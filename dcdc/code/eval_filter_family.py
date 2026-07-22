#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dcdc_taskset import Task
from filter_taskset import default_taskset_filter
from filter_eval_ac import FILTER_F_TOL_REL, task_to_specs, eval_one_detail_filter
from inc_parser import extract_inc_lines

RESPONSE_TEMPLATE = "### Response:\n"
_RESP_KEY = "### Response:"


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def build_prompt(family: str, vin: float, vout: float, *, min_elems: int = 20) -> str:
    fam = (family or "").strip().lower()
    spec = task_to_specs(Task(fam, float(vin), float(vout)))

    if fam in {"filter_lpf", "filter_hpf"}:
        fc = float(spec.get("fc_hz") or 0.0)
        atten = float(spec.get("atten_db") or 20.0)
        spec_line = f"fc={fc:.3g} Hz (?{100*FILTER_F_TOL_REL:.0f}%), stopband ? {atten:.0f} dB"
    elif fam == "filter_bpf":
        f0 = float(spec.get("f0_hz") or 0.0)
        bw = float(spec.get("bw_hz") or 0.0)
        spec_line = f"f0={f0:.3g} Hz (?{100*FILTER_F_TOL_REL:.0f}%), bw={bw:.3g} Hz"
    elif fam == "filter_notch":
        f0 = float(spec.get("f0_hz") or 0.0)
        atten = float(spec.get("atten_db") or 20.0)
        spec_line = f"f0={f0:.3g} Hz (?{100*FILTER_F_TOL_REL:.0f}%), notch depth ? {atten:.0f} dB"
    else:
        spec_line = f"vin={vin}, vout={vout}"

    body = f"""Generate a {fam} circuit in INC DSL.

Rules:
- Output ONLY INC lines (no explanation).
- Line format: INC <name> <node1> <node2> <value>
- Element names MUST start with R/C/L (e.g., R1, C1, L1).
- Do NOT wrap output in Markdown/code fences.
- Use ONLY passive elements {{R,C,L}}.
- Required nodes: {{vin, out, 0}}. You MAY add helper nodes (n1, n2, ...).
- DO NOT use numeric node names like 12 or 5 (only ground '0' is numeric).
- Use at least {int(min_elems)} INC lines (>= {int(min_elems)} elements).

Specs (AC point-specs): {spec_line}
"""
    return body + RESPONSE_TEMPLATE


def load_model(base_model: str, adapter: Optional[str]) -> tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.bos_token_id is None:
        tok.bos_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return tok, model


def _device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_response(text: str) -> str:
    s = text or ""
    if _RESP_KEY in s:
        s = s.split(_RESP_KEY, 1)[-1]
    return s


def _canon_unique(inc_text: str) -> str:
    lines = [ln.strip() for ln in extract_inc_lines(inc_text) if ln.strip()]
    return "\n".join(sorted(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_per_task", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--sim_timeout_s", type=float, default=60.0)
    ap.add_argument("--min_elems", type=int, default=20)
    ap.add_argument("--no_fallback", action="store_true")
    ap.add_argument("--only_task", action="append", default=[], help="<family>,<vin>,<vout>")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tasks").mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))

    tasks = default_taskset_filter()
    only: List[Task] = []
    for s in (args.only_task or []):
        try:
            parts = [p.strip() for p in str(s).split(",")]
            if len(parts) != 3:
                continue
            fam, a, b = parts[0], float(parts[1]), float(parts[2])
            only.append(Task(str(fam).lower(), float(a), float(b)))
        except Exception:
            continue
    if only:
        tasks = only

    tok, model = load_model(args.base_model, str(args.adapter or "").strip() or None)
    dev = _device(model)

    summary: Dict[str, Any] = {"started_at": _now(), "results": {}}

    for ti, t in enumerate(tasks):
        prompt = build_prompt(t.family, t.vin, t.vout, min_elems=int(args.min_elems))

        task_dir = outdir / "tasks" / t.family.lower() / f"vin{float(t.vin):.1f}_vout{float(t.vout):.1f}"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        uniq: Dict[str, Dict[str, Any]] = {}
        for si in range(int(args.n_per_task)):
            seed = int(args.seed) + ti * 1000 + si * 17 + rng.randint(0, 9999)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            inp = tok(prompt, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    do_sample=True,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_new_tokens=int(args.max_new_tokens),
                    pad_token_id=tok.eos_token_id,
                )
            txt = tok.decode(out[0], skip_special_tokens=True)
            resp = _extract_response(txt)
            can = _canon_unique(resp)
            if not can.strip():
                continue
            if can in uniq:
                continue

            detail = eval_one_detail_filter(
                resp,
                family=str(t.family),
                vin=float(t.vin),
                vout=float(t.vout),
                sim_timeout_s=float(args.sim_timeout_s),
                min_elems=int(args.min_elems),
            )
            uniq[can] = {"raw": resp, "detail": detail}

        samples = [v["detail"] for v in uniq.values()]
        n_ok = sum(1 for d in samples if bool(d.get("ok")))
        n_cv = sum(1 for d in samples if bool(d.get("pass_CV")))
        n_ce = sum(1 for d in samples if bool(d.get("pass_CE")))
        n_fc = sum(1 for d in samples if bool(d.get("pass_fc")))
        n_stop = sum(1 for d in samples if bool(d.get("pass_stop")))
        n_f0 = sum(1 for d in samples if bool(d.get("pass_f0")))
        n_bw = sum(1 for d in samples if bool(d.get("pass_bw")))
        n_depth = sum(1 for d in samples if bool(d.get("pass_depth")))
        n_il = sum(1 for d in samples if bool(d.get("pass_il")))

        key = f"{t.family.lower()}/vin{float(t.vin):.1f}_vout{float(t.vout):.1f}"
        payload = {
            "family": t.family.lower(),
            "vin": float(t.vin),
            "vout": float(t.vout),
            "summary": {
                "n_unique": int(len(samples)),
                "ok": int(n_ok),
                "pass_CV": int(n_cv),
                "pass_CE": int(n_ce),
                "pass_fc": int(n_fc),
                "pass_stop": int(n_stop),
                "pass_f0": int(n_f0),
                "pass_bw": int(n_bw),
                "pass_depth": int(n_depth),
                "pass_il": int(n_il),
            },
            "details": list(uniq.values()),
        }
        (task_dir / "eval.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["results"][key] = payload["summary"]

    summary["finished_at"] = _now()
    (outdir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] saved", str(outdir))


if __name__ == "__main__":
    main()

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

from amp_eval_acop import (
    AMP_BW_TOL_REL,
    AMP_GAIN_TOL_DB,
    AMP_PM_MIN_DEG,
    AMP_PSTATIC_MAX_MW,
    AMP_VDD,
    amp_family_params,
    eval_one_detail_amp_family,
    task_to_specs,
)
from amp_taskset import default_taskset_amp
from dcdc_taskset import Task
from inc_parser import extract_inc_lines

RESPONSE_TEMPLATE = "### Response:\n"
_RESP_KEY = "### Response:"


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def build_prompt(family: str, vin: float, vout: float, *, min_elems: int = 15) -> str:
    fam = (family or "").strip().lower()
    if fam == "amp":
        fam = "amp_op2"

    params = amp_family_params(fam)
    vdd = float(params.get("vdd") or AMP_VDD)
    rload = float(params.get("rload_ohm") or 10_000.0)
    pm_min = float(params.get("pm_min_deg") or AMP_PM_MIN_DEG)
    p_max = float(params.get("pstatic_max_mw") or AMP_PSTATIC_MAX_MW)

    gain_db = float(vin)
    bw_hz = float(vout)

    # Avoid leaking internal family tokens (e.g., "amp_op2") into the prompt.
    # The model tends to copy them into element names, which breaks the DSL.
    if fam == "amp_op2":
        fam_desc = "two-stage op-amp closed-loop amplifier (light load)"
    elif fam == "amp_rfpa":
        fam_desc = "high-speed / RF power-driver amplifier (heavier load)"
    else:
        fam_desc = "analog amplifier (unspecified sub-type)"

    # Helpful analytic init (used by the EDA repair operator and can be learned by the policy).
    # Calibration note (empirical for the current amp_eval_acop core model):
    # - amp_op2 needs a slightly larger RFB/RG ratio to satisfy phase-margin constraints.
    # - A global CCOMP scale (~1.4) better matches the evaluator's measured -3dB bandwidth.
    rg0 = 10_000.0
    a_lin = float(10.0 ** (float(gain_db) / 20.0))
    ratio_ideal = float(max(1e-6, a_lin - 1.0))

    if fam == "amp_op2":
        ratio_scale = 1.2
        rbias_hint = 1500.0
    else:
        ratio_scale = 1.0
        rbias_hint = 220.0

    rf0 = float(max(10.0, ratio_ideal * float(ratio_scale) * float(rg0)))

    ccomp_scale = 1.4
    try:
        ccomp0 = float(ccomp_scale) / (2.0 * 3.141592653589793 * max(1.0, rf0) * max(1e-9, float(bw_hz)))
    except Exception:
        ccomp0 = 1e-15
    ccomp0 = float(min(max(float(ccomp0), 1e-15), 1e-6))

    rbias_min = float((vdd * vdd) / max(1e-9, (float(p_max) / 1e3)))
    rbias0 = float(max(float(rbias_hint), float(rbias_min) * (1.02 if fam == "amp_rfpa" else 1.0)))

    body = f"""Generate an analog amplifier netlist in INC DSL.

Task:
- Circuit class: {fam_desc}

Rules:
- Output ONLY INC lines (no explanation, no Markdown, no code fences).
- Line format: INC <name> <node1> <node2> <value>
- Element names MUST start with R/C/L (e.g., RBIAS, R1, CCOMP, L1). Do NOT use circuit-class labels as element names.
- Do NOT reuse element names: every INC line must have a unique <name>.
- Use ONLY passive elements {{R,C,L}}.
- Use ONLY these nodes: {{vin, inv, out, vdd, 0}} (do NOT introduce helper nodes).
- DO NOT use numeric node names like 12 or 5 (only ground '0' is numeric).
- Use at least {int(min_elems)} INC lines (>= {int(min_elems)} elements).
- Non-inverting topology: do NOT connect node vin to any node except 0 (vin is the op-amp non-inv input driven by an ideal source).
- Must include a bias resistor between vdd and 0 (static power control).
- Must include negative feedback: at least one R between out-inv and one R between inv-0.
- If you need extra parts to reach min_elems, ONLY add non-interacting dummies:
  - extra R: connect out-0 and set value=1e12
  - extra C: connect vdd-0 and set value=1e-15
  - to include node vin without breaking topology, add ONE dummy: RIN vin 0 1e12

Specs (AC+OP):
- target_gain_db={gain_db:.1f} (±{AMP_GAIN_TOL_DB:.1f} dB)
- target_bw_hz={bw_hz:.3g} (±{100*AMP_BW_TOL_REL:.0f}%)
- min_phase_margin_deg={pm_min:.0f}
- max_static_power_mW={p_max:.1f}
- VDD={vdd:.1f} V
- RLOAD={rload:.0f} ohm

Helpful starting-point (optional):
- Set RG(inv,0) ≈ {rg0:.0f}
- Set RFB(out,inv) ≈ {rf0:.4g}  (for non-inverting gain)
- Set CCOMP(out,inv) ≈ {ccomp0:.4g}
- Set RBIAS(vdd,0) ? {rbias0:.4g} (>= {rbias_min:.1f}; larger RBIAS lowers static power and often improves phase margin).
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
    ap.add_argument("--min_elems", type=int, default=15)
    ap.add_argument("--no_fallback", action="store_true")
    ap.add_argument("--only_task", action="append", default=[], help="<family>,<vin>,<vout>")
    args = ap.parse_args()
    # VP-SPI min_elems relax: amplifier/oscillator branches use a lighter complexity constraint.
    _min_raw = int(getattr(args, "min_elems", 15) or 15)
    if _min_raw > 15:
        print(f"[min_elems] clamp {_min_raw} -> 15 (amp/osc)", flush=True)
        args.min_elems = 15

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tasks").mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))

    tasks = default_taskset_amp()
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
        gain_db, bw_hz = task_to_specs(t)
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

            detail = eval_one_detail_amp_family(
                resp,
                family=str(t.family),
                gain_db=float(gain_db),
                bw_hz=float(bw_hz),
                sim_timeout_s=float(args.sim_timeout_s),
                min_elems=int(args.min_elems),
            )
            uniq[can] = {"raw": resp, "detail": detail}

        samples = [v["detail"] for v in uniq.values()]
        n_ok = sum(1 for d in samples if bool(d.get("ok")))
        n_cv = sum(1 for d in samples if bool(d.get("pass_CV")))
        n_ce = sum(1 for d in samples if bool(d.get("pass_CE")))
        n_gain = sum(1 for d in samples if bool(d.get("pass_gain")))
        n_bw = sum(1 for d in samples if bool(d.get("pass_bw")))
        n_pm = sum(1 for d in samples if bool(d.get("pass_pm")))
        n_p = sum(1 for d in samples if bool(d.get("pass_p")))

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
                "pass_gain": int(n_gain),
                "pass_bw": int(n_bw),
                "pass_pm": int(n_pm),
                "pass_p": int(n_p),
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

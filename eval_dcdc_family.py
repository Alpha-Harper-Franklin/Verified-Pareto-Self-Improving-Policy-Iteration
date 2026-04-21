#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dcdc_eval_tran import eval_one_detail_dcdc
from dcdc_taskset import default_taskset
from dcdc_verifier import verify_inc_dcdc
from inc_parser import extract_inc_lines, parse_inc


RESPONSE_TEMPLATE = "### Response:\n"
_RESP_KEY = "### Response:"


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def build_prompt(family: str, vin: float, vout: float) -> str:
    fam = (family or "").strip().lower()
    common = (
        "- Output ONLY INC lines with one INC statement per line.\n"
        "- Use valid node names such as vin, sw, out, 0, n1, mid, sw1, sw2.\n"
        "- Use valid element names beginning with R/L/C/D/S and including digits.\n"
        "- Use at least 20 INC lines (>=20 elements).\n"
        "- You may add passive banks, snubbers, and damping networks, but do not copy a fixed template.\n"
    )
    if fam == "buck":
        return (
            "Generate a Buck DC-DC converter in INC DSL.\n"
            "Requirements:\n"
            + common
            + "- Realize a buck-family power stage with a switch from input to the switching node, a freewheel diode connected to the switching node, an inductor from the switching node to the output, and an output capacitor to ground.\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )
    if fam == "boost":
        return (
            "Generate a Boost DC-DC converter in INC DSL.\n"
            "Requirements:\n"
            + common
            + "- Realize a boost-family power stage with an input inductor, a switch from the switching node to ground, a diode from the switching node to the output, and an output capacitor to ground.\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )
    if fam == "sepic":
        return (
            "Generate a SEPIC DC-DC converter in INC DSL.\n"
            "Requirements:\n"
            + common
            + "- Realize a SEPIC-family power stage with nodes vin, sw, n1, out, and 0, including two inductive/coupling branches, one switch to ground, one diode to the output, and an output capacitor.\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )
    if fam in {"buckboost", "buck-boost", "bb"}:
        return (
            "Generate a non-inverting Buck-Boost DC-DC converter in INC DSL.\n"
            "Requirements:\n"
            + common
            + "- Realize a cascaded buck-to-boost family structure with nodes vin, sw1, mid, sw2, out, and 0.\n"
            + "- Use two switches with separate gate names or models so the two stages are independently driven.\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )
    raise ValueError(f"unknown family: {family}")


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


def _device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _flatten_detail(detail: dict) -> dict:
    keys = [
        "ok",
        "pass_C",
        "pass_CV",
        "pass_CE",
        "eff",
        "vavg",
        "ripple",
        "overshoot",
        "canonical_hash",
        "tuned",
        "tune_iters",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if k in detail:
            out[k] = detail[k]
    if "error" in detail:
        out["error"] = detail["error"]
    if "violations" in detail:
        out["violations"] = detail["violations"]
    return out


def _set_elem_value(inc: str, name: str, value: float) -> str:
    lines = extract_inc_lines(inc)
    out: List[str] = []
    changed = False
    for raw in lines:
        toks = raw.split()
        if len(toks) == 5 and toks[0] == "INC" and toks[1] == name:
            toks[4] = f"{float(value):.6g}"
            out.append(" ".join(toks))
            changed = True
        else:
            out.append(raw)
    if not changed:
        return inc
    return ("\n".join(out).strip() + "\n")


def _append_inc_line(inc: str, line: str) -> str:
    s = (inc or "").strip()
    ln = (line or "").strip()
    if not ln:
        return (s + "\n") if s else ""
    if s:
        return s + "\n" + ln + "\n"
    return ln + "\n"


def _new_name(existing: set[str], prefix: str, *, start: int = 1) -> str:
    p = (prefix or "").strip().upper()[:1]
    if p not in {"R", "L", "C", "D", "S"}:
        p = "X"
    i = max(1, int(start))
    while True:
        name = f"{p}{i}"
        if name not in existing:
            existing.add(name)
            return name
        i += 1


def _bank_expand_to_min_elems(
    inc: str,
    *,
    family: str,
    min_elems: int,
    max_parts_per_elem: int = 32,
) -> tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"used": False}
    elems = parse_inc(inc)
    n0 = int(len(elems))
    need = int(min_elems) - n0
    if need <= 0:
        return inc, meta

    caps = [e for e in elems if e.kind == "C" and (e.value is not None) and float(e.value) > 0.0]
    if not caps:
        return inc, meta

    def _is_out0(e: Any) -> bool:
        ns = {str(x).lower() for x in (e.nodes or [])}
        return ("out" in ns) and ("0" in ns)

    caps_out = [e for e in caps if _is_out0(e)]
    pick = max(caps_out, key=lambda e: float(e.value)) if caps_out else max(caps, key=lambda e: float(e.value))

    c_min = 1e-9
    base = float(pick.value)
    k_target = int(min(int(max_parts_per_elem), need + 1))
    k_max_by_min = int(max(1, math.floor(base / c_min)))
    k = int(max(1, min(k_target, k_max_by_min)))
    if k <= 1:
        return inc, meta

    part = float(base) / float(k)
    if part < c_min:
        return inc, meta

    inc2 = _set_elem_value(inc, pick.name, float(part))
    added: List[str] = []
    existing = {str(e.name).strip() for e in elems if str(e.name).strip()}
    for _ in range(k - 1):
        name_new = _new_name(existing, "C", start=1000)
        n1, n2 = pick.nodes[0], pick.nodes[1]
        inc2 = _append_inc_line(inc2, f"INC {name_new} {n1} {n2} {float(part):.6g}")
        added.append(name_new)

    meta = {
        "used": True,
        "strategy": "cap_bank_parallel",
        "picked": {
            "name": str(pick.name),
            "nodes": list(pick.nodes),
            "value_before": float(base),
            "value_each": float(part),
            "k": int(k),
        },
        "added": added,
        "n_elems_before": int(n0),
        "n_elems_after": int(len(parse_inc(inc2))),
        "family": str(family).lower(),
    }
    return inc2, meta


def _task_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = max(1, len(rows))
    return {
        "n_samples": len(rows),
        "ok_rate": sum(1 for r in rows if bool(r.get("ok", False))) / n,
        "cv_rate": sum(1 for r in rows if bool(r.get("pass_CV", False))) / n,
        "ce_rate": sum(1 for r in rows if bool(r.get("pass_CE", False))) / n,
        "min_elems_rate": sum(1 for r in rows if bool(r.get("meets_min_elems", False))) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_per_task", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--rload", type=float, default=10.0)
    ap.add_argument("--t_pre", type=float, default=0.008)
    ap.add_argument("--t_win", type=float, default=0.002)
    ap.add_argument("--sim_timeout_s", type=float, default=180.0)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--min_elems", type=int, default=20)
    ap.add_argument("--bank_expand_min_elems", action="store_true")
    ap.add_argument("--bank_expand_max_parts", type=int, default=32)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--only_task", action="append", default=[])
    args = ap.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tasks = default_taskset()
    if args.only_task:
        wanted = set()
        for spec in args.only_task:
            fam_s, vin_s, vout_s = [p.strip() for p in str(spec).split(",")]
            wanted.add((fam_s.lower(), float(vin_s), float(vout_s)))
        tasks = [t for t in tasks if (str(t.family).lower(), float(t.vin), float(t.vout)) in wanted]

    tok = None
    model = None
    dev = None
    logits_proc = None
    if not bool(args.dry_run):
        tok, model = load_model(args.base_model, (args.adapter or "").strip() or None)
        dev = _device(model)
        if bool(args.constrained):
            from integrated.constraints import CharClassLogitsProcessor

            logits_proc = [CharClassLogitsProcessor(tok, penalty=30.0)]

    summary: Dict[str, Any] = {
        "started_at": _now(),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "n_tasks": len(tasks),
        "n_per_task": int(args.n_per_task),
        "tol": float(args.tol),
        "min_elems": int(args.min_elems),
        "constrained": bool(args.constrained),
        "results": {},
    }

    for task in tasks:
        fam = str(task.family)
        vin = float(task.vin)
        vout = float(task.vout)
        task_key = f"{fam}_vin{vin:.1f}_vout{vout:.1f}"
        task_dir = outdir / fam / f"vin{vin:.1f}_vout{vout:.1f}"
        task_dir.mkdir(parents=True, exist_ok=True)

        prompts = [build_prompt(fam, vin, vout) for _ in range(int(args.n_per_task))]
        results: List[Dict[str, Any]] = []

        if bool(args.dry_run):
            (task_dir / "prompts.txt").write_text("\n\n".join(prompts), encoding="utf-8")
            summary["results"][task_key] = {"dry_run": True, "n_prompts": len(prompts)}
            continue

        t_gen0 = time.time()
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_new_tokens=int(args.max_new_tokens),
                logits_processor=logits_proc,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        outputs_texts = tok.batch_decode(out_ids, skip_special_tokens=True)
        per_sample_gen_time = float(time.time() - t_gen0) / max(1, len(outputs_texts))

        for i, txt0 in enumerate(outputs_texts):
            txt = str(txt0 or "")
            if _RESP_KEY in txt:
                txt = txt.rsplit(_RESP_KEY, 1)[-1].lstrip()
            inc_lines = extract_inc_lines(txt)
            inc = ("\n".join(inc_lines).strip() + "\n") if inc_lines else ""

            ver = verify_inc_dcdc(inc, family=fam, vin=vin, vout=vout)
            violations = list(ver.violations)
            too_few = int(ver.n_elems) < int(args.min_elems)

            bank_meta: Dict[str, Any] = {"used": False}
            if bool(too_few) and bool(args.bank_expand_min_elems) and inc.strip():
                inc2, bank_meta = _bank_expand_to_min_elems(
                    inc,
                    family=fam,
                    min_elems=int(args.min_elems),
                    max_parts_per_elem=int(args.bank_expand_max_parts),
                )
                if bool(bank_meta.get("used")) and inc2.strip():
                    inc = inc2.strip() + "\n"
                    ver = verify_inc_dcdc(inc, family=fam, vin=vin, vout=vout)
                    violations = list(ver.violations)
                    too_few = int(ver.n_elems) < int(args.min_elems)

            meets_min = int(ver.n_elems) >= int(args.min_elems)
            if (not bool(ver.ok)) or (not bool(meets_min)):
                detail = {
                    "ok": False,
                    "pass_C": False,
                    "pass_CV": False,
                    "pass_CE": False,
                    "eff": 0.0,
                    "vavg": 0.0,
                    "ripple": 0.0,
                    "overshoot": 0.0,
                    "canonical_hash": ver.canonical_hash,
                    "violations": violations + ([f"too_few_elems_{int(ver.n_elems)}"] if too_few else []),
                    "error": ("too_few_elems" if too_few else "invalid_structure"),
                    "tuned": False,
                    "tune_iters": 0,
                }
            else:
                detail = eval_one_detail_dcdc(
                    inc=inc,
                    family=fam,
                    vin=vin,
                    vout=vout,
                    tol=float(args.tol),
                    rload=float(args.rload),
                    t_pre=float(args.t_pre),
                    t_win=float(args.t_win),
                    sim_timeout_s=float(args.sim_timeout_s),
                )

            row: Dict[str, Any] = {
                "index": int(i + 1),
                "gen_time": float(per_sample_gen_time),
                "inc_source": str(inc).strip(),
                "raw_text": txt[:4000],
                "violations": violations,
                "n_elems": int(ver.n_elems),
                "n_inc_lines": int(ver.n_inc_lines),
                "meets_min_elems": bool(meets_min),
                "bank_expand": bank_meta,
                "source_format": "INC",
            }
            row.update(_flatten_detail(detail))
            results.append(row)

        task_payload = {
            "family": fam,
            "vin": vin,
            "vout": vout,
            "summary": _task_summary(results),
            "results": results,
        }
        (task_dir / f"metric_vin{vin:.1f}_vout{vout:.1f}_full.json").write_text(
            json.dumps(task_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["results"][task_key] = task_payload["summary"]

    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

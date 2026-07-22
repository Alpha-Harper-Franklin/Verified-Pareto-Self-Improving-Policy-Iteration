#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dcdc_eval_tran import eval_one_detail_dcdc
from dcdc_templates import templates
from dcdc_verifier import verify_inc_dcdc
from inc_parser import extract_inc_lines

import sys
sys.path.append("/root/workspace_autocircuit_rl")
sys.path.append("/root/workspace_autocircuit_rl/integrated")
from integrated.constraints import CharClassLogitsProcessor


def _device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _score(detail: dict, vout: float, tol_ref: float = 0.1) -> float:
    if not detail.get("ok"):
        return -1.0
    vavg = float(detail.get("vavg", 0.0) or 0.0)
    eff = float(detail.get("eff", 0.0) or 0.0)
    ripple = float(detail.get("ripple", 0.0) or 0.0)
    overshoot = float(detail.get("overshoot", 0.0) or 0.0)
    err = abs(vavg - float(vout)) / max(1e-6, float(vout))
    score_v = 3.0 * max(0.0, 1.0 - err / max(1e-6, float(tol_ref)))
    score_eff = 0.5 * eff
    score_ripple = -0.2 * (ripple / max(1e-6, float(vout)))
    score_over = -0.2 * overshoot
    if not detail.get("pass_CV", False):
        score_v -= 1.0
    if not detail.get("pass_CE", False):
        score_eff -= 0.5
    return float(score_v + score_eff + score_ripple + score_over)


def build_prompt(family: str, vin: float, vout: float) -> str:
    fam = (family or "").strip().lower()
    if fam == "buck":
        return (
            "Generate a Buck converter in INC DSL. Use only nodes {vin,sw,out,0}. "
            "Use elements: L,C,D,S with values (e.g., 47u). Output only INC lines.\n"
            "Example:\n"
            "INC S1 vin sw Sstd\nINC D1 0 sw Dstd\nINC L1 sw out 47u\nINC C1 out 0 47u\n\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V.\n"
        )
    if fam == "boost":
        return (
            "Generate a Boost converter in INC DSL. Use only nodes {vin,sw,out,0}. "
            "Typical pattern: L vin-sw, S sw-0, D sw-out, C out-0. Output only INC lines.\n"
            "Example:\n"
            "INC L1 vin sw 47u\nINC S1 sw 0 Sstd\nINC D1 sw out Dstd\nINC C1 out 0 47u\n\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V.\n"
        )
    if fam == "sepic":
        return (
            "Generate a SEPIC converter in INC DSL. Allowed nodes: {vin,sw,n1,out,0}. "
            "Use 2 inductors and 2 capacitors: L1 vin-sw, C1 sw-n1, L2 n1-0, S1 sw-0, D1 n1-out, C2 out-0. "
            "Output only INC lines.\n"
            "Example:\n"
            "INC L1 vin sw 47u\nINC C1 sw n1 1u\nINC L2 n1 0 47u\nINC S1 sw 0 Sstd\nINC D1 n1 out Dstd\nINC C2 out 0 47u\n\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V.\n"
        )
    if fam == "buckboost":
        return (
            "Generate a non-inverting Buck-Boost (cascaded buck->boost) in INC DSL. "
            "Allowed nodes: {vin,sw1,mid,sw2,out,0}. Use 2 switches and 2 diodes. "
            "Important: use switch models Sstd1 and Sstd2 so gate1/gate2 are separate. Output only INC lines.\n"
            "Example:\n"
            "INC S1 vin sw1 Sstd1\nINC D1 0 sw1 Dstd\nINC L1 sw1 mid 47u\nINC C1 mid 0 47u\n"
            "INC L2 mid sw2 47u\nINC S2 sw2 0 Sstd2\nINC D2 sw2 out Dstd\nINC C2 out 0 47u\n\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V.\n"
        )
    raise ValueError(f"unknown family: {family}")


def load_model(base_model: str):
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return tok, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--family", required=True, choices=["buck", "boost", "sepic", "buckboost"])
    ap.add_argument("--vin", type=float, required=True)
    ap.add_argument("--vout", type=float, required=True)
    ap.add_argument("--n_gen", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--opt_budget", type=int, default=30)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.base_model)
    dev = _device(model)
    logits_proc = [CharClassLogitsProcessor(tok, penalty=30.0)] if args.constrained else None

    prompt = build_prompt(args.family, args.vin, args.vout)
    tpl = templates().get(args.family, "")

    candidates: List[Dict[str, Any]] = []

    for i in range(int(args.n_gen)):
        seed_i = int(args.seed) + i
        random.seed(seed_i)
        torch.manual_seed(seed_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_i)

        enc = tok(prompt, return_tensors="pt").to(dev)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                logits_processor=logits_proc,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                num_return_sequences=1,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0

        txt = tok.decode(out[0], skip_special_tokens=True)
        inc_lines = extract_inc_lines(txt)
        inc = "\n".join(inc_lines).strip() + "\n"

        ver = verify_inc_dcdc(inc, family=args.family, vin=args.vin, vout=args.vout)
        used_fallback = False
        if not ver.ok:
            inc = tpl
            ver = verify_inc_dcdc(inc, family=args.family, vin=args.vin, vout=args.vout)
            used_fallback = True

        detail = eval_one_detail_dcdc(inc=inc, family=args.family, vin=args.vin, vout=args.vout, tol=0.1, rload=10.0, t_pre=0.008, t_win=0.002)
        s = _score(detail, vout=args.vout, tol_ref=0.1)

        candidates.append(
            {
                "i": i,
                "seed": seed_i,
                "gen_time_s": float(dt),
                "used_fallback": bool(used_fallback),
                "verify": {"ok": bool(ver.ok), "violations": list(ver.violations)},
                "detail": detail,
                "score": float(s),
                "inc": inc,
                "raw_tail": txt[-500:],
            }
        )

    (outdir / "candidates.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in candidates) + "\n", encoding="utf-8")

    best = max(candidates, key=lambda x: x.get("score", -1e9))
    (outdir / "seed_inc.txt").write_text(best["inc"], encoding="utf-8")
    (outdir / "seed_detail.json").write_text(json.dumps(best["detail"], ensure_ascii=False, indent=2), encoding="utf-8")

    # Run parameter optimization as a subprocess (reuses ngspice loop)
    opt_out = outdir / "opt"
    opt_out.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    cmd = [
        py,
        str(Path(__file__).with_name("optimize_dcdc_params.py")),
        "--family",
        str(args.family),
        "--vin",
        str(float(args.vin)),
        "--vout",
        str(float(args.vout)),
        "--inc_file",
        str(outdir / "seed_inc.txt"),
        "--outdir",
        str(opt_out),
        "--budget",
        str(int(args.opt_budget)),
        "--pop",
        "10",
        "--elite",
        "3",
        "--seed",
        str(int(args.seed)),
        "--opt_duty",
        "--opt_freq",
    ]
    (outdir / "opt_cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

    subprocess.run(cmd, check=True)

    print(json.dumps({"seed_best": best, "opt_summary": json.loads((opt_out / "summary.json").read_text())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

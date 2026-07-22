#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


RESPONSE_TEMPLATE = "### Response:\n"
_RESP_KEY = "### Response:"


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


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
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        )
        model = get_peft_model(model, lora)
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)

    model.eval()
    return tok, model


def _device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_prompt(family: str, vin: float, vout: float) -> str:
    # Reuse the module-graph language (contribution #2).
    from build_opt_modulegraph_datasets import build_module_prompt

    return build_module_prompt(family, vin, vout, min_mods=6)


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
    return out


def _run_optimizer(
    python: str,
    code_dir: Path,
    *,
    family: str,
    vin: float,
    vout: float,
    inc_seed_path: Path,
    opt_dir: Path,
    budget: int,
    pop: int,
    elite: int,
    seed: int,
    robust: bool,
    vin_jitter: float,
    rload_list: str,
    agg: str,
    cvar_alpha: float,
) -> tuple[dict, str]:
    opt_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(code_dir / "optimize_dcdc_params.py"),
        "--family",
        str(family),
        "--vin",
        str(float(vin)),
        "--vout",
        str(float(vout)),
        "--inc_file",
        str(inc_seed_path),
        "--outdir",
        str(opt_dir),
        "--budget",
        str(int(budget)),
        "--pop",
        str(int(pop)),
        "--elite",
        str(int(elite)),
        "--seed",
        str(int(seed)),
        "--rload",
        "10.0",
        "--tol",
        "0.1",
        "--tol_ref",
        "0.1",
        "--t_pre",
        "0.008",
        "--t_win",
        "0.002",
        "--opt_duty",
        "--opt_freq",
    ]
    if robust:
        cmd += [
            "--robust",
            "--vin_jitter",
            str(float(vin_jitter)),
            "--rload_list",
            str(rload_list),
            "--agg",
            str(agg),
            "--cvar_alpha",
            str(float(cvar_alpha)),
        ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_dir)
    subprocess.run(cmd, check=True, env=env)
    summary = json.loads((opt_dir / "summary.json").read_text(encoding="utf-8"))
    best_inc = (opt_dir / "best_inc.txt").read_text(encoding="utf-8")
    return summary, best_inc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_per_task", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--min_elems", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--only_task",
        action="append",
        default=[],
        help="Restrict eval to specific tasks: 'family,vin,vout' (repeatable).",
    )
    ap.add_argument("--max_tasks", type=int, default=0, help="0 means all tasks (applied after --only_task)")

    ap.add_argument("--autotune_duty", action="store_true", help="Enable 1-step duty auto-tune in final sim (off by default).")

    ap.add_argument("--opt_budget", type=int, default=40)
    ap.add_argument("--opt_pop", type=int, default=10)
    ap.add_argument("--opt_elite", type=int, default=3)
    ap.add_argument("--opt_robust", action="store_true")
    ap.add_argument("--vin_jitter", type=float, default=0.10)
    ap.add_argument("--rload_list", default="5,10,20")
    ap.add_argument("--agg", choices=["cvar", "worst"], default="cvar")
    ap.add_argument("--cvar_alpha", type=float, default=0.25)

    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--rload", type=float, default=10.0)
    ap.add_argument("--t_pre", type=float, default=0.008)
    ap.add_argument("--t_win", type=float, default=0.002)

    args = ap.parse_args()

    # imports that depend on PYTHONPATH on remote
    import sys

    for base in [
        "/root/workspace_autocircuit_rl",
        "/root/autodl-tmp/workspace_autocircuit_rl",
        "/root/autodl-tmp/workspace_autocircuit_rl/integrated",
    ]:
        if base not in sys.path:
            sys.path.append(base)

    from dcdc_eval_tran import eval_one_detail_dcdc
    from dcdc_module_compiler import compile_module_graph
    from dcdc_taskset import default_taskset
    from dcdc_verifier import verify_inc_dcdc
    from inc_parser import extract_inc_lines

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.base_model, adapter=(args.adapter or "").strip() or None)
    dev = _device(model)

    tasks = default_taskset()
    if args.only_task:
        wanted = set()
        for spec in args.only_task:
            s = str(spec or "").strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) != 3:
                raise SystemExit(f"--only_task expects 'family,vin,vout' (got {spec!r})")
            fam_s, vin_s, vout_s = parts
            wanted.add((fam_s.lower(), float(vin_s), float(vout_s)))
        tasks = [t for t in tasks if (str(t.family).lower(), float(t.vin), float(t.vout)) in wanted]
    if int(args.max_tasks) > 0:
        tasks = tasks[: int(args.max_tasks)]
    summary: Dict[str, Any] = {
        "started_at": _now(),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "n_tasks": len(tasks),
        "n_per_task": int(args.n_per_task),
        "min_elems": int(args.min_elems),
        "opt_budget": int(args.opt_budget),
        "opt_pop": int(args.opt_pop),
        "opt_elite": int(args.opt_elite),
        "opt_robust": bool(args.opt_robust),
        "vin_jitter": float(args.vin_jitter),
        "rload_list": str(args.rload_list),
        "agg": str(args.agg),
        "cvar_alpha": float(args.cvar_alpha),
        "results": {},
    }

    code_dir = Path(__file__).resolve().parent
    python = sys.executable

    progress_path = outdir / "progress.txt"

    for ti, task in enumerate(tasks):
        fam = str(task.family)
        vin = float(task.vin)
        vout = float(task.vout)

        tdir = outdir / fam / f"vin{vin:.1f}_vout{vout:.1f}"
        tdir.mkdir(parents=True, exist_ok=True)
        out_path = tdir / f"metric_vin{vin:.1f}_vout{vout:.1f}_full.json"
        if bool(args.resume) and out_path.exists():
            continue

        prompt = build_prompt(fam, vin, vout)
        enc = tok(prompt, return_tensors="pt").to(dev)

        seed_base = int(args.seed) + ti * 1000
        random.seed(seed_base)
        torch.manual_seed(seed_base)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_base)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            outs = model.generate(
                **enc,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                num_return_sequences=int(args.n_per_task),
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_dt = time.time() - t0

        results: List[Dict[str, Any]] = []
        seen_hashes = set()

        for i in range(int(outs.shape[0])):
            txt = tok.decode(outs[i], skip_special_tokens=True)
            if _RESP_KEY in txt:
                txt = txt.rsplit(_RESP_KEY, 1)[-1].lstrip()

            inc_lines = extract_inc_lines(txt)
            seed_inc = ""
            source_format = "INC"
            if inc_lines:
                seed_inc = "\n".join(inc_lines).strip() + "\n"
            else:
                fam2, inc2, errs = compile_module_graph(txt, expected_family=fam)
                if not errs and inc2.strip():
                    seed_inc = inc2.strip() + "\n"
                    source_format = "MOD"

            ver = verify_inc_dcdc(seed_inc, family=fam, vin=vin, vout=vout)
            meets_min = int(ver.n_elems) >= int(args.min_elems)
            if (not ver.ok) or (not meets_min):
                detail = {
                    "ok": False,
                    "pass_C": False,
                    "canonical_hash": ver.canonical_hash,
                    "violations": list(ver.violations) + ([] if meets_min else [f"too_few_elems_{int(ver.n_elems)}"]),
                    "error": "invalid_seed",
                }
                ch = detail.get("canonical_hash") or ver.canonical_hash
                if ch in seen_hashes:
                    continue
                seen_hashes.add(ch)
                row: Dict[str, Any] = {
                    "index": int(len(results) + 1),
                    "gen_time": float(gen_dt) / max(1.0, float(outs.shape[0])),
                    "source_format": source_format,
                    "seed_inc_source": seed_inc.strip(),
                    "inc_source": "",
                    "n_elems": int(ver.n_elems),
                    "meets_min_elems": bool(meets_min),
                    "opt_used": False,
                }
                row.update(_flatten_detail(detail))
                results.append(row)
                continue

            # run optimizer (contribution #1 + #3)
            seed_inc_path = tdir / f"seed_{i+1}.inc"
            seed_inc_path.write_text(seed_inc, encoding="utf-8")
            opt_dir = tdir / f"opt_{i+1}"

            opt_ok = True
            opt_summary: dict = {}
            opt_best_inc = ""
            opt_wall_s = 0.0
            try:
                t_opt0 = time.time()
                opt_summary, opt_best_inc = _run_optimizer(
                    python=python,
                    code_dir=code_dir,
                    family=fam,
                    vin=vin,
                    vout=vout,
                    inc_seed_path=seed_inc_path,
                    opt_dir=opt_dir,
                    budget=int(args.opt_budget),
                    pop=int(args.opt_pop),
                    elite=int(args.opt_elite),
                    seed=seed_base + i,
                    robust=bool(args.opt_robust),
                    vin_jitter=float(args.vin_jitter),
                    rload_list=str(args.rload_list),
                    agg=str(args.agg),
                    cvar_alpha=float(args.cvar_alpha),
                )
                opt_wall_s = float(time.time() - t_opt0)
            except Exception as e:
                opt_ok = False
                (opt_dir / "opt_failed.txt").write_text(str(e), encoding="utf-8")

            final_inc = opt_best_inc.strip() + "\n" if (opt_ok and opt_best_inc.strip()) else seed_inc
            final_detail = eval_one_detail_dcdc(
                inc=final_inc,
                family=fam,
                vin=vin,
                vout=vout,
                tol=float(args.tol),
                rload=float(args.rload),
                t_pre=float(args.t_pre),
                t_win=float(args.t_win),
                autotune_duty=bool(args.autotune_duty),
            )
            ch = final_detail.get("canonical_hash") or ver.canonical_hash
            if ch in seen_hashes:
                continue
            seen_hashes.add(ch)

            row = {
                "index": int(len(results) + 1),
                "gen_time": float(gen_dt) / max(1.0, float(outs.shape[0])),
                "source_format": source_format,
                "seed_inc_source": seed_inc.strip(),
                "inc_source": final_inc.strip(),
                "n_elems": int(ver.n_elems),
                "meets_min_elems": True,
                "opt_used": bool(opt_ok),
                "opt_wall_s": float(opt_wall_s),
                "opt_best_score": float(opt_summary.get("best_score", -1e9)) if opt_ok else None,
                "opt_summary": opt_summary if opt_ok else None,
            }
            row.update(_flatten_detail(final_detail))
            results.append(row)

        payload = {
            "family": fam,
            "vin": vin,
            "vout": vout,
            "n_requested": int(args.n_per_task),
            "n_unique": int(len(results)),
            "samples": results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        n_ok = sum(1 for r in results if r.get("ok"))
        n_cv = sum(1 for r in results if r.get("pass_CV"))
        n_ce = sum(1 for r in results if r.get("pass_CE"))
        summary["results"][f"{fam}/vin{vin:.1f}_vout{vout:.1f}"] = {
            "n_unique": int(len(results)),
            "ok": int(n_ok),
            "pass_CV": int(n_cv),
            "pass_CE": int(n_ce),
        }
        progress_path.write_text(f"task={ti+1}/{len(tasks)} last={fam} vin={vin:.1f} vout={vout:.1f}\n", encoding="utf-8")

    summary["finished_at"] = _now()
    (outdir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] saved", str(outdir))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dcdc_module_compiler import compile_module_graph, extract_module_calls
from dcdc_taskset import Task, default_taskset
from dcdc_verifier import verify_inc_dcdc


RESPONSE_TEMPLATE = "### Response:\n"
_RESP_KEY = "### Response:"


def build_module_prompt(family: str, vin: float, vout: float, *, min_mods: int = 6) -> str:
    fam = (family or "").strip().lower()
    common = (
        "Output ONLY MOD lines (no explanations).\n"
        "MOD syntax:\n"
        "- MOD <MODULE_NAME> <node1> <node2> ...\n"
        "- Use node names consisting of letters/digits/underscore.\n"
        "Available modules:\n"
        "- BUCK_BASE vin sw out 0\n"
        "- BOOST_BASE vin sw out 0\n"
        "- SEPIC_BASE vin sw n1 out 0\n"
        "- BUCKBOOST_BASE vin sw1 mid sw2 out 0\n"
        "- CAPBANK_IN vin 0\n"
        "- CAPBANK_OUT out 0\n"
        "- CAPBANK_MID mid 0\n"
        "- SNUBBER_SW_GND sw 0\n"
        "- SNUBBER_SW_OUT sw out\n"
        "- DAMPER_OUT out 0\n"
        f"Rules:\n- Include EXACTLY ONE base module for the requested family.\n"
        f"- Include at least {int(min_mods)} MOD lines total.\n"
    )

    if fam == "buck":
        ex = (
            "Example:\n"
            "MOD BUCK_BASE vin sw out 0\n"
            "MOD CAPBANK_IN vin 0\n"
            "MOD CAPBANK_OUT out 0\n"
            "MOD SNUBBER_SW_GND sw 0\n"
            "MOD SNUBBER_SW_OUT sw out\n"
            "MOD DAMPER_OUT out 0\n"
        )
    elif fam == "boost":
        ex = (
            "Example:\n"
            "MOD BOOST_BASE vin sw out 0\n"
            "MOD CAPBANK_IN vin 0\n"
            "MOD CAPBANK_OUT out 0\n"
            "MOD SNUBBER_SW_GND sw 0\n"
            "MOD SNUBBER_SW_OUT sw out\n"
            "MOD DAMPER_OUT out 0\n"
        )
    elif fam == "sepic":
        ex = (
            "Example:\n"
            "MOD SEPIC_BASE vin sw n1 out 0\n"
            "MOD CAPBANK_IN vin 0\n"
            "MOD CAPBANK_OUT out 0\n"
            "MOD SNUBBER_SW_GND sw 0\n"
            "MOD SNUBBER_SW_OUT sw out\n"
            "MOD DAMPER_OUT out 0\n"
        )
    elif fam == "buckboost":
        ex = (
            "Example:\n"
            "MOD BUCKBOOST_BASE vin sw1 mid sw2 out 0\n"
            "MOD CAPBANK_IN vin 0\n"
            "MOD CAPBANK_MID mid 0\n"
            "MOD CAPBANK_OUT out 0\n"
            "MOD SNUBBER_SW_GND sw1 0\n"
            "MOD SNUBBER_SW_GND sw2 0\n"
            "MOD DAMPER_OUT out 0\n"
        )
    else:
        raise ValueError(f"unknown family: {family}")

    return (
        f"Generate a {fam} DC-DC converter as a module graph.\n"
        + common
        + ex
        + f"Task: Vin={float(vin):.1f}V, Vout={float(vout):.1f}V, Rload=10ohm.\n"
        + RESPONSE_TEMPLATE
    )


def _device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(base_model: str, adapter: str) -> tuple[Any, Any]:
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
        # Adapter is optional; for this offline dataset builder we typically start from a base model.
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)

    model.eval()
    return tok, model


def _run_optimize(
    python: str,
    code_dir: Path,
    *,
    family: str,
    vin: float,
    vout: float,
    inc_path: Path,
    outdir: Path,
    budget: int,
    pop: int,
    elite: int,
    seed: int,
    robust: bool,
    vin_jitter: float,
    rload_list: str,
    agg: str,
    cvar_alpha: float,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
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
        str(inc_path),
        "--outdir",
        str(outdir),
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

    summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    best_inc = (outdir / "best_inc.txt").read_text(encoding="utf-8")
    return {"summary": summary, "best_inc": best_inc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--n_gen", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--min_mods", type=int, default=6)
    ap.add_argument("--min_elems", type=int, default=20)
    ap.add_argument("--pairs_per_task", type=int, default=1, help="How many (top,bottom) preference pairs per task")
    ap.add_argument("--min_pair_gap", type=float, default=0.1, help="Require chosen_score - rejected_score >= gap")

    ap.add_argument("--opt_budget", type=int, default=40)
    ap.add_argument("--opt_pop", type=int, default=10)
    ap.add_argument("--opt_elite", type=int, default=3)

    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--vin_jitter", type=float, default=0.10)
    ap.add_argument("--rload_list", default="5,10,20")
    ap.add_argument("--agg", choices=["cvar", "worst"], default="cvar")
    ap.add_argument("--cvar_alpha", type=float, default=0.25)
    ap.add_argument("--max_tasks", type=int, default=0, help="0 means all tasks")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume by skipping tasks recorded in done_tasks.jsonl (append outputs instead of overwriting).",
    )
    args = ap.parse_args()

    code_dir = Path(__file__).resolve().parent
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.base_model, adapter=(args.adapter or "").strip())
    dev = _device(model)

    tasks = default_taskset()
    if int(args.max_tasks) > 0:
        tasks = tasks[: int(args.max_tasks)]

    pairs_path = out_root / "dpo_pairs.jsonl"
    sft_path = out_root / "sft_train.jsonl"
    meta_path = out_root / "pairs_meta.jsonl"
    progress_path = out_root / "progress.txt"
    progress_log_path = out_root / "progress_log.jsonl"
    done_path = out_root / "done_tasks.jsonl"

    def _tkey(fam: str, vin: float, vout: float) -> Tuple[str, float, float]:
        return (str(fam).strip().lower(), round(float(vin), 6), round(float(vout), 6))

    done = set()
    if bool(args.resume) and done_path.exists():
        try:
            with done_path.open("r", encoding="utf-8", errors="ignore") as f_done_r:
                for line in f_done_r:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    fam = str(r.get("family") or (r.get("task") or {}).get("family") or "").strip()
                    vin = float(r.get("vin") or (r.get("task") or {}).get("vin") or 0.0)
                    vout = float(r.get("vout") or (r.get("task") or {}).get("vout") or 0.0)
                    if fam:
                        done.add(_tkey(fam, vin, vout))
        except Exception:
            pass

    mode = "a" if bool(args.resume) else "w"
    with pairs_path.open(mode, encoding="utf-8") as f_pairs, sft_path.open(mode, encoding="utf-8") as f_sft, meta_path.open(
        mode, encoding="utf-8"
    ) as f_meta, done_path.open("a", encoding="utf-8") as f_done, progress_log_path.open("a", encoding="utf-8") as f_prog:
        for ti, task in enumerate(tasks):
            fam = str(task.family)
            vin = float(task.vin)
            vout = float(task.vout)
            key = _tkey(fam, vin, vout)
            if key in done:
                progress_path.write_text(
                    f"task={ti+1}/{len(tasks)} fam={fam} vin={vin} vout={vout} SKIP (already_done)\n",
                    encoding="utf-8",
                )
                continue

            pairs_written = 0
            status = "unknown"
            err_msg = ""

            try:
                prompt = build_module_prompt(fam, vin, vout, min_mods=int(args.min_mods))
                enc = tok(prompt, return_tensors="pt").to(dev)

                seed_base = int(args.seed) + ti * 1000
                random.seed(seed_base)
                torch.manual_seed(seed_base)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_base)

                with torch.inference_mode():
                    outs = model.generate(
                        **enc,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=True,
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.pad_token_id,
                        num_return_sequences=int(args.n_gen),
                    )

                candidates: List[Dict[str, Any]] = []
                for i in range(int(outs.shape[0])):
                    txt = tok.decode(outs[i], skip_special_tokens=True)
                    if _RESP_KEY in txt:
                        txt = txt.rsplit(_RESP_KEY, 1)[-1].lstrip()
                    # Keep only MOD lines for consistency.
                    calls = extract_module_calls(txt)
                    mod_text = "\n".join([c.raw.strip() for c in calls if c.raw.strip()]) + "\n"

                    fam2, inc, errs = compile_module_graph(mod_text, expected_family=fam)
                    if errs:
                        candidates.append({"ok": False, "error": "compile", "errs": errs, "mod": mod_text})
                        continue

                    ver = verify_inc_dcdc(inc, family=fam, vin=vin, vout=vout)
                    if (not ver.ok) or (int(ver.n_elems) < int(args.min_elems)):
                        candidates.append(
                            {
                                "ok": False,
                                "error": "verify",
                                "violations": list(ver.violations),
                                "n_elems": int(ver.n_elems),
                                "mod": mod_text,
                            }
                        )
                        continue

                    candidates.append({"ok": True, "mod": mod_text, "inc": inc, "n_elems": int(ver.n_elems)})

                ok_cands = [c for c in candidates if c.get("ok")]
                if len(ok_cands) < 2:
                    status = f"skip_ok<{len(ok_cands)}>"
                    progress_path.write_text(
                        f"task={ti+1}/{len(tasks)} fam={fam} vin={vin} vout={vout} SKIP (ok={len(ok_cands)})\n",
                        encoding="utf-8",
                    )
                    continue

                task_dir = out_root / "tasks" / fam / f"vin{vin:.1f}_vout{vout:.1f}"
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

                scored: List[Dict[str, Any]] = []
                for ci, c in enumerate(ok_cands):
                    mod_path = task_dir / f"cand_{ci+1}.mod.txt"
                    inc_path = task_dir / f"cand_{ci+1}.inc"
                    mod_path.write_text(c["mod"], encoding="utf-8")
                    inc_path.write_text(c["inc"], encoding="utf-8")

                    opt_dir = task_dir / f"opt_{ci+1}"
                    try:
                        t0 = time.time()
                        opt = _run_optimize(
                            python=str(args.python),
                            code_dir=code_dir,
                            family=fam,
                            vin=vin,
                            vout=vout,
                            inc_path=inc_path,
                            outdir=opt_dir,
                            budget=int(args.opt_budget),
                            pop=int(args.opt_pop),
                            elite=int(args.opt_elite),
                            seed=int(args.seed) + ti * 1000 + ci,
                            robust=bool(args.robust),
                            vin_jitter=float(args.vin_jitter),
                            rload_list=str(args.rload_list),
                            agg=str(args.agg),
                            cvar_alpha=float(args.cvar_alpha),
                        )
                        dt = time.time() - t0
                        scored.append(
                            {
                                "cand_i": int(ci + 1),
                                "mod": c["mod"],
                                "inc": c["inc"],
                                "n_elems": int(c["n_elems"]),
                                "opt_best_score": float(opt["summary"].get("best_score", -1e9)),
                                "opt_summary": opt["summary"],
                                "opt_wall_s": float(dt),
                            }
                        )
                    except Exception as e:
                        (opt_dir / "opt_failed.txt").write_text(str(e), encoding="utf-8")
                        continue

                if len(scored) < 2:
                    status = "skip_scored<2"
                    progress_path.write_text(
                        f"task={ti+1}/{len(tasks)} fam={fam} vin={vin} vout={vout} SKIP (scored<2)\n",
                        encoding="utf-8",
                    )
                    continue

                scored.sort(key=lambda x: float(x.get("opt_best_score", -1e9)), reverse=True)

                k = min(int(args.pairs_per_task), int(len(scored) // 2))
                for pi in range(int(k)):
                    chosen = scored[pi]
                    rejected = scored[-(pi + 1)]
                    gap = float(chosen.get("opt_best_score", -1e9)) - float(rejected.get("opt_best_score", -1e9))
                    if gap < float(args.min_pair_gap):
                        continue

                    f_pairs.write(
                        json.dumps({"prompt": prompt, "chosen": chosen["mod"], "rejected": rejected["mod"]}, ensure_ascii=False)
                        + "\n"
                    )
                    f_pairs.flush()
                    f_sft.write(json.dumps({"text": prompt + chosen["mod"]}, ensure_ascii=False) + "\n")
                    f_sft.flush()
                    f_meta.write(
                        json.dumps(
                            {
                                "task": asdict(task),
                                "pair_i": int(pi + 1),
                                "chosen": {k: chosen[k] for k in ["cand_i", "n_elems", "opt_best_score", "opt_wall_s"]},
                                "rejected": {k: rejected[k] for k in ["cand_i", "n_elems", "opt_best_score", "opt_wall_s"]},
                                "gap": float(gap),
                                "n_scored": len(scored),
                                "robust": bool(args.robust),
                                "vin_jitter": float(args.vin_jitter),
                                "rload_list": str(args.rload_list),
                                "agg": str(args.agg),
                                "cvar_alpha": float(args.cvar_alpha),
                                "task_dir": str(task_dir),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f_meta.flush()
                    pairs_written += 1

                status = "done"
                progress_path.write_text(
                    f"task={ti+1}/{len(tasks)} fam={fam} vin={vin} vout={vout} pairs=+{pairs_written} top={scored[0]['opt_best_score']:.4f} bot={scored[-1]['opt_best_score']:.4f}\n",
                    encoding="utf-8",
                )
            except Exception as e:
                status = "error"
                err_msg = f"{type(e).__name__}: {e}"
                progress_path.write_text(
                    f"task={ti+1}/{len(tasks)} fam={fam} vin={vin} vout={vout} ERROR {err_msg}\n",
                    encoding="utf-8",
                )
            finally:
                rec = {
                    "ts": time.strftime("%Y%m%d_%H%M%S"),
                    "task_i": int(ti + 1),
                    "task_n": int(len(tasks)),
                    "family": str(fam),
                    "vin": float(vin),
                    "vout": float(vout),
                    "status": str(status),
                    "pairs_written": int(pairs_written),
                    "robust": bool(args.robust),
                    "vin_jitter": float(args.vin_jitter),
                    "rload_list": str(args.rload_list),
                    "agg": str(args.agg),
                    "cvar_alpha": float(args.cvar_alpha),
                    "error": str(err_msg),
                }
                f_done.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_done.flush()
                f_prog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_prog.flush()
                done.add(key)

    print("[OK] wrote", str(pairs_path))
    print("[OK] wrote", str(sft_path))
    print("[OK] wrote", str(meta_path))


if __name__ == "__main__":
    raise SystemExit(main())

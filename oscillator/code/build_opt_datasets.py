#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from dcdc_taskset import Task, default_taskset
from dcdc_templates import templates
from dcdc_eval_tran import eval_one_detail_dcdc
from dcdc_verifier import verify_inc_dcdc
from inc_parser import IncElem, parse_inc, to_inc_text


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _score(detail: dict, vout: float, tol_ref: float = 0.1) -> float:
    if not detail.get("ok"):
        return -1.0

    vavg = float(detail.get("vavg", 0.0) or 0.0)
    eff = float(detail.get("eff", 0.0) or 0.0)
    ripple = float(detail.get("ripple", 0.0) or 0.0)
    overshoot = float(detail.get("overshoot", 0.0) or 0.0)

    err = abs(vavg - float(vout)) / max(1e-6, float(vout))
    score_v = 3.0 * max(0.0, 1.0 - (err / max(1e-6, float(tol_ref))))
    score_eff = 0.5 * eff
    score_ripple = -0.2 * (ripple / max(1e-6, float(vout)))
    score_over = -0.2 * overshoot

    if not detail.get("pass_CV", False):
        score_v -= 1.0
    if not detail.get("pass_CE", False):
        score_eff -= 0.5

    return float(score_v + score_eff + score_ripple + score_over)


def _prompt_for(task: Task) -> str:
    fam = task.family
    if fam == "buck":
        return (
            "Generate a Buck converter in INC DSL. Use only nodes {vin,sw,out,0}. "
            "Use elements {L,C,D,S} with values (e.g., 47u). Output only INC lines.\n"
            f"Task: Vin={task.vin:.1f}V, Vout={task.vout:.1f}V, Rload=10ohm.\n"
        )
    if fam == "boost":
        return (
            "Generate a Boost converter in INC DSL. Use only nodes {vin,sw,out,0}. "
            "Typical pattern: L vin-sw, S sw-0, D sw-out, C out-0. Output only INC lines.\n"
            f"Task: Vin={task.vin:.1f}V, Vout={task.vout:.1f}V, Rload=10ohm.\n"
        )
    if fam == "sepic":
        return (
            "Generate a SEPIC converter in INC DSL. Allowed nodes: {vin,sw,n1,out,0}. "
            "Use 2 inductors and 2 capacitors: L1 vin-sw, C1 sw-n1, L2 n1-0, S1 sw-0, D1 n1-out, C2 out-0. "
            "Output only INC lines.\n"
            f"Task: Vin={task.vin:.1f}V, Vout={task.vout:.1f}V, Rload=10ohm.\n"
        )
    if fam == "buckboost":
        return (
            "Generate a non-inverting Buck-Boost (cascaded buck->boost) in INC DSL. "
            "Allowed nodes: {vin,sw1,mid,sw2,out,0}. Use 2 switches and 2 diodes. "
            "Use switch models Sstd1 and Sstd2 so gate1/gate2 are separate. Output only INC lines.\n"
            f"Task: Vin={task.vin:.1f}V, Vout={task.vout:.1f}V, Rload=10ohm.\n"
        )
    raise ValueError(f"unknown family: {task.family}")


def _rand_passives(inc_text: str, rng: np.random.Generator, sigma: float = 0.35) -> str:
    elems = parse_inc(inc_text)
    out: List[IncElem] = []
    for e in elems:
        if e.kind in {"L", "C"} and e.value is not None:
            val = float(e.value)
            val2 = val * float(math.exp(rng.normal(0.0, sigma)))
            if e.kind == "L":
                val2 = _clamp(val2, 1e-6, 800e-6)
            else:
                val2 = _clamp(val2, 1e-9, 2000e-6)
            out.append(IncElem(name=e.name, kind=e.kind, nodes=list(e.nodes), value=val2, model=e.model, raw=e.raw))
        else:
            out.append(e)
    return to_inc_text(out) + "\n"


def _run_optimizer(
    code_dir: Path,
    family: str,
    vin: float,
    vout: float,
    seed_inc_path: Path,
    outdir: Path,
    budget: int,
    pop: int,
    elite: int,
    seed: int,
) -> None:
    py = os.environ.get("PY", "") or "python"
    cmd = [
        py,
        str(code_dir / "optimize_dcdc_params.py"),
        "--family",
        str(family),
        "--vin",
        str(float(vin)),
        "--vout",
        str(float(vout)),
        "--inc_file",
        str(seed_inc_path),
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
        "--opt_duty",
        "--opt_freq",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_dir)
    subprocess.run(cmd, check=True, env=env)


def _max_done_i(report_path: Path) -> int:
    if not report_path.exists() or report_path.stat().st_size == 0:
        return -1
    try:
        with report_path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            mx = -1
            for row in r:
                try:
                    mx = max(mx, int(float(row.get("i", -1))))
                except Exception:
                    continue
            return int(mx)
    except Exception:
        return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--max_tasks", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--min_improve", type=float, default=0.2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))

    code_dir = Path(__file__).resolve().parent
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = default_taskset()[: int(args.max_tasks)]
    tpls = templates()

    pairs_path = out_root / "dpo_pairs.jsonl"
    sft_path = out_root / "sft_train.jsonl"
    meta_path = out_root / "pairs_meta.jsonl"
    report_path = out_root / "build_report.csv"
    progress_path = out_root / "progress.txt"

    done_i = _max_done_i(report_path) if args.resume else -1

    mode_pairs = "a" if (args.resume and pairs_path.exists()) else "w"
    mode_sft = "a" if (args.resume and sft_path.exists()) else "w"
    mode_meta = "a" if (args.resume and meta_path.exists()) else "w"
    mode_csv = "a" if (args.resume and report_path.exists() and report_path.stat().st_size > 0) else "w"

    with pairs_path.open(mode_pairs, encoding="utf-8", buffering=1) as f_pairs, sft_path.open(
        mode_sft, encoding="utf-8", buffering=1
    ) as f_sft, meta_path.open(mode_meta, encoding="utf-8", buffering=1) as f_meta, report_path.open(
        mode_csv, newline="", encoding="utf-8", buffering=1
    ) as f_csv:
        w = csv.DictWriter(
            f_csv,
            fieldnames=[
                "i",
                "family",
                "vin",
                "vout",
                "seed_score",
                "opt_score",
                "improve",
                "seed_pass_cv",
                "opt_pass_cv",
                "seed_vavg",
                "opt_vavg",
                "seed_eff",
                "opt_eff",
                "seed_ripple",
                "opt_ripple",
            ],
        )
        if mode_csv == "w":
            w.writeheader()
            f_csv.flush()

        n_written = sum(1 for _ in pairs_path.open("r", encoding="utf-8")) if pairs_path.exists() else 0

        for i, task in enumerate(tasks):
            if i <= done_i:
                continue

            fam = task.family
            tpl = tpls.get(fam)
            if not tpl:
                continue

            prompt = _prompt_for(task)

            # seed inc: randomize template passives
            inc_seed = _rand_passives(tpl, rng)
            ver = verify_inc_dcdc(inc_seed, family=fam, vin=task.vin, vout=task.vout)
            if not ver.ok:
                inc_seed = tpl

            seed_detail = eval_one_detail_dcdc(
                inc=inc_seed,
                family=fam,
                vin=task.vin,
                vout=task.vout,
                tol=0.1,
                rload=10.0,
                t_pre=0.008,
                t_win=0.002,
            )
            seed_score = _score(seed_detail, vout=task.vout, tol_ref=0.1)

            task_dir = out_root / "tasks" / fam / f"vin{task.vin:.1f}_vout{task.vout:.1f}"
            task_dir.mkdir(parents=True, exist_ok=True)
            seed_inc_path = task_dir / "seed.inc"
            seed_inc_path.write_text(inc_seed, encoding="utf-8")

            opt_dir = task_dir / "opt"
            opt_dir.mkdir(parents=True, exist_ok=True)

            try:
                if not (opt_dir / "summary.json").exists():
                    _run_optimizer(
                        code_dir=code_dir,
                        family=fam,
                        vin=task.vin,
                        vout=task.vout,
                        seed_inc_path=seed_inc_path,
                        outdir=opt_dir,
                        budget=int(args.budget),
                        pop=int(args.pop),
                        elite=int(args.elite),
                        seed=int(args.seed) + i,
                    )
            except Exception as e:
                (task_dir / "opt_failed.txt").write_text(str(e), encoding="utf-8")
                continue

            summary = json.loads((opt_dir / "summary.json").read_text(encoding="utf-8"))
            opt_score = float(summary.get("best_score", -1e9))
            opt_detail = summary.get("best_detail", {}) or {}
            inc_opt = (opt_dir / "best_inc.txt").read_text(encoding="utf-8")

            improve = float(opt_score - float(seed_score))

            if improve >= float(args.min_improve) and bool(opt_detail.get("pass_CV", False)):
                chosen = inc_opt.strip() + "\n"
                rejected = inc_seed.strip() + "\n"

                f_pairs.write(
                    json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n"
                )
                f_pairs.flush()

                f_sft.write(json.dumps({"text": prompt + chosen}, ensure_ascii=False) + "\n")
                f_sft.flush()

                f_meta.write(
                    json.dumps(
                        {
                            "task": asdict(task),
                            "seed": {"detail": seed_detail, "score": seed_score, "inc": rejected},
                            "opt": {"detail": opt_detail, "score": opt_score, "inc": chosen},
                            "task_dir": str(task_dir),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f_meta.flush()
                n_written += 1

            w.writerow(
                {
                    "i": i,
                    "family": fam,
                    "vin": float(task.vin),
                    "vout": float(task.vout),
                    "seed_score": float(seed_score),
                    "opt_score": float(opt_score),
                    "improve": float(improve),
                    "seed_pass_cv": bool(seed_detail.get("pass_CV", False)) if seed_detail.get("ok") else False,
                    "opt_pass_cv": bool(opt_detail.get("pass_CV", False)),
                    "seed_vavg": float(seed_detail.get("vavg", 0.0) or 0.0) if seed_detail.get("ok") else 0.0,
                    "opt_vavg": float(opt_detail.get("vavg", 0.0) or 0.0) if opt_detail else 0.0,
                    "seed_eff": float(seed_detail.get("eff", 0.0) or 0.0) if seed_detail.get("ok") else 0.0,
                    "opt_eff": float(opt_detail.get("eff", 0.0) or 0.0) if opt_detail else 0.0,
                    "seed_ripple": float(seed_detail.get("ripple", 0.0) or 0.0) if seed_detail.get("ok") else 0.0,
                    "opt_ripple": float(opt_detail.get("ripple", 0.0) or 0.0) if opt_detail else 0.0,
                }
            )
            f_csv.flush()

            progress_path.write_text(f"i={i} written_pairs={n_written}\n", encoding="utf-8")

    print("[OK] pairs:", sum(1 for _ in pairs_path.open('r', encoding='utf-8')))


if __name__ == "__main__":
    main()

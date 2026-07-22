#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def _pid_running(pid: int) -> bool:
    try:
        p = Path(f"/proc/{int(pid)}")
        return p.exists()
    except Exception:
        return False


def _count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    n = 0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def _run(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[CMD] " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--python", default="/root/miniconda3/bin/python")
    ap.add_argument("--poll_s", type=int, default=60)
    ap.add_argument("--min_pairs", type=int, default=100)
    ap.add_argument("--min_improve", type=float, default=0.2)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    code_dir = Path(__file__).resolve().parent
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    pidfile = run_root / "build_opt_datasets.pid"
    if not pidfile.exists():
        raise SystemExit(f"missing pidfile: {pidfile}")

    pid = int(pidfile.read_text().strip())
    print(f"[watch] waiting for build_opt_datasets pid={pid}", flush=True)
    while _pid_running(pid):
        time.sleep(max(5, int(args.poll_s)))

    print("[watch] build_opt_datasets finished, rebuilding datasets from tasks", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_dir)

    # Rebuild datasets from completed task dirs (recovers pairs even if buffering/resume issues)
    _run(
        [
            args.python,
            str(code_dir / "make_datasets_from_tasks.py"),
            "--run_root",
            str(run_root),
            "--min_improve",
            str(float(args.min_improve)),
            "--require_pass_cv",
        ],
        logs / "make_datasets_from_tasks.out",
        env,
    )

    pairs_path = run_root / "dpo_pairs.jsonl"
    n_pairs = _count_lines(pairs_path)
    print(f"[watch] pairs={n_pairs}", flush=True)

    if n_pairs < int(args.min_pairs):
        raise SystemExit(f"too few pairs ({n_pairs}) for training; increase build budget/tasks")

    # SFT
    sft_out = run_root / "sft_stage"
    _run(
        [
            args.python,
            str(code_dir / "train_sft_dcdc.py"),
            "--base_model",
            args.base_model,
            "--sft_jsonl",
            str(run_root / "sft_train.jsonl"),
            "--outdir",
            str(sft_out),
        ],
        logs / "train_sft.out",
        env,
    )

    # DPO
    dpo_out = run_root / "dpo_stage"
    _run(
        [
            args.python,
            str(code_dir / "train_dpo_dcdc.py"),
            "--base_model",
            args.base_model,
            "--sft_adapter",
            str(sft_out / "sft_final"),
            "--dpo_pairs",
            str(run_root / "dpo_pairs.jsonl"),
            "--outdir",
            str(dpo_out),
        ],
        logs / "train_dpo.out",
        env,
    )

    print("[watch] DONE: SFT+DPO finished", flush=True)


if __name__ == "__main__":
    main()

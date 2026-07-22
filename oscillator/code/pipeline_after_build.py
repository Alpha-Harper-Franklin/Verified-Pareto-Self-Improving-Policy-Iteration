#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _pid_running(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    return Path(f"/proc/{int(pid)}").exists()


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


def _concat_jsonl(out_path: Path, *inputs: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for inp in inputs:
            if not inp.exists():
                continue
            with inp.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    out.write(line.rstrip("\n") + "\n")
                    n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--pid", type=int, default=0, help="Optional: wait for this dataset-build PID to exit")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--poll_s", type=int, default=60)
    ap.add_argument("--min_pairs", type=int, default=20)

    # SFT hyperparams (kept lightweight by default)
    ap.add_argument("--sft_epochs", type=int, default=1)
    ap.add_argument("--sft_bsz", type=int, default=1)
    ap.add_argument("--sft_grad_accum", type=int, default=16)
    ap.add_argument("--sft_lr", type=float, default=2e-5)

    # DPO hyperparams
    ap.add_argument("--dpo_epochs", type=int, default=1)
    ap.add_argument("--dpo_bsz", type=int, default=1)
    ap.add_argument("--dpo_grad_accum", type=int, default=16)
    ap.add_argument("--dpo_lr", type=float, default=5e-6)

    # checkpointing
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    code_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_dir)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    status_path = run_dir / "pipeline_status.json"
    status: dict = {
        "run_dir": str(run_dir),
        "base_model": str(args.base_model),
        "pid_wait": int(args.pid),
        "started_at": time.strftime("%Y%m%d_%H%M%S"),
        "stages": {},
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    if int(args.pid) > 0:
        print(f"[pipeline] waiting for pid={int(args.pid)}", flush=True)
        while _pid_running(int(args.pid)):
            pairs = _count_lines(run_dir / "modulegraph_data" / "dpo_pairs.jsonl")
            (run_dir / "logs" / "pipeline_heartbeat.txt").write_text(
                f"time={time.strftime('%Y%m%d_%H%M%S')} pid={int(args.pid)} pairs={pairs}\n",
                encoding="utf-8",
            )
            time.sleep(max(5, int(args.poll_s)))
        print("[pipeline] dataset build finished", flush=True)

    pairs_path = run_dir / "modulegraph_data" / "dpo_pairs.jsonl"
    sft_mod_path = run_dir / "modulegraph_data" / "sft_train.jsonl"
    official_path = run_dir / "official_sft.jsonl"
    if not pairs_path.exists():
        raise SystemExit(f"missing pairs: {pairs_path}")
    if not sft_mod_path.exists():
        raise SystemExit(f"missing modulegraph sft: {sft_mod_path}")
    n_pairs = _count_lines(pairs_path)
    if n_pairs < int(args.min_pairs):
        raise SystemExit(f"too few pairs ({n_pairs}) < min_pairs={int(args.min_pairs)}: {pairs_path}")

    # Merge SFT sources (authoritative official subset + generated module-graph supervision).
    sft_mix = run_dir / "sft_mix.jsonl"
    n_sft = _concat_jsonl(sft_mix, official_path, sft_mod_path)
    print(f"[pipeline] sft_mix={n_sft} lines -> {sft_mix}", flush=True)

    # SFT stage
    sft_out = run_dir / "sft_stage"
    sft_cmd = [
        str(args.python),
        str(code_dir / "train_sft_dcdc.py"),
        "--base_model",
        str(args.base_model),
        "--sft_jsonl",
        str(sft_mix),
        "--outdir",
        str(sft_out),
        "--epochs",
        str(int(args.sft_epochs)),
        "--bsz",
        str(int(args.sft_bsz)),
        "--grad_accum",
        str(int(args.sft_grad_accum)),
        "--lr",
        str(float(args.sft_lr)),
        "--save_steps",
        str(int(args.save_steps)),
        "--save_total_limit",
        str(int(args.save_total_limit)),
    ]
    if bool(args.resume):
        sft_cmd += ["--resume"]
    _run(sft_cmd, logs_dir / "train_sft.out", env=env)
    status["stages"]["sft"] = {"outdir": str(sft_out), "finished_at": time.strftime("%Y%m%d_%H%M%S")}
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    # DPO stage
    dpo_out = run_dir / "dpo_stage"
    dpo_cmd = [
        str(args.python),
        str(code_dir / "train_dpo_dcdc.py"),
        "--base_model",
        str(args.base_model),
        "--sft_adapter",
        str(sft_out / "sft_final"),
        "--dpo_pairs",
        str(pairs_path),
        "--outdir",
        str(dpo_out),
        "--epochs",
        str(int(args.dpo_epochs)),
        "--bsz",
        str(int(args.dpo_bsz)),
        "--grad_accum",
        str(int(args.dpo_grad_accum)),
        "--lr",
        str(float(args.dpo_lr)),
        "--save_steps",
        str(int(args.save_steps)),
        "--save_total_limit",
        str(int(args.save_total_limit)),
    ]
    if bool(args.resume):
        dpo_cmd += ["--resume"]
    _run(dpo_cmd, logs_dir / "train_dpo.out", env=env)
    status["stages"]["dpo"] = {"outdir": str(dpo_out), "finished_at": time.strftime("%Y%m%d_%H%M%S")}

    status["finished_at"] = time.strftime("%Y%m%d_%H%M%S")
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[pipeline] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


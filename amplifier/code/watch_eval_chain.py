#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def _pid_running(pid: int) -> bool:
    try:
        return Path(f"/proc/{int(pid)}").exists()
    except Exception:
        return False


def _read_pid(pidfile: Path) -> int | None:
    try:
        s = pidfile.read_text().strip()
        return int(s)
    except Exception:
        return None


def _run_stage(
    *,
    python: str,
    code_dir: Path,
    base_model: str,
    adapter: str,
    outdir: Path,
    tag: str,
    n_per_task: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tol: float,
    constrained: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    logs = outdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "eval.out"
    pid_path = logs / "eval.pid"
    summary_path = outdir / "eval_summary.json"

    if summary_path.exists():
        print(f"[watch] {tag}: already finished ({summary_path})", flush=True)
        return

    pid = _read_pid(pid_path)
    if pid and _pid_running(pid):
        print(f"[watch] {tag}: already running pid={pid}", flush=True)
        return

    cmd = [
        python,
        str(code_dir / "eval_dcdc_family.py"),
        "--base_model",
        base_model,
        "--outdir",
        str(outdir),
        "--n_per_task",
        str(int(n_per_task)),
        "--max_new_tokens",
        str(int(max_new_tokens)),
        "--temperature",
        str(float(temperature)),
        "--top_p",
        str(float(top_p)),
        "--tol",
        str(float(tol)),
        "--resume",
    ]
    if adapter:
        cmd += ["--adapter", adapter]
    if constrained:
        cmd += ["--constrained"]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_dir)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    with log_path.open("a", encoding="utf-8") as f:
        f.write("[CMD] " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(code_dir), stdout=f, stderr=subprocess.STDOUT, env=env)
        pid_path.write_text(str(int(p.pid)) + "\n", encoding="utf-8")

    print(f"[watch] {tag}: started pid={p.pid}", flush=True)
    rc = p.wait()
    print(f"[watch] {tag}: exit rc={rc}", flush=True)

    if not summary_path.exists():
        raise RuntimeError(f"{tag} finished but missing summary: {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--code_dir", default="/root/autodl-tmp/dcdc_family/code")
    ap.add_argument("--python", default="/root/miniconda3/bin/python")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--eval_base", default="eval_base")
    ap.add_argument("--eval_sft", default="eval_sft")
    ap.add_argument("--eval_dpo", default="eval_stage_llm")
    ap.add_argument("--n_per_task", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--poll_s", type=int, default=30)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    code_dir = Path(args.code_dir)

    sft_adapter = run_root / "sft_stage" / "sft_final"
    dpo_adapter = run_root / "dpo_stage" / "dpo_final"

    # 1) wait base to finish (or start if missing)
    base_dir = run_root / str(args.eval_base)
    while True:
        if (base_dir / "eval_summary.json").exists():
            break
        pid = _read_pid(base_dir / "logs" / "eval.pid") if (base_dir / "logs").exists() else None
        if not (pid and _pid_running(pid)):
            _run_stage(
                python=str(args.python),
                code_dir=code_dir,
                base_model=str(args.base_model),
                adapter="",
                outdir=base_dir,
                tag="eval_base",
                n_per_task=int(args.n_per_task),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                tol=float(args.tol),
                constrained=bool(args.constrained),
            )
            break
        time.sleep(max(5, int(args.poll_s)))

    # 2) run sft eval
    _run_stage(
        python=str(args.python),
        code_dir=code_dir,
        base_model=str(args.base_model),
        adapter=str(sft_adapter),
        outdir=run_root / str(args.eval_sft),
        tag="eval_sft",
        n_per_task=int(args.n_per_task),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        tol=float(args.tol),
        constrained=bool(args.constrained),
    )

    # 3) dpo eval is already in eval_stage_llm, but ensure it exists; otherwise run it
    dpo_dir = run_root / str(args.eval_dpo)
    if not (dpo_dir / "eval_summary.json").exists():
        _run_stage(
            python=str(args.python),
            code_dir=code_dir,
            base_model=str(args.base_model),
            adapter=str(dpo_adapter),
            outdir=dpo_dir,
            tag="eval_dpo",
            n_per_task=int(args.n_per_task),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            tol=float(args.tol),
            constrained=bool(args.constrained),
        )

    # 4) summarize comparison
    cmd = [
        str(args.python),
        str(code_dir / "summarize_dcdc_eval.py"),
        "--run_root",
        str(run_root),
        "--eval_base",
        str(args.eval_base),
        "--eval_sft",
        str(args.eval_sft),
        "--eval_dpo",
        str(args.eval_dpo),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_dir)
    subprocess.run(cmd, check=True, env=env)
    print("[watch] DONE: eval base+sft+dpo summarized", flush=True)


if __name__ == "__main__":
    main()

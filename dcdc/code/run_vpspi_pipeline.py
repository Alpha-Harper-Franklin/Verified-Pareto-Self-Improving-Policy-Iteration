#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dcdc_taskset import Task
from task_manifest import ensure_disjoint, load_tasks_jsonl, sha256_file, task_key


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


REDLINE_TASKS: List[Tuple[str, float, float]] = [
    ("buck", 12.0, 5.0),
    ("buck", 18.0, 5.0),
    ("boost", 5.0, 12.0),
    ("boost", 9.0, 18.0),
    ("sepic", 12.0, 5.0),
    ("sepic", 5.0, 12.0),
    ("buckboost", 12.0, 5.0),
    ("buckboost", 9.0, 12.0),
]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def _snapshot_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst)
        return
    if src.is_file():
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / src.name)
        return
    raise FileNotFoundError(f"snapshot src not found: {src}")


def _run(cmd: List[str], *, cwd: Path, log_path: Path, env: Dict[str, str] | None = None) -> None:
    _append_text(log_path, "$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
    with log_path.open("a", encoding="utf-8") as f:
        subprocess.check_call(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=env)


def _gpu_count() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    cvd = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not cvd:
        return 0
    parts = [p.strip() for p in cvd.split(",") if p.strip()]
    return int(len(parts))




def _train_env_single_gpu() -> Dict[str, str]:
    """
    Ensure single-process training does NOT trigger torch.nn.DataParallel when multiple GPUs are visible.

    HF Trainer will use DataParallel for n_gpu>1 in a single process, which is slower and can be unstable on some
    systems. We force a single visible GPU for warmup/DPO/PPO when nproc==1.
    """

    env = os.environ.copy()
    cvd = str(env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if cvd:
        parts = [p.strip() for p in cvd.split(",") if p.strip()]
        env["CUDA_VISIBLE_DEVICES"] = parts[0] if parts else "0"
    else:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env

def _cpu_quota() -> int:
    """
    Return effective cgroup CPU quota (ceil), or fall back to nproc/os.cpu_count.

    Note: on this platform `os.cpu_count()` may report the host (e.g., 208)
    while the container quota is smaller (e.g., 176).
    """
    try:
        p = Path("/sys/fs/cgroup/cpu.max")
        if p.exists():
            parts = p.read_text().strip().split()
            if len(parts) >= 2 and parts[0].strip().lower() != "max":
                quota = float(parts[0])
                period = float(parts[1])
                if quota > 0 and period > 0:
                    return int(max(1, math.ceil(quota / period)))
    except Exception:
        pass
    try:
        return int(max(1, subprocess.check_output(["nproc"]).decode().strip()))
    except Exception:
        pass
    try:
        return int(max(1, os.cpu_count() or 1))
    except Exception:
        return 1


def _merge_selfplay_shards(*, sp_dir: Path, shard_dirs: List[Path], log_path: Path) -> None:
    """
    Merge per-shard self-play outputs into `sp_dir`.

    This is intentionally append-only for large JSONL files; we write a per-shard
    marker file to make the merge idempotent across resumes.
    """
    sp_dir.mkdir(parents=True, exist_ok=True)
    (sp_dir / "tasks").mkdir(parents=True, exist_ok=True)
    (sp_dir / "logs").mkdir(parents=True, exist_ok=True)

    merged_any = False
    for shard in shard_dirs:
        marker = shard / ".MERGED_INTO_SELFPLAY_DATA.json"
        if marker.exists():
            continue

        if not shard.exists():
            continue

        # Copy per-task scored.json artifacts (used for debugging/analysis).
        shard_tasks = shard / "tasks"
        if shard_tasks.exists():
            for fam_dir in sorted([p for p in shard_tasks.iterdir() if p.is_dir()]):
                for tdir in sorted([p for p in fam_dir.iterdir() if p.is_dir()]):
                    dst = sp_dir / "tasks" / fam_dir.name / tdir.name
                    if dst.exists():
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(tdir, dst)

        # Append global JSONL artifacts.
        appended: Dict[str, int] = {}
        for name in ["dpo_pairs.jsonl", "pairs_meta.jsonl", "sft_train.jsonl", "sft_train_strict.jsonl"]:
            src = shard / name
            if not src.exists() or src.stat().st_size <= 0:
                continue
            dst = sp_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Count lines before/after for the marker (cheap enough at merge time).
            before = 0
            if dst.exists():
                try:
                    before = sum(1 for _ in dst.open("r", encoding="utf-8", errors="ignore"))
                except Exception:
                    before = 0
            with src.open("rb") as rf, dst.open("ab") as wf:
                shutil.copyfileobj(rf, wf)
            after = before
            try:
                after = sum(1 for _ in dst.open("r", encoding="utf-8", errors="ignore"))
            except Exception:
                pass
            appended[name] = int(max(0, after - before))

        # Merge done_tasks as a de-duplicated union (small file; safe to rewrite).
        done: Dict[Tuple[str, float, float], Dict[str, Any]] = {}
        for p in [sp_dir / "done_tasks.jsonl", shard / "done_tasks.jsonl"]:
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    key = (str(r["family"]).lower(), float(r["vin"]), float(r["vout"]))
                    done[key] = {"family": str(r["family"]).lower(), "vin": float(r["vin"]), "vout": float(r["vout"])}
                except Exception:
                    continue
        if done:
            out_lines = [json.dumps(v, ensure_ascii=False) for _, v in sorted(done.items())]
            (sp_dir / "done_tasks.jsonl").write_text("\n".join(out_lines) + "\n", encoding="utf-8")

        marker.write_text(
            json.dumps(
                {
                    "merged_at": _now(),
                    "shard_dir": str(shard),
                    "appended_lines": appended,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        merged_any = True
        _append_text(log_path, f"[selfplay_merge] merged shard {shard}\n")

    if not merged_any:
        return

    # Build a lightweight report in sp_dir for reproducibility (counts only).
    report: Dict[str, Any] = {"merged_at": _now(), "counts": {}}
    for name in ["dpo_pairs.jsonl", "pairs_meta.jsonl", "sft_train.jsonl", "sft_train_strict.jsonl", "done_tasks.jsonl"]:
        p = sp_dir / name
        if not p.exists():
            continue
        try:
            report["counts"][name] = int(sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore")))
        except Exception:
            report["counts"][name] = None
    _write_json(sp_dir / "build_report.json", report)


def _run_selfplay_sharded(
    *,
    code_dir: Path,
    base_sp_cmd: List[str],
    sp_dir: Path,
    ridx: int,
    n_shards: int,
    sim_workers_total: int,
    log_path: Path,
) -> None:
    shards_root = sp_dir.parent / "selfplay_shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    # Skip tasks already completed in the merged output (important when resuming).
    skip_done = shards_root / "skip_done_tasks.jsonl"
    legacy_done = sp_dir / "done_tasks.jsonl"
    skip_done.write_text(legacy_done.read_text(encoding="utf-8", errors="ignore") if legacy_done.exists() else "", encoding="utf-8")

    gpus = _gpu_count()
    if gpus <= 0:
        raise SystemExit("[selfplay_sharded] no GPUs visible (torch.cuda.device_count()==0)")
    n_shards = max(1, min(int(n_shards), int(gpus)))

    # Preserve the parent's physical GPU assignment. Replacing a parent value
    # such as CUDA_VISIBLE_DEVICES=2,3 with 0 or 1 would redirect concurrent
    # seed workers onto physical GPUs 0 and 1.
    parent_cvd = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if parent_cvd:
        visible_gpu_ids = [x.strip() for x in parent_cvd.split(",") if x.strip()]
    else:
        visible_gpu_ids = [str(i) for i in range(int(gpus))]
    if len(visible_gpu_ids) < int(gpus):
        raise SystemExit(
            f"[selfplay_sharded] CUDA_VISIBLE_DEVICES exposes {visible_gpu_ids}, "
            f"but torch reports {gpus} visible GPUs"
        )

    cpu_budget = int(sim_workers_total) if int(sim_workers_total) > 0 else int(_cpu_quota())
    per_shard = int(max(1, int(cpu_budget) // int(n_shards)))

    _append_text(
        log_path,
        f"[selfplay_sharded] ridx={ridx} shards={n_shards} gpus={gpus} "
        f"visible_gpu_ids={visible_gpu_ids} cpu_budget={cpu_budget} "
        f"sim_workers_per_shard={per_shard}\n",
    )

    procs: List[Tuple[int, subprocess.Popen, Path, Path]] = []
    for sid in range(int(n_shards)):
        shard_dir = shards_root / f"shard_{sid:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        cmd = list(base_sp_cmd)
        # replace --out_root and --sim_workers (base_sp_cmd contains placeholders at the end)
        cmd += ["--out_root", str(shard_dir)]
        cmd += ["--sim_workers", str(int(per_shard))]
        cmd += ["--task_shard_id", str(int(sid)), "--task_shard_count", str(int(n_shards))]
        cmd += ["--skip_done_tasks_jsonl", str(skip_done)]

        # Always resume within shard dirs if present; safe across restarts.
        if (shard_dir / "done_tasks.jsonl").exists():
            cmd.append("--resume")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu_ids[int(sid) % len(visible_gpu_ids)]
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")

        logf = shards_root / f"shard_{sid:02d}.log"
        _append_text(log_path, "$ " + " ".join(shlex.quote(x) for x in cmd) + f"  # shard={sid}\n")
        f = logf.open("a", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=str(code_dir), stdout=f, stderr=subprocess.STDOUT, env=env)
        procs.append((int(sid), p, logf, shard_dir))

    failed: List[str] = []
    for sid, p, logf, shard_dir in procs:
        rc = int(p.wait())
        if rc != 0:
            failed.append(f"shard={sid} rc={rc} log={logf}")
        else:
            _append_text(log_path, f"[selfplay_sharded] shard_done sid={sid} dir={shard_dir}\n")

    if failed:
        _append_text(log_path, "[selfplay_sharded] failed:\n" + "\n".join(failed) + "\n")
        raise SystemExit("selfplay_sharded failed: " + "; ".join(failed))

    # Merge all shards into the canonical `sp_dir` (so downstream DPO/PPO paths remain unchanged).
    _merge_selfplay_shards(sp_dir=sp_dir, shard_dirs=[sd for _, _, _, sd in procs], log_path=log_path)


def _maybe_accelerate(cmd: List[str], *, nproc: int) -> List[str]:
    nproc = int(nproc)
    if nproc <= 1:
        return cmd
    exe = shutil.which("accelerate") or "accelerate"
    # Avoid port collisions from repeated launches in the same container.
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = int(s.getsockname()[1])
    # Use bf16 everywhere for consistency with our training configs.
    #
    # `cmd` is expected to be: [python, script.py, ...] (pipeline's default),
    # but we also support [script.py, ...] as a fallback.
    script = ""
    rest: List[str] = []
    if len(cmd) >= 2 and (Path(cmd[0]).name.startswith("python") or cmd[0].endswith("/python") or cmd[0].endswith("\\python")):
        script = cmd[1]
        rest = cmd[2:]
    else:
        script = cmd[0]
        rest = cmd[1:]
    return [
        exe,
        "launch",
        "--multi_gpu",
        "--num_processes",
        str(int(nproc)),
        "--mixed_precision",
        "bf16",
        "--main_process_port",
        str(port),
        script,
        *rest,
    ]


def _parse_eval_summary(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    results = obj.get("results") or {}
    fam_sum: Dict[str, Dict[str, float]] = {}
    tried_all = 0.0
    cv_all = 0.0
    for k, v in results.items():
        fam = str(k).split("/", 1)[0].strip().lower()
        tried = float((v or {}).get("n_unique") or 0.0)
        cv = float((v or {}).get("pass_CV") or 0.0)
        tried_all += tried
        cv_all += cv
        d = fam_sum.setdefault(fam, {"tried": 0.0, "cv": 0.0})
        d["tried"] += tried
        d["cv"] += cv
    cv_by_family = {fam: (d["cv"] / d["tried"] if d["tried"] > 0.0 else 0.0) for fam, d in fam_sum.items()}
    cv_rate = float(cv_all / tried_all) if tried_all > 0.0 else 0.0
    cv_min_family = float(min(cv_by_family.values())) if cv_by_family else 0.0
    return {"cv_rate": cv_rate, "cv_min_family": cv_min_family, "cv_by_family": cv_by_family, "raw": obj}


def _non_regress(*, cand: Dict[str, Any], best: Dict[str, Any], eps: float = 1e-9) -> Tuple[bool, List[str]]:
    degraded: List[str] = []
    bbf = best.get("cv_by_family") or {}
    cbf = cand.get("cv_by_family") or {}
    for fam, best_v in bbf.items():
        cur_v = float(cbf.get(fam, 0.0))
        if cur_v + eps < float(best_v):
            degraded.append(str(fam))
    if float(cand.get("cv_rate") or 0.0) + eps < float(best.get("cv_rate") or 0.0):
        degraded.append("__overall__")
    return (len(degraded) == 0), degraded


def _update_best(best: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    best = json.loads(json.dumps(best))
    best["cv_rate"] = float(max(float(best.get("cv_rate") or 0.0), float(cand.get("cv_rate") or 0.0)))
    bbf = {str(k).strip().lower(): float(v) for k, v in (best.get("cv_by_family") or {}).items()}
    cbf = {str(k).strip().lower(): float(v) for k, v in (cand.get("cv_by_family") or {}).items()}
    for fam, v in cbf.items():
        bbf[fam] = float(max(float(bbf.get(fam) or 0.0), float(v)))
    best["cv_by_family"] = bbf
    best["cv_min_family"] = float(min(bbf.values())) if bbf else float(best.get("cv_min_family") or 0.0)
    return best


@dataclass
class StageOutputs:
    adapter: str
    eval_redline: Dict[str, Any]


def _run_redline_eval(
    *,
    code_dir: Path,
    base_model: str,
    adapter: str,
    outdir: Path,
    tol: float,
    sim_timeout_s: float,
    min_elems: int,
    bank_expand_min_elems: bool,
    bank_expand_max_parts: int,
    n_per_task: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    constrained: bool,
    autotune_duty: bool,
    log_path: Path,
    tasks: List[Task] | None = None,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(code_dir / "eval_dcdc_family.py"),
        "--base_model",
        str(base_model),
        "--adapter",
        str(adapter),
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
        "--seed",
        str(int(seed)),
        "--tol",
        str(float(tol)),
        "--sim_timeout_s",
        str(float(sim_timeout_s)),
        "--min_elems",
        str(int(min_elems)),
        "--no_fallback",
    ]
    if bool(bank_expand_min_elems):
        cmd.append("--bank_expand_min_elems")
        cmd += ["--bank_expand_max_parts", str(int(bank_expand_max_parts))]
    if bool(constrained):
        cmd.append("--constrained")
    if bool(autotune_duty):
        cmd.append("--autotune_duty")
    eval_tasks = [task_key(task) for task in tasks] if tasks else list(REDLINE_TASKS)
    for fam, vin, vout in eval_tasks:
        cmd += ["--only_task", f"{fam},{vin},{vout}"]
    _run(cmd, cwd=code_dir, log_path=log_path)

    summ = outdir / "eval_summary.json"
    if not summ.exists():
        raise SystemExit(f"Missing eval summary: {summ}")
    parsed = _parse_eval_summary(summ)
    _write_json(outdir / "redline_metrics.json", parsed)
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--anchor_adapter", required=True, help="SFT anchor adapter π0 (VP-SPI anchor).")
    ap.add_argument("--run_root", default="/root/autodl-tmp/dcdc_family/runs")
    ap.add_argument("--name", default="", help="Optional run name suffix.")
    ap.add_argument(
        "--outdir",
        default="",
        help="Optional: reuse an existing run directory. If it exists, must pass --resume. When set, --run_root/--name are ignored.",
    )
    ap.add_argument("--resume", action="store_true", help="Resume/append to an existing --outdir (reuses best_cv_guard.json if present).")
    ap.add_argument("--rounds", type=int, default=1, help="VP-SPI iterations (A→C→D). 0 means loop forever until killed.")
    ap.add_argument(
        "--train_tasks_jsonl",
        default="",
        help="Frozen training-task JSONL. Only these tasks may contribute candidates or parameter updates.",
    )
    ap.add_argument(
        "--guard_tasks_jsonl",
        default="",
        help="Frozen validation-task JSONL used only for model acceptance and rollback.",
    )

    # Global requirements
    ap.add_argument("--tol", type=float, default=0.01, help="Fixed CV tolerance (forced to 0.01).")
    ap.add_argument("--min_elems", type=int, default=20)

    # Redline eval config (non-regression constraint)
    ap.add_argument("--redline_n_per_task", type=int, default=2)
    ap.add_argument("--redline_seed", type=int, default=20250105)
    ap.add_argument("--redline_max_new_tokens", type=int, default=320)
    ap.add_argument("--redline_temp", type=float, default=0.7)
    ap.add_argument("--redline_top_p", type=float, default=0.9)
    ap.add_argument("--redline_constrained", action="store_true")
    ap.add_argument("--redline_autotune_duty", action="store_true", help="Enable duty autotune during redline eval (not open-loop).")
    ap.add_argument("--redline_no_autotune_duty", action="store_true", help="Deprecated; disables redline autotune if set.")
    ap.add_argument("--redline_sim_timeout_s", type=float, default=60.0)

    # (A) Self-play + EDA repair (Policy Improvement Operator)
    # Heavier defaults to make tol=0.01 feasible: generate more candidates + require multiple CV-pass hits per task.
    ap.add_argument("--selfplay_n_gen", type=int, default=32)
    ap.add_argument("--selfplay_max_rounds", type=int, default=4)
    ap.add_argument("--selfplay_ensure_pass_cv", action="store_true", default=True)
    ap.add_argument("--selfplay_min_pass_cv", type=int, default=3)
    ap.add_argument("--selfplay_temp_step", type=float, default=0.0)
    ap.add_argument("--selfplay_max_new_tokens", type=int, default=320)
    ap.add_argument("--selfplay_temperature", type=float, default=0.7)
    ap.add_argument("--selfplay_top_p", type=float, default=0.9)
    ap.add_argument("--selfplay_sim_timeout_s", type=float, default=60.0)
    ap.add_argument("--selfplay_sim_workers", type=int, default=22, help="Parallel ngspice workers for self-play (set to CPU cores).")
    ap.add_argument(
        "--selfplay_shards",
        type=int,
        default=1,
        help="Self-play sharding across GPUs. 0=auto(use all visible GPUs).",
    )
    ap.add_argument("--selfplay_autotune_duty", action="store_true")
    ap.add_argument("--selfplay_eda_repair", action="store_true", default=True)
    ap.add_argument("--selfplay_repair_factors", default="0.5,0.8,1.0,1.25,1.5,2.0")
    ap.add_argument("--selfplay_repair_max_evals", type=int, default=60)
    ap.add_argument("--selfplay_repair_max_iters", type=int, default=3)
    ap.add_argument("--selfplay_repair_always", action="store_true")
    ap.add_argument("--selfplay_pairs_per_task", type=int, default=64)
    ap.add_argument("--selfplay_min_pair_gap", type=float, default=0.1)
    ap.add_argument("--selfplay_top_k_chosen", type=int, default=8)
    ap.add_argument("--selfplay_no_structural_repair", action="store_true")
    ap.add_argument("--selfplay_no_bank_expand_min_elems", action="store_true")
    ap.add_argument("--selfplay_bank_expand_max_parts", type=int, default=32)
    ap.add_argument("--selfplay_sft_topn_per_task", type=int, default=8, help="Write top-N chosen samples per task into self-play SFT.")
    ap.add_argument(
        "--selfplay_sft_strict_topn_per_task",
        type=int,
        default=0,
        help="Write top-N pass_CV samples per task into strict self-play SFT (0=all pass_CV).",
    )
    ap.add_argument("--selfplay_sft_warmup_steps", type=int, default=400, help="0 disables strict self-play SFT warmup before PVPO/DPO.")
    ap.add_argument("--selfplay_sft_warmup_lr", type=float, default=1e-5)
    ap.add_argument("--selfplay_sft_warmup_bsz", type=int, default=1)
    ap.add_argument("--selfplay_sft_warmup_accum", type=int, default=16)

    # Optional: ML repair corrector (trained from search repairs; used as proposal to replace pure search).
    ap.add_argument("--repair_corrector_enable", action="store_true")
    ap.add_argument("--repair_corrector_init", default="", help="Optional existing corrector checkpoint (model.pt).")
    ap.add_argument("--repair_corrector_start_round", type=int, default=1, help="Use corrector in self-play from this round index.")
    ap.add_argument("--repair_corrector_topk", type=int, default=3)
    ap.add_argument("--repair_corrector_max_factor", type=float, default=2.0)
    ap.add_argument("--repair_corrector_min_examples", type=int, default=64)
    ap.add_argument("--repair_corrector_epochs", type=int, default=30)
    ap.add_argument("--repair_corrector_lr", type=float, default=3e-4)
    ap.add_argument("--repair_corrector_batch_size", type=int, default=128)
    ap.add_argument("--repair_corrector_hidden", type=int, default=128)
    ap.add_argument("--repair_corrector_device", default="cpu")

    # (C) PVPO/DPO (train longer; rely on redline gate to prevent regression)
    ap.add_argument("--dpo_lr", type=float, default=5e-6)
    ap.add_argument("--dpo_epochs", type=int, default=3)
    ap.add_argument("--dpo_bsz", type=int, default=1)
    ap.add_argument("--dpo_grad_accum", type=int, default=16)
    ap.add_argument("--dpo_max_steps", type=int, default=0, help="0 means use --dpo_epochs")
    ap.add_argument("--dpo_save_steps", type=int, default=100)
    ap.add_argument("--dpo_save_total_limit", type=int, default=3)
    ap.add_argument(
        "--train_gpus",
        type=int,
        default=1,
        help="GPUs for DPO/PPO/SFT training via accelerate. 0=auto(use all visible GPUs).",
    )
    ap.add_argument(
        "--ddp_backend",
        default="gloo",
        help="torch.distributed backend for accelerate-launched training. Use gloo if NCCL is unstable on this machine.",
    )

    # (C') Optional SFT regularization after DPO (recommended to prevent forgetting).
    ap.add_argument(
        "--pvpo_sft_reg_jsonl",
        default="",
        help="Optional JSONL {'text': prompt+INC} used for a short SFT regularization after DPO (keeps DSL/family memory).",
    )
    ap.add_argument("--pvpo_sft_reg_lr", type=float, default=1e-5)
    ap.add_argument("--pvpo_sft_reg_max_steps", type=int, default=400)
    ap.add_argument("--pvpo_sft_reg_bsz", type=int, default=1)
    ap.add_argument("--pvpo_sft_reg_accum", type=int, default=16)
    ap.add_argument("--pvpo_sft_reg_max_rows", type=int, default=0, help="0 uses all rows from --pvpo_sft_reg_jsonl")

    # (D) Safe-PPO (must keep CV non-regression)
    ap.add_argument("--ppo_steps", type=int, default=80)
    ap.add_argument("--ppo_batch_size", type=int, default=8)
    ap.add_argument("--ppo_group_size", type=int, default=4)
    ap.add_argument("--ppo_group_reward_mode", default="pareto_add")
    ap.add_argument("--ppo_group_rel_coef", type=float, default=0.5)
    ap.add_argument("--ppo_mbs", type=int, default=4)
    ap.add_argument("--ppo_accum", type=int, default=2)
    ap.add_argument("--ppo_lr", type=float, default=5e-6)
    ap.add_argument("--ppo_target_kl", type=float, default=0.03)
    ap.add_argument("--ppo_clip", type=float, default=0.2)
    ap.add_argument("--ppo_vf_coef", type=float, default=0.5)
    ap.add_argument("--ppo_ent_coef", type=float, default=0.01)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--ppo_max_new_tokens", type=int, default=320)
    ap.add_argument("--ppo_temperature", type=float, default=0.7)
    ap.add_argument("--ppo_top_p", type=float, default=0.9)
    ap.add_argument("--ppo_constrained", action="store_true")
    ap.add_argument("--ppo_batch_task_mode", default="balanced_families")
    ap.add_argument("--ppo_autotune_duty", action="store_true")
    ap.add_argument("--ppo_sim_timeout_s", type=float, default=180.0)
    ap.add_argument("--ppo_sim_workers", type=int, default=22, help="Parallel ngspice workers for PPO rollout/eval (set to CPU cores).")
    ap.add_argument("--ppo_save_steps", type=int, default=10)
    ap.add_argument(
        "--ppo_guard_schedule",
        choices=["fixed", "double_per_round", "add_per_round"],
        default="add_per_round",
        help="Guard size schedule for Safe-PPO. 'double_per_round' uses |G_r| = min(Gmax, G0*2^r).",
    )
    ap.add_argument("--ppo_guard_base_tasks", type=int, default=8, help="Base guard task count G0 for round 0.")
    ap.add_argument("--ppo_guard_add_step", type=int, default=8, help="For add_per_round: add this many guard tasks per round.")
    ap.add_argument(
        "--ppo_guard_base_total_samples",
        type=int,
        default=16,
        help="Base guard sampling budget B0 for round 0 (used as --cv_guard_total_samples; scales with rounds under doubling).",
    )
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    train_manifest = str(args.train_tasks_jsonl or "").strip()
    guard_manifest = str(args.guard_tasks_jsonl or "").strip()
    if bool(train_manifest) != bool(guard_manifest):
        raise SystemExit("--train_tasks_jsonl and --guard_tasks_jsonl must be provided together")
    strict_split = bool(train_manifest and guard_manifest)
    train_tasks = load_tasks_jsonl(train_manifest) if strict_split else []
    guard_tasks = load_tasks_jsonl(guard_manifest) if strict_split else []
    if strict_split:
        ensure_disjoint(train_tasks, guard_tasks)
        if len(train_tasks) != 69 or len(guard_tasks) != 8:
            raise SystemExit(
                f"strict revision protocol requires 69 train and 8 guard tasks; got {len(train_tasks)} and {len(guard_tasks)}"
            )

    # Hard requirement: fixed tol=0.01 everywhere (self-play scoring, DPO filtering, PPO reward, eval).
    tol = 0.01
    if abs(float(args.tol) - tol) > 1e-12:
        print(f"[warn] forcing --tol {tol} (got {args.tol})")

    code_dir = Path(__file__).resolve().parent
    if str(args.outdir or "").strip():
        outdir = Path(str(args.outdir).strip()).resolve()
        if outdir.exists() and (not bool(args.resume)):
            raise SystemExit(f"--outdir exists but --resume not set: {outdir}")
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "logs").mkdir(parents=True, exist_ok=True)
        log_path = outdir / "logs" / "pipeline.log"
        _append_text(log_path, f"[resume] ts={_now()} pid={os.getpid()} outdir={outdir}\n")
        _write_json(outdir / f"run_config_resume_{_now()}.json", {"argv": sys.argv, "tol_forced": tol})
    else:
        run_root = Path(args.run_root)
        run_name = f"vpspi_tol001_{_now()}"
        if str(args.name or "").strip():
            run_name += f"_{str(args.name).strip()}"
        outdir = run_root / run_name
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "logs").mkdir(parents=True, exist_ok=True)
        log_path = outdir / "logs" / "pipeline.log"
        _append_text(log_path, f"[start] {run_name} ts={_now()} pid={os.getpid()}\n")
        _write_json(outdir / "run_config.json", {"argv": sys.argv, "tol_forced": tol})

    _write_json(
        outdir / "task_protocol.json",
        {
            "strict_split": bool(strict_split),
            "train_tasks_jsonl": str(Path(train_manifest).resolve()) if train_manifest else "",
            "train_tasks_sha256": sha256_file(train_manifest) if train_manifest else "",
            "guard_tasks_jsonl": str(Path(guard_manifest).resolve()) if guard_manifest else "",
            "guard_tasks_sha256": sha256_file(guard_manifest) if guard_manifest else "",
            "n_train_tasks": int(len(train_tasks)) if strict_split else 77,
            "n_guard_tasks": int(len(guard_tasks)) if strict_split else int(len(REDLINE_TASKS)),
            "seed": int(args.seed),
            "rounds": int(args.rounds),
        },
    )

    # (Resume-safe) load baseline non-regression guard if available; otherwise run anchor redline.
    best_guard_path = outdir / "best_cv_guard.json"
    anchor_summ = outdir / "anchor_redline" / "eval_summary.json"
    if bool(args.resume) and best_guard_path.exists() and anchor_summ.exists():
        try:
            best_eval = json.loads(best_guard_path.read_text(encoding="utf-8"))
            _append_text(log_path, f"[resume] reuse best_cv_guard.json cv_rate={best_eval.get('cv_rate')} cv_min_family={best_eval.get('cv_min_family')}\n")
        except Exception:
            best_eval = {}
    else:
        anchor_eval = _run_redline_eval(
            code_dir=code_dir,
            base_model=args.base_model,
            adapter=args.anchor_adapter,
            outdir=outdir / "anchor_redline",
            tol=tol,
            sim_timeout_s=float(args.redline_sim_timeout_s),
            min_elems=int(args.min_elems),
            bank_expand_min_elems=True,
            bank_expand_max_parts=32,
            n_per_task=int(args.redline_n_per_task),
            seed=int(args.redline_seed),
            max_new_tokens=int(args.redline_max_new_tokens),
            temperature=float(args.redline_temp),
            top_p=float(args.redline_top_p),
            constrained=bool(args.redline_constrained),
            autotune_duty=bool(args.redline_autotune_duty) and (not bool(args.redline_no_autotune_duty)),
            log_path=log_path,
            tasks=guard_tasks if strict_split else None,
        )
        best_eval = {
            "cv_rate": float(anchor_eval["cv_rate"]),
            "cv_min_family": float(anchor_eval["cv_min_family"]),
            "cv_by_family": dict(anchor_eval["cv_by_family"]),
        }
        _write_json(outdir / "best_cv_guard.json", best_eval)

    cur_adapter = str(args.anchor_adapter)
    if bool(args.resume):
        fin = outdir / "final_state.json"
        if fin.exists():
            try:
                st = json.loads(fin.read_text(encoding="utf-8"))
                cur2 = str(st.get("cur_adapter") or "").strip()
                if cur2:
                    cur_adapter = cur2
                    _append_text(log_path, f"[resume] cur_adapter <- final_state.json ({cur_adapter})\n")
            except Exception:
                pass

    repair_corrector_path = str(args.repair_corrector_init or "").strip()
    if repair_corrector_path:
        if not Path(repair_corrector_path).exists():
            print(f"[warn] repair_corrector_init not found: {repair_corrector_path}")
            repair_corrector_path = ""

    # Pick the next round index if resuming from an already-completed run (round_state.json exists).
    start_ridx = 0
    if bool(args.resume):
        try:
            states = sorted(outdir.glob("round_*/round_state.json"))
            done_rounds: List[int] = []
            for p in states:
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    done_rounds.append(int(obj.get("round")))
                except Exception:
                    continue
            if done_rounds:
                start_ridx = int(max(done_rounds) + 1)
        except Exception:
            pass

    rounds = int(args.rounds)
    ridx = int(start_ridx)
    while True:
        if rounds > 0 and (ridx - start_ridx) >= rounds:
            break
        rdir = outdir / f"round_{ridx:02d}"
        rdir.mkdir(parents=True, exist_ok=True)

        # Snapshot the current best adapter before starting this round (protects against accidental overwrite).
        try:
            snap_src = Path(str(cur_adapter)).resolve()
            snap_dst = rdir / "adapter_start_snapshot"
            _snapshot_dir(snap_src, snap_dst)
            _write_json(rdir / "adapter_start_snapshot.json", {"src": str(snap_src), "dst": str(snap_dst)})
        except Exception as e:
            _append_text(log_path, f"[warn] snapshot adapter failed: {type(e).__name__}: {e}\n")

        # (A) self-play data build with industrial-EDA repair
        sp_dir = rdir / "selfplay_data"
        sp_dir.mkdir(parents=True, exist_ok=True)
        base_sp_cmd = [
            sys.executable,
            str(code_dir / "build_selfplay_inc_datasets.py"),
            "--base_model",
            str(args.base_model),
            "--adapter",
            str(cur_adapter),
            "--seed",
            str(int(args.seed) + ridx * 100),
            "--n_gen",
            str(int(args.selfplay_n_gen)),
            "--max_rounds",
            str(int(args.selfplay_max_rounds)),
            "--temp_step",
            str(float(args.selfplay_temp_step)),
            "--max_new_tokens",
            str(int(args.selfplay_max_new_tokens)),
            "--temperature",
            str(float(args.selfplay_temperature)),
            "--top_p",
            str(float(args.selfplay_top_p)),
            "--min_elems",
            str(int(args.min_elems)),
            "--pairs_per_task",
            str(int(args.selfplay_pairs_per_task)),
            "--min_pair_gap",
            str(float(args.selfplay_min_pair_gap)),
            "--top_k_chosen",
            str(int(args.selfplay_top_k_chosen)),
            "--tol",
            str(float(tol)),
            "--t_pre",
            "0.008",
            "--t_win",
            "0.002",
            "--sim_timeout_s",
            str(float(args.selfplay_sim_timeout_s)),
            "--sft_topn_per_task",
            str(int(args.selfplay_sft_topn_per_task)),
            "--sft_strict_topn_per_task",
            str(int(args.selfplay_sft_strict_topn_per_task)),
        ]
        if strict_split:
            base_sp_cmd += ["--tasks_jsonl", str(Path(train_manifest).resolve())]
        if bool(args.selfplay_ensure_pass_cv):
            base_sp_cmd.append("--ensure_pass_cv")
            base_sp_cmd += ["--min_pass_cv", str(int(args.selfplay_min_pass_cv))]
        if not bool(args.selfplay_no_structural_repair):
            base_sp_cmd.append("--structural_repair")
        if not bool(args.selfplay_no_bank_expand_min_elems):
            base_sp_cmd.append("--bank_expand_min_elems")
            base_sp_cmd += ["--bank_expand_max_parts", str(int(args.selfplay_bank_expand_max_parts))]
        if bool(args.selfplay_autotune_duty):
            base_sp_cmd.append("--autotune_duty")
        if bool(args.selfplay_eda_repair):
            base_sp_cmd.append("--eda_repair")
            base_sp_cmd += ["--repair_factors", str(args.selfplay_repair_factors)]
            base_sp_cmd += ["--repair_max_evals", str(int(args.selfplay_repair_max_evals))]
            base_sp_cmd += ["--repair_max_iters", str(int(args.selfplay_repair_max_iters))]
            if bool(args.selfplay_repair_always):
                base_sp_cmd.append("--repair_always")
        if bool(args.repair_corrector_enable) and repair_corrector_path and int(ridx) >= int(args.repair_corrector_start_round):
            base_sp_cmd += ["--repair_corrector", str(repair_corrector_path)]
            base_sp_cmd += ["--repair_corrector_topk", str(int(args.repair_corrector_topk))]
            base_sp_cmd += ["--repair_corrector_max_factor", str(float(args.repair_corrector_max_factor))]

        n_shards = int(args.selfplay_shards)
        if n_shards <= 0:
            n_shards = _gpu_count() or 1
        if int(n_shards) <= 1:
            sp_cmd = list(base_sp_cmd) + ["--out_root", str(sp_dir), "--sim_workers", str(int(args.selfplay_sim_workers))]
            # Resume-safe dataset build: append to existing selfplay_data if present.
            if bool(args.resume) and (sp_dir / "done_tasks.jsonl").exists():
                sp_cmd.append("--resume")
            _run(sp_cmd, cwd=code_dir, log_path=log_path)
        else:
            _run_selfplay_sharded(
                code_dir=code_dir,
                base_sp_cmd=base_sp_cmd,
                sp_dir=sp_dir,
                ridx=int(ridx),
                n_shards=int(n_shards),
                sim_workers_total=int(args.selfplay_sim_workers),
                log_path=log_path,
            )

        # Train/refresh repair corrector from this round's repaired samples.
        if bool(args.repair_corrector_enable):
            corr_dir = rdir / "repair_corrector"
            corr_dir.mkdir(parents=True, exist_ok=True)
            corr_cmd = [
                sys.executable,
                str(code_dir / "train_repair_corrector.py"),
                "--selfplay_root",
                str(sp_dir),
                "--outdir",
                str(corr_dir),
                "--seed",
                str(int(args.seed) + ridx * 100 + 7),
                "--min_examples",
                str(int(args.repair_corrector_min_examples)),
                "--epochs",
                str(int(args.repair_corrector_epochs)),
                "--lr",
                str(float(args.repair_corrector_lr)),
                "--batch_size",
                str(int(args.repair_corrector_batch_size)),
                "--hidden",
                str(int(args.repair_corrector_hidden)),
                "--max_factor",
                str(float(args.repair_corrector_max_factor)),
                "--device",
                str(args.repair_corrector_device),
            ]
            _run(corr_cmd, cwd=code_dir, log_path=log_path)
            cand = corr_dir / "model.pt"
            if cand.exists():
                repair_corrector_path = str(cand)
                _write_json(rdir / "repair_corrector_state.json", {"path": repair_corrector_path})

        # Optional strict self-play SFT warmup (only pass_CV samples) before PVPO/DPO.
        if int(getattr(args, "selfplay_sft_warmup_steps", 0) or 0) > 0:
            strict_path = sp_dir / "sft_train_strict.jsonl"
            if strict_path.exists() and int(strict_path.stat().st_size) > 0:
                warm_dir = rdir / "selfplay_sft_warmup"
                warm_dir.mkdir(parents=True, exist_ok=True)
                warm_cmd = [
                    sys.executable,
                    str(code_dir / "train_sft_dcdc.py"),
                    "--base_model",
                    str(args.base_model),
                    "--init_adapter",
                    str(cur_adapter),
                    "--train_jsonl",
                    str(strict_path),
                    "--outdir",
                    str(warm_dir),
                    "--ddp_backend",
                    str(args.ddp_backend),
                    "--seed",
                    str(int(args.seed) + ridx * 1000 + 11),
                    "--lr",
                    str(float(args.selfplay_sft_warmup_lr)),
                    "--max_steps",
                    str(int(args.selfplay_sft_warmup_steps)),
                    "--bsz",
                    str(int(args.selfplay_sft_warmup_bsz)),
                    "--grad_accum",
                    str(int(args.selfplay_sft_warmup_accum)),
                    "--save_steps",
                    "200",
                    "--save_total_limit",
                    "2",
                ]
                tg = int(args.train_gpus)
                if tg <= 0:
                    tg = _gpu_count() or 1
                tg = int(max(1, min(int(tg), int(_gpu_count() or tg))))
                train_env = None
                if int(tg) <= 1:
                    train_env = _train_env_single_gpu()
                _run(_maybe_accelerate(warm_cmd, nproc=int(tg)), cwd=code_dir, log_path=log_path, env=train_env)
                warm_adapter = warm_dir / "sft_final"
                if warm_adapter.exists():
                    warm_eval = _run_redline_eval(
                        code_dir=code_dir,
                        base_model=args.base_model,
                        adapter=str(warm_adapter),
                        outdir=rdir / "selfplay_sft_warmup_redline",
                        tol=tol,
                        sim_timeout_s=float(args.redline_sim_timeout_s),
                        min_elems=int(args.min_elems),
                        bank_expand_min_elems=True,
                        bank_expand_max_parts=32,
                        n_per_task=int(args.redline_n_per_task),
                        seed=int(args.redline_seed),
                        max_new_tokens=int(args.redline_max_new_tokens),
                        temperature=float(args.redline_temp),
                        top_p=float(args.redline_top_p),
                        constrained=bool(args.redline_constrained),
                        autotune_duty=bool(args.redline_autotune_duty) and (not bool(args.redline_no_autotune_duty)),
                        log_path=log_path,
                        tasks=guard_tasks if strict_split else None,
                    )
                    ok_warm, degraded = _non_regress(cand=warm_eval, best=best_eval)
                    _write_json(rdir / "selfplay_sft_warmup_gate.json", {"accepted": bool(ok_warm), "degraded": degraded, "cand": warm_eval, "best": best_eval})
                    if ok_warm:
                        cur_adapter = str(warm_adapter)
                        best_eval = _update_best(best_eval, warm_eval)
                        _write_json(outdir / "best_cv_guard.json", best_eval)

        pairs_path = sp_dir / "dpo_pairs.jsonl"
        if not pairs_path.exists():
            raise SystemExit(f"Missing PVPO pairs: {pairs_path}")

        # (C) PVPO/DPO on high-confidence preference pairs
        dpo_dir = rdir / "pvpo_dpo"
        dpo_dir.mkdir(parents=True, exist_ok=True)
        dpo_cmd = [
            sys.executable,
            str(code_dir / "train_dpo_dcdc.py"),
            "--base_model",
            str(args.base_model),
            "--sft_adapter",
            str(cur_adapter),
            "--dpo_pairs",
            str(pairs_path),
            "--outdir",
            str(dpo_dir),
            "--ddp_backend",
            str(args.ddp_backend),
            "--seed",
            str(int(args.seed) + ridx * 1000 + 21),
            "--lr",
            str(float(args.dpo_lr)),
            "--epochs",
            str(int(args.dpo_epochs)),
            "--bsz",
            str(int(args.dpo_bsz)),
            "--grad_accum",
            str(int(args.dpo_grad_accum)),
            "--max_steps",
            str(int(args.dpo_max_steps)),
            "--save_steps",
            str(int(args.dpo_save_steps)),
            "--save_total_limit",
            str(int(args.dpo_save_total_limit)),
        ]
        tg = int(args.train_gpus)
        if tg <= 0:
            tg = _gpu_count() or 1
        tg = int(max(1, min(int(tg), int(_gpu_count() or tg))))
        train_env = None
        if int(tg) <= 1:
            train_env = _train_env_single_gpu()
        _run(_maybe_accelerate(dpo_cmd, nproc=int(tg)), cwd=code_dir, log_path=log_path, env=train_env)

        dpo_adapter = dpo_dir / "dpo_final"
        if not dpo_adapter.exists():
            raise SystemExit(f"Missing PVPO adapter: {dpo_adapter}")

        pvpo_adapter = dpo_adapter
        if str(args.pvpo_sft_reg_jsonl or "").strip():
            reg_dir = rdir / "pvpo_sft_reg"
            reg_dir.mkdir(parents=True, exist_ok=True)
            reg_cmd = [
                sys.executable,
                str(code_dir / "train_sft_dcdc.py"),
                "--base_model",
                str(args.base_model),
                "--init_adapter",
                str(dpo_adapter),
                "--train_jsonl",
                str(args.pvpo_sft_reg_jsonl),
                "--outdir",
                str(reg_dir),
                "--ddp_backend",
                str(args.ddp_backend),
                "--seed",
                str(int(args.seed) + ridx * 1000 + 31),
                "--lr",
                str(float(args.pvpo_sft_reg_lr)),
                "--max_steps",
                str(int(args.pvpo_sft_reg_max_steps)),
                "--bsz",
                str(int(args.pvpo_sft_reg_bsz)),
                "--grad_accum",
                str(int(args.pvpo_sft_reg_accum)),
                "--max_rows",
                str(int(args.pvpo_sft_reg_max_rows)),
                "--save_steps",
                "200",
                "--save_total_limit",
                "2",
            ]
            tg = int(args.train_gpus)
            if tg <= 0:
                tg = _gpu_count() or 1
            tg = int(max(1, min(int(tg), int(_gpu_count() or tg))))
            train_env = None
            if int(tg) <= 1:
                train_env = _train_env_single_gpu()
            _run(_maybe_accelerate(reg_cmd, nproc=int(tg)), cwd=code_dir, log_path=log_path, env=train_env)
            reg_adapter = reg_dir / "sft_final"
            if not reg_adapter.exists():
                raise SystemExit(f"Missing PVPO SFT-reg adapter: {reg_adapter}")
            pvpo_adapter = reg_adapter

        dpo_eval = _run_redline_eval(
            code_dir=code_dir,
            base_model=args.base_model,
            adapter=str(pvpo_adapter),
            outdir=rdir / "pvpo_redline",
            tol=tol,
            sim_timeout_s=float(args.redline_sim_timeout_s),
            min_elems=int(args.min_elems),
            bank_expand_min_elems=True,
            bank_expand_max_parts=32,
            n_per_task=int(args.redline_n_per_task),
            seed=int(args.redline_seed),
            max_new_tokens=int(args.redline_max_new_tokens),
            temperature=float(args.redline_temp),
            top_p=float(args.redline_top_p),
            constrained=bool(args.redline_constrained),
            autotune_duty=bool(args.redline_autotune_duty) and (not bool(args.redline_no_autotune_duty)),
            log_path=log_path,
            tasks=guard_tasks if strict_split else None,
        )
        ok_dpo, degraded = _non_regress(cand=dpo_eval, best=best_eval)
        _write_json(rdir / "pvpo_gate.json", {"accepted": bool(ok_dpo), "degraded": degraded, "cand": dpo_eval, "best": best_eval})
        if ok_dpo:
            cur_after_pvpo = str(pvpo_adapter)
            best_eval = _update_best(best_eval, dpo_eval)
            _write_json(outdir / "best_cv_guard.json", best_eval)
        else:
            cur_after_pvpo = str(cur_adapter)

        # (D) Safe-PPO (with internal per-family non-regression + rollback)
        ppo_dir = rdir / "safe_ppo"
        ppo_dir.mkdir(parents=True, exist_ok=True)
        # Round-wise guard schedule for Safe-PPO.
        guard_tasks0 = int(args.ppo_guard_base_tasks)
        guard_budget0 = int(args.ppo_guard_base_total_samples)
        sched = str(args.ppo_guard_schedule)
        if strict_split:
            guard_task_count = int(len(guard_tasks))
        elif sched == "double_per_round":
            guard_task_count = int(guard_tasks0) * (2 ** int(ridx))
        elif sched == "add_per_round":
            guard_task_count = int(guard_tasks0) + int(args.ppo_guard_add_step) * int(ridx)
        else:
            guard_task_count = int(guard_tasks0)
        # Keep total guard samples roughly proportional to guard_tasks (preserve n_per_task).
        if strict_split:
            guard_budget = int(guard_budget0)
        elif int(guard_tasks0) > 0:
            guard_budget = int((int(guard_budget0) * int(guard_task_count) + int(guard_tasks0) - 1) // int(guard_tasks0))
        else:
            guard_budget = int(guard_budget0)
        
        ppo_cmd = [
            sys.executable,
            str(code_dir / "train_ppo_dcdc.py"),
            "--base_model",
            str(args.base_model),
            "--sft_adapter",
            str(cur_after_pvpo),
            "--outdir",
            str(ppo_dir),
            "--ddp_backend",
            str(args.ddp_backend),
            "--steps",
            str(int(args.ppo_steps)),
            "--seed",
            str(int(args.seed) + ridx * 1000 + 7),
            "--batch_size",
            str(int(args.ppo_batch_size)),
            "--group_size",
            str(int(args.ppo_group_size)),
            "--group_reward_mode",
            str(args.ppo_group_reward_mode),
            "--group_rel_coef",
            str(float(args.ppo_group_rel_coef)),
            "--mini_batch_size",
            str(int(args.ppo_mbs)),
            "--grad_accum",
            str(int(args.ppo_accum)),
            "--lr",
            str(float(args.ppo_lr)),
            "--target_kl",
            str(float(args.ppo_target_kl)),
            "--cliprange",
            str(float(args.ppo_clip)),
            "--cliprange_value",
            str(float(args.ppo_clip)),
            "--vf_coef",
            str(float(args.ppo_vf_coef)),
            "--ent_coef",
            str(float(args.ppo_ent_coef)),
            "--ppo_epochs",
            str(int(args.ppo_epochs)),
            "--max_new_tokens",
            str(int(args.ppo_max_new_tokens)),
            "--temperature",
            str(float(args.ppo_temperature)),
            "--top_p",
            str(float(args.ppo_top_p)),
            "--batch_task_mode",
            str(args.ppo_batch_task_mode),
            "--min_elems",
            str(int(args.min_elems)),
            "--sim_timeout_s",
            str(float(args.ppo_sim_timeout_s)),
            "--sim_workers",
            str(int(args.ppo_sim_workers)),
            "--save_steps",
            str(int(args.ppo_save_steps)),
            "--cv_guard_init_tasks",
            str(int(guard_task_count)),
            "--cv_guard_tasks_max",
            str(int(guard_task_count)),
            "--cv_guard_total_samples",
            str(int(guard_budget)),
            "--cv_guard_expand_streak",
            "0",
        ]
        if strict_split:
            ppo_cmd += ["--train_tasks_jsonl", str(Path(train_manifest).resolve())]
            ppo_cmd += ["--guard_tasks_jsonl", str(Path(guard_manifest).resolve())]
            ppo_cmd += ["--hard_task_frac", "0"]
        if bool(args.ppo_constrained):
            ppo_cmd.append("--constrained")
        if bool(args.ppo_autotune_duty):
            ppo_cmd.append("--autotune_duty")

        tg = int(args.train_gpus)
        if tg <= 0:
            tg = _gpu_count() or 1
        tg = int(max(1, min(int(tg), int(_gpu_count() or tg))))

        # When launching PPO with multiple processes, each rank runs its own ngspice pool.
        # Treat --ppo_sim_workers as a total CPU budget and split across ranks.
        ppo_cpu_budget = int(args.ppo_sim_workers)
        if ppo_cpu_budget <= 0:
            ppo_cpu_budget = int(_cpu_quota())
        ppo_workers_per_rank = int(max(1, int(ppo_cpu_budget) // int(tg))) if tg > 1 else int(max(1, ppo_cpu_budget))
        for i, x in enumerate(list(ppo_cmd)):
            if x == "--sim_workers" and i + 1 < len(ppo_cmd):
                ppo_cmd[i + 1] = str(int(ppo_workers_per_rank))

        _append_text(
            log_path,
            f"[ppo] train_gpus={tg} ppo_sim_workers_total={ppo_cpu_budget} ppo_sim_workers_per_rank={ppo_workers_per_rank}\n",
        )
        train_env = None
        if int(tg) <= 1:
            train_env = _train_env_single_gpu()
        _run(_maybe_accelerate(ppo_cmd, nproc=int(tg)), cwd=code_dir, log_path=log_path, env=train_env)

        ppo_best = ppo_dir / "ppo_best"
        if not ppo_best.exists():
            raise SystemExit(f"Missing ppo_best: {ppo_best}")

        ppo_eval = _run_redline_eval(
            code_dir=code_dir,
            base_model=args.base_model,
            adapter=str(ppo_best),
            outdir=rdir / "ppo_redline",
            tol=tol,
            sim_timeout_s=float(args.redline_sim_timeout_s),
            min_elems=int(args.min_elems),
            bank_expand_min_elems=True,
            bank_expand_max_parts=32,
            n_per_task=int(args.redline_n_per_task),
            seed=int(args.redline_seed),
            max_new_tokens=int(args.redline_max_new_tokens),
            temperature=float(args.redline_temp),
            top_p=float(args.redline_top_p),
            constrained=bool(args.redline_constrained),
            autotune_duty=bool(args.redline_autotune_duty) and (not bool(args.redline_no_autotune_duty)),
            log_path=log_path,
            tasks=guard_tasks if strict_split else None,
        )
        ok_ppo, degraded = _non_regress(cand=ppo_eval, best=best_eval)
        _write_json(rdir / "ppo_gate.json", {"accepted": bool(ok_ppo), "degraded": degraded, "cand": ppo_eval, "best": best_eval})
        if ok_ppo:
            cur_adapter = str(ppo_best)
            best_eval = _update_best(best_eval, ppo_eval)
            _write_json(outdir / "best_cv_guard.json", best_eval)
        else:
            cur_adapter = str(cur_after_pvpo)

        accepted_snapshot = rdir / "accepted_adapter"
        _snapshot_dir(Path(cur_adapter).resolve(), accepted_snapshot)
        _write_json(
            rdir / "stage_checkpoints.json",
            {
                "round": int(ridx),
                "round_start": str((rdir / "adapter_start_snapshot").resolve()),
                "pvpo_candidate": str(Path(pvpo_adapter).resolve()),
                "pvpo_accepted": bool(ok_dpo),
                "safe_ppo_pre": str(Path(cur_after_pvpo).resolve()),
                "safe_ppo_post_candidate": str(ppo_best.resolve()),
                "safe_ppo_accepted": bool(ok_ppo),
                "accepted_adapter": str(accepted_snapshot.resolve()),
            },
        )
        cur_adapter = str(accepted_snapshot.resolve())
        _write_json(rdir / "round_state.json", {"round": int(ridx), "cur_adapter": str(cur_adapter), "best_eval": best_eval})

        ridx += 1

    _write_json(outdir / "final_state.json", {"cur_adapter": str(cur_adapter), "best_eval": best_eval})
    _append_text(log_path, f"[done] ts={_now()} cur_adapter={cur_adapter}\n")
    print("[OK] VP-SPI finished:", str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

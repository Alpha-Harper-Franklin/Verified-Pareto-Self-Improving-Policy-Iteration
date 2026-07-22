#!/usr/bin/env python3
from __future__ import annotations

import inspect


def _patch_ddp_init_sync_default_false() -> None:
    """
    Work around rare NCCL+DDP init-time broadcast failures on some systems.

    We load identical weights on every rank, so the init-time sync broadcast is redundant.
    """

    try:
        import torch
        from torch.nn.parallel import DistributedDataParallel as DDP
    except Exception:
        return

    try:
        sig = inspect.signature(DDP)
    except Exception:
        return

    if "init_sync" not in sig.parameters:
        return

    if getattr(DDP, "_vpspi_init_sync_patched", False):
        return

    orig_init = DDP.__init__

    def _init(self, module, *args, **kwargs):
        kwargs.setdefault("init_sync", False)
        return orig_init(self, module, *args, **kwargs)

    DDP.__init__ = _init  # type: ignore[assignment]
    setattr(DDP, "_vpspi_init_sync_patched", True)


_patch_ddp_init_sync_default_false()

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import signal
import shlex
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate.utils import InitProcessGroupKwargs, DistributedDataParallelKwargs
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from dcdc_eval_tran import eval_one_detail_dcdc
from dcdc_taskset import Task, default_taskset
from dcdc_verifier import verify_inc_dcdc
from inc_parser import extract_inc_lines


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _dist_rank_world() -> Tuple[int, int]:
    rank = 0
    world = 1
    try:
        if str(os.environ.get("RANK") or "").strip():
            rank = int(os.environ.get("RANK") or 0)
        elif str(os.environ.get("LOCAL_RANK") or "").strip():
            rank = int(os.environ.get("LOCAL_RANK") or 0)
    except Exception:
        rank = 0
    try:
        if str(os.environ.get("WORLD_SIZE") or "").strip():
            world = int(os.environ.get("WORLD_SIZE") or 1)
    except Exception:
        world = 1
    return int(rank), int(world)


def _is_main_process() -> bool:
    rank, world = _dist_rank_world()
    return int(world) <= 1 or int(rank) == 0


def _patch_torch_numpy_fancy_indexing() -> None:
    """
    Workaround for a PyTorch regression (seen in torch==2.7.0) where indexing a
    torch.Tensor with a numpy int array raises:

        RuntimeError: Could not infer dtype of numpy.int64

    TRL's PPOTrainer uses numpy permutations for minibatch indices, so we patch
    numpy.random.permutation to return a plain Python list of ints.
    """

    try:
        import numpy as np
    except Exception:
        return

    if getattr(np.random, "_vpspi_perm_returns_list", False):
        return

    old_perm = np.random.permutation

    def _perm(n):
        arr = old_perm(n)
        try:
            return [int(x) for x in arr.tolist()]
        except Exception:
            try:
                return [int(x) for x in list(arr)]
            except Exception:
                return arr

    np.random.permutation = _perm  # type: ignore[assignment]
    setattr(np.random, "_vpspi_perm_returns_list", True)
    print("[patch] numpy permutation -> list (torch numpy-index workaround)", flush=True)


def _patch_trl_adaptive_kl_controller_clip() -> None:
    """Avoid `np.clip` to prevent numpy internal `_clip` ABI issues.

    TRL's `AdaptiveKLController.update()` only needs scalar clipping, so a pure
    Python implementation is equivalent and more robust.
    """

    try:
        from trl.trainer.utils import AdaptiveKLController
    except Exception:
        return

    if getattr(AdaptiveKLController, "_vpspi_clip_patched", False):
        return

    def _clip01(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    old_update = AdaptiveKLController.update

    def update(self, current, n_steps):
        try:
            target = float(self.target)
            cur = float(current)
        except Exception:
            target = self.target
            cur = current
        if target == 0:
            return old_update(self, current, n_steps)
        proportional_error = cur / target - 1
        proportional_error = _clip01(proportional_error, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

    AdaptiveKLController.update = update
    setattr(AdaptiveKLController, "_vpspi_clip_patched", True)
    print("[patch] trl AdaptiveKLController.update: python clip fallback", flush=True)


def _parse_cpu_list(s: str) -> int:
    s = str(s or "").strip()
    if not s:
        return 0
    total = 0
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                lo = int(a.strip())
                hi = int(b.strip())
            except Exception:
                continue
            if hi >= lo:
                total += int(hi - lo + 1)
        else:
            try:
                int(part)
            except Exception:
                continue
            total += 1
    return int(total)


def _cgroup_cpu_quota() -> int:
    """
    Return the CPU quota as an integer #CPUs (ceil), or 0 if unlimited/unknown.
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
        q = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        per = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q.exists() and per.exists():
            quota_us = float(q.read_text().strip())
            period_us = float(per.read_text().strip())
            if quota_us > 0 and period_us > 0:
                return int(max(1, math.ceil(quota_us / period_us)))
    except Exception:
        pass
    return 0


def _cgroup_cpuset_cpus() -> int:
    for cand in [
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
        "/sys/fs/cgroup/cpuset/cpuset.effective_cpus",
    ]:
        try:
            p = Path(cand)
            if p.exists():
                n = _parse_cpu_list(p.read_text().strip())
                if n > 0:
                    return int(n)
        except Exception:
            continue
    return 0


def _effective_cpus() -> int:
    cpu = int(os.cpu_count() or 1)
    quota = int(_cgroup_cpu_quota() or 0)
    if quota > 0:
        cpu = min(cpu, quota)
    cpuset = int(_cgroup_cpuset_cpus() or 0)
    if cpuset > 0:
        cpu = min(cpu, cpuset)
    return max(1, int(cpu))


def _auto_sim_workers(n: int) -> int:
    try:
        n = int(n)
    except Exception:
        n = 0
    if n > 0:
        return int(n)
    return int(_effective_cpus() or 1)


def _auto_sim_workers_dist(n: int) -> int:
    """Auto-allocate sim workers per distributed process.

    Semantics:
    - If user sets --sim_workers > 0: treat it as *per-process* (no division).
    - If --sim_workers <= 0: treat it as total CPU budget and divide by WORLD_SIZE.
    """

    try:
        n = int(n)
    except Exception:
        n = 0
    if n > 0:
        return int(n)
    total = int(_auto_sim_workers(0) or 1)
    _rank, world = _dist_rank_world()
    world = int(max(1, world))
    return int(max(1, total // world))


_STOP_REQUESTED = False
_FORCE_SAVE_REQUESTED = False


def _handle_stop_signal(signum: int, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[signal] received {int(signum)}; will stop after this step", flush=True)


def _handle_save_signal(signum: int, _frame) -> None:
    global _FORCE_SAVE_REQUESTED
    _FORCE_SAVE_REQUESTED = True
    print(f"[signal] received {int(signum)}; will save checkpoint after this step", flush=True)


def _install_signal_handlers() -> None:
    try:
        signal.signal(signal.SIGTERM, _handle_stop_signal)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGINT, _handle_stop_signal)
    except Exception:
        pass
    if hasattr(signal, "SIGUSR1"):
        try:
            signal.signal(signal.SIGUSR1, _handle_save_signal)
        except Exception:
            pass


def _force_inputs_require_grads(model: Any) -> None:
    try:
        model.enable_input_require_grads()
        return
    except Exception:
        pass
    try:
        emb = model.get_input_embeddings()
        if emb is None:
            return

        def _hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)

        emb.register_forward_hook(_hook)
    except Exception:
        pass


def _sanity_check_trainable_grads(model: Any, tok: Any) -> None:
    try:
        trainable = [(n, p) for n, p in model.named_parameters() if getattr(p, "requires_grad", False)]
        n_trainable = sum(int(p.numel()) for _, p in trainable)
        print(f"[sanity] trainable_params={n_trainable}", flush=True)
        if not trainable:
            raise RuntimeError("No trainable parameters found (expected LoRA + value head).")

        device = next(model.parameters()).device
        eos = int(getattr(tok, "eos_token_id", 0) or 0)
        ids = torch.tensor([[eos] * 32], device=device, dtype=torch.long)

        model.train()
        # Prefer checking grads on the actual policy model (pretrained_model holds LoRA).
        try:
            out = model.pretrained_model(input_ids=ids, labels=ids)
        except Exception:
            out = model(input_ids=ids, labels=ids)
        loss = getattr(out, "loss", None)
        if loss is None:
            logits = getattr(out, "logits", None)
            if logits is None and isinstance(out, (tuple, list)) and len(out) > 0:
                logits = out[0]
            if logits is None:
                raise RuntimeError("Sanity forward produced no loss/logits.")
            loss = logits.float().mean()
        loss.backward()

        has_grad = any((p.grad is not None) for _, p in trainable)
        model.zero_grad(set_to_none=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[sanity] has_grad={has_grad}", flush=True)
        if not has_grad:
            raise RuntimeError("No gradients on trainable params. Check gradient checkpointing / input require grads.")
    except Exception as e:
        raise SystemExit(f"[sanity_failed] {type(e).__name__}: {e}")


def _write_text(path: Path, text: str) -> None:
    if not _is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")




def _unwrap_trainer_model(trainer: Any):
    m = getattr(trainer, "model", None)
    if m is None:
        return None
    try:
        acc = getattr(trainer, "accelerator", None)
        if acc is not None:
            return acc.unwrap_model(m)
    except Exception:
        pass
    try:
        return m.module
    except Exception:
        return m


def _write_json(path: Path, obj: Any) -> None:
    if not _is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _strip_response_template(text: str) -> str:
    key = "### Response:"
    if key not in text:
        return text
    return text.rsplit(key, 1)[-1].lstrip()


def _sanity_generate_report(
    *,
    trainer: PPOTrainer,
    tok: Any,
    build_prompt,
    logits_proc: Optional[list],
    outdir: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    # If PPO starts from the wrong weights (e.g., wrong adapter), rollouts collapse to "INC111..."
    # and the whole run becomes wasted. Detect early and save a repro artifact.
    dev = next(trainer.model.parameters()).device
    tasks = [
        ("buck", 12.0, 5.0),
        ("buck", 15.0, 5.0),
    ]
    items: List[Dict[str, Any]] = []

    for fam, vin, vout in tasks:
        prompt = str(build_prompt(fam, float(vin), float(vout)))

        # Direct generate (full sequence, then strip the response template).
        enc = tok(prompt, return_tensors="pt").to(dev)
        gen_model = trainer.model
        try:
            gen_model = trainer.accelerator.unwrap_model(trainer.model)
        except Exception:
            try:
                gen_model = trainer.model.module
            except Exception:
                gen_model = trainer.model

        with torch.inference_mode():
            out = gen_model.generate(
                **enc,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                logits_processor=logits_proc,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                num_return_sequences=1,
            )
        txt_full = tok.decode(out[0], skip_special_tokens=True)
        txt_direct = _strip_response_template(txt_full)
        inc_lines_direct = extract_inc_lines(txt_direct)

        # PPOTrainer.generate (response-only).
        qt = tok(prompt, return_tensors="pt").input_ids[0]
        responses = trainer.generate(
            [qt],
            batch_size=1,
            return_prompt=False,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            logits_processor=logits_proc,
        )
        resp0 = responses[0]
        if isinstance(resp0, torch.Tensor):
            resp_ids = resp0.detach().cpu().tolist()
        else:
            resp_ids = list(resp0)
        txt_trainer = tok.decode(resp_ids, skip_special_tokens=True)
        inc_lines_trainer = extract_inc_lines(txt_trainer)

        items.append(
            {
                "family": fam,
                "vin": float(vin),
                "vout": float(vout),
                "prompt_head": prompt[:240],
                "direct": {
                    "txt_head": txt_direct[:240],
                    "n_inc_lines": int(len(inc_lines_direct)),
                    "inc_head": "\n".join(inc_lines_direct[:8]),
                },
                "trainer_generate": {
                    "txt_head": txt_trainer[:240],
                    "n_inc_lines": int(len(inc_lines_trainer)),
                    "inc_head": "\n".join(inc_lines_trainer[:8]),
                },
            }
        )

    report = {"items": items}
    _write_json(outdir / "logs" / "sanity_generate.json", report)

    ok_any = any((it["direct"]["n_inc_lines"] > 0 or it["trainer_generate"]["n_inc_lines"] > 0) for it in items)
    if not ok_any:
        raise SystemExit(
            "[sanity_failed] PPO init generation produced zero INC lines for 2 tasks; check adapter/base_model/constraints."
        )

def _ensure_integrated_on_path() -> None:
    # Keep this in sync with eval_dcdc_family.py so --constrained works everywhere.
    for base in [
        "/root/workspace_autocircuit_rl",
        "/root/autodl-tmp/workspace_autocircuit_rl",
        "/root/autodl-tmp/workspace_autocircuit_rl/integrated",
    ]:
        if base not in sys.path and os.path.isdir(base):
            sys.path.append(base)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _snapshot_run_code(outdir: Path) -> None:
    """Copy the exact training/eval code into outdir for reproducibility."""
    try:
        snap = outdir / f'code_snapshot_{_now()}'
        _mkdir(snap)
        code_dir = Path(__file__).resolve().parent
        for name in [
            'train_ppo_dcdc.py',
            'train_sft_dcdc.py',
            'train_dpo_dcdc.py',
            'inc_parser.py',
            'dcdc_eval_tran.py',
            'dcdc_spice_builder.py',
            'dcdc_verifier.py',
            'dcdc_taskset.py',
            'eval_dcdc_family.py',
            'dcdc_templates.py',
        ]:
            p = code_dir / name
            if p.exists():
                shutil.copy2(p, snap / name)
        extra = Path('/root/autodl-tmp/workspace_autocircuit_rl/integrated/constraints.py')
        if extra.exists():
            shutil.copy2(extra, snap / 'integrated_constraints.py')
    except Exception as e:
        print(f'[warn] code_snapshot_failed: {type(e).__name__}: {e}', flush=True)


def _snapshot_trainable_params(model: Any) -> Dict[str, torch.Tensor]:
    snap: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if not bool(getattr(param, "requires_grad", False)):
            continue
        try:
            snap[name] = param.detach().cpu().clone()
        except Exception:
            continue
    return snap


def _restore_trainable_params(model: Any, snap: Dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if not bool(getattr(param, "requires_grad", False)):
            continue
        if name not in snap:
            continue
        try:
            param.data.copy_(snap[name].to(param.device))
        except Exception:
            continue


def _to_cpu_state(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_state(v) for v in obj)
    return obj


def _snapshot_optimizer_state(opt: Any) -> Dict[str, Any]:
    try:
        sd = opt.state_dict()
    except Exception:
        return {}
    # Keep on CPU to reduce GPU memory pressure; we'll move back on restore.
    return _to_cpu_state(sd)


def _restore_optimizer_state(opt: Any, sd: Dict[str, Any]) -> None:
    if not sd:
        return
    try:
        opt.load_state_dict(sd)
    except Exception:
        # If load fails, continue without restoring (best-effort).
        return
    try:
        for p in opt.state.keys():
            st = opt.state.get(p) or {}
            for k, v in list(st.items()):
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(p.device)
    except Exception:
        pass


def _guard_tasks_default() -> List[Tuple[str, float, float]]:
    # Keep this small but cover families that historically regress (boost/sepic).
    return [
        ("buck", 12.0, 5.0),
        ("buck", 18.0, 5.0),
        ("boost", 5.0, 12.0),
        ("boost", 9.0, 18.0),
        ("sepic", 12.0, 5.0),
        ("sepic", 5.0, 12.0),
        ("buckboost", 12.0, 5.0),
        ("buckboost", 9.0, 12.0),
    ]


def _guard_pool_from_taskset(
    *,
    tasks_all: List["Task"],
    max_tasks: int,
) -> List[Tuple[str, float, float]]:
    base = list(_guard_tasks_default())
    max_tasks = int(max_tasks)
    if max_tasks <= 0:
        return base
    if max_tasks <= len(base):
        return base[:max_tasks]

    seen = {(str(f).strip().lower(), float(vin), float(vout)) for f, vin, vout in base}

    fam_to_all: Dict[str, List[Tuple[str, float, float]]] = {}
    for t in tasks_all:
        fam = str(t.family).strip().lower()
        key = (fam, float(t.vin), float(t.vout))
        fam_to_all.setdefault(fam, []).append(key)
    for fam in list(fam_to_all.keys()):
        fam_to_all[fam] = sorted(fam_to_all[fam], key=lambda x: (x[1], x[2]))

    fams = ["buck", "boost", "sepic", "buckboost"]
    per_fam = int(math.ceil(float(max_tasks) / float(max(1, len(fams)))))
    fam_pools: Dict[str, List[Tuple[str, float, float]]] = {}
    for fam in fams:
        lst = fam_to_all.get(fam) or []
        if not lst:
            fam_pools[fam] = []
            continue
        n = min(int(per_fam), int(len(lst)))
        if n <= 1:
            fam_pools[fam] = [lst[0]]
            continue
        idxs = [int(round(i * (len(lst) - 1) / float(n - 1))) for i in range(n)]
        fam_pools[fam] = [lst[i] for i in idxs]

    pool = list(base)
    for i in range(max(len(v) for v in fam_pools.values()) if fam_pools else 0):
        for fam in fams:
            lst = fam_pools.get(fam) or []
            if i >= len(lst):
                continue
            key = lst[i]
            if key in seen:
                continue
            seen.add(key)
            pool.append(key)
            if len(pool) >= int(max_tasks):
                return pool
    return pool


def _cv_rate(flags: List[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(1 for x in flags if bool(x))) / float(len(flags))


def _auto_group_size(
    *,
    batch_size: int,
    requested_group_size: int,
    min_groups_per_step: int,
) -> int:
    batch_size = max(1, int(batch_size))
    requested_group_size = max(1, int(requested_group_size))
    min_groups_per_step = int(min_groups_per_step)
    if min_groups_per_step <= 0:
        return requested_group_size
    min_groups_per_step = max(1, min(int(min_groups_per_step), int(batch_size)))

    if (batch_size % requested_group_size == 0) and ((batch_size // requested_group_size) >= min_groups_per_step):
        return requested_group_size

    candidates = [
        gs
        for gs in range(1, int(requested_group_size) + 1)
        if (batch_size % gs == 0) and ((batch_size // gs) >= min_groups_per_step)
    ]
    if candidates:
        return max(candidates)

    # Fallback: maximize groups (min group size).
    for gs in range(1, int(batch_size) + 1):
        if batch_size % gs == 0:
            return int(gs)
    return 1


def _guard_eval(
    *,
    trainer: PPOTrainer,
    tok: Any,
    build_prompt,
    logits_proc: Optional[list],
    tol: float,
    min_elems: int,
    n_per_task: int,
    seed: int,
    t_pre: float,
    t_win: float,
    sim_timeout_s: float,
    autotune_duty: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    constrained: bool,
    sim_executor: Optional[ThreadPoolExecutor] = None,
    tasks: Optional[List[Tuple[str, float, float]]] = None,
) -> Dict[str, Any]:
    tasks = list(tasks) if tasks is not None else _guard_tasks_default()
    if int(n_per_task) <= 0:
        n_per_task = 1

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    was_train = bool(getattr(trainer.model, "training", False))
    trainer.model.eval()
    try:
        trainer.model.pretrained_model.config.use_cache = True
    except Exception:
        pass

    per_family_cv: Dict[str, List[bool]] = {}
    per_family_ce: Dict[str, List[bool]] = {}
    overall_cv: List[bool] = []
    overall_ce: List[bool] = []
    per_task_stats: List[Dict[str, Any]] = []

    for fam, vin, vout in tasks:
        fam_s = str(fam).strip().lower()
        prompt = str(build_prompt(fam_s, float(vin), float(vout)))
        qt = tok(prompt, return_tensors="pt").input_ids[0]

        q_batch = [qt for _ in range(int(n_per_task))]
        resp = trainer.generate(
            q_batch,
            batch_size=int(n_per_task),
            return_prompt=False,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            logits_processor=(logits_proc if bool(constrained) else None),
        )

        cv_flags: List[bool] = [False for _ in resp]
        ce_flags: List[bool] = [False for _ in resp]
        fut_to_i: Dict[Any, int] = {}
        for i, r in enumerate(resp):
            txt = tok.decode(r, skip_special_tokens=True)
            inc_lines = extract_inc_lines(_strip_response_template(txt))
            inc = ("\n".join(inc_lines).strip() + "\n") if inc_lines else ""

            ver = verify_inc_dcdc(inc, family=fam_s, vin=float(vin), vout=float(vout))
            meets_min = int(getattr(ver, "n_elems", 0) or 0) >= int(min_elems)
            if (not bool(ver.ok)) or (not meets_min):
                continue

            if sim_executor is None:
                detail = eval_one_detail_dcdc(
                    inc=inc,
                    family=fam_s,
                    vin=float(vin),
                    vout=float(vout),
                    tol=float(tol),
                    rload=10.0,
                    t_pre=float(t_pre),
                    t_win=float(t_win),
                    sim_timeout_s=float(sim_timeout_s),
                    autotune_duty=bool(autotune_duty),
                )
                ok = bool(detail.get("ok", False))
                cv_flags[i] = ok and bool(detail.get("pass_CV", False))
                ce_flags[i] = ok and bool(detail.get("pass_CE", False))
            else:
                fut = sim_executor.submit(
                    eval_one_detail_dcdc,
                    inc=inc,
                    family=fam_s,
                    vin=float(vin),
                    vout=float(vout),
                    tol=float(tol),
                    rload=10.0,
                    t_pre=float(t_pre),
                    t_win=float(t_win),
                    sim_timeout_s=float(sim_timeout_s),
                    autotune_duty=bool(autotune_duty),
                )
                fut_to_i[fut] = int(i)

        for fut in as_completed(fut_to_i):
            i = fut_to_i[fut]
            try:
                detail = fut.result()
            except Exception:
                detail = {"ok": False, "pass_CV": False, "pass_CE": False}
            ok = bool(detail.get("ok", False))
            cv_flags[i] = ok and bool(detail.get("pass_CV", False))
            ce_flags[i] = ok and bool(detail.get("pass_CE", False))

        per_family_cv.setdefault(fam_s, []).extend(cv_flags)
        per_family_ce.setdefault(fam_s, []).extend(ce_flags)
        overall_cv.extend(cv_flags)
        overall_ce.extend(ce_flags)
        per_task_stats.append(
            {
                "family": fam_s,
                "vin": float(vin),
                "vout": float(vout),
                "pass_rate": _cv_rate(cv_flags),
                "passes": int(sum(1 for x in cv_flags if bool(x))),
                "ce_pass_rate": _cv_rate(ce_flags),
                "ce_passes": int(sum(1 for x in ce_flags if bool(x))),
                "tried": int(len(cv_flags)),
            }
        )

    try:
        trainer.model.pretrained_model.config.use_cache = False
    except Exception:
        pass
    if was_train:
        trainer.model.train()

    by_family_cv = {fam: _cv_rate(fs) for fam, fs in per_family_cv.items()}
    by_family_ce = {fam: _cv_rate(fs) for fam, fs in per_family_ce.items()}
    min_family_cv = min(by_family_cv.values()) if by_family_cv else 0.0
    min_family_ce = min(by_family_ce.values()) if by_family_ce else 0.0
    return {
        "cv_rate": _cv_rate(overall_cv),
        "cv_min_family": float(min_family_cv),
        "cv_by_family": by_family_cv,
        "ce_rate": _cv_rate(overall_ce),
        "ce_min_family": float(min_family_ce),
        "ce_by_family": by_family_ce,
        "task_stats": per_task_stats,
        "n_samples": int(len(overall_cv)),
        "n_per_task": int(n_per_task),
        "n_tasks": int(len(tasks)),
        "tol": float(tol),
        "seed": int(seed),
    }


def _latest_ckpt(outdir: Path) -> Tuple[int, Path] | Tuple[None, None]:
    cks = []
    for p in outdir.glob("ppo_step_*"):
        if not p.is_dir():
            continue
        m = re.match(r"^ppo_step_(\d+)$", p.name)
        if not m:
            continue
        cks.append((int(m.group(1)), p))
    if not cks:
        return None, None
    cks.sort(key=lambda x: x[0])
    return cks[-1]


@dataclass
class WeightSchedule:
    w_cv: float
    w_ce: float
    w_ripple: float
    w_over: float


def anneal_weights(step: int, total_steps: int) -> WeightSchedule:
    # Multi-objective weight annealing: CV -> CE -> transient (ripple/overshoot).
    # Keep CV dominant to prevent regression on voltage tracking.
    t = float(step) / max(1.0, float(total_steps))
    w_cv = 3.0

    # CE ramps up after the model can hit CV.
    w_ce = 0.2 + 0.8 * min(1.0, max(0.0, (t - 0.25) / 0.50))

    # transient terms only in late stage
    w_ripple = 0.0 + 0.2 * min(1.0, max(0.0, (t - 0.60) / 0.40))
    w_over = 0.0 + 0.2 * min(1.0, max(0.0, (t - 0.60) / 0.40))
    return WeightSchedule(w_cv=w_cv, w_ce=w_ce, w_ripple=w_ripple, w_over=w_over)


@dataclass
class TolController:
    levels: List[float]
    idx: int = 0
    window: int = 128
    hi: float = 0.90
    lo: float = 0.60
    min_samples: int = 0
    history: List[int] = None  # 1 if pass_CV else 0

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []
        self.levels = [float(x) for x in self.levels]
        self.idx = max(0, min(int(self.idx), len(self.levels) - 1))
        self.window = int(self.window)
        if self.window <= 0:
            self.window = 1
        self.min_samples = int(self.min_samples)
        if self.min_samples <= 0:
            self.min_samples = int(self.window)

    @property
    def tol(self) -> float:
        return float(self.levels[self.idx])

    def update(self, pass_cv_flags: List[bool]) -> None:
        for f in pass_cv_flags:
            self.history.append(1 if bool(f) else 0)
        if len(self.history) > int(self.window):
            self.history = self.history[-int(self.window) :]

    def maybe_adjust(self) -> None:
        if not self.history:
            return
        if len(self.history) < int(self.min_samples):
            return
        rate = float(sum(self.history)) / max(1.0, float(len(self.history)))
        if rate >= float(self.hi) and self.idx < (len(self.levels) - 1):
            self.idx += 1
        elif rate <= float(self.lo) and self.idx > 0:
            self.idx -= 1

    def summary(self) -> Dict[str, Any]:
        rate = float(sum(self.history)) / max(1.0, float(len(self.history))) if self.history else 0.0
        return {"tol": float(self.tol), "tol_idx": int(self.idx), "cv_rate_win": float(rate), "win": int(len(self.history))}


@dataclass
class RiskController:
    """
    Constrained / risk-sensitive controller to directly optimize *sample-level* success rate.
    We maintain a Lagrange multiplier (lambda_fail) that penalizes failures when the observed
    fail-rate exceeds a target (default 10% -> success >= 90%).
    """

    target_fail: float = 0.10
    lambda_lr: float = 0.10
    lambda_max: float = 10.0
    lam: float = 0.0
    window: int = 128
    history: List[int] = None  # 1 if strict_success else 0

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []
        self.target_fail = float(self.target_fail)
        self.lambda_lr = float(self.lambda_lr)
        self.lambda_max = float(self.lambda_max)
        self.lam = float(self.lam)
        self.window = int(self.window)
        if self.window <= 0:
            self.window = 1

    def update(self, strict_success_flags: List[bool]) -> None:
        for f in strict_success_flags:
            self.history.append(1 if bool(f) else 0)
        if len(self.history) > int(self.window):
            self.history = self.history[-int(self.window) :]

    def maybe_adjust(self) -> None:
        if not self.history:
            return
        succ = float(sum(self.history)) / max(1.0, float(len(self.history)))
        fail = 1.0 - succ
        # Lagrangian ascent on constraint violation: fail <= target_fail
        self.lam = float(self.lam) + float(self.lambda_lr) * float(fail - float(self.target_fail))
        if self.lam < 0.0:
            self.lam = 0.0
        if self.lam > float(self.lambda_max):
            self.lam = float(self.lambda_max)

    def penalty(self, strict_success: bool) -> float:
        return 0.0 if bool(strict_success) else float(self.lam)

    def summary(self) -> Dict[str, Any]:
        succ = float(sum(self.history)) / max(1.0, float(len(self.history))) if self.history else 0.0
        return {
            "target_fail": float(self.target_fail),
            "success_rate_win": float(succ),
            "fail_rate_win": float(1.0 - succ),
            "lambda_fail": float(self.lam),
            "win": int(len(self.history)),
        }


def compute_reward(
    detail: dict,
    *,
    family: str,
    vout: float,
    tol: float,
    weights: WeightSchedule,
    min_elems: int,
    n_elems: int,
    ver_ok: bool,
    violations: List[str],
) -> Tuple[float, Dict[str, float]]:
    # Element-count shaping: keep >=min_elems as the strict "final success" criterion,
    # but provide a *graded* penalty so PPO has a learning signal before it reaches 20+ lines.
    size_frac = 1.0
    if int(min_elems) > 0:
        size_frac = max(0.0, min(1.0, float(n_elems) / float(min_elems)))
    pen_size = -1.0 * (1.0 - size_frac)

    # Rule-level shaped reward (verifiable signal) to prevent collapse to empty outputs.
    viol_set = set(str(x) for x in (violations or []))
    fam = str(family or "").strip().lower()
    def _w(name: str, w: float) -> tuple[str, float]:
        return (name, float(w))

    # Weighted verifier score: emphasize power-stage topology constraints so PPO learns structure first,
    # especially for buck-boost where many failures are structural (not just parameter tuning).
    checks: List[tuple[str, float]] = [
        _w("no_elements", 3.0),
        _w("missing_inductor", 1.0),
        _w("missing_capacitor", 1.0),
        _w("missing_vin_node", 1.0),
        _w("missing_out_node", 1.0),
        _w("missing_gnd_node", 1.0),
        _w("disconnected_vin_to_out", 1.0),
    ]
    if fam == "buck":
        checks += [
            _w("buck_missing_switch", 1.0),
            _w("buck_missing_diode", 1.0),
            _w("buck_missing_output_cap", 1.0),
            _w("buck_missing_vin_sw_switch", 3.0),
            _w("buck_missing_diode_0_to_sw", 3.0),
            _w("buck_missing_sw_out_inductor", 3.0),
            _w("buck_missing_out_gnd_cap", 3.0),
        ]
    elif fam == "boost":
        checks += [
            _w("boost_missing_switch", 1.0),
            _w("boost_missing_diode", 1.0),
            _w("boost_missing_vin_sw_inductor", 3.0),
            _w("boost_missing_sw_gnd_switch", 3.0),
            _w("boost_missing_diode_sw_to_out", 3.0),
            _w("boost_missing_out_gnd_cap", 3.0),
        ]
    elif fam == "sepic":
        checks += [
            _w("sepic_missing_switch", 1.0),
            _w("sepic_missing_diode", 1.0),
            _w("sepic_missing_2_inductors", 2.0),
            _w("sepic_missing_2_caps", 2.0),
            _w("sepic_missing_vin_sw_inductor", 3.0),
            _w("sepic_missing_sw_n1_cap", 3.0),
            _w("sepic_missing_n1_gnd_inductor", 3.0),
            _w("sepic_missing_sw_gnd_switch", 3.0),
            _w("sepic_missing_diode_n1_to_out", 3.0),
            _w("sepic_missing_out_gnd_cap", 3.0),
        ]
    elif fam == "buckboost":
        checks += [
            _w("buckboost_missing_2_switches", 2.0),
            _w("buckboost_missing_2_diodes", 2.0),
            _w("buckboost_missing_2_inductors", 2.0),
            _w("buckboost_missing_2_caps", 2.0),
            _w("buckboost_missing_gate1_gate2", 2.0),
            _w("buckboost_missing_vin_sw1_switch", 3.0),
            _w("buckboost_missing_diode_0_to_sw1", 3.0),
            _w("buckboost_missing_sw1_mid_inductor", 3.0),
            _w("buckboost_missing_mid_gnd_cap", 3.0),
            _w("buckboost_missing_mid_sw2_inductor", 3.0),
            _w("buckboost_missing_sw2_gnd_switch", 3.0),
            _w("buckboost_missing_diode_sw2_to_out", 3.0),
            _w("buckboost_missing_out_gnd_cap", 3.0),
        ]

    total_checks = max(1e-9, float(sum(w for _, w in checks)))
    if ("no_elements" in viol_set) or (int(n_elems) <= 0):
        # Special case: verifier returns early for empty outputs, so many
        # "missing_*" violations are not present. Penalize empty output strongly.
        rule_score = 0.0
        reward_rule = -1.0
    else:
        passed_w = float(sum(w for c, w in checks if c not in viol_set))
        rule_score = float(passed_w) / float(total_checks)
        reward_rule = -1.0 + 0.8 * rule_score

    # If structure verification or simulation fails, fall back to rule-level reward.
    #
    # IMPORTANT: Never let "sim fail / garbage" become preferable to a valid-but-wrong-voltage circuit.
    # Otherwise PPO will learn to intentionally break generation / simulation to avoid CV penalties.
    if not bool(ver_ok) or not bool(detail.get("ok", False)):
        # Strict ordering: invalid < sim_fail < sim_ok_far_miss < sim_ok_near_miss < pass_CV
        pen_fail = -4.0 if (not bool(ver_ok)) else -3.0
        total = float(reward_rule + pen_size + pen_fail)
        return total, {
            "reward_total": total,
            "reward_rule": float(reward_rule),
            "rule_score": float(rule_score),
            "reward_cv": 0.0,
            "reward_ce": 0.0,
            "pen_ripple": 0.0,
            "pen_over": 0.0,
            "pen_size": float(pen_size),
            "pen_fail": float(pen_fail),
        }

    vavg = _safe_float(detail.get("vavg", 0.0), 0.0)
    eff = _safe_float(detail.get("eff", 0.0), 0.0)
    ripple = _safe_float(detail.get("ripple", 0.0), 0.0)
    overshoot = _safe_float(detail.get("overshoot", 0.0), 0.0)

    err = abs(vavg - float(vout)) / max(1e-6, float(vout))
    tol_f = max(1e-6, float(tol))
    pass_cv = bool(err <= tol_f)
    if pass_cv:
        # Match the original scheme: CV pass gives a strong +w_cv base reward (≈3 at stage-1),
        # but also learn margin within tol (otherwise CV can drop just because tol tightens).
        margin = min(1.0, max(0.0, float(err) / float(tol_f)))
        cv_term = 1.0 - 0.20 * margin
    else:
        # For CV misses, keep a graded negative signal without early saturation.
        # Map over=(err-tol)/tol in [0, +inf) to (-1, 0] smoothly.
        over = max(0.0, (err - tol_f) / tol_f)
        k = 2.0
        cv_term = -float(over) / float(over + k)

    # Enforce the >=min_elems requirement inside the reward (graded), so PPO cannot ignore it.
    reward_cv = float(weights.w_cv * cv_term * size_frac)
    if not pass_cv:
        # Progressive objective: no CE/transient credit until CV is satisfied.
        reward_ce = 0.0
        pen_ripple = 0.0
        pen_over = 0.0
    else:
        reward_ce = float(weights.w_ce * eff * size_frac)
        pen_ripple = float(-weights.w_ripple * (ripple / max(1e-6, float(vout))))
        pen_over = float(-weights.w_over * overshoot)

    # For valid simulations, rule checks already passed (ver_ok=True), so we do not penalize structure here.
    reward_rule = 0.0
    total = float(reward_rule + reward_cv + reward_ce + pen_ripple + pen_over + pen_size)
    parts = {
        "reward_total": total,
        "reward_rule": float(reward_rule),
        "rule_score": float(rule_score),
        "reward_cv": reward_cv,
        "reward_ce": reward_ce,
        "pen_ripple": pen_ripple,
        "pen_over": pen_over,
        "pen_size": float(pen_size),
        "pen_fail": 0.0,
    }
    return total, parts


class PPOTrainerWithEntropy(PPOTrainer):
    def __init__(self, *args, ent_coef: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._ent_coef = float(ent_coef)

    def loss(  # type: ignore[override]
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        pg_loss, scaled_vf_loss, train_stats = super().loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
        )
        if self._ent_coef == 0.0:
            return pg_loss, scaled_vf_loss, train_stats

        ent = train_stats.get("policy/entropy", None)
        if ent is None:
            return pg_loss, scaled_vf_loss, train_stats

        ent_bonus = float(self._ent_coef) * ent
        pg_loss = pg_loss - ent_bonus

        # train_stats is a flat dict from TRL's flatten_dict(stats).
        train_stats["loss/policy"] = train_stats["loss/policy"] - ent_bonus.detach()
        train_stats["loss/total"] = train_stats["loss/policy"] + self.config.vf_coef * train_stats["loss/value"]
        train_stats["loss/entropy_bonus"] = ent_bonus.detach()
        return pg_loss, scaled_vf_loss, train_stats


def _group_normalize_rewards(
    raw_rewards: List[float],
    group_ids: List[int],
    mode: str,
    *,
    rel_coef: float = 0.5,
) -> List[float]:
    if not raw_rewards:
        return []
    if not group_ids or len(group_ids) != len(raw_rewards):
        return list(raw_rewards)
    m = (mode or "none").strip().lower()
    if m == "none":
        return list(raw_rewards)

    out = [0.0 for _ in raw_rewards]
    by_group: Dict[int, List[int]] = {}
    for i, gid in enumerate(group_ids):
        by_group.setdefault(int(gid), []).append(i)

    def _rank_unit(vals0: List[float]) -> List[float]:
        # Tie-aware average-rank mapping -> [-1, 1]. If all equal, all become 0.
        order0 = sorted(range(len(vals0)), key=lambda j: vals0[j])
        ranks0 = [0.0 for _ in vals0]
        pos0 = 0
        while pos0 < len(order0):
            j0 = order0[pos0]
            v0 = vals0[j0]
            end0 = pos0 + 1
            while end0 < len(order0) and vals0[order0[end0]] == v0:
                end0 += 1
            avg_rank0 = 0.5 * (float(pos0) + float(end0 - 1))
            for k0 in range(pos0, end0):
                ranks0[order0[k0]] = avg_rank0
            pos0 = end0
        denom0 = float(max(1, len(vals0) - 1))
        return [(float(r) / denom0) * 2.0 - 1.0 for r in ranks0]

    for _, idxs in by_group.items():
        if len(idxs) <= 1:
            for i in idxs:
                out[i] = float(raw_rewards[i])
            continue
        vals = [float(raw_rewards[i]) for i in idxs]
        vmin = min(vals)
        vmax = max(vals)
        # Guard: NEVER let relative normalization turn failures into "positive" rewards.
        #
        # - For replacement modes (rank/zscore), we only apply normalization when *all* samples are strictly positive.
        # - For *_add modes, we apply the relative term only to the strictly-positive subset; non-positive samples
        #   keep their absolute rewards unchanged.
        if abs(vmax - vmin) < 1e-12:
            # No within-group signal; keep absolute rewards so PPO can still learn from all-fail groups.
            for i in idxs:
                out[i] = float(raw_rewards[i])
            continue
        if m in {"rank", "zscore"}:
            if vmin <= 0.0:
                for i in idxs:
                    out[i] = float(raw_rewards[i])
                continue

        if m in {"rank", "rank_add"}:
            if m == "rank":
                r = _rank_unit(vals)
                for j, i in enumerate(idxs):
                    out[i] = float(r[j])
            else:
                pos_local = [(j, i) for j, i in enumerate(idxs) if float(raw_rewards[i]) > 0.0]
                if len(pos_local) <= 1:
                    for i in idxs:
                        out[i] = float(raw_rewards[i])
                else:
                    pos_vals = [float(raw_rewards[i]) for _, i in pos_local]
                    r_pos = _rank_unit(pos_vals)
                    pos_rank = {i: float(r_pos[k]) for k, (_, i) in enumerate(pos_local)}
                    for i in idxs:
                        rr = float(raw_rewards[i])
                        out[i] = rr if rr <= 0.0 else rr + float(rel_coef) * float(pos_rank.get(i, 0.0))
        elif m == "zscore":
            mean = sum(vals) / float(len(vals))
            var = sum((v - mean) ** 2 for v in vals) / float(max(1, len(vals) - 1))
            std = (var ** 0.5) if var > 0.0 else 1.0
            for j, i in enumerate(idxs):
                out[i] = (float(raw_rewards[i]) - mean) / float(std)
        elif m == "zscore_add":
            pos = [i for i in idxs if float(raw_rewards[i]) > 0.0]
            if len(pos) <= 1:
                for i in idxs:
                    out[i] = float(raw_rewards[i])
            else:
                pos_vals = [float(raw_rewards[i]) for i in pos]
                mean = sum(pos_vals) / float(len(pos_vals))
                var = sum((v - mean) ** 2 for v in pos_vals) / float(max(1, len(pos_vals) - 1))
                std = (var ** 0.5) if var > 0.0 else 1.0
                z_pos = {i: (float(raw_rewards[i]) - mean) / float(std) for i in pos}
                for i in idxs:
                    rr = float(raw_rewards[i])
                    out[i] = rr if rr <= 0.0 else rr + float(rel_coef) * float(z_pos.get(i, 0.0))
        else:
            for i in idxs:
                out[i] = float(raw_rewards[i])
    return out


def _group_pareto_add_rewards(
    *,
    raw_rewards: List[float],
    details: List[Dict[str, Any]],
    group_ids: List[int],
    strict_success_flags: List[bool],
    rel_coef: float,
    min_elems: int,
) -> Tuple[List[float], List[float]]:
    """
    Pareto-GRPO-style within-group shaping:
      - compute multi-objective Pareto rank + crowding distance inside each (task) group
      - map to a relative term r_rel in [-1, 1] for STRICT successes only (pass_CV & min_elems & ok)
      - add r_rel to absolute reward: r_used = r_raw + rel_coef * r_rel

    Safety invariants:
      - failures (non-strict-success) always keep absolute rewards (no positive shaping)
      - if a group has <=1 strict-success sample, no relative signal is applied (r_rel=0).
    """

    if not raw_rewards:
        return [], []
    if (not group_ids) or (len(group_ids) != len(raw_rewards)) or (len(details) != len(raw_rewards)):
        return list(raw_rewards), [0.0 for _ in raw_rewards]

    out = list(float(x) for x in raw_rewards)
    rel_terms = [0.0 for _ in raw_rewards]

    def _dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
        le = True
        lt = False
        for x, y in zip(a, b):
            if float(x) > float(y) + 1e-12:
                le = False
                break
            if float(x) < float(y) - 1e-12:
                lt = True
        return bool(le and lt)

    def _pareto_ranks(objs: List[Tuple[float, ...]]) -> List[int]:
        n = int(len(objs))
        ranks = [-1 for _ in range(n)]
        remaining = set(range(n))
        r = 0
        while remaining:
            front: List[int] = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if _dominates(objs[j], objs[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            for i in front:
                ranks[i] = int(r)
                remaining.remove(i)
            r += 1
        return ranks

    def _crowding_distance(objs: List[Tuple[float, ...]], ranks: List[int]) -> List[float]:
        n = int(len(objs))
        if n == 0:
            return []
        m = int(len(objs[0]))
        cd = [0.0 for _ in range(n)]
        fronts: Dict[int, List[int]] = {}
        for i, rr in enumerate(ranks):
            fronts.setdefault(int(rr), []).append(int(i))
        for _, idxs in fronts.items():
            if len(idxs) <= 2:
                for i in idxs:
                    cd[i] = float("inf")
                continue
            for k in range(m):
                idxs_sorted = sorted(idxs, key=lambda i: float(objs[i][k]))
                lo = float(objs[idxs_sorted[0]][k])
                hi = float(objs[idxs_sorted[-1]][k])
                denom = max(1e-12, float(hi - lo))
                cd[idxs_sorted[0]] = float("inf")
                cd[idxs_sorted[-1]] = float("inf")
                for j in range(1, len(idxs_sorted) - 1):
                    prev_v = float(objs[idxs_sorted[j - 1]][k])
                    next_v = float(objs[idxs_sorted[j + 1]][k])
                    if cd[idxs_sorted[j]] != float("inf"):
                        cd[idxs_sorted[j]] += (next_v - prev_v) / denom
        return cd

    def _rank_unit_from_keys(keys: List[Tuple[float, float]]) -> List[float]:
        # Tie-aware average-rank mapping -> [-1, 1]. If all equal, all become 0.
        order0 = sorted(range(len(keys)), key=lambda j: keys[j])
        ranks0 = [0.0 for _ in keys]
        pos0 = 0
        while pos0 < len(order0):
            j0 = order0[pos0]
            v0 = keys[j0]
            end0 = pos0 + 1
            while end0 < len(order0) and keys[order0[end0]] == v0:
                end0 += 1
            avg_rank0 = 0.5 * (float(pos0) + float(end0 - 1))
            for k0 in range(pos0, end0):
                ranks0[order0[k0]] = avg_rank0
            pos0 = end0
        denom0 = float(max(1, len(keys) - 1))
        return [(float(r) / denom0) * 2.0 - 1.0 for r in ranks0]

    by_group: Dict[int, List[int]] = {}
    for i, gid in enumerate(group_ids):
        by_group.setdefault(int(gid), []).append(int(i))

    for _, idxs in by_group.items():
        succ = [i for i in idxs if bool(strict_success_flags[i])]
        if len(succ) <= 1:
            continue

        objs: List[Tuple[float, ...]] = []
        for i in succ:
            rec = details[i] or {}
            d = rec.get("detail") or {}
            vout = float(rec.get("vout") or 0.0)
            vavg = float(d.get("vavg") or 0.0)
            eff = float(d.get("eff") or 0.0)
            ripple = float(d.get("ripple") or 0.0)
            over = float(d.get("overshoot") or 0.0)
            n_elems = float(rec.get("n_elems") or 0.0)

            # Minimize: cv_err, ripple_norm, overshoot; Maximize: eff, n_elems
            cv_err = float(abs(vavg - vout) / max(1e-6, abs(vout)))
            ripple_n = float(ripple / max(1e-6, abs(vout)))
            objs.append((cv_err, ripple_n, over, -eff, -n_elems))

        ranks = _pareto_ranks(objs)
        crowd = _crowding_distance(objs, ranks)

        # Ordering key: (pareto_rank, -crowding). Quantize crowding to avoid tiny-float instability.
        keys: List[Tuple[float, float]] = []
        for rr, cc in zip(ranks, crowd):
            c0 = 1e9 if cc == float("inf") else float(cc)
            keys.append((float(rr), -float(round(c0, 6))))

        rel = _rank_unit_from_keys(keys)
        for j, i in enumerate(succ):
            # Relative term is only applied to strict successes.
            rel_terms[i] = float(rel[j])
            out[i] = float(out[i]) + float(rel_coef) * float(rel_terms[i])

    # Extra safety: never shape failures (even if caller passed inconsistent flags).
    for i, ss in enumerate(strict_success_flags):
        if not bool(ss):
            out[i] = float(raw_rewards[i])
            rel_terms[i] = 0.0
            continue
        if int(details[i].get("n_elems") or 0) < int(min_elems):
            out[i] = float(raw_rewards[i])
            rel_terms[i] = 0.0

    return out, rel_terms


def load_value_head_model(
    base_model: str,
    *,
    init_adapter: Optional[str],
    resume_ckpt: Optional[Path],
    bf16: bool = True,
    gradient_checkpointing: bool = True,
) -> AutoModelForCausalLMWithValueHead:
    tok_dtype = torch.bfloat16 if bool(bf16) else torch.float16
    base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype=tok_dtype)
    if bool(gradient_checkpointing):
        try:
            base.gradient_checkpointing_enable()
        except Exception:
            pass
        _force_inputs_require_grads(base)
    base.config.use_cache = False

    if resume_ckpt is not None:
        peft = PeftModel.from_pretrained(base, str(resume_ckpt), is_trainable=True)
    elif init_adapter:
        peft = PeftModel.from_pretrained(base, str(init_adapter), is_trainable=True)
    else:
        raise SystemExit("Need --sft_adapter for fresh PPO, or --resume with existing ppo_step_* checkpoint.")

    model = AutoModelForCausalLMWithValueHead.from_pretrained(peft)
    model.pretrained_model.config.use_cache = False

    # Make gradient checkpointing work with PEFT (avoid "Gradients will be None").
    if bool(gradient_checkpointing):
        _force_inputs_require_grads(model.pretrained_model)
        try:
            model.pretrained_model.gradient_checkpointing_enable()
        except Exception:
            pass

    # Load value-head weights if present (saved by PreTrainedModelWrapper.save_pretrained).
    if resume_ckpt is not None:
        vh = Path(resume_ckpt) / "pytorch_model.bin"
        if vh.exists():
            sd = torch.load(str(vh), map_location="cpu")
            model.load_state_dict(sd, strict=False)

    return model


def main() -> None:
    global _STOP_REQUESTED, _FORCE_SAVE_REQUESTED
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--sft_adapter", default="", help="LoRA adapter dir from SFT (start point for PPO)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--anneal_steps", type=int, default=0, help="Total steps used for weight annealing (0 -> use --steps).")
    ap.add_argument("--save_steps", type=int, default=25)
    ap.add_argument("--save_total_limit", type=int, default=3, help="Keep only the latest N ppo_step_* checkpoints; 0 keeps all.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ddp_backend", default="nccl", help="torch.distributed backend for TRL/Accelerate. Use gloo if NCCL is unstable on this machine.")
    ap.add_argument("--debug_dist", action="store_true", help="Print torch.distributed backend diagnostics and exit early points.")
    ap.add_argument("--seed", type=int, default=2025)

    # PPO hyperparams (match paper-style defaults)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--mini_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--target_kl", type=float, default=0.03)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument(
        "--no_ref_model",
        action="store_true",
        help="Disable creating a frozen reference model for KL (NOT recommended).",
    )
    ap.add_argument("--cliprange", type=float, default=0.2)
    ap.add_argument("--cliprange_value", type=float, default=0.2)
    ap.add_argument("--ppo_epochs", type=int, default=4)

    # generation + simulation
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--constrained", action="store_true", help="Use CharClassLogitsProcessor during generation")
    ap.add_argument("--group_size", type=int, default=4, help="Group size for within-task relative learning (GRPO-style).")
    ap.add_argument(
        "--min_groups_per_step",
        type=int,
        default=0,
        help="If >0, auto-reduce --group_size (per-step) so batch_size/group_size >= this many groups (reduces task variance).",
    )
    ap.add_argument(
        "--group_reward_mode",
        choices=["none", "rank", "zscore", "rank_add", "zscore_add", "pareto_add"],
        default="none",
        help=(
            "How to mix within-group relative learning into PPO rewards. "
            "'rank'/'zscore' replace absolute rewards (can collapse); "
            "'rank_add'/'zscore_add' add a relative term on top of absolute rewards; "
            "'pareto_add' adds a Pareto-GRPO-style relative term based on (cv_err, eff, ripple, overshoot, n_elems)."
        ),
    )
    ap.add_argument(
        "--group_rel_coef",
        type=float,
        default=0.5,
        help="Scale for the relative (group) term when using *_add modes.",
    )
    ap.add_argument(
        "--batch_task_mode",
        choices=["mixed", "same_family", "same_task", "balanced_families"],
        default="same_family",
        help=(
            "How to sample tasks within one PPO step. 'mixed' may break some models because TRL uses left-padding when batching; "
            "use 'same_family' or 'same_task' for stable generation. "
            "'balanced_families' forces each group to use a different family (round-robin) to prevent a hard family "
            "(e.g., buckboost) from dominating gradients."
        ),
    )
    ap.add_argument(
        "--train_task_pool",
        choices=["all", "guard"],
        default="all",
        help="Task pool used to sample PPO rollouts. 'guard' restricts rollouts to the current guard pool (stabilizes CV and speeds up improving best_guard CV).",
    )
    ap.add_argument("--min_elems", type=int, default=20)
    ap.add_argument("--t_pre", type=float, default=0.008)
    ap.add_argument("--t_win", type=float, default=0.002)
    ap.add_argument("--sim_timeout_s", type=float, default=180.0)
    ap.add_argument("--autotune_duty", action="store_true", help="Enable 1-step duty autotune in eval")
    ap.add_argument(
        "--sim_workers",
        type=int,
        default=0,
        help="Parallel ngspice evaluations (threads). 0=auto (uses ~half of logical CPU cores, capped).",
    )

    # adaptive tolerance
    # NOTE: user requirement: fixed CV tolerance (no adaptive schedule).
    ap.add_argument("--tol_levels", default="0.01")
    ap.add_argument("--tol_window", type=int, default=128)
    ap.add_argument("--tol_hi", type=float, default=0.95)
    ap.add_argument("--tol_lo", type=float, default=0.85)
    ap.add_argument("--tol_adjust_every", type=int, default=10)
    ap.add_argument("--tol_min_samples", type=int, default=0, help="min CV samples before tol adjust (0 -> use tol_window)")

    # Convergence / early-stop (optional): stop only when tol is tight and CV is stably high.
    # NOTE: by default we DISABLE early-stop (window=0), because weight annealing needs many steps
    # to optimize CE/transient objectives even after CV becomes easy.
    ap.add_argument("--converge_tol", type=float, default=0.02, help="Early-stop only when tol <= this value.")
    ap.add_argument(
        "--converge_cv_window",
        type=int,
        default=0,
        help="0 disables early-stop; otherwise require this many consecutive steps for convergence.",
    )
    ap.add_argument("--converge_cv_threshold", type=float, default=0.9, help="Mean CV over the window to declare convergence.")
    ap.add_argument(
        "--converge_use_strict_cv",
        action="store_true",
        help="Use cv_strict_rate (CV+min_elems) instead of cv_rate for convergence check.",
    )

    # Risk-sensitive / constrained objective: directly target sample-level success rate.
    ap.add_argument(
        "--enable_risk_penalty",
        action="store_true",
        help="Enable Lagrangian failure penalty (experimental; keep OFF to match the original PPO).",
    )
    ap.add_argument("--target_fail_rate", type=float, default=0.10)
    ap.add_argument("--lambda_fail_init", type=float, default=0.0)
    ap.add_argument("--lambda_fail_lr", type=float, default=0.10)
    ap.add_argument("--lambda_fail_max", type=float, default=10.0)
    ap.add_argument("--risk_window", type=int, default=128)
    ap.add_argument("--risk_adjust_every", type=int, default=1)

    # CV non-regression guard (user requirement): reject PPO updates that reduce CV on a fixed mini-eval.
    ap.add_argument("--no_cv_guard", action="store_true", help="Disable CV non-regression guard (NOT recommended).")
    ap.add_argument("--cv_guard_every", type=int, default=1)
    ap.add_argument("--cv_guard_n_per_task", type=int, default=2)
    ap.add_argument("--cv_guard_seed", type=int, default=20250105)
    ap.add_argument(
        "--cv_guard_total_samples",
        type=int,
        default=16,
        help="If >0, auto-set guard n_per_task ~= total_samples/n_tasks (keeps guard cost bounded as n_tasks grows).",
    )
    ap.add_argument(
        "--cv_guard_tasks_max",
        type=int,
        default=0,
        help="Max guard tasks for adaptive expansion. 0 means no cap (use full taskset).",
    )
    ap.add_argument(
        "--cv_guard_init_tasks",
        type=int,
        default=0,
        help="Initial guard task count (0 uses the default core guard size, typically 8).",
    )
    ap.add_argument("--cv_guard_expand_step", type=int, default=8, help="Adaptive guard: add this many tasks per expansion.")
    ap.add_argument(
        "--cv_guard_expand_streak",
        type=int,
        default=1,
        help=(
            "Expand guard after this many consecutive guard evals where CORE-GUARD cv_rate >= --cv_guard_expand_trigger_cv "
            "and CORE-GUARD ce_rate > --cv_guard_expand_ce_threshold."
        ),
    )
    ap.add_argument(
        "--cv_guard_expand_trigger_cv",
        type=float,
        default=1.0,
        help="Adaptive guard expansion: require CORE-GUARD cv_rate >= this threshold (together with CORE-GUARD ce_rate > --cv_guard_expand_ce_threshold).",
    )
    ap.add_argument(
        "--cv_guard_expand_ce_threshold",
        type=float,
        default=0.8,
        help="Adaptive guard expansion: require CORE-GUARD ce_rate > this threshold (together with CORE-GUARD cv_rate >= --cv_guard_expand_trigger_cv).",
    )
    ap.add_argument(
        "--cv_guard_expand_min_cv",
        type=float,
        default=0.7,
        help="Adaptive guard expansion: commit guard size increase immediately; only allow ppo_best update on expanded guard once expanded guard cv_rate >= this threshold.",
    )
    ap.add_argument(
        "--cv_guard_best_confirm_seeds",
        type=int,
        default=2,
        help="When guard is expanded (n_tasks>core), confirm ppo_best candidates with this many extra guard eval seeds (>=1).",
    )
    ap.add_argument(
        "--cv_guard_best_confirm_total_samples",
        type=int,
        default=32,
        help="Total guard samples budget per confirm eval for ppo_best selection (0 -> use --cv_guard_total_samples).",
    )
    ap.add_argument(
        "--hard_task_frac",
        type=float,
        default=0.5,
        help="Probability of sampling mined hard tasks for the chosen family (0 disables).",
    )
    ap.add_argument(
        "--hard_task_pass_rate_lt",
        type=float,
        default=1.0,
        help="A task is considered hard if guard pass_rate < this threshold (default: <1.0).",
    )
    args = ap.parse_args()

    # Pre-initialize Accelerate/torch.distributed with the requested backend before TRL creates its Accelerator.
    # This avoids Accelerate's singleton state being initialized with NCCL by an earlier import on some systems.
    try:
        import torch.distributed as dist
        if int(str(os.environ.get('WORLD_SIZE') or '1').strip() or '1') > 1 and not dist.is_initialized():
            backend = str(getattr(args, 'ddp_backend', '') or '').strip() or 'nccl'
            from accelerate.state import PartialState
            _ = PartialState(backend=backend)
            print(f"[dist] preinit backend={backend}", flush=True)
    except Exception as e:
        print(f"[dist] preinit failed: {type(e).__name__}: {e}", flush=True)

    _patch_torch_numpy_fancy_indexing()
    _patch_trl_adaptive_kl_controller_clip()

    # User requirement: force a fixed tolerance (no adaptive schedule).
    # We keep the TolController machinery for logging/resume, but collapse it to a single level.
    args.tol_levels = "0.01"

    sim_workers = _auto_sim_workers_dist(getattr(args, "sim_workers", 0))
    sim_executor: Optional[ThreadPoolExecutor] = None
    if int(sim_workers) > 1:
        sim_executor = ThreadPoolExecutor(max_workers=int(sim_workers))
        print(f"[sim] sim_workers={int(sim_workers)}", flush=True)
        try:
            import atexit

            atexit.register(sim_executor.shutdown, wait=True, cancel_futures=False)
        except Exception:
            pass

    anneal_total_steps = int(getattr(args, 'anneal_steps', 0) or 0)
    if anneal_total_steps <= 0:
        anneal_total_steps = int(args.steps)


    _ensure_integrated_on_path()

    outdir = Path(args.outdir)
    _mkdir(outdir)
    _mkdir(outdir / "logs")

    # Round-wise guard schedule (user requirement): add +|G0| tasks each VP-SPI round (8,16,24,...) instead of doubling.
    # We infer round index from the output path (.../round_00/... , .../round_01/..., etc.)
    # so this works even when the pipeline script does not explicitly pass guard args.
    round_idx: Optional[int] = None
    try:
        for part in reversed(outdir.parts):
            m = re.match(r"^round[_-]?(\d+)$", str(part).strip())
            if m:
                round_idx = int(m.group(1))
                break
    except Exception:
        round_idx = None

    if round_idx is not None and int(round_idx) >= 0:
        core_n = int(len(_guard_tasks_default()))
        sched_tasks_linear = int(core_n) * (int(round_idx) + 1)
        sched_tasks_double = int(core_n) * (2 ** int(round_idx))

        # If the pipeline passed the old doubling schedule, override to the new linear +8 schedule.
        passed_init = int(getattr(args, 'cv_guard_init_tasks', 0) or 0)
        passed_max = int(getattr(args, 'cv_guard_tasks_max', 0) or 0)
        if passed_init in (0, int(sched_tasks_double)) and passed_max in (0, int(sched_tasks_double)):
            args.cv_guard_init_tasks = int(sched_tasks_linear)
            args.cv_guard_tasks_max = int(sched_tasks_linear)
        elif passed_init <= 0:
            args.cv_guard_init_tasks = int(sched_tasks_linear)
        elif passed_max <= 0:
            args.cv_guard_tasks_max = int(sched_tasks_linear)

        # Keep per-task guard samples roughly stable by scaling the total budget with guard size.
        passed_total = int(getattr(args, 'cv_guard_total_samples', 0) or 0)
        sched_total_linear = int(16) * (int(round_idx) + 1)
        sched_total_double = int(16) * (2 ** int(round_idx))
        if passed_total in (0, 16, int(sched_total_double)):
            args.cv_guard_total_samples = int(sched_total_linear)

        # Disable within-round expansion for the round-wise schedule unless explicitly overridden.
        if int(getattr(args, "cv_guard_expand_streak", 0) or 0) == 1:
            args.cv_guard_expand_streak = 0

        print(
            f"[guard] round_schedule_linear round={int(round_idx)} init={int(getattr(args,'cv_guard_init_tasks',0))} "
            f"max={int(getattr(args,'cv_guard_tasks_max',0))} total_samples={int(getattr(args,'cv_guard_total_samples',0))} "
            f"expand_streak={int(getattr(args,'cv_guard_expand_streak',0))}",
            flush=True,
        )

    _write_text(outdir / "run_cmd.txt", " ".join(shlex.quote(x) for x in sys.argv) + "\n")
    _write_json(outdir / "run_meta.json", {"argv": list(sys.argv), "started_at": _now()})

    # Write PID early so external scripts can send SIGUSR1 (save) / SIGTERM (stop).
    _write_text(outdir / "ppo.pid", str(os.getpid()) + "\n")
    _install_signal_handlers()

    if _is_main_process():
        _snapshot_run_code(outdir)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    # Prompt builder shared with eval.
    from eval_dcdc_family import build_prompt

    tasks = default_taskset()
    if not tasks:
        raise SystemExit("no tasks")
    # Pre-shuffle per-family task lists for deterministic but balanced coverage.
    fam_to_tasks: Dict[str, List[Task]] = {}
    for t in tasks:
        fam_to_tasks.setdefault(str(t.family), []).append(t)
    for fam in list(fam_to_tasks.keys()):
        random.shuffle(fam_to_tasks[fam])
    fam_cursors: Dict[str, int] = {fam: 0 for fam in fam_to_tasks}

    key_to_task: Dict[Tuple[str, float, float], Task] = {
        (str(t.family).strip().lower(), float(t.vin), float(t.vout)): t for t in tasks
    }

    cv_guard_enabled = not bool(getattr(args, "no_cv_guard", False))
    guard_max = int(getattr(args, "cv_guard_tasks_max", 0) or 0)
    if guard_max <= 0:
        # No cap: allow guard pool to expand to the full taskset (user requirement).
        guard_max = int(len(tasks))
    guard_pool = _guard_pool_from_taskset(tasks_all=tasks, max_tasks=int(guard_max))
    if not guard_pool:
        guard_pool = list(_guard_tasks_default())
    core_guard = list(_guard_tasks_default())
    core_guard_keys = [(str(f).strip().lower(), float(vin), float(vout)) for (f, vin, vout) in core_guard]
    guard_init_tasks = int(getattr(args, "cv_guard_init_tasks", 0) or 0)
    if int(guard_init_tasks) > 0:
        guard_n_tasks = min(int(len(guard_pool)), int(guard_init_tasks))
    else:
        guard_n_tasks = min(int(len(guard_pool)), int(len(core_guard)))
    guard_streak = 0
    best_guard_cv_for_expand = 1.0
    guard_expand_step = int(getattr(args, "cv_guard_expand_step", 8) or 8)
    guard_expand_streak = int(getattr(args, "cv_guard_expand_streak", 0) or 0)

    hard_task_frac = float(getattr(args, "hard_task_frac", 0.0) or 0.0)
    hard_pass_rate_lt = float(getattr(args, "hard_task_pass_rate_lt", 1.0) or 1.0)
    hard_tasks: List[Task] = []
    hard_fam_to_tasks: Dict[str, List[Task]] = {}
    hard_fam_cursors: Dict[str, int] = {}

    core_task_baseline: Dict[Tuple[str, float, float], float] = {}

    def _guard_n_per_task(n_tasks: int) -> int:
        n_tasks = max(1, int(n_tasks))
        total = int(getattr(args, "cv_guard_total_samples", 0) or 0)
        if total > 0:
            return max(1, int(total) // int(n_tasks))
        return max(1, int(args.cv_guard_n_per_task))

    def _rebuild_hard_maps(htasks: List[Task]) -> None:
        nonlocal hard_tasks, hard_fam_to_tasks, hard_fam_cursors
        hard_tasks = list(htasks)
        hard_fam_to_tasks = {}
        for t in hard_tasks:
            hard_fam_to_tasks.setdefault(str(t.family), []).append(t)
        for fam in list(hard_fam_to_tasks.keys()):
            hard_fam_to_tasks[fam] = sorted(hard_fam_to_tasks[fam], key=lambda x: (float(x.vin), float(x.vout)))
        hard_fam_cursors = {fam: 0 for fam in hard_fam_to_tasks}

    def _update_hard_tasks_from_guard(guard_info: Optional[Dict[str, Any]]) -> None:
        if hard_task_frac <= 0.0:
            return
        stats = (guard_info or {}).get("task_stats") or []
        hard_keys: List[Tuple[str, float, float]] = []
        for ts in stats:
            try:
                pr = float(ts.get("pass_rate", 0.0) or 0.0)
                fam = str(ts.get("family") or "").strip().lower()
                vin = float(ts.get("vin") or 0.0)
                vout = float(ts.get("vout") or 0.0)
            except Exception:
                continue
            if pr + 1e-9 < float(hard_pass_rate_lt):
                hard_keys.append((fam, vin, vout))
        uniq: Dict[Tuple[str, float, float], Task] = {}
        for k in hard_keys:
            t = key_to_task.get((str(k[0]).strip().lower(), float(k[1]), float(k[2])))
            if t is not None:
                uniq[(str(t.family).strip().lower(), float(t.vin), float(t.vout))] = t
        _rebuild_hard_maps(list(uniq.values()))

    tol_levels = [float(x.strip()) for x in str(args.tol_levels).split(",") if x.strip()]
    tol_min_samples = int(args.tol_min_samples)
    if tol_min_samples <= 0:
        tol_min_samples = int(args.tol_window)
    tol_ctl = TolController(
        levels=tol_levels,
        idx=0,
        window=int(args.tol_window),
        hi=float(args.tol_hi),
        lo=float(args.tol_lo),
        min_samples=int(tol_min_samples),
    )
    risk_ctl = RiskController(
        target_fail=float(args.target_fail_rate),
        lambda_lr=float(args.lambda_fail_lr),
        lambda_max=float(args.lambda_fail_max),
        lam=float(args.lambda_fail_init),
        window=int(args.risk_window),
    )
    if not bool(getattr(args, "enable_risk_penalty", False)):
        risk_ctl.lambda_lr = 0.0
        risk_ctl.lambda_max = 0.0
        risk_ctl.lam = 0.0
        risk_ctl.history = []
        print("[init] risk_penalty=disabled", flush=True)
    else:
        print("[init] risk_penalty=enabled", flush=True)

    # Resume logic.
    start_step = 0
    resume_ckpt = None
    s, p = _latest_ckpt(outdir)
    if s is not None and p is not None:
        if not bool(args.resume):
            print("[resume] auto_detected_existing_checkpoint=1 (set --resume to silence this)", flush=True)
        start_step = int(s) + 1
        resume_ckpt = p
        print(f"[resume] {p} (start_step={start_step})", flush=True)
        try:
            st_path = Path(resume_ckpt) / "tol_state.json"
            if st_path.exists():
                st = json.loads(st_path.read_text(encoding="utf-8"))
                # Support both legacy summary-only and full state.
                if isinstance(st.get("levels"), list):
                    tol_ctl.levels = [float(x) for x in st.get("levels") if str(x).strip()]
                tol_ctl.window = int(st.get("window", tol_ctl.window))
                tol_ctl.hi = float(st.get("hi", tol_ctl.hi))
                tol_ctl.lo = float(st.get("lo", tol_ctl.lo))
                tol_ctl.min_samples = int(st.get("min_samples", tol_ctl.min_samples))
                tol_ctl.idx = int(st.get("tol_idx", st.get("idx", tol_ctl.idx)))
                hist = st.get("history", None)
                if isinstance(hist, list):
                    tol_ctl.history = [1 if int(x) else 0 for x in hist][-int(tol_ctl.window) :]
                print(
                    f"[resume] tol_state tol={tol_ctl.tol:.3f} idx={tol_ctl.idx} win={len(tol_ctl.history)}",
                    flush=True,
                )

            rk_path = Path(resume_ckpt) / "risk_state.json"
            if rk_path.exists():
                rs = json.loads(rk_path.read_text(encoding="utf-8"))
                risk_ctl.target_fail = float(rs.get("target_fail", risk_ctl.target_fail))
                risk_ctl.lambda_lr = float(rs.get("lambda_lr", risk_ctl.lambda_lr))
                risk_ctl.lambda_max = float(rs.get("lambda_max", risk_ctl.lambda_max))
                risk_ctl.lam = float(rs.get("lambda_fail", rs.get("lam", risk_ctl.lam)))
                risk_ctl.window = int(rs.get("window", risk_ctl.window))
                rh = rs.get("history", None)
                if isinstance(rh, list):
                    risk_ctl.history = [1 if int(x) else 0 for x in rh][-int(risk_ctl.window) :]
                summ = risk_ctl.summary()
                print(
                    f"[resume] risk_state lambda_fail={summ['lambda_fail']:.3f} fail_win={summ['fail_rate_win']:.3f} win={summ['win']}",
                    flush=True,
                )

            gs_path = Path(resume_ckpt) / "guard_state.json"
            if gs_path.exists():
                gs = json.loads(gs_path.read_text(encoding="utf-8"))
                guard_n_tasks = int(gs.get("guard_n_tasks", guard_n_tasks))
                guard_n_tasks = max(1, min(int(guard_n_tasks), int(len(guard_pool))))
                guard_streak = int(gs.get("guard_streak", guard_streak))
                try:
                    best_guard_cv_for_expand = float(gs.get("best_guard_cv_for_expand", best_guard_cv_for_expand))
                except Exception:
                    best_guard_cv_for_expand = float(best_guard_cv_for_expand)

                core_task_baseline = {}
                core_recs = gs.get("core_tasks") or []
                if isinstance(core_recs, list):
                    for rec in core_recs:
                        try:
                            fam = str(rec.get("family") or "").strip().lower()
                            vin = float(rec.get("vin") or 0.0)
                            vout = float(rec.get("vout") or 0.0)
                            pr = float(rec.get("pass_rate_min", rec.get("pass_rate", 0.0)) or 0.0)
                        except Exception:
                            continue
                        core_task_baseline[(fam, vin, vout)] = pr

                ht_recs = gs.get("hard_tasks") or []
                hard_objs: List[Task] = []
                if isinstance(ht_recs, list):
                    for rec in ht_recs:
                        try:
                            fam = str(rec.get("family") or "").strip().lower()
                            vin = float(rec.get("vin") or 0.0)
                            vout = float(rec.get("vout") or 0.0)
                        except Exception:
                            continue
                        t = key_to_task.get((fam, vin, vout))
                        if t is not None:
                            hard_objs.append(t)
                _rebuild_hard_maps(hard_objs)

                print(
                    f"[resume] guard_state n_tasks={int(guard_n_tasks)} streak={int(guard_streak)} hard={int(len(hard_tasks))}",
                    flush=True,
                )
        except Exception as e:
            print(f"[resume] state_load_failed: {type(e).__name__}: {e}", flush=True)

    # Enforce risk_penalty disabling even when resuming from an older run that stored non-zero lambda.
    if not bool(getattr(args, "enable_risk_penalty", False)):
        risk_ctl.lambda_lr = 0.0
        risk_ctl.lambda_max = 0.0
        risk_ctl.lam = 0.0
        risk_ctl.history = []

    # User requirement: keep tolerance fixed even when resuming from older runs.
    tol_ctl.levels = [0.01]
    tol_ctl.idx = 0

    train_task_pool = str(getattr(args, "train_task_pool", "all") or "all").strip().lower()

    def _build_guard_train_pool(n_tasks: int) -> Tuple[List[Task], Dict[str, List[Task]], Dict[str, int]]:
        n_tasks = max(1, min(int(n_tasks), int(len(guard_pool))))
        keys = list(guard_pool[: int(n_tasks)])
        lst: List[Task] = []
        for k in keys:
            t = key_to_task.get((str(k[0]).strip().lower(), float(k[1]), float(k[2])))
            if t is not None:
                lst.append(t)
        fam_map: Dict[str, List[Task]] = {}
        for t in lst:
            fam_map.setdefault(str(t.family), []).append(t)
        for fam in list(fam_map.keys()):
            fam_map[fam] = sorted(fam_map[fam], key=lambda x: (float(x.vin), float(x.vout)))
        cursors = {fam: 0 for fam in fam_map}
        return lst, fam_map, cursors

    tasks_train: List[Task] = list(tasks)
    fam_to_tasks_train: Dict[str, List[Task]] = dict(fam_to_tasks)
    fam_cursors_train: Dict[str, int] = dict(fam_cursors)
    train_guard_n_tasks: Optional[int] = None
    if train_task_pool == "guard":
        tasks_train, fam_to_tasks_train, fam_cursors_train = _build_guard_train_pool(int(guard_n_tasks))
        train_guard_n_tasks = int(guard_n_tasks)
        print(f"[task] train_task_pool=guard n={len(tasks_train)} guard_n_tasks={int(guard_n_tasks)}", flush=True)
    else:
        print(f"[task] train_task_pool=all n={len(tasks_train)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    # TRL's PPO step pads sequences; left-padding is typically the stable choice for causal LMs during RLHF.
    # We observed large negative "objective/kl" with right-padding which correlates with PPO collapse.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.bos_token_id is None:
        tok.bos_token = tok.eos_token

    rank0, world0 = _dist_rank_world()
    gc_enabled = True
    if int(world0) > 1:
        gc_enabled = False

    model = load_value_head_model(
        args.base_model,
        init_adapter=(args.sft_adapter or "").strip() or None,
        resume_ckpt=resume_ckpt,
        gradient_checkpointing=bool(gc_enabled),
    )

    accel_kwargs = {}
    try:
        rank, world = _dist_rank_world()
    except Exception:
        rank, world = 0, 1
    if int(world) > 1:
        backend = str(getattr(args, "ddp_backend", "") or "").strip() or "nccl"
        accel_kwargs = {
            "kwargs_handlers": [
                InitProcessGroupKwargs(backend=backend),
                DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False),
            ]
        }


    ppo_cfg = PPOConfig(
        seed=int(args.seed),
        steps=int(args.steps),
        learning_rate=float(args.lr),
        batch_size=int(args.batch_size),
        mini_batch_size=int(args.mini_batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        ppo_epochs=int(args.ppo_epochs),
        cliprange=float(args.cliprange),
        cliprange_value=float(args.cliprange_value),
        vf_coef=float(args.vf_coef),
        target_kl=float(args.target_kl),
        log_with=None,
        tracker_project_name="dcdc_family",
        model_name=str(args.base_model),
        accelerator_kwargs=accel_kwargs,
        )

    ref_model = None
    # NOTE: When using PEFT/LoRA, TRL computes the KL reference pass by temporarily disabling adapters
    # (see PPOTrainer.optional_peft_ctx -> disable_adapter). In that case, cloning a separate ref_model
    # is both unnecessary (it is not used) and can OOM for large backbones (e.g., Qwen2.5-14B).
    model_is_peft = bool(getattr(model, "is_peft_model", False))
    try:  # best-effort additional detection
        from peft import PeftModel  # type: ignore

        if not model_is_peft:
            model_is_peft = isinstance(getattr(model, "pretrained_model", None), PeftModel)
    except Exception:
        pass

    if model_is_peft:
        print("[init] peft_model=True; skip ref_model clone (KL uses disable_adapter)", flush=True)
    elif not bool(args.no_ref_model):
        try:
            from trl.models import create_reference_model
        except Exception:  # pragma: no cover
            from trl import create_reference_model  # type: ignore
        try:
            ref_model = create_reference_model(model)
            print("[init] ref_model=enabled", flush=True)
        except Exception as e:
            raise SystemExit(f"Failed to create ref_model: {type(e).__name__}: {e}")
    else:
        print("[init] ref_model=disabled", flush=True)

    trainer = PPOTrainerWithEntropy(
        config=ppo_cfg,
        model=model,
        ref_model=ref_model,
        tokenizer=tok,
        dataset=None,
        ent_coef=float(args.ent_coef),
    )
    if bool(getattr(args, "debug_dist", False)) and _is_main_process():
        try:
            import torch.distributed as dist
            print(f"[dist] backend={dist.get_backend()} accel_state_backend={getattr(getattr(trainer, 'accelerator', None), 'state', None) and trainer.accelerator.state.backend} accel_kwargs={ppo_cfg.accelerator_kwargs}", flush=True)
        except Exception as e:
            print(f"[dist] debug failed: {type(e).__name__}: {e}", flush=True)

    if int(world0) <= 1:
        _sanity_check_trainable_grads(trainer.model, tok)

    logits_proc = None
    if bool(args.constrained):
        try:
            from integrated.constraints import CharClassLogitsProcessor

            logits_proc = [CharClassLogitsProcessor(tok, penalty=30.0)]
        except Exception as e:
            raise SystemExit(f"Missing CharClassLogitsProcessor: {type(e).__name__}: {e}")

    if int(world0) <= 1:
        _sanity_generate_report(
        trainer=trainer,
        tok=tok,
        build_prompt=build_prompt,
        logits_proc=logits_proc,
        outdir=outdir,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )

    csv_path = outdir / "ppo_metrics.csv"
    jsonl_path = outdir / "ppo_metrics.jsonl"
    rollout_path = outdir / "ppo_rollouts.jsonl"

    def _prune_ckpts() -> None:
        limit = int(args.save_total_limit)
        if limit <= 0:
            return
        cks = []
        for p in outdir.glob("ppo_step_*"):
            if not p.is_dir():
                continue
            m = re.match(r"^ppo_step_(\d+)$", p.name)
            if not m:
                continue
            cks.append((int(m.group(1)), p))
        cks.sort(key=lambda x: x[0])
        if len(cks) <= limit:
            return
        for _, p in cks[: max(0, len(cks) - limit)]:
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass

    csv_fields = [
        "step",
        "tol",
        "tol_idx",
        "tol_next",
        "tol_idx_next",
        "cv_rate_win",
        "w_cv",
        "w_ce",
        "w_ripple",
        "w_over",
        "reward_raw_mean",
        "reward_used_mean",
        "reward_rule_mean",
        "rule_score_mean",
        "reward_cv_mean",
        "reward_ce_mean",
        "pen_ripple_mean",
        "pen_over_mean",
        "pen_size_mean",
        "pen_fail_mean",
        "ok_rate",
        "cv_rate",
        "ce_rate",
        "min_elems_rate",
        "ok_strict_rate",
        "cv_strict_rate",
        "ce_strict_rate",
        "n_elems_mean",
        "strict_success_rate",
        "guard/cv_rate",
        "guard/cv_min_family",
        "guard/ce_rate",
        "guard/accepted",
        "best_guard/n_tasks",
        "best_guard/cv_rate",
        "best_guard/ce_rate",
        "best_guard/cv_min_family",
        "best_guard/ce_min_family",
        "risk/lambda_fail",
        "risk/fail_rate_win",
        "risk/success_rate_win",
        "risk/target_fail",
        "group/mode",
        "group/rel_coef",
        # TRL stats (keyed by record_step_stats)
        "objective/kl",
        "objective/entropy",
        "objective/kl_coef",
        "ppo/mean_scores",
        "ppo/std_scores",
        "ppo/loss/total",
        "ppo/loss/policy",
        "ppo/loss/value",
        "ppo/loss/entropy_bonus",
        "ppo/policy/entropy",
        "ppo/policy/approxkl",
        "ppo/policy/policykl",
        "ppo/policy/clipfrac",
        "ppo/policy/advantages_mean",
        "ppo/val/vpred",
        "ppo/val/error",
        "ppo/val/clipfrac",
        "ppo/val/var_explained",
    ]

    def _ensure_csv_header() -> None:
        if not _is_main_process():
            return
        expected = ",".join(csv_fields)
        if csv_path.exists():
            try:
                with csv_path.open("r", encoding="utf-8") as f:
                    first = f.readline().strip()
                if first == expected:
                    return
            except Exception:
                pass
            try:
                bak = csv_path.with_suffix(csv_path.suffix + f".bak_{_now()}")
                csv_path.rename(bak)
            except Exception:
                pass
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            w.writeheader()

    _ensure_csv_header()

    def _csv_write_header() -> None:
        if not _is_main_process():
            return
        if csv_path.exists():
            return
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            w.writeheader()

    _csv_write_header()

    # Keep a "best checkpoint" independent of --save_total_limit so we never lose the best-performing weights
    # even if later PPO collapses.
    # User requirement:
    # - Guard expansion is triggered by CORE-GUARD (cv_rate==1.0 & ce_rate>thr).
    # - After expansion, accept new points once guard CV>=min_cv, then only update best when
    #   guard CV improves, or guard CV ties but GUARD CE improves.
    # Rank checkpoints by (guard_n_tasks, guard_cv_rate, guard_ce_rate, guard_cv_min_family, reward_raw_mean).
    best_key: Tuple[float, float, float, float, float] = (-1.0, -1.0, -1.0, -1.0, -1e30)
    best_step: int = -1


    # Restore best_key/best_step from an existing ppo_best (important for restart/resume).
    try:
        _best_meta_path = outdir / "ppo_best" / "best_meta.json"
        if _best_meta_path.exists():
            _meta = json.loads(_best_meta_path.read_text(encoding="utf-8"))
            _step0 = int(_meta.get("step", -1))
            _ck = _meta.get("cand_key") or []

            _ce0 = None
            if _step0 >= 0 and csv_path.exists():
                try:
                    with csv_path.open("r", encoding="utf-8") as f:
                        for _row in csv.DictReader(f):
                            try:
                                if int(_row.get("step", -1)) == int(_step0):
                                    _ce0 = float(_row.get("ce_rate", -1.0) or -1.0)
                                    break
                            except Exception:
                                continue
                except Exception:
                    _ce0 = None

            if isinstance(_ck, list) and len(_ck) >= 5:
                best_key = (
                    float(_ck[0]),
                    float(_ck[1]),
                    float(_ck[2]),
                    float(_ck[3]),
                    float(_ck[4]),
                )
                best_step = int(_step0)
            elif isinstance(_ck, list) and len(_ck) == 4:
                # legacy cand_key: (guard_n_tasks, guard_cv_min_family, guard_cv_rate, reward_raw_mean)
                _n_tasks = float(_ck[0])
                _minfam = float(_ck[1])
                _gcv = float(_ck[2])
                _reward = float(_ck[3])
                _ce = float(_ce0) if _ce0 is not None else -1.0
                best_key = (_n_tasks, _gcv, _ce, _minfam, _reward)
                best_step = int(_step0)

            if int(best_step) >= 0:
                print(f"[best] restored best_step={int(best_step)} best_key={best_key}", flush=True)
    except Exception as e:
        print(f"[best] restore_failed: {type(e).__name__}: {e}", flush=True)


    # Rolling CV window for convergence detection (only tracked once tol is tight).
    conv_cv_hist: List[float] = []
    conv_triggered: bool = False

    def _save_best_checkpoint(
        *,
        step: int,
        cand_key: Tuple[float, float, float, float, float],
        reward_raw_mean: float,
        strict_success_rate: float,
        batch_ce_rate: float,
        cv_guard: Optional[Dict[str, Any]],
        tol_state: Dict[str, Any],
        risk_state: Dict[str, Any],
        guard_state: Dict[str, Any],
    ) -> None:
        nonlocal best_key, best_step
        if not _is_main_process():
            return
        if tuple(float(x) for x in cand_key) <= best_key:
            return

        best_key = tuple(float(x) for x in cand_key)
        best_step = int(step)

        best_dir = outdir / "ppo_best"
        tmp_dir = outdir / "ppo_best_tmp"
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if best_dir.exists():
                shutil.rmtree(best_dir, ignore_errors=True)
        except Exception:
            pass

        _mkdir(tmp_dir)
        _unwrap_trainer_model(trainer).save_pretrained(str(tmp_dir))
        (tmp_dir / "ppo_config.json").write_text(
            json.dumps(ppo_cfg.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (tmp_dir / "tol_state.json").write_text(json.dumps(tol_state, ensure_ascii=False, indent=2), encoding="utf-8")
        (tmp_dir / "risk_state.json").write_text(json.dumps(risk_state, ensure_ascii=False, indent=2), encoding="utf-8")
        (tmp_dir / "guard_state.json").write_text(
            json.dumps(guard_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (tmp_dir / "best_meta.json").write_text(
            json.dumps(
                {
                    "step": int(step),
                    "cand_key": [float(x) for x in cand_key],
                    "guard_n_tasks": int(cand_key[0]),
                    "guard_cv_rate": float(cand_key[1]),
                    "guard_ce_rate": float(cand_key[2]),
                    "batch_ce_rate": float(batch_ce_rate),
                    "guard_cv_min_family": float(cand_key[3]),
                    "strict_success_rate": float(strict_success_rate),
                    "reward_raw_mean": float(reward_raw_mean),
                    "cv_guard": (cv_guard or {}),
                    "saved_at": _now(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        try:
            tmp_dir.rename(best_dir)
        except Exception:
            # Fallback to copy if rename fails across filesystems.
            shutil.copytree(tmp_dir, best_dir, dirs_exist_ok=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)

        print(
            f"[best] step={int(step)} strict_success={float(strict_success_rate):.3f} reward_raw={float(reward_raw_mean):.3f}",
            flush=True,
        )

    # main PPO loop
    cv_guard_best_minfam: float = -1.0
    cv_guard_best_rate: float = -1.0
    cv_guard_best_by_family: Dict[str, float] = {}

    if cv_guard_enabled:
        guard_tasks0 = list(guard_pool[: int(guard_n_tasks)])
        npt0 = _guard_n_per_task(int(len(guard_tasks0)))
        print(
            f"[guard] running baseline mini-eval (fixed tol=0.01, tasks={int(len(guard_tasks0))}, n_per_task={int(npt0)})...",
            flush=True,
        )
        guard0 = _guard_eval(
            trainer=trainer,
            tok=tok,
            build_prompt=build_prompt,
            logits_proc=logits_proc,
            tol=float(tol_ctl.tol),
            min_elems=int(args.min_elems),
            n_per_task=int(npt0),
            seed=int(args.cv_guard_seed),
            t_pre=float(args.t_pre),
            t_win=float(args.t_win),
            sim_timeout_s=float(args.sim_timeout_s),
            autotune_duty=bool(args.autotune_duty),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            constrained=bool(args.constrained),
            sim_executor=sim_executor,
            tasks=guard_tasks0,
        )
        if _is_main_process():
            (outdir / "logs" / "cv_guard_baseline.json").write_text(
                json.dumps(guard0, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (outdir / "logs" / "cv_guard_tasks.json").write_text(
                json.dumps(
                    [
                        {"family": str(f).strip().lower(), "vin": float(vin), "vout": float(vout)}
                        for (f, vin, vout) in guard_tasks0
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        cv_guard_best_by_family = {
            str(k).strip().lower(): float(v) for k, v in (guard0.get("cv_by_family") or {}).items()
        }
        cv_guard_best_rate = float(guard0.get("cv_rate", 0.0))
        cv_guard_best_minfam = (
            float(min(cv_guard_best_by_family.values()))
            if cv_guard_best_by_family
            else float(guard0.get("cv_min_family", 0.0))
        )

        stat_map = {}
        for rec in (guard0.get("task_stats") or []):
            try:
                k = (str(rec.get("family") or "").strip().lower(), float(rec.get("vin") or 0.0), float(rec.get("vout") or 0.0))
                stat_map[k] = float(rec.get("pass_rate", 0.0) or 0.0)
            except Exception:
                continue
        core_task_baseline = {k: float(stat_map.get(k, 0.0)) for k in core_guard_keys}
        if _is_main_process():
            (outdir / "logs" / "cv_guard_core_baseline.json").write_text(
                json.dumps(
                    [
                        {"family": k[0], "vin": float(k[1]), "vout": float(k[2]), "pass_rate_min": float(v)}
                        for k, v in core_task_baseline.items()
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        _update_hard_tasks_from_guard(guard0)

        guard_state0 = {
            "guard_n_tasks": int(guard_n_tasks),
            "guard_tasks_max": int(len(guard_pool)),
            "guard_streak": int(guard_streak),
            "core_tasks": [
                {"family": k[0], "vin": float(k[1]), "vout": float(k[2]), "pass_rate_min": float(v)}
                for k, v in core_task_baseline.items()
            ],
            "hard_tasks": [
                {"family": str(t.family).strip().lower(), "vin": float(t.vin), "vout": float(t.vout)}
                for t in hard_tasks
            ],
        }
        # Seed "ppo_best" with the initial (SFT) policy so we never regress below it.
        tol_state0 = tol_ctl.summary().copy()
        tol_state0.update(
            {
                "levels": list(tol_ctl.levels),
                "window": int(tol_ctl.window),
                "hi": float(tol_ctl.hi),
                "lo": float(tol_ctl.lo),
                "min_samples": int(tol_ctl.min_samples),
                "history": list(tol_ctl.history),
            }
        )
        risk_state0 = risk_ctl.summary().copy()
        risk_state0.update(
            {
                "lambda_lr": float(risk_ctl.lambda_lr),
                "lambda_max": float(risk_ctl.lambda_max),
                "window": int(risk_ctl.window),
                "history": list(risk_ctl.history),
            }
        )
        # Seed ppo_best with the initial (SFT/resumed) policy only when ppo_best does not exist yet.
        # When resuming, we must NOT overwrite the best checkpoint accumulated so far.
        if not (outdir / 'ppo_best' / 'best_meta.json').exists():
            _save_best_checkpoint(
                step=-1,
                cand_key=(float(guard_n_tasks), float(cv_guard_best_rate), -1.0, float(cv_guard_best_minfam), -1e30),
                reward_raw_mean=-1e30,
                strict_success_rate=0.0,
                batch_ce_rate=-1.0,
                cv_guard=guard0,
                tol_state=tol_state0,
                risk_state=risk_state0,
                guard_state=guard_state0,
            )
        else:
            print('[best] existing ppo_best found; skip seeding', flush=True)

    last_eff_group_size: Optional[int] = None

    for step in range(int(start_step), int(args.steps)):
        tol_used = float(tol_ctl.tol)
        tol_idx_used = int(tol_ctl.idx)
        weights = anneal_weights(step=step, total_steps=int(anneal_total_steps))

        # sample tasks
        bs = int(args.batch_size)
        requested_group_size = max(1, int(args.group_size))
        min_groups_per_step = int(getattr(args, "min_groups_per_step", 0) or 0)
        group_size = _auto_group_size(
            batch_size=int(bs),
            requested_group_size=int(requested_group_size),
            min_groups_per_step=int(min_groups_per_step),
        )
        if (last_eff_group_size is None) or (int(group_size) != int(last_eff_group_size)):
            if int(group_size) != int(requested_group_size):
                print(
                    f"[task] auto group_size {int(requested_group_size)}->{int(group_size)} "
                    f"(n_groups={int(bs)//int(group_size)}) min_groups_per_step={int(min_groups_per_step)}",
                    flush=True,
                )
            last_eff_group_size = int(group_size)
        if bs % group_size != 0:
            raise SystemExit(f"--batch_size ({bs}) must be divisible by --group_size ({group_size})")
        n_groups = bs // group_size
        mode = str(getattr(args, "batch_task_mode", "same_family") or "same_family").strip().lower()

        # Optionally focus PPO rollouts on the (expanded) guard pool for faster CV improvements.
        if train_task_pool == "guard" and (train_guard_n_tasks is None or int(train_guard_n_tasks) != int(guard_n_tasks)):
            tasks_train, fam_to_tasks_train, fam_cursors_train = _build_guard_train_pool(int(guard_n_tasks))
            train_guard_n_tasks = int(guard_n_tasks)
            print(f"[task] train_task_pool=guard updated n={len(tasks_train)} guard_n_tasks={int(guard_n_tasks)}", flush=True)

        if mode == "same_task":
            t0 = random.choice(tasks_train)
            group_tasks: List[Task] = [t0 for _ in range(n_groups)]
        elif mode == "same_family":
            fams = sorted({str(t.family) for t in tasks_train})
            fam = random.choice(fams) if fams else ""
            fam_tasks = fam_to_tasks_train.get(str(fam)) or list(tasks_train)
            if hard_task_frac > 0.0:
                hard_lst = hard_fam_to_tasks.get(str(fam), None)
                if hard_lst:
                    # Mix in mined hard tasks for the chosen family.
                    if random.random() < float(hard_task_frac):
                        fam_tasks = list(hard_lst)
            group_tasks = [random.choice(fam_tasks) for _ in range(n_groups)]
        elif mode == "balanced_families":
            fams = sorted(fam_to_tasks_train.keys())
            if not fams:
                group_tasks = [random.choice(tasks_train) for _ in range(n_groups)]
            else:
                # Round-robin families across groups AND steps.
                group_tasks = []
                for gi in range(n_groups):
                    fam = fams[(int(step) + int(gi)) % int(len(fams))]
                    lst = fam_to_tasks_train.get(fam) or list(tasks_train)
                    cur = int(fam_cursors_train.get(fam, 0))
                    use_hard = False
                    if hard_task_frac > 0.0:
                        hard_lst = hard_fam_to_tasks.get(str(fam), None)
                        if hard_lst and (random.random() < float(hard_task_frac)):
                            lst = list(hard_lst)
                            cur = int(hard_fam_cursors.get(str(fam), 0))
                            use_hard = True
                    t = lst[cur % len(lst)]
                    if use_hard:
                        hard_fam_cursors[str(fam)] = cur + 1
                    else:
                        fam_cursors_train[fam] = cur + 1
                    group_tasks.append(t)
        else:
            group_tasks = [random.choice(tasks_train) for _ in range(n_groups)]
        batch_tasks: List[Task] = []
        group_ids: List[int] = []
        for gi, t in enumerate(group_tasks):
            for _ in range(group_size):
                batch_tasks.append(t)
                group_ids.append(int(gi))
        prompts = [build_prompt(t.family, float(t.vin), float(t.vout)) for t in batch_tasks]
        query_tensors = [tok(p, return_tensors="pt").input_ids[0] for p in prompts]

        # TRL batches generation with left-padding; for some LMs this can catastrophically break outputs when mixing
        # different prompt lengths (even within the same family). To keep generation stable, generate per-group (each
        # group has identical prompts) and then concatenate.
        # Speed fix: enable KV-cache during sampling (generation is inference-only), then disable for PPO backward.
        was_train = bool(getattr(trainer.model, 'training', False))
        trainer.model.eval()
        try:
            trainer.model.pretrained_model.config.use_cache = True
        except Exception:
            pass

        responses: List[torch.Tensor] = []
        for gi in range(n_groups):
            start = int(gi) * int(group_size)
            end = start + int(group_size)
            qt = query_tensors[start:end]
            resp_g = trainer.generate(
                qt,
                batch_size=int(group_size),
                return_prompt=False,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                logits_processor=logits_proc,
            )
            responses.extend(resp_g)


        try:
            trainer.model.pretrained_model.config.use_cache = False
        except Exception:
            pass
        if was_train:
            trainer.model.train()

        details: List[dict] = []
        rewards: List[float] = []
        pass_cv_flags: List[bool] = []
        pass_ce_flags: List[bool] = []
        ok_flags: List[bool] = []
        min_elems_flags: List[bool] = []
        reward_parts_sum = {
            "reward_rule": 0.0,
            "rule_score": 0.0,
            "reward_cv": 0.0,
            "reward_ce": 0.0,
            "pen_ripple": 0.0,
            "pen_over": 0.0,
            "pen_size": 0.0,
            "pen_fail": 0.0,
        }
        n_elems_sum = 0.0

        # simulate each sample (CPU-bound) with parallel ngspice
        sample_infos: List[Dict[str, Any]] = []
        fut_to_i: Dict[Any, int] = {}
        for bi, (t, resp_ids) in enumerate(zip(batch_tasks, responses)):
            if isinstance(resp_ids, torch.Tensor):
                txt = tok.decode(resp_ids.detach().cpu().tolist(), skip_special_tokens=True)
            else:
                txt = tok.decode(list(resp_ids), skip_special_tokens=True)
            inc_lines = extract_inc_lines(txt)
            inc = ("\n".join(inc_lines).strip() + "\n") if inc_lines else ""

            ver = verify_inc_dcdc(inc, family=t.family, vin=float(t.vin), vout=float(t.vout))
            n_elems = int(ver.n_elems)
            meets_min = n_elems >= int(args.min_elems)

            info: Dict[str, Any] = {
                "task": t,
                "txt": txt,
                "inc": inc,
                "ver": ver,
                "n_elems": int(n_elems),
                "meets_min": bool(meets_min),
                "prompt": prompts[bi],
                "detail": None,
            }

            if not bool(ver.ok):
                # Avoid wasting ngspice on obviously invalid designs; use verifier-only detail.
                info["detail"] = {
                    "ok": False,
                    "pass_C": False,
                    "pass_CV": False,
                    "pass_CE": False,
                    "eff": 0.0,
                    "vavg": 0.0,
                    "ripple": 0.0,
                    "overshoot": 0.0,
                    "violations": list(ver.violations),
                    "canonical_hash": str(ver.canonical_hash),
                }
            else:
                if sim_executor is None:
                    info["detail"] = eval_one_detail_dcdc(
                        inc=inc,
                        family=t.family,
                        vin=float(t.vin),
                        vout=float(t.vout),
                        tol=float(tol_used),
                        rload=10.0,
                        t_pre=float(args.t_pre),
                        t_win=float(args.t_win),
                        sim_timeout_s=float(args.sim_timeout_s),
                        autotune_duty=bool(args.autotune_duty),
                    )
                else:
                    fut = sim_executor.submit(
                        eval_one_detail_dcdc,
                        inc=inc,
                        family=t.family,
                        vin=float(t.vin),
                        vout=float(t.vout),
                        tol=float(tol_used),
                        rload=10.0,
                        t_pre=float(args.t_pre),
                        t_win=float(args.t_win),
                        sim_timeout_s=float(args.sim_timeout_s),
                        autotune_duty=bool(args.autotune_duty),
                    )
                    fut_to_i[fut] = int(bi)

            sample_infos.append(info)

        for fut in as_completed(fut_to_i):
            bi = fut_to_i[fut]
            try:
                detail = fut.result()
            except Exception:
                ver = sample_infos[bi].get("ver")
                detail = {
                    "ok": False,
                    "pass_C": False,
                    "pass_CV": False,
                    "pass_CE": False,
                    "eff": 0.0,
                    "vavg": 0.0,
                    "ripple": 0.0,
                    "overshoot": 0.0,
                    "violations": list(getattr(ver, "violations", []) or []),
                    "canonical_hash": str(getattr(ver, "canonical_hash", "")),
                    "error": "eval_exception",
                }
            sample_infos[bi]["detail"] = detail

        for bi, info in enumerate(sample_infos):
            t = info["task"]
            txt = str(info.get("txt") or "")
            inc = str(info.get("inc") or "")
            ver = info.get("ver")
            n_elems = int(info.get("n_elems") or 0)
            meets_min = bool(info.get("meets_min"))
            detail = info.get("detail") or {"ok": False, "pass_CV": False, "pass_CE": False}

            reward_total, parts = compute_reward(
                detail,
                family=str(t.family),
                vout=float(t.vout),
                tol=float(tol_used),
                weights=weights,
                min_elems=int(args.min_elems),
                n_elems=n_elems,
                ver_ok=bool(getattr(ver, "ok", False)),
                violations=list(getattr(ver, "violations", []) or []),
            )

            ok_flags.append(bool(detail.get("ok", False)) and bool(getattr(ver, "ok", False)))
            pass_cv_flags.append(bool(detail.get("pass_CV", False)))
            pass_ce_flags.append(bool(detail.get("pass_CE", False)))
            min_elems_flags.append(bool(meets_min))
            rewards.append(float(reward_total))
            for k in reward_parts_sum:
                reward_parts_sum[k] += float(parts.get(k, 0.0))
            n_elems_sum += float(n_elems)

            rec = {
                "step": int(step),
                "family": str(t.family),
                "vin": float(t.vin),
                "vout": float(t.vout),
                "tol": float(tol_used),
                "weights": {
                    "w_cv": float(weights.w_cv),
                    "w_ce": float(weights.w_ce),
                    "w_ripple": float(weights.w_ripple),
                    "w_over": float(weights.w_over),
                },
                "n_elems": int(n_elems),
                "n_inc_lines": int(getattr(ver, "n_inc_lines", 0) if ver is not None else 0),
                "violations": list(getattr(ver, "violations", []) or []),
                "reward": float(reward_total),
                "reward_parts": parts,
                "detail": detail,
                "inc_source": inc.strip(),
                "prompt": info.get("prompt") or prompts[bi],
                "raw_text": txt[:4000],
            }
            details.append(rec)

        # Snapshot controllers so we can rollback if the CV-guard rejects this PPO update.
        tol_ctl_snap = copy.deepcopy(tol_ctl) if cv_guard_enabled else None
        risk_ctl_snap = copy.deepcopy(risk_ctl) if cv_guard_enabled else None

        # adaptive tol update (kept for logging; tol is forced-fixed to 0.01 via tol_levels=[0.01])
        tol_ctl.update(pass_cv_flags)
        if (int(step) + 1) % int(args.tol_adjust_every) == 0:
            tol_ctl.maybe_adjust()

        strict_success_flags = [
            bool(ok and cv and mn) for ok, cv, mn in zip(ok_flags, pass_cv_flags, min_elems_flags)
        ]

        # Risk-sensitive / constrained controller: update lambda_fail based on strict success rate.
        risk_ctl.update(strict_success_flags)
        if (int(step) + 1) % int(args.risk_adjust_every) == 0:
            risk_ctl.maybe_adjust()
        risk_penalties = [risk_ctl.penalty(s) for s in strict_success_flags]

        rewards_used = rewards
        group_mode = str(getattr(args, "group_reward_mode", "none") or "none").strip().lower()
        group_rel_terms = [0.0 for _ in rewards]
        if group_size > 1 and group_mode != "none":
            if group_mode == "pareto_add":
                rewards_used, group_rel_terms = _group_pareto_add_rewards(
                    raw_rewards=rewards,
                    details=details,
                    group_ids=group_ids,
                    strict_success_flags=strict_success_flags,
                    rel_coef=float(args.group_rel_coef),
                    min_elems=int(args.min_elems),
                )
            else:
                rewards_used = _group_normalize_rewards(
                    rewards,
                    group_ids,
                    mode=group_mode,
                    rel_coef=float(args.group_rel_coef),
                )

        # Apply per-sample failure penalty (Lagrangian): reward -= lambda_fail * 1[failure]
        if float(risk_ctl.lam) > 0.0:
            rewards_used = [float(r) - float(p) for r, p in zip(rewards_used, risk_penalties)]

        # Safety: NEVER allow failures to become positive due to within-group normalization.
        # Keep absolute (raw) rewards for failures; only apply group-relative shaping to strict successes.
        rewards_used = [
            (float(r_raw) - float(pen) if not bool(ss) else float(r_used))
            for r_raw, r_used, pen, ss in zip(rewards, rewards_used, risk_penalties, strict_success_flags)
        ]

        for rec, ru, rp, ss, gr in zip(details, rewards_used, risk_penalties, strict_success_flags, group_rel_terms):
            rec["reward_used"] = float(ru)
            rec["risk_penalty"] = float(rp)
            rec["strict_success"] = bool(ss)
            rec["group_rel"] = float(gr)

        # PPO step (uses rewards_used) + CV non-regression guard.
        trainable_snap = _snapshot_trainable_params(trainer.model) if cv_guard_enabled else {}
        opt_snap = _snapshot_optimizer_state(getattr(trainer, "optimizer", None)) if cv_guard_enabled else {}

        score_tensors = [torch.tensor(float(x), dtype=torch.float32) for x in rewards_used]
        stats = trainer.step(query_tensors, responses, score_tensors)

        # aggregate logging
        bs = max(1, int(len(rewards)))
        reward_raw_mean = float(sum(rewards) / bs)
        reward_used_mean = float(sum(float(x) for x in rewards_used) / bs)
        parts_mean = {k: float(v / bs) for k, v in reward_parts_sum.items()}
        ok_rate = float(sum(1 for x in ok_flags if x) / bs)
        cv_rate = float(sum(1 for x in pass_cv_flags if x) / bs)
        ce_rate = float(sum(1 for x in pass_ce_flags if x) / bs)
        min_elems_rate = float(sum(1 for x in min_elems_flags if x) / bs)
        ok_strict_rate = float(sum(1 for ok, mn in zip(ok_flags, min_elems_flags) if ok and mn) / bs)
        cv_strict_rate = float(sum(1 for cv, mn in zip(pass_cv_flags, min_elems_flags) if cv and mn) / bs)
        ce_strict_rate = float(sum(1 for ce, mn in zip(pass_ce_flags, min_elems_flags) if ce and mn) / bs)
        n_elems_mean = float(n_elems_sum / bs)
        strict_success_rate = float(sum(1 for x in strict_success_flags if x) / bs)
        risk_info = risk_ctl.summary()

        guard_info: Optional[Dict[str, Any]] = None  # CORE guard (for rollback)
        best_guard_info: Optional[Dict[str, Any]] = None  # expanded guard (for best selection)
        guard_accepted: bool = True
        if cv_guard_enabled and int(args.cv_guard_every) > 0 and ((int(step) + 1) % int(args.cv_guard_every) == 0):
            # (1) CORE guard: non-regression/rollback only looks at the fixed core task set.
            core_n = int(len(core_guard_keys))
            core_tasks = list(guard_pool[: int(core_n)])
            npt_core = _guard_n_per_task(int(len(core_tasks)))
            core_guard = _guard_eval(
                trainer=trainer,
                tok=tok,
                build_prompt=build_prompt,
                logits_proc=logits_proc,
                tol=float(tol_used),
                min_elems=int(args.min_elems),
                n_per_task=int(npt_core),
                seed=int(args.cv_guard_seed),
                t_pre=float(args.t_pre),
                t_win=float(args.t_win),
                sim_timeout_s=float(args.sim_timeout_s),
                autotune_duty=bool(args.autotune_duty),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                constrained=bool(args.constrained),
                sim_executor=sim_executor,
                tasks=core_tasks,
            )
            guard_info = core_guard
            g_rate = float(core_guard.get("cv_rate", 0.0))
            g_ce = float(core_guard.get("ce_rate", 0.0))
            g_minfam = float(core_guard.get("cv_min_family", 0.0))
            g_by_family = {str(k).strip().lower(): float(v) for k, v in (core_guard.get("cv_by_family") or {}).items()}

            # Mine hard tasks from CORE guard even if this step is rejected (helps prevent getting stuck).
            _update_hard_tasks_from_guard(core_guard)

            task_rates = {}
            for rec in (core_guard.get("task_stats") or []):
                try:
                    k = (str(rec.get("family") or "").strip().lower(), float(rec.get("vin") or 0.0), float(rec.get("vout") or 0.0))
                    task_rates[k] = float(rec.get("pass_rate", 0.0) or 0.0)
                except Exception:
                    continue

            degraded = []
            for k, base_pr in core_task_baseline.items():
                cur_pr = float(task_rates.get(k, 0.0) or 0.0)
                if cur_pr + 1e-9 < float(base_pr):
                    degraded.append(f"core:{k[0]}:{k[1]:.1f}->{k[2]:.1f}")
            for fam, best_v in cv_guard_best_by_family.items():
                cur_v = float(g_by_family.get(fam, 0.0))
                if cur_v + 1e-9 < float(best_v):
                    degraded.append(fam)
            if g_rate + 1e-9 < float(cv_guard_best_rate):
                degraded.append("__overall__")

            if degraded:
                guard_accepted = False
                guard_streak = 0
                print(
                    f"[guard] REJECT step={int(step)} degraded={degraded} core_cv={g_rate:.3f} best={float(cv_guard_best_rate):.3f} n_tasks_core={int(len(core_tasks))}; rollback",
                    flush=True,
                )
                _restore_trainable_params(trainer.model, trainable_snap)
                _restore_optimizer_state(getattr(trainer, "optimizer", None), opt_snap)
                if tol_ctl_snap is not None:
                    tol_ctl = tol_ctl_snap
                if risk_ctl_snap is not None:
                    risk_ctl = risk_ctl_snap
            else:
                for fam, cur_v in g_by_family.items():
                    prev = float(cv_guard_best_by_family.get(fam, -1.0))
                    if cur_v > prev + 1e-9:
                        cv_guard_best_by_family[fam] = float(cur_v)
                if g_rate > float(cv_guard_best_rate) + 1e-9:
                    cv_guard_best_rate = float(g_rate)
                cv_guard_best_minfam = (
                    float(min(cv_guard_best_by_family.values())) if cv_guard_best_by_family else float(g_minfam)
                )

            # (2) Adaptive guard expansion trigger uses CORE-GUARD (not training batch) to reduce false expands.
            if bool(guard_accepted):
                expand_cv_thr = float(getattr(args, "cv_guard_expand_trigger_cv", 1.0) or 1.0)
                expand_ce_thr = float(getattr(args, "cv_guard_expand_ce_threshold", 0.8) or 0.8)
                min_expand_cv = float(getattr(args, "cv_guard_expand_min_cv", 0.0) or 0.0)
                expand_ready = (float(g_rate) + 1e-9 >= float(expand_cv_thr)) and (float(g_ce) > float(expand_ce_thr))
                # Prevent runaway expansion: once guard is expanded, only expand again after the expanded-guard CV recovers
                # to min_expand_cv (otherwise we expand every step due to the core guard being relatively easy).
                if int(guard_n_tasks) > int(core_n) and (float(best_guard_cv_for_expand) + 1e-9 < float(min_expand_cv)):
                    expand_ready = False
                guard_streak = int(guard_streak) + 1 if bool(expand_ready) else 0

                if (
                    int(guard_expand_streak) > 0
                    and int(guard_streak) >= int(guard_expand_streak)
                    and int(guard_n_tasks) < int(len(guard_pool))
                ):
                    old_n = int(guard_n_tasks)
                    new_n = min(int(len(guard_pool)), int(old_n) + int(max(1, guard_expand_step)))
                    guard_streak = 0
                    new_tasks = list(guard_pool[: int(new_n)])
                    npt2 = _guard_n_per_task(int(len(new_tasks)))
                    print(
                        f"[guard] EXPAND old={old_n} -> new={int(new_n)} (n_per_task={int(npt2)})",
                        flush=True,
                    )
                    expand_info = _guard_eval(
                        trainer=trainer,
                        tok=tok,
                        build_prompt=build_prompt,
                        logits_proc=logits_proc,
                        tol=float(tol_used),
                        min_elems=int(args.min_elems),
                        n_per_task=int(npt2),
                        seed=int(args.cv_guard_seed),
                        t_pre=float(args.t_pre),
                        t_win=float(args.t_win),
                        sim_timeout_s=float(args.sim_timeout_s),
                        autotune_duty=bool(args.autotune_duty),
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        constrained=bool(args.constrained),
                        sim_executor=sim_executor,
                        tasks=new_tasks,
                    )
                    e_rate = float(expand_info.get("cv_rate", 0.0))
                    e_ce = float(expand_info.get("ce_rate", 0.0))
                    min_expand_cv = float(getattr(args, "cv_guard_expand_min_cv", 0.0) or 0.0)
                    met = bool(e_rate + 1e-9 >= float(min_expand_cv))
                    # Commit expansion immediately (no shrink-back). min_expand_cv only gates ppo_best update.
                    guard_n_tasks = int(new_n)
                    best_guard_info = expand_info
                    best_guard_cv_for_expand = float(e_rate)

                    # Mine hard tasks from the expanded guard set after commit.
                    _update_hard_tasks_from_guard(best_guard_info)
                    try:
                        expand_info2 = dict(expand_info)
                        expand_info2["expand_min_cv"] = float(min_expand_cv)
                        expand_info2["expand_min_cv_met"] = bool(met)
                        name = f"cv_guard_expand_to_{int(guard_n_tasks)}" + ("_below_min_cv" if not met else "") + ".json"
                        if _is_main_process():
                            (outdir / "logs" / name).write_text(
                                json.dumps(expand_info2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                            )
                    except Exception:
                        pass
                    tag = "EXPAND_ACCEPT" if bool(met) else "EXPAND_COMMIT"
                    print(
                        f"[guard] {tag} old={old_n} -> new={int(new_n)} cv_rate={e_rate:.3f} ce_rate={e_ce:.3f} "
                        f"min_cv={float(min_expand_cv):.3f} n_tasks={int(guard_n_tasks)}",
                        flush=True,
                    )

                # (3) BEST guard: expanded guard metrics used for selecting ppo_best (no rollback on these).
                if best_guard_info is None:
                    if int(guard_n_tasks) <= int(core_n):
                        best_guard_info = core_guard
                    else:
                        best_tasks = list(guard_pool[: int(guard_n_tasks)])
                        npt_best = _guard_n_per_task(int(len(best_tasks)))
                        best_guard_info = _guard_eval(
                            trainer=trainer,
                            tok=tok,
                            build_prompt=build_prompt,
                            logits_proc=logits_proc,
                            tol=float(tol_used),
                            min_elems=int(args.min_elems),
                            n_per_task=int(npt_best),
                            seed=int(args.cv_guard_seed),
                            t_pre=float(args.t_pre),
                            t_win=float(args.t_win),
                            sim_timeout_s=float(args.sim_timeout_s),
                            autotune_duty=bool(args.autotune_duty),
                            max_new_tokens=int(args.max_new_tokens),
                            temperature=float(args.temperature),
                            top_p=float(args.top_p),
                            constrained=bool(args.constrained),
                            sim_executor=sim_executor,
                            tasks=best_tasks,
                        )
                        _update_hard_tasks_from_guard(best_guard_info)

            if bool(guard_accepted):
                bg = best_guard_info or {}
                bg_rate = float(bg.get("cv_rate", g_rate) or g_rate)
                bg_ce = float(bg.get("ce_rate", 0.0) or 0.0)
                bg_minfam = float(bg.get("cv_min_family", bg_rate) or bg_rate)
                bg_n = int(bg.get("n_tasks", guard_n_tasks) or guard_n_tasks)
                best_guard_cv_for_expand = float(bg_rate)
                print(
                    f"[guard] ACCEPT step={int(step)} core_cv={g_rate:.3f} core_ce={g_ce:.3f} "
                    f"best_guard={int(bg_n)} cv={bg_rate:.3f} ce={bg_ce:.3f} minfam={bg_minfam:.3f} "
                    f"streak={int(guard_streak)} hard={int(len(hard_tasks))}",
                    flush=True,
                )

        # Recompute controller summaries after potential rollback.
        risk_info = risk_ctl.summary()

        tol_info = tol_ctl.summary()
        tol_state = tol_info.copy()
        tol_state.update(
            {
                "levels": list(tol_ctl.levels),
                "window": int(tol_ctl.window),
                "hi": float(tol_ctl.hi),
                "lo": float(tol_ctl.lo),
                "min_samples": int(tol_ctl.min_samples),
                "history": list(tol_ctl.history),
            }
        )
        risk_state = risk_info.copy()
        risk_state.update(
            {
                "lambda_lr": float(risk_ctl.lambda_lr),
                "lambda_max": float(risk_ctl.lambda_max),
                "window": int(risk_ctl.window),
                "history": list(risk_ctl.history),
            }
        )

        guard_state = {
            "guard_n_tasks": int(guard_n_tasks),
            "guard_tasks_max": int(len(guard_pool)),
            "guard_streak": int(guard_streak),
            "best_guard_cv_for_expand": float(best_guard_cv_for_expand),
            "core_tasks": [
                {"family": k[0], "vin": float(k[1]), "vout": float(k[2]), "pass_rate_min": float(v)}
                for k, v in core_task_baseline.items()
            ],
            "hard_tasks": [
                {"family": str(t.family).strip().lower(), "vin": float(t.vin), "vout": float(t.vout)}
                for t in hard_tasks
            ],
        }
        row = {
            "step": int(step),
            "tol": float(tol_used),
            "tol_idx": int(tol_idx_used),
            "tol_next": float(tol_info["tol"]),
            "tol_idx_next": int(tol_info["tol_idx"]),
            "cv_rate_win": float(tol_info["cv_rate_win"]),
            "w_cv": float(weights.w_cv),
            "w_ce": float(weights.w_ce),
            "w_ripple": float(weights.w_ripple),
            "w_over": float(weights.w_over),
            "reward_raw_mean": reward_raw_mean,
            "reward_used_mean": reward_used_mean,
            "reward_rule_mean": parts_mean["reward_rule"],
            "rule_score_mean": parts_mean["rule_score"],
            "reward_cv_mean": parts_mean["reward_cv"],
            "reward_ce_mean": parts_mean["reward_ce"],
            "pen_ripple_mean": parts_mean["pen_ripple"],
            "pen_over_mean": parts_mean["pen_over"],
            "pen_size_mean": parts_mean["pen_size"],
            "pen_fail_mean": parts_mean.get("pen_fail", 0.0),
            "ok_rate": ok_rate,
            "cv_rate": cv_rate,
            "ce_rate": ce_rate,
            "min_elems_rate": min_elems_rate,
            "ok_strict_rate": ok_strict_rate,
            "cv_strict_rate": cv_strict_rate,
            "ce_strict_rate": ce_strict_rate,
            "n_elems_mean": n_elems_mean,
            "strict_success_rate": strict_success_rate,
            "guard/cv_rate": float((guard_info or {}).get("cv_rate", 0.0) or 0.0),
            "guard/cv_min_family": float((guard_info or {}).get("cv_min_family", 0.0) or 0.0),
            "guard/ce_rate": float((guard_info or {}).get("ce_rate", 0.0) or 0.0),
            "guard/accepted": 1 if bool(guard_accepted) else 0,
            "best_guard/n_tasks": int((best_guard_info or {}).get("n_tasks", 0) or 0),
            "best_guard/cv_rate": float((best_guard_info or {}).get("cv_rate", 0.0) or 0.0),
            "best_guard/ce_rate": float((best_guard_info or {}).get("ce_rate", 0.0) or 0.0),
            "best_guard/cv_min_family": float((best_guard_info or {}).get("cv_min_family", 0.0) or 0.0),
            "best_guard/ce_min_family": float((best_guard_info or {}).get("ce_min_family", 0.0) or 0.0),
            "risk/lambda_fail": float(risk_info.get("lambda_fail", 0.0)),
            "risk/fail_rate_win": float(risk_info.get("fail_rate_win", 0.0)),
            "risk/success_rate_win": float(risk_info.get("success_rate_win", 0.0)),
            "risk/target_fail": float(risk_info.get("target_fail", float(args.target_fail_rate))),
            "group/mode": str(args.group_reward_mode),
            "group/rel_coef": float(args.group_rel_coef),
            "objective/kl": _safe_float(stats.get("objective/kl", 0.0), 0.0),
            "objective/entropy": _safe_float(stats.get("objective/entropy", 0.0), 0.0),
            "objective/kl_coef": _safe_float(stats.get("objective/kl_coef", 0.0), 0.0),
            "ppo/mean_scores": _safe_float(stats.get("ppo/mean_scores", 0.0), 0.0),
            "ppo/std_scores": _safe_float(stats.get("ppo/std_scores", 0.0), 0.0),
            "ppo/loss/total": _safe_float(stats.get("ppo/loss/total", 0.0), 0.0),
            "ppo/loss/policy": _safe_float(stats.get("ppo/loss/policy", 0.0), 0.0),
            "ppo/loss/value": _safe_float(stats.get("ppo/loss/value", 0.0), 0.0),
            "ppo/loss/entropy_bonus": _safe_float(stats.get("ppo/loss/entropy_bonus", 0.0), 0.0),
            "ppo/policy/entropy": _safe_float(stats.get("ppo/policy/entropy", 0.0), 0.0),
            "ppo/policy/approxkl": _safe_float(stats.get("ppo/policy/approxkl", 0.0), 0.0),
            "ppo/policy/policykl": _safe_float(stats.get("ppo/policy/policykl", 0.0), 0.0),
            "ppo/policy/clipfrac": _safe_float(stats.get("ppo/policy/clipfrac", 0.0), 0.0),
            "ppo/policy/advantages_mean": _safe_float(stats.get("ppo/policy/advantages_mean", 0.0), 0.0),
            "ppo/val/vpred": _safe_float(stats.get("ppo/val/vpred", 0.0), 0.0),
            "ppo/val/error": _safe_float(stats.get("ppo/val/error", 0.0), 0.0),
            "ppo/val/clipfrac": _safe_float(stats.get("ppo/val/clipfrac", 0.0), 0.0),
            "ppo/val/var_explained": _safe_float(stats.get("ppo/val/var_explained", 0.0), 0.0),
        }

        if _is_main_process():
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=csv_fields)
                w.writerow(row)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            with rollout_path.open("a", encoding="utf-8") as f:
                for rec in details:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(
            f"[PPO] step {step+1}/{int(args.steps)} tol={tol_used:.3f}->{float(tol_info['tol']):.3f} " if float(tol_info['tol']) != float(tol_used) else f"[PPO] step {step+1}/{int(args.steps)} tol={tol_used:.3f} "
            f"reward_raw={reward_raw_mean:.3f} reward_used={reward_used_mean:.3f} "
            f"ok={ok_rate:.3f} cv={cv_rate:.3f} ce={ce_rate:.3f} min20={min_elems_rate:.3f} "
            f"cv_strict={cv_strict_rate:.3f} nelems={n_elems_mean:.1f}",
            flush=True,
        )

        if bool(guard_accepted) and (guard_info is not None):
            bg = best_guard_info or {}
            bg_rate = float(bg.get("cv_rate", 0.0) or 0.0)
            bg_ce = float(bg.get("ce_rate", 0.0) or 0.0)
            bg_minfam = float(bg.get("cv_min_family", bg_rate) or bg_rate)

            min_expand_cv = float(getattr(args, "cv_guard_expand_min_cv", 0.0) or 0.0)
            core_n = int(len(core_guard_keys))
            best_ok = True
            if int(guard_n_tasks) > int(core_n) and (bg_rate + 1e-9 < float(min_expand_cv)):
                best_ok = False
                print(
                    f"[best] skip_save step={int(step)} guard_n_tasks={int(guard_n_tasks)} guard_cv={float(bg_rate):.3f} < min_expand_cv={float(min_expand_cv):.3f}",
                    flush=True,
                )

            if bool(best_ok):
                cand_key = (
                    float(guard_n_tasks),
                    float(bg_rate),
                    float(bg_ce),
                    float(bg_minfam),
                    float(reward_raw_mean),
                )
                cand_guard_payload: Dict[str, Any] = {"best_guard": bg}

                # Reduce expanded-guard noise: when a candidate would beat current best, confirm with extra seeds.
                if tuple(float(x) for x in cand_key) > best_key and int(guard_n_tasks) > int(core_n):
                    k_conf = int(getattr(args, "cv_guard_best_confirm_seeds", 1) or 1)
                    k_conf = max(1, int(k_conf))
                    if k_conf > 1:
                        best_tasks = list(guard_pool[: int(guard_n_tasks)])
                        total = int(getattr(args, "cv_guard_best_confirm_total_samples", 0) or 0)
                        if total <= 0:
                            total = int(getattr(args, "cv_guard_total_samples", 0) or 0)
                        npt_conf = _guard_n_per_task(int(len(best_tasks)))
                        if int(total) > 0:
                            npt_conf = max(1, int(total) // max(1, int(len(best_tasks))))

                        infos = [bg]
                        for j in range(1, int(k_conf)):
                            info_j = _guard_eval(
                                trainer=trainer,
                                tok=tok,
                                build_prompt=build_prompt,
                                logits_proc=logits_proc,
                                tol=float(tol_used),
                                min_elems=int(args.min_elems),
                                n_per_task=int(npt_conf),
                                seed=int(args.cv_guard_seed) + int(j),
                                t_pre=float(args.t_pre),
                                t_win=float(args.t_win),
                                sim_timeout_s=float(args.sim_timeout_s),
                                autotune_duty=bool(args.autotune_duty),
                                max_new_tokens=int(args.max_new_tokens),
                                temperature=float(args.temperature),
                                top_p=float(args.top_p),
                                constrained=bool(args.constrained),
                                sim_executor=sim_executor,
                                tasks=best_tasks,
                            )
                            infos.append(info_j)

                        cv_vals = [float(d.get("cv_rate", 0.0) or 0.0) for d in infos]
                        ce_vals = [float(d.get("ce_rate", 0.0) or 0.0) for d in infos]
                        minfam_vals = [float(d.get("cv_min_family", 0.0) or 0.0) for d in infos]
                        cv_ag = float(min(cv_vals)) if cv_vals else float(bg_rate)
                        ce_ag = float(min(ce_vals)) if ce_vals else float(bg_ce)
                        minfam_ag = float(min(minfam_vals)) if minfam_vals else float(bg_minfam)

                        if cv_ag + 1e-9 < float(min_expand_cv):
                            print(
                                f"[best] confirm_fail step={int(step)} guard_n_tasks={int(guard_n_tasks)} cv_min={cv_ag:.3f} < min_expand_cv={float(min_expand_cv):.3f}; skip",
                                flush=True,
                            )
                            cand_key = None
                        else:
                            cand_key = (
                                float(guard_n_tasks),
                                float(cv_ag),
                                float(ce_ag),
                                float(minfam_ag),
                                float(reward_raw_mean),
                            )
                            cand_guard_payload = {
                                "best_guard": bg,
                                "best_guard_confirm": infos[1:],
                                "confirm": {
                                    "k": int(k_conf),
                                    "n_per_task": int(npt_conf),
                                    "total_samples": int(total),
                                    "agg": "min",
                                },
                            }
                            if _is_main_process():
                                try:
                                    (outdir / "logs" / f"best_confirm_step_{int(step)}.json").write_text(
                                        json.dumps(cand_guard_payload, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8",
                                    )
                                except Exception:
                                    pass

                if cand_key is not None and tuple(float(x) for x in cand_key) > best_key:
                    _save_best_checkpoint(
                        step=int(step),
                        cand_key=tuple(float(x) for x in cand_key),
                        reward_raw_mean=float(reward_raw_mean),
                        strict_success_rate=float(strict_success_rate),
                        batch_ce_rate=float(ce_rate),
                        cv_guard=cand_guard_payload,
                        tol_state=tol_state,
                        risk_state=risk_state,
                        guard_state=guard_state,
                    )

        

        # Convergence criterion (optional): only stop when tol is tight and CV is consistently high.
        # Disabled by default (converge_cv_window<=0).
        k_req = int(getattr(args, "converge_cv_window", 0) or 0)
        if k_req > 0 and (not conv_triggered) and float(tol_info.get("tol", 1.0)) <= float(args.converge_tol):
            cv_metric = float(cv_strict_rate) if bool(args.converge_use_strict_cv) else float(cv_rate)
            conv_cv_hist.append(float(cv_metric))
            k = int(k_req)
            if len(conv_cv_hist) > k:
                conv_cv_hist = conv_cv_hist[-k:]
            if len(conv_cv_hist) >= k:
                cv_ma = float(sum(conv_cv_hist) / float(len(conv_cv_hist)))
                if cv_ma >= float(args.converge_cv_threshold):
                    conv_triggered = True
                    _FORCE_SAVE_REQUESTED = True
                    _STOP_REQUESTED = True
                    _write_text(
                        outdir / "CONVERGED.txt",
                        f"converged_at_step={int(step)} tol={float(tol_info['tol']):.6f} "
                        f"cv_ma{k}={cv_ma:.6f} ts={_now()}\n",
                    )
                    print(
                        f"[converge] tol={float(tol_info['tol']):.3f} cv_ma{k}={cv_ma:.3f} >= "
                        f"{float(args.converge_cv_threshold):.3f}; stopping...",
                        flush=True,
                    )
        else:
            # reset window if tol loosens again
            conv_cv_hist = []

        # checkpoint
        do_ckpt = bool(_FORCE_SAVE_REQUESTED) or bool(_STOP_REQUESTED) or (int(step) + 1) % int(args.save_steps) == 0 or (int(step) + 1) == int(args.steps)
        # Avoid wasting disk on rejected (rolled-back) steps unless explicitly requested.
        if (not bool(guard_accepted)) and (not bool(_FORCE_SAVE_REQUESTED)) and (not bool(_STOP_REQUESTED)):
            do_ckpt = False
        if do_ckpt:
            if _is_main_process():
                ckpt_dir = outdir / f"ppo_step_{int(step)}"
                _mkdir(ckpt_dir)
                _unwrap_trainer_model(trainer).save_pretrained(str(ckpt_dir))
                # also snapshot config/state for reproducibility
                (ckpt_dir / "ppo_config.json").write_text(
                    json.dumps(ppo_cfg.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
                )
                (ckpt_dir / "tol_state.json").write_text(json.dumps(tol_state, ensure_ascii=False, indent=2), encoding="utf-8")
                (ckpt_dir / "risk_state.json").write_text(
                    json.dumps(risk_state, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (ckpt_dir / "guard_state.json").write_text(
                    json.dumps(guard_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )

                _prune_ckpts()
            _FORCE_SAVE_REQUESTED = False

        if bool(_STOP_REQUESTED):
            _write_text(outdir / "STOPPED.txt", f"stopped_at_step={int(step)} ts={_now()}\n")
            print(f"[stop] stop requested; exiting at step={int(step)}", flush=True)
            break


if __name__ == "__main__":
    main()

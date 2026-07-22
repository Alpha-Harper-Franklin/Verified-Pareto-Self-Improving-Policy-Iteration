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
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
import torch

# PyTorch>=2.6 changed torch.load default `weights_only=True`, which can break
# Trainer checkpoint RNG-state resume. We trust our own checkpoints here.
try:
    import numpy as np
    import numpy.core.multiarray
    import torch.serialization

    torch.serialization.add_safe_globals([
        numpy.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
    ])
except Exception:
    pass

from trl import DPOConfig, DPOTrainer


def _is_main_process() -> bool:
    try:
        return int(os.environ.get("RANK", "0") or 0) == 0
    except Exception:
        return True


class CSVLogger(TrainerCallback):
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._header_written = os.path.exists(self.csv_path)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not _is_main_process():
            return
        if not logs:
            return
        step = int(getattr(state, "global_step", 0))
        row: Dict[str, Any] = {"step": int(step)}
        if isinstance(logs.get("loss"), (int, float)):
            row["loss"] = float(logs["loss"])
        if isinstance(logs.get("learning_rate"), (int, float)):
            row["learning_rate"] = float(logs["learning_rate"])
        if isinstance(logs.get("epoch"), (int, float)):
            row["epoch"] = float(logs["epoch"])

        # JSONL keeps all numeric keys for post-hoc analysis.
        jsonl_path = os.path.splitext(self.csv_path)[0] + ".jsonl"
        extra = {k: float(v) for k, v in (logs or {}).items() if isinstance(v, (int, float))}
        with open(jsonl_path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps({"step": int(step), **extra}, ensure_ascii=False) + "\n")

        fieldnames = ["step", "loss", "learning_rate", "epoch"]
        if not self._header_written:
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerow(row)
            self._header_written = True
        else:
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writerow(row)


def load_dpo_dataset(path: str) -> Dataset:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]})
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--sft_adapter", required=True)
    ap.add_argument("--dpo_pairs", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=0, help="0 means use --epochs")
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in --outdir if present")
    ap.add_argument("--ddp_backend", default="nccl", help="DDP backend when launched multi-process (e.g., via accelerate/torchrun). Use gloo if NCCL is unstable.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _latest_ckpt(d: Path) -> str:
        cks = sorted([p for p in d.glob("checkpoint-*") if p.is_dir()], key=lambda p: int(p.name.split("-")[-1]))
        return str(cks[-1]) if cks else ""

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.bos_token_id is None:
        tok.bos_token = tok.eos_token

    ds = load_dpo_dataset(args.dpo_pairs)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    model.config.use_cache = False

    train_args = DPOConfig(
        output_dir=args.outdir,
        num_train_epochs=int(args.epochs),
        per_device_train_batch_size=int(args.bsz),
        gradient_accumulation_steps=int(args.grad_accum),
        max_steps=int(args.max_steps) if int(args.max_steps) > 0 else -1,
        learning_rate=float(args.lr),
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=1,
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        report_to=[],
        remove_unused_columns=False,
        ddp_backend=str(args.ddp_backend).strip() or None,
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
    )

    trainer = DPOTrainer(
        model=model,
        # For PEFT models, TRL can use the same model with adapters disabled as the reference,
        # which avoids a second full model copy and is more memory-stable on this platform.
        ref_model=None,
        args=train_args,
        beta=0.1,
        train_dataset=ds,
        tokenizer=tok,
        max_length=1024,
        max_prompt_length=512,
    )
    trainer.add_callback(CSVLogger(str(outdir / "dpo_metrics.csv")))

    resume_from = None
    if bool(args.resume):
        ck = _latest_ckpt(outdir)
        if ck:
            resume_from = ck
            print(f"[resume] {ck}")
    trainer.train(resume_from_checkpoint=resume_from)

    out_adapter = outdir / "dpo_final"
    if _is_main_process():
        model.save_pretrained(str(out_adapter))
        print("[OK] saved", out_adapter)


if __name__ == "__main__":
    main()

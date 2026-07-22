#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from inc_parser import parse_inc
from repair_corrector import FAMILY_ORDER, _MLP, build_feature_vector


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _select_tunable_names(inc: str, family: str) -> List[str]:
    fam = str(family or "").strip().lower()
    elems = parse_inc(inc)

    def _match(kind: str, a: str, b: str) -> Optional[str]:
        a0 = a.lower()
        b0 = b.lower()
        for e in elems:
            if e.kind != kind or not e.name:
                continue
            n = {str(x).lower() for x in (e.nodes or [])}
            if a0 in n and b0 in n:
                return str(e.name)
        return None

    want: List[str] = []
    if fam == "buck":
        want += [x for x in [_match("L", "sw", "out"), _match("C", "out", "0")] if x]
    elif fam == "boost":
        want += [x for x in [_match("L", "vin", "sw"), _match("C", "out", "0")] if x]
    elif fam == "sepic":
        want += [
            x
            for x in [
                _match("L", "vin", "sw"),
                _match("L", "n1", "0"),
                _match("C", "sw", "n1"),
                _match("C", "out", "0"),
            ]
            if x
        ]
    elif fam in {"buckboost", "buck-boost", "bb"}:
        want += [
            x
            for x in [
                _match("L", "sw1", "mid"),
                _match("C", "mid", "0"),
                _match("L", "mid", "sw2"),
                _match("C", "out", "0"),
            ]
            if x
        ]

    if not want:
        for e in elems:
            if e.kind in {"L", "C"} and e.name not in want:
                want.append(str(e.name))
            if len(want) >= 4:
                break
    return want[:4]


def _inc_value_map(inc: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for e in parse_inc(inc):
        if e.value is None or not e.name:
            continue
        out[str(e.name)] = float(e.value)
    return out


@dataclass
class Example:
    x: List[float]
    y: List[float]
    m: List[float]
    weight: float


def _load_examples(selfplay_root: Path) -> Tuple[List[Example], Dict[str, Any]]:
    scored_files = sorted(selfplay_root.glob("tasks/*/*/scored.json"))
    n_files = 0
    n_rows = 0
    examples: List[Example] = []
    fam_cnt: Dict[str, int] = {}
    tuned_cnt = 0

    for sf in scored_files:
        n_files += 1
        try:
            obj = json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, list):
            continue

        for rec in obj:
            n_rows += 1
            fam = str(rec.get("family") or "").strip().lower()
            if fam not in FAMILY_ORDER:
                continue
            rep = (rec.get("repair") or {}) if isinstance(rec, dict) else {}
            if not isinstance(rep, dict) or not rep:
                continue

            changed = bool(rep.get("changed", False))
            score_b = _safe_float(rep.get("score_before", -1.0), -1.0)
            score_a = _safe_float(rep.get("score_after", -1.0), -1.0)
            if (not changed) or (score_a <= score_b + 1e-6):
                continue

            inc_b = str(rep.get("inc_before") or rec.get("inc") or "").strip()
            inc_a = str(rep.get("inc_after") or "").strip()
            if not inc_b or not inc_a:
                continue

            tun_names = _select_tunable_names(inc_b, family=fam)
            if not tun_names:
                continue

            vb = _inc_value_map(inc_b)
            va = _inc_value_map(inc_a)
            x_tun: List[float] = []
            y_log: List[float] = []
            mask: List[float] = []
            any_tuned = False

            for name in tun_names:
                b = _safe_float(vb.get(name, 0.0), 0.0)
                a = _safe_float(va.get(name, 0.0), 0.0)
                if b > 0.0 and a > 0.0:
                    lr = float(math.log(a / b))
                    any_tuned = True
                    y_log.append(lr)
                    mask.append(1.0)
                else:
                    y_log.append(0.0)
                    mask.append(0.0)
                x_tun.append(float(b) if b > 0.0 else 0.0)

            while len(x_tun) < 4:
                x_tun.append(0.0)
            while len(y_log) < 4:
                y_log.append(0.0)
                mask.append(0.0)

            if not any_tuned:
                continue
            tuned_cnt += 1

            detail_b = rep.get("detail_before") or {}
            if not isinstance(detail_b, dict):
                detail_b = {}
            n_elems = rep.get("detail_before", {}).get("n_elems") if isinstance(rep.get("detail_before"), dict) else None
            if n_elems is None:
                n_elems = rec.get("n_elems")
            x = build_feature_vector(
                family=fam,
                vin=_safe_float(rec.get("vin", 0.0), 0.0),
                vout=_safe_float(rec.get("vout", 0.0), 0.0),
                detail_before=detail_b,
                tunable_values=x_tun,
                n_elems=_safe_float(n_elems, 0.0),
            )

            w = float(max(0.1, min(5.0, score_a - score_b)))
            examples.append(Example(x=x, y=y_log, m=mask, weight=w))
            fam_cnt[fam] = int(fam_cnt.get(fam, 0) + 1)

    stats = {
        "selfplay_root": str(selfplay_root),
        "n_scored_files": int(n_files),
        "n_rows_total": int(n_rows),
        "n_examples": int(len(examples)),
        "n_with_any_tuned": int(tuned_cnt),
        "examples_by_family": fam_cnt,
    }
    return examples, stats


class _XYDataset(Dataset):
    def __init__(self, ex: List[Example]) -> None:
        self.ex = ex

    def __len__(self) -> int:
        return len(self.ex)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e = self.ex[i]
        return (
            torch.tensor(e.x, dtype=torch.float32),
            torch.tensor(e.y, dtype=torch.float32),
            torch.tensor(e.m, dtype=torch.float32),
            torch.tensor([e.weight], dtype=torch.float32),
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfplay_root", required=True, help="Round selfplay_data dir containing tasks/*/*/scored.json")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--min_examples", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--max_factor", type=float, default=2.0, help="Clamp target/pred scale to [1/max_factor, max_factor]")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    sp = Path(args.selfplay_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    examples, stats = _load_examples(sp)
    (outdir / "data_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if len(examples) < int(args.min_examples):
        (outdir / "SKIPPED.txt").write_text(
            f"Not enough examples ({len(examples)} < {int(args.min_examples)})\n", encoding="utf-8"
        )
        print(f"[repair-corrector] skip: only {len(examples)} examples")
        return 0

    # Build tensors + normalization
    x_all = torch.stack([torch.tensor(e.x, dtype=torch.float32) for e in examples], dim=0)
    x_mean = x_all.mean(dim=0)
    x_std = x_all.std(dim=0).clamp(min=1e-6)

    ds = _XYDataset(examples)
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)

    in_dim = int(x_all.shape[1])
    model = _MLP(in_dim=in_dim, out_dim=4, hidden=int(args.hidden))
    dev = torch.device(str(args.device))
    model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    max_factor = float(max(1.01, float(args.max_factor)))
    clamp_log = float(math.log(max_factor))

    for epoch in range(int(args.epochs)):
        model.train()
        total = 0.0
        denom = 0.0
        for xb, yb, mb, wb in dl:
            xb = xb.to(dev)
            yb = yb.to(dev)
            mb = mb.to(dev)
            wb = wb.to(dev)
            pred = model((xb - x_mean.to(dev)) / x_std.to(dev))
            pred = torch.clamp(pred, min=-clamp_log, max=clamp_log)
            yb = torch.clamp(yb, min=-clamp_log, max=clamp_log)
            abs_err = torch.abs(pred - yb) * mb
            w = wb.view(-1, 1)
            loss = (abs_err * w).sum() / torch.clamp((mb * w).sum(), min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += float(loss.detach().cpu()) * float(xb.shape[0])
            denom += float(xb.shape[0])

        avg = total / max(1.0, denom)
        with (outdir / "train_log.txt").open("a", encoding="utf-8") as f:
            f.write(f"epoch={epoch} loss={avg:.6f}\n")
        print(f"[repair-corrector] epoch {epoch}/{int(args.epochs)} loss={avg:.6f}")

    ckpt = {
        "model": model.state_dict(),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "meta": {"in_dim": int(in_dim), "hidden": int(args.hidden), "max_factor": float(max_factor)},
    }
    torch.save(ckpt, str(outdir / "model.pt"))
    print(f"[repair-corrector] saved: {outdir / 'model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

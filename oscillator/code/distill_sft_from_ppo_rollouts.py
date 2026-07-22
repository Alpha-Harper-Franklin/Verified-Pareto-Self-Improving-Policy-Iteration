#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from inc_parser import extract_inc_lines


RESPONSE_TEMPLATE = "### Response:\n"


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _iter_jsonl(paths: Sequence[Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()


def _build_prompt(family: str, vin: float, vout: float) -> str:
    try:
        from eval_dcdc_family import build_prompt  # type: ignore

        return str(build_prompt(str(family), float(vin), float(vout)))
    except Exception:
        fam = (family or "").strip().lower()
        return (
            f"Generate a {fam} DC-DC converter in INC DSL.\n"
            "Rules:\n"
            "- Output ONLY INC lines (no explanation).\n"
            "- Line format: INC <name> <node1> <node2> <value_or_model>\n"
            "- Element names must start with R/L/C/D/S and contain a digit.\n"
            "- Use at least 20 INC lines.\n"
            f"Task: Vin={float(vin):.1f}V, Vout={float(vout):.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )


@dataclass(frozen=True)
class _KeepRule:
    require_ok: bool
    require_pass_cv: bool
    require_pass_ce: bool
    min_elems: int
    min_reward: float


def _canonical_hash(r: Dict[str, Any]) -> str:
    detail = r.get("detail") or {}
    h = str(detail.get("canonical_hash") or "").strip()
    if h:
        return h
    inc = str(r.get("inc_source") or "").strip()
    if inc:
        return _hash_text(inc)
    raw = str(r.get("raw_text") or "").strip()
    return _hash_text(raw)


def _inc_from_rollout(r: Dict[str, Any]) -> str:
    inc = str(r.get("inc_source") or "").strip()
    if not inc:
        # fall back to parsing INC lines from raw_text
        raw = str(r.get("raw_text") or "")
        inc_lines = extract_inc_lines(raw)
        inc = ("\n".join(inc_lines)).strip()
    if inc and not inc.endswith("\n"):
        inc += "\n"
    return inc


def _keep(r: Dict[str, Any], rule: _KeepRule) -> Tuple[bool, str]:
    detail = r.get("detail") or {}
    ok = bool(detail.get("ok", False))
    pass_cv = bool(detail.get("pass_CV", False))
    pass_ce = bool(detail.get("pass_CE", False))
    n_elems = _safe_int(r.get("n_elems"), 0)
    reward = _safe_float(r.get("reward"), -999.0)

    if rule.require_ok and (not ok):
        return False, "not_ok"
    if rule.require_pass_cv and (not pass_cv):
        return False, "not_pass_cv"
    if rule.require_pass_ce and (not pass_ce):
        return False, "not_pass_ce"
    if int(n_elems) < int(rule.min_elems):
        return False, "too_few_elems"
    if float(reward) < float(rule.min_reward):
        return False, "reward_too_low"
    inc = _inc_from_rollout(r)
    if not inc.strip():
        return False, "empty_inc"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True, help="One or more ppo_rollouts.jsonl paths (glob allowed)")
    ap.add_argument("--out_sft_jsonl", required=True)
    ap.add_argument("--out_meta_jsonl", default="")
    ap.add_argument("--out_report_json", default="")
    ap.add_argument("--seed", type=int, default=2025)

    ap.add_argument("--require_ok", action="store_true")
    ap.add_argument("--require_pass_cv", action="store_true")
    ap.add_argument("--require_pass_ce", action="store_true")
    ap.add_argument("--min_elems", type=int, default=20)
    ap.add_argument("--min_reward", type=float, default=-1e9)

    ap.add_argument("--max_per_task", type=int, default=50, help="0 means no limit per (family,vin,vout)")
    ap.add_argument("--global_limit", type=int, default=0, help="0 means no global limit")
    ap.add_argument("--resume", action="store_true", help="Append, skipping already-exported canonical hashes")
    args = ap.parse_args()

    # Expand globs
    rollout_paths: List[Path] = []
    for g in args.rollouts:
        ms = [Path(p) for p in glob.glob(str(g))]
        rollout_paths.extend(ms if ms else [Path(g)])
    rollout_paths = [p for p in rollout_paths if p.exists()]
    if not rollout_paths:
        raise SystemExit("no rollouts found")

    out_sft = Path(args.out_sft_jsonl)
    out_sft.parent.mkdir(parents=True, exist_ok=True)
    out_meta = Path(args.out_meta_jsonl) if str(args.out_meta_jsonl).strip() else None
    out_report = Path(args.out_report_json) if str(args.out_report_json).strip() else None
    if out_meta:
        out_meta.parent.mkdir(parents=True, exist_ok=True)
    if out_report:
        out_report.parent.mkdir(parents=True, exist_ok=True)

    rule = _KeepRule(
        require_ok=bool(args.require_ok),
        require_pass_cv=bool(args.require_pass_cv),
        require_pass_ce=bool(args.require_pass_ce),
        min_elems=int(args.min_elems),
        min_reward=float(args.min_reward),
    )

    exported: Set[str] = set()
    if bool(args.resume) and out_sft.exists():
        # Prefer meta file hashes if available; else re-hash SFT outputs.
        if out_meta and out_meta.exists():
            with out_meta.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    h = str(r.get("canonical_hash") or "").strip()
                    if h:
                        exported.add(h)
        else:
            with out_sft.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "\"text\"" not in line:
                        continue
                    exported.add(_hash_text(line.strip()))

    # Collect candidates grouped by task
    by_task: Dict[Tuple[str, float, float], List[Dict[str, Any]]] = defaultdict(list)
    n_total = 0
    n_kept_raw = 0
    n_dedup_drop = 0
    n_filtered = 0

    for r in _iter_jsonl(rollout_paths):
        n_total += 1
        family = str(r.get("family") or "").strip().lower() or "unknown"
        vin = _safe_float(r.get("vin"), 0.0)
        vout = _safe_float(r.get("vout"), 0.0)
        key = (family, float(vin), float(vout))

        ok, why = _keep(r, rule)
        if not ok:
            n_filtered += 1
            continue

        h = _canonical_hash(r)
        if h in exported:
            n_dedup_drop += 1
            continue

        # Store normalized essentials for sorting
        rr = {
            "family": family,
            "vin": float(vin),
            "vout": float(vout),
            "step": _safe_int(r.get("step"), -1),
            "reward": _safe_float(r.get("reward"), -999.0),
            "n_elems": _safe_int(r.get("n_elems"), 0),
            "canonical_hash": h,
            "detail": r.get("detail") or {},
            "inc": _inc_from_rollout(r),
        }
        by_task[key].append(rr)
        n_kept_raw += 1

    # Select top per task
    selected: List[Dict[str, Any]] = []
    for key, items in by_task.items():
        items.sort(key=lambda x: (float(x.get("reward", -999.0)), float(x.get("n_elems", 0))), reverse=True)
        if int(args.max_per_task) > 0:
            items = items[: int(args.max_per_task)]
        selected.extend(items)

    # Global sort and limit
    selected.sort(key=lambda x: (float(x.get("reward", -999.0)), float(x.get("n_elems", 0))), reverse=True)
    if int(args.global_limit) > 0:
        selected = selected[: int(args.global_limit)]

    mode = "a" if bool(args.resume) else "w"
    wrote = 0
    with out_sft.open(mode, encoding="utf-8") as f_sft:
        f_meta = out_meta.open(mode, encoding="utf-8") if out_meta else None
        try:
            for r in selected:
                prompt = _build_prompt(str(r["family"]), float(r["vin"]), float(r["vout"]))
                txt = prompt
                if RESPONSE_TEMPLATE not in txt:
                    txt = txt.rstrip() + "\n" + RESPONSE_TEMPLATE
                txt = txt.rstrip() + "\n" + str(r["inc"]).strip() + "\n"

                f_sft.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                if f_meta:
                    f_meta.write(
                        json.dumps(
                            {
                                "family": r["family"],
                                "vin": r["vin"],
                                "vout": r["vout"],
                                "step": r["step"],
                                "reward": r["reward"],
                                "n_elems": r["n_elems"],
                                "canonical_hash": r["canonical_hash"],
                                "detail": r.get("detail") or {},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                exported.add(str(r["canonical_hash"]))
                wrote += 1
        finally:
            if f_meta:
                f_meta.close()

    report = {
        "ts": _now(),
        "rollouts": [str(p) for p in rollout_paths],
        "out_sft_jsonl": str(out_sft),
        "out_meta_jsonl": str(out_meta) if out_meta else "",
        "rule": {
            "require_ok": bool(rule.require_ok),
            "require_pass_cv": bool(rule.require_pass_cv),
            "require_pass_ce": bool(rule.require_pass_ce),
            "min_elems": int(rule.min_elems),
            "min_reward": float(rule.min_reward),
        },
        "n_total": int(n_total),
        "n_filtered": int(n_filtered),
        "n_kept_raw": int(n_kept_raw),
        "n_dedup_drop": int(n_dedup_drop),
        "n_tasks": int(len(by_task)),
        "n_selected": int(len(selected)),
        "n_written": int(wrote),
    }
    if out_report:
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

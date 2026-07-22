#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator

from inc_parser import IncElem, parse_inc, to_inc_text


def _sha1(text: str) -> str:
    hasher = hashlib.sha1()
    hasher.update((text or "").encode("utf-8", errors="ignore"))
    return hasher.hexdigest()


def _iter_task_dirs(selfplay_root: Path) -> Iterator[Path]:
    tasks_root = selfplay_root / "tasks"
    if not tasks_root.exists():
        return
    for family_dir in sorted([p for p in tasks_root.iterdir() if p.is_dir()]):
        for task_dir in sorted([p for p in family_dir.iterdir() if p.is_dir()]):
            if (task_dir / "prompt.txt").exists() and (task_dir / "scored.json").exists():
                yield task_dir


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any) -> bool:
    return bool(value)

def _augment_inc(
    inc_text: str,
    *,
    keep_names_upper: set[str],
    seed: int,
    aug_id: int,
) -> str:
    """
    Lightweight textual augmentation to increase SFT coverage:
    - rename non-essential element names (names are semantics-free in our INC DSL)
    - shuffle non-essential lines (order-insensitive netlists)
    """
    elems = list(parse_inc(inc_text))
    # Deterministic per-record augmentation.
    import random

    rng = random.Random(int(seed) + 1000003 * int(aug_id))

    new_elems: list[IncElem] = []
    used: set[str] = set()
    cnt: dict[str, int] = {"R": 0, "C": 0, "L": 0, "D": 0, "S": 0}

    def _new_name(kind: str) -> str:
        k = (kind or "R").upper()
        cnt[k] = int(cnt.get(k, 0)) + 1
        # Add randomness but keep it readable.
        return f"{k}X{cnt[k]:02d}{rng.randint(0,9999):04d}"

    for e in elems:
        nm = str(e.name or "").strip()
        nm_u = nm.upper()
        if nm_u and nm_u in keep_names_upper:
            used.add(nm)
            new_elems.append(e)
            continue
        kind = str(e.kind or "R").upper()
        nn = _new_name(kind)
        while nn in used:
            nn = _new_name(kind)
        used.add(nn)
        new_elems.append(IncElem(name=nn, kind=kind, nodes=list(e.nodes), value=e.value, model=e.model, raw=e.raw))

    keep: list[IncElem] = []
    rest: list[IncElem] = []
    for e in new_elems:
        nm_u = str(e.name or "").strip().upper()
        if nm_u in keep_names_upper:
            keep.append(e)
        else:
            rest.append(e)
    rng.shuffle(rest)
    out = to_inc_text(keep + rest).strip()
    return out + ("\n" if out else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfplay_root", required=True)
    ap.add_argument("--out_root", default="", help="Defaults to --selfplay_root")
    ap.add_argument("--min_score_gain", type=float, default=0.0)
    ap.add_argument("--require_after_pass_cv", action="store_true")
    ap.add_argument("--require_improvement", action="store_true")
    ap.add_argument("--max_pairs_per_task", type=int, default=0, help="0 means unlimited")
    ap.add_argument("--max_pairs", type=int, default=0, help="0 means unlimited")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sft_aug_per_record", type=int, default=1, help="Generate N augmented SFT samples per repaired record (>=1).")
    ap.add_argument("--sft_aug_seed", type=int, default=2025)
    ap.add_argument(
        "--sft_aug_keep_names",
        default="RBIAS,RFB,RG,CCOMP",
        help="Comma-separated element names to keep unchanged during augmentation (uppercase match).",
    )
    args = ap.parse_args()

    selfplay_root = Path(args.selfplay_root).resolve()
    out_root = Path(str(args.out_root).strip()).resolve() if str(args.out_root).strip() else selfplay_root
    out_root.mkdir(parents=True, exist_ok=True)

    out_pairs = out_root / "repair_distill_dpo_pairs.jsonl"
    out_sft = out_root / "repair_distill_sft_train.jsonl"
    out_meta = out_root / "repair_distill_meta.jsonl"
    out_report = out_root / "repair_distill_report.json"

    if bool(args.resume) and out_pairs.exists() and out_pairs.stat().st_size > 0 and out_report.exists():
        try:
            rep = json.loads(out_report.read_text(encoding="utf-8", errors="ignore"))
            if (
                int(rep.get("sft_aug_per_record", 1)) == int(args.sft_aug_per_record)
                and str(rep.get("sft_aug_keep_names", "")) == str(args.sft_aug_keep_names)
            ):
                print(f"[repair_distill] reuse existing: {out_pairs}", flush=True)
                return 0
        except Exception:
            pass

    seen_pair: set[str] = set()
    seen_sft: set[str] = set()
    keep_names_upper: set[str] = {p.strip().upper() for p in str(args.sft_aug_keep_names or "").split(",") if p.strip()}

    summary: Dict[str, Any] = {
        "selfplay_root": str(selfplay_root),
        "out_root": str(out_root),
        "sft_aug_per_record": int(args.sft_aug_per_record),
        "sft_aug_keep_names": str(args.sft_aug_keep_names),
        "started_at": time.strftime("%Y%m%d_%H%M%S"),
        "tasks": 0,
        "records": 0,
        "repair_changed": 0,
        "pairs": 0,
        "sft": 0,
        "before": {"ok": 0, "pass_C": 0, "pass_CV": 0},
        "after": {"ok": 0, "pass_C": 0, "pass_CV": 0},
        "by_family": {},
    }
    by_family: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "tasks": 0,
            "records": 0,
            "repair_changed": 0,
            "pairs": 0,
            "before": {"ok": 0, "pass_C": 0, "pass_CV": 0},
            "after": {"ok": 0, "pass_C": 0, "pass_CV": 0},
        }
    )

    def _acc_detail(bucket: Dict[str, int], detail: Dict[str, Any]) -> None:
        for key in ["ok", "pass_C", "pass_CV"]:
            bucket[key] += 1 if _safe_bool(detail.get(key, False)) else 0

    max_pairs_total = int(args.max_pairs) if int(args.max_pairs) > 0 else 10**18

    tmp_pairs = out_pairs.with_suffix(out_pairs.suffix + ".tmp")
    tmp_sft = out_sft.with_suffix(out_sft.suffix + ".tmp")
    tmp_meta = out_meta.with_suffix(out_meta.suffix + ".tmp")

    with tmp_pairs.open("w", encoding="utf-8") as f_pairs, tmp_sft.open("w", encoding="utf-8") as f_sft, tmp_meta.open(
        "w", encoding="utf-8"
    ) as f_meta:
        for task_dir in _iter_task_dirs(selfplay_root):
            prompt = (task_dir / "prompt.txt").read_text(encoding="utf-8", errors="ignore")
            scored = _load_json(task_dir / "scored.json")
            if not isinstance(scored, list):
                continue

            family = str(task_dir.parent.name).lower()
            summary["tasks"] += 1
            by_family[family]["tasks"] += 1
            pairs_this_task = 0

            for rec in scored:
                if summary["pairs"] >= max_pairs_total:
                    break
                if not isinstance(rec, dict):
                    continue
                summary["records"] += 1
                by_family[family]["records"] += 1

                repair = rec.get("repair")
                if not isinstance(repair, dict):
                    continue
                if not _safe_bool(repair.get("changed", False)):
                    continue

                summary["repair_changed"] += 1
                by_family[family]["repair_changed"] += 1

                inc_before = str(repair.get("inc_before") or "").strip()
                inc_after = str(repair.get("inc_after") or "").strip()
                if (not inc_before) or (not inc_after):
                    continue

                detail_before = repair.get("detail_before") if isinstance(repair.get("detail_before"), dict) else {}
                detail_after = repair.get("detail_after") if isinstance(repair.get("detail_after"), dict) else {}
                _acc_detail(summary["before"], detail_before)
                _acc_detail(summary["after"], detail_after)
                _acc_detail(by_family[family]["before"], detail_before)
                _acc_detail(by_family[family]["after"], detail_after)

                if bool(args.require_after_pass_cv) and (not _safe_bool(detail_after.get("pass_CV", False))):
                    continue

                score_before = _as_float(repair.get("score_before"), default=_as_float(rec.get("score"), 0.0))
                score_after = _as_float(repair.get("score_after"), default=_as_float(rec.get("score"), 0.0))
                if bool(args.require_improvement) and (not (score_after > score_before + float(args.min_score_gain))):
                    continue

                if int(args.max_pairs_per_task) > 0 and pairs_this_task >= int(args.max_pairs_per_task):
                    continue

                pair_key = _sha1(prompt + "\n" + inc_after + "\n" + inc_before)
                if pair_key in seen_pair:
                    continue
                seen_pair.add(pair_key)

                f_pairs.write(json.dumps({"prompt": prompt, "chosen": inc_after, "rejected": inc_before}, ensure_ascii=False) + "\n")
                summary["pairs"] += 1
                by_family[family]["pairs"] += 1
                pairs_this_task += 1

                sft_key = _sha1(prompt + "\n" + inc_after)
                # SFT augmentation (makes the policy learn stable formatting + numeric patterns).
                for k in range(max(1, int(args.sft_aug_per_record))):
                    inc_out = inc_after
                    if int(args.sft_aug_per_record) > 1:
                        inc_out = _augment_inc(
                            inc_after,
                            keep_names_upper=keep_names_upper,
                            seed=int(args.sft_aug_seed),
                            aug_id=int(summary["sft"]) * 17 + int(k),
                        )
                    sft_key2 = _sha1(prompt + "\n" + inc_out)
                    if sft_key2 in seen_sft:
                        continue
                    seen_sft.add(sft_key2)
                    f_sft.write(json.dumps({"text": prompt + inc_out}, ensure_ascii=False) + "\n")
                    summary["sft"] += 1

                f_meta.write(
                    json.dumps(
                        {
                            "task_dir": str(task_dir),
                            "family": family,
                            "score_before": float(score_before),
                            "score_after": float(score_after),
                            "before": {k: bool(detail_before.get(k, False)) for k in ["ok", "pass_C", "pass_CV"]},
                            "after": {k: bool(detail_after.get(k, False)) for k in ["ok", "pass_C", "pass_CV"]},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _rate(num: int, den: int) -> float:
        return float(num) / float(max(1, den))

    for family, st in sorted(by_family.items()):
        st["rates"] = {
            "before_ok": _rate(int(st["before"]["ok"]), int(st["repair_changed"])),
            "before_pass_C": _rate(int(st["before"]["pass_C"]), int(st["repair_changed"])),
            "before_pass_CV": _rate(int(st["before"]["pass_CV"]), int(st["repair_changed"])),
            "after_ok": _rate(int(st["after"]["ok"]), int(st["repair_changed"])),
            "after_pass_C": _rate(int(st["after"]["pass_C"]), int(st["repair_changed"])),
            "after_pass_CV": _rate(int(st["after"]["pass_CV"]), int(st["repair_changed"])),
        }
        summary["by_family"][family] = st

    summary["rates"] = {
        "before_ok": _rate(int(summary["before"]["ok"]), int(summary["repair_changed"])),
        "before_pass_C": _rate(int(summary["before"]["pass_C"]), int(summary["repair_changed"])),
        "before_pass_CV": _rate(int(summary["before"]["pass_CV"]), int(summary["repair_changed"])),
        "after_ok": _rate(int(summary["after"]["ok"]), int(summary["repair_changed"])),
        "after_pass_C": _rate(int(summary["after"]["pass_C"]), int(summary["repair_changed"])),
        "after_pass_CV": _rate(int(summary["after"]["pass_CV"]), int(summary["repair_changed"])),
    }
    summary["finished_at"] = time.strftime("%Y%m%d_%H%M%S")

    out_report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_pairs.replace(out_pairs)
    tmp_sft.replace(out_sft)
    tmp_meta.replace(out_meta)

    print(f"[repair_distill] wrote pairs={summary['pairs']} sft={summary['sft']} -> {out_root}", flush=True)
    print(
        f"[repair_distill] rates_before_pass_CV={summary['rates']['before_pass_CV']:.4f} after_pass_CV={summary['rates']['after_pass_CV']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

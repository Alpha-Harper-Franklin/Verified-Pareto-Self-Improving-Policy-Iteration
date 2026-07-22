#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from dcdc_taskset import Task, default_taskset
from task_manifest import ensure_disjoint, sha256_file, task_key, unique_tasks, validate_task


TaskKey = Tuple[str, float, float]
FAMILIES = ("buck", "boost", "sepic", "buckboost")
TASK_PATTERN = re.compile(
    r"Task:\s*Vin\s*=\s*([0-9]+(?:\.[0-9]+)?)V\s*,\s*Vout\s*=\s*([0-9]+(?:\.[0-9]+)?)V",
    flags=re.IGNORECASE,
)


def family_from_text(text: str) -> str | None:
    prefix = text[:2000].lower()
    if "buck-boost" in prefix or "buck boost" in prefix:
        return "buckboost"
    if "sepic" in prefix:
        return "sepic"
    if "boost" in prefix:
        return "boost"
    if "buck" in prefix:
        return "buck"
    return None


def extract_sft_tasks(path: Path) -> Tuple[Set[TaskKey], int, int]:
    keys: Set[TaskKey] = set()
    rows = 0
    matched_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            text = str(record.get("text") or "")
            family = family_from_text(text)
            match = TASK_PATTERN.search(text)
            if family is None or match is None:
                continue
            matched_rows += 1
            keys.add((family, float(match.group(1)), float(match.group(2))))
    return keys, rows, matched_rows


def stratified_guard_split(tasks: Sequence[Task], split_seed: int) -> Tuple[List[Task], List[Task]]:
    by_family: Dict[str, List[Task]] = defaultdict(list)
    for task in tasks:
        by_family[task_key(task)[0]].append(task)
    guard: List[Task] = []
    for family in FAMILIES:
        family_tasks = by_family[family]
        ranked = sorted(
            family_tasks,
            key=lambda task: hashlib.sha256(
                f"{split_seed}|{task.family}|{task.vin:.12g}|{task.vout:.12g}".encode("ascii")
            ).hexdigest(),
        )
        guard.extend(ranked[:2])
    guard_keys = {task_key(task) for task in guard}
    train = [task for task in tasks if task_key(task) not in guard_keys]
    return sort_tasks(train), sort_tasks(guard)


def sort_tasks(tasks: Iterable[Task]) -> List[Task]:
    order = {family: index for index, family in enumerate(FAMILIES)}
    return sorted(tasks, key=lambda task: (order[task_key(task)[0]], float(task.vin), float(task.vout)))


def build_unseen_tasks() -> List[Task]:
    # Ten matched voltage pairs. Buck uses the descending direction; the other
    # families use the ascending direction so every topology remains physical.
    voltage_pairs = [
        (5.5, 9.5),
        (6.0, 10.0),
        (6.5, 10.5),
        (7.0, 11.0),
        (7.5, 11.5),
        (8.0, 12.5),
        (8.5, 13.0),
        (9.5, 13.5),
        (10.0, 14.0),
        (10.5, 14.5),
    ]
    tasks: List[Task] = []
    for low, high in voltage_pairs:
        tasks.append(Task("buck", high, low))
        tasks.append(Task("boost", low, high))
        tasks.append(Task("sepic", low, high))
        tasks.append(Task("buckboost", low, high))
    return sort_tasks(unique_tasks(tasks))


def empirical_ranges(tasks: Sequence[Task]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for family in FAMILIES:
        subset = [task for task in tasks if task_key(task)[0] == family]
        result[family] = {
            "vin_min": min(float(task.vin) for task in subset),
            "vin_max": max(float(task.vin) for task in subset),
            "vout_min": min(float(task.vout) for task in subset),
            "vout_max": max(float(task.vout) for task in subset),
        }
    return result


def validate_test_ranges(test: Sequence[Task], ranges: Dict[str, Dict[str, float]]) -> None:
    for task in test:
        validate_task(task)
        family, vin, vout = task_key(task)
        limits = ranges[family]
        if not (limits["vin_min"] <= vin <= limits["vin_max"]):
            raise ValueError(f"test Vin outside Grid77 range: {task}")
        if not (limits["vout_min"] <= vout <= limits["vout_max"]):
            raise ValueError(f"test Vout outside Grid77 range: {task}")


def write_tasks(path: Path, tasks: Sequence[Task], split: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, task in enumerate(tasks):
            record = {
                "task_id": f"{split}_{index:03d}",
                "split": split,
                "family": str(task.family),
                "vin": float(task.vin),
                "vout": float(task.vout),
                "rload": 10.0,
            }
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def verify_existing(checksum_path: Path) -> None:
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        target = checksum_path.parent / name.strip()
        actual = sha256_file(target)
        if actual != expected:
            raise SystemExit(f"frozen manifest checksum mismatch: {target} expected={expected} actual={actual}")
    print(f"[OK] existing frozen manifests verified: {checksum_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_jsonl", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--split_seed", type=int, default=20260722)
    parser.add_argument("--source_commit", default="")
    args = parser.parse_args()

    output = Path(args.outdir).resolve()
    checksum_path = output / "FROZEN_SHA256SUMS.txt"
    if checksum_path.exists():
        verify_existing(checksum_path)
        return 0
    output.mkdir(parents=True, exist_ok=True)

    sft_path = Path(args.sft_jsonl).resolve()
    grid = unique_tasks(default_taskset())
    if len(grid) != 77:
        raise SystemExit(f"expected Grid77 to contain 77 tasks, found {len(grid)}")
    train, guard = stratified_guard_split(grid, int(args.split_seed))
    test = build_unseen_tasks()
    if (len(train), len(guard), len(test)) != (69, 8, 40):
        raise SystemExit(f"unexpected split sizes: train={len(train)} guard={len(guard)} test={len(test)}")

    ensure_disjoint(train, guard, test)
    grid_keys = {task_key(task) for task in grid}
    sft_keys, sft_rows, sft_matched_rows = extract_sft_tasks(sft_path)
    test_keys = {task_key(task) for task in test}
    if test_keys.intersection(grid_keys):
        raise SystemExit(f"test/Grid77 overlap: {sorted(test_keys.intersection(grid_keys))}")
    if test_keys.intersection(sft_keys):
        raise SystemExit(f"test/SFT overlap: {sorted(test_keys.intersection(sft_keys))}")

    ranges = empirical_ranges(grid)
    validate_test_ranges(test, ranges)
    if Counter(task.family for task in guard) != Counter({family: 2 for family in FAMILIES}):
        raise SystemExit("guard split is not stratified at two tasks per family")
    if Counter(task.family for task in test) != Counter({family: 10 for family in FAMILIES}):
        raise SystemExit("test split is not stratified at ten tasks per family")

    train_path = output / "train_tasks.jsonl"
    guard_path = output / "guard_tasks.jsonl"
    test_path = output / "unseen_test_tasks.jsonl"
    meta_path = output / "manifest_meta.json"
    write_tasks(train_path, train, "train")
    write_tasks(guard_path, guard, "guard")
    write_tasks(test_path, test, "unseen_test")

    source_commit = str(args.source_commit).strip()
    if not source_commit:
        try:
            source_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parent), text=True
            ).strip()
        except Exception:
            source_commit = "unknown"

    metadata = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "split_seed": int(args.split_seed),
        "split_method": "two lowest SHA-256 ranks per family are guard tasks",
        "counts": {"grid77": 77, "train": 69, "guard": 8, "unseen_test": 40},
        "family_counts": {
            "train": dict(sorted(Counter(task.family for task in train).items())),
            "guard": dict(sorted(Counter(task.family for task in guard).items())),
            "unseen_test": dict(sorted(Counter(task.family for task in test).items())),
        },
        "grid77_ranges": ranges,
        "test_construction": (
            "ten matched in-range voltage pairs; buck uses high-to-low and boost/SEPIC/buckboost use low-to-high"
        ),
        "sft": {
            "path_at_freeze": str(sft_path),
            "rows": int(sft_rows),
            "rows_with_parsed_dcdc_spec": int(sft_matched_rows),
            "unique_parsed_dcdc_tasks": int(len(sft_keys)),
            "sha256": sha256_file(sft_path),
        },
        "overlaps": {
            "train_guard": 0,
            "test_grid77": 0,
            "test_sft_parsed_specs": 0,
        },
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    frozen_files = [train_path, guard_path, test_path, meta_path]
    checksum_lines = [f"{sha256_file(path)}  {path.name}" for path in frozen_files]
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    for path in [*frozen_files, checksum_path]:
        os.chmod(path, 0o444)

    print(json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True))
    print(f"[OK] frozen manifests: {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

from dcdc_taskset import Task


VALID_FAMILIES = {"buck", "boost", "sepic", "buckboost"}
TaskKey = Tuple[str, float, float]


def task_key(task: Task) -> TaskKey:
    return (str(task.family).strip().lower(), float(task.vin), float(task.vout))


def validate_task(task: Task) -> None:
    family, vin, vout = task_key(task)
    if family not in VALID_FAMILIES:
        raise ValueError(f"unsupported family: {family!r}")
    if not (math.isfinite(vin) and math.isfinite(vout) and vin > 0.0 and vout > 0.0):
        raise ValueError(f"invalid voltages: family={family} vin={vin} vout={vout}")
    if family == "buck" and not (vout < vin):
        raise ValueError(f"buck requires vout < vin: vin={vin} vout={vout}")
    if family == "boost" and not (vout > vin):
        raise ValueError(f"boost requires vout > vin: vin={vin} vout={vout}")
    if family in {"sepic", "buckboost"} and math.isclose(vin, vout, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{family} manifest excludes equal-voltage tasks: vin={vin} vout={vout}")


def unique_tasks(tasks: Iterable[Task]) -> List[Task]:
    result: List[Task] = []
    seen: Set[TaskKey] = set()
    for task in tasks:
        validate_task(task)
        key = task_key(task)
        if key in seen:
            raise ValueError(f"duplicate task: {key}")
        seen.add(key)
        result.append(Task(key[0], key[1], key[2]))
    if not result:
        raise ValueError("task manifest is empty")
    return result


def load_tasks_jsonl(path: str | Path) -> List[Task]:
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"task manifest not found: {manifest}")
    tasks: List[Task] = []
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            tasks.append(Task(str(record["family"]), float(record["vin"]), float(record["vout"])))
        except Exception as exc:
            raise ValueError(f"invalid task record at {manifest}:{line_no}: {exc}") from exc
    return unique_tasks(tasks)


def ensure_disjoint(*groups: Sequence[Task]) -> None:
    seen: Set[TaskKey] = set()
    for group_index, group in enumerate(groups):
        keys = {task_key(task) for task in group}
        overlap = seen.intersection(keys)
        if overlap:
            raise ValueError(f"task groups overlap before group {group_index}: {sorted(overlap)}")
        seen.update(keys)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

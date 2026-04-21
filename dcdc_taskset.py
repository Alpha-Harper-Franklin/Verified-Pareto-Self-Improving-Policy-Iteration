from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class Task:
    family: str
    vin: float
    vout: float


def _uniq(tasks: Iterable[Task]) -> List[Task]:
    seen = set()
    out: List[Task] = []
    for t in tasks:
        key = (str(t.family), float(t.vin), float(t.vout))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def default_taskset() -> List[Task]:
    out: List[Task] = []

    # Buck: vout < vin
    vin_buck = [9.0, 12.0, 15.0, 18.0, 24.0]
    vout_all = [1.8, 3.3, 5.0, 8.0, 12.0, 15.0, 18.0]
    for vin in vin_buck:
        for vout in vout_all:
            if vout < vin:
                out.append(Task("buck", vin, vout))

    # Boost: vout > vin
    vin_boost = [3.3, 5.0, 9.0, 12.0]
    vout_boost = [9.0, 12.0, 15.0, 18.0, 24.0]
    for vin in vin_boost:
        for vout in vout_boost:
            if vout > vin:
                out.append(Task("boost", vin, vout))

    # SEPIC: non-inverting step-up/down
    vin_mid = [5.0, 9.0, 12.0, 15.0]
    vout_mid = [3.3, 5.0, 8.0, 12.0, 15.0]
    for vin in vin_mid:
        for vout in vout_mid:
            if abs(vout - vin) > 1e-6:
                out.append(Task("sepic", vin, vout))

    # Buck-Boost: non-inverting (cascaded) step-up/down
    for vin in vin_mid:
        for vout in vout_mid:
            if abs(vout - vin) > 1e-6:
                out.append(Task("buckboost", vin, vout))

    return _uniq(out)

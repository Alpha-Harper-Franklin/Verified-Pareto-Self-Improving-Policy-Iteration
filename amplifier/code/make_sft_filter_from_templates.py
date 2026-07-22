#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


RESPONSE_TEMPLATE_DEFAULT = "### Response:\n"


@dataclass(frozen=True)
class FilterTask:
    family: str
    f_hz: float
    y: float


def _fmt(x: float) -> str:
    try:
        return f"{float(x):.6g}"
    except Exception:
        return "0"


def _loguniform(rng: random.Random, lo: float, hi: float) -> float:
    lo = float(lo)
    hi = float(hi)
    if lo <= 0 or hi <= 0:
        return float(max(lo, 1e-9))
    if hi < lo:
        lo, hi = hi, lo
    return float(10 ** rng.uniform(math.log10(lo), math.log10(hi)))


def _task_grid_filter() -> List[FilterTask]:
    out: List[FilterTask] = []
    freqs = [1e4, 2e4, 3e4, 5e4, 1e5]
    atten = [20.0]

    for fc in freqs:
        for att in atten:
            out.append(FilterTask("filter_lpf", float(fc), float(att)))
            out.append(FilterTask("filter_hpf", float(fc), float(att)))

    bw_ratios = [0.20]
    for f0 in freqs:
        for r in bw_ratios:
            out.append(FilterTask("filter_bpf", float(f0), float(f0 * r)))

    for f0 in freqs:
        for att in atten:
            out.append(FilterTask("filter_notch", float(f0), float(att)))

    return out


def _build_prompt(task: FilterTask, *, min_elems: int, response_template: str) -> str:
    fam = str(task.family).strip().lower()
    if fam in {"filter_lpf", "filter_hpf"}:
        spec = f"fc_target={task.f_hz:.6g} Hz, stopband_target={task.y:.1f} dB"
    elif fam == "filter_bpf":
        spec = f"f0_target={task.f_hz:.6g} Hz, bw_target={task.y:.6g} Hz"
    elif fam == "filter_notch":
        spec = f"f0_target={task.f_hz:.6g} Hz, notch_depth_target={task.y:.1f} dB"
    else:
        spec = f"f_target={task.f_hz:.6g}"

    body = f"""Generate a {fam} circuit in INC DSL.

Rules:
- Output ONLY INC lines (no explanation).
- Line format: INC <name> <node1> <node2> <value>
- Use ONLY passive elements {{R,C,L}}.
- Allowed nodes: {{vin, out, 0}} and helper nodes n1,n2,... ONLY.
- DO NOT use numeric node names like 12 or 5 (only ground '0' is numeric).
- Use at least {int(min_elems)} INC lines (>= {int(min_elems)} elements).

Specs:
- {spec}
"""
    return body + str(response_template)


def _new_name(prefix: str, used: set[str]) -> str:
    p = str(prefix or "X").strip().upper() or "X"
    for i in range(1, 100_000):
        cand = f"{p}{i}"
        if cand not in used:
            used.add(cand)
            return cand
    raise RuntimeError("name_exhausted")


def _pad_rc_network(lines: List[str], *, used: set[str], rng: random.Random, node_pool: List[str], min_elems: int) -> None:
    while len(lines) < int(min_elems):
        kind = rng.choices(["R", "C", "L"], weights=[0.45, 0.45, 0.10], k=1)[0]
        name = _new_name(kind, used)
        a, b = rng.sample(node_pool, 2)
        if a == b:
            continue
        if kind == "R":
            val = float(_loguniform(rng, 10.0, 1e6))
        elif kind == "C":
            val = float(_loguniform(rng, 1e-13, 1e-6))
        else:
            val = float(_loguniform(rng, 1e-9, 1e-3))
        lines.append(f"INC {name} {a} {b} {_fmt(val)}")


def _gen_netlist(task: FilterTask, *, min_elems: int, rng: random.Random) -> str:
    fam = str(task.family).strip().lower()
    f = max(1.0, float(task.f_hz))

    lines: List[str] = []
    used: set[str] = set()

    helpers = [f"n{i}" for i in range(1, max(10, int(min_elems) + 1))]
    node_pool = ["vin", "out", "0", *helpers]

    # Choose a base R around 100..10k.
    R0 = float(_loguniform(rng, 100.0, 10_000.0))
    C0 = float(min(max(1.0 / (2.0 * math.pi * R0 * f), 1e-13), 1e-6))

    if fam == "filter_lpf":
        # Simple RC low-pass, then add ladder stages.
        lines.append(f"INC RIN vin out {_fmt(R0)}")
        lines.append(f"INC CSH out 0 {_fmt(C0)}")
        used.update({"RIN", "CSH"})

        # Add extra ladder stages (vin -> n1 -> ... -> out)
        prev = "vin"
        stages = max(6, int(min_elems) // 2)
        for i in range(1, stages + 1):
            node = helpers[i - 1] if i <= len(helpers) else "out"
            if i == stages:
                node = "out"
            r = float(_loguniform(rng, 50.0, 50_000.0))
            c = float(min(max(1.0 / (2.0 * math.pi * r * f), 1e-13), 1e-6))
            lines.append(f"INC {_new_name('R', used)} {prev} {node} {_fmt(r)}")
            lines.append(f"INC {_new_name('C', used)} {node} 0 {_fmt(c)}")
            prev = node

    elif fam == "filter_hpf":
        # Simple CR high-pass.
        lines.append(f"INC CIN vin out {_fmt(C0)}")
        lines.append(f"INC RSH out 0 {_fmt(R0)}")
        used.update({"CIN", "RSH"})

        prev = "vin"
        stages = max(6, int(min_elems) // 2)
        for i in range(1, stages + 1):
            node = helpers[i - 1] if i <= len(helpers) else "out"
            if i == stages:
                node = "out"
            r = float(_loguniform(rng, 50.0, 50_000.0))
            c = float(min(max(1.0 / (2.0 * math.pi * r * f), 1e-13), 1e-6))
            lines.append(f"INC {_new_name('C', used)} {prev} {node} {_fmt(c)}")
            lines.append(f"INC {_new_name('R', used)} {node} 0 {_fmt(r)}")
            prev = node

    elif fam == "filter_bpf":
        f0 = f
        bw = max(1.0, float(task.y))
        # Series RLC band-pass approximation.
        C = float(_loguniform(rng, 1e-12, 1e-8))
        L = float(1.0 / ((2.0 * math.pi * f0) ** 2 * max(1e-18, C)))
        L = float(min(max(L, 1e-9), 1e-3))
        R = float(2.0 * math.pi * L * bw)
        R = float(min(max(R, 1.0), 1e6))

        # vin -- C -- n1 -- L -- out, with series R.
        n1 = helpers[0]
        lines.append(f"INC C1 vin {n1} {_fmt(C)}")
        lines.append(f"INC L1 {n1} out {_fmt(L)}")
        lines.append(f"INC R1 {n1} out {_fmt(R)}")
        used.update({"C1", "L1", "R1"})

        # Add a few shunt elements around the resonator.
        lines.append(f"INC {_new_name('C', used)} out 0 {_fmt(_loguniform(rng, 1e-13, 1e-8))}")
        lines.append(f"INC {_new_name('R', used)} out 0 {_fmt(_loguniform(rng, 10.0, 1e6))}")

    elif fam == "filter_notch":
        f0 = f
        # Notch via parallel LC to ground at out.
        C = float(_loguniform(rng, 1e-12, 1e-8))
        L = float(1.0 / ((2.0 * math.pi * f0) ** 2 * max(1e-18, C)))
        L = float(min(max(L, 1e-9), 1e-3))
        Rser = float(_loguniform(rng, 10.0, 10_000.0))
        lines.append(f"INC RIN vin out {_fmt(Rser)}")
        lines.append(f"INC LNOT out 0 {_fmt(L)}")
        lines.append(f"INC CNOT out 0 {_fmt(C)}")
        used.update({"RIN", "LNOT", "CNOT"})

        # Add a damping resistor to control depth.
        lines.append(f"INC RDM out 0 {_fmt(_loguniform(rng, 10.0, 1e6))}")
        used.add("RDM")

    else:
        # Fallback: minimal RC pass-through.
        lines.append(f"INC RIN vin out {_fmt(R0)}")
        lines.append(f"INC CSH out 0 {_fmt(C0)}")
        used.update({"RIN", "CSH"})

    # Pad to min_elems using allowed nodes only.
    _pad_rc_network(lines, used=used, rng=rng, node_pool=node_pool, min_elems=int(min_elems))
    return "\n".join(lines).strip() + "\n"


def _iter_records(*, n: int, min_elems: int, seed: int, response_template: str) -> Iterable[dict]:
    rng = random.Random(int(seed))
    tasks = _task_grid_filter()
    rng.shuffle(tasks)

    for i in range(int(n)):
        if i > 0 and (i % len(tasks) == 0):
            rng.shuffle(tasks)
        t = tasks[i % len(tasks)]
        prompt = _build_prompt(t, min_elems=int(min_elems), response_template=str(response_template))
        netlist = _gen_netlist(t, min_elems=int(min_elems), rng=rng)
        yield {"text": prompt + netlist}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--min_elems", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260114)
    ap.add_argument("--response_template", default=RESPONSE_TEMPLATE_DEFAULT)
    args = ap.parse_args()

    out_path = Path(str(args.out_jsonl)).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in _iter_records(
            n=int(args.n),
            min_elems=int(args.min_elems),
            seed=int(args.seed),
            response_template=str(args.response_template),
        ):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(json.dumps({"out_jsonl": str(out_path), "n": int(args.n), "min_elems": int(args.min_elems), "seed": int(args.seed)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
class OscTask:
    family: str
    f_hz: float
    vpp: float


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


def _logspace(lo: float, hi: float, n: int) -> List[float]:
    lo = float(lo)
    hi = float(hi)
    n = int(n)
    if n <= 1:
        return [float(f"{lo:.6g}")]
    ratio = (hi / lo) ** (1.0 / float(n - 1))
    out: List[float] = []
    f = lo
    for _ in range(n):
        out.append(float(f"{f:.6g}"))
        f *= ratio
    out[0] = float(f"{lo:.6g}")
    out[-1] = float(f"{hi:.6g}")
    uniq: List[float] = []
    for x in out:
        if not uniq or float(x) != float(uniq[-1]):
            uniq.append(float(x))
    return uniq


def _task_grid_osc() -> List[OscTask]:
    out: List[OscTask] = []
    vpp = 1.0

    # Reduced-difficulty curriculum: 5 frequency targets per family.
    for f in [1e4, 2e4, 5e4, 1e5, 2e5]:
        out.append(OscTask("osc_lc", float(f), float(vpp)))
    for f in [1e3, 2e3, 5e3, 1e4, 2e4]:
        out.append(OscTask("osc_rc", float(f), float(vpp)))
    for f in [1e5, 2e5, 5e5, 1e6, 2e6]:
        out.append(OscTask("osc_ring", float(f), float(vpp)))
    for f in [1e3, 2e3, 5e3, 1e4, 2e4]:
        out.append(OscTask("osc_wien", float(f), float(vpp)))

    return out


def _build_prompt(task: OscTask, *, min_elems: int, response_template: str) -> str:
    fam = str(task.family).strip().lower()
    body = f"""Generate an oscillator passive network in INC DSL.

Rules:
- Output ONLY INC lines (no explanation).
- Line format: INC <name> <node1> <node2> <value>
- Use ONLY passive elements {{R,C,L}}.
- Allowed nodes: {{out, vdd, 0}} and helper nodes n1,n2,... ONLY.
- DO NOT use numeric node names like 12 or 5 (only ground '0' is numeric).
- Use at least {int(min_elems)} INC lines (>= {int(min_elems)} elements).

Specs:
- family={fam}
- f_target={float(task.f_hz):.6g} Hz
- vpp_hint={float(task.vpp):.2f} V
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


def _pad(lines: List[str], *, used: set[str], rng: random.Random, node_pool: List[str], min_elems: int, allow_L: bool) -> None:
    while len(lines) < int(min_elems):
        kinds = ["R", "C", "L"] if allow_L else ["R", "C"]
        weights = [0.5, 0.5, 0.0] if not allow_L else [0.45, 0.45, 0.10]
        kind = rng.choices(kinds, weights=weights[: len(kinds)], k=1)[0]
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


def _gen_netlist(task: OscTask, *, min_elems: int, rng: random.Random) -> str:
    fam = str(task.family).strip().lower()
    f = max(1.0, float(task.f_hz))

    lines: List[str] = []
    used: set[str] = set()

    helpers = [f"n{i}" for i in range(1, max(10, int(min_elems) + 1))]
    node_pool = ["out", "vdd", "0", *helpers]

    # Always include a VDD decoupling element so 'vdd' appears in INC nodes.
    cdd = float(_loguniform(rng, 1e-12, 1e-8))
    lines.append(f"INC CDD vdd 0 {_fmt(cdd)}")
    used.add("CDD")

    if fam == "osc_lc":
        C = float(_loguniform(rng, 1e-12, 1e-8))
        L = float(1.0 / ((2.0 * math.pi * f) ** 2 * max(1e-18, C)))
        L = float(min(max(L, 1e-9), 1e-3))
        n1 = helpers[0]
        lines.append(f"INC L1 out {n1} {_fmt(L)}")
        lines.append(f"INC C1 {n1} 0 {_fmt(C)}")
        used.update({"L1", "C1"})
        # damping / load
        lines.append(f"INC R1 out 0 {_fmt(_loguniform(rng, 100.0, 100_000.0))}")
        used.add("R1")
        allow_L = True

    elif fam == "osc_rc":
        R = float(_loguniform(rng, 100.0, 100_000.0))
        C = float(min(max(1.0 / (2.0 * math.pi * R * f), 1e-13), 1e-6))
        n1 = helpers[0]
        lines.append(f"INC R1 out {n1} {_fmt(R)}")
        lines.append(f"INC C1 {n1} 0 {_fmt(C)}")
        used.update({"R1", "C1"})
        allow_L = False

    elif fam == "osc_wien":
        # Wien bridge: f0 ~ 1/(2*pi*R*C)
        R = float(_loguniform(rng, 100.0, 100_000.0))
        C = float(min(max(1.0 / (2.0 * math.pi * R * f), 1e-13), 1e-6))
        n1, n2 = helpers[0], helpers[1]
        lines.append(f"INC R1 out {n1} {_fmt(R)}")
        lines.append(f"INC C1 {n1} {n2} {_fmt(C)}")
        lines.append(f"INC R2 {n2} 0 {_fmt(R)}")
        lines.append(f"INC C2 {n2} 0 {_fmt(C)}")
        used.update({"R1", "C1", "R2", "C2"})
        allow_L = False

    else:  # osc_ring
        # RC ring-like ladder on helper nodes.
        n1, n2 = helpers[0], helpers[1]
        R = float(_loguniform(rng, 50.0, 10_000.0))
        C = float(min(max(1.0 / (2.0 * math.pi * R * f), 1e-13), 1e-6))
        lines.append(f"INC R1 out {n1} {_fmt(R)}")
        lines.append(f"INC C1 {n1} 0 {_fmt(C)}")
        lines.append(f"INC R2 {n1} {n2} {_fmt(R)}")
        lines.append(f"INC C2 {n2} 0 {_fmt(C)}")
        lines.append(f"INC R3 {n2} out {_fmt(R)}")
        lines.append(f"INC C3 out 0 {_fmt(C)}")
        used.update({"R1", "C1", "R2", "C2", "R3", "C3"})
        allow_L = False

    _pad(lines, used=used, rng=rng, node_pool=node_pool, min_elems=int(min_elems), allow_L=bool(allow_L))
    return "\n".join(lines).strip() + "\n"


def _iter_records(*, n: int, min_elems: int, seed: int, response_template: str) -> Iterable[dict]:
    rng = random.Random(int(seed))
    tasks = _task_grid_osc()
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

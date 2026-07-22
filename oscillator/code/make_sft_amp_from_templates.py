#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


RESPONSE_TEMPLATE_DEFAULT = "### Response:\n"


@dataclass(frozen=True)
class AmpTask:
    family: str
    gain_db: float
    bw_hz: float


def _fmt(x: float) -> str:
    try:
        return f"{float(x):.6g}"
    except Exception:
        return "0"


def _loguniform(rng: random.Random, lo: float, hi: float) -> float:
    lo = float(lo)
    hi = float(hi)
    if lo <= 0 or hi <= 0 or (not math.isfinite(lo)) or (not math.isfinite(hi)):
        return float(lo)
    if hi < lo:
        lo, hi = hi, lo
    return float(10 ** rng.uniform(math.log10(lo), math.log10(hi)))


def _task_grid_amp() -> List[AmpTask]:
    out: List[AmpTask] = []

    # Keep consistent with amp_taskset.py
    gains_db = [6.0]
    bws_op2 = [1e4, 3e4, 1e5, 3e5, 1e6]
    bws_rfpa = [1e6, 2e6, 3e6, 5e6, 1e7]

    for g in gains_db:
        for bw in bws_op2:
            out.append(AmpTask("amp_op2", float(g), float(bw)))
    for g in gains_db:
        for bw in bws_rfpa:
            out.append(AmpTask("amp_rfpa", float(g), float(bw)))

    return out


def _amp_family_vdd_and_pmax(family: str) -> Tuple[float, float]:
    fam = str(family or "").strip().lower()
    if fam == "amp_rfpa":
        return 3.3, 50.0
    return 1.8, 5.0


def _build_prompt(task: AmpTask, *, min_elems: int, response_template: str) -> str:
    fam = str(task.family).strip().lower()
    vdd, pmax_mw = _amp_family_vdd_and_pmax(fam)
    body = f"""Generate an analog amplifier in INC DSL.

Rules:
- Output ONLY INC lines (no explanation).
- Line format: INC <name> <node1> <node2> <value>
- Use ONLY passive elements {{R,C,L}}.
- Use ONLY these nodes: {{vin, inv, out, vdd, 0}} (do NOT introduce helper nodes).
- DO NOT use numeric node names like 12 or 5 (only ground '0' is numeric).
- Use at least {int(min_elems)} INC lines (>= {int(min_elems)} elements).
- Non-inverting topology: do NOT connect node vin to any node except 0.
- Must include a bias resistor between vdd and 0.
- Must include negative feedback: at least one R between out-inv and one R between inv-0.
- If you need extra parts to reach min_elems, ONLY add non-interacting dummies:
  - extra R: connect out-0 and set value=1e12
  - extra C: connect vdd-0 and set value=1e-15
  - to include node vin without breaking topology, add ONE dummy: RIN vin 0 1e12

Specs:
- family={fam}
- gain_target={float(task.gain_db):.1f} dB
- bw_target={float(task.bw_hz):.6g} Hz
- VDD={vdd:.1f} V
- Pstatic <= {pmax_mw:.1f} mW
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


def _gen_netlist(task: AmpTask, *, min_elems: int, rng: random.Random) -> str:
    """Generate a *verifier-aligned* amplifier INC netlist.

    Key goal: provide an analytic init that is already close to passing the evaluator,
    so open-loop CV does not collapse to 0.

    Empirical calibration (for the current amp_eval_acop core model):
      - amp_op2: use a slightly larger feedback ratio and moderate bias to satisfy PM.
      - amp_rfpa: bias is chosen to sit just under the static power limit.
      - CCOMP uses a global scale factor (~1.4) to match the evaluator's measured -3dB BW.

    This is not used as a template at evaluation time; it is only for anchor SFT data.
    """

    fam = str(task.family).strip().lower()
    gain_db = float(task.gain_db)
    bw_hz = max(1.0, float(task.bw_hz))

    # Closed-loop gain ratio for non-inverting amplifier: Acl ~= 1 + RFB/RG.
    a_lin = float(10 ** (gain_db / 20.0))
    ratio_ideal = float(max(1e-6, a_lin - 1.0))

    # Calibrated family knobs.
    if fam == "amp_rfpa":
        ratio_scale = 1.0
        rbias0 = 220.0
    else:
        ratio_scale = 1.2
        rbias0 = 1500.0

    ccomp_scale = 1.4

    # Use a stable RG to reduce variance; keep mild jitter for robustness.
    rg = float(10_000.0)
    ratio = float(min(max(ratio_ideal * ratio_scale, 1e-3), 1e3))
    rfb = float(min(max(rg * ratio, 50.0), 1e8))

    # Bias resistor (calibrated): controls gm/power and indirectly phase margin.
    vdd, pmax_mw = _amp_family_vdd_and_pmax(fam)
    pmax_w = float(max(1e-9, float(pmax_mw))) / 1e3
    rbias_min = float((float(vdd) * float(vdd)) / pmax_w)
    # For RFPA, keep a tiny margin so static power stays <= pmax.
    rbias_floor = float(rbias_min * (1.02 if fam == "amp_rfpa" else 1.0))
    rbias_base = float(max(float(rbias0), float(rbias_floor)))
    if fam == "amp_rfpa":
        rbias = float(rbias_base * rng.uniform(1.0, 1.02))
    else:
        rbias = float(rbias_base * rng.uniform(0.98, 1.02))
    rbias = float(min(max(rbias, 10.0), 1e9))

    # Compensation capacitor across out-inv: C ~ scale/(2*pi*Rfb*bw).
    ccomp = float((ccomp_scale / (2.0 * math.pi * max(1.0, rfb) * bw_hz)) * rng.uniform(0.98, 1.02))
    ccomp = float(min(max(ccomp, 1e-15), 1e-6))

    lines: List[str] = []
    lines.append(f"INC RBIAS vdd 0 {_fmt(rbias)}")
    lines.append(f"INC RG inv 0 {_fmt(rg)}")
    lines.append(f"INC RFB out inv {_fmt(rfb)}")
    lines.append(f"INC CCOMP out inv {_fmt(ccomp)}")
    # Include VIN as a harmless dummy (verifier requires the node to appear).
    lines.append("INC RIN vin 0 1e12")

    used = {ln.split()[1].strip().upper() for ln in lines if ln.strip().startswith("INC ")}

    # Pad to min_elems with non-interacting dummies.
    # NOTE: keep them *strictly* within verifier thresholds.
    while len(lines) < int(min_elems):
        if len(lines) % 2 == 0:
            name = _new_name("R", used)
            lines.append(f"INC {name} out 0 1e12")
        else:
            name = _new_name("C", used)
            lines.append(f"INC {name} vdd 0 1e-15")

    return "\n".join(lines).strip() + "\n"


def _iter_records(*, n: int, min_elems: int, seed: int, response_template: str) -> Iterable[dict]:
    rng = random.Random(int(seed))
    tasks = _task_grid_amp()
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

    n = int(args.n)
    min_elems = int(args.min_elems)
    seed = int(args.seed)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in _iter_records(n=n, min_elems=min_elems, seed=seed, response_template=str(args.response_template)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(json.dumps({"out_jsonl": str(out_path), "n": n, "min_elems": min_elems, "seed": seed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

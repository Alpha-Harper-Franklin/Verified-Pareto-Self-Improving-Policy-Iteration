from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

from inc_parser import IncElem, parse_inc


@dataclass(frozen=True)
class SpiceBuildMeta:
    vin_node: str
    out_node: str
    duty: float
    freq: float
    rload: float


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _infer_vin_node(elems: List[IncElem]) -> str:
    nodes = {n.lower() for e in elems for n in e.nodes}
    if "vin" in nodes:
        return "vin"
    if "in" in nodes:
        return "in"
    return elems[0].nodes[0] if elems else "vin"


def build_buck_spice(
    inc_text: str,
    vin: float,
    vout: float,
    rload: float = 10.0,
    freq: float = 200_000.0,
    tstop: float = 12e-3,
) -> Tuple[str, SpiceBuildMeta]:
    elems = parse_inc(inc_text)
    vin_node = _infer_vin_node(elems)
    out_node = "out"

    Rval = max(1e-2, float(rload))
    Iout = max(1e-6, float(vout) / max(1e-6, Rval))
    Rsw = 3e-3
    Rd = 5e-3
    Vd = 0.30
    num = float(vout) + Vd + Iout * Rd
    den = max(1e-3, float(vin) - Iout * Rsw + Vd + Iout * Rd)
    duty = _clamp(num / den, 0.05, 0.95)

    freq = _clamp(float(freq), 150e3, 600e3)
    per = 1.0 / max(1.0, freq)
    ton = duty * per

    diode_models: Set[str] = {"Dstd"}
    switch_models: Set[str] = {"Sstd"}
    for e in elems:
        if e.kind == "D" and e.model:
            diode_models.add(e.model)
        if e.kind == "S" and e.model:
            switch_models.add(e.model)

    lines: List[str] = []
    lines.append("* buck tran test (topology-preserving)")
    lines.append(f"V_IN {vin_node} 0 {float(vin)}")
    lines.append(f"* gate drive ~{duty*100:.1f}% duty at {freq/1000:.1f} kHz")
    lines.append(f"Vgate gate 0 PULSE(0 5 0 1n 1n {ton:.6e} {per:.6e})")
    for m in sorted(switch_models):
        lines.append(f".model {m} sw (ron=3m roff=1e6 vt=2)")
    for m in sorted(diode_models):
        lines.append(f".model {m} D (IS=1e-12 RS=5m N=1 TT=3n CJO=3p M=0.5)")

    for e in elems:
        n1, n2 = e.nodes[0], e.nodes[1]
        if e.kind in {"R", "L", "C"}:
            if e.value is None:
                continue
            val = float(e.value)
            if e.kind == "L":
                val = _clamp(val, 5e-6, 300e-6)
            if e.kind == "C":
                val = _clamp(val, 5e-6, 330e-6)
            lines.append(f"{e.name} {n1} {n2} {val:.6e}")
        elif e.kind == "D":
            mdl = e.model or "Dstd"
            lines.append(f"{e.name} {n1} {n2} {mdl}")
        elif e.kind == "S":
            mdl = e.model or "Sstd"
            lines.append(f"{e.name} {n1} {n2} gate 0 {mdl}")

    lines.append(f"R_ENVLOAD {out_node} 0 {Rval:.6e}")
    lines.append(".control")
    lines.append("set filetype=ascii")
    lines.append(f"tran 0.1u {tstop:.6e}")
    lines.append("quit")
    lines.append(".endc")
    lines.append(".end")

    meta = SpiceBuildMeta(vin_node=vin_node, out_node=out_node, duty=float(duty), freq=float(freq), rload=float(Rval))
    return "\n".join(lines) + "\n", meta


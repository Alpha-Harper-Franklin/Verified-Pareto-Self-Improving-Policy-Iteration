from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from inc_parser import IncElem, parse_inc


@dataclass(frozen=True)
class SpiceBuildMeta:
    family: str
    vin_node: str
    out_node: str
    rload: float
    freq: float
    duty_map: Dict[str, float]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _infer_vin_node(elems: List[IncElem]) -> str:
    nodes = {n.lower() for e in elems for n in e.nodes}
    if "vin" in nodes:
        return "vin"
    if "in" in nodes:
        return "in"
    return elems[0].nodes[0] if elems else "vin"


def _gate_for_switch(model: Optional[str], name: str) -> str:
    m = (model or "").lower().strip()
    n = (name or "").lower().strip()
    if "g2" in m or m.endswith("2") or "_2" in n or n.endswith("2"):
        return "gate2"
    if "g1" in m or m.endswith("1") or "_1" in n or n.endswith("1"):
        return "gate1"
    return "gate"


def _diode_vf(i: float, *, is_: float, rs: float, n: float, vt: float = 0.02585) -> float:
    i = max(1e-12, float(i))
    is_ = max(1e-30, float(is_))
    rs = max(0.0, float(rs))
    n = max(0.1, float(n))
    vt = max(1e-6, float(vt))
    return (n * vt * math.log((i / is_) + 1.0)) + (i * rs)


def _duty_buck(vin: float, vout: float, rload: float, rsw: float, diode_is: float, diode_rs: float, diode_n: float) -> float:
    rval = max(1e-2, float(rload))
    iout = max(1e-6, float(vout) / max(1e-6, rval))
    vf = _diode_vf(iout, is_=float(diode_is), rs=float(diode_rs), n=float(diode_n))
    num = float(vout) + vf
    den = max(1e-3, float(vin) - iout * float(rsw) + vf)
    return _clamp(num / den, 0.05, 0.95)


def _duty_boost(vin: float, vout: float, rload: float, rsw: float, diode_is: float, diode_rs: float, diode_n: float) -> float:
    rval = max(1e-2, float(rload))
    iout = max(1e-6, float(vout) / max(1e-6, rval))

    rsw2 = float(rsw)
    vin2 = float(vin)
    vout2 = float(vout)

    # Use a short fixed-point iteration to account for higher inductor current: I_L ≈ Iout/(1-D).
    vf0 = _diode_vf(iout, is_=float(diode_is), rs=float(diode_rs), n=float(diode_n))
    d = 1.0 - (max(1e-6, vin2 - iout * rsw2) / max(1e-6, vout2 + vf0))
    for _ in range(4):
        d = _clamp(d, 0.05, 0.95)
        il = iout / max(1e-3, 1.0 - d)
        vf = _diode_vf(il, is_=float(diode_is), rs=float(diode_rs), n=float(diode_n))
        num = max(1e-6, vin2 - il * rsw2)
        den = max(1e-6, vout2 + vf)
        d = 1.0 - (num / den)
    return _clamp(d, 0.05, 0.95)


def _duty_sepic(vin: float, vout: float, rload: float, rsw: float, diode_is: float, diode_rs: float, diode_n: float) -> float:
    rval = max(1e-2, float(rload))
    iout = max(1e-6, float(vout) / max(1e-6, rval))

    rsw2 = float(rsw)
    vin2 = float(vin)
    vout2 = float(vout)

    # SEPIC has similar gain to a non-inverting buck-boost: Vout ≈ Vin * D/(1-D) - Vf.
    # Approximate the switch/diode current with I_L ≈ Iout/(1-D) and iterate a few times.
    vf0 = _diode_vf(iout, is_=float(diode_is), rs=float(diode_rs), n=float(diode_n))
    vin_eff = max(1e-6, vin2 - iout * rsw2)
    vout_eff = max(1e-6, vout2 + vf0)
    d = vout_eff / (vin_eff + vout_eff)
    for _ in range(4):
        d = _clamp(d, 0.05, 0.95)
        il = iout / max(1e-3, 1.0 - d)
        vf = _diode_vf(il, is_=float(diode_is), rs=float(diode_rs), n=float(diode_n))
        vin_eff = max(1e-6, vin2 - il * rsw2)
        vout_eff = max(1e-6, vout2 + vf)
        d = vout_eff / (vin_eff + vout_eff)
    return _clamp(d, 0.05, 0.95)


def _duty_buckboost_pair(
    vin: float, vout: float, rload: float, rsw: float, diode_is: float, diode_rs: float, diode_n: float
) -> Tuple[float, float]:
    # Cascaded buck->boost has an intermediate node `mid` that must be <= vin (buck) and <= vout (so the boost stage
    # does not need to "step down", which it cannot). Pick a conservative mid close to the limiting rail so that duties
    # stay in-range for both step-up and step-down regimes.
    vin2 = max(1e-6, float(vin))
    vout2 = max(1e-6, float(vout))

    if vout2 >= vin2:
        # step-up overall: keep buck duty away from 1.0 to avoid pathological switching at stage1
        vmid = 0.8 * vin2
    else:
        # step-down overall: stage2 is still boost, so vmid must be slightly below vout
        vmid = 0.95 * vout2

    vmid = _clamp(vmid, 0.05 * min(vin2, vout2), 0.95 * min(vin2, vout2))
    d1 = _duty_buck(vin, vmid, rload=rload, rsw=rsw, diode_is=diode_is, diode_rs=diode_rs, diode_n=diode_n)
    d2 = _duty_boost(vmid, vout, rload=rload, rsw=rsw, diode_is=diode_is, diode_rs=diode_rs, diode_n=diode_n)
    return d1, d2


def build_dcdc_spice(
    inc_text: str,
    family: str,
    vin: float,
    vout: float,
    rload: float = 10.0,
    freq: float = 200_000.0,
    tstop: float = 12e-3,
    duty_override: Optional[float] = None,
    duty1_override: Optional[float] = None,
    duty2_override: Optional[float] = None,
) -> Tuple[str, SpiceBuildMeta]:
    fam = (family or "").strip().lower()
    elems = parse_inc(inc_text)
    vin_node = _infer_vin_node(elems)
    out_node = "out"

    rval = max(1e-2, float(rload))

    # Keep the duty estimate consistent with the actual SPICE models below.
    # Dstd: IS=1e-12, RS=5m, N=1  => Vf ~0.7V @ ~0.5A load.
    rsw = 3e-3
    diode_is = 1e-12
    diode_rs = 5e-3
    diode_n = 1.0

    if fam == "buck":
        duty = (
            float(duty_override)
            if duty_override is not None
            else _duty_buck(vin, vout, rval, rsw, diode_is=diode_is, diode_rs=diode_rs, diode_n=diode_n)
        )
        duty_map = {"gate": duty}
    elif fam == "boost":
        duty = (
            float(duty_override)
            if duty_override is not None
            else _duty_boost(vin, vout, rval, rsw, diode_is=diode_is, diode_rs=diode_rs, diode_n=diode_n)
        )
        duty_map = {"gate": duty}
    elif fam == "sepic":
        duty = (
            float(duty_override)
            if duty_override is not None
            else _duty_sepic(vin, vout, rval, rsw, diode_is=diode_is, diode_rs=diode_rs, diode_n=diode_n)
        )
        duty_map = {"gate": duty}
    elif fam in {"buckboost", "buck-boost", "bb"}:
        d1, d2 = _duty_buckboost_pair(vin, vout, rval, rsw, diode_is=diode_is, diode_rs=diode_rs, diode_n=diode_n)
        if duty_override is not None:
            d1 = float(duty_override)
            d2 = float(duty_override)
        if duty1_override is not None:
            d1 = float(duty1_override)
        if duty2_override is not None:
            d2 = float(duty2_override)
        duty_map = {"gate1": _clamp(d1, 0.05, 0.95), "gate2": _clamp(d2, 0.05, 0.95)}
    else:
        raise ValueError(f"unknown family: {family}")

    freq2 = _clamp(float(freq), 150e3, 600e3)
    per = 1.0 / max(1.0, freq2)

    diode_models: Set[str] = {"Dstd"}
    switch_models: Set[str] = {"Sstd"}
    for e in elems:
        if e.kind == "D" and e.model:
            diode_models.add(e.model)
        if e.kind == "S" and e.model:
            switch_models.add(e.model)

    gates_used: Set[str] = set()
    for e in elems:
        if e.kind == "S":
            gates_used.add(_gate_for_switch(e.model, e.name))
    if not gates_used:
        gates_used.add("gate")

    lines: List[str] = []
    lines.append(f"* dcdc tran test (family={fam})")
    lines.append(f"V_IN {vin_node} 0 {float(vin)}")

    for g in sorted(gates_used):
        duty_g = float(duty_map.get(g, duty_map.get("gate", 0.5)))
        ton = duty_g * per
        lines.append(f"* {g} ~{duty_g*100:.1f}% duty at {freq2/1000:.1f} kHz")
        lines.append(f"V{g} {g} 0 PULSE(0 5 0 1n 1n {ton:.6e} {per:.6e})")

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
                val = _clamp(val, 1e-6, 800e-6)
            if e.kind == "C":
                val = _clamp(val, 1e-9, 2000e-6)
            lines.append(f"{e.name} {n1} {n2} {val:.6e}")
        elif e.kind == "D":
            mdl = e.model or "Dstd"
            lines.append(f"{e.name} {n1} {n2} {mdl}")
        elif e.kind == "S":
            mdl = e.model or "Sstd"
            g = _gate_for_switch(e.model, e.name)
            lines.append(f"{e.name} {n1} {n2} {g} 0 {mdl}")

    lines.append(f"R_ENVLOAD {out_node} 0 {rval:.6e}")
    # Improve transient convergence for switching converters with large passive banks.
    lines.append(".options method=gear maxord=2 reltol=1e-3 vntol=1e-6 abstol=1e-12")
    lines.append(".control")
    lines.append("set filetype=ascii")
    lines.append(f"tran 0.1u {float(tstop):.6e}")
    lines.append("quit")
    lines.append(".endc")
    lines.append(".end")

    meta = SpiceBuildMeta(
        family=fam,
        vin_node=vin_node,
        out_node=out_node,
        rload=float(rval),
        freq=float(freq2),
        duty_map={k: float(v) for k, v in duty_map.items()},
    )
    return "\n".join(lines) + "\n", meta

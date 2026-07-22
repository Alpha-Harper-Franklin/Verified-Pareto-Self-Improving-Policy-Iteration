from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    ports: Sequence[str]
    expand: Callable[[Dict[str, str], int], List[str]]


def _capbank_lines(prefix: str, npos: str, nneg: str, values: Sequence[str], start_i: int = 1) -> List[str]:
    out: List[str] = []
    for i, v in enumerate(values, start=start_i):
        out.append(f"INC C{prefix}{i} {npos} {nneg} {v}")
    return out


def _rc_to_gnd(prefix: str, n: str, gnd: str, r: str, c: str, node_mid: str) -> List[str]:
    return [
        f"INC R{prefix} {n} {node_mid} {r}",
        f"INC C{prefix} {node_mid} {gnd} {c}",
    ]


def _buck_base(port: Dict[str, str], inst: int) -> List[str]:
    vin = port["vin"]
    sw = port["sw"]
    out = port["out"]
    gnd = port["0"]
    p = f"{inst}_"
    return [
        f"INC S{p}1 {vin} {sw} Sstd",
        f"INC D{p}1 {gnd} {sw} Dstd",
        f"INC L{p}1 {sw} {out} 47u",
        f"INC C{p}1 {out} {gnd} 47u",
    ]


def _boost_base(port: Dict[str, str], inst: int) -> List[str]:
    vin = port["vin"]
    sw = port["sw"]
    out = port["out"]
    gnd = port["0"]
    p = f"{inst}_"
    return [
        f"INC L{p}1 {vin} {sw} 47u",
        f"INC S{p}1 {sw} {gnd} Sstd",
        f"INC D{p}1 {sw} {out} Dstd",
        f"INC C{p}1 {out} {gnd} 47u",
    ]


def _sepic_base(port: Dict[str, str], inst: int) -> List[str]:
    vin = port["vin"]
    sw = port["sw"]
    n1 = port["n1"]
    out = port["out"]
    gnd = port["0"]
    p = f"{inst}_"
    return [
        f"INC L{p}1 {vin} {sw} 47u",
        f"INC C{p}1 {sw} {n1} 1u",
        f"INC L{p}2 {n1} {gnd} 47u",
        f"INC S{p}1 {sw} {gnd} Sstd",
        f"INC D{p}1 {n1} {out} Dstd",
        f"INC C{p}2 {out} {gnd} 47u",
    ]


def _buckboost_base(port: Dict[str, str], inst: int) -> List[str]:
    vin = port["vin"]
    sw1 = port["sw1"]
    mid = port["mid"]
    sw2 = port["sw2"]
    out = port["out"]
    gnd = port["0"]
    p = f"{inst}_"
    return [
        f"INC S{p}1 {vin} {sw1} Sstd1",
        f"INC D{p}1 {gnd} {sw1} Dstd",
        f"INC L{p}1 {sw1} {mid} 47u",
        f"INC C{p}1 {mid} {gnd} 47u",
        f"INC L{p}2 {mid} {sw2} 47u",
        f"INC S{p}2 {sw2} {gnd} Sstd2",
        f"INC D{p}2 {sw2} {out} Dstd",
        f"INC C{p}2 {out} {gnd} 47u",
    ]


def _capbank_in(port: Dict[str, str], inst: int) -> List[str]:
    vin = port["vin"]
    gnd = port["0"]
    p = f"{inst}_"
    return _capbank_lines(p, vin, gnd, ["47u", "22u", "10u", "4.7u", "1u", "0.1u"], start_i=1)


def _capbank_out(port: Dict[str, str], inst: int) -> List[str]:
    out = port["out"]
    gnd = port["0"]
    p = f"{inst}_"
    return _capbank_lines(p, out, gnd, ["47u", "22u", "10u", "4.7u", "1u", "0.1u"], start_i=1)


def _capbank_mid(port: Dict[str, str], inst: int) -> List[str]:
    mid = port["mid"]
    gnd = port["0"]
    p = f"{inst}_"
    return _capbank_lines(p, mid, gnd, ["47u", "10u"], start_i=1)


def _snubber_sw_gnd(port: Dict[str, str], inst: int) -> List[str]:
    sw = port["sw"]
    gnd = port["0"]
    node_mid = f"nsn_{inst}"
    p = f"{inst}_"
    return _rc_to_gnd(p, sw, gnd, r="5", c="1n", node_mid=node_mid)


def _snubber_sw_out(port: Dict[str, str], inst: int) -> List[str]:
    sw = port["sw"]
    out = port["out"]
    p = f"{inst}_"
    return [
        f"INC R{p}1 {sw} nsn_{inst} 50",
        f"INC C{p}1 nsn_{inst} {out} 100p",
    ]


def _damper_out(port: Dict[str, str], inst: int) -> List[str]:
    out = port["out"]
    gnd = port["0"]
    p = f"{inst}_"
    return [
        f"INC R{p}1 {out} nd_{inst} 0.2",
        f"INC C{p}1 nd_{inst} {gnd} 1u",
    ]


def default_module_specs() -> Dict[str, ModuleSpec]:
    specs = [
        ModuleSpec("BUCK_BASE", ports=["vin", "sw", "out", "0"], expand=_buck_base),
        ModuleSpec("BOOST_BASE", ports=["vin", "sw", "out", "0"], expand=_boost_base),
        ModuleSpec("SEPIC_BASE", ports=["vin", "sw", "n1", "out", "0"], expand=_sepic_base),
        ModuleSpec("BUCKBOOST_BASE", ports=["vin", "sw1", "mid", "sw2", "out", "0"], expand=_buckboost_base),
        ModuleSpec("CAPBANK_IN", ports=["vin", "0"], expand=_capbank_in),
        ModuleSpec("CAPBANK_OUT", ports=["out", "0"], expand=_capbank_out),
        ModuleSpec("CAPBANK_MID", ports=["mid", "0"], expand=_capbank_mid),
        ModuleSpec("SNUBBER_SW_GND", ports=["sw", "0"], expand=_snubber_sw_gnd),
        ModuleSpec("SNUBBER_SW_OUT", ports=["sw", "out"], expand=_snubber_sw_out),
        ModuleSpec("DAMPER_OUT", ports=["out", "0"], expand=_damper_out),
    ]
    return {s.name.upper(): s for s in specs}

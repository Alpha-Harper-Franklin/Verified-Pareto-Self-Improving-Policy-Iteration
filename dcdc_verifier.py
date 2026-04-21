from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from inc_parser import extract_inc_lines, parse_inc


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    family: str
    violations: List[str]
    canonical_hash: str
    n_elems: int
    n_inc_lines: int
    counts: Dict[str, int]


def _canon_hash(inc_text: str) -> str:
    lines = [l.strip() for l in extract_inc_lines(inc_text) if l.strip()]
    norm: List[str] = []
    for l in lines:
        if l.startswith("INC "):
            l2 = "INC " + " ".join(l[4:].split())
        else:
            l2 = "INC " + " ".join(l.split())
        norm.append(l2)
    norm.sort()
    return hashlib.sha256("\n".join(norm).encode("utf-8", errors="ignore")).hexdigest()


def _node_norm(n: str) -> str:
    x = (n or "").strip().lower()
    if x == "gnd":
        return "0"
    return x


def verify_inc_dcdc(
    inc_text: str,
    family: str,
    vin: Optional[float] = None,
    vout: Optional[float] = None,
) -> VerifyResult:
    fam = (family or "").strip().lower()
    elems = parse_inc(inc_text)

    counts: Dict[str, int] = {}
    nodes: Set[str] = set()
    violations: List[str] = []

    for e in elems:
        counts[e.kind] = counts.get(e.kind, 0) + 1
        for n in e.nodes:
            nodes.add(_node_norm(n))
        if e.kind in {"R", "L", "C"} and e.value is None:
            violations.append(f"{e.name}_missing_value")

    if not elems:
        violations.append("no_elements")
        return VerifyResult(False, fam, violations, _canon_hash(inc_text), 0, len(extract_inc_lines(inc_text)), counts)

    if counts.get("L", 0) < 1:
        violations.append("missing_inductor")
    if counts.get("C", 0) < 1:
        violations.append("missing_capacitor")

    if "out" not in nodes:
        violations.append("missing_out_node")
    if "0" not in nodes and "gnd" not in nodes:
        violations.append("missing_gnd_node")
    if "vin" not in nodes and "in" not in nodes:
        violations.append("missing_vin_node")

    adj: Dict[str, Set[str]] = {}

    def add_edge(a: str, b: str) -> None:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    for e in elems:
        a = _node_norm(e.nodes[0])
        b = _node_norm(e.nodes[1])
        if a and b:
            add_edge(a, b)

    starts: List[str] = []
    if "vin" in adj:
        starts.append("vin")
    if "in" in adj:
        starts.append("in")

    seen: Set[str] = set()
    stack = list(starts)
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        for y in adj.get(x, set()):
            if y not in seen:
                stack.append(y)

    if "out" not in seen:
        violations.append("disconnected_vin_to_out")

    def _match_undirected(e, a_set: Set[str], b_set: Set[str]) -> bool:
        n0 = _node_norm(e.nodes[0])
        n1 = _node_norm(e.nodes[1])
        return (n0 in a_set and n1 in b_set) or (n1 in a_set and n0 in b_set)

    def _has_kind_undirected(kind: str, a_set: Set[str], b_set: Set[str]) -> bool:
        return any(e.kind == kind and _match_undirected(e, a_set, b_set) for e in elems)

    def _has_diode(anode: str, cathode: str) -> bool:
        a = _node_norm(anode)
        c = _node_norm(cathode)
        return any(e.kind == "D" and _node_norm(e.nodes[0]) == a and _node_norm(e.nodes[1]) == c for e in elems)

    vin_nodes = {"vin", "in"}

    def _gate_name_for_switch(name: str, model: Optional[str]) -> str:
        m = (model or "").lower().strip()
        n = (name or "").lower().strip()
        if "g2" in m or m.endswith("2") or "_2" in n or n.endswith("2"):
            return "gate2"
        if "g1" in m or m.endswith("1") or "_1" in n or n.endswith("1"):
            return "gate1"
        return "gate"

    if fam == "buck":
        if counts.get("S", 0) < 1:
            violations.append("buck_missing_switch")
        if counts.get("D", 0) < 1:
            violations.append("buck_missing_diode")
        has_c_out = any(e.kind == "C" and ("out" in {_node_norm(e.nodes[0]), _node_norm(e.nodes[1])}) for e in elems)
        if not has_c_out:
            violations.append("buck_missing_output_cap")

        # topology sanity: S vin-sw, D 0->sw, L sw-out, C out-0
        if not _has_kind_undirected("S", vin_nodes, {"sw"}):
            violations.append("buck_missing_vin_sw_switch")
        if not _has_diode("0", "sw"):
            violations.append("buck_missing_diode_0_to_sw")
        if not _has_kind_undirected("L", {"sw"}, {"out"}):
            violations.append("buck_missing_sw_out_inductor")
        if not _has_kind_undirected("C", {"out"}, {"0"}):
            violations.append("buck_missing_out_gnd_cap")
    elif fam == "boost":
        if counts.get("S", 0) < 1:
            violations.append("boost_missing_switch")
        if counts.get("D", 0) < 1:
            violations.append("boost_missing_diode")

        # topology sanity: L vin-sw, S sw-0, D sw->out, C out-0
        if not _has_kind_undirected("L", vin_nodes, {"sw"}):
            violations.append("boost_missing_vin_sw_inductor")
        if not _has_kind_undirected("S", {"sw"}, {"0"}):
            violations.append("boost_missing_sw_gnd_switch")
        if not _has_diode("sw", "out"):
            violations.append("boost_missing_diode_sw_to_out")
        if not _has_kind_undirected("C", {"out"}, {"0"}):
            violations.append("boost_missing_out_gnd_cap")
    elif fam == "sepic":
        if counts.get("S", 0) < 1:
            violations.append("sepic_missing_switch")
        if counts.get("D", 0) < 1:
            violations.append("sepic_missing_diode")
        if counts.get("L", 0) < 2:
            violations.append("sepic_missing_2_inductors")
        if counts.get("C", 0) < 2:
            violations.append("sepic_missing_2_caps")

        # topology sanity: L1 vin-sw, C1 sw-n1, L2 n1-0, S1 sw-0, D1 n1->out, C2 out-0
        if not _has_kind_undirected("L", vin_nodes, {"sw"}):
            violations.append("sepic_missing_vin_sw_inductor")
        if not _has_kind_undirected("C", {"sw"}, {"n1"}):
            violations.append("sepic_missing_sw_n1_cap")
        if not _has_kind_undirected("L", {"n1"}, {"0"}):
            violations.append("sepic_missing_n1_gnd_inductor")
        if not _has_kind_undirected("S", {"sw"}, {"0"}):
            violations.append("sepic_missing_sw_gnd_switch")
        if not _has_diode("n1", "out"):
            violations.append("sepic_missing_diode_n1_to_out")
        if not _has_kind_undirected("C", {"out"}, {"0"}):
            violations.append("sepic_missing_out_gnd_cap")
    elif fam in {"buckboost", "buck-boost", "bb"}:
        if counts.get("S", 0) < 2:
            violations.append("buckboost_missing_2_switches")
        if counts.get("D", 0) < 2:
            violations.append("buckboost_missing_2_diodes")
        if counts.get("L", 0) < 2:
            violations.append("buckboost_missing_2_inductors")
        if counts.get("C", 0) < 2:
            violations.append("buckboost_missing_2_caps")

        # topology sanity for cascaded buck->boost:
        # stage1 buck: S vin-sw1, D 0->sw1, L sw1-mid, C mid-0
        # stage2 boost: L mid-sw2, S sw2-0, D sw2->out, C out-0
        if not _has_kind_undirected("S", vin_nodes, {"sw1"}):
            violations.append("buckboost_missing_vin_sw1_switch")
        if not _has_diode("0", "sw1"):
            violations.append("buckboost_missing_diode_0_to_sw1")
        if not _has_kind_undirected("L", {"sw1"}, {"mid"}):
            violations.append("buckboost_missing_sw1_mid_inductor")
        if not _has_kind_undirected("C", {"mid"}, {"0"}):
            violations.append("buckboost_missing_mid_gnd_cap")
        if not _has_kind_undirected("L", {"mid"}, {"sw2"}):
            violations.append("buckboost_missing_mid_sw2_inductor")
        if not _has_kind_undirected("S", {"sw2"}, {"0"}):
            violations.append("buckboost_missing_sw2_gnd_switch")
        if not _has_diode("sw2", "out"):
            violations.append("buckboost_missing_diode_sw2_to_out")
        if not _has_kind_undirected("C", {"out"}, {"0"}):
            violations.append("buckboost_missing_out_gnd_cap")

        # Ensure the two switching devices are independently driven (gate1/gate2),
        # otherwise the simulated duty schedule degenerates and CV will collapse.
        gates = {_gate_name_for_switch(e.name, e.model) for e in elems if e.kind == "S"}
        if not (("gate1" in gates) and ("gate2" in gates)):
            violations.append("buckboost_missing_gate1_gate2")
    else:
        violations.append("unknown_family")

    ok = len(violations) == 0
    return VerifyResult(
        ok=ok,
        family=fam,
        violations=violations,
        canonical_hash=_canon_hash(inc_text),
        n_elems=len(elems),
        n_inc_lines=len(extract_inc_lines(inc_text)),
        counts=counts,
    )

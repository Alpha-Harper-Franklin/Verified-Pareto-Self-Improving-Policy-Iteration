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
    return (n or "").strip().lower()


def infer_family(vin: float, vout: float) -> str:
    try:
        if float(vout) < float(vin):
            return "buck"
        if float(vout) > float(vin):
            return "boost"
    except Exception:
        pass
    return "unknown"


def verify_inc(inc_text: str, vin: Optional[float] = None, vout: Optional[float] = None) -> VerifyResult:
    elems = parse_inc(inc_text)
    counts: Dict[str, int] = {}
    nodes: Set[str] = set()
    for e in elems:
        counts[e.kind] = counts.get(e.kind, 0) + 1
        for n in e.nodes:
            nodes.add(_node_norm(n))

    fam = infer_family(vin or 0.0, vout or 0.0) if (vin is not None and vout is not None) else "unknown"
    violations: List[str] = []

    if not elems:
        violations.append("no_elements")
        return VerifyResult(
            False, fam, violations, _canon_hash(inc_text), 0, len(extract_inc_lines(inc_text)), counts
        )

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

    if fam == "buck":
        if counts.get("S", 0) < 1:
            violations.append("buck_missing_switch")
        if counts.get("D", 0) < 1:
            violations.append("buck_missing_diode")
        has_c_out = any(
            e.kind == "C" and (_node_norm(e.nodes[0]) == "out" or _node_norm(e.nodes[1]) == "out") for e in elems
        )
        if not has_c_out:
            violations.append("buck_missing_output_cap")

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


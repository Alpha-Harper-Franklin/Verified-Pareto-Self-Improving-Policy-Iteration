from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Set

from inc_parser import extract_inc_lines, parse_inc

from osc_taskset import OSC_FAMILY_LC, OSC_FAMILY_RC, OSC_FAMILY_RING, OSC_FAMILY_WIEN

_NUM_NODE_RE = re.compile(r"^\d+$")
_HELPER_NODE_RE = re.compile(r"^n\d+$", re.IGNORECASE)


def _node_norm(n: str) -> str:
    return (n or "").strip().lower()


def _canon_hash(inc_text: str) -> str:
    lines = [ln.strip() for ln in extract_inc_lines(inc_text) if ln.strip()]
    can = "\n".join(sorted(lines)).encode("utf-8", errors="ignore")
    return hashlib.md5(can).hexdigest()


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    violations: List[str]
    canonical_hash: str
    n_elems: int
    counts: Dict[str, int]


def verify_inc_osc(inc_text: str, *, family: str, min_elems: int = 15) -> VerifyResult:
    fam = str(family or "").strip().lower()
    elems = parse_inc(inc_text)

    counts: Dict[str, int] = {}
    for e in elems:
        counts[e.kind] = counts.get(e.kind, 0) + 1

    violations: List[str] = []

    bad_kinds = sorted({e.kind for e in elems if e.kind not in {"R", "C", "L"}})
    if bad_kinds:
        violations.append("osc_disallowed_kinds_" + "".join(bad_kinds))

    req: Set[str] = {"out", "vdd", "0"}
    seen_nodes: Set[str] = set()
    for e in elems:
        for n in e.nodes:
            nn = _node_norm(n)
            if _NUM_NODE_RE.match(nn) and nn != "0":
                violations.append("numeric_node_not_allowed")
            if (nn not in req) and (not _HELPER_NODE_RE.match(nn)):
                violations.append("node_not_allowed")
            seen_nodes.add(nn)

    for r in sorted(req):
        if r not in seen_nodes:
            violations.append(f"missing_node_{r}")

    if len(elems) < int(min_elems):
        violations.append("min_elems")

    # Family-specific minimal structure hints.
    nL = int(counts.get("L", 0) or 0)
    nC = int(counts.get("C", 0) or 0)
    if fam == OSC_FAMILY_LC:
        if nL <= 0:
            violations.append("missing_L")
        if nC <= 0:
            violations.append("missing_C")
    elif fam in {OSC_FAMILY_RC, OSC_FAMILY_WIEN, OSC_FAMILY_RING}:
        if nC <= 0:
            violations.append("missing_C")
        if fam != OSC_FAMILY_RING and nL > 0:
            violations.append("unexpected_L")

    ok = len(violations) == 0
    return VerifyResult(
        ok=ok,
        violations=violations,
        canonical_hash=_canon_hash(inc_text),
        n_elems=len(elems),
        counts=counts,
    )

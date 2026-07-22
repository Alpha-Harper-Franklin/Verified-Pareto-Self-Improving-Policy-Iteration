from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Set

from inc_parser import extract_inc_lines, parse_inc

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


def verify_inc_filter(inc_text: str, *, min_elems: int = 20) -> VerifyResult:
    elems = parse_inc(inc_text)
    counts: Dict[str, int] = {}
    for e in elems:
        counts[e.kind] = counts.get(e.kind, 0) + 1

    violations: List[str] = []

    bad_kinds = sorted({e.kind for e in elems if e.kind not in {"R", "C", "L"}})
    if bad_kinds:
        violations.append("filter_disallowed_kinds_" + "".join(bad_kinds))

    req: Set[str] = {"vin", "out", "0"}
    seen_nodes: Set[str] = set()
    for e in elems:
        for n in e.nodes:
            nn = _node_norm(n)
            if _NUM_NODE_RE.match(nn) and nn != "0":
                violations.append("numeric_node_not_allowed")
            # Allow vin/out/0 and helper nodes n1,n2,...
            if (nn not in req) and (not _HELPER_NODE_RE.match(nn)):
                violations.append("node_not_allowed")
            seen_nodes.add(nn)

    for r in sorted(req):
        if r not in seen_nodes:
            violations.append(f"missing_node_{r}")

    if len(elems) < int(min_elems):
        violations.append("min_elems")

    if counts.get("C", 0) + counts.get("L", 0) <= 0:
        violations.append("missing_reactive")

    # Connectivity: require a path from vin to out.
    adj: Dict[str, Set[str]] = {}
    for e in elems:
        a = _node_norm(e.nodes[0])
        b = _node_norm(e.nodes[1])
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def _reachable(src: str) -> Set[str]:
        src = _node_norm(src)
        seen: Set[str] = set([src])
        stack = [src]
        while stack:
            x = stack.pop()
            for y in adj.get(x, set()):
                if y in seen:
                    continue
                seen.add(y)
                stack.append(y)
        return seen

    if "vin" in seen_nodes and "out" in seen_nodes:
        if "out" not in _reachable("vin"):
            violations.append("disconnected_vin_to_out")

    ok = len(violations) == 0
    return VerifyResult(
        ok=ok,
        violations=violations,
        canonical_hash=_canon_hash(inc_text),
        n_elems=len(elems),
        counts=counts,
    )

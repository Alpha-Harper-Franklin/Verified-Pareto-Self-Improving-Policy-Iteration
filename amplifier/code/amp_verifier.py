from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List

from inc_parser import extract_inc_lines, parse_inc


_NUM_NODE_RE = re.compile(r"^\d+$")


def _node_norm(n: str) -> str:
    return (n or "").strip().lower()


def _canon_hash(inc_text: str) -> str:
    lines = [ln.strip() for ln in extract_inc_lines(inc_text) if ln.strip()]
    # Deterministic canonicalization: sort lines (order-insensitive) and hash.
    can = "\n".join(sorted(lines)).encode("utf-8", errors="ignore")
    return hashlib.md5(can).hexdigest()


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    violations: List[str]
    canonical_hash: str
    n_elems: int
    counts: Dict[str, int]


def verify_inc_amp(inc_text: str, *, min_elems: int = 15) -> VerifyResult:
    elems = parse_inc(inc_text)
    counts: Dict[str, int] = {}
    for e in elems:
        counts[e.kind] = counts.get(e.kind, 0) + 1

    violations: List[str] = []

    # Allow only passive R/C/L in the generated INC (op-amp core is provided by the testbench).
    bad_kinds = sorted({e.kind for e in elems if e.kind not in {"R", "C", "L"}})
    if bad_kinds:
        violations.append("amp_disallowed_kinds_" + "".join(bad_kinds))

    # Required node set.
    req = {"vin", "inv", "out", "vdd", "0"}
    seen_nodes = set()
    for e in elems:
        for n in e.nodes:
            nn = _node_norm(n)
            if _NUM_NODE_RE.match(nn) and nn != "0":
                violations.append("numeric_node_not_allowed")
            seen_nodes.add(nn)

    # Enforce non-inverting topology: `vin` is the non-inv input driven by an ideal source.
    # Connecting `vin` to other non-ground nodes makes the closed-loop definition ambiguous and
    # breaks the evaluator's gain model.
    for e in elems:
        n1, n2 = _node_norm(e.nodes[0]), _node_norm(e.nodes[1])
        if "vin" in {n1, n2} and "0" not in {n1, n2}:
            violations.append("vin_must_only_connect_to_0")
            break

    for r in sorted(req):
        if r not in seen_nodes:
            violations.append(f"missing_node_{r}")

    if len(elems) < int(min_elems):
        violations.append("min_elems")

    def _nset(e) -> frozenset[str]:
        n1 = _node_norm(e.nodes[0]) if e.nodes else ""
        n2 = _node_norm(e.nodes[1]) if len(e.nodes or []) >= 2 else ""
        return frozenset({n1, n2})

    def _as_float(v):
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    def _pick_min_res_idx(conn: frozenset[str]) -> int | None:
        best_i: int | None = None
        best_v: float | None = None
        for i, e in enumerate(elems):
            if e.kind != "R":
                continue
            if _nset(e) != conn:
                continue
            vv = _as_float(getattr(e, "value", None))
            if vv is None:
                continue
            if best_v is None or float(vv) < float(best_v):
                best_i, best_v = int(i), float(vv)
        return best_i

    def _pick_max_cap_idx(conn: frozenset[str]) -> int | None:
        best_i: int | None = None
        best_v: float | None = None
        for i, e in enumerate(elems):
            if e.kind != "C":
                continue
            if _nset(e) != conn:
                continue
            vv = _as_float(getattr(e, "value", None))
            if vv is None:
                continue
            if best_v is None or float(vv) > float(best_v):
                best_i, best_v = int(i), float(vv)
        return best_i

    # Bias resistor: any R between vdd-0 (do not require the name "RBIAS" here).
    bias_conn = frozenset({"vdd", "0"})
    bias_i = _pick_min_res_idx(bias_conn)
    if bias_i is None:
        violations.append("missing_bias_R_vdd_0")

    # Basic negative-feedback structure: at least one R between out-inv and one R between inv-0.
    fb_conn = frozenset({"out", "inv"})
    rg_conn = frozenset({"inv", "0"})
    fb_i = _pick_min_res_idx(fb_conn)
    rg_i = _pick_min_res_idx(rg_conn)
    if fb_i is None:
        violations.append("missing_feedback_R_out_inv")
    if rg_i is None:
        violations.append("missing_feedback_R_inv_0")

    # Treat the dominant out-inv capacitor as "essential" (compensation), if present.
    ccomp_i = _pick_max_cap_idx(fb_conn)
    essential = {i for i in [bias_i, fb_i, rg_i, ccomp_i] if i is not None}

    # Keep "padding" elements effectively non-interacting.
    # This avoids the min_elems constraint accidentally destroying the designed transfer function.
    for i, e in enumerate(elems):
        if int(i) in essential:
            continue
        if e.kind == "R" and e.value is not None:
            if float(e.value) < 1e9:
                violations.append("nonessential_R_must_be_open")
                break
        if e.kind == "C" and e.value is not None:
            if float(e.value) > 1e-12:
                violations.append("nonessential_C_must_be_tiny")
                break
        if e.kind == "L" and e.value is not None:
            if float(e.value) < 1e2:
                violations.append("nonessential_L_must_be_large")
                break

    ok = len(violations) == 0
    return VerifyResult(
        ok=ok,
        violations=violations,
        canonical_hash=_canon_hash(inc_text),
        n_elems=len(elems),
        counts=counts,
    )

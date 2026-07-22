from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_NUM_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]{0,3})$")
_SUFFIX = {
    "G": 1e9,
    "M": 1e6,
    "K": 1e3,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "U": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
}


def parse_numeric(tok: str) -> Optional[float]:
    tok = (tok or "").strip()
    if not tok:
        return None
    m = _NUM_RE.match(tok)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except Exception:
        return None
    suf = m.group(2) or ""
    if suf:
        s0 = suf[0]
        if s0 in _SUFFIX:
            v *= _SUFFIX[s0]
    return v


def _safe_ident(tok: str) -> Optional[str]:
    tok = (tok or "").strip()
    if not tok:
        return None
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tok):
        return tok
    return None


@dataclass(frozen=True)
class IncElem:
    name: str
    kind: str
    nodes: List[str]
    value: Optional[float] = None
    model: Optional[str] = None
    raw: str = ""


_ELEM_NAME_RE = re.compile(r"^[RLCSD][A-Za-z0-9_]*$")
_NODE_RE = re.compile(r"^[A-Za-z0-9_]+$")
_INC_TOK_RE = re.compile(r"(?i)\bINC")
_INC_LINE_START_RE = re.compile(r"(?i)^\s*(?:[-*]|\d+[.)])?\s*INC")


def _normalize_inc_line(line: str) -> Optional[str]:
    s = (line or "").strip()
    if not s:
        return None

    # Tolerate prefixes like "1. INC ..." or "- INC ...".
    # Also tolerate missing whitespace after INC (e.g., "INCR1 ..." or "INCS1 ...").
    m = re.match(r"(?i)^(?:\s*(?:\d+[.)]|[-*])\s*)?INC", s)
    if not m:
        return None
    s = s[m.end() :].strip()

    toks = s.split()
    if len(toks) < 3:
        return None

    name, n1, n2 = toks[0], toks[1], toks[2]
    tail = toks[3] if len(toks) >= 4 else ""
    if not name:
        return None

    kind = (name[:1] or "?").upper()
    if kind not in {"R", "L", "C", "D", "S"}:
        return None

    if not _ELEM_NAME_RE.match(name):
        return None
    if not _NODE_RE.match(n1) or not _NODE_RE.match(n2):
        return None

    if kind in {"R", "L", "C"}:
        if not tail or parse_numeric(tail) is None:
            return None
    else:
        # tolerate missing/invalid model token by defaulting to known models
        mdl = "Dstd" if kind == "D" else "Sstd"
        return f"INC {name} {n1} {n2} {mdl}"

    return f"INC {name} {n1} {n2} {tail}"


def extract_inc_lines(text: str) -> List[str]:
    """
    Extract normalized INC lines from free-form model output.

    Important: model outputs may contain multiple `INC ...` statements on the *same* line
    (missing newlines). We therefore split repeated INC tokens *within candidate INC lines*.
    """
    s = text or ""
    out: List[str] = []
    seen = set()

    # Only consider lines that plausibly start an INC statement; this avoids false positives
    # like: "Here is the INC DSL code ..." being mis-parsed into a diode line.
    for raw in s.splitlines():
        raw_s = (raw or "").strip()
        if not raw_s:
            continue
        if not _INC_LINE_START_RE.match(raw_s):
            continue

        matches = list(_INC_TOK_RE.finditer(raw_s))
        if not matches:
            continue

        chunks: List[str] = []
        for i, m in enumerate(matches):
            # Require a word boundary before INC to avoid matching inside identifiers.
            if m.start() > 0 and re.match(r"[A-Za-z0-9_]", raw_s[m.start() - 1]):
                continue
            start = m.start()
            end = matches[i + 1].start() if (i + 1) < len(matches) else len(raw_s)
            chunk = raw_s[start:end].strip()
            if chunk:
                chunks.append(chunk)

        for cand in chunks:
            norm = _normalize_inc_line(cand)
            if norm and norm not in seen:
                out.append(norm)
                seen.add(norm)

    return out


def parse_inc(text: str) -> List[IncElem]:
    elems: List[IncElem] = []
    for line in extract_inc_lines(text):
        raw = line
        if line.startswith("INC "):
            line = line[4:].strip()
        toks = line.split()
        if len(toks) != 4:
            continue
        name, n1, n2, tail = toks

        kind = (name[:1] or "?").upper()
        value: Optional[float] = None
        model: Optional[str] = None

        if kind in {"R", "L", "C"}:
            value = parse_numeric(tail)
        elif kind in {"D", "S"}:
            model = _safe_ident(tail)
        else:
            continue

        elems.append(IncElem(name=name, kind=kind, nodes=[n1, n2], value=value, model=model, raw=raw))
    return elems


def to_inc_text(elems: List[IncElem]) -> str:
    lines: List[str] = []
    for e in elems:
        if e.kind in {"R", "L", "C"}:
            if e.value is None:
                continue
            lines.append(f"INC {e.name} {e.nodes[0]} {e.nodes[1]} {e.value}")
        elif e.kind in {"D", "S"}:
            mdl = e.model or ("Dstd" if e.kind == "D" else "Sstd")
            lines.append(f"INC {e.name} {e.nodes[0]} {e.nodes[1]} {mdl}")
    return "\n".join(lines)

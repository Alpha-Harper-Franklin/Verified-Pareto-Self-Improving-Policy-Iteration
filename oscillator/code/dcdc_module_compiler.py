from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from dcdc_modules import ModuleSpec, default_module_specs


@dataclass(frozen=True)
class ModuleCall:
    name: str
    args: List[str]
    raw: str = ""


_MOD_RE = re.compile(r"(?im)^\s*MOD\s+([A-Za-z0-9_]+)\s*(.*)$")


def extract_module_calls(text: str) -> List[ModuleCall]:
    calls: List[ModuleCall] = []
    for m in _MOD_RE.finditer(text or ""):
        name = (m.group(1) or "").strip()
        tail = (m.group(2) or "").strip()
        args = [a.strip() for a in tail.split() if a.strip()]
        calls.append(ModuleCall(name=name, args=args, raw=m.group(0)))
    return calls


def _base_family(mod_name: str) -> Optional[str]:
    n = (mod_name or "").strip().upper()
    if n == "BUCK_BASE":
        return "buck"
    if n == "BOOST_BASE":
        return "boost"
    if n == "SEPIC_BASE":
        return "sepic"
    if n == "BUCKBOOST_BASE":
        return "buckboost"
    return None


def compile_module_graph(text: str, expected_family: Optional[str] = None) -> Tuple[Optional[str], str, List[str]]:
    """
    Compile `MOD ...` lines into INC netlist text.
    Returns (family, inc_text, errors). No fallback/repair is performed.
    """
    specs: Dict[str, ModuleSpec] = default_module_specs()
    calls = extract_module_calls(text)
    if not calls:
        return None, "", ["no_MOD_lines_found"]

    base_calls = [c for c in calls if _base_family(c.name) is not None]
    if len(base_calls) != 1:
        return None, "", [f"need_exactly_1_base_module_got_{len(base_calls)}"]

    base = base_calls[0]
    family = _base_family(base.name)
    if not family:
        return None, "", ["unknown_base_module"]
    if expected_family and (family.lower() != str(expected_family).lower().strip()):
        return family, "", [f"base_family_mismatch_expected_{expected_family}_got_{family}"]

    inc_lines: List[str] = []
    errors: List[str] = []

    for inst, call in enumerate(calls, start=1):
        key = (call.name or "").strip().upper()
        spec = specs.get(key)
        if not spec:
            errors.append(f"unknown_module_{call.name}")
            continue

        if len(call.args) != len(spec.ports):
            errors.append(f"module_{call.name}_needs_{len(spec.ports)}_args_got_{len(call.args)}")
            continue

        port_map = {p: a for p, a in zip(spec.ports, call.args)}
        try:
            inc_lines.extend(spec.expand(port_map, inst))
        except Exception as e:
            errors.append(f"module_{call.name}_expand_error_{type(e).__name__}")
            continue

    if errors:
        return family, "", errors

    inc_text = "\n".join([l.strip() for l in inc_lines if l.strip()]) + "\n"
    return family, inc_text, []


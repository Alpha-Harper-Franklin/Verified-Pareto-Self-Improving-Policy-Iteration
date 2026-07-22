#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dcdc_taskset import Task, default_taskset
from task_manifest import load_tasks_jsonl, sha256_file
from dcdc_verifier import verify_inc_dcdc
from inc_parser import extract_inc_lines, parse_inc
from repair_corrector import RepairCorrector, build_feature_vector


RESPONSE_TEMPLATE = "### Response:\n"
_RESP_KEY = "### Response:"


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _cgroup_quota_cpus() -> Optional[int]:
    # cgroup v2
    try:
        p = "/sys/fs/cgroup/cpu.max"
        if os.path.exists(p):
            txt = Path(p).read_text(encoding="utf-8", errors="ignore").strip().split()
            if len(txt) >= 2:
                quota, period = txt[0], txt[1]
                if quota != "max":
                    q = int(quota)
                    per = int(period)
                    if q > 0 and per > 0:
                        return max(1, int(math.ceil(q / per)))
    except Exception:
        pass

    # cgroup v1
    for base in ["/sys/fs/cgroup/cpu", "/sys/fs/cgroup"]:
        try:
            qpath = os.path.join(base, "cpu.cfs_quota_us")
            ppath = os.path.join(base, "cpu.cfs_period_us")
            if not (os.path.exists(qpath) and os.path.exists(ppath)):
                continue
            q = int(Path(qpath).read_text(encoding="utf-8", errors="ignore").strip() or "-1")
            per = int(Path(ppath).read_text(encoding="utf-8", errors="ignore").strip() or "0")
            if q <= 0 or per <= 0:
                continue
            return max(1, int(math.ceil(q / per)))
        except Exception:
            continue
    return None


def _effective_cpus() -> int:
    # Use the tightest bound among: cgroup quota, CPU affinity, os.cpu_count().
    caps: List[int] = []
    q = _cgroup_quota_cpus()
    if q is not None:
        caps.append(int(q))
    try:
        caps.append(int(len(os.sched_getaffinity(0))))  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        caps.append(int(os.cpu_count() or 1))
    except Exception:
        pass
    caps = [int(x) for x in caps if int(x) > 0]
    return max(1, int(min(caps)) if caps else 1)


def _auto_sim_workers(n: int) -> int:
    try:
        n = int(n)
    except Exception:
        n = 0
    if n > 0:
        return int(n)
    return int(_effective_cpus())


def _device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(base_model: str, adapter: str) -> tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.bos_token_id is None:
        tok.bos_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if str(adapter or "").strip():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter).strip(), is_trainable=False)

    model.eval()
    return tok, model


def _score(detail: dict, vout: float, tol_ref: float = 0.01) -> float:
    if not bool(detail.get("ok", False)):
        return -1.0
    try:
        vavg = float(detail.get("vavg", 0.0) or 0.0)
        eff = float(detail.get("eff", 0.0) or 0.0)
        ripple = float(detail.get("ripple", 0.0) or 0.0)
        overshoot = float(detail.get("overshoot", 0.0) or 0.0)
    except Exception:
        return -1.0

    err = abs(vavg - float(vout)) / max(1e-6, float(vout))
    tol = max(1e-6, float(tol_ref))
    if err <= tol:
        score_v = 3.0 * max(0.0, 1.0 - err / tol)
    else:
        # Provide partial credit just above tolerance to improve ranking signal.
        # err in (tol, 2*tol] linearly maps to score_v in (0, -1].
        score_v = -min(1.0, (err - tol) / tol)
    score_eff = 0.5 * eff
    score_ripple = -0.2 * (ripple / max(1e-6, float(vout)))
    score_over = -0.2 * overshoot

    if not bool(detail.get("pass_CE", False)):
        score_eff -= 0.5

    return float(score_v + score_eff + score_ripple + score_over)


def _cv_err(detail: dict, vout: float) -> float:
    try:
        if not bool(detail.get("ok", False)):
            return 1e9
        vavg = float(detail.get("vavg", 0.0) or 0.0)
        return float(abs(vavg - float(vout)) / max(1e-6, float(vout)))
    except Exception:
        return 1e9


def _sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


def _ripple_norm(detail: dict, vout: float) -> float:
    try:
        return float(detail.get("ripple", 0.0) or 0.0) / max(1e-6, float(vout))
    except Exception:
        return 1e9


def _pareto_obj(rec: Dict[str, Any], vout: float) -> Tuple[float, float, float, float, float]:
    # Minimize: cv_err, ripple_norm, overshoot; Maximize: eff, n_elems
    d = rec.get("detail") or {}
    cv_err = float(rec.get("cv_err", 1e9))
    eff = float(d.get("eff", 0.0) or 0.0)
    ripple_n = _ripple_norm(d, vout=float(vout))
    over = float(d.get("overshoot", 0.0) or 0.0)
    n_elems = float((d.get("n_elems") or rec.get("n_elems") or 0.0) or 0.0)
    return (cv_err, ripple_n, over, -eff, -n_elems)


def _dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    le = True
    lt = False
    for x, y in zip(a, b):
        if float(x) > float(y) + 1e-12:
            le = False
            break
        if float(x) < float(y) - 1e-12:
            lt = True
    return bool(le and lt)


def _pareto_front(recs: List[Dict[str, Any]], vout: float) -> List[Dict[str, Any]]:
    objs = [_pareto_obj(r, vout=float(vout)) for r in recs]
    keep: List[Dict[str, Any]] = []
    for i, r in enumerate(recs):
        dominated = False
        for j in range(len(recs)):
            if i == j:
                continue
            if _dominates(objs[j], objs[i]):
                dominated = True
                break
        if not dominated:
            keep.append(r)
    return keep


def _set_elem_value(inc: str, name: str, value: float) -> str:
    lines = extract_inc_lines(inc)
    out: List[str] = []
    changed = False
    for raw in lines:
        toks = raw.split()
        if len(toks) == 5 and toks[0] == "INC" and toks[1] == name:
            toks[4] = f"{float(value):.6g}"
            out.append(" ".join(toks))
            changed = True
        else:
            out.append(raw)
    if not changed:
        return inc
    return ("\n".join(out).strip() + "\n")


def _set_elem_model(inc: str, name: str, model: str) -> str:
    lines = extract_inc_lines(inc)
    out: List[str] = []
    changed = False
    for raw in lines:
        toks = raw.split()
        if len(toks) == 5 and toks[0] == "INC" and toks[1] == name:
            toks[4] = str(model)
            out.append(" ".join(toks))
            changed = True
        else:
            out.append(raw)
    if not changed:
        return inc
    return ("\n".join(out).strip() + "\n")


def _append_inc_line(inc: str, line: str) -> str:
    s = (inc or "").strip()
    ln = (line or "").strip()
    if not ln:
        return (s + "\n") if s else ""
    if s:
        return s + "\n" + ln + "\n"
    return ln + "\n"


def _new_name(existing: set[str], prefix: str, *, start: int = 1) -> str:
    p = (prefix or "").strip().upper()[:1]
    if p not in {"R", "L", "C", "D", "S"}:
        p = "X"
    i = max(1, int(start))
    while True:
        name = f"{p}{i}"
        if name not in existing:
            existing.add(name)
            return name
        i += 1


def _rewrite_first_diode(inc: str, *, anode: str, cathode: str) -> tuple[str, bool]:
    a = str(anode).lower().strip()
    c = str(cathode).lower().strip()
    lines = extract_inc_lines(inc)
    out: List[str] = []
    changed = False
    for raw in lines:
        toks = raw.split()
        if (not changed) and len(toks) == 5 and toks[0] == "INC" and toks[1].startswith("D"):
            n1 = toks[2].lower()
            n2 = toks[3].lower()
            if n1 == c and n2 == a:
                toks[2] = str(anode)
                toks[3] = str(cathode)
                out.append(" ".join(toks))
                changed = True
                continue
        out.append(raw)
    if not changed:
        return inc, False
    return ("\n".join(out).strip() + "\n"), True


def _select_tunable_names(inc: str, family: str) -> List[str]:
    fam = str(family or "").strip().lower()
    elems = parse_inc(inc)

    def _match(kind: str, a: str, b: str) -> Optional[str]:
        a0 = a.lower()
        b0 = b.lower()
        for e in elems:
            if e.kind != kind or not e.name:
                continue
            n = {str(x).lower() for x in (e.nodes or [])}
            if a0 in n and b0 in n:
                return str(e.name)
        return None

    want: List[str] = []
    if fam == "buck":
        want += [x for x in [_match("L", "sw", "out"), _match("C", "out", "0")] if x]
    elif fam == "boost":
        want += [x for x in [_match("L", "vin", "sw"), _match("C", "out", "0")] if x]
    elif fam == "sepic":
        want += [
            x
            for x in [
                _match("L", "vin", "sw"),
                _match("L", "n1", "0"),
                _match("C", "sw", "n1"),
                _match("C", "out", "0"),
            ]
            if x
        ]
    elif fam in {"buckboost", "buck-boost", "bb"}:
        want += [
            x
            for x in [
                _match("L", "sw1", "mid"),
                _match("C", "mid", "0"),
                _match("L", "mid", "sw2"),
                _match("C", "out", "0"),
            ]
            if x
        ]

    # Fallback: tune up to 2 inductors + 2 capacitors by appearance order.
    if not want:
        for e in elems:
            if e.kind in {"L", "C"} and e.name not in want:
                want.append(str(e.name))
            if len(want) >= 4:
                break
    return want


def _suggest_tunable_values(*, family: str, vin: float, vout: float, rload: float) -> Dict[str, float]:
    """
    Closed-form-ish sizing for key passives to help the EDA local-search escape the failure region quickly.

    Notes:
      - This is ONLY used inside (A) to construct better self-play training data under tol=0.01.
      - It is NOT used in eval scripts; so the learned policy still must generate good values itself.
    """
    fam = (family or "").strip().lower()
    vin2 = max(1e-6, float(vin))
    vout2 = max(1e-6, float(vout))
    r = max(1e-3, float(rload))
    fs = 200_000.0

    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(float(lo), min(float(hi), float(v)))

    iout = max(1e-6, vout2 / r)
    dv = max(0.005 * vout2, 0.02)  # ripple proxy

    if fam == "buck":
        d = _clamp(vout2 / vin2, 0.05, 0.95)
        di = max(1e-6, 0.30 * iout)
        L = (vin2 - vout2) * d / (di * fs)
        C = di / (8.0 * fs * dv)
        return {"L_main": _clamp(L, 5e-6, 300e-6), "C_out": _clamp(C, 0.5e-6, 800e-6)}

    if fam == "boost":
        d = _clamp(1.0 - (vin2 / max(vout2, vin2)), 0.05, 0.95)
        iin = iout * (vout2 / vin2)
        di = max(1e-6, 0.30 * iin)
        L = vin2 * d / (di * fs)
        C = iout * d / (fs * dv)
        return {"L_in": _clamp(L, 5e-6, 300e-6), "C_out": _clamp(C, 0.5e-6, 1200e-6)}

    if fam == "sepic":
        d = _clamp(vout2 / (vin2 + vout2), 0.05, 0.95)
        di = max(1e-6, 0.30 * iout)
        L = vin2 * d / (di * fs)
        C_out = di / (8.0 * fs * dv)
        C_c = iout * d / (fs * dv)
        return {
            "L1": _clamp(L, 5e-6, 300e-6),
            "L2": _clamp(L, 5e-6, 300e-6),
            "C_couple": _clamp(C_c, 0.2e-6, 800e-6),
            "C_out": _clamp(C_out, 0.5e-6, 1200e-6),
        }

    if fam in {"buckboost", "buck-boost", "bb"}:
        # Mirror dcdc_spice_builder mid choice
        if vout2 >= vin2:
            vmid = 0.8 * vin2
        else:
            vmid = 0.95 * vout2
        vmid = _clamp(vmid, 0.05 * min(vin2, vout2), 0.95 * min(vin2, vout2))

        d1 = _clamp(vmid / vin2, 0.05, 0.95)
        di1 = max(1e-6, 0.30 * (vmid / r))
        L1 = (vin2 - vmid) * d1 / (di1 * fs)
        Cmid = di1 / (8.0 * fs * max(0.005 * vmid, 0.02))

        d2 = _clamp(1.0 - (vmid / max(vout2, vmid)), 0.05, 0.95)
        iout2 = vout2 / r
        iin2 = iout2 * (vout2 / vmid)
        di2 = max(1e-6, 0.30 * iin2)
        L2 = vmid * d2 / (di2 * fs)
        Cout = iout2 * d2 / (fs * dv)
        return {"L1": _clamp(L1, 5e-6, 300e-6), "Cmid": _clamp(Cmid, 0.2e-6, 800e-6), "L2": _clamp(L2, 5e-6, 300e-6), "C_out": _clamp(Cout, 0.5e-6, 1200e-6)}

    return {}


def _structural_complete_minimal(inc: str, *, family: str, vin: float, vout: float) -> str:
    """
    Minimal non-template structural completion to salvage near-miss samples:
      - Add/fix only the components required by the family verifier.
      - Fix reversed diodes on the same node pair when possible.
      - For buckboost, ensure gate1/gate2 split by editing switch model names.
    """
    fam = (family or "").strip().lower()
    elems = parse_inc(inc)
    existing = {str(e.name).strip() for e in elems if str(e.name).strip()}

    def _has_undirected(kind: str, a: str, b: str) -> bool:
        a0 = str(a).lower().strip()
        b0 = str(b).lower().strip()
        for e in elems:
            if e.kind != kind:
                continue
            n = {str(x).lower() for x in (e.nodes or [])}
            if a0 in n and b0 in n:
                return True
        return False

    def _has_diode(anode: str, cathode: str) -> bool:
        a0 = str(anode).lower().strip()
        c0 = str(cathode).lower().strip()
        for e in elems:
            if e.kind != "D":
                continue
            if str(e.nodes[0]).lower() == a0 and str(e.nodes[1]).lower() == c0:
                return True
        return False

    def _add(kind: str, n1: str, n2: str, tail: str) -> None:
        nonlocal inc, elems
        name = _new_name(existing, kind, start=1)
        inc = _append_inc_line(inc, f"INC {name} {n1} {n2} {tail}")
        elems = parse_inc(inc)

    sugg = _suggest_tunable_values(family=fam, vin=float(vin), vout=float(vout), rload=10.0)
    L_main = float(sugg.get("L_main", 50e-6))
    L_in = float(sugg.get("L_in", 50e-6))
    C_out = float(sugg.get("C_out", 100e-6))
    C_couple = float(sugg.get("C_couple", C_out))
    Cmid = float(sugg.get("Cmid", 0.5 * C_out))

    if fam == "buck":
        if not (_has_undirected("S", "vin", "sw") or _has_undirected("S", "in", "sw")):
            _add("S", "vin", "sw", "Sstd")
        if not _has_diode("0", "sw"):
            inc, _ = _rewrite_first_diode(inc, anode="0", cathode="sw")
            elems = parse_inc(inc)
            if not _has_diode("0", "sw"):
                _add("D", "0", "sw", "Dstd")
        if not _has_undirected("L", "sw", "out"):
            _add("L", "sw", "out", f"{L_main:.6g}")
        if not _has_undirected("C", "out", "0"):
            _add("C", "out", "0", f"{C_out:.6g}")
        return inc

    if fam == "boost":
        if not (_has_undirected("L", "vin", "sw") or _has_undirected("L", "in", "sw")):
            _add("L", "vin", "sw", f"{L_in:.6g}")
        if not _has_undirected("S", "sw", "0"):
            _add("S", "sw", "0", "Sstd")
        if not _has_diode("sw", "out"):
            inc, _ = _rewrite_first_diode(inc, anode="sw", cathode="out")
            elems = parse_inc(inc)
            if not _has_diode("sw", "out"):
                _add("D", "sw", "out", "Dstd")
        if not _has_undirected("C", "out", "0"):
            _add("C", "out", "0", f"{C_out:.6g}")
        return inc

    if fam == "sepic":
        if not (_has_undirected("L", "vin", "sw") or _has_undirected("L", "in", "sw")):
            _add("L", "vin", "sw", f"{L_main:.6g}")
        if not _has_undirected("C", "sw", "n1"):
            _add("C", "sw", "n1", f"{C_couple:.6g}")
        if not _has_undirected("L", "n1", "0"):
            _add("L", "n1", "0", f"{L_main:.6g}")
        if not _has_undirected("S", "sw", "0"):
            _add("S", "sw", "0", "Sstd")
        if not _has_diode("n1", "out"):
            inc, _ = _rewrite_first_diode(inc, anode="n1", cathode="out")
            elems = parse_inc(inc)
            if not _has_diode("n1", "out"):
                _add("D", "n1", "out", "Dstd")
        if not _has_undirected("C", "out", "0"):
            _add("C", "out", "0", f"{C_out:.6g}")
        return inc

    if fam in {"buckboost", "buck-boost", "bb"}:
        if not (_has_undirected("S", "vin", "sw1") or _has_undirected("S", "in", "sw1")):
            _add("S", "vin", "sw1", "Sstd1")
        if not _has_diode("0", "sw1"):
            inc, _ = _rewrite_first_diode(inc, anode="0", cathode="sw1")
            elems = parse_inc(inc)
            if not _has_diode("0", "sw1"):
                _add("D", "0", "sw1", "Dstd")
        if not _has_undirected("L", "sw1", "mid"):
            _add("L", "sw1", "mid", f"{float(sugg.get('L1', L_main)):.6g}")
        if not _has_undirected("C", "mid", "0"):
            _add("C", "mid", "0", f"{Cmid:.6g}")
        if not _has_undirected("L", "mid", "sw2"):
            _add("L", "mid", "sw2", f"{float(sugg.get('L2', L_in)):.6g}")
        if not _has_undirected("S", "sw2", "0"):
            _add("S", "sw2", "0", "Sstd2")
        if not _has_diode("sw2", "out"):
            inc, _ = _rewrite_first_diode(inc, anode="sw2", cathode="out")
            elems = parse_inc(inc)
            if not _has_diode("sw2", "out"):
                _add("D", "sw2", "out", "Dstd")
        if not _has_undirected("C", "out", "0"):
            _add("C", "out", "0", f"{C_out:.6g}")

        # Ensure gate1/gate2 split for two switches by editing model names (connectivity preserved).
        sws = [e for e in elems if e.kind == "S"]
        if len(sws) >= 2:
            inc = _set_elem_model(inc, sws[0].name, (sws[0].model or "Sstd") + "1")
            inc = _set_elem_model(inc, sws[1].name, (sws[1].model or "Sstd") + "2")
        return inc

    return inc


def _bank_expand_to_min_elems(
    inc: str,
    *,
    family: str,
    min_elems: int,
    max_parts_per_elem: int = 32,
) -> tuple[str, Dict[str, Any]]:
    """
    Increase element count to >=min_elems without changing topology, via electrically-equivalent passive banks.

    Currently: split the main output capacitor (out-0) into a parallel bank (value/k each).
    """
    meta: Dict[str, Any] = {"used": False}
    elems = parse_inc(inc)
    n0 = int(len(elems))
    need = int(min_elems) - n0
    if need <= 0:
        return inc, meta

    caps = [e for e in elems if e.kind == "C" and (e.value is not None) and float(e.value) > 0.0]
    if not caps:
        return inc, meta

    def _is_out0(e) -> bool:
        ns = {str(x).lower() for x in (e.nodes or [])}
        return ("out" in ns) and ("0" in ns)

    caps_out = [e for e in caps if _is_out0(e)]
    pick = max(caps_out, key=lambda e: float(e.value)) if caps_out else max(caps, key=lambda e: float(e.value))

    c_min = 1e-9  # consistent with dcdc_spice_builder clamp
    base = float(pick.value)
    k_target = int(min(int(max_parts_per_elem), need + 1))
    k_max_by_min = int(max(1, math.floor(base / c_min)))
    k = int(max(1, min(k_target, k_max_by_min)))
    if k <= 1:
        return inc, meta

    part = float(base) / float(k)
    if part < c_min:
        return inc, meta

    inc2 = _set_elem_value(inc, pick.name, float(part))
    added: List[str] = []
    existing = {str(e.name).strip() for e in elems if str(e.name).strip()}
    for _ in range(k - 1):
        name_new = _new_name(existing, "C", start=1000)
        n1, n2 = pick.nodes[0], pick.nodes[1]
        inc2 = _append_inc_line(inc2, f"INC {name_new} {n1} {n2} {float(part):.6g}")
        added.append(name_new)

    meta = {
        "used": True,
        "strategy": "cap_bank_parallel",
        "picked": {"name": str(pick.name), "nodes": list(pick.nodes), "value_before": float(base), "value_each": float(part), "k": int(k)},
        "added": added,
        "n_elems_before": int(n0),
        "n_elems_after": int(len(parse_inc(inc2))),
        "family": str(family).lower(),
    }
    return inc2, meta


def _eda_repair_local_search(
    *,
    inc: str,
    family: str,
    vin: float,
    vout: float,
    tol: float,
    rload: float,
    t_pre: float,
    t_win: float,
    sim_timeout_s: float,
    autotune_duty: bool,
    factors: List[float],
    max_evals: int,
    max_iters: int,
    only_if_fail: bool,
    corrector: Optional[RepairCorrector],
    corrector_topk: int,
    corrector_max_factor: float,
    eval_one_detail_dcdc,
) -> Dict[str, Any]:
    """
    Industrial-EDA-style local search (no template fallback):
      - run simulation for current INC,
      - then coordinate-search key L/C values by multiplicative factors,
      - always keep the best (by scalar score), record before/after details.
    """
    factors = [float(x) for x in (factors or []) if float(x) > 0.0]
    if not factors:
        factors = [0.5, 0.8, 1.0, 1.25, 1.5, 2.0]
    max_evals = max(1, int(max_evals))
    max_iters = max(0, int(max_iters))

    detail0 = eval_one_detail_dcdc(
        inc=inc,
        family=str(family),
        vin=float(vin),
        vout=float(vout),
        tol=float(tol),
        rload=float(rload),
        t_pre=float(t_pre),
        t_win=float(t_win),
        sim_timeout_s=float(sim_timeout_s),
        autotune_duty=bool(autotune_duty),
    )
    score0 = _score(detail0, vout=float(vout), tol_ref=float(tol))
    best_inc = str(inc)
    best_detail = dict(detail0)
    best_score = float(score0)
    n_evals = 1

    if bool(only_if_fail) and bool(best_detail.get("pass_CV", False)):
        return {
            "inc_before": inc,
            "detail_before": detail0,
            "score_before": float(score0),
            "inc_after": best_inc,
            "detail_after": best_detail,
            "score_after": float(best_score),
            "n_evals": int(n_evals),
            "changed": False,
            "tuned": {},
            "corrector": {"used": False, "reason": "already_pass_cv"},
            "init": {"used": False, "reason": "already_pass_cv"},
        }

    corrector_meta: Optional[Dict[str, Any]] = None
    init_meta: Optional[Dict[str, Any]] = None
    tuned: Dict[str, Dict[str, float]] = {}

    # Analytic init for key passives (fast path to escape tol=0.01 failure region).
    try:
        if bool(best_detail.get("ok", False)) and (not bool(best_detail.get("pass_CV", False))) and n_evals < int(max_evals):
            sugg = _suggest_tunable_values(family=str(family), vin=float(vin), vout=float(vout), rload=float(rload))
            if sugg:
                tunables = _select_tunable_names(best_inc, family=str(family))[:4]
                elems_now = {e.name: e for e in parse_inc(best_inc)}
                cand_inc = str(best_inc)
                init_tuned: Dict[str, Dict[str, float]] = {}
                fam = str(family or "").strip().lower()
                for idx, name in enumerate(tunables):
                    e = elems_now.get(name)
                    if not e or e.value is None:
                        continue
                    kind = str(e.kind)
                    before = float(e.value)
                    target = None
                    if fam == "buck":
                        target = float(sugg.get("L_main" if kind == "L" else "C_out", 0.0))
                    elif fam == "boost":
                        target = float(sugg.get("L_in" if kind == "L" else "C_out", 0.0))
                    elif fam == "sepic":
                        if idx == 0 and kind == "L":
                            target = float(sugg.get("L1", 0.0))
                        elif idx == 1 and kind == "L":
                            target = float(sugg.get("L2", 0.0))
                        elif idx == 2 and kind == "C":
                            target = float(sugg.get("C_couple", 0.0))
                        elif idx == 3 and kind == "C":
                            target = float(sugg.get("C_out", 0.0))
                    elif fam in {"buckboost", "buck-boost", "bb"}:
                        if idx == 0 and kind == "L":
                            target = float(sugg.get("L1", 0.0))
                        elif idx == 1 and kind == "C":
                            target = float(sugg.get("Cmid", 0.0))
                        elif idx == 2 and kind == "L":
                            target = float(sugg.get("L2", 0.0))
                        elif idx == 3 and kind == "C":
                            target = float(sugg.get("C_out", 0.0))
                    if target is None or not (float(target) > 0.0):
                        continue
                    cand_inc = _set_elem_value(cand_inc, name=str(name), value=float(target))
                    init_tuned[str(name)] = {"before": float(before), "after": float(target)}

                if cand_inc.strip() != best_inc.strip():
                    cand_detail = eval_one_detail_dcdc(
                        inc=cand_inc,
                        family=str(family),
                        vin=float(vin),
                        vout=float(vout),
                        tol=float(tol),
                        rload=float(rload),
                        t_pre=float(t_pre),
                        t_win=float(t_win),
                        sim_timeout_s=float(sim_timeout_s),
                        autotune_duty=bool(autotune_duty),
                    )
                    n_evals += 1
                    cand_score = _score(cand_detail, vout=float(vout), tol_ref=float(tol))
                    if float(cand_score) > float(best_score) + 1e-6:
                        best_score = float(cand_score)
                        best_inc = cand_inc
                        best_detail = dict(cand_detail)
                        init_meta = {"used": True, "tuned": init_tuned, "suggest": sugg}
    except Exception:
        init_meta = {"used": False, "error": "exception"}

    # ML repair corrector (proposal) before coordinate-search.
    if corrector is not None and n_evals < int(max_evals) and bool(best_detail.get("ok", False)):
        try:
            tunables = _select_tunable_names(best_inc, family=str(family))[:4]
            elems_now = {e.name: e for e in parse_inc(best_inc)}
            tun_vals: List[float] = []
            tun_mask: List[float] = []
            for name in tunables:
                e = elems_now.get(name)
                v = float(e.value) if (e and e.value is not None) else 0.0
                tun_vals.append(float(v) if float(v) > 0.0 else 0.0)
                tun_mask.append(1.0 if float(v) > 0.0 else 0.0)
            while len(tun_vals) < 4:
                tun_vals.append(0.0)
                tun_mask.append(0.0)

            feat = build_feature_vector(
                family=str(family),
                vin=float(vin),
                vout=float(vout),
                detail_before=dict(detail0),
                tunable_values=tun_vals,
                n_elems=float(best_detail.get("n_elems", 0.0) or 0.0),
            )
            scale_tries = corrector.suggest_scales(
                feature_vec=feat,
                topk=int(corrector_topk),
                max_factor=float(corrector_max_factor),
            )
            for scales in scale_tries:
                if n_evals >= int(max_evals):
                    break
                cand_inc = str(best_inc)
                corr_tuned: Dict[str, Dict[str, float]] = {}
                for i, name in enumerate(tunables):
                    if i >= 4:
                        break
                    if float(tun_mask[i]) <= 0.0:
                        continue
                    base_val = float(tun_vals[i])
                    s = float(scales[i] if i < len(scales) else 1.0)
                    cand_val = float(base_val) * float(s)
                    cand_inc = _set_elem_value(cand_inc, name=str(name), value=float(cand_val))
                    corr_tuned[str(name)] = {"before": float(base_val), "after": float(cand_val), "scale": float(s)}
                if cand_inc.strip() == best_inc.strip():
                    continue

                cand_detail = eval_one_detail_dcdc(
                    inc=cand_inc,
                    family=str(family),
                    vin=float(vin),
                    vout=float(vout),
                    tol=float(tol),
                    rload=float(rload),
                    t_pre=float(t_pre),
                    t_win=float(t_win),
                    sim_timeout_s=float(sim_timeout_s),
                    autotune_duty=bool(autotune_duty),
                )
                n_evals += 1
                cand_score = _score(cand_detail, vout=float(vout), tol_ref=float(tol))
                if float(cand_score) > float(best_score) + 1e-6:
                    best_score = float(cand_score)
                    best_inc = cand_inc
                    best_detail = dict(cand_detail)
                    corrector_meta = {
                        "used": True,
                        "tuned": corr_tuned,
                        "topk": int(corrector_topk),
                        "max_factor": float(corrector_max_factor),
                    }

            # If the corrector already fixes CV and we only care about failures, stop early.
            if bool(only_if_fail) and bool(best_detail.get("pass_CV", False)):
                return {
                    "inc_before": inc,
                    "detail_before": detail0,
                    "score_before": float(score0),
                    "inc_after": best_inc,
                    "detail_after": best_detail,
                    "score_after": float(best_score),
                    "n_evals": int(n_evals),
                    "changed": bool(best_inc.strip() != str(inc).strip()),
                    "tuned": tuned,
                    "corrector": corrector_meta or {"used": False},
                    "init": init_meta or {"used": False},
                }
        except Exception:
            # Fall back to coordinate-search repair.
            corrector_meta = {"used": False, "error": "exception"}

    for _ in range(int(max_iters)):
        improved_any = False
        tunables = _select_tunable_names(best_inc, family=str(family))
        elems_now = {e.name: e for e in parse_inc(best_inc)}
        for name in tunables:
            e = elems_now.get(name)
            if not e or e.value is None:
                continue
            base_val = float(e.value)
            best_val = base_val

            for f in factors:
                if n_evals >= int(max_evals):
                    break
                if abs(float(f) - 1.0) < 1e-9:
                    continue
                cand_val = float(base_val) * float(f)
                cand_inc = _set_elem_value(best_inc, name=name, value=cand_val)
                if cand_inc == best_inc:
                    continue
                cand_detail = eval_one_detail_dcdc(
                    inc=cand_inc,
                    family=str(family),
                    vin=float(vin),
                    vout=float(vout),
                    tol=float(tol),
                    rload=float(rload),
                    t_pre=float(t_pre),
                    t_win=float(t_win),
                    sim_timeout_s=float(sim_timeout_s),
                    autotune_duty=bool(autotune_duty),
                )
                n_evals += 1
                cand_score = _score(cand_detail, vout=float(vout), tol_ref=float(tol))
                if float(cand_score) > float(best_score) + 1e-6:
                    best_score = float(cand_score)
                    best_inc = cand_inc
                    best_detail = dict(cand_detail)
                    best_val = float(cand_val)
                    improved_any = True

            if abs(float(best_val) - float(base_val)) > 0.0:
                tuned[str(name)] = {"before": float(base_val), "after": float(best_val)}

        if not improved_any:
            break

    return {
        "inc_before": inc,
        "detail_before": detail0,
        "score_before": float(score0),
        "inc_after": best_inc,
        "detail_after": best_detail,
        "score_after": float(best_score),
        "n_evals": int(n_evals),
        "changed": bool(best_inc.strip() != str(inc).strip()),
        "tuned": tuned,
        "corrector": corrector_meta or {"used": False},
        "init": init_meta or {"used": False},
    }


def _task_key(t: Task) -> tuple[str, float, float]:
    return (str(t.family).lower(), float(t.vin), float(t.vout))


def _normalize_model_text(txt: str) -> str:
    s = (txt or "")
    if _RESP_KEY in s:
        s = s.rsplit(_RESP_KEY, 1)[-1].lstrip()
    return s


def _build_prompt(family: str, vin: float, vout: float) -> str:
    try:
        from eval_dcdc_family import build_prompt  # type: ignore

        return str(build_prompt(str(family), float(vin), float(vout)))
    except Exception:
        fam = (family or "").strip().lower()
        return (
            f"Generate a {fam} DC-DC converter in INC DSL.\n"
            "Rules:\n"
            "- Output ONLY INC lines (no explanation).\n"
            "- Line format: INC <name> <node1> <node2> <value_or_model>\n"
            "- Element names must start with R/L/C/D/S and contain a digit.\n"
            "- Use at least 20 INC lines.\n"
            f"Task: Vin={float(vin):.1f}V, Vout={float(vout):.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=2025)

    ap.add_argument("--n_gen", type=int, default=16)
    ap.add_argument("--max_rounds", type=int, default=1, help="Generation rounds per task (each round generates --n_gen).")
    ap.add_argument("--ensure_pass_cv", action="store_true", help="Keep generating until >=--min_pass_cv samples pass CV or rounds exhausted.")
    ap.add_argument("--min_pass_cv", type=int, default=1, help="Minimum #pass_CV samples required when --ensure_pass_cv is set.")
    ap.add_argument("--temp_step", type=float, default=0.0, help="Temperature increment per round when using multiple rounds.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--min_elems", type=int, default=20)
    ap.add_argument(
        "--structural_repair",
        action="store_true",
        help="Salvage verifier failures by minimally completing required topology (not a full template fallback).",
    )
    ap.add_argument(
        "--bank_expand_min_elems",
        action="store_true",
        help="If element count < --min_elems, expand via electrically-equivalent passive banks (cap splitting) before sim/repair.",
    )
    ap.add_argument("--bank_expand_max_parts", type=int, default=32, help="Max parallel parts per capacitor for bank expansion.")
    ap.add_argument(
        "--min_res_ohm",
        type=float,
        default=0.0,
        help="Skip simulation if any resistor has value < this (helps avoid ngspice stalls on overly aggressive snubbers). 0 disables.",
    )
    ap.add_argument("--pairs_per_task", type=int, default=12)
    ap.add_argument("--min_pair_gap", type=float, default=0.1)
    ap.add_argument("--top_k_chosen", type=int, default=4)
    ap.add_argument("--sft_topn_per_task", type=int, default=1, help="Write top-N chosen samples to sft_train.jsonl per task.")
    ap.add_argument(
        "--sft_strict_topn_per_task",
        type=int,
        default=1,
        help="Write top-N pass_CV samples to sft_train_strict.jsonl per task (0=all pass_CV).",
    )
    ap.add_argument(
        "--allow_fail_chosen",
        action="store_true",
        help="Allow 'chosen' samples that do NOT pass CV (not recommended for PVPO/DPO). Default requires pass_CV chosen.",
    )
    ap.add_argument("--max_tasks", type=int, default=0, help="0 means all tasks")
    ap.add_argument(
        "--tasks_jsonl",
        default="",
        help="Optional frozen JSONL task manifest. When set, only these tasks can generate candidates or training data.",
    )
    ap.add_argument("--task_shard_id", type=int, default=0, help="Task shard id in [0, --task_shard_count).")
    ap.add_argument("--task_shard_count", type=int, default=1, help="Total task shards. 1 disables sharding.")
    ap.add_argument(
        "--skip_done_tasks_jsonl",
        default="",
        help="Optional JSONL file with {'family','vin','vout'} tasks already completed elsewhere; will be skipped.",
    )
    ap.add_argument(
        "--upgrade_done_tasks",
        action="store_true",
        help=(
            "When used with --resume, revisit already-done tasks to meet higher targets "
            "(e.g. larger --n_gen / --pairs_per_task / --min_pass_cv). This loads existing "
            "tasks/*/scored.json and existing pairs_meta.jsonl to append only missing examples."
        ),
    )
    ap.add_argument(
        "--upgrade_prev_n_gen",
        type=int,
        default=0,
        help=(
            "Estimated previous --n_gen used to build existing done tasks (used to infer how many "
            "raw generations were already attempted). 0 tries to infer per-task from pairs_meta.jsonl, "
            "fallback to current --n_gen."
        ),
    )

    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--rload", type=float, default=10.0)
    ap.add_argument("--t_pre", type=float, default=0.008)
    ap.add_argument("--t_win", type=float, default=0.002)
    ap.add_argument("--sim_timeout_s", type=float, default=60.0)
    ap.add_argument(
        "--sim_workers",
        type=int,
        default=0,
        help="Parallel simulation/repair workers per task. 0=auto (uses CPU affinity; e.g., nproc inside container).",
    )
    ap.add_argument("--autotune_duty", action="store_true")
    ap.add_argument("--eda_repair", action="store_true", help="Enable industrial EDA local-search repair (duty + L/C grid).")
    ap.add_argument(
        "--repair_factors",
        default="0.5,0.8,1.0,1.25,1.5,2.0",
        help="Multiplicative factors for L/C coordinate-search (comma-separated).",
    )
    ap.add_argument("--repair_max_evals", type=int, default=25, help="Max #sim evaluations per sample (including baseline).")
    ap.add_argument("--repair_max_iters", type=int, default=2, help="Coordinate-search passes over tunable elements.")
    ap.add_argument(
        "--repair_always",
        action="store_true",
        help="Also run L/C local-search even if pass_CV is already True (default: only repair failures).",
    )
    ap.add_argument("--repair_corrector", default="", help="Optional trained ML corrector checkpoint (model.pt) used before search.")
    ap.add_argument("--repair_corrector_topk", type=int, default=3)
    ap.add_argument("--repair_corrector_max_factor", type=float, default=2.0)

    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # Avoid tokenizers fork warnings (we spawn many ngspice subprocesses).
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Prevent PyTorch CPU ops (repair corrector) from oversubscribing threads.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    sim_workers = max(1, int(_auto_sim_workers(int(getattr(args, "sim_workers", 0) or 0))))

    shard_count = max(1, int(getattr(args, "task_shard_count", 1) or 1))
    shard_id = int(getattr(args, "task_shard_id", 0) or 0)
    if shard_id < 0 or shard_id >= shard_count:
        raise SystemExit(f"--task_shard_id must be in [0,{shard_count}), got {shard_id}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    (out_root / "tasks").mkdir(parents=True, exist_ok=True)

    # output files
    pairs_path = out_root / "dpo_pairs.jsonl"
    sft_path = out_root / "sft_train.jsonl"
    sft_strict_path = out_root / "sft_train_strict.jsonl"
    meta_path = out_root / "pairs_meta.jsonl"
    done_path = out_root / "done_tasks.jsonl"
    progress_path = out_root / "progress.txt"
    report_path = out_root / "build_report.json"

    done: set[tuple[str, float, float]] = set()
    if bool(args.resume) and done_path.exists():
        for line in done_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((str(r["family"]).lower(), float(r["vin"]), float(r["vout"])))
            except Exception:
                continue

    # Optional global skip list (e.g., merged done tasks across shards). When upgrading, we still
    # want to allow revisiting tasks inside this shard; sharding already prevents duplicates.
    if (not bool(getattr(args, "upgrade_done_tasks", False))) and str(getattr(args, "skip_done_tasks_jsonl", "") or "").strip():
        p = Path(str(args.skip_done_tasks_jsonl).strip())
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((str(r["family"]).lower(), float(r["vin"]), float(r["vout"])))
                except Exception:
                    continue
        else:
            print(f"[warn] skip_done_tasks_jsonl not found: {p}", flush=True)

    # If upgrading, read existing pairs_meta (to avoid duplicating pairs and to infer previous n_gen),
    # and read existing SFT JSONL (to avoid duplicate prompt+INC rows).
    prev_pairs_seen: Dict[tuple[str, float, float], set[tuple[str, str]]] = {}
    prev_pairs_n: Dict[tuple[str, float, float], int] = {}
    prev_task_n_gen: Dict[tuple[str, float, float], int] = {}
    seen_sft: set[str] = set()
    seen_sft_strict: set[str] = set()
    if bool(args.resume) and bool(getattr(args, "upgrade_done_tasks", False)):
        if meta_path.exists() and meta_path.stat().st_size > 0:
            for line in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    t = r.get("task") or {}
                    key = (str(t.get("family") or "").lower(), float(t.get("vin")), float(t.get("vout")))
                    ch = str(((r.get("chosen") or {}) or {}).get("canonical_hash") or "")
                    rh = str(((r.get("rejected") or {}) or {}).get("canonical_hash") or "")
                    if key[0] and ch and rh:
                        s = prev_pairs_seen.setdefault(key, set())
                        s.add((ch, rh))
                        prev_pairs_n[key] = int(len(s))
                    ng = int(r.get("n_gen") or 0)
                    if key[0] and ng > 0:
                        prev_task_n_gen[key] = int(max(int(prev_task_n_gen.get(key, 0) or 0), int(ng)))
                except Exception:
                    continue
        if sft_path.exists() and sft_path.stat().st_size > 0:
            for line in sft_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    txt = str(r.get("text") or "")
                    if txt:
                        seen_sft.add(_sha1_text(txt))
                except Exception:
                    continue
        if sft_strict_path.exists() and sft_strict_path.stat().st_size > 0:
            for line in sft_strict_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    txt = str(r.get("text") or "")
                    if txt:
                        seen_sft_strict.add(_sha1_text(txt))
                except Exception:
                    continue

    # open output in append vs write mode
    mode = "a" if bool(args.resume) else "w"
    f_pairs = pairs_path.open(mode, encoding="utf-8")
    f_sft = sft_path.open(mode, encoding="utf-8")
    f_sft_strict = sft_strict_path.open(mode, encoding="utf-8")
    f_meta = meta_path.open(mode, encoding="utf-8")
    f_done = done_path.open("a", encoding="utf-8")

    tok, model = _load_model(args.base_model, str(args.adapter))
    dev = _device(model)

    from dcdc_eval_tran import eval_one_detail_dcdc  # local import on server

    corrector: Optional[RepairCorrector] = None
    if str(args.repair_corrector or "").strip():
        p = Path(str(args.repair_corrector).strip())
        if p.exists():
            try:
                corrector = RepairCorrector.load(str(p), device="cpu")
                print(f"[repair-corrector] loaded: {p}")
            except Exception as e:
                print(f"[repair-corrector] failed to load {p}: {e}")
        else:
            print(f"[repair-corrector] not found: {p}")

    tasks_manifest = str(getattr(args, "tasks_jsonl", "") or "").strip()
    tasks = load_tasks_jsonl(tasks_manifest) if tasks_manifest else default_taskset()
    if int(args.max_tasks) > 0:
        tasks = tasks[: int(args.max_tasks)]

    total_pairs = 0
    total_pairs_repair = 0
    total_pairs_sample = 0
    total_sft = 0
    total_sft_strict = 0
    total_tasks_done = 0
    total_tasks_skipped = 0

    executor: Optional[ThreadPoolExecutor] = None
    if int(sim_workers) > 1:
        executor = ThreadPoolExecutor(max_workers=int(sim_workers))
        print(f"[sim] ThreadPoolExecutor workers={int(sim_workers)} (affinity_cpus={_effective_cpus()})", flush=True)
    else:
        print(f"[sim] workers=1 (affinity_cpus={_effective_cpus()})", flush=True)

    try:
        for ti, task in enumerate(tasks):
            if shard_count > 1 and (int(ti) % int(shard_count)) != int(shard_id):
                continue
            fam, vin, vout = str(task.family), float(task.vin), float(task.vout)
            key = (fam.lower(), vin, vout)
            upgrading = bool(getattr(args, "upgrade_done_tasks", False)) and bool(args.resume) and (key in done)
            if (key in done) and (not upgrading):
                total_tasks_skipped += 1
                continue

            seed_base = int(args.seed) + ti * 1000
            random.seed(seed_base)
            torch.manual_seed(seed_base)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_base)

            prompt = _build_prompt(fam, vin, vout)
            task_dir = out_root / "tasks" / fam.lower() / f"vin{vin:.1f}_vout{vout:.1f}"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            scored: List[Dict[str, Any]] = []
            best_by_hash: Dict[str, Dict[str, Any]] = {}

            # Upgrade mode: load existing scored/pairs and append only missing examples.
            prev_pairs = int(prev_pairs_n.get(key, 0) or 0) if upgrading else 0
            prev_max_index = 0
            prev_rounds_est = 0
            prev_n_gen_est = 0
            idx_base = 0
            ridx_offset = 0
            need_first = 0

            if upgrading:
                scored_path = task_dir / "scored.json"
                if scored_path.exists() and scored_path.stat().st_size > 0:
                    try:
                        old_scored = json.loads(scored_path.read_text(encoding="utf-8", errors="ignore"))
                        if isinstance(old_scored, list):
                            for r in old_scored:
                                if not isinstance(r, dict):
                                    continue
                                d = r.get("detail") or {}
                                ch = str(d.get("canonical_hash") or r.get("canonical_hash") or _sha1_text(str(r.get("inc") or "")))
                                if not ch:
                                    continue
                                prev = best_by_hash.get(ch)
                                if (prev is None) or float(r.get("score", -1e9) or -1e9) > float(prev.get("score", -1e9) or -1e9):
                                    best_by_hash[ch] = r
                                try:
                                    prev_max_index = int(max(prev_max_index, int(r.get("index") or 0)))
                                except Exception:
                                    pass
                    except Exception:
                        pass

                pass_cv_hits = sum(
                    1
                    for r in best_by_hash.values()
                    if bool((r.get("detail") or {}).get("ok")) and bool((r.get("detail") or {}).get("pass_CV", False))
                )

                # Infer previous n_gen from existing pairs_meta (preferred) or user-provided flag.
                prev_n_gen_est = int(prev_task_n_gen.get(key, 0) or 0)
                if prev_n_gen_est <= 0:
                    prev_n_gen_est = int(getattr(args, "upgrade_prev_n_gen", 0) or 0)
                if prev_n_gen_est <= 0:
                    prev_n_gen_est = int(args.n_gen)
                prev_n_gen_est = max(1, int(prev_n_gen_est))

                if prev_max_index > 0:
                    prev_rounds_est = max(1, int(math.ceil(float(prev_max_index) / float(prev_n_gen_est))))
                else:
                    prev_rounds_est = 0

                prev_total_gen_est = int(prev_rounds_est) * int(prev_n_gen_est) if prev_rounds_est > 0 else 0
                need_first = max(0, int(args.n_gen) - int(prev_total_gen_est))
                need_pairs = max(0, int(args.pairs_per_task) - int(prev_pairs))
                need_more_for_pass = bool(args.ensure_pass_cv) and (pass_cv_hits < int(args.min_pass_cv))
                if (need_first <= 0) and (need_pairs <= 0) and (not need_more_for_pass):
                    total_tasks_skipped += 1
                    continue

                idx_base = int(prev_max_index)
                ridx_offset = int(prev_rounds_est)
            else:
                pass_cv_hits = 0

            rounds = max(1, int(args.max_rounds))
            if upgrading:
                rounds = max(0, int(rounds) - int(prev_rounds_est))
                # If we only need to top-up pairs (not more generations), skip generation.
                if (int(need_first) <= 0) and (not (bool(args.ensure_pass_cv) and pass_cv_hits < int(args.min_pass_cv))):
                    rounds = 0

            for ridx in range(int(rounds)):
                ridx_eff = int(ridx_offset) + int(ridx)
                # Vary RNG and (optionally) temperature across rounds for harder tasks.
                seed_round = int(seed_base) + int(ridx_eff) * 97
                random.seed(seed_round)
                torch.manual_seed(seed_round)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_round)

                temperature = float(args.temperature) + float(ridx_eff) * float(args.temp_step)
                temperature = max(1e-3, float(temperature))

                enc = tok(prompt, return_tensors="pt").to(dev)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                n_gen_this = int(args.n_gen)
                if upgrading and int(need_first) > 0:
                    n_gen_this = int(need_first)
                    need_first = 0
                with torch.inference_mode():
                    outs = model.generate(
                        **enc,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=True,
                        temperature=float(temperature),
                        top_p=float(args.top_p),
                        eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.pad_token_id,
                        num_return_sequences=int(n_gen_this),
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gen_dt = time.time() - t0

                n_outs = int(outs.shape[0])
                gen_time_each = float(gen_dt) / max(1.0, float(n_outs))
                ridx_base = int(idx_base) if upgrading else int(ridx_eff) * int(args.n_gen)
                raw_texts = [tok.decode(outs[j], skip_special_tokens=True) for j in range(n_outs)]

                def _eval_one(i: int, raw_text: str) -> tuple[str, Dict[str, Any]]:
                    raw = str(raw_text)
                    txt = _normalize_model_text(raw)
                    inc_lines = extract_inc_lines(txt)
                    inc = ("\n".join(inc_lines).strip() + "\n") if inc_lines else ""
                    pre_repair: Optional[Dict[str, Any]] = None

                    ver = verify_inc_dcdc(inc, family=fam, vin=vin, vout=vout)
                    violations = list(ver.violations)
                    meets_min = int(ver.n_elems) >= int(args.min_elems)
                    if not meets_min:
                        violations.append(f"too_few_elems_{int(ver.n_elems)}")

                    if (not ver.ok) or (not meets_min):
                        inc_before = str(inc)
                        inc_after = str(inc)
                        bank_meta: Optional[Dict[str, Any]] = None
                        if bool(args.structural_repair):
                            inc_after = _structural_complete_minimal(inc_after, family=fam, vin=vin, vout=vout)
                        if bool(args.bank_expand_min_elems):
                            inc_after, bank_meta = _bank_expand_to_min_elems(
                                inc_after,
                                family=fam,
                                min_elems=int(args.min_elems),
                                max_parts_per_elem=int(args.bank_expand_max_parts),
                            )
                        ver2 = verify_inc_dcdc(inc_after, family=fam, vin=vin, vout=vout)
                        meets_min2 = int(ver2.n_elems) >= int(args.min_elems)
                        if ver2.ok and meets_min2 and str(inc_after).strip():
                            pre_repair = {
                                "used": True,
                                "type": "structural_bank",
                                "changed": bool(str(inc_before).strip() != str(inc_after).strip()),
                                "inc_before": inc_before,
                                "inc_after": inc_after,
                                "detail_before": {
                                    "ok": False,
                                    "pass_C": bool(ver.ok),
                                    "pass_CV": False,
                                    "pass_CE": False,
                                    "canonical_hash": ver.canonical_hash,
                                    "violations": violations,
                                    "error": "verify_or_min_elems",
                                },
                                "bank": bank_meta or {"used": False},
                                "score_before": -1.0,
                            }
                            inc = inc_after
                            ver = ver2
                            violations = list(ver2.violations)
                            meets_min = meets_min2

                    if (not ver.ok) or (not meets_min):
                        detail = {
                            "ok": False,
                            "pass_C": False,
                            "canonical_hash": ver.canonical_hash,
                            "violations": violations,
                            "error": "verify_or_min_elems",
                        }
                        score = -1.0
                        repair = None
                    else:
                        inc_after = inc
                        repair = None
                        if float(args.min_res_ohm) > 0.0:
                            try:
                                elems = parse_inc(inc)
                                if any(e.kind == "R" and (e.value is not None) and float(e.value) < float(args.min_res_ohm) for e in elems):
                                    detail = {
                                        "ok": False,
                                        "pass_C": True,
                                        "canonical_hash": ver.canonical_hash,
                                        "violations": violations + [f"min_res_ohm_{float(args.min_res_ohm):g}"],
                                        "error": "heuristic_min_res_ohm",
                                    }
                                    score = -1.0
                                    repair = None
                                else:
                                    if bool(args.eda_repair):
                                        factors = [float(x.strip()) for x in str(args.repair_factors).split(",") if x.strip()]
                                        repair = _eda_repair_local_search(
                                            inc=inc,
                                            family=fam,
                                            vin=vin,
                                            vout=vout,
                                            tol=float(args.tol),
                                            rload=float(args.rload),
                                            t_pre=float(args.t_pre),
                                            t_win=float(args.t_win),
                                            sim_timeout_s=float(args.sim_timeout_s),
                                            autotune_duty=bool(args.autotune_duty),
                                            factors=factors,
                                            max_evals=int(args.repair_max_evals),
                                            max_iters=int(args.repair_max_iters),
                                            only_if_fail=(not bool(args.repair_always)),
                                            corrector=corrector,
                                            corrector_topk=int(args.repair_corrector_topk),
                                            corrector_max_factor=float(args.repair_corrector_max_factor),
                                            eval_one_detail_dcdc=eval_one_detail_dcdc,
                                        )
                                        inc_after = str(repair.get("inc_after") or inc)
                                        detail = repair.get("detail_after") or {}
                                        score = float(repair.get("score_after", -1.0))
                                    else:
                                        detail = eval_one_detail_dcdc(
                                            inc=inc,
                                            family=fam,
                                            vin=vin,
                                            vout=vout,
                                            tol=float(args.tol),
                                            rload=float(args.rload),
                                            t_pre=float(args.t_pre),
                                            t_win=float(args.t_win),
                                            sim_timeout_s=float(args.sim_timeout_s),
                                            autotune_duty=bool(args.autotune_duty),
                                        )
                                        score = _score(detail, vout=vout, tol_ref=float(args.tol))
                            except Exception:
                                if bool(args.eda_repair):
                                    factors = [float(x.strip()) for x in str(args.repair_factors).split(",") if x.strip()]
                                    repair = _eda_repair_local_search(
                                        inc=inc,
                                        family=fam,
                                        vin=vin,
                                        vout=vout,
                                        tol=float(args.tol),
                                        rload=float(args.rload),
                                        t_pre=float(args.t_pre),
                                        t_win=float(args.t_win),
                                        sim_timeout_s=float(args.sim_timeout_s),
                                        autotune_duty=bool(args.autotune_duty),
                                        factors=factors,
                                        max_evals=int(args.repair_max_evals),
                                        max_iters=int(args.repair_max_iters),
                                        only_if_fail=(not bool(args.repair_always)),
                                        corrector=corrector,
                                        corrector_topk=int(args.repair_corrector_topk),
                                        corrector_max_factor=float(args.repair_corrector_max_factor),
                                        eval_one_detail_dcdc=eval_one_detail_dcdc,
                                    )
                                    inc_after = str(repair.get("inc_after") or inc)
                                    detail = repair.get("detail_after") or {}
                                    score = float(repair.get("score_after", -1.0))
                                else:
                                    detail = eval_one_detail_dcdc(
                                        inc=inc,
                                        family=fam,
                                        vin=vin,
                                        vout=vout,
                                        tol=float(args.tol),
                                        rload=float(args.rload),
                                        t_pre=float(args.t_pre),
                                        t_win=float(args.t_win),
                                        sim_timeout_s=float(args.sim_timeout_s),
                                        autotune_duty=bool(args.autotune_duty),
                                    )
                                    score = _score(detail, vout=vout, tol_ref=float(args.tol))
                        else:
                            if bool(args.eda_repair):
                                factors = [float(x.strip()) for x in str(args.repair_factors).split(",") if x.strip()]
                                repair = _eda_repair_local_search(
                                    inc=inc,
                                    family=fam,
                                    vin=vin,
                                    vout=vout,
                                    tol=float(args.tol),
                                    rload=float(args.rload),
                                    t_pre=float(args.t_pre),
                                    t_win=float(args.t_win),
                                    sim_timeout_s=float(args.sim_timeout_s),
                                    autotune_duty=bool(args.autotune_duty),
                                    factors=factors,
                                    max_evals=int(args.repair_max_evals),
                                    max_iters=int(args.repair_max_iters),
                                    only_if_fail=(not bool(args.repair_always)),
                                    corrector=corrector,
                                    corrector_topk=int(args.repair_corrector_topk),
                                    corrector_max_factor=float(args.repair_corrector_max_factor),
                                    eval_one_detail_dcdc=eval_one_detail_dcdc,
                                )
                                inc_after = str(repair.get("inc_after") or inc)
                                detail = repair.get("detail_after") or {}
                                score = float(repair.get("score_after", -1.0))
                            else:
                                detail = eval_one_detail_dcdc(
                                    inc=inc,
                                    family=fam,
                                    vin=vin,
                                    vout=vout,
                                    tol=float(args.tol),
                                    rload=float(args.rload),
                                    t_pre=float(args.t_pre),
                                    t_win=float(args.t_win),
                                    sim_timeout_s=float(args.sim_timeout_s),
                                    autotune_duty=bool(args.autotune_duty),
                                )
                                score = _score(detail, vout=vout, tol_ref=float(args.tol))

                        # If repair changed the INC, store it for downstream PVPO/SFT.
                        if str(inc_after).strip():
                            inc = str(inc_after).strip() + "\n"

                    ch = str(detail.get("canonical_hash") or ver.canonical_hash)
                    if pre_repair is not None and bool(pre_repair.get("used", False)):
                        pre_repair["detail_after"] = detail
                        pre_repair["score_after"] = float(score)
                    row = {
                        "family": fam.lower(),
                        "vin": float(vin),
                        "vout": float(vout),
                        "index": int(ridx_base + i + 1),
                        "gen_time": float(gen_time_each),
                        "n_elems": int(getattr(ver, "n_elems", 0)),
                        "inc": inc,
                        "score": float(score),
                        "detail": detail,
                        "cv_err": float(_cv_err(detail, vout=float(vout))),
                        "repair": repair,
                        "repair_pre": pre_repair,
                    }
                    return ch, row

                if executor is None:
                    for i, raw in enumerate(raw_texts):
                        ch, row = _eval_one(int(i), str(raw))
                        prev = best_by_hash.get(ch)
                        if (prev is None) or float(row["score"]) > float(prev["score"]):
                            best_by_hash[ch] = row
                else:
                    futs = [executor.submit(_eval_one, int(i), str(raw)) for i, raw in enumerate(raw_texts)]
                    for fut in as_completed(futs):
                        ch, row = fut.result()
                        prev = best_by_hash.get(ch)
                        if (prev is None) or float(row["score"]) > float(prev["score"]):
                            best_by_hash[ch] = row

                if upgrading:
                    idx_base = int(idx_base) + int(n_outs)

                # Early stop if we already found enough CV-passing candidates.
                if bool(args.ensure_pass_cv):
                    pass_cv_hits = sum(
                        1
                        for r in best_by_hash.values()
                        if bool((r.get("detail") or {}).get("ok")) and bool((r.get("detail") or {}).get("pass_CV", False))
                    )
                    if pass_cv_hits >= int(args.min_pass_cv):
                        if (not upgrading) or int(need_first) <= 0:
                            break

                if upgrading and (not bool(args.ensure_pass_cv)) and int(need_first) <= 0:
                    break

            scored = list(best_by_hash.values())
            scored.sort(key=lambda r: float(r.get("score", -1e9)), reverse=True)
            (task_dir / "scored.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")

            valids = [r for r in scored if bool((r.get("detail") or {}).get("ok"))]
            if not valids:
                progress_path.write_text(
                    f"task={ti+1}/{len(tasks)} fam={fam} vin={vin} vout={vout} SKIP (no_valid)\n",
                    encoding="utf-8",
                )
                if key not in done:
                    done.add(key)
                    f_done.write(json.dumps({"family": fam, "vin": vin, "vout": vout}, ensure_ascii=False) + "\n")
                    f_done.flush()
                total_tasks_done += 1
                continue

            pass_cv_valids = [r for r in valids if bool((r.get("detail") or {}).get("pass_CV", False))]
            if (not pass_cv_valids) and (not bool(args.allow_fail_chosen)):
                # Still add a best-effort SFT row (structure learning), but DO NOT emit DPO pairs
                # when we have no CV-pass candidates under the current tol.
                best = valids[0]
                txt = prompt + str(best.get("inc") or "")
                h = _sha1_text(txt) if txt else ""
                if h and h not in seen_sft:
                    seen_sft.add(h)
                    f_sft.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                    f_sft.flush()
                    total_sft += 1

                progress_path.write_text(
                    f"task={ti+1}/{len(tasks)} fam={fam} vin={vin} vout={vout} unique={len(scored)} ok={len(valids)} pass_cv=0 pairs=+0 SKIP_DPO(no_pass_cv)\n",
                    encoding="utf-8",
                )
                if key not in done:
                    done.add(key)
                    f_done.write(json.dumps({"family": fam, "vin": vin, "vout": vout}, ensure_ascii=False) + "\n")
                    f_done.flush()
                total_tasks_done += 1
                continue
            # Prefer Pareto-front candidates as "chosen" (multi-objective, within-task),
            # but only among CV-pass samples.
            pareto_pool: List[Dict[str, Any]] = []
            if pass_cv_valids:
                pareto_pool = _pareto_front(pass_cv_valids, vout=float(vout))
                pareto_pool.sort(key=lambda r: float(r.get("score", -1e9)), reverse=True)

            top_k = max(1, min(int(args.top_k_chosen), len(valids)))
            chosen_pool = (pareto_pool[:top_k] if pareto_pool else (pass_cv_valids[:top_k] if pass_cv_valids else valids[:top_k]))
            rejected_pool = scored  # include invalid as negatives

            pairs_written = int(prev_pairs) if upgrading else 0
            seen_pairs: set[tuple[str, str]] = set(prev_pairs_seen.get(key, set()) or set()) if upgrading else set()
            tries = 0

            # High-confidence pre-repair preference pairs: (final, raw) for samples salvaged by structural/bank completion.
            for rec0 in list(valids):
                if pairs_written >= int(args.pairs_per_task):
                    break
                rep0 = rec0.get("repair_pre") if isinstance(rec0, dict) else None
                if not isinstance(rep0, dict) or not bool(rep0.get("used", False)):
                    continue
                if not bool(rep0.get("changed", False)):
                    continue
                d_after = rep0.get("detail_after") or rec0.get("detail") or {}
                if not (bool(d_after.get("ok", False)) and bool(d_after.get("pass_CV", False))):
                    continue
                inc_after = str(rec0.get("inc") or "")
                inc_before = str(rep0.get("inc_before") or "")
                if not inc_after.strip() or not inc_before.strip():
                    continue
                h_after = str((d_after.get("canonical_hash") or _sha1_text(inc_after)) or "")
                h_before = str(((rep0.get("detail_before") or {}).get("canonical_hash") or _sha1_text(inc_before)) or "")
                if not h_after or not h_before or h_after == h_before:
                    continue
                gap0 = float(rec0.get("score", -1e9)) - float(rep0.get("score_before", -1.0))
                if gap0 < float(args.min_pair_gap):
                    continue
                key_pair0 = (h_after, h_before)
                if key_pair0 in seen_pairs:
                    continue
                seen_pairs.add(key_pair0)
                pairs_written += 1
                f_pairs.write(json.dumps({"prompt": prompt, "chosen": inc_after, "rejected": inc_before}, ensure_ascii=False) + "\n")
                f_pairs.flush()
                total_pairs += 1
                total_pairs_repair += 1
                f_meta.write(
                    json.dumps(
                        {
                            "task": asdict(task),
                            "prompt_seed": int(seed_base),
                            "type": "pre_repair_pair",
                            "chosen": {"canonical_hash": h_after, "score": float(rec0.get("score", -1e9)), "detail": d_after},
                            "rejected": {
                                "canonical_hash": h_before,
                                "score": float(rep0.get("score_before", -1.0)),
                                "detail": rep0.get("detail_before") or {},
                            },
                            "gap": float(gap0),
                            "n_gen": int(args.n_gen),
                            "n_unique": int(len(scored)),
                            "autotune_duty": bool(args.autotune_duty),
                            "pre_repair": {"bank": rep0.get("bank") or {"used": False}},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f_meta.flush()

            # High-confidence repair preference pairs: (after, before) for the same candidate.
            # This is the cleanest PVPO signal (teacher from itself + EDA local search).
            for rec0 in list(valids):
                if pairs_written >= int(args.pairs_per_task):
                    break
                rep0 = rec0.get("repair") if isinstance(rec0, dict) else None
                if not isinstance(rep0, dict):
                    continue
                if not bool(rep0.get("changed", False)):
                    continue
                d_after = rep0.get("detail_after") or rec0.get("detail") or {}
                d_before = rep0.get("detail_before") or {}
                if not (bool(d_after.get("ok", False)) and bool(d_after.get("pass_CV", False))):
                    continue
                inc_after = str(rep0.get("inc_after") or rec0.get("inc") or "")
                inc_before = str(rep0.get("inc_before") or "")
                if not inc_after.strip() or not inc_before.strip():
                    continue
                h_after = str((d_after.get("canonical_hash") or _sha1_text(inc_after)) or "")
                h_before = str((d_before.get("canonical_hash") or _sha1_text(inc_before)) or "")
                if not h_after or not h_before or h_after == h_before:
                    continue
                gap0 = float(rep0.get("score_after", rec0.get("score", -1e9)) or -1e9) - float(rep0.get("score_before", -1e9) or -1e9)
                if gap0 < float(args.min_pair_gap):
                    continue
                key_pair0 = (h_after, h_before)
                if key_pair0 in seen_pairs:
                    continue
                seen_pairs.add(key_pair0)
                pairs_written += 1
                f_pairs.write(json.dumps({"prompt": prompt, "chosen": inc_after, "rejected": inc_before}, ensure_ascii=False) + "\n")
                f_pairs.flush()
                total_pairs += 1
                total_pairs_repair += 1
                f_meta.write(
                    json.dumps(
                        {
                            "task": asdict(task),
                            "prompt_seed": int(seed_base),
                            "type": "repair_pair",
                            "chosen": {"canonical_hash": h_after, "score": float(rep0.get("score_after", -1e9)), "detail": d_after},
                            "rejected": {"canonical_hash": h_before, "score": float(rep0.get("score_before", -1e9)), "detail": d_before},
                            "gap": float(gap0),
                            "n_gen": int(args.n_gen),
                            "n_unique": int(len(scored)),
                            "autotune_duty": bool(args.autotune_duty),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f_meta.flush()

            while pairs_written < int(args.pairs_per_task) and tries < int(args.pairs_per_task) * 50:
                tries += 1
                c = random.choice(chosen_pool)
                r = random.choice(rejected_pool)
                ch_c = str((c.get("detail") or {}).get("canonical_hash") or "")
                ch_r = str((r.get("detail") or {}).get("canonical_hash") or "")
                if not ch_c or not ch_r or ch_c == ch_r:
                    continue
                key_pair = (ch_c, ch_r)
                if key_pair in seen_pairs:
                    continue
                gap = float(c.get("score", -1e9)) - float(r.get("score", -1e9))
                if gap < float(args.min_pair_gap):
                    continue

                seen_pairs.add(key_pair)
                pairs_written += 1

                f_pairs.write(
                    json.dumps({"prompt": prompt, "chosen": str(c.get("inc") or ""), "rejected": str(r.get("inc") or "")}, ensure_ascii=False)
                    + "\n"
                )
                f_pairs.flush()
                total_pairs += 1
                total_pairs_sample += 1

                f_meta.write(
                    json.dumps(
                        {
                            "task": asdict(task),
                            "prompt_seed": int(seed_base),
                            "type": "sample_pair",
                            "chosen": {
                                "canonical_hash": ch_c,
                                "score": float(c.get("score", -1e9)),
                                "detail": c.get("detail") or {},
                            },
                            "rejected": {
                                "canonical_hash": ch_r,
                                "score": float(r.get("score", -1e9)),
                                "detail": r.get("detail") or {},
                            },
                            "gap": float(gap),
                            "n_gen": int(args.n_gen),
                            "n_unique": int(len(scored)),
                            "autotune_duty": bool(args.autotune_duty),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f_meta.flush()

            # SFT: write top-N chosen samples (structure + sizing); strict SFT uses only pass_CV samples.
            sft_topn = max(1, int(getattr(args, "sft_topn_per_task", 1) or 1))
            strict_topn = int(getattr(args, "sft_strict_topn_per_task", 1) or 0)

            sft_take = list(chosen_pool[:sft_topn]) if chosen_pool else list(valids[:sft_topn])
            for rec in sft_take:
                txt = prompt + str(rec.get("inc") or "")
                h = _sha1_text(txt) if txt else ""
                if h and h not in seen_sft:
                    seen_sft.add(h)
                    f_sft.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                    total_sft += 1
            f_sft.flush()

            strict_pool = [r for r in pass_cv_valids if bool((r.get("detail") or {}).get("pass_CV", False))]
            strict_take = strict_pool if strict_topn <= 0 else strict_pool[: max(0, strict_topn)]
            for rec in strict_take:
                txt = prompt + str(rec.get("inc") or "")
                h = _sha1_text(txt) if txt else ""
                if h and h not in seen_sft_strict:
                    seen_sft_strict.add(h)
                    f_sft_strict.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                    total_sft_strict += 1
            if strict_take:
                f_sft_strict.flush()

            progress_path.write_text(
                f"task={ti+1}/{len(tasks)} fam={fam} vin={vin} vout={vout} unique={len(scored)} ok={len(valids)} pass_cv={len(pass_cv_valids)} pairs=+{pairs_written}\n",
                encoding="utf-8",
            )

            if key not in done:
                done.add(key)
                f_done.write(json.dumps({"family": fam, "vin": vin, "vout": vout}, ensure_ascii=False) + "\n")
                f_done.flush()
            total_tasks_done += 1

        report = {
            "started_at": _now(),
            "base_model": str(args.base_model),
            "adapter": str(args.adapter),
            "out_root": str(out_root),
            "tasks_total": len(tasks),
            "tasks_jsonl": str(Path(tasks_manifest).resolve()) if tasks_manifest else "",
            "tasks_jsonl_sha256": sha256_file(tasks_manifest) if tasks_manifest else "",
            "tasks_done": int(total_tasks_done),
            "tasks_skipped": int(total_tasks_skipped),
            "pairs_total": int(total_pairs),
            "pairs_repair_total": int(total_pairs_repair),
            "pairs_sample_total": int(total_pairs_sample),
            "sft_rows": int(total_sft),
            "sft_rows_strict": int(total_sft_strict),
            "n_gen": int(args.n_gen),
            "max_rounds": int(args.max_rounds),
            "ensure_pass_cv": bool(args.ensure_pass_cv),
            "min_pass_cv": int(args.min_pass_cv),
            "temp_step": float(args.temp_step),
            "pairs_per_task": int(args.pairs_per_task),
            "min_pair_gap": float(args.min_pair_gap),
            "min_elems": int(args.min_elems),
            "structural_repair": bool(args.structural_repair),
            "bank_expand_min_elems": bool(args.bank_expand_min_elems),
            "bank_expand_max_parts": int(args.bank_expand_max_parts),
            "autotune_duty": bool(args.autotune_duty),
            "allow_fail_chosen": bool(args.allow_fail_chosen),
            "tol": float(args.tol),
            "rload": float(args.rload),
            "t_pre": float(args.t_pre),
            "t_win": float(args.t_win),
            "eda_repair": bool(args.eda_repair),
            "repair_factors": str(args.repair_factors),
            "repair_max_evals": int(args.repair_max_evals),
            "repair_max_iters": int(args.repair_max_iters),
            "repair_always": bool(args.repair_always),
            "repair_corrector": str(args.repair_corrector),
            "repair_corrector_topk": int(args.repair_corrector_topk),
            "repair_corrector_max_factor": float(args.repair_corrector_max_factor),
            "repair_corrector_loaded": bool(corrector is not None),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        try:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass
        for f in [f_pairs, f_sft, f_sft_strict, f_meta, f_done]:
            try:
                f.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

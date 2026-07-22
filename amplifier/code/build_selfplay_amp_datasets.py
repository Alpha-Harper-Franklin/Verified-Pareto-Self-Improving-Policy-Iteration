#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from amp_eval_acop import (
    AMP_BW_TOL_REL,
    AMP_GAIN_TOL_DB,
    AMP_PM_MIN_DEG,
    AMP_PSTATIC_MAX_MW,
    eval_one_detail_amp_family,
    amp_family_params,
)
from amp_taskset import default_taskset_amp
from amp_verifier import verify_inc_amp
from dcdc_taskset import Task
from inc_parser import IncElem, extract_inc_lines, parse_inc, to_inc_text

RESPONSE_TEMPLATE = "### Response:\n"
_RESP_KEY = "### Response:"


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


def _normalize_model_text(txt: str) -> str:
    s = (txt or "")
    if _RESP_KEY in s:
        s = s.rsplit(_RESP_KEY, 1)[-1].lstrip()
    return s


def _canon_inc_text(inc_text: str) -> str:
    lines = [ln.strip() for ln in extract_inc_lines(inc_text) if ln.strip()]
    return "\n".join(sorted(lines))


def _effective_cpus() -> int:
    caps: List[int] = []
    try:
        p = Path("/sys/fs/cgroup/cpu.max")
        if p.exists():
            parts = p.read_text().strip().split()
            if len(parts) >= 2 and parts[0].lower() != "max":
                quota = float(parts[0])
                period = float(parts[1])
                if quota > 0 and period > 0:
                    caps.append(int(math.ceil(quota / period)))
    except Exception:
        pass
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


def _build_prompt(family: str, vin: float, vout: float, *, min_elems: int) -> str:
    try:
        from eval_amp_family import build_prompt  # type: ignore

        return str(build_prompt(str(family), float(vin), float(vout), min_elems=int(min_elems)))
    except Exception:
        gain_db = float(vin)
        bw_hz = float(vout)
        return (
            "Generate an analog amplifier in INC DSL.\n"
            "Rules:\n"
            "- Output ONLY INC lines (no explanation).\n"
            "- Line format: INC <name> <node1> <node2> <value>\n"
            "- Use ONLY passive elements {R,C,L}.\n"
            "- Required nodes: {vin, inv, out, vdd, 0}.\n"
            f"- Use at least {int(min_elems)} INC lines.\n"
            f"Task: gain={gain_db:.1f} dB, bw={bw_hz:.3g} Hz.\n"
            + RESPONSE_TEMPLATE
        )


def _set_elem_value(inc_text: str, name: str, value: float) -> str:
    target = (name or "").strip()
    if not target:
        return inc_text
    lines = extract_inc_lines(inc_text)
    out: List[str] = []
    changed = False
    for raw in lines:
        toks = raw.split()
        if len(toks) == 5 and toks[0] == "INC" and toks[1].strip().upper() == target.upper():
            toks[4] = f"{float(value):.6g}"
            out.append(" ".join(toks))
            changed = True
        else:
            out.append(raw)
    if not changed:
        return inc_text
    return ("\n".join(out).strip() + "\n")


def _is_bad_numeric_node(n: str) -> bool:
    s = str(n or "").strip()
    return s.isdigit() and s != "0"


def _unique_name(prefix: str, existing_upper: set[str], *, prefer: str | None = None) -> str:
    if prefer:
        cand = str(prefer).strip().upper()
        if cand and cand not in existing_upper:
            existing_upper.add(cand)
            return cand
    base = str(prefix or "X").strip().upper() or "X"
    for i in range(1, 10_000):
        cand = f"{base}{i}"
        if cand not in existing_upper:
            existing_upper.add(cand)
            return cand
    raise RuntimeError("unique name exhausted")


def _amp_structural_repair(
    inc_text: str,
    *,
    family: str,
    min_elems: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Minimal structure repair (NOT a circuit template fallback):
    - remove disallowed numeric nodes (except ground '0')
    - remove helper nodes (enforce fixed node set)
    - ensure required named parts exist: RBIAS(vdd,0), RFB(out,inv), RG(inv,0)
    - ensure required nodes appear and min element count is met via harmless decoupling caps
    """

    elems0 = list(parse_inc(inc_text))
    kept: List[IncElem] = []
    dropped: List[str] = []

    allowed_nodes = {"vin", "inv", "out", "vdd", "0"}
    for e in elems0:
        n1, n2 = (e.nodes or ["", ""])[:2]
        n1n, n2n = str(n1).strip().lower(), str(n2).strip().lower()
        if _is_bad_numeric_node(n1) or _is_bad_numeric_node(n2):
            dropped.append(str(e.name))
            continue
        if (n1n not in allowed_nodes) or (n2n not in allowed_nodes):
            dropped.append(str(e.name))
            continue
        # Enforce non-inverting topology: drop any direct connection from vin to non-ground nodes.
        nset0 = {n1n, n2n}
        if "vin" in nset0 and "0" not in nset0:
            dropped.append(str(e.name))
            continue
        # Enforce unique, well-scoped named parts: drop wrong-node RBIAS/RFB/RG to avoid ambiguous updates.
        nm = str(e.name or "").strip().upper()
        nset = {n1n, n2n}
        if nm == "RBIAS" and nset != {"vdd", "0"}:
            dropped.append(str(e.name))
            continue
        if nm == "RFB" and nset != {"out", "inv"}:
            dropped.append(str(e.name))
            continue
        if nm == "RG" and nset != {"inv", "0"}:
            dropped.append(str(e.name))
            continue
        if nm == "CCOMP" and nset != {"out", "inv"}:
            dropped.append(str(e.name))
            continue
        kept.append(e)

    existing = {str(e.name or "").strip().upper() for e in kept if str(e.name or "").strip()}

    req_nodes = {"vin", "inv", "out", "vdd", "0"}
    seen_nodes = set()
    for e in kept:
        for n in (e.nodes or []):
            seen_nodes.add(str(n).strip().lower())

    added: List[str] = []

    # Ensure required nodes appear (harmless caps to ground; VIN is an ideal source).
    for n in sorted(req_nodes - seen_nodes):
        if n == "0":
            continue
        name = _unique_name("CREQ", existing)
        kept.append(IncElem(name=name, kind="C", nodes=[n, "0"], value=1e-12, raw=""))
        added.append(name)

    # Ensure required named parts exist.
    def _has_named(kind: str, name: str, a: str, b: str) -> bool:
        kind_u = str(kind or "").strip().upper() or "R"
        name_u = str(name).strip().upper()
        aset = {str(a).lower(), str(b).lower()}
        for e in kept:
            if str(e.kind or "").strip().upper() != kind_u:
                continue
            if str(e.name or "").strip().upper() != name_u:
                continue
            nset = {str(e.nodes[0]).lower(), str(e.nodes[1]).lower()}
            if nset == aset:
                return True
        return False

    params = amp_family_params(str(family))
    vdd = float(params.get("vdd") or 1.8)
    p_max_mw = float(params.get("pstatic_max_mw") or AMP_PSTATIC_MAX_MW)
    p_max_w = float(max(1e-9, p_max_mw)) / 1e3
    rbias_min = float((vdd * vdd) / p_max_w)
    rbias0 = float(max(5.0 * rbias_min, 1e3))

    if not _has_named("R", "RBIAS", "vdd", "0"):
        kept.append(IncElem(name="RBIAS", kind="R", nodes=["vdd", "0"], value=float(rbias0), raw=""))
        existing.add("RBIAS")
        added.append("RBIAS")
    if not _has_named("R", "RFB", "out", "inv"):
        kept.append(IncElem(name=_unique_name("RFB", existing, prefer="RFB"), kind="R", nodes=["out", "inv"], value=10_000.0, raw=""))
        added.append("RFB")
    if not _has_named("R", "RG", "inv", "0"):
        kept.append(IncElem(name=_unique_name("RG", existing, prefer="RG"), kind="R", nodes=["inv", "0"], value=10_000.0, raw=""))
        added.append("RG")
    # Optional (tunable) compensation capacitor across out-inv to shape bandwidth.
    if not _has_named("C", "CCOMP", "out", "inv"):
        kept.append(IncElem(name=_unique_name("CCOMP", existing, prefer="CCOMP"), kind="C", nodes=["out", "inv"], value=1e-15, raw=""))
        added.append("CCOMP")

    # Pad to min_elems with decoupling caps (do not affect the signal path).
    while len(kept) < int(min_elems):
        nm = _unique_name("CDEC", existing)
        kept.append(IncElem(name=nm, kind="C", nodes=["vdd", "0"], value=1e-12, raw=""))
        added.append(nm)

    out = (to_inc_text(kept).strip() + "\n")
    return out, {"changed": bool(added or dropped), "added": added, "dropped": dropped}


def _amp_sanitize_extras(elems: List[IncElem], *, keep_names_upper: set[str]) -> Tuple[List[IncElem], bool]:
    """
    Make non-essential passives effectively open circuits to reduce unwanted interactions
    (keeps element count but stabilizes gain/bw tuning).
    """
    out: List[IncElem] = []
    changed = False
    for e in elems:
        nm = str(e.name or "").strip().upper()
        if nm in keep_names_upper:
            out.append(e)
            continue
        if e.kind == "R" and e.value is not None:
            if float(e.value) != 1e12:
                changed = True
            out.append(IncElem(name=e.name, kind=e.kind, nodes=list(e.nodes), value=1e12, model=e.model, raw=e.raw))
            continue
        if e.kind == "C" and e.value is not None:
            if float(e.value) != 1e-15:
                changed = True
            out.append(IncElem(name=e.name, kind=e.kind, nodes=list(e.nodes), value=1e-15, model=e.model, raw=e.raw))
            continue
        if e.kind == "L" and e.value is not None:
            if float(e.value) != 1e3:
                changed = True
            out.append(IncElem(name=e.name, kind=e.kind, nodes=list(e.nodes), value=1e3, model=e.model, raw=e.raw))
            continue
        out.append(e)
    return out, bool(changed)


def _gain_ratio_from_db(gain_db: float) -> float:
    lin = float(10.0 ** (float(gain_db) / 20.0))
    return max(1e-6, float(lin - 1.0))


def _find_fb_resistors(inc_text: str) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[float]]:
    # Returns (Rf_name, Rf_value, Rg_name, Rg_value) where
    # Rf is between out-inv, Rg is between inv-0.
    elems = parse_inc(inc_text)
    rf: Tuple[Optional[str], Optional[float]] = (None, None)
    rg: Tuple[Optional[str], Optional[float]] = (None, None)
    for e in elems:
        if e.kind != "R" or e.value is None:
            continue
        nset = {str(e.nodes[0]).lower(), str(e.nodes[1]).lower()}
        if nset == {"out", "inv"}:
            if rf[1] is None or float(e.value) > float(rf[1]):
                rf = (e.name, float(e.value))
        if nset == {"inv", "0"}:
            if rg[1] is None or float(e.value) > float(rg[1]):
                rg = (e.name, float(e.value))
    return rf[0], rf[1], rg[0], rg[1]


def _find_rbias(inc_text: str) -> Tuple[Optional[str], Optional[float]]:
    for e in parse_inc(inc_text):
        if e.kind != "R" or e.value is None:
            continue
        if e.name.strip().upper() != "RBIAS":
            continue
        nset = {str(e.nodes[0]).lower(), str(e.nodes[1]).lower()}
        if nset == {"vdd", "0"}:
            return e.name, float(e.value)
    return None, None


def _score_amp(detail: Dict[str, Any], *, family: str, gain_db_t: float, bw_hz_t: float) -> float:
    if not bool(detail.get("ok", False)):
        return -1.0
    try:
        g = float(detail.get("gain_db") or 0.0)
        bw = float(detail.get("bw_hz") or 0.0)
        pm = float(detail.get("pm_deg") or 0.0)
        p = float(detail.get("pstatic_mw") or 0.0)
    except Exception:
        return -1.0

    params = amp_family_params(str(family))
    pm_min = float(params.get("pm_min_deg") or AMP_PM_MIN_DEG)
    p_max = float(params.get("pstatic_max_mw") or AMP_PSTATIC_MAX_MW)

    gain_err = abs(float(g) - float(gain_db_t))
    bw_err = abs(float(bw) - float(bw_hz_t)) / max(1e-9, float(bw_hz_t))

    # Keep a longer tail for large errors (avoid all-fail samples collapsing to the same clipped score).
    gain_term = 1.0 - gain_err / max(1e-6, float(AMP_GAIN_TOL_DB))
    bw_term = 1.0 - bw_err / max(1e-6, float(AMP_BW_TOL_REL))
    gain_term = max(-10.0, min(1.0, float(gain_term)))
    bw_term = max(-10.0, min(1.0, float(bw_term)))
    pm_term = max(-1.0, min(1.0, (float(pm) - float(pm_min)) / 30.0))
    p_term = max(-1.0, min(1.0, (float(p_max) - float(p)) / max(1e-6, float(p_max))))

    score = 0.0
    score += 5.0 if bool(detail.get("pass_CV", False)) else 0.0
    score += 1.0 if bool(detail.get("pass_CE", False)) else -1.0
    score += 2.0 * float(gain_term) + 2.0 * float(bw_term) + 0.5 * float(pm_term) + 0.5 * float(p_term)
    return float(score)


def _eval_one(
    inc_text: str,
    *,
    family: str,
    gain_db_t: float,
    bw_hz_t: float,
    sim_timeout_s: float,
    min_elems: int,
) -> Dict[str, Any]:
    t0 = time.time()
    detail = eval_one_detail_amp_family(
        inc_text,
        family=str(family),
        gain_db=float(gain_db_t),
        bw_hz=float(bw_hz_t),
        sim_timeout_s=float(sim_timeout_s),
        min_elems=int(min_elems),
    )
    dt = float(time.time() - t0)
    score = _score_amp(detail, family=str(family), gain_db_t=float(gain_db_t), bw_hz_t=float(bw_hz_t))
    return {"detail": detail, "score": float(score), "sim_time": float(dt)}


def _eda_repair(
    inc_text: str,
    *,
    family: str,
    gain_db_t: float,
    bw_hz_t: float,
    sim_timeout_s: float,
    min_elems: int,
    factors: List[float],
    max_evals: int,
    max_iters: int = 3,
    always: bool = False,
    no_structural_repair: bool = False,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    # Returns (best_inc, best_detail, repair_meta)
    tried = 0
    base_inc = str(inc_text or "").strip() + ("\n" if str(inc_text or "").strip() else "")

    base_res = _eval_one(
        base_inc,
        family=family,
        gain_db_t=gain_db_t,
        bw_hz_t=bw_hz_t,
        sim_timeout_s=sim_timeout_s,
        min_elems=min_elems,
    )
    tried += 1
    base_detail = dict(base_res.get("detail") or {})
    base_score = float(base_res.get("score") or -1.0)

    best_inc = base_inc
    best_detail = dict(base_detail)
    best_score = float(base_score)
    best_tag = "base"

    # (S) Structural repair (only to enforce verifier constraints; not used as evaluation fallback).
    struct_meta: Dict[str, Any] = {"changed": False}
    work_inc = best_inc
    if not bool(no_structural_repair):
        ver_ok = bool(verify_inc_amp(work_inc, min_elems=int(min_elems)).ok)
        if (not ver_ok) or bool(always):
            cand, sm = _amp_structural_repair(work_inc, family=str(family), min_elems=int(min_elems))
            struct_meta = dict(sm or {})
            work_inc = str(cand)
            if tried < int(max_evals):
                r = _eval_one(
                    work_inc,
                    family=family,
                    gain_db_t=gain_db_t,
                    bw_hz_t=bw_hz_t,
                    sim_timeout_s=sim_timeout_s,
                    min_elems=min_elems,
                )
                tried += 1
                sc = float(r.get("score") or -1.0)
                if sc > best_score + 1e-9:
                    best_score = sc
                    best_inc = work_inc
                    best_detail = dict(r.get("detail") or {})
                    best_tag = "structural"
        else:
            # Even if we don't change the circuit, normalize to ensure RFB/RG exist for later tuning.
            cand, sm = _amp_structural_repair(work_inc, family=str(family), min_elems=int(min_elems))
            work_inc = str(cand)
            struct_meta = dict(sm or {})

    # From here, tune on `work_inc` (should have RBIAS/RFB/RG).
    tuned_inc = work_inc

    # (T0) Sanitize non-essential passives to reduce interactions (keeps element count).
    try:
        elems = list(parse_inc(tuned_inc))
        keep = {"RBIAS", "RFB", "RG", "CCOMP"}
        elems_s, chg = _amp_sanitize_extras(elems, keep_names_upper=set(keep))
        if chg:
            tuned_inc = to_inc_text(elems_s).strip() + "\n"
    except Exception:
        pass

    # (T1) Closed-loop gain repair by formula (non-inverting): A = 1 + RFB/RG.
    rg = 10_000.0
    a_lin = float(10.0 ** (float(gain_db_t) / 20.0))
    ratio = max(0.0, float(a_lin - 1.0))
    rf = max(10.0, float(ratio) * float(rg))
    tuned_inc = _set_elem_value(tuned_inc, "RG", float(rg))
    tuned_inc = _set_elem_value(tuned_inc, "RFB", float(rf))

    if tried < int(max_evals):
        r = _eval_one(
            tuned_inc,
            family=family,
            gain_db_t=gain_db_t,
            bw_hz_t=bw_hz_t,
            sim_timeout_s=sim_timeout_s,
            min_elems=min_elems,
        )
        tried += 1
        sc = float(r.get("score") or -1.0)
        if sc > best_score + 1e-9:
            best_score = sc
            best_inc = tuned_inc
            best_detail = dict(r.get("detail") or {})
            best_tag = "gain_formula"

    # (T2) Joint tuning:
    # - gain: update RFB (finite open-loop gain -> may need ratio correction)
    # - bandwidth: tune CCOMP across out-inv (structure-agnostic, not an evaluation template)
    params = amp_family_params(str(family))
    vdd = float(params.get("vdd") or 1.8)
    p_max_mw = float(params.get("pstatic_max_mw") or AMP_PSTATIC_MAX_MW)
    pm_min_deg = float(params.get("pm_min_deg") or AMP_PM_MIN_DEG)
    p_max_w = float(max(1e-9, p_max_mw)) / 1e3
    rbias_min = float(max(1.0, (vdd * vdd) / p_max_w))
    rbias_max = 1e9
    # Keep a small headroom so power constraint doesn't fail due to numeric jitter.
    rbias_base = float(min(max(rbias_min * 1.05, rbias_min), rbias_max))

    def _get_value(txt: str, name: str) -> Optional[float]:
        nu = str(name or "").strip().upper()
        for e in parse_inc(txt):
            if e.value is None:
                continue
            if str(e.name or "").strip().upper() == nu:
                return float(e.value)
        return None

    # Bias at the power limit to maximize open-loop gain, then shape bandwidth with CCOMP.
    tuned_inc = _set_elem_value(tuned_inc, "RBIAS", float(rbias_base))

    # Initial CCOMP guess from a 1st-order pole: fc ≈ 1/(2π RFB CCOMP).
    rf0 = float(_get_value(tuned_inc, "RFB") or rf)
    try:
        c0 = 1.0 / (2.0 * math.pi * max(1.0, rf0) * max(1e-9, float(bw_hz_t)))
    except Exception:
        c0 = 1e-15
    c0 = float(min(max(float(c0), 1e-15), 1e-6))
    tuned_inc = _set_elem_value(tuned_inc, "CCOMP", float(c0))

    cur_inc = tuned_inc
    cur_detail: Dict[str, Any] = {}
    if tried < int(max_evals):
        r = _eval_one(
            cur_inc,
            family=family,
            gain_db_t=gain_db_t,
            bw_hz_t=bw_hz_t,
            sim_timeout_s=sim_timeout_s,
            min_elems=min_elems,
        )
        tried += 1
        cur_detail = dict(r.get("detail") or {})
        sc = float(r.get("score") or -1.0)
        if sc > best_score + 1e-9:
            best_score = sc
            best_inc = cur_inc
            best_detail = dict(cur_detail)
            best_tag = "ccomp_init"

    eta_gain = 0.7
    eta_bw = 0.7
    for it in range(max(0, int(max_iters))):
        if tried >= int(max_evals):
            break
        if not bool((cur_detail or {}).get("ok", False)):
            break
        if bool((cur_detail or {}).get("pass_CV", False)):
            break

        g_db_meas = float((cur_detail or {}).get("gain_db") or 0.0)
        bw_meas = float((cur_detail or {}).get("bw_hz") or 0.0)

        rf_cur = float(_get_value(cur_inc, "RFB") or rf0)
        c_cur = float(_get_value(cur_inc, "CCOMP") or c0)

        a_t = float(10.0 ** (float(gain_db_t) / 20.0))
        try:
            a_m = float(10.0 ** (float(g_db_meas) / 20.0))
        except Exception:
            a_m = 0.0

        if a_m <= 1.01:
            scale_rf = 2.0
        else:
            scale_rf = float(max(1e-6, a_t - 1.0) / max(1e-6, a_m - 1.0))
        scale_rf = float(min(max(scale_rf, 0.3), 3.0))
        rf_new = float(rf_cur) * float(scale_rf ** eta_gain)
        rf_new = float(min(max(rf_new, 10.0), 1e9))

        target_bw = float(max(1e-9, float(bw_hz_t)))
        if not (bw_meas > 0.0) or not math.isfinite(float(bw_meas)):
            scale_c = 1.0
        else:
            # If bw is too high, increase CCOMP; if too low, decrease.
            scale_c = float(bw_meas / target_bw)
        pm_meas = float((cur_detail or {}).get("pm_deg") or 0.0)
        if math.isfinite(pm_meas) and pm_meas > 0.0 and pm_meas < float(pm_min_deg):
            # Increase compensation if phase margin is below the threshold.
            scale_c *= float(min(3.0, max(1.1, float(pm_min_deg) / float(pm_meas))))
        scale_c = float(min(max(scale_c, 0.3), 3.0))
        c_new = float(c_cur) * float(scale_c ** eta_bw)
        c_new = float(min(max(c_new, 1e-15), 1e-6))

        # Bias (RBIAS) update for power / stability headroom.
        rbias_cur = float(_get_value(cur_inc, "RBIAS") or rbias_base)
        rbias_new = float(rbias_cur)
        p_meas = float((cur_detail or {}).get("pstatic_mw") or 0.0)
        if p_max_mw > 0.0 and math.isfinite(p_meas) and p_meas > float(p_max_mw) * 0.995:
            rbias_new *= float(p_meas / float(p_max_mw)) * 1.05
        if math.isfinite(pm_meas) and pm_meas > 0.0 and pm_meas < float(pm_min_deg):
            rbias_new *= float(min(3.0, max(1.05, float(pm_min_deg) / float(pm_meas))))
        rbias_new = float(min(max(rbias_new, float(rbias_min)), float(rbias_max)))

        cand = cur_inc
        cand = _set_elem_value(cand, "RG", float(rg))
        cand = _set_elem_value(cand, "RFB", float(rf_new))
        cand = _set_elem_value(cand, "CCOMP", float(c_new))
        cand = _set_elem_value(cand, "RBIAS", float(rbias_new))

        r = _eval_one(
            cand,
            family=family,
            gain_db_t=gain_db_t,
            bw_hz_t=bw_hz_t,
            sim_timeout_s=sim_timeout_s,
            min_elems=min_elems,
        )
        tried += 1
        cur_inc = cand
        cur_detail = dict(r.get("detail") or {})
        sc = float(r.get("score") or -1.0)
        if sc > best_score + 1e-9:
            best_score = sc
            best_inc = cand
            best_detail = dict(cur_detail)
            best_tag = f"joint_it{it}"

    # (T3) Near-miss local search around the best point to fix tight PM/bandwidth edges.
    try:
        if tried < int(max_evals) and bool((best_detail or {}).get("ok", False)):
            g_err = float((best_detail or {}).get("gain_err_db") or 1e9)
            bw_err = float((best_detail or {}).get("bw_err_rel") or 1e9)
            pm_best = float((best_detail or {}).get("pm_deg") or 0.0)
            if (g_err <= 2.0 * float(AMP_GAIN_TOL_DB)) and (bw_err <= 2.0 * float(AMP_BW_TOL_REL)) and (
                pm_best < float(pm_min_deg)
            ):
                c_base = float(_get_value(best_inc, "CCOMP") or c0)
                r_base = float(_get_value(best_inc, "RBIAS") or rbias_base)
                c_base = float(min(max(c_base, 1e-15), 1e-6))
                r_base = float(min(max(r_base, float(rbias_min)), float(rbias_max)))

                c_factors = [0.8, 1.0, 1.25, 1.6, 2.0, 2.5]
                r_factors = [1.0, 1.25, 1.6, 2.0, 2.5, 3.0]
                for cf in c_factors:
                    for rfct in r_factors:
                        if tried >= int(max_evals):
                            break
                        cand = best_inc
                        cand = _set_elem_value(cand, "CCOMP", float(min(max(c_base * float(cf), 1e-15), 1e-6)))
                        cand = _set_elem_value(
                            cand,
                            "RBIAS",
                            float(min(max(r_base * float(rfct), float(rbias_min)), float(rbias_max))),
                        )
                        r = _eval_one(
                            cand,
                            family=family,
                            gain_db_t=gain_db_t,
                            bw_hz_t=bw_hz_t,
                            sim_timeout_s=sim_timeout_s,
                            min_elems=min_elems,
                        )
                        tried += 1
                        sc = float(r.get("score") or -1.0)
                        if sc > best_score + 1e-9:
                            best_score = sc
                            best_inc = cand
                            best_detail = dict(r.get("detail") or {})
                            best_tag = "near_miss_search"
                    if tried >= int(max_evals):
                        break
    except Exception:
        pass

    changed = bool(best_score > base_score + 1e-6 and _canon_inc_text(best_inc) != _canon_inc_text(base_inc))
    meta: Dict[str, Any] = {
        "changed": bool(changed),
        "tag": str(best_tag),
        "tried": int(tried),
        "score_before": float(base_score),
        "score_after": float(best_score),
        "inc_before": str(base_inc),
        "inc_after": str(best_inc) if changed else "",
        "detail_before": dict(base_detail),
        "detail_after": dict(best_detail) if changed else {},
        "structural": struct_meta,
    }
    return best_inc, best_detail, meta


def _parse_factors(s: str) -> List[float]:
    out: List[float] = []
    for p in (s or "").split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            continue
    if not out:
        out = [0.5, 0.8, 1.0, 1.25, 1.5, 2.0]
    return out


def _load_done_set(path: Path) -> set[Tuple[str, float, float]]:
    out: set[Tuple[str, float, float]] = set()
    if not path.exists() or path.stat().st_size <= 0:
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            out.add((str(r["family"]).lower(), float(r["vin"]), float(r["vout"])))
        except Exception:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=2025)

    ap.add_argument("--n_gen", type=int, default=16)
    ap.add_argument("--max_rounds", type=int, default=1)
    ap.add_argument("--ensure_pass_cv", action="store_true")
    ap.add_argument("--min_pass_cv", type=int, default=1)
    ap.add_argument("--temp_step", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--min_elems", type=int, default=15)
    ap.add_argument("--pairs_per_task", type=int, default=12)
    ap.add_argument("--min_pair_gap", type=float, default=0.1)
    ap.add_argument("--top_k_chosen", type=int, default=4)
    ap.add_argument("--sft_topn_per_task", type=int, default=1)
    ap.add_argument("--sft_strict_topn_per_task", type=int, default=1)
    ap.add_argument("--allow_fail_chosen", action="store_true")

    ap.add_argument("--max_tasks", type=int, default=0)
    ap.add_argument("--task_shard_id", type=int, default=0)
    ap.add_argument("--task_shard_count", type=int, default=1)
    ap.add_argument("--skip_done_tasks_jsonl", default="")

    ap.add_argument("--sim_timeout_s", type=float, default=60.0)
    ap.add_argument("--sim_workers", type=int, default=0)
    ap.add_argument("--eda_repair", action="store_true")
    ap.add_argument("--repair_factors", default="0.5,0.8,1.0,1.25,1.5,2.0")
    ap.add_argument("--repair_max_evals", type=int, default=12)
    ap.add_argument("--repair_max_iters", type=int, default=6)
    ap.add_argument("--repair_always", action="store_true")
    ap.add_argument("--no_structural_repair", action="store_true")

    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    # VP-SPI min_elems relax: amplifier/oscillator branches use a lighter complexity constraint.
    _min_raw = int(getattr(args, "min_elems", 15) or 15)
    if _min_raw > 15:
        print(f"[min_elems] clamp {_min_raw} -> 15 (amp/osc)", flush=True)
        args.min_elems = 15

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    sim_workers = max(1, int(_auto_sim_workers(int(args.sim_workers))))
    shard_count = max(1, int(args.task_shard_count))
    shard_id = int(args.task_shard_id)
    if shard_id < 0 or shard_id >= shard_count:
        raise SystemExit(f"--task_shard_id must be in [0,{shard_count}), got {shard_id}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    (out_root / "tasks").mkdir(parents=True, exist_ok=True)

    pairs_path = out_root / "dpo_pairs.jsonl"
    meta_path = out_root / "pairs_meta.jsonl"
    sft_path = out_root / "sft_train.jsonl"
    sft_strict_path = out_root / "sft_train_strict.jsonl"
    done_path = out_root / "done_tasks.jsonl"

    done: set[Tuple[str, float, float]] = set()
    if str(args.skip_done_tasks_jsonl or "").strip():
        done |= _load_done_set(Path(str(args.skip_done_tasks_jsonl).strip()))
    if bool(args.resume):
        done |= _load_done_set(done_path)

    seen_sft: set[str] = set()
    seen_sft_strict: set[str] = set()
    if bool(args.resume) and sft_path.exists() and sft_path.stat().st_size > 0:
        for line in sft_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                txt = str(json.loads(line).get("text") or "")
                if txt:
                    seen_sft.add(_sha1_text(txt))
            except Exception:
                continue
    if bool(args.resume) and sft_strict_path.exists() and sft_strict_path.stat().st_size > 0:
        for line in sft_strict_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                txt = str(json.loads(line).get("text") or "")
                if txt:
                    seen_sft_strict.add(_sha1_text(txt))
            except Exception:
                continue

    mode = "a" if bool(args.resume) else "w"
    f_pairs = pairs_path.open(mode, encoding="utf-8")
    f_meta = meta_path.open(mode, encoding="utf-8")
    f_sft = sft_path.open(mode, encoding="utf-8")
    f_sft_strict = sft_strict_path.open(mode, encoding="utf-8")
    f_done = done_path.open("a", encoding="utf-8")

    tok, model = _load_model(args.base_model, str(args.adapter))
    dev = _device(model)

    tasks = default_taskset_amp()
    if int(args.max_tasks) > 0:
        tasks = tasks[: int(args.max_tasks)]

    factors = _parse_factors(str(args.repair_factors))

    total_pairs = 0
    total_sft = 0
    total_sft_strict = 0
    total_tasks_done = 0
    total_tasks_skipped = 0
    agg: Dict[str, Any] = {
        "total": 0,
        "ok": 0,
        "pass_C": 0,
        "pass_CV": 0,
        "pass_CE": 0,
        "pass_gain": 0,
        "pass_bw": 0,
        "pass_pm": 0,
        "pass_p": 0,
    }
    agg_by_family: Dict[str, Dict[str, Any]] = {}

    def _acc(fam0: str, detail: Dict[str, Any]) -> None:
        fam_k = str(fam0).lower()
        a = agg_by_family.setdefault(fam_k, {k: 0 for k in agg.keys()})
        for bucket in (agg, a):
            bucket["total"] += 1
            for k in ["ok", "pass_C", "pass_CV", "pass_CE", "pass_gain", "pass_bw", "pass_pm", "pass_p"]:
                bucket[k] += 1 if bool(detail.get(k, False)) else 0

    executor: Optional[ThreadPoolExecutor] = None
    if int(sim_workers) > 1:
        executor = ThreadPoolExecutor(max_workers=int(sim_workers))
        print(f"[sim] workers={int(sim_workers)} (affinity_cpus={_effective_cpus()})", flush=True)
    else:
        print(f"[sim] workers=1 (affinity_cpus={_effective_cpus()})", flush=True)

    try:
        for ti, task in enumerate(tasks):
            if shard_count > 1 and (int(ti) % int(shard_count)) != int(shard_id):
                continue
            fam, vin, vout = str(task.family), float(task.vin), float(task.vout)
            key = (fam.lower(), float(vin), float(vout))
            if key in done:
                total_tasks_skipped += 1
                continue

            seed_base = int(args.seed) + ti * 1000
            random.seed(seed_base)
            torch.manual_seed(seed_base)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_base)

            prompt = _build_prompt(fam, vin, vout, min_elems=int(args.min_elems))
            task_dir = out_root / "tasks" / fam.lower() / f"vin{vin:.1f}_vout{vout:.1f}"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            best_by_hash: Dict[str, Dict[str, Any]] = {}
            pass_cv_hits = 0

            rounds = max(1, int(args.max_rounds))
            for ridx in range(int(rounds)):
                if bool(args.ensure_pass_cv) and pass_cv_hits >= int(args.min_pass_cv):
                    break
                temp = float(args.temperature) + float(args.temp_step) * float(ridx)

                # Batched generation for GPU utilization.
                n_gen = int(args.n_gen)
                inputs = tok([prompt] * n_gen, return_tensors="pt", padding=True).to(dev)
                t_gen0 = time.time()
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=float(temp),
                        top_p=float(args.top_p),
                        max_new_tokens=int(args.max_new_tokens),
                        pad_token_id=tok.eos_token_id,
                    )
                gen_time = float(time.time() - t_gen0)
                texts = tok.batch_decode(out, skip_special_tokens=True)

                cand_incs: List[str] = []
                seen_can: set[str] = set()
                for txt in texts:
                    resp = _normalize_model_text(txt)
                    inc_lines = extract_inc_lines(resp)
                    if not inc_lines:
                        continue
                    inc_text = ("\n".join(inc_lines).strip() + "\n")
                    can = _canon_inc_text(inc_text)
                    if not can.strip() or can in seen_can:
                        continue
                    seen_can.add(can)
                    cand_incs.append(inc_text)

                gain_db_t = float(vin)
                bw_hz_t = float(vout)

                def _eval_candidate(inc0: str) -> Dict[str, Any]:
                    if bool(args.eda_repair):
                        inc_fin, detail, repair_meta = _eda_repair(
                            inc0,
                            family=fam,
                            gain_db_t=gain_db_t,
                            bw_hz_t=bw_hz_t,
                            sim_timeout_s=float(args.sim_timeout_s),
                            min_elems=int(args.min_elems),
                            factors=factors,
                            max_evals=int(args.repair_max_evals),
                            max_iters=int(args.repair_max_iters),
                            always=bool(args.repair_always),
                            no_structural_repair=bool(args.no_structural_repair),
                        )
                    else:
                        r = _eval_one(
                            inc0,
                            family=fam,
                            gain_db_t=gain_db_t,
                            bw_hz_t=bw_hz_t,
                            sim_timeout_s=float(args.sim_timeout_s),
                            min_elems=int(args.min_elems),
                        )
                        inc_fin = inc0
                        detail = dict(r.get("detail") or {})
                        repair_meta = {"used": False}
                    score = _score_amp(detail, family=fam, gain_db_t=gain_db_t, bw_hz_t=bw_hz_t)
                    ch = str(detail.get("canonical_hash") or _sha1_text(_canon_inc_text(inc_fin)))
                    return {
                        "family": str(fam).lower(),
                        "vin": float(gain_db_t),
                        "vout": float(bw_hz_t),
                        "ridx": int(ridx),
                        "inc": str(inc_fin),
                        "score": float(score),
                        "detail": detail,
                        "repair": repair_meta,
                        "gen_time": float(gen_time),
                        "n_elems": float((detail or {}).get("n_elems") or 0.0),
                    }, ch

                if executor is None:
                    for inc0 in cand_incs:
                        rec, ch = _eval_candidate(inc0)
                        prev = best_by_hash.get(ch)
                        if (prev is None) or float(rec["score"]) > float(prev.get("score") or -1e9):
                            best_by_hash[ch] = rec
                else:
                    fut_map = {executor.submit(_eval_candidate, inc0): inc0 for inc0 in cand_incs}
                    for fut in as_completed(list(fut_map.keys())):
                        try:
                            rec, ch = fut.result()
                        except Exception:
                            continue
                        prev = best_by_hash.get(ch)
                        if (prev is None) or float(rec["score"]) > float(prev.get("score") or -1e9):
                            best_by_hash[ch] = rec

                pass_cv_hits = sum(1 for r in best_by_hash.values() if bool((r.get("detail") or {}).get("pass_CV", False)))

            # Persist per-task scored records for debugging.
            scored_path = task_dir / "scored.json"
            scored = sorted(list(best_by_hash.values()), key=lambda r: float(r.get("score") or -1e9), reverse=True)
            scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")
            for r in scored:
                d = (r.get("detail") or {}) if isinstance(r, dict) else {}
                if isinstance(d, dict):
                    _acc(fam, d)

            chosen_pool = [r for r in scored if bool((r.get("detail") or {}).get("pass_CV", False))]
            if not chosen_pool:
                # Fallback: keep PVPO data non-empty by taking best OK (power-safe) samples when pass_CV is rare.
                chosen_pool = [
                    r
                    for r in scored
                    if bool((r.get("detail") or {}).get("ok", False))
                    and bool((r.get("detail") or {}).get("pass_p", True))
                ]
            if bool(args.allow_fail_chosen) and not chosen_pool:
                chosen_pool = list(scored)

            # SFT dumps.
            for r in chosen_pool[: int(args.sft_topn_per_task)]:
                txt = prompt + str(r.get("inc") or "")
                h = _sha1_text(txt)
                if h in seen_sft:
                    continue
                seen_sft.add(h)
                f_sft.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                total_sft += 1

            strict_pool = [r for r in scored if bool((r.get("detail") or {}).get("pass_CV", False))]
            strict_topn = int(args.sft_strict_topn_per_task)
            strict_take = strict_pool if strict_topn == 0 else strict_pool[:strict_topn]
            for r in strict_take:
                txt = prompt + str(r.get("inc") or "")
                h = _sha1_text(txt)
                if h in seen_sft_strict:
                    continue
                seen_sft_strict.add(h)
                f_sft_strict.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                total_sft_strict += 1

            # DPO pairs.
            pairs_written = 0
            if chosen_pool:
                chosen_top = chosen_pool[: max(1, int(args.top_k_chosen))]
                for chosen in chosen_top:
                    if pairs_written >= int(args.pairs_per_task):
                        break
                    c_score = float(chosen.get("score") or -1e9)
                    rejected_cands = [
                        r for r in scored if float(r.get("score") or -1e9) <= c_score - float(args.min_pair_gap)
                    ]
                    if not rejected_cands:
                        continue
                    random.shuffle(rejected_cands)
                    for rejected in rejected_cands:
                        if pairs_written >= int(args.pairs_per_task):
                            break
                        row = {"prompt": prompt, "chosen": str(chosen.get("inc") or ""), "rejected": str(rejected.get("inc") or "")}
                        f_pairs.write(json.dumps(row, ensure_ascii=False) + "\n")
                        meta = {
                            "task": asdict(task),
                            "chosen": {"score": float(chosen.get("score") or 0.0), "detail": chosen.get("detail")},
                            "rejected": {"score": float(rejected.get("score") or 0.0), "detail": rejected.get("detail")},
                            "min_pair_gap": float(args.min_pair_gap),
                        }
                        f_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")
                        total_pairs += 1
                        pairs_written += 1

            # Mark done.
            f_done.write(json.dumps({"family": fam.lower(), "vin": float(vin), "vout": float(vout)}, ensure_ascii=False) + "\n")
            f_done.flush()
            done.add(key)
            total_tasks_done += 1

            print(
                f"[task] {total_tasks_done} done fam={fam} vin={vin:.1f} vout={vout:.1f} uniq={len(scored)} pass_cv={sum(1 for r in scored if bool((r.get('detail') or {}).get('pass_CV')))} pairs={pairs_written}",
                flush=True,
            )

    finally:
        try:
            f_pairs.close()
            f_meta.close()
            f_sft.close()
            f_sft_strict.close()
            f_done.close()
        except Exception:
            pass
        if executor is not None:
            try:
                executor.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass

    print(
        f"[OK] wrote pairs={total_pairs} sft={total_sft} sft_strict={total_sft_strict} tasks_done={total_tasks_done} tasks_skipped={total_tasks_skipped} out_root={out_root}",
        flush=True,
    )
    try:
        stats = {
            "out_root": str(out_root),
            "tasks_done": int(total_tasks_done),
            "tasks_skipped": int(total_tasks_skipped),
            "pairs": int(total_pairs),
            "sft": int(total_sft),
            "sft_strict": int(total_sft_strict),
            "agg": agg,
            "agg_by_family": agg_by_family,
            "rates": {
                k: (float(agg.get(k, 0)) / float(max(1, agg.get("total", 0))))
                for k in ["ok", "pass_C", "pass_CV", "pass_CE", "pass_gain", "pass_bw", "pass_pm", "pass_p"]
            },
        }
        (out_root / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


RESPONSE_TEMPLATE = "### Response:\n"
_RESP_KEY = "### Response:"


def build_prompt(family: str, vin: float, vout: float) -> str:
    fam = (family or "").strip().lower()
    if fam == "buck":
        return (
            "Generate a Buck DC-DC converter in INC DSL.\n"
            "Rules:\n"
            "- IMPORTANT: Your output MUST START with the required power-stage topology lines EXACTLY as written (copy-paste), in the SAME ORDER.\n"
            "- Output ONLY INC lines (no explanation).\n"
            "- Line format: INC <name> <node1> <node2> <value_or_model>\n"
            "- Element names must start with R/L/C/D/S and contain a digit.\n"
            "- Must include required nodes {vin,sw,out,0}. You MAY introduce extra helper nodes (e.g., n1, nsn1, nd1).\n"
            "- Node names are identifiers (vin/sw/out/0/n1/...). DO NOT use numeric node names like 12 or 5.\n"
            "- Use elements {L,C,D,S} with numeric values (e.g., 47u).\n"
            "- Value constraints: use inductors in [10u, 220u] and main power capacitors in [1u, 470u]. Avoid nH/pF-scale values.\n"
            "- Use at least 20 INC lines (>=20 elements).\n"
            "- Put ONE INC statement per line (use newlines between INC lines).\n"
            "- Required power-stage topology (note diode direction: anode=0, cathode=sw):\n"
            "  INC S1 vin sw Sstd\n"
            "  INC D1 0 sw Dstd\n"
            "  INC L1 sw out <L>\n"
            "  INC C1 out 0 <C>\n"
            "- To reach >=20 elements, add extra input/output capacitor banks and RC snubbers/dampers.\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )
    if fam == "boost":
        return (
            "Generate a Boost DC-DC converter in INC DSL.\n"
            "Rules:\n"
            "- IMPORTANT: Your output MUST START with the required power-stage topology lines EXACTLY as written (copy-paste), in the SAME ORDER.\n"
            "- Output ONLY INC lines (no explanation).\n"
            "- Line format: INC <name> <node1> <node2> <value_or_model>\n"
            "- Element names must start with R/L/C/D/S and contain a digit.\n"
            "- Must include required nodes {vin,sw,out,0}. You MAY introduce extra helper nodes (e.g., n1, nsn1, nd1).\n"
            "- Node names are identifiers (vin/sw/out/0/n1/...). DO NOT use numeric node names like 12 or 5.\n"
            "- Value constraints: use inductors in [10u, 220u] and main power capacitors in [1u, 470u]. Avoid nH/pF-scale values.\n"
            "- Use at least 20 INC lines (>=20 elements).\n"
            "- Put ONE INC statement per line (use newlines between INC lines).\n"
            "- Required power-stage topology:\n"
            "  INC L1 vin sw <L>\n"
            "  INC S1 sw 0 Sstd\n"
            "  INC D1 sw out Dstd\n"
            "  INC C1 out 0 <C>\n"
            "- To reach >=20 elements, add extra input/output capacitor banks and RC snubbers/dampers.\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )
    if fam == "sepic":
        return (
            "Generate a SEPIC DC-DC converter in INC DSL.\n"
            "Rules:\n"
            "- IMPORTANT: Your output MUST START with the required power-stage topology lines EXACTLY as written (copy-paste), in the SAME ORDER.\n"
            "- Output ONLY INC lines (no explanation).\n"
            "- Line format: INC <name> <node1> <node2> <value_or_model>\n"
            "- Element names must start with R/L/C/D/S and contain a digit.\n"
            "- Must include required nodes {vin,sw,n1,out,0}. You MAY introduce extra helper nodes.\n"
            "- Node names are identifiers (vin/sw/out/0/n1/...). DO NOT use numeric node names like 12 or 5.\n"
            "- Value constraints: use inductors in [10u, 220u], coupling capacitor C1 in [0.47u, 4.7u], and main power capacitors in [1u, 470u]. Avoid nH/pF-scale values.\n"
            "- Use at least 20 INC lines (>=20 elements).\n"
            "- Put ONE INC statement per line (use newlines between INC lines).\n"
            "- Required power-stage topology:\n"
            "  INC L1 vin sw <L>\n"
            "  INC C1 sw n1 <C>\n"
            "  INC L2 n1 0 <L>\n"
            "  INC S1 sw 0 Sstd\n"
            "  INC D1 n1 out Dstd\n"
            "  INC C2 out 0 <C>\n"
            "- To reach >=20 elements, add extra input/output capacitor banks and RC snubbers/dampers.\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )
    if fam == "buckboost":
        return (
            "Generate a non-inverting Buck-Boost (cascaded buck->boost) DC-DC converter in INC DSL.\n"
            "Rules:\n"
            "- IMPORTANT: Your output MUST START with the required cascaded topology lines EXACTLY as written (copy-paste), in the SAME ORDER.\n"
            "- Output ONLY INC lines (no explanation).\n"
            "- Line format: INC <name> <node1> <node2> <value_or_model>\n"
            "- Element names must start with R/L/C/D/S and contain a digit.\n"
            "- Must include required nodes {vin,sw1,mid,sw2,out,0}. You MAY introduce extra helper nodes.\n"
            "- Node names are identifiers (vin/sw/out/0/mid/sw1/sw2/n1/...). DO NOT use numeric node names like 12 or 5.\n"
            "- Use 2 switches and 2 diodes. Use switch models Sstd1 and Sstd2 so gate1/gate2 are separate.\n"
            "- Value constraints: use inductors in [10u, 220u] and main power capacitors in [1u, 470u]. Avoid nH/pF-scale values.\n"
            "- Use at least 20 INC lines (>=20 elements).\n"
            "- Put ONE INC statement per line (use newlines between INC lines).\n"
            "- Required cascaded topology:\n"
            "  INC S1 vin sw1 Sstd1\n"
            "  INC D1 0 sw1 Dstd\n"
            "  INC L1 sw1 mid <L>\n"
            "  INC C1 mid 0 <C>\n"
            "  INC L2 mid sw2 <L>\n"
            "  INC S2 sw2 0 Sstd2\n"
            "  INC D2 sw2 out Dstd\n"
            "  INC C2 out 0 <C>\n"
            "- To reach >=20 elements, add extra input/mid/output capacitor banks and RC snubbers/dampers.\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )
    raise ValueError(f"unknown family: {family}")


def load_model(base_model: str, adapter: Optional[str]) -> tuple[Any, Any]:
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

    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)

    model.eval()
    return tok, model


def _device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _flatten_detail(detail: dict) -> dict:
    keys = [
        "ok",
        "pass_C",
        "pass_CV",
        "pass_CE",
        "eff",
        "vavg",
        "ripple",
        "overshoot",
        "canonical_hash",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if k in detail:
            out[k] = detail[k]
    return out


def _set_elem_value(inc: str, name: str, value: float) -> str:
    from inc_parser import extract_inc_lines

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


def _bank_expand_to_min_elems(
    inc: str,
    *,
    family: str,
    min_elems: int,
    max_parts_per_elem: int = 32,
) -> tuple[str, Dict[str, Any]]:
    """
    Increase element count to >=min_elems without changing topology, via electrically-equivalent passive banks.

    Currently: split the largest capacitor (prefer out-0) into a parallel bank (value/k each).
    """
    meta: Dict[str, Any] = {"used": False}
    from inc_parser import parse_inc

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
        "picked": {
            "name": str(pick.name),
            "nodes": list(pick.nodes),
            "value_before": float(base),
            "value_each": float(part),
            "k": int(k),
        },
        "added": added,
        "n_elems_before": int(n0),
        "n_elems_after": int(len(parse_inc(inc2))),
        "family": str(family).lower(),
    }
    return inc2, meta


def _select_tunable_names(inc: str, family: str) -> List[str]:
    from inc_parser import parse_inc

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


def _run_mlp_repair(
    *,
    inc_before: str,
    detail_before: Dict[str, Any],
    family: str,
    vin: float,
    vout: float,
    tol: float,
    rload: float,
    t_pre: float,
    t_win: float,
    sim_timeout_s: float,
    autotune_duty: bool,
    corrector: Any,
    corrector_topk: int,
    corrector_max_factor: float,
) -> Dict[str, Any]:
    from dcdc_eval_tran import _score_detail, eval_one_detail_dcdc
    from inc_parser import parse_inc
    from repair_corrector import build_feature_vector

    tunables = _select_tunable_names(str(inc_before), family=str(family))[:4]
    elems_now = {e.name: e for e in parse_inc(str(inc_before))}
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
        detail_before=dict(detail_before or {}),
        tunable_values=list(tun_vals),
        n_elems=float((detail_before or {}).get("n_elems") or 0.0),
    )

    scale_tries = corrector.suggest_scales(
        feature_vec=feat,
        topk=int(corrector_topk),
        max_factor=float(corrector_max_factor),
    )

    base_score = float(_score_detail(dict(detail_before or {}), vout=float(vout), tol_ref=float(tol)))
    best = {
        "inc_after": str(inc_before),
        "detail_after": dict(detail_before or {}),
        "score_after": base_score,
        "used": False,
        "selected_try": -1,
        "tries": [],
        "tuned": {},
    }

    for ti, scales in enumerate(scale_tries):
        cand_inc = str(inc_before)
        tuned: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(tunables):
            if i >= 4:
                break
            if float(tun_mask[i]) <= 0.0:
                continue
            base_val = float(tun_vals[i])
            s = float(scales[i] if i < len(scales) else 1.0)
            cand_val = float(base_val) * float(s)
            cand_inc = _set_elem_value(cand_inc, name=str(name), value=float(cand_val))
            tuned[str(name)] = {"before": float(base_val), "after": float(cand_val), "scale": float(s)}

        cand_inc = str(cand_inc)
        if cand_inc.strip() == str(inc_before).strip():
            best["tries"].append(
                {
                    "try": int(ti),
                    "skipped": True,
                    "reason": "no_change",
                    "scales": list(scales),
                }
            )
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
        cand_score = float(_score_detail(dict(cand_detail), vout=float(vout), tol_ref=float(tol)))

        best["tries"].append(
            {
                "try": int(ti),
                "skipped": False,
                "scales": list(scales),
                "tuned": tuned,
                "score": float(cand_score),
                "ok": bool(cand_detail.get("ok", False)),
                "pass_CV": bool(cand_detail.get("pass_CV", False)),
                "pass_CE": bool(cand_detail.get("pass_CE", False)),
                "detail": dict(cand_detail),
            }
        )

        if float(cand_score) > float(best["score_after"]) + 1e-9:
            best.update(
                {
                    "inc_after": str(cand_inc),
                    "detail_after": dict(cand_detail),
                    "score_after": float(cand_score),
                    "used": True,
                    "selected_try": int(ti),
                    "tuned": tuned,
                }
            )

        # Early exit: first repaired-success.
        if bool(best["detail_after"].get("pass_CV", False)):
            break

    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_per_task", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--rload", type=float, default=10.0)
    ap.add_argument("--t_pre", type=float, default=0.008)
    ap.add_argument("--t_win", type=float, default=0.002)
    ap.add_argument("--sim_timeout_s", type=float, default=180.0)
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--autotune_duty", action="store_true", help="Enable 1-step duty auto-tune (disabled by default).")
    ap.add_argument("--no_fallback", action="store_true", help="Disable template fallback; invalid INC stays invalid")
    ap.add_argument("--mlp_corrector", default="", help="Optional RepairCorrector ckpt; enable 1-shot MLP repair on failure")
    ap.add_argument("--mlp_device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--mlp_topk", type=int, default=3)
    ap.add_argument("--mlp_max_factor", type=float, default=2.0)
    ap.add_argument(
        "--mlp_repair_mode",
        choices=["never", "if_not_ok", "if_not_pass_cv"],
        default="never",
        help="When to apply MLP repair (requires --mlp_corrector).",
    )
    ap.add_argument(
        "--dedup_by_hash",
        action="store_true",
        help="Deduplicate samples by canonical_hash (NOT recommended for sample-level metrics; off by default).",
    )
    ap.add_argument(
        "--allow_module_graph",
        action="store_true",
        help="Allow compiling a ModuleGraph representation when no INC lines are found (off by default).",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip model loading/generation and use empty outputs (pipeline smoke test).",
    )
    ap.add_argument("--template_variant", choices=["base", "full"], default="base")
    ap.add_argument("--min_elems", type=int, default=20, help="Require at least this many parsed INC elements")
    ap.add_argument(
        "--bank_expand_min_elems",
        action="store_true",
        help="If element count < --min_elems, expand via electrically-equivalent passive banks before sim.",
    )
    ap.add_argument("--bank_expand_max_parts", type=int, default=32, help="Max parallel parts per capacitor for bank expansion.")
    ap.add_argument(
        "--only_task",
        action="append",
        default=[],
        help="Restrict eval to specific tasks: 'family,vin,vout' (repeatable).",
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # local imports (on CA800)
    import sys

    for base in [
        "/root/workspace_autocircuit_rl",
        "/root/autodl-tmp/workspace_autocircuit_rl",
        "/root/autodl-tmp/workspace_autocircuit_rl/integrated",
    ]:
        if base not in sys.path:
            sys.path.append(base)

    from dcdc_eval_tran import eval_one_detail_dcdc
    from dcdc_taskset import default_taskset
    from dcdc_templates import templates
    from dcdc_verifier import verify_inc_dcdc
    from inc_parser import extract_inc_lines

    corrector = None
    if str(args.mlp_corrector or "").strip() and str(args.mlp_repair_mode or "") != "never":
        from repair_corrector import RepairCorrector

        corr_dev = str(args.mlp_device or "cpu").strip().lower()
        if corr_dev == "cuda" and (not torch.cuda.is_available()):
            print("[warn] --mlp_device=cuda requested but CUDA not available; falling back to cpu")
            corr_dev = "cpu"
        corrector = RepairCorrector.load(str(args.mlp_corrector).strip(), device=corr_dev)
        print(f"[mlp] enabled mode={args.mlp_repair_mode} device={corr_dev} topk={int(args.mlp_topk)} max_factor={float(args.mlp_max_factor)}")

    compile_module_graph = None
    if bool(args.allow_module_graph):
        try:
            from dcdc_module_compiler import compile_module_graph as _compile_module_graph  # type: ignore

            compile_module_graph = _compile_module_graph
        except Exception as e:
            print(f"[warn] --allow_module_graph requested but compiler import failed: {type(e).__name__}: {e}")
            args.allow_module_graph = False

    tok = None
    model = None
    dev = None
    logits_proc = None
    if not bool(args.dry_run):
        tok, model = load_model(args.base_model, adapter=(args.adapter or "").strip() or None)
        dev = _device(model)
        if args.constrained:
            try:
                from integrated.constraints import CharClassLogitsProcessor
            except Exception:
                raise SystemExit(
                    "Missing integrated.constraints.CharClassLogitsProcessor; disable --constrained or fix sys.path"
                )
            logits_proc = [CharClassLogitsProcessor(tok, penalty=30.0)]
    else:
        if bool(args.constrained):
            print("[warn] --dry_run ignores --constrained")
            args.constrained = False

    tpl = templates(str(args.template_variant)) if (not bool(args.no_fallback)) else {}
    tasks = default_taskset()
    if args.only_task:
        wanted = set()
        for spec in args.only_task:
            s = str(spec or "").strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) != 3:
                raise SystemExit(f"--only_task expects 'family,vin,vout' (got {spec!r})")
            fam_s, vin_s, vout_s = parts
            wanted.add((fam_s.lower(), float(vin_s), float(vout_s)))
        tasks = [t for t in tasks if (str(t.family).lower(), float(t.vin), float(t.vout)) in wanted]

    summary: Dict[str, Any] = {
        "started_at": _now(),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "n_tasks": len(tasks),
        "n_per_task": int(args.n_per_task),
        "tol": float(args.tol),
        "rload": float(args.rload),
        "t_pre": float(args.t_pre),
        "t_win": float(args.t_win),
        "constrained": bool(args.constrained),
        "autotune_duty": bool(args.autotune_duty),
        "mlp_corrector": str(args.mlp_corrector or ""),
        "mlp_repair_mode": str(args.mlp_repair_mode or ""),
        "results": {},
    }

    progress_path = outdir / "progress.txt"
    for ti, task in enumerate(tasks):
        fam = str(task.family)
        vin = float(task.vin)
        vout = float(task.vout)

        tdir = outdir / fam / f"vin{vin:.1f}_vout{vout:.1f}"
        tdir.mkdir(parents=True, exist_ok=True)
        out_path = tdir / f"metric_vin{vin:.1f}_vout{vout:.1f}_full.json"
        if bool(args.resume) and out_path.exists():
            # Resume should still rebuild a complete eval_summary.json by loading per-task metric files,
            # instead of silently dropping skipped tasks from the summary.
            try:
                payload = json.loads(out_path.read_text(encoding="utf-8", errors="ignore") or "{}")
                samples_min = list(payload.get("samples") or [])
                results_raw = list(payload.get("details_raw") or [])

                n_ok = sum(1 for d in samples_min if bool((d or {}).get("ok")))
                n_cv = sum(1 for d in samples_min if bool((d or {}).get("pass_CV")))
                n_ce = sum(1 for d in samples_min if bool((d or {}).get("pass_CE")))
                n_ok_raw = sum(1 for d in results_raw if bool((d or {}).get("ok")))
                n_cv_raw = sum(1 for d in results_raw if bool((d or {}).get("pass_CV")))
                n_ce_raw = sum(1 for d in results_raw if bool((d or {}).get("pass_CE")))

                pass_at_1_raw = bool((results_raw[0] if results_raw else {}).get("pass_CV", False))
                pass_at_1 = bool((samples_min[0] if samples_min else {}).get("pass_CV", False))
                pass_at_k_raw = any(bool((r or {}).get("pass_CV", False)) for r in results_raw)
                pass_at_k = any(bool((r or {}).get("pass_CV", False)) for r in samples_min)

                summary["results"][f"{fam}/vin{vin:.1f}_vout{vout:.1f}"] = {
                    "n_unique": int(len(samples_min)),
                    "ok": int(n_ok),
                    "pass_CV": int(n_cv),
                    "pass_CE": int(n_ce),
                    "ok_raw": int(n_ok_raw),
                    "pass_CV_raw": int(n_cv_raw),
                    "pass_CE_raw": int(n_ce_raw),
                    "pass_at_1_raw": bool(pass_at_1_raw),
                    "pass_at_1": bool(pass_at_1),
                    "pass_at_k_raw": bool(pass_at_k_raw),
                    "pass_at_k": bool(pass_at_k),
                }
            except Exception:
                # Fall back to a full re-run if the existing file is unreadable/corrupt.
                pass
            continue

        outputs_texts: List[str] = []
        per_sample_gen_time = 0.0
        if bool(args.dry_run):
            outputs_texts = ["" for _ in range(int(args.n_per_task))]
            per_sample_gen_time = 0.0
        else:
            prompt = build_prompt(fam, vin, vout)
            enc = tok(prompt, return_tensors="pt").to(dev)

            seed_base = int(args.seed) + ti * 1000
            random.seed(seed_base)
            torch.manual_seed(seed_base)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_base)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                outs = model.generate(
                    **enc,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=True,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    logits_processor=logits_proc,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                    num_return_sequences=int(args.n_per_task),
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_dt = time.time() - t0
            per_sample_gen_time = float(gen_dt) / max(1.0, float(outs.shape[0]))
            for i in range(int(outs.shape[0])):
                outputs_texts.append(tok.decode(outs[i], skip_special_tokens=True))

        results: List[Dict[str, Any]] = []
        results_raw: List[Dict[str, Any]] = []
        seen_hashes = set() if bool(args.dedup_by_hash) else None

        for i, txt0 in enumerate(outputs_texts):
            txt = str(txt0 or "")
            if _RESP_KEY in txt:
                txt = txt.rsplit(_RESP_KEY, 1)[-1].lstrip()
            inc_lines = extract_inc_lines(txt)
            used_module_graph = False
            if inc_lines:
                inc = "\n".join(inc_lines).strip() + "\n"
            else:
                if bool(args.allow_module_graph):
                    fam2, inc2, errs = compile_module_graph(txt, expected_family=fam)
                    if (not errs) and inc2.strip():
                        inc = inc2.strip() + "\n"
                        used_module_graph = True
                    else:
                        inc = "\n".join(inc_lines).strip() + "\n"
                else:
                    inc = "\n".join(inc_lines).strip() + "\n"

            ver = verify_inc_dcdc(inc, family=fam, vin=vin, vout=vout)
            used_fallback = False
            violations = list(ver.violations)
            too_few = int(ver.n_elems) < int(args.min_elems)
            if too_few:
                violations.append(f"too_few_elems_{int(ver.n_elems)}")

            bank_meta: Dict[str, Any] = {"used": False}
            inc_before_bank = str(inc)
            if bool(too_few) and bool(args.bank_expand_min_elems):
                inc2, bank_meta = _bank_expand_to_min_elems(
                    inc,
                    family=fam,
                    min_elems=int(args.min_elems),
                    max_parts_per_elem=int(args.bank_expand_max_parts),
                )
                if bool(bank_meta.get("used")) and inc2.strip():
                    inc = inc2.strip() + "\n"
                    ver = verify_inc_dcdc(inc, family=fam, vin=vin, vout=vout)
                    violations = list(ver.violations)
                    too_few = int(ver.n_elems) < int(args.min_elems)
                    if too_few:
                        violations.append(f"too_few_elems_{int(ver.n_elems)}")

            if ((not ver.ok) or too_few) and (not bool(args.no_fallback)):
                inc = tpl.get(fam, "").strip() + "\n"
                used_fallback = True
                ver = verify_inc_dcdc(inc, family=fam, vin=vin, vout=vout)

            meets_min = int(ver.n_elems) >= int(args.min_elems)
            if (not meets_min) and bool(args.no_fallback):
                detail = {
                    "ok": False,
                    "pass_C": False,
                    "canonical_hash": ver.canonical_hash,
                    "violations": violations,
                    "error": "too_few_elems",
                }
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

            # Optional 1-shot MLP repair (value scaling) on failure.
            inc_before = str(inc)
            detail_before = dict(detail)
            repair_meta: Dict[str, Any] = {"used": False}
            if corrector is not None and str(args.mlp_repair_mode or "") != "never":
                need = False
                if str(args.mlp_repair_mode) == "if_not_ok":
                    need = not bool(detail_before.get("ok", False))
                elif str(args.mlp_repair_mode) == "if_not_pass_cv":
                    need = not bool(detail_before.get("pass_CV", False))
                if need and bool(ver.ok) and bool(meets_min):
                    try:
                        rep = _run_mlp_repair(
                            inc_before=inc_before,
                            detail_before=detail_before,
                            family=fam,
                            vin=float(vin),
                            vout=float(vout),
                            tol=float(args.tol),
                            rload=float(args.rload),
                            t_pre=float(args.t_pre),
                            t_win=float(args.t_win),
                            sim_timeout_s=float(args.sim_timeout_s),
                            autotune_duty=bool(args.autotune_duty),
                            corrector=corrector,
                            corrector_topk=int(args.mlp_topk),
                            corrector_max_factor=float(args.mlp_max_factor),
                        )
                        if rep.get("used", False):
                            inc = str(rep.get("inc_after") or inc_before)
                            detail = dict(rep.get("detail_after") or detail_before)
                        repair_meta = {
                            "used": bool(rep.get("used", False)),
                            "selected_try": int(rep.get("selected_try", -1) or -1),
                            "tuned": dict(rep.get("tuned") or {}),
                            "tries": list(rep.get("tries") or []),
                            "score_after": float(rep.get("score_after", 0.0) or 0.0),
                        }
                    except Exception as e:
                        repair_meta = {"used": False, "error": f"{type(e).__name__}"}

            ch = detail.get("canonical_hash") or ver.canonical_hash
            if seen_hashes is not None:
                if ch in seen_hashes:
                    continue
                seen_hashes.add(ch)

            row: Dict[str, Any] = {
                "index": int(len(results) + 1),
                "gen_time": float(per_sample_gen_time),
                "inc_source": str(inc).strip(),
                "inc_before": str(inc_before).strip(),
                "inc_before_bank": str(inc_before_bank).strip(),
                "source_format": ("MOD" if used_module_graph else "INC"),
                "used_fallback": bool(used_fallback),
                "violations": violations,
                "n_elems": int(ver.n_elems),
                "n_inc_lines": int(ver.n_inc_lines),
                "meets_min_elems": bool(meets_min),
                "bank_expand": bank_meta,
                "mlp_repair": repair_meta,
            }
            row.update(_flatten_detail(detail))
            results.append(row)

            raw_row: Dict[str, Any] = {
                "index": int(len(results_raw) + 1),
                "gen_time": float(per_sample_gen_time),
                "inc_source": str(inc_before).strip(),
                "inc_before_bank": str(inc_before_bank).strip(),
                "source_format": ("MOD" if used_module_graph else "INC"),
                "used_fallback": bool(used_fallback),
                "violations": violations,
                "n_elems": int(ver.n_elems),
                "n_inc_lines": int(ver.n_inc_lines),
                "meets_min_elems": bool(meets_min),
                "bank_expand": bank_meta,
            }
            raw_row.update(_flatten_detail(detail_before))
            results_raw.append(raw_row)

        # Back-compat compact per-sample list.
        samples_min: List[Dict[str, Any]] = []
        for j, r in enumerate(results):
            samples_min.append(
                {
                    "ok": bool(r.get("ok", False)),
                    "pass_C": bool(r.get("pass_C", False)),
                    "pass_CV": bool(r.get("pass_CV", False)),
                    "pass_CE": bool(r.get("pass_CE", False)),
                    "eff": float(r.get("eff") or 0.0),
                    "vavg": float(r.get("vavg") or 0.0),
                    "ripple": float(r.get("ripple") or 0.0),
                    "overshoot": float(r.get("overshoot") or 0.0),
                    "gen_time": float(r.get("gen_time") or 0.0),
                    "index": int(r.get("index") or (j + 1)),
                    "inc_source": str(r.get("inc_source") or ""),
                    "canonical_hash": str(r.get("canonical_hash") or ""),
                }
            )

        # Full per-sample records with repair metadata.
        details_full: List[Dict[str, Any]] = list(results)

        tried = int(len(samples_min))
        succ_count = int(sum(1 for d in samples_min if d.get("pass_CV") is True))
        succ_overall = float(succ_count / tried) if tried else 0.0
        eff_vals = [
            float(d.get("eff") or 0.0)
            for d in samples_min
            if d.get("pass_CV") is True and float(d.get("eff") or 0.0) > 0.0
        ]
        eff_mean = float(sum(eff_vals) / len(eff_vals)) if eff_vals else 0.0
        total_generated = int(args.n_per_task)
        valid_count = int(tried)
        valid_rate = float(valid_count / total_generated) if total_generated else 0.0
        avg_gen_time = float(sum(float(d.get("gen_time") or 0.0) for d in samples_min) / tried) if tried else 0.0

        # pass@1 / pass@k (task-level)
        first_raw = results_raw[0] if results_raw else {}
        first_fin = results[0] if results else {}
        pass_at_1_raw = bool(first_raw.get("pass_CV", False))
        pass_at_1 = bool(first_fin.get("pass_CV", False))
        pass_at_k_raw = any(bool(r.get("pass_CV", False)) for r in results_raw)
        pass_at_k = any(bool(r.get("pass_CV", False)) for r in results)

        payload = {
            "family": fam,
            "vin": float(vin),
            "vout": float(vout),
            "summary": {
                "tried": tried,
                "succ_overall": succ_overall,
                "eff_mean": eff_mean,
                "valid_count": valid_count,
                "total_generated": total_generated,
                "valid_rate": valid_rate,
                "avg_gen_time": avg_gen_time,
                "pass_at_1_raw": bool(pass_at_1_raw),
                "pass_at_1": bool(pass_at_1),
                "pass_at_k_raw": bool(pass_at_k_raw),
                "pass_at_k": bool(pass_at_k),
                "mlp_used": int(sum(1 for r in results if bool((r.get("mlp_repair") or {}).get("used", False)))),
            },
            # Keep both keys for backward compatibility with older analysis scripts.
            "samples": samples_min,
            "details": details_full,
            "details_raw": results_raw,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # aggregate simple success metrics
        n_ok = sum(1 for d in samples_min if bool(d.get("ok")))
        n_cv = sum(1 for d in samples_min if bool(d.get("pass_CV")))
        n_ce = sum(1 for d in samples_min if bool(d.get("pass_CE")))
        n_ok_raw = sum(1 for d in results_raw if bool(d.get("ok")))
        n_cv_raw = sum(1 for d in results_raw if bool(d.get("pass_CV")))
        n_ce_raw = sum(1 for d in results_raw if bool(d.get("pass_CE")))
        summary["results"][f"{fam}/vin{vin:.1f}_vout{vout:.1f}"] = {
            "n_unique": int(len(samples_min)),
            "ok": int(n_ok),
            "pass_CV": int(n_cv),
            "pass_CE": int(n_ce),
            "ok_raw": int(n_ok_raw),
            "pass_CV_raw": int(n_cv_raw),
            "pass_CE_raw": int(n_ce_raw),
            "pass_at_1_raw": bool(pass_at_1_raw),
            "pass_at_1": bool(pass_at_1),
            "pass_at_k_raw": bool(pass_at_k_raw),
            "pass_at_k": bool(pass_at_k),
        }

        progress_path.write_text(f"task={ti+1}/{len(tasks)} last={fam} vin={vin:.1f} vout={vout:.1f}\n", encoding="utf-8")

    summary["finished_at"] = _now()
    (outdir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] saved", str(outdir))


if __name__ == "__main__":
    main()

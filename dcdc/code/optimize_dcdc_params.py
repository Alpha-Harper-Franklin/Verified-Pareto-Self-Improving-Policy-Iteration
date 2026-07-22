#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dcdc_spice_builder import build_dcdc_spice
from dcdc_verifier import verify_inc_dcdc
from inc_parser import IncElem, parse_inc, to_inc_text


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p2 = max(1e-6, min(1.0 - 1e-6, float(p)))
    return math.log(p2 / (1.0 - p2))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _score(detail: dict, vout: float, tol_ref: float) -> float:
    if not detail.get("ok"):
        return -1.0

    vavg = float(detail.get("vavg", 0.0) or 0.0)
    eff = float(detail.get("eff", 0.0) or 0.0)
    ripple = float(detail.get("ripple", 0.0) or 0.0)
    overshoot = float(detail.get("overshoot", 0.0) or 0.0)

    err = abs(vavg - float(vout)) / max(1e-6, float(vout))
    # voltage tracking dominates
    score_v = 3.0 * max(0.0, 1.0 - (err / max(1e-6, float(tol_ref))))
    score_eff = 0.5 * eff
    score_ripple = -0.2 * (ripple / max(1e-6, float(vout)))
    score_over = -0.2 * overshoot

    # hard constraints
    if not detail.get("pass_CV", False):
        score_v -= 1.0
    if not detail.get("pass_CE", False):
        score_eff -= 0.5

    return float(score_v + score_eff + score_ripple + score_over)


def _inject_measures(cir: str, t0: float, t1: float, vin_node: str) -> str:
    meas_lines = [
        f".meas tran vavg AVG v(out) from={t0:.6e} to={t1:.6e}",
        f".meas tran vmax MAX v(out) from={t0:.6e} to={t1:.6e}",
        f".meas tran vmin MIN v(out) from={t0:.6e} to={t1:.6e}",
        f".meas tran iavg AVG i(V_IN) from={t0:.6e} to={t1:.6e}",
        f".meas tran vinavg AVG v({vin_node}) from={t0:.6e} to={t1:.6e}",
    ]

    out_lines: List[str] = []
    inserted = False
    for raw in (cir or "").splitlines():
        s = raw.strip()
        low = s.lower()
        if (not inserted) and low.startswith(".control"):
            out_lines.extend(meas_lines)
            inserted = True
        if low.startswith("tran "):
            toks = s.split()
            if len(toks) >= 3 and toks[0].lower() == "tran":
                raw = f"tran {toks[1]} {t1:.6e}"
        if low.startswith("wrdata "):
            continue
        out_lines.append(raw)
    if not inserted:
        out_lines.extend(meas_lines)
    return "\n".join(out_lines) + "\n"


_MEAS_RE = re = __import__("re").compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+0-9.eE]+)")


def _parse_meas(text: str) -> dict:
    out: Dict[str, float] = {}
    for raw in (text or "").splitlines():
        m = _MEAS_RE.match(raw)
        if not m:
            continue
        k = m.group(1).strip().lower()
        try:
            v = float(m.group(2))
        except Exception:
            continue
        out[k] = v
    return out


def _simulate_ngspice(cir: str, timeout_s: int = 180) -> dict:
    import shutil
    import subprocess
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="dcdc_opt_")
    try:
        from pathlib import Path

        tmp = Path(tmpdir)
        cpath = tmp / "net.cir"
        opath = tmp / "out.log"
        cpath.write_text(cir, encoding="utf-8")
        try:
            subprocess.run(
                ["ngspice", "-b", "-o", str(opath), str(cpath)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=int(timeout_s),
                cwd=str(tmpdir),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"timeout": True}

        try:
            text = opath.read_text(errors="replace")
        except Exception:
            return {"io_error": True}

        return _parse_meas(text)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@dataclass
class SearchSpace:
    # per variable: (name, kind, lo, hi)
    vars: List[Tuple[str, str, float, float]]


def _make_space(elems: List[IncElem], family: str, opt_duty: bool, opt_freq: bool) -> SearchSpace:
    vars: List[Tuple[str, str, float, float]] = []

    # passive values in log-space
    for e in elems:
        if e.kind == "L":
            vars.append((e.name, "log", 1e-6, 800e-6))
        if e.kind == "C":
            # allow smaller coupling caps
            vars.append((e.name, "log", 1e-9, 2000e-6))

    if opt_duty:
        fam = (family or "").lower().strip()
        if fam in {"buckboost", "buck-boost", "bb"}:
            vars.append(("duty_gate1", "logit", 0.05, 0.95))
            vars.append(("duty_gate2", "logit", 0.05, 0.95))
        else:
            vars.append(("duty", "logit", 0.05, 0.95))

    if opt_freq:
        vars.append(("freq", "log", 150e3, 600e3))

    return SearchSpace(vars=vars)


def _encode_init(elems: List[IncElem], family: str, space: SearchSpace, vin: float, vout: float, rload: float) -> np.ndarray:
    # use build_dcdc_spice to get default duty and freq
    _, meta = build_dcdc_spice(to_inc_text(elems), family=family, vin=vin, vout=vout, rload=rload)

    vals: Dict[str, float] = {}
    for e in elems:
        if e.kind in {"L", "C"}:
            vals[e.name] = float(e.value) if e.value is not None else (47e-6 if e.kind == "L" else 47e-6)

    vals["freq"] = float(meta.freq)
    if meta.duty_map:
        for k, v in meta.duty_map.items():
            vals[f"duty_{k}"] = float(v)
    # canonical single duty key
    if "duty_gate" not in vals and "duty_gate" not in vals:
        pass

    x0 = []
    for name, kind, lo, hi in space.vars:
        if name.startswith("duty_") and name != "duty":
            p = vals.get(name, 0.5)
            x0.append(_logit(_clamp(p, lo, hi)))
        elif name == "duty":
            p = vals.get("duty_gate", vals.get("duty_gate1", vals.get("duty_gate2", 0.5)))
            x0.append(_logit(_clamp(p, lo, hi)))
        elif kind == "log":
            v = vals.get(name, (lo + hi) * 0.5)
            v = _clamp(v, lo, hi)
            x0.append(math.log(v))
        else:
            x0.append(0.0)
    return np.array(x0, dtype=np.float64)


def _decode(x: np.ndarray, space: SearchSpace) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for (name, kind, lo, hi), xi in zip(space.vars, x.tolist()):
        if kind == "log":
            v = math.exp(float(xi))
            out[name] = _clamp(v, lo, hi)
        elif kind == "logit":
            p = _sigmoid(float(xi))
            out[name] = _clamp(p, lo, hi)
        else:
            out[name] = float(xi)
    return out


def _apply(elems: List[IncElem], params: Dict[str, float]) -> str:
    out_elems: List[IncElem] = []
    for e in elems:
        if e.kind in {"L", "C"} and e.name in params:
            out_elems.append(IncElem(name=e.name, kind=e.kind, nodes=list(e.nodes), value=float(params[e.name]), model=e.model, raw=e.raw))
        else:
            out_elems.append(e)
    return to_inc_text(out_elems) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["buck", "boost", "sepic", "buckboost"])
    ap.add_argument("--vin", type=float, required=True)
    ap.add_argument("--vout", type=float, required=True)
    ap.add_argument("--inc", default="")
    ap.add_argument("--inc_file", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rload", type=float, default=10.0)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--tol_ref", type=float, default=0.02)
    ap.add_argument("--t_pre", type=float, default=0.008)
    ap.add_argument("--t_win", type=float, default=0.002)
    ap.add_argument("--opt_duty", action="store_true")
    ap.add_argument("--opt_freq", action="store_true")
    ap.add_argument("--robust", action="store_true", help="Optimize a robust aggregated score over a condition set.")
    ap.add_argument("--vin_jitter", type=float, default=0.10)
    ap.add_argument("--rload_list", default="5,10,20", help="Comma-separated Rload values for robust mode.")
    ap.add_argument("--agg", choices=["cvar", "worst"], default="cvar")
    ap.add_argument("--cvar_alpha", type=float, default=0.25)
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.inc_file:
        inc_text = Path(args.inc_file).read_text(errors="replace")
    else:
        inc_text = args.inc

    ver = verify_inc_dcdc(inc_text, family=args.family, vin=args.vin, vout=args.vout)
    if not ver.ok:
        raise SystemExit(f"invalid inc for {args.family}: {ver.violations}")

    elems = parse_inc(inc_text)
    space = _make_space(elems, family=args.family, opt_duty=bool(args.opt_duty), opt_freq=bool(args.opt_freq))
    x_mean = _encode_init(elems, args.family, space, vin=args.vin, vout=args.vout, rload=args.rload)
    x_std = np.ones_like(x_mean) * 0.7

    history_path = outdir / "history.jsonl"
    best_path = outdir / "best.json"

    best = {"score": -1e9}

    def evaluate(params: Dict[str, float]) -> dict:
        inc2 = _apply(elems, params)

        duty_override: Optional[float] = None
        duty1_override: Optional[float] = None
        duty2_override: Optional[float] = None
        freq = None

        if args.opt_duty:
            if args.family == "buckboost":
                duty1_override = params.get("duty_gate1")
                duty2_override = params.get("duty_gate2")
            else:
                duty_override = params.get("duty")

        if args.opt_freq:
            freq = params.get("freq")

        def _eval_once(vin_i: float, rload_i: float) -> dict:
            cir, meta = build_dcdc_spice(
                inc2,
                family=args.family,
                vin=float(vin_i),
                vout=args.vout,
                rload=float(rload_i),
                freq=float(freq) if freq is not None else 200_000.0,
                duty_override=duty_override,
                duty1_override=duty1_override,
                duty2_override=duty2_override,
            )
            cir2 = _inject_measures(
                cir,
                float(args.t_pre),
                float(args.t_pre) + float(args.t_win),
                vin_node=meta.vin_node,
            )
            meas = _simulate_ngspice(cir2)

            if "vavg" not in meas:
                detail = {"ok": False, "pass_C": False}
            else:
                vavg = float(meas["vavg"])
                vmax = float(meas.get("vmax", vavg))
                vmin = float(meas.get("vmin", vavg))
                iavg = float(meas.get("iavg", 0.0))
                vinavg = float(meas.get("vinavg", float(vin_i)))

                ripple = float(vmax - vmin)
                overshoot = float(max(0.0, (vmax - float(args.vout)) / max(1e-6, float(args.vout))))
                pass_CV = bool(abs(vavg - float(args.vout)) / max(1e-6, float(args.vout)) <= float(args.tol))

                pin = abs(vinavg * iavg)
                pout = (vavg * vavg) / max(1e-6, float(rload_i))
                eff = float((pout / pin) if pin > 1e-12 else 0.0)
                pass_CE = bool(eff >= 0.8)

                detail = {
                    "ok": True,
                    "pass_C": True,
                    "pass_CV": pass_CV,
                    "pass_CE": pass_CE,
                    "eff": eff,
                    "vavg": vavg,
                    "ripple": ripple,
                    "overshoot": overshoot,
                }

            score = float(_score(detail, vout=args.vout, tol_ref=args.tol_ref))
            return {
                "vin": float(vin_i),
                "rload": float(rload_i),
                "score": float(score),
                "detail": detail,
                "meta": {"freq": float(meta.freq), "duty_map": meta.duty_map},
            }

        if not bool(args.robust):
            base = _eval_once(float(args.vin), float(args.rload))
            return {
                "score": float(base["score"]),
                "params": params,
                "detail": base["detail"],
                "inc": inc2,
                "meta": base["meta"],
                "robust": None,
            }

        # Robust aggregated score over a small condition set.
        try:
            rloads = [float(x) for x in str(args.rload_list).split(",") if str(x).strip()]
        except Exception:
            rloads = [5.0, 10.0, 20.0]
        if not rloads:
            rloads = [float(args.rload)]

        vj = float(args.vin_jitter)
        vin0 = float(args.vin)
        vin_list = [vin0 * (1.0 - vj), vin0, vin0 * (1.0 + vj)]

        scenarios = [_eval_once(vv, rr) for vv in vin_list for rr in rloads]
        scores = [float(s["score"]) for s in scenarios]

        mode = str(args.agg).strip().lower()
        if mode == "worst":
            agg_score = float(min(scores))
        else:
            alpha = max(1e-6, min(1.0, float(args.cvar_alpha)))
            k = max(1, int(round(alpha * len(scores))))
            worst_k = sorted(scores)[:k]
            agg_score = float(sum(worst_k) / max(1, len(worst_k)))

        base = next(
            (s for s in scenarios if abs(s["vin"] - vin0) < 1e-9 and abs(s["rload"] - float(args.rload)) < 1e-9),
            scenarios[0],
        )
        return {
            "score": float(agg_score),
            "params": params,
            "detail": base["detail"],
            "inc": inc2,
            "meta": base["meta"],
            "robust": {
                "mode": mode,
                "vin_jitter": float(args.vin_jitter),
                "rload_list": [float(x) for x in rloads],
                "cvar_alpha": float(args.cvar_alpha),
                "scenarios": scenarios,
            },
        }

    t_start = time.time()
    n_eval = 0

    with history_path.open("w", encoding="utf-8") as hf:
        while n_eval < int(args.budget):
            # sample population
            pop = []
            for _ in range(int(args.pop)):
                if n_eval >= int(args.budget):
                    break
                x = x_mean + rng.normal(size=x_mean.shape) * x_std
                params = _decode(x, space)
                res = evaluate(params)
                res["step"] = int(n_eval + 1)
                res["elapsed_s"] = float(time.time() - t_start)
                hf.write(json.dumps(res, ensure_ascii=False) + "\n")
                hf.flush()

                pop.append((res["score"], x, res))
                n_eval += 1

                if res["score"] > float(best["score"]):
                    best = res
                    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

            if not pop:
                break

            pop.sort(key=lambda t: t[0], reverse=True)
            elites = pop[: max(1, int(args.elite))]
            xs = np.stack([x for _, x, _ in elites], axis=0)
            x_mean = xs.mean(axis=0)
            x_std = np.maximum(xs.std(axis=0), 0.15)

    (outdir / "best_inc.txt").write_text(best.get("inc", ""), encoding="utf-8")
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "family": args.family,
                "vin": args.vin,
                "vout": args.vout,
                "budget": int(args.budget),
                "best_score": float(best.get("score", -1e9)),
                "best_detail": best.get("detail", {}),
                "best_meta": best.get("meta", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

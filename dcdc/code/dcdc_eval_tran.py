from __future__ import annotations

import re
import os
import signal
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from dcdc_spice_builder import build_dcdc_spice
from dcdc_verifier import verify_inc_dcdc

_MEAS_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+0-9.eE]+)")


def _score_detail(detail: dict, vout: float, tol_ref: float = 0.1) -> float:
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
        # Partial credit beyond tol to keep a useful learning signal for near-misses.
        score_v = -min(1.0, (err - tol) / tol)
    score_eff = 0.5 * eff
    score_ripple = -0.2 * (ripple / max(1e-6, float(vout)))
    score_over = -0.2 * overshoot

    if not bool(detail.get("pass_CE", False)):
        score_eff -= 0.5

    return float(score_v + score_eff + score_ripple + score_over)


def _parse_meas(text: str) -> dict:
    out = {}
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


def inject_measures(cir: str, t0: float, t1: float, vin_node: str) -> str:
    meas_lines = [
        f".meas tran vavg AVG v(out) from={t0:.6e} to={t1:.6e}",
        f".meas tran vmax MAX v(out) from={t0:.6e} to={t1:.6e}",
        f".meas tran vmin MIN v(out) from={t0:.6e} to={t1:.6e}",
        f".meas tran iavg AVG i(V_IN) from={t0:.6e} to={t1:.6e}",
        f".meas tran vinavg AVG v({vin_node}) from={t0:.6e} to={t1:.6e}",
    ]
    out_lines = []
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


def _run_ngspice(*, cpath: Path, opath: Path, cwd: str, timeout_s: float) -> Optional[str]:
    """
    Run ngspice in batch mode with a hard timeout.

    subprocess.run(timeout=...) is not always sufficient for ngspice: on some systems it can leave
    orphaned ngspice processes that keep consuming CPU and stall PPO. We therefore create a new
    process group (Linux) and kill the whole group on timeout.
    """
    cmd = ["ngspice", "-b", "-o", str(opath), str(cpath)]
    kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "cwd": str(cwd),
    }
    if hasattr(os, "setsid"):
        kwargs["preexec_fn"] = os.setsid  # type: ignore[assignment]
    try:
        p = subprocess.Popen(cmd, **kwargs)
    except FileNotFoundError:
        return "ngspice_not_found"

    try:
        p.communicate(timeout=float(timeout_s))
        return None
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            else:
                p.kill()
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        try:
            p.wait(timeout=5.0)
        except Exception:
            pass
        return "ngspice_timeout"


def eval_one_detail_dcdc(
    inc: str,
    family: str,
    vin: float,
    vout: float,
    tol: float,
    rload: float,
    t_pre: float,
    t_win: float,
    sim_timeout_s: float = 180.0,
    autotune_duty: bool = False,
):
    ver = verify_inc_dcdc(inc, family=family, vin=vin, vout=vout)
    pass_C = bool(ver.ok)
    if int(ver.n_elems) <= 0:
        return {"ok": False, "pass_C": pass_C, "violations": ver.violations, "canonical_hash": ver.canonical_hash}

    t0 = float(t_pre)
    t1 = float(t_pre) + float(t_win)

    def _simulate_once(
        duty_override: Optional[float] = None,
        duty1_override: Optional[float] = None,
        duty2_override: Optional[float] = None,
    ) -> dict:
        try:
            cir, meta = build_dcdc_spice(
                inc,
                family=family,
                vin=vin,
                vout=vout,
                rload=rload,
                duty_override=duty_override,
                duty1_override=duty1_override,
                duty2_override=duty2_override,
            )
        except Exception as e:
            return {
                "ok": False,
                "pass_C": pass_C,
                "violations": ver.violations,
                "canonical_hash": ver.canonical_hash,
                "error": f"build_dcdc_spice_failed:{type(e).__name__}",
            }
        cir2 = inject_measures(cir, t0, t1, vin_node=meta.vin_node)

        tmpdir = tempfile.mkdtemp(prefix="tran_dcdc_")
        try:
            tmp = Path(tmpdir)
            cpath = tmp / "net.cir"
            opath = tmp / "out.log"
            cpath.write_text(cir2, encoding="utf-8")
            err = _run_ngspice(cpath=cpath, opath=opath, cwd=str(tmpdir), timeout_s=float(sim_timeout_s))
            if err:
                out = {
                    "ok": False,
                    "pass_C": pass_C,
                    "violations": ver.violations,
                    "canonical_hash": ver.canonical_hash,
                    "error": str(err),
                }
                if str(err) == "ngspice_timeout":
                    out["duty_map"] = meta.duty_map
                return out

            try:
                text = opath.read_text(errors="replace")
            except Exception:
                return {
                    "ok": False,
                    "pass_C": pass_C,
                    "violations": ver.violations,
                    "canonical_hash": ver.canonical_hash,
                    "duty_map": meta.duty_map,
                    "error": "ngspice_log_read_failed",
                }

            meas = _parse_meas(text)
            need = {"vavg", "vmax", "vmin", "iavg", "vinavg"}
            if not need.issubset(meas.keys()):
                return {
                    "ok": False,
                    "pass_C": pass_C,
                    "violations": ver.violations,
                    "canonical_hash": ver.canonical_hash,
                    "duty_map": meta.duty_map,
                    "error": "missing_measures",
                }

            vavg = float(meas["vavg"])
            vmax = float(meas["vmax"])
            vmin = float(meas["vmin"])
            iavg = float(meas["iavg"])
            vinavg = float(meas["vinavg"])

            ripple = float(vmax - vmin)
            overshoot = float(max(0.0, (vmax - float(vout)) / max(1e-6, float(vout))))
            pass_CV = bool(abs(vavg - float(vout)) / max(1e-6, float(vout)) <= float(tol))

            pin = abs(vinavg * iavg)
            pout = (vavg * vavg) / max(1e-6, float(rload))
            eff = float((pout / pin) if pin > 1e-12 else 0.0)
            if eff > 0.99:
                eff = 0.99
            if eff < 0.0:
                eff = 0.0
            pass_CE = bool(eff >= 0.7)

            return {
                "ok": True,
                "family": str(meta.family),
                "pass_C": pass_C,
                "pass_CV": bool(pass_CV),
                "pass_CE": bool(pass_CE),
                "eff": float(eff),
                "vavg": float(vavg),
                "ripple": float(ripple),
                "overshoot": float(overshoot),
                "canonical_hash": ver.canonical_hash,
                "duty_map": meta.duty_map,
            }
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    out0 = _simulate_once()
    if not out0.get("ok"):
        return out0

    if out0.get("pass_CV", False):
        out0["tuned"] = False
        out0["tune_iters"] = 0
        return out0

    if not bool(autotune_duty):
        out0["tuned"] = False
        out0["tune_iters"] = 0
        return out0

    # Lightweight duty auto-tune: one feedback update if voltage tracking fails.
    fam = (family or "").strip().lower()
    vavg0 = float(out0.get("vavg", 0.0) or 0.0)
    if not (vavg0 > 1e-9):
        out0["tuned"] = False
        out0["tune_iters"] = 0
        return out0

    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    duty_map0 = out0.get("duty_map") or {}
    duty0 = float(duty_map0.get("gate", 0.5))
    duty1 = float(duty_map0.get("gate1", 0.5))
    duty2 = float(duty_map0.get("gate2", 0.5))

    duty_override = None
    duty1_override = None
    duty2_override = None

    if fam == "buck":
        duty_override = _clamp(duty0 * (float(vout) / vavg0), 0.05, 0.95)
    elif fam == "boost":
        duty_override = _clamp(1.0 - (1.0 - duty0) * (vavg0 / max(1e-9, float(vout))), 0.05, 0.95)
    elif fam == "sepic":
        k0 = duty0 / max(1e-9, (1.0 - duty0))
        k1 = k0 * (float(vout) / vavg0)
        duty_override = _clamp(k1 / (1.0 + k1), 0.05, 0.95)
    elif fam in {"buckboost", "buck-boost", "bb"}:
        # Cascaded buck->boost: vout ≈ vin * duty1 / (1-duty2). Adjust both gates softly.
        ratio = float(vout) / max(1e-9, float(vavg0))
        ratio_s = ratio ** 0.5
        duty1_override = _clamp(duty1 * ratio_s, 0.05, 0.95)
        duty2_override = _clamp(1.0 - (1.0 - duty2) / max(1e-6, ratio_s), 0.05, 0.95)
    else:
        out0["tuned"] = False
        out0["tune_iters"] = 0
        return out0

    out1 = _simulate_once(duty_override=duty_override, duty1_override=duty1_override, duty2_override=duty2_override)
    if not out1.get("ok"):
        out0["tuned"] = False
        out0["tune_iters"] = 0
        return out0

    err0 = abs(float(out0.get("vavg", 0.0) or 0.0) - float(vout)) / max(1e-9, float(vout))
    err1 = abs(float(out1.get("vavg", 0.0) or 0.0) - float(vout)) / max(1e-9, float(vout))

    best = out1 if (out1.get("pass_CV", False) or err1 < err0) else out0
    best["tuned"] = True
    best["tune_iters"] = 1
    return best


def eval_one_detail_dcdc_robust(
    inc: str,
    family: str,
    vin: float,
    vout: float,
    tol: float,
    rload: float,
    t_pre: float,
    t_win: float,
    *,
    vin_jitter: float = 0.10,
    rload_list: tuple[float, ...] = (5.0, 10.0, 20.0),
    agg: str = "cvar",
    cvar_alpha: float = 0.25,
    tol_ref: float = 0.1,
    autotune_duty: bool = False,
) -> dict:
    """
    Robust evaluation over a small operating-condition set.

    Aggregation:
      - agg='worst': worst-case score across scenarios
      - agg='cvar' : mean of worst alpha-fraction scores (CVaR proxy)
    """
    vin0 = float(vin)
    vj = float(vin_jitter)
    vin_list = [vin0 * (1.0 - vj), vin0, vin0 * (1.0 + vj)]
    rl_list = [float(x) for x in rload_list]

    scenarios = []
    scores = []
    pass_cv_all = True
    pass_ce_all = True
    ok_all = True

    for vin_i in vin_list:
        for rl in rl_list:
            d = eval_one_detail_dcdc(
                inc=inc,
                family=family,
                vin=float(vin_i),
                vout=float(vout),
                tol=float(tol),
                rload=float(rl),
                t_pre=float(t_pre),
                t_win=float(t_win),
                autotune_duty=bool(autotune_duty),
            )
            s = _score_detail(d, vout=float(vout), tol_ref=float(tol_ref))
            scenarios.append({"vin": float(vin_i), "rload": float(rl), "detail": d, "score": float(s)})
            scores.append(float(s))
            ok_all = ok_all and bool(d.get("ok", False))
            pass_cv_all = pass_cv_all and bool(d.get("pass_CV", False))
            pass_ce_all = pass_ce_all and bool(d.get("pass_CE", False))

    if not scores:
        return {"ok": False, "error": "no_scenarios"}

    agg_mode = str(agg or "cvar").strip().lower()
    if agg_mode == "worst":
        agg_score = float(min(scores))
    else:
        alpha = max(1e-6, min(1.0, float(cvar_alpha)))
        k = max(1, int(round(alpha * len(scores))))
        worst_k = sorted(scores)[:k]
        agg_score = float(sum(worst_k) / max(1, len(worst_k)))

    return {
        "ok": bool(ok_all),
        "pass_CV_all": bool(pass_cv_all),
        "pass_CE_all": bool(pass_ce_all),
        "agg": {"mode": agg_mode, "score": float(agg_score), "n_scenarios": int(len(scores))},
        "scenarios": scenarios,
    }

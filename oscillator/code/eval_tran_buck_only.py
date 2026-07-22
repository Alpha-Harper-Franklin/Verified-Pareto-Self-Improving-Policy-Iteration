from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from spice_builder import build_buck_spice
from verifier import verify_inc

_MEAS_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+0-9.eE]+)")


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


def eval_one_detail_verified(inc: str, vin: float, vout: float, tol: float, rload: float, t_pre: float, t_win: float):
    ver = verify_inc(inc, vin=vin, vout=vout)
    if not ver.ok:
        return {"ok": False, "pass_C": False, "violations": ver.violations, "canonical_hash": ver.canonical_hash}

    cir, meta = build_buck_spice(inc, vin=vin, vout=vout, rload=rload)
    t0 = float(t_pre)
    t1 = float(t_pre) + float(t_win)
    cir2 = inject_measures(cir, t0, t1, vin_node=meta.vin_node)

    tmpdir = tempfile.mkdtemp(prefix="tran_verified_")
    try:
        tmp = Path(tmpdir)
        cpath = tmp / "net.cir"
        opath = tmp / "out.log"
        cpath.write_text(cir2, encoding="utf-8")
        try:
            subprocess.run(
                ["ngspice", "-b", "-o", str(opath), str(cpath)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
                cwd=str(tmpdir),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "pass_C": True, "canonical_hash": ver.canonical_hash}

        try:
            text = opath.read_text(errors="replace")
        except Exception:
            return {"ok": False, "pass_C": True, "canonical_hash": ver.canonical_hash}

        meas = _parse_meas(text)
        need = {"vavg", "vmax", "vmin", "iavg", "vinavg"}
        if not need.issubset(meas.keys()):
            return {"ok": False, "pass_C": True, "canonical_hash": ver.canonical_hash}

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
        pass_CE = bool(eff >= 0.8)

        return {
            "ok": True,
            "pass_C": True,
            "pass_CV": bool(pass_CV),
            "pass_CE": bool(pass_CE),
            "eff": float(eff),
            "vavg": float(vavg),
            "ripple": float(ripple),
            "overshoot": float(overshoot),
            "canonical_hash": ver.canonical_hash,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


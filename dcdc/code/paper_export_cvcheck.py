#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _tag_from_path(p: Path) -> str:
    name = p.stem
    if name.startswith("cv_check_templates_"):
        return name[len("cv_check_templates_") :]
    return name


def _rate(n: int, d: int) -> float:
    return float(n) / float(d) if d > 0 else float("nan")


def _iter_inputs(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            out.append(p)
            continue
        out.extend(sorted(Path(".").glob(raw)))
    # keep deterministic order
    return sorted({p.resolve() for p in out})


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k in seen:
                continue
            seen.add(k)
            fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _heatmap_long(rows: List[dict]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        fam = str(r.get("family", ""))
        vin = _safe_float(r.get("vin"))
        vout = _safe_float(r.get("vout"))
        det = r.get("detail") or {}
        ok = bool(det.get("ok", False))
        vavg = _safe_float(det.get("vavg"))
        err = abs(vavg - vout) / max(1e-9, vout) if ok else float("nan")
        val = (1.0 - err) if ok else float("nan")
        out.setdefault(fam, []).append(
            {
                "family": fam,
                "vin": vin,
                "vout": vout,
                "ok": int(ok),
                "pass_CV": int(bool(det.get("pass_CV", False))) if ok else 0,
                "one_minus_err": val,
                "err_pct": err * 100.0 if ok else float("nan"),
            }
        )
    return out


def summarize_one(tag: str, payload: dict) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary = payload.get("summary") or {}
    rows = list(payload.get("rows") or [])
    tmpl_meta = summary.get("template_meta") or {}

    # per-task rows
    task_rows: List[Dict[str, Any]] = []
    for r in rows:
        fam = str(r.get("family", ""))
        vin = _safe_float(r.get("vin"))
        vout = _safe_float(r.get("vout"))
        det = r.get("detail") or {}
        ok = bool(det.get("ok", False))
        pass_cv = bool(det.get("pass_CV", False)) if ok else False
        pass_ce = bool(det.get("pass_CE", False)) if ok else False
        eff = _safe_float(det.get("eff")) if ok else float("nan")
        vavg = _safe_float(det.get("vavg")) if ok else float("nan")
        ripple = _safe_float(det.get("ripple")) if ok else float("nan")
        overshoot = _safe_float(det.get("overshoot")) if ok else float("nan")
        err = abs(vavg - vout) / max(1e-9, vout) if ok else float("nan")

        duty_map = det.get("duty_map") or {}
        row = {
            "tag": tag,
            "variant": summary.get("variant", ""),
            "family": fam,
            "vin": vin,
            "vout": vout,
            "n_elems": int((tmpl_meta.get(fam) or {}).get("n_elems", 0) or 0),
            "ok": int(ok),
            "pass_CV": int(pass_cv),
            "pass_CE": int(pass_ce),
            "eff": eff,
            "vavg": vavg,
            "err_pct": err * 100.0 if ok else float("nan"),
            "ripple": ripple,
            "ripple_norm": (ripple / max(1e-9, vout)) if ok else float("nan"),
            "overshoot": overshoot,
            "tuned": int(bool(det.get("tuned", False))),
            "tune_iters": int(det.get("tune_iters", 0) or 0),
            "duty_gate": _safe_float(duty_map.get("gate")),
            "duty_gate1": _safe_float(duty_map.get("gate1")),
            "duty_gate2": _safe_float(duty_map.get("gate2")),
            "error": str(det.get("error", "")),
        }
        task_rows.append(row)

    # summary rows (overall + family)
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for tr in task_rows:
        by_family.setdefault(str(tr["family"]), []).append(tr)

    sum_rows: List[Dict[str, Any]] = []

    def _sum_bucket(fam: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(items)
        ok = sum(int(x["ok"]) for x in items)
        cv = sum(int(x["pass_CV"]) for x in items)
        ce = sum(int(x["pass_CE"]) for x in items)
        tuned = sum(int(x["tuned"]) for x in items)
        eff_cv = [float(x["eff"]) for x in items if int(x["ok"]) and int(x["pass_CV"]) and math.isfinite(float(x["eff"]))]
        err_cv = [float(x["err_pct"]) for x in items if int(x["ok"]) and math.isfinite(float(x["err_pct"]))]
        n_elems = max(int(x.get("n_elems", 0) or 0) for x in items) if items else 0
        return {
            "tag": tag,
            "variant": summary.get("variant", ""),
            "family": fam,
            "tasks": n,
            "n_elems": n_elems,
            "ok_rate": _rate(ok, n),
            "cv_rate": _rate(cv, n),
            "ce_rate": _rate(ce, n),
            "tuned_rate": _rate(tuned, n),
            "mean_eff_on_CV": sum(eff_cv) / max(1, len(eff_cv)) if eff_cv else float("nan"),
            "mean_err_pct": sum(err_cv) / max(1, len(err_cv)) if err_cv else float("nan"),
        }

    sum_rows.append(_sum_bucket("ALL", task_rows))
    for fam, items in sorted(by_family.items()):
        sum_rows.append(_sum_bucket(fam, items))

    return task_rows, sum_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more cv_check_templates_*.json files (or globs)")
    ap.add_argument("--outdir", default="paper_assets_cvcheck")
    args = ap.parse_args()

    inputs = _iter_inputs(list(args.inputs))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_task_rows: List[Dict[str, Any]] = []
    all_sum_rows: List[Dict[str, Any]] = []

    for p in inputs:
        payload = json.loads(p.read_text(encoding="utf-8"))
        tag = _tag_from_path(p)
        task_rows, sum_rows = summarize_one(tag, payload)

        _write_csv(outdir / f"task_details_{tag}.csv", task_rows)
        _write_csv(outdir / f"summary_{tag}.csv", sum_rows)

        # heatmap long format (for easy plotting in matplotlib/pgfplots)
        hm = _heatmap_long(payload.get("rows") or [])
        for fam, fam_rows in hm.items():
            _write_csv(outdir / f"heatmap_{tag}_{fam}.csv", fam_rows)

        all_task_rows.extend(task_rows)
        all_sum_rows.extend(sum_rows)

    _write_csv(outdir / "task_details_ALL.csv", all_task_rows)
    _write_csv(outdir / "summary_ALL.csv", all_sum_rows)
    print("[OK] wrote", str(outdir))


if __name__ == "__main__":
    main()


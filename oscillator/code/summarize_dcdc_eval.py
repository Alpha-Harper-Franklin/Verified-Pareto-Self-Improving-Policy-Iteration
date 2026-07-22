#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _score_sample(sample: dict, vout: float, tol_ref: float = 0.1) -> float:
    if not bool(sample.get("ok", False)):
        return -1e9
    vavg = _safe_float(sample.get("vavg", 0.0), 0.0)
    eff = _safe_float(sample.get("eff", 0.0), 0.0)
    ripple = _safe_float(sample.get("ripple", 0.0), 0.0)
    overshoot = _safe_float(sample.get("overshoot", 0.0), 0.0)

    err = abs(vavg - float(vout)) / max(1e-6, float(vout))
    score_v = 3.0 * max(0.0, 1.0 - (err / max(1e-6, float(tol_ref))))
    score_eff = 0.5 * eff
    score_ripple = -0.2 * (ripple / max(1e-6, float(vout)))
    score_over = -0.2 * overshoot
    if not bool(sample.get("pass_CV", False)):
        score_v -= 1.0
    if not bool(sample.get("pass_CE", False)):
        score_eff -= 0.5
    return float(score_v + score_eff + score_ripple + score_over)


def iter_metric_files(eval_dir: Path) -> List[Path]:
    return sorted(eval_dir.glob("**/metric_*_full.json"))


def summarize_eval_dir(eval_dir: Path) -> Dict[str, Any]:
    files = iter_metric_files(eval_dir)
    tasks_total = 0
    tasks_any_ok = 0
    tasks_any_cv = 0
    tasks_any_cvce = 0

    sum_best_eff_cv = 0.0
    sum_best_eff_cvce = 0.0
    sum_best_err_cv = 0.0
    sum_best_ripple_norm_cv = 0.0
    sum_n_unique = 0.0

    n_eff_cv = 0
    n_eff_cvce = 0
    n_err_cv = 0
    n_ripple_cv = 0

    # sample-level stats
    samples_total = 0
    samples_ok = 0
    samples_cv = 0
    samples_cvce = 0

    by_family: Dict[str, Dict[str, int]] = {}

    for p in files:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        fam = str(payload.get("family", ""))
        vin = _safe_float(payload.get("vin"))
        vout = _safe_float(payload.get("vout"))
        samples: List[dict] = list(payload.get("samples") or [])
        if not samples:
            continue

        tasks_total += 1
        sum_n_unique += float(len(samples))

        fam_stat = by_family.setdefault(fam, {"tasks": 0, "tasks_cv": 0, "tasks_cvce": 0})
        fam_stat["tasks"] += 1

        any_ok = any(bool(s.get("ok", False)) for s in samples)
        any_cv = any(bool(s.get("pass_CV", False)) for s in samples)
        any_cvce = any(bool(s.get("pass_CV", False)) and bool(s.get("pass_CE", False)) for s in samples)
        if any_ok:
            tasks_any_ok += 1
        if any_cv:
            tasks_any_cv += 1
            fam_stat["tasks_cv"] += 1
        if any_cvce:
            tasks_any_cvce += 1
            fam_stat["tasks_cvce"] += 1

        for s in samples:
            samples_total += 1
            if bool(s.get("ok", False)):
                samples_ok += 1
            if bool(s.get("pass_CV", False)):
                samples_cv += 1
            if bool(s.get("pass_CV", False)) and bool(s.get("pass_CE", False)):
                samples_cvce += 1

        # best-of-task summaries
        cv_samples = [s for s in samples if bool(s.get("pass_CV", False))]
        cvce_samples = [s for s in samples if bool(s.get("pass_CV", False)) and bool(s.get("pass_CE", False))]

        if cv_samples:
            best_cv = max(cv_samples, key=lambda s: _score_sample(s, vout=vout, tol_ref=0.1))
            best_eff = _safe_float(best_cv.get("eff", 0.0), 0.0)
            best_vavg = _safe_float(best_cv.get("vavg", 0.0), 0.0)
            best_ripple = _safe_float(best_cv.get("ripple", 0.0), 0.0)

            err = abs(best_vavg - float(vout)) / max(1e-6, float(vout))
            sum_best_eff_cv += best_eff
            sum_best_err_cv += err
            sum_best_ripple_norm_cv += best_ripple / max(1e-6, float(vout))
            n_eff_cv += 1
            n_err_cv += 1
            n_ripple_cv += 1

        if cvce_samples:
            best_cvce = max(cvce_samples, key=lambda s: _score_sample(s, vout=vout, tol_ref=0.1))
            best_eff = _safe_float(best_cvce.get("eff", 0.0), 0.0)
            sum_best_eff_cvce += best_eff
            n_eff_cvce += 1

    def _rate(n: int, d: int) -> float:
        return float(n) / float(d) if d > 0 else float("nan")

    out = {
        "eval_dir": str(eval_dir),
        "tasks_total": tasks_total,
        "tasks_any_ok": tasks_any_ok,
        "tasks_any_cv": tasks_any_cv,
        "tasks_any_cvce": tasks_any_cvce,
        "task_ok_rate": _rate(tasks_any_ok, tasks_total),
        "task_cv_rate": _rate(tasks_any_cv, tasks_total),
        "task_cvce_rate": _rate(tasks_any_cvce, tasks_total),
        "mean_n_unique": sum_n_unique / max(1, tasks_total),
        "mean_best_eff_cv": sum_best_eff_cv / max(1, n_eff_cv) if n_eff_cv else float("nan"),
        "mean_best_eff_cvce": sum_best_eff_cvce / max(1, n_eff_cvce) if n_eff_cvce else float("nan"),
        "mean_best_err_cv": sum_best_err_cv / max(1, n_err_cv) if n_err_cv else float("nan"),
        "mean_best_ripple_norm_cv": sum_best_ripple_norm_cv / max(1, n_ripple_cv) if n_ripple_cv else float("nan"),
        "samples_total": samples_total,
        "samples_ok": samples_ok,
        "samples_cv": samples_cv,
        "samples_cvce": samples_cvce,
        "sample_ok_rate": _rate(samples_ok, samples_total),
        "sample_cv_rate": _rate(samples_cv, samples_total),
        "sample_cvce_rate": _rate(samples_cvce, samples_total),
        "by_family": by_family,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--eval_base", default="eval_base")
    ap.add_argument("--eval_sft", default="eval_sft")
    ap.add_argument("--eval_dpo", default="eval_stage_llm")
    args = ap.parse_args()

    rr = Path(args.run_root)
    items: List[Tuple[str, Path]] = [
        ("base", rr / str(args.eval_base)),
        ("sft", rr / str(args.eval_sft)),
        ("dpo", rr / str(args.eval_dpo)),
    ]

    summaries: Dict[str, Any] = {}
    for name, p in items:
        if not p.exists():
            continue
        summaries[name] = summarize_eval_dir(p)

    out_json = rr / "compare_eval_summary.json"
    out_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    # also a compact CSV
    out_csv = rr / "compare_eval_summary.csv"
    header = [
        "model",
        "tasks_total",
        "task_ok_rate",
        "task_cv_rate",
        "task_cvce_rate",
        "mean_best_eff_cv",
        "mean_best_eff_cvce",
        "mean_best_err_cv",
        "mean_best_ripple_norm_cv",
        "samples_total",
        "sample_ok_rate",
        "sample_cv_rate",
        "sample_cvce_rate",
    ]
    lines = [",".join(header)]
    for name in ["base", "sft", "dpo"]:
        s = summaries.get(name)
        if not s:
            continue
        row = [
            name,
            str(s.get("tasks_total", "")),
            str(s.get("task_ok_rate", "")),
            str(s.get("task_cv_rate", "")),
            str(s.get("task_cvce_rate", "")),
            str(s.get("mean_best_eff_cv", "")),
            str(s.get("mean_best_eff_cvce", "")),
            str(s.get("mean_best_err_cv", "")),
            str(s.get("mean_best_ripple_norm_cv", "")),
            str(s.get("samples_total", "")),
            str(s.get("sample_ok_rate", "")),
            str(s.get("sample_cv_rate", "")),
            str(s.get("sample_cvce_rate", "")),
        ]
        lines.append(",".join(row))
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[OK] wrote", str(out_json))
    print("[OK] wrote", str(out_csv))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


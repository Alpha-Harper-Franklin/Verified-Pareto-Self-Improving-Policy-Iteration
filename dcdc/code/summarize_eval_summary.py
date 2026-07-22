#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _rate(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _alias_metric_key(k: str) -> str:
    k = str(k or "").strip()
    if k == "pass_CV":
        return "cv"
    if k == "pass_CE":
        return "ce"
    if k == "ok":
        return "ok"
    return k.lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_summary', required=True, help='Path to eval_summary.json produced by eval_*_family.py')
    ap.add_argument('--out', required=True, help='Path to write metrics json')
    args = ap.parse_args()

    p = Path(str(args.eval_summary))
    if not p.exists():
        raise SystemExit(f'missing eval_summary: {p}')

    obj = json.loads(p.read_text(encoding='utf-8', errors='ignore'))
    results: Dict[str, Any] = obj.get('results') or {}
    if not isinstance(results, dict):
        results = {}

    total_tasks = int(len(results))

    total_tried = 0

    # Dynamic metric aggregation: any integer count in per-task summaries is treated as a "sample-count" metric.
    # We always keep the legacy CV/CE naming for backward compatibility.
    metric_counts: Dict[str, int] = defaultdict(int)
    metric_task_pass: Dict[str, int] = defaultdict(int)

    fam_tot = defaultdict(lambda: {'tasks': 0, 'tried': 0, 'metric_counts': defaultdict(int), 'metric_task_pass': defaultdict(int)})

    for key, summ in results.items():
        fam = str(key).split('/', 1)[0].strip().lower() if isinstance(key, str) else 'unknown'
        if not isinstance(summ, dict):
            continue
        tried = _safe_int(summ.get('n_unique'), 0)
        total_tried += tried

        ft = fam_tot[fam]
        ft['tasks'] += 1
        ft['tried'] += tried

        # Aggregate all integer metrics except n_unique.
        for mk, mv in summ.items():
            if mk == 'n_unique':
                continue
            try:
                iv = int(mv)
            except Exception:
                continue
            metric_counts[mk] += iv
            ft['metric_counts'][mk] += iv
            if iv > 0:
                metric_task_pass[mk] += 1
                ft['metric_task_pass'][mk] += 1

    metrics = {
        'total_tasks': int(total_tasks),
        'total_unique_samples': int(total_tried),
        'by_family': {},
    }

    # Top-level sample/task rates.
    for mk in sorted(metric_counts.keys()):
        alias = _alias_metric_key(mk)
        metrics[f'sample_{alias}_rate'] = _rate(metric_counts[mk], total_tried)
        metrics[f'task_pass_{alias}_rate'] = _rate(metric_task_pass.get(mk, 0), total_tasks)

    for fam, ft in sorted(fam_tot.items()):
        t = int(ft['tasks'])
        tried = int(ft['tried'])
        row: Dict[str, Any] = {
            'tasks': t,
            'unique_samples': tried,
        }
        for mk in sorted(ft['metric_counts'].keys()):
            alias = _alias_metric_key(mk)
            row[f'sample_{alias}_rate'] = _rate(ft['metric_counts'][mk], tried)
            row[f'task_pass_{alias}_rate'] = _rate(ft['metric_task_pass'].get(mk, 0), t)
        metrics['by_family'][fam] = row

    out = Path(str(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')
    print('[OK] wrote', str(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

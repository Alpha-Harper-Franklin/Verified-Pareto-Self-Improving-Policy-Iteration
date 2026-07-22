#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

from inc_parser import parse_numeric


TASK_RE = re.compile(r"vin(?P<vin>[0-9.]+)_vout(?P<vout>[0-9.]+)$")
DEFAULT_OFFICIAL_JSONL = "/root/autodl-tmp/datasets/official_power_v1/train.jsonl"
RESPONSE_TEMPLATE = "### Response:\n"

_OFFICIAL_ELEM_NAME_RE = re.compile(r"^[RLCSD][A-Za-z0-9_]*\d[A-Za-z0-9_]*$")
_OFFICIAL_NODE_RE = re.compile(r"^[A-Za-z0-9_]+$")
_OFFICIAL_MODEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _inc_allows_only_types(inc: str, allow_types: str) -> bool:
    allow = {c.strip().upper() for c in allow_types if c.strip()}
    if not allow:
        return False
    for raw in (inc or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if not line.upper().startswith("INC "):
            return False
        parts = line.split()
        if len(parts) < 2:
            return False
        name = parts[1]
        t = name[:1].upper() if name else ""
        if t not in allow:
            return False
    return True


def _normalize_official_inc(inc: str, allow_types: str) -> str | None:
    allow = {c.strip().upper() for c in allow_types if c.strip()}
    if not allow:
        return None

    out_lines: List[str] = []
    counts: Dict[str, int] = {}
    for raw in (inc or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        if not s.upper().startswith("INC "):
            return None
        parts = s.split()
        if len(parts) != 5:
            return None
        _inc, name, n1, n2, tail = parts

        kind = (name[:1] or "?").upper()
        if kind not in allow:
            return None
        if kind not in {"R", "L", "C", "D", "S"}:
            return None

        if not _OFFICIAL_ELEM_NAME_RE.match(name):
            return None
        if not _OFFICIAL_NODE_RE.match(n1) or not _OFFICIAL_NODE_RE.match(n2):
            return None

        if kind in {"R", "L", "C"}:
            if parse_numeric(tail) is None:
                return None
        else:
            if parse_numeric(tail) is not None:
                return None
            if not _OFFICIAL_MODEL_RE.match(tail):
                return None

        out_lines.append(f"INC {name} {n1} {n2} {tail}")
        counts[kind] = counts.get(kind, 0) + 1

    if not out_lines:
        return None
    return "\n".join(out_lines).strip() + "\n"


def _official_prompt() -> str:
    return (
        "Write a valid circuit in INC DSL.\n"
        "Rules:\n"
        "- Output ONLY INC lines (no explanation).\n"
        "- Line format: INC <name> <node1> <node2> <value_or_model>\n"
        "- Element names must start with R/L/C/D/S and contain a digit.\n"
        + RESPONSE_TEMPLATE
    )


def load_official_sft_texts(
    official_jsonl: Path,
    max_rows: int,
    seed: int,
    allow_types: str,
) -> List[str]:
    if max_rows <= 0:
        return []
    if not official_jsonl.exists():
        return []

    kept: List[str] = []
    prompt = _official_prompt()
    with official_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            inc = (row.get("inc") or "").strip()
            if not inc:
                continue
            if not _inc_allows_only_types(inc, allow_types=allow_types):
                continue
            inc_norm = _normalize_official_inc(inc, allow_types=allow_types)
            if not inc_norm:
                continue
            kept.append(prompt + inc_norm)

    rng = random.Random(int(seed))
    rng.shuffle(kept)
    return kept[: int(max_rows)]


def load_history_candidates(history_path: Path) -> List[dict]:
    if not history_path.exists():
        return []
    out: List[dict] = []
    with history_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            inc = (r.get("inc") or "").strip()
            score = r.get("score", None)
            if not inc:
                continue
            try:
                score_f = float(score)
            except Exception:
                continue
            out.append(
                {
                    "score": score_f,
                    "inc": inc + "\n",
                    "detail": r.get("detail") or {},
                    "step": int(r.get("step", 0) or 0),
                }
            )
    return out


def build_prompt(family: str, vin: float, vout: float) -> str:
    fam = (family or "").strip().lower()
    if fam == "buck":
        return (
            "Generate a Buck DC-DC converter in INC DSL.\n"
            "Rules:\n"
            "- Output ONLY INC lines (no explanation).\n"
            "- Line format: INC <name> <node1> <node2> <value_or_model>\n"
            "- Element names must start with R/L/C/D/S and contain a digit.\n"
            "- Must include required nodes {vin,sw,out,0}. You MAY introduce extra helper nodes.\n"
            "- Use elements {L,C,D,S} with numeric values (e.g., 47u).\n"
            "- Use at least 20 INC lines (>=20 elements).\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )
    if fam == "boost":
        return (
            "Generate a Boost DC-DC converter in INC DSL.\n"
            "Rules:\n"
            "- Output ONLY INC lines (no explanation).\n"
            "- Line format: INC <name> <node1> <node2> <value_or_model>\n"
            "- Element names must start with R/L/C/D/S and contain a digit.\n"
            "- Must include required nodes {vin,sw,out,0}. You MAY introduce extra helper nodes.\n"
            "- Typical topology: L vin-sw, S sw-0, D sw-out, C out-0.\n"
            "- Use at least 20 INC lines (>=20 elements).\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )
    if fam == "sepic":
        return (
            "Generate a SEPIC DC-DC converter in INC DSL.\n"
            "Rules:\n"
            "- Output ONLY INC lines (no explanation).\n"
            "- Line format: INC <name> <node1> <node2> <value_or_model>\n"
            "- Element names must start with R/L/C/D/S and contain a digit.\n"
            "- Must include required nodes {vin,sw,n1,out,0}. You MAY introduce extra helper nodes.\n"
            "- Use 2 inductors and 2 capacitors: L1 vin-sw, C1 sw-n1, L2 n1-0, S1 sw-0, D1 n1-out, C2 out-0.\n"
            "- Use at least 20 INC lines (>=20 elements).\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )
    if fam == "buckboost":
        return (
            "Generate a non-inverting Buck-Boost (cascaded buck->boost) DC-DC converter in INC DSL.\n"
            "Rules:\n"
            "- Output ONLY INC lines (no explanation).\n"
            "- Line format: INC <name> <node1> <node2> <value_or_model>\n"
            "- Element names must start with R/L/C/D/S and contain a digit.\n"
            "- Must include required nodes {vin,sw1,mid,sw2,out,0}. You MAY introduce extra helper nodes.\n"
            "- Use 2 switches and 2 diodes. Use switch models Sstd1 and Sstd2 so gate1/gate2 are separate.\n"
            "- Use at least 20 INC lines (>=20 elements).\n"
            f"Task: Vin={vin:.1f}V, Vout={vout:.1f}V, Rload=10ohm.\n"
            + RESPONSE_TEMPLATE
        )
    raise ValueError(f"unknown family: {family}")


def iter_task_dirs(tasks_root: Path) -> List[Tuple[str, float, float, Path]]:
    out: List[Tuple[str, float, float, Path]] = []
    for fam_dir in sorted(tasks_root.iterdir()):
        if not fam_dir.is_dir():
            continue
        fam = fam_dir.name
        for td in sorted(fam_dir.iterdir()):
            if not td.is_dir():
                continue
            m = TASK_RE.search(td.name)
            if not m:
                continue
            vin = float(m.group('vin'))
            vout = float(m.group('vout'))
            out.append((fam, vin, vout, td))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_root', required=True)
    ap.add_argument('--min_improve', type=float, default=0.2)
    ap.add_argument('--require_pass_cv', action='store_true')
    ap.add_argument('--max_pairs_per_task', type=int, default=8)
    ap.add_argument('--sft_topk_per_task', type=int, default=3)
    ap.add_argument('--official_jsonl', default=DEFAULT_OFFICIAL_JSONL)
    ap.add_argument('--official_max', type=int, default=128)
    ap.add_argument('--official_seed', type=int, default=2025)
    ap.add_argument('--official_allow_types', default='RLCSD')
    ap.add_argument('--no_official', action='store_true')
    args = ap.parse_args()

    run_root = Path(args.run_root)
    tasks_root = run_root / 'tasks'
    if not tasks_root.is_dir():
        raise SystemExit(f"missing tasks dir: {tasks_root}")

    out_pairs = run_root / 'dpo_pairs.jsonl'
    out_sft = run_root / 'sft_train.jsonl'
    out_meta = run_root / 'pairs_meta.jsonl'
    out_report = run_root / 'pairs_build_report.csv'

    rows = iter_task_dirs(tasks_root)

    n_pairs = 0
    fam_count: Dict[str, int] = {}
    n_sft_task = 0
    n_sft_official = 0

    with out_pairs.open('w', encoding='utf-8') as f_pairs, out_sft.open('w', encoding='utf-8') as f_sft, out_meta.open(
        'w', encoding='utf-8'
    ) as f_meta, out_report.open('w', newline='', encoding='utf-8') as f_csv:
        w = csv.DictWriter(
            f_csv,
            fieldnames=[
                'family',
                'vin',
                'vout',
                'best_score',
                'worst_score',
                'best_pass_cv',
                'best_pass_ce',
                'best_vavg',
                'best_eff',
                'best_ripple',
                'pairs_task',
                'task_dir',
            ],
        )
        w.writeheader()

        for fam, vin, vout, td in rows:
            seed_path = td / 'seed.inc'
            summary_path = td / 'opt' / 'summary.json'
            best_inc_path = td / 'opt' / 'best_inc.txt'
            history_path = td / 'opt' / 'history.jsonl'

            if not (seed_path.exists() and summary_path.exists() and best_inc_path.exists()):
                continue

            try:
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
            except Exception:
                continue

            best_score = float(summary.get('best_score', -1e9))
            best_detail = summary.get('best_detail', {}) or {}
            best_pass_cv = bool(best_detail.get('pass_CV', False))
            best_pass_ce = bool(best_detail.get('pass_CE', False))
            best_vavg = float(best_detail.get('vavg', 0.0) or 0.0) if best_detail else 0.0
            best_eff = float(best_detail.get('eff', 0.0) or 0.0) if best_detail else 0.0
            best_ripple = float(best_detail.get('ripple', 0.0) or 0.0) if best_detail else 0.0

            if bool(args.require_pass_cv) and not best_pass_cv:
                continue

            cands = load_history_candidates(history_path)
            cands_desc = sorted(cands, key=lambda x: float(x.get("score", -1e9)), reverse=True)
            cands_sorted = sorted(cands, key=lambda x: float(x.get("score", -1e9)))
            worst_score = float(cands_sorted[0]["score"]) if cands_sorted else float("nan")

            prompt = build_prompt(fam, vin, vout)
            chosen = best_inc_path.read_text(encoding='utf-8').strip() + '\n'

            sft_incs: List[str] = [chosen]
            if int(args.sft_topk_per_task) > 1:
                for cand in cands_desc:
                    if len(sft_incs) >= int(args.sft_topk_per_task):
                        break
                    inc = str(cand.get("inc", "")).strip() + "\n"
                    if not inc.strip():
                        continue
                    if inc.strip() == chosen.strip():
                        continue
                    if bool(args.require_pass_cv) and not bool((cand.get("detail") or {}).get("pass_CV", False)):
                        continue
                    sft_incs.append(inc)

            for inc in sft_incs:
                f_sft.write(json.dumps({'text': prompt + inc}, ensure_ascii=False) + '\n')
                n_sft_task += 1

            pairs_task = 0
            for cand in cands_sorted:
                if pairs_task >= int(args.max_pairs_per_task):
                    break
                rejected = str(cand.get("inc", "")).strip() + "\n"
                if not rejected.strip():
                    continue
                if rejected.strip() == chosen.strip():
                    continue
                cand_score = float(cand.get("score", -1e9))
                if (best_score - cand_score) < float(args.min_improve):
                    continue
                f_pairs.write(
                    json.dumps({'prompt': prompt, 'chosen': chosen, 'rejected': rejected}, ensure_ascii=False) + '\n'
                )
                f_meta.write(
                    json.dumps(
                        {
                            'task': {'family': fam, 'vin': vin, 'vout': vout},
                            'chosen': {'score': best_score, 'detail': best_detail},
                            'rejected': {'score': cand_score, 'detail': cand.get('detail') or {}},
                            'task_dir': str(td),
                            'history_step': int(cand.get('step', 0) or 0),
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )
                pairs_task += 1
                n_pairs += 1

            if pairs_task > 0:
                fam_count[fam] = fam_count.get(fam, 0) + pairs_task

            w.writerow(
                {
                    'family': fam,
                    'vin': vin,
                    'vout': vout,
                    'best_score': best_score,
                    'worst_score': worst_score,
                    'best_pass_cv': best_pass_cv,
                    'best_pass_ce': best_pass_ce,
                    'best_vavg': best_vavg,
                    'best_eff': best_eff,
                    'best_ripple': best_ripple,
                    'pairs_task': pairs_task,
                    'task_dir': str(td),
                }
            )

        if not bool(args.no_official):
            official_path = Path(str(args.official_jsonl))
            official_texts = load_official_sft_texts(
                official_jsonl=official_path,
                max_rows=int(args.official_max),
                seed=int(args.official_seed),
                allow_types=str(args.official_allow_types),
            )
            for t in official_texts:
                f_sft.write(json.dumps({'text': t}, ensure_ascii=False) + '\n')
                n_sft_official += 1

    (run_root / 'pairs_stats.json').write_text(
        json.dumps(
            {
                'pairs': n_pairs,
                'by_family': fam_count,
                'sft_task': n_sft_task,
                'sft_official': n_sft_official,
                'sft_topk_per_task': int(args.sft_topk_per_task),
                'official_jsonl': None if bool(args.no_official) else str(args.official_jsonl),
                'official_allow_types': str(args.official_allow_types),
                'official_max': int(args.official_max),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    print('[OK] pairs', n_pairs)


if __name__ == '__main__':
    main()

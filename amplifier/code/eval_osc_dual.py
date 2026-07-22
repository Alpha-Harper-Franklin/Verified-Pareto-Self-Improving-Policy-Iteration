#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dcdc_taskset import Task
from inc_parser import extract_inc_lines
from osc_eval_tran import eval_one_detail_osc
from osc_taskset import default_taskset_osc

_RESP_KEY = "### Response:"


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_response(text: str) -> str:
    s = text or ""
    if _RESP_KEY in s:
        s = s.split(_RESP_KEY, 1)[-1]
    return s


def _canon_unique(inc_text: str) -> str:
    lines = [ln.strip() for ln in extract_inc_lines(inc_text) if ln.strip()]
    return "\n".join(sorted(lines))


def _sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


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


def build_prompt(family: str, vin: float, vout: float, *, min_elems: int) -> str:
    from eval_osc_family import build_prompt as _bp  # type: ignore

    return str(_bp(str(family), float(vin), float(vout), min_elems=int(min_elems)))


@dataclass
class SampleRec:
    task: Task
    seed: int
    raw: str
    canon: str
    canon_hash: str


def _eval_one_osc(task: Task, raw: str, *, sim_timeout_s: float, min_elems: int) -> Dict[str, Any]:
    return eval_one_detail_osc(
        raw,
        family=str(task.family),
        vin=float(task.vin),
        vout=float(task.vout),
        sim_timeout_s=float(sim_timeout_s),
        min_elems=int(min_elems),
    )


def _maybe_repair_osc(
    task: Task,
    raw: str,
    *,
    sim_timeout_s: float,
    min_elems: int,
    repair_factors: List[float],
    repair_max_evals: int,
    repair_only_on_fail: bool,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    from build_selfplay_osc_datasets import _eda_repair  # type: ignore

    base_detail = _eval_one_osc(task, raw, sim_timeout_s=sim_timeout_s, min_elems=min_elems)
    need = not bool(base_detail.get("pass_CV"))
    if bool(repair_only_on_fail):
        need = not bool(base_detail.get("ok"))
    if not need:
        return raw, base_detail, {"used": False}

    inc2, det2, meta = _eda_repair(
        raw,
        family=str(task.family),
        vin=float(task.vin),
        vout=float(task.vout),
        sim_timeout_s=float(sim_timeout_s),
        min_elems=int(min_elems),
        factors=list(repair_factors),
        max_evals=int(repair_max_evals),
    )
    meta = dict(meta or {})
    meta["used"] = bool(meta.get("changed"))
    return str(inc2), dict(det2 or {}), meta


def _rate(samples: List[Dict[str, Any]], key: str) -> float:
    if not samples:
        return 0.0
    return float(sum(1 for d in samples if bool(d.get(key))) / float(len(samples)))


def _pass_at_k(details_by_task: Dict[str, List[Dict[str, Any]]], key: str) -> float:
    if not details_by_task:
        return 0.0
    ok = 0
    tot = 0
    for _, ds in details_by_task.items():
        tot += 1
        if any(bool(d.get(key)) for d in ds):
            ok += 1
    return float(ok / float(tot)) if tot > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--n_per_task", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--sim_timeout_s", type=float, default=60.0)
    ap.add_argument("--min_elems", type=int, default=15)
    ap.add_argument("--sim_workers", type=int, default=64)

    ap.add_argument("--with_repair", action="store_true")
    ap.add_argument("--repair_only_on_fail", action="store_true")
    ap.add_argument("--repair_factors", default="0.2,0.5,0.8,1.0,1.25,1.5,2.0,3.0,5.0")
    ap.add_argument("--repair_max_evals", type=int, default=12)
    ap.add_argument("--only_task", action="append", default=[], help="<family>,<vin>,<vout>")
    args = ap.parse_args()
    # VP-SPI min_elems relax: amplifier/oscillator branches use a lighter complexity constraint.
    _min_raw = int(getattr(args, "min_elems", 15) or 15)
    if _min_raw > 15:
        print(f"[min_elems] clamp {_min_raw} -> 15 (amp/osc)", flush=True)
        args.min_elems = 15

    outdir = Path(args.outdir).resolve()
    out_open = outdir / "openloop"
    out_rep = outdir / "repaired"
    out_open.mkdir(parents=True, exist_ok=True)
    out_rep.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))

    tasks = default_taskset_osc()
    only: List[Task] = []
    for s in (args.only_task or []):
        try:
            fam, a, b = [p.strip() for p in str(s).split(",")]
            only.append(Task(str(fam).lower(), float(a), float(b)))
        except Exception:
            continue
    if only:
        tasks = only

    tok, model = load_model(args.base_model, str(args.adapter or "").strip() or None)
    dev = _device(model)

    gen: Dict[str, Dict[str, Any]] = {}
    for ti, t in enumerate(tasks):
        prompt = build_prompt(t.family, t.vin, t.vout, min_elems=int(args.min_elems))
        task_key = f"{t.family.lower()}/vin{float(t.vin):.6g}_vout{float(t.vout):.2f}"
        task_dir = outdir / "tasks" / t.family.lower() / f"vin{float(t.vin):.6g}_vout{float(t.vout):.2f}"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        uniq: Dict[str, SampleRec] = {}
        raw_list: List[SampleRec] = []
        hash_counts: Dict[str, int] = {}
        for si in range(int(args.n_per_task)):
            seed = int(args.seed) + ti * 1000 + si * 17 + rng.randint(0, 9999)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            inp = tok(prompt, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    do_sample=True,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_new_tokens=int(args.max_new_tokens),
                    pad_token_id=tok.eos_token_id,
                )
            txt = tok.decode(out[0], skip_special_tokens=True)
            resp = _extract_response(txt)
            can = _canon_unique(resp)
            if not can.strip():
                continue
            h = _sha1_text(can)
            hash_counts[h] = int(hash_counts.get(h, 0)) + 1
            rec = SampleRec(task=t, seed=seed, raw=resp, canon=can, canon_hash=h)
            raw_list.append(rec)
            if can in uniq:
                continue
            uniq[can] = rec

        gen[task_key] = {
            "family": t.family.lower(),
            "vin": float(t.vin),
            "vout": float(t.vout),
            "n_gen": int(args.n_per_task),
            "n_raw_nonempty": int(len(raw_list)),
            "n_unique": int(len(uniq)),
            "hash_counts": dict(hash_counts),
            "samples": [
                {"seed": int(r.seed), "raw": r.raw, "canon": r.canon, "hash": r.canon_hash} for r in uniq.values()
            ],
        }
        (task_dir / "generated.json").write_text(json.dumps(gen[task_key], ensure_ascii=False, indent=2), encoding="utf-8")

    open_details_by_task: Dict[str, List[Dict[str, Any]]] = {}
    jobs: List[Tuple[str, SampleRec]] = []
    for k, obj in gen.items():
        for s in obj["samples"]:
            jobs.append(
                (
                    k,
                    SampleRec(
                        task=Task(obj["family"], obj["vin"], obj["vout"]),
                        seed=int(s["seed"]),
                        raw=str(s["raw"]),
                        canon=str(s["canon"]),
                        canon_hash=str(s.get("hash") or ""),
                    ),
                )
            )

    def _job_open(j: Tuple[str, SampleRec]) -> Tuple[str, Dict[str, Any]]:
        key, rec = j
        d = _eval_one_osc(rec.task, rec.raw, sim_timeout_s=float(args.sim_timeout_s), min_elems=int(args.min_elems))
        return key, {"seed": int(rec.seed), "hash": rec.canon_hash, "canon": rec.canon, "raw": rec.raw, "detail": d}

    max_workers = max(1, int(args.sim_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_job_open, j) for j in jobs]
        for fu in as_completed(futs):
            key, out = fu.result()
            open_details_by_task.setdefault(key, []).append(out)

    open_summary: Dict[str, Any] = {"started_at": _now(), "results": {}}
    for key, items in open_details_by_task.items():
        tdir = outdir / "tasks" / key.split("/")[0] / key.split("/")[1]
        payload = {
            "task": gen[key],
            "details": items,
            "summary": {
                "n_unique": int(len(items)),
                "ok": int(sum(1 for x in items if bool((x.get("detail") or {}).get("ok")))),
                "pass_CV": int(sum(1 for x in items if bool((x.get("detail") or {}).get("pass_CV")))),
                "pass_CE": int(sum(1 for x in items if bool((x.get("detail") or {}).get("pass_CE")))),
            },
        }
        (tdir / "eval_openloop.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        open_summary["results"][key] = payload["summary"]

    all_open = [x["detail"] for xs in open_details_by_task.values() for x in xs if isinstance(x, dict)]
    raw_total = 0
    raw_ok = 0
    raw_cv = 0
    raw_ce = 0
    for key, items in open_details_by_task.items():
        counts = gen.get(key, {}).get("hash_counts") or {}
        for it in items:
            h = str(it.get("hash") or "")
            w = int(counts.get(h, 1))
            raw_total += w
            d = it.get("detail") or {}
            if bool(d.get("ok")):
                raw_ok += w
            if bool(d.get("pass_CV")):
                raw_cv += w
            if bool(d.get("pass_CE")):
                raw_ce += w
    open_summary["agg"] = {
        "n_tasks": int(len(open_details_by_task)),
        "n_unique": int(len(all_open)),
        "ok_rate": _rate(all_open, "ok"),
        "cv_rate": _rate(all_open, "pass_CV"),
        "ce_rate": _rate(all_open, "pass_CE"),
        "pass@k_cv": _pass_at_k({k: [x["detail"] for x in xs] for k, xs in open_details_by_task.items()}, "pass_CV"),
        "n_raw_nonempty": int(raw_total),
        "ok_rate_raw": float(raw_ok / raw_total) if raw_total > 0 else 0.0,
        "cv_rate_raw": float(raw_cv / raw_total) if raw_total > 0 else 0.0,
        "ce_rate_raw": float(raw_ce / raw_total) if raw_total > 0 else 0.0,
    }
    open_summary["finished_at"] = _now()
    (out_open / "eval_summary_openloop.json").write_text(json.dumps(open_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rep_summary: Dict[str, Any] = {"skipped": True}
    if bool(args.with_repair):
        rf: List[float] = []
        for p in str(args.repair_factors).split(","):
            p = p.strip()
            if not p:
                continue
            try:
                rf.append(float(p))
            except Exception:
                continue
        if not rf:
            rf = [0.8, 1.0, 1.25]

        def _job_rep(item: Tuple[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
            key, rec = item
            t = Task(gen[key]["family"], float(gen[key]["vin"]), float(gen[key]["vout"]))
            inc2, det2, meta = _maybe_repair_osc(
                t,
                str(rec["raw"]),
                sim_timeout_s=float(args.sim_timeout_s),
                min_elems=int(args.min_elems),
                repair_factors=rf,
                repair_max_evals=int(args.repair_max_evals),
                repair_only_on_fail=bool(args.repair_only_on_fail),
            )
            return key, {
                "seed": int(rec["seed"]),
                "hash": str(rec.get("hash") or ""),
                "canon": str(rec["canon"]),
                "raw": str(rec["raw"]),
                "repaired_inc": str(inc2),
                "repair": meta,
                "detail": det2,
            }

        rep_workers = max(1, min(int(max_workers), 24))
        rep_details_by_task: Dict[str, List[Dict[str, Any]]] = {}
        base_items: List[Tuple[str, Dict[str, Any]]] = []
        for k, xs in open_details_by_task.items():
            for x in xs:
                base_items.append((k, x))

        with ThreadPoolExecutor(max_workers=rep_workers) as ex:
            futs = [ex.submit(_job_rep, it) for it in base_items]
            for fu in as_completed(futs):
                key, out = fu.result()
                rep_details_by_task.setdefault(key, []).append(out)

        rep_summary = {"started_at": _now(), "results": {}}
        for key, items in rep_details_by_task.items():
            tdir = outdir / "tasks" / key.split("/")[0] / key.split("/")[1]
            payload = {
                "task": gen[key],
                "details": items,
                "summary": {
                    "n_unique": int(len(items)),
                    "ok": int(sum(1 for x in items if bool((x.get("detail") or {}).get("ok")))),
                    "pass_CV": int(sum(1 for x in items if bool((x.get("detail") or {}).get("pass_CV")))),
                    "pass_CE": int(sum(1 for x in items if bool((x.get("detail") or {}).get("pass_CE")))),
                    "repair_used": int(sum(1 for x in items if bool((x.get("repair") or {}).get("used")))),
                },
            }
            (tdir / "eval_repaired.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            rep_summary["results"][key] = payload["summary"]

        all_rep = [x["detail"] for xs in rep_details_by_task.values() for x in xs if isinstance(x, dict)]
        rep_raw_total = 0
        rep_raw_ok = 0
        rep_raw_cv = 0
        rep_raw_ce = 0
        rep_raw_used = 0
        for key, items in rep_details_by_task.items():
            counts = gen.get(key, {}).get("hash_counts") or {}
            for it in items:
                h = str(it.get("hash") or "")
                w = int(counts.get(h, 1))
                rep_raw_total += w
                d = it.get("detail") or {}
                if bool(d.get("ok")):
                    rep_raw_ok += w
                if bool(d.get("pass_CV")):
                    rep_raw_cv += w
                if bool(d.get("pass_CE")):
                    rep_raw_ce += w
                if bool((it.get("repair") or {}).get("used")):
                    rep_raw_used += w
        rep_summary["agg"] = {
            "n_tasks": int(len(rep_details_by_task)),
            "n_unique": int(len(all_rep)),
            "ok_rate": _rate(all_rep, "ok"),
            "cv_rate": _rate(all_rep, "pass_CV"),
            "ce_rate": _rate(all_rep, "pass_CE"),
            "pass@k_cv": _pass_at_k({k: [x["detail"] for x in xs] for k, xs in rep_details_by_task.items()}, "pass_CV"),
            "repair_used_rate": float(
                sum(1 for xs in rep_details_by_task.values() for x in xs if bool((x.get("repair") or {}).get("used")))
                / max(1, len(all_rep))
            ),
            "n_raw_nonempty": int(rep_raw_total),
            "ok_rate_raw": float(rep_raw_ok / rep_raw_total) if rep_raw_total > 0 else 0.0,
            "cv_rate_raw": float(rep_raw_cv / rep_raw_total) if rep_raw_total > 0 else 0.0,
            "ce_rate_raw": float(rep_raw_ce / rep_raw_total) if rep_raw_total > 0 else 0.0,
            "repair_used_rate_raw": float(rep_raw_used / rep_raw_total) if rep_raw_total > 0 else 0.0,
        }
        rep_summary["finished_at"] = _now()
        (out_rep / "eval_summary_repaired.json").write_text(json.dumps(rep_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    combined = {
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "outdir": str(outdir),
        "openloop": str(out_open / "eval_summary_openloop.json"),
        "repaired": str(out_rep / "eval_summary_repaired.json") if bool(args.with_repair) else "",
        "openloop_agg": open_summary.get("agg"),
        "repaired_agg": rep_summary.get("agg") if isinstance(rep_summary, dict) else None,
    }
    (outdir / "eval_dual_overview.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] saved", str(outdir))
    print(json.dumps(combined.get("openloop_agg") or {}, ensure_ascii=False))
    if bool(args.with_repair):
        print(json.dumps((combined.get("repaired_agg") or {}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

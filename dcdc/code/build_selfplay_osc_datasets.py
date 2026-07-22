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

from dcdc_taskset import Task
from inc_parser import extract_inc_lines, parse_numeric
from osc_taskset import default_taskset_osc
from osc_eval_tran import OSC_F_TOL_REL, OSC_MIN_VPP, OSC_PSTATIC_MAX_MW, eval_one_detail_osc

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
    from eval_osc_family import build_prompt  # type: ignore

    return str(build_prompt(str(family), float(vin), float(vout), min_elems=int(min_elems)))


def _scale_reactives(inc_text: str, factor: float) -> str:
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0:
        return inc_text
    lines = extract_inc_lines(inc_text)
    out: List[str] = []
    for raw in lines:
        toks = raw.split()
        if len(toks) != 5 or toks[0].upper() != "INC":
            continue
        name, n1, n2, tail = toks[1], toks[2], toks[3], toks[4]
        kind = (name[:1] or "?").upper()
        if kind in {"C", "L"}:
            v = parse_numeric(tail)
            if v is not None:
                tail = f"{float(v) * float(factor):.6g}"
        out.append(" ".join(["INC", name, n1, n2, tail]))
    return ("\n".join(out).strip() + "\n") if out else (inc_text or "")


def _score_osc(detail: Dict[str, Any], *, family: str, vin: float, vout: float) -> float:
    if not bool(detail.get("ok", False)):
        return -1.0

    pass_cv = bool(detail.get("pass_CV", False))
    pass_ce = bool(detail.get("pass_CE", False))

    score = 0.0
    score += 5.0 if pass_cv else 0.0
    score += 1.0 if pass_ce else -1.0

    try:
        f_t = float(vin)
        f = float(detail.get("freq_hz") or 0.0)
        vpp = float(detail.get("vpp") or 0.0)
        p = float(detail.get("pstatic_mw") or 0.0)

        f_err = abs(f - f_t) / max(1e-9, f_t)
        f_term = max(-1.0, 1.0 - f_err / max(1e-6, float(OSC_F_TOL_REL)))
        v_term = max(-1.0, min(1.0, (vpp - float(OSC_MIN_VPP)) / max(1e-6, float(OSC_MIN_VPP))))
        p_term = max(-1.0, min(1.0, (float(OSC_PSTATIC_MAX_MW) - p) / max(1e-6, float(OSC_PSTATIC_MAX_MW))))
        score += 1.5 * float(f_term) + 0.5 * float(v_term) + 0.5 * float(p_term)
    except Exception:
        pass

    return float(score)


def _eval_one(
    inc_text: str,
    *,
    family: str,
    vin: float,
    vout: float,
    sim_timeout_s: float,
    min_elems: int,
) -> Dict[str, Any]:
    t0 = time.time()
    detail = eval_one_detail_osc(
        inc_text,
        family=str(family),
        vin=float(vin),
        vout=float(vout),
        sim_timeout_s=float(sim_timeout_s),
        min_elems=int(min_elems),
    )
    dt = float(time.time() - t0)
    score = _score_osc(detail, family=str(family), vin=float(vin), vout=float(vout))
    return {"detail": detail, "score": float(score), "sim_time": float(dt)}


def _eda_repair(
    inc_text: str,
    *,
    family: str,
    vin: float,
    vout: float,
    sim_timeout_s: float,
    min_elems: int,
    factors: List[float],
    max_evals: int,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    tried = 0

    base_res = _eval_one(
        inc_text,
        family=family,
        vin=vin,
        vout=vout,
        sim_timeout_s=sim_timeout_s,
        min_elems=min_elems,
    )
    tried += 1
    best_inc = inc_text
    best_detail = dict(base_res.get("detail") or {})
    best_score = float(base_res.get("score") or -1e9)
    best_factor = 1.0

    for fac in factors:
        if tried >= int(max_evals):
            break
        fac = float(fac)
        if not math.isfinite(fac) or fac <= 0:
            continue
        if abs(fac - 1.0) < 1e-12:
            continue
        cand = _scale_reactives(inc_text, fac)
        res = _eval_one(
            cand,
            family=family,
            vin=vin,
            vout=vout,
            sim_timeout_s=sim_timeout_s,
            min_elems=min_elems,
        )
        tried += 1
        sc = float(res.get("score") or -1e9)
        if sc > best_score:
            best_score = sc
            best_inc = cand
            best_detail = dict(res.get("detail") or {})
            best_factor = fac

    meta = {
        "used": bool(best_factor != 1.0),
        "tried": int(tried),
        "best_factor": float(best_factor),
        "best_score": float(best_score),
    }
    return str(best_inc), best_detail, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=20250105)

    ap.add_argument("--n_gen", type=int, default=16)
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--temp_step", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--min_elems", type=int, default=15)
    ap.add_argument("--pairs_per_task", type=int, default=64)
    ap.add_argument("--min_pair_gap", type=float, default=0.1)
    ap.add_argument("--top_k_chosen", type=int, default=8)
    ap.add_argument("--sft_topn_per_task", type=int, default=8)
    ap.add_argument("--sft_strict_topn_per_task", type=int, default=0)

    ap.add_argument("--ensure_pass_cv", action="store_true", default=True)
    ap.add_argument("--min_pass_cv", type=int, default=1)

    ap.add_argument("--sim_timeout_s", type=float, default=60.0)
    ap.add_argument("--sim_workers", type=int, default=0)

    ap.add_argument("--eda_repair", action="store_true", default=True)
    ap.add_argument("--repair_factors", default="0.5,0.8,1.0,1.25,1.5,2.0")
    ap.add_argument("--repair_max_evals", type=int, default=12)

    ap.add_argument("--task_shard_id", type=int, default=0)
    ap.add_argument("--task_shard_count", type=int, default=1)
    ap.add_argument("--skip_done_tasks_jsonl", default="")
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

    sim_workers = max(1, int(_auto_sim_workers(int(getattr(args, "sim_workers", 0) or 0))))

    shard_count = max(1, int(getattr(args, "task_shard_count", 1) or 1))
    shard_id = int(getattr(args, "task_shard_id", 0) or 0)
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

    done: set[tuple[str, float, float]] = set()
    if bool(args.resume) and done_path.exists():
        for line in done_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((str(r["family"]).lower(), float(r["vin"]), float(r["vout"])))
            except Exception:
                continue

    if str(getattr(args, "skip_done_tasks_jsonl", "") or "").strip():
        p = Path(str(args.skip_done_tasks_jsonl).strip())
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((str(r["family"]).lower(), float(r["vin"]), float(r["vout"])))
                except Exception:
                    continue

    tok, model = _load_model(str(args.base_model), str(args.adapter or ""))
    dev = _device(model)

    tasks_all = default_taskset_osc()
    tasks: List[Task] = []
    for i, t in enumerate(tasks_all):
        if (i % shard_count) != shard_id:
            continue
        tasks.append(t)

    factors: List[float] = []
    for x in str(args.repair_factors).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            factors.append(float(x))
        except Exception:
            continue
    if not factors:
        factors = [1.0]

    rng = random.Random(int(args.seed) + shard_id * 1000)

    f_pairs = pairs_path.open("a", encoding="utf-8")
    f_meta = meta_path.open("a", encoding="utf-8")
    f_sft = sft_path.open("a", encoding="utf-8")
    f_sft_strict = sft_strict_path.open("a", encoding="utf-8")
    f_done = done_path.open("a", encoding="utf-8")

    total_pairs = 0
    total_sft = 0
    total_sft_strict = 0
    total_tasks_done = 0
    total_tasks_skipped = 0

    executor: Optional[ThreadPoolExecutor] = None
    if sim_workers > 1:
        executor = ThreadPoolExecutor(max_workers=int(sim_workers))

    try:
        for ti, task in enumerate(tasks):
            fam = str(task.family).lower()
            vin = float(task.vin)
            vout = float(task.vout)
            key = (fam, vin, vout)
            if key in done:
                total_tasks_skipped += 1
                continue

            task_dir = out_root / "tasks" / fam / f"vin{vin:.1f}_vout{vout:.1f}"
            task_dir.mkdir(parents=True, exist_ok=True)

            prompt = _build_prompt(fam, vin, vout, min_elems=int(args.min_elems))
            (task_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            best_by_hash: Dict[str, Dict[str, Any]] = {}
            pass_cv_hits = 0
            for ridx in range(int(args.max_rounds)):
                if (not bool(args.ensure_pass_cv)) or (pass_cv_hits >= int(args.min_pass_cv)):
                    break

                cand_incs: List[str] = []
                for gi in range(int(args.n_gen)):
                    seed = int(args.seed) + ti * 1000 + ridx * 100 + gi * 7 + rng.randint(0, 9999)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                    inp = tok(prompt, return_tensors="pt").to(dev)
                    with torch.no_grad():
                        out = model.generate(
                            **inp,
                            do_sample=True,
                            temperature=float(args.temperature) + float(args.temp_step) * float(ridx),
                            top_p=float(args.top_p),
                            max_new_tokens=int(args.max_new_tokens),
                            pad_token_id=tok.eos_token_id,
                        )
                    txt = tok.decode(out[0], skip_special_tokens=True)
                    resp = _normalize_model_text(txt)
                    can = _canon_inc_text(resp)
                    if not can.strip():
                        continue
                    cand_incs.append(resp)

                def _eval_candidate(inc0: str) -> Tuple[Dict[str, Any], str]:
                    t0 = time.time()
                    inc0 = str(inc0)
                    if bool(args.eda_repair):
                        inc_fin, detail, repair_meta = _eda_repair(
                            inc0,
                            family=fam,
                            vin=vin,
                            vout=vout,
                            sim_timeout_s=float(args.sim_timeout_s),
                            min_elems=int(args.min_elems),
                            factors=factors,
                            max_evals=int(args.repair_max_evals),
                        )
                    else:
                        detail = eval_one_detail_osc(
                            inc0,
                            family=fam,
                            vin=vin,
                            vout=vout,
                            sim_timeout_s=float(args.sim_timeout_s),
                            min_elems=int(args.min_elems),
                        )
                        inc_fin = inc0
                        repair_meta = {"used": False}

                    score = _score_osc(detail, family=fam, vin=vin, vout=vout)
                    dt = float(time.time() - t0)
                    ch = str(detail.get("canonical_hash") or _sha1_text(_canon_inc_text(inc_fin)))
                    return {
                        "ridx": int(ridx),
                        "inc": str(inc_fin),
                        "score": float(score),
                        "detail": detail,
                        "repair": repair_meta,
                        "gen_time": float(dt),
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

            scored = sorted(list(best_by_hash.values()), key=lambda r: float(r.get("score") or -1e9), reverse=True)
            (task_dir / "scored.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")

            chosen_pool = [r for r in scored if bool((r.get("detail") or {}).get("pass_CV", False))]

            seen_sft: set[str] = set()
            seen_sft_strict: set[str] = set()
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

            pairs_written = 0
            if chosen_pool:
                chosen_top = chosen_pool[: max(1, int(args.top_k_chosen))]
                for chosen in chosen_top:
                    if pairs_written >= int(args.pairs_per_task):
                        break
                    c_score = float(chosen.get("score") or -1e9)
                    rejected_cands = [r for r in scored if float(r.get("score") or -1e9) <= c_score - float(args.min_pair_gap)]
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

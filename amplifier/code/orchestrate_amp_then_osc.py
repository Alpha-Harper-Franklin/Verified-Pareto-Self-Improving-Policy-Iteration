#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _append(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(s)


def _run(cmd: List[str], *, cwd: Path, log_path: Path, env: Optional[Dict[str, str]] = None) -> None:
    _append(log_path, "$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
    with log_path.open("a", encoding="utf-8") as f:
        subprocess.check_call(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=env)


def _find_pid_contains(*needles: str) -> Optional[int]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True, errors="ignore")
    except Exception:
        return None
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            pid_s, args = ln.split(maxsplit=1)
            pid = int(pid_s)
        except Exception:
            continue
        ok = True
        for nd in needles:
            if str(nd) not in args:
                ok = False
                break
        if ok:
            return pid
    return None


def _descendants(pid: int) -> List[int]:
    seen: List[int] = []
    q: List[int] = [int(pid)]
    while q:
        p = q.pop(0)
        if p in seen:
            continue
        seen.append(p)
        try:
            out = subprocess.check_output(["pgrep", "-P", str(p)], text=True, errors="ignore")
            for s in out.split():
                try:
                    c = int(s.strip())
                except Exception:
                    continue
                if c not in seen:
                    q.append(c)
        except subprocess.CalledProcessError:
            continue
        except Exception:
            continue
    return seen


def _kill_tree(pid: int, *, timeout_s: float = 60.0) -> None:
    pids = _descendants(int(pid))
    for p in reversed(pids):
        try:
            os.kill(int(p), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            pass

    t0 = time.time()
    while time.time() - t0 < float(timeout_s):
        alive = 0
        for p in pids:
            try:
                os.kill(int(p), 0)
                alive += 1
            except ProcessLookupError:
                continue
            except Exception:
                continue
        if alive == 0:
            return
        time.sleep(2.0)

    for p in reversed(pids):
        try:
            os.kill(int(p), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass


def _wait_for_last_round(
    outdir: Path,
    *,
    target_last_round: int,
    poll_s: float,
    log_path: Path,
    ensure_running: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    _append(log_path, f"[wait] target_last_round={target_last_round}\n")
    fin = outdir / "final_state.json"
    while True:
        if fin.exists() and fin.stat().st_size > 0:
            st = _read_json(fin)
            try:
                lr = int(st.get("last_round"))
            except Exception:
                lr = -999
            if lr >= int(target_last_round):
                _append(log_path, f"[wait] reached last_round={lr}\n")
                return st
        if ensure_running is not None:
            try:
                ensure_running()
            except Exception:
                pass
        time.sleep(float(poll_s))


def _latest_run_argv(outdir: Path) -> List[str]:
    cands = sorted(outdir.glob("run_config_resume_*.json"))
    cfg_path = cands[-1] if cands else (outdir / "run_config.json")
    cfg = _read_json(cfg_path) if cfg_path.exists() else {}
    argv = cfg.get("argv") or []
    return [str(x) for x in argv if str(x).strip()]


def _ensure_amp_pipeline_running(*, code_dir: Path, amp_outdir: Path, log_path: Path) -> None:
    pid = _find_pid_contains("run_vpspi_pipeline_amp.py", "--outdir", str(amp_outdir))
    if pid is not None:
        return
    argv = _latest_run_argv(amp_outdir)
    if not argv:
        _append(log_path, "[warn] cannot restart amp pipeline: missing run_config argv\n")
        return
    cmd = [sys.executable, str(code_dir / "run_vpspi_pipeline_amp.py"), *argv[1:]]
    if "--outdir" not in cmd:
        cmd += ["--outdir", str(amp_outdir)]
    if "--resume" not in cmd:
        cmd.append("--resume")
    _append(log_path, "[warn] amp pipeline missing; restarting: " + " ".join(shlex.quote(x) for x in cmd) + "\n")
    try:
        subprocess.Popen(cmd, cwd=str(code_dir))
    except Exception as e:
        _append(log_path, f"[warn] restart failed: {type(e).__name__}: {e}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp_outdir", required=True)
    ap.add_argument("--amp_extra_rounds", type=int, default=1)
    ap.add_argument("--amp_eval_n", type=int, default=10)
    ap.add_argument("--osc_rounds", type=int, default=3)
    ap.add_argument("--osc_eval_n", type=int, default=10)
    ap.add_argument("--poll_s", type=float, default=30.0)
    args = ap.parse_args()

    code_dir = Path(__file__).resolve().parent

    amp_outdir = Path(str(args.amp_outdir)).resolve()
    amp_log = amp_outdir / "logs" / f"chain_{_now()}.log"

    states = sorted(amp_outdir.glob("round_*/round_state.json"))
    done_rounds: List[int] = []
    for p in states:
        try:
            obj = _read_json(p)
            done_rounds.append(int(obj.get("round")))
        except Exception:
            continue
    done = max(done_rounds) if done_rounds else -1

    if done < 0:
        target_last_round = 1
    else:
        target_last_round = int(done) + 1 + int(args.amp_extra_rounds)

    _append(
        amp_log,
        f"[start] ts={_now()} amp_outdir={amp_outdir} done_round={done} target_last_round={target_last_round}\n",
    )

    st = _wait_for_last_round(
        amp_outdir,
        target_last_round=int(target_last_round),
        poll_s=float(args.poll_s),
        log_path=amp_log,
        ensure_running=lambda: _ensure_amp_pipeline_running(code_dir=code_dir, amp_outdir=amp_outdir, log_path=amp_log),
    )

    pid = _find_pid_contains("run_vpspi_pipeline_amp.py", "--outdir", str(amp_outdir))
    if pid is not None:
        _append(amp_log, f"[stop] killing amp pipeline pid={pid}\n")
        _kill_tree(int(pid), timeout_s=90.0)
    else:
        _append(amp_log, "[stop] amp pipeline pid not found (already stopped?)\n")

    fin = amp_outdir / "final_state.json"
    st2 = _read_json(fin) if fin.exists() else dict(st or {})
    amp_adapter = str(st2.get("cur_adapter") or "").strip() or str(st.get("cur_adapter") or "").strip()
    if not amp_adapter:
        raise SystemExit("cannot find amp cur_adapter for eval")

    argv = _latest_run_argv(amp_outdir)
    if not argv:
        raise SystemExit("cannot find amp run_config argv (for base_model)")

    def _get(flag: str, default: str = "") -> str:
        try:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return str(argv[i + 1])
        except Exception:
            return default
        return default

    base_model = _get("--base_model")
    if not base_model:
        raise SystemExit("cannot find base_model in run_config")

    amp_eval_dir = amp_outdir / f"full_eval_amp_round{int(target_last_round):02d}_{_now()}"
    amp_eval_cmd = [
        sys.executable,
        str(code_dir / "eval_amp_dual.py"),
        "--base_model",
        str(base_model),
        "--adapter",
        str(amp_adapter),
        "--outdir",
        str(amp_eval_dir),
        "--n_per_task",
        str(int(args.amp_eval_n)),
        "--sim_workers",
        "128",
        "--with_repair",
    ]
    _append(amp_log, f"[eval] amp_adapter={amp_adapter}\n")
    _run(amp_eval_cmd, cwd=code_dir, log_path=amp_log)

    osc_anchor = Path("/root/autodl-tmp/vpspi_multitype_v3_5pts/branches/osc/anchor_sft/sft_final")
    if not osc_anchor.exists():
        raise SystemExit(f"missing osc anchor: {osc_anchor}")

    osc_outdir = Path("/root/autodl-tmp/vpspi_multitype_v3_5pts/seq_runs") / f"osc_v3_{_now()}"
    osc_log = osc_outdir / "logs" / "pipeline.log"
    osc_outdir.mkdir(parents=True, exist_ok=True)
    (osc_outdir / "logs").mkdir(parents=True, exist_ok=True)

    osc_cmd = [
        sys.executable,
        str(code_dir / "run_vpspi_pipeline_osc.py"),
        "--base_model",
        str(base_model),
        "--anchor_adapter",
        str(osc_anchor),
        "--outdir",
        str(osc_outdir),
        "--rounds",
        str(int(args.osc_rounds)),
        "--ddp_backend",
        "gloo",
        "--tol",
        "0.01",
        "--min_elems",
        "15",
        "--selfplay_shards",
        "8",
        "--selfplay_sim_workers",
        "160",
        "--ppo_sim_workers",
        "160",
        "--ppo_steps",
        "80",
        "--ppo_batch_size",
        "8",
        "--ppo_group_size",
        "4",
        "--ppo_accum",
        "2",
        "--ppo_mbs",
        "4",
        "--ppo_lr",
        "5e-6",
        "--ppo_target_kl",
        "0.03",
        "--ppo_temperature",
        "0.7",
        "--ppo_top_p",
        "0.9",
    ]
    _append(amp_log, f"[osc] starting osc pipeline outdir={osc_outdir}\n")
    _run(osc_cmd, cwd=code_dir, log_path=osc_log)

    osc_state = _read_json(osc_outdir / "final_state.json")
    osc_adapter = str(osc_state.get("cur_adapter") or "").strip()
    if not osc_adapter:
        raise SystemExit("cannot find osc cur_adapter")

    osc_eval_dir = osc_outdir / f"full_eval_osc_{_now()}"
    osc_eval_cmd = [
        sys.executable,
        str(code_dir / "eval_osc_dual.py"),
        "--base_model",
        str(base_model),
        "--adapter",
        str(osc_adapter),
        "--outdir",
        str(osc_eval_dir),
        "--n_per_task",
        str(int(args.osc_eval_n)),
        "--sim_workers",
        "128",
        "--with_repair",
    ]
    _append(amp_log, f"[eval] osc_adapter={osc_adapter}\n")
    _run(osc_eval_cmd, cwd=code_dir, log_path=osc_log)

    _append(
        amp_log,
        f"[done] ts={_now()} amp_eval={amp_eval_dir} osc_outdir={osc_outdir} osc_eval={osc_eval_dir}\n",
    )
    print("[OK] chain finished")
    print(json.dumps({"amp_eval": str(amp_eval_dir), "osc_outdir": str(osc_outdir), "osc_eval": str(osc_eval_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

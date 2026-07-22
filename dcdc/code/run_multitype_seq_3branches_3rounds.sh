#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/root/autodl-tmp/hf_home
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache
export PATH=/root/miniconda3/bin:$PATH

BASE_MODEL="/root/models/Qwen2.5-7B.real/Qwen/Qwen2___5-7B"

ANCHOR_AMP="/root/autodl-tmp/vpspi_multitype/branches/amp/anchor_adapter"
ANCHOR_FILTER="/root/autodl-tmp/vpspi_multitype/branches/filter/anchor_adapter"
ANCHOR_OSC="/root/autodl-tmp/vpspi_multitype/branches/osc/anchor_adapter"

CODE_DIR="/root/autodl-tmp/dcdc_family/code"
SEQ_ROOT="/root/autodl-tmp/vpspi_multitype/seq_runs"
CURRENT_PTR="${SEQ_ROOT}/current_run.txt"

mkdir -p "${SEQ_ROOT}"

# Preflight: require GPUs (this script is meant to be launched AFTER GPUs are enabled).
GPU_COUNT=$(/root/miniconda3/bin/python - <<'PY'
import torch
print(int(torch.cuda.device_count()) if torch.cuda.is_available() else 0)
PY
)
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "[FATAL] No GPUs visible (torch.cuda.device_count()==0). Please enable GPUs, then rerun." >&2
  exit 2
fi
if [[ "${GPU_COUNT}" -lt 8 ]]; then
  echo "[WARN] GPU_COUNT=${GPU_COUNT} (<8). Will run with available GPUs." >&2
fi

# Default to NCCL for speed; auto-fallback to gloo if NCCL is unstable on this machine.
DDP_BACKEND="${DDP_BACKEND:-nccl}"

# Choose / resume experiment root.
RUN_DIR=""
if [[ -f "${CURRENT_PTR}" ]]; then
  CAND=$(cat "${CURRENT_PTR}" | head -n 1 || true)
  if [[ -n "${CAND}" && -d "${CAND}" && ! -f "${CAND}/ALL_DONE" ]]; then
    RUN_DIR="${CAND}"
    echo "[resume] RUN_DIR=${RUN_DIR}"
  fi
fi

if [[ -z "${RUN_DIR}" ]]; then
  TS=$(date +%Y%m%d_%H%M%S)
  RUN_DIR="${SEQ_ROOT}/seq_${TS}_amp_filter_osc"
  mkdir -p "${RUN_DIR}"
  echo "${RUN_DIR}" > "${CURRENT_PTR}"
  echo "[start] RUN_DIR=${RUN_DIR}"
fi

EVAL_N_PER_TASK=${EVAL_N_PER_TASK:-10}
EVAL_SIM_TIMEOUT_S=${EVAL_SIM_TIMEOUT_S:-60}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-320}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.7}
EVAL_TOP_P=${EVAL_TOP_P:-0.9}
EVAL_SEED=${EVAL_SEED:-2026}

run_branch() {
  local NAME="$1"
  local PIPELINE="$2"
  local EVAL_SCRIPT="$3"
  local ANCHOR="$4"

  local OUTDIR="${RUN_DIR}/${NAME}"
  mkdir -p "${OUTDIR}"

  echo "[${NAME}] outdir=${OUTDIR}"

  # Determine how many rounds are already completed.
  local DONE_ROUNDS
  DONE_ROUNDS=$(/root/miniconda3/bin/python - "${OUTDIR}" <<'PY'
import glob, json, os, sys
outdir=sys.argv[1]
paths=glob.glob(os.path.join(outdir,'round_*','round_state.json'))
rounds=set()
for p in paths:
    try:
        obj=json.load(open(p,'r',encoding='utf-8'))
        rounds.add(int(obj.get('round')))
    except Exception:
        pass
print(len(rounds))
PY
)

  local TARGET_ROUNDS=3
  local REMAIN=$(( TARGET_ROUNDS - DONE_ROUNDS ))
  if [[ "${REMAIN}" -gt 0 ]]; then
    echo "[${NAME}] training: done_rounds=${DONE_ROUNDS} remain=${REMAIN} (target=${TARGET_ROUNDS})"

    # Always use resume if directory already has content.
    local RESUME_ARGS=("--outdir" "${OUTDIR}" "--resume")

    # Retry loop for robustness.
    local ATT=0
    while true; do
      ATT=$((ATT+1))
      echo "[${NAME}] pipeline attempt ${ATT}"
      BACKEND_THIS="${DDP_BACKEND}"
      if [[ "${ATT}" -ge 2 && "${DDP_BACKEND}" == "nccl" ]]; then
        BACKEND_THIS="gloo"
      fi
      /root/miniconda3/bin/python "${CODE_DIR}/${PIPELINE}"         --base_model "${BASE_MODEL}"         --anchor_adapter "${ANCHOR}"         "${RESUME_ARGS[@]}"         --rounds "${REMAIN}"         --train_gpus 0         --ddp_backend "${BACKEND_THIS}"         --selfplay_shards 0         --selfplay_sim_workers 0         --ppo_sim_workers 0         && break

      if [[ "${ATT}" -ge 3 ]]; then
        echo "[${NAME}] FATAL: pipeline failed after ${ATT} attempts" >&2
        exit 3
      fi
      echo "[${NAME}] warn: pipeline failed (attempt ${ATT}), sleeping then retrying with resume..." >&2
      sleep 30
    done

    touch "${OUTDIR}/DONE_TRAIN"
  else
    echo "[${NAME}] training already completed: done_rounds=${DONE_ROUNDS}"
    touch "${OUTDIR}/DONE_TRAIN"
  fi

  # Extract the final adapter path from final_state.json.
  local FINAL_STATE="${OUTDIR}/final_state.json"
  if [[ ! -f "${FINAL_STATE}" ]]; then
    echo "[${NAME}] FATAL: missing final_state.json at ${FINAL_STATE}" >&2
    exit 4
  fi
  local ADAPTER
  ADAPTER=$(/root/miniconda3/bin/python - "${FINAL_STATE}" <<'PY'
import json,sys
p=sys.argv[1]
obj=json.load(open(p,'r',encoding='utf-8'))
print(str(obj.get('cur_adapter') or '').strip())
PY
)
  if [[ -z "${ADAPTER}" ]]; then
    echo "[${NAME}] FATAL: cur_adapter is empty in ${FINAL_STATE}" >&2
    exit 5
  fi
  echo "${ADAPTER}" > "${OUTDIR}/FINAL_ADAPTER.txt"
  echo "[${NAME}] FINAL_ADAPTER=${ADAPTER}"

  # Full eval (DCDC-like): open-loop generation, no fallback/template.
  local EVAL_DIR="${OUTDIR}/eval_full_n${EVAL_N_PER_TASK}"
  if [[ -f "${EVAL_DIR}/eval_summary.json" ]]; then
    echo "[${NAME}] eval already exists: ${EVAL_DIR}"
  else
    echo "[${NAME}] eval starting: n_per_task=${EVAL_N_PER_TASK}"
    /root/miniconda3/bin/python "${CODE_DIR}/${EVAL_SCRIPT}"       --base_model "${BASE_MODEL}"       --adapter "${ADAPTER}"       --outdir "${EVAL_DIR}"       --n_per_task "${EVAL_N_PER_TASK}"       --seed "${EVAL_SEED}"       --max_new_tokens "${EVAL_MAX_NEW_TOKENS}"       --temperature "${EVAL_TEMPERATURE}"       --top_p "${EVAL_TOP_P}"       --sim_timeout_s "${EVAL_SIM_TIMEOUT_S}"       --min_elems 20       --no_fallback

    /root/miniconda3/bin/python "${CODE_DIR}/summarize_eval_summary.py"       --eval_summary "${EVAL_DIR}/eval_summary.json"       --out "${EVAL_DIR}/eval_metrics.json"
  fi

  touch "${OUTDIR}/DONE_EVAL"
}

run_branch amp run_vpspi_pipeline_amp.py eval_amp_family.py "${ANCHOR_AMP}"
run_branch filter run_vpspi_pipeline_filter.py eval_filter_family.py "${ANCHOR_FILTER}"
run_branch osc run_vpspi_pipeline_osc.py eval_osc_family.py "${ANCHOR_OSC}"

touch "${RUN_DIR}/ALL_DONE"
echo "[OK] all branches finished. RUN_DIR=${RUN_DIR}"

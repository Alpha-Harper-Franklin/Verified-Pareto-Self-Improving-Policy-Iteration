#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/root/autodl-tmp/hf_home
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache
export PATH=/root/miniconda3/bin:$PATH

BASE_MODEL="${BASE_MODEL:-/root/models/Qwen2.5-7B.real/Qwen/Qwen2___5-7B}"
CODE_DIR="${CODE_DIR:-/root/autodl-tmp/dcdc_family/code}"
ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/vpspi_multitype_v2}"

ANCHOR_MIN_ELEMS="${ANCHOR_MIN_ELEMS:-20}"
ANCHOR_N="${ANCHOR_N:-8000}"
ANCHOR_SEED="${ANCHOR_SEED:-20260114}"

TARGET_BRANCH_ROUNDS="${TARGET_BRANCH_ROUNDS:-3}"
DDP_BACKEND="${DDP_BACKEND:-gloo}"  # default to gloo for stability on this machine

PREPARE_ONLY=0
if [[ "${1:-}" == "--prepare_only" ]]; then
  PREPARE_ONLY=1
  shift
fi

mkdir -p "${ROOT_DIR}/branches" "${ROOT_DIR}/seq_runs" "${ROOT_DIR}/logs"

log() { echo "[$(date +%F' '%T)] $*"; }

log "BASE_MODEL=${BASE_MODEL}"
log "CODE_DIR=${CODE_DIR}"
log "ROOT_DIR=${ROOT_DIR}"
log "ANCHOR_N=${ANCHOR_N} ANCHOR_MIN_ELEMS=${ANCHOR_MIN_ELEMS} ANCHOR_SEED=${ANCHOR_SEED}"
log "DDP_BACKEND=${DDP_BACKEND}"

export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
cd "${CODE_DIR}"

# 0) Build anchor template datasets (CPU-only)
build_anchor_dataset() {
  local NAME="$1"
  local GEN_SCRIPT="$2"
  local OUT_JSONL="${ROOT_DIR}/branches/${NAME}/anchor_sft_templates.jsonl"
  mkdir -p "${ROOT_DIR}/branches/${NAME}"
  if [[ -s "${OUT_JSONL}" ]]; then
    log "[anchor:${NAME}] dataset exists: ${OUT_JSONL} ($(wc -l <"${OUT_JSONL}") lines)"
    return 0
  fi
  log "[anchor:${NAME}] building dataset: ${OUT_JSONL}"
  /root/miniconda3/bin/python "${CODE_DIR}/${GEN_SCRIPT}" \
    --out_jsonl "${OUT_JSONL}" \
    --n "${ANCHOR_N}" \
    --min_elems "${ANCHOR_MIN_ELEMS}" \
    --seed "${ANCHOR_SEED}" \
    >"${ROOT_DIR}/logs/anchor_${NAME}_dataset_build.log" 2>&1
  log "[anchor:${NAME}] built: ${OUT_JSONL} ($(wc -l <"${OUT_JSONL}") lines)"
}

build_anchor_dataset amp make_sft_amp_from_templates.py
build_anchor_dataset filter make_sft_filter_from_templates.py
build_anchor_dataset osc make_sft_osc_from_templates.py

# 0.1) Quick structural sanity check (CPU-only)
log "[anchor] sanity-checking a few samples (structure only)"
/root/miniconda3/bin/python - <<'PY'
import json, random
from pathlib import Path

import os
ROOT=Path(os.environ.get('ROOT_DIR','/root/autodl-tmp/vpspi_multitype_v2'))

checks=[
  ('amp', ROOT/'branches'/'amp'/'anchor_sft_templates.jsonl', 'amp_verifier', 'verify_inc_amp', {'min_elems':15}),
  ('filter', ROOT/'branches'/'filter'/'anchor_sft_templates.jsonl', 'filter_verifier', 'verify_inc_filter', {'min_elems':20}),
  ('osc', ROOT/'branches'/'osc'/'anchor_sft_templates.jsonl', 'osc_verifier', 'verify_inc_osc', {'min_elems':15}),
]

for name, path, mod, fn, kwargs in checks:
  assert path.exists() and path.stat().st_size>0, (name, path)
  m=__import__(mod, fromlist=[fn])
  verify=getattr(m, fn)
  rows=[]
  with path.open('r',encoding='utf-8') as f:
    for i,line in enumerate(f):
      if i>=200: break
      try:
        rows.append(json.loads(line)['text'])
      except Exception:
        pass
  random.shuffle(rows)
  ok=0
  n=0
  for txt in rows[:20]:
    # extract response part
    body=txt.split('### Response:',1)[-1]
    if name == 'osc':
      fam = 'osc_rc'
      for ln in txt.splitlines():
        s = ln.strip().lower()
        if 'family=' in s:
          fam = ln.split('family=', 1)[1].strip()
          fam = fam.split()[0].strip().strip(',')
          break
      res = verify(body, family=str(fam).strip(), **kwargs)
    else:
      res = verify(body, **kwargs)
    n+=1
    if getattr(res,'ok',False):
      ok+=1
  print(f'[sanity] {name}: ok={ok}/{n} (path={path})')
PY

if [[ "${PREPARE_ONLY}" -eq 1 ]]; then
  log "[READY] prepare_only=1: datasets generated + sanity checked. Enable GPUs then rerun without --prepare_only."
  exit 0
fi

# 1) Train per-branch anchor adapters (multi-GPU)
GPU_COUNT=$(/root/miniconda3/bin/python - <<'PY'
import torch
print(int(torch.cuda.device_count()) if torch.cuda.is_available() else 0)
PY
)
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  log "[FATAL] No GPUs visible (torch.cuda.device_count()==0)."
  exit 2
fi

log "[env] GPU_COUNT=${GPU_COUNT}"

train_anchor() {
  local NAME="$1"
  local TRAIN_JSONL="${ROOT_DIR}/branches/${NAME}/anchor_sft_templates.jsonl"
  local OUTDIR="${ROOT_DIR}/branches/${NAME}/anchor_sft"
  local FINAL="${OUTDIR}/sft_final/adapter_model.safetensors"
  mkdir -p "${OUTDIR}"
  if [[ -s "${FINAL}" ]]; then
    log "[anchor:${NAME}] already trained: ${FINAL}"
    return 0
  fi
  log "[anchor:${NAME}] training SFT anchor (gpus=${GPU_COUNT})"

  # retry with resume for robustness
  local ATT=0
  while true; do
    ATT=$((ATT+1))
    log "[anchor:${NAME}] attempt=${ATT}"
    PORT=$(/root/miniconda3/bin/python - <<'PY'
import socket
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
PY
    )
    /root/miniconda3/bin/accelerate launch \
      --multi_gpu \
      --num_processes "${GPU_COUNT}" \
      --mixed_precision bf16 \
      --main_process_port "${PORT}" \
      "${CODE_DIR}/train_sft_dcdc.py" \
        --base_model "${BASE_MODEL}" \
        --train_jsonl "${TRAIN_JSONL}" \
        --outdir "${OUTDIR}" \
        --lr 2e-5 \
        --max_steps 600 \
        --bsz 1 \
        --grad_accum 16 \
        --save_steps 50 \
        --save_total_limit 3 \
        --ddp_backend "${DDP_BACKEND}" \
        --resume \
        >"${ROOT_DIR}/logs/anchor_${NAME}_sft.log" 2>&1 \
        && break

    if [[ "${ATT}" -ge 3 ]]; then
      log "[anchor:${NAME}] FATAL: failed after ${ATT} attempts. See ${ROOT_DIR}/logs/anchor_${NAME}_sft.log"
      exit 3
    fi
    log "[anchor:${NAME}] warn: failed attempt=${ATT}, sleep then retry with resume"
    sleep 20
  done

  if [[ ! -s "${FINAL}" ]]; then
    log "[anchor:${NAME}] FATAL: missing ${FINAL} after training"
    exit 4
  fi
  log "[anchor:${NAME}] OK: ${OUTDIR}/sft_final"
}

train_anchor amp
train_anchor filter
train_anchor osc

# 2) Branch map
/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
import os
root=Path(os.environ.get('ROOT_DIR','/root/autodl-tmp/vpspi_multitype_v2'))
mp={
  'amp': str(root/'branches'/'amp'/'anchor_sft'/'sft_final'),
  'filter': str(root/'branches'/'filter'/'anchor_sft'/'sft_final'),
  'osc': str(root/'branches'/'osc'/'anchor_sft'/'sft_final'),
}
(root/'branches').mkdir(parents=True, exist_ok=True)
(root/'branches'/'branch_map.json').write_text(json.dumps(mp,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('[OK] wrote', root/'branches'/'branch_map.json')
PY

# 3) Run VP-SPI pipeline per branch (3 rounds each by default), sequential.
run_branch() {
  local NAME="$1"
  local PIPELINE="$2"
  local EVAL_SCRIPT="$3"
  local ANCHOR="${ROOT_DIR}/branches/${NAME}/anchor_sft/sft_final"

  local OUTDIR="${ROOT_DIR}/seq_runs/${NAME}"
  mkdir -p "${OUTDIR}"

  log "[${NAME}] outdir=${OUTDIR} anchor=${ANCHOR} rounds=${TARGET_BRANCH_ROUNDS}"

  /root/miniconda3/bin/python "${CODE_DIR}/${PIPELINE}" \
    --base_model "${BASE_MODEL}" \
    --anchor_adapter "${ANCHOR}" \
    --outdir "${OUTDIR}" \
    --rounds "${TARGET_BRANCH_ROUNDS}" \
    --ddp_backend "${DDP_BACKEND}" \
    --train_gpus 0 \
    --selfplay_shards 0 \
    --selfplay_sim_workers 0 \
    --ppo_sim_workers 0 \
    --resume \
    >"${ROOT_DIR}/logs/${NAME}_pipeline.log" 2>&1

  # Full eval (open-loop generation, no fallback/template).
  local ADAPTER
  ADAPTER=$(/root/miniconda3/bin/python - <<PY
import json
p='${OUTDIR}/final_state.json'
obj=json.load(open(p,'r',encoding='utf-8'))
print(str(obj.get('cur_adapter') or '').strip())
PY
)
  if [[ -z "${ADAPTER}" ]]; then
    log "[${NAME}] FATAL: cur_adapter empty in final_state.json"
    exit 5
  fi

  local EVAL_DIR="${OUTDIR}/eval_full_n10"
  if [[ -f "${EVAL_DIR}/eval_summary.json" ]]; then
    log "[${NAME}] eval exists: ${EVAL_DIR}"
  else
    log "[${NAME}] eval starting: ${EVAL_DIR}"
    /root/miniconda3/bin/python "${CODE_DIR}/${EVAL_SCRIPT}" \
      --base_model "${BASE_MODEL}" \
      --adapter "${ADAPTER}" \
      --outdir "${EVAL_DIR}" \
      --n_per_task 10 \
      --seed 2026 \
      --max_new_tokens 320 \
      --temperature 0.7 \
      --top_p 0.9 \
      --sim_timeout_s 60 \
      --min_elems 15 \
      --no_fallback

    /root/miniconda3/bin/python "${CODE_DIR}/summarize_eval_summary.py" \
      --eval_summary "${EVAL_DIR}/eval_summary.json" \
      --out "${EVAL_DIR}/eval_metrics.json"
  fi
}

run_branch amp run_vpspi_pipeline_amp.py eval_amp_family.py
run_branch filter run_vpspi_pipeline_filter.py eval_filter_family.py
run_branch osc run_vpspi_pipeline_osc.py eval_osc_family.py

log "[OK] multi-type VP-SPI finished. ROOT_DIR=${ROOT_DIR}"

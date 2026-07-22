#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/autodl-tmp/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-tmp/hf_datasets_cache}"
export PATH="/root/miniconda3/bin:${PATH}"

CODE_DIR="${CODE_DIR:-/root/autodl-tmp/dcdc_family/code}"
BASE_MODEL="${BASE_MODEL:-/root/models/Qwen2.5-7B.real/Qwen/Qwen2___5-7B}"

# Output directory for an entire OSC run (anchor SFT + VP-SPI rounds)
RUN_OUTDIR="${RUN_OUTDIR:-/root/autodl-tmp/vpspi_multitype_v3_5pts/seq_runs/osc_vpspi_3rounds}"

# VP-SPI main loop
ROUNDS="${ROUNDS:-3}"
TOL="${TOL:-0.01}"
MIN_ELEMS="${MIN_ELEMS:-20}"
SELFPLAY_SHARDS="${SELFPLAY_SHARDS:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-0}"
DDP_BACKEND="${DDP_BACKEND:-auto}"

# Anchor SFT (osc)
ANCHOR_N="${ANCHOR_N:-8000}"
ANCHOR_STEPS="${ANCHOR_STEPS:-600}"
ANCHOR_LR="${ANCHOR_LR:-1e-5}"
ANCHOR_BSZ="${ANCHOR_BSZ:-1}"
ANCHOR_ACCUM="${ANCHOR_ACCUM:-16}"
ANCHOR_SEED="${ANCHOR_SEED:-20260117}"

# Safety: do not burn CPU-only quota unless explicitly allowed.
ALLOW_CPU="${ALLOW_CPU:-0}"

mkdir -p "${RUN_OUTDIR}"

GPU_COUNT="$(python - <<'PY'
try:
    import torch
    n = int(torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    n = 0
print(n)
PY
)"

if [ "${GPU_COUNT}" -le 0 ] && [ "${ALLOW_CPU}" -ne 1 ]; then
  echo "[error] No GPU visible (GPU_COUNT=${GPU_COUNT}). Set ALLOW_CPU=1 to run anyway." >&2
  exit 2
fi

if [ "${TRAIN_GPUS}" -le 0 ]; then
  TRAIN_GPUS="${GPU_COUNT}"
fi
if [ "${TRAIN_GPUS}" -le 0 ]; then
  TRAIN_GPUS=1
fi

if [ "${DDP_BACKEND}" = "auto" ]; then
  if [ "${GPU_COUNT}" -gt 0 ]; then
    DDP_BACKEND="nccl"
  else
    DDP_BACKEND="gloo"
  fi
fi

if [ ! -d "${CODE_DIR}" ]; then
  echo "[error] CODE_DIR not found: ${CODE_DIR}" >&2
  exit 1
fi
if [ ! -d "${BASE_MODEL}" ]; then
  echo "[error] BASE_MODEL not found: ${BASE_MODEL}" >&2
  exit 1
fi

echo "[cfg] CODE_DIR=${CODE_DIR}"
echo "[cfg] BASE_MODEL=${BASE_MODEL}"
echo "[cfg] RUN_OUTDIR=${RUN_OUTDIR}"
echo "[cfg] GPU_COUNT=${GPU_COUNT}"
echo "[cfg] TRAIN_GPUS=${TRAIN_GPUS}"
echo "[cfg] DDP_BACKEND=${DDP_BACKEND}"
echo "[cfg] SELFPLAY_SHARDS=${SELFPLAY_SHARDS}"
echo "[cfg] ROUNDS=${ROUNDS}"
echo "[cfg] MIN_ELEMS=${MIN_ELEMS}"
echo "[cfg] TOL=${TOL}"

ANCHOR_JSONL="${RUN_OUTDIR}/anchor_sft_train.jsonl"
ANCHOR_OUTDIR="${RUN_OUTDIR}/anchor_sft"
ANCHOR_ADAPTER="${ANCHOR_OUTDIR}/sft_final"

# 0) Build a fresh anchor dataset (templates -> SFT). Previous anchors are treated as obsolete.
echo "[anchor] build dataset -> ${ANCHOR_JSONL} (n=${ANCHOR_N})"
python "${CODE_DIR}/make_sft_osc_from_templates.py" \
  --out_jsonl "${ANCHOR_JSONL}" \
  --n "${ANCHOR_N}" \
  --min_elems "${MIN_ELEMS}" \
  --seed "${ANCHOR_SEED}"

# 1) Train the anchor SFT adapter (LoRA) from the base model.
echo "[anchor] train SFT -> ${ANCHOR_OUTDIR} (max_steps=${ANCHOR_STEPS})"
if [ "${TRAIN_GPUS}" -gt 1 ] && command -v accelerate >/dev/null 2>&1; then
  accelerate launch --num_processes "${TRAIN_GPUS}" --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    "${CODE_DIR}/train_sft_dcdc.py" \
    --base_model "${BASE_MODEL}" \
    --train_jsonl "${ANCHOR_JSONL}" \
    --outdir "${ANCHOR_OUTDIR}" \
    --lr "${ANCHOR_LR}" \
    --bsz "${ANCHOR_BSZ}" \
    --grad_accum "${ANCHOR_ACCUM}" \
    --max_steps "${ANCHOR_STEPS}" \
    --save_steps 50 \
    --save_total_limit 3 \
    --resume \
    --ddp_backend "${DDP_BACKEND}"
else
  python "${CODE_DIR}/train_sft_dcdc.py" \
    --base_model "${BASE_MODEL}" \
    --train_jsonl "${ANCHOR_JSONL}" \
    --outdir "${ANCHOR_OUTDIR}" \
    --lr "${ANCHOR_LR}" \
    --bsz "${ANCHOR_BSZ}" \
    --grad_accum "${ANCHOR_ACCUM}" \
    --max_steps "${ANCHOR_STEPS}" \
    --save_steps 50 \
    --save_total_limit 3 \
    --resume \
    --ddp_backend "${DDP_BACKEND}"
fi

if [ ! -d "${ANCHOR_ADAPTER}" ]; then
  echo "[error] missing anchor adapter: ${ANCHOR_ADAPTER}" >&2
  exit 1
fi

echo "[anchor] OK: ${ANCHOR_ADAPTER}"

# 2) VP-SPI (A?C?D) for oscillators, 3 rounds, resume-safe.
#    Enable repair-corrector (small MLP) to reduce reliance on pure search.
echo "[vpspi] start VP-SPI osc rounds=${ROUNDS}"
python "${CODE_DIR}/run_vpspi_pipeline_osc.py" \
  --base_model "${BASE_MODEL}" \
  --anchor_adapter "${ANCHOR_ADAPTER}" \
  --outdir "${RUN_OUTDIR}" \
  --resume \
  --rounds "${ROUNDS}" \
  --tol "${TOL}" \
  --min_elems "${MIN_ELEMS}" \
  --train_gpus "${TRAIN_GPUS}" \
  --ddp_backend "${DDP_BACKEND}" \
  --selfplay_shards "${SELFPLAY_SHARDS}" \
  --selfplay_sim_workers 0 \
  --ppo_sim_workers 0 \
  --repair_corrector_enable

echo "[done] VP-SPI OSC complete: ${RUN_OUTDIR}"

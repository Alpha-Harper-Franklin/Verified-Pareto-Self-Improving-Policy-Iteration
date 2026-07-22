#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/root/autodl-tmp/hf_home
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache
export PATH=/root/miniconda3/bin:$PATH

BASE_MODEL="/root/models/Qwen2.5-7B.real/Qwen/Qwen2___5-7B"
ANCHOR_ADAPTER="/root/autodl-tmp/vpspi_multitype/branches/osc/anchor_adapter"

CODE_DIR="/root/autodl-tmp/dcdc_family/code"
RUN_ROOT="/root/autodl-tmp/vpspi_multitype/runs"

DDP_BACKEND=$(/root/miniconda3/bin/python - <<'PY'
import torch
print('nccl' if torch.cuda.is_available() else 'gloo')
PY
)

echo "[run_osc_branch] DDP_BACKEND=${DDP_BACKEND}"

echo "[run_osc_branch] base_model=${BASE_MODEL}"
echo "[run_osc_branch] anchor_adapter=${ANCHOR_ADAPTER}"
echo "[run_osc_branch] run_root=${RUN_ROOT}"

exec /root/miniconda3/bin/python "${CODE_DIR}/run_vpspi_pipeline_osc.py"   --base_model "${BASE_MODEL}"   --anchor_adapter "${ANCHOR_ADAPTER}"   --run_root "${RUN_ROOT}"   --name osc   --rounds 0   --train_gpus 0   --ddp_backend "${DDP_BACKEND}"   --selfplay_shards 0   --selfplay_sim_workers 0   --ppo_sim_workers 0

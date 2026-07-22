#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 /path/to/Qwen2.5-7B /path/to/anchor_adapter /path/to/pvpo_sft.jsonl [run_root]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_model="$1"
anchor_adapter="$2"
pvpo_sft_jsonl="$3"
run_root="${4:-$repo_root/dcdc/code/runs}"

cd "$repo_root/dcdc/code"
python run_vpspi_pipeline.py \
  --base_model "$base_model" \
  --anchor_adapter "$anchor_adapter" \
  --run_root "$run_root" \
  --name mosimi_reproduction \
  --rounds 1 \
  --tol 0.01 \
  --min_elems 20 \
  --selfplay_n_gen 20 \
  --selfplay_max_rounds 4 \
  --selfplay_min_pass_cv 2 \
  --selfplay_eda_repair \
  --selfplay_repair_max_evals 36 \
  --dpo_epochs 3 \
  --pvpo_sft_reg_jsonl "$pvpo_sft_jsonl" \
  --pvpo_sft_reg_max_steps 400 \
  --ppo_steps 80 \
  --ppo_batch_size 8 \
  --ppo_group_size 4 \
  --ppo_lr 5e-6 \
  --ppo_target_kl 0.03 \
  --ppo_clip 0.2 \
  --ppo_max_new_tokens 320 \
  --ppo_temperature 0.7 \
  --ppo_top_p 0.9 \
  --train_gpus 1

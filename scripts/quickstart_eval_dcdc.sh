#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /path/to/Qwen2.5-7B [output_dir]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_model="$1"
output_dir="${2:-$repo_root/dcdc/1_eval_code/replay_final_raw}"
adapter="$repo_root/dcdc/4_train_artifacts_latest/vpspi_tol001_20260111_005147_PURE_SFT_ANCHOR_LOOP_r40_8GPU176CPU/round_02/safe_ppo/ppo_best"

cd "$repo_root/dcdc/1_eval_code"
python eval_dcdc_family.py \
  --base_model "$base_model" \
  --adapter "$adapter" \
  --outdir "$output_dir" \
  --n_per_task 10 \
  --max_new_tokens 320 \
  --temperature 0.7 \
  --top_p 0.9 \
  --seed 2025 \
  --tol 0.01 \
  --min_elems 20 \
  --constrained \
  --no_fallback

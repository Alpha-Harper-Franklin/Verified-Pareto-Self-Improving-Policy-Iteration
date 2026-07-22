#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 /path/to/Qwen2.5-7B /path/to/final_adapter [output_dir]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_model="$1"
adapter="$2"
output_dir="${3:-$repo_root/dcdc/code/replay_final_raw}"

cd "$repo_root/dcdc/code"
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

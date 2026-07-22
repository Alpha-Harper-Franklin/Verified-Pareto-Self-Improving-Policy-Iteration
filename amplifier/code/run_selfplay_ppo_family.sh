#!/usr/bin/env bash
set -euo pipefail

# Circuit self-play + SFT + PPO (GRPO-style group reward) pipeline.
# - Uses official dataset (>1000 lines) + DC-DC templates + self-play rollouts.
# - PPO uses multi-objective weight annealing + fixed 2% tolerance (in train_ppo_dcdc.py).
# - No template fallback is used in evaluation; templates are used only as SFT pretraining data.
#
# Usage on CA800:
#   bash run_selfplay_ppo_family.sh [/root/autodl-tmp/dcdc_family/runs/selfplay_ppo_YYYYmmdd_HHMMSS]

PY="${PY:-/root/miniconda3/bin/python}"
CODE="${CODE:-/root/autodl-tmp/dcdc_family/code}"

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-7B.real/Qwen/Qwen2___5-7B}"
OFFICIAL_JSONL="${OFFICIAL_JSONL:-/root/autodl-tmp/datasets/official_power_v1/train.jsonl}"

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
  RUN_DIR="/root/autodl-tmp/dcdc_family/runs/selfplay_ppo_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/data"
echo "$RUN_DIR" > "$RUN_DIR/RUN_DIR.txt"

log() { echo "[$(date +%F' '%T)] $*"; }

STEP_MARK="$RUN_DIR/steps_done.txt"
touch "$STEP_MARK"
mark_done() { echo "$1" >>"$STEP_MARK"; }
is_done() { grep -Fxq "$1" "$STEP_MARK" 2>/dev/null; }

# 1) Official SFT (ALL rows -> guarantee official>1000 actually used)
OFFICIAL_SFT="$RUN_DIR/data/official_sft_all.jsonl"
if ! is_done "official_sft_all"; then
  log "step=official_sft_all"
  "$PY" "$CODE/filter_official_dcdc_sft.py" \
    --official_jsonl "$OFFICIAL_JSONL" \
    --out_jsonl "$OFFICIAL_SFT" \
    --pattern ".*" \
    --require_inc_prefix \
    >"$RUN_DIR/logs/official_sft_all.log" 2>&1
  mark_done "official_sft_all"
fi

# 2) Template SFT (>=20 elements, DC-DC family)
TPL_SFT="$RUN_DIR/data/sft_dcdc_tpl.jsonl"
if ! is_done "tpl_sft"; then
  log "step=tpl_sft"
  "$PY" "$CODE/make_sft_dcdc_from_templates.py" \
    --out_jsonl "$TPL_SFT" \
    --template_variant full \
    >"$RUN_DIR/logs/tpl_sft.log" 2>&1
  mark_done "tpl_sft"
fi

# 3) Self-play build (generate+simulate+rank within task; produces sft_train.jsonl and dpo_pairs.jsonl)
SELFPLAY_DIR="$RUN_DIR/data/selfplay_inc"
mkdir -p "$SELFPLAY_DIR"
if ! is_done "selfplay_inc"; then
  log "step=selfplay_inc"
  "$PY" "$CODE/build_selfplay_inc_datasets.py" \
    --base_model "$BASE_MODEL" \
    --out_root "$SELFPLAY_DIR" \
    --n_gen 16 \
    --pairs_per_task 12 \
    --top_k_chosen 4 \
    --min_pair_gap 0.15 \
    --min_elems 20 \
    --max_new_tokens 320 \
    --temperature 0.7 \
    --top_p 0.9 \
    --seed 2025 \
    --tol 0.02 \
    --rload 10.0 \
    --t_pre 0.008 \
    --t_win 0.002 \
    --autotune_duty \
    --resume \
    >"$RUN_DIR/logs/selfplay_inc.log" 2>&1
  mark_done "selfplay_inc"
fi

# 4) Merge SFT
SFT_MIX="$RUN_DIR/data/sft_mix.jsonl"
if ! is_done "sft_mix"; then
  log "step=sft_mix"
  cat "$OFFICIAL_SFT" "$TPL_SFT" "$SELFPLAY_DIR/sft_train.jsonl" >"$SFT_MIX"
  mark_done "sft_mix"
fi

# 5) SFT training (resume + per-step metrics)
SFT_DIR="$RUN_DIR/sft_stage"
mkdir -p "$SFT_DIR"
if ! is_done "sft_train"; then
  log "step=sft_train"
  "$PY" "$CODE/train_sft_dcdc.py" \
    --base_model "$BASE_MODEL" \
    --sft_jsonl "$SFT_MIX" \
    --outdir "$SFT_DIR" \
    --lr 2e-5 \
    --epochs 1 \
    --bsz 1 \
    --grad_accum 16 \
    --save_steps 50 \
    --save_total_limit 3 \
    --resume \
    >"$RUN_DIR/logs/sft_train.log" 2>&1
  touch "$SFT_DIR/DONE"
  mark_done "sft_train"
fi

# 6) PPO training (GRPO-style group reward inside train_ppo_dcdc.py; per-step reward/loss breakdown logged)
PPO_DIR="$RUN_DIR/ppo_stage"
mkdir -p "$PPO_DIR"
if ! is_done "ppo_train"; then
  log "step=ppo_train"
  "$PY" "$CODE/train_ppo_dcdc.py" \
    --base_model "$BASE_MODEL" \
    --sft_adapter "$SFT_DIR/sft_final" \
    --outdir "$PPO_DIR" \
    --steps 500 \
    --resume \
    --batch_size 8 \
    --mini_batch_size 4 \
    --grad_accum 2 \
    --lr 5e-6 \
    --target_kl 0.03 \
    --vf_coef 0.5 \
    --cliprange 0.2 \
    --cliprange_value 0.2 \
    --ppo_epochs 4 \
    --max_new_tokens 320 \
    --temperature 0.7 \
    --top_p 0.9 \
    --constrained \
    --autotune_duty \
    --group_size 4 \
    --group_reward_mode rank_add \
    --group_rel_coef 0.5 \
    --tol_levels 0.02 \
    --save_steps 25 \
    --save_total_limit 3 \
    >"$RUN_DIR/logs/ppo_train.log" 2>&1
  touch "$PPO_DIR/DONE"
  mark_done "ppo_train"
fi

log "DONE run_dir=$RUN_DIR"

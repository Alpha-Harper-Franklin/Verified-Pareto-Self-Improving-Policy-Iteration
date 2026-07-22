#!/usr/bin/env bash
set -euo pipefail

# 最优路线（最大化每一步对指标的贡献）：
# SFT0(官方+结构先验) -> Self-play(用 SFT0 权重生成候选+仿真打分+组内相对偏好对)
# -> SFT1(蒸馏 self-play 高质量轨迹，初始化自 SFT0) -> DPO(偏好对齐)
# -> PPO(多目标权重退火 + 自适应容差 + 风险敏感约束 + 组内相对项；逐步记录 loss/reward 分项)
#
# 用法：
#   bash run_optimal_route.sh [/root/autodl-tmp/dcdc_family/runs/optimal_YYYYmmdd_HHMMSS]
#
# 断点续训：
#   重复执行同一 RUN_DIR，会自动跳过已完成阶段，并对各阶段使用 --resume。

PY="${PY:-/root/miniconda3/bin/python}"
CODE="${CODE:-/root/autodl-tmp/dcdc_family/code}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-7B.real/Qwen/Qwen2___5-7B}"
OFFICIAL_JSONL="${OFFICIAL_JSONL:-/root/autodl-tmp/datasets/official_power_v1/train.jsonl}"

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
  RUN_DIR="/root/autodl-tmp/dcdc_family/runs/optimal_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/data"
echo "$RUN_DIR" > "$RUN_DIR/RUN_DIR.txt"

log() { echo "[$(date +%F' '%T)] $*"; }

STEP_MARK="$RUN_DIR/steps_done.txt"
touch "$STEP_MARK"
mark_done() { echo "$1" >>"$STEP_MARK"; }
is_done() { grep -Fxq "$1" "$STEP_MARK" 2>/dev/null; }

log "RUN_DIR=$RUN_DIR"
log "BASE_MODEL=$BASE_MODEL"
log "OFFICIAL_JSONL=$OFFICIAL_JSONL"

# 1) 官方数据集预热（>1000条，实际参与训练）
OFFICIAL_SFT="$RUN_DIR/data/official_sft_all.jsonl"
if ! is_done "official_sft_all" || [ ! -s "$OFFICIAL_SFT" ]; then
  log "step=official_sft_all"
  "$PY" "$CODE/filter_official_dcdc_sft.py" \
    --official_jsonl "$OFFICIAL_JSONL" \
    --out_jsonl "$OFFICIAL_SFT" \
    --pattern ".*" \
    --require_inc_prefix \
    >"$RUN_DIR/logs/official_sft_all.log" 2>&1
  wc -l "$OFFICIAL_SFT" | tee "$RUN_DIR/logs/official_sft_all.wc.txt"
  mark_done "official_sft_all"
fi

# 2) 结构先验预热（>=20元件 DC-DC family 模板，仅用于训练）
TPL_SFT="$RUN_DIR/data/sft_dcdc_tpl.jsonl"
if ! is_done "tpl_sft" || [ ! -s "$TPL_SFT" ]; then
  log "step=tpl_sft"
  "$PY" "$CODE/make_sft_dcdc_from_templates.py" \
    --out_jsonl "$TPL_SFT" \
    --template_variant full \
    >"$RUN_DIR/logs/tpl_sft.log" 2>&1
  wc -l "$TPL_SFT" | tee "$RUN_DIR/logs/tpl_sft.wc.txt"
  mark_done "tpl_sft"
fi

# 3) SFT0：官方 + 模板（让模型先会“写对、写长、写>=20元件”）
SFT0_MIX="$RUN_DIR/data/sft0_mix.jsonl"
if ! is_done "sft0_mix" || [ ! -s "$SFT0_MIX" ]; then
  log "step=sft0_mix"
  cat "$OFFICIAL_SFT" "$TPL_SFT" >"$SFT0_MIX"
  wc -l "$SFT0_MIX" | tee "$RUN_DIR/logs/sft0_mix.wc.txt"
  mark_done "sft0_mix"
fi

SFT0_DIR="$RUN_DIR/sft0_stage"
mkdir -p "$SFT0_DIR"
if ! is_done "sft0_train"; then
  log "step=sft0_train"
  "$PY" "$CODE/train_sft_dcdc.py" \
    --base_model "$BASE_MODEL" \
    --sft_jsonl "$SFT0_MIX" \
    --outdir "$SFT0_DIR" \
    --lr 2e-5 \
    --epochs 1 \
    --bsz 1 \
    --grad_accum 16 \
    --save_steps 50 \
    --save_total_limit 3 \
    --resume \
    >"$RUN_DIR/logs/sft0_train.log" 2>&1
  touch "$SFT0_DIR/DONE"
  mark_done "sft0_train"
fi

# 4) Self-play（用 SFT0 权重生成候选 -> 仿真打分 -> 组内相对偏好对）
SELFPLAY_DIR="$RUN_DIR/data/selfplay_inc"
mkdir -p "$SELFPLAY_DIR"
if ! is_done "selfplay_build"; then
  log "step=selfplay_build"
  "$PY" "$CODE/build_selfplay_inc_datasets.py" \
    --base_model "$BASE_MODEL" \
    --adapter "$SFT0_DIR/sft_final" \
    --out_root "$SELFPLAY_DIR" \
    --n_gen 16 \
    --max_rounds 2 \
    --ensure_pass_cv \
    --min_pass_cv 2 \
    --temp_step 0.05 \
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
    --sim_timeout_s 30 \
    --autotune_duty \
    --resume \
    >"$RUN_DIR/logs/selfplay_build.log" 2>&1
  mark_done "selfplay_build"
fi

# 5) SFT1：蒸馏 self-play 高质量轨迹（初始化自 SFT0）
SFT1_MIX="$RUN_DIR/data/sft1_mix.jsonl"
if ! is_done "sft1_mix" || [ ! -s "$SFT1_MIX" ]; then
  log "step=sft1_mix"
  cat "$OFFICIAL_SFT" "$TPL_SFT" "$SELFPLAY_DIR/sft_train.jsonl" >"$SFT1_MIX"
  wc -l "$SFT1_MIX" | tee "$RUN_DIR/logs/sft1_mix.wc.txt"
  mark_done "sft1_mix"
fi

SFT1_DIR="$RUN_DIR/sft1_stage"
mkdir -p "$SFT1_DIR"
if ! is_done "sft1_train"; then
  log "step=sft1_train"
  "$PY" "$CODE/train_sft_dcdc.py" \
    --base_model "$BASE_MODEL" \
    --init_adapter "$SFT0_DIR/sft_final" \
    --sft_jsonl "$SFT1_MIX" \
    --outdir "$SFT1_DIR" \
    --lr 2e-5 \
    --epochs 1 \
    --bsz 1 \
    --grad_accum 16 \
    --save_steps 50 \
    --save_total_limit 3 \
    --resume \
    >"$RUN_DIR/logs/sft1_train.log" 2>&1
  touch "$SFT1_DIR/DONE"
  mark_done "sft1_train"
fi

# 6) DPO：组内相对偏好对齐（INC 口径一致）
DPO_DIR="$RUN_DIR/dpo_stage"
mkdir -p "$DPO_DIR"
if ! is_done "dpo_train"; then
  log "step=dpo_train"
  "$PY" "$CODE/train_dpo_dcdc.py" \
    --base_model "$BASE_MODEL" \
    --sft_adapter "$SFT1_DIR/sft_final" \
    --dpo_pairs "$SELFPLAY_DIR/dpo_pairs.jsonl" \
    --outdir "$DPO_DIR" \
    --lr 5e-6 \
    --epochs 1 \
    --bsz 1 \
    --grad_accum 16 \
    --save_steps 50 \
    --save_total_limit 3 \
    --resume \
    >"$RUN_DIR/logs/dpo_train.log" 2>&1
  touch "$DPO_DIR/DONE"
  mark_done "dpo_train"
fi

# 7) PPO：多目标退火 + 自适应容差 + 风险敏感约束 + 组内相对项（禁止失败样本因归一化变正）
PPO_DIR="$RUN_DIR/ppo_stage"
mkdir -p "$PPO_DIR"
if ! is_done "ppo_train"; then
  log "step=ppo_train"
  "$PY" "$CODE/train_ppo_dcdc.py" \
    --base_model "$BASE_MODEL" \
    --sft_adapter "$DPO_DIR/dpo_final" \
    --outdir "$PPO_DIR" \
    --steps 500 \
    --batch_size 8 \
    --mini_batch_size 4 \
    --grad_accum 2 \
    --lr 5e-6 \
    --target_kl 0.03 \
    --vf_coef 0.5 \
    --ent_coef 0.01 \
    --cliprange 0.2 \
    --cliprange_value 0.2 \
    --ppo_epochs 4 \
    --max_new_tokens 320 \
    --temperature 0.7 \
    --top_p 0.9 \
    --batch_task_mode same_family \
    --group_size 4 \
    --group_reward_mode rank_add \
    --group_rel_coef 0.5 \
    --constrained \
    --autotune_duty \
    --save_steps 25 \
    --save_total_limit 3 \
    --resume \
    >"$RUN_DIR/logs/ppo_train.log" 2>&1
  touch "$PPO_DIR/DONE"
  mark_done "ppo_train"
fi

log "DONE run_dir=$RUN_DIR"

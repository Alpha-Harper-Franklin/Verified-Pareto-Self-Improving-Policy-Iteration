#!/usr/bin/env bash
set -euo pipefail

# "4点创新包" 训练流水线（在 CA800 上使用）
# 1) 官方数据集预热（>1000 条，实际参与训练）
# 2) 工业EDA式结构先验：模块图（design operators）→ 编译成 INC → SPICE 评分 + 参数优化
# 3) 组内相对学习：由同题多解构造偏好对（DPO）+ PPO 内部 GRPO-style 组内相对项
# 4) PPO：多目标权重退火 + 自适应容差 + 风险敏感约束（对齐 sample 级成功率）
#
# 用法：
#   bash run_innov4_ppo_pipeline.sh [/root/autodl-tmp/dcdc_family/runs/innov4_YYYYmmdd_HHMMSS]
#
# 断点续训：
#   重复执行同一 RUN_DIR，会基于 steps_done.txt 与各阶段 --resume 自动续跑。

PY="${PY:-/root/miniconda3/bin/python}"
CODE="${CODE:-/root/autodl-tmp/dcdc_family/code}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-7B.real/Qwen/Qwen2___5-7B}"
OFFICIAL_JSONL="${OFFICIAL_JSONL:-/root/autodl-tmp/datasets/official_power_v1/train.jsonl}"

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
  RUN_DIR="/root/autodl-tmp/dcdc_family/runs/innov4_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/data"
echo "$RUN_DIR" > "$RUN_DIR/RUN_DIR.txt"

log() { echo "[$(date +%F' '%T)] $*"; }

STEP_MARK="$RUN_DIR/steps_done.txt"
touch "$STEP_MARK"
mark_done() { echo "$1" >>"$STEP_MARK"; }
is_done() { grep -Fxq "$1" "$STEP_MARK" 2>/dev/null; }

# 0) quick sanity
log "RUN_DIR=$RUN_DIR"
log "BASE_MODEL=$BASE_MODEL"
log "OFFICIAL_JSONL=$OFFICIAL_JSONL"

# 1) 官方数据集（全部行，保证 official>1000 且真实参与训练）
OFFICIAL_SFT="$RUN_DIR/data/official_sft_all.jsonl"
if ! is_done "official_sft_all"; then
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

# 2) 结构先验预热（DC-DC family >=20元件模板，只用于训练，不用于评测兜底）
TPL_SFT="$RUN_DIR/data/sft_dcdc_tpl.jsonl"
if ! is_done "tpl_sft"; then
  log "step=tpl_sft"
  "$PY" "$CODE/make_sft_dcdc_from_templates.py" \
    --out_jsonl "$TPL_SFT" \
    --template_variant full \
    >"$RUN_DIR/logs/tpl_sft.log" 2>&1
  wc -l "$TPL_SFT" | tee "$RUN_DIR/logs/tpl_sft.wc.txt"
  mark_done "tpl_sft"
fi

# 3) 工业EDA式自举数据：模块图（design operators）→ 编译 → 参数优化 → SPICE 评分 → (SFT + DPO pairs)
MOD_DIR="$RUN_DIR/data/modulegraph_data"
mkdir -p "$MOD_DIR"
if ! is_done "modulegraph_build"; then
  log "step=modulegraph_build"
  "$PY" "$CODE/build_opt_modulegraph_datasets.py" \
    --base_model "$BASE_MODEL" \
    --out_root "$MOD_DIR" \
    --n_gen 4 \
    --pairs_per_task 2 \
    --min_pair_gap 0.10 \
    --min_mods 6 \
    --min_elems 20 \
    --opt_budget 40 \
    --opt_pop 10 \
    --opt_elite 3 \
    --robust \
    --vin_jitter 0.10 \
    --rload_list "5,10,20" \
    --agg cvar \
    --cvar_alpha 0.25 \
    --resume \
    >"$RUN_DIR/logs/modulegraph_build.log" 2>&1
  wc -l "$MOD_DIR/sft_train.jsonl" | tee "$RUN_DIR/logs/modulegraph_sft.wc.txt"
  wc -l "$MOD_DIR/dpo_pairs.jsonl" | tee "$RUN_DIR/logs/modulegraph_dpo.wc.txt"
  mark_done "modulegraph_build"
fi

# 4) 合并 SFT 数据（official + templates + modulegraph self-play）
SFT_MIX="$RUN_DIR/data/sft_mix.jsonl"
if ! is_done "sft_mix"; then
  log "step=sft_mix"
  cat "$OFFICIAL_SFT" "$TPL_SFT" "$MOD_DIR/sft_train.jsonl" >"$SFT_MIX"
  wc -l "$SFT_MIX" | tee "$RUN_DIR/logs/sft_mix.wc.txt"
  mark_done "sft_mix"
fi

# 5) SFT（逐步 loss 记录 + 断点续训）
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

# 6) DPO（组内相对/偏好蒸馏，逐步 loss 记录 + 断点续训）
DPO_DIR="$RUN_DIR/dpo_stage"
mkdir -p "$DPO_DIR"
if ! is_done "dpo_train"; then
  log "step=dpo_train"
  "$PY" "$CODE/train_dpo_dcdc.py" \
    --base_model "$BASE_MODEL" \
    --sft_adapter "$SFT_DIR/sft_final" \
    --dpo_pairs "$MOD_DIR/dpo_pairs.jsonl" \
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

# 7) PPO（GRPO-style 组内相对项 + 多目标退火 + 自适应容差 + 风险敏感约束；逐步 reward/loss 分项落盘）
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


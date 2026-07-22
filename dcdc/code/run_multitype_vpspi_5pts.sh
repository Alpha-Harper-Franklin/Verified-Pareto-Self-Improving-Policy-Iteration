#!/usr/bin/env bash
set -euo pipefail

# 5-point (reduced-difficulty) curriculum run.
# - Uses new tasksets (amp/filter/osc): single main target variable with 5 point-spec values.
# - Keeps the same VP-SPI pipeline (A->B->C->D) and multi-GPU settings.

export ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/vpspi_multitype_v3_5pts}"
export ANCHOR_N="${ANCHOR_N:-8000}"
export ANCHOR_MIN_ELEMS="${ANCHOR_MIN_ELEMS:-20}"
export ANCHOR_SEED="${ANCHOR_SEED:-20260115}"
export DDP_BACKEND="${DDP_BACKEND:-gloo}"

exec "$(dirname "$0")/run_multitype_vpspi_from_scratch.sh" "$@"

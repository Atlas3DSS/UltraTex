#!/usr/bin/env bash
# Foreground-aware decoder fine-tune. Usage: BACKBONE=qwen21|krea2 bash scripts/train_vae_mv.sh [extra args]
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/mv_env.sh
BACKBONE=${BACKBONE:-qwen21}
EXTRA=()
[ -n "${DATA_ROOT:-}" ] && EXTRA+=(--data_root "${DATA_ROOT}")
exec "${MV_PYTHON}" train_vae/train_decoder_mv.py --backbone "${BACKBONE}" \
    --train_data_json "${TRAIN_JSON:-data/train_demo.json}" --eval_data_json "${EVAL_JSON:-data/eval_demo.json}" \
    --resolution "${RESOLUTION:-2048}" --lr "${LR:-1e-5}" --max_train_steps "${MAX_TRAIN_STEPS:-3000}" \
    --output_dir "${OUTPUT_DIR:-outputs/train_vae_${BACKBONE}}" "${EXTRA[@]}" "$@"

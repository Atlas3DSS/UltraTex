#!/usr/bin/env bash
# Usage: BACKBONE=qwen21|krea2 LORA=outputs/.../checkpoint-N [DECODER_CKPT=...] bash scripts/inference_mv.sh [extra args]
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/mv_env.sh
BACKBONE=${BACKBONE:-qwen21}
RESOLUTION=${RESOLUTION:-2048}
TEXT_COND=${TEXT_COND:-checkpoints/${BACKBONE}/text_cond.safetensors}
EVAL_JSON=${EVAL_JSON:-data/eval_demo.json}
OUT=${OUT:-outputs/inference_${BACKBONE}}
EXTRA=()
[ -n "${LORA:-}" ] && EXTRA+=(--lora_path "${LORA}")
[ -n "${DECODER_CKPT:-}" ] && EXTRA+=(--decoder_ckpt "${DECODER_CKPT}")
[ -n "${DATA_ROOT:-}" ] && EXTRA+=(--data_root "${DATA_ROOT}")
exec "${MV_PYTHON}" inference_mv.py --backbone "${BACKBONE}" --text_cond_path "${TEXT_COND}" --eval_data_json "${EVAL_JSON}" \
    --resolution "${RESOLUTION}" --project_dir "${OUT}" "${EXTRA[@]}" "$@"

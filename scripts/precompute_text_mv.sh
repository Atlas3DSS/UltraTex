#!/usr/bin/env bash
# Encode the (empty) texturing prompt once per backbone with ComfyUI's text encoders.
# Usage: BACKBONE=qwen21|krea2 bash scripts/precompute_text_mv.sh [--prompt "..."]...
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/mv_env.sh
BACKBONE=${BACKBONE:-qwen21}
case "${BACKBONE}" in
    krea2) TE=${KREA2_TEXT_ENCODER} ;;
    qwen21) TE=${QWEN21_TEXT_ENCODER} ;;
    *) echo "BACKBONE must be krea2 or qwen21" >&2; exit 2 ;;
esac
OUT=${OUT:-checkpoints/${BACKBONE}/text_cond.safetensors}
exec "${MV_PYTHON}" tools/precompute_text_conditioning.py --backbone "${BACKBONE}" --text-encoder "${TE}" --out "${OUT}" "$@"

#!/usr/bin/env bash
# UltraTex albedo LoRA training on Krea 2 or Qwen-Image 2.1.
# Usage: BACKBONE=qwen21|krea2 [RESOLUTION=2048] [TRAIN_JSON=...] bash scripts/train_mv.sh [extra trainer args]
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/mv_env.sh
BACKBONE=${BACKBONE:-qwen21}
RESOLUTION=${RESOLUTION:-2048}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1000000}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-2000}
TRAIN_JSON=${TRAIN_JSON:-data/train_demo.json}
EVAL_JSON=${EVAL_JSON:-data/eval_demo.json}
TEXT_COND=${TEXT_COND:-checkpoints/${BACKBONE}/text_cond.safetensors}
DECODER_CKPT=${DECODER_CKPT:-}
PROJECT_DIR=${PROJECT_DIR:-outputs/train_${BACKBONE}_albedo_res=${RESOLUTION}_lr=${LEARNING_RATE}}
SPARSE_ATTENTION=${SPARSE_ATTENTION:-true}
WORKERS=${WORKERS:-8}

for f in "${TEXT_COND}" "${TRAIN_JSON}" "${EVAL_JSON}"; do
    [ -f "$f" ] || { echo "missing required file: $f (text cache: run scripts/precompute_text_mv.sh)" >&2; exit 1; }
done
EXTRA=()
[ -n "${DECODER_CKPT}" ] && EXTRA+=(--decoder_ckpt "${DECODER_CKPT}")
[ -n "${BUCKET_METADATA_JSON:-}" ] && EXTRA+=(--bucket_metadata_json "${BUCKET_METADATA_JSON}")
[ -n "${DATA_ROOT:-}" ] && EXTRA+=(--data_root "${DATA_ROOT}")
if [ -z "${RESUME_FROM_CHECKPOINT+x}" ]; then
    LATEST=$(ls -d "${PROJECT_DIR}"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
    [ -n "${LATEST}" ] && RESUME_FROM_CHECKPOINT="${PROJECT_DIR}/checkpoint-${LATEST}"
fi
[ -n "${RESUME_FROM_CHECKPOINT:-}" ] && EXTRA+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")

echo "backbone=${BACKBONE} resolution=${RESOLUTION} lr=${LEARNING_RATE} steps=${MAX_TRAIN_STEPS} out=${PROJECT_DIR} resume=${RESUME_FROM_CHECKPOINT:-<none>}"
exec "${MV_PYTHON}" train_mv.py \
    --backbone "${BACKBONE}" \
    --text_cond_path "${TEXT_COND}" \
    --train_data_json "${TRAIN_JSON}" \
    --eval_data_json "${EVAL_JSON}" \
    --resolution "${RESOLUTION}" \
    --learning_rate "${LEARNING_RATE}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --checkpointing_steps "${CHECKPOINTING_STEPS}" \
    --num_workers "${WORKERS}" \
    --project_dir "${PROJECT_DIR}" \
    --batch_size 1 --lora_rank 64 \
    --sparse_attention "${SPARSE_ATTENTION}" --sparse_attention_topk 0.20 --sparse_attention_blkq 128 --sparse_attention_blkk 64 \
    --inference_num_steps 25 --inference_drop_background_tokens true --inference_remove_background true \
    "${EXTRA[@]}" "$@"

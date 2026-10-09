#!/usr/bin/env bash
# Texture ("skin") one mesh from one reference image with UltraTex, end to end:
#   render six G-buffer views -> UltraTex multi-view albedo (released FLUX.2 weights by default)
#   -> back-project to the UV texture -> export a textured GLB.
# Usage: scripts/skin_mesh.sh MESH.glb REFERENCE.png OUT_DIR [NAME]
# Env: BACKEND=flux2 (default; released checkpoints/flux2) | qwen21 | krea2 (our LoRAs: LORA=..., DECODER_CKPT=...)
#      RESOLUTION=1024 (per view), STEPS=25, GUIDANCE=4.0, TEXTURE_SIZE=2048
set -Eeuo pipefail
MESH=$1; REF=$2; OUT=$3; NAME=${4:-$(basename "${MESH%.*}")}
cd "$(dirname "$0")/.."
source scripts/mv_env.sh
BLENDER=${BLENDER:-/home/orwel/.local/opt/blender-4.2.23-linux-x64/blender}
BACKEND=${BACKEND:-flux2}
RESOLUTION=${RESOLUTION:-1024}
STEPS=${STEPS:-25}
GUIDANCE=${GUIDANCE:-4.0}
TEXTURE_SIZE=${TEXTURE_SIZE:-2048}
ASSET="${OUT}/${NAME}"
mkdir -p "${ASSET}"

echo "== [1/4] G-buffer render -> ${ASSET}"
"${BLENDER}" -b -P tools/render_gbuffer.py -- --mesh "${MESH}" --out "${ASSET}" --res "${RESOLUTION}" 2>&1 | grep -E "GBUFFER_OK|Error" || { echo "render failed" >&2; exit 1; }
# the AI-mesh loader wants the reference next to the asset as <id>_resize.webp (RGBA; alpha = foreground)
"${MV_PYTHON}" - "${REF}" "${ASSET}/${NAME}_resize.webp" "${RESOLUTION}" <<'PY'
import sys; from PIL import Image
src, dst, res = sys.argv[1], sys.argv[2], int(sys.argv[3])
im = Image.open(src).convert("RGBA")
if im.getextrema()[3][0] == 255:  # no alpha: treat near-white/gray flat background as transparent
    import numpy as np
    a = np.asarray(im).astype(int); bg = a[0, 0, :3]
    diff = np.abs(a[..., :3] - bg).sum(-1)
    alpha = (diff > 40).astype("uint8") * 255
    import cv2
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    im.putalpha(Image.fromarray(alpha))
im = im.resize((res, res), Image.LANCZOS)
im.save(dst, format="WEBP", lossless=True)
PY
cat > "${ASSET}/eval.json" <<JSON
[{"prompt": "", "image_dir": "${ASSET}", "id": "${NAME}", "ref_image_path": "${ASSET}/${NAME}_resize.webp"}]
JSON

echo "== [2/4] UltraTex inference (${BACKEND})"
if [ "${BACKEND}" = flux2 ]; then
    export AE_MODEL_PATH=checkpoints/base/flux2/ae.safetensors KLEIN_9B_BASE_MODEL_PATH=checkpoints/base/flux2/flux-2-klein-base-9b.safetensors QWEN3_8B_PATH=checkpoints/base/flux2/qwen3-8b
    DRY=(); [ -n "${DRY_RUN:-}" ] && DRY=(--dry_run true)
    AI_JSON="${ASSET}/eval.json" AI_OUTPUT="${ASSET}/ultratex" bash scripts/inference_flux2.sh ai --resolution "${RESOLUTION}" --inference_num_steps "${STEPS}" --guidance "${GUIDANCE}" --output_format webp "${DRY[@]}"
    [ -n "${DRY_RUN:-}" ] && { echo "DRY_RUN_OK"; exit 0; }
    CANVAS=$(ls "${ASSET}"/ultratex/*/*"${NAME}"*_result.webp "${ASSET}"/ultratex/*"${NAME}"*_result.webp 2>/dev/null | head -1 || true)
else
    BACKBONE="${BACKEND}" RESOLUTION="${RESOLUTION}" EVAL_JSON="${ASSET}/eval.json" OUT="${ASSET}/ultratex" bash scripts/inference_mv.sh --inference_num_steps "${STEPS}" --dataset_type ai
    CANVAS=$(ls "${ASSET}"/ultratex/result_*.webp | head -1 || true)
fi
[ -f "${CANVAS}" ] || { echo "no albedo canvas produced under ${ASSET}/ultratex" >&2; ls "${ASSET}/ultratex" >&2; exit 1; }

echo "== [3/4] bake -> ${ASSET}/albedo_baked.png"
"${MV_PYTHON}" tools/bake_views_to_uv.py --asset-dir "${ASSET}" --canvas "${CANVAS}" --out "${ASSET}/albedo_baked.png" --size "${TEXTURE_SIZE}"

echo "== [4/4] export -> ${ASSET}/${NAME}_skinned.glb"
"${BLENDER}" -b -P tools/apply_texture.py -- --mesh "${MESH}" --texture "${ASSET}/albedo_baked.png" --out "${ASSET}/${NAME}_skinned.glb" 2>&1 | grep -E "APPLY_OK|Error"
echo "SKIN_OK ${ASSET}/${NAME}_skinned.glb"

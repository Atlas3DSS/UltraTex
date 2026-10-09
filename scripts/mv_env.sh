# Environment for the Krea 2 / Qwen-Image 2.1 UltraTex port. Source from the repository root.
# Every value can be overridden before sourcing. Defaults match the Atlas devbox (wtatum84).
export COMFY_ROOT=${COMFY_ROOT:-/home/orwel/.local/opt/frame-comfy-qwen21}
export MV_PYTHON=${MV_PYTHON:-${COMFY_ROOT}/.venv/bin/python}
COMFY_MODELS=${COMFY_MODELS:-/home/orwel/dev_genius/ComfyUI/models}

export KREA2_MODEL_PATH=${KREA2_MODEL_PATH:-${COMFY_MODELS}/diffusion_models/KREA/krea2_raw_bf16.safetensors}
export KREA2_VAE_PATH=${KREA2_VAE_PATH:-${COMFY_MODELS}/vae/qwen_image_vae.safetensors}
export KREA2_TEXT_ENCODER=${KREA2_TEXT_ENCODER:-${COMFY_MODELS}/text_encoders/qwen3vl_4b_bf16.safetensors}

export QWEN21_MODEL_PATH=${QWEN21_MODEL_PATH:-${COMFY_MODELS}/diffusion_models/qwen_image_2.1_bf16.safetensors}
export QWEN21_VAE_PATH=${QWEN21_VAE_PATH:-${COMFY_MODELS}/vae/qwen_image_2.1_vae_bf16.safetensors}
export QWEN21_TEXT_ENCODER=${QWEN21_TEXT_ENCODER:-${COMFY_MODELS}/text_encoders/qwen3vl_8b_bf16.safetensors}

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

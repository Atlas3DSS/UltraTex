"""Krea 2 loading helpers and multi-view sequence layout."""

from __future__ import annotations

import os

import torch
from safetensors.torch import load_file
from torch import Tensor

from ultratex.backbones.common import WAN21_NORM, ComfyVAE, patchify_tokens
from ultratex.backbones.krea2.model import Krea2, Krea2Params

ENV_MODEL_PATH = "KREA2_MODEL_PATH"
ENV_VAE_PATH = "KREA2_VAE_PATH"


def load_flow_model(model_path: str | None = None, device="cpu", dtype=torch.bfloat16, debug_mode: bool = False) -> Krea2:
    """Load Krea 2 Raw (the undistilled checkpoint; train on Raw, run on Turbo if you must)."""
    params = Krea2Params()
    if debug_mode:
        params.layers = 1
        with torch.device(device):
            return Krea2(params).to(dtype)
    path = model_path or os.environ.get(ENV_MODEL_PATH)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Krea 2 weights not found (set {ENV_MODEL_PATH}): {path!r}")
    with torch.device("meta"):
        model = Krea2(params)
    sd = load_file(path, device=str(device))
    model.load_state_dict(sd, strict=True, assign=True)
    return model.to(device=device, dtype=dtype)


def load_ae(vae_path: str | None = None, device="cpu", dtype=torch.float32, comfy_root: str | None = None) -> ComfyVAE:
    path = vae_path or os.environ.get(ENV_VAE_PATH)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Krea 2 (Qwen-Image / Wan 2.1) VAE not found (set {ENV_VAE_PATH}): {path!r}")
    return ComfyVAE(path, WAN21_NORM, image_channels=3, comfy_root=comfy_root, device=device, dtype=dtype)


def pack_image(latent: Tensor, index: int, patch: int = 2) -> tuple[Tensor, Tensor]:
    """[B, 16, H, W] normalised latent -> tokens [B, N, 64] and ids [B, N, 3] = (index, h, w).

    Absolute (h, w) coordinates on the 2x2-patch token grid, as the Krea 2
    reference-latent convention does; ``index`` separates images on axis 0.
    """
    tokens, h, w = patchify_tokens(latent, patch)
    ids = torch.zeros(h, w, 3, device=latent.device, dtype=torch.float32)
    ids[..., 0] = index
    ids[..., 1] = torch.arange(h, device=latent.device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w, device=latent.device, dtype=torch.float32)[None, :]
    return tokens, ids.reshape(1, h * w, 3).expand(latent.shape[0], -1, -1).contiguous()


def lora_target_modules(model: Krea2, block_indices: list[int] | None = None) -> list[str]:
    indices = list(range(len(model.blocks))) if block_indices is None else block_indices
    targets = []
    for i in indices:
        if not 0 <= i < len(model.blocks):
            raise ValueError(f"block index {i} is invalid for Krea 2 ({len(model.blocks)} blocks)")
        p = f"blocks.{i}"
        targets += [f"{p}.attn.wq", f"{p}.attn.wk", f"{p}.attn.wv", f"{p}.attn.wo", f"{p}.attn.gate",
                    f"{p}.mlp.gate", f"{p}.mlp.up", f"{p}.mlp.down"]
    return targets

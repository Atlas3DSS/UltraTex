"""Qwen-Image 2.1 loading helpers and multi-view sequence layout."""

from __future__ import annotations

import os

import torch
from safetensors.torch import load_file
from torch import Tensor

from ultratex.backbones.common import QWEN21_NORM, ComfyVAE, patchify_tokens
from ultratex.backbones.qwen21.model import Qwen21Params, QwenImage21

ENV_MODEL_PATH = "QWEN21_MODEL_PATH"
ENV_VAE_PATH = "QWEN21_VAE_PATH"


def load_flow_model(model_path: str | None = None, device="cpu", dtype=torch.bfloat16, debug_mode: bool = False) -> QwenImage21:
    params = Qwen21Params()
    if debug_mode:
        params.num_layers = 1
        with torch.device(device):
            return QwenImage21(params).to(dtype)
    path = model_path or os.environ.get(ENV_MODEL_PATH)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Qwen-Image 2.1 weights not found (set {ENV_MODEL_PATH}): {path!r}")
    with torch.device("meta"):
        model = QwenImage21(params)
    sd = load_file(path, device=str(device))
    model.load_state_dict(sd, strict=True, assign=True)
    return model.to(device=device, dtype=dtype)


def load_ae(vae_path: str | None = None, device="cpu", dtype=torch.float32, comfy_root: str | None = None) -> ComfyVAE:
    path = vae_path or os.environ.get(ENV_VAE_PATH)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Qwen-Image 2.1 VAE not found (set {ENV_VAE_PATH}): {path!r}")
    return ComfyVAE(path, QWEN21_NORM, image_channels=4, comfy_root=comfy_root, device=device, dtype=dtype)


def pack_image(latent: Tensor, frame: float, target_hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """[B, 64, H, W] normalised latent -> tokens [B, H*W, 64] and ids (frame, h, w).

    Spatial ids are centred on the token grid (``h - (h - h//2)``) with the
    half-token parity shift Qwen-Image 2.1 uses so references centre on the
    target; ``frame`` is the axis-0 position reserved for this image.
    """
    tokens, h, w = patchify_tokens(latent, 1)
    th, tw = target_hw
    hh = torch.arange(h, device=latent.device, dtype=torch.float32) - (h - h // 2) + 0.5 * (h % 2 - th % 2)
    ww = torch.arange(w, device=latent.device, dtype=torch.float32) - (w - w // 2) + 0.5 * (w % 2 - tw % 2)
    ids = torch.stack(
        [torch.full((h, w), float(frame), device=latent.device), hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1
    )
    return tokens, ids.reshape(1, h * w, 3).expand(latent.shape[0], -1, -1).contiguous()


def lora_target_modules(model: QwenImage21, block_indices: list[int] | None = None) -> list[str]:
    indices = list(range(len(model.transformer_blocks))) if block_indices is None else block_indices
    targets = []
    for i in indices:
        if not 0 <= i < len(model.transformer_blocks):
            raise ValueError(f"block index {i} is invalid for Qwen-Image 2.1 ({len(model.transformer_blocks)} blocks)")
        p = f"transformer_blocks.{i}"
        targets += [f"{p}.attn.to_q", f"{p}.attn.to_k", f"{p}.attn.to_v", f"{p}.attn.to_out.0",
                    f"{p}.img_mlp.gate_up", f"{p}.img_mlp.out"]
    return targets

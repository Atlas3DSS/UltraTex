"""Multi-view texturing pipeline shared by the Krea 2 and Qwen-Image 2.1 backbones.

This is the backbone-agnostic half of UltraTex: TexVerse 2x3 view grids are
VAE-encoded tile by tile, packed into token sequences with per-token RoPE ids,
background tokens are dropped (BTD) while the survivors keep their ids, the DiT
predicts the flow-matching velocity for the retained target tokens, and at
inference the generated foreground is scattered into a canonical background
latent before the (foreground-aware) decoder reconstructs the views.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Callable

import torch
from einops import rearrange
from safetensors.torch import load_file
from torch import Tensor

from ultratex.backbones import common
from ultratex.backbones.common import (
    ComfyVAE,
    LatentNorm,
    foreground_indices,
    gather_sequence,
    qwen21_mu,
    scatter_sequence,
    shifted_schedule,
    unpatchify_tokens,
)

GRID_ROWS, GRID_COLS = 2, 3
TOKEN_STRIDE = 16  # pixels per token for both backbones (8x VAE * 2x2 patch, or 16x VAE * 1x1 patch)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    patch: int
    latent_channels: int
    image_channels: int  # 3 (RGB) or 4 (RGBA)
    norm: LatentNorm
    text_dim: int
    default_attention: str  # "full" or "block_causal"
    default_background: str  # "black" or "white"
    supports_prefix_cache: bool


SPECS = {
    "krea2": BackboneSpec("krea2", 2, 16, 3, common.WAN21_NORM, 12 * 2560, "full", "black", False),
    "qwen21": BackboneSpec("qwen21", 1, 64, 4, common.QWEN21_NORM, 4096, "block_causal", "white", True),
}


def get_spec(name: str) -> BackboneSpec:
    try:
        return SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unknown backbone {name!r}; expected one of {sorted(SPECS)}") from exc


def load_backbone(name: str, model_path: str | None, device, dtype=torch.bfloat16, debug_mode: bool = False):
    if name == "krea2":
        from ultratex.backbones.krea2.util import load_flow_model
    elif name == "qwen21":
        from ultratex.backbones.qwen21.util import load_flow_model
    else:
        get_spec(name)
    return load_flow_model(model_path, device=device, dtype=dtype, debug_mode=debug_mode)


def load_vae(name: str, vae_path: str | None, device, dtype=torch.float32, comfy_root: str | None = None) -> ComfyVAE:
    if name == "krea2":
        from ultratex.backbones.krea2.util import load_ae
    elif name == "qwen21":
        from ultratex.backbones.qwen21.util import load_ae
    else:
        get_spec(name)
    return load_ae(vae_path, device=device, dtype=dtype, comfy_root=comfy_root)


def lora_targets(name: str, model, block_indices=None) -> list[str]:
    if name == "krea2":
        from ultratex.backbones.krea2.util import lora_target_modules
    else:
        from ultratex.backbones.qwen21.util import lora_target_modules
    return lora_target_modules(model, block_indices)


# ----------------------------------------------------------------------------
# Text conditioning cache (prompts are encoded once with ComfyUI's text encoders)
# ----------------------------------------------------------------------------


def prompt_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


class TextCache:
    """``tools/precompute_text_conditioning.py`` output: one tensor per prompt.

    The safetensors file maps ``sha1(prompt)`` to ``[L, D]`` features; the
    sidecar ``.json`` records the backbone and the prompt list so a mismatch
    fails loudly instead of silently conditioning on the wrong encoder.
    """

    def __init__(self, path: str, backbone: str, device):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"text conditioning cache not found: {path}")
        meta_path = os.path.splitext(path)[0] + ".json"
        with open(meta_path, "rt", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("backbone") != backbone:
            raise ValueError(f"{path} was encoded for backbone {meta.get('backbone')!r}, not {backbone!r}")
        self.tensors = {k: v.to(device) for k, v in load_file(path).items()}
        self.prompts = {prompt_key(p): p for p in meta["prompts"]}
        missing = set(self.prompts) - set(self.tensors)
        if missing:
            raise ValueError(f"{path} is missing features for prompts {[self.prompts[k] for k in missing]}")

    def get(self, prompts: list[str], dtype) -> Tensor:
        feats = []
        for prompt in prompts:
            key = prompt_key(prompt)
            if key not in self.tensors:
                raise KeyError(
                    f"prompt {prompt!r} is not in the text cache; re-run tools/precompute_text_conditioning.py with it"
                )
            feats.append(self.tensors[key])
        if any(f.shape != feats[0].shape for f in feats):
            raise ValueError("prompts in one batch must have equal token counts (use batch_size=1 or identical prompts)")
        return torch.stack(feats).to(dtype)


# ----------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------


def with_alpha(img: Tensor, mask_full: Tensor) -> Tensor:
    """Append an alpha channel in [-1, 1] built from the full-resolution foreground mask."""
    alpha = (mask_full.to(img.dtype) > 0).to(img.dtype) * 2 - 1
    if alpha.ndim == 2:
        alpha = alpha[None, None]
    elif alpha.ndim == 3:
        alpha = alpha[:, None]
    return torch.cat([img, alpha.expand(img.shape[0], 1, -1, -1).to(img.device)], dim=1)


@torch.no_grad()
def encode_grid(vae: ComfyVAE, img: Tensor, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> Tensor:
    """VAE-encode a 2x3 view canvas tile by tile (each view is encoded on its own)."""
    batch = img.shape[0]
    tiles = rearrange(img, "b c (r h) (co w) -> (b r co) c h w", r=rows, co=cols)
    z = torch.cat([vae.encode(tile[None]) for tile in tiles], dim=0)
    return rearrange(z, "(b r co) c h w -> b c (r h) (co w)", b=batch, r=rows, co=cols)


@torch.no_grad()
def decode_grid(vae: ComfyVAE, z: Tensor, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> Tensor:
    decoded_rows = []
    for latent_row in torch.chunk(z, rows, dim=-2):
        decoded_rows.append(torch.cat([vae.decode(tile) for tile in torch.chunk(latent_row, cols, dim=-1)], dim=-1))
    return torch.cat(decoded_rows, dim=-2)


def canonical_background_latent(vae: ComfyVAE, spec: BackboneSpec, resolution: int, background: str) -> Tensor:
    """One VAE tile of the canonical background, repeated to the 2x3 grid (normalised latent)."""
    value = {"black": -1.0, "white": 1.0}[background]
    pixels = torch.full((1, 3, resolution, resolution), value, device=vae.device)
    if spec.image_channels == 4:
        pixels = torch.cat([pixels, torch.full((1, 1, resolution, resolution), -1.0, device=vae.device)], dim=1)
    with torch.no_grad():
        z = vae.encode(pixels)
    return z.repeat(1, 1, GRID_ROWS, GRID_COLS)


# ----------------------------------------------------------------------------
# Sequence construction with BTD
# ----------------------------------------------------------------------------


@dataclass
class MVSequence:
    """Everything the DiT needs for one multi-view sample (tokens are background-pruned)."""

    target_tokens: Tensor  # [B, Kt, C] clean target tokens (training) or placeholder (inference)
    target_ids: Tensor  # [B, Kt, 3]
    target_indices: Tensor  # [B, Kt] positions in the full target token grid
    target_token_count: int  # full (unpruned) token count of the target grid
    ref_tokens: list[Tensor]  # per reference, [B, Kr_i, C]
    ref_ids: list[Tensor]
    latent_hw: tuple[int, int]  # target latent grid (h, w)
    token_hw: tuple[int, int]  # target token grid (h, w)
    ctx: Tensor


class MVPipeline:
    def __init__(self, spec: BackboneSpec, vae: ComfyVAE, text_cache: TextCache | None, device, weight_dtype=torch.bfloat16,
                 attention: str | None = None, ref_timestep_zero: bool = False):
        self.spec = spec
        self.vae = vae
        self.text_cache = text_cache
        self.device = device
        self.weight_dtype = weight_dtype
        self.attention = attention or spec.default_attention
        self.ref_timestep_zero = ref_timestep_zero
        if spec.name == "krea2":
            from ultratex.backbones.krea2.util import pack_image

            self._pack = lambda latent, index, target_hw: pack_image(latent, index, spec.patch)
        else:
            from ultratex.backbones.qwen21.util import pack_image

            self._pack = pack_image

    # -- encoding -------------------------------------------------------------
    def encode_target(self, img: Tensor, mask_full: Tensor | None) -> Tensor:
        img = img.to(self.device)
        if self.spec.image_channels == 4:
            if mask_full is None:
                raise ValueError("RGBA backbones need the full-resolution target mask to build alpha")
            img = with_alpha(img, mask_full.to(self.device))
        return encode_grid(self.vae, img)

    def encode_reference(self, img: Tensor, mask_full: Tensor | None) -> Tensor:
        img = img.to(self.device)
        if self.spec.image_channels == 4:
            if mask_full is None:
                alpha = torch.ones_like(img[:, :1])
                img = torch.cat([img, alpha], dim=1)
            else:
                img = with_alpha(img, mask_full.to(self.device))
        h, w = img.shape[-2:]
        if h == w:
            with torch.no_grad():
                return self.vae.encode(img)
        return encode_grid(self.vae, img)

    # -- ids ---------------------------------------------------------------
    def _layout(self, target_latent: Tensor, ref_latents: list[Tensor], txt_len: int):
        """Pack target + refs into tokens with backbone-specific RoPE ids."""
        th, tw = target_latent.shape[-2] // self.spec.patch, target_latent.shape[-1] // self.spec.patch
        if self.spec.name == "krea2":
            target_tokens, target_ids = self._pack(target_latent, 0, (th, tw))
            refs = [self._pack(r, i + 1, (th, tw)) for i, r in enumerate(ref_latents)]
        else:
            # text occupies frames [0, txt_len); every image gets its own frame and advances by max(h, w)
            pos = txt_len
            refs = []
            for r in ref_latents:
                tokens, ids = self._pack(r, pos, (th, tw))
                refs.append((tokens, ids))
                pos += max(r.shape[-2], r.shape[-1])
            target_tokens, target_ids = self._pack(target_latent, pos, (th, tw))
        return target_tokens, target_ids, refs, (th, tw)

    def build(self, batch: dict, drop_background: bool = True, target_latent: Tensor | None = None) -> MVSequence:
        """Encode a dataset batch and prune background tokens.

        ``batch`` follows ``FluxPairedDatasetV2.collate_fn``: ``img`` (target
        canvas), ``ref_imgs`` ([normal canvas, reference render]), ``masks``
        (token-grid masks: target, then one per reference), ``masks_full``.
        """
        prompts = list(batch["txt"])
        if self.text_cache is None:
            raise ValueError("a text conditioning cache is required")
        ctx = self.text_cache.get(prompts, self.weight_dtype)
        txt_len = ctx.shape[1]

        mask_full = torch.stack([m[0] for m in batch["masks_full"]]) if self.spec.image_channels == 4 else None
        if target_latent is None:
            target_latent = self.encode_target(batch["img"], mask_full)
        ref_latents = []
        for ref_index, ref_img in enumerate(batch["ref_imgs"]):
            ref_mask_full = None
            if self.spec.image_channels == 4:
                ref_mask_full = _ref_mask_full(batch, ref_index, ref_img.shape[-2:])
            ref_latents.append(self.encode_reference(ref_img, ref_mask_full))

        target_tokens, target_ids, refs, token_hw = self._layout(target_latent, ref_latents, txt_len)
        n_target = target_tokens.shape[1]
        masks = batch["masks"]
        expected = 1 + len(refs)
        if any(len(m) != expected for m in masks):
            raise ValueError(f"expected {expected} masks per sample (target + references)")

        if drop_background:
            target_indices = foreground_indices(torch.stack([m[0] for m in masks]).to(self.device), n_target, "target")
            ref_indices = [
                foreground_indices(torch.stack([m[i + 1] for m in masks]).to(self.device), refs[i][0].shape[1], f"reference[{i}]")
                for i in range(len(refs))
            ]
        else:
            target_indices = torch.arange(n_target, device=self.device)[None].expand(target_tokens.shape[0], -1)
            ref_indices = [torch.arange(t.shape[1], device=self.device)[None].expand(t.shape[0], -1) for t, _ in refs]

        return MVSequence(
            target_tokens=gather_sequence(target_tokens, target_indices),
            target_ids=gather_sequence(target_ids, target_indices),
            target_indices=target_indices,
            target_token_count=n_target,
            ref_tokens=[gather_sequence(t, idx) for (t, _), idx in zip(refs, ref_indices)],
            ref_ids=[gather_sequence(i, idx) for (_, i), idx in zip(refs, ref_indices)],
            latent_hw=tuple(target_latent.shape[-2:]),
            token_hw=token_hw,
            ctx=ctx,
        )

    # -- model call ----------------------------------------------------------
    def predict(self, model, seq: MVSequence, target_tokens: Tensor, timesteps: Tensor, prefix_cache: dict | None = None) -> Tensor:
        """Run the DiT on [target (noisy), refs] and return the target velocity [B, Kt, C]."""
        dtype = self.weight_dtype
        if self.spec.name == "krea2":
            img = torch.cat([target_tokens.to(dtype)] + [r.to(dtype) for r in seq.ref_tokens], dim=1)
            ids = torch.cat([seq.target_ids] + seq.ref_ids, dim=1)
            zero_index = target_tokens.shape[1] if (self.ref_timestep_zero and seq.ref_tokens) else None
            pred = model(img, ids, timesteps.to(dtype), seq.ctx.to(dtype), timestep_zero_index=zero_index)
            return pred[:, : target_tokens.shape[1]]
        if prefix_cache is not None and "blocks" in prefix_cache:
            return model(target_tokens.to(dtype), seq.target_ids, timesteps.to(dtype), seq.ctx.to(dtype), attention="block_causal", prefix_cache=prefix_cache)
        img = torch.cat([r.to(dtype) for r in seq.ref_tokens] + [target_tokens.to(dtype)], dim=1)
        ids = torch.cat(seq.ref_ids + [seq.target_ids], dim=1)
        return model(
            img, ids, timesteps.to(dtype), seq.ctx.to(dtype),
            ref_lengths=[r.shape[1] for r in seq.ref_tokens], attention=self.attention, prefix_cache=prefix_cache,
        )

    # -- schedules -------------------------------------------------------------
    def mu(self, resolution: int, krea2_mu: float = 1.15) -> float:
        tokens_per_view = (resolution // TOKEN_STRIDE) ** 2
        if self.spec.name == "krea2":
            return krea2_mu
        return qwen21_mu(tokens_per_view)

    def training_timesteps(self, resolution: int, krea2_mu: float = 1.15) -> Tensor:
        """The UNO/UltraTex recipe: sample uniformly from a 999-step shifted schedule."""
        return torch.tensor(shifted_schedule(999, self.mu(resolution, krea2_mu)), device=self.device)

    # -- sampling --------------------------------------------------------------
    @torch.no_grad()
    def sample(self, model, seq: MVSequence, num_steps: int, resolution: int, seed: int = 0, guidance: float = 1.0,
               uncond: MVSequence | None = None, use_prefix_cache: bool = True, krea2_mu: float = 1.15) -> Tensor:
        """Euler flow-matching sampler over the retained target tokens. Returns [B, Kt, C]."""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn(seq.target_tokens.shape, generator=generator, device=self.device, dtype=torch.float32)
        timesteps = shifted_schedule(num_steps, self.mu(resolution, krea2_mu))
        cache = {} if (use_prefix_cache and self.spec.supports_prefix_cache and self.attention == "block_causal") else None
        uncond_cache = {} if (cache is not None and uncond is not None) else None
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((x.shape[0],), t_curr, device=self.device, dtype=torch.float32)
            pred = self.predict(model, seq, x, t_vec, cache).float()
            if uncond is not None and guidance != 1.0:
                pred_u = self.predict(model, uncond, x, t_vec, uncond_cache).float()
                pred = pred_u + guidance * (pred - pred_u)
            x = x + (t_prev - t_curr) * pred
        return x

    @torch.no_grad()
    def compose_and_decode(self, seq: MVSequence, generated: Tensor, background_latent: Tensor) -> Tensor:
        """Scatter generated foreground tokens into the canonical background latent and decode the 2x3 canvas."""
        from ultratex.backbones.common import patchify_tokens

        bg_tokens, h, w = patchify_tokens(background_latent.to(generated.dtype), self.spec.patch)
        if bg_tokens.shape[1] != seq.target_token_count:
            raise ValueError(f"background has {bg_tokens.shape[1]} tokens, expected {seq.target_token_count}")
        bg_tokens = bg_tokens.expand(generated.shape[0], -1, -1)
        full = scatter_sequence(bg_tokens, seq.target_indices, generated)
        latent = unpatchify_tokens(full, h, w, self.spec.latent_channels, self.spec.patch)
        return decode_grid(self.vae, latent)


def _ref_mask_full(batch: dict, ref_index: int, hw: tuple[int, int]) -> Tensor | None:
    """Full-resolution alpha for a reference; derived from its token mask when no pixel mask is available."""
    token_mask = torch.stack([m[ref_index + 1] for m in batch["masks"]]).float()
    return torch.nn.functional.interpolate(token_mask[:, None], size=hw, mode="nearest")[:, 0]


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------


def flow_loss(pred: Tensor, x_0: Tensor, x_1: Tensor) -> Tensor:
    """Foreground-restricted flow-matching objective: v = noise - clean on retained tokens."""
    return torch.nn.functional.mse_loss(pred.float(), (x_0 - x_1).float(), reduction="mean")


# ----------------------------------------------------------------------------
# Visualisation helpers (eval composites, same canvas as the FLUX ports)
# ----------------------------------------------------------------------------


def load_and_concat(image_dir: str, subdir: str, resolution: int):
    """Six TexVerse views -> one 2x3 canvas (PIL), white background."""
    import numpy as np
    from PIL import Image

    imgs = []
    for i in range(6):
        path = os.path.join(image_dir, subdir, f"{i:03d}.webp")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        img = Image.open(path).convert("RGBA").resize((resolution, resolution), Image.Resampling.LANCZOS)
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        imgs.append(np.asarray(bg, dtype=np.uint8))
    row1 = np.concatenate([imgs[0], imgs[1], imgs[3]], axis=1)
    row2 = np.concatenate([imgs[2], imgs[4], imgs[5]], axis=1)
    return Image.fromarray(np.concatenate([row1, row2], axis=0))


def make_eval_composite(image_dir: str, result_img, render_path: str, resolution: int, target_subdir: str = "albedo"):
    from PIL import Image

    pos_img = load_and_concat(image_dir, "position", resolution)
    bump_img = load_and_concat(image_dir, "bump_normal_world", resolution)
    target_img = load_and_concat(image_dir, target_subdir, resolution)
    render_rgba = Image.open(render_path).convert("RGBA").resize((resolution, resolution))
    render_img = Image.new("RGB", render_rgba.size, (255, 255, 255))
    render_img.paste(render_rgba, mask=render_rgba.getchannel("A"))

    ref_w, ref_h = pos_img.size
    cell_w, cell_h = render_img.size
    canvas = Image.new("RGB", (cell_w * 7, cell_h * 4), (255, 255, 255))
    canvas.paste(render_img, (0, 2 * cell_h))
    canvas.paste(pos_img, (cell_w, 0))
    canvas.paste(bump_img, (cell_w, 2 * cell_h))
    canvas.paste(target_img, (cell_w * 4, 0))
    canvas.paste(result_img.resize((ref_w, ref_h)), (cell_w * 4, 2 * cell_h))
    return canvas


def tensor_to_pil(decoded: Tensor, mask_full: Tensor | None = None, whiten_background: bool = True):
    """[1, 3|4, H, W] in [-1, 1] -> RGB PIL. RGBA output is composited on white by its own alpha
    unless a foreground mask is given, in which case the mask decides (as in the FLUX ports)."""
    import numpy as np
    from PIL import Image

    img = decoded[0].float().clamp(-1, 1)
    rgb = ((img[:3] + 1) * 127.5).permute(1, 2, 0).cpu().numpy()
    if whiten_background:
        if mask_full is not None:
            m = mask_full.float()
            if m.shape != rgb.shape[:2]:
                m = torch.nn.functional.interpolate(m[None, None], size=rgb.shape[:2], mode="nearest")[0, 0]
            bg = (m.cpu().numpy() == 0)
            rgb[bg] = 255.0
        elif img.shape[0] == 4:
            alpha = ((img[3] + 1) / 2).clamp(0, 1).cpu().numpy()[..., None]
            rgb = rgb * alpha + 255.0 * (1 - alpha)
    return Image.fromarray(rgb.round().astype(np.uint8))

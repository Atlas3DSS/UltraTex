"""Pieces shared by the Krea 2 and Qwen-Image 2.1 UltraTex backbones.

Both backbones are single-stream DiTs that consume one flat token sequence with
per-token 3-axis RoPE ids. UltraTex's Background Token Dropping (BTD) is a pure
sequence operation (gather the foreground tokens and their ids), so the only
backbone-specific parts are the block, the id convention and the modulation.
This module holds the attention dispatch (dense SDPA or the vendored SLA
block-sparse kernel), RoPE helpers, the flow-matching schedules and the
ComfyUI-backed VAE / latent normalisation used by both.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------
# Sparse attention backend (vendored SLA, same configuration as the FLUX ports)
# ----------------------------------------------------------------------------

_SPARSE_ATTENTION = None
_SPARSE_ATTENTION_CONFIG: dict = {"enabled": False}


def configure_sparse_attention(
    enabled: bool,
    head_dim: int = 128,
    topk: float = 0.20,
    blkq: int = 128,
    blkk: int = 64,
) -> dict:
    """Select the attention backend used by every DiT block.

    Mirrors ``ultratex/backbones/flux2/model.py``. The SLA Triton kernel needs
    equal query/key lengths, which holds for the pruned [text, target, refs]
    sequence that the trainers build.
    """
    global _SPARSE_ATTENTION, _SPARSE_ATTENTION_CONFIG
    if not enabled:
        _SPARSE_ATTENTION = None
        _SPARSE_ATTENTION_CONFIG = {"enabled": False}
        return dict(_SPARSE_ATTENTION_CONFIG)
    if not 0.0 < topk <= 1.0:
        raise ValueError(f"sparse attention topk must be in (0, 1], got {topk}")
    if head_dim != 128:
        raise ValueError(f"The vendored SLA kernel expects head_dim=128, got {head_dim}")
    from ultratex.sparse_attention import SparseLinearAttention

    _SPARSE_ATTENTION = SparseLinearAttention(
        head_dim=head_dim, topk=topk, feature_map="softmax", BLKQ=blkq, BLKK=blkk
    )
    _SPARSE_ATTENTION_CONFIG = {
        "enabled": True,
        "backend": "SparseLinearAttention",
        "head_dim": head_dim,
        "topk": topk,
        "BLKQ": blkq,
        "BLKK": blkk,
    }
    return dict(_SPARSE_ATTENTION_CONFIG)


def sparse_attention_config() -> dict:
    return dict(_SPARSE_ATTENTION_CONFIG)


def sparse_attention_enabled() -> bool:
    return _SPARSE_ATTENTION is not None


def dense_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """q, k, v: [B, H, N, D] -> [B, N, H*D]."""
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return rearrange(out, "b h n d -> b n (h d)")


def full_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Bidirectional attention over the whole sequence, sparse when configured."""
    if _SPARSE_ATTENTION is not None and q.shape[2] == k.shape[2]:
        return rearrange(_SPARSE_ATTENTION(q, k, v), "b h n d -> b n (h d)")
    return dense_attention(q, k, v)


def block_causal_attention(q: Tensor, k: Tensor, v: Tensor, segments: list[tuple[int, int, bool]]) -> Tensor:
    """Qwen-Image 2.1 block-causal attention.

    ``segments`` lists ``(start, end, causal)`` spans covering the sequence in
    order. A span attends to every key before its own end; a causal span (text)
    additionally applies a causal triangle inside itself. The last span is the
    target image, which therefore sees everything.

    With the sparse backend, the target rows come from one SLA pass over the
    full sequence (the only large block) and the prefix rows are recomputed
    exactly with dense SDPA over their restricted keys.
    """
    n = q.shape[2]
    if not segments or segments[-1][1] != n:
        raise ValueError(f"segments {segments} do not cover the sequence of length {n}")
    if _SPARSE_ATTENTION is not None:
        out = rearrange(_SPARSE_ATTENTION(q, k, v), "b h n d -> b n (h d)")
        prefix_end = segments[-1][0]
        if prefix_end == 0:
            return out
        parts = [
            _segment_attention(q, k, v, start, end, causal) for start, end, causal in segments[:-1]
        ]
        return torch.cat(parts + [out[:, prefix_end:]], dim=1)
    parts = [_segment_attention(q, k, v, start, end, causal) for start, end, causal in segments]
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def _segment_attention(q, k, v, start, end, causal):
    mask = None
    if causal:
        mask = torch.ones((end - start, end), dtype=torch.bool, device=q.device).tril(start)
    return dense_attention(q[:, :, start:end], k[:, :, :end], v[:, :, :end], mask)


def prefix_cached_attention(q: Tensor, k: Tensor, v: Tensor, prefix_k: Tensor, prefix_v: Tensor) -> Tensor:
    """Target-only queries against [cached prefix, target] keys (dense)."""
    return dense_attention(q, torch.cat([prefix_k, k], dim=2), torch.cat([prefix_v, v], dim=2))


# ----------------------------------------------------------------------------
# RoPE
# ----------------------------------------------------------------------------


def rope(pos: Tensor, dim: int, theta: float) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float64), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: float, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))], dim=-3
        )
        return emb.unsqueeze(1)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


# ----------------------------------------------------------------------------
# Token packing (BTD operates on these flat sequences)
# ----------------------------------------------------------------------------


def patchify_tokens(latent: Tensor, patch: int) -> tuple[Tensor, int, int]:
    """[B, C, H, W] -> [B, (H/p)(W/p), C*p*p] plus the token grid size."""
    b, c, h, w = latent.shape
    if h % patch or w % patch:
        raise ValueError(f"latent {h}x{w} is not divisible by patch {patch}")
    tokens = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    return tokens, h // patch, w // patch


def unpatchify_tokens(tokens: Tensor, h: int, w: int, channels: int, patch: int) -> Tensor:
    return rearrange(tokens, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, c=channels, ph=patch, pw=patch)


def gather_sequence(sequence: Tensor, indices: Tensor) -> Tensor:
    """Gather along the token axis with one index list per batch item."""
    return sequence.gather(1, indices[..., None].expand(-1, -1, sequence.shape[-1]))


def scatter_sequence(base: Tensor, indices: Tensor, values: Tensor) -> Tensor:
    return base.scatter(1, indices[..., None].expand_as(values), values)


def foreground_indices(mask: Tensor, token_count: int, label: str) -> Tensor:
    """Flat foreground token indices [B, K] from a [B, h, w] (or [h, w]) mask.

    Batch size 1 is the supported training configuration (exactly like the
    FLUX ports); larger batches must have equal foreground counts.
    """
    if mask.ndim == 2:
        mask = mask[None]
    flat = mask.reshape(mask.shape[0], -1).bool()
    if flat.shape[1] != token_count:
        raise ValueError(f"{label} mask has {flat.shape[1]} cells but the backbone produced {token_count} tokens")
    counts = flat.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError(f"{label} contains a sample with no foreground tokens")
    if not torch.all(counts == counts[0]):
        raise ValueError(f"{label} foreground counts differ within the batch ({counts.tolist()}); use batch_size=1")
    return torch.stack([row.nonzero(as_tuple=True)[0] for row in flat], dim=0)


# ----------------------------------------------------------------------------
# Flow-matching schedules
# ----------------------------------------------------------------------------


def time_shift(mu: float, sigma: float, t: Tensor) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def shifted_schedule(num_steps: int, mu: float) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


def qwen21_mu(image_seq_len: int, base_shift: float = 0.5, max_shift: float = 0.9, base_len: int = 256, max_len: int = 8192) -> float:
    """Qwen-Image 2.1 scheduler mu: 0.5 at 256 tokens, 0.9 at 8192 (linear, extrapolated)."""
    slope = (max_shift - base_shift) / (max_len - base_len)
    return slope * image_seq_len + (base_shift - slope * base_len)


# ----------------------------------------------------------------------------
# Latent normalisation (what the DiTs were trained on; the VAE itself is raw)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentNorm:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def process_in(self, latent: Tensor) -> Tensor:
        mean = torch.tensor(self.mean, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1)
        std = torch.tensor(self.std, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1)
        return (latent - mean) / std

    def process_out(self, latent: Tensor) -> Tensor:
        mean = torch.tensor(self.mean, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1)
        std = torch.tensor(self.std, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1)
        return latent * std + mean


# Wan 2.1 / Qwen-Image VAE (Krea 2 latent space), 16 channels, 8x.
WAN21_NORM = LatentNorm(
    mean=(-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
          0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921),
    std=(2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
         3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160),
)

# Qwen-Image 2.1 RGBA VAE, 64 channels, 16x.
QWEN21_NORM = LatentNorm(
    mean=(0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
          1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
          2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
          -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
          -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
          -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
          -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
          0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304),
    std=(3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910,
         3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756,
         3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
         3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467,
         3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125,
         3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
         3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959,
         3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161),
)


# ----------------------------------------------------------------------------
# ComfyUI-backed VAE
# ----------------------------------------------------------------------------


def import_comfy(comfy_root: str | None = None):
    """Make ComfyUI importable as a library (its CLI parser must not see our argv)."""
    root = comfy_root or os.environ.get("COMFY_ROOT")
    if not root or not os.path.isdir(os.path.join(root, "comfy")):
        raise FileNotFoundError(
            "COMFY_ROOT must point at a ComfyUI checkout with Qwen-Image 2.1 support "
            f"(got {root!r})"
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        import comfy.sd  # noqa: F401
        import comfy.utils  # noqa: F401
    finally:
        sys.argv = saved_argv
    import comfy

    return comfy


class ComfyVAE(nn.Module):
    """Thin wrapper over ComfyUI's Wan-family VAEs.

    ``encode``/``decode`` work on single images ``[B, C, H, W]`` in ``[-1, 1]``
    and return/consume *normalised* latents (the DiT's input space). The raw
    autoencoder is ``self.model`` (``comfy.ldm.wan.vae*.WanVAE``); its
    ``decoder`` + ``conv2`` are what the foreground-aware fine-tune trains.
    """

    def __init__(self, vae_path: str, norm: LatentNorm, image_channels: int, comfy_root: str | None = None, device="cpu", dtype=torch.float32):
        super().__init__()
        comfy = import_comfy(comfy_root)
        sd = comfy.utils.load_torch_file(vae_path)
        wrapper = comfy.sd.VAE(sd=sd, device=torch.device("cpu"), dtype=dtype)
        if wrapper.latent_channels != len(norm.mean):
            raise ValueError(
                f"{vae_path} has {wrapper.latent_channels} latent channels, expected {len(norm.mean)}"
            )
        if wrapper.output_channels != image_channels:
            raise ValueError(
                f"{vae_path} reconstructs {wrapper.output_channels} image channels, expected {image_channels}"
            )
        self.model = wrapper.first_stage_model.to(device=device, dtype=dtype)
        self.norm = norm
        self.image_channels = image_channels
        self.latent_channels = wrapper.latent_channels
        self.downscale = wrapper.downscale_ratio if isinstance(wrapper.downscale_ratio, int) else wrapper.downscale_ratio[-1]
        # The Wan 2.1 VAE (Krea 2 latent space) is a video VAE that only takes [B, C, T, H, W];
        # the Qwen-Image 2.1 VAE accepts single images directly.
        self.video_only = type(self.model).__module__.endswith("wan.vae")

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def trainable_decoder_parameters(self):
        return list(self.model.decoder.parameters()) + list(self.model.conv2.parameters())

    def decoder_state_dict(self) -> dict[str, Tensor]:
        out = {f"decoder.{k}": v for k, v in self.model.decoder.state_dict().items()}
        out.update({f"conv2.{k}": v for k, v in self.model.conv2.state_dict().items()})
        return out

    def load_decoder_state_dict(self, state: dict[str, Tensor]) -> None:
        dec = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
        conv2 = {k[len("conv2."):]: v for k, v in state.items() if k.startswith("conv2.")}
        self.model.decoder.load_state_dict(dec, strict=True)
        if conv2:
            self.model.conv2.load_state_dict(conv2, strict=True)

    def encode(self, pixels: Tensor) -> Tensor:
        """pixels [B, C, H, W] in [-1, 1] (C == image_channels) -> normalised latent."""
        if pixels.shape[1] != self.image_channels:
            raise ValueError(f"expected {self.image_channels} image channels, got {pixels.shape[1]}")
        pixels = pixels.to(self.device, self.dtype)
        if self.video_only:
            z = self.model.encode(pixels.unsqueeze(2)).squeeze(2)
        else:
            z = self.model.encode(pixels)
        return self.norm.process_in(z.float())

    def decode(self, latent: Tensor) -> Tensor:
        """normalised latent [B, C, h, w] -> pixels [B, image_channels, H, W] in [-1, 1]."""
        z = self.norm.process_out(latent.float()).to(self.dtype)
        if self.video_only:
            return self.model.decode(z.unsqueeze(2)).squeeze(2).float()
        return self.model.decode(z).float()

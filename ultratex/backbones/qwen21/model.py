"""Qwen-Image 2.1 single-stream DiT for UltraTex multi-view texturing.

Architecture matches the diffusers ``QwenImage21Transformer2DModel`` / ComfyUI
port, so ``qwen_image_2.1_bf16.safetensors`` loads with ``strict=True``:
32 blocks, 4096 wide, 32 heads (head dim 128), fused SwiGLU (``gate_up``), one
shared AdaLN modulation (scale/gate only, tanh gates), 3-axis RoPE (frame, h, w)
with dims (16, 56, 56) and theta 10000, 64-channel 16x RGBA latents with patch 1,
Qwen3-VL-8B last-hidden-state text features, and *block-causal* conditioning:
text and condition images form a causal prefix modulated from t = 0 while the
target image attends to everything.

UltraTex changes: tokens arrive pre-packed with RoPE ids so background tokens
can be dropped (BTD), attention is dispatched to dense SDPA or the SLA sparse
kernel, and the prefix (text + references) K/V can be cached across sampling
steps because it is timestep-independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import functional as F

from ultratex.backbones.common import (
    EmbedND,
    apply_rope,
    block_causal_attention,
    full_attention,
    prefix_cached_attention,
    timestep_embedding,
)


@dataclass
class Qwen21Params:
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    theta: float = 10000.0
    eps: float = 1e-6


class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x.float(), (x.shape[-1],), weight=self.weight.float() + 1.0, eps=self.eps).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, in_dim, hidden_size, eps=1e-6):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(in_dim, eps=eps)
        self.in_layer = nn.Linear(in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class TimestepEmbedder(nn.Module):
    """``time_text_embed.timestep_embedder.linear_{1,2}`` (no bias)."""

    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(256, dim, bias=False)
        self.linear_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, t: Tensor, dtype) -> Tensor:
        emb = timestep_embedding(t.float(), 256).to(dtype)
        return self.linear_2(F.silu(self.linear_1(emb)))


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_embedder = TimestepEmbedder(dim)

    def forward(self, t, dtype):
        return self.timestep_embedder(t, dtype)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.out(F.silu(gate) * up)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x.float(), (x.shape[-1],), weight=self.weight.float(), eps=self.eps).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, eps=1e-6):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim, bias=False)])
        self.norm_q = RMSNorm(dim_head, eps)
        self.norm_k = RMSNorm(dim_head, eps)

    def forward(self, x, pe, attn_fn, cache: dict | None = None, cache_prefix_len: int = 0):
        b, n, _ = x.shape
        q = self.to_q(x).view(b, n, self.heads, -1).transpose(1, 2)
        k = self.to_k(x).view(b, n, self.heads, -1).transpose(1, 2)
        v = self.to_v(x).view(b, n, self.heads, -1).transpose(1, 2)
        q, k = self.norm_q(q), self.norm_k(k)
        q, k = apply_rope(q, k, pe)
        if cache is not None and "k" not in cache:
            cache["k"] = k[:, :, :cache_prefix_len].detach()
            cache["v"] = v[:, :, :cache_prefix_len].detach()
        return self.to_out[0](attn_fn(q, k, v))


class Qwen21Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_ratio=3, eps=1e-6):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, heads, dim_head, eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio)

    @staticmethod
    def _rows(param: Tensor, prefix_len: int, n: int) -> Tensor:
        """param: [B+1, D] -> [B, N, D]; prefix rows use the t = 0 row (last)."""
        target = param[:-1, None]
        if prefix_len == 0:
            return target.expand(-1, n, -1)
        zero = param[-1:, None].expand(target.shape[0], prefix_len, -1)
        return torch.cat([zero, target.expand(-1, n - prefix_len, -1)], dim=1)

    def forward(self, x, mod, pe, attn_fn, prefix_len, cache=None, cache_prefix_len=0):
        scale1, gate1, scale2, gate2 = mod
        n = x.shape[1]
        s1, g1 = self._rows(scale1, prefix_len, n), self._rows(gate1, prefix_len, n)
        s2, g2 = self._rows(scale2, prefix_len, n), self._rows(gate2, prefix_len, n)
        x = x + g1 * self.attn(self.img_norm1(x) * (1 + s1), pe, attn_fn, cache, cache_prefix_len)
        x = x + g2 * self.img_mlp(self.img_norm2(x) * (1 + s2))
        return x


class LastLayer(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, x, temb):
        scale = self.linear(F.silu(temb)).unsqueeze(1)
        return self.norm(x) * (1 + scale)


class QwenImage21(nn.Module):
    def __init__(self, params: Qwen21Params):
        super().__init__()
        p = params
        self.params = p
        self.patch = 1
        self.channels = p.in_channels
        self.out_channels = p.out_channels
        self.inner_dim = p.num_attention_heads * p.attention_head_dim
        self.hidden_size = self.inner_dim
        self.num_heads = p.num_attention_heads
        self.head_dim = p.attention_head_dim
        self.gradient_checkpointing = False

        self.pe_embedder = EmbedND(dim=p.attention_head_dim, theta=p.theta, axes_dim=list(p.axes_dims_rope))
        self.time_text_embed = TimestepProjEmbeddings(self.inner_dim)
        self.txt_in = TextProjection(p.context_in_dim, self.inner_dim, eps=p.eps)
        self.img_in = nn.Linear(p.in_channels, self.inner_dim, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))
        self.transformer_blocks = nn.ModuleList(
            [Qwen21Block(self.inner_dim, p.num_attention_heads, p.attention_head_dim, p.mlp_ratio, p.eps) for _ in range(p.num_layers)]
        )
        self.norm_out = LastLayer(self.inner_dim, eps=p.eps)
        self.proj_out = nn.Linear(self.inner_dim, p.out_channels, bias=False)

    # ------------------------------------------------------------------
    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ref_lengths: list[int] | None = None,
        attention: str = "block_causal",
        attn_fn=None,
        prefix_cache: dict | None = None,
    ) -> Tensor:
        """Sequence layout is ``[text, ref_1, ..., ref_k, target]``.

        img: [B, N_img, 64] packed image tokens ordered ``refs..., target``
             (each may be background-pruned); img_ids: [B, N_img, 3] float RoPE
             ids (frame, h, w) already laid out by ``util.build_sequence``;
             ref_lengths: token count of every reference block, in order;
             ctx: [B, L, 4096] text features (frame ids 0..L-1 are assigned here).

        attention: ``block_causal`` (Qwen-Image 2.1's native structure: text
        causal, each image attends to everything before its end) or ``full``
        (bidirectional, as the FLUX ports train). ``attn_fn`` overrides the
        dispatch used for the last/target block (defaults to the shared
        dense/sparse dispatch).

        prefix_cache: a dict shared across sampling steps. When it holds
        cached K/V, ``img`` must contain only the target tokens and the
        prefix is skipped entirely (``block_causal`` only). Returns velocity
        predictions for the image tokens that were passed in.
        """
        ref_lengths = list(ref_lengths or [])
        b = img.shape[0]
        dtype = img.dtype
        cached = prefix_cache is not None and "blocks" in prefix_cache
        if cached and attention != "block_causal":
            raise ValueError("prefix caching requires block_causal attention")

        # pipeline rounds t*1000 and t to the compute dtype; text/ref rows modulate from t = 0
        t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
        mod = (scale1, gate1.tanh(), scale2, gate2.tanh())

        if cached:
            hidden = self.img_in(img)
            pe = self.pe_embedder(img_ids.float())
            prefix_len = 0
            segments = None
        else:
            txt = self.txt_in(ctx)
            txtlen = txt.shape[1]
            txt_ids = torch.arange(txtlen, device=img.device, dtype=torch.float32)[None, :, None].expand(b, txtlen, 3)
            hidden = torch.cat([txt, self.img_in(img)], dim=1)
            pe = self.pe_embedder(torch.cat([txt_ids, img_ids.float()], dim=1))
            n_ref = sum(ref_lengths)
            prefix_len = txtlen + n_ref
            if prefix_len >= hidden.shape[1]:
                raise ValueError("no target tokens in the sequence")
            segments = [(0, txtlen, True)] if txtlen else []
            cursor = txtlen
            for length in ref_lengths:
                segments.append((cursor, cursor + length, False))
                cursor += length
            segments.append((cursor, hidden.shape[1], False))

        if attention == "block_causal":
            if cached:
                attn_builder = lambda i: (lambda q, k, v: prefix_cached_attention(q, k, v, *prefix_cache["blocks"][i]))  # noqa: E731
            else:
                attn_builder = lambda i: (lambda q, k, v: block_causal_attention(q, k, v, segments))  # noqa: E731
        elif attention == "full":
            fn = attn_fn or full_attention
            attn_builder = lambda i: fn  # noqa: E731
        else:
            raise ValueError(f"unknown attention mode {attention!r}")

        fill_cache = prefix_cache is not None and not cached and attention == "block_causal"
        caches = [dict() for _ in self.transformer_blocks] if fill_cache else [None] * len(self.transformer_blocks)

        for i, block in enumerate(self.transformer_blocks):
            fn = attn_builder(i)
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    block, hidden, mod, pe, fn, prefix_len, None, 0, use_reentrant=False
                )
            else:
                hidden = block(hidden, mod, pe, fn, prefix_len, caches[i], prefix_len)

        if fill_cache:
            prefix_cache["blocks"] = [(c["k"], c["v"]) for c in caches]
            prefix_cache["prefix_len"] = prefix_len

        out = self.norm_out(hidden[:, prefix_len:], temb[:-1])
        return self.proj_out(out)

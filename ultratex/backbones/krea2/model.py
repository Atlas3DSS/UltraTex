"""Krea 2 (K2) single-stream DiT for UltraTex multi-view texturing.

Architecture follows the released ``krea-ai/krea-2`` ``mmdit.py`` (and the
ComfyUI port) exactly, so ``krea2_raw_bf16.safetensors`` / ``raw.safetensors``
load with ``strict=True``: 28 blocks, 6144 wide, 48 query / 12 key-value heads
(head dim 128), SwiGLU, AdaLN-single modulation with per-block learned tables,
sigmoid-gated attention, 3-axis RoPE (index, h, w) with theta 1000, 2x2 patches
over the 16-channel Qwen-Image / Wan 2.1 latent, and a 12-layer Qwen3-VL-4B
``txtfusion`` text adapter.

UltraTex changes are confined to the sequence interface: image tokens arrive
pre-packed with their RoPE ids (so background tokens can be dropped while the
survivors keep their absolute coordinates), attention goes through the shared
dense/SLA dispatch, and reference-image tokens may optionally be modulated from
t = 0 (the ``index_timestep_zero`` convention used by Krea 2 edit adapters).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from ultratex.backbones.common import EmbedND, apply_rope, full_attention, timestep_embedding


@dataclass
class Krea2Params:
    features: int = 6144
    tdim: int = 256
    txtdim: int = 2560
    heads: int = 48
    kvheads: int = 12
    multiplier: int = 4
    layers: int = 28
    patch: int = 2
    channels: int = 16
    bias: bool = False
    theta: float = 1000.0
    txtlayers: int = 12
    txtheads: int = 20
    txtkvheads: int = 20
    eps: float = 1e-5


class RMSNorm(nn.Module):
    """RMSNorm with the reference ``(1 + scale)`` convention (scale stored zero-centred)."""

    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), weight=self.scale.float() + 1.0, eps=self.eps).to(x.dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.qnorm = RMSNorm(dim, eps)
        self.knorm = RMSNorm(dim, eps)

    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int | None = None, bias: bool = False, eps: float = 1e-5):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim, eps)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: Tensor, freqs: Tensor | None = None, attn_fn=None) -> Tensor:
        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        if attn_fn is None:
            out = F.scaled_dot_product_attention(q, k, v)
            out = rearrange(out, "b h n d -> b n (h d)")
        else:
            out = attn_fn(q, k, v)
        return self.wo(out * torch.sigmoid(gate))


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))

    def forward(self, vec):
        out = vec + self.lin.to(vec.dtype).unsqueeze(0)
        scale, shift = out.chunk(2, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec):
        out = vec + self.lin.to(vec.dtype)
        return out.chunk(6, dim=-1)


class TextFusionBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None, eps=1e-5):
        super().__init__()
        self.prenorm = RMSNorm(features, eps)
        self.postnorm = RMSNorm(features, eps)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias, eps=eps)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x):
        x = x + self.attn(self.prenorm(x))
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):
    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias=False, kvheads=None, eps=1e-5):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads, eps) for _ in range(2)]
        )
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads, eps) for _ in range(2)]
        )

    def forward(self, x):
        # x: (B, seq, txtlayers, txtdim)
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous())
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None, eps=1e-5):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features, eps)
        self.postnorm = RMSNorm(features, eps)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias, eps=eps)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, vec, freqs, attn_fn=None, timestep_zero_index: int | None = None):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        if timestep_zero_index is None:
            x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs, attn_fn)
            x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
            return x

        # Rows [0, timestep_zero_index) use the sampled-t modulation (batch rows [:bs]),
        # rows [timestep_zero_index, N) use the t = 0 modulation (batch rows [bs:]).
        bs = x.shape[0]

        def split(m):
            return m[:bs], m[bs:]

        (ps, ps0), (psh, psh0), (pg, pg0) = split(prescale), split(preshift), split(pregate)
        (qs, qs0), (qsh, qsh0), (qg, qg0) = split(postscale), split(postshift), split(postgate)
        i = timestep_zero_index

        pre = self.prenorm(x)
        pre = torch.cat([(1 + ps) * pre[:, :i] + psh, (1 + ps0) * pre[:, i:] + psh0], dim=1)
        attn = self.attn(pre, freqs, attn_fn)
        x = x + torch.cat([pg * attn[:, :i], pg0 * attn[:, i:]], dim=1)

        post = self.postnorm(x)
        post = torch.cat([(1 + qs) * post[:, :i] + qsh, (1 + qs0) * post[:, i:] + qsh0], dim=1)
        mlp = self.mlp(post)
        x = x + torch.cat([qg * mlp[:, :i], qg0 * mlp[:, i:]], dim=1)
        return x


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels, eps=1e-5):
        super().__init__()
        self.norm = RMSNorm(features, eps)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)


class Krea2(nn.Module):
    def __init__(self, params: Krea2Params):
        super().__init__()
        p = params
        self.params = p
        self.patch = p.patch
        self.channels = p.channels
        self.hidden_size = p.features
        self.num_heads = p.heads
        self.head_dim = p.features // p.heads
        self.txtdim = p.txtdim
        self.txtlayers = p.txtlayers
        self.gradient_checkpointing = False

        headdim = self.head_dim
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim
        self.pe_embedder = EmbedND(dim=headdim, theta=p.theta, axes_dim=axes)

        self.first = nn.Linear(p.channels * p.patch**2, p.features, bias=True)
        self.blocks = nn.ModuleList(
            [SingleStreamBlock(p.features, p.heads, p.multiplier, p.bias, p.kvheads, p.eps) for _ in range(p.layers)]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(p.tdim, p.features), nn.GELU(approximate="tanh"), nn.Linear(p.features, p.features)
        )
        self.txtfusion = TextFusionTransformer(p.txtlayers, p.txtdim, p.txtheads, p.multiplier, p.bias, p.txtkvheads, p.eps)
        self.txtmlp = nn.Sequential(
            RMSNorm(p.txtdim, p.eps), nn.Linear(p.txtdim, p.features), nn.GELU(approximate="tanh"), nn.Linear(p.features, p.features)
        )
        self.last = LastLayer(p.features, p.patch, p.channels, p.eps)
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), nn.Linear(p.features, p.features * 6))

    # ------------------------------------------------------------------
    def encode_text(self, ctx: Tensor) -> Tensor:
        """ctx: (B, seq, txtlayers*txtdim) flattened 12-layer stack (ComfyUI layout) or (B, seq, txtlayers, txtdim)."""
        if ctx.ndim == 3:
            b, seq, fused = ctx.shape
            if fused != self.txtlayers * self.txtdim:
                raise ValueError(f"Krea2 expects {self.txtlayers}x{self.txtdim} text features, got {fused}")
            ctx = ctx.reshape(b, seq, self.txtlayers, self.txtdim)
        ctx = self.txtfusion(ctx)
        return self.txtmlp(ctx)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        attn_fn=full_attention,
        timestep_zero_index: int | None = None,
        txt_cache: Tensor | None = None,
    ) -> Tensor:
        """img: [B, N, channels*patch^2] packed image tokens (target + refs, possibly pruned),
        img_ids: [B, N, 3] float RoPE ids (index, h, w), timesteps: [B] in [0, 1],
        ctx: text features. Returns velocity predictions for every image token [B, N, channels*patch^2].

        ``timestep_zero_index`` (an image-token offset) makes the image tokens
        from that position on modulate from t = 0 (reference tokens).
        """
        bs = img.shape[0]
        x = self.first(img)

        # modulation vectors are kept as [B, 1, F] so they broadcast over the token axis
        t = self.tmlp(timestep_embedding(timesteps, self.params.tdim).to(x.dtype)).unsqueeze(1)
        tvec = self.tproj(t)
        if timestep_zero_index is not None:
            t0 = self.tmlp(timestep_embedding(torch.zeros_like(timesteps), self.params.tdim).to(x.dtype)).unsqueeze(1)
            tvec = torch.cat((tvec, self.tproj(t0)), dim=0)

        txt = self.encode_text(ctx) if txt_cache is None else txt_cache
        txtlen = txt.shape[1]
        txtpos = torch.zeros(bs, txtlen, 3, device=x.device, dtype=torch.float32)

        combined = torch.cat((txt, x), dim=1)
        zero_index = None if timestep_zero_index is None else timestep_zero_index + txtlen
        freqs = self.pe_embedder(torch.cat((txtpos, img_ids.float()), dim=1))

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                combined = torch.utils.checkpoint.checkpoint(
                    block, combined, tvec, freqs, attn_fn, zero_index, use_reentrant=False
                )
            else:
                combined = block(combined, tvec, freqs, attn_fn, zero_index)

        out = self.last(combined, t)
        return out[:, txtlen:]

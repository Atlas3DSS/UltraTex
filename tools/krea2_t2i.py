#!/usr/bin/env python
"""Minimal Krea 2 Turbo text-to-image sampler built on the UltraTex Krea 2 module.

Used to make avatar concept images (reference renders) without a ComfyUI
workflow. Prompts must already be in a text cache (tools/precompute_text_conditioning.py
--backbone krea2 --prompt ...). Krea 2 Turbo: 8 steps, no CFG, mu 1.15.

  python tools/krea2_t2i.py --model .../krea2_turbo_bf16.safetensors --text-cond cache.safetensors \
      --prompt "..." --out img.png [--size 1024] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultratex.backbones.common import shifted_schedule, unpatchify_tokens  # noqa: E402
from ultratex.backbones.krea2.util import load_ae, load_flow_model, pack_image  # noqa: E402
from ultratex.mv import TextCache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vae", default=None)
    ap.add_argument("--text-cond", required=True)
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--out", required=True, help="output file, or a directory when several prompts/seeds are given")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--mu", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=1, help="images per prompt (consecutive seeds)")
    ap.add_argument("--comfy-root", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")

    model = load_flow_model(args.model, device=device, dtype=torch.bfloat16).eval()
    vae = load_ae(args.vae, device=device, dtype=torch.bfloat16, comfy_root=args.comfy_root or os.environ.get("COMFY_ROOT"))
    cache = TextCache(args.text_cond, "krea2", device)
    h = w = args.size // 8
    many = len(args.prompt) * args.count > 1
    if many:
        os.makedirs(args.out, exist_ok=True)
    with torch.no_grad():
        for pi, prompt in enumerate(args.prompt):
            ctx = cache.get([prompt], torch.bfloat16)
            for k in range(args.count):
                seed = args.seed + k
                gen = torch.Generator(device=device).manual_seed(seed)
                x = torch.randn((1, 16, h, w), generator=gen, device=device, dtype=torch.float32)
                tokens, ids = pack_image(x, 0)
                for t_curr, t_prev in zip(*(lambda s: (s[:-1], s[1:]))(shifted_schedule(args.steps, args.mu))):
                    t = torch.full((1,), t_curr, device=device, dtype=torch.bfloat16)
                    pred = model(tokens.to(torch.bfloat16), ids, t, ctx).float()
                    tokens = tokens + (t_prev - t_curr) * pred
                latent = unpatchify_tokens(tokens, h // 2, w // 2, 16, 2)
                img = vae.decode(latent)[0].clamp(-1, 1)
                from PIL import Image

                pil = Image.fromarray(((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype("uint8"))
                path = os.path.join(args.out, f"p{pi:02d}_s{seed}.png") if many else args.out
                pil.save(path)
                print(f"wrote {path}  ({prompt[:60]!r}, seed {seed})")


if __name__ == "__main__":
    main()

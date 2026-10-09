#!/usr/bin/env python
"""Encode prompts once with ComfyUI's text encoders and cache the DiT conditioning.

UltraTex texturing is image-guided: the prompt is the empty string. Encoding it
once (per backbone) means training and inference never load the 4B/8B
Qwen3-VL text encoders.

  krea2  -> Qwen3-VL-4B, 12-layer tap, flattened to [L, 12*2560] (Krea 2 layout)
  qwen21 -> Qwen3-VL-8B, last hidden state (no final norm), [L, 4096]

Usage:
  COMFY_ROOT=... python tools/precompute_text_conditioning.py --backbone qwen21 \
      --text-encoder /path/to/qwen3vl_8b_bf16.safetensors --out checkpoints/qwen21/text_cond.safetensors \
      --prompt "" [--prompt "..."]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultratex.backbones.common import import_comfy  # noqa: E402
from ultratex.mv import get_spec, prompt_key  # noqa: E402

CLIP_TYPES = {"krea2": "KREA2", "qwen21": "QWEN_IMAGE"}  # Qwen-Image 2.1 uses the QWEN_IMAGE clip type with a Qwen3-VL-8B encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", required=True, choices=sorted(CLIP_TYPES))
    parser.add_argument("--text-encoder", required=True, help="ComfyUI text encoder safetensors")
    parser.add_argument("--out", required=True, help="output .safetensors (a .json sidecar is written next to it)")
    parser.add_argument("--prompt", action="append", default=None, help="prompt to encode (repeatable); default: ''")
    parser.add_argument("--comfy-root", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    prompts = args.prompt if args.prompt is not None else [""]

    comfy = import_comfy(args.comfy_root)
    import comfy.sd  # noqa: F401

    clip_type = getattr(comfy.sd.CLIPType, CLIP_TYPES[args.backbone])
    clip = comfy.sd.load_clip(ckpt_paths=[args.text_encoder], clip_type=clip_type, model_options={"load_device": torch.device(args.device)})

    spec = get_spec(args.backbone)
    tensors = {}
    for prompt in prompts:
        tokens = clip.tokenize(prompt)
        out = clip.encode_from_tokens(tokens, return_dict=True)
        cond = out["cond"]
        if cond.ndim != 3 or cond.shape[0] != 1 or cond.shape[-1] != spec.text_dim:
            raise ValueError(f"unexpected conditioning shape {tuple(cond.shape)} for {args.backbone} (want [1, L, {spec.text_dim}])")
        if out.get("attention_mask") is not None and int(out["attention_mask"].sum()) != out["attention_mask"].numel():
            raise ValueError("padded prompts are not supported in the cache; encode one prompt at a time (this tool does)")
        if out.get("image_slots"):
            raise ValueError("image slots are not supported; prompts must be text-only")
        tensors[prompt_key(prompt)] = cond[0].to(torch.bfloat16).cpu().contiguous()
        print(f"encoded {prompt!r}: {tuple(cond[0].shape)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(tensors, args.out)
    with open(os.path.splitext(args.out)[0] + ".json", "wt", encoding="utf-8") as f:
        json.dump({"backbone": args.backbone, "text_encoder": os.path.basename(args.text_encoder), "prompts": prompts, "text_dim": spec.text_dim}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""UltraTex multi-view albedo inference on Krea 2 or Qwen-Image 2.1.

Loads the base DiT plus a trained LoRA (merged for memory), the optional
foreground-aware decoder, and renders every object in an eval JSON: the six
albedo views as one 2x3 canvas plus the eval composite used during training.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Literal

import torch
import transformers
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from ultratex import mv
from ultratex.backbones.common import configure_sparse_attention
from ultratex.data.dataset import FluxPairedDatasetV2
from ultratex.data.dataset_ai import FluxPairedDatasetAI

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class InferenceArgs:
    backbone: Literal["krea2", "qwen21"] = "qwen21"
    model_path: str | None = None
    vae_path: str | None = None
    text_cond_path: str = ""
    decoder_ckpt: str | None = None
    comfy_root: str | None = None
    lora_path: str | None = None  # checkpoint directory holding dit_lora.safetensors (None = base model, for ablations)
    lora_rank: int = 64
    lora_alpha: int | None = None
    block_indices: list[int] | None = dataclasses.field(default=None)
    debug_model: bool = False

    eval_data_json: str = "data/eval_demo.json"
    dataset_type: Literal["standard", "ai"] = "standard"  # ai = single reference image (concept skinning), as in inference_flux2.py
    data_root: str | None = None
    resolution: int = 2048
    max_samples: int | None = None
    seed: int = 0
    inference_num_steps: int = 25
    attention: str | None = None
    ref_timestep_zero: bool = False
    use_prefix_cache: bool = True
    sparse_attention: bool = True
    sparse_attention_topk: float = 0.20
    sparse_attention_blkq: int = 128
    sparse_attention_blkk: int = 64
    krea2_mu: float = 1.15
    background: str | None = None
    inference_drop_background_tokens: bool = True
    inference_remove_background: bool = True
    degrade_single_render: bool = False
    degrade_single_render_resolution: int = 512
    project_dir: str = "outputs/inference_mv"
    dtype: Literal["bf16", "fp32"] = "bf16"


def main(args: InferenceArgs) -> None:
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    spec = mv.get_spec(args.backbone)
    comfy_root = args.comfy_root or os.environ.get("COMFY_ROOT")
    os.makedirs(args.project_dir, exist_ok=True)

    vae = mv.load_vae(args.backbone, args.vae_path, device=device, dtype=weight_dtype, comfy_root=comfy_root)
    if args.decoder_ckpt:
        vae.load_decoder_state_dict(load_file(args.decoder_ckpt))
    vae.eval().requires_grad_(False)
    text_cache = mv.TextCache(args.text_cond_path, args.backbone, device)
    pipe = mv.MVPipeline(spec, vae, text_cache, device, weight_dtype, attention=args.attention, ref_timestep_zero=args.ref_timestep_zero)

    model = mv.load_backbone(args.backbone, args.model_path, device="cpu", dtype=weight_dtype, debug_mode=args.debug_model)
    configure_sparse_attention(enabled=args.sparse_attention, head_dim=model.head_dim, topk=args.sparse_attention_topk,
                               blkq=args.sparse_attention_blkq, blkk=args.sparse_attention_blkk)
    if args.lora_path:
        lora_file = os.path.join(args.lora_path, "dit_lora.safetensors")
        meta = {}
        with open(lora_file, "rb") as f:
            import struct

            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n)).get("__metadata__", {}) or {}
        if meta.get("backbone") not in (None, args.backbone):
            raise ValueError(f"{lora_file} was trained for {meta['backbone']!r}, not {args.backbone!r}")
        rank = int(meta.get("lora_rank", args.lora_rank))
        model.requires_grad_(False)
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=args.lora_alpha or rank,
                                                 target_modules=mv.lora_targets(args.backbone, model, args.block_indices), bias="none"))
        state = load_file(lora_file)
        result = model.load_state_dict(state, strict=False)
        if any("lora_" in k for k in result.unexpected_keys) or not any("lora_" in k for k in state):
            raise ValueError(f"LoRA {lora_file} does not match the {args.backbone} LoRA layout")
        model.merge_and_unload()
        model = model.base_model.model if hasattr(model, "base_model") else model
        logger.info("merged LoRA from %s (rank %d)", lora_file, rank)
    model.to(device).eval().requires_grad_(False)

    if args.dataset_type == "ai":
        dataset = FluxPairedDatasetAI(json_file=args.eval_data_json, resolution=args.resolution)
    else:
        dataset = FluxPairedDatasetV2(json_file=args.eval_data_json, resolution=args.resolution, data_root=args.data_root,
                                      degrade_single_render_enabled=args.degrade_single_render,
                                      degrade_single_render_resolution=args.degrade_single_render_resolution)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    background_latent = mv.canonical_background_latent(vae, spec, args.resolution, args.background or spec.default_background)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_samples is not None and i >= args.max_samples:
                break
            sample_id = batch["id"][0]
            seq = pipe.build(batch, drop_background=args.inference_drop_background_tokens)
            generated = pipe.sample(model, seq, args.inference_num_steps, args.resolution, seed=args.seed,
                                    use_prefix_cache=args.use_prefix_cache, krea2_mu=args.krea2_mu)
            decoded = pipe.compose_and_decode(seq, generated, background_latent)
            mask_full = batch["masks_full"][0][0] if args.inference_remove_background else None
            result = mv.tensor_to_pil(decoded, mask_full, whiten_background=args.inference_remove_background)
            result.save(os.path.join(args.project_dir, f"result_{sample_id}_{i}.webp"), format="WEBP", lossless=True)
            composite = mv.make_eval_composite(batch["image_dir"][0], result, batch["render_path"][0], args.resolution)
            composite.save(os.path.join(args.project_dir, f"eval_{sample_id}_{i}.webp"), format="WEBP", quality=92)
            logger.info("%s -> %s (tokens kept %d/%d)", sample_id, args.project_dir, seq.target_tokens.shape[1], seq.target_token_count)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser([InferenceArgs])
    (args,) = parser.parse_args_into_dataclasses(args_file_flag="--config")
    main(args)

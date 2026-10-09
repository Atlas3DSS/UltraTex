"""UltraTex multi-view albedo LoRA training on Krea 2 or Qwen-Image 2.1.

Same recipe as ``train_flux2.py`` (TexVerse 2x3 canvases, background token
dropping, foreground-restricted flow matching, rank-64 LoRA, UNO shifted
timestep schedule, periodic LoRA checkpoints with validation renders), with the
backbone-specific pieces delegated to ``ultratex.mv``:

  --backbone krea2   Krea 2 Raw (12B single-stream DiT, 16ch 8x latent, 2x2 patch)
  --backbone qwen21  Qwen-Image 2.1 (7B single-stream DiT, 64ch 16x RGBA latent, patch 1,
                     block-causal conditioning with cached text/reference prefix)

Text conditioning comes from a cache built by ``tools/precompute_text_conditioning.py``
so no text encoder is loaded at train time.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
import os
import re
import time
from typing import Literal

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultratex import mv
from ultratex.backbones.common import configure_sparse_attention, gather_sequence
from ultratex.data.dataset import FluxPairedDatasetV2, LengthAwareBatchSampler

logger = get_logger(__name__)


@dataclasses.dataclass
class TrainArgs:
    backbone: Literal["krea2", "qwen21"] = "qwen21"
    model_path: str | None = None  # defaults to $KREA2_MODEL_PATH / $QWEN21_MODEL_PATH
    vae_path: str | None = None  # defaults to $KREA2_VAE_PATH / $QWEN21_VAE_PATH
    text_cond_path: str = ""  # tools/precompute_text_conditioning.py output
    decoder_ckpt: str | None = None  # foreground-aware decoder (train_vae/train_decoder_mv.py); optional
    comfy_root: str | None = None  # defaults to $COMFY_ROOT
    debug_model: bool = False  # 1-block random model, for pipeline smoke tests only

    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    gradient_accumulation_steps: int = 1
    seed: int = 42

    lora_rank: int = 64
    lora_alpha: int | None = None
    block_indices: list[int] | None = dataclasses.field(default=None)
    gradient_checkpoint: bool = True
    attention: str | None = None  # krea2: full; qwen21: block_causal (default per backbone)
    ref_timestep_zero: bool = False  # krea2 only: modulate reference tokens from t = 0
    sparse_attention: bool = True
    sparse_attention_topk: float = 0.20
    sparse_attention_blkq: int = 128
    sparse_attention_blkk: int = 64
    krea2_mu: float = 1.15  # Krea 2 flow shift (mu); Qwen-Image 2.1 derives mu from the token count
    background: str | None = None  # canonical background colour for compose/decode (per-backbone default)

    learning_rate: float = 1e-4
    adam_betas: list[float] = dataclasses.field(default_factory=lambda: [0.9, 0.999])
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 1000000

    train_data_json: str = "data/train_demo.json"
    eval_data_json: str = "data/eval_demo.json"
    data_root: str | None = None
    batch_size: int = 1
    resolution: int = 2048
    num_workers: int = 8
    bucket_metadata_json: str | None = None
    degrade_single_render: bool = False
    degrade_single_render_resolution: int = 512

    inference_num_steps: int = 25
    inference_drop_background_tokens: bool = True
    inference_remove_background: bool = True
    eval_max_samples: int | None = None
    skip_eval: bool = False

    resume_from_checkpoint: str | None = None
    checkpointing_steps: int = 2000
    save_only_steps: int = 1000
    project_dir: str | None = None
    log_every: int = 10

    def __post_init__(self):
        if self.project_dir is None:
            self.project_dir = f"outputs/train_{self.backbone}_albedo_res={self.resolution}_lr={self.learning_rate}"
        if self.lora_alpha is None:
            self.lora_alpha = self.lora_rank


def apply_lora(model, args: TrainArgs):
    model.requires_grad_(False)
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=mv.lora_targets(args.backbone, model, args.block_indices),
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def save_lora(model, accelerator: Accelerator, save_path: str, args: TrainArgs) -> None:
    os.makedirs(save_path, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    state = {k: v.detach().to(torch.float32).cpu().contiguous() for k, v in unwrapped.state_dict().items() if "lora_" in k}
    save_file(state, os.path.join(save_path, "dit_lora.safetensors"), metadata={"backbone": args.backbone, "lora_rank": str(args.lora_rank)})
    with open(os.path.join(save_path, "train_args.json"), "wt", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(args), f, indent=2)


def load_lora(model, path: str) -> int:
    state = load_file(os.path.join(path, "dit_lora.safetensors"))
    result = model.load_state_dict(state, strict=False)
    unexpected = [k for k in result.unexpected_keys if "lora_" in k]
    if unexpected:
        raise ValueError(f"LoRA checkpoint {path} does not match the model: unexpected keys {unexpected[:5]}")
    m = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(os.path.normpath(path)))
    return int(m.group(1)) if m else 0


def snapshot_lora_base_weights(model):
    snapshots = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "get_base_layer"):
            weight = module.get_base_layer().weight
            snapshots.append((name, weight, weight.detach().to(device="cpu", copy=True)))
    return snapshots


@torch.no_grad()
def restore_lora_base_weights(snapshots):
    for _name, weight, original in snapshots:
        weight.copy_(original)


def run_validation(pipe: mv.MVPipeline, model, eval_dataloader, save_path: str, args: TrainArgs, background_latent, accelerator):
    for i, batch in enumerate(eval_dataloader):
        if args.eval_max_samples is not None and i >= args.eval_max_samples:
            break
        sample_id = batch["id"][0]
        output_path = os.path.join(save_path, f"eval_{sample_id}_{i}.webp")
        seq = pipe.build(batch, drop_background=args.inference_drop_background_tokens)
        generated = pipe.sample(model, seq, args.inference_num_steps, args.resolution, seed=0, krea2_mu=args.krea2_mu)
        decoded = pipe.compose_and_decode(seq, generated, background_latent)
        mask_full = batch["masks_full"][0][0] if args.inference_remove_background else None
        result = mv.tensor_to_pil(decoded, mask_full, whiten_background=args.inference_remove_background)
        result.save(os.path.join(save_path, f"result_{sample_id}_{i}.webp"), format="WEBP", lossless=True)
        composite = mv.make_eval_composite(batch["image_dir"][0], result, batch["render_path"][0], args.resolution)
        composite.save(output_path, format="WEBP", quality=92)
        logger.info("validation %s -> %s", sample_id, output_path)


def main(args: TrainArgs) -> None:
    accelerator = Accelerator(
        project_dir=args.project_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed, device_specific=True)
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}[accelerator.mixed_precision]
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO, force=True)
    os.makedirs(args.project_dir, exist_ok=True)
    logger.info("UltraTex %s training: %s", args.backbone, dataclasses.asdict(args))

    spec = mv.get_spec(args.backbone)
    device = accelerator.device
    comfy_root = args.comfy_root or os.environ.get("COMFY_ROOT")

    vae = mv.load_vae(args.backbone, args.vae_path, device=device, dtype=weight_dtype, comfy_root=comfy_root)
    if args.decoder_ckpt:
        vae.load_decoder_state_dict(load_file(args.decoder_ckpt))
        logger.info("loaded foreground-aware decoder from %s", args.decoder_ckpt)
    vae.requires_grad_(False).eval()
    text_cache = mv.TextCache(args.text_cond_path, args.backbone, device)
    pipe = mv.MVPipeline(spec, vae, text_cache, device, weight_dtype, attention=args.attention, ref_timestep_zero=args.ref_timestep_zero)

    model = mv.load_backbone(args.backbone, args.model_path, device="cpu", dtype=weight_dtype, debug_mode=args.debug_model)
    sparse_config = configure_sparse_attention(
        enabled=args.sparse_attention, head_dim=model.head_dim,
        topk=args.sparse_attention_topk, blkq=args.sparse_attention_blkq, blkk=args.sparse_attention_blkk,
    )
    logger.info("attention: mode=%s backend=%s", pipe.attention, sparse_config)
    model = apply_lora(model, args)
    model.base_model.model.gradient_checkpointing = args.gradient_checkpoint
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, betas=tuple(args.adam_betas), weight_decay=args.adam_weight_decay, eps=args.adam_eps,
    )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps)

    global_step = 0
    if args.resume_from_checkpoint:
        global_step = load_lora(accelerator.unwrap_model(model), args.resume_from_checkpoint)
        logger.info("resumed LoRA from %s at step %d", args.resume_from_checkpoint, global_step)
    start_step = global_step

    dataset = FluxPairedDatasetV2(
        json_file=args.train_data_json, resolution=args.resolution, data_root=args.data_root,
        bucket_metadata_json=args.bucket_metadata_json or None,
        degrade_single_render_enabled=args.degrade_single_render, degrade_single_render_resolution=args.degrade_single_render_resolution,
    )
    sampler = LengthAwareBatchSampler(dataset, batch_size=args.batch_size, drop_last=False)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collate_fn, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers > 0)
    eval_dataset = FluxPairedDatasetV2(json_file=args.eval_data_json, resolution=args.resolution, data_root=args.data_root,
                                       degrade_single_render_enabled=args.degrade_single_render,
                                       degrade_single_render_resolution=args.degrade_single_render_resolution)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=eval_dataset.collate_fn)
    logger.info("train objects=%d eval objects=%d", len(dataset.data_dicts), len(eval_dataset.data_dicts))

    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(model, optimizer, lr_scheduler, dataloader)

    background = args.background or spec.default_background
    background_latent = mv.canonical_background_latent(vae, spec, args.resolution, background)
    training_timesteps = pipe.training_timesteps(args.resolution, args.krea2_mu)
    logger.info("timestep schedule: mu=%.3f background=%s", pipe.mu(args.resolution, args.krea2_mu), background)

    log_path = os.path.join(args.project_dir, "train_log.jsonl")
    progress = tqdm(range(args.max_train_steps), initial=start_step, total=args.max_train_steps, desc="Steps", disable=not accelerator.is_local_main_process)
    train_loss, window = 0.0, 0
    last_time = time.time()

    while global_step < args.max_train_steps:
        for batch in dataloader:
            seq = pipe.build(batch, drop_background=True)
            if global_step == start_step:
                logger.info("BTD: target %d/%d tokens kept, reference tokens kept %s", seq.target_tokens.shape[1],
                            seq.target_token_count, [r.shape[1] for r in seq.ref_tokens])
            x_1 = seq.target_tokens.float()
            t = training_timesteps[torch.randint(0, training_timesteps.shape[0], (x_1.shape[0],), device=device)]
            x_0 = torch.randn_like(x_1)
            x_t = (1 - t[:, None, None]) * x_1 + t[:, None, None] * x_0

            with accelerator.accumulate(model):
                pred = pipe.predict(model, seq, x_t, t)
                loss = mv.flow_loss(pred, x_0, x_1)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if not torch.isfinite(torch.as_tensor(grad_norm)).all():
                        optimizer.zero_grad()
                        raise FloatingPointError(f"non-finite gradient norm before step {global_step + 1} (ids={batch.get('id')})")
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                train_loss += loss.detach().item()
                window += 1
                if global_step % args.log_every == 0:
                    now = time.time()
                    record = {"step": global_step, "loss": train_loss / window, "lr": lr_scheduler.get_last_lr()[0],
                              "sec_per_step": (now - last_time) / window, "tokens": int(seq.target_tokens.shape[1] + sum(r.shape[1] for r in seq.ref_tokens)),
                              "max_mem_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
                    with open(log_path, "at", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    progress.set_postfix(loss=f"{record['loss']:.4f}", spp=f"{record['sec_per_step']:.1f}")
                    train_loss, window, last_time = 0.0, 0, now

                should_checkpoint = global_step % args.checkpointing_steps == 0 or global_step == start_step + 100
                should_save_only = global_step % args.save_only_steps == 0 and not should_checkpoint
                if should_save_only or should_checkpoint:
                    save_path = os.path.join(args.project_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process:
                        save_lora(model, accelerator, save_path, args)
                        logger.info("saved LoRA to %s", save_path)
                if should_checkpoint and not args.skip_eval and accelerator.is_main_process:
                    model.eval()
                    inference_model = accelerator.unwrap_model(model)
                    snapshots = snapshot_lora_base_weights(inference_model)
                    inference_model.merge_adapter(safe_merge=True)
                    torch.cuda.empty_cache()
                    try:
                        with torch.no_grad():
                            run_validation(pipe, inference_model, eval_dataloader, save_path, args, background_latent, accelerator)
                    finally:
                        inference_model.unmerge_adapter()
                        restore_lora_base_weights(snapshots)
                        del snapshots
                        torch.cuda.empty_cache()
                        gc.collect()
                        model.train()
            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = transformers.HfArgumentParser([TrainArgs])
    (args,) = parser.parse_args_into_dataclasses(args_file_flag="--config")
    main(args)

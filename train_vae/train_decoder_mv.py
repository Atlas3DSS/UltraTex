"""Foreground-aware VAE decoder fine-tuning for the Krea 2 / Qwen-Image 2.1 backbones.

Same objective as ``train_decoder_flux2.py``: the encoder and latent statistics
stay frozen, the background cells of every encoded view are replaced with the
canonical background latent (black RGB for Krea 2, white + transparent RGBA
for Qwen-Image 2.1), and only the decoder (plus its post-quant conv) is trained
with an L2 loss restricted to foreground pixels. For the RGBA decoder the alpha
channel is additionally supervised everywhere so the decoded mask stays usable.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from typing import Literal

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultratex import mv
from ultratex.data.dataset import FluxPairedDatasetV2

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DecoderArgs:
    backbone: Literal["krea2", "qwen21"] = "qwen21"
    vae_path: str | None = None
    comfy_root: str | None = None
    train_data_json: str = "data/train_demo.json"
    eval_data_json: str = "data/eval_demo.json"
    data_root: str | None = None
    resolution: int = 2048
    background: str | None = None
    lr: float = 1e-5
    max_train_steps: int = 3000
    batch_size: int = 1
    num_workers: int = 8
    save_every: int = 500
    eval_every: int = 500
    eval_max_samples: int = 4
    alpha_weight: float = 1.0  # RGBA only: weight of the full-image alpha L2 term
    seed: int = 0
    output_dir: str | None = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = f"outputs/train_vae_{self.backbone}"


def mix_latent(img_latent, bg_latent, mask_tokens):
    """Keep foreground latent cells, replace background cells with the canonical latent."""
    b, _, h, w = img_latent.shape
    mask = mask_tokens.reshape(b, 1, mask_tokens.shape[-2], mask_tokens.shape[-1]).float().to(img_latent.device)
    if mask.shape[-2:] != (h, w):  # token grid (1/16) -> latent grid (1/8 for the Krea 2 VAE)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
    return torch.where(mask != 0, img_latent, bg_latent)


def foreground_l2(pred, target, mask_full):
    fg = mask_full.to(pred.device) != 0  # [B, H, W]
    pred_fg = pred[:, :3].permute(0, 2, 3, 1)[fg]
    target_fg = target[:, :3].permute(0, 2, 3, 1)[fg]
    return F.mse_loss(pred_fg.float(), target_fg.float())


def main(args: DecoderArgs) -> None:
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = mv.get_spec(args.backbone)
    os.makedirs(args.output_dir, exist_ok=True)
    comfy_root = args.comfy_root or os.environ.get("COMFY_ROOT")

    vae = mv.load_vae(args.backbone, args.vae_path, device=device, dtype=torch.float32, comfy_root=comfy_root)
    vae.requires_grad_(False)
    params = vae.trainable_decoder_parameters()
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    logger.info("trainable decoder parameters: %.1fM", sum(p.numel() for p in params) / 1e6)

    train_ds = FluxPairedDatasetV2(args.train_data_json, args.resolution, None, data_root=args.data_root)
    eval_ds = FluxPairedDatasetV2(args.eval_data_json, args.resolution, None, data_root=args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate_fn, num_workers=args.num_workers, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=eval_ds.collate_fn)

    background = args.background or spec.default_background
    bg_latent = mv.canonical_background_latent(vae, spec, args.resolution, background)  # [1, C, 2r, 3r]
    tile = args.resolution // mv.TOKEN_STRIDE * (mv.TOKEN_STRIDE // vae.downscale)  # latent cells per tile side
    rgba = spec.image_channels == 4
    log_path = os.path.join(args.output_dir, "train_log.jsonl")

    def canvas_from_batch(batch):
        img = batch["img"].to(device)
        if rgba:
            img = mv.with_alpha(img, torch.stack([m[0] for m in batch["masks_full"]]))
        return img

    def eval_once(step):
        vae.eval()
        psnr_sum, n = 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                if i >= args.eval_max_samples:
                    break
                img = canvas_from_batch(batch)
                z = mv.encode_grid(vae, img)
                z = mix_latent(z, bg_latent.expand(z.shape[0], -1, -1, -1), torch.stack([m[0] for m in batch["masks"]]))
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    decoded = mv.decode_grid(vae, z)
                mask_full = torch.stack([m[0] for m in batch["masks_full"]])
                mse = foreground_l2(decoded, img, mask_full).item()
                psnr_sum += 10 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()
                n += 1
                if step == 0 or i == 0:
                    mv.tensor_to_pil(decoded, mask_full[0]).save(os.path.join(args.output_dir, f"eval_step{step}_{batch['id'][0]}.webp"), quality=92)
        vae.train()
        return psnr_sum / max(n, 1)

    vae.train()
    step = 0
    it = iter(train_loader)
    t0 = time.time()
    progress = tqdm(total=args.max_train_steps)
    while step < args.max_train_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        img = canvas_from_batch(batch)
        mask_tokens = torch.stack([m[0] for m in batch["masks"]])
        mask_full = torch.stack([m[0] for m in batch["masks_full"]])
        with torch.no_grad():
            z = mv.encode_grid(vae, img)
            z_mix = mix_latent(z, bg_latent.expand(z.shape[0], -1, -1, -1), mask_tokens)

        loss_total = 0.0
        for r in range(mv.GRID_ROWS):
            for c in range(mv.GRID_COLS):
                z_tile = z_mix[:, :, r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    pred = vae.decode(z_tile)
                hs, ws = r * args.resolution, c * args.resolution
                tgt = img[:, :, hs:hs + args.resolution, ws:ws + args.resolution]
                m = mask_full[:, hs:hs + args.resolution, ws:ws + args.resolution]
                loss = foreground_l2(pred, tgt, m)
                if rgba:
                    loss = loss + args.alpha_weight * F.mse_loss(pred[:, 3].float(), tgt[:, 3].float())
                loss.backward()
                loss_total += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        progress.update(1)
        progress.set_postfix(loss=f"{loss_total / 6:.5f}")
        with open(log_path, "at", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "loss": loss_total / 6, "elapsed": time.time() - t0}) + "\n")
        if step % args.save_every == 0 or step == args.max_train_steps:
            path = os.path.join(args.output_dir, f"decoder_step{step}.safetensors")
            save_file({k: v.detach().cpu().contiguous() for k, v in vae.decoder_state_dict().items()}, path,
                      metadata={"backbone": args.backbone, "background": background})
            logger.info("saved %s", path)
        if step % args.eval_every == 0 or step == args.max_train_steps:
            psnr = eval_once(step)
            logger.info("step %d foreground PSNR %.2f dB", step, psnr)
            with open(log_path, "at", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "eval_fg_psnr": psnr}) + "\n")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser([DecoderArgs])
    (args,) = parser.parse_args_into_dataclasses(args_file_flag="--config")
    main(args)

#!/usr/bin/env python
"""Unpack G-buffer TexVerse canonical buckets and write UltraTex JSON manifests.

  python tools/prepare_texverse.py --zips /path/to/zips/canonical/bsdf --out /path/to/assets \
      --train-json data/texverse_train.json --eval-json data/texverse_eval.json --eval-count 32

Each bucket zip holds per-asset zips (``<id>_<res>.zip``) or asset directories;
both layouts are handled. Assets missing the six canonical views in
``albedo/`` or ``bump_normal_world/`` or lacking ``render_ref_*`` references
are skipped. The script also writes ``<out>/foreground_ratios.json`` (asset id
-> mean foreground fraction on the 1/16 token grid, i.e. the length-aware
batch sampler metadata; pass it as ``--bucket_metadata_json``).

The dataset is gated on Hugging Face: request access with the owner's account,
then ``hf download YiboZhang2001/G-buffer-TexVerse canonical/bsdf/00.zip --repo-type dataset``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import zipfile

import numpy as np
from PIL import Image

REQUIRED = ("albedo", "bump_normal_world", "position")


def unpack_bucket(zip_path: str, out_dir: str) -> list[str]:
    assets = []
    with zipfile.ZipFile(zip_path) as bucket:
        names = bucket.namelist()
        inner_zips = [n for n in names if n.lower().endswith(".zip")]
        if inner_zips:
            for name in inner_zips:
                asset_id = os.path.splitext(os.path.basename(name))[0]
                target = os.path.join(out_dir, asset_id)
                if os.path.isdir(target) and os.path.isdir(os.path.join(target, "albedo")):
                    assets.append(target)
                    continue
                tmp = os.path.join(out_dir, f".{asset_id}.zip")
                with bucket.open(name) as src, open(tmp, "wb") as dst:
                    dst.write(src.read())
                with zipfile.ZipFile(tmp) as inner:
                    inner.extractall(target)
                os.remove(tmp)
                # tolerate one wrapping directory inside the inner zip
                entries = os.listdir(target)
                if len(entries) == 1 and os.path.isdir(os.path.join(target, entries[0])) and not os.path.isdir(os.path.join(target, "albedo")):
                    wrap = os.path.join(target, entries[0])
                    for e in os.listdir(wrap):
                        os.rename(os.path.join(wrap, e), os.path.join(target, e))
                    os.rmdir(wrap)
                assets.append(target)
        else:
            bucket.extractall(out_dir)
            tops = sorted({n.split("/")[0] for n in names if "/" in n})
            assets.extend(os.path.join(out_dir, t) for t in tops)
    return assets


def asset_ok(path: str) -> bool:
    for sub in REQUIRED:
        if not all(os.path.isfile(os.path.join(path, sub, f"{i:03d}.webp")) for i in range(6)):
            return False
    return any(os.path.isfile(os.path.join(path, f"render_ref_{k}", "000.webp")) for k in range(3))


def foreground_ratio(path: str, grid: int = 128) -> float:
    ratios = []
    for i in range(6):
        alpha = Image.open(os.path.join(path, "albedo", f"{i:03d}.webp")).convert("RGBA").split()[3]
        small = np.asarray(alpha.resize((grid, grid), Image.Resampling.BILINEAR))
        ratios.append(float((small > 0).mean()))
    return float(np.mean(ratios))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", required=True, help="directory holding bucket zips (or a single zip)")
    ap.add_argument("--out", required=True, help="asset directory root (one folder per asset)")
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--eval-count", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-ratios", action="store_true", help="write manifests only (no foreground_ratios.json)")
    ap.add_argument("--no-unpack", action="store_true", help="assume --out already holds the unpacked assets")
    args = ap.parse_args()

    zips = [args.zips] if os.path.isfile(args.zips) else sorted(
        os.path.join(args.zips, f) for f in os.listdir(args.zips) if f.endswith(".zip"))
    os.makedirs(args.out, exist_ok=True)
    assets = []
    if args.no_unpack:
        assets = [os.path.join(args.out, d) for d in os.listdir(args.out) if os.path.isdir(os.path.join(args.out, d)) and not d.startswith(".")]
    else:
        for z in zips:
            found = unpack_bucket(z, args.out)
            print(f"{z}: {len(found)} assets")
            assets.extend(found)
    assets = sorted({a for a in assets if asset_ok(a)})
    print(f"{len(assets)} usable assets")

    ratios = {}
    entries = []
    for a in assets:
        asset_id = os.path.basename(a)
        if not args.skip_ratios:
            ratios[asset_id] = foreground_ratio(a)
        entries.append({"prompt": "", "image_dir": os.path.abspath(a), "id": asset_id, "id_only": asset_id.rsplit("_", 1)[0]})
    if not args.skip_ratios:
        with open(os.path.join(args.out, "foreground_ratios.json"), "wt", encoding="utf-8") as f:
            json.dump(ratios, f)
    random.Random(args.seed).shuffle(entries)
    eval_entries, train_entries = entries[: args.eval_count], entries[args.eval_count:]
    for path, data in ((args.train_json, train_entries), (args.eval_json, eval_entries)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"wrote {path}: {len(data)} objects")


if __name__ == "__main__":
    main()

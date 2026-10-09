#!/usr/bin/env python
"""Back-project the six generated albedo views onto the mesh's UV texture.

Inputs: the G-buffer asset directory from ``tools/render_gbuffer.py`` (uses
``uv/00k.png`` and ``bump_normal_world/00k.*``) and the generated 2x3 albedo
canvas (``result_*.webp`` from UltraTex inference, view order
[0,1,3 / 2,4,5]). Every foreground pixel of every view is splatted into the
texel its UV points at, weighted by how squarely the surface faces that camera
(cos^power); holes are filled by iterative dilation.

  python tools/bake_views_to_uv.py --asset-dir DIR --canvas result.webp --out texture.png [--size 2048]
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

CANVAS_ORDER = [[0, 1, 3], [2, 4, 5]]
# camera look directions in the normal-map storage frame ((X, Y, -Z) of Blender): front looks +Y, ...
LOOK = {0: (0, 1, 0), 1: (1, 0, 0), 2: (0, -1, 0), 3: (-1, 0, 0), 4: (0, 0, 1), 5: (0, 0, -1)}


def load_rgba(path):
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-dir", required=True)
    ap.add_argument("--canvas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--power", type=float, default=4.0, help="facing weight exponent")
    ap.add_argument("--min-cos", type=float, default=0.15)
    ap.add_argument("--fill-iterations", type=int, default=64)
    args = ap.parse_args()

    canvas = np.asarray(Image.open(args.canvas).convert("RGB")).astype(np.float32) / 255.0
    cell_h, cell_w = canvas.shape[0] // 2, canvas.shape[1] // 3
    size = args.size
    acc = np.zeros((size, size, 3), np.float64)
    wacc = np.zeros((size, size), np.float64)

    for r, row in enumerate(CANVAS_ORDER):
        for c, view in enumerate(row):
            uv_path = os.path.join(args.asset_dir, "uv", f"{view:03d}.png")
            raw = cv2.imread(uv_path, cv2.IMREAD_UNCHANGED)  # 16-bit BGRA
            if raw is None:
                raise FileNotFoundError(f"cannot read {uv_path}")
            raw = raw.astype(np.float32) / (65535.0 if raw.dtype == np.uint16 else 255.0)
            uv = np.concatenate([raw[..., 2::-1], raw[..., 3:]], axis=-1) if raw.shape[-1] == 4 else raw[..., ::-1]
            normal_path = next(p for p in (os.path.join(args.asset_dir, "bump_normal_world", f"{view:03d}.{e}") for e in ("webp", "png")) if os.path.exists(p))
            nrm = load_rgba(normal_path)
            h, w = uv.shape[:2]
            tile = canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            if tile.shape[:2] != (h, w):
                tile = cv2.resize(tile, (w, h), interpolation=cv2.INTER_AREA)
            if nrm.shape[:2] != (h, w):
                nrm = cv2.resize(nrm, (w, h), interpolation=cv2.INTER_NEAREST)
            alpha = (uv[..., 3] > 0.5) if uv.shape[-1] == 4 else (nrm[..., 3] > 0)
            alpha &= nrm[..., 3] > 0
            n = nrm[..., :3] / 255.0 * 2 - 1
            cosv = -(n @ np.asarray(LOOK[view], np.float32))  # facing the camera -> positive
            weight = np.clip(cosv, 0, 1) ** args.power
            keep = alpha & (cosv > args.min_cos)
            u = np.clip((uv[..., 0] * size).astype(np.int64), 0, size - 1)[keep]
            v = np.clip(((1.0 - uv[..., 1]) * size).astype(np.int64), 0, size - 1)[keep]
            wk = weight[keep]
            col = tile[keep]
            np.add.at(acc, (v, u), col * wk[:, None])
            np.add.at(wacc, (v, u), wk)
            print(f"view {view}: {int(keep.sum())} pixels splatted")

    covered = wacc > 0
    tex = np.zeros_like(acc)
    tex[covered] = acc[covered] / wacc[covered][:, None]
    # hole fill: dilate covered colours outward
    filled = covered.copy()
    tex_fill = tex.copy()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(args.fill_iterations):
        if filled.all():
            break
        blurred = cv2.blur(tex_fill * filled[..., None], (3, 3))
        count = cv2.blur(filled.astype(np.float32), (3, 3))
        grow = cv2.dilate(filled.astype(np.uint8), kernel).astype(bool) & ~filled
        tex_fill[grow] = blurred[grow] / np.maximum(count[grow], 1e-6)[:, None]
        filled |= grow
    print(f"texels covered directly: {covered.mean() * 100:.1f}%, after fill: {filled.mean() * 100:.1f}%")
    Image.fromarray((np.clip(tex_fill, 0, 1) * 255).round().astype(np.uint8)).save(args.out)
    print("BAKE_OK", args.out)


if __name__ == "__main__":
    main()

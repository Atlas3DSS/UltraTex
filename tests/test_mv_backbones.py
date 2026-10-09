"""CPU tests for the Krea 2 / Qwen-Image 2.1 UltraTex port (tiny random weights)."""

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultratex.backbones import common  # noqa: E402
from ultratex.backbones.common import block_causal_attention, dense_attention, foreground_indices, gather_sequence  # noqa: E402
from ultratex.backbones.krea2.model import Krea2, Krea2Params  # noqa: E402
from ultratex.backbones.qwen21.model import Qwen21Params, QwenImage21  # noqa: E402
from ultratex.backbones.qwen21.util import pack_image as qwen_pack  # noqa: E402
from ultratex.backbones.krea2.util import pack_image as krea_pack  # noqa: E402
from ultratex import mv  # noqa: E402

torch.manual_seed(0)


def tiny_krea2():
    p = Krea2Params(features=64, tdim=16, txtdim=32, heads=4, kvheads=2, multiplier=1, layers=2, patch=2, channels=4,
                    txtlayers=3, txtheads=2, txtkvheads=2)
    return Krea2(p).float()


def tiny_qwen21():
    p = Qwen21Params(in_channels=8, out_channels=8, num_layers=2, attention_head_dim=16, num_attention_heads=4,
                     context_in_dim=24, mlp_ratio=2, axes_dims_rope=(4, 6, 6))
    return QwenImage21(p).float()


def test_block_causal_matches_masked_sdpa():
    b, h, n, d = 1, 2, 23, 8
    q, k, v = (torch.randn(b, h, n, d) for _ in range(3))
    segments = [(0, 5, True), (5, 12, False), (12, 17, False), (17, 23, False)]
    out = block_causal_attention(q, k, v, segments)
    mask = torch.zeros(n, n, dtype=torch.bool)
    for start, end, causal in segments:
        for i in range(start, end):
            mask[i, :end] = True
            if causal:
                mask[i, i + 1:end] = False
    ref = dense_attention(q, k, v, mask)
    assert torch.allclose(out, ref, atol=1e-5)


def test_krea2_forward_and_timestep_zero():
    m = tiny_krea2()
    latent = torch.randn(1, 4, 8, 12)
    tokens, ids = krea_pack(latent, 0, patch=2)
    ref_tokens, ref_ids = krea_pack(torch.randn(1, 4, 8, 12), 1, patch=2)
    assert tokens.shape == (1, 24, 16) and ids.shape == (1, 24, 3)
    assert ids[0, -1].tolist() == [0.0, 3.0, 5.0]
    ctx = torch.randn(1, 5, 3 * 32)
    img = torch.cat([tokens, ref_tokens], 1)
    img_ids = torch.cat([ids, ref_ids], 1)
    t = torch.tensor([0.4])
    out = m(img, img_ids, t, ctx)
    assert out.shape == (1, 48, 16)
    out0 = m(img, img_ids, t, ctx, timestep_zero_index=24)
    assert out0.shape == out.shape and not torch.allclose(out, out0)
    # dropping background tokens keeps the survivors' ids
    keep = torch.tensor([[0, 3, 7, 20]])
    kept_ids = gather_sequence(ids, keep)
    assert kept_ids[0, 2].tolist() == ids[0, 7].tolist()


def test_qwen21_pack_centres_on_target():
    latent = torch.randn(1, 8, 4, 6)
    tokens, ids = qwen_pack(latent, frame=7, target_hw=(4, 6))
    assert tokens.shape == (1, 24, 8)
    assert ids[0, 0].tolist() == [7.0, -2.0, -3.0]
    assert ids[0, -1].tolist() == [7.0, 1.0, 2.0]
    # odd reference on an even target gets the half-token parity shift
    _, ids_odd = qwen_pack(torch.randn(1, 8, 3, 5), frame=1, target_hw=(4, 6))
    assert ids_odd[0, 0, 1].item() == pytest.approx(-2 + 0.5)


def test_qwen21_prefix_cache_matches_full_pass():
    m = tiny_qwen21().eval()
    target_latent = torch.randn(1, 8, 4, 6)
    ref_latent = torch.randn(1, 8, 4, 6)
    ctx = torch.randn(1, 3, 24)
    txt_len = ctx.shape[1]
    ref_tokens, ref_ids = qwen_pack(ref_latent, frame=txt_len, target_hw=(4, 6))
    tgt_tokens, tgt_ids = qwen_pack(target_latent, frame=txt_len + 6, target_hw=(4, 6))
    # background-prune the reference and the target
    ref_keep = torch.tensor([[1, 2, 5, 9, 13]])
    tgt_keep = torch.tensor([[0, 4, 8, 9, 10, 20]])
    ref_tokens, ref_ids = gather_sequence(ref_tokens, ref_keep), gather_sequence(ref_ids, ref_keep)
    tgt_tokens, tgt_ids = gather_sequence(tgt_tokens, tgt_keep), gather_sequence(tgt_ids, tgt_keep)
    img = torch.cat([ref_tokens, tgt_tokens], 1)
    ids = torch.cat([ref_ids, tgt_ids], 1)
    t = torch.tensor([0.7])
    with torch.no_grad():
        full = m(img, ids, t, ctx, ref_lengths=[5], attention="block_causal")
        cache = {}
        filled = m(img, ids, t, ctx, ref_lengths=[5], attention="block_causal", prefix_cache=cache)
        assert "blocks" in cache and cache["prefix_len"] == txt_len + 5
        cached = m(tgt_tokens, tgt_ids, torch.tensor([0.2]), ctx, attention="block_causal", prefix_cache=cache)
        again = m(img, ids, torch.tensor([0.2]), ctx, ref_lengths=[5], attention="block_causal")
    assert full.shape == (1, 6, 8)
    assert torch.allclose(full, filled, atol=1e-5)
    # the prefix is timestep-independent, so cached target rows equal a fresh full pass at the new t
    assert torch.allclose(cached, again, atol=1e-4)
    # and differ from bidirectional attention
    bidir = m(img, ids, torch.tensor([0.2]), ctx, ref_lengths=[5], attention="full")
    assert not torch.allclose(bidir, again, atol=1e-4)


def test_pipeline_layout_orders_and_ids():
    spec = mv.get_spec("qwen21")
    pipe = mv.MVPipeline(spec, vae=None, text_cache=None, device="cpu", weight_dtype=torch.float32)
    target = torch.randn(1, 64, 4, 6)
    refs = [torch.randn(1, 64, 4, 6), torch.randn(1, 64, 2, 2)]
    tgt_tokens, tgt_ids, packed_refs, token_hw = pipe._layout(target, refs, txt_len=10)
    assert token_hw == (4, 6)
    assert packed_refs[0][1][0, 0, 0].item() == 10.0
    assert packed_refs[1][1][0, 0, 0].item() == 16.0
    assert tgt_ids[0, 0, 0].item() == 18.0

    spec = mv.get_spec("krea2")
    pipe = mv.MVPipeline(spec, vae=None, text_cache=None, device="cpu", weight_dtype=torch.float32)
    target = torch.randn(1, 16, 8, 12)
    tgt_tokens, tgt_ids, packed_refs, token_hw = pipe._layout(target, [torch.randn(1, 16, 8, 12)], txt_len=10)
    assert token_hw == (4, 6) and tgt_tokens.shape == (1, 24, 64)
    assert tgt_ids[0, 0, 0].item() == 0.0 and packed_refs[0][1][0, 0, 0].item() == 1.0


def test_foreground_indices_and_schedule():
    mask = torch.zeros(1, 4, 6)
    mask[0, 1:3, 2:5] = 1
    idx = foreground_indices(mask, 24, "target")
    assert idx.tolist() == [[8, 9, 10, 14, 15, 16]]
    with pytest.raises(ValueError):
        foreground_indices(torch.zeros(1, 4, 6), 24, "target")
    sched = common.shifted_schedule(25, common.qwen21_mu(16384))
    assert sched[0] == 1.0 and sched[-1] == 0.0 and all(a > b for a, b in zip(sched, sched[1:]))
    assert math.isclose(common.qwen21_mu(4096), 0.6935, abs_tol=1e-3)

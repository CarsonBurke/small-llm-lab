"""Correctness tests for the FlexAttention block-sparse fine pass.

The numeric checks require CUDA (flex_attention's fused kernel), so they are
``cuda_only`` and are run by the parent through the mlq queue.  Each check
compares the flex path against an independent dense masked-softmax reference
built from the *same* ``_select`` draw, and asserts zero future leakage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "xlayer"))
import blockwire_train_gpt as bw  # noqa: E402
import flex_rewire_train_gpt as flr  # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# The unfused (uncompiled) flex_attention path applies mask_mod to full blocks
# and does not support the backward pass; production runs it under
# torch.compile (fused).  Tests that execute the kernel must do the same to
# exercise the real path, so they swap in a compiled flex_attention.
_CFLEX = None


def use_compiled_flex(monkeypatch):
    global _CFLEX
    if _CFLEX is None:
        _CFLEX = torch.compile(flr.flex_attention)
    monkeypatch.setattr(flr, "flex_attention", _CFLEX)

# head_dim 64 (real geometry), 4 query / 2 kv heads, T=256, 128-token blocks
# -> 2 query blocks and 2 kv blocks per source layer.
DIM, HEADS, KV_HEADS = 256, 4, 2
T, BLOCK = 256, 128
N_RAND = 1


def make_attn(layer, *, n_rand=N_RAND, xlayer=True, device="cuda"):
    attn = flr.FlexRewireAttention(DIM, HEADS, KV_HEADS, 10000.0, 1.0)
    attn.n_rand = n_rand
    attn.key_block = attn.query_block = BLOCK
    attn.xlayer = xlayer
    attn.configure_wiring(layer, T)
    return attn.to(device)


def _decode_mask(attn, bm, base):
    """Reconstruct the [H, T, S*T] boolean attended set implied by a BlockMask.

    Full blocks attend all their tokens; partial (diagonal) blocks attend only
    tokens passing the causal ``mask_mod`` (q_idx + base >= kv_idx).  Pure
    Python/tensor decode — no flex kernel, so it runs on CPU.
    """
    nq = attn.n_qblocks
    S = attn.n_sources
    att = torch.zeros(HEADS, T, S * T, dtype=torch.bool)
    for h in range(HEADS):
        for qi in range(nq):
            full = bm.full_kv_indices[0, h, qi, : bm.full_kv_num_blocks[0, h, qi]].tolist()
            part = bm.kv_indices[0, h, qi, : bm.kv_num_blocks[0, h, qi]].tolist()
            for t in range(qi * BLOCK, (qi + 1) * BLOCK):
                for blk in full:
                    att[h, t, blk * BLOCK : (blk + 1) * BLOCK] = True
                for blk in part:
                    for kv in range(blk * BLOCK, (blk + 1) * BLOCK):
                        if t + base >= kv:
                            att[h, t, kv] = True
    return att


def _dense_reference(attn, q, gid, valid, ks, vs):
    """Dense masked-softmax over exactly the block set the flex path attends.

    Builds an additive [B, H, T, S*T] logit mask that is finite only on:
      * the current-source diagonal block, keys causal (local pos <= t),
      * the current-source predecessor block q-1 (fully causal),
      * each valid random block (fully causal by construction).
    Then plain softmax @ v_bank, GQA-expanded.  fp64 for a tight tolerance.
    """
    bsz = q.size(0)
    group = HEADS // KV_HEADS
    nb, nq = attn.n_blocks, attn.n_qblocks
    base_blk = attn.n_earlier * nb
    S = attn.n_sources
    bank_k = torch.cat(ks, dim=2).double()  # [B, Hkv, S*T, d]
    bank_v = torch.cat(vs, dim=2).double()
    scale = attn.head_dim ** -0.5

    neg = torch.full((bsz, KV_HEADS, T, S * T), float("-inf"), dtype=torch.float64, device=q.device)
    for b in range(bsz):
        for g in range(KV_HEADS):
            for qi in range(nq):
                q0, q1 = qi * BLOCK, (qi + 1) * BLOCK
                for t in range(q0, q1):
                    # diagonal block (current source), causal within block
                    d0 = (base_blk + qi) * BLOCK
                    for j in range(BLOCK):
                        p = d0 + j
                        if p - (base_blk * BLOCK) <= t:  # local pos <= t
                            neg[b, g, t, p] = 0.0
                    # predecessor block q-1 (current source), fully causal
                    if qi > 0:
                        pr0 = (base_blk + qi - 1) * BLOCK
                        neg[b, g, t, pr0 : pr0 + BLOCK] = 0.0
                    # random blocks
                    for s in range(attn.n_dyn):
                        if not bool(valid[b, g, qi, s]):
                            continue
                        blk = int(gid[b, g, qi, s])
                        neg[b, g, t, blk * BLOCK : (blk + 1) * BLOCK] = 0.0

    # GQA-expand kv heads to query heads
    kq = bank_k.repeat_interleave(group, dim=1)  # [B, H, S*T, d]
    vq = bank_v.repeat_interleave(group, dim=1)
    negq = neg.repeat_interleave(group, dim=1)  # [B, H, T, S*T]
    logits = torch.einsum("bhtd,bhsd->bhts", q.double(), kq) * scale + negq
    w = torch.softmax(logits, dim=-1)
    return torch.einsum("bhts,bhsd->bhtd", w, vq)


def test_block_mask_semantics_cpu(monkeypatch):
    """CPU: the BlockMask attends EXACTLY the intended causal key set.

    Builds the mask from a hand-crafted (causal) selection and checks its
    decoded attended set against an independent reference — verifying the
    full/partial split, the diagonal causal mask_mod, GQA head expansion, and
    the BCSR index layout without needing a GPU or the flex kernel.
    """
    monkeypatch.setattr(flr, "_TRAIN_SEQ_LEN", T)
    layer = 2
    attn = make_attn(layer, n_rand=2, device="cpu")
    nb, nq = attn.n_blocks, attn.n_qblocks
    base_blk = attn.n_earlier * nb
    base = attn.n_earlier * T
    S = attn.n_sources
    group = HEADS // KV_HEADS
    # hand-built causal selection: query block 1 draws the two earlier sources'
    # block 0 (ids 0 and nb); query block 0 draws nothing valid.
    gid = torch.zeros(1, KV_HEADS, nq, 2, dtype=torch.int64)
    valid = torch.zeros(1, KV_HEADS, nq, 2, dtype=torch.bool)
    gid[0, :, 1, 0] = 0
    gid[0, :, 1, 1] = nb
    valid[0, :, 1, :] = True
    bm = attn.build_block_mask(gid, valid, 1)

    att = _decode_mask(attn, bm, base)
    # independent reference attended set (per kv head, then expand to q heads)
    ref = torch.zeros(KV_HEADS, T, S * T, dtype=torch.bool)
    for g in range(KV_HEADS):
        for qi in range(nq):
            for t in range(qi * BLOCK, (qi + 1) * BLOCK):
                d0 = (base_blk + qi) * BLOCK
                for j in range(BLOCK):
                    p = d0 + j
                    if p - base_blk * BLOCK <= t:
                        ref[g, t, p] = True
                if qi > 0:
                    pr = (base_blk + qi - 1) * BLOCK
                    ref[g, t, pr : pr + BLOCK] = True
                for s in range(2):
                    if bool(valid[0, g, qi, s]):
                        blk = int(gid[0, g, qi, s])
                        ref[g, t, blk * BLOCK : (blk + 1) * BLOCK] = True
    ref = ref.repeat_interleave(group, dim=0)
    assert torch.equal(att, ref)
    # no attended key may be a future token of the current source
    cur0 = attn.n_earlier * T
    for h in range(HEADS):
        for t in range(T):
            fut = att[h, t, cur0:]  # current-source columns
            future_local = torch.nonzero(fut)[:, 0]
            assert bool((future_local <= t).all()), f"future leak h{h} t{t}"


@cuda_only
def test_flex_matches_dense_reference(monkeypatch):
    monkeypatch.setattr(flr, "_TRAIN_SEQ_LEN", T)
    use_compiled_flex(monkeypatch)
    torch.manual_seed(3)
    layer = 2
    attn = make_attn(layer).train()
    bsz = 2
    q = torch.randn(bsz, HEADS, T, attn.head_dim, device="cuda")
    ks = [torch.randn(bsz, KV_HEADS, T, attn.head_dim, device="cuda") for _ in range(layer + 1)]
    vs = [torch.randn(bsz, KV_HEADS, T, attn.head_dim, device="cuda") for _ in range(layer + 1)]
    bw._STEP_BUF.fill_(11)
    with torch.no_grad():
        gid, valid = attn._select(bsz)
        # exercise the same BlockMask the pre-hook would build
        y, stats = attn._fine(q, gid, valid, *ks, *vs)
    ref = _dense_reference(attn, q, gid, valid, ks, vs)
    torch.testing.assert_close(y.double(), ref, rtol=1e-4, atol=1e-4)
    assert stats[4].item() == 0.0  # bw_leak column must be zero


@cuda_only
def test_no_future_leakage(monkeypatch):
    """d y[t0] / d x[t] == 0 for every t > t0 (single flex layer, autograd)."""
    monkeypatch.setattr(flr, "_TRAIN_SEQ_LEN", T)
    monkeypatch.setattr(bw, "_TRAIN_SEQ_LEN", T)  # inherited forward's seqlen guard
    use_compiled_flex(monkeypatch)
    torch.manual_seed(5)
    attn = make_attn(0).train()  # layer 0: within-current-layer causality only
    bw._STEP_BUF.fill_(7)
    bw._STATE["i"] = 0
    bw._STATE["bank"] = None
    bw._STATE["masks"] = {}  # force the _fine eager-build fallback
    x = torch.randn(1, T, DIM, device="cuda", requires_grad=True)
    y = attn(x)
    t0 = 100
    (grad,) = torch.autograd.grad(y[0, t0].sum(), x)
    future = grad[0, t0 + 1 :]
    assert torch.all(future == 0), f"future leak: {future.abs().max().item()}"
    assert grad[0, : t0 + 1].abs().sum() > 0


@cuda_only
def test_prehook_builds_all_layer_masks(monkeypatch):
    """The forward_pre_hook populates one BlockMask per layer, sized correctly."""
    monkeypatch.setattr(flr, "_TRAIN_SEQ_LEN", T)
    torch.manual_seed(1)
    modules = [make_attn(i).train() for i in range(3)]

    class _M:
        blocks = [type("B", (), {"attn": m})() for m in modules]

    bw._STEP_BUF.fill_(4)
    flr._build_masks_prehook(_M(), (torch.zeros(2, T, dtype=torch.long, device="cuda"),))
    masks = bw._STATE["masks"]
    assert set(masks) == {0, 1, 2}
    for li, m in enumerate(modules):
        bm = masks[li]
        # block_mask.shape is (q_len, kv_len); kv grows with source count.
        assert bm.shape[-2] == T
        assert bm.shape[-1] == (li + 1) * T
        # H dimension expanded to query heads for GQA indexing.
        assert bm.kv_num_blocks.shape[:2] == (2, HEADS)


@cuda_only
def test_full_and_partial_blocks_are_disjoint(monkeypatch):
    """No kv block is both a full (unmasked) and a partial (diagonal) block."""
    monkeypatch.setattr(flr, "_TRAIN_SEQ_LEN", T)
    torch.manual_seed(2)
    attn = make_attn(2, n_rand=2).train()
    bw._STEP_BUF.fill_(9)
    with torch.no_grad():
        gid, valid = attn._select(2)
        bm = attn.build_block_mask(gid, valid, 2)
    B, H, NQ = bm.kv_num_blocks.shape
    for b in range(B):
        for h in range(H):
            for qi in range(NQ):
                part = set(bm.kv_indices[b, h, qi, : bm.kv_num_blocks[b, h, qi]].tolist())
                full = set(bm.full_kv_indices[b, h, qi, : bm.full_kv_num_blocks[b, h, qi]].tolist())
                assert part.isdisjoint(full), f"overlap at ({b},{h},{qi}): {part & full}"
                assert len(part) == 1  # exactly the diagonal block


@cuda_only
def test_within_layer_control_ignores_bank(monkeypatch):
    """SPARSE_XLAYER=0: earlier layers' k/v never influence the output."""
    monkeypatch.setattr(flr, "_TRAIN_SEQ_LEN", T)
    monkeypatch.setattr(bw, "_TRAIN_SEQ_LEN", T)  # inherited forward's seqlen guard
    use_compiled_flex(monkeypatch)
    torch.manual_seed(4)
    attn = make_attn(1, xlayer=False).eval()
    x = torch.randn(1, T, DIM, device="cuda")
    bw._STEP_BUF.fill_(3)
    bw._STATE["i"] = 1
    bw._STATE["bank"] = [(torch.randn(1, KV_HEADS, T, attn.head_dim, device="cuda"),
                          torch.randn(1, KV_HEADS, T, attn.head_dim, device="cuda"))]  # poison
    bw._STATE["masks"] = {}
    h1 = attn(x)
    bw._STATE["i"] = 1
    bw._STATE["bank"] = None
    bw._STATE["masks"] = {}
    h2 = attn(x)
    torch.testing.assert_close(h1, h2, rtol=0, atol=0)


@cuda_only
def test_no_null_param_when_disabled(monkeypatch):
    """Null is off by default: no null_bias parameter (avoids a dead DDP param)."""
    attn = make_attn(1)
    assert not any("null_bias" in n for n, _ in attn.named_parameters())

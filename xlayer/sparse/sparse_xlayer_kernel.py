"""Churn tail for the CROSS-LAYER arm: draws roam (layer', pos'), not just pos'.

Fork of ``sparse_churn_kernel`` for ``sparse_xlayer_train_gpt``.  Wiring
indices are flat ``lin = layer' * T + pos'`` into a K/V stack laid out
``[B, Hkv, n_layers*T, D]`` (torch.cat along the row axis, layer-major), so
the attention kernel is completely unchanged - only the churn tail needs to
know the pool is two-dimensional:

* Rewire draws sample (layer', pos') uniformly over layers 0..consumer and
  positions 0..t, where consumer = LI + 1 is the layer that will use this
  wiring (its own K/V exist by the time it gathers).  NSRC = consumer + 1
  segments.
* Slot 0 (self) is REBASED each churn to ``(NSRC - 1) * T + t`` - the
  consumer's own freshly computed K at the query position.  Without this the
  self edge would go stale, pointing at layer 0's K forever.  The rebase
  happens before dedup so same-round draws landing on the consumer-self lin
  are killed (inherited slots can never hold it: their lins predate that
  segment).
* Everything else - entropy/null gate, p <= 1/(m+1) eligibility, dedup
  win rule, banked stats - is identical to ``sparse_churn_kernel`` and is
  spec'd by the torch reference ``_churn`` in the xlayer training script.

Randomness: Philox streams keyed off the same device seed tensor; the layer
draw uses a third stream constant so gate/pos/layer draws are independent.
int32 stays safe: max CONSUMED lin = n_layers * T - 1 (9215 at production);
the last layer's stats-path churn stores lins up to (n_layers+1)*T - 1 but
that wiring is never consumed (GPT.forward resets it AND layer 0 rebuilds
structurally - both guards must survive future edits, since the attention
kernel has no bounds check).  All in-kernel constants are below 2^31.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import torch
import triton
import triton.language as tl
from torch import Tensor

_STATS_BANKS = 64


@triton.jit
def _xlayer_churn_tail_kernel(
    IDX, VALID, P, PN, SEED, NIDX, NVALID, STATS,
    eps, T, LI, NSRC,
    KB: tl.constexpr,
    DO_STATS: tl.constexpr,
    BANKS: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid % T
    offs_k = tl.arange(0, KB)
    row = pid * KB

    idx_raw = tl.load(IDX + row + offs_k)
    val = tl.load(VALID + row + offs_k) != 0
    p = tl.load(P + row + offs_k)
    pn = tl.load(PN + pid)

    m_incl = tl.sum(val.to(tl.float32), axis=0) + 1.0
    on_sup = val & (p > 0)
    plog = tl.where(on_sup, p * tl.log(tl.maximum(p, 1e-30)), 0.0)
    h_row = -tl.sum(plog, axis=0) - tl.where(pn > 0, pn * tl.log(tl.maximum(pn, 1e-30)), 0.0)
    h_norm = h_row / tl.maximum(tl.log(m_incl), 0.6931471805599453)
    q_rw = tl.minimum(tl.maximum(tl.maximum(h_norm, pn), eps), 1.0)

    share = 1.0 / m_incl
    eligible = (p <= share) & (offs_k != 0)  # slot 0 (self) never rewires

    seed = (tl.load(SEED) + LI * 1000003).to(tl.int32)
    lane = row + offs_k
    u_gate = tl.rand(seed, lane)
    u_pos = tl.rand(seed ^ 0x5851F42D, lane)
    u_lay = tl.rand(seed ^ 0x2545F491, lane)
    rewire = eligible & (u_gate < q_rw)
    # causality relies on tl.rand < 1.0 (max 1 - 2^-24) AND u*n never rounding
    # up to n for n <= 2^24 - see xlayer/sparse/sparse_churn_kernel.py for the derivation.
    # Here it bounds BOTH draws: pos <= t and layer < NSRC.
    d_pos = (u_pos * (t + 1).to(tl.float32)).to(tl.int32)
    d_lay = (u_lay * NSRC.to(tl.float32)).to(tl.int32)
    draws = d_lay * T + d_pos

    new_idx = tl.where(rewire, draws, idx_raw)  # dead slots keep their stored idx
    # rebase the self edge to the CONSUMER layer's own K at this position;
    # must precede dedup so same-round draws on this lin get killed
    self_lin = (NSRC - 1) * T + t
    new_idx = tl.where(offs_k == 0, self_lin, new_idx)
    new_val = val | rewire

    # dedup in registers: kept slots always beat rewired candidates at the same
    # index; among candidates the earliest slot wins; never kills all copies
    eq = new_idx[:, None] == new_idx[None, :]
    earlier = offs_k[None, :] < offs_k[:, None]
    wins = new_val[None, :] & (~rewire[None, :] | earlier)
    dup = tl.sum((eq & wins).to(tl.int32), axis=1) > 0
    new_val = new_val & ~(rewire & dup)

    tl.store(NIDX + row + offs_k, new_idx)
    tl.store(NVALID + row + offs_k, new_val.to(tl.int8))

    if DO_STATS:
        bank = (pid % BANKS) * 5
        tl.atomic_add(STATS + bank + 0, tl.sum(rewire.to(tl.float32), axis=0))
        tl.atomic_add(STATS + bank + 1, tl.sum(on_sup.to(tl.float32), axis=0))
        tl.atomic_add(STATS + bank + 2, m_incl - 1.0)
        tl.atomic_add(STATS + bank + 3, h_norm)
        tl.atomic_add(STATS + bank + 4, 1.0)


@torch.library.custom_op("parameter_golf::xlayer_churn_rewire", mutates_args=())
def xlayer_churn_rewire(
    idx: Tensor, valid: Tensor, p: Tensor, pn: Tensor, seed: Tensor,
    layer: int, churn_eps: float, do_stats: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    bsz, num_heads, seqlen, kb = idx.shape
    idx = idx.contiguous()
    valid = valid.contiguous()
    p = p.contiguous()
    pn = pn.contiguous()
    new_idx = torch.empty_like(idx)
    new_valid = torch.empty(idx.shape, device=idx.device, dtype=torch.int8)
    stats = torch.zeros(_STATS_BANKS * 5, device=idx.device, dtype=torch.float32)
    grid = (bsz * num_heads * seqlen,)
    nsrc = layer + 2  # consumer = layer + 1 gathers from segments 0..layer+1
    _xlayer_churn_tail_kernel[grid](
        idx, valid, p, pn, seed, new_idx, new_valid, stats,
        churn_eps, seqlen, layer, nsrc,
        KB=kb, DO_STATS=do_stats, BANKS=_STATS_BANKS,
        num_warps=1,
    )
    return new_idx, new_valid, stats.view(_STATS_BANKS, 5).sum(0)


@xlayer_churn_rewire.register_fake
def _(idx, valid, p, pn, seed, layer, churn_eps, do_stats):
    return (
        idx.new_empty(idx.shape),
        idx.new_empty(idx.shape, dtype=torch.int8),
        p.new_empty(5, dtype=torch.float32),
    )


# --- list-input attention wrapper -------------------------------------------
# Feeding the attention op a pre-cat [B, Hkv, L*T, D] stack makes every
# layer's autograd ctx retain its OWN cat copy (sum over 9 layers: 45
# segments x 32 MiB x k,v ~= 2.9 GiB held through fwd+bwd - measured OOM on
# the 31 GiB dev card next to a 4 GiB browser).  This op takes the per-layer
# K/V LISTS instead: the cat is a transient inside fwd and is rebuilt inside
# bwd, so the ctx saves only references to the 9 shared per-layer tensors
# (~0.6 GiB total, shared across all consuming layers).  Costs one extra
# k/v cat per layer in backward (~3 GiB/micro of bandwidth, ~2 ms).

def _cat_kv(ks: list[Tensor], vs: list[Tensor]) -> tuple[Tensor, Tensor]:
    if len(ks) == 1:
        return ks[0].contiguous(), vs[0].contiguous()
    return torch.cat(ks, dim=2), torch.cat(vs, dim=2)


@torch.library.custom_op("parameter_golf::xlayer_entmax_attn", mutates_args=())
def _xlayer_entmax_attn_fwd(
    q: Tensor, ks: list[Tensor], vs: list[Tensor], idx: Tensor, valid: Tensor,
    null_bias: Tensor, scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    from xlayer.sparse.sparse_entmax_kernel import _sparse_entmax_attn_fwd

    k_all, v_all = _cat_kv(ks, vs)
    return _sparse_entmax_attn_fwd(q, k_all, v_all, idx, valid, null_bias, scale)


@_xlayer_entmax_attn_fwd.register_fake
def _(q, ks, vs, idx, valid, null_bias, scale):
    bsz, num_heads, seqlen, _ = q.shape
    return (
        q.new_empty(q.shape),
        q.new_empty(bsz, num_heads, seqlen, idx.size(-1), dtype=torch.float32),
        q.new_empty(bsz, num_heads, seqlen, dtype=torch.float32),
    )


def _xlayer_setup_context(ctx, inputs, output):
    q, ks, vs, idx, valid, null_bias, scale = inputs
    _, p, pn = output
    ctx.n_seg = len(ks)
    ctx.scale = scale
    ctx.save_for_backward(q, *ks, *vs, idx, valid, p, pn)


def _xlayer_backward(ctx, dy, dp, dpn):
    from xlayer.sparse.sparse_entmax_kernel import _sparse_entmax_attn_bwd

    n = ctx.n_seg
    q, *rest = ctx.saved_tensors
    ks, vs = list(rest[:n]), list(rest[n:2 * n])
    idx, valid, p, pn = rest[2 * n:]
    k_all, v_all = _cat_kv(ks, vs)
    seg_t = ks[0].size(2)
    dq, dk_all, dv_all, dnull = _sparse_entmax_attn_bwd(
        dy, q, k_all, v_all, idx, valid, p, pn, ctx.scale
    )
    dks = list(dk_all.split(seg_t, dim=2))
    dvs = list(dv_all.split(seg_t, dim=2))
    return dq, dks, dvs, None, None, dnull, None


_xlayer_entmax_attn_fwd.register_autograd(_xlayer_backward, setup_context=_xlayer_setup_context)


def xlayer_entmax_attention_stats(
    q: Tensor, ks: list[Tensor], vs: list[Tensor], idx: Tensor, valid: Tensor,
    null_bias: Tensor, scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """``sparse_entmax_attention_stats`` over the cross-layer K/V stack.

    ``ks``/``vs`` are the per-layer [B, Hkv, T, D] tensors (equal T); segment
    s of the virtual stack holds layer s, so wiring lins are layer*T + pos.
    Gradients flow to q, every ks/vs element, and null_bias."""
    return _xlayer_entmax_attn_fwd(
        q.contiguous(), [k.contiguous() for k in ks], [v.contiguous() for v in vs],
        idx, valid, null_bias, scale,
    )

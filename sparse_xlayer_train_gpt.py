"""Cross-layer churn arm: ALL previous layers' K/V are wiring candidates.

Extends the churn-wired sparse entmax arm (``sparse_churn_train_gpt``): the
candidate pool for a query at layer L is every (layer', pos') pair with
layer' <= L and pos' <= t, not just the current layer's positions.  Wiring
indices are flat ``lin = layer' * T + pos'`` into a K/V stack built by
concatenating each layer's rotary-encoded K/V along the row axis
(``[B, Hkv, (L+1)*T, D]``, layer-major), so the fused entmax attention kernel
is UNCHANGED - it gathers rows by index and never cared what they mean.

What changes vs the same-layer churn arm:

* Each layer appends its (K, V) to a per-forward stack; attention gathers
  from the concatenation of layers 0..L.  Gradients flow through cross-layer
  edges into earlier layers' K/V natively (autograd through torch.cat).
* Rewire draws sample (layer', pos') uniformly over the consumer's pool.
  Cross-layer access comes purely from churn exploration - no cross-layer
  structural prior in v1; with L layers live, (L-1)/L of draws leave the
  current layer, so exploration pressure toward earlier layers is automatic.
* The self edge (slot 0) is rebased every churn to the consumer layer's own
  K at the query position - otherwise it would dangle on layer 0 forever.
* Inherited edges persist as (layer', pos') pairs: an edge that earned its
  keep pointing at layer 2's representation keeps pointing at layer 2, it is
  not re-aimed at the current layer.  That persistence IS the feature.

Layer-0 wiring, the entropy/null churn gate, eligibility, dedup, entmax
routing, stats, and eval determinism are identical to the churn arm (layer-0
lins coincide with positions since segment 0 is layer 0).  Costs vs the churn
arm at production shape: the K/V cat copies (~3 GB/micro of bandwidth), wider
gather working set (worse L2 locality), and fp32 dk/dv atomic buffers sized
to the stack in each layer's backward.

Env knobs: SPARSE_K (128), SPARSE_NULL_INIT (0.0), CHURN_EPS (0.05),
CHURN_LOCAL (64), CHURN_RANDOM (32).  Per-layer rewire/support/H_norm reach
tensorboard ``churn/*`` tags via the ``churn_stats`` val-cadence line.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import train_gpt as baseline
from sparse_xlayer_kernel import xlayer_churn_rewire, xlayer_entmax_attention_stats

MAX_LAYERS = 32
_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

# Wiring + per-forward K/V stack + stats state, reset by GPT.forward.
_STATE: dict = {"i": 0, "wiring": None, "kv": None}
_STATS: Tensor | None = None  # [MAX_LAYERS, 3] on device: rewire%, support%, H_norm
_LAYERS_SEEN: int = 0

if torch.cuda.is_available():
    _SEED_BUF: Tensor = torch.zeros(1, dtype=torch.int64, device=f"cuda:{_LOCAL_RANK}")
    _EVAL_SEED: Tensor = torch.full(
        (1,), 0x0C0FFEE, dtype=torch.int64, device=f"cuda:{_LOCAL_RANK}"
    )


_M31 = (1 << 31) - 1     # 31-bit hash state - see sparse_churn_train_gpt for
_HASH_MUL = 0x45D9F3B    # why every literal must stay below 2^31 (Inductor
                         # folds the key into Triton int32 index arithmetic)


def _mix31(x: Tensor, seed: int) -> Tensor:
    """lowbias32-style xorshift-multiply hash on 31-bit state (eval RNG)."""
    x = (x + (seed & _M31)) & _M31
    x = (((x >> 15) ^ x) * _HASH_MUL) & _M31
    x = (((x >> 15) ^ x) * _HASH_MUL) & _M31
    return (x >> 15) ^ x


def _hash_uniform(
    bsz: int, num_heads: int, seqlen: int, k: int,
    layer: int, seed: int, device: torch.device,
) -> Tensor:
    """Deterministic [B, H, T, K] floats in [0, 1) from lane coordinates."""
    b = torch.arange(bsz, device=device, dtype=torch.int64).view(-1, 1, 1, 1)
    h = torch.arange(num_heads, device=device, dtype=torch.int64).view(1, -1, 1, 1)
    t = torch.arange(seqlen, device=device, dtype=torch.int64).view(1, 1, -1, 1)
    s = torch.arange(k, device=device, dtype=torch.int64).view(1, 1, 1, -1)
    key = ((b * num_heads + h) * seqlen + t) * k + s    # bijective lane id
    bits = _mix31(key, seed * 0x9E3779B9 + layer * 0x1000193) & 0xFFFFFF
    return bits.to(torch.float32) / float(1 << 24)


class XLayerChurnAttention(baseline.CausalSelfAttention):
    k_budget = int(os.environ.get("SPARSE_K", "128"))
    null_init = float(os.environ.get("SPARSE_NULL_INIT", "0.0"))
    churn_eps = float(os.environ.get("CHURN_EPS", "0.05"))
    n_local = int(os.environ.get("CHURN_LOCAL", "64"))
    n_random = int(os.environ.get("CHURN_RANDOM", "32"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.k_budget & (self.k_budget - 1):
            raise ValueError("SPARSE_K must be a power of two (kernel KB)")
        if self.n_local + self.n_random + 1 >= self.k_budget:
            raise ValueError("CHURN_LOCAL + CHURN_RANDOM must leave strided slots")
        self.null_bias = nn.Parameter(
            torch.full((self.num_heads,), self.null_init, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        global _LAYERS_SEEN
        bsz, seqlen, dim = x.shape
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, v = self.c_qkv(x).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, x.dtype)
        q = baseline.apply_rotary_emb(q, cos, sin)
        k = baseline.apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        layer = _STATE["i"]
        _STATE["i"] = layer + 1
        _LAYERS_SEEN = max(_LAYERS_SEEN, layer + 1)

        # per-forward K/V stack: this layer's rows live at lins
        # [layer*T, (layer+1)*T).  The LIST feeds the attention op, which
        # cats transiently in fwd and bwd - saving pre-cat stacks per layer
        # would retain ~2.9GiB across the 9 ctxs (measured OOM on dev).
        # store v contiguous: the transpose view would otherwise keep the
        # whole 128MiB qkv buffer alive across micro-batches (_STATE outlives
        # the graph; Dynamo replays the reset only at the NEXT graph's exit)
        kv = _STATE["kv"]
        if layer == 0 or kv is None:
            kv = []
        kv.append((k, v.contiguous()))
        _STATE["kv"] = kv
        ks = [kk for kk, _ in kv]
        vs = [vv for _, vv in kv]

        with torch.no_grad():
            wiring = _STATE["wiring"]
            if layer == 0 or wiring is None or wiring[0].shape[:3] != (bsz, self.num_heads, seqlen):
                idx, valid = self._structural_wiring(bsz, seqlen, x.device, layer)
            else:
                idx, valid = wiring

        y, p, pn = xlayer_entmax_attention_stats(
            q, ks, vs, idx, valid, self.null_bias, self.head_dim**-0.5
        )

        with torch.no_grad():
            do_stats = self.training and _STATS is not None
            # the last layer's churned wiring is never consumed; skip the
            # kernel outside the stats path (training diagnostics)
            if do_stats or layer != _STATE.get("last", -1):
                new_idx, new_valid, stats = xlayer_churn_rewire(
                    idx, valid, p.detach(), pn.detach(),
                    _SEED_BUF if self.training else _EVAL_SEED,
                    layer, self.churn_eps, do_stats,
                )
                _STATE["wiring"] = (new_idx, new_valid.to(torch.bool))
                if do_stats:
                    li = min(layer, MAX_LAYERS - 1)
                    rows = stats[4].clamp_min(1.0)
                    _STATS[li, 0] = stats[0] / (rows * self.k_budget)
                    _STATS[li, 1] = stats[1] / stats[2].clamp_min(1.0)
                    _STATS[li, 2] = stats[3] / rows

        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)

    def _structural_wiring(
        self, bsz: int, seqlen: int, device: torch.device, layer: int
    ) -> tuple[Tensor, Tensor]:
        """Content-free layer-0 wiring: self + local + strided + random.

        Identical to the churn arm: layer 0's pool is segment 0 only, where
        lin == position, so this needs no cross-layer awareness."""
        kb = self.k_budget
        n_loc, n_rnd = self.n_local, self.n_random
        n_str = kb - 1 - n_loc - n_rnd
        t_pos = torch.arange(seqlen, device=device, dtype=torch.int32)

        d_loc = torch.arange(1, n_loc + 1, device=device, dtype=torch.int32)
        stride = max(1, (seqlen - n_loc) // (n_str + 1))
        d_str = n_loc + stride * torch.arange(1, n_str + 1, device=device, dtype=torch.int32)
        deltas = torch.cat([d_loc, d_str])                      # [n_loc + n_str]
        j = t_pos[:, None] - deltas[None, :]                    # [T, n_loc+n_str]
        struct_valid = j >= 0
        struct_idx = j.clamp_min(0)

        if self.training:
            u = torch.rand(bsz, self.num_heads, seqlen, n_rnd, device=device)
        else:
            u = _hash_uniform(bsz, self.num_heads, seqlen, n_rnd, layer, 11, device)
        rnd_idx = (u * (t_pos + 1).to(torch.float32).view(1, 1, -1, 1)).to(torch.int32)

        idx = torch.cat(
            [
                t_pos.view(1, 1, -1, 1).expand(bsz, self.num_heads, -1, 1),
                struct_idx.view(1, 1, seqlen, -1).expand(bsz, self.num_heads, -1, -1),
                rnd_idx,
            ],
            dim=-1,
        ).contiguous()
        valid = torch.cat(
            [
                torch.ones(bsz, self.num_heads, seqlen, 1, dtype=torch.bool, device=device),
                struct_valid.view(1, 1, seqlen, -1).expand(bsz, self.num_heads, -1, -1),
                torch.ones(bsz, self.num_heads, seqlen, n_rnd, dtype=torch.bool, device=device),
            ],
            dim=-1,
        )
        valid = _dedup_earliest(idx, valid, torch.ones_like(valid))
        return idx, valid

    def _churn(
        self, idx: Tensor, valid: Tensor, p: Tensor, pn: Tensor, layer: int
    ) -> tuple[Tensor, Tensor]:
        """Reference churn (torch ops) - the spec the tests check the fused
        ``xlayer_churn_rewire`` kernel against.  Draws roam (layer', pos');
        slot 0 rebases to the consumer layer's self lin before dedup."""
        global _STATS
        bsz, num_heads, seqlen, kb = idx.shape
        device = idx.device
        t_pos = torch.arange(seqlen, device=device, dtype=torch.float32)

        m_incl = valid.sum(-1, dtype=torch.float32) + 1.0       # slots + null
        h_row = torch.special.entr(p).sum(-1) + torch.special.entr(pn)
        h_norm = h_row / m_incl.log().clamp_min(math.log(2.0))
        q_rw = torch.maximum(h_norm, pn).clamp(self.churn_eps, 1.0)[..., None]

        share = (1.0 / m_incl)[..., None]
        slot_ok = torch.arange(kb, device=device) != 0          # self never rewires
        eligible = (p <= share) & slot_ok                       # invalid slots: p == 0

        if self.training:
            u_gate = torch.rand(bsz, num_heads, seqlen, kb, device=device)
            u_pos = torch.rand(bsz, num_heads, seqlen, kb, device=device)
            u_lay = torch.rand(bsz, num_heads, seqlen, kb, device=device)
        else:
            u_gate = _hash_uniform(bsz, num_heads, seqlen, kb, layer, 23, device)
            u_pos = _hash_uniform(bsz, num_heads, seqlen, kb, layer, 37, device)
            u_lay = _hash_uniform(bsz, num_heads, seqlen, kb, layer, 41, device)
        rewire = eligible & (u_gate < q_rw)
        nsrc = layer + 2  # consumer = layer + 1 sees segments 0..layer+1
        d_pos = (u_pos * (t_pos + 1).view(1, 1, -1, 1)).to(torch.int32)
        d_lay = (u_lay * float(nsrc)).to(torch.int32)
        draws = d_lay * seqlen + d_pos

        new_idx = torch.where(rewire, draws, idx)
        self_lin = (nsrc - 1) * seqlen + t_pos.to(torch.int32)  # [T]
        new_idx = torch.cat(
            [self_lin.view(1, 1, -1, 1).expand(bsz, num_heads, -1, 1), new_idx[..., 1:]],
            dim=-1,
        )
        new_valid = valid | rewire
        new_valid = _dedup_earliest(new_idx, new_valid, rewire)

        if self.training and _STATS is not None:
            li = min(layer, MAX_LAYERS - 1)
            _STATS[li, 0] = rewire.float().mean()
            _STATS[li, 1] = ((p > 0) & valid).sum() / valid.sum().clamp_min(1)
            _STATS[li, 2] = h_norm.mean()
        return new_idx, new_valid


def _dedup_earliest(idx: Tensor, valid: Tensor, candidates: Tensor) -> Tensor:
    """Invalidate ``candidates`` slots that duplicate a winning valid slot.

    Identical semantics to the churn arm (see sparse_churn_train_gpt).  EAGER
    at production shape materializes ~24GB - keep uncompiled tests small."""
    kb = idx.size(-1)
    slot = torch.arange(kb, device=idx.device)
    earlier = slot[None, :] < slot[:, None]                     # [j < i]
    wins = valid[..., None, :] & (~candidates[..., None, :] | earlier)
    dup = ((idx[..., :, None] == idx[..., None, :]) & wins).any(-1)
    return valid & ~(candidates & dup)


def _wrap_gpt_forward(orig_forward):
    def forward(self, *args, **kwargs):
        _STATE["i"] = 0
        _STATE["wiring"] = None
        _STATE["kv"] = None
        _STATE["last"] = len(self.blocks) - 1
        if self.training:
            _SEED_BUF.add_(1)  # fresh Philox stream per train forward
        return orig_forward(self, *args, **kwargs)

    return forward


def _wrap_eval_val(orig_eval_val):
    is_master = int(os.environ.get("RANK", "0")) == 0

    def eval_val(*args, **kwargs):
        result = orig_eval_val(*args, **kwargs)
        if is_master and _STATS is not None and _LAYERS_SEEN:
            # ablation.py folds this into the following val entry ->
            # metrics.jsonl -> tensorboard churn/ tags
            rows = _STATS[:_LAYERS_SEEN].tolist()
            parts = [
                f"{name}_l{li}:{row[col]:.4f}"
                for li, row in enumerate(rows)
                for col, name in enumerate(("rewire", "support", "hnorm"))
            ]
            print("churn_stats " + " ".join(parts), flush=True)
        return result

    return eval_val


def main() -> None:
    global _STATS
    # pin the rank's device before baseline.main() runs set_device - see
    # sparse_churn_train_gpt for the torchrun hazard this avoids
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _STATS = torch.zeros(MAX_LAYERS, 3, device=f"cuda:{local_rank}")
    original_attention = baseline.CausalSelfAttention
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    original_int8_patterns = baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    original_gpt_forward = baseline.GPT.forward
    original_eval_val = baseline.eval_val
    baseline.CausalSelfAttention = XLayerChurnAttention
    baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns + ("null_bias",)
    baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns + ("null_bias",)
    baseline.GPT.forward = _wrap_gpt_forward(original_gpt_forward)
    baseline.eval_val = _wrap_eval_val(original_eval_val)
    try:
        baseline.main()
    finally:
        baseline.CausalSelfAttention = original_attention
        baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns
        baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns
        baseline.GPT.forward = original_gpt_forward
        baseline.eval_val = original_eval_val


if __name__ == "__main__":
    main()

"""Baseline + sparse candidate attention: entmax-1.5 routing over top-k picks
with entropy-driven random exploration and a learnable null opt-out.

Design (agreed in-session, modeled on the routing machinery in
cleanrl ppo_continuous_action_morphcompute_v25):

* Each query gets a hard budget of ``SPARSE_K`` candidate edges: itself, plus
  top-(k-1) past positions ranked by a cheap coarse scorer, with the last
  ``SPARSE_R_MAX`` rank slots replaced by uniform-random past positions with
  probability equal to the query's normalized coarse-score entropy (a token
  that cannot tell its candidates apart explores more).  Training only; eval
  is deterministic top-k.
* The coarse scorer is a frozen random semi-orthogonal sketch of the exact
  fine q/k (post norm, RoPE, gain) into ``SPARSE_COARSE_DIM`` dims, so coarse
  ranking approximates fine ranking (Johnson-Lindenstrauss) with no extra
  learnable parts that could collapse.  ``SPARSE_COARSE_DIM=0`` selects with
  the exact fine scores (oracle-recall diagnostic).
* Fine attention over the candidates uses entmax-1.5 with an extra learnable
  per-head null logit.  Entmax gives exact zeros, so a bad exploration edge
  costs compute but injects no output noise and no gradient noise; mass on
  the null column attenuates the output, letting effective connectivity fall
  below k ("can't go higher" is the budget, "wants fewer" is the null).

This v1 computes the semantics at dense cost (full QK^T is materialized and
then masked to the candidate set) - it measures BPB quality at 1024 context;
the fused gather kernel is the long-context payoff and only worth building if
this wins.  The T x T intermediates would be ~15 GB of saved activations at
the baseline micro-batch (64 seqs on one GPU; grad accumulation is hardcoded
to 8 // world_size in the baseline, not tunable), so training wraps the
attention body in activation checkpointing: only the layer input is saved and
the intermediates are recomputed during backward.

Env knobs: SPARSE_K (128), SPARSE_R_MAX (16), SPARSE_COARSE_DIM (16),
SPARSE_NULL_INIT (0.0).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import math
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

import train_gpt as baseline

NEG_INF = float("-inf")
# Finite stand-in for masked candidate logits: far below any real score, but
# keeps the entmax prefix statistics finite (no inf*inf -> nan in cumsums).
MASKED_LOGIT = -1e4
COARSE_SKETCH_SEED = 271828


def entmax15(logits: Tensor, dim: int = -1) -> Tensor:
    """Exact entmax-1.5 (Peters & Martins 2019) over ``dim``.

    The support set and threshold index are found with the sort-based exact
    algorithm under no_grad; the threshold is then re-derived differentiably
    from the support statistics, so autograd yields the true Jacobian wherever
    the support is locally constant (almost everywhere).  This keeps the
    backward memory footprint at O(width) per row and stays fullgraph-safe
    (no custom autograd.Function).
    """
    dtype = torch.promote_types(logits.dtype, torch.float32)
    x = (logits - logits.amax(dim=dim, keepdim=True)).to(dtype) / 2
    with torch.no_grad():
        x_srt, _ = torch.sort(x, dim=dim, descending=True)
        rho = torch.arange(
            1, x.size(dim) + 1, device=x.device, dtype=dtype
        ).view([-1 if d == dim % x.dim() else 1 for d in range(x.dim())])
        mean = x_srt.cumsum(dim) / rho
        mean_sq = (x_srt * x_srt).cumsum(dim) / rho
        delta = (1 - rho * (mean_sq - mean * mean)) / rho
        tau = mean - torch.sqrt(delta.clamp_min(0))
        support_size = (tau <= x_srt).sum(dim=dim, keepdim=True).clamp_min(1)
        tau_star = tau.gather(dim, support_size - 1)
        support = x > tau_star
    size = support.sum(dim=dim, keepdim=True).clamp_min(1).to(dtype)
    x_sup = torch.where(support, x, torch.zeros_like(x))
    m = x_sup.sum(dim=dim, keepdim=True) / size
    m_sq = (x_sup * x_sup).sum(dim=dim, keepdim=True) / size
    root = (m * m - m_sq + 1.0 / size).clamp_min(0).sqrt()
    p = torch.clamp(x - (m - root), min=0)
    return p * p


class SparseEntmaxCausalSelfAttention(baseline.CausalSelfAttention):
    k_budget = int(os.environ.get("SPARSE_K", "128"))
    r_max = int(os.environ.get("SPARSE_R_MAX", "16"))
    coarse_dim = int(os.environ.get("SPARSE_COARSE_DIM", "16"))
    null_init = float(os.environ.get("SPARSE_NULL_INIT", "0.0"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < self.r_max < self.k_budget:
            raise ValueError("need 0 < SPARSE_R_MAX < SPARSE_K")
        if not 0 <= self.coarse_dim <= self.head_dim:
            raise ValueError("need 0 <= SPARSE_COARSE_DIM <= head_dim")
        self.null_bias = nn.Parameter(
            torch.full((self.num_heads,), self.null_init, dtype=torch.float32)
        )
        if self.coarse_dim:
            # Deterministic seed: the sketch is identical across warmup and the
            # real run (baseline restores parameter state, not buffers), and it
            # is reproducible without being exported (non-persistent).
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(COARSE_SKETCH_SEED)
                sketch = nn.init.orthogonal_(
                    torch.empty(self.head_dim, self.coarse_dim, dtype=torch.float32)
                )
            self.register_buffer("coarse_sketch", sketch, persistent=False)

    def _select_candidates(
        self, q: Tensor, k_exp: Tensor, scores: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Pick each query's candidate keys; no gradients flow through this.

        Returns ``idx`` [B, H, T, k] (self edge in slot 0) and a matching
        boolean validity mask (False marks the padding slots early tokens get
        from ranking a mostly -inf row).
        """
        bsz, num_heads, seqlen, _ = scores.shape
        device = scores.device
        if self.coarse_dim:
            sketch = self.coarse_sketch.to(q.dtype)
            # Rescale so sketched scores estimate the fine scores: a random
            # semi-orthogonal sketch shrinks dot products by coarse_dim/head_dim
            # in expectation, and fine scores carry the 1/sqrt(head_dim) scale.
            coarse_scale = (self.head_dim / self.coarse_dim) * self.head_dim**-0.5
            coarse = (q @ sketch) @ (k_exp @ sketch).transpose(-2, -1) * coarse_scale
        else:
            coarse = scores
        # Selection runs in the compute dtype: full-width fp32 copies of the
        # T x T matrices cost gigabytes at the real micro-batch, and ranking /
        # exploration pressure tolerate bf16 easily.
        causal = torch.ones(seqlen, seqlen, dtype=torch.bool, device=device).tril()
        coarse = coarse.masked_fill(~causal, NEG_INF)
        eye = torch.eye(seqlen, dtype=torch.bool, device=device)
        ranked = coarse.masked_fill(eye, NEG_INF)  # self is force-included below

        n_rank = min(self.k_budget, seqlen) - 1
        top_val, top_idx = ranked.topk(n_rank, dim=-1)
        if self.training:
            r = min(self.r_max, n_rank)
            # Exploration pressure: normalized entropy of the coarse routing
            # distribution.  A flat row (can't tell candidates apart) explores;
            # a confident row keeps its ranked picks.  bf16 softmax with fp32
            # accumulation: an exploration probability needs no more precision.
            probs = torch.softmax(coarse, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-20).log()).sum(
                dim=-1, dtype=torch.float32
            )
            n_keys = torch.arange(1, seqlen + 1, device=device, dtype=torch.float32)
            explore_p = (entropy / n_keys.log().clamp_min(math.log(2.0))).clamp(0, 1)
            positions = torch.arange(seqlen, device=device)
            rand_idx = (
                torch.rand(bsz, num_heads, seqlen, r, device=device)
                * (positions + 1).to(torch.float32).view(1, 1, seqlen, 1)
            ).long()
            replace = (
                torch.rand(bsz, num_heads, seqlen, r, device=device)
                < explore_p[..., None]
            )
            top_idx = torch.cat(
                [top_idx[..., : n_rank - r], torch.where(replace, rand_idx, top_idx[..., n_rank - r :])],
                dim=-1,
            )
            # Random picks are always causally valid; keep ranked slots' scores.
            top_val = torch.cat(
                [top_val[..., : n_rank - r], torch.where(replace, torch.zeros_like(top_val[..., n_rank - r :]), top_val[..., n_rank - r :])],
                dim=-1,
            )
        self_idx = (
            torch.arange(seqlen, device=device)
            .view(1, 1, seqlen, 1)
            .expand(bsz, num_heads, seqlen, 1)
        )
        idx = torch.cat([self_idx, top_idx], dim=-1)
        valid = torch.cat(
            [torch.ones_like(self_idx, dtype=torch.bool), top_val > NEG_INF], dim=-1
        )
        if self.training:
            # A random exploration pick can collide with the self edge or a
            # ranked pick.  Duplicated keys are NOT weight-neutral under
            # entmax (the extra support entry shifts the threshold, inflating
            # that key), so invalidate all but one copy.  Which copy survives
            # is irrelevant among VALID copies (equal indices carry equal
            # logits), but invalid junk slots can share an index with a valid
            # slot (topk returns the -inf self position for early tokens), and
            # sort's tie-break is undefined - so invalid slots are remapped to
            # unique sentinels first, guaranteeing the survivor is valid on
            # any hardware.  Eval has no duplicates (topk indices are distinct
            # and exclude self).
            n_slots = idx.size(-1)
            sentinel = seqlen + torch.arange(n_slots, device=device)
            key = torch.where(valid, idx, sentinel.expand_as(idx))
            key_sorted, order = key.sort(dim=-1)
            dup_sorted = torch.cat(
                [
                    torch.zeros_like(key_sorted[..., :1], dtype=torch.bool),
                    key_sorted[..., 1:] == key_sorted[..., :-1],
                ],
                dim=-1,
            )
            dup = torch.zeros_like(dup_sorted).scatter(-1, order, dup_sorted)
            valid = valid & ~dup
        return idx, valid

    def forward(self, x: Tensor) -> Tensor:
        # The rotary cache mutates module state lazily; keep it outside the
        # checkpointed region so fullgraph compile can trace the mutation.
        cos, sin = self.rotary(x.size(1), x.device, x.dtype)
        if self.training and torch.is_grad_enabled():
            # The T x T intermediates (scores, routing weights, indices) would
            # dominate saved-activation memory at the baseline micro-batch;
            # recompute them in backward instead of saving them.  The default
            # preserve_rng_state=True is load-bearing: the recompute must
            # redraw the exact exploration randomness or gradients would be
            # taken w.r.t. a different candidate set than the forward used.
            return torch.utils.checkpoint.checkpoint(
                self._attend, x, cos, sin, use_reentrant=False
            )
        return self._attend(x, cos, sin)

    def _attend(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, v = self.c_qkv(x).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = baseline.apply_rotary_emb(q, cos, sin)
        k = baseline.apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        group = self.num_heads // self.num_kv_heads
        k_exp = k.repeat_interleave(group, dim=1)
        v_exp = v.repeat_interleave(group, dim=1)
        # v1 measures quality at dense cost: full scores, then candidate mask.
        scores = q @ k_exp.transpose(-2, -1) * self.head_dim**-0.5

        with torch.no_grad():
            idx, valid = self._select_candidates(q, k_exp, scores)
        # Gather first, then cast: a fp32 copy of the full score matrix costs
        # gigabytes at the real micro-batch; the gathered slice is tiny.
        cand = torch.gather(scores, -1, idx).to(torch.float32)
        cand = torch.where(valid, cand, torch.full_like(cand, MASKED_LOGIT))
        null = (
            self.null_bias.to(torch.float32)
            .view(1, self.num_heads, 1, 1)
            .expand(bsz, self.num_heads, seqlen, 1)
        )
        weights = entmax15(torch.cat([cand, null], dim=-1))
        # Null mass is simply dropped: rows sum to <=1, attenuating the output.
        # Candidate indices are deduplicated at selection time, so scatter_add
        # never merges copies (entmax weights are not duplicate-invariant).
        dense_weights = torch.zeros_like(scores)
        dense_weights = dense_weights.scatter_add(
            -1, idx, weights[..., :-1].to(scores.dtype)
        )
        y = dense_weights @ v_exp
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


def main() -> None:
    original_attention = baseline.CausalSelfAttention
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    original_int8_patterns = baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    baseline.CausalSelfAttention = SparseEntmaxCausalSelfAttention
    baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns + ("null_bias",)
    baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns + ("null_bias",)
    try:
        baseline.main()
    finally:
        baseline.CausalSelfAttention = original_attention
        baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns
        baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns


if __name__ == "__main__":
    main()

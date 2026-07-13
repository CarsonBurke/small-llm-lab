"""Baseline + sparse entmax candidate attention, fused-kernel edition.

Same semantics as ``sparse_entmax_attn_train_gpt.py`` (coarse top-k selection
with entropy-driven exploration, entmax-1.5 + learnable null over exactly k
candidate edges, deterministic eval) but the fine pass runs in the fused
Triton kernel from ``sparse_entmax_kernel``: per query, candidate K/V rows are
gathered straight into SRAM and the routing weights never leave registers.

What this removes relative to the reference implementation:

* the dense T x T fine score matrix (built only to be gathered from),
* the T x T scatter of routing weights and the dense AV matmul,
* activation checkpointing and its full recompute (the T x T intermediates it
  existed to avoid saving no longer exist).

Selection is the only remaining quadratic piece and runs at coarse width
(SPARSE_COARSE_DIM of head_dim) under no_grad.  ``SPARSE_COARSE_DIM=0``
(oracle selection with exact fine scores) still materializes dense scores for
ranking - it is a diagnostic mode.

Env knobs are shared with the reference: SPARSE_K (128), SPARSE_R_MAX (16),
SPARSE_COARSE_DIM (16), SPARSE_NULL_INIT (0.0).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import Tensor

import train_gpt as baseline
import sparse_select_kernel
from sparse_entmax_attn_train_gpt import NEG_INF, SparseEntmaxCausalSelfAttention
from sparse_entmax_kernel import sparse_entmax_attention


def _dedup_valid(idx: Tensor, valid: Tensor, s0: int) -> Tensor:
    """Invalidate duplicate candidate slots; the earliest occurrence stays.

    Only the exploration slots [s0:] can ever collide (self, the ranked
    strong set, and the kept weak set are pairwise distinct by construction),
    so instead of the reference's sentinel sort over all slots this compares
    just those r slots against every earlier slot.  Same semantics as the
    reference dedup - one valid slot per distinct index - except the survivor
    is deterministically the earliest slot (the reference's unstable sort
    keeps an arbitrary one; duplicates gather identical K/V rows, so the
    choice never reaches the output).
    """
    n_slots = idx.size(-1)
    slot = torch.arange(n_slots, device=idx.device)
    earlier = slot[None, :] < slot[s0:, None]  # [r, n_slots]
    dup = (
        (idx[..., s0:, None] == idx[..., None, :])
        & valid[..., None, :]
        & earlier
    ).any(-1)
    return torch.cat([valid[..., :s0], valid[..., s0:] & ~dup], dim=-1)


class FusedSparseEntmaxCausalSelfAttention(SparseEntmaxCausalSelfAttention):
    # Fused selection kernel (no T x T coarse materialization); 0 falls back
    # to the reference torch pipeline (also used by oracle mode and tiny T).
    fused_select = os.environ.get("SPARSE_FUSED_SELECT", "1") == "1"

    def forward(self, x: Tensor) -> Tensor:
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

        group = self.num_heads // self.num_kv_heads
        n_rank = min(self.k_budget, seqlen) - 1
        with torch.no_grad():
            if (
                self.fused_select
                and self.coarse_dim
                and n_rank >= 1
                and seqlen <= sparse_select_kernel.MAX_T
                and q.dtype == torch.bfloat16  # kernel ranks at bf16 precision
            ):
                idx, valid = self._select_candidates_fused(q, k)
            else:
                k_exp = k.repeat_interleave(group, dim=1)
                # Oracle selection is the one mode that needs dense scores.
                scores = (
                    None
                    if self.coarse_dim
                    else q @ k_exp.transpose(-2, -1) * self.head_dim**-0.5
                )
                idx, valid = self._select_candidates(q, k_exp, scores)
        n_slots = idx.size(-1)
        kb = max(16, 1 << (n_slots - 1).bit_length())
        if kb != n_slots:
            pad = (0, kb - n_slots)
            idx = F.pad(idx, pad)
            valid = F.pad(valid, pad)

        y = sparse_entmax_attention(
            q, k, v, idx.to(torch.int32), valid,
            self.null_bias, self.head_dim**-0.5,
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)

    def _select_candidates_fused(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Fused-kernel ranking + the reference exploration/dedup tail.

        Ranking, causal/self masking, and the exploration entropy come from
        one kernel pass over the sketched q/k (see sparse_select_kernel);
        sketching the unexpanded k and mapping heads to KV groups inside the
        kernel is the same math as the reference's k_exp @ sketch.  The tail
        below (exploration draws, forced self edge, validity, sentinel dedup)
        is line-for-line the reference semantics and consumes the RNG stream
        in the same order.  One defined difference: the kernel returns the
        weakest-r window as a set in position order, not score order, so
        with a matched seed the i.i.d. replacement draws attach to permuted
        slots - distribution-identical, not sample-path-identical.
        """
        bsz, num_heads, seqlen, _ = q.shape
        device = q.device
        sketch = self.coarse_sketch.to(q.dtype)
        coarse_scale = (self.head_dim / self.coarse_dim) * self.head_dim**-0.5
        n_rank = min(self.k_budget, seqlen) - 1
        r = min(self.r_max, n_rank)
        # Kernel output is partitioned, not sorted: slots [0, n_rank - r)
        # hold the strongest set, the last r the weakest (the exploration
        # window below).  In eval r is inert - no slot is ever replaced.
        top_val, top_idx, entropy = sparse_select_kernel.select_topk_entropy(
            q @ sketch, k @ sketch, coarse_scale, n_rank, n_rank - r
        )
        if self.training:
            n_keys = torch.arange(1, seqlen + 1, device=device, dtype=torch.float32)
            explore_p = (entropy / n_keys.log().clamp_min(math.log(2.0))).clamp(0, 1)
            positions = torch.arange(seqlen, device=device)
            rand_idx = (
                torch.rand(bsz, num_heads, seqlen, r, device=device)
                * (positions + 1).to(torch.float32).view(1, 1, seqlen, 1)
            ).to(torch.int32)
            replace = (
                torch.rand(bsz, num_heads, seqlen, r, device=device)
                < explore_p[..., None]
            )
            top_idx = torch.cat(
                [top_idx[..., : n_rank - r], torch.where(replace, rand_idx, top_idx[..., n_rank - r :])],
                dim=-1,
            )
            top_val = torch.cat(
                [top_val[..., : n_rank - r], torch.where(replace, torch.zeros_like(top_val[..., n_rank - r :]), top_val[..., n_rank - r :])],
                dim=-1,
            )
        self_idx = (
            torch.arange(seqlen, device=device, dtype=torch.int32)
            .view(1, 1, seqlen, 1)
            .expand(bsz, num_heads, seqlen, 1)
        )
        idx = torch.cat([self_idx, top_idx], dim=-1)
        valid = torch.cat(
            [torch.ones_like(self_idx, dtype=torch.bool), top_val > NEG_INF], dim=-1
        )
        if self.training:
            valid = _dedup_valid(idx, valid, idx.size(-1) - r)
        return idx, valid

    def _select_candidates(
        self, q: Tensor, k_exp: Tensor, scores: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Reference selection semantics, re-typed for the kernel's diet.

        Identical ranking / exploration / dedup logic to the parent, with two
        cost-only changes: indices are int32 end to end (the dedup sort was
        the reference's second-slowest selection op at int64), and the entropy
        uses the fused ``torch.special.entr``.  ``scores`` is only consumed in
        oracle mode (coarse_dim == 0); the fused path passes None and never
        builds the dense T x T fine scores.
        """
        bsz, num_heads, seqlen, _ = q.shape
        device = q.device
        if self.coarse_dim:
            sketch = self.coarse_sketch.to(q.dtype)
            coarse_scale = (self.head_dim / self.coarse_dim) * self.head_dim**-0.5
            coarse = (q @ sketch) @ (k_exp @ sketch).transpose(-2, -1) * coarse_scale
        else:
            coarse = scores
        causal = torch.ones(seqlen, seqlen, dtype=torch.bool, device=device).tril()
        coarse = coarse.masked_fill(~causal, NEG_INF)
        eye = torch.eye(seqlen, dtype=torch.bool, device=device)
        ranked = coarse.masked_fill(eye, NEG_INF)

        n_rank = min(self.k_budget, seqlen) - 1
        top_val, top_idx64 = ranked.topk(n_rank, dim=-1)
        top_idx = top_idx64.to(torch.int32)
        if self.training:
            r = min(self.r_max, n_rank)
            probs = torch.softmax(coarse, dim=-1)
            entropy = torch.special.entr(probs).sum(dim=-1, dtype=torch.float32)
            n_keys = torch.arange(1, seqlen + 1, device=device, dtype=torch.float32)
            explore_p = (entropy / n_keys.log().clamp_min(math.log(2.0))).clamp(0, 1)
            positions = torch.arange(seqlen, device=device)
            rand_idx = (
                torch.rand(bsz, num_heads, seqlen, r, device=device)
                * (positions + 1).to(torch.float32).view(1, 1, seqlen, 1)
            ).to(torch.int32)
            replace = (
                torch.rand(bsz, num_heads, seqlen, r, device=device)
                < explore_p[..., None]
            )
            top_idx = torch.cat(
                [top_idx[..., : n_rank - r], torch.where(replace, rand_idx, top_idx[..., n_rank - r :])],
                dim=-1,
            )
            top_val = torch.cat(
                [top_val[..., : n_rank - r], torch.where(replace, torch.zeros_like(top_val[..., n_rank - r :]), top_val[..., n_rank - r :])],
                dim=-1,
            )
        self_idx = (
            torch.arange(seqlen, device=device, dtype=torch.int32)
            .view(1, 1, seqlen, 1)
            .expand(bsz, num_heads, seqlen, 1)
        )
        idx = torch.cat([self_idx, top_idx], dim=-1)
        valid = torch.cat(
            [torch.ones_like(self_idx, dtype=torch.bool), top_val > NEG_INF], dim=-1
        )
        if self.training:
            # Same sentinel-remap dedup as the reference (see its comment for
            # why invalid slots must not share sort keys with valid ones).
            n_slots = idx.size(-1)
            sentinel = seqlen + torch.arange(n_slots, device=device, dtype=torch.int32)
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


def main() -> None:
    original_attention = baseline.CausalSelfAttention
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    original_int8_patterns = baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    baseline.CausalSelfAttention = FusedSparseEntmaxCausalSelfAttention
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

"""Sparse entmax attention with support-preserving, capped routing turnover.

The scan arm rebuilds every candidate set independently at every layer.  The
churn arm carries wiring forward, but a large null probability rewires nearly
every zero-mass slot and creates a positive feedback loop: poor random pool ->
null collapse -> wholesale random pool replacement.  This arm keeps attention
strictly sparse while separating *which edges are active* from exploration:

* layer 0 starts from the deterministic coarse top-k candidate set;
* later layers preserve the previous layer's positive-entmax support;
* at most ``STICKY_REPLACE`` slots (default 8 of 128) are replaced per layer;
* replacements must be novel members of the current layer's coarse top-2r.

Thus useful edges have hysteresis, but a layer can still introduce candidates
that become newly salient.  The fine attention remains fused entmax-1.5 over
exactly the same hard candidate budget.  This is a routing-quality ablation;
the coarse scan can be optimized only if the BPB gate passes.

Env knobs: SPARSE_K (128), SPARSE_COARSE_DIM (16), SPARSE_NULL_INIT (0.0),
STICKY_REPLACE (8), STICKY_PROPOSAL_MULT (2).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import Tensor

import sparse_select_kernel
import train_gpt as baseline
from sparse_entmax_attn_train_gpt import NEG_INF
from sparse_entmax_fast_train_gpt import FusedSparseEntmaxCausalSelfAttention
from sparse_entmax_kernel import sparse_entmax_attention_stats


MAX_LAYERS = 32
_STATE: dict = {"i": 0, "routing": None}
_STATS: Tensor | None = None  # [layer, rewire, support, hnorm, pnull]
_LAYERS_SEEN = 0


def _sticky_merge(
    idx: Tensor,
    valid: Tensor,
    previous_p: Tensor,
    proposal_idx: Tensor,
    proposal_valid: Tensor,
    replace_budget: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replace weak incumbents with up to ``replace_budget`` novel proposals.

    Slot 0 (the forced self edge) and every positive-mass incumbent are
    protected.  Invalid/zero-mass slots are preferred for replacement.  The
    proposal set is unique by construction; filtering it against all valid
    incumbents therefore preserves the no-duplicate invariant required by
    entmax.  Returns the new wiring plus a per-row replacement count.
    """
    if replace_budget == 0:
        rows = idx.shape[:-1]
        return idx, valid, torch.zeros(rows, device=idx.device, dtype=torch.float32)

    novel = proposal_valid & ~(
        (proposal_idx[..., :, None] == idx[..., None, :])
        & valid[..., None, :]
    ).any(-1)

    # Pick any r members of the current top-2r proposal set.  The fused
    # selector deliberately returns a rank set rather than a full ordering.
    chosen_valid, chosen_slot = novel.to(torch.float32).topk(
        replace_budget, dim=-1
    )
    chosen_valid = chosen_valid.to(torch.bool)
    chosen_idx = proposal_idx.gather(-1, chosen_slot)

    # Invalid incumbents sort below zero-mass valid ones.  Positive-support
    # slots are ineligible even in the unusual case where support occupies
    # nearly the entire budget.
    replaceable = ~valid | (previous_p <= 0)
    priority = torch.where(valid, previous_p, torch.full_like(previous_p, -1.0))
    protect_self = torch.arange(idx.size(-1), device=idx.device) == 0
    replaceable = replaceable & ~protect_self
    priority = priority.masked_fill(protect_self, float("inf"))
    _, replace_slot = priority.topk(replace_budget, dim=-1, largest=False)

    old_idx = idx.gather(-1, replace_slot)
    old_valid = valid.gather(-1, replace_slot)
    do_replace = chosen_valid & replaceable.gather(-1, replace_slot)
    write_idx = torch.where(do_replace, chosen_idx, old_idx)
    write_valid = old_valid | do_replace
    return (
        idx.scatter(-1, replace_slot, write_idx),
        valid.scatter(-1, replace_slot, write_valid),
        do_replace.sum(-1, dtype=torch.float32),
    )


class StickySparseEntmaxAttention(FusedSparseEntmaxCausalSelfAttention):
    replace_budget = int(os.environ.get("STICKY_REPLACE", "8"))
    proposal_mult = int(os.environ.get("STICKY_PROPOSAL_MULT", "2"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < self.replace_budget < self.k_budget:
            raise ValueError("need 0 < STICKY_REPLACE < SPARSE_K")
        if self.proposal_mult < 1:
            raise ValueError("STICKY_PROPOSAL_MULT must be positive")

    def _ranked_nonself(
        self, q: Tensor, k: Tensor, n_rank: int
    ) -> tuple[Tensor, Tensor]:
        """Return the deterministic coarse top-``n_rank`` non-self edges."""
        bsz, num_heads, seqlen, _ = q.shape
        if n_rank == 0:
            shape = (bsz, num_heads, seqlen, 0)
            return (
                torch.empty(shape, device=q.device, dtype=torch.int32),
                torch.empty(shape, device=q.device, dtype=torch.bool),
            )

        if (
            self.fused_select
            and self.coarse_dim
            and seqlen <= sparse_select_kernel.MAX_T
            and q.dtype == torch.bfloat16
        ):
            sketch = self.coarse_sketch.to(q.dtype)
            coarse_scale = (
                self.head_dim / self.coarse_dim
            ) * self.head_dim**-0.5
            top_val, top_idx, _ = sparse_select_kernel.select_topk_entropy(
                q @ sketch, k @ sketch, coarse_scale, n_rank, n_rank
            )
            return top_idx, top_val > NEG_INF

        group = self.num_heads // self.num_kv_heads
        k_exp = k.repeat_interleave(group, dim=1)
        if self.coarse_dim:
            sketch = self.coarse_sketch.to(q.dtype)
            coarse_scale = (
                self.head_dim / self.coarse_dim
            ) * self.head_dim**-0.5
            coarse = (
                (q @ sketch) @ (k_exp @ sketch).transpose(-2, -1)
            ) * coarse_scale
        else:
            coarse = q @ k_exp.transpose(-2, -1) * self.head_dim**-0.5
        causal = torch.ones(
            seqlen, seqlen, dtype=torch.bool, device=q.device
        ).tril()
        eye = torch.eye(seqlen, dtype=torch.bool, device=q.device)
        ranked = coarse.masked_fill(~causal | eye, NEG_INF)
        top_val, top_idx = ranked.topk(n_rank, dim=-1)
        return top_idx.to(torch.int32), top_val > NEG_INF

    def _initial_wiring(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        bsz, num_heads, seqlen, _ = q.shape
        n_rank = min(self.k_budget, seqlen) - 1
        top_idx, top_valid = self._ranked_nonself(q, k, n_rank)
        self_idx = (
            torch.arange(seqlen, device=q.device, dtype=torch.int32)
            .view(1, 1, seqlen, 1)
            .expand(bsz, num_heads, seqlen, 1)
        )
        idx = torch.cat((self_idx, top_idx), dim=-1)
        valid = torch.cat((torch.ones_like(self_idx, dtype=torch.bool), top_valid), dim=-1)
        slots = idx.size(-1)
        padded = max(16, 1 << (slots - 1).bit_length())
        if padded != slots:
            idx = F.pad(idx, (0, padded - slots))
            valid = F.pad(valid, (0, padded - slots))
        return idx.contiguous(), valid.contiguous()

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
        routing = _STATE["routing"]
        with torch.no_grad():
            if (
                layer == 0
                or routing is None
                or routing[0].shape[:3] != (bsz, self.num_heads, seqlen)
            ):
                idx, valid = self._initial_wiring(q, k)
                replacements = torch.zeros(
                    idx.shape[:-1], device=x.device, dtype=torch.float32
                )
            else:
                idx, valid, previous_p = routing
                n_prop = min(
                    self.proposal_mult * self.replace_budget, seqlen - 1
                )
                proposal_idx, proposal_valid = self._ranked_nonself(q, k, n_prop)
                budget = min(self.replace_budget, n_prop)
                idx, valid, replacements = _sticky_merge(
                    idx, valid, previous_p, proposal_idx, proposal_valid, budget
                )

        y, p, pn = sparse_entmax_attention_stats(
            q, k, v, idx, valid, self.null_bias, self.head_dim**-0.5
        )
        with torch.no_grad():
            _STATE["routing"] = (idx, valid, p.detach())
            if self.training and _STATS is not None:
                li = min(layer, MAX_LAYERS - 1)
                valid_count = valid.sum(-1, dtype=torch.float32)
                m_incl = valid_count + 1.0
                h = torch.special.entr(p).sum(-1) + torch.special.entr(pn)
                hnorm = h / m_incl.log().clamp_min(math.log(2.0))
                _STATS[li, 0] = (replacements / idx.size(-1)).mean()
                _STATS[li, 1] = ((p > 0) & valid).sum() / valid.sum().clamp_min(1)
                _STATS[li, 2] = hnorm.mean()
                _STATS[li, 3] = pn.mean()

        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


def _wrap_gpt_forward(orig_forward):
    def forward(self, *args, **kwargs):
        _STATE["i"] = 0
        _STATE["routing"] = None
        return orig_forward(self, *args, **kwargs)

    return forward


def _wrap_eval_val(orig_eval_val):
    is_master = int(os.environ.get("RANK", "0")) == 0

    def eval_val(*args, **kwargs):
        result = orig_eval_val(*args, **kwargs)
        if is_master and _STATS is not None and _LAYERS_SEEN:
            rows = _STATS[:_LAYERS_SEEN].tolist()
            names = ("rewire", "support", "hnorm", "pnull")
            parts = [
                f"{name}_l{li}:{row[col]:.4f}"
                for li, row in enumerate(rows)
                for col, name in enumerate(names)
            ]
            print("churn_stats " + " ".join(parts), flush=True)
        return result

    return eval_val


def main() -> None:
    global _STATS
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _STATS = torch.zeros(MAX_LAYERS, 4, device=f"cuda:{local_rank}")
    original_attention = baseline.CausalSelfAttention
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    original_int8_patterns = baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    original_gpt_forward = baseline.GPT.forward
    original_eval_val = baseline.eval_val
    baseline.CausalSelfAttention = StickySparseEntmaxAttention
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

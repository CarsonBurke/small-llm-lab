"""Baseline + churn-wired sparse entmax attention: no selection, no scan.

Paradigm (vs the scan-selection scripts): each query owns a persistent SET of
candidate positions - wiring - that flows across layers within one forward
pass.  No coarse scorer, no ranking, no T x T anything exists:

* Layer 0 wiring is structural (content-free): self + a local window + strided
  offsets + uniform random draws.
* Every layer runs the fused entmax-1.5 kernel over its wiring only (the same
  ``sparse_entmax_kernel`` as the scan scripts - it never cared where idx came
  from) and hands the next layer a churned copy: slots the row engaged stay;
  slots at-or-below the uniform mass share may rewire to fresh random
  positions.
* Keeping is content-based at full precision (entmax mass IS the keep signal,
  and it is free - the kernel already saves p for backward).  Proposing is
  random, so it needs no fidelity.  Depth supplies the search iterations.

Churn rate is state-dependent and NEVER scheduled: per query row,

    q_rewire = clamp(max(H / log(m + 1), p_null), CHURN_EPS, 1)

where H is the Shannon entropy of the row's full attention distribution
(candidates + null) and m its valid-slot count.  High entropy = "nothing
stands out in this set"; high null mass = "I rejected everything I was
offered"; both mean look elsewhere.  As rows sharpen the churn dies down on
its own - the annealing is emergent.  CHURN_EPS keeps confident rows slowly
auditioning fresh tokens in dead slots forever.

Eligibility is p <= 1/(m+1) (at-or-below the uniform share), NOT p == 0: a
fully uncertain row can carry full support with no exact zeros, and that is
exactly the row that must be able to rewire everything.  Slot 0 (self) never
rewires; dedup-killed slots stay eligible so budget never leaks.

Attention + churn run as ONE fused Triton kernel (``sparse_churn_kernel``):
the routing weights are live in registers when the entmax solve finishes, so
the churn tail costs no extra global-memory pass.  Randomness is Philox keyed
by a device seed tensor (bumped once per train forward); eval passes a fixed
seed, so val forwards are reproducible and consume no RNG state.  Layer-0
structural wiring keeps the integer-hash path for its eval random slots.

Wiring state threads through a module-level holder that GPT.forward resets,
so nothing leaks across steps or between warmup and the real run.

Env knobs: SPARSE_K (128, must be a power of two), SPARSE_NULL_INIT (0.0),
CHURN_EPS (0.05), CHURN_LOCAL (64), CHURN_RANDOM (32); strided slots fill the
remainder.  Logged per layer at val cadence (train-step snapshot): rewire
fraction, support fraction, normalized row entropy.
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
from torch import Tensor, nn

import train_gpt as baseline
from xlayer.sparse.sparse_churn_kernel import churn_rewire
from xlayer.sparse.sparse_entmax_kernel import sparse_entmax_attention_stats

MAX_LAYERS = 32
_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

# Wiring + stats state, reset at the top of every GPT.forward.
_STATE: dict = {"i": 0, "wiring": None}
_STATS: Tensor | None = None  # [MAX_LAYERS, 3] on device: rewire%, support%, H_norm
_LAYERS_SEEN: int = 0

# Philox seed plumbing for the fused kernel: the trainer bumps _SEED_BUF once
# per train forward (device tensor - never a per-step host scalar, which would
# re-guard the compiled graph); eval uses the fixed seed for determinism.
if torch.cuda.is_available():
    _SEED_BUF: Tensor = torch.zeros(1, dtype=torch.int64, device=f"cuda:{_LOCAL_RANK}")
    _EVAL_SEED: Tensor = torch.full(
        (1,), 0x0C0FFEE, dtype=torch.int64, device=f"cuda:{_LOCAL_RANK}"
    )


_M31 = (1 << 31) - 1     # 31-bit hash state: EVERY scalar literal below stays
_HASH_MUL = 0x45D9F3B    # under 2^31.  Inductor folds the affine key into
                         # Triton int32 INDEX arithmetic, where an int64-sized
                         # constant is a hard compile error under fullgraph
                         # (measured).  Arithmetic mod 2^31 is also exact under
                         # int32 wraparound (2^31 | 2^32), so compiled and
                         # eager paths agree bit-for-bit.


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


class ChurnSparseEntmaxAttention(baseline.CausalSelfAttention):
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
        with torch.no_grad():
            wiring = _STATE["wiring"]
            if layer == 0 or wiring is None or wiring[0].shape[:3] != (bsz, self.num_heads, seqlen):
                idx, valid = self._structural_wiring(bsz, seqlen, x.device, layer)
            else:
                idx, valid = wiring

        y, p, pn = sparse_entmax_attention_stats(
            q, k, v, idx, valid, self.null_bias, self.head_dim**-0.5
        )

        with torch.no_grad():
            do_stats = self.training and _STATS is not None
            # the last layer's churned wiring is never consumed (the GPT
            # forward wrapper resets _STATE), so outside of the stats path
            # (training diagnostics for the final layer) skip the kernel;
            # layer/last are Python constants per call site under Dynamo
            if do_stats or layer != _STATE.get("last", -1):
                new_idx, new_valid, stats = churn_rewire(
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
        """Content-free layer-0 wiring: self + local + strided + random."""
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
        """Reference churn (torch ops).  The hot path runs the identical logic
        fused inside ``sparse_churn_kernel``; this stays as the spec the tests
        check invariants and rewire-rate statistics against."""
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
            u_draw = torch.rand(bsz, num_heads, seqlen, kb, device=device)
        else:
            u_gate = _hash_uniform(bsz, num_heads, seqlen, kb, layer, 23, device)
            u_draw = _hash_uniform(bsz, num_heads, seqlen, kb, layer, 37, device)
        rewire = eligible & (u_gate < q_rw)
        draws = (u_draw * (t_pos + 1).view(1, 1, -1, 1)).to(torch.int32)

        new_idx = torch.where(rewire, draws, idx)
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

    Same one-valid-slot-per-index semantics as the scan scripts' dedup.  Only
    ``candidates`` slots (churned/random - the only ones that can collide) are
    ever killed; non-candidate slots are pairwise distinct by construction and
    always win.  Candidates are scattered, not a suffix, so a candidate loses
    to ANY non-candidate valid slot at the same index (earlier or later) and
    to earlier candidates - deterministic, and never kills all copies.  The
    [KB, KB] broadcast never materializes under torch.compile (measured ~2MB
    peak); EAGER at production shape materializes ~3x the [B,H,T,KB,KB] bool
    (~24GB at B=64) - keep uncompiled tests at small B*T.
    """
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
            # scripts/ablation.py parses this into the following val entry ->
            # metrics.jsonl -> tensorboard churn/ tags; must print BEFORE
            # the trainer's val_loss line (i.e. inside eval_val)
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
    # baseline.main() hasn't run torch.cuda.set_device yet, so pin the rank's
    # device explicitly - a bare "cuda" would put every rank's stats (and a
    # spurious CUDA context, plus cross-device writes inside the compiled
    # train graph) on physical GPU 0 under torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _STATS = torch.zeros(MAX_LAYERS, 3, device=f"cuda:{local_rank}")
    original_attention = baseline.CausalSelfAttention
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    original_int8_patterns = baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    original_gpt_forward = baseline.GPT.forward
    original_eval_val = baseline.eval_val
    baseline.CausalSelfAttention = ChurnSparseEntmaxAttention
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

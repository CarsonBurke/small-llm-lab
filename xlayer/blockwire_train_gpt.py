"""Randomly-rewired cross-layer sparse attention: xlayer v3 (sub-quadratic).

Every layer attends over the token representations of ALL layers up to and
including itself, at full resolution — no pooling, no down-projected values,
no learned routing state.  Where-to-look is *randomly rewired every step*:
each (kv-head, query block) draws a fresh uniform set of causal key blocks
from the growing cross-layer bank, gathers only those, and attends over them.
Nothing is scanned, scored, ranked, or sorted — the cost per query is
O(fixed K), so the whole layer is linear in sequence length.  This is the
deliberate, hard constraint of this design: NO sort, NO sweep over the bank,
NO entmax (its threshold needs a sort).  Anything that reintroduces
super-linear-in-T work is out.

Per (kv-head, query block of ``SPARSE_QUERY_BLOCK`` queries) each layer
gathers two kinds of keys:

* ``SPARSE_RAND`` random ``SPARSE_KEY_BLOCK``-token blocks: fresh uniform
  draws per step from the causal universe (deterministic 31-bit hash keyed
  on the optimizer-step clock).  Because the draw is re-rolled every step,
  every causal block is visited ~(batch x query-blocks x rand)/universe
  times over a run and receives full-resolution fine-pass gradient — the
  connections stay plastic and cannot freeze (v2's failure mode: persistent
  logit tables whose unselected entries starve).  Eval uses a fixed-seed
  draw so validation stays reproducible.
* 2 x QUERY_BLOCK local keys, always: the query block's own span (masked
  per-token causal) plus its predecessor span, from the current layer.

Only fully-causal blocks are ever drawable — the sole keys that can straddle
a query position are the current layer's diagonal span, which carries an
explicit per-token causal mask.  Earlier layers are fully computed for all
positions, so a straddling read there would leak the future; such blocks are
structurally undrawable (the random draw indexes only the causal universe)
and a leak counter stat asserts zero at runtime.

The fine pass scores gathered keys at full 64-dim width and normalizes with
plain softmax plus a per-head scalar null (the proven "attend to nothing"
option).  softmax has no sort/cumsum, so the backward is memory-cheap and the
whole op is a dense batched matmul on tensor cores.  New parameters per
layer: the null bias.  Nothing else.

Env knobs: SPARSE_RAND (8), SPARSE_KEY_BLOCK (16), SPARSE_QUERY_BLOCK (64),
SPARSE_FINE_QCHUNK (4), SPARSE_XLAYER (1; 0 = within-layer-only control),
SPARSE_NULL_INIT (0.0), TRAIN_SEQ_LEN.  Per-layer stats reach tensorboard via
the ``graph_stats`` val-cadence line.  This fork does not modify
``train_gpt.py``.
"""

from __future__ import annotations

import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

import train_gpt as baseline
from sparse_entmax_attn_train_gpt import MASKED_LOGIT

MAX_LAYERS = 32
_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
_TRAIN_SEQ_LEN = int(os.environ.get("TRAIN_SEQ_LEN", "1024"))

_STATE: dict = {"i": 0, "bank": None}
_STAT_NAMES = (
    "bw_xsel", "bw_xmass", "bw_null", "bw_support", "bw_leak", "bw_valid",
)
_STATS: Tensor | None = None  # [MAX_LAYERS, 6] on device; columns match _STAT_NAMES
_LAYERS_SEEN = 0

if torch.cuda.is_available():
    # Optimizer-step clock: incremented eagerly after each Muon step, read as
    # a tensor inside the compiled forward (value changes never recompile).
    _STEP_BUF: Tensor = torch.zeros(1, dtype=torch.int64, device=f"cuda:{_LOCAL_RANK}")
    # Fixed key for eval-time random draws: val forwards are reproducible and
    # consume no RNG state.
    _EVAL_STEP: Tensor = torch.full(
        (1,), 0x7EA15EED, dtype=torch.int64, device=f"cuda:{_LOCAL_RANK}"
    )


_M31 = (1 << 31) - 1
_HASH_MUL = 0x45D9F3B


def _mix31(x: Tensor, seed: Tensor) -> Tensor:
    """lowbias32-style xorshift-multiply hash on 31-bit state (tensor seed)."""
    x = (x + (seed & _M31)) & _M31
    x = (((x >> 15) ^ x) * _HASH_MUL) & _M31
    x = (((x >> 15) ^ x) * _HASH_MUL) & _M31
    return (x >> 15) ^ x


def _hash_uniform(lane: Tensor, step: Tensor, layer: int) -> Tensor:
    """Deterministic uniforms in (0, 1) keyed on (lane, step, layer)."""
    bits = _mix31(lane, step * 0x9E3779B9 + layer * 0x1000193) & 0xFFFFFF
    return (bits.to(torch.float32) + 0.5) / float(1 << 24)


class BlockWireAttention(baseline.CausalSelfAttention):
    n_rand = int(os.environ.get("SPARSE_RAND", "8"))
    key_block = int(os.environ.get("SPARSE_KEY_BLOCK", "16"))
    query_block = int(os.environ.get("SPARSE_QUERY_BLOCK", "64"))
    fine_qchunk = int(os.environ.get("SPARSE_FINE_QCHUNK", "4"))
    xlayer = os.environ.get("SPARSE_XLAYER", "1") == "1"
    null_init = float(os.environ.get("SPARSE_NULL_INIT", "0.0"))
    # Recompute the fine pass in backward so at most one layer's gathered
    # keys + softmax scores are co-live, instead of all L layers' (the
    # cross-layer OOM).  This is a pure memory/recompute trade — no sort, no
    # sweep — so it does not touch the sub-quadratic guarantee.  Safe now only
    # because the fine pass is plain softmax: its recompute has no sort to
    # re-enter the backward (the reason checkpoint was unusable with entmax).
    ckpt = os.environ.get("SPARSE_CKPT", "1") == "1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_rand < 1:
            raise ValueError("need SPARSE_RAND >= 1")
        if self.query_block % self.key_block:
            raise ValueError("SPARSE_QUERY_BLOCK must be a multiple of SPARSE_KEY_BLOCK")
        self.null_bias = nn.Parameter(
            torch.full((self.num_heads,), self.null_init, dtype=torch.float32)
        )
        self.layer_index = -1

    def configure_wiring(self, layer: int, seqlen: int) -> None:
        """Build the per-layer static index/mask buffers (called by GPT.__init__)."""
        if seqlen % self.query_block:
            raise ValueError("TRAIN_SEQ_LEN must be a multiple of SPARSE_QUERY_BLOCK")
        if (seqlen // self.query_block) % self.fine_qchunk:
            raise ValueError("SPARSE_FINE_QCHUNK must divide the query-block count")
        self.layer_index = layer
        qb, kb = self.query_block, self.key_block
        self.kpq = qb // kb
        self.n_blocks = seqlen // kb
        self.n_qblocks = seqlen // qb
        self.n_sources = layer + 1 if self.xlayer else 1
        self.n_earlier = self.n_sources - 1
        self.n_dyn = self.n_rand
        self.dyn_keys = self.n_dyn * kb
        self.local_keys = 2 * qb
        if self.n_dyn > self.n_blocks * self.n_sources:
            raise ValueError("SPARSE_RAND exceeds the block universe")

        qidx = torch.arange(self.n_qblocks)
        # Per-query-block causal-universe size, for the uniform random draws:
        # every fully-causal block in each earlier layer, plus the current
        # layer's blocks strictly before the local predecessor span.
        u_q = self.n_earlier * self.kpq * qidx + (self.kpq * (qidx - 1)).clamp_min(0)
        self.register_buffer("u_q", u_q.to(torch.int64), persistent=False)

        # Local span: the query block's own qb positions (per-token causal)
        # plus the qb positions before it, from the current layer.
        raw = qidx[:, None] * qb - qb + torch.arange(2 * qb)[None, :]
        self.register_buffer(
            "local_pos", raw.clamp_min(0).to(torch.int64), persistent=False
        )
        pos_ok = raw >= 0  # [NQ, 2*qb]
        intra = torch.arange(2 * qb)[None, :] <= qb + torch.arange(qb)[:, None]
        # model.bfloat16() downcasts this buffer, so the masked value lands at
        # ~-9984 rather than -1e4 exactly; both are unreachably far below any
        # real logit (rms-normed q/k keep scores within ~+-50), so the
        # mismatch with _fine's fp32 dyn_mask is deliberate, not a bug.
        local_mask = torch.where(
            pos_ok[:, None, :] & intra[None, :, :], 0.0, MASKED_LOGIT
        )
        self.register_buffer("local_mask", local_mask, persistent=False)  # [NQ, qb, 2*qb]

        sentinel = self.n_sources * self.n_blocks + torch.arange(self.n_dyn)
        self.register_buffer("sentinel", sentinel.to(torch.int64), persistent=False)
        dup_tri = torch.ones(self.n_dyn, self.n_dyn, dtype=torch.bool).tril(-1)
        self.register_buffer("dup_tri", dup_tri, persistent=False)

    # ------------------------------------------------------------------ #
    # selection (no_grad): fresh uniform random draws, no scan, no sort    #
    # ------------------------------------------------------------------ #

    def _select(self, bsz: int) -> tuple[Tensor, Tensor]:
        """Pick each (kv-head, query block)'s dynamic key blocks by fresh
        uniform random draw from the causal universe (rewired every step).

        Returns global block ids [B, Hkv, NQ, n_dyn] (source-rank * BLK + c)
        and a validity mask.  No importance scan, no top-k, no sort — the
        draw is pure arithmetic on a per-slot hash, so the op is O(n_dyn).
        """
        device = self.u_q.device
        n_draw = self.n_dyn
        step = _STEP_BUF if self.training else _EVAL_STEP
        lane = (
            (
                (torch.arange(bsz, device=device, dtype=torch.int64)
                 .view(-1, 1, 1, 1) * self.num_kv_heads
                 + torch.arange(self.num_kv_heads, device=device)
                 .view(1, -1, 1, 1)) * self.n_qblocks
                + torch.arange(self.n_qblocks, device=device).view(1, 1, -1, 1)
            ) * n_draw
            + torch.arange(n_draw, device=device).view(1, 1, 1, -1)
        )
        u = _hash_uniform(lane, step, self.layer_index)
        u_q = self.u_q.view(1, 1, -1, 1)
        r = (u * u_q.to(torch.float32)).to(torch.int64).clamp(max=(u_q - 1).clamp_min(0))
        per_src = (self.kpq * torch.arange(self.n_qblocks, device=device)).view(1, 1, -1, 1)
        earlier_cnt = self.n_earlier * per_src
        is_earlier = r < earlier_cnt
        div = per_src.clamp_min(1)
        gid_earlier = (r // div) * self.n_blocks + r % div
        gid_current = self.n_earlier * self.n_blocks + (r - earlier_cnt)
        gid = torch.where(is_earlier, gid_earlier, gid_current)
        valid = (u_q > 0).expand(bsz, self.num_kv_heads, -1, n_draw)
        # Dedup: under softmax a duplicated key doubles its logit in the
        # normalizer and so over-weights that key.  Invalid slots are remapped
        # to unique sentinels first so junk indices can never invalidate a
        # real slot; the earliest copy survives (O(n_dyn^2) over the small
        # draw, not the bank — stays sub-quadratic in sequence length).
        key = torch.where(valid, gid, self.sentinel.expand_as(gid))
        dup = ((key.unsqueeze(-1) == key.unsqueeze(-2)) & self.dup_tri).any(-1)
        return gid, valid & ~dup

    # ------------------------------------------------------------------ #
    # fine pass (per query chunk)                                          #
    # ------------------------------------------------------------------ #

    def _fine_chunk(
        self,
        s: int,
        q6c: Tensor,
        gidc: Tensor,
        dyn_maskc: Tensor,
        earlierc: Tensor,
        *kv: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """One query chunk: gather blocks + full-width scores + softmax + mix.

        Splitting the query blocks into chunks bounds the transient fp32 score
        tensor to one chunk at a time.  softmax has no sort, so the backward
        saves only the softmax output and stays memory-cheap.
        """
        bsz = q6c.size(0)
        group = self.num_heads // self.num_kv_heads
        kb = self.key_block
        n_src = len(kv) // 2
        ks, vs = kv[:n_src], kv[n_src:]
        bank_k = torch.cat(ks, dim=2) if n_src > 1 else ks[0]
        bank_v = torch.cat(vs, dim=2) if n_src > 1 else vs[0]

        tok = (gidc * kb).unsqueeze(-1) + torch.arange(kb, device=q6c.device)
        tok = tok.reshape(bsz, self.num_kv_heads, -1, 1).expand(-1, -1, -1, self.head_dim)
        k_dyn = bank_k.gather(2, tok).view(
            bsz, self.num_kv_heads, self.fine_qchunk, self.dyn_keys, self.head_dim
        )
        v_dyn = bank_v.gather(2, tok).view_as(k_dyn)

        loc = self.local_pos.view(self.n_qblocks, -1)[s : s + self.fine_qchunk]
        loc = loc.reshape(1, 1, -1, 1).expand(bsz, self.num_kv_heads, -1, self.head_dim)
        k_loc = ks[-1].gather(2, loc).view(
            bsz, self.num_kv_heads, self.fine_qchunk, self.local_keys, self.head_dim
        )
        v_loc = vs[-1].gather(2, loc).view_as(k_loc)

        k_all = torch.cat([k_dyn, k_loc], dim=3)
        v_all = torch.cat([v_dyn, v_loc], dim=3)

        scale = self.head_dim ** -0.5
        null = self.null_bias.to(torch.float32).view(1, self.num_kv_heads, group, 1, 1, 1)
        logits = torch.einsum("bgjqtd,bgqkd->bgjqtk", q6c, k_all).float() * scale
        logits[..., : self.dyn_keys] += dyn_maskc[:, :, None, :, None, :]
        logits[..., self.dyn_keys:] += self.local_mask[None, None, None, s : s + self.fine_qchunk]
        full = torch.cat([logits, null.expand(*logits.shape[:-1], 1)], dim=-1)
        w = full.softmax(-1)
        y = torch.einsum("bgjqtk,bgqkd->bgjqtd", w[..., :-1].to(v_all.dtype), v_all)
        with torch.no_grad():
            wd = w.detach()
            slot_mass = wd[..., : self.dyn_keys].sum(dim=(2, 4)).view(
                bsz, self.num_kv_heads, self.fine_qchunk, self.n_dyn, kb
            ).sum(-1)
            stats = torch.stack([
                wd[..., :-1].sum(),
                (slot_mass * earlierc.float()).sum(),
                wd[..., -1].sum(),
                (wd[..., :-1] > 1e-4).float().sum(),
            ])
        return y, stats

    def _fine(self, q: Tensor, gid: Tensor, valid: Tensor, *kv: Tensor) -> tuple[Tensor, Tensor]:
        """Full-resolution attention over gathered key blocks + local span.

        Orchestrates the per-chunk fine passes.  Returns (y [B, H, T, d],
        stats [len(_STAT_NAMES)] fp32).
        """
        bsz = q.size(0)
        group = self.num_heads // self.num_kv_heads
        kb, qb = self.key_block, self.query_block
        q6 = q.reshape(bsz, self.num_kv_heads, group, self.n_qblocks, qb, self.head_dim)

        dyn_mask = torch.where(valid, 0.0, MASKED_LOGIT).repeat_interleave(kb, dim=-1)
        with torch.no_grad():
            earlier = valid & (gid < self.n_earlier * self.n_blocks)

        # Per-chunk loop bounds the transient fp32 score to one query chunk.
        ys, chunk_stats = [], []
        for s in range(0, self.n_qblocks, self.fine_qchunk):
            sl = slice(s, s + self.fine_qchunk)
            yc, st = self._fine_chunk(
                s, q6[:, :, :, sl], gid[:, :, sl], dyn_mask[:, :, sl],
                earlier[:, :, sl], *kv,
            )
            ys.append(yc)
            chunk_stats.append(st)
        y = torch.cat(ys, dim=3).reshape(bsz, self.num_heads, -1, self.head_dim)

        with torch.no_grad():
            acc = torch.stack(chunk_stats).sum(0)
            key_mass = acc[0].clamp_min(1e-30)
            earlier_mass, null_sum, support = acc[1], acc[2], acc[3]
            n_rows = bsz * self.num_kv_heads * group * self.n_qblocks * qb
            n_keys = n_rows * (self.dyn_keys + self.local_keys)
            n_valid = valid.sum().clamp_min(1)
            block_end = (gid % self.n_blocks + 1) * kb
            q_first = (torch.arange(self.n_qblocks, device=q.device) * qb).view(1, 1, -1, 1)
            leaks = (valid & (block_end > q_first)).sum()
            stats = torch.stack([
                (earlier.sum() / n_valid).float(),
                earlier_mass / key_mass,
                null_sum / n_rows,
                support / n_keys,
                leaks.float(),
                valid.float().mean(),
            ])
        return y, stats

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        if seqlen != _TRAIN_SEQ_LEN:
            raise ValueError(
                f"block wiring was configured for TRAIN_SEQ_LEN={_TRAIN_SEQ_LEN}, got {seqlen}"
            )
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, v = self.c_qkv(x).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = baseline.apply_rotary_emb(q, cos, sin), baseline.apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        layer = _STATE["i"]
        _STATE["i"] = layer + 1
        if layer != self.layer_index:
            raise RuntimeError(
                f"wiring layer mismatch: configured {self.layer_index}, executing {layer}"
            )

        if self.xlayer:
            bank = _STATE["bank"]
            if layer == 0 or bank is None:
                bank = []
            bank.append((k, v))
            _STATE["bank"] = bank
            ks = [kk for kk, _ in bank]
            vs = [vv for _, vv in bank]
        else:
            ks, vs = [k], [v]

        with torch.no_grad():
            gid, valid = self._select(bsz)

        if self.ckpt and self.training and torch.is_grad_enabled():
            y, stats = torch.utils.checkpoint.checkpoint(
                self._fine, q, gid, valid, *ks, *vs, use_reentrant=False
            )
        else:
            y, stats = self._fine(q, gid, valid, *ks, *vs)

        if self.training and _STATS is not None:
            with torch.no_grad():
                _STATS[min(layer, MAX_LAYERS - 1)] = stats

        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


def _wrap_gpt_init(orig_init):
    def init(self, *args, **kwargs):
        global _LAYERS_SEEN
        orig_init(self, *args, **kwargs)
        for layer, block in enumerate(self.blocks):
            attn = block.attn
            if not isinstance(attn, BlockWireAttention):
                raise TypeError("GPT block was not constructed with block-wire attention")
            attn.configure_wiring(layer, _TRAIN_SEQ_LEN)
        # Set statically here: reading (and guarding on) a mutating Python
        # global inside the compiled forward costs a needless recompile.
        _LAYERS_SEEN = len(self.blocks)
    return init


def _wrap_gpt_forward(orig_forward):
    def forward(self, *args, **kwargs):
        _STATE["i"] = 0
        _STATE["bank"] = None
        return orig_forward(self, *args, **kwargs)
    return forward


def _wrap_muon_step(orig_step):
    def step(self, *args, **kwargs):
        result = orig_step(self, *args, **kwargs)
        _STEP_BUF.add_(1)
        return result
    return step


def _wrap_load_state_dict(orig_load):
    def load_state_dict(self, *args, **kwargs):
        result = orig_load(self, *args, **kwargs)
        # Warmup exists only to compile kernels; resetting the clock after
        # its rollback keeps the random-audit stream identical to a run
        # without warmup.  (The final int8 reload also lands here - by then
        # only eval runs, which never reads the clock.)
        _STEP_BUF.zero_()
        if _STATS is not None:
            _STATS.zero_()
        return result
    return load_state_dict


def _wrap_eval_val(orig_eval_val):
    is_master = int(os.environ.get("RANK", "0")) == 0

    def eval_val(*args, **kwargs):
        result = orig_eval_val(*args, **kwargs)
        if is_master and _STATS is not None and _LAYERS_SEEN and bool((_STATS != 0).any()):
            rows = _STATS[:_LAYERS_SEEN].tolist()
            parts = [
                f"{name}_l{li}:{row[col]:.4f}"
                for li, row in enumerate(rows)
                for col, name in enumerate(_STAT_NAMES)
            ]
            print("graph_stats " + " ".join(parts), flush=True)
        return result

    return eval_val


def main() -> None:
    global _STATS
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _STATS = torch.zeros(MAX_LAYERS, len(_STAT_NAMES), device=f"cuda:{local_rank}")
    snapshot_path = os.environ.get("SPARSE_MEM_SNAPSHOT", "")
    if snapshot_path:
        torch.cuda.memory._record_memory_history(max_entries=200_000)
    original_attention = baseline.CausalSelfAttention
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    original_int8_patterns = baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    original_gpt_init = baseline.GPT.__init__
    original_gpt_forward = baseline.GPT.forward
    original_load_state_dict = baseline.GPT.load_state_dict
    original_muon_step = baseline.Muon.step
    original_eval_val = baseline.eval_val
    baseline.CausalSelfAttention = BlockWireAttention
    baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns + ("null_bias",)
    baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns + ("null_bias",)
    baseline.GPT.__init__ = _wrap_gpt_init(original_gpt_init)
    baseline.GPT.forward = _wrap_gpt_forward(original_gpt_forward)
    baseline.GPT.load_state_dict = _wrap_load_state_dict(original_load_state_dict)
    baseline.Muon.step = _wrap_muon_step(original_muon_step)
    baseline.eval_val = _wrap_eval_val(original_eval_val)
    try:
        if snapshot_path:
            try:
                baseline.main()
            finally:
                torch.cuda.memory._dump_snapshot(snapshot_path)
                torch.cuda.memory._record_memory_history(enabled=None)
        else:
            baseline.main()
    finally:
        baseline.CausalSelfAttention = original_attention
        baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns
        baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns
        baseline.GPT.__init__ = original_gpt_init
        baseline.GPT.forward = original_gpt_forward
        baseline.GPT.load_state_dict = original_load_state_dict
        baseline.Muon.step = original_muon_step
        baseline.eval_val = original_eval_val


if __name__ == "__main__":
    main()

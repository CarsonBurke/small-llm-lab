"""FlexAttention block-sparse fine pass for the randomly-rewired cross-layer
attention (xlayer v3) — a memory/speed prototype of ``blockwire_train_gpt``.

The blockwire fine pass (``BlockWireAttention._fine``) *materializes* an fp32
score tensor ``[B, Hkv, group, qchunk, qb, n_keys]`` per query chunk, *gathers*
a physical copy of every selected key/value block, and softmaxes the dense
result.  That is memory-heavy (the gathered keys + the score + the softmax
output are all co-live) and leaves the tensor-core matmul boundary-padded to
the gathered layout.  This fork keeps the *exact same* random-rewiring
selection (``_select`` is reused verbatim) but hands the actual attention to
``torch.nn.attention.flex_attention.flex_attention`` over a block-sparse
``BlockMask``, so that:

* the [queries x keys] score is never materialized (flex fuses it in-kernel),
* the selected keys are never gathered into a copy (flex reads the bank
  in place through the block index tables),
* the op stays genuinely sub-quadratic (only the selected blocks are visited).

Granularity change vs blockwire.  FlexAttention tiles at ``BLOCK`` (default
128) for both queries and keys, so the random draws now select *128-token*
blocks rather than blockwire's 16-token ``SPARSE_KEY_BLOCK``.  ``_select`` is
reused with ``key_block == query_block == BLOCK``; it then returns, per
(kv-head, query-block), ``n_rand`` global *128-block* ids that are fully causal
by construction (the blockwire causal-universe math with ``kpq == 1``).  This
is a deliberate coarsening — a prototype trades blockwire's fine-grained
selection for the FlexAttention kernel — and is NOT quality-comparable to the
16-token path; it is memory/speed-comparable.

BlockMask construction (the crux).  For each (b, kv-head, query-block) the keys
split into two causal classes:

* FULL blocks — the ``n_rand`` random draws plus the current layer's local
  predecessor block (query-block ``q-1``).  Every one is *fully* causal (its
  last token precedes the query block's first token), so it needs no per-token
  mask and goes in ``full_kv_*`` where FlexAttention skips ``mask_mod`` entirely.
* PARTIAL block — the current layer's own diagonal block (query-block ``q`` in
  the current source).  It straddles the causal boundary, so it goes in
  ``kv_*`` (the mask_mod-checked set) with a causal ``mask_mod``
  (``q_idx + base >= kv_idx``, ``base = n_earlier * T``).

These two classes are disjoint by construction (the random universe for the
current source is blocks ``[0, q-1)``, which excludes both the predecessor
``q-1`` and the diagonal ``q``), so no key is ever double-counted.

fullgraph / no-sort.  ``BlockMask.from_kv_blocks(compute_q_blocks=True)`` runs
``torch.argsort`` (twice) to transpose the block table for the backward pass;
building it *inside* the ``fullgraph=True`` compiled model would inject
``aten.sort`` into the graph (verified) and violate the sub-quadratic-by-no-sort
guarantee.  A ``forward_pre_hook`` on the *inner* GPT does NOT help — pytorch
traces inner-module pre-hooks into the compiled graph (verified: the hook's sort
lands in the graph).  The hook must sit on the *outer* ``OptimizedModule``
returned by ``torch.compile``, whose ``__call__`` runs its own pre-hooks
*eagerly* before dispatching to the compiled forward (verified: the sort stays
out of the graph, and swapping in a fresh mask each step does not recompile).
So ``main()`` intercepts the model-level ``torch.compile`` call and hangs the
hook on the compiled module.  The hook builds every layer's ``BlockMask`` from
``_STEP_BUF`` + batch size (the selection depends only on the step clock and the
static wiring, never on activations); the compiled forward only *reads* the
prebuilt ``BlockMask`` from ``_STATE["masks"]`` and calls ``flex_attention``, so
zero sort lands in the graph.  The argsort is over the tiny block grid
(``NQ`` x ``S*NQ``), constant per step, not a sequence-length sort.  (The
``mask_mod`` closure is cached once per layer in ``configure_wiring`` so its
object identity is stable across steps and cannot trigger a recompile.)

Null option.  blockwire adds a per-head scalar null column that absorbs
0.3-0.6 of the softmax mass ("attend to nothing").  FlexAttention cannot add a
column via ``score_mod``, so for this prototype the null is OFF (``FLEX_NULL``
defaults to 0) and the ``null_bias`` parameter is not created.  A null-less run
is therefore memory/speed-comparable to the materialized path but NOT
quality-comparable.  The report describes the appended-null-block design for
turning it back on; ``FLEX_NULL=1`` raises until that path is validated on GPU.

Env knobs: FLEX_BLOCK (128), SPARSE_RAND (8), SPARSE_XLAYER (1), FLEX_NULL (0),
TRAIN_SEQ_LEN.  This fork does not modify ``train_gpt.py`` or
``blockwire_train_gpt.py``.
"""

from __future__ import annotations

import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_XLAYER = pathlib.Path(__file__).resolve().parent
if str(_XLAYER) not in sys.path:
    sys.path.insert(0, str(_XLAYER))

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

import train_gpt as baseline
import blockwire_train_gpt as bw
from blockwire_train_gpt import (
    MAX_LAYERS,
    _STATE,
    _STAT_NAMES,
    BlockWireAttention,
)

_TRAIN_SEQ_LEN = int(os.environ.get("TRAIN_SEQ_LEN", "1024"))
_FLEX_BLOCK = int(os.environ.get("FLEX_BLOCK", "128"))
_FLEX_NULL = os.environ.get("FLEX_NULL", "0") == "1"

# Masks are prebuilt per step by the GPT forward_pre_hook (eager, outside the
# compiled graph) and keyed by layer index; the compiled forward only reads
# them.  Lives on the shared blockwire _STATE so the compiled forward closes
# over one global container (same pattern as _STATE["bank"]).
_STATE.setdefault("masks", {})


class FlexRewireAttention(BlockWireAttention):
    # Both query and key tiles are FLEX_BLOCK wide: _select then draws whole
    # 128-token blocks (kpq == 1).  Overrides blockwire's SPARSE_KEY/QUERY_BLOCK
    # env reads so the flex granularity is independent of blockwire's knobs.
    key_block = _FLEX_BLOCK
    query_block = _FLEX_BLOCK
    # flex_attention tiles queries itself; the blockwire per-chunk loop knob is
    # unused here, so pin it to 1 (it must still divide the query-block count in
    # the inherited configure_wiring, and 1 always does).
    fine_qchunk = 1
    # flex_attention has its own fused, recompute-free backward; the blockwire
    # activation checkpoint (there to bound the materialized score) is pure
    # overhead here, so disable it and let the inherited forward call _fine
    # directly.
    ckpt = False

    def __init__(self, *args, **kwargs):
        # Bypass BlockWireAttention.__init__ (it unconditionally allocates the
        # null_bias parameter); run the grandparent CausalSelfAttention.__init__
        # and add null_bias only when the null column is actually wired.  An
        # always-unused requires_grad parameter would break DDP
        # (find_unused_parameters=False) on the 8xH100 validation run.
        #
        # Reach the grandparent through the MRO (``super(BlockWireAttention,
        # self)``), NOT through ``baseline.CausalSelfAttention`` — main()
        # rebinds that module attribute to this very class, so calling it here
        # would recurse infinitely.  The MRO is fixed at class-definition time
        # and still resolves to the original CausalSelfAttention.
        super(BlockWireAttention, self).__init__(*args, **kwargs)
        if self.n_rand < 1:
            raise ValueError("need SPARSE_RAND >= 1")
        if self.query_block % self.key_block:
            raise ValueError("query_block must be a multiple of key_block")
        if _FLEX_NULL:
            raise NotImplementedError(
                "FLEX_NULL=1 is not wired in this prototype; see the module "
                "docstring / report for the appended-null-block design."
            )
        self.layer_index = -1
        self._mask_mod = None  # cached per-layer once configure_wiring runs

    def configure_wiring(self, layer: int, seqlen: int) -> None:
        super().configure_wiring(layer, seqlen)
        self._seqlen = seqlen  # authoritative seq length for this layer's mask
        # Build the causal mask_mod ONCE per layer and cache it: a fresh closure
        # each step would change the mask_mod object identity and force
        # flex_attention to recompile the kernel every step.  ``base`` is the
        # token offset of the current source in the concatenated bank; it is
        # static after wiring.  The mask is written to be GLOBALLY valid — correct
        # whether or not the kernel skips it for full blocks — so the fused and
        # unfused paths agree: earlier-source keys (kv < base) are always causal
        # (their blocks are structurally before the query span), current-source
        # keys are causal iff q_idx + base >= kv_idx.
        base = self.n_earlier * seqlen

        def causal_mask(b, h, q_idx, kv_idx):
            return (kv_idx < base) | (q_idx + base >= kv_idx)

        self._mask_mod = causal_mask

    # ------------------------------------------------------------------ #
    # BlockMask construction (eager; runs in the GPT forward_pre_hook)      #
    # ------------------------------------------------------------------ #

    def build_block_mask(self, gid: Tensor, valid: Tensor, bsz: int) -> BlockMask:
        """Turn one layer's ``_select`` output into a FlexAttention BlockMask.

        ``gid``/``valid`` are ``[B, Hkv, NQ, n_rand]``.  ``gid`` is already the
        absolute 128-block index into the concatenated bank (source ``s``
        occupies blocks ``[s*n_blocks, (s+1)*n_blocks)`` and ``gid = s*n_blocks
        + c``).  Builds, per (b, query-head, query-block):

        * partial set = the single current-source diagonal block ``q`` (causal
          ``mask_mod``);
        * full set = the valid random blocks + the current-source predecessor
          block ``q-1`` (fully causal, no mask_mod), compacted to the front.

        Runs eagerly (argsort/vmap inside ``from_kv_blocks`` must NOT enter the
        compiled graph).  Returns a BlockMask whose block tables have H ==
        num_heads: the per-kv-head selection is expanded to query heads
        (repeat_interleave by the GQA group), because FlexAttention indexes the
        block mask by *query* head.
        """
        device = gid.device
        nq = self.n_qblocks
        nb = self.n_blocks
        group = self.num_heads // self.num_kv_heads
        base_blk = self.n_earlier * nb  # current source's first block index
        total_kv = self.n_sources * nb  # BCSR column width == total kv blocks
        qidx = torch.arange(nq, device=device)

        # kv_indices is BCSR-style: last dim spans ALL kv blocks (values are
        # absolute block ids in [0, total_kv)); only the first ``num`` entries
        # per row are meaningful.  (create_block_mask uses the same layout.)

        # --- partial: the diagonal block, one per query row -------------- #
        diag = (base_blk + qidx).to(torch.int32)  # [NQ]
        part_num = torch.ones(bsz, self.num_kv_heads, nq, dtype=torch.int32, device=device)
        part_idx = torch.zeros(bsz, self.num_kv_heads, nq, total_kv, dtype=torch.int32, device=device)
        part_idx[..., 0] = diag.view(1, 1, nq)

        # --- full: random draws + local predecessor block --------------- #
        pred = (base_blk + qidx - 1)  # [NQ]; invalid (masked) for q == 0
        pred_valid = (qidx > 0).view(1, 1, nq, 1).expand(bsz, self.num_kv_heads, nq, 1)
        pred_idx = pred.clamp_min(0).view(1, 1, nq, 1).expand_as(pred_valid)

        idx_all = torch.cat([gid.to(torch.int32), pred_idx.to(torch.int32)], dim=-1)
        valid_all = torch.cat([valid, pred_valid], dim=-1)  # [B, Hkv, NQ, n_rand+1]

        # Compact valid blocks to the front so the first full_num entries of
        # each row are the real blocks (from_kv_blocks reads only those).  This
        # per-row reorder is a constant-size (n_rand+1) argsort, and it runs
        # eagerly here — never in the compiled graph.
        order = torch.argsort(valid_all.to(torch.int32), dim=-1, descending=True, stable=True)
        comp = torch.gather(idx_all, -1, order)  # [B, Hkv, NQ, n_rand+1], valid-first
        full_num = valid_all.sum(-1).clamp_max(total_kv).to(torch.int32)
        full_idx = torch.zeros(bsz, self.num_kv_heads, nq, total_kv, dtype=torch.int32, device=device)
        w = min(comp.size(-1), total_kv)
        full_idx[..., :w] = comp[..., :w]

        # Expand per-kv-head tables to query heads (flex indexes by query head).
        def to_qheads(t: Tensor) -> Tensor:
            return t.repeat_interleave(group, dim=1).contiguous()

        part_num, part_idx = to_qheads(part_num), to_qheads(part_idx)
        full_num, full_idx = to_qheads(full_num), to_qheads(full_idx)

        q_len = self._seqlen
        kv_len = self.n_sources * self._seqlen
        return BlockMask.from_kv_blocks(
            part_num, part_idx, full_num, full_idx,
            BLOCK_SIZE=_FLEX_BLOCK, mask_mod=self._mask_mod, seq_lengths=(q_len, kv_len),
        )

    # ------------------------------------------------------------------ #
    # fine pass (flex_attention over the prebuilt BlockMask)               #
    # ------------------------------------------------------------------ #

    def _fine(self, q: Tensor, gid: Tensor, valid: Tensor, *kv: Tensor) -> tuple[Tensor, Tensor]:
        """Block-sparse attention via flex_attention; matches the inherited
        forward's ``(y [B, H, T, d], stats)`` contract.

        Reads the BlockMask prebuilt for this layer by the GPT pre-hook.  Falls
        back to building it eagerly here when absent (e.g. unit tests that drive
        ``_fine`` directly, or an eager forward with no pre-hook registered).
        """
        bsz = q.size(0)
        n_src = len(kv) // 2
        ks, vs = kv[:n_src], kv[n_src:]
        bank_k = torch.cat(ks, dim=2) if n_src > 1 else ks[0]
        bank_v = torch.cat(vs, dim=2) if n_src > 1 else vs[0]

        bm = _STATE.get("masks", {}).get(self.layer_index)
        if bm is None:
            bm = self.build_block_mask(gid, valid, bsz)

        y = flex_attention(
            q, bank_k, bank_v, block_mask=bm,
            enable_gqa=(self.num_kv_heads != self.num_heads),
            scale=self.head_dim ** -0.5,
        )
        stats = self._structural_stats(gid, valid, q.device)
        return y, stats

    def _structural_stats(self, gid: Tensor, valid: Tensor, device) -> Tensor:
        """Selection-side stats (no softmax weights available from flex here).

        Fills the columns derivable from the draw — cross-layer selection
        fraction, validity, and the future-leak counter (must be 0) — and zeros
        the mass/support columns that need per-key softmax weights.  Enough to
        confirm the wiring is healthy in tensorboard.
        """
        with torch.no_grad():
            nb = self.n_blocks
            n_valid = valid.sum().clamp_min(1)
            earlier = valid & (gid < self.n_earlier * nb)
            block_end = (gid % nb + 1) * self.key_block
            q_first = (torch.arange(self.n_qblocks, device=device) * self.query_block).view(1, 1, -1, 1)
            leaks = (valid & (block_end > q_first)).sum()
            zero = torch.zeros((), device=device)
            return torch.stack([
                (earlier.sum() / n_valid).float(),  # bw_xsel
                zero,                                # bw_xmass  (needs weights)
                zero,                                # bw_null   (no null column)
                zero,                                # bw_support(needs weights)
                leaks.float(),                       # bw_leak
                valid.float().mean(),                # bw_valid
            ])


# ---------------------------------------------------------------------- #
# GPT wiring: build all layers' BlockMasks eagerly, before the graph runs #
# ---------------------------------------------------------------------- #

def _build_masks_prehook(module, args):
    """forward_pre_hook: precompute every layer's BlockMask for this step.

    Registered on the *compiled* module (the ``OptimizedModule`` returned by
    ``torch.compile``), NOT the inner GPT: a pre-hook on the inner module is
    traced into the ``fullgraph=True`` graph (its ops, incl. the ``from_kv_blocks``
    argsort, would land in the graph — verified), whereas the OptimizedModule
    runs its own hooks *eagerly* before dispatching to the compiled forward
    (also verified: the hook's sort stays out of the graph and per-step mask
    swaps do not recompile).  The selection reads ``_STEP_BUF`` / ``_EVAL_STEP``
    and the static wiring only — no activations — so it is safe to run first.
    """
    input_ids = args[0]
    bsz = input_ids.size(0)
    blocks = getattr(module, "blocks", None)
    if blocks is None:  # OptimizedModule delegates attribute access to _orig_mod
        blocks = module._orig_mod.blocks
    masks: dict = {}
    with torch.no_grad():
        for block in blocks:
            attn = block.attn
            gid, valid = attn._select(bsz)
            masks[attn.layer_index] = attn.build_block_mask(gid, valid, bsz)
    _STATE["masks"] = masks
    return args


def _wrap_gpt_init(orig_init):
    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        for layer, block in enumerate(self.blocks):
            attn = block.attn
            if not isinstance(attn, FlexRewireAttention):
                raise TypeError("GPT block was not constructed with flex-rewire attention")
            attn.configure_wiring(layer, _TRAIN_SEQ_LEN)
        bw._LAYERS_SEEN = len(self.blocks)
    return init


def _wrap_torch_compile(orig_compile):
    """Attach the mask pre-hook to the compiled GPT (OptimizedModule).

    train_gpt calls ``torch.compile(base_model, fullgraph=True)`` at the model
    level; we intercept that call so the hook lands on the OUTER compiled
    module, where it fires eagerly (see ``_build_masks_prehook``).  Other
    ``torch.compile`` calls in the baseline (e.g. the Newton-Schulz helper) pass
    through untouched.
    """
    def compile(*args, **kwargs):
        compiled = orig_compile(*args, **kwargs)
        model = args[0] if args else kwargs.get("model")
        if isinstance(model, baseline.GPT):
            compiled.register_forward_pre_hook(_build_masks_prehook)
        return compiled
    return compile


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    bw._STATS = torch.zeros(MAX_LAYERS, len(_STAT_NAMES), device=f"cuda:{local_rank}")

    original_attention = baseline.CausalSelfAttention
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    original_int8_patterns = baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    original_gpt_init = baseline.GPT.__init__
    original_gpt_forward = baseline.GPT.forward
    original_load_state_dict = baseline.GPT.load_state_dict
    original_muon_step = baseline.Muon.step
    original_eval_val = baseline.eval_val
    original_torch_compile = torch.compile

    baseline.CausalSelfAttention = FlexRewireAttention
    baseline.GPT.__init__ = _wrap_gpt_init(original_gpt_init)
    baseline.GPT.forward = bw._wrap_gpt_forward(original_gpt_forward)
    baseline.GPT.load_state_dict = bw._wrap_load_state_dict(original_load_state_dict)
    baseline.Muon.step = bw._wrap_muon_step(original_muon_step)
    baseline.eval_val = bw._wrap_eval_val(original_eval_val)
    # Intercept the model-level compile to hang the mask pre-hook on the
    # OptimizedModule (so it runs eagerly, outside the fullgraph graph).
    torch.compile = _wrap_torch_compile(original_torch_compile)
    try:
        baseline.main()
    finally:
        baseline.CausalSelfAttention = original_attention
        baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns
        baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns
        baseline.GPT.__init__ = original_gpt_init
        baseline.GPT.forward = original_gpt_forward
        baseline.GPT.load_state_dict = original_load_state_dict
        baseline.Muon.step = original_muon_step
        torch.compile = original_torch_compile
        baseline.eval_val = original_eval_val


if __name__ == "__main__":
    main()

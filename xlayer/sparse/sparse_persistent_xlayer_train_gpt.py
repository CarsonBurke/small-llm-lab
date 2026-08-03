"""Persistent, optimizer-step-rewired cross-layer sparse entmax attention.

This is the no-selector sparse graph arm.  Every consumer (layer, query head)
owns ``SPARSE_K`` relative templates ``(source_layer, lag)``.  The templates
are shared by all rows/examples, survive optimizer steps and serialization,
and address exactly K entries in the source-private cross-layer K/V bank.

During each optimizer step, installed wires accumulate their actual entmax
mass.  Their utility is the mass relative to the uniform-share opportunity::

    U_e = sum_rows p_e / sum_rows 1 / (m_row + 1)

``m_row`` is the number of valid installed wires and the extra route is the
learned null.  Invalid occurrences contribute zero mass but retain the
denominator.  After every Muon step, every wire with U < 1 is replaced.  New
wires come from a deterministic without-replacement traversal of the entire
causal ``(source_layer, lag)`` universe; active duplicates are skipped.  No
wire is protected and no absent wire is scored.

The existing xlayer list-input custom op keeps each source layer's own rotary
K/V space and avoids retaining a concatenated bank per consumer.  This fork
does not modify ``train_gpt.py``.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import math
import os
from collections.abc import Iterable
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

import train_gpt as baseline
from xlayer.sparse.sparse_xlayer_kernel import xlayer_entmax_attention_stats


MAX_LAYERS = 32
UTILITY_THRESHOLD = float(os.environ.get("WIRE_UTILITY_THRESHOLD", "1.0"))
_TRAIN_SEQ_LEN = int(os.environ.get("TRAIN_SEQ_LEN", "1024"))
_GRAPH_MODE = os.environ.get("SPARSE_GRAPH_MODE", "xlayer")
if _GRAPH_MODE not in {"xlayer", "same_layer"}:
    raise ValueError("SPARSE_GRAPH_MODE must be 'xlayer' or 'same_layer'")
_MODULES: list["PersistentXLayerAttention"] = []
_STATE: dict[str, object] = {"i": 0, "kv": None}
_LAYERS_SEEN = 0
_GRAPH_STATS: Tensor | None = None
_REWIRING_ACTIVE = int(os.environ.get("WARMUP_STEPS", "20")) == 0


def _seed(layer: int, head: int, epoch: int) -> int:
    """Stable integer mixer used only for graph traversal, not model RNG."""
    x = (0x9E3779B9 * (layer + 1) + 0x85EBCA6B * (head + 1) + 0xC2B2AE35 * (epoch + 1))
    x ^= x >> 16
    x = (x * 0x45D9F3B) & 0x7FFFFFFF
    x ^= x >> 15
    return x & 0x7FFFFFFF


def _affine_permutation_params(universe_size: int, layer: int, head: int, epoch: int) -> tuple[int, int]:
    """Return ``a,b`` such that ``(a*i+b) % N`` permutes ``range(N)``."""
    if universe_size <= 0:
        raise ValueError("universe_size must be positive")
    mixed = _seed(layer, head, epoch)
    a = (mixed % universe_size) or 1
    while math.gcd(a, universe_size) != 1:
        a = (a + 1) % universe_size or 1
    b = _seed(layer, head, epoch + 0x10001) % universe_size
    return a, b


def _permuted_candidate(universe_size: int, layer: int, head: int, epoch: int, cursor: int) -> int:
    return int(_candidate_permutation(universe_size, layer, head, epoch)[cursor])


@lru_cache(maxsize=256)
def _candidate_permutation(universe_size: int, layer: int, head: int, epoch: int) -> Tensor:
    """Seeded random traversal of a finite universe, cached one epoch at a time."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed(layer, head, epoch))
    return torch.randperm(universe_size, generator=generator, dtype=torch.int64)


def initial_templates(
    layer: int, num_heads: int, k_budget: int, seqlen: int, *, same_layer: bool = False
) -> tuple[Tensor, Tensor]:
    """Deterministically initialize a diverse, duplicate-free graph on CPU.

    The production K=128 layout starts with 64 current-layer local lags, 32
    current-layer strided lags and 32 all-prior draws.  Small test shapes use
    the same ordering truncated to their feasible universe.
    """
    universe_size = seqlen if same_layer else (layer + 1) * seqlen
    if k_budget > universe_size:
        raise ValueError(
            f"SPARSE_K={k_budget} exceeds layer {layer}'s universe ({universe_size}); "
            "reduce SPARSE_K or increase TRAIN_SEQ_LEN"
        )
    encoded = torch.empty(num_heads, k_budget, dtype=torch.int64)
    for head in range(num_heads):
        chosen: list[int] = []
        occupied: set[int] = set()

        def add(candidate: int) -> None:
            if len(chosen) < k_budget and candidate not in occupied:
                occupied.add(candidate)
                chosen.append(candidate)

        n_prior = min(32, k_budget, layer * seqlen) if layer > 0 and not same_layer else 0
        current_target = k_budget - n_prior
        n_local = min(64, current_target, seqlen)
        for lag in range(n_local):
            add(layer * seqlen + lag)

        n_strided = min(32, current_target - len(chosen))
        if n_strided and seqlen > n_local:
            for i in range(n_strided):
                lag = n_local + ((i + 1) * (seqlen - n_local) // (n_strided + 1))
                add(layer * seqlen + min(lag, seqlen - 1))

        # Finish the current-layer quota before adding the exact prior quota.
        for lag in range(seqlen):
            add(layer * seqlen + lag)
            if len(chosen) == current_target:
                break

        if n_prior:
            # Stratify source layers exactly (counts differ by at most one),
            # then independently shuffle each source's lag space.  Sampling a
            # prefix of one flat modular permutation aliases badly with T and
            # can omit entire prior layers at production dimensions.
            per_source_rank = [0] * layer
            for i in range(n_prior):
                source = (head + i) % layer
                source_rank = per_source_rank[source]
                per_source_rank[source] += 1
                lag_order = _candidate_permutation(
                    seqlen, layer * MAX_LAYERS + source, head, -1
                )
                add(source * seqlen + int(lag_order[source_rank]))

        # Fill the remaining slots from a complete deterministic permutation.
        # Earlier layers receive a head-specific offset naturally; active
        # structural entries are skipped without consuming graph traversal.
        a, b = _affine_permutation_params(universe_size, layer, head, -1)
        for i in range(universe_size):
            candidate = (a * i + b) % universe_size
            add(layer * seqlen + candidate if same_layer else candidate)
            if len(chosen) == k_budget:
                break
        if len(chosen) != k_budget:
            raise RuntimeError("failed to initialize a complete sparse graph")
        encoded[head] = torch.tensor(chosen, dtype=torch.int64)

    # int32 is accepted by both Gloo and NCCL broadcasts; int16 is not a
    # portable distributed collective dtype and saves only a few KiB here.
    return (encoded // seqlen).to(torch.int32), (encoded % seqlen).to(torch.int32)


def eager_rewire_templates(
    source_layer: Tensor,
    lag: Tensor,
    utility: Tensor,
    cursor: Tensor,
    epoch: Tensor,
    traversed: Tensor,
    *,
    layer: int,
    seqlen: int,
    threshold: float = 1.0,
    same_layer: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """CPU reference and optimizer-step graph update.

    Every below-threshold slot is mutable.  Traversal visits each candidate
    once per epoch, skips currently occupied templates, and reseeds only after
    exhausting the complete universe.  Returned ``rewired`` has shape [H,K].
    """
    tensors = (source_layer, lag, utility, cursor, epoch, traversed)
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("eager_rewire_templates expects CPU tensors")
    if source_layer.shape != lag.shape or utility.shape != source_layer.shape:
        raise ValueError("source_layer, lag, and utility must share the installed [H,K] shape")
    if any(t.shape != (source_layer.shape[0],) for t in (cursor, epoch, traversed)):
        raise ValueError("cursor, epoch, and traversed must have one entry per head")
    src = source_layer.clone()
    dst_lag = lag.clone()
    next_cursor = cursor.clone()
    next_epoch = epoch.clone()
    next_traversed = traversed.clone()
    rewired = utility < threshold
    num_heads, k_budget = src.shape
    universe_size = seqlen if same_layer else (layer + 1) * seqlen

    for head in range(num_heads):
        # All installed templates are excluded, even the below-threshold ones:
        # rewiring must audition a genuinely absent wire rather than reinstall
        # a doomed template in another slot.
        occupied = {
            int(src[head, slot]) * seqlen + int(dst_lag[head, slot])
            for slot in range(k_budget)
        }
        if int(rewired[head].sum()) > universe_size - len(occupied):
            raise ValueError("candidate universe has insufficient absent templates for full rewiring")
        cur = int(next_cursor[head])
        ep = int(next_epoch[head])
        seen = int(next_traversed[head])
        for slot in range(k_budget):
            if not bool(rewired[head, slot]):
                continue
            while True:
                proposal = _permuted_candidate(universe_size, layer, head, ep, cur)
                candidate = layer * seqlen + proposal if same_layer else proposal
                cur += 1
                seen += 1
                if cur == universe_size:
                    cur = 0
                    ep += 1
                if candidate not in occupied:
                    occupied.add(candidate)
                    src[head, slot] = candidate // seqlen
                    dst_lag[head, slot] = candidate % seqlen
                    break
        next_cursor[head] = cur
        next_epoch[head] = ep
        next_traversed[head] = seen

    return src, dst_lag, next_cursor, next_epoch, next_traversed, rewired


def graph_indices(source_layer: Tensor, lag: Tensor, bsz: int, seqlen: int) -> tuple[Tensor, Tensor]:
    """Expand state-independent relative templates to fused-kernel indices."""
    if source_layer.shape != lag.shape:
        raise ValueError("source_layer and lag must have identical [H,K] shapes")
    pos = torch.arange(seqlen, device=lag.device, dtype=torch.int32).view(1, seqlen, 1)
    src_pos = pos - lag.to(torch.int32).unsqueeze(1)
    idx = source_layer.to(torch.int32).unsqueeze(1) * seqlen + src_pos.clamp_min(0)
    valid = src_pos >= 0
    return (
        idx.unsqueeze(0).expand(bsz, -1, -1, -1),
        valid.unsqueeze(0).expand(bsz, -1, -1, -1),
    )


def utility_components(p: Tensor, pn: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Reduce one microbatch into optimizer-step utility sufficient stats."""
    m_incl = valid.sum(dim=-1, dtype=torch.float32) + 1.0
    denominator = (1.0 / m_incl).sum(dim=(0, 2))
    numerator = p.float().sum(dim=(0, 2))
    support = ((p > 0) & valid).sum(dim=(0, 2), dtype=torch.float32)
    opportunities = valid.sum(dim=(0, 2), dtype=torch.float32)
    null_sum = pn.float().sum(dim=(0, 2))
    row_count = torch.full_like(null_sum, p.shape[0] * p.shape[2], dtype=torch.float32)
    return numerator, denominator, support, opportunities, null_sum, row_count


class PersistentXLayerAttention(baseline.CausalSelfAttention):
    k_budget = int(os.environ.get("SPARSE_K", "128"))
    null_init = float(os.environ.get("SPARSE_NULL_INIT", "0.0"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.k_budget <= 0 or self.k_budget & (self.k_budget - 1):
            raise ValueError("SPARSE_K must be a positive power of two")
        self.null_bias = nn.Parameter(torch.full((self.num_heads,), self.null_init, dtype=torch.float32))
        self.layer_index = -1
        # Configured by GPT.__init__ before the model is moved to CUDA and
        # before baseline warmup snapshots state_dict.
        self.register_buffer("wire_source_layer", torch.empty(0, dtype=torch.int32))
        self.register_buffer("wire_lag", torch.empty(0, dtype=torch.int32))
        self.register_buffer("wire_cursor", torch.empty(0, dtype=torch.int64))
        self.register_buffer("wire_epoch", torch.empty(0, dtype=torch.int64))
        self.register_buffer("wire_traversed", torch.empty(0, dtype=torch.int64))
        self.register_buffer("utility_num", torch.empty(0), persistent=False)
        self.register_buffer("utility_den", torch.empty(0), persistent=False)
        self.register_buffer("support_num", torch.empty(0), persistent=False)
        self.register_buffer("valid_num", torch.empty(0), persistent=False)
        self.register_buffer("null_sum", torch.empty(0), persistent=False)
        self.register_buffer("row_count", torch.empty(0), persistent=False)

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        # baseline constructs ``model.to(device).bfloat16()``.  Graph evidence
        # spans eight microbatches and must accumulate in fp32, so restore only
        # the nonpersistent statistic buffers after dtype-wide module casts.
        for name in ("utility_num", "utility_den", "support_num", "valid_num", "null_sum", "row_count"):
            tensor = getattr(self, name)
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                setattr(self, name, tensor.float())
        return result

    def configure_graph(self, layer: int, seqlen: int) -> None:
        self.layer_index = layer
        universe_size = seqlen if _GRAPH_MODE == "same_layer" else (layer + 1) * seqlen
        if universe_size < 2 * self.k_budget:
            raise ValueError(
                f"layer {layer} candidate universe ({universe_size}) must contain at least "
                f"2*SPARSE_K ({2 * self.k_budget}) so every wire can change in one step"
            )
        src, lag = initial_templates(
            layer, self.num_heads, self.k_budget, seqlen,
            same_layer=_GRAPH_MODE == "same_layer",
        )
        self.wire_source_layer = src
        self.wire_lag = lag
        self.wire_cursor = torch.zeros(self.num_heads, dtype=torch.int64)
        self.wire_epoch = torch.zeros(self.num_heads, dtype=torch.int64)
        self.wire_traversed = torch.zeros(self.num_heads, dtype=torch.int64)
        self.utility_num = torch.zeros(self.num_heads, self.k_budget, dtype=torch.float32)
        self.utility_den = torch.zeros(self.num_heads, dtype=torch.float32)
        self.support_num = torch.zeros(self.num_heads, self.k_budget, dtype=torch.float32)
        self.valid_num = torch.zeros(self.num_heads, self.k_budget, dtype=torch.float32)
        self.null_sum = torch.zeros(self.num_heads, dtype=torch.float32)
        self.row_count = torch.zeros(self.num_heads, dtype=torch.float32)

    @torch.no_grad()
    def reset_utility(self) -> None:
        for name in ("utility_num", "utility_den", "support_num", "valid_num", "null_sum", "row_count"):
            getattr(self, name).zero_()

    def forward(self, x: Tensor) -> Tensor:
        global _LAYERS_SEEN
        bsz, seqlen, dim = x.shape
        if seqlen != _TRAIN_SEQ_LEN:
            raise ValueError(f"persistent graph was configured for TRAIN_SEQ_LEN={_TRAIN_SEQ_LEN}, got {seqlen}")
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

        layer = int(_STATE["i"])
        _STATE["i"] = layer + 1
        _LAYERS_SEEN = max(_LAYERS_SEEN, layer + 1)
        if layer != self.layer_index:
            raise RuntimeError(f"graph layer mismatch: configured {self.layer_index}, executing {layer}")

        kv = _STATE["kv"]
        if layer == 0 or kv is None:
            kv = []
        assert isinstance(kv, list)
        kv.append((k, v.contiguous()))
        _STATE["kv"] = kv
        ks = [kk for kk, _ in kv]
        vs = [vv for _, vv in kv]

        idx, valid = graph_indices(self.wire_source_layer, self.wire_lag, bsz, seqlen)
        y, p, pn = xlayer_entmax_attention_stats(
            q, ks, vs, idx, valid, self.null_bias, self.head_dim**-0.5
        )

        if self.training:
            with torch.no_grad():
                num, den, sup, opp, null, rows = utility_components(p.detach(), pn.detach(), valid)
                self.utility_num.add_(num)
                self.utility_den.add_(den)
                self.support_num.add_(sup)
                self.valid_num.add_(opp)
                self.null_sum.add_(null)
                self.row_count.add_(rows)

        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


def _all_reduce_(tensors: Iterable[Tensor]) -> None:
    if dist.is_available() and dist.is_initialized():
        for tensor in tensors:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


@torch.no_grad()
def rewire_all_graphs() -> None:
    """Apply one utility decision after one accumulated optimizer step."""
    if not _MODULES:
        return
    if not _REWIRING_ACTIVE:
        # Warmup exists only to compile kernels.  Its graph evidence, traversal
        # cursor and topology must not leak into measured training.
        for module in _MODULES:
            module.reset_utility()
        return
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    for module in _MODULES:
        if not bool((module.utility_den > 0).any()):
            continue
        _all_reduce_(
            (module.utility_num, module.utility_den, module.support_num, module.valid_num, module.null_sum, module.row_count)
        )
        utility = module.utility_num / module.utility_den.clamp_min(1e-30)[:, None]
        old_src = module.wire_source_layer.clone()
        old_lag = module.wire_lag.clone()
        rewired = utility < UTILITY_THRESHOLD
        if rank == 0:
            result = eager_rewire_templates(
                module.wire_source_layer.cpu(),
                module.wire_lag.cpu(),
                utility.cpu(),
                module.wire_cursor.cpu(),
                module.wire_epoch.cpu(),
                module.wire_traversed.cpu(),
                layer=module.layer_index,
                seqlen=_TRAIN_SEQ_LEN,
                threshold=UTILITY_THRESHOLD,
                same_layer=_GRAPH_MODE == "same_layer",
            )
            src, lag, cursor, epoch, traversed, _ = result
            module.wire_source_layer.copy_(src.to(module.wire_source_layer.device))
            module.wire_lag.copy_(lag.to(module.wire_lag.device))
            module.wire_cursor.copy_(cursor.to(module.wire_cursor.device))
            module.wire_epoch.copy_(epoch.to(module.wire_epoch.device))
            module.wire_traversed.copy_(traversed.to(module.wire_traversed.device))
        if distributed:
            for tensor in (
                module.wire_source_layer, module.wire_lag, module.wire_cursor,
                module.wire_epoch, module.wire_traversed,
            ):
                dist.broadcast(tensor, src=0)

        if _GRAPH_STATS is not None:
            li = module.layer_index
            earlier = old_src < li
            mass_total = module.utility_num.sum().clamp_min(1e-30)
            universe = float(_TRAIN_SEQ_LEN if _GRAPH_MODE == "same_layer" else (li + 1) * _TRAIN_SEQ_LEN)
            _GRAPH_STATS[li, 0] = rewired.float().mean().to(_GRAPH_STATS.device)
            _GRAPH_STATS[li, 1] = utility.mean().to(_GRAPH_STATS.device)
            _GRAPH_STATS[li, 2] = (module.support_num.sum() / module.valid_num.sum().clamp_min(1.0)).to(_GRAPH_STATS.device)
            _GRAPH_STATS[li, 3] = (module.null_sum.sum() / module.row_count.sum().clamp_min(1.0)).to(_GRAPH_STATS.device)
            _GRAPH_STATS[li, 4] = earlier.float().mean().to(_GRAPH_STATS.device)
            _GRAPH_STATS[li, 5] = (module.utility_num[earlier].sum() / mass_total).to(_GRAPH_STATS.device)
            _GRAPH_STATS[li, 6] = (module.wire_traversed.float().clamp_max(universe).mean() / universe).to(_GRAPH_STATS.device)
            changed = (old_src != module.wire_source_layer) | (old_lag != module.wire_lag)
            _GRAPH_STATS[li, 7] = changed.float().mean()
        module.reset_utility()


def _wrap_gpt_init(orig_init):
    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _MODULES.clear()
        for layer, block in enumerate(self.blocks):
            attn = block.attn
            if not isinstance(attn, PersistentXLayerAttention):
                raise TypeError("GPT block was not constructed with persistent sparse attention")
            attn.configure_graph(layer, _TRAIN_SEQ_LEN)
            _MODULES.append(attn)
    return init


def _wrap_gpt_forward(orig_forward):
    def forward(self, *args, **kwargs):
        _STATE["i"] = 0
        _STATE["kv"] = None
        return orig_forward(self, *args, **kwargs)
    return forward


def _wrap_load_state_dict(orig_load):
    def load_state_dict(self, *args, **kwargs):
        global _REWIRING_ACTIVE
        result = orig_load(self, *args, **kwargs)
        # Nonpersistent accumulators are intentionally absent from snapshots;
        # clear warmup/final-eval residue whenever baseline restores a model.
        for module in _MODULES:
            module.reset_utility()
        if _GRAPH_STATS is not None:
            _GRAPH_STATS.zero_()
        # The baseline's first restore is the warmup rollback.  Enabling here
        # means the first measured optimizer step gets the first real rewire.
        _REWIRING_ACTIVE = True
        return result
    return load_state_dict


def _wrap_muon_step(orig_step):
    def step(self, *args, **kwargs):
        result = orig_step(self, *args, **kwargs)
        rewire_all_graphs()
        return result
    return step


def _wrap_eval_val(orig_eval_val):
    is_master = int(os.environ.get("RANK", "0")) == 0
    def eval_val(*args, **kwargs):
        result = orig_eval_val(*args, **kwargs)
        if is_master and _GRAPH_STATS is not None and _LAYERS_SEEN:
            names = (
                "graph_rewire", "graph_utility", "graph_support", "graph_null",
                "graph_xsrc", "graph_xmass", "graph_coverage", "graph_changed",
            )
            rows = _GRAPH_STATS[:_LAYERS_SEEN].tolist()
            parts = [
                f"{name}_l{layer}:{row[column]:.4f}"
                for layer, row in enumerate(rows)
                for column, name in enumerate(names)
            ]
            print("graph_stats " + " ".join(parts), flush=True)
        return result
    return eval_val


def main() -> None:
    global _GRAPH_STATS
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _GRAPH_STATS = torch.zeros(MAX_LAYERS, 8, device=f"cuda:{local_rank}")
    original_attention = baseline.CausalSelfAttention
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    original_int8_patterns = baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    original_gpt_init = baseline.GPT.__init__
    original_gpt_forward = baseline.GPT.forward
    original_load_state_dict = baseline.GPT.load_state_dict
    original_muon_step = baseline.Muon.step
    original_eval_val = baseline.eval_val
    baseline.CausalSelfAttention = PersistentXLayerAttention
    baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns + ("null_bias",)
    baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns + ("null_bias",)
    baseline.GPT.__init__ = _wrap_gpt_init(original_gpt_init)
    baseline.GPT.forward = _wrap_gpt_forward(original_gpt_forward)
    baseline.GPT.load_state_dict = _wrap_load_state_dict(original_load_state_dict)
    baseline.Muon.step = _wrap_muon_step(original_muon_step)
    baseline.eval_val = _wrap_eval_val(original_eval_val)
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
        baseline.eval_val = original_eval_val


if __name__ == "__main__":
    main()

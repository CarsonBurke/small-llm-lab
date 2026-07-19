"""Learned persistent cross-layer sparse graph: v2 of the no-selector arm.

v1 (``sparse_persistent_xlayer_train_gpt``) kept a discrete wire table alive
by a utility threshold and random traversal.  Red-teaming showed that scheme
is structurally self-defeating: entmax-1.5 puts exact zeros on most slots, so
"replace every wire below uniform share" churns ~90% of the graph every step,
new wires arrive gradient-blind, and the validity-blind denominator kills
long lags a priori.  v2 removes the heuristic layer entirely and makes the
graph itself a differentiable object:

* Every (layer, head) owns graph logits over the (source_layer, lag)
  universe: ``score(s, lag) = wire_logit_src[h, s] + wire_logit_lag[h, lag]
  + sum_r wire_logit_u[h, s, r] * wire_logit_v[h, r, lag]``.  The low-rank
  interaction (rank ``GRAPH_RANK``, U zero-init so the additive prior is
  exact at step 0) lets different source layers learn different lag
  profiles - "layer 0 diffuse at long lags, self layer sharp and local" is
  expressible.  All tables are real fp32 parameters in the scalar Adam
  group (CONTROL/INT8 name patterns), initialized to a smooth local-decay
  prior instead of a hand-built slot list.
* Every optimizer step samples K wires per (layer, head) WITHOUT
  replacement by Gumbel-top-K over those logits (Plackett-Luce) at a
  CONSTANT temperature ``GUMBEL_TAU``.  No anneal, no hold window: the
  noise scale is stationary and annealing is left to learning itself -
  as gradients grow the logit gaps past the noise scale, confident wires
  become sticky and exploration concentrates on the undecided margin.
  The noise is a deterministic 31-bit hash of (layer, head, candidate,
  step // GRAPH_HOLD_STEPS), so every rank draws the same graph, replay
  is exact, and no collectives or CPU syncs exist anywhere
  (``GRAPH_HOLD_STEPS`` defaults to 1; it exists only as an ablation
  knob).  Every eval runs the deterministic argmax graph.
* Two exact gradient paths feed the same tables through the fused kernel
  (``sparse_wire_bias_kernel``).  The selected wires' logits enter the
  entmax scores as a bias - a marginal, redistribution signal that pulls
  cold-but-useful wires into the entmax support (no gradient-blind
  exploration).  The same logits also ride a straight-through contribution
  gate (forward-identity ``1 + theta - sg(theta)`` multiplying each wire's
  p_e * v_e term), whose gradient sum_rows p_e * (dy . v_e) measures gross
  usefulness and does NOT vanish once a wire's mass is calibrated - so
  well-tuned wires keep earning their top-K rank instead of drifting out.
  No utility statistic, no replacement rule, no protection anywhere.

K/V gathering is unchanged from v1: idx = source*T + (t - lag), valid at
lag <= t, per-layer K/V lists through the transient-cat custom op.  The
learned tables ride the standard state_dict/int8 passthrough (fp32 exact;
a submission fork can bake the argmax indices instead, ~18KB).  This fork
does not modify ``train_gpt.py``.

Env knobs: SPARSE_K (128), SPARSE_NULL_INIT (0.0), GUMBEL_TAU (1.0),
GRAPH_HOLD_STEPS (1), GRAPH_RANK (4), GRAPH_LAG_DECAY (0.5),
GRAPH_SRC_PENALTY (0.5), GRAPH_BIAS_GAIN (1.0), TRAIN_SEQ_LEN.  Per-layer
stats reach tensorboard ``graph/*`` tags via the ``graph_stats``
val-cadence line.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import train_gpt as baseline
from sparse_persistent_xlayer_train_gpt import graph_indices
from sparse_wire_bias_kernel import xlayer_entmax_bias_attention_stats

MAX_LAYERS = 32
_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
_TRAIN_SEQ_LEN = int(os.environ.get("TRAIN_SEQ_LEN", "1024"))
_TAU = float(os.environ.get("GUMBEL_TAU", "1.0"))
_HOLD_STEPS = max(1, int(os.environ.get("GRAPH_HOLD_STEPS", "1")))

_STATE: dict = {"i": 0, "kv": None}
_STAT_NAMES = (
    "graph_rewire", "graph_theta", "graph_support", "graph_null",
    "graph_xsrc", "graph_xmass", "graph_theta_std",
)
_STATS: Tensor | None = None  # [MAX_LAYERS, 7] on device; columns match _STAT_NAMES
_LAYERS_SEEN = 0

if torch.cuda.is_available():
    # Optimizer-step clock: incremented eagerly after each Muon step, read as
    # a tensor inside the compiled forward (value changes never recompile).
    _STEP_BUF: Tensor = torch.zeros(1, dtype=torch.int64, device=f"cuda:{_LOCAL_RANK}")

_M31 = (1 << 31) - 1
_HASH_MUL = 0x45D9F3B


def _mix31(x: Tensor, seed: Tensor) -> Tensor:
    """lowbias32-style xorshift-multiply hash on 31-bit state (tensor seed)."""
    x = (x + (seed & _M31)) & _M31
    x = (((x >> 15) ^ x) * _HASH_MUL) & _M31
    x = (((x >> 15) ^ x) * _HASH_MUL) & _M31
    return (x >> 15) ^ x


def gumbel_noise(layer: int, num_heads: int, universe: int, step: Tensor) -> Tensor:
    """Deterministic [H, universe] Gumbel(0,1) draws keyed on (lane, step).

    Same graph on every rank at a given optimizer step, exact replay, and no
    RNG state consumed.  24-bit uniforms are centered off the lattice ends so
    log(-log(u)) is finite: the support is [-2.86, 16.6], plenty on both
    sides of the O(1) logit gaps that matter.
    """
    h = torch.arange(num_heads, device=step.device, dtype=torch.int64).view(-1, 1)
    c = torch.arange(universe, device=step.device, dtype=torch.int64).view(1, -1)
    lane = (layer * num_heads + h) * universe + c
    bits = _mix31(lane, step * 0x9E3779B9 + layer * 0x1000193) & 0xFFFFFF
    u = (bits.to(torch.float32) + 0.5) / float(1 << 24)
    return -torch.log(-torch.log(u))


class LearnedGraphAttention(baseline.CausalSelfAttention):
    k_budget = int(os.environ.get("SPARSE_K", "128"))
    null_init = float(os.environ.get("SPARSE_NULL_INIT", "0.0"))
    lag_decay = float(os.environ.get("GRAPH_LAG_DECAY", "0.5"))
    src_penalty = float(os.environ.get("GRAPH_SRC_PENALTY", "0.5"))
    bias_gain = float(os.environ.get("GRAPH_BIAS_GAIN", "1.0"))
    graph_rank = int(os.environ.get("GRAPH_RANK", "4"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.k_budget <= 0 or self.k_budget & (self.k_budget - 1):
            raise ValueError("SPARSE_K must be a positive power of two")
        self.null_bias = nn.Parameter(
            torch.full((self.num_heads,), self.null_init, dtype=torch.float32)
        )
        self.layer_index = -1

    def configure_graph(self, layer: int, seqlen: int) -> None:
        """Create the per-layer graph logit tables (called by GPT.__init__).

        Init is a smooth prior, not a slot list: current layer at zero
        penalty, earlier layers at -GRAPH_SRC_PENALTY, lags decaying as
        -GRAPH_LAG_DECAY * log1p(lag).  The argmax top-K of that surface is
        an interleaved local window across all live layers, and every lag in
        the universe stays within a few Gumbel units of selection at tau0.
        """
        if self.k_budget > (layer + 1) * seqlen:
            raise ValueError(
                f"SPARSE_K={self.k_budget} exceeds layer {layer}'s universe"
            )
        self.layer_index = layer
        src = torch.full((self.num_heads, layer + 1), -self.src_penalty)
        src[:, layer] = 0.0
        lag = -self.lag_decay * torch.log1p(
            torch.arange(seqlen, dtype=torch.float32)
        ).expand(self.num_heads, -1).clone()
        self.wire_logit_src = nn.Parameter(src)
        self.wire_logit_lag = nn.Parameter(lag)
        # Low-rank source-by-lag interaction.  U starts at zero so the initial
        # surface is exactly the additive prior; V starts small so U receives
        # gradient from step one (dU is proportional to V and vice versa).  A
        # dedicated generator keeps the model's RNG stream untouched.
        rank = self.graph_rank
        generator = torch.Generator().manual_seed(0x51AB1E + layer)
        self.wire_logit_u = nn.Parameter(torch.zeros(self.num_heads, layer + 1, rank))
        self.wire_logit_v = nn.Parameter(
            0.02 * torch.randn(self.num_heads, rank, seqlen, generator=generator)
        )
        # Previous step's sampled graph, for turnover diagnostics only.
        self.register_buffer(
            "prev_wires", torch.full((self.num_heads, self.k_budget), -1, dtype=torch.int64),
            persistent=False,
        )

    def _logit_surface(self) -> Tensor:
        """Full [H, S, T] graph logit surface: additive prior + low-rank."""
        return (
            self.wire_logit_src[:, :, None]
            + self.wire_logit_lag[:, None, :]
            + torch.bmm(self.wire_logit_u, self.wire_logit_v)
        )

    def _sample_wires(self, seqlen: int) -> tuple[Tensor, Tensor]:
        """Gumbel-top-K (train) or argmax top-K (eval) over the graph logits.

        The temperature is a constant: no anneal schedule and (by default)
        no hold window.  Annealing is left to learning - once the gradient
        grows a wire's logit gap past the noise scale it is selected every
        step, so exploration concentrates on the undecided margin.
        """
        universe = (self.layer_index + 1) * seqlen
        scores = self._logit_surface().detach().reshape(self.num_heads, universe)
        if self.training:
            epoch = _STEP_BUF // _HOLD_STEPS
            scores = scores + _TAU * gumbel_noise(
                self.layer_index, self.num_heads, universe, epoch
            )
        wires = scores.topk(self.k_budget, dim=-1).indices
        return (wires // seqlen).to(torch.int32), (wires % seqlen).to(torch.int32)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        if seqlen != _TRAIN_SEQ_LEN:
            raise ValueError(
                f"learned graph was configured for TRAIN_SEQ_LEN={_TRAIN_SEQ_LEN}, got {seqlen}"
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
        q = baseline.apply_rotary_emb(q, cos, sin)
        k = baseline.apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        layer = _STATE["i"]
        _STATE["i"] = layer + 1
        if layer != self.layer_index:
            raise RuntimeError(
                f"graph layer mismatch: configured {self.layer_index}, executing {layer}"
            )

        # Per-forward K/V stack; the list feeds the transient-cat custom op
        # (see sparse_xlayer_train_gpt for the retention rationale).
        kv = _STATE["kv"]
        if layer == 0 or kv is None:
            kv = []
        kv.append((k, v.contiguous()))
        _STATE["kv"] = kv
        ks = [kk for kk, _ in kv]
        vs = [vv for _, vv in kv]

        with torch.no_grad():
            src_idx, lag_idx = self._sample_wires(seqlen)
            idx, valid = graph_indices(src_idx, lag_idx, bsz, seqlen)

        # Differentiable gather: the LM gradient on both paths lands in the
        # logit tables via autograd scatter-add.  theta rides the kernel
        # twice - as the entmax score bias (marginal signal, pulls cold wires
        # into support) and as a straight-through contribution gate (gross
        # usefulness signal, keeps calibrated wires selected).
        head = torch.arange(self.num_heads, device=x.device).unsqueeze(1)
        src_long, lag_long = src_idx.to(torch.int64), lag_idx.to(torch.int64)
        theta = (
            self.wire_logit_src[head, src_long]
            + self.wire_logit_lag[head, lag_long]
            + (
                self.wire_logit_u[head, src_long]
                * self.wire_logit_v.transpose(1, 2)[head, lag_long]
            ).sum(-1)
        )
        wire_bias = self.bias_gain * theta
        # (theta - sg(theta)) is bitwise zero, so the gate is EXACTLY 1.0 in
        # forward value while carrying d(gate)/d(theta) = 1 for the ST path.
        wire_gate = (theta - theta.detach()) + 1.0
        y, p, pn = xlayer_entmax_bias_attention_stats(
            q, ks, vs, idx, valid, wire_bias, wire_gate, self.null_bias,
            self.head_dim**-0.5,
        )

        if self.training and _STATS is not None:
            with torch.no_grad():
                li = min(layer, MAX_LAYERS - 1)
                wires = src_idx.to(torch.int64) * seqlen + lag_idx.to(torch.int64)
                overlap = (wires[:, :, None] == self.prev_wires[:, None, :]).any(-1)
                earlier = src_idx < layer
                mass = p.sum(dim=(0, 2))
                mass_total = mass.sum().clamp_min(1e-30)
                _STATS[li, 0] = 1.0 - overlap.float().mean()
                _STATS[li, 1] = theta.detach().mean()
                _STATS[li, 2] = ((p > 0) & valid).sum() / valid.sum().clamp_min(1)
                _STATS[li, 3] = pn.mean()
                earlier_f = earlier.float()
                _STATS[li, 4] = earlier_f.mean()
                # Masked product, not mass[earlier]: bool-mask indexing has a
                # data-dependent shape, which not every dynamo version will
                # capture under fullgraph=True.
                _STATS[li, 5] = (mass * earlier_f).sum() / mass_total
                # Spread of the selected wires' logits: once it grows past
                # the (constant) noise scale the graph has self-annealed.
                _STATS[li, 6] = theta.detach().std()
                self.prev_wires.copy_(wires)

        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


def _wrap_gpt_init(orig_init):
    def init(self, *args, **kwargs):
        global _LAYERS_SEEN
        orig_init(self, *args, **kwargs)
        for layer, block in enumerate(self.blocks):
            attn = block.attn
            if not isinstance(attn, LearnedGraphAttention):
                raise TypeError("GPT block was not constructed with learned-graph attention")
            attn.configure_graph(layer, _TRAIN_SEQ_LEN)
        # Set statically here: reading (and guarding on) a mutating Python
        # global inside the compiled forward costs a needless recompile.
        _LAYERS_SEEN = len(self.blocks)
    return init


def _wrap_gpt_forward(orig_forward):
    def forward(self, *args, **kwargs):
        _STATE["i"] = 0
        _STATE["kv"] = None
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
        # its rollback keeps the noise stream identical to a run without
        # warmup.  (The final int8 reload also lands here - by then only
        # eval runs, which never reads the clock.)
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
    original_attention = baseline.CausalSelfAttention
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    original_int8_patterns = baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    original_gpt_init = baseline.GPT.__init__
    original_gpt_forward = baseline.GPT.forward
    original_load_state_dict = baseline.GPT.load_state_dict
    original_muon_step = baseline.Muon.step
    original_eval_val = baseline.eval_val
    baseline.CausalSelfAttention = LearnedGraphAttention
    baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns + ("null_bias", "wire_logit")
    baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns + ("null_bias", "wire_logit")
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

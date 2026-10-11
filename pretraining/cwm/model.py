"""Causal Workspace Machine (CWM): exact token-sequential feedback LM.

Design and hypotheses: ``research_notes/Optimal causal model communication/
cwm_plan.md``. Summary of the implemented ``cwm_v1`` contract:

* A parallel *perception* layer (causal softmax attention, or KDA for the
  KDA-only variant) maps tokens to ``u_t``. It never sees feedback.
* A sequential *core* of ``R`` untied rounds runs per token. Event ``(t, r)``
  computes its input state ``x_{t,r}`` (workspace after an addressed depth
  read over this token's earlier rounds), proposes writes from it, then reads
  memory and applies an MLP. After round ``R`` the final state ``h_t`` scores
  token ``t + 1`` and proposes the writes of source ``R + 1`` (``out``).
* Memory has one source per round plus ``out`` (``S = R + 1`` sources):

  - associative banks ``M_s`` (single writer each): delta rule with
    per-key-channel decay and write strength ``beta in (0, beta_max)``;
    ``beta_max = 2`` permits negative transition eigenvalues. The state
    holds ``bank_dtype`` values (fp32 or bf16): commits and reads compute in
    fp32 and a commit rounds its result once, with a straight-through
    gradient (here the state gradient stays fp32; the fused executor
    stores it in ``bank_dtype`` too);
  - an exact windowed store of key/value entries (``store=True`` only), one
    entry per source per token, RoPE on the token index, a learned per-source
    key tag, and the last ``store_window`` tokens kept.

* Commit-once: every write proposed while processing token ``t`` is applied
  after token ``t`` is scored, so it is visible from token ``t + 1`` on and
  never to token ``t`` itself.
* Legality: round ``r`` may always read sources ``s <= r`` (what a parallel
  lattice can compute: inputs of rounds at or below ``r`` at earlier tokens).
  Sources ``s > r`` -- higher rounds and ``out`` at earlier tokens -- are the
  *feedback* edges. They are read by a separate softmax/normalization path
  whose output projection starts at zero, so ``read_mask="full"`` and
  ``read_mask="lattice"`` compute bitwise-identical functions at
  initialization and differ only by learned use of feedback.
* ``read_mask="placebo"`` keeps every feedback path and parameter of
  ``full`` but serves each row's feedback reads from another row's memory
  (row ``b`` reads row ``b - B/2``, cyclically): equal capacity and gradient
  plumbing, no information about the row's own sequence. Rows of a batch are
  consecutive slices of one token stream, so the shift is half the batch
  rather than one row, whose text directly precedes the reader's. Full beating
  placebo is the evidence that feedback, not extra capacity, helps.
* Feedback-only parameters are frozen under ``lattice`` (they never enter
  its graph), so optimizers only see what the arm can train.

* Chunk lag (``lag``, NOTES.md 2026-10-01). ``"none"`` is the exact
  token-sequential contract above. Otherwise the sequence is cut into
  ``lag_chunk``-token chunks and every feedback read (banks and store) sees
  memory as of its chunk's start: the banks committed before the chunk and
  the store entries of earlier tokens. Under ``"feedback"`` legal reads stay
  exact, so the lattice arm is unchanged; ``"all"`` serves legal bank reads
  from the chunk-start state too, leaving the chunk's own tokens to the
  legal store reads (exact, and the window covers a chunk) and the
  residual stream. Either way every token is still committed exactly, and a
  chunk can be computed round by round in parallel over its tokens
  (``parallel.py``).

Training runs the exact recurrence (no Jacobi passes): the core advances all
rows in lock-step one token at a time, with checkpointing every ``chunk``
tokens so backpropagation through the whole window fits in memory.

This module is the readable definition (and the CPU path). Training and
evaluation run the same function through ``fused.FusedCore`` (hand-written
backward, Triton kernels, CUDA graphs via ``graphed.GraphedCore``), which
``pretraining/tests/test_cwm_fused.py`` holds to this module's values and
gradients.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from pretraining.nanogpt_mini.nanogpt_mini_kda_model import (
    MLP,
    BiasFreeLinear,
    CausalSelfAttention,
    KimiDeltaAttention,
    Linear,
    RMSNorm,
)

ARCHITECTURE = "cwm_v1"
DECAY_LOWER_BOUND = -5.0
SOFTCAP = 15.0
EPS = torch.finfo(torch.float32).eps  # every RMS normalization


@dataclass(frozen=True)
class CWMConfig:
    vocab_size: int = 50304
    dim: int = 512
    mlp_hidden: int = 2048
    rounds: int = 5
    perception: str = "attention"  # "attention" | "kda"
    store: bool = True  # exact windowed store; False = banks only
    read_mask: str = "full"  # "full" | "lattice"
    bank_heads: int = 4
    bank_dim: int = 64
    beta_max: float = 2.0
    store_heads: int = 4
    store_dim: int = 32
    store_window: int = 64
    depth_heads: int = 2
    depth_dim: int = 64
    low_rank: int = 64
    # Storage of the bank state (and, in the fused executor, of its
    # gradient); commits and reads compute in fp32. bf16 halves the state
    # traffic that bounds the core's speed; on a trained model it moves the
    # predictions less than the fused core's own rounding does
    # (NOTES.md 2026-10-01).
    bank_dtype: str = "float32"  # "float32" | "bfloat16"
    # Chunk-lagged reads (module docstring): "none" | "feedback" | "all".
    lag: str = "none"
    lag_chunk: int = 16

    def __post_init__(self):
        if self.perception not in ("attention", "kda"):
            raise ValueError(f"perception must be 'attention' or 'kda', got {self.perception!r}")
        if self.read_mask not in ("full", "lattice", "placebo"):
            raise ValueError(f"read_mask must be 'full', 'lattice' or 'placebo', got {self.read_mask!r}")
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
        if not 0 < self.beta_max <= 2:
            raise ValueError("beta_max must lie in (0, 2]")
        if self.store_dim % 4 or self.depth_dim % 2:
            raise ValueError("store_dim must be a multiple of 4 (half-truncated RoPE)")
        if self.bank_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"bank_dtype must be 'float32' or 'bfloat16', got {self.bank_dtype!r}")
        if self.lag not in ("none", "feedback", "all"):
            raise ValueError(f"lag must be 'none', 'feedback' or 'all', got {self.lag!r}")
        if self.lag_chunk < 1:
            raise ValueError("lag_chunk must be positive")
        if self.lag != "none" and self.bank_dtype != "float32":
            # The chunk-parallel core holds the state only at chunk starts,
            # so per-token rounding has no counterpart there.
            raise ValueError("a lagged core keeps fp32 banks")

    @property
    def sources(self) -> int:
        return self.rounds + 1

    @property
    def bank_torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.bank_dtype)

    @property
    def store_scale(self) -> float:
        # The baseline scores RMS-normed 128-wide heads with 0.12, a logit
        # range of +-15.4; keep that range at the store's head width.
        return 0.12 * 128 / self.store_dim


def rms(x: Tensor) -> Tensor:
    """RMS-normalize the last dim in fp32 (fp32 epsilon whatever the input
    dtype: ``F.rms_norm`` on bf16 would use bf16's 2^-7), rounded back."""
    return F.rms_norm(x.float(), (x.size(-1),), eps=EPS).type_as(x)


def l2norm(x: Tensor) -> Tensor:
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)


def rope_frequencies(dim: int) -> Tensor:
    """Half-truncated RoPE frequencies (the baseline's ``Rotary``)."""
    freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
    return torch.cat([freq, freq.new_zeros(dim // 4)])


def rope(x: Tensor, pos: Tensor, freq: Tensor) -> Tensor:
    """Rotate ``x[..., D]`` at float positions ``pos`` (broadcast over heads)."""
    theta = pos[..., None, None] * freq
    cos, sin = theta.cos(), theta.sin()
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1).type_as(x)


def sink_attention(q: Tensor, k: Tensor, v: Tensor, bias: Tensor | None, scale: float) -> Tensor:
    """Single-query attention with a zero-key, zero-value sink (fp32).

    ``q [B, H, D]``, ``k/v [B, N, H, D]``, ``bias`` additive ``[N]`` (0 or
    -inf). The sink makes an empty or fully masked set well defined (output
    zero) and lets a reader decline to read.
    """
    scores = torch.einsum("bhd,bnhd->bhn", q.float(), k.float()) * scale
    if bias is not None:
        scores = scores + bias
    scores = torch.cat([scores, scores.new_zeros(scores.shape[:-1] + (1,))], dim=-1)
    weights = scores.softmax(-1)[..., :-1]
    return torch.einsum("bhn,bnhd->bhd", weights, v.float()).type_as(v)


def placebo_shift(rows: int) -> int:
    if rows < 2:
        raise ValueError("the placebo needs at least two rows to read across")
    return rows // 2


def gated_head_norm(o: Tensor, gate: Tensor, weight: Tensor) -> Tensor:
    return rms(o.float()) * weight.float() * torch.sigmoid(gate.float())


@dataclass
class Writes:
    """One source's proposed writes for one token (all ``[B, ...]``)."""

    key: Tensor  # [B, H, Dk] fp32, unit norm
    value: Tensor  # [B, H, Dv] fp32
    beta: Tensor  # [B, H] fp32
    log_decay: Tensor  # [B, H, Dk] fp32, in [DECAY_LOWER_BOUND, 0]
    store_key: Tensor | None  # [B, Hs, Ds], RoPE'd at the token's position, tagged
    store_value: Tensor | None


class Writer(nn.Module):
    """Write projections of one source (one per round, plus ``out``)."""

    def __init__(self, cfg: CWMConfig):
        super().__init__()
        d, H, Dk, Dv = cfg.dim, cfg.bank_heads, cfg.bank_dim, cfg.bank_dim
        self.cfg = cfg
        self.bank_k = BiasFreeLinear(d, H * Dk)
        self.bank_v = BiasFreeLinear(d, H * Dv)
        self.bank_beta = Linear(d, H)
        self.decay_a = BiasFreeLinear(d, cfg.low_rank)
        self.decay_b = BiasFreeLinear(cfg.low_rank, H * Dk)
        self.A_log = nn.Parameter(torch.zeros(H))
        self.dt_bias = nn.Parameter(torch.zeros(H * Dk))
        if cfg.store:
            hs = cfg.store_heads * cfg.store_dim
            self.store_k = Linear(d, hs)
            self.store_v = Linear(d, hs)
            self.store_tag = nn.Parameter(torch.zeros(cfg.store_heads, cfg.store_dim))

    def forward(self, xn: Tensor, pos: Tensor, freq: Tensor) -> Writes:
        cfg = self.cfg
        B, H, Dk = xn.size(0), cfg.bank_heads, cfg.bank_dim
        key = l2norm(self.bank_k(xn).view(B, H, Dk).float())
        value = self.bank_v(xn).view(B, H, Dk).float()
        beta = cfg.beta_max * torch.sigmoid(self.bank_beta(xn).float())
        decay_logits = self.decay_b(self.decay_a(xn)).view(B, H, Dk).float()
        biased = decay_logits + self.dt_bias.float().view(H, Dk)
        log_decay = DECAY_LOWER_BOUND * torch.sigmoid(self.A_log.float().exp()[:, None] * biased)
        store_key = store_value = None
        if cfg.store:
            Hs, Ds = cfg.store_heads, cfg.store_dim
            # The source tag is added before the rotation, so its score
            # contribution stays a function of the relative offset only.
            store_key = rope(rms(self.store_k(xn).view(B, Hs, Ds)) + self.store_tag.type_as(xn), pos, freq)
            store_value = self.store_v(xn).view(B, Hs, Ds)
        return Writes(key, value, beta, log_decay, store_key, store_value)


class Round(nn.Module):
    """Round block ``index`` (1-based) of the core."""

    def __init__(self, cfg: CWMConfig, index: int):
        super().__init__()
        d, S = cfg.dim, cfg.sources
        H, Dv = cfg.bank_heads, cfg.bank_dim
        hd = cfg.depth_heads * cfg.depth_dim
        self.cfg, self.index = cfg, index
        # Addressed depth read over [u_t, h_{t,1}, ..., h_{t,index-1}].
        self.depth_norm = RMSNorm(d)
        self.depth_q = Linear(d, hd)
        self.depth_k = Linear(d, hd)
        self.depth_v = Linear(d, hd)
        self.depth_tag = nn.Parameter(torch.zeros(cfg.rounds, hd))
        self.depth_o = Linear(hd, d)
        # Input state for writes and memory queries.
        self.in_norm = RMSNorm(d)
        self.writer = Writer(cfg)
        # Bank reads: lattice-legal sources 1..index, feedback index+1..S.
        self.bank_q = BiasFreeLinear(d, H * cfg.bank_dim)
        # Per-head source mixes; a single source needs none (its softmax is 1).
        self.bank_mix_legal = Linear(d, H * index) if index > 1 else None
        self.bank_mix_fb = Linear(d, H * (S - index)) if S - index > 1 else None
        self.bank_gate_a = BiasFreeLinear(d, cfg.low_rank)
        self.bank_gate_b = BiasFreeLinear(cfg.low_rank, H * Dv)
        self.bank_norm_legal = nn.Parameter(torch.ones(Dv))
        self.bank_norm_fb = nn.Parameter(torch.ones(Dv))
        self.bank_o_legal = Linear(H * Dv, d)
        self.bank_o_fb = Linear(H * Dv, d)
        if cfg.store:
            hs = cfg.store_heads * cfg.store_dim
            self.store_q = Linear(d, hs)
            self.store_o_legal = Linear(hs, d)
            self.store_o_fb = Linear(hs, d)
        self.mlp_norm = RMSNorm(d)
        self.mlp = MLP(d, cfg.mlp_hidden)

    def depth_read(self, h: Tensor, log: Tensor) -> Tensor:
        """``log [B, index, d]``: this token's earlier round outputs."""
        cfg = self.cfg
        B, n = log.size(0), log.size(1)
        Hd, Dd = cfg.depth_heads, cfg.depth_dim
        q = rms(self.depth_q(self.depth_norm(h)).view(B, Hd, Dd))
        entries = self.depth_norm(log)
        k = self.depth_k(entries) + self.depth_tag[:n].type_as(entries)
        k = rms(k.view(B, n, Hd, Dd))
        v = self.depth_v(entries).view(B, n, Hd, Dd)
        o = sink_attention(q, k, v, None, 0.12 * 128 / Dd)
        return self.depth_o(o.reshape(B, Hd * Dd))

    def bank_query(self, xn: Tensor) -> Tensor:
        """Unit-norm bank query ``[rows, H, D]`` (fp32), scaled by D^-1/2."""
        H, D = self.cfg.bank_heads, self.cfg.bank_dim
        return l2norm(self.bank_q(xn).view(xn.size(0), H, D).float()) * D**-0.5

    def read_banks(self, xn: Tensor, legal_banks: Tensor, fb_banks: Tensor, feedback: str | None) -> Tensor:
        """``*_banks [S, B, H, Dv, Dk]`` fp32: the state legal and feedback
        reads see (the committed state before this token, or with ``lag``
        the chunk-start state). ``feedback``: None (legal sources only),
        "full" or "placebo"."""
        r = self.index
        q = self.bank_query(xn)
        with torch.autocast(xn.device.type, enabled=False):
            legal = torch.einsum("sbhvk,bhk->sbhv", legal_banks[:r], q)
            fb = None
            if feedback == "placebo":
                # out[b] = M[b - B/2] q[b], without copying the banks.
                shift = placebo_shift(xn.size(0))
                fb = torch.einsum("sbhvk,bhk->sbhv", fb_banks[r:], q.roll(-shift, 0)).roll(shift, 1)
            elif feedback:
                fb = torch.einsum("sbhvk,bhk->sbhv", fb_banks[r:], q)
        return self.bank_output(xn, legal, fb)

    def bank_output(self, xn: Tensor, legal: Tensor, fb: Tensor | None) -> Tensor:
        """Mix, normalize, gate and project the per-source bank reads
        ``[sources, rows, H, Dv]`` (fp32) of legal and (unless None)
        feedback sources."""
        B, H, D = xn.size(0), self.cfg.bank_heads, self.cfg.bank_dim
        gate = self.bank_gate_b(self.bank_gate_a(xn)).view(B, H, D)

        def mixed(reads: Tensor, mix: Linear | None) -> Tensor:
            if mix is None:
                return reads[0]
            weights = mix(xn).view(B, H, reads.size(0)).float().softmax(-1)
            return torch.einsum("sbhv,bhs->bhv", reads, weights)

        out = self.bank_o_legal(gated_head_norm(mixed(legal, self.bank_mix_legal), gate, self.bank_norm_legal).type_as(xn).reshape(B, H * D))
        if fb is not None:
            fb = gated_head_norm(mixed(fb, self.bank_mix_fb), gate, self.bank_norm_fb)
            out = out + self.bank_o_fb(fb.type_as(xn).reshape(B, H * D))
        return out

    def store_query(self, xn: Tensor, pos: Tensor, freq: Tensor) -> Tensor:
        Hs, Ds = self.cfg.store_heads, self.cfg.store_dim
        return rope(rms(self.store_q(xn).view(xn.size(0), Hs, Ds)), pos, freq)

    def read_store(
        self,
        xn: Tensor,
        keys: Tensor,
        values: Tensor,
        valid_bias: Tensor,
        fb_bias: Tensor,
        pos: Tensor,
        freq: Tensor,
        feedback: str | None,
    ) -> Tensor:
        """``keys/values [B, W, S, Hs, Ds]``: the last W tokens' entries;
        ``valid_bias``/``fb_bias [W]`` mask the slots legal/feedback reads
        may not see."""
        cfg, r = self.cfg, self.index
        B, W = xn.size(0), cfg.store_window
        Hs, Ds = cfg.store_heads, cfg.store_dim
        q = self.store_query(xn, pos, freq)

        def path(lo: int, hi: int, query: Tensor, slot_bias: Tensor) -> Tensor:
            n = hi - lo
            k = keys[:, :, lo:hi].reshape(B, W * n, Hs, Ds)
            v = values[:, :, lo:hi].reshape(B, W * n, Hs, Ds)
            bias = slot_bias[:, None].expand(W, n).reshape(W * n)
            return sink_attention(query, k, v, bias, cfg.store_scale).reshape(B, Hs * Ds)

        out = self.store_o_legal(path(0, r, q, valid_bias))
        if feedback == "placebo":
            shift = placebo_shift(B)
            out = out + self.store_o_fb(path(r, cfg.sources, q.roll(-shift, 0), fb_bias).roll(shift, 0))
        elif feedback:
            out = out + self.store_o_fb(path(r, cfg.sources, q, fb_bias))
        return out


class CWM(nn.Module):
    def __init__(self, cfg: CWMConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, d).bfloat16()
        self.norm1 = RMSNorm(d)
        self.p_norm1 = RMSNorm(d)
        if cfg.perception == "attention":
            self.p_mixer = CausalSelfAttention(d)
        else:
            self.p_mixer = KimiDeltaAttention(d)
        self.p_norm2 = RMSNorm(d)
        self.p_mlp = MLP(d, cfg.mlp_hidden)
        self.rounds = nn.ModuleList(Round(cfg, r) for r in range(1, cfg.rounds + 1))
        self.out_norm = RMSNorm(d)
        self.out_writer = Writer(cfg)
        self.norm2 = RMSNorm(d)
        self.head = Linear(d, cfg.vocab_size)
        self.register_buffer("store_freq", rope_frequencies(cfg.store_dim), persistent=False)
        self.register_buffer("window_index", torch.arange(cfg.store_window), persistent=False)
        # Evaluation-only switch: False removes every feedback read (a causal
        # ablation of the trained channel).
        self.feedback_enabled = True
        if cfg.read_mask == "lattice":
            for name, p in self.named_parameters():
                if is_feedback_parameter(name):
                    p.requires_grad_(False)

    @property
    def feedback(self) -> str | None:
        if self.cfg.read_mask == "lattice" or not self.feedback_enabled:
            return None
        return self.cfg.read_mask

    # ------------------------------------------------------------------
    # Perception (parallel)
    # ------------------------------------------------------------------
    def perceive(self, inputs: Tensor) -> Tensor:
        x = self.norm1(self.embed(inputs))
        mixed = self.p_mixer(self.p_norm1(x))
        if isinstance(mixed, tuple):
            mixed = mixed[0]
        x = x + mixed
        return x + self.p_mlp(self.p_norm2(x))

    # ------------------------------------------------------------------
    # Core (sequential)
    # ------------------------------------------------------------------
    def initial_state(self, batch: int, device, dtype) -> tuple[Tensor, Tensor | None, Tensor | None]:
        cfg = self.cfg
        banks = torch.zeros(cfg.sources, batch, cfg.bank_heads, cfg.bank_dim, cfg.bank_dim, device=device)
        if not cfg.store:
            return banks, None, None
        shape = (batch, cfg.store_window, cfg.sources, cfg.store_heads, cfg.store_dim)
        return banks, torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)

    def step(
        self, banks: Tensor, lag_banks: Tensor, keys: Tensor | None, values: Tensor | None, u: Tensor, pos: int
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Process token ``pos`` for every row. ``lag_banks``: the state at
        the start of the token's lag chunk (read only under ``lag``)."""
        cfg, freq = self.cfg, self.store_freq
        feedback = self.feedback
        posf = torch.tensor(float(pos), device=u.device)
        legal_banks = lag_banks if cfg.lag == "all" else banks
        fb_banks = banks if cfg.lag == "none" else lag_banks
        valid_bias = fb_bias = None
        if cfg.store:
            # Slot w holds token pos - W + w.
            W = cfg.store_window
            valid_bias = torch.where(self.window_index >= W - pos, 0.0, float("-inf"))
            # Under lag, feedback sees no token of its own chunk.
            offset = 0 if cfg.lag == "none" else pos % cfg.lag_chunk
            fb_bias = torch.where(self.window_index >= W - offset, float("-inf"), valid_bias)
        h = u
        log = [u]
        writes = []
        for blk in self.rounds:
            h = h + blk.depth_read(h, torch.stack(log, dim=1))
            xn = blk.in_norm(h)
            writes.append(blk.writer(xn, posf, freq))
            h = h + blk.read_banks(xn, legal_banks, fb_banks, feedback)
            if cfg.store:
                h = h + blk.read_store(xn, keys, values, valid_bias, fb_bias, posf, freq, feedback)
            h = h + blk.mlp(blk.mlp_norm(h))
            log.append(h)
        writes.append(self.out_writer(self.out_norm(h), posf, freq))
        banks = commit_banks(banks, writes, cfg.bank_torch_dtype)
        if cfg.store:
            new_keys = torch.stack([w.store_key for w in writes], dim=1).type_as(keys)
            new_values = torch.stack([w.store_value for w in writes], dim=1).type_as(values)
            keys = torch.cat([keys[:, 1:], new_keys[:, None]], dim=1)
            values = torch.cat([values[:, 1:], new_values[:, None]], dim=1)
        return h, banks, keys, values

    def chunk_forward(
        self, banks: Tensor, lag_banks: Tensor, keys: Tensor | None, values: Tensor | None, u: Tensor, start: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        outputs = []
        for i in range(u.size(1)):
            pos = start + i
            if pos % self.cfg.lag_chunk == 0:
                lag_banks = banks
            h, banks, keys, values = self.step(banks, lag_banks, keys, values, u[:, i], pos)
            outputs.append(h)
        return torch.stack(outputs, dim=1), banks, lag_banks, keys, values

    def core(self, u: Tensor, chunk: int, checkpointing: bool) -> Tensor:
        B, T, _ = u.shape
        if T % chunk:
            raise ValueError(f"sequence length {T} is not a multiple of chunk {chunk}")
        banks, keys, values = self.initial_state(B, u.device, u.dtype)
        lag_banks = banks
        outputs = []
        for c0 in range(0, T, chunk):
            args = (banks, lag_banks, keys, values, u[:, c0 : c0 + chunk], c0)
            if checkpointing:
                out = checkpoint(self.chunk_forward, *args, use_reentrant=False)
            else:
                out = self.chunk_forward(*args)
            h, banks, lag_banks, keys, values = out
            outputs.append(h)
        return torch.cat(outputs, dim=1)

    # ------------------------------------------------------------------
    # Head
    # ------------------------------------------------------------------
    def hidden(self, inputs: Tensor, chunk: int = 16, checkpointing: bool = False) -> Tensor:
        return self.norm2(self.core(self.perceive(inputs), chunk, checkpointing))

    def project_logits(self, hidden: Tensor) -> Tensor:
        logits = self.head(hidden).float()
        return SOFTCAP * logits * (logits.square() + SOFTCAP**2).rsqrt()


class _RoundTo(torch.autograd.Function):
    """``x`` rounded to ``dtype`` (kept in ``x``'s dtype); straight-through
    gradient."""

    @staticmethod
    def forward(ctx, x: Tensor, dtype: torch.dtype) -> Tensor:
        return x.to(dtype).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: Tensor):
        return grad, None


def commit_banks(banks: Tensor, writes: list[Writes], dtype: torch.dtype = torch.float32) -> Tensor:
    """Apply every source's delta-rule write once (fp32, value-major state),
    rounding the result to the values of the state dtype ``dtype``."""
    key = torch.stack([w.key for w in writes])  # [S, B, H, Dk]
    value = torch.stack([w.value for w in writes])
    beta = torch.stack([w.beta for w in writes])
    log_decay = torch.stack([w.log_decay for w in writes])
    with torch.autocast(banks.device.type, enabled=False):
        decayed = banks * log_decay.exp()[..., None, :]
        predicted = torch.einsum("sbhvk,sbhk->sbhv", decayed, key)
        new = decayed + beta[..., None, None] * (value - predicted)[..., :, None] * key[..., None, :]
        return new if dtype == torch.float32 else _RoundTo.apply(new, dtype)


# ----------------------------------------------------------------------
# Initialization and accounting
# ----------------------------------------------------------------------

ZERO_INIT_SUFFIXES = ("depth_o", "bank_o_legal", "bank_o_fb", "store_o_legal", "store_o_fb", "mlp.proj", "p_mixer.proj", "p_mixer.o_proj", "head")


@torch.no_grad()
def initialize(model: CWM, seed: int) -> None:
    """Baseline conventions: matrices N(0, 0.33 / fan_in), every output
    projection (and the head) zero, unit norm gains, N(0, 1) embedding.
    Initialization is defined by module names only, so ``read_mask`` variants
    built at the same seed are identical parameter for parameter."""
    cfg = model.cfg
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for p in model.parameters():
        p.fill_(float("nan"))
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.endswith(s) for s in ZERO_INIT_SUFFIXES):
                module.weight.zero_()
            else:
                std = (0.33 / module.weight.size(1)) ** 0.5
                module.weight.copy_(torch.randn(module.weight.shape, generator=generator) * std)
            if module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, RMSNorm):
            module.gains.fill_(1)
    model.embed.weight.copy_(torch.randn(model.embed.weight.shape, generator=generator))
    writers = [blk.writer for blk in model.rounds] + [model.out_writer]
    for writer in writers:
        writer.A_log.zero_()
        dt = torch.empty(writer.dt_bias.shape).uniform_(math.log(0.001), math.log(0.1), generator=generator).exp().clamp(min=1e-4)
        writer.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        # Initial write strength 0.5, as KDA's sigmoid beta starts.
        writer.bank_beta.bias.fill_(math.log(0.5 / (cfg.beta_max - 0.5)))
        if cfg.store:
            writer.store_tag.zero_()
    for blk in model.rounds:
        blk.depth_tag.zero_()
        blk.bank_norm_legal.fill_(1)
        blk.bank_norm_fb.fill_(1)
    mixer = model.p_mixer
    if isinstance(mixer, KimiDeltaAttention):
        for conv in (mixer.q_conv1d, mixer.k_conv1d, mixer.v_conv1d):
            conv.weight.zero_()
            conv.weight[:, 0, -1] = 1
        mixer.A_log.zero_()
        dt = torch.empty(mixer.dt_bias.shape).uniform_(math.log(0.001), math.log(0.1), generator=generator).exp().clamp(min=1e-4)
        mixer.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        mixer.o_norm.weight.fill_(1)
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise RuntimeError(f"uninitialized parameter {name}")


def is_feedback_parameter(name: str) -> bool:
    """Used only by feedback reads (dead under ``lattice``): the feedback read
    paths and the ``out`` source's writer."""
    return "_fb" in name or name.startswith(("out_writer.", "out_norm."))


def feedback_parameter_names(model: CWM) -> list[str]:
    return [n for n, _ in model.named_parameters() if is_feedback_parameter(n)]


def parameter_report(model: CWM) -> dict:
    total = sum(p.numel() for p in model.parameters())
    embedding = model.embed.weight.numel() + model.head.weight.numel() + model.head.bias.numel()
    names = set(feedback_parameter_names(model))
    feedback = sum(p.numel() for n, p in model.named_parameters() if n in names)
    return {
        "total": total,
        "non_embedding": total - embedding,
        "feedback_only": feedback,
        "active_non_embedding": total - embedding - (feedback if model.cfg.read_mask == "lattice" else 0),
    }


def config_dict(cfg: CWMConfig) -> dict:
    return dataclasses.asdict(cfg)

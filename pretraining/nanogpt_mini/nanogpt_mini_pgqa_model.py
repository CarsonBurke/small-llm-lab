"""pretraining/nanogpt_mini/nanogpt_mini_pgqa_model.py

Pointer-GQA attention: a sliding window whose every edge is a relaxed,
sampled pointer.

Each layer owns ``P = window`` single-token pointers per query. Pointer ``p``
is anchored at base slot ``p`` (key position ``t - p``; ``p = 0`` is the
query itself) so at init the pointer set *is* the window ``[t - P + 1, t]``.
Each pointer carries ``L`` Bernoulli bits that move it off its slot:

  distance(p) = (p XOR low) + P * j        low = bits[:k], j = bits[k:]

``j`` picks which ``P``-wide region of the context the pointer reads (its
phase ``p mod P`` is kept), so the union of the ``P`` regions covers the
whole ``seq_len`` context and a region-only pointer (``k = 0``) can never
land on another pointer's phase: the ``P`` sampled keys are distinct by
construction. ``k`` *free* low bits let the ``2**k`` pointers that share
``p >> k`` also trade slots; coincident keys simply appear twice in the
softmax and add their mass (no merge step, no repulsion term).

Attention is grouped-query: ``dim // head_dim`` query heads score one shared
K/V head over the *same* ``P`` gathered keys, so the gather is paid once per
layer, not once per head. Scores are ``scale * q.k + log pi_p(bits_p)`` with
the sampled bits' log-probability as an additive prior; that prior term is
the only gradient path into the controller, and the softmax makes it
advantage-shaped for free (a pointer's bits are reinforced in proportion to
its mass times its value relative to the other ``P - 1`` pointers, which act
as the counterfactuals). Bits are sampled in eval as well: the model is the
sampled set, and mode-decoding would be a different model.

Controller (shared, low rank): ``u = silu(ctrl(x))`` in ``R`` dims; the
logit of bit ``b`` of pointer ``p`` is ``u . (mix[b] @ embed[p]) + bias[p,b]``.
Parameter count is ``dim*R + P*R + L*R*R + P*L``; the readout costs
``P*L*R`` MACs per token.

Kernel: an explicit gather of ``P`` K/V rows per query (the scattered
single-token key set of this design is exactly the case flex's 128-block
tables cannot express), activation-checkpointed so the ``[B, T, P, D]``
candidate tensors are not kept for backward.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pretraining.nanogpt_mini.nanogpt_mini_model import MLP, Linear, RMSNorm, Rotary

# Finite fill for masked scores: exp() underflows to exactly 0 in fp32 and
# rows with no valid key stay NaN-free (their output is zeroed afterwards).
MASKED_SCORE = -1e9
# Baseline's SDPA scale on rms-normed q/k.
SCORE_SCALE = 0.12


def _log2_exact(n: int, what: str) -> int:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"{what} must be a power of two, got {n}")
    return n.bit_length() - 1


def num_region_bits(seq_len: int, window: int) -> int:
    """Bits selecting which ``window``-wide region of a ``seq_len`` context a
    pointer reads: ``log2(seq_len / window)``."""
    if seq_len % window:
        raise ValueError(f"seq_len {seq_len} must be a multiple of window {window}")
    return _log2_exact(seq_len // window, "seq_len / window")


def pointer_distances(bits: Tensor, window: int, free_bits: int) -> Tensor:
    """Query-relative key distance of every pointer.

    ``bits``: ``[..., P, L]`` bool with ``P == window``. The low ``free_bits``
    bits XOR the pointer's slot, the remaining ``L - free_bits`` bits select
    the region. Returns ``[..., P]`` long.
    """
    P, L = bits.shape[-2:]
    if P != window:
        raise ValueError(f"one pointer per slot: expected {window} pointers, got {P}")
    if not 0 <= free_bits <= min(L, _log2_exact(window, "window")):
        raise ValueError(f"free_bits {free_bits} out of range for window {window}, {L} bits")
    powers = torch.pow(2, torch.arange(L, device=bits.device))
    code = (bits.long() * powers).sum(-1)
    low = code & ((1 << free_bits) - 1)
    region = code >> free_bits
    slot = torch.arange(P, device=bits.device)
    return (slot ^ low) + window * region


def pointer_positions(
    bits: Tensor, seq_len: int, window: int, free_bits: int
) -> tuple[Tensor, Tensor]:
    """Key positions of the sampled pointers (gather-ready).

    ``bits``: ``[B, T, P, L]``. Returns ``(positions, valid)`` of shape
    ``[B, T, P]``: ``positions`` clamped to ``>= 0`` for gathering, ``valid``
    marks pointers that land inside ``[0, t]`` (they never exceed ``t``).
    """
    distance = pointer_distances(bits, window, free_bits)
    queries = torch.arange(seq_len, device=bits.device).view(1, seq_len, 1)
    positions = queries - distance
    return positions.clamp_min(0), positions >= 0


def pointer_log_prior(logits: Tensor, bits: Tensor) -> Tensor:
    """``log pi`` of each pointer's sampled bits: ``[..., P, L]`` -> ``[..., P]``."""
    sign = bits.to(logits.dtype) * 2 - 1
    return F.logsigmoid(sign * logits).sum(-1)


class GatherRows(torch.autograd.Function):
    """``table[index]`` for a ``[N, D]`` table and a flat ``[M]`` index whose
    backward accumulates in fp32.

    The stock ``gather`` backward is a scatter-add in the table's dtype;
    with bf16 K/V and ~P contributions per row it degenerates to emulated
    bf16 atomics under heavy contention (measured 1.3 s per layer per
    16-row microbatch, 97% of the step). fp32 ``index_add_`` uses hardware
    atomics and is ~100x faster; the result is cast back once.
    """

    @staticmethod
    def forward(ctx, table: Tensor, index: Tensor) -> Tensor:
        ctx.save_for_backward(index)
        ctx.rows = table.size(0)
        ctx.dtype = table.dtype
        return table.index_select(0, index)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (index,) = ctx.saved_tensors
        acc = torch.zeros(ctx.rows, grad.size(1), dtype=torch.float32, device=grad.device)
        acc.index_add_(0, index, grad.to(torch.float32))
        return acc.to(ctx.dtype), None


def gqa_attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    positions: Tensor,
    valid: Tensor,
    log_prior: Tensor,
    scale: float,
) -> Tensor:
    """Every query head attends the shared K/V head over the query's pointers.

    ``q``: ``[B, T, H, D]``; ``k``/``v``: ``[B, T, D]``; ``positions``/
    ``valid``/``log_prior``: ``[B, T, P]``. Returns ``[B, T, H, D]``;
    queries with no valid pointer output zero. Coincident positions are
    separate softmax entries (their mass adds).
    """
    B, T, H, D = q.shape
    P = positions.size(-1)
    # one row gather serves K and V (and every query head): rows of the
    # flattened [B*T, 2D] table, batch folded into the index
    table = torch.cat((k, v), dim=-1).reshape(B * T, 2 * D)
    index = (positions + (torch.arange(B, device=q.device) * T).view(B, 1, 1)).reshape(-1)
    cand = GatherRows.apply(table, index).view(B, T, P, 2 * D)
    k_cand, v_cand = cand.split(D, dim=-1)
    score_dtype = torch.promote_types(q.dtype, torch.float32)
    scores = torch.einsum("bthd,btpd->bthp", q, k_cand).to(score_dtype) * scale
    scores = scores + log_prior.unsqueeze(2).to(score_dtype)
    valid_bt1p = valid.unsqueeze(2)
    scores = scores.masked_fill(~valid_bt1p, MASKED_SCORE)
    weights = torch.softmax(scores, dim=-1)
    weights = weights * valid_bt1p.any(-1, keepdim=True)
    return torch.einsum("bthp,btpd->bthd", weights.type_as(v_cand), v_cand)


class PointerController(nn.Module):
    """Shared low-rank bit-logit controller (see module docstring).

    Parameter names are deliberately outside the training scripts' generic
    ``weight``/``bias`` init rules: ``ptr_embed``, ``ptr_mix`` and
    ``ptr_bias`` need explicit init (``ptr_bias`` carries the stay-on-slot
    prior) and live in AdamW, not Muon.
    """

    def __init__(self, dim: int, num_pointers: int, num_bits: int, rank: int):
        super().__init__()
        self.ctrl = Linear(dim, rank)
        self.ptr_embed = nn.Parameter(torch.empty(num_pointers, rank))
        self.ptr_mix = nn.Parameter(torch.empty(num_bits, rank, rank))
        self.ptr_bias = nn.Parameter(torch.empty(num_pointers, num_bits))

    def forward(self, x: Tensor) -> Tensor:
        """``[B, T, dim]`` -> bit logits ``[B, T, P, L]`` in ``>= fp32``."""
        u = F.silu(self.ctrl(x))
        readout = torch.einsum("lrs,ps->plr", self.ptr_mix.type_as(u), self.ptr_embed.type_as(u))
        logits = torch.einsum("btr,plr->btpl", u, readout)
        logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
        return logits + self.ptr_bias.to(logits.dtype)


# ``fused``: Triton kernels (nanogpt_mini_pgqa_kernel), the training path.
# ``gather``: explicit PyTorch candidate gather, CPU-runnable; the tested
# reference.
BACKENDS = ("fused", "gather")


@torch.compiler.disable
def fused_attend(q: Tensor, k: Tensor, v: Tensor, positions: Tensor, valid: Tensor,
                 log_prior: Tensor, scale: float) -> Tensor:
    """``gqa_attend`` on the fused kernels. A (recursive) compile boundary:
    the kernels are hand-written Triton launched from an autograd.Function
    and gain nothing from inductor, whose re-tracing of them fails at
    precompile; the trunk on either side compiles as usual."""
    from pretraining.nanogpt_mini.nanogpt_mini_pgqa_kernel import fused_attend as kernel
    kv = torch.cat((k, v), dim=-1)
    pos = torch.where(valid, positions, -1).to(torch.int32)
    return kernel(q, kv, pos, log_prior.to(torch.float32), scale)


class PointerGQAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        window: int = 256,
        free_bits: int = 2,
        rank: int = 64,
        head_dim: int = 128,
        backend: str = "fused",
        checkpoint_attend: bool = True,
    ):
        super().__init__()
        if dim % head_dim or dim < head_dim:
            raise ValueError(f"dim {dim} must be a positive multiple of head_dim {head_dim}")
        _log2_exact(window, "window")
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.window = window
        self.free_bits = free_bits
        self.num_bits = num_region_bits(seq_len, window) + free_bits
        # validates free_bits against the window
        pointer_distances(torch.zeros(window, self.num_bits, dtype=torch.bool), window, free_bits)
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.backend = backend
        self.checkpoint_attend = checkpoint_attend  # gather backend only
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, head_dim)
        self.v = Linear(dim, head_dim)
        self.proj = Linear(hdim, dim)
        self.controller = PointerController(dim, window, self.num_bits, rank)
        self.rotary = Rotary(head_dim)

    @property
    def num_pointers(self) -> int:
        return self.window

    def sample_bits(self, logits: Tensor) -> Tensor:
        # The model IS the sampled pointer set, so eval samples too (the
        # mode would collapse every pointer onto its argmax, a set that never
        # trains).
        with torch.no_grad():
            return torch.bernoulli(torch.sigmoid(logits)).bool()

    def qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, T, H, D = x.size(0), x.size(1), self.num_heads, self.head_dim
        q = self.q(x).view(B, T, H, D)
        k = self.k(x).view(B, T, 1, D)
        v = self.v(x).view(B, T, D)
        q, k = F.rms_norm(q, (D,)), F.rms_norm(k, (D,))
        return self.rotary(q), self.rotary(k).squeeze(2), v

    def route(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(positions, valid, log_prior)`` of the sampled pointers, ``[B, T, P]``."""
        logits = self.controller(x)
        bits = self.sample_bits(logits)
        positions, valid = pointer_positions(bits, x.size(1), self.window, self.free_bits)
        return positions, valid, pointer_log_prior(logits, bits)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        if T != self.seq_len:
            raise ValueError(f"PointerGQAttention built for seq_len {self.seq_len}, got {T}")
        q, k, v = self.qkv(x)
        positions, valid, log_prior = self.route(x)
        if self.backend == "fused":
            y = fused_attend(q, k, v, positions, valid, log_prior, SCORE_SCALE)
        elif self.checkpoint_attend and torch.is_grad_enabled():
            y = checkpoint(gqa_attend, q, k, v, positions, valid, log_prior, SCORE_SCALE,
                           use_reentrant=False)
        else:
            y = gqa_attend(q, k, v, positions, valid, log_prior, SCORE_SCALE)
        y = y.reshape(B, T, self.num_heads * self.head_dim)
        return self.proj(y)


class PGQABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        window: int,
        free_bits: int,
        rank: int,
        mlp_hidden: "int | None" = None,
        head_dim: int = 128,
        backend: str = "fused",
        checkpoint_attend: bool = True,
    ):
        super().__init__()
        self.attn = PointerGQAttention(dim, seq_len, window, free_bits, rank, head_dim,
                                       backend, checkpoint_attend)
        self.mlp = MLP(dim, mlp_hidden)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PGQAGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        seq_len: int,
        window: int = 256,
        free_bits: int = 2,
        rank: int = 64,
        mlp_hidden: "int | None" = None,
        head_dim: int = 128,
        backend: str = "fused",
        checkpoint_attend: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([
            PGQABlock(model_dim, seq_len, window, free_bits, rank, mlp_hidden, head_dim,
                      backend, checkpoint_attend)
            for _ in range(num_layers)
        ])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")

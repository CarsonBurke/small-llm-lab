"""pretraining/nanogpt_mini/nanogpt_mini_ptr_model.py

Hierarchical-pointer sparse attention for the nanogpt-mini trunk. Every
attention layer is sparse; there is no dense causal layer anywhere.

Mechanism (per head, per query token ``t``):
  - ``K`` pointers. A pointer is ``L = log2(T / Bk)`` Bernoulli bits whose
    logits come from the token itself (``ptr: Linear(dim, H*K*L)``). The bits
    encode a block offset *backward* from ``t`` in units of ``Bk`` tokens:
    offset ``o`` selects the window ``[t - o*Bk - Bk + 1, t - o*Bk]``. Offset 0
    is the local window ending at ``t`` (self included); bit ``b`` moves
    ``2**b`` blocks, so flipping the low bit is a local step and flipping the
    high bit is a long jump ("local or full navigation"). Windows that fall
    before position 0 are masked, so the pattern is causal by construction.
  - Training samples the bits (``torch.bernoulli``), so the pointer explores
    without an optimizer step; eval takes the per-bit mode.
  - The query attends to the sampled block AND the ``L`` blocks reachable by
    flipping one bit (the Hamming-1 ball): ``K * (L + 1)`` blocks,
    ``K * (L + 1) * Bk`` keys, i.e. ``K * O(log N)`` per query.
  - Differentiable pointer: each candidate ``c`` scores
    ``q . k_c * scale + log pi(c)`` with ``log pi(c) = sum_b log P(bit_b = c_b)``
    under the token's bit distribution; one softmax spans the union of all
    pointers' candidates. The bit logits receive gradient only through the
    prior term of the flipped neighbours: attention mass landing on the
    flip-``b`` block pulls bit ``b`` toward that value ("if this edge had been
    elsewhere, would things go better?"). No REINFORCE, no straight-through.
    Duplicate blocks (two pointers landing on the same window) simply add
    their prior mass, matching ``log sum_j pi_j(p)`` of the dense analogue.

The dense-equivalent view: ``softmax_p(q . k_p * scale + log sum_j pi_j(p))``
over all causal positions, Monte-Carlo restricted to the Hamming-1 balls of
one sample per pointer.

Backends (same math; ``gather`` is the CPU-testable reference):
  - ``flex``: per query a ``[num_blocks]`` membership row and a log-prior row
    (``pointer_tables``) feed flex_attention's ``mask_mod``/``score_mod``; the
    fused kernel never materializes candidates and the prior table receives
    gradient through ``score_mod``. Tile skipping is at 128x128 granularity:
    a tile is skipped only when no query in it has a candidate there. With
    per-token pointers at random offsets almost every tile is touched, so at
    T=1024 compute is ~dense causal; the ``K * O(log N)`` cost is asymptotic
    (true skipping needs pointers shared across a 128-query tile and
    128-aligned windows, i.e. block-of-nodes pointers).
  - ``gather``: explicit ``[B, H, T, C, D]`` candidate gathers. Exactly
    ``K * (L + 1) * Bk`` keys per query, but memory-bound and ~20x slower
    than the baseline at mbs 4 on a 5090; keep for tests and equivalence
    checks.

Shared trunk pieces (``RMSNorm``/``Linear``/``Rotary``/``MLP``) are imported
from ``nanogpt_mini_model`` so the only architectural difference from the
baseline is the mixer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pretraining.nanogpt_mini.nanogpt_mini_model import MLP, Linear, RMSNorm, Rotary

# Finite fill for masked scores: exp() underflows to exactly 0 in fp32 and
# rows with no valid key stay NaN-free (their output is zeroed afterwards).
MASKED_SCORE = -1e9
# Baseline's SDPA scale on rms-normed q/k.
SCORE_SCALE = 0.12
# ``flex``: fused kernel over per-query block tables (training path).
# ``gather``: explicit candidate gather, CPU-runnable; the tested reference.
BACKENDS = ("flex", "gather")


def num_pointer_bits(seq_len: int, block_size: int) -> int:
    """Bits needed to address every block offset of a ``seq_len`` window."""
    if block_size <= 0 or seq_len % block_size:
        raise ValueError(
            f"block_size must divide seq_len, got block_size={block_size}, seq_len={seq_len}"
        )
    num_blocks = seq_len // block_size
    if num_blocks & (num_blocks - 1):
        raise ValueError(
            f"seq_len / block_size must be a power of two, got {num_blocks}"
        )
    return max(1, num_blocks.bit_length() - 1)


def pointer_log_prior(logits: Tensor, bits: Tensor) -> Tensor:
    """``log pi`` of the sampled address and of each single-bit flip.

    ``logits``/``bits``: ``[..., L]``. Returns ``[..., L + 1]``: column 0 is
    the sampled address, column ``1 + b`` is the address with bit ``b``
    flipped.
    """
    sign = bits.to(logits.dtype) * 2 - 1
    lp_keep = F.logsigmoid(sign * logits)
    lp_flip = F.logsigmoid(-sign * logits)
    lp_addr = lp_keep.sum(-1, keepdim=True)
    delta = torch.cat((torch.zeros_like(lp_addr), lp_flip - lp_keep), dim=-1)
    return lp_addr + delta


def pointer_offsets(bits: Tensor) -> Tensor:
    """Block offsets of the Hamming-1 ball: ``[..., L]`` bits -> ``[..., L + 1]``
    (column 0 the sampled address, column ``1 + b`` its bit-``b`` flip)."""
    powers = torch.pow(2, torch.arange(bits.size(-1), device=bits.device))
    flips = torch.cat((powers.new_zeros(1), powers))
    address = (bits.long() * powers).sum(-1)
    return address.unsqueeze(-1) ^ flips


def pointer_candidates(
    bits: Tensor, seq_len: int, block_size: int
) -> tuple[Tensor, Tensor]:
    """Key positions of the Hamming-1 ball of every pointer (gather backend).

    ``bits``: ``[B, T, H, K, L]`` bool. Returns ``(positions, valid)`` of shape
    ``[B, T, H, K, L + 1, Bk]``: ``positions`` is clamped to ``>= 0`` for
    gathering, ``valid`` marks entries that lie inside ``[0, t]``.
    """
    offsets = pointer_offsets(bits)
    window = torch.arange(block_size, device=bits.device)
    queries = torch.arange(seq_len, device=bits.device).view(1, seq_len, 1, 1, 1, 1)
    positions = queries - offsets.unsqueeze(-1) * block_size - window
    valid = positions >= 0
    return positions.clamp_min(0), valid


def pointer_tables(
    offsets: Tensor,
    log_prior: Tensor,
    num_blocks: int,
    local_blocks: int = 0,
    valid: "Tensor | None" = None,
) -> tuple[Tensor, Tensor]:
    """Per-query block-offset tables for the flex backend.

    ``offsets``/``log_prior``: ``[B, T, H, K, C]`` candidate slots and their
    log prior; ``valid`` (same shape, optional) drops candidates entirely.
    Returns ``member[B, H, T, num_blocks]`` (offset is in some pointer's
    ball or in the always-on local window of ``local_blocks`` offsets) and
    ``prior[B, H, T, num_blocks]`` = ``log(sum_j pi_j + [o < local_blocks])``
    over the valid candidates landing on the offset (0 where absent; masked
    anyway), so duplicate windows add their mass exactly as the gather
    backend does and the local window carries a fixed prior mass of 1.
    """
    B, T, H = offsets.shape[:3]
    index = offsets.permute(0, 2, 1, 3, 4).reshape(B, H, T, -1)
    lp = log_prior.permute(0, 2, 1, 3, 4).reshape(B, H, T, -1)
    keep = (
        torch.ones_like(index, dtype=torch.bool)
        if valid is None
        else valid.permute(0, 2, 1, 3, 4).reshape(B, H, T, -1)
    )
    shape = (B, H, T, num_blocks)
    local = torch.arange(num_blocks, device=offsets.device) < local_blocks
    member = torch.zeros(shape, dtype=torch.int32, device=offsets.device)
    member = (member.scatter_add(-1, index, keep.to(torch.int32)) > 0) | local
    # logsumexp via a detached per-slot max: d/dm (m + log sum exp(x - m)) == 0
    peak = torch.full(shape, float("-inf"), dtype=lp.dtype, device=lp.device)
    peak = peak.scatter_reduce(-1, index, lp.detach().masked_fill(~keep, float("-inf")), reduce="amax")
    peak = torch.where(local, peak.clamp_min(0.0), peak).masked_fill(~member, 0.0)
    mass = torch.zeros(shape, dtype=lp.dtype, device=lp.device)
    mass = mass.scatter_add(-1, index, (lp - peak.gather(-1, index)).exp() * keep)
    mass = mass + torch.where(local, (-peak).exp(), torch.zeros((), dtype=lp.dtype, device=lp.device))
    prior = peak + mass.masked_fill(~member, 1.0).log()
    return member, prior


@torch.compiler.disable(recursive=False)
def flex_attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    member: Tensor,
    prior: Tensor,
    block_size: int,
    scale: float,
) -> Tensor:
    """``sparse_attend`` semantics on the fused flex_attention kernel.

    ``q``/``k``/``v``: ``[B, T, H, D]``; ``member``/``prior`` from
    ``pointer_tables`` (the local window is already folded in). The key at
    ``kv`` lies in query ``t``'s window of block offset ``(t - kv) // Bk``;
    the block mask keeps causal members, the score_mod adds the prior (its
    captured table receives gradient). Fully masked rows output zero, as in
    the gather backend.

    Compile boundary on purpose: when a whole-model ``torch.compile`` graph
    contains this call, ``prior`` is a graph intermediate and inductor
    (torch 2.14) miscompiles the flex backward — captured-buffer AND v/k
    gradients come out ~1e6 too large (a bigram-plateau training failure),
    and the eval graph fails to lower ("convert FlexibleLayout to
    FixedLayout first"). With flex outside the outer graph its inputs are
    graph inputs and both problems vanish; the kernels below are compiled on
    their own. Measured cost vs the (wrong) fused graph: none once the
    skipped captured-gradient atomics are accounted for.
    """
    B, T, H, _ = q.shape

    def mask_mod(b, h, q_idx, kv_idx):
        slot = ((q_idx - kv_idx) // block_size).clamp(min=0)
        return (kv_idx <= q_idx) & member[b, h, q_idx, slot]

    def score_mod(score, b, h, q_idx, kv_idx):
        slot = ((q_idx - kv_idx) // block_size).clamp(min=0)
        return score + prior[b, h, q_idx, slot]

    block_mask = compiled_block_mask_builder()(mask_mod, B, H, T, T, device=q.device)
    out = compiled_flex_attention()(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        score_mod=score_mod, block_mask=block_mask, scale=scale,
    )
    return out.transpose(1, 2)


_FLEX_ATTENTION = None
_BLOCK_MASK_BUILDER = None


def compiled_flex_attention():
    """Lazily compiled ``flex_attention`` (eager flex is a slow reference)."""
    global _FLEX_ATTENTION
    if _FLEX_ATTENTION is None:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)
    return _FLEX_ATTENTION


def compiled_block_mask_builder():
    """Lazily compiled ``create_block_mask`` (the mask is data-dependent, so it
    is rebuilt every forward; eager evaluation is the slow vmap path)."""
    global _BLOCK_MASK_BUILDER
    if _BLOCK_MASK_BUILDER is None:
        from torch.nn.attention.flex_attention import create_block_mask

        _BLOCK_MASK_BUILDER = torch.compile(create_block_mask, dynamic=False)
    return _BLOCK_MASK_BUILDER


def sparse_attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    positions: Tensor,
    valid: Tensor,
    log_prior: Tensor,
    scale: float,
) -> Tensor:
    """Softmax attention of every query over its own candidate key list.

    ``q``/``k``/``v``: ``[B, T, H, D]``. ``positions``/``valid``:
    ``[B, T, H, C]``. ``log_prior``: ``[B, T, H, C]`` additive score term.
    Returns ``[B, T, H, D]``; queries with no valid candidate output zero.
    """
    B, T, H, D = q.shape
    C = positions.size(-1)
    index = positions.permute(0, 2, 1, 3).reshape(B, H, T * C, 1).expand(-1, -1, -1, D)
    k_bhtd = k.transpose(1, 2)
    v_bhtd = v.transpose(1, 2)
    k_cand = torch.gather(k_bhtd, 2, index).view(B, H, T, C, D)
    v_cand = torch.gather(v_bhtd, 2, index).view(B, H, T, C, D)
    q_bhtd = q.transpose(1, 2)
    score_dtype = torch.promote_types(q.dtype, torch.float32)
    scores = torch.einsum("bhtd,bhtcd->bhtc", q_bhtd, k_cand).to(score_dtype) * scale
    scores = scores + log_prior.permute(0, 2, 1, 3).to(score_dtype)
    valid_bhtc = valid.permute(0, 2, 1, 3)
    scores = scores.masked_fill(~valid_bhtc, MASKED_SCORE)
    weights = torch.softmax(scores, dim=-1)
    weights = weights * valid_bhtc.any(-1, keepdim=True)
    out = torch.einsum("bhtc,bhtcd->bhtd", weights.type_as(v_cand), v_cand)
    return out.transpose(1, 2)


class PointerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        num_pointers: int,
        block_size: int,
        head_dim: int = 128,
        backend: str = "flex",
        local_window: int = 0,
    ):
        super().__init__()
        if dim % head_dim or dim < head_dim:
            raise ValueError(f"dim {dim} must be a positive multiple of head_dim {head_dim}")
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.num_pointers = num_pointers
        self.block_size = block_size
        self.num_bits = num_pointer_bits(seq_len, block_size)
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        # Bit logits, laid out (head, pointer, bit); ``ptr`` (not ``proj``) so
        # the scripts' name-based init gives it the block-matrix draw.
        self.ptr = Linear(dim, self.num_heads * num_pointers * self.num_bits)
        self.rotary = Rotary(head_dim)
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.backend = backend
        if local_window < 0 or local_window % block_size or local_window > seq_len:
            raise ValueError(
                f"local_window must be a multiple of block_size within seq_len, got {local_window}"
            )
        self.local_window = local_window

    @property
    def num_blocks(self) -> int:
        return self.seq_len // self.block_size

    @property
    def candidates_per_query(self) -> int:
        return self.num_pointers * (self.num_bits + 1) * self.block_size + self.local_window

    def sample_bits(self, logits: Tensor) -> Tensor:
        # The model IS the sampled path plus its Hamming-1 ball, so eval
        # samples too instead of mode-decoding the bits (measured at step
        # 100: mode 3.057 vs sampled 3.061 nats -- same model either way, but
        # the mode collapses the ball onto the argmax path, which is not what
        # trains).
        with torch.no_grad():
            return torch.bernoulli(torch.sigmoid(logits)).bool()

    def bit_logits(self, x: Tensor) -> Tensor:
        B, T = x.size(0), x.size(1)
        logits = self.ptr(x).view(B, T, self.num_heads, self.num_pointers, self.num_bits)
        # prior/logsigmoid in >= fp32 (bf16 trunk); keeps fp64 as fp64
        return logits.to(torch.promote_types(logits.dtype, torch.float32))

    def qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, T, H, D = x.size(0), x.size(1), self.num_heads, self.head_dim
        q = self.q(x).view(B, T, H, D)
        k = self.k(x).view(B, T, H, D)
        v = self.v(x).view(B, T, H, D)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        return self.rotary(q), self.rotary(k), v

    def prepare(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Everything before the attention kernel: ``(q, k, v, member, prior)``
        with the flex tables of ``pointer_tables``. Compile-friendly (no
        data-dependent shapes; the Bernoulli draw is an ordinary RNG op)."""
        q, k, v = self.qkv(x)
        logits = self.bit_logits(x)
        bits = self.sample_bits(logits)
        log_prior = pointer_log_prior(logits, bits)
        member, prior = pointer_tables(
            pointer_offsets(bits), log_prior, self.num_blocks, self.local_window // self.block_size
        )
        return q, k, v, member, prior

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        if T != self.seq_len:
            raise ValueError(f"PointerAttention built for seq_len {self.seq_len}, got {T}")
        H, D = self.num_heads, self.head_dim
        if self.backend == "flex":
            q, k, v, member, prior = self.prepare(x)
            y = flex_attend(q, k, v, member, prior, self.block_size, scale=SCORE_SCALE)
        else:
            q, k, v = self.qkv(x)
            logits = self.bit_logits(x)
            bits = self.sample_bits(logits)
            log_prior = pointer_log_prior(logits, bits)  # [B, T, H, K, L + 1]
            positions, valid = pointer_candidates(bits, T, self.block_size)
            C = self.num_pointers * (self.num_bits + 1) * self.block_size
            log_prior = log_prior.unsqueeze(-1).expand(-1, -1, -1, -1, -1, self.block_size)
            positions = positions.reshape(B, T, H, C)
            valid = valid.reshape(B, T, H, C)
            log_prior = log_prior.reshape(B, T, H, C)
            if self.local_window:
                # always-on local keys with prior mass 1 (log 1 = 0); keys also
                # inside a pointer ball appear twice and add mass, matching
                # pointer_tables' ``+ 1``
                local_pos = (torch.arange(T, device=x.device).view(1, T, 1, 1)
                             - torch.arange(self.local_window, device=x.device))
                local_pos = local_pos.expand(B, T, H, -1)
                positions = torch.cat((positions, local_pos.clamp_min(0)), -1)
                valid = torch.cat((valid, local_pos >= 0), -1)
                log_prior = torch.cat((log_prior, torch.zeros_like(local_pos, dtype=log_prior.dtype)), -1)
            y = sparse_attend(q, k, v, positions, valid, log_prior, scale=SCORE_SCALE)
        y = y.contiguous().view(B, T, H * D)
        return self.proj(y)


class PtrBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        num_pointers: int,
        block_size: int,
        mlp_hidden: "int | None" = None,
        backend: str = "flex",
        local_window: int = 0,
        head_dim: int = 128,
    ):
        super().__init__()
        self.attn = PointerAttention(
            dim, seq_len, num_pointers, block_size, head_dim=head_dim, backend=backend,
            local_window=local_window,
        )
        self.mlp = MLP(dim, mlp_hidden)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PtrGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        seq_len: int,
        num_pointers: int,
        block_size: int,
        mlp_hidden: "int | None" = None,
        backend: str = "flex",
        local_window: int = 0,
        head_dim: int = 128,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([
            PtrBlock(model_dim, seq_len, num_pointers, block_size, mlp_hidden, backend,
                     local_window, head_dim)
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

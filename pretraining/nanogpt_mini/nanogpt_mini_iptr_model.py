"""pretraining/nanogpt_mini/nanogpt_mini_iptr_model.py

Interval-decoded pointer attention: parallel, length-aware pointers whose
bits are decoded as an exact binary search over the blocks that actually
exist behind the query. Sibling of ``nanogpt_mini_ptr_model.py`` (fixed
binary offsets, power-of-two windows) and ``nanogpt_mini_hashptr_model.py``
(content buckets). Every attention layer is sparse; no dense layer.

Addressing (per query ``t``, head ``h``, pointer ``k``):
  - ``N_t = t + 1`` accessible tokens (self included), ``M_t = ceil(N_t /
    Bk)`` accessible blocks of query-relative offsets ``o``: offset ``o``
    reads backward distances ``[o Bk, min((o + 1) Bk, N_t))``. The oldest
    block may be partial; only its real tokens are read (``kv >= 0``).
  - Routing depth ``d`` splits the current offset interval ``[lo, hi)`` at
    ``mid = (lo + hi) // 2``: bit 0 = nearer half ``[lo, mid)``, bit 1 =
    farther half ``[mid, hi)``. Halves are nonempty whenever ``hi - lo >
    1``; decoding stops at one block, so a query with a single accessible
    block makes zero routing decisions and every accessible block is a leaf.
    Arbitrary ``T`` and non-power-of-two block counts work; the number of
    depths ``D = ceil(log2 ceil(T / Bk))`` is derived from the input.
  - One shared-depth controller (``IntervalPointerHead``) serves every
    depth and context length: ``u = W x_t`` (width ``R``), ``z = w . silu(u
    + V f) + v . f + c`` with routing features ``f`` telling the head which
    bit it is predicting, how much context exists and what jump scale the
    bit controls. Two decoders:
      * ``"interval"``: every bit sampled independently and IN PARALLEL
        from ``f_{t,d}`` (``routing_features``); no selected interval enters
        the head, so the address distribution is a product of Bernoullis.
      * ``"sequential"``: bit ``d`` sampled given the interval reached by
        the bits ``< d`` (``sequential_route``; ``f`` adds ``log2(hi - lo)``
        exact and ``log2(lo + 1)``), so the address distribution is a tree
        and any preference over blocks is expressible. Same head, same
        parameters, ``D`` sequential evaluations of ``[B, T, H, K, R]``.
  - Bits are drawn with ``torch.bernoulli`` in training and eval alike.
    The query attends to the sampled leaf and the ``D`` leaves reached by
    flipping one bit and re-decoding (Hamming-1 ball). A flip of a bit the
    sampled path never consumed re-decodes to the same leaf and is dropped
    (``valid``). Candidate ``c`` carries ``log pi(c)`` summed over the depths
    ITS path consumed; one softmax spans the union. Only the prior term
    reaches the controller. ``prior`` selects how it enters the score:
      * ``"score"``: ``q . k * scale + log pi(c)``. The controller's gradient
        on bit ``d`` is ``-s_d * w_flip-d * (g_flip-d - g_mean)`` -- the
        flip-``d`` candidate's attention gradient -- and ``w_flip-d`` carries
        ``pi(flip-d)``, so a confident bit suppresses its own counterfactual
        exponentially and the prior doubles as a learned position bias.
      * ``"straight"``: ``q . k * scale + (log pi(c) - sg(log pi(c)))``. The
        forward pass attends by content only over the sampled ball; the
        backward pass is the same finite difference with content-only
        weights. Measured: the ``pi`` factors of keep and flip cancel, so
        the gradient never saturates and the logits run away (diverged by
        step ~250 of a 2k schedule).
      * ``"leg"`` (local expectation gradient): candidate 0 carries ``sum_d
        pi(keep_d)``, the flip-``d`` candidate ``pi(flip_d)``, each as ``p -
        sg(p)`` (``leg_surrogate``). Content-only forward; backward ``s_d *
        sigma'(z_d) * (e_0 - e_flip-d)`` with ``e_c`` the attention-score
        gradient of candidate ``c``: the exact single-site derivative of the
        expected loss with the mass swap sampled <-> flipped as the proxy for
        ``L(sampled) - L(flipped)``. Saturates at both ends; ``pi`` is purely
        the sampling distribution and a confident controller makes sample =
        mode so eval is deterministic.

The flex path is the position model's: block-offset ``member``/``prior``
tables (``pointer_tables`` with ``valid``) and the same ``mask_mod``/
``score_mod``; ``gather`` is the CPU-testable reference.
"""

from __future__ import annotations


import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pretraining.nanogpt_mini.nanogpt_mini_model import MLP, Linear, RMSNorm, Rotary
from pretraining.nanogpt_mini.nanogpt_mini_ptr_model import (
    BACKENDS,
    SCORE_SCALE,
    flex_attend,
    pointer_tables,
    sparse_attend,
)


def num_blocks(seq_len: int, block_size: int) -> int:
    return -(-seq_len // block_size)


def num_depths(seq_len: int, block_size: int) -> int:
    """``ceil(log2(ceil(T / Bk)))``: routing decisions of the longest query."""
    return max(0, num_blocks(seq_len, block_size) - 1).bit_length()


def accessible_blocks(seq_len: int, block_size: int, device: torch.device) -> Tensor:
    """``M_t = ceil((t + 1) / Bk)`` for every query position, ``[T]``."""
    return (torch.arange(seq_len, device=device) + block_size) // block_size


def flip_matrix(depths: int, device: torch.device) -> Tensor:
    """``[D + 1, D]`` bool: candidate 0 keeps the sampled bits, candidate
    ``1 + d`` flips bit ``d``."""
    return torch.cat((
        torch.zeros(1, depths, dtype=torch.bool, device=device),
        torch.eye(depths, dtype=torch.bool, device=device),
    ))


def decode_intervals(bits: Tensor, blocks: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Binary-search decode of the sampled bits and every single-bit flip.

    ``bits``: ``[B, T, H, K, D]`` bool (1 = farther half); ``blocks``: ``[T]``
    accessible block counts. Returns ``offsets[B, T, H, K, D + 1]`` (the leaf
    block offset of each candidate, always ``< M_t``), ``used[B, T, H, K,
    D + 1, D]`` (which depths each candidate's path consumed) and
    ``valid[B, T, H, K, D + 1]`` (candidate 0 always; flip ``d`` only if the
    sampled path consumed bit ``d`` — otherwise it is the same leaf).
    """
    D = bits.size(-1)
    cand = bits.unsqueeze(-2) ^ flip_matrix(D, bits.device)  # [B,T,H,K,D+1,D]
    lo = torch.zeros(cand.shape[:-1], dtype=torch.long, device=bits.device)
    hi = blocks.view(1, -1, 1, 1, 1).expand_as(lo)
    used = []
    for d in range(D):
        active = (hi - lo) > 1
        mid = (lo + hi) // 2
        farther = cand[..., d]
        lo = torch.where(active & farther, mid, lo)
        hi = torch.where(active & ~farther, mid, hi)
        used.append(active)
    used = torch.stack(used, -1) if D else cand.new_zeros((*cand.shape[:-1], 0))
    valid = torch.cat((torch.ones_like(lo[..., :1], dtype=torch.bool), used[..., 0, :]), -1)
    return lo, used, valid


def decode_padded(bits: Tensor, blocks: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """DIAGNOSTIC decode, not the design: binary search over the fixed range
    ``[0, 2^D)`` for every query (so a bit pattern is the same absolute
    offset at every ``t``, every depth is consumed, every flip is a distinct
    leaf) and candidates whose leaf is ``>= M_t`` are dropped. This is the
    position model's addressing inside the interval code path; it exists to
    measure what the ``t``-relative interval map costs.
    """
    D = bits.size(-1)
    padded = torch.full_like(blocks, 1 << D)
    offsets, used, _ = decode_intervals(bits, padded)
    valid = offsets < blocks.view(1, -1, 1, 1, 1)
    return offsets, used, valid


DECODES = ("interval", "padded", "sequential")
PRIORS = ("score", "straight", "leg")


def interval_log_prior(logits: Tensor, bits: Tensor, used: Tensor) -> Tensor:
    """``log pi`` of each candidate over the depths its own path consumed.
    ``logits``/``bits``: ``[..., D]``; ``used``: ``[..., D + 1, D]``."""
    sign = bits.to(logits.dtype) * 2 - 1
    lp_keep = F.logsigmoid(sign * logits).unsqueeze(-2)
    lp_flip = F.logsigmoid(-sign * logits).unsqueeze(-2)
    lp = torch.where(flip_matrix(bits.size(-1), bits.device), lp_flip, lp_keep)
    return (lp * used.to(lp.dtype)).sum(-1)


def leg_surrogate(logits: Tensor, bits: Tensor, used: Tensor) -> Tensor:
    """Local-expectation-gradient surrogate, ``[..., D]`` -> ``[..., D + 1]``:
    candidate 0 carries ``sum_d pi(bit d = sampled value)`` over its consumed
    depths, candidate ``1 + d`` carries ``pi(bit d = flipped value)``. Used
    as ``p - sg(p)`` so the forward is content-only and the controller's
    gradient on bit ``d`` is ``s_d * sigma'(z_d) * (e_0 - e_flip-d)`` with
    ``e_c`` the attention-score gradient of candidate ``c``: the exact
    single-site derivative of the expected loss over bit ``d`` (``d pi_d /
    d z_d = s_d sigma'(z_d)``) times the first-order proxy of ``L(sampled)
    - L(flipped)`` (swap the sampled leaf's mass with the flipped leaf's).
    Saturates as a bit becomes confident either way, unlike ``log pi -
    sg(log pi)`` whose ``pi`` factors cancel and let the logits run away;
    ``sum_c e_c = 0`` across a query's whole candidate set, so no
    pointer-level REINFORCE-like term ``(1 - pi) * sum_{c in ball} e_c``
    survives (the log forms carry one whenever ``K > 1``).
    """
    sign = bits.to(logits.dtype) * 2 - 1
    p_keep = torch.sigmoid(sign * logits)
    p_flip = 1 - p_keep
    keep = (p_keep * used[..., 0, :].to(p_keep.dtype)).sum(-1, keepdim=True)
    return torch.cat((keep, p_flip), -1)


def routing_features(seq_len: int, depths: int, block_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """What the PARALLEL controller knows about decision ``d`` at query ``t``
    before any bit is drawn, ``[T, D, F]`` with ``F = NUM_FEATURES``:
    ``log2 N_t`` (available context, tokens), ``d`` (which bit is being
    predicted), ``log2 max(M_t / 2^d, 1)`` (blocks in the interval this
    bit halves, i.e. the jump scale it controls; exact for power-of-two
    ``M_t``, off by at most one block otherwise) and the interval start,
    which the parallel decoder cannot know (constant 0; the sequential
    decoder fills it with ``log2(lo + 1)``).

    How many bits a query consumes is not predicted: it is ``ceil(log2
    M_t)``, fixed by the available context (``decode_intervals`` stops at
    one block, and unconsumed depths carry no prior, no candidate and no
    gradient). Logits for all ``D`` depths are computed only as padding.
    """
    t = torch.arange(seq_len, device=device, dtype=dtype)
    d = torch.arange(depths, device=device, dtype=dtype)
    log_len = torch.log2(t + 1)
    blocks = ((t + block_size) // block_size).view(-1, 1)
    scale = torch.log2((blocks / d.exp2()).clamp_min(1.0))
    return torch.stack((
        log_len.view(-1, 1).expand(-1, depths),
        d.view(1, -1).expand(seq_len, -1),
        scale,
        torch.zeros_like(scale),
    ), -1)


NUM_FEATURES = 4
DEPTH_FEATURE = 1  # index of ``d``
SCALE_FEATURE = 2  # index of ``log2 (blocks in the interval being halved)``
START_FEATURE = 3  # index of ``log2 (lo + 1)`` (sequential decoder only)


class IntervalPointerHead(nn.Module):
    """Shared-depth controller. Per (head, pointer) with width ``R`` and
    routing features ``f`` (``routing_features`` / ``sequential_route``):

        u = W x_t                                   [R]
        z = w . silu(u + V f) + v . f + c

    ``V`` (``feat_in``, ``[F, R]``) conditions the hidden units on which
    bit is being predicted, how much context exists and (sequential decode)
    which interval the search is in; ``v`` (``feat_out``, ``[F]``) is a
    direct linear read-out so a geometric prior over jump size is
    representable exactly at init: a decision over an interval of ``2^s``
    blocks jumps ``2^(s - 1)`` blocks, so ``v_scale = -slope, c = slope``
    gives ``logit = -slope * log2(jump)`` -- the position model's
    ``-slope * b`` for bit ``b``, in absolute blocks whatever ``M_t`` is.
    One module serves every depth and every context length; ``forward``
    predicts all depths in parallel from the same ``u`` (no selected
    interval enters), ``logits`` scores one decision from per-sample
    features (the sequential decoder's path).
    """

    def __init__(self, dim: int, num_heads: int, num_pointers: int, width: int):
        super().__init__()
        if width <= 0:
            raise ValueError(f"controller width must be positive, got {width}")
        self.num_heads = num_heads
        self.num_pointers = num_pointers
        self.width = width
        self.ctrl = Linear(dim, num_heads * num_pointers * width)
        self.feat_in = nn.Parameter(torch.zeros(num_heads, num_pointers, NUM_FEATURES, width))  # V
        self.feat_out = nn.Parameter(torch.zeros(num_heads, num_pointers, NUM_FEATURES))        # v
        self.out = nn.Parameter(torch.zeros(num_heads, num_pointers, width))                    # w
        self.out_bias = nn.Parameter(torch.zeros(num_heads, num_pointers))                      # c

    def hidden_input(self, x: Tensor) -> Tensor:
        """``u = W x_t`` as ``[B, T, H, K, R]`` in ``>= fp32``."""
        B, T = x.size(0), x.size(1)
        u = self.ctrl(x).view(B, T, self.num_heads, self.num_pointers, self.width)
        return u.to(torch.promote_types(u.dtype, torch.float32))

    def logits(self, u: Tensor, feats: Tensor) -> Tensor:
        """One decision per (sample, head, pointer): ``u`` ``[B, T, H, K, R]``,
        ``feats`` ``[B, T, H, K, F]`` -> ``[B, T, H, K]``."""
        shift = torch.einsum("bthkf,hkfr->bthkr", feats, self.feat_in.to(u.dtype))
        direct = torch.einsum("bthkf,hkf->bthk", feats, self.feat_out.to(u.dtype))
        hidden = F.silu(u + shift)
        return (hidden * self.out.to(u.dtype)).sum(-1) + direct + self.out_bias.to(u.dtype)

    def forward(self, x: Tensor, depths: int, block_size: int) -> Tensor:
        u = self.hidden_input(x)
        feats = routing_features(x.size(1), depths, block_size, x.device, u.dtype)  # [T, D, F]
        shift = torch.einsum("tdf,hkfr->thkdr", feats, self.feat_in.to(u.dtype))   # [T, H, K, D, R]
        direct = torch.einsum("tdf,hkf->thkd", feats, self.feat_out.to(u.dtype))   # [T, H, K, D]
        hidden = F.silu(u.unsqueeze(-2) + shift)                                    # [B, T, H, K, D, R]
        return (hidden * self.out.to(u.dtype).unsqueeze(-2)).sum(-1) + direct + self.out_bias.to(u.dtype).unsqueeze(-1)


def pooled_scores(q: Tensor, k: Tensor, block_size: int) -> Tensor:
    """``s[b, t, h, o] = q_t . sum_{i <= t - o Bk} k_i`` for block offsets ``o``
    (0 where ``t - o Bk < 0``), ``[B, T, H, M]`` fp32, from rms-normed
    pre-rotary ``q``/``k`` ``[B, T, H, Dh]``.

    The pooled content of any query-relative offset interval ``[a, b)`` is
    then ``(s[t, a] - s[t, b]) / count``, scalar gathers per routing
    decision. Computed as one matmul per residue ``r = t mod Bk`` on the
    inclusive prefix sum of ``k``: for ``t = Bk m + r`` the boundary
    ``t - o Bk`` is ``Bk (m - o) + r``, i.e. the same residue, so
    ``S_r = Q_r @ P_r^T`` (``[M, M]``) holds every needed dot and
    ``s[t, o] = S_r[m, m - o]``. ``T M Dh`` MACs per (b, h): ``1 / Bk`` of
    dense attention. fp32 throughout: the prefix sums reach ``O(T)`` and
    their differences over short intervals would be lost in bf16.
    """
    B, T, H, Dh = q.shape
    Bk = block_size
    if T % Bk:
        raise ValueError(f"pooled routing needs seq_len {T} to be a multiple of block_size {Bk}")
    M = T // Bk
    dtype = torch.promote_types(q.dtype, torch.float32)
    prefix = torch.cumsum(k.to(dtype), dim=1)                                 # P[j] = sum_{i <= j} k_i
    qr = q.to(dtype).view(B, M, Bk, H, Dh).permute(0, 3, 2, 1, 4)             # [B, H, Bk, M, Dh]: t = Bk m + r
    pr = prefix.view(B, M, Bk, H, Dh).permute(0, 3, 2, 1, 4)
    S = qr @ pr.transpose(-1, -2)                                             # [B, H, Bk, M, M]: S[m, m']
    m = torch.arange(M, device=q.device)
    diff = m.view(M, 1) - m.view(1, M)                                        # [m, o] -> m - o
    s_r = S.gather(-1, diff.clamp_min(0).expand(B, H, Bk, M, M)) * (diff >= 0)
    return s_r.permute(0, 3, 2, 1, 4).reshape(B, T, H, M)                     # [B, T, H, M]: t = Bk m + r


def sequential_route(
    head: IntervalPointerHead,
    x: Tensor,
    depths: int,
    block_size: int,
    pooled: "Tensor | None" = None,
    gains: "Tensor | None" = None,
) -> tuple[Tensor, Tensor]:
    """Interval-conditioned sampling: bit ``d`` is drawn given the interval
    ``[lo, hi)`` the bits ``< d`` reached, so the address distribution is a
    tree (any preference over blocks is expressible) instead of a product
    of independent Bernoullis. The head sees ``(log2 N_t, d, log2(hi - lo),
    log2(lo + 1))``; ``log2(hi - lo)`` is the exact jump scale the parallel
    decoder only approximates. Returns ``(bits, logits)``, both ``[B, T, H,
    K, D]``; ``decode_intervals`` on these bits reproduces the sampled path
    (candidate 0) and re-decodes each single-bit flip with the sampled lower
    bits, exactly as the parallel decoder does -- no extra policy
    evaluations for the counterfactuals.

    With ``pooled`` (``pooled_scores``, ``[B, T, H, M]``) and ``gains``
    (``[H, K]``) the decision also reads the keys: ``z += gain * scale *
    (mean q.k over the farther half - mean q.k over the nearer half)``.
    Content addressing on the key side only -- the real ``q`` compares the
    two halves' pooled keys, the search finally reads the array -- while
    position stays in the head's features. Gradient reaches ``q`` and the
    keys through the prior term, so keys learn to advertise themselves to
    the queries that need them.
    """
    B, T = x.size(0), x.size(1)
    u = head.hidden_input(x)                                              # [B, T, H, K, R]
    lo = torch.zeros(u.shape[:-1], dtype=torch.long, device=x.device)
    hi = accessible_blocks(T, block_size, x.device).view(1, T, 1, 1).expand_as(lo)
    log_len = torch.log2(torch.arange(T, device=x.device, dtype=u.dtype) + 1).view(1, T, 1, 1).expand_as(lo)
    if pooled is not None:
        H, K = u.size(2), u.size(3)
        M = pooled.size(-1)
        t_idx = torch.arange(T, device=x.device).view(1, T, 1, 1)
        gain = (gains.to(u.dtype) * SCORE_SCALE).view(1, 1, H, K)

        def boundary(o: Tensor) -> Tensor:
            # s[t, o] for the K pointers' boundaries ``o`` [B, T, H, K] (o may
            # equal M; s[t, o] = 0 where t - o Bk < 0). Index dims K <= M, so
            # this gathers straight from [B, T, H, M] without an expand.
            return pooled.gather(-1, o.clamp_max(M - 1)) * (t_idx - o * block_size >= 0)

        def mean_score(a: Tensor, b: Tensor) -> Tensor:
            count = (torch.minimum(b * block_size, t_idx + 1) - a * block_size).clamp_min(1)
            return (boundary(a) - boundary(b)) / count.to(u.dtype)

    bits, logits = [], []
    for d in range(depths):
        feats = torch.stack((
            log_len,
            torch.full_like(log_len, d),
            torch.log2((hi - lo).clamp_min(1).to(u.dtype)),
            torch.log2(lo.to(u.dtype) + 1),
        ), -1)
        z = head.logits(u, feats)                                         # [B, T, H, K]
        mid = (lo + hi) // 2
        if pooled is not None:
            z = z + gain * (mean_score(mid, hi) - mean_score(lo, mid))
        with torch.no_grad():
            bit = torch.bernoulli(torch.sigmoid(z)).bool()
        active = (hi - lo) > 1
        lo = torch.where(active & bit, mid, lo)
        hi = torch.where(active & ~bit, mid, hi)
        bits.append(bit)
        logits.append(z)
    if not depths:
        return lo.new_zeros((*lo.shape, 0), dtype=torch.bool), u.new_zeros((*lo.shape, 0))
    return torch.stack(bits, -1), torch.stack(logits, -1)


class PerDepthLinearHead(nn.Module):
    """DIAGNOSTIC controller, not the design: ``z_{t,d} = W_d x_t + c_d`` with
    independent parameters per depth up to ``max_depths`` (the position
    model's ``ptr`` head applied to interval addressing), plus the same
    geometric jump-size prior as the shared head via ``feat_out``/``out_bias``
    so the two heads start from identical bit distributions. Cannot serve a
    context longer than ``Bk * 2^max_depths``; exists only to measure what
    the shared head's parameter sharing costs.
    """

    def __init__(self, dim: int, num_heads: int, num_pointers: int, max_depths: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_pointers = num_pointers
        self.max_depths = max_depths
        self.ctrl = Linear(dim, num_heads * num_pointers * max_depths)
        self.feat_out = nn.Parameter(torch.zeros(num_heads, num_pointers, NUM_FEATURES))
        self.out_bias = nn.Parameter(torch.zeros(num_heads, num_pointers))

    def forward(self, x: Tensor, depths: int, block_size: int) -> Tensor:
        if depths > self.max_depths:
            raise ValueError(f"PerDepthLinearHead built for {self.max_depths} depths, got {depths}")
        B, T = x.size(0), x.size(1)
        z = self.ctrl(x).view(B, T, self.num_heads, self.num_pointers, self.max_depths)[..., :depths]
        z = z.to(torch.promote_types(z.dtype, torch.float32))
        feats = routing_features(T, depths, block_size, x.device, z.dtype)
        direct = torch.einsum("tdf,hkf->thkd", feats, self.feat_out.to(z.dtype))
        return z + direct + self.out_bias.to(z.dtype).unsqueeze(-1)


HEADS = ("shared", "linear")
# ``position``: the routing decision reads x_t and position features only.
# ``pooled``: it also reads the keys (``pooled_scores``); sequential decode only.
ROUTES = ("position", "pooled")



class IntervalPointerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_pointers: int,
        block_size: int,
        width: int,
        head_dim: int = 128,
        backend: str = "flex",
        local_window: int = 0,
        head: str = "shared",
        max_depths: int = 16,
        decode: str = "interval",
        prior: str = "score",
        route: str = "position",
    ):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        if decode not in DECODES:
            raise ValueError(f"decode must be one of {DECODES}, got {decode!r}")
        if decode == "sequential" and head != "shared":
            raise ValueError("sequential decode needs the shared head (per-sample routing features)")
        self.decode = decode
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}, got {prior!r}")
        self.prior = prior
        if route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
        if route == "pooled" and decode != "sequential":
            raise ValueError("pooled routing needs the sequential decoder (the interval must be known per depth)")
        self.route_mode = route
        if dim % head_dim or dim < head_dim:
            raise ValueError(f"dim {dim} must be a positive multiple of head_dim {head_dim}")
        if block_size <= 0 or num_pointers <= 0:
            raise ValueError("block_size and num_pointers must be positive")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if local_window < 0 or local_window % block_size:
            raise ValueError(f"local_window must be a nonnegative multiple of block_size, got {local_window}")
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.num_pointers = num_pointers
        self.block_size = block_size
        self.backend = backend
        self.local_window = local_window
        hdim = self.num_heads * head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.head = (
            IntervalPointerHead(dim, self.num_heads, num_pointers, width)
            if head == "shared"
            else PerDepthLinearHead(dim, self.num_heads, num_pointers, max_depths)
        )
        self.rotary = Rotary(head_dim)
        # per-(head, pointer) gain on the pooled-key term; a vector in the
        # optimizer split despite the [H, K] storage
        self.route_gains = nn.Parameter(torch.ones(self.num_heads, num_pointers)) if route == "pooled" else None

    def candidates_per_query(self, seq_len: int) -> int:
        return self.num_pointers * (num_depths(seq_len, self.block_size) + 1) * self.block_size + self.local_window

    def sample_bits(self, logits: Tensor) -> Tensor:
        # The model IS the sampled path plus its Hamming-1 ball; eval samples
        # too (mode-decoding collapses the ball onto the argmax path and
        # measured worse: 3.1069 vs 3.1038 nats at step 100).
        with torch.no_grad():
            return torch.bernoulli(torch.sigmoid(logits)).bool()

    def qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """``(q, k, v, q_plain, k_plain)``: rotary q/k for attention, the
        rms-normed pre-rotary q/k for pooled routing (position-free)."""
        B, T, H, D = x.size(0), x.size(1), self.num_heads, self.head_dim
        q = self.q(x).view(B, T, H, D)
        k = self.k(x).view(B, T, H, D)
        v = self.v(x).view(B, T, H, D)
        q, k = F.rms_norm(q, (D,)), F.rms_norm(k, (D,))
        return self.rotary(q), self.rotary(k), v, q, k

    def route(self, x: Tensor, q_plain: "Tensor | None" = None, k_plain: "Tensor | None" = None) -> tuple[Tensor, Tensor, Tensor]:
        """``(offsets, log_prior, valid)``, each ``[B, T, H, K, D + 1]``."""
        T = x.size(1)
        depths = num_depths(T, self.block_size)
        blocks = accessible_blocks(T, self.block_size, x.device)
        if self.decode == "sequential":
            pooled = pooled_scores(q_plain, k_plain, self.block_size) if self.route_mode == "pooled" else None
            bits, logits = sequential_route(self.head, x, depths, self.block_size, pooled, self.route_gains)
            offsets, used, valid = decode_intervals(bits, blocks)
        else:
            logits = self.head(x, depths, self.block_size)
            bits = self.sample_bits(logits)
            decode = decode_intervals if self.decode == "interval" else decode_padded
            offsets, used, valid = decode(bits, blocks)
        if self.prior == "leg":
            p = leg_surrogate(logits, bits, used)
            log_prior = p - p.detach()   # zero forward; sigma'(z)-weighted backward
        else:
            log_prior = interval_log_prior(logits, bits, used)
            if self.prior == "straight":
                # zero in the forward (content-only softmax), identity in the backward
                log_prior = log_prior - log_prior.detach()
        return offsets, log_prior, valid

    def prepare(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        q, k, v, q_plain, k_plain = self.qkv(x)
        offsets, log_prior, valid = self.route(x, q_plain, k_plain)
        member, prior = pointer_tables(
            offsets, log_prior, num_blocks(x.size(1), self.block_size),
            self.local_window // self.block_size, valid,
        )
        return q, k, v, member, prior

    def forward(self, x: Tensor):
        B, T, H, D = x.size(0), x.size(1), self.num_heads, self.head_dim
        if self.backend == "flex":
            q, k, v, member, prior = self.prepare(x)
            y = flex_attend(q, k, v, member, prior, self.block_size, scale=SCORE_SCALE)
        else:
            q, k, v, q_plain, k_plain = self.qkv(x)
            offsets, log_prior, valid = self.route(x, q_plain, k_plain)
            Bk = self.block_size
            window = torch.arange(Bk, device=x.device)
            queries = torch.arange(T, device=x.device).view(1, T, 1, 1, 1, 1)
            positions = queries - offsets.unsqueeze(-1) * Bk - window
            valid_tok = (positions >= 0) & valid.unsqueeze(-1)
            log_prior = log_prior.unsqueeze(-1).expand_as(positions)
            C = positions.size(3) * positions.size(4) * Bk
            positions = positions.clamp_min(0).reshape(B, T, H, C)
            valid_tok = valid_tok.reshape(B, T, H, C)
            log_prior = log_prior.reshape(B, T, H, C)
            if self.local_window:
                local_pos = (torch.arange(T, device=x.device).view(1, T, 1, 1)
                             - torch.arange(self.local_window, device=x.device))
                local_pos = local_pos.expand(B, T, H, -1)
                positions = torch.cat((positions, local_pos.clamp_min(0)), -1)
                valid_tok = torch.cat((valid_tok, local_pos >= 0), -1)
                log_prior = torch.cat((log_prior, torch.zeros_like(local_pos, dtype=log_prior.dtype)), -1)
            y = sparse_attend(q, k, v, positions, valid_tok, log_prior, scale=SCORE_SCALE)
        y = y.contiguous().view(B, T, H * D)
        return self.proj(y)


class IPtrBlock(nn.Module):
    def __init__(self, dim: int, num_pointers: int, block_size: int, width: int,
                 mlp_hidden: "int | None" = None, backend: str = "flex",
                 local_window: int = 0, head_dim: int = 128, head: str = "shared",
                 decode: str = "interval", prior: str = "score", route: str = "position"):
        super().__init__()
        self.attn = IntervalPointerAttention(
            dim, num_pointers, block_size, width, head_dim=head_dim,
            backend=backend, local_window=local_window, head=head, decode=decode,
            prior=prior, route=route,
        )
        self.mlp = MLP(dim, mlp_hidden)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class IPtrGPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 num_pointers: int, block_size: int, width: int,
                 mlp_hidden: "int | None" = None, backend: str = "flex",
                 local_window: int = 0, head_dim: int = 128, head: str = "shared",
                 decode: str = "interval", prior: str = "score", route: str = "position"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([
            IPtrBlock(model_dim, num_pointers, block_size, width, mlp_hidden,
                      backend, local_window, head_dim, head, decode, prior, route)
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

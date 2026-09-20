"""pretraining/nanogpt_mini/nanogpt_mini_slot_model.py

Slot-limited KV cache with a learned write policy ("slot memory") for the
nanogpt-mini testbed.

Every sequence owns ``M`` memory slots. The memory entry of token ``t`` is
its per-layer rotary'd key/value pair (what a KV cache stores). After the
top state of position ``t`` is computed, a slot head samples
``sigma_t in {0..M-1, null}``; slot ``sigma_t`` is overwritten with the
entry of ``t`` (``null`` writes nothing). Attention at every layer and every
position ``t`` runs over the token itself plus the at most ``M`` alive
entries, never the full causal history:

    alive[t, s] = max{ i < t : sigma_i = s }   (or -1 when slot s is empty)
    key i visible to query t  <=>  i == t  or  alive[t, sigma_i] == i

Rotary is the ordinary one (keys rotated at their own position, queries at
theirs), so the age ``t - i`` of an entry is native to the score. The trunk
math is the base ``GPT``'s; only the visibility changes.

Training (``mode="policy"``) uses Jacobi passes with shared parameters:

    pass 1  full causal attention (the reference pass), CE_1 per position,
            slot head on its top state samples sigma^(1)
    pass k  attention under the alive table built from sigma^(k-1), CE_k;
            its slot head samples sigma^(k) for pass k+1 (if any)

    A_t     = sum_{u=t+1}^{t+H} gamma^(u-t) (CE_u^(1) - CE_u^(k)) - b
    loss    = CE_1 + mean_{k>=2} CE_k
              - sum_t stopgrad(A_t) log pi_sigma(sigma_t^(k-1) | s_t^(k-1))
              - entropy_coef * sum_t H(pi_sigma(. | s_t^(k-1)))

``b`` is an EMA baseline of the raw discounted credit (one per later pass,
seeded with the first microbatch mean). The slot chosen at the final
position is never consumed and receives no policy gradient. ``detach`` cuts
the slot term's gradient into the trunk (slot head only).

Modes: ``policy`` (above); ``fifo`` (``sigma_t = t mod M``, no head, one
restricted pass: a 64-token sliding window in slot form); ``full`` (no
restriction, one pass, bitwise the base ``GPT`` forward and state dict).

Backends: ``flex`` (production; block-sparse FlexAttention whose BlockMask
is derived from the ``(T, M)`` table without a ``T x T`` scan) and ``dense``
(explicit ``[B, T, T]`` visibility through SDPA; tests and reference only).

``sequential`` evaluates the true token-by-token regime with per-layer slot
banks (the KV cache), sampling ``sigma_t`` from the sequential top state.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pretraining.nanogpt_mini.nanogpt_mini_model import MLP, Linear, RMSNorm, Rotary

MODES = ("policy", "fifo", "full")
BACKENDS = ("flex", "dense")
NO_WRITE = -1
SCORE_SCALE = 0.12
BLOCK_SIZE = 128


# ---------------------------------------------------------------------------
# Alive table, visibility, block mask
# ---------------------------------------------------------------------------

def build_alive_table(sigma: Tensor, slots: int) -> Tensor:
    """``alive[b, t, s] = max{ i < t : sigma[b, i] == s }`` or ``NO_WRITE``.

    ``sigma``: ``[B, T]`` integer with ``NO_WRITE`` for the null write. The
    write at ``t`` is not visible to ``t`` itself (strict ``i < t``).
    """
    if sigma.ndim != 2 or sigma.dtype.is_floating_point:
        raise ValueError("sigma must be an integer [B, T] tensor")
    B, T = sigma.shape
    device = sigma.device
    steps = torch.arange(T, device=device, dtype=torch.long)
    lanes = torch.arange(slots, device=device, dtype=torch.long)
    writes = torch.where(sigma.long()[:, :, None] == lanes[None, None, :],
                         steps[None, :, None], torch.full((), NO_WRITE, device=device, dtype=torch.long))
    after = writes.cummax(dim=1).values                    # includes i <= t
    empty = torch.full((B, 1, slots), NO_WRITE, device=device, dtype=torch.long)
    return torch.cat((empty, after[:, :-1]), dim=1)


def fifo_sigma(batch: int, seq_len: int, slots: int, device: torch.device) -> Tensor:
    """``sigma_t = t mod M``: a ``M``-token sliding window in slot form."""
    return (torch.arange(seq_len, device=device, dtype=torch.long) % slots).expand(batch, seq_len)


def visibility_reference(alive: Tensor, sigma: Tensor) -> Tensor:
    """Dense ``[B, T, T]`` bool: key ``i`` visible to query ``t`` iff
    ``i == t`` or ``alive[t, sigma_i] == i``."""
    B, T, _ = alive.shape
    keys = torch.arange(T, device=alive.device, dtype=torch.long)
    occupant = alive.gather(2, sigma.long().clamp_min(0)[:, None, :].expand(B, T, T))
    written = (sigma >= 0)[:, None, :]
    return torch.eye(T, dtype=torch.bool, device=alive.device)[None] | (written & (occupant == keys[None, None, :]))


def slot_mask_mod(alive: Tensor, sigma: Tensor) -> Callable[..., Tensor]:
    """FlexAttention ``mask_mod`` capturing the ``[B, T, M]`` table and ``[B, T]`` choices."""
    B, T, M = alive.shape
    alive_flat = alive.reshape(-1)
    sigma_flat = sigma.reshape(-1)

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        s = sigma_flat[b * T + kv_idx]
        occupant = alive_flat[(b * T + q_idx) * M + s.clamp_min(0)]
        return (kv_idx == q_idx) | ((s >= 0) & (occupant == kv_idx))

    return mask_mod


def block_presence(alive: Tensor, block_size: int) -> Tensor:
    """``[B, QB, KB]`` bool: the key blocks a query block needs, straight from
    the table (``O(T * M)``): every occupant of every row of the block plus
    the diagonal block for ``i == t``. A superset of the exact block
    structure is sound because ``mask_mod`` runs inside every kept block."""
    B, T, M = alive.shape
    blocks = -(-T // block_size)
    qb = torch.arange(T, device=alive.device, dtype=torch.long) // block_size
    kb = alive.clamp_min(0) // block_size                                   # [B, T, M]
    b = torch.arange(B, device=alive.device, dtype=torch.long)
    flat = ((b[:, None, None] * blocks + qb[None, :, None]) * blocks + kb).reshape(-1)
    counts = torch.zeros(B * blocks * blocks, dtype=torch.int32, device=alive.device)
    counts.scatter_add_(0, flat, (alive >= 0).reshape(-1).to(torch.int32))
    presence = counts.view(B, blocks, blocks) > 0
    diag = torch.eye(blocks, dtype=torch.bool, device=alive.device)
    return presence | diag[None]


def slot_block_mask(alive: Tensor, sigma: Tensor, block_size: int = BLOCK_SIZE):
    """Block-sparse ``BlockMask`` (``[B, 1, QB, KB]`` structure) from the table."""
    from torch.nn.attention.flex_attention import BlockMask

    B, T, _ = alive.shape
    presence = block_presence(alive, block_size)
    kv_num_blocks = presence.sum(dim=-1, dtype=torch.int32)
    kv_indices = torch.argsort((~presence).to(torch.int8), dim=-1, stable=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        kv_num_blocks[:, None], kv_indices[:, None], None, None,
        BLOCK_SIZE=block_size, mask_mod=slot_mask_mod(alive, sigma), seq_lengths=(T, T),
    )


def causal_block_mask(batch: int, seq_len: int, device: torch.device, block_size: int = BLOCK_SIZE):
    """Plain causal ``BlockMask`` (eval-only age statistics of the full mode)."""
    from torch.nn.attention.flex_attention import BlockMask

    blocks = -(-seq_len // block_size)
    presence = torch.tril(torch.ones(blocks, blocks, dtype=torch.bool, device=device))
    presence = presence[None].expand(batch, blocks, blocks)
    kv_num_blocks = presence.sum(dim=-1, dtype=torch.int32)
    kv_indices = torch.argsort((~presence).to(torch.int8), dim=-1, stable=True).to(torch.int32)

    def mask_mod(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx

    return BlockMask.from_kv_blocks(
        kv_num_blocks[:, None].contiguous(), kv_indices[:, None].contiguous(), None, None,
        BLOCK_SIZE=block_size, mask_mod=mask_mod, seq_lengths=(seq_len, seq_len),
    )


_FLEX_ATTENTION = None


def compiled_flex_attention():
    """Lazily compiled ``flex_attention`` (eager flex is the dense math
    reference: it materialises ``[B, H, T, T]`` scores, which at MBS 64 is
    1 GB per layer and an OOM in the backward). Every distinct
    (shape, dtype, kwargs) is a recompile under ``dynamic=False``; Dynamo's
    default limit of 8 would then silently hand later shapes to eager flex,
    so the limit is raised and exceeding it is an error rather than a
    fallback (a production run needs about four variants)."""
    global _FLEX_ATTENTION
    if _FLEX_ATTENTION is None:
        import torch._dynamo.config as dynamo_config
        from torch.nn.attention.flex_attention import flex_attention

        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 64)
        dynamo_config.fail_on_recompile_limit_hit = True
        _FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)
    return _FLEX_ATTENTION


class SlotMask:
    """Visibility of one microbatch: the alive table, the choices, and the
    backend structure (block mask for ``flex``, dense bool mask for
    ``dense``), all built eagerly so the object is immutable afterwards."""

    def __init__(self, alive: Tensor, sigma: Tensor, backend: str):
        if backend not in BACKENDS:
            raise ValueError(f"unknown slot backend {backend!r}")
        self.alive = alive
        self.sigma = sigma
        self.backend = backend
        self._block = slot_block_mask(alive, sigma) if backend == "flex" else None
        self._dense = visibility_reference(alive, sigma) if backend == "dense" else None

    def block(self):
        if self._block is None:
            raise ValueError("block structure is only built for the flex backend")
        return self._block

    def dense(self) -> Tensor:
        if self._dense is None:
            raise ValueError("dense visibility is only built for the dense backend")
        return self._dense


@torch.compiler.disable
def make_slot_mask(sigma: Tensor, slots: int, backend: str) -> SlotMask:
    """Alive table + backend structure, outside the compiled graph."""
    return SlotMask(build_alive_table(sigma, slots), sigma, backend)


@torch.compiler.disable(recursive=False)
def slot_attend(q: Tensor, k: Tensor, v: Tensor, mask: SlotMask | None, backend: str) -> Tensor:
    """``[B, T, H, D]`` attention. ``mask=None`` is the base model's causal
    SDPA call (bitwise the base). Outside the outer compiled graph on
    purpose, as ``nanogpt_mini_ptr_model.flex_attend`` explains."""
    if mask is None:
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           scale=SCORE_SCALE, is_causal=True)
        return y.transpose(1, 2)
    if backend == "flex":
        y = compiled_flex_attention()(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                      block_mask=mask.block(), scale=SCORE_SCALE)
        return y.transpose(1, 2)
    if backend == "dense":
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           attn_mask=mask.dense()[:, None], scale=SCORE_SCALE)
        return y.transpose(1, 2)
    raise ValueError(f"unknown slot backend {backend!r}")


@torch.compiler.disable(recursive=False)
def attention_age_stats(q: Tensor, k: Tensor, mask: SlotMask | None, backend: str) -> tuple[Tensor, Tensor]:
    """Softmax-mass-weighted age of the attended non-self entries.

    Returns ``(sum of mass * age, sum of non-self mass)`` over every
    ``(b, h, t)`` in fp32; their ratio is the mean age. Positions with no
    alive entry put all mass on themselves and contribute nothing to either.
    Flex path: a value channel ``-i`` gives ``sum_i p_ti (t - i)`` after
    adding ``t``; the self mass is ``exp(score_tt - lse_t)``.
    """
    B, T, H, D = q.shape
    qf, kf = q.float(), k.float()
    positions = torch.arange(T, device=q.device, dtype=torch.float32)
    self_score = (qf * kf).sum(-1) * SCORE_SCALE                            # [B, T, H]
    if backend == "dense" or (mask is None and backend != "flex"):
        scores = torch.einsum("bqhd,bkhd->bhqk", qf, kf) * SCORE_SCALE
        visible = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))[None, None] \
            if mask is None else mask.dense()[:, None]
        probs = scores.masked_fill(~visible, float("-inf")).softmax(-1)
        age = (positions[:, None] - positions[None, :]).clamp_min(0)
        weighted = (probs * age[None, None]).sum()
        nonself = (1.0 - probs.diagonal(dim1=-2, dim2=-1)).sum()
        return weighted, nonself
    value = torch.zeros(B, H, T, D, device=q.device, dtype=torch.float32)
    value[..., 0] = -positions
    block = causal_block_mask(B, T, q.device) if mask is None else mask.block()
    out, lse = compiled_flex_attention()(qf.transpose(1, 2), kf.transpose(1, 2), value,
                                         block_mask=block, scale=SCORE_SCALE, return_lse=True)
    weighted = (out[..., 0] + positions[None, None, :]).sum()               # sum_i p (t - i)
    nonself = (1.0 - torch.exp(self_score.transpose(1, 2) - lse)).sum()
    return weighted, nonself


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class SlotAttention(nn.Module):
    """The base ``CausalSelfAttention`` with the visibility as an argument.
    Same parameters, names, and registration order."""

    def __init__(self, dim: int, head_dim: int = 128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Normed, rotary'd ``q, k`` and ``v`` as ``[B, T, H, D]`` at positions ``0..T-1``."""
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        return q, k, v

    def forward(self, x: Tensor, mask: SlotMask | None, backend: str,
                age: list[Tensor] | None = None) -> Tensor:
        B, T = x.size(0), x.size(1)
        q, k, v = self.qkv(x)
        if age is not None:
            weighted, nonself = attention_age_stats(q, k, mask, backend)
            age[0] = age[0] + weighted
            age[1] = age[1] + nonself
        y = slot_attend(q, k, v, mask, backend)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)


class SlotBlock(nn.Module):
    def __init__(self, dim: int, head_dim: int = 128, mlp_hidden: int | None = None):
        super().__init__()
        self.attn = SlotAttention(dim, head_dim)
        self.mlp = MLP(dim, mlp_hidden)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor, mask: SlotMask | None, backend: str, age=None) -> Tensor:
        x = x + self.attn(self.norm1(x), mask, backend, age)
        x = x + self.mlp(self.norm2(x))
        return x


def discounted_future_credit(delta: Tensor, gamma: float, horizon: int) -> Tensor:
    """``A[t] = sum_{u=t+1}^{min(t+H, T-1)} gamma^(u-t) delta[u]`` over the last dim."""
    T = delta.size(-1)
    credit = torch.zeros_like(delta)
    for h in range(1, min(horizon, T - 1) + 1):
        credit[..., :T - h] += (gamma ** h) * delta[..., h:]
    return credit


def sample_slot_choices(logits: Tensor, greedy: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    """Gumbel-max sample (or argmax) over ``[..., M + 1]`` fp32 logits.

    Returns ``(choice in [0, M], log-prob of the choice, entropy)``; index
    ``M`` is the null write.
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    if greedy:
        choice = logits.argmax(dim=-1)
    else:
        gumbel = -torch.log(-torch.log(torch.rand_like(logp)))
        choice = (logp + gumbel).argmax(dim=-1)
    chosen = logp.gather(-1, choice[..., None]).squeeze(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    return choice, chosen, entropy


def injected_slot_choices(logits: Tensor, sigma: Tensor, slots: int) -> tuple[Tensor, Tensor, Tensor]:
    """Log-probability and entropy of a given trajectory (``NO_WRITE`` = null)."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    choice = torch.where(sigma < 0, torch.full_like(sigma, slots), sigma).long()
    chosen = logp.gather(-1, choice[..., None]).squeeze(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    return choice, chosen, entropy


def head_choice_to_sigma(choice: Tensor, slots: int) -> Tensor:
    """Head index ``M`` (null) becomes ``NO_WRITE``."""
    return torch.where(choice >= slots, torch.full_like(choice, NO_WRITE), choice)


class SlotGPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, *, slots: int = 64,
                 mode: str = "policy", passes: int = 2, gamma: float = 0.9, horizon: int = 32,
                 entropy_coef: float = 0.0, detach: bool = False, backend: str = "flex",
                 baseline_decay: float = 0.99, head_dim: int = 128, mlp_hidden: int | None = None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if slots < 1:
            raise ValueError("slots must be positive")
        if mode == "policy" and passes < 2:
            raise ValueError("policy mode needs at least two Jacobi passes")
        if not (0.0 <= gamma <= 1.0) or horizon < 1 or not (0.0 <= baseline_decay < 1.0):
            raise ValueError("gamma in [0, 1], horizon >= 1, baseline_decay in [0, 1)")
        self.slots = slots
        self.mode = mode
        self.passes = passes if mode == "policy" else 1
        self.gamma = gamma
        self.horizon = horizon
        self.entropy_coef = entropy_coef
        self.detach = detach
        self.backend = backend
        self.baseline_decay = baseline_decay
        # trunk registration order == base GPT (seeded init parity)
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([SlotBlock(model_dim, head_dim, mlp_hidden) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        if mode == "policy":
            # fp32 head over the post-final-norm state; zero-init = uniform policy
            self.slot_head = nn.Linear(model_dim, slots + 1)
            nn.init.zeros_(self.slot_head.weight)
            nn.init.zeros_(self.slot_head.bias)
            self.register_buffer("adv_baseline", torch.zeros(passes - 1))
            self.register_buffer("adv_baseline_ready", torch.zeros((), dtype=torch.bool))

    # -- pieces -------------------------------------------------------------

    def trunk(self, x0: Tensor, mask: SlotMask | None, age: list[Tensor] | None = None) -> Tensor:
        x = x0
        for block in self.blocks:
            x = block(x, mask, self.backend, age)
        return self.norm2(x)

    def logits(self, top: Tensor) -> Tensor:
        logits = self.proj(top).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()

    def token_ce(self, top: Tensor, targets: Tensor) -> Tensor:
        """Per-position CE ``[B, T]`` fp32."""
        logits = self.logits(top)
        return F.cross_entropy(logits.reshape(targets.numel(), -1), targets.reshape(-1),
                               reduction="none").view(targets.shape)

    def slot_logits(self, top: Tensor) -> Tensor:
        state = top.detach() if self.detach else top
        return F.linear(state.float(), self.slot_head.weight, self.slot_head.bias)

    def make_mask(self, sigma: Tensor) -> SlotMask:
        return make_slot_mask(sigma, self.slots, self.backend)

    def forward_with_slots(self, inputs: Tensor, targets: Tensor, sigma: Tensor) -> Tensor:
        """Per-position CE ``[B, T]`` under an injected slot trajectory (tests, reference)."""
        x0 = self.norm1(self.embed(inputs))
        return self.token_ce(self.trunk(x0, self.make_mask(sigma)), targets)

    # -- training / evaluation forward --------------------------------------

    def forward(self, inputs: Tensor, targets: Tensor,
                sigma: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        """Returns ``(loss to backward, stats)``. Stats are detached sums over
        the microbatch: ``ce_last`` (headline, last pass), ``ce_full``
        (unrestricted pass), ``null_count``, ``entropy_sum`` (of the head
        that produced the headline trajectory), ``age_num``/``age_den`` (eval
        only), and the policy telemetry in training. ``sigma`` (policy mode,
        tests only) replaces pass 1's sampled trajectory with an injected one
        (its log-probability under the head is still the policy term)."""
        if sigma is not None and self.mode != "policy":
            raise ValueError("an injected slot trajectory only applies to policy mode")
        x0 = self.norm1(self.embed(inputs))
        B, T = targets.shape
        want_age = not self.training
        age = [torch.zeros((), device=inputs.device), torch.zeros((), device=inputs.device)] if want_age else None
        zero = torch.zeros((), device=inputs.device)
        stats: dict[str, Tensor] = {}
        if self.mode == "full":
            top = self.trunk(x0, None, age)
            logits = self.logits(top)
            loss = F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")
            stats.update(ce_last=loss.detach(), ce_full=loss.detach(), null_count=zero, entropy_sum=zero)
        elif self.mode == "fifo":
            sigma = fifo_sigma(B, T, self.slots, inputs.device)
            ce = self.token_ce(self.trunk(x0, self.make_mask(sigma), age), targets)
            loss = ce.sum()
            stats.update(ce_last=loss.detach(), null_count=zero, entropy_sum=zero)
            if want_age:
                with torch.no_grad():
                    stats["ce_full"] = self.token_ce(self.trunk(x0, None), targets).sum()
            else:
                stats["ce_full"] = loss.detach()
        else:
            loss, stats = self._policy_forward(x0, targets, age, sigma)
        if want_age:
            stats["age_num"], stats["age_den"] = age
        return loss, stats

    def _policy_forward(self, x0: Tensor, targets: Tensor, age,
                        injected: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        B, T = targets.shape
        device = targets.device
        valid = (torch.arange(T, device=device) < T - 1).to(torch.float32)[None]   # last sigma never consumed
        top = self.trunk(x0, None)
        ce_full = self.token_ce(top, targets)
        loss = ce_full.sum()
        ref = ce_full.detach()
        if injected is None:
            choice, logp, entropy = sample_slot_choices(self.slot_logits(top))
        else:
            choice, logp, entropy = injected_slot_choices(self.slot_logits(top), injected, self.slots)
        later = []
        adv_mean = adv_std = pg_sum = torch.zeros((), device=device)
        for k in range(2, self.passes + 1):
            last = k == self.passes
            sigma = head_choice_to_sigma(choice, self.slots)
            top = self.trunk(x0, self.make_mask(sigma), age if last else None)
            ce = self.token_ce(top, targets)
            later.append(ce.sum())
            raw = discounted_future_credit(ref - ce.detach(), self.gamma, self.horizon)
            mean = raw[:, :T - 1].mean()
            ready = self.adv_baseline_ready
            baseline = torch.where(ready, self.adv_baseline[k - 2], mean)
            advantage = (raw - baseline) * valid
            if self.training:
                with torch.no_grad():
                    updated = torch.where(ready, self.baseline_decay * self.adv_baseline[k - 2]
                                          + (1 - self.baseline_decay) * mean, mean)
                    self.adv_baseline[k - 2].copy_(updated)
                    if last:
                        self.adv_baseline_ready.fill_(True)
            pg = -(advantage * logp).sum()
            loss = loss + pg
            if self.entropy_coef > 0:
                loss = loss - self.entropy_coef * (entropy * valid).sum()
            if last:
                adv_mean = mean.detach()
                adv_std = raw[:, :T - 1].std().detach()
                pg_sum = pg.detach()
                null_count = (choice >= self.slots).sum()
                entropy_sum = entropy.detach().sum()
                headline = ce.detach().sum()
            else:
                choice, logp, entropy = sample_slot_choices(self.slot_logits(top))
        loss = loss + torch.stack(later).mean()
        stats = dict(ce_last=headline, ce_full=ref.sum(), null_count=null_count, entropy_sum=entropy_sum,
                     adv_mean=adv_mean, adv_std=adv_std, pg_sum=pg_sum,
                     baseline=self.adv_baseline[-1].detach().clone(), sigma=head_choice_to_sigma(choice, self.slots))
        return loss, stats

    # -- sequential (true) regime ---------------------------------------------

    @torch.no_grad()
    def sequential(self, inputs: Tensor, targets: Tensor, choice: str = "sample",
                   sigma: Tensor | None = None) -> dict[str, Tensor]:
        """Token-by-token evaluation with per-layer slot banks (the KV cache).

        ``choice``: ``"sample"`` / ``"greedy"`` from the sequential top state
        (policy mode), ``"fifo"`` (``t mod M``), or ``"inject"`` with a given
        ``sigma`` ``[B, T]``. Returns per-position ``ce`` ``[B, T]`` fp32, the
        ``sigma`` used, and ``null_count``. Eager on purpose.
        """
        if choice in ("sample", "greedy") and self.mode != "policy":
            raise ValueError("sampled sequential choices need the policy head")
        if choice == "inject" and (sigma is None or sigma.shape != inputs.shape):
            raise ValueError("inject needs a [B, T] sigma")
        B, T = inputs.shape
        device = inputs.device
        M = self.slots
        x_all = self.norm1(self.embed(inputs))
        rows = torch.arange(B, device=device)
        banks = []
        for block in self.blocks:
            H, D = block.attn.num_heads, block.attn.head_dim
            banks.append((torch.zeros(B, M, H, D, device=device, dtype=x_all.dtype),
                          torch.zeros(B, M, H, D, device=device, dtype=x_all.dtype)))
        alive = torch.full((B, M), NO_WRITE, device=device, dtype=torch.long)
        # rotary tables via the exact ops of Rotary.forward (bitwise the parallel path)
        freq = self.blocks[0].attn.rotary.angular_freq
        theta = torch.outer(torch.arange(T, dtype=torch.float32, device=device), freq)
        cos_all, sin_all = theta.cos(), theta.sin()
        ce = torch.zeros(B, T, device=device, dtype=torch.float32)
        used = torch.full((B, T), NO_WRITE, device=device, dtype=torch.long)
        if choice == "fifo":
            sigma = fifo_sigma(B, T, M, device)
        for t in range(T):
            x = x_all[:, t:t + 1]                                            # [B, 1, dim]
            written = []
            for layer, block in enumerate(self.blocks):
                attn = block.attn
                h = block.norm1(x)
                q = attn.q(h).view(B, 1, attn.num_heads, attn.head_dim)
                k = attn.k(h).view(B, 1, attn.num_heads, attn.head_dim)
                v = attn.v(h).view(B, 1, attn.num_heads, attn.head_dim)
                q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
                q, k = rotate_at(q, cos_all[t], sin_all[t]), rotate_at(k, cos_all[t], sin_all[t])
                q, k, v = q[:, 0], k[:, 0], v[:, 0]                          # [B, H, D]
                K, V = banks[layer]
                s_self = (q.float() * k.float()).sum(-1, keepdim=True) * SCORE_SCALE      # [B, H, 1]
                s_slot = torch.einsum("bhd,bmhd->bhm", q.float(), K.float()) * SCORE_SCALE
                s_slot = s_slot.masked_fill((alive < 0)[:, None, :], float("-inf"))
                p = torch.cat((s_self, s_slot), -1).softmax(-1)
                y = p[..., :1] * v.float() + torch.einsum("bhm,bmhd->bhd", p[..., 1:], V.float())
                y = y.to(x.dtype).reshape(B, 1, attn.num_heads * attn.head_dim)
                x = x + attn.proj(y)
                x = x + block.mlp(block.norm2(x))
                written.append((k, v))
            top = self.norm2(x)
            ce[:, t] = self.token_ce(top, targets[:, t:t + 1])[:, 0]
            if choice in ("sample", "greedy"):
                head, _, _ = sample_slot_choices(self.slot_logits(top)[:, 0], greedy=(choice == "greedy"))
                sigma_t = head_choice_to_sigma(head, M)
            else:
                sigma_t = sigma[:, t]
            used[:, t] = sigma_t
            write = sigma_t >= 0
            target = sigma_t.clamp_min(0)
            for layer in range(len(self.blocks)):
                K, V = banks[layer]
                k, v = written[layer]
                K[rows, target] = torch.where(write[:, None, None], k, K[rows, target])
                V[rows, target] = torch.where(write[:, None, None], v, V[rows, target])
            alive[rows, target] = torch.where(write, torch.full_like(target, t), alive[rows, target])
        return dict(ce=ce, sigma=used, null_count=(used < 0).sum())


def rotate_at(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """``Rotary.forward``'s arithmetic for one position: ``x`` ``[B, 1, H, D]``,
    ``cos``/``sin`` ``[D/2]`` rows of the batched tables."""
    x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3).type_as(x)

"""pretraining/nanogpt_mini/nanogpt_mini_hashptr_model.py

Content-hashed pointer attention: the position-addressed pointer of
``nanogpt_mini_ptr_model.py`` (soft bits, Bernoulli sampling, Hamming-1 ball,
prior-term gradient) with the address computed from CONTENT instead of
position. Every attention layer is sparse; there is no dense causal layer.

Why: a pointer whose bits are a function of the token alone can only learn
positional statistics ("nearby is useful"); the position-addressed model
drifted to offset 0 with ~60% of attention mass on the 8-token local window.
Where the useful key sits is context-dependent, so the address must be a
function of key content — and then the same pointer works in any context.

Mechanism (per head ``h``, per pointer round ``j`` of ``K``):
  - ``L`` learned hyperplanes ``r[h, j, b, :]`` shared by queries and keys
    (learning-to-hash). Query bit logit ``(q . r_b) / tau``; key bit
    ``sign(k . r_b)``; a key's bucket is its ``L`` hard bits, one bucket per
    round. Both sides hash the PRE-rotary rms-normed q/k so the bucket is
    position-free; scores use the rotary q/k as in the baseline.
  - Training samples the query bits (``torch.bernoulli``); eval takes the
    mode. The query attends to every key whose round-``j`` bucket is the
    sampled bucket or one of its ``L`` single-bit flips (multi-probe LSH),
    union over rounds, plus an optional always-on ``local_window``.
  - Score ``q . k * scale + log(sum_j pi_j(bucket_j(kv)) [in ball_j] +
    [local])`` with ``pi_j`` the query's bit distribution — the Monte-Carlo
    restriction of ``softmax_p(q . k_p * scale + log sum_j pi_j(p))`` to the
    sampled balls. Only the prior term reaches the hyperplanes (through the
    query side; key buckets are hard): mass landing on a flip-``b`` bucket
    pulls the query's bit ``b`` toward it. No REINFORCE, no straight-through.

Cost: keys hash once (``O(N L D)``), queries attend to ``K (L+1)`` buckets;
at uniform occupancy ``N / 2^L`` keys each, i.e. ``K * O(log N)`` keys per
query with ``2^L = N / Bk``. Bucket occupancy is data-dependent — the model
reports the realized mean candidate count (``candidate_stats``). Backends:
  - ``flex``: ``member[B,H,T,K*2^L]``/``prior`` tables plus
    ``key_slots[B,H,K,T]`` feed ``mask_mod``/``score_mod``; the prior table
    receives gradient. Tile skipping needs bucket-sorted keys (not done here;
    at T=1024 compute is ~dense causal either way).
  - ``dense``: ``O(T^2)`` masked reference, CPU-runnable; tests only.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pretraining.nanogpt_mini.nanogpt_mini_model import MLP, Linear, RMSNorm, Rotary
from pretraining.nanogpt_mini.nanogpt_mini_ptr_model import (
    MASKED_SCORE,
    SCORE_SCALE,
    compiled_block_mask_builder,
    compiled_flex_attention,
    pointer_log_prior,
    pointer_offsets,
    pointer_tables,
)

BACKENDS = ("flex", "dense")


def _round_mods(member: Tensor, prior: Tensor, key_slots: Tensor, num_bits: int, local_window: int):
    """flex ``mask_mod``/``score_mod``.

    A key sits in one bucket per round (``key_slots[b, h, j, kv]``, within
    the round's ``2^L`` buckets); it is a candidate if any round's ball
    contains its bucket or it is inside the local window, and its prior is
    ``log(sum_j pi_j [in ball_j] + [local])``.

    flex forbids indexing one gradient-carrying tensor more than once inside
    ``score_mod``, so the shared ``[.., K * 2^L]`` prior is split into one
    ``[.., 2^L]`` table per round (non-members set to ``-inf`` so the
    membership test is folded in); each is indexed exactly once. The Python
    loop over rounds unrolls inside the traced mods.
    """
    B, H, T, _ = member.shape
    num_rounds = key_slots.size(2)
    fold = prior.masked_fill(~member, float("-inf")).view(B, H, T, num_rounds, 1 << num_bits)
    round_priors = [fold[:, :, :, j] for j in range(num_rounds)]
    round_members = [member.view(B, H, T, num_rounds, 1 << num_bits)[:, :, :, j] for j in range(num_rounds)]
    zero = torch.zeros((), dtype=prior.dtype, device=prior.device)
    neg_inf = torch.full((), float("-inf"), dtype=prior.dtype, device=prior.device)

    def mask_mod(b, h, q_idx, kv_idx):
        hit = (q_idx - kv_idx) < local_window
        for j in range(num_rounds):
            hit = hit | round_members[j][b, h, q_idx, key_slots[b, h, j, kv_idx]]
        return (kv_idx <= q_idx) & hit

    def score_mod(score, b, h, q_idx, kv_idx):
        mass = torch.where((q_idx - kv_idx) < local_window, zero, neg_inf)
        for j in range(num_rounds):
            mass = torch.logaddexp(mass, round_priors[j][b, h, q_idx, key_slots[b, h, j, kv_idx]])
        return score + mass

    return mask_mod, score_mod


@torch.compiler.disable(recursive=False)
def flex_attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    member: Tensor,
    prior: Tensor,
    key_slots: Tensor,
    local_window: int,
    scale: float,
) -> Tensor:
    """``dense_attend`` semantics on the fused flex kernel. Compile boundary
    for the same inductor miscompile documented on the position model's
    ``flex_attend`` (captured-buffer gradients inside a whole-model graph)."""
    B, T, H, _ = q.shape
    num_bits = (member.size(-1) // key_slots.size(2)).bit_length() - 1
    mask_mod, score_mod = _round_mods(member, prior, key_slots, num_bits, local_window)
    block_mask = compiled_block_mask_builder()(mask_mod, B, H, T, T, device=q.device)
    out = compiled_flex_attention()(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        score_mod=score_mod, block_mask=block_mask, scale=scale,
    )
    return out.transpose(1, 2)


def shared_slots(key_slots: Tensor, num_slots: int) -> Tensor:
    """Round-local buckets ``[B, H, K, T]`` -> slots on the shared
    ``K * 2^L`` axis of ``member``/``prior`` (round ``j`` offset by ``j * 2^L``)."""
    num_rounds = key_slots.size(2)
    offsets = torch.arange(num_rounds, device=key_slots.device) * (num_slots // num_rounds)
    return key_slots + offsets.view(1, 1, -1, 1)


def candidate_mask(member: Tensor, key_slots: Tensor, local_window: int) -> Tensor:
    """``[B, H, T, T]`` bool: causal keys in any round's ball or the local
    window (the dense form of ``_round_mods``'s ``mask_mod``)."""
    B, H, T = key_slots.size(0), key_slots.size(1), key_slots.size(3)
    slots = shared_slots(key_slots, member.size(-1))
    rel = torch.arange(T, device=member.device).view(T, 1) - torch.arange(T, device=member.device).view(1, T)
    hit = ((rel >= 0) & (rel < local_window)).expand(B, H, T, T)
    for j in range(slots.size(2)):
        hit = hit | member.gather(-1, slots[:, :, j].unsqueeze(2).expand(B, H, T, T))
    return hit & (rel >= 0)


def dense_attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    member: Tensor,
    prior: Tensor,
    key_slots: Tensor,
    local_window: int,
    scale: float,
) -> Tensor:
    """``O(T^2)`` reference: causal attention masked to the candidate set with
    the ``_round_mods`` prior term. Queries with no candidate output zero."""
    B, T, H, _ = q.shape
    score_dtype = torch.promote_types(q.dtype, torch.float32)
    scores = torch.einsum("bthd,bshd->bhts", q, k).to(score_dtype) * scale
    rel = torch.arange(T, device=q.device).view(T, 1) - torch.arange(T, device=q.device).view(1, T)
    local = (rel >= 0) & (rel < local_window)
    mass = torch.where(local, 0.0, float("-inf")).to(score_dtype).expand(B, H, T, T)
    shared = shared_slots(key_slots, member.size(-1))
    for j in range(shared.size(2)):
        slots = shared[:, :, j].unsqueeze(2).expand(B, H, T, T)
        in_ball = member.gather(-1, slots)
        mass = torch.logaddexp(
            mass, torch.where(in_ball, prior.gather(-1, slots).to(score_dtype), float("-inf"))
        )
    allowed = candidate_mask(member, key_slots, local_window)
    scores = scores + torch.where(allowed, mass, 0.0)
    scores = scores.masked_fill(~allowed, MASKED_SCORE)
    weights = torch.softmax(scores, dim=-1) * allowed.any(-1, keepdim=True)
    return torch.einsum("bhts,bshd->bthd", weights.type_as(v), v)


class HashPointerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        num_rounds: int,
        num_bits: int,
        head_dim: int = 128,
        backend: str = "flex",
        local_window: int = 0,
        temperature: float = 1.0,
    ):
        super().__init__()
        if num_rounds <= 0 or num_bits <= 0 or temperature <= 0:
            raise ValueError("num_rounds, num_bits and temperature must be positive")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if not 0 <= local_window <= seq_len:
            raise ValueError(f"local_window must lie in [0, seq_len], got {local_window}")
        if dim % head_dim or dim < head_dim:
            raise ValueError(f"dim {dim} must be a positive multiple of head_dim {head_dim}")
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.num_rounds = num_rounds
        self.num_bits = num_bits
        self.backend = backend
        self.local_window = local_window
        self.temperature = temperature
        hdim = self.num_heads * head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        # Hyperplanes r[h, j, b, :] shared by queries and keys. ``hash`` (not
        # ``proj``/``weight``) so the scripts' init loop treats it explicitly.
        self.hash = nn.Parameter(torch.empty(self.num_heads, num_rounds, num_bits, head_dim))
        self.rotary = Rotary(head_dim)

    @property
    def num_slots(self) -> int:
        return self.num_rounds << self.num_bits

    def sample_bits(self, logits: Tensor) -> Tensor:
        if self.training:
            with torch.no_grad():
                return torch.bernoulli(torch.sigmoid(logits)).bool()
        return logits > 0

    def prepare(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """``(q, k, v, member, prior, key_slots)``: rotary q/k/v for scoring,
        the query's ball tables over the shared ``K * 2^L`` slot axis, and
        each key's round-local bucket ``[B, H, K, T]`` in ``[0, 2^L)``."""
        B, T, H, D = x.size(0), x.size(1), self.num_heads, self.head_dim
        q = F.rms_norm(self.q(x).view(B, T, H, D), (D,))
        k = F.rms_norm(self.k(x).view(B, T, H, D), (D,))
        v = self.v(x).view(B, T, H, D)
        hash_weight = self.hash.type_as(q)
        logits = torch.einsum("bthd,hjld->bthjl", q, hash_weight)
        logits = logits.to(torch.promote_types(logits.dtype, torch.float32)) / self.temperature
        rounds = (torch.arange(self.num_rounds, device=x.device) << self.num_bits)
        with torch.no_grad():
            key_bits = torch.einsum("bthd,hjld->bhjtl", k, hash_weight) > 0
            powers = torch.pow(2, torch.arange(self.num_bits, device=x.device))
            key_slots = (key_bits.long() * powers).sum(-1)
        bits = self.sample_bits(logits)
        log_prior = pointer_log_prior(logits, bits)
        slots = pointer_offsets(bits) + rounds.view(1, 1, 1, -1, 1)
        member, prior = pointer_tables(slots, log_prior, self.num_slots)
        return self.rotary(q), self.rotary(k), v, member, prior, key_slots

    @torch.no_grad()
    def candidate_stats(self, x: Tensor) -> dict[str, float]:
        """Realized candidate-set size (mean keys per query and fraction of
        the causal set), for logging bucket balance."""
        _, _, _, member, _, key_slots = self.prepare(x)
        allowed = candidate_mask(member, key_slots, self.local_window)
        T = x.size(1)
        causal = torch.arange(1, T + 1, device=x.device, dtype=torch.float32)
        per_query = allowed.sum(-1, dtype=torch.float32)
        return {
            "keys_per_query": float(per_query.mean()),
            "causal_fraction": float((per_query / causal).mean()),
        }

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        if T != self.seq_len:
            raise ValueError(f"HashPointerAttention built for seq_len {self.seq_len}, got {T}")
        q, k, v, member, prior, key_slots = self.prepare(x)
        attend = flex_attend if self.backend == "flex" else dense_attend
        y = attend(q, k, v, member, prior, key_slots, self.local_window, SCORE_SCALE)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)


class HashPtrBlock(nn.Module):
    def __init__(self, dim: int, seq_len: int, num_rounds: int, num_bits: int,
                 mlp_hidden: "int | None" = None, backend: str = "flex",
                 local_window: int = 0, temperature: float = 1.0, head_dim: int = 128):
        super().__init__()
        self.attn = HashPointerAttention(
            dim, seq_len, num_rounds, num_bits, head_dim=head_dim, backend=backend,
            local_window=local_window, temperature=temperature,
        )
        self.mlp = MLP(dim, mlp_hidden)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HashPtrGPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, seq_len: int,
                 num_rounds: int, num_bits: int, mlp_hidden: "int | None" = None,
                 backend: str = "flex", local_window: int = 0, temperature: float = 1.0,
                 head_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([
            HashPtrBlock(model_dim, seq_len, num_rounds, num_bits, mlp_hidden,
                         backend, local_window, temperature, head_dim)
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

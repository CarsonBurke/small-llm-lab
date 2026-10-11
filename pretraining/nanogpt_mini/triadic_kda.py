"""pretraining/nanogpt_mini/triadic_kda.py

Triadic KDA: the KDA delta rule on a third-order state (arXiv:2609.36529).

Each head keeps ``E`` state slices ``S[e]`` of the ordinary KDA shape
``[Dv, Dk]``. A token addresses them with its key ``k`` and a second key
``k2`` over the slice axis, and reads with a query ``q`` and a second query
``q2``. Flattening ``(e, j)`` gives plain KDA with the joint key
``k2 (x) k`` of width ``E * Dk``, so the state holds ``E`` times as many
associations while the projections grow by ``2 E`` columns per head.

Recurrence (per step t, per head, fp32; state ``[B, H, E, Dv, Dk]``):

    S[e]  <- S[e] * exp(g2[e]) * exp(g)        (slice decay x channel decay)
    u     <- beta * (v - sum_e k2[e] S[e] k)   (delta-rule residual)
    S[e]  <- S[e] + k2[e] * u k^T
    o     <- sum_e q2[e] S[e] q

``k``/``q`` are unit vectors (``q`` carries the readout scale), ``k2``/``q2``
are unit and non-negative. The decay is a Kronecker product of KDA's
channel-wise factor and a per-slice scalar. That is what keeps the layer
chunk-parallel at the cost of ordinary KDA plus the state traffic: within a
chunk every pairwise term factorizes,

    sum_{e,j} k2_r[e] k2_s[e] k_r[j] k_s[j] D_{r<-s}[e, j]  =  R_rs * G_rs,

into the ``[C, C]`` decayed Gram of the ``Dk``-wide keys and a ``[C, C]``
slice term, so only the reads and writes of the state scale with ``E``.

With ``E = 1`` and ``k2 = q2 = 1``, ``g2 = 0`` this is exactly KDA
(``reference_kda_recurrence``); tests pin that, and pin the chunk form
against the step form.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pretraining.nanogpt_mini.nanogpt_mini_kda_model import (
    BiasFreeLinear,
    KimiDeltaAttention,
    ShortConv,
    _l2norm,
    kda_decay_gate,
)

CHUNK_SIZE = 64
# Rows of a chunk are grouped so the largest positive exponent the decayed
# Gram forms is GRAM_BLOCK * |lower bound| = 80 < log(FLT_MAX) = 88.7.
GRAM_BLOCK = 16


def triadic_delta_update(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q2: Tensor,
    k2: Tensor,
    log_decay: Tensor,
    slice_log_decay: Tensor,
    beta: Tensor,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
    """One step, functional: returns ``(o, updated_state)``.

    q/k: ``[B, H, Dk]``; v: ``[B, H, Dv]``; q2/k2: ``[B, H, E]``; log decays
    ``[B, H, Dk]`` and ``[B, H, E]``; beta ``[B, H]``; state
    ``[B, H, E, Dv, Dk]``.
    """
    decayed = (
        state
        * slice_log_decay.exp()[:, :, :, None, None]
        * log_decay.exp()[:, :, None, None, :]
    )
    stored = torch.einsum("bhevj,bhe,bhj->bhv", decayed, k2, k)
    residual = beta[:, :, None] * (v - stored)
    updated = decayed + torch.einsum("bhv,bhe,bhj->bhevj", residual, k2, k)
    return torch.einsum("bhevj,bhe,bhj->bhv", updated, q2, q), updated


def triadic_reference_recurrence(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q2: Tensor,
    k2: Tensor,
    log_decay: Tensor,
    slice_log_decay: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Step-by-step oracle over ``[B, T, H, ...]`` inputs."""
    B, T, H, Dk = k.shape
    state = k.new_zeros(B, H, k2.size(-1), v.size(-1), Dk)
    if initial_state is not None:
        state = state + initial_state
    outputs = []
    for t in range(T):
        o, state = triadic_delta_update(
            q[:, t], k[:, t], v[:, t], q2[:, t], k2[:, t],
            log_decay[:, t], slice_log_decay[:, t], beta[:, t], state,
        )
        outputs.append(o)
    return torch.stack(outputs, dim=1), state


def decayed_gram(left: Tensor, right: Tensor, cum_log_decay: Tensor) -> Tensor:
    """``G[r, s] = sum_j left_r[j] right_s[j] exp(c_r[j] - c_s[j])``.

    Inputs are ``[..., C, D]`` with ``c`` the within-chunk cumulative log
    decay. Entries with ``s > r`` are not meaningful and are returned as
    zero. The exponent is split around the cumulative decay just before each
    block of ``GRAM_BLOCK`` rows, so one GEMM per block forms the result and
    no factor exceeds ``exp(GRAM_BLOCK * |per-step bound|)``.
    """
    C, D = left.shape[-2:]
    blocks = C // GRAM_BLOCK
    batch = left.shape[:-2]
    # c just before each block's first row; zero before the chunk's first.
    reference = F.pad(cum_log_decay, (0, 0, 1, 0))[..., :C:GRAM_BLOCK, :]
    rows = left.unflatten(-2, (blocks, GRAM_BLOCK)) * (
        cum_log_decay.unflatten(-2, (blocks, GRAM_BLOCK))
        - reference[..., None, :]
    ).exp()  # [..., blocks, GRAM_BLOCK, D], every factor <= 1
    # Columns past a block's last row would need exp(+large); they belong to
    # the masked triangle, so their exponent is sent to -inf instead.
    position = torch.arange(C, device=left.device)
    block_end = (torch.arange(blocks, device=left.device) + 1) * GRAM_BLOCK
    visible = position[None, :] < block_end[:, None]  # [blocks, C]
    exponent = reference[..., :, None, :] - cum_log_decay[..., None, :, :]
    exponent = exponent.masked_fill(~visible[..., None], -torch.inf)
    columns = right[..., None, :, :] * exponent.exp()  # [..., blocks, C, D]
    gram = rows @ columns.transpose(-1, -2)  # [..., blocks, GRAM_BLOCK, C]
    return gram.reshape(*batch, C, C).tril()


def slice_gram(left: Tensor, right: Tensor, cum_log_decay: Tensor) -> Tensor:
    """``R[r, s] = sum_e left_r[e] right_s[e] exp(c_r[e] - c_s[e])``, s <= r.

    The slice axis is small, so every factor is the exponential of an exact
    non-positive difference and no bound on the slice decay is needed.
    """
    difference = cum_log_decay[..., :, None, :] - cum_log_decay[..., None, :, :]
    C = left.size(-2)
    causal = torch.ones(C, C, dtype=torch.bool, device=left.device).tril()
    weight = difference.masked_fill(~causal[..., None], -torch.inf).exp()
    return torch.einsum("...re,...se,...rse->...rs", left, right, weight)


def triadic_chunk_kda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q2: Tensor,
    k2: Tensor,
    log_decay: Tensor,
    slice_log_decay: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[Tensor, Tensor]:
    """Chunk-parallel triadic KDA; same contract as the reference recurrence.

    Shapes: q/k ``[B, T, H, Dk]``, v ``[B, T, H, Dv]``, q2/k2
    ``[B, T, H, E]``, log decays ``[B, T, H, Dk]`` / ``[B, T, H, E]``, beta
    ``[B, T, H]``. Returns ``(o [B, T, H, Dv], final_state
    [B, H, E, Dv, Dk])``. Per-step channel log decay must stay above
    ``-log(FLT_MAX) / GRAM_BLOCK`` (KDA's safe gate bound of -5 does).

    Everything pairwise within a chunk is computed for all chunks at once;
    only the chunk-boundary state is carried by a loop.
    """
    if chunk_size % GRAM_BLOCK:
        raise ValueError(f"chunk_size must be a multiple of {GRAM_BLOCK}")
    B, T, H, Dk = k.shape
    Dv, E = v.size(-1), k2.size(-1)
    C = chunk_size
    pad = -T % C
    if pad:
        # No-op rows: zero keys write nothing and zero log decay keeps the
        # state, so the final state is the one after the last real token.
        q, k, v, q2, k2, log_decay, slice_log_decay = (
            F.pad(x, (0, 0, 0, 0, 0, pad))
            for x in (q, k, v, q2, k2, log_decay, slice_log_decay)
        )
        beta = F.pad(beta, (0, 0, 0, pad))
    N = (T + pad) // C

    def chunked(x: Tensor) -> Tensor:  # [B, T, H, ...] -> [B, H, N, C, ...]
        return x.unflatten(1, (N, C)).movedim(3, 1)

    q, k, v, q2, k2 = map(chunked, (q, k, v, q2, k2))
    beta = chunked(beta[..., None])  # [B, H, N, C, 1]
    c = chunked(log_decay).cumsum(-2)
    c2 = chunked(slice_log_decay).cumsum(-2)

    # Pairwise terms: the delta-rule system and the intra-chunk readout.
    system = beta * (decayed_gram(k, k, c) * slice_gram(k2, k2, c2)).tril(-1)
    readout = decayed_gram(q, k, c) * slice_gram(q2, k2, c2)

    # State traffic: reads see the chunk-start state decayed to their own
    # position; writes are decayed from their position to the chunk end.
    decay, slice_decay = c.exp(), c2.exp()
    read_keys = torch.cat((k * decay, q * decay), dim=-2)  # [B, H, N, 2C, Dk]
    read_k2, read_q2 = k2 * slice_decay, q2 * slice_decay
    write_keys = k * (c[..., -1:, :] - c).exp()
    write_k2 = k2 * (c2[..., -1:, :] - c2).exp()
    carry = slice_decay[..., -1, :, None, None] * decay[..., -1, None, None, :]

    state = k.new_zeros(B, H, E, Dv, Dk)
    if initial_state is not None:
        state = state + initial_state
    outputs = []
    for n in range(N):
        # One GEMM reads every slice for the chunk's keys and queries.
        reads = (
            state.flatten(2, 3) @ read_keys[:, :, n].transpose(-1, -2)
        ).unflatten(2, (E, Dv))  # [B, H, E, Dv, 2C]
        stored = torch.einsum("bhevr,bhre->bhrv", reads[..., :C], read_k2[:, :, n])
        carried = torch.einsum("bhevr,bhre->bhrv", reads[..., C:], read_q2[:, :, n])
        residual = torch.linalg.solve_triangular(
            system[:, :, n],
            beta[:, :, n] * (v[:, :, n] - stored),
            upper=False,
            unitriangular=True,
        )  # [B, H, C, Dv]
        outputs.append(carried + readout[:, :, n] @ residual)
        writes = torch.einsum("bhsv,bhse->bhevs", residual, write_k2[:, :, n])
        state = state * carry[:, :, n] + (
            writes.flatten(2, 3) @ write_keys[:, :, n]
        ).unflatten(2, (E, Dv))
    o = torch.stack(outputs, dim=2).movedim(1, 3).flatten(1, 2)  # [B, T, H, Dv]
    return o[:, :T], state


# Initial slice half-lives, log-spaced so the slices of a head start with
# distinct identities: equal slices under equal second keys are a symmetric
# saddle that no gradient separates.
SLICE_HALF_LIFE_RANGE = (32.0, 8192.0)


class SecondKeyConv(ShortConv):
    """Short convolution of the second key/query: softplus keeps slice
    weights non-negative, so a read cannot cancel what a write stored."""

    activation = staticmethod(F.softplus)


class TriadicKimiDeltaAttention(KimiDeltaAttention):
    """KDA mixer with ``slices`` state slices per head.

    Adds a second key and query over the slice axis and one decay gate per
    slice; every other parameter is KDA's, under KDA's names.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 128,
        num_heads: int = 3,
        conv_size: int = 4,
        full_rank_gate: bool = False,
        slices: int = 4,
    ):
        super().__init__(dim, head_dim, num_heads, conv_size, full_rank_gate)
        if slices < 2:
            raise ValueError("a triadic mixer needs at least two slices")
        self.slices = slices
        self.slice_projection_size = num_heads * slices
        self.q2_proj = BiasFreeLinear(dim, self.slice_projection_size)
        self.k2_proj = BiasFreeLinear(dim, self.slice_projection_size)
        self.q2_conv1d = SecondKeyConv(self.slice_projection_size, conv_size)
        self.k2_conv1d = SecondKeyConv(self.slice_projection_size, conv_size)
        self.slice_gate_proj = BiasFreeLinear(dim, self.slice_projection_size)
        self.slice_A_log = nn.Parameter(torch.zeros(num_heads))
        self.slice_dt_bias = nn.Parameter(torch.empty(self.slice_projection_size))
        self.reset_slice_gates()

    @torch.no_grad()
    def reset_slice_gates(self) -> None:
        """Bias each slice's gate to its log-spaced initial half-life."""
        low, high = SLICE_HALF_LIFE_RANGE
        half_life = torch.logspace(
            torch.log10(torch.tensor(low)), torch.log10(torch.tensor(high)),
            self.slices,
        )
        # half-life h  <=>  per-step log decay -ln2 / h = bound * sigmoid(bias)
        fraction = torch.log(torch.tensor(2.0)) / (5.0 * half_life)
        self.slice_A_log.zero_()
        self.slice_dt_bias.copy_(torch.logit(fraction).repeat(self.num_heads))

    def _slice_shape(self, x: Tensor) -> tuple[int, ...]:
        return (*x.shape[:-1], self.num_heads, self.slices)

    def _head_shape(self, x: Tensor) -> tuple[int, ...]:
        return (*x.shape[:-1], self.num_heads, self.head_dim)

    def _controls(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """fp32 ``(log_decay, slice_log_decay, beta)`` for inputs ``x``."""
        log_decay = kda_decay_gate(
            self.f_b_proj(self.f_a_proj(x)).view(self._head_shape(x)),
            self.A_log, self.dt_bias,
        )
        slice_log_decay = kda_decay_gate(
            self.slice_gate_proj(x).view(self._slice_shape(x)),
            self.slice_A_log, self.slice_dt_bias,
        )
        return log_decay, slice_log_decay, torch.sigmoid(self.b_proj(x).float())

    def forward(
        self,
        x: Tensor,
        key_valid: Tensor | None = None,
        initial_state: Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, ...] | None]:
        """Full-sequence mixer output; see ``KimiDeltaAttention.forward``.

        Caches, when requested, are ``(conv_q, conv_k, conv_v, conv_q2,
        conv_k2, state)`` with state ``[B, H, E, Dv, Dk]`` in fp32.
        """
        B, T, _ = x.shape
        projected = [
            proj(x) for proj in
            (self.q_proj, self.k_proj, self.v_proj, self.q2_proj, self.k2_proj)
        ]
        if key_valid is not None:
            # Zeroed keys write nothing; softplus(0) second keys are then
            # multiplied by a zero key in every state update.
            valid = key_valid[..., None].to(x.dtype)
            projected = [p * valid for p in projected]
        q_in, k_in, v_in, q2_in, k2_in = projected
        heads, slices = self._head_shape(x), self._slice_shape(x)
        log_decay, slice_log_decay, beta = self._controls(x)
        with torch.autocast(x.device.type, enabled=False):
            o, state = triadic_chunk_kda(
                _l2norm(self.q_conv1d(q_in).float().view(heads))
                * self.head_dim**-0.5,
                _l2norm(self.k_conv1d(k_in).float().view(heads)),
                self.v_conv1d(v_in).float().view(heads),
                _l2norm(self.q2_conv1d(q2_in).float().view(slices)),
                _l2norm(self.k2_conv1d(k2_in).float().view(slices)),
                log_decay, slice_log_decay, beta,
                initial_state=initial_state,
            )
        y = self.o_norm(o.to(x.dtype), self._output_gate(x))
        out = self.o_proj(y.reshape(B, T, self.projection_size))
        if not output_final_state:
            return out, None
        convs = (self.q_conv1d, self.k_conv1d, self.v_conv1d,
                 self.q2_conv1d, self.k2_conv1d)
        return out, (
            *(conv.final_state(p) for conv, p in zip(convs, projected, strict=True)),
            state,
        )

    def step(self, x: Tensor, caches: tuple[Tensor, ...]) -> Tensor:
        """One-token decode step over ``[B, 1, dim]``; caches mutate in place."""
        x_flat = x.squeeze(1)
        *conv_caches, state = caches
        convs = (self.q_conv1d, self.k_conv1d, self.v_conv1d,
                 self.q2_conv1d, self.k2_conv1d)
        projs = (self.q_proj, self.k_proj, self.v_proj, self.q2_proj, self.k2_proj)
        q, k, v, q2, k2 = (
            conv.step(proj(x_flat), cache).float()
            for conv, proj, cache in zip(convs, projs, conv_caches, strict=True)
        )
        heads, slices = self._head_shape(x_flat), self._slice_shape(x_flat)
        log_decay, slice_log_decay, beta = self._controls(x_flat)
        with torch.autocast(x.device.type, enabled=False):
            o, updated = triadic_delta_update(
                _l2norm(q.view(heads)) * self.head_dim**-0.5,
                _l2norm(k.view(heads)), v.view(heads),
                _l2norm(q2.view(slices)), _l2norm(k2.view(slices)),
                log_decay, slice_log_decay, beta, state,
            )
        state.copy_(updated)
        y = self.o_norm(o.to(x.dtype), self._output_gate(x_flat))
        return self.o_proj(y.reshape(-1, 1, self.projection_size))

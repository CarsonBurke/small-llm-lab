"""Gated-delta memory carry: a bounded, trained memory over detached beliefs.

Reasoning mode ``memory`` is the successor arm to the deterministic hidden
carry (``carry``). That carry re-injects the whole previous belief into every
generated token's input through an unbounded full-rank map, so the update can
only ever see one step of its effect and nothing bounds what it feeds back.
Here the carried information instead passes through a per-head gated delta
rule (GDN) memory with a separate read and write:

- **Write, at the network output.** The detached post-final-norm belief
  ``h_t`` that produced a token is written into an fp32 state
  ``S`` (per head ``K x V``, the fla layout)::

      h^      = h_t / rms(h_t)
      k_t     = l2norm(W_k h^)            v_t = W_v h^
      beta_t  = sigmoid(w_b . h^ + b_b)   alpha_t = sigmoid(w_a . h^ + b_a)
      S_t     = alpha_t S_{t-1} + k_t (beta_t (v_t - alpha_t S_{t-1}^T k_t))^T

  With ``|k| = 1`` and ``alpha, beta`` in (0, 1), the old state passes
  through ``alpha (I - beta k k^T)``, whose operator norm is at most
  ``alpha``, and the write adds at most ``beta |v|``. So
  ``||S_t|| <= alpha_t ||S_{t-1}|| + beta_t |v_t|``, and from zero
  ``||S|| <= max(beta |v|) / (1 - max alpha)`` for any trajectory length.
  (It is *not* bounded by the largest value written: nearly parallel keys
  carrying conflicting values can amplify it up to the decay bound.) The old
  carry has no bound at all.
- **Read, at the network input.** The next token's input reads the state
  through a query built from its own embedding and the belief that produced
  it, and adds a zero-initialized projection of the result::

      q_t     = l2norm(W_q [e(x_t) / rms, h_{t-1} / rms])
      input_t = e(x_t) + W_o (S^T q_t)

  ``W_o`` starts at zero, so a fresh memory policy is exactly ``cot`` at
  step 0. With ``alpha = 0, beta = 1, q = k`` the read is ``W_o W_v h^``:
  the old carry restricted to rank ``heads * head_dim``.

Prompt beliefs are written too (prefill produces them in parallel, and one
scan folds them into the state), but the prompt never reads: a read there
would make the prefill sequential. The state resets per trajectory.

Replay is where this differs in kind from the old carry. The stored beliefs
are sampled data, so the memory recurrence over them is linear in ``S`` and
one chunked parallel scan (fla ``chunk_gated_delta_rule``) differentiates the
loss at every read through every earlier write: exact backpropagation through
time inside the memory -- into ``W_k, W_v``, both gates, ``W_q`` and ``W_o`` --
with no graph capture and no sequential replay. The trunk's own dependence on
the memory (read -> trunk -> next belief -> write) stays cut, as it is for
every stored action.

The critic reads the actor's memory (detached) through its own reader, so
its value at a slot conditions on everything the actor will read next and
the TD error carries the value of each write.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_MEMORY_HEADS = 4
DEFAULT_MEMORY_HEAD_DIM = 64
# Initial decay timescales (tokens), spread geometrically across heads: the
# GDN/Mamba recipe of covering several horizons at init instead of making
# every head forget at one rate. 16 tokens is a local phrase; 1024 spans most
# of a 2048-token response.
MEMORY_MIN_TIMESCALE = 16.0
MEMORY_MAX_TIMESCALE = 1024.0
_RMS_EPS = 1e-6


def _unit_rms(x: Tensor) -> Tensor:
    """Parameter-free RMS normalization in the input's dtype."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + _RMS_EPS)


def _head_l2norm(x: Tensor, heads: int) -> Tensor:
    """Split the last dimension into heads and L2-normalize each."""
    x = x.unflatten(-1, (heads, -1))
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + _RMS_EPS)


def initial_decay_bias(heads: int) -> Tensor:
    """Per-head decay-gate bias giving geometric timescales at init.

    ``alpha = sigmoid(b) = 1 - 1/tau`` holds a write for ``tau`` tokens on
    average, so ``b = log(tau - 1)``.
    """
    if heads == 1:
        timescales = torch.tensor([MEMORY_MAX_TIMESCALE])
    else:
        timescales = torch.logspace(
            math.log10(MEMORY_MIN_TIMESCALE),
            math.log10(MEMORY_MAX_TIMESCALE),
            heads,
        )
    return torch.log(timescales - 1.0)


class MemoryWriter(nn.Module):
    """Belief -> (key, value, log decay, write strength). Actor-owned."""

    def __init__(self, model_dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        width = heads * head_dim
        self.key = nn.Linear(model_dim, width, bias=False)
        self.value = nn.Linear(model_dim, width, bias=False)
        self.write_gate = nn.Linear(model_dim, heads)
        self.decay_gate = nn.Linear(model_dim, heads)
        with torch.no_grad():
            std = model_dim**-0.5
            self.key.weight.normal_(std=std)
            self.value.weight.normal_(std=std)
            # Gates start data-independent: half-strength writes, geometric
            # timescales. No writer parameter receives gradient until the
            # zero-initialized read projection leaves zero after its first
            # optimizer step; every path to the loss runs through it.
            self.write_gate.weight.zero_()
            self.write_gate.bias.zero_()
            self.decay_gate.weight.zero_()
            self.decay_gate.bias.copy_(initial_decay_bias(heads))

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """fp32 ``(k, v, log_decay, beta)`` for beliefs ``(..., D)``.

        ``k``/``v`` are ``(..., H, K)``; the gates ``(..., H)``. The caller
        owns detaching: the actor's writer trains through the scan, but the
        belief it reads is sampled data.
        """
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            # Separate GEMMs on purpose: one concatenated projection saved
            # two launches but let Inductor fold its strided split into the
            # decode write's pass over the state, 6x slower (bench_graph_decode).
            unit = _unit_rms(hidden.float())
            k = _head_l2norm(F.linear(unit, self.key.weight), self.heads)
            v = F.linear(unit, self.value.weight).unflatten(
                -1, (self.heads, self.head_dim)
            )
            beta = torch.sigmoid(
                F.linear(unit, self.write_gate.weight, self.write_gate.bias)
            )
            log_decay = F.logsigmoid(
                F.linear(unit, self.decay_gate.weight, self.decay_gate.bias)
            )
        return k, v, log_decay, beta


class MemoryReader(nn.Module):
    """(token latent, carried belief, state) -> zero-init input residual.

    The actor and the critic each own one; they read the same (actor) state.
    """

    def __init__(self, model_dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        width = heads * head_dim
        self.query = nn.Linear(2 * model_dim, width, bias=False)
        self.out = nn.Linear(width, model_dim, bias=False)
        with torch.no_grad():
            self.query.weight.normal_(std=(2 * model_dim) ** -0.5)
            # Zero output: the read contributes nothing until the policy
            # gradient finds a use for it, so step 0 is the token-only policy.
            self.out.weight.zero_()

    def queries(self, token_latent: Tensor, hidden: Tensor) -> Tensor:
        """Unit-norm fp32 queries ``(..., H, K)``."""
        with torch.autocast(device_type=token_latent.device.type, enabled=False):
            features = torch.cat(
                (_unit_rms(token_latent.float()), _unit_rms(hidden.float())),
                dim=-1,
            )
            return _head_l2norm(F.linear(features, self.query.weight), self.heads)

    def residual(self, reads: Tensor) -> Tensor:
        """fp32 ``(..., D)`` input residual from per-head reads ``(..., H, V)``."""
        with torch.autocast(device_type=reads.device.type, enabled=False):
            return F.linear(reads.float().flatten(-2), self.out.weight)

    def add(self, base: Tensor, reads: Tensor) -> Tensor:
        """``base + residual`` with the residual rounded once to base's dtype.

        The same cast order at decode and replay, as ``CombinedEmbedding``
        does for the old carry.
        """
        return base + self.residual(reads).to(base.dtype)


def delta_write(
    state: Tensor, k: Tensor, v: Tensor, log_decay: Tensor, beta: Tensor
) -> Tensor:
    """One fp32 gated-delta write; ``state`` is ``(R, H, K, V)``.

    The recall reads the undecayed state (``k^T (alpha S) = alpha k^T S``),
    so no decayed copy is materialized, and it is a broadcast product and
    sum rather than a GEMV: the recall of column ``(r, h, :, v)`` and that
    column's update form one reduction group, so compiled, the whole write
    is a single persistent-reduction pass over ``S`` -- one read, one store
    into the arena's buffer -- instead of a cuBLAS GEMV pass plus an update
    pass.
    """
    with torch.autocast(device_type=state.device.type, enabled=False):
        alpha = log_decay.exp()
        recalled = alpha[..., None] * (k[..., :, None] * state).sum(-2)
        update = beta[..., None] * (v - recalled)
        return (
            alpha[..., None, None] * state
            + k[..., :, None] * update[..., None, :]
        )


def delta_read(state: Tensor, q: Tensor) -> Tensor:
    """``S^T q`` per head: ``(R, H, V)`` in fp32.

    A product and sum, like ``delta_write``'s recall: Inductor emits one
    bandwidth-bound reduction over ``S`` instead of a batched cuBLAS GEMV.
    """
    with torch.autocast(device_type=state.device.type, enabled=False):
        return (q[..., :, None] * state).sum(-2)


def reference_delta_scan(
    q: Tensor, k: Tensor, v: Tensor, log_decay: Tensor, beta: Tensor
) -> tuple[Tensor, Tensor]:
    """Sequential write-then-read over ``(B, T, ...)``: the scan's definition.

    Slot ``t`` first writes ``(k_t, v_t)`` and then reads with ``q_t``, so its
    read sees every write up to and including its own -- fla's
    ``chunk_gated_delta_rule`` convention. Returns ``(reads (B,T,H,V),
    final_state (B,H,K,V))``. A masked slot passes ``log_decay = beta = 0``,
    which leaves the state untouched.
    """
    batch, length, heads, key_dim = k.shape
    state = v.new_zeros((batch, heads, key_dim, v.size(-1)))
    reads = []
    for t in range(length):
        state = delta_write(
            state, k[:, t], v[:, t], log_decay[:, t], beta[:, t]
        )
        reads.append(delta_read(state, q[:, t]))
    return torch.stack(reads, dim=1), state


@torch.compiler.disable
def _fla_delta_scan(
    q: Tensor, k: Tensor, v: Tensor, log_decay: Tensor, beta: Tensor
) -> tuple[Tensor, Tensor]:
    """fla's chunked scan, as one varlen sequence per row.

    The flattened ``[1, B*T]`` call with uniform ``cu_seqlens`` is the same
    per-row math (states reset at every boundary) with one kernel cache key
    for every replay-shard row count, the KDA mixer's recipe. Fenced off from
    Dynamo like that mixer: the chunk kernel graph-breaks by design.
    """
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    batch, length = q.shape[:2]
    bounds_cpu = torch.arange(batch + 1, dtype=torch.long) * length
    bounds = torch.arange(batch + 1, device=q.device, dtype=torch.long) * length
    flat = lambda x: x.reshape(1, batch * length, *x.shape[2:]).contiguous()
    reads, final_state = chunk_gated_delta_rule(
        q=flat(q),
        k=flat(k),
        v=flat(v),
        g=flat(log_decay),
        beta=flat(beta),
        scale=1.0,
        output_final_state=True,
        cu_seqlens=bounds,
        cu_seqlens_cpu=bounds_cpu,
    )
    return reads.reshape(batch, length, *reads.shape[2:]), final_state


def delta_scan(
    q: Tensor, k: Tensor, v: Tensor, log_decay: Tensor, beta: Tensor
) -> tuple[Tensor, Tensor]:
    """fp32 masked gated-delta scan; fla on CUDA, the reference elsewhere."""
    args = tuple(x.float() for x in (q, k, v, log_decay, beta))
    if q.is_cuda:
        return _fla_delta_scan(*args)
    return reference_delta_scan(*args)


def memory_write_mask(kind: Tensor, pad_slot: int) -> Tensor:
    """Slots whose stored belief enters the memory.

    ``hiddens[:, j]`` holds the belief computed at slot ``j - 1``; it is a
    real belief exactly when both slots are real stream positions.
    """
    real = kind != pad_slot
    mask = torch.zeros_like(real)
    mask[:, 1:] = real[:, 1:] & real[:, :-1]
    return mask


def masked_write_terms(
    writer: MemoryWriter, hiddens: Tensor, write_mask: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Writer terms with masked slots turned into exact no-ops."""
    k, v, log_decay, beta = writer(hiddens)
    flag = write_mask.to(torch.float32)
    return k, v, log_decay * flag[..., None], beta * flag[..., None]


class GatedDeltaMemory(nn.Module):
    """The actor's memory: its writer plus its reader."""

    def __init__(
        self,
        model_dim: int,
        heads: int = DEFAULT_MEMORY_HEADS,
        head_dim: int = DEFAULT_MEMORY_HEAD_DIM,
    ):
        super().__init__()
        if heads < 1 or head_dim < 1:
            raise ValueError("memory heads and head dim must be positive")
        self.heads = heads
        self.head_dim = head_dim
        self.writer = MemoryWriter(model_dim, heads, head_dim)
        self.reader = MemoryReader(model_dim, heads, head_dim)

    # ------------------------------------------------------------ decode
    def initial_state(self, rows: int, device: torch.device) -> Tensor:
        return torch.zeros(
            (rows, self.heads, self.head_dim, self.head_dim),
            dtype=torch.float32,
            device=device,
        )

    def prompt_state(self, beliefs: Tensor, valid: Tensor | None) -> Tensor:
        """State after writing every real prompt belief ``(B, P, D)``.

        Its final belief -- the one that produces the first generated token --
        is included, exactly as replay writes it at the first generated slot.
        """
        if valid is None:
            valid = torch.ones(
                beliefs.shape[:2], dtype=torch.bool, device=beliefs.device
            )
        # Pad beliefs are never consumed elsewhere, so nothing pins them
        # finite; a masked no-op multiplies by zero, which NaN survives.
        beliefs = torch.where(valid[..., None], beliefs, 0)
        k, v, log_decay, beta = masked_write_terms(self.writer, beliefs, valid)
        # The kernel needs a query; its reads are discarded.
        return delta_scan(k, k, v, log_decay, beta)[1]

    def write(self, state: Tensor, belief: Tensor) -> Tensor:
        """Write one belief ``(R, D)`` per row into ``state``."""
        return delta_write(state, *self.writer(belief))

    def read_input(
        self, base: Tensor, hidden: Tensor, state: Tensor
    ) -> Tensor:
        """Decode input ``(R, 1, D)``: ``base`` plus the read of ``state``.

        ``hidden`` is the belief that produced the token, already written.
        """
        q = self.reader.queries(base.squeeze(1), hidden)
        return self.reader.add(base, delta_read(state, q)[:, None])

    # ------------------------------------------------------------ replay
    def stream_inputs(
        self,
        base: Tensor,
        hiddens: Tensor,
        write_mask: Tensor,
        read_mask: Tensor,
    ) -> Tensor:
        """Replay inputs ``(B, T, D)``: reads at ``read_mask`` slots.

        Gradient reaches every memory parameter through every earlier write;
        ``hiddens`` are stored sampled data and receive none.
        """
        terms = masked_write_terms(self.writer, hiddens.detach(), write_mask)
        return read_stream(self.reader, terms, base, hiddens, read_mask)


def read_stream(
    reader: MemoryReader,
    write_terms: tuple[Tensor, Tensor, Tensor, Tensor],
    base: Tensor,
    hiddens: Tensor,
    read_mask: Tensor,
) -> Tensor:
    """Scan ``write_terms`` and add ``reader``'s reads at ``read_mask`` slots.

    The dense compute-then-select keeps static shapes. Selected slots get
    ``reader.add`` of the scan's reads, the decode path's formula; the values
    agree with decode to kernel precision, not bitwise: the chunked scan
    reduces in a different order through TF32 dots, while the decode's
    sequential recall and read are fp32 elementwise product-sums.
    """
    q = reader.queries(base, hiddens.detach())
    reads = delta_scan(q, *write_terms)[0]
    return torch.where(read_mask[..., None], reader.add(base, reads), base)


def borrowed_write_terms(
    writer: MemoryWriter, hiddens: Tensor, write_mask: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The actor's write terms as constants, for a reader that only reads.

    The critic's value loss must not train the actor's writer.
    """
    with torch.no_grad():
        return masked_write_terms(writer, hiddens, write_mask)

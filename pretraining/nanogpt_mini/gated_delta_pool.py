"""Layerwise GDN-2 with a shared pool of latent state banks, routed reads and per-bank adapters.

Every layer keeps its exact private GDN-2 recurrence. In addition, each
sequence owns ``pool_banks`` shared GDN-2 state banks of ``pool_heads`` heads
(half the private width by default). All layers address the banks through one
shared set of pool projections (query, key, value, decay, erase and write
gates computed from the layer's normalized input), so every layer writes and
reads in one common key space. Each bank then owns full-rank ``latent x
latent`` adapters on its keys, values, queries and outputs, which is what lets
banks become different functions rather than interchangeable copies.

Routing follows the mixture-of-experts recipe: each layer's router scores the
banks in fp32, reads the top-``pool_reads`` banks and combines the reads with
the selected probabilities renormalized to sum to one, so the task gradient
reaches the router as a contrast between the banks it chose. A Switch
balance loss over the top-k selections and a router z-loss regularize the
routers. Two writers are implemented:

* ``routed``: the layer writes into its top-1 bank (and reads it after the
  write, as the private recurrence does), and reads the other selected banks
  through read-only events.
* ``layer``: bank ``l`` is written only by layer ``l`` (a feedback-style single
  writer, ``pool_banks == num_layers``); every layer routes its reads over all
  banks, its own included.

Bank events are ordered by (token, layer, event): a read at token t sees every
write with an earlier token and the writes of layers 0..L at token t. The model
uses two Jacobi passes: pass one runs the private model and records every
layer's pool inputs and routes; the pool then processes each bank's events in
event order with one packed chunk-kernel call; pass two rebuilds the private
model from the embeddings and adds each layer's weighted reads through a gated
normalization and a zero-initialized output projection. Both passes are
scored; the headline metric is the second pass.

The pool is packed, not padded to a capacity: events are laid out bank by
bank in chunk-aligned segments of the kernel's packed (variable-length) mode,
bank-major across the microbatch, followed by one filler segment. The row
length is fixed at ``batch * (events // 64 + banks)`` chunks, which always fits
every bank's chunk-rounded count, so the whole step compiles and CUDA-graphs
whatever the routing, and the kernel processes every real event exactly once.
Padding rows are no-op events (zero decay, erase, write, key and query). Since
every 64-row tile belongs to one bank, the adapters run as a static per-tile
matmul (``gated_delta_bank_linear``).

The vendored GDN-2 code is licensed for research/evaluation under NVIDIA's
Source Code License-NC. See pretraining/gated_delta/vendor/LICENSE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange

from fla.models.utils import Cache

from pretraining.nanogpt_mini import gated_delta_ops
from pretraining.nanogpt_mini.gated_delta_bank_linear import bank_linear
from pretraining.nanogpt_mini.gated_delta_model import (
    CustomOpRMSNormSwishGate, CustomOpShortConvolution, GatedDeltaBlock, GatedDeltaGPT, gated_delta_config,
)

POOL_PASSES = 2
POOL_READS = 2
BALANCE_COEFFICIENT = 0.01
Z_COEFFICIENT = 0.001
POOL_WRITERS = ("routed", "layer")
# The official mixer's Xavier gain; the pool projections and routers follow its projections.
MIXER_PROJECTION_GAIN = 2 ** -2.5
# Per-block pool modules; the shared projections and adapters live on the model as ``pool`` and ``adapters``.
POOL_MODULES = ("router", "pool_gate_proj", "pool_norm", "pool_o_proj")
SHARED_POOL_MODULES = ("pool", "adapters")


@dataclass
class PoolPort:
    """One layer's first-pass interaction with the pool."""
    hidden: Tensor   # [B, T, D] normalized block input, the pool projections' and router's input
    logits: Tensor   # [B, T, banks] fp32 router logits
    probs: Tensor    # [B, T, banks] fp32 router distribution
    reads: Tensor    # [B, T, reads] banks read, best first
    weights: Tensor  # [B, T, reads] fp32 read weights: the selected probabilities renormalized to one
    write: Tensor    # [B, T] bank written


def pool_row_chunks(events: int, banks: int, chunk: int) -> int:
    """Chunks reserved per sequence: every bank's chunk-rounded count plus at least one filler chunk.

    The banks' rounded counts total at most ``events + banks * (chunk - 1)``
    events, i.e. fewer than ``events // chunk + banks`` chunks, so that many
    chunks always leaves at least one chunk per sequence for the filler.
    """
    if events % chunk:
        raise ValueError("a sequence's event count must be a multiple of the kernel chunk")
    return events // chunk + banks


@dataclass
class PoolLayout:
    """Packed placement of one microbatch's events for the kernel's variable-length mode."""
    slots: Tensor          # [batch, events] int64 row of every event in the packed pool
    cu_seqlens: Tensor     # [batch * banks + 2] int32 segment offsets: bank-major (bank, sequence) segments, then the filler
    chunk_indices: Tensor  # [chunks, 2] int32 (segment, chunk within segment) of every chunk
    tile_bank: Tensor      # [chunks] int32 bank of every 64-row tile (filler tiles borrow the last bank; their rows are zero)
    bank_tiles: Tensor     # [banks + 1] int32: bank j owns tiles [bank_tiles[j], bank_tiles[j + 1])
    rows: int              # packed pool rows, chunks * chunk
    chunks: int            # packed chunks, batch * (events // chunk + banks)


def pool_layout(route: Tensor, banks: int, chunk: int) -> PoolLayout:
    """Lay the events out bank by bank in chunk-aligned segments, bank-major across the batch.

    ``route`` is ``[batch, events]`` in ascending event order. Within a bank
    the events keep that order; each (bank, sequence) segment starts at a
    chunk boundary, so it has an independent recurrent state and a whole
    number of chunks, and every bank's tiles are contiguous across the batch.
    All shapes are static: they depend on the batch and event counts only.
    """
    if route.dtype != torch.int64 or route.ndim != 2:
        raise ValueError("route must be [batch, events] int64 bank indices")
    batch, events = route.shape
    chunks = batch * pool_row_chunks(events, banks, chunk)
    one_hot = F.one_hot(route, banks).to(torch.int32)
    position = (one_hot.cumsum(1) - one_hot).gather(2, route.unsqueeze(-1)).squeeze(-1)
    rounded = (one_hot.sum(1) + (chunk - 1)) // chunk                            # [batch, banks] chunks per segment
    lengths = rounded.t().reshape(-1).to(torch.int64)                            # bank-major segment lengths
    starts = F.pad(lengths.cumsum(0), (1, 0))                                    # [banks * batch + 1]; last = filler start
    slots = starts[:-1].view(banks, batch).t().gather(1, route) * chunk + position
    end = torch.full((1,), chunks, device=route.device, dtype=torch.int64)
    boundaries = torch.cat((starts, end))                                        # every segment start, then the filler end
    cu_seqlens = (boundaries * chunk).to(torch.int32)
    local = torch.arange(chunks, device=route.device)
    segment = (local[:, None] >= starts[None, :]).sum(-1) - 1                    # last segment starting at or before
    within = local - starts[segment]
    chunk_indices = torch.stack((segment, within), -1).to(torch.int32)
    tile_bank = (segment // batch).clamp_max(banks - 1).to(torch.int32)
    bank_tiles = starts[::batch].to(torch.int32)
    return PoolLayout(slots, cu_seqlens, chunk_indices, tile_bank, bank_tiles, chunks * chunk, chunks)


class _ScatterRows(torch.autograd.Function):
    """Copy every field's events into their packed rows.

    The rows of one field are a permutation of the events plus zero padding,
    so the backward is exactly the row gather: one read of each event's row.
    Tracking the in-place copies with autograd instead would run a full-pool
    ``index_fill`` per field in backward.
    """

    @staticmethod
    def forward(ctx, slots: Tensor, rows: int, *fields: Tensor) -> Tensor:
        width = fields[0].shape[-2] * fields[0].shape[-1]
        pool = fields[0].new_zeros(rows, width)
        for index, field in enumerate(fields):
            if field.shape[:2] != slots.shape[:2]:
                raise ValueError("every event needs exactly one pool row")
            pool.index_copy_(0, slots[..., index].reshape(-1), field.reshape(-1, width))
        ctx.save_for_backward(slots)
        ctx.field_shape = fields[0].shape
        return pool

    @staticmethod
    def backward(ctx, grad: Tensor):
        slots, = ctx.saved_tensors
        grads = [grad.index_select(0, slots[..., index].reshape(-1)).view(ctx.field_shape)
                 for index in range(slots.shape[-1])]
        return (None, None, *grads)


class _GatherRows(torch.autograd.Function):
    """Every field's read of its events' packed rows; the backward is the row scatter."""

    @staticmethod
    def forward(ctx, pool_output: Tensor, slots: Tensor) -> tuple[Tensor, ...]:
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(slots)
        ctx.pool_shape = pool_output.shape
        flat = pool_output.reshape(pool_output.shape[0], -1)
        return tuple(flat.index_select(0, slots[..., index].reshape(-1))
                     .view(*slots.shape[:2], *pool_output.shape[1:])
                     for index in range(slots.shape[-1]))

    @staticmethod
    def backward(ctx, *grads: Tensor | None):
        slots, = ctx.saved_tensors
        rows, width = ctx.pool_shape[0], ctx.pool_shape[1] * ctx.pool_shape[2]
        present = [(index, grad) for index, grad in enumerate(grads) if grad is not None]
        if not present:
            return None, None
        grad_pool = present[0][1].new_zeros(rows, width)
        for index, grad in present:
            grad_pool.index_copy_(0, slots[..., index].reshape(-1), grad.reshape(-1, width))
        return grad_pool.view(ctx.pool_shape), None


def scatter_events(fields: list[Tensor], slots: Tensor, rows: int) -> Tensor:
    """Place ``[batch, time, H, D]`` event fields at their packed pool rows, as a ``[rows, H * D]`` pool.

    ``slots`` is ``[batch, time, len(fields)]``. Fields are copied one at a
    time, so the per-field tensors are the only copies besides the pool
    itself. Unfilled rows stay zero (no-op events).
    """
    return _ScatterRows.apply(slots, rows, *fields)


def gather_events(pool_output: Tensor, slots: Tensor) -> list[Tensor]:
    """Each slot column's ``[batch, time, H, D]`` read of its events' packed pool rows."""
    return list(_GatherRows.apply(pool_output, slots))


def balance_loss(probs: Tensor, reads: Tensor, banks: int) -> Tensor:
    """Switch auxiliary loss over the top-k selections: banks * sum_j f_j * P_j.

    f_j is the share of the selections that landed on bank j and P_j the mean
    router probability of bank j; the value is 1 under uniform routing and
    larger when routing concentrates (the Latent-MoE reference formulation).
    """
    mask = torch.zeros_like(probs).scatter_(-1, reads, 1.0)
    fraction = mask.sum((0, 1)) / mask.sum()
    return banks * (fraction * probs.mean((0, 1))).sum()


def z_loss(logits: Tensor) -> Tensor:
    """Router z-loss: the squared log-partition, which keeps the logits small."""
    return torch.logsumexp(logits, -1).square().mean()


def resolve_pool_options(pool_banks: int | None, pool_writer: str, pool_heads: int, *, num_layers: int,
                         head_dim: int) -> dict:
    """Validate the pool's free options and record them with the module's fixed policy.

    The layer writer owns one bank per layer, so its bank count is the layer
    count; the routed writer defaults to twelve banks.
    """
    if pool_writer not in POOL_WRITERS:
        raise ValueError(f"pool_writer must be one of {POOL_WRITERS}")
    if isinstance(pool_heads, bool) or not isinstance(pool_heads, int) or pool_heads < 1:
        raise ValueError("pool_heads must be a positive integer")
    if (pool_heads * head_dim) % 64:
        raise ValueError("The pool's latent width (pool_heads * head_dim) must be a multiple of 64")
    if pool_writer == "layer":
        if pool_banks is None:
            pool_banks = num_layers
        elif pool_banks != num_layers:
            raise ValueError("The layer writer owns one bank per layer: pool_banks must equal num_layers")
    elif pool_banks is None:
        pool_banks = 12
    if isinstance(pool_banks, bool) or not isinstance(pool_banks, int) or pool_banks < POOL_READS:
        raise ValueError(f"pool_banks must be an integer of at least {POOL_READS}")
    return dict(shared_pool=True, pool_banks=pool_banks, pool_writer=pool_writer, pool_heads=pool_heads,
                pool_reads=POOL_READS, pool_passes=POOL_PASSES, pool_balance_coefficient=BALANCE_COEFFICIENT,
                pool_z_coefficient=Z_COEFFICIENT)


def pool_policy(args, *, num_layers: int = 6) -> dict:
    """The pool configuration a training or benchmark command line records."""
    return resolve_pool_options(args.pool_banks, args.pool_writer, args.pool_heads, num_layers=num_layers,
                                head_dim=args.memory_head_dim if hasattr(args, "memory_head_dim") else args.head_dim)


class PoolProjections(nn.Module):
    """The pool's shared GDN-2 input projections: one key space for every layer.

    Mirrors the vendored mixer's input side (packed projections, short
    convolutions on q/k/v, channel-wise decay from a low-rank projection with
    ``A_log`` and ``dt_bias``, sigmoid erase and write gates) at the pool's
    latent width, without an output side: reads are normalized and projected
    per layer.
    """

    def __init__(self, dim: int, heads: int, head_dim: int, conv_size: int):
        super().__init__()
        latent = heads * head_dim
        self.heads, self.head_dim, self.latent = heads, head_dim, latent
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(dim, latent, bias=False) for _ in range(3))
        self.b_proj, self.w_proj = (nn.Linear(dim, latent, bias=False) for _ in range(2))
        self.f_proj = nn.Sequential(nn.Linear(dim, head_dim, bias=False), nn.Linear(head_dim, latent, bias=False))
        self.q_conv1d, self.k_conv1d, self.v_conv1d = (
            CustomOpShortConvolution(hidden_size=latent, kernel_size=conv_size, bias=False, activation="silu",
                                     backend="triton") for _ in range(3))
        self.A_log = nn.Parameter(torch.log(torch.empty(heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(latent, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=MIXER_PROJECTION_GAIN)

    def forward(self, hidden: Tensor) -> tuple[Tensor, ...]:
        """``[L, B, T, D]`` normalized inputs to (query, key, value, decay, erase, write), each ``[L, B, T, H, Dh]``."""
        layers, batch, time, _ = hidden.shape
        weight = torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
                            self.b_proj.weight, self.w_proj.weight, self.f_proj[0].weight))
        raw_q, raw_k, raw_v, raw_b, raw_w, f_input = F.linear(hidden, weight).split(
            (self.latent,) * 5 + (self.head_dim,), -1)
        # The short convolutions run every layer's sequences as one batch.
        q, k, v = (conv(x=field.reshape(layers * batch, time, self.latent))[0].view(layers, batch, time, self.latent)
                   for conv, field in ((self.q_conv1d, raw_q), (self.k_conv1d, raw_k), (self.v_conv1d, raw_v)))
        decay = (-self.A_log.float().exp().repeat_interleave(self.head_dim)
                 * F.softplus(self.f_proj[1](f_input).float() + self.dt_bias))
        return tuple(rearrange(x, "... (h d) -> ... h d", d=self.head_dim)
                     for x in (q, k, v, decay, raw_b.sigmoid(), raw_w.sigmoid()))


class PoolAdapters(nn.Module):
    """Every bank's full-rank latent adapters on its keys, values, queries and outputs.

    Initialized as one random orthogonal basis change per bank, shared by the
    bank's query and key adapters and inverted by its output adapter, so at
    initialization every bank computes the same function of the shared
    projections (up to the channel-wise gates, which act in the bank's basis)
    while the banks' parameters, and hence their gradients, already differ.
    """

    def __init__(self, banks: int, latent: int):
        super().__init__()
        self.query, self.key, self.value, self.output = (
            nn.Parameter(torch.empty(banks, latent, latent)) for _ in range(4))
        with torch.no_grad():
            for bank in range(banks):
                nn.init.orthogonal_(self.key[bank])
                nn.init.orthogonal_(self.value[bank])
            self.query.copy_(self.key)
            self.output.copy_(self.value.mT)


class SharedPoolBlock(GatedDeltaBlock):
    """GDN-2 block with a bank router, a pool read gate and a pool readout."""

    def __init__(self, config: dict, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_index = layer_idx
        if self.attn.num_v_heads != self.attn.num_heads or self.attn.allow_neg_eigval:
            raise ValueError("The shared pool covers ungrouped value heads without negative eigenvalues")

    def build_pool(self, config: dict):
        """Attach the block's pool modules once the whole private backbone is built.

        Constructing them here rather than in ``__init__`` keeps the private
        parameters bit-identical to ``GatedDeltaGPT`` at the same seed: no
        random draw of a pool module lands between two private modules.
        """
        dim, head_dim = config["model_dim"], config["head_dim"]
        latent = config["pool_heads"] * head_dim
        self.router = nn.Linear(dim, config["pool_banks"], bias=False)
        self.pool_gate_proj = nn.Sequential(nn.Linear(dim, head_dim, bias=False), nn.Linear(head_dim, latent, bias=True))
        self.pool_norm = CustomOpRMSNormSwishGate(head_dim, elementwise_affine=True, eps=config["mixer_norm_eps"])
        self.pool_o_proj = nn.Linear(latent, dim, bias=False)

    def route(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Router logits, probabilities, the top-k banks, their renormalized weights and the written bank."""
        # Routing decisions are taken in fp32 (Switch Transformer's selective precision).
        with torch.autocast("cuda", enabled=False):
            logits = F.linear(hidden.float(), self.router.weight.float())
            probs = logits.softmax(-1)
            selected, reads = probs.topk(POOL_READS, -1)
            weights = selected / selected.sum(-1, keepdim=True)
        write = reads[..., 0] if self.pool_writer == "routed" else torch.full_like(reads[..., 0], self.layer_index)
        return logits, probs, reads, weights, write

    def first_pass(self, x: Tensor) -> tuple[Tensor, PoolPort, Tensor]:
        """Private block plus the layer's pool port; also returns the mixer output.

        The first block's mixer sees the embeddings in both passes, so pass two
        reuses its output instead of recomputing it.
        """
        hidden = self.norm1(x)
        mixed = self.attn(hidden)[0]
        logits, probs, reads, weights, write = self.route(hidden)
        x = x + mixed
        x = x + self.mlp(self.norm2(x))
        return x, PoolPort(hidden, logits, probs, reads, weights, write), mixed

    def second_pass(self, x: Tensor, reads: list[Tensor], weights: Tensor, mixer: Tensor | None = None) -> Tensor:
        """Private block plus the weighted pool reads; ``mixer`` supplies a reused mixer output."""
        hidden = self.norm1(x)
        mixed = mixer if mixer is not None else self.attn(hidden)[0]
        gate = rearrange(self.pool_gate_proj(hidden), "... (h d) -> ... h d", d=self.pool_norm.hidden_size)
        # Each bank's read is normalized on its own, then the reads mix with the router's weights.
        read = sum(self.pool_norm(output, gate) * weights[..., index, None, None].to(output.dtype)
                   for index, output in enumerate(reads))
        x = x + mixed + self.pool_o_proj(rearrange(read, "b t h d -> b t (h d)"))
        return x + self.mlp(self.norm2(x))


class SharedPoolGatedDeltaGPT(GatedDeltaGPT):
    """Two-pass GDN-2 model whose layers share a routed pool of latent state banks."""

    block_class = SharedPoolBlock

    def __init__(self, *, pool_banks: int | None = None, pool_writer: str = "routed", pool_heads: int = 2,
                 **options):
        super().__init__(pool_banks=pool_banks, pool_writer=pool_writer, pool_heads=pool_heads, **options)
        config = self.config
        for block in self.blocks:
            block.pool_writer = config["pool_writer"]
            block.build_pool(config)
        self.pool = PoolProjections(config["model_dim"], config["pool_heads"], config["head_dim"], config["conv_size"])
        self.adapters = PoolAdapters(config["pool_banks"], config["pool_heads"] * config["head_dim"])
        self._initialize_pool()

    def _configure(self, *, pool_banks: int | None, pool_writer: str, pool_heads: int, shared_pool: bool = True,
                   pool_reads: int = POOL_READS, pool_passes: int = POOL_PASSES,
                   pool_balance_coefficient: float = BALANCE_COEFFICIENT,
                   pool_z_coefficient: float = Z_COEFFICIENT, **options) -> dict:
        # The recorded pool policy is accepted back so a checkpoint's config rebuilds the model,
        # but it is a constant of this module rather than a free option.
        if (shared_pool is not True or pool_reads != POOL_READS or pool_passes != POOL_PASSES
                or pool_balance_coefficient != BALANCE_COEFFICIENT or pool_z_coefficient != Z_COEFFICIENT):
            raise ValueError(f"The shared pool always reads {POOL_READS} banks over {POOL_PASSES} passes with balance "
                             f"coefficient {BALANCE_COEFFICIENT} and z coefficient {Z_COEFFICIENT}; "
                             "the recorded policy differs")
        config = gated_delta_config(**options)
        if not config["custom_ops"] or not config["fused_projections"]:
            raise ValueError("The shared pool runs on the custom-operator, packed-projection execution")
        if config["gate_in_kernel"]:
            raise ValueError("Pool events mix layers, so the decay gate is computed before dispatch")
        if config["allow_neg_eigval"]:
            raise ValueError("The shared pool covers erase gates in [0, 1]")
        return dict(config, **resolve_pool_options(pool_banks, pool_writer, pool_heads,
                                                   num_layers=config["num_layers"], head_dim=config["head_dim"]))

    @staticmethod
    def _pool_owns(name: str) -> bool:
        return (any(f".{module}." in name for module in POOL_MODULES)
                or any(name.startswith(f"{module}.") for module in SHARED_POOL_MODULES))

    @classmethod
    def _backbone_owns(cls, name: str) -> bool:
        return ".attn." not in name and not cls._pool_owns(name)

    @property
    def events_per_layer(self) -> int:
        """Pool events per (token, layer): the routed write also reads; the layer writer's write does not."""
        return POOL_READS if self.config["pool_writer"] == "routed" else POOL_READS + 1

    @property
    def read_events(self) -> list[int]:
        return list(range(POOL_READS)) if self.config["pool_writer"] == "routed" else list(range(1, POOL_READS + 1))

    @torch.no_grad()
    def _initialize_pool(self):
        for block in self.blocks:
            nn.init.xavier_uniform_(block.router.weight, gain=MIXER_PROJECTION_GAIN)
            for linear in block.pool_gate_proj:
                nn.init.xavier_uniform_(linear.weight, gain=MIXER_PROJECTION_GAIN)
            block.pool_gate_proj[1].bias.zero_()
            block.pool_norm.weight.fill_(1)
            # Zero readout: the initial model is exactly the private two-pass model.
            block.pool_o_proj.weight.zero_()
        # The shared projections, convolutions and adapters keep their constructors' initialization.

    def _pool(self, ports: list[PoolPort]) -> tuple[list[list[Tensor]], dict[str, Tensor]]:
        config = self.config
        banks, chunk, heads, head_dim = (config[key] for key in ("pool_banks", "kernel_chunk_size", "pool_heads",
                                                                 "head_dim"))
        layers, events, read_events = len(ports), self.events_per_layer, self.read_events
        batch, time = ports[0].write.shape
        query, key, value, decay, erase, write = self.pool(torch.stack([port.hidden for port in ports]))
        # Event order is (token, layer, event): the write event first, then the read-only events.
        banks_of_events = []
        for port in ports:
            if config["pool_writer"] == "layer":
                banks_of_events.append(port.write)
            banks_of_events.extend(port.reads[..., index] for index in range(POOL_READS))
        route = torch.stack(banks_of_events, 2)                                    # [batch, time, layers * events]
        layout = pool_layout(route.reshape(batch, -1), banks, chunk)
        slots = layout.slots.view(batch, time, layers, events)
        write_slots = slots[..., 0]
        read_slots = slots[..., read_events].reshape(batch, time, layers * POOL_READS)

        def written(field):
            return scatter_events(list(field), write_slots, layout.rows)

        def adapted(packed, weight):
            return bank_linear(packed, weight.to(packed.dtype), layout.tile_bank, layout.bank_tiles)

        def kernel_layout(packed):
            return packed.view(1, layout.rows, heads, head_dim)

        queries = scatter_events([query[layer] for layer in range(layers) for _ in range(POOL_READS)],
                                 read_slots, layout.rows)
        output = gated_delta_ops.chunk_gdn2_training(
            kernel_layout(adapted(queries, self.adapters.query)),
            kernel_layout(adapted(written(key), self.adapters.key)),
            kernel_layout(adapted(written(value), self.adapters.value)),
            kernel_layout(written(decay)), kernel_layout(written(erase)), kernel_layout(written(write)),
            cu_seqlens=layout.cu_seqlens, chunk_indices=layout.chunk_indices, state_v_first=config["state_v_first"])
        output = adapted(output[0].view(layout.rows, heads * head_dim), self.adapters.output)
        reads = gather_events(kernel_layout(output), read_slots)
        load = F.one_hot(route.reshape(batch, -1), banks).float().mean(1)         # [batch, banks] share of events
        stats = dict(balance=sum(balance_loss(port.probs, port.reads, banks) for port in ports),
                     z=sum(z_loss(port.logits) for port in ports),
                     load_max=load.amax())
        return [reads[layer * POOL_READS:(layer + 1) * POOL_READS] for layer in range(layers)], stats

    def forward_passes(self, inputs: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Both passes' final hidden states plus pool statistics.

        Pass one is the private model; pass two adds every layer's weighted
        reads of the pool built from pass one's writes. ``balance`` sums the
        routers' Switch losses, ``z`` their z-losses, and ``load_max`` is the
        largest share of one sequence's events routed to one bank.
        """
        if inputs.ndim != 2 or inputs.shape[1] == 0:
            raise ValueError("Expected nonempty [batch,time] token inputs")
        if inputs.device.type != "cuda":
            raise ValueError("GDN-2 requires CUDA execution; no CPU fallback")
        if inputs.shape[1] % self.config["kernel_chunk_size"]:
            raise ValueError("The pool executes fixed-length rows of whole 64-token kernel chunks")
        embedded = self.norm1(self.embed(inputs))
        x, ports, first_mixer = embedded, [], None
        for index, block in enumerate(self.blocks):
            x, port, mixer = block.first_pass(x)
            ports.append(port)
            if index == 0:
                first_mixer = mixer  # pass two's first block sees the same embeddings
        first = self.norm2(x)
        reads, stats = self._pool(ports)
        x = embedded
        for index, (block, layer_reads, port) in enumerate(zip(self.blocks, reads, ports)):
            x = block.second_pass(x, layer_reads, port.weights, first_mixer if index == 0 else None)
        return first, self.norm2(x), stats

    def forward_hidden(self, inputs: Tensor, state: Cache | None = None,
                       segment_size: int | None = None, use_cache: bool = False):
        """Second-pass hidden states; the pool has no incremental-decoding path."""
        if state is not None or use_cache:
            raise NotImplementedError("Shared-pool GDN-2 evaluates complete sequences only")
        if segment_size is not None and segment_size < 1:
            raise ValueError("segment_size must be positive when supplied")
        _, hidden, _ = self.forward_passes(inputs)
        return hidden, None

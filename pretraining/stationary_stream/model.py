"""Token-conditioned FFN stack reading a stationary buffer of detached past hiddens.

The backbone is the recurrent-CE FFN stack of ``future_credit_stream``: the
observation embedding is normalized, context is added before the first
residual FFN, and the post-final-norm hidden produces the logits. The
difference is what "context" is. CE recursion carried one vector that every
step rewrote, so information from ``K`` steps back survived only by passing
through ``K`` rewrites, none of which was asked to keep it. Here the last
``buffer_slots`` post-final-norm hiddens of the document are kept verbatim
and detached, each written once and never rewritten while it is readable,
and every step reads them with one small attention over slots (plus a null
slot, so a fresh document can attend to nothing). Slots written before the
document's BOS are masked. This is Transformer-XL's stop-gradient memory at a
segment length of one token: each producer is trained only by its own CE,
and long-horizon utility reaches it only through the reader adapting.

The buffer is a ring: the host overwrites the oldest slot in place with the
tick's detached hidden (one ``[B, D]`` write per tick), and a per-tick age
vector tells the read how old each slot is. Per token the read costs a query
projection, ``K`` key projections rotated by slot age, and one softmax per
head over the slots.

Two entry paths carry the mixed slots into the residual stream:

* ``value_map`` (the first arms): a null slot the reader can prefer, and a
  zero-initialized value projection applied *after* the heads mix their
  slices (exact by linearity). The untrained model is the context-free FFN
  and the read must grow from nothing.
* ``latent_norm``: no null slot and no value map. The mixed vector passes
  through the CE-recursion control's ``latent_norm`` (RMSNorm with learned
  gains) and enters at unit RMS from step 0; the query starts at zero and a
  learned per-head recency bias makes the newest slot dominate at init. With
  ``K=1`` this *is* the control's forward, bitwise; larger ``K`` only adds
  reachable slots.
"""

import math

import torch
from torch import Tensor, nn

from pretraining.nanogpt_mini.nanogpt_mini_model import Linear, MLP, RMSNorm
from pretraining.nextlat import rational_softcap


class BufferRead(nn.Module):
    """Multi-head attention from the observation over ``K`` slots (plus a null slot for ``value_map``).

    Time is rotary: a slot ``age`` steps old has its key rotated by the angle
    of that age (the half-truncated RoPE schedule of the shared ``Rotary``
    module, precomputed for every age ``1..K``) and the query is not rotated,
    which makes every score a function of relative age only. ``null_bias[h]``
    is the null slot's logit. Masked slots get ``-inf``; the null slot is
    always finite, so a lane with no readable slot yields the constant read
    ``value.bias`` (the bias enters once, unweighted by the non-null mass). Head
    ``h`` mixes slice ``h`` of the buffered hiddens with its own weights, then
    one ``value`` projection maps the mixed vector into the residual stream.
    ``value`` starts at zero, so an untrained model is the context-free FFN
    and grows the read from the CE gradient. That is the ``value_map`` entry.

    The ``latent_norm`` entry has neither null slot nor value map: the
    softmax runs over the readable slots only (a lane with none reads zero,
    exactly the control's zeroed carry), the query is zero-initialized so
    content addressing starts neutral, and ``age_bias[h, age-1]`` is a
    learned recency bias initialized to ``-slope * (age - 1)`` so the newest
    slot takes most of the mass at init (63% of ten slots at slope 1, 98% at
    slope 4). The model normalizes the mixed vector.
    """

    ENTRIES = ("value_map", "latent_norm")

    def __init__(self, model_dim: int, slots: int, heads: int, key_dim: int, entry: str = "latent_norm",
                 recency_slope: float = 1.0) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("heads must divide model_dim")
        if key_dim % 4:
            raise ValueError("key_dim must be a multiple of 4 for half-truncated rotary ages")
        if entry not in self.ENTRIES:
            raise ValueError(f"entry must be one of {self.ENTRIES}")
        self.slots, self.heads, self.key_dim, self.entry = slots, heads, key_dim, entry
        self.query = Linear(model_dim, heads * key_dim)
        self.key = Linear(model_dim, heads * key_dim)
        if not math.isfinite(recency_slope) or recency_slope < 0:
            raise ValueError("recency_slope must be nonnegative and finite")
        if entry == "value_map":
            self.value = Linear(model_dim, model_dim)
            self.null_bias = nn.Parameter(torch.zeros(heads))
            with torch.no_grad():
                self.value.weight.zero_()
                self.value.bias.zero_()
        else:
            self.value = None
            recency = -recency_slope * torch.arange(slots, dtype=torch.float32)
            self.age_bias = nn.Parameter(recency[None, :].repeat(heads, 1))
            with torch.no_grad():
                self.query.weight.zero_()
                self.query.bias.zero_()
        # Rotary ages: the frequency schedule of ``Rotary`` (half of the pairs
        # rotate, half are zero-frequency), evaluated once per age ``1..K``.
        angular = (1 / 1024) ** torch.linspace(0, 1, steps=key_dim // 4, dtype=torch.float32)
        angular = torch.cat((angular, angular.new_zeros(key_dim // 4)))
        theta = torch.outer(torch.arange(1, slots + 1, dtype=torch.float32), angular)
        self.register_buffer("age_cos", theta.cos(), persistent=False)
        self.register_buffer("age_sin", theta.sin(), persistent=False)

    def rotate_by_age(self, keys: Tensor, ages: Tensor) -> Tensor:
        """Rotate keys [K, B, H, key_dim] by their slot's age [K] (in ``1..K``), in FP32."""
        first, second = keys.float().chunk(2, dim=-1)
        cos = self.age_cos[ages - 1][:, None, None, :]
        sin = self.age_sin[ages - 1][:, None, None, :]
        return torch.cat((first * cos + second * sin, first * (-sin) + second * cos), dim=-1)

    def forward(self, observation: Tensor, buffer: Tensor, valid: Tensor, ages: Tensor) -> tuple[Tensor, Tensor]:
        """Return the BF16 read [B, D] and FP32 attention [B, H, K+1] (null first).

        ``observation`` [B, D] and ``buffer`` [K, B, D] are BF16; ``valid``
        [K, B] marks slots written since the lane's document began; ``ages``
        [K] gives each slot's age in steps, a permutation of ``1..K``. For
        the ``latent_norm`` entry the returned read is the un-normalized
        mixed vector and the null column is 1 exactly on lanes with no
        readable slot (their read is zero).
        """
        slots, batch, dim = buffer.shape
        heads, key_dim = self.heads, self.key_dim
        query = self.query(observation).view(batch, heads, key_dim).float()
        keys = self.rotate_by_age(self.key(buffer).view(slots, batch, heads, key_dim), ages)
        logits = torch.einsum("bhd,kbhd->bhk", query, keys) * key_dim ** -0.5
        masked = ~valid.t()[:, None, :]
        if self.entry == "value_map":
            logits = logits.masked_fill(masked, float("-inf"))
            null = self.null_bias[None, :, None].expand(batch, heads, 1)
            attention = torch.cat((null, logits), dim=-1).softmax(-1)
        else:
            logits = logits + self.age_bias[:, ages - 1][None]
            # A finite fill keeps a fully masked lane's softmax finite (uniform);
            # its mass is then zeroed, so nothing NaN reaches the backward. The
            # fill sits far below any reachable logit (the recency bias is
            # bounded by the slope times the slot count) so masked slots, which
            # still hold the previous document's hiddens, never take mass.
            readable = valid.any(0)[:, None, None]
            context = logits.masked_fill(masked, -1e30).softmax(-1) * readable
            attention = torch.cat(((~readable).float().expand(batch, heads, 1), context), dim=-1)
        weights = attention[..., 1:].permute(2, 0, 1)[..., None].to(buffer.dtype)
        slices = buffer.view(slots, batch, heads, dim // heads)
        # A fused multiply-reduce over slots: the buffer is read once, never permuted.
        mixed = (weights * slices).sum(0).reshape(batch, dim)
        return (self.value(mixed) if self.value is not None else mixed), attention

    def constant_read(self) -> Tensor | float:
        """What a lane with nothing readable reads: the value bias, or zero."""
        return self.value.bias if self.value is not None else 0.0


class StationaryFFNModel(nn.Module):
    """FFN stack whose context is a read over the last ``K`` detached hiddens.

    Parameters stay FP32 and core activations use BF16. The buffer is always
    detached: no gradient ever crosses a tick. Module construction and the
    core initialization follow ``StreamingFFNModel`` exactly, in the same
    order, and the read block is allocated last, so a same-seed backbone is
    bitwise identical to the CE-recursion control's.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        model_dim: int = 512,
        num_layers: int = 6,
        mlp_hidden: int = 2048,
        buffer_slots: int = 10,
        read_heads: int = 4,
        read_key_dim: int = 32,
        read_entry: str = "latent_norm",
        read_recency_slope: float = 1.0,
    ) -> None:
        super().__init__()
        if min(vocab_size, model_dim, num_layers, mlp_hidden, buffer_slots, read_heads, read_key_dim) <= 0:
            raise ValueError("All model dimensions, counts, and the slot count must be positive")
        self.config = {
            "vocab_size": vocab_size,
            "model_dim": model_dim,
            "num_layers": num_layers,
            "mlp_hidden": mlp_hidden,
            "buffer_slots": buffer_slots,
            "read_heads": read_heads,
            "read_key_dim": read_key_dim,
            "read_entry": read_entry,
            "read_recency_slope": read_recency_slope,
        }
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.observation_norm = RMSNorm(model_dim)
        # The control's carry norm, in the control's position (RMSNorm draws
        # no randomness, so the backbone parity below is unaffected).
        self.latent_norm = RMSNorm(model_dim) if read_entry == "latent_norm" else None
        self.blocks = nn.ModuleList(
            nn.Sequential(RMSNorm(model_dim), MLP(model_dim, mlp_hidden))
            for _ in range(num_layers)
        )
        self.final_norm = RMSNorm(model_dim)
        self.proj = Linear(model_dim, vocab_size)
        # Finish every common-core random draw before allocating the read
        # block, so a same-seed backbone matches the CE-recursion control.
        self._initialize_core()
        self.read = BufferRead(model_dim, buffer_slots, read_heads, read_key_dim, read_entry, read_recency_slope)

    def read_parameters(self):
        """Every parameter the CE-recursion control lacks: the read block (``latent_norm`` is the control's own)."""
        return self.read.parameters()

    @torch.no_grad()
    def _initialize_core(self) -> None:
        nn.init.normal_(self.embed.weight)
        for block in self.blocks:
            mlp = block[1]
            nn.init.normal_(mlp.fc.weight, std=math.sqrt(0.33 / mlp.fc.in_features))
            nn.init.zeros_(mlp.fc.bias)
            nn.init.zeros_(mlp.proj.weight)
            nn.init.zeros_(mlp.proj.bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def initial_state(self, batch_size: int, device: torch.device | str) -> "RingState":
        """Return an empty ring over ``buffer_slots`` for ``batch_size`` lanes."""
        return RingState(self.config["buffer_slots"], batch_size, self.config["model_dim"], device)

    def forward(
        self, observed: Tensor, buffer: Tensor, valid: Tensor, ages: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return FP32 logits [B, V], BF16 hidden [B, D], FP32 attention [B, H, K+1], BF16 read [B, D].

        The returned read is what the entry produced *before* ``latent_norm``
        (the value-mapped vector, or the un-normalized mixed vector), which
        is what the diagnostics measure. The buffer is detached here
        unconditionally: there is no temporal gradient path in this
        architecture.
        """
        buffer = buffer.detach()
        # Shared layers cast FP32 masters to activation dtype themselves.
        # Disable autocast: CUDA rms_norm otherwise widens the residual to FP32.
        with torch.autocast("cuda", enabled=False):
            observation = self.observation_norm(self.embed(observed).bfloat16())
            read, attention = self.read(observation, buffer.bfloat16(), valid, ages)
            entry = self.latent_norm(read) if self.latent_norm is not None else read
            hidden = observation + entry
            for block in self.blocks:
                hidden = hidden + block(hidden)
            hidden = self.final_norm(hidden)
            return self.readout(hidden), hidden, attention, read

    def readout(self, hidden: Tensor) -> Tensor:
        """Apply the historical untied head and rational logit softcap."""
        logits = self.proj(hidden).float()
        return rational_softcap(logits, softcap=15.0)

    def step(
        self, observed: Tensor, buffer: Tensor, valid: Tensor, ages: Tensor, resets: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One transition: logits, the detached hidden, and the next validity.

        ``resets`` [B] marks lanes whose ``observed`` token begins a document;
        they read nothing this tick. The next validity is the readable mask
        with the oldest slot (age ``K``) marked valid, because the caller
        commits this tick's hidden into that slot (``RingState.commit``).
        """
        readable = valid & ~resets[None]
        logits, hidden, _, _ = self.forward(observed, buffer, readable, ages)
        return logits, hidden.detach(), self.next_validity(readable, ages)

    @staticmethod
    def next_validity(readable: Tensor, ages: Tensor) -> Tensor:
        """The validity after this tick's hidden is committed into the oldest slot (age ``K``)."""
        return readable | (ages == ages.shape[0])[:, None]


class RingState:
    """Host-managed ring over ``K`` slots: the buffer, its validity, and the ages.

    ``buffer`` [K, B, D] BF16, ``valid`` [K, B], and ``ages`` [K] are
    persistent device tensors mutated only in place and marked as static
    addresses, so a CUDA graph recorded against one ring reads them directly
    instead of copying them into placeholders on every replay. A ring must
    therefore live as long as the graphs recorded against it: allocate one
    per role (training, validation, generation) and ``reset`` it, rather
    than creating rings ad hoc, or every new address forces a re-record.
    ``newest`` is the slot written last; ``ages`` is refreshed from a
    precomputed ``[K, K]`` table row, so no per-tick allocation happens.
    ``commit`` writes one ``[B, D]`` hidden into the oldest slot and makes
    it the newest. Because recorded forward *and* backward graphs read the
    buffer in place, ``commit`` must run only after the tick's backward.
    """

    def __init__(self, slots: int, batch_size: int, model_dim: int, device: torch.device | str) -> None:
        self.slots = slots
        self.buffer = torch.zeros(slots, batch_size, model_dim, device=device, dtype=torch.bfloat16)
        self.valid = torch.zeros(slots, batch_size, device=device, dtype=torch.bool)
        indices = torch.arange(slots)
        # ages_table[newest, slot] = (newest - slot) mod K + 1: the newest is 1 step old.
        self.ages_table = ((indices[:, None] - indices[None, :]) % slots + 1).to(device)
        self.ages = torch.empty(slots, device=device, dtype=self.ages_table.dtype)
        for tensor in (self.buffer, self.valid, self.ages):
            torch._dynamo.mark_static_address(tensor)
        self.reset()

    def reset(self) -> None:
        """Return to the empty ring: nothing readable, slot 0 the first to be written."""
        self.buffer.zero_()
        self.valid.zero_()
        self._set_newest(self.slots - 1)

    def _set_newest(self, newest: int) -> None:
        self.newest = newest
        self.ages.copy_(self.ages_table[newest])

    @property
    def oldest(self) -> int:
        return (self.newest + 1) % self.slots

    def commit(self, hidden: Tensor, next_valid: Tensor) -> None:
        """Overwrite the oldest slot with ``hidden`` and adopt ``next_valid``."""
        oldest = self.oldest
        self.buffer[oldest].copy_(hidden)
        self.valid.copy_(next_valid)
        self._set_newest(oldest)

    def state_dict(self) -> dict:
        """A snapshot (cloned, so a later commit cannot mutate it)."""
        return {"buffer": self.buffer.detach().clone(), "valid": self.valid.clone(), "newest": self.newest}

    def load_state_dict(self, state: dict) -> None:
        if (tuple(state["buffer"].shape) != tuple(self.buffer.shape)
                or tuple(state["valid"].shape) != tuple(self.valid.shape)
                or state["valid"].dtype != torch.bool
                or not 0 <= int(state["newest"]) < self.slots):
            raise ValueError("ring state does not match the configured buffer")
        self.buffer.copy_(state["buffer"].to(self.buffer.device))
        self.valid.copy_(state["valid"].to(self.valid.device))
        self._set_newest(int(state["newest"]))

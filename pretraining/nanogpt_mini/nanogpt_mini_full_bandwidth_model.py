"""Full-bandwidth latent feedback on the nanoGPT-mini transformer blocks.

Implements Eqs. (4), (9)--(12) of arXiv:2608.08888v1. The paper does not
specify a residual depth-scaling constant; this implementation scales each of
the two residual branches per block by ``1 / sqrt(2 * num_layers)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pretraining.nanogpt_mini.mini_cached import rotary_at
from pretraining.nanogpt_mini.nanogpt_mini_model import (
    MLP,
    Block,
    Linear,
    RMSNorm,
    Rotary,
)


class _GroupedQueryAttention(nn.Module):
    """Mini attention with query groups sharing each key/value head."""

    def __init__(self, dim: int, num_kv_heads: int) -> None:
        super().__init__()
        self.head_dim = 128
        self.num_heads = dim // self.head_dim
        self.num_kv_heads = num_kv_heads
        kv_dim = num_kv_heads * self.head_dim
        self.q = Linear(dim, dim)
        self.k = Linear(dim, kv_dim)
        self.v = Linear(dim, kv_dim)
        self.proj = Linear(dim, dim)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, length = x.shape[:2]
        q = self.q(x).view(batch, length, self.num_heads, self.head_dim)
        k = self.k(x).view(batch, length, self.num_kv_heads, self.head_dim)
        v = self.v(x).view(batch, length, self.num_kv_heads, self.head_dim)
        q = self.rotary(F.rms_norm(q, (self.head_dim,)))
        k = self.rotary(F.rms_norm(k, (self.head_dim,)))
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=0.12,
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)
        return self.proj(attended.contiguous().view(batch, length, -1))


class _GroupedQueryBlock(Block):
    def __init__(self, dim: int, mlp_hidden: int | None, num_kv_heads: int) -> None:
        nn.Module.__init__(self)
        self.attn = _GroupedQueryAttention(dim, num_kv_heads)
        self.mlp = MLP(dim, mlp_hidden)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)


class FullBandwidthGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1024,
        num_layers: int = 6,
        model_dim: int = 512,
        mlp_hidden: int | None = None,
        noise: float = 0.02,
        detach_carry: bool = False,
        fusion: str = "glu",
        layerscale_init: float = 0.1,
        num_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("vocab_size", vocab_size),
            ("num_layers", num_layers),
            ("model_dim", model_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if model_dim % 128:
            raise ValueError(
                "model_dim must be divisible by the mini attention head width, 128"
            )
        num_heads = model_dim // 128
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if (
            isinstance(num_kv_heads, bool)
            or not isinstance(num_kv_heads, int)
            or num_kv_heads < 1
            or num_heads % num_kv_heads
        ):
            raise ValueError(
                "num_kv_heads must be a positive divisor of the query head count"
            )
        if mlp_hidden is not None and (
            isinstance(mlp_hidden, bool)
            or not isinstance(mlp_hidden, int)
            or mlp_hidden < 1
        ):
            raise ValueError("mlp_hidden must be None or a positive integer")
        if (
            isinstance(noise, bool)
            or not isinstance(noise, (int, float))
            or not math.isfinite(noise)
            or noise < 0
        ):
            raise ValueError("noise must be a finite nonnegative number")
        if not isinstance(detach_carry, bool):
            raise TypeError("detach_carry must be a boolean")
        if fusion not in ("glu", "layerscale"):
            raise ValueError("fusion must be glu or layerscale")
        if (
            isinstance(layerscale_init, bool)
            or not isinstance(layerscale_init, (int, float))
            or not math.isfinite(layerscale_init)
            or layerscale_init < 0
        ):
            raise ValueError("layerscale_init must be a finite nonnegative number")

        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_kv_heads = num_kv_heads
        self.kv_dim = num_kv_heads * 128
        self.mlp_hidden = mlp_hidden
        self.noise = float(noise)
        self.detach_carry = detach_carry
        self.fusion = fusion
        self.layerscale_init = float(layerscale_init)
        self.residual_scale = (2 * num_layers) ** -0.5
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList(
            [
                Block(model_dim, mlp_hidden)
                if num_kv_heads == num_heads
                else _GroupedQueryBlock(model_dim, mlp_hidden, num_kv_heads)
                for _ in range(num_layers)
            ]
        )
        for block in self.blocks:
            block.attn.num_kv_heads = num_kv_heads
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.fuse_value = nn.Linear(model_dim, model_dim, bias=False)
        self.fuse_gate = nn.Linear(model_dim, model_dim, bias=False)
        if fusion == "layerscale":
            # A channelwise residual scale, not a sigmoid interpolation gate.
            # Allocation/fill consumes no RNG, preserving common seeded weights.
            self.carry_scale = nn.Parameter(torch.empty(model_dim))
        # Keep optimizer masters in FP32, including the tied embedding/readout.
        # Activations are explicitly BF16 rather than inheriting parameter dtype.
        self.float()
        self.initialize_parameters()

    @property
    def config(self) -> dict:
        """JSON-serializable constructor arguments for checkpoint reconstruction."""
        return {
            "vocab_size": self.embed.num_embeddings,
            "num_layers": self.num_layers,
            "model_dim": self.model_dim,
            "num_kv_heads": self.num_kv_heads,
            "mlp_hidden": self.mlp_hidden,
            "noise": self.noise,
            "detach_carry": self.detach_carry,
            "fusion": self.fusion,
            "layerscale_init": self.layerscale_init,
        }

    @torch.no_grad()
    def initialize_parameters(self) -> None:
        """Mini initialization, with a unit-scale tied readout and zero residual projections."""
        for name, parameter in self.named_parameters():
            if name == "carry_scale":
                parameter.fill_(self.layerscale_init)
            elif name == "embed.weight":
                parameter.normal_(std=self.model_dim**-0.5)
            elif name.endswith(".weight"):
                if name.startswith("blocks.") and ".proj." in name:
                    parameter.zero_()
                else:
                    parameter.normal_(std=math.sqrt(0.33 / parameter.shape[-1]))
            elif name.endswith(".gains"):
                parameter.fill_(1.0)
            elif name.endswith(".bias"):
                parameter.zero_()
            else:
                raise ValueError(f"Unrecognized model parameter: {name}")

    def _plain_input(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("tokens must have nonempty shape [batch, sequence]")
        return self.norm1(self.embed(tokens).to(torch.bfloat16))

    def _token_projection(self, plain: Tensor) -> Tensor:
        projected = F.linear(plain, self.fuse_gate.weight.type_as(plain))
        return projected.sigmoid() if self.fusion == "glu" else projected

    def _fuse(
        self, plain: Tensor, token_projection: Tensor, previous: Tensor
    ) -> Tensor:
        # MiniCPM-style stop-gradient on the incoming state, not its reader.
        # Token-side gradients and attention within each pass remain attached.
        if self.detach_carry:
            previous = previous.detach()
        if self.training and self.noise:
            previous = previous + torch.empty_like(previous).uniform_(
                -self.noise, self.noise
            )
        value = F.linear(previous, self.fuse_value.weight.type_as(previous))
        if self.fusion == "glu":
            return self.norm1(value * token_projection)
        # LayerScale acts on the entire residual before the identity addition.
        # Do not normalize afterward: scale zero must preserve plain exactly.
        residual = token_projection + value
        return plain + (residual.float() * self.carry_scale).type_as(plain)

    def _fused_input(
        self,
        plain: Tensor,
        token_projection: Tensor,
        state: Tensor,
        prefix_lengths: Tensor | int | None,
    ) -> Tensor:
        if state.shape != plain.shape:
            raise ValueError("state must have shape [batch, sequence, model_dim]")
        batch, length, _ = plain.shape
        if prefix_lengths is None:
            prefix_lengths = 1
        if isinstance(prefix_lengths, int) and not isinstance(prefix_lengths, bool):
            if not 1 <= prefix_lengths <= length:
                raise ValueError("prefix_lengths must lie in [1, sequence length]")
            prefix = prefix_lengths
        elif isinstance(prefix_lengths, Tensor):
            if prefix_lengths.shape != (batch,) or prefix_lengths.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError(
                    "prefix_lengths must be an integer tensor of shape [batch]"
                )
            if prefix_lengths.device != plain.device:
                raise ValueError("prefix_lengths must be on the same device as tokens")
            torch._assert_async(
                ((prefix_lengths >= 1) & (prefix_lengths <= length)).all(),
                "prefix_lengths must lie in [1, sequence length]",
            )
            prefix = prefix_lengths[:, None, None]
        else:
            raise TypeError(
                "prefix_lengths must be None, an integer, or an integer tensor [batch]"
            )
        if length == 1:
            return plain
        fused = self._fuse(plain[:, 1:], token_projection, state[:, :-1].type_as(plain))
        positions = torch.arange(1, length, device=plain.device)[None, :, None]
        suffix = torch.where(positions < prefix, plain[:, 1:], fused)
        # Never fuse position zero; untouched prefixes are bitwise plain inputs.
        return torch.cat((plain[:, :1], suffix), dim=1)

    def fused_input(
        self,
        tokens: Tensor,
        state: Tensor | None = None,
        prefix_lengths: Tensor | int | None = None,
    ) -> Tensor:
        """Layer-0 inputs; feedback at t uses the prior pass's state at t-1."""
        plain = self._plain_input(tokens)
        if state is None:
            return plain
        return self._fused_input(
            plain, self._token_projection(plain[:, 1:]), state, prefix_lengths
        )

    def recurrent_input(
        self, tokens: Tensor, previous: Tensor, resets: Tensor
    ) -> Tensor:
        """One token per document, with plain inputs at document resets."""
        if tokens.ndim != 1 or tokens.shape[0] < 1:
            raise ValueError("tokens must have nonempty shape [batch]")
        if previous.shape != (tokens.shape[0], self.model_dim):
            raise ValueError("previous must have shape [batch, model_dim]")
        if resets.shape != tokens.shape or resets.dtype != torch.bool:
            raise ValueError("resets must be a boolean tensor of shape [batch]")
        if previous.device != tokens.device or resets.device != tokens.device:
            raise ValueError("previous and resets must be on the tokens device")
        plain = self._plain_input(tokens[:, None])
        fused = self._fuse(
            plain, self._token_projection(plain), previous[:, None].type_as(plain)
        )
        return torch.where(resets[:, None, None], plain, fused)

    def inactive_feedback_parameters(self) -> list[nn.Parameter]:
        """Parameters legitimately unused by a single plain Jacobi pass."""
        parameters = [self.fuse_value.weight, self.fuse_gate.weight]
        if self.fusion == "layerscale":
            parameters.append(self.carry_scale)
        return parameters

    def _run_blocks(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = x + self.residual_scale * block.attn(block.norm1(x))
            x = x + self.residual_scale * block.mlp(block.norm2(x))
        return self.norm2(x)

    def pass_forward(
        self,
        tokens: Tensor,
        state: Tensor | None = None,
        prefix_lengths: Tensor | int | None = None,
    ) -> Tensor:
        """One parallel Jacobi pass, returning the final-normalized top states."""
        return self._run_blocks(self.fused_input(tokens, state, prefix_lengths))

    def logits(self, state: Tensor) -> Tensor:
        """Attached, tied embedding readout; there is no independent head matrix."""
        return F.linear(state, self.embed.weight.type_as(state)).float()

    def head_loss(self, state: Tensor, targets: Tensor, z_loss: float = 0.0) -> Tensor:
        logits = self.logits(state)
        loss = F.cross_entropy(
            logits.reshape(-1, self.embed.num_embeddings),
            targets.reshape(-1),
            reduction="sum",
        )
        if z_loss:
            loss = loss + z_loss * logits.logsumexp(dim=-1).square().sum()
        return loss

    def pass_losses(
        self,
        inputs: Tensor,
        targets: Tensor,
        passes: int,
        prefix_mixin: bool = False,
        z_loss: float = 0.0,
    ) -> list[Tensor]:
        """Per-pass summed CE (+ optional z-loss), with configurable carry gradients.

        Prefix mixin samples independently for each row on every feedback pass.
        Evaluation always uses a deterministic one-token plain prefix.
        """
        if isinstance(passes, bool) or not isinstance(passes, int) or passes < 1:
            raise ValueError("passes must be a positive integer")
        plain = self._plain_input(inputs)
        state = self._run_blocks(plain)
        losses = [self.head_loss(state, targets, z_loss)]
        if passes == 1:
            return losses
        # Embedding lookup and token-side projection do not depend on pass depth.
        token_projection = self._token_projection(plain[:, 1:])
        for _ in range(passes - 1):
            prefix_lengths = None
            if prefix_mixin and self.training:
                prefix_lengths = torch.randint(
                    1, inputs.shape[1] + 1, (inputs.shape[0],), device=inputs.device
                )
            state = self._run_blocks(
                self._fused_input(plain, token_projection, state, prefix_lengths)
            )
            losses.append(self.head_loss(state, targets, z_loss))
        return losses

    def forward(
        self, inputs: Tensor, targets: Tensor, passes: int = 1, z_loss: float = 0.0
    ) -> Tensor:
        """Eq. (12), lambda=1: plain loss plus the mean of all feedback losses."""
        losses = self.pass_losses(
            inputs, targets, passes, prefix_mixin=self.training, z_loss=z_loss
        )
        if passes == 1:
            return losses[0]
        return losses[0] + sum(losses[1:]) / (passes - 1)

    def _cached_step(
        self, x: Tensor, keys: list[Tensor], values: list[Tensor], position: int
    ) -> Tensor:
        """Append one layer-0 input to the ordinary mini KV cache."""
        batch = x.shape[0]
        end = position + 1
        for block, key_cache, value_cache in zip(self.blocks, keys, values):
            attention = block.attn
            normalized = block.norm1(x)
            q_shape = (batch, 1, attention.num_heads, attention.head_dim)
            kv_shape = (batch, 1, attention.num_kv_heads, attention.head_dim)
            q = F.rms_norm(attention.q(normalized).view(q_shape), (attention.head_dim,))
            k = F.rms_norm(
                attention.k(normalized).view(kv_shape), (attention.head_dim,)
            )
            v = attention.v(normalized).view(kv_shape)
            q = rotary_at(q, attention.rotary.angular_freq, position)
            k = rotary_at(k, attention.rotary.angular_freq, position)
            key_cache[:, :, position:end].copy_(k.transpose(1, 2))
            value_cache[:, :, position:end].copy_(v.transpose(1, 2))
            # The query is the LAST position, not position zero of a rectangular
            # causal mask. All and only initialized cache entries are visible.
            attended = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                key_cache[:, :, :end],
                value_cache[:, :, :end],
                scale=0.12,
                is_causal=False,
                enable_gqa=attention.num_kv_heads != attention.num_heads,
            ).transpose(1, 2)
            update = attention.proj(
                attended.contiguous().view(batch, 1, self.model_dim)
            )
            x = x + self.residual_scale * update
            x = x + self.residual_scale * block.mlp(block.norm2(x))
        return self.norm2(x)

    @torch.no_grad()
    def loss_sequential(
        self, inputs: Tensor, targets: Tensor, prefix_length: int = 1
    ) -> Tensor:
        """Sum-CE from true cached recurrence, with a plain prompt of prefix_length tokens."""
        if self.training:
            raise ValueError(
                "loss_sequential is an evaluation API; call model.eval() first"
            )
        plain = self._plain_input(inputs)
        batch, length, _ = plain.shape
        if (
            isinstance(prefix_length, bool)
            or not isinstance(prefix_length, int)
            or not 1 <= prefix_length <= length
        ):
            raise ValueError("prefix_length must be an integer in [1, sequence length]")
        keys, values = [], []
        for block in self.blocks:
            shape = (batch, block.attn.num_kv_heads, length, block.attn.head_dim)
            keys.append(torch.empty(shape, device=plain.device, dtype=plain.dtype))
            values.append(torch.empty(shape, device=plain.device, dtype=plain.dtype))
        token_projection = self._token_projection(plain[:, prefix_length:])
        states = torch.empty_like(plain)
        previous = None
        for position in range(length):
            if position < prefix_length:
                x = plain[:, position : position + 1]
            else:
                index = position - prefix_length
                x = self._fuse(
                    plain[:, position : position + 1],
                    token_projection[:, index : index + 1],
                    previous,
                )
            previous = self._cached_step(x, keys, values, position)
            states[:, position : position + 1].copy_(previous)
        return self.head_loss(states, targets)


__all__ = ["FullBandwidthGPT"]

"""Detached token-conditioned FFN recurrence with a future-bag carry writer.

The backbone is the recurrent-CE FFN stack: one carried vector per document is
normalized and added before the first residual FFN, and the post-final-norm
hidden produces the logits. Plain CE recursion carries that hidden verbatim.

The optional ``CarryWriter`` decides what the next step receives. It is a gated
cell over the current hidden and the incoming carry, initialized to reproduce
CE recursion (gate nearly open, identity write). It is trained by reading the
carry through the model's own frozen output head (``carry_readout``): the
actor's head is the only judge of what the carry says about the future. There
is no critic and no training-only parameter.
"""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pretraining.nanogpt_mini.nanogpt_mini_model import Linear, MLP, RMSNorm
from pretraining.nextlat import rational_softcap


GATE_BIAS_INIT = 3.0
"""Initial gate logit: sigmoid(3) ~ 0.95 of the carry is the fresh write."""


def _partial_gradient(parameter: Tensor, weight: float) -> Tensor:
    """``parameter`` whose gradient is scaled by ``weight`` (0 detaches it)."""
    if weight == 0.0:
        return parameter.detach()
    if weight == 1.0:
        return parameter
    return parameter.detach() + weight * (parameter - parameter.detach())


class CarryWriter(nn.Module):
    """``carry = gate * unit(write(hidden)) + (1 - gate) * unit(previous_carry)``.

    Both terms are RMS-normalized before mixing. Only a carry's direction
    reaches the reader and the judge, so nothing is lost, and retention can
    then only be expressed through the gate, never through the relative norms
    of the write and the old carry. The gate reads both the fresh hidden and
    the incoming carry so retention can depend on what is already stored.
    ``write`` starts at identity and the gate at ``GATE_BIAS_INIT``, so an
    untrained writer is close to CE recursion while every parameter still has
    a nonzero gradient path. A zero (reset) carry stays zero under the norm.
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.write = Linear(model_dim, model_dim)
        self.gate_hidden = Linear(model_dim, model_dim)
        self.gate_carry = nn.Parameter(torch.zeros(model_dim, model_dim))
        with torch.no_grad():
            self.write.weight.copy_(torch.eye(model_dim))
            self.write.bias.zero_()
            self.gate_hidden.weight.zero_()
            self.gate_hidden.bias.fill_(GATE_BIAS_INIT)

    def forward(self, hidden: Tensor, previous_carry: Tensor) -> tuple[Tensor, Tensor]:
        """Return the next BF16 carry and the BF16 gate for ``hidden`` [B, D]."""
        # Disable autocast: CUDA rms_norm otherwise widens both terms to FP32.
        with torch.autocast("cuda", enabled=False):
            hidden = hidden.bfloat16()
            previous_carry = previous_carry.bfloat16()
            gate_logit = self.gate_hidden(hidden) + F.linear(
                previous_carry, self.gate_carry.type_as(previous_carry)
            )
            gate = gate_logit.sigmoid()
            size = (hidden.shape[-1],)
            carry = gate * F.rms_norm(self.write(hidden), size) + (1 - gate) * F.rms_norm(previous_carry, size)
            return carry, gate


class StreamingFFNModel(nn.Module):
    """A single carried vector enters before the first residual FFN.

    Parameters stay FP32 and core activations use BF16. ``forward`` detaches
    its incoming carry unless the caller explicitly requests a temporal
    gradient (the TBPTT reference). The caller owns BOS resets. ``step``
    returns the carry the next observation should receive: the writer's
    output when a writer exists, otherwise the hidden itself.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        model_dim: int = 512,
        num_layers: int = 6,
        mlp_hidden: int = 2048,
        use_writer: bool = True,
    ) -> None:
        super().__init__()
        if min(vocab_size, model_dim, num_layers, mlp_hidden) <= 0:
            raise ValueError("All model dimensions and the layer count must be positive")
        self.config = {
            "vocab_size": vocab_size,
            "model_dim": model_dim,
            "num_layers": num_layers,
            "mlp_hidden": mlp_hidden,
            "use_writer": use_writer,
        }
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.observation_norm = RMSNorm(model_dim)
        self.latent_norm = RMSNorm(model_dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(RMSNorm(model_dim), MLP(model_dim, mlp_hidden))
            for _ in range(num_layers)
        )
        self.final_norm = RMSNorm(model_dim)
        self.proj = Linear(model_dim, vocab_size)
        # Finish every common-core random draw before allocating the optional
        # writer, so same-seed arms start from identical backbones.
        self._initialize_core()
        self.writer = CarryWriter(model_dim) if use_writer else None

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

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        """Return zero BF16 carry [batch_size, model_dim] on the given device."""
        return torch.zeros(
            batch_size, self.config["model_dim"], device=device, dtype=torch.bfloat16
        )

    def forward(
        self, observed: Tensor, carry: Tensor, temporal_gradient: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Return FP32 logits [B,V] and BF16 hidden [B,D].

        The carry is detached unless ``temporal_gradient`` is requested; only
        the TBPTT reference objective asks for it.
        """
        if not temporal_gradient:
            carry = carry.detach()
        # Shared layers cast FP32 masters to activation dtype themselves.
        # Disable autocast: CUDA rms_norm otherwise widens the residual to FP32.
        with torch.autocast("cuda", enabled=False):
            hidden = self.observation_norm(self.embed(observed).bfloat16())
            hidden = hidden + self.latent_norm(carry.bfloat16())
            for block in self.blocks:
                hidden = hidden + block(hidden)
            hidden = self.final_norm(hidden)
            return self.readout(hidden), hidden

    def readout(self, hidden: Tensor) -> Tensor:
        """Apply the historical untied head and rational logit softcap."""
        logits = self.proj(hidden).float()
        return rational_softcap(logits, softcap=15.0)

    def carry_readout(self, carry: Tensor, parameter_gradient: float = 0.0) -> Tensor:
        """Read a carry [B, D] through the final norm and head as if it were a hidden.

        Returns FP32 softcapped logits [B, V]; ``carry`` keeps its gradient.
        ``parameter_gradient`` is the fraction of the gradient that reaches the
        norm gains and head weights (0: frozen, the writer's training signal).
        Rescaling the carry cannot change the result: only its direction reaches
        the reader, and only its direction is judged here.
        """
        with torch.autocast("cuda", enabled=False):
            carry = carry.bfloat16()
            gains = _partial_gradient(self.final_norm.gains, parameter_gradient)
            weight = _partial_gradient(self.proj.weight, parameter_gradient)
            bias = _partial_gradient(self.proj.bias, parameter_gradient)
            direction = F.rms_norm(carry, (carry.shape[-1],), weight=gains.type_as(carry))
            logits = F.linear(direction, weight.type_as(carry), bias.type_as(carry)).float()
            return rational_softcap(logits, softcap=15.0)

    def next_carry(self, hidden: Tensor, carry: Tensor) -> Tensor:
        """Return what the next observation receives: written carry or hidden."""
        if self.writer is None:
            return hidden
        return self.writer(hidden, carry)[0]

    def step(self, observed: Tensor, carry: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Inference transition: logits, hidden, and the detached next carry."""
        logits, hidden = self.forward(observed, carry)
        return logits, hidden, self.next_carry(hidden, carry.detach())

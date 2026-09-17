"""Detached token-conditioned FFN recurrence with a scalar future-loss critic."""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pretraining.nanogpt_mini.nanogpt_mini_model import Linear, MLP, RMSNorm
from pretraining.nextlat import rational_softcap


class _ScalarValueCritic(nn.Module):
    """Tiny training-only value head over the already normalized hidden state."""

    def __init__(self, model_dim: int, critic_hidden: int) -> None:
        super().__init__()
        self.fc = Linear(model_dim, critic_hidden)
        self.proj = Linear(critic_hidden, 1)
        nn.init.normal_(self.fc.weight, std=math.sqrt(0.33 / model_dim))
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: Tensor, detach_weights: bool = False) -> Tensor:
        """Keep the hidden derivative live even when critic parameters freeze.

        The first projection and squared ReLU use BF16 activations. Final scalar
        accumulation is deliberately FP32: finite-document cumulative losses
        must retain TD residuals smaller than one BF16 unit at their magnitude.
        """
        with torch.autocast("cuda", enabled=False):
            hidden = hidden.bfloat16()
            if detach_weights:
                features = F.linear(
                    hidden,
                    self.fc.weight.detach().type_as(hidden),
                    self.fc.bias.detach().type_as(hidden),
                ).relu().square()
                value = F.linear(
                    features.float(),
                    self.proj.weight.detach(),
                    self.proj.bias.detach(),
                )
            else:
                features = self.fc(hidden).relu().square()
                value = self.proj(features.float())
            return value.squeeze(-1)


class StreamingFFNModel(nn.Module):
    """A single carried vector enters before the first residual FFN.

    Parameters stay FP32 and core activations use BF16. Every ``forward``
    detaches its incoming carry, so tokens never share a temporal graph. The
    caller owns BOS resets. The optional scalar critic estimates future CE
    from current hidden only and is not used by inference.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        model_dim: int = 512,
        num_layers: int = 6,
        mlp_hidden: int = 2048,
        use_td_critic: bool = True,
        critic_hidden: int = 64,
    ) -> None:
        super().__init__()
        if min(vocab_size, model_dim, num_layers, mlp_hidden, critic_hidden) <= 0:
            raise ValueError("All model dimensions and the layer count must be positive")
        self.config = {
            "vocab_size": vocab_size,
            "model_dim": model_dim,
            "num_layers": num_layers,
            "mlp_hidden": mlp_hidden,
            "use_td_critic": use_td_critic,
            "critic_hidden": critic_hidden,
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
        # critic, so same-seed TD and plain-CE arms start identically.
        self._initialize_core()
        self.critic = (
            _ScalarValueCritic(model_dim, critic_hidden) if use_td_critic else None
        )

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

    def forward(self, observed: Tensor, carry: Tensor) -> tuple[Tensor, Tensor]:
        """Return FP32 logits [B,V] and BF16 hidden [B,D], detaching carry."""
        # Shared layers cast FP32 masters to activation dtype themselves.
        # Disable autocast: CUDA rms_norm otherwise widens the residual to FP32.
        with torch.autocast("cuda", enabled=False):
            hidden = self.observation_norm(self.embed(observed).bfloat16())
            hidden = hidden + self.latent_norm(carry.detach().bfloat16())
            for block in self.blocks:
                hidden = hidden + block(hidden)
            hidden = self.final_norm(hidden)
            return self.readout(hidden), hidden

    def readout(self, hidden: Tensor) -> Tensor:
        """Apply the historical untied head and rational logit softcap."""
        logits = self.proj(hidden).float()
        return rational_softcap(logits, softcap=15.0)

    def value(self, hidden: Tensor, detach_weights: bool = False) -> Tensor:
        """Return FP32 future-loss values [B] without detaching ``hidden``.

        Fit the critic with ``value(hidden.detach())``. Use
        ``value(hidden, detach_weights=True)`` for the producer's input gradient:
        this differentiates the scalar value through hidden into the current
        FFN, without updating critic weights or propagating to preceding tokens.
        """
        if self.critic is None:
            raise RuntimeError("value requires use_td_critic=True")
        return self.critic(hidden, detach_weights=detach_weights)

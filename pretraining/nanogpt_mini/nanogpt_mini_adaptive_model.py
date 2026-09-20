"""Causal variable-length character decoding with learned sparse global refreshes.

Training uses two independent hard route samples and a leave-one-out policy
baseline. Evaluation uses a deterministic causal gate, so no uncharged latent
boundary choices enter the reported character likelihood. No fixed chunk size.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pretraining.nanogpt_mini.bit_density import PrefixBinaryHead, binary_nll
from pretraining.nanogpt_mini.nanogpt_mini_model import Block, Linear, RMSNorm


@dataclass(frozen=True)
class AdaptiveConfig:
    vocab_size: int
    code_bits: int = 0
    num_layers: int = 6
    model_dim: int = 512
    local_layers: int = 2
    local_dim: int = 128
    prefix_width: int = 128
    refresh_cost: float = 0.02

    def __post_init__(self):
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.code_bits == 0:
            object.__setattr__(
                self, "code_bits", max(2, (self.vocab_size - 1).bit_length())
            )
        if not 2 <= self.code_bits <= 32 or self.vocab_size > 1 << self.code_bits:
            raise ValueError(
                "code_bits must cover the alphabet and be between 2 and 32"
            )
        if self.num_layers < 1 or self.local_layers < 1 or self.prefix_width < 1:
            raise ValueError("layer counts and prefix_width must be positive")
        for width in (self.model_dim, self.local_dim):
            if width < 128 or width % 128:
                raise ValueError("Mini widths must be positive multiples of 128")
        if not math.isfinite(self.refresh_cost) or self.refresh_cost < 0:
            raise ValueError("refresh_cost must be finite and nonnegative")


class LocalCharacterEncoder(nn.Module):
    def __init__(self, config: AdaptiveConfig):
        super().__init__()
        self.bos = nn.Parameter(torch.empty(1, 1, config.local_dim))
        self.input = Linear(config.code_bits, config.local_dim)
        self.norm1 = RMSNorm(config.local_dim)
        self.blocks = nn.ModuleList(
            Block(config.local_dim) for _ in range(config.local_layers)
        )
        self.norm2 = RMSNorm(config.local_dim)

    def forward(self, codes: Tensor) -> Tensor:
        embedded = self.input((2 * codes[:, :-1] - 1).to(torch.bfloat16))
        # A learned initial state avoids propagating an exact zero through both
        # normalized trunks, whose zero-state Jacobians otherwise multiply.
        previous = torch.cat(
            (self.bos.to(torch.bfloat16).expand(codes.shape[0], -1, -1), embedded),
            dim=1,
        )
        hidden = self.norm1(previous)
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm2(hidden)


class EventTransformer(nn.Module):
    def __init__(self, config: AdaptiveConfig):
        super().__init__()
        self.input = Linear(config.local_dim, config.model_dim)
        self.norm1 = RMSNorm(config.model_dim)
        self.blocks = nn.ModuleList(
            Block(config.model_dim) for _ in range(config.num_layers)
        )
        self.norm2 = RMSNorm(config.model_dim)

    def forward(self, events: Tensor) -> Tensor:
        hidden = self.norm1(self.input(events))
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm2(hidden)


def routing_surrogate(
    router_logits: Tensor, routes: Tensor, nll: Tensor
) -> tuple[Tensor, Tensor]:
    """Two-sample RLOO; each action sees its own loss-to-go minus the other route.

    The other route is sampled independently, making its suffix loss an
    action-independent baseline. Cost has a separate analytic derivative.
    """
    delta = nll[0].detach() - nll[1].detach()
    advantage = delta.flip(-1).cumsum(-1).flip(-1)
    log_probability = -F.binary_cross_entropy_with_logits(
        router_logits.unsqueeze(0).expand_as(routes), routes.float(), reduction="none"
    )
    # BOS is forced, not a stochastic action. Empty T=1 suffixes remain finite.
    credited = advantage[:, 1:]
    policy = (
        (log_probability[0, :, 1:] - log_probability[1, :, 1:]) * credited
    ).sum() * 0.5
    rms = (credited.square().sum() / max(1, credited.numel())).sqrt()
    return policy - policy.detach(), rms.detach()


class AdaptiveGPT(nn.Module):
    identity_shifts: Tensor
    # One host synchronization determines the compact allocation. Neural kernels
    # are compiled individually; compiling this Python orchestration hides a break.
    compile_forward = False
    sum_stat_names = ("rate_nats", "compute_nats")
    mean_stat_names = (
        "refresh_rate",
        "expected_refresh_rate",
        "global_positions_per_character",
        "padded_global_positions_per_character",
        "router_entropy",
        "policy_advantage_rms",
    )

    def __init__(self, config: AdaptiveConfig):
        super().__init__()
        self.config = config
        self.local = LocalCharacterEncoder(config)
        self.router = Linear(config.local_dim, 1)
        self.prior = EventTransformer(config)
        self.local_output = Linear(config.local_dim, config.model_dim)
        self.readout_norm = RMSNorm(config.model_dim)
        self.head = PrefixBinaryHead(
            config.model_dim, config.code_bits, config.prefix_width
        )
        self.register_buffer(
            "identity_shifts", torch.arange(config.code_bits - 1, -1, -1)
        )
        self.reset_parameters()

    @property
    def transport_widths(self) -> tuple[int, int]:
        return self.config.code_bits, 0

    @torch.no_grad()
    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if name == "local.bos":
                parameter.normal_(std=math.sqrt(0.33))
            elif name.endswith("weight"):
                if ".proj." in name or name == "head.output.weight":
                    parameter.zero_()
                else:
                    parameter.normal_(std=math.sqrt(0.33 / parameter.shape[1]))
            elif name.endswith("bias"):
                parameter.zero_()
            elif name.endswith("gains"):
                parameter.fill_(1)
            else:
                raise ValueError(f"uninitialized parameter: {name}")

    def identity_bits(self, ids: Tensor) -> Tensor:
        return ((ids.long().unsqueeze(-1) >> self.identity_shifts) & 1).float()

    def encode_context(self, codes: Tensor) -> tuple[Tensor, Tensor]:
        local = self.local(codes)
        return local, self.router(local).float().squeeze(-1)

    def choose_routes(self, logits: Tensor) -> Tensor:
        if self.training:
            routes = torch.rand(
                (2, *logits.shape), device=logits.device
            ) < logits.sigmoid().unsqueeze(0)
        else:
            routes = (logits >= 0).unsqueeze(0)
        routes[..., 0] = True
        return routes

    @staticmethod
    def packed_capacity(routes: Tensor) -> int:
        # Allocation buckets only: this never changes a route or forces a stop.
        maximum = int(routes.sum(-1).max().item())
        return min(routes.shape[-1], ((maximum + 31) // 32) * 32)

    def packed_logits(
        self, local: Tensor, codes: Tensor, routes: Tensor, capacity: int
    ) -> tuple[Tensor, Tensor]:
        rollouts, batch, length = routes.shape
        positions = torch.arange(length, device=routes.device)
        ordered = (
            torch.where(routes, positions, length).sort(dim=-1).values[..., :capacity]
        )
        # Padding lives strictly after real events. Causal attention prevents it
        # from affecting any real query; held-state indices never select padding.
        indices = ordered.clamp_max(length - 1)
        events = (
            local.unsqueeze(0)
            .expand(rollouts, -1, -1, -1)
            .gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, local.shape[-1]))
        )
        global_states = self.prior(
            events.reshape(rollouts * batch, capacity, -1)
        ).reshape(rollouts, batch, capacity, self.config.model_dim)
        latest = routes.long().cumsum(-1) - 1
        held = global_states.gather(
            2, latest.unsqueeze(-1).expand(-1, -1, -1, self.config.model_dim)
        )
        context = self.readout_norm(held + self.local_output(local).unsqueeze(0))
        targets = codes.unsqueeze(0).expand(rollouts, -1, -1, -1)
        logits = self.head(context, targets).float()
        logits = 15 * logits * (logits.square() + 225).rsqrt()
        return binary_nll(logits, targets, 1), logits

    def pack_and_score(
        self, local_states: Tensor, codes: Tensor, routes: Tensor
    ) -> tuple[Tensor, Tensor, int]:
        if routes.dtype != torch.bool or routes.shape != codes.shape[:2]:
            raise ValueError(
                "routes must be a boolean mask matching character positions"
            )
        if not bool(routes[:, 0].all()):
            raise ValueError("every sequence must refresh its BOS state")
        capacity = self.packed_capacity(routes)
        nll, logits = self.packed_logits(
            local_states, codes, routes.unsqueeze(0), capacity
        )
        return nll[0], logits[0], capacity

    def predict(self, ids: Tensor, routes: Tensor | None = None) -> dict:
        codes = self.identity_bits(ids)
        local, router_logits = self.encode_context(codes)
        if routes is None:
            routes = router_logits >= 0
            routes[:, 0] = True
        _, logits, capacity = self.pack_and_score(local, codes, routes)
        return {
            "logits": logits,
            "routes": routes,
            "router_logits": router_logits,
            "local_states": local,
            "packed_capacity": capacity,
        }

    def finish_objective(
        self, router_logits: Tensor, routes: Tensor, nll: Tensor, capacity: int
    ):
        batch, length = router_logits.shape
        count = batch * length
        probability = router_logits.sigmoid()
        expected_updates = probability[:, 1:].sum() + batch
        rate = nll.sum() / routes.shape[0]
        cost = self.config.refresh_cost * expected_updates
        if self.training:
            policy, advantage_rms = routing_surrogate(router_logits, routes, nll)
        else:
            policy = router_logits.new_zeros(())
            advantage_rms = router_logits.new_zeros(())
            # Report the cost of actual deployed decisions, not sampled training.
            cost = self.config.refresh_cost * routes.sum()
        actual_updates = routes.sum().float()
        stats = {
            "rate_nats": rate.detach(),
            "compute_nats": cost.detach(),
            "refresh_rate": (actual_updates / routes.numel()).detach(),
            "expected_refresh_rate": (expected_updates / count).detach(),
            "global_positions_per_character": (actual_updates / count).detach(),
            "padded_global_positions_per_character": router_logits.new_ones(())
            * (routes.shape[0] * capacity / length),
            "router_entropy": (
                F.binary_cross_entropy_with_logits(
                    router_logits[:, 1:], probability[:, 1:], reduction="sum"
                )
                / max(1, batch * (length - 1))
            ).detach(),
            "policy_advantage_rms": advantage_rms,
        }
        return rate + cost + policy, stats

    def forward(self, ids: Tensor, codec_weight: float = 1.0):
        codes = self.identity_bits(ids)
        local, router_logits = self.encode_context(codes)
        routes = self.choose_routes(router_logits)
        capacity = self.packed_capacity(routes)
        nll, _ = self.packed_logits(local, codes, routes, capacity)
        return self.finish_objective(router_logits, routes, nll, capacity)

    def compile_components(self):
        self.encode_context = torch.compile(
            self.encode_context, dynamic=True, fullgraph=True
        )
        self.choose_routes = torch.compile(
            self.choose_routes, dynamic=True, fullgraph=True
        )
        self.packed_logits = torch.compile(
            self.packed_logits, dynamic=True, fullgraph=True
        )
        self.finish_objective = torch.compile(
            self.finish_objective, dynamic=True, fullgraph=True
        )

    def export(self, ids: Tensor) -> dict[str, list[list[int]]]:
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("export requires one row")
        if ids.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("source identities must be integers")
        originals = ids[0].tolist()
        if any(value < 0 or value >= self.config.vocab_size for value in originals):
            raise ValueError("source identity is outside the checkpoint alphabet")
        return {
            "latents": [
                [
                    (value >> shift) & 1
                    for shift in range(self.config.code_bits - 1, -1, -1)
                ]
                for value in originals
            ],
            "residual": [[] for _ in originals],
        }

    def recover(self, latents: list[list[int]], residual: list[list[int]]) -> list[int]:
        if len(latents) != len(residual):
            raise ValueError("latent and residual stream lengths differ")
        originals = []
        for row, correction in zip(latents, residual, strict=True):
            if correction:
                raise ValueError("adaptive character transport has no residual bits")
            if len(row) != self.config.code_bits or any(
                type(bit) not in (bool, int) or bit not in (0, 1) for bit in row
            ):
                raise ValueError("invalid binary stream or width")
            value = 0
            for bit in row:
                value = (value << 1) | bit
            if value >= self.config.vocab_size:
                raise ValueError(
                    "decoded identity is outside the checkpoint alphabet (reserved ID)"
                )
            originals.append(value)
        return originals

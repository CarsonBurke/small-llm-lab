"""A dense Mini teacher with cheap recurrent proposals for exact speculation.

Detached teacher states supervise short recurrent rollouts. Canonical evaluation
scores the dense teacher, whose distribution verified speculative sampling
preserves. The learned refresh gate is retained only for historical approximate
rollout diagnostics and its existing auxiliary training objective.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call

from pretraining.nanogpt_mini.bit_density import binary_nll
from pretraining.nanogpt_mini.nanogpt_mini_bitflow_model import (
    BitFlowConfig,
    BitFlowGPT,
)
from pretraining.nanogpt_mini.nanogpt_mini_model import Linear, RMSNorm


@dataclass(frozen=True)
class DynamicsConfig:
    vocab_size: int
    code_bits: int = 0
    num_layers: int = 6
    model_dim: int = 512
    prefix_width: int = 128
    dynamics_width: int = 128
    rollout_horizon: int = 2
    latent_weight: float = 1.0
    rollout_weight: float = 1.0
    gate_weight: float = 1.0
    refresh_cost: float = 0.02

    def __post_init__(self):
        for name in (
            "vocab_size",
            "num_layers",
            "model_dim",
            "prefix_width",
            "dynamics_width",
            "rollout_horizon",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.code_bits) is not int:
            raise ValueError("code_bits must be an integer")
        if self.code_bits == 0:
            object.__setattr__(
                self, "code_bits", max(2, (self.vocab_size - 1).bit_length())
            )
        if not 2 <= self.code_bits <= 32 or self.vocab_size > 1 << self.code_bits:
            raise ValueError(
                "code_bits must cover the alphabet and be between 2 and 32"
            )
        if self.model_dim % 128:
            raise ValueError("Mini model_dim must be a positive multiple of 128")
        for name in ("latent_weight", "rollout_weight", "gate_weight", "refresh_cost"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (name == "refresh_cost" and value == 0)
            ):
                requirement = "positive" if name == "refresh_cost" else "nonnegative"
                raise ValueError(f"{name} must be finite and {requirement}")


class ResidualTransition(nn.Module):
    def __init__(self, config: DynamicsConfig):
        super().__init__()
        self.norm = RMSNorm(config.model_dim + config.code_bits)
        self.input = Linear(config.model_dim + config.code_bits, config.dynamics_width)
        self.output = Linear(config.dynamics_width, config.model_dim)

    def forward(self, state: Tensor, previous_code: Tensor) -> Tensor:
        features = torch.cat(
            (state.to(torch.bfloat16), (2 * previous_code - 1).to(torch.bfloat16)),
            dim=-1,
        )
        return state.to(torch.bfloat16) + self.output(
            self.input(self.norm(features)).relu().square()
        )


class CausalErrorPredictor(nn.Module):
    def __init__(self, config: DynamicsConfig):
        super().__init__()
        self.input = Linear(config.model_dim + 1, config.dynamics_width)
        self.output = Linear(config.dynamics_width, 1)

    def forward(self, state: Tensor, age: Tensor) -> Tensor:
        features = torch.cat(
            (
                state.to(torch.bfloat16),
                age.float().log1p().unsqueeze(-1).to(torch.bfloat16),
            ),
            dim=-1,
        )
        return F.softplus(
            self.output(self.input(features).relu().square()).float().squeeze(-1)
        )


class DynamicsGPT(nn.Module):
    identity_shifts: Tensor
    architecture = "nanogpt_mini_dynamics_characters_v1"
    # Compile neural work, not the potentially unbounded Python rollout loop.
    compile_forward = False
    transition_type = ResidualTransition
    sum_stat_names = ("rate_nats",)
    mean_stat_names = (
        "latent_loss",
        "rollout_kl",
        "gate_loss",
        "refresh_rate",
        "backbone_positions_per_character",
        "evaluation_backbone_positions_per_character",
    )

    def __init__(self, config: DynamicsConfig):
        super().__init__()
        self.config = config
        # The fixed-prefix control constructs and resets a small frozen codec
        # before its teacher. Retain that temporary construction to consume the
        # exact same seeded RNG draws, then discard the unused codec entirely.
        reference = BitFlowGPT(
            BitFlowConfig(
                vocab_size=config.vocab_size,
                code_bits=config.code_bits,
                num_layers=config.num_layers,
                model_dim=config.model_dim,
                mixture_components=1,
                learn_codec=False,
                density_head="prefix",
                prefix_width=config.prefix_width,
            )
        )
        self.prior = reference.prior
        self.register_buffer("identity_shifts", reference.identity_shifts)
        del reference
        self.transition = self.transition_type(config)
        self.error_predictor = CausalErrorPredictor(config)
        self.reset_auxiliary_parameters()

    @property
    def head(self) -> nn.Module:
        return self.prior.proj

    @property
    def transport_widths(self) -> tuple[int, int]:
        return self.config.code_bits, 0

    @torch.no_grad()
    def reset_auxiliary_parameters(self):
        self.transition.norm.gains.fill_(1)
        for module in (self.transition, self.error_predictor):
            module.input.weight.normal_(std=math.sqrt(0.33 / module.input.in_features))
            module.input.bias.zero_()
            module.output.weight.zero_()
            module.output.bias.zero_()

    def identity_bits(self, ids: Tensor) -> Tensor:
        return ((ids.long().unsqueeze(-1) >> self.identity_shifts) & 1).float()

    def teacher_context(self, codes: Tensor) -> Tensor:
        # Identical to LatentPrior.forward through norm2, including zero BOS
        # features rather than the signed representation of binary code zero.
        previous = F.pad(2 * codes[:, :-1] - 1, (0, 0, 1, 0))
        state = self.prior.norm1(self.prior.input(previous.to(torch.bfloat16)))
        for block in self.prior.blocks:
            state = block(state)
        return self.prior.norm2(state)

    def score(self, context: Tensor, codes: Tensor) -> Tensor:
        logits = self.head(context, codes).float()
        return 15 * logits * (logits.square() + 225).rsqrt()

    def detached_score(self, context: Tensor, codes: Tensor) -> Tensor:
        # Use the same prefix head without letting auxiliary KL train it. The
        # context stays differentiable, including through recurrent transitions.
        parameters = {
            name: value.detach() for name, value in self.head.named_parameters()
        }
        logits = functional_call(self.head, parameters, (context, codes)).float()
        return 15 * logits * (logits.square() + 225).rsqrt()

    def advance(
        self, state: Tensor, previous_code: Tensor, age: Tensor
    ) -> tuple[Tensor, Tensor]:
        predicted = self.transition(state, previous_code)
        # Calibration may fit the transition's error but cannot alter the state
        # to make that prediction artificially easier.
        predicted_kl = self.error_predictor(predicted.detach(), age)
        return predicted, predicted_kl

    def auxiliary_losses(
        self, teacher: Tensor, teacher_logits: Tensor, codes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        teacher = teacher.detach()
        teacher_logits = teacher_logits.detach()
        horizon_count = min(self.config.rollout_horizon, codes.shape[1] - 1)
        if horizon_count == 0:
            # There is no next state to supervise, but every auxiliary parameter
            # remains connected for optimizers/DDP even on a one-character batch.
            predicted, error = self.advance(
                teacher, codes, torch.ones_like(codes[..., 0])
            )
            zero = predicted.float().sum() * 0
            return zero, zero, error.sum() * 0
        probability = teacher_logits.sigmoid()
        entropy = F.binary_cross_entropy_with_logits(
            teacher_logits, probability, reduction="none"
        ).sum(-1)
        latent_loss = teacher_logits.new_zeros(())
        rollout_kl = teacher_logits.new_zeros(())
        gate_loss = teacher_logits.new_zeros(())
        state = teacher
        for horizon in range(1, horizon_count + 1):
            # h_t predicts c_t: transition h_t with c_t to target h_{t+1}.
            # At depth j, column s tracks h_{s+j}, not h_{s+1} repeatedly.
            state, error = self.advance(
                state[:, :-1],
                codes[:, horizon - 1 : -1],
                torch.full_like(codes[:, horizon:, 0], horizon),
            )
            target_codes = codes[:, horizon:]
            logits = self.detached_score(state, target_codes)
            # Bernoulli KL sums over the actual target's causal bit prefixes,
            # retaining all mass in the binary domain (including reserved IDs).
            kl = (
                F.binary_cross_entropy_with_logits(
                    logits, probability[:, horizon:], reduction="none"
                ).sum(-1)
                - entropy[:, horizon:]
            ).clamp_min(0)  # Exact KL is nonnegative; remove FP32 cancellation.
            latent_loss = latent_loss + F.smooth_l1_loss(
                state.float(), teacher[:, horizon:].float()
            )
            rollout_kl = rollout_kl + kl.mean()
            gate_loss = gate_loss + F.mse_loss(error, kl.detach())
        return (
            latent_loss / horizon_count,
            rollout_kl / horizon_count,
            gate_loss / horizon_count,
        )

    def training_objective(self, teacher: Tensor, codes: Tensor):
        logits = self.score(teacher, codes)
        rate = binary_nll(logits, codes, 1).sum()
        latent_loss, rollout_kl, gate_loss = self.auxiliary_losses(
            teacher, logits, codes
        )
        count = codes.shape[0] * codes.shape[1]
        total = rate + count * (
            self.config.latent_weight * latent_loss
            + self.config.rollout_weight * rollout_kl
            + self.config.gate_weight * gate_loss
        )
        one = rate.new_ones(())
        return total, {
            "rate_nats": rate.detach(),
            "latent_loss": latent_loss.detach(),
            "rollout_kl": rollout_kl.detach(),
            "gate_loss": gate_loss.detach(),
            "refresh_rate": one,
            "backbone_positions_per_character": one,
            "evaluation_backbone_positions_per_character": one,
        }

    def deployed_step(
        self, state: Tensor, previous_code: Tensor, age: Tensor, teacher: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        candidate_age = age + 1
        predicted, predicted_kl = self.advance(state, previous_code, candidate_age)
        refresh = predicted_kl >= self.config.refresh_cost
        # No teacher state/error contributes to the decision above.
        context = torch.where(refresh.unsqueeze(-1), teacher, predicted)
        next_age = torch.where(refresh, 0, candidate_age)
        return context, next_age, refresh, predicted_kl

    def deployed_contexts(self, teacher: Tensor, codes: Tensor) -> dict[str, Tensor]:
        batch, length = codes.shape[:2]
        state = teacher[:, 0]
        age = torch.zeros(batch, device=codes.device, dtype=torch.long)
        contexts = [state]
        routes = [torch.ones(batch, device=codes.device, dtype=torch.bool)]
        errors = [torch.zeros(batch, device=codes.device)]
        # Per-step compiled kernels keep both route choices on device. There is
        # no per-character host synchronization or fixed refresh/chunk boundary.
        for position in range(1, length):
            state, age, refresh, error = self.deployed_step(
                state, codes[:, position - 1], age, teacher[:, position]
            )
            contexts.append(state)
            routes.append(refresh)
            errors.append(error)
        return {
            "contexts": torch.stack(contexts, dim=1),
            "routes": torch.stack(routes, dim=1),
            "predicted_kl": torch.stack(errors, dim=1),
        }

    def evaluation_objective(self, teacher: Tensor, codes: Tensor):
        rate = binary_nll(self.score(teacher, codes), codes, 1).sum()
        zero = rate.new_zeros(())
        return rate, {
            "rate_nats": rate.detach(),
            "latent_loss": zero,
            "rollout_kl": zero,
            "gate_loss": zero,
            # These describe dense likelihood evaluation, not speculative work.
            "refresh_rate": rate.new_ones(()),
            "backbone_positions_per_character": rate.new_ones(()),
            "evaluation_backbone_positions_per_character": rate.new_ones(()),
        }

    def predict(self, ids: Tensor) -> dict[str, Tensor]:
        """Historical unverified rollout diagnostic, not canonical target density."""
        codes = self.identity_bits(ids)
        teacher = self.teacher_context(codes)
        prediction = self.deployed_contexts(teacher, codes)
        prediction["logits"] = self.score(prediction["contexts"], codes)
        prediction["teacher_logits"] = self.score(teacher, codes)
        return prediction

    def forward(self, ids: Tensor, codec_weight: float = 1.0):
        codes = self.identity_bits(ids)
        teacher = self.teacher_context(codes)
        if self.training:
            return self.training_objective(teacher, codes)
        return self.evaluation_objective(teacher, codes)

    def compile_components(self):
        self.teacher_context = torch.compile(
            self.teacher_context, dynamic=True, fullgraph=True
        )
        self.score = torch.compile(self.score, dynamic=True, fullgraph=True)
        self.advance = torch.compile(self.advance, dynamic=True, fullgraph=True)
        # Training uses fixed microbatch/context shapes. Specialize the recurrent
        # loss: symbolic horizon slices break Inductor's mixed-reduction backward.
        self.training_objective = torch.compile(
            self.training_objective, dynamic=False, fullgraph=True
        )
        self.deployed_step = torch.compile(
            self.deployed_step, dynamic=True, fullgraph=True
        )
        self.evaluation_objective = torch.compile(
            self.evaluation_objective, dynamic=True, fullgraph=True
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
                raise ValueError("dynamics character transport has no residual bits")
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

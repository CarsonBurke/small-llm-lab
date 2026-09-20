"""Character softmax with a parallel recurrent accumulator and sparse event Mini.

The accumulator is a lossy, fixed-width, persistent context, not an exact codec.
Emitting does not reset it. This makes all gate inputs independent of earlier
sampled actions and permits a logarithmic-depth affine scan during training.
Only emitted states (and BOS) enter the global Transformer. Exact character ID
transport is independent of every neural component.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pretraining.nanogpt_mini.nanogpt_mini_model import Block, Linear, RMSNorm


@dataclass(frozen=True)
class EmbedderConfig:
    vocab_size: int
    num_layers: int = 6
    model_dim: int = 512
    embedder_dim: int = 128
    gate_mode: str = "learned"
    fixed_stride: int = 4
    emission_cost: float = 0.02
    baseline_loss_weight: float = 0.01

    def __post_init__(self):
        for name in (
            "vocab_size",
            "num_layers",
            "model_dim",
            "embedder_dim",
            "fixed_stride",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_dim % 128:
            raise ValueError("Mini model_dim must be a positive multiple of 128")
        if self.gate_mode not in ("learned", "fixed"):
            raise ValueError("gate_mode must be learned or fixed")
        for name in ("emission_cost", "baseline_loss_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def affine_scan(a: Tensor, b: Tensor, initial: Tensor) -> Tensor:
    """Inclusive h[t] = a[t] * h[t-1] + b[t], without division or span caps.

    Composition (a2,b2) o (a1,b1) = (a2*a1,b2+a2*b1) is associative in
    real arithmetic. This BF16 parallel scan has the usual floating-point
    reduction-order difference from sequential generation. Its O(log T) stages
    are captured by torch.compile, not a T-step Python neural loop. Work and
    activation storage are O(B*T*D*log T), reported separately from event work.
    """
    offset = 1
    while offset < a.shape[1]:
        previous_a = a
        a = torch.cat((a[:, :offset], a[:, offset:] * a[:, :-offset]), dim=1)
        b = torch.cat(
            (b[:, :offset], b[:, offset:] + previous_a[:, offset:] * b[:, :-offset]),
            dim=1,
        )
        offset *= 2
    return a * initial.unsqueeze(1) + b


class CharacterAccumulator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.bos = nn.Parameter(torch.empty(1, dim))
        self.transition = Linear(dim, 2 * dim)
        self.norm = RMSNorm(dim)

    def coefficients(self, embedded: Tensor) -> tuple[Tensor, Tensor]:
        retention, proposal = self.transition(embedded).chunk(2, dim=-1)
        retention = retention.sigmoid()
        return retention, (1 - retention) * proposal.tanh()

    def forward(self, embedded: Tensor) -> Tensor:
        initial = self.bos.to(torch.bfloat16).expand(embedded.shape[0], -1)
        a, b = self.coefficients(embedded)
        states = affine_scan(a, b, initial)
        return self.norm(torch.cat((initial.unsqueeze(1), states), dim=1))

    def step(self, embedded: Tensor, previous: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.coefficients(embedded)
        state = a * previous + b
        return state, self.norm(state)


class EventTransformer(nn.Module):
    def __init__(self, config: EmbedderConfig):
        super().__init__()
        self.input = Linear(config.embedder_dim, config.model_dim)
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


def gate_surrogate(
    gate_logits: Tensor, routes: Tensor, nll: Tensor, baseline: Tensor
) -> tuple[Tensor, Tensor]:
    """Single-route REINFORCE: action t can affect losses t+1 onward only.

    The baseline predicts mean future NLL from the UNGATED post-character
    state. Multiplying by remaining length restores return units without an
    O(T)-sized critic regression target. Its input is action-independent given
    the source sequence. Detaching the advantage avoids critic/likelihood
    gradients through the policy reward. Emission cost has its own exact
    Bernoulli-expectation derivative.
    """
    remaining = torch.arange(nll.shape[1] - 1, 0, -1, device=nll.device)
    suffix = nll[:, 1:].flip(-1).cumsum(-1).flip(-1)
    advantage = (suffix - baseline * remaining).detach()
    log_probability = -F.binary_cross_entropy_with_logits(
        gate_logits, routes.float(), reduction="none"
    )
    policy = (log_probability * advantage).sum()
    rms = (advantage.square().sum() / max(1, advantage.numel())).sqrt()
    return policy - policy.detach(), rms.detach()


class EmbedderGPT(nn.Module):
    # Compact allocation requires one host synchronization. Neural components,
    # including the scan, packing, readout and objective, are compiled separately.
    compile_forward = False
    embedding_lr = 0.7
    sum_stat_names = (
        "rate_nats",
        "compute_nats",
        "baseline_loss",
        "emitted_events",
        "useful_global_positions",
        "padded_global_positions",
        "encoder_positions",
        "gate_positions",
        "scan_compositions",
    )
    mean_stat_names = (
        "symbol_accuracy",
        "emission_rate",
        "expected_emission_rate",
        "global_positions_per_character",
        "padded_global_positions_per_character",
        "gate_entropy",
        "policy_advantage_rms",
    )

    def __init__(self, config: EmbedderConfig):
        super().__init__()
        self.config = config
        self.table = nn.Embedding(config.vocab_size, config.embedder_dim).bfloat16()
        self.encoder = CharacterAccumulator(config.embedder_dim)
        self.gate = Linear(config.embedder_dim, 1)
        # This critic receives detached local features, so its regression cannot
        # alter the representation or become an unreported language-model loss.
        self.baseline = Linear(config.embedder_dim, 1)
        self.prior = EventTransformer(config)
        self.local_output = Linear(config.embedder_dim, config.model_dim)
        self.readout_norm = RMSNorm(config.model_dim)
        self.head = Linear(config.model_dim, config.vocab_size)
        self._components_compiled = False
        self.reset_parameters()

    @property
    def transport_widths(self) -> tuple[int, int]:
        return max(1, (self.config.vocab_size - 1).bit_length()), 0

    @torch.no_grad()
    def reset_parameters(self):
        # Neither gate_mode nor fixed_stride changes parameters or RNG draws.
        for name, parameter in self.named_parameters():
            if name == "encoder.bos":
                parameter.normal_(std=math.sqrt(0.33))
            elif name == "table.weight":
                parameter.normal_()
            elif name.endswith("weight"):
                if ".proj." in name or name in ("head.weight", "baseline.weight"):
                    parameter.zero_()
                else:
                    parameter.normal_(std=math.sqrt(0.33 / parameter.shape[-1]))
            elif name.endswith("bias"):
                parameter.zero_()
            elif name.endswith("gains"):
                parameter.fill_(1)
            else:
                raise ValueError(f"uninitialized parameter: {name}")
        # Stable convex updates with a spectrum of initial memory timescales.
        self.encoder.transition.bias[: self.config.embedder_dim].copy_(
            torch.linspace(
                0, 4, self.config.embedder_dim, device=self.table.weight.device
            )
        )
        self.gate.bias.fill_(-math.log(3))
        self.baseline.bias.fill_(math.log(self.config.vocab_size))

    def compile_components(self):
        if not self._components_compiled:
            self.encode = torch.compile(self.encode, dynamic=True, fullgraph=True)
            self.choose_routes = torch.compile(
                self.choose_routes, dynamic=True, fullgraph=True
            )
            self.packed_logits = torch.compile(
                self.packed_logits, dynamic=True, fullgraph=True
            )
            self.finish_objective = torch.compile(
                self.finish_objective, dynamic=True, fullgraph=True
            )
            self._components_compiled = True

    def _check_input(self, ids: Tensor):
        if not self._components_compiled:
            raise RuntimeError("call compile_components() before neural execution")
        if ids.device.type != "cuda" or ids.device != self.table.weight.device:
            raise ValueError("embedder requires IDs and model on the same CUDA device")
        if self.table.weight.dtype != torch.bfloat16:
            raise ValueError("embedder character table must remain BF16")
        if ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 1:
            raise ValueError("ids must have nonempty shape [batch, characters]")
        if ids.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("source identities must be integers")

    def encode(self, ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Target t is never consumed before its own prediction. The final target
        # is scored, not encoded, and therefore has no unused emission decision.
        local = self.encoder(self.table(ids[:, :-1].long()))
        gate_logits = self.gate(local[:, 1:]).float().squeeze(-1)
        baseline = self.baseline(local[:, 1:].detach()).float().squeeze(-1)
        return local, gate_logits, baseline

    def choose_routes(self, gate_logits: Tensor, stochastic: bool) -> Tensor:
        if self.config.gate_mode == "fixed":
            positions = (
                torch.arange(gate_logits.shape[1], device=gate_logits.device) + 1
            )
            return (
                (positions % self.config.fixed_stride == 0)
                .unsqueeze(0)
                .expand_as(gate_logits)
            )
        if stochastic:
            return torch.rand_like(gate_logits) < gate_logits.sigmoid()
        return gate_logits >= 0

    @staticmethod
    def packed_capacity(routes: Tensor) -> int:
        maximum = int(routes.sum(-1).max().item()) + 1
        # Allocation buckets, not a span cap or compulsory emission schedule.
        if maximum == 1:
            return 1
        return min(routes.shape[1] + 1, ((maximum + 15) // 16) * 16)

    def readout(self, local: Tensor, held: Tensor) -> Tensor:
        context = self.readout_norm(held + self.local_output(local))
        logits = self.head(context).float()
        return 15 * logits * (logits.square() + 225).rsqrt()

    def packed_logits(self, local: Tensor, routes: Tensor, capacity: int) -> Tensor:
        # Raw character batches are fixed-size; only packed event capacity varies.
        # Keeping batch/character axes symbolic triggers Inductor CantSplit in
        # mixed-reduction backward at 64 x 1024. Event capacity stays dynamic.
        torch._dynamo.mark_static(local, 0)
        torch._dynamo.mark_static(local, 1)
        batch, length, dim = local.shape
        positions = torch.arange(1, length, device=local.device)
        ordered = torch.where(routes, positions, length).sort(dim=-1).values
        ordered = ordered[:, : capacity - 1]
        indices = torch.cat((ordered.new_zeros((batch, 1)), ordered), dim=1)
        events = local.gather(
            1, indices.clamp_max(length - 1).unsqueeze(-1).expand(-1, -1, dim)
        )
        events = torch.where((indices < length).unsqueeze(-1), events, 0)
        global_states = self.prior(events)
        # Event index zero is BOS; route t first becomes visible at prediction
        # t+1. Padding is after ALL real events and is never selected here.
        latest = torch.cat(
            (routes.new_zeros((batch, 1), dtype=torch.long), routes.long().cumsum(-1)),
            dim=1,
        )
        held = global_states.gather(
            1, latest.unsqueeze(-1).expand(-1, -1, self.config.model_dim)
        )
        return self.readout(local, held)

    def _run(self, ids: Tensor, routes: Tensor | None, stochastic: bool) -> dict:
        self._check_input(ids)
        local, gate_logits, baseline = self.encode(ids)
        if routes is None:
            routes = self.choose_routes(gate_logits, stochastic)
        elif (
            routes.dtype != torch.bool
            or routes.shape != gate_logits.shape
            or routes.device != ids.device
        ):
            raise ValueError(
                "routes must be boolean [batch, characters-1] on the ID device"
            )
        capacity = self.packed_capacity(routes)
        logits = self.packed_logits(local, routes, capacity)
        return {
            "logits": logits,
            "routes": routes,
            "gate_logits": gate_logits,
            "local_states": local,
            "baseline": baseline,
            "packed_capacity": capacity,
        }

    def predict(self, ids: Tensor, routes: Tensor | None = None) -> dict:
        """Score every target using deterministic deployment gates by default.

        Explicit routes are a diagnostic intervention, never the likelihood
        chosen for normal training/evaluation or generation.
        """
        return self._run(ids, routes, stochastic=False)

    def finish_objective(
        self,
        ids: Tensor,
        logits: Tensor,
        gate_logits: Tensor,
        routes: Tensor,
        baseline: Tensor,
        capacity: int,
    ):
        batch, length = ids.shape
        count = batch * length
        nll = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            ids.long().reshape(-1),
            reduction="none",
        ).reshape_as(ids)
        rate = nll.sum()
        probabilities = gate_logits.sigmoid()
        actual_events = routes.sum().float()
        expected_events = (
            probabilities.sum() if self.config.gate_mode == "learned" else actual_events
        )
        compute = self.config.emission_cost * (
            batch + (expected_events if self.training else actual_events)
        )
        # Always retain these graph edges, including fixed routes and T=1, so
        # optimizer/DDP sees zero rather than missing gradients for idle modules.
        policy = gate_logits.sum() * 0
        advantage_rms = gate_logits.new_zeros(())
        if self.training and self.config.gate_mode == "learned":
            policy, advantage_rms = gate_surrogate(gate_logits, routes, nll, baseline)
        remaining = torch.arange(length - 1, 0, -1, device=nll.device)
        suffix = nll[:, 1:].detach().flip(-1).cumsum(-1).flip(-1)
        baseline_loss = (baseline - suffix / remaining).square().sum()
        offset, compositions = 1, 0
        while offset < length - 1:
            compositions += batch * (length - 1 - offset)
            offset *= 2
        scalar = rate.detach().new_ones(())
        useful = actual_events + batch
        stats = {
            "rate_nats": rate.detach(),
            "compute_nats": compute.detach(),
            "baseline_loss": baseline_loss.detach(),
            "emitted_events": actual_events.detach(),
            "useful_global_positions": useful.detach(),
            "padded_global_positions": scalar * batch * capacity,
            "encoder_positions": scalar * batch * (length - 1),
            "gate_positions": scalar * batch * (length - 1),
            "scan_compositions": scalar * compositions,
            "symbol_accuracy": (logits.argmax(-1) == ids).float().mean().detach(),
            "emission_rate": (actual_events / max(1, batch * (length - 1))).detach(),
            "expected_emission_rate": (
                expected_events / max(1, batch * (length - 1))
            ).detach(),
            "global_positions_per_character": (useful / count).detach(),
            "padded_global_positions_per_character": scalar * capacity / length,
            "gate_entropy": (
                F.binary_cross_entropy_with_logits(
                    gate_logits, probabilities, reduction="sum"
                )
                / max(1, gate_logits.numel())
            ).detach(),
            "policy_advantage_rms": advantage_rms,
        }
        return (
            rate + compute + policy + self.config.baseline_loss_weight * baseline_loss,
            stats,
        )

    def forward(self, ids: Tensor, codec_weight: float = 1.0):
        result = self._run(ids, None, stochastic=self.training)
        # codec_weight is deliberately irrelevant: opaque IDs have no learned
        # codec objective. The policy/critic are distinct from reported BPB.
        return self.finish_objective(
            ids,
            result["logits"],
            result["gate_logits"],
            result["routes"],
            result["baseline"],
            result["packed_capacity"],
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
        width, _ = self.transport_widths
        return {
            "latents": [
                [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]
                for value in originals
            ],
            "residual": [[] for _ in originals],
        }

    def recover(self, latents: list[list[int]], residual: list[list[int]]) -> list[int]:
        if len(latents) != len(residual):
            raise ValueError("latent and residual stream lengths differ")
        width, _ = self.transport_widths
        originals = []
        for row, correction in zip(latents, residual):
            if len(correction) != 0:
                raise ValueError("character transport has no residual bits")
            if len(row) != width or any(
                type(bit) not in (bool, int) or bit not in (0, 1) for bit in row
            ):
                raise ValueError("invalid binary stream or width")
            value = 0
            for bit in row:
                value = (value << 1) | bit
            if value >= self.config.vocab_size:
                raise ValueError("decoded identity is outside the checkpoint alphabet")
            originals.append(value)
        return originals

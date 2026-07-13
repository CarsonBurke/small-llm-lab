"""Latent-thought policy modules layered over a frozen LeJEPA backbone.

The backbone remains the sequence model.  At every stream position the model
either EMITs a token (the pretrained closed loop: the sampled token is fed
back through ``embed_tokens``) or THINKs (a latent sampled from the transition
head is fed back directly; it occupies a stream position but renders nothing).

The transition policy is a diagonal heteroscedastic Gaussian over the next
projected token latent.  Its mean is the pretrained prediction path
(``prediction_projector`` applied to the belief); this module adds only a
state-dependent per-dimension log-std.  Thoughts are injected as
``z + adapter(z)`` with a zero-initialized adapter, so an untrained thought is
exactly the sampled imagined next-token latent — already in-distribution for a
trunk trained on projected token latents.

With the gate forced to EMIT at every position the wrapper is step-for-step
identical to the backbone's ``generation_step``; that property is what makes
short-context BPB comparable against pretraining and is pinned by tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

THINK, EMIT = 0, 1


class GaussianTransitionHead(nn.Module):
    """State-dependent per-dim log-std for the pretrained latent predictor.

    The mean is not owned here — callers pass the backbone's predicted latent —
    so pretrained prediction weights are reused without duplication.  The
    log-std bias starts near the backbone's observed per-dim prediction error
    (latent MSE ~0.36 => log-std ~ -0.5) with zero input weights, giving a
    state-independent but correctly scaled initial noise level.
    """

    def __init__(
        self,
        model_dim: int,
        log_std_init: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 0.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.log_std_head = nn.Linear(model_dim, model_dim)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, log_std_init)

    def log_std(self, belief: Tensor) -> Tensor:
        # Distribution parameters are FP32 statistics even under an autocast
        # region, matching the SIGReg module's convention.
        with torch.autocast(device_type=belief.device.type, enabled=False):
            return self.log_std_head(belief.float()).clamp(
                self.log_std_min, self.log_std_max
            )

    def sample(
        self, mean: Tensor, belief: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Draw one latent and return it with its log-probability."""
        log_std = self.log_std(belief)
        noise = torch.randn(
            mean.shape, device=mean.device, dtype=torch.float32, generator=generator
        )
        sample = mean.float() + log_std.exp() * noise
        return sample, self._log_prob(sample, mean.float(), log_std)

    def log_prob(self, sample: Tensor, mean: Tensor, belief: Tensor) -> Tensor:
        return self._log_prob(sample.float(), mean.float(), self.log_std(belief))

    @staticmethod
    def _log_prob(sample: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
        normalized = (sample - mean) * (-log_std).exp()
        per_dim = -0.5 * normalized.square() - log_std - 0.5 * math.log(2 * math.pi)
        return per_dim.sum(-1)

    def entropy(self, belief: Tensor) -> Tensor:
        log_std = self.log_std(belief)
        return (log_std + 0.5 * (1.0 + math.log(2 * math.pi))).sum(-1)

    def beta_nll(
        self,
        target: Tensor,
        mean: Tensor,
        belief: Tensor,
        beta: float = 0.5,
        weights: Tensor | None = None,
    ) -> Tensor:
        """Heteroscedastic NLL with detached sigma^(2*beta) weighting.

        Plain NLL scales each dimension's gradient by 1/sigma^2, stalling
        exactly where the model is uncertain; the detached weight restores a
        useful gradient there (Seitzer et al.).  ``weights`` optionally
        weights positions (e.g. masking padded stream slots).

        Reduced as a MEAN over dimensions — unlike ``log_prob``/``entropy``,
        which are proper summed log-densities — so at unit weight this loss
        is a drop-in replacement for pretraining's per-element-mean latent
        MSE: at init (constant log-std s, beta=0.5) its gradient through the
        mean is e^{-s}/2 times the MSE gradient, not model_dim/2 times.
        """
        log_std = self.log_std(belief)
        per_dim = (
            0.5 * ((target.float() - mean.float()) * (-log_std).exp()).square()
            + log_std
            + 0.5 * math.log(2 * math.pi)
        )
        scale = (2.0 * beta * log_std).exp().detach()
        per_position = (per_dim * scale).mean(-1)
        if weights is None:
            return per_position.mean()
        weights = weights.float()
        return (per_position * weights).sum() / weights.sum().clamp_min(1.0)


class ThinkEmitGate(nn.Module):
    """Bernoulli THINK/EMIT policy over the belief; zero-init is exactly 50/50."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.head = nn.Linear(model_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def emit_logit(self, belief: Tensor) -> Tensor:
        with torch.autocast(device_type=belief.device.type, enabled=False):
            return self.head(belief.float()).squeeze(-1)

    def sample(
        self, belief: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Sample actions (EMIT=1/THINK=0) with their log-probabilities."""
        logit = self.emit_logit(belief)
        probability = logit.sigmoid()
        uniform = torch.rand(
            probability.shape,
            device=probability.device,
            dtype=probability.dtype,
            generator=generator,
        )
        action = (uniform < probability).long()
        return action, self.log_prob(action, belief)

    def log_prob(self, action: Tensor, belief: Tensor) -> Tensor:
        logit = self.emit_logit(belief)
        return -F.binary_cross_entropy_with_logits(
            logit, action.float(), reduction="none"
        )

    def entropy(self, belief: Tensor) -> Tensor:
        logit = self.emit_logit(belief)
        probability = logit.sigmoid()
        return F.binary_cross_entropy_with_logits(logit, probability, reduction="none")


class ThoughtAdapter(nn.Module):
    """Residual correction for thought injection; zero-init passes z through."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.correction = nn.Linear(model_dim, model_dim, bias=False)
        nn.init.zeros_(self.correction.weight)

    def forward(self, thought: Tensor) -> Tensor:
        return thought + self.correction(thought)


@dataclass
class StepOutput:
    """Everything one stream step exposes to rollout and training code."""

    belief: Tensor
    predicted: Tensor
    input_latent: Tensor
    logits: Tensor
    value: Tensor
    caches: list[tuple[Tensor, ...]]


class LatentThoughtModel(nn.Module):
    """Frozen-backbone wrapper adding gate, transition noise, and thoughts.

    ``step`` consumes one already-embedded stream input (token latent or
    injected thought) and mirrors the backbone's ``generation_step`` block
    loop exactly; ``test_latent_thought.py`` pins the equivalence.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        model_dim = backbone.tok_emb.embedding_dim
        self.transition = GaussianTransitionHead(model_dim)
        self.gate = ThinkEmitGate(model_dim)
        self.adapter = ThoughtAdapter(model_dim)

    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        return self.backbone.embed_tokens(token_ids)

    def make_generation_cache(self, batch_size: int, max_length: int, device: torch.device):
        return self.backbone.make_generation_cache(batch_size, max_length, device)

    def step(
        self,
        input_latent: Tensor,
        caches: list[tuple[Tensor, ...]],
        position: int | Tensor,
    ) -> StepOutput:
        """Advance one stream position from an embedded input."""
        backbone = self.backbone
        x = input_latent
        skips: list[Tensor] = []
        next_caches = list(caches)
        for i in range(backbone.num_encoder_layers):
            x, next_caches[i] = backbone._block_step(
                backbone.blocks[i], x, input_latent, caches[i], position
            )
            skips.append(x)
        for j in range(backbone.num_decoder_layers):
            i = backbone.num_encoder_layers + j
            if skips:
                x = x + backbone.skip_weights[j].to(x.dtype)[None, None] * skips.pop()
            x, next_caches[i] = backbone._block_step(
                backbone.blocks[i], x, input_latent, caches[i], position
            )
        belief = backbone.final_norm(x)
        predicted = backbone.prediction_latent(belief)
        features = backbone.generation_probe_features(input_latent, belief, predicted)
        logits = backbone.logits_from_features(features).squeeze(1)
        value = backbone.values_from_features(features).squeeze(1)
        return StepOutput(
            belief=belief.squeeze(1),
            predicted=predicted.squeeze(1),
            input_latent=input_latent.squeeze(1),
            logits=logits,
            value=value,
            caches=next_caches,
        )

    def token_step(
        self,
        token_ids: Tensor,
        caches: list[tuple[Tensor, ...]],
        position: int | Tensor,
    ) -> StepOutput:
        return self.step(self.embed_tokens(token_ids[:, None]), caches, position)

    def thought_input(self, thought: Tensor) -> Tensor:
        # The adapter runs in fp32 on the raw thought and the result is
        # rounded to the embedding dtype afterwards — the same cast order as
        # ``assemble_stream_latents`` — so rollout and replay agree exactly
        # and the fp32 adapter never sees a low-precision operand.
        return self.adapter(thought.float())[:, None].to(
            self.backbone.tok_emb.weight.dtype
        )

    def new_parameters(self):
        """Post-training parameters that do not exist in the pretrained checkpoint."""
        for module in (self.transition, self.gate, self.adapter):
            yield from module.parameters()

    def load_backbone_checkpoint(self, state: dict[str, Tensor]) -> None:
        """Strict backbone load: every checkpoint key must land in the backbone."""
        self.backbone.load_state_dict(state, strict=True)

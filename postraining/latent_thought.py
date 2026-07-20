"""Latent-thought policy modules layered over the LeJEPA backbone.

The backbone remains the sequence model.  At every stream position the model
either EMITs a token (the pretrained closed loop: the sampled token is fed
back through ``embed_tokens``) or THINKs (a latent sampled from the transition
head is fed back directly; it occupies a stream position but renders nothing).

The transition policy is a diagonal Gaussian with FIXED sigma whose mean IS
the backbone's prediction path (``prediction_latent`` applied to the belief).
The whole trunk trains by policy gradient at RL time, shifting the prediction
objective from "what the next latent WILL be" (pretraining likelihood) to
"what it SHOULD be" (reward).  Thoughts pass through an identity-initialized
affine embedder, so an untrained thought is exactly the sampled imagined
next-token latent.  During post-training, its bias can become a shared
thought-type marker while its weight translates thought content into the
trunk's learned thought representation.

The renderer is deliberately separated from that thought path: it consumes
the current stream input and the raw belief, while ``prediction_latent`` is
reserved for the continuous thought policy.  Consequently emitted-token
losses train the belief/trunk but do not directly train the prediction
projector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

THINK, EMIT = 0, 1
RENDERER_FEATURES_SCHEMA = "input_latent+belief/v1"
ROLLOUT_POLICY_SCHEMA = "half_group_members_forced_initial_latent_think/v1"
THOUGHT_INPUT_SCHEMA = "identity_init_affine/v1"


def validate_renderer_checkpoint(payload: dict, checkpoint: str) -> None:
    """Reject wrapper checkpoints trained with incompatible policy semantics."""
    actual = payload.get("renderer_features_schema")
    if actual != RENDERER_FEATURES_SCHEMA:
        raise ValueError(
            f"incompatible latent-policy checkpoint {checkpoint!r}: renderer "
            f"schema is {actual!r}, expected {RENDERER_FEATURES_SCHEMA!r}. "
            "Old or untagged VAPO checkpoints used predicted-latent renderer "
            "features and cannot be resumed or evaluated as this policy."
        )
    rollout_policy = payload.get("rollout_policy_schema")
    if rollout_policy != ROLLOUT_POLICY_SCHEMA:
        raise ValueError(
            f"incompatible latent-policy checkpoint {checkpoint!r}: rollout "
            f"schema is {rollout_policy!r}, expected {ROLLOUT_POLICY_SCHEMA!r}. "
            "Old or untagged VAPO checkpoints used a different forced-initial "
            "assignment and cannot be resumed or evaluated as this policy."
        )
    thought_input = payload.get("thought_input_schema")
    if thought_input != THOUGHT_INPUT_SCHEMA:
        raise ValueError(
            f"incompatible latent-policy checkpoint {checkpoint!r}: thought "
            f"input schema is {thought_input!r}, expected "
            f"{THOUGHT_INPUT_SCHEMA!r}. Old or untagged VAPO checkpoints "
            "used a different thought embedder and cannot be resumed or "
            "evaluated as this policy."
        )


class GaussianTransitionHead(nn.Module):
    """Thought policy: the backbone's predicted latent under fixed noise.

    The mean is not owned here — callers pass the backbone's predicted
    latent — and at RL time the policy gradient flows through it into the
    whole trunk: the prediction path itself is trained from "what the next
    latent WILL be" toward "what it SHOULD be".  Sigma is a fixed,
    state-independent constant (a buffer, not a parameter): plain
    exploration noise scaled to the backbone's pretraining prediction
    error (latent MSE ~0.36 => log-sigma ~ -0.5).  There is no learned
    uncertainty, no entropy bonus, and no likelihood objective on this
    head — the joint-action PPO trust region is the only constraint.
    """

    def __init__(self, model_dim: int, log_sigma: float = -0.5):
        super().__init__()
        del model_dim  # kept for call-site symmetry with the other heads
        self.register_buffer("log_sigma", torch.tensor(float(log_sigma)))

    def sample(
        self, mean: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Draw one latent and return it with its (summed) log-probability."""
        sample = self.sample_latent(mean, generator)
        return sample, self.log_prob(sample, mean.float())

    def sample_latent(
        self, mean: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        """Draw one latent without computing a likelihood (evaluation)."""
        # Distribution statistics are FP32 even under an autocast region,
        # matching the SIGReg module's convention.
        mean = mean.float()
        noise = torch.randn(
            mean.shape, device=mean.device, dtype=torch.float32, generator=generator
        )
        sample = mean + self.log_sigma.exp() * noise
        return sample

    def log_prob(self, sample: Tensor, mean: Tensor) -> Tensor:
        return self.per_dim_log_prob(sample, mean).sum(-1)

    def per_dim_log_prob(self, sample: Tensor, mean: Tensor) -> Tensor:
        """Per-dimension log-density of the thought policy, (…, dim).

        Replay stores these factors individually, then sums them into the
        complete Gaussian-vector log probability before applying one joint
        action PPO ratio. With sigma constant, the ratio is driven purely by
        mean movement — i.e. by the trunk.
        """
        log_sigma = self.log_sigma
        normalized = (sample.float() - mean.float()) * (-log_sigma).exp()
        return -0.5 * normalized.square() - log_sigma - 0.5 * math.log(2 * math.pi)


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
        log_probability = -F.binary_cross_entropy_with_logits(
            logit, action.float(), reduction="none"
        )
        return action, log_probability

    def sample_action(
        self, belief: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        """Sample an action without computing its likelihood (evaluation)."""
        probability = self.emit_logit(belief).sigmoid()
        uniform = torch.rand(
            probability.shape,
            device=probability.device,
            dtype=probability.dtype,
            generator=generator,
        )
        return (uniform < probability).long()

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
    """Affine thought embedder initialized to the identity transformation.

    Deliberately un-normalized (LeWM feeds raw predictions back the same
    way): the prediction's magnitude carries the model's confidence, the
    trunk's pre-norms make branch computations scale-invariant anyway, and
    the loop cannot blow up — the belief is RMS-normed before the projector
    and the projector re-norms before its output layer, so ``predicted`` is
    bounded by learned weight norms, not by anything that compounds across
    rollout steps.
    """

    def __init__(self, model_dim: int):
        super().__init__()
        self.projection = nn.Linear(model_dim, model_dim, bias=True)
        nn.init.eye_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, thought: Tensor) -> Tensor:
        return self.projection(thought)


@dataclass
class StepOutput:
    """Everything one stream step exposes to rollout and training code.

    No value: the critic is a separate model that scores stored streams in
    parallel (``refresh_old_statistics``); the stepwise path never values.
    """

    belief: Tensor
    predicted: Tensor
    input_latent: Tensor
    logits: Tensor
    caches: list[tuple[Tensor, ...]]


class LatentThoughtModel(nn.Module):
    """Backbone wrapper adding gate, transition noise, and thought injection.

    ``step`` consumes one already-embedded stream input (token latent or
    injected thought) and mirrors the backbone's block loop.  Its renderer
    intentionally differs from the pretraining generation helper: vocab
    logits read the raw belief, not the projected thought mean.
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

    def make_generation_cache(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ):
        return self.backbone.make_generation_cache(
            batch_size, max_length, device, dtype=dtype
        )

    @staticmethod
    def renderer_features(input_latent: Tensor, belief: Tensor) -> Tensor:
        """Features for vocab rendering, independent of the thought projector.

        The policy probe keeps its pretrained 2*model_dim input shape, but its
        contextual half is the raw temporal belief.  ``prediction_latent`` is
        exclusively the mean of the continuous thought action.
        """
        return torch.cat((input_latent, belief), dim=-1)

    def thought_mean(self, belief: Tensor) -> Tensor:
        """Project a belief into the continuous thought policy's mean."""
        return self.backbone.prediction_latent(belief)

    def policy_logits(self, input_ids: Tensor) -> Tensor:
        """Teacher-forced vocab logits under the deployed belief renderer."""
        input_latent = self.embed_tokens(input_ids)
        belief = self.backbone.temporal_belief_from_token_latent(input_latent)
        return self.backbone.logits_from_features(
            self.renderer_features(input_latent, belief)
        )

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Teacher-forced CE used by the policy-renderer BPB guard."""
        logits = self.policy_logits(input_ids)
        return F.cross_entropy(logits.float().flatten(0, 1), target_ids.flatten())

    def make_static_generation_cache(
        self,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ):
        """Preallocated caches for the fixed-shape (``key_mask``) step path.

        Zero-filled once — masked slots must stay finite or the full-cache
        SDPA turns their garbage scores into NaN — and marked as static
        addresses so a CUDA-graph capture of ``step`` may mutate them in
        place across replays.  Reuse one cache set across rollouts of the
        same shape: a fresh allocation forces a graph re-record.
        """
        caches = self.make_generation_cache(
            batch_size, cache_length, device, dtype=dtype
        )
        # Cache tuples are architecture-defined (RoPE stores K/V pairs, PoPE
        # stores k_real/k_imag/value triples) — treat them generically.
        for cache in caches:
            for tensor in cache:
                tensor.zero_()
                torch._dynamo.mark_static_address(tensor)
        return caches

    def step_core(
        self,
        input_latent: Tensor,
        caches: list[tuple[Tensor, ...]],
        position: int | Tensor,
        key_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """The compiled surface: ``(belief, predicted, logits)`` out.

        Caches are mutated strictly in place. The thought mean is computed
        densely to preserve the launch-efficient batched path, but it is not
        a renderer feature: token losses have no graph edge into it.

        ``key_mask`` selects the static full-cache attention path (see
        ``_attention_step``): shapes stay constant across positions, which is
        what makes this method capturable as a single CUDA graph.  No
        dataclass construction and no cache aliasing cross the compile
        boundary — the trainer patches THIS method with torch.compile.
        """
        backbone = self.backbone
        x = input_latent
        skips: list[Tensor] = []
        for i in range(backbone.num_encoder_layers):
            # Every _attention_step variant mutates the cache tensors in
            # place and returns the same objects; the returned handle is
            # deliberately dropped so mutated inputs never alias outputs
            # inside a CUDA-graph capture.
            x, _ = backbone._block_step(
                backbone.blocks[i], x, input_latent, caches[i], position, key_mask
            )
            skips.append(x)
        for j in range(backbone.num_decoder_layers):
            i = backbone.num_encoder_layers + j
            if skips:
                x = x + backbone.skip_weights[j].to(x.dtype)[None, None] * skips.pop()
            x, _ = backbone._block_step(
                backbone.blocks[i], x, input_latent, caches[i], position, key_mask
            )
        belief = backbone.final_norm(x)
        predicted = self.thought_mean(belief)
        features = self.renderer_features(input_latent, belief)
        logits = backbone.logits_from_features(features).squeeze(1)
        return belief.squeeze(1), predicted.squeeze(1), logits

    def step(
        self,
        input_latent: Tensor,
        caches: list[tuple[Tensor, ...]],
        position: int | Tensor,
        key_mask: Tensor | None = None,
    ) -> StepOutput:
        """Advance one stream position from an embedded input."""
        belief, predicted, logits = self.step_core(
            input_latent, caches, position, key_mask
        )
        return StepOutput(
            belief=belief,
            predicted=predicted,
            input_latent=input_latent.squeeze(1),
            logits=logits,
            caches=list(caches),
        )

    def token_step(
        self,
        token_ids: Tensor,
        caches: list[tuple[Tensor, ...]],
        position: int | Tensor,
        key_mask: Tensor | None = None,
    ) -> StepOutput:
        return self.step(
            self.embed_tokens(token_ids[:, None]), caches, position, key_mask
        )

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

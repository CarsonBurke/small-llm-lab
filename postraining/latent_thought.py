"""Latent-thought policy modules layered over the LeJEPA backbone.

The backbone remains the sequence model.  At every stream position the model
either EMITs a token (the pretrained closed loop: the sampled token is fed
back through ``embed_tokens``) or THINKs (a latent sampled from the transition
head is fed back directly; it occupies a stream position but renders nothing).

The transition policy is a diagonal Gaussian whose mean IS the backbone's
prediction path (``prediction_latent`` applied to the belief) and whose
per-dimension log-sigma is predicted from that same belief.
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
THOUGHT_DISTRIBUTION_SCHEMA = "state_dependent_diag_tanh_log_sigma_-5_2/v1"
THOUGHT_LOG_SIGMA_MIN = -5.0
THOUGHT_LOG_SIGMA_MAX = 2.0


def validate_renderer_checkpoint(
    payload: dict,
    checkpoint: str,
    *,
    allow_transition_reset: bool = False,
) -> None:
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
    thought_distribution = payload.get("thought_distribution_schema")
    if (
        thought_distribution != THOUGHT_DISTRIBUTION_SCHEMA
        and not allow_transition_reset
    ):
        raise ValueError(
            f"incompatible latent-policy checkpoint {checkpoint!r}: thought "
            f"distribution schema is {thought_distribution!r}, expected "
            f"{THOUGHT_DISTRIBUTION_SCHEMA!r}. Old or untagged state-sigma "
            "checkpoints used unbounded log-sigma outputs and cannot be "
            "loaded without changing their policy."
        )


class GaussianTransitionHead(nn.Module):
    """Thought policy: the backbone's predicted latent under learned,
    state-dependent noise.

    The mean is not owned here — callers pass the backbone's predicted
    latent — and at RL time the policy gradient flows through it into the
    whole trunk: the prediction path itself is trained from "what the next
    latent WILL be" toward "what it SHOULD be".  The noise IS owned here:
    a zero-init linear head over the belief predicts per-dimension
    log-sigma around a CLI-initialized bias, so exploration starts
    state-independent (and small) and the joint-action policy objective learns
    where and how much to explore — no schedule, entropy bonus, or likelihood
    auxiliary on this head. With one fresh behavior minibatch per optimizer
    step, the learning rate controls each post-ratio update directly. Callers
    compute
    ``predict_log_sigma(belief)`` once per site and pass it to every
    sampling/scoring method so rollout, refresh, and update always price
    the same distribution.
    """

    def __init__(self, model_dim: int, log_sigma: float = -2.0):
        super().__init__()
        self.log_sigma_head = nn.Linear(model_dim, model_dim)
        self.reset_noise(log_sigma)

    def reset_noise(self, log_sigma: float) -> None:
        """Zero state dependence and initialize an exact bounded log-sigma.

        Zero weights make the head state-independent at initialization —
        exactly the retired scalar policy — with the CLI owning the level.
        The learned affine head lives in raw space, so the desired log-sigma
        is inverse-transformed into its bias.
        """
        raw_bias = self.raw_from_log_sigma(log_sigma)
        with torch.no_grad():
            self.log_sigma_head.weight.zero_()
            self.log_sigma_head.bias.fill_(raw_bias)

    @staticmethod
    def raw_from_log_sigma(log_sigma: float) -> float:
        """Inverse of the fixed tanh bound for scalar initialization."""
        log_sigma = float(log_sigma)
        if not math.isfinite(log_sigma):
            raise ValueError("log-sigma must be finite")
        if not THOUGHT_LOG_SIGMA_MIN < log_sigma < THOUGHT_LOG_SIGMA_MAX:
            raise ValueError(
                "log-sigma must be strictly inside "
                f"({THOUGHT_LOG_SIGMA_MIN}, {THOUGHT_LOG_SIGMA_MAX}); "
                f"got {log_sigma}"
            )
        midpoint = (THOUGHT_LOG_SIGMA_MIN + THOUGHT_LOG_SIGMA_MAX) / 2.0
        half_range = (THOUGHT_LOG_SIGMA_MAX - THOUGHT_LOG_SIGMA_MIN) / 2.0
        return math.atanh((log_sigma - midpoint) / half_range)

    @staticmethod
    def bound_raw_log_sigma(raw: Tensor) -> Tensor:
        """Map unconstrained head output smoothly into the fixed safe range."""
        midpoint = (THOUGHT_LOG_SIGMA_MIN + THOUGHT_LOG_SIGMA_MAX) / 2.0
        half_range = (THOUGHT_LOG_SIGMA_MAX - THOUGHT_LOG_SIGMA_MIN) / 2.0
        return midpoint + half_range * raw.tanh()

    def predict_log_sigma(self, belief: Tensor) -> Tensor:
        """Per-dimension log-sigma of the thought policy at this state.

        Smoothly bounded to [-5, 2]. Under autocast, the state-dependent matrix
        multiply uses bf16, but it deliberately excludes the raw-space bias:
        a small residual near zero retains fine bf16 absolute resolution,
        then the fp32 bias addition returns fp32 distribution statistics.
        Including the bias in an autocast linear would quantize early changes
        around -2 to roughly 0.008 increments.
        """
        low_precision = belief.dtype in (torch.bfloat16, torch.float16)
        with torch.autocast(
            device_type=belief.device.type,
            dtype=belief.dtype if low_precision else None,
            enabled=low_precision,
        ):
            residual = F.linear(
                belief, self.log_sigma_head.weight, bias=None
            )
        raw = residual.float() + self.log_sigma_head.bias.float()
        return self.bound_raw_log_sigma(raw)

    def sample(
        self,
        mean: Tensor,
        log_sigma: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Draw one latent and return it with its (summed) log-probability."""
        sample = self.sample_latent(mean, log_sigma, generator)
        return sample, self.log_prob(sample, mean.float(), log_sigma)

    def sample_latent(
        self,
        mean: Tensor,
        log_sigma: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw one latent without computing a likelihood (evaluation)."""
        mean = mean.float()
        noise = torch.randn(
            mean.shape, device=mean.device, dtype=torch.float32, generator=generator
        )
        return mean + log_sigma.float().exp() * noise

    def log_prob(self, sample: Tensor, mean: Tensor, log_sigma: Tensor) -> Tensor:
        return self.per_dim_log_prob(sample, mean, log_sigma).sum(-1)

    def per_dim_log_prob(
        self, sample: Tensor, mean: Tensor, log_sigma: Tensor
    ) -> Tensor:
        """Per-dimension log-density of the thought policy, (…, dim).

        Replay stores these factors individually, then sums them into the
        complete Gaussian-vector log probability before applying one joint
        action PPO ratio. The ratio moves with the mean (the trunk) and
        with the state-dependent sigma; refresh_old_statistics recomputes
        old factors under the current head, so epoch-0 ratios stay exactly 1.
        """
        log_sigma = log_sigma.float()
        normalized = (sample.float() - mean.float()) * (-log_sigma).exp()
        return -0.5 * normalized.square() - log_sigma - 0.5 * math.log(2 * math.pi)


def migrate_scalar_log_sigma_state(
    state_dict: dict, head: GaussianTransitionHead, prefix: str = "transition."
) -> bool:
    """Rewrite a legacy scalar-sigma transition state into the head layout.

    The retired policy stored one global ``transition.log_sigma``; a
    zero-weight head whose inverse-transformed bias carries that scalar is the
    identical distribution, so old checkpoints stay loadable and
    behaviorally exact.
    Returns True when a migration was applied.
    """
    key = prefix + "log_sigma"
    if key not in state_dict:
        return False
    scalar = float(state_dict.pop(key))
    raw_bias = head.raw_from_log_sigma(scalar)
    state_dict[prefix + "log_sigma_head.weight"] = torch.zeros_like(
        head.log_sigma_head.weight
    )
    state_dict[prefix + "log_sigma_head.bias"] = torch.full_like(
        head.log_sigma_head.bias, raw_bias
    )
    return True


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


def migrate_residual_thought_adapter_state(
    state_dict: dict,
    adapter: ThoughtAdapter,
    prefix: str = "adapter.",
) -> bool:
    """Rewrite the retired ``z + correction(z)`` adapter exactly.

    The old biasless residual map is algebraically the affine map
    ``(I + correction.weight) @ z + 0``.  This is a representation migration,
    not a reinitialization, so it is safe for both pristine and trained actor
    or critic checkpoints.
    """
    old_key = prefix + "correction.weight"
    if old_key not in state_dict:
        return False
    weight_key = prefix + "projection.weight"
    bias_key = prefix + "projection.bias"
    if weight_key in state_dict or bias_key in state_dict:
        raise ValueError("checkpoint contains both residual and affine adapters")
    correction = state_dict.pop(old_key)
    if correction.shape != adapter.projection.weight.shape:
        raise ValueError(
            f"legacy thought adapter shape {tuple(correction.shape)} does not "
            f"match {tuple(adapter.projection.weight.shape)}"
        )
    identity = torch.eye(
        correction.size(0), dtype=correction.dtype, device=correction.device
    )
    state_dict[weight_key] = identity + correction
    state_dict[bias_key] = torch.zeros(
        correction.size(0), dtype=correction.dtype, device=correction.device
    )
    return True


def migrate_legacy_wrapper_checkpoint(
    payload: dict,
    wrapper: "LatentThoughtModel",
) -> tuple[bool, bool]:
    """Apply exact state-layout migrations before strict validation/loading."""
    state_dict = payload["model"]
    adapter_migrated = migrate_residual_thought_adapter_state(
        state_dict, wrapper.adapter
    )
    sigma_migrated = migrate_scalar_log_sigma_state(
        state_dict, wrapper.transition
    )
    if adapter_migrated:
        schema = payload.get("thought_input_schema")
        if schema not in (None, THOUGHT_INPUT_SCHEMA):
            raise ValueError(
                f"cannot migrate unknown thought input schema {schema!r}"
            )
        payload["thought_input_schema"] = THOUGHT_INPUT_SCHEMA
    if sigma_migrated:
        schema = payload.get("thought_distribution_schema")
        if schema not in (None, THOUGHT_DISTRIBUTION_SCHEMA):
            raise ValueError(
                f"cannot migrate unknown thought distribution schema {schema!r}"
            )
        payload["thought_distribution_schema"] = THOUGHT_DISTRIBUTION_SCHEMA
    return adapter_migrated, sigma_migrated


@dataclass
class StepOutput:
    """Everything one stream step exposes to rollout and training code.

    No value: the critic is a separate model that scores stored streams in
    parallel (``refresh_old_statistics``); the stepwise path never values.
    """

    belief: Tensor
    predicted: Tensor
    thought_log_sigma: Tensor
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """The compiled surface: belief, thought mean/log-sigma, and logits.

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
        thought_log_sigma = self.transition.predict_log_sigma(belief.squeeze(1))
        features = self.renderer_features(input_latent, belief)
        logits = backbone.logits_from_features(features).squeeze(1)
        return belief.squeeze(1), predicted.squeeze(1), thought_log_sigma, logits

    def step(
        self,
        input_latent: Tensor,
        caches: list[tuple[Tensor, ...]],
        position: int | Tensor,
        key_mask: Tensor | None = None,
    ) -> StepOutput:
        """Advance one stream position from an embedded input."""
        belief, predicted, thought_log_sigma, logits = self.step_core(
            input_latent, caches, position, key_mask
        )
        return StepOutput(
            belief=belief,
            predicted=predicted,
            thought_log_sigma=thought_log_sigma,
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

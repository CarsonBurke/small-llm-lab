"""Latent-thought policy modules layered over the LeJEPA backbone.

The backbone remains the sequence model.  At every stream position the model
either EMITs a token (the pretrained closed loop: the sampled token is fed
back through ``embed_tokens``) or THINKs (a latent sampled from the transition
head is fed back directly; it occupies a stream position but renders nothing).

The transition policy is a diagonal Gaussian whose mean comes from a fresh
linear head over the belief and whose per-dimension log-sigma is predicted
from that same belief. The mean starts as a zero-bias orthogonal map at gain
0.1, with no initial obligation to imitate a discrete-token embedding.
Thoughts pass through a separate full-width orthogonal projection and
``2*SiLU`` after sampling. The factor of two makes the adapter locally
unit-gain at zero while the nonlinearity gives the policy exclusive processing
before the shared trunk. The adapter is recurrent policy state, not part of
the Gaussian likelihood.

The renderer is deliberately separated from that thought path: it consumes
the current stream input and the raw belief, while the fresh mean head is
reserved for the continuous thought policy. Consequently emitted-token losses
train the belief/trunk but do not directly train the mean head.
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
# Pinned-EMIT reasoning modes never sample the gate or a thought: the rollout
# is a plain token policy. The tag embeds the mode because cot and none differ
# in their trained emission budgets, so their checkpoints are not one policy.
PINNED_EMIT_ROLLOUT_POLICY_SCHEMAS = {
    "cot": "pinned_emit_token_only_cot/v1",
    "none": "pinned_emit_token_only_answer_prefix/v1",
}


def rollout_policy_schema_for_mode(reasoning_mode: str) -> str:
    """The rollout-policy schema tag a reasoning mode trains and resumes."""
    if reasoning_mode == "latent":
        return ROLLOUT_POLICY_SCHEMA
    try:
        return PINNED_EMIT_ROLLOUT_POLICY_SCHEMAS[reasoning_mode]
    except KeyError:
        raise ValueError(f"unknown reasoning mode {reasoning_mode!r}") from None
IDENTITY_AFFINE_THOUGHT_INPUT_SCHEMA = "fresh_identity_affine/v7"
ORTHOGONAL_SILU_THOUGHT_INPUT_SCHEMA = "fresh_orthogonal_affine_2silu/v8"
THOUGHT_INPUT_SCHEMA = ORTHOGONAL_SILU_THOUGHT_INPUT_SCHEMA
COMPATIBLE_IDENTITY_AFFINE_THOUGHT_INPUT_SCHEMAS = frozenset(
    {
        IDENTITY_AFFINE_THOUGHT_INPUT_SCHEMA,
        "fresh_scaled_eye_0.1_affine/v6",
        "fresh_zero_affine/v5",
    }
)
THOUGHT_ADAPTER_KINDS = ("identity_affine", "orthogonal_silu")
THOUGHT_ACTION_TRANSFORM_KINDS = ("identity", "tanh")
THOUGHT_ACTION_TRANSFORM_SCHEMAS = {
    "identity": "raw_gaussian_recurrent_input/v1",
    "tanh": "tanh_raw_gaussian_recurrent_input/v1",
}
SIGMA_STATE_INIT_KINDS = ("constant", "orthogonal")
CRITIC_ADAPTER_INIT_KINDS = ("identity", "orthogonal")
SIGMA_STATE_INIT_SCHEMAS = {
    "constant": "zero_weight_constant_sigma/v1",
    "orthogonal": "unit_orthogonal_weight_gain_0.01/v2",
}
CRITIC_ADAPTER_INIT_SCHEMAS = {
    "identity": "identity_affine/v1",
    "orthogonal": "unit_orthogonal_affine/v2",
}
THOUGHT_DISTRIBUTION_SCHEMA = (
    "state_dependent_diag_tanh_log_sigma_scaled_residual_-5_2/v2"
)
THOUGHT_MEAN_SCHEMA = "fresh_linear_learned_output_gain_zero_bias/v3"
# The initial gain is not part of a trained policy's runtime semantics: its
# checkpointed scalar completely determines the function. Keep v2 resumable
# so the currently running policy remains recoverable, while every explicit
# fresh actor restart is relabeled with the initialization-agnostic schema.
COMPATIBLE_THOUGHT_MEAN_SCHEMAS = frozenset(
    {
        THOUGHT_MEAN_SCHEMA,
        "fresh_linear_learned_output_gain_0.01_zero_bias/v2",
    }
)
THOUGHT_LOG_SIGMA_MIN = -5.0
THOUGHT_LOG_SIGMA_MAX = 2.0


def thought_input_schema_for_adapter(kind: str) -> str:
    """Return the deployed actor-adapter schema for an ablation kind."""
    if kind == "identity_affine":
        return IDENTITY_AFFINE_THOUGHT_INPUT_SCHEMA
    if kind == "orthogonal_silu":
        return ORTHOGONAL_SILU_THOUGHT_INPUT_SCHEMA
    raise ValueError(f"unknown thought adapter kind {kind!r}")


def transform_thought_action(thought: Tensor, kind: str) -> Tensor:
    """Map a raw Gaussian action into the recurrent input consumed by a trunk.

    Rollout storage and policy likelihoods remain in the raw Gaussian space.
    This transform belongs only at actor/critic stream-input boundaries.
    """
    thought = thought.float()
    if kind == "identity":
        return thought
    if kind == "tanh":
        return thought.tanh()
    raise ValueError(f"unknown thought action transform {kind!r}")


def wrapper_init_kwargs_from_checkpoint(payload: dict) -> dict[str, str]:
    """Recover policy semantics before constructing a checkpoint wrapper.

    Checkpoints predating the selectable initializations have no matching
    argument fields and used the identity actor adapter with constant sigma.
    The tensor layouts match the new treatment, so callers must recover these
    semantics before strict loading rather than relying on shape checks.
    """
    saved_args = payload.get("args", {})
    return {
        "thought_adapter": saved_args.get(
            "thought_adapter", "identity_affine"
        ),
        "sigma_state_init": saved_args.get(
            "thought_sigma_state_init", "constant"
        ),
        "thought_action_transform": saved_args.get(
            "thought_action_transform", "identity"
        ),
    }


def validate_renderer_checkpoint(
    payload: dict,
    checkpoint: str,
    *,
    allow_transition_reset: bool = False,
    expected_rollout_policy_schema: str = ROLLOUT_POLICY_SCHEMA,
    expected_thought_input_schema: str = THOUGHT_INPUT_SCHEMA,
    expected_thought_action_transform_schema: str = (
        THOUGHT_ACTION_TRANSFORM_SCHEMAS["identity"]
    ),
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
    if rollout_policy != expected_rollout_policy_schema:
        raise ValueError(
            f"incompatible latent-policy checkpoint {checkpoint!r}: rollout "
            f"schema is {rollout_policy!r}, expected "
            f"{expected_rollout_policy_schema!r}. The checkpoint was trained "
            "under a different reasoning mode or forced-initial assignment "
            "and cannot be resumed or evaluated as this policy."
        )
    thought_input = payload.get("thought_input_schema")
    thought_input_matches = thought_input == expected_thought_input_schema
    if (
        expected_thought_input_schema == IDENTITY_AFFINE_THOUGHT_INPUT_SCHEMA
        and thought_input
        in COMPATIBLE_IDENTITY_AFFINE_THOUGHT_INPUT_SCHEMAS
    ):
        thought_input_matches = True
    if not thought_input_matches:
        raise ValueError(
            f"incompatible latent-policy checkpoint {checkpoint!r}: thought "
            f"input schema is {thought_input!r}, expected "
            f"{expected_thought_input_schema!r}. The checkpoint uses a "
            "different deployed thought adapter and cannot be resumed or "
            "evaluated as this policy."
        )
    thought_action_transform = payload.get(
        "thought_action_transform_schema",
        THOUGHT_ACTION_TRANSFORM_SCHEMAS["identity"],
    )
    if thought_action_transform != expected_thought_action_transform_schema:
        raise ValueError(
            f"incompatible latent-policy checkpoint {checkpoint!r}: thought "
            f"action transform schema is {thought_action_transform!r}, "
            f"expected {expected_thought_action_transform_schema!r}. The "
            "checkpoint recurrently consumed a different action and cannot "
            "be resumed or evaluated as this policy."
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
    thought_mean = payload.get("thought_mean_schema")
    if thought_mean not in COMPATIBLE_THOUGHT_MEAN_SCHEMAS:
        raise ValueError(
            f"incompatible latent-policy checkpoint {checkpoint!r}: thought "
            f"mean schema is {thought_mean!r}, expected one of "
            f"{sorted(COMPATIBLE_THOUGHT_MEAN_SCHEMAS)!r}. Use the explicit "
            "fresh-mean branch "
            "migration to replace a legacy pretrained-projector mean; normal "
            "resume and evaluation cannot change policy semantics."
        )


class FreshThoughtMeanHead(nn.Linear):
    """Orthogonal belief-to-mean map behind a learned small output gain.

    Storing the initial 0.1 gain directly in every matrix element is
    functionally equivalent only before optimization. Adam's first 3e-4 step
    can move each element by an amount comparable to the initialized weight,
    coherently changing a D-wide output by O(D * lr). Keeping a unit-scale
    orthogonal map behind one learned 0.1 gain preserves the exact initial
    function while scaling the functional effect of matrix updates.
    """

    INIT_OUTPUT_GAIN = 0.1

    def __init__(self, model_dim: int):
        super().__init__(model_dim, model_dim, bias=True)
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)
        self.output_gain = nn.Parameter(torch.tensor(self.INIT_OUTPUT_GAIN))

    def forward(self, belief: Tensor) -> Tensor:
        low_precision = belief.dtype in (torch.bfloat16, torch.float16)
        with torch.autocast(
            device_type=belief.device.type,
            dtype=belief.dtype if low_precision else None,
            enabled=low_precision,
        ):
            projected = F.linear(belief, self.weight, bias=None)
        return self.output_gain.float() * projected.float() + self.bias.float()

    def reset_output_gain(self, output_gain: float) -> None:
        """Set the fresh-policy gain without changing its orthogonal map."""
        output_gain = float(output_gain)
        if not math.isfinite(output_gain) or output_gain <= 0.0:
            raise ValueError("thought mean output gain must be finite and positive")
        with torch.no_grad():
            self.output_gain.fill_(output_gain)


class StateDependentLogSigmaHead(nn.Linear):
    """Raw log-sigma bias plus a learned, weak orthogonal state residual."""

    INIT_RESIDUAL_GAIN = 0.01

    def __init__(self, model_dim: int):
        super().__init__(model_dim, model_dim, bias=True)
        self.residual_gain = nn.Parameter(
            torch.tensor(self.INIT_RESIDUAL_GAIN)
        )


class GaussianTransitionHead(nn.Module):
    """Fresh diagonal-Gaussian thought policy over the current belief.

    A unit-orthogonal mean map sits behind a learned output gain initialized
    at 0.1. For an RMS-normalized D-wide belief this gives a mean RMS of
    exactly 0.1 while preserving every input direction and avoiding the
    retired next-token latent prior. A separate unit-orthogonal linear head
    predicts per-dimension log-sigma residuals behind a 0.01 output gain around
    a CLI-initialized bias. The head is therefore mildly state-dependent from
    its first rollout, while the explicit gain controls both its initial
    function and Adam's scale-insensitive matrix updates.
    Callers compute
    ``predict_log_sigma(belief)`` once per site and pass it to every
    sampling/scoring method so rollout, refresh, and update always price
    the same distribution.
    """

    MEAN_INIT_GAIN = FreshThoughtMeanHead.INIT_OUTPUT_GAIN

    def __init__(
        self,
        model_dim: int,
        log_sigma: float = -2.0,
        sigma_state_init: str = "orthogonal",
    ):
        super().__init__()
        if sigma_state_init not in SIGMA_STATE_INIT_KINDS:
            raise ValueError(f"unknown sigma state init {sigma_state_init!r}")
        self.sigma_state_init = sigma_state_init
        # Keep log-sigma registered first. Legacy v11 actor optimizers stored
        # this pair as their fifth group; the fresh mean becomes a sixth group
        # so the one-time branch migration can restore every old Adam state
        # without positional remapping.
        self.log_sigma_head = StateDependentLogSigmaHead(model_dim)
        self.mean_head = FreshThoughtMeanHead(model_dim)
        self.reset_noise(log_sigma, sigma_state_init)

    def predict_mean(self, belief: Tensor) -> Tensor:
        """State-dependent mean of the continuous thought action.

        The matrix multiply follows the backbone's low-precision inference
        policy, while the initially zero FP32 bias is added afterward. This
        keeps tiny learned offsets representable instead of quantizing them at
        the scale of the projected mean, and it also makes direct bf16 callers
        dtype-safe outside an enclosing autocast context.
        """
        return self.mean_head(belief)

    def reset_noise(
        self, log_sigma: float, sigma_state_init: str | None = None
    ) -> None:
        """Initialize bounded log-sigma around a CLI-owned statewise mean.

        A unit-orthogonal raw-space map preserves the RMS-one belief geometry.
        Its explicit 0.01 gain gives only about 0.03 RMS log-sigma variation
        near the usual initialization range, while retaining full rank and an
        immediate gradient for the learned gain. The inverse-transformed bias
        remains the exact center of the bounded distribution. ``constant`` is
        retained only as the matched zero-weight ablation.
        """
        sigma_state_init = sigma_state_init or self.sigma_state_init
        if sigma_state_init not in SIGMA_STATE_INIT_KINDS:
            raise ValueError(f"unknown sigma state init {sigma_state_init!r}")
        self.sigma_state_init = sigma_state_init
        raw_bias = self.raw_from_log_sigma(log_sigma)
        with torch.no_grad():
            if sigma_state_init == "orthogonal":
                # Linear.__init__ already consumed the same global RNG as the
                # retired zero-weight head. Draw the new direction without
                # shifting later fresh modules or the critic's initialization.
                devices = (
                    [self.log_sigma_head.weight.device]
                    if self.log_sigma_head.weight.is_cuda
                    else []
                )
                with torch.random.fork_rng(devices=devices):
                    nn.init.orthogonal_(self.log_sigma_head.weight)
            else:
                self.log_sigma_head.weight.zero_()
            self.log_sigma_head.bias.fill_(raw_bias)
            self.log_sigma_head.residual_gain.fill_(
                self.log_sigma_head.INIT_RESIDUAL_GAIN
            )

    def set_noise_level(self, log_sigma: float) -> None:
        """Change only the fresh policy's central log-sigma.

        Construction owns the random orthogonal direction. CLI scale selection
        must not draw it again: repeated initialization would consume RNG and
        silently change every later fresh module.
        """
        raw_bias = self.raw_from_log_sigma(log_sigma)
        with torch.no_grad():
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
        raw = (
            self.log_sigma_head.residual_gain.float() * residual.float()
            + self.log_sigma_head.bias.float()
        )
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

        Replay stores these factors individually and clips each dimension's
        PPO ratio separately while retaining the diagonal Gaussian's summed
        score gradient. The ratios move with the mean (the trunk) and with
        the state-dependent sigma; refresh_old_statistics recomputes old
        factors under the current head, so behavior-age-0 ratios stay exactly
        1.
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
    state_dict[prefix + "log_sigma_head.residual_gain"] = (
        head.log_sigma_head.residual_gain.detach().clone()
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


class AffineThoughtAdapter(nn.Module):
    """Full-width affine thought embedder with an explicit initialization."""

    def __init__(self, model_dim: int, initialization: str = "orthogonal"):
        super().__init__()
        if initialization not in CRITIC_ADAPTER_INIT_KINDS:
            raise ValueError(
                f"unknown affine thought-adapter initialization {initialization!r}"
            )
        self.initialization = initialization
        self.projection = nn.Linear(model_dim, model_dim, bias=True)
        self.reset_affine(initialization)

    def reset_affine(self, initialization: str | None = None) -> None:
        initialization = initialization or self.initialization
        if initialization not in CRITIC_ADAPTER_INIT_KINDS:
            raise ValueError(
                f"unknown affine thought-adapter initialization {initialization!r}"
            )
        self.initialization = initialization
        with torch.no_grad():
            if initialization == "orthogonal":
                # Preserve the global stream left by Linear.__init__, matching
                # the old identity reset's RNG consumption.
                devices = (
                    [self.projection.weight.device]
                    if self.projection.weight.is_cuda
                    else []
                )
                with torch.random.fork_rng(devices=devices):
                    nn.init.orthogonal_(self.projection.weight)
            else:
                nn.init.eye_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    def forward(self, thought: Tensor) -> Tensor:
        return self.projection(thought)


class ThoughtAdapter(AffineThoughtAdapter):
    """Policy thought mixer: affine control or orthogonal ``2*SiLU``."""

    def __init__(self, model_dim: int, kind: str = "orthogonal_silu"):
        if kind not in THOUGHT_ADAPTER_KINDS:
            raise ValueError(f"unknown thought adapter kind {kind!r}")
        initialization = (
            "identity" if kind == "identity_affine" else "orthogonal"
        )
        super().__init__(model_dim, initialization=initialization)
        self.kind = kind

    def forward(self, thought: Tensor) -> Tensor:
        projected = self.projection(thought)
        if self.kind == "identity_affine":
            return projected
        # SiLU'(0)=1/2, so the factor of two gives a unit-gain nonlinear
        # interface around the small initial thought distribution.
        return 2.0 * F.silu(projected)


def migrate_legacy_wrapper_checkpoint(
    payload: dict,
    wrapper: "LatentThoughtModel",
    *,
    initialize_fresh_mean: bool = False,
    initialize_fresh_adapter: bool = False,
) -> tuple[bool, bool, bool]:
    """Apply exact state-layout migrations before strict validation/loading."""
    state_dict = payload["model"]
    adapter_reset = False
    if initialize_fresh_adapter:
        # Explicit actor restart: discard any legacy recurrent adapter rather
        # than silently inheriting its learned latent protocol. The critic-
        # warm source has an untouched actor optimizer, so no corresponding
        # Adam state exists to migrate.
        state_dict.pop("adapter.correction.weight", None)
        state_dict["adapter.projection.weight"] = (
            wrapper.adapter.projection.weight.detach().clone()
        )
        state_dict["adapter.projection.bias"] = (
            wrapper.adapter.projection.bias.detach().clone()
        )
        state_dict.pop("adapter.interpolation_strength", None)
        payload["thought_input_schema"] = wrapper.thought_input_schema
        adapter_reset = True
    else:
        if "adapter.correction.weight" in state_dict:
            raise ValueError(
                "legacy residual thought adapters cannot be resumed into the "
                "current thought policy; use an explicit actor restart"
            )
    sigma_weight_key = "transition.log_sigma_head.weight"
    sigma_bias_key = "transition.log_sigma_head.bias"
    sigma_gain_key = "transition.log_sigma_head.residual_gain"
    if initialize_fresh_mean:
        # A critic-warm checkpoint has never deployed its actor. Replace the
        # whole fresh noise head, including the random orthogonal direction,
        # instead of loading the checkpoint's obsolete untouched init and
        # drawing a second direction after load.
        state_dict.pop("transition.log_sigma", None)
        state_dict[sigma_weight_key] = (
            wrapper.transition.log_sigma_head.weight.detach().clone()
        )
        state_dict[sigma_bias_key] = (
            wrapper.transition.log_sigma_head.bias.detach().clone()
        )
        state_dict[sigma_gain_key] = (
            wrapper.transition.log_sigma_head.residual_gain.detach().clone()
        )
        payload["thought_distribution_schema"] = THOUGHT_DISTRIBUTION_SCHEMA
        sigma_migrated = True
    else:
        sigma_migrated = migrate_scalar_log_sigma_state(
            state_dict, wrapper.transition
        )
    mean_weight_key = "transition.mean_head.weight"
    mean_bias_key = "transition.mean_head.bias"
    mean_gain_key = "transition.mean_head.output_gain"
    present_mean_keys = {
        key
        for key in (mean_weight_key, mean_bias_key, mean_gain_key)
        if key in state_dict
    }
    if present_mean_keys and len(present_mean_keys) != 3:
        raise ValueError(
            "checkpoint contains a partial fresh thought-mean head: "
            f"{sorted(present_mean_keys)}"
        )
    mean_migrated = False
    if initialize_fresh_mean:
        # An explicit actor restart owns the complete fresh thought policy.
        # Critic-warm checkpoints can already contain an older untouched mean
        # head, so checking only for missing keys would silently retain that
        # experiment's initialization instead of the requested one.
        state_dict[mean_weight_key] = (
            wrapper.transition.mean_head.weight.detach().clone()
        )
        state_dict[mean_bias_key] = (
            wrapper.transition.mean_head.bias.detach().clone()
        )
        state_dict[mean_gain_key] = (
            wrapper.transition.mean_head.output_gain.detach().clone()
        )
        payload["thought_mean_schema"] = THOUGHT_MEAN_SCHEMA
        mean_migrated = True
    if sigma_migrated:
        schema = payload.get("thought_distribution_schema")
        if schema not in (None, THOUGHT_DISTRIBUTION_SCHEMA):
            raise ValueError(
                f"cannot migrate unknown thought distribution schema {schema!r}"
            )
        payload["thought_distribution_schema"] = THOUGHT_DISTRIBUTION_SCHEMA
    return sigma_migrated, mean_migrated, adapter_reset


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

    def __init__(
        self,
        backbone: nn.Module,
        *,
        thought_adapter: str = "orthogonal_silu",
        sigma_state_init: str = "orthogonal",
        thought_action_transform: str = "identity",
    ):
        super().__init__()
        self.backbone = backbone
        model_dim = backbone.tok_emb.embedding_dim
        self.thought_adapter_kind = thought_adapter
        self.thought_input_schema = thought_input_schema_for_adapter(
            thought_adapter
        )
        if thought_action_transform not in THOUGHT_ACTION_TRANSFORM_KINDS:
            raise ValueError(
                f"unknown thought action transform {thought_action_transform!r}"
            )
        self.thought_action_transform = thought_action_transform
        self.thought_action_transform_schema = (
            THOUGHT_ACTION_TRANSFORM_SCHEMAS[thought_action_transform]
        )
        if sigma_state_init not in SIGMA_STATE_INIT_KINDS:
            raise ValueError(f"unknown sigma state init {sigma_state_init!r}")
        self.sigma_state_init = sigma_state_init
        self.sigma_state_init_schema = SIGMA_STATE_INIT_SCHEMAS[
            sigma_state_init
        ]
        self.transition = GaussianTransitionHead(
            model_dim, sigma_state_init=sigma_state_init
        )
        self.gate = ThinkEmitGate(model_dim)
        self.adapter = ThoughtAdapter(model_dim, kind=thought_adapter)

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
        """Features for vocab rendering, independent of the thought mean.

        The policy probe keeps its pretrained 2*model_dim input shape, but its
        contextual half is the raw temporal belief.
        """
        return torch.cat((input_latent, belief), dim=-1)

    def thought_mean(self, belief: Tensor) -> Tensor:
        """Map a belief into the fresh continuous thought policy's mean."""
        return self.transition.predict_mean(belief)

    def policy_logits(self, input_ids: Tensor) -> Tensor:
        """Teacher-forced vocab logits under the deployed belief renderer."""
        input_latent = self.embed_tokens(input_ids)
        belief = self.backbone.temporal_belief_from_token_latent(input_latent)
        return self.backbone.logits_from_features(
            self.renderer_features(input_latent, belief)
        )

    # Tokens rendered per chunk in the teacher-forced CE below. The full
    # [batch, seq, vocab] logits tensor (plus the softcap temporaries) does
    # not fit on one GPU for large vocabularies — at GPT-2's 50257 vocab a
    # 64x1024 eval batch needs tens of GiB — while the trunk activations
    # feeding it are small. 8192 tokens keeps each logits chunk under ~2 GiB.
    BPB_EVAL_CHUNK_TOKENS = 8192

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Teacher-forced CE used by the policy-renderer BPB guard.

        Renders logits in token chunks and reduces with a sum so the result
        is the exact token mean one-shot ``cross_entropy`` would return.
        """
        input_latent = self.embed_tokens(input_ids)
        belief = self.backbone.temporal_belief_from_token_latent(input_latent)
        features = self.renderer_features(input_latent, belief).flatten(0, 1)
        targets = target_ids.flatten()
        loss_sum = torch.zeros((), device=features.device, dtype=torch.float32)
        for start in range(0, targets.numel(), self.BPB_EVAL_CHUNK_TOKENS):
            stop = start + self.BPB_EVAL_CHUNK_TOKENS
            logits = self.backbone.logits_from_features(features[start:stop])
            loss_sum += F.cross_entropy(
                logits.float(), targets[start:stop], reduction="sum"
            )
        return loss_sum / targets.numel()

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

        Caches are mutated strictly in place. The fresh thought mean is
        computed densely to preserve the launch-efficient batched path, but it
        is not a renderer feature: token losses have no graph edge into it.

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

    def prefill_core(
        self,
        input_latent: Tensor,
        caches: list[tuple[Tensor, ...]],
        key_valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Densely ingest a deterministic prefix and expose its final state."""
        belief = self.backbone.prefill_belief(
            input_latent, caches, key_valid
        )[:, -1:]
        final_input = input_latent[:, -1:]
        predicted = self.thought_mean(belief)
        thought_log_sigma = self.transition.predict_log_sigma(
            belief.squeeze(1)
        )
        logits = self.backbone.logits_from_features(
            self.renderer_features(final_input, belief)
        ).squeeze(1)
        return (
            belief.squeeze(1),
            predicted.squeeze(1),
            thought_log_sigma,
            logits,
        )

    def prefill(
        self,
        token_ids: Tensor,
        caches: list[tuple[Tensor, ...]],
        key_valid: Tensor | None = None,
    ) -> StepOutput:
        """Populate prefix caches without running policy heads per token."""
        input_latent = self.embed_tokens(token_ids)
        belief, predicted, thought_log_sigma, logits = self.prefill_core(
            input_latent, caches, key_valid
        )
        return StepOutput(
            belief=belief,
            predicted=predicted,
            thought_log_sigma=thought_log_sigma,
            input_latent=input_latent[:, -1],
            logits=logits,
            caches=list(caches),
        )

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
        # The transform and adapter run in fp32 and the result is
        # rounded to the embedding dtype afterwards — the same cast order as
        # ``assemble_stream_latents`` — so rollout and replay agree exactly
        # and the fp32 adapter never sees a low-precision operand.
        return self.adapt_thought_action(thought)[:, None].to(
            self.backbone.tok_emb.weight.dtype
        )

    def adapt_thought_action(self, thought: Tensor) -> Tensor:
        """Transform one raw Gaussian action, then apply the actor adapter."""
        return self.adapter(
            transform_thought_action(thought, self.thought_action_transform)
        )

    def new_parameters(self):
        """Post-training parameters that do not exist in the pretrained checkpoint."""
        for module in (self.transition, self.gate, self.adapter):
            yield from module.parameters()

    def load_backbone_checkpoint(self, state: dict[str, Tensor]) -> None:
        """Strict backbone load: every checkpoint key must land in the backbone."""
        self.backbone.load_state_dict(state, strict=True)

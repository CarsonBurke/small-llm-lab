"""Geometry-preserving categorical RBF decoder for LeJEPA.

The JEPA path retains its original responsibilities:

* attached MSE aligns the predicted latent with the actual next-token latent;
* paired SIGReg shapes the encoded target geometry and prevents collapse.

Discrete token probabilities are read directly from that geometry.  A small
uncertainty head predicts one log-radius per position from the causal belief,
and the vocabulary receives one learned prior bias per token::

    log_sigma_t = w_sigma @ stopgrad(belief_t) + b_sigma
    logit_tk = b_k - precision_t
                     * ||stopgrad(predicted_t) - stopgrad(code_k)||^2 / 2

The categorical NLL is therefore a calibration objective only: its gradient
cannot move the predictor, token projector, embedding table, or transformer.
Those parameters remain owned by MSE/SIGReg.  At inference, ``policy_logits``
returns the normalized decoder's logits; callers sample the categorical token
distribution directly.  ``log_sigma`` is an RBF temperature/radius, not the
standard deviation of a separately sampled continuous latent.

Run the canonical ablation through the shared GPU queue::

    mlq submit --name jepa_rbf_decoder_2k --cwd "$PWD" \
      --max-parallel-runs 1 -- python3 scripts/ablation.py --steps 2000 \
      --name jepa_rbf_decoder_2k \
      --script energy_readout/fresh_lejepa_train_jepa_rbf_decoder.py \
      --env DATA_PATH=data/datasets/fineweb_onepass_sp1024 \
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

from __future__ import annotations

import inspect
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import train_gpt as baseline
from pretraining.fresh_lejepa import fresh_lejepa_train as v1
from pretraining.fresh_lejepa import fresh_lejepa_train_v1_probe_shared_rms_pope as pope
from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope_belief_attached import (
    FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE,
)


ARCHITECTURE = "fresh_lejepa_pope_jepa_owned_geometry_rbf_decoder_onepass_2k"
PRECISION_MIN = float(os.environ.get("RBF_PRECISION_MIN", "1e-5"))
PRECISION_MAX = float(os.environ.get("RBF_PRECISION_MAX", "100"))


class GeometryPreservingRBFLeJEPA(
    FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE
):
    """LeJEPA MSE/SIGReg with a detached-geometry categorical RBF decoder."""

    latent_loss_weight = v1.FreshHyperparameters.latent_loss_weight
    sigreg_loss_weight = v1.FreshHyperparameters.sigreg_weight

    def __init__(self, *args, **kwargs):
        vocab_size = kwargs.get("vocab_size", args[0] if args else None)
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        super().__init__(*args, **kwargs)
        if vocab_size is None or model_dim is None:
            raise ValueError("vocab_size and model_dim are required")

        owner = self.blocks[-1]
        # Constructing and then deleting the parent's probes preserves the RNG
        # stream and initialization of every surviving JEPA parameter.
        del owner.policy_probe
        del owner.critic_probe

        # All decoder parameters are vectors/scalars registered under blocks,
        # so baseline.main routes them to the fp32 Adam control-parameter group.
        # Zero w_sigma makes the initial radius context-independent.  The base
        # sigma=d^0.25 gives inverse variance 1/sqrt(d), matching the calibrated
        # near-uniform initialization used by the existing energy family.
        owner.rbf_log_sigma_weight = nn.Parameter(
            torch.zeros(int(model_dim), dtype=torch.float32)
        )
        owner.rbf_log_sigma_bias = nn.Parameter(
            torch.tensor(0.25 * math.log(int(model_dim)), dtype=torch.float32)
        )
        owner.rbf_token_bias = nn.Parameter(
            torch.zeros(int(vocab_size), dtype=torch.float32)
        )
        owner.register_buffer(
            "rbf_log_precision_bounds_nano",
            torch.tensor(
                [
                    round(math.log(PRECISION_MIN) * 1_000_000_000),
                    round(math.log(PRECISION_MAX) * 1_000_000_000),
                ],
                dtype=torch.int64,
            ),
        )

        leftover = [name for name, _ in self.named_parameters() if "probe" in name]
        if leftover:
            raise RuntimeError(f"probe parameters survived deletion: {leftover}")

    @property
    def rbf_owner(self) -> nn.Module:
        return self.blocks[-1]

    def semantic_codebook(self) -> Tensor:
        """Return the live JEPA target codebook, detached from token NLL."""
        # policy_codebook uses the exact target projector on the normalized
        # tied embedding table and already detaches its result.
        return self.policy_codebook()

    def position_log_sigma(self, belief: Tensor) -> Tensor:
        """Predict one categorical RBF log-radius per position.

        Belief is detached deliberately: NLL calibrates uncertainty without
        changing the causal representation learned by JEPA MSE/SIGReg.
        """
        owner = self.rbf_owner
        weight = owner.rbf_log_sigma_weight.to(dtype=belief.dtype)
        projected = F.linear(belief.detach(), weight.unsqueeze(0)).squeeze(-1)
        return projected.float() + owner.rbf_log_sigma_bias

    def precision_log_bounds(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        """Decode checkpointed bounds without BF16 model-cast rounding."""
        bounds = self.rbf_owner.rbf_log_precision_bounds_nano.to(
            device=reference.device,
            dtype=reference.dtype,
        ) * 1e-9
        return bounds[0], bounds[1]

    def energy_logits(self, predicted: Tensor, belief: Tensor) -> Tensor:
        """Map detached JEPA geometry to categorical vocabulary logits."""
        predicted = predicted.detach()
        codebook = self.semantic_codebook()
        log_sigma = self.position_log_sigma(belief)

        # Convert before leaving autocast and use cdist's direct kernel.  The
        # GEMM identity recovers a small distance from three O(d) terms and
        # catastrophically cancels for nearby BF16 vectors; direct differences
        # retain the geometry without changing process-wide TF32 settings.
        with torch.autocast(device_type=predicted.device.type, enabled=False):
            predicted_fp32 = predicted.float()
            codebook_fp32 = codebook.float()
            distance_sq = torch.cdist(
                predicted_fp32,
                codebook_fp32,
                p=2.0,
                compute_mode="donot_use_mm_for_euclid_dist",
            ).square()

        # A straight-through bound is a numerical domain constraint, not a
        # fitted temperature hyperparameter.  Healthy values see the exact
        # exponential parameterization and gradient; corrupt/extreme values
        # cannot create inf * 0 or an all-infinite categorical distribution.
        log_precision = -2.0 * log_sigma
        log_precision_min, log_precision_max = self.precision_log_bounds(
            log_precision
        )
        bounded_log_precision = log_precision.clamp(
            min=log_precision_min,
            max=log_precision_max,
        )
        safe_log_precision = log_precision + (
            bounded_log_precision - log_precision
        ).detach()
        precision = safe_log_precision.exp().unsqueeze(-1)
        return self.rbf_owner.rbf_token_bias - 0.5 * precision * distance_sq

    def token_nll(
        self,
        predicted: Tensor,
        belief: Tensor,
        target_ids: Tensor,
    ) -> Tensor:
        logits = self.energy_logits(predicted, belief)
        return F.cross_entropy(
            logits.flatten(0, 1), target_ids.flatten(), reduction="mean"
        )

    @staticmethod
    def decoder_features(input_latent: Tensor, belief: Tensor) -> Tensor:
        """Use the repository's standard renderer feature schema."""
        return torch.cat((input_latent.detach(), belief.detach()), dim=-1)

    def split_decoder_features(self, features: Tensor) -> tuple[Tensor, Tensor]:
        model_dim = self.tok_emb.embedding_dim
        if features.size(-1) != 2 * model_dim:
            raise ValueError(
                "RBF decoder features must concatenate predicted and belief "
                f"latents ({2 * model_dim} channels), got {features.size(-1)}"
            )
        return features.split(model_dim, dim=-1)

    def detached_probe_features(self, input_ids: Tensor) -> Tensor:
        token_latent = self.embed_tokens(input_ids)
        belief = self.temporal_belief_from_token_latent(token_latent)
        return self.decoder_features(token_latent, belief)

    def generation_probe_features(
        self, token_latent: Tensor, belief: Tensor, predicted: Tensor
    ) -> Tensor:
        del predicted
        return self.decoder_features(token_latent, belief)

    def generation_policy_step(
        self,
        token_ids: Tensor,
        caches: list[tuple[Tensor, ...]],
        position: int | Tensor,
    ) -> tuple[Tensor, list[tuple[Tensor, ...]]]:
        """Consume one token and return categorical logits plus KV caches.

        The inherited ``generation_step`` is an actor-critic API and requires
        a value prediction.  This pretraining decoder intentionally has no
        critic, so discrete generation uses this policy-only counterpart.
        """
        token_latent = self.embed_tokens(token_ids[:, None])
        predicted = token_latent
        skips: list[Tensor] = []
        next_caches = list(caches)
        for index in range(self.num_encoder_layers):
            predicted, next_caches[index] = self._block_step(
                self.blocks[index],
                predicted,
                token_latent,
                caches[index],
                position,
            )
            skips.append(predicted)
        for decoder_index in range(self.num_decoder_layers):
            index = self.num_encoder_layers + decoder_index
            if skips:
                skip_weight = self.skip_weights[decoder_index].to(predicted.dtype)
                predicted = predicted + skip_weight[None, None] * skips.pop()
            predicted, next_caches[index] = self._block_step(
                self.blocks[index],
                predicted,
                token_latent,
                caches[index],
                position,
            )
        belief = self.final_norm(predicted)
        features = self.generation_probe_features(
            token_latent, belief, belief
        )
        return self.logits_from_features(features).squeeze(1), next_caches

    def generation_step(self, *args, **kwargs):
        """Reject the actor-critic API before mutating its supplied caches."""
        del args, kwargs
        raise RuntimeError(
            "this decoder has no value head; use generation_policy_step for "
            "cached discrete-token generation"
        )

    def logits_from_features(self, features: Tensor) -> Tensor:
        _input_latent, belief = self.split_decoder_features(features)
        predicted = self.prediction_latent(belief)
        return self.energy_logits(predicted, belief)

    def renderer_parameters(self):
        """Yield the NLL-owned parameters used by SFT/evaluation tooling."""
        yield self.rbf_owner.rbf_log_sigma_weight
        yield self.rbf_owner.rbf_log_sigma_bias
        yield self.rbf_owner.rbf_token_bias

    def values_from_features(self, features: Tensor) -> Tensor:
        raise RuntimeError(
            "the geometry-preserving RBF decoder has no value head; "
            "policy_logits provides discrete-token logits"
        )

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        token_latent, belief, predicted, target_latent = (
            self.training_latents_with_belief(input_ids, target_ids)
        )
        policy_loss = self.token_nll(predicted, belief, target_ids)
        if not self.training:
            return policy_loss

        # This attached MSE is the authority over predictor/codebook geometry.
        latent_loss = F.mse_loss(predicted.float(), target_latent.float())
        if self.defer_sigreg:
            sigreg_loss = policy_loss.detach().new_zeros(())
        else:
            sigreg_loss = self.sigreg(
                self.training_sigreg_features(token_latent, target_latent)
            )
        total_loss = (
            policy_loss
            + self.latent_loss_weight * latent_loss
            + self.sigreg_loss_weight * sigreg_loss
        )
        if self.return_loss_components:
            return total_loss, torch.stack(
                (policy_loss.detach(), latent_loss.detach(), sigreg_loss.detach())
            )
        return total_loss

    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int | float]:
        metadata = super().experiment_metadata()
        for inherited_probe_field in (
            "ce_probe_gradient",
            "policy_probe_features",
            "prediction_projector_role",
        ):
            metadata.pop(inherited_probe_field, None)
        metadata.update(
            {
                "emission_head": "detached_geometry_categorical_rbf",
                "token_distribution": "softmax_negative_squared_distance",
                "continuous_latent_sampling": "none",
                "uncertainty": "per_position_scalar_log_sigma_from_belief",
                "uncertainty_gradient": "rbf_parameters_only",
                "rbf_precision_bounds": f"[{PRECISION_MIN:g}, {PRECISION_MAX:g}]",
                "distance_accumulation": "direct_fp32_cdist",
                "prediction_gradient_from_token_nll": "detached",
                "codebook_gradient_from_token_nll": "detached",
                "geometry_owner": "attached_mse_and_paired_sigreg",
                "latent_mse_weight": cls.latent_loss_weight,
                "sigreg_weight": cls.sigreg_loss_weight,
                "decoder_parameters": "d_log_sigma_weights_plus_scalar_plus_vocab_bias",
                "probes": "deleted_policy_and_critic",
                "cached_generation": "generation_policy_step",
                "renderer_features": "input_latent_plus_raw_belief",
                "distributed_sigreg": "single_gpu_ablation_only_pending_ddp_fix",
            }
        )
        return metadata


def run_experiment(
    model_class: type[GeometryPreservingRBFLeJEPA],
    architecture: str,
    source: Path,
) -> None:
    """Run one member of the RBF family through the shared PoPE harness."""
    if pope.PolarCausalSelfAttention.position_mode != "pope":
        raise ValueError("the geometry-preserving RBF variant is defined only for PoPE")
    if not 0.0 < PRECISION_MIN < PRECISION_MAX:
        raise ValueError(
            "RBF precision bounds must satisfy "
            f"0 < min < max, got [{PRECISION_MIN}, {PRECISION_MAX}]"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError(
            "paired SIGReg currently backpropagates outside DDP; run this 2k "
            "ablation on one GPU, then implement synchronized rank pairing "
            "before an 8-GPU scale-up"
        )

    # Compile warmup would consume the deterministic one-pass corpus prefix.
    v1.FreshHyperparameters.warmup_steps = 0
    model_class.latent_loss_weight = v1.FreshHyperparameters.latent_loss_weight
    model_class.sigreg_loss_weight = v1.FreshHyperparameters.sigreg_weight
    pope.FreshLeJEPASharedRMSV1PoPE = model_class
    pope.POPE_ARCHITECTURE = architecture
    pope.__file__ = str(source.resolve())

    # Log parameter/geometry canaries without running a second validation pass.
    original_eval_val = baseline.eval_val

    def eval_val_logging_rbf_canaries(*args, **kwargs):
        result = original_eval_val(*args, **kwargs)
        frame = inspect.currentframe().f_back
        while frame is not None:
            if "quant_state" in frame.f_locals:
                return result
            frame = frame.f_back
        if int(os.environ.get("RANK", "0")) != 0:
            return result

        model = kwargs.get("model", args[1] if len(args) > 1 else None)
        base = getattr(model, "module", model)
        base = getattr(base, "_orig_mod", base)
        with torch.no_grad():
            owner = base.rbf_owner
            base_sigma = float(torch.exp(owner.rbf_log_sigma_bias.float()))
            sigma_weight_norm = float(owner.rbf_log_sigma_weight.float().norm())
            token_bias_absmax = float(owner.rbf_token_bias.float().abs().max())
            codebook = base.semantic_codebook().float()
            codebook_norm_mean = float(codebook.norm(dim=-1).mean())
            codebook_pairdist_mean = float(torch.cdist(codebook, codebook).mean())
        print(
            f"churn_stats rbf_base_sigma:{base_sigma:.8f} "
            f"rbf_sigma_weight_norm:{sigma_weight_norm:.8f} "
            f"rbf_token_bias_absmax:{token_bias_absmax:.8f} "
            f"codebook_norm_mean:{codebook_norm_mean:.4f} "
            f"codebook_pairdist_mean:{codebook_pairdist_mean:.4f}",
            flush=True,
        )
        return result

    baseline.eval_val = eval_val_logging_rbf_canaries
    try:
        pope.main()
    finally:
        baseline.eval_val = original_eval_val


def main() -> None:
    run_experiment(
        GeometryPreservingRBFLeJEPA,
        ARCHITECTURE,
        Path(__file__),
    )


if __name__ == "__main__":
    main()

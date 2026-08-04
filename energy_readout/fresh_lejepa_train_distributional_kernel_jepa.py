"""Distributional-kernel JEPA over one exact categorical energy law.

This model has no independent CE probe and no independently predicted JEPA
point.  The distance-energy head produces one categorical distribution
``p_t`` over the live vocabulary codebook.  That same law has a nonlinear
kernel-mean representation::

    mu_t = sum_k p_t(k) phi(c_k)

Training combines the exact token log score with a kernel score against the
observed target code::

    loss_t = -log p_t(y) + lambda * ||mu_t - stopgrad(phi(c_y))||^2

``phi`` is a deterministic multi-bandwidth random Fourier approximation to a
characteristic RBF kernel on unit-normalized codes.  It distinguishes useful
nonlinear distributional structure that a linear barycenter loses, although a
finite feature map cannot be injective over all categorical laws.  Exact NLL
makes the combined score strictly proper.  The whole kernel dictionary is a
stop-gradient target geometry: the kernel score changes the categorical law,
while code geometry moves through the contrastive energy logits rather than
an unconditional every-code attraction.  Sampling is directly from ``p_t``;
no continuous latent or K future predictions are produced.

This first ablation deliberately retains the proven tied BN codebook,
per-dimension global metric, and no-SIGReg training loop.  Contextual EMA
prototypes are a separate geometry-source ablation, not bundled into the test
of distributional kernel alignment.

Run through the shared GPU queue::

    mlq submit --name distributional_kernel_jepa_fineweb_2k --cwd "$PWD" \
      --max-parallel-runs 1 -- python3 scripts/ablation.py --steps 2000 \
      --val-every 20 --name distributional_kernel_jepa_fineweb_2k \
      --script energy_readout/fresh_lejepa_train_distributional_kernel_jepa.py \
      --env DATA_PATH=data/datasets/fineweb_onepass_sp1024 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
from torch import Tensor

import train_gpt as baseline
from energy_readout import fresh_lejepa_train_energy_readout as energy
from energy_readout import fresh_lejepa_train_energy_readout_perdim_nosigreg as nosig


ARCHITECTURE = (
    "energy_readout_lejepa_pope_distributional_kernel_mean_perdim_nosigreg_onepass_2k"
)
KERNEL_JEPA_WEIGHT = float(os.environ.get("KERNEL_JEPA_WEIGHT", "1.0"))
KERNEL_FEATURES = int(os.environ.get("KERNEL_FEATURES", "64"))
KERNEL_SEED = int(os.environ.get("KERNEL_SEED", "314159"))
KERNEL_FREQUENCY_QUANTIZATION = 4096
KERNEL_BANDWIDTHS = (0.5, 1.0)


class DistributionalKernelLeJEPA(nosig.EnergyReadoutPerDimNoSigregLeJEPA):
    """Make the categorical law's kernel mean the JEPA representation."""

    latent_loss_weight = 0.0
    kernel_jepa_loss_weight = KERNEL_JEPA_WEIGHT

    def __init__(self, *args, **kwargs):
        kernel_features = int(kwargs.pop("kernel_features", KERNEL_FEATURES))
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        if model_dim is None:
            raise ValueError("model_dim is required")
        if kernel_features <= 0 or kernel_features % 2:
            raise ValueError(
                "KERNEL_FEATURES must be a positive even integer, got "
                f"{kernel_features}"
            )
        super().__init__(*args, **kwargs)

        # A local generator preserves the base model's global initialization
        # stream.  Integer fixed-point storage survives model.bfloat16() and
        # makes checkpoint reconstruction bit-exact without carrying another
        # trainable projection.  Frequencies alternate bandwidths so every
        # prefix contains both local and broad semantic comparisons.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(KERNEL_SEED)
        frequency_count = kernel_features // 2
        frequencies = torch.empty(frequency_count, int(model_dim))
        for index in range(frequency_count):
            bandwidth = KERNEL_BANDWIDTHS[index % len(KERNEL_BANDWIDTHS)]
            frequencies[index].normal_(generator=generator)
            frequencies[index].div_(bandwidth)
        quantized = torch.round(
            frequencies * KERNEL_FREQUENCY_QUANTIZATION
        ).clamp_(torch.iinfo(torch.int16).min, torch.iinfo(torch.int16).max)
        self.blocks[-1].register_buffer(
            "kernel_frequencies_q16", quantized.to(torch.int16)
        )

    def kernel_features(self, codes: Tensor) -> Tensor:
        """Map codes to unit-norm multi-scale RBF Fourier features."""
        owner = self.blocks[-1]
        with torch.autocast(device_type=codes.device.type, enabled=False):
            unit_codes = F.normalize(codes.float(), dim=-1, eps=1e-6)
            frequencies = owner.kernel_frequencies_q16.float().div(
                KERNEL_FREQUENCY_QUANTIZATION
            )
            angles = unit_codes @ frequencies.transpose(0, 1)
            scale = math.sqrt(1.0 / angles.size(-1))
            return torch.cat((angles.cos(), angles.sin()), dim=-1).mul(scale)

    def distributional_jepa_outputs(
        self,
        logits: Tensor,
        codebook: Tensor,
        target_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the law's kernel mean and its stop-gradient target."""
        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
        # Fix the feature dictionary for this scoring step.  If it were live,
        # dL/dphi(c_k) would attract every code with p_k > 0 toward every
        # observed target, and directional code collapse would zero the score.
        # Gradients still reach the codebook through logits, where softmax
        # centering supplies the required target-vs-competitor contrast.
        code_features = self.kernel_features(codebook.detach()).detach()
        predicted_mean = probabilities @ code_features
        target_features = code_features[target_ids]
        return predicted_mean, target_features

    def distributional_jepa_loss(
        self,
        logits: Tensor,
        codebook: Tensor,
        target_ids: Tensor,
    ) -> Tensor:
        predicted_mean, target_features = self.distributional_jepa_outputs(
            logits, codebook, target_ids
        )
        return (
            predicted_mean.sub(target_features).square().sum(dim=-1).mean()
        )

    def forward(self, input_ids: Tensor, target_ids: Tensor):
        token_latent, _belief, predicted, target_latent = (
            self.training_latents_with_belief(input_ids, target_ids)
        )
        logits = self.energy_logits(
            predicted, input_ids if energy.BIGRAM_TABLE else None
        )
        policy_loss = F.cross_entropy(
            logits.float().flatten(0, 1), target_ids.flatten()
        )
        if not self.training:
            return policy_loss

        codebook = self.energy_codebook()
        kernel_jepa_loss = self.distributional_jepa_loss(
            logits, codebook, target_ids
        )
        with torch.no_grad():
            latent_diagnostic = F.mse_loss(
                predicted.float(), target_latent.float()
            )
        sigreg_loss = policy_loss.detach().new_zeros(())
        total_loss = (
            policy_loss + self.kernel_jepa_loss_weight * kernel_jepa_loss
        )
        if self.return_loss_components:
            return total_loss, torch.stack(
                (
                    policy_loss.detach(),
                    latent_diagnostic,
                    sigreg_loss,
                    kernel_jepa_loss.detach(),
                )
            )
        return total_loss

    def generation_policy_step(
        self,
        token_ids: Tensor,
        caches: list[tuple[Tensor, ...]],
        position: int | Tensor,
    ) -> tuple[Tensor, list[tuple[Tensor, ...]]]:
        """Consume one token and return discrete logits without a critic."""
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
                predicted = predicted + self.skip_weights[decoder_index].to(
                    predicted.dtype
                )[None, None] * skips.pop()
            predicted, next_caches[index] = self._block_step(
                self.blocks[index],
                predicted,
                token_latent,
                caches[index],
                position,
            )
        belief = self.final_norm(predicted)
        predicted = self.prediction_latent(belief)
        input_ids = token_ids[:, None] if energy.BIGRAM_TABLE else None
        logits = self.energy_logits(predicted, input_ids).squeeze(1)
        return logits, next_caches

    def policy_logits(self, input_ids: Tensor) -> Tensor:
        """Return the exact categorical law, including optional bigram energy."""
        token_latent = self.embed_tokens(input_ids)
        belief = self.temporal_belief_from_token_latent(token_latent)
        predicted = self.prediction_latent(belief)
        energy_input_ids = input_ids if energy.BIGRAM_TABLE else None
        return self.energy_logits(predicted, energy_input_ids)

    def renderer_parameters(self):
        """Yield the small categorical metric parameters for renderer tuning."""
        yield self.blocks[-1].energy_log_scale
        yield self.blocks[-1].energy_bias

    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int | float]:
        metadata = super().experiment_metadata()
        metadata.update(
            {
                "objective": "exact_nll_plus_distributional_kernel_jepa",
                "jepa_prediction": "categorical_kernel_mean",
                "jepa_target": "stopgrad_full_code_kernel_dictionary",
                "kernel": "multiscale_rbf_random_fourier",
                "kernel_features": KERNEL_FEATURES,
                "kernel_bandwidths": ",".join(
                    str(value) for value in KERNEL_BANDWIDTHS
                ),
                "kernel_seed": KERNEL_SEED,
                "kernel_jepa_weight": cls.kernel_jepa_loss_weight,
                "categorical_objective": "exact_token_nll",
                "proper_score": "log_score_plus_low_rank_kernel_score",
                "kernel_geometry_gradient": "through_contrastive_logits_only",
                "continuous_latent_sampling": "none",
                "target_code_source": "live_tied_bn_codebook",
                "semantic_partial_credit": "nonlinear_kernel_mean_distance",
                "uncertainty": "categorical_entropy_no_position_sigma",
                "sigreg_batch": "removed_energy_geometry_measured_sufficient",
            }
        )
        return metadata


def main() -> None:
    if not math.isfinite(KERNEL_JEPA_WEIGHT) or KERNEL_JEPA_WEIGHT < 0.0:
        raise ValueError(
            "KERNEL_JEPA_WEIGHT must be finite and nonnegative, got "
            f"{KERNEL_JEPA_WEIGHT}"
        )

    original_accumulation = nosig._install_nosigreg_accumulation
    original_class = nosig.EnergyReadoutPerDimNoSigregLeJEPA
    original_architecture = nosig.ARCHITECTURE
    original_file = nosig.__file__
    original_eval_val = baseline.eval_val

    def install_distributional_accumulation(
        default_steps: int = 8,
        extra_components: tuple[tuple[str, str], ...] = (),
    ):
        if extra_components:
            raise ValueError("distributional installer owns its component schema")
        return original_accumulation(
            default_steps=default_steps,
            extra_components=(
                ("kernel_jepa_loss", "kernel_jepa_loss_weight"),
            ),
        )

    def eval_val_logging_kernel_canaries(*args, **kwargs):
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
            features = base.kernel_features(base.energy_codebook()).float()
            vocab = features.size(0)
            norm_sq_sum = features.square().sum()
            feature_sum = features.sum(dim=0)
            offdiag_similarity = float(
                (feature_sum.square().sum() - norm_sq_sum)
                / max(vocab * (vocab - 1), 1)
            )
            centered = features - features.mean(dim=0, keepdim=True)
            covariance = centered.transpose(0, 1) @ centered / max(vocab - 1, 1)
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
            normalized = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
            effective_rank = float(
                torch.exp(
                    -(normalized * normalized.clamp_min(1e-12).log()).sum()
                )
            )
        print(
            "churn_stats "
            f"kernel_offdiag_similarity:{offdiag_similarity:.8f} "
            f"kernel_effective_rank:{effective_rank:.4f}",
            flush=True,
        )
        return result

    nosig._install_nosigreg_accumulation = install_distributional_accumulation
    nosig.EnergyReadoutPerDimNoSigregLeJEPA = DistributionalKernelLeJEPA
    nosig.ARCHITECTURE = ARCHITECTURE
    nosig.__file__ = str(Path(__file__).resolve())
    baseline.eval_val = eval_val_logging_kernel_canaries
    try:
        nosig.main()
    finally:
        baseline.eval_val = original_eval_val
        nosig._install_nosigreg_accumulation = original_accumulation
        nosig.EnergyReadoutPerDimNoSigregLeJEPA = original_class
        nosig.ARCHITECTURE = original_architecture
        nosig.__file__ = original_file


if __name__ == "__main__":
    main()

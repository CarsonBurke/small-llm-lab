"""PoPE/B128 LeJEPA with attached CE rendered directly from raw belief.

The vocabulary policy consumes ``[token_latent, belief]`` in training,
teacher-forced evaluation, and incremental generation. The prediction
projector remains exclusively the latent/world-model path: latent MSE trains
it, vocabulary CE does not. This is a distinct checkpoint architecture from
the older attached-CE variant trained on projected predictions.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path

import torch
from torch import Tensor

from pretraining.fresh_lejepa import fresh_lejepa_train_v1_probe_shared_rms_pope as pope


ARCHITECTURE = (
    "fresh_lejepa_shared_rms_v1_probes_pope_belief_attached_ce_onepass_2k"
)


class FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE(
    pope.FreshLeJEPASharedRMSV1PoPE
):
    """Attach policy CE to token latents and raw temporal beliefs."""

    requires_exact_one_pass_data = True
    requires_deterministic_data = True

    @staticmethod
    def belief_probe_features(token_latent: Tensor, belief: Tensor) -> Tensor:
        return torch.cat((token_latent, belief), dim=-1)

    def training_latents_with_belief(
        self, input_ids: Tensor, target_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Input/target IDs are one shifted trajectory. Project it once so the
        # overlapping positions share exactly the same encoded values and the
        # relatively large token projector does not run twice.
        trajectory_ids = torch.cat((input_ids, target_ids[:, -1:]), dim=1)
        trajectory_latent = self.embed_tokens(trajectory_ids)
        token_latent = trajectory_latent[:, :-1]
        target_latent = trajectory_latent[:, 1:]
        belief = self.temporal_belief_from_token_latent(token_latent)
        predicted = self.prediction_latent(belief)
        return token_latent, belief, predicted, target_latent

    def training_policy_features(
        self, token_latent: Tensor, belief: Tensor | None, predicted: Tensor
    ) -> Tensor:
        del predicted
        if belief is None:
            raise ValueError("belief-attached CE requires the raw temporal belief")
        return self.belief_probe_features(token_latent, belief)

    def detached_probe_features(self, input_ids: Tensor) -> Tensor:
        token_latent = self.embed_tokens(input_ids)
        belief = self.temporal_belief_from_token_latent(token_latent)
        return self.belief_probe_features(token_latent.detach(), belief.detach())

    def generation_probe_features(
        self, token_latent: Tensor, belief: Tensor, predicted: Tensor
    ) -> Tensor:
        del predicted
        return self.belief_probe_features(token_latent.detach(), belief.detach())

    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int]:
        metadata = super().experiment_metadata()
        metadata.update(
            {
                "ce_probe_gradient": "attached_token_latent_and_raw_belief",
                "policy_probe_features": "token_latent_plus_raw_belief",
                "prediction_projector_role": "latent_loss_and_thought_policy_only",
                "data_order": "deterministic_one_pass_no_rng",
            }
        )
        return metadata


def main() -> None:
    if pope.PolarCausalSelfAttention.position_mode != "pope":
        raise ValueError("the belief-attached CE variant is defined only for PoPE")
    # Compile warmup would consume the corpus prefix twice even though its
    # state is restored. The one-pass contract forbids that reuse.
    pope.v1.FreshHyperparameters.warmup_steps = 0
    pope.FreshLeJEPASharedRMSV1PoPE = FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE
    pope.POPE_ARCHITECTURE = ARCHITECTURE
    pope.__file__ = str(Path(__file__).resolve())
    pope.main()


if __name__ == "__main__":
    main()

"""PoPE/B128 LeJEPA variant with CE attached to the latent backbone.

This is a controlled fork of ``fresh_lejepa_train_v1_probe_shared_rms_pope``.
The policy head, objective values, initialization, and optimizer configuration
are unchanged; only the stop-gradients on the policy loss's token and
predicted-latent inputs are removed. Standalone policy/critic helpers retain
their original detached behavior.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

import fresh_lejepa_train_v1_probe_shared_rms_pope as pope


ARCHITECTURE = "fresh_lejepa_shared_rms_v1_probes_pope_attached_ce_scratch_2k"


class FreshLeJEPASharedRMSV1PoPEAttachedCE(pope.FreshLeJEPASharedRMSV1PoPE):
    """Let policy CE update both token and predicted-latent feature paths."""

    def policy_loss_features(
        self, token_latent: Tensor, predicted: Tensor
    ) -> Tensor:
        return torch.cat((token_latent, predicted), dim=-1)

    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int]:
        metadata = super().experiment_metadata()
        metadata["ce_probe_gradient"] = "attached_token_and_predicted_latent"
        return metadata


def main() -> None:
    if pope.PolarCausalSelfAttention.position_mode != "pope":
        raise ValueError("the attached-CE variant is defined only for PoPE")
    pope.FreshLeJEPASharedRMSV1PoPE = FreshLeJEPASharedRMSV1PoPEAttachedCE
    pope.POPE_ARCHITECTURE = ARCHITECTURE
    # Archive this controlled fork rather than the delegated base entry point.
    pope.__file__ = str(Path(__file__).resolve())
    pope.main()


if __name__ == "__main__":
    main()

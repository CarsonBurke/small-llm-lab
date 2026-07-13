"""Shared LeWM projector ablation with V1's full-width learned probes."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

import fresh_lejepa_train as v1
from fresh_lejepa_train_v2_shared_projector import FreshLeJEPAV2SharedProjector


ARCHITECTURE = "fresh_lejepa_shared_bn_predproj_v1_large_probes"


class FreshLeJEPASharedProjectorV1Probes(FreshLeJEPAV2SharedProjector):
    def __init__(self, *args, **kwargs):
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        vocab_size = kwargs.get("vocab_size", args[0] if args else None)
        super().__init__(*args, **kwargs)
        if model_dim is None or vocab_size is None:
            raise ValueError("model_dim and vocab_size are required")
        owner = self.blocks[-1]
        # Keep every non-probe parameter and the subsequent SIGReg random
        # stream identical to the compact-probe control.
        with torch.random.fork_rng(devices=[]):
            owner.policy_probe = v1.ResidualProbe(model_dim, vocab_size)
            owner.critic_probe = v1.ResidualProbe(model_dim, 1)
            torch.nn.init.zeros_(owner.policy_probe.output.weight)
            torch.nn.init.zeros_(owner.critic_probe.output.weight)

    def logits_from_features(self, features: Tensor) -> Tensor:
        raw = self.policy_probe(features)
        return self.logit_softcap * torch.tanh(raw / self.logit_softcap)


def main() -> None:
    v1.FreshLeJEPAGPT = FreshLeJEPASharedProjectorV1Probes
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    v1.main()


if __name__ == "__main__":
    main()

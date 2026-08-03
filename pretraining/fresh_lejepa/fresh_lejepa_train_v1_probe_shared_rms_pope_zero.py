"""Zero-phase PoPE pretraining variant for long-context generalization.

This is identical to ``fresh_lejepa_train_v1_probe_shared_rms_pope`` except
that every learned phase offset starts at zero, matching the PoPE language-model
and length-generalization configuration. It remains a from-scratch run.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import os
from pathlib import Path

import torch

os.environ.setdefault("FRESH_POSITION_MODE", "pope")

from pretraining.fresh_lejepa import fresh_lejepa_train_v1_probe_shared_rms_pope as pope


class ZeroPhasePolarCausalSelfAttention(pope.PolarCausalSelfAttention):
    """PoPE attention with paper-faithful zero phase-offset initialization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.delta_c is not None:
            with torch.no_grad():
                self.delta_c.zero_()


class FreshLeJEPASharedRMSV1PoPEZero(pope.FreshLeJEPASharedRMSV1PoPE):
    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int]:
        metadata = super().experiment_metadata()
        metadata["pope_theta_bias"] = "zero"
        return metadata


def main() -> None:
    pope.PolarCausalSelfAttention = ZeroPhasePolarCausalSelfAttention
    pope.FreshLeJEPASharedRMSV1PoPE = FreshLeJEPASharedRMSV1PoPEZero
    pope.POPE_ARCHITECTURE = "fresh_lejepa_shared_rms_v1_probes_pope_zero_scratch_2k"
    # The base entry point archives its module-level ``__file__`` as the
    # experiment source. Point it at this actual variant for provenance.
    pope.__file__ = str(Path(__file__).resolve())
    pope.main()


if __name__ == "__main__":
    main()

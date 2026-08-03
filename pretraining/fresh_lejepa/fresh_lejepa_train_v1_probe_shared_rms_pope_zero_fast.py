"""Zero-phase PoPE with cached angle tables and a bf16 elementwise path.

Identical geometry to ``fresh_lejepa_train_v1_probe_shared_rms_pope_zero``,
optimized per the SOTA pattern: the learned phase makes naive cos(theta-delta)
table caching impossible (delta changes every step), but the angle-addition
identity splits it — cos/sin(theta) tables are precomputed once in fp32 and
stored bf16 (safe: bounded in [-1, 1]), cos/sin(delta) is computed per step in
fp32 on the tiny (1, H, 1, D) parameter, and the combination plus softplus
magnitudes and rotations all run in bf16 instead of the reference port's fp32
over (B, H, T, D).

bf16 softplus shares fp32's exponent range (no under/overflow) but this still
changes numerics relative to the validated pope_zero runs, so this is a
separate ablation scored on both step time and BPB parity.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

os.environ.setdefault("FRESH_POSITION_MODE", "pope")

from pretraining.fresh_lejepa import fresh_lejepa_train_v1_probe_shared_rms_pope as pope
from pretraining.fresh_lejepa import fresh_lejepa_train_v1_probe_shared_rms_pope_zero as zero

# Generation (postraining) runs past the training context; size the tables to
# cover it.  Non-persistent buffers: rebuilt at init, never checkpointed.
TABLE_LENGTH = max(int(os.environ.get("TRAIN_SEQ_LEN", "1024")), 4096)


class FastZeroPhasePolarCausalSelfAttention(zero.ZeroPhasePolarCausalSelfAttention):
    """Zero-phase PoPE via angle addition over cached bf16 theta tables."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.polar_inv_freq is not None:
            theta = torch.outer(
                torch.arange(TABLE_LENGTH, dtype=torch.float32), self.polar_inv_freq
            )
            self.register_buffer(
                "theta_cos", theta.cos().to(torch.bfloat16), persistent=False
            )
            self.register_buffer(
                "theta_sin", theta.sin().to(torch.bfloat16), persistent=False
            )

    def _polar_components(
        self, q: Tensor, k: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.polar_inv_freq is None or self.delta_c is None:
            raise RuntimeError("polar components requested in RoPE mode")
        cos_theta = self.theta_cos[positions][None, None]
        sin_theta = self.theta_sin[positions][None, None]
        # The phase stays fp32 through clamp and trig (tiny: (1, H, 1, D)),
        # then joins the bf16 broadcast.
        delta = self.delta_c.clamp(-2 * math.pi, 0).float()
        cos_delta = delta.cos().to(cos_theta.dtype)
        sin_delta = delta.sin().to(cos_theta.dtype)
        q_cos = cos_theta * cos_delta + sin_theta * sin_delta
        q_sin = sin_theta * cos_delta - cos_theta * sin_delta
        q_mag = F.softplus(q.to(cos_theta.dtype))
        k_mag = F.softplus(k.to(cos_theta.dtype))
        gain = self.q_gain.to(cos_theta.dtype)[None, :, None, None]
        return (
            q_mag * q_cos * gain,
            q_mag * q_sin * gain,
            k_mag * cos_theta,
            k_mag * sin_theta,
        )


class FreshLeJEPASharedRMSV1PoPEZeroFast(zero.FreshLeJEPASharedRMSV1PoPEZero):
    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int]:
        metadata = super().experiment_metadata()
        metadata["pope_kernel"] = "angle_addition_bf16_tables"
        return metadata


def main() -> None:
    pope.PolarCausalSelfAttention = FastZeroPhasePolarCausalSelfAttention
    pope.FreshLeJEPASharedRMSV1PoPE = FreshLeJEPASharedRMSV1PoPEZeroFast
    pope.POPE_ARCHITECTURE = "fresh_lejepa_shared_rms_v1_probes_pope_zero_fast_scratch_2k"
    pope.__file__ = str(Path(__file__).resolve())
    pope.main()


if __name__ == "__main__":
    main()

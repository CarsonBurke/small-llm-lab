"""Zero-phase PoPE using FlashAttention-4 for full-sequence training.

FA4 accepts distinct Q/K and V widths, so PoPE uses its ideal QK=128/V=64
shape with native 8Q/4KV GQA. Incremental generation retains the existing
native-GQA SDPA path until the separately optimized decode kernel is selected.
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

try:
    from flash_attn.cute import flash_attn_func
except ImportError as error:  # pragma: no cover - exercised by the FA4 venv
    raise ImportError(
        "FA4 variant requires the isolated flash-attn-4 environment"
    ) from error

from pretraining.fresh_lejepa import fresh_lejepa_train_v1_probe_shared_rms_pope as pope
from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope_zero import (
    FreshLeJEPASharedRMSV1PoPEZero,
    ZeroPhasePolarCausalSelfAttention,
)


class ZeroPhaseFA4PolarCausalSelfAttention(ZeroPhasePolarCausalSelfAttention):
    """Ideal-arithmetic PoPE training attention on FA4/SM120."""

    def _complex_attention(
        self,
        q_real: Tensor,
        q_imag: Tensor,
        k_real: Tensor,
        k_imag: Tensor,
        value: Tensor,
        *,
        is_causal: bool,
    ) -> Tensor:
        if not is_causal or q_real.shape[-2] != k_real.shape[-2]:
            return super()._complex_attention(
                q_real, q_imag, k_real, k_imag, value, is_causal=is_causal
            )

        # FA4 layout is [batch, sequence, heads, dimension]. Q/K retain the
        # exact concatenated real-complex score while V stays at the true D.
        query = torch.cat((q_real, q_imag), dim=-1).to(value.dtype)
        key = torch.cat((k_real, k_imag), dim=-1).to(value.dtype)
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value_fa4 = value.transpose(1, 2).contiguous()
        output = flash_attn_func(
            query,
            key,
            value_fa4,
            softmax_scale=self.head_dim**-0.5,
            causal=True,
            pack_gqa=True,
        )
        if isinstance(output, tuple):
            output = output[0]
        return output.transpose(1, 2)


class FreshLeJEPASharedRMSV1PoPEZeroFA4(FreshLeJEPASharedRMSV1PoPEZero):
    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int]:
        metadata = super().experiment_metadata()
        metadata["pope_attention_kernel"] = "fa4_sm120_qk128_v64_native_gqa"
        metadata["pope_decode_kernel"] = "pytorch_flash_sdpa"
        return metadata


def main() -> None:
    pope.PolarCausalSelfAttention = ZeroPhaseFA4PolarCausalSelfAttention
    pope.FreshLeJEPASharedRMSV1PoPE = FreshLeJEPASharedRMSV1PoPEZeroFA4
    pope.POPE_ARCHITECTURE = (
        "fresh_lejepa_shared_rms_v1_probes_pope_zero_fa4_scratch_2k"
    )
    pope.__file__ = str(Path(__file__).resolve())
    pope.main()


if __name__ == "__main__":
    main()

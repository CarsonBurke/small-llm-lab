"""Zero-phase PoPE with the reference complex FlashAttention kernel.

The reference kernel operates on an equal number of query and KV heads. GQA is
exactly equivalent to repeating each KV head across its query group; autograd
then sums the repeated-head gradients back into the original KV projection.
The persistent generation cache remains native 8Q/4KV. Expansion occurs only
at the fused full-sequence training boundary and avoids the old path's doubled
attention feature width and padded values.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch import Tensor

import fresh_lejepa_train_v1_probe_shared_rms_pope as pope
from fresh_lejepa_train_v1_probe_shared_rms_pope_zero import (
    FreshLeJEPASharedRMSV1PoPEZero,
    ZeroPhasePolarCausalSelfAttention,
)


def _load_reference_complex_attention():
    source = Path(__file__).resolve().parent.parent / "pope" / "complex_flash_attention.py"
    spec = importlib.util.spec_from_file_location("pope_reference_complex_flash", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load PoPE complex FlashAttention from {source}")
    module = importlib.util.module_from_spec(spec)
    # Dynamo resolves the custom autograd function through its defining module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # This run is CUDA-only. The reference's runtime backend query is harmless
    # in eager mode but intentionally untraceable inside a fullgraph custom
    # autograd function; specialize it once at import time.
    module.is_hip = lambda: False
    return module.attention


# The reference launches and autotunes its own Triton kernels. Keep that launch
# opaque to Inductor; the surrounding model remains compiled with the existing
# graph-break-tolerant JEPA compile policy.
_reference_complex_attention = torch.compiler.disable(
    _load_reference_complex_attention()
)


class ZeroPhaseFlashPolarCausalSelfAttention(ZeroPhasePolarCausalSelfAttention):
    """Reference complex FlashAttention with exact grouped-query semantics."""

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
            # Incremental decoding has Q length 1 and a longer KV prefix. Keep
            # the already-correct native-GQA SDPA path for that shape.
            return super()._complex_attention(
                q_real, q_imag, k_real, k_imag, value, is_causal=is_causal
            )

        groups = self.num_heads // self.num_kv_heads
        if groups * self.num_kv_heads != self.num_heads:
            raise ValueError("query heads must be divisible by KV heads")

        dtype = value.dtype
        q_real = q_real.to(dtype).contiguous()
        q_imag = q_imag.to(dtype).contiguous()
        k_real = k_real.repeat_interleave(groups, dim=1).to(dtype).contiguous()
        k_imag = k_imag.repeat_interleave(groups, dim=1).to(dtype).contiguous()
        expanded_value = value.repeat_interleave(groups, dim=1).contiguous()

        # The reference kernel computes Qr·Kr - Qi·Ki, hence -Qi produces the
        # desired real conjugate product Qr·Kr + Qi·Ki used by our oracle.
        with torch.amp.autocast("cuda", dtype=dtype, enabled=True):
            return _reference_complex_attention(
                q_real,
                k_real,
                -q_imag,
                k_imag,
                expanded_value,
                True,
                self.head_dim**-0.5,
            )


class FreshLeJEPASharedRMSV1PoPEZeroFlash(FreshLeJEPASharedRMSV1PoPEZero):
    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int]:
        metadata = super().experiment_metadata()
        metadata["pope_attention_kernel"] = "reference_complex_flash_exact_gqa_expand"
        return metadata


def main() -> None:
    pope.PolarCausalSelfAttention = ZeroPhaseFlashPolarCausalSelfAttention
    pope.FreshLeJEPASharedRMSV1PoPE = FreshLeJEPASharedRMSV1PoPEZeroFlash
    pope.POPE_ARCHITECTURE = (
        "fresh_lejepa_shared_rms_v1_probes_pope_zero_flash_scratch_2k"
    )
    pope.__file__ = str(Path(__file__).resolve())
    pope.main()


if __name__ == "__main__":
    main()

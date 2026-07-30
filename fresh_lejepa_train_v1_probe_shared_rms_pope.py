"""From-scratch PoPE ablation for the shared-RMS V1 large-probe model.

This changes only attention Q/K positional geometry.  The PoPE branch follows
the reference implementation in ``../pope``: raw Q/K projections become
positive magnitudes through softplus, every feature has a geometric frequency,
and a learned per-query-head phase offset is initialized over the reference
two-pi interval.  With GQA, shifting a repeated key head by ``delta`` is exactly
equivalent in the real complex inner product to shifting its query by
``-delta``.  We use that identity to retain the native 8Q/4KV cache layout.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

import fresh_lejepa_train as v1
import fresh_lejepa_train_v4 as v4
import train_gpt as baseline
from fresh_lejepa_train_v1_probe_shared_rms_projector import (
    FreshLeJEPASharedRMSProjectorV1Probes,
)


POPE_ARCHITECTURE = "fresh_lejepa_shared_rms_v1_probes_pope_scratch_1k"
CONTROL_ARCHITECTURE = "fresh_lejepa_shared_rms_v1_probes_rope_scratch_1k_control"


class PolarCausalSelfAttention(baseline.CausalSelfAttention):
    """Baseline GQA with either the original RoPE or faithful PoPE geometry."""

    position_mode = os.environ.get("FRESH_POSITION_MODE", "pope")
    block_size = int(os.environ.get("TRAIN_SEQ_LEN", "1024"))

    def __init__(self, *args, **kwargs):
        rope_base = float(kwargs.get("rope_base", args[3] if len(args) > 3 else 10_000.0))
        super().__init__(*args, **kwargs)
        if self.position_mode not in {"rope", "pope"}:
            raise ValueError(f"unknown FRESH_POSITION_MODE={self.position_mode!r}")
        if self.position_mode == "pope":
            inv_freq = 1.0 / (
                rope_base
                ** (
                    torch.arange(self.head_dim, dtype=torch.float32)
                    / self.head_dim
                )
            )
            self.register_buffer("polar_inv_freq", inv_freq, persistent=False)
            # Do not perturb initialization of any matched baseline weights.
            with torch.random.fork_rng(devices=[]):
                upper = torch.zeros_like(inv_freq)
                lower = (
                    -2
                    * math.pi
                    / torch.maximum(inv_freq, torch.tensor(1.0 / self.block_size))
                    * inv_freq
                )
                delta = torch.rand(1, self.num_heads, 1, self.head_dim)
                delta = delta * (upper - lower)[None, None, None] + lower[
                    None, None, None
                ]
            self.delta_c = nn.Parameter(delta)
        else:
            self.register_buffer("polar_inv_freq", None, persistent=False)
            self.register_parameter("delta_c", None)

    def _polar_components(
        self, q: Tensor, k: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return real/imaginary Q/K, retaining four unexpanded KV heads."""
        if self.polar_inv_freq is None or self.delta_c is None:
            raise RuntimeError("polar components requested in RoPE mode")
        theta = torch.outer(
            positions.to(dtype=self.polar_inv_freq.dtype),
            self.polar_inv_freq.to(device=q.device),
        )[None, None]
        delta = self.delta_c.clamp(-2 * math.pi, 0).to(dtype=theta.dtype)

        # Reference: K angle is theta + delta for every Q head.  In GQA we
        # instead use Q angle theta - delta, which gives the identical score
        # cos((theta_k + delta) - theta_q) without duplicating cached K/V.
        q_theta = theta - delta
        q_mag = F.softplus(q.float())
        k_mag = F.softplus(k.float())
        q_real = q_mag * q_theta.cos()
        q_imag = q_mag * q_theta.sin()
        k_real = k_mag * theta.cos()
        k_imag = k_mag * theta.sin()
        gain = self.q_gain.float()[None, :, None, None]
        return q_real * gain, q_imag * gain, k_real, k_imag

    def _complex_attention(
        self,
        q_real: Tensor,
        q_imag: Tensor,
        k_real: Tensor,
        k_imag: Tensor,
        value: Tensor,
        *,
        is_causal: bool,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        # Concatenation turns Re(conj(Q)K) into one ordinary dot product.  The
        # explicit scale is sqrt(complex feature width), not sqrt(2D).  CUDA
        # Flash SDPA requires Q/K/V to have equal feature widths, so append a
        # zero-valued half to V and discard the corresponding zero output.
        query = torch.cat((q_real, q_imag), dim=-1).to(value.dtype)
        key = torch.cat((k_real, k_imag), dim=-1).to(value.dtype)
        padded_value = torch.cat((value, torch.zeros_like(value)), dim=-1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            padded_value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            enable_gqa=self.num_kv_heads != self.num_heads,
            scale=self.head_dim**-0.5,
        )
        return output[..., : self.head_dim]

    def forward(self, x: Tensor) -> Tensor:
        if self.position_mode == "rope":
            return super().forward(x)
        batch, length, dim = x.shape
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, value = self.c_qkv(x).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(
            batch, length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        positions = torch.arange(length, device=x.device)
        q_real, q_imag, k_real, k_imag = self._polar_components(q, k, positions)
        y = self._complex_attention(
            q_real, q_imag, k_real, k_imag, value, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.proj(y)

    def forward_step(
        self,
        x: Tensor,
        cache: tuple[Tensor, Tensor, Tensor],
        position: int | Tensor,
        key_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        batch, _, dim = x.shape
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, value = self.c_qkv(x).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        index = torch.as_tensor(position, device=x.device).reshape(1)
        q_real, q_imag, k_real, k_imag = self._polar_components(q, k, index)
        cache[0].index_copy_(2, index, k_real.to(cache[0].dtype))
        cache[1].index_copy_(2, index, k_imag.to(cache[1].dtype))
        cache[2].index_copy_(2, index, value.to(cache[2].dtype))
        if key_mask is not None and key_mask.dim() == 2:
            # Per-row (batch, keys) validity for left-padded batched rollouts:
            # narrow position-sliced shapes, each row's padded prefix slots
            # masked out of attention.
            # The mask already has exactly the live prefix width. Deriving
            # the narrow length from its shape keeps this dimension symbolic
            # under dynamic torch.compile; deriving it from the position
            # tensor would make it a data-dependent output shape.
            prefix_length = key_mask.shape[-1]
            prefix_k_real = torch.narrow(cache[0], 2, 0, prefix_length)
            prefix_k_imag = torch.narrow(cache[1], 2, 0, prefix_length)
            prefix_value = torch.narrow(cache[2], 2, 0, prefix_length)
            attn_mask = torch.narrow(
                key_mask, 1, 0, prefix_length
            )[:, None, None, :]
        elif key_mask is not None:
            # Static full-cache path (see FreshLeJEPAGPT._attention_step):
            # constant shapes for CUDA-graph capture; masked slots must be
            # finite, so static caches are zero-filled at allocation.
            if not torch.is_tensor(position):
                raise ValueError(
                    "key_mask stepping requires a 0-dim tensor position"
                )
            prefix_k_real, prefix_k_imag, prefix_value = cache
            attn_mask = key_mask[None, None, None, :]
        else:
            prefix_length = index[0] + 1
            prefix_k_real = torch.narrow(cache[0], 2, 0, prefix_length)
            prefix_k_imag = torch.narrow(cache[1], 2, 0, prefix_length)
            prefix_value = torch.narrow(cache[2], 2, 0, prefix_length)
            attn_mask = None
        y = self._complex_attention(
            q_real,
            q_imag,
            prefix_k_real,
            prefix_k_imag,
            prefix_value,
            is_causal=False,
            attn_mask=attn_mask,
        )
        y = y.transpose(1, 2).contiguous().view(batch, 1, dim)
        return self.proj(y), cache


class FreshLeJEPASharedRMSV1PoPE(FreshLeJEPASharedRMSProjectorV1Probes):
    """V1 shared-RMS model with paired-B128 SIGReg and selectable Q/K geometry."""

    defer_sigreg = True
    latent_loss_weight = v1.FreshHyperparameters.latent_loss_weight
    sigreg_loss_weight = v1.FreshHyperparameters.sigreg_weight

    def deferred_sigreg_loss(
        self,
        first_input_ids: Tensor,
        first_target_ids: Tensor,
        second_input_ids: Tensor,
        second_target_ids: Tensor,
    ) -> Tensor:
        first = torch.cat((first_input_ids, first_target_ids[:, -1:]), dim=1)
        second = torch.cat((second_input_ids, second_target_ids[:, -1:]), dim=1)
        trajectory = self.embed_tokens(torch.cat((first, second), dim=0))
        return self.sigreg(trajectory)

    def _attention_step(
        self,
        attention: baseline.CausalSelfAttention,
        x: Tensor,
        cache: tuple[Tensor, ...],
        position: int | Tensor,
        key_mask: Tensor | None = None,
        block_mask: "BlockMask | None" = None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        if isinstance(attention, PolarCausalSelfAttention) and (
            attention.position_mode == "pope"
        ):
            if block_mask is not None:
                # PoPE scores are a complex inner product over a k_real/k_imag
                # cache pair, not one QK product flex can score.
                raise NotImplementedError(
                    "flex decoding is not implemented for PoPE attention"
                )
            return attention.forward_step(x, cache, position, key_mask)
        return super()._attention_step(
            attention, x, cache, position, key_mask, block_mask
        )

    def _attention_prefill(
        self,
        attention: baseline.CausalSelfAttention,
        x: Tensor,
        cache: tuple[Tensor, ...],
        attention_mask: Tensor | None,
    ) -> Tensor:
        if not isinstance(attention, PolarCausalSelfAttention) or (
            attention.position_mode != "pope"
        ):
            return super()._attention_prefill(
                attention, x, cache, attention_mask
            )
        batch, length, dim = x.shape
        q_dim = attention.num_heads * attention.head_dim
        kv_dim = attention.num_kv_heads * attention.head_dim
        q, k, value = attention.c_qkv(x).split(
            [q_dim, kv_dim, kv_dim], dim=-1
        )
        q = q.view(
            batch, length, attention.num_heads, attention.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch, length, attention.num_kv_heads, attention.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch, length, attention.num_kv_heads, attention.head_dim
        ).transpose(1, 2)
        positions = torch.arange(length, device=x.device)
        q_real, q_imag, k_real, k_imag = attention._polar_components(
            q, k, positions
        )
        cache[0][:, :, :length].copy_(k_real.to(cache[0].dtype))
        cache[1][:, :, :length].copy_(k_imag.to(cache[1].dtype))
        cache[2][:, :, :length].copy_(value.to(cache[2].dtype))
        output = attention._complex_attention(
            q_real,
            q_imag,
            k_real,
            k_imag,
            value,
            is_causal=attention_mask is None,
            attn_mask=attention_mask,
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, dim)
        return attention.proj(output)

    def make_generation_cache(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> list[tuple[Tensor, ...]]:
        if PolarCausalSelfAttention.position_mode == "rope":
            return super().make_generation_cache(
                batch_size, max_length, device, dtype=dtype
            )
        cache_dtype = self.tok_emb.weight.dtype if dtype is None else dtype
        caches = []
        for block in self.blocks:
            attention = block.attn
            shape = (
                batch_size,
                attention.num_kv_heads,
                max_length,
                attention.head_dim,
            )
            caches.append(
                tuple(
                    torch.empty(
                        shape, device=device, dtype=cache_dtype
                    )
                    for _ in range(3)
                )
            )
        return caches

    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int]:
        return {
            "position_encoding": PolarCausalSelfAttention.position_mode,
            "training_context": PolarCausalSelfAttention.block_size,
            "pope_theta_bias": "two_pi",
            "pope_gqa_phase": "equivalent_per_query_phase_shift",
            "pope_qk_normalization": "raw_projection_softplus_no_rms",
            "pope_attention_scale": "head_dim^-0.5",
            "q_gain": "retained_after_polar_projection",
            "initial_checkpoint": "none_from_scratch",
        }


def main() -> None:
    if os.environ.get("FRESH_POPE_INIT_CHECKPOINT"):
        raise ValueError("PoPE is a from-scratch ablation; checkpoint loading is forbidden")
    original_attention = baseline.CausalSelfAttention
    original_main = v4._install_configurable_accumulation(default_steps=8)
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    original_int8_patterns = baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    baseline.CausalSelfAttention = PolarCausalSelfAttention
    baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns + ("delta_c",)
    baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns + (
        "delta_c",
    )
    FreshLeJEPASharedRMSV1PoPE.return_loss_components = True
    v1.FreshLeJEPAGPT = FreshLeJEPASharedRMSV1PoPE
    v1.EXPERIMENT_ARCHITECTURE = (
        POPE_ARCHITECTURE
        if PolarCausalSelfAttention.position_mode == "pope"
        else CONTROL_ARCHITECTURE
    )
    v1.EXPERIMENT_SOURCE = Path(__file__)
    os.environ.setdefault("GRAD_ACCUM_STEPS", "8")
    try:
        v1.main()
    finally:
        baseline.CausalSelfAttention = original_attention
        baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns
        baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns
        baseline.main = original_main


if __name__ == "__main__":
    main()

"""Baseline + zero-phase PoPE with cached angle tables and bf16 elementwise.

Same PoPE geometry as ``ablations/baseline/pope_zero_train_gpt.py`` but engineered the way
production RoPE implementations handle positional math in low-precision runs:
angles are computed once in fp32 at construction and cached as bounded
cos/sin tables (safe to hold in bf16 at any position — only unbounded angles
are precision-fragile), and the per-step elementwise work (softplus
magnitudes, real/imag products) runs in the compute dtype instead of fp32.

The learned phase offset still trains: ``cos/sin(delta_c)`` are computed in
fp32 from the fp32 parameter each step (tiny: (1, H, 1, D)) and folded into
the query tables with the angle-addition identities
``cos(theta - delta) = cos(theta)cos(delta) + sin(theta)sin(delta)`` and
``sin(theta - delta) = sin(theta)cos(delta) - cos(theta)sin(delta)``,
so no large angle is ever represented in low precision.

Numerics deviate from the validated fp32-elementwise PoPE runs (bf16
magnitudes/products, ~0.4% relative), which is exactly what this ablation
measures: BPB parity vs ``baseline_pope_zero_2k`` at a lower step cost.
Tables are sized to TRAIN_SEQ_LEN; longer inference contexts would need
regeneration.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import math

import torch
import torch.nn.functional as F
from torch import Tensor

import train_gpt as baseline
from ablations.baseline.nextlat_pope_train_gpt import PolarCausalSelfAttention


class FastPolarCausalSelfAttention(PolarCausalSelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        positions = torch.arange(self.block_size, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.rope_base
            ** (torch.arange(self.head_dim, dtype=torch.float32) / self.head_dim)
        )
        theta = torch.outer(positions, inv_freq)[None, None]
        # Non-persistent buffers: baseline.main's .bfloat16() rounds the
        # bounded values (~0.4% relative, position-independent) by design.
        self.register_buffer("cos_table", theta.cos(), persistent=False)
        self.register_buffer("sin_table", theta.sin(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, dim = x.shape
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, value = self.c_qkv(x).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos_t = self.cos_table[:, :, :length]
        sin_t = self.sin_table[:, :, :length]
        delta = self.delta_c.clamp(-2 * math.pi, 0)
        cos_d = delta.cos()
        sin_d = delta.sin()
        # Per-(head, position) query factors, fp32 combine then one cast.
        cos_q = cos_t.float() * cos_d + sin_t.float() * sin_d
        sin_q = sin_t.float() * cos_d - cos_t.float() * sin_d
        if self.qk_gain_mode == "on":
            gain = self.q_gain[None, :, None, None]
            cos_q = cos_q * gain
            sin_q = sin_q * gain
        cos_q = cos_q.to(q.dtype)
        sin_q = sin_q.to(q.dtype)

        q_mag = F.softplus(q)
        k_mag = F.softplus(k)
        q_real = q_mag * cos_q
        q_imag = q_mag * sin_q
        k_real = k_mag * cos_t.to(k.dtype)
        k_imag = k_mag * sin_t.to(k.dtype)
        y = self._complex_attention(q_real, q_imag, k_real, k_imag, value)
        y = y.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.proj(y)


def main() -> None:
    original_attention = baseline.CausalSelfAttention
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    original_int8_patterns = baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    baseline.CausalSelfAttention = FastPolarCausalSelfAttention
    baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns + ("delta_c",)
    baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns + ("delta_c",)
    try:
        baseline.main()
    finally:
        baseline.CausalSelfAttention = original_attention
        baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns
        baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns


if __name__ == "__main__":
    main()

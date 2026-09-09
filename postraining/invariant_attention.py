"""Opaque native FA4 launch inside an otherwise full-graph invariant decoder."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast
import torch
from torch import Tensor, nn


INVARIANT_ATTENTION = "parameter_golf_invariant_suffix"


@torch.library.custom_op(
    "parameter_golf::invariant_fa4", mutates_args=(), device_types="cuda"
)
def invariant_fa4(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sequence_lengths: Tensor,
    max_query: int,
    max_key: int,
    scale: float,
) -> Tensor:
    flash_attn_varlen_func = cast(
        Any, import_module("flash_attn.cute.interface")
    ).flash_attn_varlen_func
    output, _ = flash_attn_varlen_func(
        query,
        key,
        value,
        max_seqlen_q=max_query,
        max_seqlen_k=max_key,
        seqused_k=sequence_lengths,
        softmax_scale=scale,
        causal=True,
        pack_gqa=True,
    )
    # Canonical output layout is part of the custom-op/fake contract.
    return output.contiguous()


@invariant_fa4.register_fake
def _fake_invariant_fa4(query, key, value, sequence_lengths, max_query, max_key, scale):
    return torch.empty(
        (*query.shape[:-1], value.shape[-1]), dtype=query.dtype, device=query.device
    )


def invariant_suffix_attention(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[Tensor, None]:
    attention = cast(Any, module)
    scale = query.size(-1) ** -0.5 if scaling is None else scaling
    output = invariant_fa4(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attention._rollout_sequence_lengths,
        query.size(2),
        attention._rollout_max_cache_len,
        scale,
    )
    return output, None

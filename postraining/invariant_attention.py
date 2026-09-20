"""Opaque native FA4 launch inside an otherwise full-graph invariant decoder."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Optional, cast
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
    optimized_decode: bool,
    split_offsets: Optional[Tensor],
    split_live: Optional[Tensor],
) -> Tensor:
    """Ragged-suffix FA4 for one decode step.

    ``split_offsets`` and ``split_live`` carry a split-KV partition plan the
    caller computed once for this step (see ``split_kv_plan``); pass ``None``
    to derive it per launch. Both give identical attention output.
    """
    interface = cast(Any, import_module("flash_attn.cute.interface"))
    # Keep Uno's serial/block numerical target unchanged. This tile was
    # qualified for ordinary MiniCPM decode, not other FA4 architectures.
    tune_tile = (
        optimized_decode
        and query.shape[1:] == (1, 16, 128)
        and key.shape[-2:] == value.shape[-2:] == (2, 128)
        and query.dtype == torch.bfloat16
        and torch.cuda.get_device_capability(query.device)[0] == 12
    )
    if (
        tune_tile
        and torch.cuda.get_device_capability(query.device) == (12, 0)
        and key.is_contiguous()
        and value.is_contiguous()
    ):
        from postraining.split_kv_attention import split_kv_attention

        return split_kv_attention(
            query, key, value, sequence_lengths, scale, split_offsets, split_live
        )
    forward = (
        interface._flash_attn_fwd if tune_tile else interface.flash_attn_varlen_func
    )
    options = {"tile_mn": (64, 64)} if tune_tile else {}
    output = forward(
        query,
        key,
        value,
        max_seqlen_q=max_query,
        max_seqlen_k=max_key,
        seqused_k=sequence_lengths,
        softmax_scale=scale,
        causal=True,
        pack_gqa=True,
        **options,
    )[0]
    # Canonical output layout is part of the custom-op/fake contract.
    return output.contiguous()


@invariant_fa4.register_fake
def _fake_invariant_fa4(
    query,
    key,
    value,
    sequence_lengths,
    max_query,
    max_key,
    scale,
    optimized_decode,
    split_offsets,
    split_live,
):
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
        attention._rollout_optimized_decode,
        attention._rollout_split_kv_offsets,
        attention._rollout_split_kv_live,
    )
    return output, None

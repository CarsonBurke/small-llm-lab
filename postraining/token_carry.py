"""Shared observed-token carry inputs with independent trainable projections."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TokenCarryCombiner(nn.Module):
    """Scale each residual channel while preserving the token embedding path."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.token_delta = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.float32)
        self.carry = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.float32)
        self.scale = nn.Parameter(torch.full((hidden_size,), 0.01, dtype=torch.float32))
        nn.init.zeros_(self.token_delta.weight)
        nn.init.zeros_(self.carry.weight)

    def forward(self, token_embedding: Tensor, previous_hidden: Tensor) -> Tensor:
        dtype = token_embedding.dtype
        residual = F.linear(token_embedding, self.token_delta.weight.to(dtype)) + F.linear(
            previous_hidden.detach().to(dtype), self.carry.weight.to(dtype)
        )
        # Do not quantize the learned scale to BF16 before multiplication.
        return token_embedding + (residual.float() * self.scale).to(dtype)


def load_token_carry_state_dict(side: Any, payload: dict[str, Any]) -> None:
    enabled = payload.get("token_carry", False)
    if type(enabled) is not bool or enabled != side.token_carry:
        raise ValueError("checkpoint token-carry mode differs from the model")
    slot_memory = getattr(side, "slot_memory", None)
    if (payload.get("slot_memory") is None) != (slot_memory is None):
        raise ValueError("checkpoint slot-memory mode differs from the model")
    if enabled:
        if payload.get("latent_thinking", False):
            raise ValueError("token carry cannot be combined with latent thinking")
        if "token_combiner" not in payload:
            raise ValueError("checkpoint is missing token_combiner")
        if slot_memory is not None:
            from postraining.slot_memory import SlotMemoryConfig

            if SlotMemoryConfig.from_payload(payload["slot_memory"]) != slot_memory:
                raise ValueError("checkpoint slot-memory geometry differs from the model")
            if hasattr(side, "slot_head") != ("slot_head" in payload):
                raise ValueError("checkpoint slot head presence differs from the model side")
            if "slot_head" in payload:
                side.slot_head.load_state_dict(payload["slot_head"], strict=True)
        side.token_combiner.load_state_dict(payload["token_combiner"], strict=True)
    elif "token_combiner" in payload:
        raise ValueError("native checkpoint contains token-carry state")


def token_carry_replay_hidden(side: Any, batch: Any) -> Tensor:
    """Run one packed forward on the fixed behavior-time token/hidden stream."""
    if not side.token_carry:
        raise ValueError("stored-carry replay requires token carry")
    if batch.action_kinds is not None or batch.latent_vectors is not None:
        raise ValueError("token-carry replay cannot consume Gaussian latent actions")
    if getattr(batch, "slot_alive_table", None) is not None:
        raise ValueError("plain token-carry replay cannot consume slot-memory records")
    carries = batch.carry_hiddens
    positions = batch.carry_input_positions
    if carries is None or positions is None:
        raise ValueError("token-carry replay requires stored carry hiddens and positions")
    if (
        batch.input_ids.ndim != 2 or batch.input_ids.shape[0] != 1
        or carries.ndim != 2 or carries.dtype != torch.bfloat16
        or carries.shape[1] != side.hidden_size
        or positions.ndim != 1 or positions.dtype != torch.long
        or positions.numel() != carries.shape[0]
        or carries.device != batch.input_ids.device
        or positions.device != batch.input_ids.device
    ):
        raise ValueError("stored carry hiddens and positions must match the packed input")
    # Inference-mode rollout storage must never leak into autograd's saved inputs.
    differentiable = torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    with torch.inference_mode(False), torch.set_grad_enabled(differentiable):
        carries = carries.detach()
        if carries.is_inference():
            carries = carries.clone()
        embeddings = side.token_embeddings(batch.input_ids)
        if positions.numel():
            mixed = side.token_combiner(embeddings[0, positions], carries)
            embeddings = embeddings.index_copy(1, positions, mixed.unsqueeze(0))
        return side.replay_hidden(
            None, batch.attention_mask,
            inputs_embeds=embeddings,
            position_ids=batch.position_ids,
            cu_seqlens=batch.cu_seqlens,
            sequence_boundaries=batch.sequence_boundaries,
            max_sequence_length=batch.max_sequence_length,
        )

"""Vocabulary readouts.

Scoring the actions of a rollout is where families diverge most cheaply and
most dangerously. MiniCPM5 scores against a frozen 130,560-row ``lm_head``,
so the chunked path below never materializes an ``[actions, vocab]`` tensor
in either direction. The nano backbones render from
``cat(token_latent, belief)`` through a softcapped, optionally tied codebook,
and their readout parameters are trainable.

Both are expressed as :class:`~postraining.vapo.model.protocols.Readout`, so
replay and the policy objective stay family-agnostic.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _ChunkedFrozenHeadLogProbs(torch.autograd.Function):
    """Exact selected-token log-probabilities without retaining full logits."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden: Tensor,
        targets: Tensor,
        weight: Tensor,
        chunk_tokens: int,
    ) -> Tensor:
        if hidden.ndim != 2 or weight.ndim != 2:
            raise ValueError("hidden and output weight must be matrices")
        if targets.shape != hidden.shape[:1] or targets.dtype != torch.long:
            raise ValueError("targets must be int64 with one id per hidden row")
        if hidden.shape[1] != weight.shape[1]:
            raise ValueError("hidden width and output-head width differ")
        if weight.requires_grad:
            raise ValueError("chunked output head must be frozen")
        if chunk_tokens < 1:
            raise ValueError("logit chunk size must be positive")
        if targets.device.type == "cpu" and targets.numel() and (
            int(targets.min()) < 0 or int(targets.max()) >= weight.shape[0]
        ):
            raise ValueError("target token lies outside the vocabulary")

        ctx.save_for_backward(hidden, targets, weight)
        ctx.chunk_tokens = chunk_tokens
        result = torch.empty(hidden.shape[0], device=hidden.device, dtype=torch.float32)
        for start in range(0, hidden.shape[0], chunk_tokens):
            stop = min(start + chunk_tokens, hidden.shape[0])
            logits = F.linear(hidden[start:stop], weight).float()
            result[start:stop] = logits.gather(
                1, targets[start:stop, None]
            ).squeeze(1) - logits.logsumexp(dim=1)
        return result


    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor):
        if len(grad_outputs) != 1:
            raise RuntimeError("chunked log-probability backward expects one gradient")
        grad_output = grad_outputs[0]
        hidden, targets, weight = ctx.saved_tensors
        grad_hidden = torch.empty_like(hidden)
        rows = torch.arange(
            min(ctx.chunk_tokens, hidden.shape[0]), device=hidden.device
        )
        for start in range(0, hidden.shape[0], ctx.chunk_tokens):
            stop = min(start + ctx.chunk_tokens, hidden.shape[0])
            logits = F.linear(hidden[start:stop], weight).float()
            probabilities = -logits.softmax(dim=1)
            del logits
            probabilities[rows[: stop - start], targets[start:stop]] += 1.0
            probabilities.mul_(grad_output[start:stop, None].float())
            grad_hidden[start:stop] = F.linear(
                probabilities.to(weight.dtype), weight.transpose(0, 1)
            ).to(hidden.dtype)
        return grad_hidden, None, None, None


def chunked_frozen_head_logprobs(
    hidden: Tensor,
    targets: Tensor,
    weight: Tensor,
    *,
    chunk_tokens: int,
) -> Tensor:
    return _ChunkedFrozenHeadLogProbs.apply(hidden, targets, weight, chunk_tokens)


class FrozenLinearReadout:
    """Frozen ``[vocab, hidden]`` output projection scored chunkwise.

    The weight is held by reference, so an in-place adapter merge or a device
    move on the owning module stays visible here.
    """

    def __init__(self, head: nn.Linear) -> None:
        if not isinstance(head, nn.Linear):
            raise TypeError("a frozen linear readout requires nn.Linear")
        if head.bias is not None:
            raise ValueError("chunked frozen scoring assumes a bias-free head")
        self._head = head

    @property
    def frozen(self) -> bool:
        return True

    @property
    def weight(self) -> Tensor:
        weight = self._head.weight
        if weight.requires_grad:
            raise RuntimeError(
                f"the {weight.shape[0]:,}-row output head must remain frozen"
            )
        return weight

    def logits(self, features: Tensor) -> Tensor:
        return self._head(features)

    def target_logprobs(
        self, features: Tensor, targets: Tensor, *, chunk_tokens: int
    ) -> Tensor:
        return chunked_frozen_head_logprobs(
            features, targets, self.weight, chunk_tokens=chunk_tokens
        )


class TrunkRenderReadout:
    """Readout delegated to a nano backbone's own trainable renderer.

    The nano renderer consumes ``cat(token_latent, belief)`` and applies a
    softcap, so its log-probabilities cannot be reproduced by a plain matmul
    against an embedding matrix. Scoring therefore goes through the backbone
    and, because the head is trainable, keeps its dense autograd graph. The
    chunk argument still bounds peak activation memory.
    """

    def __init__(self, backbone: Any) -> None:
        self._backbone = backbone

    @property
    def frozen(self) -> bool:
        return False

    def logits(self, features: Tensor) -> Tensor:
        return self._backbone.logits_from_features(features)

    def target_logprobs(
        self, features: Tensor, targets: Tensor, *, chunk_tokens: int
    ) -> Tensor:
        if features.ndim != 2 or targets.ndim != 1:
            raise ValueError("readout scoring expects packed [actions, features]")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("one target is required per scored action")
        if chunk_tokens < 1:
            raise ValueError("chunk size must be positive")
        parts = [
            torch.log_softmax(self.logits(features[start : start + chunk_tokens]), dim=-1)
            .gather(1, targets[start : start + chunk_tokens, None].long())
            .squeeze(1)
            for start in range(0, features.shape[0], chunk_tokens)
        ]
        return torch.cat(parts, dim=0)


__all__ = [
    "FrozenLinearReadout",
    "TrunkRenderReadout",
    "chunked_frozen_head_logprobs",
]

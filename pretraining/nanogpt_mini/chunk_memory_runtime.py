"""Whole-sequence compiled loss without activation checkpointing."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class CompiledFullLoss:
    """Shared benchmark/trainer loss, compatible with CUDA graph executors."""

    def __init__(self, model, segment_size=256):
        if segment_size < 1:
            raise ValueError("segment_size must be positive")
        self.model = model
        self.segment_size = segment_size
        self._compiled = torch.compile(self._loss, fullgraph=True, dynamic=False)

    def _loss(self, inputs, targets, diagnostics=False):
        hidden, memory = self.model.forward_hidden(inputs, segment_size=self.segment_size)
        loss = F.cross_entropy(self.model.logits(hidden).flatten(0, 1),
                               targets.flatten(), reduction="sum")
        if diagnostics:
            memory = memory.float()
            return loss, memory.square().mean(), memory.var(dim=1, correction=0).mean()
        return loss

    def __call__(self, inputs, targets, *, diagnostics=False):
        if inputs.device.type != "cuda" or targets.device != inputs.device:
            raise ValueError("Compiled full loss requires CUDA inputs and targets")
        if inputs.ndim != 2 or inputs.shape != targets.shape:
            raise ValueError("inputs and targets must share [batch,time] shape")
        return self._compiled(inputs, targets, diagnostics)

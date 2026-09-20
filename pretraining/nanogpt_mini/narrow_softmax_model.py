"""Narrow causal-softmax control with the unchanged nanoGPT-mini residual width.

Default mixing has one 128-wide Q/K/V head within six 512-wide blocks. It
preserves mini's attention mathematics, RoPE, Q/K RMS normalization, MLPs,
initialization and output softcap. It has no recurrent state or memory bank.
This is a capacity/architecture control, not an equivalent scalar-delta model.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from pretraining.nanogpt_mini.nanogpt_mini_model import (
    Block, CausalSelfAttention, GPT, Linear, MLP, RMSNorm, Rotary,
)


class NarrowCausalSelfAttention(CausalSelfAttention):
    """Inherit mini's exact attention computation; narrow only its projections."""

    def __init__(self, model_dim: int, mixer_dim: int, head_dim: int):
        nn.Module.__init__(self)
        self.num_heads = mixer_dim // head_dim
        self.head_dim = head_dim
        self.q = Linear(model_dim, mixer_dim)
        self.k = Linear(model_dim, mixer_dim)
        self.v = Linear(model_dim, mixer_dim)
        self.proj = Linear(mixer_dim, model_dim)
        self.rotary = Rotary(head_dim)


class NarrowSoftmaxBlock(Block):
    def __init__(self, model_dim: int, mixer_dim: int, head_dim: int):
        nn.Module.__init__(self)
        self.attn = NarrowCausalSelfAttention(model_dim, mixer_dim, head_dim)
        self.mlp = MLP(model_dim)
        self.norm1, self.norm2 = RMSNorm(model_dim), RMSNorm(model_dim)


class NarrowSoftmaxGPT(GPT):
    """Mini-compatible sum-loss forward plus shared hidden/logit runtime API."""

    def __init__(self, vocab_size: int = 1024, num_layers: int = 6,
                 model_dim: int = 512, mixer_dim: int = 128, head_dim: int = 128):
        nn.Module.__init__(self)
        if min(vocab_size, num_layers, model_dim, mixer_dim, head_dim) < 1:
            raise ValueError("All dimensions must be positive")
        if mixer_dim % head_dim or head_dim % 4:
            raise ValueError("head_dim must divide mixer_dim and be divisible by four for mini RoPE")
        self.config = dict(vocab_size=vocab_size, num_layers=num_layers, model_dim=model_dim,
                           mixer_dim=mixer_dim, head_dim=head_dim)
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(NarrowSoftmaxBlock(model_dim, mixer_dim, head_dim)
                                    for _ in range(num_layers))
        self.proj = Linear(model_dim, vocab_size)
        self.norm1, self.norm2 = RMSNorm(model_dim), RMSNorm(model_dim)
        self._initialize()

    @torch.no_grad()
    def _initialize(self):
        for name, parameter in self.named_parameters():
            if name == "embed.weight":
                parameter.normal_()
            elif name.endswith("gains"):
                parameter.fill_(1)
            elif name.endswith("weight"):
                if "proj" in name:
                    parameter.zero_()
                else:
                    parameter.normal_(std=math.sqrt(.33 / parameter.shape[-1]))
            else:
                parameter.zero_()

    def forward_hidden(self, inputs: Tensor, segment_size: int | None = None):
        """Full causal attention within each row; segment_size never cuts context."""
        if inputs.ndim != 2 or inputs.shape[1] == 0:
            raise ValueError("Expected nonempty [batch,time] token inputs")
        if inputs.device.type != "cuda":
            raise ValueError("Narrow softmax control requires CUDA; no CPU fallback")
        if segment_size is not None and segment_size < 1:
            raise ValueError("segment_size must be positive")
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        return self.norm2(x), None

    def logits(self, hidden: Tensor):
        logits = self.proj(hidden).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()

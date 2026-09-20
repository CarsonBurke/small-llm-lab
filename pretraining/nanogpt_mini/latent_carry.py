"""Exact final-state feedback through a shared, learned delta-rule memory.

All blocks read the same pre-token state. The normalized final hidden state
then supplies one content-dependent write, visible starting at the next token.
There is no token attention, delayed chunk write, or layer-local state bank.
Dense computations use BF16 autocast; the recurrent accumulator stays FP32.
Checkpoint segments recompute activations without truncating temporal gradients.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pretraining.nanogpt_mini.nanogpt_mini_model import Linear, MLP, RMSNorm


def unit_key(x: Tensor) -> Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-6)


def delta_write(memory: Tensor, key: Tensor, value: Tensor,
                retention: Tensor, strength: Tensor) -> Tensor:
    """FP32 [B,H,K,V] state; erase the predicted association and write correction."""
    decayed = memory * retention[..., None, None]
    prediction = (decayed * key[..., None]).sum(-2)
    correction = strength[..., None] * (value - prediction)
    return decayed + key[..., None] * correction[..., None, :]


class DeltaReadBlock(nn.Module):
    def __init__(self, dim: int, heads: int, key_dim: int, value_dim: int):
        super().__init__()
        self.heads, self.key_dim, self.value_dim = heads, key_dim, value_dim
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q = Linear(dim, heads * key_dim)
        self.proj = Linear(heads * value_dim, dim)
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        query = unit_key(self.q(self.norm1(x)).view(x.shape[0], self.heads, self.key_dim))
        # Unit-norm queries/keys make a fully written matching key recover its
        # value; no additional softmax or head-width-dependent read scaling.
        read = (memory * query[..., None]).sum(-2).flatten(1).to(x.dtype)
        x = x + self.proj(read)
        return x + self.mlp(self.norm2(x))


class FinalStateDeltaCarry(nn.Module):
    def __init__(self, vocab_size: int = 1024, num_layers: int = 6,
                 model_dim: int = 512, heads: int = 4, key_dim: int = 32,
                 value_dim: int = 128, initial_retention=None):
        super().__init__()
        if min(vocab_size, num_layers, model_dim, heads, key_dim, value_dim) < 1:
            raise ValueError("All dimensions must be positive")
        if initial_retention is None:
            initial_retention = ([.5, .9, .98, .999] if heads == 4 else
                                 [math.exp(-math.exp(math.log(-math.log(.5)) + i / max(heads - 1, 1)
                                  * (math.log(-math.log(.999)) - math.log(-math.log(.5)))))
                                  for i in range(heads)])
        initial_retention = tuple(float(x) for x in initial_retention)
        if len(initial_retention) != heads or not all(0 < x < 1 for x in initial_retention):
            raise ValueError("initial_retention must provide one probability per head")
        self.config = dict(vocab_size=vocab_size, num_layers=num_layers, model_dim=model_dim,
                           heads=heads, key_dim=key_dim, value_dim=value_dim,
                           initial_retention=list(initial_retention))
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(DeltaReadBlock(model_dim, heads, key_dim, value_dim)
                                    for _ in range(num_layers))
        self.norm1, self.norm2 = RMSNorm(model_dim), RMSNorm(model_dim)
        self.proj = Linear(model_dim, vocab_size)
        self.write_key = Linear(model_dim, heads * key_dim)
        self.write_value = Linear(model_dim, heads * value_dim)
        self.write_gates = Linear(model_dim, 2 * heads)
        self.reset_parameters()
        self._compiled_segment = None

    @torch.no_grad()
    def reset_parameters(self):
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
        # These are trainable initial conditions, not imposed memory ages or
        # eviction rules. Later writes condition both gates on final hidden state.
        self.write_gates.weight.zero_()
        self.write_gates.bias[:self.config["heads"]].copy_(
            torch.tensor([math.log(a / (1 - a)) for a in self.config["initial_retention"]],
                         device=self.write_gates.bias.device))

    def initial_memory(self, batch: int, device) -> Tensor:
        return torch.zeros(batch, self.config["heads"], self.config["key_dim"],
                           self.config["value_dim"], device=device, dtype=torch.float32)

    def write(self, hidden: Tensor, memory: Tensor) -> Tensor:
        # Keep original leaves for optimizer geometry, but pack their projections.
        weights = torch.cat((self.write_key.weight, self.write_value.weight, self.write_gates.weight))
        biases = torch.cat((self.write_key.bias, self.write_value.bias, self.write_gates.bias))
        padding = -weights.shape[0] % 16
        projected = F.linear(hidden, F.pad(weights, (0, 0, 0, padding)).to(hidden.dtype),
                              F.pad(biases, (0, padding)).to(hidden.dtype))[..., :weights.shape[0]]
        h, k, v = self.config["heads"], self.config["key_dim"], self.config["value_dim"]
        key, value, gates = projected.split((h * k, h * v, 2 * h), dim=-1)
        key = unit_key(key.view(-1, h, k))
        value = value.view(-1, h, v).float()
        retention, strength = gates.float().sigmoid().split(h, dim=-1)
        return delta_write(memory, key, value, retention, strength)

    def step(self, tokens: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        x = self.norm1(self.embed(tokens))
        for block in self.blocks:
            x = block(x, memory)
        hidden = self.norm2(x)
        return hidden, self.write(hidden, memory)

    def logits(self, hidden: Tensor) -> Tensor:
        logits = self.proj(hidden).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()

    def _segment(self, inputs: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        states = []
        for tokens in inputs.unbind(1):
            hidden, memory = self.step(tokens, memory)
            states.append(hidden)
        return torch.stack(states, 1), memory

    def compile_segments(self):
        import torch._dynamo.config as config
        config.suppress_errors = False
        config.fail_on_recompile_limit_hit = True
        self._compiled_segment = torch.compile(self._segment, fullgraph=True, dynamic=False)

    def forward_hidden(self, inputs: Tensor, memory: Tensor | None = None,
                       segment_size: int = 16) -> tuple[Tensor, Tensor]:
        if inputs.device.type != "cuda":
            raise ValueError("Final-state delta carry requires CUDA execution")
        if inputs.ndim != 2 or inputs.shape[1] == 0 or segment_size < 1:
            raise ValueError("Expected nonempty [batch,time] tokens and positive segment_size")
        if memory is None:
            memory = self.initial_memory(inputs.shape[0], inputs.device)
        expected = (inputs.shape[0], self.config["heads"], self.config["key_dim"], self.config["value_dim"])
        if memory.shape != expected or memory.device != inputs.device or memory.dtype != torch.float32:
            raise ValueError("Memory must be FP32 [batch,heads,key,value] on the token device")
        if self._compiled_segment is None:
            raise RuntimeError("Call compile_segments before sequence execution")
        states = []
        for chunk in inputs.split(segment_size, 1):
            if self.training and torch.is_grad_enabled():
                hidden, memory = checkpoint(self._compiled_segment, chunk, memory,
                                            use_reentrant=False, preserve_rng_state=False)
            else:
                hidden, memory = self._compiled_segment(chunk, memory)
            states.append(hidden)
        return torch.cat(states, 1), memory

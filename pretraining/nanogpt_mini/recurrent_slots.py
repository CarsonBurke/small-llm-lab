"""Exact token recurrence with mutable slots as the only context channel.

Each block reads the same pre-token snapshot. A shared, content-dependent
writer updates all slots simultaneously after the final block. Sequence
execution and incremental decoding call the same transition; checkpoint
segments only trade recomputation for activation memory, never detach state.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pretraining.nanogpt_mini.nanogpt_mini_model import Linear, MLP, RMSNorm


class SlotReadBlock(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.heads = dim // head_dim
        self.norm1 = RMSNorm(dim)
        self.q = Linear(dim, dim)
        self.proj = Linear(dim, dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        q = self.q(self.norm1(x)).view(x.shape[0], self.heads, 1, self.head_dim)
        q = F.rms_norm(q, (self.head_dim,))
        # Slots are identities, not token positions: no temporal mask or RoPE.
        read = F.scaled_dot_product_attention(q, keys, values, scale=0.12)
        x = x + self.proj(read.flatten(1))
        return x + self.mlp(self.norm2(x))


class RecurrentSlots(nn.Module):
    def __init__(self, vocab_size=1024, num_layers=6, model_dim=512,
                 head_dim=128, slots=32, writer_dim=128):
        super().__init__()
        if min(vocab_size, num_layers, model_dim, head_dim, slots, writer_dim) < 1:
            raise ValueError("all model dimensions must be positive")
        if model_dim % head_dim:
            raise ValueError("model_dim must be divisible by head_dim")
        self.config = dict(vocab_size=vocab_size, num_layers=num_layers,
                           model_dim=model_dim, head_dim=head_dim,
                           slots=slots, writer_dim=writer_dim)
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(SlotReadBlock(model_dim, head_dim)
                                    for _ in range(num_layers))
        self.proj = Linear(model_dim, vocab_size)
        self.norm1, self.norm2 = RMSNorm(model_dim), RMSNorm(model_dim)
        self.slot_identity = nn.Parameter(torch.empty(slots, model_dim))
        # All depths share the bank encoding, but have distinct queries/outputs.
        self.memory_kv = Linear(model_dim, 2 * model_dim)
        self.writer_old = Linear(model_dim, writer_dim)
        self.writer_token = Linear(model_dim, writer_dim)
        self.writer_identity = Linear(model_dim, writer_dim)
        self.writer_candidate = Linear(writer_dim, model_dim)
        self.writer_gate = Linear(writer_dim, 1)
        self.reset_parameters()
        self._compiled_segment = None

    @torch.no_grad()
    def reset_parameters(self):
        for name, p in self.named_parameters():
            if name == "slot_identity":
                p.normal_(std=0.02)
            elif name == "embed.weight":
                p.normal_()
            elif name.endswith("gains"):
                p.fill_(1)
            elif name.endswith("weight"):
                if "proj" in name:
                    p.zero_()
                else:
                    p.normal_(std=math.sqrt(0.33 / p.shape[-1]))
            else:
                p.zero_()
        # Initial interpolation retains 90% of previous content per write.
        self.writer_gate.bias.fill_(math.log(0.1 / 0.9))

    def initial_memory(self, batch: int, device, dtype=torch.bfloat16) -> Tensor:
        return torch.zeros(batch, self.config["slots"], self.config["model_dim"],
                           device=device, dtype=dtype)

    def step(self, tokens: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        """Predict from old memory, then return the next token's memory."""
        x = self.norm1(self.embed(tokens))
        identity = self.slot_identity.type_as(memory)
        source = F.rms_norm(memory + identity, (memory.shape[-1],))
        kv = self.memory_kv(source).view(
            tokens.shape[0], self.config["slots"], 2,
            self.config["model_dim"] // self.config["head_dim"],
            self.config["head_dim"])
        keys, values = kv.unbind(2)
        keys = F.rms_norm(keys, (self.config["head_dim"],)).transpose(1, 2)
        values = values.transpose(1, 2)
        for block in self.blocks:
            x = block(x, keys, values)
        hidden = self.norm2(x)
        proposal = F.silu(
            self.writer_old(F.rms_norm(memory, (memory.shape[-1],)))
            + self.writer_token(hidden)[:, None]
            + self.writer_identity(identity)[None])
        candidate = self.writer_candidate(proposal).tanh()
        gate = self.writer_gate(proposal).sigmoid()
        updated = memory + gate * (candidate - memory)
        return hidden, updated

    def logits(self, hidden: Tensor) -> Tensor:
        logits = self.proj(hidden).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()

    def _segment(self, inputs: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        states = []
        for token in inputs.unbind(1):
            hidden, memory = self.step(token, memory)
            states.append(hidden)
        return torch.stack(states, dim=1), memory

    def compile_segments(self):
        # Bounded unrolling avoids compiling a 1024-token recurrent graph.
        import torch._dynamo.config as dynamo_config
        dynamo_config.fail_on_recompile_limit_hit = True
        self._compiled_segment = torch.compile(self._segment, fullgraph=True,
                                                dynamic=False)

    def forward_hidden(self, inputs: Tensor, memory: Tensor | None = None,
                       segment_size: int = 16) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 2 or inputs.shape[1] == 0 or segment_size < 1:
            raise ValueError("expected nonempty [batch,time] and positive segment_size")
        if memory is None:
            memory = self.initial_memory(inputs.shape[0], inputs.device,
                                         self.embed.weight.dtype)
        segment = self._compiled_segment or self._segment
        states = []
        for chunk in inputs.split(segment_size, dim=1):
            if self.training and torch.is_grad_enabled():
                hidden, memory = checkpoint(segment, chunk, memory,
                                            use_reentrant=False,
                                            preserve_rng_state=False)
            else:
                hidden, memory = segment(chunk, memory)
            states.append(hidden)
        return torch.cat(states, dim=1), memory

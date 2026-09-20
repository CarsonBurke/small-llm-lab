"""Causal attention within chunks, learned slot memory between chunks.

One contextual read follows the first block's local attention. A single writer
commits the completed chunk after all predictions have been computed. All reads
in a chunk see the same old memory, and future chunks backpropagate through the
writer without detaching. Streaming callers must buffer complete chunks; this
module deliberately rejects partial chunks rather than committing early writes.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pretraining.nanogpt_mini.nanogpt_mini_model import (
    Block, CausalSelfAttention, Linear, RMSNorm,
)


class ChunkMemory(nn.Module):
    def __init__(self, vocab_size: int = 1024, num_layers: int = 6,
                 model_dim: int = 512, head_dim: int = 128, slots: int = 32,
                 chunk_size: int = 256, checkpoint_chunks: bool = False):
        super().__init__()
        if min(vocab_size, num_layers, model_dim, head_dim, slots, chunk_size) < 1:
            raise ValueError("All model dimensions must be positive")
        if model_dim % head_dim or head_dim % 4:
            raise ValueError("head_dim must divide model_dim and be divisible by four for RoPE")
        self.config = dict(vocab_size=vocab_size, num_layers=num_layers,
                           model_dim=model_dim, head_dim=head_dim, slots=slots,
                           chunk_size=chunk_size, checkpoint_chunks=checkpoint_chunks)
        self.heads = model_dim // head_dim
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(Block(model_dim) for _ in range(num_layers))
        if head_dim != 128:
            for block in self.blocks:
                block.attn = CausalSelfAttention(model_dim, head_dim=head_dim)
        self.proj = Linear(model_dim, vocab_size)
        self.norm1, self.norm2 = RMSNorm(model_dim), RMSNorm(model_dim)

        self.slot_identity = nn.Parameter(torch.empty(slots, model_dim))
        self.memory_norm = RMSNorm(model_dim)
        self.memory_q = Linear(model_dim, model_dim)
        self.memory_kv = Linear(model_dim, 2 * model_dim)
        self.memory_proj = Linear(model_dim, model_dim)
        self.writer_q = Linear(model_dim, model_dim)
        self.writer_kv = Linear(model_dim, 2 * model_dim)
        # One shared, content-dependent interpolation strength per slot.
        self.writer_gate = Linear(2 * model_dim, 1)
        self.reset_parameters()
        self._compiled_segment = None

    @torch.no_grad()
    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if name == "slot_identity":
                parameter.normal_(std=0.02)
            elif name == "embed.weight":
                parameter.normal_()
            elif name.endswith("gains"):
                parameter.fill_(1)
            elif name.endswith("weight"):
                if "proj" in name:
                    parameter.zero_()
                else:
                    parameter.normal_(std=math.sqrt(0.33 / parameter.shape[-1]))
            else:
                parameter.zero_()
        self.writer_gate.bias.fill_(math.log(0.1 / 0.9))

    def initial_memory(self, batch: int, device, dtype=torch.bfloat16) -> Tensor:
        return torch.zeros(batch, self.config["slots"], self.config["model_dim"],
                           device=device, dtype=dtype)

    def _queries(self, projected: Tensor) -> Tensor:
        batch, positions, _ = projected.shape
        queries = projected.view(batch, positions, self.heads, self.config["head_dim"])
        return F.rms_norm(queries, (self.config["head_dim"],)).transpose(1, 2)

    def _keys_values(self, projected: Tensor) -> tuple[Tensor, Tensor]:
        batch, positions, _ = projected.shape
        projected = projected.view(batch, positions, 2, self.heads, self.config["head_dim"])
        keys, values = projected.unbind(2)
        keys = F.rms_norm(keys, (self.config["head_dim"],)).transpose(1, 2)
        return keys, values.transpose(1, 2)

    def _retrieve(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        retrieved = F.scaled_dot_product_attention(queries, keys, values, scale=0.12)
        return retrieved.transpose(1, 2).contiguous().flatten(2)

    def process_chunk(self, tokens: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        """Predict one full chunk using old memory, then commit its write."""
        if tokens.ndim != 2 or tokens.shape[1] != self.config["chunk_size"]:
            raise ValueError("process_chunk requires exactly chunk_size tokens per row")
        expected = (tokens.shape[0], self.config["slots"], self.config["model_dim"])
        if tuple(memory.shape) != expected or memory.device != tokens.device:
            raise ValueError("Memory must have [batch,slots,model_dim] on the tokens' device")
        if memory.dtype != self.embed.weight.dtype:
            raise ValueError("Memory dtype must match the embedding computation dtype")

        # The source and its reader projections are computed once per chunk.
        source = F.rms_norm(memory + self.slot_identity.type_as(memory),
                            (self.config["model_dim"],))
        keys, values = self._keys_values(self.memory_kv(source))
        x = self.norm1(self.embed(tokens))
        first = self.blocks[0]
        x = x + first.attn(first.norm1(x))
        queries = self._queries(self.memory_q(self.memory_norm(x)))
        x = x + self.memory_proj(self._retrieve(queries, keys, values))
        x = x + first.mlp(first.norm2(x))
        for block in self.blocks[1:]:
            x = block(x)
        hidden = self.norm2(x)

        # Writing can inspect the entire completed chunk because the result is
        # visible only to the next chunk. Distinct slots issue distinct queries.
        write_queries = self._queries(self.writer_q(source))
        write_keys, write_values = self._keys_values(self.writer_kv(hidden))
        retrieved = self._retrieve(write_queries, write_keys, write_values)
        old = F.rms_norm(memory, (self.config["model_dim"],))
        gate = self.writer_gate(torch.cat((old, retrieved), dim=-1)).sigmoid()
        updated = memory + gate * (retrieved.tanh() - memory)
        return hidden, updated

    def compile_segments(self):
        import torch._dynamo.config as dynamo_config
        dynamo_config.fail_on_recompile_limit_hit = True
        self._compiled_segment = torch.compile(self.process_chunk, fullgraph=True, dynamic=False)

    def forward_hidden(self, inputs: Tensor, memory: Tensor | None = None,
                       segment_size: int | None = None) -> tuple[Tensor, Tensor]:
        chunk_size = self.config["chunk_size"]
        if segment_size is not None and segment_size != chunk_size:
            raise ValueError("segment_size must equal configured chunk_size; it defines the write boundary")
        if inputs.ndim != 2 or inputs.shape[1] == 0 or inputs.shape[1] % chunk_size:
            raise ValueError("forward_hidden requires a nonempty sequence of complete chunks")
        if memory is None:
            memory = self.initial_memory(inputs.shape[0], inputs.device, self.embed.weight.dtype)
        process = self._compiled_segment or self.process_chunk
        states = []
        for chunk in inputs.split(chunk_size, dim=1):
            if self.config["checkpoint_chunks"] and self.training and torch.is_grad_enabled():
                hidden, memory = checkpoint(process, chunk, memory, use_reentrant=False,
                                            preserve_rng_state=False)
            else:
                hidden, memory = process(chunk, memory)
            states.append(hidden)
        return torch.cat(states, dim=1), memory

    def logits(self, hidden: Tensor) -> Tensor:
        logits = self.proj(hidden).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()

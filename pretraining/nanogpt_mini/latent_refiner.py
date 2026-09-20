"""Parallel causal encoder plus a shared latent refiner with three source controls.

The four-block full-width mini encoder runs once. The refiner receives either
its current encoder state, the previous encoder state, or the previous refined
state. Every arm receives a zero source at position zero. Only the refined arm
has a token recurrence; optional segment checkpoints preserve full BPTT.

All arms register and initialize identical parameters in identical order. Source
selection changes data flow only. Cached evaluation uses absolute-position
RoPE and a fixed-capacity, masked encoder KV cache. Its throughput is unmeasured.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pretraining.nanogpt_mini.nanogpt_mini_model import Block, Linear, MLP, RMSNorm


@dataclass
class LatentRefinerCache:
    """Fixed-capacity lockstep batch cache, owned and mutated by one stream."""

    keys: list[Tensor]
    values: list[Tensor]
    previous_encoder: Tensor
    previous_refined: Tensor
    index: Tensor
    positions: Tensor
    position: int = 0


class LatentRefiner(nn.Module):
    def __init__(self, vocab_size: int = 1024, encoder_layers: int = 4,
                 model_dim: int = 512, refiner_hidden: int = 2048,
                 source: str = "refined", gate_init: float = .05,
                 checkpoint_stride: int = 2):
        super().__init__()
        if min(vocab_size, encoder_layers, model_dim, refiner_hidden) < 1:
            raise ValueError("All model dimensions must be positive")
        if model_dim % 128:
            raise ValueError("model_dim must be divisible by mini's 128-wide heads")
        if source not in {"current", "encoder", "refined"}:
            raise ValueError("source must be current, encoder, or refined")
        if not math.isfinite(gate_init) or not 0 < gate_init < 1:
            raise ValueError("gate_init must be finite and strictly between zero and one")
        if isinstance(checkpoint_stride, bool) or not isinstance(checkpoint_stride, int) or checkpoint_stride < 0:
            raise ValueError("checkpoint_stride must be a nonnegative integer")
        self.config = dict(vocab_size=vocab_size, encoder_layers=encoder_layers,
                           model_dim=model_dim, refiner_hidden=refiner_hidden,
                           source=source, gate_init=gate_init,
                           checkpoint_stride=checkpoint_stride)
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(Block(model_dim) for _ in range(encoder_layers))
        self.proj = Linear(model_dim, vocab_size)
        self.norm1, self.norm2 = RMSNorm(model_dim), RMSNorm(model_dim)
        self.gate = Linear(model_dim, model_dim)
        self.refiner_norm = RMSNorm(model_dim)
        self.refiner_mlp = MLP(model_dim, refiner_hidden)
        self.final_norm = RMSNorm(model_dim)
        self.reset_parameters()
        self._compiled_encoder = None
        self._compiled_parallel = None
        self._compiled_segment = None
        self._compiled_decode = None

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
        self.gate.weight.zero_()
        self.gate.bias.fill_(math.log(self.config["gate_init"] / (1 - self.config["gate_init"])))

    def encode(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return normalized encoder states and gates, both FP32 [B,T,D]."""
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        u = self.norm2(x).float()
        # Dense projection follows BF16 autocast; sigmoid and state arithmetic
        # are FP32, avoiding an additional BF16 quantization of gate values.
        gates = self.gate(u).float().sigmoid()
        return u, gates

    def refine_step(self, u: Tensor, gate: Tensor, source: Tensor) -> Tensor:
        """One transition, supporting either [B,D] or parallel [B,T,D]."""
        z = u + gate * source
        return self.final_norm(z + self.refiner_mlp(self.refiner_norm(z))).float()

    def _parallel(self, u: Tensor, gates: Tensor) -> tuple[Tensor, Tensor]:
        zero = torch.zeros_like(u[:, :1])
        if self.config["source"] == "current":
            source = torch.cat((zero, u[:, 1:]), dim=1)
        else:
            source = torch.cat((zero, u[:, :-1]), dim=1)
        states = self.refine_step(u, gates, source)
        return states, states[:, -1]

    def _segment(self, u: Tensor, gates: Tensor, previous: Tensor) -> tuple[Tensor, Tensor]:
        states = []
        for token_u, token_gate in zip(u.unbind(1), gates.unbind(1)):
            previous = self.refine_step(token_u, token_gate, previous)
            states.append(previous)
        return torch.stack(states, dim=1), previous

    def compile_segments(self):
        """Compile bounded recurrence and whole parallel stages without fallback."""
        import torch._dynamo.config as dynamo_config
        dynamo_config.suppress_errors = False
        dynamo_config.fail_on_recompile_limit_hit = True
        self._compiled_encoder = torch.compile(self.encode, fullgraph=True, dynamic=False)
        self._compiled_parallel = torch.compile(self._parallel, fullgraph=True, dynamic=False)
        self._compiled_segment = torch.compile(self._segment, fullgraph=True, dynamic=False)

    def refine_from_encoded(self, u: Tensor, gates: Tensor,
                            segment_size: int = 16) -> tuple[Tensor, Tensor]:
        """Refine a complete fresh sequence, with no detach at segment boundaries."""
        if u.ndim != 3 or u.shape != gates.shape or u.shape[1] == 0:
            raise ValueError("Expected matching nonempty [batch,time,dim] states and gates")
        if u.shape[-1] != self.config["model_dim"] or segment_size < 1:
            raise ValueError("Expected model_dim states and positive segment_size")
        if u.device.type != "cuda" or gates.device != u.device:
            raise ValueError("Latent refiner requires CUDA; no CPU fallback")
        if u.dtype != torch.float32 or gates.dtype != torch.float32:
            raise ValueError("Encoder states and gates must be FP32")
        if self._compiled_segment is None:
            self.compile_segments()
        if self.config["source"] != "refined":
            return self._compiled_parallel(u, gates)
        previous = torch.zeros_like(u[:, 0])
        outputs = []
        # Single splits avoid independent full-size slice-gradient allocations.
        stride = self.config["checkpoint_stride"]
        # 0 keeps all activations; 1 recomputes every segment; N>1 recomputes
        # segment indices 0,N,2N,... without changing recurrence or gradients.
        for index, (chunk_u, chunk_gate) in enumerate(zip(u.split(segment_size, dim=1), gates.split(segment_size, dim=1))):
            if stride > 0 and index % stride == 0 and self.training and torch.is_grad_enabled():
                hidden, previous = checkpoint(self._compiled_segment, chunk_u, chunk_gate, previous,
                                              use_reentrant=False, preserve_rng_state=False)
            else:
                hidden, previous = self._compiled_segment(chunk_u, chunk_gate, previous)
            outputs.append(hidden)
        return torch.cat(outputs, dim=1), previous

    def forward_hidden(self, inputs: Tensor, segment_size: int = 16) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 2 or inputs.shape[1] == 0 or segment_size < 1:
            raise ValueError("Expected nonempty [batch,time] tokens and positive segment_size")
        if inputs.device.type != "cuda":
            raise ValueError("Latent refiner requires CUDA; no CPU fallback")
        if self._compiled_encoder is None:
            self.compile_segments()
        u, gates = self._compiled_encoder(inputs)
        return self.refine_from_encoded(u, gates, segment_size)

    def logits(self, hidden: Tensor) -> Tensor:
        logits = self.proj(hidden).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()

    def new_cache(self, batch_size: int, max_seq_len: int, device=None) -> LatentRefinerCache:
        """Create zero state; capacity exhaustion is an error, never a window slide."""
        device = self.embed.weight.device if device is None else torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = self.embed.weight.device
        if batch_size < 1 or max_seq_len < 1 or device.type != "cuda":
            raise ValueError("Cache requires positive batch/capacity and a CUDA device")
        if device != self.embed.weight.device:
            raise ValueError("Cache must reside on the model device")
        dim = self.config["model_dim"]
        shape = (batch_size, dim // 128, max_seq_len, 128)
        return LatentRefinerCache(
            keys=[torch.zeros(shape, device=device, dtype=torch.bfloat16) for _ in self.blocks],
            values=[torch.zeros(shape, device=device, dtype=torch.bfloat16) for _ in self.blocks],
            previous_encoder=torch.zeros(batch_size, dim, device=device, dtype=torch.float32),
            previous_refined=torch.zeros(batch_size, dim, device=device, dtype=torch.float32),
            index=torch.zeros(1, device=device, dtype=torch.int64),
            positions=torch.arange(max_seq_len, device=device),
        )

    @staticmethod
    def _rotate_position(x: Tensor, angular_freq: Tensor, index: Tensor) -> Tensor:
        # Exact mini half-truncated RoPE with the absolute cached position.
        theta = index.float().view(1, 1, 1, 1) * angular_freq.view(1, 1, 1, -1)
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1).type_as(x)

    def _decode(self, tokens: Tensor, cache: LatentRefinerCache) -> Tensor:
        x = self.norm1(self.embed(tokens))[:, None]
        mask = (cache.positions <= cache.index).view(1, 1, 1, -1)
        for block, keys, values in zip(self.blocks, cache.keys, cache.values):
            attention = block.attn
            normalized = block.norm1(x)
            shape = (tokens.shape[0], 1, attention.num_heads, attention.head_dim)
            q, k, v = (projection(normalized).view(shape)
                       for projection in (attention.q, attention.k, attention.v))
            q, k = F.rms_norm(q, (q.shape[-1],)), F.rms_norm(k, (k.shape[-1],))
            q = self._rotate_position(q, attention.rotary.angular_freq, cache.index)
            k = self._rotate_position(k, attention.rotary.angular_freq, cache.index)
            keys.index_copy_(2, cache.index, k.transpose(1, 2))
            values.index_copy_(2, cache.index, v.transpose(1, 2))
            attended = F.scaled_dot_product_attention(q.transpose(1, 2), keys, values,
                                                       attn_mask=mask, scale=.12, is_causal=False)
            x = x + attention.proj(attended.transpose(1, 2).contiguous().view(tokens.shape[0], 1, -1))
            x = x + block.mlp(block.norm2(x))
        u = self.norm2(x[:, 0]).float()
        gates = self.gate(u).float().sigmoid()
        if self.config["source"] == "current":
            source = u
        elif self.config["source"] == "encoder":
            source = cache.previous_encoder
        else:
            source = cache.previous_refined
        source = torch.where(cache.index == 0, torch.zeros_like(source), source)
        hidden = self.refine_step(u, gates, source)
        cache.previous_encoder.copy_(u)
        cache.previous_refined.copy_(hidden)
        cache.index.add_(1)
        return hidden

    @torch.no_grad()
    def decode_step(self, tokens: Tensor, cache: LatentRefinerCache) -> Tensor:
        """Return one hidden state and mutate the cache, using fixed-shape compilation.

        All rows advance together. The compiled decoder attends the fixed cache
        capacity with an explicit valid-prefix mask. This preserves the training
        contract; decoding throughput has not been measured.
        """
        if self.training:
            raise ValueError("Cached decoding requires eval mode")
        if tokens.ndim != 1 or tokens.shape[0] != cache.previous_encoder.shape[0]:
            raise ValueError("decode_step expects one token per cache batch row")
        if tokens.device.type != "cuda" or tokens.device != self.embed.weight.device:
            raise ValueError("Cached decoding requires tokens on the model CUDA device")
        if cache.previous_encoder.device != tokens.device:
            raise ValueError("Cache and tokens must share a CUDA device")
        if len(cache.keys) != len(self.blocks) or cache.previous_encoder.shape[1] != self.config["model_dim"]:
            raise ValueError("Cache dimensions do not match the model")
        if cache.position >= cache.positions.numel():
            raise ValueError("Decoder cache capacity exhausted")
        if self._compiled_decode is None:
            import torch._dynamo.config as dynamo_config
            dynamo_config.suppress_errors = False
            dynamo_config.fail_on_recompile_limit_hit = True
            self._compiled_decode = torch.compile(self._decode, fullgraph=True, dynamic=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            hidden = self._compiled_decode(tokens, cache)
        cache.position += 1
        return hidden

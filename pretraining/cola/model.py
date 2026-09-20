# Portions adapted from Cola-DLM modeling_cola_vae.py and modeling_cola_dit.py.
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# Licensed under the Apache License, Version 2.0:
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Position-aligned Cola VAE and block-causal flow prior, with FP32 master weights.

Neural execution is CUDA/BF16 and compiled only. Construction, including on the
meta device for parameter accounting, does not execute a network. Call
``compile_components`` after moving the model to CUDA. Compilation is lazy;
the first invocation of each kernel performs its compilation, without fallback.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from .config import ModelConfig

CleanCache = tuple[tuple[Tensor, Tensor], ...]


class BF16Linear(nn.Linear):
    """Mini-style explicit compute casts, retaining FP32 optimizer parameters."""

    def forward(self, x: Tensor) -> Tensor:
        bias = None if self.bias is None else self.bias.to(torch.bfloat16)
        return F.linear(x.to(torch.bfloat16), self.weight.to(torch.bfloat16), bias)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x.float(), self.normalized_shape, self.weight, self.bias, self.eps
        ).to(x.dtype)


def _rotary_tables(length: int, width: int, theta: float) -> tuple[Tensor, Tensor]:
    frequencies = theta ** (-torch.arange(0, width, 2, dtype=torch.float32) / width)
    angles = torch.arange(length, dtype=torch.float32)[:, None] * frequencies[None, :]
    return angles.cos(), angles.sin()


def _rotate(x: Tensor, cos: Tensor, sin: Tensor, *, interleaved: bool) -> Tensor:
    """Upstream VAE uses split halves; upstream DiT uses adjacent rotary pairs."""
    width = cos.shape[-1] * 2
    rotary = x[..., :width].float()
    if interleaved:
        first, second = rotary.unflatten(-1, (-1, 2)).unbind(-1)
        rotated = torch.stack(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        ).flatten(-2)
    else:
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
    rotated = rotated.to(x.dtype)
    return (
        rotated
        if width == x.shape[-1]
        else torch.cat((rotated, x[..., width:]), dim=-1)
    )


def _check_runtime(module: nn.Module, tensor: Tensor, compiled: bool) -> None:
    if not compiled:
        raise RuntimeError(
            "call compile_components() or compile_component() before neural execution"
        )
    parameter = next(module.parameters())
    if tensor.device.type != "cuda" or tensor.device != parameter.device:
        raise RuntimeError(
            "Cola neural execution requires inputs and FP32 master weights on the same CUDA device"
        )
    if torch.is_autocast_enabled("cuda"):
        raise RuntimeError("Cola uses explicit BF16 compute; do not enable autocast")


def _check_compile(module: nn.Module) -> None:
    parameters = tuple(module.parameters())
    if not parameters or parameters[0].device.type != "cuda":
        raise RuntimeError("move FP32 master weights to CUDA before compiling Cola")
    if any(
        p.device != parameters[0].device or p.dtype != torch.float32 for p in parameters
    ):
        raise RuntimeError("all Cola master parameters must be FP32 on one CUDA device")


def _check_sequence(x: Tensor, config: ModelConfig, width: int | None = None) -> None:
    ndim = 2 if width is None else 3
    if x.ndim != ndim or not 0 < x.shape[1] <= config.seq_len or x.shape[0] < 1:
        raise ValueError(
            "expected a nonempty batch with 1..config.seq_len token positions"
        )
    if width is not None and (x.shape[-1] != width or not x.is_floating_point()):
        raise ValueError(
            "latent tensors must be floating point with config.latent_dim channels"
        )


class VAEBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.vae_dim
        self.heads = config.vae_heads
        self.head_dim = dim // self.heads
        self.norm_attn = LayerNorm(dim, eps=config.vae_norm_eps)
        self.qkv = BF16Linear(dim, 3 * dim, bias=False)
        # Upstream VAE Q/K normalization is across model width, not per head.
        self.q_norm = LayerNorm(dim, eps=config.vae_norm_eps)
        self.k_norm = LayerNorm(dim, eps=config.vae_norm_eps)
        self.attn_out = BF16Linear(dim, dim)
        self.norm_ffn = LayerNorm(dim, eps=config.vae_norm_eps)
        self.ffn_in = BF16Linear(dim, config.vae_ffn_dim)
        self.ffn_out = BF16Linear(config.vae_ffn_dim // 2, dim)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = (
            self.q_norm(q)
            .view(batch, length, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_norm(k)
            .view(batch, length, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q = _rotate(q, cos, sin, interleaved=False)
        k = _rotate(k, cos, sin, interleaved=False)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        # Preserve Cola's post_norm=True equations, including its attention
        # residual normalization (this is intentionally not a pre-norm block).
        x = self.norm_attn(x) + self.attn_out(attended)
        value, gate = self.ffn_in(x).chunk(2, dim=-1)
        return x + self.norm_ffn(self.ffn_out(value * F.silu(gate)))


class VAEEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size + 1, config.vae_dim)
        # Original patch-size-one projection, with no grouping of token positions.
        self.input_projection = BF16Linear(config.vae_dim, config.vae_dim)
        self.blocks = nn.ModuleList(VAEBlock(config) for _ in range(config.vae_layers))
        self.posterior = BF16Linear(config.vae_dim, 2 * config.latent_dim)
        cos, sin = _rotary_tables(
            config.seq_len, config.vae_dim // config.vae_heads, config.rope_theta
        )
        self.register_buffer("rotary_cos", cos, persistent=False)
        self.register_buffer("rotary_sin", sin, persistent=False)
        self._compiled_forward: Callable | None = None

    @property
    def compiled(self) -> bool:
        return self._compiled_forward is not None

    def compile_component(self) -> None:
        _check_compile(self)
        # Compile the unbound function: deepcopy of a prepared encoder cannot
        # accidentally keep a callable bound to the live training encoder.
        self._compiled_forward = torch.compile(
            type(self)._forward, fullgraph=True, dynamic=True
        )

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        _check_runtime(self, ids, self.compiled)
        _check_sequence(ids, self.config)
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "token IDs must be int32/int64, including the encoder-only mask ID"
            )
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return self._compiled_forward(self, ids)

    def _forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        x = F.embedding(ids, self.embedding.weight.to(torch.bfloat16))
        x = self.input_projection(x)
        cos, sin = self.rotary_cos[: ids.shape[1]], self.rotary_sin[: ids.shape[1]]
        for block in self.blocks:
            x = block(x, cos, sin)
        mean, logvar = self.posterior(x).float().chunk(2, dim=-1)
        mean = F.layer_norm(
            mean, (self.config.latent_dim,), eps=self.config.vae_norm_eps
        )
        return mean, logvar.clamp(-30.0, 20.0)


class VAEDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_projection = BF16Linear(config.latent_dim, config.vae_dim)
        self.blocks = nn.ModuleList(VAEBlock(config) for _ in range(config.vae_layers))
        self.output_projection = BF16Linear(config.vae_dim, config.vae_dim)
        self.final_norm = LayerNorm(config.vae_dim, eps=config.vae_norm_eps)
        self.head = BF16Linear(config.vae_dim, config.vocab_size)
        cos, sin = _rotary_tables(
            config.seq_len, config.vae_dim // config.vae_heads, config.rope_theta
        )
        self.register_buffer("rotary_cos", cos, persistent=False)
        self.register_buffer("rotary_sin", sin, persistent=False)
        self._compiled_forward: Callable | None = None

    @property
    def compiled(self) -> bool:
        return self._compiled_forward is not None

    def compile_component(self) -> None:
        _check_compile(self)
        self._compiled_forward = torch.compile(
            type(self)._forward, fullgraph=True, dynamic=True
        )

    def forward(self, z: Tensor) -> Tensor:
        _check_runtime(self, z, self.compiled)
        _check_sequence(z, self.config, self.config.latent_dim)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return self._compiled_forward(self, z)

    def _forward(self, z: Tensor) -> Tensor:
        x = self.input_projection(z)
        cos, sin = self.rotary_cos[: z.shape[1]], self.rotary_sin[: z.shape[1]]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.final_norm(self.output_projection(x))).float()


def _init_vae(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(module.weight, std=0.02, a=-0.06, b=0.06)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


class ColaVAE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = VAEEncoder(config)
        self.decoder = VAEDecoder(config)
        self.apply(_init_vae)
        # Reference patch Conv1d is excluded from its Linear/Embedding initializer.
        # Kernel-one Conv1d and Linear share this fan-in and default initialization.
        self.encoder.input_projection.reset_parameters()
        # Declared Mini-style deviation: start with uniform reconstruction.
        # The posterior projection stays nonzero, preserving input information.
        nn.init.zeros_(self.decoder.head.weight)
        nn.init.zeros_(self.decoder.head.bias)

    def encode(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(ids)

    def decode(self, z: Tensor) -> Tensor:
        """Aligned causal reconstruction: token i may read latent i, never i+1."""
        return self.decoder(z)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj_in = BF16Linear(256, dim)
        self.proj_hid = BF16Linear(dim, dim)
        self.proj_out = BF16Linear(dim, dim)
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(128, dtype=torch.float32) / 128
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, times: Tensor) -> Tensor:
        # Original sinusoidal convention: sin then cos, denominator half_dim,
        # and internal time units 0..1000 while the flow API uses 0..1.
        angles = (1000.0 * times.float()).unsqueeze(-1) * self.frequencies
        x = torch.cat((angles.sin(), angles.cos()), dim=-1).to(torch.bfloat16)
        return self.proj_out(F.silu(self.proj_hid(F.silu(self.proj_in(x)))))


class DiTAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.dit_heads
        self.head_dim = config.dit_dim // config.dit_heads
        self.qkv = BF16Linear(config.dit_dim, 3 * config.dit_dim, bias=False)
        self.out = BF16Linear(config.dit_dim, config.dit_dim, bias=False)
        self.q_norm = LayerNorm(self.head_dim, eps=config.dit_norm_eps)
        self.k_norm = LayerNorm(self.head_dim, eps=config.dit_norm_eps)

    def project(
        self, x: Tensor, cos: Tensor, sin: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = x.shape
        q, k, v = (
            self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).unbind(2)
        )
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = _rotate(self.q_norm(q), cos, sin, interleaved=True)
        k = _rotate(self.k_norm(k), cos, sin, interleaved=True)
        return q, k, v

    def output(self, attended: Tensor) -> Tensor:
        return self.out(attended.transpose(1, 2).flatten(2))


class DiTBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.dit_dim
        self.attn_norm = LayerNorm(
            dim, eps=config.dit_norm_eps, elementwise_affine=False
        )
        self.attention = DiTAttention(config)
        self.ffn_norm = LayerNorm(
            dim, eps=config.dit_norm_eps, elementwise_affine=False
        )
        self.ffn_in = BF16Linear(dim, config.dit_mlp_ratio * dim)
        self.ffn_out = BF16Linear(config.dit_mlp_ratio * dim, dim)
        # Fusing the four original AdaLN projections does not change their math.
        self.modulation = BF16Linear(dim, 6 * dim)

    def prepare(self, x: Tensor, emb: Tensor, cos: Tensor, sin: Tensor):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(
            F.silu(emb)
        ).chunk(6, dim=-1)
        q, k, v = self.attention.project(
            self.attn_norm(x) * (1 + scale_a) + shift_a, cos, sin
        )
        return q, k, v, gate_a, shift_f, scale_f, gate_f

    def finish(
        self,
        x: Tensor,
        attended: Tensor,
        gate_a: Tensor,
        shift_f: Tensor,
        scale_f: Tensor,
        gate_f: Tensor,
    ) -> Tensor:
        x = x + gate_a * self.attention.output(attended)
        hidden = self.ffn_norm(x) * (1 + scale_f) + shift_f
        return x + gate_f * self.ffn_out(
            F.gelu(self.ffn_in(hidden), approximate="tanh")
        )

    def forward(
        self, x: Tensor, emb: Tensor, cos: Tensor, sin: Tensor, mask: BlockMask
    ) -> Tensor:
        q, k, v, gate_a, shift_f, scale_f, gate_f = self.prepare(x, emb, cos, sin)
        # AUTO's selected kernel violates this mixed clean/noisy mask on the
        # installed CUDA stack. Explicit Triton preserves conditional visibility
        # and agrees with cached FlashSDPA; never silently switch backends.
        attended = flex_attention(
            q, k, v, block_mask=mask, kernel_options={"BACKEND": "TRITON"}
        )
        return self.finish(x, attended, gate_a, shift_f, scale_f, gate_f)

    def cached(
        self,
        x: Tensor,
        emb: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: tuple[Tensor, Tensor] | None,
    ):
        q, k, v, gate_a, shift_f, scale_f, gate_f = self.prepare(x, emb, cos, sin)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        # Every current query sees every predecessor clean key and every key in
        # its own block; no triangular mask is appropriate within a DiT block.
        attended = F.scaled_dot_product_attention(q, k, v)
        return self.finish(x, attended, gate_a, shift_f, scale_f, gate_f), (k, v)


class ColaPrior(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_projection = BF16Linear(config.latent_dim, config.dit_dim)
        self.time_embedding = TimestepEmbedding(config.dit_dim)
        self.blocks = nn.ModuleList(DiTBlock(config) for _ in range(config.dit_layers))
        self.final_norm = LayerNorm(config.dit_dim, eps=config.dit_norm_eps)
        self.final_modulation = BF16Linear(config.dit_dim, 2 * config.dit_dim)
        self.head = BF16Linear(config.dit_dim, config.latent_dim)
        cos, sin = _rotary_tables(
            config.seq_len, config.dit_rope_dim, config.dit_rope_theta
        )
        self.register_buffer("rotary_cos", cos, persistent=False)
        self.register_buffer("rotary_sin", sin, persistent=False)
        self.register_buffer(
            "training_cos", torch.cat((cos, cos), dim=0), persistent=False
        )
        self.register_buffer(
            "training_sin", torch.cat((sin, sin), dim=0), persistent=False
        )
        self._block_mask: BlockMask | None = None
        self._compiled_forward: Callable | None = None
        self._compiled_append: Callable | None = None
        self._compiled_velocity: Callable | None = None
        self.apply(self._init_weights)
        for projection in (
            self.time_embedding.proj_in,
            self.time_embedding.proj_hid,
            self.time_embedding.proj_out,
        ):
            nn.init.normal_(projection.weight, std=0.02)
        for projection in [block.modulation for block in self.blocks] + [
            self.final_modulation,
            self.head,
        ]:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def compiled(self) -> bool:
        return all(
            kernel is not None
            for kernel in (
                self._compiled_forward,
                self._compiled_append,
                self._compiled_velocity,
            )
        )

    def compile_component(self) -> None:
        _check_compile(self)
        length, block_size = self.config.seq_len, self.config.block_size

        def mask_mod(batch, head, query, key):
            del batch, head
            query_clean, key_clean = query < length, key < length
            query_block = (query % length) // block_size
            key_block = (key % length) // block_size
            clean_visibility = query_clean & key_clean & (key_block <= query_block)
            noisy_visibility = (~query_clean) & (
                (key_clean & (key_block < query_block))
                | ((~key_clean) & (key_block == query_block))
            )
            return (
                (query < 2 * length)
                & (key < 2 * length)
                & (clean_visibility | noisy_visibility)
            )

        self._block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=2 * length,
            KV_LEN=2 * length,
            device=str(self.input_projection.weight.device),
            _compile=True,
        )
        self._compiled_forward = torch.compile(
            type(self)._forward, fullgraph=True, dynamic=False
        )
        self._compiled_append = torch.compile(
            type(self)._append, fullgraph=True, dynamic=True
        )
        self._compiled_velocity = torch.compile(
            type(self)._velocity, fullgraph=True, dynamic=True
        )

    def forward(self, clean: Tensor, noisy: Tensor, times: Tensor) -> Tensor:
        """Velocity dz/ds for s in [0,1]; clean predecessor latents are detached."""
        _check_runtime(self, noisy, self.compiled)
        _check_sequence(noisy, self.config, self.config.latent_dim)
        if clean.shape != noisy.shape or noisy.shape[1] != self.config.seq_len:
            raise ValueError(
                "full-2L prior requires clean/noisy [B, config.seq_len, latent_dim]"
            )
        if clean.device != noisy.device or not clean.is_floating_point():
            raise ValueError(
                "clean/noisy latents must share CUDA device and floating-point type"
            )
        self._check_times(times, noisy)
        return self._compiled_forward(self, clean, noisy, times)

    @staticmethod
    def _check_times(times: Tensor, z: Tensor) -> None:
        if (
            times.shape != z.shape[:2]
            or times.device != z.device
            or not times.is_floating_point()
        ):
            raise ValueError(
                "times must be floating point [B,T] on the latent device, in [0,1]"
            )

    def _output(self, x: Tensor, emb: Tensor) -> Tensor:
        shift, scale = self.final_modulation(F.silu(emb)).chunk(2, dim=-1)
        return self.head(self.final_norm(x) * (1 + scale) + shift).float()

    def _forward(self, clean: Tensor, noisy: Tensor, times: Tensor) -> Tensor:
        x = self.input_projection(torch.cat((clean.detach(), noisy), dim=1))
        emb = self.time_embedding(torch.cat((torch.zeros_like(times), times), dim=1))
        for block in self.blocks:
            x = block(x, emb, self.training_cos, self.training_sin, self._block_mask)
        return self._output(x[:, self.config.seq_len :], emb[:, self.config.seq_len :])

    def _check_block(self, z: Tensor, cache: CleanCache | None) -> None:
        _check_runtime(self, z, self.compiled)
        _check_sequence(z, self.config, self.config.latent_dim)
        if z.shape[1] != self.config.block_size:
            raise ValueError("cached prior calls require one complete diffusion block")
        if cache is None:
            return
        if not isinstance(cache, tuple) or len(cache) != self.config.dit_layers:
            raise ValueError(
                "clean cache must be an immutable per-layer (key,value) tuple"
            )
        prefix = cache[0][0].shape[2]
        if (
            prefix < self.config.block_size
            or prefix % self.config.block_size
            or prefix + z.shape[1] > self.config.seq_len
        ):
            raise ValueError(
                "clean cache prefix must contain complete blocks and fit the configured context"
            )
        expected = (
            z.shape[0],
            self.config.dit_heads,
            prefix,
            self.config.dit_dim // self.config.dit_heads,
        )
        for pair in cache:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("each cache layer must be a (key,value) tuple")
            if any(
                t.shape != expected or t.device != z.device or t.dtype != torch.bfloat16
                for t in pair
            ):
                raise ValueError(
                    "cache tensors must be BF16 [B,H,prefix,head_dim] on the latent device"
                )

    def append_clean(
        self, z_block: Tensor, cache: CleanCache | None = None
    ) -> CleanCache:
        """Return a new detached clean t=0 cache; neither input cache nor z changes."""
        self._check_block(z_block, cache)
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return self._compiled_append(self, z_block, cache)

    def block_velocity(
        self, noisy_block: Tensor, times: Tensor, cache: CleanCache | None = None
    ) -> Tensor:
        """Conditional velocity using fixed predecessor clean keys, without mutation."""
        self._check_block(noisy_block, cache)
        self._check_times(times, noisy_block)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return self._compiled_velocity(self, noisy_block, times, cache)

    def _append(self, z: Tensor, cache: CleanCache | None) -> CleanCache:
        prefix = 0 if cache is None else cache[0][0].shape[2]
        cos = self.rotary_cos[prefix : prefix + z.shape[1]]
        sin = self.rotary_sin[prefix : prefix + z.shape[1]]
        x = self.input_projection(z.detach())
        emb = self.time_embedding(
            torch.zeros(z.shape[:2], device=z.device, dtype=torch.float32)
        )
        new_cache = []
        for index, block in enumerate(self.blocks):
            x, pair = block.cached(
                x, emb, cos, sin, None if cache is None else cache[index]
            )
            new_cache.append((pair[0].detach(), pair[1].detach()))
        return tuple(new_cache)

    def _velocity(self, z: Tensor, times: Tensor, cache: CleanCache | None) -> Tensor:
        prefix = 0 if cache is None else cache[0][0].shape[2]
        cos = self.rotary_cos[prefix : prefix + z.shape[1]]
        sin = self.rotary_sin[prefix : prefix + z.shape[1]]
        x = self.input_projection(z)
        emb = self.time_embedding(times)
        for index, block in enumerate(self.blocks):
            x, _ = block.cached(
                x, emb, cos, sin, None if cache is None else cache[index]
            )
        return self._output(x, emb)


class ColaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vae = ColaVAE(config)
        self.prior = ColaPrior(config)

    @property
    def compiled(self) -> bool:
        return (
            self.vae.encoder.compiled
            and self.vae.decoder.compiled
            and self.prior.compiled
        )

    def compile_components(self) -> None:
        """Prepare static training sparsity and compile all train/generation paths.

        Encoder/decoder and cached generation use dynamic sequence lengths;
        training FlexAttention uses exactly config.seq_len positions per stream.
        A frozen deepcopy of ``vae.encoder`` can call ``compile_component`` to
        prepare its independent reference-encoder invocation after stage one.
        """
        self.vae.encoder.compile_component()
        self.vae.decoder.compile_component()
        self.prior.compile_component()

    def muon_parameters(self) -> list[nn.Parameter]:
        """Block attention/FFN matrices only; all other parameters use AdamW."""
        parameters = []
        for stack in (self.vae.encoder.blocks, self.vae.decoder.blocks):
            for block in stack:
                parameters.extend(
                    layer.weight
                    for layer in (
                        block.qkv,
                        block.attn_out,
                        block.ffn_in,
                        block.ffn_out,
                    )
                )
        for block in self.prior.blocks:
            parameters.extend(
                layer.weight
                for layer in (
                    block.attention.qkv,
                    block.attention.out,
                    block.ffn_in,
                    block.ffn_out,
                )
            )
        return parameters

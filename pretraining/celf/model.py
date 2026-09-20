"""CE-anchored latent flow over byte patches, with FP32 master weights.

Three components share one latent contract:

* ``Encoder``: deterministic, block-bidirectional / block-causal transformer
  over byte patches, emitting one unit-power latent per patch.
* ``Decoder``: block-bidirectional / block-causal transformer that reads only
  latents at the anchoring noise level and emits per-byte logits.
* ``FlowPrior``: block-causal DiT velocity field whose history conditioning is
  the same anchored latent the decoder reads; the noisy stream is the flow path
  of the current block.

Neural execution is CUDA/BF16 and compiled only. Construction, including on the
meta device, does not execute a network. Call ``compile_components`` after
moving the model to CUDA; compilation is lazy and has no eager fallback.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig

CleanCache = tuple[tuple[Tensor, Tensor], ...]


class BF16Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = None if self.bias is None else self.bias.to(torch.bfloat16)
        return F.linear(x.to(torch.bfloat16), self.weight.to(torch.bfloat16), bias)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, affine: bool = True):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x.float(), (self.dim,), self.weight, self.eps).to(x.dtype)


def unit_power(z: Tensor, eps: float) -> Tensor:
    """Per-position unit RMS in FP32; the only latent geometry constraint."""
    z = z.float()
    return z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)


def _rotary_tables(length: int, width: int, theta: float) -> tuple[Tensor, Tensor]:
    frequencies = theta ** (-torch.arange(0, width, 2, dtype=torch.float32) / width)
    angles = torch.arange(length, dtype=torch.float32)[:, None] * frequencies[None, :]
    return angles.cos(), angles.sin()


def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Split-half rotary on the leading ``2 * cos.shape[-1]`` channels."""
    width = cos.shape[-1] * 2
    rotary = x[..., :width].float()
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    ).to(x.dtype)
    return rotated if width == x.shape[-1] else torch.cat((rotated, x[..., width:]), -1)


def block_causal_mask(length: int, block: int, device) -> Tensor:
    """Bidirectional within a block, causal across blocks; True means visible."""
    index = torch.arange(length, device=device) // block
    return (index[None, :] <= index[:, None])[None, None]


def two_stream_mask(length: int, block: int, device) -> Tensor:
    """History stream then flow stream, both ``length`` long.

    History queries see history keys in their block or earlier. Flow queries
    see history keys strictly before their block and flow keys in their block.
    """
    position = torch.arange(2 * length, device=device)
    history = position < length
    block_index = (position % length) // block
    q_history, k_history = history[:, None], history[None, :]
    q_block, k_block = block_index[:, None], block_index[None, :]
    visible = (q_history & k_history & (k_block <= q_block)) | (
        ~q_history & ((k_history & (k_block < q_block)) | (~k_history & (k_block == q_block)))
    )
    return visible[None, None]


def _check_runtime(module: nn.Module, tensor: Tensor, compiled: bool) -> None:
    if not compiled:
        raise RuntimeError("call compile_components() before neural execution")
    parameter = next(module.parameters())
    if tensor.device.type != "cuda" or tensor.device != parameter.device:
        raise RuntimeError(
            "CELF neural execution requires inputs and FP32 master weights on one CUDA device"
        )
    if torch.is_autocast_enabled("cuda"):
        raise RuntimeError("CELF uses explicit BF16 compute; do not enable autocast")


def _check_compile(module: nn.Module) -> None:
    parameters = tuple(module.parameters())
    if not parameters or parameters[0].device.type != "cuda":
        raise RuntimeError("move FP32 master weights to CUDA before compiling CELF")
    if any(
        p.device != parameters[0].device or p.dtype != torch.float32 for p in parameters
    ):
        raise RuntimeError("all CELF master parameters must be FP32 on one CUDA device")


def _check_latents(z: Tensor, config: ModelConfig) -> None:
    if (
        z.ndim != 3
        or z.shape[0] < 1
        or not 0 < z.shape[1] <= config.seq_patches
        or z.shape[1] % config.block_patches
        or z.shape[-1] != config.latent_dim
        or not z.is_floating_point()
    ):
        raise ValueError(
            "latents must be floating [B, complete blocks <= seq_patches, latent_dim]"
        )


class CodecAttention(nn.Module):
    def __init__(self, dim: int, heads: int, eps: float):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = BF16Linear(dim, 3 * dim, bias=False)
        self.out = BF16Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps)
        self.k_norm = RMSNorm(self.head_dim, eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = _rotate(self.q_norm(q), cos, sin)
        k = _rotate(self.k_norm(k), cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(attended.transpose(1, 2).flatten(2))


class CodecBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.codec_dim
        self.attn_norm = RMSNorm(dim, config.norm_eps)
        self.attention = CodecAttention(dim, config.codec_heads, config.norm_eps)
        self.ffn_norm = RMSNorm(dim, config.norm_eps)
        self.ffn_in = BF16Linear(dim, 2 * config.codec_ffn_dim, bias=False)
        self.ffn_out = BF16Linear(config.codec_ffn_dim, dim, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attention(self.attn_norm(x), cos, sin, mask)
        value, gate = self.ffn_in(self.ffn_norm(x)).chunk(2, dim=-1)
        return x + self.ffn_out(value * F.silu(gate))


class _Codec(nn.Module):
    """Shared block-causal transformer body with lazy compiled execution."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(CodecBlock(config) for _ in range(config.codec_layers))
        self.final_norm = RMSNorm(config.codec_dim, config.norm_eps)
        cos, sin = _rotary_tables(
            config.seq_patches, config.codec_dim // config.codec_heads, config.rope_theta
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

    def _body(self, x: Tensor) -> Tensor:
        length = x.shape[1]
        cos, sin = self.rotary_cos[:length], self.rotary_sin[:length]
        mask = block_causal_mask(length, self.config.block_patches, x.device)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        return self.final_norm(x)


class Encoder(_Codec):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.byte_embedding = nn.Embedding(config.vocab_size, config.byte_dim)
        self.mask_embedding = nn.Parameter(
            torch.zeros(config.patch_size * config.byte_dim)
        )
        self.input_projection = BF16Linear(config.patch_size * config.byte_dim, config.codec_dim)
        self.latent_projection = BF16Linear(config.codec_dim, config.latent_dim)

    def forward(self, patches: Tensor, masked: Tensor) -> Tensor:
        """Unit-power latents [B, L, D] from byte patches [B, L, P] and a mask [B, L]."""
        _check_runtime(self, patches, self.compiled)
        config = self.config
        if (
            patches.ndim != 3
            or patches.shape[0] < 1
            or not 0 < patches.shape[1] <= config.seq_patches
            or patches.shape[1] % config.block_patches
            or patches.shape[2] != config.patch_size
            or patches.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "patches must be integer [B, complete blocks <= seq_patches, patch_size]"
            )
        if masked.shape != patches.shape[:2] or masked.dtype != torch.bool:
            raise ValueError("masked must be a boolean [B, L] patch mask")
        return self._compiled_forward(self, patches, masked)

    def _forward(self, patches: Tensor, masked: Tensor) -> Tensor:
        embedded = F.embedding(patches, self.byte_embedding.weight.to(torch.bfloat16))
        embedded = embedded.flatten(2)
        embedded = torch.where(
            masked[..., None], self.mask_embedding.to(torch.bfloat16), embedded
        )
        x = self._body(self.input_projection(embedded))
        return unit_power(self.latent_projection(x), self.config.norm_eps)


class Decoder(_Codec):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.input_projection = BF16Linear(config.latent_dim, config.codec_dim)
        self.head = BF16Linear(config.codec_dim, config.patch_size * config.vocab_size)

    def forward(self, w: Tensor) -> Tensor:
        """Per-byte logits [B, L, P, V] from anchored latents [B, L, D]."""
        _check_runtime(self, w, self.compiled)
        _check_latents(w, self.config)
        return self._compiled_forward(self, w)

    def _forward(self, w: Tensor) -> Tensor:
        x = self._body(self.input_projection(w))
        logits = self.head(x).float()
        return logits.view(*w.shape[:2], self.config.patch_size, self.config.vocab_size)


def _init_codec(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(module.weight, std=0.02, a=-0.06, b=0.06)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


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
        angles = (1000.0 * times.float()).unsqueeze(-1) * self.frequencies
        x = torch.cat((angles.sin(), angles.cos()), dim=-1).to(torch.bfloat16)
        return self.proj_out(F.silu(self.proj_hid(F.silu(self.proj_in(x)))))


class PriorAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.prior_heads
        self.head_dim = config.prior_dim // config.prior_heads
        self.qkv = BF16Linear(config.prior_dim, 3 * config.prior_dim, bias=False)
        self.out = BF16Linear(config.prior_dim, config.prior_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps)

    def project(self, x: Tensor, cos: Tensor, sin: Tensor):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        return _rotate(self.q_norm(q), cos, sin), _rotate(self.k_norm(k), cos, sin), v

    def output(self, attended: Tensor) -> Tensor:
        return self.out(attended.transpose(1, 2).flatten(2))


class PriorBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.prior_dim
        self.attn_norm = RMSNorm(dim, config.norm_eps, affine=False)
        self.attention = PriorAttention(config)
        self.ffn_norm = RMSNorm(dim, config.norm_eps, affine=False)
        self.ffn_in = BF16Linear(dim, config.prior_mlp_ratio * dim)
        self.ffn_out = BF16Linear(config.prior_mlp_ratio * dim, dim)
        self.modulation = BF16Linear(dim, 6 * dim)

    def prepare(self, x: Tensor, emb: Tensor, cos: Tensor, sin: Tensor):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(
            F.silu(emb)
        ).chunk(6, dim=-1)
        q, k, v = self.attention.project(
            self.attn_norm(x) * (1 + scale_a) + shift_a, cos, sin
        )
        return q, k, v, gate_a, shift_f, scale_f, gate_f

    def finish(self, x, attended, gate_a, shift_f, scale_f, gate_f) -> Tensor:
        x = x + gate_a * self.attention.output(attended)
        hidden = self.ffn_norm(x) * (1 + scale_f) + shift_f
        return x + gate_f * self.ffn_out(F.gelu(self.ffn_in(hidden), approximate="tanh"))

    def forward(self, x: Tensor, emb: Tensor, cos: Tensor, sin: Tensor, mask: Tensor):
        q, k, v, gate_a, shift_f, scale_f, gate_f = self.prepare(x, emb, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.finish(x, attended, gate_a, shift_f, scale_f, gate_f)

    def cached(self, x, emb, cos, sin, cache: tuple[Tensor, Tensor] | None):
        q, k, v, gate_a, shift_f, scale_f, gate_f = self.prepare(x, emb, cos, sin)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        attended = F.scaled_dot_product_attention(q, k, v)
        return self.finish(x, attended, gate_a, shift_f, scale_f, gate_f), (k, v)


class FlowPrior(nn.Module):
    """Block-causal velocity field conditioned on anchored history latents."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_projection = BF16Linear(config.latent_dim, config.prior_dim)
        self.time_embedding = TimestepEmbedding(config.prior_dim)
        self.blocks = nn.ModuleList(PriorBlock(config) for _ in range(config.prior_layers))
        self.final_norm = RMSNorm(config.prior_dim, config.norm_eps, affine=False)
        self.final_modulation = BF16Linear(config.prior_dim, 2 * config.prior_dim)
        self.head = BF16Linear(config.prior_dim, config.latent_dim)
        cos, sin = _rotary_tables(config.seq_patches, config.prior_rope_dim, config.rope_theta)
        self.register_buffer("rotary_cos", cos, persistent=False)
        self.register_buffer("rotary_sin", sin, persistent=False)
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
        self._compiled_forward = torch.compile(
            type(self)._forward, fullgraph=True, dynamic=False
        )
        self._compiled_append = torch.compile(
            type(self)._append, fullgraph=True, dynamic=True
        )
        self._compiled_velocity = torch.compile(
            type(self)._velocity, fullgraph=True, dynamic=True
        )

    @staticmethod
    def _check_times(times: Tensor, z: Tensor) -> None:
        if (
            times.shape != z.shape[:2]
            or times.device != z.device
            or not times.is_floating_point()
        ):
            raise ValueError("times must be floating [B, L] on the latent device")

    def forward(self, history: Tensor, noisy: Tensor, times: Tensor) -> Tensor:
        """Velocity dz/dt for every block at once, each conditioned on its history.

        ``history`` is the anchored latent sequence (decode-time noise level) and
        stays differentiable: the encoder learns through it. ``noisy`` is the
        flow-path state of every block at ``times``.
        """
        _check_runtime(self, noisy, self.compiled)
        _check_latents(noisy, self.config)
        if history.shape != noisy.shape or noisy.shape[1] != self.config.seq_patches:
            raise ValueError("two-stream prior requires history/noisy [B, seq_patches, D]")
        if history.device != noisy.device or not history.is_floating_point():
            raise ValueError("history and noisy must share device and be floating")
        self._check_times(times, noisy)
        return self._compiled_forward(self, history, noisy, times)

    def _output(self, x: Tensor, emb: Tensor) -> Tensor:
        shift, scale = self.final_modulation(F.silu(emb)).chunk(2, dim=-1)
        return self.head(self.final_norm(x) * (1 + scale) + shift).float()

    def _forward(self, history: Tensor, noisy: Tensor, times: Tensor) -> Tensor:
        length = noisy.shape[1]
        x = self.input_projection(torch.cat((history, noisy), dim=1))
        emb = self.time_embedding(
            torch.cat((torch.full_like(times, self.config.decode_time), times), dim=1)
        )
        cos = torch.cat((self.rotary_cos[:length], self.rotary_cos[:length]), dim=0)
        sin = torch.cat((self.rotary_sin[:length], self.rotary_sin[:length]), dim=0)
        mask = two_stream_mask(length, self.config.block_patches, noisy.device)
        for block in self.blocks:
            x = block(x, emb, cos, sin, mask)
        return self._output(x[:, length:], emb[:, length:])

    def _check_block(self, z: Tensor, cache: CleanCache | None) -> None:
        _check_runtime(self, z, self.compiled)
        _check_latents(z, self.config)
        if z.shape[1] != self.config.block_patches:
            raise ValueError("cached prior calls require exactly one block")
        if cache is None:
            return
        if not isinstance(cache, tuple) or len(cache) != self.config.prior_layers:
            raise ValueError("history cache must be a per-layer (key, value) tuple")
        prefix = cache[0][0].shape[2]
        if (
            prefix < self.config.block_patches
            or prefix % self.config.block_patches
            or prefix + z.shape[1] > self.config.seq_patches
        ):
            raise ValueError("cache prefix must be complete blocks fitting the context")
        expected = (
            z.shape[0],
            self.config.prior_heads,
            prefix,
            self.config.prior_dim // self.config.prior_heads,
        )
        for pair in cache:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("each cache layer must be a (key, value) tuple")
            if any(
                t.shape != expected or t.device != z.device or t.dtype != torch.bfloat16
                for t in pair
            ):
                raise ValueError("cache tensors must be BF16 [B, H, prefix, head_dim]")

    def append_history(self, w_block: Tensor, cache: CleanCache | None = None) -> CleanCache:
        """Return a new detached history cache extended by one anchored block."""
        self._check_block(w_block, cache)
        with torch.no_grad():
            return self._compiled_append(self, w_block, cache)

    def block_velocity(
        self, noisy_block: Tensor, times: Tensor, cache: CleanCache | None = None
    ) -> Tensor:
        self._check_block(noisy_block, cache)
        self._check_times(times, noisy_block)
        return self._compiled_velocity(self, noisy_block, times, cache)

    def _append(self, w: Tensor, cache: CleanCache | None) -> CleanCache:
        prefix = 0 if cache is None else cache[0][0].shape[2]
        cos = self.rotary_cos[prefix : prefix + w.shape[1]]
        sin = self.rotary_sin[prefix : prefix + w.shape[1]]
        x = self.input_projection(w.detach())
        emb = self.time_embedding(
            torch.full(w.shape[:2], self.config.decode_time, device=w.device)
        )
        new_cache = []
        for index, block in enumerate(self.blocks):
            x, pair = block.cached(x, emb, cos, sin, None if cache is None else cache[index])
            new_cache.append((pair[0].detach(), pair[1].detach()))
        return tuple(new_cache)

    def _velocity(self, z: Tensor, times: Tensor, cache: CleanCache | None) -> Tensor:
        prefix = 0 if cache is None else cache[0][0].shape[2]
        cos = self.rotary_cos[prefix : prefix + z.shape[1]]
        sin = self.rotary_sin[prefix : prefix + z.shape[1]]
        x = self.input_projection(z)
        emb = self.time_embedding(times)
        for index, block in enumerate(self.blocks):
            x, _ = block.cached(x, emb, cos, sin, None if cache is None else cache[index])
        return self._output(x, emb)


class CelfModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.encoder.apply(_init_codec)
        self.decoder.apply(_init_codec)
        # Uniform reconstruction at initialization; the latent projection stays
        # informative so cross-entropy can anchor from the first update.
        nn.init.zeros_(self.decoder.head.weight)
        nn.init.zeros_(self.decoder.head.bias)
        self.prior = FlowPrior(config)

    @property
    def compiled(self) -> bool:
        return self.encoder.compiled and self.decoder.compiled and self.prior.compiled

    def compile_components(self) -> None:
        self.encoder.compile_component()
        self.decoder.compile_component()
        self.prior.compile_component()

    def anchor(self, z: Tensor, noise: Tensor) -> Tensor:
        """Flow-path state at the anchoring time: the latent every consumer reads."""
        t = self.config.decode_time
        return (1.0 - t) * z + t * noise

    def muon_parameters(self) -> list[nn.Parameter]:
        parameters = []
        for stack in (self.encoder.blocks, self.decoder.blocks):
            for block in stack:
                parameters.extend(
                    layer.weight
                    for layer in (
                        block.attention.qkv,
                        block.attention.out,
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

"""Legacy approximate-rollout diagnostics and exact full-history KV-cache kernels.

The learned gate below remains only for reproducing approximate-rollout metrics.
Public dynamics sampling uses target-verified speculative_generation.generate;
it does not commit these unverified latent predictions.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from pretraining.nanogpt_mini.nanogpt_mini_dynamics_model import DynamicsGPT


@dataclass
class DynamicsGenerationState:
    max_characters: int
    previous_code: Tensor
    input_features: Tensor
    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]
    candidate_age: Tensor
    _model: DynamicsGPT = field(repr=False)
    position: int = 0
    backbone_calls: int = 0
    backbone_positions: int = 0
    dynamics_steps: int = 0
    age: int = 0
    context: Tensor | None = None
    last_predicted_kl: float | None = None
    refresh_positions: list[int] = field(default_factory=list)
    _pending: bool = field(default=False, repr=False)
    _failed: bool = field(default=False, repr=False)


def _nonnegative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _model_device(model: DynamicsGPT) -> torch.device:
    if not isinstance(model, DynamicsGPT):
        raise TypeError("dynamics generation requires a DynamicsGPT")
    if model.training:
        raise ValueError("dynamics generation requires model.eval()")
    device = model.prior.input.weight.device
    if device.type != "cuda":
        raise ValueError("dynamics generation requires CUDA; no CPU fallback")
    if any(tensor.device != device for tensor in model.parameters()):
        raise ValueError("all model parameters must be on the same CUDA device")
    if any(tensor.device != device for tensor in model.buffers()):
        raise ValueError("all model buffers must be on the same CUDA device")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise ValueError("dynamics generation requires BF16-capable CUDA")
    return device


@torch.inference_mode()
def initial_state(model: DynamicsGPT, max_characters: int) -> DynamicsGenerationState:
    """Allocate owned buffers for the TOTAL stream, not a refresh interval."""
    _nonnegative_integer(max_characters, "max_characters")
    device = _model_device(model)
    keys, values = [], []
    for block in model.prior.blocks:
        shape = (1, block.attn.num_heads, max_characters, block.attn.head_dim)
        keys.append(torch.empty(shape, dtype=torch.bfloat16, device=device))
        values.append(torch.empty(shape, dtype=torch.bfloat16, device=device))
    features = torch.empty(
        (1, max_characters, model.config.code_bits),
        dtype=torch.bfloat16,
        device=device,
    )
    # Dense teacher BOS is a zero feature, not the signed code for address zero.
    if max_characters:
        features[:, 0].zero_()
    return DynamicsGenerationState(
        max_characters=max_characters,
        previous_code=torch.zeros(
            (1, model.config.code_bits), dtype=torch.float32, device=device
        ),
        input_features=features,
        keys=tuple(keys),
        values=tuple(values),
        candidate_age=torch.empty((1,), dtype=torch.float32, device=device),
        _model=model,
    )


def _rotary_at(x: Tensor, angular_freq: Tensor, positions: Tensor) -> Tensor:
    theta = torch.outer(positions.float(), angular_freq)[None, :, None, :]
    cos, sin = theta.cos(), theta.sin()
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat(
        (first * cos + second * sin, first * (-sin) + second * cos), dim=-1
    ).type_as(x)


@torch.compile(dynamic=True, fullgraph=True)
def _catch_up(
    model: DynamicsGPT,
    features: Tensor,
    keys,
    values,
    start: int,
    return_all: bool = False,
) -> Tensor:
    """Append a raw-position suffix once; return its last context or all contexts."""
    stack = model.prior
    count = features.shape[1]
    end = start + count
    positions = torch.arange(count, device=features.device) + start
    # SDPA's rectangular is_causal mask aligns at the upper left. With a cached
    # prefix that would hide most legitimate history. Use absolute positions for
    # multi-query suffixes; one query can see every initialized key directly.
    mask = (
        positions[:, None] >= torch.arange(end, device=features.device)[None, :]
        if count > 1
        else None
    )
    x = stack.norm1(stack.input(features))
    for block, key_cache, value_cache in zip(stack.blocks, keys, values):
        attention = block.attn
        normalized = block.norm1(x)
        shape = (1, count, attention.num_heads, attention.head_dim)
        q = F.rms_norm(attention.q(normalized).view(shape), (attention.head_dim,))
        k = F.rms_norm(attention.k(normalized).view(shape), (attention.head_dim,))
        v = attention.v(normalized).view(shape)
        q = _rotary_at(q, attention.rotary.angular_freq, positions)
        k = _rotary_at(k, attention.rotary.angular_freq, positions)
        key_cache[:, :, start:end].copy_(k.transpose(1, 2))
        value_cache[:, :, start:end].copy_(v.transpose(1, 2))
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            key_cache[:, :, :end],
            value_cache[:, :, :end],
            attn_mask=mask,
            scale=0.12,
            is_causal=False,
        ).transpose(1, 2)
        x = x + attention.proj(attended.contiguous().view(1, count, -1))
        x = x + block.mlp(block.norm2(x))
    return stack.norm2(x) if return_all else stack.norm2(x[:, -1])


@torch.compile(dynamic=True, fullgraph=True)
def _dynamics_step(
    model: DynamicsGPT, context: Tensor, previous_code: Tensor, age: Tensor
) -> tuple[Tensor, Tensor]:
    return model.advance(context, previous_code, age)


def _check_state(model: DynamicsGPT, state: DynamicsGenerationState) -> None:
    if not isinstance(state, DynamicsGenerationState) or state._model is not model:
        raise ValueError("generation state belongs to a different model")
    if model.training:
        raise ValueError("dynamics generation requires model.eval()")
    if model.prior.input.weight.device != state.previous_code.device:
        raise ValueError("model moved devices after cache allocation")
    if state._failed:
        raise RuntimeError("generation state is unusable after a failed neural step")


@torch.inference_mode()
def next_context(
    model: DynamicsGPT,
    state: DynamicsGenerationState,
    *,
    force_refresh: bool = False,
) -> tuple[Tensor, bool]:
    """Choose the next context, then require accept_code before advancing again.

    force_refresh is an explicit backbone-only cost diagnostic, not a policy or
    checkpoint setting. It bypasses unused dynamics and always performs exact
    catch-up. Normal generation never enables it.
    """
    _check_state(model, state)
    if not isinstance(force_refresh, bool):
        raise TypeError("force_refresh must be a boolean")
    if state._pending:
        raise RuntimeError(
            "accept the pending character before requesting another context"
        )
    if state.position >= state.max_characters:
        raise ValueError("total character cache capacity exhausted")
    state._pending = True
    try:
        refresh = state.position == 0 or force_refresh
        state.last_predicted_kl = None
        if not refresh:
            if state.context is None:
                raise RuntimeError("missing BOS backbone state")
            state.candidate_age.fill_(state.age + 1)
            candidate, predicted_kl = _dynamics_step(
                model, state.context, state.previous_code, state.candidate_age
            )
            state.dynamics_steps += 1
            state.last_predicted_kl = float(predicted_kl.item())
            refresh = state.last_predicted_kl >= model.config.refresh_cost
        if refresh:
            end = state.position + 1
            context = _catch_up(
                model,
                state.input_features[:, state.backbone_positions : end],
                state.keys,
                state.values,
                state.backbone_positions,
            )
            state.backbone_calls += 1
            state.backbone_positions = end
            state.refresh_positions.append(state.position)
            state.age = 0
        else:
            context = candidate
            state.age += 1
        state.context = context
        # Publicly returned contexts cannot mutate the next transition's state.
        return context.clone(), refresh
    except Exception:
        # A failing neural call can have partially written a layer's K/V cache.
        state._failed = True
        raise


@torch.compile(dynamic=True, fullgraph=True)
def _validated_address(code: Tensor, shifts: Tensor) -> Tensor:
    binary = ((code == 0) | (code == 1)).all()
    address = torch.bitwise_left_shift(code.to(torch.int64), shifts).sum()
    return torch.where(binary, address, -1)


def _accept_validated_code(state: DynamicsGenerationState, code: Tensor) -> None:
    try:
        state.previous_code.copy_(code)
        if state.position + 1 < state.max_characters:
            state.input_features[:, state.position + 1].copy_(2 * code - 1)
        state.position += 1
        state._pending = False
    except Exception:
        state._failed = True
        raise


@torch.inference_mode()
def accept_code(
    model: DynamicsGPT, state: DynamicsGenerationState, code: Tensor
) -> None:
    """Commit one valid alphabet address without retaining caller-owned storage."""
    _check_state(model, state)
    if not state._pending:
        raise RuntimeError("request a context before accepting a character")
    if state.position >= state.max_characters:
        raise ValueError("total character cache capacity exhausted")
    if not isinstance(code, Tensor) or code.shape != state.previous_code.shape:
        raise ValueError(f"code must have shape (1, {model.config.code_bits})")
    if code.device != state.previous_code.device or code.is_complex():
        raise ValueError("code must be a real tensor on the model's CUDA device")
    address = int(_validated_address(code, model.identity_shifts).item())
    if address < 0:
        raise ValueError("code must contain only binary zero/one values")
    if address >= model.config.vocab_size:
        raise ValueError(
            f"reserved code address {address} is outside the checkpoint alphabet"
        )
    _accept_validated_code(state, code)

"""Compiled CUDA character decoding with learned, genuinely sparse global updates.

Cache capacity and the requested output limit bound the entire stream, not a
chunk: only the causal router (apart from BOS) decides when the global Mini runs.
"""

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from pretraining.nanogpt_mini.bit_density import sample_prefix_code
from pretraining.nanogpt_mini.nanogpt_mini_adaptive_model import AdaptiveGPT


@dataclass
class AdaptiveGenerationState:
    max_characters: int
    previous_code: Tensor
    local_keys: tuple[Tensor, ...]
    local_values: tuple[Tensor, ...]
    global_keys: tuple[Tensor, ...]
    global_values: tuple[Tensor, ...]
    _model: AdaptiveGPT = field(repr=False)
    position: int = 0
    global_updates: int = 0
    local_positions: int = 0
    held_global: Tensor | None = None
    last_router_logit: float | None = None
    refresh_positions: list[int] = field(default_factory=list)
    _pending: bool = field(default=False, repr=False)
    _failed: bool = field(default=False, repr=False)


def _nonnegative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _model_device(model: AdaptiveGPT) -> torch.device:
    if not isinstance(model, AdaptiveGPT):
        raise TypeError("adaptive generation requires an AdaptiveGPT")
    if model.training:
        raise ValueError("adaptive generation requires model.eval()")
    device = model.local.input.weight.device
    if device.type != "cuda":
        raise ValueError("adaptive generation requires CUDA; no CPU fallback")
    if any(tensor.device != device for tensor in model.parameters()):
        raise ValueError("all model parameters must be on the same CUDA device")
    if any(tensor.device != device for tensor in model.buffers()):
        raise ValueError("all model buffers must be on the same CUDA device")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise ValueError("adaptive generation requires BF16-capable CUDA")
    return device


def _allocate_caches(stack, capacity: int, device: torch.device):
    keys, values = [], []
    for block in stack.blocks:
        shape = (1, block.attn.num_heads, capacity, block.attn.head_dim)
        keys.append(torch.empty(shape, dtype=torch.bfloat16, device=device))
        values.append(torch.empty(shape, dtype=torch.bfloat16, device=device))
    return tuple(keys), tuple(values)


@torch.inference_mode()
def initial_state(model: AdaptiveGPT, max_characters: int) -> AdaptiveGenerationState:
    """Reserve independent local/global caches for a TOTAL stream capacity.

    Capacity zero is useful for a requested empty stream; it permits no steps.
    Uninitialized cache tails are never read by attention.
    """
    _nonnegative_integer(max_characters, "max_characters")
    device = _model_device(model)
    local_keys, local_values = _allocate_caches(model.local, max_characters, device)
    global_keys, global_values = _allocate_caches(model.prior, max_characters, device)
    return AdaptiveGenerationState(
        max_characters=max_characters,
        previous_code=torch.zeros(
            (1, model.config.code_bits), dtype=torch.float32, device=device
        ),
        local_keys=local_keys,
        local_values=local_values,
        global_keys=global_keys,
        global_values=global_values,
        _model=model,
    )


def _rotary_at(x: Tensor, angular_freq: Tensor, position: int) -> Tensor:
    # The original Mini's half-truncated RoPE, including its FP32 rotation and
    # BF16 rounding. Local positions count characters; global positions events.
    pos = torch.scalar_tensor(position, dtype=torch.float32, device=x.device)
    theta = (pos * angular_freq).view(1, 1, 1, -1)
    cos, sin = theta.cos(), theta.sin()
    x1, x2 = x.float().chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1).type_as(x)


def _cached_stack(
    stack, features: Tensor, keys, values, position: int, initial: Tensor | None = None
) -> Tensor:
    projected = (
        stack.input(features.to(torch.bfloat16).unsqueeze(1))
        if initial is None
        else initial.to(torch.bfloat16)
    )
    x = stack.norm1(projected)
    for block, key_cache, value_cache in zip(stack.blocks, keys, values):
        attention = block.attn
        normalized = block.norm1(x)
        shape = (1, 1, attention.num_heads, attention.head_dim)
        q = attention.q(normalized).view(shape)
        k = attention.k(normalized).view(shape)
        v = attention.v(normalized).view(shape)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = _rotary_at(q, attention.rotary.angular_freq, position)
        k = _rotary_at(k, attention.rotary.angular_freq, position)
        key_cache[:, :, position : position + 1, :].copy_(k.transpose(1, 2))
        value_cache[:, :, position : position + 1, :].copy_(v.transpose(1, 2))
        # The one query is the LAST position of this prefix. is_causal=True
        # would incorrectly give a length-one query access only to key zero.
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            key_cache[:, :, : position + 1, :],
            value_cache[:, :, : position + 1, :],
            scale=0.12,
            is_causal=False,
        ).transpose(1, 2)
        attended = attended.contiguous().view(1, 1, -1)
        x = x + attention.proj(attended)
        x = x + block.mlp(block.norm2(x))
    return stack.norm2(x).squeeze(1)


@torch.compile(dynamic=True, fullgraph=True)
def _local_step(model: AdaptiveGPT, previous_code: Tensor, keys, values, position: int):
    initial = model.local.bos if position == 0 else None
    local = _cached_stack(
        model.local, 2 * previous_code - 1, keys, values, position, initial
    )
    return local, model.router(local).float().reshape(())


@torch.compile(dynamic=True, fullgraph=True)
def _global_step(model: AdaptiveGPT, local: Tensor, keys, values, position: int):
    return _cached_stack(model.prior, local, keys, values, position)


@torch.compile(dynamic=True, fullgraph=True)
def _readout(model: AdaptiveGPT, local: Tensor, held_global: Tensor) -> Tensor:
    return model.readout_norm(held_global + model.local_output(local))


def _check_state(model: AdaptiveGPT, state: AdaptiveGenerationState) -> None:
    if not isinstance(state, AdaptiveGenerationState) or state._model is not model:
        raise ValueError("generation state belongs to a different model")
    if model.training:
        raise ValueError("adaptive generation requires model.eval()")
    if model.local.input.weight.device != state.previous_code.device:
        raise ValueError("model moved devices after cache allocation")
    if state._failed:
        raise RuntimeError("generation state is unusable after a failed neural step")


@torch.inference_mode()
def next_context(
    model: AdaptiveGPT, state: AdaptiveGenerationState
) -> tuple[Tensor, bool]:
    """Advance once before a character, then require accept_code before reuse."""
    _check_state(model, state)
    if state._pending:
        raise RuntimeError(
            "accept the pending character before requesting another context"
        )
    if state.position >= state.max_characters:
        raise ValueError("total character cache capacity exhausted")
    state._pending = True
    try:
        local, router_logit = _local_step(
            model,
            state.previous_code,
            state.local_keys,
            state.local_values,
            state.position,
        )
        state.local_positions += 1
        state.last_router_logit = float(router_logit.item())
        refresh = state.position == 0 or state.last_router_logit >= 0
        if refresh:
            state.held_global = _global_step(
                model,
                local,
                state.global_keys,
                state.global_values,
                state.global_updates,
            )
            state.global_updates += 1
            state.refresh_positions.append(state.position)
        if state.held_global is None:
            raise RuntimeError("missing BOS global state")
        return _readout(model, local, state.held_global), refresh
    except Exception:
        # A failed kernel may already have written some layers of a cache.
        state._failed = True
        raise


@torch.compile(dynamic=True, fullgraph=True)
def _validated_address(code: Tensor, shifts: Tensor) -> Tensor:
    binary = ((code == 0) | (code == 1)).all()
    address = torch.bitwise_left_shift(code.to(torch.int64), shifts).sum()
    return torch.where(binary, address, -1)


def _accept_validated_code(state: AdaptiveGenerationState, code: Tensor) -> None:
    # Copy into owned storage: callers may mutate or reuse their input tensor.
    state.previous_code.copy_(code)
    state.position += 1
    state._pending = False


@torch.inference_mode()
def accept_code(
    model: AdaptiveGPT, state: AdaptiveGenerationState, code: Tensor
) -> None:
    """Commit exactly one valid opaque alphabet address to the pending context."""
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


@torch.inference_mode()
def generate(
    model: AdaptiveGPT,
    prompt_ids: list[int],
    max_new_characters: int,
    temperature: float = 0.0,
    seed: int = 1337,
) -> dict:
    """Generate from the FULL binary domain, stopping explicitly on a reserved code.

    Both prompt and generated positions use the same cached causal router. Timing
    includes prompt ingestion, allocations and first-use compilation, with CUDA
    synchronized at both ends. The limit is total new output, never a chunk cap.
    """
    _nonnegative_integer(max_new_characters, "max_new_characters")
    if not isinstance(prompt_ids, list) or any(
        not isinstance(identity, int)
        or isinstance(identity, bool)
        or not 0 <= identity < model.config.vocab_size
        for identity in prompt_ids
    ):
        raise ValueError(
            "prompt_ids must be a list of valid integer alphabet identities"
        )
    if (
        not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be finite and nonnegative")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    device = _model_device(model)
    generator = torch.Generator(device=device).manual_seed(seed)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    state = initial_state(model, len(prompt_ids) + max_new_characters)
    bits = model.config.code_bits
    # Inputs and independent random uniforms are staged once. Bernoulli
    # probabilities and every autoregressive bit decision run inside the
    # compiled sampler; passing a torch.Generator through Dynamo is unsupported.
    prompt_codes = torch.tensor(
        [
            [(identity >> shift) & 1 for shift in range(bits - 1, -1, -1)]
            for identity in prompt_ids
        ],
        dtype=torch.float32,
        device=device,
    ).reshape(len(prompt_ids), bits)
    uniforms = (
        torch.rand((max_new_characters, bits), device=device, generator=generator)
        if temperature > 0
        else torch.empty((1, bits), device=device)
    )
    for index in range(len(prompt_ids)):
        next_context(model, state)
        _accept_validated_code(state, prompt_codes[index : index + 1])

    generated_ids, generated_codes, generated_addresses = [], [], []
    reserved_code = None
    for index in range(max_new_characters):
        context, _ = next_context(model, state)
        random_row = uniforms[index : index + 1] if temperature > 0 else uniforms
        code = sample_prefix_code(model.head, context, random_row, float(temperature))
        emitted = [int(bit) for bit in code[0].tolist()]
        address = sum(bit << (bits - 1 - offset) for offset, bit in enumerate(emitted))
        generated_codes.append(emitted)
        generated_addresses.append(address)
        if address >= model.config.vocab_size:
            reserved_code = address
            break
        generated_ids.append(address)
        _accept_validated_code(state, code)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "prompt_ids": list(prompt_ids),
        "generated_ids": generated_ids,
        "generated_codes": generated_codes,
        "generated_addresses": generated_addresses,
        "refresh_positions": list(state.refresh_positions),
        "global_updates": state.global_updates,
        "local_positions": state.local_positions,
        "accepted_characters": state.position,
        "reserved_code": reserved_code,
        "stop_reason": "reserved_code" if reserved_code is not None else "output_limit",
        "max_new_characters": max_new_characters,
        "elapsed_seconds": elapsed,
        "characters_per_second": len(generated_ids) / elapsed if elapsed > 0 else 0.0,
        "timing_includes_prompt_and_compilation": True,
        "temperature": float(temperature),
        "seed": seed,
        "rate_label": "code_space",
        "semantics": (
            "Deterministic causal global refresh; full binary address-domain sampling "
            "without alphabet renormalization or resampling. Generated codes/addresses "
            "include an offending reserved address; generated IDs exclude it. Refresh "
            "positions are zero-based raw positions including prompt and any reserved "
            "attempt. Local positions count actual contexts, including that attempt. "
            "There is no character EOS and no fixed or maximum refresh interval."
        ),
    }

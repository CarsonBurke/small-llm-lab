"""Faithful cached inference for CharacterGPT's full-history character softmax.

Every consumed character enters Mini; this is not an emitter-shaped surrogate.
The zero-feature BOS, Mini BF16 blocks, output projection and soft cap match the
original CharacterGPT. Only loss/softmax/RoPE arithmetic intentionally uses FP32.
"""

import math
import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

from pretraining.nanogpt_mini.mini_cached import (
    allocate_cache,
    cached_mini_state,
    sample_character,
)
from pretraining.nanogpt_mini.nanogpt_mini_character_model import CharacterGPT


@dataclass
class CharacterGenerationState:
    max_characters: int
    context: Tensor
    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]
    _model: CharacterGPT = field(repr=False)
    position: int = 0
    backbone_positions: int = 1
    backbone_calls: int = 1
    readout_positions: int = 0
    _failed: bool = field(default=False, repr=False)


def _model_device(model: CharacterGPT) -> torch.device:
    if not isinstance(model, CharacterGPT):
        raise TypeError("character generation requires a CharacterGPT")
    if model.training:
        raise ValueError("character generation requires model.eval()")
    device = model.table.weight.device
    if device.type != "cuda":
        raise ValueError("character generation requires CUDA; no CPU fallback")
    if model.table.weight.dtype != torch.bfloat16:
        raise ValueError("character embeddings must remain BF16")
    if any(value.device != device for value in (*model.parameters(), *model.buffers())):
        raise ValueError("all model tensors must share one CUDA device")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise ValueError("character generation requires BF16-capable CUDA")
    return device


@torch.compile(dynamic=True, fullgraph=True)
def _initial_context(model: CharacterGPT, keys, values) -> Tensor:
    bos = torch.zeros(
        (1, model.config.model_dim),
        dtype=torch.bfloat16,
        device=model.table.weight.device,
    )
    return cached_mini_state(model.prior, bos, keys, values, 0)


@torch.compile(dynamic=True, fullgraph=True)
def _character_step(model: CharacterGPT, identity: Tensor, keys, values, position: int):
    features = model.table(identity.long().reshape(1))
    return cached_mini_state(model.prior, features, keys, values, position)


@torch.compile(dynamic=True, fullgraph=True)
def _readout(model: CharacterGPT, context: Tensor) -> Tensor:
    logits = model.prior.proj(context).float()
    return 15 * logits * (logits.square() + 225).rsqrt()


@torch.inference_mode()
def initial_state(model: CharacterGPT, max_characters: int) -> CharacterGenerationState:
    if type(max_characters) is not int or max_characters < 0:
        raise ValueError("max_characters must be a nonnegative integer")
    device = _model_device(model)
    keys, values = allocate_cache(model.prior, max_characters + 1, device)
    context = _initial_context(model, keys, values)
    return CharacterGenerationState(max_characters, context, keys, values, model)


def _check_state(model: CharacterGPT, state: CharacterGenerationState):
    if not isinstance(state, CharacterGenerationState) or state._model is not model:
        raise ValueError("generation state belongs to a different model")
    if model.training:
        raise ValueError("character generation requires model.eval()")
    if model.table.weight.device != state.context.device:
        raise ValueError("model moved devices after cache allocation")
    if state._failed:
        raise RuntimeError("generation state is unusable after a failed neural step")


@torch.inference_mode()
def next_logits(model: CharacterGPT, state: CharacterGenerationState) -> Tensor:
    _check_state(model, state)
    logits = _readout(model, state.context)
    state.readout_positions += 1
    return logits


def _accept_identity(
    model: CharacterGPT, state: CharacterGenerationState, identity: Tensor
):
    if state.position >= state.max_characters:
        raise ValueError("generation state has exhausted its total character capacity")
    try:
        state.context = _character_step(
            model,
            identity,
            state.keys,
            state.values,
            state.backbone_positions,
        )
        state.position += 1
        state.backbone_positions += 1
        state.backbone_calls += 1
    except Exception:
        state._failed = True
        raise


@torch.inference_mode()
def accept_id(model: CharacterGPT, state: CharacterGenerationState, identity: int):
    _check_state(model, state)
    if type(identity) is not int or not 0 <= identity < model.config.vocab_size:
        raise ValueError("identity must be an integer in the checkpoint alphabet")
    value = torch.tensor([identity], dtype=torch.long, device=state.context.device)
    _accept_identity(model, state, value)


def counters(state: CharacterGenerationState | None) -> dict:
    return {
        "backbone_positions": state.backbone_positions if state else 0,
        "backbone_calls": state.backbone_calls if state else 0,
        "raw_backbone_positions": state.backbone_positions if state else 0,
        "padded_backbone_positions": state.backbone_positions if state else 0,
        "encoder_positions": state.position if state else 0,
        "gate_positions": 0,
        "readout_positions": state.readout_positions if state else 0,
        "accepted_characters": state.position if state else 0,
        "raw_kv_capacity": state.max_characters + 1 if state else 0,
        "raw_kv_bytes": sum(
            value.numel() * value.element_size()
            for value in (*state.keys, *state.values)
        )
        if state
        else 0,
    }


@torch.inference_mode()
def generate(
    model: CharacterGPT,
    prompt_ids: list[int],
    max_new_characters: int,
    temperature: float = 0.0,
    seed: int = 1337,
) -> dict:
    if not isinstance(model, CharacterGPT):
        raise TypeError("character generation requires a CharacterGPT")
    if type(max_new_characters) is not int or max_new_characters < 0:
        raise ValueError("max_new_characters must be a nonnegative integer")
    if not isinstance(prompt_ids, list) or any(
        type(identity) is not int or not 0 <= identity < model.config.vocab_size
        for identity in prompt_ids
    ):
        raise ValueError(
            "prompt_ids must be a list of valid integer alphabet identities"
        )
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be finite and nonnegative")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    generated_ids = []
    state = None
    elapsed = prefill_seconds = decode_seconds = 0.0
    if max_new_characters:
        device = _model_device(model)
        generator = torch.Generator(device=device).manual_seed(seed)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        state = initial_state(model, len(prompt_ids) + max_new_characters - 1)
        prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        uniforms = (
            torch.rand((max_new_characters, 1), device=device, generator=generator)
            if temperature > 0
            else torch.empty((1, 1), device=device)
        )
        for index in range(len(prompt_ids)):
            _accept_identity(model, state, prompt[index : index + 1])
        torch.cuda.synchronize(device)
        decode_started = time.perf_counter()
        prefill_seconds = decode_started - started
        for index in range(max_new_characters):
            logits = next_logits(model, state)
            uniform = uniforms[index : index + 1] if temperature > 0 else uniforms
            identity = sample_character(logits, uniform, float(temperature))
            generated_ids.append(int(identity.item()))
            if index + 1 < max_new_characters:
                _accept_identity(model, state, identity)
        torch.cuda.synchronize(device)
        finished = time.perf_counter()
        decode_seconds = finished - decode_started
        elapsed = finished - started
    return {
        "prompt_ids": list(prompt_ids),
        "generated_ids": generated_ids,
        **counters(state),
        "stop_reason": "output_limit",
        "max_new_characters": max_new_characters,
        "elapsed_seconds": elapsed,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "characters_per_second": len(generated_ids) / elapsed if elapsed > 0 else 0.0,
        "timing_includes_prompt_and_compilation": True,
        "temperature": float(temperature),
        "seed": seed,
        "rate_label": "observed_alphabet_softmax",
        "semantics": (
            "Exact CharacterGPT architecture: zero-feature BOS and every consumed "
            "character enter the full-history Mini KV cache. Raw positions include BOS. "
            "The final output character is not consumed; zero output requests perform "
            "no prefill or neural work. No character EOS or reserved symbols."
        ),
    }

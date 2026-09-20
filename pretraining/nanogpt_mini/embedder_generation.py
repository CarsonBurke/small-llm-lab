"""Compiled BF16 character generation with a persistent accumulator/event KV cache.

Raw character positions never enter the global cache. RoPE positions count BOS
and emitted summaries only. No autocast is used: Mini owns its tensor dtypes.
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
from pretraining.nanogpt_mini.nanogpt_mini_embedder_model import EmbedderGPT


@dataclass
class EmbedderGenerationState:
    max_characters: int
    accumulator: Tensor
    local: Tensor
    held_global: Tensor
    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]
    _model: EmbedderGPT = field(repr=False)
    position: int = 0
    emitted_events: int = 0
    backbone_positions: int = 1
    backbone_calls: int = 1
    encoder_positions: int = 0
    gate_positions: int = 0
    readout_positions: int = 0
    emission_positions: list[int] = field(default_factory=list)
    _failed: bool = field(default=False, repr=False)


def _nonnegative_integer(value: int, name: str):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _model_device(model: EmbedderGPT) -> torch.device:
    if not isinstance(model, EmbedderGPT):
        raise TypeError("embedder generation requires an EmbedderGPT")
    if model.training:
        raise ValueError("embedder generation requires model.eval()")
    if not model._components_compiled:
        raise RuntimeError("call compile_components() before generation")
    device = model.table.weight.device
    if device.type != "cuda":
        raise ValueError("embedder generation requires CUDA; no CPU fallback")
    if model.table.weight.dtype != torch.bfloat16:
        raise ValueError("embedder character table must remain BF16")
    if any(value.device != device for value in (*model.parameters(), *model.buffers())):
        raise ValueError("all model tensors must share one CUDA device")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise ValueError("embedder generation requires BF16-capable CUDA")
    return device


@torch.compile(dynamic=True, fullgraph=True)
def _global_step(
    model: EmbedderGPT, local: Tensor, keys, values, position: int
) -> Tensor:
    return cached_mini_state(
        model.prior, model.prior.input(local), keys, values, position
    )


@torch.compile(dynamic=True, fullgraph=True)
def _initial_local(model: EmbedderGPT) -> tuple[Tensor, Tensor]:
    accumulator = model.encoder.bos.to(torch.bfloat16).clone()
    return accumulator, model.encoder.norm(accumulator)


@torch.compile(dynamic=True, fullgraph=True)
def _character_step(model: EmbedderGPT, identity: Tensor, accumulator: Tensor):
    embedded = model.table(identity.long().reshape(1))
    accumulator, local = model.encoder.step(embedded, accumulator)
    gate_logit = model.gate(local).float().reshape(())
    return accumulator, local, gate_logit


@torch.compile(dynamic=True, fullgraph=True)
def _readout(model: EmbedderGPT, local: Tensor, held: Tensor) -> Tensor:
    return model.readout(local, held)


@torch.inference_mode()
def initial_state(model: EmbedderGPT, max_characters: int) -> EmbedderGenerationState:
    """Allocate a TOTAL-stream capacity, never a forced emission interval.

    Cache storage reserves the all-emission worst case. Only actual events are
    computed/appended. The memory reservation is reported separately from work.
    """
    _nonnegative_integer(max_characters, "max_characters")
    device = _model_device(model)
    keys, values = allocate_cache(model.prior, max_characters + 1, device)
    accumulator, local = _initial_local(model)
    held = _global_step(model, local, keys, values, 0)
    return EmbedderGenerationState(
        max_characters=max_characters,
        accumulator=accumulator,
        local=local,
        held_global=held,
        keys=keys,
        values=values,
        _model=model,
    )


def _check_state(model: EmbedderGPT, state: EmbedderGenerationState):
    if not isinstance(state, EmbedderGenerationState) or state._model is not model:
        raise ValueError("generation state belongs to a different model")
    if model.training:
        raise ValueError("embedder generation requires model.eval()")
    if model.table.weight.device != state.accumulator.device:
        raise ValueError("model moved devices after cache allocation")
    if state._failed:
        raise RuntimeError("generation state is unusable after a failed neural step")


@torch.inference_mode()
def next_logits(model: EmbedderGPT, state: EmbedderGenerationState) -> Tensor:
    """Predict the next unobserved character without consuming it."""
    _check_state(model, state)
    logits = _readout(model, state.local, state.held_global)
    state.readout_positions += 1
    return logits


def _accept_identity(
    model: EmbedderGPT, state: EmbedderGenerationState, identity: Tensor
):
    if state.position >= state.max_characters:
        raise ValueError("generation state has exhausted its total character capacity")
    try:
        accumulator, local, gate_logit = _character_step(
            model, identity, state.accumulator
        )
        emit = (
            (state.position + 1) % model.config.fixed_stride == 0
            if model.config.gate_mode == "fixed"
            else bool((gate_logit >= 0).item())
        )
        if emit:
            state.held_global = _global_step(
                model, local, state.keys, state.values, state.backbone_positions
            )
            state.emitted_events += 1
            state.backbone_positions += 1
            state.backbone_calls += 1
            # Position of the character JUST consumed, not the one it predicts.
            state.emission_positions.append(state.position)
        state.accumulator = accumulator
        state.local = local
        state.encoder_positions += 1
        state.gate_positions += 1
        state.position += 1
        return emit
    except Exception:
        state._failed = True
        raise


@torch.inference_mode()
def accept_id(
    model: EmbedderGPT, state: EmbedderGenerationState, identity: int
) -> bool:
    """Consume c_t, optionally emit its summary, then permit prediction of c_t+1."""
    _check_state(model, state)
    if type(identity) is not int or not 0 <= identity < model.config.vocab_size:
        raise ValueError("identity must be an integer in the checkpoint alphabet")
    value = torch.tensor([identity], dtype=torch.long, device=state.accumulator.device)
    return _accept_identity(model, state, value)


def counters(state: EmbedderGenerationState | None) -> dict:
    """Actual work, including BOS only when a state has really been created."""
    return {
        "emission_positions": list(state.emission_positions) if state else [],
        "emitted_events": state.emitted_events if state else 0,
        "useful_global_positions": state.backbone_positions if state else 0,
        "padded_global_positions": state.backbone_positions if state else 0,
        "backbone_positions": state.backbone_positions if state else 0,
        "backbone_calls": state.backbone_calls if state else 0,
        "encoder_positions": state.encoder_positions if state else 0,
        "gate_positions": state.gate_positions if state else 0,
        "readout_positions": state.readout_positions if state else 0,
        "accepted_characters": state.position if state else 0,
        "event_kv_capacity": state.max_characters + 1 if state else 0,
        "event_kv_bytes": sum(
            value.numel() * value.element_size()
            for value in (*state.keys, *state.values)
        )
        if state
        else 0,
    }


@torch.inference_mode()
def generate(
    model: EmbedderGPT,
    prompt_ids: list[int],
    max_new_characters: int,
    temperature: float = 0.0,
    seed: int = 1337,
) -> dict:
    """Sample only the observed alphabet with the same deterministic causal gate.

    No work consumes the final output character: it has no future prediction.
    Zero requested characters performs no neural work, including no prefill/BOS.
    Timing otherwise includes cache allocation, prompt ingestion and compilation.
    """
    if not isinstance(model, EmbedderGPT):
        raise TypeError("embedder generation requires an EmbedderGPT")
    _nonnegative_integer(max_new_characters, "max_new_characters")
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
            "Emission positions are zero-based consumed-character positions including "
            "the prompt but excluding the final output character. Useful/padded global positions "
            "include BOS; incremental execution has no padding. The global KV cache "
            "contains only emitted summaries, indexed by event RoPE positions. "
            "The persistent finite-width accumulator is lossy; no span cap, periodic "
            "refresh, raw-character global catch-up, reserved symbols or EOS is used."
        ),
    }

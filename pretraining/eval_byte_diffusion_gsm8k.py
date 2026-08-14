"""Few-shot GSM8K for causal and native byte-diffusion checkpoints.

The causal mode is a correctness adapter that recomputes the complete prefix.
Fixed-stride BLT uses the incremental hierarchical cache. Causal-entropy BLT
uses an authenticated patcher and exact uncached variable topology. Duo,
I-DLM, and DiffusionGemma use their native samplers. Every mode reports model
forwards, and GPU execution must be submitted via mlq.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch._dynamo.utils import counters as dynamo_counters


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.data import AtomicIdManifest
from pretraining.byte_diffusion.inference import (
    CachedCanvasGenerator,
    EntropyPatchedCanvasGenerator,
    append_clean_block,
    denoise_blt_cached,
    document_start_ar_metadata,
    entropy_next_byte_starts_patch,
    prefill_prefix,
    prepare_cached_canvas,
)
from pretraining.byte_diffusion.patching import CausalEntropyPatcher
from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.diffusion_gemma import EntropyBudgetSamplerConfig
from pretraining.byte_diffusion.diffusion_gemma_model import DiffusionGemmaModel
from pretraining.byte_diffusion.idlm_model import IDLMModel, IDLMModelConfig
from pretraining.byte_diffusion.inference_diffusion_gemma import (
    generate_diffusion_gemma_continuation,
)
from pretraining.byte_diffusion.inference_duo import generate_duo_continuation
from pretraining.byte_diffusion.inference_idlm import generate_idlm_fused_replay
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.tokenizer import IncrementalByteDecoder
from pretraining.byte_diffusion.sampling import (
    BatchedCanvasSample,
    UnmaskingStrategy,
    remaining_schedule,
    sample_absorbing_canvas_batched,
    unmasking_quota_and_priority,
)
from pretraining.byte_diffusion.state import TransactionalDecodeState
from pretraining.byte_diffusion.training import (
    CHECKPOINT_SCHEMA,
    model_config_from_dict,
)
from pretraining.byte_diffusion.variable_patching import load_dataset_patching_spec
from pretraining.eval_fewshot_gsm8k import (
    GSM8K_ROOT,
    PROMPT_FORMATS,
    TEST_PARQUET,
    TRAIN_PARQUET,
    build_prompt,
    extract_gold,
    extract_prediction,
    select_exemplars,
    sha256,
)
from scripts.train_byte_diffusion import training_source_provenance
from scripts.train_byte_duo import (
    _local_imports,
    source_provenance as duo_source_provenance,
)
from scripts.train_byte_diffusion_gemma import (
    source_provenance as diffusion_gemma_source_provenance,
)
from scripts.train_byte_idlm import source_provenance as idlm_source_provenance


EVALUATOR_SCHEMA = "byte_diffusion_gsm8k/v9"
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ByteGeneration:
    raw: bytes
    text: str | None
    termination: str
    native_actions: int
    model_forwards: int
    invalid_utf8: bool
    generated_atoms: int
    clean_cache_forwards: int = 0
    denoising_forwards: int = 0


def _decode_atomic_continuation(
    atoms: Sequence[int], manifest: AtomicIdManifest
) -> tuple[bytes, str | None, bool, bool]:
    """Strictly validate UTF-8/control boundaries and return literal bytes."""

    decoder = IncrementalByteDecoder(manifest)
    raw = bytearray()
    saw_eot = False
    try:
        for atom in atoms:
            decoder.push(atom)
            if atom == manifest.eot_id:
                saw_eot = True
                break
            if atom < manifest.byte_count:
                raw.append(atom)
        decoder.finish()
        text = bytes(raw).decode("utf-8", errors="strict")
        invalid = False
    except UnicodeDecodeError:
        text = None
        invalid = True
    return bytes(raw), text, invalid, saw_eot


def _pack_byte_prompts(
    prompts: Sequence[bytes], *, width: int, pad_id: int
) -> tuple[Tensor, Tensor]:
    """Pack a ragged byte cohort with one vectorized host allocation."""

    lengths = np.fromiter((len(prompt) for prompt in prompts), dtype=np.int64)
    if lengths.size == 0 or int(lengths.max()) > width:
        raise ValueError("prompt cohort is empty or exceeds its work width")
    offsets = np.empty(lengths.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    flat = np.frombuffer(b"".join(prompts), dtype=np.uint8)
    rows = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
    columns = np.arange(flat.size, dtype=np.int64) - np.repeat(offsets[:-1], lengths)
    packed = np.full((lengths.size, width), pad_id, dtype=np.int64)
    packed[rows, columns] = flat
    return torch.from_numpy(packed), torch.from_numpy(lengths)


def _duo_required_canvases(
    max_new_atoms: int,
    canvas_length: int,
    stride: int,
    commit_width: int | None = None,
    visible_width: int | None = None,
) -> int:
    """Worst-case canvases including a partial UTF-8/patch-aligned first canvas."""

    if min(max_new_atoms, canvas_length, stride) <= 0 or canvas_length % stride:
        raise ValueError("invalid Duo continuation geometry")
    if commit_width is not None and not 0 < commit_width <= canvas_length:
        raise ValueError("invalid Duo commit width")
    if visible_width is not None and not 0 < visible_width <= canvas_length:
        raise ValueError("invalid Duo visible width")
    if commit_width is not None and visible_width is not None and commit_width > visible_width:
        raise ValueError("Duo commit width cannot exceed visible width")
    requested_width = canvas_length if commit_width is None else commit_width
    visible = canvas_length if visible_width is None else visible_width
    worst = 0
    for initial_phase in range(stride):
        phase = initial_phase
        remaining = max_new_atoms
        canvases = 0
        while remaining:
            committed = min(requested_width, visible, canvas_length - phase, remaining)
            if committed <= 0:
                raise AssertionError("Duo canvas budget made no progress")
            remaining -= committed
            phase = (phase + committed) % stride
            canvases += 1
        worst = max(worst, canvases)
    return worst


def _duo_warmup_prompt_groups(
    prompts: Sequence[bytes], batch_size: int
) -> tuple[Sequence[bytes], ...]:
    """Return the full and final partial batch geometries used by evaluation."""

    if not prompts or batch_size <= 0:
        raise ValueError("Duo warmup needs prompts and a positive batch size")
    full = prompts[:batch_size]
    tail_size = len(prompts) % batch_size
    if tail_size and tail_size != len(full):
        return full, prompts[-tail_size:]
    return (full,)


def _dynamo_compile_evidence() -> dict[str, int]:
    return {
        "unique_graphs": int(dynamo_counters["stats"]["unique_graphs"]),
        "recompiles": int(sum(dynamo_counters["recompiles"].values())),
        "graph_breaks": int(sum(dynamo_counters["graph_break"].values())),
    }


def evaluator_source_provenance() -> dict[str, object]:
    """Hash the complete local evaluator/tokenizer/serving closure."""

    pending = {Path(__file__).resolve()}
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        observed.add(path)
        pending.update(item for item in _local_imports(path) if item not in observed)
    relative_paths = tuple(sorted(path.relative_to(REPO_ROOT) for path in observed))
    digest = hashlib.sha256()
    for relative in relative_paths:
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update((REPO_ROOT / relative).read_bytes())
        digest.update(b"\0")
    return {
        "schema": "byte_diffusion_gsm8k_evaluator_source/v1",
        "sha256": digest.hexdigest(),
        "files": tuple(map(str, relative_paths)),
    }


def _trace_steps(trace: dict[str, object]) -> list[dict[str, object]]:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        raise TypeError("trace steps must be a list")
    return steps


@torch.no_grad()
def sample_absorbing_canvas_batched_traced(
    initial_ids: Tensor,
    denoise: Callable[[Tensor], Tensor],
    *,
    steps: int,
    mask_id: int,
    eot_id: int,
    generator: torch.Generator | None,
    strategy: UnmaskingStrategy,
    confidence_threshold: float,
    entropy_budget: float,
    row_active: Tensor,
    stochastic: bool,
) -> tuple[BatchedCanvasSample, list[dict[str, object] | None]]:
    """Run the production reveal rule while retaining every schedule state."""

    if initial_ids.ndim != 2 or initial_ids.dtype != torch.long:
        raise ValueError("batched canvases must be int64 [batch, width]")
    if row_active.shape != initial_ids.shape[:1] or row_active.dtype != torch.bool:
        raise ValueError("row_active must be one boolean per canvas")
    if strategy not in {"confidence", "entropy_bounded", "fixed_quota"}:
        raise ValueError(f"unknown unmasking strategy {strategy!r}")
    if not 0.0 < confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must lie in (0, 1]")
    if entropy_budget < 0.0 or not np.isfinite(entropy_budget):
        raise ValueError("entropy budget must be finite and nonnegative")

    ids = initial_ids.clone()
    unresolved = ids.eq(mask_id) & row_active[:, None]
    if strategy != "fixed_quota" and steps < initial_ids.shape[1]:
        raise ValueError(
            "paper-exact confidence/entropy sampling requires steps >= canvas width"
        )
    active = torch.ones_like(unresolved) & row_active[:, None]
    useful = torch.zeros(
        initial_ids.shape[0], dtype=torch.int64, device=initial_ids.device
    )
    traces: list[dict[str, object] | None] = [
        {
            "initial_ids": initial_ids[row].tolist(),
            "steps": [],
        }
        if bool(row_active[row])
        else None
        for row in range(initial_ids.shape[0])
    ]
    executed = 0
    targets = remaining_schedule(initial_ids.shape[1], steps)[1:]
    for step_index, target in enumerate(targets, start=1):
        live_before = unresolved & active
        live_rows = live_before.any(-1)
        if not bool(live_rows.any()):
            break
        base_quota = (live_before.sum(-1) - target).clamp_min(0)
        if strategy == "fixed_quota" and not bool((base_quota > 0).any()):
            for row, trace in enumerate(traces):
                if trace is not None and bool(live_rows[row]):
                    _trace_steps(trace).append(
                        {
                            "step": step_index,
                            "denoiser_executed": False,
                            "sampler_executed": False,
                            "target_remaining": target,
                            "input_ids": ids[row].tolist(),
                            "output_ids": ids[row].tolist(),
                            "revealed_positions": [],
                        }
                    )
            continue

        input_ids = ids.clone()
        logits = denoise(ids)
        if logits.shape[:2] != ids.shape:
            raise ValueError("batched denoiser logits must align with canvases")
        executed += 1
        from pretraining.byte_diffusion.kernels import (
            categorical_entropy_argmax_confidence,
            categorical_sample_entropy_argmax_confidence,
            reveal_low_entropy,
        )

        if stochastic:
            uniforms = torch.rand(
                ids.shape,
                device=ids.device,
                generator=generator,
                dtype=torch.float32,
            )
            samples, entropy, _, confidence = (
                categorical_sample_entropy_argmax_confidence(logits, uniforms)
            )
        else:
            entropy, samples, confidence = categorical_entropy_argmax_confidence(
                logits
            )
        quota, priority = unmasking_quota_and_priority(
            strategy=strategy,
            entropy=entropy,
            confidence=confidence,
            samples=samples,
            live=live_before,
            fixed_quota=base_quota,
            force_resolve=step_index == steps,
            confidence_threshold=confidence_threshold,
            entropy_budget=entropy_budget,
            eot_id=eot_id,
        )
        ids, unresolved, active, revealed = reveal_low_entropy(
            ids,
            samples,
            priority,
            unresolved,
            active,
            quota,
            eot_id=eot_id,
            allow_simultaneous_eot=(
                strategy == "fixed_quota" and step_index == steps
            ),
        )
        useful += (revealed & live_before).any(-1)
        for row, trace in enumerate(traces):
            if trace is None or not bool(live_rows[row]):
                continue
            _trace_steps(trace).append(
                {
                    "step": step_index,
                    "denoiser_executed": True,
                    "sampler_executed": True,
                    "target_remaining": target,
                    "input_ids": input_ids[row].tolist(),
                    "proposal_ids": samples[row].tolist(),
                    "proposal_entropy_nats": [
                        round(float(value), 6) for value in entropy[row].tolist()
                    ],
                    "proposal_confidence": [
                        round(float(value), 6) for value in confidence[row].tolist()
                    ],
                    "unmasking_strategy": strategy,
                    "revealed_positions": revealed[row].nonzero().flatten().tolist(),
                    "output_ids": ids[row].tolist(),
                    "active_after": active[row].tolist(),
                }
            )
    if bool((unresolved & active).any()):
        raise RuntimeError("absorbing schedule failed to resolve a live canvas")
    return (
        BatchedCanvasSample(
            ids=ids,
            active=active,
            useful_nfe=useful,
            executed_nfe=executed,
        ),
        traces,
    )


def _first_stop(data: bytes, stops: tuple[bytes, ...]) -> int | None:
    offsets = [offset for stop in stops if (offset := data.find(stop)) >= 0]
    return min(offsets) if offsets else None


@torch.no_grad()
def greedy_generate_bytes(
    model: ByteDiffusionModel,
    prompts: list[bytes],
    *,
    max_new_bytes: int,
    max_native_actions: int,
    context_bytes: int,
    stops: tuple[str, ...],
    device: torch.device,
) -> list[ByteGeneration]:
    """Greedily decode literal bytes with virtual-BOS prompt semantics."""

    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("byte prompts must be a nonempty batch of nonempty strings")
    if min(max_new_bytes, max_native_actions, context_bytes) <= 0:
        raise ValueError("generation budgets must be positive")
    prompt_lengths = [len(prompt) for prompt in prompts]
    if max(prompt_lengths) + max_new_bytes > context_bytes:
        raise ValueError("prompt plus byte budget exceeds the trained context")
    batch = len(prompts)
    unpadded_storage_width = max(prompt_lengths) + max_new_bytes
    patch_stride = model.config.patch_stride
    storage_width = (
        (unpadded_storage_width + patch_stride - 1) // patch_stride
    ) * patch_stride
    pad_id = model.config.vocab.pad_id
    buffer = torch.full(
        (batch, storage_width), pad_id, dtype=torch.long, device=device
    )
    for row, prompt in enumerate(prompts):
        buffer[row, : len(prompt)] = torch.tensor(
            list(prompt), dtype=torch.long, device=device
        )
    cursor = torch.tensor(prompt_lengths, dtype=torch.long, device=device)
    positions = torch.arange(storage_width, device=device)[None].expand(batch, -1)
    finished = [False] * batch
    actions = [0] * batch
    generated = [bytearray() for _ in prompts]
    termination = ["native_action_cap" for _ in prompts]
    blocked = torch.tensor(
        list(range(257, model.config.vocab.output_size)),
        dtype=torch.long,
        device=device,
    )
    encoded_stops = tuple(stop.encode("utf-8") for stop in stops)
    executed_forwards = 0

    for _ in range(max_native_actions):
        if all(finished):
            break
        unpadded_span = int(cursor.max())
        span = ((unpadded_span + patch_stride - 1) // patch_stride) * patch_stride
        ids = buffer[:, :span]
        valid = torch.arange(span, device=device)[None] < cursor[:, None]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            packed_logits = model.forward_ar_varlen(
                ids,
                valid,
                positions=positions[:, :span],
                allow_dense_reference=device.type != "cuda",
                return_padded_logits=False,
                **document_start_ar_metadata(valid, patch_stride),
            ).logits
        executed_forwards += 1
        offsets = torch.cat(
            (
                cursor.new_zeros(1),
                cursor.cumsum(0),
            )
        )
        logits = packed_logits[offsets[1:] - 1].float()
        if blocked.numel():
            logits[:, blocked] = -torch.inf
        chosen = logits.argmax(-1)
        for row, token in enumerate(chosen.tolist()):
            if finished[row]:
                continue
            actions[row] += 1
            if token == model.config.vocab.eot_id:
                termination[row] = "eot"
                finished[row] = True
                continue
            if not 0 <= token < 256:
                raise AssertionError(f"unblocked non-byte output id {token}")
            buffer[row, cursor[row]] = token
            cursor[row] += 1
            generated[row].append(token)
            if _first_stop(bytes(generated[row]), encoded_stops) is not None:
                termination[row] = "text_stop"
                finished[row] = True
            elif len(generated[row]) >= max_new_bytes:
                termination[row] = "byte_cap"
                finished[row] = True

    results: list[ByteGeneration] = []
    for row, raw_buffer in enumerate(generated):
        raw = bytes(raw_buffer)
        stop = _first_stop(raw, encoded_stops)
        if stop is not None:
            raw = raw[:stop]
        try:
            text = raw.decode("utf-8", errors="strict")
            invalid = False
        except UnicodeDecodeError:
            text = None
            invalid = True
        results.append(
            ByteGeneration(
                raw=raw,
                text=text,
                termination=termination[row],
                native_actions=actions[row],
                model_forwards=executed_forwards,
                invalid_utf8=invalid,
                generated_atoms=actions[row],
            )
        )
    return results


@torch.no_grad()
def blt_generate_bytes(
    model: ByteDiffusionModel,
    prompts: list[bytes],
    *,
    max_new_bytes: int,
    max_native_actions: int,
    context_bytes: int,
    stops: tuple[str, ...],
    block_length: int,
    diffusion_steps: int,
    unmasking_strategy: UnmaskingStrategy,
    confidence_threshold: float,
    entropy_budget: float,
    seed: int,
    device: torch.device,
) -> list[ByteGeneration]:
    """Decode prompts independently through the exact incremental BLT cache."""

    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("byte prompts must be a nonempty batch of nonempty strings")
    if min(
        max_new_bytes,
        max_native_actions,
        context_bytes,
        block_length,
        diffusion_steps,
    ) <= 0:
        raise ValueError("generation budgets must be positive")
    if block_length % model.config.patch_stride:
        raise ValueError("BLT block length must be patch aligned")
    if max(map(len, prompts)) + max_new_bytes + block_length - 1 > context_bytes:
        raise ValueError(
            "prompt plus byte budget and final BLT block exceed the trained context"
        )
    encoded_stops = tuple(stop.encode("utf-8") for stop in stops)
    blocked = torch.tensor(
        list(range(model.config.vocab.eot_id + 1, model.config.vocab.output_size)),
        dtype=torch.long,
        device=device,
    )
    results: list[ByteGeneration] = []
    for row, prompt in enumerate(prompts):
        state = TransactionalDecodeState(
            eot_id=model.config.vocab.eot_id,
            patch_stride=model.config.patch_stride,
            device=device,
        )
        state.seed(seed + 1_000_003 * row)
        generator = CachedCanvasGenerator(
            model,
            state,
            blocked_output_ids=blocked,
        )
        generator.prefill(torch.tensor(list(prompt), dtype=torch.long, device=device))
        generated = bytearray()
        termination = "native_action_cap"
        while state.counters.forwards < max_native_actions:
            remaining_actions = max_native_actions - state.counters.forwards
            alignment_actions = (
                model.config.patch_stride - state.patch_phase
                if state.patch_phase
                else 0
            )
            # An initially unaligned prompt needs exact AR alignment followed
            # by one cache prefill before the first denoising NFE.
            alignment_overhead = alignment_actions + int(alignment_actions > 0)
            # A successful full block needs one incremental clean
            # encoder/global/decoder replay after the denoising NFEs.
            commit_overhead = 1
            if remaining_actions <= alignment_overhead + commit_overhead:
                break
            generation = generator.generate_blt(
                block_length,
                min(
                    diffusion_steps,
                    remaining_actions - alignment_overhead - commit_overhead,
                ),
                strategy=unmasking_strategy,
                confidence_threshold=confidence_threshold,
                entropy_budget=entropy_budget,
            )
            proposed = torch.cat(
                (generation.alignment_ids, generation.committed_canvas_ids)
            ).tolist()
            saw_eot = False
            for token in proposed:
                if token == model.config.vocab.eot_id:
                    saw_eot = True
                    termination = "eot"
                    break
                if not 0 <= token < 256:
                    raise AssertionError(f"unblocked non-byte output id {token}")
                generated.append(token)
                if _first_stop(bytes(generated), encoded_stops) is not None:
                    termination = "text_stop"
                    break
                if len(generated) >= max_new_bytes:
                    termination = "byte_cap"
                    break
            if termination != "native_action_cap" or saw_eot:
                break
            if not proposed:
                raise RuntimeError("BLT generation made no semantic progress")
        raw = bytes(generated[:max_new_bytes])
        stop = _first_stop(raw, encoded_stops)
        if stop is not None:
            raw = raw[:stop]
        try:
            text = raw.decode("utf-8", errors="strict")
            invalid = False
        except UnicodeDecodeError:
            text = None
            invalid = True
        results.append(
            ByteGeneration(
                raw=raw,
                text=text,
                termination=termination,
                native_actions=state.counters.forwards,
                model_forwards=state.counters.forwards,
                invalid_utf8=invalid,
                generated_atoms=len(generated) + int(saw_eot),
            )
        )
    return results


@torch.no_grad()
def blt_generate_bytes_batched(
    model: ByteDiffusionModel,
    prompts: list[bytes],
    *,
    max_new_bytes: int,
    max_native_actions: int,
    context_bytes: int,
    stops: tuple[str, ...],
    block_length: int,
    diffusion_steps: int,
    unmasking_strategy: UnmaskingStrategy,
    confidence_threshold: float,
    entropy_budget: float,
    seed: int,
    stochastic: bool,
    device: torch.device,
    trace_records: list[dict[str, object]] | None = None,
) -> list[ByteGeneration]:
    """Ragged cached BLT decoding with batched denoising kernels."""

    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("byte prompts must be a nonempty batch of nonempty strings")
    if min(
        max_new_bytes,
        max_native_actions,
        context_bytes,
        block_length,
        diffusion_steps,
    ) <= 0:
        raise ValueError("generation budgets must be positive")
    stride = model.config.patch_stride
    if block_length % stride:
        raise ValueError("BLT block length must be patch aligned")
    if max(map(len, prompts)) > context_bytes:
        raise ValueError("prompt exceeds the trained context")
    if trace_records is not None and len(trace_records) != len(prompts):
        raise ValueError("trace records must align one-to-one with prompts")

    pad_id = model.config.vocab.pad_id
    eot_id = model.config.vocab.eot_id
    encoded_stops = tuple(stop.encode("utf-8") for stop in stops)
    blocked = torch.tensor(
        list(range(eot_id + 1, model.config.vocab.output_size)),
        dtype=torch.long,
        device=device,
    )
    aligned = [bytearray(prompt) for prompt in prompts]
    alignment_generated = [bytearray() for _ in prompts]
    terminated = [False] * len(prompts)
    termination_reason: list[str | None] = [None] * len(prompts)
    logical_actions = [0] * len(prompts)
    actual_forwards = 0

    # At most three batched exact-AR actions close incomplete byte patches.
    for _ in range(stride - 1):
        lengths = torch.tensor(
            [len(prompt) for prompt in aligned], dtype=torch.long, device=device
        )
        needs_alignment = torch.tensor(
            [not done for done in terminated], dtype=torch.bool, device=device
        ) & lengths.remainder(stride).ne(0)
        has_budget = torch.tensor(
            [actions < max_native_actions for actions in logical_actions],
            dtype=torch.bool,
            device=device,
        )
        has_context = lengths.lt(context_bytes)
        for row in (needs_alignment & ~has_budget).nonzero().flatten().tolist():
            terminated[row] = True
            termination_reason[row] = "native_action_cap"
        for row in (
            needs_alignment & has_budget & ~has_context
        ).nonzero().flatten().tolist():
            terminated[row] = True
            termination_reason[row] = "context_cap"
        needs_alignment &= has_budget & has_context
        if not bool(needs_alignment.any()):
            break
        storage_width = int(lengths.max())
        storage_width += (-storage_width) % stride
        ids = torch.full(
            (len(prompts), storage_width),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        for row, prompt in enumerate(aligned):
            ids[row, : len(prompt)] = torch.tensor(
                prompt, dtype=torch.long, device=device
            )
        valid = torch.arange(storage_width, device=device)[None] < lengths[:, None]
        positions = torch.arange(storage_width, device=device)[None].expand_as(ids)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            packed_logits = model.forward_ar_varlen(
                ids,
                valid,
                positions=positions,
                allow_dense_reference=device.type != "cuda",
                return_padded_logits=False,
                **document_start_ar_metadata(valid, stride),
            ).logits
        actual_forwards += 1
        offsets = valid.sum(1).cumsum(0)
        next_logits = packed_logits[offsets - 1].float()
        if blocked.numel():
            next_logits[:, blocked] = -torch.inf
        chosen = next_logits.argmax(-1).tolist()
        for row, token in enumerate(chosen):
            if not bool(needs_alignment[row]):
                continue
            if trace_records is not None:
                alignment_steps = trace_records[row]["alignment_steps"]
                if not isinstance(alignment_steps, list):
                    raise TypeError("trace alignment_steps must be a list")
                alignment_steps.append(
                    {
                        "action": len(alignment_steps) + 1,
                        "prefix_bytes_before": len(aligned[row]),
                        "chosen_id": token,
                    }
                )
            logical_actions[row] += 1
            if token == eot_id:
                terminated[row] = True
                termination_reason[row] = "eot"
            elif 0 <= token < 256:
                aligned[row].append(token)
                alignment_generated[row].append(token)
                if _first_stop(bytes(alignment_generated[row]), encoded_stops) is not None:
                    terminated[row] = True
                    termination_reason[row] = "text_stop"
                elif len(alignment_generated[row]) >= max_new_bytes:
                    terminated[row] = True
                    termination_reason[row] = "byte_cap"
            else:
                raise AssertionError(f"unblocked non-byte output id {token}")

    generated = [bytearray(value) for value in alignment_generated]
    termination = [reason or "native_action_cap" for reason in termination_reason]
    live_rows: list[int] = []
    for row, prompt in enumerate(aligned):
        if not terminated[row]:
            if logical_actions[row] >= max_native_actions:
                terminated[row] = True
                termination[row] = "native_action_cap"
                continue
            if len(prompt) % stride:
                raise AssertionError("live BLT prompt remained patch-unaligned")
            live_rows.append(row)

    for rows in (live_rows,):
        if not rows:
            continue
        cohort_width = max(len(aligned[row]) for row in rows)
        cohort_ids = torch.full(
            (len(rows), cohort_width),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        cohort_lengths = torch.tensor(
            [len(aligned[row]) for row in rows], dtype=torch.long, device=device
        )
        for cohort_row, output_row in enumerate(rows):
            prompt_size = len(aligned[output_row])
            cohort_ids[cohort_row, :prompt_size] = torch.tensor(
                aligned[output_row], dtype=torch.long, device=device
            )
        cohort_valid = (
            torch.arange(cohort_width, device=device)[None]
            < cohort_lengths[:, None]
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            cache = prefill_prefix(
                model,
                cohort_ids,
                valid=cohort_valid,
                allow_dense_reference=device.type != "cuda",
            )
        actual_forwards += 1
        for output_row in rows:
            logical_actions[output_row] += 1
        cohort_actions = torch.tensor(
            [logical_actions[row] for row in rows],
            dtype=torch.int64,
            device=device,
        )
        live = torch.ones(len(rows), dtype=torch.bool, device=device)
        generator = torch.Generator(device=device).manual_seed(
            seed + 1_000_003 * cohort_width
        )
        while bool(live.any()):
            remaining = max_native_actions - cohort_actions
            starts = cache.valid.sum(1).to(torch.long)
            # Each full block needs the configured denoising NFEs plus one
            # clean-cache commit. A lane that cannot afford that work is done,
            # but must not prevent peers with remaining budget from advancing.
            has_budget = remaining.ge(diffusion_steps + 1)
            has_context = (starts + block_length).le(context_bytes)
            context_capped = live & has_budget & ~has_context
            for cohort_row in context_capped.nonzero().flatten().tolist():
                termination[rows[cohort_row]] = "context_cap"
            live &= has_budget & has_context
            if not bool(live.any()):
                break
            steps = diffusion_steps
            starts = starts[:, None]
            plan = prepare_cached_canvas(
                model,
                cache,
                starts,
                block_length,
                allow_dense_reference=device.type != "cuda",
                include_global=False,
            )
            initial = torch.full(
                (len(rows), block_length),
                model.config.vocab.mask_id,
                dtype=torch.long,
                device=device,
            )

            def denoise(canvas: Tensor) -> Tensor:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = denoise_blt_cached(
                        model,
                        cache,
                        canvas[:, None],
                        starts,
                        allow_dense_reference=device.type != "cuda",
                        plan=plan,
                    )
                if blocked.numel():
                    logits[..., blocked] = -torch.inf
                return logits

            if trace_records is None:
                sample = sample_absorbing_canvas_batched(
                    initial,
                    denoise,
                    steps=steps,
                    mask_id=model.config.vocab.mask_id,
                    eot_id=eot_id,
                    generator=generator,
                    strategy=unmasking_strategy,
                    confidence_threshold=confidence_threshold,
                    entropy_budget=entropy_budget,
                    row_active=live,
                    stochastic=stochastic,
                )
                sampled_traces = None
            else:
                sample, sampled_traces = sample_absorbing_canvas_batched_traced(
                    initial,
                    denoise,
                    steps=steps,
                    mask_id=model.config.vocab.mask_id,
                    eot_id=eot_id,
                    generator=generator,
                    strategy=unmasking_strategy,
                    confidence_threshold=confidence_threshold,
                    entropy_budget=entropy_budget,
                    row_active=live,
                    stochastic=stochastic,
                )
            actual_forwards += sample.executed_nfe
            cohort_actions += live.to(cohort_actions.dtype) * sample.executed_nfe
            # EOT deactivates its suffix, which deliberately remains MASK in
            # the semantic sample. Finished lanes still need shape-compatible
            # cache storage while their live cohort peers continue.
            committed = torch.where(
                sample.ids == model.config.vocab.mask_id,
                torch.full_like(sample.ids, eot_id),
                sample.ids,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                cache = append_clean_block(model, cache, committed)
            actual_forwards += 1
            cohort_actions += live.to(cohort_actions.dtype)

            committed_cpu = sample.ids.cpu().numpy()
            live_before_commit = live.clone()
            appended_trace_records: dict[int, dict[str, object]] = {}
            if sampled_traces is not None:
                starts_cpu = starts.squeeze(-1).cpu().tolist()
                for cohort_row, output_row in enumerate(rows):
                    sampled_trace = sampled_traces[cohort_row]
                    if sampled_trace is None or not bool(live[cohort_row]):
                        continue
                    blocks = trace_records[output_row]["diffusion_blocks"]
                    if not isinstance(blocks, list):
                        raise TypeError("trace diffusion_blocks must be a list")
                    sampled_trace.update(
                        {
                            "block": len(blocks),
                            "absolute_byte_start": starts_cpu[cohort_row],
                            "generated_bytes_before": len(generated[output_row]),
                            "sample_ids": sample.ids[cohort_row].tolist(),
                            "cache_committed_ids": committed[cohort_row].tolist(),
                        }
                    )
                    blocks.append(sampled_trace)
                    appended_trace_records[cohort_row] = sampled_trace
            for cohort_row, output_row in enumerate(rows):
                if not bool(live[cohort_row]):
                    continue
                semantic_actions: list[int] = []
                for token in committed_cpu[cohort_row]:
                    token = int(token)
                    semantic_actions.append(token)
                    if token == eot_id:
                        termination[output_row] = "eot"
                        live[cohort_row] = False
                        break
                    if not 0 <= token < 256:
                        raise AssertionError(f"unblocked non-byte output id {token}")
                    generated[output_row].append(token)
                    if _first_stop(
                        bytes(generated[output_row]), encoded_stops
                    ) is not None:
                        termination[output_row] = "text_stop"
                        live[cohort_row] = False
                        break
                    if len(generated[output_row]) >= max_new_bytes:
                        termination[output_row] = "byte_cap"
                        live[cohort_row] = False
                        break
                if cohort_row in appended_trace_records:
                    appended_trace_records[cohort_row].update(
                        {
                            "semantic_actions": semantic_actions,
                            "generated_bytes_after_untrimmed": len(
                                generated[output_row]
                            ),
                            "termination_after_block": (
                                termination[output_row]
                                if bool(live_before_commit[cohort_row])
                                and not bool(live[cohort_row])
                                else None
                            ),
                        }
                    )
            logical_actions_tensor = cohort_actions.cpu().tolist()
            for cohort_row, output_row in enumerate(rows):
                logical_actions[output_row] = logical_actions_tensor[cohort_row]

    results: list[ByteGeneration] = []
    for row, raw_buffer in enumerate(generated):
        raw = bytes(raw_buffer[:max_new_bytes])
        stop = _first_stop(raw, encoded_stops)
        if stop is not None:
            raw = raw[:stop]
        if trace_records is not None:
            blocks = trace_records[row]["diffusion_blocks"]
            if not isinstance(blocks, list):
                raise TypeError("trace diffusion_blocks must be a list")
            if blocks:
                blocks[-1]["final_retained_output_bytes"] = len(raw)
        try:
            text = raw.decode("utf-8", errors="strict")
            invalid = False
        except UnicodeDecodeError:
            text = None
            invalid = True
        results.append(
            ByteGeneration(
                raw=raw,
                text=text,
                termination=termination[row],
                native_actions=logical_actions[row],
                # All rows share this physical work counter; callers take the
                # maximum once, never sum it across batch lanes.
                model_forwards=actual_forwards,
                invalid_utf8=invalid,
                generated_atoms=len(raw_buffer) + int(termination[row] == "eot"),
            )
        )
    return results


@torch.no_grad()
def entropy_blt_generate_bytes(
    model: ByteDiffusionModel,
    patcher: CausalEntropyPatcher,
    prompts: Sequence[bytes],
    *,
    max_new_bytes: int,
    max_native_actions: int,
    context_bytes: int,
    block_length: int,
    stops: tuple[str, ...],
    diffusion_steps: int,
    unmasking_strategy: UnmaskingStrategy,
    confidence_threshold: float,
    entropy_budget: float,
    seed: int,
    stochastic: bool,
    device: torch.device,
    trace_records: list[dict[str, object]] | None = None,
) -> list[ByteGeneration]:
    """Reference causal-entropy serving; each row preserves its own topology."""

    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("byte prompts must be a nonempty batch of nonempty strings")
    if min(
        max_new_bytes,
        max_native_actions,
        context_bytes,
        block_length,
        diffusion_steps,
    ) <= 0:
        raise ValueError("generation budgets must be positive")
    if max(map(len, prompts)) > context_bytes:
        raise ValueError("prompt exceeds the trained context")
    if trace_records is not None and len(trace_records) != len(prompts):
        raise ValueError("trace records must align one-to-one with prompts")

    eot_id = model.config.vocab.eot_id
    encoded_stops = tuple(stop.encode("utf-8") for stop in stops)
    blocked = torch.arange(
        eot_id + 1,
        model.config.vocab.output_size,
        dtype=torch.long,
        device=device,
    )
    results: list[ByteGeneration] = []
    for row, prompt in enumerate(prompts):
        generator = EntropyPatchedCanvasGenerator(
            model,
            patcher,
            block_length=block_length,
            seed=seed + 1_000_003 * row,
            blocked_output_ids=blocked,
        )
        generator.prefill(torch.tensor(list(prompt), dtype=torch.long, device=device))
        generated = bytearray()
        generated_atoms = 0
        termination = "native_action_cap"
        while generator.forwards < max_native_actions:
            # Close an incomplete patch one clean AR action at a time so a
            # byte/text/context cap cannot hide later alignment generations.
            while not entropy_next_byte_starts_patch(patcher, generator.ids):
                if generator.forwards >= max_native_actions:
                    break
                if generator.ids.numel() >= context_bytes:
                    termination = "context_cap"
                    break
                alignment = generator.align_prefix_ar(
                    stochastic=stochastic, max_forwards=1
                )
                if alignment.numel() != 1:
                    raise AssertionError("entropy AR alignment made no progress")
                token = int(alignment[0])
                generated_atoms += 1
                if trace_records is not None:
                    alignment_steps = trace_records[row]["alignment_steps"]
                    if not isinstance(alignment_steps, list):
                        raise TypeError("trace alignment_steps must be a list")
                    alignment_steps.append(
                        {
                            "prefix_bytes_after": generator.ids.numel(),
                            "chosen_id": token,
                        }
                    )
                if token == eot_id:
                    termination = "eot"
                    break
                if not 0 <= token < 256:
                    raise AssertionError(f"unblocked non-byte output id {token}")
                generated.append(token)
                if _first_stop(bytes(generated), encoded_stops) is not None:
                    termination = "text_stop"
                    break
                if len(generated) >= max_new_bytes:
                    termination = "byte_cap"
                    break
            if termination != "native_action_cap":
                break
            if not entropy_next_byte_starts_patch(patcher, generator.ids):
                # The only remaining cause is exhaustion of the native-action
                # budget while a variable patch was still incomplete.
                break
            if generator.forwards + diffusion_steps > max_native_actions:
                break
            if generator.ids.numel() + block_length > context_bytes:
                termination = "context_cap"
                break

            proposal = generator.generate_blt(
                diffusion_steps,
                strategy=unmasking_strategy,
                confidence_threshold=confidence_threshold,
                entropy_budget=entropy_budget,
                stochastic=stochastic,
            )
            if proposal.canvas is None:
                termination = "eot"
                break
            if trace_records is not None:
                blocks = trace_records[row]["diffusion_blocks"]
                if not isinstance(blocks, list):
                    raise TypeError("trace diffusion_blocks must be a list")
                blocks.append(
                    {
                        "block": len(blocks),
                        "prompt_bytes_noised": False,
                        "canvas_width": block_length,
                        "sample_ids": proposal.canvas.ids.tolist(),
                        "committed_ids": proposal.committed_canvas_ids.tolist(),
                        "overflow_ids": proposal.overflow_canvas_ids.tolist(),
                    }
                )
            if not proposal.committed_canvas_ids.numel():
                raise RuntimeError("entropy BLT generation made no semantic progress")
            for token in proposal.committed_canvas_ids.tolist():
                generated_atoms += 1
                if token == eot_id:
                    termination = "eot"
                    break
                if not 0 <= token < 256:
                    raise AssertionError(f"unblocked non-byte output id {token}")
                generated.append(token)
                if _first_stop(bytes(generated), encoded_stops) is not None:
                    termination = "text_stop"
                    break
                if len(generated) >= max_new_bytes:
                    termination = "byte_cap"
                    break
            if termination != "native_action_cap":
                break

        raw = bytes(generated[:max_new_bytes])
        stop = _first_stop(raw, encoded_stops)
        if stop is not None:
            raw = raw[:stop]
        try:
            text = raw.decode("utf-8", errors="strict")
            invalid = False
        except UnicodeDecodeError:
            text = None
            invalid = True
        results.append(
            ByteGeneration(
                raw=raw,
                text=text,
                termination=termination,
                native_actions=generator.forwards,
                model_forwards=generator.forwards,
                invalid_utf8=invalid,
                generated_atoms=generated_atoms,
                clean_cache_forwards=generator.causal_forwards,
                denoising_forwards=generator.denoising_forwards,
            )
        )
    return results


@torch.no_grad()
def duo_generate_bytes_batched(
    model: DuoModel,
    prompts: Sequence[bytes],
    *,
    max_new_bytes: int,
    max_new_atoms: int,
    max_native_actions: int,
    context_bytes: int,
    stops: tuple[str, ...],
    canvas_length: int,
    diffusion_steps: int,
    visible_width: int | None = None,
    commit_width: int | None = None,
    terminal_eps: float = 1e-5,
    use_float64: bool = False,
    seed: int,
    device: torch.device,
    work_width: int | None = None,
    trace_records: list[dict[str, object]] | None = None,
) -> list[ByteGeneration]:
    """Evaluate Byte-Duo from clean prompts; only fresh suffix atoms diffuse."""

    if not prompts:
        return []
    if trace_records is not None and len(trace_records) != len(prompts):
        raise ValueError("Duo trace records must align with prompts")
    host_lengths = np.fromiter((len(prompt) for prompt in prompts), dtype=np.int64)
    required_width = int(host_lengths.max()) + max_new_atoms
    if required_width > context_bytes:
        raise ValueError("Duo prompt plus generation budget exceeds trained context")
    width = (
        ((required_width + model.config.patch_stride - 1) // model.config.patch_stride)
        * model.config.patch_stride
        if work_width is None
        else work_width
    )
    if width < required_width or width > context_bytes:
        raise ValueError("Duo work width must cover the cohort within context")
    if width % model.config.patch_stride:
        raise ValueError("Duo work width must be patch aligned")
    host_ids, host_prompt_lengths = _pack_byte_prompts(
        prompts, width=width, pad_id=model.config.vocab.pad_id
    )
    if device.type == "cuda":
        host_ids = host_ids.pin_memory()
        host_prompt_lengths = host_prompt_lengths.pin_memory()
    ids = host_ids.to(device, non_blocking=device.type == "cuda")
    prompt_lengths = host_prompt_lengths.to(
        device, non_blocking=device.type == "cuda"
    )
    valid = torch.arange(width, device=device)[None] < prompt_lengths[:, None]
    per_canvas_forwards = diffusion_steps + 2  # clean-bank prep + N+1 branch NFEs
    max_canvases = max_native_actions // per_canvas_forwards
    if max_canvases <= 0:
        raise ValueError("native action cap cannot fund one Duo canvas")
    encoded_stops = tuple(stop.encode("utf-8") for stop in stops)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        continuation = generate_duo_continuation(
            model,
            ids,
            valid,
            canvas_length=canvas_length,
            max_new_atoms=max_new_atoms,
            max_new_bytes=max_new_bytes,
            steps=diffusion_steps,
            visible_width=visible_width,
            commit_width=commit_width,
            terminal_eps=terminal_eps,
            max_canvases=max_canvases,
            generator=generator,
            use_float64=use_float64,
            return_trajectories=trace_records is not None,
            stop_sequences=encoded_stops,
        )
    if continuation.model_forwards > max_native_actions:
        raise AssertionError("Duo continuation exceeded its native-action cap")

    if trace_records is not None:
        schedule_eps = float(model.schedule_eps)
        reverse_grid = tuple(
            1.0 + (terminal_eps - 1.0) * index / diffusion_steps
            for index in range(diffusion_steps + 1)
        )
        ledgers = (
            continuation.trajectories,
            continuation.trajectory_rows,
            continuation.trajectory_starts,
            continuation.trajectory_active,
        )
        if any(value is None for value in ledgers):
            raise AssertionError("Duo tracing requested but trajectory ledger is absent")
        trajectories, trajectory_rows, trajectory_starts, trajectory_active = ledgers
        assert trajectories is not None
        assert trajectory_rows is not None
        assert trajectory_starts is not None
        assert trajectory_active is not None
        for canvas_index, (trajectory, rows, starts, active) in enumerate(
            zip(
                trajectories,
                trajectory_rows,
                trajectory_starts,
                trajectory_active,
                strict=True,
            )
        ):
            for cohort_row, output_row in enumerate(rows.tolist()):
                blocks = trace_records[output_row]["diffusion_blocks"]
                if not isinstance(blocks, list):
                    raise TypeError("Duo diffusion_blocks must be a list")
                states = trajectory[:, cohort_row].cpu()
                steps: list[dict[str, object]] = []
                for step in range(1, states.shape[0]):
                    final_cleanup = step == states.shape[0] - 1
                    time_t = reverse_grid[-1] if final_cleanup else reverse_grid[step - 1]
                    time_s = 0.0 if final_cleanup else reverse_grid[step]
                    alpha_t = 1.0 - (1.0 - schedule_eps) * time_t
                    alpha_s = 1.0 if final_cleanup else 1.0 - (1.0 - schedule_eps) * time_s
                    steps.append(
                        {
                            "step": step,
                            "transition_kind": (
                                "exact_residual_noise_cleanup"
                                if final_cleanup
                                else "reverse_grid_posterior"
                            ),
                            "time_t": time_t,
                            "time_s": time_s,
                            "alpha_t": alpha_t,
                            "alpha_s": alpha_s,
                            "input_ids": states[step - 1].tolist(),
                            "output_ids": states[step].tolist(),
                            "all_active_suffix_positions_revisable": True,
                        }
                    )
                blocks.append(
                    {
                        "block": canvas_index,
                        "prompt_atoms_noised": False,
                        "schedule_eps": schedule_eps,
                        "sampling_terminal_eps": terminal_eps,
                        "posterior_precision": "float64" if use_float64 else "float32",
                        "absolute_atom_start": int(starts[cohort_row]),
                        "revisable_positions": active[cohort_row]
                        .nonzero(as_tuple=False)
                        .flatten()
                        .tolist(),
                        "non_revisable_positions": (~active[cohort_row])
                        .nonzero(as_tuple=False)
                        .flatten()
                        .tolist(),
                        "initial_ids": states[0].tolist(),
                        "steps": steps,
                    }
                )

    output: list[ByteGeneration] = []
    generated_mask = continuation.generated_valid.cpu()
    output_ids = continuation.ids.cpu()
    text_stopped = continuation.text_stopped.cpu()
    byte_capped = continuation.byte_capped.cpu()
    denoising_actions = continuation.denoising_actions.cpu()
    clean_bank_actions = continuation.clean_bank_actions.cpu()
    for row in range(len(prompts)):
        atoms = output_ids[row].masked_select(generated_mask[row]).tolist()
        raw, text, invalid, saw_eot = _decode_atomic_continuation(
            atoms, AtomicIdManifest.reference()
        )
        termination = (
            "text_stop"
            if bool(text_stopped[row])
            else "byte_cap"
            if bool(byte_capped[row])
            else "atom_cap"
            if len(atoms) >= max_new_atoms
            else "native_action_cap"
        )
        if saw_eot:
            termination = "eot"
        raw = raw[:max_new_bytes]
        stop = _first_stop(raw, encoded_stops)
        if stop is not None:
            raw = raw[:stop]
        if not invalid:
            text = raw.decode("utf-8", errors="strict")
        output.append(
            ByteGeneration(
                raw=raw,
                text=text,
                termination=termination,
                native_actions=int(denoising_actions[row] + clean_bank_actions[row]),
                model_forwards=continuation.model_forwards,
                invalid_utf8=invalid,
                generated_atoms=len(atoms),
                clean_cache_forwards=int(clean_bank_actions[row]),
                denoising_forwards=int(denoising_actions[row]),
            )
        )
    if trace_records is not None:
        final_lengths = continuation.lengths.cpu()
        for row, trace in enumerate(trace_records):
            blocks = trace["diffusion_blocks"]
            if not isinstance(blocks, list):
                raise TypeError("Duo diffusion_blocks must be a list")
            generated_bytes = 0
            for block_index, block in enumerate(blocks):
                if not isinstance(block, dict):
                    raise TypeError("Duo diffusion block must be an object")
                active = tuple(int(value) for value in block["revisable_positions"])
                final_state = tuple(int(value) for value in block["steps"][-1]["output_ids"])
                start = int(block["absolute_atom_start"])
                if block_index + 1 < len(blocks):
                    next_block = blocks[block_index + 1]
                    if not isinstance(next_block, dict):
                        raise TypeError("Duo diffusion block must be an object")
                    next_active = tuple(
                        int(value) for value in next_block["revisable_positions"]
                    )
                    next_start = int(next_block["absolute_atom_start"])
                    length_after = next_start + min(next_active)
                    transition: dict[str, int] | None = {
                        "absolute_atom_start": next_start,
                        "clean_prefix_atoms_in_canvas": min(next_active),
                    }
                    termination_after_block = "continue"
                else:
                    length_after = int(final_lengths[row])
                    transition = None
                    termination_after_block = output[row].termination
                committed_positions = tuple(
                    position
                    for position in active
                    if start + position < length_after
                )
                committed_ids = tuple(final_state[position] for position in committed_positions)
                committed_bytes = tuple(value for value in committed_ids if value < 256)
                before = generated_bytes
                generated_bytes += len(committed_bytes)
                block.update(
                    {
                        "commit_positions": committed_positions,
                        "committed_ids": committed_ids,
                        "committed_byte_values": committed_bytes,
                        "generated_bytes_before": before,
                        "generated_bytes_after": generated_bytes,
                        "termination_after_block": termination_after_block,
                        "next_canvas_transition": transition,
                    }
                )
            returned = tuple(output[row].raw)
            trace["returned_byte_values_after_stop_trim"] = returned
            trace["returned_byte_count_after_stop_trim"] = len(returned)
    return output


@torch.no_grad()
def idlm_generate_bytes_batched(
    model: IDLMModel,
    prompts: Sequence[bytes],
    *,
    max_new_bytes: int,
    max_native_actions: int,
    context_bytes: int,
    stops: tuple[str, ...],
    seed: int,
    device: torch.device,
) -> list[ByteGeneration]:
    """Run exact fused-ISD continuation from untouched clean prompts."""

    if not prompts:
        return []
    width = max(map(len, prompts))
    if width + max_new_bytes > context_bytes:
        raise ValueError("I-DLM prompt plus generation budget exceeds context")
    host_ids, host_lengths = _pack_byte_prompts(
        prompts, width=width, pad_id=model.config.pad_id
    )
    if device.type == "cuda":
        host_ids = host_ids.pin_memory()
        host_lengths = host_lengths.pin_memory()
    ids = host_ids.to(device, non_blocking=device.type == "cuda")
    lengths = host_lengths.to(device, non_blocking=device.type == "cuda")
    valid = torch.arange(width, device=device)[None] < lengths[:, None]
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        continuation = generate_idlm_fused_replay(
            model,
            ids,
            valid,
            stride=model.config.block_size,
            max_new_tokens=max_new_bytes,
            generator=generator,
        )
    if continuation.model_forwards > max_native_actions:
        raise RuntimeError("I-DLM continuation exceeded the native-action cap")
    encoded_stops = tuple(stop.encode("utf-8") for stop in stops)
    generated = continuation.generated_valid.cpu()
    output_ids = continuation.ids.cpu()
    results: list[ByteGeneration] = []
    for row in range(len(prompts)):
        atoms = output_ids[row].masked_select(generated[row]).tolist()
        raw, text, invalid, saw_eot = _decode_atomic_continuation(
            atoms, AtomicIdManifest.reference()
        )
        stop = _first_stop(raw, encoded_stops)
        if stop is not None:
            raw = raw[:stop]
        if not invalid:
            text = raw.decode("utf-8", errors="strict")
        results.append(
            ByteGeneration(
                raw=raw,
                text=text,
                termination=(
                    "eot" if saw_eot else "text_stop" if stop is not None else "byte_cap"
                ),
                native_actions=continuation.iterations,
                model_forwards=continuation.model_forwards,
                invalid_utf8=invalid,
                generated_atoms=len(atoms),
                denoising_forwards=continuation.model_forwards,
            )
        )
    return results


@torch.no_grad()
def diffusion_gemma_generate_bytes_batched(
    model: DiffusionGemmaModel,
    prompts: Sequence[bytes],
    *,
    max_new_bytes: int,
    max_new_atoms: int,
    max_native_actions: int,
    context_bytes: int,
    stops: tuple[str, ...],
    canvas_length: int,
    diffusion_steps: int,
    seed: int,
    device: torch.device,
    trace_records: list[dict[str, object]] | None = None,
) -> list[ByteGeneration]:
    """Generate revisable canvases while keeping every prompt atom clean."""

    if not prompts:
        return []
    required = max(map(len, prompts)) + max_new_atoms
    if required > context_bytes:
        raise ValueError("DiffusionGemma prompt plus generation budget exceeds context")
    width = max(map(len, prompts))
    host_ids, host_lengths = _pack_byte_prompts(
        prompts, width=width, pad_id=model.config.vocab.pad_id
    )
    if device.type == "cuda":
        host_ids = host_ids.pin_memory()
        host_lengths = host_lengths.pin_memory()
    ids = host_ids.to(device, non_blocking=device.type == "cuda")
    lengths = host_lengths.to(device, non_blocking=device.type == "cuda")
    valid = torch.arange(width, device=device)[None] < lengths[:, None]
    sampler = EntropyBudgetSamplerConfig(max_steps=diffusion_steps)
    max_canvases = max(1, (max_native_actions - 1) // (diffusion_steps + 1))
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        continuation = generate_diffusion_gemma_continuation(
            model,
            ids,
            valid,
            canvas_length=canvas_length,
            max_new_tokens=max_new_atoms,
            sampler_config=sampler,
            max_canvases=max_canvases,
            generator=generator,
            trace_records=trace_records,
        )
    if continuation.model_forwards > max_native_actions:
        raise RuntimeError("DiffusionGemma continuation exceeded native-action cap")
    encoded_stops = tuple(stop.encode("utf-8") for stop in stops)
    generated = continuation.generated_valid.cpu()
    output_ids = continuation.ids.cpu()
    results: list[ByteGeneration] = []
    for row in range(len(prompts)):
        atoms = output_ids[row].masked_select(generated[row]).tolist()
        raw, text, invalid, saw_eot = _decode_atomic_continuation(
            atoms, AtomicIdManifest.reference()
        )
        raw = raw[:max_new_bytes]
        stop = _first_stop(raw, encoded_stops)
        if stop is not None:
            raw = raw[:stop]
        if not invalid:
            text = raw.decode("utf-8", errors="strict")
        results.append(
            ByteGeneration(
                raw=raw,
                text=text,
                termination=(
                    "eot"
                    if saw_eot
                    else "text_stop"
                    if stop is not None
                    else "atom_cap"
                ),
                native_actions=continuation.model_forwards,
                model_forwards=continuation.model_forwards,
                invalid_utf8=invalid,
                generated_atoms=len(atoms),
                clean_cache_forwards=(
                    continuation.clean_prefill_forwards
                    + continuation.clean_commit_forwards
                ),
                denoising_forwards=continuation.denoising_forwards,
            )
        )
    return results


def load_byte_checkpoint(
    path: Path,
    device: torch.device,
    *,
    decode_mode: str,
    allow_compatible_duo_source: bool = False,
) -> tuple[ByteDiffusionModel | DuoModel | IDLMModel | DiffusionGemmaModel, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if decode_mode == "idlm":
        if payload.get("schema") != "byte_idlm_training/v4":
            raise ValueError("I-DLM GSM8K requires an I-DLM checkpoint")
        extra = payload.get("extra")
        if not isinstance(extra, Mapping) or extra.get("source_sha256") != (
            idlm_source_provenance()["sha256"]
        ):
            raise ValueError("I-DLM GSM8K code differs from checkpoint provenance")
        config = IDLMModelConfig(**payload["model_config"])
        model = IDLMModel(config)
        model.load_state_dict(payload["model"], strict=True)
        return model.to(device).eval(), payload
    if decode_mode == "diffusion_gemma":
        if payload.get("schema") != "byte_diffusion_gemma_checkpoint/v3":
            raise ValueError(
                "DiffusionGemma GSM8K requires a DiffusionGemma checkpoint"
            )
        if payload.get("source_sha256") != diffusion_gemma_source_provenance()[
            "sha256"
        ]:
            raise ValueError(
                "DiffusionGemma GSM8K code differs from checkpoint provenance"
            )
        config = model_config_from_dict(payload["model_config"])
        model = DiffusionGemmaModel(config)
        model.load_state_dict(payload["model"], strict=True)
        return model.to(device).eval(), payload
    if decode_mode == "duo":
        if payload.get("schema") != "byte_duo_checkpoint/v2":
            raise ValueError("Duo GSM8K requires a Byte-Duo checkpoint")
        current_source = duo_source_provenance()
        if (
            payload.get("source_sha256") != current_source["sha256"]
            and not allow_compatible_duo_source
        ):
            raise ValueError("Duo GSM8K code differs from checkpoint provenance")
        config = model_config_from_dict(payload["model_config"])
        schedule_eps = float(payload["training"]["schedule_eps"])
        model = DuoModel(config, schedule_eps=schedule_eps)
        model.load_state_dict(payload["model"], strict=True)
        return model.to(device).eval(), payload
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("GSM8K requires a byte-diffusion training checkpoint")
    current_source = training_source_provenance()
    if payload.get("source_provenance") != current_source:
        raise ValueError(
            "GSM8K code differs from the checkpoint's pinned training source"
        )
    recipe = payload.get("run_contract", {}).get("recipe")
    if recipe not in {"causal_only", "blt_d"}:
        raise ValueError("GSM8K supports causal-only and BLT-D checkpoints")
    if decode_mode == "blt" and recipe != "blt_d":
        raise ValueError("BLT decoding requires a BLT-D checkpoint")
    manifest = AtomicIdManifest.from_dict(payload["atomic_manifest"])
    if manifest.sha256 != AtomicIdManifest.reference().sha256:
        raise ValueError("checkpoint atomic vocabulary differs from the evaluator")
    model = ByteDiffusionModel(model_config_from_dict(payload["model_config"]))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload


def load_entropy_patcher_for_checkpoint(
    payload: Mapping[str, object], dataset_path: Path | None
) -> CausalEntropyPatcher | None:
    """Load a patcher only through the checkpoint-pinned dataset manifest."""

    run_contract = payload.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise ValueError("byte checkpoint omitted its run contract")
    policy_name = run_contract.get("patching_policy")
    if policy_name == "fixed_stride_v1":
        if dataset_path is not None:
            raise ValueError("--dataset-path is only valid for entropy patching")
        return None
    if policy_name != "causal_entropy_v1":
        raise ValueError(f"unknown checkpoint patching policy {policy_name!r}")
    if dataset_path is None:
        raise ValueError(
            "entropy-patched evaluation requires --dataset-path for its "
            "authenticated patcher artifact"
        )
    manifest_path = dataset_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"entropy dataset manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, Mapping):
        raise ValueError("entropy dataset manifest must be an object")
    claimed_payload = manifest.get("payload_sha256")
    unsigned = dict(manifest)
    unsigned.pop("payload_sha256", None)
    observed_payload = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if claimed_payload != observed_payload:
        raise ValueError("entropy dataset manifest payload_sha256 mismatch")
    provenance = payload.get("dataset_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("entropy checkpoint omitted dataset provenance")
    if provenance.get("payload_sha256") != observed_payload:
        raise ValueError(
            "entropy dataset differs from the checkpoint-pinned training dataset"
        )
    spec = load_dataset_patching_spec(dataset_path, manifest)
    if spec.name != "causal_entropy_v1" or not spec.variable:
        raise ValueError("checkpoint dataset does not use causal entropy patching")
    patching = manifest.get("patching")
    if not isinstance(patching, Mapping):
        raise ValueError("entropy manifest omitted its patching contract")
    artifact = patching.get("patcher_artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("entropy manifest omitted its patcher artifact")
    relative = Path(str(artifact.get("path", "")))
    patcher = CausalEntropyPatcher.from_bytes((dataset_path / relative).read_bytes())
    if patcher.sha256 != spec.artifact_sha256:
        raise ValueError("loaded entropy patcher differs from authenticated artifact")
    return patcher


def validate_serving_contract(
    payload: Mapping[str, object],
    *,
    decode_mode: str,
    block_length: int,
    entropy_patcher: CausalEntropyPatcher | None = None,
) -> None:
    """Reject evaluator geometry that differs from the trained model."""

    if decode_mode == "duo":
        training = payload.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("Duo checkpoint omitted its training contract")
        if block_length != int(training["canvas_length"]):
            raise ValueError("Duo eval canvas must equal the checkpoint's trained canvas")
        return
    if decode_mode == "idlm":
        config = payload.get("model_config")
        if not isinstance(config, Mapping):
            raise ValueError("I-DLM checkpoint omitted its model configuration")
        if block_length != int(config["block_size"]):
            raise ValueError("I-DLM eval stride must equal its trained stride")
        return
    if decode_mode == "diffusion_gemma":
        training = payload.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("DiffusionGemma checkpoint omitted its training contract")
        if block_length != int(training["canvas_length"]):
            raise ValueError(
                "DiffusionGemma eval canvas must equal its trained canvas"
            )
        return
    run_contract = payload.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise ValueError("byte checkpoint omitted its run contract")
    patching_policy = run_contract.get("patching_policy")
    if patching_policy == "causal_entropy_v1":
        if decode_mode != "blt":
            raise ValueError(
                "entropy-patched checkpoints must use variable-topology BLT decoding"
            )
        if entropy_patcher is None:
            raise ValueError(
                "entropy-patched checkpoints require the authenticated runtime "
                "entropy patcher"
            )
        corruption = run_contract.get("corruption")
        if not isinstance(corruption, Mapping):
            raise ValueError("BLT checkpoint omitted its corruption contract")
        trained_length = int(corruption["canvas_length"])
        if block_length != trained_length:
            raise ValueError(
                "entropy BLT eval block must equal the checkpoint's trained canvas: "
                f"requested {block_length}, trained {trained_length}"
            )
        return
    if patching_policy != "fixed_stride_v1":
        raise ValueError(f"unknown checkpoint patching policy {patching_policy!r}")
    if decode_mode != "blt":
        return
    corruption = run_contract.get("corruption")
    if not isinstance(corruption, Mapping):
        raise ValueError("BLT checkpoint omitted its corruption contract")
    trained_length = int(corruption["canvas_length"])
    if block_length != trained_length:
        raise ValueError(
            "BLT eval block must equal the checkpoint's trained canvas: "
            f"requested {block_length}, trained {trained_length}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help=(
            "checkpoint-pinned dataset directory containing the authenticated "
            "entropy patcher; required only for causal_entropy_v1"
        ),
    )
    parser.add_argument("--shots", default="5")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--prompt-format", choices=sorted(PROMPT_FORMATS), default="harness"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--decode-mode",
        choices=("causal", "blt", "duo", "idlm", "diffusion_gemma"),
        default="blt",
    )
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument(
        "--duo-visible-width",
        type=int,
        help=(
            "Optional number of fresh atoms visible in each canvas. Width one "
            "is the true no-noisy-lookahead LTR geometry."
        ),
    )
    parser.add_argument(
        "--duo-commit-width",
        type=int,
        help=(
            "Optional number of earliest denoised atoms committed per 512-atom "
            "canvas; smaller values are a semi-autoregressive quality/compute control."
        ),
    )
    parser.add_argument(
        "--duo-terminal-eps",
        type=float,
        default=1e-5,
        help="Reference Duo reverse-chain terminal time; independent of schedule eps.",
    )
    parser.add_argument(
        "--duo-posterior-precision",
        choices=("float32", "float64"),
        default="float32",
        help="FP64 is the slow exact Torch oracle used by scaling-dLLMs.",
    )
    parser.add_argument(
        "--allow-compatible-duo-source",
        action="store_true",
        help=(
            "Permit a strict state-dict load for sampler-only ablations after "
            "the training source changed; the mismatch is recorded in output."
        ),
    )
    parser.add_argument(
        "--sampling",
        choices=("greedy", "categorical"),
        default="greedy",
        help="Fast-BLT reports greedy task evaluation; categorical is a named ablation.",
    )
    parser.add_argument(
        "--unmasking-strategy",
        choices=("confidence", "entropy_bounded", "fixed_quota"),
        default="confidence",
        help="Fast-BLT confidence/EB rule; fixed_quota is a named ablation.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--entropy-budget", type=float, default=1.0)
    parser.add_argument("--generation-seed", type=int, default=12345)
    parser.add_argument("--max-new-bytes", type=int, default=512)
    parser.add_argument(
        "--max-new-atoms",
        type=int,
        help=(
            "Duo/DiffusionGemma clean-atom continuation cap; defaults to "
            "--max-new-bytes."
        ),
    )
    parser.add_argument(
        "--max-native-actions",
        type=int,
        help="Optional hard forward cap; default is derived so --max-new-bytes is reachable.",
    )
    parser.add_argument("--context-bytes", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--trace-diffusion",
        action="store_true",
        help="Retain every alignment action and denoising schedule state.",
    )
    parser.add_argument("--keep-calculator-annotations", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GSM8K model evaluation requires CUDA through mlq")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable result {args.output}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.trace_diffusion and (
        args.decode_mode not in {"blt", "duo", "diffusion_gemma"}
        or args.limit is None
        or args.limit > 32
    ):
        raise ValueError(
            "--trace-diffusion requires a diffusion mode and explicit --limit <= 32"
        )
    if min(
        args.batch_size,
        args.block_length,
        args.diffusion_steps,
        args.max_new_bytes,
        args.context_bytes,
    ) <= 0:
        raise ValueError("batch, diffusion, byte, and context dimensions must be positive")
    if not 0.0 < args.duo_terminal_eps < 1.0:
        raise ValueError("--duo-terminal-eps must lie in (0, 1)")
    if args.duo_commit_width is not None and not 0 < args.duo_commit_width <= args.block_length:
        raise ValueError("--duo-commit-width must lie in [1, --block-length]")
    if args.duo_visible_width is not None and not 0 < args.duo_visible_width <= args.block_length:
        raise ValueError("--duo-visible-width must lie in [1, --block-length]")
    if args.duo_commit_width is not None and args.duo_visible_width is not None and args.duo_commit_width > args.duo_visible_width:
        raise ValueError("--duo-commit-width cannot exceed --duo-visible-width")
    if args.decode_mode != "duo" and (
        args.duo_terminal_eps != 1e-5
        or args.duo_posterior_precision != "float32"
        or args.allow_compatible_duo_source
        or args.duo_commit_width is not None
        or args.duo_visible_width is not None
    ):
        raise ValueError("Duo sampling controls require --decode-mode duo")
    max_new_atoms = (
        args.max_new_bytes if args.max_new_atoms is None else args.max_new_atoms
    )
    if max_new_atoms <= 0:
        raise ValueError("--max-new-atoms must be positive")
    if (
        args.decode_mode not in {"duo", "diffusion_gemma"}
        and args.max_new_atoms is not None
    ):
        raise ValueError(
            "--max-new-atoms applies to atomic-canvas diffusion decoding"
        )
    max_native_actions = args.max_native_actions
    if not 0.0 < args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must lie in (0, 1]")
    if args.entropy_budget < 0.0 or not np.isfinite(args.entropy_budget):
        raise ValueError("--entropy-budget must be finite and nonnegative")
    if args.decode_mode in {"duo", "idlm", "diffusion_gemma"} and (
        args.sampling != "categorical"
    ):
        raise ValueError(f"{args.decode_mode} uses categorical native sampling")
    device = torch.device("cuda")
    model, payload = load_byte_checkpoint(
        args.checkpoint,
        device,
        decode_mode=args.decode_mode,
        allow_compatible_duo_source=args.allow_compatible_duo_source,
    )
    entropy_patcher = (
        load_entropy_patcher_for_checkpoint(payload, args.dataset_path)
        if args.decode_mode in {"causal", "blt"}
        else None
    )
    if entropy_patcher is not None and args.trace_diffusion:
        raise ValueError(
            "--trace-diffusion is not yet supported by the uncached entropy "
            "reference path because it would omit per-NFE reveal states"
        )
    validate_serving_contract(
        payload,
        decode_mode=args.decode_mode,
        block_length=args.block_length,
        entropy_patcher=entropy_patcher,
    )
    if max_native_actions is None:
        budget = (
            max_new_atoms
            if args.decode_mode in {"duo", "diffusion_gemma"}
            else args.max_new_bytes
        )
        blocks = (
            _duo_required_canvases(
                budget,
                args.block_length,
                model.config.patch_stride,
                args.duo_commit_width if args.decode_mode == "duo" else None,
                args.duo_visible_width if args.decode_mode == "duo" else None,
            )
            if args.decode_mode in {"duo", "diffusion_gemma"}
            else -(-budget // args.block_length)
        )
        max_native_actions = (
            blocks * (args.diffusion_steps + 2)
            if args.decode_mode == "duo"
            else 1 + blocks * (args.diffusion_steps + 1)
            if args.decode_mode == "diffusion_gemma"
            else 2 * budget
            if args.decode_mode == "idlm"
            else args.block_length + 1 + blocks * (args.diffusion_steps + 1)
        )
    if args.decode_mode == "duo":
        if not isinstance(model, DuoModel):
            raise AssertionError("Duo decode loaded the wrong model cell")
        # Compile the expensive complete denoiser, not merely the 261-way
        # posterior tail. A fixed per-eval required-width bucket gives this
        # graph a stable cohort policy without forcing every prompt through the
        # full 8K training limit; warmup is excluded from timed generation.
        model.prepare_clean_bank = torch.compile(  # type: ignore[method-assign]
            model.prepare_clean_bank,
            dynamic=True,
            fullgraph=False,
            mode="max-autotune-no-cudagraphs",
        )
        model.forward_prepared = torch.compile(  # type: ignore[method-assign]
            model.forward_prepared,
            dynamic=True,
            fullgraph=False,
            mode="max-autotune-no-cudagraphs",
        )
    elif args.decode_mode == "idlm":
        if not isinstance(model, IDLMModel):
            raise AssertionError("I-DLM decode loaded the wrong model cell")
        model.forward_sequence = torch.compile(  # type: ignore[method-assign]
            model.forward_sequence,
            dynamic=True,
            fullgraph=False,
            mode="max-autotune-no-cudagraphs",
        )
    elif args.decode_mode == "diffusion_gemma":
        if not isinstance(model, DiffusionGemmaModel):
            raise AssertionError("DiffusionGemma decode loaded the wrong model cell")
        model.forward = torch.compile(  # type: ignore[method-assign]
            model.forward,
            dynamic=True,
            fullgraph=False,
            mode="max-autotune-no-cudagraphs",
        )
    train = pd.read_parquet(TRAIN_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)
    if args.limit is not None:
        test = test.iloc[: args.limit]
    gold = [extract_gold(answer) for answer in test["answer"]]
    if any(answer is None for answer in gold):
        raise ValueError("GSM8K row has no valid gold answer")
    shot_counts = [int(value) for value in args.shots.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    prompt_format = PROMPT_FORMATS[args.prompt_format]
    strip_calculator = not args.keep_calculator_annotations
    records: list[dict] = []
    compile_warmup_seconds = 0.0
    duo_warmed_shapes: set[tuple[int, int]] = set()
    started = time.perf_counter()
    for shots in shot_counts:
        for seed in seeds:
            exemplar_rows = select_exemplars(
                train, shots, max(shot_counts), seed
            )
            exemplars = [train.iloc[index] for index in exemplar_rows]
            prompt_text = [
                build_prompt(
                    exemplars,
                    question,
                    prompt_format,
                    strip_calculator=strip_calculator,
                )
                for question in test["question"]
            ]
            prompts = [text.encode("utf-8") for text in prompt_text]
            duo_work_width = (
                (
                    max(map(len, prompts))
                    + max_new_atoms
                    + model.config.patch_stride
                    - 1
                )
                // model.config.patch_stride
                * model.config.patch_stride
                if isinstance(model, DuoModel)
                else None
            )
            generated: list[ByteGeneration] = []
            traces: list[dict[str, object]] | None = (
                [
                    {"alignment_steps": [], "diffusion_blocks": []}
                    for _ in prompts
                ]
                if args.trace_diffusion
                else None
            )
            model_forwards_total = 0
            physical_batches = 0
            batch_lanes = 0
            warm_compile_evidence: dict[str, int] | None = None
            measured_compile_evidence: dict[str, int] | None = None
            if args.decode_mode == "duo":
                warmup_groups = _duo_warmup_prompt_groups(prompts, args.batch_size)
                pending_warmups = tuple(
                    group
                    for group in warmup_groups
                    if (int(duo_work_width or 0), len(group)) not in duo_warmed_shapes
                )
                dynamo_counters.clear()
                torch.cuda.synchronize()
                warmup_started = time.perf_counter()
                for warmup_index, group in enumerate(pending_warmups):
                    duo_generate_bytes_batched(
                        model,
                        group,
                        max_new_bytes=min(args.max_new_bytes, args.block_length),
                        max_new_atoms=min(max_new_atoms, args.block_length),
                        max_native_actions=args.diffusion_steps + 2,
                        context_bytes=args.context_bytes,
                        stops=(),
                        canvas_length=args.block_length,
                        diffusion_steps=args.diffusion_steps,
                        visible_width=args.duo_visible_width,
                        commit_width=args.duo_commit_width,
                        terminal_eps=args.duo_terminal_eps,
                        use_float64=args.duo_posterior_precision == "float64",
                        seed=args.generation_seed - 1 - warmup_index,
                        device=device,
                        work_width=duo_work_width,
                    )
                    duo_warmed_shapes.add((int(duo_work_width or 0), len(group)))
                torch.cuda.synchronize()
                compile_warmup_seconds += time.perf_counter() - warmup_started
                warm_compile_evidence = _dynamo_compile_evidence()
                dynamo_counters.clear()
            torch.cuda.synchronize()
            generation_started = time.perf_counter()
            eval_batch_size = args.batch_size
            for start in range(0, len(prompts), eval_batch_size):
                selected_prompts = prompts[start : start + eval_batch_size]
                if args.decode_mode == "causal":
                    batch_generations = greedy_generate_bytes(
                        model,
                        selected_prompts,
                        max_new_bytes=args.max_new_bytes,
                        max_native_actions=max_native_actions,
                        context_bytes=args.context_bytes,
                        stops=prompt_format.stops,
                        device=device,
                    )
                elif args.decode_mode == "blt":
                    blt_kwargs = {
                        "max_new_bytes": args.max_new_bytes,
                        "max_native_actions": max_native_actions,
                        "context_bytes": args.context_bytes,
                        "stops": prompt_format.stops,
                        "diffusion_steps": args.diffusion_steps,
                        "unmasking_strategy": args.unmasking_strategy,
                        "confidence_threshold": args.confidence_threshold,
                        "entropy_budget": args.entropy_budget,
                        "seed": (
                            args.generation_seed
                            + 10_000_019 * seed
                            + 97_409 * start
                        ),
                        "stochastic": args.sampling == "categorical",
                        "device": device,
                        "trace_records": (
                            traces[start : start + len(selected_prompts)]
                            if traces is not None
                            else None
                        ),
                    }
                    batch_generations = (
                        entropy_blt_generate_bytes(
                            model,
                            entropy_patcher,
                            selected_prompts,
                            block_length=args.block_length,
                            **blt_kwargs,
                        )
                        if entropy_patcher is not None
                        else blt_generate_bytes_batched(
                            model,
                            selected_prompts,
                            block_length=args.block_length,
                            **blt_kwargs,
                        )
                    )
                elif args.decode_mode == "duo":
                    if not isinstance(model, DuoModel):
                        raise AssertionError("Duo decode loaded the wrong model cell")
                    batch_generations = duo_generate_bytes_batched(
                        model,
                        selected_prompts,
                        max_new_bytes=args.max_new_bytes,
                        max_new_atoms=max_new_atoms,
                        max_native_actions=max_native_actions,
                        context_bytes=args.context_bytes,
                        stops=prompt_format.stops,
                        canvas_length=args.block_length,
                        diffusion_steps=args.diffusion_steps,
                        visible_width=args.duo_visible_width,
                        commit_width=args.duo_commit_width,
                        terminal_eps=args.duo_terminal_eps,
                        use_float64=args.duo_posterior_precision == "float64",
                        seed=(
                            args.generation_seed
                            + 10_000_019 * seed
                            + 97_409 * start
                        ),
                        device=device,
                        work_width=duo_work_width,
                        trace_records=(
                            traces[start : start + len(selected_prompts)]
                            if traces is not None
                            else None
                        ),
                    )
                elif args.decode_mode == "idlm":
                    if not isinstance(model, IDLMModel):
                        raise AssertionError("I-DLM decode loaded the wrong model cell")
                    batch_generations = idlm_generate_bytes_batched(
                        model,
                        selected_prompts,
                        max_new_bytes=args.max_new_bytes,
                        max_native_actions=max_native_actions,
                        context_bytes=args.context_bytes,
                        stops=prompt_format.stops,
                        seed=(
                            args.generation_seed
                            + 10_000_019 * seed
                            + 97_409 * start
                        ),
                        device=device,
                    )
                else:
                    if not isinstance(model, DiffusionGemmaModel):
                        raise AssertionError(
                            "DiffusionGemma decode loaded the wrong model cell"
                        )
                    batch_generations = diffusion_gemma_generate_bytes_batched(
                        model,
                        selected_prompts,
                        max_new_bytes=args.max_new_bytes,
                        max_new_atoms=max_new_atoms,
                        max_native_actions=max_native_actions,
                        context_bytes=args.context_bytes,
                        stops=prompt_format.stops,
                        canvas_length=args.block_length,
                        diffusion_steps=args.diffusion_steps,
                        seed=(
                            args.generation_seed
                            + 10_000_019 * seed
                            + 97_409 * start
                        ),
                        device=device,
                        trace_records=(
                            traces[start : start + len(selected_prompts)]
                            if traces is not None
                            else None
                        ),
                    )
                model_forwards_total += max(
                    item.model_forwards for item in batch_generations
                )
                physical_batches += 1
                batch_lanes += len(batch_generations)
                generated.extend(batch_generations)
                print(
                    f"shots={shots} seed={seed} scored={len(generated)}/{len(prompts)}",
                    flush=True,
                )
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generation_started
            if args.decode_mode == "duo":
                measured_compile_evidence = _dynamo_compile_evidence()
            rows = []
            for index, (generation, truth) in enumerate(
                zip(generated, gold, strict=True)
            ):
                prediction = (
                    extract_prediction(generation.text, prompt_format)
                    if generation.text is not None
                    else None
                )
                rows.append(
                    {
                        "test_row": index,
                        "gold": truth,
                        "prediction": prediction,
                        "correct": prediction is not None and prediction == truth,
                        "delimiter_emitted": (
                            generation.text is not None
                            and prompt_format.delimiter in generation.text
                        ),
                        **asdict(generation),
                        "raw_hex": generation.raw.hex(),
                        **(
                            {"diffusion_trace": traces[index]}
                            if traces is not None
                            else {}
                        ),
                    }
                )
                rows[-1].pop("raw")
            correct = sum(row["correct"] for row in rows)
            records.append(
                {
                    "shots": shots,
                    "seed": seed,
                    "exemplar_train_rows": exemplar_rows,
                    "examples": len(rows),
                    "correct": correct,
                    "exact_match": correct / len(rows),
                    "delimiter_emitted_rate": sum(
                        row["delimiter_emitted"] for row in rows
                    )
                    / len(rows),
                    "parsed_answer_rate": sum(
                        row["prediction"] is not None for row in rows
                    )
                    / len(rows),
                    "invalid_utf8_rate": sum(row["invalid_utf8"] for row in rows)
                    / len(rows),
                    "generation_seconds": generation_seconds,
                    "generated_bytes": sum(
                        len(row["raw_hex"]) // 2 for row in rows
                    ),
                    "generated_atoms": sum(
                        int(row["generated_atoms"]) for row in rows
                    ),
                    "native_actions": sum(
                        int(row["native_actions"]) for row in rows
                    ),
                    "model_forwards": model_forwards_total,
                    "warm_compile_evidence": warm_compile_evidence,
                    "measured_compile_evidence": measured_compile_evidence,
                    "physical_batches": physical_batches,
                    "realized_mean_batch_size": batch_lanes / physical_batches,
                    "mean_prompt_bytes": sum(map(len, prompts)) / len(prompts),
                    "max_prompt_bytes": max(map(len, prompts)),
                    "duo_work_width": duo_work_width,
                    "termination_counts": {
                        reason: sum(row["termination"] == reason for row in rows)
                        for reason in sorted({row["termination"] for row in rows})
                    },
                    "serialized_sample_count": min(args.samples, len(rows)),
                    "rows": rows[: args.samples],
                }
            )
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    result = {
        "schema": EVALUATOR_SCHEMA,
        "evaluator_source": evaluator_source_provenance(),
        "generation_semantics": (
            "strict_utf8_atomic_controls_dual_atom_byte_caps_in_canvas_stops"
            if args.decode_mode in {"duo", "diffusion_gemma"}
            else "strict_utf8_atomic_controls_literal_byte_cap"
            if args.decode_mode == "idlm"
            else "literal_byte_budget"
        ),
        "implementation_maturity": (
            "full_prefix_recompute_correctness_adapter"
            if args.decode_mode == "causal"
            else "per_row_uncached_exact_variable_patch_topology"
            if args.decode_mode == "blt" and entropy_patcher is not None
            else "ragged_batched_incremental_fast_blt_cache"
            if args.decode_mode == "blt"
            else "batched_exact_fused_isd_full_prefix_replay"
            if args.decode_mode == "idlm"
            else "ragged_batched_revisable_canvas_full_clean_replay"
            if args.decode_mode == "diffusion_gemma"
            else "ragged_batched_duo_per_canvas_clean_bank_cache"
        ),
        "decode_mode": args.decode_mode,
        "patching_policy": (
            payload.get("run_contract", {}).get("patching_policy")
            if args.decode_mode in {"causal", "blt"}
            else None
        ),
        "entropy_patcher_sha256": (
            entropy_patcher.sha256 if entropy_patcher is not None else None
        ),
        "entropy_dataset_path": (
            str(args.dataset_path) if entropy_patcher is not None else None
        ),
        "block_length": (
            args.block_length
            if args.decode_mode in {"blt", "duo", "idlm", "diffusion_gemma"}
            else None
        ),
        "diffusion_steps": (
            args.diffusion_steps
            if args.decode_mode in {"blt", "duo", "diffusion_gemma"}
            else None
        ),
        "schedule_eps": (
            float(payload["training"]["schedule_eps"])
            if args.decode_mode == "duo"
            else None
        ),
        "sampling_terminal_eps": (
            args.duo_terminal_eps if args.decode_mode == "duo" else None
        ),
        "duo_posterior_precision": (
            args.duo_posterior_precision if args.decode_mode == "duo" else None
        ),
        "duo_commit_width": (
            args.duo_commit_width if args.decode_mode == "duo" else None
        ),
        "duo_visible_width": (
            args.duo_visible_width if args.decode_mode == "duo" else None
        ),
        "checkpoint_training_source_matches_current": (
            payload.get("source_sha256") == duo_source_provenance()["sha256"]
            if args.decode_mode == "duo"
            else None
        ),
        "compatible_duo_source_override": (
            args.allow_compatible_duo_source if args.decode_mode == "duo" else None
        ),
        "unmasking_strategy": (
            args.unmasking_strategy if args.decode_mode == "blt" else None
        ),
        "confidence_threshold": (
            args.confidence_threshold
            if args.decode_mode == "blt"
            and args.unmasking_strategy == "confidence"
            else None
        ),
        "entropy_budget": (
            args.entropy_budget
            if args.decode_mode == "blt"
            and args.unmasking_strategy == "entropy_bounded"
            else None
        ),
        "generation_seed": args.generation_seed,
        "sampling": (
            args.sampling
            if args.decode_mode in {"blt", "duo", "idlm", "diffusion_gemma"}
            else "greedy"
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_schema": payload["schema"],
        "training_source_sha256": (
            payload["source_sha256"]
            if args.decode_mode in {"duo", "diffusion_gemma"}
            else payload["extra"]["source_sha256"]
            if args.decode_mode == "idlm"
            else payload["source_provenance"]["sha256"]
        ),
        "completed_steps": (
            payload["steps"]
            if args.decode_mode in {"duo", "diffusion_gemma"}
            else payload["completed_steps"]
        ),
        "model_parameters": (
            model.parameter_count
            if isinstance(model, (DuoModel, DiffusionGemmaModel))
            else model.parameter_count()
        ),
        "gsm8k_snapshot": str(GSM8K_ROOT),
        "gsm8k_train_sha256": sha256(TRAIN_PARQUET),
        "gsm8k_test_sha256": sha256(TEST_PARQUET),
        "prompt_format": prompt_format.name,
        "strip_calculator_annotations": strip_calculator,
        "max_new_bytes": args.max_new_bytes,
        "max_new_atoms": (
            max_new_atoms
            if args.decode_mode in {"duo", "diffusion_gemma"}
            else None
        ),
        "max_native_actions": max_native_actions,
        "context_bytes": args.context_bytes,
        "requested_batch_size": args.batch_size,
        "requested_serialized_samples": args.samples,
        "batching": (
            "ragged_per_canvas_clean_bank_cache"
            if args.decode_mode == "duo"
            else "ragged_batched_full_replay"
            if args.decode_mode in {"idlm", "diffusion_gemma"}
            else "independent_variable_topology_reference"
            if entropy_patcher is not None
            else "ragged_cache"
        ),
        "diffusion_trace_included": args.trace_diffusion,
        "compile_warmup_seconds_excluded_from_generation": compile_warmup_seconds,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                f"{record['shots']}shot_seed{record['seed']}": record["exact_match"]
                for record in records
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

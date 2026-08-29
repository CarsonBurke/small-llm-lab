"""Deployment-only Byte-Duo continuation helpers.

This module intentionally depends only on the Duo model, sampler, atomic byte
contract, NumPy, and Torch. It is the small serving closure counted in final
artifacts; benchmark orchestration and other model families stay outside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .data import AtomicIdManifest
from .duo import DuoSchedule
from .duo_model import DuoModel
from .inference_duo import generate_duo_continuation
from .patching import CausalEntropyPatcher
from .tokenizer import IncrementalByteDecoder


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


def decode_atomic_continuation(
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


def pack_byte_prompts(
    prompts: Sequence[bytes], *, width: int, pad_id: int
) -> tuple[Tensor, Tensor]:
    lengths = np.fromiter((len(prompt) for prompt in prompts), dtype=np.int64)
    if lengths.size == 0 or int(lengths.max()) > width:
        raise ValueError("prompt cohort is empty or exceeds its work width")
    offsets = np.empty(lengths.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    flat = np.frombuffer(b"".join(prompts), dtype=np.uint8)
    rows = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
    columns = np.arange(flat.size, dtype=np.int64) - np.repeat(
        offsets[:-1], lengths
    )
    packed = np.full((lengths.size, width), pad_id, dtype=np.int64)
    packed[rows, columns] = flat
    return torch.from_numpy(packed), torch.from_numpy(lengths)


def required_canvases(
    max_new_atoms: int,
    canvas_length: int,
    stride: int,
    commit_width: int | None = None,
    visible_width: int | None = None,
) -> int:
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


def warmup_prompt_groups(
    prompts: Sequence[bytes], batch_size: int
) -> tuple[Sequence[bytes], ...]:
    if not prompts or batch_size <= 0:
        raise ValueError("Duo warmup needs prompts and a positive batch size")
    full = prompts[:batch_size]
    tail_size = len(prompts) % batch_size
    if tail_size and tail_size != len(full):
        return full, prompts[-tail_size:]
    return (full,)


def first_stop(data: bytes, stops: tuple[bytes, ...]) -> int | None:
    offsets = [offset for stop in stops if (offset := data.find(stop)) >= 0]
    return min(offsets) if offsets else None


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
    entropy_patcher: CausalEntropyPatcher | None = None,
) -> list[ByteGeneration]:
    """Evaluate Byte-Duo from clean prompts; only fresh suffix atoms diffuse."""

    if not prompts:
        return []
    if trace_records is not None and len(trace_records) != len(prompts):
        raise ValueError("Duo trace records must align with prompts")
    entropy_policy = model.config.duo_clean_patching == "causal_entropy_v1"
    if entropy_policy != (entropy_patcher is not None):
        raise ValueError(
            "Duo entropy serving requires exactly its authenticated patcher"
        )
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
    host_ids, host_prompt_lengths = pack_byte_prompts(
        prompts, width=width, pad_id=model.config.vocab.pad_id
    )
    if device.type == "cuda":
        host_ids = host_ids.pin_memory()
        host_prompt_lengths = host_prompt_lengths.pin_memory()
    ids = host_ids.to(device, non_blocking=device.type == "cuda")
    prompt_lengths = host_prompt_lengths.to(device, non_blocking=device.type == "cuda")
    valid = torch.arange(width, device=device)[None] < prompt_lengths[:, None]
    per_canvas_forwards = diffusion_steps + 2
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
            schedule=DuoSchedule(model.schedule_eps),
            visible_width=visible_width,
            commit_width=commit_width,
            terminal_eps=terminal_eps,
            max_canvases=max_canvases,
            generator=generator,
            use_float64=use_float64,
            return_trajectories=trace_records is not None,
            stop_sequences=encoded_stops,
            entropy_patcher=entropy_patcher,
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
        raw, text, invalid, saw_eot = decode_atomic_continuation(
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
        stop = first_stop(raw, encoded_stops)
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
                    position for position in active if start + position < length_after
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

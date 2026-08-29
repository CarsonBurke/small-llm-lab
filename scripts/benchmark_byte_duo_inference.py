#!/usr/bin/env python3
"""Measure production Byte-Duo cache/denoiser throughput for a 512-atom budget.

All four prompt phases are exercised.  The canvas count is derived from the
trained width: 512-wide training uses one or two canvases, while 256-wide
training uses two or three. The harness deliberately ignores semantic EOT stopping so every
row completes the same requested compute budget. It exercises the production
compiled clean-bank cache, compiled branch-only denoiser, and fail-closed
Triton exact posterior. Compile warmup is excluded. GPU execution must go
through ``mlq``.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from torch._dynamo.utils import counters as dynamo_counters


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.config import ByteDiffusionConfig, model_config_from_env
from pretraining.byte_diffusion.inference_duo import (
    duo_entropy_clean_metadata,
    sample_duo_canvas,
)
from pretraining.byte_diffusion.patching import CausalEntropyPatcher
from pretraining.byte_diffusion.variable_patching import load_dataset_patching_spec
from scripts.train_byte_duo import source_provenance
from scripts.train_byte_duo import _local_imports
from pretraining.byte_diffusion.readiness import (
    DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
    DUO_INFERENCE_READINESS_SCHEMA,
    HEADROOM_FRACTION,
    HEADROOM_MINIMUM_GIB,
    MIN_TELEMETRY_SAMPLES,
    SUSTAINED_GPU_POLICY,
    duo_geometry_contract,
)
from pretraining.byte_diffusion.telemetry import nvidia_smi_selector


def _telemetry_loop(
    stop: threading.Event,
    gpu_index: str,
    powers: array,
    utilizations: array,
) -> None:
    while not stop.wait(0.05):
        observed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
                f"--id={gpu_index}",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if observed.returncode:
            continue
        try:
            power, separator, utilization = observed.stdout.partition("\n")[0].partition(",")
            if separator:
                powers.append(float(power.strip()))
                utilizations.append(float(utilization.strip().removesuffix(" %")))
        except ValueError:
            continue


def _summary(samples: array) -> dict[str, float | int | None]:
    if not samples:
        return {"count": 0, "mean": None, "p10": None, "peak": None}
    values = np.frombuffer(samples, dtype=np.float64)
    return {
        "count": len(samples),
        "mean": math.fsum(samples) / len(samples),
        "p10": float(np.quantile(values, 0.1, method="linear")),
        "peak": max(samples),
    }


def benchmark_source_provenance() -> dict[str, object]:
    """Bind inference readiness to the complete benchmark/serving closure."""

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
        "schema": "byte_duo_inference_benchmark_source/v1",
        "sha256": digest.hexdigest(),
        "files": list(map(str, relative_paths)),
    }


def _prompt_batch(
    batch_size: int,
    prompt_length: int,
    max_new_atoms: int,
    *,
    phase: int,
    pad_id: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 <= phase < stride or prompt_length % stride:
        raise ValueError("base prompt must be aligned and phase lie inside the stride")
    actual_prompt_length = prompt_length + phase
    width = math.ceil((actual_prompt_length + max_new_atoms) / stride) * stride
    # Non-EOT ASCII is a valid one-byte UTF-8 prefix. Build the whole cohort in
    # one NumPy allocation and transfer it once, matching the production packer.
    host = np.full((batch_size, width), pad_id, dtype=np.int64)
    host[:, :actual_prompt_length] = (
        np.arange(actual_prompt_length, dtype=np.int64)[None] % 95 + 32
    )
    lengths = np.full(batch_size, actual_prompt_length, dtype=np.int64)
    ids = torch.from_numpy(host).pin_memory().to("cuda", non_blocking=True)
    prompt_lengths = torch.from_numpy(lengths).pin_memory().to(
        "cuda", non_blocking=True
    )
    valid = torch.arange(width, device="cuda")[None] < prompt_lengths[:, None]
    return ids, valid


def _serving_canvas_origin(
    current_length: int, stride: int, *, full_resolution: bool
) -> tuple[int, int]:
    """Return the deployed branch start and immutable in-canvas prefix width."""

    if current_length <= 0 or stride <= 0:
        raise ValueError("serving origin dimensions must be positive")
    if full_resolution:
        return current_length, 0
    start = current_length // stride * stride
    return start, current_length - start


@torch.inference_mode()
def _run_once(
    model: DuoModel,
    ids: torch.Tensor,
    valid: torch.Tensor,
    *,
    prompt_length: int,
    canvas_length: int,
    requested_atoms: int,
    steps: int,
    seed: int,
    entropy_patcher: CausalEntropyPatcher | None = None,
) -> dict[str, int]:
    """Complete exactly one atom budget without semantic early termination."""

    generator = torch.Generator(device=ids.device).manual_seed(seed)
    stride = model.config.patch_stride
    document_ids = torch.arange(ids.shape[0], device=ids.device)[:, None].expand_as(ids)
    positions = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
    full_resolution = (
        model.config.duo_mutable_topology == "full_resolution_decoder"
    )
    current_length = prompt_length
    remaining = requested_atoms
    canvases = 0
    model_forwards = 0
    denoised_atom_slots = 0
    while remaining:
        start, phase = _serving_canvas_origin(
            current_length,
            stride,
            full_resolution=full_resolution,
        )
        fresh_count = min(canvas_length - phase, remaining)
        offsets = torch.arange(canvas_length, device=ids.device)[None].expand(
            ids.shape[0], -1
        )
        fresh = (offsets >= phase) & (offsets < phase + fresh_count)
        visible = offsets < phase + fresh_count
        branch_starts = torch.full(
            (ids.shape[0], 1), start, dtype=torch.long, device=ids.device
        )
        absolute = branch_starts + offsets
        safe_absolute = absolute.clamp_max(ids.shape[1] - 1)
        initial = torch.gather(ids, 1, safe_absolute)
        initial = torch.where(visible, initial, model.config.vocab.pad_id)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if entropy_patcher is None:
                attention_metadata = model.prepare_attention_metadata(
                    valid, document_ids
                )
                clean_patch_metadata = None
            else:
                attention_metadata = None
                clean_patch_metadata = duo_entropy_clean_metadata(
                    entropy_patcher, ids, valid, document_ids
                )
            clean = model.prepare_clean_bank(
                ids,
                valid,
                document_ids,
                positions,
                attention_metadata=attention_metadata,
                clean_patch_metadata=clean_patch_metadata,
            )
            cache = model.prepare_canvas_cache(
                clean,
                visible[:, None],
                branch_starts,
                allow_synthetic_branch_suffix=True,
                validate_inputs=False,
            )
            output = sample_duo_canvas(
                model,
                ids,
                valid,
                document_ids,
                positions,
                branch_starts,
                visible[:, None],
                steps=steps,
                generator=generator,
                initial_ids=initial,
                denoise_active=fresh,
                return_trajectory=False,
                posterior_backend="triton",
                prepared_cache=cache,
            )
        committed = torch.where(fresh, output.ids, initial)
        ids = ids.scatter(1, safe_absolute, committed)
        current_length += fresh_count
        remaining -= fresh_count
        valid = (
            torch.arange(ids.shape[1], device=ids.device)[None] < current_length
        ).expand(ids.shape[0], -1)
        canvases += 1
        model_forwards += steps + 2
        denoised_atom_slots += canvas_length
    return {
        "canvases": canvases,
        "model_forwards": model_forwards,
        "denoising_forwards": canvases * (steps + 1),
        "clean_cache_forwards": canvases,
        "requested_atoms": requested_atoms,
        "denoised_atom_slots": denoised_atom_slots,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--canvas-length", type=int, default=512)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument(
        "--requested-atoms",
        type=int,
        default=DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
        help="fixed generated-atom budget, independent of trained canvas width",
    )
    parser.add_argument("--diffusion-steps", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data" / "byte_diffusion_aligned_v5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.batch_size,
        args.prompt_length,
        args.canvas_length,
        args.branches,
        args.requested_atoms,
        args.diffusion_steps,
        args.repetitions,
    ) <= 0:
        raise ValueError("all benchmark dimensions must be positive")
    if (
        args.batch_size != 8
        or args.prompt_length != 2_048
        or args.requested_atoms != DUO_CANONICAL_VALIDATION_CANVAS_LENGTH
        or args.diffusion_steps != 8
        or args.repetitions != 12
    ):
        raise ValueError(
            "inference readiness requires the production benchmark geometry"
        )
    geometry_contract = duo_geometry_contract(args.canvas_length, args.branches)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Canonicalize the implicit CUDA device once so telemetry and the emitted
    # provenance artifact agree on the logical device ordinal.
    device = torch.device("cuda", torch.cuda.current_device())
    config = model_config_from_env()
    dataset_manifest = json.loads((args.data_path / "manifest.json").read_text())
    patching = load_dataset_patching_spec(args.data_path, dataset_manifest)
    if patching.name != config.duo_clean_patching:
        raise ValueError("inference dataset and Duo clean patching policy differ")
    if patching.variable:
        artifact = dataset_manifest["patching"]["patcher_artifact"]
        entropy_patcher = CausalEntropyPatcher.from_bytes(
            (args.data_path / artifact["path"]).read_bytes()
        )
        if entropy_patcher.sha256 != patching.artifact_sha256:
            raise ValueError("inference entropy patcher authentication failed")
    else:
        entropy_patcher = None
    full_resolution = config.duo_mutable_topology == "full_resolution_decoder"
    model = DuoModel(config).to(device).eval()
    if config == ByteDiffusionConfig():
        model.validate_production_parameterization()
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
    stride = model.config.patch_stride
    phases = tuple(range(stride))
    phase_batches = tuple(
        _prompt_batch(
            args.batch_size,
            args.prompt_length,
            args.requested_atoms,
            phase=phase,
            pad_id=model.config.vocab.pad_id,
            stride=stride,
        )
        for phase in phases
    )
    torch.cuda.synchronize()
    dynamo_counters.clear()
    warmup_started = time.perf_counter()
    warmup_work = tuple(
        _run_once(
            model,
            ids,
            valid,
            prompt_length=args.prompt_length + phase,
            canvas_length=args.canvas_length,
            requested_atoms=args.requested_atoms,
            steps=args.diffusion_steps,
            seed=args.seed + phase,
            entropy_patcher=entropy_patcher,
        )
        for phase, (ids, valid) in zip(phases, phase_batches, strict=True)
    )
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started
    warmup_graphs = int(dynamo_counters["stats"]["unique_graphs"])
    warmup_graph_breaks = int(sum(dynamo_counters["graph_break"].values()))
    dynamo_counters.clear()

    powers = array("d")
    utilizations = array("d")
    stop = threading.Event()
    telemetry = threading.Thread(
        target=_telemetry_loop,
        args=(stop, nvidia_smi_selector(device), powers, utilizations),
        daemon=True,
    )
    torch.cuda.reset_peak_memory_stats(device)
    telemetry.start()
    phase_records: list[dict[str, object]] = []
    measured_seconds = 0.0
    for phase, (ids, valid), expected in zip(
        phases, phase_batches, warmup_work, strict=True
    ):
        torch.cuda.synchronize()
        started = time.perf_counter()
        work = expected
        for repetition in range(args.repetitions):
            work = _run_once(
                model,
                ids,
                valid,
                prompt_length=args.prompt_length + phase,
                canvas_length=args.canvas_length,
                requested_atoms=args.requested_atoms,
                steps=args.diffusion_steps,
                seed=args.seed + 10_003 * phase + repetition + 1,
                entropy_patcher=entropy_patcher,
            )
        torch.cuda.synchronize()
        phase_seconds = time.perf_counter() - started
        measured_seconds += phase_seconds
        expected_canvases = math.ceil(
            (
                args.requested_atoms
                + (0 if full_resolution else phase)
            )
            / args.canvas_length
        )
        if work != expected or work["canvases"] != expected_canvases:
            raise AssertionError("phase workload did not complete its exact atom budget")
        phase_trajectories = args.batch_size * args.repetitions
        phase_records.append(
            {
                "phase": phase,
                "prompt_length": args.prompt_length + phase,
                "work_width": ids.shape[1],
                **work,
                "measured_seconds": phase_seconds,
                "milliseconds_per_batched_trajectory": (
                    1_000 * phase_seconds / args.repetitions
                ),
                "trajectories_per_second": phase_trajectories / phase_seconds,
                "requested_atoms_per_second": (
                    phase_trajectories * args.requested_atoms / phase_seconds
                ),
                "denoised_atom_slots_per_second": (
                    phase_trajectories * int(work["denoised_atom_slots"])
                    / phase_seconds
                ),
            }
        )
    stop.set()
    telemetry.join(timeout=2)

    trajectories = args.batch_size * args.repetitions * len(phases)
    requested_atoms = trajectories * args.requested_atoms
    denoised_atom_slots = sum(
        args.batch_size
        * args.repetitions
        * int(record["denoised_atom_slots"])
        for record in phase_records
    )
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    required_headroom = max(
        math.ceil(total_memory * HEADROOM_FRACTION),
        math.ceil(HEADROOM_MINIMUM_GIB * (1 << 30)),
    )
    power = _summary(powers)
    utilization = _summary(utilizations)
    measured_dynamo = {
        "warmup_unique_graphs": warmup_graphs,
        "warmup_graph_breaks": warmup_graph_breaks,
        "measured_unique_graphs": int(dynamo_counters["stats"]["unique_graphs"]),
        "measured_graph_breaks": int(sum(dynamo_counters["graph_break"].values())),
        "measured_recompiles": int(sum(dynamo_counters["recompiles"].values())),
    }
    eligible = (
        int(power["count"] or 0) >= MIN_TELEMETRY_SAMPLES
        and int(utilization["count"] or 0) >= MIN_TELEMETRY_SAMPLES
        # This fixed batch-eight benchmark measures online latency. Requiring
        # saturated throughput here rewards batching rather than a faster
        # trajectory and made valid low-latency execution impossible to
        # authenticate. Saturation remains mandatory for training/validation.
        and math.isfinite(float(power["mean"] or math.nan))
        and math.isfinite(float(utilization["mean"] or math.nan))
        and total_memory - peak_reserved >= required_headroom
        and warmup_graph_breaks == 0
        and measured_dynamo["measured_unique_graphs"] == 0
        and measured_dynamo["measured_graph_breaks"] == 0
        and measured_dynamo["measured_recompiles"] == 0
    )
    result = {
        "schema": DUO_INFERENCE_READINESS_SCHEMA,
        "recipe_source": source_provenance(),
        "benchmark_source": benchmark_source_provenance(),
        "model_config": config.to_dict(),
        "parameter_count": model.parameter_count,
        "dataset_payload_sha256": dataset_manifest.get("payload_sha256"),
        "dataset_patching": {
            "name": patching.name,
            "max_patch_size": patching.max_patch_size,
            "patcher_sha256": patching.artifact_sha256,
        },
        "serving_origin_policy": (
            "exact_prompt_length"
            if full_resolution
            else "floor_to_patch_and_carry_clean_phase"
        ),
        "semantic_generation": False,
        "semantic_generation_reason": "fixed compute budget ignores EOT stopping",
        "posterior_backend": "triton",
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "base_prompt_length": args.prompt_length,
        "phase_coverage": phases,
        "canvas_length": args.canvas_length,
        "branches": args.branches,
        "requested_atoms_per_trajectory": args.requested_atoms,
        **geometry_contract,
        "diffusion_steps": args.diffusion_steps,
        "repetitions": args.repetitions,
        "compile_warmup_seconds": warmup_seconds,
        "measured_seconds": measured_seconds,
        "trajectories_per_second": trajectories / measured_seconds,
        "requested_atoms_per_second": requested_atoms / measured_seconds,
        "denoised_atom_slots_per_second": denoised_atom_slots / measured_seconds,
        "phases": phase_records,
        "peak_allocated_gib": peak_allocated / (1 << 30),
        "peak_reserved_gib": peak_reserved / (1 << 30),
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_reserved_headroom_bytes": total_memory - peak_reserved,
        "required_headroom_bytes": required_headroom,
        "power_w": power,
        "gpu_utilization_percent": utilization,
        "sustained_gpu_policy": SUSTAINED_GPU_POLICY,
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "index": device.index,
            "total_memory_bytes": total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvidia_smi_selector": nvidia_smi_selector(device),
        },
        "runtime": {"compiled": True, "world_size": 1},
        "torch_dynamo": measured_dynamo,
        "eligible": eligible,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, sort_keys=True), flush=True)
    if not eligible:
        raise RuntimeError("Byte-Duo inference did not meet readiness policy")


if __name__ == "__main__":
    main()

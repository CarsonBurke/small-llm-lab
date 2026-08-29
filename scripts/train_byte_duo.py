#!/usr/bin/env python3
"""Train the standalone non-absorbing Byte-Duo recipe.

All GPU executions of this script must be submitted through ``mlq``.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from checkpointing import RecoveryCheckpointPolicy


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.config import model_config_from_env
from pretraining.byte_diffusion.data import DeterministicChunkCursor
from pretraining.byte_diffusion.diffusion_gemma_model import (
    compile_stable_document_metadata,
)
from pretraining.byte_diffusion.duo import (
    DuoSchedule,
    compiled_duo_nelbo_token_loss,
    duo_nelbo_token_loss,
    sample_branch_antithetic_times,
)
from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.readiness import (
    DUO_CANONICAL_VALIDATION_BRANCHES,
    DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
    DUO_DATA_BRANCH_SPAN_LENGTH,
    DUO_DATA_REQUIRED_BRANCH_BYTES,
    diagnostic_cadence_contract,
    duo_geometry_contract,
    materialize_training_diagnostics,
    validate_architecture_readiness,
    validate_duo_distributed_readiness,
    validate_duo_inference_readiness,
)
from pretraining.byte_diffusion.export import ARTIFACT_CAP_BYTES, artifact_size_report
from pretraining.byte_diffusion.pipeline import DeviceBatchPrefetcher
from pretraining.byte_diffusion.training import (
    DistributedContext,
    TrainingBatch,
    distributed_local_counts,
    load_data_directory,
)
from pretraining.byte_diffusion.training_duo import (
    DUO_TRAINING_OBJECTIVES,
    JOINT_DUO_CLEAN_AR_OBJECTIVE,
    PURE_DUO_OBJECTIVE,
    DuoBatch,
    PreparedDuoUpdate,
    PreparedDuoValidationBatch,
    accumulate_duo_validation_stats,
    duo_validation_from_stats,
    duo_clean_ar_weight,
    duo_loss,
    duo_mutable_topology_contract,
    duo_objective_contract,
    prepare_duo_update,
    prepare_duo_validation_batch,
)
from pretraining.byte_diffusion.variable_patching import (
    DatasetPatchingSpec,
    build_duo_clean_patch_metadata,
)


def _authenticated_entropy_patcher_bytes(
    data_path: Path,
    dataset_manifest: Mapping[str, object],
    dataset_patching: DatasetPatchingSpec,
) -> bytes | None:
    """Load only the patcher already authenticated by the dataset contract."""

    if not dataset_patching.variable:
        return None
    patching_record = dataset_manifest.get("patching")
    artifact_record = (
        patching_record.get("patcher_artifact")
        if isinstance(patching_record, Mapping)
        else None
    )
    relative_path = (
        artifact_record.get("path")
        if isinstance(artifact_record, Mapping)
        else None
    )
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("entropy dataset omitted its patcher artifact path")
    patcher_path = (data_path / relative_path).resolve()
    if not patcher_path.is_relative_to(data_path.resolve()):
        raise ValueError("entropy patcher artifact escaped its dataset directory")
    payload = patcher_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != dataset_patching.artifact_sha256:
        raise ValueError("entropy patcher bytes differ from the dataset contract")
    return payload


def _local_imports(path: Path) -> tuple[Path, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    candidates: set[Path] = set()

    def add_module(parts: list[str]) -> None:
        if not parts:
            return
        module = REPO_ROOT.joinpath(*parts)
        resolved = False
        for candidate in (module.with_suffix(".py"), module / "__init__.py"):
            if candidate.is_file():
                candidates.add(candidate.resolve())
                resolved = True
        if not resolved:
            return
        # Importing a nested module executes every package initializer on its
        # path.  They are therefore part of the executable artifact even when
        # the source spells only the leaf module.
        for depth in range(1, len(parts)):
            initializer = REPO_ROOT.joinpath(*parts[:depth], "__init__.py")
            if initializer.is_file():
                candidates.add(initializer.resolve())

    package = list(path.relative_to(REPO_ROOT).with_suffix("").parts[:-1])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_module(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - (node.level - 1)] if node.level else []
            module = [] if node.module is None else node.module.split(".")
            add_module(base + module)
            for alias in node.names:
                if alias.name != "*":
                    add_module(base + module + alias.name.split("."))
    return tuple(sorted(candidates))


def _source_closure(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    """Return the transitive repository-local Python import closure."""

    pending = {path.resolve() for path in roots}
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        if not path.is_relative_to(REPO_ROOT):
            raise ValueError(f"Byte-Duo source escaped repository: {path}")
        observed.add(path)
        pending.update(item for item in _local_imports(path) if item not in observed)
    return tuple(sorted(observed))


def counted_duo_code_paths() -> tuple[Path, ...]:
    """Return every source file required to train, export, and evaluate Duo."""

    return _source_closure(
        (
            Path(__file__),
            REPO_ROOT / "pretraining/byte_diffusion/inference_duo.py",
            REPO_ROOT / "scripts/export_byte_duo.py",
            REPO_ROOT / "pretraining/eval_byte_duo_gsm8k.py",
        )
    )


def _paths_provenance(
    paths: tuple[Path, ...], *, schema: str
) -> dict[str, object]:
    """Hash named repository files without path/content ambiguity."""

    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for path in paths:
        relative = path.relative_to(REPO_ROOT).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative] = file_hash
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
    return {
        "schema": schema,
        "sha256": digest.hexdigest(),
        "files": files,
    }


def source_provenance() -> dict[str, object]:
    """Hash the complete repository-local source closure of this recipe."""

    # Training and the exact sampler jointly define the promoted recipe. The
    # sampler is not imported by training, so bind it as an explicit root.
    return _paths_provenance(
        _source_closure(
            (
                Path(__file__),
                REPO_ROOT / "pretraining/byte_diffusion/inference_duo.py",
            )
        ),
        schema="byte_duo_source_provenance/v1",
    )


def complete_code_provenance() -> dict[str, object]:
    """Bind the shippable trainer, sampler, exporter, and evaluator closure."""

    return _paths_provenance(
        counted_duo_code_paths(), schema="byte_duo_complete_code_provenance/v1"
    )


def capture_rng_state(
    generator: torch.Generator, *, device: torch.device
) -> dict[str, object]:
    """Capture every RNG stream that can affect an update."""

    return {
        "generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def restore_rng_state(
    state: dict[str, object],
    generator: torch.Generator,
    *,
    device: torch.device,
) -> None:
    """Restore the complete RNG contract before the next cursor draw."""

    # ``map_location=device`` relocates serialized byte tensors.  Both CPU
    # generators and the process CPU RNG require their state tensors on CPU.
    generator.set_state(state["generator_state"].cpu())
    torch.set_rng_state(state["torch_rng_state"].cpu())
    np.random.set_state(state["numpy_rng_state"])
    random.setstate(state["python_rng_state"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(
            [item.cpu() for item in state["cuda_rng_state"]]
        )


def distributed_rank_positions(
    global_count: int,
    *,
    rank: int,
    world_size: int,
    rotation: int = 0,
) -> np.ndarray:
    """Return this rank's disjoint positions in one exact global update.

    The remainder rotates between ranks, while every rank advances the same
    global cursor.  This preserves exactly 249 distinct rows on eight devices
    instead of padding the update to 256 or duplicating examples.
    """

    if global_count <= 0 or world_size <= 0:
        raise ValueError("global count and world size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must lie inside world size")
    rotation %= world_size
    positions = np.arange(global_count, dtype=np.int64)
    selected = positions[(positions + rotation) % world_size == rank]
    counts = distributed_local_counts(global_count, world_size)
    expected = counts[(rank - rotation) % world_size]
    if selected.size != expected:
        raise AssertionError("distributed Byte-Duo sharding produced the wrong size")
    return selected


def distributed_microstep_count(
    global_count: int, *, world_size: int, microbatch_size: int
) -> int:
    """Validate that uneven exact sharding gives every rank equal collectives."""

    if microbatch_size <= 0:
        raise ValueError("microbatch size must be positive")
    counts = distributed_local_counts(global_count, world_size)
    if not all(counts):
        raise ValueError("WORLD_SIZE cannot exceed the exact global batch")
    microsteps = tuple(-(-count // microbatch_size) for count in counts)
    if len(set(microsteps)) != 1:
        raise ValueError(
            "uneven global batch would give ranks different backward-call "
            f"counts: rows={counts}, microsteps={microsteps}"
        )
    return microsteps[0]


def distributed_loss_scale(world_size: int) -> float:
    """Undo DDP's gradient average after global-denominator normalization."""

    if world_size <= 0:
        raise ValueError("world size must be positive")
    return float(world_size)


def rank_checkpoint_path(path: Path, rank: int, *, step: int) -> Path:
    """Infer the deterministic per-rank stochastic-state sidecar path."""

    if rank < 0 or step < 0:
        raise ValueError("checkpoint rank and step must be nonnegative")
    return path.with_name(
        f"{path.stem}.step{step:08d}.rank{rank:05d}{path.suffix}"
    )

def prune_stale_rank_checkpoints(
    path: Path, rank: int, *, committed_step: int
) -> None:
    """Retain only the sidecar referenced by the committed main checkpoint."""

    current = rank_checkpoint_path(path, rank, step=committed_step)
    for candidate in path.parent.glob(
        f"{path.stem}.step*.rank{rank:05d}{path.suffix}"
    ):
        if candidate != current:
            candidate.unlink(missing_ok=True)


def capture_rank_state(
    time_generator: torch.Generator,
    corruption_generator: torch.Generator,
    *,
    cursor: DeterministicChunkCursor,
    device: torch.device,
    prepared_host_state: dict[str, object] | None = None,
) -> dict[str, object]:
    """Capture all rank-local state needed for exact update-boundary resume."""

    if prepared_host_state is None:
        generator_state = corruption_generator.get_state()
        time_generator_state = time_generator.get_state()
        cursor_state = cursor.state_dict()
    else:
        required = {
            "corruption_generator_state",
            "time_generator_state",
            "cursor",
        }
        missing = required.difference(prepared_host_state)
        if missing:
            raise ValueError(f"prepared Duo host state omitted {sorted(missing)}")
        generator_state = prepared_host_state["corruption_generator_state"]
        time_generator_state = prepared_host_state["time_generator_state"]
        cursor_state = prepared_host_state["cursor"]
    # Prefetch may already be preparing N+1 while the main thread checkpoints
    # committed update N. Process/global RNG is owned by the main thread, while
    # cursor and explicit CPU generators come from N's immutable snapshot.
    state = {
        "generator_state": generator_state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        ),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    return {
        **state,
        "time_generator_state": time_generator_state,
        "cursor": cursor_state,
    }


def restore_rank_state(
    state: dict[str, object],
    time_generator: torch.Generator,
    corruption_generator: torch.Generator,
    *,
    cursor: DeterministicChunkCursor,
    device: torch.device,
) -> None:
    """Restore a rank sidecar without depending on another rank's RNG."""

    restore_rng_state(state, corruption_generator, device=device)
    time_generator.set_state(state["time_generator_state"].cpu())
    cursor.load_state_dict(state["cursor"])


def _all_reduce_sum(value: torch.Tensor, distributed: DistributedContext) -> torch.Tensor:
    result = value.clone()
    if distributed.world_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def _attach_duo_interface(wrapper: DDP, model: DuoModel) -> None:
    """Expose the non-module recipe metadata consumed by ``duo_loss``."""

    # DDP intentionally delegates calls but not arbitrary attributes.  The
    # objective needs these three pieces of the Duo public interface.
    wrapper.config = model.config  # type: ignore[attr-defined]
    wrapper.schedule_eps = model.schedule_eps  # type: ignore[attr-defined]
    wrapper.forward_bos_logits = model.forward_bos_logits  # type: ignore[attr-defined]


def _recipe_batch(batch: TrainingBatch, *, patch_stride: int) -> DuoBatch:
    if batch.document_ids is None:
        raise ValueError("Byte-Duo requires document-isolated byte pages")
    if batch.max_patch_size is not None:
        if batch.patch_offsets is None:
            raise ValueError("entropy-patched Byte-Duo requires patch offsets")
        entropy_metadata = build_duo_clean_patch_metadata(
            batch.valid,
            batch.document_ids,
            batch.patch_offsets,
            max_patch_size=batch.max_patch_size,
        )
        metadata = None
    else:
        if (
            batch.byte_indices is None
            or batch.byte_cu_seqlens is None
            or batch.patch_indices is None
        ):
            raise ValueError("Byte-Duo requires prepacked document metadata")
        entropy_metadata = None
        metadata = compile_stable_document_metadata(
            batch.valid,
            batch.document_ids,
            patch_stride=patch_stride,
            max_segments_per_row=64,
        )
    return DuoBatch(
        clean_ids=batch.ids,
        clean_valid=batch.valid,
        document_ids=batch.document_ids,
        positions=batch.positions,
        ar_targets=batch.ar_targets,
        bos_targets=batch.bos_targets,
        attention_metadata=metadata,
        clean_patch_metadata=entropy_metadata,
        patch_offsets=batch.patch_offsets if entropy_metadata is not None else None,
        max_patch_size=batch.max_patch_size,
        bos_row_indices=batch.bos_row_indices,
    )


def production_batch_geometry(world_size: int) -> tuple[int, int]:
    """Return training/validation local batch limits for supported systems."""

    geometry = {1: (16, 64), 8: (32, 32)}
    if world_size not in geometry:
        raise ValueError("production Byte-Duo supports world size 1 or 8")
    return geometry[world_size]


def production_microbatches(world_size: int) -> tuple[int, ...]:
    """Return eligible microbatches; readiness selects the fastest fitting one."""

    # Twelve and fifteen are measured fallbacks for deeper full-resolution
    # cells on the 32 GiB development GPU.  Keep the larger candidates in the
    # authenticated sweep so shallower cells still select their fastest
    # eligible geometry rather than inheriting a fallback unconditionally.
    # The global batch remains exactly 249 for every candidate.
    candidates = {1: (12, 15, 16, 24, 32), 8: (32,)}
    if world_size not in candidates:
        raise ValueError("production Byte-Duo supports world size 1 or 8")
    return candidates[world_size]


def parse_args() -> argparse.Namespace:
    parser_world_size = int(os.getenv("WORLD_SIZE", "1"))
    try:
        default_microbatch, default_validation_batch = production_batch_geometry(
            parser_world_size
        )
    except ValueError:
        # Non-production distributed smoke tests may use smaller world sizes;
        # the strict 2k gate below still rejects them.
        default_microbatch, default_validation_batch = 16, 64
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path, default=REPO_ROOT / "data" / "byte_diffusion_aligned_v5"
    )
    parser.add_argument("--steps", type=int, default=int(os.getenv("ITERATIONS", "2000")))
    parser.add_argument("--run-name", default=os.getenv("RUN_ID", "byte_duo"))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("BYTE_DUO_MICROBATCH", str(default_microbatch))),
    )
    parser.add_argument("--global-batch-size", type=int, default=249)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--canvas-length", type=int, default=512)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--objective",
        choices=DUO_TRAINING_OBJECTIVES,
        default=os.getenv("BYTE_DUO_OBJECTIVE", PURE_DUO_OBJECTIVE),
        help=(
            "pure NELBO (default) or the fixed equal-weight sum of independently "
            "normalized Duo NELBO and clean causal AR"
        ),
    )
    parser.add_argument("--schedule-eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--log-every", type=int, default=int(os.getenv("TRAIN_LOG_EVERY", "10"))
    )
    parser.add_argument(
        "--val-every", type=int, default=int(os.getenv("VAL_LOSS_EVERY", "20"))
    )
    parser.add_argument("--validation-rows", type=int, default=256)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=int(
            os.getenv("BYTE_DUO_VALIDATION_BATCH", str(default_validation_batch))
        ),
    )
    parser.add_argument(
        "--warmdown-steps", type=int, default=int(os.getenv("WARMDOWN_ITERS", "1200"))
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpoint-interval-seconds", type=float, default=480.0
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--readiness-report",
        type=Path,
        default=os.getenv("BYTE_DUO_READINESS_REPORT"),
    )
    parser.add_argument(
        "--inference-readiness-report",
        type=Path,
        default=os.getenv("BYTE_DUO_INFERENCE_READINESS_REPORT"),
    )
    parser.add_argument("--print-provenance", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def validate_readiness_evidence(
    args: argparse.Namespace,
    *,
    source_sha256: str,
    dataset_payload_sha256: str,
    dataset_patching: DatasetPatchingSpec | None = None,
    parameter_count: int | None = None,
    world_size: int | None = None,
    rank: int = 0,
) -> dict[str, object] | None:
    """Authenticate the systems evidence required by a production 2k run."""

    config = model_config_from_env()
    paths = (args.readiness_report, args.inference_readiness_report)
    if args.steps >= 2_000 and any(path is None for path in paths):
        raise ValueError(
            "2k Byte-Duo runs require training/validation and inference "
            "readiness reports"
        )
    if all(path is None for path in paths):
        return None
    if any(path is None for path in paths):
        raise ValueError("Byte-Duo readiness reports must be supplied together")
    training_path, inference_path = paths
    assert training_path is not None and inference_path is not None
    observed_world_size = (
        int(os.environ.get("WORLD_SIZE", "1"))
        if world_size is None
        else world_size
    )
    workload = {
        **duo_geometry_contract(args.canvas_length, args.branches),
        "diagnostic_cadence": diagnostic_cadence_contract(
            log_every=args.log_every,
            validation_every=args.val_every,
        ),
        **duo_objective_contract(args.objective),
        "schedule_eps": args.schedule_eps,
        "time_sampling": "global_branch_antithetic_striped_uniform_0_1",
        **duo_mutable_topology_contract(config),
    }
    patching_contract = (
        None
        if dataset_patching is None
        else {
            "name": dataset_patching.name,
            "max_patch_size": dataset_patching.max_patch_size,
            "patcher_sha256": dataset_patching.artifact_sha256,
        }
    )
    if patching_contract is not None:
        workload["dataset_patching"] = patching_contract
    if observed_world_size == 1:
        training_evidence = validate_architecture_readiness(
            training_path,
            architecture="duo",
            source_sha256=source_sha256,
            dataset_payload_sha256=dataset_payload_sha256,
            global_batch_size=args.global_batch_size,
            microbatch_size=args.batch_size,
            validation_batch_size=args.validation_batch_size,
            model_config=config.to_dict(),
            workload=workload,
            runtime={
                "compiled": args.compile,
                "device_type": "cpu" if args.cpu else "cuda",
                "validation_rows": args.validation_rows,
                "world_size": 1,
            },
            required_microbatches=production_microbatches(1),
        )
    elif observed_world_size == 8:
        if parameter_count is None:
            raise ValueError("distributed readiness requires the model parameter count")
        training_evidence = validate_duo_distributed_readiness(
            training_path,
            source_sha256=source_sha256,
            dataset_payload_sha256=dataset_payload_sha256,
            model_config=config.to_dict(),
            parameter_count=parameter_count,
            workload=workload,
            rank=rank,
        )
    else:
        raise ValueError("Byte-Duo readiness supports world size 1 or 8")

    if observed_world_size == 1:
        # Dataset/patcher authentication is mandatory for real training (the
        # loader always supplies a DatasetPatchingSpec).  Keeping these
        # expectations absent only supports synthetic readiness-unit fixtures
        # that do not represent a loadable dataset.
        inference_data_sha256 = (
            dataset_payload_sha256 if dataset_patching is not None else None
        )
        inference_evidence = validate_duo_inference_readiness(
            inference_path,
            source_sha256=source_sha256,
            canvas_length=args.canvas_length,
            branches=args.branches,
            model_config=config.to_dict(),
            parameter_count=int(parameter_count or 0),
            dataset_payload_sha256=inference_data_sha256,
            dataset_patching=patching_contract,
        )
    else:
        # Inference readiness is a single-GPU benchmark. Authenticate it once
        # on rank zero (the same physical device used to create that report),
        # then give every training rank the exact same authenticated record.
        payload: list[object] = [None]
        if rank == 0:
            try:
                payload[0] = {
                    "evidence": validate_duo_inference_readiness(
                        inference_path,
                        source_sha256=source_sha256,
                        canvas_length=args.canvas_length,
                        branches=args.branches,
                        model_config=config.to_dict(),
                        parameter_count=int(parameter_count or 0),
                        dataset_payload_sha256=(
                            dataset_payload_sha256
                            if dataset_patching is not None
                            else None
                        ),
                        dataset_patching=patching_contract,
                    )
                }
            except Exception as error:  # propagate before any rank can continue
                payload[0] = {"error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        if not isinstance(result, dict):
            raise ValueError("rank zero returned invalid inference readiness evidence")
        if "error" in result:
            raise ValueError(f"inference readiness failed on rank zero: {result['error']}")
        inference_evidence = result.get("evidence")
        if not isinstance(inference_evidence, dict):
            raise ValueError("rank zero omitted inference readiness evidence")
    return {
        "training_validation": training_evidence,
        "inference": inference_evidence,
    }


def _validation_batches(
    dataset,
    *,
    rows: int,
    batch_size: int,
    patch_stride: int,
    device: torch.device,
    row_indices: np.ndarray | None = None,
):
    rows = min(rows, len(dataset))
    indices = (
        np.arange(rows, dtype=np.int64)
        if row_indices is None
        else np.asarray(row_indices, dtype=np.int64)
    )
    if indices.ndim != 1 or bool(((indices < 0) | (indices >= rows)).any()):
        raise ValueError("validation row indices lie outside the declared ledger")
    if np.unique(indices).size != indices.size:
        raise ValueError("validation row indices must be unique")
    native = getattr(dataset, "training_batch", None)
    for start in range(0, indices.size, batch_size):
        batch_indices = indices[start : start + batch_size]
        if native is None:
            raise TypeError("Byte-Duo validation requires the vectorized native batch API")
        cpu_batch = native(batch_indices)
        row_ids = torch.from_numpy(batch_indices.copy())
        if device.type == "cuda":
            cpu_batch = cpu_batch.pin_memory()
            row_ids = row_ids.pin_memory()
        indexed = replace(
            _recipe_batch(cpu_batch, patch_stride=patch_stride),
            row_ids=row_ids,
        )
        yield indexed.to(
            device, non_blocking=device.type == "cuda"
        )


def _prepared_validation_ledger(
    model: DuoModel,
    dataset,
    *,
    rows: int,
    batch_size: int,
    patch_stride: int,
    canvas_length: int,
    branches: int,
    seed: int,
    schedule: DuoSchedule,
    device: torch.device,
    row_indices: np.ndarray,
    expose_random_phase: bool = False,
) -> tuple[PreparedDuoValidationBatch, ...]:
    """Build/check the immutable validation corruption once, then cache it."""

    def prepare(batch: DuoBatch) -> PreparedDuoValidationBatch:
        item = prepare_duo_validation_batch(
            model,
            batch,
            total_rows=rows,
            canvas_length=canvas_length,
            branches=branches,
            seed=seed,
            schedule=schedule,
            expose_random_phase=expose_random_phase,
        )
        if device.type == "cuda":
            item = item.pin_memory().to(device, non_blocking=True)
        return item

    ledger = tuple(
        map(
            prepare,
            _validation_batches(
                dataset,
                rows=rows,
                batch_size=batch_size,
                patch_stride=patch_stride,
                device=torch.device("cpu"),
                row_indices=row_indices,
            ),
        )
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return ledger


def _run(
    args: argparse.Namespace,
    provenance: dict[str, object],
    distributed: DistributedContext,
    device: torch.device,
) -> None:
    clean_ar_weight = duo_clean_ar_weight(args.objective)
    ar_bpb_role = (
        "joint_training_objective"
        if args.objective == JOINT_DUO_CLEAN_AR_OBJECTIVE
        else "zero_weight_diagnostic_only"
    )
    if args.steps <= 0 or args.batch_size <= 0 or args.global_batch_size <= 0:
        raise ValueError("steps and batch sizes must be positive")
    if args.steps >= 2_000 and args.global_batch_size != 249:
        raise ValueError("matched 2k Byte-Duo runs require global batch 249")
    try:
        geometry_contract = duo_geometry_contract(
            args.canvas_length, args.branches
        )
    except ValueError:
        if args.steps >= 2_000:
            raise
        geometry_contract = None
    if args.steps >= 2_000 and args.schedule_eps != 1e-3:
        raise ValueError(
            "production Byte-Duo geometries require schedule eps=1e-3"
        )
    if args.canvas_length <= 0 or args.branches <= 0:
        raise ValueError("canvas geometry must be positive")
    if not 0 <= args.warmdown_steps <= args.steps:
        raise ValueError("warmdown steps must lie in [0, steps]")
    if args.max_grad_norm <= 0:
        raise ValueError("max grad norm must be positive")
    microsteps_per_rank = distributed_microstep_count(
        args.global_batch_size,
        world_size=distributed.world_size,
        microbatch_size=args.batch_size,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    expected_source = os.environ.get("BYTE_DUO_EXPECTED_SOURCE_SHA256")
    if args.steps >= 2_000 and expected_source is None:
        raise ValueError("2k Byte-Duo runs require BYTE_DUO_EXPECTED_SOURCE_SHA256")
    if expected_source is not None and expected_source != provenance["sha256"]:
        raise ValueError(
            "Byte-Duo source differs from the pinned contract: "
            f"expected {expected_source}, observed {provenance['sha256']}"
        )
    expected_data = os.environ.get("BYTE_DIFFUSION_EXPECTED_DATA_SHA256")
    if args.steps >= 2_000 and expected_data is None:
        raise ValueError("2k Byte-Duo runs require BYTE_DIFFUSION_EXPECTED_DATA_SHA256")

    manifest, train_chunks, validation_chunks = load_data_directory(
        args.data_path,
        chunk_size=args.chunk_size,
        recipe="blt_d",
        required_branch_bytes=DUO_DATA_REQUIRED_BRANCH_BYTES,
        branch_span_length=DUO_DATA_BRANCH_SPAN_LENGTH,
        validation_chunk_limit=args.validation_rows,
        expected_payload_sha256=expected_data,
    )
    dataset_manifest = json.loads((args.data_path / "manifest.json").read_text())
    config = model_config_from_env()
    dataset_patching = getattr(train_chunks, "patching", None)
    validation_patching = getattr(validation_chunks, "patching", None)
    if not isinstance(dataset_patching, DatasetPatchingSpec):
        raise ValueError("Byte-Duo dataset omitted its authenticated patching policy")
    if dataset_patching != validation_patching:
        raise ValueError("training and validation patching policies differ")
    if dataset_patching.name != config.duo_clean_patching:
        raise ValueError(
            "dataset patching policy differs from the Duo clean hierarchy: "
            f"expected {config.duo_clean_patching}, observed {dataset_patching.name}"
        )
    entropy_patcher_bytes = _authenticated_entropy_patcher_bytes(
        args.data_path, dataset_manifest, dataset_patching
    )
    train_exposure = dataset_manifest.get("splits", {}).get("train", {})
    special_targets = int(train_exposure.get("special_atomic_tokens", -1))
    eot_targets = int(train_exposure.get("eot_atomic_tokens", -2))
    unused_controls = tuple(
        special.name
        for special in manifest.specials
        if special.atomic_id != manifest.eot_id
    )
    allow_unused_controls_value = os.environ.get(
        "BYTE_DUO_ALLOW_UNUSED_CONTROLS", "0"
    )
    if allow_unused_controls_value not in {"0", "1"}:
        raise ValueError("BYTE_DUO_ALLOW_UNUSED_CONTROLS must be 0 or 1")
    allow_unused_controls = allow_unused_controls_value == "1"
    if (
        args.steps >= 2_000
        and config.duo_diffusion_atoms > manifest.eot_id + 1
        and unused_controls
        and special_targets == eot_targets
        and not allow_unused_controls
    ):
        raise ValueError(
            "2k Byte-Duo runs refuse a diffusion prior with never-positive "
            f"controls {unused_controls}; inject typed controls into pretraining "
            "or rebuild a bytes+EOT vocabulary"
        )
    try:
        _, production_validation_batch = production_batch_geometry(distributed.world_size)
        allowed_microbatches = production_microbatches(distributed.world_size)
    except ValueError:
        allowed_microbatches = None
        production_validation_batch = None
    if args.steps >= 2_000 and (
        device.type != "cuda"
        or allowed_microbatches is None
        or production_validation_batch is None
        or args.batch_size not in allowed_microbatches
        or not args.compile
        or args.chunk_size != 8_192
        or geometry_contract is None
        or args.schedule_eps != 0.001
        or args.validation_rows != 256
        or args.validation_batch_size != production_validation_batch
    ):
        raise ValueError("2k Byte-Duo runs require the benchmarked production contract")
    if (
        manifest.output_size != config.vocab.output_size
        or manifest.pad_id != config.vocab.pad_id
        or manifest.eot_id != config.vocab.eot_id
    ):
        raise ValueError("dataset vocabulary and Byte-Duo model differ")
    model = DuoModel(config, schedule_eps=args.schedule_eps).to(device)
    if config == type(config)():
        model.validate_production_parameterization()
    if args.steps >= 2_000:
        # Every ablation must be potentially shippable before spending the
        # complete training budget. Count the same executable closure as final
        # export; reserve 128 KiB for manifest, provenance and metric metadata.
        code_bytes = sum(path.stat().st_size for path in counted_duo_code_paths())
        size = artifact_size_report(
            model,
            config,
            code_bytes=code_bytes + 128 * 1024,
            entropy_patcher=entropy_patcher_bytes,
        )
        if size.complete_bytes > ARTIFACT_CAP_BYTES:
            raise ValueError(
                "2k Byte-Duo variant cannot fit the complete 16 MB artifact: "
                f"{size.complete_bytes:,} bytes"
            )
    dynamic_compile = config.duo_clean_patching == "causal_entropy_v1"
    execution_model: torch.nn.Module = (
        torch.compile(model, dynamic=dynamic_compile, fullgraph=False)
        if device.type == "cuda" and args.compile
        else model
    )
    if distributed.world_size > 1:
        # Validation is sharded across ranks and reduces only additive metrics.
        # Training forwards therefore must not introduce an unrelated buffer
        # broadcast collective. Parameter/buffer initialization still
        # synchronizes once in the DDP constructor.
        ddp_kwargs: dict[str, object] = {"forward_sync_buffers": False}
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[device.index], output_device=device.index)
        forward_model: torch.nn.Module = DDP(execution_model, **ddp_kwargs)
        _attach_duo_interface(forward_model, model)  # type: ignore[arg-type]
    else:
        forward_model = execution_model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=device.type == "cuda",
    )
    cursor = DeterministicChunkCursor(train_chunks, seed=args.seed, shuffle=True)
    # Canvas ledgers are built on the CPU next to the memory-mapped pages, then
    # only compact branches are transferred. This avoids copying
    # the full 249x8,192 update twice before microbatch training.
    time_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    # Preserve the original single-device RNG stream exactly: time sampling,
    # origin selection, and corruption consume one generator in the same order.
    # Distributed ranks need independent corruption streams, while every rank
    # reproduces the same global antithetic-time ledger before disjoint sharding.
    corruption_generator = (
        time_generator
        if distributed.world_size == 1
        else torch.Generator(device="cpu").manual_seed(
            args.seed + 1 + 1_000_003 * distributed.rank
        )
    )
    schedule = DuoSchedule(args.schedule_eps)
    objective_fn = (
        compiled_duo_nelbo_token_loss
        if device.type == "cuda" and args.compile
        else duo_nelbo_token_loss
    )
    output_dir = REPO_ROOT / "ablation_results" / args.run_name
    if distributed.is_primary:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    metrics_path = output_dir / (
        "native_metrics.jsonl"
        if os.environ.get("ABLATION_RUNNER_OWNS_METRICS") == "1"
        else "metrics.jsonl"
    )
    checkpoint_path = args.checkpoint or output_dir / "checkpoint.pt"
    dataset_payload_sha256 = dataset_manifest.get("payload_sha256")
    if expected_data is not None and dataset_payload_sha256 != expected_data:
        raise ValueError("dataset payload differs from the pinned Byte-Duo contract")
    readiness_evidence = validate_readiness_evidence(
        args,
        source_sha256=str(provenance["sha256"]),
        dataset_payload_sha256=str(dataset_payload_sha256),
        dataset_patching=dataset_patching,
        parameter_count=model.parameter_count,
        world_size=distributed.world_size,
        rank=distributed.rank,
    )

    training_contract = {
        "steps": args.steps,
        "microbatch_size": args.batch_size,
        "global_batch_size": args.global_batch_size,
        "chunk_size": args.chunk_size,
        "canvas_length": args.canvas_length,
        "branches": args.branches,
        **(
            geometry_contract
            if geometry_contract is not None
            else {
                "training_geometry": {
                    "canvas_length": args.canvas_length,
                    "branches": args.branches,
                },
                "canonical_validation_geometry": {
                    "canvas_length": DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
                    "branches": DUO_CANONICAL_VALIDATION_BRANCHES,
                },
                "dataset_geometry": {
                    "required_branch_bytes": DUO_DATA_REQUIRED_BRANCH_BYTES,
                    "branch_span_length": DUO_DATA_BRANCH_SPAN_LENGTH,
                },
            }
        ),
        "learning_rate": args.learning_rate,
        **duo_objective_contract(args.objective),
        "schedule_eps": args.schedule_eps,
        "random_phase_training": config.duo_random_phase_training,
        "canonical_validation_origin_stride": config.patch_stride,
        "serving_distribution_validation_origin_stride": config.duo_origin_stride,
        **duo_mutable_topology_contract(config),
        "allow_unused_diffusion_controls": allow_unused_controls,
        "time_sampling": "global_branch_antithetic_striped_uniform_0_1",
        "warmdown_steps": args.warmdown_steps,
        "max_grad_norm": args.max_grad_norm,
        "compile": args.compile,
        "compile_dynamic_shapes": dynamic_compile,
        "dataset_patching": {
            "name": dataset_patching.name,
            "max_patch_size": dataset_patching.max_patch_size,
            "patcher_sha256": dataset_patching.artifact_sha256,
        },
        "diagnostic_cadence": diagnostic_cadence_contract(
            log_every=args.log_every,
            validation_every=args.val_every,
        ),
        "seed": args.seed,
        "log_every": args.log_every,
        "validation_every": args.val_every,
        "validation_rows": args.validation_rows,
        "validation_batch_size": args.validation_batch_size,
        "validation_seed": args.seed + 10_000,
        "checkpoint_interval_seconds": args.checkpoint_interval_seconds,
        "world_size": distributed.world_size,
        "base_rank_row_counts": distributed_local_counts(
            args.global_batch_size, distributed.world_size
        ),
        "rank_remainder_rotation": "(step - 1) modulo world_size",
        "microsteps_per_rank": microsteps_per_rank,
        "readiness_evidence": readiness_evidence,
    }
    contract = {
        "schema": "byte_duo_run/v1",
        "run_name": args.run_name,
        "model": config.to_dict(),
        "parameter_count": model.parameter_count,
        "estimated_quantized_artifact_bytes": model.estimated_quantized_artifact_bytes,
        "source": provenance,
        "atomic_manifest_sha256": manifest.sha256,
        "dataset_payload_sha256": dataset_payload_sha256,
        "train_dataset_sha256": getattr(train_chunks, "dataset_sha256", None),
        "validation_dataset_sha256": getattr(validation_chunks, "dataset_sha256", None),
        "objective": args.objective,
        "objective_contract": duo_objective_contract(args.objective),
        "prior": (
            f"uniform_over_{config.duo_diffusion_atoms}_diffusion_atomic_ids"
        ),
        "mask_state": False,
        "within_family_promotion_metric": (
            "conditional_canvas_duo_nelbo_nats_per_atom"
        ),
        "across_family_promotion_metric": "gsm8k_exact_match_generation_accuracy",
        "nelbo_is_ar_bpb": False,
        "ar_bpb_role": ar_bpb_role,
        "training": training_contract,
        "world_size": distributed.world_size,
        "distributed_exact_global_batch": True,
    }
    if distributed.is_primary:
        (output_dir / "contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n"
        )
        (output_dir / "source_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        print("byte_duo_contract " + json.dumps(contract, sort_keys=True), flush=True)
    distributed.barrier()

    completed_steps = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("schema") != "byte_duo_checkpoint/v2":
            raise ValueError("unsupported Byte-Duo checkpoint schema")
        if checkpoint.get("source_sha256") != provenance["sha256"]:
            raise ValueError("resume checkpoint source provenance differs")
        if checkpoint.get("dataset_payload_sha256") != dataset_payload_sha256:
            raise ValueError("resume checkpoint dataset provenance differs")
        if checkpoint.get("model_config") != config.to_dict():
            raise ValueError("resume checkpoint model configuration differs")
        if checkpoint.get("training") != training_contract:
            raise ValueError("resume checkpoint training contract differs")
        if checkpoint.get("world_size") != distributed.world_size:
            raise ValueError("resume checkpoint WORLD_SIZE differs")
        if checkpoint.get("rank_checkpoint_schema") != "byte_duo_rank_checkpoint/v1":
            raise ValueError("resume checkpoint rank-sidecar schema differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed_steps = int(checkpoint["steps"])
        rank_path = rank_checkpoint_path(
            args.resume, distributed.rank, step=completed_steps
        )
        rank_checkpoint = torch.load(
            rank_path, map_location=device, weights_only=False
        )
        expected_rank_fields = {
            "schema": "byte_duo_rank_checkpoint/v1",
            "rank": distributed.rank,
            "world_size": distributed.world_size,
            "steps": completed_steps,
            "source_sha256": provenance["sha256"],
            "dataset_payload_sha256": dataset_payload_sha256,
        }
        for key, expected in expected_rank_fields.items():
            if rank_checkpoint.get(key) != expected:
                raise ValueError(f"resume rank checkpoint {key} differs")
        restore_rank_state(
            rank_checkpoint,
            time_generator,
            corruption_generator,
            cursor=cursor,
            device=device,
        )
        distributed.barrier()

    checkpoint_policy = RecoveryCheckpointPolicy(
        args.checkpoint_interval_seconds
    )
    committed_host_state: dict[str, object] | None = None

    def save_checkpoint(step: int) -> None:
        # Sidecars are step-versioned and published before the main checkpoint.
        # If a rank or host dies mid-save, the prior main checkpoint still
        # points (by its completed step) to untouched, mutually consistent
        # sidecars instead of a mixture of old model state and new RNG state.
        rank_path = rank_checkpoint_path(
            checkpoint_path, distributed.rank, step=step
        )
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        rank_payload = {
            "schema": "byte_duo_rank_checkpoint/v1",
            "rank": distributed.rank,
            "world_size": distributed.world_size,
            "steps": step,
            "source_sha256": provenance["sha256"],
            "dataset_payload_sha256": dataset_payload_sha256,
            **capture_rank_state(
                time_generator,
                corruption_generator,
                cursor=cursor,
                device=device,
                prepared_host_state=committed_host_state,
            ),
        }
        rank_temporary = rank_path.with_suffix(rank_path.suffix + ".working")
        torch.save(rank_payload, rank_temporary)
        rank_temporary.replace(rank_path)
        distributed.barrier()
        if distributed.is_primary:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "byte_duo_checkpoint/v2",
                "model_config": config.to_dict(),
                "training": training_contract,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "steps": step,
                "world_size": distributed.world_size,
                "rank_checkpoint_schema": "byte_duo_rank_checkpoint/v1",
                "source_sha256": provenance["sha256"],
                "dataset_payload_sha256": dataset_payload_sha256,
            }
            temporary = checkpoint_path.with_suffix(
                checkpoint_path.suffix + ".working"
            )
            torch.save(payload, temporary)
            temporary.replace(checkpoint_path)
        distributed.barrier()
        prune_stale_rank_checkpoints(
            checkpoint_path, distributed.rank, committed_step=step
        )
        distributed.barrier()
        checkpoint_policy.committed(step)

    validation_rows = min(args.validation_rows, len(validation_chunks))
    rank_validation_rows = distributed_rank_positions(
        validation_rows,
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    validation_ledger = _prepared_validation_ledger(
        model,
        validation_chunks,
        rows=validation_rows,
        batch_size=args.validation_batch_size,
        patch_stride=config.patch_stride,
        canvas_length=DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
        branches=DUO_CANONICAL_VALIDATION_BRANCHES,
        seed=args.seed + 10_000,
        schedule=schedule,
        device=device,
        row_indices=rank_validation_rows,
    )
    started = time.perf_counter()
    validation_seconds = 0.0
    metrics = (
        metrics_path.open("a", encoding="utf-8")
        if distributed.is_primary
        else None
    )
    batch_prefetcher = DeviceBatchPrefetcher(device=device)
    try:
        def emit(record: dict[str, object], *, prefix: str) -> None:
            if metrics is None:
                raise RuntimeError("only the primary rank may emit Byte-Duo metrics")
            line = json.dumps(record, sort_keys=True, allow_nan=False)
            metrics.write(line + "\n")
            metrics.flush()
            print(prefix + " " + line, flush=True)

        def validate(step: int) -> None:
            nonlocal validation_seconds
            # Every rank evaluates a disjoint slice through the unwrapped
            # compiled model. There are no per-forward collectives, so uneven
            # final microbatches cannot deadlock. One six-scalar reduction is
            # the synchronization boundary before root-only metric emission.
            distributed.barrier()
            validation_started = time.perf_counter()
            local_stats = accumulate_duo_validation_stats(
                execution_model,  # type: ignore[arg-type]
                validation_ledger,
                canvas_length=DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
                branches=DUO_CANONICAL_VALIDATION_BRANCHES,
                seed=args.seed + 10_000,
                schedule=schedule,
                objective_fn=objective_fn,
                total_rows=validation_rows,
                expected_row_ids=torch.from_numpy(rank_validation_rows.copy()),
                compute_ar_diagnostic=(
                    clean_ar_weight != 0.0 or step == args.steps
                ),
            )
            global_stats = _all_reduce_sum(local_stats, distributed)
            validation_seconds += time.perf_counter() - validation_started
            if distributed.is_primary:
                validation = duo_validation_from_stats(global_stats)
                validation_optimizer_objective = (
                    validation.conditional_canvas_nelbo_nats_per_atom
                    + clean_ar_weight * (validation.clean_ar_nats_per_atom or 0.0)
                )
                emit(
                    {
                        "schema": "byte_duo_validation/v1",
                        "kind": "validation",
                        "step": step,
                        "conditional_canvas_duo_nelbo_nats_per_atom": (
                            validation.conditional_canvas_nelbo_nats_per_atom
                        ),
                        "clean_ar_nats_per_atom": validation.clean_ar_nats_per_atom,
                        "normalized_optimizer_objective": (
                            validation_optimizer_objective
                        ),
                        "objective": args.objective,
                        "clean_ar_weight": clean_ar_weight,
                        "ar_anchor_bpb": validation.ar_anchor_bpb,
                        "denoising_accuracy": validation.denoising_accuracy,
                        "diffusion_targets": validation.targets,
                        "changed_targets": validation.changed_targets,
                        "ar_targets": validation.ar_targets,
                        "within_family_promotion_metric": (
                            "conditional_canvas_duo_nelbo_nats_per_atom"
                        ),
                        "across_family_promotion_metric": (
                            "gsm8k_exact_match_generation_accuracy"
                        ),
                        "nelbo_is_ar_bpb": False,
                        "ar_bpb_role": ar_bpb_role,
                        "evaluation_geometry": "canonical_512x8",
                        "evaluation_canvas_length": (
                            DUO_CANONICAL_VALIDATION_CANVAS_LENGTH
                        ),
                        "evaluation_branches": DUO_CANONICAL_VALIDATION_BRANCHES,
                    },
                    prefix="byte_duo_val",
                )
                nelbo_nats = validation.conditional_canvas_nelbo_nats_per_atom
                nelbo_bits = nelbo_nats / np.log(2.0)
                training_time_ms = (
                    time.perf_counter() - started - validation_seconds
                ) * 1_000
                ar_extra = (
                    ""
                    if validation.ar_anchor_bpb is None
                    else f" val_ar_anchor_bpb:{validation.ar_anchor_bpb:.6f}"
                )
                print(
                    f"step:{step}/{args.steps} "
                    f"val_loss:{validation_optimizer_objective:.6f} "
                    f"val_diffusion_nelbo_bits_per_atom:{nelbo_bits:.6f}"
                    f"{ar_extra} train_time:{training_time_ms:.3f}ms",
                    flush=True,
                )
            distributed.barrier()

        if completed_steps == 0:
            validate(0)

        def prepare_update(step: int) -> PreparedDuoUpdate:
            global_indices = cursor.next_indices(args.global_batch_size)
            rank_positions = distributed_rank_positions(
                args.global_batch_size,
                rank=distributed.rank,
                world_size=distributed.world_size,
                rotation=step - 1,
            )
            update_indices = global_indices[rank_positions]
            native_update = train_chunks.training_batch(update_indices)
            update_batch = _recipe_batch(native_update, patch_stride=config.patch_stride)
            global_times = sample_branch_antithetic_times(
                args.global_batch_size,
                args.branches,
                device="cpu",
                generator=time_generator,
            )
            update_times = global_times.index_select(
                0, torch.from_numpy(rank_positions)
            )
            prepared_update = prepare_duo_update(
                model,
                update_batch,
                microbatch_size=args.batch_size,
                canvas_length=args.canvas_length,
                branches=args.branches,
                schedule=schedule,
                times=update_times,
                generator=corruption_generator,
                include_clean_ar=clean_ar_weight != 0.0,
            )
            return replace(
                prepared_update,
                host_state={
                    "time_generator_state": time_generator.get_state(),
                    "corruption_generator_state": corruption_generator.get_state(),
                    "cursor": cursor.state_dict(),
                },
            )

        update_steps = range(completed_steps + 1, args.steps + 1)
        prefetched_updates = batch_prefetcher.batches(update_steps, prepare_update)
        for step, prepared_update in zip(
            update_steps, prefetched_updates, strict=True
        ):
            if prepared_update.host_state is None:
                raise AssertionError("prefetched Duo update omitted resume state")
            committed_host_state = prepared_update.host_state
            local_denominators = torch.stack(
                (
                    prepared_update.nelbo_denominator,
                    prepared_update.ar_denominator,
                )
            )
            global_denominators = _all_reduce_sum(
                local_denominators, distributed
            )
            nelbo_denominator = global_denominators[0]
            ar_denominator = (
                global_denominators[1]
                if clean_ar_weight != 0.0
                else global_denominators.new_ones(())
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            materialize_diagnostics = materialize_training_diagnostics(
                step,
                args.steps,
                log_every=args.log_every,
                validation_every=args.val_every,
            )
            totals = (
                torch.zeros(5, dtype=torch.float64, device=device)
                if materialize_diagnostics
                else None
            )
            microsteps = 0
            for item_index, item in enumerate(prepared_update.microbatches):
                synchronize = item_index + 1 == len(prepared_update.microbatches)
                sync_context = (
                    forward_model.no_sync()
                    if isinstance(forward_model, DDP) and not synchronize
                    else nullcontext()
                )
                with sync_context:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        loss = duo_loss(
                            forward_model,  # type: ignore[arg-type]
                            item.batch,
                            canvas_length=args.canvas_length,
                            branches=args.branches,
                            prepared=item.prepared,
                            schedule=schedule,
                            nelbo_denominator=nelbo_denominator,
                            ar_denominator=ar_denominator,
                            clean_ar_weight=clean_ar_weight,
                            objective_fn=objective_fn,
                        )
                        scaled_loss = loss.total * distributed_loss_scale(
                            distributed.world_size
                        )
                    scaled_loss.backward()
                if totals is not None:
                    totals += torch.stack(
                        (
                            loss.nelbo_sum.detach().double(),
                            loss.nelbo_targets.detach().double(),
                            loss.ar_nll_sum.detach().double(),
                            loss.ar_targets.detach().double(),
                            loss.corruption.changed.sum().detach().double(),
                        )
                    )
                microsteps += 1
            if microsteps != microsteps_per_rank:
                raise AssertionError("rank executed the wrong number of backward calls")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            warmdown_start = args.steps - args.warmdown_steps
            lr_scale = (
                1.0
                if args.warmdown_steps == 0 or step - 1 < warmdown_start
                else (args.steps - (step - 1)) / args.warmdown_steps
            )
            learning_rate = args.learning_rate * lr_scale
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            if step % args.log_every == 0 or step == args.steps:
                if totals is None:
                    raise AssertionError("logging update omitted training diagnostics")
                global_totals = _all_reduce_sum(totals, distributed)
                elapsed = time.perf_counter() - started - validation_seconds
                targets = int(global_totals[1].item())
                ar_targets = int(global_totals[3].item())
                if distributed.is_primary:
                    train_nelbo = float(global_totals[0] / max(targets, 1))
                    train_clean_ar = (
                        None
                        if ar_targets == 0
                        else float(global_totals[2] / ar_targets)
                    )
                    train_optimizer_objective = (
                        train_nelbo + clean_ar_weight * (train_clean_ar or 0.0)
                    )
                    emit(
                        {
                            "schema": "byte_duo_train/v1",
                            "kind": "train",
                            "step": step,
                            "duo_nelbo_nats_per_atom": float(
                                global_totals[0] / max(targets, 1)
                            ),
                            "clean_ar_nats_per_atom": train_clean_ar,
                            "normalized_optimizer_objective": (
                                train_optimizer_objective
                            ),
                            "objective": args.objective,
                            "clean_ar_weight": clean_ar_weight,
                            "ar_anchor_bpb": (
                                None
                                if ar_targets == 0
                                else float(
                                    global_totals[2] / ar_targets / np.log(2.0)
                                )
                            ),
                            "ar_bpb_role": ar_bpb_role,
                            "diffusion_targets": targets,
                            "changed_targets": int(global_totals[4].item()),
                            "ar_targets": ar_targets,
                            "microsteps_per_rank": microsteps,
                            "rank_row_counts": tuple(
                                distributed_rank_positions(
                                    args.global_batch_size,
                                    rank=rank,
                                    world_size=distributed.world_size,
                                    rotation=step - 1,
                                ).size
                                for rank in range(distributed.world_size)
                            ),
                            "world_size": distributed.world_size,
                            "global_batch_size": args.global_batch_size,
                            "learning_rate": learning_rate,
                            "grad_norm": float(grad_norm),
                            "session_updates": step - completed_steps,
                            "session_elapsed_seconds": elapsed,
                            "steps_per_second": (step - completed_steps) / elapsed,
                            "parameters": model.parameter_count,
                        },
                        prefix="byte_duo_train",
                    )
                    print(
                        f"step:{step}/{args.steps} "
                        f"train_loss:{train_optimizer_objective:.6f} "
                        f"train_time:{elapsed * 1_000:.3f}ms "
                        f"duo_nelbo_nats_per_atom:{train_nelbo:.6f} "
                        f"duo_nelbo_bits_per_atom:"
                        f"{train_nelbo / np.log(2.0):.6f}",
                        flush=True,
                    )
            if step % args.val_every == 0 or step == args.steps:
                validate(step)
            checkpoint_due = (
                checkpoint_policy.due()
                if distributed.is_primary
                else False
            )
            if distributed.world_size > 1:
                due_flag = torch.tensor(
                    int(checkpoint_due), device=device, dtype=torch.int32
                )
                dist.broadcast(due_flag, src=0)
                checkpoint_due = bool(due_flag.item())
            if checkpoint_due or (
                step == args.steps and checkpoint_policy.terminal_due(step)
            ):
                save_checkpoint(step)
    finally:
        batch_prefetcher.close()
        if metrics is not None:
            metrics.close()
    distributed.barrier()

    result = {
        "schema": "byte_duo_result/v1",
        "run_name": args.run_name,
        "steps": args.steps,
        "world_size": distributed.world_size,
        "global_batch_size": args.global_batch_size,
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "metrics": str(metrics_path.relative_to(REPO_ROOT)),
        "within_family_promotion_metric": (
            "conditional_canvas_duo_nelbo_nats_per_atom"
        ),
        "across_family_promotion_metric": "gsm8k_exact_match_generation_accuracy",
        "diagnostic_metric": "conditional_canvas_duo_nelbo_nats_per_atom",
        "objective": args.objective,
        "objective_contract": duo_objective_contract(args.objective),
        "ar_anchor_metric": (
            "trained_clean_causal_ar_bos_bpb"
            if clean_ar_weight != 0.0
            else "zero_weight_clean_ar_bos_bpb"
        ),
        "diffusion_bpb_available": False,
        "reason": "Duo NELBO is a likelihood upper bound, not AR teacher-forced BPB",
    }
    if distributed.is_primary:
        result_name = (
            "native_result.json"
            if os.environ.get("ABLATION_RUNNER_OWNS_METRICS") == "1"
            else "result.json"
        )
        (output_dir / result_name).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    distributed.barrier()


def main() -> None:
    args = parse_args()
    provenance = source_provenance()
    if args.print_provenance:
        print(json.dumps(provenance, indent=2, sort_keys=True))
        return
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required unless --cpu is explicit")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = (
        torch.device("cpu")
        if args.cpu
        else torch.device("cuda", local_rank)
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    distributed = DistributedContext.from_environment(device.type)
    try:
        _run(args, provenance, distributed, device)
    finally:
        distributed.close()


if __name__ == "__main__":
    main()

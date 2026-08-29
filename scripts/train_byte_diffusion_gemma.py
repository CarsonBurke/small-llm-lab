#!/usr/bin/env python3
"""Train the standalone reference-aligned DiffusionGemma cell.

This entry point uses document-packed FlashAttention for the causal clean bank
and shared-bank FlexAttention for every noisy canvas. Submit it through
``mlq`` like every other model workload.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from checkpointing import RecoveryCheckpointPolicy, atomic_torch_save


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.config import model_config_from_env
from pretraining.byte_diffusion.data import DeterministicChunkCursor
from pretraining.byte_diffusion.diffusion_gemma_model import (
    DiffusionGemmaAttentionMetadata,
    DiffusionGemmaModel,
    packed_patch_cu_seqlens,
)
from pretraining.byte_diffusion.diffusion_gemma import sample_static_half_batch_mask
from pretraining.byte_diffusion.readiness import (
    diagnostic_cadence_contract,
    materialize_training_diagnostics,
    validate_architecture_readiness,
)
from pretraining.byte_diffusion.pipeline import DeviceBatchPrefetcher
from pretraining.byte_diffusion.training import (
    TrainingBatch,
    load_data_directory,
)
from pretraining.byte_diffusion.training_diffusion_gemma import (
    DiffusionGemmaBatch,
    PreparedDiffusionGemmaValidationBatch,
    diffusion_gemma_loss,
    prepare_diffusion_gemma_inputs,
    validate_diffusion_gemma,
)


def _local_imports(path: Path) -> tuple[Path, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    candidates: set[Path] = set()

    def add_module(parts: list[str]) -> None:
        if not parts:
            return
        module = REPO_ROOT.joinpath(*parts)
        for candidate in (module.with_suffix(".py"), module / "__init__.py"):
            if candidate.is_file():
                candidates.add(candidate.resolve())

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


def source_provenance() -> dict[str, object]:
    """Fingerprint the transitive source closure defining this cell."""

    pending = [Path(__file__).resolve()]
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        if not path.is_relative_to(REPO_ROOT):
            raise ValueError(f"DiffusionGemma source escaped repository: {path}")
        observed.add(path)
        pending.extend(
            imported for imported in _local_imports(path) if imported not in observed
        )
    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for path in sorted(observed, key=lambda value: value.relative_to(REPO_ROOT).as_posix()):
        relative = path.relative_to(REPO_ROOT).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative] = file_hash
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
    return {
        "schema": "byte_diffusion_gemma_source_provenance/v1",
        "sha256": digest.hexdigest(),
        "files": files,
    }


def _recipe_batch(
    batch: TrainingBatch, *, patch_stride: int
) -> DiffusionGemmaBatch:
    if batch.ids.shape[1] % patch_stride:
        raise ValueError("DiffusionGemma physical rows must be patch aligned")
    if batch.document_ids is None:
        raise ValueError("DiffusionGemma requires document-isolated byte pages")
    if (
        batch.byte_indices is None
        or batch.byte_cu_seqlens is None
        or batch.patch_indices is None
    ):
        raise ValueError("DiffusionGemma requires prepacked document metadata")
    metadata = DiffusionGemmaAttentionMetadata(
        byte_indices=batch.byte_indices,
        byte_cu_seqlens=batch.byte_cu_seqlens,
        patch_indices=batch.patch_indices,
        # `TrainingBatch.patch_cu_seqlens` is the global BLT sequence and may
        # include synthetic BOS patches. DiffusionGemma packs only the physical
        # `patch_indices`, so derive matching document CUs from that exact bank.
        patch_cu_seqlens=packed_patch_cu_seqlens(
            batch.patch_indices,
            batch.valid,
            batch.document_ids,
            patch_stride=patch_stride,
        ),
    )
    return DiffusionGemmaBatch(
        clean_ids=batch.ids,
        clean_valid=batch.valid,
        document_ids=batch.document_ids,
        positions=batch.positions,
        ar_targets=batch.ar_targets,
        bos_targets=batch.bos_targets,
        attention_metadata=metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data" / "byte_diffusion_aligned_v5",
    )
    parser.add_argument("--steps", type=int, default=int(os.getenv("ITERATIONS", "2000")))
    parser.add_argument("--run-name", default=os.getenv("RUN_ID", "diffusion_gemma"))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("BYTE_DIFFUSION_GEMMA_MICROBATCH", "24")),
    )
    parser.add_argument("--global-batch-size", type=int, default=249)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--canvas-length", type=int, default=512)
    parser.add_argument("--branches", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clean-ar-weight", type=float, default=1.0)
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
        default=int(os.getenv("BYTE_DIFFUSION_GEMMA_VALIDATION_BATCH", "24")),
    )
    parser.add_argument(
        "--warmdown-steps", type=int, default=int(os.getenv("WARMDOWN_ITERS", "1200"))
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--checkpoint-interval-seconds", type=float, default=480.0
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--readiness-report",
        type=Path,
        default=os.getenv("BYTE_DIFFUSION_GEMMA_READINESS_REPORT"),
    )
    parser.add_argument("--print-provenance", action="store_true")
    parser.add_argument("--exact-k", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _validation_batches(
    dataset,
    *,
    rows: int,
    batch_size: int,
    patch_stride: int,
    device: torch.device,
):
    rows = min(rows, len(dataset))
    native = getattr(dataset, "training_batch", None)
    if native is None:
        raise TypeError(
            "DiffusionGemma validation requires the vectorized native batch API"
        )
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        cpu_batch = native(np.arange(start, stop, dtype=np.int64))
        if device.type == "cuda":
            cpu_batch = cpu_batch.pin_memory()
        yield _recipe_batch(cpu_batch, patch_stride=patch_stride).to(
            device, non_blocking=device.type == "cuda"
        )


def _prepared_validation_ledger(
    model: DiffusionGemmaModel,
    dataset,
    *,
    rows: int,
    batch_size: int,
    patch_stride: int,
    canvas_length: int,
    branches: int,
    seed: int,
    device: torch.device,
) -> tuple[PreparedDiffusionGemmaValidationBatch, ...]:
    """Materialize the held corruption once so validation is GPU-bound."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    ledger = tuple(
        PreparedDiffusionGemmaValidationBatch(
            batch,
            prepare_diffusion_gemma_inputs(
                model,
                batch,
                canvas_length=canvas_length,
                branches=branches,
                integrated_exact_k=True,
                generator=generator,
            ),
        )
        .pin_memory()
        .to(device, non_blocking=device.type == "cuda")
        for batch in _validation_batches(
            dataset,
            rows=rows,
            batch_size=batch_size,
            patch_stride=patch_stride,
            device=torch.device("cpu"),
        )
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return ledger


def main() -> None:
    args = parse_args()
    checkpoint_policy = RecoveryCheckpointPolicy(
        args.checkpoint_interval_seconds
    )
    provenance = source_provenance()
    if args.print_provenance:
        print(json.dumps(provenance, indent=2, sort_keys=True))
        return
    if args.steps <= 0 or args.batch_size <= 0 or args.global_batch_size <= 0:
        raise ValueError("steps, microbatch, and global batch must be positive")
    if args.steps >= 2_000 and args.global_batch_size != 249:
        raise ValueError("matched 2k DiffusionGemma runs require global batch 249")
    positive = (
        args.canvas_length,
        args.branches,
        args.log_every,
        args.val_every,
        args.validation_rows,
        args.validation_batch_size,
        args.checkpoint_interval_seconds,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("canvas, logging, validation, and checkpoint values must be positive")
    if not 0 <= args.warmdown_steps <= args.steps:
        raise ValueError("warmdown steps must lie in [0, steps]")
    if args.max_grad_norm <= 0:
        raise ValueError("max grad norm must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("standalone DiffusionGemma currently requires one device")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required unless --cpu is explicit")
    device = torch.device("cpu" if args.cpu else "cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    expected_source = os.environ.get("BYTE_DIFFUSION_GEMMA_EXPECTED_SOURCE_SHA256")
    if args.steps >= 2_000 and expected_source is None:
        raise ValueError(
            "2k DiffusionGemma runs require "
            "BYTE_DIFFUSION_GEMMA_EXPECTED_SOURCE_SHA256"
        )
    if expected_source is not None and provenance["sha256"] != expected_source:
        raise ValueError(
            "DiffusionGemma source differs from the pinned contract: "
            f"expected {expected_source}, observed {provenance['sha256']}"
        )
    expected_data = os.environ.get("BYTE_DIFFUSION_EXPECTED_DATA_SHA256")
    if args.steps >= 2_000 and expected_data is None:
        raise ValueError("2k DiffusionGemma runs require BYTE_DIFFUSION_EXPECTED_DATA_SHA256")

    manifest, train_chunks, validation_chunks = load_data_directory(
        args.data_path,
        chunk_size=args.chunk_size,
        # The loader's BLT-D label selects document-aligned pages. This cell
        # does not call or inherit the BLT-D model/objective.
        recipe="blt_d",
        required_branch_bytes=args.canvas_length * args.branches,
        branch_span_length=args.canvas_length,
        validation_chunk_limit=args.validation_rows,
        expected_payload_sha256=expected_data,
    )
    config = model_config_from_env()
    if args.steps >= 2_000 and (
        device.type != "cuda"
        or not args.compile
        or config != type(config)()
        or args.chunk_size != 8_192
        or args.canvas_length != 512
        or args.branches != 1
        or args.clean_ar_weight != 1.0
        or args.exact_k
        or args.validation_rows != 256
    ):
        raise ValueError(
            "2k DiffusionGemma runs require the benchmarked production contract"
        )
    if (
        manifest.output_size != config.vocab.output_size
        or manifest.pad_id != config.vocab.pad_id
        or manifest.eot_id != config.vocab.eot_id
    ):
        raise ValueError("dataset vocabulary and DiffusionGemma model differ")
    model = DiffusionGemmaModel(config).to(device)
    if config == type(config)():
        model.validate_production_parameterization()
    execution_model = (
        # Torch 2.13's dynamic backward scheduler fails on this cell's mixed
        # self-conditioning reduction geometry (CantSplit on symbolic
        # products). Production has only full and tail microbatch shapes, so
        # concrete static specializations are both bounded and faster.
        torch.compile(model, dynamic=False, fullgraph=False)
        if device.type == "cuda" and args.compile
        else model
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=device.type == "cuda",
    )
    cursor = DeterministicChunkCursor(train_chunks, seed=args.seed, shuffle=True)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    output_dir = REPO_ROOT / "ablation_results" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / (
        "native_metrics.jsonl"
        if os.environ.get("ABLATION_RUNNER_OWNS_METRICS") == "1"
        else "metrics.jsonl"
    )
    checkpoint_path = args.checkpoint or output_dir / "checkpoint.pt"
    dataset_manifest = json.loads((args.data_path / "manifest.json").read_text())
    dataset_payload_sha256 = dataset_manifest.get("payload_sha256")
    if expected_data is not None and dataset_payload_sha256 != expected_data:
        raise ValueError(
            "dataset payload differs from the pinned contract: "
            f"expected {expected_data}, observed {dataset_payload_sha256}"
        )
    if args.steps >= 2_000 and args.readiness_report is None:
        raise ValueError("2k DiffusionGemma runs require a readiness report")
    readiness_evidence = (
        validate_architecture_readiness(
            args.readiness_report,
            architecture="diffusion_gemma",
            source_sha256=str(provenance["sha256"]),
            dataset_payload_sha256=str(dataset_payload_sha256),
            global_batch_size=args.global_batch_size,
            microbatch_size=args.batch_size,
            validation_batch_size=args.validation_batch_size,
            model_config=config.to_dict(),
            workload={
                "branches": args.branches,
                "canvas_length": args.canvas_length,
                "clean_ar_weight": args.clean_ar_weight,
                "diagnostic_cadence": diagnostic_cadence_contract(
                    log_every=args.log_every,
                    validation_every=args.val_every,
                ),
                "integrated_exact_k_training": args.exact_k,
                "runtime_candidate_validation": False,
                "self_condition": True,
            },
            runtime={
                "compiled": args.compile,
                "device_type": device.type,
                "validation_rows": args.validation_rows,
                "world_size": 1,
            },
        )
        if args.readiness_report is not None
        else None
    )
    training_contract = {
        "steps": args.steps,
        "microbatch_size": args.batch_size,
        "global_batch_size": args.global_batch_size,
        "canvas_length": args.canvas_length,
        "branches": args.branches,
        "learning_rate": args.learning_rate,
        "clean_ar_weight": args.clean_ar_weight,
        "warmdown_steps": args.warmdown_steps,
        "max_grad_norm": args.max_grad_norm,
        "compile": args.compile,
        "checkpoint_interval_seconds": args.checkpoint_interval_seconds,
        "diagnostic_cadence": diagnostic_cadence_contract(
            log_every=args.log_every,
            validation_every=args.val_every,
        ),
        "exact_k": args.exact_k,
        "seed": args.seed,
        "readiness_evidence": readiness_evidence,
    }
    contract = {
        "schema": "byte_diffusion_gemma_run/v3",
        "run_name": args.run_name,
        "model": config.to_dict(),
        "parameter_count": model.parameter_count,
        "source": provenance,
        "atomic_manifest_sha256": manifest.sha256,
        "dataset_payload_sha256": dataset_payload_sha256,
        "train_dataset_sha256": getattr(train_chunks, "dataset_sha256", None),
        "validation_dataset_sha256": getattr(validation_chunks, "dataset_sha256", None),
        "objective": "uniform_replacement_dense_unchanged_inclusive_ce",
        "promotion_metric": "gsm8k_exact_match_generation_accuracy",
        "denoising_ce_is_diffusion_bpb": False,
        "ar_bpb_role": "retained_language_anchor_only",
        "training": training_contract,
    }
    (output_dir / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print("byte_diffusion_gemma_contract " + json.dumps(contract, sort_keys=True), flush=True)

    completed_steps = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("schema") != "byte_diffusion_gemma_checkpoint/v3":
            raise ValueError("unsupported DiffusionGemma checkpoint schema")
        if checkpoint.get("source_sha256") != provenance["sha256"]:
            raise ValueError("resume checkpoint source provenance differs")
        if checkpoint.get("dataset_payload_sha256") != dataset_payload_sha256:
            raise ValueError("resume checkpoint dataset provenance differs")
        if checkpoint.get("model_config") != config.to_dict():
            raise ValueError("resume checkpoint model configuration differs")
        if checkpoint.get("training") != training_contract:
            raise ValueError("resume checkpoint training contract differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        cursor.load_state_dict(checkpoint["cursor"])
        generator.set_state(checkpoint["generator_state"])
        completed_steps = int(checkpoint["steps"])

    def save_checkpoint(step: int) -> None:
        atomic_torch_save(
            {
                "schema": "byte_diffusion_gemma_checkpoint/v3",
                "model_config": config.to_dict(),
                "training": training_contract,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "steps": step,
                "cursor": cursor.state_dict(),
                "generator_state": generator.get_state(),
                "source_sha256": provenance["sha256"],
                "dataset_payload_sha256": dataset_payload_sha256,
            },
            checkpoint_path,
        )
        checkpoint_policy.committed(step)

    validation_ledger = _prepared_validation_ledger(
        model,
        validation_chunks,
        rows=args.validation_rows,
        batch_size=args.validation_batch_size,
        patch_stride=config.patch_stride,
        canvas_length=args.canvas_length,
        branches=args.branches,
        seed=args.seed + 10_000,
        device=device,
    )
    microbatch_keys = tuple(
        (start, min(start + args.batch_size, args.global_batch_size))
        for start in range(0, args.global_batch_size, args.batch_size)
    )

    started = time.perf_counter()
    validation_seconds = 0.0
    with (
        DeviceBatchPrefetcher(device=device) as batch_prefetcher,
        metrics_path.open("a", encoding="utf-8") as metrics,
    ):
        def emit(record: dict[str, object], *, prefix: str) -> None:
            line = json.dumps(record, sort_keys=True, allow_nan=False)
            metrics.write(line + "\n")
            metrics.flush()
            print(prefix + " " + line, flush=True)

        def validate(step: int) -> None:
            nonlocal validation_seconds
            validation_started = time.perf_counter()
            validation = validate_diffusion_gemma(
                execution_model,
                validation_ledger,
                canvas_length=args.canvas_length,
                branches=args.branches,
                seed=args.seed + 10_000,
                self_condition=True,
                integrated_exact_k=True,
            )
            validation_seconds += time.perf_counter() - validation_started
            emit(
                {
                    "schema": "byte_diffusion_gemma_validation/v2",
                    "kind": "validation",
                    "step": step,
                    "denoising_ce_nats_per_atom": validation.denoising_ce_nats_per_atom,
                    "ar_anchor_bpb": validation.ar_anchor_bpb,
                    "denoising_accuracy": validation.denoising_accuracy,
                    "changed_accuracy": validation.changed_accuracy,
                    "unchanged_accuracy": validation.unchanged_accuracy,
                    "diffusion_targets": validation.diffusion_targets,
                    "ar_targets": validation.ar_targets,
                    "changed_targets": validation.changed_targets,
                    "unchanged_targets": validation.unchanged_targets,
                    "promotion_metric": "gsm8k_exact_match_generation_accuracy",
                    "denoising_ce_is_diffusion_bpb": False,
                    "ar_bpb_role": "anchor_only",
                },
                prefix="byte_diffusion_gemma_val",
            )
            training_time_ms = (
                time.perf_counter() - started - validation_seconds
            ) * 1_000
            print(
                f"step:{step}/{args.steps} "
                f"val_loss:{validation.denoising_ce_nats_per_atom:.6f} "
                f"val_bpb:{validation.ar_anchor_bpb:.6f} "
                f"val_ar_anchor_bpb:{validation.ar_anchor_bpb:.6f} "
                f"val_denoising_ce_nats_per_atom:"
                f"{validation.denoising_ce_nats_per_atom:.6f} "
                "generation_primary:1 "
                f"train_time:{training_time_ms:.3f}ms",
                flush=True,
            )

        if completed_steps == 0:
            validate(0)
        for step in range(completed_steps + 1, args.steps + 1):
            update_indices = cursor.next_indices(args.global_batch_size)
            native_update = train_chunks.training_batch(update_indices)
            if device.type == "cuda":
                native_update = native_update.pin_memory()
            update_batch = _recipe_batch(
                native_update, patch_stride=config.patch_stride
            ).to(device, non_blocking=device.type == "cuda")
            prepared_update = prepare_diffusion_gemma_inputs(
                execution_model,
                update_batch,
                canvas_length=args.canvas_length,
                branches=args.branches,
                integrated_exact_k=args.exact_k,
                generator=generator,
                validate_candidates=False,
            )
            diffusion_denominator = prepared_update.corruption.active.sum()
            ar_denominator = (
                update_batch.ar_targets.ne(-100).sum()
                + update_batch.bos_targets.numel()
            )
            del update_batch, native_update
            model.train()
            optimizer.zero_grad(set_to_none=True)
            materialize_diagnostics = materialize_training_diagnostics(
                step,
                args.steps,
                log_every=args.log_every,
                validation_every=args.val_every,
            )
            totals = (
                torch.zeros(7, dtype=torch.float64, device=device)
                if materialize_diagnostics
                else None
            )
            microsteps = 0
            def prepare_microbatch(key: tuple[int, int]) -> DiffusionGemmaBatch:
                start, stop = key
                return _recipe_batch(
                    train_chunks.training_batch(update_indices[start:stop]),
                    patch_stride=config.patch_stride,
                )

            device_batches = batch_prefetcher.batches(
                microbatch_keys, prepare_microbatch
            )
            for (start, stop), batch in zip(
                microbatch_keys, device_batches, strict=True
            ):
                self_conditioned_rows = sample_static_half_batch_mask(
                    stop - start,
                    device=device,
                    generator=generator,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    loss = diffusion_gemma_loss(
                        execution_model,
                        batch,
                        canvas_length=args.canvas_length,
                        branches=args.branches,
                        clean_ar_weight=args.clean_ar_weight,
                        self_condition=True,
                        integrated_exact_k=args.exact_k,
                        generator=generator,
                        prepared=prepared_update.slice_rows(start, stop),
                        self_conditioned_rows=self_conditioned_rows,
                        diffusion_denominator=diffusion_denominator,
                        ar_denominator=ar_denominator,
                    )
                loss.total.backward()
                if totals is not None:
                    totals += torch.stack(
                        (
                            loss.diffusion_nll_sum.detach().double(),
                            loss.ar_nll_sum.detach().double(),
                            loss.diffusion_targets.detach().double(),
                            loss.ar_targets.detach().double(),
                            loss.changed_targets.detach().double(),
                            loss.unchanged_targets.detach().double(),
                            loss.corruption.k.sum().detach().double(),
                        )
                    )
                microsteps += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
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
                elapsed = time.perf_counter() - started - validation_seconds
                diffusion_targets = int(totals[2])
                ar_targets = int(totals[3])
                record = {
                    "schema": "byte_diffusion_gemma_train/v3",
                    "kind": "train",
                    "step": step,
                    "total_loss": float(
                        totals[0] / max(diffusion_targets, 1)
                        + args.clean_ar_weight * totals[1] / max(ar_targets, 1)
                    ),
                    "denoising_ce_nats_per_atom": float(
                        totals[0] / max(diffusion_targets, 1)
                    ),
                    "clean_ar_loss": float(totals[1] / max(ar_targets, 1)),
                    "ar_anchor_bpb": float(
                        totals[1] / max(ar_targets, 1) / np.log(2.0)
                    ),
                    "ar_bpb_role": "anchor_only",
                    "diffusion_targets": diffusion_targets,
                    "ar_targets": ar_targets,
                    "changed_targets": int(totals[4]),
                    "unchanged_targets": int(totals[5]),
                    "noise_fraction": float(totals[6] / max(diffusion_targets, 1)),
                    "microsteps": microsteps,
                    "global_batch_size": args.global_batch_size,
                    "learning_rate": learning_rate,
                    "grad_norm": float(grad_norm),
                    "elapsed_seconds": elapsed,
                    "steps_per_second": step / elapsed,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                }
                emit(record, prefix="byte_diffusion_gemma_train")
                print(
                    f"step:{step}/{args.steps} "
                    f"train_loss:{record['total_loss']:.6f} "
                    f"train_time:{float(record['elapsed_seconds']) * 1_000:.3f}ms "
                    f"denoising_ce_nats_per_atom:"
                    f"{record['denoising_ce_nats_per_atom']:.6f} "
                    f"ar_anchor_bpb:{record['ar_anchor_bpb']:.6f}",
                    flush=True,
                )
            if step % args.val_every == 0 or step == args.steps:
                validate(step)
            if checkpoint_policy.due() or (
                step == args.steps and checkpoint_policy.terminal_due(step)
            ):
                save_checkpoint(step)
    result = {
        "schema": "byte_diffusion_gemma_result/v3",
        "run_name": args.run_name,
        "steps": args.steps,
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "metrics": str(metrics_path.relative_to(REPO_ROOT)),
        "promotion_metric": "gsm8k_exact_match_generation_accuracy",
        "diagnostic_metric": "held_uniform_replacement_denoising_ce_nats_per_atom",
        "ar_anchor_metric": "clean_ar_bos_bpb",
        "diffusion_bpb_available": False,
        "reason": (
            "uniform-replacement dense CE is not the absorbing-diffusion ELBO; "
            "AR BPB is retained only as a language anchor"
        ),
    }
    result_name = (
        "native_result.json"
        if os.environ.get("ABLATION_RUNNER_OWNS_METRICS") == "1"
        else "result.json"
    )
    (output_dir / result_name).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

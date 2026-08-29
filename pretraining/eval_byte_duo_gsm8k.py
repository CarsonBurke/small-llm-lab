#!/usr/bin/env python3
"""Artifact-only few-shot GSM8K evaluation for Byte-Duo.

The executable closure deliberately excludes every non-Duo model family and
all post-training/nanoGPT infrastructure. GPU execution must run through mlq.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping

import pandas as pd
import torch
from torch._dynamo.utils import counters as dynamo_counters


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.data import AtomicIdManifest
from pretraining.byte_diffusion.config import AtomicVocabulary, ByteDiffusionConfig
from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.export import (
    duo_serving_contract,
    load_artifact,
    load_embedded_entropy_patcher,
    parse_artifact,
)
from pretraining.byte_diffusion.patching import CausalEntropyPatcher
from pretraining.byte_diffusion.serving_duo import (
    ByteGeneration,
    duo_generate_bytes_batched,
    required_canvases,
    warmup_prompt_groups,
)
from pretraining.gsm8k_contract import (
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


EVALUATOR_SCHEMA = "byte_diffusion_gsm8k/v9"


def model_config_from_dict(value: Mapping[str, object]) -> ByteDiffusionConfig:
    fields = dict(value)
    vocab = fields.get("vocab")
    if not isinstance(vocab, Mapping):
        raise ValueError("Byte-Duo artifact config omitted its atomic vocabulary")
    fields["vocab"] = AtomicVocabulary(**vocab)
    if "ngram_orders" in fields:
        fields["ngram_orders"] = tuple(fields["ngram_orders"])  # type: ignore[arg-type]
    return ByteDiffusionConfig(**fields)  # type: ignore[arg-type]


def _local_imports(path: Path) -> tuple[Path, ...]:
    """Resolve repository-local imports for executable-closure accounting."""

    tree = ast.parse(path.read_text(), filename=str(path))
    candidates: set[Path] = set()

    def add_package_initializers(target: Path) -> None:
        parent = target.parent
        while parent != REPO_ROOT and parent.is_relative_to(REPO_ROOT):
            initializer = parent / "__init__.py"
            if initializer.is_file():
                candidates.add(initializer.resolve())
            parent = parent.parent

    def add_module(parts: list[str]) -> None:
        if not parts:
            return
        stem = REPO_ROOT.joinpath(*parts)
        module = stem.with_suffix(".py")
        package = stem / "__init__.py"
        if module.is_file():
            candidates.add(module.resolve())
            add_package_initializers(module)
        elif package.is_file():
            candidates.add(package.resolve())
            add_package_initializers(package)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_module(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = path.parent.relative_to(REPO_ROOT).parts
                base = list(relative[: len(relative) - node.level + 1])
            else:
                base = []
            module = [] if node.module is None else node.module.split(".")
            add_module(base + module)
            for alias in node.names:
                if alias.name != "*":
                    add_module(base + module + alias.name.split("."))
    return tuple(sorted(candidates))


def evaluator_source_provenance() -> dict[str, object]:
    pending = {Path(__file__).resolve()}
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        if not path.is_relative_to(REPO_ROOT):
            raise ValueError(f"Duo serving closure escaped repository: {path}")
        observed.add(path)
        pending.update(item for item in _local_imports(path) if item not in observed)
    relative_paths = tuple(sorted(path.relative_to(REPO_ROOT) for path in observed))
    digest = hashlib.sha256()
    for relative in relative_paths:
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update((REPO_ROOT / relative).read_bytes())
        digest.update(b"\0")
    return {
        "schema": "byte_duo_gsm8k_evaluator_source/v1",
        "sha256": digest.hexdigest(),
        "files": tuple(map(str, relative_paths)),
        "bytes": sum((REPO_ROOT / path).stat().st_size for path in relative_paths),
    }


def _dynamo_compile_evidence() -> dict[str, int]:
    return {
        "unique_graphs": int(dynamo_counters["stats"]["unique_graphs"]),
        "recompiles": int(sum(dynamo_counters["recompiles"].values())),
        "graph_breaks": int(sum(dynamo_counters["graph_break"].values())),
    }


def load_duo_artifact(
    path: Path, device: torch.device
) -> tuple[DuoModel, dict[str, object], CausalEntropyPatcher | None]:
    """Reconstruct all runtime state from one authenticated artifact."""

    artifact = path.read_bytes()
    metadata, _ = parse_artifact(artifact)
    serving = duo_serving_contract(metadata)
    config_payload = metadata.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("Byte-Duo artifact omitted its model config")
    config = model_config_from_dict(dict(config_payload))
    manifest_payload = metadata.get("atomic_manifest")
    if not isinstance(manifest_payload, Mapping):
        raise ValueError("Byte-Duo artifact omitted its atomic manifest")
    manifest = AtomicIdManifest.from_dict(dict(manifest_payload))
    if manifest.sha256 != AtomicIdManifest.reference().sha256:
        raise ValueError("Byte-Duo artifact atomic vocabulary differs from serving")
    if (
        config.vocab.byte_values != manifest.byte_count
        or config.vocab.output_size != manifest.output_size
        or config.vocab.mask_id != manifest.mask_id
        or config.vocab.pad_id != manifest.pad_id
        or config.vocab.eot_id != manifest.eot_id
    ):
        raise ValueError("Byte-Duo model config and atomic manifest disagree")
    model = DuoModel(config, schedule_eps=float(serving["schedule_eps"]))
    load_artifact(model, artifact)
    patcher = load_embedded_entropy_patcher(artifact)
    if (patcher is None) != (serving["patching_policy"] == "fixed_stride_v1"):
        raise ValueError("Byte-Duo serving policy and embedded patcher disagree")
    return model.to(device).eval(), metadata, patcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--decode-mode", choices=("duo",), default="duo")
    parser.add_argument("--shots", default="5")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--prompt-format", choices=sorted(PROMPT_FORMATS), default="harness"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=512)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--duo-visible-width", type=int)
    parser.add_argument("--duo-commit-width", type=int)
    parser.add_argument("--duo-terminal-eps", type=float, default=1e-5)
    parser.add_argument(
        "--duo-posterior-precision",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--generation-seed", type=int, default=12345)
    parser.add_argument("--sampling", choices=("categorical",), default="categorical")
    parser.add_argument("--max-new-bytes", type=int, default=512)
    parser.add_argument("--max-new-atoms", type=int)
    parser.add_argument("--max-native-actions", type=int)
    parser.add_argument("--context-bytes", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--trace-diffusion", action="store_true")
    parser.add_argument("--keep-calculator-annotations", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable result {args.output}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.trace_diffusion and (args.limit is None or args.limit > 32):
        raise ValueError("--trace-diffusion requires explicit --limit <= 32")
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
    for name in ("duo_commit_width", "duo_visible_width"):
        value = getattr(args, name)
        if value is not None and not 0 < value <= args.block_length:
            raise ValueError(f"--{name.replace('_', '-')} must fit the block")
    if (
        args.duo_commit_width is not None
        and args.duo_visible_width is not None
        and args.duo_commit_width > args.duo_visible_width
    ):
        raise ValueError("--duo-commit-width cannot exceed --duo-visible-width")
    max_new_atoms = args.max_new_bytes if args.max_new_atoms is None else args.max_new_atoms
    if max_new_atoms <= 0:
        raise ValueError("--max-new-atoms must be positive")
    return max_new_atoms


def _record(
    *,
    shots: int,
    seed: int,
    exemplar_rows: list[int],
    prompts: list[bytes],
    generations: list[ByteGeneration],
    gold: list[str | None],
    prompt_format: object,
    traces: list[dict[str, object]] | None,
    generation_seconds: float,
    model_forwards: int,
    physical_batches: int,
    batch_lanes: int,
    work_width: int,
    samples: int,
    warm_compile_evidence: dict[str, int],
    measured_compile_evidence: dict[str, int],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for index, (generation, truth) in enumerate(zip(generations, gold, strict=True)):
        prediction = (
            extract_prediction(generation.text, prompt_format)  # type: ignore[arg-type]
            if generation.text is not None
            else None
        )
        row = {
            "test_row": index,
            "gold": truth,
            "prediction": prediction,
            "correct": prediction is not None and prediction == truth,
            "delimiter_emitted": (
                generation.text is not None
                and prompt_format.delimiter in generation.text  # type: ignore[attr-defined]
            ),
            **asdict(generation),
            "raw_hex": generation.raw.hex(),
            **({"diffusion_trace": traces[index]} if traces is not None else {}),
        }
        row.pop("raw")
        rows.append(row)
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "shots": shots,
        "seed": seed,
        "exemplar_train_rows": exemplar_rows,
        "examples": len(rows),
        "correct": correct,
        "exact_match": correct / len(rows),
        "delimiter_emitted_rate": sum(bool(row["delimiter_emitted"]) for row in rows) / len(rows),
        "parsed_answer_rate": sum(row["prediction"] is not None for row in rows) / len(rows),
        "invalid_utf8_rate": sum(bool(row["invalid_utf8"]) for row in rows) / len(rows),
        "generation_seconds": generation_seconds,
        "generated_bytes": sum(len(str(row["raw_hex"])) // 2 for row in rows),
        "generated_atoms": sum(int(row["generated_atoms"]) for row in rows),
        "native_actions": sum(int(row["native_actions"]) for row in rows),
        "model_forwards": model_forwards,
        "warm_compile_evidence": warm_compile_evidence,
        "measured_compile_evidence": measured_compile_evidence,
        "physical_batches": physical_batches,
        "realized_mean_batch_size": batch_lanes / physical_batches,
        "mean_prompt_bytes": sum(map(len, prompts)) / len(prompts),
        "max_prompt_bytes": max(map(len, prompts)),
        "duo_work_width": work_width,
        "termination_counts": {
            reason: sum(row["termination"] == reason for row in rows)
            for reason in sorted({str(row["termination"]) for row in rows})
        },
        "serialized_sample_count": min(samples, len(rows)),
        "rows": rows[:samples],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GSM8K model evaluation requires CUDA through mlq")
    max_new_atoms = _validate_args(args)
    device = torch.device("cuda")
    model, metadata, entropy_patcher = load_duo_artifact(args.artifact, device)
    serving = duo_serving_contract(metadata)
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("final Duo artifact omitted checkpoint provenance")
    trained_canvas = provenance.get("canvas_length")
    if not isinstance(trained_canvas, int) or trained_canvas <= 0:
        raise ValueError("final Duo artifact omitted its trained canvas length")
    if args.block_length != trained_canvas:
        raise ValueError("Duo eval canvas differs from the artifact serving contract")
    if model.config.duo_clean_patching == "causal_entropy_v1" and entropy_patcher is None:
        raise AssertionError("validated entropy artifact lost its patcher")
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
    max_native_actions = args.max_native_actions
    if max_native_actions is None:
        blocks = required_canvases(
            max_new_atoms,
            args.block_length,
            model.config.patch_stride,
            args.duo_commit_width,
            args.duo_visible_width,
        )
        max_native_actions = blocks * (args.diffusion_steps + 2)
    records: list[dict[str, object]] = []
    compile_warmup_seconds = 0.0
    warmed_shapes: set[tuple[int, int]] = set()
    started = time.perf_counter()
    for shots in shot_counts:
        for seed in seeds:
            exemplar_rows = select_exemplars(train, shots, max(shot_counts), seed)
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
            work_width = (
                (max(map(len, prompts)) + max_new_atoms + model.config.patch_stride - 1)
                // model.config.patch_stride
                * model.config.patch_stride
            )
            traces = (
                [{"alignment_steps": [], "diffusion_blocks": []} for _ in prompts]
                if args.trace_diffusion
                else None
            )
            pending = tuple(
                group
                for group in warmup_prompt_groups(prompts, args.batch_size)
                if (work_width, len(group)) not in warmed_shapes
            )
            dynamo_counters.clear()
            torch.cuda.synchronize()
            warm_started = time.perf_counter()
            for warmup_index, group in enumerate(pending):
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
                    work_width=work_width,
                    entropy_patcher=entropy_patcher,
                )
                warmed_shapes.add((work_width, len(group)))
            torch.cuda.synchronize()
            compile_warmup_seconds += time.perf_counter() - warm_started
            warm_evidence = _dynamo_compile_evidence()
            dynamo_counters.clear()
            generations: list[ByteGeneration] = []
            model_forwards = 0
            physical_batches = 0
            batch_lanes = 0
            torch.cuda.synchronize()
            generation_started = time.perf_counter()
            for start in range(0, len(prompts), args.batch_size):
                selected = prompts[start : start + args.batch_size]
                batch = duo_generate_bytes_batched(
                    model,
                    selected,
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
                    seed=args.generation_seed + 10_000_019 * seed + 97_409 * start,
                    device=device,
                    work_width=work_width,
                    trace_records=(
                        traces[start : start + len(selected)] if traces is not None else None
                    ),
                    entropy_patcher=entropy_patcher,
                )
                model_forwards += max(item.model_forwards for item in batch)
                physical_batches += 1
                batch_lanes += len(batch)
                generations.extend(batch)
                print(
                    f"shots={shots} seed={seed} scored={len(generations)}/{len(prompts)}",
                    flush=True,
                )
            torch.cuda.synchronize()
            records.append(
                _record(
                    shots=shots,
                    seed=seed,
                    exemplar_rows=exemplar_rows,
                    prompts=prompts,
                    generations=generations,
                    gold=gold,
                    prompt_format=prompt_format,
                    traces=traces,
                    generation_seconds=time.perf_counter() - generation_started,
                    model_forwards=model_forwards,
                    physical_batches=physical_batches,
                    batch_lanes=batch_lanes,
                    work_width=work_width,
                    samples=args.samples,
                    warm_compile_evidence=warm_evidence,
                    measured_compile_evidence=_dynamo_compile_evidence(),
                )
            )
    artifact_sha256 = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    result = {
        "schema": EVALUATOR_SCHEMA,
        "evaluator_source": evaluator_source_provenance(),
        "generation_semantics": "strict_utf8_atomic_controls_dual_atom_byte_caps_in_canvas_stops",
        "implementation_maturity": "ragged_batched_duo_per_canvas_clean_bank_cache",
        "decode_mode": args.decode_mode,
        "patching_policy": serving["patching_policy"],
        "entropy_patcher_sha256": serving["entropy_patcher_sha256"],
        "entropy_dataset_path": None,
        "block_length": args.block_length,
        "diffusion_steps": args.diffusion_steps,
        "schedule_eps": serving["schedule_eps"],
        "sampling_terminal_eps": args.duo_terminal_eps,
        "duo_posterior_precision": args.duo_posterior_precision,
        "duo_commit_width": args.duo_commit_width,
        "duo_visible_width": args.duo_visible_width,
        "checkpoint_training_source_matches_current": None,
        "compatible_duo_source_override": False,
        "unmasking_strategy": None,
        "confidence_threshold": None,
        "entropy_budget": None,
        "generation_seed": args.generation_seed,
        "sampling": args.sampling,
        "artifact": str(args.artifact),
        "artifact_sha256": artifact_sha256,
        "checkpoint": None,
        "checkpoint_sha256": (
            provenance.get("checkpoint_sha256")
            if isinstance(provenance, Mapping)
            else None
        ),
        "checkpoint_schema": (
            provenance.get("checkpoint_schema")
            if isinstance(provenance, Mapping)
            else None
        ),
        "training_source_sha256": (
            provenance.get("source_sha256") if isinstance(provenance, Mapping) else None
        ),
        "completed_steps": (
            provenance.get("checkpoint_step") if isinstance(provenance, Mapping) else None
        ),
        "model_parameters": model.parameter_count,
        "gsm8k_snapshot": str(GSM8K_ROOT),
        "gsm8k_train_sha256": sha256(TRAIN_PARQUET),
        "gsm8k_test_sha256": sha256(TEST_PARQUET),
        "prompt_format": prompt_format.name,
        "strip_calculator_annotations": strip_calculator,
        "max_new_bytes": args.max_new_bytes,
        "max_new_atoms": max_new_atoms,
        "max_native_actions": max_native_actions,
        "context_bytes": args.context_bytes,
        "requested_batch_size": args.batch_size,
        "requested_serialized_samples": args.samples,
        "batching": "ragged_per_canvas_clean_bank_cache",
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

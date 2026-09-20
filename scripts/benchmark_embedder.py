#!/usr/bin/env python3
"""Matched cached CharacterGPT / fixed / learned embedder inference benchmark.

Run only as an mlq priority-1, max-parallel-runs-1 workload. This script neither
submits jobs nor changes queue state. It measures three pinned heldout prompts,
three seed-scheduled repeats, 256 outputs at temperature .8, rotating variant
order. Each exact prompt/seed/variant workload is fully warmed before timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.nanogpt_mini import character_generation, embedder_generation
from pretraining.nanogpt_mini.nanogpt_mini_native_bits_train import (
    checkpoint_model_kind,
    load_checkpoint,
    load_model,
)
from pretraining.nanogpt_mini.native_bits_data import sha256_file

PROMPT_OFFSETS = (0, 4096, 8192)
PROMPT_CHARACTERS = 128
PARITY_CHARACTERS = 1024
REPEATS = 3
NEW_CHARACTERS = 256
TEMPERATURE = 0.8
BASE_SEED = 1337
VARIANTS = ("baseline", "fixed", "learned")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--fixed-checkpoint", type=Path, required=True)
    parser.add_argument("--learned-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Optional relocated prepared data; must match checkpoint val.npy SHA256",
    )
    return parser.parse_args()


def ids_sha256(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<u4").tobytes()).hexdigest()


def current_source_hashes() -> dict:
    names = (
        "scripts/benchmark_embedder.py",
        "pretraining/nanogpt_mini/character_generation.py",
        "pretraining/nanogpt_mini/embedder_generation.py",
        "pretraining/nanogpt_mini/mini_cached.py",
        "pretraining/nanogpt_mini/nanogpt_mini_character_model.py",
        "pretraining/nanogpt_mini/nanogpt_mini_embedder_model.py",
        "pretraining/nanogpt_mini/nanogpt_mini_model.py",
        "pretraining/nanogpt_mini/nanogpt_mini_native_bits_train.py",
        "pretraining/nanogpt_mini/native_bits_data.py",
    )
    return {name: sha256_file(REPO_ROOT / name) for name in names}


def load_variants(args):
    checkpoints, models, records = {}, {}, {}
    for variant in VARIANTS:
        path = getattr(args, f"{variant}_checkpoint").expanduser().resolve(strict=True)
        digest = sha256_file(path)
        checkpoint = load_checkpoint(path)
        if sha256_file(path) != digest:
            raise ValueError(f"{variant} checkpoint changed during loading")
        kind = checkpoint_model_kind(checkpoint)
        expected_kind = "softmax" if variant == "baseline" else "embedder"
        if kind != expected_kind:
            raise ValueError(f"{variant} must be a genuine {expected_kind} checkpoint")
        if variant != "baseline" and checkpoint["model_config"]["gate_mode"] != variant:
            raise ValueError(f"{variant} checkpoint has the wrong gate_mode")
        if checkpoint["step"] <= 0:
            raise ValueError(
                "benchmark requires trained checkpoints, not initialization"
            )
        checkpoints[variant] = checkpoint
        records[variant] = {
            "path": str(path),
            "sha256": digest,
            "architecture": checkpoint["architecture"],
            "step": checkpoint["step"],
            "model_config": checkpoint["model_config"],
            "train_config": checkpoint["train_config"],
            "training_source_hashes": checkpoint["provenance"]["source_hashes"],
        }
    reference = checkpoints["baseline"]
    for variant, checkpoint in checkpoints.items():
        if checkpoint["alphabet"] != reference["alphabet"]:
            raise ValueError(f"{variant} alphabet identities differ from baseline")
        for name in ("num_layers", "model_dim", "vocab_size"):
            if checkpoint["model_config"][name] != reference["model_config"][name]:
                raise ValueError(f"{variant} has an unmatched {name}")
        expected = reference["provenance"]["data"]["artifacts"]["val.npy"]
        if checkpoint["provenance"]["data"]["artifacts"]["val.npy"] != expected:
            raise ValueError(f"{variant} was trained with a different heldout split")
    fixed = dict(checkpoints["fixed"]["model_config"])
    learned = dict(checkpoints["learned"]["model_config"])
    fixed.pop("gate_mode")
    learned.pop("gate_mode")
    if fixed != learned:
        raise ValueError(
            "fixed/learned embedder configurations differ beyond gate_mode"
        )
    for variant, checkpoint in checkpoints.items():
        started = time.perf_counter()
        model, device = load_model(checkpoint)
        model.requires_grad_(False)
        torch.cuda.synchronize(device)
        models[variant] = model
        records[variant]["model_load_seconds"] = time.perf_counter() - started
        records[variant]["parameter_count"] = sum(p.numel() for p in model.parameters())
    return checkpoints, models, records


def load_heldout(args, checkpoint):
    root = args.data_path or Path(checkpoint["train_config"]["data_path"])
    if args.data_path is None and not root.is_absolute():
        root = REPO_ROOT / root
    root = root.expanduser().resolve(strict=True)
    metadata = json.loads((root / "metadata.json").read_text())
    alphabet = json.loads((root / "alphabet.json").read_text())
    if alphabet != checkpoint["alphabet"]:
        raise ValueError("prepared heldout alphabet differs from checkpoint")
    path = root / "val.npy"
    digest = sha256_file(path)
    expected = checkpoint["provenance"]["data"]["artifacts"]["val.npy"]
    if digest != expected or digest != metadata["artifacts"]["val.npy"]:
        raise ValueError(
            "heldout val.npy fingerprint differs from checkpoint/prepared metadata"
        )
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.uint32 or values.ndim != 1:
        raise ValueError("heldout IDs must be a flat uint32 array")
    if values.size < max(PROMPT_OFFSETS) + PARITY_CHARACTERS:
        raise ValueError("heldout split is too short for the fixed prefix schedule")
    prefixes = [
        values[offset : offset + PARITY_CHARACTERS].astype(np.int64).tolist()
        for offset in PROMPT_OFFSETS
    ]
    if any(not 0 <= value < len(alphabet) for prefix in prefixes for value in prefix):
        raise ValueError("heldout prefix contains an invalid identity")
    if sha256_file(path) != digest:
        raise ValueError("heldout data changed while loading prefixes")
    return prefixes, {
        "path": str(path),
        "sha256": digest,
        "metadata_sha256": sha256_file(root / "metadata.json"),
        "characters": int(values.size),
        "offsets": list(PROMPT_OFFSETS),
        "prefix_characters": PARITY_CHARACTERS,
    }


@torch.inference_mode()
def encoder_decision_parity(model, prefix_ids: list[int]) -> dict:
    """Untouched trained weights and real prefixes, including near-threshold gates.

    No global cache is necessary: this policy is a function of the persistent
    character encoder, not global state. This directly isolates the numerical
    parallel-scan versus serial-recurrence decision discrepancy.
    """
    device = model.table.weight.device
    source = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    parallel_local, parallel_gate, _ = model.encode(source)
    parallel_routes = model.choose_routes(parallel_gate, False)[0]
    accumulator, _ = embedder_generation._initial_local(model)
    serial_logits, serial_local = [], []
    for index in range(len(prefix_ids) - 1):
        accumulator, local, gate = embedder_generation._character_step(
            model,
            source[0, index : index + 1],
            accumulator,
        )
        serial_logits.append(gate)
        serial_local.append(local[0])
    serial_gate = torch.stack(serial_logits)
    serial_states = torch.stack(serial_local)
    if model.config.gate_mode == "fixed":
        positions = torch.arange(1, len(prefix_ids), device=device)
        serial_routes = positions % model.config.fixed_stride == 0
    else:
        serial_routes = serial_gate >= 0
    mismatch = (parallel_routes != serial_routes).nonzero().flatten()
    gate_delta = (parallel_gate[0] - serial_gate).abs()
    state_delta = (parallel_local[0, 1:].float() - serial_states.float()).abs()
    return {
        "prefix_ids_sha256": ids_sha256(prefix_ids),
        "scored_characters": len(prefix_ids),
        "decisions": len(prefix_ids) - 1,
        "mismatches": int(mismatch.numel()),
        "mismatch_rate": mismatch.numel() / (len(prefix_ids) - 1),
        "consumed_character_mismatch_positions": mismatch.tolist(),
        "parallel_logits_at_mismatches": parallel_gate[0, mismatch].tolist(),
        "incremental_logits_at_mismatches": serial_gate[mismatch].tolist(),
        "parallel_emissions": int(parallel_routes.sum().item()),
        "incremental_emissions": int(serial_routes.sum().item()),
        "gate_absolute_difference_max": float(gate_delta.max().item()),
        "gate_absolute_difference_mean": float(gate_delta.mean().item()),
        "state_absolute_difference_max": float(state_delta.max().item()),
        "state_absolute_difference_mean": float(state_delta.mean().item()),
        "parallel_gate_abs_below_0_01": int((parallel_gate.abs() < 0.01).sum().item()),
        "parallel_gate_abs_below_0_05": int((parallel_gate.abs() < 0.05).sum().item()),
        "incremental_encoder_positions": len(serial_logits),
        "parameter_interventions": False,
    }


def timed_workload(runtime, model, prompt, seed):
    # Both genuine runtimes use the same shared cached Mini arithmetic, sampler,
    # host .item() output collection and synchronization/timing boundaries.
    result = runtime.generate(model, list(prompt), NEW_CHARACTERS, TEMPERATURE, seed)
    if len(result["generated_ids"]) != NEW_CHARACTERS:
        raise RuntimeError("runtime did not return the full requested output")
    if result["accepted_characters"] != len(prompt) + NEW_CHARACTERS - 1:
        raise RuntimeError(
            "runtime consumed the final output or dropped required context"
        )
    if result["readout_positions"] != NEW_CHARACTERS:
        raise RuntimeError("runtime readout count differs from requested output")
    result["generated_ids_sha256"] = ids_sha256(result["generated_ids"])
    result["decode_characters_per_second"] = NEW_CHARACTERS / result["decode_seconds"]
    prefill_events = (
        sum(position < len(prompt) for position in result["emission_positions"])
        if "emission_positions" in result
        else None
    )
    prefill_positions = 1 + (
        prefill_events if prefill_events is not None else len(prompt)
    )
    result["prefill_counts"] = {
        "backbone_positions": prefill_positions,
        "backbone_calls": prefill_positions,
        "encoder_positions": len(prompt),
        "gate_positions": len(prompt) if prefill_events is not None else 0,
        "readout_positions": 0,
        "accepted_characters": len(prompt),
    }
    if prefill_events is not None:
        result["prefill_counts"]["emitted_events"] = prefill_events
    result["decode_counts"] = {
        name: result[name] - count for name, count in result["prefill_counts"].items()
    }
    return result


def main():
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark evidence: {output}")
    sources_before = current_source_hashes()
    checkpoints, models, checkpoint_records = load_variants(args)
    prefixes, heldout = load_heldout(args, checkpoints["baseline"])
    alphabet = checkpoints["baseline"]["alphabet"]
    prompts = [prefix[:PROMPT_CHARACTERS] for prefix in prefixes]
    report = {
        "schema": "nanogpt_mini_embedder_inference_benchmark_v1",
        "checkpoints": checkpoint_records,
        "source_hashes": sources_before,
        "heldout": heldout,
        "environment": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(models["baseline"].table.weight.device),
            "neural_dtype": "bfloat16",
            "autocast": False,
        },
        "protocol": {
            "prompts": 3,
            "repeats": REPEATS,
            "new_characters": NEW_CHARACTERS,
            "temperature": TEMPERATURE,
            "base_seed": BASE_SEED,
            "seed_schedule": "1337 + 3*prompt_index + repeat_index; identical across variants",
            "variant_order": "rotate [baseline,fixed,learned] by (3*prompt_index+repeat_index)%3",
            "warmup": "one complete exact prompt/seed/variant workload immediately before each measurement",
            "decode_timing": "synchronized wall time excluding prefill/allocation/random staging; includes sampling and output host synchronization",
            "prefill_timing": "separate synchronized allocation, BOS and sequential prompt ingestion, plus random staging",
            "cold_timing": "first use in this process per variant; compiler/disk caches may be shared, not an isolated cold-cache claim",
            "final_output_consumed": False,
            "metric_scope": "cached inference throughput, not training speed or heldout BPB",
        },
        "prompts": [
            {
                "index": index,
                "offset": PROMPT_OFFSETS[index],
                "characters": len(prompt),
                "ids_sha256": ids_sha256(prompt),
                "text": "".join(alphabet[value] for value in prompt),
            }
            for index, prompt in enumerate(prompts)
        ],
        "cold_first_use": {},
        "measurements": [],
        "encoder_decision_parity": {},
    }
    for prompt_index, prompt in enumerate(prompts):
        for repeat in range(REPEATS):
            rotation = (prompt_index * REPEATS + repeat) % len(VARIANTS)
            order = VARIANTS[rotation:] + VARIANTS[:rotation]
            seed = BASE_SEED + prompt_index * REPEATS + repeat
            for order_index, variant in enumerate(order):
                runtime = (
                    character_generation
                    if variant == "baseline"
                    else embedder_generation
                )
                warmup = timed_workload(runtime, models[variant], prompt, seed)
                if variant not in report["cold_first_use"]:
                    report["cold_first_use"][variant] = {
                        name: warmup[name]
                        for name in (
                            "elapsed_seconds",
                            "prefill_seconds",
                            "decode_seconds",
                            "generated_ids_sha256",
                        )
                    }
                measured = timed_workload(runtime, models[variant], prompt, seed)
                if measured["generated_ids"] != warmup["generated_ids"]:
                    raise RuntimeError(
                        f"{variant} exact warmup and measured outputs differ"
                    )
                report["measurements"].append(
                    {
                        "variant": variant,
                        "prompt_index": prompt_index,
                        "repeat_index": repeat,
                        "order_index": order_index,
                        "variant_order": list(order),
                        "seed": seed,
                        "warmup_elapsed_seconds": warmup["elapsed_seconds"],
                        "warmup_prefill_seconds": warmup["prefill_seconds"],
                        "warmup_decode_seconds": warmup["decode_seconds"],
                        **measured,
                    }
                )
                print(
                    f"{variant} prompt={prompt_index} repeat={repeat} "
                    f"decode_chars_s={measured['decode_characters_per_second']:.3f}",
                    flush=True,
                )
    # Run after generation timing so dense/parity compilation cannot prewarm the
    # nominal first-use decode measurements. No gate margins or weight edits.
    for variant in ("fixed", "learned"):
        report["encoder_decision_parity"][variant] = [
            {
                "prompt_index": index,
                "heldout_offset": PROMPT_OFFSETS[index],
                **encoder_decision_parity(models[variant], prefix),
            }
            for index, prefix in enumerate(prefixes)
        ]
    report["summary"] = {}
    for variant in VARIANTS:
        rows = [row for row in report["measurements"] if row["variant"] == variant]
        decode_seconds = sum(row["decode_seconds"] for row in rows)
        report["summary"][variant] = {
            "measured_workloads": len(rows),
            "generated_characters": len(rows) * NEW_CHARACTERS,
            "decode_seconds": decode_seconds,
            "aggregate_decode_characters_per_second": len(rows)
            * NEW_CHARACTERS
            / decode_seconds,
            "median_decode_characters_per_second": statistics.median(
                row["decode_characters_per_second"] for row in rows
            ),
            "prefill_seconds": sum(row["prefill_seconds"] for row in rows),
            "backbone_positions": sum(row["backbone_positions"] for row in rows),
            "encoder_positions": sum(row["encoder_positions"] for row in rows),
            "gate_positions": sum(row["gate_positions"] for row in rows),
            "decode_backbone_positions": sum(
                row["decode_counts"]["backbone_positions"] for row in rows
            ),
            "decode_encoder_positions": sum(
                row["decode_counts"]["encoder_positions"] for row in rows
            ),
            "decode_gate_positions": sum(
                row["decode_counts"]["gate_positions"] for row in rows
            ),
        }
    baseline_rate = report["summary"]["baseline"][
        "aggregate_decode_characters_per_second"
    ]
    for variant in ("fixed", "learned"):
        report["summary"][variant]["decode_throughput_ratio_to_baseline"] = (
            report["summary"][variant]["aggregate_decode_characters_per_second"]
            / baseline_rate
        )
    if current_source_hashes() != sources_before:
        raise RuntimeError(
            "benchmark sources changed during execution; refusing mixed-source evidence"
        )
    for record in checkpoint_records.values():
        if sha256_file(record["path"]) != record["sha256"]:
            raise RuntimeError("checkpoint changed during benchmark")
    report["status"] = "complete"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()

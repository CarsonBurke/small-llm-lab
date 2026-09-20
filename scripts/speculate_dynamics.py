"""Exact frozen-dynamics generation and matched sampling benchmarks (CUDA via mlq).

Example: mlq submit --name exact-dynamics --max-parallel-runs 1 --priority 1 --time-limit 2h \
    -- python scripts/speculate_dynamics.py benchmark --checkpoint CHECKPOINT \
    --prompt 'The ' --characters 256 --output results/exact-dynamics.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.nanogpt_mini.native_bits_data import encode_text, sha256_file
from scripts.native_bits import _fresh_output, _load_pinned_checkpoint


def _validate(args: argparse.Namespace) -> None:
    if args.characters < 0:
        raise ValueError("--characters must be nonnegative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("--temperature must be finite and nonnegative")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must fit an unsigned 32-bit integer")
    budgets = args.draft_tokens if args.command == "benchmark" else [args.draft_tokens]
    if any(budget < 0 for budget in budgets):
        raise ValueError("--draft-tokens must be nonnegative")
    if args.command == "benchmark":
        if args.characters < 1:
            raise ValueError("benchmark requires --characters greater than zero")
        if args.repeats < 1 or args.warmup < 1:
            raise ValueError("--repeats and --warmup must both be at least one")


def _metadata(checkpoint: dict, digest: str, path: Path, device) -> dict:
    import torch

    from pretraining.nanogpt_mini.nanogpt_mini_native_bits_train import (
        checkpoint_model_kind,
    )

    properties = torch.cuda.get_device_properties(device)
    source_paths = [Path(__file__)] + [
        REPO_ROOT / "pretraining" / "nanogpt_mini" / name
        for name in (
            "speculative_generation.py",
            "dynamics_generation.py",
            "bit_density.py",
            "nanogpt_mini_dynamics_model.py",
            "nanogpt_mini_model.py",
        )
    ]
    model_kind = checkpoint_model_kind(checkpoint)
    if model_kind == "dynamics_bounded":
        source_paths.append(
            REPO_ROOT
            / "pretraining/nanogpt_mini/nanogpt_mini_dynamics_bounded_model.py"
        )
    return {
        "schema": "exact_dynamics_speculation_v1",
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "architecture": checkpoint["architecture"],
        "model_kind": model_kind,
        "checkpoint_step": checkpoint["step"],
        "model_config": checkpoint["model_config"],
        "train_config": checkpoint["train_config"],
        "provenance": checkpoint["provenance"],
        "alphabet": checkpoint["alphabet"],
        "inference_source_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_paths
        },
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "gpu": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "gpu_total_memory_bytes": properties.total_memory,
            "neural_precision": "bfloat16",
            "probability_precision": "float32",
            "compiled": True,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        },
        "semantics": {
            "target": "Frozen dense fixed-prefix character teacher; no learned KL gate.",
            "proposal": "Frozen latent transition; draft_tokens changes inference only.",
            "acceptance": "Ordered Bernoulli p/q acceptance and exact binary residual correction.",
            "distribution": (
                "All budgets target the same autoregressive distribution, subject to "
                "floating-point implementation. Sampling seeds do not imply identical "
                "stochastic text across budgets. This is not a model-quality/BPB comparison."
            ),
            "reserved_addresses": (
                "Full binary domain; stop at the first committed reserved address, "
                "record its code/address once, exclude it from IDs/text, never resample."
            ),
            "compute": (
                "target_calls/target_positions count decode verification, including "
                "rejected and wasted suffixes; prefill is reported separately. "
                "Accepted/proposed draft counts are character attempts, not accepted bits."
            ),
        },
    }


def _run(
    model, alphabet: list[str], prompt: str, prompt_ids: list[int], args, budget, seed
):
    from pretraining.nanogpt_mini.speculative_generation import generate

    result = generate(
        model,
        prompt_ids=prompt_ids,
        max_new_characters=args.characters,
        temperature=args.temperature,
        seed=seed,
        draft_tokens=budget,
        trace=False,
    )
    generated_text = "".join(alphabet[identity] for identity in result["generated_ids"])
    full_length = (
        result["stop_reason"] == "output_limit"
        and result["reserved_code"] is None
        and result["valid_generated_characters"] == args.characters
    )
    return {
        **result,
        "prompt": prompt,
        "generated_text": generated_text,
        "text": prompt + generated_text,
        "seed": seed,
        "temperature": args.temperature,
        "full_length": full_length,
        "valid_characters_per_decode_second": (
            result["valid_generated_characters"] / result["decode_seconds"]
            if result["decode_seconds"] > 0
            else None
        ),
    }


def _summary(records: list[dict], budgets: list[int]) -> list[dict]:
    baseline = {
        (record["prompt_index"], record["repeat"]): record
        for record in records
        if record["draft_tokens"] == 0
    }
    summaries = []
    for budget in budgets:
        selected = [record for record in records if record["draft_tokens"] == budget]
        paired = [
            baseline[(record["prompt_index"], record["repeat"])] for record in selected
        ]
        complete_pairs = all(
            record["full_length"] and target["full_length"]
            for record, target in zip(selected, paired, strict=True)
        )
        complete = [record for record in selected if record["full_length"]]
        proposed = sum(record["proposed_draft_characters"] for record in selected)
        accepted = sum(record["accepted_draft_characters"] for record in selected)
        median_decode = statistics.median(
            record["decode_seconds"] for record in selected
        )
        median_target = statistics.median(record["decode_seconds"] for record in paired)
        speedup = (
            median_target / median_decode
            if complete_pairs and median_decode > 0
            else None
        )
        summaries.append(
            {
                "draft_tokens": budget,
                "runs": len(selected),
                "complete_runs": len(complete),
                "incomplete_runs": len(selected) - len(complete),
                "all_matched_pairs_full_length": complete_pairs,
                "median_warm_decode_seconds": median_decode,
                "median_prefill_seconds": statistics.median(
                    record["prefill_seconds"] for record in selected
                ),
                "median_end_to_end_seconds": statistics.median(
                    record["elapsed_seconds"] for record in selected
                ),
                "median_full_length_characters_per_decode_second": (
                    statistics.median(
                        record["valid_characters_per_decode_second"]
                        for record in complete
                    )
                    if len(complete) == len(selected)
                    and all(
                        record["valid_characters_per_decode_second"] is not None
                        for record in complete
                    )
                    else None
                ),
                "speedup_vs_target_only": speedup,
                "speedup_unavailable_reason": (
                    None
                    if speedup is not None
                    else "At least one matched run terminated before the requested valid character count, or decode time was zero."
                ),
                "proposed_draft_characters": proposed,
                "accepted_draft_characters": accepted,
                "draft_character_acceptance_rate": accepted / proposed
                if proposed
                else None,
                "corrected_characters": sum(
                    record["corrected_characters"] for record in selected
                ),
                "bonus_characters": sum(
                    record["bonus_characters"] for record in selected
                ),
                "target_calls": sum(record["target_calls"] for record in selected),
                "target_positions": sum(
                    record["target_positions"] for record in selected
                ),
                "valid_generated_characters": sum(
                    record["valid_generated_characters"] for record in selected
                ),
                "prefill_calls": sum(record["prefill_calls"] for record in selected),
                "prefill_positions": sum(
                    record["prefill_positions"] for record in selected
                ),
            }
        )
    return summaries


def _benchmark(model, alphabet, prompts, prompt_ids, args) -> dict:
    budgets = sorted({0, *args.draft_tokens})
    schedule = [
        {
            "prompt_index": prompt_index,
            "repeat": repeat,
            "seed": (args.seed + prompt_index * args.repeats + repeat) % 2**32,
        }
        for repeat in range(args.repeats)
        for prompt_index in range(len(prompts))
    ]
    warmups, records = [], []
    seen_settings = set()
    # Replay every measured seed/shape before timing, not just one lucky
    # acceptance path. Shared kernels may already be warm for a new setting.
    for warmup in range(args.warmup):
        for workload in schedule:
            index = workload["prompt_index"]
            for budget in budgets:
                result = _run(
                    model,
                    alphabet,
                    prompts[index],
                    prompt_ids[index],
                    args,
                    budget,
                    workload["seed"],
                )
                warmups.append(
                    {
                        **result,
                        **workload,
                        "warmup": warmup,
                        "first_call_for_setting": budget not in seen_settings,
                        "first_model_generation_call": not warmups,
                    }
                )
                seen_settings.add(budget)
    for workload_index, workload in enumerate(schedule):
        index = workload["prompt_index"]
        offset = workload_index % len(budgets)
        order = budgets[offset:] + budgets[:offset]
        for budget in order:
            result = _run(
                model,
                alphabet,
                prompts[index],
                prompt_ids[index],
                args,
                budget,
                workload["seed"],
            )
            records.append({**result, **workload, "execution_index": len(records)})
    return {
        "settings": {
            "draft_tokens": budgets,
            "characters": args.characters,
            "temperature": args.temperature,
            "base_seed": args.seed,
            "repeats": args.repeats,
            "warmup_passes_per_exact_workload": args.warmup,
            "prompts": prompts,
            "prompt_ids": prompt_ids,
            "seed_schedule": schedule,
        },
        "timing_policy": {
            "synchronization": "Runtime CUDA synchronization brackets setup, prefill, and decode.",
            "warmup": "Each budget/prompt/repeat seed is replayed before any measured runs; no trace collection.",
            "order": "Measured budget order rotates once per prompt/repeat workload.",
            "cold": (
                "First model generation is process-cold. First call for each setting "
                "may reuse previously compiled kernels; all warmup raw times retained."
            ),
            "warm": (
                "Post-warmup timings retain any compiler work still incurred; "
                "compilation-free execution is not assumed or subtracted."
            ),
            "speedup": (
                "Ratio of median matched target-only decode latency to setting decode latency. "
                "Published only when EVERY paired run emits the requested valid character count. "
                "Incomplete results remain in raw records and latency summaries, but never "
                "qualify for full-length throughput or speedup."
            ),
        },
        "cold_calls": [
            record for record in warmups if record["first_call_for_setting"]
        ],
        "warmup_runs": warmups,
        "raw_repeats": records,
        "summary": _summary(records, budgets),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "benchmark"):
        subparser = commands.add_parser(
            command, help=f"{command} exact cached sampling; queue CUDA through mlq"
        )
        subparser.add_argument("--checkpoint", required=True)
        subparser.add_argument(
            "--output",
            required=True,
            help="new JSON artifact; existing paths are never overwritten",
        )
        subparser.add_argument(
            "--prompt",
            required=True,
            **({"action": "append"} if command == "benchmark" else {}),
        )
        subparser.add_argument("--characters", type=int, default=256)
        subparser.add_argument("--temperature", type=float, default=0.8)
        subparser.add_argument("--seed", type=int, default=1337)
        if command == "benchmark":
            subparser.add_argument(
                "--draft-tokens",
                type=int,
                nargs="+",
                default=[1, 2, 4],
                help="proposal budgets; target-only 0 is always included",
            )
            subparser.add_argument("--repeats", type=int, default=3)
            subparser.add_argument("--warmup", type=int, default=1)
        else:
            subparser.add_argument(
                "--draft-tokens",
                type=int,
                default=2,
                help="proposal budget; 0 selects matched target-only sampling",
            )
    args = parser.parse_args()
    _validate(args)
    output = _fresh_output(args.output)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve(strict=True)
    checkpoint, digest = _load_pinned_checkpoint(checkpoint_path)
    from pretraining.nanogpt_mini.nanogpt_mini_native_bits_train import (
        DYNAMICS_MODEL_KINDS,
        checkpoint_model_kind,
        load_model,
    )

    # Inspect architecture and opaque-alphabet prompts before constructing any
    # CUDA model. A control checkpoint is not a latent draft model.
    if checkpoint_model_kind(checkpoint) not in DYNAMICS_MODEL_KINDS:
        raise ValueError("exact speculation requires a frozen dynamics checkpoint")
    prompts = args.prompt if args.command == "benchmark" else [args.prompt]
    prompt_ids = [
        encode_text(prompt, checkpoint["alphabet"]).tolist() for prompt in prompts
    ]
    model, device = load_model(checkpoint)
    model.requires_grad_(False)
    report = {
        **_metadata(checkpoint, digest, checkpoint_path, device),
        "command": args.command,
        "output": str(output),
    }
    if args.command == "benchmark":
        report.update(
            _benchmark(model, checkpoint["alphabet"], prompts, prompt_ids, args)
        )
    else:
        report["result"] = _run(
            model,
            checkpoint["alphabet"],
            prompts[0],
            prompt_ids[0],
            args,
            args.draft_tokens,
            args.seed,
        )
        report["timing_policy"] = (
            "Single cold generation; compilation, setup and prefill are reported, not hidden. Use benchmark for warm speedups."
        )
    serialized = (
        json.dumps(
            report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    print(
        json.dumps(
            {
                "output": str(output),
                "checkpoint_sha256": digest,
                "command": args.command,
                "summary": report.get("summary"),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

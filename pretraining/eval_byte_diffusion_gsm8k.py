"""Few-shot GSM8K for causal and cached Fast-BLT byte checkpoints.

The causal mode is a correctness adapter that recomputes the complete prefix.
BLT mode uses the incremental hierarchical cache and reports every prefill,
alignment, and denoising forward. GPU execution must be submitted via mlq.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import Tensor

from pretraining.byte_diffusion.data import AtomicIdManifest
from pretraining.byte_diffusion.inference import CachedCanvasGenerator
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.state import TransactionalDecodeState
from pretraining.byte_diffusion.training import (
    CHECKPOINT_SCHEMA,
    model_config_from_dict,
)
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


EVALUATOR_SCHEMA = "byte_diffusion_gsm8k/v2"


@dataclass(frozen=True)
class ByteGeneration:
    raw: bytes
    text: str | None
    termination: str
    native_actions: int
    invalid_utf8: bool


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
    storage_width = max(prompt_lengths) + max_new_bytes
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

    for _ in range(max_native_actions):
        if all(finished):
            break
        span = int(cursor.max())
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
            ).logits
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
                invalid_utf8=invalid,
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
    adaptive_confidence: float | None,
    min_diffusion_steps: int,
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
            if remaining_actions <= alignment_overhead:
                break
            generation = generator.generate_blt(
                block_length,
                min(diffusion_steps, remaining_actions - alignment_overhead),
                adaptive_confidence=adaptive_confidence,
                min_steps=min_diffusion_steps,
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
                invalid_utf8=invalid,
            )
        )
    return results


def load_byte_checkpoint(
    path: Path, device: torch.device, *, decode_mode: str
) -> tuple[ByteDiffusionModel, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("GSM8K requires a byte-diffusion training checkpoint")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shots", default="5")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--prompt-format", choices=sorted(PROMPT_FORMATS), default="harness"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-mode", choices=("causal", "blt"), default="blt")
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--adaptive-confidence", type=float)
    parser.add_argument("--min-diffusion-steps", type=int, default=1)
    parser.add_argument("--generation-seed", type=int, default=12345)
    parser.add_argument("--max-new-bytes", type=int, default=512)
    parser.add_argument("--max-native-actions", type=int, default=512)
    parser.add_argument("--context-bytes", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=8)
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
    if not 1 <= args.min_diffusion_steps <= args.diffusion_steps:
        raise ValueError("--min-diffusion-steps must lie in [1, --diffusion-steps]")
    if args.adaptive_confidence is not None and not (
        0.0 < args.adaptive_confidence <= 1.0
    ):
        raise ValueError("--adaptive-confidence must lie in (0, 1]")
    device = torch.device("cuda")
    model, payload = load_byte_checkpoint(
        args.checkpoint, device, decode_mode=args.decode_mode
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
            generated: list[ByteGeneration] = []
            generation_started = time.perf_counter()
            eval_batch_size = args.batch_size if args.decode_mode == "causal" else 1
            for start in range(0, len(prompts), eval_batch_size):
                selected_prompts = prompts[start : start + eval_batch_size]
                if args.decode_mode == "causal":
                    batch_generations = greedy_generate_bytes(
                        model,
                        selected_prompts,
                        max_new_bytes=args.max_new_bytes,
                        max_native_actions=args.max_native_actions,
                        context_bytes=args.context_bytes,
                        stops=prompt_format.stops,
                        device=device,
                    )
                else:
                    batch_generations = blt_generate_bytes(
                        model,
                        selected_prompts,
                        max_new_bytes=args.max_new_bytes,
                        max_native_actions=args.max_native_actions,
                        context_bytes=args.context_bytes,
                        stops=prompt_format.stops,
                        block_length=args.block_length,
                        diffusion_steps=args.diffusion_steps,
                        adaptive_confidence=args.adaptive_confidence,
                        min_diffusion_steps=args.min_diffusion_steps,
                        seed=(
                            args.generation_seed
                            + 10_000_019 * seed
                            + 97_409 * start
                        ),
                        device=device,
                    )
                generated.extend(batch_generations)
                print(
                    f"shots={shots} seed={seed} scored={len(generated)}/{len(prompts)}",
                    flush=True,
                )
            generation_seconds = time.perf_counter() - generation_started
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
                    "native_actions": sum(
                        int(row["native_actions"]) for row in rows
                    ),
                    "mean_prompt_bytes": sum(map(len, prompts)) / len(prompts),
                    "max_prompt_bytes": max(map(len, prompts)),
                    "termination_counts": {
                        reason: sum(row["termination"] == reason for row in rows)
                        for reason in sorted({row["termination"] for row in rows})
                    },
                    "rows": rows,
                }
            )
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    result = {
        "schema": EVALUATOR_SCHEMA,
        "implementation_maturity": (
            "full_prefix_recompute_correctness_adapter"
            if args.decode_mode == "causal"
            else "incremental_fast_blt_cache"
        ),
        "decode_mode": args.decode_mode,
        "block_length": args.block_length if args.decode_mode == "blt" else None,
        "diffusion_steps": args.diffusion_steps if args.decode_mode == "blt" else None,
        "adaptive_confidence": (
            args.adaptive_confidence if args.decode_mode == "blt" else None
        ),
        "min_diffusion_steps": (
            args.min_diffusion_steps if args.decode_mode == "blt" else None
        ),
        "generation_seed": args.generation_seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_schema": payload["schema"],
        "completed_steps": payload["completed_steps"],
        "model_parameters": model.parameter_count(),
        "gsm8k_snapshot": str(GSM8K_ROOT),
        "gsm8k_train_sha256": sha256(TRAIN_PARQUET),
        "gsm8k_test_sha256": sha256(TEST_PARQUET),
        "prompt_format": prompt_format.name,
        "max_new_bytes": args.max_new_bytes,
        "max_native_actions": args.max_native_actions,
        "context_bytes": args.context_bytes,
        "requested_batch_size": args.batch_size,
        "effective_batch_size": args.batch_size if args.decode_mode == "causal" else 1,
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

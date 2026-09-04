#!/usr/bin/env python3
"""Generate a fixed-policy MiniCPM critic corpus with continuous lane refill."""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from checkpointing import atomic_torch_save
from postraining.core import answer_style, load_unique_math_rows, verify_answer
from postraining.fast_inference import CapturedTrainingRolloutEngine
from postraining.minicpm_vapo import (
    LoRAConfig,
    MiniCPMVAPOPolicy,
    load_adapter_state_dict,
)
from postraining.runtime.profiling import DeviceSampler
from postraining.train_minicpm_vapo import (
    _decode_text,
    _stop_ids,
    device_phase_metrics,
    encode_math_prompt,
    file_sha256,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", type=int, default=160)
    parser.add_argument("--prompt-tokens", type=int, default=1_024)
    parser.add_argument("--max-new-tokens", type=int, default=8_192)
    parser.add_argument("--capacity-prompts", type=int, default=4)
    parser.add_argument("--prefill-batch-prompts", type=int, default=8)
    parser.add_argument("--completion-poll-steps", type=int, default=16)
    parser.add_argument("--min-productive-utilization", type=float, default=0.90)
    parser.add_argument("--min-gpu-utilization", type=float, default=90.0)
    parser.add_argument("--device-telemetry-interval-ms", type=int, default=250)
    parser.add_argument("--device-power-floor", type=float, default=400.0)
    return parser


def _validate_args(args: argparse.Namespace, samples_per_prompt: int) -> None:
    positive = (
        args.prompts,
        args.prompt_tokens,
        args.max_new_tokens,
        args.capacity_prompts,
        args.prefill_batch_prompts,
        args.completion_poll_steps,
        args.device_telemetry_interval_ms,
    )
    if any(value < 1 for value in positive):
        raise ValueError("corpus generation dimensions must be positive")
    if args.prompts < 2:
        raise ValueError("critic corpus requires at least two prompts")
    if not 0.0 < args.min_productive_utilization <= 1.0:
        raise ValueError("productive utilization gate must lie in (0, 1]")
    if not 0.0 <= args.min_gpu_utilization <= 100.0:
        raise ValueError("GPU utilization gate must lie in [0, 100]")
    capacity_rows = args.capacity_prompts * samples_per_prompt
    if capacity_rows < samples_per_prompt:
        raise ValueError("decode capacity must fit one complete prompt group")


def _score_response(tokenizer, row: dict, response: torch.Tensor) -> bool:
    text = _decode_text(tokenizer, response)
    correct, _ = verify_answer(
        text,
        row["reward_model"]["ground_truth"],
        answer_style(row),
    )
    return bool(correct)


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("pending_records") is not None or checkpoint.get(
        "pending_epoch", 0
    ) != 0:
        raise ValueError(
            "critic corpus source must be a replay-boundary checkpoint"
        )
    checkpoint_args = checkpoint["args"]
    samples_per_prompt = int(checkpoint_args["samples_per_prompt"])
    _validate_args(args, samples_per_prompt)
    data_path = Path(checkpoint_args["data"])
    data_sha256 = file_sha256(data_path)
    if data_sha256 != checkpoint["data_sha256"]:
        raise ValueError("critic corpus dataset differs from checkpoint dataset")

    rows = load_unique_math_rows(data_path)
    order_rng = random.Random(int(checkpoint_args["seed"]))
    order_rng.shuffle(rows)
    cursor_start = int(checkpoint["cursor"])
    selected_rows = [
        rows[(cursor_start + index) % len(rows)] for index in range(args.prompts)
    ]

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checkpoint_payload = checkpoint["policy"]
    if checkpoint_payload.get("schema") != "minicpm5_vapo_adapter/v6":
        raise ValueError("critic corpus generation requires a v6 checkpoint")
    actor = checkpoint_payload["actor"]
    lora_payload = dict(actor["lora_config"])
    lora_payload["targets"] = tuple(lora_payload["targets"])
    lora_config = LoRAConfig(**lora_payload)
    policy, tokenizer = MiniCPMVAPOPolicy.from_pretrained(
        model_id=actor["model_id"],
        revision=actor["revision"],
        device=device,
        lora_config=lora_config,
        nextlat_projection_factor=float(actor["nextlat_projection_factor"]),
        gradient_checkpointing=False,
    )
    load_adapter_state_dict(policy.causal_lm, actor["adapter"])
    policy.eval()
    prompt_ids = [
        encode_math_prompt(
            tokenizer,
            row,
            prompt_tokens=args.prompt_tokens,
            enable_thinking=bool(checkpoint_args["thinking"]),
        )
        for row in selected_rows
    ]
    maximum_prompt = max(int(prompt.numel()) for prompt in prompt_ids)
    cache_length = maximum_prompt + args.max_new_tokens
    engine = CapturedTrainingRolloutEngine(
        policy,
        stop_ids=_stop_ids(policy, tokenizer),
        prompts_per_rollout=args.capacity_prompts,
        samples_per_prompt=samples_per_prompt,
        cache_length=cache_length,
        temperature=float(checkpoint_args["temperature"]),
        top_k=int(checkpoint_args["top_k"]),
        top_p=float(checkpoint_args["top_p"]),
        compile_decode=bool(checkpoint_args["compile_rollout"]),
    )

    sampler = DeviceSampler(args.device_telemetry_interval_ms, device)
    sampler.start()
    atexit.register(sampler.stop)
    started = time.perf_counter()

    def report(step: int, metrics: dict[str, float | int]) -> None:
        print(json.dumps({"type": "rollout_live", "step": step, **metrics}), flush=True)

    generation = engine.generate_prompt_pool(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        prefill_batch_prompts=args.prefill_batch_prompts,
        completion_poll_steps=args.completion_poll_steps,
        progress_callback=report,
    )
    ended = time.perf_counter()
    sampler.stop()

    expected_rows = args.prompts * samples_per_prompt
    if len(generation.responses) != expected_rows:
        raise RuntimeError("continuous rollout returned the wrong trajectory count")
    invalid = [
        index
        for index, response in enumerate(generation.responses)
        if bool(((response < 0) | (response >= len(tokenizer))).any())
    ]
    if invalid:
        raise RuntimeError(f"rollout produced invalid token IDs in rows {invalid[:8]}")

    records: list[dict] = []
    with ThreadPoolExecutor(
        max_workers=min(args.prompts, 32),
        thread_name_prefix="critic-corpus-score",
    ) as scoring_pool:
        futures = []
        for prompt_index, row in enumerate(selected_rows):
            for sample in range(samples_per_prompt):
                response_index = prompt_index * samples_per_prompt + sample
                response = generation.responses[response_index]
                futures.append(
                    (
                        prompt_index,
                        response,
                        scoring_pool.submit(
                            _score_response, tokenizer, row, response
                        ),
                    )
                )
        for prompt_index, response, future in futures:
            records.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_ids": prompt_ids[prompt_index].to(torch.int32),
                    "response_ids": response.to(torch.int32),
                    "correct": future.result(),
                }
            )

    correct = sum(int(record["correct"]) for record in records)
    telemetry = device_phase_metrics(
        sampler,
        started=started,
        ended=ended,
        power_floor=args.device_power_floor,
    )
    metrics = {
        "elapsed_seconds": ended - started,
        "prompts": args.prompts,
        "samples_per_prompt": samples_per_prompt,
        "trajectories": len(records),
        "correct_trajectories": correct,
        "accuracy": correct / len(records),
        "generated_tokens": generation.useful_tokens,
        "decode_steps": generation.decode_steps,
        "capacity_row_steps": generation.capacity_row_steps,
        "productive_utilization": generation.productive_utilization,
        "admission_events": generation.admission_events,
        "minimum_active_rows_with_backlog": (
            generation.minimum_active_rows_with_backlog
        ),
        **telemetry,
    }
    minimum_refilled_rows = engine.batch_size
    passed = (
        generation.productive_utilization
        >= args.min_productive_utilization
        and generation.minimum_active_rows_with_backlog
        >= minimum_refilled_rows
        and float(metrics.get("device_utilization_gpu_percent_mean", 0.0))
        >= args.min_gpu_utilization
        and 0 < correct < len(records)
    )
    corpus = {
        "schema": "minicpm_critic_corpus/v1",
        "model_id": actor["model_id"],
        "revision": actor["revision"],
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_step": int(checkpoint["step"]),
        "data_sha256": data_sha256,
        "cursor_start": cursor_start,
        "cursor_end": cursor_start + args.prompts,
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "completion_poll_steps": args.completion_poll_steps,
        "samples_per_prompt": samples_per_prompt,
        "seed": int(checkpoint_args["seed"]),
        "metrics": metrics,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(corpus, output)
    metrics_path = output.with_suffix(output.suffix + ".metrics.json")
    metrics_path.write_text(
        json.dumps({"passed": passed, **metrics}, indent=2, sort_keys=True)
        + "\n"
    )
    print(json.dumps({"passed": passed, **metrics}, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

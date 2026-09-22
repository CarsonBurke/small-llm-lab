#!/usr/bin/env python3
"""Fixed-work real-CUDA latent/native rollout and independent replay benchmark.

Run through mlq with --max-parallel-runs 1. Stream positions are not lexical
tokens. Component replay timings isolate likelihood/value forward+backward.
Mode all also measures warmed rollout/refresh/actor-critic optimizer cycles.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional strict latent/v1 actor/critic checkpoint; otherwise initialize once from the pinned MiniCPM5 base.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--host-reference",
        type=Path,
        help="Preserved host-scheduled runtime snapshot; required only for the host arm.",
    )
    parser.add_argument(
        "--expected-sources",
        type=Path,
        help="Reject queued runs if source fingerprints changed before admission.",
    )
    parser.add_argument(
        "--mode", choices=("all", "inference", "replay", "qualify"), default="all"
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=("host", "optimized", "native"),
        default=["optimized", "native"],
    )
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--physical-batch-size", type=int, default=4)
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--thought-steps", type=int, default=32)
    parser.add_argument("--answer-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--compile-replay", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--replay-batch-size", type=int, default=1)
    parser.add_argument("--optimizer-minibatches", type=int, default=1)
    parser.add_argument("--replay-checkpoint-interval", type=int, default=0)
    parser.add_argument("--logit-chunk-tokens", type=int, default=128)
    parser.add_argument("--nextlat-samples", type=int, default=64)
    parser.add_argument("--nextlat-horizon", type=int, default=2)
    parser.add_argument("--nextlat-kl-chunk-tokens", type=int, default=16)
    parser.add_argument("--qualification-atol", type=float, default=0.1)
    parser.add_argument("--qualification-rtol", type=float, default=0.02)
    parser.add_argument(
        "--prompt",
        default="Prove that the square root of 2 is irrational, with complete reasoning.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> dict[str, int]:
    if args.output.suffix != ".json":
        raise ValueError(
            "output must use .json; the paired .records.pt stores exact replay actions"
        )
    if args.checkpoint is not None and args.checkpoint.resolve() in {
        args.output.resolve(),
        args.output.with_suffix(".records.pt").resolve(),
    }:
        raise ValueError("benchmark output must not overwrite its checkpoint")
    if args.output.exists() or args.output.with_suffix(".records.pt").exists():
        raise ValueError("benchmark output already exists; choose a new evidence path")
    for name in (
        "prompts",
        "samples_per_prompt",
        "physical_batch_size",
        "context_tokens",
        "thought_steps",
        "answer_tokens",
        "warmups",
        "repetitions",
        "replay_batch_size",
        "logit_chunk_tokens",
        "nextlat_samples",
        "nextlat_horizon",
        "nextlat_kl_chunk_tokens",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    logical = args.prompts * args.samples_per_prompt
    if args.physical_batch_size > logical:
        raise ValueError("physical batch cannot exceed logical batch")
    if args.replay_batch_size > logical or logical % args.replay_batch_size:
        raise ValueError(
            "replay batch must divide logical batch: no unequal final microbatch"
        )
    if len(set(args.engines)) != len(args.engines):
        raise ValueError("engines must be unique")
    if "host" in args.engines and (
        args.host_reference is None or not args.host_reference.is_file()
    ):
        raise ValueError("host arm requires an existing --host-reference snapshot")
    if not 1 <= args.optimizer_minibatches <= logical:
        raise ValueError("optimizer minibatches must fit the logical rollout batch")
    if args.replay_checkpoint_interval < 0:
        raise ValueError("replay checkpoint interval must be nonnegative")
    if (
        not math.isfinite(args.temperature)
        or not args.temperature > 0
        or not 0 < args.top_p <= 1
        or args.top_k < 1
    ):
        raise ValueError("invalid sampling configuration")
    if not all(
        math.isfinite(value) and value > 0
        for value in (args.qualification_atol, args.qualification_rtol)
    ):
        raise ValueError("qualification tolerances must be positive")
    stream = args.thought_steps + 1 + args.answer_tokens
    return {
        "logical_batch": logical,
        "physical_batch": args.physical_batch_size,
        "prompt_tokens_per_row": args.context_tokens,
        "stream_positions_per_row": stream,
        "total_stream_positions": logical * stream,
        "latent_thought_actions": logical * args.thought_steps,
        "latent_close_actions": logical,
        "latent_lexical_tokens": logical * args.answer_tokens,
        "native_lexical_tokens": logical * stream,
        "cache_length": args.context_tokens + stream,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensors: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def fingerprints(args: argparse.Namespace) -> dict[str, str]:
    paths = [
        Path(__file__),
        *(
            REPO_ROOT / "postraining" / name
            for name in (
                "minicpm_latent_rollout.py",
                "minicpm_vapo.py",
                "fast_inference.py",
                "latent_thought.py",
                "train_minicpm_vapo.py",
                "invariant_linear.py",
                "invariant_attention.py",
            )
        ),
    ]
    if "host" in args.engines:
        paths.append(args.host_reference.resolve())
    return {str(path.resolve()): file_sha256(path) for path in paths}


def load_policy(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    from postraining.vapo.model.hf import MINICPM5_SPEC
    from postraining.vapo.model.lora import (
        LoRAConfig,
        load_adapter_state_dict,
    )
    from postraining.vapo.policy import VAPOPolicy

    payload = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        payload = checkpoint["policy"]
        if payload.get("schema") != "minicpm5_vapo_latent/v1":
            raise ValueError("benchmark requires a latent/v1 actor/critic checkpoint")
        actor, critic = payload["actor"], payload["critic"]
        if not actor.get("latent_thinking", False) or not critic.get(
            "latent_thinking", False
        ):
            raise ValueError(
                "checkpoint requires latent actor and critic state; no implicit head initialization or dropping"
            )
        if (actor["model_id"], actor["revision"]) != (
            critic["model_id"],
            critic["revision"],
        ):
            raise ValueError(
                "actor and critic must reference identical frozen base weights"
            )
    else:
        actor = {
            "model_id": MINICPM5_SPEC.model_id,
            "revision": MINICPM5_SPEC.revision,
            "lora_config": {"initialization": "nora"},
            "nextlat_projection_factor": 1.6,
            "thought_sigma": 1.0,
            "init_stop_thinking_probability": 0.9,
        }
    policy, tokenizer = VAPOPolicy.from_family("minicpm5", 
        model_id=actor["model_id"],
        revision=actor["revision"],
        device=torch.device("cuda"),
        lora_config=LoRAConfig(**actor["lora_config"]),
        nextlat_projection_factor=float(actor["nextlat_projection_factor"]),
        gradient_checkpointing=True,
        latent_thinking=True,
        thought_sigma=float(actor["thought_sigma"]),
        init_stop_thinking_probability=float(actor["init_stop_thinking_probability"]),
    )
    if payload is not None:
        load_adapter_state_dict(policy.causal_lm, actor["adapter"])
        policy.nextlat_head.load_state_dict(actor["nextlat"], strict=True)
        policy.load_latent_state_dict(actor)
    else:
        payload = {"actor": policy.checkpoint_payload(), "critic": None}
    policy.eval()
    return policy, tokenizer, payload


def load_critic(payload: dict[str, Any], policy: Any, args: argparse.Namespace) -> Any:
    from postraining.vapo.model.lora import (
        LoRAConfig,
        load_adapter_state_dict,
    )
    from postraining.vapo.policy import VAPOCritic

    saved = payload["critic"]
    config = saved if saved is not None else payload["actor"]
    # Independent deterministic initialization, unaffected by rollout RNG usage.
    torch.manual_seed(args.seed + 1)
    critic = VAPOCritic.from_family("minicpm5", 
        model_id=config["model_id"],
        revision=config["revision"],
        device=torch.device("cuda"),
        lora_config=LoRAConfig(**config["lora_config"]),
        critic_width=saved["value_head"]["input.weight"].shape[0]
        if saved is not None
        else 256,
        nextlat_projection_factor=float(config["nextlat_projection_factor"]),
        gradient_checkpointing=True,
        shared_frozen_source=policy.causal_lm,
        latent_thinking=True,
    )
    if saved is not None:
        load_adapter_state_dict(critic.causal_lm, saved["adapter"])
        critic.nextlat_head.load_state_dict(saved["nextlat"], strict=True)
        critic.value_head.load_state_dict(saved["value_head"], strict=True)
        critic.load_latent_state_dict(saved)
    else:
        critic.thought_adapter.load_state_dict(
            policy.thought_adapter.state_dict(), strict=True
        )
    return critic


def make_prompts(tokenizer: Any, args: argparse.Namespace) -> list[torch.Tensor]:
    from postraining.validate_minicpm_vapo import prompt_ids

    base = prompt_ids(tokenizer, args.prompt, torch.device("cpu")).flatten()
    if base.numel() > args.context_tokens:
        raise ValueError("context-tokens must fit the complete native prompt")
    # Synthetic equal-length context, not a quality evaluation. Preserve the exact
    # native chat-template suffix; the repeated background is left-truncated.
    repeats = 1
    expanded = base
    while expanded.numel() < args.context_tokens:
        expanded = prompt_ids(
            tokenizer,
            "Background: mathematical reasoning. " * repeats + args.prompt,
            torch.device("cpu"),
        ).flatten()
        repeats *= 2
    prompt = expanded[-args.context_tokens :].contiguous()
    return [prompt.clone() for _ in range(args.prompts)]


@contextmanager
def fixed_gate(policy: Any):
    weight = policy.thinking_gate.head.weight
    bias = policy.thinking_gate.head.bias
    saved_weight, saved_bias = weight.detach().clone(), bias.detach().clone()
    with torch.no_grad():
        weight.zero_()
        # sigmoid(-1000) is exactly zero in fp32, but finite BCE remains valid.
        bias.fill_(-1000.0)
    try:
        yield
    finally:
        with torch.no_grad():
            weight.copy_(saved_weight)
            bias.copy_(saved_bias)


def reference_engine_class(reference: Path) -> Any:
    name = "_minicpm_latent_host_reference"
    spec = importlib.util.spec_from_file_location(name, reference)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load host reference snapshot")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.MiniCPMLatentRolloutEngine


def make_engine(
    policy: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    name: str,
    work: dict[str, int],
) -> Any:
    from postraining.train_minicpm_vapo import (
        _stop_ids,
        resolve_thinking_end_token,
        resolve_thinking_start_token,
    )

    stops = _stop_ids(policy, tokenizer)
    common = dict(
        stop_ids=stops,
        prompts_per_rollout=args.prompts,
        samples_per_prompt=args.samples_per_prompt,
        physical_batch_size=args.physical_batch_size,
        cache_length=work["cache_length"],
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        compile_decode=args.compile,
    )
    if name == "native":
        from postraining.fast_inference import CapturedTrainingRolloutEngine

        engine = CapturedTrainingRolloutEngine(policy, **common)
        engine.stop_tensor.fill_(-1)
        if hasattr(engine, "primary_stop_tensor"):
            engine.primary_stop_tensor.fill_(-1)
    else:
        if name == "host":
            cls = reference_engine_class(args.host_reference)
        else:
            from postraining.minicpm_latent_rollout import MiniCPMLatentRolloutEngine

            cls = MiniCPMLatentRolloutEngine
        engine = cls(
            policy,
            **common,
            answer_reserve_tokens=args.answer_tokens,
            thinking_start_token_id=resolve_thinking_start_token(
                tokenizer, stop_ids=stops
            ),
            thinking_end_token_id=resolve_thinking_end_token(tokenizer, stop_ids=stops),
        )
    # Constructor validation uses genuine vocabulary ids. No EOS termination in
    # fixed-work measurements; never change logits or returned token ids.
    engine.stop_ids = (-1,)
    return engine


def timed(call: Any, *, reset_peak: bool = True) -> tuple[Any, dict[str, float | int]]:
    torch.cuda.synchronize()
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    wall = time.perf_counter()
    start.record()
    result = call()
    end.record()
    torch.cuda.synchronize()
    return result, {
        "wall_seconds": time.perf_counter() - wall,
        "cuda_event_seconds": start.elapsed_time(end) / 1000,
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def validate_generation(
    result: Any, name: str, args: argparse.Namespace, work: dict[str, int]
) -> dict[str, int]:
    from postraining.vapo.policy import (
        CONTINUE_THOUGHT,
        FIRST_THOUGHT,
        FORCED_STOP_THINKING,
        TOKEN_ACTION,
    )

    if len(result.responses) != work["logical_batch"] or any(
        row.numel() != work["stream_positions_per_row"] for row in result.responses
    ):
        raise RuntimeError(
            "fixed workload violated: wrong row count or natural early termination"
        )
    if name == "native":
        if getattr(result, "action_kinds", None) is not None:
            raise RuntimeError("native benchmark returned latent records")
        return {
            "stream_positions": work["total_stream_positions"],
            "thought_actions": 0,
            "close_actions": 0,
            "lexical_tokens": work["native_lexical_tokens"],
        }
    expected = torch.tensor(
        [FIRST_THOUGHT]
        + [CONTINUE_THOUGHT] * (args.thought_steps - 1)
        + [FORCED_STOP_THINKING]
        + [TOKEN_ACTION] * args.answer_tokens,
        dtype=torch.int8,
    )
    for kinds, raw in zip(result.action_kinds, result.latent_vectors, strict=True):
        if (
            not torch.equal(kinds.cpu(), expected)
            or raw.dtype != torch.float32
            or raw.shape[0] != args.thought_steps
            or not bool(torch.isfinite(raw).all())
        ):
            raise RuntimeError("fixed latent workload/action precision violated")
    return {
        "stream_positions": work["total_stream_positions"],
        "thought_actions": work["latent_thought_actions"],
        "close_actions": work["latent_close_actions"],
        "lexical_tokens": work["latent_lexical_tokens"],
    }


def benchmark_generation(
    policy: Any,
    tokenizer: Any,
    prompts: list[torch.Tensor],
    args: argparse.Namespace,
    name: str,
    work: dict[str, int],
) -> tuple[dict[str, Any], Any]:
    policy.eval()
    policy.latent_thinking = name != "native"
    engine = None
    try:
        with fixed_gate(policy):
            engine, construction = timed(
                lambda: make_engine(policy, tokenizer, args, name, work)
            )
            rows, warmups = [], []
            result = None
            for iteration in range(args.warmups + args.repetitions):
                torch.manual_seed(args.seed + iteration)
                result, timing = timed(
                    lambda: engine.generate_prompt_pool(
                        prompts, max_new_tokens=work["stream_positions_per_row"]
                    )
                )
                counts = validate_generation(result, name, args, work)
                timing.update(counts)
                timing.update(
                    {
                        key: getattr(result, key)
                        for key in (
                            "prefill_seconds",
                            "decode_seconds",
                            "decode_steps",
                            "capacity_row_steps",
                            "admission_events",
                        )
                    }
                )
                timing["telemetry"] = dict(getattr(engine, "telemetry", {}))
                timing["includes_graph_capture"] = (
                    bool(timing["telemetry"].get("graph_captures", 0))
                    if name == "optimized"
                    else None
                )
                timing["stream_positions_per_second"] = (
                    counts["stream_positions"] / timing["wall_seconds"]
                )
                timing["lexical_tokens_per_second"] = (
                    counts["lexical_tokens"] / timing["wall_seconds"]
                )
                (warmups if iteration < args.warmups else rows).append(timing)
            assert result is not None
            return {
                "engine": name,
                "arithmetic": engine.arithmetic,
                "timing_scope": "public_generation_lifecycle_including_recapture; warmups_amortize_kernel_compilation_only",
                "construction": construction,
                "warmups": warmups,
                "measurements": rows,
                "median_wall_seconds": statistics.median(
                    row["wall_seconds"] for row in rows
                ),
                "median_stream_positions_per_second": statistics.median(
                    row["stream_positions_per_second"] for row in rows
                ),
                "response_sha256": tensor_sha256(result.responses),
                "raw_actions_sha256": tensor_sha256(result.latent_vectors)
                if name != "native"
                else None,
            }, result
    finally:
        if engine is not None:
            engine.release_cache()
        policy.latent_thinking = True
        engine = None
        gc.collect()
        torch.cuda.empty_cache()


def natural_generation(
    policy: Any,
    tokenizer: Any,
    prompts: list[torch.Tensor],
    args: argparse.Namespace,
    name: str,
    work: dict[str, int],
) -> tuple[dict[str, Any], Any]:
    from postraining.train_minicpm_vapo import _stop_ids

    policy.eval()
    policy.latent_thinking = True
    engine = make_engine(policy, tokenizer, args, name, work)
    engine.stop_ids = _stop_ids(policy, tokenizer)
    try:
        torch.manual_seed(args.seed)
        result, timing = timed(
            lambda: engine.generate_prompt_pool(
                prompts, max_new_tokens=work["stream_positions_per_row"]
            )
        )
        return {
            "engine": name,
            "original_gate_and_eos": True,
            "prompt_lengths": [prompt.numel() for prompt in prompts],
            "timing_not_a_matched_throughput_comparison": timing,
            "response_lengths": [row.numel() for row in result.responses],
            "thought_counts": [row.shape[0] for row in result.latent_vectors],
            "raw_actions_sha256": tensor_sha256(result.latent_vectors),
            "telemetry": dict(getattr(engine, "telemetry", {})),
        }, result
    finally:
        engine.release_cache()
        engine = None
        gc.collect()
        torch.cuda.empty_cache()


def make_records(
    result: Any, prompts: list[torch.Tensor], args: argparse.Namespace, name: str
) -> list[Any]:
    from postraining.vapo.policy import (
        FORCED_STOP_THINKING,
        TrajectoryRecord,
    )

    records = []
    for index, response in enumerate(result.responses):
        prompt = prompts[index // args.samples_per_prompt]
        latent = name != "native"
        forced = (
            (result.action_kinds[index] == FORCED_STOP_THINKING).nonzero().flatten()
            if latent
            else torch.empty(0)
        )
        records.append(
            TrajectoryRecord(
                token_ids=torch.cat((prompt, response.cpu())).to(torch.int32),
                prompt_length=prompt.numel(),
                old_logprobs=result.logprobs[index].cpu().clone(),
                advantages=torch.ones(response.numel(), dtype=torch.float32),
                correct=True,
                text="",
                forced_token_index=int(forced[0]) if forced.numel() else -1,
                action_kinds=result.action_kinds[index].clone() if latent else None,
                latent_vectors=result.latent_vectors[index].clone() if latent else None,
            )
        )
    return records


def drift_summary(delta: torch.Tensor) -> dict[str, float | int] | None:
    if not delta.numel():
        return None
    values = delta.double()
    return {
        "count": values.numel(),
        "mean_delta": float(values.mean()),
        "mean_absolute_delta": float(values.abs().mean()),
        "max_absolute_delta": float(values.abs().max()),
        "rms_delta": float(values.square().mean().sqrt()),
    }


def rollout_replay_drift(
    policy: Any,
    critic: Any,
    records: list[Any],
    args: argparse.Namespace,
    *,
    fixed: bool,
) -> dict[str, Any]:
    from contextlib import nullcontext
    from postraining.vapo.policy import (
        CONTINUE_THOUGHT,
        FIRST_THOUGHT,
        FORCED_STOP_THINKING,
        STOP_THINKING,
        TOKEN_ACTION,
    )

    with fixed_gate(policy) if fixed else nullcontext():
        refreshed, _ = refresh_records(policy, critic, records, args)
    deltas = torch.cat(
        [
            after.old_logprobs - before.old_logprobs
            for before, after in zip(records, refreshed, strict=True)
        ]
    )
    kinds = torch.cat([record.action_kinds for record in records])
    return {
        "same_gate_weights_as_rollout": True,
        "fixed_continue_gate": fixed,
        "delta_convention": "packed_replay_minus_original_rollout",
        "arbitrary_equality_threshold_applied": False,
        "by_action_kind": {
            name: drift_summary(deltas[kinds == code])
            for name, code in (
                ("first_gaussian", FIRST_THOUGHT),
                ("continue_gaussian_plus_gate", CONTINUE_THOUGHT),
                ("stop_gate", STOP_THINKING),
                ("forced_close", FORCED_STOP_THINKING),
                ("answer_vocabulary", TOKEN_ACTION),
            )
        },
        "all_actions": drift_summary(deltas),
        "interpretation": "Numerical drift only. Gaussian sums can amplify bf16 hidden-state differences; refreshed age-zero ratios do not establish rollout/replay arithmetic identity.",
    }


def replay_forward(
    side: Any, batch: Any, *, actor: bool, args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from postraining.train_minicpm_vapo import _replay_hidden

    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = _replay_hidden(side, batch)
        states = hidden[batch.action_batch_indices, batch.action_positions]
        output = (
            side.action_logprobs(states, batch, chunk_tokens=args.logit_chunk_tokens)
            if actor
            else side.values(states)
        )
        # Deliberately named surrogate workloads, not invented PPO advantages or
        # a full update: no optimizer, NextLat, clipping, refresh, or reward claim.
        loss = (
            -output[batch.policy_mask].mean()
            if actor
            else (output.float() - 1).square().mean()
        )
    return states, output, loss


def gradient_snapshot(side: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in side.named_parameters()
        if parameter.grad is not None
    }


def assert_close_metrics(
    actual: torch.Tensor, expected: torch.Tensor, args: argparse.Namespace
) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise RuntimeError("qualification tensor shapes differ")
    torch.testing.assert_close(
        actual, expected, atol=args.qualification_atol, rtol=args.qualification_rtol
    )
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_absolute_error": float(delta.max()),
        "rms_error": float(delta.square().mean().sqrt()),
    }


@contextmanager
def deterministic_replay_check():
    """Compare gradients deterministically without changing timed execution."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    workspace = torch.backends.cuda.cublas_workspace_size()
    try:
        torch.backends.cuda.cublas_workspace_size(32 * 1024 * 1024)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cuda.cublas_workspace_size(workspace)


@deterministic_replay_check()
def qualify_replay(
    side: Any, records: list[Any], args: argparse.Namespace, *, actor: bool
) -> dict[str, Any]:
    from postraining.vapo.model.hf import use_packed_replay_attention
    from postraining.vapo.policy import collate_replay_microbatch
    from postraining.train_minicpm_vapo import configure_replay_checkpointing

    record = records[0]
    batch = collate_replay_microbatch(
        records, [0], pad_token_id=0, device=torch.device("cuda")
    )
    if batch.latent_vectors is not None:
        torch.testing.assert_close(
            batch.latent_vectors.cpu(), record.latent_vectors, atol=0, rtol=0
        )
        batch.latent_vectors.requires_grad_(True)
    side.train()
    baseline_states = baseline_output = None
    baseline_grads = None
    reports = []
    for interval in (0, max(1, args.replay_checkpoint_interval)):
        configure_replay_checkpointing(side.causal_lm, interval)
        side.zero_grad(set_to_none=True)
        states, output, loss = replay_forward(side, batch, actor=actor, args=args)
        loss.backward()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite replay loss")
        if batch.latent_vectors is not None and batch.latent_vectors.grad is not None:
            raise RuntimeError(
                "stored raw actions received gradients; replay must detach sampled actions"
            )
        gradients = gradient_snapshot(side)
        if not gradients or any(
            not bool(torch.isfinite(value).all()) for value in gradients.values()
        ):
            raise RuntimeError("replay produced missing/nonfinite gradients")
        if batch.latent_vectors is not None:
            thought = [
                value
                for name, value in gradients.items()
                if name.startswith("thought_adapter.")
            ]
            if not thought or not any(bool(value.ne(0).any()) for value in thought):
                raise RuntimeError("thought adapter has no nonzero replay gradient")
            if actor:
                transition = [
                    value
                    for name, value in gradients.items()
                    if name.startswith("transition.")
                ]
                if not transition or not any(
                    bool(value.ne(0).any()) for value in transition
                ):
                    raise RuntimeError("transition head has no nonzero policy gradient")
        states_cpu, output_cpu = states.detach().cpu(), output.detach().cpu()
        if baseline_grads is None:
            baseline_states, baseline_output, baseline_grads = (
                states_cpu,
                output_cpu,
                gradients,
            )
        else:
            if gradients.keys() != baseline_grads.keys():
                raise RuntimeError("checkpointing changed gradient participation")
            for name in gradients:
                torch.testing.assert_close(
                    gradients[name],
                    baseline_grads[name],
                    atol=1e-5,
                    rtol=1e-3,
                    msg=lambda message: f"{name}: {message}",
                )
            reports.append(
                {
                    "deterministic_gradient_check": True,
                    "checkpoint_interval": interval,
                    "states": assert_close_metrics(states_cpu, baseline_states, args),
                    "outputs": assert_close_metrics(output_cpu, baseline_output, args),
                    "gradient_max_absolute_error": max(
                        assert_close_metrics(
                            gradients[name], baseline_grads[name], args
                        )["max_absolute_error"]
                        for name in gradients
                    ),
                    "gradient_atol": 1e-5,
                    "gradient_rtol": 1e-3,
                }
            )
        del states, output, loss, gradients
    # Independent causal state reference: full native SDPA with explicitly
    # constructed embeddings instead of the packed replay input replacement.
    side.eval()
    configure_replay_checkpointing(side.causal_lm, 0)
    use_packed_replay_attention(side.causal_lm, enabled=False)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ids = record.token_ids[:-1].long().to("cuda").unsqueeze(0)
            embeddings = side.token_embeddings(ids).clone()
            if record.latent_vectors is not None:
                raw = record.latent_vectors.to("cuda")
                embeddings[
                    0, record.prompt_length : record.prompt_length + raw.shape[0]
                ] = side.thought_embeddings(raw)
            hidden = side.causal_lm.model(
                inputs_embeds=embeddings, use_cache=False, return_dict=True
            ).last_hidden_state
            independent = hidden[0, record.prompt_length - 1 :].cpu()
        state_report = assert_close_metrics(baseline_states, independent, args)
    finally:
        use_packed_replay_attention(side.causal_lm, enabled=True)
        configure_replay_checkpointing(side.causal_lm, args.replay_checkpoint_interval)
        side.zero_grad(set_to_none=True)
        side.train()
    return {
        "passed": True,
        "record_index": 0,
        "all_stream_positions_checked": record.response_length,
        "raw_actions_exact_fp32": record.latent_vectors is not None,
        "raw_actions_sha256": tensor_sha256([record.latent_vectors])
        if record.latent_vectors is not None
        else None,
        "detached_action_and_nonzero_thought_gradients": record.latent_vectors
        is not None,
        "checkpoint_comparison": reports,
        "independent_embedding_state_reference": state_report,
        "atol": args.qualification_atol,
        "rtol": args.qualification_rtol,
    }


def benchmark_replay(
    side: Any, records: list[Any], args: argparse.Namespace, *, actor: bool
) -> dict[str, Any]:
    from postraining.vapo.policy import collate_replay_microbatch
    from postraining.train_minicpm_vapo import configure_replay_checkpointing

    configure_replay_checkpointing(side.causal_lm, args.replay_checkpoint_interval)
    side.train()
    plan = [
        list(range(start, start + args.replay_batch_size))
        for start in range(0, len(records), args.replay_batch_size)
    ]

    def replay():
        side.zero_grad(set_to_none=True)
        for indices in plan:
            batch = collate_replay_microbatch(
                records, indices, pad_token_id=0, device=torch.device("cuda")
            )
            _, _, loss = replay_forward(side, batch, actor=actor, args=args)
            (loss / len(plan)).backward()
        return loss.detach()

    warmups, measurements = [], []
    for iteration in range(args.warmups + args.repetitions):
        loss, row = timed(replay)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite measured replay loss")
        row["stream_positions_per_second"] = (
            sum(record.response_length for record in records) / row["wall_seconds"]
        )
        (warmups if iteration < args.warmups else measurements).append(row)
    side.zero_grad(set_to_none=True)
    return {
        "workload": "actor_likelihood_forward_backward"
        if actor
        else "critic_value_mse_forward_backward",
        "includes_cpu_collation_and_h2d": True,
        "optimizer_steps": 0,
        "microbatches": len(plan),
        "trajectories": len(records),
        "stream_positions": sum(record.response_length for record in records),
        "warmups": warmups,
        "measurements": measurements,
        "median_wall_seconds": statistics.median(
            row["wall_seconds"] for row in measurements
        ),
    }


def trainable_state(side: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in side.named_parameters()
        if parameter.requires_grad
    }


@torch.no_grad()
def restore_trainable_state(
    sides: Sequence[Any], states: Sequence[dict[str, torch.Tensor]]
) -> None:
    for side, state in zip(sides, states, strict=True):
        for name, parameter in side.named_parameters():
            if name in state:
                parameter.copy_(state[name])
        side.zero_grad(set_to_none=True)


def training_options(args: argparse.Namespace, records: list[Any]) -> dict[str, Any]:
    return {
        "optimizer_minibatches": args.optimizer_minibatches,
        "replay_token_budget": args.replay_batch_size
        * max(record.input_length for record in records),
        "replay_max_trajectories": args.replay_batch_size,
        "logit_chunk_tokens": args.logit_chunk_tokens,
        "clip_low": 0.20,
        "clip_high": 0.28,
        "value_coefficient": 1.0,
        "nextlat_horizon": args.nextlat_horizon,
        "nextlat_samples": args.nextlat_samples,
        "nextlat_mse_coefficient": 1.0,
        "nextlat_kl_coefficient": 1.0,
        "nextlat_kl_chunk_tokens": args.nextlat_kl_chunk_tokens,
        "train_nextlat": True,
        "grad_clip_norm": 1.0,
    }


def make_optimizers(policy: Any, critic: Any) -> tuple[Any, Any]:
    actor = torch.optim.AdamW(
        [
            {"params": list(policy.actor_parameters())},
            {
                "params": list(policy.nextlat_head.parameters()),
                "weight_decay": 0.1,
                "betas": (0.9, 0.95),
            },
        ],
        lr=1e-6,
        weight_decay=0.0,
        fused=True,
    )
    value = torch.optim.AdamW(
        [
            {
                "params": list(critic.backbone_parameters())
                + list(critic.value_head.parameters())
            },
            {
                "params": list(critic.nextlat_head.parameters()),
                "weight_decay": 0.1,
                "betas": (0.9, 0.95),
            },
        ],
        lr=1e-5,
        weight_decay=0.0,
        fused=True,
    )
    return actor, value


def refresh_records(
    policy: Any, critic: Any, records: list[Any], args: argparse.Namespace
) -> tuple[list[Any], dict[str, Any]]:
    from postraining.train_minicpm_vapo import refresh_behavior_statistics

    refreshed, metrics = refresh_behavior_statistics(
        policy,
        critic,
        records,
        replay_token_budget=args.replay_batch_size
        * max(record.input_length for record in records),
        replay_max_trajectories=args.replay_batch_size,
        logit_chunk_tokens=args.logit_chunk_tokens,
    )
    for before, after in zip(records, refreshed, strict=True):
        if before.latent_vectors is not None:
            torch.testing.assert_close(
                after.latent_vectors, before.latent_vectors, atol=0, rtol=0
            )
        if not bool(torch.isfinite(after.old_logprobs).all()):
            raise RuntimeError("refresh produced nonfinite action likelihoods")
        if (
            before.forced_token_index >= 0
            and float(after.old_logprobs[before.forced_token_index]) != 0
        ):
            raise RuntimeError("refresh changed forced-close policy mask")
    return refreshed, metrics


def warmup_value(
    policy: Any, critic: Any, records: list[Any], args: argparse.Namespace
) -> dict[str, Any]:
    from postraining.train_minicpm_vapo import (
        configure_replay_checkpointing,
        update_step,
    )

    for side in (policy, critic):
        configure_replay_checkpointing(side.causal_lm, args.replay_checkpoint_interval)
    actor_before = trainable_state(policy)
    critic_before = trainable_state(critic)
    actor_optimizer, critic_optimizer = make_optimizers(policy, critic)
    (refreshed, refresh_metrics), refresh_timing = timed(
        lambda: refresh_records(policy, critic, records, args)
    )
    metrics, timing = timed(
        lambda: update_step(
            policy,
            critic,
            refreshed,
            actor_optimizer,
            critic_optimizer,
            **training_options(args, records),
            value_only=True,
        )
    )
    actor_after, critic_after = trainable_state(policy), trainable_state(critic)
    if any(
        not torch.equal(actor_before[name], actor_after[name]) for name in actor_before
    ):
        raise RuntimeError("value-only warmup changed actor parameters")
    if not any(
        not torch.equal(critic_before[name], critic_after[name])
        for name in critic_before
    ):
        raise RuntimeError("value-only warmup did not update critic parameters")
    if any(not bool(torch.isfinite(value).all()) for value in critic_after.values()):
        raise RuntimeError("value-only warmup produced nonfinite critic parameters")
    policy.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    return {
        "passed": True,
        "actor_unchanged": True,
        "critic_changed": True,
        "refresh_metrics": refresh_metrics,
        "refresh_timing": refresh_timing,
        "update_metrics": metrics,
        "update_timing_including_lazy_compilation": timing,
        "train_nextlat": True,
        "benchmark_return_labels": "all +1, not evaluated math rewards",
    }


def qualify_update(
    policy: Any, critic: Any, records: list[Any], args: argparse.Namespace
) -> dict[str, Any]:
    from postraining.train_minicpm_vapo import update_step

    # Restore every trainable parameter after this destructive qualification so
    # native and latent component arms see exactly the same starting weights.
    saved = [trainable_state(side) for side in (policy, critic)]
    try:
        warmup = warmup_value(policy, critic, records, args)
        actor_optimizer, critic_optimizer = make_optimizers(policy, critic)
        (refreshed, refresh_metrics), refresh_timing = timed(
            lambda: refresh_records(policy, critic, records, args)
        )
        before_actor = trainable_state(policy)
        metrics, timing = timed(
            lambda: update_step(
                policy,
                critic,
                refreshed,
                actor_optimizer,
                critic_optimizer,
                **training_options(args, records),
            )
        )
        after_actor = trainable_state(policy)
        if not any(
            not torch.equal(before_actor[name], after_actor[name])
            for name in before_actor
        ):
            raise RuntimeError("actor update did not change actor parameters")
        for side in (policy, critic):
            if any(
                not bool(torch.isfinite(parameter).all())
                for parameter in side.parameters()
                if parameter.requires_grad
            ):
                raise RuntimeError("integrated update produced nonfinite parameters")
        return {
            "passed": True,
            "value_only": warmup,
            "refresh_metrics": refresh_metrics,
            "refresh_timing": refresh_timing,
            "actor_critic_nextlat_update_metrics": metrics,
            "update_timing_including_lazy_compilation": timing,
            "actor_changed": True,
            "parameters_restored_after_qualification": True,
            "steady_state_full_cycle_throughput": None,
        }
    finally:
        restore_trainable_state((policy, critic), saved)


def benchmark_core_cycle(
    policy: Any,
    critic: Any,
    tokenizer: Any,
    prompts: list[torch.Tensor],
    args: argparse.Namespace,
    name: str,
    work: dict[str, int],
) -> dict[str, Any]:
    """Warm optimizer-bearing rollout/refresh/update cycles at fixed stream work."""
    from postraining.train_minicpm_vapo import update_step

    saved = [trainable_state(side) for side in (policy, critic)]
    engine = None
    try:
        policy.latent_thinking = critic.latent_thinking = name != "native"
        actor_optimizer, critic_optimizer = make_optimizers(policy, critic)
        engine, construction = timed(
            lambda: make_engine(policy, tokenizer, args, name, work)
        )

        def cycle():
            policy.eval()
            with fixed_gate(policy):
                generation, rollout_time = timed(
                    lambda: engine.generate_prompt_pool(
                        prompts,
                        max_new_tokens=work["stream_positions_per_row"],
                    ),
                    reset_peak=False,
                )
            counts = validate_generation(generation, name, args, work)
            records = make_records(generation, prompts, args, name)
            _, release_time = timed(engine.release_cache, reset_peak=False)
            (refreshed, refresh_metrics), refresh_time = timed(
                lambda: refresh_records(policy, critic, records, args),
                reset_peak=False,
            )
            metrics, update_time = timed(
                lambda: update_step(
                    policy,
                    critic,
                    refreshed,
                    actor_optimizer,
                    critic_optimizer,
                    **training_options(args, records),
                ),
                reset_peak=False,
            )
            return {
                "counts": counts,
                "rollout": rollout_time,
                "cache_release": release_time,
                "behavior_refresh": refresh_time,
                "update": update_time,
                "refresh_metrics": refresh_metrics,
                "update_metrics": metrics,
                "runtime_telemetry": dict(getattr(engine, "telemetry", {})),
            }

        warmups, measurements = [], []
        for iteration in range(args.warmups + args.repetitions):
            torch.manual_seed(args.seed + iteration)
            phases, total = timed(cycle)
            row = {
                **total,
                **phases,
                "stream_positions_per_second": work["total_stream_positions"]
                / total["wall_seconds"],
            }
            (warmups if iteration < args.warmups else measurements).append(row)
        if any(
            not bool(torch.isfinite(parameter).all())
            for side in (policy, critic)
            for parameter in side.parameters()
            if parameter.requires_grad
        ):
            raise RuntimeError("measured cycle produced nonfinite parameters")
        return {
            "engine": name,
            "scope": "rollout, record construction, cache release, behavior refresh, actor/critic/NextLat optimizer updates",
            "excluded": [
                "dataset loading",
                "math answer verification",
                "telemetry writer",
                "checkpoint I/O",
            ],
            "benchmark_return_labels": "all +1, not measured math rewards",
            "starting_trainable_weights_shared_across_arms": True,
            "weights_evolve_during_warmup_and_measurement": True,
            "optimizer_minibatches": args.optimizer_minibatches,
            "construction": construction,
            "warmups": warmups,
            "measurements": measurements,
            "median_wall_seconds": statistics.median(
                row["wall_seconds"] for row in measurements
            ),
            "median_stream_positions_per_second": statistics.median(
                row["stream_positions_per_second"] for row in measurements
            ),
        }
    finally:
        if engine is not None:
            engine.release_cache()
        restore_trainable_state((policy, critic), saved)
        policy.latent_thinking = critic.latent_thinking = True
        engine = None
        gc.collect()
        torch.cuda.empty_cache()


def save_report(args: argparse.Namespace, report: dict[str, Any]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work = validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA is required; no CPU model fallback")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    sources = fingerprints(args)
    if args.expected_sources is not None:
        expected_sources = json.loads(args.expected_sources.read_text())
        if sources != expected_sources:
            raise RuntimeError("benchmark sources changed while queued")
    report: dict[str, Any] = {
        "schema": "minicpm_latent_matched_benchmark/v1",
        "status": "running",
        "qualification_scope": "execution and replay integrity, not training validity",
        "rollout_replay_distribution_equivalence_verified": False,
        "execution_failures": [],
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "sources": sources,
        "checkpoint_sha256": file_sha256(args.checkpoint)
        if args.checkpoint is not None
        else None,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "precision": "bf16 trunk/autocast, fp32 master adapters and raw Gaussian actions",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "workload": work,
        "inference": [],
        "replay": [],
        "natural_qualification": [],
        "limitations": [
            "Equal prompt/context/stream positions and logical/physical batches, not equal lexical work or equal policy outputs.",
            "Fixed workload temporarily sets both latent gate weights to zero and bias to -1000 (exact fp32 continue), then forces close. Original checkpoint gate is restored for replay.",
            "EOS is disabled for fixed-work timing. Original-gate/EOS natural rollouts are separate qualification only, never equal-work throughput evidence.",
            "Synthetic context uses repeated background text left-truncated to exact context length while preserving native thinking suffix.",
            "Native retains every loaded latent head but explicitly disables latent policy semantics and generates independent token records.",
            "Component replay uses real generated records and shared weights but likelihood/value surrogate losses. Integrated qualification additionally exercises real refresh, value-only and actor/critic+NextLat update_step with fixed benchmark return labels, not measured math rewards.",
            "Host and optimized stochastic trajectories need not match: chunked RNG scheduling differs. Fixed counts are asserted for each measurement.",
            "CUDA events bracket host scheduling too; event elapsed is not a kernel-only profiler metric. Wall time includes synchronization, prefix transfer and policy refresh.",
            "Missing telemetry is unavailable, never reported as zero. Head work differs inherently between latent and native modes.",
            "Graph captures are invalidated at each public optimized generation for cache/source residency correctness; capture cost remains in measurements. Warmups exclude initial model loading and amortize kernel compilation, not per-call graph capture.",
            "Integrated training qualification is not a warmed full-cycle throughput result; its phase timings may include lazy compilation.",
            "Warmed core-cycle measurements additionally include rollout, record construction, cache release, behavior refresh and real optimizer updates; exclude math grading, checkpoint I/O and telemetry writers. Trainable parameters evolve during each arm and reset between arms.",
        ],
    }
    save_report(args, report)
    try:
        (policy, tokenizer, payload), startup = timed(lambda: load_policy(args))
        report["startup"] = startup
        report["initial_actor_trainable_sha256"] = tensor_sha256(
            list(trainable_state(policy).values())
        )
        report["base_model"] = {
            key: payload["actor"][key] for key in ("model_id", "revision")
        }
        prompts = make_prompts(tokenizer, args)
        report["prompt_ids_sha256"] = tensor_sha256(prompts)
        records_by_mode, natural_records = {}, {}
        for name in args.engines:
            metrics, result = benchmark_generation(
                policy, tokenizer, prompts, args, name, work
            )
            report["inference"].append(metrics)
            records_by_mode[name] = make_records(result, prompts, args, name)
            if (
                tensor_sha256(list(trainable_state(policy).values()))
                != report["initial_actor_trainable_sha256"]
            ):
                raise RuntimeError(
                    "generation changed source actor weights between benchmark arms"
                )
            if name != "native" and args.mode in ("all", "qualify"):
                # Exercise unequal prefix lengths separately from fixed-work timing.
                natural_prompts = [
                    prompt if index % 2 == 0 else prompt[1:]
                    for index, prompt in enumerate(prompts)
                ]
                natural, natural_result = natural_generation(
                    policy, tokenizer, natural_prompts, args, name, work
                )
                report["natural_qualification"].append(natural)
                natural_records[name] = make_records(
                    natural_result, natural_prompts, args, name
                )
            save_report(args, report)
        records_path = args.output.with_suffix(".records.pt")
        torch.save(
            {"fixed": records_by_mode, "natural": natural_records, "sources": sources},
            records_path,
        )
        report["exact_fp32_replay_records"] = {
            "path": str(records_path),
            "sha256": file_sha256(records_path),
        }
        if args.mode != "inference":
            from postraining.vapo.model.hf import (
                enable_packed_replay_attention,
                enable_replay_mlp_compilation,
            )

            critic, report["critic_startup"] = timed(
                lambda: load_critic(payload, policy, args)
            )
            report["initial_critic_trainable_sha256"] = tensor_sha256(
                list(trainable_state(critic).values())
            )
            for side in (policy, critic):
                enable_packed_replay_attention(side.causal_lm)
                if args.compile_replay:
                    enable_replay_mlp_compilation(side.causal_lm)
            report["rollout_replay_drift"] = {
                name: rollout_replay_drift(policy, critic, records, args, fixed=True)
                for name, records in records_by_mode.items()
                if name != "native"
            }
            for natural in report["natural_qualification"]:
                natural["rollout_replay_drift"] = rollout_replay_drift(
                    policy,
                    critic,
                    natural_records[natural["engine"]],
                    args,
                    fixed=False,
                )
            # One canonical latent replay corpus avoids presenting different RNG
            # trajectories as the same replay work across runtime implementations.
            latent_name = "optimized" if "optimized" in records_by_mode else "host"
            replay_modes = ([latent_name] if latent_name in records_by_mode else []) + (
                ["native"] if "native" in records_by_mode else []
            )
            # One shared real value-only warmup opens the zero-initialized value
            # head's gradient path. All component arms then share these weights.
            warmup_name = replay_modes[0]
            policy.latent_thinking = critic.latent_thinking = warmup_name != "native"
            report["shared_value_warmup"] = warmup_value(
                policy, critic, records_by_mode[warmup_name], args
            )
            report["component_critic_trainable_sha256"] = tensor_sha256(
                list(trainable_state(critic).values())
            )
            for name in replay_modes:
                records = records_by_mode[name]
                policy.latent_thinking = critic.latent_thinking = name != "native"
                row: dict[str, Any] = {
                    "records_from_engine": name,
                    "record_tokens_sha256": tensor_sha256(
                        [record.token_ids for record in records]
                    ),
                }
                report["replay"].append(row)
                for actor, side in ((True, policy), (False, critic)):
                    key = "actor" if actor else "critic"
                    row[key] = {
                        "qualification": qualify_replay(
                            side, records, args, actor=actor
                        )
                    }
                    if args.mode != "qualify":
                        row[key]["timing"] = benchmark_replay(
                            side, records, args, actor=actor
                        )
                try:
                    row["integrated_training_qualification"] = qualify_update(
                        policy, critic, records, args
                    )
                except Exception as error:
                    # qualify_update restores parameters and clears gradients.
                    # Keep this failure visible without losing independent arms.
                    failure = {
                        "engine": name,
                        "stage": "integrated_training_qualification",
                        "error": f"{type(error).__name__}: {error}",
                    }
                    row["integrated_training_qualification"] = {
                        "passed": False,
                        "error": failure["error"],
                    }
                    report["execution_failures"].append(failure)
                save_report(args, report)
            if args.mode == "all":
                report["core_cycles"] = []
                qualified = {
                    row["records_from_engine"]: row[
                        "integrated_training_qualification"
                    ]["passed"]
                    for row in report["replay"]
                }
                for name in args.engines:
                    qualification_name = "native" if name == "native" else latent_name
                    if not qualified[qualification_name]:
                        core = {
                            "engine": name,
                            "status": "blocked",
                            "reason": "integrated training qualification failed",
                        }
                    else:
                        try:
                            core = benchmark_core_cycle(
                                policy, critic, tokenizer, prompts, args, name, work
                            )
                            core["status"] = "passed"
                        except Exception as error:
                            core = {
                                "engine": name,
                                "status": "failed",
                                "error": f"{type(error).__name__}: {error}",
                            }
                            report["execution_failures"].append(
                                {"stage": "core_cycle", **core}
                            )
                    report["core_cycles"].append(core)
                    save_report(args, report)
            policy.latent_thinking = critic.latent_thinking = True
        if fingerprints(args) != sources or (
            args.checkpoint is not None
            and file_sha256(args.checkpoint) != report["checkpoint_sha256"]
        ):
            raise RuntimeError(
                "source or checkpoint changed during benchmark; result is not reproducible"
            )
        report["status"] = "failed" if report["execution_failures"] else "passed"
        save_report(args, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(args.output),
                    "workload": work,
                }
            ),
            flush=True,
        )
        return 1 if report["execution_failures"] else 0
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        save_report(args, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

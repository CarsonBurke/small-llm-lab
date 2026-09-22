from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

if TYPE_CHECKING:
    from postraining.vapo.policy import VAPOPolicy


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "postraining/fast_inference.py",
        "postraining/vapo/policy.py",
        "postraining/nextlat_speculative.py",
        "postraining/train_minicpm_vapo.py",
        "postraining/validate_minicpm_vapo.py",
        "scripts/benchmark_minicpm_nextlat.py",
    ):
        digest.update((REPO_ROOT / relative).read_bytes())
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark training-compatible and optimized MiniCPM decoding."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 16))
    parser.add_argument("--tokens", type=int, default=140)
    parser.add_argument("--draft-length", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=("fast", "autoregressive", "nextlat"),
        default=("fast",),
    )
    parser.add_argument(
        "--target-dtype",
        choices=("bf16", "w8a16_head", "w8a16_all"),
        default="bf16",
    )
    parser.add_argument(
        "--prompt",
        default="Prove that the square root of 2 is irrational, with complete reasoning.",
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def required_warmup_steps(checkpoint: Mapping[str, Any]) -> int:
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, Mapping):
        raise ValueError("checkpoint training arguments are missing")
    value = saved_args.get("value_warmup_steps")
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise ValueError("checkpoint NextLat warmup requirement is invalid")
    return value


def load_policy(
    checkpoint_path: Path,
    *,
    target_dtype: str,
) -> tuple[VAPOPolicy, Any, dict[str, int | str | float]]:
    from postraining.vapo.policy import VAPOPolicy
    from postraining.vapo.model.lora import (
        LoRAConfig,
        load_adapter_state_dict,
        merge_lora_for_inference,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload = checkpoint["policy"]
    if payload.get("schema") != "minicpm5_vapo_adapter/v6":
        raise ValueError("benchmark requires a v6 MiniCPM checkpoint")
    actor = payload["actor"]
    required_warmup = required_warmup_steps(checkpoint)
    warmup_step = int(checkpoint.get("warmup_step", 0))
    if warmup_step < 1 or warmup_step < required_warmup:
        raise ValueError(
            "benchmark requires a checkpoint with completed value warmup"
        )
    device = torch.device("cuda")
    policy, tokenizer = VAPOPolicy.from_family("minicpm5", 
        model_id=actor["model_id"],
        revision=actor["revision"],
        device=device,
        lora_config=LoRAConfig(**actor["lora_config"]),
        nextlat_projection_factor=float(actor["nextlat_projection_factor"]),
        gradient_checkpointing=False,
    )
    load_adapter_state_dict(policy.causal_lm, actor["adapter"])
    policy.nextlat_head.load_state_dict(actor["nextlat"], strict=True)
    merged_lora_modules = len(merge_lora_for_inference(policy.causal_lm))
    policy.eval()
    metadata = {
        "warmup_step": warmup_step,
        "required_warmup_steps": required_warmup,
        "actor_step": int(checkpoint.get("step", 0)),
        "merged_lora_modules": merged_lora_modules,
        "target_dtype": target_dtype,
        "quantized_linear_modules": 0,
        "fused_projection_groups": 0,
        "model_id": str(actor["model_id"]),
        "model_revision": str(actor["revision"]),
    }
    return policy, tokenizer, metadata


@torch.inference_mode()
def apply_target_dtype(
    policy: VAPOPolicy,
    prompt: torch.Tensor,
    target_dtype: str,
) -> dict[str, int | float | str]:
    from postraining.fast_inference import (
        fuse_llama_projections_,
        quantize_linear_layers_w8a16_,
        quantize_lm_head_w8a16_,
    )

    device = next(policy.parameters()).device
    input_ids = prompt.to(device)[None]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference_hidden = policy.replay_hidden(input_ids, None)
        reference_logits = policy.logits(reference_hidden).float()

    def comparison(
        prefix: str,
        candidate_logits: torch.Tensor,
    ) -> dict[str, int | float]:
        reference = reference_logits.flatten(0, -2)
        candidate = candidate_logits.flatten(0, -2)
        reference_logprobs = reference.log_softmax(dim=-1)
        candidate_logprobs = candidate.log_softmax(dim=-1)
        reference_top = reference.topk(20, dim=-1).indices
        candidate_top = candidate.topk(20, dim=-1).indices
        overlap = (
            (
                reference_top[:, :, None]
                == candidate_top[:, None, :]
            )
            .any(dim=-1)
            .float()
            .mean()
        )
        return {
            f"{prefix}_logit_cosine": float(
                torch.cosine_similarity(reference, candidate).mean()
            ),
            f"{prefix}_top1_match_rate": float(
                (
                    reference.argmax(dim=-1)
                    == candidate.argmax(dim=-1)
                )
                .float()
                .mean()
            ),
            f"{prefix}_top20_overlap": float(overlap),
            f"{prefix}_categorical_kl": float(
                (
                    reference_logprobs.exp()
                    * (reference_logprobs - candidate_logprobs)
                )
                .sum(dim=-1)
                .mean()
            ),
        }

    fused_projection_groups = len(
        fuse_llama_projections_(policy.causal_lm)
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        fused_hidden = policy.replay_hidden(input_ids, None)
        fused_logits = policy.logits(fused_hidden).float()
    fusion_metrics = comparison("fusion", fused_logits)
    fusion_quality_passed = (
        fusion_metrics["fusion_logit_cosine"] >= 0.999
        and fusion_metrics["fusion_top1_match_rate"] >= 0.90
        and fusion_metrics["fusion_top20_overlap"] >= 0.95
        and fusion_metrics["fusion_categorical_kl"] <= 0.01
    )
    metrics: dict[str, int | float | str] = {
        "fused_projection_groups": fused_projection_groups,
        "quantized_linear_modules": 0,
        "fusion_quality_gate_passed": int(fusion_quality_passed),
        **fusion_metrics,
    }
    if target_dtype == "bf16":
        metrics["quality_status"] = (
            "bf16_fusion_passed_prompt_gate"
            if fusion_quality_passed
            else "bf16_fusion_failed_prompt_gate"
        )
        return metrics

    if target_dtype == "w8a16_head":
        quantize_lm_head_w8a16_(policy.causal_lm)
        metrics["quantized_linear_modules"] = 1
    else:
        metrics["quantized_linear_modules"] = len(
            quantize_linear_layers_w8a16_(policy.causal_lm)
        )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        candidate_hidden = policy.replay_hidden(input_ids, None)
        candidate_logits = policy.logits(candidate_hidden).float()
    w8a16_metrics = comparison("w8a16", candidate_logits)
    metrics.update(w8a16_metrics)
    quality_passed = (
        fusion_quality_passed
        and w8a16_metrics["w8a16_logit_cosine"] >= 0.995
        and w8a16_metrics["w8a16_top1_match_rate"] >= 0.90
        and w8a16_metrics["w8a16_top20_overlap"] >= 0.90
        and w8a16_metrics["w8a16_categorical_kl"] <= 0.05
    )
    metrics["quality_gate_passed"] = int(quality_passed)
    metrics["quality_status"] = (
        "experimental_passed_prompt_gate"
        if quality_passed
        else "experimental_failed_prompt_gate"
    )
    return metrics


def force_fixed_length(engine: Any, device: torch.device) -> None:
    engine.stop_ids = (-1,)
    engine.primary_stop = -1
    engine.stop_tensor = torch.tensor((-1,), device=device)
    if hasattr(engine, "primary_stop_tensor"):
        engine.primary_stop_tensor = torch.tensor(-1, device=device)


def timed_generation(
    engine: Any,
    prompt: torch.Tensor,
    tokens: int,
    *,
    fast: bool,
) -> tuple[Any, float, int]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = (
        engine.generate(prompt, max_new_tokens=tokens)
        if fast
        else engine.generate_prompts([prompt], max_new_tokens=tokens)
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = int(result[0].numel())
    return result, elapsed, generated


def benchmark_engine(
    policy: VAPOPolicy,
    tokenizer: Any,
    prompt: torch.Tensor,
    *,
    batch_size: int,
    engine_name: str,
    measured_tokens: int,
    draft_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    compile_decode: bool,
) -> dict[str, float | int | str]:
    device = next(policy.parameters()).device
    cache_length = int(prompt.numel()) + measured_tokens + draft_length
    if engine_name == "nextlat":
        from postraining.nextlat_speculative import NextLatSpeculativeEngine

        engine: Any = NextLatSpeculativeEngine(
            policy,
            stop_ids=(-1,),
            prompts_per_rollout=1,
            samples_per_prompt=batch_size,
            cache_length=cache_length,
            draft_length=draft_length,
            temperature=temperature,
            top_p=top_p,
            compile_decode=compile_decode,
        )
        name = "nextlat"
    elif engine_name == "autoregressive":
        from postraining.train_minicpm_vapo import RolloutEngine

        engine = RolloutEngine(
            policy,
            tokenizer,
            prompts_per_rollout=1,
            samples_per_prompt=batch_size,
            cache_length=cache_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            compile_decode=compile_decode,
        )
        force_fixed_length(engine, device)
        name = "autoregressive"
    else:
        from postraining.fast_inference import FixedLengthInferenceEngine

        engine = FixedLengthInferenceEngine(
            policy,
            batch_size=batch_size,
            cache_length=cache_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            compile_decode=compile_decode,
        )
        name = "autoregressive_fast_topk"

    seed = 17_291
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    timed_generation(
        engine, prompt, measured_tokens, fast=engine_name == "fast"
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    result, elapsed, generated = timed_generation(
        engine,
        prompt,
        measured_tokens,
        fast=engine_name == "fast",
    )
    stats = result[-1]
    metrics: dict[str, float | int | str] = {
        "engine": name,
        "decode_schedule": str(
            getattr(engine, "decode_schedule", "engine_native")
        ),
        "batch_size": batch_size,
        "seed": seed,
        "generated_tokens": generated,
        "generated_token_ids_sha256": hashlib.sha256(
            json.dumps(
                result[0].detach().cpu().tolist(),
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "elapsed_seconds": elapsed,
        "aggregate_tokens_per_second": generated / elapsed,
        "tokens_per_second_per_stream": generated / elapsed / batch_size,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "target_decode_calls": int(stats.target_decode_calls),
        "target_decode_positions": int(
            getattr(stats, "target_decode_positions", stats.target_decode_calls)
        ),
        "target_positions_per_call": float(
            getattr(stats, "target_decode_positions", stats.target_decode_calls)
        )
        / max(stats.target_decode_calls, 1),
        "prefill_seconds": float(stats.prefill_seconds),
        "decode_seconds": float(stats.decode_seconds),
        "decode_tokens_per_second": generated
        / max(float(stats.decode_seconds), 1e-9),
    }
    proposed = int(getattr(stats, "proposed_tokens", 0))
    accepted = int(getattr(stats, "accepted_tokens", 0))
    proposed_pos2 = int(
        getattr(stats, "proposed_tokens_pos2plus", 0)
    )
    accepted_pos2 = int(
        getattr(stats, "accepted_tokens_pos2plus", 0)
    )
    row_cycles = int(getattr(stats, "speculative_row_cycles", 0))
    metrics["proposed_tokens"] = proposed
    metrics["accepted_tokens"] = accepted
    if proposed:
        metrics["acceptance"] = accepted / proposed
    if proposed_pos2:
        metrics["acceptance_pos2plus"] = accepted_pos2 / proposed_pos2
    if row_cycles:
        metrics["accepted_pos2plus_per_row_cycle"] = (
            accepted_pos2 / row_cycles
        )
    for position, (position_proposed, position_accepted) in enumerate(
        zip(
            getattr(stats, "proposed_by_position", ()),
            getattr(stats, "accepted_by_position", ()),
            strict=True,
        ),
        start=1,
    ):
        if position_proposed:
            metrics[f"acceptance_position_{position}"] = (
                position_accepted / position_proposed
            )
    if hasattr(engine, "release_cache"):
        engine.release_cache()
    del engine
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    if args.tokens < 1 or args.draft_length < 2:
        raise ValueError(
            "token count must be positive and draft length at least two"
        )
    if "fast" in args.engines and len(args.engines) != 1:
        raise ValueError(
            "fast top-k throughput must be benchmarked separately from "
            "full-vocabulary engines"
        )
    if args.target_dtype.startswith("w8a16") and args.batch_sizes != [1]:
        raise ValueError("packed W8A16 kernels are optimized only for B1")
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    from postraining.validate_minicpm_vapo import prompt_ids

    policy, tokenizer, checkpoint_metadata = load_policy(
        args.checkpoint, target_dtype=args.target_dtype
    )
    prompt = prompt_ids(tokenizer, args.prompt, torch.device("cpu")).squeeze(0)
    checkpoint_metadata.update(
        apply_target_dtype(policy, prompt, args.target_dtype)
    )
    checkpoint_metadata.update(
        {
            "benchmark_source_sha256": source_sha256(),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "gpu": torch.cuda.get_device_name(),
            "pytorch_version": torch.__version__,
            "transformers_version": version("transformers"),
            "triton_version": version("triton"),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
    )
    print(
        json.dumps(
            {"type": "checkpoint", **checkpoint_metadata},
            sort_keys=True,
        ),
        flush=True,
    )
    results: list[dict[str, float | int | str]] = []
    for batch_size in args.batch_sizes:
        for engine_name in args.engines:
            metrics = benchmark_engine(
                policy,
                tokenizer,
                prompt,
                batch_size=batch_size,
                engine_name=engine_name,
                measured_tokens=args.tokens,
                draft_length=args.draft_length,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                compile_decode=args.compile,
            )
            results.append(metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)
    document = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "prompt_tokens": int(prompt.numel()),
        "warmup_tokens_per_stream": args.tokens,
        "measured_tokens_per_stream": args.tokens,
        "prompt": args.prompt,
        "prompt_token_ids_sha256": hashlib.sha256(
            json.dumps(
                prompt.tolist(), separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "seed": 17_291,
        "draft_length": args.draft_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "engines": args.engines,
        "target_dtype": args.target_dtype,
        "compiled": args.compile,
        "measurement_repetitions": 1,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark equivalent LoRA policy updates for MiniCPM5 and Qwen3.5.

GPU execution belongs behind ``mlq``.  The benchmark measures the common actor
update used by the MiniCPM VAPO path: frozen bf16 backbone, fp32-master rank-16
LoRA, non-reentrant gradient checkpointing, exact selected-token log
probabilities through a frozen output head, and fused AdamW.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from postraining.hf_runtime import prepare_text_only_transformers_runtime
from postraining.vapo.model.hf import MINICPM5_SPEC
from postraining.vapo.model.lora import (
    LoRAConfig,
    inject_lora,
)
from postraining.vapo.model.readout import chunked_frozen_head_logprobs


QWEN35_MODEL_ID = "Qwen/Qwen3.5-0.8B"
QWEN35_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MINICPM_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
QWEN35_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("minicpm", "qwen35"), required=True)
    parser.add_argument(
        "--sequence-lengths",
        default="1024,2048,4096,5120,8192",
        help="Comma-separated packed sequence lengths.",
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--logit-chunk-tokens", type=int, default=128)
    parser.add_argument("--max-action-tokens", type=int, default=4_096)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.sequence_lengths = tuple(int(value) for value in args.sequence_lengths.split(","))
    if any(length < 2 for length in args.sequence_lengths):
        parser.error("all sequence lengths must be at least 2")
    if args.warmup_steps < 1 or args.measure_steps < 1:
        parser.error("warmup and measurement counts must be positive")
    if args.max_action_tokens < 1:
        parser.error("max action tokens must be positive")
    return args


def tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def unique_parameters(module: nn.Module) -> list[nn.Parameter]:
    seen: set[int] = set()
    parameters: list[nn.Parameter] = []
    for parameter in module.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            parameters.append(parameter)
    return parameters


def nested_tensor_bytes(value: Any) -> int:
    if isinstance(value, Tensor):
        return tensor_bytes(value)
    if isinstance(value, dict):
        return sum(nested_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(nested_tensor_bytes(item) for item in value)
    return 0


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(nested_tensor_bytes(state) for state in optimizer.state.values())


def mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def load_model(model_key: str, rank: int, alpha: float) -> tuple[Any, dict[str, Any]]:
    prepare_text_only_transformers_runtime()
    model: Any
    if model_key == "minicpm":
        from transformers import AutoModelForCausalLM

        model_id = MINICPM5_SPEC.model_id
        revision = MINICPM5_SPEC.revision
        targets = MINICPM_TARGETS
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    else:
        from transformers import Qwen3_5ForCausalLM

        model_id = QWEN35_MODEL_ID
        revision = QWEN35_REVISION
        targets = QWEN35_TARGETS
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )

    model.config.use_cache = False
    replacements = inject_lora(model, LoRAConfig(rank=rank, alpha=alpha, targets=targets))
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.to(torch.device("cuda"))
    model.train()
    metadata = {
        "model_id": model_id,
        "revision": revision,
        "model_class": type(model).__name__,
        "lora_targets": targets,
        "lora_module_count": len(replacements),
        "hidden_size": int(model.config.hidden_size),
        "vocab_size": int(model.config.vocab_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
    }
    return model, metadata


def train_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    *,
    logit_chunk_tokens: int,
    action_tokens: int,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = model.model(
            input_ids=input_ids,
            attention_mask=None,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        action_hidden = hidden[:, -action_tokens - 1 : -1].reshape(
            -1, hidden.shape[-1]
        )
        targets = input_ids[:, -action_tokens:].reshape(-1)
        head = model.get_output_embeddings().weight
        if head.requires_grad:
            raise RuntimeError("output head must remain frozen")
        logprobs = chunked_frozen_head_logprobs(
            action_hidden,
            targets,
            head,
            chunk_tokens=logit_chunk_tokens,
        )
        loss = -logprobs.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def benchmark_length(
    model: Any,
    optimizer: torch.optim.Optimizer,
    *,
    sequence_length: int,
    warmup_steps: int,
    measure_steps: int,
    logit_chunk_tokens: int,
    max_action_tokens: int,
    seed: int,
) -> dict[str, Any]:
    action_tokens = min(max_action_tokens, sequence_length - 1)
    generator = torch.Generator(device="cuda").manual_seed(seed + sequence_length)
    input_ids = torch.randint(
        0,
        int(model.config.vocab_size),
        (1, sequence_length),
        generator=generator,
        device="cuda",
    )
    losses: list[float] = []
    for _ in range(warmup_steps):
        losses.append(
            train_step(
                model,
                optimizer,
                input_ids,
                logit_chunk_tokens=logit_chunk_tokens,
                action_tokens=action_tokens,
            )
        )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    steady_allocated = torch.cuda.memory_allocated()
    steady_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    seconds: list[float] = []
    for _ in range(measure_steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        losses.append(
            train_step(
                model,
                optimizer,
                input_ids,
                logit_chunk_tokens=logit_chunk_tokens,
                action_tokens=action_tokens,
            )
        )
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    final_allocated = torch.cuda.memory_allocated()
    final_reserved = torch.cuda.memory_reserved()
    median_seconds = statistics.median(seconds)
    return {
        "sequence_length": sequence_length,
        "batch_size": 1,
        "action_tokens": action_tokens,
        "median_step_seconds": median_seconds,
        "step_seconds": seconds,
        "input_tokens_per_second": sequence_length / median_seconds,
        "action_tokens_per_second": action_tokens / median_seconds,
        "loss_last": losses[-1],
        "steady_allocated_bytes": steady_allocated,
        "steady_allocated_mib": mib(steady_allocated),
        "steady_reserved_bytes": steady_reserved,
        "steady_reserved_mib": mib(steady_reserved),
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_mib": mib(peak_allocated),
        "peak_working_bytes": peak_allocated - steady_allocated,
        "peak_working_mib": mib(peak_allocated - steady_allocated),
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_mib": mib(peak_reserved),
        "final_allocated_bytes": final_allocated,
        "final_allocated_mib": mib(final_allocated),
        "final_reserved_bytes": final_reserved,
        "final_reserved_mib": mib(final_reserved),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision("high")

    torch.cuda.empty_cache()
    model, metadata = load_model(args.model, args.rank, args.alpha)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    parameters = unique_parameters(model)
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    frozen = [parameter for parameter in parameters if not parameter.requires_grad]
    buffer_bytes = sum(tensor_bytes(buffer) for buffer in model.buffers())
    model_allocated = torch.cuda.memory_allocated()
    model_reserved = torch.cuda.memory_reserved()
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=0.0,
        fused=True,
    )

    measurements: list[dict[str, Any]] = []
    for sequence_length in args.sequence_lengths:
        try:
            measurement = benchmark_length(
                model,
                optimizer,
                sequence_length=sequence_length,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                logit_chunk_tokens=args.logit_chunk_tokens,
                max_action_tokens=args.max_action_tokens,
                seed=1234,
            )
            measurement["optimizer_state_bytes"] = optimizer_state_bytes(optimizer)
            measurement["optimizer_state_mib"] = mib(measurement["optimizer_state_bytes"])
            measurements.append(measurement)
            print(
                f"{args.model} L={sequence_length}: "
                f"{measurement['median_step_seconds']:.3f}s, "
                f"{measurement['input_tokens_per_second']:.1f} input tok/s, "
                f"{measurement['peak_allocated_mib']:.1f} MiB peak",
                flush=True,
            )
        except torch.OutOfMemoryError as error:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            measurements.append(
                {
                    "sequence_length": sequence_length,
                    "batch_size": 1,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            break

    output = {
        "schema_version": 1,
        "benchmark": "equivalent_lora_actor_update",
        "model_key": args.model,
        "model": metadata,
        "methodology": {
            "base_dtype": "bfloat16",
            "adapter_dtype": "float32",
            "lora_rank": args.rank,
            "lora_alpha": args.alpha,
            "gradient_checkpointing": "non_reentrant",
            "attention_implementation": "sdpa",
            "optimizer": "torch.optim.AdamW(fused=True)",
            "learning_rate": args.learning_rate,
            "logit_loss": "exact selected-token log-probability for response tokens",
            "logit_chunk_tokens": args.logit_chunk_tokens,
            "max_action_tokens": args.max_action_tokens,
            "warmup_steps_per_length": args.warmup_steps,
            "measure_steps_per_length": args.measure_steps,
            "seed": 1234,
        },
        "parameters": {
            "total": sum(parameter.numel() for parameter in parameters),
            "frozen": sum(parameter.numel() for parameter in frozen),
            "trainable": sum(parameter.numel() for parameter in trainable),
            "frozen_bytes": sum(tensor_bytes(parameter) for parameter in frozen),
            "frozen_mib": mib(sum(tensor_bytes(parameter) for parameter in frozen)),
            "trainable_bytes": sum(tensor_bytes(parameter) for parameter in trainable),
            "trainable_mib": mib(sum(tensor_bytes(parameter) for parameter in trainable)),
            "buffer_bytes": buffer_bytes,
            "buffer_mib": mib(buffer_bytes),
        },
        "memory_after_model_load": {
            "allocated_bytes": model_allocated,
            "allocated_mib": mib(model_allocated),
            "reserved_bytes": model_reserved,
            "reserved_mib": mib(model_reserved),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "fla_installed": importlib.util.find_spec("fla") is not None,
            "flash_qla_installed": importlib.util.find_spec("flash_qla") is not None,
            "causal_conv1d_installed": importlib.util.find_spec("causal_conv1d") is not None,
        },
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

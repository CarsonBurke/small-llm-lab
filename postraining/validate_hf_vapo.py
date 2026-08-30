"""Validate zero-initialized MiniCPM VAPO adapters against native HF behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from postraining.hf_runtime import prepare_text_only_transformers_runtime
from postraining.hf_vapo import (
    MINICPM5_MODEL_ID,
    MINICPM5_REVISION,
    LoRAConfig,
    MiniCPMVAPOPolicy,
    StaticCachePool,
    chunked_frozen_head_logprobs,
)


PROMPTS = (
    "Compute 37 * 46 and give the final answer.",
    "If x + 3 = 11, find x. Explain the reasoning briefly.",
)


def prompt_ids(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
    )
    if not isinstance(encoded, torch.Tensor):
        encoded = encoded["input_ids"]
    return encoded.to(device)


@torch.inference_mode()
def greedy_static_generation(
    policy: MiniCPMVAPOPolicy,
    inputs: torch.Tensor,
    *,
    max_new_tokens: int,
    cache_pool: StaticCachePool,
) -> torch.Tensor:
    device = inputs.device
    cache = cache_pool.acquire(inputs.shape[0])
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = policy.cached_hidden(
            inputs,
            past_key_values=cache,
            cache_position=torch.arange(inputs.shape[1], device=device),
        )[:, -1]
        logits = policy.logits(hidden)
        generated = []
        for offset in range(max_new_tokens):
            token = logits.argmax(dim=-1)
            generated.append(token)
            if offset + 1 == max_new_tokens:
                break
            hidden = policy.cached_hidden(
                token[:, None],
                past_key_values=cache,
                cache_position=torch.tensor([inputs.shape[1] + offset], device=device),
            )[:, -1]
            logits = policy.logits(hidden)
    return torch.stack(generated, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MINICPM5_MODEL_ID)
    parser.add_argument("--revision", default=MINICPM5_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--output", default="postraining/runs/minicpm5_hf_vapo_parity.json"
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("generation length must be positive")

    prepare_text_only_transformers_runtime()
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model: Any = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = model.to(device).eval()
    inputs = [prompt_ids(tokenizer, prompt, device) for prompt in PROMPTS]
    native_logits = []
    native_generations = []
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for tokens in inputs:
            # Match the deployed rollout head shape. A full-sequence output-head
            # GEMM can choose a different bf16 kernel and is not a parity test
            # for the one-token rollout path.
            native_logits.append(
                model(tokens, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            )
            native_generations.append(
                model.generate(
                    tokens,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=int(model.config.pad_token_id),
                )[:, tokens.shape[1]:].cpu()
            )

    policy = MiniCPMVAPOPolicy(model, LoRAConfig()).to(device).eval()
    max_cache_length = max(tokens.shape[1] for tokens in inputs) + args.max_new_tokens
    cache_pool = StaticCachePool(
        lambda: StaticCache(
            config=model.config,
            max_cache_len=max_cache_length,
        ),
        batch_size=1,
    )
    prompt_results = []
    passed = True
    for index, tokens in enumerate(inputs):
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            hidden = policy.replay_hidden(tokens, None)[:, -1]
            integrated_logits = policy.logits(hidden).float()
        difference = integrated_logits - native_logits[index]
        relative_l2 = float(
            difference.norm() / native_logits[index].norm().clamp_min(1e-12)
        )
        maximum_absolute = float(difference.abs().max())
        target = native_logits[index].argmax(dim=-1)
        selected = chunked_frozen_head_logprobs(
            hidden,
            target,
            policy.lm_head_weight,
            chunk_tokens=1,
        )
        native_selected = native_logits[index].log_softmax(dim=-1).gather(
            1, target[:, None]
        ).squeeze(1)
        selected_error = float((selected - native_selected).abs().max())
        integrated_generation = greedy_static_generation(
            policy,
            tokens,
            max_new_tokens=args.max_new_tokens,
            cache_pool=cache_pool,
        ).cpu()
        generation_equal = torch.equal(
            integrated_generation, native_generations[index]
        )
        prompt_passed = (
            relative_l2 <= 1e-5
            and selected_error <= 1e-5
            and generation_equal
        )
        passed &= prompt_passed
        prompt_results.append(
            {
                "prompt": PROMPTS[index],
                "relative_l2_error": relative_l2,
                "maximum_absolute_error": maximum_absolute,
                "selected_logprob_error": selected_error,
                "generation_equal": generation_equal,
                "passed": prompt_passed,
            }
        )

    result = {
        "schema": "minicpm5_hf_vapo_native_parity/v1",
        "model": args.model,
        "revision": args.revision,
        "max_new_tokens": args.max_new_tokens,
        "cache_resets": cache_pool.reset_count,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "prompts": prompt_results,
        "passed": passed,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()

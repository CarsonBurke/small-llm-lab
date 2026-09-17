"""Native batched math evaluation for pretrained Hugging Face causal LMs.

This is a checkpoint-selection probe, not a latent-policy evaluation.  It
measures each model through its native prompt interface before we spend the
engineering effort required to integrate a new backbone into latent VAPO.

The three default suites deliberately answer different questions:

* AIME-2024: 30 problems x 32 samples, matching the post-training avg@k guard.
* DeepMind easy: 144 problems x 8 samples, matching the automatic easy bench.
* DAPO: a fixed 64-problem hash sample x 16 samples, matching one RL pool's
  prompt-group geometry and exposing within-prompt collapse.

Generation is native ``transformers.generate`` with the model cache enabled.
Prompts are length-sorted and expanded to a fixed trajectory budget so each
GPU batch has little left-padding while preserving every prompt's contiguous
sample group.  Base checkpoints receive raw completions; checkpoints with a
chat template receive their native user/assistant framing.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from importlib.machinery import ModuleSpec
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
from types import ModuleType
from typing import Any

import torch
import torch.nn.functional as F

# The Hub kernels published for Falcon-H1 do not currently include Blackwell
# (sm_120).  Prefer the locally installed Mamba Triton kernels instead of
# silently selecting the very slow reference SSM.
os.environ["USE_HUB_KERNELS"] = "NO"

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from postraining.core import (
    answer_style,
    deterministic_math_subset,
    extract_final_answer,
    load_unique_math_rows,
    verify_answer,
)
from postraining.hf_runtime import prepare_text_only_transformers_runtime
from postraining.train_vapo import prompt_text


EVALUATION_SCHEMA = "hf_native_math_checkpoint_selection/v1"
DEFAULT_MODELS = (
    "tiiuae/Falcon-H1-Tiny-R-90M",
    "tiiuae/Falcon-H1-Tiny-90M-Base",
    "tiiuae/Falcon-H1-Tiny-90M-Instruct-Curriculum-pre-DPO",
    "openbmb/MiniCPM5-1B",
)
MODEL_REVISIONS = {
    "tiiuae/Falcon-H1-Tiny-R-90M":
        "7385612bf04c64405a51b29b6229d6d2ab0e72fd",
    "tiiuae/Falcon-H1-Tiny-90M-Base":
        "7994372e93b62822ae25f8bfb19f653649cea3a3",
    "tiiuae/Falcon-H1-Tiny-90M-Instruct-Curriculum-pre-DPO":
        "008dc03d2adc558ffa899aa6929bd2d6ecbdb896",
    "openbmb/MiniCPM5-1B":
        "87179e5c1f455ef22e6223592d2d61351b525bfc",
}


@dataclass(frozen=True)
class Suite:
    name: str
    path: str
    samples: int
    max_rows: int
    answer_style_override: str | None = None


DEFAULT_SUITES = (
    Suite(
        name="aime_2024",
        path="postraining/data/aime-2024.parquet",
        samples=32,
        max_rows=0,
        answer_style_override="aime",
    ),
    Suite(
        name="deepmind_easy",
        path="postraining/data/deepmind-interpolate-easy.parquet",
        samples=8,
        max_rows=0,
    ),
    Suite(
        name="dapo",
        path="postraining/data/dapo-math-17k.parquet",
        samples=16,
        max_rows=64,
    ),
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_suite_metrics_to_tensorboard(
    log_dir: str | Path,
    step: int,
    suite_metrics: dict[str, dict[str, Any]],
) -> None:
    """Append fixed-suite quality metrics to a training dashboard."""
    from torch.utils.tensorboard import SummaryWriter

    tensorboard = SummaryWriter(log_dir)
    for suite_name, metrics in suite_metrics.items():
        tensorboard.add_scalar(
            f"{suite_name}/accuracy", metrics["contract_accuracy"], step
        )
        for name in (
            "contract_accuracy",
            "relaxed_accuracy",
            "terminated_fraction",
            "capped_fraction",
        ):
            tensorboard.add_scalar(
                f"{suite_name}/{name}", metrics[name], step
            )
    tensorboard.close()


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def stable_suite_seed(seed: int, suite: str) -> int:
    digest = hashlib.sha256(suite.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "little")


def normalized_token_ids(
    value: int | list[int] | tuple[int, ...] | None,
) -> tuple[int, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, int) else tuple(value)
    return tuple(dict.fromkeys(int(token) for token in values))


def truncate_prompt(
    ids: list[int], maximum: int, bos_token_id: int | None
) -> list[int]:
    """Match latent VAPO's BOS-plus-tail prompt truncation."""
    if maximum < 1:
        raise ValueError("prompt token budget must be positive")
    if len(ids) <= maximum:
        return ids
    if bos_token_id is not None and ids[0] == bos_token_id and maximum > 1:
        return [ids[0], *ids[-(maximum - 1):]]
    return ids[-maximum:]


def last_boxed_answer(text: str) -> str | None:
    """Return the last balanced ``\\boxed{...}`` body, including nested braces."""
    marker = r"\boxed{"
    starts = [match.start() for match in re.finditer(re.escape(marker), text)]
    for start in reversed(starts):
        body_start = start + len(marker)
        depth = 1
        for index in range(body_start, len(text)):
            character = text[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return text[body_start:index].strip()
    return None


def relaxed_verify(text: str, truth: str, style: str) -> tuple[bool, str, str]:
    """Verify the Answer field, then fall back only to a balanced boxed answer."""
    correct, prediction = verify_answer(text, truth, style)
    if extract_final_answer(text, window=None) is not None:
        return correct, prediction, "answer_field"
    boxed = last_boxed_answer(text)
    if boxed is None:
        return False, prediction, "missing"
    correct, prediction = verify_answer(f"Answer: {boxed}", truth, style)
    return correct, prediction, "boxed"


def repeated_ngram_fraction(tokens: list[int], width: int = 4) -> float:
    if width < 1:
        raise ValueError("ngram width must be positive")
    count = len(tokens) - width + 1
    if count <= 0:
        return 0.0
    unique = len({tuple(tokens[index:index + width]) for index in range(count)})
    return 1.0 - unique / count


def has_terminal_loop(
    tokens: list[int],
    *,
    maximum_period: int = 16,
    minimum_repeats: int = 3,
) -> bool:
    """Detect an exact repeated token pattern at the end of a generation."""
    if maximum_period < 1 or minimum_repeats < 2:
        raise ValueError("loop period/repetition bounds are invalid")
    for period in range(1, min(maximum_period, len(tokens) // minimum_repeats) + 1):
        pattern = tokens[-period:]
        if all(
            tokens[-repeat * period:-(repeat - 1) * period] == pattern
            for repeat in range(2, minimum_repeats + 1)
        ):
            return True
    return False


def percentile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(probability * len(ordered)) - 1
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def bootstrap_mean_interval(
    values: list[float], *, seed: int = 0, replicates: int = 2_000
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    generator = random.Random(seed)
    means = sorted(
        statistics.fmean(generator.choices(values, k=len(values)))
        for _ in range(replicates)
    )
    return [
        means[int(0.025 * (replicates - 1))],
        means[int(0.975 * (replicates - 1))],
    ]


def prepare_prompt_ids(
    tokenizer,
    text: str,
    *,
    prompt_mode: str,
    prompt_tokens: int,
    enable_thinking: bool | None = None,
) -> list[int]:
    if prompt_mode == "chat":
        template_options: dict[str, Any] = {}
        if enable_thinking is not None:
            template_options["enable_thinking"] = enable_thinking
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            **template_options,
        )
        ids = (
            encoded["input_ids"]
            if isinstance(encoded, Mapping)
            else encoded
        )
    elif prompt_mode == "raw":
        if enable_thinking is not None:
            raise ValueError("thinking mode requires a chat prompt")
        ids = tokenizer.encode(text, add_special_tokens=True)
    else:
        raise ValueError(f"unknown prompt mode {prompt_mode!r}")
    return truncate_prompt(list(ids), prompt_tokens, tokenizer.bos_token_id)


def resolve_prompt_mode(tokenizer, requested: str) -> str:
    if requested != "auto":
        if requested == "chat" and not tokenizer.chat_template:
            raise ValueError("chat prompt mode requires a tokenizer chat template")
        return requested
    return "chat" if tokenizer.chat_template else "raw"


def resolved_eos_ids(model, tokenizer, prompt_mode: str) -> tuple[int, ...]:
    """Resolve actual terminal tokens, including Falcon's chat turn terminator."""
    candidates = [
        *normalized_token_ids(model.generation_config.eos_token_id),
        *normalized_token_ids(tokenizer.eos_token_id),
    ]
    if prompt_mode == "chat":
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (
            isinstance(im_end, int)
            and im_end >= 0
            and im_end != tokenizer.unk_token_id
            and tokenizer.convert_ids_to_tokens(im_end) == "<|im_end|>"
        ):
            candidates.append(im_end)
    eos_ids = normalized_token_ids(candidates)
    if not eos_ids:
        raise ValueError("model evaluation requires at least one EOS token")
    return eos_ids


def _causal_conv1d_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    seq_idx: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Inference-only PyTorch fallback around the fast Triton Mamba scan."""
    if seq_idx is not None:
        raise NotImplementedError("segmented causal convolution is not used here")
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError(f"unsupported activation {activation!r}")
    input_dtype = x.dtype
    x = x.to(weight.dtype)
    sequence_length = x.shape[-1]
    channels, width = weight.shape
    if initial_states is None:
        output = F.conv1d(
            x, weight.unsqueeze(1), bias, padding=width - 1, groups=channels
        )
    else:
        x = torch.cat((initial_states, x), dim=-1)
        output = F.conv1d(
            x, weight.unsqueeze(1), bias, padding=0, groups=channels
        )
    output = output[..., :sequence_length]
    if activation is not None:
        output = F.silu(output)
    output = output.to(input_dtype)
    if not return_final_states:
        return output
    final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(input_dtype)
    if final_states_out is None:
        final_states_out = final_states
    else:
        final_states_out.copy_(final_states)
    return output, final_states_out


def _causal_conv1d_update_reference(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cache_seqlens: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update a short depthwise-convolution cache without an ABI-bound extension."""
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError(f"unsupported activation {activation!r}")
    if conv_state_indices is not None:
        raise NotImplementedError("continuous-batch state indexing is not used here")
    input_dtype = x.dtype
    squeeze = x.ndim == 2
    if squeeze:
        x = x.unsqueeze(-1)
    batch, channels, sequence_length = x.shape
    width = weight.shape[1]
    state_length = conv_state.shape[-1]
    if conv_state.shape[:2] != (batch, channels):
        raise ValueError("convolution state has incompatible dimensions")
    if cache_seqlens is None:
        x_new = torch.cat((conv_state, x), dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[..., -state_length:])
    else:
        history = (
            torch.arange(
                -(width - 1), 0, dtype=torch.long, device=x.device
            ).unsqueeze(0)
            + cache_seqlens.unsqueeze(1)
        )
        history = history.remainder(state_length).unsqueeze(1)
        history = history.expand(-1, channels, -1)
        x_new = torch.cat((conv_state.gather(2, history), x), dim=-1)
        insertion = (
            torch.arange(
                sequence_length, dtype=torch.long, device=x.device
            ).unsqueeze(0)
            + cache_seqlens.unsqueeze(1)
        )
        insertion = insertion.remainder(state_length).unsqueeze(1)
        conv_state.scatter_(2, insertion.expand(-1, channels, -1), x)
        x_new = x_new.to(weight.dtype)
    output = F.conv1d(
        x_new, weight.unsqueeze(1), bias, padding=0, groups=channels
    )[..., -sequence_length:]
    if squeeze:
        output = output.squeeze(-1)
    if activation is not None:
        output = F.silu(output)
    return output.to(input_dtype)



def prepare_falcon_h1_runtime() -> None:
    """Install Blackwell-compatible Falcon-H1 shims."""
    prepare_text_only_transformers_runtime()

    spec = importlib.util.find_spec("mamba_ssm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "Falcon-H1 evaluation requires the mamba-ssm Python package"
        )
    mamba = ModuleType("mamba_ssm")
    mamba.__path__ = list(spec.submodule_search_locations)
    mamba.__package__ = "mamba_ssm"
    mamba.__spec__ = ModuleSpec("mamba_ssm", loader=None, is_package=True)
    mamba.__version__ = "2.3.2.post1"
    sys.modules["mamba_ssm"] = mamba

    causal_conv = ModuleType("causal_conv1d")
    causal_conv.__spec__ = ModuleSpec(
        "causal_conv1d", loader=None, is_package=False
    )
    causal_conv.causal_conv1d_fn = _causal_conv1d_reference
    causal_conv.causal_conv1d_update = _causal_conv1d_update_reference
    sys.modules["causal_conv1d"] = causal_conv

    from mamba_ssm.ops.triton.selective_state_update import (
        selective_state_update,
    )
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )

    mamba.selective_state_update = selective_state_update
    mamba.mamba_chunk_scan_combined = mamba_chunk_scan_combined
    mamba.mamba_split_conv1d_scan_combined = (
        mamba_split_conv1d_scan_combined
    )


def assert_fast_falcon_h1_runtime(model) -> None:
    if getattr(model.config, "model_type", None) != "falcon_h1":
        return
    from transformers.models.falcon_h1 import modeling_falcon_h1

    if not modeling_falcon_h1.is_fast_path_available:
        raise RuntimeError(
            "Falcon-H1 fast Mamba path is unavailable; refusing a misleading "
            "naive-SSM evaluation"
        )
def left_padded_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = attention_mask.to(torch.long).cumsum(dim=-1) - 1
    return positions.masked_fill(attention_mask == 0, 0)


@torch.inference_mode()
def validate_left_padded_logits(
    model,
    prompt_ids: list[list[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    """Ensure padding and position ids preserve final-token logits."""
    if len(prompt_ids) < 2:
        raise ValueError("left-padding validation requires two prompts")
    selected = sorted(prompt_ids, key=len)
    selected = [selected[0], selected[-1]]
    width = max(map(len, selected))
    batch = torch.full(
        (2, width), pad_token_id, dtype=torch.long, device=device
    )
    mask = torch.zeros_like(batch)
    for index, ids in enumerate(selected):
        batch[index, -len(ids):] = torch.tensor(ids, device=device)
        mask[index, -len(ids):] = 1
    batched_arguments: dict[str, Any] = {
        "input_ids": batch,
        "attention_mask": mask,
        "use_cache": False,
    }
    if getattr(model.config, "model_type", None) == "llama":
        batched_arguments["position_ids"] = left_padded_position_ids(mask)
    batched_logits = model(**batched_arguments).logits[:, -1].float()
    individual_logits = torch.stack(
        [
            model(
                input_ids=torch.tensor([ids], device=device),
                attention_mask=torch.ones(
                    (1, len(ids)), dtype=torch.long, device=device
                ),
                use_cache=False,
            ).logits[0, -1].float()
            for ids in selected
        ]
    )
    difference = batched_logits - individual_logits
    relative_l2 = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(individual_logits).clamp_min(1e-12)
    )
    maximum_absolute = float(difference.abs().max())
    argmax_equal = (
        batched_logits.argmax(dim=-1) == individual_logits.argmax(dim=-1)
    ).tolist()
    result = {
        "prompt_lengths": list(map(len, selected)),
        "relative_l2_error": relative_l2,
        "maximum_absolute_error": maximum_absolute,
        "argmax_equal": argmax_equal,
    }
    if relative_l2 > 0.02 or not all(argmax_equal):
        raise RuntimeError(f"left-padded logits failed validation: {result}")
    return result


@torch.inference_mode()
def autotune_batch_trajectories(
    model,
    tokenizer,
    rows: list[dict],
    *,
    suite: Suite,
    prompt_mode: str,
    prompt_tokens: int,
    enable_thinking: bool | None,
    maximum_batch_trajectories: int,
    temperature: float,
    top_p: float,
    seed: int,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    """Choose the fastest lockstep width using identical prompt/sample work."""
    candidates = []
    candidate = suite.samples
    while candidate <= maximum_batch_trajectories:
        candidates.append(candidate)
        candidate *= 2
    if not candidates:
        raise ValueError("batch limit cannot fit one complete prompt group")
    largest = candidates[-1]
    prompt_count = largest // suite.samples
    selected_rows = rows[:prompt_count]
    encoded = [
        prepare_prompt_ids(
            tokenizer,
            prompt_text(row),
            prompt_mode=prompt_mode,
            prompt_tokens=prompt_tokens,
            enable_thinking=enable_thinking,
        )
        for row in selected_rows
    ]
    eos_ids = resolved_eos_ids(model, tokenizer, prompt_mode)
    pad_token_id = model.generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("batch autotuning requires a pad token")

    results: dict[str, Any] = {}
    for batch_trajectories in candidates:
        prompts_per_batch = batch_trajectories // suite.samples
        generated_tokens = 0
        warmup_chunk = encoded[:prompts_per_batch]
        warmup_width = max(map(len, warmup_chunk))
        warmup_ids = torch.full(
            (len(warmup_chunk), warmup_width),
            int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        warmup_mask = torch.zeros_like(warmup_ids)
        for index, ids in enumerate(warmup_chunk):
            warmup_ids[index, -len(ids):] = torch.tensor(ids, device=device)
            warmup_mask[index, -len(ids):] = 1
        model.generate(
            input_ids=warmup_ids,
            attention_mask=warmup_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            num_return_sequences=suite.samples,
            max_new_tokens=2,
            use_cache=True,
            pad_token_id=int(pad_token_id),
            eos_token_id=list(eos_ids),
        )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for begin in range(0, len(encoded), prompts_per_batch):
            chunk = encoded[begin:begin + prompts_per_batch]
            width = max(map(len, chunk))
            input_ids = torch.full(
                (len(chunk), width),
                int(pad_token_id),
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for index, ids in enumerate(chunk):
                input_ids[index, -len(ids):] = torch.tensor(
                    ids, device=device
                )
                attention_mask[index, -len(ids):] = 1
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=0,
                num_return_sequences=suite.samples,
                max_new_tokens=128,
                use_cache=True,
                pad_token_id=int(pad_token_id),
                eos_token_id=list(eos_ids),
            )
            for continuation in generated[:, width:].to("cpu").tolist():
                tokens, terminated = _unpadded_continuation(
                    continuation, eos_ids, int(pad_token_id)
                )
                generated_tokens += len(tokens) - int(terminated)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        results[str(batch_trajectories)] = {
            "elapsed_seconds": elapsed,
            "useful_generated_tokens": generated_tokens,
            "useful_generated_tokens_per_second":
                generated_tokens / max(elapsed, 1e-9),
            "peak_allocated_bytes":
                torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    selected = max(
        candidates,
        key=lambda value:
            results[str(value)]["useful_generated_tokens_per_second"],
    )
    return selected, {
        "benchmark_max_new_tokens": 128,
        "prompt_count": prompt_count,
        "samples_per_prompt": suite.samples,
        "candidates": results,
        "selected_batch_trajectories": selected,
    }


def _unpadded_continuation(
    tokens: list[int],
    eos_ids: tuple[int, ...],
    pad_token_id: int | None,
) -> tuple[list[int], bool]:
    eos_set = set(eos_ids)
    for index, token in enumerate(tokens):
        if token in eos_set:
            return tokens[: index + 1], True
    if pad_token_id is not None:
        while tokens and tokens[-1] == pad_token_id:
            tokens.pop()
    return tokens, False


def _attempt_record(
    *,
    row: dict,
    problem_index: int,
    sample_index: int,
    continuation: list[int],
    terminated: bool,
    tokenizer,
    style: str,
) -> dict[str, Any]:
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    truth = str(row["reward_model"]["ground_truth"])
    contract_correct, contract_prediction = verify_answer(text, truth, style)
    relaxed_correct, relaxed_prediction, relaxed_source = relaxed_verify(
        text, truth, style
    )
    scored_tokens = continuation[:-1] if terminated else continuation
    answer_fields = len(re.findall(r"(?i)Answer\s*:", text))
    boxed = last_boxed_answer(text)
    return {
        "problem_index": problem_index,
        "dataset_index": str(
            (row.get("extra_info") or {}).get("index", problem_index)
        ),
        "sample_index": sample_index,
        "prompt": prompt_text(row),
        "ground_truth": truth,
        "answer_style": style,
        "emitted_token_ids": continuation,
        "emitted_text": text,
        "terminated": terminated,
        "emitted_token_count": len(scored_tokens),
        "answer_field_count": answer_fields,
        "boxed_answer": boxed,
        "contract_prediction": contract_prediction,
        "contract_correct_unterminated": bool(contract_correct),
        "contract_correct": bool(terminated and contract_correct),
        "relaxed_prediction": relaxed_prediction,
        "relaxed_source": relaxed_source,
        "relaxed_correct_unterminated": bool(relaxed_correct),
        "relaxed_correct": bool(terminated and relaxed_correct),
        "repeated_4gram_fraction": repeated_ngram_fraction(scored_tokens),
        "terminal_loop": has_terminal_loop(scored_tokens),
    }


def summarize_attempts(
    attempts: list[dict[str, Any]],
    *,
    problem_count: int,
    samples_per_problem: int,
    elapsed_seconds: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> dict[str, Any]:
    total = len(attempts)
    expected = problem_count * samples_per_problem
    if total != expected:
        raise ValueError(f"expected {expected} attempts, received {total}")
    lengths = [int(attempt["emitted_token_count"]) for attempt in attempts]
    predictions = [
        str(attempt["relaxed_prediction"])
        for attempt in attempts
        if attempt["relaxed_source"] != "missing"
    ]
    modal_prediction, modal_count = (
        Counter(predictions).most_common(1)[0] if predictions else ("", 0)
    )
    prompt_correct = [0] * problem_count
    prompt_predictions: list[list[str]] = [[] for _ in range(problem_count)]
    prompt_transcripts: list[list[str]] = [[] for _ in range(problem_count)]
    for attempt in attempts:
        problem_index = int(attempt["problem_index"])
        prompt_correct[problem_index] += int(attempt["contract_correct"])
        prompt_predictions[problem_index].append(
            str(attempt["relaxed_prediction"])
            if attempt["relaxed_source"] != "missing"
            else "<missing>"
        )
        prompt_transcripts[problem_index].append(
            str(attempt["emitted_text"])
        )
    prompt_accuracies = [
        count / samples_per_problem for count in prompt_correct
    ]
    prompt_any = [float(count > 0) for count in prompt_correct]
    unique_answers = [len(set(group)) for group in prompt_predictions]
    unique_transcripts = [len(set(group)) for group in prompt_transcripts]
    within_prompt_modal_shares = [
        Counter(group).most_common(1)[0][1] / samples_per_problem
        for group in prompt_predictions
    ]
    generated_tokens = sum(lengths)
    return {
        "samples": total,
        "problems": problem_count,
        "samples_per_problem": samples_per_problem,
        "contract_accuracy": sum(
            int(attempt["contract_correct"]) for attempt in attempts
        ) / total,
        "contract_content_accuracy": sum(
            int(attempt["contract_correct_unterminated"]) for attempt in attempts
        ) / total,
        "relaxed_accuracy": sum(
            int(attempt["relaxed_correct"]) for attempt in attempts
        ) / total,
        "relaxed_content_accuracy": sum(
            int(attempt["relaxed_correct_unterminated"]) for attempt in attempts
        ) / total,
        "terminated_fraction": sum(
            int(attempt["terminated"]) for attempt in attempts
        ) / total,
        "capped_fraction": sum(
            int(not attempt["terminated"]) for attempt in attempts
        ) / total,
        "answer_field_fraction": sum(
            int(int(attempt["answer_field_count"]) > 0) for attempt in attempts
        ) / total,
        "multiple_answer_field_fraction": sum(
            int(int(attempt["answer_field_count"]) > 1) for attempt in attempts
        ) / total,
        "boxed_fraction": sum(
            int(attempt["boxed_answer"] is not None) for attempt in attempts
        ) / total,
        "leading_boxed_fraction": sum(
            int(bool(re.match(r"^\s*(?:\\\[|\$\$)?\s*\\boxed\{", str(
                attempt["emitted_text"]
            ))))
            for attempt in attempts
        ) / total,
        "terminal_loop_fraction": sum(
            int(attempt["terminal_loop"]) for attempt in attempts
        ) / total,
        "mean_repeated_4gram_fraction": statistics.fmean(
            float(attempt["repeated_4gram_fraction"]) for attempt in attempts
        ),
        "modal_prediction": modal_prediction,
        "modal_prediction_fraction": modal_count / total,
        "unique_answers_per_prompt_mean": statistics.fmean(unique_answers),
        "unique_answers_per_prompt_p50": percentile(unique_answers, 0.50),
        "unique_transcripts_per_prompt_mean": statistics.fmean(
            unique_transcripts
        ),
        "identical_transcript_group_fraction": sum(
            int(count == 1) for count in unique_transcripts
        ) / problem_count,
        "within_prompt_modal_answer_share_mean": statistics.fmean(
            within_prompt_modal_shares
        ),
        "prompt_any_correct_fraction": sum(
            int(count > 0) for count in prompt_correct
        ) / problem_count,
        "contract_accuracy_prompt_bootstrap_95ci": bootstrap_mean_interval(
            prompt_accuracies, seed=17
        ),
        "prompt_any_correct_bootstrap_95ci": bootstrap_mean_interval(
            prompt_any, seed=23
        ),
        "prompt_mixed_correct_fraction": sum(
            int(0 < count < samples_per_problem) for count in prompt_correct
        ) / problem_count,
        "prompt_all_correct_fraction": sum(
            int(count == samples_per_problem) for count in prompt_correct
        ) / problem_count,
        "prompt_zero_correct_fraction": sum(
            int(count == 0) for count in prompt_correct
        ) / problem_count,
        "emitted_tokens_mean": statistics.fmean(lengths),
        "emitted_tokens_p50": percentile(lengths, 0.50),
        "emitted_tokens_p95": percentile(lengths, 0.95),
        "emitted_tokens_max": max(lengths, default=0),
        "elapsed_seconds": elapsed_seconds,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / max(elapsed_seconds, 1e-9),
        "attempts_per_second": total / max(elapsed_seconds, 1e-9),
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
    }


@torch.inference_mode()
def evaluate_suite(
    *,
    model,
    tokenizer,
    rows: list[dict],
    suite: Suite,
    prompt_mode: str,
    prompt_tokens: int,
    enable_thinking: bool | None,
    max_new_tokens: int,
    batch_trajectories: int,
    temperature: float,
    top_p: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if batch_trajectories < suite.samples:
        raise ValueError(
            "batch trajectories must fit at least one complete prompt group"
        )
    encoded = [
        (
            original_index,
            row,
            prepare_prompt_ids(
                tokenizer,
                prompt_text(row),
                prompt_mode=prompt_mode,
                prompt_tokens=prompt_tokens,
                enable_thinking=enable_thinking,
            ),
        )
        for original_index, row in enumerate(rows)
    ]
    encoded.sort(key=lambda item: (len(item[2]), item[0]))
    prompts_per_batch = max(1, batch_trajectories // suite.samples)
    eos_ids = resolved_eos_ids(model, tokenizer, prompt_mode)
    pad_token_id = model.generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("batched generation requires a pad token")

    def padded_batch(chunk):
        width = max(len(item[2]) for item in chunk)
        input_ids = torch.full(
            (len(chunk), width),
            int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, (_, _, ids) in enumerate(chunk):
            input_ids[row_index, -len(ids):] = torch.tensor(ids, device=device)
            attention_mask[row_index, -len(ids):] = 1
        return input_ids, attention_mask

    # Compile/autotune the prefill and recurrent Triton paths before measuring.
    warmup_chunk = encoded[:prompts_per_batch]
    warmup_ids, warmup_mask = padded_batch(warmup_chunk)
    model.generate(
        input_ids=warmup_ids,
        attention_mask=warmup_mask,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=0,
        num_return_sequences=suite.samples,
        max_new_tokens=2,
        use_cache=True,
        pad_token_id=int(pad_token_id),
        eos_token_id=list(eos_ids),
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    prompt_tokens_unpadded = 0
    prompt_tokens_padded = 0

    for begin in range(0, len(encoded), prompts_per_batch):
        chunk = encoded[begin:begin + prompts_per_batch]
        input_ids, attention_mask = padded_batch(chunk)
        width = input_ids.shape[1]
        prompt_tokens_unpadded += int(attention_mask.sum().item())
        prompt_tokens_padded += input_ids.numel()

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            num_return_sequences=suite.samples,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=int(pad_token_id),
            eos_token_id=list(eos_ids),
        )
        continuations = generated[:, width:].to("cpu")
        for flat_index, continuation_tensor in enumerate(continuations):
            chunk_index, sample_index = divmod(flat_index, suite.samples)
            original_index, row, _ = chunk[chunk_index]
            continuation, terminated = _unpadded_continuation(
                continuation_tensor.tolist(), eos_ids, int(pad_token_id)
            )
            style = suite.answer_style_override or answer_style(row)
            attempts.append(
                _attempt_record(
                    row=row,
                    problem_index=original_index,
                    sample_index=sample_index,
                    continuation=continuation,
                    terminated=terminated,
                    tokenizer=tokenizer,
                    style=style,
                )
            )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = summarize_attempts(
        attempts,
        problem_count=len(rows),
        samples_per_problem=suite.samples,
        elapsed_seconds=elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )
    metrics["prompt_padding_fraction"] = (
        1.0 - prompt_tokens_unpadded / prompt_tokens_padded
    )
    attempts.sort(
        key=lambda item: (int(item["problem_index"]), int(item["sample_index"]))
    )
    return metrics, attempts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--revision",
        help="immutable model revision; known candidates are pinned by default",
    )
    parser.add_argument(
        "--adapter-checkpoint",
        help="native VAPO adapter checkpoint to apply before evaluation",
    )
    parser.add_argument("--output", default="postraining/runs/hf_falcon_eval")
    parser.add_argument(
        "--prompt-mode", choices=("auto", "raw", "chat"), default="auto"
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="forward enable_thinking to the checkpoint's native chat template",
    )
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--samples-per-problem",
        type=int,
        default=0,
        help="override suite sample counts; zero keeps each suite default",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=0,
        help="deterministically limit each suite; zero keeps its default",
    )
    parser.add_argument("--batch-trajectories", type=int, default=128)
    parser.add_argument(
        "--autotune-batch-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="benchmark power-of-two widths up to --batch-trajectories",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--suite",
        action="append",
        choices=tuple(suite.name for suite in DEFAULT_SUITES),
        help="suite(s) to run; repeat the flag, or omit it for all",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        help="optional training TensorBoard directory to receive suite metrics",
    )
    parser.add_argument(
        "--tensorboard-step",
        type=int,
        help="training step attached to --tensorboard-log-dir metrics",
    )
    return parser

def load_vapo_adapter_for_evaluation(
    model,
    checkpoint_path: str | Path,
    *,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    """Apply a native VAPO adapter while preserving rollout bf16 arithmetic."""
    from postraining.minicpm_vapo import (
        LoRAConfig,
        inject_lora,
        load_adapter_state_dict,
    )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    payload = checkpoint["policy"]
    if payload.get("schema") == "minicpm5_vapo_latent/v1":
        raise ValueError(
            "latent checkpoints require the MiniCPM latent rollout engine; "
            "native-token evaluation would discard their thinking policy"
        )
    if str(payload.get("schema", "")).startswith("minicpm5_vapo_token_carry/"):
        raise ValueError(
            "token-carry checkpoints require CapturedTrainingRolloutEngine; "
            "stock generate would silently discard the recurrent hidden input"
        )
    policy = (
        payload["actor"]
        if payload.get("schema") == "minicpm5_vapo_adapter/v6"
        else payload
    )
    if policy["model_id"] != model_id or policy["revision"] != revision:
        raise ValueError("adapter base checkpoint differs from evaluation model")
    config = dict(policy["lora_config"])
    config["targets"] = tuple(config["targets"])
    inject_lora(model, LoRAConfig(**config))
    load_adapter_state_dict(model, policy["adapter"])
    model_dtype = next(
        parameter.dtype
        for name, parameter in model.named_parameters()
        if not name.endswith(("lora_a", "lora_b"))
    )
    for name, parameter in model.named_parameters():
        if name.endswith(("lora_a", "lora_b")):
            parameter.data = parameter.data.to(dtype=model_dtype)
            parameter.requires_grad_(False)
    return {
        "path": str(checkpoint_path),
        "step": int(checkpoint["step"]),
        "thinking": bool(checkpoint["args"]["thinking"]),
        "lora_config": config,
    }


def main() -> None:
    args = build_parser().parse_args()
    if (args.tensorboard_log_dir is None) != (args.tensorboard_step is None):
        raise ValueError(
            "--tensorboard-log-dir and --tensorboard-step must be set together"
        )
    if args.tensorboard_step is not None and args.tensorboard_step < 0:
        raise ValueError("--tensorboard-step must be non-negative")
    if args.prompt_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("prompt and generation budgets must be positive")
    if args.batch_trajectories < 1:
        raise ValueError("batch trajectory budget must be positive")
    if not 0 < args.temperature or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be positive and top-p must be in (0, 1]")
    if args.samples_per_problem < 0 or args.max_problems < 0:
        raise ValueError("sample and problem overrides must be nonnegative")
    selected = set(args.suite or ())
    suites = []
    for suite in DEFAULT_SUITES:
        if selected and suite.name not in selected:
            continue
        samples = args.samples_per_problem or suite.samples
        maximum = suite.max_rows
        if args.max_problems:
            maximum = min(maximum, args.max_problems) if maximum else args.max_problems
        suites.append(
            Suite(
                name=suite.name,
                path=suite.path,
                samples=samples,
                max_rows=maximum,
                answer_style_override=suite.answer_style_override,
            )
        )

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    revision = args.revision or MODEL_REVISIONS.get(args.model)
    if revision is None:
        raise ValueError(
            "an immutable --revision is required for an unrecognized model"
        )
    prepare_text_only_transformers_runtime()
    checkpoint_config = AutoConfig.from_pretrained(args.model, revision=revision)
    if getattr(checkpoint_config, "model_type", None) == "falcon_h1":
        prepare_falcon_h1_runtime()
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        config=checkpoint_config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    adapter = None
    if args.adapter_checkpoint:
        adapter = load_vapo_adapter_for_evaluation(
            model,
            args.adapter_checkpoint,
            model_id=args.model,
            revision=revision,
        )
        if args.thinking is None:
            args.thinking = adapter["thinking"]
        elif bool(args.thinking) != adapter["thinking"]:
            raise ValueError(
                "evaluation thinking mode differs from adapter training"
            )
    model = model.to(device).eval()
    assert_fast_falcon_h1_runtime(model)
    load_seconds = time.perf_counter() - load_started
    prompt_mode = resolve_prompt_mode(tokenizer, args.prompt_mode)
    eos_ids = resolved_eos_ids(model, tokenizer, prompt_mode)

    destination_name = model_slug(args.model)
    if adapter is not None:
        destination_name += f"-vapo-step-{adapter['step']}"
    destination = Path(args.output) / destination_name
    suite_metrics: dict[str, dict[str, Any]] = {}
    padding_validation = None
    batch_autotune = None
    selected_batch_trajectories = args.batch_trajectories
    for suite in suites:
        rows = load_unique_math_rows(suite.path)
        rows = deterministic_math_subset(rows, suite.max_rows)
        uses_left_padding = (
            args.autotune_batch_trajectories
            or args.batch_trajectories > suite.samples
        )
        if padding_validation is None and uses_left_padding:
            candidate_ids = [
                prepare_prompt_ids(
                    tokenizer,
                    prompt_text(row),
                    prompt_mode=prompt_mode,
                    prompt_tokens=args.prompt_tokens,
                    enable_thinking=args.thinking,
                )
                for row in rows[:8]
            ]
            pad_token_id = model.generation_config.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                raise ValueError("left-padding validation requires a pad token")
            padding_validation = validate_left_padded_logits(
                model,
                candidate_ids,
                pad_token_id=int(pad_token_id),
                device=device,
            )
            print(
                f"[padding] relative_l2="
                f"{padding_validation['relative_l2_error']:.3e} "
                f"argmax_equal={padding_validation['argmax_equal']}",
                flush=True,
            )
        if batch_autotune is None and args.autotune_batch_trajectories:
            (
                selected_batch_trajectories,
                batch_autotune,
            ) = autotune_batch_trajectories(
                model,
                tokenizer,
                rows,
                suite=suite,
                prompt_mode=prompt_mode,
                prompt_tokens=args.prompt_tokens,
                enable_thinking=args.thinking,
                maximum_batch_trajectories=args.batch_trajectories,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=stable_suite_seed(args.seed, "batch_autotune"),
                device=device,
            )
            print(
                f"[batch] selected {selected_batch_trajectories} "
                f"trajectories: {batch_autotune['candidates']}",
                flush=True,
            )
        print(
            f"[{suite.name}] {len(rows)} problems x {suite.samples} samples "
            f"({prompt_mode=})",
            flush=True,
        )
        metrics, attempts = evaluate_suite(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            suite=suite,
            prompt_mode=prompt_mode,
            prompt_tokens=args.prompt_tokens,
            enable_thinking=args.thinking,
            max_new_tokens=args.max_new_tokens,
            batch_trajectories=selected_batch_trajectories,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=stable_suite_seed(args.seed, suite.name),
            device=device,
        )
        suite_metrics[suite.name] = metrics
        _atomic_jsonl(destination / f"{suite.name}_attempts.jsonl", attempts)
        _atomic_json(destination / f"{suite.name}_metrics.json", metrics)
        print(
            f"[{suite.name}] contract={metrics['contract_accuracy']:.4%} "
            f"relaxed={metrics['relaxed_accuracy']:.4%} "
            f"terminated={metrics['terminated_fraction']:.4%} "
            f"tokens/s={metrics['generated_tokens_per_second']:.1f}",
            flush=True,
        )

    if padding_validation is None:
        padding_validation = {
            "skipped": True,
            "reason": "one_prompt_per_batch",
        }

    config = model.config
    summary = {
        "schema": EVALUATION_SCHEMA,
        "model": args.model,
        "revision": revision,
        "model_slug": model_slug(args.model),
        "adapter": adapter,
        "prompt_mode": prompt_mode,
        "enable_thinking": args.thinking,
        "samples_per_problem_override": args.samples_per_problem,
        "max_problems_override": args.max_problems,
        "load_seconds": load_seconds,
        "dtype": str(next(model.parameters()).dtype),
        "model_type": getattr(config, "model_type", None),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "vocab_size": len(tokenizer),
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "maximum_batch_trajectories": args.batch_trajectories,
        "batch_trajectories": selected_batch_trajectories,
        "batch_autotune": batch_autotune,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "eos_token_ids": list(eos_ids),
        "pad_token_id": model.generation_config.pad_token_id,
        "left_padding_validation": padding_validation,
        "suites": suite_metrics,
    }
    _atomic_json(destination / "summary.json", summary)
    if args.tensorboard_log_dir is not None:
        assert args.tensorboard_step is not None
        write_suite_metrics_to_tensorboard(
            args.tensorboard_log_dir,
            args.tensorboard_step,
            suite_metrics,
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Single-GPU LoRA VAPO for native MiniCPM5-1B.

The final RL+OPD checkpoint is the default because its math success rate gives
sparse-reward VAPO useful within-prompt variation immediately. The SFT checkpoint
can still be selected explicitly, but no tokenizer, chat-template, or attention
block is replaced. Native rollout stores token ids and selected-action statistics;
opt-in latent thinking also stores exact continuous actions for actor and critic
replay. Only answer-token states require vocabulary projection in latent mode.
"""

from __future__ import annotations

import argparse
import atexit
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch._functorch import config as aot_config
from torch.utils.tensorboard import SummaryWriter

from checkpointing import RecoveryCheckpointPolicy, atomic_link_or_copy, atomic_torch_save
from postraining.core import JsonlLogger, answer_style, load_unique_math_rows, math_corpus_identity, verify_answer
from postraining.eval_hf_math import prepare_prompt_ids, resolved_eos_ids
from postraining.fast_inference import (
    CapturedTrainingRolloutEngine,
    FastTrainingDecodeStats,
    response_token_limits,
    top_k_top_p_sample,
)
from postraining.minicpm_vapo import (
    MINICPM5_MODEL_ID,
    MINICPM5_VOCAB_SIZE,
    MINICPM5_REVISION,
    LoRAConfig,
    MiniCPMVAPOPolicy,
    MiniCPMVAPOCritic,
    adapter_state_dict,
    StaticCachePool,
    ReplayMicrobatch,
    TrajectoryRecord,
    _precompute_advantages,
    chunked_frozen_head_logprobs,
    collate_replay_microbatch,
    enable_packed_replay_attention,
    enable_replay_mlp_compilation,
    exact_top_p_sample,
    load_adapter_state_dict,
    plan_replay_microbatches,
    replay_storage_bytes,
    use_packed_replay_attention,
)
from postraining.minicpm_latent_rollout import MiniCPMLatentRolloutEngine
from postraining.runtime.profiling import DEVICE_SAMPLE_FIELDS, DeviceSampler
from postraining.slot_memory import (
    SLOT_MEMORY_SCHEMA,
    SlotMemoryConfig,
    slot_choice_statistics,
    visibility_reference,
)
from postraining.repetition import repetition_metrics
from postraining.train_vapo import prompt_text
from postraining.thinking_budget import force_thinking_end_, validate_thinking_budget
from postraining.verifiable_tasks import (
    is_verifiable_task,
    score_verifiable_response,
    verifiable_reward_identity,
    validate_verifiable_task,
)

from postraining.nextlat_speculative import (
    NextLatDecodeStats,
    NextLatSpeculativeEngine,
)

@dataclass(frozen=True)
class DecodeStats:
    target_decode_calls: int = 0
    target_decode_positions: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0

@dataclass(frozen=True)
class NextLatTrainingLoss:
    loss: Tensor
    smooth_l1: Tensor
    categorical_kl: Tensor
    samples: int
    transitions: int


@dataclass(frozen=True)
class RolloutResult:
    records: list[TrajectoryRecord]
    generated_tokens: int
    scheduled_tokens: int
    elapsed_seconds: float
    sampling_scanned_vocabulary: int
    sampling_candidate_support: int
    sampling_full_policy_mass_lower_bound: float
    sampling_conditional_mass_lower_bound: float
    admission_events: int
    minimum_active_rows_with_backlog: int
    decoding: DecodeStats | FastTrainingDecodeStats | NextLatDecodeStats
    scoring_seconds: float = 0.0
    uno_metrics: dict[str, float | int] | None = None
    group_domains: tuple[str, ...] = ()
    outcome_reasons: tuple[str, ...] = ()
    response_limits: tuple[int, ...] = ()

def static_kv_cache_bytes(
    model_config: Any,
    *,
    batch_size: int,
    cache_length: int,
    element_size: int = 2,
) -> int:
    """Exact dense Llama K/V storage for one static rollout cache."""
    if batch_size < 1 or cache_length < 1 or element_size < 1:
        raise ValueError("cache dimensions must be positive")
    return (
        batch_size
        * cache_length
        * int(model_config.num_hidden_layers)
        * 2
        * int(model_config.num_key_value_heads)
        * int(model_config.head_dim)
        * element_size
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_math_corpus(
    path: str | Path, *, seed: int
) -> tuple[list[dict], dict[str, Any], str]:
    """Load a unique global corpus; preserve the historical math-only API."""
    audit: dict[str, Any] = {}
    rows = load_unique_math_rows(path, audit=audit)
    if not rows:
        raise ValueError("effective math corpus contains no trainable questions")
    for row in rows:
        if is_verifiable_task(row):
            validate_verifiable_task(row)
    random.Random(seed).shuffle(rows)
    counts = Counter(training_row_domain(row) for row in rows)
    audit["domain_counts"] = dict(sorted(counts.items()))
    audit["domain_proportions"] = {
        domain: count / len(rows) for domain, count in sorted(counts.items())
    }
    identity = math_corpus_identity(rows)
    if any(is_verifiable_task(row) for row in rows):
        reward_identity = verifiable_reward_identity()
        audit["verifiable_reward_identity"] = reward_identity
        identity = hashlib.sha256(json.dumps({
            "corpus": identity,
            "reward": reward_identity,
            "response_boundary": "native_special_tokens_prefilled_thinking/v1",
            "domains": [training_row_domain(row) for row in rows],
        }, sort_keys=True).encode("utf-8")).hexdigest()
    return rows, audit, identity


def training_row_domain(row: dict) -> str:
    return row["extra_info"]["domain"] if is_verifiable_task(row) else "legacy_math"


def select_corpus_rows(rows: list[dict], *, cursor: int, count: int) -> list[dict]:
    """Advance one global permutation, without replacement within each full pass."""
    if not rows or cursor < 0 or not 1 <= count <= len(rows):
        raise ValueError("rollout requires a nonempty corpus and no repeated prompt per batch")
    return [rows[(cursor + index) % len(rows)] for index in range(count)]


def corpus_progress(cursor: int, size: int) -> dict[str, float | int]:
    return {
        "corpus_questions_seen": cursor,
        "corpus_completed_epochs": cursor // size,
        "corpus_cursor": cursor % size,
        "corpus_epoch_progress": (cursor % size) / size,
    }


def validate_resume_dataset(
    resume: dict[str, Any], data_sha256: str, corpus_identity: str
) -> None:
    if resume.get("data_sha256") != data_sha256:
        raise ValueError("resume checkpoint was trained from different dataset bytes")
    if not resume.get("math_corpus_identity"):
        raise ValueError(
            "resume checkpoint is missing math_corpus_identity; its old corpus cursor "
            "cannot be reused for exact resume (actor-only evaluation is unaffected)"
        )
    if resume["math_corpus_identity"] != corpus_identity:
        raise ValueError(
            "resume checkpoint has a different effective math corpus or question order"
        )

DEFAULT_ANSWER_RESERVE_TOKENS = 1_000

# Missing metadata in pre-reserve checkpoints must preserve their unforced behavior.
LEGACY_RESUME_DEFAULTS = {
    "top_k": 20,
    "lora_initialization": "standard",
    "nextlat_trunk_balance": "parameter",
    "rollout_physical_batch_size": 0,
    "uno_rollout": False,
    "uno_checkpoint": None,
    "uno_block_size": 4,
    "answer_reserve_tokens": 0,
    "prompt_suffix": "",
    "latent_thinking": False,
    "token_carry": False,
    "slot_memory": False,
    "slot_memory_slots": 64,
    "slot_memory_heads": 1,
    "slot_memory_head_dim": 128,
    "thought_sigma": 1.0,
    "init_stop_thinking_probability": 0.9,
    "behavior_logprobs": "replay",
}

RESUME_MUTABLE_OPTIONS = frozenset(
    {
        "steps",
        "output",
        "checkpoint_interval_seconds",
        "resume",
        "device_telemetry",
        "device_telemetry_interval_ms",
        "device_power_floor",
        "min_rollout_tokens_per_second",
        "rollout_only",
        "rollout_physical_batch_size",
        "gate_min_positive_trajectories",
        "gate_min_positive_groups",
        "gate_max_truncation_fraction",
        "logit_chunk_tokens",
        "nextlat_kl_chunk_tokens",
        "replay_token_budget",
        "nextlat_trunk_balance",
        "replay_max_trajectories",
        "replay_checkpoint_interval",
        "replay_attention_backend",
        "compile_replay",
        "actor_lr",
        "critic_lr",
        "uno_checkpoint_sha256",
        "uno_arithmetic",
        "rollout_arithmetic",
    }
)


def policy_checkpoint_schema(
    *, latent_thinking: bool, token_carry: bool, slot_memory: bool
) -> str:
    if slot_memory:
        if not token_carry or latent_thinking:
            raise ValueError("slot memory requires token carry without latent thinking")
        return SLOT_MEMORY_SCHEMA
    if latent_thinking:
        return "minicpm5_vapo_latent/v1"
    if token_carry:
        return "minicpm5_vapo_token_carry/v4"
    return "minicpm5_vapo_adapter/v6"


def slot_memory_config_from_args(args) -> SlotMemoryConfig | None:
    if not getattr(args, "slot_memory", False):
        return None
    return SlotMemoryConfig(
        slots=int(args.slot_memory_slots),
        heads=int(args.slot_memory_heads),
        head_dim=int(args.slot_memory_head_dim),
    )


def validate_resume_configuration(resume: dict[str, Any], args) -> None:
    mutable_options = RESUME_MUTABLE_OPTIONS
    prior_args = resume["args"]
    if "policy" in resume:
        expected_schema = policy_checkpoint_schema(
            latent_thinking=args.latent_thinking,
            token_carry=args.token_carry,
            slot_memory=getattr(args, "slot_memory", False),
        )
        if resume["policy"].get("schema") != expected_schema:
            raise ValueError(f"resume checkpoint requires {expected_schema}")
    if prior_args.get("top_k", 20) != args.top_k:
        prior_top_k = prior_args.get("top_k", 20)
        raise ValueError(
            f"checkpoint used top-k={prior_top_k}; resume with --top-k {prior_top_k} "
            "or start a new run without --resume to change the sampling policy"
        )
    mismatches = [
        name
        for name, value in vars(args).items()
        if name not in mutable_options
        and prior_args.get(name, LEGACY_RESUME_DEFAULTS.get(name)) != value
        and not (
            name in {"max_new_tokens", "answer_reserve_tokens", "prompt_suffix"}
            and resume.get("pending_records") is None
            and prior_args.get("context_tokens") is None
            and getattr(args, "context_tokens", None) is None
        )
        and not (
            name in {"optimizer_minibatches", "behavior_logprobs"}
            and resume.get("pending_records") is None
        )
        and not (
            name in {"uno_rollout", "uno_checkpoint", "uno_block_size"}
            and resume.get("pending_records") is None
        )
    ]
    if mismatches:
        raise ValueError(
            "resume configuration differs for: " + ", ".join(mismatches)
        )
    if getattr(args, "token_carry", False) and resume.get("pending_records") is not None:
        slot_config = slot_memory_config_from_args(args)
        for record in resume["pending_records"]:
            if not isinstance(record, TrajectoryRecord) or getattr(record, "carry_hiddens", None) is None:
                raise ValueError("pending token-carry replay requires stored carry hiddens")
            if (getattr(record, "slot_choices", None) is None) != (slot_config is None):
                raise ValueError("pending replay slot-memory mode differs from the run")
            if slot_config is not None and record.slot_count != slot_config.slots:
                raise ValueError("pending replay slot count differs from the run")
            record.__post_init__()
    if prior_args.get("uno_rollout", False) and args.uno_rollout:
        expected = prior_args.get("uno_checkpoint_sha256")
        if not expected or file_sha256(args.uno_checkpoint) != expected:
            raise ValueError("resume Uno adapter bytes differ from the pinned checkpoint")
        if resume.get("pending_records") is not None:
            from postraining.invariant_linear import INVARIANT_ARITHMETIC

            if prior_args.get("uno_arithmetic") != INVARIANT_ARITHMETIC:
                raise ValueError("pending Uno rollout has a different numerical target")


def validate_resume_rollout_arithmetic(resume: dict[str, Any], arithmetic: str) -> None:
    """Do not reinterpret pending AR generations after a runtime-default change."""
    if resume.get("pending_records") is None:
        return
    from postraining.invariant_linear import LEGACY_ARITHMETIC

    # Checkpoints predating this identity were generated by the legacy AR path.
    prior_arithmetic = resume["args"].get("rollout_arithmetic", LEGACY_ARITHMETIC)
    if prior_arithmetic != arithmetic:
        raise ValueError("pending AR rollout has a different numerical target")


def reassert_optimizer_learning_rates(
    actor_optimizer,
    critic_optimizer,
    *,
    actor_lr: float,
    critic_lr: float,
) -> None:
    """Keep CLI-owned RL rates when AdamW restores checkpoint group metadata."""
    for group in actor_optimizer.param_groups:
        group["lr"] = actor_lr
    for group in critic_optimizer.param_groups:
        group["lr"] = critic_lr




def _stop_ids(policy: MiniCPMVAPOPolicy, tokenizer) -> tuple[int, ...]:
    return resolved_eos_ids(policy.causal_lm, tokenizer, "chat")


def resolve_thinking_end_token(tokenizer, *, stop_ids: tuple[int, ...]) -> int:
    token_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError("answer reserve requires a single native </think> token")
    token_id = int(token_ids[0])
    if token_id in stop_ids:
        raise ValueError("thinking end token cannot also be a response stop token")
    return token_id


def resolve_thinking_start_token(tokenizer, *, stop_ids: tuple[int, ...]) -> int:
    token_ids = tokenizer.encode("<think>", add_special_tokens=False)
    if len(token_ids) != 1 or tokenizer.decode(token_ids) != "<think>":
        raise ValueError("latent thinking requires one native <think> token")
    token_id = int(token_ids[0])
    if token_id in stop_ids:
        raise ValueError("thinking start token cannot be an end-of-response token")
    return token_id


def _math_prompt_text(row: dict, prompt_suffix: str) -> str:
    text = prompt_text(row)
    if is_verifiable_task(row):
        return text
    return f"{text}\n\n{prompt_suffix}" if prompt_suffix else text


def encode_math_prompt(
    tokenizer,
    row: dict,
    *,
    prompt_tokens: int,
    enable_thinking: bool,
    prompt_suffix: str = "",
    truncate_prompt: bool = True,
) -> Tensor:
    if is_verifiable_task(row) and "prompt_token_cap" in row["extra_info"]:
        row_cap = row["extra_info"]["prompt_token_cap"]
        if type(row_cap) is not int or row_cap < 1:
            raise ValueError("prompt_token_cap must be a positive integer")
        prompt_tokens = min(prompt_tokens, row_cap)
    if is_verifiable_task(row) or not truncate_prompt:
        messages = row["prompt"] if is_verifiable_task(row) else [
            {"role": "user", "content": _math_prompt_text(row, prompt_suffix)}
        ]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        ids = list(encoded["input_ids"] if isinstance(encoded, Mapping) else encoded)
        if not ids or len(ids) > prompt_tokens:
            raise ValueError(
                f"Full chat prompt has {len(ids)} tokens, exceeding "
                f"the configured bound {prompt_tokens}; reprepare instead of truncating"
            )
        return torch.tensor(ids, dtype=torch.long)
    ids = prepare_prompt_ids(
        tokenizer,
        _math_prompt_text(row, prompt_suffix),
        prompt_mode="chat",
        prompt_tokens=prompt_tokens,
        enable_thinking=enable_thinking,
    )
    return torch.tensor(ids, dtype=torch.long)


def _first_stop_length(tokens: Tensor, stop_ids: tuple[int, ...]) -> int:
    stop = torch.zeros_like(tokens, dtype=torch.bool)
    for token_id in stop_ids:
        stop |= tokens == token_id
    positions = stop.nonzero(as_tuple=False)
    return int(positions[0]) + 1 if positions.numel() else tokens.numel()


def _decode_text(tokenizer, token_ids: Tensor) -> str:
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)


def score_training_response(
    tokenizer, row: dict, prompt_ids: Tensor, response: Tensor,
    *, prefilled_thinking: bool | None = None,
) -> tuple[str, bool, str]:
    """Keep native answer boundaries for verifiable tasks and prefilled thinking."""
    if not is_verifiable_task(row):
        text = _decode_text(tokenizer, response)
        correct, _ = verify_answer(
            text, row["reward_model"]["ground_truth"], answer_style(row),
        )
        return text, correct, "correct" if correct else "incorrect"
    text = tokenizer.decode(response.tolist(), skip_special_tokens=False)
    # Chat templates can supply <think> outside the generated response. Carry it
    # into grading so a capped scratchpad cannot masquerade as a final answer.
    if prefilled_thinking is None:
        prefix = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=False)
        prefilled_thinking = prefix.rstrip().endswith("<think>")
    if prefilled_thinking:
        text = "<think>" + text
    correct, reason = score_verifiable_response(row, text)
    return text, correct, reason



class RolloutEngine:
    """Fully batched multi-prompt generation over one reusable static KV cache."""

    def __init__(
        self,
        policy: MiniCPMVAPOPolicy,
        tokenizer,
        *,
        prompts_per_rollout: int,
        samples_per_prompt: int,
        cache_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        compile_decode: bool,
        answer_reserve_tokens: int = 0,
        thinking_end_token_id: int | None = None,
    ) -> None:
        from transformers import StaticCache

        if getattr(policy, "token_carry", False):
            raise ValueError("token carry requires CapturedTrainingRolloutEngine")
        if prompts_per_rollout < 1 or samples_per_prompt < 1:
            raise ValueError("rollout batch dimensions must be positive")
        self.policy = policy
        self.tokenizer = tokenizer
        self.prompts_per_rollout = prompts_per_rollout
        self.samples_per_prompt = samples_per_prompt
        self.batch_size = prompts_per_rollout * samples_per_prompt
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.cache_length = cache_length
        self.stop_ids = _stop_ids(policy, tokenizer)
        self.primary_stop = self.stop_ids[0]
        validate_thinking_budget(answer_reserve_tokens, thinking_end_token_id)
        self.answer_reserve_tokens = answer_reserve_tokens
        self.thinking_end_token_id = thinking_end_token_id
        model_config: Any = getattr(policy.causal_lm, "config")
        if not -1 <= top_k <= int(model_config.vocab_size):
            raise ValueError("top-k must be -1, zero, or fit the model vocabulary")
        if top_k == -1 and top_p != 1.0:
            raise ValueError("unrestricted top-k sampling requires top-p=1")
        device = next(policy.parameters()).device
        self.estimated_cache_bytes = static_kv_cache_bytes(
            model_config,
            batch_size=self.batch_size,
            cache_length=cache_length,
        )
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if self.estimated_cache_bytes > int(0.7 * free_bytes):
            raise MemoryError(
                "fully parallel rollout cache requires "
                f"{self.estimated_cache_bytes / 2**30:.2f} GiB with only "
                f"{free_bytes / 2**30:.2f} GiB free; reduce the response "
                "context or use a larger GPU"
            )
        self.cache_pool = StaticCachePool(
            lambda: StaticCache(
                config=model_config,
                max_cache_len=cache_length,
            ),
            batch_size=self.batch_size,
        )
        self.cache_positions = torch.arange(cache_length, device=device)
        self.stop_tensor = torch.tensor(self.stop_ids, device=device)
        self.primary_stop_tensor = torch.tensor(self.primary_stop, device=device)
        self.generated_buffer = torch.empty(
            (self.batch_size, cache_length), dtype=torch.long, device=device
        )
        self.logprob_buffer = torch.empty(
            (self.batch_size, cache_length), dtype=torch.float32, device=device
        )
        self.value_buffer = torch.empty_like(self.logprob_buffer)
        self.attention_mask_buffer = torch.zeros(
            (self.batch_size, cache_length), dtype=torch.bool, device=device
        )
        self.prompt_lengths_buffer = torch.empty(
            self.batch_size, dtype=torch.long, device=device
        )
        self.position_id_buffer = torch.empty(
            (self.batch_size, 1), dtype=torch.long, device=device
        )
        if hasattr(torch, "_dynamo"):
            torch._dynamo.mark_static_address(self.attention_mask_buffer)
            torch._dynamo.mark_static_address(self.position_id_buffer)

        def decode(
            token_ids: Tensor,
            cache_position: Tensor,
            cache,
            attention_mask: Tensor,
            position_ids: Tensor,
        ):
            hidden = policy.cached_hidden(
                token_ids[:, None],
                past_key_values=cache,
                cache_position=cache_position,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[:, -1]
            return hidden, policy.logits(hidden), policy.rollout_values(hidden)

        def sample_bounded(logits: Tensor) -> tuple[Tensor, Tensor]:
            sampled = top_k_top_p_sample(
                logits,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            policy_logits = logits.float()
            sampled_logprobs = (
                policy_logits.gather(1, sampled[:, None]).squeeze(1)
                - policy_logits.logsumexp(dim=-1)
            )
            return sampled, sampled_logprobs

        self.sample_bounded = (
            torch.compile(sample_bounded, fullgraph=True)
            if compile_decode and top_k
            else sample_bounded
        )
        self.decode = (
            torch.compile(decode, mode="reduce-overhead", fullgraph=False)
            if compile_decode
            else decode
        )

    def release_cache(self) -> None:
        self.cache_pool.clear()

    def _prepare_prompts(
        self, prompt_ids_cpu: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor, int]:
        if len(prompt_ids_cpu) != self.prompts_per_rollout:
            raise ValueError(
                "rollout prompt count differs from the configured batch"
            )
        lengths = [prompt.numel() for prompt in prompt_ids_cpu]
        if not lengths or min(lengths) < 1:
            raise ValueError("rollout prompts cannot be empty")
        maximum = max(lengths)
        if maximum >= self.cache_length:
            raise ValueError("prompt exhausts the rollout cache")

        device = self.generated_buffer.device
        pad_token_id = int(getattr(self.policy.causal_lm, "config").pad_token_id)
        prompt_batch = torch.full(
            (self.batch_size, maximum),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.zeros_like(prompt_batch)
        attention_mask = self.attention_mask_buffer
        attention_mask.zero_()
        for group, (prompt, length) in enumerate(zip(prompt_ids_cpu, lengths)):
            row_start = group * self.samples_per_prompt
            row_stop = row_start + self.samples_per_prompt
            token_start = maximum - length
            prompt_batch[row_start:row_stop, token_start:].copy_(
                prompt.to(device)[None].expand(self.samples_per_prompt, -1)
            )
            attention_mask[row_start:row_stop, token_start:maximum] = True
            position_ids[row_start:row_stop, token_start:].copy_(
                torch.arange(length, device=device)[None].expand(
                    self.samples_per_prompt, -1
                )
            )
        repeated_lengths = torch.tensor(
            lengths, dtype=torch.long, device=device
        ).repeat_interleave(self.samples_per_prompt)
        self.prompt_lengths_buffer.copy_(repeated_lengths)
        return prompt_batch, position_ids, maximum

    @torch.inference_mode()
    def generate_prompts(
        self,
        prompt_ids_cpu: Sequence[Tensor],
        *,
        max_new_tokens: int,
        progress_callback: (
            Callable[[int, dict[str, float | int]], None] | None
        ) = None,
    ) -> tuple[Tensor, Tensor, Tensor, int, float, DecodeStats]:
        """Generate every prompt/sample row in one synchronized GPU batch."""
        started = time.perf_counter()
        device = next(self.policy.parameters()).device
        prompt_batch, prompt_position_ids, prompt_width = self._prepare_prompts(
            prompt_ids_cpu
        )
        gpu_started = torch.cuda.Event(enable_timing=True)
        prefill_complete = torch.cuda.Event(enable_timing=True)
        decode_complete = torch.cuda.Event(enable_timing=True)
        gpu_started.record()
        if prompt_width + max_new_tokens > self.cache_length:
            raise ValueError("prompt and response exceed the rollout cache")
        validate_thinking_budget(
            self.answer_reserve_tokens, self.thinking_end_token_id, max_new_tokens
        )
        cache = self.cache_pool.acquire(self.batch_size)
        generated = self.generated_buffer[:, :max_new_tokens]
        logprobs = self.logprob_buffer[:, :max_new_tokens]
        values = self.value_buffer[:, :max_new_tokens]
        scanned_vocabulary = 0
        nucleus_mass_lower_bound = 1.0
        generated_count = 0
        next_progress = 256
        target_decode_calls = 0
        target_decode_positions = 0
        thinking_closed = torch.zeros(self.batch_size, dtype=torch.bool, device=device)
        thinking_boundary = max_new_tokens - self.answer_reserve_tokens - 1

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = self.policy.cached_hidden(
                prompt_batch,
                past_key_values=cache,
                cache_position=self.cache_positions[:prompt_width],
                attention_mask=self.attention_mask_buffer,
                position_ids=prompt_position_ids,
            )[:, -1]
            logits = self.policy.logits(hidden)
            state_values = self.policy.rollout_values(hidden)
            finished = torch.zeros(
                self.batch_size, dtype=torch.bool, device=device
            )
            prefill_complete.record()

            def sample_target(
                target_logits: Tensor, target_values: Tensor
            ) -> Tensor:
                nonlocal generated_count
                nonlocal scanned_vocabulary
                nonlocal nucleus_mass_lower_bound
                if self.top_k:
                    sampled, sampled_logprobs = self.sample_bounded(target_logits)
                    sampling_support = target_logits.shape[1]
                    sampling_mass = self.top_p
                else:
                    sampled, sampled_logprobs, sampling = exact_top_p_sample(
                        target_logits,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    sampling_support = sampling.scanned_vocabulary
                    sampling_mass = sampling.nucleus_mass_lower_bound
                active = ~finished
                if self.answer_reserve_tokens:
                    sampled, forced = force_thinking_end_(
                        sampled,
                        generated_count,
                        thinking_closed,
                        active,
                        thinking_boundary,
                        self.thinking_end_token_id,
                    )
                    sampled_logprobs = sampled_logprobs.masked_fill(forced, 0.0)
                actual = torch.where(active, sampled, self.primary_stop_tensor)
                selected_logprobs = torch.where(
                    active, sampled_logprobs, torch.zeros_like(sampled_logprobs)
                )
                generated[:, generated_count].copy_(actual)
                logprobs[:, generated_count].copy_(selected_logprobs)
                values[:, generated_count].copy_(target_values)
                scanned_vocabulary = max(scanned_vocabulary, sampling_support)
                nucleus_mass_lower_bound = min(
                    nucleus_mass_lower_bound, sampling_mass
                )
                finished.logical_or_(
                    (actual[:, None] == self.stop_tensor[None]).any(dim=1)
                )
                generated_count += 1
                return actual

            def report_progress(force: bool = False) -> None:
                nonlocal next_progress
                crossed = generated_count >= next_progress
                while generated_count >= next_progress:
                    next_progress += 256
                if progress_callback is None or not (crossed or force):
                    return
                wall_seconds = time.perf_counter() - started
                scheduled_tokens = generated_count * self.batch_size
                progress_callback(
                    generated_count,
                    {
                        "decode_steps": generated_count,
                        "batch_rows": self.batch_size,
                        "prompt_groups": self.prompts_per_rollout,
                        "scheduled_tokens": scheduled_tokens,
                        "wall_seconds": wall_seconds,
                        "scheduled_tokens_per_second": scheduled_tokens
                        / max(wall_seconds, 1e-9),
                        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
                    },
                )

            while generated_count < max_new_tokens:
                response_offset = generated_count
                token = sample_target(logits, state_values)
                all_finished = (
                    generated_count % 64 == 0 and bool(finished.all())
                )
                if generated_count == max_new_tokens or all_finished:
                    report_progress(force=True)
                    break

                physical_position = prompt_width + response_offset
                self.attention_mask_buffer[:, physical_position] = True
                self.position_id_buffer[:, 0].copy_(
                    self.prompt_lengths_buffer + response_offset
                )
                hidden, logits, state_values = self.decode(
                    token,
                    self.cache_positions[
                        physical_position : physical_position + 1
                    ],
                    cache,
                    self.attention_mask_buffer,
                    self.position_id_buffer,
                )
                target_decode_calls += 1
                target_decode_positions += 1
                report_progress()

        decode_complete.record()
        return (
            generated[:, :generated_count].cpu(),
            logprobs[:, :generated_count].cpu(),
            values[:, :generated_count].cpu(),
            scanned_vocabulary,
            nucleus_mass_lower_bound,
            DecodeStats(
                target_decode_calls=target_decode_calls,
                target_decode_positions=target_decode_positions,
                prefill_seconds=gpu_started.elapsed_time(prefill_complete)
                / 1_000.0,
                decode_seconds=prefill_complete.elapsed_time(decode_complete)
                / 1_000.0,
            ),
        )


def _build_group_records(
    tokenizer,
    row: dict,
    prompt_ids: Tensor,
    responses: Tensor,
    logprobs: Tensor,
    values: Tensor,
    *,
    samples_per_prompt: int,
    stop_ids: tuple[int, ...],
    thinking_end_token_id: int | None = None,
    forced_thinking_position: int = -1,
    carry_hiddens: Sequence[Tensor] | Tensor | None = None,
    slot_choices: Sequence[Tensor] | Tensor | None = None,
    slot_count: int = 0,
    outcome_reasons: list[str] | None = None,
) -> tuple[list[TrajectoryRecord], int]:
    """Decode, score, and compact one host-resident prompt group."""
    records = []
    generated_tokens = 0
    prefilled_thinking = (
        tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=False).rstrip().endswith("<think>")
        if is_verifiable_task(row) else False
    )
    for sample in range(samples_per_prompt):
        response_length = _first_stop_length(responses[sample], stop_ids)
        response = responses[sample, :response_length]
        forced_token_index = -1
        if forced_thinking_position >= 0 and response_length > forced_thinking_position:
            closed = (response == thinking_end_token_id).nonzero(as_tuple=False)
            if not closed.numel() or int(closed[0, 0]) > forced_thinking_position:
                raise RuntimeError("rollout exceeded its thinking budget without closing")
            if int(closed[0, 0]) == forced_thinking_position:
                forced_token_index = forced_thinking_position
        text, correct, reason = score_training_response(
            tokenizer, row, prompt_ids, response, prefilled_thinking=prefilled_thinking,
        )
        if outcome_reasons is not None:
            outcome_reasons.append(reason)
        records.append(
            TrajectoryRecord.from_device(
                token_ids=torch.cat((prompt_ids, response)),
                prompt_length=prompt_ids.numel(),
                old_logprobs=logprobs[sample, :response_length],
                old_values=values[sample, :response_length],
                correct=correct,
                text=text,
                forced_token_index=forced_token_index,
                carry_hiddens=(
                    None if carry_hiddens is None else carry_hiddens[sample][:response_length]
                ),
                slot_choices=(
                    None if slot_choices is None else slot_choices[sample][:response_length]
                ),
                slot_count=slot_count,
            )
        )
        generated_tokens += response_length
    return records, generated_tokens


def collect_rollouts(
    engine: RolloutEngine | CapturedTrainingRolloutEngine | NextLatSpeculativeEngine | MiniCPMLatentRolloutEngine,
    tokenizer,
    rows: list[dict],
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    enable_thinking: bool,
    prompt_suffix: str = "",
    context_tokens: int | None = None,
    progress_callback: (
        Callable[[int, dict[str, float | int]], None] | None
    ) = None,
) -> RolloutResult:
    answer_reserve = int(getattr(engine, "answer_reserve_tokens", 0))
    if answer_reserve and not enable_thinking:
        raise ValueError("answer reserve requires a thinking-mode rollout")
    if context_tokens is not None and type(engine) is not CapturedTrainingRolloutEngine:
        raise ValueError("context_tokens requires the native continuous rollout engine")
    started = time.perf_counter()
    encoded_rows = [
        (
            row,
            encode_math_prompt(
                tokenizer,
                row,
                prompt_tokens=prompt_tokens,
                enable_thinking=enable_thinking,
                prompt_suffix=prompt_suffix,
                truncate_prompt=context_tokens is None,
            ),
        )
        for row in rows
    ]
    prompt_limits = response_token_limits(
        [int(prompt.numel()) for _, prompt in encoded_rows],
        max_new_tokens=max_new_tokens, context_tokens=context_tokens,
        answer_reserve_tokens=answer_reserve,
        thinking_end_token_id=getattr(engine, "thinking_end_token_id", None),
    )
    if isinstance(engine, MiniCPMLatentRolloutEngine):
        return collect_latent_rollouts(
            engine, tokenizer, encoded_rows,
            max_new_tokens=max_new_tokens, progress_callback=progress_callback,
        )
    if isinstance(engine, CapturedTrainingRolloutEngine):
        continuous = engine.generate_prompt_pool(
            [prompt_ids for _, prompt_ids in encoded_rows],
            max_new_tokens=max_new_tokens,
            context_tokens=context_tokens,
            progress_callback=progress_callback,
        )
        carry_hiddens = continuous.carry_hiddens
        slot_choices = continuous.slot_choices
        responses = torch.nn.utils.rnn.pad_sequence(
            list(continuous.responses),
            batch_first=True,
            padding_value=engine.stop_ids[0],
        )
        logprobs = torch.nn.utils.rnn.pad_sequence(
            list(continuous.logprobs),
            batch_first=True,
            padding_value=0.0,
        )
        values = torch.zeros_like(logprobs)
        scanned_vocabulary = int(engine.policy.causal_lm.config.vocab_size)
        nucleus_mass_lower_bound = engine.top_p
        uno_metrics = getattr(engine, "last_uno_metrics", {})
        decoding = FastTrainingDecodeStats(
            prefill_seconds=continuous.prefill_seconds,
            decode_seconds=continuous.decode_seconds,
            target_decode_calls=int(
                uno_metrics.get("uno_target_decode_calls", continuous.decode_steps)
            ),
            target_decode_positions=int(
                uno_metrics.get(
                    "uno_target_decode_positions", continuous.capacity_row_steps
                )
            ),
        )
        scheduled_tokens = continuous.capacity_row_steps
        admission_events = continuous.admission_events
        minimum_active_rows_with_backlog = (
            continuous.minimum_active_rows_with_backlog
        )
    else:
        generation = engine.generate_prompts(
            [prompt_ids for _, prompt_ids in encoded_rows],
            max_new_tokens=max_new_tokens,
            progress_callback=progress_callback,
        )
        (
            responses,
            logprobs,
            values,
            scanned_vocabulary,
            nucleus_mass_lower_bound,
            decoding,
        ) = generation
        carry_hiddens = getattr(engine, "last_carry_hiddens", None)
        slot_choices = getattr(engine, "last_slot_choices", None)
        scheduled_tokens = responses.numel()
        admission_events = 1
        minimum_active_rows_with_backlog = engine.batch_size
    if bool(getattr(engine.policy, "token_carry", False)) != (carry_hiddens is not None):
        raise ValueError("rollout token-carry mode differs from stored carry hiddens")
    slot_config = getattr(engine.policy, "slot_memory", None)
    if (slot_config is not None) != (slot_choices is not None):
        raise ValueError("rollout slot-memory mode differs from stored slot choices")
    invalid_tokens = (responses < 0) | (responses >= MINICPM5_VOCAB_SIZE)
    if invalid_tokens.any():
        coordinates = invalid_tokens.nonzero()[:8].tolist()
        details = [
            (row, column, int(responses[row, column]))
            for row, column in coordinates
        ]
        raise RuntimeError(
            f"rollout produced token ids outside the model vocabulary: {details}"
        )

    records: list[TrajectoryRecord] = []
    generated_tokens = 0
    group_outcomes: list[list[str]] = [[] for _ in encoded_rows]
    scoring_started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=len(encoded_rows),
        thread_name_prefix="minicpm-vapo-score",
    ) as scoring_pool:
        pending = []
        for group, (row, prompt_ids) in enumerate(encoded_rows):
            start = group * engine.samples_per_prompt
            stop = start + engine.samples_per_prompt
            pending.append(
                scoring_pool.submit(
                    _build_group_records,
                    tokenizer,
                    row,
                    prompt_ids,
                    responses[start:stop, :prompt_limits[group]],
                    logprobs[start:stop, :prompt_limits[group]],
                    values[start:stop, :prompt_limits[group]],
                    samples_per_prompt=engine.samples_per_prompt,
                    stop_ids=engine.stop_ids,
                    thinking_end_token_id=getattr(engine, "thinking_end_token_id", None),
                    forced_thinking_position=(
                        prompt_limits[group] - answer_reserve - 1 if answer_reserve else -1
                    ),
                    carry_hiddens=None if carry_hiddens is None else carry_hiddens[start:stop],
                    slot_choices=None if slot_choices is None else slot_choices[start:stop],
                    slot_count=0 if slot_config is None else slot_config.slots,
                    outcome_reasons=group_outcomes[group],
                )
            )
        for future in pending:
            group_records, group_tokens = future.result()
            records.extend(group_records)
            generated_tokens += group_tokens
    scoring_seconds = time.perf_counter() - scoring_started
    top_k = int(getattr(engine, "top_k", 0))
    conditional_mass_lower_bound = nucleus_mass_lower_bound
    return RolloutResult(
        records=records,
        generated_tokens=generated_tokens,
        scheduled_tokens=scheduled_tokens,
        elapsed_seconds=time.perf_counter() - started,
        scoring_seconds=scoring_seconds,
        sampling_scanned_vocabulary=scanned_vocabulary,
        sampling_candidate_support=top_k if top_k > 0 else scanned_vocabulary,
        sampling_full_policy_mass_lower_bound=(
            0.0 if top_k > 0 else nucleus_mass_lower_bound
        ),
        sampling_conditional_mass_lower_bound=conditional_mass_lower_bound,
        admission_events=admission_events,
        minimum_active_rows_with_backlog=minimum_active_rows_with_backlog,
        decoding=decoding,
        uno_metrics=getattr(engine, "last_uno_metrics", None),
        group_domains=tuple(training_row_domain(row) for row in rows),
        outcome_reasons=tuple(reason for group in group_outcomes for reason in group),
        response_limits=tuple(
            limit for limit in prompt_limits for _ in range(engine.samples_per_prompt)
        ),
    )


def collect_latent_rollouts(
    engine: MiniCPMLatentRolloutEngine,
    tokenizer,
    encoded_rows: list[tuple[dict, Tensor]],
    *,
    max_new_tokens: int,
    progress_callback=None,
) -> RolloutResult:
    """Score visible answers while retaining every continuous critic transition."""
    from postraining.minicpm_vapo import (
        FIRST_THOUGHT, CONTINUE_THOUGHT, FORCED_STOP_THINKING,
    )

    started = time.perf_counter()
    for _, prompt_ids in encoded_rows:
        starts = (prompt_ids == engine.thinking_start_token_id).nonzero().flatten()
        if not starts.numel():
            raise ValueError("latent prompt must end in the native thinking prefix")
        suffix = tokenizer.decode(prompt_ids[int(starts[-1]) + 1:].tolist())
        if suffix.strip():
            raise ValueError("latent prompt has content after its thinking prefix")
    result = engine.generate_prompt_pool(
        [prompt for _, prompt in encoded_rows],
        max_new_tokens=max_new_tokens,
        progress_callback=progress_callback,
    )
    records = []
    outcome_reasons = []
    for index, (response, kinds, vectors, logprobs) in enumerate(zip(
        result.responses, result.action_kinds, result.latent_vectors,
        result.logprobs, strict=True,
    )):
        row, prompt = encoded_rows[index // engine.samples_per_prompt]
        lexical = (kinds != FIRST_THOUGHT) & (kinds != CONTINUE_THOUGHT)
        text, correct, reason = score_training_response(
            tokenizer, row, prompt, response[lexical],
            prefilled_thinking=True,
        )
        outcome_reasons.append(reason)
        forced = (kinds == FORCED_STOP_THINKING).nonzero().flatten()
        records.append(TrajectoryRecord.from_device(
            token_ids=torch.cat((prompt, response)),
            prompt_length=prompt.numel(),
            old_logprobs=logprobs,
            old_values=torch.zeros_like(logprobs),
            correct=correct,
            text=text,
            forced_token_index=int(forced[0]) if forced.numel() else -1,
            action_kinds=kinds,
            latent_vectors=vectors,
            controller_observations=result.controller_observations[index],
        ))
    return RolloutResult(
        records=records,
        generated_tokens=sum(record.response_length for record in records),
        scheduled_tokens=result.capacity_row_steps,
        elapsed_seconds=time.perf_counter() - started,
        sampling_scanned_vocabulary=MINICPM5_VOCAB_SIZE,
        sampling_candidate_support=engine.top_k,
        sampling_full_policy_mass_lower_bound=0.0,
        sampling_conditional_mass_lower_bound=engine.top_p,
        admission_events=result.admission_events,
        minimum_active_rows_with_backlog=result.minimum_active_rows_with_backlog,
        decoding=FastTrainingDecodeStats(
            prefill_seconds=result.prefill_seconds,
            decode_seconds=result.decode_seconds,
            target_decode_calls=result.decode_steps,
            target_decode_positions=result.capacity_row_steps,
        ),
        group_domains=tuple(training_row_domain(row) for row, _ in encoded_rows),
        outcome_reasons=tuple(outcome_reasons),
    )


def rollout_diagnostics(
    result: RolloutResult,
    *,
    samples_per_prompt: int,
    stop_ids: tuple[int, ...],
) -> dict[str, float | int]:
    records = result.records
    if not records or len(records) % samples_per_prompt:
        raise ValueError("rollout records must contain complete prompt groups")
    correct = torch.tensor([record.correct for record in records], dtype=torch.bool)
    grouped = correct.view(-1, samples_per_prompt)
    lengths = torch.tensor([record.response_length for record in records])
    terminated = torch.tensor(
        [int(record.token_ids[-1]) in stop_ids for record in records], dtype=torch.float32
    )
    capped = 1 - terminated
    if result.response_limits:
        if len(result.response_limits) != len(records):
            raise ValueError("response limits must follow every trajectory")
        limits = torch.tensor(result.response_limits)
        if bool((lengths > limits).any()) or bool(((capped > 0) & (lengths != limits)).any()):
            raise ValueError("nonterminal rollout must end at its own response limit")
    repetition = [repetition_metrics(record.text) for record in records]
    metrics: dict[str, float | int] = {
        "trajectories": len(records),
        "prompt_groups": grouped.shape[0],
        "positive_trajectories": int(correct.sum()),
        "positive_groups": int(grouped.any(dim=1).sum()),
        "mixed_groups": int((grouped.any(dim=1) & ~grouped.all(dim=1)).sum()),
        "accuracy": float(correct.float().mean()),
        "truncation_fraction": float(capped.mean()),
        "forced_thinking_trajectories": sum(
            record.forced_token_index >= 0 for record in records
        ),
        "response_length_mean": float(lengths.float().mean()),
        "response_length_p95": float(torch.quantile(lengths.float(), 0.95)),
        "response_length_max": int(lengths.max()),
        "generated_tokens": result.generated_tokens,
        "rollout_seconds": result.elapsed_seconds,
        "rollout_scoring_seconds": result.scoring_seconds,
        "rollout_tokens_per_second": result.generated_tokens
        / max(result.elapsed_seconds, 1e-9),
        "scheduled_rollout_tokens_per_second": result.scheduled_tokens
        / max(result.elapsed_seconds, 1e-9),
        "productive_utilization": result.generated_tokens
        / max(result.scheduled_tokens, 1),
        "admission_events": result.admission_events,
        "minimum_active_rows_with_backlog": (
            result.minimum_active_rows_with_backlog
        ),
        "replay_storage_bytes": replay_storage_bytes(records),
        "replay_bytes_per_token": replay_storage_bytes(records)
        / max(result.generated_tokens, 1),
        "sampling_scanned_vocabulary": result.sampling_scanned_vocabulary,
        "sampling_candidate_support": result.sampling_candidate_support,
        "sampling_full_policy_mass_lower_bound": (
            result.sampling_full_policy_mass_lower_bound
        ),
        "sampling_conditional_mass_lower_bound": (
            result.sampling_conditional_mass_lower_bound
        ),
        "prefill_seconds": result.decoding.prefill_seconds,
        "decode_seconds": result.decoding.decode_seconds,
        "decode_tokens_per_second": result.generated_tokens
        / max(result.decoding.decode_seconds, 1e-9),
        "target_decode_calls": result.decoding.target_decode_calls,
        "target_decode_positions": result.decoding.target_decode_positions,
        "scheduled_decode_tokens_per_second": (
            result.decoding.target_decode_positions
            / max(result.decoding.decode_seconds, 1e-9)
        ),
        "target_positions_per_decode_call": (
            result.decoding.target_decode_positions
            / max(result.decoding.target_decode_calls, 1)
        ),
    }
    for name in repetition[0]:
        metrics[f"{name}_mean"] = sum(item[name] for item in repetition) / len(repetition)
        metrics[f"{name}_max"] = max(item[name] for item in repetition)
    if result.response_limits:
        metrics["response_limit_mean"] = float(limits.float().mean())
        metrics["response_limit_min"] = int(limits.min())
        metrics["response_limit_max"] = int(limits.max())
    if result.group_domains:
        if len(result.group_domains) != grouped.shape[0]:
            raise ValueError("domain metadata must follow complete prompt-major groups")
        if len(result.outcome_reasons) != len(records):
            raise ValueError("outcome metadata must follow every prompt-major trajectory")
        for domain in sorted(set(result.group_domains)):
            groups = [
                index for index, value in enumerate(result.group_domains) if value == domain
            ]
            indices = [
                group * samples_per_prompt + sample
                for group in groups for sample in range(samples_per_prompt)
            ]
            domain_correct = correct[indices]
            domain_grouped = grouped[groups]
            domain_lengths = lengths[indices].float()
            domain_metrics = {
                "attempts": len(indices),
                "correct": int(domain_correct.sum()),
                "accuracy": float(domain_correct.float().mean()),
                "prompt_groups": len(groups),
                "positive_groups": int(domain_grouped.any(dim=1).sum()),
                "mixed_groups": int((
                    domain_grouped.any(dim=1) & ~domain_grouped.all(dim=1)
                ).sum()),
                "response_length_mean": float(domain_lengths.mean()),
                "response_length_p95": float(torch.quantile(domain_lengths, 0.95)),
                "response_length_max": int(domain_lengths.max()),
                "capped_trajectories": int(capped[indices].sum()),
                "truncation_fraction": float(capped[indices].mean()),
                "forced_thinking_trajectories": sum(
                    records[index].forced_token_index >= 0 for index in indices
                ),
            }
            for name in repetition[0]:
                domain_metrics[f"{name}_mean"] = (
                    sum(repetition[index][name] for index in indices) / len(indices)
                )
                domain_metrics[f"{name}_max"] = max(
                    repetition[index][name] for index in indices
                )
            if result.response_limits:
                domain_metrics["response_limit_mean"] = float(limits[indices].float().mean())
                domain_metrics["response_limit_min"] = int(limits[indices].min())
                domain_metrics["response_limit_max"] = int(limits[indices].max())
            domain_metrics.update({
                f"outcome/{reason}": count for reason, count in Counter(
                    result.outcome_reasons[index] for index in indices
                ).items()
            })
            metrics.update({
                f"domain/{domain}/{name}": value for name, value in domain_metrics.items()
            })
    latent_records = [record for record in records if record.action_kinds is not None]
    if latent_records:
        from postraining.minicpm_vapo import TOKEN_ACTION, STOP_THINKING

        thought_steps = sum(record.latent_vectors.shape[0] for record in latent_records)
        answer_tokens = sum(
            int((record.action_kinds == TOKEN_ACTION).sum()) for record in latent_records
        )
        metrics.update({
            "latent_thought_steps": thought_steps,
            "latent_thought_steps_mean": thought_steps / len(latent_records),
            "latent_answer_tokens": answer_tokens,
            "latent_learned_stops": sum(
                int((record.action_kinds == STOP_THINKING).sum())
                for record in latent_records
            ),
        })
    proposed = int(getattr(result.decoding, "proposed_tokens", 0))
    accepted = int(getattr(result.decoding, "accepted_tokens", 0))
    proposed_pos2 = int(
        getattr(result.decoding, "proposed_tokens_pos2plus", 0)
    )
    accepted_pos2 = int(
        getattr(result.decoding, "accepted_tokens_pos2plus", 0)
    )
    row_cycles = int(
        getattr(result.decoding, "speculative_row_cycles", 0)
    )
    if proposed:
        metrics["nextlat_acceptance"] = accepted / proposed
    if proposed_pos2:
        metrics["nextlat_acceptance_pos2plus"] = (
            accepted_pos2 / proposed_pos2
        )
    if row_cycles:
        metrics["nextlat_accepted_pos2plus_per_row_cycle"] = (
            accepted_pos2 / row_cycles
        )
    proposed_by_position = getattr(
        result.decoding, "proposed_by_position", ()
    )
    accepted_by_position = getattr(
        result.decoding, "accepted_by_position", ()
    )
    for position, (position_proposed, position_accepted) in enumerate(
        zip(proposed_by_position, accepted_by_position, strict=True),
        start=1,
    ):
        if position_proposed:
            metrics[f"nextlat_acceptance_position_{position}"] = (
                position_accepted / position_proposed
            )
    if result.uno_metrics:
        metrics.update(result.uno_metrics)
    return metrics


def device_phase_metrics(
    sampler: DeviceSampler | None,
    *,
    started: float,
    ended: float,
    power_floor: float,
    prefix: str = "",
) -> dict[str, float | int]:
    """Attach trustworthy NVML samples formed entirely inside one phase."""
    if sampler is None:
        return {}
    stats = sampler.window([(started, ended)], power_floor)
    return {
        f"{prefix}device_{name}": value
        for name, value in stats.items()
        if isinstance(value, (int, float))
    }


class _ChunkedNextLatKL(torch.autograd.Function):
    """Exact teacher-to-student KL with recomputed vocabulary projections."""

    @staticmethod
    def forward(
        ctx: Any,
        predicted: Tensor,
        target: Tensor,
        weight: Tensor,
        chunk_tokens: int,
    ) -> Tensor:
        if predicted.shape != target.shape or predicted.ndim != 2:
            raise ValueError("NextLat KL states must be matching matrices")
        if chunk_tokens < 1:
            raise ValueError("NextLat KL chunk size must be positive")
        ctx.save_for_backward(predicted, target, weight)
        ctx.chunk_tokens = chunk_tokens
        count = predicted.shape[0]
        total = torch.zeros((), device=predicted.device, dtype=torch.float32)
        for start in range(0, count, chunk_tokens):
            stop = min(start + chunk_tokens, count)
            teacher_logits = F.linear(target[start:stop], weight).float()
            student_logits = F.linear(predicted[start:stop], weight).float()
            teacher_logprobs = teacher_logits.log_softmax(dim=-1)
            teacher_probabilities = teacher_logprobs.exp()
            student_logprobs = student_logits.log_softmax(dim=-1)
            total += (
                teacher_probabilities * (teacher_logprobs - student_logprobs)
            ).sum()
        return total / max(count, 1)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor):
        if len(grad_outputs) != 1:
            raise RuntimeError("NextLat KL backward expects one gradient")
        grad_output = grad_outputs[0]
        predicted, target, weight = ctx.saved_tensors
        count = predicted.shape[0]
        grad_predicted = torch.empty_like(predicted)
        scale = grad_output.float() / max(count, 1)
        for start in range(0, count, ctx.chunk_tokens):
            stop = min(start + ctx.chunk_tokens, count)
            teacher_logits = F.linear(target[start:stop], weight).float()
            student_logits = F.linear(predicted[start:stop], weight).float()
            grad_logits = student_logits.softmax(dim=-1)
            grad_logits.sub_(teacher_logits.softmax(dim=-1)).mul_(scale)
            grad_predicted[start:stop] = F.linear(
                grad_logits.to(weight.dtype), weight.transpose(0, 1)
            ).to(predicted.dtype)
        return grad_predicted, None, None, None
def _nextlat_training_loss(
    side: MiniCPMVAPOPolicy | MiniCPMVAPOCritic,
    hidden: Tensor,
    batch,
    *,
    max_samples: int,
    horizon: int,
    mse_coefficient: float,
    kl_coefficient: float,
    kl_chunk_tokens: int,
) -> NextLatTrainingLoss:
    """Reference recursive NextLat objective on an unbiased state sample."""
    if max_samples < 1 or horizon < 1:
        raise ValueError("NextLat sample budget and horizon must be positive")
    if hidden.shape[0] != 1:
        raise ValueError("NextLat replay requires one packed sequence row")
    starts, ends = [], []
    total = 0
    for start, stop in batch.nextlat_sequence_ranges:
        capacity = max(stop - start - horizon, 0)
        if capacity:
            starts.append(start - total)
            total += capacity
            ends.append(total)
    if not total:
        zero = hidden.float().sum() * 0.0
        return NextLatTrainingLoss(zero, zero, zero, 0, 0)
    sample_indices = torch.randint(total, (max_samples,), device=hidden.device)
    if len(starts) == 1:
        columns = sample_indices + starts[0]
    else:
        # CPU collation supplies contiguous per-trajectory ranges. Sample their
        # concatenation without a GPU nonzero allocation or host synchronization.
        layout = torch.tensor((starts, ends), dtype=torch.long, device=hidden.device)
        sequences = torch.bucketize(sample_indices, layout[1], right=True)
        columns = sample_indices + layout[0].gather(0, sequences)
    predicted = hidden[0, columns]
    smooth_l1 = torch.zeros((), device=hidden.device, dtype=torch.float32)
    categorical_kl = torch.zeros_like(smooth_l1)
    for offset in range(1, horizon + 1):
        next_tokens = batch.input_ids[0, columns + offset]
        embeddings = side.token_embeddings(next_tokens).detach()
        predicted = side.nextlat_head(predicted, embeddings)
        target = hidden[0, columns + offset].detach()
        smooth_l1 += F.smooth_l1_loss(
            predicted.float(), target.float(), reduction="mean"
        )
        categorical_kl += _ChunkedNextLatKL.apply(
            predicted,
            target,
            side.lm_head_weight.detach(),
            kl_chunk_tokens,
        )
    smooth_l1 /= horizon
    categorical_kl /= horizon
    loss = mse_coefficient * smooth_l1 + kl_coefficient * categorical_kl
    return NextLatTrainingLoss(
        loss=loss,
        smooth_l1=smooth_l1,
        categorical_kl=categorical_kl,
        samples=max_samples,
        transitions=max_samples * horizon,
    )


def _optimizer_minibatches(
    record_count: int, minibatch_count: int
) -> list[tuple[int, ...]]:
    if minibatch_count < 1:
        raise ValueError("optimizer minibatches must be positive")
    if record_count < minibatch_count:
        raise ValueError("optimizer minibatches cannot exceed trajectories")
    order = torch.randperm(record_count).tolist()
    minibatches = [
        tuple(order[offset::minibatch_count])
        for offset in range(minibatch_count)
    ]
    if any(not indices for indices in minibatches):
        raise RuntimeError("optimizer minibatch partition produced an empty batch")
    flattened = [index for indices in minibatches for index in indices]
    if sorted(flattened) != list(range(record_count)):
        raise RuntimeError("optimizer minibatches lost or duplicated trajectories")
    return minibatches


def _nextlat_shard_samples(
    capacities: Sequence[int], max_samples: int
) -> tuple[int, ...]:
    """Choose one capacity-weighted shard for an unbiased, GPU-dense sample."""
    if max_samples < 1 or any(capacity < 0 for capacity in capacities):
        raise ValueError("NextLat shard budgets require nonnegative capacities")
    weights = torch.tensor(capacities, dtype=torch.float64)
    if not bool(weights.sum()):
        return tuple(0 for _ in capacities)
    selected = int(torch.multinomial(weights, 1).item())
    return tuple(max_samples if index == selected else 0 for index in range(len(capacities)))


def _nextlat_record_capacity(record: TrajectoryRecord, horizon: int) -> int:
    states = record.response_length
    if record.latent_vectors is not None:
        states -= record.latent_vectors.shape[0] + 1
    return max(states - horizon, 0)


def _replay_hidden(
    side: MiniCPMVAPOPolicy | MiniCPMVAPOCritic,
    batch: ReplayMicrobatch,
) -> Tensor:
    if bool(getattr(side, "latent_thinking", False)) != (
        getattr(batch, "action_kinds", None) is not None
    ):
        raise ValueError("replay reasoning mode differs from actor or critic")
    if bool(getattr(side, "token_carry", False)) != (batch.carry_hiddens is not None):
        raise ValueError("replay token-carry mode differs from stored carry hiddens")
    if (getattr(side, "slot_memory", None) is not None) != (batch.slot_alive_table is not None):
        raise ValueError("replay slot-memory mode differs from stored slot choices")
    if getattr(side, "token_carry", False):
        return side.token_carry_replay_hidden(batch)
    latent_inputs = {}
    if getattr(batch, "latent_vectors", None) is not None:
        latent_inputs = {
            "latent_vectors": batch.latent_vectors,
            "latent_input_positions": batch.latent_input_positions,
        }
    return side.replay_hidden(
        batch.input_ids,
        batch.attention_mask,
        position_ids=batch.position_ids,
        cu_seqlens=batch.cu_seqlens,
        sequence_boundaries=batch.sequence_boundaries,
        max_sequence_length=batch.max_sequence_length,
        **latent_inputs,
    )


def _packed_action_rows(hidden: Tensor, batch: ReplayMicrobatch) -> Tensor:
    """Gather action-position rows from the single packed replay row.

    Values match advanced indexing exactly; the backward is one ``index_add``
    over unique positions (also exact) instead of the sort-based accumulate
    path that advanced indexing schedules.
    """
    if hidden.size(0) != 1:
        raise ValueError("packed replay hidden states must be a single row")
    return hidden[0].index_select(0, batch.action_positions)


def _action_logprobs(policy, hidden: Tensor, batch, *, chunk_tokens: int) -> Tensor:
    if getattr(batch, "action_kinds", None) is not None:
        return policy.action_logprobs(hidden, batch, chunk_tokens=chunk_tokens)
    logprobs = chunked_frozen_head_logprobs(
        hidden, batch.targets, policy.lm_head_weight, chunk_tokens=chunk_tokens,
    )
    slot_actions = getattr(batch, "slot_actions", None)
    if (getattr(policy, "slot_memory", None) is not None) != (slot_actions is not None):
        raise ValueError("slot-memory likelihood requires recorded slot choices")
    if slot_actions is not None:
        # Joint action: log pi(x_t, sigma_t | m_t) = log pi_tok + log pi_slot.
        # The terminal slot choice is never read and carries no credit.
        slot_logprobs = policy.slot_head.log_prob(hidden, slot_actions)
        logprobs = logprobs + torch.where(
            batch.slot_action_mask, slot_logprobs, torch.zeros_like(slot_logprobs)
        )
    return logprobs


@torch.no_grad()
def _carry_scale_statistics(scale: Tensor) -> dict[str, Tensor]:
    return {
        "scale_mean": scale.mean(),
        "scale_min": scale.amin(),
        "scale_max": scale.amax(),
        "scale_rms": scale.square().mean().sqrt(),
    }


@torch.no_grad()
def _slot_read_probe(side, batch: ReplayMicrobatch, indices: Tensor, embeddings: Tensor) -> dict[str, float]:
    """Dense-read probe of sampled query rows against their full recorded key set."""
    combiner = side.token_combiner
    carries = batch.carry_hiddens
    with torch.autocast(device_type=embeddings.device.type, dtype=torch.bfloat16):
        queries = combiner.queries(embeddings, carries[indices], batch.slot_positions[indices])
        keys, values = combiner.keys_values(carries, batch.slot_positions)
        visible = visibility_reference(batch.slot_alive_table[indices], batch.slot_key_choices)
        read = combiner.read_dense(queries, keys, values, visible)
        scores = torch.einsum("nhd,khd->nhk", queries.float(), keys.float()) * combiner.score_scale
        scores = scores.masked_fill(~visible[:, None, :], float("-inf"))
        full = torch.cat((combiner.null_logits(queries)[..., None], scores), dim=-1).softmax(dim=-1)
        token_delta = combiner.token_delta(embeddings)
        read_delta = F.linear(read.to(embeddings.dtype), combiner.output.weight.to(embeddings.dtype))
        residual = token_delta + read_delta
    scale = combiner.scale
    ages = (batch.slot_positions[indices][:, None] - batch.slot_positions[None]).float().clamp_min(0)
    key_mass = full[..., 1:]
    token_rms = embeddings.float().square().mean().sqrt()
    scaled_read = (read_delta.float() * scale).to(embeddings.dtype).float()
    scaled_residual = (residual.float() * scale).to(embeddings.dtype).float()
    statistics = _carry_scale_statistics(scale) | {
        "probe_token_rms": token_rms,
        "probe_hidden_rms": carries[indices].float().square().mean().sqrt(),
        "probe_scaled_read_rms": scaled_read.square().mean().sqrt(),
        "probe_read_to_token_rms": scaled_read.square().mean().sqrt() / token_rms.clamp_min(1e-12),
        "probe_residual_to_token_rms": scaled_residual.square().mean().sqrt() / token_rms.clamp_min(1e-12),
        "probe_null_mass": full[..., 0].mean(),
        "probe_visible_keys_mean": visible.sum(dim=1).float().mean(),
        "probe_read_age_mean": (key_mass * ages[:, None, :]).sum(dim=-1).mean(),
    }
    values_list = torch.stack(tuple(statistics.values())).cpu().tolist()
    return dict(zip(statistics, values_list, strict=True))


@torch.no_grad()
def _carry_input_probe(side, batch: ReplayMicrobatch) -> dict[str, float]:
    """Probe at most 256 evenly spaced carry inputs in one packed shard."""
    positions = batch.carry_input_positions
    if not getattr(side, "token_carry", False) or positions is None or not positions.numel():
        return {}
    indices = torch.linspace(
        0, positions.numel() - 1, min(256, positions.numel()),
        device=positions.device,
    ).long()
    embeddings = side.token_embeddings(batch.input_ids[0, positions[indices]])
    carries = batch.carry_hiddens[indices]
    combiner = side.token_combiner
    if getattr(side, "slot_memory", None) is not None:
        return _slot_read_probe(side, batch, indices, embeddings)
    with torch.autocast(device_type=embeddings.device.type, dtype=torch.bfloat16):
        token_delta = combiner.token_delta(embeddings)
        carry_delta = combiner.carry(carries.to(embeddings.dtype))
        residual = token_delta + carry_delta
    scale = combiner.scale
    token_rms = embeddings.float().square().mean().sqrt()
    scaled_carry = (carry_delta.float() * scale).to(embeddings.dtype).float()
    scaled_residual = (residual.float() * scale).to(embeddings.dtype).float()
    carry_rms = scaled_carry.square().mean().sqrt()
    residual_rms = scaled_residual.square().mean().sqrt()
    statistics = _carry_scale_statistics(scale) | {
        "probe_token_rms": token_rms,
        "probe_hidden_rms": carries.float().square().mean().sqrt(),
        "probe_scaled_carry_rms": carry_rms,
        "probe_carry_to_token_rms": carry_rms / token_rms.clamp_min(1e-12),
        "probe_residual_to_token_rms": residual_rms / token_rms.clamp_min(1e-12),
    }
    values = torch.stack(tuple(statistics.values())).cpu().tolist()
    return dict(zip(statistics, values, strict=True))


@torch.no_grad()
def refresh_behavior_statistics(
    policy: MiniCPMVAPOPolicy,
    critic: MiniCPMVAPOCritic,
    records: list[TrajectoryRecord],
    *,
    replay_token_budget: int,
    replay_max_trajectories: int,
    logit_chunk_tokens: int,
    behavior_logprobs: str = "replay",
) -> tuple[list[TrajectoryRecord], dict[str, float | int]]:
    """Materialize replay-consistent behavior log-probabilities and values.

    ``behavior_logprobs="rollout"`` keeps the exact log-probabilities the
    rollout engine captured and replays only the independent critic.
    """
    if behavior_logprobs not in {"replay", "rollout"}:
        raise ValueError("behavior_logprobs must be replay or rollout")
    replay_actor = behavior_logprobs == "replay"
    if not records:
        raise ValueError("cannot refresh an empty rollout")
    started = time.perf_counter()
    device = next(policy.parameters()).device
    policy.eval()
    critic.eval()
    pad_token_id = int(getattr(policy.causal_lm.config, "pad_token_id"))
    plan = plan_replay_microbatches(
        records,
        list(range(len(records))),
        token_budget=replay_token_budget,
        max_trajectories=replay_max_trajectories,
    )
    offsets = [0]
    for record in records:
        offsets.append(offsets[-1] + record.response_length)
    # Retain only two scalar statistics per action; transfer once, not twice per
    # trajectory. This avoids serializing the GPU queue at each shard boundary.
    statistics = torch.empty((2, offsets[-1]), device=device, dtype=torch.float32)
    carry_metrics: dict[str, float] = {}
    for indices in plan:
        batch = collate_replay_microbatch(
            records, indices, pad_token_id=pad_token_id, device=device,
        )
        if replay_actor:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                actor_hidden = _replay_hidden(policy, batch)
                actor_actions = _packed_action_rows(actor_hidden, batch)
                old_logprobs = _action_logprobs(
                    policy, actor_actions, batch, chunk_tokens=logit_chunk_tokens,
                )
            del actor_hidden, actor_actions
        else:
            old_logprobs = torch.cat(
                [records[record_index].old_logprobs for record_index in indices]
            ).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            critic_hidden = _replay_hidden(critic, batch)
            critic_actions = _packed_action_rows(critic_hidden, batch)
            old_values = critic.values(critic_actions).float()
        cursor = 0
        for record_index in indices:
            length = records[record_index].response_length
            destination = slice(offsets[record_index], offsets[record_index + 1])
            statistics[0, destination].copy_(old_logprobs[cursor : cursor + length])
            statistics[1, destination].copy_(old_values[cursor : cursor + length])
            cursor += length
        if not carry_metrics and getattr(policy, "token_carry", False):
            for name, side in (("actor", policy), ("critic", critic)):
                carry_metrics.update({
                    f"carry_{name}_{key}": value
                    for key, value in _carry_input_probe(side, batch).items()
                })
        del batch, critic_hidden, critic_actions, old_logprobs, old_values
    statistics = statistics.cpu()
    refreshed = []
    for index, record in enumerate(records):
        old_logprobs = statistics[0, offsets[index] : offsets[index + 1]]
        old_values = statistics[1, offsets[index] : offsets[index + 1]]
        refreshed.append(
            replace(
                record,
                old_logprobs=old_logprobs,
                advantages=_precompute_advantages(old_values, record.correct),
            )
        )
    if getattr(policy, "slot_memory", None) is not None:
        carry_metrics.update({
            f"slot_{key}": value
            for key, value in slot_choice_statistics(
                [record.slot_choices for record in records if record.slot_choices is not None],
                policy.slot_memory.slots,
            ).items()
        })
    return refreshed, {
        "behavior_refresh_seconds": time.perf_counter() - started,
        "behavior_refresh_microbatches": len(plan),
        **carry_metrics,
    }


@torch.no_grad()
def measure_post_update_behavior_kl(
    policy: MiniCPMVAPOPolicy,
    records: list[TrajectoryRecord],
    *,
    replay_token_budget: int,
    replay_max_trajectories: int,
    logit_chunk_tokens: int,
) -> dict[str, float]:
    started = time.perf_counter()
    device = next(policy.parameters()).device
    policy.eval()
    pad_token_id = int(getattr(policy.causal_lm.config, "pad_token_id"))
    plan = plan_replay_microbatches(
        records,
        list(range(len(records))),
        token_budget=replay_token_budget,
        max_trajectories=replay_max_trajectories,
    )
    approximate_kl = torch.zeros((), device=device, dtype=torch.float64)
    sampled_forward_kl = torch.zeros_like(approximate_kl)
    ratio_abs_log_max = 0.0
    actions = 0
    for indices in plan:
        batch = collate_replay_microbatch(
            records, indices, pad_token_id=pad_token_id, device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = _replay_hidden(policy, batch)
            action_hidden = _packed_action_rows(hidden, batch)
            logprobs = _action_logprobs(
                policy, action_hidden, batch, chunk_tokens=logit_chunk_tokens,
            )
        log_ratio = torch.where(
            batch.policy_mask, logprobs - batch.old_logprobs, 0.0
        )
        approximate_kl += _approximate_kl_terms(log_ratio).double().sum()
        sampled_forward_kl -= log_ratio.double().sum()
        ratio_abs_log_max = max(
            ratio_abs_log_max, float(log_ratio.abs().max())
        )
        actions += sum(
            records[index].response_length - (records[index].forced_token_index >= 0)
            for index in indices
        )
        del batch, hidden, action_hidden, logprobs, log_ratio
    if actions == 0:
        raise ValueError("behavior KL requires at least one sampled policy action")
    return {
        "post_update_approximate_kl": float(approximate_kl / actions),
        "post_update_sampled_forward_kl": float(
            sampled_forward_kl / actions
        ),
        "post_update_ratio_abs_log_max": ratio_abs_log_max,
        "post_update_kl_seconds": time.perf_counter() - started,
    }





_ROLLOUT_QUALITY_TAGS = {
    name: (
        f"latent_thinking/{name.removeprefix('latent_')}"
        if name.startswith("latent_") else f"rollout_quality/{name}"
    )
    for name in (
        "trajectories",
        "prompt_groups",
        "positive_trajectories",
        "positive_groups",
        "mixed_groups",
        "accuracy",
        "truncation_fraction",
        "forced_thinking_trajectories",
        "latent_thought_steps",
        "latent_thought_steps_mean",
        "latent_answer_tokens",
        "latent_learned_stops",
        "response_length_mean",
        "response_length_p95",
        "response_length_max",
        "passed",
    )
}
_ROLLOUT_PERFORMANCE_TAGS = {
    name: f"rollout_performance/{name}"
    for name in (
        "generated_tokens",
        "rollout_seconds",
        "rollout_scoring_seconds",
        "rollout_tokens_per_second",
        "scheduled_rollout_tokens_per_second",
        "prefill_seconds",
        "decode_seconds",
        "decode_tokens_per_second",
        "target_decode_calls",
        "target_decode_positions",
        "scheduled_decode_tokens_per_second",
        "target_positions_per_decode_call",
    )
}
_ROLLOUT_SAMPLING_TAGS = {
    name: f"rollout_sampling/{name}"
    for name in (
        "sampling_scanned_vocabulary",
        "sampling_candidate_support",
        "sampling_full_policy_mass_lower_bound",
        "sampling_conditional_mass_lower_bound",
    )
}
_ROLLOUT_EFFICIENCY_TAGS = {
    name: f"rollout_efficiency/{name}"
    for name in (
        "productive_utilization",
        "admission_events",
        "minimum_active_rows_with_backlog",
    )
}
_TRAIN_TAGS = {
    "loss": "loss/total",
    "policy_loss": "actor/policy_loss",
    "value_loss": "critic/loss",
    "value_mean": "critic/prediction_mean",
    "value_target_mean": "critic/target_mean",
    "explained_variance": "critic/explained_variance",
    "actor_grad_norm": "grad/actor",
    "critic_grad_norm": "grad/critic",
    "gradient_clip_norm": "grad/clip_threshold",
    "approximate_kl": "kl/token_behavior",
    "sampled_forward_kl": "kl/sampled_forward_behavior",
    "post_update_approximate_kl": "kl/post_update_token_behavior",
    "post_update_sampled_forward_kl": "kl/post_update_sampled_forward",
    "clip_fraction": "clip/policy",
    "ratio_mean": "ratio/mean",
    "ratio_std": "ratio/std",
    "ratio_abs_log_max": "ratio/abs_log_max",
    "post_update_ratio_abs_log_max": "ratio/post_update_abs_log_max",
    "actor_nextlat_loss": "auxiliary_nextlat/actor_loss",
    "actor_nextlat_balanced_loss": "auxiliary_nextlat/actor_balanced_loss",
    "actor_nextlat_smooth_l1": "auxiliary_nextlat/actor_smooth_l1",
    "actor_nextlat_categorical_kl": "auxiliary_nextlat/actor_categorical_kl",
    "actor_nextlat_samples": "auxiliary_nextlat/actor_samples",
    "actor_nextlat_transitions": "auxiliary_nextlat/actor_transitions",
    "critic_nextlat_loss": "auxiliary_nextlat/critic_loss",
    "critic_nextlat_balanced_loss": "auxiliary_nextlat/critic_balanced_loss",
    "critic_nextlat_smooth_l1": "auxiliary_nextlat/critic_smooth_l1",
    "critic_nextlat_categorical_kl": "auxiliary_nextlat/critic_categorical_kl",
    "critic_nextlat_samples": "auxiliary_nextlat/critic_samples",
    "critic_nextlat_transitions": "auxiliary_nextlat/critic_transitions",
    "optimizer_minibatches": "replay/optimizer_minibatches",
    "replay_microbatches": "replay/memory_microbatches",
    "replay_actions": "replay/actions",
    "update_seconds": "optimization/update_seconds",
    "post_update_kl_seconds": "optimization/post_update_kl_seconds",
    "peak_vram_bytes": "system_update/peak_vram_bytes",
    "allocated_vram_bytes": "system_update/allocated_vram_bytes",
    "advantage_mean": "advantage/mean",
    "advantage_std": "advantage/std",
    "advantage_min": "advantage/min",
    "advantage_max": "advantage/max",
}
_LIVE_TAGS = {
    name: f"system_live/{name}"
    for name in (
        "decode_steps",
        "wall_seconds",
        "scheduled_tokens_per_second",
        "peak_vram_bytes",
        "device_utilization_gpu_percent",
        "device_power_draw_watts",
        "device_clocks_sm_mhz",
        "device_memory_used_mib",
        "completed_rows",
        "active_rows",
        "pending_rows",
        "useful_completed_tokens",
    )
}


def _organized_tensorboard_tag(namespace: str, name: str) -> str | None:
    if name.startswith(("repetition_", "response_limit_")) and namespace in {
        "rollout", "rollout_gate", "value_warmup",
    }:
        phase = "_warmup" if namespace == "value_warmup" else ""
        return f"rollout_quality{phase}/{name}"
    if name.startswith("corpus_"):
        return f"corpus/{name.removeprefix('corpus_')}"
    if name.startswith("domain/") and namespace in {
        "rollout", "rollout_gate", "value_warmup",
    }:
        phase = "_warmup" if namespace == "value_warmup" else ""
        return f"rollout_domains{phase}/{name.removeprefix('domain/')}"
    if namespace == "corpus":
        return f"corpus/{name}"
    if name.startswith("uno_") and namespace in {
        "rollout", "rollout_gate", "value_warmup", "rollout_live", "value_warmup_live"
    }:
        suffix = "_warmup" if namespace.startswith("value_warmup") else ""
        metric = name.removeprefix("uno_")
        statistic, separator, position = metric.rpartition("_position_")
        if separator:
            return f"uno_position_{position}{suffix}/{statistic}"
        return f"uno_performance{suffix}/{metric}"
    if namespace == "train":
        if name.startswith("carry_"):
            return f"carry/{name.removeprefix('carry_')}"
        if name in _TRAIN_TAGS:
            return _TRAIN_TAGS[name]
        if name.startswith("device_") and not name.endswith(
            ("_period_ms", "_readings")
        ):
            return f"system_update/{name.removeprefix('device_')}"
        return None
    if namespace == "behavior":
        if name.startswith("carry_"):
            return f"carry/{name.removeprefix('carry_')}"
        return {
            "behavior_refresh_seconds": "optimization/behavior_refresh_seconds",
            "behavior_refresh_microbatches": "replay/behavior_refresh_microbatches",
            "cache_release_allocated_bytes": (
                "system_behavior/cache_release_allocated_bytes"
            ),
        }.get(name)
    if namespace.endswith("_live"):
        return _LIVE_TAGS.get(name)
    if namespace in {"rollout", "rollout_gate"}:
        if name in _ROLLOUT_QUALITY_TAGS:
            return _ROLLOUT_QUALITY_TAGS[name]
        if name in _ROLLOUT_PERFORMANCE_TAGS:
            return _ROLLOUT_PERFORMANCE_TAGS[name]
        if name in _ROLLOUT_SAMPLING_TAGS:
            return _ROLLOUT_SAMPLING_TAGS[name]
        if name in _ROLLOUT_EFFICIENCY_TAGS:
            return _ROLLOUT_EFFICIENCY_TAGS[name]
        if name == "replay_storage_bytes":
            return "replay/storage_bytes"
        if name == "replay_bytes_per_token":
            return "replay/bytes_per_token"
        if name == "peak_vram_bytes":
            return "system_rollout/peak_vram_bytes"
        if name.startswith("device_") and not name.endswith(
            ("_period_ms", "_readings")
        ):
            return f"system_rollout/{name.removeprefix('device_')}"
        return None
    if namespace == "value_warmup":
        tag = _TRAIN_TAGS.get(name)
        if tag is not None:
            category, suffix = tag.split("/", 1)
            return f"{category}_warmup/{suffix}"
        return None
    return None


def tensorboard_scalars(
    writer: SummaryWriter,
    namespace: str,
    metrics: dict[str, Any],
    step: int,
) -> None:
    for name, value in metrics.items():
        tag = _organized_tensorboard_tag(namespace, name)
        if tag is not None and isinstance(value, (int, float)):
            writer.add_scalar(tag, value, step)


def _bounded_tensorboard_text(text: str, *, max_characters: int = 16_000) -> str:
    if max_characters < 3:
        raise ValueError("TensorBoard text limit must be at least three characters")
    if len(text) <= max_characters:
        return text
    marker = "\n\n[... middle truncated ...]\n\n"
    remaining = max_characters - len(marker)
    beginning = remaining // 2
    return text[:beginning] + marker + text[-(remaining - beginning) :]


def tensorboard_rollout_samples(
    writer: SummaryWriter,
    namespace: str,
    rows: list[dict],
    records: list[TrajectoryRecord],
    *,
    samples_per_prompt: int,
    step: int,
    prompt_suffix: str = "",
    response_limits: Sequence[int] = (),
) -> None:
    """Publish one correct and one incorrect training response as text."""
    if len(records) != len(rows) * samples_per_prompt:
        raise ValueError("rollout samples do not match their prompt groups")
    if response_limits and len(response_limits) != len(records):
        raise ValueError("sample response limits do not match their trajectories")
    selected: dict[str, int] = {}
    for index, record in enumerate(records):
        label = "correct" if record.correct else "incorrect"
        selected.setdefault(label, index)
    for label, index in selected.items():
        record = records[index]
        row = rows[index // samples_per_prompt]
        text = "\n\n".join(
            (
                f"Reward: {1 if record.correct else -1}",
                f"Response tokens: {record.response_length}",
                *((
                    f"Response limit: {response_limits[index]}",
                    f"Prompt tokens: {record.prompt_length}",
                ) if response_limits else ()),
                f"Repetition: {json.dumps(repetition_metrics(record.text), sort_keys=True)}",
                f"Prompt:\n{_math_prompt_text(row, prompt_suffix)}",
                f"Ground truth:\n{row['reward_model']['ground_truth']}",
                f"Model response:\n{record.text}",
            )
        )
        phase = {
            "rollout_samples": "rollout",
            "value_warmup_samples": "warmup",
            "rollout_gate_samples": "gate",
        }.get(namespace, namespace)
        writer.add_text(
            f"samples/{phase}_{label}",
            _bounded_tensorboard_text(text),
            step,
        )


def _approximate_kl_terms(log_ratio: Tensor) -> Tensor:
    """Pointwise non-negative PPO KL approximation, evaluated in metric precision."""
    value = log_ratio.double()
    return value.expm1() - value


def _clipped_policy_objective(
    log_ratio: Tensor, advantages: Tensor, *, clip_low: float, clip_high: float
) -> Tensor:
    """Evaluate PPO's existing sign-dependent clipping before exponentiation."""
    clipped_log_ratio = torch.where(
        advantages >= 0,
        log_ratio.clamp(max=math.log1p(clip_high)),
        log_ratio.clamp(min=math.log1p(-clip_low)),
    )
    return clipped_log_ratio.exp() * advantages


def _ratio_moments_from_log_sums(
    log_sum: Tensor, log_square_sum: Tensor, count: int
) -> tuple[Tensor, Tensor]:
    """Recover ratio moments without overflowing an otherwise finite standard deviation."""
    if count == 0:
        zero = torch.zeros_like(log_sum)
        return zero, zero
    log_mean = log_sum - math.log(count)
    log_second_moment = log_square_sum - math.log(count)
    mean = log_mean.exp()
    relative_variance = -torch.expm1(
        (2 * log_mean - log_second_moment).clamp_max(0)
    )
    std = ((log_second_moment + relative_variance.log()) * 0.5).exp()
    return mean, torch.where(torch.isneginf(log_square_sum), mean, std)

def _scale_auxiliary_loss(
    auxiliary_loss: Tensor, reference_magnitude: Tensor
) -> Tensor:
    """Downscale an auxiliary loss to, but never above, the primary scale."""
    reference = reference_magnitude.detach().abs()
    denominator = torch.maximum(auxiliary_loss.detach().abs(), reference)
    denominator = denominator.clamp_min(torch.finfo(torch.float32).eps)
    return auxiliary_loss * reference / denominator

_PARAMETER_GRADIENT_STORAGE_SCALE = 1e-20
_AUXILIARY_TRUNK_PROBE_SCALE = 1e-20


class _GradientHealth:
    """Device-side finiteness flags resolved once per optimizer step.

    Per-shard ``bool(torch.isfinite(...))`` checks drained the launch queue
    hundreds of times per update; the flags accumulate on the device instead
    and ``check`` raises the same errors at the clip boundary.
    """

    _NAMES = (
        "primary parameter gradient norm",
        "auxiliary parameter gradient norm",
        "stored auxiliary-head gradient",
        "hidden gradient norm",
    )

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.flags = torch.zeros(len(self._NAMES), dtype=torch.bool, device=device)

    def record(self, index: int, finite: Tensor) -> None:
        self.flags[index] |= ~finite.to(self.flags.device)

    def check(self) -> None:
        for name, failed in zip(self._NAMES, self.flags.tolist(), strict=True):
            if failed:
                raise RuntimeError(f"{name} is non-finite")


def _gradient_device(*gradient_lists: Sequence[Tensor | None]) -> torch.device | None:
    for gradients in gradient_lists:
        for gradient in gradients:
            if gradient is not None:
                return gradient.device
    return None


@torch.no_grad()
def _pop_parameter_gradients(
    parameters: Sequence[Tensor],
) -> list[Tensor | None]:
    gradients: list[Tensor | None] = []
    for parameter in parameters:
        gradients.append(parameter.grad)
        parameter.grad = None
    return gradients


@torch.no_grad()
def _gradient_list_norm(gradients: Sequence[Tensor | None]) -> Tensor:
    present = [gradient for gradient in gradients if gradient is not None]
    if not present:
        return torch.zeros((), dtype=torch.float64)
    return torch.linalg.vector_norm(
        torch.stack(torch._foreach_norm(present, dtype=torch.float64))
    )


@torch.no_grad()
def _combine_balanced_hidden_gradients_(
    primary: Tensor, auxiliary: Tensor, *, health: _GradientHealth | None = None
) -> Tensor:
    """Merge two stored-scale hidden VJPs without amplifying the auxiliary.

    With ``health`` the finiteness guard is a deferred device flag (no host
    sync per shard); without it the guard raises immediately.
    """
    if primary.shape != auxiliary.shape:
        raise ValueError("hidden gradients must have matching shapes")
    primary_norm = torch.linalg.vector_norm(primary, dtype=torch.float64)
    auxiliary_norm = torch.linalg.vector_norm(auxiliary, dtype=torch.float64)
    if health is not None:
        health.record(3, primary_norm.isfinite() & auxiliary_norm.isfinite())
    else:
        if not torch.isfinite(primary_norm):
            raise RuntimeError("primary hidden gradient norm is non-finite")
        if not torch.isfinite(auxiliary_norm):
            raise RuntimeError("auxiliary hidden gradient norm is non-finite")
    coefficient = torch.where(
        auxiliary_norm > 0,
        torch.minimum(
            torch.ones((), dtype=torch.float64, device=primary.device),
            primary_norm / auxiliary_norm,
        ),
        torch.zeros((), dtype=torch.float64, device=primary.device),
    )
    auxiliary.mul_(coefficient.to(dtype=auxiliary.dtype))
    primary.add_(auxiliary)
    return primary


@torch.no_grad()
def _accumulate_balanced_parameter_gradients_(
    accumulator: list[Tensor | None],
    primary_gradients: list[Tensor | None],
    auxiliary_probe_gradients: list[Tensor | None],
    *,
    primary_scale: float = 1.0,
    probe_scale: float = _AUXILIARY_TRUNK_PROBE_SCALE,
    storage_scale: float = 1.0,
    health: _GradientHealth | None = None,
) -> None:
    """Accumulate balanced gradients at a common finite storage scale.

    Without ``health`` the finiteness guard resolves immediately; with it the
    caller resolves the deferred device flags later.
    """
    if not 0.0 < primary_scale <= 1.0:
        raise ValueError("primary_scale must be in (0, 1]")
    if not 0.0 < probe_scale <= 1.0:
        raise ValueError("probe_scale must be in (0, 1]")
    if not 0.0 < storage_scale <= 1.0:
        raise ValueError("storage_scale must be in (0, 1]")
    resolve_now = health is None
    if health is None:
        health = _GradientHealth(
            _gradient_device(primary_gradients, auxiliary_probe_gradients)
        )
    primary_norm = _gradient_list_norm(primary_gradients) / primary_scale
    health.record(0, primary_norm.isfinite())
    primary_storage_coefficient = storage_scale / primary_scale
    for index, primary in enumerate(primary_gradients):
        if primary is not None and accumulator[index] is None:
            accumulator[index] = torch.zeros_like(primary)
    primary_pairs = [
        (accumulator[index], primary)
        for index, primary in enumerate(primary_gradients)
        if primary is not None
    ]
    if primary_pairs:
        torch._foreach_add_(
            [accumulated for accumulated, _ in primary_pairs],
            [primary for _, primary in primary_pairs],
            alpha=primary_storage_coefficient,
        )
    auxiliary_present = [
        (index, auxiliary)
        for index, auxiliary in enumerate(auxiliary_probe_gradients)
        if auxiliary is not None
    ]
    if auxiliary_present:
        auxiliary_probe_norm = _gradient_list_norm(auxiliary_probe_gradients)
        health.record(1, auxiliary_probe_norm.isfinite())
        # Zero probes contribute nothing; the unselected division is discarded.
        auxiliary_coefficient = torch.where(
            auxiliary_probe_norm > 0,
            torch.minimum(
                torch.full_like(auxiliary_probe_norm, 1.0 / probe_scale),
                primary_norm.to(auxiliary_probe_norm.device) / auxiliary_probe_norm,
            ),
            torch.zeros_like(auxiliary_probe_norm),
        )
        for index, auxiliary in auxiliary_present:
            if accumulator[index] is None:
                accumulator[index] = torch.zeros_like(auxiliary)
        torch._foreach_add_(
            [accumulator[index] for index, _ in auxiliary_present],
            torch._foreach_mul(
                [auxiliary for _, auxiliary in auxiliary_present],
                auxiliary_coefficient * storage_scale,
            ),
        )
    if resolve_now:
        health.check()

@torch.no_grad()
def _accumulate_rescaled_parameter_gradients_(
    accumulator: list[Tensor | None],
    probe_gradients: list[Tensor | None],
    *,
    probe_scale: float = _AUXILIARY_TRUNK_PROBE_SCALE,
    storage_scale: float = 1.0,
    health: _GradientHealth | None = None,
) -> None:
    """Accumulate a loss-scaled gradient at the requested storage scale."""
    if not 0.0 < storage_scale <= 1.0:
        raise ValueError("storage_scale must be in (0, 1]")
    present = [
        (index, gradient)
        for index, gradient in enumerate(probe_gradients)
        if gradient is not None
    ]
    if not present:
        return
    resolve_now = health is None
    if health is None:
        health = _GradientHealth(present[0][1].device)
    stored = torch._foreach_mul(
        [gradient for _, gradient in present], storage_scale / probe_scale
    )
    # A finite fp64 norm certifies every fp32 element is finite.
    health.record(
        2,
        torch.stack(torch._foreach_norm(stored, dtype=torch.float64)).isfinite().all(),
    )
    existing = [
        (accumulator[index], value)
        for (index, _), value in zip(present, stored, strict=True)
        if accumulator[index] is not None
    ]
    if existing:
        torch._foreach_add_(
            [accumulated for accumulated, _ in existing],
            [value for _, value in existing],
        )
    for (index, _), value in zip(present, stored, strict=True):
        if accumulator[index] is None:
            accumulator[index] = value
    if resolve_now:
        health.check()


@torch.no_grad()
def _restore_parameter_gradients_(
    parameters: Sequence[Tensor],
    gradients: list[Tensor | None],
) -> None:
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient

@torch.no_grad()
def _clip_finite_grad_norm_(
    parameters: list[Tensor],
    max_norm: float,
    *,
    label: str,
    gradient_scale: float = 1.0,
) -> Tensor:
    if not 0.0 < gradient_scale <= 1.0:
        raise ValueError("gradient_scale must be in (0, 1]")
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        return torch.zeros((), dtype=torch.float64)
    stored_norm = torch.linalg.vector_norm(
        torch.stack(torch._foreach_norm(gradients, dtype=torch.float64))
    )
    total_norm = stored_norm / gradient_scale
    if not torch.isfinite(total_norm):
        failure_indices = [
            index
            for index, gradient in enumerate(gradients)
            if not torch.isfinite(gradient).all()
        ]
        examples = failure_indices[:4] + failure_indices[-4:]
        detail = ", ".join(
            f"{index}:{tuple(gradients[index].shape)}:{gradients[index].dtype}"
            for index in dict.fromkeys(examples)
        )
        raise RuntimeError(
            f"{label} gradient norm is non-finite "
            f"({len(failure_indices)}/{len(gradients)} tensors; {detail})"
        )
    coefficient = (
        (max_norm / (total_norm + 1e-6)).clamp(max=1.0) / gradient_scale
    )
    torch._foreach_mul_(gradients, coefficient)
    return total_norm






# Primary and auxiliary VJPs reuse saved tensors; the compiler may not donate
# them during either forward compilation or lazy backward compilation.
@aot_config.patch(donated_buffer=False)
def update_step(
    policy: MiniCPMVAPOPolicy,
    critic: MiniCPMVAPOCritic,
    records: list[TrajectoryRecord],
    actor_optimizer,
    critic_optimizer,
    *,
    optimizer_minibatches: int,
    replay_token_budget: int,
    replay_max_trajectories: int,
    logit_chunk_tokens: int,
    clip_low: float,
    clip_high: float,
    value_coefficient: float,
    nextlat_horizon: int,
    nextlat_samples: int,
    nextlat_mse_coefficient: float,
    nextlat_kl_coefficient: float,
    nextlat_kl_chunk_tokens: int,
    train_nextlat: bool,
    grad_clip_norm: float,
    nextlat_trunk_balance: str = "parameter",
    value_only: bool = False,
) -> dict[str, float | int]:
    if not records:
        raise ValueError("cannot update from an empty rollout")
    if not math.isfinite(grad_clip_norm) or grad_clip_norm <= 0:
        raise ValueError("gradient clip norm must be finite and positive")
    if nextlat_trunk_balance not in {"parameter", "hidden"}:
        raise ValueError("NextLat trunk balance must be parameter or hidden")
    device = next(policy.parameters()).device
    policy.train()
    critic.train()
    pad_token_id = int(getattr(policy.causal_lm.config, "pad_token_id"))
    minibatches = _optimizer_minibatches(len(records), optimizer_minibatches)
    actor_parameters = [
        parameter
        for group in actor_optimizer.param_groups
        for parameter in group["params"]
    ]
    critic_parameters = [
        parameter
        for group in critic_optimizer.param_groups
        for parameter in group["params"]
    ]
    actor_nextlat_parameters = list(policy.nextlat_head.parameters())
    critic_nextlat_parameters = list(critic.nextlat_head.parameters())
    actor_nextlat_parameter_ids = {
        id(parameter) for parameter in actor_nextlat_parameters
    }
    critic_nextlat_parameter_ids = {
        id(parameter) for parameter in critic_nextlat_parameters
    }
    actor_primary_parameters = [
        parameter
        for parameter in actor_parameters
        if id(parameter) not in actor_nextlat_parameter_ids
    ]
    critic_primary_parameters = [
        parameter
        for parameter in critic_parameters
        if id(parameter) not in critic_nextlat_parameter_ids
    ]
    totals = {
        name: torch.zeros((), device=device, dtype=torch.float64)
        for name in (
            "policy_numerator",
            "value_numerator",
            "kl",
            "sampled_forward_kl",
            "clipped",
            "value_prediction",
            "value_target",
            "value_target_sq",
            "value_residual_sq",
            "actor_nextlat",
            "actor_nextlat_mse",
            "actor_nextlat_kl",
            "actor_nextlat_balanced",
            "actor_nextlat_samples",
            "actor_nextlat_transitions",
            "advantage",
            "advantage_sq",
            "critic_nextlat",
            "critic_nextlat_mse",
            "critic_nextlat_kl",
            "critic_nextlat_balanced",
            "critic_nextlat_samples",
            "critic_nextlat_transitions",
        )
    }
    totals["ratio_log_sum"] = torch.full((), -torch.inf, device=device, dtype=torch.float64)
    totals["ratio_log_square_sum"] = torch.full_like(totals["ratio_log_sum"], -torch.inf)
    total_actions = sum(record.response_length for record in records)
    total_policy_actions = sum(
        record.response_length - (record.forced_token_index >= 0)
        for record in records
    )
    replay_microbatches = 0
    actor_grad_norms: list[Tensor] = []
    ratio_abs_log_max = torch.zeros((), device=device)
    advantage_min = torch.full((), math.inf, device=device)
    advantage_max = torch.full((), -math.inf, device=device)
    critic_grad_norms: list[Tensor] = []

    for optimizer_indices in minibatches:
        minibatch_records = [records[index] for index in optimizer_indices]
        # Normalize every memory shard by the entire optimizer batch. Never
        # average shard means: unequal lengths and forced actions change weights.
        minibatch_actions = sum(
            record.response_length for record in minibatch_records
        )
        minibatch_policy_actions = sum(
            record.response_length - (record.forced_token_index >= 0)
            for record in minibatch_records
        )
        plan = plan_replay_microbatches(
            minibatch_records,
            list(range(len(minibatch_records))),
            token_budget=replay_token_budget,
            max_trajectories=replay_max_trajectories,
        )
        replay_microbatches += len(plan)
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        actor_primary_accumulator: list[Tensor | None] = [
            None
        ] * len(actor_primary_parameters)
        critic_primary_accumulator: list[Tensor | None] = [
            None
        ] * len(critic_primary_parameters)
        actor_nextlat_accumulator: list[Tensor | None] = [
            None
        ] * len(actor_nextlat_parameters)
        critic_nextlat_accumulator: list[Tensor | None] = [
            None
        ] * len(critic_nextlat_parameters)
        actor_health = _GradientHealth(device)
        critic_health = _GradientHealth(device)
        nextlat_budgets = _nextlat_shard_samples(
            [
                sum(
                    _nextlat_record_capacity(minibatch_records[index], nextlat_horizon)
                    for index in indices
                )
                for indices in plan
            ],
            nextlat_samples,
        )
        # Native replay includes imposed delimiters. Latent replay samples only
        # answer-to-answer transitions, never placeholder thought token ids.
        nextlat_selected = max(sum(nextlat_budgets), 1)

        for shard_index, indices in enumerate(plan):
            batch = collate_replay_microbatch(
                minibatch_records, indices, pad_token_id=pad_token_id, device=device,
            )
            nextlat_budget = nextlat_budgets[shard_index]
            hidden_trunk_balance = (
                nextlat_trunk_balance == "hidden"
                and train_nextlat
                and nextlat_budget > 0
            )
            shard_policy_actions = sum(
                minibatch_records[index].response_length
                - (minibatch_records[index].forced_token_index >= 0)
                for index in indices
            )
            if not value_only and shard_policy_actions:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    actor_hidden = _replay_hidden(policy, batch)
                    actor_primary_hidden = (
                        actor_hidden.detach().requires_grad_()
                        if hidden_trunk_balance
                        else actor_hidden
                    )
                    actor_actions = _packed_action_rows(actor_primary_hidden, batch)
                    new_logprobs = _action_logprobs(
                        policy, actor_actions, batch, chunk_tokens=logit_chunk_tokens,
                    )
                    log_ratio = torch.where(
                        batch.policy_mask, new_logprobs - batch.old_logprobs, 0.0
                    )
                    policy_advantages = torch.where(
                        batch.policy_mask, batch.advantages, 0.0
                    )
                    objective = _clipped_policy_objective(
                        log_ratio, policy_advantages,
                        clip_low=clip_low, clip_high=clip_high,
                    )
                    policy_loss = -objective.sum() / minibatch_policy_actions
                    actor_nextlat_hidden = (
                        actor_hidden.detach().requires_grad_()
                        if train_nextlat and nextlat_budget
                        else None
                    )
                    actor_nextlat = (
                        _nextlat_training_loss(
                            policy,
                            actor_nextlat_hidden,
                            batch,
                            max_samples=nextlat_budget,
                            horizon=nextlat_horizon,
                            mse_coefficient=nextlat_mse_coefficient,
                            kl_coefficient=nextlat_kl_coefficient,
                            kl_chunk_tokens=nextlat_kl_chunk_tokens,
                        )
                        if actor_nextlat_hidden is not None
                        else None
                    )
                    balanced_actor_nextlat = None
                    if actor_nextlat is not None:
                        balanced_actor_nextlat = _scale_auxiliary_loss(
                            actor_nextlat.loss,
                            objective.detach().abs().sum() / shard_policy_actions,
                        )
                actor_auxiliary_probe_gradients: list[Tensor | None] = [
                    None
                ] * len(actor_primary_parameters)
                if hidden_trunk_balance:
                    assert actor_nextlat is not None
                    assert actor_nextlat_hidden is not None
                    assert balanced_actor_nextlat is not None
                    (
                        policy_loss * _PARAMETER_GRADIENT_STORAGE_SCALE
                    ).backward()
                    assert actor_primary_hidden.grad is not None
                    (
                        balanced_actor_nextlat
                        * actor_nextlat.samples
                        / nextlat_selected
                        * _PARAMETER_GRADIENT_STORAGE_SCALE
                    ).backward()
                    actor_nextlat_probe_gradients = _pop_parameter_gradients(
                        actor_nextlat_parameters
                    )
                    _accumulate_rescaled_parameter_gradients_(
                        actor_nextlat_accumulator,
                        actor_nextlat_probe_gradients,
                        storage_scale=_PARAMETER_GRADIENT_STORAGE_SCALE,
                        health=actor_health,
                    )
                    assert actor_nextlat_hidden.grad is not None
                    actor_hidden.backward(
                        _combine_balanced_hidden_gradients_(
                            actor_primary_hidden.grad,
                            actor_nextlat_hidden.grad,
                            health=actor_health,
                        )
                    )
                    del actor_nextlat_probe_gradients
                    actor_primary_gradients = _pop_parameter_gradients(
                        actor_primary_parameters
                    )
                else:
                    (
                        policy_loss * _PARAMETER_GRADIENT_STORAGE_SCALE
                    ).backward(retain_graph=actor_nextlat is not None)
                    actor_primary_gradients = _pop_parameter_gradients(
                        actor_primary_parameters
                    )
                    if actor_nextlat is not None:
                        assert actor_nextlat_hidden is not None
                        assert balanced_actor_nextlat is not None
                        (
                            balanced_actor_nextlat
                            * actor_nextlat.samples
                            / nextlat_selected
                            * _AUXILIARY_TRUNK_PROBE_SCALE
                        ).backward()
                        actor_nextlat_probe_gradients = _pop_parameter_gradients(
                            actor_nextlat_parameters
                        )
                        _accumulate_rescaled_parameter_gradients_(
                            actor_nextlat_accumulator,
                            actor_nextlat_probe_gradients,
                            storage_scale=_PARAMETER_GRADIENT_STORAGE_SCALE,
                            health=actor_health,
                        )
                        assert actor_nextlat_hidden.grad is not None
                        actor_hidden.backward(actor_nextlat_hidden.grad)
                        del actor_nextlat_probe_gradients
                        actor_auxiliary_probe_gradients = (
                            _pop_parameter_gradients(actor_primary_parameters)
                        )
                _accumulate_balanced_parameter_gradients_(
                    actor_primary_accumulator,
                    actor_primary_gradients,
                    actor_auxiliary_probe_gradients,
                    primary_scale=_PARAMETER_GRADIENT_STORAGE_SCALE,
                    storage_scale=_PARAMETER_GRADIENT_STORAGE_SCALE,
                    health=actor_health,
                )
                totals["policy_numerator"] += (
                    -objective.detach().double().sum()
                )
                totals["sampled_forward_kl"] -= log_ratio.detach().double().sum()
                totals["kl"] += _approximate_kl_terms(
                    log_ratio.detach()
                ).sum()
                totals["clipped"] += (
                    (log_ratio < math.log1p(-clip_low))
                    | (log_ratio > math.log1p(clip_high))
                ).double().sum()
                metric_log_ratio = torch.where(
                    batch.policy_mask, log_ratio.detach().double(), -torch.inf
                )
                torch.logaddexp(
                    totals["ratio_log_sum"], torch.logsumexp(metric_log_ratio, 0),
                    out=totals["ratio_log_sum"],
                )
                torch.logaddexp(
                    totals["ratio_log_square_sum"],
                    torch.logsumexp(2 * metric_log_ratio, 0),
                    out=totals["ratio_log_square_sum"],
                )
                totals["advantage"] += policy_advantages.double().sum()
                totals["advantage_sq"] += (
                    policy_advantages.double().square().sum()
                )
                torch.minimum(
                    advantage_min,
                    batch.advantages.masked_fill(~batch.policy_mask, math.inf).min(),
                    out=advantage_min,
                )
                torch.maximum(
                    advantage_max,
                    batch.advantages.masked_fill(~batch.policy_mask, -math.inf).max(),
                    out=advantage_max,
                )
                torch.maximum(
                    ratio_abs_log_max,
                    log_ratio.detach().abs().max(),
                    out=ratio_abs_log_max,
                )
                if actor_nextlat is not None:
                    assert balanced_actor_nextlat is not None
                    totals["actor_nextlat"] += (
                        actor_nextlat.loss.detach().double()
                        * actor_nextlat.samples
                    )
                    totals["actor_nextlat_mse"] += (
                        actor_nextlat.smooth_l1.detach().double()
                        * actor_nextlat.samples
                    )
                    totals["actor_nextlat_kl"] += (
                        actor_nextlat.categorical_kl.detach().double()
                        * actor_nextlat.samples
                    )
                    totals["actor_nextlat_balanced"] += (
                        balanced_actor_nextlat.detach().double()
                        * actor_nextlat.samples
                    )
                    totals["actor_nextlat_samples"] += actor_nextlat.samples
                    totals["actor_nextlat_transitions"] += (
                        actor_nextlat.transitions
                    )
                del actor_hidden, actor_primary_hidden, actor_actions
                del new_logprobs, log_ratio, metric_log_ratio
                del objective, policy_loss, policy_advantages
                del actor_nextlat, actor_nextlat_hidden, balanced_actor_nextlat
                del actor_primary_gradients, actor_auxiliary_probe_gradients

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                critic_hidden = _replay_hidden(critic, batch)
                critic_primary_hidden = (
                    critic_hidden.detach().requires_grad_()
                    if hidden_trunk_balance
                    else critic_hidden
                )
                critic_actions = _packed_action_rows(critic_primary_hidden, batch)
                predictions = critic.values(critic_actions).float()
                value_loss = F.mse_loss(
                    predictions, batch.value_targets, reduction="sum"
                ) / minibatch_actions
                critic_nextlat_hidden = (
                    critic_hidden.detach().requires_grad_()
                    if train_nextlat and nextlat_budget
                    else None
                )
                critic_nextlat = (
                    _nextlat_training_loss(
                        critic,
                        critic_nextlat_hidden,
                        batch,
                        max_samples=nextlat_budget,
                        horizon=nextlat_horizon,
                        mse_coefficient=nextlat_mse_coefficient,
                        kl_coefficient=nextlat_kl_coefficient,
                        kl_chunk_tokens=nextlat_kl_chunk_tokens,
                    )
                    if critic_nextlat_hidden is not None
                    else None
                )
                critic_primary_loss = value_coefficient * value_loss
                balanced_critic_nextlat = None
                if critic_nextlat is not None:
                    balanced_critic_nextlat = _scale_auxiliary_loss(
                        critic_nextlat.loss,
                        value_coefficient
                        * F.mse_loss(
                            predictions.detach(),
                            batch.value_targets,
                            reduction="mean",
                        ),
                    )
            critic_auxiliary_probe_gradients: list[Tensor | None] = [
                None
            ] * len(critic_primary_parameters)
            if hidden_trunk_balance:
                assert critic_nextlat is not None
                assert critic_nextlat_hidden is not None
                assert balanced_critic_nextlat is not None
                (
                    critic_primary_loss * _PARAMETER_GRADIENT_STORAGE_SCALE
                ).backward()
                assert critic_primary_hidden.grad is not None
                (
                    balanced_critic_nextlat
                    * critic_nextlat.samples
                    / nextlat_selected
                    * _PARAMETER_GRADIENT_STORAGE_SCALE
                ).backward()
                critic_nextlat_probe_gradients = _pop_parameter_gradients(
                    critic_nextlat_parameters
                )
                _accumulate_rescaled_parameter_gradients_(
                    critic_nextlat_accumulator,
                    critic_nextlat_probe_gradients,
                    storage_scale=_PARAMETER_GRADIENT_STORAGE_SCALE,
                    health=critic_health,
                )
                assert critic_nextlat_hidden.grad is not None
                critic_hidden.backward(
                    _combine_balanced_hidden_gradients_(
                        critic_primary_hidden.grad,
                        critic_nextlat_hidden.grad,
                        health=critic_health,
                    )
                )
                del critic_nextlat_probe_gradients
                critic_primary_gradients = _pop_parameter_gradients(
                    critic_primary_parameters
                )
            else:
                (
                    critic_primary_loss * _PARAMETER_GRADIENT_STORAGE_SCALE
                ).backward(retain_graph=critic_nextlat is not None)
                critic_primary_gradients = _pop_parameter_gradients(
                    critic_primary_parameters
                )
                if critic_nextlat is not None:
                    assert critic_nextlat_hidden is not None
                    assert balanced_critic_nextlat is not None
                    (
                        balanced_critic_nextlat
                        * critic_nextlat.samples
                        / nextlat_selected
                        * _AUXILIARY_TRUNK_PROBE_SCALE
                    ).backward()
                    critic_nextlat_probe_gradients = _pop_parameter_gradients(
                        critic_nextlat_parameters
                    )
                    _accumulate_rescaled_parameter_gradients_(
                        critic_nextlat_accumulator,
                        critic_nextlat_probe_gradients,
                        storage_scale=_PARAMETER_GRADIENT_STORAGE_SCALE,
                        health=critic_health,
                    )
                    assert critic_nextlat_hidden.grad is not None
                    critic_hidden.backward(critic_nextlat_hidden.grad)
                    del critic_nextlat_probe_gradients
                    critic_auxiliary_probe_gradients = _pop_parameter_gradients(
                        critic_primary_parameters
                    )
            _accumulate_balanced_parameter_gradients_(
                critic_primary_accumulator,
                critic_primary_gradients,
                critic_auxiliary_probe_gradients,
                primary_scale=_PARAMETER_GRADIENT_STORAGE_SCALE,
                storage_scale=_PARAMETER_GRADIENT_STORAGE_SCALE,
                health=critic_health,
            )
            residuals = predictions.detach() - batch.value_targets
            totals["value_numerator"] += residuals.double().square().sum()
            totals["value_prediction"] += predictions.detach().double().sum()
            totals["value_target"] += batch.value_targets.detach().double().sum()
            totals["value_target_sq"] += (
                batch.value_targets.detach().double().square().sum()
            )
            totals["value_residual_sq"] += residuals.double().square().sum()
            if critic_nextlat is not None:
                assert balanced_critic_nextlat is not None
                totals["critic_nextlat"] += (
                    critic_nextlat.loss.detach().double()
                    * critic_nextlat.samples
                )
                totals["critic_nextlat_mse"] += (
                    critic_nextlat.smooth_l1.detach().double()
                    * critic_nextlat.samples
                )
                totals["critic_nextlat_kl"] += (
                    critic_nextlat.categorical_kl.detach().double()
                    * critic_nextlat.samples
                )
                totals["critic_nextlat_balanced"] += (
                    balanced_critic_nextlat.detach().double()
                    * critic_nextlat.samples
                )
                totals["critic_nextlat_samples"] += critic_nextlat.samples
                totals["critic_nextlat_transitions"] += (
                    critic_nextlat.transitions
                )
            del batch, critic_hidden, critic_primary_hidden, critic_actions
            del predictions, residuals
            del value_loss, critic_primary_loss
            del critic_nextlat, critic_nextlat_hidden, balanced_critic_nextlat
            del critic_primary_gradients, critic_auxiliary_probe_gradients

        if not value_only and minibatch_policy_actions:
            _restore_parameter_gradients_(
                actor_primary_parameters, actor_primary_accumulator
            )
            _restore_parameter_gradients_(
                actor_nextlat_parameters, actor_nextlat_accumulator
            )
        _restore_parameter_gradients_(
            critic_primary_parameters, critic_primary_accumulator
        )
        _restore_parameter_gradients_(
            critic_nextlat_parameters, critic_nextlat_accumulator
        )
        if not value_only and minibatch_policy_actions:
            actor_health.check()
            actor_grad_norms.append(
                _clip_finite_grad_norm_(
                    actor_parameters,
                    grad_clip_norm,
                    label="actor",
                    gradient_scale=_PARAMETER_GRADIENT_STORAGE_SCALE,
                )
            )
            actor_optimizer.step()
        critic_health.check()
        critic_grad_norms.append(
            _clip_finite_grad_norm_(
                critic_parameters,
                grad_clip_norm,
                label="critic",
                gradient_scale=_PARAMETER_GRADIENT_STORAGE_SCALE,
            )
        )
        critic_optimizer.step()
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)

    count = float(total_actions)
    target_mean = totals["value_target"] / count
    target_variance = totals["value_target_sq"] / count - target_mean.square()
    explained_variance = (
        1.0 - totals["value_residual_sq"] / count / target_variance
        if target_variance > 1e-8
        else torch.zeros_like(target_variance)
    )
    actor_nextlat_count = totals["actor_nextlat_samples"].clamp_min(1.0)
    critic_nextlat_count = totals["critic_nextlat_samples"].clamp_min(1.0)
    value_loss_value = totals["value_numerator"] / count
    metrics: dict[str, float | int] = {
        "value_loss": float(value_loss_value),
        "value_mean": float(totals["value_prediction"] / count),
        "value_target_mean": float(target_mean),
        "explained_variance": float(explained_variance),
        "critic_grad_norm": float(torch.stack(critic_grad_norms).max()),
        "replay_microbatches": replay_microbatches,
        "optimizer_minibatches": len(minibatches),
        "replay_actions": total_actions,
        "replay_policy_actions": total_policy_actions,
        "actor_nextlat_loss": (
            float(totals["actor_nextlat"] / actor_nextlat_count)
            if train_nextlat and not value_only
            else 0.0
        ),
        "critic_nextlat_loss": (
            float(totals["critic_nextlat"] / critic_nextlat_count)
            if train_nextlat
            else 0.0
        ),
        "actor_nextlat_balanced_loss": (
            float(totals["actor_nextlat_balanced"] / actor_nextlat_count)
            if train_nextlat and not value_only
            else 0.0
        ),
        "actor_nextlat_smooth_l1": (
            float(totals["actor_nextlat_mse"] / actor_nextlat_count)
            if train_nextlat and not value_only
            else 0.0
        ),
        "actor_nextlat_categorical_kl": (
            float(totals["actor_nextlat_kl"] / actor_nextlat_count)
            if train_nextlat and not value_only
            else 0.0
        ),
        "critic_nextlat_smooth_l1": (
            float(totals["critic_nextlat_mse"] / critic_nextlat_count)
            if train_nextlat
            else 0.0
        ),
        "critic_nextlat_categorical_kl": (
            float(totals["critic_nextlat_kl"] / critic_nextlat_count)
            if train_nextlat
            else 0.0
        ),
        "critic_nextlat_balanced_loss": (
            float(totals["critic_nextlat_balanced"] / critic_nextlat_count)
            if train_nextlat
            else 0.0
        ),
        "actor_nextlat_samples": int(totals["actor_nextlat_samples"]),
        "critic_nextlat_samples": int(totals["critic_nextlat_samples"]),
        "actor_nextlat_transitions": int(
            totals["actor_nextlat_transitions"]
        ),
        "critic_nextlat_transitions": int(
            totals["critic_nextlat_transitions"]
        ),
        "gradient_clip_norm": grad_clip_norm,
    }
    policy.eval()
    critic.eval()
    if getattr(policy, "token_carry", False):
        scale_statistics = {
            f"carry_{name}_{key}": value
            for name, side in (("actor", policy), ("critic", critic))
            for key, value in _carry_scale_statistics(side.token_combiner.scale).items()
        }
        scale_values = torch.stack(tuple(scale_statistics.values())).cpu().tolist()
        metrics.update(zip(scale_statistics, scale_values, strict=True))
    if value_only:
        return metrics
    policy_count = float(max(total_policy_actions, 1))
    ratio_mean, ratio_std = _ratio_moments_from_log_sums(
        totals["ratio_log_sum"], totals["ratio_log_square_sum"], total_policy_actions
    )
    advantage_mean = totals["advantage"] / policy_count
    advantage_variance = (
        totals["advantage_sq"] / policy_count - advantage_mean.square()
    )
    policy_loss_value = totals["policy_numerator"] / policy_count
    actor_nextlat_balanced_value = (
        totals["actor_nextlat_balanced"] / actor_nextlat_count
    )
    critic_nextlat_balanced_value = (
        totals["critic_nextlat_balanced"] / critic_nextlat_count
    )
    metrics.update(
        loss=float(
            policy_loss_value
            + value_coefficient * value_loss_value
            + actor_nextlat_balanced_value
            + critic_nextlat_balanced_value
        ),
        policy_loss=float(policy_loss_value),
        approximate_kl=float(totals["kl"] / policy_count),
        sampled_forward_kl=float(totals["sampled_forward_kl"] / policy_count),
        clip_fraction=float(totals["clipped"] / policy_count),
        ratio_mean=float(ratio_mean),
        ratio_abs_log_max=float(ratio_abs_log_max),
        ratio_std=float(ratio_std),
        advantage_mean=float(advantage_mean),
        advantage_std=float(advantage_variance.clamp_min(0).sqrt()),
        advantage_min=float(advantage_min) if total_policy_actions else 0.0,
        advantage_max=float(advantage_max) if total_policy_actions else 0.0,
        actor_grad_norm=float(torch.stack(actor_grad_norms).max()) if actor_grad_norms else 0.0,
    )
    return metrics


def save_checkpoint(
    path: Path,
    policy: MiniCPMVAPOPolicy,
    critic: MiniCPMVAPOCritic,
    actor_optimizer,
    critic_optimizer,
    *,
    step: int,
    cursor: int,
    warmup_step: int,
    pending_records: list[TrajectoryRecord] | None,
    pending_epoch: int,
    args,
    data_sha256: str,
    corpus_identity: str,
) -> None:
    atomic_torch_save(
        {
            "policy": {
                "schema": policy_checkpoint_schema(
                    latent_thinking=bool(getattr(policy, "latent_thinking", False)),
                    token_carry=bool(getattr(policy, "token_carry", False)),
                    slot_memory=getattr(policy, "slot_memory", None) is not None,
                ),
                "actor": policy.checkpoint_payload(),
                "critic": critic.checkpoint_payload(),
            },
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "step": step,
            "cursor": cursor,
            "warmup_step": warmup_step,
            "pending_records": pending_records,
            "pending_epoch": pending_epoch,
            "args": vars(args).copy(),
            "data_sha256": data_sha256,
            "math_corpus_identity": corpus_identity,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "python_rng": random.getstate(),
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MINICPM5_MODEL_ID)
    parser.add_argument("--revision", default=MINICPM5_REVISION)
    parser.add_argument("--data", default="postraining/data/dapo-math-17k.parquet")
    parser.add_argument(
        "--output",
        default="postraining/runs/minicpm5_vapo",
        help="run directory beneath postraining/runs/ (parameter-golf-postraining dashboard)",
    )
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--value-warmup-steps", type=int, default=10)
    parser.add_argument("--prompts-per-rollout", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument(
        "--rollout-physical-batch-size",
        type=int,
        default=0,
        help="physical continuous-decode lanes; zero uses every logical row",
    )
    parser.add_argument("--prompt-tokens", type=int, default=1_024)
    parser.add_argument(
        "--prompt-suffix", default="",
        help="append to legacy math prompts; canonical task prompts own their instructions",
    )
    parser.add_argument("--max-new-tokens", type=int, default=10_000)
    parser.add_argument(
        "--context-tokens", type=int, default=None,
        help="total full-chat prompt plus response budget; omitted preserves legacy response-only limits",
    )
    parser.add_argument(
        "--answer-reserve-tokens",
        type=int,
        default=DEFAULT_ANSWER_RESERVE_TOKENS,
        help="reserve tokens after forced </think> inside each prompt's response allowance; zero disables",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="positive: bounded top-k; -1: full categorical (requires top-p=1); zero: slow exact top-p",
    )
    parser.set_defaults(thinking=True)
    parser.add_argument(
        "--latent-thinking", action=argparse.BooleanOptionalAction, default=False,
        help="use only continuous Gaussian thoughts inside the native thinking block",
    )
    parser.add_argument(
        "--token-carry", action=argparse.BooleanOptionalAction, default=False,
        help="sample tokens with a gated residual over token embeddings and detached carry",
    )
    parser.add_argument(
        "--slot-memory", action=argparse.BooleanOptionalAction, default=False,
        help="with --token-carry: joint (token, slot) actions over M overwrite slots read by softmax",
    )
    parser.add_argument("--slot-memory-slots", type=int, default=64)
    parser.add_argument("--slot-memory-heads", type=int, default=1)
    parser.add_argument("--slot-memory-head-dim", type=int, default=128)
    parser.add_argument("--thought-sigma", type=float, default=1.0)
    parser.add_argument("--init-stop-thinking-probability", type=float, default=0.9)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument(
        "--lora-initialization",
        choices=("standard", "nora"),
        default="nora",
        help="NoRA column-normalized A or standard Kaiming A",
    )
    parser.add_argument("--critic-width", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--critic-lr", type=float, default=1e-5)
    parser.add_argument("--nextlat-horizon", type=int, default=2)
    parser.add_argument("--nextlat-projection-factor", type=float, default=1.6)
    parser.add_argument("--nextlat-samples", type=int, default=64)
    parser.add_argument("--nextlat-kl-chunk-tokens", type=int, default=16)
    parser.add_argument("--nextlat-mse-coefficient", type=float, default=1.0)
    parser.add_argument("--nextlat-kl-coefficient", type=float, default=1.0)
    parser.add_argument(
        "--train-nextlat", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--nextlat-trunk-balance",
        choices=("parameter", "hidden"),
        default="parameter",
        help="balance auxiliary trunk gradients at parameters or final hidden states",
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--optimizer-minibatches", type=int, default=1,
        help=(
            "optimizer updates per PPO epoch: 1 accumulates the whole rollout "
            "before clipping and stepping; larger values split it into disjoint "
            "updates. Control VRAM with --replay-token-budget and "
            "--replay-max-trajectories, not this option"
        ),
    )
    parser.add_argument("--post-update-kl-interval", type=int, default=10)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--clip-low", type=float, default=0.20)
    parser.add_argument("--clip-high", type=float, default=0.28)
    parser.add_argument("--value-coefficient", type=float, default=1.0)
    parser.add_argument("--replay-token-budget", type=int, default=11_024)
    parser.add_argument("--replay-max-trajectories", type=int, default=16)
    parser.add_argument("--logit-chunk-tokens", type=int, default=1024)
    parser.add_argument(
        "--behavior-logprobs",
        choices=("replay", "rollout"),
        default="replay",
        help=(
            "replay: teacher-force the actor after rollout for replay-consistent "
            "behavior log-probabilities; rollout: keep the exact log-probabilities "
            "captured in the rollout graph and refresh only the critic"
        ),
    )
    parser.add_argument(
        "--compile-rollout", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--compile-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile replay MLPs while preserving native bf16 backward",
    )
    parser.add_argument(
        "--replay-checkpoint-interval",
        type=int,
        default=0,
        help="checkpoint every Nth replay layer; zero retains all activations",
    )
    parser.add_argument(
        "--replay-attention-backend",
        choices=("sdpa", "fa4"),
        default="sdpa",
        help="segmented SDPA is stable; FA4 is experimental on SM120 backward",
    )
    parser.add_argument(
        "--fast-rollout", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--uno-rollout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use a pretrained frozen Uno diffusion adapter for lossless block rollouts",
    )
    parser.add_argument(
        "--uno-checkpoint",
        help="adapter from scripts/train_minicpm_uno.py; required with --uno-rollout",
    )
    parser.add_argument("--uno-block-size", type=int, default=4)
    parser.add_argument(
        "--min-rollout-tokens-per-second", type=float, default=4_000.0,
        help="AR: scheduled decode tok/s; Uno: useful end-to-end rollout tok/s. Set from a matched benchmark.",
    )
    parser.add_argument(
        "--device-telemetry", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device-telemetry-interval-ms", type=int, default=250)
    parser.add_argument("--device-power-floor", type=float, default=400.0)
    parser.add_argument("--checkpoint-interval-seconds", type=float, default=480)
    parser.add_argument("--resume")
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--gate-min-positive-trajectories", type=int, default=1)
    parser.add_argument("--gate-min-positive-groups", type=int, default=1)
    parser.add_argument("--gate-max-truncation-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def validate_output_directory(output: str | Path) -> None:
    runs_root = (Path(__file__).resolve().parent / "runs").resolve()
    output = Path(output).resolve()
    if output == runs_root or not output.is_relative_to(runs_root):
        raise ValueError(
            f"--output must be a run directory beneath {runs_root}; "
            "post-training belongs in parameter-golf-postraining, not parameter-golf"
        )


def _validate_args(args) -> None:
    validate_output_directory(args.output)
    positive_integer_names = (
        "prompts_per_rollout",
        "samples_per_prompt",
        "prompt_tokens",
        "max_new_tokens",
        "lora_rank",
        "critic_width",
        "replay_token_budget",
        "replay_max_trajectories",
        "logit_chunk_tokens",
        "nextlat_horizon",
        "nextlat_samples",
        "nextlat_kl_chunk_tokens",
        "optimizer_minibatches",
        "post_update_kl_interval",
    )
    finite_float_names = (
        "temperature",
        "top_p",
        "lora_alpha",
        "actor_lr",
        "critic_lr",
        "nextlat_projection_factor",
        "nextlat_mse_coefficient",
        "nextlat_kl_coefficient",
        "gradient_clip_norm",
        "clip_low",
        "clip_high",
        "value_coefficient",
        "device_power_floor",
        "checkpoint_interval_seconds",
        "gate_max_truncation_fraction",
        "min_rollout_tokens_per_second",
    )
    nonfinite = [
        name for name in finite_float_names if not math.isfinite(getattr(args, name))
    ]
    if nonfinite:
        raise ValueError(f"finite float options required: {', '.join(nonfinite)}")
    invalid = [name for name in positive_integer_names if getattr(args, name) < 1]
    if invalid:
        raise ValueError(f"positive integer options required: {', '.join(invalid)}")
    if args.context_tokens is not None:
        if args.context_tokens < 1:
            raise ValueError("context_tokens must be positive")
        if not args.fast_rollout or not args.compile_rollout or args.latent_thinking or args.uno_rollout:
            raise ValueError("context_tokens requires native compiled fast rollout; latent/Uno/eager paths are unsupported")
        response_token_limits(
            [1], max_new_tokens=args.max_new_tokens,
            context_tokens=args.context_tokens,
            answer_reserve_tokens=args.answer_reserve_tokens,
            thinking_end_token_id=0 if args.answer_reserve_tokens else None,
        )
    if not math.isfinite(args.thought_sigma) or args.thought_sigma <= 0:
        raise ValueError("thought sigma must be finite and positive")
    if not 0 < args.init_stop_thinking_probability < 1:
        raise ValueError("initial stop-thinking probability must lie in (0, 1)")
    if args.token_carry:
        if args.latent_thinking or args.uno_rollout:
            raise ValueError("token carry cannot use Gaussian thoughts or Uno proposals")
        if not args.fast_rollout or not args.compile_rollout or not args.compile_replay:
            raise ValueError("token carry requires compiled fast rollout and replay")
        if args.train_nextlat:
            raise ValueError("token carry requires --no-train-nextlat for an isolated recurrence experiment")
    if args.slot_memory:
        if not args.token_carry:
            raise ValueError("slot memory extends token carry; pass --token-carry --slot-memory")
        slot_memory_config_from_args(args)
    if args.latent_thinking:
        if args.uno_rollout:
            raise ValueError("latent thinking cannot use token-only Uno proposals")
        if args.max_new_tokens < 3:
            raise ValueError("latent thinking needs a thought, close delimiter and answer")
    if args.answer_reserve_tokens < 0 or (
        args.answer_reserve_tokens > 0
        and args.max_new_tokens <= args.answer_reserve_tokens + 1
    ):
        raise ValueError("answer reserve must leave room for thinking and its delimiter")
    if args.replay_checkpoint_interval < 0:
        raise ValueError("replay checkpoint interval cannot be negative")
    if args.rollout_physical_batch_size < 0:
        raise ValueError("physical rollout batch cannot be negative")
    rollout_trajectories = args.prompts_per_rollout * args.samples_per_prompt
    if args.optimizer_minibatches > rollout_trajectories:
        raise ValueError("optimizer minibatches cannot exceed rollout trajectories")
    if args.rollout_physical_batch_size > rollout_trajectories:
        raise ValueError("physical rollout batch cannot exceed trajectories")
    if (
        not (args.fast_rollout or args.latent_thinking)
        and args.rollout_physical_batch_size not in (0, rollout_trajectories)
    ):
        raise ValueError("smaller physical rollout batches require fast rollout")
    if args.steps < 0 or args.value_warmup_steps < 0 or args.ppo_epochs < 1:
        raise ValueError("training step counts must be nonnegative and epochs positive")
    if not 0 < args.top_p <= 1 or args.temperature <= 0:
        raise ValueError("sampling temperature/top-p are invalid")
    if not -1 <= args.top_k <= MINICPM5_VOCAB_SIZE:
        raise ValueError("top-k must be -1, zero, or fit the model vocabulary")
    if args.top_k == -1:
        if args.top_p != 1.0:
            raise ValueError("unrestricted top-k sampling requires top-p=1")
        if args.latent_thinking or args.uno_rollout:
            raise ValueError("unrestricted sampling is supported only by native token rollouts")
    if (args.fast_rollout or args.latent_thinking) and args.top_k == 0:
        raise ValueError("captured rollouts require top-k=-1 or positive bounded top-k")
    if not 2 <= args.uno_block_size <= 16:
        raise ValueError("Uno block size must lie in [2, 16]")
    if args.uno_rollout:
        if not args.fast_rollout or not args.compile_rollout:
            raise ValueError("Uno requires fast, CUDA-graph-captured rollout")
        if not args.uno_checkpoint or not Path(args.uno_checkpoint).is_file():
            raise ValueError("Uno requires an existing pretrained --uno-checkpoint")
    elif args.uno_checkpoint is not None:
        raise ValueError("--uno-checkpoint requires --uno-rollout")
    if args.ppo_epochs != 1:
        raise ValueError("fixed behavior statistics currently require one PPO epoch")
    if args.behavior_logprobs == "rollout" and (
        args.latent_thinking or args.token_carry or args.uno_rollout
    ):
        raise ValueError(
            "rollout behavior log-probabilities require a native token rollout"
        )
    if not 0 < args.clip_low < 1 or args.clip_high <= 0:
        raise ValueError("VAPO clipping bounds are invalid")
    if args.actor_lr <= 0 or args.critic_lr <= 0:
        raise ValueError("actor and critic learning rates must be positive")
    if args.lora_alpha <= 0:
        raise ValueError("LoRA alpha must be positive")
    if (
        args.value_coefficient < 0
        or args.nextlat_mse_coefficient < 0
        or args.nextlat_kl_coefficient < 0
    ):
        raise ValueError("loss coefficients must be nonnegative")
    if args.nextlat_projection_factor <= 0 or args.gradient_clip_norm <= 0:
        raise ValueError("NextLat dimensions and gradient clip must be positive")
    if args.min_rollout_tokens_per_second <= 0:
        raise ValueError("minimum rollout throughput must be positive")
    maximum_sequence_length = args.prompt_tokens + args.max_new_tokens
    if args.context_tokens is not None:
        maximum_sequence_length = min(maximum_sequence_length, args.context_tokens)
    if maximum_sequence_length > 20_480:
        raise ValueError(
            "configured context exceeds the validated single-GPU VAPO envelope"
        )
    maximum_replay_length = maximum_sequence_length - 1
    if not args.rollout_only and maximum_replay_length > args.replay_token_budget:
        raise ValueError(
            "replay token budget must fit one maximum-length trajectory"
        )
    if args.device_telemetry_interval_ms < 1:
        raise ValueError("device telemetry interval must be positive")
    if args.device_power_floor < 0:
        raise ValueError("device power floor must be nonnegative")
    if args.checkpoint_interval_seconds <= 0:
        raise ValueError("checkpoint interval must be positive")
    if not 0 <= args.gate_max_truncation_fraction <= 1:
        raise ValueError("gate truncation fraction must lie in [0, 1]")


def configure_replay_checkpointing(causal_lm: Any, interval: int) -> int:
    """Checkpoint uniformly spaced decoder layers and retain the rest."""
    if interval < 0:
        raise ValueError("replay checkpoint interval cannot be negative")
    layers = causal_lm.get_submodule("model").layers
    enabled = 0
    for index, layer in enumerate(layers):
        checkpointed = interval > 0 and index % interval == 0
        if not hasattr(layer, "gradient_checkpointing"):
            raise TypeError("replay decoder layer does not support checkpointing")
        layer.gradient_checkpointing = checkpointed
        enabled += int(checkpointed)
    return enabled




def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    args.uno_checkpoint_sha256 = (
        file_sha256(args.uno_checkpoint) if args.uno_rollout else None
    )
    from postraining.invariant_linear import INVARIANT_ARITHMETIC
    args.uno_arithmetic = INVARIANT_ARITHMETIC if args.uno_rollout else None
    data_sha256 = file_sha256(args.data)
    rows, corpus_audit, corpus_identity = load_training_math_corpus(
        args.data, seed=args.seed
    )
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_dataset(resume, data_sha256, corpus_identity)
        validate_resume_configuration(resume, args)
    if len(rows) < args.prompts_per_rollout:
        raise ValueError("effective corpus must contain at least prompts_per_rollout unique rows")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output / "metrics.jsonl")
    corpus_metadata = {
        "data_sha256": data_sha256,
        "math_corpus_identity": corpus_identity,
        "math_corpus_audit": corpus_audit,
    }
    logger.log(type="corpus", **corpus_metadata)
    print(json.dumps({"type": "corpus", **corpus_metadata}, sort_keys=True), flush=True)
    checkpoint_policy = RecoveryCheckpointPolicy(args.checkpoint_interval_seconds)
    # Corpus validation above must finish before model construction or CUDA work.
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        initialization=args.lora_initialization,
    )
    policy, tokenizer = MiniCPMVAPOPolicy.from_pretrained(
        model_id=args.model,
        revision=args.revision,
        device=device,
        lora_config=lora_config,
        nextlat_projection_factor=args.nextlat_projection_factor,
        gradient_checkpointing=args.replay_checkpoint_interval > 0,
        latent_thinking=args.latent_thinking,
        token_carry=args.token_carry,
        slot_memory=slot_memory_config_from_args(args),
        thought_sigma=args.thought_sigma,
        init_stop_thinking_probability=args.init_stop_thinking_probability,
    )
    critic = MiniCPMVAPOCritic.from_pretrained(
        model_id=args.model,
        revision=args.revision,
        device=device,
        lora_config=lora_config,
        critic_width=args.critic_width,
        nextlat_projection_factor=args.nextlat_projection_factor,
        gradient_checkpointing=args.replay_checkpoint_interval > 0,
        shared_frozen_source=policy.causal_lm,
        latent_thinking=args.latent_thinking,
        token_carry=args.token_carry,
        slot_memory=slot_memory_config_from_args(args),
    )
    actor_checkpointed_layers = configure_replay_checkpointing(
        policy.causal_lm, args.replay_checkpoint_interval
    )
    critic_checkpointed_layers = configure_replay_checkpointing(
        critic.causal_lm, args.replay_checkpoint_interval
    )
    if actor_checkpointed_layers != critic_checkpointed_layers:
        raise RuntimeError("actor and critic checkpoint topology differs")
    actor_backbone_parameters = list(policy.actor_parameters())
    actor_nextlat_parameters = list(policy.nextlat_head.parameters())
    critic_backbone_parameters = list(critic.backbone_parameters())
    critic_value_parameters = list(critic.value_head.parameters())
    critic_nextlat_parameters = list(critic.nextlat_head.parameters())
    actor_optimizer = torch.optim.AdamW(
        (
            {"params": actor_backbone_parameters, "lr": args.actor_lr},
            {
                "params": actor_nextlat_parameters,
                "lr": args.actor_lr,
                "weight_decay": 0.1,
                "betas": (0.9, 0.95),
            },
        ),
        fused=True,
        weight_decay=0.0,
    )
    critic_optimizer = torch.optim.AdamW(
        (
            {
                "params": critic_backbone_parameters + critic_value_parameters,
                "lr": args.critic_lr,
            },
            {
                "params": critic_nextlat_parameters,
                "lr": args.critic_lr,
                "weight_decay": 0.1,
                "betas": (0.9, 0.95),
            },
        ),
        fused=True,
        weight_decay=0.0,
    )
    enable_packed_replay_attention(
        policy.causal_lm, backend=args.replay_attention_backend
    )
    enable_packed_replay_attention(
        critic.causal_lm, backend=args.replay_attention_backend
    )
    if not (args.fast_rollout or args.latent_thinking):
        use_packed_replay_attention(policy.causal_lm, enabled=False)
    if args.compile_replay:
        enable_replay_mlp_compilation(policy.causal_lm)
        enable_replay_mlp_compilation(critic.causal_lm)
    start_step = 0
    cursor = 0
    completed_warmup = 0
    pending_records: list[TrajectoryRecord] | None = None
    pending_epoch = 0
    if resume is not None:
        payload = resume["policy"]
        actor_payload = payload["actor"]
        critic_payload = payload["critic"]
        for side_payload in (actor_payload, critic_payload):
            if (
                side_payload["model_id"] != args.model
                or side_payload["revision"] != args.revision
            ):
                raise ValueError("resume base checkpoint differs")
            saved_lora_config = dict(side_payload["lora_config"])
            saved_lora_config.setdefault("initialization", "standard")
            if saved_lora_config != {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "targets": tuple(lora_config.targets),
                "initialization": args.lora_initialization,
            }:
                raise ValueError("resume LoRA configuration differs")
            if (
                side_payload["nextlat_projection_factor"]
                != args.nextlat_projection_factor
            ):
                raise ValueError("resume NextLat projection factor differs")
        load_adapter_state_dict(policy.causal_lm, actor_payload["adapter"])
        policy.nextlat_head.load_state_dict(
            actor_payload["nextlat"], strict=True
        )
        load_adapter_state_dict(critic.causal_lm, critic_payload["adapter"])
        critic.value_head.load_state_dict(
            critic_payload["value_head"], strict=True
        )
        critic.nextlat_head.load_state_dict(
            critic_payload["nextlat"], strict=True
        )
        if args.latent_thinking:
            policy.load_latent_state_dict(actor_payload)
            critic.load_latent_state_dict(critic_payload)
        policy.load_token_carry_state_dict(actor_payload)
        critic.load_token_carry_state_dict(critic_payload)
        actor_optimizer.load_state_dict(resume["actor_optimizer"])
        critic_optimizer.load_state_dict(resume["critic_optimizer"])
        reassert_optimizer_learning_rates(
            actor_optimizer,
            critic_optimizer,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
        )
        start_step = int(resume["step"])
        cursor = int(resume["cursor"])
        torch.set_rng_state(resume["cpu_rng"])
        torch.cuda.set_rng_state(resume["cuda_rng"])
        random.setstate(resume["python_rng"])
        completed_warmup = int(resume["warmup_step"])
        pending_records = resume["pending_records"]
        pending_epoch = int(resume["pending_epoch"])
        if not 0 <= completed_warmup <= args.value_warmup_steps:
            raise ValueError("resume warmup progress is invalid")
        if pending_records is None and pending_epoch != 0:
            raise ValueError("resume has a pending epoch without replay records")
        if pending_records is not None and not 0 <= pending_epoch < args.ppo_epochs:
            raise ValueError("resume pending PPO epoch is invalid")
    else:
        load_adapter_state_dict(
            critic.causal_lm, adapter_state_dict(policy.causal_lm)
        )
        if args.latent_thinking:
            critic.thought_adapter.load_state_dict(policy.thought_adapter.state_dict())
        if args.token_carry:
            critic.token_combiner.load_state_dict(policy.token_combiner.state_dict())
    actor_trainable_storage = {
        parameter.untyped_storage().data_ptr()
        for parameter in actor_backbone_parameters + actor_nextlat_parameters
    }
    critic_trainable_storage = {
        parameter.untyped_storage().data_ptr()
        for parameter in critic_backbone_parameters
        + critic_value_parameters
        + critic_nextlat_parameters
    }
    if actor_trainable_storage & critic_trainable_storage:
        raise RuntimeError("actor and critic unexpectedly share trainable storage")
    actor_parameters_by_name = dict(policy.causal_lm.named_parameters())
    critic_parameters_by_name = dict(critic.causal_lm.named_parameters())
    if not critic.shared_frozen_parameters:
        raise RuntimeError("critic did not share the immutable actor backbone")
    for name in critic.shared_frozen_parameters:
        if (
            actor_parameters_by_name[name].untyped_storage().data_ptr()
            != critic_parameters_by_name[name].untyped_storage().data_ptr()
        ):
            raise RuntimeError(f"immutable actor/critic parameter was copied: {name}")

    # TensorBoard purge_step is global, but live decode and optimizer metrics
    # intentionally use different step domains. Append resume sessions instead.
    tensorboard = SummaryWriter(output / "tensorboard")

    if args.latent_thinking:
        stop_ids = _stop_ids(policy, tokenizer)
        engine = MiniCPMLatentRolloutEngine(
            policy,
            stop_ids=stop_ids,
            thinking_start_token_id=resolve_thinking_start_token(
                tokenizer, stop_ids=stop_ids,
            ),
            thinking_end_token_id=resolve_thinking_end_token(
                tokenizer, stop_ids=stop_ids,
            ),
            prompts_per_rollout=args.prompts_per_rollout,
            samples_per_prompt=args.samples_per_prompt,
            cache_length=args.prompt_tokens + args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            compile_decode=args.compile_rollout,
            physical_batch_size=args.rollout_physical_batch_size or None,
            answer_reserve_tokens=args.answer_reserve_tokens,
        )
    elif args.fast_rollout:
        engine_class = CapturedTrainingRolloutEngine
        thinking_end_token_id = (
            resolve_thinking_end_token(tokenizer, stop_ids=_stop_ids(policy, tokenizer))
            if args.answer_reserve_tokens else None
        )
        engine_options: dict[str, Any]
        if args.uno_rollout:
            from postraining.uno_speculative import UnoTrainingRolloutEngine

            engine_class = UnoTrainingRolloutEngine
            engine_options = {
                "uno_checkpoint": args.uno_checkpoint,
                "uno_block_size": args.uno_block_size,
            }
        else:
            engine_options = {
                "capture_logprobs": args.behavior_logprobs == "rollout"
            }
        engine = engine_class(
            policy,
            stop_ids=_stop_ids(policy, tokenizer),
            prompts_per_rollout=args.prompts_per_rollout,
            samples_per_prompt=args.samples_per_prompt,
            cache_length=args.context_tokens or (args.prompt_tokens + args.max_new_tokens),
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            compile_decode=args.compile_rollout,
            physical_batch_size=args.rollout_physical_batch_size or None,
            answer_reserve_tokens=args.answer_reserve_tokens,
            thinking_end_token_id=thinking_end_token_id,
            **engine_options,
        )
        if args.uno_rollout and file_sha256(args.uno_checkpoint) != args.uno_checkpoint_sha256:
            raise ValueError("Uno checkpoint changed while loading the rollout engine")
    else:
        engine = RolloutEngine(
            policy,
            tokenizer,
            prompts_per_rollout=args.prompts_per_rollout,
            samples_per_prompt=args.samples_per_prompt,
            cache_length=args.prompt_tokens + args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            compile_decode=args.compile_rollout,
            answer_reserve_tokens=args.answer_reserve_tokens,
            thinking_end_token_id=(
                resolve_thinking_end_token(tokenizer, stop_ids=_stop_ids(policy, tokenizer))
                if args.answer_reserve_tokens else None
            ),
        )
    args.rollout_arithmetic = (
        engine.arithmetic if args.fast_rollout or args.latent_thinking else None
    )
    if args.resume and (args.fast_rollout or args.latent_thinking) and not args.uno_rollout:
        validate_resume_rollout_arithmetic(resume, args.rollout_arithmetic)

    device_sampler = None
    if args.device_telemetry:
        device_sampler = DeviceSampler(args.device_telemetry_interval_ms, device)
        device_sampler.start()
        atexit.register(device_sampler.stop)
    parameter_metrics = {
        "trainable_actor_parameters": sum(
            parameter.numel() for parameter in actor_backbone_parameters
        ),
        "trainable_critic_parameters": sum(
            parameter.numel()
            for parameter in critic_backbone_parameters
            + critic_value_parameters
        ),
        "rollout_batch_rows": engine.batch_size,
        "rollout_replica_parameters": (
            sum(parameter.numel() for parameter in engine.policy.parameters())
            if args.fast_rollout or args.latent_thinking
            else 0
        ),
        "rollout_fused_projection_groups": len(
            getattr(engine, "fused_projection_groups", ())
        ),
        "actor_nextlat_parameters": sum(
            parameter.numel() for parameter in actor_nextlat_parameters
        ),
        "critic_nextlat_parameters": sum(
            parameter.numel() for parameter in critic_nextlat_parameters
        ),
        "estimated_static_kv_cache_bytes": engine.estimated_cache_bytes,
        "rollout_offloaded_source_bytes": getattr(
            engine, "offloaded_source_bytes", 0
        ),
        "shared_frozen_parameters": sum(
            actor_parameters_by_name[name].numel()
            for name in critic.shared_frozen_parameters
        ),
        "shared_frozen_bytes": sum(
            actor_parameters_by_name[name].numel()
            * actor_parameters_by_name[name].element_size()
            for name in critic.shared_frozen_parameters
        ),
        "frozen_actor_parameters": sum(
            parameter.numel()
            for parameter in policy.causal_lm.parameters()
            if not parameter.requires_grad
        ),
        "frozen_critic_parameters": sum(
            parameter.numel()
            for parameter in critic.causal_lm.parameters()
            if not parameter.requires_grad
        ),
        "total_actor_parameters": sum(
            parameter.numel() for parameter in policy.parameters()
        ),
        "total_critic_parameters": sum(
            parameter.numel() for parameter in critic.parameters()
        ),
    }

    tensorboard_scalars(
        tensorboard, "config", {**parameter_metrics, **vars(args)}, 0
    )
    tensorboard_scalars(tensorboard, "corpus", {
        f"{domain}/{statistic}": value
        for statistic, values in (
            ("count", corpus_audit["domain_counts"]),
            ("proportion", corpus_audit["domain_proportions"]),
        )
        for domain, value in values.items()
    }, 0)
    tensorboard.add_text(
        "samples/configuration",
        json.dumps(
            {
                **vars(args),
                "data_sha256": data_sha256,
                "math_corpus_identity": corpus_identity,
                "math_corpus_audit": corpus_audit,
                **({"uno": getattr(engine, "uno_metadata")} if args.uno_rollout else {}),
            },
            sort_keys=True,
        ),
        0,
    )
    tensorboard.flush()

    def live_rollout_metrics(
        progress_step: int,
        metrics: dict[str, float | int],
        *,
        namespace: str = "rollout_live",
    ) -> None:
        live_metrics = dict(metrics)
        if device_sampler is not None and device_sampler.samples:
            _, device_values = device_sampler.samples[-1]
            live_metrics.update(
                {
                    f"device_{name}": value
                    for name, value in zip(DEVICE_SAMPLE_FIELDS, device_values)
                }
            )
        tensorboard_scalars(
            tensorboard, namespace, live_metrics, progress_step
        )
    policy.eval()

    def rollout_performance_passed(metrics: dict[str, float | int]) -> bool:
        return (
            float(
                metrics[
                    "rollout_tokens_per_second"
                    if args.uno_rollout
                    else "scheduled_decode_tokens_per_second"
                ]
            )
            >= args.min_rollout_tokens_per_second
        )
    torch.cuda.reset_peak_memory_stats(device)

    if args.rollout_only:
        selected = rows[: args.prompts_per_rollout]
        rollout_started = time.perf_counter()
        result = collect_rollouts(
            engine,
            tokenizer,
            selected,
            prompt_tokens=args.prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            context_tokens=args.context_tokens,
            enable_thinking=args.thinking,
            prompt_suffix=args.prompt_suffix,
            progress_callback=live_rollout_metrics,
        )
        rollout_ended = time.perf_counter()
        metrics = rollout_diagnostics(
            result,
            samples_per_prompt=args.samples_per_prompt,
            stop_ids=engine.stop_ids,
        )
        metrics.update(corpus_progress(args.prompts_per_rollout, len(rows)))
        metrics.update(
            device_phase_metrics(
                device_sampler,
                started=rollout_started,
                ended=rollout_ended,
                power_floor=args.device_power_floor,
            )
        )
        metrics["peak_vram_bytes"] = torch.cuda.max_memory_allocated(device)
        passed = (
            metrics["positive_trajectories"]
            >= args.gate_min_positive_trajectories
            and metrics["positive_groups"] >= args.gate_min_positive_groups
            and metrics["truncation_fraction"] <= args.gate_max_truncation_fraction
            and rollout_performance_passed(metrics)
        )
        tensorboard_rollout_samples(
            tensorboard,
            "rollout_gate_samples",
            selected,
            result.records,
            samples_per_prompt=args.samples_per_prompt,
            step=0,
            prompt_suffix=args.prompt_suffix,
            response_limits=result.response_limits,
        )
        tensorboard_scalars(
            tensorboard,
            "rollout_gate",
            {"passed": int(passed), **metrics},
            0,
        )
        logger.log(type="rollout_gate", step=0, passed=passed, **metrics)
        print(json.dumps({"passed": passed, **metrics}, sort_keys=True), flush=True)
        tensorboard.close()
        raise SystemExit(0 if passed else 2)

    for warmup in range(completed_warmup, args.value_warmup_steps):
        selected = select_corpus_rows(rows, cursor=cursor, count=args.prompts_per_rollout)
        cursor += args.prompts_per_rollout
        torch.cuda.reset_peak_memory_stats(device)
        if not (args.fast_rollout or args.latent_thinking):
            use_packed_replay_attention(policy.causal_lm, enabled=False)
        rollout_started = time.perf_counter()
        result = collect_rollouts(
            engine,
            tokenizer,
            selected,
            prompt_tokens=args.prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            context_tokens=args.context_tokens,
            enable_thinking=args.thinking,
            prompt_suffix=args.prompt_suffix,
            progress_callback=lambda progress_step, live_metrics: live_rollout_metrics(
                warmup * args.max_new_tokens + progress_step,
                live_metrics,
                namespace="value_warmup_live",
            ),
        )
        rollout_ended = time.perf_counter()
        metrics = rollout_diagnostics(
            result,
            samples_per_prompt=args.samples_per_prompt,
            stop_ids=engine.stop_ids,
        )
        metrics.update(corpus_progress(cursor, len(rows)))
        metrics.update(
            device_phase_metrics(
                device_sampler,
                started=rollout_started,
                ended=rollout_ended,
                power_floor=args.device_power_floor,
                prefix="rollout_",
            )
        )
        if not rollout_performance_passed(metrics):
            raise RuntimeError(
                "rollout throughput fell below the configured production floor"
            )
        metrics["rollout_peak_vram_bytes"] = torch.cuda.max_memory_allocated(
            device
        )
        tensorboard_rollout_samples(
            tensorboard,
            "value_warmup_samples",
            selected,
            result.records,
            samples_per_prompt=args.samples_per_prompt,
            step=warmup + 1,
            prompt_suffix=args.prompt_suffix,
            response_limits=result.response_limits,
        )
        engine.release_cache()
        if not (args.fast_rollout or args.latent_thinking):
            use_packed_replay_attention(policy.causal_lm, enabled=True)

        torch.cuda.reset_peak_memory_stats(device)
        update_started = time.perf_counter()
        update_metrics = update_step(
            policy,
            critic,
            result.records,
            actor_optimizer,
            critic_optimizer,
            optimizer_minibatches=args.optimizer_minibatches,
            replay_token_budget=args.replay_token_budget,
            replay_max_trajectories=args.replay_max_trajectories,
            logit_chunk_tokens=args.logit_chunk_tokens,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            value_coefficient=args.value_coefficient,
            nextlat_horizon=args.nextlat_horizon,
            nextlat_samples=args.nextlat_samples,
            nextlat_mse_coefficient=args.nextlat_mse_coefficient,
            nextlat_kl_coefficient=args.nextlat_kl_coefficient,
            nextlat_kl_chunk_tokens=args.nextlat_kl_chunk_tokens,
            train_nextlat=args.train_nextlat,
            nextlat_trunk_balance=args.nextlat_trunk_balance,
            grad_clip_norm=args.gradient_clip_norm,
            value_only=True,
        )
        update_ended = time.perf_counter()
        metrics.update(update_metrics)
        metrics["update_seconds"] = update_ended - update_started
        metrics["allocated_vram_bytes"] = torch.cuda.memory_allocated(device)
        metrics["update_peak_vram_bytes"] = torch.cuda.max_memory_allocated(
            device
        )
        metrics.update(
            device_phase_metrics(
                device_sampler,
                started=update_started,
                ended=update_ended,
                power_floor=args.device_power_floor,
                prefix="update_",
            )
        )
        completed_warmup = warmup + 1

        tensorboard_scalars(
            tensorboard, "value_warmup", metrics, completed_warmup
        )
        logger.log(type="value_warmup", step=completed_warmup, **metrics)
        warmup_complete = completed_warmup == args.value_warmup_steps
        if warmup_complete or checkpoint_policy.due():
            checkpoint_path = output / (
                "critic_warmup_checkpoint.pt" if warmup_complete
                else "vapo_adapter_checkpoint.pt"
            )
            save_checkpoint(
                checkpoint_path,
                policy,
                critic,
                actor_optimizer,
                critic_optimizer,
                step=start_step,
                cursor=cursor,
                warmup_step=completed_warmup,
                pending_records=pending_records,
                pending_epoch=pending_epoch,
                args=args,
                data_sha256=data_sha256,
                corpus_identity=corpus_identity,
            )
            if warmup_complete:
                atomic_link_or_copy(checkpoint_path, output / "vapo_adapter_checkpoint.pt")
            checkpoint_policy.committed(
                (start_step, cursor, completed_warmup, pending_epoch)
            )
    step = start_step
    while step < args.steps:
        if pending_records is None:
            selected = select_corpus_rows(rows, cursor=cursor, count=args.prompts_per_rollout)
            cursor += args.prompts_per_rollout
            torch.cuda.reset_peak_memory_stats(device)
            if not (args.fast_rollout or args.latent_thinking):
                use_packed_replay_attention(policy.causal_lm, enabled=False)
            rollout_started = time.perf_counter()
            result = collect_rollouts(
                engine,
                tokenizer,
                selected,
                prompt_tokens=args.prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                context_tokens=args.context_tokens,
                enable_thinking=args.thinking,
                prompt_suffix=args.prompt_suffix,
                progress_callback=lambda progress_step, live_metrics: live_rollout_metrics(
                    step
                    * args.prompts_per_rollout
                    * args.max_new_tokens
                    + progress_step,
                    live_metrics,
                ),
            )
            rollout_ended = time.perf_counter()
            rollout_metrics = rollout_diagnostics(
                result,
                samples_per_prompt=args.samples_per_prompt,
                stop_ids=engine.stop_ids,
            )
            rollout_metrics.update(corpus_progress(cursor, len(rows)))
            rollout_metrics.update(
                device_phase_metrics(
                    device_sampler,
                    started=rollout_started,
                    ended=rollout_ended,
                    power_floor=args.device_power_floor,
                )
            )
            if not rollout_performance_passed(rollout_metrics):
                raise RuntimeError(
                    "rollout throughput fell below the configured production floor"
                )
            rollout_metrics["peak_vram_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            tensorboard_rollout_samples(
                tensorboard,
                "rollout_samples",
                selected,
                result.records,
                samples_per_prompt=args.samples_per_prompt,
                step=step,
                prompt_suffix=args.prompt_suffix,
                response_limits=result.response_limits,
            )
            tensorboard_scalars(
                tensorboard, "rollout", rollout_metrics, step
            )
            logger.log(type="rollout", step=step, **rollout_metrics)
            engine.release_cache()
            if not (args.fast_rollout or args.latent_thinking):
                use_packed_replay_attention(policy.causal_lm, enabled=True)
            cache_release_allocated_bytes = torch.cuda.memory_allocated(device)
            records, refresh_metrics = refresh_behavior_statistics(
                policy,
                critic,
                result.records,
                replay_token_budget=args.replay_token_budget,
                replay_max_trajectories=args.replay_max_trajectories,
                logit_chunk_tokens=args.logit_chunk_tokens,
                behavior_logprobs=args.behavior_logprobs,
            )
            refresh_metrics["cache_release_allocated_bytes"] = (
                cache_release_allocated_bytes
            )
            tensorboard_scalars(
                tensorboard, "behavior", refresh_metrics, step
            )
            epoch_start = 0
        else:
            records = pending_records
            epoch_start = pending_epoch
            pending_records = None
            pending_epoch = 0

        for epoch in range(epoch_start, args.ppo_epochs):
            if step >= args.steps:
                pending_records = records
                pending_epoch = epoch
                break
            torch.cuda.reset_peak_memory_stats(device)
            update_started = time.perf_counter()
            metrics = update_step(
                policy,
                critic,
                records,
                actor_optimizer,
                critic_optimizer,
                optimizer_minibatches=args.optimizer_minibatches,
                replay_token_budget=args.replay_token_budget,
                replay_max_trajectories=args.replay_max_trajectories,
                logit_chunk_tokens=args.logit_chunk_tokens,
                clip_low=args.clip_low,
                clip_high=args.clip_high,
                value_coefficient=args.value_coefficient,
                nextlat_horizon=args.nextlat_horizon,
                nextlat_samples=args.nextlat_samples,
                nextlat_mse_coefficient=args.nextlat_mse_coefficient,
                nextlat_kl_coefficient=args.nextlat_kl_coefficient,
                nextlat_kl_chunk_tokens=args.nextlat_kl_chunk_tokens,
                train_nextlat=args.train_nextlat,
                nextlat_trunk_balance=args.nextlat_trunk_balance,
                grad_clip_norm=args.gradient_clip_norm,
            )
            update_ended = time.perf_counter()
            metrics["allocated_vram_bytes"] = torch.cuda.memory_allocated(device)
            metrics["update_seconds"] = update_ended - update_started
            metrics.update(
                device_phase_metrics(
                    device_sampler,
                    started=update_started,
                    ended=update_ended,
                    power_floor=args.device_power_floor,
                )
            )
            step += 1
            if step % args.post_update_kl_interval == 0:
                metrics.update(
                    measure_post_update_behavior_kl(
                        policy,
                        records,
                        replay_token_budget=args.replay_token_budget,
                        replay_max_trajectories=args.replay_max_trajectories,
                        logit_chunk_tokens=args.logit_chunk_tokens,
                    )
                )
            metrics["peak_vram_bytes"] = torch.cuda.max_memory_allocated(device)
            metrics.update(corpus_progress(cursor, len(rows)))

            tensorboard_scalars(tensorboard, "train", metrics, step)
            logger.log(type="train", step=step, **metrics)
        if checkpoint_policy.due():
            save_checkpoint(
                output / "vapo_adapter_checkpoint.pt",
                policy,
                critic,
                actor_optimizer,
                critic_optimizer,
                step=step,
                cursor=cursor,
                warmup_step=completed_warmup,
                pending_records=pending_records,
                pending_epoch=pending_epoch,
                args=args,
                data_sha256=data_sha256,
                corpus_identity=corpus_identity,
            )
            checkpoint_policy.committed(
                (step, cursor, completed_warmup, pending_epoch)
            )

    save_checkpoint(
        output / "vapo_adapter_checkpoint.pt",
        policy,
        critic,
        actor_optimizer,
        critic_optimizer,
        step=step,
        cursor=cursor,
        warmup_step=completed_warmup,
        pending_records=pending_records,
        pending_epoch=pending_epoch,
        args=args,
        data_sha256=data_sha256,
        corpus_identity=corpus_identity,
    )
    tensorboard.close()


if __name__ == "__main__":
    main()

"""Memory-bounded Hugging Face policy support for standard-token VAPO.

This path deliberately does not reuse latent-thought slots or nano backbone
assumptions. MiniCPM5 stays a native ``LlamaForCausalLM`` with its own tokenizer,
chat template, attention implementation, and KV cache. Only LoRA adapters and a
small, separately optimized value head train.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any
import numpy as np
from scipy.signal import lfilter

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from postraining.hf_runtime import prepare_text_only_transformers_runtime




MINICPM5_MODEL_ID = "openbmb/MiniCPM5-1B"
MINICPM5_REVISION = "87179e5c1f455ef22e6223592d2d61351b525bfc"
MINICPM5_VOCAB_SIZE = 130_560
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("LoRA targets must be nonempty and unique")


class LoRALinear(nn.Module):
    """Frozen linear plus fp32-master low-rank update.

    CUDA callers run under bf16 autocast. Keeping the trainable matrices fp32
    avoids losing small AdamW updates while autocast still executes both small
    GEMMs in the model's compute dtype.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.rank = config.rank
        self.scaling = config.alpha / config.rank
        self.lora_a = nn.Parameter(
            torch.empty(config.rank, base.in_features, dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, config.rank, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: Tensor) -> Tensor:
        if (
            inputs.dtype != self.lora_a.dtype
            and not torch.is_autocast_enabled(inputs.device.type)
        ):
            raise RuntimeError(
                "mixed-dtype LoRA requires autocast so fp32 masters are cast "
                "only inside the adapter GEMMs"
            )
        update = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.scaling


def inject_lora(model: nn.Module, config: LoRAConfig) -> tuple[str, ...]:
    """Freeze ``model`` and replace every requested Llama projection exactly once."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in config.targets:
            replacements.append((name, module))
    if not replacements:
        raise ValueError(f"none of the LoRA targets exist: {config.targets}")
    found_targets = {name.rsplit(".", 1)[-1] for name, _ in replacements}
    missing = set(config.targets) - found_targets
    if missing:
        raise ValueError(f"missing LoRA target modules: {sorted(missing)}")
    for name, module in replacements:
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            child_name = name
            parent = model
        setattr(parent, child_name, LoRALinear(module, config))
    return tuple(name for name, _ in replacements)


def adapter_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    }


def load_adapter_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    }
    if parameters.keys() != state.keys():
        missing = sorted(parameters.keys() - state.keys())
        unexpected = sorted(state.keys() - parameters.keys())
        raise ValueError(
            f"adapter state mismatch: missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


class ValueHead(nn.Module):
    """Independent critic over frozen actor features; no parameters are shared."""

    def __init__(self, hidden_size: int, width: int = 256) -> None:
        super().__init__()
        if hidden_size < 1 or width < 1:
            raise ValueError("critic dimensions must be positive")
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6, dtype=torch.float32)
        self.input = nn.Linear(hidden_size, width, dtype=torch.float32)
        self.output = nn.Linear(width, 1, dtype=torch.float32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.output(F.silu(self.input(self.norm(hidden.float())))).squeeze(-1)




class MiniCPMVAPOPolicy(nn.Module):
    """Native MiniCPM causal LM plus LoRA actor and separate value head."""

    def __init__(
        self,
        causal_lm: Any,
        lora_config: LoRAConfig,
        *,
        critic_width: int = 256,
    ) -> None:
        super().__init__()
        self.causal_lm = causal_lm
        config: Any = causal_lm.config
        if getattr(config, "model_type", None) != "llama":
            raise ValueError("MiniCPM VAPO requires a standard LlamaForCausalLM")
        if int(config.vocab_size) != MINICPM5_VOCAB_SIZE:
            raise ValueError(
                f"unexpected MiniCPM vocabulary: {config.vocab_size} != "
                f"{MINICPM5_VOCAB_SIZE}"
            )
        self.lora_config = lora_config
        self.lora_modules = inject_lora(causal_lm, lora_config)
        self.critic = ValueHead(int(config.hidden_size), critic_width)
        self.model_id = MINICPM5_MODEL_ID
        self.revision = MINICPM5_REVISION
        self.causal_lm.config.use_cache = False

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = MINICPM5_MODEL_ID,
        revision: str = MINICPM5_REVISION,
        device: torch.device,
        lora_config: LoRAConfig,
        critic_width: int = 256,
        gradient_checkpointing: bool = True,
    ) -> tuple["MiniCPMVAPOPolicy", Any]:
        prepare_text_only_transformers_runtime()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        loaded: Any = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        causal_lm = loaded.to(device)
        policy = cls(causal_lm, lora_config, critic_width=critic_width).to(device)
        policy.model_id = model_id
        policy.revision = revision
        if len(tokenizer) != int(causal_lm.config.vocab_size):
            raise ValueError("tokenizer and checkpoint vocabularies differ")
        if tokenizer.chat_template is None:
            raise ValueError("MiniCPM tokenizer is missing its native chat template")
        if gradient_checkpointing:
            causal_lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        return policy, tokenizer

    @property
    def lm_head_weight(self) -> Tensor:
        weight = self.causal_lm.lm_head.weight
        if weight.requires_grad:
            raise RuntimeError("the 130,560-token output head must remain frozen")
        return weight

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        return (
            parameter
            for name, parameter in self.causal_lm.named_parameters()
            if name.endswith(("lora_a", "lora_b")) and parameter.requires_grad
        )


    def replay_hidden(
        self, input_ids: Tensor, attention_mask: Tensor | None
    ) -> Tensor:
        outputs = self.causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def cached_hidden(
        self,
        input_ids: Tensor,
        *,
        past_key_values: Any,
        cache_position: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        outputs = self.causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def logits(self, hidden: Tensor) -> Tensor:
        return self.causal_lm.lm_head(hidden)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema": "minicpm5_hf_vapo_adapter/v3",
            "model_id": self.model_id,
            "revision": self.revision,
            "lora_config": asdict(self.lora_config),
            "lora_modules": list(self.lora_modules),
            "adapter": adapter_state_dict(self.causal_lm),
            "critic": {
                name: tensor.detach().cpu()
                for name, tensor in self.critic.state_dict().items()
            },
        }


class _ChunkedFrozenHeadLogProbs(torch.autograd.Function):
    """Exact selected-token log-probabilities without retaining full logits."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden: Tensor,
        targets: Tensor,
        weight: Tensor,
        chunk_tokens: int,
    ) -> Tensor:
        if hidden.ndim != 2 or weight.ndim != 2:
            raise ValueError("hidden and output weight must be matrices")
        if targets.shape != hidden.shape[:1] or targets.dtype != torch.long:
            raise ValueError("targets must be int64 with one id per hidden row")
        if hidden.shape[1] != weight.shape[1]:
            raise ValueError("hidden width and output-head width differ")
        if weight.requires_grad:
            raise ValueError("chunked output head must be frozen")
        if chunk_tokens < 1:
            raise ValueError("logit chunk size must be positive")
        if targets.device.type == "cpu" and targets.numel() and (
            int(targets.min()) < 0 or int(targets.max()) >= weight.shape[0]
        ):
            raise ValueError("target token lies outside the vocabulary")

        ctx.save_for_backward(hidden, targets, weight)
        ctx.chunk_tokens = chunk_tokens
        result = torch.empty(hidden.shape[0], device=hidden.device, dtype=torch.float32)
        for start in range(0, hidden.shape[0], chunk_tokens):
            stop = min(start + chunk_tokens, hidden.shape[0])
            logits = F.linear(hidden[start:stop], weight).float()
            result[start:stop] = logits.gather(
                1, targets[start:stop, None]
            ).squeeze(1) - logits.logsumexp(dim=1)
        return result


    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor):
        if len(grad_outputs) != 1:
            raise RuntimeError("chunked log-probability backward expects one gradient")
        grad_output = grad_outputs[0]
        hidden, targets, weight = ctx.saved_tensors
        grad_hidden = torch.empty_like(hidden)
        for start in range(0, hidden.shape[0], ctx.chunk_tokens):
            stop = min(start + ctx.chunk_tokens, hidden.shape[0])
            logits = F.linear(hidden[start:stop], weight).float()
            probabilities = -logits.softmax(dim=1)
            del logits
            probabilities[
                torch.arange(stop - start, device=hidden.device),
                targets[start:stop],
            ] += 1.0
            probabilities.mul_(grad_output[start:stop, None].float())
            grad_hidden[start:stop] = F.linear(
                probabilities.to(weight.dtype), weight.transpose(0, 1)
            ).to(hidden.dtype)
        return grad_hidden, None, None, None


def chunked_frozen_head_logprobs(
    hidden: Tensor,
    targets: Tensor,
    weight: Tensor,
    *,
    chunk_tokens: int,
) -> Tensor:
    return _ChunkedFrozenHeadLogProbs.apply(hidden, targets, weight, chunk_tokens)


@dataclass(frozen=True)
class SamplingStats:
    scanned_vocabulary: int
    nucleus_mass_lower_bound: float


def _nucleus_membership(
    scaled_logits: Tensor,
    probabilities: Tensor,
    sampled: Tensor,
    top_p: float,
) -> Tensor:
    """Return membership in a deterministic descending-logit nucleus.

    Ties are ordered by ascending token id. A token belongs to the nucleus
    exactly when the probability mass preceding it in that total order is
    below ``top_p``.
    """
    vocabulary_ids = torch.arange(
        scaled_logits.shape[1], device=scaled_logits.device
    )
    thresholds = scaled_logits.gather(1, sampled[:, None])
    precedes = (scaled_logits > thresholds) | (
        (scaled_logits == thresholds)
        & (vocabulary_ids[None] < sampled[:, None])
    )
    preceding_mass = torch.where(
        precedes, probabilities, torch.zeros((), device=probabilities.device)
    ).sum(dim=-1)
    return preceding_mass < top_p


@torch.no_grad()
def exact_top_p_sample(
    logits: Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, SamplingStats]:
    """Sample an exact nucleus without sorting the 130,560-token vocabulary.

    Proposals come from the full temperature-scaled categorical distribution.
    Rejecting proposals outside the deterministic nucleus samples exactly from
    that distribution conditioned on the nucleus. Its acceptance probability
    is at least ``top_p``; at the configured 0.95 it needs about 1.053 proposals
    on average while replacing repeated adaptive top-k sorts with linear GPU
    reductions.
    """
    if logits.ndim != 2:
        raise ValueError("sampling logits must be [batch, vocabulary]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")

    policy_logits = logits.float()
    scaled_logits = policy_logits.div(temperature)
    probabilities = scaled_logits.softmax(dim=-1)
    sampled = torch.empty(
        logits.shape[0], dtype=torch.long, device=logits.device
    )
    pending = torch.ones(
        logits.shape[0], dtype=torch.bool, device=logits.device
    )
    rows = torch.arange(logits.shape[0], device=logits.device)
    while True:
        proposals = torch.multinomial(
            probabilities.index_select(0, rows),
            1,
            generator=generator,
        ).squeeze(1)
        accepted = (
            torch.ones_like(proposals, dtype=torch.bool)
            if top_p == 1.0
            else _nucleus_membership(
                scaled_logits.index_select(0, rows),
                probabilities.index_select(0, rows),
                proposals,
                top_p,
            )
        )
        accepted_rows = rows[accepted]
        sampled[accepted_rows] = proposals[accepted]
        pending[accepted_rows] = False
        if not bool(pending.any()):
            break
        rows = pending.nonzero(as_tuple=False).squeeze(1)
    selected = (
        policy_logits.gather(1, sampled[:, None]).squeeze(1)
        - policy_logits.logsumexp(dim=-1)
    )
    return (
        sampled,
        selected,
        SamplingStats(
            scanned_vocabulary=logits.shape[1],
            nucleus_mass_lower_bound=top_p,
        ),
    )

def _precompute_advantages(old_values: Tensor, correct: bool) -> Tensor:
    """Vectorized VAPO GAE for one terminal-reward trajectory."""
    values = old_values.detach().float().cpu().numpy()
    length = values.size
    if length < 1:
        raise ValueError("trajectory values cannot be empty")
    horizon = max(0.05 * length, min(float(length), 20.0))
    policy_lambda = min(max(1.0 - 1.0 / horizon, 0.0), 1.0)
    terminal_reward = 1.0 if correct else -1.0
    deltas = np.empty(length, dtype=np.float32)
    if length > 1:
        deltas[:-1] = values[1:] - values[:-1]
    deltas[-1] = terminal_reward - values[-1]
    reversed_advantages = np.asarray(
        lfilter([1.0], [1.0, -policy_lambda], deltas[::-1]),
        dtype=np.float32,
    )
    return torch.from_numpy(reversed_advantages[::-1].copy())


@dataclass(frozen=True)
class TrajectoryRecord:
    """Compact host replay record; no vocabulary-sized tensor is retained."""

    token_ids: Tensor
    prompt_length: int
    old_logprobs: Tensor
    advantages: Tensor
    correct: bool
    text: str

    def __post_init__(self) -> None:
        if self.token_ids.device.type != "cpu" or self.token_ids.dtype != torch.int32:
            raise ValueError("replay token ids must be CPU int32")
        if self.old_logprobs.device.type != "cpu" or self.old_logprobs.dtype != torch.float32:
            raise ValueError("replay log-probabilities must be CPU fp32")
        if self.advantages.device.type != "cpu" or self.advantages.dtype != torch.float32:
            raise ValueError("replay advantages must be CPU fp32")
        response_length = self.token_ids.numel() - self.prompt_length
        if self.prompt_length < 1 or response_length < 1:
            raise ValueError("trajectory must contain prompt and response tokens")
        if self.old_logprobs.numel() != response_length:
            raise ValueError("one old log-probability is required per response token")
        if self.advantages.numel() != response_length:
            raise ValueError("one advantage is required per response token")

    @property
    def response_length(self) -> int:
        return self.token_ids.numel() - self.prompt_length

    @property
    def input_length(self) -> int:
        return self.token_ids.numel() - 1

    @property
    def storage_bytes(self) -> int:
        tensors = (self.token_ids, self.old_logprobs, self.advantages)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @classmethod
    def from_device(
        cls,
        *,
        token_ids: Tensor,
        prompt_length: int,
        old_logprobs: Tensor,
        old_values: Tensor,
        correct: bool,
        text: str,
    ) -> "TrajectoryRecord":
        token_ids_cpu = token_ids.detach().to(device="cpu", dtype=torch.int32)
        response_length = token_ids_cpu.numel() - prompt_length
        if old_values.numel() != response_length:
            raise ValueError("one old value is required per response token")
        return cls(
            token_ids=token_ids_cpu,
            prompt_length=prompt_length,
            old_logprobs=old_logprobs.detach().to(device="cpu", dtype=torch.float32),
            advantages=_precompute_advantages(old_values, correct),
            correct=correct,
            text=text,
        )


def replay_storage_bytes(records: Sequence[TrajectoryRecord]) -> int:
    return sum(record.storage_bytes for record in records)


def plan_replay_microbatches(
    records: Sequence[TrajectoryRecord],
    order: Sequence[int],
    *,
    token_budget: int,
    max_trajectories: int,
) -> list[tuple[int, ...]]:
    """Length-bucket every trajectory exactly once under a padded-token budget."""

    if token_budget < 1 or max_trajectories < 1:
        raise ValueError("replay limits must be positive")
    if sorted(order) != list(range(len(records))):
        raise ValueError("replay order must be a permutation of every trajectory")
    oversized = [
        index for index, record in enumerate(records)
        if record.input_length > token_budget
    ]
    if oversized:
        largest = max(records[index].input_length for index in oversized)
        raise ValueError(
            f"replay trajectory length {largest} exceeds token budget "
            f"{token_budget}"
        )
    remaining = sorted(order, key=lambda index: records[index].input_length, reverse=True)
    plan: list[tuple[int, ...]] = []
    cursor = 0
    while cursor < len(remaining):
        batch = [remaining[cursor]]
        cursor += 1
        maximum = records[batch[0]].input_length
        while cursor < len(remaining) and len(batch) < max_trajectories:
            candidate = remaining[cursor]
            candidate_maximum = max(maximum, records[candidate].input_length)
            if candidate_maximum * (len(batch) + 1) > token_budget:
                break
            batch.append(candidate)
            maximum = candidate_maximum
            cursor += 1
        plan.append(tuple(batch))
    flattened = [index for microbatch in plan for index in microbatch]
    if sorted(flattened) != list(range(len(records))):
        raise RuntimeError("replay planner lost or duplicated a trajectory")
    return plan


@dataclass
class ReplayMicrobatch:
    input_ids: Tensor
    attention_mask: Tensor | None
    action_batch_indices: Tensor
    action_positions: Tensor
    response_state_mask: Tensor
    targets: Tensor
    old_logprobs: Tensor
    advantages: Tensor
    value_targets: Tensor
    positive_weights: Tensor

    @property
    def action_count(self) -> int:
        return self.targets.numel()




def collate_replay_microbatch(
    records: Sequence[TrajectoryRecord],
    indices: Sequence[int],
    *,
    pad_token_id: int,
    correct_denominator: int,
    device: torch.device,
) -> ReplayMicrobatch:
    if not indices:
        raise ValueError("cannot collate an empty replay microbatch")
    selected = [records[index] for index in indices]
    maximum = max(record.input_length for record in selected)
    pin = device.type == "cuda"
    input_ids = torch.full(
        (len(selected), maximum),
        pad_token_id,
        dtype=torch.long,
        pin_memory=pin,
    )
    attention_mask = torch.zeros(
        (len(selected), maximum), dtype=torch.bool, pin_memory=pin
    )
    response_state_mask = torch.zeros(
        (len(selected), maximum), dtype=torch.bool, pin_memory=pin
    )
    batch_indices: list[Tensor] = []
    positions: list[Tensor] = []
    targets: list[Tensor] = []
    old_logprobs: list[Tensor] = []
    advantages: list[Tensor] = []
    value_targets: list[Tensor] = []
    positive_weights: list[Tensor] = []

    for row, record in enumerate(selected):
        length = record.input_length
        input_ids[row, :length].copy_(record.token_ids[:-1])
        attention_mask[row, :length] = True
        action_start = record.prompt_length - 1
        action_stop = action_start + record.response_length
        batch_indices.append(torch.full((record.response_length,), row, dtype=torch.long))
        positions.append(torch.arange(action_start, action_stop, dtype=torch.long))
        response_state_mask[row, action_start:action_stop] = True
        targets.append(record.token_ids[record.prompt_length:].long())
        old_logprobs.append(record.old_logprobs)
        advantages.append(record.advantages)
        value_targets.append(
            torch.full(
                (record.response_length,),
                1.0 if record.correct else -1.0,
                dtype=torch.float32,
            )
        )
        positive_weight = (
            1.0 / (correct_denominator * record.response_length)
            if record.correct and correct_denominator
            else 0.0
        )
        positive_weights.append(
            torch.full((record.response_length,), positive_weight, dtype=torch.float32)
        )

    all_unpadded = all(record.input_length == maximum for record in selected)

    def transfer(tensor: Tensor) -> Tensor:
        return tensor.to(device, non_blocking=pin)

    return ReplayMicrobatch(
        input_ids=transfer(input_ids),
        attention_mask=None if all_unpadded else transfer(attention_mask),
        action_batch_indices=transfer(torch.cat(batch_indices)),
        action_positions=transfer(torch.cat(positions)),
        response_state_mask=transfer(response_state_mask),
        targets=transfer(torch.cat(targets)),
        old_logprobs=transfer(torch.cat(old_logprobs)),
        advantages=transfer(torch.cat(advantages)),
        value_targets=transfer(torch.cat(value_targets)),
        positive_weights=transfer(torch.cat(positive_weights)),
    )


class StaticCachePool:
    """One reusable HF static cache with an explicit reset boundary per group."""

    def __init__(self, factory: Callable[[], Any], batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("cache batch size must be positive")
        self._factory = factory
        self._batch_size = batch_size
        self._cache: Any | None = None
        self.reset_count = 0

    def acquire(self, batch_size: int) -> Any:
        if batch_size != self._batch_size:
            raise ValueError("cache pool batch size changed")
        if self._cache is None:
            self._cache = self._factory()
        else:
            self._cache.reset()
            self.reset_count += 1
        return self._cache

    def clear(self) -> None:
        """Release cache tensors before replay needs the rollout VRAM."""
        self._cache = None

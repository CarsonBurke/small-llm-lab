"""Feature-cached critic pretraining for MiniCPM VAPO checkpoints.

The transformer is replayed exactly once per corpus trajectory.  Stratified
response-state features are then kept as BF16 CPU tensors while the independent
FP32 value head is optimized and evaluated in bounded device batches.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
from torch import Tensor
from checkpointing import atomic_torch_save

from postraining.vapo.policy import (
    VAPOCritic,
    ValueHead,
)
from postraining.vapo.model.lora import (
    LoRAConfig,
    load_adapter_state_dict,
)


CORPUS_SCHEMA = "minicpm_critic_corpus/v1"
PRETRAINING_SCHEMA = "minicpm_critic_pretraining/v1"
POLICY_SCHEMA = "minicpm5_vapo_adapter/v6"
POSITION_BUCKET_NAMES = ("0-25%", "25-50%", "50-75%", "75-100%")
_REQUIRED_CORPUS_FIELDS = {
    "schema",
    "model_id",
    "revision",
    "source_checkpoint",
    "source_checkpoint_step",
    "data_sha256",
    "cursor_start",
    "cursor_end",
    "prompt_tokens",
    "max_new_tokens",
    "samples_per_prompt",
    "seed",
    "records",
}


@dataclass(frozen=True)
class SampledPosition:
    position: int
    weight: float


@dataclass(frozen=True)
class RecordFeatureSpec:
    positions: tuple[int, ...]
    weights: tuple[float, ...]
    destination_start: int


@dataclass
class CachedFeatureSet:
    features: Tensor
    targets: Tensor
    weights: Tensor
    relative_positions: Tensor
    prompt_indices: Tensor

    def __post_init__(self) -> None:
        count = self.features.shape[0]
        if self.features.ndim != 2 or any(
            tensor.ndim != 1 or tensor.shape[0] != count
            for tensor in (
                self.targets,
                self.weights,
                self.relative_positions,
                self.prompt_indices,
            )
        ):
            raise ValueError("cached feature tensors have inconsistent shapes")
        if self.features.device.type != "cpu" or self.features.dtype != torch.bfloat16:
            raise ValueError("features must be a CPU BF16 matrix")

    def __len__(self) -> int:
        return self.features.shape[0]


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]


@dataclass
class TrainingResult:
    best_epoch: int
    epochs_completed: int
    best_validation_loss: float
    best_metrics: dict[str, Any]
    optimizer_state: dict[str, Any]
    history: list[dict[str, Any]]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def validate_critic_corpus(corpus: Any) -> dict[str, Any]:
    """Validate the complete portable corpus contract and return the mapping."""
    if not isinstance(corpus, dict):
        raise ValueError("critic corpus must be a dictionary")
    missing = sorted(_REQUIRED_CORPUS_FIELDS - corpus.keys())
    if missing:
        raise ValueError("critic corpus is missing fields: " + ", ".join(missing))
    if corpus["schema"] != CORPUS_SCHEMA:
        raise ValueError(f"unsupported critic corpus schema: {corpus['schema']!r}")
    for name in ("model_id", "revision", "source_checkpoint"):
        if not isinstance(corpus[name], str) or not corpus[name]:
            raise ValueError(f"critic corpus {name} must be a nonempty string")
    digest = corpus["data_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("critic corpus data_sha256 must be a lowercase SHA-256 digest")
    for name in (
        "source_checkpoint_step",
        "cursor_start",
        "cursor_end",
        "prompt_tokens",
        "max_new_tokens",
        "samples_per_prompt",
        "seed",
    ):
        value = corpus[name]
        if type(value) is not int:
            raise ValueError(f"critic corpus {name} must be an integer")
    if corpus["source_checkpoint_step"] < 0 or corpus["cursor_start"] < 0:
        raise ValueError("critic corpus source step and cursor must be nonnegative")
    if corpus["cursor_end"] <= corpus["cursor_start"]:
        raise ValueError("critic corpus cursor_end must be greater than cursor_start")
    for name in ("prompt_tokens", "max_new_tokens", "samples_per_prompt"):
        if corpus[name] < 1:
            raise ValueError(f"critic corpus {name} must be positive")

    records = corpus["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("critic corpus records must be a nonempty list")
    prompt_counts: Counter[int] = Counter()
    for record_index, record in enumerate(records):
        prefix = f"critic corpus record {record_index}"
        if not isinstance(record, dict):
            raise ValueError(f"{prefix} must be a dictionary")
        missing_record = {"prompt_index", "prompt_ids", "response_ids", "correct"} - record.keys()
        if missing_record:
            raise ValueError(
                f"{prefix} is missing fields: " + ", ".join(sorted(missing_record))
            )
        if type(record["prompt_index"]) is not int or record["prompt_index"] < 0:
            raise ValueError(f"{prefix} prompt_index must be a nonnegative integer")
        if type(record["correct"]) is not bool:
            raise ValueError(f"{prefix} correct must be a boolean")
        for name in ("prompt_ids", "response_ids"):
            token_ids = record[name]
            if (
                not isinstance(token_ids, Tensor)
                or token_ids.device.type != "cpu"
                or token_ids.dtype != torch.int32
                or token_ids.ndim != 1
                or token_ids.numel() < 1
            ):
                raise ValueError(f"{prefix} {name} must be a nonempty CPU int32 vector")
            if bool((token_ids < 0).any()):
                raise ValueError(f"{prefix} {name} contains a negative token id")
        if record["prompt_ids"].numel() > corpus["prompt_tokens"]:
            raise ValueError(f"{prefix} exceeds prompt_tokens")
        if record["response_ids"].numel() > corpus["max_new_tokens"]:
            raise ValueError(f"{prefix} exceeds max_new_tokens")
        prompt_counts[record["prompt_index"]] += 1
    wrong_counts = {
        prompt_index: count
        for prompt_index, count in prompt_counts.items()
        if count != corpus["samples_per_prompt"]
    }
    if wrong_counts:
        example = next(iter(sorted(wrong_counts.items())))
        raise ValueError(
            "critic corpus prompt sample counts differ from samples_per_prompt; "
            f"prompt {example[0]} has {example[1]}"
        )
    if len(prompt_counts) < 2:
        raise ValueError("critic corpus needs at least two prompts for held-out validation")
    consumed_prompts = corpus["cursor_end"] - corpus["cursor_start"]
    if consumed_prompts != len(prompt_counts):
        raise ValueError(
            "critic corpus cursor range does not match its distinct prompt count"
        )
    return corpus


def validate_corpus_checkpoint_identity(
    corpus: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    expected_source_step: int | None = None,
) -> None:
    """Reject a corpus not generated by the exact source training state."""
    policy = checkpoint.get("policy")
    if not isinstance(policy, Mapping) or policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("source checkpoint does not contain a current MiniCPM VAPO policy")
    corpus_source = Path(corpus["source_checkpoint"]).expanduser().resolve()
    supplied_source = Path(checkpoint_path).expanduser().resolve()
    if corpus_source != supplied_source:
        raise ValueError(
            f"critic corpus source checkpoint is {corpus_source}, not {supplied_source}"
        )
    source_step = checkpoint.get("step")
    if type(source_step) is not int:
        raise ValueError("source checkpoint step is missing or invalid")
    if source_step != corpus["source_checkpoint_step"]:
        raise ValueError(
            "critic corpus source step differs from source checkpoint: "
            f"{corpus['source_checkpoint_step']} != {source_step}"
        )
    if expected_source_step is not None and source_step != expected_source_step:
        raise ValueError(
            f"source checkpoint step {source_step} differs from requested step "
            f"{expected_source_step}"
        )
    critic = policy.get("critic")
    if not isinstance(critic, Mapping):
        raise ValueError("source checkpoint critic payload is missing")
    if critic.get("model_id") != corpus["model_id"]:
        raise ValueError("critic corpus model_id differs from source checkpoint")
    if critic.get("revision") != corpus["revision"]:
        raise ValueError("critic corpus revision differs from source checkpoint")
    if checkpoint.get("data_sha256") != corpus["data_sha256"]:
        raise ValueError("critic corpus dataset identity differs from source checkpoint")
    source_cursor = checkpoint.get("cursor")
    if type(source_cursor) is not int or source_cursor != corpus["cursor_start"]:
        raise ValueError(
            "critic corpus cursor_start differs from source checkpoint cursor: "
            f"{corpus['cursor_start']} != {source_cursor!r}"
        )
    if checkpoint.get("pending_records") is not None or checkpoint.get(
        "pending_epoch", 0
    ) != 0:
        raise ValueError(
            "source checkpoint must be at a replay boundary before advancing cursor"
        )


def stratified_response_position_samples(
    response_length: int, sample_count: int, seed: int
) -> tuple[SampledPosition, ...]:
    """Choose one deterministic position per equal-width response stratum.

    The returned weight is the number of original response states represented by
    the draw. Consequently weighted objectives are exact when every state is
    retained and are an unbiased stratified estimate otherwise.
    """
    if response_length < 1 or sample_count < 1:
        raise ValueError("response length and sample count must be positive")
    count = min(response_length, sample_count)
    if count == response_length:
        return tuple(SampledPosition(position, 1.0) for position in range(count))
    generator = random.Random(seed)
    sampled: list[SampledPosition] = []
    for stratum in range(count):
        start = stratum * response_length // count
        stop = (stratum + 1) * response_length // count
        position = generator.randrange(start, stop)
        sampled.append(SampledPosition(position, float(stop - start)))
    return tuple(sampled)


def deterministic_stratified_positions(
    response_length: int, sample_count: int, seed: int
) -> tuple[int, ...]:
    return tuple(
        sample.position
        for sample in stratified_response_position_samples(
            response_length, sample_count, seed
        )
    )


def split_records_by_prompt(
    records: Sequence[Mapping[str, Any]],
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split whole prompt groups deterministically, never trajectories."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must lie strictly between zero and one")
    prompt_indices = sorted({int(record["prompt_index"]) for record in records})
    if len(prompt_indices) < 2:
        raise ValueError("at least two prompt groups are required")
    random.Random(seed).shuffle(prompt_indices)
    validation_count = min(
        len(prompt_indices) - 1,
        max(1, round(len(prompt_indices) * validation_fraction)),
    )
    validation_prompts = set(prompt_indices[:validation_count])
    train = [record for record in records if record["prompt_index"] not in validation_prompts]
    validation = [record for record in records if record["prompt_index"] in validation_prompts]
    if {record["prompt_index"] for record in train} & validation_prompts:
        raise AssertionError("prompt-disjoint split invariant failed")
    return train, validation


def _metric_inputs(
    predictions: Tensor, targets: Tensor, weights: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    predictions = torch.as_tensor(predictions, dtype=torch.float64).flatten()
    targets = torch.as_tensor(targets, dtype=torch.float64).flatten()
    weights = torch.as_tensor(weights, dtype=torch.float64).flatten()
    if not (predictions.shape == targets.shape == weights.shape) or predictions.numel() == 0:
        raise ValueError("metric inputs must be nonempty vectors with equal shapes")
    if not bool(torch.isfinite(predictions).all()):
        raise ValueError("predictions must be finite")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("targets must be finite")
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("weights must be finite and positive")
    return predictions, targets, weights


def weighted_mse(predictions: Tensor, targets: Tensor, weights: Tensor) -> float:
    predictions, targets, weights = _metric_inputs(predictions, targets, weights)
    return float((weights * (predictions - targets).square()).sum() / weights.sum())


def weighted_explained_variance(
    predictions: Tensor, targets: Tensor, weights: Tensor
) -> float:
    """Return 1 - weighted Var(target - prediction) / weighted Var(target)."""
    predictions, targets, weights = _metric_inputs(predictions, targets, weights)
    total_weight = weights.sum()
    target_mean = (weights * targets).sum() / total_weight
    target_variance = (weights * (targets - target_mean).square()).sum() / total_weight
    if float(target_variance) <= 0.0:
        return float("nan")
    residual = targets - predictions
    residual_mean = (weights * residual).sum() / total_weight
    residual_variance = (
        weights * (residual - residual_mean).square()
    ).sum() / total_weight
    return float(1.0 - residual_variance / target_variance)


def _weighted_mean(values: Tensor, weights: Tensor) -> float:
    return float((values.double() * weights.double()).sum() / weights.double().sum())


def position_bucket_metrics(
    predictions: Tensor,
    targets: Tensor,
    weights: Tensor,
    relative_positions: Tensor,
) -> dict[str, dict[str, float | int]]:
    predictions, targets, weights = _metric_inputs(predictions, targets, weights)
    relative_positions = torch.as_tensor(relative_positions, dtype=torch.float64).flatten()
    if relative_positions.shape != predictions.shape:
        raise ValueError("relative positions must align with metric inputs")
    if (
        not bool(torch.isfinite(relative_positions).all())
        or bool((relative_positions < 0).any())
        or bool((relative_positions > 1).any())
    ):
        raise ValueError("relative positions must be finite and lie in [0, 1]")
    bucket_ids = torch.clamp((relative_positions * 4).floor().long(), max=3)
    result: dict[str, dict[str, float | int]] = {}
    for bucket_index, name in enumerate(POSITION_BUCKET_NAMES):
        selected = bucket_ids == bucket_index
        if not bool(selected.any()):
            result[name] = {
                "count": 0,
                "weight": 0.0,
                "weighted_mse": float("nan"),
                "explained_variance": float("nan"),
                "target_mean": float("nan"),
                "prediction_mean": float("nan"),
                "class_margin": float("nan"),
            }
            continue
        bucket_predictions = predictions[selected]
        bucket_targets = targets[selected]
        bucket_weights = weights[selected]
        positive = bucket_targets > 0
        negative = bucket_targets < 0
        class_margin = float("nan")
        if bool(positive.any()) and bool(negative.any()):
            class_margin = _weighted_mean(
                bucket_predictions[positive], bucket_weights[positive]
            ) - _weighted_mean(bucket_predictions[negative], bucket_weights[negative])
        result[name] = {
            "count": int(selected.sum()),
            "weight": float(bucket_weights.sum()),
            "weighted_mse": weighted_mse(
                bucket_predictions, bucket_targets, bucket_weights
            ),
            "explained_variance": weighted_explained_variance(
                bucket_predictions, bucket_targets, bucket_weights
            ),
            "target_mean": _weighted_mean(bucket_targets, bucket_weights),
            "prediction_mean": _weighted_mean(bucket_predictions, bucket_weights),
            "class_margin": class_margin,
        }
    return result


def critic_metrics(
    predictions: Tensor,
    targets: Tensor,
    weights: Tensor,
    relative_positions: Tensor,
) -> dict[str, Any]:
    predictions, targets, weights = _metric_inputs(predictions, targets, weights)
    target_mean = _weighted_mean(targets, weights)
    target_variance = _weighted_mean((targets - target_mean).square(), weights)
    positive = targets > 0
    negative = targets < 0
    class_margin = float("nan")
    if bool(positive.any()) and bool(negative.any()):
        class_margin = _weighted_mean(predictions[positive], weights[positive]) - _weighted_mean(
            predictions[negative], weights[negative]
        )
    return {
        "count": predictions.numel(),
        "weight": float(weights.sum()),
        "weighted_mse": weighted_mse(predictions, targets, weights),
        "explained_variance": weighted_explained_variance(
            predictions, targets, weights
        ),
        "target_mean": target_mean,
        "target_variance": target_variance,
        "prediction_mean": _weighted_mean(predictions, weights),
        "class_margin": class_margin,
        "buckets": position_bucket_metrics(
            predictions, targets, weights, relative_positions
        ),
    }


def evaluate_acceptance_gate(
    metrics: Mapping[str, Any],
    *,
    min_explained_variance: float = 0.0,
    min_bucket_explained_variance: float = -0.05,
    min_class_margin: float = 0.0,
) -> GateResult:
    """Require useful aggregate signal without a collapsing response quartile."""
    reasons: list[str] = []
    explained_variance = float(metrics.get("explained_variance", float("nan")))
    loss = float(metrics.get("weighted_mse", float("nan")))
    target_variance = float(metrics.get("target_variance", float("nan")))
    class_margin = float(metrics.get("class_margin", float("nan")))
    if not math.isfinite(explained_variance) or explained_variance <= min_explained_variance:
        reasons.append(
            f"held-out explained variance {explained_variance!r} is not above "
            f"{min_explained_variance}"
        )
    if not math.isfinite(loss) or not math.isfinite(target_variance) or loss >= target_variance:
        reasons.append("held-out weighted MSE does not beat the constant-mean baseline")
    if not math.isfinite(class_margin) or class_margin <= min_class_margin:
        reasons.append(
            f"held-out class margin {class_margin!r} is not above {min_class_margin}"
        )
    buckets = metrics.get("buckets")
    if not isinstance(buckets, Mapping):
        reasons.append("held-out position-bucket metrics are missing")
    else:
        for name in POSITION_BUCKET_NAMES:
            bucket = buckets.get(name)
            bucket_ev = (
                float(bucket.get("explained_variance", float("nan")))
                if isinstance(bucket, Mapping)
                else float("nan")
            )
            if not math.isfinite(bucket_ev) or bucket_ev < min_bucket_explained_variance:
                reasons.append(
                    f"held-out bucket {name} explained variance {bucket_ev!r} is below "
                    f"{min_bucket_explained_variance}"
                )
    return GateResult(not reasons, tuple(reasons))


def plan_extraction_microbatches(
    sequence_lengths: Sequence[int], token_budget: int
) -> list[list[int]]:
    """Pack similar lengths while bounding padded tokens in every replay."""
    if token_budget < 1:
        raise ValueError("feature extraction token budget must be positive")
    if any(length < 1 for length in sequence_lengths):
        raise ValueError("replay sequence lengths must be positive")
    too_long = [length for length in sequence_lengths if length > token_budget]
    if too_long:
        raise ValueError(
            f"feature token budget {token_budget} is below a trajectory length "
            f"{max(too_long)}"
        )
    ordered = sorted(range(len(sequence_lengths)), key=lambda index: (sequence_lengths[index], index))
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in ordered:
        length = sequence_lengths[index]
        proposed_max = max(current_max, length)
        if current and (len(current) + 1) * proposed_max > token_budget:
            batches.append(current)
            current = []
            current_max = 0
        current.append(index)
        current_max = max(current_max, length)
    if current:
        batches.append(current)
    return batches


def _record_seed(seed: int, prompt_index: int, record_index: int) -> int:
    material = f"{seed}:{prompt_index}:{record_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def _feature_specs(
    records: Sequence[Mapping[str, Any]], samples_per_trajectory: int, seed: int
) -> tuple[list[RecordFeatureSpec], int]:
    specs: list[RecordFeatureSpec] = []
    destination = 0
    for record_index, record in enumerate(records):
        samples = stratified_response_position_samples(
            int(record["response_ids"].numel()),
            samples_per_trajectory,
            _record_seed(seed, int(record["prompt_index"]), record_index),
        )
        specs.append(
            RecordFeatureSpec(
                tuple(sample.position for sample in samples),
                tuple(sample.weight for sample in samples),
                destination,
            )
        )
        destination += len(samples)
    return specs, destination


def extract_cached_features(
    policy: VAPOCritic,
    records: Sequence[Mapping[str, Any]],
    *,
    samples_per_trajectory: int,
    token_budget: int,
    seed: int,
    device: torch.device,
) -> CachedFeatureSet:
    """Replay each trajectory once and materialize only sampled action states."""
    if not records:
        raise ValueError("cannot extract an empty feature split")
    specs, feature_count = _feature_specs(records, samples_per_trajectory, seed)
    sequence_lengths = [
        int(record["prompt_ids"].numel() + record["response_ids"].numel() - 1)
        for record in records
    ]
    plan = plan_extraction_microbatches(sequence_lengths, token_budget)
    features: Tensor | None = None
    targets = torch.empty(feature_count, dtype=torch.float32)
    weights = torch.empty(feature_count, dtype=torch.float32)
    relative_positions = torch.empty(feature_count, dtype=torch.float32)
    prompt_indices = torch.empty(feature_count, dtype=torch.int64)

    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        for batch_indices in plan:
            max_length = max(sequence_lengths[index] for index in batch_indices)
            input_ids = torch.zeros(
                (len(batch_indices), max_length), dtype=torch.long
            )
            attention_mask = torch.zeros(
                (len(batch_indices), max_length), dtype=torch.bool
            )
            for row, record_index in enumerate(batch_indices):
                record = records[record_index]
                full_ids = torch.cat((record["prompt_ids"], record["response_ids"])).long()
                replay_ids = full_ids[:-1]
                input_ids[row, : replay_ids.numel()] = replay_ids
                attention_mask[row, : replay_ids.numel()] = True
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                hidden = policy.replay_hidden(input_ids, attention_mask)
            if features is None:
                features = torch.empty(
                    (feature_count, hidden.shape[-1]),
                    dtype=torch.bfloat16,
                    device="cpu",
                )
            selected_by_record: list[Tensor] = []
            for row, record_index in enumerate(batch_indices):
                record = records[record_index]
                spec = specs[record_index]
                prompt_length = int(record["prompt_ids"].numel())
                action_indices = torch.tensor(
                    [prompt_length - 1 + position for position in spec.positions],
                    device=device,
                    dtype=torch.long,
                )
                selected_by_record.append(
                    hidden[row].index_select(0, action_indices)
                )
            host_hidden = torch.cat(selected_by_record).to(
                device="cpu", dtype=torch.bfloat16
            )
            source_start = 0
            for record_index in batch_indices:
                record = records[record_index]
                spec = specs[record_index]
                count = len(spec.positions)
                destination = slice(spec.destination_start, spec.destination_start + count)
                features[destination].copy_(
                    host_hidden[source_start : source_start + count]
                )
                source_start += count
                targets[destination] = 1.0 if record["correct"] else -1.0
                weights[destination] = torch.tensor(spec.weights, dtype=torch.float32)
                response_length = int(record["response_ids"].numel())
                relative_positions[destination] = torch.tensor(
                    [(position + 0.5) / response_length for position in spec.positions],
                    dtype=torch.float32,
                )
                prompt_indices[destination] = int(record["prompt_index"])
            del hidden, host_hidden, selected_by_record, input_ids, attention_mask
    if features is None:
        raise AssertionError("feature extraction produced no batches")
    return CachedFeatureSet(
        features, targets, weights, relative_positions, prompt_indices
    )


def evaluate_value_head(
    head: ValueHead,
    cached: CachedFeatureSet,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("head batch size must be positive")
    predictions = torch.empty(len(cached), dtype=torch.float32)
    head.eval()
    with torch.no_grad():
        for start in range(0, len(cached), batch_size):
            stop = min(start + batch_size, len(cached))
            batch = cached.features[start:stop].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = head(batch)
            predictions[start:stop].copy_(output.float().cpu())
    return critic_metrics(
        predictions, cached.targets, cached.weights, cached.relative_positions
    )


def _to_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return deepcopy(value)


def train_value_head(
    head: ValueHead,
    train: CachedFeatureSet,
    validation: CachedFeatureSet,
    *,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    min_improvement: float,
    seed: int,
    log_epoch: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingResult:
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("critic optimizer hyperparameters are invalid")
    if batch_size < 1 or max_epochs < 1 or min_epochs < 1 or patience < 1:
        raise ValueError("critic training counts must be positive")
    if min_epochs > max_epochs:
        raise ValueError("minimum epochs cannot exceed maximum epochs")
    if min_improvement < 0:
        raise ValueError("minimum improvement must be nonnegative")

    head.to(device)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch)
        order = torch.randperm(len(train), generator=generator)
        head.train()
        training_error_sum = torch.zeros((), dtype=torch.float64, device=device)
        training_weight_sum = torch.zeros((), dtype=torch.float64, device=device)
        optimizer_steps = 0
        for start in range(0, len(train), batch_size):
            indices = order[start : start + batch_size]
            features = train.features.index_select(0, indices).to(device, non_blocking=True)
            targets = train.targets.index_select(0, indices).to(device, non_blocking=True)
            weights = train.weights.index_select(0, indices).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                predictions = head(features)
                weighted_errors = weights * (predictions.float() - targets).square()
                loss = weighted_errors.sum() / weights.sum()
            training_error_sum += weighted_errors.detach().double().sum()
            training_weight_sum += weights.detach().double().sum()
            loss.backward()
            optimizer.step()
            optimizer_steps += 1

        train_metrics = {
            "count": len(train),
            "weight": float(training_weight_sum),
            "weighted_mse": float(training_error_sum / training_weight_sum),
            "optimizer_steps": optimizer_steps,
        }
        validation_metrics = evaluate_value_head(
            head, validation, batch_size=batch_size, device=device
        )
        epoch_metrics = {
            "event": "epoch",
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_metrics)
        if log_epoch is not None:
            log_epoch(epoch_metrics)
        validation_loss = float(validation_metrics["weighted_mse"])
        if validation_loss < best_loss - min_improvement:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _to_cpu(head.state_dict())
            best_optimizer_state = _to_cpu(optimizer.state_dict())
            best_metrics = deepcopy(validation_metrics)
            epochs_without_improvement = 0
        elif epoch >= min_epochs:
            epochs_without_improvement += 1
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            break

    if best_state is None or best_optimizer_state is None or best_metrics is None:
        raise RuntimeError("critic training did not produce finite validation metrics")
    head.load_state_dict(best_state, strict=True)
    optimizer.load_state_dict(best_optimizer_state)
    return TrainingResult(
        best_epoch=best_epoch,
        epochs_completed=len(history),
        best_validation_loss=best_loss,
        best_metrics=best_metrics,
        optimizer_state=_to_cpu(optimizer.state_dict()),
        history=history,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
        stream.flush()




def build_output_checkpoint(
    source: Mapping[str, Any],
    critic_state: Mapping[str, Tensor],
    critic_optimizer_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    cursor_end: int,
) -> dict[str, Any]:
    """Replace the value head and reset the changed critic optimizer state."""
    source_optimizer = source.get("critic_optimizer")
    if not isinstance(source_optimizer, Mapping):
        raise ValueError("source checkpoint is missing the critic optimizer")
    source_groups = source_optimizer.get("param_groups")
    if not isinstance(source_groups, list) or any(
        not isinstance(group, Mapping) for group in source_groups
    ):
        raise ValueError("critic optimizer parameter groups are invalid")
    reset_optimizer = {
        "state": {},
        "param_groups": deepcopy(source_groups),
    }

    output = dict(source)
    policy = dict(source["policy"])
    critic = dict(policy["critic"])
    critic["value_head"] = _to_cpu(dict(critic_state))
    policy["critic"] = critic
    output["policy"] = policy
    output["critic_optimizer"] = reset_optimizer
    output["cursor"] = cursor_end
    output["critic_pretraining"] = deepcopy(dict(metadata))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--source-step", type=int)
    parser.add_argument("--feature-samples-per-trajectory", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--feature-token-budget", type=int, default=32_768)
    parser.add_argument("--head-batch-size", type=int, default=8_192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-improvement", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gate-min-explained-variance", type=float, default=0.0)
    parser.add_argument(
        "--gate-min-bucket-explained-variance", type=float, default=-0.05
    )
    parser.add_argument("--gate-min-class-margin", type=float, default=0.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.source_step is not None and args.source_step < 0:
        raise ValueError("source step must be nonnegative")
    for name in (
        "feature_samples_per_trajectory",
        "feature_token_budget",
        "head_batch_size",
        "max_epochs",
        "min_epochs",
        "patience",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.min_epochs > args.max_epochs:
        raise ValueError("--min-epochs cannot exceed --max-epochs")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must lie strictly between zero and one")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.min_improvement < 0:
        raise ValueError("optimizer settings are invalid")
    for name in (
        "gate_min_explained_variance",
        "gate_min_bucket_explained_variance",
        "gate_min_class_margin",
    ):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    source = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if source == output:
        raise ValueError("output checkpoint must not overwrite the source checkpoint")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("MiniCPM critic pretraining requires CUDA BF16 execution")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    corpus_path = Path(args.corpus).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    metrics_path = (
        Path(args.metrics).expanduser().resolve()
        if args.metrics
        else output_path.with_suffix(output_path.suffix + ".metrics.jsonl")
    )
    if metrics_path in {checkpoint_path, output_path}:
        raise ValueError("metrics path must differ from checkpoint paths")
    corpus = validate_critic_corpus(
        torch.load(corpus_path, map_location="cpu", weights_only=True)
    )
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(source, dict):
        raise ValueError("source VAPO checkpoint must be a dictionary")
    validate_corpus_checkpoint_identity(
        corpus,
        source,
        checkpoint_path,
        expected_source_step=args.source_step,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")
    policy_payload = source["policy"]
    critic_payload = policy_payload.get("critic")
    if not isinstance(critic_payload, Mapping):
        raise ValueError("source checkpoint critic payload is missing")
    lora_payload = critic_payload.get("lora_config")
    if not isinstance(lora_payload, Mapping):
        raise ValueError("source checkpoint critic LoRA configuration is missing")
    lora_config = LoRAConfig(
        rank=int(lora_payload["rank"]),
        alpha=float(lora_payload["alpha"]),
        initialization=lora_payload.get("initialization", "standard"),
        targets=tuple(lora_payload["targets"]),
    )
    critic_state = critic_payload.get("value_head")
    if not isinstance(critic_state, Mapping) or "input.weight" not in critic_state:
        raise ValueError("source checkpoint value head is missing")
    critic_width = int(critic_state["input.weight"].shape[0])
    critic_model = VAPOCritic.from_family("minicpm5", 
        model_id=corpus["model_id"],
        revision=corpus["revision"],
        device=device,
        lora_config=lora_config,
        critic_width=critic_width,
        nextlat_projection_factor=float(
            critic_payload["nextlat_projection_factor"]
        ),
        gradient_checkpointing=False,
    )
    load_adapter_state_dict(
        critic_model.causal_lm, critic_payload["adapter"]
    )
    critic_model.value_head.load_state_dict(critic_state, strict=True)

    train_records, validation_records = split_records_by_prompt(
        corpus["records"], args.validation_fraction, args.seed
    )
    train_prompts = {record["prompt_index"] for record in train_records}
    validation_prompts = {record["prompt_index"] for record in validation_records}
    append_jsonl(
        metrics_path,
        {
            "event": "split",
            "train_prompts": len(train_prompts),
            "validation_prompts": len(validation_prompts),
            "train_trajectories": len(train_records),
            "validation_trajectories": len(validation_records),
        },
    )
    train_cache = extract_cached_features(
        critic_model,
        train_records,
        samples_per_trajectory=args.feature_samples_per_trajectory,
        token_budget=args.feature_token_budget,
        seed=args.seed,
        device=device,
    )
    validation_cache = extract_cached_features(
        critic_model,
        validation_records,
        samples_per_trajectory=args.feature_samples_per_trajectory,
        token_budget=args.feature_token_budget,
        seed=args.seed + 1,
        device=device,
    )
    append_jsonl(
        metrics_path,
        {
            "event": "features",
            "train_states": len(train_cache),
            "validation_states": len(validation_cache),
            "hidden_size": train_cache.features.shape[1],
            "dtype": str(train_cache.features.dtype),
        },
    )

    critic = critic_model.value_head
    del critic_model
    gc.collect()
    torch.cuda.empty_cache()
    result = train_value_head(
        critic,
        train_cache,
        validation_cache,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.head_batch_size,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        min_improvement=args.min_improvement,
        seed=args.seed,
        log_epoch=lambda record: append_jsonl(metrics_path, record),
    )
    gate = evaluate_acceptance_gate(
        result.best_metrics,
        min_explained_variance=args.gate_min_explained_variance,
        min_bucket_explained_variance=args.gate_min_bucket_explained_variance,
        min_class_margin=args.gate_min_class_margin,
    )
    gate_record = {
        "event": "gate",
        "passed": gate.passed,
        "reasons": list(gate.reasons),
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "validation": result.best_metrics,
    }
    append_jsonl(metrics_path, gate_record)
    if not gate.passed:
        print(json.dumps(_json_safe(gate_record), indent=2, sort_keys=True))
        raise SystemExit(2)

    metadata = {
        "schema": PRETRAINING_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": str(corpus_path),
        "corpus_sha256": file_sha256(corpus_path),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_step": source["step"],
        "cursor_start": corpus["cursor_start"],
        "cursor_end": corpus["cursor_end"],
        "model_id": corpus["model_id"],
        "revision": corpus["revision"],
        "data_sha256": corpus["data_sha256"],
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "feature_samples_per_trajectory": args.feature_samples_per_trajectory,
        "lambda_critic": 1.0,
        "feature_token_budget": args.feature_token_budget,
        "head_batch_size": args.head_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "min_improvement": args.min_improvement,
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "train_prompts": len(train_prompts),
        "validation_prompts": len(validation_prompts),
        "train_trajectories": len(train_records),
        "validation_trajectories": len(validation_records),
        "train_states": len(train_cache),
        "validation_states": len(validation_cache),
        "validation_metrics": _json_safe(result.best_metrics),
        "gate": _json_safe(gate_record),
    }
    output_checkpoint = build_output_checkpoint(
        source,
        critic.state_dict(),
        result.optimizer_state,
        metadata,
        cursor_end=corpus["cursor_end"],
    )
    atomic_torch_save(output_checkpoint, output_path)
    print(
        json.dumps(
            {
                "checkpoint": str(output_path),
                "metrics": str(metrics_path),
                "best_epoch": result.best_epoch,
                "validation": _json_safe(result.best_metrics),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

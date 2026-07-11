"""Pure VAPO math, verification, data, and rollout helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor


SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""),
    (" ", ""), ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "points", "feet", "minutes", "digits", "cents", "degrees", "cm",
    "gm", "pounds", "meters", "meals", "edges", "students", "multiples", "sue",
    "childrentickets", "\\text{\ns}", "\\text{\n}",
    "\\ldots", "\\dots", "\\text{s}", "\\text{.}", "\\text{}^2",
    "\\text{}^3", "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}",
    r"\;", r",\!", "{,}", '"',
]


def normalize_final_answer(answer: str) -> str:
    """DAPO/Minerva-compatible normalization used for train and AIME rewards."""
    answer = answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        answer = answer.replace(before, after)
    for expression in REMOVED_EXPRESSIONS:
        answer = answer.replace(expression, "")
    answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\3$", answer)
    answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", answer)
    answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", answer)
    answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", answer)
    answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", answer)
    answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", answer)
    answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", answer)
    answer = answer.replace("$", "")
    if answer.replace(",", "").isdigit():
        answer = answer.replace(",", "")
    return answer.strip()


def verify_answer(solution: str, ground_truth: str) -> tuple[bool, str]:
    matches = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", solution[-300:])
    prediction = normalize_final_answer(matches[-1] if matches else "[INVALID]")
    return prediction == normalize_final_answer(ground_truth), prediction


def load_unique_math_rows(path: str | Path) -> list[dict]:
    """Load and deduplicate DAPO's physically repeated parquet rows."""
    columns = ["prompt", "reward_model", "extra_info"]
    unique: dict[str, dict] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            info = row.get("extra_info") or {}
            key = str(info.get("index", row["prompt"][0]["content"]))
            unique.setdefault(key, row)
    return list(unique.values())


def length_adaptive_lambda(lengths: Tensor, alpha: float = 0.05) -> Tensor:
    lengths = lengths.to(torch.float32).clamp_min(1)
    return (1.0 - 1.0 / (alpha * lengths)).clamp(0.0, 1.0)


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    lambdas: Tensor,
    gamma: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Masked per-token GAE; lambdas is one scalar per trajectory."""
    advantage = torch.zeros_like(values)
    running = torch.zeros(values.size(0), device=values.device, dtype=values.dtype)
    for t in range(values.size(1) - 1, -1, -1):
        next_value = values[:, t + 1] if t + 1 < values.size(1) else torch.zeros_like(running)
        next_valid = mask[:, t + 1] if t + 1 < mask.size(1) else torch.zeros_like(mask[:, t])
        delta = rewards[:, t] + gamma * next_value * next_valid - values[:, t]
        running = delta + gamma * lambdas * running * next_valid
        running = running * mask[:, t]
        advantage[:, t] = running
    return advantage, advantage + values


def clipped_policy_loss(
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    epsilon_low: float = 0.20,
    epsilon_high: float = 0.28,
) -> tuple[Tensor, Tensor]:
    ratio = (new_logprobs - old_logprobs).exp()
    clipped = ratio.clamp(1.0 - epsilon_low, 1.0 + epsilon_high)
    objective = torch.minimum(ratio * advantages, clipped * advantages)
    denom = mask.sum().clamp_min(1)
    loss = -(objective * mask).sum() / denom
    clip_fraction = (((ratio < 1.0 - epsilon_low) | (ratio > 1.0 + epsilon_high)) * mask.bool()).sum() / denom
    return loss, clip_fraction


def masked_token_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def positive_example_lm_loss(logprobs: Tensor, mask: Tensor, correct: Tensor) -> Tensor:
    if not correct.any():
        return logprobs.sum() * 0
    lengths = mask.sum(1).clamp_min(1)
    per_trajectory = -(logprobs * mask).sum(1) / lengths
    return per_trajectory[correct].mean()


def top_p_sample(logits: Tensor, temperature: float, top_p: float) -> Tensor:
    logits = logits.float() / temperature
    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    remove = probs.cumsum(dim=-1) - probs > top_p
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    sampled = torch.multinomial(sorted_logits.softmax(dim=-1), 1)
    return sorted_indices.gather(-1, sampled).squeeze(-1)


@dataclass
class TrajectoryBatch:
    input_ids: Tensor
    target_ids: Tensor
    response_mask: Tensor
    old_logprobs: Tensor
    old_values: Tensor
    rewards: Tensor
    correct: Tensor
    texts: list[str]

    def to(self, device: torch.device) -> "TrajectoryBatch":
        return TrajectoryBatch(
            **{
                name: value.to(device) if torch.is_tensor(value) else value
                for name, value in self.__dict__.items()
            }
        )


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **values) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, sort_keys=True) + "\n")

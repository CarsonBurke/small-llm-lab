"""Frozen answer encoding, target caching, and cosine reward inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import Tensor

from postraining.answer_encoder.model import (
    ANSWER_ENCODER_SCHEMA,
    DEFAULT_REWARD_TEMPERATURE,
    GPT2_EOT_ID,
    AnswerEncoderConfig,
    AnswerSimilarityReward,
    TextAnswerEncoder,
    cosine_kernel_reward,
)


TARGET_CACHE_SCHEMA = "text_lejepa_target_cache/v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_encoder_checkpoint(
    checkpoint: str | Path,
    device: torch.device,
    *,
    require_behavioral_gate: bool = False,
    required_embedding_space: str | None = None,
) -> tuple[TextAnswerEncoder, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != ANSWER_ENCODER_SCHEMA:
        raise ValueError(f"unsupported answer encoder schema in {checkpoint}")
    if require_behavioral_gate and payload.get("behavioral_gate_passed") is not True:
        raise ValueError("answer encoder did not pass its behavioral reward gate")
    if require_behavioral_gate and payload.get("behavioral_gate_space") != required_embedding_space:
        raise ValueError(
            "answer encoder behavioral gate does not authorize the requested embedding space"
        )
    config = AnswerEncoderConfig(**payload["encoder_config"])
    model = TextAnswerEncoder(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def batch_tokenize(
    texts: Sequence[str],
    encode: Callable[[str], list[int]],
    *,
    pad_token_id: int,
) -> tuple[Tensor, Tensor]:
    if not texts:
        raise ValueError("at least one text is required")
    rows = []
    for text in texts:
        tokens = list(encode(text))
        rows.append(tokens or [GPT2_EOT_ID])
    max_length = max(map(len, rows))
    token_ids = torch.full((len(rows), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    for index, row in enumerate(rows):
        token_ids[index, : len(row)] = torch.tensor(row)
        attention_mask[index, : len(row)] = True
    return token_ids, attention_mask


class FrozenAnswerScorer:
    def __init__(
        self,
        model: TextAnswerEncoder,
        encode: Callable[[str], list[int]],
        *,
        device: torch.device,
        space: str = "backbone",
        reward_temperature: float = DEFAULT_REWARD_TEMPERATURE,
    ):
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("FrozenAnswerScorer requires an eval-mode frozen encoder")
        if space not in {"backbone", "projection"}:
            raise ValueError("space must be 'backbone' or 'projection'")
        cosine_kernel_reward(torch.ones(1), temperature=reward_temperature)
        self.model = model
        self.encode_tokens = encode
        self.device = device
        self.space = space
        self.reward_temperature = reward_temperature

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> Tensor:
        token_ids, attention_mask = batch_tokenize(
            texts,
            self.encode_tokens,
            pad_token_id=self.model.config.pad_token_id,
        )
        return self.model.encode(
            token_ids.to(self.device),
            attention_mask.to(self.device),
            space=self.space,
        ).cpu()

    def preencode_target(self, target: str) -> Tensor:
        return self.encode([target])

    def score(self, answers: Sequence[str], target_embedding: Tensor) -> Tensor:
        answer_embeddings = self.encode(answers)
        return AnswerSimilarityReward(
            target_embedding,
            temperature=self.reward_temperature,
        )(answer_embeddings)

    def cosine(self, answers: Sequence[str], target_embedding: Tensor) -> Tensor:
        answer_embeddings = self.encode(answers)
        return AnswerSimilarityReward(target_embedding).cosine(
            answer_embeddings
        ).squeeze(-1)


def maximum_weight_set_similarity(
    candidate: Tensor,
    target: Tensor,
    *,
    penalize_unmatched: bool,
) -> Tensor:
    """Cosine similarity under exact one-to-one patch matching.

    With unmatched penalties, missing or additional patches receive reward
    zero (raw similarity -1) and the score is normalized by the larger set.
    Matched-only mode isolates whether correspondence beats global pooling,
    without interpreting tokenizer cardinality as semantic cardinality.
    """
    if candidate.ndim != 2 or target.ndim != 2:
        raise ValueError("patch sets must have shape [patches, dimensions]")
    if candidate.size(-1) != target.size(-1):
        raise ValueError("patch-set dimensions must match")
    if candidate.size(0) < 1 or target.size(0) < 1:
        raise ValueError("patch sets must be nonempty")
    from scipy.optimize import linear_sum_assignment

    candidate = torch.nn.functional.normalize(candidate.float(), dim=-1)
    target = torch.nn.functional.normalize(target.float(), dim=-1)
    pairwise = candidate @ target.T
    candidate_indices, target_indices = linear_sum_assignment(
        -pairwise.detach().cpu().numpy()
    )
    matched = pairwise[
        torch.as_tensor(candidate_indices, device=pairwise.device),
        torch.as_tensor(target_indices, device=pairwise.device),
    ].sum()
    if not penalize_unmatched:
        return matched / len(candidate_indices)
    population = max(candidate.size(0), target.size(0))
    unmatched = population - len(candidate_indices)
    return (matched - unmatched) / population


class FrozenPatchSetScorer:
    """Compare contextual token-patch sets without CLS pooling or order."""

    def __init__(
        self,
        model: TextAnswerEncoder,
        encode: Callable[[str], list[int]],
        *,
        device: torch.device,
        space: str = "projection",
        penalize_unmatched: bool = True,
        reward_temperature: float = DEFAULT_REWARD_TEMPERATURE,
    ):
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("FrozenPatchSetScorer requires an eval-mode frozen encoder")
        if space not in {"backbone", "projection"}:
            raise ValueError("space must be 'backbone' or 'projection'")
        cosine_kernel_reward(torch.ones(1), temperature=reward_temperature)
        self.model = model
        self.encode_tokens = encode
        self.device = device
        self.patch_space = space
        self.penalize_unmatched = penalize_unmatched
        self.reward_temperature = reward_temperature
        suffix = "cardinality" if penalize_unmatched else "matched-only"
        self.space = f"patch-set/{space}/{suffix}"

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> tuple[Tensor, ...]:
        token_ids, attention_mask = batch_tokenize(
            texts,
            self.encode_tokens,
            pad_token_id=self.model.config.pad_token_id,
        )
        attention_mask = attention_mask.to(self.device)
        hidden = self.model.encode_hidden(token_ids.to(self.device), attention_mask)
        patches = hidden[:, 1:][attention_mask]
        if self.patch_space == "projection":
            patches = self.model.projector(patches)
        patches = torch.nn.functional.normalize(patches.float(), dim=-1).cpu()
        return tuple(patches.split(attention_mask.sum(dim=1).cpu().tolist()))

    def preencode_target(self, target: str) -> Tensor:
        return self.encode([target])[0]

    def score(self, answers: Sequence[str], target_embedding: Tensor) -> Tensor:
        similarities = self.cosine(answers, target_embedding)
        return cosine_kernel_reward(
            similarities,
            temperature=self.reward_temperature,
        )

    def cosine(self, answers: Sequence[str], target_embedding: Tensor) -> Tensor:
        return torch.stack(
            tuple(
                maximum_weight_set_similarity(
                    candidate,
                    target_embedding,
                    penalize_unmatched=self.penalize_unmatched,
                )
                for candidate in self.encode(answers)
            )
        )


def save_target_cache(
    output: str | Path,
    *,
    checkpoint: str | Path,
    space: str,
    targets: dict[str, Tensor],
) -> None:
    normalized = {
        key: value.detach().float().cpu()
        for key, value in targets.items()
    }
    torch.save(
        {
            "schema": TARGET_CACHE_SCHEMA,
            "checkpoint_sha256": file_sha256(checkpoint),
            "space": space,
            "targets": normalized,
        },
        output,
    )


def load_target_cache(
    path: str | Path,
    *,
    checkpoint: str | Path,
    expected_space: str,
) -> dict[str, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != TARGET_CACHE_SCHEMA:
        raise ValueError(f"unsupported target cache schema in {path}")
    if payload.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ValueError("target cache was created by a different encoder checkpoint")
    if payload.get("space") != expected_space:
        raise ValueError("target cache embedding space does not match the scorer")
    return payload["targets"]

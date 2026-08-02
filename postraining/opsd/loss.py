"""Paper-faithful full-vocabulary OPSD forward-KL objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class PointwiseKLMetrics:
    clipped_token_loss: Tensor
    forward_kl: Tensor
    clipped_entries: Tensor
    entries: int
    tokens_with_clipping: Tensor
    tokens: int
    max_contribution: Tensor


def pointwise_clipped_forward_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 1.0,
    pointwise_clip: float | None = 0.05,
) -> PointwiseKLMetrics:
    """Per-token sums after clipping each vocabulary contribution.

    For each response position and vocabulary entry this computes
    p_teacher * (log p_teacher - log p_student), then caps that individual
    contribution before summing the vocabulary. This is Equation 6 plus the
    pointwise clipping immediately following Equation 8 in the paper. The
    teacher is always detached; gradients can only reach student logits.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shape")
    if student_logits.dim() != 2:
        raise ValueError("logits must be (tokens, vocabulary)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if pointwise_clip is not None and pointwise_clip <= 0:
        raise ValueError("pointwise_clip must be positive or None")

    student_log_probs = F.log_softmax(
        student_logits.float() / temperature, dim=-1
    )
    teacher_log_probs = F.log_softmax(
        teacher_logits.detach().float() / temperature, dim=-1
    )
    contributions = teacher_log_probs.exp() * (
        teacher_log_probs - student_log_probs
    )
    forward_kl = contributions.sum(dim=-1)
    if pointwise_clip is None:
        clipped = contributions
        clipped_mask = torch.zeros_like(contributions, dtype=torch.bool)
    else:
        clipped_mask = contributions > pointwise_clip
        clipped = contributions.clamp(max=pointwise_clip)
    return PointwiseKLMetrics(
        clipped_token_loss=clipped.sum(dim=-1),
        forward_kl=forward_kl,
        clipped_entries=clipped_mask.sum(),
        entries=contributions.numel(),
        tokens_with_clipping=clipped_mask.any(dim=-1).sum(),
        tokens=contributions.size(0),
        max_contribution=contributions.max().detach(),
    )


def response_features(backbone, prompt_ids: Tensor, response_ids: Tensor) -> Tensor:
    """Features predicting every token in one on-policy response."""
    if prompt_ids.dim() != 1 or prompt_ids.numel() < 1:
        raise ValueError("prompt_ids must be a nonempty vector")
    if response_ids.dim() != 1 or response_ids.numel() < 1:
        raise ValueError("response_ids must be a nonempty vector")
    # The logit after the final response token is not part of the trajectory.
    input_ids = torch.cat((prompt_ids, response_ids[:-1])).unsqueeze(0)
    device_type = input_ids.device.type
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    ):
        token_latent = backbone.embed_tokens(input_ids)
        belief = backbone.temporal_belief_from_token_latent(token_latent)
        features = torch.cat((token_latent, belief), dim=-1)
    start = prompt_ids.numel() - 1
    selected = features[0, start:]
    if selected.size(0) != response_ids.numel():
        raise AssertionError("response feature alignment is off by one")
    return selected


def backward_opsd_example(
    student,
    teacher,
    student_prompt_ids: Tensor,
    teacher_prompt_ids: Tensor,
    response_ids: Tensor,
    *,
    batch_denominator: int,
    temperature: float,
    pointwise_clip: float | None,
    logit_chunk_tokens: int,
) -> dict[str, Tensor | int]:
    """Backpropagate one Algorithm-1 example without materializing all logits."""
    if batch_denominator < 1 or logit_chunk_tokens < 1:
        raise ValueError("batch denominator and chunk size must be positive")
    if response_ids.numel() < 1:
        raise ValueError("response must contain at least one distilled token")

    student_features = response_features(
        student, student_prompt_ids, response_ids
    )
    with torch.no_grad():
        teacher_features = response_features(
            teacher, teacher_prompt_ids, response_ids
        )

    token_count = response_ids.numel()
    clipped_loss_sum = student_features.new_zeros((), dtype=torch.float32)
    forward_kl_sum = student_features.new_zeros((), dtype=torch.float32)
    clipped_entries = torch.zeros(
        (), dtype=torch.long, device=student_features.device
    )
    total_entries = 0
    clipped_tokens = torch.zeros(
        (), dtype=torch.long, device=student_features.device
    )
    max_contribution = student_features.new_full(
        (), -torch.inf, dtype=torch.float32
    )
    for start in range(0, token_count, logit_chunk_tokens):
        stop = min(start + logit_chunk_tokens, token_count)
        device_type = student_features.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            student_logits = student.logits_from_features(
                student_features[start:stop]
            )
        with torch.no_grad(), torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            teacher_logits = teacher.logits_from_features(
                teacher_features[start:stop]
            )
        metrics = pointwise_clipped_forward_kl(
            student_logits,
            teacher_logits,
            temperature=temperature,
            pointwise_clip=pointwise_clip,
        )
        chunk_sum = metrics.clipped_token_loss.sum()
        # Each example is a token mean; Algorithm 1 then averages examples.
        (chunk_sum / (token_count * batch_denominator)).backward(
            retain_graph=stop < token_count
        )
        clipped_loss_sum += chunk_sum.detach()
        forward_kl_sum += metrics.forward_kl.sum().detach()
        clipped_entries += metrics.clipped_entries
        total_entries += metrics.entries
        clipped_tokens += metrics.tokens_with_clipping
        max_contribution = torch.maximum(
            max_contribution, metrics.max_contribution
        )
    return {
        "loss": clipped_loss_sum / token_count,
        "forward_kl": forward_kl_sum / token_count,
        "response_tokens": token_count,
        "clipped_entry_fraction": clipped_entries / total_entries,
        "clipped_token_fraction": clipped_tokens / token_count,
        "max_pointwise_contribution": max_contribution,
    }

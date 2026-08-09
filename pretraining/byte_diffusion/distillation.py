"""Sampler-trajectory distillation with detached teacher targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class DistillationTerms:
    total: Tensor
    hard_ce: Tensor
    kl: Tensor
    ar: Tensor


def trajectory_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    final_ids: Tensor,
    selected: Tensor,
    *,
    ar_loss: Tensor,
    temperature: float = 1.0,
    hard_weight: float = 1.0,
    kl_weight: float = 1.0,
    ar_weight: float = 1.0,
) -> DistillationTerms:
    if student_logits.shape != teacher_logits.shape or student_logits.shape[:-1] != final_ids.shape:
        raise ValueError("trajectory logits and final ids must align")
    if selected.shape != final_ids.shape or selected.dtype != torch.bool or not bool(selected.any()):
        raise ValueError("selected must identify at least one distilled action")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher = teacher_logits.detach()
    targets = final_ids.detach()
    hard = F.cross_entropy(student_logits[selected].float(), targets[selected], reduction="mean")
    teacher_prob = (teacher[selected].float() / temperature).softmax(-1)
    student_log_prob = (student_logits[selected].float() / temperature).log_softmax(-1)
    teacher_log_prob = teacher_prob.clamp_min(torch.finfo(torch.float32).tiny).log()
    kl = (teacher_prob * (teacher_log_prob - student_log_prob)).sum(-1).mean() * temperature**2
    total = hard_weight * hard + kl_weight * kl + ar_weight * ar_loss
    return DistillationTerms(total=total, hard_ce=hard, kl=kl, ar=ar_loss)


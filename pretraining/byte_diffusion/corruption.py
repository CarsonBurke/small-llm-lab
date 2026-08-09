"""Device-side corruption with explicit, testable sampling laws."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import AtomicVocabulary


@dataclass(frozen=True)
class CorruptedBatch:
    ids: Tensor
    targets: Tensor
    active: Tensor
    noise_fraction: Tensor
    kind: str
    t: Tensor | None = None

    def validate(self, vocab: AtomicVocabulary) -> None:
        if self.ids.shape != self.targets.shape or self.ids.shape != self.active.shape:
            raise ValueError("corruption tensors must have identical shapes")
        if self.ids.dtype != torch.long or self.targets.dtype != torch.long:
            raise ValueError("ids and targets must be torch.long")
        if self.active.dtype != torch.bool:
            raise ValueError("active must be boolean")
        if not self.ids.is_cuda:
            if bool((self.ids < 0).any()) or bool(
                (self.ids >= vocab.input_size).any()
            ):
                raise ValueError("corrupted input contains an invalid id")
            if bool((self.targets[self.active] >= vocab.output_size).any()):
                raise ValueError("an active target is not predictable")
            if bool((self.targets[self.active] < 0).any()):
                raise ValueError("an active target is negative")


def _require_inputs(clean_ids: Tensor, eligible: Tensor, vocab: AtomicVocabulary) -> None:
    if clean_ids.dtype != torch.long or eligible.dtype != torch.bool:
        raise TypeError("clean_ids must be long and eligible must be bool")
    if clean_ids.shape != eligible.shape or clean_ids.ndim != 2:
        raise ValueError("clean_ids and eligible must be aligned rank-2 tensors")
    if not clean_ids.is_cuda:
        values = clean_ids[eligible]
        if bool((values < 0).any()) or bool((values >= vocab.output_size).any()):
            raise ValueError("eligible clean ids must be predictable")


def exact_k_mask(
    eligible: Tensor,
    k: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Uniformly select exactly ``k[row]`` eligible positions per row."""

    if eligible.ndim != 2 or eligible.dtype != torch.bool:
        raise ValueError("eligible must be a rank-2 boolean tensor")
    if k.shape != eligible.shape[:1] or k.dtype != torch.long:
        raise ValueError("k must be one int64 value per row")
    counts = eligible.sum(1)
    if not eligible.is_cuda and bool(((k < 0) | (k > counts)).any()):
        raise ValueError("k lies outside the eligible count")
    scores = torch.rand(
        eligible.shape,
        device=eligible.device,
        generator=generator,
        dtype=torch.float32,
    )
    scores = scores.masked_fill(~eligible, 2.0)
    order = scores.argsort(dim=1)
    ranks = torch.empty_like(order)
    positions = torch.arange(eligible.shape[1], device=eligible.device).expand_as(order)
    ranks.scatter_(1, order, positions)
    return eligible & (ranks < k[:, None])


def sample_uniform_k(
    eligible: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    counts = eligible.sum(1)
    uniforms = torch.rand(
        counts.shape,
        device=eligible.device,
        generator=generator,
        dtype=torch.float32,
    )
    return (uniforms * counts).floor().to(torch.long) + counts.gt(0).to(torch.long)


def absorbing_rb(
    clean_ids: Tensor,
    eligible: Tensor,
    vocab: AtomicVocabulary,
    *,
    generator: torch.Generator | None = None,
    k: Tensor | None = None,
) -> CorruptedBatch:
    _require_inputs(clean_ids, eligible, vocab)
    counts = eligible.sum(1)
    chosen_k = sample_uniform_k(eligible, generator=generator) if k is None else k
    active = exact_k_mask(eligible, chosen_k, generator=generator)
    ids = torch.where(active, vocab.mask_id, clean_ids)
    result = CorruptedBatch(
        ids=ids,
        targets=clean_ids,
        active=active,
        noise_fraction=chosen_k.float() / counts.clamp_min(1).float(),
        kind="absorbing_rb",
    )
    result.validate(vocab)
    return result


def allmask_50(
    clean_ids: Tensor,
    eligible: Tensor,
    vocab: AtomicVocabulary,
    *,
    generator: torch.Generator | None = None,
) -> CorruptedBatch:
    _require_inputs(clean_ids, eligible, vocab)
    counts = eligible.sum(1)
    k = sample_uniform_k(eligible, generator=generator)
    all_mask = torch.rand(
        counts.shape,
        device=eligible.device,
        generator=generator,
        dtype=torch.float32,
    ) < 0.5
    k = torch.where(all_mask, counts, k)
    active = exact_k_mask(eligible, k, generator=generator)
    result = CorruptedBatch(
        ids=torch.where(active, vocab.mask_id, clean_ids),
        targets=clean_ids,
        active=active,
        noise_fraction=k.float() / counts.clamp_min(1).float(),
        kind="allmask_50",
    )
    result.validate(vocab)
    return result


def blt_bernoulli(
    clean_ids: Tensor,
    eligible: Tensor,
    vocab: AtomicVocabulary,
    *,
    generator: torch.Generator | None = None,
    t: Tensor | None = None,
) -> CorruptedBatch:
    """Fast-BLT corruption: one shared t and independent byte masks."""

    _require_inputs(clean_ids, eligible, vocab)
    if t is None:
        t = torch.rand((), device=clean_ids.device, generator=generator).clamp_min(
            torch.finfo(torch.float32).tiny
        )
    if t.numel() != 1 or (
        not t.is_cuda and not 0.0 < float(t) <= 1.0
    ):
        raise ValueError("t must be one scalar in (0, 1]")
    active = eligible & (
        torch.rand(
            clean_ids.shape,
            device=clean_ids.device,
            generator=generator,
            dtype=torch.float32,
        ) < t
    )
    counts = eligible.sum(1)
    result = CorruptedBatch(
        ids=torch.where(active, vocab.mask_id, clean_ids),
        targets=clean_ids,
        active=active,
        noise_fraction=active.sum(1).float() / counts.clamp_min(1).float(),
        kind="blt_bernoulli",
        t=t,
    )
    result.validate(vocab)
    return result


def uniform_replacement(
    clean_ids: Tensor,
    eligible: Tensor,
    vocab: AtomicVocabulary,
    *,
    generator: torch.Generator | None = None,
    t: Tensor | None = None,
) -> CorruptedBatch:
    _require_inputs(clean_ids, eligible, vocab)
    batch = clean_ids.shape[0]
    if t is None:
        t = torch.rand((batch,), device=clean_ids.device, generator=generator)
    if t.shape not in {(batch,), ()} or (
        not t.is_cuda and bool(((t <= 0) | (t > 1)).any())
    ):
        raise ValueError("t must be scalar or one value per row in (0, 1]")
    replace = eligible & (
        torch.rand(
            clean_ids.shape,
            device=clean_ids.device,
            generator=generator,
            dtype=torch.float32,
        ) < t.reshape(-1, 1)
    )
    random_ids = torch.randint(
        vocab.output_size,
        clean_ids.shape,
        device=clean_ids.device,
        generator=generator,
    )
    result = CorruptedBatch(
        ids=torch.where(replace, random_ids, clean_ids),
        targets=clean_ids,
        active=eligible,
        noise_fraction=replace.sum(1).float() / eligible.sum(1).clamp_min(1).float(),
        kind="uniform_replacement",
        t=t,
    )
    result.validate(vocab)
    return result


def whole_patch(
    clean_ids: Tensor,
    eligible: Tensor,
    vocab: AtomicVocabulary,
    *,
    patch_stride: int = 4,
    generator: torch.Generator | None = None,
    contiguous: bool = False,
) -> CorruptedBatch:
    _require_inputs(clean_ids, eligible, vocab)
    if clean_ids.shape[1] % patch_stride:
        raise ValueError("whole-patch corruption requires a patch-aligned width")
    patch_eligible = eligible.view(clean_ids.shape[0], -1, patch_stride).any(-1)
    patch_counts = patch_eligible.sum(1)
    j = sample_uniform_k(patch_eligible, generator=generator)
    if contiguous:
        active_patch = torch.zeros_like(patch_eligible)
        starts_u = torch.rand(
            patch_counts.shape,
            device=clean_ids.device,
            generator=generator,
        )
        starts = (starts_u * (patch_counts - j + 1)).floor().to(torch.long)
        index = torch.arange(patch_eligible.shape[1], device=clean_ids.device)
        active_patch = (index[None, :] >= starts[:, None]) & (
            index[None, :] < (starts + j)[:, None]
        )
        active_patch &= patch_eligible
        kind = "contiguous_patch_span"
    else:
        active_patch = exact_k_mask(patch_eligible, j, generator=generator)
        kind = "whole_patch"
    active = active_patch.repeat_interleave(patch_stride, dim=1) & eligible
    result = CorruptedBatch(
        ids=torch.where(active, vocab.mask_id, clean_ids),
        targets=clean_ids,
        active=active,
        noise_fraction=active.sum(1).float() / eligible.sum(1).clamp_min(1).float(),
        kind=kind,
    )
    result.validate(vocab)
    return result


def compile_corruption(function):
    """Compile pointwise corruption/gather plumbing into static CUDA kernels."""

    return torch.compile(function, fullgraph=True, dynamic=False, mode="max-autotune")

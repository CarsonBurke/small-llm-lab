"""Portable fused utilities and fail-closed attention backend selection.

The byte-diffusion model has a deliberately small, fixed output vocabulary.
This module keeps its categorical sampler statistics in one operation and
provides a lazy Triton implementation for CUDA inference.  Importing this
module never imports Triton, initializes CUDA, or requires a CUDA build.

Dense attention is a correctness oracle, not a production fallback.  Backend
selection below therefore requires an explicit opt-in and a small sequence
length before it will return the dense reference backend.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import importlib.util
from typing import Callable, Iterator, Literal, ParamSpec, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor


ATOMIC_OUTPUT_SIZE = 261
_TRITON_BLOCK_SIZE = 512
_TRITON_CATEGORICAL_DTYPES = {torch.bfloat16, torch.float16, torch.float32}
# Largest standard canvas. The reveal implementation is also shape-specialized
# for Fast-BLT B4/B8/B16 inference.
CANVAS_REVEAL_SIZE = 512
DEFAULT_DENSE_REFERENCE_LIMIT = 256

P = ParamSpec("P")
R = TypeVar("R")


def _validate_categorical_logits(logits: Tensor) -> None:
    if not logits.is_floating_point():
        raise TypeError("categorical logits must be floating point")
    if logits.ndim < 1 or logits.shape[-1] != ATOMIC_OUTPUT_SIZE:
        raise ValueError(
            "categorical logits must have final dimension "
            f"{ATOMIC_OUTPUT_SIZE}, got {tuple(logits.shape)}"
        )


def categorical_entropy_argmax_confidence_reference(
    logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return FP32 entropy, low-id-tied argmax, and argmax probability.

    All reductions intentionally run in FP32, including when model logits are
    BF16.  This is both the CPU implementation and the semantic oracle for the
    optional CUDA kernel.  The output shapes equal ``logits.shape[:-1]``;
    argmax is int64 and the other outputs are float32.

    Inputs are expected to be finite.  Non-finite values retain PyTorch's
    native ``log_softmax``/``argmax`` behavior so this function remains fully
    tensorized and safe to place inside ``torch.compile``.
    """

    _validate_categorical_logits(logits)
    fp32_logits = logits.float()
    log_probabilities = F.log_softmax(fp32_logits, dim=-1)
    probabilities = log_probabilities.exp()
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    argmax = fp32_logits.argmax(dim=-1)
    confidence = probabilities.gather(-1, argmax.unsqueeze(-1)).squeeze(-1)
    return entropy, argmax, confidence


def _validate_uniforms(logits: Tensor, uniforms: Tensor) -> None:
    if uniforms.shape != logits.shape[:-1]:
        raise ValueError(
            "uniforms must match the leading logits shape, got "
            f"{tuple(uniforms.shape)} for logits {tuple(logits.shape)}"
        )
    if not uniforms.is_floating_point():
        raise TypeError("categorical uniforms must be floating point")
    if uniforms.device != logits.device:
        raise ValueError("categorical logits and uniforms must share a device")


def categorical_sample_entropy_argmax_confidence_reference(
    logits: Tensor,
    uniforms: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample from 261 logits while returning sampler statistics.

    ``uniforms`` supplies one caller-owned draw per categorical row and must be
    in ``[0, 1)``. Value bounds are a caller precondition rather than a runtime
    check because inspecting CUDA values here would add a synchronization or a
    separate reduction kernel. Supplying randomness instead of generating it
    here keeps deterministic resume, request-local RNG, and CUDA-graph replay
    under the caller's control. Sampling is the first index whose inclusive
    CDF is at least the supplied uniform; the final index is a roundoff guard.
    """

    _validate_categorical_logits(logits)
    _validate_uniforms(logits, uniforms)
    entropy, argmax, confidence = (
        categorical_entropy_argmax_confidence_reference(logits)
    )
    probabilities = F.softmax(logits.float(), dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    sample = (cumulative < uniforms.float().unsqueeze(-1)).sum(dim=-1)
    sample = sample.clamp_max(ATOMIC_OUTPUT_SIZE - 1).to(torch.int64)
    return sample, entropy, argmax, confidence


@lru_cache(maxsize=1)
def triton_is_importable() -> bool:
    """Whether a Triton package can be found without importing it."""

    return importlib.util.find_spec("triton") is not None


_TRITON_CATEGORICAL_KERNEL = None
_TRITON_CATEGORICAL_SAMPLE_KERNEL = None
_TRITON_REVEAL_KERNEL = None
_TRITON_IMPORT_ERROR: Exception | None = None
_TRITON_SAMPLE_IMPORT_ERROR: Exception | None = None
_TRITON_REVEAL_IMPORT_ERROR: Exception | None = None


def _load_triton_categorical_kernel():
    """Import and define the Triton kernel on first CUDA dispatch only."""

    global _TRITON_CATEGORICAL_KERNEL, _TRITON_IMPORT_ERROR
    # Triton's JIT resolves ``tl`` through the kernel function's module globals.
    global triton, tl  # type: ignore[no-redef]

    if _TRITON_CATEGORICAL_KERNEL is not None:
        return _TRITON_CATEGORICAL_KERNEL
    if _TRITON_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Triton categorical kernel is unavailable"
        ) from _TRITON_IMPORT_ERROR

    try:
        import triton as triton  # type: ignore[no-redef]
        import triton.language as tl  # type: ignore[no-redef]

        @triton.jit
        def _categorical_statistics_kernel(
            logits,
            entropy_out,
            argmax_out,
            confidence_out,
            VOCAB_SIZE: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            row = tl.program_id(0)
            columns = tl.arange(0, BLOCK_SIZE)
            valid = columns < VOCAB_SIZE
            values = tl.load(
                logits + row * VOCAB_SIZE + columns,
                mask=valid,
                other=float("-inf"),
            ).to(tl.float32)

            maximum = tl.max(values, axis=0)
            exponentials = tl.where(valid, tl.exp(values - maximum), 0.0)
            normalizer = tl.sum(exponentials, axis=0)
            probabilities = exponentials / normalizer
            log_normalizer = tl.log(normalizer)
            log_probabilities = values - maximum - log_normalizer
            entropy = -tl.sum(
                tl.where(valid, probabilities * log_probabilities, 0.0), axis=0
            )
            argmax = tl.argmax(values, axis=0, tie_break_left=True)
            confidence = tl.sum(
                tl.where(columns == argmax, probabilities, 0.0), axis=0
            )

            tl.store(entropy_out + row, entropy)
            tl.store(argmax_out + row, argmax)
            tl.store(confidence_out + row, confidence)

        _TRITON_CATEGORICAL_KERNEL = _categorical_statistics_kernel
    except Exception as error:
        _TRITON_IMPORT_ERROR = error
        raise RuntimeError("Triton categorical kernel is unavailable") from error
    return _TRITON_CATEGORICAL_KERNEL


def _load_triton_categorical_sample_kernel():
    """Define the fused inverse-CDF sampling kernel after lazy Triton import."""

    global _TRITON_CATEGORICAL_SAMPLE_KERNEL, _TRITON_SAMPLE_IMPORT_ERROR
    if _TRITON_CATEGORICAL_SAMPLE_KERNEL is not None:
        return _TRITON_CATEGORICAL_SAMPLE_KERNEL
    if _TRITON_SAMPLE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Triton categorical sample kernel is unavailable"
        ) from _TRITON_SAMPLE_IMPORT_ERROR
    try:
        _load_triton_categorical_kernel()

        @triton.jit
        def _categorical_sample_statistics_kernel(
            logits,
            uniforms,
            sample_out,
            entropy_out,
            argmax_out,
            confidence_out,
            VOCAB_SIZE: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            row = tl.program_id(0)
            columns = tl.arange(0, BLOCK_SIZE)
            valid = columns < VOCAB_SIZE
            values = tl.load(
                logits + row * VOCAB_SIZE + columns,
                mask=valid,
                other=float("-inf"),
            ).to(tl.float32)
            uniform = tl.load(uniforms + row).to(tl.float32)

            maximum = tl.max(values, axis=0)
            exponentials = tl.where(valid, tl.exp(values - maximum), 0.0)
            normalizer = tl.sum(exponentials, axis=0)
            probabilities = exponentials / normalizer
            log_normalizer = tl.log(normalizer)
            log_probabilities = values - maximum - log_normalizer
            entropy = -tl.sum(
                tl.where(valid, probabilities * log_probabilities, 0.0), axis=0
            )
            argmax = tl.argmax(values, axis=0, tie_break_left=True)
            confidence = tl.sum(
                tl.where(columns == argmax, probabilities, 0.0), axis=0
            )
            cumulative = tl.cumsum(probabilities, axis=0)
            sample = tl.sum((valid & (cumulative < uniform)).to(tl.int32), axis=0)
            sample = tl.minimum(sample, VOCAB_SIZE - 1)

            tl.store(sample_out + row, sample)
            tl.store(entropy_out + row, entropy)
            tl.store(argmax_out + row, argmax)
            tl.store(confidence_out + row, confidence)

        _TRITON_CATEGORICAL_SAMPLE_KERNEL = _categorical_sample_statistics_kernel
    except Exception as error:
        _TRITON_SAMPLE_IMPORT_ERROR = error
        raise RuntimeError("Triton categorical sample kernel is unavailable") from error
    return _TRITON_CATEGORICAL_SAMPLE_KERNEL


def _load_triton_reveal_kernel():
    """Define the stable 512-position reveal/update kernel lazily."""

    global _TRITON_REVEAL_KERNEL, _TRITON_REVEAL_IMPORT_ERROR
    if _TRITON_REVEAL_KERNEL is not None:
        return _TRITON_REVEAL_KERNEL
    if _TRITON_REVEAL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Triton reveal kernel is unavailable"
        ) from _TRITON_REVEAL_IMPORT_ERROR
    try:
        _load_triton_categorical_kernel()

        @triton.jit
        def _reveal_low_entropy_kernel(
            canvas,
            samples,
            entropy,
            unresolved,
            active,
            quota,
            canvas_out,
            unresolved_out,
            active_out,
            revealed_out,
            EOT_ID,
            WIDTH: tl.constexpr,
        ):
            row = tl.program_id(0)
            positions = tl.arange(0, WIDTH)
            row_offset = row * WIDTH
            old_canvas = tl.load(canvas + row_offset + positions)
            sampled = tl.load(samples + row_offset + positions)
            scores = tl.load(entropy + row_offset + positions).to(tl.float32)
            was_unresolved = tl.load(unresolved + row_offset + positions).to(tl.int1)
            was_active = tl.load(active + row_offset + positions).to(tl.int1)
            live = was_unresolved & was_active

            # An EOT proposal is eligible only when every earlier active slot
            # was resolved before this update. This conservative gate never
            # exposes a terminal symbol ahead of unresolved prefix content.
            live_prefix = tl.cumsum(live.to(tl.int32), axis=0)
            earlier_unresolved = live_prefix - live.to(tl.int32)
            eot_allowed = (sampled != EOT_ID) | (earlier_unresolved == 0)
            finite = (
                (scores == scores)
                & (scores < float("inf"))
                & (scores > float("-inf"))
            )
            eligible = live & eot_allowed & finite
            priority = tl.where(eligible, scores, float("inf"))

            # tl.sort returns values only. Recover a stable position-tied
            # selection from the runtime threshold: everything below it wins,
            # followed by the earliest equal-valued positions.
            sorted_priority = tl.sort(priority, dim=0, descending=False)
            requested = tl.load(quota + row).to(tl.int32)
            requested = tl.maximum(0, tl.minimum(requested, WIDTH))
            threshold = tl.sum(
                tl.where(positions == requested - 1, sorted_priority, 0.0), axis=0
            )
            below = priority < threshold
            below_count = tl.sum(below.to(tl.int32), axis=0)
            equal = priority == threshold
            equal_rank = tl.cumsum(equal.to(tl.int32), axis=0) - equal.to(tl.int32)
            selected = eligible & (requested > 0) & (
                below | (equal & (equal_rank < requested - below_count))
            )

            first_eot = tl.min(
                tl.where(selected & (sampled == EOT_ID), positions, WIDTH), axis=0
            )
            kept = positions <= first_eot
            selected = selected & kept
            now_active = was_active & kept
            now_unresolved = was_unresolved & ~selected & now_active
            now_canvas = tl.where(selected, sampled, old_canvas)

            tl.store(canvas_out + row_offset + positions, now_canvas)
            tl.store(unresolved_out + row_offset + positions, now_unresolved)
            tl.store(active_out + row_offset + positions, now_active)
            tl.store(revealed_out + row_offset + positions, selected)

        _TRITON_REVEAL_KERNEL = _reveal_low_entropy_kernel
    except Exception as error:
        _TRITON_REVEAL_IMPORT_ERROR = error
        raise RuntimeError("Triton reveal kernel is unavailable") from error
    return _TRITON_REVEAL_KERNEL


def _categorical_entropy_argmax_confidence_triton(
    logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if logits.device.type != "cuda":
        raise RuntimeError("the Triton categorical kernel requires CUDA logits")
    if logits.dtype not in _TRITON_CATEGORICAL_DTYPES:
        raise RuntimeError(
            "the Triton categorical kernel supports BF16, FP16, and FP32 logits"
        )
    if logits.requires_grad and torch.is_grad_enabled():
        raise RuntimeError(
            "the Triton categorical statistics path is inference-only; "
            "use backend='torch' when gradients are required"
        )
    if logits.numel() == 0:
        return categorical_entropy_argmax_confidence_reference(logits)

    contiguous = logits.contiguous()
    rows = contiguous.numel() // ATOMIC_OUTPUT_SIZE
    output_shape = contiguous.shape[:-1]
    entropy = torch.empty(output_shape, dtype=torch.float32, device=logits.device)
    argmax = torch.empty(output_shape, dtype=torch.int64, device=logits.device)
    confidence = torch.empty(output_shape, dtype=torch.float32, device=logits.device)
    kernel = _load_triton_categorical_kernel()
    launcher = (
        torch.library.wrap_triton(kernel) if torch.compiler.is_compiling() else kernel
    )
    launcher[(rows,)](
        contiguous,
        entropy,
        argmax,
        confidence,
        VOCAB_SIZE=ATOMIC_OUTPUT_SIZE,
        BLOCK_SIZE=_TRITON_BLOCK_SIZE,
        num_warps=4,
    )
    return entropy, argmax, confidence


CategoricalBackend = Literal["auto", "torch", "triton"]


def categorical_entropy_argmax_confidence(
    logits: Tensor,
    *,
    backend: CategoricalBackend = "auto",
) -> tuple[Tensor, Tensor, Tensor]:
    """Portable fused categorical statistics for the 261-way atomic head.

    ``auto`` selects Triton only for no-grad CUDA inputs when Triton is
    installed.  CPU, differentiable, and unsupported calls use the exact Torch
    reference.  ``triton`` is strict and raises instead of silently changing
    backend.
    """

    _validate_categorical_logits(logits)
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError(f"unknown categorical backend: {backend!r}")
    if backend == "torch":
        return categorical_entropy_argmax_confidence_reference(logits)
    if backend == "triton":
        if not triton_is_importable():
            raise RuntimeError("backend='triton' requested but Triton is not installed")
        return _categorical_entropy_argmax_confidence_triton(logits)
    if (
        logits.device.type == "cuda"
        and logits.dtype in _TRITON_CATEGORICAL_DTYPES
        and (not logits.requires_grad or not torch.is_grad_enabled())
        and triton_is_importable()
    ):
        return _categorical_entropy_argmax_confidence_triton(logits)
    return categorical_entropy_argmax_confidence_reference(logits)


def _categorical_sample_entropy_argmax_confidence_triton(
    logits: Tensor,
    uniforms: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if logits.device.type != "cuda":
        raise RuntimeError("the Triton categorical sample kernel requires CUDA logits")
    if logits.dtype not in _TRITON_CATEGORICAL_DTYPES:
        raise RuntimeError(
            "the Triton categorical sample kernel supports BF16, FP16, and FP32 logits"
        )
    if logits.requires_grad and torch.is_grad_enabled():
        raise RuntimeError(
            "the Triton categorical sample path is inference-only; "
            "use backend='torch' when gradients are required"
        )
    if logits.numel() == 0:
        return categorical_sample_entropy_argmax_confidence_reference(
            logits, uniforms
        )

    contiguous_logits = logits.contiguous()
    contiguous_uniforms = uniforms.contiguous()
    rows = contiguous_logits.numel() // ATOMIC_OUTPUT_SIZE
    output_shape = contiguous_logits.shape[:-1]
    sample = torch.empty(output_shape, dtype=torch.int64, device=logits.device)
    entropy = torch.empty(output_shape, dtype=torch.float32, device=logits.device)
    argmax = torch.empty(output_shape, dtype=torch.int64, device=logits.device)
    confidence = torch.empty(output_shape, dtype=torch.float32, device=logits.device)
    kernel = _load_triton_categorical_sample_kernel()
    launcher = (
        torch.library.wrap_triton(kernel) if torch.compiler.is_compiling() else kernel
    )
    launcher[(rows,)](
        contiguous_logits,
        contiguous_uniforms,
        sample,
        entropy,
        argmax,
        confidence,
        VOCAB_SIZE=ATOMIC_OUTPUT_SIZE,
        BLOCK_SIZE=_TRITON_BLOCK_SIZE,
        num_warps=4,
    )
    return sample, entropy, argmax, confidence


def categorical_sample_entropy_argmax_confidence(
    logits: Tensor,
    uniforms: Tensor,
    *,
    backend: CategoricalBackend = "auto",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Portable fused inverse-CDF sampling and categorical statistics."""

    _validate_categorical_logits(logits)
    _validate_uniforms(logits, uniforms)
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError(f"unknown categorical backend: {backend!r}")
    if backend == "torch":
        return categorical_sample_entropy_argmax_confidence_reference(
            logits, uniforms
        )
    if backend == "triton":
        if not triton_is_importable():
            raise RuntimeError("backend='triton' requested but Triton is not installed")
        return _categorical_sample_entropy_argmax_confidence_triton(
            logits, uniforms
        )
    if (
        logits.device.type == "cuda"
        and logits.dtype in _TRITON_CATEGORICAL_DTYPES
        and (not logits.requires_grad or not torch.is_grad_enabled())
        and triton_is_importable()
    ):
        return _categorical_sample_entropy_argmax_confidence_triton(
            logits, uniforms
        )
    return categorical_sample_entropy_argmax_confidence_reference(logits, uniforms)


def _validate_reveal_inputs(
    canvas: Tensor,
    samples: Tensor,
    entropy: Tensor,
    unresolved: Tensor,
    active: Tensor,
    quota: int | Tensor,
    eot_id: int,
) -> None:
    if canvas.ndim != 2 or not 0 < canvas.shape[1] <= CANVAS_REVEAL_SIZE:
        raise ValueError(
            f"canvas must have shape [batch, width] with 1 <= width <= "
            f"{CANVAS_REVEAL_SIZE}, got {tuple(canvas.shape)}"
        )
    expected = canvas.shape
    for name, tensor in {
        "samples": samples,
        "entropy": entropy,
        "unresolved": unresolved,
        "active": active,
    }.items():
        if tensor.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if tensor.device != canvas.device:
            raise ValueError("all reveal tensors must share a device")
    if canvas.dtype != torch.int64 or samples.dtype != torch.int64:
        raise TypeError("canvas and samples must be int64")
    if not entropy.is_floating_point():
        raise TypeError("reveal entropy must be floating point")
    if unresolved.dtype != torch.bool or active.dtype != torch.bool:
        raise TypeError("unresolved and active must be boolean")
    if not 0 <= eot_id < ATOMIC_OUTPUT_SIZE:
        raise ValueError("eot_id must be a predictable atomic id")
    if torch.is_tensor(quota):
        if quota.device != canvas.device:
            raise ValueError("quota and reveal tensors must share a device")
        if quota.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("quota must have an integer dtype")
        if quota.ndim > 1 or (quota.ndim == 1 and quota.shape[0] != canvas.shape[0]):
            raise ValueError("quota must be scalar or one value per canvas row")
    elif isinstance(quota, bool) or not isinstance(quota, int):
        raise TypeError("quota must be an int or integer tensor")


def _quota_per_row(canvas: Tensor, quota: int | Tensor) -> Tensor:
    width = canvas.shape[1]
    if isinstance(quota, int):
        return torch.full(
            (canvas.shape[0],),
            max(0, min(quota, width)),
            dtype=torch.int64,
            device=canvas.device,
        )
    if quota.ndim == 0:
        return quota.to(torch.int64).expand(canvas.shape[0])
    return quota.to(torch.int64)


def reveal_low_entropy_reference(
    canvas: Tensor,
    samples: Tensor,
    entropy: Tensor,
    unresolved: Tensor,
    active: Tensor,
    quota: int | Tensor,
    *,
    eot_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Reveal a stable lowest-entropy subset of one 512-position canvas.

    Ordering is ascending ``(entropy, position)``. Non-finite entropy is
    ineligible. An EOT sample is eligible only when its active prefix was fully
    resolved before this update. Revealing EOT immediately deactivates its
    suffix; selected suffix candidates are consequently not committed.

    Returns functional copies ``(canvas, unresolved, active, revealed)``.
    ``revealed`` marks positions actually committed by this call.
    """

    _validate_reveal_inputs(
        canvas, samples, entropy, unresolved, active, quota, eot_id
    )
    width = canvas.shape[1]
    positions = torch.arange(width, device=canvas.device)
    live = unresolved & active
    earlier_unresolved = live.to(torch.int32).cumsum(-1) - live.to(torch.int32)
    eot_allowed = (samples != eot_id) | (earlier_unresolved == 0)
    fp32_entropy = entropy.float()
    eligible = live & eot_allowed & torch.isfinite(fp32_entropy)
    priority = torch.where(
        eligible, fp32_entropy, torch.full_like(fp32_entropy, torch.inf)
    )
    order = priority.argsort(dim=-1, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(
        -1,
        order,
        positions.expand(canvas.shape[0], width),
    )
    per_row_quota = _quota_per_row(canvas, quota).clamp(0, width)
    selected = eligible & (rank < per_row_quota[:, None])

    eot_stop = torch.where(
        selected & (samples == eot_id),
        positions[None, :],
        torch.full_like(samples, width),
    ).amin(dim=-1)
    kept = positions[None, :] <= eot_stop[:, None]
    revealed = selected & kept
    updated_active = active & kept
    updated_unresolved = unresolved & ~revealed & updated_active
    updated_canvas = torch.where(revealed, samples, canvas)
    return (
        updated_canvas.contiguous(),
        updated_unresolved.contiguous(),
        updated_active.contiguous(),
        revealed.contiguous(),
    )


RevealBackend = Literal["auto", "torch", "triton"]


def _reveal_low_entropy_triton(
    canvas: Tensor,
    samples: Tensor,
    entropy: Tensor,
    unresolved: Tensor,
    active: Tensor,
    quota: int | Tensor,
    *,
    eot_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if canvas.device.type != "cuda":
        raise RuntimeError("the Triton reveal kernel requires CUDA tensors")
    if entropy.requires_grad and torch.is_grad_enabled():
        raise RuntimeError(
            "the Triton reveal path is inference-only; use backend='torch' "
            "when gradients are required"
        )
    if canvas.shape[0] == 0:
        return reveal_low_entropy_reference(
            canvas,
            samples,
            entropy,
            unresolved,
            active,
            quota,
            eot_id=eot_id,
        )
    width = canvas.shape[1]
    # tl.sort requires a power-of-two compile-time width. Canonical BLT block
    # sizes and the 512-byte canvas satisfy this; unusual widths use the
    # vectorized Torch reference without returning to scalar Python.
    if width & (width - 1):
        return reveal_low_entropy_reference(
            canvas,
            samples,
            entropy,
            unresolved,
            active,
            quota,
            eot_id=eot_id,
        )
    quota_rows = _quota_per_row(canvas, quota).clamp(0, width).to(
        torch.int32
    ).contiguous()
    contiguous_canvas = canvas.contiguous()
    contiguous_samples = samples.contiguous()
    contiguous_entropy = entropy.contiguous()
    contiguous_unresolved = unresolved.contiguous()
    contiguous_active = active.contiguous()
    canvas_out = torch.empty_like(contiguous_canvas)
    unresolved_out = torch.empty_like(contiguous_unresolved)
    active_out = torch.empty_like(contiguous_active)
    revealed_out = torch.empty_like(contiguous_unresolved)
    kernel = _load_triton_reveal_kernel()
    launcher = (
        torch.library.wrap_triton(kernel) if torch.compiler.is_compiling() else kernel
    )
    launcher[(canvas.shape[0],)](
        contiguous_canvas,
        contiguous_samples,
        contiguous_entropy,
        contiguous_unresolved,
        contiguous_active,
        quota_rows,
        canvas_out,
        unresolved_out,
        active_out,
        revealed_out,
        eot_id,
        WIDTH=width,
        num_warps=8 if width >= 256 else 4,
    )
    return canvas_out, unresolved_out, active_out, revealed_out


def reveal_low_entropy(
    canvas: Tensor,
    samples: Tensor,
    entropy: Tensor,
    unresolved: Tensor,
    active: Tensor,
    quota: int | Tensor,
    *,
    eot_id: int,
    backend: RevealBackend = "auto",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Portable stable reveal/update with a lazy strict Triton backend."""

    _validate_reveal_inputs(
        canvas, samples, entropy, unresolved, active, quota, eot_id
    )
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError(f"unknown reveal backend: {backend!r}")
    if backend == "torch":
        return reveal_low_entropy_reference(
            canvas,
            samples,
            entropy,
            unresolved,
            active,
            quota,
            eot_id=eot_id,
        )
    if backend == "triton":
        if not triton_is_importable():
            raise RuntimeError("backend='triton' requested but Triton is not installed")
        return _reveal_low_entropy_triton(
            canvas,
            samples,
            entropy,
            unresolved,
            active,
            quota,
            eot_id=eot_id,
        )
    if (
        canvas.device.type == "cuda"
        and (not entropy.requires_grad or not torch.is_grad_enabled())
        and triton_is_importable()
    ):
        return _reveal_low_entropy_triton(
            canvas,
            samples,
            entropy,
            unresolved,
            active,
            quota,
            eot_id=eot_id,
        )
    return reveal_low_entropy_reference(
        canvas,
        samples,
        entropy,
        unresolved,
        active,
        quota,
        eot_id=eot_id,
    )


class AttentionPattern(str, Enum):
    CAUSAL = "causal"
    CAUSAL_WINDOW = "causal_window"
    BIDIRECTIONAL = "bidirectional"
    BRANCH = "branch"
    INTROSPECTION = "introspection"


class AttentionBackend(str, Enum):
    VARLEN_FLASH = "varlen_flash"
    FLASH_SDPA = "flash_sdpa"
    FLEX = "flex"
    DENSE_REFERENCE = "dense_reference"


@dataclass(frozen=True)
class AttentionBackendCapabilities:
    """Import/runtime capabilities, separated for deterministic unit tests."""

    cuda_runtime: bool
    varlen_flash: bool
    flash_sdpa: bool
    flex_attention: bool


@lru_cache(maxsize=1)
def detect_attention_backend_capabilities() -> AttentionBackendCapabilities:
    """Inspect available APIs without executing an attention workload."""

    try:
        from torch.nn.attention.varlen import varlen_attn  # noqa: F401

        has_varlen = True
    except Exception:
        has_varlen = False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: F401

        has_flash_sdpa = hasattr(SDPBackend, "FLASH_ATTENTION")
    except Exception:
        has_flash_sdpa = False
    try:
        from torch.nn.attention.flex_attention import BlockMask, flex_attention  # noqa: F401

        has_flex = True
    except Exception:
        has_flex = False
    return AttentionBackendCapabilities(
        cuda_runtime=torch.cuda.is_available(),
        varlen_flash=has_varlen,
        flash_sdpa=has_flash_sdpa,
        flex_attention=has_flex,
    )


def select_attention_backend(
    pattern: AttentionPattern | str,
    *,
    device: torch.device | str,
    sequence_length: int,
    capabilities: AttentionBackendCapabilities | None = None,
    allow_dense_reference: bool = False,
    dense_reference_limit: int = DEFAULT_DENSE_REFERENCE_LIMIT,
) -> AttentionBackend:
    """Select a sparse/fused backend, refusing unsafe dense fallbacks.

    Dense execution is returned only when explicitly enabled *and* the sequence
    fits the caller's small-oracle limit.  Thus a missing CUDA kernel cannot
    accidentally turn an 8K local mask into quadratic attention.
    """

    try:
        pattern = AttentionPattern(pattern)
    except ValueError as error:
        raise ValueError(f"unknown attention pattern: {pattern!r}") from error
    device = torch.device(device)
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if dense_reference_limit <= 0:
        raise ValueError("dense_reference_limit must be positive")
    capabilities = capabilities or detect_attention_backend_capabilities()

    if device.type == "cuda" and capabilities.cuda_runtime:
        if pattern in {AttentionPattern.CAUSAL, AttentionPattern.CAUSAL_WINDOW}:
            if capabilities.varlen_flash:
                return AttentionBackend.VARLEN_FLASH
        elif pattern is AttentionPattern.BIDIRECTIONAL:
            if capabilities.flash_sdpa:
                return AttentionBackend.FLASH_SDPA
        elif pattern in {AttentionPattern.BRANCH, AttentionPattern.INTROSPECTION}:
            if capabilities.flex_attention:
                return AttentionBackend.FLEX

    if allow_dense_reference and sequence_length <= dense_reference_limit:
        return AttentionBackend.DENSE_REFERENCE

    reason = (
        f"no production backend for pattern={pattern.value!r}, device={str(device)!r}; "
        f"capabilities={capabilities}"
    )
    if sequence_length > dense_reference_limit:
        reason += (
            f"; refusing dense attention at length {sequence_length} "
            f"(reference limit {dense_reference_limit})"
        )
    elif not allow_dense_reference:
        reason += "; dense reference fallback was not explicitly enabled"
    raise RuntimeError(reason)


@contextmanager
def flash_sdpa_only() -> Iterator[None]:
    """Force Flash SDPA so ineligible shapes raise rather than fall back."""

    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        yield


_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune-no-cudagraphs",
    "max-autotune",
}


def compile_static_microstep(
    function: Callable[P, R],
    *,
    cuda_graphs: bool = False,
    mode: str | None = None,
    backend: str | Callable | None = None,
    name: str | None = None,
    disable: bool = False,
) -> Callable[P, R]:
    """Compile one fixed-shape microstep with a CUDA-graph-safe mode.

    The default keeps CUDA graphs off during kernel stabilization while still
    enabling max-autotune.  Setting ``cuda_graphs=True`` changes only the
    default mode to ``max-autotune``; callers remain responsible for persistent
    gradient buffers, static mutable addresses, warmup, and step markers.
    """

    selected_mode = mode or (
        "max-autotune" if cuda_graphs else "max-autotune-no-cudagraphs"
    )
    if selected_mode not in _COMPILE_MODES:
        raise ValueError(f"unsupported torch.compile mode: {selected_mode!r}")
    if cuda_graphs and selected_mode == "max-autotune-no-cudagraphs":
        raise ValueError("cuda_graphs=True conflicts with max-autotune-no-cudagraphs")
    if not cuda_graphs and selected_mode in {"max-autotune", "reduce-overhead"}:
        raise ValueError(
            f"mode={selected_mode!r} enables CUDA graphs; pass cuda_graphs=True"
        )
    return torch.compile(
        function,
        fullgraph=True,
        dynamic=False,
        mode=None if selected_mode == "default" else selected_mode,
        backend=backend,
        name=name,
        disable=disable,
    )


__all__ = [
    "ATOMIC_OUTPUT_SIZE",
    "AttentionBackend",
    "AttentionBackendCapabilities",
    "AttentionPattern",
    "CANVAS_REVEAL_SIZE",
    "DEFAULT_DENSE_REFERENCE_LIMIT",
    "categorical_entropy_argmax_confidence",
    "categorical_entropy_argmax_confidence_reference",
    "categorical_sample_entropy_argmax_confidence",
    "categorical_sample_entropy_argmax_confidence_reference",
    "compile_static_microstep",
    "detect_attention_backend_capabilities",
    "flash_sdpa_only",
    "reveal_low_entropy",
    "reveal_low_entropy_reference",
    "select_attention_backend",
    "triton_is_importable",
]

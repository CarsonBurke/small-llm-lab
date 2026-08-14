"""Portable exact Byte-Duo posterior sampling with lazy Triton fusion.

The production CUDA path fuses the active-state softmax, Eq. 10 posterior
numerator, and inverse-CDF draw.  It writes only the sampled atom, avoiding
the two dense ``[..., 261]`` probability/posterior intermediates used by the
transparent Torch oracle.  Importing this module does not import Triton or
initialize CUDA.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from .duo import duo_reverse_posterior


DUO_CLEAN_ATOMS = 261
DUO_SUPPORTED_ATOM_COUNTS = frozenset((257, 261))
_TRITON_BLOCK_SIZE = 512
_TRITON_DTYPES = {torch.bfloat16, torch.float16, torch.float32}

DuoPosteriorBackend = Literal["auto", "torch", "triton"]


def _validate_inputs(
    logits: Tensor,
    noisy_ids: Tensor,
    uniforms: Tensor,
    active: Tensor,
) -> None:
    if logits.ndim != 3 or logits.shape[-1] not in DUO_SUPPORTED_ATOM_COUNTS:
        raise ValueError(
            "Duo posterior logits must have shape [batch, canvas, 257 or 261], got "
            f"{tuple(logits.shape)}"
        )
    expected = logits.shape[:-1]
    for name, value in {
        "noisy_ids": noisy_ids,
        "uniforms": uniforms,
        "active": active,
    }.items():
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if value.device != logits.device:
            raise ValueError("all Duo posterior tensors must share a device")
    if not logits.is_floating_point():
        raise TypeError("Duo posterior logits must be floating point")
    if noisy_ids.dtype != torch.int64:
        raise TypeError("Duo noisy ids must be int64")
    if not uniforms.is_floating_point():
        raise TypeError("Duo posterior uniforms must be floating point")
    if active.dtype != torch.bool:
        raise TypeError("Duo posterior activity must be boolean")


def _broadcast_alpha_reference(
    value: Tensor | float,
    *,
    leading_shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.ndim == 0:
        return result
    batch, canvas = leading_shape
    if result.shape in {(batch,), (batch, 1)}:
        return result.reshape(batch, 1).expand(batch, canvas)
    if result.shape == leading_shape:
        return result
    raise ValueError("alpha must be scalar, per-row, or per-token")


def _validate_schedule_reference(alpha_s: Tensor, alpha_t: Tensor) -> None:
    if not bool(torch.isfinite(alpha_s).all()) or not bool(torch.isfinite(alpha_t).all()):
        raise ValueError("Duo reverse schedule endpoints must be finite")
    if bool(
        ((alpha_t < 0) | (alpha_s <= 0) | (alpha_t > alpha_s) | (alpha_s > 1)).any()
    ):
        raise ValueError(
            "Duo reverse transition requires 0 <= alpha_t <= alpha_s <= 1 "
            "and alpha_s > 0"
        )


def _scalar_schedule(
    alpha_s: Tensor | float, alpha_t: Tensor | float
) -> tuple[float, float] | None:
    """Return validated host scalars, or ``None`` for non-scalar tensors.

    Scalar CUDA tensors are deliberately not converted with ``.item()``: that
    would insert a device synchronization into every denoising transition.
    Production inference supplies Python endpoints, while general tensor
    schedules remain available through the Torch reference.
    """

    values: list[float] = []
    for value in (alpha_s, alpha_t):
        if torch.is_tensor(value):
            if value.ndim != 0 or value.device.type != "cpu":
                return None
            values.append(float(value))
        else:
            values.append(float(value))
    s, t = values
    if not math.isfinite(s) or not math.isfinite(t):
        raise ValueError("Duo reverse schedule endpoints must be finite")
    if not (0.0 <= t <= s <= 1.0) or s <= 0.0:
        raise ValueError(
            "Duo reverse transition requires 0 <= alpha_t <= alpha_s <= 1 "
            "and alpha_s > 0"
        )
    return s, t


def duo_posterior_sample_reference(
    logits: Tensor,
    noisy_ids: Tensor,
    alpha_s: Tensor | float,
    alpha_t: Tensor | float,
    uniforms: Tensor,
    active: Tensor,
    *,
    use_float64: bool = False,
) -> Tensor:
    """Strict Torch semantic oracle for one exact Eq. 10 transition.

    FP32 is intentional even for FP64 logits unless ``use_float64=True`` is
    explicitly requested for the numerical oracle.  ``uniforms`` is caller
    owned and selects the first class whose CDF is strictly greater than the
    draw.  Inactive
    slots retain their current ids exactly, including private PAD storage.
    Uniform values are a ``[0, 1)`` caller precondition; checking CUDA values
    here would impose a synchronization in every sampling step.
    """

    _validate_inputs(logits, noisy_ids, uniforms, active)
    work_dtype = torch.float64 if use_float64 else torch.float32
    a_s = _broadcast_alpha_reference(
        alpha_s,
        leading_shape=noisy_ids.shape,
        device=logits.device,
        dtype=work_dtype,
    )
    a_t = _broadcast_alpha_reference(
        alpha_t,
        leading_shape=noisy_ids.shape,
        device=logits.device,
        dtype=work_dtype,
    )
    _validate_schedule_reference(a_s, a_t)

    # Inactive storage may contain private PAD.  It is never sampled, but a
    # clean in-range placeholder keeps one_hot/gather in the exact reference
    # posterior well-defined.
    safe_noisy = torch.where(active, noisy_ids, torch.zeros_like(noisy_ids))
    clean_atoms = logits.shape[-1]
    if bool(((safe_noisy < 0) | (safe_noisy >= clean_atoms)).any()):
        raise ValueError("active Duo noisy ids must be clean atomic ids")
    clean_probabilities = F.softmax(logits.to(work_dtype), dim=-1)
    posterior = duo_reverse_posterior(
        clean_probabilities,
        safe_noisy,
        a_s,
        a_t,
        use_float64=use_float64,
    )
    cumulative = posterior.cumsum(-1)
    sampled = (cumulative <= uniforms.to(work_dtype).unsqueeze(-1)).sum(-1)
    sampled = sampled.clamp_max(clean_atoms - 1).to(torch.int64)
    return torch.where(active, sampled, noisy_ids)


@lru_cache(maxsize=1)
def triton_is_importable() -> bool:
    """Whether Triton is discoverable without importing it."""

    return importlib.util.find_spec("triton") is not None


_TRITON_DUO_POSTERIOR_KERNEL = None
_TRITON_IMPORT_ERROR: Exception | None = None


def _load_triton_duo_posterior_kernel():
    """Define the fused sampler only on the first eligible CUDA dispatch."""

    global _TRITON_DUO_POSTERIOR_KERNEL, _TRITON_IMPORT_ERROR
    # Triton's JIT resolves these symbols from the function module globals.
    global triton, tl  # type: ignore[no-redef]

    if _TRITON_DUO_POSTERIOR_KERNEL is not None:
        return _TRITON_DUO_POSTERIOR_KERNEL
    if _TRITON_IMPORT_ERROR is not None:
        raise RuntimeError("Triton Duo posterior kernel is unavailable") from _TRITON_IMPORT_ERROR
    try:
        import triton as triton  # type: ignore[no-redef]
        import triton.language as tl  # type: ignore[no-redef]

        @triton.jit
        def _duo_posterior_sample_kernel(
            logits,
            noisy_ids,
            uniforms,
            active,
            sampled_out,
            alpha_s,
            alpha_t,
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
            probabilities = exponentials / tl.sum(exponentials, axis=0)
            probabilities = probabilities / tl.sum(probabilities, axis=0)

            noisy = tl.load(noisy_ids + row).to(tl.int64)
            observed = columns == noisy
            ratio = alpha_t / alpha_s
            delta = alpha_s - alpha_t
            uniform_floor = (1.0 - ratio) * (1.0 - alpha_s) / VOCAB_SIZE
            numerator = (
                alpha_t * VOCAB_SIZE * probabilities * observed.to(tl.float32)
                + (ratio - alpha_t) * observed.to(tl.float32)
                + delta * probabilities
                + uniform_floor
            )
            # Eq. 10's denominator is shared by every class, so it cancels in
            # categorical normalization.  Only bounded endpoint roundoff is
            # removed, matching the exact Torch posterior.
            weights = tl.where(valid, tl.maximum(numerator, 0.0), 0.0)
            total = tl.sum(weights, axis=0)
            threshold = tl.load(uniforms + row).to(tl.float32) * total
            cumulative = tl.cumsum(weights, axis=0)
            sampled = tl.sum((valid & (cumulative <= threshold)).to(tl.int32), axis=0)
            sampled = tl.minimum(sampled, VOCAB_SIZE - 1)
            is_active = tl.load(active + row).to(tl.int1)
            tl.store(sampled_out + row, tl.where(is_active, sampled, noisy))

        _TRITON_DUO_POSTERIOR_KERNEL = _duo_posterior_sample_kernel
    except Exception as error:
        _TRITON_IMPORT_ERROR = error
        raise RuntimeError("Triton Duo posterior kernel is unavailable") from error
    return _TRITON_DUO_POSTERIOR_KERNEL


def _duo_posterior_sample_triton(
    logits: Tensor,
    noisy_ids: Tensor,
    alpha_s: float,
    alpha_t: float,
    uniforms: Tensor,
    active: Tensor,
) -> Tensor:
    if logits.device.type != "cuda":
        raise RuntimeError("the Triton Duo posterior kernel requires CUDA logits")
    if logits.dtype not in _TRITON_DTYPES:
        raise RuntimeError("the Triton Duo posterior kernel supports BF16, FP16, and FP32 logits")
    if logits.requires_grad and torch.is_grad_enabled():
        raise RuntimeError(
            "the Triton Duo posterior path is inference-only; use backend='torch' "
            "when gradients are required"
        )
    if logits.numel() == 0:
        return noisy_ids.clone()

    contiguous_logits = logits.contiguous()
    contiguous_noisy = noisy_ids.contiguous()
    contiguous_uniforms = uniforms.contiguous()
    contiguous_active = active.contiguous()
    sampled = torch.empty_like(contiguous_noisy)
    rows = contiguous_noisy.numel()
    kernel = _load_triton_duo_posterior_kernel()
    launcher = torch.library.wrap_triton(kernel) if torch.compiler.is_compiling() else kernel
    launcher[(rows,)](
        contiguous_logits,
        contiguous_noisy,
        contiguous_uniforms,
        contiguous_active,
        sampled,
        alpha_s,
        alpha_t,
        VOCAB_SIZE=logits.shape[-1],
        BLOCK_SIZE=_TRITON_BLOCK_SIZE,
        num_warps=4,
    )
    return sampled


def duo_posterior_sample(
    logits: Tensor,
    noisy_ids: Tensor,
    alpha_s: Tensor | float,
    alpha_t: Tensor | float,
    uniforms: Tensor,
    active: Tensor,
    *,
    backend: DuoPosteriorBackend = "auto",
    use_float64: bool = False,
) -> Tensor:
    """Sample one exact Duo reverse transition with fail-closed dispatch.

    ``auto`` uses the fused kernel for no-grad CUDA logits of a supported dtype
    and scalar host schedule endpoints.  General per-row/per-token schedules,
    CPU, differentiable calls, and the explicit FP64 oracle use Torch.
    ``backend='triton'`` never silently falls back.
    """

    _validate_inputs(logits, noisy_ids, uniforms, active)
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError(f"unknown Duo posterior backend: {backend!r}")
    scalar_schedule = _scalar_schedule(alpha_s, alpha_t)
    if backend == "torch":
        return duo_posterior_sample_reference(
            logits,
            noisy_ids,
            alpha_s,
            alpha_t,
            uniforms,
            active,
            use_float64=use_float64,
        )
    if backend == "triton":
        if use_float64:
            raise RuntimeError("the FP64 Duo oracle is available only with backend='torch'")
        if scalar_schedule is None:
            raise RuntimeError("the Triton Duo posterior kernel requires scalar host endpoints")
        if not triton_is_importable():
            raise RuntimeError("backend='triton' requested but Triton is not installed")
        return _duo_posterior_sample_triton(
            logits, noisy_ids, *scalar_schedule, uniforms, active
        )
    if (
        not use_float64
        and scalar_schedule is not None
        and logits.device.type == "cuda"
        and logits.dtype in _TRITON_DTYPES
        and (not logits.requires_grad or not torch.is_grad_enabled())
        and triton_is_importable()
    ):
        return _duo_posterior_sample_triton(
            logits, noisy_ids, *scalar_schedule, uniforms, active
        )
    return duo_posterior_sample_reference(
        logits,
        noisy_ids,
        alpha_s,
        alpha_t,
        uniforms,
        active,
        use_float64=use_float64,
    )


__all__ = (
    "DUO_CLEAN_ATOMS",
    "DuoPosteriorBackend",
    "duo_posterior_sample",
    "duo_posterior_sample_reference",
    "triton_is_importable",
)

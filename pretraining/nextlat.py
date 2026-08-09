"""Reusable, one-step NextLat dynamics and auxiliary losses.

The dynamics block follows the bias-free configuration from the official
NextLat implementation.  Its ``LayerNorm(..., bias=False)`` is an RMSNorm in
that codebase, so :class:`torch.nn.RMSNorm` is used here deliberately.

The terminal loss mirrors the official implementation while bounding its
activation memory.  It projects at most a configurable number of teacher
tokens at a time, packing complete short sequences or segmenting long ones.
Each teacher projection is reused for next-token cross-entropy and detached
KL, and compact first-order gradients are computed in the forward terminal.
Consequently no ``[batch * sequence, vocabulary]`` tensor survives the
forward pass, while each projection remains a compiler-visible dense GEMM.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import triton
import triton.language as tl


_TRITON_VOCAB_BLOCK_SIZE = 1024


@triton.jit
def _softcap_triton(raw, softcap: tl.constexpr):
    denominator = raw * raw + softcap * softcap
    return softcap * raw * tl.rsqrt(denominator)


@triton.jit
def _softcap_derivative_triton(raw, softcap: tl.constexpr):
    denominator = raw * raw + softcap * softcap
    inverse_root = tl.rsqrt(denominator)
    return softcap * softcap * softcap * inverse_root / denominator


@triton.jit
def _nextlat_kl_value_vjp_kernel(
    teacher_raw,
    student_raw,
    kl_mask,
    row_kl,
    inverse_kl_normalizer,
    teacher_rows_per_sequence: tl.constexpr,
    kl_rows_per_sequence: tl.constexpr,
    teacher_offset: tl.constexpr,
    vocab_size: tl.constexpr,
    softcap: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    student_row = tl.program_id(0)
    sequence = student_row // kl_rows_per_sequence
    position = student_row - sequence * kl_rows_per_sequence
    teacher_row = sequence * teacher_rows_per_sequence + teacher_offset + position
    offsets = tl.arange(0, BLOCK_V)
    teacher_max = float("-inf")
    teacher_sum = 0.0
    student_max = float("-inf")
    student_sum = 0.0
    for vocab_start in tl.range(0, vocab_size, BLOCK_V):
        columns = vocab_start + offsets
        valid = columns < vocab_size
        teacher = tl.load(
            teacher_raw + teacher_row * vocab_size + columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        student = tl.load(
            student_raw + student_row * vocab_size + columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        teacher_capped = tl.where(
            valid, _softcap_triton(teacher, softcap), float("-inf")
        )
        student_capped = tl.where(
            valid, _softcap_triton(student, softcap), float("-inf")
        )
        teacher_block_max = tl.max(teacher_capped, axis=0)
        teacher_new_max = tl.maximum(teacher_max, teacher_block_max)
        teacher_sum = teacher_sum * tl.exp(teacher_max - teacher_new_max) + tl.sum(
            tl.exp(teacher_capped - teacher_new_max), axis=0
        )
        teacher_max = teacher_new_max
        student_block_max = tl.max(student_capped, axis=0)
        student_new_max = tl.maximum(student_max, student_block_max)
        student_sum = student_sum * tl.exp(student_max - student_new_max) + tl.sum(
            tl.exp(student_capped - student_new_max), axis=0
        )
        student_max = student_new_max
    teacher_log_z = teacher_max + tl.log(teacher_sum)
    student_log_z = student_max + tl.log(student_sum)

    included = tl.load(kl_mask + student_row) != 0
    inverse_normalizer = tl.load(inverse_kl_normalizer)
    kl_value = 0.0
    for vocab_start in tl.range(0, vocab_size, BLOCK_V):
        columns = vocab_start + offsets
        valid = columns < vocab_size
        teacher = tl.load(
            teacher_raw + teacher_row * vocab_size + columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        student = tl.load(
            student_raw + student_row * vocab_size + columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        teacher_capped = _softcap_triton(teacher, softcap)
        student_capped = _softcap_triton(student, softcap)
        teacher_probability = tl.exp(teacher_capped - teacher_log_z)
        student_probability = tl.exp(student_capped - student_log_z)
        kl_terms = teacher_probability * (
            teacher_capped - teacher_log_z - student_capped + student_log_z
        )
        kl_value += tl.sum(tl.where(valid & included, kl_terms, 0.0), axis=0)
        student_gradient = (
            (student_probability - teacher_probability)
            * _softcap_derivative_triton(student, softcap)
            * inverse_normalizer
        )
        student_gradient = tl.where(included, student_gradient, 0.0)
        tl.store(
            student_raw + student_row * vocab_size + columns,
            student_gradient,
            mask=valid,
        )
    tl.store(row_kl + student_row, kl_value)


@triton.jit
def _nextlat_ce_value_vjp_kernel(
    teacher_raw,
    targets,
    row_ce,
    vocab_size: tl.constexpr,
    softcap: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_V)
    target = tl.load(targets + row)
    included = target != -100
    row_max = float("-inf")
    row_sum = 0.0
    target_capped = 0.0
    for vocab_start in tl.range(0, vocab_size, BLOCK_V):
        columns = vocab_start + offsets
        valid = columns < vocab_size
        raw = tl.load(
            teacher_raw + row * vocab_size + columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        capped = tl.where(valid, _softcap_triton(raw, softcap), float("-inf"))
        block_max = tl.max(capped, axis=0)
        new_max = tl.maximum(row_max, block_max)
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(
            tl.exp(capped - new_max), axis=0
        )
        row_max = new_max
        target_capped += tl.sum(tl.where(columns == target, capped, 0.0), axis=0)
    log_z = row_max + tl.log(row_sum)
    tl.store(row_ce + row, tl.where(included, log_z - target_capped, 0.0))

    for vocab_start in tl.range(0, vocab_size, BLOCK_V):
        columns = vocab_start + offsets
        valid = columns < vocab_size
        raw = tl.load(
            teacher_raw + row * vocab_size + columns,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        capped = _softcap_triton(raw, softcap)
        probability = tl.exp(capped - log_z)
        capped_gradient = (
            probability - (columns == target).to(tl.float32)
        ) * included
        raw_gradient = capped_gradient * _softcap_derivative_triton(raw, softcap)
        tl.store(
            teacher_raw + row * vocab_size + columns,
            raw_gradient,
            mask=valid,
        )


def _validate_positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return value


class _CastedBiasFreeLinear(nn.Linear):
    """Bias-free linear with fp32 masters and activation-dtype execution."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.weight.to(dtype=inputs.dtype))


class _CastedRMSNorm(nn.RMSNorm):
    """RMSNorm with the same fp32-master mixed-precision contract."""

    def forward(self, inputs: Tensor) -> Tensor:
        return F.rms_norm(
            inputs,
            self.normalized_shape,
            self.weight.to(dtype=inputs.dtype) if self.weight is not None else None,
            self.eps,
        )


class NextLatDynamicsModel(nn.Module):
    """Paper-faithful ``d=1`` residual latent dynamics model.

    ``next_token_latent`` is concatenated before ``current_hidden``, matching
    the official implementation.  All leading dimensions are treated as
    batch dimensions.
    """

    def __init__(
        self,
        model_dim: int,
        proj_factor: float = 1.0,
        *,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if isinstance(model_dim, bool) or not isinstance(model_dim, int) or model_dim <= 0:
            raise ValueError(f"model_dim must be a positive integer, got {model_dim!r}")
        proj_factor = _validate_positive_finite(proj_factor, "proj_factor")
        eps = _validate_positive_finite(eps, "eps")

        input_dim = 2 * model_dim
        hidden_dim = 128 * round(proj_factor * input_dim / 128)
        if hidden_dim <= 0:
            raise ValueError(
                "proj_factor rounds the dynamics hidden width to zero; "
                f"got model_dim={model_dim}, proj_factor={proj_factor}"
            )

        self.model_dim = model_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # NextLat's custom LayerNorm dispatches to RMSNorm when bias=False.
        self.norm_x = _CastedRMSNorm(input_dim, eps=eps)
        self.mlp = nn.Sequential(
            _CastedBiasFreeLinear(input_dim, hidden_dim),
            nn.GELU(),
            _CastedBiasFreeLinear(hidden_dim, hidden_dim),
            nn.GELU(),
            _CastedBiasFreeLinear(hidden_dim, model_dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, current_hidden: Tensor, next_token_latent: Tensor) -> Tensor:
        if current_hidden.shape != next_token_latent.shape:
            raise ValueError(
                "current_hidden and next_token_latent must have identical shapes, "
                f"got {tuple(current_hidden.shape)} and {tuple(next_token_latent.shape)}"
            )
        if current_hidden.ndim < 1 or current_hidden.shape[-1] != self.model_dim:
            raise ValueError(
                f"dynamics inputs must end in model_dim={self.model_dim}, "
                f"got {tuple(current_hidden.shape)}"
            )
        if current_hidden.device != next_token_latent.device:
            raise ValueError("current_hidden and next_token_latent must share a device")
        if current_hidden.dtype != next_token_latent.dtype:
            raise ValueError("current_hidden and next_token_latent must share a dtype")
        if not current_hidden.is_floating_point():
            raise TypeError("dynamics inputs must be floating-point tensors")

        inputs = torch.cat((next_token_latent, current_hidden), dim=-1)
        return current_hidden + self.mlp(self.norm_x(inputs))


def rational_softcap(logits: Tensor, softcap: float = 15.0) -> Tensor:
    """Apply the pretraining model's exact rational logit softcap."""

    softcap = _validate_positive_finite(softcap, "softcap")
    return softcap * logits * (logits.square() + softcap**2).rsqrt()


def _head_parameters(vocab_head: nn.Linear | Tensor) -> tuple[Tensor, Tensor | None]:
    if isinstance(vocab_head, nn.Linear):
        weight = vocab_head.weight
        bias = vocab_head.bias
    elif torch.is_tensor(vocab_head):
        weight = vocab_head
        bias = None
    else:
        raise TypeError("vocab_head must be an nn.Linear or a [vocab, dim] Tensor")
    if weight.ndim != 2 or weight.shape[0] <= 0:
        raise ValueError(
            f"vocab head weight must be nonempty [vocab, dim], got {tuple(weight.shape)}"
        )
    if not weight.is_floating_point():
        raise TypeError("vocab head weight must be floating point")
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError(
            f"vocab head bias must have shape {(weight.shape[0],)}, got {tuple(bias.shape)}"
        )
    return weight, bias


def document_transition_mask(document_ids: Tensor) -> Tensor:
    """Return ``True`` for adjacent positions within the same document.

    The sequence dimension is the final dimension.  This representation is
    independent of whether a tokenizer assigns a boundary token to the
    preceding or following document.
    """

    if document_ids.ndim < 1:
        raise ValueError("document_ids must have at least one dimension")
    if document_ids.shape[-1] < 2:
        raise ValueError("document_ids must contain at least two sequence positions")
    return document_ids[..., :-1] == document_ids[..., 1:]


def masked_smooth_l1(
    predicted_hidden: Tensor,
    actual_hidden: Tensor,
    transition_mask: Tensor,
) -> Tensor:
    """Smooth-L1 mean over valid within-document hidden transitions."""

    if predicted_hidden.shape != actual_hidden.shape:
        raise ValueError(
            "predicted_hidden and actual_hidden must have identical shapes, "
            f"got {tuple(predicted_hidden.shape)} and {tuple(actual_hidden.shape)}"
        )
    if predicted_hidden.ndim < 1 or not predicted_hidden.is_floating_point():
        raise TypeError("hidden tensors must be floating point and have a feature dimension")
    if not actual_hidden.is_floating_point():
        raise TypeError("hidden tensors must be floating point")
    if predicted_hidden.device != actual_hidden.device:
        raise ValueError("hidden tensors must share a device")
    if transition_mask.shape != predicted_hidden.shape[:-1]:
        raise ValueError(
            "transition_mask must have the hidden leading shape "
            f"{tuple(predicted_hidden.shape[:-1])}, got {tuple(transition_mask.shape)}"
        )
    if transition_mask.dtype != torch.bool:
        raise TypeError("transition_mask must be boolean")
    if transition_mask.device != predicted_hidden.device:
        raise ValueError("transition_mask and hidden tensors must share a device")

    implementation = (
        _compiled_masked_smooth_l1_unchecked
        if predicted_hidden.is_cuda
        else _masked_smooth_l1_unchecked
    )
    return implementation(predicted_hidden, actual_hidden, transition_mask)


def _masked_smooth_l1_unchecked(
    predicted_hidden: Tensor,
    actual_hidden: Tensor,
    transition_mask: Tensor,
) -> Tensor:
    elementwise = F.smooth_l1_loss(
        predicted_hidden.float(), actual_hidden.detach().float(), reduction="none"
    )
    weights = transition_mask[..., None].to(dtype=torch.float32)
    denominator = weights.expand_as(elementwise).sum().clamp_min(1.0)
    return (elementwise * weights).sum() / denominator


_compiled_masked_smooth_l1_unchecked = torch.compile(
    _masked_smooth_l1_unchecked,
    fullgraph=True,
    dynamic=False,
)


def _rational_softcap_unchecked(logits: Tensor, softcap: float) -> Tensor:
    return softcap * logits * (logits.square() + softcap**2).rsqrt()


def _terminal_logits(
    hidden: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    softcap: float,
) -> tuple[Tensor, Tensor]:
    """Return execution-dtype raw logits and fp32 capped logits."""

    execution_weight = weight.to(dtype=hidden.dtype)
    execution_bias = None if bias is None else bias.to(dtype=hidden.dtype)
    raw_logits = F.linear(hidden, execution_weight, execution_bias)
    capped_logits = _rational_softcap_unchecked(raw_logits.float(), softcap)
    return raw_logits, capped_logits


def _terminal_chunk_objective(
    hidden: Tensor,
    predicted_hidden_for_kl: Tensor,
    targets: Tensor,
    kl_mask: Tensor,
    weight: Tensor,
    bias_or_empty: Tensor,
    softcap: float,
    kl_normalizer: Tensor,
    has_bias: bool,
    teacher_kl_start: int,
    teacher_kl_stop: int,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Combined chunk scalar whose gradient paths are intentionally disjoint."""

    bias = bias_or_empty if has_bias else None
    _, teacher_logits = _terminal_logits(hidden, weight, bias, softcap)
    ce_sum = F.cross_entropy(
        teacher_logits.reshape(-1, teacher_logits.shape[-1]),
        targets.reshape(-1),
        reduction="sum",
    )
    if predicted_hidden_for_kl.shape[1] > 0:
        # Detaching the shared head on the student path and the main logits on
        # the teacher path makes the VJP separable: actual-state/head grads
        # belong exclusively to CE, and predicted-state grads exclusively to
        # KL.  This lets the terminal save each compact gradient independently
        # for arbitrary upstream scaling in its custom backward.
        _, student_logits = _terminal_logits(
            predicted_hidden_for_kl,
            weight.detach(),
            None if bias is None else bias.detach(),
            softcap,
        )
        teacher_log_probs = F.log_softmax(
            teacher_logits[:, teacher_kl_start:teacher_kl_stop].detach(), dim=-1
        )
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        per_token_kl = (
            teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1)
        mask = kl_mask.to(torch.float32)
        kl_loss = (per_token_kl * mask).sum() / kl_normalizer
    else:
        kl_loss = ce_sum.new_zeros(())
    return ce_sum + kl_loss, (ce_sum, kl_loss)


_terminal_chunk_grad_and_value = torch.func.grad_and_value(
    _terminal_chunk_objective,
    argnums=(0, 1, 4, 5),
    has_aux=True,
)
# Retain the original compiled dense terminal as an executable parity oracle.
# Production CUDA dispatches to the fused-linear Triton implementation below;
# CPU tests use the eager transform so unit tests remain lightweight.
_compiled_terminal_chunk_grad_and_value = torch.compile(
    _terminal_chunk_grad_and_value,
    fullgraph=True,
    dynamic=False,
)


def _triton_terminal_chunk_grad_and_value(
    hidden: Tensor,
    predicted_hidden_for_kl: Tensor,
    targets: Tensor,
    kl_mask: Tensor,
    execution_weight: Tensor,
    execution_bias_or_empty: Tensor,
    softcap: float,
    kl_normalizer: Tensor,
    has_bias: bool,
    teacher_kl_start: int,
    teacher_kl_stop: int,
) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, tuple[Tensor, Tensor]]]:
    """Five-GEMM CUDA terminal with fused row-value/VJP Triton kernels."""

    batch_size, teacher_rows_per_sequence, hidden_dim = hidden.shape
    kl_rows_per_sequence = predicted_hidden_for_kl.shape[1]
    if teacher_kl_stop - teacher_kl_start != kl_rows_per_sequence:
        raise ValueError("teacher and predicted KL slices must have equal lengths")
    vocab_size = execution_weight.shape[0]
    execution_bias = execution_bias_or_empty if has_bias else None
    hidden_flat = hidden.reshape(-1, hidden_dim)
    targets_flat = targets.contiguous().view(-1)

    # GEMM 1: shared main/teacher projection. It stays in execution dtype and
    # is overwritten with dCE/d(raw logits) after serving both objectives.
    teacher_raw = F.linear(hidden_flat, execution_weight, execution_bias)

    if kl_rows_per_sequence:
        predicted_flat = predicted_hidden_for_kl.reshape(-1, hidden_dim)
        # GEMM 2: detached-head student projection, flattened so cuBLAS uses
        # MM rather than the broadcast-BMM selected for a [B,T,D] input.
        student_raw = F.linear(predicted_flat, execution_weight, execution_bias)
        row_kl = torch.empty(
            student_raw.shape[0], device=student_raw.device, dtype=torch.float32
        )
        inverse_kl_normalizer = kl_normalizer.reciprocal()
        _nextlat_kl_value_vjp_kernel[(student_raw.shape[0],)](
            teacher_raw,
            student_raw,
            kl_mask.contiguous(),
            row_kl,
            inverse_kl_normalizer,
            teacher_rows_per_sequence=teacher_rows_per_sequence,
            kl_rows_per_sequence=kl_rows_per_sequence,
            teacher_offset=teacher_kl_start,
            vocab_size=vocab_size,
            softcap=softcap,
            BLOCK_V=_TRITON_VOCAB_BLOCK_SIZE,
            num_warps=8,
            num_stages=1,
        )
        kl_loss = row_kl.sum(dtype=torch.float32) / kl_normalizer
        # GEMM 3: compact predicted-state gradient from the overwritten
        # dKL/d(raw student logits) buffer.
        grad_predicted = torch.mm(student_raw, execution_weight).reshape_as(
            predicted_hidden_for_kl
        )
    else:
        kl_loss = teacher_raw.new_zeros((), dtype=torch.float32)
        grad_predicted = torch.zeros_like(predicted_hidden_for_kl)

    row_ce = torch.empty(
        teacher_raw.shape[0], device=teacher_raw.device, dtype=torch.float32
    )
    _nextlat_ce_value_vjp_kernel[(teacher_raw.shape[0],)](
        teacher_raw,
        targets_flat,
        row_ce,
        vocab_size=vocab_size,
        softcap=softcap,
        BLOCK_V=_TRITON_VOCAB_BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )
    ce_sum = row_ce.sum(dtype=torch.float32)
    # GEMMs 4 and 5: compact CE gradients. teacher_raw now stores BF16
    # dCE/d(raw logits), so no fp32 vocabulary-sized value survives.
    grad_hidden = torch.mm(teacher_raw, execution_weight).reshape_as(hidden)
    grad_weight = torch.mm(teacher_raw.t(), hidden_flat)
    grad_bias = (
        teacher_raw.sum(dim=0)
        if has_bias
        else execution_bias_or_empty.new_empty(0)
    )
    gradients = (grad_hidden, grad_predicted, grad_weight, grad_bias)
    return gradients, (ce_sum + kl_loss, (ce_sum, kl_loss))


class _NextLatTerminal(torch.autograd.Function):
    """Memory-bounded, first-order CE plus forward-KL terminal.

    A compiled per-chunk VJP is evaluated during forward and only compact
    gradients are saved.  Backward therefore performs no vocabulary work at
    all.  The VJP deliberately sends KL gradients only through the predicted
    state: teacher probabilities and the student head are stop-gradient.
    """

    @staticmethod
    @torch.compiler.disable(recursive=True)
    def forward(
        ctx,
        hidden: Tensor,
        predicted_hidden: Tensor,
        targets: Tensor,
        transition_mask: Tensor,
        weight: Tensor,
        bias_or_empty: Tensor,
        has_bias: bool,
        softcap: float,
        token_chunk_size: int,
    ) -> tuple[Tensor, Tensor]:
        batch_size, sequence_length, _ = hidden.shape
        kl_normalizer = (
            transition_mask[:, :-1].sum().clamp_min(1).to(dtype=torch.float32)
        )
        ce_sum = hidden.new_zeros((), dtype=torch.float32)
        kl_loss = hidden.new_zeros((), dtype=torch.float32)
        grad_hidden = torch.zeros_like(hidden)
        grad_predicted = torch.zeros_like(predicted_hidden)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias_or_empty)
        # The projection keeps fp32 optimizer masters, but all head GEMMs run
        # in the activation dtype. Cast the large vocabulary matrix once per
        # terminal rather than once per sequence chunk.
        execution_weight = weight.to(dtype=hidden.dtype)
        execution_bias_or_empty = bias_or_empty.to(dtype=hidden.dtype)
        chunk_grad_and_value = (
            _triton_terminal_chunk_grad_and_value
            if hidden.is_cuda
            else _terminal_chunk_grad_and_value
        )

        def accumulate_chunk(
            hidden_chunk: Tensor,
            predicted_kl_chunk: Tensor,
            targets_chunk: Tensor,
            kl_mask_chunk: Tensor,
            *,
            batch_start: int,
            batch_stop: int,
            hidden_start: int,
            hidden_stop: int,
            predicted_start: int,
            predicted_stop: int,
            teacher_kl_start: int,
            teacher_kl_stop: int,
        ) -> None:
            nonlocal ce_sum, kl_loss
            chunk_gradients, (_, chunk_components) = chunk_grad_and_value(
                hidden_chunk,
                predicted_kl_chunk,
                targets_chunk,
                kl_mask_chunk,
                execution_weight,
                execution_bias_or_empty,
                softcap,
                kl_normalizer,
                has_bias,
                teacher_kl_start,
                teacher_kl_stop,
            )
            chunk_grad_hidden, chunk_grad_predicted, chunk_grad_weight, chunk_grad_bias = (
                chunk_gradients
            )
            chunk_ce, chunk_kl = chunk_components
            ce_sum = ce_sum + chunk_ce
            kl_loss = kl_loss + chunk_kl
            grad_hidden[
                batch_start:batch_stop, hidden_start:hidden_stop
            ].copy_(chunk_grad_hidden)
            grad_predicted[
                batch_start:batch_stop, predicted_start:predicted_stop
            ].copy_(chunk_grad_predicted)
            # add_ casts the execution-dtype compact gradients directly into
            # the fp32 master accumulator without a temporary fp32 copy.
            grad_weight.add_(chunk_grad_weight)
            if has_bias:
                grad_bias.add_(chunk_grad_bias)

        if sequence_length <= token_chunk_size:
            # Pack as many complete sequences as fit. For the production
            # 2,048-token context and default 4,096-token budget this retains
            # the two-sequence static chunk used by the optimized run.
            sequences_per_chunk = max(1, token_chunk_size // sequence_length)
            for batch_start in range(0, batch_size, sequences_per_chunk):
                batch_stop = min(batch_start + sequences_per_chunk, batch_size)
                accumulate_chunk(
                    hidden[batch_start:batch_stop],
                    predicted_hidden[batch_start:batch_stop, :-1],
                    targets[batch_start:batch_stop],
                    transition_mask[batch_start:batch_stop, :-1],
                    batch_start=batch_start,
                    batch_stop=batch_stop,
                    hidden_start=0,
                    hidden_stop=sequence_length,
                    predicted_start=0,
                    predicted_stop=sequence_length - 2,
                    teacher_kl_start=1,
                    teacher_kl_stop=sequence_length - 1,
                )
        else:
            # Segment one sequence at a time. CE covers every hidden position.
            # KL's eligible teacher positions are 1..T-2; the first, middle,
            # and last slices below partition that interval without gaps or
            # overlap while pairing teacher j with predicted j-1.
            for batch_index in range(batch_size):
                for hidden_start in range(0, sequence_length, token_chunk_size):
                    hidden_stop = min(hidden_start + token_chunk_size, sequence_length)
                    chunk_length = hidden_stop - hidden_start
                    if hidden_start == 0:
                        teacher_kl_start = 1
                        teacher_kl_stop = chunk_length
                        predicted_start = 0
                        predicted_stop = hidden_stop - 1
                    elif hidden_stop == sequence_length:
                        teacher_kl_start = 0
                        teacher_kl_stop = chunk_length - 1
                        predicted_start = hidden_start - 1
                        predicted_stop = sequence_length - 2
                    else:
                        teacher_kl_start = 0
                        teacher_kl_stop = chunk_length
                        predicted_start = hidden_start - 1
                        predicted_stop = hidden_stop - 1
                    accumulate_chunk(
                        hidden[
                            batch_index : batch_index + 1,
                            hidden_start:hidden_stop,
                        ],
                        predicted_hidden[
                            batch_index : batch_index + 1,
                            predicted_start:predicted_stop,
                        ],
                        targets[
                            batch_index : batch_index + 1,
                            hidden_start:hidden_stop,
                        ],
                        transition_mask[
                            batch_index : batch_index + 1,
                            predicted_start:predicted_stop,
                        ],
                        batch_start=batch_index,
                        batch_stop=batch_index + 1,
                        hidden_start=hidden_start,
                        hidden_stop=hidden_stop,
                        predicted_start=predicted_start,
                        predicted_stop=predicted_stop,
                        teacher_kl_start=teacher_kl_start,
                        teacher_kl_stop=teacher_kl_stop,
                    )

        ctx.set_materialize_grads(False)
        ctx.has_bias = has_bias
        ctx.save_for_backward(
            grad_hidden,
            grad_predicted,
            grad_weight,
            grad_bias,
        )
        return ce_sum, kl_loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_ce_sum: Tensor | None, grad_kl_loss: Tensor | None):
        grad_hidden, grad_predicted, grad_weight, grad_bias = ctx.saved_tensors
        if grad_ce_sum is None:
            grad_ce_sum = grad_hidden.new_zeros((), dtype=torch.float32)
        if grad_kl_loss is None:
            grad_kl_loss = grad_predicted.new_zeros((), dtype=torch.float32)

        return (
            grad_hidden * grad_ce_sum,
            grad_predicted * grad_kl_loss,
            None,
            None,
            grad_weight * grad_ce_sum,
            grad_bias * grad_ce_sum if ctx.has_bias else None,
            None,
            None,
            None,
        )


@dataclass(frozen=True)
class NextLatTerminalLoss:
    """Main CE sum and separately loggable NextLat auxiliary means."""

    ce_sum: Tensor
    hidden_loss: Tensor
    kl_loss: Tensor
    total: Tensor

    def detached_metrics(self, prefix: str = "nextlat") -> Mapping[str, Tensor]:
        prefix = f"{prefix}/" if prefix else ""
        return {
            f"{prefix}hidden_loss": self.hidden_loss.detach(),
            f"{prefix}kl_loss": self.kl_loss.detach(),
            f"{prefix}total": self.total.detach(),
        }


def nextlat_terminal_loss(
    hidden: Tensor,
    predicted_hidden: Tensor,
    targets: Tensor,
    vocab_head: nn.Linear | Tensor,
    *,
    transition_mask: Tensor,
    hidden_weight: float = 1.0,
    kl_weight: float = 1.0,
    softcap: float = 15.0,
    token_chunk_size: int = 4096,
) -> NextLatTerminalLoss:
    """Compute the exact, memory-bounded paper-default ``d=1`` terminal.

    ``hidden`` supplies the main CE logits. ``predicted_hidden[:, i]`` is
    aligned with ``hidden[:, i + 1]`` for Smooth-L1, while KL compares their
    next-token distributions for all but the final predicted state. Main CE
    is a token sum; hidden regression and KL are independently masked means.
    ``token_chunk_size`` bounds teacher projection rows. Complete sequences
    are packed when possible; longer sequences are partitioned without
    dropping or duplicating CE or KL positions.

    The returned ``total`` is the weighted auxiliary mean.  A sum-reduction
    trainer should optimize ``ce_sum + targets.numel() * total``.

    This terminal is an eager orchestration boundary around independently
    compiled CUDA helpers; do not place the call inside an enclosing
    ``torch.compile(fullgraph=True)`` region.
    """

    hidden_weight = float(hidden_weight)
    kl_weight = float(kl_weight)
    if not math.isfinite(hidden_weight) or hidden_weight < 0.0:
        raise ValueError(f"hidden_weight must be finite and nonnegative, got {hidden_weight}")
    if not math.isfinite(kl_weight) or kl_weight < 0.0:
        raise ValueError(f"kl_weight must be finite and nonnegative, got {kl_weight}")
    softcap = _validate_positive_finite(softcap, "softcap")
    if (
        isinstance(token_chunk_size, bool)
        or not isinstance(token_chunk_size, int)
        or token_chunk_size <= 0
    ):
        raise ValueError(
            "token_chunk_size must be a positive integer, "
            f"got {token_chunk_size!r}"
        )
    if hidden.ndim != 3 or not hidden.is_floating_point():
        raise TypeError("hidden must be a floating-point [batch, sequence, dim] tensor")
    batch_size, sequence_length, hidden_dim = hidden.shape
    if batch_size <= 0 or sequence_length < 2 or hidden_dim <= 0:
        raise ValueError(
            "hidden must have nonempty batch/dim and at least two sequence positions"
        )
    expected_predicted_shape = (batch_size, sequence_length - 1, hidden_dim)
    if predicted_hidden.shape != expected_predicted_shape:
        raise ValueError(
            f"predicted_hidden must have shape {expected_predicted_shape}, "
            f"got {tuple(predicted_hidden.shape)}"
        )
    if not predicted_hidden.is_floating_point():
        raise TypeError("predicted_hidden must be floating point")
    if predicted_hidden.dtype != hidden.dtype:
        raise ValueError("hidden and predicted_hidden must share a dtype")
    if targets.shape != (batch_size, sequence_length):
        raise ValueError(
            f"targets must have shape {(batch_size, sequence_length)}, "
            f"got {tuple(targets.shape)}"
        )
    if targets.dtype != torch.long:
        raise TypeError("targets must have dtype torch.long")
    if transition_mask.shape != (batch_size, sequence_length - 1):
        raise ValueError(
            "transition_mask must have shape "
            f"{(batch_size, sequence_length - 1)}, got {tuple(transition_mask.shape)}"
        )
    if transition_mask.dtype != torch.bool:
        raise TypeError("transition_mask must be boolean")

    weight, bias = _head_parameters(vocab_head)
    if weight.shape[1] != hidden_dim:
        raise ValueError(
            f"vocab head input dim {weight.shape[1]} does not match hidden dim {hidden_dim}"
        )
    tensors = (predicted_hidden, targets, transition_mask, weight)
    if any(tensor.device != hidden.device for tensor in tensors) or (
        bias is not None and bias.device != hidden.device
    ):
        raise ValueError("all terminal inputs and vocab-head parameters must share a device")
    if bias is not None and not bias.is_floating_point():
        raise TypeError("vocab head bias must be floating point")

    bias_or_empty = bias
    if bias_or_empty is None:
        bias_or_empty = weight.new_empty(0)
    ce_sum, kl_loss = _NextLatTerminal.apply(
        hidden,
        predicted_hidden,
        targets,
        transition_mask,
        weight,
        bias_or_empty,
        bias is not None,
        softcap,
        token_chunk_size,
    )
    hidden_loss = masked_smooth_l1(
        predicted_hidden,
        hidden[:, 1:],
        transition_mask,
    )
    total = hidden_weight * hidden_loss + kl_weight * kl_loss
    return NextLatTerminalLoss(
        ce_sum=ce_sum,
        hidden_loss=hidden_loss,
        kl_loss=kl_loss,
        total=total,
    )


__all__ = [
    "NextLatDynamicsModel",
    "NextLatTerminalLoss",
    "document_transition_mask",
    "masked_smooth_l1",
    "nextlat_terminal_loss",
    "rational_softcap",
]

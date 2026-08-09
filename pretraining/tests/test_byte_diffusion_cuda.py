"""Queued CUDA parity checks; never run these outside mlq."""

from __future__ import annotations

import math
import os

import pytest
import torch

from pretraining.byte_diffusion.attention import (
    CanvasBranchLayout,
    PackedCleanQKV,
    branch_attention,
    dense_packed_clean_attention,
    packed_clean_attention,
)
from pretraining.byte_diffusion.config import ByteDiffusionConfig, CorruptionConfig
from pretraining.byte_diffusion.data import (
    AtomicDocument,
    AtomicIdManifest,
    DeterministicChunkCursor,
    pack_documents,
)
from pretraining.byte_diffusion.kernels import (
    categorical_entropy_argmax_confidence,
    categorical_sample_entropy_argmax_confidence,
    reveal_low_entropy,
)
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.training import ByteDiffusionTrainer, TrainingRunConfig


pytestmark = pytest.mark.skipif(
    os.environ.get("BYTE_DIFFUSION_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="queued CUDA validation requires BYTE_DIFFUSION_CUDA_TESTS=1",
)


def test_varlen_flash_forward_backward_matches_dense_oracle() -> None:
    torch.manual_seed(101)
    device = torch.device("cuda")
    q = torch.randn(12, 2, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(12, 2, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(12, 2, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    packed = PackedCleanQKV(
        q,
        k,
        v,
        torch.tensor([0, 5, 12], device=device, dtype=torch.int32),
        torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6], device=device),
        7,
    )
    flash = packed_clean_attention(packed, window=4)
    dense = dense_packed_clean_attention(packed, window=4)
    torch.testing.assert_close(flash, dense, rtol=3e-2, atol=3e-2)
    upstream = torch.randn_like(flash)
    flash_grads = torch.autograd.grad((flash * upstream).sum(), (q, k, v), retain_graph=True)
    dense_grads = torch.autograd.grad((dense * upstream).sum(), (q, k, v))
    for actual, expected in zip(flash_grads, dense_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def test_flex_shared_branch_forward_backward_matches_dense_oracle() -> None:
    torch.manual_seed(103)
    device = torch.device("cuda")
    clean_length, branches, canvas, heads, dim = 64, 2, 32, 2, 64
    clean_valid = torch.ones(1, clean_length, device=device, dtype=torch.bool)
    branch_valid = torch.ones(1, branches, canvas, device=device, dtype=torch.bool)
    starts = torch.tensor([[16, 32]], device=device)
    layout = CanvasBranchLayout(clean_valid, branch_valid, starts)
    query = torch.randn(1, heads, branches * canvas, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    clean_k = torch.randn(1, heads, clean_length, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    clean_v = torch.randn_like(clean_k, requires_grad=True)
    branch_k = torch.randn(1, heads, branches * canvas, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    branch_v = torch.randn_like(branch_k, requires_grad=True)
    tensors = (query, clean_k, clean_v, branch_k, branch_v)
    flex = branch_attention(*tensors, layout)
    dense = branch_attention(
        *tensors,
        layout,
        backend="dense_reference",
        allow_dense_reference=True,
    )
    torch.testing.assert_close(flex, dense, rtol=4e-2, atol=4e-2)
    upstream = torch.randn_like(flex)
    flex_grads = torch.autograd.grad((flex * upstream).sum(), tensors, retain_graph=True)
    dense_grads = torch.autograd.grad((dense * upstream).sum(), tensors)
    for actual, expected in zip(flex_grads, dense_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=7e-2, atol=7e-2)


def test_triton_categorical_statistics_match_torch() -> None:
    torch.manual_seed(107)
    logits = torch.randn(513, 261, device="cuda", dtype=torch.bfloat16)
    expected = categorical_entropy_argmax_confidence(logits, backend="torch")
    actual = categorical_entropy_argmax_confidence(logits, backend="triton")
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=2e-4, atol=2e-4)
    uniforms = torch.rand(513, device="cuda")
    sampled_expected = categorical_sample_entropy_argmax_confidence(
        logits, uniforms, backend="torch"
    )
    sampled_actual = categorical_sample_entropy_argmax_confidence(
        logits, uniforms, backend="triton"
    )
    for actual_value, expected_value in zip(sampled_actual, sampled_expected, strict=True):
        tolerance = 0 if actual_value.dtype == torch.long else 2e-4
        torch.testing.assert_close(
            actual_value, expected_value, rtol=tolerance, atol=tolerance
        )


def test_triton_reveal_matches_torch() -> None:
    device = torch.device("cuda")
    canvas = torch.full((2, 512), 261, device=device, dtype=torch.long)
    samples = torch.arange(1024, device=device).view(2, 512) % 261
    entropy = torch.arange(512, device=device, dtype=torch.float32)[None].expand(2, -1)
    unresolved = torch.ones_like(canvas, dtype=torch.bool)
    active = torch.ones_like(canvas, dtype=torch.bool)
    quota = torch.tensor([37, 111], device=device)
    expected = reveal_low_entropy(
        canvas, samples, entropy, unresolved, active, quota, eot_id=256, backend="torch"
    )
    actual = reveal_low_entropy(
        canvas, samples, entropy, unresolved, active, quota, eot_id=256, backend="triton"
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def test_production_model_varlen_and_branch_backward_is_finite() -> None:
    torch.manual_seed(109)
    device = torch.device("cuda")
    model = ByteDiffusionModel(ByteDiffusionConfig()).to(device)
    clean = torch.randint(0, 256, (1, 32), device=device)
    clean[:, -1] = 256
    valid = torch.ones_like(clean, dtype=torch.bool)
    positions = torch.arange(32, device=device)[None]
    noisy = clean[:, None, 8:24].clone()
    noisy[:, :, ::2] = 261
    branch_valid = torch.ones_like(noisy, dtype=torch.bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.forward_canvas_branches(
            clean,
            valid,
            noisy,
            branch_valid,
            torch.tensor([[8]], device=device),
            positions=positions,
        )
        loss = output.clean_logits.float().square().mean() + output.branch_logits.float().square().mean()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_compiled_trainer_runs_bf16_validation_and_update() -> None:
    manifest = AtomicIdManifest.reference()
    document = AtomicDocument(
        "cuda-smoke", tuple([*range(31), manifest.eot_id])
    )
    chunks = pack_documents((document,), manifest, chunk_size=32)
    run = TrainingRunConfig(
        iterations=1,
        val_loss_every=1,
        train_log_every=1,
        warmdown_iters=0,
        corruption=CorruptionConfig(
            kind="absorbing_rb", canvas_length=16, branches_per_row=1
        ),
        microbatch_per_rank=1,
        gradient_accumulation=1,
        compile_model=True,
    )
    trainer = ByteDiffusionTrainer(
        ByteDiffusionModel(ByteDiffusionConfig()),
        DeterministicChunkCursor(chunks, seed=run.seed),
        chunks,
        run,
        device=torch.device("cuda", 0),
        atomic_manifest=manifest,
    )
    validation = trainer.validate()
    update = trainer.run_update()
    assert validation.literal_bytes == 31
    assert validation.special_targets == 1
    assert update.ar_targets == 32
    assert update.diffusion_targets > 0


def test_compiled_blt_ht_objective_runs_on_cuda() -> None:
    manifest = AtomicIdManifest.reference()
    document = AtomicDocument(
        "cuda-blt-smoke", tuple([*range(63), manifest.eot_id])
    )
    chunks = pack_documents((document,), manifest, chunk_size=64)
    run = TrainingRunConfig(
        iterations=1,
        val_loss_every=1,
        train_log_every=1,
        warmdown_iters=0,
        recipe="blt_d",
        corruption=CorruptionConfig(
            kind="blt_bernoulli", canvas_length=16, branches_per_row=4
        ),
        objective_reduction="paper_sum",
        microbatch_per_rank=1,
        gradient_accumulation=1,
        compile_model=True,
    )
    trainer = ByteDiffusionTrainer(
        ByteDiffusionModel(ByteDiffusionConfig()),
        DeterministicChunkCursor(chunks, seed=run.seed),
        chunks,
        run,
        device=torch.device("cuda", 0),
        atomic_manifest=manifest,
    )
    update = trainer.run_update()
    assert update.ar_targets == 64
    assert update.diffusion_targets >= 0
    assert math.isfinite(update.total)

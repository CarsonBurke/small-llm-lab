"""Queued CUDA parity checks; never run these outside mlq."""

from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn.functional as F

from pretraining.byte_diffusion.attention import (
    CanvasBranchLayout,
    PackedCleanQKV,
    branch_attention,
    build_canvas_block_mask,
    canvas_block_mask_metadata,
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
from pretraining.byte_diffusion.layers import PackedSelfAttention, pack_valid
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.objectives import masked_cross_entropy_per_target
from pretraining.byte_diffusion.training import (
    IGNORE_INDEX,
    ByteDiffusionTrainer,
    JointForward,
    TrainingRunConfig,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("BYTE_DIFFUSION_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="queued CUDA validation requires BYTE_DIFFUSION_CUDA_TESTS=1",
)


def test_compiled_masked_per_target_ce_keeps_static_shape_on_cuda() -> None:
    torch.manual_seed(97)
    logits = torch.randn(3, 17, 261, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(261, (3, 17), device="cuda")
    active = torch.rand(3, 17, device="cuda") > 0.35
    actual = masked_cross_entropy_per_target(logits, targets, active)
    expected = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    expected = torch.where(active, expected, 0.0)
    assert actual.shape == targets.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


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


def test_prepacked_document_branch_attention_compiles_fullgraph() -> None:
    torch.manual_seed(103)
    device = torch.device("cuda")
    layer = PackedSelfAttention(dim=128, heads=2, rope_theta=10_000.0).to(
        device=device, dtype=torch.bfloat16
    )
    states = torch.randn(
        1, 28, 128, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    positions = torch.tensor(
        [[*range(8), *range(6), 0, 0, 4, 5, 6, 7, 4, 5, 6, 7, 0, 1, 2, 3]],
        device=device,
    )
    clean_indices = torch.arange(14, device=device)
    clean_cu = torch.tensor([0, 8, 14], device=device, dtype=torch.int32)
    query_indices = torch.tensor(
        [16, 17, 18, 19, 20, 21], device=device
    )
    kv_indices = torch.tensor(
        [0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21],
        device=device,
    )
    query_cu = torch.tensor([0, 4, 6], device=device, dtype=torch.int32)
    kv_cu = torch.tensor([0, 8, 14], device=device, dtype=torch.int32)

    def forward(x: torch.Tensor) -> torch.Tensor:
        return layer.forward_packed_document_branches(
            x,
            clean_length=16,
            clean_indices=clean_indices,
            clean_cu_seqlens=clean_cu,
            positions=positions,
            branch_query_indices=query_indices,
            branch_kv_indices=kv_indices,
            branch_query_cu_seqlens=query_cu,
            branch_kv_cu_seqlens=kv_cu,
            max_branch_query_length=4,
            max_branch_kv_length=8,
            clean_window=4,
            allow_dense_reference=False,
        )

    eager = forward(states)
    compiled = torch.compile(forward, fullgraph=True, dynamic=True)(states)
    torch.testing.assert_close(compiled, eager, rtol=3e-2, atol=3e-2)
    compiled.square().mean().backward()
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()


def test_shared_document_flex_representative_dynamic_fullgraph_fwd_bwd() -> None:
    torch.manual_seed(107)
    device = torch.device("cuda")
    batch, clean_length, branches, canvas = 32, 64, 4, 16
    valid_cpu = torch.ones(batch, clean_length, dtype=torch.bool)
    branch_valid_cpu = torch.ones(batch, branches, canvas, dtype=torch.bool)
    starts_cpu = torch.tensor([[16, 48, 16, 48]]).expand(batch, -1).clone()
    documents_cpu = torch.tensor([[0] * 32 + [1] * 32]).expand(batch, -1)
    branch_documents_cpu = torch.tensor([[0, 1, 0, 1]]).expand(batch, -1)
    positions_cpu = torch.tensor([[*range(32), *range(32)]]).expand(batch, -1)
    branch_positions_cpu = torch.gather(
        positions_cpu[:, None].expand(-1, branches, -1),
        2,
        starts_cpu[:, :, None] + torch.arange(canvas)[None, None],
    )
    cpu_layout = CanvasBranchLayout(
        valid_cpu,
        branch_valid_cpu,
        starts_cpu,
        512,
        positions_cpu,
        branch_positions_cpu,
        documents_cpu,
        branch_documents_cpu,
    )
    metadata = canvas_block_mask_metadata(cpu_layout).to(device)
    layout = CanvasBranchLayout(
        valid_cpu.to(device),
        branch_valid_cpu.to(device),
        starts_cpu.to(device),
        512,
        positions_cpu.to(device),
        branch_positions_cpu.to(device),
        documents_cpu.to(device),
        branch_documents_cpu.to(device),
    )
    block_mask = build_canvas_block_mask(layout, metadata=metadata)
    physical_positions = torch.cat(
        (layout.clean_positions, layout.branch_positions.flatten(1, 2)), 1
    )
    clean_indices = torch.arange(batch * clean_length, device=device)
    clean_cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * clean_length
    layer = PackedSelfAttention(128, 2, 10_000.0).to(
        device=device, dtype=torch.bfloat16
    )
    states = torch.randn(
        batch,
        clean_length + branches * canvas,
        128,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    targets = torch.randint(
        0, 128, (batch * branches * canvas,), device=device
    )

    def loss(x: torch.Tensor) -> torch.Tensor:
        output = layer.forward_shared_document_branches(
            x,
            clean_length=clean_length,
            clean_indices=clean_indices,
            clean_cu_seqlens=clean_cu,
            positions=physical_positions,
            layout=layout,
            clean_window=512,
            allow_dense_reference=False,
            block_mask=block_mask,
        )
        return F.cross_entropy(output[:, clean_length:].reshape(-1, 128), targets)

    compiled_loss = torch.compile(loss, fullgraph=True, dynamic=True)(states)
    compiled_loss.backward()
    assert torch.isfinite(compiled_loss)
    assert states.grad is not None and torch.isfinite(states.grad).all()


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


def test_legacy_canvas_trainer_runs_bf16_validation_and_update() -> None:
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
        # Canvas remains an eager research oracle. The production compiled
        # path is Fast-BLT and is covered by the dedicated test below.
        compile_model=False,
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


def test_compiled_packed_causal_loss_and_gradients_match_ragged_eager() -> None:
    torch.manual_seed(111)
    device = torch.device("cuda")
    config = ByteDiffusionConfig.tiny(
        ngram_enabled=True,
        ngram_table_size=128,
        ngram_rank=4,
        ngram_orders=(3, 4),
    )
    eager_model = ByteDiffusionModel(config).to(device)
    compiled_model = ByteDiffusionModel(config).to(device)
    compiled_model.load_state_dict(eager_model.state_dict())
    eager = JointForward(eager_model)
    compiled = torch.compile(
        JointForward(compiled_model), dynamic=True, fullgraph=True
    )
    ids = torch.randint(0, 256, (2, 32), device=device)
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[1, 20:] = False
    ids[0, -1] = config.vocab.eot_id
    ids[1, 19] = config.vocab.eot_id
    ids[1, 20:] = config.vocab.pad_id
    positions = torch.arange(32, device=device)[None].expand_as(ids)
    targets = torch.full_like(ids, IGNORE_INDEX)
    targets[0, :-1] = ids[0, 1:]
    targets[1, :19] = ids[1, 1:20]
    targets[0, :4] = torch.tensor([257, 258, 259, 260], device=device)
    bos_targets = torch.tensor([65, IGNORE_INDEX], device=device)

    def loss(module: torch.nn.Module) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, bos_logits = module(
                ids,
                valid,
                positions,
                None,
                -1,
                None,
                None,
                0,
                False,
                True,
            )
            packed_targets = pack_valid(targets, valid)
            total = F.cross_entropy(
                logits,
                packed_targets,
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            ) + F.cross_entropy(
                bos_logits,
                bos_targets,
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            count = packed_targets.ne(IGNORE_INDEX).sum() + bos_targets.ne(
                IGNORE_INDEX
            ).sum()
            return total / count

    eager_loss = loss(eager)
    eager_loss.backward()
    eager_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in eager_model.named_parameters()
        if parameter.grad is not None
    }
    compiled_loss = loss(compiled)
    compiled_loss.backward()

    torch.testing.assert_close(compiled_loss, eager_loss, rtol=3e-3, atol=3e-3)
    for name, parameter in compiled_model.named_parameters():
        if name not in eager_gradients:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            torch.testing.assert_close(
                parameter.grad,
                eager_gradients[name],
                rtol=5e-2,
                atol=5e-3,
            )

    # Runtime lengths change without changing the physical tensor shape. Once
    # the dynamic graph is warm, this must not create a new graph or recompile.
    from torch._dynamo.utils import counters

    counters.clear()
    compiled_model.zero_grad(set_to_none=True)
    valid[1, 24:] = False
    ids[1, 20:24] = torch.randint(0, 256, (4,), device=device)
    ids[1, 23] = config.vocab.eot_id
    ids[1, 24:] = config.vocab.pad_id
    targets[1, :23] = ids[1, 1:24]
    targets[1, 23:] = IGNORE_INDEX
    loss(compiled).backward()
    assert int(counters["stats"]["unique_graphs"]) == 0
    assert sum(counters["recompiles"].values()) == 0


def test_ragged_bf16_joint_clean_ce_matches_standalone_flash() -> None:
    torch.manual_seed(113)
    device = torch.device("cuda")
    model = ByteDiffusionModel(ByteDiffusionConfig()).to(device).eval()
    clean = torch.randint(0, 256, (2, 32), device=device)
    valid = torch.ones_like(clean, dtype=torch.bool)
    valid[1, 20:] = False
    clean[0, -1] = model.config.vocab.eot_id
    clean[1, 19] = model.config.vocab.eot_id
    clean[1, 20:] = model.config.vocab.pad_id
    starts = torch.tensor([[8], [4]], device=device)
    offsets = torch.arange(16, device=device)
    indices = starts[:, :, None] + offsets[None, None, :]
    noisy = clean[:, None, :].expand(-1, 1, -1).gather(2, indices).clone()
    noisy[:, :, ::2] = model.config.vocab.mask_id
    branch_valid = torch.ones_like(noisy, dtype=torch.bool)
    positions = torch.arange(32, device=device)[None].expand(2, -1)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        standalone = model.forward_ar_varlen(
            clean, valid, positions=positions
        ).logits
        canvas = model.forward_canvas_branches(
            clean,
            valid,
            noisy,
            branch_valid,
            starts,
            positions=positions,
        ).clean_logits
        blt = model.forward_blt_d_branches(
            clean,
            valid,
            noisy,
            branch_valid,
            starts,
            positions=positions,
            branch_condition_indices=(
                starts // model.config.patch_stride - 1
            ),
        ).clean_logits

    # The invariant concerns clean logits, so any shared predictable labels are
    # sufficient. Same-position clean ids avoid introducing PAD as the target
    # after the short row's terminal EOT.
    targets = clean
    reference_ce = F.cross_entropy(standalone[valid].float(), targets[valid])
    for joint_logits in (canvas, blt):
        joint_ce = F.cross_entropy(joint_logits[valid].float(), targets[valid])
        torch.testing.assert_close(joint_ce, reference_ce, rtol=2e-3, atol=2e-3)


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

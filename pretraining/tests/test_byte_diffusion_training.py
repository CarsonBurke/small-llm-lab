"""CPU contracts for scratch training, metrics, and exact resume."""

from __future__ import annotations

import copy
from dataclasses import replace
import math
from pathlib import Path

import pytest
import torch

import pretraining.byte_diffusion.training as training_module
from pretraining.byte_diffusion.config import ByteDiffusionConfig, CorruptionConfig
from pretraining.byte_diffusion.data import (
    AtomicDocument,
    AtomicIdManifest,
    DeterministicChunkCursor,
    pack_documents,
)
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.training import (
    CANONICAL_PRESET,
    BltCorruptionPlan,
    ByteDiffusionTrainer,
    CanvasCorruptionPlan,
    DistributedContext,
    DeterministicSubsetChunkDataset,
    TrainingRunConfig,
    attention_context,
    chunks_to_batch,
    create_optimizer,
    format_train_metric,
    format_validation_metric,
    gradient_interference_metrics,
    learning_rate_multiplier,
    prepare_blt_corruption,
    prepare_canvas_corruption,
    sample_nonoverlapping_patch_starts,
    sample_validation_starts,
    take_distributed_chunks,
)
from scripts.ablation import parse_log_line


def _chunks(*, offset: int = 0):
    manifest = AtomicIdManifest.reference()
    documents = [
        AtomicDocument(
            key=f"doc-{offset + index}",
            atomic_ids=(
                65 + index,
                66,
                67,
                68,
                69,
                70,
                71,
                manifest.eot_id,
            ),
        )
        for index in range(4)
    ]
    return pack_documents(documents, manifest, chunk_size=8)


def _run_config(*, recipe="canvas", iterations=2, accumulation=1):
    return TrainingRunConfig(
        iterations=iterations,
        val_loss_every=1,
        train_log_every=1,
        warmdown_iters=0,
        run_id="cpu-contract",
        seed=17,
        recipe=recipe,
        corruption=CorruptionConfig(
            kind="absorbing_rb", canvas_length=8, branches_per_row=1
        ),
        microbatch_per_rank=1,
        gradient_accumulation=accumulation,
        attention_policy="dense_reference",
        allow_cpu_reference=True,
    )


def _trainer(model, config, chunks=None):
    chunks = tuple(chunks or _chunks())
    return ByteDiffusionTrainer(
        model,
        DeterministicChunkCursor(chunks, seed=config.seed),
        chunks,
        config,
        device=torch.device("cpu"),
    )


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_standard_ablation_environment_is_the_run_contract(monkeypatch) -> None:
    monkeypatch.setenv("ITERATIONS", "2000")
    monkeypatch.setenv("VAL_LOSS_EVERY", "20")
    monkeypatch.setenv("TRAIN_LOG_EVERY", "10")
    monkeypatch.setenv("WARMDOWN_ITERS", "1200")
    monkeypatch.setenv("RUN_ID", "bd_canvas128")
    monkeypatch.setenv("BYTE_DIFFUSION_RECIPE", "canvas")
    observed = TrainingRunConfig.from_env()
    assert observed.iterations == 2_000
    assert observed.val_loss_every == 20
    assert observed.train_log_every == 10
    assert observed.warmdown_iters == 1_200
    assert observed.run_id == "bd_canvas128"
    assert observed.recipe == "canvas"


def test_canonical_canvas512_preset_is_fully_pinned(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_PRESET", CANONICAL_PRESET)
    monkeypatch.setenv("BYTE_DIFFUSION_MICROBATCH", "32")
    observed = TrainingRunConfig.from_env()
    assert observed.preset == CANONICAL_PRESET
    assert observed.iterations == 2_000
    assert observed.corruption == CorruptionConfig.canvas512()
    assert observed.microbatch_per_rank == 32
    assert observed.microbatch_token_budget == 278_528
    assert observed.gradient_accumulation == 8

    monkeypatch.setenv("BYTE_DIFFUSION_CANVAS_LENGTH", "128")
    with pytest.raises(ValueError, match="contract mismatch"):
        TrainingRunConfig.from_env()


def test_canonical_preset_rejects_global_batch_drift(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_PRESET", CANONICAL_PRESET)
    monkeypatch.setenv("BYTE_DIFFUSION_MICROBATCH", "32")
    monkeypatch.setenv("BYTE_DIFFUSION_GLOBAL_BATCH", "128")
    with pytest.raises(ValueError, match="global_batch_size"):
        TrainingRunConfig.from_env()


def test_canonical_preset_rejects_effective_batch_drift(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_PRESET", CANONICAL_PRESET)
    monkeypatch.setenv("BYTE_DIFFUSION_MICROBATCH", "32")
    monkeypatch.setenv("BYTE_DIFFUSION_GRAD_ACCUM", "15")
    with pytest.raises(ValueError, match="GRAD_ACCUM"):
        TrainingRunConfig.from_env()


def test_training_lineage_is_explicitly_scratch() -> None:
    assert TrainingRunConfig().initialization_kind == "scratch"
    with pytest.raises(ValueError, match="scratch lineages only"):
        TrainingRunConfig(initialization_kind="warm_start")  # type: ignore[arg-type]


def test_compiled_branch_training_rejects_activation_checkpointing() -> None:
    with pytest.raises(ValueError, match="escapes compiled FlexAttention"):
        TrainingRunConfig(activation_checkpointing=True)


def test_blt_environment_defaults_to_32_independent_16_byte_blocks(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_RECIPE", "blt_d")
    monkeypatch.delenv("BYTE_DIFFUSION_CORRUPTION", raising=False)
    monkeypatch.delenv("BYTE_DIFFUSION_CANVAS_LENGTH", raising=False)
    monkeypatch.delenv("BYTE_DIFFUSION_BRANCHES", raising=False)
    observed = TrainingRunConfig.from_env()
    assert observed.corruption.kind == "blt_bernoulli"
    assert observed.corruption.canvas_length == 16
    assert observed.corruption.branches_per_row == 32
    assert observed.corruption.corrupted_positions_per_row == 512


def test_environment_derives_accumulation_for_global_batch(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_MICROBATCH", "8")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("BYTE_DIFFUSION_GLOBAL_BATCH", "256")
    monkeypatch.delenv("BYTE_DIFFUSION_GRAD_ACCUM", raising=False)
    assert TrainingRunConfig.from_env().gradient_accumulation == 4

    monkeypatch.setenv("BYTE_DIFFUSION_GRAD_ACCUM", "3")
    with pytest.raises(ValueError, match="GRAD_ACCUM"):
        TrainingRunConfig.from_env()

    monkeypatch.delenv("BYTE_DIFFUSION_GRAD_ACCUM")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("BYTE_DIFFUSION_MICROBATCH", "24")
    monkeypatch.setenv("BYTE_DIFFUSION_GLOBAL_BATCH", "256")
    assert TrainingRunConfig.from_env().gradient_accumulation == 11

    monkeypatch.setenv("BYTE_DIFFUSION_GLOBAL_BATCH", "250")
    assert TrainingRunConfig.from_env().gradient_accumulation == 11


def test_learning_rate_is_flat_then_warms_down_on_planned_updates() -> None:
    assert learning_rate_multiplier(0, 10, 4) == 1.0
    assert learning_rate_multiplier(5, 10, 4) == 1.0
    assert learning_rate_multiplier(6, 10, 4) == 1.0
    assert learning_rate_multiplier(8, 10, 4) == 0.5
    assert learning_rate_multiplier(9, 10, 4) == 0.25
    with pytest.raises(ValueError, match="planned update"):
        learning_rate_multiplier(10, 10, 4)


def test_gradient_interference_reports_norms_and_cosine_without_grad_mutation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    ar_loss = (parameter.square()).sum()
    diffusion_loss = (-parameter).sum()
    metrics = gradient_interference_metrics(
        ar_loss, diffusion_loss, (parameter,)
    )
    assert metrics.ar_norm == pytest.approx(math.sqrt(20))
    assert metrics.diffusion_norm == pytest.approx(math.sqrt(2))
    assert metrics.cosine == pytest.approx(-3 / math.sqrt(10))
    assert parameter.grad is None


def test_cpu_backend_requires_two_explicit_reference_opt_ins() -> None:
    with pytest.raises(RuntimeError, match="correctness reference"):
        attention_context(
            "dense_reference", torch.device("cpu"), allow_cpu_reference=False
        )
    with pytest.raises(RuntimeError, match="correctness reference"):
        attention_context(
            "flash_sdpa", torch.device("cpu"), allow_cpu_reference=True
        )
    with attention_context(
        "dense_reference", torch.device("cpu"), allow_cpu_reference=True
    ):
        pass


def test_cuda_optimizer_is_declared_fused_and_cpu_is_reference(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    config = _run_config()
    cpu = create_optimizer([parameter], config, torch.device("cpu"))
    assert cpu.defaults["fused"] is False

    captured = {}

    class FakeAdamW:
        def __init__(self, parameters, **kwargs):
            del parameters
            captured.update(kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", FakeAdamW)
    create_optimizer([parameter], config, torch.device("cuda"))
    assert captured["fused"] is True


def test_chunk_batch_uses_stored_halo_targets_and_document_positions() -> None:
    batch = chunks_to_batch(_chunks()[:1])
    assert batch.ids.shape == (1, 8)
    assert batch.ar_targets[0, -1] == -100
    torch.testing.assert_close(batch.positions[0], torch.arange(8))
    assert batch.ar_targets.ne(-100).sum() == 7
    assert batch.bos_targets.tolist() == [ord("A")]
    # The virtual BOS objective closes the codelength gap without shifting the
    # physical four-byte patch phase of the clean document.
    assert int(batch.ar_targets.ne(-100).sum() + batch.bos_targets.ne(-100).sum()) == 8


def test_distributed_cursor_sharding_is_disjoint_and_cursor_identical() -> None:
    chunks = _chunks()
    left = DeterministicChunkCursor(chunks, seed=3, shuffle=False)
    right = DeterministicChunkCursor(chunks, seed=3, shuffle=False)
    rank0 = take_distributed_chunks(
        left, 2, DistributedContext(rank=0, local_rank=0, world_size=2)
    )
    rank1 = take_distributed_chunks(
        right, 2, DistributedContext(rank=1, local_rank=1, world_size=2)
    )
    assert [chunk.chunk_index for chunk in rank0] == [0, 2]
    assert [chunk.chunk_index for chunk in rank1] == [1, 3]
    assert left.state_dict() == right.state_dict()


def test_validation_subset_evenly_covers_and_binds_the_full_split() -> None:
    source = tuple(_chunks(offset=10 * group)[0] for group in range(10))
    subset = DeterministicSubsetChunkDataset(source, 3)
    assert subset.indices == (1, 5, 8)
    assert all(
        subset[index] is source[position]
        for index, position in enumerate(subset.indices)
    )
    repeated = DeterministicSubsetChunkDataset(source, 3)
    assert repeated.dataset_sha256 == subset.dataset_sha256
    changed = DeterministicSubsetChunkDataset(source, 4)
    assert changed.dataset_sha256 != subset.dataset_sha256


def test_mixed_scratch_update_has_both_losses_and_changes_parameters() -> None:
    torch.manual_seed(7)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    trainer = _trainer(model, _run_config(recipe="canvas"))
    before = model.output.weight.detach().clone()
    metrics = trainer.run_update()
    assert math.isfinite(metrics.total)
    assert metrics.ar > 0
    assert metrics.diffusion > 0
    assert metrics.ar_targets == 8
    assert 1 <= metrics.diffusion_targets <= 8
    assert not torch.equal(before, model.output.weight)


def test_global_batch_uses_a_smaller_tail_microbatch_without_extra_rows() -> None:
    torch.manual_seed(9)
    config = replace(
        _run_config(recipe="canvas", iterations=1),
        microbatch_per_rank=3,
        gradient_accumulation=2,
        global_batch_size=4,
    )
    trainer = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), config)

    metrics = trainer.run_update()

    assert metrics.microsteps == 2
    assert metrics.max_microbatch == 3
    assert metrics.ar_targets == 32
    assert trainer.train_cursor.epoch == 0
    assert trainer.train_cursor.position == 4


def test_tail_microbatch_matches_fixed_partition_gradient_for_causal_loss() -> None:
    torch.manual_seed(101)
    fixed_model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    tail_model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    tail_model.load_state_dict(fixed_model.state_dict())
    base = _run_config(recipe="causal_only", iterations=1)
    fixed = replace(
        base,
        microbatch_per_rank=4,
        gradient_accumulation=1,
        global_batch_size=4,
    )
    tailed = replace(
        base,
        microbatch_per_rank=3,
        gradient_accumulation=2,
        global_batch_size=4,
    )

    fixed_metrics = _trainer(fixed_model, fixed).run_update()
    tail_metrics = _trainer(tail_model, tailed).run_update()

    assert tail_metrics.ar == pytest.approx(fixed_metrics.ar, abs=1e-6)
    for fixed_parameter, tail_parameter in zip(
        fixed_model.parameters(), tail_model.parameters(), strict=True
    ):
        torch.testing.assert_close(fixed_parameter, tail_parameter, atol=1e-6, rtol=1e-6)


def test_production_24_plus_16_canvas_partition_matches_fixed_partition(
    monkeypatch,
) -> None:
    manifest = AtomicIdManifest.reference()
    chunks = tuple(
        pack_documents(
            (
                AtomicDocument(
                    f"row-{row}",
                    (row % 256, 17, 33, 49, 65, 81, 97, manifest.eot_id),
                ),
            ),
            manifest,
            chunk_size=8,
        )[0]
        for row in range(256)
    )

    def deterministic_canvas(batch, config, vocab, generator):
        del generator
        clean = batch.ids[:, None, : config.canvas_length]
        branch_valid = batch.valid[:, None, : config.canvas_length]
        limits = batch.ids[:, :1].remainder(config.canvas_length).add(1)
        active = branch_valid & (
            torch.arange(config.canvas_length)[None, None, :] < limits[:, :, None]
        )
        return CanvasCorruptionPlan(
            clean_branches=clean,
            noisy_branches=torch.where(active, vocab.mask_id, clean),
            branch_valid=branch_valid,
            branch_starts=torch.zeros((batch.ids.shape[0], 1), dtype=torch.long),
            active=active,
        )

    monkeypatch.setattr(
        training_module, "prepare_canvas_corruption", deterministic_canvas
    )
    base = replace(
        _run_config(recipe="canvas", iterations=1),
        max_grad_norm=None,
    )
    tail_config = replace(
        base,
        microbatch_per_rank=24,
        gradient_accumulation=11,
        global_batch_size=256,
    )
    fixed_config = replace(
        base,
        microbatch_per_rank=16,
        gradient_accumulation=16,
        global_batch_size=256,
    )
    torch.manual_seed(103)
    initial = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    tail_model = copy.deepcopy(initial)
    fixed_model = copy.deepcopy(initial)
    batch_sizes = []
    original_collate = training_module.chunks_to_batch

    def recording_collate(selected):
        batch_sizes.append(len(selected))
        return original_collate(selected)

    monkeypatch.setattr(training_module, "chunks_to_batch", recording_collate)
    tail = _trainer(tail_model, tail_config, chunks)
    tail_metrics = tail.run_update()
    monkeypatch.setattr(training_module, "chunks_to_batch", original_collate)
    fixed_metrics = _trainer(fixed_model, fixed_config, chunks).run_update()

    assert batch_sizes == [24] * 10 + [16]
    assert tail.train_cursor.epoch == 0
    assert tail.train_cursor.position == 256
    assert tail_metrics.ar_targets == fixed_metrics.ar_targets == 2_048
    assert tail_metrics.diffusion_targets == fixed_metrics.diffusion_targets
    assert tail_metrics.total == pytest.approx(fixed_metrics.total, abs=2e-6)
    assert tail_metrics.ar == pytest.approx(fixed_metrics.ar, abs=2e-6)
    assert tail_metrics.diffusion == pytest.approx(
        fixed_metrics.diffusion, abs=2e-6
    )
    for tail_parameter, fixed_parameter in zip(
        tail_model.parameters(), fixed_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            tail_parameter, fixed_parameter, atol=2e-6, rtol=2e-6
        )


def test_canvas_plan_slices_nonoverlapping_patch_aligned_branches() -> None:
    batch = chunks_to_batch(_chunks()[:1])
    config = CorruptionConfig(
        kind="absorbing_rb", canvas_length=4, branches_per_row=2
    )
    plan = prepare_canvas_corruption(
        batch,
        config,
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(31),
    )
    assert plan.clean_branches.shape == (1, 2, 4)
    assert plan.noisy_branches.shape == (1, 2, 4)
    assert plan.branch_valid.shape == (1, 2, 4)
    assert bool((plan.branch_starts % 4 == 0).all())
    assert abs(int(plan.branch_starts[0, 1] - plan.branch_starts[0, 0])) >= 4
    assert not bool((plan.active & ~plan.branch_valid).any())
    assert bool(
        (
            plan.noisy_branches[plan.active]
            == ByteDiffusionConfig.tiny().vocab.mask_id
        ).all()
    )


def test_canvas_update_uses_one_shared_prefix_forward_for_multiple_branches() -> None:
    class CountingModel(ByteDiffusionModel):
        def __init__(self):
            super().__init__(ByteDiffusionConfig.tiny())
            self.ordinary_forwards = 0
            self.canvas_forwards = 0

        def forward(self, *args, **kwargs):
            self.ordinary_forwards += 1
            return super().forward(*args, **kwargs)

        def forward_canvas_branches(self, *args, **kwargs):
            self.canvas_forwards += 1
            return super().forward_canvas_branches(*args, **kwargs)

    config = _run_config()
    config = TrainingRunConfig(
        **{
            **config.contract_dict(),
            "corruption": CorruptionConfig(
                kind="absorbing_rb", canvas_length=4, branches_per_row=2
            ),
        }
    )
    model = CountingModel()
    trainer = _trainer(model, config)
    trainer.run_update()
    assert model.canvas_forwards == 1
    assert model.ordinary_forwards == 0


def test_blt_plan_samples_reference_patch_origins_with_ht_weight() -> None:
    batch = chunks_to_batch(_chunks()[:1])
    config = CorruptionConfig(
        kind="absorbing_rb", canvas_length=4, branches_per_row=2
    )
    plan = prepare_blt_corruption(
        batch,
        config,
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(37),
    )
    assert plan.block_starts.shape == (1, 2)
    assert bool((plan.block_starts % 4 == 0).all())
    assert plan.block_starts.tolist() == [[4, 4]]
    torch.testing.assert_close(plan.sampling_weight, torch.tensor([0.5]))
    assert not bool((plan.active & ~plan.branch_valid).any())
    torch.testing.assert_close(
        plan.noisy_blocks[~plan.active], plan.clean_blocks[~plan.active]
    )


def test_canvas_start_never_crosses_pad_when_a_full_span_exists() -> None:
    valid = torch.zeros(1, 8192, dtype=torch.bool)
    valid[:, :513] = True
    for seed in range(32):
        starts = sample_nonoverlapping_patch_starts(
            valid,
            span_length=512,
            count=1,
            patch_stride=4,
            generator=torch.Generator().manual_seed(seed),
        )
        assert starts.tolist() == [[0]]


def test_validation_starts_are_invariant_to_batch_and_rank_order() -> None:
    chunks = tuple(_chunks())
    batch = chunks_to_batch(chunks)
    together = sample_validation_starts(
        batch.valid,
        chunks,
        span_length=4,
        count=1,
        patch_stride=4,
        seed=17,
    )
    reordered_chunks = (chunks[3], chunks[1])
    reordered = chunks_to_batch(reordered_chunks)
    subset = sample_validation_starts(
        reordered.valid,
        reordered_chunks,
        span_length=4,
        count=1,
        patch_stride=4,
        seed=17,
    )
    torch.testing.assert_close(subset, together[[3, 1]])


def test_short_rows_retain_partial_diffusion_span_and_ar_targets() -> None:
    manifest = AtomicIdManifest.reference()
    chunks = pack_documents(
        (AtomicDocument("short", (65, 66, manifest.eot_id)),),
        manifest,
        chunk_size=8,
    )
    batch = chunks_to_batch(chunks)
    plan = prepare_canvas_corruption(
        batch,
        CorruptionConfig(kind="absorbing_rb", canvas_length=8, branches_per_row=1),
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(5),
    )
    scored = batch.ar_targets.ne(-100).sum() + batch.bos_targets.ne(-100).sum()
    assert int(scored) == 3
    assert plan.branch_starts.tolist() == [[0]]
    assert plan.branch_valid.sum() == 3
    assert not bool((plan.active & ~plan.branch_valid).any())


def test_pad_only_extra_canvas_branch_contributes_zero_targets() -> None:
    manifest = AtomicIdManifest.reference()
    chunks = pack_documents(
        (AtomicDocument("short-multi", (65, 66, manifest.eot_id)),),
        manifest,
        chunk_size=16,
    )
    plan = prepare_canvas_corruption(
        chunks_to_batch(chunks),
        CorruptionConfig(kind="absorbing_rb", canvas_length=8, branches_per_row=2),
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(5),
    )
    assert plan.branch_valid.sum((0, 2)).tolist() == [3, 0]
    assert plan.active.sum((0, 2))[1] == 0


def test_subpatch_blt_row_keeps_ar_and_has_zero_diffusion_targets() -> None:
    manifest = AtomicIdManifest.reference()
    chunks = pack_documents(
        (AtomicDocument("tiny-blt", (65, 66, manifest.eot_id)),),
        manifest,
        chunk_size=8,
    )
    base = _run_config(recipe="canvas", iterations=1)
    config = replace(
        base,
        recipe="blt_d",
        corruption=CorruptionConfig(
            kind="blt_bernoulli", canvas_length=4, branches_per_row=1
        ),
        objective_reduction="paper_sum",
    )
    trainer = _trainer(
        ByteDiffusionModel(ByteDiffusionConfig.tiny()), config, chunks
    )
    metrics = trainer.run_update()
    assert metrics.ar_targets == 3
    assert metrics.diffusion_targets == 0
    assert math.isfinite(metrics.total)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"recipe": "canavs"}, "unknown training recipe"),
        ({"objective_reduction": "sumish"}, "unknown objective reduction"),
        ({"compile_mode": "fastest"}, "unknown compile mode"),
        (
            {
                "corruption": CorruptionConfig(
                    kind="uniform_replacement", canvas_length=8, branches_per_row=1
                )
            },
            "self-conditioning",
        ),
    ],
)
def test_run_config_fails_closed_on_unknown_or_incomplete_modes(updates, match) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_run_config(), **updates)


def test_blt_update_passes_block_geometry_to_the_model() -> None:
    class CountingModel(ByteDiffusionModel):
        def __init__(self):
            super().__init__(ByteDiffusionConfig.tiny())
            self.observed_starts = None
            self.observed_length = None
            self.ordinary_forwards = 0

        def forward(self, *args, **kwargs):
            self.ordinary_forwards += 1
            return super().forward(*args, **kwargs)

        def forward_blt_d_branches(
            self,
            *args,
            positions=None,
            **kwargs,
        ):
            noisy_blocks = args[2]
            block_starts = args[4]
            self.observed_starts = block_starts.detach().clone()
            self.observed_length = noisy_blocks.shape[-1]
            return super().forward_blt_d_branches(
                *args,
                positions=positions,
                **kwargs,
            )

    base = _run_config(recipe="canvas")
    config = TrainingRunConfig(
        **{
            **base.contract_dict(),
            "corruption": CorruptionConfig(
                kind="absorbing_rb", canvas_length=4, branches_per_row=2
            ),
            "recipe": "blt_d",
        }
    )
    model = CountingModel()
    trainer = _trainer(model, config)
    trainer.run_update()
    assert model.observed_starts is not None
    assert model.observed_starts.shape == (1, 2)
    assert model.observed_length == 4
    assert model.ordinary_forwards == 0


def test_blt_loss_sums_all_blocks_per_clean_row_before_batch_mean(
    monkeypatch,
) -> None:
    batch = chunks_to_batch(_chunks()[:1])
    clean_blocks = torch.zeros(1, 2, 4, dtype=torch.long)
    active = torch.ones_like(clean_blocks, dtype=torch.bool)
    plan = BltCorruptionPlan(
        clean_blocks=clean_blocks,
        noisy_blocks=clean_blocks.clone(),
        branch_valid=active,
        active=active,
        block_starts=torch.tensor([[0, 4]]),
        block_length=4,
        t=torch.tensor([0.5]),
        sampling_weight=torch.tensor([2.0]),
    )
    monkeypatch.setattr(
        training_module,
        "prepare_blt_corruption",
        lambda *args, **kwargs: plan,
    )

    desired_nll = torch.tensor([[[1.0] * 4, [3.0] * 4]])
    probability = torch.exp(-desired_nll)
    other_probability = (1.0 - probability) / 260
    diffusion_logits = torch.cat(
        (
            probability.log().unsqueeze(-1),
            other_probability.log().unsqueeze(-1).expand(-1, -1, -1, 260),
        ),
        dim=-1,
    )

    class FixedForward(torch.nn.Module):
        def forward(self, *args, **kwargs):
            del args, kwargs
            ar_logits = torch.zeros(1, batch.ids.shape[1], 261)
            bos_logits = torch.zeros(1, 261)
            return ar_logits, diffusion_logits, bos_logits

    base = _run_config(recipe="canvas")
    config = TrainingRunConfig(
        **{
            **base.contract_dict(),
            "corruption": CorruptionConfig(
                kind="blt_bernoulli", canvas_length=4, branches_per_row=2
            ),
            "recipe": "blt_d",
            "objective_reduction": "paper_sum",
        }
    )
    trainer = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), config)
    trainer.forward_model = FixedForward()
    loss = trainer._compute_loss(batch)

    # The sampled masked sum is 32, and the HT weight of two estimates the
    # complete fixed-patch bank at 64. Clean paper_sum adds all eight atomic
    # targets (virtual BOS + seven shifted positions) rather than mean AR CE.
    torch.testing.assert_close(loss.diffusion, torch.tensor(64.0))
    expected = 64.0 + 8.0 * math.log(261)
    torch.testing.assert_close(loss.total, torch.tensor(expected), rtol=1e-6, atol=1e-6)


def test_causal_only_is_a_named_control_and_does_not_consume_corruption_rng() -> None:
    class CountingModel(ByteDiffusionModel):
        def __init__(self):
            super().__init__(ByteDiffusionConfig.tiny())
            self.ordinary_forwards = 0
            self.varlen_forwards = 0

        def forward(self, *args, **kwargs):
            self.ordinary_forwards += 1
            return super().forward(*args, **kwargs)

        def forward_ar_varlen(self, *args, **kwargs):
            self.varlen_forwards += 1
            return super().forward_ar_varlen(*args, **kwargs)

    torch.manual_seed(11)
    model = CountingModel()
    trainer = _trainer(
        model,
        _run_config(recipe="causal_only"),
    )
    generator_before = trainer.corruption_generator.get_state().clone()
    metrics = trainer.run_update()
    assert metrics.ar > 0
    assert metrics.diffusion == 0
    assert metrics.diffusion_targets == 0
    assert model.varlen_forwards == 1
    assert model.ordinary_forwards == 0
    torch.testing.assert_close(
        trainer.corruption_generator.get_state(), generator_before, rtol=0, atol=0
    )


def test_update_boundary_checkpoint_resume_is_bit_exact(tmp_path: Path) -> None:
    torch.manual_seed(19)
    initial = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    uninterrupted = _trainer(copy.deepcopy(initial), _run_config())
    interrupted = _trainer(copy.deepcopy(initial), _run_config())

    uninterrupted.run_update()
    uninterrupted.run_update()
    interrupted.run_update()
    checkpoint = tmp_path / "resume.pt"
    interrupted.save_checkpoint(checkpoint)

    # Constructor initialization and unrelated RNG consumption must not alter
    # continuation after the checkpoint restores every RNG stream.
    torch.manual_seed(999)
    resumed = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), _run_config())
    resumed.load_checkpoint(checkpoint)
    resumed.run_update()

    _assert_nested_equal(
        uninterrupted.joint.model.state_dict(), resumed.joint.model.state_dict()
    )
    _assert_nested_equal(
        uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict()
    )
    assert uninterrupted.train_cursor.state_dict() == resumed.train_cursor.state_dict()
    torch.testing.assert_close(
        uninterrupted.corruption_generator.get_state(),
        resumed.corruption_generator.get_state(),
        rtol=0,
        atol=0,
    )


def test_checkpoint_rejects_different_data_identity(tmp_path: Path) -> None:
    torch.manual_seed(23)
    source = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), _run_config())
    path = tmp_path / "checkpoint.pt"
    source.save_checkpoint(path)
    different = _trainer(
        ByteDiffusionModel(ByteDiffusionConfig.tiny()),
        _run_config(),
        chunks=_chunks(offset=100),
    )
    with pytest.raises(ValueError, match="dataset_sha256"):
        different.load_checkpoint(path)


def test_metrics_are_parseable_by_the_ablation_harness() -> None:
    torch.manual_seed(29)
    trainer = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), _run_config())
    train = trainer.run_update()
    train_entry = parse_log_line(format_train_metric(train, 2))
    assert train_entry is not None
    assert train_entry["type"] == "train"
    assert train_entry["ar_loss"] == pytest.approx(train.ar, abs=1e-6)

    validation = trainer.validate()
    val_entry = parse_log_line(
        format_validation_metric(1, 2, validation, trainer.training_time_ms)
    )
    assert val_entry is not None
    assert val_entry["type"] == "val"
    assert val_entry["val_bpb"] == pytest.approx(validation.bpb, abs=1e-6)
    assert val_entry["val_proxy_bpb"] == pytest.approx(
        validation.bpb, abs=1e-6
    )
    assert val_entry["val_diffusion_loss"] == pytest.approx(
        validation.diffusion_loss, abs=1e-6
    )

    challenge_entry = parse_log_line(
        format_validation_metric(
            1,
            2,
            validation,
            trainer.training_time_ms,
            scope="challenge",
        )
    )
    assert challenge_entry is not None
    assert challenge_entry["val_challenge_bpb"] == pytest.approx(
        validation.bpb, abs=1e-6
    )


def test_validation_bpb_counts_bos_shift_eot_and_partial_microbatch_once() -> None:
    chunks = tuple(_chunks())
    config = _run_config(recipe="causal_only")
    config = replace(config, microbatch_per_rank=3)
    trainer = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), config, chunks)

    class UniformForward(torch.nn.Module):
        def forward(self, clean_ids, *args, **kwargs):
            del args, kwargs
            batch, length = clean_ids.shape
            return (
                torch.zeros(batch, length, 261),
                None,
                torch.zeros(batch, 261),
            )

    trainer.validation_model = UniformForward()
    metrics = trainer.validate()

    assert metrics.ar_targets == 32
    assert metrics.literal_bytes == 28
    assert metrics.special_targets == 4
    assert metrics.ar_loss == pytest.approx(math.log(261), rel=1e-6)
    assert metrics.bpb == pytest.approx(
        math.log(261) / math.log(2), rel=1e-6
    )

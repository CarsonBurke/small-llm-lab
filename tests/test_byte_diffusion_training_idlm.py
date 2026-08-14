from __future__ import annotations

import torch
import pytest

from pretraining.byte_diffusion.data import AtomicIdManifest, DeterministicChunkCursor
from pretraining.byte_diffusion.idlm_model import IDLMModel, IDLMModelConfig
from pretraining.byte_diffusion.training import TrainingBatch
from pretraining.byte_diffusion.training_idlm import (
    IDLMBatch,
    IDLMTrainer,
    IDLMTrainingConfig,
    IDLMValidationMetrics,
    format_validation_metric,
    idlm_objective,
)
from scripts.ablation import parse_log_line


class _NativeRows:
    dataset_sha256 = "idlm-test-rows"

    def __init__(self) -> None:
        self.ids = torch.tensor(
            [[1, 2, 3, 256], [10, 11, 12, 256], [20, 21, 22, 256], [30, 31, 32, 256]]
        )

    def __len__(self) -> int:
        return self.ids.shape[0]

    def __getitem__(self, index: int):
        return index

    def training_batch(self, indices) -> TrainingBatch:
        selected = torch.as_tensor(indices, dtype=torch.long)
        ids = self.ids.index_select(0, selected)
        targets = torch.full_like(ids, -100)
        targets[:, :-1] = ids[:, 1:]
        rows = torch.arange(ids.shape[0])[:, None]
        return TrainingBatch(
            ids=ids,
            valid=torch.ones_like(ids, dtype=torch.bool),
            ar_targets=targets,
            bos_targets=ids[:, 0].clone(),
            positions=torch.arange(4)[None].expand_as(ids),
            full_valid=True,
            document_ids=rows.expand_as(ids),
            isolate_documents=True,
        )


def test_native_targets_keep_page_halo_and_add_true_bos_targets() -> None:
    native = TrainingBatch(
        ids=torch.tensor([[4, 5, 6, 7]]),
        valid=torch.ones((1, 4), dtype=torch.bool),
        ar_targets=torch.tensor([[5, 6, 7, 99]]),
        bos_targets=torch.tensor([4]),
        positions=torch.arange(4)[None],
        full_valid=True,
        document_ids=torch.zeros((1, 4), dtype=torch.long),
        isolate_documents=True,
    )
    batch = IDLMBatch.from_native(native, mask_id=261, pad_id=262, block_size=4)
    assert batch.targets.proposal.tolist() == [[5, 6, 7, 99]]
    assert batch.bos_targets.tolist() == [4]


def test_complete_idlm_objective_backpropagates_both_dense_paths() -> None:
    torch.manual_seed(21)
    native = TrainingBatch(
        ids=torch.tensor([[1, 2, 3, 256], [10, 11, 12, 256]]),
        valid=torch.ones((2, 4), dtype=torch.bool),
        ar_targets=torch.tensor([[2, 3, 256, -100], [11, 12, 256, -100]]),
        bos_targets=torch.tensor([1, 10]),
        positions=torch.arange(4)[None].expand(2, -1),
        full_valid=True,
        document_ids=torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]]),
        isolate_documents=True,
    )
    batch = IDLMBatch.from_native(native, mask_id=261, pad_id=262, block_size=4)
    model = IDLMModel(IDLMModelConfig.tiny())
    output = model.forward_layout(
        batch.layout, allow_dense_reference=True, bos_count=2
    )
    objective = idlm_objective(output, batch)
    objective.total.backward()

    assert objective.proposal_targets.item() == 6
    assert objective.clean_targets.item() == 8
    assert objective.clean_scale.grad_fn is None
    assert torch.isfinite(objective.clean_bpb)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_update_global_idlm_denominators_preserve_tail_objective() -> None:
    torch.manual_seed(33)
    native = TrainingBatch(
        ids=torch.tensor([[1, 2, 3, 256], [10, 11, 12, 256], [20, 21, 22, 256]]),
        valid=torch.ones((3, 4), dtype=torch.bool),
        ar_targets=torch.tensor(
            [[2, 3, 256, -100], [11, 12, 256, -100], [21, 22, 256, -100]]
        ),
        bos_targets=torch.tensor([1, 10, 20]),
        positions=torch.arange(4)[None].expand(3, -1),
        full_valid=True,
        document_ids=torch.arange(3)[:, None].expand(3, 4),
        isolate_documents=True,
    )
    batch = IDLMBatch.from_native(native, mask_id=261, pad_id=262, block_size=4)
    model = IDLMModel(IDLMModelConfig.tiny()).eval()
    output = model.forward_layout(batch.layout, allow_dense_reference=True, bos_count=3)
    full_unscaled = idlm_objective(output, batch)
    proposal_denominator = full_unscaled.proposal_targets
    clean_denominator = full_unscaled.clean_targets
    balance_scale = full_unscaled.clean_scale
    full = idlm_objective(
        output,
        batch,
        proposal_denominator=proposal_denominator,
        clean_denominator=clean_denominator,
        balance_scale=balance_scale,
    )
    split_total = full.total.new_zeros(())
    for start, stop in ((0, 2), (2, 3)):
        part_native = TrainingBatch(
            ids=native.ids[start:stop],
            valid=native.valid[start:stop],
            ar_targets=native.ar_targets[start:stop],
            bos_targets=native.bos_targets[start:stop],
            positions=native.positions[start:stop],
            full_valid=True,
            document_ids=native.document_ids[start:stop],
            isolate_documents=True,
        )
        part = IDLMBatch.from_native(
            part_native, mask_id=261, pad_id=262, block_size=4
        )
        part_output = model.forward_layout(
            part.layout, allow_dense_reference=True, bos_count=stop - start
        )
        split_total += idlm_objective(
            part_output,
            part,
            proposal_denominator=proposal_denominator,
            clean_denominator=clean_denominator,
            balance_scale=balance_scale,
        ).total
    torch.testing.assert_close(split_total, full.total)


def test_validation_metric_labels_ar_bpb_as_anchor_not_promotion_score() -> None:
    metric = IDLMValidationMetrics(
        clean_loss=1.0,
        clean_bpb=1.442695,
        proposal_loss=2.0,
        clean_scale=2.0,
        proposal_targets=12,
        clean_targets=14,
        proposal_offset_accuracy=(0.1, 0.2, 0.3, 0.4),
        proposal_offset_targets=(3, 3, 3, 3),
        elapsed_ms=5.0,
    )
    parsed = parse_log_line(format_validation_metric(20, 2_000, metric, 10.0))
    assert parsed is not None
    assert parsed["type"] == "val"
    assert parsed["val_loss"] == 1.0
    assert parsed["val_bpb"] == 1.442695
    assert parsed["val_ar_anchor_bpb"] == 1.442695
    assert parsed["generation_primary"] == 1


def test_trainer_consumes_exact_global_batch_with_a_short_tail() -> None:
    dataset = _NativeRows()
    config = IDLMTrainingConfig(
        iterations=1,
        batch_size=2,
        global_batch_size=3,
        validation_batch_size=2,
        validation_rows=2,
        warmdown_iters=0,
        compile_model=False,
    )
    cursor = DeterministicChunkCursor(dataset, seed=7, shuffle=False)
    trainer = IDLMTrainer(
        IDLMModel(IDLMModelConfig.tiny()),
        dataset,
        dataset,
        cursor,
        config,
        device=torch.device("cpu"),
        atomic_manifest=AtomicIdManifest.reference(),
    )
    forwards = 0
    forward_layout = trainer._forward_layout

    def counted_forward(*args, **kwargs):
        nonlocal forwards
        forwards += 1
        return forward_layout(*args, **kwargs)

    trainer._forward_layout = counted_forward
    metric = trainer.run_update()
    assert metric.step == 1
    assert metric.proposal_targets == 9
    assert metric.clean_targets == 12
    assert cursor.position == 3
    assert forwards == 2  # one differentiable forward per microbatch
    assert metric.clean_scale > 0
    assert not hasattr(trainer, "balance_scale")


def test_optimizer_update_is_invariant_to_accumulation_partition() -> None:
    dataset = _NativeRows()
    torch.manual_seed(177)
    initial = IDLMModel(IDLMModelConfig.tiny()).state_dict()

    def train_once(batch_size: int) -> IDLMTrainer:
        model = IDLMModel(IDLMModelConfig.tiny())
        model.load_state_dict(initial)
        trainer = IDLMTrainer(
            model,
            dataset,
            dataset,
            DeterministicChunkCursor(dataset, seed=7, shuffle=False),
            IDLMTrainingConfig(
                iterations=1,
                batch_size=batch_size,
                global_batch_size=3,
                validation_batch_size=2,
                validation_rows=2,
                warmdown_iters=0,
                compile_model=False,
            ),
            device=torch.device("cpu"),
            atomic_manifest=AtomicIdManifest.reference(),
        )
        trainer.run_update()
        return trainer

    split = train_once(1)
    whole = train_once(3)
    for name, value in split.model.state_dict().items():
        torch.testing.assert_close(value, whole.model.state_dict()[name], atol=2e-6, rtol=2e-6)


def test_stride_curriculum_is_explicit_and_finishes_at_serving_stride() -> None:
    dataset = _NativeRows()
    config = IDLMTrainingConfig(
        iterations=6,
        batch_size=2,
        global_batch_size=3,
        validation_batch_size=2,
        validation_rows=2,
        warmdown_iters=0,
        compile_model=False,
        stride_curriculum=((0, 2), (2, 3), (4, 4)),
    )
    trainer = IDLMTrainer(
        IDLMModel(IDLMModelConfig.tiny(block_size=4)),
        dataset,
        dataset,
        DeterministicChunkCursor(dataset, seed=7, shuffle=False),
        config,
        device=torch.device("cpu"),
        atomic_manifest=AtomicIdManifest.reference(),
    )
    assert [trainer._stride_for_update(step) for step in range(6)] == [
        2,
        2,
        3,
        3,
        4,
        4,
    ]
    trainer.completed_steps = 2
    assert len(trainer.validate().proposal_offset_accuracy) == 2
    trainer.completed_steps = 3
    assert len(trainer.validate().proposal_offset_accuracy) == 3

    bad = IDLMTrainingConfig(
        iterations=2,
        warmdown_iters=0,
        stride_curriculum=((0, 2), (1, 3)),
    )
    with pytest.raises(ValueError, match="final curriculum stride"):
        IDLMTrainer(
            IDLMModel(IDLMModelConfig.tiny(block_size=4)),
            dataset,
            dataset,
            DeterministicChunkCursor(dataset, seed=7, shuffle=False),
            bad,
            device=torch.device("cpu"),
            atomic_manifest=AtomicIdManifest.reference(),
        )

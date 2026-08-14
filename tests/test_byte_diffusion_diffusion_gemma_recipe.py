from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from pretraining.byte_diffusion.config import ByteDiffusionConfig
from pretraining.byte_diffusion.diffusion_gemma import EntropyBudgetSamplerConfig
from pretraining.byte_diffusion.diffusion_gemma_model import (
    DIFFUSION_GEMMA_PARAMETER_COUNT,
    DiffusionGemmaAttentionMetadata,
    DiffusionGemmaModel,
)
from pretraining.byte_diffusion.inference_diffusion_gemma import (
    generate_diffusion_gemma_continuation,
    sample_diffusion_gemma_canvas,
)
from pretraining.byte_diffusion.training_diffusion_gemma import (
    DiffusionGemmaBatch,
    diffusion_gemma_loss,
    diffusion_gemma_optimizer_step,
    prepare_diffusion_gemma_inputs,
    select_document_canvases,
    validate_diffusion_gemma,
)
from pretraining.byte_diffusion.training import TrainingBatch
from scripts.train_byte_diffusion_gemma import _recipe_batch, parse_args


def test_diffusion_gemma_cli_inherits_ablation_schedule_and_batch_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["train_byte_diffusion_gemma.py"])
    monkeypatch.setenv("ITERATIONS", "200")
    monkeypatch.setenv("WARMDOWN_ITERS", "0")
    monkeypatch.setenv("VAL_LOSS_EVERY", "50")
    monkeypatch.setenv("TRAIN_LOG_EVERY", "7")
    monkeypatch.setenv("BYTE_DIFFUSION_GEMMA_MICROBATCH", "11")
    monkeypatch.setenv("BYTE_DIFFUSION_GEMMA_VALIDATION_BATCH", "13")
    args = parse_args()
    assert (
        args.steps,
        args.warmdown_steps,
        args.val_every,
        args.log_every,
        args.batch_size,
        args.validation_batch_size,
    ) == (200, 0, 50, 7, 11, 13)


def tiny_config(**overrides: object) -> ByteDiffusionConfig:
    return ByteDiffusionConfig.tiny(
        local_dim=16,
        global_dim=32,
        local_heads=1,
        global_heads=1,
        encoder_layers=1,
        global_layers=1,
        decoder_layers=1,
        encoder_ffn_dim=32,
        global_ffn_dim=48,
        decoder_ffn_dim=32,
        local_window=16,
        decoder_prefix_window=16,
        **overrides,
    )


def one_document_batch(batch_size: int = 2) -> DiffusionGemmaBatch:
    torch.manual_seed(10)
    length = 16
    ids = torch.randint(0, 256, (batch_size, length))
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.arange(batch_size)[:, None].expand_as(ids)
    positions = torch.arange(length)[None].expand_as(ids)
    targets = torch.full_like(ids, -100)
    targets[:, :-1] = ids[:, 1:]
    return DiffusionGemmaBatch(
        ids, valid, documents, positions, targets, ids[:, 0].clone()
    )


def test_recipe_uses_physical_patch_cu_for_unaligned_packed_byte_lengths() -> None:
    ids = torch.tensor([[1, 2, 3, 262, 10, 11, 12, 13]])
    valid = torch.tensor([[True, True, True, False, True, True, True, True]])
    native = TrainingBatch(
        ids=ids,
        valid=valid,
        ar_targets=torch.full_like(ids, -100),
        bos_targets=torch.tensor([1, 10]),
        positions=torch.tensor([[0, 1, 2, 0, 0, 1, 2, 3]]),
        full_valid=False,
        document_ids=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]),
        isolate_documents=True,
        byte_indices=torch.tensor([0, 1, 2, 4, 5, 6, 7]),
        byte_cu_seqlens=torch.tensor([0, 3, 7], dtype=torch.int32),
        patch_indices=torch.tensor([0, 1]),
        patch_cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    batch = _recipe_batch(native, patch_stride=4)
    assert batch.attention_metadata is not None
    assert batch.attention_metadata.byte_cu_seqlens.tolist() == [0, 3, 7]
    assert batch.attention_metadata.patch_cu_seqlens.tolist() == [0, 1, 2]


def test_production_cell_has_closed_parameter_count_and_no_mode_or_time_parameters() -> None:
    model = DiffusionGemmaModel(ByteDiffusionConfig())
    model.validate_production_parameterization()
    assert model.parameter_count == DIFFUSION_GEMMA_PARAMETER_COUNT
    names = tuple(name for name, _ in model.named_parameters())
    assert not any("mode" in name or "timestep" in name for name in names)
    assert model.self_conditioner[1].weight.shape == (64, 256)
    assert model.self_conditioner[3].weight.shape == (256, 64)


def test_document_relative_selection_never_crosses_a_packed_boundary() -> None:
    ids = torch.arange(16)[None]
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.tensor([[0] * 8 + [1] * 8])
    positions = torch.tensor([list(range(8)) + list(range(8))])
    selection = select_document_canvases(
        ids,
        valid,
        documents,
        positions,
        canvas_length=8,
        branches=64,
        patch_stride=4,
        pad_id=262,
        generator=torch.Generator().manual_seed(9),
    )
    # Every physical patch origin is eligible, including the final short
    # document tail. Synthetic PAD storage keeps its objective unbiased.
    assert set(selection.starts.flatten().tolist()) <= {0, 4, 8, 12}
    assert torch.all(
        selection.document_ids.eq(selection.document_ids[:, :, :1])
        | ~selection.valid
    )
    crossing = selection.starts.eq(4)
    assert crossing.any()
    assert torch.equal(
        selection.valid[crossing][0],
        torch.tensor([True, True, True, True, False, False, False, False]),
    )
    assert torch.all(selection.targets[crossing][:, 4:] == 262)


def test_multi_canvas_selection_systematically_covers_origin_strata() -> None:
    ids = torch.arange(64)[None]
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.zeros_like(ids)
    positions = torch.arange(64)[None]
    selection = select_document_canvases(
        ids,
        valid,
        documents,
        positions,
        canvas_length=16,
        branches=4,
        patch_stride=4,
        pad_id=262,
        generator=torch.Generator().manual_seed(10),
    )
    origin_ranks = selection.starts // 4
    sorted_ranks = origin_ranks.sort(1).values
    circular_gaps = torch.cat(
        (
            sorted_ranks[:, 1:] - sorted_ranks[:, :-1],
            16 + sorted_ranks[:, :1] - sorted_ranks[:, -1:],
        ),
        dim=1,
    )
    assert circular_gaps.tolist() == [[4, 4, 4, 4]]


def test_authenticated_candidate_selection_avoids_python_tensor_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = torch.arange(8)[None]
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.zeros_like(ids)
    positions = torch.arange(8)[None]

    def reject_tensor_truth(_tensor: torch.Tensor) -> bool:
        raise AssertionError("candidate validation materialized a tensor scalar")

    monkeypatch.setattr(torch.Tensor, "__bool__", reject_tensor_truth)
    selection = select_document_canvases(
        ids,
        valid,
        documents,
        positions,
        canvas_length=4,
        branches=2,
        patch_stride=4,
        pad_id=262,
        generator=torch.Generator().manual_seed(11),
        validate_candidates=False,
    )
    assert selection.starts.shape == (1, 2)


def test_reference_candidate_selection_still_rejects_empty_rows() -> None:
    ids = torch.zeros((1, 8), dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    with pytest.raises(ValueError, match="every row needs"):
        select_document_canvases(
            ids,
            valid,
            torch.zeros_like(ids),
            torch.arange(8)[None],
            canvas_length=4,
            branches=1,
            patch_stride=4,
            pad_id=262,
            generator=torch.Generator().manual_seed(13),
        )


def test_synthetic_canvas_supervises_last_patch_and_short_document_tail() -> None:
    ids = torch.arange(16)[None]
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.tensor([[0] * 4 + [1] * 12])
    positions = torch.tensor([list(range(4)) + list(range(12))])
    selection = select_document_canvases(
        ids,
        valid,
        documents,
        positions,
        canvas_length=8,
        branches=256,
        patch_stride=4,
        pad_id=262,
        generator=torch.Generator().manual_seed(91),
    )
    final = selection.starts.eq(12)
    assert final.any()
    assert torch.equal(
        selection.valid[final][0],
        torch.tensor([True, True, True, True, False, False, False, False]),
    )
    assert selection.targets[final][0, :4].tolist() == [12, 13, 14, 15]
    assert selection.positions[final][0].tolist() == list(range(8, 16))

    model = DiffusionGemmaModel(tiny_config()).eval()
    output = model(
        ids,
        valid,
        documents,
        positions,
        selection.targets[final][:1, None],
        selection.valid[final][:1, None],
        torch.tensor([[12]]),
        self_condition=False,
    )
    assert output.branch_logits.shape == (1, 1, 8, 261)


@pytest.mark.parametrize("with_ngrams", [False, True])
def test_shared_bank_isolates_documents_and_canvas_branches(
    with_ngrams: bool,
) -> None:
    ngram_overrides = (
        {
            "ngram_enabled": True,
            "ngram_orders": (3, 4),
            "ngram_table_size": 64,
            "ngram_rank": 4,
        }
        if with_ngrams
        else {}
    )
    model = DiffusionGemmaModel(tiny_config(**ngram_overrides)).eval()
    ids = torch.randint(0, 256, (1, 16), generator=torch.Generator().manual_seed(2))
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.tensor([[0] * 8 + [1] * 8])
    positions = torch.tensor([list(range(8)) + list(range(8))])
    starts = torch.tensor([[0, 8]])
    branch_valid = torch.ones((1, 2, 8), dtype=torch.bool)
    noisy = torch.randint(0, 256, (1, 2, 8), generator=torch.Generator().manual_seed(3))
    first = model(
        ids, valid, documents, positions, noisy, branch_valid, starts,
        self_condition=False,
    )
    changed_ids = ids.clone()
    changed_ids[:, 8:] = (changed_ids[:, 8:] + 73) % 256
    changed_noisy = noisy.clone()
    changed_noisy[:, 1] = (changed_noisy[:, 1] + 91) % 256
    second = model(
        changed_ids,
        valid,
        documents,
        positions,
        changed_noisy,
        branch_valid,
        starts,
        self_condition=False,
    )
    torch.testing.assert_close(first.clean_logits[:, :8], second.clean_logits[:, :8])
    torch.testing.assert_close(first.branch_logits[:, 0], second.branch_logits[:, 0])

    changed_ids = ids.clone()
    changed_ids[:, :8] = (changed_ids[:, :8] + 37) % 256
    changed_noisy = noisy.clone()
    changed_noisy[:, 0] = (changed_noisy[:, 0] + 53) % 256
    third = model(
        changed_ids,
        valid,
        documents,
        positions,
        changed_noisy,
        branch_valid,
        starts,
        self_condition=False,
    )
    torch.testing.assert_close(first.clean_logits[:, 8:], third.clean_logits[:, 8:])
    torch.testing.assert_close(first.branch_logits[:, 1], third.branch_logits[:, 1])


def test_prepacked_document_lowering_matches_implicit_cpu_oracle() -> None:
    model = DiffusionGemmaModel(tiny_config()).eval()
    ids = torch.randint(0, 256, (1, 16), generator=torch.Generator().manual_seed(21))
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.tensor([[0] * 8 + [1] * 8])
    positions = torch.tensor([list(range(8)) + list(range(8))])
    starts = torch.tensor([[0, 8]])
    branch_valid = torch.ones((1, 2, 8), dtype=torch.bool)
    noisy = torch.randint(0, 256, (1, 2, 8), generator=torch.Generator().manual_seed(22))
    metadata = DiffusionGemmaAttentionMetadata(
        byte_indices=torch.arange(16),
        byte_cu_seqlens=torch.tensor([0, 8, 16], dtype=torch.int32),
        patch_indices=torch.arange(4),
        patch_cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    implicit = model(
        ids,
        valid,
        documents,
        positions,
        noisy,
        branch_valid,
        starts,
        self_condition=False,
    )
    explicit = model(
        ids,
        valid,
        documents,
        positions,
        noisy,
        branch_valid,
        starts,
        self_condition=False,
        attention_metadata=metadata,
    )
    torch.testing.assert_close(explicit.clean_logits, implicit.clean_logits)
    torch.testing.assert_close(explicit.branch_logits, implicit.branch_logits)


def test_compact_prior_uses_exactly_half_rows_and_trains_width_preserving_ffw() -> None:
    model = DiffusionGemmaModel(tiny_config())
    batch = one_document_batch(4)
    loss = diffusion_gemma_loss(
        model,
        batch,
        canvas_length=8,
        branches=2,
        self_condition=True,
        generator=torch.Generator().manual_seed(14),
    )
    assert loss.output.prior_batch_size == 2
    assert loss.output.self_conditioned_rows is not None
    assert int(loss.output.self_conditioned_rows.sum()) == 2
    loss.total.backward()
    assert model.self_conditioner[1].weight.grad is not None
    assert model.self_conditioner[3].weight.grad is not None
    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()


def test_dense_objective_supervises_unchanged_atoms_including_unreplaced_positions() -> None:
    model = DiffusionGemmaModel(tiny_config())
    batch = one_document_batch()
    loss = diffusion_gemma_loss(
        model,
        batch,
        canvas_length=8,
        branches=1,
        self_condition=False,
        integrated_exact_k=True,
        generator=torch.Generator().manual_seed(7),
    )
    # Dense supervision partitions *every* active atom by whether its visible
    # value changed. Unreplaced atoms (and replacement collisions) remain in CE.
    assert int(loss.unchanged_targets) > 0
    assert int(loss.changed_targets + loss.unchanged_targets) == int(
        loss.diffusion_targets
    )
    assert torch.isfinite(loss.diffusion)


def test_external_sampler_prior_is_detached_but_trains_shared_ffw_and_embedding() -> None:
    model = DiffusionGemmaModel(tiny_config())
    batch = one_document_batch()
    starts = torch.tensor([[0], [0]])
    valid = torch.ones((2, 1, 8), dtype=torch.bool)
    noisy = batch.clean_ids[:, None, :8].clone()
    source_logits = torch.randn(2, 1, 8, 261, requires_grad=True)
    output = model(
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        noisy,
        valid,
        starts,
        self_condition=False,
        prior_probabilities=source_logits.softmax(-1),
    )
    output.branch_logits.square().mean().backward()
    assert source_logits.grad is None
    assert model.self_conditioner[1].weight.grad is not None
    assert model.embedding.weight.grad is not None


def test_headline_bpb_is_exact_clean_ar_plus_atomic_bos_nll() -> None:
    model = DiffusionGemmaModel(tiny_config())
    batch = one_document_batch()
    loss = diffusion_gemma_loss(
        model,
        batch,
        canvas_length=8,
        branches=1,
        self_condition=False,
        generator=torch.Generator().manual_seed(5),
    )
    clean_total = F.cross_entropy(
        loss.output.clean_logits.flatten(0, 1),
        batch.ar_targets.flatten(),
        ignore_index=-100,
        reduction="sum",
    )
    bos_total = F.cross_entropy(
        model.forward_bos_logits(2, device=batch.clean_ids.device),
        batch.bos_targets,
        reduction="sum",
    )
    count = batch.ar_targets.ne(-100).sum() + 2
    expected = (clean_total + bos_total) / count / math.log(2.0)
    torch.testing.assert_close(loss.headline_bpb, expected)


def test_virtual_bos_matches_the_ordinary_incomplete_patch_clean_path() -> None:
    model = DiffusionGemmaModel(tiny_config()).eval()
    batch = one_document_batch()
    ids = batch.clean_ids.clone()
    ids[:, 0] = model.config.vocab.eot_id
    noisy = ids[:, None, :8].clone()
    output = model(
        ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        noisy,
        torch.ones((2, 1, 8), dtype=torch.bool),
        torch.zeros((2, 1), dtype=torch.long),
        self_condition=False,
    )
    expected = model.forward_bos_logits(2, device=ids.device)
    torch.testing.assert_close(output.clean_logits[:, 0], expected.expand(2, -1))


def test_update_global_denominators_make_tail_microbatches_exact() -> None:
    model = DiffusionGemmaModel(tiny_config()).eval()
    batch = one_document_batch(3)
    prepared = prepare_diffusion_gemma_inputs(
        model,
        batch,
        canvas_length=8,
        branches=1,
        integrated_exact_k=True,
        generator=torch.Generator().manual_seed(37),
    )
    diffusion_denominator = prepared.corruption.active.sum()
    ar_denominator = batch.ar_targets.ne(-100).sum() + batch.bos_targets.numel()
    full = diffusion_gemma_loss(
        model,
        batch,
        canvas_length=8,
        branches=1,
        self_condition=False,
        prepared=prepared,
        diffusion_denominator=diffusion_denominator,
        ar_denominator=ar_denominator,
    )
    split_total = full.total.new_zeros(())
    for start, stop in ((0, 2), (2, 3)):
        part = DiffusionGemmaBatch(
            batch.clean_ids[start:stop],
            batch.clean_valid[start:stop],
            batch.document_ids[start:stop],
            batch.positions[start:stop],
            batch.ar_targets[start:stop],
            batch.bos_targets[start:stop],
        )
        split_total += diffusion_gemma_loss(
            model,
            part,
            canvas_length=8,
            branches=1,
            self_condition=False,
            prepared=prepared.slice_rows(start, stop),
            diffusion_denominator=diffusion_denominator,
            ar_denominator=ar_denominator,
        ).total
    torch.testing.assert_close(split_total, full.total)


def test_cpu_optimizer_path_updates_parameters() -> None:
    model = DiffusionGemmaModel(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.output.weight.detach().clone()
    loss = diffusion_gemma_optimizer_step(
        model,
        optimizer,
        one_document_batch(),
        canvas_length=8,
        branches=1,
        self_condition=True,
        generator=torch.Generator().manual_seed(6),
    )
    assert torch.isfinite(loss.total)
    assert not torch.equal(before, model.output.weight)


def test_sampler_is_wired_to_model_and_returns_static_final_canvas() -> None:
    model = DiffusionGemmaModel(tiny_config()).eval()
    batch = one_document_batch()
    result = sample_diffusion_gemma_canvas(
        model,
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        torch.tensor([[0], [0]]),
        torch.ones((2, 1, 8), dtype=torch.bool),
        EntropyBudgetSamplerConfig(max_steps=2, stop_entropy=0.0),
        generator=torch.Generator().manual_seed(20),
    )
    assert result.ids.shape == (2, 8)
    assert result.valid.shape == result.ids.shape
    assert result.model_forwards == 2
    assert result.denoising_forwards == 2
    assert result.clean_commit_forwards == 0
    assert result.requires_clean_commit
    assert result.state.finished.all()


class _AlwaysEOTDiffusionGemma:
    def __init__(self, token_id: int | None = None) -> None:
        self.config = tiny_config()
        self.token_id = self.config.vocab.eot_id if token_id is None else token_id
        self.calls = 0

    def prepare_attention_metadata(self, clean_valid, document_ids):
        del clean_valid, document_ids
        return None

    def __call__(
        self,
        clean_ids,
        clean_valid,
        document_ids,
        positions,
        noisy_ids,
        branch_valid,
        branch_starts,
        **kwargs,
    ):
        del clean_valid, document_ids, positions, branch_valid, branch_starts, kwargs
        self.calls += 1
        logits = torch.full(
            (*noisy_ids.shape, self.config.vocab.output_size),
            -30.0,
            device=clean_ids.device,
        )
        logits[..., self.token_id] = 30.0
        return SimpleNamespace(branch_logits=logits)


def test_end_to_end_generation_keeps_prompt_clean_and_stops_at_eot() -> None:
    model = _AlwaysEOTDiffusionGemma()
    prompt = torch.tensor([[10, 11, 12], [20, 21, 262]])
    prompt_valid = torch.tensor([[True, True, True], [True, True, False]])
    traces = [
        {"alignment_steps": [], "diffusion_blocks": []},
        {"alignment_steps": [], "diffusion_blocks": []},
    ]
    result = generate_diffusion_gemma_continuation(
        model,
        prompt,
        prompt_valid,
        canvas_length=8,
        max_new_tokens=10,
        sampler_config=EntropyBudgetSamplerConfig(max_steps=1),
        generator=torch.Generator().manual_seed(101),
        trace_records=traces,
    )
    assert result.ids[0, :3].tolist() == [10, 11, 12]
    assert result.ids[1, :2].tolist() == [20, 21]
    assert result.ids[0, 3].item() == model.config.vocab.eot_id
    assert result.ids[1, 2].item() == model.config.vocab.eot_id
    assert result.lengths.tolist() == [4, 3]
    assert result.generated_valid.sum(1).tolist() == [1, 1]
    assert result.finished.tolist() == [True, True]
    assert result.canvases == 1
    assert result.denoising_forwards == 1
    assert result.clean_prefill_forwards == 1
    assert result.clean_commit_forwards == 1
    assert result.model_forwards == model.calls == 3
    assert not result.cache_backed
    for trace in traces:
        blocks = trace["diffusion_blocks"]
        assert len(blocks) == 1
        assert blocks[0]["prompt_atoms_noised"] is False
        assert len(blocks[0]["steps"]) == 1
        assert blocks[0]["steps"][0]["step"] == 1


def test_end_to_end_generation_uses_multiple_canvases_and_honors_canvas_cap() -> None:
    prompt = torch.tensor([[10, 11, 12]])
    prompt_valid = torch.ones_like(prompt, dtype=torch.bool)
    complete = generate_diffusion_gemma_continuation(
        _AlwaysEOTDiffusionGemma(token_id=7),
        prompt,
        prompt_valid,
        canvas_length=8,
        max_new_tokens=10,
        sampler_config=EntropyBudgetSamplerConfig(max_steps=1),
    )
    assert complete.lengths.tolist() == [13]
    assert complete.generated_valid.sum().item() == 10
    assert complete.canvases == 2
    assert not complete.finished.item()
    assert complete.clean_commit_forwards == 2

    capped = generate_diffusion_gemma_continuation(
        _AlwaysEOTDiffusionGemma(token_id=7),
        prompt,
        prompt_valid,
        canvas_length=8,
        max_new_tokens=10,
        max_canvases=1,
        sampler_config=EntropyBudgetSamplerConfig(max_steps=1),
    )
    assert capped.lengths.tolist() == [8]
    assert capped.generated_valid.sum().item() == 5
    assert capped.canvases == 1


def test_real_cell_accepts_synthetic_fresh_suffix_without_noising_prompt() -> None:
    model = DiffusionGemmaModel(tiny_config()).eval()
    prompt = torch.tensor([[10, 11, 12]])
    result = generate_diffusion_gemma_continuation(
        model,
        prompt,
        torch.ones_like(prompt, dtype=torch.bool),
        canvas_length=8,
        max_new_tokens=1,
        sampler_config=EntropyBudgetSamplerConfig(max_steps=1),
        generator=torch.Generator().manual_seed(109),
    )
    assert result.ids[0, :3].tolist() == [10, 11, 12]
    assert result.lengths.tolist() == [4]
    assert result.generated_valid.sum().item() == 1


def test_odd_tail_batch_uses_nearest_representable_half_self_conditioning() -> None:
    model = DiffusionGemmaModel(tiny_config())
    batch = one_document_batch(3)
    loss = diffusion_gemma_loss(
        model,
        batch,
        canvas_length=8,
        branches=1,
        self_condition=True,
        generator=torch.Generator().manual_seed(29),
    )
    assert loss.output.prior_batch_size == 1


def test_validation_keeps_denoising_ce_separate_from_ar_bpb_anchor() -> None:
    model = DiffusionGemmaModel(tiny_config())
    metric = validate_diffusion_gemma(
        model,
        [one_document_batch()],
        canvas_length=8,
        branches=1,
        seed=31,
        self_condition=False,
    )
    assert math.isfinite(metric.denoising_ce_nats_per_atom)
    assert math.isfinite(metric.ar_anchor_bpb)
    assert 0 < metric.diffusion_targets <= 16
    assert (
        metric.changed_targets + metric.unchanged_targets
        == metric.diffusion_targets
    )
    assert 0.0 <= metric.denoising_accuracy <= 1.0

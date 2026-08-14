from __future__ import annotations

import pytest
import torch

from pretraining.byte_diffusion.config import ByteDiffusionConfig
from pretraining.byte_diffusion.inference import (
    CachedCanvasGenerator,
    LayerKV,
    _append_layer_kv,
    _gather_absolute_positions,
    append_clean_block,
    denoise_blt_cached,
    denoise_canvas_cached,
    document_start_ar_metadata,
    prefill_prefix,
    prepare_cached_canvas,
)
from pretraining.byte_diffusion.model import ByteDiffusionModel, ModelOutput
from pretraining.byte_diffusion.state import TransactionalDecodeState


class _FixedARModel(ByteDiffusionModel):
    """Tiny real hierarchy with deterministic AR logits for alignment tests."""

    def forward_ar_varlen(self, ids, valid, **kwargs):  # type: ignore[no-untyped-def]
        logits = torch.full(
            (*ids.shape, self.config.vocab.output_size),
            -20.0,
            device=ids.device,
        )
        logits[..., 65] = 20.0
        byte_states = torch.zeros(
            (*ids.shape, self.config.local_dim), device=ids.device
        )
        patch_states = torch.zeros(
            ids.shape[0],
            ids.shape[1] // self.config.patch_stride,
            self.config.global_dim,
            device=ids.device,
        )
        return ModelOutput(logits, byte_states, patch_states)


class _BlockedFirstARModel(_FixedARModel):
    def forward_ar_varlen(self, ids, valid, **kwargs):  # type: ignore[no-untyped-def]
        output = super().forward_ar_varlen(ids, valid, **kwargs)
        output.logits[..., 257] = 40.0
        return output


def test_geometric_kv_append_copies_when_branching_from_an_old_prefix() -> None:
    def block(value: float) -> torch.Tensor:
        return torch.tensor([[[[value]]]])

    base = LayerKV(block(10), block(20))
    first = _append_layer_kv(base, block(11), block(21))
    descendant = _append_layer_kv(first, block(12), block(22))
    descendant_key = descendant.key.clone()
    descendant_value = descendant.value.clone()

    branch = _append_layer_kv(first, block(99), block(109))

    torch.testing.assert_close(descendant.key, descendant_key)
    torch.testing.assert_close(descendant.value, descendant_value)
    assert branch.key.flatten().tolist() == [10, 11, 99]
    assert branch.value.flatten().tolist() == [20, 21, 109]


def test_ragged_cache_gather_uses_semantic_positions_not_storage_slots() -> None:
    values = torch.tensor(
        [
            [[10.0, 11.0], [20.0, 21.0], [90.0, 91.0], [30.0, 31.0]],
            [[40.0, 41.0], [80.0, 81.0], [50.0, 51.0], [60.0, 61.0]],
        ]
    )
    valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
    positions = torch.tensor([[0, 1, 2, 2], [0, 1, 1, 2]])
    targets = torch.tensor([[2, 0, 7], [1, 2, -1]])
    gathered = _gather_absolute_positions(
        values, valid, positions, targets, fill_value=-1
    )
    expected = torch.tensor(
        [
            [[30.0, 31.0], [10.0, 11.0], [-1.0, -1.0]],
            [[50.0, 51.0], [60.0, 61.0], [-1.0, -1.0]],
        ]
    )
    torch.testing.assert_close(gathered, expected)


@pytest.mark.parametrize("ngram_enabled", [False, True])
def test_cached_canvas_matches_shared_bank_reference(ngram_enabled: bool) -> None:
    torch.manual_seed(41)
    config = ByteDiffusionConfig.tiny(
        ngram_enabled=ngram_enabled,
        ngram_table_size=128,
        ngram_rank=4,
        ngram_orders=(3, 4),
    )
    model = ByteDiffusionModel(config).eval()
    clean = torch.tensor([[65, 66, 67, 68, 69, 70, 71, 256]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    noisy = torch.tensor([[[261, 70, 261, 256]]])
    branch_valid = torch.ones_like(noisy, dtype=torch.bool)
    starts = torch.tensor([[4]])

    expected = model.forward_canvas_branches(
        clean,
        valid,
        noisy,
        branch_valid,
        starts,
        assume_full_clean=True,
    ).branch_logits[:, 0]
    cache = prefill_prefix(
        model,
        clean,
        allow_dense_reference=True,
        document_start=False,
    )
    actual = denoise_canvas_cached(
        model,
        cache,
        noisy,
        starts,
        allow_dense_reference=True,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_prefill_rejects_unaligned_prefix() -> None:
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    try:
        prefill_prefix(model, torch.tensor([[1, 2, 256]]), allow_dense_reference=True)
    except ValueError as error:
        assert "patch aligned" in str(error)
    else:
        raise AssertionError("unaligned prefix was accepted")


def test_prefill_global_cache_starts_with_training_virtual_bos() -> None:
    torch.manual_seed(417)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    ids = torch.tensor([[65, 66, 67, 68]])

    cache = prefill_prefix(model, ids, allow_dense_reference=True)
    standalone = model.virtual_bos_global_states(
        1, device=ids.device, allow_dense_reference=True
    )

    assert cache.has_virtual_bos
    assert cache.patch_positions.tolist() == [[0, 1]]
    torch.testing.assert_close(
        cache.patch_states[:, 0], standalone, rtol=1e-6, atol=1e-7
    )


def test_cached_blt_matches_document_branch_reference() -> None:
    torch.manual_seed(419)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    clean = torch.tensor([[65, 66, 67, 68, 69, 70, 71, 256]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    noisy = torch.tensor([[[261, 70, 261, 256]]])
    branch_valid = torch.ones_like(noisy, dtype=torch.bool)
    starts = torch.tensor([[4]])

    metadata = document_start_ar_metadata(valid, model.config.patch_stride)
    expected = model.forward_blt_d_branches(
        clean,
        valid,
        noisy,
        branch_valid,
        starts,
        document_ids=torch.zeros_like(clean),
        branch_condition_indices=torch.tensor([[1]]),
        **metadata,
    ).branch_logits[:, 0]
    cache = prefill_prefix(
        model, clean[:, :4], allow_dense_reference=True
    )
    actual = denoise_blt_cached(
        model,
        cache,
        noisy,
        starts,
        allow_dense_reference=True,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_batched_virtual_bos_logits_match_independent_document_forwards() -> None:
    torch.manual_seed(420)
    config = ByteDiffusionConfig.tiny()
    model = ByteDiffusionModel(config).eval()
    ids = torch.tensor(
        [
            [65, 66, 67, config.vocab.pad_id],
            [70, 71, config.vocab.pad_id, config.vocab.pad_id],
        ]
    )
    valid = ids.ne(config.vocab.pad_id)
    positions = torch.arange(ids.shape[1])[None].expand_as(ids)

    batched = model.forward_ar_varlen(
        ids,
        valid,
        positions=positions,
        allow_dense_reference=True,
        return_padded_logits=False,
        **document_start_ar_metadata(valid, config.patch_stride),
    ).logits
    independent = torch.cat(
        tuple(
            model.forward_ar_varlen(
                ids[row : row + 1],
                valid[row : row + 1],
                positions=positions[row : row + 1],
                allow_dense_reference=True,
                return_padded_logits=False,
                **document_start_ar_metadata(
                    valid[row : row + 1], config.patch_stride
                ),
            ).logits
            for row in range(ids.shape[0])
        )
    )
    without_bos = model.forward_ar_varlen(
        ids,
        valid,
        positions=positions,
        allow_dense_reference=True,
        return_padded_logits=False,
    ).logits

    torch.testing.assert_close(batched, independent, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(batched, without_bos, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("ngram_enabled", [False, True])
def test_incremental_clean_append_matches_full_prefill(ngram_enabled: bool) -> None:
    torch.manual_seed(421)
    config = ByteDiffusionConfig.tiny(
        ngram_enabled=ngram_enabled,
        ngram_table_size=128,
        ngram_rank=4,
        ngram_orders=(3, 4),
    )
    model = ByteDiffusionModel(config).eval()
    first = torch.tensor([[65, 66, 67, 68]])
    second = torch.tensor([[69, 70, 71, 72]])

    appended = append_clean_block(
        model,
        prefill_prefix(model, first, allow_dense_reference=True),
        second,
    )
    full = prefill_prefix(
        model,
        torch.cat((first, second), dim=1),
        allow_dense_reference=True,
    )

    torch.testing.assert_close(
        appended.patch_states, full.patch_states, rtol=2e-5, atol=2e-5
    )
    for incremental, reference in zip(
        (*appended.local, *appended.global_, *appended.decoder),
        (*full.local, *full.global_, *full.decoder),
        strict=True,
    ):
        torch.testing.assert_close(
            incremental.key, reference.key, rtol=2e-5, atol=2e-5
        )
        torch.testing.assert_close(
            incremental.value, reference.value, rtol=2e-5, atol=2e-5
        )


@pytest.mark.parametrize(
    ("device_name", "allow_dense_reference"),
    [
        ("cpu", True),
        pytest.param(
            "cuda",
            False,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_ragged_cached_blt_matches_independent_rows_after_append(
    device_name: str, allow_dense_reference: bool
) -> None:
    """Padding holes must not hide subsequently appended semantic prefix K/V."""

    torch.manual_seed(422)
    config = ByteDiffusionConfig.tiny()
    device = torch.device(device_name)
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = ByteDiffusionModel(config).to(device=device, dtype=model_dtype).eval()
    pad = config.vocab.pad_id
    ids = torch.tensor(
        [
            [65, 66, 67, 68, pad, pad, pad, pad],
            [70, 71, 72, 73, 74, 75, 76, 77],
        ]
    ).to(device)
    valid = ids.ne(pad)
    committed = torch.tensor(
        [[78, 79, 80, 81], [82, 83, 84, 85]]
    ).to(device)
    noisy = torch.full(
        (2, 1, 4), config.vocab.mask_id, dtype=torch.long, device=device
    )

    batched_cache = append_clean_block(
        model,
        prefill_prefix(
            model, ids, valid=valid, allow_dense_reference=allow_dense_reference
        ),
        committed,
    )
    starts = batched_cache.valid.sum(1).to(torch.long)[:, None]
    plan = prepare_cached_canvas(
        model,
        batched_cache,
        starts,
        4,
        allow_dense_reference=allow_dense_reference,
    )
    # Prefix cutoffs index physical cache storage.  In the first row the
    # appended semantic positions 4..7 live in physical slots 8..11.
    assert plan.local_layout.prefix_lengths.tolist() == [[12], [12]]
    assert plan.global_layout.prefix_lengths.tolist() == [[4], [4]]
    batched = denoise_blt_cached(
        model,
        batched_cache,
        noisy,
        starts,
        allow_dense_reference=allow_dense_reference,
        plan=plan,
    )

    independent = []
    independent_caches = []
    lengths = valid.sum(1).tolist()
    for row, length in enumerate(lengths):
        row_cache = append_clean_block(
            model,
            prefill_prefix(
                model,
                ids[row : row + 1, :length],
                allow_dense_reference=allow_dense_reference,
            ),
            committed[row : row + 1],
        )
        row_start = row_cache.valid.sum(1).to(torch.long)[:, None]
        independent_caches.append(row_cache)
        independent.append(
            denoise_blt_cached(
                model,
                row_cache,
                noisy[row : row + 1],
                row_start,
                allow_dense_reference=allow_dense_reference,
            )
        )

    tolerance = 2e-2 if device.type == "cuda" else 2e-5
    torch.testing.assert_close(
        batched, torch.cat(independent), rtol=tolerance, atol=tolerance
    )

    second_committed = torch.tensor(
        [[86, 87, 88, 89], [90, 91, 92, 93]], device=device
    )
    batched_cache = append_clean_block(model, batched_cache, second_committed)
    batched_starts = batched_cache.valid.sum(1).to(torch.long)[:, None]
    batched = denoise_blt_cached(
        model,
        batched_cache,
        noisy,
        batched_starts,
        allow_dense_reference=allow_dense_reference,
    )
    independent = []
    for row, row_cache in enumerate(independent_caches):
        row_cache = append_clean_block(
            model, row_cache, second_committed[row : row + 1]
        )
        independent.append(
            denoise_blt_cached(
                model,
                row_cache,
                noisy[row : row + 1],
                row_cache.valid.sum(1).to(torch.long)[:, None],
                allow_dense_reference=allow_dense_reference,
            )
        )
    torch.testing.assert_close(
        batched, torch.cat(independent), rtol=tolerance, atol=tolerance
    )


def test_generator_completes_partial_patch_with_at_most_three_ar_atoms() -> None:
    config = ByteDiffusionConfig.tiny()
    model = _FixedARModel(config).eval()
    state = TransactionalDecodeState(
        eot_id=config.vocab.eot_id,
        patch_stride=config.patch_stride,
    )
    generator = CachedCanvasGenerator(model, state, allow_dense_reference=True)
    generator.prefill(torch.tensor([97]))

    alignment = generator.align_prefix_ar()

    assert alignment.tolist() == [65, 65, 65]
    assert state.ids.tolist() == [97, 65, 65, 65]
    assert state.patch_phase == 0
    assert generator.cache is not None
    assert state.counters.forwards == 4
    assert state.counters.denoise_forwards == 0
    assert state.counters.causal_replays == 1


def test_generator_output_mask_applies_to_ar_alignment() -> None:
    config = ByteDiffusionConfig.tiny()
    model = _BlockedFirstARModel(config).eval()
    state = TransactionalDecodeState(eot_id=config.vocab.eot_id)
    generator = CachedCanvasGenerator(
        model,
        state,
        allow_dense_reference=True,
        blocked_output_ids=torch.tensor([257]),
    )
    generator.prefill(torch.tensor([97]))

    assert generator.align_prefix_ar().tolist() == [65, 65, 65]


def test_generator_cannot_block_eot() -> None:
    config = ByteDiffusionConfig.tiny()
    with pytest.raises(ValueError, match="EOT cannot be blocked"):
        CachedCanvasGenerator(
            ByteDiffusionModel(config),
            TransactionalDecodeState(eot_id=config.vocab.eot_id),
            blocked_output_ids=torch.tensor([config.vocab.eot_id]),
        )


def test_generator_rejects_empty_or_cross_eot_prefix() -> None:
    config = ByteDiffusionConfig.tiny()
    generator = CachedCanvasGenerator(
        ByteDiffusionModel(config),
        TransactionalDecodeState(eot_id=config.vocab.eot_id),
        allow_dense_reference=True,
    )
    for prefix in (torch.empty(0, dtype=torch.long), torch.tensor([256, 65])):
        try:
            generator.prefill(prefix)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid generation prefix was accepted")


def test_generator_runs_canvas_beyond_prefix_bank_and_replays_commit() -> None:
    torch.manual_seed(7)
    config = ByteDiffusionConfig.tiny()
    state = TransactionalDecodeState(
        eot_id=config.vocab.eot_id,
        patch_stride=config.patch_stride,
    )
    generator = CachedCanvasGenerator(
        ByteDiffusionModel(config).eval(), state, allow_dense_reference=True
    )
    generator.prefill(torch.tensor([65, 66, 67, 68]))

    result = generator.generate(canvas_length=4, steps=1)

    assert result.canvas is not None
    assert result.canvas.executed_nfe == 1
    assert result.committed_canvas_ids.numel() >= 1
    assert state.counters.denoise_forwards == 1
    assert state.counters.prefix_prefills == 1
    assert state.counters.causal_replays == 1
    assert state.counters.forwards == 3
    assert generator.cache is not None
    assert int(generator.cache.valid.sum()) == state.ids.numel()


def test_partial_terminal_replay_masks_storage_pad() -> None:
    config = ByteDiffusionConfig.tiny()
    ids = torch.tensor([[65, 256, 262, 262]])
    valid = torch.tensor([[True, True, False, False]])

    cache = prefill_prefix(
        ByteDiffusionModel(config).eval(),
        ids,
        valid=valid,
        allow_dense_reference=True,
    )

    assert cache.valid.tolist() == valid.tolist()
    assert cache.patch_valid.tolist() == [[True, True]]
    assert cache.ids[~cache.valid].eq(config.vocab.pad_id).all()

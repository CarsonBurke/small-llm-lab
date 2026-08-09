from __future__ import annotations

import torch

from pretraining.byte_diffusion.config import ByteDiffusionConfig
from pretraining.byte_diffusion.inference import (
    CachedCanvasGenerator,
    denoise_canvas_cached,
    prefill_prefix,
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


def test_cached_canvas_matches_shared_bank_reference() -> None:
    torch.manual_seed(41)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
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
    cache = prefill_prefix(model, clean, allow_dense_reference=True)
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
    assert cache.patch_valid.tolist() == [[True]]
    assert cache.ids[~cache.valid].eq(config.vocab.pad_id).all()

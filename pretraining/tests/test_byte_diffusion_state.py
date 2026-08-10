from __future__ import annotations

import torch

from pretraining.byte_diffusion.state import TransactionalDecodeState


def test_abort_restores_semantic_cache_and_rng_but_keeps_work() -> None:
    state = TransactionalDecodeState(eot_id=256)
    state.ids = torch.tensor([97, 98, 99])
    state.install_replayed_cache("k", torch.arange(6).view(2, 3))
    before = state.snapshot()
    state.begin()
    state.note_work(4)
    state.ids = torch.tensor([1, 2])
    state.caches["k"] = torch.zeros_like(state.caches["k"])
    state.abort()
    torch.testing.assert_close(state.ids, before.ids, rtol=0, atol=0)
    torch.testing.assert_close(state.caches["k"], before.tensors["k"], rtol=0, atol=0)
    torch.testing.assert_close(state._generator.get_state(), before.generator_state, rtol=0, atol=0)
    assert state.counters.forwards == 1


def test_abort_restores_in_place_tensor_edits() -> None:
    state = TransactionalDecodeState(eot_id=256)
    state.replace_prefix(torch.tensor([97, 98, 99]))
    state.install_replayed_cache("k", torch.arange(3))
    state.begin()
    state.ids[0] = 7
    state.caches["k"][0] = 8
    state.abort()
    torch.testing.assert_close(state.ids, torch.tensor([97, 98, 99]))
    torch.testing.assert_close(state.caches["k"], torch.arange(3))


def test_patch_phase_refreshes_after_in_place_prefix_edit() -> None:
    state = TransactionalDecodeState(eot_id=256)
    state.replace_prefix(torch.tensor([97, 98, 99]))
    assert state.patch_phase == 3
    state.ids[-1] = state.eot_id
    assert state.patch_phase == 0


def test_commit_truncates_at_eot_and_never_promotes_scratch_cache() -> None:
    state = TransactionalDecodeState(eot_id=256)
    state.ids = torch.tensor([97, 98])
    state.install_replayed_cache("k", torch.tensor([1.0]))
    state.begin()
    _ = torch.rand((), generator=state.generator)
    consumed_rng = state.generator.get_state().clone()
    state.caches["k"] = torch.tensor([999.0])
    accepted = state.commit_ids(torch.tensor([65, 66, 256, 99]))
    assert accepted.tolist() == [65, 66, 256]
    assert state.ids.tolist() == [97, 98, 65, 66, 256]
    assert state.caches["k"].tolist() == [1.0]
    torch.testing.assert_close(state.generator.get_state(), consumed_rng)
    assert state.patch_phase == 0


def test_checkpoint_roundtrip() -> None:
    source = TransactionalDecodeState(eot_id=256)
    source.seed(123)
    source.ids = torch.tensor([97, 256])
    source.install_replayed_cache("v", torch.tensor([[2.0]]))
    restored = TransactionalDecodeState(eot_id=256)
    restored.load_state_dict(source.state_dict())
    torch.testing.assert_close(restored.ids, source.ids, rtol=0, atol=0)
    torch.testing.assert_close(restored.caches["v"], source.caches["v"], rtol=0, atol=0)
    torch.testing.assert_close(
        restored._generator.get_state(), source._generator.get_state(), rtol=0, atol=0
    )


def test_incomplete_patch_buffer_tracks_only_the_active_suffix() -> None:
    state = TransactionalDecodeState(eot_id=256)
    state.replace_prefix(torch.tensor([65, 66, 256, 67, 68, 69]))
    assert state.patch_phase == 3
    assert state.incomplete_patch_buffer.tolist() == [67, 68, 69]

    state.replace_prefix(torch.tensor([65, 66, 256]))
    assert state.patch_phase == 0
    assert state.incomplete_patch_buffer.numel() == 0

from __future__ import annotations

import pytest
import torch

from pretraining.fresh_lejepa.fresh_lejepa_train import FreshLeJEPAGPT
from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
from postraining.latent_thought import (
    CombinedEmbedding,
    DecodeRangeMask,
    LatentThoughtModel,
    PINNED_EMIT_ROLLOUT_POLICY_SCHEMAS,
    RENDERER_FEATURES_SCHEMA,
    ROLLOUT_POLICY_SCHEMA,
    THOUGHT_INPUT_SCHEMA,
    THOUGHT_DISTRIBUTION_SCHEMA,
    combiner_init_kwargs_from_checkpoint,
    validate_renderer_checkpoint,
)
from postraining.model_io import _pope_construction

KWARGS = dict(
    vocab_size=32, num_layers=3, model_dim=32, num_heads=4, num_kv_heads=2,
    mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.01,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
)


def _pope_model() -> FreshLeJEPASharedRMSV1PoPE:
    with _pope_construction():
        return FreshLeJEPASharedRMSV1PoPE(**KWARGS).eval()


def test_token_step_matches_full_forward_with_belief_renderer():
    torch.manual_seed(3)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    ids = torch.randint(0, 32, (2, 9))
    caches = backbone.make_generation_cache(2, ids.size(1), torch.device("cpu"))
    with torch.no_grad():
        token_latents = backbone.embed_tokens(ids)
        beliefs = backbone.temporal_belief_from_token_latent(token_latents)
        expected = backbone.logits_from_features(
            wrapper.renderer_features(token_latents, beliefs)
        )
        for position in range(ids.size(1)):
            output = wrapper.token_step(ids[:, position], caches, position)
            caches = output.caches
            torch.testing.assert_close(output.logits, expected[:, position])


def _assert_prefill_matches_steps(
    wrapper: LatentThoughtModel,
    ids: torch.Tensor,
    key_valid: torch.Tensor | None = None,
) -> None:
    length = ids.size(1)
    dense_caches = wrapper.make_generation_cache(
        ids.size(0), length, torch.device("cpu")
    )
    stepped_caches = wrapper.make_generation_cache(
        ids.size(0), length, torch.device("cpu")
    )
    with torch.no_grad():
        dense = wrapper.prefill(ids, dense_caches, key_valid)
        stepped = None
        pad_lengths = None if key_valid is None else length - key_valid.sum(1)
        for position in range(length):
            key_mask = None
            if key_valid is not None:
                key_mask = key_valid[:, : position + 1]
                key_mask = key_mask | (pad_lengths[:, None] > position)
            stepped = wrapper.token_step(
                ids[:, position], stepped_caches, position, key_mask
            )
    assert stepped is not None
    torch.testing.assert_close(dense.belief, stepped.belief)
    torch.testing.assert_close(dense.logits, stepped.logits)
    for dense_layer, stepped_layer in zip(
        dense_caches, stepped_caches, strict=True
    ):
        for dense_tensor, stepped_tensor in zip(
            dense_layer, stepped_layer, strict=True
        ):
            torch.testing.assert_close(dense_tensor, stepped_tensor)


def test_dense_pope_prefill_matches_incremental_with_left_padding():
    torch.manual_seed(41)
    wrapper = LatentThoughtModel(_pope_model()).eval()
    ids = torch.randint(1, 32, (3, 8))
    lengths = torch.tensor([4, 6, 8])
    key_valid = torch.arange(8)[None] >= (8 - lengths)[:, None]
    ids = ids * key_valid
    _assert_prefill_matches_steps(wrapper, ids, key_valid)


def test_dense_rope_prefill_matches_incremental():
    torch.manual_seed(43)
    wrapper = LatentThoughtModel(FreshLeJEPAGPT(**KWARGS).eval())
    ids = torch.randint(1, 32, (2, 7))
    _assert_prefill_matches_steps(wrapper, ids)


def _wake_attention(backbone) -> None:
    """Give attention a nonzero output projection before asserting on it.

    Every backbone here zero-initialises ``attn.proj.weight`` so each residual
    block starts as an identity. That is correct for training and fatal for a
    decode test: with the projection at zero the attention branch contributes
    exactly nothing, so a paged-versus-dense parity assertion holds no matter
    which keys -- or whose keys -- the row actually read. Waking the
    projection is what gives these tests the ability to fail at all.
    """
    for block in backbone.blocks:
        block.attn.proj.weight.normal_(std=0.5)


def _assert_paged_gqa_matches_masked_dense(backbone) -> None:
    wrapper = LatentThoughtModel(backbone).eval()
    with torch.no_grad():
        backbone.policy_probe.output.weight.normal_(std=0.1)
        _wake_attention(backbone)
        prompts = torch.randint(1, 32, (2, 5))
        lengths = torch.tensor([3, 5])
        prompts[0, :2] = 0
        key_valid = torch.arange(5)[None] >= (5 - lengths)[:, None]
        slots = torch.tensor([[2, 0], [3, 1]])
        paged = wrapper.make_paged_generation_cache(
            4, 12, torch.device("cpu"), dtype=torch.float32, page_size=4
        )
        bank = wrapper.build_prompt_prefix_bank(
            prompts, lengths, dtype=torch.float32
        )
        prefilled = wrapper.admit_prompt_prefixes(
            bank, torch.arange(2), slots, paged
        )

        references = []
        reference_caches = []
        for group in range(2):
            for _ in range(2):
                dense_cache = wrapper.make_generation_cache(
                    1, 12, torch.device("cpu"), dtype=torch.float32
                )
                reference = wrapper.prefill(
                    prompts[group : group + 1],
                    dense_cache,
                    key_valid[group : group + 1],
                )
                references.append(reference)
                reference_caches.append(dense_cache)
        torch.testing.assert_close(
            prefilled.belief,
            torch.cat([reference.belief for reference in references]),
        )
        assert paged.kv_starts.tolist() == [2, 0, 2, 0]

        next_tokens = torch.tensor([3, 5, 7, 11])
        paged_step = wrapper.token_paged_step(
            next_tokens,
            paged,
            slot_ids=slots.flatten(),
            positions=torch.full((4,), 5, dtype=torch.long),
        )
        dense_steps = []
        for row, dense_cache in enumerate(reference_caches):
            group = row // 2
            key_mask = torch.cat(
                (
                    key_valid[group : group + 1],
                    torch.ones(1, 1, dtype=torch.bool),
                ),
                dim=1,
            )
            dense_steps.append(
                wrapper.token_step(
                    next_tokens[row : row + 1],
                    dense_cache,
                    5,
                    key_mask,
                )
            )
        assert torch.isfinite(paged_step.belief).all()
        assert all(torch.isfinite(step.belief).all() for step in dense_steps)
        torch.testing.assert_close(
            paged_step.belief,
            torch.cat([step.belief for step in dense_steps]),
            rtol=2e-4,
            atol=2e-4,
        )
        torch.testing.assert_close(
            paged_step.logits,
            torch.cat([step.logits for step in dense_steps]),
            rtol=2e-4,
            atol=2e-4,
        )


def _flex_decoding_view(mask, mask_mod, rows: int, width: int, block: int):
    """The key set FlexDecoding actually attends under ``mask``.

    Full blocks are taken whole with no mask evaluation, partial blocks are
    filtered by ``mask_mod``, and every block outside both tables is skipped
    without being read. Modelling that literally is what makes this a test of
    the block arithmetic rather than of the mask_mod alone — eager
    ``flex_attention`` reads only ``mask_mod`` and ignores the tables
    entirely, so a table defect is invisible to an eager parity check.

    The kernel walks FULL_KV_IDX using KV_IDX's row stride and length bound
    (``flex_decode.py.jinja``: ``FULL_KV_IDX + sparse_idx_zhm_offset``,
    ``MAX_KV_IDX = size("KV_IDX", -1)``), so a partial table narrower than the
    full one silently misaddresses full blocks. Assert the shared width here
    rather than modelling the corruption.
    """
    assert mask.kv_indices.shape == mask.full_kv_indices.shape
    assert mask.kv_indices.shape[-1] * block >= width
    attended = torch.zeros(rows, width, dtype=torch.bool)
    keys = torch.arange(width)
    for row in range(rows):
        for i in range(int(mask.full_kv_num_blocks[row, 0, 0])):
            block_index = int(mask.full_kv_indices[row, 0, 0, i])
            attended[row, block_index * block : (block_index + 1) * block] = True
        for i in range(int(mask.kv_num_blocks[row, 0, 0])):
            block_index = int(mask.kv_indices[row, 0, 0, i])
            start = block_index * block
            stop = min(start + block, width)
            attended[row, start:stop] |= mask_mod(
                torch.tensor(row), None, None, keys[start:stop]
            )
    return attended


def test_decode_range_mask_block_tables_are_exact():
    """Both partial-block edges, every alignment, no reliance on mask_mod.

    A block table that is merely conservative would still pass an attention
    parity check (mask_mod would clean it up); one that is too NARROW silently
    drops keys, and one that marks a partial block FULL silently attends
    padding. This walks every alignment of both range edges against the
    block size instead.
    """
    device = torch.device("cpu")
    for block in (16, 32, 128):
        for width in (block, 2 * block, 5 * block):
            step = max(1, block // 4)
            starts = sorted(set(list(range(0, width, step)) + [width - 1]))
            for stop in sorted(set(list(range(1, width + 1, step)) + [width])):
                live = torch.tensor([s for s in starts if s < stop])
                builder = DecodeRangeMask(
                    live.numel(), width, device, block_size=block
                )
                attended = _flex_decoding_view(
                    builder.build(live, stop),
                    builder.mask_mod,
                    live.numel(),
                    width,
                    block,
                )
                keys = torch.arange(width)[None]
                expected = (keys >= live[:, None]) & (keys < stop)
                assert torch.equal(attended, expected), (block, width, stop)


def test_paged_block_mask_tables_share_their_block_axis():
    """The paged decode mask rides the same helper, so it needs the same shape.

    ``_assert_paged_gqa_matches_masked_dense`` runs flex eagerly and therefore
    cannot see this: eager attention reads ``mask_mod`` alone. The Triton
    decode kernel that continuous-refill actually runs walks FULL_KV_IDX with
    KV_IDX's stride and bound, so unequal widths misaddress every full block
    past row zero.
    """
    wrapper = LatentThoughtModel(_pope_model()).eval()
    paged = wrapper.make_paged_generation_cache(
        4, 12, torch.device("cpu"), dtype=torch.float32, page_size=4
    )
    paged.kv_starts.copy_(torch.tensor([2, 0, 1, 3]))
    mask = paged.block_mask(
        torch.tensor([0, 2, 3]), torch.tensor([7, 12, 5])
    )
    assert mask.kv_indices.shape == mask.full_kv_indices.shape
    assert mask.kv_indices.shape[-1] == paged.pages_per_lane


def _remap_pages(cache, permutation: torch.Tensor) -> None:
    """Rehome every slot's pages onto ``permutation`` of the physical pool.

    The mapping and its inverse are written together because the two sides of
    decode read different ones: writes and the block table go through
    ``page_table``, and ``mask_mod`` inverts through ``page_home``. A pool
    allocator will maintain exactly this pair.
    """
    table = permutation.view(cache.capacity, cache.pages_per_lane)
    cache.page_table.copy_(table)
    cache.page_home.scatter_(
        0, table.flatten(), torch.arange(table.numel())
    )


def _paged_decoding_view(mask, cache, rows: int) -> torch.Tensor:
    """The physical keys FlexDecoding attends, modelling the block-table walk.

    Same reasoning as ``_flex_decoding_view``: eager attention consults
    ``mask_mod`` alone, so a block table that addressed the wrong PAGE would be
    invisible to a parity check. This one cannot ride that helper because a
    paged row's table spans only the pages it owns, not the whole pool.
    """
    page = cache.page_size
    width = cache.total_pages * page
    assert mask.kv_indices.shape == mask.full_kv_indices.shape
    attended = torch.zeros(rows, width, dtype=torch.bool)
    keys = torch.arange(width)
    for row in range(rows):
        for i in range(int(mask.full_kv_num_blocks[row, 0, 0])):
            block = int(mask.full_kv_indices[row, 0, 0, i])
            attended[row, block * page : (block + 1) * page] = True
        for i in range(int(mask.kv_num_blocks[row, 0, 0])):
            block = int(mask.kv_indices[row, 0, 0, i])
            start, stop = block * page, (block + 1) * page
            attended[row, start:stop] |= mask.mask_mod(
                torch.tensor(row), None, None, keys[start:stop]
            )
    return attended


def test_shuffled_page_table_attends_exactly_its_own_live_span():
    """A row must reach its own pages' live window and nothing else, anywhere.

    Under the identity mapping a page-table bug and a contiguous-offset bug
    are indistinguishable, because logical and physical agree. Permuting the
    pool separates them: the row's pages now sit among other slots' pages, so
    any residual ``slot * pages_per_lane + logical`` arithmetic attends
    another row's KV instead of its own. That is the one failure a pool
    allocator would introduce, and it is silent -- wrong context, finite
    logits, no error.
    """
    wrapper = LatentThoughtModel(_pope_model()).eval()
    paged = wrapper.make_paged_generation_cache(
        4, 12, torch.device("cpu"), dtype=torch.float32, page_size=4
    )
    starts = torch.tensor([2, 0, 1, 3])
    paged.kv_starts.copy_(starts)
    slots = torch.tensor([0, 2, 3])
    lengths = torch.tensor([7, 12, 5])
    page = paged.page_size
    keys = torch.arange(paged.total_pages * page)

    for permutation in (
        torch.arange(paged.capacity * paged.pages_per_lane),
        torch.tensor([7, 0, 11, 3, 5, 1, 9, 2, 6, 10, 4, 8]),
    ):
        _remap_pages(paged, permutation)
        key_owner = paged.page_home[keys // page] // paged.pages_per_lane
        key_logical = (
            paged.page_home[keys // page] % paged.pages_per_lane
        ) * page + keys % page
        expected = (
            (key_owner[None] == slots[:, None])
            & (key_logical[None] >= starts[slots][:, None])
            & (key_logical[None] < lengths[:, None])
        )
        attended = _paged_decoding_view(
            paged.block_mask(slots, lengths), paged, slots.numel()
        )
        assert torch.equal(attended, expected), permutation.tolist()


@torch.no_grad()
def test_shuffled_page_table_decodes_identically_to_the_contiguous_map():
    """End-to-end parity: where a slot's pages live must not change results.

    Complements the table check above by covering the WRITE side -- prefix
    scatter and per-step cache writes both resolve through ``token_addresses``,
    and a write that disagreed with the block table would leave the row
    attending stale pages while every shape still lined up.
    """
    torch.manual_seed(61)
    wrapper = LatentThoughtModel(FreshLeJEPAGPT(**KWARGS).eval()).eval()
    _wake_attention(wrapper.backbone)
    prompts = torch.randint(1, 32, (2, 5))
    lengths = torch.tensor([5, 3])
    slots = torch.tensor([[0, 3], [1, 2]])
    permutation = torch.tensor([9, 2, 14, 7, 0, 5, 12, 3, 15, 10, 1, 6, 11, 4, 13, 8])

    beliefs, logit_rows = [], []
    for remap in (False, True):
        paged = wrapper.make_paged_generation_cache(
            4, 16, torch.device("cpu"), dtype=torch.float32, page_size=4
        )
        if remap:
            _remap_pages(paged, permutation)
        wrapper.prefill_into_paged_slots(prompts, lengths, slots, paged)
        rows = slots.flatten()
        positions = torch.tensor([5, 5, 3, 3])
        for offset in range(3):
            output = wrapper.token_paged_step(
                torch.tensor([11, 13, 17, 19]),
                paged,
                slot_ids=rows,
                positions=positions + offset,
            )
        beliefs.append(output.belief)
        logit_rows.append(output.logits)

    assert not torch.equal(paged.page_table, torch.arange(16).view(4, 4))
    torch.testing.assert_close(beliefs[1], beliefs[0])
    torch.testing.assert_close(logit_rows[1], logit_rows[0])


@torch.no_grad()
def test_padding_rows_read_nothing_and_write_only_to_scratch():
    """A padded row must not touch one byte another row can reach.

    The scheduler pads the decode batch up to a bucket the FlexDecoding
    compiler has specialized. Those rows used to be pointed at a currently-free
    slot, which is a safe write target only while a slot permanently owns a
    lane; a page pool hands that lane's pages to whoever needs them next, and
    the padding write then lands inside a LIVE row's window, where its own
    mask_mod accepts it. This pins the two properties that make padding inert
    regardless of who owns what: an empty KV range, and a write sink no slot
    can address.
    """
    torch.manual_seed(67)
    wrapper = LatentThoughtModel(FreshLeJEPAGPT(**KWARGS).eval()).eval()
    _wake_attention(wrapper.backbone)
    paged = wrapper.make_paged_generation_cache(
        4, 16, torch.device("cpu"), dtype=torch.float32, page_size=4
    )
    wrapper.prefill_into_paged_slots(
        torch.randint(1, 32, (2, 5)),
        torch.tensor([5, 3]),
        torch.tensor([[0, 1], [2, 3]]),
        paged,
    )
    owned = paged.capacity * paged.pages_per_lane * paged.page_size
    before = [
        tuple(tensor[:, :, :owned].clone() for tensor in layer)
        for layer in paged.layers
    ]

    live_rows = torch.tensor([0, 2])
    positions = torch.tensor([5, 3])
    tokens = torch.tensor([11, 23])
    unpadded = wrapper.token_paged_step(
        tokens, paged, slot_ids=live_rows, positions=positions
    )
    for layer, snapshot in zip(paged.layers, before, strict=True):
        for tensor, original in zip(layer, snapshot, strict=True):
            tensor[:, :, :owned].copy_(original)

    # Every padding row deliberately names slot 0, a LIVE slot: if `live` were
    # ignored, the write would corrupt the row under test rather than some
    # unrelated lane, and the comparison below would catch it.
    padded_rows = torch.cat((live_rows, torch.zeros(2, dtype=torch.long)))
    padded = wrapper.token_paged_step(
        torch.cat((tokens, torch.zeros(2, dtype=torch.long))),
        paged,
        slot_ids=padded_rows,
        positions=torch.cat((positions, torch.zeros(2, dtype=torch.long))),
        live=torch.tensor([True, True, False, False]),
    )

    torch.testing.assert_close(padded.belief[:2], unpadded.belief)
    torch.testing.assert_close(padded.logits[:2], unpadded.logits)
    assert torch.isfinite(padded.belief).all()
    paged.validate()

    # ...and they read nothing either. Poisoning slot 0's KV must move the
    # live row that owns it and leave the padding rows -- which name that very
    # slot -- bit-identical. Without this the empty-range half of the contract
    # is only assumed, and a padding row attending real keys costs a full row
    # of attention per step for an output that is thrown away.
    for layer, snapshot in zip(paged.layers, before, strict=True):
        for tensor, original in zip(layer, snapshot, strict=True):
            tensor[:, :, :owned].copy_(original)
    for layer in paged.layers:
        for tensor in layer:
            tensor[:, :, : paged.rounded_length] += 100.0
    poisoned = wrapper.token_paged_step(
        torch.cat((tokens, torch.zeros(2, dtype=torch.long))),
        paged,
        slot_ids=padded_rows,
        positions=torch.cat((positions, torch.zeros(2, dtype=torch.long))),
        live=torch.tensor([True, True, False, False]),
    )
    torch.testing.assert_close(poisoned.belief[2:], padded.belief[2:])
    assert not torch.allclose(poisoned.belief[0], padded.belief[0])


def test_decode_range_mask_serves_fewer_rows_than_it_holds():
    """Compaction shrinks the batch; the builder must not be rebuilt for it."""
    device = torch.device("cpu")
    builder = DecodeRangeMask(8, 64, device, block_size=16)
    wide = builder.build(torch.arange(8), 40)
    narrow = builder.build(torch.arange(3), 40)
    assert wide.kv_num_blocks.shape[0] == 8
    assert narrow.kv_num_blocks.shape[0] == 3
    assert torch.equal(
        _flex_decoding_view(narrow, builder.mask_mod, 3, 64, 16),
        _flex_decoding_view(wide, builder.mask_mod, 8, 64, 16)[:3],
    )


def test_paged_rope_gqa_matches_masked_dense_decode():
    torch.manual_seed(47)
    _assert_paged_gqa_matches_masked_dense(
        FreshLeJEPAGPT(**KWARGS).eval()
    )


def test_paged_pope_gqa_matches_masked_dense_decode():
    torch.manual_seed(53)
    backbone = _pope_model()
    _assert_paged_gqa_matches_masked_dense(backbone)
    # PoPE stores its complex key once as [real, imag], not as a per-step cat.
    cache = LatentThoughtModel(backbone).make_paged_generation_cache(
        2, 9, torch.device("cpu"), dtype=torch.float32, page_size=4
    )
    assert cache.layers[0][0].shape[-1] == 2 * backbone.blocks[0].attn.head_dim


@torch.no_grad()
def test_paged_refill_hides_a_stale_longer_suffix():
    torch.manual_seed(59)
    wrapper = LatentThoughtModel(FreshLeJEPAGPT(**KWARGS).eval()).eval()
    wrapper.backbone.policy_probe.output.weight.normal_(std=0.1)
    _wake_attention(wrapper.backbone)
    paged = wrapper.make_paged_generation_cache(
        1, 12, torch.device("cpu"), dtype=torch.float32, page_size=4
    )
    long_prompt = torch.randint(1, 32, (1, 5))
    short_prompt = torch.tensor([[0, 0, 0, 19, 23]])
    bank = wrapper.build_prompt_prefix_bank(
        torch.cat((long_prompt, short_prompt)),
        torch.tensor([5, 2]),
        dtype=torch.float32,
    )
    wrapper.admit_prompt_prefixes(
        bank, torch.tensor([0]), torch.tensor([[0]]), paged
    )
    for position in (5, 6, 7):
        wrapper.token_paged_step(
            torch.tensor([position]),
            paged,
            slot_ids=torch.tensor([0]),
            positions=torch.tensor([position]),
        )

    refilled = wrapper.admit_prompt_prefixes(
        bank, torch.tensor([1]), torch.tensor([[0]]), paged
    )
    paged_next = wrapper.token_paged_step(
        torch.tensor([29]),
        paged,
        slot_ids=torch.tensor([0]),
        positions=torch.tensor([5]),
    )

    dense_cache = wrapper.make_generation_cache(
        1, 12, torch.device("cpu"), dtype=torch.float32
    )
    key_valid = torch.tensor([[False, False, False, True, True]])
    dense_prefill = wrapper.prefill(short_prompt, dense_cache, key_valid)
    dense_next = wrapper.token_step(
        torch.tensor([29]),
        dense_cache,
        5,
        torch.tensor([[False, False, False, True, True, True]]),
    )
    torch.testing.assert_close(refilled.belief, dense_prefill.belief)
    torch.testing.assert_close(
        paged_next.belief, dense_next.belief, rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        paged_next.logits, dense_next.logits, rtol=2e-4, atol=2e-4
    )


def test_cached_generation_matches_full_forward_beyond_pretraining_context():
    torch.manual_seed(5)
    length = 1050  # crosses the 1024-token pretraining context
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    ids = torch.randint(0, 32, (1, length))
    with torch.no_grad():
        token_latents = backbone.embed_tokens(ids)
        beliefs = backbone.temporal_belief_from_token_latent(token_latents)
        full = backbone.logits_from_features(
            wrapper.renderer_features(token_latents, beliefs)
        )
        caches = backbone.make_generation_cache(1, length, torch.device("cpu"))
        stepped = []
        for position in range(length):
            output = wrapper.token_step(ids[:, position], caches, position)
            caches = output.caches
            stepped.append(output.logits)
    stepped = torch.stack(stepped, dim=1)
    torch.testing.assert_close(stepped, full, rtol=2e-4, atol=2e-4)


def test_renderer_logits_do_not_depend_on_prediction_projector():
    torch.manual_seed(7)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    ids = torch.randint(0, 32, (2, 7))
    with torch.no_grad():
        backbone.policy_probe.output.weight.normal_(std=0.1)
        input_latent = backbone.embed_tokens(ids)
        belief = backbone.temporal_belief_from_token_latent(input_latent)
        predicted_before = backbone.prediction_latent(belief)
        logits_before = wrapper.policy_logits(ids)
        for parameter in backbone.prediction_projector.parameters():
            parameter.add_(torch.randn_like(parameter))
        predicted_after = backbone.prediction_latent(belief)
        logits_after = wrapper.policy_logits(ids)
    assert not torch.equal(predicted_before, predicted_after)
    torch.testing.assert_close(logits_after, logits_before)


def test_renderer_checkpoint_schema_rejects_old_semantics():
    current = {
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
        "thought_input_schema": THOUGHT_INPUT_SCHEMA,
        "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
    }
    validate_renderer_checkpoint(current, "current.pt")
    # No migrations: untagged payloads and every pre-hidden-carry schema tag
    # are refused outright.
    with pytest.raises(ValueError, match="renderer"):
        validate_renderer_checkpoint({}, "old.pt")
    with pytest.raises(ValueError, match="renderer"):
        validate_renderer_checkpoint(
            {"renderer_features_schema": "input_latent+predicted/v1"}, "old.pt"
        )
    with pytest.raises(ValueError, match="rollout"):
        validate_renderer_checkpoint(
            {"renderer_features_schema": RENDERER_FEATURES_SCHEMA},
            "old-policy.pt",
        )
    with pytest.raises(ValueError, match="rollout"):
        validate_renderer_checkpoint(
            {
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "rollout_policy_schema": (
                    "forced_initial_thought_one_way_stop/v2"
                ),
            },
            "stochastic-policy.pt",
        )
    with pytest.raises(ValueError, match="thought"):
        validate_renderer_checkpoint(
            {
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
                "thought_input_schema": "fresh_zero_affine/v5",
            },
            "old-adapter.pt",
        )
    # A pinned-mode checkpoint validates only under its own mode's schema.
    pinned = {
        **current,
        "rollout_policy_schema": PINNED_EMIT_ROLLOUT_POLICY_SCHEMAS["cot"],
    }
    validate_renderer_checkpoint(
        pinned,
        "cot.pt",
        expected_rollout_policy_schema=(
            PINNED_EMIT_ROLLOUT_POLICY_SCHEMAS["cot"]
        ),
    )
    with pytest.raises(ValueError, match="rollout"):
        validate_renderer_checkpoint(pinned, "cot-as-latent.pt")


def test_combiner_init_kwargs_recover_saved_geometry():
    payload = {
        "args": {
            "combined_mlp_hidden": 1024,
            "combined_mlp_blocks": 2,
            "thought_sigma": 2.0,
            "init_stop_thinking_probability": 0.8,
        }
    }
    assert combiner_init_kwargs_from_checkpoint(payload) == {
        "mlp_hidden": 1024,
        "num_blocks": 2,
        "thought_sigma": 2.0,
        "init_stop_thinking_probability": 0.8,
    }
    with pytest.raises(ValueError, match="stochastic policy arguments"):
        combiner_init_kwargs_from_checkpoint({"args": {}})


def test_fresh_combiner_is_bitwise_identity_and_flag_selects_exactly():
    torch.manual_seed(11)
    combiner = CombinedEmbedding(32, mlp_hidden=64, num_blocks=1)
    base = torch.randn(3, 7, 32)
    hidden = torch.randn(3, 7, 32)
    flag = torch.zeros(3, 7, dtype=torch.bool)
    flag[:, 4:] = True
    with torch.no_grad():
        # Fresh init: zero carry matrix, zero type bias, zero MLP
        # projections — the combined input IS the token embedding, bit for
        # bit.
        assert torch.equal(combiner(base, hidden, flag), base)
        assert torch.equal(combiner(base, hidden), base)
        # A live combiner changes exactly the flagged positions and leaves
        # unflagged positions bitwise on the plain token path.
        combiner.carry.weight.normal_(std=0.05)
        combiner.type_bias.normal_(std=0.1)
        mixed = combiner(base, hidden, flag)
        assert torch.equal(mixed[~flag], base[~flag])
        assert not torch.equal(mixed[flag], base[flag])
        # And flagged positions are bitwise the all-injected decode path.
        assert torch.equal(mixed[flag], combiner(base, hidden)[flag])


def test_combiner_zero_blocks_is_the_pure_residual_ablation():
    combiner = CombinedEmbedding(16, num_blocks=0)
    with torch.no_grad():
        combiner.carry.weight.normal_(std=0.05)
    base = torch.randn(2, 5, 16)
    hidden = torch.randn(2, 5, 16)
    with torch.no_grad():
        expected = base + (
            torch.nn.functional.linear(hidden, combiner.carry.weight)
            + combiner.type_bias
        )
        torch.testing.assert_close(combiner(base, hidden), expected)
    with pytest.raises(ValueError, match="non-negative"):
        CombinedEmbedding(16, num_blocks=-1)


def test_teacher_forced_logits_equal_pretrained_backbone_at_init():
    """The init-identity gate: a fresh wrapper IS the pretrained model."""
    torch.manual_seed(3)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone, mlp_hidden=64).eval()
    ids = torch.randint(0, 32, (2, 9))
    with torch.no_grad():
        token_latents = backbone.embed_tokens(ids)
        beliefs = backbone.temporal_belief_from_token_latent(token_latents)
        expected = backbone.logits_from_features(
            wrapper.renderer_features(token_latents, beliefs)
        )
        actual = wrapper.policy_logits(ids)
    assert torch.equal(actual, expected)


def test_gradient_reaches_combiner_and_trunk_but_not_stored_hiddens():
    torch.manual_seed(17)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone, mlp_hidden=64)
    with torch.no_grad():
        backbone.policy_probe.output.weight.normal_(std=0.05)
        wrapper.combiner.carry.weight.normal_(std=0.02)
    base = backbone.embed_tokens(torch.randint(0, 32, (2, 6)))
    hidden = torch.randn(2, 6, 32, requires_grad=True)
    flag = torch.zeros(2, 6, dtype=torch.bool)
    flag[:, 3:] = True
    # Replay treats the stored hidden as a constant: detach before use, the
    # same as the trainer's fp32 storage.
    inputs = wrapper.combiner(base, hidden.detach(), flag)
    beliefs = backbone.temporal_belief_from_token_latent(inputs)
    loss = backbone.logits_from_features(
        wrapper.renderer_features(inputs, beliefs)
    ).float().square().mean()
    loss.backward()
    assert hidden.grad is None
    assert wrapper.combiner.carry.weight.grad is not None
    assert float(wrapper.combiner.carry.weight.grad.abs().sum()) > 0.0
    assert wrapper.combiner.type_bias.grad is not None
    assert float(backbone.blocks[0].attn.proj.weight.grad.abs().sum()) > 0.0


def test_chunked_teacher_forced_ce_matches_one_shot_cross_entropy():
    torch.manual_seed(29)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    ids = torch.randint(0, 32, (3, 10))
    targets = torch.randint(0, 32, (3, 10))
    with torch.no_grad():
        one_shot = torch.nn.functional.cross_entropy(
            wrapper.policy_logits(ids).float().flatten(0, 1), targets.flatten()
        )
        full = wrapper(ids, targets)
        # Force several uneven chunks; the token-weighted sum must reduce to
        # the identical mean.
        wrapper.BPB_EVAL_CHUNK_TOKENS = 7
        chunked = wrapper(ids, targets)
    torch.testing.assert_close(full, one_shot)
    torch.testing.assert_close(chunked, one_shot)


def test_new_parameters_exclude_backbone_and_strict_load_round_trips():
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone)
    backbone_ids = {id(parameter) for parameter in backbone.parameters()}
    new = list(wrapper.new_parameters())
    assert new
    assert all(id(parameter) not in backbone_ids for parameter in new)
    wrapper.load_backbone_checkpoint(
        {key: value.clone() for key, value in backbone.state_dict().items()}
    )


def test_vector_sigma_scales_components_by_runtime_width():
    sigma = 2.0
    wrapper = LatentThoughtModel(_pope_model(), thought_sigma=sigma)
    assert wrapper.transition.component_std == pytest.approx(
        sigma / KWARGS["model_dim"] ** 0.5
    )
    belief = torch.randn(2, KWARGS["model_dim"])
    assert torch.equal(wrapper.transition.predict_mean(belief), belief)
    mean = torch.zeros(1, KWARGS["model_dim"])
    log_sigma = wrapper.transition.predict_log_sigma(mean)
    deterministic_action = mean + log_sigma.exp()
    noise = deterministic_action - mean
    assert float(noise.square().sum()) == pytest.approx(sigma**2)
    factors = wrapper.transition.per_dim_log_prob(
        deterministic_action, mean, log_sigma
    )
    assert factors.shape == mean.shape
    assert torch.equal(factors.sum(-1), wrapper.transition.log_prob(
        deterministic_action, mean, log_sigma
    ))

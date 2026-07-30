from __future__ import annotations

import math

import pytest
import torch

from fresh_lejepa_train import FreshLeJEPAGPT
from fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
from postraining.latent_thought import (
    AffineThoughtAdapter,
    DecodeRangeMask,
    EMIT,
    IDENTITY_AFFINE_THOUGHT_INPUT_SCHEMA,
    THINK,
    GaussianTransitionHead,
    LatentThoughtModel,
    migrate_legacy_wrapper_checkpoint,
    migrate_scalar_log_sigma_state,
    RENDERER_FEATURES_SCHEMA,
    ROLLOUT_POLICY_SCHEMA,
    THOUGHT_ACTION_TRANSFORM_SCHEMAS,
    THOUGHT_DISTRIBUTION_SCHEMA,
    THOUGHT_INPUT_SCHEMA,
    THOUGHT_MEAN_SCHEMA,
    THOUGHT_LOG_SIGMA_MAX,
    THOUGHT_LOG_SIGMA_MIN,
    StopThinkingGate,
    ThoughtAdapter,
    validate_renderer_checkpoint,
    wrapper_init_kwargs_from_checkpoint,
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
    torch.testing.assert_close(dense.predicted, stepped.predicted)
    torch.testing.assert_close(
        dense.thought_log_sigma, stepped.thought_log_sigma
    )
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


def test_dense_step_thought_mean_gets_no_renderer_gradient():
    torch.manual_seed(9)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    with torch.no_grad():
        backbone.policy_probe.output.weight.normal_(std=0.1)
    ids = torch.randint(0, 32, (2,))
    caches = wrapper.make_generation_cache(2, 1, torch.device("cpu"))
    projected = []
    handle = wrapper.transition.mean_head.register_forward_pre_hook(
        lambda _module, inputs: projected.append(tuple(inputs[0].shape))
    )
    try:
        output = wrapper.token_step(ids, caches, 0)
    finally:
        handle.remove()
    output.logits.float().square().mean().backward()
    assert projected == [(2, 1, 32)]
    assert all(
        parameter.grad is None
        for parameter in wrapper.transition.mean_head.parameters()
    )
    assert float(backbone.blocks[0].attn.proj.weight.grad.abs().sum()) > 0.0


def test_renderer_checkpoint_schema_rejects_old_semantics():
    validate_renderer_checkpoint(
        {
            "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
            "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
            "thought_input_schema": THOUGHT_INPUT_SCHEMA,
            "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
            "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
        },
        "current.pt",
    )
    # Initial gain is fully represented by the learned scalar, so the v2
    # origin label remains functionally resumable for the live policy.
    validate_renderer_checkpoint(
        {
            "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
            "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
            "thought_input_schema": THOUGHT_INPUT_SCHEMA,
            "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
            "thought_mean_schema": (
                "fresh_linear_learned_output_gain_0.01_zero_bias/v2"
            ),
        },
        "live-v2.pt",
    )
    with pytest.raises(ValueError, match="Old or untagged VAPO checkpoints"):
        validate_renderer_checkpoint({}, "old.pt")
    with pytest.raises(ValueError, match="predicted/v1"):
        validate_renderer_checkpoint(
            {"renderer_features_schema": "input_latent+predicted/v1"}, "old.pt"
        )
    with pytest.raises(
        ValueError, match="reasoning mode or forced-initial assignment"
    ):
        validate_renderer_checkpoint(
            {"renderer_features_schema": RENDERER_FEATURES_SCHEMA},
            "old-policy.pt",
        )
    with pytest.raises(ValueError, match="different deployed thought adapter"):
        validate_renderer_checkpoint(
            {
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
            },
            "old-adapter.pt",
        )
    with pytest.raises(ValueError, match="unbounded log-sigma"):
        validate_renderer_checkpoint(
            {
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
                "thought_input_schema": THOUGHT_INPUT_SCHEMA,
            },
            "old-sigma.pt",
        )
    validate_renderer_checkpoint(
        {
            "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
            "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
            "thought_input_schema": THOUGHT_INPUT_SCHEMA,
            "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
        },
        "critic-warmup.pt",
        allow_transition_reset=True,
    )
    affine_payload = {
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
        "thought_input_schema": "fresh_zero_affine/v5",
        "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
        "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
    }
    validate_renderer_checkpoint(
        affine_payload,
        "affine-v24.pt",
        expected_thought_input_schema=IDENTITY_AFFINE_THOUGHT_INPUT_SCHEMA,
    )
    with pytest.raises(ValueError, match="different deployed thought adapter"):
        validate_renderer_checkpoint(affine_payload, "nonlinear-v26.pt")
    tanh_payload = {
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
        "thought_input_schema": THOUGHT_INPUT_SCHEMA,
        "thought_action_transform_schema": (
            THOUGHT_ACTION_TRANSFORM_SCHEMAS["tanh"]
        ),
        "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
        "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
    }
    validate_renderer_checkpoint(
        tanh_payload,
        "tanh.pt",
        expected_thought_action_transform_schema=(
            THOUGHT_ACTION_TRANSFORM_SCHEMAS["tanh"]
        ),
    )
    with pytest.raises(ValueError, match="different action"):
        validate_renderer_checkpoint(tanh_payload, "tanh-as-raw.pt")


def test_wrapper_checkpoint_kwargs_preserve_legacy_policy_semantics():
    assert wrapper_init_kwargs_from_checkpoint({"args": {}}) == {
        "thought_adapter": "identity_affine",
        "sigma_state_init": "constant",
        "thought_action_transform": "identity",
    }
    assert wrapper_init_kwargs_from_checkpoint(
        {
            "args": {
                "thought_adapter": "orthogonal_silu",
                "thought_sigma_state_init": "orthogonal",
                "thought_action_transform": "tanh",
            }
        }
    ) == {
        "thought_adapter": "orthogonal_silu",
        "sigma_state_init": "orthogonal",
        "thought_action_transform": "tanh",
    }


def test_tanh_thought_input_transforms_once_before_the_adapter():
    wrapper = LatentThoughtModel(
        _pope_model(), thought_action_transform="tanh"
    ).eval()
    raw = torch.linspace(-3.0, 3.0, 64).view(2, 32)
    expected = wrapper.adapter(raw.tanh())[:, None]
    torch.testing.assert_close(wrapper.thought_input(raw), expected)


def test_gate_zero_init_is_exactly_uniform():
    gate = StopThinkingGate(16)
    belief = torch.randn(4, 16)
    assert torch.all(gate.stop_logit(belief) == 0)
    log_prob = gate.log_prob(torch.tensor([THINK, EMIT, THINK, EMIT]), belief)
    torch.testing.assert_close(log_prob, torch.full((4,), math.log(0.5)))
    torch.testing.assert_close(gate.entropy(belief), torch.full((4,), math.log(2.0)))


def test_gate_sample_log_prob_recomputes_identically():
    torch.manual_seed(11)
    gate = StopThinkingGate(16)
    with torch.no_grad():
        gate.head.weight.normal_(std=0.5)
        gate.head.bias.normal_()
    belief = torch.randn(64, 16)
    generator = torch.Generator().manual_seed(7)
    action, log_prob = gate.sample(belief, generator=generator)
    torch.testing.assert_close(gate.log_prob(action, belief), log_prob)
    assert set(action.unique().tolist()) <= {THINK, EMIT}


def test_transition_log_prob_matches_torch_distributions():
    torch.manual_seed(13)
    head = GaussianTransitionHead(8, log_sigma=-0.5)
    mean = torch.randn(5, 8)
    belief = torch.randn(5, 8)
    log_sigma = head.predict_log_sigma(belief)
    generator = torch.Generator().manual_seed(21)
    sample, log_prob = head.sample(mean, log_sigma, generator=generator)
    torch.testing.assert_close(head.log_prob(sample, mean, log_sigma), log_prob)
    reference = torch.distributions.Normal(mean, log_sigma.exp())
    torch.testing.assert_close(log_prob, reference.log_prob(sample).sum(-1))
    torch.testing.assert_close(
        head.per_dim_log_prob(sample, mean, log_sigma), reference.log_prob(sample)
    )


def test_transition_mean_head_starts_small_orthogonal_and_zero_bias():
    torch.manual_seed(12)
    head = GaussianTransitionHead(8)
    gram = head.mean_head.weight @ head.mean_head.weight.T
    expected = torch.eye(8)
    torch.testing.assert_close(gram, expected, rtol=1e-5, atol=2e-7)
    assert torch.count_nonzero(head.mean_head.bias) == 0
    assert head.mean_head.output_gain.item() == pytest.approx(
        head.MEAN_INIT_GAIN
    )
    assert head.MEAN_INIT_GAIN == pytest.approx(0.1)
    torch.testing.assert_close(
        head.log_sigma_head.bias.detach(),
        torch.full((8,), head.raw_from_log_sigma(-2.0)),
    )
    log_sigma = head.predict_log_sigma(torch.randn(3, 8))
    assert not torch.allclose(log_sigma[0], log_sigma[1])
    assert float(
        (log_sigma.detach() + 2.0).square().mean().sqrt()
    ) < 0.1
    belief = torch.randn(5, 8)
    belief = torch.nn.functional.rms_norm(belief, (8,))
    mean = head.predict_mean(belief)
    torch.testing.assert_close(
        mean.norm(dim=-1),
        belief.norm(dim=-1) * head.MEAN_INIT_GAIN,
        rtol=1e-5,
        atol=1e-6,
    )


def test_transition_sigma_head_starts_orthogonal_and_mildly_state_dependent():
    head = GaussianTransitionHead(8, log_sigma=-0.5)
    parameters = list(head.log_sigma_head.parameters())
    assert len(parameters) == 3
    gram = head.log_sigma_head.weight @ head.log_sigma_head.weight.T
    torch.testing.assert_close(
        gram, torch.eye(8), rtol=1e-5, atol=2e-7
    )
    assert head.log_sigma_head.residual_gain.item() == pytest.approx(0.01)
    expected_raw_bias = head.raw_from_log_sigma(-0.5)
    torch.testing.assert_close(
        head.log_sigma_head.bias.detach(), torch.full((8,), expected_raw_bias)
    )
    beliefs = torch.nn.functional.rms_norm(torch.randn(128, 8), (8,))
    log_sigma = head.predict_log_sigma(beliefs)
    assert not torch.allclose(log_sigma[0], log_sigma[1])
    assert float(
        (log_sigma.detach() + 0.5).square().mean().sqrt()
    ) < 0.05

    weight = head.log_sigma_head.weight.detach().clone()
    head.set_noise_level(-2.5)
    torch.testing.assert_close(head.log_sigma_head.weight, weight)
    assert float(
        (head.predict_log_sigma(beliefs).detach() + 2.5)
        .square()
        .mean()
        .sqrt()
    ) < 0.05


def test_transition_sigma_head_has_first_step_gradients_for_all_parameters():
    head = GaussianTransitionHead(8, log_sigma=-2.0)
    beliefs = torch.nn.functional.rms_norm(torch.randn(32, 8), (8,))

    head.predict_log_sigma(beliefs).square().mean().backward()

    for name, parameter in head.log_sigma_head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name


def test_transition_sigma_constant_control_is_exact():
    head = GaussianTransitionHead(
        8, log_sigma=-0.5, sigma_state_init="constant"
    )
    assert torch.count_nonzero(head.log_sigma_head.weight) == 0
    beliefs = torch.randn(6, 8)
    torch.testing.assert_close(
        head.predict_log_sigma(beliefs), torch.full((6, 8), -0.5)
    )


def test_transition_sigma_orthogonal_reset_preserves_global_rng_stream():
    torch.manual_seed(123)
    head = GaussianTransitionHead(
        8, log_sigma=-2.0, sigma_state_init="constant"
    )
    before = torch.get_rng_state()
    head.reset_noise(-2.0, sigma_state_init="orthogonal")
    after = torch.get_rng_state()
    torch.testing.assert_close(after, before)


def test_transition_sigma_preserves_small_residuals_under_bf16_autocast():
    head = GaussianTransitionHead(8, log_sigma=-2.0)
    belief = torch.ones(2, 8)
    with torch.no_grad():
        head.log_sigma_head.weight.zero_()
        head.log_sigma_head.weight[0, 0] = 1e-3
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        log_sigma = head.predict_log_sigma(belief)
    assert log_sigma.dtype == torch.float32
    # A bf16 affine output centered at -2 would round this residual away.
    assert float(log_sigma[0, 0].detach()) != -2.0
    expected = head.bound_raw_log_sigma(torch.tensor(
        head.raw_from_log_sigma(-2.0)
        + head.log_sigma_head.residual_gain.item() * 1e-3
    ))
    assert float(log_sigma[0, 0].detach()) == pytest.approx(
        float(expected), abs=2e-5
    )


def test_transition_sigma_is_smoothly_bounded_and_finite():
    head = GaussianTransitionHead(8, log_sigma=-2.0)
    beliefs = torch.ones(2, 8)
    with torch.no_grad():
        head.log_sigma_head.weight.fill_(1e6)
    upper = head.predict_log_sigma(beliefs)
    lower = head.predict_log_sigma(-beliefs)
    assert torch.isfinite(upper).all()
    assert torch.isfinite(lower).all()
    assert torch.all(upper <= THOUGHT_LOG_SIGMA_MAX)
    assert torch.all(lower >= THOUGHT_LOG_SIGMA_MIN)
    torch.testing.assert_close(upper, torch.full_like(upper, THOUGHT_LOG_SIGMA_MAX))
    torch.testing.assert_close(lower, torch.full_like(lower, THOUGHT_LOG_SIGMA_MIN))


def test_transition_sigma_init_must_be_strictly_inside_bounds():
    with pytest.raises(ValueError, match="strictly inside"):
        GaussianTransitionHead(8, log_sigma=THOUGHT_LOG_SIGMA_MIN)
    with pytest.raises(ValueError, match="strictly inside"):
        GaussianTransitionHead(8, log_sigma=THOUGHT_LOG_SIGMA_MAX)


def test_scalar_log_sigma_state_migrates_to_the_head():
    head = GaussianTransitionHead(8, log_sigma=-0.5)
    legacy = {"transition.log_sigma": torch.tensor(-1.5)}
    assert migrate_scalar_log_sigma_state(legacy, head)
    assert "transition.log_sigma" not in legacy
    torch.testing.assert_close(
        legacy["transition.log_sigma_head.bias"],
        torch.full((8,), head.raw_from_log_sigma(-1.5)),
    )
    assert torch.all(legacy["transition.log_sigma_head.weight"] == 0.0)
    torch.testing.assert_close(
        legacy["transition.log_sigma_head.residual_gain"],
        head.log_sigma_head.residual_gain,
    )
    # The migrated policy is the retired scalar policy exactly.
    head.log_sigma_head.load_state_dict(
        {
            "weight": legacy["transition.log_sigma_head.weight"],
            "bias": legacy["transition.log_sigma_head.bias"],
            "residual_gain": legacy[
                "transition.log_sigma_head.residual_gain"
            ],
        }
    )
    torch.testing.assert_close(
        head.predict_log_sigma(torch.randn(4, 8)), torch.full((4, 8), -1.5)
    )
    # New-format states pass through untouched.
    assert not migrate_scalar_log_sigma_state(legacy, head)


def test_fresh_mean_requires_explicit_legacy_branch_migration():
    torch.manual_seed(14)
    wrapper = LatentThoughtModel(_pope_model())
    state = {
        key: value.detach().clone()
        for key, value in wrapper.state_dict().items()
        if not key.startswith("transition.mean_head.")
    }
    payload = {"model": state}
    assert migrate_legacy_wrapper_checkpoint(payload, wrapper) == (
        False,
        False,
        False,
    )
    assert "transition.mean_head.weight" not in state
    _, migrated, _ = migrate_legacy_wrapper_checkpoint(
        payload, wrapper, initialize_fresh_mean=True
    )
    assert migrated
    assert payload["thought_mean_schema"] == THOUGHT_MEAN_SCHEMA
    torch.testing.assert_close(
        state["transition.mean_head.weight"],
        wrapper.transition.mean_head.weight,
    )
    torch.testing.assert_close(
        state["transition.mean_head.bias"],
        wrapper.transition.mean_head.bias,
    )
    torch.testing.assert_close(
        state["transition.mean_head.output_gain"],
        wrapper.transition.mean_head.output_gain,
    )


def test_explicit_actor_restart_replaces_mean_and_sigma_heads():
    wrapper = LatentThoughtModel(_pope_model())
    wrapper.transition.mean_head.reset_output_gain(0.1)
    state = {
        key: value.detach().clone()
        for key, value in wrapper.state_dict().items()
    }
    state["transition.mean_head.weight"].zero_()
    state["transition.mean_head.bias"].fill_(2.0)
    state["transition.mean_head.output_gain"].fill_(0.01)
    state["transition.log_sigma_head.weight"].zero_()
    state["transition.log_sigma_head.bias"].fill_(2.0)
    state["transition.log_sigma_head.residual_gain"].fill_(0.5)
    payload = {"model": state}

    _, migrated, _ = migrate_legacy_wrapper_checkpoint(
        payload, wrapper, initialize_fresh_mean=True
    )

    assert migrated
    torch.testing.assert_close(
        state["transition.mean_head.weight"],
        wrapper.transition.mean_head.weight,
    )
    torch.testing.assert_close(
        state["transition.mean_head.bias"],
        wrapper.transition.mean_head.bias,
    )
    assert state["transition.mean_head.output_gain"].item() == pytest.approx(0.1)
    torch.testing.assert_close(
        state["transition.log_sigma_head.weight"],
        wrapper.transition.log_sigma_head.weight,
    )
    torch.testing.assert_close(
        state["transition.log_sigma_head.bias"],
        wrapper.transition.log_sigma_head.bias,
    )
    torch.testing.assert_close(
        state["transition.log_sigma_head.residual_gain"],
        wrapper.transition.log_sigma_head.residual_gain,
    )


def test_explicit_actor_restart_replaces_the_complete_gate():
    wrapper = LatentThoughtModel(_pope_model())
    with torch.no_grad():
        wrapper.gate.head.weight.zero_()
        wrapper.gate.head.bias.fill_(1.25)
    expected_weight = wrapper.gate.head.weight.detach().clone()
    expected_bias = wrapper.gate.head.bias.detach().clone()
    state = {
        key: value.detach().clone()
        for key, value in wrapper.state_dict().items()
    }
    state["gate.head.weight"].fill_(0.5)
    state["gate.head.bias"].fill_(-2.0)
    payload = {"model": state}

    migrate_legacy_wrapper_checkpoint(
        payload, wrapper, initialize_fresh_gate=True
    )
    wrapper.load_state_dict(state, strict=True)

    torch.testing.assert_close(wrapper.gate.head.weight, expected_weight)
    torch.testing.assert_close(wrapper.gate.head.bias, expected_bias)
    belief = torch.randn(4, expected_weight.shape[1])
    expected_stop_probability = torch.sigmoid(expected_bias).expand(4)
    torch.testing.assert_close(
        wrapper.gate.stop_logit(belief).sigmoid(), expected_stop_probability
    )


def test_fresh_adapter_explicitly_replaces_critic_warm_identity_state():
    wrapper = LatentThoughtModel(_pope_model())
    state = {
        key: value.detach().clone()
        for key, value in wrapper.state_dict().items()
    }
    state["adapter.projection.weight"] = torch.eye(32)
    state["adapter.projection.bias"] = torch.full((32,), 0.5)
    payload = {
        "model": state,
        "thought_input_schema": "identity_init_affine/v1",
    }

    _, _, reset = migrate_legacy_wrapper_checkpoint(
        payload, wrapper, initialize_fresh_adapter=True
    )

    assert reset
    assert payload["thought_input_schema"] == THOUGHT_INPUT_SCHEMA
    torch.testing.assert_close(
        state["adapter.projection.weight"],
        wrapper.adapter.projection.weight,
    )
    torch.testing.assert_close(
        state["adapter.projection.bias"],
        wrapper.adapter.projection.bias,
    )
    assert "adapter.interpolation_strength" not in state


def test_thought_policy_gradient_flows_through_the_mean():
    # v2 (full-model RL): the policy gradient must reach the prediction
    # path — per_dim_log_prob differentiates through the passed mean.
    torch.manual_seed(15)
    head = GaussianTransitionHead(8)
    mean = torch.randn(5, 8, requires_grad=True)
    sample = (mean + 0.3).detach()
    log_sigma = torch.full((5, 8), -0.5)
    head.per_dim_log_prob(sample, mean, log_sigma).sum().backward()
    assert mean.grad is not None
    # d/dmean of -0.5*((s-m)/sigma)^2 is (s-m)/sigma^2, positive here.
    assert torch.all(mean.grad > 0)


def test_thought_policy_gradient_reaches_the_sigma_head():
    # State-dependent sigma: the log-prob path must differentiate through
    # predict_log_sigma so the joint-action PPO objective can move the head
    # — including its zero-init weight, via the belief.
    torch.manual_seed(16)
    head = GaussianTransitionHead(8)
    belief = torch.randn(5, 8)
    mean = torch.randn(5, 8)
    sample = mean + 0.3 * torch.randn(5, 8)
    log_sigma = head.predict_log_sigma(belief)
    head.log_prob(sample, mean, log_sigma).sum().backward()
    assert head.log_sigma_head.bias.grad is not None
    # d/dlog_sigma of the log-density is ((s-m)/sigma)^2 - 1 per dim,
    # generically nonzero for off-mean samples.
    assert head.log_sigma_head.bias.grad.abs().sum().item() > 0.0
    assert head.log_sigma_head.weight.grad is not None
    assert head.log_sigma_head.weight.grad.abs().sum().item() > 0.0


def test_actor_adapter_starts_orthogonal_nonlinear_and_critic_is_affine():
    torch.manual_seed(19)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone)
    critic_adapter = AffineThoughtAdapter(32)
    thought = torch.randn(2, 32)
    injected = wrapper.thought_input(thought)
    assert injected.shape == (2, 1, 32)
    actor_gram = (
        wrapper.adapter.projection.weight
        @ wrapper.adapter.projection.weight.T
    )
    critic_gram = (
        critic_adapter.projection.weight
        @ critic_adapter.projection.weight.T
    )
    torch.testing.assert_close(actor_gram, torch.eye(32), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(critic_gram, torch.eye(32), rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(wrapper.adapter.projection.bias) == 0
    assert torch.count_nonzero(critic_adapter.projection.bias) == 0
    assert not torch.allclose(injected.squeeze(1), thought)
    torch.testing.assert_close(critic_adapter(thought).norm(dim=-1), thought.norm(dim=-1))


def test_nonlinear_adapter_learns_on_its_first_backward_pass():
    adapter = ThoughtAdapter(4)
    thought = torch.randn(3, 4, requires_grad=True)

    adapter(thought).sum().backward()

    assert float(adapter.projection.weight.grad.abs().sum()) > 0.0
    assert float(adapter.projection.bias.grad.abs().sum()) > 0.0
    assert float(thought.grad.abs().sum()) > 0.0


def test_identity_affine_adapter_remains_an_exact_control():
    adapter = ThoughtAdapter(4, kind="identity_affine")
    thought = torch.randn(3, 4)
    torch.testing.assert_close(adapter(thought), thought)


def test_adapter_bias_is_a_shared_thought_type_offset():
    adapter = ThoughtAdapter(4)
    marker = torch.tensor([0.25, -0.5, 1.0, 0.75])
    with torch.no_grad():
        adapter.projection.weight.zero_()
        adapter.projection.bias.copy_(marker)
    thoughts = torch.randn(3, 4)

    torch.testing.assert_close(
        adapter(thoughts),
        (2.0 * torch.nn.functional.silu(marker)).expand_as(thoughts),
    )


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


def test_thought_step_advances_state_without_rendering_machinery_changes():
    torch.manual_seed(23)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    ids = torch.randint(0, 32, (1, 4))
    caches = backbone.make_generation_cache(1, 8, torch.device("cpu"))
    with torch.no_grad():
        output = None
        for position in range(ids.size(1)):
            output = wrapper.token_step(ids[:, position], caches, position)
            caches = output.caches
        assert output is not None
        sample, _ = wrapper.transition.sample(
            output.predicted,
            wrapper.transition.predict_log_sigma(output.belief),
        )
        thought_output = wrapper.step(
            wrapper.thought_input(sample), caches, ids.size(1)
        )
    assert thought_output.logits.shape == output.logits.shape
    assert not torch.allclose(thought_output.belief, output.belief)


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

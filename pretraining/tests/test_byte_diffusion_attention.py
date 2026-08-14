from __future__ import annotations

import pytest
import torch

from pretraining.byte_diffusion import attention
from pretraining.byte_diffusion.attention import (
    CanvasBranchLayout,
    IntrospectionBranchLayout,
    PackedCleanQKV,
    branch_attention,
    build_canvas_block_mask,
    canvas_block_mask_metadata,
    build_introspection_block_mask,
    canvas_branch_allowed,
    dense_packed_clean_attention,
    introspection_branch_allowed,
    packed_clean_attention,
)
from pretraining.byte_diffusion.kernels import AttentionBackend, AttentionBackendCapabilities
from pretraining.byte_diffusion.layers import PackedSelfAttention
from pretraining.byte_diffusion.masks import canvas_branch_mask, introspection_mask


NO_ACCELERATOR = AttentionBackendCapabilities(False, False, False, False)


def _packed_example(*, query_heads: int = 2, kv_heads: int = 1) -> PackedCleanQKV:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(5, query_heads, 4, generator=generator)
    key = torch.randn(5, kv_heads, 4, generator=generator)
    value = torch.randn(5, kv_heads, 4, generator=generator)
    return PackedCleanQKV(
        query=query,
        key=key,
        value=value,
        cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32),
        absolute_positions=torch.tensor([11, 12, 13, 101, 102]),
        max_seqlen=3,
    )


def test_packed_contract_preserves_explicit_document_boundaries_and_positions() -> None:
    packed = _packed_example()
    packed.validate_boundaries()
    assert packed.total_tokens == 5
    assert packed.absolute_positions.tolist() == [11, 12, 13, 101, 102]


def test_packed_contract_rejects_bad_boundaries_when_explicitly_validated() -> None:
    packed = _packed_example()
    bad = PackedCleanQKV(
        packed.query,
        packed.key,
        packed.value,
        torch.tensor([0, 4, 5], dtype=torch.int32),
        packed.absolute_positions,
        max_seqlen=3,
    )
    with pytest.raises(ValueError, match="max_seqlen"):
        bad.validate_boundaries()


def test_dense_packed_attention_resets_causality_at_each_document() -> None:
    query = torch.ones(4, 1, 1)
    key = torch.ones(4, 1, 1)
    value = torch.tensor([1.0, 3.0, 100.0, 200.0]).view(4, 1, 1)
    packed = PackedCleanQKV(
        query,
        key,
        value,
        torch.tensor([0, 2, 4], dtype=torch.int32),
        torch.tensor([0, 1, 0, 1]),
        2,
    )
    output = dense_packed_clean_attention(packed).flatten()
    torch.testing.assert_close(output, torch.tensor([1.0, 2.0, 100.0, 150.0]))


def test_dense_packed_window_counts_query_itself() -> None:
    query = torch.ones(4, 1, 1)
    key = torch.ones(4, 1, 1)
    value = torch.arange(1, 5, dtype=torch.float32).view(4, 1, 1)
    packed = PackedCleanQKV(
        query,
        key,
        value,
        torch.tensor([0, 4], dtype=torch.int32),
        torch.arange(4),
        4,
    )
    output = dense_packed_clean_attention(packed, window=2).flatten()
    torch.testing.assert_close(output, torch.tensor([1.0, 1.5, 2.5, 3.5]))


def test_public_clean_dispatch_uses_dense_only_with_small_oracle_opt_in() -> None:
    packed = _packed_example()
    expected = dense_packed_clean_attention(packed, window=2)
    actual = packed_clean_attention(
        packed,
        window=2,
        capabilities=NO_ACCELERATOR,
        allow_dense_reference=True,
    )
    torch.testing.assert_close(actual, expected)
    with pytest.raises(RuntimeError, match="not explicitly enabled"):
        packed_clean_attention(packed, capabilities=NO_ACCELERATOR)


def test_native_varlen_call_preserves_packing_gqa_and_exact_window(monkeypatch) -> None:
    import torch.nn.attention.varlen as varlen_module

    packed = _packed_example()
    captured = {}

    def fake_varlen(query, key, value, cu_q, cu_k, max_q, max_k, **kwargs):
        captured.update(
            query=query,
            key=key,
            value=value,
            cu_q=cu_q,
            cu_k=cu_k,
            max_q=max_q,
            max_k=max_k,
            **kwargs,
        )
        return torch.empty_like(query)

    monkeypatch.setattr(
        attention, "_resolve_backend", lambda *args, **kwargs: AttentionBackend.VARLEN_FLASH
    )
    monkeypatch.setattr(varlen_module, "varlen_attn", fake_varlen)
    output = packed_clean_attention(packed, window=512)
    assert output.shape == packed.query.shape
    assert captured["query"] is packed.query
    assert captured["cu_q"] is packed.cu_seqlens
    assert captured["cu_k"] is packed.cu_seqlens
    assert captured["max_q"] == captured["max_k"] == packed.max_seqlen
    assert captured["window_size"] == (511, 0)
    assert captured["enable_gqa"] is True


def test_clean_dispatch_fails_closed_for_long_dense_request() -> None:
    packed = _packed_example()
    with pytest.raises(RuntimeError, match="dense reference backend is not permitted"):
        packed_clean_attention(
            packed,
            backend="dense_reference",
            capabilities=NO_ACCELERATOR,
            allow_dense_reference=True,
            dense_reference_limit=2,
        )


def _canvas_layout(*, windowed: bool = False) -> CanvasBranchLayout:
    clean_valid = torch.tensor([[True, True, True, True, False]])
    branch_valid = torch.tensor(
        [[[True, True, True], [True, True, False]]]
    )
    kwargs = {}
    if windowed:
        kwargs = {
            "prefix_window": 2,
            "clean_positions": torch.tensor([[0, 1, 2, 3, 4]]),
            "branch_positions": torch.tensor([[[3, 4, 5], [4, 5, 6]]]),
        }
    return CanvasBranchLayout(
        clean_valid,
        branch_valid,
        torch.tensor([[3, 4]]),
        **kwargs,
    )


def _document_canvas_layout(*, byte_geometry: bool) -> CanvasBranchLayout:
    if byte_geometry:
        document_length, branches, canvas = 128, 2, 128
    else:
        document_length, branches, canvas = 16, 4, 8
    clean_length = 2 * document_length
    clean_positions = torch.arange(document_length).repeat(2)[None]
    clean_segments = torch.arange(2).repeat_interleave(document_length)[None]
    branch_segments = torch.arange(branches) % 2
    prefix_lengths = (
        branch_segments * document_length + document_length
    )[None]
    branch_positions = (
        torch.full((branches, 1), document_length)
        + torch.arange(canvas)[None]
    )[None]
    branch_valid = torch.ones(1, branches, canvas, dtype=torch.bool)
    if not byte_geometry:
        branch_valid[0, -1, -3:] = False
    return CanvasBranchLayout(
        clean_valid=torch.ones(1, clean_length, dtype=torch.bool),
        branch_valid=branch_valid,
        prefix_lengths=prefix_lengths,
        prefix_window=document_length,
        clean_positions=clean_positions,
        branch_positions=branch_positions,
        clean_segment_ids=clean_segments,
        branch_segment_ids=branch_segments[None],
    )


def _metadata_topology(counts: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    topology = torch.zeros_like(indices, dtype=torch.bool)
    slots = torch.arange(indices.shape[-1])[None, None, None, :]
    topology.scatter_(-1, indices.to(torch.long), slots < counts[..., None])
    return topology[:, 0]


def _metadata_token_mask(
    metadata: attention.CanvasBlockMaskMetadata,
    token_rule: torch.Tensor,
) -> torch.Tensor:
    assert metadata.full_kv_num_blocks is not None
    assert metadata.full_kv_indices is not None
    partial = _metadata_topology(metadata.kv_num_blocks, metadata.kv_indices)
    full = _metadata_topology(
        metadata.full_kv_num_blocks, metadata.full_kv_indices
    )
    q_block, kv_block = metadata.block_size
    query_blocks = torch.arange(metadata.query_length) // q_block
    key_blocks = torch.arange(metadata.kv_length) // kv_block
    partial_tokens = partial[:, query_blocks[:, None], key_blocks[None, :]]
    full_tokens = full[:, query_blocks[:, None], key_blocks[None, :]]
    return full_tokens | (partial_tokens & token_rule)


def test_canvas_dense_branch_mask_matches_family_oracle() -> None:
    layout = _canvas_layout()
    combined = canvas_branch_mask(
        layout.clean_valid, layout.branch_valid, layout.prefix_lengths
    )
    expected = combined[:, layout.clean_length :, :]
    assert torch.equal(canvas_branch_allowed(layout), expected)


def test_canvas_dense_branch_mask_broadcasts_over_multiple_rows() -> None:
    layout = _canvas_layout()
    batched = CanvasBranchLayout(
        clean_valid=layout.clean_valid.expand(2, -1).clone(),
        branch_valid=layout.branch_valid.expand(2, -1, -1).clone(),
        prefix_lengths=torch.tensor([[3, 4], [1, 2]]),
    )
    allowed = canvas_branch_allowed(batched)
    assert allowed.shape == (2, batched.query_length, batched.kv_length)
    assert not torch.equal(allowed[0], allowed[1])


def test_canvas_prefix_boundary_validation_is_explicit() -> None:
    layout = _canvas_layout()
    bad = CanvasBranchLayout(
        layout.clean_valid,
        layout.branch_valid,
        torch.tensor([[3, layout.clean_length + 1]]),
    )
    with pytest.raises(ValueError, match="outside the clean bank"):
        bad.validate_prefix_lengths()


def test_canvas_segment_ids_prevent_cross_document_prefix_attention() -> None:
    layout = CanvasBranchLayout(
        clean_valid=torch.ones(1, 6, dtype=torch.bool),
        branch_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        prefix_lengths=torch.tensor([[6]]),
        clean_segment_ids=torch.tensor([[0, 0, 0, 1, 1, 1]]),
        branch_segment_ids=torch.tensor([[1]]),
    )
    allowed = canvas_branch_allowed(layout)
    assert allowed[0, 0, :6].tolist() == [False, False, False, True, True, True]


def test_windowed_canvas_prefix_uses_absolute_positions_but_own_branch_is_bidi() -> None:
    layout = _canvas_layout(windowed=True)
    allowed = canvas_branch_allowed(layout)
    # The branch origin is absolute position 3: the same two-token decoder
    # prefix (clean keys 1 and 2) is shared by every query in that branch.
    assert allowed[0, 0].tolist() == [
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    # The invalid final query of branch one has no keys at all.
    assert not allowed[0, 5].any()


def test_canvas_block_mask_token_rule_matches_dense_oracle() -> None:
    from torch.nn.attention.flex_attention import create_mask

    layout = _canvas_layout()
    block_mask = build_canvas_block_mask(layout, block_size=16, compile_mask=False)
    actual = create_mask(
        block_mask.mask_mod,
        B=layout.batch_size,
        H=1,
        Q_LEN=layout.query_length,
        KV_LEN=layout.kv_length,
        device="cpu",
    )[:, 0]
    assert torch.equal(actual, canvas_branch_allowed(layout))


def test_canvas_metadata_block_mask_matches_dense_oracle() -> None:
    from torch.nn.attention.flex_attention import BlockMask

    layout = _canvas_layout(windowed=True)
    metadata = canvas_block_mask_metadata(layout, block_size=4)
    block_mask = build_canvas_block_mask(layout, metadata=metadata)
    batch = torch.arange(layout.batch_size)[:, None, None]
    query = torch.arange(layout.query_length)[None, :, None]
    key = torch.arange(layout.kv_length)[None, None, :]
    actual = block_mask.mask_mod(batch, torch.zeros_like(batch), query, key)
    assert torch.equal(actual, canvas_branch_allowed(layout))

    # The direct constructor must carry the exact backward transpose PyTorch
    # would derive from the K/V candidates, not merely an exact token mask.
    reference = BlockMask.from_kv_blocks(
        metadata.kv_num_blocks,
        metadata.kv_indices,
        metadata.full_kv_num_blocks,
        metadata.full_kv_indices,
        BLOCK_SIZE=metadata.block_size,
        mask_mod=block_mask.mask_mod,
        seq_lengths=(metadata.query_length, metadata.kv_length),
    )
    assert torch.equal(metadata.q_num_blocks, reference.q_num_blocks)
    assert torch.equal(metadata.q_indices, reference.q_indices)
    assert torch.equal(metadata.full_q_num_blocks, reference.full_q_num_blocks)
    assert torch.equal(metadata.full_q_indices, reference.full_q_indices)

    q_block, kv_block = metadata.block_size
    candidate_blocks = _metadata_topology(
        metadata.kv_num_blocks, metadata.kv_indices
    ) | _metadata_topology(
        metadata.full_kv_num_blocks, metadata.full_kv_indices
    )
    q_blocks = torch.arange(layout.query_length) // q_block
    kv_blocks = torch.arange(layout.kv_length) // kv_block
    candidate_tokens = candidate_blocks[:, q_blocks[:, None], kv_blocks[None, :]]
    assert bool((~canvas_branch_allowed(layout) | candidate_tokens).all())


@pytest.mark.parametrize(
    ("layout", "block_size"),
    [
        pytest.param(_canvas_layout(), 2, id="invalid-tail"),
        pytest.param(_canvas_layout(windowed=True), (3, 4), id="prefix-window"),
        pytest.param(
            _document_canvas_layout(byte_geometry=True), 128, id="byte-multidoc"
        ),
        pytest.param(
            _document_canvas_layout(byte_geometry=False), 8, id="patch-multidoc"
        ),
    ],
)
def test_canvas_metadata_full_blocks_exactly_match_dense_block_oracle(
    layout: CanvasBranchLayout,
    block_size: int | tuple[int, int],
) -> None:
    allowed = canvas_branch_allowed(layout)
    metadata = canvas_block_mask_metadata(layout, block_size=block_size)
    partial = _metadata_topology(metadata.kv_num_blocks, metadata.kv_indices)
    assert metadata.full_kv_num_blocks is not None
    assert metadata.full_kv_indices is not None
    full = _metadata_topology(
        metadata.full_kv_num_blocks, metadata.full_kv_indices
    )

    q_block, kv_block = metadata.block_size
    query_padding = partial.shape[-2] * q_block - layout.query_length
    key_padding = partial.shape[-1] * kv_block - layout.kv_length
    blocked = torch.nn.functional.pad(
        allowed, (0, key_padding, 0, query_padding)
    ).view(
        layout.batch_size,
        partial.shape[-2],
        q_block,
        partial.shape[-1],
        kv_block,
    ).permute(0, 1, 3, 2, 4)
    allowed_per_block = blocked.sum((-2, -1))
    expected_full = allowed_per_block == q_block * kv_block
    expected_partial = (allowed_per_block > 0) & ~expected_full

    assert torch.equal(full, expected_full)
    assert not bool((partial & full).any())
    assert bool((~expected_partial | partial).all())
    assert torch.equal(_metadata_token_mask(metadata, allowed), allowed)


@pytest.mark.parametrize(
    ("layout", "block_size"),
    [
        pytest.param(_canvas_layout(windowed=True), (3, 4), id="window-tail"),
        pytest.param(
            _document_canvas_layout(byte_geometry=True), 128, id="byte-multidoc"
        ),
        pytest.param(
            _document_canvas_layout(byte_geometry=False), 8, id="patch-multidoc"
        ),
    ],
)
def test_canvas_full_block_fast_path_matches_dense_forward_and_gradients(
    layout: CanvasBranchLayout,
    block_size: int | tuple[int, int],
) -> None:
    allowed = canvas_branch_allowed(layout)
    metadata = canvas_block_mask_metadata(layout, block_size=block_size)
    sparse_rule = _metadata_token_mask(metadata, allowed)
    generator = torch.Generator().manual_seed(91)
    query = torch.randn(
        layout.batch_size, 2, layout.query_length, 4, generator=generator
    )
    key = torch.randn(
        layout.batch_size, 1, layout.kv_length, 4, generator=generator
    )
    value = torch.randn(
        layout.batch_size, 1, layout.kv_length, 4, generator=generator
    )
    probe = torch.randn(query.shape, generator=generator)

    def forward_and_gradients(rule: torch.Tensor) -> tuple[torch.Tensor, ...]:
        inputs = tuple(
            tensor.detach().clone().requires_grad_()
            for tensor in (query, key, value)
        )
        output = attention._dense_attention(*inputs, rule, scale=None)
        gradients = torch.autograd.grad((output * probe).sum(), inputs)
        return (output, *gradients)

    expected = forward_and_gradients(allowed)
    actual = forward_and_gradients(sparse_rule)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_canvas_metadata_scales_to_2048_branches_without_dense_mask() -> None:
    clean_length, branches, branch_length = 16_384, 2_048, 16
    layout = CanvasBranchLayout(
        clean_valid=torch.ones(1, clean_length, dtype=torch.bool),
        branch_valid=torch.ones(1, branches, branch_length, dtype=torch.bool),
        prefix_lengths=(torch.arange(branches) * 4).clamp_max(clean_length)[None],
        prefix_window=512,
        clean_positions=torch.arange(clean_length)[None],
        branch_positions=(
            (torch.arange(branches) * 4)[:, None]
            + torch.arange(branch_length)[None]
        )[None],
    )
    metadata = canvas_block_mask_metadata(layout)
    assert metadata.kv_num_blocks.shape == (1, 1, 256)
    assert int(metadata.kv_num_blocks.max()) <= 6
    # Flex uses the final dimension as the complete physical K/V-block
    # capacity when deriving its transposed backward topology.  This remains
    # tiny relative to a dense token mask (98K integers versus >1.6B bools).
    assert metadata.kv_indices.shape == (1, 1, 256, 384)
    assert metadata.kv_indices.numel() < 100_000


def _introspection_layout() -> IntrospectionBranchLayout:
    return IntrospectionBranchLayout(
        clean_valid=torch.tensor([[True, True, True, True, True, False]]),
        proposal_valid=torch.tensor([[True, True, True, True, True, False]]),
        block_size=2,
    )


def test_introspection_dense_branch_mask_matches_family_oracle() -> None:
    layout = _introspection_layout()
    combined = introspection_mask(
        layout.clean_valid, layout.proposal_valid, layout.block_size
    )
    expected = combined[:, layout.clean_length :, :]
    assert torch.equal(introspection_branch_allowed(layout), expected)


def test_introspection_block_mask_token_rule_matches_dense_oracle() -> None:
    from torch.nn.attention.flex_attention import create_mask

    layout = _introspection_layout()
    block_mask = build_introspection_block_mask(
        layout, block_size=16, compile_mask=False
    )
    actual = create_mask(
        block_mask.mask_mod,
        B=layout.batch_size,
        H=1,
        Q_LEN=layout.query_length,
        KV_LEN=layout.kv_length,
        device="cpu",
    )[:, 0]
    assert torch.equal(actual, introspection_branch_allowed(layout))


def test_introspection_segment_ids_prevent_cross_document_attention() -> None:
    segments = torch.tensor([[0, 0, 1, 1]])
    layout = IntrospectionBranchLayout(
        clean_valid=torch.ones(1, 4, dtype=torch.bool),
        proposal_valid=torch.ones(1, 4, dtype=torch.bool),
        block_size=1,
        clean_segment_ids=segments,
        proposal_segment_ids=segments.clone(),
    )
    allowed = introspection_branch_allowed(layout)
    # Proposal query 2 is in document 1. Earlier clean blocks 0/1 belong to
    # document 0 and must not be reconnected by the shared physical bank.
    assert allowed[0, 2, :4].tolist() == [False, False, False, False]
    assert allowed[0, 3, :4].tolist() == [False, False, True, False]


@pytest.mark.parametrize("layout_factory", [_canvas_layout, _introspection_layout])
def test_dense_branch_attention_zeroes_invalid_queries_and_supports_gqa(layout_factory) -> None:
    layout = layout_factory()
    generator = torch.Generator().manual_seed(3)
    query = torch.randn(layout.batch_size, 2, layout.query_length, 4, generator=generator)
    clean_key = torch.randn(layout.batch_size, 1, layout.clean_length, 4, generator=generator)
    clean_value = torch.randn_like(clean_key)
    branch_key = torch.randn(layout.batch_size, 1, layout.query_length, 4, generator=generator)
    branch_value = torch.randn_like(branch_key)
    output = branch_attention(
        query,
        clean_key,
        clean_value,
        branch_key,
        branch_value,
        layout,
        capabilities=NO_ACCELERATOR,
        allow_dense_reference=True,
    )
    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert torch.equal(output[:, :, -1], torch.zeros_like(output[:, :, -1]))


def test_flex_call_builds_one_shared_clean_plus_branch_bank(monkeypatch) -> None:
    import torch.nn.attention.flex_attention as flex_module

    layout = _canvas_layout()
    query = torch.zeros(layout.batch_size, 2, layout.query_length, 4)
    clean_key = torch.ones(layout.batch_size, 1, layout.clean_length, 4)
    clean_value = clean_key * 2
    branch_key = torch.full(
        (layout.batch_size, 1, layout.query_length, 4), 3.0
    )
    branch_value = branch_key * 2
    sentinel_mask = object()
    captured = {}

    def fake_flex(q, k, v, **kwargs):
        captured.update(query=q, key=k, value=v, **kwargs)
        return torch.empty_like(q)

    monkeypatch.setattr(
        attention, "_resolve_backend", lambda *args, **kwargs: AttentionBackend.FLEX
    )
    monkeypatch.setattr(flex_module, "flex_attention", fake_flex)
    output = branch_attention(
        query,
        clean_key,
        clean_value,
        branch_key,
        branch_value,
        layout,
        block_mask=sentinel_mask,
    )
    assert output.shape == query.shape
    assert captured["key"].shape[2] == layout.kv_length
    assert torch.equal(captured["key"][:, :, : layout.clean_length], clean_key)
    assert torch.equal(captured["key"][:, :, layout.clean_length :], branch_key)
    assert captured["block_mask"] is sentinel_mask
    assert captured["enable_gqa"] is True
    assert captured["kernel_options"]["BACKEND"] == "TRITON"


def test_branch_dispatch_fails_closed_without_flex() -> None:
    layout = _canvas_layout()
    shape = (layout.batch_size, 1, layout.query_length, 2)
    clean_shape = (layout.batch_size, 1, layout.clean_length, 2)
    with pytest.raises(RuntimeError, match="not explicitly enabled"):
        branch_attention(
            torch.zeros(shape),
            torch.zeros(clean_shape),
            torch.zeros(clean_shape),
            torch.zeros(shape),
            torch.zeros(shape),
            layout,
            capabilities=NO_ACCELERATOR,
        )


def test_prepacked_document_branches_match_dynamic_reference_with_tail() -> None:
    torch.manual_seed(29)
    layer = PackedSelfAttention(dim=8, heads=2, rope_theta=10_000.0)
    clean_length = 16
    branch_length = 4
    branches = 3
    total_length = clean_length + branches * branch_length
    states = torch.randn(1, total_length, 8)
    clean_valid = torch.tensor(
        [[True] * 8 + [True] * 6 + [False] * 2]
    )
    clean_documents = torch.tensor([[0] * 8 + [1] * 6 + [-1] * 2])
    clean_positions = torch.tensor([[*range(8), *range(6), 0, 0]])
    branch_valid = torch.tensor(
        [[[True] * 4, [True, True, False, False], [False] * 4]]
    )
    branch_starts = torch.tensor([[4, 12, 0]])
    branch_positions = torch.tensor(
        [[[4, 5, 6, 7], [4, 5, 6, 7], [0, 1, 2, 3]]]
    )
    positions = torch.cat((clean_positions, branch_positions.flatten(1, 2)), dim=1)
    clean_indices = torch.tensor([*range(14)])
    clean_cu = torch.tensor([0, 8, 14], dtype=torch.int32)

    dynamic = layer.forward_document_branches(
        states,
        clean_valid=clean_valid,
        clean_indices=clean_indices,
        clean_cu_seqlens=clean_cu,
        clean_document_ids=clean_documents,
        positions=positions,
        branch_valid=branch_valid,
        branch_starts=branch_starts,
        clean_window=4,
        allow_dense_reference=True,
    )
    prepacked = layer.forward_packed_document_branches(
        states,
        clean_length=clean_length,
        clean_indices=clean_indices,
        clean_cu_seqlens=clean_cu,
        positions=positions,
        branch_query_indices=torch.tensor([16, 17, 18, 19, 20, 21]),
        branch_kv_indices=torch.tensor(
            [0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21]
        ),
        branch_query_cu_seqlens=torch.tensor([0, 4, 6], dtype=torch.int32),
        branch_kv_cu_seqlens=torch.tensor([0, 8, 14], dtype=torch.int32),
        max_branch_query_length=4,
        max_branch_kv_length=8,
        clean_window=4,
        allow_dense_reference=True,
    )

    torch.testing.assert_close(prepacked, dynamic, rtol=0, atol=0)


@pytest.mark.parametrize("window", [4, None])
def test_shared_document_branches_match_duplicated_oracle_and_do_not_leak(
    window: int | None,
) -> None:
    torch.manual_seed(41)
    layer = PackedSelfAttention(dim=8, heads=2, rope_theta=10_000.0)
    clean_length, branches, branch_length = 16, 3, 4
    states = torch.randn(1, clean_length + branches * branch_length, 8)
    clean_valid = torch.tensor([[True] * 8 + [True] * 6 + [False] * 2])
    documents = torch.tensor([[0] * 8 + [1] * 6 + [-1] * 2])
    clean_positions = torch.tensor([[*range(8), *range(6), 0, 0]])
    branch_valid = torch.tensor(
        [[[True] * 4, [True, True, False, False], [False] * 4]]
    )
    starts = torch.tensor([[4, 12, 0]])
    branch_positions = torch.tensor(
        [[[4, 5, 6, 7], [4, 5, 6, 7], [0, 1, 2, 3]]]
    )
    positions = torch.cat((clean_positions, branch_positions.flatten(1, 2)), 1)
    clean_indices = torch.arange(14)
    clean_cu = torch.tensor([0, 8, 14], dtype=torch.int32)
    layout = CanvasBranchLayout(
        clean_valid,
        branch_valid,
        starts,
        window,
        None if window is None else clean_positions,
        None if window is None else branch_positions,
        documents,
        torch.tensor([[0, 1, 0]]),
    )
    shared = layer.forward_shared_document_branches(
        states,
        clean_length=clean_length,
        clean_indices=clean_indices,
        clean_cu_seqlens=clean_cu,
        positions=positions,
        layout=layout,
        clean_window=window,
        allow_dense_reference=True,
    )
    oracle = layer.forward_document_branches(
        states,
        clean_valid=clean_valid,
        clean_indices=clean_indices,
        clean_cu_seqlens=clean_cu,
        clean_document_ids=documents,
        positions=positions,
        branch_valid=branch_valid,
        branch_starts=starts,
        clean_window=window,
        allow_dense_reference=True,
    )
    torch.testing.assert_close(shared, oracle, rtol=1e-5, atol=1e-6)

    # Branch one belongs to document one. Document-zero clean bytes, PAD, and
    # branch zero are sentinels that its output must never observe.
    changed = states.clone()
    changed[:, :8] += 1_000
    changed[:, 14:16] -= 2_000
    changed[:, clean_length : clean_length + branch_length] += 3_000
    changed_output = layer.forward_shared_document_branches(
        changed,
        clean_length=clean_length,
        clean_indices=clean_indices,
        clean_cu_seqlens=clean_cu,
        positions=positions,
        layout=layout,
        clean_window=window,
        allow_dense_reference=True,
    )
    branch_one = slice(clean_length + branch_length, clean_length + 2 * branch_length)
    torch.testing.assert_close(
        changed_output[:, branch_one], shared[:, branch_one], rtol=0, atol=0
    )

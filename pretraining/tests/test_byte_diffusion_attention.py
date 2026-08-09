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
    build_introspection_block_mask,
    canvas_branch_allowed,
    dense_packed_clean_attention,
    introspection_branch_allowed,
    packed_clean_attention,
)
from pretraining.byte_diffusion.kernels import AttentionBackend, AttentionBackendCapabilities
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
    # First branch query is at absolute position 3: clean key 1 is excluded by
    # the strict two-token window, clean key 2 and all own branch keys survive.
    assert allowed[0, 0].tolist() == [
        False,
        False,
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

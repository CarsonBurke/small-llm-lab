from __future__ import annotations

import copy
import math

import torch
from torch import nn

from pretraining.bolmo import (
    BYTE_EOT_ID,
    BolmoArchitecture,
    BolmoBatch,
    BolmoLocalBlock,
    BolmoLocalDecoder,
    BolmoModel,
    BolmoTeacher,
    NonCausalBoundaryPredictor,
    _stateless_mlstm_forward,
    bernoulli_kl_from_log_probs,
    parameter_report,
)
from pretraining.nanogpt_mini.nanogpt_mini_kda_model import RMSNorm


def tiny_architecture(**overrides) -> BolmoArchitecture:
    values = dict(
        model_dim=16,
        local_heads=4,
        local_ffn_hidden=22,
        backend="native",
        chunk_size=4,
        num_special_tokens=1,
    )
    values.update(overrides)
    return BolmoArchitecture(**values)


def test_local_kernel_dtype_is_validated() -> None:
    architecture = tiny_architecture(autocast_kernel_dtype="bfloat16")
    assert architecture.autocast_kernel_dtype == "bfloat16"
    try:
        tiny_architecture(autocast_kernel_dtype="float16")
    except ValueError as error:
        assert "autocast_kernel_dtype" in str(error)
    else:
        raise AssertionError("unsupported local-kernel dtype was accepted")


def test_temperature_bernoulli_kl_is_zero_at_equality_and_has_gradient() -> None:
    teacher = torch.tensor([-0.2, -2.0, -8.0])
    equal = bernoulli_kl_from_log_probs(teacher, teacher)
    torch.testing.assert_close(equal, torch.zeros_like(equal), atol=2e-6, rtol=0)

    student = torch.tensor([-0.4, -1.0, -5.0], requires_grad=True)
    loss = bernoulli_kl_from_log_probs(teacher, student).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None
    assert torch.all(torch.isfinite(student.grad))


def test_local_mlstm_block_runs_forward_and_backward_on_native_backend() -> None:
    block = BolmoLocalBlock(tiny_architecture())
    inputs = torch.randn(2, 8, 16, requires_grad=True)
    outputs = block(inputs)
    assert outputs.shape == inputs.shape
    outputs.square().mean().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in block.parameters())


def test_stateless_mlstm_matches_upstream_hidden_and_gradients() -> None:
    torch.manual_seed(17)
    reference_block = BolmoLocalBlock(tiny_architecture())
    stateless_block = copy.deepcopy(reference_block)
    reference_input = torch.randn(2, 8, 16, requires_grad=True)
    stateless_input = reference_input.detach().clone().requires_grad_(True)

    reference, _ = reference_block.mlstm(
        reference_block.mlstm_norm(reference_input)
    )
    stateless = _stateless_mlstm_forward(
        stateless_block.mlstm,
        stateless_block.mlstm_norm(stateless_input),
    )
    torch.testing.assert_close(stateless, reference)

    reference.square().mean().backward()
    stateless.square().mean().backward()
    torch.testing.assert_close(stateless_input.grad, reference_input.grad)
    for (_, reference_parameter), (_, stateless_parameter) in zip(
        reference_block.named_parameters(),
        stateless_block.named_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(stateless_parameter.grad, reference_parameter.grad)


def test_boundary_predictor_uses_next_byte_and_forces_bos() -> None:
    predictor = NonCausalBoundaryPredictor(4)
    hidden = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    log_probs = predictor(hidden, valid)
    assert log_probs[0, 0] == 0  # forced synthetic BOS boundary
    assert log_probs[0, 1] > math.log(0.99)  # opposite adjacent vectors
    assert log_probs[0, -1] < -10_000  # no leaked lookahead at row end


def test_batch_contract_covers_valid_bytes_with_source_patches() -> None:
    batch = BolmoBatch(
        source_ids=torch.tensor([[0, 2, 3, 10]]),
        source_valid_mask=torch.tensor([[True, True, True, False]]),
        byte_ids=torch.tensor([[BYTE_EOT_ID, 97, 98, 99, 257]]),
        expanded_ids=torch.tensor([[0, 10, 10, 10, 10]]),
        oracle_boundaries=torch.tensor([[True, False, True, True, False]]),
        valid_mask=torch.tensor([[True, True, True, True, False]]),
        score_mask=torch.tensor([[False, True, True, True, False]]),
        patch_lens=torch.tensor([[1, 2, 1, 0]]),
    )
    batch.validate(source_vocab_size=10, atomic_vocab_size=257)


class _FakeEncoder(nn.Module):
    def __init__(self, hidden: torch.Tensor, boundary_log_probs: torch.Tensor):
        super().__init__()
        self.source_vocab_size = 10
        self.hidden = hidden
        self.boundary_log_probs = boundary_log_probs

    def forward(self, byte_ids, expanded_ids, valid_mask):
        return self.hidden, self.boundary_log_probs

    def pool(self, hidden, boundaries, *, max_patches=None):
        del max_patches
        return hidden, torch.ones(hidden.shape[:2], dtype=torch.bool)


class _FakeDecoder(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits

    def forward(self, byte_hidden, patch_hidden, boundaries):
        return self.logits


class _FakeTeacher:
    def __init__(self, width: int, source_vocab_size: int):
        self.width = width
        self.source_vocab_size = source_vocab_size

    def forward(self, source_ids, stitch_depth):
        del stitch_depth
        batch, length = source_ids.shape
        stitched = torch.randn(batch, length, self.width)
        final = torch.randn(batch, length, self.width)
        return stitched, final

    def selected_log_probs(self, final_hidden, targets, *, positions_per_chunk):
        del final_hidden
        assert positions_per_chunk > 0
        return torch.randn_like(targets, dtype=torch.float32).sub_(2.0)


def test_validation_marginalizes_fused_boundary_and_retains_joint_nll() -> None:
    architecture = tiny_architecture()
    model = BolmoModel(
        global_blocks=nn.Sequential(nn.Identity()),
        global_norm=RMSNorm(16),
        source_embedding=torch.randn(10, 16),
        source_model_config={"model_dim": 16},
        architecture=architecture,
    )
    logits = torch.full((1, 2, architecture.fused_vocab_size), -4.0)
    target = 97
    logits[0, 0, target] = 0.7
    logits[0, 0, target + architecture.atomic_vocab_size] = 0.2
    model.local_encoder = _FakeEncoder(
        hidden=torch.zeros(1, 2, 16),
        boundary_log_probs=torch.tensor([[0.0, -100_000.0]]),
    )
    model.local_decoder = _FakeDecoder(logits)
    batch = BolmoBatch(
        source_ids=torch.tensor([[0, 1]]),
        source_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        byte_ids=torch.tensor([[BYTE_EOT_ID, target]]),
        expanded_ids=torch.tensor([[0, 10]]),
        oracle_boundaries=torch.ones(1, 2, dtype=torch.bool),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        score_mask=torch.tensor([[False, True]]),
        patch_lens=torch.ones(1, 2, dtype=torch.long),
    )
    statistics = model.validation_statistics(batch)
    oracle_statistics = model.validation_statistics(batch, oracle_boundaries=True)
    log_probs = logits[0, 0].log_softmax(-1)
    expected_byte = -torch.logaddexp(
        log_probs[target], log_probs[target + architecture.atomic_vocab_size]
    )
    expected_joint = -log_probs[target]
    torch.testing.assert_close(statistics.byte_nll, expected_byte)
    torch.testing.assert_close(oracle_statistics.byte_nll, expected_byte)
    torch.testing.assert_close(statistics.joint_nll, expected_joint)
    torch.testing.assert_close(
        oracle_statistics.joint_nll,
        -log_probs[target + architecture.atomic_vocab_size],
    )
    assert statistics.byte_nll < statistics.joint_nll
    assert statistics.scored_bytes.item() == 1
    assert statistics.valid_atoms.item() == 2


def test_validation_charges_atomic_special_as_one_byte() -> None:
    architecture = tiny_architecture()
    model = BolmoModel(
        global_blocks=nn.Sequential(nn.Identity()),
        global_norm=RMSNorm(16),
        source_embedding=torch.randn(10, 16),
        source_model_config={"model_dim": 16},
        architecture=architecture,
    )
    logits = torch.zeros(1, 2, architecture.fused_vocab_size)
    model.local_encoder = _FakeEncoder(
        hidden=torch.zeros(1, 2, 16),
        boundary_log_probs=torch.tensor([[0.0, -100_000.0]]),
    )
    model.local_decoder = _FakeDecoder(logits)
    batch = BolmoBatch(
        source_ids=torch.tensor([[0, 0]]),
        source_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        byte_ids=torch.tensor([[BYTE_EOT_ID, BYTE_EOT_ID]]),
        expanded_ids=torch.tensor([[0, 0]]),
        oracle_boundaries=torch.ones(1, 2, dtype=torch.bool),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        score_mask=torch.tensor([[False, True]]),
        patch_lens=torch.ones(1, 2, dtype=torch.long),
    )
    statistics = model.validation_statistics(batch)
    assert statistics.scored_bytes.item() == 1


def test_validation_ignores_padded_bytes_outside_the_atomic_vocabulary() -> None:
    architecture = tiny_architecture()
    model = BolmoModel(
        global_blocks=nn.Sequential(nn.Identity()),
        global_norm=RMSNorm(16),
        source_embedding=torch.randn(10, 16),
        source_model_config={"model_dim": 16},
        architecture=architecture,
    )
    pad = architecture.byte_pad_id
    logits = torch.full((1, 3, architecture.fused_vocab_size), -4.0)
    target = 97
    logits[0, 0, target] = 0.7
    logits[0, 0, target + architecture.atomic_vocab_size] = 0.2
    model.local_encoder = _FakeEncoder(
        hidden=torch.zeros(1, 3, 16),
        boundary_log_probs=torch.tensor([[0.0, -100_000.0, -100_000.0]]),
    )
    model.local_decoder = _FakeDecoder(logits)
    # Collation pads every batch out to a multiple of 128 bytes, so the padded
    # tail is the common case rather than an edge case.
    batch = BolmoBatch(
        source_ids=torch.tensor([[0, 1, 0]]),
        source_valid_mask=torch.tensor([[True, True, False]]),
        byte_ids=torch.tensor([[BYTE_EOT_ID, target, pad]]),
        expanded_ids=torch.tensor([[0, 10, 0]]),
        oracle_boundaries=torch.tensor([[True, True, False]]),
        valid_mask=torch.tensor([[True, True, False]]),
        score_mask=torch.tensor([[False, True, False]]),
        patch_lens=torch.tensor([[1, 1, 0]]),
    )
    statistics = model.validation_statistics(batch)
    log_probs = logits[0, 0].log_softmax(-1)
    torch.testing.assert_close(
        statistics.byte_nll,
        -torch.logaddexp(
            log_probs[target], log_probs[target + architecture.atomic_vocab_size]
        ),
    )
    torch.testing.assert_close(statistics.joint_nll, -log_probs[target])
    assert statistics.scored_bytes.item() == 1


def test_fixed_stride_patching_is_well_formed_over_valid_bytes_only() -> None:
    valid = torch.tensor(
        [
            [True] * 9 + [False] * 3,
            [True] * 4 + [False] * 8,
        ]
    )
    boundaries = BolmoModel._fixed_stride_boundaries(valid, 4)
    # Every 4th valid byte closes a patch; the first byte and the last valid
    # byte of each row are always boundaries; padding never is.
    torch.testing.assert_close(
        boundaries,
        torch.tensor(
            [
                [True, False, False, True, False, False, False, True, True]
                + [False] * 3,
                [True, False, False, True] + [False] * 8,
            ]
        ),
    )
    assert not bool((boundaries & ~valid).any())
    # Patch lengths must exactly cover the valid prefix, which is what the
    # pooler assumes when it left-packs.
    for row in range(valid.shape[0]):
        ends = torch.nonzero(boundaries[row], as_tuple=False).flatten()
        assert int(ends[-1]) == int(valid[row].sum()) - 1

    strides = torch.tensor([[True] * 8 + [False] * 4])
    dense = BolmoModel._fixed_stride_boundaries(strides, 1)
    assert int(dense.sum()) == 8
    try:
        BolmoModel._fixed_stride_boundaries(valid, 0)
    except ValueError as error:
        assert "stride" in str(error)
    else:
        raise AssertionError("a non-positive stride was accepted")


def test_uniform_patching_matches_the_oracle_patch_count_exactly() -> None:
    # Ragged rows with very different byte densities, which is the case a
    # global stride cannot handle inside a fixed patch budget.
    valid = torch.tensor(
        [
            [True] * 12,
            [True] * 7 + [False] * 5,
            [True] * 2 + [False] * 10,
        ]
    )
    oracle = torch.tensor(
        [
            [True, False, True, False, False, True, False, False, False, True,
             False, True],
            [True, True, False, True, False, False, True] + [False] * 5,
            [True, True] + [False] * 10,
        ]
    )
    counts = oracle.sum(1)
    boundaries = BolmoModel._uniform_boundaries(valid, counts)
    # Same patch budget as the oracle, row by row — only placement differs.
    torch.testing.assert_close(boundaries.sum(1), counts)
    assert not bool((boundaries & ~valid).any())
    assert bool(boundaries[:, 0].all())
    for row in range(valid.shape[0]):
        ends = torch.nonzero(boundaries[row], as_tuple=False).flatten()
        # The last patch must close on the row's last valid byte, or the
        # pooler would leave a tail of bytes in no patch at all.
        assert int(ends[-1]) == int(valid[row].sum()) - 1
        gaps = (ends[1:] - ends[:-1]).tolist()
        # "As evenly as the integer split allows": every interior patch length
        # is within one byte of every other.
        if len(gaps) > 1:
            assert max(gaps) - min(gaps) <= 1

    try:
        BolmoModel._uniform_boundaries(valid, torch.tensor([1, 2, 2]))
    except ValueError as error:
        assert "BOS patch" in str(error)
    else:
        raise AssertionError("a row with no room for a second patch was accepted")

    try:
        BolmoModel._uniform_boundaries(valid, torch.tensor([12, 7, 9]))
    except ValueError as error:
        assert "more patches" in str(error)
    else:
        raise AssertionError("more patches than bytes was accepted")


def test_validation_rejects_patching_past_the_pooled_patch_budget() -> None:
    architecture = tiny_architecture()
    model = BolmoModel(
        global_blocks=nn.Sequential(nn.Identity()),
        global_norm=RMSNorm(16),
        source_embedding=torch.randn(10, 16),
        source_model_config={"model_dim": 16},
        architecture=architecture,
    )
    model.local_encoder = _FakeEncoder(
        hidden=torch.zeros(1, 4, 16),
        boundary_log_probs=torch.zeros(1, 4),
    )
    model.local_decoder = _FakeDecoder(
        torch.zeros(1, 4, architecture.fused_vocab_size)
    )
    # Four valid bytes at stride 1 need four patches, but the source axis only
    # budgets two. ``pool`` would silently drop the tail; this must not pass.
    batch = BolmoBatch(
        source_ids=torch.tensor([[0, 1]]),
        source_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        byte_ids=torch.tensor([[BYTE_EOT_ID, 97, 98, 99]]),
        expanded_ids=torch.zeros(1, 4, dtype=torch.long),
        oracle_boundaries=torch.tensor([[True, False, True, False]]),
        valid_mask=torch.ones(1, 4, dtype=torch.bool),
        score_mask=torch.tensor([[False, True, True, True]]),
        patch_lens=torch.tensor([[2, 2]]),
    )
    try:
        model.validation_statistics(batch, fixed_stride=1)
    except ValueError as error:
        assert "patch budget" in str(error)
    else:
        raise AssertionError("patching past the pooled patch budget was accepted")

    try:
        model.validation_statistics(
            batch, oracle_boundaries=True, fixed_stride=4
        )
    except ValueError as error:
        assert "different patchings" in str(error)
    else:
        raise AssertionError("two conflicting patchings were accepted")


def test_boundary_supervision_excludes_undefined_final_lookahead() -> None:
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    expected = torch.tensor(
        [[True, True, False, False], [True, False, False, False]]
    )
    torch.testing.assert_close(BolmoModel._boundary_valid_mask(valid), expected)


def test_global_states_stay_raw_and_source_norm_is_copied_to_byte_head() -> None:
    source_norm = RMSNorm(16)
    source_norm.gains.data.fill_(2.0)
    model = BolmoModel(
        global_blocks=nn.Sequential(nn.Identity()),
        global_norm=source_norm,
        source_embedding=torch.randn(10, 16),
        source_model_config={"model_dim": 16},
        architecture=tiny_architecture(),
    )
    inputs = torch.randn(2, 3, 16)
    torch.testing.assert_close(model.global_forward(inputs), inputs)
    torch.testing.assert_close(
        model.local_decoder.head_norm.gains, source_norm.gains
    )
    assert "global_norm" not in dict(model.named_modules())
    report = parameter_report(model)
    assert report["total"] == sum(parameter.numel() for parameter in model.parameters())


def test_eos_coalesced_patch_alignment_selects_eos_teacher_state() -> None:
    batch = BolmoBatch(
        source_ids=torch.tensor([[0, 7, 0, 8]]),
        source_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        byte_ids=torch.tensor([[BYTE_EOT_ID, 97, BYTE_EOT_ID, 98]]),
        expanded_ids=torch.zeros(1, 4, dtype=torch.long),
        oracle_boundaries=torch.tensor([[True, False, True, True]]),
        valid_mask=torch.ones(1, 4, dtype=torch.bool),
        score_mask=torch.tensor([[False, True, True, True]]),
        patch_lens=torch.ones(1, 4, dtype=torch.long),
    )
    indices, valid = BolmoModel._teacher_patch_alignment(batch)
    torch.testing.assert_close(indices, torch.tensor([[0, 2, 3, 3]]))
    torch.testing.assert_close(valid, torch.tensor([[True, True, True, False]]))
    original = BolmoModel._original_boundaries(batch)
    torch.testing.assert_close(original, torch.ones_like(original))


def test_selected_teacher_cross_entropy_matches_log_softmax() -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, 5, 11)
    targets = torch.randint(0, logits.shape[-1], (2, 5))
    selected = -torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), reduction="none"
    ).view_as(targets)
    reference = torch.gather(
        torch.nn.functional.log_softmax(logits.float(), dim=-1),
        -1,
        targets[..., None],
    ).squeeze(-1)
    torch.testing.assert_close(selected, reference)


def test_teacher_chunked_selected_log_probs_match_full_head() -> None:
    source = nn.Module()
    source.embed = nn.Embedding(11, 8)
    source.norm1 = nn.Identity()
    source.norm2 = nn.LayerNorm(8)
    source.proj = nn.Linear(8, 11)
    teacher = BolmoTeacher(source, nn.Sequential(nn.Identity()))  # type: ignore[arg-type]
    hidden = torch.randn(2, 5, 8)
    targets = torch.randint(0, 11, (2, 5))
    selected = teacher.selected_log_probs(hidden, targets, positions_per_chunk=3)
    logits = teacher.output_head(teacher.output_norm(hidden)).float()
    logits = 15.0 * logits * torch.rsqrt(logits.square() + 15.0**2)
    reference = torch.gather(
        logits.log_softmax(-1), -1, targets[..., None]
    ).squeeze(-1)
    torch.testing.assert_close(selected, reference)


def test_small_stage2_is_finite_and_backpropagates() -> None:
    architecture = tiny_architecture()
    model = BolmoModel(
        global_blocks=nn.Sequential(nn.Identity()),
        global_norm=RMSNorm(16),
        source_embedding=torch.randn(10, 16),
        source_model_config={"model_dim": 16},
        architecture=architecture,
    )
    batch = BolmoBatch(
        source_ids=torch.tensor([[0, 1, 2]]),
        source_valid_mask=torch.ones(1, 3, dtype=torch.bool),
        byte_ids=torch.tensor([[BYTE_EOT_ID, 97, 98, architecture.byte_pad_id]]),
        expanded_ids=torch.tensor([[0, 10, 10, 10]]),
        oracle_boundaries=torch.tensor([[True, False, True, False]]),
        valid_mask=torch.tensor([[True, True, True, False]]),
        score_mask=torch.tensor([[False, True, True, False]]),
        patch_lens=torch.tensor([[1, 2, 0]]),
    )
    # Correct the source contract: the second source token owns both bytes.
    batch.source_valid_mask = torch.tensor([[True, True, False]])
    loss = model.stage2(batch, oracle_boundaries=True)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_small_stage1_aligns_multibyte_patches_and_backpropagates() -> None:
    architecture = tiny_architecture()
    model = BolmoModel(
        global_blocks=nn.Sequential(nn.Identity()),
        global_norm=RMSNorm(16),
        source_embedding=torch.randn(10, 16),
        source_model_config={"model_dim": 16},
        architecture=architecture,
    )
    batch = BolmoBatch(
        source_ids=torch.tensor([[0, 1, 2]]),
        source_valid_mask=torch.ones(1, 3, dtype=torch.bool),
        byte_ids=torch.tensor([[BYTE_EOT_ID, 97, 98, 99]]),
        expanded_ids=torch.tensor([[0, 10, 10, 10]]),
        oracle_boundaries=torch.tensor([[True, False, True, True]]),
        valid_mask=torch.ones(1, 4, dtype=torch.bool),
        score_mask=torch.tensor([[False, True, True, True]]),
        patch_lens=torch.tensor([[1, 2, 1]]),
    )
    teacher = _FakeTeacher(width=16, source_vocab_size=10)
    loss = model.stage1(batch, teacher)  # type: ignore[arg-type]
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert model.local_encoder.byte_embedding.weight.grad is not None
    assert model.local_decoder.lm_head.weight.grad is not None


class _RecordingEncoder(_FakeEncoder):
    """``_FakeEncoder`` that remembers which mask ``pool`` was given."""

    def __init__(self, hidden: torch.Tensor, boundary_log_probs: torch.Tensor):
        super().__init__(hidden, boundary_log_probs)
        self.pooled_boundaries: torch.Tensor | None = None

    def pool(self, hidden, boundaries, *, max_patches=None):
        self.pooled_boundaries = boundaries.clone()
        return super().pool(hidden, boundaries, max_patches=max_patches)


class _LogitTap(nn.Module):
    """Pass-through wrapper that keeps the real decoder in the loop.

    Unlike ``_RecordingDecoder`` this exercises ``prepare_hidden``, where the
    routing shift is actually consumed — including the ``clamp(min=0)`` that
    only the shifted mask's leading ``-1`` ever engages.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.logits: torch.Tensor | None = None

    def forward(self, byte_hidden, patch_hidden, boundaries):
        self.logits = self.inner(byte_hidden, patch_hidden, boundaries)
        return self.logits


class _RecordingDecoder(_FakeDecoder):
    """``_FakeDecoder`` that remembers which mask it was routed with."""

    def __init__(self, logits: torch.Tensor):
        super().__init__(logits)
        self.routing_boundaries: torch.Tensor | None = None

    def forward(self, byte_hidden, patch_hidden, boundaries):
        self.routing_boundaries = boundaries.clone()
        return super().forward(byte_hidden, patch_hidden, boundaries)


def _routing_probe_model(
    architecture: BolmoArchitecture, boundary_log_probs: torch.Tensor
) -> BolmoModel:
    model = BolmoModel(
        global_blocks=nn.Sequential(nn.Identity()),
        global_norm=RMSNorm(16),
        source_embedding=torch.randn(10, 16),
        source_model_config={"model_dim": 16},
        architecture=architecture,
    )
    length = boundary_log_probs.shape[1]
    model.local_encoder = _RecordingEncoder(
        hidden=torch.zeros(1, length, 16), boundary_log_probs=boundary_log_probs
    )
    model.local_decoder = _RecordingDecoder(
        torch.zeros(1, length, architecture.fused_vocab_size)
    )
    return model


def _routing_probe_batch(length: int) -> BolmoBatch:
    return BolmoBatch(
        source_ids=torch.arange(length)[None, :],
        source_valid_mask=torch.ones(1, length, dtype=torch.bool),
        byte_ids=torch.tensor([[BYTE_EOT_ID] + [97] * (length - 1)]),
        expanded_ids=torch.zeros(1, length, dtype=torch.long),
        oracle_boundaries=torch.ones(1, length, dtype=torch.bool),
        valid_mask=torch.ones(1, length, dtype=torch.bool),
        score_mask=torch.tensor([[False] + [True] * (length - 1)]),
        patch_lens=torch.ones(1, length, dtype=torch.long),
    )


def test_causal_routing_drops_the_boundary_at_the_scoring_position() -> None:
    # ``boundaries[t]`` is a function of byte ``t + 1``, so routing on it at
    # position ``t`` conditions the score for byte ``t + 1`` on its own value.
    architecture = tiny_architecture()
    predicted = torch.tensor([[True, False, True, True]])
    log_probs = torch.where(
        predicted, torch.zeros(1, 4), torch.full((1, 4), -100_000.0)
    )
    for causal, expected in (
        (False, predicted),
        (True, torch.tensor([[False, True, False, True]])),
    ):
        model = _routing_probe_model(architecture, log_probs)
        model.validation_statistics(_routing_probe_batch(4), causal_routing=causal)
        assert torch.equal(model.local_decoder.routing_boundaries, expected)
        # Only routing moves: the patches themselves are pooled at the same
        # byte positions either way, so patch *contents* stay causal and
        # unchanged. A shift applied before pooling would change both.
        assert torch.equal(model.local_encoder.pooled_boundaries, predicted)


def test_causal_routing_survives_the_shifted_in_leading_non_boundary() -> None:
    # Shifting inserts ``False`` at index 0, making that position's cumsum
    # ``-1``; ``prepare_hidden`` must clamp it onto the first patch rather
    # than wrap to the last one.
    architecture = tiny_architecture()
    predicted = torch.tensor([[True, True, True]])
    log_probs = torch.zeros(1, 3)
    model = _routing_probe_model(architecture, log_probs)
    model.validation_statistics(_routing_probe_batch(3), causal_routing=True)
    routed = model.local_decoder.routing_boundaries
    assert torch.equal(routed, torch.tensor([[False, True, True]]))
    patch_ids = (routed.long().cumsum(1) - 1).clamp(min=0)
    assert torch.equal(patch_ids, torch.tensor([[0, 0, 1]]))


def test_causal_routing_makes_the_score_independent_of_the_byte_it_scores() -> None:
    # The end-to-end property, through the real local encoder: change only the
    # final byte and every earlier position's routing must be unchanged under
    # causal routing, while the non-causal rule leaks that byte backwards.
    architecture = tiny_architecture()
    length = 8

    def batch_for(final_byte: int) -> BolmoBatch:
        return BolmoBatch(
            source_ids=torch.arange(length)[None, :],
            source_valid_mask=torch.ones(1, length, dtype=torch.bool),
            byte_ids=torch.tensor(
                [[BYTE_EOT_ID] + [97, 98, 99, 100, 101, 102] + [final_byte]]
            ),
            expanded_ids=torch.zeros(1, length, dtype=torch.long),
            oracle_boundaries=torch.ones(1, length, dtype=torch.bool),
            valid_mask=torch.ones(1, length, dtype=torch.bool),
            score_mask=torch.tensor([[False] + [True] * (length - 1)]),
            patch_lens=torch.ones(1, length, dtype=torch.long),
        )

    def scored_logits(model: BolmoModel, final_byte: int, *, causal: bool) -> torch.Tensor:
        tap = _LogitTap(model.local_decoder)
        model.local_decoder = tap
        try:
            model.validation_statistics(batch_for(final_byte), causal_routing=causal)
        finally:
            model.local_decoder = tap.inner
        # ``validation_statistics`` scores ``logits[:, :-1]``; the dropped final
        # position is the only one the lookahead may legitimately reach.
        return tap.logits[:, :-1]

    # Sweep the final byte and the initialization rather than picking one of
    # each, so the test asserts the invariant itself and not where some
    # untrained model happens to put its cosine decision boundary.
    candidates = list(range(32, 224, 8))
    routing_was_live = False
    for seed in range(6):
        torch.manual_seed(seed)
        model = BolmoModel(
            global_blocks=nn.Sequential(nn.Identity()),
            global_norm=RMSNorm(16),
            source_embedding=torch.randn(10, 16),
            source_model_config={"model_dim": 16},
            architecture=architecture,
        )
        # The predictor's identity-initialized projections leave an untrained
        # tiny encoder's cosine saturated on one side of ``log 0.5``, so its
        # thresholded boundaries never move and this test would pass with
        # ``causal_routing`` doing nothing. Random projections make the
        # decision live while leaving the one-byte lookahead exactly where it
        # is. The assertions below still hold for the saturated seeds.
        with torch.no_grad():
            model.local_encoder.boundary_predictor.q_proj.weight.normal_()
            model.local_encoder.boundary_predictor.k_proj.weight.normal_()
        model.eval()
        causal = [scored_logits(model, byte, causal=True) for byte in candidates]
        free = [scored_logits(model, byte, causal=False) for byte in candidates]
        # Byte states and patch contents are causal either way, so a scored
        # logit can only move if the routing moved. Under causal routing none
        # may: bit-identical, not merely close.
        for other in causal[1:]:
            assert torch.equal(causal[0], other)
        routing_was_live |= any(
            not torch.equal(free[0], other) for other in free[1:]
        )
    # The leak this diagnostic removes, demonstrated rather than assumed: under
    # the non-causal rule the scored logits do move with the byte they score.
    assert routing_was_live


def test_causal_routing_is_refused_for_the_non_predicted_patchings() -> None:
    # The oracle, uniform and fixed-stride masks are whole-row quantities, so
    # shifting their routing by one byte would not make them codelengths. The
    # combination must fail rather than report a number that looks like one.
    architecture = tiny_architecture()
    model = _routing_probe_model(architecture, torch.zeros(1, 4))
    batch = _routing_probe_batch(4)
    for kwargs in (
        {"oracle_boundaries": True},
        {"uniform_patching": "oracle"},
        {"uniform_patching": "predicted"},
        {"fixed_stride": 2},
    ):
        try:
            model.validation_statistics(batch, causal_routing=True, **kwargs)
        except ValueError as error:
            assert "codelength" in str(error)
        else:
            raise AssertionError(f"causal routing was accepted with {kwargs}")
    # The same patchings stay available without it.
    for kwargs in (
        {"oracle_boundaries": True},
        {"uniform_patching": "oracle"},
        {"uniform_patching": "predicted"},
        {"fixed_stride": 2},
    ):
        model.validation_statistics(batch, **kwargs)


def test_prepare_hidden_clamps_the_shifted_leading_position_onto_patch_zero() -> None:
    # ``causal_routing`` makes this clamp load-bearing: the unshifted mask
    # always starts with a forced boundary, so ``cumsum - 1`` was never
    # negative before. Dropping ``min=0`` would gather at index -1.
    architecture = tiny_architecture()
    decoder = BolmoLocalDecoder(architecture)
    byte_hidden = torch.randn(1, 4, architecture.model_dim)
    patch_hidden = torch.randn(1, 3, architecture.model_dim)
    shifted = torch.tensor([[False, True, False, True]])
    hidden = decoder.prepare_hidden(byte_hidden, patch_hidden, shifted)
    normalized = decoder.patch_norm(patch_hidden)
    projected = decoder.byte_projection(byte_hidden)
    # Positions 0 and 1 read patch 0, position 2 and 3 read patch 0 and 1:
    # cumsum is [0, 1, 1, 2] and the leading -1 clamps up rather than wrapping
    # to the final patch.
    for position, patch in enumerate((0, 0, 0, 1)):
        torch.testing.assert_close(
            hidden[0, position], projected[0, position] + normalized[0, patch]
        )


def test_uniform_floor_matches_the_patch_count_of_the_arm_it_names() -> None:
    # The floor is only a controlled comparison against the arm carrying its
    # patch count. The oracle and the predictor disagree on that count on real
    # data, so a floor matched to one systematically mis-brackets the other.
    architecture = tiny_architecture()
    length = 8
    # Deliberately clustered, so a count-matched even split cannot coincide
    # with it and the assertion below stays about placement.
    predicted = torch.tensor([[True, True, True, False, False, False, False, True]])
    log_probs = torch.where(
        predicted, torch.zeros(1, length), torch.full((1, length), -100_000.0)
    )
    batch = _routing_probe_batch(length)
    oracle_count = int(batch.oracle_boundaries.sum())
    predicted_count = int(predicted.sum())
    assert oracle_count != predicted_count

    for source, expected in (
        ("oracle", oracle_count),
        ("predicted", predicted_count),
    ):
        model = _routing_probe_model(architecture, log_probs)
        model.validation_statistics(batch, uniform_patching=source)
        routed = model.local_decoder.routing_boundaries
        assert routed is not None
        assert int(routed.sum()) == expected, source
        # Placement, not count, is what the floor gives up: it must not simply
        # reproduce the mask it was counted from.
        if source == "predicted":
            assert not torch.equal(routed, predicted)

    # The count must come from the *well-formed* mask, not a raw threshold:
    # on real rows the predictor scores padding positions and can decline to
    # fire at position 0, and only `_force_well_formed_boundaries` masks the
    # first and forces the second. Without this the two differ by two here.
    valid = torch.tensor([[True] * 6 + [False] * 2])
    raw = torch.tensor([[False, False, True, False, True, False, True, True]])
    padded_log_probs = torch.where(
        raw, torch.zeros(1, length), torch.full((1, length), -100_000.0)
    )
    well_formed = BolmoModel._force_well_formed_boundaries(padded_log_probs, valid)
    assert int(well_formed.sum()) == 3
    assert int(raw.sum()) == 4
    padded = BolmoBatch(
        source_ids=torch.arange(length)[None, :],
        source_valid_mask=valid.clone(),
        byte_ids=torch.tensor([[BYTE_EOT_ID] + [97] * (length - 1)]),
        expanded_ids=torch.zeros(1, length, dtype=torch.long),
        oracle_boundaries=valid.clone(),
        valid_mask=valid.clone(),
        score_mask=torch.tensor([[False] + [True] * 5 + [False] * 2]),
        patch_lens=torch.ones(1, length, dtype=torch.long),
    )
    model = _routing_probe_model(architecture, padded_log_probs)
    model.validation_statistics(padded, uniform_patching="predicted")
    assert int(model.local_decoder.routing_boundaries.sum()) == 3


def test_uniform_patching_rejects_an_unnamed_patch_count_source() -> None:
    architecture = tiny_architecture()
    model = _routing_probe_model(architecture, torch.zeros(1, 4))
    batch = _routing_probe_batch(4)
    for bad in ("uniform", "", True):
        try:
            model.validation_statistics(batch, uniform_patching=bad)
        except ValueError as error:
            assert "names the arm" in str(error)
        else:
            raise AssertionError(f"uniform_patching={bad!r} was accepted")

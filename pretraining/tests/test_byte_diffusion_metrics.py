from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from pretraining.byte_accounting import ByteCounter
from pretraining.byte_diffusion.data import AtomicIdManifest
from pretraining.byte_diffusion.metrics import (
    ARMetricCounter,
    DiffusionMetricCounter,
    Utf8DFA,
    Utf8Role,
    Utf8ValidityCounter,
    classify_utf8_literal,
    merge_metric_states,
    noise_bucket,
)


@pytest.fixture
def manifest() -> AtomicIdManifest:
    return AtomicIdManifest.reference()


def test_ar_counter_uses_additive_atomic_byte_denominator(
    manifest: AtomicIdManifest,
) -> None:
    logits = torch.zeros(1, 4, manifest.output_size)
    targets = torch.tensor([[ord("A"), manifest.eot_id, manifest.pad_id, 17]])
    score_mask = torch.tensor([[True, True, False, False]])
    counter = ARMetricCounter()

    counter.update(logits, targets, score_mask, manifest)
    metrics = counter.compute()

    expected_bits = math.log2(manifest.output_size)
    assert metrics["ar_atomic_bytes"] == 2
    assert metrics["ar_literal_bytes"] == 1
    assert metrics["ar_special_count"] == 1
    assert metrics["ar_bpb"] == pytest.approx(expected_bits)
    assert metrics["ar_atomic_bpb"] == pytest.approx(expected_bits)
    assert metrics["ar_literal_only_bpb"] == pytest.approx(expected_bits)
    assert metrics["ar_special_bits"] == pytest.approx(expected_bits)
    assert metrics["ar_accuracy"] == 0.0


def test_atomic_bpb_denominator_matches_repository_special_byte_convention(
    manifest: AtomicIdManifest,
) -> None:
    class AsciiTokenizer:
        @staticmethod
        def decode(ids):
            return bytes(ids).decode("ascii")

    byte_counter = ByteCounter.__new__(ByteCounter)
    byte_counter.lut = None
    byte_counter.tokenizer = AsciiTokenizer()
    byte_counter.special_ids = frozenset(
        special.atomic_id for special in manifest.specials
    )
    targets = torch.tensor(
        [[ord("A"), manifest.eot_id, manifest.specials[1].atomic_id, ord("B")]]
    )
    counter = ARMetricCounter()
    counter.update(
        torch.zeros(1, 4, manifest.output_size),
        targets,
        torch.ones_like(targets, dtype=torch.bool),
        manifest,
    )

    assert byte_counter.count_ids(targets.flatten().tolist()) == 4
    assert counter.compute()["ar_atomic_bytes"] == 4
    assert counter.compute()["ar_bpb"] == pytest.approx(
        math.log2(manifest.output_size)
    )


def test_ar_counter_merge_and_state_roundtrip_match_one_pass(
    manifest: AtomicIdManifest,
) -> None:
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn(4, 3, manifest.output_size, generator=generator)
    targets = torch.tensor(
        [
            [0, 1, 256],
            [65, 194, 257],
            [128, 260, 10],
            [255, 32, 256],
        ]
    )
    score = torch.ones_like(targets, dtype=torch.bool)
    whole = ARMetricCounter()
    whole.update(logits, targets, score, manifest)

    left = ARMetricCounter()
    right = ARMetricCounter()
    left.update(logits[:2], targets[:2], score[:2], manifest)
    right.update(logits[2:], targets[2:], score[2:], manifest)
    merged = merge_metric_states((left, right))
    restored = ARMetricCounter.from_state_dict(merged.state_dict())

    assert restored.state_dict() == merged.state_dict()
    assert merged.compute() == pytest.approx(whole.compute())


def test_ar_counter_rejects_input_only_scored_target(
    manifest: AtomicIdManifest,
) -> None:
    logits = torch.zeros(1, 1, manifest.output_size)
    with pytest.raises(ValueError, match="cannot be AR targets"):
        ARMetricCounter().update(
            logits,
            torch.tensor([[manifest.pad_id]]),
            torch.tensor([[True]]),
            manifest,
        )


@pytest.mark.parametrize(
    "masked,eligible,expected",
    [
        (0, 0, "empty"),
        (0, 8, "zero"),
        (1, 8, "q1"),
        (2, 8, "q1"),
        (3, 8, "q2"),
        (4, 8, "q2"),
        (5, 8, "q3"),
        (6, 8, "q3"),
        (7, 8, "q4"),
        (8, 8, "all_mask"),
    ],
)
def test_noise_bucket_has_exact_integer_boundaries(
    masked: int, eligible: int, expected: str
) -> None:
    assert noise_bucket(masked, eligible) == expected


def test_diffusion_counter_preserves_canvas_and_token_weightings(
    manifest: AtomicIdManifest,
) -> None:
    targets = torch.tensor(
        [
            [ord("A"), ord("x"), ord("y"), ord("z")],
            [0xE2, 0x82, 0xAC, manifest.eot_id],
        ]
    )
    valid = torch.ones_like(targets, dtype=torch.bool)
    active = torch.tensor(
        [
            [True, False, False, False],
            [True, True, True, True],
        ]
    )
    logits = torch.zeros(2, 4, manifest.output_size)
    # The all-mask row is deliberately much easier.  A global masked-token
    # mean therefore differs from the equal-canvas diagnostic mean.
    for column, target in enumerate(targets[1].tolist()):
        logits[1, column, target] = 10.0

    counter = DiffusionMetricCounter()
    counter.update(logits, targets, active, valid, manifest)
    metrics = counter.compute()

    hard_nll = math.log(manifest.output_size)
    easy_losses = F.cross_entropy(logits[1], targets[1], reduction="none")
    easy_nll = float(easy_losses.mean().item())
    expected_token_nll = (hard_nll + 4 * easy_nll) / 5
    expected_canvas_nll = (hard_nll + easy_nll) / 2

    assert metrics["diffusion_active_targets"] == 5
    assert metrics["diffusion_scored_canvases"] == 2
    assert metrics["diffusion_q1_canvases"] == 1
    assert metrics["diffusion_all_mask_canvases"] == 1
    assert metrics["diffusion_nll"] == pytest.approx(expected_token_nll)
    assert metrics["diffusion_canvas_nll"] == pytest.approx(expected_canvas_nll)
    assert metrics["diffusion_nll"] != pytest.approx(
        metrics["diffusion_canvas_nll"]
    )
    assert metrics["diffusion_ascii_targets"] == 1
    assert metrics["diffusion_leading_targets"] == 1
    assert metrics["diffusion_continuation_targets"] == 2
    assert metrics["diffusion_eot_targets"] == 1


def test_diffusion_counter_merge_roundtrip_and_padding_validation(
    manifest: AtomicIdManifest,
) -> None:
    targets = torch.tensor([[1, 2, manifest.pad_id], [3, 4, manifest.pad_id]])
    logits = torch.zeros(2, 3, manifest.output_size)
    active = torch.tensor([[True, False, False], [True, True, False]])
    valid = torch.tensor([[True, True, False], [True, True, False]])

    first = DiffusionMetricCounter()
    second = DiffusionMetricCounter()
    first.update(logits[:1], targets[:1], active[:1], valid[:1], manifest)
    second.update(logits[1:], targets[1:], active[1:], valid[1:], manifest)
    merged = merge_metric_states((first, second))
    restored = DiffusionMetricCounter.from_state_dict(merged.state_dict())

    assert restored.state_dict() == merged.state_dict()
    assert restored.compute() == merged.compute()

    bad_active = active.clone()
    bad_active[0, 2] = True
    with pytest.raises(ValueError, match="must be valid"):
        DiffusionMetricCounter().update(
            logits, targets, bad_active, valid, manifest
        )


@pytest.mark.parametrize(
    "byte,role",
    [
        (0x00, Utf8Role.ASCII),
        (0x7F, Utf8Role.ASCII),
        (0xC2, Utf8Role.LEADING),
        (0xF4, Utf8Role.LEADING),
        (0x80, Utf8Role.CONTINUATION),
        (0xBF, Utf8Role.CONTINUATION),
        (0xC0, Utf8Role.INVALID_LITERAL),
        (0xF5, Utf8Role.INVALID_LITERAL),
    ],
)
def test_utf8_role_uses_clean_target_byte(byte: int, role: Utf8Role) -> None:
    assert classify_utf8_literal(byte) is role


def test_utf8_dfa_state_resumes_mid_codepoint() -> None:
    euro = "€".encode()
    uninterrupted = Utf8DFA()
    assert uninterrupted.feed_literal(euro[0])
    state = uninterrupted.state_dict()
    expected = [uninterrupted.feed_literal(byte) for byte in euro[1:]]

    resumed = Utf8DFA()
    resumed.load_state_dict(state)
    actual = [resumed.feed_literal(byte) for byte in euro[1:]]

    assert actual == expected == [True, True]
    assert resumed.accepting
    assert resumed.valid


def test_utf8_validity_counts_strict_errors_eot_and_raw_overflow(
    manifest: AtomicIdManifest,
) -> None:
    counter = Utf8ValidityCounter()

    # Valid UTF-8 stops at EOT and records raw atoms after the generated end.
    counter.update_sequence(
        (*"A€".encode(), manifest.eot_id, ord("x"), ord("y")), manifest
    )
    # Standalone continuation is invalid.
    counter.update_sequence((0x80, manifest.eot_id), manifest)
    # EOT cannot terminate a partial code point.
    counter.update_sequence((0xE2, manifest.eot_id), manifest)
    # A raw byte limit can leave a valid leading byte incomplete.
    counter.update_sequence((0xF0, 0x90), manifest)
    metrics = counter.compute()

    assert metrics["utf8_sequences"] == 4
    assert metrics["utf8_valid_sequences"] == 1
    assert metrics["utf8_valid_fraction"] == pytest.approx(0.25)
    assert metrics["utf8_invalid_transitions"] == 2
    assert metrics["utf8_incomplete_sequences"] == 1
    assert metrics["utf8_eot_mid_codepoint"] == 1
    assert metrics["utf8_discarded_post_eot_atoms"] == 2
    assert metrics["utf8_eot_count"] == 3
    assert metrics["utf8_leading_count"] == 3
    assert metrics["utf8_continuation_count"] == 4


def test_utf8_counter_merge_and_state_roundtrip(manifest: AtomicIdManifest) -> None:
    left = Utf8ValidityCounter()
    right = Utf8ValidityCounter()
    left.update_sequence((*b"hello", manifest.eot_id), manifest)
    right.update_sequence((0xED, 0xA0, 0x80), manifest)  # UTF-8 surrogate

    merged = merge_metric_states((left, right))
    restored = Utf8ValidityCounter.from_state_dict(merged.state_dict())

    assert restored.state_dict() == merged.state_dict()
    assert restored.compute() == merged.compute()
    assert restored.valid_sequences == 1
    assert restored.invalid_transitions >= 1


def test_utf8_counter_rejects_mask_and_pad(manifest: AtomicIdManifest) -> None:
    for input_only_id in (manifest.mask_id, manifest.pad_id):
        with pytest.raises(ValueError, match="not generated"):
            Utf8ValidityCounter().update_sequence((input_only_id,), manifest)

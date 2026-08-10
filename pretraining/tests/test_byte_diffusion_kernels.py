from __future__ import annotations

import math

import pytest
import torch

from pretraining.byte_diffusion import kernels
from pretraining.byte_diffusion.kernels import (
    ATOMIC_OUTPUT_SIZE,
    AttentionBackend,
    AttentionBackendCapabilities,
    AttentionPattern,
    CANVAS_REVEAL_SIZE,
    categorical_entropy_argmax_confidence,
    categorical_entropy_argmax_confidence_reference,
    categorical_sample_entropy_argmax_confidence,
    categorical_sample_entropy_argmax_confidence_reference,
    compile_static_microstep,
    reveal_low_entropy,
    reveal_low_entropy_reference,
    select_attention_backend,
)


def test_uniform_categorical_statistics_have_exact_contract() -> None:
    logits = torch.zeros(2, 3, ATOMIC_OUTPUT_SIZE, dtype=torch.bfloat16)
    entropy, argmax, confidence = categorical_entropy_argmax_confidence_reference(logits)

    assert entropy.shape == (2, 3)
    assert entropy.dtype == torch.float32
    assert argmax.shape == (2, 3)
    assert argmax.dtype == torch.int64
    assert confidence.shape == (2, 3)
    assert confidence.dtype == torch.float32
    torch.testing.assert_close(
        entropy,
        torch.full_like(entropy, math.log(ATOMIC_OUTPUT_SIZE)),
        rtol=0,
        atol=2e-6,
    )
    assert torch.equal(argmax, torch.zeros_like(argmax))
    torch.testing.assert_close(
        confidence,
        torch.full_like(confidence, 1.0 / ATOMIC_OUTPUT_SIZE),
        rtol=0,
        atol=1e-8,
    )


def test_portable_cpu_path_is_the_reference_and_preserves_gradients(monkeypatch) -> None:
    def forbidden_loader():
        raise AssertionError("CPU auto dispatch must not import Triton")

    monkeypatch.setattr(kernels, "_load_triton_categorical_kernel", forbidden_loader)
    logits = torch.randn(4, ATOMIC_OUTPUT_SIZE, requires_grad=True)
    expected = categorical_entropy_argmax_confidence_reference(logits)
    actual = categorical_entropy_argmax_confidence(logits)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    actual[0].sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_reference_argmax_uses_the_same_fp32_values_as_probabilities() -> None:
    logits = torch.zeros(ATOMIC_OUTPUT_SIZE, dtype=torch.float64)
    logits[0] = 1.0
    logits[1] = 1.0 + 1e-12
    _, argmax, _ = categorical_entropy_argmax_confidence_reference(logits)
    assert argmax.item() == 0


@pytest.mark.parametrize(
    "shape",
    [(2, 260), (2, 262), (2, 3, 17)],
)
def test_categorical_statistics_reject_wrong_vocabulary_shape(shape) -> None:
    with pytest.raises(ValueError, match="final dimension 261"):
        categorical_entropy_argmax_confidence(torch.zeros(shape))


def test_categorical_statistics_reject_non_floating_logits() -> None:
    with pytest.raises(TypeError, match="floating point"):
        categorical_entropy_argmax_confidence(
            torch.zeros(2, ATOMIC_OUTPUT_SIZE, dtype=torch.long)
        )


def test_strict_triton_request_fails_before_lazy_load_on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(kernels, "triton_is_importable", lambda: True)
    monkeypatch.setattr(
        kernels,
        "_load_triton_categorical_kernel",
        lambda: (_ for _ in ()).throw(AssertionError("must reject CPU first")),
    )
    with pytest.raises(RuntimeError, match="requires CUDA"):
        categorical_entropy_argmax_confidence(
            torch.zeros(1, ATOMIC_OUTPUT_SIZE), backend="triton"
        )


def test_categorical_sample_matches_explicit_inverse_cdf() -> None:
    generator = torch.Generator().manual_seed(19)
    logits = torch.randn(3, 2, ATOMIC_OUTPUT_SIZE, generator=generator)
    uniforms = torch.tensor(
        [[0.0, 0.125], [0.5, 0.875], [0.999, 0.333]], dtype=torch.float32
    )
    sample, entropy, argmax, confidence = (
        categorical_sample_entropy_argmax_confidence_reference(logits, uniforms)
    )
    probabilities = logits.float().softmax(-1)
    expected_sample = torch.searchsorted(
        probabilities.cumsum(-1).contiguous(),
        uniforms.unsqueeze(-1).contiguous(),
        right=False,
    ).squeeze(-1)
    expected_sample.clamp_max_(ATOMIC_OUTPUT_SIZE - 1)
    expected_statistics = categorical_entropy_argmax_confidence_reference(logits)
    assert torch.equal(sample, expected_sample)
    for actual, expected in zip(
        (entropy, argmax, confidence), expected_statistics
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_categorical_sample_uniform_extremes_and_low_id_ties() -> None:
    logits = torch.zeros(2, ATOMIC_OUTPUT_SIZE)
    uniforms = torch.tensor([0.0, torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))])
    sample, entropy, argmax, confidence = (
        categorical_sample_entropy_argmax_confidence(logits, uniforms)
    )
    assert sample.tolist() == [0, ATOMIC_OUTPUT_SIZE - 1]
    assert argmax.tolist() == [0, 0]
    torch.testing.assert_close(
        entropy, torch.full_like(entropy, math.log(ATOMIC_OUTPUT_SIZE)), atol=2e-6, rtol=0
    )
    torch.testing.assert_close(
        confidence,
        torch.full_like(confidence, 1 / ATOMIC_OUTPUT_SIZE),
        atol=1e-8,
        rtol=0,
    )


def test_categorical_sample_validates_uniform_structure() -> None:
    logits = torch.zeros(2, ATOMIC_OUTPUT_SIZE)
    with pytest.raises(ValueError, match="leading logits shape"):
        categorical_sample_entropy_argmax_confidence(logits, torch.zeros(3))
    with pytest.raises(TypeError, match="floating point"):
        categorical_sample_entropy_argmax_confidence(
            logits, torch.zeros(2, dtype=torch.long)
        )


def test_strict_triton_sample_request_fails_closed_on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(kernels, "triton_is_importable", lambda: True)
    monkeypatch.setattr(
        kernels,
        "_load_triton_categorical_sample_kernel",
        lambda: (_ for _ in ()).throw(AssertionError("must reject CPU first")),
    )
    with pytest.raises(RuntimeError, match="requires CUDA"):
        categorical_sample_entropy_argmax_confidence(
            torch.zeros(1, ATOMIC_OUTPUT_SIZE),
            torch.zeros(1),
            backend="triton",
        )


def _reveal_inputs(batch: int = 1):
    canvas = torch.full((batch, CANVAS_REVEAL_SIZE), 261, dtype=torch.int64)
    samples = torch.arange(CANVAS_REVEAL_SIZE, dtype=torch.int64).remainder(
        ATOMIC_OUTPUT_SIZE
    )[None].expand(batch, -1).clone()
    entropy = torch.arange(CANVAS_REVEAL_SIZE, dtype=torch.float32)[None].expand(
        batch, -1
    ).clone()
    unresolved = torch.zeros(batch, CANVAS_REVEAL_SIZE, dtype=torch.bool)
    active = torch.ones_like(unresolved)
    return canvas, samples, entropy, unresolved, active


def test_reveal_is_stable_by_entropy_then_position_with_runtime_row_quotas() -> None:
    canvas, samples, entropy, unresolved, active = _reveal_inputs(batch=2)
    unresolved[:, [1, 3, 7, 9]] = True
    entropy[:, [1, 3, 7, 9]] = 0.25
    quota = torch.tensor([2, 3])
    updated, remaining, updated_active, revealed = reveal_low_entropy_reference(
        canvas,
        samples,
        entropy,
        unresolved,
        active,
        quota,
        eot_id=256,
    )
    assert revealed[0].nonzero().flatten().tolist() == [1, 3]
    assert revealed[1].nonzero().flatten().tolist() == [1, 3, 7]
    assert remaining[0].nonzero().flatten().tolist() == [7, 9]
    assert remaining[1].nonzero().flatten().tolist() == [9]
    assert torch.equal(updated[revealed], samples[revealed])
    assert updated_active.all()


@pytest.mark.parametrize("quota", [torch.tensor([2**40]), 10**100])
def test_reveal_clamps_quota_before_integer_narrowing(quota) -> None:
    canvas, samples, entropy, unresolved, active = _reveal_inputs()
    unresolved[0, [1, 3, 5]] = True
    _, remaining, _, revealed = reveal_low_entropy_reference(
        canvas,
        samples,
        entropy,
        unresolved,
        active,
        quota=quota,
        eot_id=256,
    )
    assert revealed[0].nonzero().flatten().tolist() == [1, 3, 5]
    assert not remaining.any()


def test_reveal_eot_waits_for_resolved_prefix_then_deactivates_suffix() -> None:
    canvas, samples, entropy, unresolved, active = _reveal_inputs()
    samples[0, 4] = 256
    unresolved[0, [2, 4, 6]] = True
    entropy[0, [2, 4, 6]] = torch.tensor([0.2, 0.1, 0.0])

    first = reveal_low_entropy_reference(
        canvas,
        samples,
        entropy,
        unresolved,
        active,
        quota=2,
        eot_id=256,
    )
    first_canvas, first_unresolved, first_active, first_revealed = first
    # EOT is lower entropy than position 2 but remains gated while its prefix
    # has an unresolved active slot. Positions 6 and 2 are revealed instead.
    assert first_revealed[0].nonzero().flatten().tolist() == [2, 6]
    assert first_unresolved[0].nonzero().flatten().tolist() == [4]
    assert first_active.all()

    second = reveal_low_entropy_reference(
        first_canvas,
        samples,
        entropy,
        first_unresolved,
        first_active,
        quota=torch.tensor(1),
        eot_id=256,
    )
    second_canvas, second_unresolved, second_active, second_revealed = second
    assert second_revealed[0].nonzero().flatten().tolist() == [4]
    assert second_canvas[0, 4].item() == 256
    assert second_active[0, :5].all()
    assert not second_active[0, 5:].any()
    assert not second_unresolved.any()


def test_reveal_drops_same_step_suffix_selection_after_eot() -> None:
    canvas, samples, entropy, unresolved, active = _reveal_inputs()
    samples[0, 2] = 256
    unresolved[0, [2, 7]] = True
    entropy[0, [2, 7]] = torch.tensor([0.2, 0.1])
    updated, remaining, updated_active, revealed = reveal_low_entropy_reference(
        canvas,
        samples,
        entropy,
        unresolved,
        active,
        quota=2,
        eot_id=256,
    )
    assert revealed[0].nonzero().flatten().tolist() == [2]
    assert updated[0, 7] == canvas[0, 7]
    assert not updated_active[0, 3:].any()
    assert not remaining.any()


def test_reveal_excludes_inactive_resolved_and_nonfinite_entropy() -> None:
    canvas, samples, entropy, unresolved, active = _reveal_inputs()
    unresolved[0, [1, 2, 3, 4, 5]] = True
    active[0, 2] = False
    entropy[0, 3] = torch.nan
    entropy[0, 4] = torch.inf
    entropy[0, 5] = -torch.inf
    _, remaining, _, revealed = reveal_low_entropy(
        canvas,
        samples,
        entropy,
        unresolved,
        active,
        quota=4,
        eot_id=256,
    )
    assert revealed[0].nonzero().flatten().tolist() == [1]
    assert remaining[0].nonzero().flatten().tolist() == [3, 4, 5]


def test_reveal_rejects_boolean_quota() -> None:
    canvas, samples, entropy, unresolved, active = _reveal_inputs()
    with pytest.raises(TypeError, match="quota must be an int"):
        reveal_low_entropy_reference(
            canvas,
            samples,
            entropy,
            unresolved,
            active,
            quota=True,
            eot_id=256,
        )


def test_reveal_supports_empty_batches() -> None:
    canvas, samples, entropy, unresolved, active = _reveal_inputs(batch=0)
    actual = reveal_low_entropy_reference(
        canvas,
        samples,
        entropy,
        unresolved,
        active,
        quota=torch.empty(0, dtype=torch.int64),
        eot_id=256,
    )
    assert all(tensor.shape == canvas.shape for tensor in actual)


def test_reveal_accepts_noncontiguous_inputs_with_contiguous_outputs() -> None:
    shape = (CANVAS_REVEAL_SIZE, 2)
    canvas = torch.full(shape, 261, dtype=torch.int64).transpose(0, 1)
    samples = torch.zeros(shape, dtype=torch.int64).transpose(0, 1)
    entropy = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(
        shape
    ).transpose(0, 1)
    unresolved = torch.ones(shape, dtype=torch.bool).transpose(0, 1)
    active = torch.ones(shape, dtype=torch.bool).transpose(0, 1)
    assert not canvas.is_contiguous()

    updated, remaining, updated_active, revealed = reveal_low_entropy(
        canvas,
        samples,
        entropy,
        unresolved,
        active,
        quota=torch.tensor([2, 3]),
        eot_id=256,
    )
    assert all(
        tensor.is_contiguous()
        for tensor in (updated, remaining, updated_active, revealed)
    )
    assert revealed.sum(dim=-1).tolist() == [2, 3]


def test_reveal_supports_blt_widths_and_strict_triton_fails_on_cpu(
    monkeypatch,
) -> None:
    canvas, samples, entropy, unresolved, active = _reveal_inputs()
    narrowed = reveal_low_entropy(
        canvas[:, :4],
        samples[:, :4],
        entropy[:, :4],
        unresolved[:, :4],
        active[:, :4],
        quota=1,
        eot_id=256,
    )
    assert all(tensor.shape == (canvas.shape[0], 4) for tensor in narrowed)
    monkeypatch.setattr(kernels, "triton_is_importable", lambda: True)
    monkeypatch.setattr(
        kernels,
        "_load_triton_reveal_kernel",
        lambda: (_ for _ in ()).throw(AssertionError("must reject CPU first")),
    )
    with pytest.raises(RuntimeError, match="requires CUDA"):
        reveal_low_entropy(
            canvas,
            samples,
            entropy,
            unresolved,
            active,
            quota=1,
            eot_id=256,
            backend="triton",
        )


CUDA_CAPABILITIES = AttentionBackendCapabilities(
    cuda_runtime=True,
    varlen_flash=True,
    flash_sdpa=True,
    flex_attention=True,
)


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        (AttentionPattern.CAUSAL, AttentionBackend.VARLEN_FLASH),
        (AttentionPattern.CAUSAL_WINDOW, AttentionBackend.VARLEN_FLASH),
        (AttentionPattern.BIDIRECTIONAL, AttentionBackend.FLASH_SDPA),
        (AttentionPattern.BRANCH, AttentionBackend.FLEX),
        (AttentionPattern.INTROSPECTION, AttentionBackend.FLEX),
    ],
)
def test_cuda_backend_selection_is_pattern_specific(pattern, expected) -> None:
    assert (
        select_attention_backend(
            pattern,
            device="cuda",
            sequence_length=8192,
            capabilities=CUDA_CAPABILITIES,
        )
        is expected
    )


def test_dense_reference_requires_explicit_small_tensor_opt_in() -> None:
    unavailable = AttentionBackendCapabilities(False, False, False, False)
    with pytest.raises(RuntimeError, match="not explicitly enabled"):
        select_attention_backend(
            AttentionPattern.BRANCH,
            device="cpu",
            sequence_length=32,
            capabilities=unavailable,
        )
    assert (
        select_attention_backend(
            AttentionPattern.BRANCH,
            device="cpu",
            sequence_length=32,
            capabilities=unavailable,
            allow_dense_reference=True,
        )
        is AttentionBackend.DENSE_REFERENCE
    )


def test_missing_sparse_backend_never_falls_back_dense_for_long_mask() -> None:
    unavailable = AttentionBackendCapabilities(True, False, False, False)
    with pytest.raises(RuntimeError, match="refusing dense attention at length 8192"):
        select_attention_backend(
            AttentionPattern.CAUSAL_WINDOW,
            device="cuda",
            sequence_length=8192,
            capabilities=unavailable,
            allow_dense_reference=True,
        )


def test_compile_static_microstep_uses_fixed_shapes_and_safe_default(monkeypatch) -> None:
    captured = {}

    def fake_compile(function, **kwargs):
        captured.update(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    function = lambda value: value + 1
    assert compile_static_microstep(function) is function
    assert captured == {
        "fullgraph": True,
        "dynamic": False,
        "mode": "max-autotune-no-cudagraphs",
        "backend": None,
        "name": None,
        "disable": False,
    }


def test_compile_static_microstep_requires_explicit_cuda_graph_intent(monkeypatch) -> None:
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    with pytest.raises(ValueError, match="pass cuda_graphs=True"):
        compile_static_microstep(lambda: None, mode="max-autotune")
    with pytest.raises(ValueError, match="conflicts"):
        compile_static_microstep(
            lambda: None,
            cuda_graphs=True,
            mode="max-autotune-no-cudagraphs",
        )

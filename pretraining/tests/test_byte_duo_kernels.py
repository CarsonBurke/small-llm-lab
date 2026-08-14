"""CPU semantic and dispatch tests for the fused Byte-Duo sampler."""

from __future__ import annotations

import pytest
import torch

from pretraining.byte_diffusion import duo_kernels
from pretraining.byte_diffusion.duo import duo_reverse_posterior
from pretraining.byte_diffusion.duo_kernels import (
    DUO_CLEAN_ATOMS,
    duo_posterior_sample,
    duo_posterior_sample_reference,
)


def _inputs(batch: int = 2, canvas: int = 5):
    generator = torch.Generator().manual_seed(211)
    logits = torch.randn(
        batch, canvas, DUO_CLEAN_ATOMS, generator=generator, dtype=torch.float64
    )
    noisy = torch.randint(
        DUO_CLEAN_ATOMS, (batch, canvas), generator=generator
    )
    uniforms = torch.rand(batch, canvas, generator=generator)
    active = torch.ones(batch, canvas, dtype=torch.bool)
    return logits, noisy, uniforms, active


def test_fp32_reference_matches_explicit_exact_posterior_inverse_cdf() -> None:
    logits, noisy, uniforms, active = _inputs()
    alpha_s = torch.tensor([0.81, 0.53])
    alpha_t = torch.tensor([0.27, 0.11])
    probabilities = logits.float().softmax(-1)
    posterior = duo_reverse_posterior(
        probabilities, noisy, alpha_s, alpha_t
    )
    expected = (posterior.cumsum(-1) <= uniforms[..., None]).sum(-1)
    expected.clamp_max_(DUO_CLEAN_ATOMS - 1)

    observed = duo_posterior_sample_reference(
        logits, noisy, alpha_s, alpha_t, uniforms, active
    )

    assert observed.dtype == torch.int64
    assert torch.equal(observed, expected)


def test_float64_oracle_matches_explicit_float64_posterior() -> None:
    logits, noisy, uniforms, active = _inputs()
    probabilities = logits.double().softmax(-1)
    posterior = duo_reverse_posterior(
        probabilities, noisy, 0.77, 0.09, use_float64=True
    )
    expected = (posterior.cumsum(-1) <= uniforms.double()[..., None]).sum(-1)
    expected.clamp_max_(DUO_CLEAN_ATOMS - 1)

    observed = duo_posterior_sample_reference(
        logits,
        noisy,
        0.77,
        0.09,
        uniforms,
        active,
        use_float64=True,
    )

    assert torch.equal(observed, expected)


def test_identity_transition_and_inactive_private_pad_retain_current_ids() -> None:
    logits, noisy, uniforms, active = _inputs(batch=1, canvas=6)
    active[0, [1, 4]] = False
    noisy[0, [1, 4]] = DUO_CLEAN_ATOMS  # private PAD is not a posterior state
    uniforms.zero_()

    observed = duo_posterior_sample(
        logits, noisy, 0.4, 0.4, uniforms, active, backend="torch"
    )

    assert torch.equal(observed, noisy)


def test_inverse_cdf_skips_zero_mass_prefix_at_uniform_zero() -> None:
    logits = torch.zeros(1, 3, DUO_CLEAN_ATOMS, dtype=torch.float64)
    noisy = torch.tensor([[17, 0, DUO_CLEAN_ATOMS]])
    uniforms = torch.tensor([[0.0, 0.0, 0.0]])
    active = torch.tensor([[True, True, False]])

    # An identity transition is a point mass at the observed state. The
    # first-strictly-greater CDF convention must skip the exact-zero prefix
    # for id 17; an inclusive-boundary implementation incorrectly returns 0.
    fp32 = duo_posterior_sample_reference(
        logits, noisy, 0.4, 0.4, uniforms, active
    )
    fp64 = duo_posterior_sample_reference(
        logits,
        noisy,
        0.4,
        0.4,
        uniforms,
        active,
        use_float64=True,
    )
    expected = torch.tensor([[17, 0, DUO_CLEAN_ATOMS]])
    assert torch.equal(fp32, expected)
    assert torch.equal(fp64, expected)


def test_inverse_cdf_upper_endpoint_guard_selects_last_positive_class() -> None:
    logits = torch.zeros(1, 1, DUO_CLEAN_ATOMS)
    noisy = torch.tensor([[DUO_CLEAN_ATOMS - 1]])
    uniforms = torch.nextafter(torch.ones(1, 1), torch.zeros(1, 1))
    active = torch.ones_like(noisy, dtype=torch.bool)

    observed = duo_posterior_sample_reference(
        logits, noisy, 0.4, 0.4, uniforms, active
    )
    assert observed.item() == DUO_CLEAN_ATOMS - 1


def test_caller_owned_uniforms_make_sampling_repeatable() -> None:
    logits, noisy, uniforms, active = _inputs()
    first = duo_posterior_sample(
        logits, noisy, 0.73, 0.17, uniforms, active
    )
    second = duo_posterior_sample(
        logits, noisy, 0.73, 0.17, uniforms.clone(), active
    )
    assert torch.equal(first, second)


@pytest.mark.parametrize(
    ("alpha_s", "alpha_t"),
    [
        (0.0, 0.0),
        (0.4, -0.1),
        (0.4, 0.5),
        (1.1, 0.2),
        (float("nan"), 0.2),
        (0.5, float("inf")),
    ],
)
def test_reverse_schedule_endpoints_fail_closed(alpha_s: float, alpha_t: float) -> None:
    logits, noisy, uniforms, active = _inputs(batch=1, canvas=1)
    with pytest.raises(ValueError, match="reverse"):
        duo_posterior_sample(
            logits, noisy, alpha_s, alpha_t, uniforms, active
        )


def test_per_token_schedule_is_supported_by_reference() -> None:
    logits, noisy, uniforms, active = _inputs()
    alpha_s = torch.full(noisy.shape, 0.7)
    alpha_t = torch.linspace(0.05, 0.6, noisy.numel()).reshape_as(noisy)
    observed = duo_posterior_sample(
        logits,
        noisy,
        alpha_s,
        alpha_t,
        uniforms,
        active,
        backend="torch",
    )
    assert observed.shape == noisy.shape
    assert bool(((observed >= 0) & (observed < DUO_CLEAN_ATOMS)).all())


def test_cpu_auto_dispatch_never_loads_triton(monkeypatch) -> None:
    monkeypatch.setattr(
        duo_kernels,
        "_load_triton_duo_posterior_kernel",
        lambda: (_ for _ in ()).throw(AssertionError("CPU must remain lazy")),
    )
    logits, noisy, uniforms, active = _inputs()
    expected = duo_posterior_sample_reference(
        logits, noisy, 0.7, 0.1, uniforms, active
    )
    observed = duo_posterior_sample(
        logits, noisy, 0.7, 0.1, uniforms, active
    )
    assert torch.equal(observed, expected)


def test_explicit_triton_request_rejects_cpu_before_lazy_load(monkeypatch) -> None:
    monkeypatch.setattr(duo_kernels, "triton_is_importable", lambda: True)
    monkeypatch.setattr(
        duo_kernels,
        "_load_triton_duo_posterior_kernel",
        lambda: (_ for _ in ()).throw(AssertionError("CPU must fail first")),
    )
    logits, noisy, uniforms, active = _inputs()
    with pytest.raises(RuntimeError, match="requires CUDA"):
        duo_posterior_sample(
            logits,
            noisy,
            0.7,
            0.1,
            uniforms,
            active,
            backend="triton",
        )


def test_explicit_triton_request_rejects_float64_oracle() -> None:
    logits, noisy, uniforms, active = _inputs()
    with pytest.raises(RuntimeError, match="FP64"):
        duo_posterior_sample(
            logits,
            noisy,
            0.7,
            0.1,
            uniforms,
            active,
            backend="triton",
            use_float64=True,
        )


def test_fused_kernel_aot_compiles_for_rtx5090_without_device_execution() -> None:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, make_backend

    kernel = duo_kernels._load_triton_duo_posterior_kernel()
    target = GPUTarget("cuda", 120, 32)
    compiler_backend = make_backend(target)
    options = compiler_backend.parse_options({"num_warps": 4, "num_stages": 1})
    source = ASTSource(
        fn=kernel,
        signature={
            "logits": "*bf16",
            "noisy_ids": "*i64",
            "uniforms": "*fp32",
            "active": "*i1",
            "sampled_out": "*i64",
            "alpha_s": "fp32",
            "alpha_t": "fp32",
            "VOCAB_SIZE": "constexpr",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"VOCAB_SIZE": DUO_CLEAN_ATOMS, "BLOCK_SIZE": 512},
    )
    compiled = triton.compile(source, target=target, options=options.__dict__)
    assert compiled.metadata.global_scratch_size == 0


def test_input_contract_rejects_wrong_shape_and_dtype() -> None:
    logits, noisy, uniforms, active = _inputs()
    with pytest.raises(ValueError, match="261"):
        duo_posterior_sample(
            logits[..., :-1], noisy, 0.7, 0.1, uniforms, active
        )
    with pytest.raises(TypeError, match="int64"):
        duo_posterior_sample(
            logits, noisy.int(), 0.7, 0.1, uniforms, active
        )
    with pytest.raises(ValueError, match="unknown"):
        duo_posterior_sample(
            logits,
            noisy,
            0.7,
            0.1,
            uniforms,
            active,
            backend="magic",  # type: ignore[arg-type]
        )

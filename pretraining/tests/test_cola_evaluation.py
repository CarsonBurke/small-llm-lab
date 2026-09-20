"""Analytic ODE contracts plus real-model CUDA tests; queue GPU tests through mlq.

The analytic tests run only tensor arithmetic, never a neural CPU fallback.
No test or model workload is run as part of authoring this module.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from pretraining.cola.evaluation import (
    estimate_negative_elbo,
    generate_tokens,
    integrate_log_density,
)

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required; queue with mlq"
)


def normal_log_prob(value):
    return -0.5 * (value.square() + math.log(2 * math.pi)).flatten(1).sum(1)


def test_linear_conditional_density_has_positive_divergence_sign():
    initial = torch.tensor([[[0.2, -0.4], [0.7, 0.1]], [[-0.3, 0.6], [0.2, -0.8]]])
    history = torch.tensor([[[0.8, 0.3], [-0.4, 0.2]], [[0.1, 0.2], [-0.5, 0.3]]])
    rates = torch.tensor([0.2, 0.5])
    dimensions = initial[0].numel()
    # A scaled orthogonal probe set gives an exact trace for this analytic check.
    probes = math.sqrt(dimensions) * torch.eye(dimensions).reshape(dimensions, 1, 2, 2)
    probes = probes.expand(-1, initial.shape[0], -1, -1)
    integrated = integrate_log_density(
        lambda clean, state, times: rates * (state - clean),
        history,
        initial,
        steps=64,
        probe_vectors=probes,
    )
    terminal = history + (initial - history) * rates.exp()
    exact_trace = torch.full((initial.shape[0],), initial.shape[1] * float(rates.sum()))
    torch.testing.assert_close(integrated["state"], terminal, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(
        integrated["divergence_integral"], exact_trace, rtol=2e-6, atol=2e-6
    )
    torch.testing.assert_close(
        integrated["log_prob"],
        normal_log_prob(terminal) + exact_trace,
        rtol=3e-5,
        atol=3e-5,
    )
    assert integrated["nfe"] == 128


def test_normalized_time_fixed_history_and_fixed_gaussian_probes():
    initial = torch.tensor([[[0.4, -0.2], [0.8, 0.1]]], requires_grad=True)
    history = torch.tensor([[[0.2, 0.9], [-0.1, 0.5]]], requires_grad=True)
    generator = torch.Generator().manual_seed(728)
    probes = torch.randn((3, *initial.shape), generator=generator)
    rng_before = torch.random.get_rng_state()
    # dz/ds = s (z - clean): z(1) = clean + exp(1/2) (z(0) - clean).
    # A changing history, time embedding scale of 1000, or resampled probes
    # changes this closed-form answer. Outer no_grad must not disable the trace.
    with torch.no_grad():
        integrated = integrate_log_density(
            lambda clean, state, times: times[..., None] * (state - clean),
            history,
            initial,
            steps=128,
            probe_vectors=probes,
        )
    terminal = history.detach() + math.exp(0.5) * (initial.detach() - history.detach())
    sampled_trace = 0.5 * probes.square().flatten(2).sum(2).mean(0)
    torch.testing.assert_close(integrated["state"], terminal, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(
        integrated["divergence_integral"], sampled_trace, rtol=2e-6, atol=2e-6
    )
    torch.testing.assert_close(
        integrated["log_prob"],
        normal_log_prob(terminal) + sampled_trace,
        rtol=3e-5,
        atol=3e-5,
    )
    assert integrated["state"].grad_fn is None
    assert integrated["log_prob"].grad_fn is None
    assert integrated["state"].dtype == torch.float32
    assert initial.grad is None and history.grad is None
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_state_independent_velocity_has_zero_divergence():
    initial = torch.tensor([[[0.3, -0.7]]])
    history = torch.tensor([[[0.1, 0.4]]])
    probes = torch.tensor([[[[0.4, -1.2]]], [[[0.8, 0.5]]]])
    integrated = integrate_log_density(
        lambda clean, state, times: clean + times[..., None],
        history,
        initial,
        steps=4,
        probe_vectors=probes,
    )
    terminal = initial + history + 0.5
    torch.testing.assert_close(integrated["state"], terminal)
    torch.testing.assert_close(integrated["divergence_integral"], torch.zeros(1))
    torch.testing.assert_close(integrated["log_prob"], normal_log_prob(terminal))


def test_density_rejects_inference_mode_instead_of_silently_dropping_trace():
    with torch.inference_mode(), pytest.raises(RuntimeError, match="input gradients"):
        integrate_log_density(
            lambda clean, state, times: state,
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
            steps=2,
            probe_vectors=torch.ones(1, 1, 1, 1),
        )


@pytest.fixture(scope="module", params=[256, 100278], ids=["byte", "bpe"])
def cuda_model(request):
    from pretraining.cola.config import ModelConfig
    from pretraining.cola.model import ColaModel

    # Independent architecture fixtures must not exhaust each other's per-code
    # Dynamo cache; production runs contain one vocabulary/model configuration.
    torch._dynamo.reset()
    device = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(418)
        model = (
            ColaModel(
                ModelConfig(
                    vocab_size=request.param,
                    latent_dim=4,
                    vae_dim=32,
                    vae_layers=1,
                    vae_heads=4,
                    vae_ffn_dim=64,
                    dit_dim=32,
                    dit_layers=2,
                    dit_heads=2,
                    dit_mlp_ratio=2,
                    dit_rope_dim=8,
                    block_size=4,
                    seq_len=16,
                )
            )
            .cuda()
            .eval()
        )
        # Exercise input-dependent velocity and reconstruction, not the
        # zero-initialized identity ODE / uniform token readout.
        with torch.no_grad():
            for projection in [
                *(block.modulation for block in model.prior.blocks),
                model.prior.final_modulation,
                model.prior.head,
                model.vae.decoder.head,
            ]:
                projection.weight.normal_(std=0.02)
        model.compile_components()
    return model


@cuda_required
@pytest.mark.parametrize("prompt_length", [0, 3, 4])
def test_generation_preserves_prefix_accounts_real_work_and_isolates_rng(
    cuda_model, prompt_length
):
    prompt = [0, cuda_model.config.vocab_size - 1, 128, 16][:prompt_length]
    new_tokens, steps = 5, 2
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    first = generate_tokens(
        cuda_model, prompt, new_tokens=new_tokens, steps=steps, seed=51
    )
    second = generate_tokens(
        cuda_model, prompt, new_tokens=new_tokens, steps=steps, seed=51
    )
    output = first["output_ids"]
    suffix = first["generated_ids"]
    assert output == prompt + suffix
    assert first["prompt_ids"] == prompt
    assert len(suffix) == first["generated_count"] == new_tokens
    assert all(0 <= identity < cuda_model.config.vocab_size for identity in suffix)
    assert first["total_tokens"] == len(prompt) + new_tokens
    assert first["output_ids"] == second["output_ids"]
    block_size = cuda_model.config.block_size
    blocks = math.ceil((len(prompt) % block_size + new_tokens) / block_size)
    appends = len(prompt) // block_size + blocks - 1
    assert first["nfe"] == first["prior_velocity_calls"] == blocks * 2 * steps
    assert first["prior_append_calls"] == appends
    assert first["prior_calls"] == blocks * 2 * steps + appends
    assert first["prior_positions"] == first["prior_calls"] * block_size
    assert first["encoder_positions"] == len(prompt)
    complete_prefix = len(prompt) // block_size * block_size
    assert first["decoder_positions"] == sum(
        min(complete_prefix + block * block_size, len(output))
        for block in range(1, blocks + 1)
    )
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)


@cuda_required
def test_zero_generation_preserves_exact_token_prefix_without_prefill(cuda_model):
    prompt = [cuda_model.config.vocab_size - 1, 0, 254]
    result = generate_tokens(cuda_model, prompt, new_tokens=0)
    assert result["output_ids"] == prompt
    assert result["generated_count"] == 0
    assert result["generated_ids"] == []
    assert (
        result["prior_calls"]
        == result["prior_positions"]
        == result["decoder_positions"]
        == 0
    )
    assert result["encoder_positions"] == 0
    assert result["elapsed_seconds"] == 0


@cuda_required
def test_generation_rejects_context_overflow_instead_of_truncating_prompt(cuda_model):
    with pytest.raises(ValueError, match="exceeds"):
        generate_tokens(cuda_model, [0] * cuda_model.config.seq_len, new_tokens=1)


@cuda_required
def test_elbo_uses_one_posterior_sample_separates_units_and_restores_training_state(
    cuda_model,
):
    ids = torch.arange(32, device="cuda").reshape(2, 16)
    ids[0, -1] = cuda_model.config.vocab_size - 1
    source_bytes = 97
    seed = 931
    generator = torch.Generator(device=ids.device).manual_seed(seed)
    with torch.no_grad():
        mean, logvar = cuda_model.vae.encode(ids)
        std = (0.5 * logvar.float()).exp()
        sample = mean.float() + std * torch.randn(
            mean.shape, generator=generator, device=ids.device, dtype=torch.float32
        )
        expected_logq = (
            torch.distributions.Normal(mean.float(), std).log_prob(sample).sum()
        )
        expected_reconstruction = F.cross_entropy(
            cuda_model.vae.decode(sample).reshape(-1, cuda_model.config.vocab_size),
            ids.reshape(-1),
            reduction="sum",
        )
    original_parameter_flags = [
        (parameter, parameter.requires_grad) for parameter in cuda_model.parameters()
    ]
    original_module_flags = [
        (module, module.training) for module in cuda_model.modules()
    ]
    try:
        cuda_model.train()
        cuda_model.vae.decoder.eval()
        next(cuda_model.parameters()).requires_grad_(False)
        parameter_flags = [
            parameter.requires_grad for parameter in cuda_model.parameters()
        ]
        module_flags = [module.training for module in cuda_model.modules()]
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        result = estimate_negative_elbo(
            cuda_model, ids, steps=2, seed=seed, probes=2, source_bytes=source_bytes
        )
        repeated = estimate_negative_elbo(
            cuda_model, ids, steps=2, seed=seed, probes=2, source_bytes=source_bytes
        )
        assert result == repeated
        assert [
            parameter.requires_grad for parameter in cuda_model.parameters()
        ] == parameter_flags
        assert [module.training for module in cuda_model.modules()] == module_flags
        assert torch.equal(torch.random.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    finally:
        for parameter, requires_grad in original_parameter_flags:
            parameter.requires_grad_(requires_grad)
        for module, training in original_module_flags:
            module.training = training
    assert result["bytes"] == source_bytes
    assert result["positions"] == ids.numel()
    assert result["reconstruction_nll_nats"] == pytest.approx(
        float(expected_reconstruction), rel=1e-5
    )
    assert result["posterior_logprob_nats"] == pytest.approx(
        float(expected_logq), rel=1e-5
    )
    denominator = source_bytes
    assert result["reconstruction_bits_per_byte"] == pytest.approx(
        float(expected_reconstruction) / (denominator * math.log(2)), rel=1e-5
    )
    assert result["prior_nats_per_byte"] == -result["prior_logprob_nats"] / denominator
    assert result["posterior_logprob_nats_per_byte"] == pytest.approx(
        float(expected_logq) / denominator, rel=1e-5
    )
    assert result["negative_elbo_estimate_bpb"] == pytest.approx(
        (
            result["reconstruction_bits_per_byte"] * math.log(2)
            + result["prior_nats_per_byte"]
            + result["posterior_logprob_nats_per_byte"]
        )
        / math.log(2),
        rel=1e-5,
    )
    assert result["nfe"] == 4
    assert result["prior_positions"] == 4 * 2 * ids.numel()
    assert math.isfinite(result["negative_elbo_estimate_bpb"])


@pytest.mark.parametrize("source_bytes", [0, -1, True, 1.5])
def test_elbo_rejects_invalid_raw_byte_denominator(source_bytes):
    with pytest.raises(ValueError, match="source_bytes"):
        estimate_negative_elbo(
            None,
            torch.zeros(1, 1, dtype=torch.long),
            steps=1,
            seed=1,
            source_bytes=source_bytes,
        )


@cuda_required
def test_evaluation_and_generation_reject_mask_id_as_content(cuda_model):
    ids = torch.zeros((1, cuda_model.config.seq_len), device="cuda", dtype=torch.long)
    ids[0, -1] = cuda_model.config.vocab_size
    with pytest.raises(ValueError, match="vocabulary"):
        estimate_negative_elbo(
            cuda_model, ids, steps=1, seed=1, source_bytes=ids.numel()
        )
    with pytest.raises(ValueError, match="vocabulary"):
        generate_tokens(cuda_model, [cuda_model.config.vocab_size], new_tokens=0)


@cuda_required
def test_partial_prefix_clamps_both_heun_stages_and_final_clean_latents(
    cuda_model, monkeypatch
):
    prompt = [0, cuda_model.config.vocab_size - 1, 128]
    with torch.no_grad():
        known, _ = cuda_model.vae.encode(torch.tensor([prompt], device="cuda"))
    known = known.float()
    observed_states = []
    observed_contexts = []
    block_velocity = cuda_model.prior.block_velocity
    decode = cuda_model.vae.decode

    def observe_velocity(state, times, cache):
        if cache is None:
            observed_states.append((state[:, : len(prompt)].clone(), times.clone()))
        return block_velocity(state, times, cache)

    def observe_decode(context):
        observed_contexts.append(context[:, : len(prompt)].clone())
        return decode(context)

    monkeypatch.setattr(cuda_model.prior, "block_velocity", observe_velocity)
    monkeypatch.setattr(cuda_model.vae, "decode", observe_decode)
    result = generate_tokens(cuda_model, prompt, new_tokens=5, steps=2, seed=91)
    fixed_noise = observed_states[0][0]
    for state, times in observed_states:
        normalized_time = times[:, : len(prompt), None]
        torch.testing.assert_close(
            state, (1 - normalized_time) * known + normalized_time * fixed_noise
        )
    for context in observed_contexts:
        torch.testing.assert_close(context, known)
    assert result["output_ids"][: len(prompt)] == prompt
    assert len(result["generated_ids"]) == 5
    assert result["exact_conditional_sampling"] is False

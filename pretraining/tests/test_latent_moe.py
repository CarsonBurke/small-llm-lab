from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor
import torch.nn.functional as F

from pretraining.latent_moe import (
    LatentMoEConfig,
    QuantileBalanceHistogram,
    StableLatentMoE,
    situ_glu,
)


def _config(**overrides: int | float) -> LatentMoEConfig:
    values: dict[str, int | float] = {
        "model_dim": 8,
        "latent_dim": 4,
        "routed_hidden_dim": 6,
        "num_routed_experts": 4,
        "experts_per_token": 2,
        "shared_hidden_dim": 7,
        "num_shared_experts": 2,
        "rms_norm_eps": 1e-6,
    }
    values.update(overrides)
    return LatentMoEConfig(**values)  # type: ignore[arg-type]


def _independent_reference(module: StableLatentMoE, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    config = module.config
    flat_x = x.reshape(-1, config.model_dim)
    scores = torch.sigmoid(F.linear(flat_x.float(), module.router_weight.float()))
    indices = (scores + module.correction_bias).topk(config.experts_per_token, dim=-1).indices
    raw_selected = scores.gather(1, indices)
    weights = raw_selected / raw_selected.sum(1, keepdim=True)
    latent = F.linear(flat_x, module.latent_down_proj.weight)

    routed_rows = []
    for token_index in range(flat_x.shape[0]):
        aggregate = torch.zeros(config.latent_dim, dtype=x.dtype)
        for route_index in range(config.experts_per_token):
            expert = indices[token_index, route_index]
            gate_up = F.linear(latent[token_index], module.expert_gate_up_weight[expert])
            gate, up = gate_up.chunk(2)
            expert_output = F.linear(
                situ_glu(gate, up, config.situ_gate_cap, config.situ_up_cap),
                module.expert_down_weight[expert],
            )
            aggregate = aggregate + weights[token_index, route_index].to(x.dtype) * expert_output
        routed_rows.append(aggregate)
    routed = torch.stack(routed_rows)
    routed = F.rms_norm(
        routed,
        (config.latent_dim,),
        module.routed_norm.weight,
        config.rms_norm_eps,
    )
    routed = F.linear(routed, module.latent_up_proj.weight)

    shared_gate_up = F.linear(flat_x, module.shared_expert.gate_up_proj.weight)
    shared_gate, shared_up = shared_gate_up.chunk(2, dim=-1)
    shared = F.linear(
        situ_glu(shared_gate, shared_up, config.situ_gate_cap, config.situ_up_cap),
        module.shared_expert.down_proj.weight,
    )
    output = (shared + routed).reshape_as(x)
    return output, indices, weights


def test_config_rejects_invalid_routing() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        _config(experts_per_token=5)
    with pytest.raises(ValueError, match="fewer than 1024"):
        _config(num_routed_experts=1024)
    with pytest.raises(ValueError, match="soft caps"):
        _config(situ_gate_cap=0.0)


def test_reference_forward_matches_independent_equations_exactly() -> None:
    torch.manual_seed(17)
    module = StableLatentMoE(_config(), dtype=torch.float64, implementation="reference")
    with torch.no_grad():
        module.correction_bias.copy_(torch.tensor([-0.4, 0.2, 0.5, -0.3]))
    x = torch.randn(2, 3, 8, dtype=torch.float64)

    actual, telemetry = module(x, return_telemetry=True)
    expected, indices, weights = _independent_reference(module, x)

    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(telemetry.expert_indices, indices.reshape(2, 3, 2))
    torch.testing.assert_close(telemetry.expert_weights, weights.reshape(2, 3, 2))
    assert telemetry.expert_loads.sum().item() == 12
    torch.testing.assert_close(telemetry.load_fraction.sum(), torch.tensor(1.0))
    assert telemetry.load_cv_squared.ndim == 0
    assert not telemetry.expert_weights.requires_grad


def test_correction_bias_changes_selection_but_not_mixture_weights() -> None:
    config = _config(experts_per_token=1)
    module = StableLatentMoE(config, dtype=torch.float32, implementation="reference")
    with torch.no_grad():
        module.router_weight.zero_()
        module.router_weight[0, 0] = 2.0
        module.router_weight[1, 0] = 1.0
        module.router_weight[2, 0] = -1.0
        module.router_weight[3, 0] = -2.0
    x = torch.zeros(1, 8)
    x[0, 0] = 1.0

    _, before = module(x, return_telemetry=True)
    module.set_correction_bias_(torch.tensor([-2.0, -2.0, 6.0, -2.0]))
    _, after = module(x, return_telemetry=True)

    assert before.expert_indices.item() == 0
    assert after.expert_indices.item() == 2
    # With k=1 normalization of the raw score is exactly one.  Were the bias
    # incorrectly included in weighting, this would not generally hold.
    torch.testing.assert_close(after.expert_weights, torch.ones_like(after.expert_weights))


def test_reference_backward_matches_independent_equations() -> None:
    torch.manual_seed(23)
    actual_module = StableLatentMoE(_config(), dtype=torch.float64, implementation="reference")
    expected_module = copy.deepcopy(actual_module)
    actual_x = torch.randn(2, 2, 8, dtype=torch.float64, requires_grad=True)
    expected_x = actual_x.detach().clone().requires_grad_(True)
    probe = torch.randn(2, 2, 8, dtype=torch.float64)

    actual_loss = (actual_module(actual_x) * probe).sum()
    expected_output, _, _ = _independent_reference(expected_module, expected_x)
    expected_loss = (expected_output * probe).sum()
    actual_loss.backward()
    expected_loss.backward()

    torch.testing.assert_close(actual_x.grad, expected_x.grad, rtol=2e-12, atol=2e-12)
    expected_parameters = dict(expected_module.named_parameters())
    for name, parameter in actual_module.named_parameters():
        expected_gradient = expected_parameters[name].grad
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert expected_gradient is not None, f"reference missing gradient for {name}"
        torch.testing.assert_close(parameter.grad, expected_gradient, rtol=2e-12, atol=2e-12)
        assert torch.isfinite(parameter.grad).all(), name


def test_quantile_bias_update_is_explicit_finite_and_centered() -> None:
    torch.manual_seed(29)
    module = StableLatentMoE(_config(), dtype=torch.float32)
    x = torch.randn(32, 8)
    old_bias = module.correction_bias.clone()

    next_bias = module.compute_next_correction_bias(x)

    torch.testing.assert_close(module.correction_bias, old_bias)
    assert torch.isfinite(next_bias).all()
    torch.testing.assert_close(next_bias.mean(), torch.tensor(0.0), atol=2e-7, rtol=0)
    module.set_correction_bias_(next_bias + 7.0)
    torch.testing.assert_close(module.correction_bias.mean(), torch.tensor(0.0), atol=2e-7, rtol=0)


def test_qb_histogram_matches_exact_quantile_within_bin_error() -> None:
    torch.manual_seed(37)
    module = StableLatentMoE(_config(), dtype=torch.float32)
    x = torch.randn(8192, 8)
    exact = module.compute_next_correction_bias(x)

    histogram = module.compute_qb_histogram(x, num_bins=1000)
    approximate = module.correction_bias_from_qb_histogram(histogram)

    assert histogram.counts.shape == (4, 1000)
    torch.testing.assert_close(histogram.counts.sum(1), torch.full((4,), 8192))
    assert histogram.token_count.item() == 8192
    bin_width = (histogram.upper_bound - histogram.lower_bound) / 1000
    # Mean-centering can add at most one further bin of error.
    assert (approximate - exact).abs().max() <= 2.0 * bin_width + 1e-6


def test_qb_histograms_are_additive_across_microbatches() -> None:
    torch.manual_seed(41)
    module = StableLatentMoE(_config(), dtype=torch.float32)
    x = torch.randn(64, 8)
    whole = module.compute_qb_histogram(x, num_bins=64)
    first = module.compute_qb_histogram(x[:21], num_bins=64)
    second = module.compute_qb_histogram(x[21:], num_bins=64)

    torch.testing.assert_close(first.counts + second.counts, whole.counts)
    torch.testing.assert_close(first.token_count + second.token_count, whole.token_count)
    torch.testing.assert_close(first.lower_bound, whole.lower_bound)
    torch.testing.assert_close(first.upper_bound, whole.upper_bound)


def test_forward_qb_accumulator_equals_summed_microbatch_histograms() -> None:
    torch.manual_seed(47)
    module = StableLatentMoE(_config(), dtype=torch.float32, implementation="selected_bmm")
    module.enable_qb_collection(num_bins=64)
    batches = [torch.randn(5, 8), torch.randn(3, 2, 8), torch.randn(7, 8)]
    expected_histograms = [module.compute_qb_histogram(batch, num_bins=64) for batch in batches]
    expected_loads = torch.zeros(4, dtype=torch.int64)
    fixed_bias = module.correction_bias.clone()

    for batch in batches:
        _, telemetry = module(batch, return_telemetry=True)
        expected_loads.add_(telemetry.expert_loads)

    accumulated = module.get_accumulated_qb_histogram()
    expected_counts = sum(
        (histogram.counts for histogram in expected_histograms),
        torch.zeros_like(expected_histograms[0].counts),
    )
    expected_tokens = sum(
        (histogram.token_count for histogram in expected_histograms),
        torch.zeros_like(expected_histograms[0].token_count),
    )
    torch.testing.assert_close(accumulated.counts, expected_counts)
    torch.testing.assert_close(accumulated.token_count, expected_tokens)
    torch.testing.assert_close(module.get_accumulated_route_loads(), expected_loads)
    torch.testing.assert_close(module.correction_bias, fixed_bias)


def test_forward_qb_accumulator_reset_eval_and_nonpersistent_state() -> None:
    torch.manual_seed(53)
    module = StableLatentMoE(_config(), dtype=torch.float32, implementation="selected_bmm")
    module.enable_qb_collection(num_bins=32)
    qb_buffer_names = {name for name, _ in module.named_buffers() if name.startswith("_qb_")}
    assert qb_buffer_names
    assert not qb_buffer_names.intersection(module.state_dict())

    module(torch.randn(9, 8))
    assert module.get_accumulated_qb_histogram().token_count.item() == 9
    module.eval()
    module(torch.randn(4, 8))
    assert module.get_accumulated_qb_histogram().token_count.item() == 9

    module.set_correction_bias_(torch.tensor([-0.6, -0.2, 0.3, 0.5]))
    module.reset_qb_accumulators_()
    reset = module.get_accumulated_qb_histogram()
    assert reset.counts.count_nonzero().item() == 0
    assert reset.token_count.item() == 0
    torch.testing.assert_close(reset.lower_bound, module.correction_bias.min() - 1.0)
    torch.testing.assert_close(reset.upper_bound, module.correction_bias.max() + 1.0)
    torch.testing.assert_close(
        module.get_accumulated_route_loads(), torch.zeros(4, dtype=torch.int64)
    )


def test_qb_collection_can_be_explicitly_enabled_for_eval_mode_trainers() -> None:
    module = StableLatentMoE(_config(), implementation="reference").eval()
    module.enable_qb_collection(num_bins=32, collect_in_eval=True)
    module(torch.randn(2, 5, 8))
    assert module.get_accumulated_qb_histogram().token_count.item() == 10


def test_compiled_forward_accumulates_fixed_shape_qb_buffers() -> None:
    torch.manual_seed(59)
    module = StableLatentMoE(_config(), dtype=torch.float32, implementation="selected_bmm")
    module.enable_qb_collection(num_bins=32)
    compiled = torch.compile(module, backend="eager", fullgraph=True)

    compiled(torch.randn(6, 8))
    compiled(torch.randn(6, 8))

    accumulated = module.get_accumulated_qb_histogram()
    assert accumulated.token_count.item() == 12
    torch.testing.assert_close(accumulated.counts.sum(1), torch.full((4,), 12))


def test_grouped_mm_request_fails_closed_on_cpu() -> None:
    module = StableLatentMoE(_config(), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="CUDA BF16"):
        module(torch.randn(2, 8), implementation="grouped_mm")


def test_grouped_algorithm_matches_reference_with_cpu_grouped_mm_stub(monkeypatch) -> None:
    torch.manual_seed(31)
    grouped_module = StableLatentMoE(_config(), dtype=torch.float32)
    reference_module = copy.deepcopy(grouped_module)
    x = torch.randn(3, 2, 8)

    def grouped_mm_stub(left: Tensor, right: Tensor, *, offs: Tensor) -> Tensor:
        starts = torch.cat((offs.new_zeros(1), offs))
        chunks = [
            left[starts[i] : starts[i + 1]] @ right[i]
            for i in range(right.shape[0])
        ]
        return torch.cat(chunks, dim=0)

    monkeypatch.setattr(torch, "_grouped_mm", grouped_mm_stub)
    monkeypatch.setattr(grouped_module, "_can_use_grouped_mm", lambda _: True)
    grouped = grouped_module(x, implementation="grouped_mm")
    reference = reference_module(x, implementation="reference")
    torch.testing.assert_close(grouped, reference, rtol=2e-6, atol=2e-6)


def test_grouped_algorithm_backward_matches_reference_with_cpu_stub(monkeypatch) -> None:
    torch.manual_seed(39)
    grouped_module = StableLatentMoE(_config(), dtype=torch.float64)
    reference_module = copy.deepcopy(grouped_module)
    grouped_x = torch.randn(3, 2, 8, dtype=torch.float64, requires_grad=True)
    reference_x = grouped_x.detach().clone().requires_grad_(True)
    probe = torch.randn_like(grouped_x)

    def grouped_mm_stub(left: Tensor, right: Tensor, *, offs: Tensor) -> Tensor:
        starts = torch.cat((offs.new_zeros(1), offs))
        return torch.cat(
            [
                left[starts[i] : starts[i + 1]] @ right[i]
                for i in range(right.shape[0])
            ],
            dim=0,
        )

    monkeypatch.setattr(torch, "_grouped_mm", grouped_mm_stub)
    monkeypatch.setattr(grouped_module, "_can_use_grouped_mm", lambda _: True)
    (grouped_module(grouped_x, implementation="grouped_mm") * probe).sum().backward()
    (reference_module(reference_x, implementation="reference") * probe).sum().backward()

    torch.testing.assert_close(grouped_x.grad, reference_x.grad, rtol=2e-12, atol=2e-12)
    reference_parameters = dict(reference_module.named_parameters())
    for name, parameter in grouped_module.named_parameters():
        reference_gradient = reference_parameters[name].grad
        assert parameter.grad is not None, name
        assert reference_gradient is not None, name
        torch.testing.assert_close(
            parameter.grad, reference_gradient, rtol=2e-12, atol=2e-12
        )


def test_selected_bmm_matches_reference() -> None:
    torch.manual_seed(43)
    module = StableLatentMoE(_config(), dtype=torch.float32)
    x = torch.randn(3, 2, 8)
    selected_bmm = module(x, implementation="selected_bmm")
    reference = module(x, implementation="reference")
    torch.testing.assert_close(selected_bmm, reference, rtol=2e-6, atol=2e-6)


def test_selected_bmm_backward_matches_reference() -> None:
    torch.manual_seed(45)
    selected_module = StableLatentMoE(_config(), dtype=torch.float64)
    reference_module = copy.deepcopy(selected_module)
    selected_x = torch.randn(3, 2, 8, dtype=torch.float64, requires_grad=True)
    reference_x = selected_x.detach().clone().requires_grad_(True)
    probe = torch.randn_like(selected_x)

    (selected_module(selected_x, implementation="selected_bmm") * probe).sum().backward()
    (reference_module(reference_x, implementation="reference") * probe).sum().backward()

    torch.testing.assert_close(selected_x.grad, reference_x.grad, rtol=2e-12, atol=2e-12)
    reference_parameters = dict(reference_module.named_parameters())
    for name, parameter in selected_module.named_parameters():
        reference_gradient = reference_parameters[name].grad
        assert parameter.grad is not None, name
        assert reference_gradient is not None, name
        torch.testing.assert_close(
            parameter.grad, reference_gradient, rtol=2e-12, atol=2e-12
        )


def test_default_fp32_master_parameters_accept_bfloat16_activations() -> None:
    module = StableLatentMoE(_config(), dtype=torch.float32, implementation="reference")
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    output = module(x)

    assert output.dtype == torch.bfloat16
    assert module.router_weight.dtype == torch.float32
    assert module.correction_bias.dtype == torch.float32
    assert module.expert_gate_up_weight.dtype == torch.float32
    assert module.latent_down_proj.weight.dtype == torch.float32
    output.float().square().mean().backward()
    assert module.expert_gate_up_weight.grad is not None
    assert module.expert_gate_up_weight.grad.dtype == torch.float32
    assert x.grad is not None


def test_whole_module_cast_preserves_fp32_routing_contract() -> None:
    module = StableLatentMoE(_config()).bfloat16()
    assert module.router_weight.dtype == torch.float32
    assert module.correction_bias.dtype == torch.float32
    assert module.expert_gate_up_weight.dtype == torch.bfloat16
    output = module(torch.randn(3, 8, dtype=torch.bfloat16))
    assert output.dtype == torch.bfloat16


def test_empty_and_invalid_inputs() -> None:
    module = StableLatentMoE(_config(), dtype=torch.float32)
    output, telemetry = module(torch.empty(0, 8), return_telemetry=True)
    assert output.shape == (0, 8)
    assert telemetry.expert_loads.tolist() == [0, 0, 0, 0]
    with pytest.raises(ValueError, match="expected"):
        module(torch.randn(2, 7))
    with pytest.raises(ValueError, match="empty batch"):
        module.compute_next_correction_bias(torch.empty(0, 8))
    with pytest.raises(ValueError, match="expected"):
        module.compute_next_correction_bias(torch.randn(16))
    with pytest.raises(ValueError, match="expected"):
        module.compute_qb_histogram(torch.randn(16))


def test_qb_histogram_recovery_rejects_invalid_values_without_host_reads() -> None:
    module = StableLatentMoE(_config(), dtype=torch.float32)
    valid = module.compute_qb_histogram(torch.randn(8, 8), num_bins=16)
    with pytest.raises(RuntimeError, match="positive"):
        module.correction_bias_from_qb_histogram(
            QuantileBalanceHistogram(
                counts=torch.zeros_like(valid.counts),
                token_count=torch.zeros_like(valid.token_count),
                lower_bound=valid.lower_bound,
                upper_bound=valid.upper_bound,
            )
        )
    bad_counts = valid.counts.clone()
    bad_counts[0, 0] = -1
    with pytest.raises(RuntimeError, match="nonnegative"):
        module.correction_bias_from_qb_histogram(
            QuantileBalanceHistogram(
                counts=bad_counts,
                token_count=valid.token_count,
                lower_bound=valid.lower_bound,
                upper_bound=valid.upper_bound,
            )
        )
    missing_count = valid.counts.clone()
    positive_bin = torch.nonzero(missing_count[0] > 0, as_tuple=False)[0, 0]
    missing_count[0, positive_bin] -= 1
    with pytest.raises(RuntimeError, match="every pooled token"):
        module.correction_bias_from_qb_histogram(
            QuantileBalanceHistogram(
                counts=missing_count,
                token_count=valid.token_count,
                lower_bound=valid.lower_bound,
                upper_bound=valid.upper_bound,
            )
        )
    with pytest.raises(RuntimeError, match="finite and increasing"):
        module.correction_bias_from_qb_histogram(
            QuantileBalanceHistogram(
                counts=valid.counts,
                token_count=valid.token_count,
                lower_bound=valid.upper_bound,
                upper_bound=valid.lower_bound,
            )
        )


def test_situ_glu_is_bounded_and_smooth() -> None:
    gate = torch.tensor([-1e4, -1.0, 0.0, 1.0, 1e4], requires_grad=True)
    up = torch.full_like(gate, 1e4)
    output = situ_glu(gate, up)
    assert output.abs().max().item() <= 100.0
    output.sum().backward()
    assert torch.isfinite(gate.grad).all()

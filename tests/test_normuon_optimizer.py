from __future__ import annotations

import torch

import normuon_train_gpt as normuon


def _trading_bot_newton_schulz(
    update: torch.Tensor,
    steps: int = 5,
) -> torch.Tensor:
    original_dtype = update.dtype
    transposed = update.size(0) > update.size(1)
    x = update.bfloat16()
    x = x / x.norm().clamp_min(1e-7)
    if transposed:
        x = x.T
    x = x.unsqueeze(0)
    for _ in range(steps):
        a = x @ x.mT
        b = torch.baddbmm(a, a, a, beta=-4.7750, alpha=2.0315)
        x = torch.baddbmm(x, b, x, beta=3.4445)
    x = x.squeeze(0)
    if transposed:
        x = x.T.contiguous()
    return x.to(dtype=original_dtype)


def _reference_rescale(
    update: torch.Tensor,
    second_momentum: torch.Tensor,
    beta2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_dtype = update.dtype
    update_fp32 = update.float()
    original_norm = update_fp32.norm(dim=(-2, -1), keepdim=True)
    row_mean_sq = update_fp32.square().mean(dim=-1, keepdim=True)
    second_momentum = second_momentum.lerp(row_mean_sq, 1 - beta2)
    update_fp32 *= second_momentum.sqrt().add(1e-10).reciprocal()
    update_fp32 *= original_norm / update_fp32.norm(dim=(-2, -1), keepdim=True).add(1e-10)
    return update_fp32.to(dtype=original_dtype), second_momentum


def test_rescale_matches_reference_and_preserves_norm() -> None:
    generator = torch.Generator().manual_seed(7)
    for shape in ((8, 8), (12, 8), (8, 12)):
        update = torch.randn(shape, generator=generator)
        second = torch.rand(shape[0], 1, generator=generator)
        expected_update, expected_second = _reference_rescale(
            update.clone(), second.clone(), 0.95
        )
        actual_update, actual_second = normuon.normuon_rescale(
            update.clone(), second.clone(), 0.95
        )

        torch.testing.assert_close(actual_update, expected_update)
        torch.testing.assert_close(actual_second, expected_second)
        torch.testing.assert_close(actual_update.norm(), update.norm())


def test_newton_schulz_matches_trading_bot_reference() -> None:
    generator = torch.Generator().manual_seed(11)
    for shape in ((8, 8), (12, 8), (8, 12)):
        update = torch.randn(shape, generator=generator)
        actual = normuon.zeropower_via_newtonschulz5(update, steps=5)
        expected = _trading_bot_newton_schulz(update, steps=5)
        assert actual.dtype is update.dtype
        assert actual.shape == update.shape
        torch.testing.assert_close(actual, expected)


def test_newton_schulz_does_not_mutate_bfloat16_input() -> None:
    update = torch.randn(12, 8).bfloat16()
    before = update.clone()

    actual = normuon.zeropower_via_newtonschulz5(update, steps=5)

    torch.testing.assert_close(update, before)
    assert actual.dtype is torch.bfloat16


def test_optimizer_preserves_gradients_and_updates_all_state() -> None:
    generator = torch.Generator().manual_seed(17)
    params = [
        torch.nn.Parameter(torch.randn(8, 8, generator=generator)),
        torch.nn.Parameter(torch.randn(12, 8, generator=generator)),
        torch.nn.Parameter(torch.randn(8, 12, generator=generator)),
    ]
    optimizer = normuon.NorMuon(
        params,
        lr=0.04,
        momentum=0.85,
        beta2=0.95,
        backend_steps=5,
    )
    gradients = [torch.randn_like(param) for param in params]
    before = [param.detach().clone() for param in params]
    for param, grad in zip(params, gradients, strict=True):
        param.grad = grad.clone()

    optimizer.step()

    for param, old_param, grad in zip(params, before, gradients, strict=True):
        assert not torch.equal(param, old_param)
        torch.testing.assert_close(param.grad, grad)
        assert optimizer.state[param]["momentum_buffer"].shape == param.shape
        assert optimizer.state[param]["second_momentum_buffer"].shape == (param.size(0), 1)
        assert optimizer.state[param]["second_momentum_buffer"].dtype is torch.float32


def test_momentum_uses_reference_ema_nesterov_with_changing_beta() -> None:
    generator = torch.Generator().manual_seed(23)
    param = torch.nn.Parameter(torch.randn(8, 8, generator=generator))
    optimizer = normuon.NorMuon(
        [param],
        lr=0.04,
        momentum=0.85,
        beta2=0.95,
        backend_steps=5,
    )
    expected_momentum = torch.zeros_like(param)

    for beta in (0.85, 0.90):
        grad = torch.randn(8, 8, generator=generator)
        param.grad = grad.clone()
        optimizer.param_groups[0]["momentum"] = beta
        expected_momentum.lerp_(grad, 1 - beta)

        optimizer.step()

        torch.testing.assert_close(
            optimizer.state[param]["momentum_buffer"],
            expected_momentum,
        )
        torch.testing.assert_close(param.grad, grad)


def test_missing_gradient_is_untouched() -> None:
    active = torch.nn.Parameter(torch.randn(8, 8))
    inactive = torch.nn.Parameter(torch.randn(8, 8))
    inactive_before = inactive.detach().clone()
    optimizer = normuon.NorMuon(
        [active, inactive],
        lr=0.04,
        momentum=0.85,
        beta2=0.95,
        backend_steps=5,
    )
    active.grad = torch.randn_like(active)

    optimizer.step()

    torch.testing.assert_close(inactive, inactive_before)
    assert inactive.grad is None
    assert inactive not in optimizer.state


def test_empty_state_restore_clears_warmup_moments() -> None:
    param = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = normuon.NorMuon(
        [param],
        lr=0.04,
        momentum=0.85,
        beta2=0.95,
        backend_steps=5,
    )
    initial_state = optimizer.state_dict()

    param.grad = torch.randn_like(param)
    optimizer.step()
    assert optimizer.state

    optimizer.load_state_dict(initial_state)
    assert not optimizer.state

    param.grad = torch.randn_like(param)
    optimizer.step()
    assert optimizer.state[param]["momentum_buffer"].count_nonzero() > 0
    assert optimizer.state[param]["second_momentum_buffer"].count_nonzero() > 0

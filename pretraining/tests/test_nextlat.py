from __future__ import annotations

import math

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from pretraining.nextlat import (
    NextLatDynamicsModel,
    document_transition_mask,
    masked_smooth_l1,
    nextlat_terminal_loss,
    rational_softcap,
)


def _copy_head(head: nn.Linear) -> nn.Linear:
    copied = nn.Linear(head.in_features, head.out_features, bias=head.bias is not None)
    copied.load_state_dict(head.state_dict())
    return copied


def _dense_terminal_loss(
    hidden: torch.Tensor,
    predicted_hidden: torch.Tensor,
    targets: torch.Tensor,
    head: nn.Linear,
    transition_mask: torch.Tensor,
    *,
    hidden_weight: float,
    kl_weight: float,
    softcap: float,
):
    """Straightforward fp32/BF16 oracle matching the production contract."""

    weight = head.weight.to(dtype=hidden.dtype)
    bias = None if head.bias is None else head.bias.to(dtype=hidden.dtype)
    actual_logits = rational_softcap(
        F.linear(hidden, weight, bias).float(), softcap
    )
    ce_sum = F.cross_entropy(
        actual_logits.reshape(-1, actual_logits.shape[-1]),
        targets.reshape(-1),
        reduction="sum",
    )
    if hidden.shape[1] > 2:
        student_logits = rational_softcap(
            F.linear(
                predicted_hidden[:, :-1],
                weight.detach(),
                None if bias is None else bias.detach(),
            ).float(),
            softcap,
        )
        teacher_log_probs = actual_logits[:, 1:-1].detach().log_softmax(dim=-1)
        student_log_probs = student_logits.log_softmax(dim=-1)
        per_token = (
            teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1)
        mask = transition_mask[:, :-1].float()
        kl_loss = (per_token * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        kl_loss = actual_logits.new_zeros(())
    hidden_loss = masked_smooth_l1(
        predicted_hidden,
        hidden[:, 1:],
        transition_mask,
    )
    total = hidden_weight * hidden_loss + kl_weight * kl_loss
    return ce_sum, hidden_loss, kl_loss, total


def _tiled_online_log_normalizer(
    raw_logits: torch.Tensor,
    softcap: float,
    block_size: int,
) -> torch.Tensor:
    capped = rational_softcap(raw_logits.float(), softcap)
    row_max = torch.full(
        capped.shape[:-1], float("-inf"), dtype=torch.float32
    )
    row_sum = torch.zeros_like(row_max)
    for start in range(0, capped.shape[-1], block_size):
        block = capped[..., start : start + block_size]
        block_max = block.max(dim=-1).values
        new_max = torch.maximum(row_max, block_max)
        row_sum = row_sum * (row_max - new_max).exp() + (
            block - new_max[..., None]
        ).exp().sum(dim=-1)
        row_max = new_max
    return row_max + row_sum.log()


def _softcap_derivative(raw_logits: torch.Tensor, softcap: float) -> torch.Tensor:
    denominator = raw_logits.float().square() + softcap**2
    return softcap**3 * denominator.rsqrt() / denominator


def test_dynamics_uses_official_dimensions_init_and_residual() -> None:
    torch.manual_seed(11)
    dynamics = NextLatDynamicsModel(model_dim=512, proj_factor=1.6)

    assert dynamics.input_dim == 1024
    assert dynamics.hidden_dim == 1664
    assert isinstance(dynamics.norm_x, nn.RMSNorm)
    linears = [module for module in dynamics.mlp if isinstance(module, nn.Linear)]
    gelus = [module for module in dynamics.mlp if isinstance(module, nn.GELU)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (1024, 1664),
        (1664, 1664),
        (1664, 512),
    ]
    assert len(gelus) == 2
    assert all(layer.bias is None for layer in linears)
    for layer in linears:
        assert abs(layer.weight.mean().item()) < 1e-3
        assert layer.weight.std().item() == pytest.approx(0.02, abs=7e-5)

    current = torch.randn(2, 3, 512)
    next_token = torch.randn_like(current)
    for layer in linears:
        layer.weight.data.zero_()
    output = dynamics(current, next_token)
    torch.testing.assert_close(output, current, rtol=0.0, atol=0.0)


def test_dynamics_supports_generic_width_and_arbitrary_leading_dims() -> None:
    dynamics = NextLatDynamicsModel(model_dim=48, proj_factor=2.0)
    current = torch.randn(2, 3, 4, 48)
    next_token = torch.randn_like(current)
    output = dynamics(current, next_token)
    assert output.shape == current.shape
    assert dynamics.hidden_dim == 256


def test_dynamics_keeps_fp32_masters_with_bfloat16_inputs() -> None:
    dynamics = NextLatDynamicsModel(model_dim=64)
    current = torch.randn(2, 3, 64, dtype=torch.bfloat16)
    next_token = torch.randn_like(current)
    output = dynamics(current, next_token)
    assert output.dtype == torch.bfloat16
    assert output.shape == current.shape
    linears = [module for module in dynamics.mlp if isinstance(module, nn.Linear)]
    assert all(layer.weight.dtype == torch.float32 for layer in linears)
    output.float().square().mean().backward()
    assert all(layer.weight.grad is not None for layer in linears)
    assert all(layer.weight.grad.dtype == torch.float32 for layer in linears)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model_dim": 0}, "model_dim"),
        ({"model_dim": 8, "proj_factor": 0.0}, "proj_factor"),
        ({"model_dim": 8, "proj_factor": 0.1}, "rounds.*zero"),
        ({"model_dim": 8, "eps": math.inf}, "eps"),
    ],
)
def test_dynamics_rejects_invalid_configuration(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        NextLatDynamicsModel(**kwargs)


def test_dynamics_rejects_invalid_inputs() -> None:
    dynamics = NextLatDynamicsModel(128)
    current = torch.randn(2, 3, 128)
    with pytest.raises(ValueError, match="identical shapes"):
        dynamics(current, torch.randn(2, 2, 128))
    with pytest.raises(ValueError, match="model_dim"):
        dynamics(torch.randn(2, 3, 64), torch.randn(2, 3, 64))
    with pytest.raises(ValueError, match="dtype"):
        dynamics(current, torch.randn_like(current, dtype=torch.float64))


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize(
    "token_chunk_size",
    [
        1,   # degenerate one-token first/middle/last segments
        3,   # sequence is segmented across token-chunk boundaries
        5,   # token budget exactly equals sequence length
        9,   # one complete sequence fits per chunk
        10,  # two complete sequences fit per chunk
    ],
)
def test_terminal_matches_dense_fp32_scalars_and_all_first_order_gradients(
    with_bias: bool,
    token_chunk_size: int,
) -> None:
    torch.manual_seed(29)
    hidden_dense = torch.randn(3, 5, 7, requires_grad=True)
    predicted_dense = torch.randn(3, 4, 7, requires_grad=True)
    hidden_terminal = hidden_dense.detach().clone().requires_grad_()
    predicted_terminal = predicted_dense.detach().clone().requires_grad_()
    head_dense = nn.Linear(7, 19, bias=with_bias)
    head_terminal = _copy_head(head_dense)
    targets = torch.randint(0, 19, (3, 5))
    mask = torch.tensor(
        [[True, False, True, True], [True, True, False, True], [False, True, True, False]]
    )
    hidden_weight = 1.7
    kl_weight = 0.35
    softcap = 3.5

    dense = _dense_terminal_loss(
        hidden_dense,
        predicted_dense,
        targets,
        head_dense,
        mask,
        hidden_weight=hidden_weight,
        kl_weight=kl_weight,
        softcap=softcap,
    )
    terminal = nextlat_terminal_loss(
        hidden_terminal,
        predicted_terminal,
        targets,
        head_terminal,
        transition_mask=mask,
        hidden_weight=hidden_weight,
        kl_weight=kl_weight,
        softcap=softcap,
        token_chunk_size=token_chunk_size,
    )
    terminal_values = (
        terminal.ce_sum,
        terminal.hidden_loss,
        terminal.kl_loss,
        terminal.total,
    )
    for actual, expected in zip(terminal_values, dense, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=5e-7)

    dense_objective = 0.41 * dense[0] + 2.3 * dense[3]
    terminal_objective = 0.41 * terminal.ce_sum + 2.3 * terminal.total
    dense_objective.backward()
    terminal_objective.backward()
    torch.testing.assert_close(
        hidden_terminal.grad, hidden_dense.grad, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        predicted_terminal.grad, predicted_dense.grad, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        head_terminal.weight.grad, head_dense.weight.grad, rtol=2e-5, atol=2e-6
    )
    if with_bias:
        torch.testing.assert_close(
            head_terminal.bias.grad, head_dense.bias.grad, rtol=2e-5, atol=2e-6
        )


def test_tiled_triton_row_math_matches_dense_values_and_raw_logit_gradients() -> None:
    torch.manual_seed(30)
    batch_size, sequence_length, kl_length, vocab_size = 2, 4, 2, 17
    softcap = 3.5
    teacher_raw = torch.randn(
        batch_size, sequence_length, vocab_size, requires_grad=True
    )
    student_raw = torch.randn(
        batch_size, kl_length, vocab_size, requires_grad=True
    )
    targets = torch.randint(0, vocab_size, (batch_size, sequence_length))
    targets[0, 0] = -100
    kl_mask = torch.tensor([[True, False], [True, True]])
    normalizer = kl_mask.sum().clamp_min(1).float()

    teacher_capped = rational_softcap(teacher_raw, softcap)
    student_capped = rational_softcap(student_raw, softcap)
    dense_ce = F.cross_entropy(
        teacher_capped.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="sum",
    )
    teacher_log_probs = teacher_capped[:, 1:3].detach().log_softmax(dim=-1)
    student_log_probs = student_capped.log_softmax(dim=-1)
    dense_per_token_kl = (
        teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    dense_kl = (dense_per_token_kl * kl_mask.float()).sum() / normalizer
    dense_teacher_gradient = torch.autograd.grad(
        dense_ce, teacher_raw, retain_graph=True
    )[0]
    dense_student_gradient = torch.autograd.grad(dense_kl, student_raw)[0]

    teacher_log_z = _tiled_online_log_normalizer(teacher_raw.detach(), softcap, 5)
    student_log_z = _tiled_online_log_normalizer(student_raw.detach(), softcap, 5)
    teacher_kernel_capped = rational_softcap(teacher_raw.detach(), softcap)
    student_kernel_capped = rational_softcap(student_raw.detach(), softcap)
    ce_included = targets != -100
    safe_targets = targets.masked_fill(~ce_included, 0)
    tiled_ce = ((
        teacher_log_z
        - teacher_kernel_capped.gather(-1, safe_targets[..., None]).squeeze(-1)
    ) * ce_included).sum()
    teacher_kl_capped = teacher_kernel_capped[:, 1:3]
    teacher_probability = (
        teacher_kl_capped - teacher_log_z[:, 1:3, None]
    ).exp()
    student_probability = (student_kernel_capped - student_log_z[..., None]).exp()
    tiled_per_token_kl = (
        teacher_probability
        * (
            teacher_kl_capped
            - teacher_log_z[:, 1:3, None]
            - student_kernel_capped
            + student_log_z[..., None]
        )
    ).sum(dim=-1)
    tiled_kl = (tiled_per_token_kl * kl_mask.float()).sum() / normalizer

    ce_capped_gradient = (
        teacher_kernel_capped.softmax(dim=-1) * ce_included[..., None]
    )
    ce_capped_gradient.scatter_add_(
        -1,
        safe_targets[..., None],
        -ce_included[..., None].float(),
    )
    tiled_teacher_gradient = ce_capped_gradient * _softcap_derivative(
        teacher_raw.detach(), softcap
    )
    tiled_student_gradient = (
        (student_probability - teacher_probability)
        * _softcap_derivative(student_raw.detach(), softcap)
        * kl_mask[..., None]
        / normalizer
    )

    torch.testing.assert_close(tiled_ce, dense_ce, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(tiled_kl, dense_kl, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        tiled_teacher_gradient, dense_teacher_gradient, rtol=3e-6, atol=3e-7
    )
    torch.testing.assert_close(
        tiled_student_gradient, dense_student_gradient, rtol=3e-6, atol=3e-7
    )


def test_fused_linear_kernel_equations_match_compiled_terminal_reference() -> None:
    import pretraining.nextlat as nextlat

    torch.manual_seed(32)
    batch_size, sequence_length, hidden_dim, vocab_size = 2, 5, 7, 19
    teacher_start, teacher_stop = 1, 4
    softcap = 4.0
    hidden = torch.randn(batch_size, sequence_length, hidden_dim)
    predicted = torch.randn(
        batch_size, teacher_stop - teacher_start, hidden_dim
    )
    targets = torch.randint(0, vocab_size, (batch_size, sequence_length))
    mask = torch.tensor([[True, False, True], [True, True, False]])
    weight = torch.randn(vocab_size, hidden_dim)
    bias = torch.randn(vocab_size)
    normalizer = mask.sum().clamp_min(1).float()

    reference_gradients, (_, reference_components) = (
        nextlat._compiled_terminal_chunk_grad_and_value(
            hidden,
            predicted,
            targets,
            mask,
            weight,
            bias,
            softcap,
            normalizer,
            True,
            teacher_start,
            teacher_stop,
        )
    )

    hidden_flat = hidden.reshape(-1, hidden_dim)
    predicted_flat = predicted.reshape(-1, hidden_dim)
    teacher_raw = F.linear(hidden_flat, weight, bias).reshape(
        batch_size, sequence_length, vocab_size
    )
    student_raw = F.linear(predicted_flat, weight, bias).reshape(
        batch_size, teacher_stop - teacher_start, vocab_size
    )
    teacher_log_z = _tiled_online_log_normalizer(teacher_raw, softcap, 5)
    student_log_z = _tiled_online_log_normalizer(student_raw, softcap, 5)
    teacher_capped = rational_softcap(teacher_raw, softcap)
    student_capped = rational_softcap(student_raw, softcap)
    ce_sum = (
        teacher_log_z
        - teacher_capped.gather(-1, targets[..., None]).squeeze(-1)
    ).sum()
    teacher_kl_capped = teacher_capped[:, teacher_start:teacher_stop]
    teacher_probability = (
        teacher_kl_capped
        - teacher_log_z[:, teacher_start:teacher_stop, None]
    ).exp()
    student_probability = (student_capped - student_log_z[..., None]).exp()
    per_token_kl = (
        teacher_probability
        * (
            teacher_kl_capped
            - teacher_log_z[:, teacher_start:teacher_stop, None]
            - student_capped
            + student_log_z[..., None]
        )
    ).sum(dim=-1)
    kl_loss = (per_token_kl * mask.float()).sum() / normalizer

    teacher_raw_gradient = teacher_capped.softmax(dim=-1)
    teacher_raw_gradient.scatter_add_(
        -1,
        targets[..., None],
        -torch.ones_like(targets[..., None], dtype=torch.float32),
    )
    teacher_raw_gradient *= _softcap_derivative(teacher_raw, softcap)
    student_raw_gradient = (
        (student_probability - teacher_probability)
        * _softcap_derivative(student_raw, softcap)
        * mask[..., None]
        / normalizer
    )
    flat_teacher_gradient = teacher_raw_gradient.reshape(-1, vocab_size)
    flat_student_gradient = student_raw_gradient.reshape(-1, vocab_size)
    fused_gradients = (
        torch.mm(flat_teacher_gradient, weight).reshape_as(hidden),
        torch.mm(flat_student_gradient, weight).reshape_as(predicted),
        torch.mm(flat_teacher_gradient.t(), hidden_flat),
        flat_teacher_gradient.sum(dim=0),
    )

    reference_ce, reference_kl = reference_components
    torch.testing.assert_close(ce_sum, reference_ce, rtol=3e-6, atol=2e-6)
    torch.testing.assert_close(kl_loss, reference_kl, rtol=3e-6, atol=2e-6)
    for fused, reference in zip(fused_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(fused, reference, rtol=3e-5, atol=3e-6)


def test_triton_kernels_aot_compile_for_rtx5090() -> None:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, make_backend

    import pretraining.nextlat as nextlat

    target = GPUTarget("cuda", 120, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 8, "num_stages": 1})
    ce_source = ASTSource(
        fn=nextlat._nextlat_ce_value_vjp_kernel,
        signature={
            "teacher_raw": "*bf16",
            "targets": "*i64",
            "row_ce": "*fp32",
            "vocab_size": "constexpr",
            "softcap": "constexpr",
            "BLOCK_V": "constexpr",
        },
        constexprs={"vocab_size": 50304, "softcap": 15.0, "BLOCK_V": 1024},
    )
    kl_source = ASTSource(
        fn=nextlat._nextlat_kl_value_vjp_kernel,
        signature={
            "teacher_raw": "*bf16",
            "student_raw": "*bf16",
            "kl_mask": "*i1",
            "row_kl": "*fp32",
            "inverse_kl_normalizer": "*fp32",
            "teacher_rows_per_sequence": "constexpr",
            "kl_rows_per_sequence": "constexpr",
            "teacher_offset": "constexpr",
            "vocab_size": "constexpr",
            "softcap": "constexpr",
            "BLOCK_V": "constexpr",
        },
        constexprs={
            "teacher_rows_per_sequence": 2048,
            "kl_rows_per_sequence": 2046,
            "teacher_offset": 1,
            "vocab_size": 50304,
            "softcap": 15.0,
            "BLOCK_V": 1024,
        },
    )
    ce = triton.compile(
        ce_source, target=target, options=options.__dict__
    )
    kl = triton.compile(
        kl_source, target=target, options=options.__dict__
    )
    assert ce.metadata.global_scratch_size == 0
    assert kl.metadata.global_scratch_size == 0


def test_terminal_kl_detaches_teacher_and_vocabulary_head() -> None:
    torch.manual_seed(31)
    hidden = torch.randn(2, 4, 6, requires_grad=True)
    predicted = torch.randn(2, 3, 6, requires_grad=True)
    head = nn.Linear(6, 13, bias=True)
    targets = torch.randint(0, 13, (2, 4))
    mask = torch.ones(2, 3, dtype=torch.bool)
    result = nextlat_terminal_loss(
        hidden,
        predicted,
        targets,
        head,
        transition_mask=mask,
        hidden_weight=0.0,
        kl_weight=1.0,
    )
    result.kl_loss.backward()

    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
    assert predicted.grad is not None
    assert torch.count_nonzero(predicted.grad[:, :-1]) > 0
    torch.testing.assert_close(predicted.grad[:, -1], torch.zeros_like(predicted.grad[:, -1]))
    torch.testing.assert_close(head.weight.grad, torch.zeros_like(head.weight))
    torch.testing.assert_close(head.bias.grad, torch.zeros_like(head.bias))


def test_terminal_ce_preserves_default_ignore_index_semantics() -> None:
    torch.manual_seed(34)
    hidden = torch.randn(2, 4, 5, requires_grad=True)
    predicted = torch.randn(2, 3, 5, requires_grad=True)
    head = nn.Linear(5, 11)
    targets = torch.randint(0, 11, (2, 4))
    targets[0, 1] = -100
    mask = torch.ones(2, 3, dtype=torch.bool)
    terminal = nextlat_terminal_loss(
        hidden,
        predicted,
        targets,
        head,
        transition_mask=mask,
        token_chunk_size=3,
    )
    logits = rational_softcap(F.linear(hidden, head.weight, head.bias).float())
    expected_ce = F.cross_entropy(
        logits.reshape(-1, 11), targets.reshape(-1), reduction="sum"
    )
    torch.testing.assert_close(terminal.ce_sum, expected_ce)


def test_terminal_all_masked_has_zero_auxiliary_and_auxiliary_gradients() -> None:
    torch.manual_seed(37)
    hidden = torch.randn(2, 4, 5, requires_grad=True)
    predicted = torch.randn(2, 3, 5, requires_grad=True)
    head = nn.Linear(5, 11)
    targets = torch.randint(0, 11, (2, 4))
    mask = torch.zeros(2, 3, dtype=torch.bool)
    result = nextlat_terminal_loss(
        hidden,
        predicted,
        targets,
        head,
        transition_mask=mask,
    )
    torch.testing.assert_close(result.hidden_loss, torch.tensor(0.0))
    torch.testing.assert_close(result.kl_loss, torch.tensor(0.0))
    torch.testing.assert_close(result.total, torch.tensor(0.0))
    result.total.backward()
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
    torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted))


def test_terminal_two_token_sequence_has_hidden_loss_but_no_kl() -> None:
    hidden = torch.randn(2, 2, 5, requires_grad=True)
    predicted = torch.randn(2, 1, 5, requires_grad=True)
    head = nn.Linear(5, 11, bias=False)
    targets = torch.randint(0, 11, (2, 2))
    mask = torch.tensor([[True], [False]])
    result = nextlat_terminal_loss(
        hidden,
        predicted,
        targets,
        head,
        transition_mask=mask,
    )
    torch.testing.assert_close(result.kl_loss, torch.tensor(0.0))
    assert result.hidden_loss > 0
    (result.ce_sum + targets.numel() * result.total).backward()
    assert hidden.grad is not None
    assert predicted.grad is not None


def test_terminal_supports_noncontiguous_inputs() -> None:
    torch.manual_seed(41)
    hidden = torch.randn(3, 5, 4).transpose(1, 2).detach().requires_grad_()
    predicted = torch.randn(3, 5, 3).transpose(1, 2).detach().requires_grad_()
    targets = torch.randint(0, 17, (4, 3)).t()
    mask = torch.tensor(
        [[True, False, True], [False, True, True], [True, True, False]]
    ).t()
    assert not hidden.is_contiguous()
    assert not predicted.is_contiguous()
    assert not targets.is_contiguous()
    assert not mask.is_contiguous()
    head = nn.Linear(5, 17)
    result = nextlat_terminal_loss(
        hidden,
        predicted,
        targets,
        head,
        transition_mask=mask,
        token_chunk_size=3,
    )
    objective = result.ce_sum + targets.numel() * result.total
    objective.backward()
    assert torch.isfinite(objective)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()


def test_terminal_bfloat16_matches_dense_with_fp32_head_masters() -> None:
    torch.manual_seed(43)
    hidden_dense = torch.randn(2, 4, 8, dtype=torch.bfloat16).requires_grad_()
    predicted_dense = torch.randn(2, 3, 8, dtype=torch.bfloat16).requires_grad_()
    hidden_terminal = hidden_dense.detach().clone().requires_grad_()
    predicted_terminal = predicted_dense.detach().clone().requires_grad_()
    head_dense = nn.Linear(8, 23, bias=True)
    head_terminal = _copy_head(head_dense)
    targets = torch.randint(0, 23, (2, 4))
    mask = torch.tensor([[True, True, False], [True, False, True]])

    dense = _dense_terminal_loss(
        hidden_dense,
        predicted_dense,
        targets,
        head_dense,
        mask,
        hidden_weight=1.0,
        kl_weight=1.0,
        softcap=4.0,
    )
    terminal = nextlat_terminal_loss(
        hidden_terminal,
        predicted_terminal,
        targets,
        head_terminal,
        transition_mask=mask,
        softcap=4.0,
        token_chunk_size=2,
    )
    torch.testing.assert_close(terminal.ce_sum, dense[0], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(terminal.kl_loss, dense[2], rtol=2e-3, atol=2e-3)
    dense_objective = dense[0] + targets.numel() * dense[3]
    terminal_objective = terminal.ce_sum + targets.numel() * terminal.total
    dense_objective.backward()
    terminal_objective.backward()
    torch.testing.assert_close(
        hidden_terminal.grad, hidden_dense.grad, rtol=3e-2, atol=2e-2
    )
    torch.testing.assert_close(
        predicted_terminal.grad, predicted_dense.grad, rtol=3e-2, atol=2e-2
    )
    torch.testing.assert_close(
        head_terminal.weight.grad, head_dense.weight.grad, rtol=3e-2, atol=2e-2
    )
    torch.testing.assert_close(
        head_terminal.bias.grad, head_dense.bias.grad, rtol=3e-2, atol=2e-2
    )


def test_terminal_does_not_save_full_vocabulary_activations() -> None:
    hidden = torch.randn(3, 5, 7, requires_grad=True)
    predicted = torch.randn(3, 4, 7, requires_grad=True)
    head = nn.Linear(7, 29)
    targets = torch.randint(0, 29, (3, 5))
    mask = torch.ones(3, 4, dtype=torch.bool)
    result = nextlat_terminal_loss(
        hidden,
        predicted,
        targets,
        head,
        transition_mask=mask,
        token_chunk_size=3,
    )
    saved_shapes = [tensor.shape for tensor in result.ce_sum.grad_fn.saved_tensors]
    assert saved_shapes
    assert not any(len(shape) >= 3 and shape[-1] == 29 for shape in saved_shapes)
    (result.ce_sum + result.total).backward()


def test_document_mask_and_masked_smooth_l1_exclude_crossings() -> None:
    document_ids = torch.tensor([[0, 0, 1, 1], [4, 5, 5, 5]])
    mask = document_transition_mask(document_ids)
    expected_mask = torch.tensor([[True, False, True], [False, True, True]])
    assert torch.equal(mask, expected_mask)

    actual = torch.zeros(2, 3, 2, requires_grad=True)
    predicted = torch.tensor(
        [
            [[1.0, 0.0], [100.0, 100.0], [0.0, 2.0]],
            [[100.0, 100.0], [0.5, -0.5], [1.0, 1.0]],
        ],
        requires_grad=True,
    )
    loss = masked_smooth_l1(predicted, actual, mask)
    dense_elements = F.smooth_l1_loss(
        predicted.float(), actual.detach().float(), reduction="none"
    )
    expected = dense_elements[mask].mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert actual.grad is None
    assert predicted.grad is not None
    assert torch.count_nonzero(predicted.grad[~mask]) == 0


def test_terminal_exposes_separate_detached_metrics() -> None:
    predicted = torch.randn(2, 3, 4, requires_grad=True)
    hidden = torch.randn(2, 4, 4, requires_grad=True)
    head = nn.Linear(4, 9)
    targets = torch.randint(0, 9, (2, 4))
    mask = torch.tensor([[True, False, True], [True, True, False]])
    losses = nextlat_terminal_loss(
        hidden,
        predicted,
        targets,
        head,
        transition_mask=mask,
        hidden_weight=2.0,
        kl_weight=0.25,
    )
    torch.testing.assert_close(
        losses.total, 2.0 * losses.hidden_loss + 0.25 * losses.kl_loss
    )
    metrics = losses.detached_metrics()
    assert set(metrics) == {
        "nextlat/hidden_loss",
        "nextlat/kl_loss",
        "nextlat/total",
    }
    assert all(not value.requires_grad for value in metrics.values())


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (
            lambda: nextlat_terminal_loss(
                torch.randn(2, 4, 5),
                torch.randn(2, 2, 5),
                torch.zeros(2, 4, dtype=torch.long),
                nn.Linear(5, 7),
                transition_mask=torch.ones(2, 3, dtype=torch.bool),
            ),
            ValueError,
            "predicted_hidden",
        ),
        (
            lambda: nextlat_terminal_loss(
                torch.randn(2, 4, 5),
                torch.randn(2, 3, 5),
                torch.zeros(2, 4),
                nn.Linear(5, 7),
                transition_mask=torch.ones(2, 3, dtype=torch.bool),
            ),
            TypeError,
            "torch.long",
        ),
        (
            lambda: nextlat_terminal_loss(
                torch.randn(2, 4, 5),
                torch.randn(2, 3, 5),
                torch.zeros(2, 4, dtype=torch.long),
                nn.Linear(5, 7),
                transition_mask=torch.ones(2, 3),
            ),
            TypeError,
            "boolean",
        ),
        (
            lambda: nextlat_terminal_loss(
                torch.randn(2, 4, 5),
                torch.randn(2, 3, 5),
                torch.zeros(2, 4, dtype=torch.long),
                nn.Linear(5, 7),
                transition_mask=torch.ones(2, 3, dtype=torch.bool),
                token_chunk_size=0,
            ),
            ValueError,
            "positive integer",
        ),
        (
            lambda: nextlat_terminal_loss(
                torch.randn(2, 4, 5),
                torch.randn(2, 3, 5),
                torch.zeros(2, 4, dtype=torch.long),
                nn.Linear(5, 7),
                transition_mask=torch.ones(2, 3, dtype=torch.bool),
                softcap=0.0,
            ),
            ValueError,
            "softcap",
        ),
        (
            lambda: masked_smooth_l1(
                torch.randn(2, 3, 4),
                torch.randn(2, 3, 4),
                torch.ones(2, 2, dtype=torch.bool),
            ),
            ValueError,
            "leading shape",
        ),
        (
            lambda: document_transition_mask(torch.tensor(1)),
            ValueError,
            "at least one dimension",
        ),
    ],
)
def test_invalid_loss_inputs(call, error, match) -> None:
    with pytest.raises(error, match=match):
        call()

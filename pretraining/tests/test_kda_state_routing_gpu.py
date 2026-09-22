"""GPU numerical/gradient contracts, not training-quality experiments.

Run only through mlq: .venv/bin/python -m pytest -q this_file.
"""

import copy
import os

# The independent oracle uses FLA's native Triton backend. TileLang/TVM's
# optional FFI extension fails during import in this Python 3.14 environment;
# production shared-state execution does not use either FLA backward backend.
os.environ["FLA_TILELANG"] = "0"

import pytest
import torch


from pretraining.nanogpt_mini.nanogpt_mini_kda_model import KDAGPT
from pretraining.state_routing.model import (
    SharedStateExecutor,
    dense_step,
    install_shared_state_routing,
    routed_kda_step,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires queued CUDA"
)


def model():
    torch.manual_seed(414)
    net = KDAGPT(
        vocab_size=128,
        num_layers=8,
        model_dim=128,
        mlp_hidden=192,
        delta_num_heads=3,
        delta_layer_indices=[0, 1, 2, 4, 5, 6],
    ).cuda()
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            if name.endswith("gains") or name.endswith("o_norm.weight"):
                parameter.fill_(1)
            elif name.endswith("A_log"):
                parameter.zero_()
            elif name.endswith("dt_bias"):
                parameter.fill_(-3)
            else:
                parameter.normal_(std=0.03)
    return net


def test_dense_attention_forward_and_gradients_unchanged():
    net = model()
    attn = net.blocks[3].attn
    x = torch.randn(2, 7, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expected = attn(x)
    key = torch.zeros(2, 1, 7, 128, device="cuda", dtype=x.dtype)
    value = torch.zeros_like(key)
    outputs = []
    for t in range(7):
        output, key, value = dense_step(
            attn, x[:, t], key, value, torch.tensor(t, device="cuda")
        )
        outputs.append(output)
    actual = torch.stack(outputs, 1)
    torch.testing.assert_close(actual, expected, atol=0.006, rtol=0.03)
    grad = torch.randn_like(expected)
    a = torch.autograd.grad(actual, x, grad)[0]
    b = torch.autograd.grad(expected, x, grad)[0]
    torch.testing.assert_close(a, b, atol=0.012, rtol=0.04)


def test_private_routes_reduce_to_reference_kda():
    from fla.modules import ShortConvolution
    from fla.ops.kda import chunk_kda

    net = model()
    attn = net.blocks[0].attn
    install_shared_state_routing(net, compile_step=False)
    x = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    convolved = []
    for projection, original_conv in (
        (attn.q_proj, attn.q_conv1d),
        (attn.k_proj, attn.k_conv1d),
        (attn.v_proj, attn.v_conv1d),
    ):
        conv = ShortConvolution(384, 4, bias=False, activation="silu").cuda()
        conv.weight = original_conv.weight
        value, _ = conv(projection(x))
        convolved.append(value.view(2, 8, 3, 128))
    raw, final_state = chunk_kda(
        q=convolved[0],
        k=convolved[1],
        v=convolved[2],
        g=attn.f_b_proj(attn.f_a_proj(x)).view(2, 8, 3, 128),
        beta=attn.b_proj(x).float(),
        A_log=attn.A_log,
        dt_bias=attn.dt_bias,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        state_v_first=True,
        disable_recompute=True,
    )
    expected = attn.o_proj(attn.o_norm(raw, attn._output_gate(x)).flatten(2))
    pool = torch.zeros(2, 12, 3, 128, 128, device="cuda")
    history = torch.zeros(3, 2, 384, 3, device="cuda", dtype=x.dtype)
    outputs = []
    for t in range(x.size(1)):
        output, pool, history, _, _ = routed_kda_step(attn, x[:, t], pool, history, 4)
        outputs.append(output)
    torch.testing.assert_close(torch.stack(outputs, 1), expected, atol=0.008, rtol=0.04)
    torch.testing.assert_close(pool[:, 4], final_state, atol=0.003, rtol=0.04)
    assert pool[:, :4].count_nonzero() == 0
    assert pool[:, 5:].count_nonzero() == 0
    actual = torch.stack(outputs, 1)
    upstream = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad(actual, x, upstream)[0]
    reference_gradient = torch.autograd.grad(expected, x, upstream)[0]
    torch.testing.assert_close(
        actual_gradient, reference_gradient, atol=0.02, rtol=0.06
    )


def test_full_model_causality_and_cross_layer_credit():
    net = model()
    execution = install_shared_state_routing(
        net, compile_step=False, fixed_indices=(0,) * 6
    )
    embedded = torch.randn(
        2, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    def run(z):
        state = execution.initial_state(2, 4, z.device)
        outputs = []
        for t in range(4):
            output, *state, _, _ = execution.step(
                z[:, t], *state, torch.tensor(t, device=z.device)
            )
            outputs.append(output)
        return torch.stack(outputs, 1)

    outputs = run(embedded)
    changed = embedded.detach().clone()
    changed[:, 2:] += 2
    torch.testing.assert_close(run(changed)[:, :2], outputs[:, :2], rtol=0, atol=0)
    # Isolate the bank-mediated path: late writer at token 0, early reader
    # at token 1, with independent layer-local convolution histories.
    pool, histories, _, _ = execution.initial_state(2, 2, embedded.device)
    writer = embedded[:, 0]
    _, pool, _, _, _ = routed_kda_step(
        net.blocks[6].attn, writer, pool, histories[5], 0
    )
    reader = embedded[:, 1].detach()
    read, _, _, _, _ = routed_kda_step(
        net.blocks[0].attn, reader, pool, histories[0], 0
    )
    gradient = torch.autograd.grad(read.float().square().sum(), embedded)[0]
    assert gradient[:, 0].float().norm() > 0
    assert torch.isfinite(gradient).all()


def test_router_gets_language_loss_gradient_and_only_one_bank_changes():
    net = model()
    install_shared_state_routing(net, compile_step=False, balance_weight=0)
    attn = net.blocks[0].attn
    x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(3, 12, 3, 128, 128, device="cuda") * 0.01
    history = torch.zeros(3, 3, 384, 3, device="cuda", dtype=x.dtype)
    output, updated, _, _, selected = routed_kda_step(attn, x, pool, history)
    changed = (updated != pool).flatten(2).any(-1)
    torch.testing.assert_close(changed, selected.bool())
    output.float().square().sum().backward()
    assert attn.state_router.weight.grad.norm() > 0
    assert torch.isfinite(attn.state_router.weight.grad).all()


def test_compiled_checkpointed_full_bptt_matches_uncheckpointed():
    net = model()
    install_shared_state_routing(net, compile_step=False, checkpoint_tokens=2)
    reference = copy.deepcopy(net)
    eager = copy.deepcopy(net)
    # Deepcopy preserves the executor/model cycle; build a fresh execution
    # policy without replacing the paired router parameters.
    reference.shared_state_executor = SharedStateExecutor(
        reference,
        compile_step=True,
        checkpoint_tokens=2,
        balance_weight=0,
        checkpointing=False,
    )
    net.shared_state_executor = SharedStateExecutor(
        net, compile_step=True, checkpoint_tokens=2, balance_weight=0
    )
    eager.shared_state_executor = SharedStateExecutor(
        eager,
        compile_step=False,
        checkpoint_tokens=2,
        balance_weight=0,
        checkpointing=False,
    )
    tokens = torch.randint(0, 128, (2, 4), device="cuda")
    targets = torch.randint(0, 128, (2, 4), device="cuda")
    expected = reference(tokens, targets)
    actual = net(tokens, targets)
    eager_loss = eager(tokens, targets)
    expected.backward()
    actual.backward()
    eager_loss.backward()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual, eager_loss, atol=0.05, rtol=0.002)
    for (name, p), (other_name, q), (_, e) in zip(
        net.named_parameters(),
        reference.named_parameters(),
        eager.named_parameters(),
        strict=True,
    ):
        assert name == other_name
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        # Isolate checkpoint correctness using identical compiled arithmetic.
        # The independent eager comparison also checks BF16 numerical parity;
        # relative-only checks are ill-conditioned for cancelling bias grads.
        torch.testing.assert_close(p.grad, q.grad, atol=1e-4, rtol=0.005, msg=name)
        torch.testing.assert_close(p.grad, e.grad, atol=0.035, rtol=0.09, msg=name)
        reference_norm = q.grad.float().norm()
        if reference_norm > 1e-6:
            relative_error = (p.grad.float() - q.grad.float()).norm() / reference_norm
            assert relative_error < 0.005, (name, float(relative_error))


def test_sequence_routing_fused_path_forward_backward():
    """The throughput path must exercise FLA's varlen bank scan."""
    net = model()
    install_shared_state_routing(net, compile_step=False, fast_sequence=True)
    tokens = torch.randint(0, 128, (2, 16), device="cuda")
    targets = torch.randint(0, 128, (2, 16), device="cuda")
    loss = net(tokens, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert net.blocks[0].attn.state_router.weight.grad is not None
    assert torch.isfinite(net.blocks[0].attn.state_router.weight.grad).all()

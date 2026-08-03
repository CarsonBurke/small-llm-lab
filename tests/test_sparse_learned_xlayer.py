from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xlayer.sparse import sparse_learned_xlayer_train_gpt as learned
from xlayer.sparse.sparse_persistent_xlayer_train_gpt import graph_indices
import train_gpt

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/Triton required")


def _random_case(
    bsz=2, heads=2, kv_heads=1, seqlen=16, dim=16, kb=8, segs=2, *, bias_scale=1.0, seed=11
):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    q = torch.randn(bsz, heads, seqlen, dim, device=device)
    k = torch.randn(bsz, kv_heads, segs * seqlen, dim, device=device)
    v = torch.randn(bsz, kv_heads, segs * seqlen, dim, device=device)
    src = torch.randint(0, segs, (heads, kb), dtype=torch.int32)
    lag = torch.randint(0, seqlen, (heads, kb), dtype=torch.int32)
    # graph_indices assumes duplicate-free (source, lag) wires per head, as
    # top-k selection guarantees in the training script.
    for h in range(heads):
        flat = (src[h].to(torch.int64) * seqlen + lag[h]).tolist()
        while len(set(flat)) != kb:
            src[h] = torch.randint(0, segs, (kb,), dtype=torch.int32)
            lag[h] = torch.randint(0, seqlen, (kb,), dtype=torch.int32)
            flat = (src[h].to(torch.int64) * seqlen + lag[h]).tolist()
    idx, valid = graph_indices(src.to(device), lag.to(device), bsz, seqlen)
    bias = bias_scale * torch.randn(heads, kb, device=device)
    null = torch.randn(heads, device=device)
    return q, k, v, idx, valid, bias, null


def _reference_forward(q, k, v, idx, valid, bias, gate, null_bias, scale):
    """fp64 gather + bisection entmax-1.5 with per-(head, slot) bias + gate."""
    q, k, v = q.double(), k.double(), v.double()
    bias, gate, null_bias = bias.double(), gate.double(), null_bias.double()
    group = q.size(1) // k.size(1)
    k_exp = k.repeat_interleave(group, dim=1)
    v_exp = v.repeat_interleave(group, dim=1)
    bsz, heads, seqlen, kb = idx.shape
    dim = q.size(-1)
    flat = idx.to(torch.int64).reshape(bsz, heads, seqlen * kb, 1).expand(-1, -1, -1, dim)
    k_rows = k_exp.gather(2, flat).view(bsz, heads, seqlen, kb, dim)
    v_rows = v_exp.gather(2, flat).view(bsz, heads, seqlen, kb, dim)
    logits = (k_rows * q.unsqueeze(3)).sum(-1) * scale + bias.view(1, heads, 1, kb)
    logits = logits.masked_fill(~valid, float("-inf"))
    nl = null_bias.view(1, heads, 1).expand(bsz, heads, seqlen)
    mx = torch.maximum(logits.max(-1).values, nl)
    x = (logits - mx.unsqueeze(-1)) * 0.5
    xn = (nl - mx) * 0.5
    lo = torch.full_like(mx, -1.0)
    hi = torch.zeros_like(mx)
    for _ in range(80):
        tau = 0.5 * (lo + hi)
        f = (
            torch.clamp(x - tau.unsqueeze(-1), min=0.0).square().sum(-1)
            + torch.clamp(xn - tau, min=0.0).square()
            - 1.0
        )
        lo = torch.where(f > 0, tau, lo)
        hi = torch.where(f > 0, hi, tau)
    tau = 0.5 * (lo + hi)
    p = torch.clamp(x - tau.unsqueeze(-1), min=0.0).square()
    pn = torch.clamp(xn - tau, min=0.0).square()
    z = p.sum(-1) + pn
    p = p / z.unsqueeze(-1)
    pn = pn / z
    y = ((p * gate.view(1, -1, 1, gate.size(-1))).unsqueeze(-1) * v_rows).sum(3)
    return y.float(), p.float(), pn.float()


@cuda_only
def test_zero_bias_matches_unbiased_kernel_exactly() -> None:
    from xlayer.sparse.sparse_entmax_kernel import _sparse_entmax_attn_fwd
    from xlayer.sparse.sparse_wire_bias_kernel import _sparse_entmax_bias_attn_fwd

    q, k, v, idx, valid, _, null = _random_case()
    bias = torch.zeros(q.size(1), idx.size(-1), device=q.device)
    gate = torch.ones(q.size(1), idx.size(-1), device=q.device)

    q1 = q.clone().requires_grad_(True)
    k1 = k.clone().requires_grad_(True)
    v1 = v.clone().requires_grad_(True)
    n1 = null.clone().requires_grad_(True)
    b1 = bias.clone().requires_grad_(True)
    g1 = gate.clone().requires_grad_(True)
    q2 = q.clone().requires_grad_(True)
    k2 = k.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True)
    n2 = null.clone().requires_grad_(True)

    out1 = _sparse_entmax_bias_attn_fwd(q1, k1, v1, idx, valid, b1, g1, n1, 0.25)
    out2 = _sparse_entmax_attn_fwd(q2, k2, v2, idx, valid, n2, 0.25)
    for lhs, rhs in zip(out1, out2, strict=True):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
    grad_out = torch.randn_like(out1[0])
    out1[0].backward(grad_out)
    out2[0].backward(grad_out)
    # At gate == 1 the input gradients are exact-math identical to the
    # unbiased kernel, but the gate load and the two extra atomics change the
    # compiled kernel's instruction scheduling, so fp32 reduction rounding in
    # dq/dk/dv can drift by a last ulp.  Forward outputs stay bit-exact.
    torch.testing.assert_close(q1.grad, q2.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(k1.grad, k2.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(v1.grad, v2.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(n1.grad, n2.grad, rtol=0, atol=0)
    assert b1.grad is not None and torch.isfinite(b1.grad).all()
    assert g1.grad is not None and torch.isfinite(g1.grad).all()


@cuda_only
def test_biased_gated_forward_matches_fp64_reference() -> None:
    from xlayer.sparse.sparse_wire_bias_kernel import _sparse_entmax_bias_attn_fwd

    q, k, v, idx, valid, bias, null = _random_case(bias_scale=2.0, seed=3)
    gate = 1.0 + 0.3 * torch.randn_like(bias)
    y, p, pn = _sparse_entmax_bias_attn_fwd(q, k, v, idx, valid, bias, gate, null, 0.25)
    ry, rp, rpn = _reference_forward(q, k, v, idx, valid, bias, gate, null, 0.25)
    torch.testing.assert_close(p, rp, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(pn, rpn, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y, ry, rtol=1e-4, atol=1e-4)


@cuda_only
def test_dbias_matches_finite_differences() -> None:
    from xlayer.sparse.sparse_wire_bias_kernel import _sparse_entmax_bias_attn_fwd

    q, k, v, idx, valid, bias, null = _random_case(
        bsz=1, heads=2, seqlen=8, kb=8, bias_scale=1.5, seed=5
    )
    gate = 1.0 + 0.2 * torch.randn_like(bias)
    weight = torch.randn(q.shape, device=q.device)

    def value(b: torch.Tensor) -> torch.Tensor:
        y, _, _ = _sparse_entmax_bias_attn_fwd(q, k, v, idx, valid, b, gate, null, 0.25)
        return (y * weight).sum()

    b = bias.clone().requires_grad_(True)
    value(b).backward()
    analytic = b.grad.clone()

    eps = 2e-3
    fd = torch.zeros_like(bias)
    for h in range(bias.size(0)):
        for e in range(bias.size(1)):
            up = bias.clone()
            up[h, e] += eps
            down = bias.clone()
            down[h, e] -= eps
            fd[h, e] = (value(up) - value(down)) / (2 * eps)
    torch.testing.assert_close(analytic, fd, rtol=5e-2, atol=5e-3)


@cuda_only
def test_dgate_matches_finite_differences() -> None:
    from xlayer.sparse.sparse_wire_bias_kernel import _sparse_entmax_bias_attn_fwd

    q, k, v, idx, valid, bias, null = _random_case(
        bsz=1, heads=2, seqlen=8, kb=8, bias_scale=1.5, seed=9
    )
    # Perturb around a non-identity base point so the FD probes the general
    # dgate = sum_rows p * (dy . v) path, not just gate == 1.
    gate0 = 1.0 + 0.2 * torch.randn_like(bias)
    weight = torch.randn(q.shape, device=q.device)

    def value(g: torch.Tensor) -> torch.Tensor:
        y, _, _ = _sparse_entmax_bias_attn_fwd(q, k, v, idx, valid, bias, g, null, 0.25)
        return (y * weight).sum()

    g = gate0.clone().requires_grad_(True)
    value(g).backward()
    analytic = g.grad.clone()

    eps = 2e-3
    fd = torch.zeros_like(gate0)
    for h in range(gate0.size(0)):
        for e in range(gate0.size(1)):
            up = gate0.clone()
            up[h, e] += eps
            down = gate0.clone()
            down[h, e] -= eps
            fd[h, e] = (value(up) - value(down)) / (2 * eps)
    torch.testing.assert_close(analytic, fd, rtol=5e-2, atol=5e-3)


@cuda_only
def test_xlayer_bias_list_wrapper_matches_flat_bank() -> None:
    from xlayer.sparse.sparse_wire_bias_kernel import (
        _sparse_entmax_bias_attn_fwd,
        xlayer_entmax_bias_attention_stats,
    )

    q, k, v, idx, valid, bias, null = _random_case(segs=2, seed=7)
    seqlen = q.size(2)
    q1 = q.clone().to(torch.bfloat16).requires_grad_(True)
    ks1 = [
        k[:, :, :seqlen].clone().to(torch.bfloat16).requires_grad_(True),
        k[:, :, seqlen:].clone().to(torch.bfloat16).requires_grad_(True),
    ]
    vs1 = [
        v[:, :, :seqlen].clone().to(torch.bfloat16).requires_grad_(True),
        v[:, :, seqlen:].clone().to(torch.bfloat16).requires_grad_(True),
    ]
    gate = 1.0 + 0.3 * torch.randn_like(bias)
    b1 = bias.clone().requires_grad_(True)
    g1 = gate.clone().requires_grad_(True)
    n1 = null.clone().requires_grad_(True)
    q2 = q1.detach().clone().requires_grad_(True)
    k2 = torch.cat([t.detach() for t in ks1], dim=2).requires_grad_(True)
    v2 = torch.cat([t.detach() for t in vs1], dim=2).requires_grad_(True)
    b2 = bias.clone().requires_grad_(True)
    g2 = gate.clone().requires_grad_(True)
    n2 = null.clone().requires_grad_(True)

    out1 = xlayer_entmax_bias_attention_stats(q1, ks1, vs1, idx, valid, b1, g1, n1, 0.25)
    out2 = _sparse_entmax_bias_attn_fwd(q2, k2, v2, idx, valid, b2, g2, n2, 0.25)
    for lhs, rhs in zip(out1, out2, strict=True):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
    out1[0].float().sum().backward()
    out2[0].float().sum().backward()
    torch.testing.assert_close(q1.grad, q2.grad, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([t.grad for t in ks1], dim=2), k2.grad, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([t.grad for t in vs1], dim=2), v2.grad, rtol=0, atol=0)
    torch.testing.assert_close(b1.grad, b2.grad, rtol=0, atol=0)
    torch.testing.assert_close(g1.grad, g2.grad, rtol=0, atol=0)
    torch.testing.assert_close(n1.grad, n2.grad, rtol=0, atol=0)


def _make_module(layer: int = 1, seqlen: int = 16, k_budget: int = 8):
    module = learned.LearnedGraphAttention(32, 4, 2, 10_000.0, 1.5)
    module.k_budget = k_budget
    module.configure_graph(layer, seqlen)
    return module


@cuda_only
def test_sampling_is_deterministic_per_epoch_and_argmax_at_eval() -> None:
    module = _make_module().cuda()
    learned._STEP_BUF.fill_(3)
    first = module._sample_wires(16)
    second = module._sample_wires(16)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    # A step change re-keys the Gumbel draws, so the topology resamples.
    learned._STEP_BUF.fill_(4)
    third = module._sample_wires(16)
    assert not (
        torch.equal(first[0], third[0]) and torch.equal(first[1], third[1])
    )

    module.eval()
    learned._STEP_BUF.fill_(3)
    eval_a = module._sample_wires(16)
    learned._STEP_BUF.fill_(999)
    eval_b = module._sample_wires(16)
    assert torch.equal(eval_a[0], eval_b[0]) and torch.equal(eval_a[1], eval_b[1])

    scores = module._logit_surface().detach().reshape(module.num_heads, -1)
    expected = scores.topk(module.k_budget, dim=-1).indices
    got = eval_a[0].to(torch.int64) * 16 + eval_a[1].to(torch.int64)
    assert torch.equal(got.sort(-1).values, expected.sort(-1).values)


@cuda_only
def test_topology_is_constant_within_a_hold_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GRAPH_HOLD_STEPS defaults to 1 (resample every step); pin a window of
    # 4 to exercise the epoch-keyed mechanism the knob exposes.
    monkeypatch.setattr(learned, "_HOLD_STEPS", 4)
    module = _make_module().cuda()
    hold = learned._HOLD_STEPS
    learned._STEP_BUF.fill_(hold)
    base = module._sample_wires(16)
    for step in range(hold + 1, 2 * hold):
        learned._STEP_BUF.fill_(step)
        same = module._sample_wires(16)
        assert torch.equal(base[0], same[0]) and torch.equal(base[1], same[1])
    learned._STEP_BUF.fill_(2 * hold)
    fresh = module._sample_wires(16)
    assert not (torch.equal(base[0], fresh[0]) and torch.equal(base[1], fresh[1]))
    learned._STEP_BUF.zero_()


@cuda_only
def test_constant_tau_explores_forever_and_zero_tau_is_argmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # There is no anneal schedule: exploration persists at ANY step ...
    module = _make_module().cuda()
    learned._STEP_BUF.fill_(100_000)
    trained = module._sample_wires(16)
    module.eval()
    argmax = module._sample_wires(16)
    assert not (torch.equal(trained[0], argmax[0]) and torch.equal(trained[1], argmax[1]))

    # ... and tau = 0 degenerates to the eval argmax graph exactly.
    monkeypatch.setattr(learned, "_TAU", 0.0)
    module.train()
    greedy = module._sample_wires(16)
    assert torch.equal(greedy[0], argmax[0]) and torch.equal(greedy[1], argmax[1])
    learned._STEP_BUF.zero_()


@cuda_only
def test_sampled_wires_are_unique_causal_and_in_range() -> None:
    module = _make_module(layer=2, seqlen=16, k_budget=16).cuda()
    for step in (0, 1, 17):
        learned._STEP_BUF.fill_(step)
        src, lag = module._sample_wires(16)
        assert torch.all((src >= 0) & (src <= 2))
        assert torch.all((lag >= 0) & (lag < 16))
        flat = src.to(torch.int64) * 16 + lag.to(torch.int64)
        for head in range(flat.size(0)):
            assert flat[head].unique().numel() == module.k_budget
        idx, valid = graph_indices(src, lag, bsz=2, seqlen=16)
        assert int(idx.max()) < 3 * 16
        pos = torch.arange(16, device=idx.device).view(1, 1, -1, 1)
        assert torch.all((idx % 16)[valid.expand_as(idx)] <= pos.expand_as(idx)[valid.expand_as(idx)])
    learned._STEP_BUF.zero_()


@cuda_only
def test_end_to_end_gradient_reaches_graph_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(learned, "_TRAIN_SEQ_LEN", 16)
    module = _make_module(layer=0, seqlen=16).cuda().bfloat16()
    learned._STATE["i"] = 0
    learned._STATE["kv"] = None
    learned._STEP_BUF.zero_()
    x = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16)
    y = module(x)
    assert y.shape == (2, 16, 32)
    y.float().square().sum().backward()
    assert module.wire_logit_lag.grad is not None
    assert module.wire_logit_src.grad is not None
    assert torch.isfinite(module.wire_logit_lag.grad).all()
    assert module.wire_logit_lag.grad.abs().sum() > 0
    # dU = dtheta * V-rows with V randomly initialized, so U gets signal from
    # step one despite its zero init; dV is proportional to U and is exactly
    # zero at init, so only existence/finiteness can be asserted there.
    assert module.wire_logit_u.grad is not None
    assert torch.isfinite(module.wire_logit_u.grad).all()
    assert module.wire_logit_u.grad.abs().sum() > 0
    assert module.wire_logit_v.grad is not None
    assert torch.isfinite(module.wire_logit_v.grad).all()
    assert module.null_bias.grad is not None


def test_graph_tables_serialize_and_survive_int8_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        train_gpt,
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        train_gpt.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS + ("null_bias", "wire_logit"),
    )
    module = _make_module(layer=2, seqlen=32)
    with torch.no_grad():
        module.wire_logit_lag.add_(torch.randn_like(module.wire_logit_lag))
        module.wire_logit_u.add_(torch.randn_like(module.wire_logit_u))
    state = copy.deepcopy(module.state_dict())
    assert "wire_logit_src" in state
    assert "wire_logit_lag" in state
    assert "wire_logit_u" in state
    assert "wire_logit_v" in state
    assert "prev_wires" not in state

    restored = _make_module(layer=2, seqlen=32)
    restored.load_state_dict(state, strict=True)
    assert torch.equal(restored.wire_logit_lag, module.wire_logit_lag)

    quantized, _ = train_gpt.quantize_state_dict_int8(state)
    roundtrip = train_gpt.dequantize_state_dict_int8(quantized)
    for name in ("wire_logit_src", "wire_logit_lag", "wire_logit_u", "wire_logit_v"):
        assert torch.equal(roundtrip[name], state[name])
        assert roundtrip[name].dtype == torch.float32


def test_straight_through_gate_is_bitwise_one() -> None:
    # (theta - theta.detach()) evaluates x - x elementwise, which is exactly
    # +0.0 in IEEE754, so the forward gate is bitwise 1.0 for ANY theta while
    # d(gate)/d(theta) == 1.  (The reversed order 1.0 + theta - theta.detach()
    # would round.)
    theta = (1e4 * torch.randn(64, dtype=torch.float32)).requires_grad_(True)
    gate = (theta - theta.detach()) + 1.0
    assert (gate == 1.0).all()
    gate.sum().backward()
    assert torch.equal(theta.grad, torch.ones_like(theta))


def test_gumbel_noise_is_finite_and_step_dependent() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the step buffer")
    step_a = torch.tensor([5], dtype=torch.int64, device="cuda")
    step_b = torch.tensor([6], dtype=torch.int64, device="cuda")
    noise_a = learned.gumbel_noise(1, 4, 64, step_a)
    noise_b = learned.gumbel_noise(1, 4, 64, step_b)
    assert noise_a.shape == (4, 64)
    assert torch.isfinite(noise_a).all()
    assert not torch.equal(noise_a, noise_b)
    assert torch.equal(noise_a, learned.gumbel_noise(1, 4, 64, step_a))
    assert float(noise_a.min()) > -3.0
    assert float(noise_a.max()) < 17.0


@cuda_only
def test_compiled_forward_does_not_recompile_across_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    from torch._dynamo.testing import CompileCounter

    monkeypatch.setattr(learned, "_TRAIN_SEQ_LEN", 16)
    module = _make_module(layer=0, seqlen=16).cuda().bfloat16()

    counter = CompileCounter()
    compiled = torch.compile(module, backend=counter, fullgraph=True)
    x = torch.randn(2, 16, 32, device="cuda", dtype=torch.bfloat16)
    # Prime the baseline Rotary cos/sin cache eagerly: its None -> tensor
    # transition costs one recompile for EVERY arm (baseline included) and is
    # absorbed by warmup; this test pins step-to-step stability after it.
    learned._STATE["i"] = 0
    learned._STATE["kv"] = None
    module(x)
    learned._STEP_BUF.zero_()
    learned._STATE["i"] = 0
    learned._STATE["kv"] = None
    first = compiled(x)
    # Advance a full hold window so the graph epoch (and thus the sampled
    # topology) changes; the tensor clock must alter the output with no
    # recompilation.
    learned._STEP_BUF.add_(learned._HOLD_STEPS)
    learned._STATE["i"] = 0
    learned._STATE["kv"] = None
    second = compiled(x)
    assert counter.frame_count == 1
    assert not torch.equal(first, second)
    learned._STEP_BUF.zero_()

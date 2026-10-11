"""CPU contracts for the triadic KDA recurrence and its chunk-parallel form."""

import pytest
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini.nanogpt_mini_kda_model import (
    _l2norm,
    kda_decay_gate,
    reference_kda_recurrence,
)
from pretraining.nanogpt_mini.triadic_kda import (
    decayed_gram,
    slice_gram,
    triadic_chunk_kda,
    triadic_reference_recurrence,
)


def _inputs(B=2, T=96, H=2, Dk=8, Dv=6, E=3, seed=0, dtype=torch.float64,
            decay_scale=1.0):
    g = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, dtype=dtype)

    q = _l2norm(randn(B, T, H, Dk)) * Dk**-0.5
    k = _l2norm(randn(B, T, H, Dk))
    v = randn(B, T, H, Dv)
    q2 = _l2norm(F.softplus(randn(B, T, H, E)))
    k2 = _l2norm(F.softplus(randn(B, T, H, E)))
    log_decay = -5.0 * decay_scale * torch.sigmoid(randn(B, T, H, Dk))
    slice_log_decay = -5.0 * decay_scale * torch.sigmoid(randn(B, T, H, E))
    beta = torch.sigmoid(randn(B, T, H))
    return q, k, v, q2, k2, log_decay, slice_log_decay, beta


def test_chunk_form_matches_step_recurrence():
    inputs = _inputs()
    o_ref, s_ref = triadic_reference_recurrence(*inputs)
    for chunk in (16, 32, 64):
        o, s = triadic_chunk_kda(*inputs, chunk_size=chunk)
        torch.testing.assert_close(o, o_ref, rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(s, s_ref, rtol=1e-9, atol=1e-11)


def test_chunk_form_carries_an_initial_state_and_ragged_length():
    inputs = _inputs(T=70, seed=1)
    state = torch.randn(2, 2, 3, 6, 8, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(9))
    o_ref, s_ref = triadic_reference_recurrence(*inputs, initial_state=state)
    o, s = triadic_chunk_kda(*inputs, initial_state=state)
    assert o.shape == o_ref.shape
    torch.testing.assert_close(o, o_ref, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(s, s_ref, rtol=1e-9, atol=1e-11)


def test_single_slice_is_exactly_kda():
    """E = 1 with unit second keys and no slice decay is the KDA oracle."""
    g = torch.Generator().manual_seed(2)
    B, T, H, D = 2, 80, 3, 8
    q, k, v = (torch.randn(B, T, H, D, generator=g) for _ in range(3))
    decay_logits = torch.randn(B, T, H, D, generator=g)
    beta_logits = torch.randn(B, T, H, generator=g)
    A_log = torch.randn(H, generator=g) * 0.1
    dt_bias = torch.randn(H * D, generator=g)
    o_kda, s_kda = reference_kda_recurrence(
        q, k, v, decay_logits, beta_logits, A_log, dt_bias
    )
    ones = torch.ones(B, T, H, 1)
    o, s = triadic_chunk_kda(
        _l2norm(q) * D**-0.5, _l2norm(k), v, ones, ones,
        kda_decay_gate(decay_logits, A_log, dt_bias),
        torch.zeros(B, T, H, 1), torch.sigmoid(beta_logits),
    )
    torch.testing.assert_close(o, o_kda, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(s[:, :, 0], s_kda, rtol=2e-4, atol=2e-5)


def test_triadic_is_kda_on_the_joint_key():
    """Flattening (slice, channel) gives KDA with key k2 (x) k."""
    q, k, v, q2, k2, log_decay, slice_log_decay, beta = _inputs(seed=3)
    B, T, H, Dk = k.shape
    E, Dv = k2.size(-1), v.size(-1)
    joint_k = (k2[..., :, None] * k[..., None, :]).flatten(-2)
    joint_q = (q2[..., :, None] * q[..., None, :]).flatten(-2)
    joint_decay = (slice_log_decay[..., :, None] + log_decay[..., None, :]).flatten(-2)
    state = k.new_zeros(B, H, Dv, E * Dk)
    outputs = []
    for t in range(T):
        decayed = state * joint_decay[:, t].exp()[:, :, None, :]
        delta = v[:, t] - torch.einsum("bhvk,bhk->bhv", decayed, joint_k[:, t])
        state = decayed + (
            beta[:, t, :, None, None] * delta[..., None] * joint_k[:, t, :, None, :]
        )
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, joint_q[:, t]))
    o, s = triadic_chunk_kda(q, k, v, q2, k2, log_decay, slice_log_decay, beta)
    torch.testing.assert_close(o, torch.stack(outputs, 1), rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(
        s, state.unflatten(-1, (E, Dk)).movedim(-2, 2), rtol=1e-9, atol=1e-11
    )


def test_gradients_match_the_step_recurrence():
    inputs = [x.clone().requires_grad_() for x in _inputs(T=64, seed=4)]
    weight = torch.randn(2, 64, 2, 6, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(5))

    def grads(fn):
        o, s = fn(*inputs)
        return torch.autograd.grad((o * weight).sum() + s.square().sum(), inputs)

    for ref, got in zip(grads(triadic_reference_recurrence),
                        grads(triadic_chunk_kda), strict=True):
        torch.testing.assert_close(got, ref, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("decay_scale", [1.0, 0.0])
def test_fp32_chunk_form_is_stable_at_the_gate_bounds(decay_scale):
    """Channel decay pinned at the -5 bound (and at none) stays finite and
    matches an fp64 oracle: no factor in the decayed Gram overflows."""
    inputs = list(_inputs(T=128, Dk=16, seed=6, dtype=torch.float32))
    inputs[5] = torch.full_like(inputs[5], -5.0 * decay_scale)
    inputs[6] = torch.full_like(inputs[6], -40.0 * decay_scale)
    o, s = triadic_chunk_kda(*inputs)
    o_ref, s_ref = triadic_reference_recurrence(*(x.double() for x in inputs))
    assert o.isfinite().all() and s.isfinite().all()
    torch.testing.assert_close(o.double(), o_ref, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(s.double(), s_ref, rtol=1e-3, atol=1e-4)


def test_decayed_gram_matches_the_direct_sum():
    g = torch.Generator().manual_seed(7)
    left, right = (torch.randn(3, 64, 5, generator=g, dtype=torch.float64)
                   for _ in range(2))
    c = (-5.0 * torch.rand(3, 64, 5, generator=g, dtype=torch.float64)).cumsum(-2)
    direct = torch.einsum(
        "brj,bsj,brsj->brs", left, right,
        (c[:, :, None] - c[:, None, :]).clamp(max=0).exp(),
    ).tril()
    torch.testing.assert_close(decayed_gram(left, right, c), direct,
                               rtol=1e-9, atol=1e-12)
    small = torch.rand(3, 64, 2, generator=g, dtype=torch.float64)
    c2 = (-50.0 * torch.rand(3, 64, 2, generator=g, dtype=torch.float64)).cumsum(-2)
    direct2 = torch.einsum(
        "bre,bse,brse->brs", small, small,
        (c2[:, :, None] - c2[:, None, :]).clamp(max=0).exp(),
    ).tril()
    torch.testing.assert_close(slice_gram(small, small, c2), direct2,
                               rtol=1e-9, atol=1e-12)


def test_output_is_causal():
    inputs = _inputs(T=96, seed=8)
    o, _ = triadic_chunk_kda(*inputs)
    changed = [
        torch.cat((x[:, :50], y[:, 50:]), dim=1)
        for x, y in zip(inputs, _inputs(T=96, seed=99), strict=True)
    ]
    o2, _ = triadic_chunk_kda(*changed)
    torch.testing.assert_close(o2[:, :50], o[:, :50], rtol=0, atol=0)
    assert not torch.allclose(o2[:, 50:], o[:, 50:])


def _mixer(slices=3, seed=0):
    from pretraining.nanogpt_mini.triadic_kda import TriadicKimiDeltaAttention

    torch.manual_seed(seed)
    mixer = TriadicKimiDeltaAttention(dim=32, head_dim=8, num_heads=2,
                                      slices=slices)
    for name, parameter in mixer.named_parameters():
        if name.endswith("conv1d.weight"):
            torch.nn.init.normal_(parameter, std=0.5)
        elif name not in {"slice_A_log", "slice_dt_bias"}:
            torch.nn.init.normal_(parameter, std=0.3)
    return mixer


def test_mixer_decode_steps_match_the_full_sequence():
    mixer = _mixer()
    x = torch.randn(2, 40, 32)
    full, _ = mixer(x)
    prefix, caches = mixer(x[:, :25], output_final_state=True)
    torch.testing.assert_close(prefix, full[:, :25], rtol=1e-4, atol=1e-5)
    for t in range(25, 40):
        step = mixer.step(x[:, t:t + 1], caches)
        torch.testing.assert_close(step[:, 0], full[:, t], rtol=1e-3, atol=1e-4)


def test_mixer_keeps_state_zero_through_left_padding():
    mixer = _mixer(seed=1)
    x = torch.randn(1, 30, 32)
    valid = torch.ones(1, 30, dtype=torch.bool)
    valid[:, :12] = False
    padded, caches = mixer(x, key_valid=valid, output_final_state=True)
    bare, bare_caches = mixer(x[:, 12:], output_final_state=True)
    torch.testing.assert_close(padded[:, 12:], bare, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(caches[-1], bare_caches[-1], rtol=1e-4, atol=1e-5)


def test_slices_start_with_distinct_long_half_lives():
    mixer = _mixer(slices=4)
    with torch.no_grad():
        mixer.slice_gate_proj.weight.zero_()
        _, slice_log_decay, _ = mixer._controls(torch.zeros(1, 32))
    half_life = (torch.log(torch.tensor(0.5)) / slice_log_decay)[0, 0]
    torch.testing.assert_close(
        half_life, torch.tensor([32.0, 203.19, 1290.16, 8192.0]), rtol=1e-3, atol=0
    )


def test_every_mixer_parameter_receives_gradient():
    mixer = _mixer(seed=2)
    out, _ = mixer(torch.randn(2, 70, 32))
    out.square().sum().backward()
    for name, parameter in mixer.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name

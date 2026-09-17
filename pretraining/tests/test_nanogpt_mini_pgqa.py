"""CPU tests for ``pretraining/nanogpt_mini/nanogpt_mini_pgqa_model.py``.

Pins the address contract (region bits keep the pointer's phase, free bits
trade slots only inside ``2**k`` groups, every context position is
reachable, no pointer ever reads the future), the gathered GQA softmax
against a dense scatter reference (coincident keys add mass, query rows
with no valid pointer are zero), the prior as the sole controller gradient
path, and that activation checkpointing does not change the gradients.
"""

import copy
import itertools

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_pgqa_model import (
    PGQAGPT,
    PointerGQAttention,
    gqa_attend,
    num_region_bits,
    pointer_distances,
    pointer_log_prior,
    pointer_positions,
)


def test_num_region_bits():
    assert num_region_bits(1024, 256) == 2
    assert num_region_bits(256, 256) == 0
    with pytest.raises(ValueError):
        num_region_bits(1024, 96)
    with pytest.raises(ValueError):
        num_region_bits(768, 256)


def all_bit_patterns(window, num_bits):
    codes = torch.arange(2 ** num_bits)
    bits = (codes.unsqueeze(-1) >> torch.arange(num_bits)) & 1
    return bits.bool().view(2 ** num_bits, 1, num_bits).expand(-1, window, -1)  # [codes, P, L]


@pytest.mark.parametrize("free_bits", [0, 1, 3])
def test_distances_keep_phase_and_cover_context(free_bits):
    window, seq_len = 8, 32
    num_bits = num_region_bits(seq_len, window) + free_bits
    dist = pointer_distances(all_bit_patterns(window, num_bits), window, free_bits)  # [codes, P]
    slot = torch.arange(window)
    group = 1 << free_bits
    # phase of every reachable distance stays inside the pointer's free group
    assert torch.equal((dist % window) >> free_bits, (slot >> free_bits).expand_as(dist))
    # region-only pointers never coincide: the P distances of one draw are distinct
    if free_bits == 0:
        for code in range(dist.size(0)):
            assert dist[code].unique().numel() == window
    # each pointer reaches exactly (seq_len / window) * 2**k distances, all < seq_len
    for p in range(window):
        reach = dist[:, p].unique()
        assert reach.numel() == (seq_len // window) * group
        assert int(reach.max()) < seq_len and int(reach.min()) >= 0
    # union over pointers is the whole context [0, seq_len)
    assert torch.equal(dist.unique(), torch.arange(seq_len))


def test_positions_are_causal_and_clamped():
    torch.manual_seed(0)
    B, T, window, free_bits = 2, 32, 8, 1
    num_bits = num_region_bits(T, window) + free_bits
    bits = torch.rand(B, T, window, num_bits) < 0.5
    positions, valid = pointer_positions(bits, T, window, free_bits)
    assert positions.shape == valid.shape == (B, T, window)
    queries = torch.arange(T).view(1, T, 1)
    assert bool((positions <= queries).all()) and bool((positions >= 0).all())
    expected_valid = queries - pointer_distances(bits, window, free_bits) >= 0
    assert torch.equal(valid, expected_valid)
    # slot 0 with all-zero bits is the query itself
    zero = torch.zeros(1, T, window, num_bits, dtype=torch.bool)
    pos0, valid0 = pointer_positions(zero, T, window, free_bits)
    assert torch.equal(pos0[0, :, 0], torch.arange(T)) and bool(valid0[0, :, 0].all())


def test_log_prior_is_sum_of_bit_log_probs():
    torch.manual_seed(1)
    logits = torch.randn(3, 4, 5, dtype=torch.float64)
    bits = torch.rand(3, 4, 5) < 0.5
    lp = pointer_log_prior(logits, bits)
    probs = torch.sigmoid(logits)
    expected = torch.where(bits, probs, 1 - probs).log().sum(-1)
    assert torch.allclose(lp, expected, atol=1e-12)


def test_gqa_attend_matches_dense_scatter_reference():
    torch.manual_seed(2)
    B, T, H, D, P = 2, 6, 3, 4, 5
    scale = 0.3
    q = torch.randn(B, T, H, D, dtype=torch.float64)
    k = torch.randn(B, T, D, dtype=torch.float64)
    v = torch.randn(B, T, D, dtype=torch.float64)
    positions = torch.randint(0, T, (B, T, P))
    valid = torch.rand(B, T, P) < 0.7
    positions[0, 3, 1] = positions[0, 3, 2]  # coincident keys: masses add
    valid[0, 3, 1] = valid[0, 3, 2] = True
    valid[1, 4] = False  # query with no valid pointer at all
    log_prior = torch.randn(B, T, P, dtype=torch.float64)
    out = gqa_attend(q, k, v, positions, valid, log_prior, scale)
    assert out.shape == (B, T, H, D) and torch.isfinite(out).all()
    for b, t, h in itertools.product(range(B), range(T), range(H)):
        # dense reference: per-key mass = sum over pointers landing on it
        mass = torch.zeros(T, dtype=torch.float64)
        for p in range(P):
            if valid[b, t, p]:
                pos = int(positions[b, t, p])
                mass[pos] += torch.exp(scale * (q[b, t, h] @ k[b, pos]) + log_prior[b, t, p])
        if mass.sum() == 0:
            assert torch.equal(out[b, t, h], torch.zeros(D, dtype=out.dtype))
            continue
        expected = (mass / mass.sum()) @ v[b]
        assert torch.allclose(out[b, t, h], expected, atol=1e-12)


def make_attention(seq_len=32, window=8, free_bits=1, dim=16, head_dim=8, rank=4):
    torch.manual_seed(3)
    attn = PointerGQAttention(dim, seq_len, window, free_bits, rank, head_dim=head_dim, backend="gather")
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(std=0.3)
    return attn


def test_forward_is_causal_and_seed_deterministic():
    attn = make_attention().eval()  # eval samples too: the sample stream is the only nondeterminism
    x = torch.randn(2, 32, 16)
    torch.manual_seed(3)
    y = attn(x)
    torch.manual_seed(3)
    assert torch.equal(y, attn(x))
    x_future = x.clone()
    x_future[:, 20:] = torch.randn_like(x_future[:, 20:])
    torch.manual_seed(3)
    y_future = attn(x_future)
    assert torch.equal(y[:, :20], y_future[:, :20])
    assert not torch.equal(y[:, 20:], y_future[:, 20:])
    with pytest.raises(ValueError):
        attn(x[:, :16])


def test_controller_grad_flows_only_through_prior():
    attn = make_attention().train()
    x = torch.randn(2, 32, 16)
    torch.manual_seed(10)
    y_a = attn(x)
    torch.manual_seed(11)
    assert not torch.equal(y_a, attn(x))  # different draws -> different pointer sets
    y_a.square().sum().backward()
    ctrl = attn.controller
    for p in (ctrl.ctrl.weight, ctrl.ptr_embed, ctrl.ptr_mix, ctrl.ptr_bias):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0

    # Prior term is the sole gradient path and it is relative to the other
    # pointers: a lone pointer (window 1) sees at most one valid key, so its
    # softmax is shift-invariant in the prior and the bit logits get exactly
    # zero gradient (float64 so roundoff cannot masquerade as a path).
    solo = make_attention(seq_len=4, window=1, free_bits=0).double().train()
    assert solo.num_bits == 2
    x = torch.randn(1, 4, 16, dtype=torch.float64)
    logits = solo.controller(x)
    logits.retain_grad()
    torch.manual_seed(12)
    bits = solo.sample_bits(logits)
    positions, valid = pointer_positions(bits, 4, 1, 0)
    q, k, v = solo.qkv(x)
    out = gqa_attend(q, k, v, positions, valid, pointer_log_prior(logits, bits), 0.12)
    out.square().sum().backward()
    assert logits.grad is not None and logits.grad.abs().max() == 0


def test_checkpointing_does_not_change_gradients():
    ref = make_attention().double().train()
    ckpt = copy.deepcopy(ref)
    ref.checkpoint_attend = False
    ckpt.checkpoint_attend = True
    x = torch.randn(2, 32, 16, dtype=torch.float64)
    torch.manual_seed(20)
    y_ref = ref(x)
    torch.manual_seed(20)
    y_ckpt = ckpt(x)
    assert torch.equal(y_ref, y_ckpt)
    y_ref.square().sum().backward()
    y_ckpt.square().sum().backward()
    for (name, a), (_, b) in zip(ref.named_parameters(), ckpt.named_parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-12), name


def test_pgqa_gpt_loss_is_finite():
    torch.manual_seed(4)
    model = PGQAGPT(vocab_size=64, num_layers=2, model_dim=16, seq_len=32, window=8,
                    free_bits=1, rank=4, mlp_hidden=32, head_dim=8, backend="gather").float()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.3)
    inputs = torch.randint(0, 64, (2, 32))
    targets = torch.randint(0, 64, (2, 32))
    loss = model(inputs, targets)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())

"""CPU tests for ``pretraining/nanogpt_mini/nanogpt_mini_ptr_model.py``.

Pins the contracts a wrong index or prior computation would break: the
candidate set is exactly the causal Hamming-1 ball of each sampled block
offset, the additive prior factorizes over independent bits, the gathered
softmax equals an explicit per-query list reference (duplicates and
no-valid-key rows included), and the bit logits get gradient only through
that prior.
"""

import math

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_ptr_model import (
    PointerAttention,
    PtrGPT,
    num_pointer_bits,
    pointer_candidates,
    pointer_log_prior,
    pointer_offsets,
    pointer_tables,
    sparse_attend,
)


def test_num_pointer_bits_covers_every_block_offset():
    assert num_pointer_bits(1024, 8) == 7
    assert num_pointer_bits(1024, 1) == 10
    assert num_pointer_bits(16, 16) == 1
    with pytest.raises(ValueError):
        num_pointer_bits(1024, 12)
    with pytest.raises(ValueError):
        num_pointer_bits(96, 8)


def test_candidates_are_causal_hamming_one_ball():
    torch.manual_seed(0)
    T, Bk, K, L = 32, 4, 2, 3
    bits = torch.rand(1, T, 1, K, L) < 0.5
    positions, valid = pointer_candidates(bits, T, Bk)
    assert positions.shape == (1, T, 1, K, L + 1, Bk)
    for t in range(T):
        for j in range(K):
            address = sum(int(bits[0, t, 0, j, b]) << b for b in range(L))
            offsets = [address] + [address ^ (1 << b) for b in range(L)]
            for c, offset in enumerate(offsets):
                for w in range(Bk):
                    pos = t - offset * Bk - w
                    assert bool(valid[0, t, 0, j, c, w]) == (pos >= 0)
                    assert int(positions[0, t, 0, j, c, w]) == max(pos, 0)
    assert bool(valid.all(-1).any())  # offset 0 windows exist somewhere
    assert bool((positions <= torch.arange(T).view(1, T, 1, 1, 1, 1)).all())


def test_log_prior_factorizes_over_independent_bits():
    torch.manual_seed(1)
    L = 5
    logits = torch.randn(3, L, dtype=torch.float64)
    bits = torch.rand(3, L) < 0.5
    log_prior = pointer_log_prior(logits, bits)
    p = torch.sigmoid(logits)
    for row in range(3):
        for c in range(L + 1):
            cand = bits[row].clone()
            if c > 0:
                cand[c - 1] = ~cand[c - 1]
            expected = sum(
                math.log(p[row, b]) if cand[b] else math.log(1 - p[row, b])
                for b in range(L)
            )
            assert float(log_prior[row, c]) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("local_blocks", [0, 2])
def test_pointer_tables_logsumexp_duplicates_local_mass_and_gradient(local_blocks):
    torch.manual_seed(5)
    B, T, H, K, L = 1, 4, 1, 3, 2  # 4 blocks, 3 pointers x 3 candidates -> forced duplicates
    bits = torch.rand(B, T, H, K, L) < 0.5
    logits = torch.randn(B, T, H, K, L, dtype=torch.float64, requires_grad=True)
    log_prior = pointer_log_prior(logits, bits)
    offsets = pointer_offsets(bits)
    member, prior = pointer_tables(offsets, log_prior, num_blocks=1 << L, local_blocks=local_blocks)
    assert member.shape == prior.shape == (B, H, T, 1 << L)

    def hits_of(t, o, lp):
        return [lp[0, t, 0, j, c] for j in range(K) for c in range(L + 1)
                if int(offsets[0, t, 0, j, c]) == o]

    for t in range(T):
        for o in range(1 << L):
            hits = hits_of(t, o, log_prior)
            assert bool(member[0, 0, t, o]) == (bool(hits) or o < local_blocks)
            if member[0, 0, t, o]:
                expected = math.log(sum(math.exp(float(h)) for h in hits) + (o < local_blocks))
                assert float(prior[0, 0, t, o]) == pytest.approx(expected, rel=1e-12)
    # gradient of the table w.r.t. logits matches the direct log-sum
    (prior * member).sum().backward()
    grad_table = logits.grad.clone()
    logits.grad = None
    direct = torch.zeros((), dtype=torch.float64)
    lp = pointer_log_prior(logits, bits)
    for t in range(T):
        for o in range(1 << L):
            hits = hits_of(t, o, lp)
            if hits:
                direct = direct + torch.log(torch.stack(hits).exp().sum() + (o < local_blocks))
    direct.backward()
    assert torch.allclose(logits.grad, grad_table, atol=1e-12)


def test_sparse_attend_matches_explicit_list_softmax():
    torch.manual_seed(2)
    B, T, H, D, C = 2, 6, 2, 4, 5
    scale = 0.3
    q, k, v = (torch.randn(B, T, H, D, dtype=torch.float64) for _ in range(3))
    positions = torch.randint(0, T, (B, T, H, C))
    valid = torch.rand(B, T, H, C) < 0.7
    positions[0, 3, 0, 1] = positions[0, 3, 0, 2]  # duplicate key entry
    valid[0, 3, 0, 1] = valid[0, 3, 0, 2] = True
    valid[1, 4, 1] = False  # query with no valid key at all
    log_prior = torch.randn(B, T, H, C, dtype=torch.float64)
    out = sparse_attend(q, k, v, positions, valid, log_prior, scale)
    assert out.shape == (B, T, H, D)
    assert torch.isfinite(out).all()
    for b in range(B):
        for t in range(T):
            for h in range(H):
                entries = [
                    (float(q[b, t, h] @ k[b, int(positions[b, t, h, c]), h]) * scale
                     + float(log_prior[b, t, h, c]), int(positions[b, t, h, c]))
                    for c in range(C) if bool(valid[b, t, h, c])
                ]
                if not entries:
                    assert torch.equal(out[b, t, h], torch.zeros(D, dtype=out.dtype))
                    continue
                m = max(s for s, _ in entries)
                z = sum(math.exp(s - m) for s, _ in entries)
                expected = sum(math.exp(s - m) / z * v[b, p, h] for s, p in entries)
                assert torch.allclose(out[b, t, h], expected, atol=1e-12)


def make_attention(seq_len=32, block=4, pointers=2, dim=16, head_dim=8):
    torch.manual_seed(3)
    attn = PointerAttention(dim, seq_len, pointers, block, head_dim=head_dim, backend="gather")
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(std=0.3)
    return attn


def test_forward_is_causal_and_seed_deterministic():
    attn = make_attention().eval()  # eval samples bits too: the sample stream is the only nondeterminism
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


def test_train_forward_samples_and_grads_reach_bit_logits_only_via_prior():
    attn = make_attention().train()
    x = torch.randn(2, 32, 16)
    torch.manual_seed(10)
    y_a = attn(x)
    torch.manual_seed(11)
    y_b = attn(x)
    assert not torch.equal(y_a, y_b)  # different bit samples -> different candidate sets
    torch.manual_seed(10)
    assert torch.equal(attn(x), y_a)  # sample stream is the only nondeterminism

    y_a.square().sum().backward()
    assert attn.ptr.weight.grad is not None and attn.ptr.weight.grad.abs().sum() > 0
    assert torch.isfinite(attn.ptr.weight.grad).all()

    # Prior term is the sole gradient path: with a single one-bit pointer
    # whose flip window is always invalid, every query sees one block with a
    # shared prior, so the softmax is shift-invariant and d(out)/d(logit)
    # vanishes (float64 so roundoff cannot masquerade as a gradient path).
    solo = make_attention(seq_len=8, block=8, pointers=1).double().train()
    x = torch.randn(1, 8, 16, dtype=torch.float64)
    torch.manual_seed(12)
    solo(x).square().sum().backward()
    assert solo.ptr.weight.grad.abs().max() < 1e-12


def test_ptr_gpt_loss_is_finite_and_rejects_wrong_seq_len():
    torch.manual_seed(4)
    model = PtrGPT(vocab_size=64, num_layers=2, model_dim=16, seq_len=32,
                   num_pointers=2, block_size=4, mlp_hidden=32, backend="gather",
                   head_dim=8).float()
    inputs = torch.randint(0, 64, (2, 32))
    targets = torch.randint(0, 64, (2, 32))
    loss = model(inputs, targets)
    assert torch.isfinite(loss) and loss > 0
    with pytest.raises(ValueError):
        model(inputs[:, :16], targets[:, :16])

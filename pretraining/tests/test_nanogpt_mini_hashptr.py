"""CPU tests for ``pretraining/nanogpt_mini/nanogpt_mini_hashptr_model.py``.

Pins: the candidate set is exactly the causal keys whose per-round bucket
lies in the query's Hamming-1 ball (or the local window); the dense
reference equals an explicit per-query softmax with the round-summed prior;
buckets are position-free (permuting a prefix does not change any key's
bucket); the hyperplanes get gradient only through the prior term.
"""

import math

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_hashptr_model import (
    HashPointerAttention,
    HashPtrGPT,
    candidate_mask,
    dense_attend,
)


def make_attention(seq_len=32, rounds=2, bits=3, dim=16, head_dim=8, local_window=0, dtype=torch.float32):
    torch.manual_seed(3)
    attn = HashPointerAttention(dim, seq_len, rounds, bits, head_dim=head_dim,
                                backend="dense", local_window=local_window)
    with torch.no_grad():
        for name, p in attn.named_parameters():
            p.normal_(std=head_dim ** -0.5 if name == "hash" else 0.3)
    return attn.to(dtype)


def explicit_bits(attn, x):
    """Query bit logits and key buckets recomputed from first principles."""
    B, T, H, D = x.size(0), x.size(1), attn.num_heads, attn.head_dim
    q = torch.nn.functional.rms_norm(attn.q(x).view(B, T, H, D), (D,))
    k = torch.nn.functional.rms_norm(attn.k(x).view(B, T, H, D), (D,))
    logits = torch.einsum("bthd,hjld->bthjl", q, attn.hash) / attn.temperature
    key_bits = torch.einsum("bthd,hjld->bhjtl", k, attn.hash) > 0
    return logits, key_bits


@pytest.mark.parametrize("local_window", [0, 5])
def test_candidates_are_causal_ball_members_union_over_rounds(local_window):
    attn = make_attention(local_window=local_window).eval()
    x = torch.randn(1, 32, 16)
    _, _, _, member, prior, key_slots = attn.prepare(x)
    logits, key_bits = explicit_bits(attn, x)
    allowed = candidate_mask(member, key_slots, local_window)
    L, K = attn.num_bits, attn.num_rounds
    for t in range(32):
        for h in range(attn.num_heads):
            for s in range(32):
                expected = s <= t and (t - s < local_window)
                for j in range(K):
                    qb = [bool(logits[0, t, h, j, b] > 0) for b in range(L)]  # eval: mode bits
                    kb = [bool(key_bits[0, h, j, s, b]) for b in range(L)]
                    hamming = sum(a != c for a, c in zip(qb, kb))
                    expected = expected or (s <= t and hamming <= 1)
                assert bool(allowed[0, h, t, s]) == expected, (t, h, s)


def test_dense_attend_matches_explicit_round_summed_prior():
    torch.manual_seed(4)
    attn = make_attention(dtype=torch.float64).eval()
    x = torch.randn(1, 32, 16, dtype=torch.float64)
    q, k, v, member, prior, key_slots = attn.prepare(x)
    logits, key_bits = explicit_bits(attn, x)
    out = dense_attend(q, k, v, member, prior, key_slots, 0, 0.12)
    p = torch.sigmoid(logits)
    L, K = attn.num_bits, attn.num_rounds
    for t in range(32):
        for h in range(attn.num_heads):
            entries = []
            for s in range(t + 1):
                mass = 0.0
                for j in range(K):
                    qb = [bool(logits[0, t, h, j, b] > 0) for b in range(L)]
                    kb = [bool(key_bits[0, h, j, s, b]) for b in range(L)]
                    if sum(a != c for a, c in zip(qb, kb)) <= 1:
                        # pi_j(bucket of s) under the query's independent bits
                        mass += math.prod(
                            float(p[0, t, h, j, b]) if kb[b] else 1 - float(p[0, t, h, j, b])
                            for b in range(L)
                        )
                if mass > 0:
                    entries.append((float(q[0, t, h] @ k[0, s, h]) * 0.12 + math.log(mass), s))
            if not entries:
                assert torch.equal(out[0, t, h], torch.zeros_like(out[0, t, h]))
                continue
            m = max(e for e, _ in entries)
            z = sum(math.exp(e - m) for e, _ in entries)
            expected = sum(math.exp(e - m) / z * v[0, s, h] for e, s in entries)
            assert torch.allclose(out[0, t, h], expected, atol=1e-10), (t, h)


def test_buckets_are_position_free():
    attn = make_attention().eval()
    x = torch.randn(1, 32, 16)
    _, _, _, _, _, slots = attn.prepare(x)
    perm = torch.randperm(32)
    _, _, _, _, _, slots_perm = attn.prepare(x[:, perm])
    assert torch.equal(slots_perm, slots[..., perm])


def test_train_samples_and_hash_grad_flows_only_through_prior():
    attn = make_attention().train()
    x = torch.randn(2, 32, 16)
    torch.manual_seed(10)
    y_a = attn(x)
    torch.manual_seed(11)
    assert not torch.equal(attn(x), y_a)
    torch.manual_seed(10)
    assert torch.equal(attn(x), y_a)
    y_a.square().sum().backward()
    assert attn.hash.grad is not None and torch.isfinite(attn.hash.grad).all()
    assert attn.hash.grad.abs().sum() > 0

    # Prior term is the sole gradient path to the hyperplanes: the key side
    # is hard (sign), so detaching the prior must leave the hash with no grad.
    solo = make_attention(rounds=1, bits=1, dtype=torch.float64).train()
    x = torch.randn(1, 32, 16, dtype=torch.float64)
    torch.manual_seed(12)
    q, k, v, member, prior, key_slots = solo.prepare(x)
    dense_attend(q, k, v, member, prior.detach(), key_slots, 0, 0.12).square().sum().backward()
    assert solo.hash.grad is None


def test_hash_gpt_loss_finite_and_rejects_wrong_seq_len():
    torch.manual_seed(5)
    model = HashPtrGPT(64, 2, 16, seq_len=32, num_rounds=2, num_bits=3, mlp_hidden=32,
                       backend="dense", head_dim=8).float()
    with torch.no_grad():
        model.blocks[0].attn.hash.normal_(std=0.35)
        model.blocks[1].attn.hash.normal_(std=0.35)
    inputs = torch.randint(0, 64, (2, 32))
    targets = torch.randint(0, 64, (2, 32))
    loss = model(inputs, targets)
    assert torch.isfinite(loss) and loss > 0
    with pytest.raises(ValueError):
        model(inputs[:, :16], targets[:, :16])

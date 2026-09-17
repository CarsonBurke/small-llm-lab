"""CPU tests for ``pretraining/nanogpt_mini/nanogpt_mini_iptr_model.py``.

Pins the interval-decoding invariants: every selected block contains real
accessible tokens (offset < M_t), every accessible block is reachable, one
available block needs zero decisions, non-power-of-two block counts and
arbitrary T work, single-bit flips give distinct leaves and unused-bit flips
are dropped; the prior sums over each candidate's own consumed depths; the
gather backend equals the dense-analogue softmax; the controller is
length-aware and gets gradient only through the prior.
"""

import itertools
import math

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_iptr_model import (
    DEPTH_FEATURE,
    SCALE_FEATURE,
    IPtrGPT,
    routing_features,
    IntervalPointerAttention,
    accessible_blocks,
    decode_intervals,
    interval_log_prior,
    num_depths,
    pooled_scores,
    sequential_route,
)


def leaf(bits, count):
    lo, hi, consumed = 0, count, []
    for d, b in enumerate(bits):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if b else (lo, mid)
        consumed.append(d)
    return lo, consumed


@pytest.mark.parametrize("seq_len,block", [(1024, 8), (100, 8), (37, 5), (7, 8), (3, 1)])
def test_decode_reaches_exactly_the_accessible_blocks(seq_len, block):
    D = num_depths(seq_len, block)
    assert D == (math.ceil(math.log2(math.ceil(seq_len / block))) if seq_len > block else 0)
    blocks = accessible_blocks(seq_len, block, torch.device("cpu"))
    assert torch.equal(blocks, torch.tensor([math.ceil((t + 1) / block) for t in range(seq_len)]))
    if D > 10:
        pytest.skip("exhaustive enumeration too large")
    all_bits = torch.tensor(list(itertools.product([False, True], repeat=D))) if D else torch.zeros(1, 0, dtype=torch.bool)
    for t in (0, 1, seq_len // 2, seq_len - 1):
        M = int(blocks[t])
        bits = all_bits.view(1, 1, 1, all_bits.size(0), D)
        offsets, used, valid = decode_intervals(bits, blocks[t:t + 1])
        sampled = offsets[0, 0, 0, :, 0]
        # every leaf is a real block, every real block is some leaf
        assert int(sampled.min()) >= 0 and int(sampled.max()) < M
        assert set(sampled.tolist()) == set(range(M))
        for row, b in enumerate(all_bits.tolist()):
            o, consumed = leaf(b, M)
            assert int(sampled[row]) == o
            assert used[0, 0, 0, row, 0].nonzero().flatten().tolist() == consumed
            flips = offsets[0, 0, 0, row, 1:]
            for d in range(D):
                if d in consumed:
                    assert bool(valid[0, 0, 0, row, 1 + d])
                    assert int(flips[d]) != o  # a consumed flip changes the leaf
                else:
                    assert not bool(valid[0, 0, 0, row, 1 + d])
            live = [int(flips[d]) for d in consumed]
            assert len(live) == len(set(live))  # distinct leaves


def test_interval_log_prior_sums_only_consumed_depths():
    torch.manual_seed(0)
    D = 4
    logits = torch.randn(3, D, dtype=torch.float64)
    bits = torch.rand(3, D) < 0.5
    blocks = torch.tensor([11])  # non power of two
    offsets, used, valid = decode_intervals(bits.view(1, 1, 1, 3, D), blocks)
    lp = interval_log_prior(logits.view(1, 1, 1, 3, D), bits.view(1, 1, 1, 3, D), used)[0, 0, 0]
    p = torch.sigmoid(logits)
    for row in range(3):
        for c in range(D + 1):
            cand = bits[row].clone()
            if c:
                cand[c - 1] = ~cand[c - 1]
            _, consumed = leaf(cand.tolist(), 11)
            expected = sum(math.log(p[row, d]) if cand[d] else math.log(1 - p[row, d]) for d in consumed)
            assert float(lp[row, c]) == pytest.approx(expected, rel=1e-12)


def test_routing_features_describe_each_decision():
    # T=37, Bk=4 -> M_t in 1..10, D=4
    f = routing_features(37, 4, 4, torch.device("cpu"), torch.float64)
    assert f.shape == (37, 4, 4)
    assert torch.allclose(f[:, :, 0], torch.log2(torch.arange(1.0, 38.0, dtype=torch.float64)).view(-1, 1).expand(-1, 4))
    assert torch.equal(f[:, :, DEPTH_FEATURE], torch.arange(4.0, dtype=torch.float64).expand(37, -1))
    # t=36: M=10 blocks -> bit d halves ~10/2^d blocks
    assert torch.allclose(f[36, :, 2], torch.log2(torch.tensor([10.0, 5.0, 2.5, 1.25], dtype=torch.float64)))
    # t=3: one block -> no jump scale at any depth
    assert torch.equal(f[3, :, 2], torch.zeros(4, dtype=torch.float64))
    # the parallel decoder never knows the interval start
    assert torch.equal(f[:, :, 3], torch.zeros(37, 4, dtype=torch.float64))


def test_pooled_scores_match_explicit_interval_means():
    # s[t, o] = q_t . sum_{i <= t - o Bk} k_i, zero past the start; the
    # routine's interval mean (s[a] - s[b]) / count equals the explicit
    # mean over the keys in query-relative block offsets [a, b).
    torch.manual_seed(0)
    B, T, H, Dh, Bk = 2, 40, 2, 8, 4
    q = torch.nn.functional.rms_norm(torch.randn(B, T, H, Dh, dtype=torch.float64), (Dh,))
    k = torch.nn.functional.rms_norm(torch.randn(B, T, H, Dh, dtype=torch.float64), (Dh,))
    s = pooled_scores(q, k, Bk)
    assert s.shape == (B, T, H, T // Bk)
    for t, o in itertools.product(range(T), range(T // Bk)):
        j = t - o * Bk
        ref = (q[:, t] * k[:, :j + 1].sum(1)).sum(-1) if j >= 0 else torch.zeros(B, H, dtype=torch.float64)
        assert torch.allclose(s[:, t, :, o], ref, atol=1e-10)
    t, a, b = 37, 2, 6  # offsets [2, 6): tokens (t - 24, t - 8] = [14, 29]
    count = min(b * Bk, t + 1) - a * Bk
    got = (s[:, t, :, a] - s[:, t, :, b]) / count
    ref = (q[:, t] * k[:, t - b * Bk + 1:t - a * Bk + 1].mean(1)).sum(-1)
    assert torch.allclose(got, ref, atol=1e-10)
    t, a, b = 9, 1, 5  # oldest block partial: tokens [0, 5], count 6
    count = min(b * Bk, t + 1) - a * Bk
    got = (s[:, t, :, a] - s[:, t, :, b]) / count
    ref = (q[:, t] * k[:, :t - a * Bk + 1].mean(1)).sum(-1)
    assert count == 6 and torch.allclose(got, ref, atol=1e-10)


def test_sequential_route_walks_the_interval_tree():
    # T=45, Bk=4 -> M_t in 1..12 (non-power-of-two), D=4
    torch.manual_seed(0)
    attn = IntervalPointerAttention(16, 2, 4, 3, head_dim=8, backend="gather", decode="sequential").double()
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(std=0.3)
    x = torch.randn(2, 45, 16, dtype=torch.float64)
    D = num_depths(45, 4)
    bits, logits = sequential_route(attn.head, x, D, 4)
    assert bits.shape == logits.shape == (2, 45, 2, 2, D)
    counts = accessible_blocks(45, 4, x.device)
    offsets, used, _ = decode_intervals(bits, counts)
    for b, t, h, k in itertools.product(range(2), range(45), range(2), range(2)):
        lo, consumed = leaf(bits[b, t, h, k].tolist(), int(counts[t]))
        # the sampled path is what the decoder reproduces as candidate 0
        assert int(offsets[b, t, h, k, 0]) == lo
        assert used[b, t, h, k, 0].tolist() == [d in consumed for d in range(D)]
    # depth d's logit is a function of the interval reached: two samples of
    # the same query that diverge at depth 0 get different depth-1 logits
    # (the head reads log2(lo + 1)), unlike the parallel decoder.
    t = 44  # 12 blocks: bit 0 splits [0, 6) / [6, 12)
    u = attn.head.hidden_input(x)
    feats = lambda lo, hi: torch.stack((
        torch.full((), math.log2(t + 1), dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(math.log2(hi - lo), dtype=torch.float64), torch.tensor(math.log2(lo + 1), dtype=torch.float64),
    )).view(1, 1, 1, 1, 4).expand(1, 1, 2, 2, 4)
    near = attn.head.logits(u[:1, t:t + 1], feats(0, 6))
    far = attn.head.logits(u[:1, t:t + 1], feats(6, 12))
    assert not torch.allclose(near, far)


def test_straight_prior_attends_by_content_only():
    torch.manual_seed(0)
    kwargs = dict(head_dim=8, backend="gather")
    straight = IntervalPointerAttention(16, 2, 4, 3, prior="straight", **kwargs).double()
    score = IntervalPointerAttention(16, 2, 4, 3, prior="score", **kwargs).double()
    with torch.no_grad():
        for p in straight.parameters():
            p.normal_(std=0.3)
    score.load_state_dict(straight.state_dict())
    x = torch.randn(2, 30, 16, dtype=torch.float64)
    torch.manual_seed(1)
    _, lp_straight, _ = straight.route(x)
    torch.manual_seed(1)
    _, lp_score, _ = score.route(x)
    assert torch.equal(lp_straight, torch.zeros_like(lp_straight))
    assert not torch.equal(lp_score, torch.zeros_like(lp_score))
    # the backward is the same finite difference: the prior path carries gradient
    lp_straight.sum().backward()
    assert straight.head.ctrl.weight.grad is not None and straight.head.ctrl.weight.grad.abs().sum() > 0


def make_attention(block=4, pointers=2, dim=16, head_dim=8, width=3, local_window=0, dtype=torch.float32):
    torch.manual_seed(3)
    attn = IntervalPointerAttention(dim, pointers, block, width, head_dim=head_dim,
                                    backend="gather", local_window=local_window)
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(std=0.3)
    return attn.to(dtype)


def test_gather_matches_dense_analogue_softmax():
    attn = make_attention(dtype=torch.float64).eval()
    with torch.no_grad():  # identity output projection exposes the raw attention output
        attn.proj.weight.copy_(torch.eye(16, dtype=torch.float64))
        attn.proj.bias.zero_()
    T = 37  # 10 blocks of 4 -> non power of two, partial oldest block
    x = torch.randn(1, T, 16, dtype=torch.float64)
    q, k, v, _, _ = attn.qkv(x)
    torch.manual_seed(1)
    offsets, log_prior, valid = attn.route(x)
    torch.manual_seed(1)  # eval samples bits too; same draw for the explicit recompute
    pre = attn(x)[0].view(T, attn.num_heads, attn.head_dim)
    for t in range(T):
        for h in range(attn.num_heads):
            mass = {}
            for j in range(attn.num_pointers):
                for c in range(offsets.size(-1)):
                    if not bool(valid[0, t, h, j, c]):
                        continue
                    o = int(offsets[0, t, h, j, c])
                    assert o * 4 <= t  # real block
                    for w in range(4):
                        s = t - o * 4 - w
                        if s >= 0:
                            mass[s] = mass.get(s, 0.0) + math.exp(float(log_prior[0, t, h, j, c]))
            entries = [(float(q[0, t, h] @ k[0, s, h]) * 0.12 + math.log(m), s) for s, m in mass.items()]
            top = max(e for e, _ in entries)
            z = sum(math.exp(e - top) for e, _ in entries)
            expected = sum(math.exp(e - top) / z * v[0, s, h] for e, s in entries)
            assert torch.allclose(pre[t, h], expected, atol=1e-8), (t, h)


def test_forward_handles_any_length_and_single_block():
    attn = make_attention().eval()
    for T in (1, 3, 4, 5, 37, 64):
        x = torch.randn(2, T, 16)
        y = attn(x)
        assert y.shape == (2, T, 16) and torch.isfinite(y).all()
    # T <= block: zero routing decisions, prior exactly 0, one candidate
    x = torch.randn(1, 4, 16)
    offsets, log_prior, valid = attn.route(x)
    assert offsets.size(-1) == 1 and int(offsets.max()) == 0
    assert torch.equal(log_prior, torch.zeros_like(log_prior)) and bool(valid.all())


def test_controller_is_length_aware_and_trained_only_through_prior():
    attn = make_attention().train()
    # routing features: same token content at two positions gives identical
    # logits until the feature maps are nonzero; then context length tells
    # them apart, and the depth feature alone reproduces the geometric prior
    x = torch.randn(1, 8, 16)
    x[:, 5] = x[:, 2]
    with torch.no_grad():
        attn.head.feat_in.zero_()
        attn.head.feat_out.zero_()
        same = attn.head(x, 2, 4)
        attn.head.feat_in.normal_()
        diff = attn.head(x, 2, 4)
        attn.head.feat_in.zero_()
        attn.head.out.zero_()
        attn.head.out_bias.zero_()
        attn.head.feat_out[..., SCALE_FEATURE] = -0.5
        attn.head.out_bias.fill_(0.5)
        prior = attn.head(x, 3, 4)  # t=7: M=2 blocks -> scale [1,0,0] -> logits [0, .5, .5]
    assert torch.allclose(same[0, 2], same[0, 5])
    assert not torch.allclose(diff[0, 2], diff[0, 5])
    assert torch.allclose(prior[0, 7], torch.tensor([0.0, 0.5, 0.5]).expand_as(prior[0, 7]))
    # a query with 8 blocks (t>=28 at Bk=4): jumps 4,2,1 blocks -> logits -1, -0.5, 0
    x8 = torch.randn(1, 32, 16)
    with torch.no_grad():
        prior8 = attn.head(x8, 3, 4)
    assert torch.allclose(prior8[0, 31], torch.tensor([-1.0, -0.5, 0.0]).expand_as(prior8[0, 31]))

    with torch.no_grad():
        attn.head.feat_in.normal_(std=0.3)
        attn.head.out.normal_(std=0.3)
    torch.manual_seed(10)
    y_a = attn(torch.randn(2, 32, 16))
    y_a.square().sum().backward()
    for name in ("ctrl.weight", "feat_in", "feat_out", "out", "out_bias"):
        g = dict(attn.head.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name

    # detaching the prior removes every controller gradient: the bits are
    # sampled, so the only path into the controller is log pi
    solo = make_attention().train()
    route = solo.route
    solo.route = lambda x, *qk: (lambda o, lp, v: (o, lp.detach(), v))(*route(x, *qk))
    solo(torch.randn(1, 32, 16)).square().sum().backward()
    assert all(p.grad is None for p in solo.head.parameters())
    assert solo.q.weight.grad is not None


def test_gpt_loss_finite_at_odd_length():
    torch.manual_seed(5)
    model = IPtrGPT(64, 2, 16, num_pointers=2, block_size=4, width=3, mlp_hidden=32,
                    backend="gather", head_dim=8).float()
    with torch.no_grad():
        for block in model.blocks:
            block.attn.head.out.normal_(std=0.3)
    inputs = torch.randint(0, 64, (2, 37))
    targets = torch.randint(0, 64, (2, 37))
    loss = model(inputs, targets)
    assert torch.isfinite(loss) and loss > 0
    assert torch.isfinite(model(inputs[:, :20], targets[:, :20].contiguous()))


def test_linear_diagnostic_head_starts_from_the_shared_prior():
    from pretraining.nanogpt_mini.nanogpt_mini_iptr_model import IntervalPointerAttention as A
    shared = A(16, 2, 4, 3, head_dim=8, backend="gather", head="shared")
    linear = A(16, 2, 4, 3, head_dim=8, backend="gather", head="linear", max_depths=5)
    with torch.no_grad():
        for attn in (shared, linear):
            for name, p in attn.head.named_parameters():
                p.zero_()
                if name == "feat_out":
                    p[..., SCALE_FEATURE] = -0.5
                elif name == "out_bias":
                    p.fill_(0.5)
    x = torch.randn(1, 32, 16)
    assert torch.allclose(shared.head(x, 3, 4), linear.head(x, 3, 4))
    with pytest.raises(ValueError):
        linear.head(x, 6, 4)


def test_padded_diagnostic_decode_is_absolute_and_drops_leaves_beyond_query():
    from pretraining.nanogpt_mini.nanogpt_mini_iptr_model import decode_padded
    D = 4
    all_bits = torch.tensor(list(itertools.product([False, True], repeat=D))).view(1, 1, 1, 16, D)
    blocks = torch.tensor([11])
    offsets, used, valid = decode_padded(all_bits, blocks)
    # bit pattern == absolute offset (MSB first), independent of M_t
    assert offsets[0, 0, 0, :, 0].tolist() == list(range(16))
    assert bool(used.all())
    # sampled and flip candidates beyond the 11 accessible blocks are dropped
    assert torch.equal(valid, offsets < 11)
    # flips are XOR neighbours
    for row in range(16):
        assert offsets[0, 0, 0, row, 1:].tolist() == [row ^ (1 << (D - 1 - d)) for d in range(D)]

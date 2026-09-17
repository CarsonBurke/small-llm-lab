"""CPU tests for ``pretraining/nanogpt_mini/nanogpt_mini_tree_model.py``.

Pins the invariants the estimator relies on: every row is the exact mean of
its interval; for each pointer the attended rows (sibling summaries + leaf)
partition ``[0, t]`` -- the whole causal context is seen exactly once, at
mixed resolution; the candidate priors of a pointer sum to one when every
depth is decided; the checkpointed descent replays the same rows; the
model runs end to end with finite gradients into routing and K/V.
"""

import itertools

import torch
from torch.utils.checkpoint import checkpoint

from pretraining.nanogpt_mini.nanogpt_mini_tree_model import (
    LEVEL_FEATURE,
    TreeGPT,
    TreePointerAttention,
    build_row_table,
    edge_row,
    node_row,
    num_rows,
    tree_depth,
)


def interval_of_row(row: int, t: int, seq_len: int) -> tuple[int, int]:
    """Half-open token interval a row summarizes (for query ``t``)."""
    D = tree_depth(seq_len)
    if row < seq_len:
        return row, row + 1
    if row < 2 * seq_len - 1:
        level = 1
        while row >= int(node_row(torch.tensor(0), level, seq_len)) + (seq_len >> level):
            level += 1
        lo = (row - int(node_row(torch.tensor(0), level, seq_len))) << level
        return lo, lo + (1 << level)
    e = row - (2 * seq_len - 1)
    tt, d = divmod(e, D + 1)
    assert tt == t, "edge rows belong to their own query"
    return (tt >> (D - d)) << (D - d), tt + 1


def make_attention(seq_len=32, pointers=3, dim=64, head_dim=16, width=5):
    torch.manual_seed(0)
    attn = TreePointerAttention(dim, seq_len, pointers, width, head_dim=head_dim, backend="gather").double()
    with torch.no_grad():
        for name, p in attn.named_parameters():
            if name.startswith("head.") and name != "head.ctrl.weight":
                p.normal_(std=0.5)
    return attn


def test_row_table_holds_exact_interval_means():
    torch.manual_seed(0)
    B, T, Dh = 2, 32, 8
    D = tree_depth(T)
    k = torch.randn(B, T, Dh, dtype=torch.float64)
    v = torch.randn(B, T, Dh, dtype=torch.float64)
    table, log_count = build_row_table(k, v)
    assert table.shape == (B, num_rows(T), 2 * Dh)
    for level in range(1, D + 1):
        size = 1 << level
        for lo in range(0, T, size):
            r = int(node_row(torch.tensor(lo), level, T))
            ref = torch.cat((k[:, lo:lo + size].mean(1), v[:, lo:lo + size].mean(1)), -1)
            assert torch.allclose(table[:, r], ref)
            assert log_count[r].exp().round().item() == size
    for t, d in itertools.product(range(T), range(D + 1)):
        start = (t >> (D - d)) << (D - d)
        r = int(edge_row(torch.tensor(t), d, T, D))
        ref = torch.cat((k[:, start:t + 1].mean(1), v[:, start:t + 1].mean(1)), -1)
        assert torch.allclose(table[:, r], ref)
        assert abs(log_count[r].exp().item() - (t - start + 1)) < 1e-4


def test_pointer_rows_partition_the_causal_context():
    T, K = 32, 3
    attn = make_attention(seq_len=T, pointers=K)
    D = attn.depths
    x = torch.randn(2, T, 64, dtype=torch.float64)
    _, k, v, qr = attn.qkv(x)
    table, _ = build_row_table(k, v)
    torch.manual_seed(3)
    rows, log_prior = attn.route(x, qr, table)
    rows = rows.view(2, T, K, D + 1)
    log_prior = log_prior.view(2, T, K, D + 1)
    for b, t, j in itertools.product(range(2), range(T), range(K)):
        ivs = sorted(interval_of_row(int(r), t, T) for r in rows[b, t, j] if r >= 0)
        assert [i for iv in ivs for i in range(*iv)] == list(range(t + 1))
    assert bool((rows[..., D] < T).all())  # leaves are tokens
    # every depth decided at t = T - 1: the candidates' priors sum to 1
    assert torch.allclose(log_prior[:, T - 1].exp().sum(-1), torch.ones(2, K, dtype=torch.float64))
    # t = 0: one accessible token, no decisions, no summaries
    assert bool((rows[:, 0, :, :D] == -1).all()) and bool((rows[:, 0, :, D] == 0).all())


def test_checkpointed_descent_replays_the_same_rows():
    attn = make_attention()
    x = torch.randn(2, 32, 64, dtype=torch.float64)
    _, k, v, qr = attn.qkv(x)
    table, _ = build_row_table(k, v)
    torch.manual_seed(9)
    rows, _ = attn.route(x, qr, table)
    torch.manual_seed(9)
    rows_ckpt, lp_ckpt = checkpoint(attn.route, x, qr, table, use_reentrant=False)
    assert torch.equal(rows, rows_ckpt)
    lp_ckpt.sum().backward()  # the recompute must reproduce rows for the gradient to be meaningful
    assert attn.qr.weight.grad is not None


def test_model_trains_end_to_end_on_cpu():
    torch.manual_seed(0)
    model = TreeGPT(64, 2, 64, 32, num_pointers=2, width=5, head_dim=16, backend="gather").double()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("head.feat_out"):
                p.zero_()
                p[..., LEVEL_FEATURE] = 0.5
            elif name.endswith("head.feat_in"):
                p.zero_()
            elif name.endswith("head.out"):
                p.normal_(std=0.5)
    inputs = torch.randint(0, 64, (2, 32))
    targets = torch.randint(0, 64, (2, 32))
    loss = model(inputs, targets)
    loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters()}
    assert torch.isfinite(loss)
    for name in ("blocks.0.attn.qr.weight", "blocks.0.attn.route_gains", "blocks.0.attn.head.ctrl.weight",
                 "blocks.0.attn.k.weight", "blocks.0.attn.v.weight"):
        assert grads[name] is not None and torch.isfinite(grads[name]).all() and grads[name].abs().sum() > 0, name

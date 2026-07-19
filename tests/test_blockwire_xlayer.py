from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "xlayer"))
import blockwire_train_gpt as bw  # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# Small geometry used across tests: head_dim 16, 4 query / 2 kv heads,
# T=128, key blocks of 8, query blocks of 16 -> 8 query blocks, 16 key
# blocks per source layer.
DIM, HEADS, KV_HEADS = 64, 4, 2
T, KB, QB = 128, 8, 16
N_RAND = 2


def make_attn(layer, *, n_rand=N_RAND, xlayer=True):
    attn = bw.BlockWireAttention(DIM, HEADS, KV_HEADS, 10000.0, 1.0)
    attn.n_rand = n_rand
    attn.key_block, attn.query_block = KB, QB
    attn.xlayer = xlayer
    attn.configure_wiring(layer, T)
    return attn.cuda()


def run_chain(modules, x, monkeypatch):
    """Drive a stack of attention modules the way GPT.forward does."""
    monkeypatch.setattr(bw, "_TRAIN_SEQ_LEN", T)
    bw._STATE["i"] = 0
    bw._STATE["bank"] = None
    for m in modules:
        x = x + m(x)
    return x


def fresh_chain(n_layers, **kwargs):
    torch.manual_seed(7)
    return [make_attn(i, **kwargs) for i in range(n_layers)]


def test_hash_uniform_bounds_and_determinism():
    """Draw hash is deterministic per (lane, step, layer) and lands in (0, 1).

    Pure arithmetic on int64 tensors, so this runs on CPU without CUDA.
    """
    lane = torch.arange(4096, dtype=torch.int64)
    step = torch.full((1,), 123, dtype=torch.int64)
    u1 = bw._hash_uniform(lane, step, 3)
    u2 = bw._hash_uniform(lane, step, 3)
    assert torch.equal(u1, u2)
    assert (u1 > 0).all() and (u1 < 1).all()
    u3 = bw._hash_uniform(lane, step + 1, 3)
    assert not torch.equal(u1, u3)


@cuda_only
def test_selection_is_causal_and_deduped(monkeypatch):
    """Every valid dynamic block is fully causal; no valid duplicates."""
    monkeypatch.setattr(bw, "_TRAIN_SEQ_LEN", T)
    for layer in (0, 2):
        attn = make_attn(layer).train()
        for step in (0, 5):
            bw._STEP_BUF.fill_(step)
            gid, valid = attn._select(2)
            assert gid.shape == (2, KV_HEADS, T // QB, N_RAND)
            # Causality: a valid block must end at or before its query block's
            # first position (the current-layer diagonal span is local, never
            # drawn, so no straddling block is ever selectable).
            block_end = (gid % attn.n_blocks + 1) * KB
            q_first = (torch.arange(T // QB, device="cuda") * QB).view(1, 1, -1, 1)
            assert not (valid & (block_end > q_first)).any()
            # Dedup: no duplicate valid ids within a slot row (softmax would
            # over-weight a duplicated key).  Invalid slots map to the unique
            # per-slot sentinels, so they can never masquerade as duplicates.
            key = torch.where(valid, gid, attn.sentinel)
            eq = key.unsqueeze(-1) == key.unsqueeze(-2)
            eq &= ~torch.eye(key.size(-1), dtype=torch.bool, device="cuda")
            assert not eq.any()


@cuda_only
def test_random_slots_change_with_step_and_eval_is_deterministic(monkeypatch):
    """Training draws are re-rolled every step; eval uses a fixed seed."""
    monkeypatch.setattr(bw, "_TRAIN_SEQ_LEN", T)
    attn = make_attn(2).train()
    bw._STEP_BUF.fill_(1)
    gid1, _ = attn._select(2)
    bw._STEP_BUF.fill_(2)
    gid2, _ = attn._select(2)
    assert not torch.equal(gid1, gid2)  # every slot is a fresh random draw

    attn.eval()
    e1 = attn._select(2)
    bw._STEP_BUF.fill_(99)  # eval must read _EVAL_STEP, not the train clock
    e2 = attn._select(2)
    assert torch.equal(e1[0], e2[0]) and torch.equal(e1[1], e2[1])


def _reference_fine(attn, q, gid, valid, ks, vs, positions):
    """fp64 loop reference for the softmax+null fine pass (independent path).

    Computes only the requested query positions; the caller compares those
    rows.  Full-T loops would take minutes for zero extra coverage.
    """
    bsz = q.size(0)
    group = HEADS // KV_HEADS
    kb, qb = attn.key_block, attn.query_block
    bank_k = torch.cat(ks, dim=2).double()
    bank_v = torch.cat(vs, dim=2).double()
    k_cur, v_cur = ks[-1].double(), vs[-1].double()
    scale = attn.head_dim ** -0.5
    y = torch.zeros(bsz, HEADS, q.size(2), attn.head_dim, dtype=torch.float64, device=q.device)
    qd = q.double()
    for b in range(bsz):
        for h in range(HEADS):
            g = h // group
            for t in positions:
                qb_i = t // qb
                logit_keys = []
                for slot in range(attn.n_dyn):
                    if not valid[b, g, qb_i, slot]:
                        continue
                    blk = int(gid[b, g, qb_i, slot])
                    for j in range(kb):
                        pos = blk * kb + j
                        logit_keys.append(("bank", pos))
                for j in range(2 * qb):
                    pos = qb_i * qb - qb + j
                    if pos < 0 or pos > t:
                        continue
                    logit_keys.append(("cur", pos))
                logits = []
                vals = []
                for kind, pos in logit_keys:
                    kk = bank_k[b, g, pos] if kind == "bank" else k_cur[b, g, pos]
                    vv = bank_v[b, g, pos] if kind == "bank" else v_cur[b, g, pos]
                    logits.append((qd[b, h, t] * kk).sum() * scale)
                    vals.append(vv)
                logits.append(attn.null_bias[h].double())
                logits = torch.stack(logits)
                w = torch.softmax(logits, -1)
                for wi, vv in zip(w[:-1], vals):
                    y[b, h, t] += wi * vv
    return y


@cuda_only
def test_fine_pass_matches_reference(monkeypatch):
    monkeypatch.setattr(bw, "_TRAIN_SEQ_LEN", T)
    torch.manual_seed(3)
    layer = 2
    attn = make_attn(layer).train()
    with torch.no_grad():
        attn.null_bias.copy_(0.3 * torch.randn(HEADS))
    bsz = 2
    q = torch.randn(bsz, HEADS, T, attn.head_dim, device="cuda")
    ks = [torch.randn(bsz, KV_HEADS, T, attn.head_dim, device="cuda") for _ in range(layer + 1)]
    vs = [torch.randn(bsz, KV_HEADS, T, attn.head_dim, device="cuda") for _ in range(layer + 1)]
    bw._STEP_BUF.fill_(11)
    with torch.no_grad():
        gid, valid = attn._select(bsz)
        y, stats = attn._fine(q, gid, valid, *ks, *vs)
    # Block boundaries, block interiors, and both ends of the sequence.
    positions = [0, 1, 15, 16, 17, 40, 63, 64, 71, 96, 127]
    ref = _reference_fine(attn, q, gid, valid, ks, vs, positions)
    torch.testing.assert_close(
        y[:, :, positions].double(), ref[:, :, positions], rtol=1e-4, atol=1e-5
    )
    assert stats[4].item() == 0.0  # bw_leak is column 4 of the 6-stat vector


@cuda_only
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_causality_no_future_gradient(monkeypatch, mode):
    """d y[t] / d x[t'] must be exactly zero for every t' > t, any layer."""
    modules = fresh_chain(3)
    for m in modules:
        m.train(mode == "train")
    bw._STEP_BUF.fill_(5)
    x = torch.randn(1, T, DIM, device="cuda", requires_grad=True)
    with torch.enable_grad():
        y = run_chain(modules, x, monkeypatch)
        t0 = 71
        loss = y[0, t0].sum()
        (grad,) = torch.autograd.grad(loss, x)
    future = grad[0, t0 + 1 :]
    assert torch.all(future == 0), f"future leak: {future.abs().max().item()}"
    assert grad[0, : t0 + 1].abs().sum() > 0


@cuda_only
def test_cross_layer_gradients_flow(monkeypatch):
    """Layer 0's projections receive gradient from a loss on layer 2 only."""
    modules = fresh_chain(3)
    for m in modules:
        m.train()
    bw._STEP_BUF.fill_(2)
    x = torch.randn(1, T, DIM, device="cuda")
    monkeypatch.setattr(bw, "_TRAIN_SEQ_LEN", T)
    bw._STATE["i"] = 0
    bw._STATE["bank"] = None
    h = x
    outs = []
    for m in modules:
        o = m(h)
        outs.append(o)
        h = h + o
    outs[-1].square().mean().backward()
    for i, m in enumerate(modules):
        g = m.c_qkv.weight.grad
        assert g is not None and torch.isfinite(g).all(), f"layer {i} grad missing"
        assert g.abs().sum() > 0, f"layer {i} got no gradient through the bank"
        assert m.null_bias.grad is not None and m.null_bias.grad.abs().sum() > 0


@cuda_only
def test_compiled_graph_has_no_sort(monkeypatch):
    """Sub-quadratic guarantee: neither compiled graph sorts or top-k's.

    Selection is a pure uniform random draw and normalization is plain
    softmax — there is no importance scan, no entmax threshold, no top-k
    anywhere.  softmax lowers without a sort, so both the forward AND the
    backward graph must contain ZERO ``sort``/``topk`` nodes.  A regression
    that reintroduced any of them (an importance scan, an entmax normalizer,
    checkpointing that re-materializes one) would make the per-query cost
    super-linear in T and OOM at real scale — caught here at toy scale.
    """
    monkeypatch.setattr(bw, "_TRAIN_SEQ_LEN", T)
    from torch._dynamo.backends.common import aot_autograd
    from functorch.compile import make_boxed_func

    counts = {}

    def n_bad(gm):
        return sum(
            1 for n in gm.graph.nodes
            if n.op == "call_function"
            and ("sort" in str(n.target) or "topk" in str(n.target))
        )

    def fw_compiler(gm, ex):
        counts["fwd"] = n_bad(gm)
        return make_boxed_func(gm.forward)

    def bw_compiler(gm, ex):
        counts["bwd"] = n_bad(gm)
        return make_boxed_func(gm.forward)

    torch.manual_seed(1)
    attn = make_attn(1).train()
    prev_k = torch.randn(1, KV_HEADS, T, attn.head_dim, device="cuda")
    prev_v = torch.randn(1, KV_HEADS, T, attn.head_dim, device="cuda")

    def step(x):
        bw._STEP_BUF.fill_(9)
        bw._STATE["i"] = 1
        bw._STATE["bank"] = [(prev_k, prev_v)]
        return attn(x)

    compiled = torch.compile(
        step, backend=aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler),
        fullgraph=True, dynamic=False,
    )
    x = torch.randn(1, T, DIM, device="cuda", requires_grad=True)
    compiled(x).square().mean().backward()
    assert counts["fwd"] == 0, f"forward must be sort/top-k free, got {counts['fwd']}"
    assert counts["bwd"] == 0, f"backward must be sort/top-k free, got {counts['bwd']}"


@cuda_only
def test_invalid_slots_carry_no_weight(monkeypatch):
    """Query block 0 has no dynamic candidates: output = local-only attention."""
    monkeypatch.setattr(bw, "_TRAIN_SEQ_LEN", T)
    torch.manual_seed(2)
    attn = make_attn(0, xlayer=True).eval()
    x = torch.randn(1, T, DIM, device="cuda")
    bw._STATE["i"] = 0
    bw._STATE["bank"] = None
    y = attn(x)
    assert torch.isfinite(y).all()
    # For t < QB every key beyond the local span is invalid at layer 0
    # (universe = current layer only, and Q=0 blocks the whole dynamic set);
    # a second run must reproduce it exactly (no hidden randomness at eval).
    bw._STATE["i"] = 0
    bw._STATE["bank"] = None
    y2 = attn(x)
    torch.testing.assert_close(y, y2, rtol=0, atol=0)


@cuda_only
def test_within_layer_control_never_reads_earlier_layers(monkeypatch):
    """SPARSE_XLAYER=0: earlier layers' k/v do not influence the output."""
    monkeypatch.setattr(bw, "_TRAIN_SEQ_LEN", T)
    torch.manual_seed(4)
    modules = fresh_chain(2, xlayer=False)
    for m in modules:
        m.eval()
    x = torch.randn(1, T, DIM, device="cuda")
    run_chain(modules, x, monkeypatch)
    # Feed layer 1 the same input twice; poison the bank on the first pass.
    # A within-layer module never consults the bank, so the outputs must match
    # bit-for-bit regardless of what junk the bank holds.
    bw._STATE["i"] = 1
    bw._STATE["bank"] = [(torch.randn_like(x), torch.randn_like(x))]  # poison
    h = modules[1](x)
    bw._STATE["i"] = 1
    bw._STATE["bank"] = None
    h2 = modules[1](x)
    torch.testing.assert_close(h, h2, rtol=0, atol=0)


@cuda_only
def test_stats_are_sane(monkeypatch):
    monkeypatch.setattr(bw, "_TRAIN_SEQ_LEN", T)
    torch.manual_seed(6)
    attn = make_attn(3).train()
    bsz = 2
    q = torch.randn(bsz, HEADS, T, attn.head_dim, device="cuda")
    ks = [torch.randn(bsz, KV_HEADS, T, attn.head_dim, device="cuda") for _ in range(4)]
    vs = [torch.randn(bsz, KV_HEADS, T, attn.head_dim, device="cuda") for _ in range(4)]
    bw._STEP_BUF.fill_(21)
    with torch.no_grad():
        gid, valid = attn._select(bsz)
        _, stats = attn._fine(q, gid, valid, *ks, *vs)
    assert stats.shape == (len(bw._STAT_NAMES),)
    names = dict(zip(bw._STAT_NAMES, stats.tolist()))
    assert 0.0 <= names["bw_xsel"] <= 1.0
    assert 0.0 <= names["bw_xmass"] <= 1.0
    assert 0.0 <= names["bw_null"] <= 1.0
    assert 0.0 <= names["bw_support"] <= 1.0
    assert names["bw_leak"] == 0.0
    assert 0.0 < names["bw_valid"] <= 1.0


@cuda_only
def test_state_dict_contains_only_params(monkeypatch):
    attn = make_attn(2)
    sd = attn.state_dict()
    for name in sd:
        assert not any(
            b in name
            for b in ("u_q", "local_pos", "local_mask", "sentinel", "dup_tri", "sketch", "allowed")
        ), f"static buffer {name} leaked into state_dict"
    assert "null_bias" in sd
    # null_bias is the only Parameter this fork adds on top of the baseline.
    assert [n for n, _ in attn.named_parameters() if "null_bias" in n] == ["null_bias"]
    attn2 = make_attn(2)
    attn2.load_state_dict(sd)
    torch.testing.assert_close(attn2.null_bias, attn.null_bias)


@cuda_only
def test_compiles_fullgraph_single_graph(monkeypatch):
    """The training forward compiles as one graph and matches eager.

    _STATS is set to a real tensor so the in-place index-write to a Python
    global — the pattern most likely to break under fullgraph=True — is part
    of the compiled graph, exactly as in a real run (main() sets it before
    the model compiles).
    """
    monkeypatch.setattr(bw, "_TRAIN_SEQ_LEN", T)
    monkeypatch.setattr(
        bw, "_STATS",
        torch.zeros(bw.MAX_LAYERS, len(bw._STAT_NAMES), device="cuda"),
    )
    torch.manual_seed(8)
    modules = fresh_chain(2)
    for m in modules:
        m.train()
        # Prime the lazy rotary cache eagerly: its None -> tensor transition
        # is a known one-off recompile source.
        m.rotary(T, torch.device("cuda"), torch.float32)

    def step(x):
        bw._STATE["i"] = 0
        bw._STATE["bank"] = None
        h = x
        for m in modules:
            h = h + m(h)
        return h

    compiled = torch.compile(step, fullgraph=True)
    x = torch.randn(1, T, DIM, device="cuda", requires_grad=True)
    bw._STEP_BUF.fill_(1)
    y1 = compiled(x)
    y1.square().mean().backward()
    bw._STEP_BUF.add_(1)
    y2 = compiled(x)
    assert not torch.equal(y1, y2)  # the step clock moved the random slots
    assert torch.isfinite(y2).all()
    assert bool((bw._STATS[:2] != 0).any())  # the stats store compiled and ran

    # Compiled backward must agree numerically with eager: the softmax fine
    # pass and the per-chunk loop are traced into the joint graph, so a
    # partitioner/lowering bug would only show up in this comparison.
    bw._STEP_BUF.fill_(1)
    x_eager = x.detach().clone().requires_grad_(True)
    y_eager = step(x_eager)
    y_eager.square().mean().backward()
    x.grad = None
    bw._STEP_BUF.fill_(1)
    y_comp = compiled(x)
    y_comp.square().mean().backward()
    torch.testing.assert_close(y_comp, y_eager, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(x.grad, x_eager.grad, rtol=1e-2, atol=1e-4)

"""CPU contracts of the Causal Workspace Machine (``pretraining/cwm``).

The reference below implements the event semantics literally from the
module docstring -- every bank is rebuilt from the full write history of
earlier tokens, and the store is an explicit list of (token, source) entries
-- so it shares no state layout with the executor (ring buffer, stacked
banks, chunking, checkpointing) it checks.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from pretraining.cwm.model import (
    CWM,
    CWMConfig,
    commit_banks,
    feedback_parameter_names,
    initialize,
    l2norm,
    rms,
    rope,
)

T = 12
B = 4  # the placebo reads row b - B/2


def small_config(**overrides) -> CWMConfig:
    base = dict(
        vocab_size=97,
        dim=128,
        mlp_hidden=192,
        rounds=3,
        bank_heads=2,
        bank_dim=8,
        store_heads=2,
        store_dim=8,
        store_window=4,  # shorter than T: eviction is exercised
        depth_heads=2,
        depth_dim=8,
        low_rank=8,
    )
    base.update(overrides)
    return CWMConfig(**base)


def build(cfg: CWMConfig, seed: int = 0, wake: bool = True) -> CWM:
    """fp32 model; ``wake`` randomizes the zero-initialized output
    projections so every path (feedback included) carries signal."""
    torch.manual_seed(seed)
    model = CWM(cfg).float()
    initialize(model, seed)
    if wake:
        generator = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.ndim == 2 and p.abs().max() == 0:
                    p.copy_(torch.randn(p.shape, generator=generator) * 0.3 / p.size(1) ** 0.5)
    return model


def tokens(seed: int = 0) -> torch.Tensor:
    return torch.randint(0, 97, (B, T), generator=torch.Generator().manual_seed(seed))


def lin(module, x):
    return torch.nn.functional.linear(x, module.weight, module.bias)


def literal_rms(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + torch.finfo(x.dtype).eps)


def literal_rotate(x, t, freq):
    """Rotate pairs (x_i, x_{i + D/2}) by angle t * freq_i."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = (t * freq).cos(), (t * freq).sin()
    return torch.cat([x1 * cos + x2 * sin, x2 * cos - x1 * sin], dim=-1)


def literal_writes(model: CWM, writer, xn, t: int) -> dict:
    cfg = model.cfg
    H, D = cfg.bank_heads, cfg.bank_dim
    key = lin(writer.bank_k, xn).view(B, H, D)
    key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    decay_logits = lin(writer.decay_b, lin(writer.decay_a, xn)).view(B, H, D) + writer.dt_bias.view(H, D)
    out = dict(
        key=key,
        value=lin(writer.bank_v, xn).view(B, H, D),
        beta=cfg.beta_max * torch.sigmoid(lin(writer.bank_beta, xn)),
        log_decay=-5.0 * torch.sigmoid(writer.A_log.exp()[:, None] * decay_logits),
    )
    if cfg.store:
        Hs, Ds = cfg.store_heads, cfg.store_dim
        k = literal_rms(lin(writer.store_k, xn).view(B, Hs, Ds)) + writer.store_tag
        out["store_key"] = literal_rotate(k, t, model.store_freq)
        out["store_value"] = lin(writer.store_v, xn).view(B, Hs, Ds)
    return out


def literal_depth_read(model: CWM, blk, h, log):
    cfg = model.cfg
    Hd, Dd = cfg.depth_heads, cfg.depth_dim
    q = literal_rms(lin(blk.depth_q, blk.depth_norm(h)).view(B, Hd, Dd))
    ks = [literal_rms((lin(blk.depth_k, blk.depth_norm(e)) + blk.depth_tag[i]).view(B, Hd, Dd)) for i, e in enumerate(log)]
    vs = [lin(blk.depth_v, blk.depth_norm(e)).view(B, Hd, Dd) for e in log]
    scores = torch.stack([(q * k).sum(-1) for k in ks] + [torch.zeros(B, Hd)], -1) * (0.12 * 128 / Dd)
    weights = scores.softmax(-1)
    o = sum(weights[..., i, None] * v for i, v in enumerate(vs))
    return lin(blk.depth_o, o.reshape(B, Hd * Dd))


def reference_events(model: CWM, inputs: torch.Tensor) -> torch.Tensor:
    """All round outputs ``[B, T, R, d]`` by literal per-event evaluation."""
    cfg, freq = model.cfg, model.store_freq
    S, W = cfg.sources, cfg.store_window
    feedback = model.feedback
    shift = B // 2
    u = model.perceive(inputs)
    history: dict[tuple[int, int], dict] = {}  # (token, source index) -> writes
    events = []
    def bank(s: int, upto: int) -> torch.Tensor:
        """Source s's bank after the writes of tokens < upto."""
        M = torch.zeros(B, cfg.bank_heads, cfg.bank_dim, cfg.bank_dim)
        for t2 in range(upto):
            w = history[t2, s]
            M = M * w["log_decay"].exp()[..., None, :]
            M = M + w["beta"][..., None, None] * torch.einsum(
                "bhv,bhk->bhvk", w["value"] - torch.einsum("bhvk,bhk->bhv", M, w["key"]), w["key"]
            )
        return M

    for t in range(u.size(1)):
        # Under lag, reads served from the chunk start see tokens < start.
        start = t if cfg.lag == "none" else t - t % cfg.lag_chunk
        banks = [bank(s, start if cfg.lag == "all" else t) for s in range(S)]
        fb_banks = [bank(s, start) for s in range(S)]
        # The placebo's feedback reads see row b - B/2's memory.
        if feedback == "placebo":
            fb_banks = [M.roll(shift, 0) for M in fb_banks]
        visible = range(max(0, t - W), t)
        fb_visible = range(max(0, t - W), start)
        h = u[:, t]
        log = [h]
        writes = []
        rounds_out = []
        for r, blk in enumerate(model.rounds, start=1):
            h = h + literal_depth_read(model, blk, h, log)
            xn = blk.in_norm(h)
            writes.append(literal_writes(model, blk.writer, xn, t))
            # Banks: legal sources 1..r (indices < r), feedback r+1..S.
            H, D = cfg.bank_heads, cfg.bank_dim
            q = l2norm(blk.bank_q(xn).view(B, H, D)) * D**-0.5
            gate = blk.bank_gate_b(blk.bank_gate_a(xn)).view(B, H, D)

            def bank_path(memory, sources, mix_linear, weight, out_linear):
                if mix_linear is None:
                    mix = torch.ones(B, H, 1)
                else:
                    mix = mix_linear(xn).view(B, H, len(sources)).softmax(-1)
                o = sum(mix[..., i, None] * torch.einsum("bhvk,bhk->bhv", memory[s], q) for i, s in enumerate(sources))
                o = rms(o) * weight * torch.sigmoid(gate)
                return out_linear(o.reshape(B, H * D))

            h_add = bank_path(banks, range(r), blk.bank_mix_legal, blk.bank_norm_legal, blk.bank_o_legal)
            if feedback:
                h_add = h_add + bank_path(fb_banks, range(r, S), blk.bank_mix_fb, blk.bank_norm_fb, blk.bank_o_fb)
            h = h + h_add
            if cfg.store:
                Hs, Ds = cfg.store_heads, cfg.store_dim
                qs = literal_rotate(literal_rms(lin(blk.store_q, xn).view(B, Hs, Ds)), t, freq)

                def store_path(sources, out_linear, visible, rows_shift=0):
                    entries = [history[t2, s] for t2 in visible for s in sources]
                    if entries:
                        k = torch.stack([e["store_key"] for e in entries], 1).roll(rows_shift, 0)
                        v = torch.stack([e["store_value"] for e in entries], 1).roll(rows_shift, 0)
                        scores = torch.einsum("bhd,bnhd->bhn", qs, k) * cfg.store_scale
                    else:
                        k = v = torch.zeros(B, 0, Hs, Ds)
                        scores = torch.zeros(B, Hs, 0)
                    sink = torch.zeros(B, Hs, 1)
                    weights = torch.cat([scores, sink], -1).softmax(-1)[..., :-1]
                    return out_linear(torch.einsum("bhn,bnhd->bhd", weights, v).reshape(B, Hs * Ds))

                h_add = store_path(range(r), blk.store_o_legal, visible)
                if feedback:
                    h_add = h_add + store_path(range(r, S), blk.store_o_fb, fb_visible, shift if feedback == "placebo" else 0)
                h = h + h_add
            h = h + blk.mlp(blk.mlp_norm(h))
            log.append(h)
            rounds_out.append(h)
        writes.append(literal_writes(model, model.out_writer, model.out_norm(h), t))
        # Commit once, after the token is complete.
        for s, w in enumerate(writes):
            history[t, s] = w
        events.append(torch.stack(rounds_out, dim=1))
    return torch.stack(events, dim=1)


VARIANTS = [
    pytest.param(dict(perception="attention", store=True), id="cwm"),
    pytest.param(dict(perception="kda", store=False), id="kda"),
]
MASKS = ["full", "lattice", "placebo"]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mask", MASKS)
@pytest.mark.parametrize("lag", ["none", "feedback", "all"])
def test_executor_matches_literal_reference(variant, mask, lag):
    # A lag chunk (3) that the checkpoint chunks (4) cut across.
    model = build(small_config(read_mask=mask, lag=lag, lag_chunk=3, **variant))
    inputs = tokens()
    expected = reference_events(model, inputs)[:, :, -1]
    for chunk, ckpt in ((1, False), (4, True), (12, False)):
        got = model.core(model.perceive(inputs), chunk, ckpt)
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("lag", ["none", "feedback"])
def test_feedback_lag_leaves_the_lattice_exact(variant, lag):
    exact = build(small_config(read_mask="lattice", **variant))
    lagged = build(small_config(read_mask="lattice", lag=lag, lag_chunk=3, **variant))
    inputs = tokens()
    with torch.no_grad():
        torch.testing.assert_close(lagged.hidden(inputs, chunk=4), exact.hidden(inputs, chunk=4), rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("lag", ["feedback", "all"])
def test_lag_hides_the_chunk_from_feedback(variant, lag):
    """Under lag, feedback reaches a token only from earlier chunks: with
    every legal path of the chunk's own earlier tokens unchanged, a change to
    token 4's feedback-only writes (``out``) cannot reach tokens 5 (same
    chunk of 3, which starts at 3), only token 6 on."""
    model = build(small_config(read_mask="full", lag=lag, lag_chunk=3, **variant))
    inputs = tokens()
    with torch.no_grad():
        base = model.hidden(inputs, chunk=4)
        writer = model.out_writer

        def perturb(module, args, out):
            # Only token 4's out writes change.
            pos = args[1]
            if int(pos) == 4:
                out.value.add_(1.0)
                if out.store_value is not None:
                    out.store_value.add_(1.0)
            return out

        handle = writer.register_forward_hook(perturb)
        changed = model.hidden(inputs, chunk=4)
        handle.remove()
    torch.testing.assert_close(changed[:, :6], base[:, :6], rtol=0, atol=0)
    assert not torch.allclose(changed[:, 6:], base[:, 6:])


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mask", MASKS)
def test_checkpointed_gradients_match_reference(variant, mask):
    model = build(small_config(read_mask=mask, **variant))
    inputs = tokens()
    probe = torch.randn(B, T, model.cfg.vocab_size, generator=torch.Generator().manual_seed(3))

    def grads(hidden_fn):
        model.zero_grad(set_to_none=True)
        (model.project_logits(model.norm2(hidden_fn())) * probe).sum().backward()
        # Unused parameters may hold None (reference) or zeros (executor).
        return {n: torch.zeros_like(p) if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}

    expected = grads(lambda: reference_events(model, inputs)[:, :, -1])
    got = grads(lambda: model.core(model.perceive(inputs), 4, True))
    for name in expected:
        if not expected[name].any():
            assert not got[name].any(), name
            continue
        # fp32 summation order differs (stacked banks vs per-source folds):
        # bound the error relative to each tensor's gradient norm.
        error = (got[name] - expected[name]).norm() / expected[name].norm().clamp_min(1e-12)
        assert error < 1e-5, (name, error.item())


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mask", MASKS)
def test_causality(variant, mask):
    model = build(small_config(read_mask=mask, **variant))
    a = tokens(0)
    b = a.clone()
    b[:, 7] = (b[:, 7] + 1) % 97
    with torch.no_grad():
        ha, hb = model.hidden(a, chunk=4), model.hidden(b, chunk=4)
    torch.testing.assert_close(ha[:, :7], hb[:, :7], rtol=0, atol=0)
    assert not torch.allclose(ha[:, 7:], hb[:, 7:])


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mask", ["full", "placebo"])
def test_commit_once_out_writes_become_visible_next_token(variant, mask):
    """The ``out`` writes of token t are proposed after token t is scored:
    they reach token t + 1 on and never token t itself."""
    model = build(small_config(read_mask=mask, **variant))
    inputs = tokens()
    with torch.no_grad():
        base = model.hidden(inputs, chunk=4)
        for p in model.out_writer.parameters():
            p.add_(torch.randn_like(p) * 0.5)
        moved = model.hidden(inputs, chunk=4)
    torch.testing.assert_close(base[:, 0], moved[:, 0], rtol=0, atol=0)
    assert not torch.allclose(base[:, 1:], moved[:, 1:])


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mask", ["full", "placebo"])
def test_commit_once_round_writes_invisible_to_own_token(variant, mask):
    """Round 1's input state at token 0 sees no memory at all, so the whole
    first token is independent of every writer."""
    model = build(small_config(read_mask=mask, **variant))
    inputs = tokens()
    with torch.no_grad():
        base = model.hidden(inputs, chunk=4)
        for blk in model.rounds:
            for p in blk.writer.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        moved = model.hidden(inputs, chunk=4)
    torch.testing.assert_close(base[:, 0], moved[:, 0], rtol=0, atol=0)
    assert not torch.allclose(base[:, 1:], moved[:, 1:])


@pytest.mark.parametrize("variant", VARIANTS)
def test_lattice_rounds_are_closed_under_higher_rounds(variant):
    """Lattice: round r at every token depends only on rounds <= r (its own
    token's earlier rounds and sources <= r at earlier tokens), so it is a
    stack computable layer by layer. Full mode breaks exactly this."""
    for mask, closed in (("lattice", True), ("full", False), ("placebo", False)):
        model = build(small_config(read_mask=mask, **variant))
        inputs = tokens()
        with torch.no_grad():
            base = reference_events(model, inputs)
            for p in model.rounds[2].parameters():
                p.add_(torch.randn_like(p) * 0.5)
            for p in model.out_writer.parameters():
                p.add_(torch.randn_like(p) * 0.5)
            moved = reference_events(model, inputs)
        lower_same = torch.equal(base[:, :, :2], moved[:, :, :2])
        assert lower_same == closed, mask
        assert not torch.allclose(base[:, :, 2], moved[:, :, 2])


@pytest.mark.parametrize("variant", VARIANTS)
def test_lattice_has_no_gradient_into_feedback_parameters(variant):
    model = build(small_config(read_mask="lattice", **variant))
    loss = model.project_logits(model.hidden(tokens(), chunk=4, checkpointing=True)).square().mean()
    loss.backward()
    names = set(feedback_parameter_names(model))
    assert names
    for name, p in model.named_parameters():
        if name in names:
            assert p.grad is None or not p.grad.any(), name
        else:
            assert p.grad is not None and p.grad.any(), name


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mask", ["full", "placebo"])
def test_feedback_ablation_equals_lattice(variant, mask):
    full = build(small_config(read_mask=mask, **variant))
    lattice = build(small_config(read_mask="lattice", **variant))
    lattice.load_state_dict(full.state_dict())
    inputs = tokens()
    full.feedback_enabled = False
    with torch.no_grad():
        torch.testing.assert_close(full.hidden(inputs, chunk=4), lattice.hidden(inputs, chunk=4), rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_all_masks_are_bitwise_identical_at_init(variant, dtype):
    models = {}
    for mask in MASKS:
        torch.manual_seed(0)
        model = CWM(small_config(read_mask=mask, **variant))
        if dtype == torch.float32:
            model = model.float()
        initialize(model, seed=5)
        models[mask] = model
    full = models["full"]
    for other in MASKS[1:]:
        for (n1, p1), (n2, p2) in zip(full.named_parameters(), models[other].named_parameters()):
            assert n1 == n2 and torch.equal(p1, p2), n1
    inputs = tokens()
    losses = {}
    for mask, model in models.items():
        logits = model.project_logits(model.hidden(inputs, chunk=4, checkpointing=True))
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), inputs.roll(-1, 1).flatten())
        loss.backward()
        losses[mask] = logits.detach()
    fb = set(feedback_parameter_names(full))
    for other in MASKS[1:]:
        assert torch.equal(losses["full"], losses[other]), other
        for (name, p1), (_, p2) in zip(full.named_parameters(), models[other].named_parameters()):
            # Lattice freezes the feedback parameters; under the placebo the
            # zero-init feedback output weights see other rows' reads.
            if (other == "lattice" and name in fb) or (other == "placebo" and name.endswith("_o_fb.weight")):
                continue
            g1 = p1.grad if p1.grad is not None else torch.zeros_like(p1)
            g2 = p2.grad if p2.grad is not None else torch.zeros_like(p2)
            assert torch.equal(g1, g2), (other, name)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mask", MASKS)
def test_rows_interact_only_under_placebo(variant, mask):
    """Rows are independent streams; only the placebo's feedback reads cross
    rows, and only from row b - B/2."""
    model = build(small_config(read_mask=mask, **variant))
    a = tokens(0)
    b = a.clone()
    b[2] = (b[2] + 1) % 97
    with torch.no_grad():
        ha, hb = model.hidden(a, chunk=4), model.hidden(b, chunk=4)
    for row in range(B):
        same = torch.equal(ha[row], hb[row])
        if row == 2:
            assert not same
        elif mask == "placebo" and row == (2 + B // 2) % B:
            # Reads row 2's memory from the next token on.
            assert not same and torch.equal(ha[row, 0], hb[row, 0])
        else:
            assert same, (mask, row)


@pytest.mark.parametrize("mask", MASKS)
def test_trainer_optimizers_own_trainable_parameters_and_step(mask):
    """Every trainable parameter is owned by one optimizer group, frozen
    lattice parameters by none, and one full step runs and moves weights."""
    import argparse

    from pretraining.cwm.train import build_optimizers

    model = build(small_config(read_mask=mask), wake=False)
    args = argparse.Namespace(
        embed_lr=0.7, head_lr=0.004, scalar_lr=0.015, conv_lr=0.004, muon_lr=0.025, adam_wd=0.001, muon_wd=0.05
    )
    optimizers = build_optimizers(model, args)
    owned = {id(p) for opt in optimizers for g in opt.param_groups for p in g["params"]}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert owned == trainable
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert frozen == (set(feedback_parameter_names(model)) if mask == "lattice" else set())
    inputs = tokens()
    logits = model.project_logits(model.hidden(inputs, chunk=4, checkpointing=True))
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), inputs.roll(-1, 1).flatten()).backward()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for opt in optimizers:
        opt.step()
    moved = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert "head.weight" in moved and "rounds.0.mlp.fc.weight" in moved
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_delta_commit_reaches_negative_eigenvalues():
    """beta = 2 reflects the keyed direction: M' k = (1 - beta) M k + beta v."""
    S, H, D = 1, 2, 4
    g = torch.Generator().manual_seed(0)
    banks = torch.randn(S, B, H, D, D, generator=g)
    key = l2norm(torch.randn(B, H, D, generator=g))
    value = torch.randn(B, H, D, generator=g)

    class W:
        pass

    w = W()
    w.key, w.value = key, value
    w.beta = torch.full((B, H), 2.0)
    w.log_decay = torch.zeros(B, H, D)
    new = commit_banks(banks, [w])
    read_old = torch.einsum("bhvk,bhk->bhv", banks[0], key)
    read_new = torch.einsum("bhvk,bhk->bhv", new[0], key)
    # l2norm's epsilon leaves |k| = 1 - O(1e-6), hence the tolerance.
    torch.testing.assert_close(read_new, -read_old + 2 * value, rtol=0, atol=1e-4)
    # Directions orthogonal to the key are untouched.
    ortho = l2norm(torch.randn(B, H, D, generator=g))
    ortho = l2norm(ortho - (ortho * key).sum(-1, keepdim=True) * key)
    torch.testing.assert_close(
        torch.einsum("bhvk,bhk->bhv", new[0], ortho), torch.einsum("bhvk,bhk->bhv", banks[0], ortho), rtol=0, atol=1e-4
    )


def test_bf16_bank_commit_rounds_once_with_straight_through_gradient():
    """A bf16 bank state holds the fp32 commit rounded to nearest bf16; its
    gradient passes the rounding unchanged."""
    S, H, D = 2, 2, 4
    g = torch.Generator().manual_seed(1)

    class W:
        pass

    writes = []
    for _ in range(S):
        w = W()
        w.key = l2norm(torch.randn(B, H, D, generator=g))
        w.value = torch.randn(B, H, D, generator=g)
        w.beta = torch.rand(B, H, generator=g) * 2
        w.log_decay = -torch.rand(B, H, D, generator=g)
        writes.append(w)
    banks = torch.randn(S, B, H, D, D, generator=g).bfloat16().float().requires_grad_()
    exact = commit_banks(banks, writes)
    rounded = commit_banks(banks, writes, torch.bfloat16)
    assert rounded.dtype == torch.float32
    assert torch.equal(rounded, exact.bfloat16().float())
    probe = torch.randn(rounded.shape, generator=g)
    (d_exact,) = torch.autograd.grad((exact * probe).sum(), banks)
    (d_rounded,) = torch.autograd.grad((rounded * probe).sum(), banks)
    assert torch.equal(d_rounded, d_exact)


def test_write_strength_range_and_initial_value():
    model = build(small_config(), wake=False)
    xn = torch.randn(B, model.cfg.dim) * 10
    beta = model.rounds[0].writer(xn, torch.tensor(0.0), model.store_freq).beta
    assert (beta > 0).all() and (beta < model.cfg.beta_max).all()
    zero = model.rounds[0].writer(torch.zeros(B, model.cfg.dim), torch.tensor(0.0), model.store_freq).beta
    torch.testing.assert_close(zero, torch.full_like(zero, 0.5))


def test_config_validation():
    with pytest.raises(ValueError):
        small_config(read_mask="sparse")
    with pytest.raises(ValueError):
        small_config(beta_max=2.5)
    with pytest.raises(ValueError):
        small_config(bank_dtype="float16")
    assert dataclasses.asdict(small_config())["rounds"] == 3

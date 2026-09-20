"""CPU tests for ``pretraining/nanogpt_mini/nanogpt_mini_slot_model.py``.

Pins the contracts the slot-memory line relies on: the alive table and the
visibility rule against loops (overwrite, null writes, strict ``i < t``);
the FlexAttention ``mask_mod`` and the table-derived block structure against
the dense reference; the fifo table as a sliding window; ``full`` mode
bitwise the base ``GPT``; the discounted advantage against a loop; where the
policy-gradient term's gradient goes (slot head always, trunk only without
``detach``); the baseline EMA and entropy knob; and the sequential evaluator
against the parallel dense pass and a prefix recompute with injected slots.
"""

import math

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_model import GPT
from pretraining.nanogpt_mini.nanogpt_mini_slot_model import (
    NO_WRITE,
    SlotGPT,
    block_presence,
    build_alive_table,
    discounted_future_credit,
    fifo_sigma,
    head_choice_to_sigma,
    sample_slot_choices,
    slot_mask_mod,
    visibility_reference,
)

TINY = dict(vocab_size=64, num_layers=2, model_dim=32, head_dim=16, slots=4, backend="dense")


def alive_loop(sigma, slots):
    B, T = sigma.shape
    out = torch.full((B, T, slots), NO_WRITE, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            for s in range(slots):
                writes = [i for i in range(t) if int(sigma[b, i]) == s]
                out[b, t, s] = max(writes) if writes else NO_WRITE
    return out


def visible_loop(alive, sigma):
    B, T, _ = alive.shape
    out = torch.zeros(B, T, T, dtype=torch.bool)
    for b in range(B):
        for t in range(T):
            for i in range(T):
                s = int(sigma[b, i])
                out[b, t, i] = (i == t) or (s >= 0 and int(alive[b, t, s]) == i)
    return out


def random_sigma(B, T, slots, seed=0, null_prob=0.3):
    g = torch.Generator().manual_seed(seed)
    sigma = torch.randint(0, slots, (B, T), generator=g)
    null = torch.rand(B, T, generator=g) < null_prob
    return torch.where(null, torch.full_like(sigma, NO_WRITE), sigma)


def tiny_model(mode="policy", seed=0, **kw):
    torch.manual_seed(seed)
    cfg = {**TINY, **kw}
    model = SlotGPT(mode=mode, **cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("slot_head."):
                p.normal_(std=0.5)  # a non-uniform policy so log-probs vary
            elif name.endswith("gains"):
                p.fill_(1.0)
            else:
                p.normal_(std=0.2)
    return model.float()


def test_alive_table_matches_loop_with_overwrite_and_null():
    sigma = random_sigma(3, 17, 5, seed=1)
    alive = build_alive_table(sigma, 5)
    assert alive.shape == (3, 17, 5)
    assert torch.equal(alive, alive_loop(sigma, 5))
    assert (alive[:, 0] == NO_WRITE).all()  # nothing written before position 0


def test_visibility_semantics_hand_case():
    # sigma: 0 -> slot0, 1 -> slot0 (overwrites), 2 -> null, 3 -> slot1
    sigma = torch.tensor([[0, 0, NO_WRITE, 1, NO_WRITE]])
    alive = build_alive_table(sigma, 2)
    assert alive[0].tolist() == [[-1, -1], [0, -1], [1, -1], [1, -1], [1, 3]]
    vis = visibility_reference(alive, sigma)
    expect = torch.tensor([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],   # t=1 sees its own write? no: sees key 0 (slot 0) + self
        [0, 1, 1, 0, 0],   # key 0 was overwritten by 1
        [0, 1, 0, 1, 0],   # the null write at 2 is invisible; 3's own write not yet
        [0, 1, 0, 1, 1],   # slot 1 now holds 3
    ], dtype=torch.bool)
    assert torch.equal(vis[0], expect)
    assert torch.equal(vis, visible_loop(alive, sigma))


def test_visibility_reference_matches_loop_random():
    sigma = random_sigma(2, 24, 4, seed=2)
    alive = build_alive_table(sigma, 4)
    vis = visibility_reference(alive, sigma)
    assert torch.equal(vis, visible_loop(alive, sigma))
    # at most M + 1 keys per query
    assert int(vis.sum(-1).max()) <= 5


def test_mask_mod_and_block_presence_match_dense():
    T, M, bs = 96, 6, 8
    sigma = random_sigma(2, T, M, seed=3)
    alive = build_alive_table(sigma, M)
    vis = visibility_reference(alive, sigma)
    mask_mod = slot_mask_mod(alive, sigma)
    b = torch.arange(2).view(2, 1, 1)
    q = torch.arange(T).view(1, T, 1)
    k = torch.arange(T).view(1, 1, T)
    dense_from_mod = mask_mod(b, torch.zeros((), dtype=torch.long), q, k)
    assert torch.equal(dense_from_mod, vis)
    # every block holding a visible pair is present in the table-derived structure,
    # and a query block never needs more than M occupant blocks + its diagonal
    presence = block_presence(alive, bs)
    nb = T // bs
    true_presence = vis.view(2, nb, bs, nb, bs).any(-1).any(2)
    assert (true_presence <= presence).all()
    assert (presence.sum(-1) <= M + 1).all()
    assert int(presence.sum()) < 2 * (nb * (nb + 1) // 2)  # sparser than causal


def test_fifo_table_is_sliding_window():
    M, T = 4, 12
    sigma = fifo_sigma(1, T, M, torch.device("cpu"))
    alive = build_alive_table(sigma, M)
    assert torch.equal(alive, alive_loop(sigma, M))
    vis = visibility_reference(alive, sigma)[0]
    for t in range(T):
        keys = set(torch.nonzero(vis[t]).flatten().tolist())
        assert keys == {i for i in range(max(0, t - M), t + 1)}


def test_full_mode_is_base_gpt_bitwise():
    torch.manual_seed(0)
    base = GPT(vocab_size=64, num_layers=2, model_dim=128)
    with torch.no_grad():
        for name, p in base.named_parameters():
            if name.endswith("weight") and "proj" in name:
                p.normal_(std=0.05)
    slot = SlotGPT(vocab_size=64, num_layers=2, model_dim=128, slots=8, mode="full", backend="dense")
    assert list(slot.state_dict()) == list(base.state_dict())
    slot.load_state_dict(base.state_dict())
    inputs = torch.randint(0, 64, (2, 8), dtype=torch.int32)
    targets = torch.randint(0, 64, (2, 8))
    loss, stats = slot(inputs, targets)
    ref = base(inputs, targets)
    assert torch.equal(loss, ref)
    assert torch.equal(stats["ce_last"], ref)


def test_discounted_future_credit_matches_loop():
    torch.manual_seed(0)
    delta = torch.randn(3, 20)
    gamma, H = 0.9, 5
    credit = discounted_future_credit(delta, gamma, H)
    for b in range(3):
        for t in range(20):
            ref = sum(gamma ** (u - t) * float(delta[b, u]) for u in range(t + 1, min(t + H, 19) + 1))
            assert abs(float(credit[b, t]) - ref) < 1e-5
    assert (credit[:, -1] == 0).all()
    long = discounted_future_credit(delta, gamma, 1000)
    for b in range(3):
        ref = sum(gamma ** u * float(delta[b, u]) for u in range(1, 20))
        assert abs(float(long[b, 0]) - ref) < 1e-4


def test_uniform_initial_policy_and_sampling():
    torch.manual_seed(0)
    model = SlotGPT(mode="policy", **TINY).float()
    top = torch.randn(2, 7, 32)
    logits = model.slot_logits(top)
    assert torch.equal(logits, torch.zeros(2, 7, 5))
    choice, logp, entropy = sample_slot_choices(logits)
    assert torch.allclose(logp, torch.full((2, 7), -math.log(5)))
    assert torch.allclose(entropy, torch.full((2, 7), math.log(5)))
    assert choice.min() >= 0 and choice.max() <= 4
    big = sample_slot_choices(torch.zeros(20000, 5))[0]
    assert abs(float((big == 4).float().mean()) - 0.2) < 0.02
    sigma = head_choice_to_sigma(choice, 4)
    assert torch.equal(sigma == NO_WRITE, choice == 4)
    greedy = sample_slot_choices(torch.tensor([[0.1, 2.0, -1.0]]), greedy=True)[0]
    assert greedy.tolist() == [1]


def trunk_grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None and not n.startswith("slot_head.")}


def test_policy_gradient_reaches_head_and_trunk_as_intended():
    inputs = torch.randint(0, 64, (2, 12), dtype=torch.int32)
    targets = torch.randint(0, 64, (2, 12))
    results = {}
    for detach in (False, True):
        model = tiny_model("policy", detach=detach)
        model.train()
        torch.manual_seed(11)
        loss, stats = model(inputs, targets)
        loss.backward()
        results[detach] = (model, stats)
    m0, s0 = results[False]
    m1, s1 = results[True]
    # same weights and RNG: same slot trajectory and the same slot-head gradient either way
    assert torch.equal(s0["sigma"], s1["sigma"])
    assert torch.allclose(m0.slot_head.weight.grad, m1.slot_head.weight.grad, atol=1e-6)
    assert m0.slot_head.weight.grad.abs().sum() > 0
    assert m0.slot_head.bias.grad.abs().sum() > 0
    # detach: trunk gradient is exactly the CE-only gradient (pass 1 + pass 2 under the same sigma)
    ref = tiny_model("policy", detach=True)
    x0 = ref.norm1(ref.embed(inputs))
    ce1 = ref.token_ce(ref.trunk(x0, None), targets).sum()
    ce2 = ref.forward_with_slots(inputs, targets, s1["sigma"]).sum()
    (ce1 + ce2).backward()
    for name, g in trunk_grads(ref).items():
        assert torch.allclose(g, dict(m1.named_parameters())[name].grad, atol=1e-5, rtol=1e-4), name
    # without detach the policy term also moves the trunk
    diffs = [float((g - dict(m0.named_parameters())[name].grad).abs().max()) for name, g in trunk_grads(ref).items()]
    assert max(diffs) > 1e-6


def test_entropy_knob_changes_loss_by_entropy_sum():
    inputs = torch.randint(0, 64, (2, 10), dtype=torch.int32)
    targets = torch.randint(0, 64, (2, 10))
    losses = {}
    ents = {}
    for coef in (0.0, 0.5):
        model = tiny_model("policy", entropy_coef=coef)
        model.train()
        torch.manual_seed(3)
        loss, stats = model(inputs, targets)
        losses[coef] = float(loss.detach())
        ents[coef] = stats
    assert torch.equal(ents[0.0]["sigma"], ents[0.5]["sigma"])
    # entropy of the head that produced the trajectory, last position excluded
    model = tiny_model("policy")
    x0 = model.norm1(model.embed(inputs))
    _, _, entropy = sample_slot_choices(model.slot_logits(model.trunk(x0, None)))
    expected = losses[0.0] - 0.5 * float(entropy[:, :-1].sum().detach())
    assert abs(losses[0.5] - expected) < 1e-3 * max(1.0, abs(expected))


def test_baseline_ema_seeds_then_decays():
    inputs = torch.randint(0, 64, (2, 10), dtype=torch.int32)
    targets = torch.randint(0, 64, (2, 10))
    model = tiny_model("policy", baseline_decay=0.5)
    model.train()
    assert not bool(model.adv_baseline_ready)
    _, s1 = model(inputs, targets)
    assert bool(model.adv_baseline_ready)
    assert torch.allclose(model.adv_baseline[0], s1["adv_mean"])
    first = float(model.adv_baseline[0])
    _, s2 = model(inputs, targets)
    assert abs(float(model.adv_baseline[0]) - (0.5 * first + 0.5 * float(s2["adv_mean"]))) < 1e-5
    model.eval()
    before = float(model.adv_baseline[0])
    with torch.no_grad():
        model(inputs, targets)
    assert float(model.adv_baseline[0]) == before


def test_last_position_gets_no_policy_gradient():
    """The slot chosen at T-1 is never consumed: the policy term's gradient
    with respect to that position's slot logits must vanish (the model's own
    advantage/valid masking, reproduced here from its pieces)."""
    inputs = torch.randint(0, 64, (1, 6), dtype=torch.int32)
    targets = torch.randint(0, 64, (1, 6))
    model = tiny_model("policy", detach=False)
    model.train()
    x0 = model.norm1(model.embed(inputs))
    torch.manual_seed(5)
    top1 = model.trunk(x0, None)
    logits = model.slot_logits(top1)
    choice, logp, _ = sample_slot_choices(logits)
    sigma = head_choice_to_sigma(choice, model.slots)
    ce = model.forward_with_slots(inputs, targets, sigma)
    raw = discounted_future_credit((model.token_ce(top1, targets) - ce).detach(), 0.9, 32)
    valid = (torch.arange(6) < 5).float()[None]
    pg = -((raw - raw[:, :-1].mean()) * valid * logp).sum()
    g = torch.autograd.grad(pg, [logits], retain_graph=True)[0]
    assert torch.equal(g[:, -1], torch.zeros_like(g[:, -1]))
    assert g[:, :-1].abs().sum() > 0
    # and the model's forward agrees: with detach the slot-head gradient equals this term's
    model.zero_grad(set_to_none=True)
    model.detach = True
    torch.manual_seed(5)
    loss, stats = model(inputs, targets)
    assert torch.equal(stats["sigma"], sigma)
    head_grad = torch.autograd.grad(loss, [model.slot_head.weight])[0]
    ref_grad = torch.autograd.grad(-((raw - raw[:, :-1].mean()) * valid * logp).sum(), [model.slot_head.weight],
                                   retain_graph=True)[0]
    assert torch.allclose(head_grad, ref_grad, atol=1e-6)


def test_fifo_and_full_modes_have_no_head_and_finite_gradients():
    inputs = torch.randint(0, 64, (2, 12), dtype=torch.int32)
    targets = torch.randint(0, 64, (2, 12))
    for mode in ("fifo", "full"):
        model = tiny_model(mode)
        assert not hasattr(model, "slot_head")
        model.train()
        loss, stats = model(inputs, targets)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert float(stats["null_count"]) == 0
        model.eval()
        with torch.no_grad():
            _, ev = model(inputs, targets)
        assert {"ce_last", "ce_full", "age_num", "age_den"} <= set(ev)
        if mode == "full":
            assert torch.equal(ev["ce_last"], ev["ce_full"])
        else:
            assert float(ev["age_num"]) / float(ev["age_den"]) <= 4.0 + 1e-5


@pytest.mark.parametrize("mode,choice", [("policy", "inject"), ("fifo", "fifo")])
def test_sequential_matches_parallel_and_prefix_recompute(mode, choice):
    inputs = torch.randint(0, 64, (2, 14), dtype=torch.int32)
    targets = torch.randint(0, 64, (2, 14))
    model = tiny_model(mode, seed=4)
    model.eval()
    if choice == "inject":
        sigma = random_sigma(2, 14, 4, seed=9)
        out = model.sequential(inputs, targets, choice="inject", sigma=sigma)
    else:
        sigma = fifo_sigma(2, 14, 4, torch.device("cpu"))
        out = model.sequential(inputs, targets, choice="fifo")
    assert torch.equal(out["sigma"], sigma)
    with torch.no_grad():
        parallel = model.forward_with_slots(inputs, targets, sigma)
    assert torch.allclose(out["ce"], parallel, atol=1e-4, rtol=1e-4), (out["ce"] - parallel).abs().max()
    # prefix recompute: the last position of every prefix under the same slots
    with torch.no_grad():
        for L in (1, 2, 5, 9, 14):
            pre = model.forward_with_slots(inputs[:, :L], targets[:, :L], sigma[:, :L])
            assert torch.allclose(pre[:, -1], out["ce"][:, L - 1], atol=1e-4, rtol=1e-4), L


def test_sequential_sampled_choices_are_consistent_with_head():
    inputs = torch.randint(0, 64, (2, 10), dtype=torch.int32)
    targets = torch.randint(0, 64, (2, 10))
    model = tiny_model("policy", seed=6)
    model.eval()
    out = model.sequential(inputs, targets, choice="greedy")
    # replaying the greedy trajectory in parallel reproduces the losses
    with torch.no_grad():
        parallel = model.forward_with_slots(inputs, targets, out["sigma"])
    assert torch.allclose(out["ce"], parallel, atol=1e-4, rtol=1e-4)
    sampled = model.sequential(inputs, targets, choice="sample")
    assert sampled["ce"].shape == (2, 10)
    assert int(sampled["null_count"]) == int((sampled["sigma"] < 0).sum())


def test_policy_eval_stats_are_complete_and_bounded():
    inputs = torch.randint(0, 64, (2, 16), dtype=torch.int32)
    targets = torch.randint(0, 64, (2, 16))
    model = tiny_model("policy", seed=7)
    model.eval()
    with torch.no_grad():
        _, stats = model(inputs, targets)
    assert 0 <= float(stats["null_count"]) <= 32
    assert 0 <= float(stats["entropy_sum"]) <= 32 * math.log(5) + 1e-4
    assert float(stats["age_den"]) >= 0
    assert 0 <= float(stats["age_num"]) / max(float(stats["age_den"]), 1e-9) <= 15
    assert torch.isfinite(stats["ce_last"]) and torch.isfinite(stats["ce_full"])


def test_injected_sigma_reproduces_sampled_forward():
    inputs = torch.randint(0, 64, (2, 12), dtype=torch.int32)
    targets = torch.randint(0, 64, (2, 12))
    model = tiny_model("policy")
    model.train()
    torch.manual_seed(8)
    loss_a, stats_a = model(inputs, targets)
    model_b = tiny_model("policy")
    model_b.train()
    loss_b, stats_b = model_b(inputs, targets, sigma=stats_a["sigma"])
    assert torch.equal(stats_a["sigma"], stats_b["sigma"])
    assert torch.allclose(loss_a, loss_b, atol=1e-5, rtol=1e-6)
    assert torch.allclose(stats_a["ce_last"], stats_b["ce_last"])


def test_three_passes_run_and_train():
    inputs = torch.randint(0, 64, (2, 12), dtype=torch.int32)
    targets = torch.randint(0, 64, (2, 12))
    model = tiny_model("policy", passes=3)
    assert model.adv_baseline.shape == (2,)
    model.train()
    loss, stats = model(inputs, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters())
    assert bool(model.adv_baseline_ready)

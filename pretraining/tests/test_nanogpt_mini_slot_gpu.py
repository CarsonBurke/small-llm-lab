"""GPU tests for ``pretraining/nanogpt_mini/nanogpt_mini_slot_model.py``.

Submit through mlq (GPU workload). Checks, at the production geometry
(6L/512d, 4 heads x 128, M = 64):
  1. the compiled FlexAttention path over the table-derived block mask
     equals the dense reference in forward (per-token CE) and in every
     parameter gradient (fp32: tight; bf16: output only, flex accumulates
     dK/dV in bf16), and the flex age statistic equals the dense one;
  2. the eager sequential evaluator (per-layer slot banks) equals the
     parallel replay under the same injected slots on a short window, and a
     literal prefix recompute at a few prefix lengths;
  3. the compiled policy training graph equals the eager one on an injected
     slot trajectory (loss, telemetry, the EMA baseline buffer, every
     gradient);
  4. one real-shape compiled policy training step (MBS 64 x seq 1024, two
     passes) fits under 28 GB peak and prints the step time; the eval
     forward's statistics are finite and in range.
"""

import time

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_slot_model import (
    NO_WRITE,
    SlotGPT,
    attention_age_stats,
    make_slot_mask,
    slot_block_mask,
    build_alive_table,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")


def random_sigma(B, T, slots, seed=0, null_prob=0.2):
    g = torch.Generator(device="cpu").manual_seed(seed)
    sigma = torch.randint(0, slots, (B, T), generator=g)
    null = torch.rand(B, T, generator=g) < null_prob
    return torch.where(null, torch.full_like(sigma, NO_WRITE), sigma).to(DEV)


def real_model(backend, mode="policy", seed=0, **kw):
    torch.manual_seed(seed)
    model = SlotGPT(vocab_size=1024, num_layers=6, model_dim=512, slots=64, mode=mode, backend=backend, **kw)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("slot_head."):
                p.normal_(std=0.3)
            elif name.endswith("weight") and "proj" in name:
                p.normal_(std=0.02)
            elif name.endswith("weight") and "embed" in name:
                p.normal_()
            elif name.endswith("weight"):
                p.normal_(std=0.33**0.5 / p.size(-1) ** 0.5)
            elif name.endswith("gains"):
                p.fill_(1.0)
            else:
                p.zero_()
    return model.to(DEV)


def rel(a, b):
    return float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flex_matches_dense_forward_backward(dtype):
    B, T = 2, 1024
    dense = real_model("dense").to(dtype)
    flex = real_model("flex").to(dtype)
    flex.load_state_dict(dense.state_dict())
    inputs = torch.randint(0, 1024, (B, T), device=DEV, dtype=torch.int32)
    targets = torch.randint(0, 1024, (B, T), device=DEV)
    sigma = random_sigma(B, T, 64, seed=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        make_slot_mask(sigma, 64, "flex")
    torch.cuda.synchronize()
    print(f"block mask build (B={B}, T={T}, M=64): {(time.perf_counter() - t0) / 5 * 1000:.2f} ms")
    alive = build_alive_table(sigma, 64)
    bm = slot_block_mask(alive, sigma)
    print(f"block sparsity: {bm.sparsity():.1f}% of blocks skipped")
    ce_dense = dense.forward_with_slots(inputs, targets, sigma)
    ce_flex = flex.forward_with_slots(inputs, targets, sigma)
    err = float((ce_dense - ce_flex).abs().max())
    print(f"{dtype}: per-token CE max abs diff {err:.3e}, rel {rel(ce_dense, ce_flex):.3e}")
    if dtype == torch.float32:
        assert err < 2e-3, err
    else:
        assert err < 0.15 and rel(ce_dense, ce_flex) < 2e-2, err
    ce_dense.sum().backward()
    ce_flex.sum().backward()
    worst = 0.0
    for (name, p), (_, q) in zip(dense.named_parameters(), flex.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
            continue
        r = rel(p.grad, q.grad)
        worst = max(worst, r)
        if dtype == torch.float32:
            assert r < 5e-3, (name, r)
    print(f"{dtype}: worst parameter-gradient relative error {worst:.3e}")
    assert worst < (5e-3 if dtype == torch.float32 else 0.1), worst
    # age statistic: flex value-channel path vs explicit probabilities (slot mask and causal)
    torch.manual_seed(2)
    q = torch.randn(B, T, 4, 128, device=DEV)
    k = torch.randn(B, T, 4, 128, device=DEV)
    for label, dense_mask, flex_mask in (("slot", make_slot_mask(sigma, 64, "dense"), make_slot_mask(sigma, 64, "flex")),
                                         ("causal", None, None)):
        num_d, den_d = attention_age_stats(q, k, dense_mask, "dense")
        num_f, den_f = attention_age_stats(q, k, flex_mask, "flex")
        print(f"{label} age stat dense {float(num_d) / float(den_d):.4f} flex {float(num_f) / float(den_f):.4f}")
        assert abs(float(num_d) - float(num_f)) / float(num_d) < 1e-3
        assert abs(float(den_d) - float(den_f)) / float(den_d) < 1e-3


def test_sequential_matches_recompute_real_dims():
    B, T = 2, 128
    inputs = torch.randint(0, 1024, (B, T), device=DEV, dtype=torch.int32)
    targets = torch.randint(0, 1024, (B, T), device=DEV)
    sigma = random_sigma(B, T, 64, seed=3)
    for dtype, tol in ((torch.float32, 1e-3), (torch.bfloat16, 0.15)):
        model = real_model("flex").to(dtype).eval()
        seq = model.sequential(inputs, targets, choice="inject", sigma=sigma)
        assert torch.equal(seq["sigma"], sigma)
        with torch.no_grad():
            parallel = model.forward_with_slots(inputs, targets, sigma)
        err = float((seq["ce"] - parallel).abs().max())
        mean_err = float((seq["ce"] - parallel).abs().mean())
        print(f"{dtype}: sequential vs parallel per-token CE max abs {err:.3e} mean abs {mean_err:.3e}")
        assert err < tol, err
        with torch.no_grad():
            for L in (1, 7, 64, 128):
                pre = model.forward_with_slots(inputs[:, :L], targets[:, :L], sigma[:, :L])
                e = float((pre[:, -1] - seq["ce"][:, L - 1]).abs().max())
                assert e < tol, (L, e)
    # sampled sequential choices replay exactly in parallel (fp32)
    model = real_model("flex").float().eval()
    torch.manual_seed(4)
    seq = model.sequential(inputs, targets, choice="sample")
    with torch.no_grad():
        parallel = model.forward_with_slots(inputs, targets, seq["sigma"])
    assert float((seq["ce"] - parallel).abs().max()) < 1e-3


def test_compiled_policy_forward_matches_eager():
    """The compiled training graph (slot head, injected trajectory, discounted
    credit, baseline buffer mutation, dict return) against the eager forward
    on the same injected slots: loss, telemetry, every gradient (fp32)."""
    torch._dynamo.reset()
    B, T = 4, 1024
    inputs = torch.randint(0, 1024, (B, T), device=DEV, dtype=torch.int32)
    targets = torch.randint(0, 1024, (B, T), device=DEV)
    sigma = random_sigma(B, T, 64, seed=6)
    eager = real_model("flex", seed=1).float().train()
    compiled = real_model("flex", seed=1).float().train()
    compiled.compile(dynamic=False)
    decay = compiled.baseline_decay
    for step in range(2):
        if step == 1:  # a different microbatch so the EMA actually has to move
            torch.manual_seed(7)
            inputs = torch.randint(0, 1024, (B, T), device=DEV, dtype=torch.int32)
            targets = torch.randint(0, 1024, (B, T), device=DEV)
            sigma = random_sigma(B, T, 64, seed=8)
            b0 = float(compiled.adv_baseline[0])
        loss_e, st_e = eager(inputs, targets, sigma=sigma)
        loss_c, st_c = compiled(inputs, targets, sigma=sigma)
        assert bool(eager.adv_baseline_ready) and bool(compiled.adv_baseline_ready)
        assert abs(float(eager.adv_baseline[0]) - float(compiled.adv_baseline[0])) < 1e-4, step
        if step == 0:
            assert abs(float(compiled.adv_baseline[0]) - float(st_c["adv_mean"])) < 1e-5  # seeded by the first mean
        else:
            expected = decay * b0 + (1 - decay) * float(st_c["adv_mean"])
            assert abs(float(compiled.adv_baseline[0]) - expected) < 1e-5, (float(compiled.adv_baseline[0]), expected)
            assert abs(float(compiled.adv_baseline[0]) - float(st_c["adv_mean"])) > 1e-6
        for key in ("ce_last", "ce_full", "adv_mean", "adv_std", "pg_sum", "baseline", "entropy_sum", "null_count"):
            assert abs(float(st_e[key]) - float(st_c[key])) <= 1e-3 * max(1.0, abs(float(st_e[key]))), (key, step)
        print(f"compiled vs eager step {step}: loss {float(loss_e):.4f} / {float(loss_c):.4f} "
              f"baseline {float(eager.adv_baseline[0]):.5f} / {float(compiled.adv_baseline[0]):.5f}")
        assert abs(float(loss_e) - float(loss_c)) <= 1e-4 * abs(float(loss_e))
        loss_e.backward()
        loss_c.backward()
        worst = 0.0
        for (name, p), (_, q) in zip(eager.named_parameters(), compiled.named_parameters()):
            assert p.grad is not None and q.grad is not None, name
            worst = max(worst, rel(p.grad, q.grad))
            assert rel(p.grad, q.grad) < 2e-3, (name, rel(p.grad, q.grad))
        print(f"compiled vs eager step {step}: worst gradient relative error {worst:.3e}")
        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)


def test_training_step_real_shape_memory_and_time():
    torch._dynamo.reset()
    B, T = 64, 1024
    model = real_model("flex")
    with torch.no_grad():
        model.slot_head.weight.zero_()
        model.slot_head.bias.zero_()
    model.compile(dynamic=False)
    torch.manual_seed(5)
    inputs = torch.randint(0, 1024, (B, T), device=DEV, dtype=torch.int32)
    targets = torch.randint(0, 1024, (B, T), device=DEV)
    model.train()
    for _ in range(2):  # compile warmup
        loss, stats = model(inputs, targets)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    steps = 3
    for _ in range(steps):
        loss, stats = model(inputs, targets)
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t0) / steps * 1000
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"policy training microbatch (MBS {B} x {T}, 2 passes): {step_ms:.1f} ms, peak {peak:.2f} GiB")
    print("stats: " + " ".join(f"{k}={float(v):.4f}" for k, v in stats.items() if v.ndim == 0))
    assert torch.isfinite(loss)
    assert peak < 28.0, peak
    null_fraction = float(stats["null_count"]) / (B * T)
    assert 0.01 < null_fraction < 0.02, null_fraction  # uniform head: 1/65 = 0.0154
    assert bool(model.adv_baseline_ready)
    assert torch.isfinite(model.adv_baseline).all()
    model.eval()
    with torch.no_grad():
        _, ev = model(inputs, targets)
    age = float(ev["age_num"]) / float(ev["age_den"])
    print("eval stats: " + " ".join(f"{k}={float(v):.4f}" for k, v in ev.items() if v.ndim == 0) + f" age={age:.2f}")
    assert all(torch.isfinite(v).all() for v in ev.values())
    assert 0 < age < T
    assert float(ev["ce_last"]) > 0 and float(ev["ce_full"]) > 0

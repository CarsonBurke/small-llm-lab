"""GPU tests for ``pretraining/nanogpt_mini/nanogpt_mini_pgqa_model.py`` and
its fused kernels (``nanogpt_mini_pgqa_kernel.py``).

Submit through mlq (GPU workload).
  1. The fused Triton path must reproduce the PyTorch gather reference
     (itself pinned to a dense-scatter reference on CPU) at the production
     shape (T=1024, P=256, 4 heads x 128): outputs and every gradient (dq,
     dk, dv, d log_prior, and through the module all parameters) in fp32
     with IEEE dots, for ``free_bits`` 0 and 2 and a mixed pointer set that
     exercises coincident keys and all-invalid early queries. bf16 checks
     the output only.
  2. Whole-model ``torch.compile`` with the fused kernel inside (compile
     boundary) reproduces eager gradients in fp32.
  3. A bf16 microbatch of the train script's default ``MBS`` runs under
     compile with finite loss and gradients, and reports peak memory.
"""

import copy

import pytest
import torch

from pretraining.nanogpt_mini import nanogpt_mini_pgqa_model as m

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def init_params(model, bias):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("weight"):
                p.normal_(std=0.02 if "proj" in name else 0.05)
            elif name.endswith("ptr_embed"):
                p.normal_()
            elif name.endswith("ptr_mix"):
                p.normal_(std=1 / 64)
            elif name.endswith("ptr_bias"):
                p.fill_(bias)
            elif name.endswith("bias"):
                p.zero_()
    return model


def build_gpt(dtype, free_bits=2, backend="fused"):
    torch.manual_seed(0)
    model = m.PGQAGPT(1024, 6, 512, seq_len=1024, window=256, free_bits=free_bits, rank=64,
                      backend=backend)
    # -1.0: ~27% flip rate -> mixed pointer sets, not the bare window
    return init_params(model, -1.0).cuda().to(dtype)


def grads(model, x, y):
    model.train()
    loss = model(x, y)
    loss.backward()
    g = {n: p.grad.float().clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return float(loss.detach()), g


def rel(a, b):
    return float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12))


@pytest.mark.parametrize("free_bits", [0, 2])
def test_fused_matches_gather_fp32(free_bits):
    torch.manual_seed(0)
    attn = m.PointerGQAttention(512, 1024, 256, free_bits, 64, backend="gather")
    attn = init_params(attn, -1.0).cuda().float().train()
    fused = copy.deepcopy(attn)
    fused.backend = "fused"
    x = torch.randn(2, 1024, 512, device="cuda")
    torch.manual_seed(7)
    y_ref = attn(x)
    torch.manual_seed(7)
    y_fused = fused(x)
    assert torch.isfinite(y_fused).all()
    assert rel(y_ref, y_fused) < 1e-5
    weight = torch.randn_like(y_ref)
    (y_ref * weight).sum().backward()
    (y_fused * weight).sum().backward()
    for (name, pa), (_, pb) in zip(attn.named_parameters(), fused.named_parameters()):
        assert rel(pa.grad, pb.grad) < 1e-4, (name, rel(pa.grad, pb.grad))
    assert fused.controller.ctrl.weight.grad.abs().sum() > 0


def test_fused_kernel_grads_match_gather_directly():
    # kernel-level check of every input gradient, including d log_prior and
    # dk/dv through coincident keys and all-invalid early queries
    from pretraining.nanogpt_mini.nanogpt_mini_pgqa_kernel import fused_attend
    torch.manual_seed(1)
    B, T, H, D, W, k = 2, 1024, 4, 128, 256, 2
    q = torch.randn(B, T, H, D, device="cuda", requires_grad=True)
    kk = torch.randn(B, T, D, device="cuda", requires_grad=True)
    v = torch.randn(B, T, D, device="cuda", requires_grad=True)
    bits = torch.rand(B, T, W, m.num_region_bits(T, W) + k, device="cuda") < 0.3
    positions, valid = m.pointer_positions(bits, T, W, k)
    log_prior = torch.randn(B, T, W, device="cuda", requires_grad=True)
    out_ref = m.gqa_attend(q, kk, v, positions, valid, log_prior, 0.12)
    kv = torch.cat((kk, v), -1)
    pos = torch.where(valid, positions, -1).to(torch.int32)
    out = fused_attend(q, kv, pos, log_prior, 0.12)
    assert rel(out_ref, out) < 1e-5
    weight = torch.randn_like(out)
    g_ref = torch.autograd.grad((out_ref * weight).sum(), (q, kk, v, log_prior))
    g = torch.autograd.grad((out * weight).sum(), (q, kk, v, log_prior))
    for name, a, b in zip(("dq", "dk", "dv", "dlog_prior"), g_ref, g):
        assert rel(a, b) < 1e-4, (name, rel(a, b))


def test_fused_bf16_output_matches_gather():
    torch.manual_seed(0)
    attn = m.PointerGQAttention(512, 1024, 256, 2, 64, backend="gather")
    attn = init_params(attn, -1.0).cuda().bfloat16().eval()
    fused = copy.deepcopy(attn)
    fused.backend = "fused"
    x = torch.randn(2, 1024, 512, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(7)
    y_ref = attn(x)
    torch.manual_seed(7)
    y_fused = fused(x)
    assert torch.isfinite(y_fused).all()
    assert rel(y_ref, y_fused) < 2e-2  # reference rounds scores to bf16; kernel keeps fp32


def test_compiled_matches_eager_fp32(monkeypatch):
    # Fixed pseudo-random bits: a threshold on the logits would flip
    # near-zero bits between eager and compiled numerics. Same draw for
    # every layer.
    torch.manual_seed(1)
    fixed_bits = torch.rand(4, 1024, 256, 4, device="cuda") < 0.27
    monkeypatch.setattr(m.PointerGQAttention, "sample_bits", lambda self, logits: fixed_bits)
    base = build_gpt(torch.float32)
    x = torch.randint(0, 1024, (4, 1024), device="cuda")
    y = torch.randint(0, 1024, (4, 1024), device="cuda")
    eager = copy.deepcopy(base)
    l0, g0 = grads(eager, x, y)
    torch._dynamo.reset()
    compiled = copy.deepcopy(base)
    compiled.compile(dynamic=False)
    l1, g1 = grads(compiled, x, y)
    assert abs(l0 - l1) / l0 < 1e-5
    worst = sorted(((rel(g0[n], g1[n]), n) for n in g0), reverse=True)[:5]
    assert worst[0][0] < 1e-4, worst
    eager.eval()
    compiled.eval()
    with torch.no_grad():
        assert abs(float(eager(x, y)) - float(compiled(x, y))) / l0 < 1e-5


def test_bf16_train_microbatch_under_compile():
    model = build_gpt(torch.bfloat16)
    model.compile(dynamic=False)
    x = torch.randint(0, 1024, (16, 1024), device="cuda")
    y = torch.randint(0, 1024, (16, 1024), device="cuda")
    grads(model, x, y)  # warm-up / compile
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    loss, g = grads(model, x, y)
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    print(f"bf16 mbs=16 loss {loss / y.numel():.4f} peak {peak_gb:.1f} GiB fwd+bwd {ms:.0f} ms")
    assert loss == loss and all(torch.isfinite(v).all() for v in g.values())
    assert peak_gb < 24

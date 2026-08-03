"""
Micro-ablation tests for individual components.
Verifies correctness and measures per-component overhead.

Usage: python3 scripts/benchmark_components.py
"""
import time
import torch
import torch.nn.functional as F
from torch import Tensor

DEVICE = "cuda"
DTYPE = torch.bfloat16
WARMUP = 10
ITERS = 200

# Match baseline config
B, T, D, H, Hkv = 64, 1024, 512, 8, 4
HEAD_DIM = D // H


def cuda_timer(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ---- XSA ----

def xsa_rejection(y: Tensor, v: Tensor) -> Tensor:
    """XSA via flat vector rejection (our implementation)."""
    Hkv = v.size(1)
    if Hkv != y.size(1):
        v = v.repeat_interleave(y.size(1) // Hkv, dim=1)
    vn = F.normalize(v, dim=-1)
    dot = (y * vn).sum(dim=-1, keepdim=True)
    return y - dot * vn


def xsa_grouped_reshape(y: Tensor, v: Tensor) -> Tensor:
    """XSA via 5D grouped reshape (SOTA's implementation)."""
    B, H, T, D = y.shape
    Hkv = v.size(1)
    group = H // Hkv
    y_g = y.reshape(B, Hkv, group, T, D)
    vn = F.normalize(v, dim=-1).unsqueeze(2)
    proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
    return (y_g - proj).reshape(B, H, T, D)


def test_xsa():
    print("=" * 60)
    print("XSA (Exclusive Self Attention)")
    print("=" * 60)
    y = torch.randn(B, H, T, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, Hkv, T, HEAD_DIM, device=DEVICE, dtype=DTYPE)

    # Correctness: both should give the same result
    r1 = xsa_rejection(y, v)
    r2 = xsa_grouped_reshape(y, v)
    max_diff = (r1 - r2).abs().max().item()
    print(f"  Correctness (max diff): {max_diff:.2e} {'OK' if max_diff < 1e-2 else 'FAIL'}")

    # Verify orthogonality: output should be orthogonal to v
    v_expanded = v.repeat_interleave(H // Hkv, dim=1)
    vn = F.normalize(v_expanded, dim=-1)
    cos_sim = (F.normalize(r1, dim=-1) * vn).sum(dim=-1).abs().mean().item()
    print(f"  Orthogonality (mean |cos_sim| with v): {cos_sim:.6f} {'OK' if cos_sim < 0.01 else 'WARN'}")

    # Compile both
    c_reject = torch.compile(xsa_rejection, fullgraph=True)
    c_grouped = torch.compile(xsa_grouped_reshape, fullgraph=True)
    noop = torch.compile(lambda y, v: y, fullgraph=True)

    t_noop = cuda_timer(lambda: noop(y, v))
    t_reject = cuda_timer(lambda: c_reject(y, v))
    t_grouped = cuda_timer(lambda: c_grouped(y, v))
    print(f"  No-op:            {t_noop:.3f} ms")
    print(f"  Rejection (ours): {t_reject:.3f} ms  (+{t_reject - t_noop:.3f} ms)")
    print(f"  Grouped (SOTA):   {t_grouped:.3f} ms  (+{t_grouped - t_noop:.3f} ms)")
    print()


# ---- Peri-LN ----

def test_periln():
    print("=" * 60)
    print("Peri-LN (post-sublayer RMSNorm)")
    print("=" * 60)
    x = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE)

    def pre_only(x):
        h = F.rms_norm(x, (D,))
        return x + h  # fake sublayer

    def peri(x):
        h = F.rms_norm(x, (D,))
        h = F.rms_norm(h, (D,))  # post-norm
        return x + h

    c_pre = torch.compile(pre_only, fullgraph=True)
    c_peri = torch.compile(peri, fullgraph=True)

    t_pre = cuda_timer(lambda: c_pre(x))
    t_peri = cuda_timer(lambda: c_peri(x))
    print(f"  Pre-norm only: {t_pre:.3f} ms")
    print(f"  Peri-LN:       {t_peri:.3f} ms  (+{t_peri - t_pre:.3f} ms)")
    print(f"  Per-layer overhead: {t_peri - t_pre:.3f} ms x 9 layers x 2 (attn+mlp) = {(t_peri - t_pre) * 18:.3f} ms/step")
    print()


# ---- NorMuon ----

def test_normuon():
    print("=" * 60)
    print("NorMuon (post-NS per-row adaptive scaling)")
    print("=" * 60)
    from shared.normuon import normuon_post_ns

    # Typical matrix sizes in baseline
    sizes = [(512, 512), (512, 256), (1024, 512)]
    for fan_out, fan_in in sizes:
        g = torch.randn(fan_out, fan_in, device=DEVICE, dtype=DTYPE)
        sm = torch.ones(fan_out, 1, device=DEVICE, dtype=torch.float32) * 0.01

        # Check norm preservation
        g_orig_norm = g.norm().item()
        g_out = normuon_post_ns(g.clone(), sm.clone(), beta2=0.95)
        g_out_norm = g_out.norm().item()
        ratio = g_out_norm / g_orig_norm
        print(f"  [{fan_out}x{fan_in}] Norm preservation: {ratio:.4f} {'OK' if 0.95 < ratio < 1.05 else 'FAIL'}")

        def bench_normuon():
            normuon_post_ns(g, sm, 0.95)

        def bench_noop():
            return g

        t_noop = cuda_timer(bench_noop)
        t_nm = cuda_timer(bench_normuon)
        print(f"  [{fan_out}x{fan_in}] Overhead: {t_nm - t_noop:.3f} ms")
    print()


# ---- End-to-end step overhead estimate ----

def test_e2e_estimate():
    print("=" * 60)
    print("Estimated per-step overhead (9 layers, 8 grad_accum)")
    print("=" * 60)
    baseline_step_ms = 588.0

    # XSA: per attention layer
    y = torch.randn(B, H, T, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, Hkv, T, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    c_xsa = torch.compile(xsa_rejection, fullgraph=True)
    noop = torch.compile(lambda y, v: y, fullgraph=True)
    xsa_overhead = cuda_timer(lambda: c_xsa(y, v)) - cuda_timer(lambda: noop(y, v))

    # Peri-LN: 2 extra norms per layer
    x = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE)
    norm_overhead = cuda_timer(lambda: F.rms_norm(x, (D,)))

    # NorMuon: per matrix param (negligible vs forward/backward)
    from shared.normuon import normuon_post_ns
    g = torch.randn(512, 512, device=DEVICE, dtype=DTYPE)
    sm = torch.ones(512, 1, device=DEVICE, dtype=torch.float32) * 0.01
    nm_overhead = cuda_timer(lambda: normuon_post_ns(g, sm, 0.95))

    layers = 9
    grad_accum = 8
    matrices_per_layer = 6

    total_xsa = xsa_overhead * layers * grad_accum
    total_periln = norm_overhead * 2 * layers * grad_accum  # 2 post-norms per layer
    total_normuon = nm_overhead * matrices_per_layer * layers  # once per step, not per accum

    total = total_xsa + total_periln + total_normuon
    print(f"  XSA:     {total_xsa:.1f} ms ({total_xsa/baseline_step_ms*100:.1f}%)")
    print(f"  Peri-LN: {total_periln:.1f} ms ({total_periln/baseline_step_ms*100:.1f}%)")
    print(f"  NorMuon: {total_normuon:.1f} ms ({total_normuon/baseline_step_ms*100:.1f}%)")
    print(f"  Total:   {total:.1f} ms / {baseline_step_ms:.0f} ms = {total/baseline_step_ms*100:.1f}% overhead")
    print()


if __name__ == "__main__":
    test_xsa()
    test_periln()
    test_normuon()
    test_e2e_estimate()

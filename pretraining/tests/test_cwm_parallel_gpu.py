"""The chunk-parallel core on CUDA graphs with the compiled chunk and the
bank-product kernels (``parallel.GraphedParallelCore``,
``chunk_kernels``) against its autograd reference (``parallel.ParallelCore``
in PyTorch products), which ``test_cwm_parallel.py`` holds to the module
path. Queue through mlq."""

from __future__ import annotations

import pytest
import torch

from pretraining.cwm import chunk_kernels
from pretraining.cwm.chunk_kernels import (
    bank_accumulate,
    bank_accumulate_reference,
    bank_end,
    bank_read,
    store_attend,
    store_attend_reference,
    wy_read,
    wy_read_reference,
    wy_write,
    wy_write_reference,
)
from pretraining.cwm.parallel import GraphedParallelCore, ParallelCore, graphed_parallel_core
from pretraining.tests.test_cwm import VARIANTS, build, small_config
from pretraining.tests.test_cwm_parallel import store_inputs, wy_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Three chunks; the store window (4) is shorter than one. Chunk and bank
# dims of 16 take the kernels.
B, T, CHUNK, BANK_DIM = 4, 48, 16, 16
CASES = [("feedback", "full"), ("all", "placebo"), ("feedback", "lattice")]


@pytest.fixture(autouse=True)
def no_tf32():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.fixture(autouse=True)
def fresh_compile():
    """Each test's configs compile the chunk anew; a training run needs a
    handful of specializations, a test session many more than dynamo's
    per-function recompile limit."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def make(variant: dict, lag: str, mask: str):
    model = build(small_config(read_mask=mask, lag=lag, lag_chunk=CHUNK, bank_dim=BANK_DIM, **variant)).cuda()
    inputs = torch.randint(0, 97, (B, T), generator=torch.Generator().manual_seed(0)).cuda()
    probe = torch.randn(B, T, model.cfg.vocab_size, generator=torch.Generator().manual_seed(3)).cuda()
    return model, inputs, probe


def run(model, inputs, probe, core=None):
    """Core output and gradients (core parameters and its input ``u``).
    The perception runs without grad: only the core is under test, and the
    KDA perception's backward fails on a second use per process on GPU
    (a kernel-library registration error unrelated to the core)."""
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        u = model.perceive(inputs)
    u.requires_grad_()
    if core is None:
        # The reference runs the bank products in PyTorch.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(chunk_kernels, "_kernel_shapes", lambda C, D: False)
            h = ParallelCore(model)(u)
    else:
        h = graphed_parallel_core(core, u)
    (model.project_logits(model.norm2(h)) * probe).sum().backward()
    grads = {n: torch.zeros_like(p) if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}
    grads["u"] = u.grad
    return h.detach(), grads


def assert_grads_close(grads, expected, tol):
    for name, e in expected.items():
        if not e.any():
            assert not grads[name].any(), name
            continue
        error = (grads[name] - e).norm() / e.norm()
        assert error < tol, (name, error.item())


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("lag,mask", CASES)
def test_graphed_core_matches_autograd_core(variant, lag, mask):
    model, inputs, probe = make(variant, lag, mask)
    expected, expected_grads = run(model, inputs, probe)
    core = GraphedParallelCore(model, B, T, train=True)
    assert core.graphs
    # Twice: every carried and accumulated buffer is reset per backward.
    for _ in range(2):
        got, grads = run(model, inputs, probe, core)
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
        assert_grads_close(grads, expected_grads, 1e-4)
    with torch.no_grad():
        evaluation = graphed_parallel_core(GraphedParallelCore(model, B, T, train=False), model.perceive(inputs))
    torch.testing.assert_close(evaluation, expected, rtol=1e-4, atol=1e-4)


def test_graphed_core_follows_weight_updates():
    """The graphs read weights loaded per call, so an optimizer step is seen."""
    model, inputs, probe = make(VARIANTS[0].values[0], "feedback", "full")
    core = GraphedParallelCore(model, B, T, train=True)
    run(model, inputs, probe, core)
    with torch.no_grad():
        for p in model.rounds.parameters():
            p.add_(0.01 * torch.randn_like(p))
    expected, expected_grads = run(model, inputs, probe)
    got, grads = run(model, inputs, probe, core)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
    assert_grads_close(grads, expected_grads, 1e-4)


@pytest.mark.parametrize("op", ["read", "end"])
def test_bank_kernels_match_pytorch_products(op):
    g = torch.Generator(device="cuda").manual_seed(0)
    N, C, D = 6 * 3, 16, 64

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=g, dtype=torch.float64)

    m, vecs, other, e = randn(N, D, D), randn(N, C, D), randn(N, C, D), randn(N, D).exp() / 3
    if op == "read":
        # One query read from 3 sources.
        inputs = (vecs[: N // 3], m.view(3, N // 3, D, D))
        f, reference = bank_read, lambda q, m: q @ m.transpose(-1, -2)
    else:
        inputs = (m, e, vecs, other)
        f, reference = bank_end, lambda m, e, u, k: m * e[..., None, :] + u.transpose(-1, -2) @ k
    expected_inputs = [x.clone().requires_grad_() for x in inputs]
    expected = reference(*expected_inputs)
    d_out = randn(*expected.shape)
    expected.backward(d_out)
    got_inputs = [x.float().requires_grad_() for x in inputs]
    got = f(*got_inputs)
    got.backward(d_out.float())
    torch.testing.assert_close(got.double(), expected.detach(), rtol=1e-5, atol=1e-4)
    for x, y in zip(got_inputs, expected_inputs):
        torch.testing.assert_close(x.grad.double(), y.grad, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("C", [16, 32])
@pytest.mark.parametrize("at_bound", [False, True])
@pytest.mark.parametrize("op", ["write", "read"])
def test_wy_kernels_match_fp64_reference(op, at_bound, C):
    """The fp32 WY kernels against the fp64 autograd of their references,
    with decays up to the lower bound across the chunk (factored ratios
    to exp(+-5 C / 2))."""
    m, key, value, beta, log_decay, q = wy_inputs(3, 40, C, 64, torch.Generator().manual_seed(0), at_bound, "cuda")
    if op == "write":
        inputs = (m, key, value, beta, log_decay)
        f, reference = wy_write, wy_write_reference
    else:
        u = wy_write_reference(m, key, value, beta, log_decay)
        inputs = (q, m, key, log_decay, u)
        f, reference = wy_read, wy_read_reference
    expected_inputs = [x.clone().requires_grad_() for x in inputs]
    expected = reference(*expected_inputs)
    d_out = torch.randn_like(expected)
    expected.backward(d_out)
    got_inputs = [x.float().requires_grad_() for x in inputs]
    got = f(*got_inputs)
    got.backward(d_out.float())

    def rel(x, y):
        return ((x.double() - y).norm() / y.norm()).item()

    assert rel(got, expected.detach()) < 1e-5
    for x, y in zip(got_inputs, expected_inputs):
        assert x.grad.isfinite().all()
        assert rel(x.grad, y.grad) < 1e-4


@pytest.mark.parametrize("C", [16, 32])
@pytest.mark.parametrize("exact", [False, True])
def test_bank_accumulate_kernel_matches_fp64_reference(exact, C):
    """The deferred state gradient (fp32, in place) against its fp64
    reference, with decays at the lower bound on half the channels."""
    generator = torch.Generator().manual_seed(0)
    R, S, P, D = 5, 6, 40, 64
    _, key, dv, _, log_decay, _ = wy_inputs(S, P, C, D, generator, True, "cuda")

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64).cuda()

    q, q_fb, d_reads, dm = randn(R, P, C, D), randn(R, P, C, D), randn(R, S, P, C, D), randn(S, P, D, D)
    expected = dm + bank_accumulate_reference(q, q_fb, d_reads, key, log_decay, dv, exact)
    got = dm.float()
    bank_accumulate(got, *(t.float() for t in (q, q_fb, d_reads, key, log_decay, dv)), exact)
    assert ((got.double() - expected).norm() / expected.norm()).item() < 1e-5


@pytest.mark.parametrize("start", [0, 8, 512])
@pytest.mark.parametrize("reads", [(1, True, 0), (3, True, 2), (5, False, 0)])
def test_store_attend_kernel_matches_fp64_reference(start, reads):
    """The store kernel (bf16 entries, fp32 queries) against the fp64
    autograd of its reference, over a 6-source window: entry gradients to
    bf16 rounding."""
    legal, feedback, shift = reads
    inputs = store_inputs(32, 16, 4, 32, 64, 6, legal, "cuda")
    start = torch.tensor(start, device="cuda")
    d_out, d_fb = torch.randn_like(inputs[0]), torch.randn_like(inputs[0])
    got_inputs = [(t.float() if i == 0 else t.bfloat16()).requires_grad_() for i, t in enumerate(inputs)]
    got, got_fb, _, _ = store_attend(*got_inputs, start, 0.6, legal, feedback, shift)
    # The reference on the same (bf16-rounded) entries.
    reference_inputs = [t.detach().double().requires_grad_() for t in got_inputs]
    expected, expected_fb, _, _ = store_attend_reference(*reference_inputs, start, 0.6, legal, feedback, shift)
    if feedback:
        torch.autograd.backward([got, got_fb], [d_out.float(), d_fb.float()])
        torch.autograd.backward([expected, expected_fb], [d_out, d_fb])
    else:
        got.backward(d_out.float())
        expected.backward(d_out)

    def rel(x, y):
        return ((x.double() - y).norm() / y.norm().clamp_min(1e-30)).item()

    assert rel(got, expected.detach()) < 1e-5
    if feedback:
        assert rel(got_fb, expected_fb.detach()) < 1e-5
    for x, y in zip(got_inputs, reference_inputs):
        assert rel(x.grad, y.grad) < (1e-5 if x.dtype == torch.float32 else 1e-2)

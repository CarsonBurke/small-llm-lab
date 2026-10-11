"""The chunk-parallel executors (``pretraining/cwm/parallel.py``) against the
token-sequential module path (``CWM.core``), which ``test_cwm.py`` holds to
the literal event reference. ``GraphedParallelCore`` runs its graphed
procedures eagerly here; ``test_cwm_parallel_gpu.py`` covers the graphs."""

from __future__ import annotations

import pytest
import torch

from pretraining.cwm.chunk_kernels import store_attend, wy_read, wy_write
from pretraining.cwm.model import CWM
from pretraining.cwm.parallel import ChunkBank, GraphedParallelCore, ParallelCore, graphed_parallel_core, supports
from pretraining.tests.test_cwm import B, T, VARIANTS, build, small_config, tokens

MASKS = ["full", "lattice", "placebo"]
# Chunks shorter than, equal to and longer than the store window (4), and
# the whole sequence (12).
CHUNKS = [3, 4, 6, 12]


def make(mask: str, variant: dict, lag: str, chunk: int) -> CWM:
    return build(small_config(read_mask=mask, lag=lag, lag_chunk=chunk, **variant))


def outputs_and_grads(model: CWM, executor: str, inputs: torch.Tensor, probe: torch.Tensor):
    model.zero_grad(set_to_none=True)
    u = model.perceive(inputs)
    if executor == "module":
        h = model.core(u, 4, False)
    elif executor == "parallel":
        h = ParallelCore(model)(u)
    else:
        h = graphed_parallel_core(GraphedParallelCore(model, B, T, train=True), u)
    (model.project_logits(model.norm2(h)) * probe).sum().backward()
    grads = {n: torch.zeros_like(p) if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}
    return h.detach(), grads


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mask", MASKS)
@pytest.mark.parametrize("lag", ["feedback", "all"])
@pytest.mark.parametrize("chunk", CHUNKS)
@pytest.mark.parametrize("executor", ["parallel", "graphed"])
def test_parallel_core_matches_module_core(variant, mask, lag, chunk, executor):
    model = make(mask, variant, lag, chunk)
    inputs = tokens()
    probe = torch.randn(B, T, model.cfg.vocab_size, generator=torch.Generator().manual_seed(3))
    expected, expected_grads = outputs_and_grads(model, "module", inputs, probe)
    got, grads = outputs_and_grads(model, executor, inputs, probe)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    for name, e in expected_grads.items():
        if not e.any():
            assert not grads[name].any(), name
            continue
        error = (grads[name] - e).norm() / e.norm()
        assert error < 1e-4, (name, error.item())


@pytest.mark.parametrize("variant", VARIANTS)
def test_parallel_core_runs_the_exact_lattice(variant):
    model = make("lattice", variant, "none", 4)
    assert supports(model)
    with torch.no_grad():
        u = model.perceive(tokens())
        torch.testing.assert_close(ParallelCore(model)(u), model.core(u, 4, False), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("lag", ["feedback", "all"])
def test_parallel_core_follows_feedback_ablation(variant, lag):
    model = make("full", variant, lag, 3)
    core = ParallelCore(model)
    u = model.perceive(tokens()).detach()
    with torch.no_grad():
        model.feedback_enabled = False
        ablated, expected = core(u), model.core(u, 4, False)
        graphed_core = GraphedParallelCore(model, B, T, train=False)
        graphed = graphed_parallel_core(graphed_core, u)
        model.feedback_enabled = True
        full = core(u)
        # A core built for the ablation refuses the full model.
        with pytest.raises(RuntimeError):
            graphed_parallel_core(graphed_core, u)
    torch.testing.assert_close(ablated, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(graphed, expected, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(full, ablated)


def test_exact_feedback_is_not_chunk_parallel():
    model = make("full", VARIANTS[0].values[0], "none", 4)
    assert not supports(model)
    with pytest.raises(ValueError):
        ParallelCore(model)


def test_chunk_bank_matches_sequential_commits_at_long_decay_spans():
    """WY reads and end state equal token-by-token delta-rule commits with
    decays at the lower bound across a 32-token chunk (decay ratios to
    exp(-155)): the reference factoring keeps every factor finite."""
    g = torch.Generator().manual_seed(0)
    Bh, C, D = 6, 32, 16
    M0 = torch.randn(Bh, D, D, generator=g, dtype=torch.float64)
    key = torch.nn.functional.normalize(torch.randn(Bh, C, D, generator=g, dtype=torch.float64), dim=-1)
    value = torch.randn(Bh, C, D, generator=g, dtype=torch.float64)
    beta = 2 * torch.rand(Bh, C, generator=g, dtype=torch.float64)
    log_decay = -5 * torch.rand(Bh, C, D, generator=g, dtype=torch.float64)
    log_decay[:, :, : D // 2] = -5.0
    q = torch.randn(Bh, C, D, generator=g, dtype=torch.float64)
    M, reads = M0, []
    for j in range(C):
        reads.append(M @ q[:, j, :, None])
        M = M * log_decay[:, j, None, :].exp()
        M = M + beta[:, j, None, None] * (value[:, j, :, None] - M @ key[:, j, :, None]) * key[:, j, None, :]
    bank = ChunkBank(M0, key, value, beta, log_decay)
    stacked = ChunkBank.stack([bank], M0[None])
    torch.testing.assert_close(stacked.read(q)[0], torch.cat(reads, -1).transpose(-1, -2), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(bank.end(), M, rtol=1e-9, atol=1e-9)
    for f in ChunkBank.FIELDS:
        assert torch.isfinite(getattr(bank, f)).all(), f


def wy_inputs(s: int, P: int, C: int, D: int, generator: torch.Generator, at_bound: bool = False, device: str = "cpu"):
    """Shared inputs of the WY op tests: ``s`` sources of ``P`` tiles."""

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64).to(device)

    m = randn(s, P, D, D)
    key = torch.nn.functional.normalize(randn(s, P, C, D), dim=-1)
    value = randn(s, P, C, D)
    beta = 2 * torch.rand(s, P, C, generator=generator, dtype=torch.float64).to(device)
    log_decay = -5 * torch.rand(s, P, C, D, generator=generator, dtype=torch.float64).to(device)
    if at_bound:
        log_decay[..., : D // 2] = -5.0
    q = randn(P, C, D)
    return m, key, value, beta, log_decay, q


@pytest.mark.parametrize("C", [4, 5])
def test_wy_op_gradients_match_finite_differences(C):
    """The hand-derived backward of the WY ops (their PyTorch references off
    CUDA; the kernels hold the same derivation) against finite differences.
    C = 5 puts the reference row at an odd chunk's middle."""
    m, key, value, beta, log_decay, q = wy_inputs(2, 3, C, 6, torch.Generator().manual_seed(0))
    write_inputs = [t.requires_grad_() for t in (m, key, value, beta, log_decay)]
    assert torch.autograd.gradcheck(wy_write, write_inputs)
    u = wy_write(*write_inputs).detach().requires_grad_()
    assert torch.autograd.gradcheck(wy_read, (q.requires_grad_(), m, key, log_decay, u))


def store_inputs(B: int, C: int, H: int, Dh: int, W: int, S: int, chunk_sources: int, device: str = "cpu"):
    """Queries, a window of ``S`` sources (a source slice of a wider window,
    so strided) and the chunk's entries of ``chunk_sources`` sources."""
    g = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64).to(device)

    window = [randn(B, W, S + 1, H, Dh)[:, :, :S] for _ in range(2)]
    chunk = [randn(B, C, chunk_sources, H, Dh) for _ in range(2)]
    return 2 * randn(B, C, H, Dh), *window, *chunk


# (legal sources, feedback, placebo shift) of a 3-source window.
STORE_READS = [(1, True, 0), (2, True, 1), (2, False, 0)]


@pytest.mark.parametrize("start", [0, 2, 9])
@pytest.mark.parametrize("reads", STORE_READS)
def test_store_attend_gradients_match_finite_differences(start, reads):
    """The store op's explicit backward (its PyTorch reference off CUDA;
    the kernel holds the same derivation) against finite differences, for
    windows partly, fully and not yet filled (W = 4, C = 3), both reads."""
    legal, feedback, shift = reads
    inputs = [t.requires_grad_() for t in store_inputs(2, 3, 2, 4, 4, 3, legal)]
    start = torch.tensor(start)

    def f(*xs):
        out, fb_out, _, _ = store_attend(*xs, start, 0.7, legal, feedback, shift)
        return (out, fb_out) if feedback else out

    assert torch.autograd.gradcheck(f, inputs)

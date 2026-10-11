"""Contracts of the fused CWM executor (``pretraining/cwm/fused.py``).

The fused core must compute the module path's function (``CWM.core``) and
its exact gradients. In fp32 they agree to fp32 rounding; in bf16 the fused
core rounds less often than the module path, so it is held to the module
path's own distance from fp32.

Runs on CUDA through mlq, or on CPU under Triton's interpreter:

    TRITON_INTERPRET=1 .venv/bin/python -m pytest pretraining/tests/test_cwm_fused.py
"""

from __future__ import annotations

import os

import pytest
import torch

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET or not torch.cuda.is_available() else "cuda"
pytestmark = pytest.mark.skipif(DEVICE == "cpu" and not INTERPRET, reason="needs CUDA or TRITON_INTERPRET=1")

from pretraining.cwm.fused import Layout, prepare  # noqa: E402
from pretraining.cwm.graphed import GraphedCore, graphed_core  # noqa: E402
from pretraining.cwm.model import CWM, feedback_parameter_names, initialize  # noqa: E402
from pretraining.tests.test_cwm import B, T, VARIANTS, build, small_config  # noqa: E402

CHUNK = 4
MASKS = ["full", "lattice", "placebo"]
# Bank and store heads of different counts and widths (they share launches),
# and a store window that is not a power of two (masked entry slots).
UNEQUAL = dict(perception="attention", store=True, bank_heads=2, bank_dim=8, store_heads=3, store_dim=16, store_window=3)
SHAPES = [*(pytest.param(v.values[0], CHUNK, id=v.id) for v in VARIANTS), pytest.param(UNEQUAL, 3, id="unequal-odd")]
BF16_BANK = dict(perception="attention", store=True, bank_dtype="bfloat16")


def make(mask: str, variant: dict, wake: bool = True, **overrides) -> CWM:
    return build(small_config(read_mask=mask, **variant, **overrides), wake=wake).to(DEVICE)


def perturb_vectors(model: CWM) -> CWM:
    """Move every vector parameter (biases, norm gains, decay parameters,
    tags) off its initialization, which ``build`` leaves in place."""
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 or name.endswith(("store_tag", "depth_tag")):
                p.add_(0.1 * torch.randn(p.shape, generator=generator).to(p.device, p.dtype))
    return model


def core_input(model: CWM, seed: int = 11) -> torch.Tensor:
    """A fixed perception output: the core is under test, not perception
    (whose KDA variant runs fla kernels with their own numerics on CUDA)."""
    u = torch.randn(B, T, model.cfg.dim, generator=torch.Generator().manual_seed(seed))
    return u.to(DEVICE, model.embed.weight.dtype)


def run(model: CWM, fused: bool, probe: torch.Tensor | None = None, chunk: int = CHUNK, seed: int = 11):
    """Core output and (with ``probe``) the gradients of ``<out, probe>``
    for every parameter and (as ``"u"``) the core input."""
    model.zero_grad(set_to_none=True)
    u = core_input(model, seed).requires_grad_(probe is not None)
    with torch.set_grad_enabled(probe is not None):
        if fused:
            out = graphed_core(GraphedCore(model, B, T, chunk, train=probe is not None), u)
        else:
            out = model.core(u, chunk, checkpointing=False)
    if probe is None:
        return out.detach(), None
    (out.float() * probe).sum().backward()
    grads = {n: torch.zeros_like(p) if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}
    grads["u"] = u.grad.clone()
    return out.detach(), grads


def probe_for(model: CWM) -> torch.Tensor:
    return torch.randn(B, T, model.cfg.dim, generator=torch.Generator().manual_seed(7)).to(DEVICE)


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def test_layout_is_consistent():
    for variant in VARIANTS:
        lay = Layout(small_config(**variant.values[0]))
        assert lay.lr % 8 == 0 and lay.nx % 8 == 0 and lay.nx >= lay.lr + 2 * lay.cfg.low_rank
        assert lay.entry(0) == (0, lay.hd * (1 + 2 * lay.cfg.rounds))
        for j in range(1, lay.cfg.rounds):
            offset, width = lay.entry(j - 1)
            assert lay.entry(j)[0] == offset + width


@pytest.mark.parametrize("variant,chunk", SHAPES)
@pytest.mark.parametrize("mask", MASKS)
def test_fused_core_matches_module_core_fp32(variant, chunk, mask):
    model = perturb_vectors(make(mask, variant))
    probe = probe_for(model)
    expected, expected_grads = run(model, fused=False, probe=probe)
    got, grads = run(model, fused=True, probe=probe, chunk=chunk)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    frozen = set(feedback_parameter_names(model)) if mask == "lattice" else set()
    for name, e in expected_grads.items():
        if name in frozen or not e.any():
            assert not grads[name].any(), name
            continue
        assert relative(grads[name], e) < 1e-4, (name, relative(grads[name], e))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("chunk", [3, T])
def test_fused_core_independent_of_chunking(variant, chunk):
    model = make("full", variant)
    probe = probe_for(model)
    a, ga = run(model, fused=True, probe=probe, chunk=CHUNK)
    b, gb = run(model, fused=True, probe=probe, chunk=chunk)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    for name in ga:
        if ga[name].any():
            assert relative(gb[name], ga[name]) < 1e-5, name


REBUILT = ("U", "Y", "E", "OD", "HA", "HB", "HC", "MN", "XN", "P", "Z", "RO", "MIXED", "STATS", "A")


@pytest.mark.parametrize("variant,chunk", [*SHAPES, pytest.param(BF16_BANK, CHUNK, id="bf16-bank")])
@pytest.mark.parametrize("dtype", [torch.float32] + ([torch.bfloat16] if DEVICE == "cuda" else []))
def test_backward_rebuilds_the_forward_bitwise(variant, chunk, dtype):
    """The backward sweep (chunks in reverse, after the whole forward) sees
    each chunk's activations and bank states exactly as the forward made
    them, so its gradients are those of the computed forward."""
    model = make("placebo", variant)
    model.embed.to(dtype)
    graphed = GraphedCore(model, B, T, chunk, train=True, graphs=False)
    core = graphed.core
    names = [n for n in REBUILT if hasattr(core, n)]
    seen: dict[int, dict[str, torch.Tensor]] = {}
    exits: dict[int, torch.Tensor] = {}
    forward_chunk, rebuild = core.forward_chunk, core._rebuild

    def chunk_index() -> int:
        return int(core.pos0.item()) // chunk

    def recording_forward():
        forward_chunk()
        seen[chunk_index()] = {n: getattr(core, n).clone() for n in names}
        exits[chunk_index()] = core.M[0].clone()

    def checking_rebuild():
        rebuild()
        c = chunk_index()
        for n in names:
            assert torch.equal(getattr(core, n), seen[c][n]), (c, n)
        assert torch.equal(core.M[chunk], exits[c]), c

    core.forward_chunk, core._rebuild = recording_forward, checking_rebuild
    u = core_input(model)
    core.load(prepare(model))
    out = graphed.forward(u, save=True)
    graphed.backward(torch.randn_like(out), u)
    assert len(seen) == T // chunk


@pytest.mark.skipif(DEVICE != "cuda", reason="Triton's interpreter truncates fp32 -> bf16 casts; the GPU rounds to nearest even")
@pytest.mark.parametrize("mask", MASKS)
def test_fused_core_matches_module_core_with_bf16_banks(mask):
    """Both paths compute every commit in fp32 and round it once to bf16, so
    they agree except where their fp32 values (equal to fp32 rounding)
    straddle a bf16 rounding boundary: a rare one-ulp flip of a state entry,
    far below what the bf16 state itself moves the function by."""
    model = perturb_vectors(make(mask, BF16_BANK))
    exact = perturb_vectors(make(mask, {**BF16_BANK, "bank_dtype": "float32"}))
    probe = probe_for(model)
    truth, _ = run(exact, fused=False, probe=probe)
    expected, expected_grads = run(model, fused=False, probe=probe)
    got, grads = run(model, fused=True, probe=probe)
    fused_error, bf16_effect = relative(got, expected), relative(expected, truth)
    assert fused_error < bf16_effect / 4, (fused_error, bf16_effect)
    frozen = set(feedback_parameter_names(model)) if mask == "lattice" else set()
    for name, e in expected_grads.items():
        if name in frozen or not e.any():
            assert not grads[name].any(), name
            continue
        assert relative(grads[name], e) < 5e-2, (name, relative(grads[name], e))


@pytest.mark.parametrize("variant", VARIANTS)
def test_fused_evaluation_matches_training_forward(variant):
    model = make("placebo", variant)
    with torch.no_grad():
        evaluated, _ = run(model, fused=True)
    trained, _ = run(model, fused=True, probe=probe_for(model))
    assert torch.equal(evaluated, trained)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mask", ["full", "placebo"])
def test_fused_feedback_ablation_equals_lattice(variant, mask):
    full = make(mask, variant)
    lattice = make("lattice", variant)
    lattice.load_state_dict(full.state_dict())
    full.feedback_enabled = False
    with torch.no_grad():
        assert torch.equal(run(full, fused=True)[0], run(lattice, fused=True)[0])


@pytest.mark.parametrize("variant", VARIANTS)
def test_fused_feedback_ablation_drops_trained_feedback_biases(variant):
    model = make("full", variant)
    with torch.no_grad():
        for blk in model.rounds:
            blk.bank_o_fb.bias.normal_(0, 0.1)
            if model.cfg.store:
                blk.store_o_fb.bias.normal_(0, 0.1)
    model.feedback_enabled = False
    with torch.no_grad():
        expected, _ = run(model, fused=False)
        got, _ = run(model, fused=True)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fused_masks_are_bitwise_identical_at_init(variant):
    models = {mask: make(mask, variant, wake=False) for mask in MASKS}
    probe = probe_for(models["full"])
    results = {mask: run(model, fused=True, probe=probe) for mask, model in models.items()}
    fb = set(feedback_parameter_names(models["full"]))
    out_full, grads_full = results["full"]
    for other in MASKS[1:]:
        out, grads = results[other]
        assert torch.equal(out, out_full), other
        for name, g in grads_full.items():
            if (other == "lattice" and name in fb) or (other == "placebo" and name.endswith("_o_fb.weight")):
                continue
            assert torch.equal(grads[name], g), (other, name)


@pytest.mark.skipif(DEVICE != "cuda", reason="bf16 numerics are checked on CUDA")
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mask", MASKS)
def test_fused_core_bf16_is_as_close_to_fp32_as_module_core(variant, mask):
    # bf16 error of one model and input is dominated by how chaotic that
    # trajectory is (either path can be 3x the other's on one seed), and
    # per-tensor error is noisy (small bias and decay vectors suffer
    # cancellation), so the paths are compared by log-mean error over seeds
    # and parameters. Exactness is the fp32 test's job.
    out_err, grad_err = {"fused": [], "module": []}, {"fused": [], "module": []}
    for seed in range(8):
        model = build(small_config(read_mask=mask, **variant), seed=seed).to(DEVICE)
        probe = probe_for(model)
        truth, truth_grads = run(model, fused=False, probe=probe, seed=100 + seed)
        model.embed.bfloat16()  # bf16 activations, fp32 parameters: the training setup
        names = [n for n, g in truth_grads.items() if g.any()]
        for path in ("module", "fused"):
            out, grads = run(model, fused=path == "fused", probe=probe, seed=100 + seed)
            out_err[path].append(relative(out, truth))
            grad_err[path] += [relative(grads[n], truth_grads[n]) for n in names]

    def log_mean(xs: list[float]) -> float:
        return torch.tensor(xs).log().mean().item()

    assert log_mean(out_err["fused"]) <= log_mean(out_err["module"]) + 0.1
    assert log_mean(grad_err["fused"]) <= log_mean(grad_err["module"]) + 0.1


def test_prepare_is_differentiable_in_every_core_parameter():
    model = make("full", VARIANTS[0].values[0])
    prepared = prepare(model)
    total = sum(t.float().square().sum() for t in prepared.values() if t.requires_grad)
    total.backward()
    core = [n for n, _ in model.named_parameters() if n.startswith(("rounds.", "out_norm.", "out_writer."))]
    missing = [n for n in core if dict(model.named_parameters())[n].grad is None]
    assert not missing, missing


def test_initialize_reaches_every_parameter():
    model = CWM(small_config())
    initialize(model, 0)
    assert all(torch.isfinite(p).all() for p in model.parameters())

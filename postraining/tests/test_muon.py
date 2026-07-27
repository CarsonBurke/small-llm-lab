"""Muon optimizer math and the trunk-optimizer routing in build_optimizers."""

from __future__ import annotations

import copy
import os
import pickle

import pytest
import torch

from postraining import muon as muon_module
from postraining.latent_thought import LatentThoughtModel
from postraining.muon import MUON_ALGORITHM_SCHEMA, Muon, _polar_express
from postraining.nano_backbone import NanoGPTBackbone
from postraining.train_latent_vapo import (
    build_optimizers,
    renderer_parameters,
    step_optimizers,
    zero_optimizers,
)
from postraining.vapo.schemas import optimizer_schema_for_trunk_optimizer
from postraining.value_model import SeparateCritic

KWARGS = dict(vocab_size=64, num_layers=2, model_dim=256)


def _wrapper(seed: int = 3) -> LatentThoughtModel:
    torch.manual_seed(seed)
    wrapper = LatentThoughtModel(NanoGPTBackbone(**KWARGS).float().eval())
    for parameter in wrapper.parameters():
        parameter.requires_grad_(True)
    return wrapper


def _critic(wrapper: LatentThoughtModel, seed: int = 11) -> SeparateCritic:
    torch.manual_seed(seed)
    trunk = NanoGPTBackbone(**KWARGS).float()
    return SeparateCritic(trunk, num_bins=17, sigma_ratio=2.0).eval()


def _newtonschulz12(G: torch.Tensor) -> torch.Tensor:
    """The naive NewtonSchulz5 this module used to run, kept for comparison."""
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _orthogonalized(matrix: torch.Tensor) -> torch.Tensor:
    """Polar Express alone, with momentum wound out of the way."""
    return _polar_express(
        matrix.clone(),
        torch.zeros_like(matrix),
        torch.tensor(0.0),
    ).float()


def test_optimizer_schema_identifies_polar_express_and_adamw():
    schema = optimizer_schema_for_trunk_optimizer("muon")
    assert MUON_ALGORITHM_SCHEMA in schema
    assert "torch_adamw" in schema
    assert "polar_express5" in schema


def test_polar_express_trades_conditioning_for_iterations():
    """Five Polar Express iterations do not orthogonalize as tightly as twelve
    naive ones — they cost 15 matmuls instead of 36.  Pin both spectra so the
    trade stays visible if either side is ever changed.
    """
    # A realistic trunk gradient: low effective rank plus noise.
    generator = torch.Generator().manual_seed(0)
    left = torch.randn(512, 32, generator=generator)
    right = torch.randn(32, 512, generator=generator)
    gradient = left @ right + 0.05 * torch.randn(512, 512, generator=generator)

    polar = torch.linalg.svdvals(_orthogonalized(gradient))
    naive = torch.linalg.svdvals(_newtonschulz12(gradient).float())
    # Both aim at the polar factor, whose singular values are all 1.  Polar
    # Express keeps a designed ripple band instead of converging onto it.
    assert (naive - 1).abs().max() < (polar - 1).abs().max()
    assert naive.median() > polar.median()
    # The ripple is bounded, and the step it produces is roughly half the size.
    assert polar.max() < 1.2
    step_ratio = (
        _orthogonalized(gradient).norm() / _newtonschulz12(gradient).float().norm()
    )
    assert 0.4 < step_ratio < 0.7


def test_polar_express_covers_both_orientations():
    generator = torch.Generator().manual_seed(1)
    tall = torch.randn(64, 48, generator=generator)
    # The tall and wide branches multiply on opposite sides; a matrix and its
    # transpose have to come back as transposes of each other.
    assert torch.allclose(
        _orthogonalized(tall), _orthogonalized(tall.T).T, atol=2e-2, rtol=0
    )
    singular_values = torch.linalg.svdvals(_orthogonalized(tall))
    # The design ripple for five iterations, not convergence onto 1.
    assert singular_values.max() < 1.2
    assert singular_values.min() > 0.8


def test_batched_matmuls_are_the_only_bit_exact_entry_point():
    """``mm`` and ``bmm`` disagree, so ``step`` never mixes them.

    One differing rounding in the Gram matmul is amplified through five
    iterations, so this is not a difference the optimizer can absorb.

    CPU EVIDENCE ONLY, and the distinction matters. Line 133 below --
    a batch of one reproducing the whole group exactly -- holds here
    because ATen's CPU ``bmm`` is a loop over the batch. It does NOT hold
    on CUDA, where cuBLAS picks its strided-batched GEMM by batch count;
    see ``test_cuda_kernels_match_the_per_tensor_reference``. Do not read
    this test as licensing a batch-invariance assumption anywhere.

    What survives on both devices is the ``mm``/``bmm`` split asserted at
    the end, which is the reason a partial shape group steps row by row as
    a batch of one rather than falling back to the 2-D path.
    """
    generator = torch.Generator().manual_seed(2)
    stacks = [torch.randn(4, 16, 4, generator=generator) for _ in range(24)]

    def one_at_a_time(stack: torch.Tensor, batched: bool) -> torch.Tensor:
        outputs = []
        for matrix in stack:
            source = matrix.clone()[None] if batched else matrix.clone()
            output = _polar_express(
                source, torch.zeros_like(source), torch.tensor(0.95)
            )
            outputs.append(output[0] if batched else output)
        return torch.stack(outputs)

    disagreements = 0
    for stack in stacks:
        together = _polar_express(
            stack.clone(), torch.zeros_like(stack), torch.tensor(0.95)
        )
        # A batch of one reproduces the whole group exactly.
        assert torch.equal(together, one_at_a_time(stack, batched=True))
        flat = one_at_a_time(stack, batched=False)
        disagreements += not torch.equal(together, flat)
        # One rounding, amplified through five iterations into a few bf16 ULP.
        assert torch.allclose(together.float(), flat.float(), atol=5e-2, rtol=0)
    assert disagreements > 0, "expected the 2-D path to differ somewhere"


def test_muon_step_updates_matrices_and_skips_missing_grads():
    torch.manual_seed(1)
    stepped = torch.nn.Parameter(torch.randn(8, 8))
    gradless = torch.nn.Parameter(torch.randn(8, 8))
    stepped.grad = torch.randn(8, 8)
    before_stepped = stepped.detach().clone()
    before_gradless = gradless.detach().clone()
    optimizer = Muon([stepped, gradless], lr=1e-2)
    optimizer.step()
    assert not torch.equal(stepped.detach(), before_stepped)
    assert torch.equal(gradless.detach(), before_gradless)
    assert "momentum" in optimizer.state[stepped]
    assert gradless not in optimizer.state

    # The reference drops Muon's rectangular scale, so a matrix and its
    # transpose now take the same size step instead of differing by
    # sqrt(rows/cols).
    tall = torch.nn.Parameter(torch.zeros(16, 4))
    wide = torch.nn.Parameter(torch.zeros(4, 16))
    tall.grad = torch.ones(16, 4)
    wide.grad = torch.ones(4, 16)
    Muon([tall, wide], lr=1.0).step()
    ratio = tall.detach().norm() / wide.detach().norm()
    assert abs(ratio - 1.0) < 1e-6


class _PerMatrixMuon(Muon):
    """One matrix at a time, kept as the bit-exactness reference.

    Batching may only change how kernels are launched, so every batched result
    is compared against this loop rather than against another batched run.
    Each matrix still enters as a batch of one: ``mm`` and ``bmm`` disagree,
    so the 2-D entry point would compare two BLAS paths rather than two batch
    sizes.
    """

    @torch.no_grad()
    def step(self):
        momentum_t = torch.tensor(0.0)
        for group in self.param_groups:
            momentum_t.fill_(group["mu"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                update = muon_module.polar_express(
                    p.grad.to(torch.float32, copy=True)[None],
                    state["momentum"][None],
                    momentum_t,
                    p.size(-2) > 1024,
                )[0].float()
                if group["weight_decay"]:
                    decay = group["weight_decay"] * group["lr"]
                    p.sub_(p * (update * p >= 0), alpha=decay)
                p.add_(update, alpha=-group["lr"])


def _matrices(shapes, seed: int) -> list[torch.nn.Parameter]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(shape, generator=generator))
        for shape in shapes
    ]


def _assign_grads(parameters, seed: int, skip: frozenset[int] = frozenset()):
    generator = torch.Generator().manual_seed(seed)
    for index, parameter in enumerate(parameters):
        drawn = torch.randn(parameter.shape, generator=generator)
        parameter.grad = None if index in skip else drawn


def _assert_identical(batched, reference, batched_optimizer, reference_optimizer):
    for left, right in zip(batched, reference, strict=True):
        assert torch.equal(left.detach(), right.detach())
        left_state = batched_optimizer.state.get(left, {})
        right_state = reference_optimizer.state.get(right, {})
        assert set(left_state) == set(right_state)
        if left_state:
            assert torch.equal(left_state["momentum"], right_state["momentum"])
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert torch.equal(left.grad, right.grad)


# Two members per square group, plus tall and wide groups, so the batched path
# covers both NewtonSchulz5 branches and a singleton group.
_MIXED_SHAPES = [(8, 8), (16, 4), (8, 8), (4, 16), (16, 4), (12, 12)]
# Membership changes between steps, so the staging buffers have to survive a
# shrinking shape group and momentum has to carry across identically.
_SKIPS = [frozenset(), frozenset({0, 4}), frozenset({3})]


def test_batched_step_is_bit_identical_to_the_per_matrix_loop(monkeypatch):
    # Pinned to the eager kernel: torch.compile schedules 2-D and 3-D shapes
    # differently, so only the eager path isolates what batching itself
    # changes, which must be nothing.
    monkeypatch.setattr(muon_module, "polar_express", _polar_express)
    batched = _matrices(_MIXED_SHAPES, seed=5)
    reference = [torch.nn.Parameter(p.detach().clone()) for p in batched]
    batched_optimizer = Muon(batched, lr=3e-2, weight_decay=1e-2)
    reference_optimizer = _PerMatrixMuon(reference, lr=3e-2, weight_decay=1e-2)

    for step, skip in enumerate(_SKIPS):
        _assign_grads(batched, seed=100 + step, skip=skip)
        _assign_grads(reference, seed=100 + step, skip=skip)
        batched_optimizer.step()
        reference_optimizer.step()
        _assert_identical(batched, reference, batched_optimizer, reference_optimizer)


def test_batched_step_matches_the_per_matrix_loop_under_the_shipped_kernel():
    """The kernel the trainer actually calls, compiled or not.

    torch.compile picks a different schedule per input rank, and twelve
    NewtonSchulz5 iterations amplify that into a few bf16 ULP, so this pins the
    deviation at bf16 resolution rather than claiming bit-exactness the
    compiler does not give.
    """
    batched = _matrices(_MIXED_SHAPES, seed=5)
    reference = [torch.nn.Parameter(p.detach().clone()) for p in batched]
    batched_optimizer = Muon(batched, lr=3e-2, weight_decay=1e-2)
    reference_optimizer = _PerMatrixMuon(reference, lr=3e-2, weight_decay=1e-2)

    for step, skip in enumerate(_SKIPS):
        _assign_grads(batched, seed=100 + step, skip=skip)
        _assign_grads(reference, seed=100 + step, skip=skip)
        batched_optimizer.step()
        reference_optimizer.step()
    if muon_module.polar_express is _polar_express:
        _assert_identical(batched, reference, batched_optimizer, reference_optimizer)
        return
    for left, right in zip(batched, reference, strict=True):
        # lr is 3e-2, so a few bf16 ULP on a near-orthogonal update lands here.
        assert torch.allclose(left.detach(), right.detach(), atol=1e-3, rtol=0)


def test_batched_step_skips_grad_free_members_of_a_populated_shape_group():
    parameters = _matrices([(8, 8)] * 3, seed=7)
    before = [p.detach().clone() for p in parameters]
    _assign_grads(parameters, seed=8, skip=frozenset({1}))
    optimizer = Muon(parameters, lr=1e-2)
    optimizer.step()

    assert not torch.equal(parameters[0].detach(), before[0])
    assert torch.equal(parameters[1].detach(), before[1])
    assert not torch.equal(parameters[2].detach(), before[2])
    # A skipped member must not gain momentum state either.
    assert parameters[1] not in optimizer.state


def test_batched_step_writes_momentum_in_place_and_leaves_grad_alone():
    parameters = _matrices([(8, 8)] * 2, seed=9)
    _assign_grads(parameters, seed=10)
    grads = [p.grad for p in parameters]
    raw_grads = [g.clone() for g in grads]
    optimizer = Muon(parameters, lr=1e-2)
    optimizer.step()

    for parameter, grad, raw in zip(parameters, grads, raw_grads, strict=True):
        # Polar Express consumes a staged copy, so the caller's grad survives
        # the step untouched — the reference does not write through it either.
        assert parameter.grad is grad
        assert torch.equal(grad, raw)
        assert torch.allclose(
            optimizer.state[parameter]["momentum"], raw * (1 - 0.95)
        )


def test_momentum_rows_are_views_of_one_buffer_per_shape_group():
    parameters = _matrices([(8, 8)] * 3 + [(4, 16)], seed=12)
    _assign_grads(parameters, seed=13)
    optimizer = Muon(parameters, lr=1e-2)
    optimizer.step()
    square = [optimizer.state[p]["momentum"] for p in parameters[:3]]
    assert len({m.untyped_storage().data_ptr() for m in square}) == 1
    assert (
        optimizer.state[parameters[3]]["momentum"].untyped_storage().data_ptr()
        not in {m.untyped_storage().data_ptr() for m in square}
    )

    # A restored checkpoint hands back independent tensors; the next step has
    # to fold them into the buffer instead of stepping a detached copy.
    saved = optimizer.state_dict()
    clones = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    restored = Muon(clones, lr=1e-2)
    restored.load_state_dict(saved)
    _assign_grads(clones, seed=14)
    _assign_grads(parameters, seed=14)
    restored.step()
    optimizer.step()
    for original, clone in zip(parameters, clones, strict=True):
        assert torch.equal(original.detach(), clone.detach())
        assert torch.equal(
            optimizer.state[original]["momentum"], restored.state[clone]["momentum"]
        )


def test_two_param_groups_of_one_shape_keep_separate_momentum():
    """The momentum key has to carry the group, not just the shape.

    ``self._momenta`` outlives any single group, so keyed on shape alone the
    second group of (8, 8) matrices is handed the FIRST group's buffer.

    The member counts decide how that fails, and the dangerous case is the
    tidy one.  Differing counts raise -- a shorter second group trips
    ``lerp_`` on mismatched batch sizes, a longer one indexes off the end.
    EQUAL counts, which is what a symmetric actor/critic split produces,
    raise nothing at all: the two groups simply share momentum rows and
    feed each other's history into the wrong parameters.  So this test uses
    equal counts.

    Every Muon in the trainer today is built from a flat list, which is one
    group, so this never fired in production.
    """
    first = _matrices([(8, 8)] * 3, seed=71)
    second = _matrices([(8, 8)] * 3, seed=72)
    grouped = Muon([{"params": first}, {"params": second}], lr=1e-2)

    # Same parameters, same order, but each group alone in its own optimizer.
    apart_first = [torch.nn.Parameter(p.detach().clone()) for p in first]
    apart_second = [torch.nn.Parameter(p.detach().clone()) for p in second]
    solo = [Muon(apart_first, lr=1e-2), Muon(apart_second, lr=1e-2)]

    for step in range(3):
        for source, mirror in ((first, apart_first), (second, apart_second)):
            _assign_grads(source, seed=80 + step)
            _assign_grads(mirror, seed=80 + step)
        grouped.step()
        for optimizer in solo:
            optimizer.step()

    storages = [
        grouped.state[p]["momentum"].untyped_storage().data_ptr()
        for p in first + second
    ]
    assert len(set(storages[:3])) == 1, "first group should share one buffer"
    assert len(set(storages[3:])) == 1, "second group should share one buffer"
    assert not set(storages[:3]) & set(storages[3:]), "groups aliased"

    for grouped_p, solo_p in zip(first + second, apart_first + apart_second, strict=True):
        assert torch.equal(grouped_p.detach(), solo_p.detach())


def test_muon_rejects_vectors_and_empty_parameter_lists():
    vector = torch.nn.Parameter(torch.zeros(4))
    try:
        Muon([vector], lr=1e-2)
    except ValueError:
        pass
    else:
        raise AssertionError("Muon accepted a 1-D parameter")
    try:
        Muon([], lr=1e-2)
    except ValueError:
        pass
    else:
        raise AssertionError("Muon accepted an empty parameter list")


def test_muon_state_dict_round_trip_restores_momentum():
    parameters = [
        torch.nn.Parameter(torch.randn(6, 6)),
        torch.nn.Parameter(torch.randn(12, 6)),
    ]
    for parameter in parameters:
        parameter.grad = torch.randn_like(parameter)
    optimizer = Muon(parameters, lr=1e-2)
    optimizer.step()
    saved = optimizer.state_dict()

    clones = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    restored = Muon(clones, lr=1e-2)
    restored.load_state_dict(saved)
    for original, clone in zip(parameters, clones, strict=True):
        assert torch.equal(
            restored.state[clone]["momentum"],
            optimizer.state[original]["momentum"],
        )


def test_muon_rebuilds_its_buffers_after_deepcopy_and_pickle():
    """``Optimizer.__getstate__`` drops everything outside state/param_groups.

    Nothing in the trainer copies an optimizer object today, but a revived one
    that lost its momentum buffer raises inside ``step`` rather than returning a
    wrong number, so the failure would surface far from its cause.
    """
    parameters = _matrices([(8, 8)] * 2 + [(4, 16)], seed=21)
    _assign_grads(parameters, seed=22)
    optimizer = Muon(parameters, lr=1e-2)
    optimizer.step()

    revivals = [copy.deepcopy(optimizer), pickle.loads(pickle.dumps(optimizer))]
    # Neither deepcopy nor pickle carries ``.grad``, so the second step has to
    # be fed by hand on both sides.
    _assign_grads(parameters, seed=23)
    for revived in revivals:
        _assign_grads(revived.param_groups[0]["params"], seed=23)
        revived.step()
    optimizer.step()
    for revived in revivals:
        copies = revived.param_groups[0]["params"]
        for parameter, copied in zip(parameters, copies, strict=True):
            assert torch.equal(parameter.detach(), copied.detach())
            assert torch.equal(
                optimizer.state[parameter]["momentum"],
                revived.state[copied]["momentum"],
            )


@pytest.mark.skipif(
    os.environ.get("RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set RUN_CUDA_TESTS=1 on CUDA host",
)
def test_cuda_kernels_match_the_per_tensor_reference():
    """Every other exactness test here is CPU-only evidence.

    ``_foreach_add_`` has a CUDA kernel of its own and a CPU kernel that is a
    loop over ``add_``, so agreeing with the per-tensor form on CPU says nothing
    about the arithmetic the trainer runs.  Batching is checked here too, at
    real trunk shapes and at both momentum values that straddle the
    ``|w| < 0.5`` branch in ATen's lerp.

    Batching is NOT bit-exact and asserting that it is was wrong.  cuBLAS
    picks its strided-batched GEMM by batch count, so a 3-row ``bmm`` and a
    1-row ``bmm`` accumulate in different orders; five iterations of a cubic
    then amplify that bf16 rounding to a couple of percent on the small
    elements of a near-orthogonal matrix.  Nothing downstream needs
    bit-exactness -- Muon's output is not on the age-0 canary path, and the
    shape buckets are fixed within a run, so a run is still reproducible.

    What DOES need checking is that no row leaks into another.  ``bmm``
    cannot mix batch entries structurally, but the normalization and the
    ``.mT`` swaps are written by hand over the same tensors, so the test
    pins the property directly: each batched row must track its own
    unbatched result far more closely than it tracks a sibling's.
    """
    device = torch.device("cuda")
    torch.manual_seed(31)
    for mu in (0.05, 0.95):
        momentum_t = torch.tensor(mu, dtype=torch.float32, device="cpu")
        for shape in ((512, 512), (2048, 512), (512, 2048)):
            grads = torch.randn((3, *shape), device=device)
            momenta = torch.randn((3, *shape), device=device)
            together = _polar_express(
                grads.clone(), momenta.clone(), momentum_t
            )
            rows = [
                _polar_express(
                    grads[index : index + 1].clone(),
                    momenta[index : index + 1].clone(),
                    momentum_t,
                )[0]
                for index in range(3)
            ]
            for index, row in enumerate(rows):
                torch.testing.assert_close(
                    row, together[index], rtol=8e-2, atol=8e-3
                )
                own = (row.float() - together[index].float()).norm()
                for other in range(3):
                    if other == index:
                        continue
                    crossed = (row.float() - together[other].float()).norm()
                    assert crossed > 20 * own

    params = [torch.randn(s, device=device) for s in ((8, 8), (16, 4), (4, 16))]
    updates = [torch.randn_like(p) for p in params]
    reference = [p.clone() for p in params]
    torch._foreach_add_(params, updates, alpha=-8.333e-5)
    for restored, update in zip(reference, updates, strict=True):
        restored.add_(update, alpha=-8.333e-5)
    for stepped, restored in zip(params, reference, strict=True):
        assert torch.equal(stepped, restored)


def test_build_optimizers_muon_layout_partitions_exactly():
    wrapper = _wrapper()
    critic = _critic(wrapper)
    optimizers = build_optimizers(
        wrapper,
        critic,
        learning_rate=1e-3,
        critic_learning_rate=4e-3,
        trunk_optimizer="muon",
        muon_learning_rate=2e-3,
        critic_muon_learning_rate=3e-3,
        fused=False,
    )
    assert set(optimizers) == {"actor", "actor_muon", "critic", "critic_muon"}
    assert isinstance(optimizers["actor_muon"], Muon)
    assert isinstance(optimizers["critic_muon"], Muon)
    assert optimizers["actor_muon"].param_groups[0]["lr"] == 2e-3
    assert optimizers["critic_muon"].param_groups[0]["lr"] == 3e-3
    assert {
        group["lr"] for group in optimizers["actor"].param_groups
    } == {1e-3}
    assert {
        group["lr"] for group in optimizers["critic"].param_groups
    } == {4e-3}

    backbone = wrapper.backbone
    expected_actor_muon = {
        id(p) for p in backbone.blocks.parameters() if p.ndim >= 2
    }
    actual_actor_muon = {
        id(p)
        for group in optimizers["actor_muon"].param_groups
        for p in group["params"]
    }
    assert actual_actor_muon == expected_actor_muon
    # 2 layers x (q, k, v, attn-proj, mlp-fc, mlp-proj)
    assert len(actual_actor_muon) == 12

    adamw_actor = {
        id(p)
        for group in optimizers["actor"].param_groups
        for p in group["params"]
    }
    assert not (adamw_actor & actual_actor_muon)
    assert {id(p) for p in renderer_parameters(backbone)} <= adamw_actor
    assert id(backbone.embed.weight) in adamw_actor
    assert {id(p) for p in wrapper.adapter.parameters()} <= adamw_actor
    assert {
        id(p) for p in wrapper.transition.log_sigma_head.parameters()
    } <= adamw_actor
    assert {
        id(p) for p in wrapper.transition.mean_head.parameters()
    } <= adamw_actor

    critic_muon = {
        id(p)
        for group in optimizers["critic_muon"].param_groups
        for p in group["params"]
    }
    critic_adamw = {
        id(p)
        for group in optimizers["critic"].param_groups
        for p in group["params"]
    }
    assert critic_muon == {
        id(p) for p in critic.trunk.blocks.parameters() if p.ndim >= 2
    }
    assert not (critic_muon & critic_adamw)
    assert critic_muon | critic_adamw == {id(p) for p in critic.parameters()}
    assert id(critic.head.weight) in critic_adamw
    assert {id(p) for p in critic.adapter.parameters()} <= critic_adamw


def test_build_optimizers_adamw_layout_is_unchanged():
    wrapper = _wrapper()
    critic = _critic(wrapper)
    optimizers = build_optimizers(
        wrapper, critic, learning_rate=1e-3, fused=False,
    )
    assert set(optimizers) == {"actor", "critic"}
    registered = {
        id(p)
        for optimizer in optimizers.values()
        for group in optimizer.param_groups
        for p in group["params"]
    }
    assert {id(p) for p in critic.parameters()} <= registered
    assert {
        id(p) for p in wrapper.backbone.blocks.parameters() if p.ndim >= 2
    } <= registered


def test_role_helpers_drive_muon_optimizers():
    wrapper = _wrapper()
    critic = _critic(wrapper)
    optimizers = build_optimizers(
        wrapper,
        critic,
        learning_rate=1e-3,
        trunk_optimizer="muon",
        muon_learning_rate=1e-2,
        critic_muon_learning_rate=1e-2,
        fused=False,
    )
    block_weight = next(
        p for p in wrapper.backbone.blocks.parameters() if p.ndim >= 2
    )
    critic_block_weight = next(
        p for p in critic.trunk.blocks.parameters() if p.ndim >= 2
    )
    for parameter in (block_weight, critic_block_weight):
        parameter.grad = torch.randn_like(parameter)
    actor_before = block_weight.detach().clone()
    critic_before = critic_block_weight.detach().clone()

    step_optimizers(optimizers, "actor")
    assert not torch.equal(block_weight.detach(), actor_before)
    assert torch.equal(critic_block_weight.detach(), critic_before)

    step_optimizers(optimizers, "critic")
    assert not torch.equal(critic_block_weight.detach(), critic_before)

    zero_optimizers(optimizers, "actor")
    zero_optimizers(optimizers, "critic")
    assert block_weight.grad is None
    assert critic_block_weight.grad is None

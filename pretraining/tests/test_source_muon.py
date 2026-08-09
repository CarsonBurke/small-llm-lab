from __future__ import annotations

import torch

from pretraining.source_muon import (
    Muon,
    PerHeadMuon,
    zeropower_via_newtonschulz5,
)


def _reference_newtonschulz5(G: torch.Tensor) -> torch.Tensor:
    """The source trainer's iteration, transcribed independently."""

    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        A = X @ X.mT
        B = -4.7750 * A + 2.0315 * A @ A
        X = 3.4445 * X + B @ X
    return X.mT if transposed else X


def test_orthogonalization_matches_the_source_iteration() -> None:
    torch.manual_seed(0)
    for shape in ((8, 8), (12, 4), (4, 12), (3, 6, 5)):
        gradient = torch.randn(*shape)
        torch.testing.assert_close(
            zeropower_via_newtonschulz5(gradient),
            _reference_newtonschulz5(gradient),
        )


def test_update_is_scaled_for_tall_matrices_and_normalized_for_wide_ones() -> None:
    # ``max(1, rows / cols) ** 0.5`` is the source's rectangular scaling; a
    # port that drops it halves the step on the trunk's (2070, 512) matrices.
    torch.manual_seed(0)
    tall = torch.nn.Parameter(torch.zeros(16, 4))
    wide = torch.nn.Parameter(torch.zeros(4, 16))
    tall.grad = torch.randn(16, 4)
    wide.grad = tall.grad.mT.contiguous()
    optimizer = Muon([tall, wide], lr=1.0, weight_decay=0.0, mu=0.0)
    optimizer.step()
    ratio = tall.detach().norm() / wide.detach().norm()
    torch.testing.assert_close(ratio, torch.tensor(2.0), rtol=2e-2, atol=2e-2)


def test_decoupled_decay_shrinks_before_the_update_is_applied() -> None:
    parameter = torch.nn.Parameter(torch.eye(4))
    parameter.grad = torch.zeros(4, 4)
    Muon([parameter], lr=0.1, weight_decay=0.5, mu=0.95).step()
    # A zero gradient orthogonalizes to zero, so only the decay survives.
    torch.testing.assert_close(parameter.detach(), torch.eye(4) * (1 - 0.1 * 0.5))


def test_a_parameter_outside_the_loss_graph_is_neither_decayed_nor_stepped() -> None:
    stepped = torch.nn.Parameter(torch.eye(4))
    stepped.grad = torch.randn(4, 4)
    skipped = torch.nn.Parameter(torch.eye(4))
    optimizer = Muon([stepped, skipped], lr=0.1, weight_decay=0.5, mu=0.95)
    optimizer.step()
    torch.testing.assert_close(skipped.detach(), torch.eye(4))
    assert not torch.equal(stepped.detach(), torch.eye(4))
    assert skipped not in optimizer.state


def test_per_head_orthogonalization_keeps_heads_independent() -> None:
    torch.manual_seed(0)
    gradient = torch.randn(8, 4)
    # Second head carries no signal, so a per-head update must leave it alone
    # while a whole-matrix update mixes the first head's directions into it.
    gradient[4:] = 0.0
    per_head = torch.nn.Parameter(torch.zeros(8, 4))
    whole = torch.nn.Parameter(torch.zeros(8, 4))
    per_head.grad = gradient.clone()
    whole.grad = gradient.clone()
    PerHeadMuon([(per_head, 2)], lr=1.0, weight_decay=0.0, mu=0.0).step()
    Muon([whole], lr=1.0, weight_decay=0.0, mu=0.0).step()
    torch.testing.assert_close(per_head.detach()[4:], torch.zeros(4, 4))
    assert whole.detach()[:4].norm() > 0
    assert not torch.allclose(per_head.detach()[:4], whole.detach()[:4])


def test_per_head_rejects_a_head_count_that_does_not_divide_the_rows() -> None:
    parameter = torch.nn.Parameter(torch.zeros(9, 4))
    try:
        PerHeadMuon([(parameter, 2)], lr=1.0)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("an indivisible per-head projection was accepted")


def test_muon_rejects_rank_one_parameters() -> None:
    try:
        Muon([torch.nn.Parameter(torch.zeros(4))], lr=1.0)
    except ValueError as error:
        assert "rank two" in str(error)
    else:
        raise AssertionError("a scalar parameter was accepted into Muon")

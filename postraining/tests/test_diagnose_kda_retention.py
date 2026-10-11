"""CPU contracts for the KDA retention diagnostic's probe propagation."""

import math

import torch

from postraining.diagnose_kda_retention import (
    horizons_for,
    mask_past_row_end,
    retention_curves,
)
from pretraining.nanogpt_mini.nanogpt_mini_kda_model import (
    _l2norm,
    kda_delta_update,
)


def _transitions(L=2, B=2, T=24, H=3, Dk=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    k = _l2norm(torch.randn(L, B, T, H, Dk, generator=generator))
    gate = -2.0 * torch.rand(L, B, T, H, Dk, generator=generator)
    beta = torch.rand(L, B, T, H, generator=generator)
    return k, gate, beta


def test_probes_match_the_model_recurrence_with_writes_removed():
    """With v = 0 the model's own step is exactly the state Jacobian."""
    k, gate, beta = _transitions()
    L, B, T, H, Dk = k.shape
    starts, operator_starts = [0, 5], [0, 5]
    horizons = horizons_for(T)
    curves = retention_curves(
        k, gate.exp(), beta, starts, operator_starts, horizons
    )

    for layer in range(L):
        for start in starts:
            # Rows of the value-major state are key-space vectors: seed one
            # row with the written key and the rest with a basis of key space.
            state = torch.cat(
                (k[layer, :, start][:, :, None, :],
                 torch.eye(Dk).expand(B, H, Dk, Dk)),
                dim=2,
            )
            for tau in range(1, T - start):
                t = start + tau
                _, state = kda_delta_update(
                    k[layer, :, t], k[layer, :, t],
                    torch.zeros(B, H, Dk + 1),
                    gate[layer, :, t], beta[layer, :, t], state,
                )
                if tau not in horizons:
                    continue
                column = horizons.index(tau)
                slot = starts.index(start)
                torch.testing.assert_close(
                    curves["own_key"][layer, :, :, slot, column],
                    state[:, :, 0].norm(dim=-1),
                    rtol=1e-4, atol=1e-6,
                )
                jacobian = state[:, :, 1:]
                torch.testing.assert_close(
                    curves["operator"][layer, :, :, slot, column],
                    torch.linalg.matrix_norm(jacobian, ord=2),
                    rtol=1e-4, atol=1e-6,
                )
                torch.testing.assert_close(
                    curves["energy"][layer, :, :, slot, column],
                    jacobian.square().sum((-2, -1)) / Dk,
                    rtol=1e-4, atol=1e-6,
                )


def test_gate_only_is_the_cumulative_decay_and_operator_bounds_own_key():
    k, gate, beta = _transitions(seed=1)
    T = k.size(2)
    starts, horizons = [3], horizons_for(T)
    curves = retention_curves(k, gate.exp(), beta, starts, [3], horizons)
    for column, tau in enumerate(horizons):
        if 3 + tau >= T:
            assert curves["gate_only"][..., 0, column].isnan().all()
            continue
        decayed = k[:, :, 3] * gate[:, :, 4:4 + tau].sum(2).exp()
        torch.testing.assert_close(
            curves["gate_only"][..., 0, column], decayed.norm(dim=-1),
            rtol=1e-4, atol=1e-6,
        )
        # The operator norm is the best-retained direction, so it bounds the
        # written key. (Decay alone does not: the erase can rotate a probe
        # into slower-decaying dimensions.)
        assert (curves["own_key"][..., 0, column]
                <= curves["operator"][..., 0, column] + 1e-5).all()


def test_no_decay_and_no_erase_retains_everything():
    k, _, _ = _transitions(seed=2)
    T = k.size(2)
    horizons = horizons_for(T)
    curves = retention_curves(
        k, torch.ones_like(k), torch.zeros(k.shape[:-1]), [0], [0], horizons
    )
    for name in ("own_key", "gate_only", "operator", "energy"):
        torch.testing.assert_close(
            curves[name], torch.ones_like(curves[name]), atol=1e-5, rtol=0
        )


def test_padding_masks_probes_that_end_past_the_row():
    k, gate, beta = _transitions(seed=3)
    T = k.size(2)
    starts, operator_starts, horizons = [0, 8], [0], horizons_for(T)
    curves = retention_curves(
        k, gate.exp(), beta, starts, operator_starts, horizons
    )
    valid = torch.tensor([T, 10])
    mask_past_row_end(curves, starts, operator_starts, horizons, valid)
    for column, tau in enumerate(horizons):
        full_row = curves["own_key"][:, 0, :, 0, column]
        short_row = curves["own_key"][:, 1, :, 1, column]
        assert not full_row.isnan().any()
        assert short_row.isnan().all() == (8 + tau >= 10)
        assert math.isnan(float(curves["operator"][0, 1, 0, 0, column])) == (
            tau >= 10
        )

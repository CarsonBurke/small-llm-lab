"""Round-trip coverage for the v1 -> v2 combiner gain fold.

``fold_combiner_gain`` performs state-dict surgery on live training
checkpoints (the k3 10h run, and any v1 warmup checkpoint reused later via
``--actor-critic-init``), so its index arithmetic gets a synthetic
round-trip here: build a v1-shaped model, train one step so the optimizer
state is populated, fold, and demand (1) the folded weights load strictly
into the v2 layout, (2) the forward is preserved, (3) the remapped
optimizer state loads and steps. The parameter-ordering trap this guards:
``Module.parameters()`` yields direct ``nn.Parameter`` attributes before
children's parameters, so the v1 combiner order is [gain, type_bias,
carry.weight, ...] no matter that ``carry`` was assigned first.
"""

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from pretraining.nanogpt_mini import nanogpt_mini_model
from postraining.fold_combiner_gain import (
    fold_model_state,
    remap_optimizer_state,
)
from postraining.latent_thought import CombinedEmbedding

DIM = 8
MLP_HIDDEN = 16


class _V1Combiner(nn.Module):
    """The retired gated combiner, state-dict compatible with v2 plus gain."""

    def __init__(self):
        super().__init__()
        self.carry = nn.Linear(DIM, DIM, bias=False)
        nn.init.orthogonal_(self.carry.weight)
        self.gain = nn.Parameter(torch.tensor(-3e-4))
        self.type_bias = nn.Parameter(torch.randn(DIM) * 0.01)
        self.norms = nn.ModuleList([nanogpt_mini_model.RMSNorm(DIM)])
        self.mlps = nn.ModuleList([nanogpt_mini_model.MLP(DIM, MLP_HIDDEN)])

    def forward(self, base, hidden):
        combined = base + (
            self.gain * F.linear(hidden, self.carry.weight) + self.type_bias
        )
        for norm, mlp in zip(self.norms, self.mlps, strict=True):
            combined = combined + mlp(norm(combined))
        return combined


class _Model(nn.Module):
    def __init__(self, combiner: nn.Module):
        super().__init__()
        self.trunk = nn.Linear(DIM, DIM)
        self.combiner = combiner
        self.renderer = nn.Linear(DIM, DIM)

    def forward(self, base, hidden):
        return self.renderer(self.trunk(self.combiner(base, hidden)))


def _optimizer(model: _Model) -> torch.optim.AdamW:
    # The trainer's actor layout: [trunk, combiner, renderer] groups, so the
    # gain's global index is param_groups[1]["params"][0] exactly as in the
    # real checkpoint.
    return torch.optim.AdamW(
        [
            {"params": list(model.trunk.parameters()), "lr": 1e-3},
            {"params": list(model.combiner.parameters()), "lr": 1e-3},
            {"params": list(model.renderer.parameters()), "lr": 1e-3},
        ],
        weight_decay=0.0,
    )


def test_fold_round_trip_preserves_function_and_loads_optimizer_state():
    torch.manual_seed(5)
    v1 = _Model(_V1Combiner())
    optimizer = _optimizer(v1)
    base = torch.randn(4, DIM)
    hidden = torch.randn(4, DIM)
    v1(base, hidden).square().mean().backward()
    optimizer.step()

    with torch.no_grad():
        reference = v1(base, hidden)
    model_sd = v1.state_dict()
    optimizer_sd = optimizer.state_dict()
    fold_model_state(model_sd)
    remap_optimizer_state(
        optimizer_sd,
        expected_gain_index=optimizer_sd["param_groups"][1]["params"][0],
    )

    v2 = _Model(CombinedEmbedding(DIM, mlp_hidden=MLP_HIDDEN, num_blocks=1))
    v2.load_state_dict(model_sd, strict=True)
    with torch.no_grad():
        # inject() computes in fp32 and casts once; the fold scales the
        # weight where v1 scaled the output, so parity is ulp-level, not
        # bitwise.
        folded = v2(base, hidden)
    torch.testing.assert_close(folded, reference, rtol=1e-6, atol=1e-7)

    fresh = _optimizer(v2)
    fresh.load_state_dict(optimizer_sd)
    # carry.weight's moments were dropped (they lived in the gain-scaled
    # gradient space); a step must lazily reinitialize them and run.
    v2(base, hidden).square().mean().backward()
    fresh.step()
    state_shapes = {
        entry["exp_avg"].shape
        for entry in fresh.state_dict()["state"].values()
    }
    assert torch.Size([]) not in state_shapes


def test_remap_refuses_ambiguous_scalars():
    def entry(shape):
        return {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros(shape),
            "exp_avg_sq": torch.zeros(shape),
        }

    sd = {
        "state": {0: entry(()), 1: entry(()), 2: entry((4, 4))},
        "param_groups": [{"params": [0, 1, 2], "lr": 1e-3}],
    }
    with pytest.raises(ValueError, match="exactly one scalar"):
        remap_optimizer_state(sd, None)


def test_remap_refuses_mismatched_construction_position():
    def entry(shape):
        return {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros(shape),
            "exp_avg_sq": torch.zeros(shape),
        }

    sd = {
        "state": {0: entry((4,)), 1: entry(()), 2: entry((4, 4))},
        "param_groups": [{"params": [0, 1, 2], "lr": 1e-3}],
    }
    with pytest.raises(ValueError, match="construction"):
        remap_optimizer_state(sd, expected_gain_index=0)

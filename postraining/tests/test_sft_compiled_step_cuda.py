"""The compiled SFT step computes the eager step's loss and gradients.

``SupervisedCE(compiled=True)`` compiles every residual block, with the KDA
mixer fenced from Dynamo only around its FLA recurrence, and the readout
cross-entropy as one dynamic-shape region. This checks that the compiled
trunk -- projections, short convolutions and gated norm now traced around
``chunk_kda`` -- trains the same function as the eager one.

Run only through mlq: .venv/bin/python -m pytest -q this_file.
"""

import copy

import numpy as np
import pytest
import torch

from postraining.kda_backbone import NanoKDABackbone
from postraining.sft_trace_train import (
    IGNORE_INDEX,
    SupervisedCE,
    accumulate_step_gradients,
    holdout_ce,
)
from postraining.tests.test_kda_backbone import MODEL_KWARGS

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires queued CUDA"
)


def _live_backbone() -> NanoKDABackbone:
    torch.manual_seed(7)
    backbone = NanoKDABackbone(**MODEL_KWARGS).cuda()
    with torch.no_grad():
        # The fresh init zeroes every output projection; give each residual
        # branch signal so the comparison covers the whole trunk.
        for block in backbone.blocks:
            if block.use_kda:
                block.attn.o_proj.weight.normal_(std=0.02)
            else:
                block.attn.proj.weight.normal_(std=0.02)
            if block.use_mlp:
                block.mlp.proj.weight.normal_(std=0.02)
        backbone.proj.weight.normal_(std=0.05)
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    return backbone.train()


def _rows(count: int, seq_len: int, vocab: int, seed: int):
    generator = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        tokens = generator.integers(0, vocab, seq_len, dtype=np.int32)
        targets = np.roll(tokens, -1)
        targets[generator.random(seq_len) < 0.35] = IGNORE_INDEX
        rows.append((tokens, targets))
    return rows


def test_compiled_step_matches_eager_loss_and_gradients():
    eager = _live_backbone()
    compiled = copy.deepcopy(eager)
    # Five rows in micro-batches of two: the tail micro-batch of one row is
    # a second compiled shape, as an epoch's final step is in training.
    rows = _rows(5, 512, MODEL_KWARGS["vocab_size"], seed=11)
    holdout = _rows(3, 512, MODEL_KWARGS["vocab_size"], seed=12)
    device = torch.device("cuda")

    losses, holdout_losses, gradients = [], [], []
    for backbone, is_compiled in ((eager, False), (compiled, True)):
        loss_fn = SupervisedCE(backbone, compiled=is_compiled)
        losses.append(float(accumulate_step_gradients(loss_fn, rows, 2, device)))
        gradients.append(
            {
                name: parameter.grad.float()
                for name, parameter in backbone.named_parameters()
            }
        )
        # The no-grad holdout pass compiles its own graphs, tail included.
        holdout_losses.append(holdout_ce(loss_fn, holdout, 2, device))

    # Both paths run under bf16 autocast; fusion changes rounding, not math.
    for pair in (losses, holdout_losses):
        assert np.isfinite(pair).all()
        assert abs(pair[1] - pair[0]) / pair[0] < 2e-3
    reference, candidate = gradients
    assert reference.keys() == candidate.keys()
    # Per parameter, so a small group (a conv kernel, A_log, a norm gain)
    # whose gradient the compiled path dropped cannot hide inside the norm
    # of the vocab head's.
    for name, expected in reference.items():
        assert float(expected.norm()) > 0, name
        error = float((candidate[name] - expected).norm() / expected.norm())
        assert error < 5e-2, (name, error)

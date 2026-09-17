"""GPU equivalence of the flex backend against the dense reference in
``pretraining/nanogpt_mini/nanogpt_mini_hashptr_model.py``.

Submit through mlq (GPU workload). Compares outputs and every parameter
gradient of ``HashPointerAttention`` under both backends at the production
shape (T=1024, 4 rounds x 7 bits, 4 heads x 128) in bf16, with and without
the local window, in eval mode (mode bits) and in train mode with the same
RNG stream, so any disagreement between ``_round_mods`` and the dense mask/
prior shows up numerically.
"""

import copy

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_hashptr_model import HashPointerAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def build_pair(local_window, seq_len=1024, rounds=4, bits=7, dim=512):
    torch.manual_seed(0)
    dense = HashPointerAttention(dim, seq_len, rounds, bits, backend="dense", local_window=local_window)
    with torch.no_grad():
        for name, p in dense.named_parameters():
            if name == "hash":
                p.normal_(std=128 ** -0.5)
            elif name.endswith("weight"):
                p.normal_(std=0.02 if "proj" in name else 0.05)
            else:
                p.normal_(std=0.3)
    flex = copy.deepcopy(dense)
    flex.backend = "flex"
    return dense.cuda().bfloat16(), flex.cuda().bfloat16()


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("local_window", [0, 64])
def test_flex_matches_dense_outputs_and_grads(training, local_window):
    dense, flex = build_pair(local_window)
    dense.train(training)
    flex.train(training)
    x = torch.randn(2, 1024, 512, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(7)
    y_dense = dense(x)
    torch.manual_seed(7)
    y_flex = flex(x)
    assert torch.isfinite(y_flex.float()).all()
    assert torch.allclose(y_dense.float(), y_flex.float(), atol=2e-2, rtol=2e-2), (
        (y_dense.float() - y_flex.float()).abs().max()
    )
    weight = torch.randn_like(y_dense)
    (y_dense.float() * weight.float()).sum().backward()
    (y_flex.float() * weight.float()).sum().backward()
    for (name, pd), (_, pf) in zip(dense.named_parameters(), flex.named_parameters()):
        g1, g2 = pd.grad.float(), pf.grad.float()
        scale = g1.abs().max().clamp_min(1e-6)
        assert torch.allclose(g1 / scale, g2 / scale, atol=5e-2), (name, ((g1 - g2).abs().max() / scale))
    assert dense.hash.grad.abs().sum() > 0
    stats = dense.candidate_stats(x)
    assert 0 < stats["causal_fraction"] < 1

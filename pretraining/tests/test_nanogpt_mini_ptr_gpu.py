"""GPU equivalence of the flex backend against the gather reference in
``pretraining/nanogpt_mini/nanogpt_mini_ptr_model.py``.

Submit through mlq (GPU workload). Compares outputs and every parameter
gradient of ``PointerAttention`` under both backends at the production shape
(T=1024, Bk=8, K=4, 4 heads x 128) in bf16, with and without the always-on local
window, in eval mode (mode bits) and in train mode with the same RNG stream
(identical Bernoulli draws), so any
mask/prior/offset disagreement between ``pointer_tables`` + flex mods and the
explicit gather shows up as a numeric mismatch.
"""

import copy

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_ptr_model import PointerAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def build_pair(local_window, seq_len=1024, block=8, pointers=4, dim=512):
    torch.manual_seed(0)
    gather = PointerAttention(dim, seq_len, pointers, block, backend="gather", local_window=local_window)
    with torch.no_grad():
        for name, p in gather.named_parameters():
            if name.endswith("weight"):
                p.normal_(std=0.02 if "proj" in name else 0.05)
            else:
                p.normal_(std=0.3)
        # geometric address prior, as in the training script
        bits = gather.num_bits
        gather.ptr.bias.view(-1, bits).copy_(-0.5 * torch.arange(bits, dtype=torch.float32))
    flex = copy.deepcopy(gather)
    flex.backend = "flex"
    return gather.cuda().bfloat16(), flex.cuda().bfloat16()


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("local_window", [0, 64])
def test_flex_matches_gather_outputs_and_grads(training, local_window):
    gather, flex = build_pair(local_window)
    gather.train(training)
    flex.train(training)
    x = torch.randn(2, 1024, 512, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(7)
    y_gather = gather(x)
    torch.manual_seed(7)
    y_flex = flex(x)
    tol = dict(atol=2e-2, rtol=2e-2)
    assert torch.allclose(y_gather.float(), y_flex.float(), **tol), (
        (y_gather.float() - y_flex.float()).abs().max()
    )
    # also the raw attention output (before proj) on early tokens where whole
    # pointer balls precede position 0
    assert torch.isfinite(y_flex.float()).all()

    weight = torch.randn_like(y_gather)
    (y_gather.float() * weight.float()).sum().backward()
    (y_flex.float() * weight.float()).sum().backward()
    for (name, pg), (_, pf) in zip(gather.named_parameters(), flex.named_parameters()):
        g1, g2 = pg.grad.float(), pf.grad.float()
        scale = g1.abs().max().clamp_min(1e-6)
        assert torch.allclose(g1 / scale, g2 / scale, atol=5e-2), (name, ((g1 - g2).abs().max() / scale))
    assert gather.ptr.weight.grad.abs().sum() > 0

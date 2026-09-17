"""GPU equivalence of the flex backend against the gather reference in
``pretraining/nanogpt_mini/nanogpt_mini_iptr_model.py``.

Submit through mlq (GPU workload). Compares outputs and every parameter
gradient of ``IntervalPointerAttention`` under both backends at the
production shape (T=1024, Bk=8, K=4, 4 heads x 128), with and without the
always-on local window, in eval mode (mode bits) and in train mode with the
same RNG stream (identical Bernoulli draws). The equivalence assertion runs
in fp32 (measured 1e-6 relative on every gradient) so any ``valid``/prior/
offset disagreement between ``pointer_tables`` + flex mods and the explicit
gather shows up; bf16 only checks the output, because flex accumulates
dK/dV in bf16 and its 3-6% gradient noise would otherwise mask semantics.
"""

import copy

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_iptr_model import IntervalPointerAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def build_pair(local_window, dtype, block=8, pointers=4, width=7, dim=512):
    torch.manual_seed(0)
    gather = IntervalPointerAttention(dim, pointers, block, width, backend="gather", local_window=local_window)
    with torch.no_grad():
        for name, p in gather.named_parameters():
            if name.endswith("weight"):
                p.normal_(std=0.02 if "proj" in name else 0.05)
            else:
                p.normal_(std=0.3)
    flex = copy.deepcopy(gather)
    flex.backend = "flex"
    return gather.cuda().to(dtype), flex.cuda().to(dtype)


def run_pair(training, local_window, dtype):
    gather, flex = build_pair(local_window, dtype)
    gather.train(training)
    flex.train(training)
    x = torch.randn(2, 1024, 512, device="cuda", dtype=dtype)
    torch.manual_seed(7)
    y_gather = gather(x)
    torch.manual_seed(7)
    y_flex = flex(x)
    weight = torch.randn_like(y_gather)
    (y_gather.float() * weight.float()).sum().backward()
    (y_flex.float() * weight.float()).sum().backward()
    return gather, flex, y_gather.float(), y_flex.float()


def rel(a, b):
    return float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12))


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("local_window", [0, 64])
def test_flex_matches_gather_fp32(training, local_window):
    gather, flex, y_gather, y_flex = run_pair(training, local_window, torch.float32)
    assert torch.isfinite(y_flex).all()
    assert rel(y_gather, y_flex) < 1e-5
    for (name, pg), (_, pf) in zip(gather.named_parameters(), flex.named_parameters()):
        assert rel(pg.grad, pf.grad) < 1e-4, (name, rel(pg.grad, pf.grad))
    assert gather.head.ctrl.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("training", [False, True])
def test_flex_bf16_output_matches_gather(training):
    gather, flex, y_gather, y_flex = run_pair(training, 0, torch.bfloat16)
    assert torch.isfinite(y_flex).all()
    assert rel(y_gather, y_flex) < 1e-2

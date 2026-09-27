"""The fused softcapped target readout == the composed readout on CUDA.

At the production vocabulary, under bf16 autocast and compiled the way the
trainer compiles the emit tail. Run only through mlq:
.venv/bin/python -m pytest -q this_file.
"""

import pytest
import torch

from postraining.kda_backbone import NanoKDABackbone
from postraining.kda_gpu_parity import MODEL_KWARGS

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires queued CUDA"
)


def _composed(backbone, features, targets):
    logits = backbone.logits_from_features(features)
    return logits.float().log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)


def _fused(backbone, features, targets):
    return backbone.target_logprobs_from_features(features, targets)


@pytest.mark.parametrize("compiled", [False, True])
def test_fused_readout_matches_composed_under_bf16_autocast(compiled):
    torch.manual_seed(0)
    backbone = NanoKDABackbone(**dict(MODEL_KWARGS, vocab_size=50304)).cuda()
    with torch.no_grad():
        # Logits well past the softcap knee, as a trained readout produces.
        backbone.proj.weight.normal_(std=0.2)
        backbone.proj.bias.normal_(std=1.0)
    features = torch.randn(3001, 2 * backbone.model_dim, device="cuda")
    targets = torch.randint(0, 50304, (3001,), device="cuda")
    upstream = torch.randn(3001, device="cuda")
    # The reference is the composed readout run eagerly: it rounds the raw
    # logit gradient to bf16 at the ``.float()`` boundary before the readout
    # GEMM backward, exactly as the fused op does. Compiled, Inductor may sum
    # the composed bias gradient from the unrounded fp32 values instead, a
    # different (not a wrong) rounding that is not the fused op's contract.
    fused = torch.compile(_fused, fullgraph=True, dynamic=True) if compiled else _fused
    results = {}
    for name, function in (("composed", _composed), ("fused", fused)):
        backbone.zero_grad(set_to_none=True)
        leaf = features.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = function(backbone, leaf, targets)
        (value * upstream).sum().backward()
        results[name] = (
            value.detach(),
            leaf.grad,
            backbone.proj.weight.grad,
            backbone.proj.bias.grad,
        )
    value, dx, dw, db = results["fused"]
    reference, reference_dx, reference_dw, reference_db = results["composed"]
    assert value.dtype == torch.float32
    torch.testing.assert_close(value, reference, rtol=0, atol=2e-5)
    # Both round the same fp32 raw-logit gradient to bf16 before the readout
    # GEMMs; the GEMMs then accumulate in their own order.
    for grad, reference_grad in ((dx, reference_dx), (dw, reference_dw)):
        assert grad.dtype == reference_grad.dtype
        torch.testing.assert_close(grad, reference_grad, rtol=2e-2, atol=1e-3)
    # The bias gradient is a 3001-row sum that cancels to small values, so
    # per-row bf16 rounding dominates it. Inductor fuses the fused op's bf16
    # cast into that sum without emulating the intermediate rounding, so
    # compiled it is the MORE exact sum; judge both against fp64.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        raw = backbone.proj(features[:, -backbone.model_dim:])
    x = raw.double()
    inverse_norm = (x.square() + 225).rsqrt()
    probabilities = (15 * x * inverse_norm).softmax(-1)
    is_target = torch.arange(x.size(-1), device="cuda") == targets[:, None]
    exact_db = (
        upstream.double()[:, None] * (is_target.double() - probabilities)
        * 15**3 * inverse_norm**3
    ).sum(0)
    assert db.dtype == reference_db.dtype
    fused_error = (db.double() - exact_db).abs().max()
    reference_error = (reference_db.double() - exact_db).abs().max()
    assert fused_error <= reference_error * 1.25 + 1e-5, (
        fused_error, reference_error
    )

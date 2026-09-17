"""GPU tests for ``nanogpt_mini_tree_model``: the fused Triton path over the
extended row table (``R != T``) equals the gather reference in forward and
in every gradient. Run via mlq (GPU workload)."""

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_tree_model import TreePointerAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("seq_len,pointers", [(64, 3), (256, 4)])
def test_fused_matches_gather(seq_len, pointers):
    torch.manual_seed(0)
    dim, head_dim = 512, 128
    fused = TreePointerAttention(dim, seq_len, pointers, 7, head_dim=head_dim, backend="fused").cuda().float()
    with torch.no_grad():
        for name, p in fused.named_parameters():
            if name.startswith("head.") and name != "head.ctrl.weight":
                p.normal_(std=0.5)
    ref = TreePointerAttention(dim, seq_len, pointers, 7, head_dim=head_dim, backend="gather").cuda().float()
    ref.load_state_dict(fused.state_dict())
    x = torch.randn(2, seq_len, dim, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    torch.manual_seed(1)
    y = fused(x)
    torch.manual_seed(1)
    y_ref = ref(x_ref)
    assert torch.allclose(y, y_ref, atol=1e-4, rtol=1e-4), (y - y_ref).abs().max()
    g = torch.randn_like(y)
    y.backward(g)
    y_ref.backward(g)
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-3, rtol=1e-3), (x.grad - x_ref.grad).abs().max()
    for (name, p), (_, p_ref) in zip(fused.named_parameters(), ref.named_parameters()):
        if p.grad is None:
            assert p_ref.grad is None, name
            continue
        assert torch.allclose(p.grad, p_ref.grad, atol=1e-3, rtol=1e-3), (name, (p.grad - p_ref.grad).abs().max())

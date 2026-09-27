"""Fused KDA decode step == the reference recurrence step, on CUDA.

Run only through mlq: .venv/bin/python -m pytest -q this_file.
"""

import pytest
import torch

from pretraining.nanogpt_mini import nanogpt_mini_kda_model as kda_model
from pretraining.nanogpt_mini.kda_decode_kernel import fused_decode_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires queued CUDA"
)


@pytest.fixture(autouse=True)
def _exact_fp32_matmul():
    """The reference contracts with einsum; TF32 would round it, not the kernel."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _inputs(rows: int, heads: int, dim: int, *, dtype=torch.bfloat16):
    torch.manual_seed(rows * 31 + dim)
    device = torch.device("cuda")
    q, k, v = torch.randn(3, rows, heads, dim, device=device).to(dtype)
    gate = kda_model.kda_decay_gate(
        torch.randn(rows, heads, dim, device=device),
        torch.zeros(heads, device=device),
        torch.randn(heads * dim, device=device) * 0.5 - 3.0,
    )
    beta = torch.rand(rows, heads, device=device)
    state = torch.randn(rows, heads, dim, dim, device=device) * 0.1
    return q, k, v, gate, beta, state


@pytest.mark.parametrize("rows,heads,dim", [(1, 3, 128), (8, 3, 128), (37, 2, 64)])
def test_fused_step_matches_reference_step(rows, heads, dim):
    q, k, v, gate, beta, state = _inputs(rows, heads, dim)
    assert fused_decode_supported(state)
    reference_state = state.clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = kda_model.kda_recurrent_step(
            q, k, v, gate, beta, reference_state
        )
        out = kda_model.kda_decode_recurrent_step(q, k, v, gate, beta, state)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state, reference_state, rtol=1e-5, atol=1e-5)


def test_fused_step_writes_through_a_strided_cache_view():
    """The arena hands the step leading-row slices of its resident caches."""
    q, k, v, gate, beta, full = _inputs(16, 3, 128)
    view = full[:8]
    untouched = full[8:].clone()
    expected_state = view.clone()
    expected = kda_model.kda_recurrent_step(
        q[:8], k[:8], v[:8], gate[:8], beta[:8], expected_state
    )
    out = kda_model.kda_decode_recurrent_step(
        q[:8], k[:8], v[:8], gate[:8], beta[:8], view
    )
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(full[:8], expected_state, rtol=1e-5, atol=1e-5)
    assert torch.equal(full[8:], untouched)


def test_fused_step_traces_into_a_fullgraph_compile():
    q, k, v, gate, beta, state = _inputs(8, 3, 128)
    eager_state = state.clone()
    eager = kda_model.kda_decode_recurrent_step(q, k, v, gate, beta, eager_state)
    compiled = torch.compile(
        kda_model.kda_decode_recurrent_step, fullgraph=True, dynamic=False
    )
    out = compiled(q, k, v, gate, beta, state)
    torch.testing.assert_close(out, eager, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(state, eager_state, rtol=1e-6, atol=1e-6)


def test_unsupported_state_falls_through_to_the_reference_step():
    q, k, v, gate, beta, state = _inputs(4, 2, 48)
    assert not fused_decode_supported(state)
    expected_state = state.clone()
    expected = kda_model.kda_recurrent_step(q, k, v, gate, beta, expected_state)
    out = kda_model.kda_decode_recurrent_step(q, k, v, gate, beta, state)
    assert torch.equal(out, expected)
    assert torch.equal(state, expected_state)

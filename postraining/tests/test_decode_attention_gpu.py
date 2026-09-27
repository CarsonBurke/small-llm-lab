"""``ranged_decode_attention``'s Triton kernel == its masked-SDPA reference.

Run only through mlq: .venv/bin/python -m pytest -q this_file.
"""

import pytest
import torch

from postraining.decode_attention import (
    _split_count,
    ranged_decode_attention,
    ranged_decode_attention_reference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires queued CUDA"
)


def _case(rows, heads, width, head_dim, dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    device = torch.device("cuda")
    cache_rows = rows + 3  # attend through a leading-row slice, as the arena does
    k = torch.randn(cache_rows, heads, width, head_dim, device=device, generator=generator)
    v = torch.randn(cache_rows, heads, width, head_dim, device=device, generator=generator)
    # The step's q is a transposed view: unit-stride heads, non-contiguous rows.
    q = torch.randn(rows, 1, heads, head_dim, device=device, generator=generator)
    q = q.transpose(1, 2).squeeze(2)
    return q.to(dtype), k.to(dtype)[:rows], v.to(dtype)[:rows]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 7, 256, 1024])
def test_kernel_matches_the_masked_reference(rows, dtype):
    heads, width, head_dim = 4, 1024, 128
    q, k, v = _case(rows, heads, width, head_dim, dtype, seed=rows)
    position = torch.tensor(611, device="cuda")
    starts = torch.randint(0, 600, (rows,), device="cuda")
    starts[0] = 611  # a range holding only the row's own token
    out = ranged_decode_attention(q, k, v, starts, position, 0.12)
    expected = ranged_decode_attention_reference(
        q.float(), k.float(), v.float(), starts, position, 0.12
    )
    assert out.dtype == dtype
    tolerance = 2e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out.float(), expected, rtol=tolerance, atol=tolerance)


def test_slots_outside_the_range_are_never_read():
    """Poisoning every slot before the start and after the write head leaves
    the output unchanged, so the kernel reads the live range only."""
    rows, heads, width, head_dim = 64, 4, 512, 128
    q, k, v = _case(rows, heads, width, head_dim, torch.bfloat16, seed=5)
    position = torch.tensor(300, device="cuda")
    starts = torch.randint(0, 290, (rows,), device="cuda")
    clean = ranged_decode_attention(q, k, v, starts, position, 0.12)
    keys = torch.arange(width, device="cuda")
    outside = (keys[None, :] < starts[:, None]) | (keys[None, :] > position)
    for cache in (k, v):
        cache.masked_fill_(outside[:, None, :, None], float("nan"))
    poisoned = ranged_decode_attention(q, k, v, starts, position, 0.12)
    assert torch.equal(clean, poisoned)


def test_split_merge_is_exercised_and_agrees_with_one_split():
    rows, heads, width, head_dim = 2, 4, 1024, 128
    assert _split_count(rows * heads, width, torch.device("cuda")) > 1
    q, k, v = _case(rows, heads, width, head_dim, torch.float32, seed=11)
    position = torch.tensor(1000, device="cuda")
    starts = torch.tensor([3, 700], device="cuda")
    out = ranged_decode_attention(q, k, v, starts, position, 0.12)
    expected = ranged_decode_attention_reference(q, k, v, starts, position, 0.12)
    torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-5)


def test_kernel_traces_into_a_fullgraph_compile_and_graph_replays():
    rows, heads, width, head_dim = 16, 4, 256, 128
    q, k, v = _case(rows, heads, width, head_dim, torch.bfloat16, seed=3)
    position = torch.tensor(100, device="cuda")
    starts = torch.randint(0, 90, (rows,), device="cuda")
    compiled = torch.compile(ranged_decode_attention, fullgraph=True, dynamic=False)
    eager = ranged_decode_attention(q, k, v, starts, position, 0.12)
    torch.testing.assert_close(compiled(q, k, v, starts, position, 0.12), eager)

    graph = torch.cuda.CUDAGraph()
    compiled(q, k, v, starts, position, 0.12)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        replayed = compiled(q, k, v, starts, position, 0.12)
    position.fill_(200)
    starts.fill_(150)
    graph.replay()
    expected = ranged_decode_attention(q, k, v, starts, position, 0.12)
    torch.testing.assert_close(replayed, expected)


def test_one_symbolic_compile_serves_every_row_bucket():
    """The split count varies with the row count; it must stay a runtime
    value so a ``dynamic=True`` step never recompiles per bucket."""
    heads, width, head_dim = 4, 1024, 128
    compiled = torch.compile(ranged_decode_attention, fullgraph=True, dynamic=True)
    position = torch.tensor(700, device="cuda")
    first = True
    for rows in (512, 96, 8, 2):
        q, k, v = _case(rows, heads, width, head_dim, torch.bfloat16, seed=rows)
        starts = torch.randint(0, 690, (rows,), device="cuda")
        with torch._dynamo.config.patch(error_on_recompile=not first):
            out = compiled(q, k, v, starts, position, 0.12)
        first = False
        expected = ranged_decode_attention_reference(
            q.float(), k.float(), v.float(), starts, position, 0.12
        )
        torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)

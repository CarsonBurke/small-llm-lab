from __future__ import annotations

import pytest
import torch

from scripts.eval_byte_duo_time_buckets import _bucket_index


def test_bucket_index_covers_endpoints_and_internal_boundaries() -> None:
    times = torch.tensor([0.0, 0.124, 0.125, 0.999, 1.0])

    torch.testing.assert_close(
        _bucket_index(times, 8),
        torch.tensor([0, 0, 1, 7, 7]),
    )


@pytest.mark.parametrize(
    ("times", "buckets", "message"),
    [
        (torch.tensor([0.5]), 0, "positive"),
        (torch.tensor([[0.5]]), 8, "vector"),
        (torch.tensor([1]), 8, "floating-point"),
    ],
)
def test_bucket_index_rejects_invalid_inputs(
    times: torch.Tensor, buckets: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _bucket_index(times, buckets)

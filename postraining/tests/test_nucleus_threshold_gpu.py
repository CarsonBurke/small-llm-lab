"""``nucleus_threshold``'s Triton search == the top-p rule, decided in fp64.

Run only through mlq: .venv/bin/python -m pytest -q this_file.
"""

import pytest
import torch

from postraining.latent_rollout import counter_gumbel_tokens, nucleus_mask
from postraining.nucleus_threshold import nucleus_threshold

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires queued CUDA"
)

VOCAB = 50304


def _mass_above(scores: torch.Tensor) -> torch.Tensor:
    """fp64 probability mass strictly above each token's own score."""
    probabilities = scores.double().softmax(dim=-1)
    ranked, order = scores.sort(dim=-1, descending=True)
    ranked_mass = probabilities.gather(-1, order)
    inclusive = ranked_mass.cumsum(dim=-1)
    # Mass strictly above = inclusive mass of the last strictly larger value.
    first_of_value = torch.searchsorted(
        (-ranked).contiguous(), (-ranked).contiguous(), right=False
    )
    strictly_above = torch.where(
        first_of_value > 0,
        inclusive.gather(-1, (first_of_value - 1).clamp_min(0)),
        torch.zeros_like(inclusive),
    )
    return torch.empty_like(strictly_above).scatter_(-1, order, strictly_above)


def _assert_is_the_nucleus(logits, temperature, top_p):
    scores = logits.float()
    if temperature != 1.0:
        scores = scores / temperature
    kept = nucleus_mask(logits, temperature, top_p)
    above = _mass_above(scores)
    exact = above <= top_p
    # The two differ only where fp32 summation cannot resolve the boundary.
    differs = kept != exact
    assert not differs[(above - top_p).abs() > 1e-5].any()
    assert kept.any(dim=-1).all()
    return kept


def _logits(rows, kind, seed, dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(rows, VOCAB, device="cuda", generator=generator)
    if kind == "lm":
        logits = base * 3.0
    elif kind == "peaked":
        logits = base.clone()
        logits[:, 17] = 40.0
    elif kind == "flat":
        logits = torch.zeros_like(base)
    elif kind == "coarse":
        # Few distinct values: heavy ties at every boundary.
        logits = (base * 2.0).round()
    elif kind == "masked":
        logits = base * 3.0
        logits[:, VOCAB // 2 :] = -torch.inf
    elif kind == "signed_zeros":
        # +0 and -0 are one score; key-space search must keep them tied.
        logits = torch.zeros_like(base)
        logits[:, ::2] = -0.0
        logits[:, :5] = 1.0
    elif kind == "extreme":
        # Scores spanning the whole fp32 range: the value-space cuts overflow
        # to inf and the key-space passes start from a ~2**32-key bracket.
        logits = base * 3.0
        logits[:, 0] = torch.finfo(torch.float32).max
        logits[:, 1] = -torch.finfo(torch.float32).max
        logits[:, 2:9] = torch.finfo(torch.float32).max / 2
        logits[:, 9] = 1e-40  # denormal
    elif kind == "mixed":
        # Rows that close at different passes share every launch.
        logits = base * 3.0
        logits[0::4] = 0.0
        logits[1::4, 17] = 40.0
        logits[2::4] = (base[2::4] * 2.0).round()
    elif kind == "masked_head":
        # Whole leading chunks of -inf: the online normaliser starts there.
        logits = base * 3.0
        logits[:, : VOCAB // 2] = -torch.inf
    else:
        raise AssertionError(kind)
    return logits.to(dtype)


@pytest.mark.parametrize(
    "kind",
    [
        "lm", "peaked", "flat", "coarse", "masked", "masked_head",
        "signed_zeros", "mixed",
    ],
)
@pytest.mark.parametrize("top_p", [0.1, 0.7, 0.95])
def test_threshold_is_the_top_p_nucleus(kind, top_p):
    _assert_is_the_nucleus(_logits(64, kind, seed=3), 1.0, top_p)


@pytest.mark.parametrize("top_p", [0.1, 0.7, 0.95])
def test_threshold_spans_the_whole_fp32_range(top_p):
    """fp32 only: bf16 would round FLT_MAX to inf, which is not a score."""
    logits = _logits(16, "extreme", seed=4, dtype=torch.float32)
    kept = nucleus_mask(logits, 1.0, top_p)
    # The FLT_MAX token holds all the mass: it alone is the nucleus.
    assert kept.sum(dim=-1).eq(1).all()
    assert kept[:, 0].all()
    spread = _logits(16, "lm", seed=4, dtype=torch.float32) * 1e29
    _assert_is_the_nucleus(spread, 1.0, top_p)


def test_threshold_under_a_symbolic_row_count():
    compiled = torch.compile(nucleus_threshold, fullgraph=True, dynamic=True)
    for rows in (5, 13, 300):
        logits = _logits(rows, "lm", seed=rows)
        torch.testing.assert_close(
            compiled(logits, 0.7), nucleus_threshold(logits, 0.7), rtol=0, atol=0
        )


@pytest.mark.parametrize("temperature", [0.6, 1.3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mask_is_the_nucleus_at_any_temperature(temperature, dtype):
    _assert_is_the_nucleus(_logits(32, "lm", seed=5, dtype=dtype), temperature, 0.7)


@pytest.mark.parametrize("temperature", [0.6, 1.3])
def test_compiled_mask_thresholds_the_scores_it_compares(temperature):
    """Inductor rewrites ``x / T`` as ``x * (1 / T)``; the threshold must come
    from those scores, or a peaked row's only nucleus token drops out."""
    logits = _logits(64, "peaked", seed=7)
    compiled = torch.compile(nucleus_mask, fullgraph=True, dynamic=False)
    kept = compiled(logits, temperature, 0.7)
    assert kept.sum(dim=-1).eq(1).all()
    assert kept[:, 17].all()


@pytest.mark.parametrize("rows", [1, 13, 1024])
def test_threshold_matches_the_sorted_mask_at_decode_shapes(rows):
    logits = _logits(rows, "lm", seed=rows)
    kept = _assert_is_the_nucleus(logits, 1.0, 0.7)
    sorted_rule = nucleus_mask(logits.cpu(), 1.0, 0.7).cuda()
    # CPU sort and GPU search agree except at rounding-level boundary ties.
    assert (kept != sorted_rule).sum() <= rows


def test_threshold_reads_row_strided_logits():
    padded = _logits(8, "lm", seed=11).repeat(1, 2)
    logits = padded[:, :VOCAB]
    assert logits.stride(0) == 2 * VOCAB
    torch.testing.assert_close(
        nucleus_threshold(logits, 0.7),
        nucleus_threshold(logits.contiguous(), 0.7),
        rtol=0,
        atol=0,
    )


def test_threshold_traces_into_a_compiled_graphed_sampler():
    logits = _logits(256, "lm", seed=9)
    seeds = torch.arange(256, device="cuda", dtype=torch.long) * 7919
    slot = torch.tensor(5, device="cuda")

    def draw(logits, seeds, slot):
        return counter_gumbel_tokens(logits, seeds, slot, 1.0, 0.7)

    compiled = torch.compile(draw, fullgraph=True, dynamic=False, mode="reduce-overhead")
    eager = draw(logits, seeds, slot)
    for _ in range(3):
        graphed = compiled(logits, seeds, slot)
    assert torch.equal(graphed, eager)
    kept = _assert_is_the_nucleus(logits, 1.0, 0.7)
    assert kept.gather(-1, eager[:, None]).all()

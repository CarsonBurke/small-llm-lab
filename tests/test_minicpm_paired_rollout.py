"""CPU-only categorical/address/lifecycle contracts; never emulate CUDA RNG."""
from types import SimpleNamespace

import pytest
import torch

from postraining import minicpm_latent_rollout as rollout
from postraining.minicpm_paired_rollout import (
    PairedControllerRolloutEngine, _fixed_label_categorical,
)


def test_near_tied_probability_rank_swaps_keep_common_uniform_token_coupling():
    logits = torch.tensor([[0.0, 0.0001], [0.0001, 0.0]]).repeat(2, 1)
    tokens = _fixed_label_categorical(
        logits, torch.tensor([0.25, 0.25, 0.75, 0.75]),
        temperature=0.9, top_k=2, top_p=0.95,
    )
    assert tokens.tolist() == [0, 0, 1, 1]


def test_nucleus_filtering_precedes_token_label_ordering():
    # The largest token alone crosses the nucleus threshold. Sorting labels
    # before filtering would incorrectly admit lower-ID, lower-probability 0.
    logits = torch.tensor([0.25, 0.55, 0.15, 0.05]).log().expand(3, -1)
    tokens = _fixed_label_categorical(
        logits, torch.tensor([0.0, 0.4, 0.99999994]),
        temperature=1.0, top_k=3, top_p=0.5,
    )
    assert tokens.tolist() == [1, 1, 1]


def test_filtered_categorical_retains_probability_mass_in_token_order():
    # Top-k removes token 1, then nucleus removes token 3. The surviving mass
    # is 0.4:0.35 = 8:7; a midpoint uniform grid integrates these bins exactly.
    logits = torch.tensor([0.4, 0.05, 0.35, 0.2]).log().expand(300, -1)
    tokens = _fixed_label_categorical(
        logits, (torch.arange(300) + 0.5) / 300,
        temperature=1.0, top_k=3, top_p=0.7,
    )
    assert torch.bincount(tokens, minlength=4).tolist() == [160, 0, 140, 0]


def test_inverse_cdf_skips_leading_internal_and_trailing_zero_mass():
    # Underflow and excluded logits give zero mass at IDs 0,2,4,5. The exact
    # half-open boundary belongs to token 3, never a zero-probability plateau.
    logits = torch.tensor([-1000.0, 0.0, -torch.inf, 0.0, -1000.0, -torch.inf]).expand(4, -1)
    tokens = _fixed_label_categorical(
        logits, torch.tensor([0.0, 0.499, 0.5, 0.99999994]),
        temperature=1.0, top_k=6, top_p=1.0,
    )
    assert tokens.tolist() == [1, 1, 3, 3]


@pytest.fixture
def host_engine(monkeypatch):
    # Bypass model construction, not RNG implementation. Only host scheduling,
    # metadata tensors and the stock/paired mode boundary are exercised here.
    engine = object.__new__(PairedControllerRolloutEngine)
    engine.prompts_per_rollout = 4
    engine.samples_per_prompt = 16
    engine.batch_size = 64
    engine.cache_length = 11024
    engine.answer_reserve_tokens = 1024
    engine.pair_samples = True
    engine.counter_seed = 17
    engine.coupling_seed = torch.tensor(17, dtype=torch.int64)
    engine.thought_indices = torch.full((64,), -1, dtype=torch.int64)
    engine.answer_indices = torch.full_like(engine.thought_indices, -1)
    engine.lane_prompt_lengths = torch.zeros_like(engine.thought_indices)
    engine.admission_count = 0
    engine._generating = False
    engine._stock_sampling = False
    engine._stock_heads = (lambda: "stock", lambda: "stock", lambda: "stock")
    engine._thought, engine._gate, engine._answer = (
        lambda: "paired", lambda: "paired", lambda: "paired",
    )
    engine._decoder = SimpleNamespace(
        lengths=torch.zeros(64, dtype=torch.int64),
        admit=lambda lanes, prompts: tuple(prompts),
    )
    engine.release_cache = lambda: None

    def generate(self, prompt_ids, budget, progress_callback=None):
        count = len(prompt_ids) * self.samples_per_prompt
        admissions = 0
        responses = []
        for start in range(0, count, self.batch_size):
            rows = list(range(start, min(count, start + self.batch_size)))
            self._decoder.admit(list(range(len(rows))), [row // self.samples_per_prompt for row in rows])
            admissions += 1
            responses.extend(self._thought() for _ in rows)
        if progress_callback is not None:
            progress_callback(1, {})
        return SimpleNamespace(responses=tuple(responses), admission_events=admissions)

    monkeypatch.setattr(rollout.MiniCPMLatentRolloutEngine, "generate_prompt_pool", generate)
    return engine


def test_repeated_training_generation_resets_admission_and_prompt_positions(host_engine):
    engine = host_engine
    first = engine.generate_prompt_pool([torch.ones(n, dtype=torch.long) for n in (1, 2, 3, 4)], 10000)
    second = engine.generate_prompt_pool([torch.ones(n, dtype=torch.long) for n in (5, 6, 7, 8)], 10000)
    assert first.responses == second.responses == ("paired",) * 64
    # Address observations after the second prompt batch must use its lengths,
    # not the old batch's lengths. Reordering and padding must not change keys.
    engine._decoder.lengths.copy_(torch.tensor([n for n in (5, 6, 7, 8) for _ in range(16)]) + 3)
    _, keys, positions = engine._counters(torch.tensor([33, 32, -1, 64, 1, 0]), 6)
    assert keys.tolist() == [16, 16, -1, -1, 0, 0]
    assert positions.tolist() == [3, 3, -1, -1, 3, 3]
    engine.pair_samples = False
    _, keys, positions = engine._counters(torch.tensor([33, 32, -1, 64, 1, 0]), 6)
    assert keys.tolist() == [33, 32, -1, -1, 1, 0]


def test_stock_eval_allows_refill_and_restores_paired_mode_after_nested_failure(host_engine):
    engine = host_engine
    prompts = [torch.ones(1, dtype=torch.long)] * 4
    assert engine.generate_prompt_pool(prompts, 10000).responses == ("paired",) * 64
    with engine.stock_sampling():
        engine.prompts_per_rollout, engine.samples_per_prompt, engine.batch_size = 2, 3, 4
        try:
            with pytest.raises(RuntimeError, match="heldout failed"):
                with engine.stock_sampling():
                    def fail(*unused):
                        raise RuntimeError("heldout failed")
                    engine.generate_prompt_pool(prompts[:2], 2048, fail)
            result = engine.generate_prompt_pool(prompts[:2], 2048)
            assert result.responses == ("stock",) * 6
            assert result.admission_events == 2
        finally:
            engine.prompts_per_rollout, engine.samples_per_prompt, engine.batch_size = 4, 16, 64
    assert engine.generate_prompt_pool(prompts, 10000).responses == ("paired",) * 64


def test_failed_paired_generation_restores_global_scheduler_and_decoder(host_engine):
    engine = host_engine
    original_chunks = rollout._LatentChunks
    prompts = [torch.ones(1, dtype=torch.long)] * 4

    def illegal_refill(*unused):
        engine._decoder.admit([0], [0])

    with pytest.raises(ValueError, match="forbids refill"):
        engine.generate_prompt_pool(prompts, 10000, illegal_refill)
    assert rollout._LatentChunks is original_chunks
    # The ordinary decoder must work for small admissions after unwinding.
    assert engine._decoder.admit([0], [2]) == (2,)
    assert engine.generate_prompt_pool(prompts, 10000).responses == ("paired",) * 64


def test_stock_generation_never_draws_a_counter_seed(host_engine):
    engine = host_engine

    class ForbiddenCounterDraw:
        def random_(self):
            raise AssertionError("heldout evaluation consumed the training counter seed")

    engine.counter_seed = None
    engine.coupling_seed = ForbiddenCounterDraw()
    with engine.stock_sampling():
        result = engine.generate_prompt_pool([torch.ones(1, dtype=torch.long)] * 4, 10000)
    assert result.responses == ("stock",) * 64


def test_training_rejects_non_b64_pool_before_any_admission(host_engine):
    engine = host_engine
    engine.samples_per_prompt = 8
    with pytest.raises(ValueError, match="exact B64"):
        engine.generate_prompt_pool([torch.ones(1, dtype=torch.long)] * 4, 10000)
    # The rejection must not leave the instance stuck in an active generation.
    with engine.stock_sampling():
        assert engine.generate_prompt_pool([torch.ones(1, dtype=torch.long)] * 4, 10000).responses == ("stock",) * 32

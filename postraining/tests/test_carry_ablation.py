"""Carry-ablation evaluation: paired stats, arm isolation, count plumbing."""

import pytest
import torch

import postraining.latent_eval as latent_eval
from postraining.carry_ablation_eval import (
    hidden_probes,
    paired_prompt_stats,
    zeroed_carry,
)
from postraining.latent_eval import evaluate_latent_math
from postraining.tests.test_latent_rollout import _critic, _wrapper


def test_paired_prompt_stats_null_and_signal():
    null = paired_prompt_stats([2, 0, 5, 3], [2, 0, 5, 3], 8, seed=0)
    assert null["mean_delta"] == 0.0
    assert null["bootstrap_ci_low"] <= 0.0 <= null["bootstrap_ci_high"]
    assert null["permutation_p"] == pytest.approx(1.0)
    assert null["prompts_tied"] == 4

    counts_a = [6] * 40
    counts_b = [2] * 40
    signal = paired_prompt_stats(counts_a, counts_b, 8, seed=0)
    assert signal["mean_delta"] == pytest.approx(0.5)
    assert signal["bootstrap_ci_low"] > 0.0
    assert signal["permutation_p"] < 0.01
    assert signal["prompts_improved"] == 40

    with pytest.raises(ValueError, match="identical prompt panels"):
        paired_prompt_stats([1, 2], [1], 8, seed=0)
    with pytest.raises(ValueError, match="samples must be positive"):
        paired_prompt_stats([1], [1], 0, seed=0)


def test_zeroed_carry_restores_exactly_and_kills_only_content():
    wrapper = _wrapper()
    combiner = wrapper.combiner
    with torch.no_grad():
        combiner.carry.weight.normal_(std=0.05)
        combiner.type_bias.normal_(std=0.02)
    original_weight = combiner.carry.weight.detach().clone()
    dim = combiner.carry.weight.size(0)
    base = torch.randn(2, 3, dim)
    hidden = torch.randn(2, 3, dim)
    flag = torch.ones(2, 3, dtype=torch.bool)

    with torch.no_grad():
        full = combiner(base, hidden, flag)
        with zeroed_carry(wrapper):
            ablated = combiner(base, hidden, flag)
            # Content off: the hidden no longer matters at all...
            assert torch.equal(
                ablated, combiner(base, torch.randn_like(hidden), flag)
            )
        # ...but type_bias/MLP still fire, so this is not the plain path.
        assert not torch.equal(ablated, full)
        assert not torch.equal(
            ablated, combiner(base, hidden, torch.zeros_like(flag))
        )
        restored = combiner(base, hidden, flag)
    assert torch.equal(combiner.carry.weight, original_weight)
    assert torch.equal(restored, full)


def test_zeroed_carry_restores_on_error():
    wrapper = _wrapper()
    with torch.no_grad():
        wrapper.combiner.carry.weight.normal_(std=0.05)
    original = wrapper.combiner.carry.weight.detach().clone()
    with pytest.raises(RuntimeError, match="boom"):
        with zeroed_carry(wrapper):
            raise RuntimeError("boom")
    assert torch.equal(wrapper.combiner.carry.weight, original)


class _ProbeTokenizer:
    def eos_id(self) -> int:
        return 5

    def bos_id(self) -> int:
        return -1

    def encode(self, text: str) -> list[int]:
        return [1 + (ord(ch) % 20) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "Answer: 42"


_PROBE_SAVED_ARGS = {
    "resolved_train_max_new_tokens": 4,
    "resolved_train_max_stream_steps": 5,
    "prompt_tokens": 6,
    "nearby_reward_max": 0.1,
    "temperature": 1.0,
    "top_p": 1.0,
    "replay_max_trajectories": 8,
    "replay_attention_budget": 1 << 20,
    "replay_bucket": 1,
    "replay_slot_budget": 4096,
}

_PROBE_ROWS = [
    {
        "prompt": [{"content": text}],
        "reward_model": {"ground_truth": "42"},
    }
    for text in ("what is 6*7", "sum 40 and 2")
]


def test_hidden_probes_retain_raw_action_signal_with_zero_carry():
    wrapper = _wrapper()
    critic = _critic()
    with torch.no_grad():
        wrapper.combiner.carry.weight.normal_(std=0.05)
        critic.combiner.carry.weight.normal_(std=0.05)
        # Fresh nano readouts are zero-initialized, which makes every token
        # distribution uniform no matter the input — a live readout is what
        # lets a raw-thought change move the replayed logprobs at all.
        wrapper.backbone.policy_probe.output.weight.normal_(std=0.02)

    probes = hidden_probes(
        wrapper, critic, _ProbeTokenizer(), _PROBE_ROWS, samples=2,
        saved_args=_PROBE_SAVED_ARGS, seed=9,
        device=torch.device("cpu"), stop_ids=(5,),
    )
    assert probes["action_slots"] > 0
    assert probes["trajectories"] == 4
    assert probes["carried_hidden_rms"] > 0.0
    assert probes["injection_to_embedding_rms_ratio"] > 0.0
    # A live carry matrix means zeroing the stored raw thoughts must move the
    # replayed logprobs and critic values somewhere.
    assert probes["token_logprob_delta_abs_max"] > 0.0
    assert probes["critic_value_delta_abs_max"] > 0.0

    with torch.no_grad():
        wrapper.combiner.carry.weight.zero_()
        critic.combiner.carry.weight.zero_()
    nulled = hidden_probes(
        wrapper, critic, _ProbeTokenizer(), _PROBE_ROWS, samples=2,
        saved_args=_PROBE_SAVED_ARGS, seed=9,
        device=torch.device("cpu"), stop_ids=(5,),
    )
    # The raw thought is also the base input to the thought adapter. Disabling
    # the carry matrix removes only its residual, not the direct action path.
    assert nulled["token_logprob_delta_abs_max"] > 0.0
    assert nulled["critic_value_delta_abs_max"] > 0.0


def test_prompt_correct_counts_keep_original_row_order(monkeypatch):
    wrapper = _wrapper()

    class _Tokenizer:
        def eos_id(self) -> int:
            return 5

        def bos_id(self) -> int:
            return -1

        def encode(self, text: str) -> list[int]:
            return list(range(1, len(text) + 1))

        def decode(self, ids: list[int]) -> str:
            return "Answer: 42"

    # Row 0 is the LONGER prompt: length bucketing will evaluate it after
    # row 1, so identical count order would betray compute order, not
    # dataset order.
    rows = [
        {
            "prompt": [{"content": "abcdef"}],
            "reward_model": {"ground_truth": "42"},
        },
        {
            "prompt": [{"content": "a"}],
            "reward_model": {"ground_truth": "7"},
        },
    ]

    def _by_truth(
        emitted, truth, tokenizer, stop_ids, style, prefix_ids=(),
        answer_fence_ids=None,
    ):
        return truth == "42", truth

    monkeypatch.setattr(latent_eval, "verify_terminated_answer", _by_truth)
    metrics = evaluate_latent_math(
        wrapper, _Tokenizer(), rows, samples=3, max_new_tokens=2,
        max_stream_steps=3, chunk=3, seed=5, device=torch.device("cpu"),
        prompt_tokens=8, batch_trajectories=4,
    )
    assert metrics["prompt_correct_counts"] == [3, 0]
    assert sum(metrics["prompt_correct_counts"]) == int(
        round(metrics["accuracy"] * metrics["samples"])
    )

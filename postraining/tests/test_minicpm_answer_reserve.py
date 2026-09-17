from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from postraining.train_minicpm_vapo import (
    _build_group_records,
    _validate_args,
    build_parser,
    resolve_thinking_end_token,
    validate_resume_configuration,
)
from scripts.preflight_minicpm_vapo import _saved_trainer_options


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert text == "</think>"
        assert not add_special_tokens
        return [9]

    def decode(self, ids, *, skip_special_tokens):
        return "".join({20: "reason ", 9: "", 30: "Answer: ", 31: "42", 2: ""}[token] for token in ids)


def test_forced_close_is_retained_and_final_answer_is_scored():
    responses = torch.tensor([
        [20] * 8 + [9, 30, 31, 2],
        [20, 9, 20, 20, 20, 20, 20, 20, 20, 30, 31, 2],
    ])
    records, token_count = _build_group_records(
        Tokenizer(),
        {"reward_model": {"ground_truth": "42", "style": "rule"}},
        torch.tensor([1]),
        responses,
        torch.zeros_like(responses, dtype=torch.float32),
        torch.zeros_like(responses, dtype=torch.float32),
        samples_per_prompt=2,
        stop_ids=(2,),
        thinking_end_token_id=9,
        forced_thinking_position=8,
    )
    assert token_count == 24
    assert [record.correct for record in records] == [True, True]
    assert [record.forced_token_index for record in records] == [8, -1]
    assert records[0].token_ids.tolist() == [1] + responses[0].tolist()
    assert records[0].response_length == 12


def test_missing_budget_close_is_rejected_before_training():
    response = torch.tensor([[20] * 9 + [30, 31, 2]])
    with pytest.raises(RuntimeError, match="thinking"):
        _build_group_records(
            Tokenizer(),
            {"reward_model": {"ground_truth": "42", "style": "rule"}},
            torch.tensor([1]),
            response,
            torch.zeros_like(response, dtype=torch.float32),
            torch.zeros_like(response, dtype=torch.float32),
            samples_per_prompt=1,
            stop_ids=(2,),
            thinking_end_token_id=9,
            forced_thinking_position=8,
        )


def test_answer_reserve_changes_require_completed_rollout_boundary():
    args = build_parser().parse_args([])
    prior = vars(args).copy()
    prior.pop("answer_reserve_tokens")
    resume = {"args": prior, "pending_records": [object()]}
    with pytest.raises(ValueError, match="answer_reserve_tokens"):
        validate_resume_configuration(resume, args)
    args.answer_reserve_tokens = 0
    validate_resume_configuration(resume, args)
    args.answer_reserve_tokens = 1000
    resume["pending_records"] = None
    validate_resume_configuration(resume, args)


def test_preflight_does_not_enable_forcing_on_legacy_pending_rollouts():
    args = build_parser().parse_args([])
    saved = vars(args).copy()
    saved.pop("answer_reserve_tokens")
    restored = build_parser().parse_args(_saved_trainer_options(saved))
    assert restored.answer_reserve_tokens == 0


@pytest.mark.parametrize("reserve", [-1, 9999, 10000])
def test_reserve_requires_space_for_thinking_and_delimiter(reserve):
    args = build_parser().parse_args(["--answer-reserve-tokens", str(reserve)])
    with pytest.raises(ValueError, match="reserve"):
        _validate_args(args)


def test_thinking_end_uses_native_single_token_and_cannot_be_eos():
    assert resolve_thinking_end_token(Tokenizer(), stop_ids=(2,)) == 9
    with pytest.raises(ValueError, match="stop"):
        resolve_thinking_end_token(Tokenizer(), stop_ids=(9,))
    multi = SimpleNamespace(encode=lambda *args, **kwargs: [10, 11])
    with pytest.raises(ValueError, match="single"):
        resolve_thinking_end_token(multi, stop_ids=(2,))

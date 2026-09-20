from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from postraining import fast_inference
from postraining.fast_inference import CapturedTrainingRolloutEngine, PromptPrefixBank
from postraining.minicpm_vapo import MiniCPMVAPOPolicy
from postraining.nextlat_speculative import NextLatSpeculativeEngine
from postraining.thinking_budget import force_thinking_end_, validate_thinking_budget


class _TokenPolicy(nn.Module):
    """An exact finite-state decoder; answers differ after the closing token."""

    def __init__(self) -> None:
        super().__init__()
        self.causal_lm = nn.Module()
        self.causal_lm.register_parameter(
            "anchor", nn.Parameter(torch.zeros(()), requires_grad=False)
        )
        self.causal_lm.config = SimpleNamespace(
            num_hidden_layers=1, num_key_value_heads=1, head_dim=1,
            pad_token_id=0, vocab_size=10,
        )
        self.transitions = torch.tensor([1, 1, 2, 9, 8, 1, 1, 1, 8, 2])

    def cached_hidden(self, input_ids, **kwargs):
        return torch.nn.functional.one_hot(input_ids, num_classes=10).float()

    def logits(self, hidden):
        target = self.transitions[hidden.argmax(-1)]
        return torch.full((*target.shape, 10), -100.0).scatter_(
            -1, target[..., None], 100.0
        )

    def rollout_values(self, hidden):
        return hidden.argmax(-1).float()

    def nextlat_hidden(self, hidden, token):
        return self.cached_hidden(token)


@pytest.mark.parametrize("reserve,token,limit", [(-1, 9, 6), (1, None, 6), (1, -1, 6), (2, 9, 3)])
def test_invalid_thinking_budgets_fail_before_decode(reserve, token, limit) -> None:
    with pytest.raises(ValueError):
        validate_thinking_budget(reserve, token, limit)


def test_budget_only_updates_active_lanes_and_reserves_tokens_after_delimiter() -> None:
    closed = torch.tensor([False, True, False, False])
    tokens, forced = force_thinking_end_(
        torch.tensor([1, 2, 8, 9]), torch.tensor([8999, 8999, 8998, 8999]),
        closed, torch.tensor([True, True, True, False]), 8999, 9,
    )
    assert tokens.tolist() == [9, 2, 8, 9]
    assert forced.tolist() == [True, False, False, False]
    assert closed.tolist() == [True, True, False, False]


@pytest.mark.parametrize("reserve", [1, 2, 3, 4])
def test_nextlat_recomputes_answer_after_forcing_across_commit_boundaries(reserve) -> None:
    engine = NextLatSpeculativeEngine(
        cast(MiniCPMVAPOPolicy, _TokenPolicy()), stop_ids=(8,),
        prompts_per_rollout=3, samples_per_prompt=1, cache_length=12,
        draft_length=2, temperature=1.0, top_p=1.0, compile_decode=False,
        answer_reserve_tokens=reserve, thinking_end_token_id=9,
    )
    boundary = 6 - reserve - 1
    # A repeated call must not inherit the previous batch's natural/forced close.
    for _ in range(2):
        responses, _, _, _, _, _ = engine.generate_prompts(
            [torch.tensor([0]), torch.tensor([3]), torch.tensor([4])], max_new_tokens=6
        )
        assert responses.tolist() == [
            [1] * boundary + [9] + [2] * reserve,
            [9, 2, 2, 2, 2, 2],
            [8, 8, 8, 8, 8, 8],
        ]


@pytest.mark.parametrize("mode", ["statistics", "continuous", "invariant"])
@torch.inference_mode()
def test_captured_steps_force_before_model_forward_and_reset_on_refill(monkeypatch, mode) -> None:
    policy = _TokenPolicy()
    monkeypatch.setattr(fast_inference, "build_fused_rollout_replica", lambda source: (source, ()))
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: None)
    # Exercise the actual tensor decode closures on CPU, not compilation or a model workload.
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    monkeypatch.setattr(fast_inference, "compile_invariant", lambda function: function)
    monkeypatch.setattr(fast_inference, "install_invariant_linears", lambda model: ())
    engine = CapturedTrainingRolloutEngine(
        cast(MiniCPMVAPOPolicy, policy), stop_ids=(8,), prompts_per_rollout=3,
        samples_per_prompt=1, cache_length=12, temperature=1.0, top_k=1, top_p=1.0,
        compile_decode=mode == "invariant", invariant_decode=mode == "invariant",
        answer_reserve_tokens=2, thinking_end_token_id=9,
    )
    prompt_tokens = torch.tensor([0, 0, 4])
    engine._graph_logits.copy_(policy.logits(policy.cached_hidden(prompt_tokens)))
    engine._graph_values.copy_(prompt_tokens.float())
    engine.response_limit.copy_(torch.tensor([6, 5, 4]))
    engine.active.fill_(True)
    if mode == "invariant":
        engine._pending.copy_(prompt_tokens)
    for _ in range(6):
        if mode == "statistics":
            engine._split_decode_step()
        else:
            engine._continuous_split_decode_step()
    assert engine.generated[:, :6].tolist() == [
        [1, 1, 1, 9, 2, 2], [1, 1, 9, 2, 2, 0], [8, 0, 0, 0, 0, 0],
    ]
    assert engine.output_position.tolist() == [6, 5, 1]
    if mode == "statistics":
        # The forced token remains a real context/value step, not a terminal.
        assert engine.values[0, 4:6].tolist() == [9.0, 2.0]
        assert engine.logprobs[0, 3].item() == -200.0

    engine.cache = SimpleNamespace(layers=[])
    bank = PromptPrefixBank(
        lengths=torch.tensor([1]),
        logits=policy.logits(policy.cached_hidden(torch.tensor([0]))).to(
            engine._graph_logits.dtype
        ),
        values=torch.empty(0), layer_keys=torch.empty(1, 1, 0, 0, 0),
        layer_values=torch.empty(1, 1, 0, 0, 0),
    )
    if mode == "invariant":
        engine._prompt_last_tokens = torch.tensor([0])
    engine._admit_prompt_rows(bank, 0, [0], max_new_tokens=4)
    assert engine.thinking_closed.tolist() == [False, True, False]
    for _ in range(6):
        engine._continuous_split_decode_step()
    assert engine.generated[0, :6].tolist() == [1, 9, 2, 2, 0, 0]
    assert engine.output_position.tolist() == [4, 5, 1]

"""Code partial credit reaches terminal rewards without becoming exact success."""
from types import SimpleNamespace

import pytest
import torch

from postraining.latent_rollout import TOKEN_SLOT
from postraining.train_latent_vapo import score_python_rollout
from postraining.vapo.code_reward import PYTHON_REWARD_SCHEMA, PYTHON_RESULT_CODES
from postraining.vapo.config import build_arg_parser


class CodeTokenizer:
    def decode(self, tokens):
        assert tokens == [14]
        return "def f(x): return 0"


def batch():
    tokens = torch.tensor([[99, 10, 11, 12, 13, 14, 15, 16]])
    return SimpleNamespace(
        token_ids=tokens, kind=torch.full_like(tokens, TOKEN_SLOT),
        prompt_length=1, rewards=torch.zeros_like(tokens, dtype=torch.float32),
        reward_scalar=torch.zeros(1), action_mask=torch.ones_like(tokens),
    )


@pytest.mark.parametrize("mode, expected", [("binary", 0.0), ("test-fraction", 0.5)])
def test_partial_test_credit_reaches_terminal_reward(mode, expected):
    rollout = batch()
    verdicts = score_python_rollout(
        rollout,
        {"schema": PYTHON_REWARD_SCHEMA, "entry_points": ["f"],
         "tests": ["assert f(1) == 1", "assert f(0) == 0"]},
        CodeTokenizer(), (16,), (10, 12), (13, 15), 1, reward_mode=mode,
    )
    assert rollout.reward_scalar.item() == expected
    assert rollout.rewards[0, -1].item() == expected
    assert rollout.rewards.sum().item() == expected
    assert rollout.verifier_status.item() == PYTHON_RESULT_CODES["tests_failed"]
    assert verdicts[0].format_ok
    if mode == "test-fraction":
        assert "1/2 tests passed" in verdicts[0].parsed_answer


def test_partial_credit_does_not_bypass_format_gate(monkeypatch):
    rollout = batch()
    rollout.token_ids[0, 1] = 11
    def unexpected(*args):
        pytest.fail("malformed response must not execute")
    monkeypatch.setattr("postraining.train_latent_vapo.batch_python_test_scores", unexpected)
    score_python_rollout(
        rollout, {"schema": PYTHON_REWARD_SCHEMA}, CodeTokenizer(),
        (16,), (10, 12), (13, 15), 1, reward_mode="test-fraction",
    )
    assert rollout.reward_scalar.item() == 0
    assert rollout.verifier_status.item() == PYTHON_RESULT_CODES["format_ineligible"]


def test_future_training_defaults_to_test_fraction():
    args = build_arg_parser().parse_args(["--checkpoint", "model.pt", "--output", "out"])
    assert args.python_reward_mode == "test-fraction"

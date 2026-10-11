"""Explicit abstention with a normalized, nonnegative terminal reward."""

from types import SimpleNamespace

import pytest
import torch

from postraining.core import structural_abstention, structural_format_ok
from postraining.latent_eval import verify_terminated_answer
from postraining.latent_rollout import PAD_SLOT, TOKEN_SLOT
from postraining.rollout_report import RowVerdict
from postraining.train_latent_vapo import (
    RESUME_EXACT_ARG_FIELDS,
    aggregate_diagnostics,
    abstention_terminal_scores,
    reward_schema_for_abstention,
    nearby_numeric_reward,
    rollout_tensorboard_metrics,
    score_math_rollout,
    score_python_rollout,
    validate_resume_arg_contract,
)
from postraining.vapo.code_reward import PYTHON_REWARD_SCHEMA, PYTHON_RESULT_CODES
from postraining.vapo.config import build_arg_parser, validate_args

EOS, THINK, ANSWER = 5, (7, 8), (20, 21)


class _Tokenizer:
    PIECES = {1: "hmm", 12: "42", 13: "37", 14: "Answer: 42", 15: "def f(x): return 0", 16: "\n", 17: "I cannot solve this."}

    def decode(self, ids: list[int]) -> str:
        return "".join(self.PIECES.get(token, "") for token in ids)

    def eos_id(self) -> int:
        return EOS


def _batch(rows: list[list[int]]) -> SimpleNamespace:
    """A scored rollout: one prompt token, then each row's emitted tokens."""
    width = 1 + max(map(len, rows))
    tokens = torch.zeros(len(rows), width, dtype=torch.long)
    kind = torch.full_like(tokens, PAD_SLOT)
    for index, row in enumerate(rows):
        tokens[index, 1 : 1 + len(row)] = torch.tensor(row)
        kind[index, : 1 + len(row)] = TOKEN_SLOT
    action_mask = (kind == TOKEN_SLOT).float()
    return SimpleNamespace(
        token_ids=tokens, kind=kind, prompt_length=1, action_mask=action_mask,
        rewards=torch.zeros_like(action_mask), reward_scalar=torch.zeros(len(rows)),
    )


CORRECT = [7, 1, 8, 20, 12, 21, EOS]
WRONG = [7, 1, 8, 20, 13, 21, EOS]
# The think holds the right value; an abstention must still not earn it.
ABSTAIN = [7, 14, 8, EOS]
OPEN_ANSWER_ABSTAIN = [7, 14, 8, 20, EOS]
EMPTY_ANSWER_ABSTAIN = [7, 14, 8, 20, 21, EOS]
TRUNCATED_ABSTAIN = [7, 1, 8, 20, 21]
UNFENCED = [7, 1, 8, 14, EOS]  # right value, no answer fence
TRAILING = [7, 1, 8, 20, 21, 1, EOS]
EXPLAINED_ABSTAIN = [7, 14, 8, 16, 17, 14, EOS]
UNFINISHED_ANSWER_ABSTAIN = [7, 14, 8, 20, 14, EOS]
ABSTENTIONS = [
    ABSTAIN, OPEN_ANSWER_ABSTAIN, EMPTY_ANSWER_ABSTAIN, UNFENCED,
    TRAILING, EXPLAINED_ABSTAIN, UNFINISHED_ANSWER_ABSTAIN,
]


def test_terminal_scores_keep_both_endpoints_exact():
    scores = [1.0, 0.0, 0.5, 0.0]
    abstained = [False, False, False, True]
    assert abstention_terminal_scores(scores, abstained, None) == scores
    # exact_accuracy tests reward == 1.0, so correct must stay exactly 1
    # for every penalty, not merely to rounding.
    for penalty in (0.1, 0.3, 0.7, 1 / 3):
        mapped = abstention_terminal_scores(scores, abstained, penalty)
        assert mapped[0] == 1.0
        assert mapped[1] == 0.0
        assert mapped[2] == 0.5
        assert mapped[3] == pytest.approx(penalty / (1 + penalty))
        assert mapped[1] < mapped[3] < mapped[0]
    with pytest.raises(ValueError):
        abstention_terminal_scores([1.0], [False, True], 0.1)


def test_abstention_requires_eos_after_a_valid_think_without_submitted_answer():
    visible = lambda row: row[: row.index(EOS) + 1] if EOS in row else row
    cases = {
        tuple(CORRECT): (True, False),
        tuple(ABSTAIN): (False, True),
        tuple(OPEN_ANSWER_ABSTAIN): (False, True),
        tuple(EMPTY_ANSWER_ABSTAIN): (False, True),
        tuple(TRUNCATED_ABSTAIN): (False, False),
        tuple(UNFENCED): (False, True),
        tuple(TRAILING): (False, True),
        tuple(EXPLAINED_ABSTAIN): (False, True),
        tuple(UNFINISHED_ANSWER_ABSTAIN): (False, True),
        (7, 1, 8, 21, EOS): (False, True),  # stray answer close
        (7, 1, 8, 1, EOS): (False, True),  # content after thinking
        (7, 20, 1, 8, EOS): (False, False),  # answer marker inside thinking
        (7, 1, EOS, 8, EOS): (False, False),  # first EOS was inside thinking
        (7, 1, 8, 20, 1, EOS): (False, True),  # unfinished answer
        (7, 1, 8, 20, 21, 21, EOS): (False, True),  # no submitted answer
        (7, 1, 8, 20, 12, 21, 17, EOS): (False, False),  # submitted, then explanation
        (7, 1, 8, 20, 13, 21, 20, 21, EOS): (False, False),  # cannot retract a wrong answer
        (7, 1, 8, 20, 13, 20, 21, EOS): (False, False),  # duplicate open cannot erase submission
        (7, 8, 20, 21, EOS): (False, False),  # empty think
        (20, 21, 7, 1, 8, EOS): (False, False),  # answer before think
        (EOS,): (False, False),  # immediate early stop
    }
    for row, (format_ok, abstain) in cases.items():
        tokens = visible(list(row))
        assert structural_format_ok(tokens, THINK, ANSWER) is format_ok, row
        assert structural_abstention(tokens, THINK, ANSWER, EOS) is abstain, row
    # The think floor applies to abstentions exactly as to answers.
    assert not structural_abstention(
        ABSTAIN, THINK, ANSWER, EOS, min_think_tokens=2
    )
    assert not structural_abstention(ABSTAIN[:-1], THINK, ANSWER, EOS)
    assert not structural_abstention(ABSTAIN[:-1] + [1], THINK, ANSWER, EOS)


def test_explained_abstention_preserves_the_33_token_think_floor():
    for count in (32, 33):
        row = [THINK[0], *([1] * count), THINK[1], 16, 17, EOS]
        assert structural_abstention(row, THINK, ANSWER, EOS, 33) is (count == 33)


def test_abstention_schema_changes_with_the_qualification_rule():
    assert reward_schema_for_abstention(1.0).endswith("normalized_eos_abstention/v4")


def _score_math(rows, penalty, nearby_reward_max=0.0):
    batch = _batch(rows)
    verdicts = score_math_rollout(
        batch, "42", _Tokenizer(), (EOS,), nearby_reward_max=nearby_reward_max,
        think_fence_ids=THINK, answer_fence_ids=ANSWER,
        abstention_penalty=penalty,
    )
    return batch, verdicts


def test_math_rollout_rewards_abstention_between_right_and_wrong():
    rows = [
        CORRECT, WRONG, ABSTAIN, OPEN_ANSWER_ABSTAIN,
        EMPTY_ANSWER_ABSTAIN, TRUNCATED_ABSTAIN, UNFENCED, TRAILING,
        EXPLAINED_ABSTAIN, UNFINISHED_ANSWER_ABSTAIN,
    ]
    batch, verdicts = _score_math(rows, 0.25)
    assert batch.reward_scalar.tolist() == pytest.approx(
        [1.0, 0.0, 0.2, 0.2, 0.2, 0.0, 0.2, 0.2, 0.2, 0.2]
    )
    # One terminal reward per row, at its last action.
    assert torch.equal(batch.rewards.sum(1), batch.reward_scalar)
    assert batch.verifier_status.tolist() == [
        PYTHON_RESULT_CODES[
            "abstained"
            if row in ABSTENTIONS
            else "not_applicable"
        ]
        for row in rows
    ]
    # The verdict reports structural_format_ok, which an empty span fails.
    assert verdicts[2] == RowVerdict(format_ok=False, parsed_answer=None)
    # A deliberate abstention is not "right value, wrong structure".
    assert batch.think_gate_zeroed_correct == 0

    unpenalized, _ = _score_math(rows, None)
    assert unpenalized.reward_scalar.tolist() == [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    assert unpenalized.verifier_status.tolist() == batch.verifier_status.tolist()


def test_math_rollout_partial_credit_keeps_its_place():
    partial = nearby_numeric_reward("37", "42", 0.1)
    assert 0.0 < partial < 0.1
    batch, _ = _score_math([WRONG], 0.2, nearby_reward_max=0.1)
    assert batch.reward_scalar.item() == pytest.approx(partial)


def test_submitted_wrong_answer_cannot_become_abstention_by_adding_text():
    batch, _ = _score_math([WRONG[:-1] + [17, EOS]], 1.0)
    assert batch.reward_scalar.item() == 0.0
    assert batch.verifier_status.item() == PYTHON_RESULT_CODES["not_applicable"]


def test_math_abstention_requires_the_answer_fence():
    with pytest.raises(ValueError, match="answer_fence_ids"):
        score_math_rollout(
            _batch([CORRECT]), "42", _Tokenizer(), (EOS,),
            think_fence_ids=THINK, abstention_penalty=0.1,
        )


def _score_python(row, penalty, monkeypatch, outcome="fail"):
    def run(answers, info):
        assert answers == ["def f(x): return 0"]
        return [outcome]

    monkeypatch.setattr("postraining.train_latent_vapo.batch_python_test_results", run)
    batch = _batch([row])
    verdicts = score_python_rollout(
        batch, {"schema": PYTHON_REWARD_SCHEMA}, _Tokenizer(), (EOS,),
        THINK, ANSWER, 1, reward_mode="binary", abstention_penalty=penalty,
    )
    return batch, verdicts


@pytest.mark.parametrize("row", ABSTENTIONS)
def test_python_abstention_scores_threshold_without_running_tests(row, monkeypatch):
    batch, verdicts = _score_python(row, 0.1, monkeypatch, outcome=None)
    assert batch.reward_scalar.item() == pytest.approx(0.1 / 1.1)
    assert batch.verifier_status.item() == PYTHON_RESULT_CODES["abstained"]
    assert verdicts[0] == RowVerdict(format_ok=False, parsed_answer=None)


@pytest.mark.parametrize(
    "row, outcome, expected, status",
    [
        ([7, 1, 8, 20, 15, 21, EOS], "pass", 1.0, "pass"),
        ([7, 1, 8, 20, 15, 21, EOS], "tests_failed", 0.0, "tests_failed"),
        (TRUNCATED_ABSTAIN, None, 0.0, "format_ineligible"),
    ],
)
def test_python_rows_map_to_the_penalized_range(
    row, outcome, expected, status, monkeypatch
):
    batch, _ = _score_python(row, 0.1, monkeypatch, outcome=outcome)
    assert batch.reward_scalar.item() == pytest.approx(expected)
    assert batch.verifier_status.item() == PYTHON_RESULT_CODES[status]


@pytest.mark.parametrize("row", ABSTENTIONS)
def test_evaluation_never_credits_an_abstention(row):
    # The plain-text fallback would grade the "Answer: 42" in the think.
    assert verify_terminated_answer(
        row, "42", _Tokenizer(), (EOS,), answer_fence_ids=ANSWER,
        think_fence_ids=THINK,
    ) == (False, "[NO ANSWER]")
    assert verify_terminated_answer(
        CORRECT, "42", _Tokenizer(), (EOS,), answer_fence_ids=ANSWER
    )[0]


def _parse(*extra):
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "c.pt", "--output", "out", *extra])
    validate_args(parser, args)
    return args


def test_penalty_is_opt_in_positive_and_needs_the_answer_fence():
    assert _parse().abstention_penalty is None
    assert _parse("--abstention-penalty", "0.1").abstention_penalty == 0.1
    for value in ("0", "-0.1", "nan", "inf"):
        with pytest.raises(SystemExit):
            _parse("--abstention-penalty", value)
    for value in ("1e-50", "1e8"):
        with pytest.raises(SystemExit):
            _parse("--abstention-penalty", value)
    with pytest.raises(SystemExit):
        _parse("--abstention-penalty", "0.1", "--no-answer-fence")
    # The source-success gate would silence the wrong-versus-abstain
    # contrast the penalty exists to teach.
    with pytest.raises(SystemExit):
        _parse(
            "--abstention-penalty", "0.1", "--source-success-actor-gate",
            "--rl-mixture-manifest", "m.json",
        )


def test_resume_rejects_a_changed_penalty():
    assert "abstention_penalty" in RESUME_EXACT_ARG_FIELDS
    current = _parse("--abstention-penalty", "0.1")
    saved = vars(current).copy()
    validate_resume_arg_contract(saved, current)
    for before in (None, 0.2):
        saved["abstention_penalty"] = before
        with pytest.raises(ValueError, match="abstention_penalty"):
            validate_resume_arg_contract(saved, current)


def test_rollout_dashboard_reports_abstention_only_when_measured():
    raw = {
        "reward_mean": 0.0, "within_group_reward_std": 0.0, "exact_accuracy": 0.0,
        "exact_within_group_reward_std": 0.0, "partial_reward_fraction": 0.0,
        "ended_fraction": 1.0, "actions_per_trajectory": 4.0,
    }
    assert "reward/abstained_fraction" not in rollout_tensorboard_metrics(raw)
    raw["verifier_abstained_fraction"] = 0.25
    assert rollout_tensorboard_metrics(raw)["reward/abstained_fraction"] == 0.25


def test_abstention_length_is_pooled_by_abstention_count(monkeypatch):
    import postraining.train_latent_vapo as trainer

    groups = iter([
        {
            "trajectories": 2,
            "abstained_emitted_tokens_sum": 50.0,
            "abstained_emitted_tokens_count": 1.0,
            "abstained_emitted_tokens_max": 50.0,
        },
        {
            "trajectories": 2,
            "abstained_emitted_tokens_sum": 120.0,
            "abstained_emitted_tokens_count": 3.0,
            "abstained_emitted_tokens_max": 60.0,
        },
    ])
    monkeypatch.setattr(trainer, "rollout_diagnostics", lambda *a, **k: next(groups))
    pooled = aggregate_diagnostics([None, None], samples_per_prompt=2)
    assert pooled["abstained_emitted_tokens_count"] == 4.0
    assert pooled["abstained_emitted_tokens_mean"] == 42.5
    assert pooled["abstained_emitted_tokens_max"] == 60.0

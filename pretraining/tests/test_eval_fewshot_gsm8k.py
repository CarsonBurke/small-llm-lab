"""Deterministic contracts of the few-shot GSM8K harness.

None of this touches a model, so it runs directly rather than through mlq.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from types import SimpleNamespace

from pretraining.eval_byte_diffusion_gsm8k import greedy_generate_bytes

from pretraining.eval_fewshot_gsm8k import (
    DAPO_PREAMBLE,
    PROMPT_FORMATS,
    build_prompt,
    extract_gold,
    extract_prediction,
    greedy_generate,
    normalize_number,
    select_exemplars,
    truncate_at_stop,
)

HARNESS = PROMPT_FORMATS["harness"]
BARE = PROMPT_FORMATS["bare"]
DAPO = PROMPT_FORMATS["dapo"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1,000", "1000"),
        ("72", "72"),
        ("-5.0", "-5"),
        ("3.5", "3.5"),
        ("0018", "18"),
        ("x", None),
    ],
)
def test_normalize_number_canonicalizes_or_refuses(text, expected):
    assert normalize_number(text) == expected


def test_normalize_number_keeps_integers_exact_above_float_precision():
    # Routing through float first would collapse these onto one value and
    # score a wrong answer as correct.
    assert normalize_number("12345678901234567") != normalize_number(
        "12345678901234568"
    )


def test_normalize_number_refuses_a_repetition_collapse_digit_run():
    assert normalize_number("9" * 309) == "9" * 309
    assert normalize_number("9" * 309 + ".5") is None


def test_prediction_takes_the_first_answer_and_gold_the_last():
    drifted = " The answer is 18.\n#### 18\nAnswer: 18\n#### 4"
    assert extract_prediction(drifted, HARNESS) == "18"
    assert extract_gold(drifted) == "4"


def test_digit_grouping_must_be_well_formed():
    # `#### 1,8` is two numbers the model ran together. Reading it as 1,8 ->
    # 18 would score as correct against a gold of 18.
    assert extract_prediction("#### 1,8", HARNESS) == "1"
    assert extract_prediction("#### 1,000", HARNESS) == "1000"


def test_extraction_refuses_a_window_with_no_delimiter():
    assert extract_prediction(" 2, 2, 2, 2, 2", HARNESS) is None


@pytest.mark.parametrize("stop", HARNESS.stops)
def test_truncate_cuts_at_every_stop_string(stop):
    assert truncate_at_stop(f" 18{stop} and then noise", HARNESS.stops) == " 18"


def test_truncate_cuts_at_the_earliest_stop_not_the_first_listed():
    # `\nAnswer:` appears before `\nQuestion:`; scanning in listed order and
    # taking the first hit would keep the drifted continuation.
    text = " 18\nAnswer: 20\nQuestion: next"
    assert truncate_at_stop(text, HARNESS.stops) == " 18"


def test_truncate_is_identity_without_a_stop():
    assert truncate_at_stop(" 18", HARNESS.stops) == " 18"


def _train_frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": [f"q{i}" for i in range(rows)],
            "answer": [f"work <<{i}+1={i + 1}>>{i + 1}\n#### {i + 1}" for i in range(rows)],
        }
    )


def test_exemplar_draws_are_nested_across_shot_counts():
    train = _train_frame()
    one = select_exemplars(train, 1, 5, seed=0)
    five = select_exemplars(train, 5, 5, seed=0)
    assert five[:1] == one


def test_exemplar_draws_differ_across_seeds():
    train = _train_frame()
    draws = {tuple(select_exemplars(train, 5, 5, seed=s)) for s in range(3)}
    assert len(draws) == 3


def test_prompt_strips_calculator_spans_and_ends_open():
    train = _train_frame()
    prompt = build_prompt([train.iloc[3]], "how many?", HARNESS, strip_calculator=True)
    assert "<<" not in prompt
    assert prompt.endswith("Question: how many?\nAnswer:")
    assert prompt.count("Question:") == 2


def test_prompt_keeps_calculator_spans_when_asked():
    train = _train_frame()
    prompt = build_prompt([train.iloc[3]], "how many?", HARNESS, strip_calculator=False)
    assert "<<3+1=4>>" in prompt


def test_exemplar_solutions_survive_gold_extraction():
    train = _train_frame()
    prompt = build_prompt([train.iloc[3]], "how many?", HARNESS, strip_calculator=True)
    # The exemplar must still demonstrate the answer format it is teaching.
    assert extract_prediction(prompt, HARNESS) == "4"


def test_bare_format_matches_the_openmath_rendering():
    train = _train_frame()
    prompt = build_prompt([train.iloc[3]], "how many?", BARE, strip_calculator=True)
    # `{problem}\n{solution}\nAnswer: {answer}`, then the bare query.
    assert prompt.startswith("q3\nwork 4\nAnswer: 4")
    assert prompt.endswith("how many?\n")
    assert "####" not in prompt
    assert "Question:" not in prompt


def test_dapo_format_wraps_every_block():
    train = _train_frame()
    prompt = build_prompt([train.iloc[3]], "how many?", DAPO, strip_calculator=True)
    assert prompt.count(DAPO_PREAMBLE) == 2
    assert prompt.endswith('after "Answer:".\n')


def test_prediction_delimiter_follows_the_format():
    text = " some work\nAnswer: 42"
    assert extract_prediction(text, BARE) == "42"
    # The harness format looks for `####`, which is absent here.
    assert extract_prediction(text, HARNESS) is None


def test_gold_extraction_is_format_independent():
    # Gold always comes from GSM8K's own `#### N`, whatever the prompt format.
    assert extract_gold("reasoning\n#### 72") == "72"


class _FakeCachedNano:
    def __init__(self, continuations):
        self.continuations = continuations
        self.step_index = 0
        self.prefill_ids = None
        self.prefill_valid = None

    def make_generation_cache(self, batch, width, device, dtype=None):
        del device, dtype
        return [(batch, width)]

    def _logits(self):
        logits = torch.zeros((len(self.continuations), 8))
        for row, continuation in enumerate(self.continuations):
            logits[row, continuation[self.step_index]] = 10
            logits[row, 7] = 20
        return logits

    def prefill(self, ids, caches, key_valid):
        del caches
        self.prefill_ids = ids.clone()
        self.prefill_valid = key_valid.clone()
        return SimpleNamespace(logits=self._logits())

    def token_step(self, chosen, caches, position):
        del chosen, caches, position
        self.step_index += 1
        return SimpleNamespace(logits=self._logits())


def test_nanogpt_greedy_uses_one_prefill_then_cached_steps():
    model = _FakeCachedNano([[2, 0], [3, 4]])
    outputs, work = greedy_generate(
        model,
        [[5], [5, 6]],
        max_new_tokens=2,
        eot_id=0,
        blocked_ids=torch.tensor([7]),
        device=torch.device("cpu"),
        stop_check_every=2,
        stops=("never",),
        decode=lambda ids: "".join(map(str, ids)),
    )

    assert outputs == [[2], [3, 4]]
    assert model.prefill_ids.tolist() == [[0, 5], [5, 6]]
    assert model.prefill_valid.tolist() == [[False, True], [True, True]]
    assert work == {
        "prefill_forwards": 1,
        "decode_forwards": 1,
        "model_forwards": 2,
        "termination_reasons": ("eot", "token_safety_cap"),
        "realized_bytes": (1, 2),
    }


def test_nanogpt_greedy_pads_every_chunk_to_one_static_prompt_width():
    model = _FakeCachedNano([[2]])
    outputs, _ = greedy_generate(
        model,
        [[5, 6]],
        max_new_tokens=1,
        eot_id=0,
        blocked_ids=torch.tensor([7]),
        device=torch.device("cpu"),
        stop_check_every=1,
        stops=("never",),
        decode=lambda ids: "".join(map(str, ids)),
        padded_prompt_width=5,
    )

    assert outputs == [[2]]
    assert model.prefill_ids.tolist() == [[0, 0, 0, 5, 6]]
    assert model.prefill_valid.tolist() == [[False, False, False, True, True]]


def test_nanogpt_greedy_rejects_short_static_prompt_width():
    model = _FakeCachedNano([[2]])
    with pytest.raises(ValueError, match="shorter"):
        greedy_generate(
            model,
            [[5, 6]],
            max_new_tokens=1,
            eot_id=0,
            blocked_ids=torch.tensor([7]),
            device=torch.device("cpu"),
            stop_check_every=1,
            stops=("never",),
            decode=lambda ids: "".join(map(str, ids)),
            padded_prompt_width=1,
        )


def test_nanogpt_greedy_never_overshoots_semantic_byte_cap():
    model = _FakeCachedNano([[1, 2]])
    outputs, work = greedy_generate(
        model,
        [[5]],
        max_new_tokens=2,
        eot_id=0,
        blocked_ids=torch.tensor([7]),
        device=torch.device("cpu"),
        stop_check_every=1,
        stops=("never",),
        decode=lambda ids: "".join({1: "abc", 2: "def"}[token] for token in ids),
        max_new_bytes=4,
    )

    assert outputs == [[1]]
    assert work["termination_reasons"] == ("byte_cap",)
    assert work["realized_bytes"] == (3,)


class _FakeByteModel:
    def __init__(self, prompt_lengths, continuations):
        self.config = SimpleNamespace(
            vocab=SimpleNamespace(pad_id=262, eot_id=256, output_size=261),
            patch_stride=4,
        )
        self.prompt_lengths = prompt_lengths
        self.continuations = continuations
        self.first_ids = None
        self.first_valid = None
        self.first_kwargs = None

    def forward_ar_varlen(self, ids, valid, **kwargs):
        if self.first_ids is None:
            self.first_ids = ids.clone()
            self.first_valid = valid.clone()
            self.first_kwargs = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in kwargs.items()
            }
        lengths = valid.sum(1)
        logits = torch.zeros((int(lengths.sum()), 261))
        offsets = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
        for row, length in enumerate(lengths.tolist()):
            generated = length - self.prompt_lengths[row]
            token = self.continuations[row][generated]
            logits[offsets[row + 1] - 1, token] = 10
            # An untrained control id must never win even with a larger logit.
            logits[offsets[row + 1] - 1, 257] = 20
        return SimpleNamespace(logits=logits)


def test_byte_greedy_uses_virtual_bos_blocks_controls_and_stops_on_eot():
    prompts = [b"A", b"BC"]
    model = _FakeByteModel([1, 2], [[ord("2"), 256], [256]])
    results = greedy_generate_bytes(
        model,
        prompts,
        max_new_bytes=4,
        max_native_actions=4,
        context_bytes=16,
        stops=("\n\n",),
        device=torch.device("cpu"),
    )

    assert model.first_ids[0, 0].item() == ord("A")
    assert model.first_ids[1, :2].tolist() == [ord("B"), ord("C")]
    assert model.first_ids.shape == (2, 4)
    assert model.first_valid.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
    ]
    assert model.first_kwargs["byte_cu_seqlens"].tolist() == [0, 1, 3]
    assert model.first_kwargs["patch_cu_seqlens"].tolist() == [0, 2, 4]
    assert model.first_kwargs["global_patch_sources"].tolist() == [-1, 0, -1, 1]
    assert model.first_kwargs["condition_patch_indices"].tolist() == [0, 2, 2]
    assert [result.raw for result in results] == [b"2", b""]
    assert [result.termination for result in results] == ["eot", "eot"]
    assert [result.native_actions for result in results] == [2, 1]


def test_byte_greedy_marks_invalid_utf8_wrong_instead_of_replacement_decoding():
    model = _FakeByteModel([1], [[0xFF, 256]])
    result = greedy_generate_bytes(
        model,
        [b"A"],
        max_new_bytes=4,
        max_native_actions=4,
        context_bytes=16,
        stops=("\n\n",),
        device=torch.device("cpu"),
    )[0]

    assert result.raw == b"\xff"
    assert result.text is None
    assert result.invalid_utf8

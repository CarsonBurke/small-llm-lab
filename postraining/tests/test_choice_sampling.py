from copy import deepcopy
from itertools import permutations
from pathlib import Path

import pytest

from postraining.choice_prompt import split_options
from postraining.core import GPT2BPETokenizer, encode_prompt
from postraining.vapo.choice_sampling import permute_choice_row, randomize_choice_row
from postraining.vapo.config import build_arg_parser, validate_args
from postraining.vapo.mixture import MixtureSource, MixedPromptSampler, mixture_identity


def choice_row():
    return {
        "prompt": [{"role": "user", "content": "Which substance freezes?\n\nA. steam\nB. water\nC. sand\nD. air"}],
        "reward_model": {"style": "rule", "ground_truth": "B"},
        "extra_info": {"index": "science1", "option_order": [2, 0, 3, 1], "source_position": 0},
        "_qualified_identity": "science:science1",
    }


def test_all_permutations_preserve_answer_content_source_mapping_and_token_budget():
    row = choice_row()
    original = deepcopy(row)
    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    tokens = len(encode_prompt(tokenizer, row["prompt"][0]["content"]))
    for order in permutations(range(4)):
        changed = permute_choice_row(row, order)
        parsed = split_options(changed["prompt"][0]["content"])
        answer = parsed.labels.index(changed["reward_model"]["ground_truth"])
        assert parsed.options[answer] == "water"
        assert changed["extra_info"]["option_order"][answer] == 0
        assert len(encode_prompt(tokenizer, changed["prompt"][0]["content"])) == tokens
    assert row == original


def test_permutation_composition_maps_back_to_original_source():
    row = choice_row()
    transformed = permute_choice_row(permute_choice_row(row, [3, 2, 1, 0]), [3, 2, 1, 0])
    assert transformed["prompt"] == row["prompt"]
    assert transformed["reward_model"] == row["reward_model"]
    assert transformed["extra_info"]["option_order"] == row["extra_info"]["option_order"]


def test_mc_sampler_is_immutable_resumable_and_independent_of_batching():
    row = choice_row()
    original = deepcopy(row)
    sources = [MixtureSource("science", Path("unused"), "math", (row,))]
    full = MixedPromptSampler(sources, 17, "identity").next_rows(40)
    resumed = MixedPromptSampler(sources, 17, "identity", cursor=13)
    assert resumed.next_rows(7) + resumed.next_rows(20) == full[13:]
    assert len({entry["prompt"][0]["content"] for entry in full}) > 8
    assert row == original
    fixed = MixedPromptSampler(sources, 17, "identity", randomize_choice_options=False)
    assert fixed.next_rows(40) == [row] * 40


def test_shuffle_does_not_depend_on_gold():
    row = choice_row()
    other = deepcopy(row)
    other["reward_model"]["ground_truth"] = "C"
    other["extra_info"]["source_position"] = 3
    assert randomize_choice_row(row, seed=4, cursor=10)["prompt"] == randomize_choice_row(other, seed=4, cursor=10)["prompt"]


def test_numeric_rows_are_not_inferred_to_be_mc():
    row = choice_row()
    row["extra_info"]["option_order"] = None
    assert randomize_choice_row(row, seed=1, cursor=2) is row
    with pytest.raises(ValueError, match="screened"):
        permute_choice_row(row, [0, 1, 2, 3])


@pytest.mark.parametrize("damage", ["target", "order", "source", "prompt"])
def test_inconsistent_choice_metadata_fails_closed(damage):
    row = choice_row()
    if damage == "target":
        row["reward_model"]["ground_truth"] = "E"
    elif damage == "order":
        row["extra_info"]["option_order"] = [0, 0, 1, 2]
    elif damage == "source":
        row["extra_info"]["source_position"] = 2
    else:
        row["prompt"][0]["content"] = "This is not a choice question."
    with pytest.raises(ValueError):
        randomize_choice_row(row, seed=1, cursor=2)


def test_sampling_policy_changes_resume_identity(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    manifest = {"sources": []}
    assert mixture_identity(path, manifest) != mixture_identity(path, manifest, randomize_choice_options=True)


@pytest.mark.parametrize("flags,expected", [
    ([], True), (["--rollout-only"], False),
    (["--rollout-only", "--randomize-choice-options"], True),
    (["--no-randomize-choice-options"], False),
])
def test_training_and_frozen_gate_shuffle_defaults(flags, expected):
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "unused", "--output", "postraining/runs/test", *flags])
    validate_args(parser, args)
    assert args.randomize_choice_options is expected

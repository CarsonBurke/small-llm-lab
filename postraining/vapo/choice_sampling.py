"""Immutable option permutations for screened single-choice RL rows."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

from postraining.choice_prompt import RENDERED_LABELS, render_choice_problem, split_options

CHOICE_PRESENTATION_SCHEMA = "cursor_seeded_choice_permutation/v1"


def permute_choice_row(row: dict, order: Sequence[int]) -> dict:
    """Reorder current options and remap gold/source metadata together.

    ``order[new_position]`` is the old rendered position. Only pool-builder
    screened rows carrying ``option_order`` can be permuted; arbitrary MC
    text may contain positional references and is not safe to shuffle.
    """
    info = row.get("extra_info") or {}
    original_order = info.get("option_order")
    if original_order is None:
        raise ValueError("choice permutation requires screened option_order metadata")
    prompt = row["prompt"]
    if len(prompt) != 1 or prompt[0]["role"] != "user":
        raise ValueError("choice permutation requires a single user prompt")
    parsed = split_options(prompt[0]["content"])
    count = len(original_order)
    order = list(order)
    if (
        parsed is None or len(parsed.options) != count
        or parsed.labels != RENDERED_LABELS[:count]
        or sorted(order) != list(range(count))
        or sorted(original_order) != list(range(count))
    ):
        raise ValueError("invalid choice options or permutation metadata")
    reward = row["reward_model"]
    truth = reward["ground_truth"]
    if reward["style"] != "rule" or truth not in parsed.labels:
        raise ValueError("choice permutation requires an exact single-letter target")
    target = parsed.labels.index(truth)
    if original_order[target] != info.get("source_position"):
        raise ValueError("choice target disagrees with source-position metadata")
    rendered = render_choice_problem(parsed.question, tuple(parsed.options[i] for i in order))
    return {
        **row,
        "prompt": [{**prompt[0], "content": rendered}],
        "reward_model": {**reward, "ground_truth": RENDERED_LABELS[order.index(target)]},
        "extra_info": {
            **info,
            "option_order": [original_order[i] for i in order],
            "choice_presentation_schema": CHOICE_PRESENTATION_SCHEMA,
            "presentation_option_order": order,
        },
    }


def randomize_choice_row(row: dict, *, seed: int, cursor: int) -> dict:
    """A fresh uniform permutation per presentation, reproducible on resume.

    All trajectories from one presentation share its permutation. The seed
    depends only on the prompt cursor, not batching or mutable RNG state.
    Numeric Knowledge, math and code rows are returned unchanged.
    """
    info = row.get("extra_info") or {}
    original_order = info.get("option_order")
    if original_order is None:
        return row
    identity = row.get("_qualified_identity", info.get("index", row["prompt"][0]["content"]))
    material = f"{CHOICE_PRESENTATION_SCHEMA}\0{seed}\0{cursor}\0{identity}"
    rng = random.Random(int.from_bytes(hashlib.sha256(material.encode()).digest(), "big"))
    order = list(range(len(original_order)))
    rng.shuffle(order)
    return permute_choice_row(row, order)

"""Canonical prompt framing for post-training mathematics episodes.

The source parquets inherited several copies of DAPO's plain-text
``Answer:`` contract.  Answer-fenced SFT, RL, and evaluation must not preserve
that source-specific framing: every problem gets one identical instruction
using the registered think and answer tokens.
"""

from __future__ import annotations

import re

from postraining.core import ANSWER_CLOSE, ANSWER_OPEN, THINK_CLOSE, THINK_OPEN

DAPO_HEADER = "Solve the following math problem step by step."

ANSWER_FIELD_INSTRUCTIONS = (
    "The last line of your response should be of the form Answer: "
    "$Answer (without quotes) where $Answer is the answer to the problem.",
    'Remember to put your answer on its own line after "Answer:".',
)

CHINESE_ANSWER_FIELD_INSTRUCTION = (
    "请以“Answer: \\boxed{<final_answer>}”的格式输出最终答案。"
)
CHINESE_REASONING_INSTRUCTION = "让我们一步一步地思考。"

ANSWER_FENCE_INSTRUCTION = (
    f"Start your response with {THINK_OPEN} and reason until {THINK_CLOSE}, "
    f"then end it with only the final answer inside "
    f"{ANSWER_OPEN}{ANSWER_CLOSE}."
)
ANSWER_FENCE_SUFFIX = f"\n\n{ANSWER_FENCE_INSTRUCTION}"
ANSWER_FENCE_PROMPT_SCHEMA = "bare_problem_single_think_answer_contract/v1"

# Older code replaced every legacy sentence independently, so already-built
# prompts can contain either or all of these redundant fence instructions.
# Treat them as source framing and canonicalize them just like the originals.
LEGACY_FENCE_INSTRUCTIONS = (
    ANSWER_FENCE_INSTRUCTION,
    f"Remember to start with {THINK_OPEN} and put only the final answer "
    f"inside {ANSWER_OPEN}{ANSWER_CLOSE}.",
    f"请以 {THINK_OPEN} 开始思考，并仅将最终答案放在 "
    f"{ANSWER_OPEN}{ANSWER_CLOSE} 中。",
)

SOURCE_INSTRUCTIONS = (
    DAPO_HEADER,
    *ANSWER_FIELD_INSTRUCTIONS,
    CHINESE_ANSWER_FIELD_INSTRUCTION,
    CHINESE_REASONING_INSTRUCTION,
    *LEGACY_FENCE_INSTRUCTIONS,
)

# The verifier's Answer: field is case-insensitive.  Canonicalization must use
# the same rule or an unlisted source template could survive silently.
ANSWER_FIELD_DEMAND = re.compile(r"(?i)answer\s*:")


def require_answer_fence_prompt_schema(
    metadata: dict,
    *,
    answer_fence: bool,
    source: str,
) -> None:
    """Reject offline evaluation under prompt wording unlike training."""

    if not answer_fence:
        return
    schema = metadata.get("answer_fence_prompt_schema")
    if schema != ANSWER_FENCE_PROMPT_SCHEMA:
        raise ValueError(
            f"{source} uses answer-fence prompt schema {schema!r}, not "
            f"{ANSWER_FENCE_PROMPT_SCHEMA!r}; evaluating it with the current "
            "canonicalizer would silently change the policy environment"
        )


def strip_math_prompt_framing(content: str) -> tuple[str, int]:
    """Return the bare problem and number of recognized clauses removed."""

    removed = 0
    for instruction in SOURCE_INSTRUCTIONS:
        occurrences = content.count(instruction)
        if occurrences:
            content = content.replace(instruction, "")
            removed += occurrences
    # Removing a sentence between a prefix and a blank line can leave spaces
    # on otherwise-empty lines.  Normalize only that structural residue; do
    # not rewrite whitespace inside the actual problem.
    content = re.sub(r"[ \t]+\n", "\n", content).strip()
    if ANSWER_FIELD_DEMAND.search(content):
        raise ValueError(
            "answer-fence prompt canonicalization left an Answer: demand "
            f"in place (unlisted instruction template?): {content[:200]!r}"
        )
    return content, removed


def answer_fence_prompt(problem_or_prompt: str) -> str:
    """Frame one bare or source-wrapped problem with one fence contract."""

    problem, _ = strip_math_prompt_framing(problem_or_prompt)
    if not problem:
        raise ValueError("math prompt reduced to an empty problem")
    return problem + ANSWER_FENCE_SUFFIX


def canonicalize_answer_fence_rows(rows: list[dict]) -> list[dict]:
    """Copy rows and give each prompt exactly one canonical fence contract.

    Dataset rows must carry a recognized source contract.  Failing closed
    prevents a new, contradictory ``Answer:`` template from reaching reward
    training unnoticed.  Multiple-message prompts are preserved; the single
    canonical instruction is appended to their final non-empty message.
    """

    canonical_rows = []
    for source_row in rows:
        row = dict(source_row)
        prompt = [dict(message) for message in row["prompt"]]
        removed = 0
        last_nonempty = None
        for index, message in enumerate(prompt):
            content, count = strip_math_prompt_framing(message["content"])
            message["content"] = content
            removed += count
            if content:
                last_nonempty = index
        if not removed:
            preview = prompt[0]["content"][:120] if prompt else ""
            raise ValueError(
                "answer-fence prompt canonicalization found no recognized "
                f"instruction in prompt {preview!r}"
            )
        if last_nonempty is None:
            raise ValueError("math prompt reduced to an empty problem")
        prompt[last_nonempty]["content"] += ANSWER_FENCE_SUFFIX
        row["prompt"] = prompt
        canonical_rows.append(row)
    return canonical_rows

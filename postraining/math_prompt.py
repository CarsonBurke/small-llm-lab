"""Canonical user prompts for post-training mathematics episodes.

Source parquets carry several variants of DAPO's plain-text ``Answer:``
contract.  Answer-fenced SFT, RL, and evaluation remove all such framing: the
user prompt is only the problem, while registered think and answer tokens
define the completion structure learned by the policy.
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

# openbmb/UltraData-RL-2609 prompts are rendered with the MiniCPM5 chat
# template, which appends a boxed-answer contract and a literal context
# budget.  The budget sentence is not merely redundant here: it states 9k
# tokens, which is false for every model in this repository, so leaving it in
# a prompt would train the policy against a context it does not have.
ULTRADATA_REASONING_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
ULTRADATA_FINAL_ANSWER_INSTRUCTION = (
    "After any reasoning, give the final answer as Answer: \\boxed{...}, "
    "replacing ... with only the answer."
)
ULTRADATA_BUDGET_SUFFIX = "You have a budget of 9k tokens"
# "The input will be given via stdin and the output should be printed to
# stdout by your code." is deliberately NOT registered: that sentence occurs
# inside competitive-programming problem statements as well as in the
# template, and unconditional removal would mutilate the problem body.
ULTRADATA_CODE_INSTRUCTIONS = (
    "End with the complete executable Python program in one ```python fenced "
    "code block. Read standard input and write standard output.",
    "Now solve the problem by providing the code.",
)

# Registered because it survived canonicalization: the Chinese answer-field
# clause removes the "Answer:" half of these prompts, after which neither the
# Answer: demand nor the context-budget check can see the remaining
# output-format sentence, and it was reaching "bare" problems as their tail.
ULTRADATA_CHINESE_OUTPUT_FORMAT = (
    "如果是选择题，请按顺序输出正确的选项，不带任何标点或空格。"
    "对于其他类型的问题，请只输出最终答案的数值。"
)
ULTRADATA_SOURCE_INSTRUCTIONS = (
    ULTRADATA_REASONING_INSTRUCTION,
    ULTRADATA_FINAL_ANSWER_INSTRUCTION,
    ULTRADATA_BUDGET_SUFFIX,
    ULTRADATA_CHINESE_OUTPUT_FORMAT,
    *ULTRADATA_CODE_INSTRUCTIONS,
)

CHINESE_ANSWER_FIELD_INSTRUCTION = (
    "请以“Answer: \\boxed{<final_answer>}”的格式输出最终答案。"
)
CHINESE_REASONING_INSTRUCTION = "让我们一步一步地思考。"

LEGACY_ANSWER_FENCE_INSTRUCTION = (
    f"Start your response with {THINK_OPEN} and reason until {THINK_CLOSE}, "
    f"then end it with only the final answer inside "
    f"{ANSWER_OPEN}{ANSWER_CLOSE}."
)
LEGACY_ANSWER_FENCE_SUFFIX = f"\n\n{LEGACY_ANSWER_FENCE_INSTRUCTION}"
LEGACY_ANSWER_FENCE_PROMPT_SCHEMA = (
    "bare_problem_single_think_answer_contract/v1"
)

# The answer-fenced policy's response contract lives entirely in completion
# token structure.  Keep the suffix constant for callers that share prompt
# boundary code with the legacy/plain modes, but make its emptiness explicit.
ANSWER_FENCE_SUFFIX = ""
ANSWER_FENCE_PROMPT_SCHEMA = "bare_problem_think_answer_tokens/v2"

# Older code replaced every legacy sentence independently, so already-built
# prompts can contain either or all of these redundant fence instructions.
# Treat them as source framing and canonicalize them just like the originals.
LEGACY_FENCE_INSTRUCTIONS = (
    LEGACY_ANSWER_FENCE_INSTRUCTION,
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
    *ULTRADATA_SOURCE_INSTRUCTIONS,
)

# The verifier's Answer: field is case-insensitive.  Canonicalization must use
# the same rule or an unlisted source template could survive silently.
ANSWER_FIELD_DEMAND = re.compile(r"(?i)answer\s*:")

# A token-budget clause carries no ``Answer:``, so the demand check above
# cannot see it.  Match the shape rather than the one literal we register: a
# corpus rebuilt at a different budget must fail closed instead of teaching
# the policy a context size it does not have.
CONTEXT_BUDGET_DEMAND = re.compile(
    r"(?i)you have a budget of\s*\S+\s*tokens"
)

# Chinese output-format instructions carry no ``Answer:`` once the registered
# clause is removed, so match their shape rather than the one literal we
# register: "请...输出...答案" / "请...输出...选项".
CHINESE_OUTPUT_FORMAT_DEMAND = re.compile(r"请[^。]{0,40}输出[^。]{0,40}(答案|选项)")

# UltraData's single-choice final-line templates ("The answer is $LETTER")
# carry no ``Answer:`` colon in most variants, so neither check above sees
# them. ``choice_prompt.strip_choice_framing`` removes the header that holds
# them; one surviving anywhere else must fail closed.
CHOICE_TEMPLATE_DEMAND = re.compile(r"\$LETTER\b")


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
    if CONTEXT_BUDGET_DEMAND.search(content):
        raise ValueError(
            "answer-fence prompt canonicalization left a context-budget "
            "clause in place; register the exact wording rather than "
            f"training against a false context: {content[:200]!r}"
        )
    if CHINESE_OUTPUT_FORMAT_DEMAND.search(content):
        raise ValueError(
            "answer-fence prompt canonicalization left a Chinese "
            "output-format instruction in place (unlisted template?): "
            f"{content[:200]!r}"
        )
    if CHOICE_TEMPLATE_DEMAND.search(content):
        raise ValueError(
            "answer-fence prompt canonicalization left a $LETTER "
            f"answer-format template in place: {content[:200]!r}"
        )
    return content, removed


def answer_fence_prompt(problem_or_prompt: str) -> str:
    """Return only the problem, stripped of every recognized source wrapper."""

    problem, _ = strip_math_prompt_framing(problem_or_prompt)
    if not problem:
        raise ValueError("math prompt reduced to an empty problem")
    return problem


def canonicalize_answer_fence_rows(rows: list[dict]) -> list[dict]:
    """Copy rows and reduce each prompt to only its canonical problem text.

    Dataset rows must carry a recognized source contract.  Failing closed
    prevents a new, contradictory ``Answer:`` template from reaching reward
    training unnoticed.  Rows emitted here are marked with the bare contract
    so canonicalization remains idempotent without accepting arbitrary
    unmarked free-form prompts.
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
        bare_contract = (row.get("extra_info") or {}).get(
            "prompt_contract"
        ) == "bare"
        if not removed and not bare_contract:
            preview = prompt[0]["content"][:120] if prompt else ""
            raise ValueError(
                "answer-fence prompt canonicalization found no recognized "
                f"instruction in prompt {preview!r}"
            )
        if last_nonempty is None:
            raise ValueError("math prompt reduced to an empty problem")
        extra_info = dict(row.get("extra_info") or {})
        extra_info["prompt_contract"] = "bare"
        row["prompt"] = prompt
        row["extra_info"] = extra_info
        canonical_rows.append(row)
    return canonical_rows

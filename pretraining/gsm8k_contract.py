"""Model-family-neutral GSM8K prompting and exact-answer parsing contract."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


GSM8K_ROOT = Path(
    "~/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/"
    "740312add88f781978c0658806c59bc2815b9866/main"
).expanduser()
TRAIN_PARQUET = GSM8K_ROOT / "train-00000-of-00001.parquet"
TEST_PARQUET = GSM8K_ROOT / "test-00000-of-00001.parquet"
GOLD_DELIMITER = "####"
CALCULATOR_SPAN = re.compile(r"<<[^>]*>>")
DAPO_PREAMBLE = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem."
)
DAPO_REMINDER = 'Remember to put your answer on its own line after "Answer:".'


def answer_pattern(delimiter: str) -> re.Pattern[str]:
    """Match a well-formed integer or decimal after an answer delimiter."""

    return re.compile(
        rf"{re.escape(delimiter)}\s*(-?(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)"
    )


GOLD_PATTERN = answer_pattern(GOLD_DELIMITER)


@dataclass(frozen=True)
class PromptFormat:
    name: str
    delimiter: str
    stops: tuple[str, ...]

    def solution(self, answer: str, *, strip_calculator: bool) -> str:
        body = CALCULATOR_SPAN.sub("", answer) if strip_calculator else answer
        reasoning, _, final = body.partition(GOLD_DELIMITER)
        if self.delimiter == GOLD_DELIMITER:
            return body
        return f"{reasoning.strip()}\n{self.delimiter} {final.strip()}"

    def exemplar(self, question: str, solution: str) -> str:
        if self.name == "harness":
            return f"Question: {question}\nAnswer: {solution}"
        if self.name == "bare":
            return f"{question}\n{solution}"
        return f"{DAPO_PREAMBLE}\n\n{question}\n\n{DAPO_REMINDER}\n{solution}"

    def query(self, question: str) -> str:
        if self.name == "harness":
            return f"Question: {question}\nAnswer:"
        if self.name == "bare":
            return f"{question}\n"
        return f"{DAPO_PREAMBLE}\n\n{question}\n\n{DAPO_REMINDER}\n"


PROMPT_FORMATS = {
    "harness": PromptFormat(
        "harness", GOLD_DELIMITER, ("\nQuestion:", "\nAnswer:", "\n\n")
    ),
    "bare": PromptFormat("bare", "Answer:", ("\n\n",)),
    "dapo": PromptFormat("dapo", "Answer:", ("\n\n",)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_number(text: str) -> str | None:
    stripped = text.replace(",", "").strip()
    try:
        return str(int(stripped))
    except ValueError:
        pass
    try:
        value = float(stripped)
    except (ValueError, OverflowError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return str(int(value)) if value == int(value) else repr(value)


def extract_gold(text: str) -> str | None:
    matches = GOLD_PATTERN.findall(text)
    return normalize_number(matches[-1]) if matches else None


def extract_prediction(text: str, fmt: PromptFormat) -> str | None:
    match = answer_pattern(fmt.delimiter).search(text)
    return normalize_number(match.group(1)) if match else None


def build_prompt(
    exemplars: Sequence[object],
    question: str,
    fmt: PromptFormat,
    *,
    strip_calculator: bool,
) -> str:
    blocks = [
        fmt.exemplar(
            row["question"],  # type: ignore[index]
            fmt.solution(row["answer"], strip_calculator=strip_calculator),  # type: ignore[index]
        )
        for row in exemplars
    ]
    blocks.append(fmt.query(question))
    return "\n\n".join(blocks)


def select_exemplars(train: object, shots: int, max_shots: int, seed: int) -> list[int]:
    if shots < 0 or max_shots < shots:
        raise ValueError("shot counts must satisfy 0 <= shots <= max_shots")
    return random.Random(seed).sample(range(len(train)), max_shots)[:shots]  # type: ignore[arg-type]

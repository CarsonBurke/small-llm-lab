"""Verified reasoning data and privileged-context construction for OPSD."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from postraining.core import encode_prompt
from postraining.math_prompt import strip_math_prompt_framing
from postraining.opsd.schemas import OPSD_PROMPT_SCHEMA
from postraining.prepare_sft_traces import (
    INSTRUCTION_SUFFIX,
    INSTRUCTION_SUFFIX_ANSWER,
)


TEACHER_TRANSITION = (
    "After reading the reference solution above, make sure you understand "
    "the reasoning behind every step. Do not copy or paraphrase it. Now use "
    "your own reasoning to derive the same final answer to the original "
    "problem. Work step by step, explore alternatives when useful, and "
    "backtrack if an approach fails."
)
TEACHER_PROMPT_SCHEMA = OPSD_PROMPT_SCHEMA


@dataclass(frozen=True)
class OPSDExample:
    problem: str
    student_prompt: str
    instruction_suffix: str
    reference_solution: str
    reference_kind: str
    source: str


@dataclass(frozen=True)
class TokenizedOPSDExample:
    example: OPSDExample
    student_prompt_ids: tuple[int, ...]
    teacher_prompt_ids: tuple[int, ...]


def prompt_suffix(answer_fence: bool) -> str:
    return INSTRUCTION_SUFFIX_ANSWER if answer_fence else INSTRUCTION_SUFFIX


def build_teacher_prompt(
    problem: str,
    reference_solution: str,
    reference_kind: str = "solution",
    instruction_suffix: str = INSTRUCTION_SUFFIX_ANSWER,
) -> str:
    """Put privilege first and preserve the exact student response contract."""
    if not problem or not instruction_suffix:
        raise ValueError("teacher prompt requires a problem and response contract")
    try:
        bare_problem, removed = strip_math_prompt_framing(problem)
    except ValueError as error:
        raise ValueError("teacher prompt requires a bare problem") from error
    if removed or bare_problem != problem.strip():
        raise ValueError("teacher prompt requires a bare problem, not a student prompt")
    if reference_kind == "solution":
        privilege = (
            "Here is a verified reference solution to the problem:\n"
            "=== Reference Solution Begin ===\n"
            f"{reference_solution}\n"
            "=== Reference Solution End ===\n\n"
            f"{TEACHER_TRANSITION}"
        )
    elif reference_kind == "final_answer":
        privilege = (
            "Here is the verified final answer to the problem:\n"
            "=== Verified Final Answer Begin ===\n"
            f"{reference_solution}\n"
            "=== Verified Final Answer End ===\n\n"
            "Use this answer only as privileged target and checking "
            "information. Independently derive it from the original problem; "
            "do not merely state or copy it. Work step by step, explore "
            "alternatives when useful, and backtrack if an approach fails."
        )
    else:
        raise ValueError(f"unknown OPSD reference kind {reference_kind!r}")
    return f"{problem}\n\n{privilege}\n\nBegin the new solution now:{instruction_suffix}"


def load_examples(
    path: str | Path,
    *,
    answer_fence: bool,
    reference_column: str = "auto",
) -> list[OPSDExample]:
    """Load only independently verified rows with a usable solution trace."""
    path = Path(path)
    table = pq.read_table(path)
    columns = set(table.schema.names)
    if "problem" not in columns:
        raise ValueError(f"{path} lacks a problem column")
    reference_column = resolve_reference_column(path, reference_column)
    if "verified" not in columns:
        raise ValueError(
            f"{path} lacks the required boolean verified column; OPSD "
            "must not treat unverified reference text as privileged truth"
        )
    suffix = prompt_suffix(answer_fence)
    examples: list[OPSDExample] = []
    for row in table.to_pylist():
        # OPSD's privileged information must be ground-truth quality. Only a
        # literal boolean true is accepted; missing/null/non-boolean values
        # fail closed.
        if row.get("verified") is not True:
            continue
        if row.get("problem") is None:
            continue
        problem = str(row["problem"])
        if not problem.strip():
            continue
        student_prompt = problem + suffix
        if reference_column == "document":
            if row.get(reference_column) is None:
                continue
            document = str(row[reference_column])
            if not document.startswith(student_prompt):
                continue
            reference = document[len(student_prompt):].strip()
            reference_kind = "solution"
        else:
            if row.get(reference_column) is None:
                continue
            reference = str(row[reference_column]).strip()
            reference_kind = str(row.get("reference_kind") or "solution")
        if reference_kind not in {"solution", "final_answer"}:
            continue
        if not reference:
            continue
        examples.append(
            OPSDExample(
                problem=problem,
                student_prompt=student_prompt,
                instruction_suffix=suffix,
                reference_solution=reference,
                reference_kind=reference_kind,
                source=str(row.get("source", "unknown")),
            )
        )
    if not examples:
        raise ValueError(f"{path} contains no verified, prompt-compatible rows")
    return examples


def resolve_reference_column(
    path: str | Path, reference_column: str
) -> str:
    """Resolve ``auto`` before applying final-answer data safety gates."""
    columns = set(pq.read_schema(path).names)
    resolved = reference_column
    if resolved == "auto":
        resolved = "document" if "document" in columns else "solution"
    if resolved not in columns:
        raise ValueError(f"{path} lacks reference column {resolved!r}")
    return resolved


def tokenize_example(
    example: OPSDExample,
    tokenizer,
    *,
    max_prompt_length: int,
    context_tokens: int,
    max_completion_length: int,
) -> tuple[TokenizedOPSDExample | None, str | None]:
    """Encode without truncating privileged information; reject overflow."""
    student_ids = tuple(encode_prompt(tokenizer, example.student_prompt))
    if len(student_ids) > max_prompt_length:
        return None, "student_prompt_overflow"
    teacher_ids = tuple(
        encode_prompt(
            tokenizer,
            build_teacher_prompt(
                example.problem,
                example.reference_solution,
                example.reference_kind,
                example.instruction_suffix,
            ),
        )
    )
    if len(teacher_ids) + max_completion_length > context_tokens:
        return None, "teacher_context_overflow"
    return (
        TokenizedOPSDExample(example, student_ids, teacher_ids),
        None,
    )


class ShuffledExampleSampler:
    """Seeded epoch shuffles with a monotonic, exactly resumable cursor."""

    def __init__(self, examples: list[OPSDExample], seed: int, cursor: int = 0):
        if not examples:
            raise ValueError("OPSD sampler requires at least one example")
        if cursor < 0:
            raise ValueError("sampler cursor must be nonnegative")
        self.examples = examples
        self.seed = seed
        self.cursor = cursor
        self._cached_epoch: int | None = None
        self._cached_order: list[int] | None = None

    def _order(self, epoch: int) -> list[int]:
        if epoch != self._cached_epoch:
            order = list(range(len(self.examples)))
            random.Random(self.seed * 1_000_003 + epoch).shuffle(order)
            self._cached_epoch = epoch
            self._cached_order = order
        assert self._cached_order is not None
        return self._cached_order

    def next(self) -> OPSDExample:
        epoch, offset = divmod(self.cursor, len(self.examples))
        example = self.examples[self._order(epoch)[offset]]
        self.cursor += 1
        return example


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

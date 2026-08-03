"""Normalize the relaxed-bar trace sources into verified SFT documents.

Round-5 corpus (NOTES.md, 2026-08-02 relaxed-bar survey): five HF sources
fetched to ``postraining/data/relaxed_bar/`` by fetch_relaxed_bar_sources,
plus the self-generated K3 trace JSONL from generate_k3_traces. Each row
is normalized into the exact episode shape the RL trainer rolls out —
same instruction suffix, same answer format, same GPT-2 tokenizer — so
the SFT stage teaches precisely the distribution VAPO later samples from.

Every kept row's label is independent of the trace that argues for it,
and where an independent gold exists the trace is verified against it
with OUR grader (core.verify_answer). Honest per-source accounting —
"verified" means different things per source and the docstrings say so:

- openmath_gsm8k: RESTRICTED to ``problem_source == "gsm8k"`` (verbatim
  GSM8K train re-answers), verified against OUR local GSM8K gold via
  the same normalized-question join had653 uses. The augmented_gsm8k
  majority is EXCLUDED for the same reason augmented_math never left
  the fetch script: ``expected_answer`` is the teacher's own boxed
  value BY CONSTRUCTION, so no filter can see a bad row — and the
  red-team measured 13.9% decimal finals (vs 0.0% in genuine rows,
  where real GSM8K golds are always integers) plus hand-confirmed
  ill-posed problems carrying plausible integer labels.
- a1_deepmind: question/answer arrive as printed bytes literals
  (``"b'-6\\n'"``) and are unwrapped; the DeepSeek solution's final
  answer (trailing ``**Answer:**`` line or last boxed) must agree with
  the unwrapped canonical truth — a 600-row hand-checked sample put the
  teacher-error rate near HALF the set (the survey's 23.5% was an
  underestimate), all dropped here. The composed final is the CANONICAL
  truth string, matching the exact-style grading these problems get
  in RL.
- had653_gold: only rows whose question joins GSM8K train gold are kept
  (final_answer is teacher-derived — worthless as truth); the cot's
  claimed answer must grade correct against the joined gold.
- sxiong_l13: Level 1-3 only. HONESTY NOTE: the answer column is
  byte-identical to the solution's boxed value in every row, so the
  grading here is an extraction/template-drift canary, NOT independent
  verification — the independent check is upstream (GPT-4o problems
  cross-verified by R1 answer agreement, per the relaxed-bar survey).
  Its finals also carry MATH-register markup (``\\sqrt{29}``,
  ``120^\\circ``) that the numeric RL gate cannot reward — accepted
  deliberately for MATH-style think-span coverage.
- gsm8k_socratic: human-written gold; the derivation must still reach
  the ``####`` answer (calculator ``<<...>>`` annotations stripped).
- k3_traces: records generate_k3_traces marked correct, re-verified
  here with the per-record style. ``--k3-think-channel`` picks whether
  the think span carries the clean visible prose (default — matches
  the register of the rest of the blend) or the native reasoning
  channel (telegraphic, much longer; ablate before switching).

All sources are then decontaminated against GSM8K test, the deepmind
interpolate bench, and the AIME sets: exact normalized-text match (the
deepmind bench problems are short enough to slip under any n-gram) plus
8-gram word overlap. GSM8K TRAIN overlap is expected and fine — several
sources are re-answers of it.

Prompt framing is source-independent and byte-exact to RL/evaluation:
every family is reduced to the bare problem followed by one shared
think/answer-token contract. Composed rows are finally screened for literal
fence strings in any field (a "<think>"
inside teacher prose would tokenize to a REAL special id and break the
single-pair gate invariant) and for empty fields after repair.

Output: ``postraining/data/sft_traces_v4_answer_canonical_hfonly.parquet``
(with --think-tags --answer-tags) plus a sibling build manifest and printed
per-source stage-by-stage drop counts. CPU-only; runs directly, no mlq.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters  # noqa: F401 (env parity)
from postraining.core import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    GPT2BPETokenizer,
    load_unique_math_rows,
    normalize_final_answer,
    parse_numeric_answer,
    verify_answer,
)
from postraining.math_prompt import (
    ANSWER_FENCE_PROMPT_SCHEMA,
    ANSWER_FENCE_SUFFIX,
    answer_fence_prompt,
    strip_math_prompt_framing,
)

RELAXED_BAR = Path("postraining/data/relaxed_bar")
DATA = Path("postraining/data")
OUTPUT = DATA / "sft_traces_v1.parquet"
THINK_OUTPUT = DATA / "sft_traces_v2_think.parquet"
ANSWER_OUTPUT = DATA / "sft_traces_v4_answer_canonical_hfonly.parquet"
INSTRUCTION_SUFFIX = (
    '\n\nRemember to put your answer on its own line after "Answer:".'
)
INSTRUCTION_SUFFIX_ANSWER = ANSWER_FENCE_SUFFIX
# A literal fence string inside teacher text would tokenize to a REAL
# special id and break the anchored gate's single-pair invariant.
FENCE_STRINGS = (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE)
MAX_DOC_TOKENS = 4096
CORPUS_MANIFEST_SCHEMA = "verified_math_sft_corpus/v1"

# Post-verification per-source caps (seeded sample), sized to the blend
# chosen in the relaxed-bar survey: ~28k mixed HF + 7.5k human gold +
# the K3 core. Uncapped sources are already at or under target.
SOURCE_CAPS = {"openmath_gsm8k": 10_000, "a1_deepmind": 6_000}

DECONTAMINATION_TARGETS = (
    RELAXED_BAR / "gsm8k_test_questions.parquet",
    DATA / "deepmind-interpolate-easy.parquet",
    DATA / "aime-2024.parquet",
    DATA / "aime-2026.parquet",
)
NGRAM = 8


def compose_document(
    problem: str,
    reasoning: str,
    final: str,
    think_tags: bool,
    answer_tags: bool = False,
) -> str:
    """One SFT document in the exact RL episode shape.

    ``think_tags=True`` fences the reasoning in dedicated
    ``<think>``/``</think>`` tokens (kept from the teachers rather than
    stripped): the fence gives the model an explicit thinking/output type
    distinction, and the close token becomes a first-class stop-thinking
    action for RL. The ``Answer:`` line stays outside the fence — grading
    starts where thinking ends.

    ``answer_tags=True`` (requires ``think_tags``) matches the anchored
    structural gate exactly: the completion's FIRST token is ``<think>``
    (no leading newline — the gate requires the fence to open the
    completion) and ``</answer>`` is its last, so the packed row's
    separator lands immediately after the close, the shape
    ``structural_format_ok`` anchors on. The fenced value replaces the
    ``Answer:`` line.
    """
    if answer_tags:
        if not think_tags:
            raise ValueError("answer_tags requires think_tags")
        prompt = answer_fence_prompt(problem)
        return (
            f"{prompt}{THINK_OPEN}\n{reasoning}"
            f"\n{THINK_CLOSE}\n{ANSWER_OPEN}{final}{ANSWER_CLOSE}"
        )
    if think_tags:
        return (
            f"{problem}{INSTRUCTION_SUFFIX}\n{THINK_OPEN}\n{reasoning}"
            f"\n{THINK_CLOSE}\nAnswer: {final}"
        )
    return f"{problem}{INSTRUCTION_SUFFIX}\n{reasoning}\nAnswer: {final}"


def canonical_problem(problem: str) -> str:
    """Return the exact bare problem stored beside the composed document."""
    bare, _ = strip_math_prompt_framing(problem)
    if not bare:
        raise ValueError("problem reduced to empty text during canonicalization")
    return bare


NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
CALCULATOR_RE = re.compile(r"<<[^>]*>>")
# Trailing answer-statement markers a teacher appends after the
# derivation ("**Answer:** \\(-6\\)", "**Final Answer:**" with the
# statement on the NEXT line): stripped from think bodies so SFT never
# teaches emitting an Answer: field inside the fence. The colon is
# required — prose lines like "Answer depends on x." are not markers.
ANSWER_LINE_RE = re.compile(
    r"^\s*\*{0,2}(?:final\s+)?answer\*{0,2}\s*:", re.IGNORECASE
)


def normalize_problem(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())[:160]


def numbers_in(text: str) -> set[str]:
    """Canonical numeric strings mentioned in ``text`` ('1,000.0' -> '1000')."""
    values = set()
    for hit in NUMBER_RE.findall(text):
        cleaned = hit.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        values.add(str(int(value)) if value == int(value) else str(value))
    return values


def truth_canonical(truth: str) -> str | None:
    cleaned = truth.replace(",", "").strip()
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return str(int(value)) if value == int(value) else str(value)


def trace_reaches(truth: str, *segments: str) -> bool:
    """Does the derivation actually conclude the canonical answer?

    Checks the closing window of each segment (the conclusion lives at
    the end of a forward derivation, so callers pass pre-sliced windows).
    """
    canonical = truth_canonical(truth)
    if canonical is None:
        return False
    return any(canonical in numbers_in(segment) for segment in segments)


def last_boxed(text: str) -> str | None:
    """Innermost-balanced content of the LAST ``\\boxed{...}`` in ``text``.

    Brace-walked rather than regexed so nested arguments
    (``\\boxed{\\frac{1}{2}}``) extract whole instead of failing.
    """
    marker = text.rfind("\\boxed{")
    if marker < 0:
        return None
    depth = 0
    for index in range(marker + len("\\boxed{") - 1, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[marker + len("\\boxed{"): index].strip()
    return None


def strip_latex_wrappers(value: str) -> str:
    """Peel display wrappers off one final-answer value: ``**\\(-6\\)**``
    -> ``-6``. Loops until stable because teachers nest them."""
    previous = None
    # ``x = <value>`` statements and display fraction variants reduce to
    # the forms the graders parse; inline-math delimiters are pure markup
    # in a final-answer value, so stray unbalanced ones go too.
    value = value.split("=")[-1].strip()
    for delimiter in ("\\(", "\\)", "\\[", "\\]"):
        value = value.replace(delimiter, "")
    value = re.sub(r"\\[dt]frac", r"\\frac", value)
    while value != previous:
        previous = value
        boxed = last_boxed(value)
        if boxed is not None and value.startswith("\\boxed{"):
            value = boxed
        for open_mark, close_mark in (("$", "$"), ("**", "**")):
            if (
                value.startswith(open_mark)
                and value.endswith(close_mark)
                and len(value) > len(open_mark) + len(close_mark)
            ):
                value = value[len(open_mark): -len(close_mark)]
        value = value.strip().rstrip(".").strip()
    return value


# Bound on the trailing-block cascade (red-team verification, V3): the
# stacked-marker loop below is bounded by the data, not by construction
# — a future source with per-part "Answer:" markers spaced within the
# scan window would otherwise be eaten back to its first line silently.
# The largest legitimate observed block is marker + LaTeX display
# wrapper + post-answer commentary, well under this cap.
MAX_ANSWER_BLOCK_LINES = 12


def strip_trailing_answer_lines(reasoning: str) -> str:
    """Drop the trailing answer-statement BLOCK, not just the last line:
    teachers often put a bare ``**Answer:**`` header with the statement
    on the following line, so the marker is scanned for within the last
    few lines and everything from it onward is cut. Repeats until
    stable (stacked ``**Answer:**`` / ``**Final Answer:**`` blocks),
    capped at MAX_ANSWER_BLOCK_LINES total removed."""
    lines = reasoning.rstrip().splitlines()
    removed = 0
    changed = True
    while changed and removed < MAX_ANSWER_BLOCK_LINES:
        changed = False
        while lines and not lines[-1].strip():
            lines.pop()
        for index in range(len(lines) - 1, max(len(lines) - 5, -1), -1):
            if ANSWER_LINE_RE.match(lines[index]):
                removed += len(lines) - index
                del lines[index:]
                changed = True
                break
    return "\n".join(lines).strip()


def unwrap_bytes_literal(text: str) -> str | None:
    """Decode a printed bytes literal (``"b'-6\\n'"``) to its text.

    The a1_math_deepmind columns were serialized through ``str(bytes)``;
    non-literal text passes through untouched. Returns None when the
    literal is malformed (the row is dropped, counted).
    """
    stripped = text.strip()
    if not stripped.startswith(("b'", 'b"')):
        return stripped
    try:
        value = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(value, bytes):
        return None
    try:
        return value.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None


LATEX_FRAC_RE = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
LATEX_SQRT_RE = re.compile(r"\\sqrt\{([^{}]*)\}")


def latex_to_python(value: str) -> str:
    """A LaTeX final answer in mathematics_dataset's Python-ish syntax.

    Bridges the a1 teacher's rendering (``-96a^2``,
    ``\\dfrac{1}{k^{138}}``) to the canonical answer register
    (``-96*a**2``, ``1/k**138``) far enough for sympy to compare values.
    Innermost-first fraction rewriting handles nesting.
    """
    value = re.sub(r"\\[dt]frac", r"\\frac", value)
    # Innermost-first to a joint fixpoint: fractions, roots, and braced
    # exponents nest through EACH OTHER (``\frac{1}{v^{\frac{51}{2}}}``),
    # so no single rewrite can finish before the others start.
    while True:
        rewritten = LATEX_FRAC_RE.sub(r"((\1)/(\2))", value)
        rewritten = LATEX_SQRT_RE.sub(r"sqrt(\1)", rewritten)
        rewritten = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", rewritten)
        if rewritten == value:
            break
        value = rewritten
    for latex, python in (
        ("\\left", ""), ("\\right", ""), ("\\cdot", "*"), ("\\times", "*"),
        ("\\pi", "pi"), ("\\,", ""), ("\\!", ""),
    ):
        value = value.replace(latex, python)
    value = value.replace("^", "**")
    value = value.replace("{", "(").replace("}", ")")
    value = re.sub(r"(\d)\s*([a-zA-Z(])", r"\1*\2", value)
    value = re.sub(r"\)\s*([a-zA-Z(])", r")*\1", value)
    return value


def symbolic_agree(candidate: str, truth: str) -> bool:
    """Numeric value equality for the symbolic tail of a1's answers.

    Deliberately NOT sympy's ``simplify`` or ``equals``: both take
    unbounded rewriting paths on this data (``w**(-3058)``-scale
    exponents), and every genuinely wrong teacher answer — half the set
    — reaches this tier and would pay that cost just to be dropped,
    which made the corpus build hour-scale twice. Instead: evaluate both
    expressions at fixed rational points under mpmath (arbitrary
    exponent range, bounded precision) and compare with relative
    tolerance. False positives need agreement at every point — with the
    exact tiers before this one, acceptable for a dataset filter.
    """
    if len(candidate) > 128 or len(truth) > 128:
        return False
    import sympy

    try:
        parsed_candidate = sympy.sympify(
            latex_to_python(candidate), rational=True
        )
        parsed_truth = sympy.sympify(truth, rational=True)
    except Exception:
        # sympify raises a zoo of parse/eval errors on prose; every one
        # of them just means "not comparable symbolically".
        return False
    if not isinstance(parsed_candidate, sympy.Expr) or not isinstance(
        parsed_truth, sympy.Expr
    ):
        # sympify returns tuples for comma lists, booleans for
        # relational strings — only plain expressions are comparable.
        return False
    if parsed_candidate == parsed_truth:
        return True
    symbols = sorted(
        parsed_candidate.free_symbols | parsed_truth.free_symbols, key=str
    )
    # Two point sets so a coincidental root of the difference at one
    # point cannot fake agreement; non-integer bases dodge the common
    # "equal at small integers" coincidences.
    for offset in (sympy.Rational(3, 7), sympy.Rational(13, 5)):
        substitutions = {
            symbol: offset + index for index, symbol in enumerate(symbols)
        }
        try:
            candidate_value = parsed_candidate.subs(substitutions).evalf(50)
            truth_value = parsed_truth.subs(substitutions).evalf(50)
            difference = sympy.Abs(candidate_value - truth_value)
            scale = sympy.Max(
                sympy.Abs(candidate_value),
                sympy.Abs(truth_value),
                sympy.Float("1e-40"),
            )
            if not bool(difference / scale < sympy.Float("1e-30")):
                return False
        except Exception:
            # Undefined at the probe point, complex branch, infinities:
            # not comparable — drop conservatively.
            return False
    return True


def answers_agree(candidate: str, truth: str) -> bool:
    """One extracted final vs the canonical truth, rendering-tolerant."""
    if candidate == truth:
        return True
    candidate_value = parse_numeric_answer(candidate)
    truth_value = parse_numeric_answer(truth)
    if candidate_value is not None and truth_value is not None:
        return candidate_value == truth_value
    if normalize_final_answer(candidate) == normalize_final_answer(truth):
        return True
    return symbolic_agree(candidate, truth)


def graded_correct(claimed: str, truth: str, style: str = "minerva") -> bool:
    """Grade one claimed final answer with OUR grader, window disabled
    (we constructed the Answer: line ourselves, so nothing can push it
    out of a tail window). An empty truth fails outright: the Minerva
    path grades empty-vs-empty correct (its field regex captures a
    trailing space), and an empty final would compose the zero-width
    ``<answer></answer>`` the anchored gate rejects."""
    if not truth.strip():
        return False
    return bool(
        verify_answer(f"Answer: {claimed}", truth, style, window=None)[0]
    )


# --------------------------------------------------------------------------
# Source adapters. Each yields verified {problem, reasoning, final} dicts
# and records its stage-by-stage drops in ``stats``.
# --------------------------------------------------------------------------


def rows_of(name: str) -> list[dict]:
    return pq.read_table(RELAXED_BAR / name).to_pylist()


def gsm8k_gold_bank() -> dict[str, str]:
    """Normalized GSM8K train question -> human gold final answer."""
    return {
        normalize_problem(row["question"]): (
            row["answer"].rsplit("####", 1)[-1].strip()
        )
        for row in rows_of("gsm8k_main_train.parquet")
    }


def adapt_openmath(
    stats: Counter, args: argparse.Namespace
) -> Iterator[dict]:
    """Genuine-GSM8K OpenMathInstruct-2 rows, verified against OUR gold.

    ``problem_source == "augmented_gsm8k"`` is EXCLUDED: its
    expected_answer is the teacher's own boxed value by construction
    (red-team finding — 13.9% decimal finals where real GSM8K golds are
    always integers, plus hand-confirmed ill-posed problems with
    plausible integer labels no filter can see). Genuine rows join the
    local GSM8K train gold, so the label is human and the filter is
    provably live (it drops real teacher errors).
    """
    gold = gsm8k_gold_bank()
    for row in rows_of("openmathinstruct2_gsm8k_band.parquet"):
        if row["problem_source"] != "gsm8k":
            stats["openmath_gsm8k/augmented_excluded"] += 1
            continue
        problem = row["problem"].strip()
        truth = gold.get(normalize_problem(problem))
        if truth is None:
            stats["openmath_gsm8k/no_gold_join_dropped"] += 1
            continue
        solution = row["generated_solution"].strip()
        boxed = last_boxed(solution)
        if boxed is None:
            stats["openmath_gsm8k/no_boxed_dropped"] += 1
            continue
        if not graded_correct(boxed, truth):
            stats["openmath_gsm8k/wrong_dropped"] += 1
            continue
        yield {"problem": problem, "reasoning": solution, "final": truth}


def adapt_a1_deepmind(
    stats: Counter, args: argparse.Namespace
) -> Iterator[dict]:
    for row in rows_of("a1_math_deepmind.parquet"):
        problem = unwrap_bytes_literal(row["question"] or "")
        truth = unwrap_bytes_literal(row["answer"] or "")
        if not problem or not truth:
            stats["a1_deepmind/unwrap_failed_dropped"] += 1
            continue
        solution = (row["deepseek_solution"] or "").strip()
        if not solution:
            stats["a1_deepmind/no_solution_dropped"] += 1
            continue
        candidates = [
            strip_latex_wrappers(value)
            for value in (
                extract_answer_statement(solution),
                last_boxed(solution),
            )
            if value
        ]
        # Agreement filter for the ~46% teacher-error rate (measured on
        # a 600-row seeded sample; spot-checked by hand). Lenient on
        # RENDERING (``\frac{1}{2}`` vs ``1/2`` is not a teacher error;
        # Fraction equality bridges exactly what Minerva's string
        # normalization does not), strict on VALUE — and the composed
        # final is always the canonical truth string, matching the
        # exact-style grading these rows get in RL, so leniency here
        # cannot corrupt a label.
        if not any(answers_agree(candidate, truth) for candidate in candidates):
            stats["a1_deepmind/disagrees_dropped"] += 1
            continue
        # The trailing answer block is stripped centrally in
        # screen_candidates (single application keeps the cascade cap
        # meaningful); empties are counted there too.
        yield {
            "problem": problem,
            "reasoning": solution,
            "final": truth,
        }


ANSWER_STATEMENT_RE = re.compile(
    r"(?i)\*{0,2}answer\*{0,2}\s*:\*{0,2}\s*([^\n]+)"
)


def extract_answer_statement(solution: str) -> str | None:
    hits = ANSWER_STATEMENT_RE.findall(solution[-400:])
    return hits[-1].strip() if hits else None


def adapt_had653(
    stats: Counter, args: argparse.Namespace
) -> Iterator[dict]:
    gold = gsm8k_gold_bank()
    for row in rows_of("had653_gsm8k_openmath.parquet"):
        problem = row["question"].strip()
        truth = gold.get(normalize_problem(problem))
        if truth is None:
            # final_answer is teacher-derived — no independent ground
            # truth exists for these rows, so they cannot be verified.
            stats["had653_gold/no_gold_join_dropped"] += 1
            continue
        cot = row["cot"]
        body_part, marker, claimed = cot.rpartition("Answer:")
        reasoning = body_part.partition("Reasoning:")[2].strip()
        claimed_lines = [
            line.strip() for line in claimed.splitlines() if line.strip()
        ]
        if not marker or not reasoning or not claimed_lines:
            stats["had653_gold/malformed_cot_dropped"] += 1
            continue
        if not graded_correct(claimed_lines[0], truth):
            stats["had653_gold/wrong_dropped"] += 1
            continue
        yield {"problem": problem, "reasoning": reasoning, "final": truth}


def adapt_sxiong(
    stats: Counter, args: argparse.Namespace
) -> Iterator[dict]:
    for row in rows_of("sxiong_synthetic_math.parquet"):
        if row["level"] not in ("Level 1", "Level 2", "Level 3"):
            stats["sxiong_l13/level_excluded"] += 1
            continue
        solution = row["solution"].strip()
        boxed = last_boxed(solution)
        if boxed is None:
            stats["sxiong_l13/no_boxed_dropped"] += 1
            continue
        truth = row["answer"].strip()
        if not graded_correct(boxed, truth):
            stats["sxiong_l13/wrong_dropped"] += 1
            continue
        yield {
            "problem": row["problem"].strip(),
            "reasoning": solution,
            "final": truth,
        }


def adapt_socratic(
    stats: Counter, args: argparse.Namespace
) -> Iterator[dict]:
    for row in rows_of("gsm8k_socratic_train.parquet"):
        body, marker, final = row["answer"].rpartition("####")
        final = final.strip()
        reasoning = CALCULATOR_RE.sub("", body).strip()
        if not marker or not reasoning or not final:
            stats["gsm8k_socratic/malformed_dropped"] += 1
            continue
        if not trace_reaches(final, reasoning[-300:]):
            stats["gsm8k_socratic/inconclusive_dropped"] += 1
            continue
        yield {
            "problem": row["question"].strip(),
            "reasoning": reasoning,
            "final": final,
        }


def adapt_k3(stats: Counter, args: argparse.Namespace) -> Iterator[dict]:
    path = Path(args.k3_jsonl) if args.k3_jsonl else None
    if path is None or not path.exists():
        print(
            "WARNING: no K3 trace JSONL "
            f"({path or '--k3-jsonl unset'}) — the self-generated core "
            "is missing from this build"
        )
        return
    # Lazy: generate_k3_traces pulls the RL trainer (torch) in; the
    # importers of this module's suffix constants never need that.
    from postraining.generate_k3_traces import GENERATION_SCHEMA

    channel_field = (
        "visible_prose"
        if args.k3_think_channel == "visible"
        else "reasoning_channel"
    )
    seen_keys: set[str] = set()
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["k3_traces/torn_line_skipped"] += 1
                continue
            if record.get("schema") != GENERATION_SCHEMA:
                stats["k3_traces/foreign_schema_skipped"] += 1
                continue
            if not record.get("correct"):
                stats["k3_traces/not_correct_skipped"] += 1
                continue
            if record["key"] in seen_keys:
                stats["k3_traces/duplicate_key_dropped"] += 1
                continue
            seen_keys.add(record["key"])
            reasoning = (record.get(channel_field) or "").strip()
            if not reasoning:
                stats[f"k3_traces/empty_{channel_field}_dropped"] += 1
                continue
            final = (record.get("final_answer") or "").strip()
            if not final or not graded_correct(
                final, record["ground_truth"], record["style"]
            ):
                # The generator graded this correct; disagreement here
                # means grader drift or a corrupt record — never trust
                # the stored verdict over a fresh grading.
                stats["k3_traces/reverify_failed_dropped"] += 1
                continue
            candidate = {
                "problem": record["problem"].strip(),
                "reasoning": reasoning,
                "final": final,
            }
            if record["key"].split("/")[0] == "deepmind":
                candidate["framing"] = "deepmind"
            yield candidate


SOURCES = {
    "openmath_gsm8k": adapt_openmath,
    "a1_deepmind": adapt_a1_deepmind,
    "had653_gold": adapt_had653,
    "sxiong_l13": adapt_sxiong,
    "gsm8k_socratic": adapt_socratic,
    "k3_traces": adapt_k3,
}


# --------------------------------------------------------------------------
# Decontamination
# --------------------------------------------------------------------------


def decontamination_text(row: dict) -> str:
    """Problem text of one target row, instruction boilerplate stripped.

    The DAPO instruction sentences appear in every bench prompt; training
    problems never contain them, so leaving them in could not cause a
    false hit — stripping just keeps the n-gram set honest. Tolerant of
    templates bare_problem does not know (AIME variants): falls back to
    the raw content, unlike the fail-loud teacher-prompt path.
    """
    if "question" in row:
        return row["question"]
    from postraining.generate_k3_traces import bare_problem

    try:
        return bare_problem(row)
    except ValueError:
        return "".join(message["content"] for message in row["prompt"])


def word_ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i: i + n]) for i in range(len(words) - n + 1)}


def build_decontamination_index() -> tuple[set[str], set[tuple[str, ...]]]:
    """(exact normalized texts, word 8-grams) over every eval target.

    Exact match exists because the deepmind bench problems ("Work out
    64339656 - 0.") are far too short for any n-gram to catch.
    """
    exact: set[str] = set()
    ngrams: set[tuple[str, ...]] = set()
    for target in DECONTAMINATION_TARGETS:
        if target.name == "gsm8k_test_questions.parquet":
            rows = pq.read_table(target).to_pylist()
        else:
            rows = load_unique_math_rows(target)
        for row in rows:
            text = decontamination_text(row)
            exact.add(" ".join(text.split()).lower())
            ngrams |= word_ngrams(text)
    return exact, ngrams


def contaminated(
    problem: str, exact: set[str], ngrams: set[tuple[str, ...]]
) -> bool:
    if " ".join(problem.split()).lower() in exact:
        return True
    return not word_ngrams(problem).isdisjoint(ngrams)


def screen_candidates(
    source: str,
    adapted: Iterator[dict],
    stats: Counter,
    exact: set[str],
    ngrams: set[tuple[str, ...]],
) -> Iterator[dict]:
    """Source-independent screens between adaptation and composition."""
    for candidate in adapted:
        # No source may teach an Answer: field inside the think fence —
        # the v2 think-only gate grades by that field's position, and
        # it is dead weight under the anchored gate.
        candidate["reasoning"] = strip_trailing_answer_lines(
            candidate["reasoning"]
        )
        # An empty final would compose a zero-width <answer></answer> —
        # the exact shape the anchored gate rejects (inner >= 1).
        if not (
            candidate["problem"]
            and candidate["reasoning"]
            and candidate["final"]
        ):
            stats[f"{source}/empty_field_dropped"] += 1
            continue
        if any(
            fence in text
            for text in (
                candidate["problem"],
                candidate["reasoning"],
                candidate["final"],
            )
            for fence in FENCE_STRINGS
        ):
            stats[f"{source}/fence_string_dropped"] += 1
            continue
        if contaminated(candidate["problem"], exact, ngrams):
            stats[f"{source}/contaminated_dropped"] += 1
            continue
        yield candidate


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def output_manifest_path(output: Path) -> Path:
    return output.with_suffix(".manifest.json")


def require_fresh_outputs(output: Path) -> Path:
    """Refuse to overwrite a corpus or detach it from its build manifest."""
    manifest = output_manifest_path(output)
    existing = [path for path in (output, manifest) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing corpus artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--think-tags",
        action="store_true",
        help="fence reasoning in <think>/</think> special tokens "
        f"and write {THINK_OUTPUT}",
    )
    parser.add_argument(
        "--answer-tags",
        action="store_true",
        help="additionally fence the final answer in <answer>/</answer> "
        f"(anchored gate shape) and write {ANSWER_OUTPUT}; "
        "requires --think-tags",
    )
    parser.add_argument(
        "--k3-jsonl",
        default=str(DATA / "k3_traces.jsonl"),
        help="generate_k3_traces output; missing file = build without "
        "the self-generated core (warned)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Immutable output parquet. Defaults by fence mode; an existing "
            "parquet or sibling .manifest.json is never overwritten."
        ),
    )
    parser.add_argument(
        "--k3-think-channel",
        choices=("visible", "reasoning"),
        default="visible",
        help="which K3 channel becomes the think span: 'visible' clean "
        "prose (matches the blend register) or the native 'reasoning' "
        "channel (telegraphic, several times longer — ablate first)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.answer_tags and not args.think_tags:
        parser.error("--answer-tags requires --think-tags")
    output = Path(args.output) if args.output else (
        ANSWER_OUTPUT if args.answer_tags else THINK_OUTPUT if args.think_tags else OUTPUT
    )
    try:
        manifest_path = require_fresh_outputs(output)
    except FileExistsError as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2BPETokenizer(
        think_tokens=args.think_tags, answer_tokens=args.answer_tags
    )
    exact, ngrams = build_decontamination_index()
    print(
        f"decontamination index: {len(exact)} exact texts, "
        f"{len(ngrams)} {NGRAM}-grams"
    )
    print(
        "LICENSE NOTE: a1_math_deepmind declares no license and HAD653's "
        "is a placeholder — resolve before shipping weights trained on "
        "this corpus (NOTES.md)."
    )

    records = []
    stats: Counter[str] = Counter()
    lengths: list[int] = []
    seen_documents: set[str] = set()
    for source, adapt in SOURCES.items():
        source_started = time.perf_counter()
        candidates = []
        candidates = list(
            screen_candidates(source, adapt(stats, args), stats, exact, ngrams)
        )
        cap = SOURCE_CAPS.get(source)
        if cap is not None and len(candidates) > cap:
            stats[f"{source}/cap_sampled_out"] += len(candidates) - cap
            candidates = random.Random(args.seed).sample(candidates, cap)
        for candidate in candidates:
            # Applied after decontamination; stored identities stay bare and
            # independent of their source's legacy prompt wrapper.
            try:
                problem = canonical_problem(candidate["problem"])
            except ValueError:
                stats[f"{source}/empty_problem_after_canonicalization"] += 1
                continue
            document = compose_document(
                problem,
                candidate["reasoning"],
                candidate["final"],
                args.think_tags,
                answer_tags=args.answer_tags,
            )
            if document in seen_documents:
                stats[f"{source}/duplicate_dropped"] += 1
                continue
            seen_documents.add(document)
            token_count = len(tokenizer.encode(document)) + 1  # BOS
            if token_count > MAX_DOC_TOKENS:
                stats[f"{source}/too_long_dropped"] += 1
                continue
            records.append(
                {
                    "source": source,
                    "problem": problem,
                    "document": document,
                    "final_answer": candidate["final"],
                    "verified": True,
                    "doc_tokens": token_count,
                }
            )
            stats[f"{source}/kept"] += 1
            lengths.append(token_count)
        print(
            f"{source}: {stats[f'{source}/kept']} kept, "
            f"{time.perf_counter() - source_started:.0f}s",
            flush=True,
        )

    table = pa.Table.from_pylist(records)
    staging = output.with_name(output.name + f".{os.getpid()}.tmp")
    pq.write_table(table, staging)
    os.replace(staging, output)
    source_counts = dict(
        sorted(Counter(record["source"] for record in records).items())
    )
    input_paths = sorted(RELAXED_BAR.glob("*.parquet"))
    k3_path = Path(args.k3_jsonl) if args.k3_jsonl else None
    if k3_path is not None and k3_path.is_file():
        input_paths.append(k3_path)
    manifest = {
        "schema": CORPUS_MANIFEST_SCHEMA,
        "answer_fence_prompt_schema": (
            ANSWER_FENCE_PROMPT_SCHEMA if args.answer_tags else None
        ),
        "output": str(output),
        "output_sha256": file_sha256(output),
        "documents": len(records),
        "source_counts": source_counts,
        "stats": dict(sorted(stats.items())),
        "doc_tokens": {
            "total": sum(lengths),
            "maximum": max(lengths, default=0),
        },
        "inputs": {str(path): file_sha256(path) for path in input_paths},
        "args": vars(args),
    }
    manifest_staging = manifest_path.with_name(
        manifest_path.name + f".{os.getpid()}.tmp"
    )
    manifest_staging.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(manifest_staging, manifest_path)
    print(f"wrote {output} with {len(records)} documents")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    lengths.sort()
    if lengths:
        p = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))]
        print(
            f"doc tokens p10 {p(0.1)} p50 {p(0.5)} p90 {p(0.9)} "
            f"p99 {p(0.99)} max {lengths[-1]} "
            f"total {sum(lengths)/1e6:.1f}M"
        )


if __name__ == "__main__":
    main()

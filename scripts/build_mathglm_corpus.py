"""Render the MathGLM arithmetic corpus into pretraining documents.

MathGLM ships one bare expression chain per line::

    5+4/2*1=5+2*1=5+2=7

Every one of those lines is refused by this repository's corpus quality gate,
and not marginally: over a 200,000-row sample, 61.44% fail ``too_short`` (the
math domain wants 100 characters) and the remaining 38.56% fail
``low_alpha_fraction``, because *not one row in the file contains a single
alphabetic character*. The upstream data is unusable here as text; what is
valuable is its content, which is the axis the generated drills lack -- 1 to
26 chained operations under operator precedence (95.8% of rows carry 1 to 9;
the tail past 12 is 1.0%), and multiplication that is 66.9% four-digit by
four-digit against a drill corpus that had none. The operator set is wider
than the four basics: ``[]`` group like parentheses, ``^`` raises to a power
(exponents reach four digits and go negative, most often over a base of
0, 1 or -1), and a postfix ``%`` means hundredths, so ``4%`` is 0.04 and
``4.0/1%`` is 400 rather than 0.04.

So this script reframes rather than copies. Expressions are packed into
documents under one natural-language header, sized so the header's alphabetic
mass keeps every document above the gate's floor: short chains pack together,
a long chain stands alone. Nothing is estimated -- each rendered document is
checked against the builder's own ``quality_reason`` before it is written, and
the rejection counts land in the manifest.

Nor is the upstream arithmetic trusted. MathGLM's own step generator breaks on
scientific notation: it reduces ``0.018706333107955823-1.923635517236736e-06``
to ``0.018706333107955823-06``, which both invents a rewrite rule and lands on
``-5.98`` where the truth is ``0.0187``. A second family interleaves scratch
work -- a fraction chain steps aside to ``(5/2)+(1*(7/3))`` and then resumes --
so consecutive segments joined by ``=`` are not always equal. Copying either
verbatim teaches false equalities, so every row is evaluated here: a chain is
admitted only when every ``=``-separated segment has the same value -- whole
integers exactly, everything else on relative tolerance. Over the 5,000,000-row
file that rejects 1.622% (57,796 chains whose segments disagree, 23,290 that
will not evaluate, 2 unparsed) and admits 4,918,912.

The output directory is versioned and immutable; the script refuses to write
over an existing corpus. This is pure CPU text processing -- no model
executes -- so it does not go through mlq.

    python3 scripts/build_mathglm_corpus.py --source <dataset5m.txt>
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import operator
import re
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postraining.problem_registry import ProblemRegistry, problem_key  # noqa: E402
from scripts.build_k3_pretrain_dataset import quality_reason  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "mathglm" / "v6"
DEFAULT_REGISTRY = REPOSITORY_ROOT / "data" / "problem_registry" / "v1"

CORPUS_SCHEMA = "mathglm_expression_chains/v6"

# The alphabetic framing. It carries essentially all of a document's alpha
# mass, which is why documents are sized against it rather than packed to a
# fixed item count -- and why it must not be one fixed sentence. A single
# header repeated across every document teaches the wrapper, not the
# arithmetic, and makes this corpus the most templated text in the mix; the
# generated drills carry ~99,000 distinct instruction lines by comparison.
# Selection is a content hash, so the corpus stays reproducible.
HEADERS = (
    "Evaluate each expression one operation at a time, following operator "
    "precedence.",
    "Work through each calculation step by step, taking multiplication and "
    "division before addition and subtraction.",
    "Each expression below is reduced one operation at a time until a single "
    "number is left.",
    "Simplify each of these expressions, taking the operations in precedence "
    "order.",
    "A short set of arithmetic exercises, each worked one operation at a time.",
    "Resolve the operations in each expression in the order the precedence "
    "rules require.",
    "Practice: reduce every expression below to a single value, showing each "
    "step.",
    "The following expressions are evaluated step by step, innermost "
    "precedence first.",
    "Careful arithmetic: each line below removes exactly one operation from "
    "the expression.",
    "Worked arithmetic. Multiplication and division bind more tightly than "
    "addition and subtraction.",
)

PROMPTS = (
    "What is {expression}?",
    "Evaluate {expression}.",
    "Compute {expression}.",
    "Work out {expression}.",
    "Simplify {expression}.",
    "Find the value of {expression}.",
    "What does {expression} come to?",
    "Reduce {expression} to a single number.",
)

# Sometimes stated, sometimes not, so the step count is a fact the model can
# read rather than a slot it learns to expect in every item. Every phrasing
# here is plural, so a note is only attached to a chain of two steps or more:
# "1 steps, one operation each" is the kind of line that teaches broken
# agreement to a model whose whole job downstream is reading English.
STEP_NOTES = (
    "",
    "",
    "",
    "This takes {steps} steps.",
    "There are {steps} operations to remove.",
    "{steps} steps, one operation each.",
)
MIN_NOTED_STEPS = 2

# `Answer: <value>` is deliberately NOT varied. It is the convention shared by
# openmath_instruct, the generated drills and the post-training episode
# contract, and it is the only line downstream parsing depends on.
ANSWER_PREFIX = "Answer: "

# Pack until a document reaches MIN_CHARS; never start a new item once there.
# A single chain longer than the target simply becomes its own document, which
# is the case that would otherwise dilute the framing below the alpha floor.
MIN_CHARS = 300

# Documents per parquet row group while streaming the corpus to disk.
WRITE_BATCH = 50_000


def choose(pool: tuple[str, ...], *parts: str) -> str:
    """Deterministic pick, keyed on content so rebuilds are reproducible."""
    digest = hashlib.blake2b(
        "\x00".join(parts).encode(), digest_size=8
    ).digest()
    return pool[int.from_bytes(digest, "little") % len(pool)]


class ChainError(Exception):
    """A row that does not state well-formed, true arithmetic."""


# `4%` is hundredths, and the whole literal is the operand: `9598%` is 95.98,
# not `959` followed by `8%`. Anchoring on the full number rather than the
# trailing digit is the difference between those two readings.
PERCENT = re.compile(r"(\d+(?:\.\d+)?(?:e[-+]?\d+)?)%")

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

# A compute valve, not a quality rule. Only a large *positive* exponent over a
# base above one materialises a huge exact integer; `46^-94` is a float and
# `1^2864` is one, and gating those threw away tens of thousands of rows that
# the segment comparison below is there to judge. The ceiling sits at the edge
# of what a float can hold, past which no comparison is possible anyway.
MAX_RESULT_DIGITS = 300

# Segment values are compared as floats, so the tolerance has to absorb the
# reordering between MathGLM's stepwise evaluation and this one, without
# admitting a genuinely different number. The absolute tolerance is zero on
# purpose: at 1e-12 every value below that floor compares equal to every
# other, which passed `46^-94 = 1/<121-digit denominator>` -- two numbers 36
# orders of magnitude apart -- as a true statement.
RELATIVE_TOLERANCE = 1e-9
ABSOLUTE_TOLERANCE = 0.0


def evaluate(segment: str) -> float:
    """Value of one chain segment, in MathGLM's operator dialect."""
    text = segment.replace("[", "(").replace("]", ")").replace("^", "**")
    text = PERCENT.sub(r"(\1/100)", text)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ChainError("unparsed_segment") from error
    try:
        return _value(tree.body)
    except (ArithmeticError, ValueError, TypeError) as error:
        raise ChainError("uncomputable_segment") from error


def same_value(left: float, right: float) -> bool:
    """Whether two segments state the same number.

    Integers are compared exactly rather than through floats. A float carries
    about sixteen significant digits, and this corpus exists to teach products
    wider than that: 7,571 rows state a product like
    ``385924542305736*3405=1314073066551031040`` whose true value ends 031080,
    and every one of them compares equal in floating point.
    """
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        )
    except OverflowError as error:
        raise ChainError("uncomputable_segment") from error


def _power(base: float, exponent: float) -> float:
    """``base ** exponent``, refused only when it would run away."""
    if abs(base) > 1 and exponent > 1:
        if exponent * math.log10(abs(base)) > MAX_RESULT_DIGITS:
            raise ChainError("power_out_of_range")
    return base**exponent


def _value(node: ast.AST) -> float:
    """Walk an arithmetic tree, refusing every node that is not arithmetic.

    The upstream charset holds no letters beyond the ``e`` of an exponent, so
    a name or a call here means the text was misread rather than evaluated.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ChainError("non_numeric_constant")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _value(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        apply = BINARY_OPERATORS.get(type(node.op))
        if apply is None:
            raise ChainError("unsupported_operator")
        left, right = _value(node.left), _value(node.right)
        if apply is operator.pow:
            return _power(left, right)
        return apply(left, right)
    raise ChainError("unsupported_syntax")


def parse_chain(line: str) -> tuple[str, list[str], str]:
    """Split ``a+b=step=result`` into its problem, rewrites, and answer.

    Raises ChainError unless every ``=``-separated segment evaluates to the
    same value. That is the whole point of the check: MathGLM ships rows whose
    steps are mangled and rows that step aside into scratch work, and both
    state an equality that is false.
    """
    parts = [part.strip() for part in line.strip().split("=")]
    if len(parts) < 2 or not all(parts):
        raise ChainError("not_a_chain")
    values = [evaluate(part) for part in parts]
    if not all(same_value(a, b) for a, b in zip(values, values[1:])):
        raise ChainError("segments_disagree")
    return parts[0], parts[1:-1], parts[-1]


def render_item(problem: str, rewrites: list[str], answer: str) -> str:
    steps = len(rewrites) + 1
    lines = [choose(PROMPTS, "prompt", problem).format(expression=problem)]
    note = choose(STEP_NOTES, "note", problem) if steps >= MIN_NOTED_STEPS else ""
    if note:
        lines.append(note.format(steps=steps))
    lines.append(problem)
    lines.extend(f"= {rewrite}" for rewrite in rewrites)
    lines.append(f"= {answer}")
    lines.append(f"{ANSWER_PREFIX}{answer}")
    return "\n".join(lines)


def documents(
    source: Path,
    excluded: frozenset[bytes],
    max_rows: int | None,
    counts: Counter,
) -> Iterator[str]:
    """Stream rendered chains packed into gate-passing documents.

    Rows are only counted as used once the document carrying them clears the
    quality gate, so ``rows_used`` describes the corpus rather than the
    intake.
    """
    buffer: list[str] = []
    buffered_chars = 0

    def flush() -> Iterator[str]:
        nonlocal buffer, buffered_chars
        if not buffer:
            return
        header = choose(HEADERS, "header", buffer[0])
        text = header + "\n\n" + "\n\n".join(buffer)
        reason = quality_reason(text, "math")
        if reason:
            counts[f"document_rejected:{reason}"] += 1
            counts["rows_dropped_with_document"] += len(buffer)
        else:
            counts["documents"] += 1
            counts["rows_used"] += len(buffer)
            yield text
        buffer, buffered_chars = [], 0

    with source.open() as handle:
        for index, line in enumerate(handle):
            if max_rows is not None and index >= max_rows:
                break
            counts["rows_read"] += 1
            try:
                problem, rewrites, answer = parse_chain(line)
            except ChainError as error:
                counts[f"rows_rejected:{error}"] += 1
                continue
            if problem_key(problem) in excluded:
                counts["rows_registry_excluded"] += 1
                continue
            item = render_item(problem, rewrites, answer)
            buffer.append(item)
            buffered_chars += len(item) + 2
            if buffered_chars >= MIN_CHARS:
                yield from flush()
    yield from flush()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="MathGLM plain-text file, one expression chain per line",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="stop after this many input lines (for smoke builds)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="problem registry whose eval/rl/sft problems this corpus must "
        "avoid; pass --no-registry only when no registry exists yet",
    )
    parser.add_argument("--no-registry", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.output.exists():
        raise SystemExit(
            f"{args.output} already exists; corpora are immutable, so build a "
            "new version directory instead"
        )
    if not args.source.is_file():
        raise SystemExit(f"no MathGLM source file at {args.source}")

    excluded: frozenset[bytes] = frozenset()
    registry_provenance: dict | None = None
    if not args.no_registry:
        if not (args.registry / "registry.parquet").exists():
            raise SystemExit(
                f"no problem registry at {args.registry}; build one with "
                "scripts/build_problem_registry.py, or pass --no-registry to "
                "build a corpus that has not been checked against the "
                "evaluation and post-training splits"
            )
        registry = ProblemRegistry.read(args.registry)
        excluded = frozenset(registry.excluded_from("pretrain"))
        registry_provenance = {
            "directory": str(args.registry),
            "registry_sha256": registry.provenance["registry_sha256"],
            "excluded_problem_keys": len(excluded),
        }
        print(f"registry: excluding {len(excluded):,} claimed problems")

    source_digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    print(f"reading {args.source} ({source_digest[:12]})", flush=True)

    counts: Counter = Counter()
    args.output.mkdir(parents=True)
    corpus_path = args.output / "mathglm.parquet"
    schema = pa.schema([pa.field("document", pa.string())])
    characters = 0
    batch: list[str] = []
    # Written in batches rather than one table: the 50M-row upstream set would
    # otherwise have to sit in memory in full before anything reached disk.
    with pq.ParquetWriter(corpus_path, schema, compression="zstd") as writer:
        for text in documents(args.source, excluded, args.max_rows, counts):
            characters += len(text)
            batch.append(text)
            if len(batch) >= WRITE_BATCH:
                writer.write_table(pa.table({"document": batch}, schema=schema))
                batch = []
        if batch:
            writer.write_table(pa.table({"document": batch}, schema=schema))
    if not counts["documents"]:
        raise SystemExit("no document survived the quality gate")
    corpus_digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    built_count = counts["documents"]
    manifest = {
        "schema": CORPUS_SCHEMA,
        "upstream": {
            "dataset": "jonathanasdf/MathGLM-dataset-5M",
            "derivation": "every 10th row of the THUDM/MathGLM 50M arithmetic "
            "pre-training set",
            "paper": "arXiv:2309.03241",
            "license": "AFL-3.0",
            "source_file": args.source.name,
            "source_sha256": source_digest,
        },
        "framing": {
            "headers": list(HEADERS),
            "prompts": list(PROMPTS),
            "step_notes": list(STEP_NOTES),
            "answer_prefix": ANSWER_PREFIX,
            "selection": "blake2b content hash; reproducible across rebuilds",
        },
        "validation": {
            "rule": "every '='-separated segment must evaluate to the same "
            "value, whole integers compared exactly; MathGLM ships mangled "
            "scientific-notation steps, fraction chains that step aside into "
            "scratch work, and wide products wrong past the sixteenth digit",
            "relative_tolerance": RELATIVE_TOLERANCE,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "percent_is_hundredths": True,
            "max_result_digits": MAX_RESULT_DIGITS,
        },
        "min_document_characters": MIN_CHARS,
        "corpus_sha256": corpus_digest,
        "documents": built_count,
        "characters": characters,
        "mean_document_characters": characters / built_count,
        "problem_registry": registry_provenance,
        "counts": dict(sorted(counts.items())),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

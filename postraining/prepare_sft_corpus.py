"""Build a large single-epoch SFT corpus in the canonical answer-fence schema.

``prepare_sft_traces``/``prepare_sft6_bare`` curate a few tens of thousands of
verified traces.  That corpus is the wrong shape for this stage: three epochs
over the retired v6 traces drove holdout completion CE from 3.9449 to 0.9009
while the derived policy still scored 0.00 on every DeepMind interpolate
module, which is memorisation of a small trace set rather than reasoning.  The
fix is more data and one pass, so this module streams a published instruction
corpus into the schema ``sft_trace_train`` already consumes.

Emitted columns match the curated corpora exactly -- ``source``, ``problem``,
``document``, ``final_answer``, ``verified``, ``doc_tokens`` -- where

    document = problem + THINK_OPEN + "\\n" + solution + "\\n" + THINK_CLOSE
               + "\\n" + ANSWER_OPEN + final_answer + ANSWER_CLOSE

because the answer-fence prompt contract appends no instruction suffix, so the
trainer recovers the completion as ``document[len(problem):]``.

Adapters declare a corpus rather than sniffing it, including its record
shape: ``parquet_columns`` for flat columnar corpora and ``ultradata_chat``
for openbmb's JSONL chat records, whose ``reasoning_content`` becomes the
``<think>`` body and whose presented answer supplies the ``<answer>`` span.  Rows are dropped, never
reshaped, when they do not fit: a document that overflows ``--seq-len`` would
be silently skipped by the trainer anyway, and a row whose problem carries a
source wrapper the canonicalizer does not recognise must fail closed.

Evaluation protection is part of the corpus contract, not a later filter, and
has two independent layers.  ``problem_source`` values naming a set this
repository evaluates on are excluded by the adapter, so a corpus that cannot
report per-row provenance cannot be admitted.  On top of that every problem
runs through ``prepare_sft_traces``' decontamination index -- exact
normalized match plus word 8-grams over the GSM8K test questions, the
DeepMind interpolate bench panel and AIME 2024/2025/2026 -- because
provenance labels only describe a row's immediate source.
OpenMathInstruct-2 is MATH-derived and MATH contains AIME problems, so the
label alone would leave AIME numbers measured off this lineage unprotected.

Fence literals are rejected for the same reason the sibling rejects them: a
literal ``<think>``/``<answer>`` inside model-generated solution text
tokenizes to a real special id and breaks the anchored gate's single-pair
invariant.

CPU-only (tokenizer + parquet); no GPU, so no mlq submission is required.

    .venv/bin/python -m postraining.prepare_sft_corpus \\
        --source openmathinstruct2 --shards 8 \\
        --output postraining/data/sft_omi2_bare_v1.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter
from multiprocessing import get_context
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from postraining.core import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    GPT2BPETokenizer,
    frame_prompt,
)
from postraining.prepare_sft_traces import (
    DECONTAMINATION_TARGETS,
    FENCE_STRINGS,
    build_decontamination_index,
    contaminated,
    word_ngrams,
)
from postraining.choice_prompt import (
    choice_trace_defect,
    concluded_label,
    cut_derivation_recap,
    split_options,
    strip_choice_framing,
)
from postraining.eval_hf_math import last_boxed_answer
from postraining.math_prompt import (
    ANSWER_FENCE_PROMPT_SCHEMA,
    ANSWER_FENCE_SUFFIX,
    strip_math_prompt_framing,
)

CORPUS_SCHEMA = "instruction_corpus_answer_fence/v2"

# Problem sets that must not reach training but whose overlap the shared
# any-8-gram index cannot judge. KodCode is an evaluation set here
# (``postraining/kodcode_eval.py``); GSM8K train is named by CLAUDE.md as a
# source whose rows are excluded from SFT corpora, and UltraData records carry
# no ``problem_source`` field to exclude it by label.
CONTAINMENT_TARGETS = (
    (Path("postraining/data/kodcode-light-rl-10k-problems.parquet"), "problem"),
    (Path("postraining/data/gsm8k_rl_prompts.parquet"), "prompt"),
)

# These need containment, not similarity. Two rules were measured and rejected
# first.
#
# A binary "any shared 8-gram" test, the rule the math index uses, rejected
# 466 of 800 distinct UltraData code problems while catching zero real
# duplicates: programming statements share boilerplate ("the first line of the
# input contains two") and their example I/O blocks produce digit-run grams
# like "1 1 1 0 1 0 1 0". Mathematics has no comparable boilerplate, which is
# why that rule is right for the math targets and wrong here.
#
# A similarity ratio -- the candidate's own grams that the index contains --
# then leaked on containment, which is the case that matters: a reference
# problem restated inside a longer one dilutes to |K|/(|K|+|U|) and slips under
# any fixed threshold once the surrounding text is large enough. Measured by
# concatenation, 56.9% of embedded reference problems were admitted, and since
# reference problems are about 3x smaller than UltraData's, dilution was the
# common case rather than the corner case.
#
# Containment against each reference problem separately -- the share of *that
# problem's* grams present in the candidate -- separates completely. Measured:
# embedded reference problems score min 1.000 (150/150 caught), while distinct
# UltraData code problems flag 0 of 400 and distinct OpenMathInstruct-2
# problems 0 of 2,475. Cost is 0.1 ms per row, and the longest posting list is
# 190, so the inverted index stays cheap.
CONTAINMENT_MIN_OVERLAP = 0.30


# A record already reduced to the three spans the contract needs, plus the
# provenance label the manifest reports and the exclusions are checked against.
@dataclass(frozen=True)
class Record:
    problem: str  # raw; canonicalization happens once, in the main loop
    solution: str  # becomes the <think> body
    answer: str  # becomes the <answer> body
    origin: str
    # Whether the math verifier can grade this row's final answer. A program
    # is graded by test execution, not answer comparison, so a code row must
    # not enter the trainer's sampling-gate panel: scored with the MATH
    # verifier it would be wrong by construction and would deflate gate
    # accuracy and the mixed-group rate the RL stage is sized from.
    gradeable: bool = True
    # Math answers are one line by contract. A program is not, and the Python
    # reward already accepts a fenced block (``normalize_python_answer``), so
    # code rows opt out of the single-line rule instead of being dropped.
    multiline_answer: bool = False
    # Whether this repository re-verified the solution. Only the verified code
    # pool is: its reference passed its tests in the reward's own sandbox.
    verified: bool = False


@dataclass(frozen=True)
class Adapter:
    """A published corpus and the exact way its rows become documents."""

    name: str
    repo: str
    revision: str
    # "parquet_columns": flat columnar rows named by the ``*_column`` fields.
    # "ultradata_chat": JSONL chat records parsed by ``parse_ultradata_chat``.
    # "code_exercise": verified code exercises (``parse_code_exercise``).
    # "choice_trace": verified distilled single-choice traces
    # (``parse_choice_trace``).
    kind: str
    shard_template: str
    shards: int
    # Local corpora are already materialised; ``repo``/``shard_template`` are
    # empty for them and ``local_parquet``/``local_shards`` is read directly.
    local_parquet: Path | None
    problem_column: str
    solution_column: str
    answer_column: str
    provenance_column: str
    # Excluded because this repository reports accuracy on these sets; an SFT
    # pass over them would make every later number a training-set score.
    excluded_provenance: frozenset[str]
    license_note: str
    # "ultradata_chat" only: directory of JSONL shards and the glob selecting
    # them, so which subtrees are admitted is declared rather than sniffed.
    local_shards: Path | None = None
    shard_glob: str = ""
    # Provenance allowlist. A denylist fails open: an unseen shard carrying a
    # renamed or newly added evaluation-derived source would be admitted by
    # default. Any label outside this set is rejected and counted, so a new
    # source shows up as a drop count to investigate rather than as training
    # data. Empty means "no allowlist declared" and is refused for adapters
    # whose provenance is the only evaluation protection.
    known_sources: frozenset[str] = frozenset()
    # "ultradata_chat" only: the domain subdirectories of ``local_shards``
    # this adapter reads, each globbed with ``shard_glob``. Declared rather
    # than globbed as ``*/``, so mirroring a new domain cannot silently widen
    # an existing adapter's shard set (and its manifest's shard hashes).
    domains: tuple[str, ...] = ()
    # Containment references screened only when an adapter that needs them
    # is in the plan, as (path, question column). A missing file refuses the
    # build: these are the only evaluation protection for corpora whose rows
    # carry no upstream provenance.
    containment_targets: tuple[tuple[Path, str], ...] = ()
    # Benchmark question stems screened by ``contains_benchmark_question``
    # when an adapter that needs them is in the plan, as (path, question
    # column). Missing files refuse the build, as above.
    question_targets: tuple[tuple[Path, str], ...] = ()
    # A per-source document-length ceiling below ``--seq-len``, counted like
    # it (document plus EOS). For a source whose long documents would take
    # the token budget of a longer-window build from the rest of the mix.
    max_doc_tokens: int | None = None
    # Single-choice sources: deduplicated by ``choice_identity`` (question
    # plus sorted option texts) and reported per option count and answer
    # label in the manifest.
    single_choice: bool = False
    # Single-choice sources whose teacher answers favour some positions: cap
    # every position per option count at (1 + ``choice_balance_tolerance``)
    # times the mean share by dropping rows (``balance_choice_answers``).
    balance_choice_answers: bool = False
    choice_balance_tolerance: float = 0.0
    # Distilled trace pools (kind "choice_trace"): the RL pool the question
    # partition was split from, which must be the one RL trains on
    # (``prepare_vapo_mixture.SOURCE_SPECS``), byte for byte.
    rl_counterpart: Path | None = None


# Held-out STEM multiple-choice benchmarks. OpenScienceReasoning-2, the
# upstream of UltraData's Knowledge RL slice, was built to target them, and
# UltraData-SFT-2605 Knowledge records name no upstream source, so text is the
# only way to keep their questions out of training. Question stems only:
# benchmark options are short, generic and, for MMLU-Pro, augmented, while a
# restated stem is what a leak looks like. GPQA extended is a superset of
# GPQA main, which contains GPQA diamond.
#
# Screened by ``contains_benchmark_question``, not the 0.30 containment rule
# above. Exam stems are short and formulaic: a third of MMLU questions have
# under 8 grams of their own, so "which of the following statements best
# describes the" is a third of a reference, and the 0.30 rule flagged 86
# UltraData-RL Knowledge rows of which every inspected one shared only that
# boilerplate.
STEM_QUESTION_TARGETS = (
    (Path("postraining/data/stem_eval_decontam/cais__mmlu/all/"
          "test-00000-of-00001.parquet"), "question"),
    (Path("postraining/data/stem_eval_decontam/cais__mmlu/all/"
          "validation-00000-of-00001.parquet"), "question"),
    (Path("postraining/data/stem_eval_decontam/cais__mmlu/all/"
          "dev-00000-of-00001.parquet"), "question"),
    (Path("postraining/data/stem_eval_decontam/TIGER-Lab__MMLU-Pro/data/"
          "test-00000-of-00001.parquet"), "question"),
    (Path("postraining/data/stem_eval_decontam/TIGER-Lab__MMLU-Pro/data/"
          "validation-00000-of-00001.parquet"), "question"),
    (Path("postraining/data/stem_eval_decontam/Idavidrein__gpqa/"
          "gpqa_extended.csv"), "Question"),
    # The question as first written, before expert revision; a copy that
    # circulated earlier would restate this one.
    (Path("postraining/data/stem_eval_decontam/Idavidrein__gpqa/"
          "gpqa_extended.csv"), "Pre-Revision Question"),
    # The evaluation splits of the grade-school science sources whose train
    # splits feed the ``science_mc`` RL pool.
    *(
        (Path(f"postraining/data/stem_eval_decontam/{path}"), column)
        for path, column in (
            ("allenai__sciq/data/validation-00000-of-00001.parquet", "question"),
            ("allenai__sciq/data/test-00000-of-00001.parquet", "question"),
            *(
                (f"allenai__ai2_arc/{subset}/{split}-00000-of-00001.parquet",
                 "question")
                for subset in ("ARC-Easy", "ARC-Challenge")
                for split in ("validation", "test")
            ),
            ("allenai__openbookqa/main/validation-00000-of-00001.parquet",
             "question_stem"),
            ("allenai__openbookqa/main/test-00000-of-00001.parquet",
             "question_stem"),
        )
    ),
)
# Evaluation-only data. Nothing under it may be read as a training example:
# ``shard_paths`` and the RL mixture refuse it, and
# ``postraining/tests/test_benchmark_isolation.py`` checks every declared input.
EVALUATION_ONLY_DATA = Path("postraining/data/stem_eval_decontam")

# Function-level Python benchmarks, for code adapters. MBPP is taken whole:
# tasks 601-974 are the RL ``mbpp`` source and the rest are its evaluation
# splits. Both files are built from pinned sources by
# ``prepare_ultradata_code targets``.
CODE_CONTAINMENT_TARGETS = (
    (Path("postraining/data/mbpp-problems.parquet"), "problem"),
    (Path("postraining/data/humaneval-problems.parquet"), "problem"),
)


ADAPTERS = {
    "openmathinstruct2": Adapter(
        name="openmathinstruct2",
        repo="nvidia/OpenMathInstruct-2",
        revision="469216e3f46f4dacf476b382e192485ea51a143e",
        kind="parquet_columns",
        shard_template="data/train-{index:05d}-of-00032.parquet",
        shards=32,
        local_parquet=None,
        problem_column="problem",
        solution_column="generated_solution",
        answer_column="expected_answer",
        provenance_column="problem_source",
        excluded_provenance=frozenset({"gsm8k", "augmented_gsm8k"}),
        license_note="CC-BY-4.0; solutions synthesised with Llama-3.1-405B",
    ),
    # The arithmetic capability probe scored this lineage 0.000 on all 15
    # families at every digit count, 3-digit integer addition included, and
    # 0.000 leniently as well -- the primitive is absent, not merely fragile.
    # These drills are the only corpus here that shows the working: carrying,
    # borrowing, place value and digit-by-digit long division. Their own
    # ``document`` column uses the retired plain ``Answer:`` framing, so it is
    # ignored and the document is rebuilt under the canonical contract.
    "math_drills": Adapter(
        name="math_drills",
        repo="local:data/math_drills/v4",
        revision="worked_arithmetic_drills/v3",
        kind="parquet_columns",
        shard_template="",
        shards=1,
        local_parquet=Path("data/math_drills/v4/drills.parquet"),
        problem_column="problem",
        solution_column="solution",
        answer_column="answer",
        provenance_column="family",
        excluded_provenance=frozenset(),
        license_note="generated in-repo by postraining/math_drills.py",
    ),
    # openbmb's own SFT mixture. Only the ``think`` split is admitted: the
    # canonical completion needs a reasoning span, and ``reasoning_content``
    # is exactly that, with ``content`` carrying the presented answer. The
    # ``no_think`` split has no reasoning field at all -- its assistant turn
    # is the bare answer -- so admitting it would mean synthesising a
    # <think> body, which is the kind of reshaping this module refuses.
    #
    # Math and Code are both taken. Math answers come from the balanced
    # \boxed{} span; Code answers are the final fenced program, because
    # OpenCodeInstruct-style rows have no boxed answer and the Python reward
    # already normalises a fenced block.
    #
    # Its Math half is substantially OpenMathInstruct-2 derived, which
    # overlaps the ``openmathinstruct2`` adapter. That is handled by building
    # one corpus in a single invocation: ``seen`` spans adapters, so the
    # duplicate lands under whichever source is listed first and is counted
    # as ``duplicate_problem`` thereafter.
    "ultradata_sft_2605": Adapter(
        name="ultradata_sft_2605",
        repo="openbmb/UltraData-SFT-2605",
        revision="affda6aca75e7cff78e73f93ad08d4c3b01f097c",
        kind="ultradata_chat",
        shard_template="",
        shards=0,
        local_parquet=None,
        local_shards=Path(
            "postraining/data/instruction_corpus_shards/"
            "ultradata_sft_2605_hf/data/think"
        ),
        # Sorted domain order reproduces the former "*/*.jsonl" shard order
        # over the Math/Code mirror exactly.
        domains=("Code", "Math"),
        shard_glob="*.jsonl",
        problem_column="",
        solution_column="",
        answer_column="",
        provenance_column="source",
        # KodCode is an evaluation set in this repository
        # (``postraining/kodcode_eval.py``), so its SFT rows cannot be
        # trained on. The remaining evaluation sets -- GSM8K, AIME, the
        # DeepMind interpolate panel -- have no matching ``source`` label
        # here and are caught textually by the decontamination index.
        excluded_provenance=frozenset({"KodCode-V1-SFT", "gsm8k"}),
        # Observed across the mirrored think shards. A source value may itself
        # contain a slash ("Nemotron-Cascade-SFT-Stage-1/math"), so the full
        # "{domain}/{source}" label is what is declared and matched.
        known_sources=frozenset(
            {
                "Math/OpenMathInstruct-2",
                "Math/Nemotron-Cascade-SFT-Stage-1/math",
                "Code/OpenCodeReasoning",
            }
        ),
        license_note="apache-2.0; think split only (reasoning_content)",
    ),
    # UltraData's Knowledge (STEM / exam QA) think split, a separate adapter
    # so its share can be capped independently: the 5,120-token stage 1 lost
    # to job 9040 because long think traces diluted the drills (NOTES
    # 2026-09-22), and these traces are longer still.
    #
    # Only templated single-choice rows are admitted (``choice_prompt``): the
    # two-line answer-format header is removed, the options must match the
    # labels it declares, and the label the presented answer concludes with
    # becomes the <answer> span. Free-form rows (about half the split) end in
    # prose, and untemplated multiple-choice rows conclude in dozens of
    # unanchored styles, so neither has a final answer this module can
    # extract without guessing.
    #
    # Every record's ``source`` is the dataset name itself, so provenance
    # cannot exclude anything; protection is textual, and the STEM benchmark
    # question targets are mandatory for this adapter.
    #
    # Documents are capped at 2,048 tokens whatever the build's window. The
    # templated rows' median document is 5,070 tokens: a 5,120 window admits
    # 77.9k of them and 239M tokens, as much as the whole 1,024-token
    # canonical mix, while 2,048 admits 18.1k and 28.6M. Reasoning that
    # discusses the removed header is then dropped (``cites_answer_format``),
    # which leaves about 2.8k documents and 4.3M tokens (NOTES 2026-09-23).
    # A "**Step-by-step derivation:**" recap after the conclusion is cut
    # (``cut_derivation_recap``); the rest must pass the mechanical screens
    # of ``choice_trace_defect``, which drop any other drafted response.
    # There is no gold label to verify against: the label is the presented
    # answer's own, and the reasoning is not parsed for a conclusion. For
    # that reason no corpus built from this adapter is admitted to the
    # science_mc SFT mix (NOTES 2026-09-24).
    "ultradata_sft_2605_knowledge": Adapter(
        name="ultradata_sft_2605_knowledge",
        repo="openbmb/UltraData-SFT-2605",
        revision="affda6aca75e7cff78e73f93ad08d4c3b01f097c",
        kind="ultradata_chat",
        shard_template="",
        shards=0,
        local_parquet=None,
        local_shards=Path(
            "postraining/data/instruction_corpus_shards/"
            "ultradata_sft_2605_hf/data/think"
        ),
        domains=("Knowledge",),
        shard_glob="*.jsonl",
        problem_column="",
        solution_column="",
        answer_column="",
        provenance_column="source",
        excluded_provenance=frozenset(),
        known_sources=frozenset({"Knowledge/UltraData-sft-2605"}),
        question_targets=STEM_QUESTION_TARGETS,
        max_doc_tokens=2048,
        single_choice=True,
        balance_choice_answers=True,
        # No answer position above 1.25x chance per option count, by
        # dropping rows only, except that a position may always hold ceil(n
        # / k) rows, so a small option count can exceed it
        # (``_capped_counts``). Exact balance kept
        # 235 of 1,526 screened rows (NOTES 2026-09-23/24). Late positions
        # stay rare, and the manifest reports each option count's modal and
        # least share.
        choice_balance_tolerance=0.25,
        license_note=(
            "apache-2.0; think split, templated single-choice rows only"
        ),
    ),
    # openbmb/UltraData-Code L3/py exercises, admitted only through the
    # verified pool ``prepare_ultradata_code build`` writes: reference passes
    # the RL reward's AST policy and its own tests in the sandbox, and the
    # pool shares no problem identity (entry-point names) or near-duplicate
    # cluster with the ``ultradata_code_l3`` RL rows. ``analysis`` becomes the
    # <think> body and the fenced reference program the <answer> span, the
    # same shape the Python reward normalises. The 2,048-token ceiling keeps
    # 97.6% of verified exercises and stops the long tail diluting a mix.
    "ultradata_code_l3": Adapter(
        name="ultradata_code_l3",
        repo="openbmb/UltraData-Code",
        revision="85182d829f2ce7ea07cca72ebfc509deea1d9f5f",
        kind="code_exercise",
        shard_template="",
        shards=1,
        local_parquet=Path(
            "postraining/data/ultradata-code-l3-v3-sft-pool.parquet"
        ),
        problem_column="problem",
        solution_column="analysis",
        answer_column="solution",
        provenance_column="provenance",
        excluded_provenance=frozenset(),
        known_sources=frozenset({"L3/py"}),
        containment_targets=CODE_CONTAINMENT_TARGETS,
        max_doc_tokens=2048,
        license_note=(
            "apache-2.0 plus source-repository licenses; L3/py verified pool"
        ),
    ),
    # Worked solutions to the science_mc SFT question partition (ARC, OBQA,
    # SciQ train), distilled from Qwen3.8-27B by ``generate_choice_traces``.
    # The partition shares no question component with the ``science_mc`` RL
    # pool (``rl_counterpart``). A trace is admitted only if the teacher's
    # final letter is the gold one under the RL reward's own grader, enough
    # of its question's samples were correct (self-consistency), and the
    # trace passed the mechanical ``choice_trace_defect``, which the parser
    # re-checks here. The prompt is the rendered problem the RL policy sees,
    # so the letter span is the graded answer. The split rendered every
    # question's options in a content-seeded uniform order before the
    # teacher saw them, so the answers need no balancing and no trace is
    # relabelled. 1,024 tokens keeps the source inside the job 9040 window
    # (NOTES 2026-09-24).
    "science_mc_traces": Adapter(
        name="science_mc_traces",
        repo="local:postraining/generate_choice_traces.py",
        revision="choice_trace_pool/v2",
        kind="choice_trace",
        shard_template="",
        shards=1,
        local_parquet=Path("postraining/data/science-mc-traces-v1-pool.parquet"),
        rl_counterpart=Path("postraining/data/science-mc-rl-v5.parquet"),
        problem_column="problem",
        solution_column="reasoning",
        answer_column="letter",
        provenance_column="source",
        excluded_provenance=frozenset(),
        known_sources=frozenset(
            {"arc_challenge", "arc_easy", "openbookqa", "sciq"}
        ),
        question_targets=STEM_QUESTION_TARGETS,
        max_doc_tokens=1024,
        single_choice=True,
        license_note=(
            "questions: ARC CC BY-SA 4.0, OpenBookQA Apache-2.0, SciQ CC BY-NC "
            "3.0 (non-commercial); traces: Qwen3.8-27B (apache-2.0) outputs"
        ),
    ),
}

PYTHON_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def parse_parquet_row(adapter: Adapter, row: dict) -> tuple[Record | None, str]:
    answer = (row.get(adapter.answer_column) or "").strip()
    solution = (row.get(adapter.solution_column) or "").strip()
    if not answer or not solution:
        return None, "empty_answer_or_solution"
    origin = (row.get(adapter.provenance_column) or "").strip()
    if not origin:
        # A corpus that cannot report this row's provenance cannot be
        # admitted: provenance is one of the two evaluation-protection
        # layers, so a missing label is a rejection, not a pass.
        return None, "missing_provenance"
    return (
        Record(
            problem=row[adapter.problem_column] or "",
            solution=solution,
            answer=answer,
            origin=origin,
        ),
        "",
    )


def parse_ultradata_chat(adapter: Adapter, row: dict) -> tuple[Record | None, str]:
    """Reduce one UltraData chat record to the three contract spans."""

    messages = row.get("messages") or []
    if [message.get("role") for message in messages] != ["user", "assistant"]:
        # Multi-turn or system-prefixed records do not fit a single-prompt
        # episode; reshaping them into one is out of scope for this schema.
        return None, "unexpected_turn_structure"
    if row.get("think_type") != "think":
        return None, "not_think_split"
    problem = (messages[0].get("content") or "").strip()
    solution = (messages[1].get("reasoning_content") or "").strip()
    presented = (messages[1].get("content") or "").strip()
    if not problem or not solution or not presented:
        return None, "empty_answer_or_solution"
    domain = (row.get("domain") or "").strip()
    source = (row.get(adapter.provenance_column) or "").strip()
    if not domain or not source:
        return None, "missing_provenance"
    origin = f"{domain}/{source}"
    if domain == "Math":
        boxed = last_boxed_answer(presented)
        # A literal \boxed{} yields "" rather than None, which would build a
        # zero-width <answer></answer> span -- the exact shape the anchored
        # gate rejects, taught as a target. Observed in the real corpus at
        # roughly one row in a thousand, so it must be checked, not assumed.
        if not boxed:
            return None, "no_boxed_answer"
        return Record(problem, solution, boxed, origin), ""
    if domain == "Code":
        blocks = PYTHON_FENCE.findall(presented)
        if not blocks:
            return None, "no_python_block"
        program = blocks[-1].strip()
        if not program:
            return None, "no_python_block"
        return (
            Record(
                problem,
                solution,
                f"```python\n{program}\n```",
                origin,
                gradeable=False,
                multiline_answer=True,
            ),
            "",
        )
    if domain == "Knowledge":
        try:
            framing = strip_choice_framing(problem)
        except ValueError:
            return None, "malformed_choice_framing"
        if framing is None:
            return None, "not_templated_choice"
        label = concluded_label(presented, framing)
        if label is None:
            return None, "no_choice_conclusion"
        solution = cut_derivation_recap(solution)
        if not solution:
            return None, "empty_answer_or_solution"
        if defect := choice_trace_defect(solution, split_options(framing.problem)):
            return None, defect
        return Record(framing.problem, solution, label, origin), ""
    return None, "unsupported_domain"


def parse_code_exercise(adapter: Adapter, row: dict) -> tuple[Record | None, str]:
    """A verified exercise: task, worked analysis, reference program."""

    problem = (row.get(adapter.problem_column) or "").strip()
    analysis = (row.get(adapter.solution_column) or "").strip()
    program = (row.get(adapter.answer_column) or "").strip()
    if not problem or not analysis or not program:
        return None, "empty_answer_or_solution"
    if "```" in program:
        # A nested fence would end the answer's code block early.
        return None, "program_contains_fence"
    origin = (row.get(adapter.provenance_column) or "").strip()
    if not origin:
        return None, "missing_provenance"
    return (
        Record(
            problem,
            analysis,
            f"```python\n{program}\n```",
            origin,
            gradeable=False,
            multiline_answer=True,
            verified=True,
        ),
        "",
    )


def parse_choice_trace(adapter: Adapter, row: dict) -> tuple[Record | None, str]:
    """A distilled single-choice trace whose letter already verified."""

    problem = (row.get(adapter.problem_column) or "").strip()
    reasoning = (row.get(adapter.solution_column) or "").strip()
    letter = (row.get(adapter.answer_column) or "").strip()
    if not problem or not reasoning or not letter:
        return None, "empty_answer_or_solution"
    if row.get("verified") is not True:
        return None, "unverified_trace"
    parsed = split_options(problem)
    if parsed is None:
        return None, "not_single_choice"
    if letter not in parsed.labels:
        return None, "answer_not_a_label"
    if defect := choice_trace_defect(reasoning, parsed, letter):
        return None, defect
    origin = (row.get(adapter.provenance_column) or "").strip()
    if not origin:
        return None, "missing_provenance"
    # The SFT trainer's sampling gate uses the math verifier. A correct
    # letter would be scored as wrong there, as with verified code programs.
    # Keep these rows for training and holdout CE, but gate science choices
    # separately with the exact-letter evaluator after SFT.
    return Record(
        problem, reasoning, letter, origin, gradeable=False, verified=True
    ), ""


PARSERS = {
    "parquet_columns": parse_parquet_row,
    "ultradata_chat": parse_ultradata_chat,
    "code_exercise": parse_code_exercise,
    "choice_trace": parse_choice_trace,
}

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_evaluation_only(path: Path) -> Path:
    """``path``, unless it lies under ``EVALUATION_ONLY_DATA``."""

    if path.resolve().is_relative_to(EVALUATION_ONLY_DATA.resolve()):
        raise ValueError(
            f"{path} is evaluation-only data (benchmark screening targets); "
            "it can never be a training input"
        )
    return path


def shard_paths(adapter: Adapter, shards: int, cache: Path):
    for root in (adapter.local_shards, adapter.local_parquet, cache):
        if root is not None:
            refuse_evaluation_only(root)
    if adapter.local_shards is not None:
        paths = []
        for directory in (
            [adapter.local_shards / domain for domain in adapter.domains]
            or [adapter.local_shards]
        ):
            # Every declared domain must be mirrored: one missing directory
            # would otherwise shrink the corpus without an error.
            found = sorted(directory.glob(adapter.shard_glob))
            if not found:
                raise FileNotFoundError(
                    f"{adapter.name}: no shard matches "
                    f"{directory / adapter.shard_glob}"
                )
            paths.extend(found)
        return iter(sorted(paths))
    if adapter.local_parquet is not None:
        if not adapter.local_parquet.exists():
            raise FileNotFoundError(adapter.local_parquet)
        return iter([adapter.local_parquet])
    # A generator: a per-source cap must not download shards it never reads.
    return (fetch_shard(adapter, index, cache) for index in range(shards))


def iter_rows(path: Path):
    """Yield raw records from one shard, parquet or JSONL."""

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192):
        yield from batch.to_pylist()


def fetch_shard(adapter: Adapter, index: int, cache: Path) -> Path:
    """Download one shard into ``cache`` and return its path."""

    relative = adapter.shard_template.format(index=index)
    # The revision is part of the cache path: a cache keyed only by file name
    # would silently serve bytes fetched under an earlier upstream state, so
    # one corpus could mix shards from two revisions.
    target = cache / adapter.name / adapter.revision / Path(relative).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        url = (
            f"https://huggingface.co/datasets/{adapter.repo}/resolve/"
            f"{adapter.revision}/{relative}"
        )
        staging = target.with_name(target.name + f".{os.getpid()}.part")
        with urllib.request.urlopen(url) as response, staging.open("wb") as out:
            while chunk := response.read(4 * 1024 * 1024):
                out.write(chunk)
        staging.replace(target)
    return target


def _done(total_cap: int, total: int, source_cap: int, kept: int) -> bool:
    return bool(
        (total_cap and total >= total_cap) or (source_cap and kept >= source_cap)
    )


# Tokenizing a 60k-token reasoning trace only to discard it dominates the
# cost of a multi-GB corpus, so documents are rejected on character length
# first. The bound must never discard a document that would have fit, and
# 12 is not safe for that: GPT-2 BPE emits long run-tokens, so 50 repetitions
# of a 40-character rule line measure 13.67 characters per token. Repeated
# separators, banners and deep indentation are ordinary in code. 40 is chosen
# instead -- far above any ratio real text reaches while still rejecting the
# 35k-80k-token traces this corpus is full of, which run to hundreds of
# thousands of characters. Rejections are counted so the slack stays
# auditable.
CHARS_PER_TOKEN_BOUND = 40

# Raising the hard bound to a sound value means the 35k-80k-token traces this
# corpus is full of now get tokenized in full before being rejected, which
# dominates a multi-gigabyte build. A prefix settles it instead: BPE token
# counts are monotone in the prefix except for a single token at the split
# point, so a prefix that already exceeds the budget proves the whole
# document does. 12 characters per token is far above the ~3-4 real text
# reaches, so the prefix is long enough to overshoot whenever the document
# genuinely does not fit.
PREFIX_PROBE_CHARS_PER_TOKEN = 12


def excluded(origin: str | None, exclusions: frozenset[str]) -> bool:
    """Whether a provenance label names a set this repository evaluates on.

    ``ultradata_chat`` labels are ``"{domain}/{source}"``, so the bare source
    name is checked too; otherwise an exclusion would have to be spelled once
    per domain and a new domain would silently reopen the leak.
    """

    label = (origin or "").casefold()
    folded = {item.casefold() for item in exclusions}
    return label in folded or label.rsplit("/", 1)[-1] in folded


def known(origin: str | None, allowed: frozenset[str]) -> bool:
    """Whether a provenance label is one this adapter declares it can admit."""

    label = (origin or "").casefold()
    folded = {item.casefold() for item in allowed}
    return label in folded or label.rsplit("/", 1)[-1] in folded


def build_document(problem: str, solution: str, answer: str) -> str:
    return (
        f"{problem}{ANSWER_FENCE_SUFFIX}{THINK_OPEN}\n{solution}\n"
        f"{THINK_CLOSE}\n{ANSWER_OPEN}{answer}{ANSWER_CLOSE}"
    )


def trained_length(
    problem: str, document: str, tokenizer: GPT2BPETokenizer
) -> int:
    """Positions ``sft_trace_train.load_documents`` needs for a document.

    The trainer tokenizes the prompt and the completion apart, prefixes the
    prompt with BOS and needs one more position for the stop target, so a
    document fits ``seq_len`` only when this is at most ``seq_len``;
    ``len(encode(document)) + 1`` undercounts by the BOS and by any BPE merge
    across the prompt/completion boundary.
    """

    prompt = problem + ANSWER_FENCE_SUFFIX
    return (
        len(frame_prompt(tokenizer, tokenizer.encode(prompt)))
        + len(tokenizer.encode(document[len(prompt):])) + 1
    )


# Screening state, built once in the parent and inherited by workers through
# fork's copy-on-write. Rebuilding the decontamination index per worker would
# cost both startup time and a private copy of its n-gram set in every
# process; forking shares one.
_TOKENIZER: GPT2BPETokenizer | None = None
_EXACT: set[str] | None = None
_NGRAMS: set[tuple[str, ...]] | None = None
_REF_EXACT: set[str] | None = None
_REF_GRAM_TO_IDS: dict[tuple[str, ...], list[int]] | None = None
_REF_SIZES: list[int] | None = None
_QUESTIONS: QuestionIndex | None = None


def build_containment_index(
    texts: list[str],
) -> tuple[dict[tuple[str, ...], list[int]], list[int]]:
    """Inverted 8-gram index plus each reference problem's own gram count."""

    gram_to_ids: dict[tuple[str, ...], list[int]] = {}
    sizes: list[int] = []
    for index, text in enumerate(texts):
        grams = word_ngrams(text)
        sizes.append(len(grams))
        for gram in grams:
            gram_to_ids.setdefault(gram, []).append(index)
    return gram_to_ids, sizes


def contains_reference_problem(
    problem: str,
    exact: set[str],
    gram_to_ids: dict[tuple[str, ...], list[int]],
    sizes: list[int],
    min_overlap: float,
) -> bool:
    """Whether the candidate restates a reference problem.

    Containment, measured against each reference problem's own gram count, so
    that embedding one inside a longer problem cannot dilute it away.
    """

    if " ".join(problem.split()).lower() in exact:
        return True
    hits: Counter = Counter()
    for gram in word_ngrams(problem):
        for index in gram_to_ids.get(gram, ()):
            hits[index] += 1
    return any(
        count / sizes[index] >= min_overlap
        for index, count in hits.items()
        if sizes[index]
    )


def word_string(text: str) -> str:
    """The token stream ``word_ngrams`` sees, as one space-joined string."""

    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


# A gram in more than this many distinct benchmark questions is exam phrasing
# ("which of the following is not a", "is most likely to be the"), not
# evidence that a candidate restates any one of them. MMLU-Pro reuses MMLU
# stems, so a genuine question's own grams already occur in two references.
QUESTION_TEMPLATE_DOCUMENTS = 3
# Share of a question's distinctive grams the candidate must contain, and the
# fewest distinctive grams that make the share meaningful. Calibrated
# 2026-09-23 against MMLU and MMLU-Pro (NOTES.md): no UltraData-RL-2609
# Knowledge query reaches 0.6 (one, sharing only phrasing, reaches 0.5), and
# every embedded reference question with enough distinctive grams scores 1.0.
QUESTION_MIN_OVERLAP = 0.6
QUESTION_MIN_GRAMS = 4
# Stated in manifests.
QUESTION_RULE = (
    "exact normalized match of the problem or its question stem; else at "
    f"least {QUESTION_MIN_OVERLAP:.0%} of a benchmark question's distinctive "
    f"8-grams (those in at most {QUESTION_TEMPLATE_DOCUMENTS} benchmark "
    f"questions), for questions with {QUESTION_MIN_GRAMS}+ of them"
)


@dataclass(frozen=True)
class QuestionIndex:
    """Benchmark question stems prepared for ``contains_benchmark_question``."""

    # Each question's word string, mapped to its id.
    exact: dict[str, int]
    # Distinctive 8-grams of questions with at least QUESTION_MIN_GRAMS.
    gram_to_ids: dict[tuple[str, ...], list[int]]
    sizes: list[int]


def build_question_index(
    texts: list[str], template_documents: int | None = QUESTION_TEMPLATE_DOCUMENTS
) -> QuestionIndex:
    """Index question stems; ``template_documents=None`` keeps every gram
    distinctive, which is how a question is indexed on its own."""

    words = list(dict.fromkeys(filter(None, map(word_string, texts))))
    grams = [word_ngrams(text) for text in words]
    frequency = Counter(gram for own in grams for gram in own)
    gram_to_ids: dict[tuple[str, ...], list[int]] = {}
    sizes: list[int] = []
    for index, own in enumerate(grams):
        distinctive = [
            gram for gram in own
            if template_documents is None or frequency[gram] <= template_documents
        ]
        sizes.append(len(distinctive))
        if len(distinctive) >= QUESTION_MIN_GRAMS:
            for gram in distinctive:
                gram_to_ids.setdefault(gram, []).append(index)
    return QuestionIndex(
        {text: index for index, text in enumerate(words)}, gram_to_ids, sizes
    )


def contains_benchmark_question(problem: str, index: QuestionIndex) -> bool:
    """Whether the candidate restates a benchmark question stem.

    A question with at least ``QUESTION_MIN_GRAMS`` distinctive grams is
    contained when the candidate holds ``QUESTION_MIN_OVERLAP`` of them, which
    survives embedding in a longer problem. A shorter one is matched only
    exactly, against the whole candidate or its question stem: its few grams
    are too close to exam phrasing to test inside other text. "Which of the
    following is not a true statement?" is an MMLU question with two
    distinctive grams, and a verbatim-substring test for it rejected
    UltraData problems that merely ask the same thing about fluorescence.
    Such questions are identified in their benchmark by their options, which
    a stem screen cannot use.
    """

    return bool(matching_questions(problem, index))


def matching_questions(problem: str, index: QuestionIndex) -> set[int]:
    """Ids of the indexed questions ``problem`` restates, by the rule above.

    ``contains_benchmark_question`` is whether this is non-empty; callers
    that group problems by the questions they share need the ids.
    """

    words = word_string(problem)
    stems = {words}
    parsed = split_options(problem)
    if parsed is not None:
        stems.add(word_string(parsed.question))
    found = {index.exact[stem] for stem in stems if stem in index.exact}
    hits: Counter = Counter()
    for gram in word_ngrams(words):
        for reference in index.gram_to_ids.get(gram, ()):
            hits[reference] += 1
    found.update(
        reference for reference, count in hits.items()
        if count / index.sizes[reference] >= QUESTION_MIN_OVERLAP
    )
    return found


def read_reference_texts(path: Path, column: str) -> list[str]:
    """The non-empty problem texts of one screening target."""

    # GPQA's CSV quotes multi-line questions and explanations; without
    # newlines_in_values pyarrow refuses the file.
    table = (
        pa_csv.read_csv(
            path, parse_options=pa_csv.ParseOptions(newlines_in_values=True)
        )
        if path.suffix == ".csv"
        else pq.read_table(path)
    )
    texts = []
    for value in table.column(column).to_pylist():
        # gsm8k_rl_prompts stores a chat prompt; the rest bare strings.
        text = value[0]["content"] if isinstance(value, list) else value
        if isinstance(text, str) and text.strip():
            texts.append(text)
    if not texts:
        # A wrong column or an emptied file would otherwise disable the
        # screen it feeds without a trace.
        raise ValueError(f"{path} column {column!r} holds no problem text")
    return texts


def build_row(
    adapter: Adapter, row: dict, seq_len: int, min_completion_tokens: int
) -> tuple[dict | None, str]:
    """Screen and tokenize one raw record, independently of every other.

    Everything expensive and everything order-independent lives here, so it
    can run in a worker process. The order-dependent decisions -- global
    deduplication and the document caps -- stay in the parent, which is why
    the output does not depend on the worker count.
    """

    record, reason = PARSERS[adapter.kind](adapter, row)
    if record is None:
        return None, reason
    if excluded(record.origin, adapter.excluded_provenance):
        return None, "excluded_evaluation_source"
    if adapter.known_sources and not known(record.origin, adapter.known_sources):
        # Fail closed on an unrecognised label rather than admitting it: a
        # renamed or newly added evaluation-derived source in a shard this
        # adapter has not seen must not train by default.
        return None, f"unknown_provenance:{record.origin}"
    if not record.answer.strip():
        # Every parser must produce a non-empty answer span; an empty one
        # builds <answer></answer>, which the anchored gate rejects. Checked
        # here so a future parser cannot reopen it.
        return None, "empty_answer_span"
    try:
        problem, _ = strip_math_prompt_framing(record.problem)
    except ValueError:
        return None, "uncanonicalizable_problem"
    if not problem:
        return None, "empty_problem"
    # The answer span is graded structurally, so a math answer that cannot be
    # one line of final answer is not a usable target.
    if "\n" in record.answer and not record.multiline_answer:
        return None, "multiline_answer"
    # A literal fence in teacher text would tokenize to a real special id and
    # break the single-pair invariant.
    if any(
        fence in text
        for text in (problem, record.solution, record.answer)
        for fence in FENCE_STRINGS
    ):
        return None, "fence_literal"
    if contaminated(problem, _EXACT, _NGRAMS):
        return None, "contaminated"
    if contains_reference_problem(
        problem, _REF_EXACT, _REF_GRAM_TO_IDS, _REF_SIZES,
        CONTAINMENT_MIN_OVERLAP,
    ):
        return None, "contains_reference_problem"
    if _QUESTIONS is not None and contains_benchmark_question(problem, _QUESTIONS):
        return None, "contains_benchmark_question"
    document = build_document(problem, record.solution, record.answer)
    if adapter.max_doc_tokens is not None:
        seq_len = min(seq_len, adapter.max_doc_tokens)
    if len(document) > seq_len * CHARS_PER_TOKEN_BOUND:
        return None, "over_char_bound"
    probe_chars = seq_len * PREFIX_PROBE_CHARS_PER_TOKEN
    if len(document) > probe_chars and (
        len(_TOKENIZER.encode(document[:probe_chars])) > seq_len + 1
    ):
        return None, "over_seq_len_by_prefix"
    doc_tokens = len(_TOKENIZER.encode(document))
    # BPE is not concatenation-stable, so the completion is tokenized
    # directly rather than by subtracting the prompt.
    if len(_TOKENIZER.encode(document[len(problem):])) < min_completion_tokens:
        return None, "completion_too_short"
    if trained_length(problem, document, _TOKENIZER) > seq_len:
        return None, "over_seq_len"
    return (
        {
            "source": f"{adapter.name}:{record.origin}",
            "problem": problem,
            "document": document,
            "final_answer": record.answer,
            "verified": record.verified,
            "gradeable": record.gradeable,
            "doc_tokens": doc_tokens,
            # Deduplication is the parent's decision; the key is computed
            # here only because it is derived from the canonical problem.
            "_identity": (
                choice_identity(problem)
                if adapter.single_choice
                else problem.strip().lower()[:160]
            ),
            **(
                {"_choice": choice_balance_key(problem, record)}
                if adapter.balance_choice_answers
                else {}
            ),
        },
        "",
    )


def choice_identity(problem: str) -> str:
    """A single-choice problem's identity independent of labels and order.

    The same question appears under A-H and under 1-8 labels, or with its
    options in another order; a 160-character prefix sees two problems.
    """

    parsed = split_options(problem)
    return "\x00".join(
        (" ".join(parsed.question.split()).casefold(),
         *sorted(" ".join(option.split()).casefold() for option in parsed.options))
    )


def choice_balance_key(problem: str, record: Record) -> dict:
    """What ``balance_choice_answers`` needs to know about one kept row."""

    labels = split_options(problem).labels
    return {"count": len(labels), "position": labels.index(record.answer)}


def _capped_counts(counts: list[int], tolerance: float) -> list[int]:
    """Per-position counts under the cap that keeps the most rows.

    A cap U keeps min(counts[p], U) rows at position p. With n rows kept
    over k positions it is admissible when no position holds more than
    max(ceil(n / k), floor((1 + tolerance) n / k)) rows: a cap on the modal
    position's share only, with no floor, so an option count too small to
    fill every position is kept rather than dropped. At tolerance 0 the
    modal position holds at most ceil(n / k).
    """

    count = len(counts)
    best = [0] * count
    for cap in range(1, max(counts, default=0) + 1):
        kept = [min(rows, cap) for rows in counts]
        total = sum(kept)
        bound = max(
            math.ceil(total / count), math.floor((1 + tolerance) * total / count)
        )
        if max(kept) <= bound and total > sum(best):
            best = kept
    return best


def balance_choice_answers(rows: list[dict], tolerance: float) -> tuple[list[dict], dict]:
    """Cap every answer position by dropping rows; no row is relabelled.

    Per option count k, ``_capped_counts`` caps every position at
    (1 + ``tolerance``) times the mean and the excess is dropped in
    content-hash order, so the result is a function of the rows' contents,
    not their order. Renaming the labels a trace refers to was removed: a
    relabelled trace walks its options out of order, which tells which rows
    were moved, and every proof that a rename kept a trace's meaning was
    another heuristic over free text (NOTES 2026-09-24). Positions the
    teacher rarely chose stay rare, and the report gives each option count's
    least share.
    """

    def digest(row: dict) -> str:
        return hashlib.sha256(row["problem"].encode("utf-8")).hexdigest()

    by_count: dict[int, list[dict]] = {}
    for row in rows:
        by_count.setdefault(row["_choice"]["count"], []).append(row)
    kept: set[int] = set()
    report: dict[str, dict] = {}
    for count, group in sorted(by_count.items()):
        at = [[] for _ in range(count)]
        for row in sorted(group, key=digest):
            at[row["_choice"]["position"]].append(row)
        counts = _capped_counts([len(rows_at) for rows_at in at], tolerance)
        kept |= {
            id(row) for position, rows_at in enumerate(at)
            for row in rows_at[:counts[position]]
        }
        after = sum(counts)
        report[str(count)] = {
            "rows_before": len(group),
            "rows_after": after,
            "positions_before": [len(rows_at) for rows_at in at],
            "positions_after": counts,
            "modal_share_over_chance": (
                max(counts) * count / after if after else None
            ),
            "least_share_over_chance": (
                min(counts) * count / after if after else None
            ),
        }
    return [row for row in rows if id(row) in kept], report


def choice_document_report(rows: list[dict]) -> dict:
    """Kept single-choice documents by provenance, option count and label."""

    counts = Counter(len(split_options(row["problem"]).labels) for row in rows)
    return {
        "documents": len(rows),
        "by_option_count": {str(k): v for k, v in sorted(counts.items())},
        "by_source": dict(sorted(Counter(row["source"] for row in rows).items())),
        "answer_labels": dict(
            sorted(Counter(row["final_answer"] for row in rows).items())
        ),
    }


def trace_pool_provenance(pool: Path, rl_counterpart: Path, schema: str) -> dict:
    """The teacher, sampling and yields a distilled trace pool was built with.

    The pool's manifest must describe these exact bytes, so the corpus
    manifest cannot attribute a pool to the wrong generation run. The
    questions it was generated from must be an SFT partition whose RL
    counterpart names it (``scripts/build_science_mc_rl_prompts.py``), and
    that counterpart must be ``rl_counterpart``, the pool RL trains on, with
    the bytes its manifest describes: a pool distilled from an older split,
    whose questions the current RL pool may hold, cannot train. Every trace
    is re-graded against those questions: one trace per question, same
    problem, same source, letter equal to the gold letter. The pool must be
    of ``schema``, the selection contract the adapter declares, and selected
    from a finished generation: every question with all its samples.
    """

    manifest_path = pool.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    digest = file_sha256(pool)
    if manifest.get("output_sha256") != digest:
        raise ValueError(f"{manifest_path} does not describe {pool} ({digest})")
    if manifest.get("schema") != schema:
        raise ValueError(
            f"{manifest_path} is {manifest.get('schema')!r}, not {schema!r}"
        )
    generation = manifest["generation"]
    questions_path = refuse_evaluation_only(Path(generation["questions"]))
    questions_sha256 = file_sha256(questions_path)
    if questions_sha256 != generation["questions_sha256"]:
        raise ValueError(f"{questions_path} changed since generation")
    questions_manifest = json.loads(
        questions_path.with_suffix(".manifest.json").read_text()
    )
    if (
        questions_manifest.get("partition") != "sft"
        or questions_manifest.get("output_sha256") != questions_sha256
    ):
        raise ValueError(f"{questions_path} is not a built SFT question partition")
    rl_path = Path(questions_manifest["split"]["rl_output"])
    if rl_path.resolve() != rl_counterpart.resolve():
        raise ValueError(
            f"{questions_path} was split from {rl_path}, not the RL pool "
            f"{rl_counterpart}"
        )
    rl_manifest = json.loads(rl_path.with_suffix(".manifest.json").read_text())
    if rl_manifest.get("output_sha256") != file_sha256(rl_path):
        raise ValueError(f"{rl_path} does not match its manifest's sha256")
    if (
        rl_manifest.get("partition") != "rl"
        or rl_manifest["split"].get("sft_output_sha256") != questions_sha256
        or rl_manifest["split"].get("schema") != questions_manifest["split"]["schema"]
    ):
        raise ValueError(f"{rl_path} does not name {questions_path} as its SFT side")
    gold = {
        row["extra_info"]["original_query_sha256"]: (
            row["prompt"][-1]["content"],
            row["data_source"].removeprefix("science_mc_"),
            row["reward_model"]["ground_truth"],
        )
        for row in pq.read_table(questions_path).to_pylist()
    }
    # A pool selected while its generation was still running judged only
    # part of the partition for consistency.
    complete = manifest["self_consistency"].get("questions_complete")
    if complete != len(gold):
        raise ValueError(
            f"{pool} was selected from {complete} complete questions of "
            f"{len(gold)}; select again once generation has finished"
        )
    traced: set[str] = set()
    for row in pq.read_table(pool).to_pylist():
        if row["key"] in traced:
            raise ValueError(f"{pool}: question {row['key']} has two traces")
        traced.add(row["key"])
        if gold.get(row["key"]) != (row["problem"], row["source"], row["letter"]):
            raise ValueError(
                f"{pool}: trace {row['key']} does not match its question's "
                "problem, source and gold letter"
            )
    return {
        "pool": str(pool),
        "pool_sha256": digest,
        "pool_manifest_sha256": file_sha256(manifest_path),
        "pool_schema": manifest["schema"],
        "questions": generation["questions"],
        "questions_sha256": generation["questions_sha256"],
        "questions_rl_counterpart": str(rl_path),
        "questions_rl_counterpart_sha256": rl_manifest["output_sha256"],
        "generations_sha256": manifest["generations_sha256"],
        "teacher": generation["teacher"],
        "draft_model": generation["draft_model"],
        "server": generation["server"],
        "sampling": generation["sampling"],
        "system_prompt": generation["system_prompt"],
        "max_doc_tokens": manifest["max_doc_tokens"],
        "selection": manifest["selection"],
        "verifier": manifest["verifier"],
        "self_consistency": manifest["self_consistency"],
        "per_source": manifest["per_source"],
        "per_module": manifest["per_module"],
        "rejections": manifest["rejections"],
    }


def _screen_chunk(task):
    adapter, chunk, seq_len, min_completion_tokens = task
    return [
        build_row(adapter, row, seq_len, min_completion_tokens) for row in chunk
    ]


def chunked(rows, size: int):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME[=MAX]",
        help="corpus to include, optionally capped at MAX documents; "
        "repeatable. Default: openmathinstruct2 alone",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shards", type=int, default=None,
        help="how many numbered shards to download and consume; applies only "
        "to adapters that fetch shards by index (default 32)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=1024,
        help="documents needing more than this many tokens (plus the stop "
        "target) are dropped, matching the trainer's own packing rule",
    )
    parser.add_argument("--min-completion-tokens", type=int, default=16)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 2) - 4)),
        help="processes screening and tokenizing rows. Screening is pure and "
        "order-preserving, and deduplication and the caps stay in the parent, "
        "so the corpus is identical for any value; 1 forces the serial path",
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=200,
        help="rows per unit of work handed to a worker",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("postraining/data/instruction_corpus_shards"),
    )
    args = parser.parse_args()

    shards_requested = args.shards is not None
    if args.shards is None:
        args.shards = 32
    requested = args.source or ["openmathinstruct2"]
    plan: list[tuple[Adapter, int]] = []
    for item in requested:
        name, _, raw = item.partition("=")
        if name not in ADAPTERS:
            parser.error(
                f"unknown --source {name!r}; known: {', '.join(sorted(ADAPTERS))}"
            )
        if "=" in item and (not raw.isdigit() or int(raw) < 1):
            parser.error(f"--source {item!r} needs a positive integer cap")
        plan.append((ADAPTERS[name], int(raw) if raw else 0))
    if len({adapter.name for adapter, _ in plan}) != len(plan):
        parser.error("--source repeats a corpus")
    for adapter, _ in plan:
        # --shards only governs adapters that download numbered shards; local
        # corpora enumerate their own and must not be range-checked against it.
        if adapter.shard_template and not 1 <= args.shards <= adapter.shards:
            parser.error(
                f"--shards must be in 1..{adapter.shards} for {adapter.name}"
            )

    # --shards governs only adapters that fetch numbered shards. Passing it
    # when no selected adapter consumes it would be silently ignored; passing
    # it alongside a locally-globbed adapter is fine, since the flag still
    # applies to the others.
    if shards_requested and not any(
        adapter.shard_template for adapter, _ in plan
    ):
        parser.error(
            "--shards applies only to adapters that fetch numbered shards; "
            f"none of {', '.join(adapter.name for adapter, _ in plan)} does"
        )
    if args.output.exists():
        parser.error(f"{args.output} exists; corpora are immutable")
    if args.output.with_suffix(".manifest.json").exists():
        parser.error(
            f"{args.output.with_suffix('.manifest.json')} exists; corpora "
            "and their manifests are immutable"
        )
    if args.output.suffix != ".parquet":
        parser.error("--output must end in .parquet")
    if "." in args.output.stem:
        # The trainer derives the manifest as with_suffix(".manifest.json"),
        # which truncates at the first dot: "corpus_v1.2.parquet" would look
        # for "corpus_v1.manifest.json". Refuse rather than write a manifest
        # the trainer will not find.
        parser.error(
            f"--output stem {args.output.stem!r} contains a dot; the corpus "
            "manifest path would not match what the trainer looks for"
        )

    teacher_traces = {}
    for adapter, _ in plan:
        if adapter.kind == "choice_trace":
            if adapter.rl_counterpart is None:
                parser.error(f"{adapter.name}: a trace pool needs its rl_counterpart")
            try:
                teacher_traces[adapter.name] = trace_pool_provenance(
                    adapter.local_parquet, adapter.rl_counterpart, adapter.revision
                )
            except (FileNotFoundError, KeyError, ValueError) as error:
                parser.error(f"{adapter.name}: {error}")

    # The trainer tokenizes with the fences registered, so the corpus must
    # measure lengths the same way: fence-blind BPE splits <think> into
    # ordinary pieces and overstates every document by ~8 tokens.
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    global _TOKENIZER, _EXACT, _NGRAMS
    global _REF_EXACT, _REF_GRAM_TO_IDS, _REF_SIZES, _QUESTIONS
    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    _TOKENIZER = tokenizer
    exact, ngrams = build_decontamination_index()
    reference_texts: list[str] = []
    reference_counts: dict[str, int] = {}
    # Deduplicated in declaration order, so the screen is one shared index
    # regardless of how many planned adapters declare the same targets.
    targets = list(dict.fromkeys(
        [*CONTAINMENT_TARGETS,
         *(target for adapter, _ in plan
           for target in adapter.containment_targets)]
    ))
    for target, column in targets:
        if not target.exists():
            parser.error(
                f"{target} is missing; rows would enter with provenance "
                "labels as their only protection against it"
            )
        texts = read_reference_texts(target, column)
        reference_texts.extend(texts)
        reference_counts[str(target)] = len(texts)
    question_texts: list[str] = []
    question_counts: dict[str, int] = {}
    for target, column in dict.fromkeys(
        target for adapter, _ in plan for target in adapter.question_targets
    ):
        if not target.exists():
            parser.error(
                f"{target} is missing; rows would enter training unscreened "
                "against its benchmark questions"
            )
        texts = read_reference_texts(target, column)
        question_texts.extend(texts)
        question_counts[str(target)] = len(texts)
    reference_exact = {
        " ".join(text.split()).lower() for text in reference_texts
    }
    reference_gram_to_ids, reference_sizes = build_containment_index(
        reference_texts
    )
    # Set before the pool forks so workers inherit it copy-on-write.
    _EXACT, _NGRAMS = exact, ngrams
    _REF_EXACT = reference_exact
    _REF_GRAM_TO_IDS = reference_gram_to_ids
    _REF_SIZES = reference_sizes
    _QUESTIONS = build_question_index(question_texts) if question_texts else None
    counts = Counter()
    provenance = Counter()
    seen: set[str] = set()
    rows: list[dict] = []
    shard_hashes = []

    # "fork" explicitly: Python 3.14 defaults to forkserver on Linux, which
    # re-imports this module in each worker and so would leave the screening
    # state above unset. Copy-on-write inheritance is the point here.
    for adapter, cap in plan:
      # One pool per adapter. A per-source cap breaks out mid-stream, and an
      # abandoned imap keeps its feeder thread reading and tokenizing the rest
      # of a multi-gigabyte shard into an unbounded result buffer; terminating
      # the pool with the adapter is what actually stops that work.
      pool = (
          None
          if args.workers == 1
          else get_context("fork").Pool(args.workers)
      )
      try:
        kept_here = 0
        shards = 1 if adapter.local_parquet is not None else args.shards
        for path in shard_paths(adapter, shards, args.cache):
            shard_hashes.append(
                {"corpus": adapter.name, "shard": path.name,
                 "sha256": file_sha256(path)}
            )
            tasks = (
                (adapter, chunk, args.seq_len, args.min_completion_tokens)
                for chunk in chunked(iter_rows(path), args.chunk_rows)
            )
            # imap preserves input order, so the parent sees rows in exactly
            # the order the serial path would, and deduplication and the caps
            # resolve identically for any --workers.
            results = (
                map(_screen_chunk, tasks)
                if pool is None
                else pool.imap(_screen_chunk, tasks)
            )
            stop = False
            for screened in results:
                for payload, reason in screened:
                    counts["seen"] += 1
                    if payload is None:
                        if reason.startswith("unknown_provenance:"):
                            counts["unknown_provenance"] += 1
                            label = reason.split(":", 1)[1]
                            provenance[f"{adapter.name}:UNKNOWN:{label}"] += 1
                        else:
                            counts[reason] += 1
                        continue
                    identity = payload.pop("_identity")
                    if identity in seen:
                        counts["duplicate_problem"] += 1
                        continue
                    seen.add(identity)
                    provenance[payload["source"]] += 1
                    counts["kept"] += 1
                    kept_here += 1
                    rows.append(payload)
                    if _done(args.max_documents, len(rows), cap, kept_here):
                        stop = True
                        break
                if stop:
                    break
            # Progress on stderr: a full build reads tens of gigabytes
            # across ~120 shards and the manifest only prints at the very end,
            # so without this a multi-hour run looks indistinguishable from a
            # hung one.
            print(
                f"[{adapter.name}] {path.name}: seen {counts['seen']} "
                f"kept {counts['kept']} ({kept_here} this source)",
                file=sys.stderr,
                flush=True,
            )
            if stop:
                break
      finally:
        if pool is not None:
            # Terminate rather than close: queued chunks are abandoned instead
            # of screened pointlessly.
            pool.terminate()
            pool.join()
      if args.max_documents and len(rows) >= args.max_documents:
          break

    choice_balance = {}
    for adapter, _ in plan:
        if not adapter.balance_choice_answers:
            continue
        mine = [row for row in rows if row["source"].startswith(f"{adapter.name}:")]
        balanced, by_count = balance_choice_answers(
            mine, adapter.choice_balance_tolerance
        )
        choice_balance[adapter.name] = {
            "tolerance": adapter.choice_balance_tolerance,
            "by_option_count": by_count,
        }
        # A source's rows are contiguous in build order, so the balanced ones
        # (in their original relative order) go back where the source was.
        start = next(
            index for index, row in enumerate(rows)
            if row["source"].startswith(f"{adapter.name}:")
        )
        if rows[start:start + len(mine)] != mine:
            raise AssertionError(f"{adapter.name} rows are not contiguous")
        rows = rows[:start] + balanced + rows[start + len(mine):]
        counts["choice_balance_dropped"] += len(mine) - len(balanced)
        counts["kept"] -= len(mine) - len(balanced)
    for row in rows:
        row.pop("_choice", None)
    # Kept counts are the written rows', after balancing; rejected rows of
    # unknown provenance stay listed under their UNKNOWN keys.
    provenance = Counter(
        {key: value for key, value in provenance.items() if ":UNKNOWN:" in key}
    ) + Counter(row["source"] for row in rows)

    if not rows:
        parser.error("no documents survived the corpus filters")

    # Match the curated corpora's column types exactly. Writing ``verified``
    # as a string would make the trainer's ``if row["verified"]`` filter admit
    # rows by truthiness of the text "False" rather than by their value.
    schema = pa.schema(
        [
            ("source", pa.string()),
            ("problem", pa.string()),
            ("document", pa.string()),
            ("final_answer", pa.string()),
            ("verified", pa.bool_()),
            ("gradeable", pa.bool_()),
            ("doc_tokens", pa.int64()),
        ]
    )
    staging = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), staging)
    staging.replace(args.output)

    manifest = {
        "schema": CORPUS_SCHEMA,
        "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
        "documents": len(rows),
        "corpora": [
            {
                "name": adapter.name,
                "dataset": adapter.repo,
                "revision": adapter.revision,
                "license": adapter.license_note,
                "max_documents": cap or None,
                "max_doc_tokens": adapter.max_doc_tokens,
                "excluded_evaluation_sources": sorted(
                    adapter.excluded_provenance
                ),
            }
            for adapter, cap in plan
        ],
        "shards_consumed": {
            name: sum(1 for entry in shard_hashes if entry["corpus"] == name)
            for name in (adapter.name for adapter, _ in plan)
        },
        "decontamination_targets": [
            *(str(target) for target in DECONTAMINATION_TARGETS),
        ],
        "containment_targets": reference_counts,
        "question_targets": question_counts,
        "question_rule": QUESTION_RULE,
        "containment_rule": (
            "exact normalized match, or a reference problem at least "
            f"{CONTAINMENT_MIN_OVERLAP:.0%} contained in the candidate, "
            "measured against that reference problem's own 8-gram count"
        ),
        # Stated because it cannot be enforced. OpenMathInstruct-2's
        # augmented_gsm8k partition is GSM8K train rewritten, not copied, so
        # no textual rule identifies it: containment catches 89 of 525
        # gsm8k-derived rows. UltraData records carry no problem_source
        # field, so its OpenMathInstruct-2 half may contain semantically
        # GSM8K-train-derived problems. The GSM8K *test* split -- the
        # evaluation set -- is separately and fully protected by the exact
        # plus any-8-gram index above, and is disjoint from train.
        "gsm8k_train_derivation_caveat": (
            "ultradata_sft_2605 rows have no problem_source; rewritten "
            "GSM8K-train variants are not textually identifiable and may be "
            "present. GSM8K test is unaffected."
        ),
        # Which shards actually existed locally. A pinned revision can be
        # mirrored partially, and the corpus must say so rather than leaving
        # it inferable only from the shard hash list's length.
        "local_shard_selection": [
            {
                "name": adapter.name,
                "local_shards": str(adapter.local_shards)
                if adapter.local_shards
                else None,
                "shard_glob": adapter.shard_glob or None,
                "domains": list(adapter.domains) or None,
                "shards_found": sum(
                    1 for entry in shard_hashes
                    if entry["corpus"] == adapter.name
                ),
                "published_shards": adapter.shards or None,
                "known_sources": sorted(adapter.known_sources) or None,
            }
            for adapter, _ in plan
        ],
        "gradeable_documents": sum(1 for row in rows if row["gradeable"]),
        "decontamination": "exact normalized match plus word 8-grams",
        "shard_sha256": shard_hashes,
        "seq_len": args.seq_len,
        "min_completion_tokens": args.min_completion_tokens,
        "chars_per_token_bound": CHARS_PER_TOKEN_BOUND,
        "workers": args.workers,
        "counts": dict(counts),
        "kept_by_provenance": dict(provenance),
        "choice_answer_balance": choice_balance or None,
        "choice_documents": {
            adapter.name: choice_document_report(
                [row for row in rows
                 if row["source"].startswith(f"{adapter.name}:")]
            )
            for adapter, _ in plan
            if adapter.single_choice
        } or None,
        "teacher_traces": teacher_traces or None,
        "deduplicated_by": {
            adapter.name: (
                "casefolded question plus sorted option texts (choice_identity)"
                if adapter.single_choice
                else "lowercased 160-char problem prefix"
            )
            for adapter, _ in plan
        },
        # Solutions are model-generated and, except for the sandbox-verified
        # code pool, not re-verified here; the column says which rows were
        # rather than asserting a verification that never ran.
        "verified": all(row["verified"] for row in rows),
        "verified_documents": sum(1 for row in rows if row["verified"]),
        "output_sha256": file_sha256(args.output),
    }
    # Must be with_suffix: sft_trace_train derives the manifest path as
    # traces.with_suffix(".manifest.json") and refuses the corpus if it is
    # not there. A dotted stem is rejected above rather than diverging here.
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in manifest.items()
                      if k != "shard_sha256"}, indent=2))


if __name__ == "__main__":
    main()

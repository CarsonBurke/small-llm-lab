"""Build the ``science_mc`` RL prompt pool from SciQ, ARC and OpenBookQA.

Grade-school and high-school science single choice, far easier than the
UltraData Knowledge pool's graduate-level ten-option rows, so a small policy
has a chance of a learnable signal. Every source contributes its TRAIN split
only, pinned by revision and byte sha256:

* ``arc_challenge`` / ``arc_easy`` -- ``allenai/ai2_arc`` (CC BY-SA 4.0);
  3-5 options, labels ``A``-``E`` or ``1``-``5``.
* ``openbookqa`` -- ``allenai/openbookqa`` ``main`` (Apache-2.0 upstream; the
  hub card says "unknown"); four options. The stem is often a sentence
  fragment that an option completes; rendering keeps it verbatim.
* ``sciq`` -- ``allenai/sciq`` (CC BY-NC 3.0: non-commercial); the correct
  answer plus three distractors. The ``support`` passage is not shown.

Rendering, grading and the content-seeded option shuffle are the shared
single-choice contract (``postraining/choice_rl_pool.py``): ``A. option``
lines, one upper-case letter graded ``exact`` under the ``rule`` style, and
the correct option at a uniformly drawn position. ``extra_info.module`` is
``{source}_choice_{k}``, so each source and option count logs its own
modal-answer baseline.

Screens, in order, after canonicalization:

1. the math evaluation index (``prepare_sft_traces``);
2. the benchmark-question screen over ``STEM_QUESTION_TARGETS`` -- MMLU,
   MMLU-Pro, GPQA and these sources' own validation and test splits. Those
   files are screen targets only; no row is ever read from them. A stem
   shared with an evaluation question drops the row even when its options
   differ, which over-screens generic stems on purpose;
3. owners: question stems of the UltraData Knowledge RL pool and SFT corpus,
   and of every earlier source in ``SOURCES`` order, under the same rule, so
   one question is served by one pool;
4. the 256-token RL prompt budget.

Within a source, rows are deduplicated by normalized stem plus option set:
generic stems ("Which of these is a mixture?") recur with different options
as different questions.

The kept rows are then split into two disjoint partitions: an RL pool
(``--output``) and a teacher-trace question set (``--sft-output``) from which
``postraining/generate_choice_traces.py`` distils the worked solutions of the
``science_mc_traces`` SFT source. Rows that share a question under the owner
rule, that have the same option texts and the same correct option, or that
have the same correct option and close stems (reworded questions), form one
component and go to one side together, and each
component's side is a content hash of its rows' keys
(``choice_rl_pool.SPLIT_RULE``).
``--sft-fraction 0`` builds the unsplit pool (v2).

    python3 scripts/build_science_mc_rl_prompts.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postraining.choice_prompt import split_options
from postraining.choice_rl_pool import (
    CHOICE_STYLE,
    SPLIT_RULE,
    SPLIT_SCHEMA,
    Screens,
    choice_problem,
    choice_report,
    identity,
    load_reference_texts,
    normalized,
    partition_questions,
    refuse_existing_outputs,
    screen_problem,
    sha256_file,
    write_pool,
)
from postraining.math_prompt import ANSWER_FENCE_PROMPT_SCHEMA
from postraining.prepare_sft_corpus import (
    EVALUATION_ONLY_DATA,
    QUESTION_RULE,
    STEM_QUESTION_TARGETS,
    refuse_evaluation_only,
)
from postraining.prepare_sft_traces import DECONTAMINATION_TARGETS

POOL_SCHEMA = "science_mc_rl_pool/v1"
DEFAULT_SPLIT_SALT = "science_mc_sft_split/v1"
TRAIN_ROOT = Path("postraining/data/science_mc_train")

# One raw training example: stable id, question, options, correct index and
# the labels the source printed (none for SciQ).
Example = tuple[str, str, tuple[str, ...], int, tuple[str, ...]]


def arc_examples(table) -> Iterator[Example | str]:
    for row in table.to_pylist():
        labels = list(row["choices"]["label"])
        if row["answerKey"] not in labels:
            yield "answer_key_not_a_label"
            continue
        yield (
            row["id"], row["question"], tuple(row["choices"]["text"]),
            labels.index(row["answerKey"]), tuple(labels),
        )


def openbookqa_examples(table) -> Iterator[Example | str]:
    for row in table.to_pylist():
        labels = list(row["choices"]["label"])
        if row["answerKey"] not in labels:
            yield "answer_key_not_a_label"
            continue
        yield (
            row["id"], row["question_stem"], tuple(row["choices"]["text"]),
            labels.index(row["answerKey"]), tuple(labels),
        )


def sciq_examples(table) -> Iterator[Example | str]:
    # SciQ has no ids; the row index in the pinned file is stable.
    for index, row in enumerate(table.to_pylist()):
        yield (
            f"sciq-train-{index}", row["question"],
            (row["correct_answer"], row["distractor1"], row["distractor2"],
             row["distractor3"]),
            0, (),
        )


@dataclass(frozen=True)
class Source:
    name: str
    dataset: str
    revision: str
    path: Path
    sha256: str
    rows: int
    license: str
    examples: Callable[..., Iterator[Example | str]]


# Harder sources first: a question shared across sources is kept by the
# earliest one.
SOURCES = (
    Source(
        "arc_challenge", "allenai/ai2_arc",
        "210d026faf9955653af8916fad021475a3f00453",
        TRAIN_ROOT / "allenai__ai2_arc/ARC-Challenge/train-00000-of-00001.parquet",
        "e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb",
        1119, "CC BY-SA 4.0", arc_examples,
    ),
    Source(
        "arc_easy", "allenai/ai2_arc",
        "210d026faf9955653af8916fad021475a3f00453",
        TRAIN_ROOT / "allenai__ai2_arc/ARC-Easy/train-00000-of-00001.parquet",
        "b315db8a4be597dc7daa50a4e70d48dd7c990c32085629e6ccd8c926beaa80b5",
        2251, "CC BY-SA 4.0", arc_examples,
    ),
    Source(
        "openbookqa", "allenai/openbookqa",
        "388097ea7776314e93a529163e0fea805b8a6454",
        TRAIN_ROOT / "allenai__openbookqa/main/train-00000-of-00001.parquet",
        "98148f8a54e62eb862346a75192d5fb824d6cbb68f2f59aecd793d39ecb5cd8b",
        4957, "Apache-2.0 (upstream); hub card: unknown",
        openbookqa_examples,
    ),
    Source(
        "sciq", "allenai/sciq",
        "2c94ad3e1aafab77146f384e23536f97a4849815",
        TRAIN_ROOT / "allenai__sciq/data/train-00000-of-00001.parquet",
        "19644360954006d06e9ad3df07bddb34f8535c081b831d48f604603c713ac167",
        11679, "CC BY-NC 3.0 (non-commercial)", sciq_examples,
    ),
)
DEFAULT_OWNERS = (
    Path("postraining/data/ultradata-knowledge-rl-v2.parquet"),
    Path("postraining/data/sft_ultradata_knowledge_v5.parquet"),
)


def question_stem(problem: str) -> str:
    parsed = split_options(problem)
    return parsed.question if parsed is not None else problem


def owner_stems(path: Path) -> list[str]:
    """Question stems of an RL pool (``prompt``) or SFT corpus (``problem``)."""

    table = pq.read_table(path)
    if "prompt" in table.column_names:
        problems = [
            turns[-1]["content"] for turns in table.column("prompt").to_pylist()
        ]
    elif "problem" in table.column_names:
        problems = table.column("problem").to_pylist()
    else:
        raise SystemExit(f"{path} has neither a prompt nor a problem column")
    return [question_stem(problem) for problem in problems]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("postraining/data/science-mc-rl-v5.parquet"),
    )
    parser.add_argument(
        "--sft-output", type=Path,
        default=Path("postraining/data/science-mc-sft-questions-v3.parquet"),
        help="the teacher-trace partition, in the pool's own row schema",
    )
    parser.add_argument(
        "--sft-fraction", type=float, default=0.5,
        help="expected share of question components in the SFT partition; "
        "0 builds the unsplit pool and writes no --sft-output",
    )
    parser.add_argument("--split-salt", default=DEFAULT_SPLIT_SALT)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument(
        "--min-options", type=int, default=3,
        help="ARC has a few three-option rows; fewer would be a coin flip",
    )
    parser.add_argument(
        "--owner", type=Path, action="append", default=None,
        help="RL pool or SFT corpus whose questions this pool must not "
        "serve again (repeatable); defaults to the UltraData Knowledge RL "
        "pool and SFT corpus",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report counts only; tolerate missing screen targets and owners "
        "and write nothing",
    )
    args = parser.parse_args()
    if not 0.0 <= args.sft_fraction < 1.0:
        parser.error("--sft-fraction must be in [0, 1)")
    outputs = [args.output] + ([args.sft_output] if args.sft_fraction else [])
    if len(set(outputs)) != len(outputs):
        parser.error("--output and --sft-output must differ")
    for output in outputs if not args.dry_run else ():
        if reason := refuse_existing_outputs(output):
            parser.error(reason)

    for source in SOURCES:
        try:
            refuse_evaluation_only(source.path)
        except ValueError as error:
            parser.error(str(error))
        if not source.path.exists():
            parser.error(f"{source.path} is missing")
    inputs = []
    for source in SOURCES:
        digest = sha256_file(source.path)
        if digest != source.sha256:
            parser.error(f"{source.path} sha256 {digest} != pinned {source.sha256}")
        inputs.append({
            "source": source.name, "dataset": source.dataset,
            "revision": source.revision, "split": "train",
            "path": str(source.path), "sha256": digest,
            "license": source.license,
        })

    owners = args.owner or list(DEFAULT_OWNERS)
    owner_texts: list[str] = []
    owner_counts: dict[str, dict] = {}
    missing_owners: list[str] = []
    for path in owners:
        if not path.exists():
            if not args.dry_run:
                parser.error(f"owner {path} is missing; build it first")
            missing_owners.append(str(path))
            continue
        stems = owner_stems(path)
        owner_texts.extend(stems)
        owner_counts[str(path)] = {"questions": len(stems), "sha256": sha256_file(path)}
    reference_texts, reference_counts, unscreened = load_reference_texts(
        STEM_QUESTION_TARGETS, allow_missing=args.dry_run
    )

    counts: dict[str, Counter] = {}
    rows: list[dict] = []
    for source in SOURCES:
        # Earlier sources own their questions: rebuilt per source so a
        # restatement across sources is screened like one across pools.
        screens = Screens.build(
            reference_texts, owner_texts, args.max_prompt_tokens
        )
        table = pq.read_table(source.path)
        if table.num_rows != source.rows:
            parser.error(
                f"{source.path} has {table.num_rows} rows, pinned {source.rows}"
            )
        tally = counts.setdefault(source.name, Counter())
        seen: set[str] = set()
        kept_stems: list[str] = []
        for example in source.examples(table):
            tally["seen"] += 1
            if isinstance(example, str):
                tally[example] += 1
                continue
            example_id, question, options, target, labels = example
            question = question.strip()
            options = tuple(option.strip() for option in options)
            key = identity(
                question + "\x00" + "\x00".join(sorted(map(normalized, options)))
            )
            if key in seen:
                tally["duplicate_question"] += 1
                continue
            built = choice_problem(
                question, options, target,
                hashlib.sha256(f"{source.name}\x00{key}".encode()).digest(),
                args.min_options, labels,
            )
            if isinstance(built, str):
                tally[built] += 1
                continue
            problem, answer, order = built
            screened = screen_problem(problem, screens)
            if isinstance(screened, str):
                tally[screened] += 1
                continue
            problem, prompt_tokens = screened
            seen.add(key)
            kept_stems.append(question)
            tally["kept"] += 1
            count = len(order)
            rows.append(
                {
                    "data_source": f"science_mc_{source.name}",
                    "prompt": [{"role": "user", "content": problem}],
                    "ability": "science",
                    "reward_model": {"ground_truth": answer, "style": CHOICE_STYLE},
                    "extra_info": {
                        "index": example_id,
                        "module": f"{source.name}_choice_{count}",
                        "prompt_contract": "bare",
                        "original_query_sha256": key,
                        "source_path": str(source.path),
                        "source_revision": source.revision,
                        "prompt_token_count": prompt_tokens,
                        "option_order": order,
                        "source_position": target,
                        "source_target": options[target],
                        "chance": 1.0 / count,
                    },
                }
            )
        owner_texts.extend(kept_stems)
    if not rows:
        parser.error("no science rows survived")
    rows.sort(
        key=lambda row: (
            row["data_source"], row["extra_info"]["original_query_sha256"]
        )
    )
    split = None
    partitions = {"rl": rows}
    if args.sft_fraction:
        sft, components = partition_questions(
            [row["extra_info"]["original_query_sha256"] for row in rows],
            [row["prompt"][-1]["content"] for row in rows],
            [question_stem(row["prompt"][-1]["content"]) for row in rows],
            [row["reward_model"]["ground_truth"] for row in rows],
            args.sft_fraction, args.split_salt,
        )
        partitions = {
            "sft": [row for row, side in zip(rows, sft) if side],
            "rl": [row for row, side in zip(rows, sft) if not side],
        }
        split = {
            "schema": SPLIT_SCHEMA,
            "rule": SPLIT_RULE,
            "sft_fraction": args.sft_fraction,
            "salt": args.split_salt,
            "rows_before_split": len(rows),
            **components,
            "rows": {
                side: {
                    source.name: sum(
                        row["data_source"] == f"science_mc_{source.name}"
                        for row in kept
                    )
                    for source in SOURCES
                }
                for side, kept in partitions.items()
            },
            "rl_output": str(args.output),
            "sft_output": str(args.sft_output),
        }

    def report_for(kept: list[dict], partition: str) -> dict:
        return {
            "schema": POOL_SCHEMA,
            "partition": partition,
            "problems": len(kept),
            "counts": {
                name: dict(sorted(tally.items())) for name, tally in counts.items()
            },
            **choice_report(kept),
            "per_source": {
                source.name: choice_report(
                    [row for row in kept
                     if row["data_source"] == f"science_mc_{source.name}"]
                )
                for source in SOURCES
            },
            "split": split,
            "unscreened_targets": unscreened,
            "missing_owners": missing_owners,
        }

    if args.dry_run:
        print(json.dumps(
            {side: report_for(kept, side) for side, kept in partitions.items()},
            indent=2,
        ))
        return

    manifest = {
        "inputs": inputs,
        "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
        "verifier": "reward_model.style 'rule' -> core.verify_answer 'exact': "
        "the stripped <answer> span equals one upper-case letter",
        "option_rendering": "question, blank line, 'A. option' lines",
        "option_order": "content-seeded permutation placing the target at a "
        "uniform position; extra_info.option_order maps rendered to source "
        "and extra_info.source_position is the target's source index",
        "decontamination_targets": [str(path) for path in DECONTAMINATION_TARGETS],
        "question_targets": reference_counts,
        "question_rule": QUESTION_RULE,
        "owners": owner_counts,
        "owner_rule": "the question rule over owner question stems and every "
        "earlier source's kept stems, in source order "
        + ", ".join(source.name for source in SOURCES),
        "evaluation_only_data": str(EVALUATION_ONLY_DATA),
        "evaluation_only_policy": "train splits only. GPQA, MMLU, MMLU-Pro and "
        "the SciQ/ARC/OpenBookQA validation and test splits under "
        "evaluation_only_data are decontamination screen targets only; no row "
        "of this pool is read from them (postraining/tests/test_benchmark_isolation.py)",
        "deduplicated_by": "sha256 of the normalized question plus sorted "
        "normalized option texts, within each source",
        "order": "data_source, then ascending original_query_sha256; no RNG",
        "max_prompt_tokens": args.max_prompt_tokens,
        "min_options": args.min_options,
        "license_note": "SciQ rows are CC BY-NC 3.0 (non-commercial); ARC "
        "rows CC BY-SA 4.0; OpenBookQA Apache-2.0 upstream",
    }
    digests = {}
    if "sft" in partitions:
        # Written first so the RL manifest can bind the partition it excludes.
        digests["sft"] = write_pool(
            partitions["sft"],
            {**report_for(partitions["sft"], "sft"), **manifest,
             "use": "teacher-trace questions for the science_mc_traces SFT "
             "source; never an RL source"},
            args.sft_output,
        )
        split["sft_output_sha256"] = digests["sft"]
    digests["rl"] = write_pool(
        partitions["rl"], {**report_for(partitions["rl"], "rl"), **manifest},
        args.output,
    )
    print(json.dumps(
        {side: {"problems": len(kept), "output_sha256": digests[side]}
         for side, kept in partitions.items()},
        indent=2,
    ))


if __name__ == "__main__":
    main()

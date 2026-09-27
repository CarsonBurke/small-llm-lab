"""Build the UltraData-RL-2609 Knowledge (STEM) RL prompt pool.

UltraData's Knowledge slice is OpenScienceReasoning-2 rewritten into short
answers. Most targets are free-text phrases that only a semantic judge could
grade, so this pool admits exactly two row shapes with deterministic
verifiers:

* **single choice** -- the query ends in a labelled option block
  (``choice_prompt.split_options``) and the target is one option's label or,
  more often, its exact text. Options are re-rendered as ``A. text`` lines and
  the target becomes one upper-case letter graded by the existing ``exact``
  style: the stripped ``<answer>`` span must be that letter and nothing else.
* **numeric** -- the target parses as one exact number
  (``parse_numeric_answer``), graded by the Minerva style the math pools use.

Rendering, the content-seeded option shuffle that removes the source's
letter prior, the ``choice_{k}`` chance modules and the screens are shared
with every single-choice pool (``postraining/choice_rl_pool.py``).

Evaluation protection is textual, because every record names only
``UltraData-RL-2609`` as its source: the GSM8K/DeepMind/AIME index the SFT
corpora use, plus the benchmark-question screen against MMLU, MMLU-Pro and
GPQA stems and the SciQ/ARC/OpenBookQA validation and test splits
(``prepare_sft_corpus.contains_benchmark_question`` over
``STEM_QUESTION_TARGETS``). Those files are screen targets only: no row is
ever read from them. A missing target refuses
the build; ``--dry-run`` reports counts without writing anything and names
the targets it could not screen.

The inputs are the pinned, full-file ``ultradata_windows`` captures of both
Knowledge shards; their receipts are verified before any row is read.

    python3 scripts/build_ultradata_knowledge_rl_prompts.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postraining.choice_prompt import split_options
from postraining.choice_rl_pool import (
    CHOICE_STYLE,
    Screens,
    balanced_order,  # noqa: F401  (re-exported for tests)
    choice_problem,
    choice_report,
    identity,
    load_reference_texts,
    normalized,
    refuse_existing_outputs,
    screen_problem,
    sha256_file,
    write_pool,
)
from postraining.core import parse_numeric_answer
from postraining.math_prompt import ANSWER_FENCE_PROMPT_SCHEMA
from postraining.prepare_sft_corpus import (
    EVALUATION_ONLY_DATA,
    QUESTION_RULE,
    STEM_QUESTION_TARGETS,
)
from postraining.prepare_sft_traces import DECONTAMINATION_TARGETS
from postraining.ultradata_data import DATASET, PUBLISHED_COUNTS, REVISION

POOL_SCHEMA = "ultradata_knowledge_rl_pool/v1"
WINDOWS = Path("postraining/data/ultradata_windows") / REVISION
SHARDS = (
    "data/Knowledge/Knowledge_part-1-of-2.jsonl",
    "data/Knowledge/Knowledge_part-2-of-2.jsonl",
)
NUMERIC_STYLE = "rule-lighteval/MATH_v2"


def captured_shards() -> list[tuple[str, Path, dict]]:
    """Locate and verify each shard's full-file capture by its receipt."""

    found: dict[str, tuple[Path, dict]] = {}
    for receipt_path in WINDOWS.glob("*.json"):
        receipt = json.loads(receipt_path.read_text())
        for shard in SHARDS:
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{shard}"
            total = (receipt.get("content_range") or "").rpartition("/")[2]
            if (
                receipt.get("url") == url
                and receipt.get("start") == 0
                and total.isdigit()
                and receipt.get("bytes") == int(total)
            ):
                found[shard] = (receipt_path.with_suffix(".bin"), receipt)
    missing = [shard for shard in SHARDS if shard not in found]
    if missing:
        raise SystemExit(f"no full-file capture of {missing} under {WINDOWS}")
    for shard, (path, receipt) in found.items():
        if sha256_file(path) != receipt["sha256"]:
            raise SystemExit(f"{path} bytes differ from its receipt")
    return [(shard, *found[shard]) for shard in SHARDS]


def choice_target(parsed, ground_truth: str) -> int | None:
    """Index of the one option the target names, by label or by text.

    A target that reads as a label of one option and as the text of another
    ("3" with digit labels and numeric options) names two options; which one
    the source meant is unknowable, so it resolves to neither.
    """

    named = set()
    label = re.fullmatch(r"\(?([A-Za-z]|[1-9]\d?)\)?\.?", ground_truth.strip())
    if label and label[1] in parsed.labels:
        named.add(parsed.labels.index(label[1]))
    texts = [normalized(option).rstrip(".") for option in parsed.options]
    hits = [
        index for index, text in enumerate(texts)
        if text == normalized(ground_truth).rstrip(".")
    ]
    if len(hits) > 1:
        return None
    named.update(hits)
    return named.pop() if len(named) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("postraining/data/ultradata-knowledge-rl-v2.parquet"),
    )
    parser.add_argument(
        "--max-prompt-tokens", type=int, default=256,
        help="drop prompts longer than the RL prompt budget; the production "
        "launch uses --prompt-tokens 256, and a truncated single-choice "
        "prompt loses its question or its options",
    )
    parser.add_argument(
        "--min-options", type=int, default=4,
        help="fewer options raise chance toward a coin flip",
    )
    parser.add_argument(
        "--no-numeric", action="store_true",
        help="single-choice rows only",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report counts only; tolerate missing decontamination targets "
        "and write nothing",
    )
    args = parser.parse_args()
    if not args.dry_run and (reason := refuse_existing_outputs(args.output)):
        parser.error(reason)

    shards = captured_shards()
    reference_texts, reference_counts, unscreened = load_reference_texts(
        STEM_QUESTION_TARGETS, allow_missing=args.dry_run
    )
    screens = Screens.build(reference_texts, [], args.max_prompt_tokens)

    counts: Counter = Counter()
    rows: list[dict] = []
    seen: set[str] = set()
    seen_stems: set[str] = set()
    for shard, path, _ in shards:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                counts["seen"] += 1
                query = record["query"]
                truth = record["ground_truth"]
                if not isinstance(truth, str) or not truth.strip():
                    counts["invalid_target"] += 1
                    continue
                query_id = identity(query)
                if query_id in seen:
                    counts["duplicate_query"] += 1
                    continue
                parsed = split_options(query)
                stem = None
                if parsed is not None:
                    target = choice_target(parsed, truth)
                    if target is None:
                        counts["choice_target_unresolved"] += 1
                        continue
                    stem = normalized(parsed.question)
                    if stem in seen_stems:
                        # The same question under another option set.
                        counts["duplicate_question"] += 1
                        continue
                    built = choice_problem(
                        parsed.question, tuple(parsed.options), target,
                        bytes.fromhex(query_id), args.min_options,
                        parsed.labels,
                    )
                    if isinstance(built, str):
                        counts[built] += 1
                        continue
                    problem, answer, order = built
                    count = len(order)
                    style, module = CHOICE_STYLE, f"choice_{count}"
                    extra = {
                        "option_order": order,
                        "source_position": target,
                        "source_target": truth,
                        "chance": 1.0 / count,
                    }
                elif parse_numeric_answer(truth) is not None and not args.no_numeric:
                    problem, answer = query.strip(), truth.strip()
                    style, module = NUMERIC_STYLE, "numeric"
                    extra = {"option_order": None, "source_position": None,
                             "source_target": truth, "chance": 0.0}
                else:
                    counts["free_text_target"] += 1
                    continue
                screened = screen_problem(problem, screens)
                if isinstance(screened, str):
                    counts[screened] += 1
                    continue
                problem, prompt_tokens = screened
                seen.add(query_id)
                if stem is not None:
                    seen_stems.add(stem)
                counts[f"kept_{module.split('_')[0]}"] += 1
                rows.append(
                    {
                        "data_source": "ultradata_rl_knowledge",
                        "prompt": [{"role": "user", "content": problem}],
                        "ability": "knowledge",
                        "reward_model": {"ground_truth": answer, "style": style},
                        "extra_info": {
                            "index": record["uuid"],
                            "module": module,
                            "prompt_contract": "bare",
                            "original_query_sha256": query_id,
                            "source_path": shard,
                            "source_revision": REVISION,
                            "prompt_token_count": prompt_tokens,
                            **extra,
                        },
                    }
                )
    if counts["seen"] != PUBLISHED_COUNTS["Knowledge"]:
        parser.error(
            f"read {counts['seen']} Knowledge rows, published "
            f"{PUBLISHED_COUNTS['Knowledge']}"
        )
    if not rows:
        parser.error("no Knowledge rows survived")
    rows.sort(key=lambda row: row["extra_info"]["original_query_sha256"])
    report = {
        "schema": POOL_SCHEMA,
        "problems": len(rows),
        "counts": dict(sorted(counts.items())),
        **choice_report(rows),
        "unscreened_targets": unscreened,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return

    manifest = {
        **report,
        "dataset": DATASET,
        "source_revision": REVISION,
        "domain": "Knowledge",
        "inputs": [
            {"shard": shard, "path": str(path), "sha256": receipt["sha256"],
             "receipt_etag": receipt.get("etag")}
            for shard, path, receipt in shards
        ],
        "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
        "verifiers": {
            "choice": "reward_model.style 'rule' -> core.verify_answer "
            "'exact': the stripped <answer> span equals one upper-case letter",
            "numeric": "reward_model.style 'rule-lighteval/MATH_v2' -> Minerva",
        },
        "option_rendering": "question, blank line, 'A. option' lines",
        "option_order": "content-seeded permutation placing the target at a "
        "uniform position; extra_info.option_order maps rendered to source "
        "and extra_info.source_position is the target's source index",
        "decontamination_targets": [str(path) for path in DECONTAMINATION_TARGETS],
        "question_targets": reference_counts,
        "question_rule": QUESTION_RULE,
        "evaluation_only_data": str(EVALUATION_ONLY_DATA),
        "evaluation_only_policy": "GPQA, MMLU, MMLU-Pro and the SciQ/ARC/"
        "OpenBookQA validation and test splits under evaluation_only_data "
        "are decontamination screen targets only; no row of this pool is "
        "read from them (postraining/tests/test_benchmark_isolation.py)",
        "deduplicated_by": "sha256 of the whitespace/case-normalized query, "
        "then the normalized question stem of single-choice rows",
        "order": "ascending original_query_sha256; no RNG",
        "max_prompt_tokens": args.max_prompt_tokens,
        "min_options": args.min_options,
    }
    digest = write_pool(rows, manifest, args.output)
    print(json.dumps({**report, "output_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()

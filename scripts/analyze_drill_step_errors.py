"""Where worked drill traces first go wrong: a step-level error anatomy.

A drill's worked solution is a pure function of its problem (operands and
the phrasing keyed on them, ``math_drills.phrase``), so every bench attempt
can be checked line by line against the exact canonical working. This
locates the first divergent line of each attempt and asks:

- Where do failures start (line index, fraction of the way through the
  working), per module?
- Does a divergence predict failure? ``P(correct | diverged)`` near zero
  means a divergence is an error, not a harmless rewording, so the first
  divergent line is the culprit step.
- What is wrong on that line: the stated result, an operand (misread digit
  or carried state), or the line's shape?

Only computation lines are compared (``step_lines``); restatements are
ungraded and drift harmlessly.

Canonical solutions are recovered without the pool builder's exclusion
sets: a cell's problems are a subsequence of ``math_drills.generate`` on
that cell's seed (exclusion only skips draws; the RNG consumes the same
values), so each held-out problem is matched by text in the raw stream.

    python3 scripts/analyze_drill_step_errors.py \\
        --bench-data postraining/data/drill_rl_study_heldout_20261014_v1.parquet \\
        --answers <run>/bench_answers/step_000360.json --output <out>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postraining.math_drills import BUILDERS, FAMILY_ORDERS

# The pool builder's rule (scripts/build_drill_rl_prompts.py CELL_SEED_RULE).
RAW_DRAWS_PER_CELL = 4096
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def cell_seed(seed: int, family: str, digits: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{family}:{digits}".encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "little")


def canonical_solutions(bench_data: Path) -> dict[str, list[list[str]]]:
    """Problem text -> the canonical steps of each valid method.

    Replays ``math_drills.generate``'s draws for a one-family curriculum and
    rebuilds every matched problem under each of its family's digit orders
    from the RNG state its builder started in: builders draw the operands
    first, so every order works the same problem.
    """
    rows = pq.read_table(bench_data).to_pylist()
    wanted: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for row in rows:
        info = row["extra_info"]
        seed = int(info["index"].split("/")[3])
        wanted[(info["family"], int(info["digits"]), seed)].add(
            row["prompt"][0]["content"]
        )
    solutions: dict[str, list[list[str]]] = {}
    for (family, digits, seed), problems in wanted.items():
        rng = random.Random(cell_seed(seed, family, digits))
        orders, missing = FAMILY_ORDERS[family], set(problems)
        for _ in range(RAW_DRAWS_PER_CELL):
            # generate()'s draw order: family, digit order, digit count.
            rng.choices([family], [1.0])
            order = rng.choice(orders)
            rng.randint(digits, digits)
            state = rng.getstate()
            drill = BUILDERS[family](rng, order, digits)
            if drill.problem not in missing:
                continue
            missing.discard(drill.problem)
            methods = []
            for other in orders:
                replay = random.Random()
                replay.setstate(state)
                rebuilt = BUILDERS[family](replay, other, digits)
                assert rebuilt.problem == drill.problem, "builders must draw operands first"
                methods.append(step_lines(rebuilt.solution))
            solutions[drill.problem] = methods
            if not missing:
                break
        if missing:
            raise ValueError(
                f"{len(missing)} {family}:{digits} problems are not in the "
                f"first {RAW_DRAWS_PER_CELL} raw draws of seed {seed}"
            )
    return solutions


def step_lines(text: str) -> list[str]:
    """The computation lines: the procedure's checkable steps.

    A step states a result (``=``, ``->``) or reads digits off a column
    chain (``Digits from ...``). Prose leads and restatements ("Ignore the
    points and multiply 438 by 99.") are skipped: the reward never grades
    them, and the trained policy rewords and corrupts them in attempts that
    are still correct, so a difference there is not an error.
    """
    return [
        line.strip() for line in text.strip().split("\n")
        if NUMBER.search(line)
        and ("=" in line or "->" in line or line.strip().startswith("Digits from"))
    ]


def classify(model: str, canonical: str) -> str:
    """What differs on the first divergent line."""
    got, want = NUMBER.findall(model), NUMBER.findall(canonical)
    if NUMBER.sub("#", model) != NUMBER.sub("#", canonical) or len(got) != len(want):
        return "shape"
    differing = [index for index, (g, w) in enumerate(zip(got, want)) if g != w]
    # A line's last number is its stated result (or carry); earlier numbers
    # are operands read from the problem or carried from the previous line.
    return "result" if differing[0] >= len(want) - 2 else "operand"


def analyze(bench_data: Path, answers: Path) -> dict:
    solutions = canonical_solutions(bench_data)
    attempts = json.loads(answers.read_text())["attempts"]
    per_module: dict[str, Counter] = defaultdict(Counter)
    first_error_fraction: dict[str, list[float]] = defaultdict(list)
    examples: dict[str, list[dict]] = defaultdict(list)
    for attempt in attempts:
        module = "_".join(attempt["dataset_index"].split("/")[1:3])
        steps = step_lines(attempt["emitted_text"])

        def first_divergence(canonical: list[str]) -> int | None:
            got = steps[: len(canonical)]
            return next(
                (index for index, (g, w) in enumerate(zip(got, canonical)) if g != w),
                None if len(got) == len(canonical) else len(got),
            )

        # Score against the method the attempt follows furthest.
        canonical, diverged = max(
            ((method, first_divergence(method)) for method in solutions[attempt["prompt"]]),
            key=lambda pair: len(pair[0]) if pair[1] is None else pair[1],
        )
        lines = steps[: len(canonical)]
        counts = per_module[module]
        counts["attempts"] += 1
        counts["correct"] += bool(attempt["correct"])
        if diverged is None:
            counts["canonical"] += 1
            counts["canonical_correct"] += bool(attempt["correct"])
            if not attempt["correct"]:
                # Every step right and the answer wrong: the final assembly.
                counts["kind/assembly"] += 1
                if len(examples[module]) < 4:
                    examples[module].append({
                        "prompt": attempt["prompt"], "line": None,
                        "lines": len(canonical), "model": attempt["parsed_answer"],
                        "canonical": attempt["ground_truth"],
                    })
            continue
        counts["diverged"] += 1
        counts["diverged_correct"] += bool(attempt["correct"])
        kind = (
            "truncated" if diverged >= len(lines)
            else classify(lines[diverged], canonical[diverged])
        )
        counts[f"kind/{kind}"] += 1
        first_error_fraction[module].append(diverged / len(canonical))
        if not attempt["correct"] and len(examples[module]) < 4:
            examples[module].append({
                "prompt": attempt["prompt"],
                "line": diverged,
                "lines": len(canonical),
                "model": lines[diverged] if diverged < len(lines) else None,
                "canonical": canonical[diverged],
            })
    report = {}
    for module, counts in sorted(per_module.items()):
        fractions = sorted(first_error_fraction[module])
        report[module] = {
            **counts,
            "accuracy": counts["correct"] / counts["attempts"],
            "diverged_fraction": counts["diverged"] / counts["attempts"],
            "p_correct_given_diverged": (
                counts["diverged_correct"] / counts["diverged"]
                if counts["diverged"] else None
            ),
            "p_correct_given_canonical": (
                counts["canonical_correct"] / counts["canonical"]
                if counts["canonical"] else None
            ),
            "first_error_fraction_quartiles": (
                [fractions[int(q * (len(fractions) - 1))] for q in (0.25, 0.5, 0.75)]
                if fractions else None
            ),
            "failure_examples": examples[module],
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bench-data", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.bench_data, args.answers)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'module':18s} {'acc':>6s} {'div':>6s} {'P(c|div)':>8s} {'P(c|can)':>8s}  kinds")
    for module, row in report.items():
        kinds = {k.split("/")[1]: v for k, v in row.items() if k.startswith("kind/")}
        fmt = lambda x: "    -" if x is None else f"{x:.3f}"
        print(f"{module:18s} {row['accuracy']:6.3f} {row['diverged_fraction']:6.3f} "
              f"{fmt(row['p_correct_given_diverged']):>8s} "
              f"{fmt(row['p_correct_given_canonical']):>8s}  {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

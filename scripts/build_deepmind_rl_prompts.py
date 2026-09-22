"""Build a DeepMind mathematics_dataset RL prompt pool from one split.

The binding rollout gate showed total reward starvation on DAPO-Math-17K
(128/128 trajectories at reward 0.0 — the step-1500 mathmix model cannot
solve any of it, so there is no within-group variance and no policy
gradient).  This pool is the documented curriculum lever: prompts at
exactly the difficulty where the model already scores ~15% avg@32
(deepmind-interpolate-easy), which is precisely the mixed-success regime
VAPO's group-relative signal needs.

Rows use the DAPO parquet schema with the verbatim DAPO template, so
``train_latent_vapo --math-data`` and the verifier consume them unchanged.
The 144 problems of the held-out eval set (``deepmind-interpolate-easy
.parquet``) are excluded by question text, keeping the RL trainer's bench
eval genuinely held out.

``--source-dir`` selects the difficulty tier.  ``interpolate`` is the test
split and shares its distribution with the bench panel, so a pool built from
it trains on exactly what the bench measures.  ``train-easy`` is a disjoint
training split at a markedly easier surface difficulty -- the same 18 modules
phrased with smaller magnitudes and fewer decimal places -- which is what a
policy scoring 0.00 on interpolate needs before group-relative RL has any
within-group variance to learn from.  The split name is recorded in the
manifest; it is not inferred by any consumer.

    python3 scripts/build_deepmind_rl_prompts.py \
        --source-dir postraining/data/mathematics_dataset-v1.0/train-easy \
        --split train-easy \
        --output postraining/data/deepmind-train-easy-rl.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_deepmind_eval_set import module_pairs
from scripts.build_math_mix_dataset import (
    DAPO_PREAMBLE,
    DAPO_REMINDER,
    DEEPMIND_EASY_MODULES,
)

# Read deep enough into each module file that excluding the eval picks and
# sampling the pool never exhausts a module.
READ_LIMIT_PAIRS = 20000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default="postraining/data/mathematics_dataset-v1.0/interpolate",
        help="mathematics_dataset split directory to draw prompts from",
    )
    parser.add_argument(
        "--split",
        default="interpolate",
        help="split name recorded in the manifest; must match --source-dir",
    )
    parser.add_argument(
        "--eval-file", default="postraining/data/deepmind-interpolate-easy.parquet"
    )
    parser.add_argument(
        "--output", default="postraining/data/deepmind-interpolate-rl.parquet"
    )
    parser.add_argument(
        "--per-module", type=int, default=1000,
        help="take the first N non-eval rows per module; 0 takes every row",
    )
    args = parser.parse_args()
    if args.per_module < 0:
        parser.error("--per-module must be nonnegative")
    if Path(args.source_dir).name != args.split:
        parser.error(
            f"--split {args.split!r} does not name --source-dir "
            f"{Path(args.source_dir).name!r}; the manifest must record the "
            "split the rows actually came from"
        )

    eval_questions = set()
    for row in pq.read_table(args.eval_file).to_pylist():
        content = row["prompt"][0]["content"]
        question = content.removeprefix(DAPO_PREAMBLE).removesuffix(DAPO_REMINDER)
        eval_questions.add(question.strip())

    source_dir = Path(args.source_dir)
    excluded = 0
    rows_by_module = []
    for module in DEEPMIND_EASY_MODULES:
        read = module_pairs(source_dir / f"{module}.txt", READ_LIMIT_PAIRS)
        pairs = [
            (question, answer)
            for question, answer in read
            if question not in eval_questions
        ]
        # The real number of rows this split lost to the bench panel. For a
        # tier disjoint from interpolate this is 0, and saying so is the point:
        # the manifest must not imply an exclusion it never performed.
        excluded += len(read) - len(pairs)
        if args.per_module and len(pairs) < args.per_module:
            parser.error(f"{module}: only {len(pairs)} non-eval pairs available")
        selected = pairs if args.per_module == 0 else pairs[: args.per_module]
        module_rows = []
        for source_index, (question, answer) in enumerate(selected):
            module_rows.append(
                {
                    "prompt": [
                        {
                            "content": (
                                f"{DAPO_PREAMBLE}\n\n{question}\n\n{DAPO_REMINDER}"
                            ),
                            "role": "user",
                        }
                    ],
                    "reward_model": {"ground_truth": answer, "style": "rule"},
                    "extra_info": {
                        "index": f"{module}/{source_index}",
                        "module": module,
                    },
                }
            )
        rows_by_module.append(module_rows)

    # Deterministic round-robin keeps every contiguous training window mixed
    # across modules without sampling, shuffling, or omitting any source row.
    rows = [
        module_rows[index]
        for index in range(max(map(len, rows_by_module)))
        for module_rows in rows_by_module
        if index < len(module_rows)
    ]

    output = Path(args.output)
    pq.write_table(pa.Table.from_pylist(rows), output)
    manifest = {
        "problems": len(rows),
        "per_module": args.per_module or "all",
        "modules": DEEPMIND_EASY_MODULES,
        "split": args.split,
        "source": (
            f"mathematics_dataset-v1.0 {args.split}, eval-set problems excluded"
        ),
        "eval_panel_problems": len(eval_questions),
        "excluded_eval_problems": excluded,
        "template": "verbatim DAPO-Math-17K prompt wrapper",
        "order": "deterministic module round-robin in source-file order; no RNG",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

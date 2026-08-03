"""Build the DeepMind-interpolate easy-math benchmark parquet.

An easier benchmark than AIME, at exactly the difficulty of the mathmix
training QA slice: held-out ``interpolate`` test problems from the same 18
verifier-compatible mathematics_dataset modules whose ``train-easy`` splits
went into the corpus.  Rows use the DAPO parquet schema (``prompt`` message
list, ``reward_model.ground_truth``, ``extra_info.index``) with each problem
wrapped in the verbatim DAPO template, so every existing eval path
(``eval_aime.py --test-file``, ``evaluate_aime_latent``) consumes it
unchanged.  AIME stays the eval of record; this set is where a 27M model can
show an actual accuracy curve instead of AIME's 0-or-lucky floor.

    python3 scripts/build_deepmind_eval_set.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_math_mix_dataset import (
    DAPO_PREAMBLE,
    DAPO_REMINDER,
    DEEPMIND_EASY_MODULES,
)

# Enough pairs to sample from without reading the full multi-MB files.
READ_LIMIT_PAIRS = 2000


def module_pairs(path: Path, limit: int) -> list[tuple[str, str]]:
    pairs = []
    with path.open(encoding="utf-8") as handle:
        while len(pairs) < limit:
            question = handle.readline()
            answer = handle.readline()
            if not answer:
                break
            pairs.append((question.strip(), answer.strip()))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interpolate-dir",
        default="postraining/data/mathematics_dataset-v1.0/interpolate",
    )
    parser.add_argument(
        "--output", default="postraining/data/deepmind-interpolate-easy.parquet"
    )
    parser.add_argument("--per-module", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    interpolate_dir = Path(args.interpolate_dir)
    missing = [
        module
        for module in DEEPMIND_EASY_MODULES
        if not (interpolate_dir / f"{module}.txt").exists()
    ]
    if missing:
        parser.error(f"missing interpolate module files: {missing}")

    rng = random.Random(args.seed)
    rows = []
    for module in DEEPMIND_EASY_MODULES:
        pairs = module_pairs(interpolate_dir / f"{module}.txt", READ_LIMIT_PAIRS)
        if len(pairs) < args.per_module:
            parser.error(f"{module}: only {len(pairs)} pairs available")
        for question, answer in rng.sample(pairs, args.per_module):
            rows.append(
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
                    "extra_info": {"index": f"{module}/{len(rows)}", "module": module},
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)
    manifest = {
        "problems": len(rows),
        "per_module": args.per_module,
        "modules": DEEPMIND_EASY_MODULES,
        "source": "mathematics_dataset-v1.0 interpolate (held out from training)",
        "template": "verbatim DAPO-Math-17K prompt wrapper",
        "seed": args.seed,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

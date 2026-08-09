"""Build a GSM8K-difficulty RL prompt set in the DAPO row schema.

The SFT'd policy has its trainable band (mixed correct/incorrect groups)
on GSM8K-level problems, while DAPO's competition-level tail leaves exact
reward mostly unreachable — there the only dense signal is partial
credit, which is optimizable by fast answer-guessing and drives the
brevity/entropy collapse. Matching the RL prompt distribution to the
band where exact success actually varies is what gives a BINARY reward
gradient signal.

Rows remain in the neutral plain-Answer source schema; answer-fenced consumers
canonicalize every math family at runtime. Only
the GSM8K train split is used. The SFT holdout problems (the sampling
gate's panel) are excluded so the gate stays uncontaminated across
SFT -> RL comparisons.

Output: ``postraining/data/gsm8k_rl_prompts.parquet``. CPU-only.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters  # noqa: F401 (env parity)
from postraining.prepare_sft_traces import INSTRUCTION_SUFFIX, normalize_problem
from postraining.sft_trace_train import load_documents, split_holdout

OUTPUT = Path("postraining/data/gsm8k_rl_prompts.parquet")
SFT_TRACES = Path(
    "postraining/data/sft_traces_v6_answer_bare_a1swap10k.parquet"
)
SFT_HOLDOUT_PROBLEMS = 256  # must match the SFT run's --holdout-problems


def main() -> None:
    _, _, panel = split_holdout(load_documents(SFT_TRACES), SFT_HOLDOUT_PROBLEMS)
    held_keys = {normalize_problem(document["problem"]) for document in panel}

    records = []
    excluded = 0
    for position, row in enumerate(
        load_dataset("openai/gsm8k", "main", split="train")
    ):
        if normalize_problem(row["question"]) in held_keys:
            excluded += 1
            continue
        answer = row["answer"].rsplit("####", 1)[-1].strip()
        records.append(
            {
                "data_source": "gsm8k_train",
                "prompt": [
                    {"content": row["question"].strip() + INSTRUCTION_SUFFIX}
                ],
                "ability": "MATH",
                "reward_model": {"ground_truth": answer},
                "extra_info": {"index": f"gsm8k_train_{position}"},
            }
        )
    pq.write_table(pa.Table.from_pylist(records), OUTPUT)
    print(
        f"wrote {OUTPUT}: {len(records)} prompts "
        f"({excluded} SFT-holdout problems excluded)"
    )


if __name__ == "__main__":
    main()

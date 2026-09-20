"""Canonical verifiable-task preparation, content splits and immutable publication."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

from postraining.data_acquisition import atomic_json, canonical_json, sha256
from postraining.verifiable_tasks import (
    VERIFIABLE_TASK_SCHEMA,
    validate_verifiable_task,
    verifiable_reward_identity,
)

FINAL_ANSWER_INSTRUCTION = "After any reasoning, give the final answer as Answer: \\boxed{...}, replacing ... with only the answer."
CODE_ANSWER_INSTRUCTION = "End with the complete executable Python program in one ```python fenced code block. Read standard input and write standard output."
TOKENIZER = "openbmb/MiniCPM5-1B"
TOKENIZER_REVISION = "87179e5c1f455ef22e6223592d2d61351b525bfc"
DEFAULT_PROMPT_CAPS = {
    "Math": 2048,
    "Knowledge": 2048,
    "Code": 4096,
    "Long_Context": 6144,
}


def make_task(
    source: dict, tokenizer, *, prompt_tokens: int, context_tokens: int = 10000
) -> dict | None:
    """Normalize an explicitly typed source record without truncating its question."""
    kind = source["verification_kind"]
    if kind not in ("math", "text", "python_stdio"):
        raise ValueError(f"unsupported verification kind: {kind}")
    if not 0 < prompt_tokens < context_tokens:
        raise ValueError("prompt cap must leave room within the total context")
    question = source["query"]
    instruction = (
        CODE_ANSWER_INSTRUCTION if kind == "python_stdio" else FINAL_ANSWER_INSTRUCTION
    )
    messages = [{"role": "user", "content": question + "\n\n" + instruction}]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True
    )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if len(ids) > prompt_tokens:
        return None
    verification = {"schema": VERIFIABLE_TASK_SCHEMA, "kind": kind}
    code = kind == "python_stdio"
    if code:
        target = source["ground_truth"]
        if not isinstance(target, dict) or set(target) != {
            "call_type",
            "fn_name",
            "inputs",
            "outputs",
        }:
            raise ValueError(
                "stdio targets must contain only their executable test contract"
            )
        verification.update(target)
    row = {
        "prompt": messages,
        "data_source": source["source"],
        "reward_model": {
            "style": VERIFIABLE_TASK_SCHEMA,
            "ground_truth": "execution_tests" if code else source["ground_truth"],
        },
        "extra_info": {
            "index": source["uuid"],
            "domain": source["domain"],
            "source_revision": source["source_revision"],
            "original_query_sha256": sha256(question.encode("utf-8")),
            "source": source["source"],
            "prompt_token_count": len(ids),
            "prompt_token_cap": prompt_tokens,
            "context_token_cap": context_tokens,
            "provenance_json": canonical_json(source.get("provenance", {})).decode(),
        },
        "verification_info": verification,
    }
    validate_verifiable_task(row)
    return row


def unique_tasks(rows: list[dict], counters: dict[str, Counter]) -> list[dict]:
    """Quarantine every conflicting contract for an exact source question."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["extra_info"]["original_query_sha256"]].append(row)
    retained = []
    for group in groups.values():
        contracts = {
            canonical_json(
                {
                    "reward": r["reward_model"],
                    "verification": r["verification_info"],
                    "domain": r["extra_info"]["domain"],
                }
            )
            for r in group
        }
        if len(contracts) != 1:
            for row in group:
                counters[row["extra_info"]["domain"]][
                    "conflicting_prompt_quarantine"
                ] += 1
            continue
        retained.append(group[0])
        for row in group[1:]:
            counters[row["extra_info"]["domain"]]["duplicate_prompt"] += 1
    return retained


def split_tasks(
    rows: list[dict], *, seed: int, validation_fraction: float
) -> tuple[list[dict], list[dict]]:
    """Retain every unique task, splitting each reporting domain deterministically."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    domains = defaultdict(list)
    for row in rows:
        domains[row["extra_info"]["domain"]].append(row)
    if not domains or any(len(pool) < 2 for pool in domains.values()):
        raise ValueError("every domain needs at least two usable unique tasks")
    train, validation = [], []
    for domain, pool in sorted(domains.items()):
        pool.sort(key=lambda row: row["extra_info"]["original_query_sha256"])
        random.Random(f"{seed}:{domain}:selection").shuffle(pool)
        count = min(len(pool) - 1, max(1, round(len(pool) * validation_fraction)))
        validation.extend(pool[:count])
        train.extend(pool[count:])
    random.Random(f"{seed}:train").shuffle(train)
    random.Random(f"{seed}:validation").shuffle(validation)
    return train, validation


def publish_tasks(
    output: Path, train: list[dict], validation: list[dict], manifest: dict
) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")

    def hashes(items):
        return [r["extra_info"]["original_query_sha256"] for r in items]

    train_hashes, validation_hashes = hashes(train), hashes(validation)
    if len(set(train_hashes + validation_hashes)) != len(train) + len(validation):
        raise ValueError("published tasks must be globally content unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent)
    )
    try:
        table = pa.Table.from_pylist(train + validation)
        pq.write_table(table.slice(0, len(train)), staging / "train.parquet")
        pq.write_table(table.slice(len(train)), staging / "validation.parquet")
        manifest = {
            **manifest,
            "splits": {
                name: {
                    "path": name + ".parquet",
                    "rows": len(items),
                    "sha256": sha256((staging / (name + ".parquet")).read_bytes()),
                    "ordered_content_sha256": sha256(canonical_json(hashes(items))),
                    "content_hashes": hashes(items),
                }
                for name, items in (("train", train), ("validation", validation))
            },
        }
        atomic_json(staging / "manifest.json", manifest)
        rename = ctypes.CDLL(None, use_errno=True).renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(output))
        for path in output.iterdir():
            path.chmod(0o444)
        output.chmod(0o555)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "postraining/data/ultradata_windows/e6ecfa733708a4c54b5a98c3ca0fd16fc6923790"
        ),
    )
    parser.add_argument("--acquisition-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--code-shards", type=int, default=39)
    parser.add_argument("--context-tokens", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.acquisition_bytes <= 4 * 1024**3:
        parser.error("acquisition bytes must be positive and at most4GiB")
    if args.context_tokens != 10000:
        parser.error("this campaign uses10000 total context tokens")
    from postraining.data_acquisition import BoundedHTTP
    from postraining.hf_runtime import prepare_text_only_transformers_runtime
    from postraining.ultradata_data import acquire_ultradata, reviewed_questions
    from postraining.codecontests_data import acquire_codecontests

    prepare_text_only_transformers_runtime()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER,
        revision=TOKENIZER_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    http = BoundedHTTP(args.cache, args.acquisition_bytes)
    try:
        ultra_rows, ultra_manifest = acquire_ultradata(http, seed=args.seed)
        code_rows, code_manifest = acquire_codecontests(
            http, seed=args.seed, max_shards=args.code_shards
        )
        counters = {domain: Counter() for domain in DEFAULT_PROMPT_CAPS}
        tasks = []
        for source in ultra_rows + code_rows:
            domain = source["domain"]
            counters[domain]["source_eligible"] += 1
            task = make_task(
                source,
                tokenizer,
                prompt_tokens=DEFAULT_PROMPT_CAPS[domain],
                context_tokens=args.context_tokens,
            )
            if task is None:
                counters[domain]["prompt_over_budget"] += 1
            else:
                tasks.append(task)
        tasks = unique_tasks(tasks, counters)
        counts = Counter(r["extra_info"]["domain"] for r in tasks)
        if set(counts) != set(DEFAULT_PROMPT_CAPS):
            raise ValueError(f"missing required domains: {counts}")
        train, validation = split_tasks(
            tasks, seed=args.seed, validation_fraction=args.validation_fraction
        )
        domain_counts = {
            domain: {
                "usable_unique": counts[domain],
                "train_rows": sum(r["extra_info"]["domain"] == domain for r in train),
                "validation_rows": sum(
                    r["extra_info"]["domain"] == domain for r in validation
                ),
                "filters": dict(counters[domain]),
            }
            for domain in DEFAULT_PROMPT_CAPS
        }
        for value in domain_counts.values():
            value["train_weight"] = value["train_rows"] / len(train)
        _, review_identity = reviewed_questions()
        manifest = {
            "schema": "verifiable_corpus/v1",
            "sources": [ultra_manifest, code_manifest],
            "domains": domain_counts,
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "allocation": "all usable unique examples; no downsampling or oversampling",
            "context_tokens": args.context_tokens,
            "prompt_token_caps": DEFAULT_PROMPT_CAPS,
            "tokenizer": {
                "model": TOKENIZER,
                "revision": TOKENIZER_REVISION,
                "chat_template_sha256": sha256(canonical_json(tokenizer.chat_template)),
                "enable_thinking": True,
            },
            "reward_identity": verifiable_reward_identity(),
            "math_review_identity": review_identity,
            "preparation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "network": {"cap_bytes": args.acquisition_bytes, **http.ledger},
            "limitations": [
                "Bounded source acquisition is not a uniform sample of either full release.",
                "Base-model pretraining exposure is possible; development split is local only.",
                "Format filtering does not certify semantic target correctness or hidden-test completeness.",
            ],
        }
        result = publish_tasks(args.output, train, validation, manifest)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "domains": domain_counts,
                    "download_bytes": result["network"]["charged_bytes"],
                    "splits": {k: v["rows"] for k, v in result["splits"].items()},
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        http.close()

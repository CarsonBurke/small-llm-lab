"""Build immutable SFT5 by replacing SFT4's GSM8K traces with A1 math."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import postraining.prepare_sft_traces as sft
from postraining.core import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    GPT2BPETokenizer,
)
from postraining.math_prompt import (
    LEGACY_ANSWER_FENCE_PROMPT_SCHEMA,
    LEGACY_ANSWER_FENCE_SUFFIX,
)


SELECTION_SCHEMA = "sft5_a1_swap_stable_identity_rank/v1"
MANIFEST_SCHEMA = "verified_math_sft_a1_swap/v1"
DEFAULT_BASE = Path(
    "postraining/data/sft_traces_v4_answer_canonical_hfonly.parquet"
)
DEFAULT_BASE_SHA256 = (
    "ac398fc38e4db4d5d53cd03594850b96d3be79425fc80f963f326e23f99e8007"
)
DEFAULT_A1_SOURCE = Path("postraining/data/relaxed_bar/a1_math_deepmind.parquet")
DEFAULT_OUTPUT = Path(
    "postraining/data/sft_traces_v5_answer_canonical_a1swap10k.parquet"
)
DEFAULT_SWAP_ROWS = 10_000
DEFAULT_HOLDOUT_PROBLEMS = 256
DEFAULT_SEED = 0
PANEL_ROWS = 128

EXPECTED_DOCUMENTS = 36_286
EXPECTED_SOURCE_COUNTS = {
    "a1_deepmind": 16_000,
    "openmath_gsm8k": 0,
    "had653_gold": 6_950,
    "sxiong_l13": 5_974,
    "gsm8k_socratic": 7_362,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def identity(document: dict) -> str:
    """Use the exact identity function consumed by current split_holdout."""
    return sft.normalize_problem(str(document["problem"]))


def identity_rank(problem_identity: str) -> bytes:
    """Use the exact unversioned rank consumed by current split_holdout."""
    return hashlib.sha256(problem_identity.encode("utf-8")).digest()


def selection_rank(value: str, seed: int, purpose: str) -> bytes:
    return hashlib.sha256(
        f"{SELECTION_SCHEMA}:{purpose}:{seed}:{value}".encode("utf-8")
    ).digest()


def heldout_panel(
    documents: list[dict], holdout_problems: int
) -> tuple[list[str], list[dict]]:
    """Reproduce split_holdout's identity order and first-document panel."""
    by_problem: dict[str, list[dict]] = {}
    for document in documents:
        by_problem.setdefault(identity(document), []).append(document)
    ranked = sorted(by_problem, key=identity_rank)
    if not 0 < holdout_problems < len(ranked):
        raise ValueError(
            f"holdout problems must be in (0, {len(ranked)}); "
            f"got {holdout_problems}"
        )
    heldout = ranked[:holdout_problems]
    return heldout, [by_problem[key][0] for key in heldout]


def grading_style(document: dict) -> str:
    return (
        "rule"
        if document.get("source") == "a1_deepmind"
        else "rule-lighteval/MATH_v2"
    )


def panel_signature(panel: list[dict], rows: int = PANEL_ROWS) -> list[dict]:
    return [
        {
            "problem": document["problem"],
            "final": document["final_answer"],
            "style": grading_style(document),
        }
        for document in panel[:rows]
    ]


def prioritize_heldout_panel_rows(
    retained: list[dict], base_identities: list[str], base_panel: list[dict]
) -> list[dict]:
    """Put a byte/signature-identical retained row first per held identity."""
    reordered = list(retained)
    for problem_identity, original in zip(
        base_identities, base_panel, strict=True
    ):
        target = panel_signature([original], rows=1)[0]
        positions = [
            index
            for index, row in enumerate(reordered)
            if identity(row) == problem_identity
        ]
        if not positions:
            raise ValueError(
                f"heldout identity {problem_identity!r} has no retained row"
            )
        match = next(
            (
                index
                for index in positions
                if panel_signature([reordered[index]], rows=1)[0] == target
            ),
            None,
        )
        if match is None:
            raise ValueError(
                "heldout identity lacks an exact retained panel signature: "
                f"{problem_identity!r}"
            )
        first = positions[0]
        if match != first:
            row = reordered.pop(match)
            reordered.insert(first, row)
    return reordered


def sequence_sha256(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def require_fresh_outputs(output: str | Path) -> Path:
    output = Path(output)
    manifest = output.with_suffix(".manifest.json")
    existing = [path for path in (output, manifest) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite immutable SFT5 artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    return manifest


def decontamination_input_hashes() -> dict[str, str]:
    return {
        str(path): file_sha256(path)
        for path in sft.DECONTAMINATION_TARGETS
    }


def _raw_a1_rows(path: Path):
    required = {"question", "answer", "deepseek_solution"}
    missing = required - set(pq.read_schema(path).names)
    if missing:
        raise ValueError(f"{path} lacks A1 columns {sorted(missing)}")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
        yield from batch.to_pylist()


def materialize_a1_candidates(
    path: str | Path,
    tokenizer,
    *,
    seed: int,
) -> tuple[list[dict], Counter[str]]:
    """Run the audited A1 adapter/screens and compose answer-fenced rows."""
    path = Path(path)
    stats: Counter[str] = Counter()
    args = argparse.Namespace(seed=seed)
    exact, ngrams = sft.build_decontamination_index()

    def configured_rows(name: str):
        if name != "a1_math_deepmind.parquet":
            raise ValueError(f"unexpected A1 adapter input {name!r}")
        return _raw_a1_rows(path)

    with patch.object(sft, "rows_of", configured_rows):
        adapted = sft.adapt_a1_deepmind(stats, args)
        screened = sft.screen_candidates(
            "a1_deepmind", adapted, stats, exact, ngrams
        )
        records = []
        for candidate in screened:
            try:
                problem = sft.canonical_problem(candidate["problem"])
            except ValueError:
                stats["a1_deepmind/empty_problem_after_canonicalization"] += 1
                continue
            # SFT5 is an immutable historical v1 artifact. Reproduce its old
            # instruction-bearing prompt even though current SFT uses a bare
            # problem and completion-only token semantics.
            document = (
                f"{problem}{LEGACY_ANSWER_FENCE_SUFFIX}{THINK_OPEN}\n"
                f"{candidate['reasoning']}\n{THINK_CLOSE}\n"
                f"{ANSWER_OPEN}{candidate['final']}{ANSWER_CLOSE}"
            )
            doc_tokens = len(tokenizer.encode(document)) + 1
            if doc_tokens > sft.MAX_DOC_TOKENS:
                stats["a1_deepmind/too_long_dropped"] += 1
                continue
            records.append(
                {
                    "source": "a1_deepmind",
                    "problem": problem,
                    "document": document,
                    "final_answer": str(candidate["final"]).strip(),
                    "verified": True,
                    "doc_tokens": doc_tokens,
                }
            )
            stats["a1_deepmind/materialized"] += 1
    return records, stats


def select_a1_replacements(
    base_rows: list[dict],
    candidates: list[dict],
    *,
    swap_rows: int,
    holdout_problems: int,
    seed: int,
) -> tuple[list[dict], Counter[str]]:
    """Choose novel, post-cutoff A1 identities and one stable trace each."""
    base_identities = {identity(row) for row in base_rows}
    base_documents = {str(row["document"]) for row in base_rows}
    heldout, _ = heldout_panel(base_rows, holdout_problems)
    cutoff = identity_rank(heldout[-1])
    grouped: dict[str, list[dict]] = defaultdict(list)
    stats: Counter[str] = Counter()
    for candidate in candidates:
        candidate_identity = identity(candidate)
        if candidate_identity in base_identities:
            stats["candidate_identity_in_sft4"] += 1
            continue
        if str(candidate["document"]) in base_documents:
            stats["candidate_document_in_sft4"] += 1
            continue
        if identity_rank(candidate_identity) <= cutoff:
            stats["candidate_before_holdout_cutoff"] += 1
            continue
        grouped[candidate_identity].append(candidate)

    representatives: list[dict] = []
    for candidate_identity in sorted(grouped):
        traces = grouped[candidate_identity]
        finals = {str(trace["final_answer"]).strip() for trace in traces}
        if len(finals) != 1:
            raise ValueError(
                f"conflicting canonical finals for A1 identity "
                f"{candidate_identity!r}: {sorted(finals)!r}"
            )
        chosen = min(
            traces,
            key=lambda trace: selection_rank(
                str(trace["document"]), seed, f"trace:{candidate_identity}"
            ),
        )
        representatives.append(chosen)
        stats["candidate_duplicate_traces"] += len(traces) - 1
    representatives.sort(
        key=lambda row: selection_rank(identity(row), seed, "identity")
    )
    if len(representatives) < swap_rows:
        raise ValueError(
            f"only {len(representatives)} eligible A1 identities for "
            f"swap of {swap_rows}"
        )
    stats["eligible_a1_identities"] = len(representatives)
    stats["selection_sampled_out"] = len(representatives) - swap_rows
    return representatives[:swap_rows], stats


def assemble_swap(
    base_rows: list[dict],
    candidates: list[dict],
    *,
    swap_rows: int,
    holdout_problems: int,
    seed: int,
    expected_documents: int,
    expected_source_counts: dict[str, int],
) -> tuple[list[dict], dict]:
    """Perform and exhaustively validate the holdout-preserving swap."""
    if swap_rows < 1:
        raise ValueError("swap rows must be positive")
    base_heldout, base_panel = heldout_panel(base_rows, holdout_problems)
    heldout_cutoff = identity_rank(base_heldout[-1])
    removed = [row for row in base_rows if row["source"] == "openmath_gsm8k"]
    if len(removed) != swap_rows:
        raise ValueError(
            f"expected exactly {swap_rows} openmath_gsm8k rows, "
            f"found {len(removed)}"
        )
    retained = [row for row in base_rows if row["source"] != "openmath_gsm8k"]
    retained_identities = {identity(row) for row in retained}
    removed_identities = [identity(row) for row in removed]
    unrepresented = sorted(set(removed_identities) - retained_identities)
    unsafe_unrepresented = [
        problem_identity
        for problem_identity in unrepresented
        if identity_rank(problem_identity) <= heldout_cutoff
    ]
    if unsafe_unrepresented:
        raise ValueError(
            f"{len(unsafe_unrepresented)} removed-only identities can affect "
            "the pinned holdout"
        )
    # split_holdout chooses the first row encountered for each identity.
    # Removing the OpenMath-first rows can change panel bytes even when a
    # byte-identical retained variant exists later, so promote that exact
    # variant within every held identity before appending novel A1 rows.
    retained = prioritize_heldout_panel_rows(
        retained, base_heldout, base_panel
    )

    selected, selection_stats = select_a1_replacements(
        base_rows,
        candidates,
        swap_rows=swap_rows,
        holdout_problems=holdout_problems,
        seed=seed,
    )
    selected_identities = [identity(row) for row in selected]
    if set(selected_identities) & {identity(row) for row in base_rows}:
        raise AssertionError("selected A1 identity was already present in SFT4")
    output_rows = retained + selected
    documents = [str(row["document"]) for row in output_rows]
    if len(documents) != len(set(documents)):
        raise ValueError("SFT5 documents are not exactly unique")

    actual_counts = Counter(str(row["source"]) for row in output_rows)
    source_counts = {
        source: actual_counts.get(source, 0) for source in expected_source_counts
    }
    unexpected = set(actual_counts) - set(expected_source_counts)
    if unexpected or source_counts != expected_source_counts:
        raise ValueError(
            f"unexpected SFT5 source counts {dict(sorted(actual_counts.items()))}"
        )
    if len(output_rows) != expected_documents:
        raise ValueError(
            f"expected {expected_documents} SFT5 rows, found {len(output_rows)}"
        )

    output_heldout, output_panel = heldout_panel(output_rows, holdout_problems)
    heldout_identical = output_heldout == base_heldout
    base_signature = panel_signature(base_panel)
    output_signature = panel_signature(output_panel)
    panel_identical = output_signature == base_signature
    if not heldout_identical:
        raise AssertionError("SFT5 heldout identity sequence changed")
    if not panel_identical:
        raise AssertionError("SFT5 first-128 panel signature changed")

    metadata = {
        "source_counts": source_counts,
        "removed_rows": len(removed),
        "removed_identities": removed_identities,
        "dropped_nonheldout_identities": unrepresented,
        "selected_identities": selected_identities,
        "selection_stats": dict(sorted(selection_stats.items())),
        "heldout_identities": output_heldout,
        "panel_signature": output_signature,
        "invariants": {
            "removed_only_identities_strictly_after_holdout_cutoff": True,
            "removed_identities_retained_or_safe_nonheldout": True,
            "added_identities_novel": True,
            "documents_unique": True,
            "heldout_identity_sequence_identical": heldout_identical,
            "first128_panel_signature_identical": panel_identical,
            "source_counts_exact": True,
            "document_count_exact": True,
        },
    }
    return output_rows, metadata


def _temporary(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    )


def _publish_no_replace(temporary: Path, final: Path) -> None:
    """Atomically publish one same-filesystem file without overwriting."""
    os.link(temporary, final)
    temporary.unlink()


def write_outputs(
    rows: list[dict], output: Path, manifest_path: Path, manifest: dict
) -> dict:
    parquet_temporary = _temporary(output)
    manifest_temporary = _temporary(manifest_path)
    published: list[Path] = []
    try:
        pq.write_table(pa.Table.from_pylist(rows), parquet_temporary)
        complete_manifest = {
            **manifest,
            "output": str(output),
            "output_sha256": file_sha256(parquet_temporary),
        }
        manifest_temporary.write_text(
            json.dumps(complete_manifest, indent=2, sort_keys=True) + "\n"
        )
        _publish_no_replace(parquet_temporary, output)
        published.append(output)
        _publish_no_replace(manifest_temporary, manifest_path)
        published.append(manifest_path)
        return complete_manifest
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        parquet_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict:
    base_path = Path(args.base)
    a1_path = Path(args.a1_source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = require_fresh_outputs(output)

    base_sha256 = file_sha256(base_path)
    if base_sha256 != args.base_sha256:
        raise ValueError(
            f"SFT4 SHA mismatch: expected {args.base_sha256}, got {base_sha256}"
        )
    a1_sha256 = file_sha256(a1_path)
    decontamination_hashes = decontamination_input_hashes()
    base_rows = pq.read_table(base_path).to_pylist()
    required = {
        "source",
        "problem",
        "document",
        "final_answer",
        "verified",
        "doc_tokens",
    }
    if not base_rows or not required <= set(base_rows[0]):
        raise ValueError("SFT4 lacks required canonical columns")
    if any(row.get("verified") is not True for row in base_rows):
        raise ValueError("SFT4 contains an unverified document")

    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    candidates, adapter_stats = materialize_a1_candidates(
        a1_path, tokenizer, seed=args.seed
    )
    if file_sha256(base_path) != base_sha256:
        raise ValueError("SFT4 changed during the build")
    if file_sha256(a1_path) != a1_sha256:
        raise ValueError("raw A1 source changed during the build")
    if decontamination_input_hashes() != decontamination_hashes:
        raise ValueError("decontamination inputs changed during the build")

    output_rows, metadata = assemble_swap(
        base_rows,
        candidates,
        swap_rows=args.swap_rows,
        holdout_problems=args.holdout_problems,
        seed=args.seed,
        expected_documents=EXPECTED_DOCUMENTS,
        expected_source_counts=EXPECTED_SOURCE_COUNTS,
    )
    if file_sha256(base_path) != base_sha256:
        raise ValueError("SFT4 changed before output publication")
    if file_sha256(a1_path) != a1_sha256:
        raise ValueError("raw A1 source changed before output publication")
    if decontamination_input_hashes() != decontamination_hashes:
        raise ValueError("decontamination inputs changed before publication")
    source_token_totals = {
        source: sum(
            int(row["doc_tokens"])
            for row in output_rows
            if row["source"] == source
        )
        for source in EXPECTED_SOURCE_COUNTS
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "selection_schema": SELECTION_SCHEMA,
        "answer_fence_prompt_schema": LEGACY_ANSWER_FENCE_PROMPT_SCHEMA,
        "parent": {"path": str(base_path), "sha256": base_sha256},
        "raw_a1": {"path": str(a1_path), "sha256": a1_sha256},
        "decontamination_inputs": decontamination_hashes,
        "documents": len(output_rows),
        "source_counts": metadata["source_counts"],
        "doc_tokens": {
            "total": sum(int(row["doc_tokens"]) for row in output_rows),
            "maximum": max(int(row["doc_tokens"]) for row in output_rows),
            "by_source": source_token_totals,
        },
        "removed_rows": metadata["removed_rows"],
        "removed_identity_sha256": sequence_sha256(
            metadata["removed_identities"]
        ),
        "dropped_nonheldout_identity_count": len(
            metadata["dropped_nonheldout_identities"]
        ),
        "dropped_nonheldout_identity_sha256": sequence_sha256(
            metadata["dropped_nonheldout_identities"]
        ),
        "selected_identity_sha256": sequence_sha256(
            metadata["selected_identities"]
        ),
        "heldout_identity_sha256": sequence_sha256(
            metadata["heldout_identities"]
        ),
        "first128_panel_signature_sha256": hashlib.sha256(
            json.dumps(
                metadata["panel_signature"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "adapter_stats": dict(sorted(adapter_stats.items())),
        "selection_stats": metadata["selection_stats"],
        "invariants": metadata["invariants"],
        "args": json_args(args),
    }
    return write_outputs(output_rows, output, manifest_path, manifest)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--base-sha256", default=DEFAULT_BASE_SHA256)
    parser.add_argument("--a1-source", default=str(DEFAULT_A1_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--swap-rows", type=int, default=DEFAULT_SWAP_ROWS)
    parser.add_argument(
        "--holdout-problems", type=int, default=DEFAULT_HOLDOUT_PROBLEMS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.swap_rows < 1 or args.holdout_problems < 1:
        parser.error("swap rows and holdout problems must be positive")
    try:
        manifest = build(args)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

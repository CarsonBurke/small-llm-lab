"""Build an immutable reviewed RL candidate and audit its reserved science panel.

Run through mlq. This never selects a training default or starts training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pyarrow.parquet as pq

from postraining.choice_prompt import split_options
from postraining.choice_rl_pool import _Components, SPLIT_RULE, SPLIT_SCHEMA
from postraining.core import GPT2BPETokenizer, encode_prompt, load_unique_math_rows, math_corpus_identity
from postraining.math_prompt import canonicalize_answer_fence_rows
from postraining.prepare_sft_corpus import QUESTION_RULE, build_question_index, matching_questions, word_string
from postraining.prepare_vapo_mixture import atomic_json, atomic_parquet
from postraining.vapo.mixture import file_sha256, load_mixture_manifest


CONFIRMED_CODE_CATEGORIES = {
    "contradictory_contract", "overrestricted_output",
    "undisclosed_invalid_input_requirement", "invalid_test_input",
    "internally_inconsistent_problem",
}
CONFIRMED_SEVERITIES = {
    "confirmed_bad_key", "confirmed_multiple_correct", "confirmed_underspecified",
    "confirmed_inconsistent_domain", "confirmed_cross_source_duplicate",
}


def confirmed_quarantine(items: list[dict]) -> tuple[dict[str, set[str]], list[dict]]:
    excluded: dict[str, set[str]] = {}
    retained = []
    for item in items:
        confirmed = (
            item.get("category") in CONFIRMED_CODE_CATEGORIES
            or item.get("severity") in CONFIRMED_SEVERITIES
        )
        if confirmed:
            excluded.setdefault(item["rl_source"], set()).add(item["id"])
        else:
            retained.append(item)
    return excluded, retained


def choice_parts(row: dict) -> tuple[str, str, str]:
    problem = row["prompt"][0]["content"]
    parsed = split_options(problem)
    answer = row["reward_model"]["ground_truth"]
    if parsed is None or answer not in parsed.labels:
        raise ValueError("component screen requires a gradeable single-choice row")
    return problem, parsed.question, answer


def choice_component_matches(heldout: list[dict], candidates: list[dict]) -> set[int]:
    """Candidate indices connected to the panel by the established split rule.

    Using the existing component implementation also catches transitive
    rewordings, including option reorderings and near-duplicate answer text.
    """
    if not candidates:
        return set()
    triples = [choice_parts(row) for row in heldout + candidates]
    problems, stems, answers = map(list, zip(*triples))
    components = _Components(len(triples))
    all_indices = list(range(len(triples)))
    components.link(problems, stems, all_indices, all_indices)
    components.link(problems, stems, all_indices, all_indices, None)
    components.link_answers(problems, answers)
    components.link_reworded(problems, stems, answers)
    # Template-gram frequency depends on which side is indexed. Close both
    # directions after each expansion of the reserved component set, just
    # as the corpus splitter does for its SFT/RL partitions.
    while True:
        roots = {components.find(i) for i in range(len(heldout))}
        reserved = [i for i in all_indices if components.find(i) in roots]
        remaining = [i for i in all_indices if components.find(i) not in roots]
        joins = components.link(problems, stems, remaining, reserved)
        joins += components.link(problems, stems, reserved, remaining)
        if not joins:
            break
    roots = {components.find(i) for i in range(len(heldout))}
    return {i for i in range(len(candidates)) if components.find(i + len(heldout)) in roots}


def panel_index(heldout: list[dict]):
    return build_question_index([choice_parts(row)[1] for row in heldout])


def audit_sft(path: Path, heldout: list[dict]) -> dict:
    """Stream actual trained problems, not the teacher's source partition."""
    index = panel_index(heldout)
    hits = []
    source_counts = Counter()
    choices = {}
    scanned = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=["source", "problem", "final_answer"]):
        for record in batch.to_pylist():
            scanned += 1
            source_counts[record["source"]] += 1
            problem = record["problem"]
            matched = matching_questions(problem, index)
            if matched:
                hits.append({"sft_row": scanned - 1, "source": record["source"], "problem": problem,
                             "matched_panel_stems": [stem for stem, i in index.exact.items() if i in matched]})
            parsed = split_options(problem)
            if parsed is not None and record["final_answer"] in parsed.labels:
                key = (problem, record["final_answer"])
                choices.setdefault(key, {
                    "prompt": [{"role": "user", "content": problem}],
                    "reward_model": {"ground_truth": record["final_answer"]},
                    "extra_info": {"sft_row": scanned - 1, "source": record["source"]},
                })
        print(json.dumps({"phase": "actual_sft_scan", "rows": scanned, "question_matches": len(hits)}), flush=True)
    candidates = list(choices.values())
    component_hits = choice_component_matches(heldout, candidates)
    return {
        "path": str(path), "sha256": file_sha256(path), "rows_scanned": scanned,
        "rows_by_source": dict(source_counts), "question_rule": QUESTION_RULE,
        "question_matches": hits, "unique_gradeable_choice_questions": len(candidates),
        "choice_component_rule": SPLIT_RULE, "choice_component_schema": SPLIT_SCHEMA,
        "choice_component_matches": [candidates[i] for i in sorted(component_hits)],
        "disjoint_under_checked_rules": not hits and not component_hits,
        "limitation": "Lexical/component screens cannot certify absence of every semantic paraphrase or pretraining exposure.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists() or args.evidence.exists():
        parser.error("refusing to overwrite immutable output/evidence")
    if args.prompt_tokens < 1:
        parser.error("--prompt-tokens must be positive")
    _, sources, original = load_mixture_manifest(args.input_manifest)
    sft = Path(original["sft_corpus"])
    if file_sha256(sft) != original["sft_corpus_sha256"]:
        parser.error("actual SFT corpus differs from the bound mixture corpus")
    heldout = canonicalize_answer_fence_rows(load_unique_math_rows(args.holdout))
    quarantine = json.loads(args.quarantine.read_text())
    excluded, retained = confirmed_quarantine(quarantine["items"])
    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    index = panel_index(heldout)
    canonical_sources = {source.name: canonicalize_answer_fence_rows(list(source.rows)) for source in sources}
    if excluded.keys() - canonical_sources.keys():
        raise ValueError("quarantine names sources absent from the candidate mixture")
    science_by_id = {row["extra_info"]["index"]: row for row in canonical_sources["science_mc"]}
    for row in heldout:
        original_row = science_by_id.get(row["extra_info"]["index"])
        if original_row is None or choice_parts(original_row) != choice_parts(row):
            raise ValueError("reserved panel is not an unchanged subset of the science source")
    choice_rows = []
    choice_locations = []
    for name, rows in canonical_sources.items():
        for i, row in enumerate(rows):
            if (row.get("extra_info") or {}).get("option_order") is not None:
                choice_locations.append((name, i))
                choice_rows.append(row)
    matched = choice_component_matches(heldout, choice_rows)
    component_locations = {choice_locations[i] for i in matched}
    print(json.dumps({"phase": "training_components", "heldout_connected_rows": len(matched)}), flush=True)
    sft_audit = audit_sft(sft, heldout)
    report = {
        "schema": "reviewed_rl_candidate/v1", "status": "candidate_not_enabled_for_training",
        "input_manifest": str(args.input_manifest), "input_manifest_sha256": file_sha256(args.input_manifest),
        "quarantine": str(args.quarantine), "quarantine_sha256": file_sha256(args.quarantine),
        "unconfirmed_findings_retained": retained, "prompt_tokens_including_bos_limit": args.prompt_tokens,
        "science_holdout": {"path": str(args.holdout), "sha256": file_sha256(args.holdout), "rows": len(heldout),
                            "status": "reserved_for_future_RL; previously used for model-development diagnostics",
                            "question_rule": QUESTION_RULE, "choice_component_rule": SPLIT_RULE},
        "actual_sft_holdout_audit": sft_audit, "sources": {},
    }
    args.output_dir.mkdir(parents=True)
    entries = []
    for source in sources:
        rows = canonical_sources[source.name]
        present = {row["extra_info"]["index"] for row in rows}
        missing = excluded.get(source.name, set()) - present
        if missing:
            raise ValueError(f"reviewed IDs absent from {source.name}: {sorted(missing)}")
        kept = []
        dropped = []
        for i, row in enumerate(rows):
            identity = row["extra_info"]["index"]
            reasons = []
            if identity in excluded.get(source.name, set()):
                reasons.append("confirmed_reviewed_defect")
            length = len(encode_prompt(tokenizer, row["prompt"][0]["content"]))
            if length > args.prompt_tokens:
                reasons.append("canonical_prompt_over_budget")
            if (source.name, i) in component_locations or matching_questions(row["prompt"][0]["content"], index):
                reasons.append("reserved_science_question")
            if reasons:
                dropped.append({"id": identity, "reasons": reasons, "prompt_tokens": length})
            else:
                kept.append({key: value for key, value in row.items() if not key.startswith("_")})
        if not kept:
            raise ValueError(f"review removes every row from {source.name}")
        output = args.output_dir / f"{source.name}.parquet"
        atomic_parquet(kept, output)
        reloaded = load_unique_math_rows(output)
        if len(reloaded) != len(kept):
            raise ValueError(f"canonicalization changed uniqueness in {source.name}")
        entries.append({"name": source.name, "path": str(output), "verifier": source.verifier,
                        "rows": len(kept), "sha256": file_sha256(output), "math_corpus_identity": math_corpus_identity(reloaded)})
        report["sources"][source.name] = {"input_rows": len(rows), "output_rows": len(kept), "removed": dropped}
        print(json.dumps({"phase": "write", "source": source.name, "kept": len(kept), "removed": len(dropped)}), flush=True)
    manifest = {**original, "sources": entries, "prompts_per_cycle": sum(entry["rows"] for entry in entries),
                "reviewed_candidate": {"evidence": str(args.evidence), "science_holdout_sha256": file_sha256(args.holdout)}}
    output_manifest = args.output_dir / "mixture.manifest.json"
    atomic_json(manifest, output_manifest)
    load_mixture_manifest(output_manifest)
    report["output_manifest"] = str(output_manifest)
    report["output_manifest_sha256"] = file_sha256(output_manifest)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report, args.evidence)
    print(json.dumps({"output": str(output_manifest), "evidence": str(args.evidence),
                      "sft_disjoint_under_checked_rules": sft_audit["disjoint_under_checked_rules"]}), flush=True)


if __name__ == "__main__":
    main()

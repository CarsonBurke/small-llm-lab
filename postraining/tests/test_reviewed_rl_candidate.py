import pyarrow as pa
import pyarrow.parquet as pq
import json
import sys

from scripts.prepare_reviewed_rl_candidate import (
    audit_sft, choice_component_matches, confirmed_quarantine, main,
)
from postraining.core import load_unique_math_rows, math_corpus_identity, math_corpus_policy_sha256
from postraining.vapo.mixture import VAPO_MIXTURE_SCHEMA, file_sha256, load_mixture_manifest


def row(problem, answer):
    return {"prompt": [{"role": "user", "content": problem}],
            "reward_model": {"ground_truth": answer}}


def test_only_confirmed_findings_are_quarantined():
    excluded, retained = confirmed_quarantine([
        {"rl_source": "code", "id": "bad", "category": "invalid_test_input"},
        {"rl_source": "science", "id": "wrong", "severity": "confirmed_bad_key"},
        {"rl_source": "math", "id": "domain", "severity": "confirmed_inconsistent_domain"},
        {"rl_source": "math", "id": "duplicate", "severity": "confirmed_cross_source_duplicate"},
        {"rl_source": "code", "id": "uncertain", "category": "ambiguous"},
        {"rl_source": "code", "id": "weak", "category": "weak_tests"},
    ])
    assert excluded == {"code": {"bad"}, "science": {"wrong"}, "math": {"domain", "duplicate"}}
    assert [item["id"] for item in retained] == ["uncertain", "weak"]


def test_panel_reservation_tracks_questions_across_option_permutations():
    panel = [row("Which animal purrs?\n\nA. cat\nB. dog", "A")]
    candidates = [
        row("Which animal purrs?\n\nA. dog\nB. cat", "B"),
        row("Which celestial body orbits Earth?\n\nA. moon\nB. sun", "A"),
    ]
    assert choice_component_matches(panel, candidates) == {0}


def test_audit_reads_actual_sft_problems_and_reports_exposure(tmp_path):
    path = tmp_path / "sft.parquet"
    panel = [row("Which animal purrs?\n\nA. cat\nB. dog", "A")]
    pq.write_table(pa.Table.from_pylist([
        {"source": "science_traces", "problem": "Which animal purrs?\n\nA. dog\nB. cat", "final_answer": "B"},
        {"source": "math", "problem": "Calculate 2 + 3.", "final_answer": "5"},
    ]), path)
    result = audit_sft(path, panel)
    assert result["rows_scanned"] == 2
    assert len(result["question_matches"]) == 1
    assert len(result["choice_component_matches"]) == 1
    assert result["disjoint_under_checked_rules"] is False


def test_actual_sft_disjointness_does_not_require_teacher_partition(tmp_path):
    path = tmp_path / "sft.parquet"
    panel = [row("Which animal purrs?\n\nA. cat\nB. dog", "A")]
    pq.write_table(pa.Table.from_pylist([
        {"source": "math", "problem": "Calculate 2 + 3.", "final_answer": "5"},
    ]), path)
    result = audit_sft(path, panel)
    assert result["rows_scanned"] == 1
    assert result["disjoint_under_checked_rules"] is True


def test_candidate_build_removes_reviewed_overbudget_and_reserved_rows(tmp_path, monkeypatch):
    def example(identity, question, options, answer="A"):
        value = row(question + "\n\n" + "\n".join(f"{letter}. {text}" for letter, text in zip("AB", options)), answer)
        value["reward_model"]["style"] = "rule"
        value["extra_info"] = {"index": identity, "prompt_contract": "bare", "option_order": [0, 1],
                               "source_position": "AB".index(answer)}
        return value

    heldout = example("held", "Which animal purrs?", ["cat", "dog"])
    science = [heldout, example("keep_science", "Which planet has rings?", ["Saturn", "Mars"])]
    knowledge = [example("bad", "Which element has atomic number 1?", ["Hydrogen", "Helium"]),
                 example("long", "A long premise. " * 300 + "What follows?", ["blue", "red"]),
                 example("keep_knowledge", "Who wrote Hamlet?", ["Shakespeare", "Dickens"])]
    sft_path = tmp_path / "sft.parquet"
    pq.write_table(pa.Table.from_pylist([{"source": "math", "problem": "Calculate 2 + 3.", "final_answer": "5"}]), sft_path)
    entries = []
    for name, rows in [("science_mc", science), ("ultradata_knowledge", knowledge)]:
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        entries.append({"name": name, "path": str(path), "verifier": "math", "rows": len(rows),
                        "sha256": file_sha256(path), "math_corpus_identity": math_corpus_identity(load_unique_math_rows(path))})
    manifest_path = tmp_path / "input.json"
    manifest_path.write_text(json.dumps({"schema": VAPO_MIXTURE_SCHEMA,
        "math_corpus_policy_sha256": math_corpus_policy_sha256(), "prompts_per_cycle": 5, "sources": entries,
        "sft_corpus": str(sft_path), "sft_corpus_sha256": file_sha256(sft_path)}))
    panel_path = tmp_path / "heldout.parquet"
    pq.write_table(pa.Table.from_pylist([heldout]), panel_path)
    quarantine_path = tmp_path / "quarantine.json"
    quarantine_path.write_text(json.dumps({"items": [{"rl_source": "ultradata_knowledge", "id": "bad", "severity": "confirmed_bad_key"}]}))
    output = tmp_path / "output"
    evidence = tmp_path / "evidence.json"
    monkeypatch.setattr(sys, "argv", ["prepare", "--input-manifest", str(manifest_path), "--quarantine", str(quarantine_path),
        "--holdout", str(panel_path), "--output-dir", str(output), "--evidence", str(evidence)])
    main()
    loaded, _, _ = load_mixture_manifest(output / "mixture.manifest.json")
    assert {row["extra_info"]["index"] for row in loaded} == {"keep_science", "keep_knowledge"}
    report = json.loads(evidence.read_text())
    reasons = {item["id"]: item["reasons"] for source in report["sources"].values() for item in source["removed"]}
    assert reasons == {"held": ["reserved_science_question"], "bad": ["confirmed_reviewed_defect"], "long": ["canonical_prompt_over_budget"]}
    assert report["actual_sft_holdout_audit"]["disjoint_under_checked_rules"]
    assert file_sha256(manifest_path) == report["input_manifest_sha256"]

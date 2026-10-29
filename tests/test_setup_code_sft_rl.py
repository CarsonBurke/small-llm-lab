"""CPU-only checks for the ordered, bounded code/RL separation audit."""

import numpy as np
import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from postraining.prepare_ultradata_code import minhash, near_twins
from scripts.setup_code_sft_rl import ordered_signatures, rl_audit_inputs
from scripts import setup_code_sft_rl as setup


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "REPO", tmp_path)
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    pool = tmp_path / setup.POOL
    pool.parent.mkdir(parents=True)
    pq.write_table(pa.table({"text": ["one", "two"]}), pool)
    pool_manifest = {"dataset": "openbmb/UltraData-Code", "revision": "pinned",
                     "sft_pool": {"sha256": setup.sha256(pool)}}
    write_json(tmp_path / setup.POOL_MANIFEST, pool_manifest)
    source = tmp_path / "rl.parquet"
    pq.write_table(pa.table({"text": ["other"]}), source)
    sources = [{"name": "rl", "path": "rl.parquet", "sha256": setup.sha256(source), "rows": 1}]
    write_json(tmp_path / setup.MIXTURE, {"sources": sources})
    filtered, audit_path, traces, manifest_path = setup.prepared_corpus_paths("test")
    pq.write_table(pa.table({"text": ["one"]}), filtered)
    pq.write_table(pa.table({"text": ["trace"]}), traces)
    audit = {
        "schema": setup.SEPARATION_SCHEMA, "source_pool": setup.POOL,
        "source_pool_manifest": setup.POOL_MANIFEST, "rl_mixture": setup.MIXTURE,
        "near_duplicate_rule": setup.SEPARATION_RULE,
        "source_pool_sha256": setup.sha256(pool),
        "source_pool_manifest_sha256": setup.sha256(tmp_path / setup.POOL_MANIFEST),
        "rl_mixture_sha256": setup.sha256(tmp_path / setup.MIXTURE),
        "dataset": pool_manifest["dataset"], "revision": "pinned", "rl_sources": sources,
        "output_sha256": setup.sha256(filtered), "input_rows": 2, "kept_rows": 1,
        "removed_union": 1,
    }
    write_json(audit_path, audit)
    from postraining.math_prompt import ANSWER_FENCE_PROMPT_SCHEMA
    manifest = {
        "schema": "instruction_corpus_answer_fence/v2",
        "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
        "seq_len": 1024, "verified": True, "documents": 1, "verified_documents": 1,
        "corpora": [{"name": "ultradata_code_l3", "dataset": pool_manifest["dataset"], "revision": "pinned"}],
        "shard_sha256": [{"corpus": "ultradata_code_l3", "shard": filtered.name, "sha256": setup.sha256(filtered)}],
        "output_sha256": setup.sha256(traces),
        "rl_separation_audit": str(audit_path.relative_to(tmp_path)),
        "rl_separation_audit_sha256": setup.sha256(audit_path),
    }
    write_json(manifest_path, manifest)
    return tmp_path


def test_complete_preparation_reused_without_writes(prepared):
    paths = setup.prepared_corpus_paths("test")
    before = [path.read_bytes() for path in paths]
    assert setup.reuse_prepared_corpus("test")
    assert [path.read_bytes() for path in paths] == before


def test_absent_preparation_is_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "REPO", tmp_path)
    assert not setup.reuse_prepared_corpus("fresh")


def test_completed_filtered_stage_continues_framing(prepared, monkeypatch):
    filtered, audit, traces, manifest = setup.prepared_corpus_paths("test")
    traces.unlink()
    manifest.unlink()
    before = [filtered.read_bytes(), audit.read_bytes()]
    assert not setup.reuse_prepared_corpus("test")
    assert setup.reuse_prepared_corpus("test", filtered_only=True)
    calls = []
    monkeypatch.setattr(setup, "frame_corpus", lambda *args: calls.append(args))
    monkeypatch.setattr(setup, "ordered_signatures", lambda *args, **kwargs: pytest.fail("repeated MinHash"))
    setup.prepare_corpus("test")
    assert calls == [("test", filtered, audit)]
    assert [filtered.read_bytes(), audit.read_bytes()] == before


def test_changed_filtered_stage_rejected(prepared):
    filtered, _, traces, manifest = setup.prepared_corpus_paths("test")
    traces.unlink()
    manifest.unlink()
    filtered.write_bytes(filtered.read_bytes() + b" ")
    with pytest.raises(ValueError, match="disjoint pool hash"):
        setup.reuse_prepared_corpus("test", filtered_only=True)


@pytest.mark.parametrize("index", range(4))
def test_partial_preparation_rejected(prepared, index):
    setup.prepared_corpus_paths("test")[index].unlink()
    with pytest.raises(ValueError, match="partial prepared corpus"):
        setup.reuse_prepared_corpus("test")


@pytest.mark.parametrize("relative", [setup.POOL, setup.POOL_MANIFEST, setup.MIXTURE,
    "rl.parquet", "postraining/data/test_disjoint_pool.parquet",
    "postraining/data/test_traces.parquet", "postraining/data/test_disjoint_pool.manifest.json"])
def test_changed_preparation_or_source_rejected(prepared, relative):
    path = prepared / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="mismatch"):
        setup.reuse_prepared_corpus("test")


@pytest.mark.parametrize("field,value", [("schema", "old"), ("verified", False),
    ("seq_len", 512), ("documents", 2), ("rl_separation_audit", "different")])
def test_invalid_trace_contract_rejected(prepared, field, value):
    path = setup.prepared_corpus_paths("test")[3]
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch"):
        setup.reuse_prepared_corpus("test")


def test_exact_resume_keeps_original_epoch_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "REPO", tmp_path)
    plan = setup.stage_plan("test")
    fresh = setup.sft_execution_command(plan)
    assert fresh == plan["commands"][1]
    run = tmp_path / "postraining/runs/test"
    run.mkdir(parents=True)
    with pytest.raises(ValueError, match="no exact resume"):
        setup.sft_execution_command(plan)
    (run / "resume.pt").write_bytes(b"trainer validates state")
    resumed = setup.sft_execution_command(plan)
    assert resumed == [*fresh, "--resume"]
    assert resumed[resumed.index("--epochs") + 1] == "1"
    (run / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="already finished"):
        setup.sft_execution_command(plan)


def test_parallel_signatures_and_overlap_mask_match_serial():
    # Multiple batches, a short final batch, duplicates and an unrelated task
    # check both row ordering and the actual downstream exclusion decision.
    queries = [
        "Write a function summing all even numbers in an input list.",
        "Return the reversed lowercase letters of a supplied input string.",
        "Write a function summing all even numbers in an input list.",
        "Compute the determinant of a square matrix by elimination.",
        "",
    ]
    index = [queries[1], queries[0], "Find the shortest path through a weighted graph."]
    serial_queries = np.stack([minhash(text) for text in queries])
    serial_index = np.stack([minhash(text) for text in index])
    parallel_queries = ordered_signatures(queries, workers=4, batch_size=2)
    parallel_index = ordered_signatures(index, workers=4, batch_size=2)
    np.testing.assert_array_equal(parallel_queries, serial_queries)
    np.testing.assert_array_equal(parallel_index, serial_index)
    assert parallel_queries.dtype == np.uint32
    np.testing.assert_array_equal(
        near_twins(parallel_queries, parallel_index),
        near_twins(serial_queries, serial_index),
    )


def test_rl_audit_projects_prompt_and_entry_points(tmp_path):
    code_path = tmp_path / "code.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": [{"role": "user", "content": "Implement Foo and bar."}],
                    "verification_info": {
                        "entry_points": ["Foo", "bar"],
                        "tests": ["bulky payload unused by this audit"],
                    },
                    "reward_model": {"ground_truth": "unused"},
                },
            ]
        ),
        code_path,
    )
    assert rl_audit_inputs(code_path) == (["Implement Foo and bar."], {("bar", "foo")})
    math_path = tmp_path / "math.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": [{"role": "user", "content": "What is 1 + 1?"}],
                    "reward_model": {"ground_truth": "2"},
                },
            ]
        ),
        math_path,
    )
    assert rl_audit_inputs(math_path) == (["What is 1 + 1?"], set())

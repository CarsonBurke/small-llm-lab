"""Focused invariants for the immutable SFT4-to-SFT5 A1 swap."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import postraining.prepare_sft5_a1_swap as swap


class _Tokenizer:
    def __init__(self, **_kwargs):
        pass

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def _row(
    source: str,
    problem: str,
    final: str,
    document_suffix: str,
    tokens: int = 10,
) -> dict:
    return {
        "source": source,
        "problem": problem,
        "document": f"{problem}\ntrace-{document_suffix}\n<answer>{final}</answer>",
        "final_answer": final,
        "verified": True,
        "doc_tokens": tokens,
    }


def _base_rows() -> list[dict]:
    # OpenMath occurs first, matching SFT4 source order. Its retained
    # variants preserve the exact panel-visible problem/final/style fields.
    return [
        _row("openmath_gsm8k", "Shared problem one", "1", "open-one"),
        _row("openmath_gsm8k", "Shared problem two", "2", "open-two"),
        _row("had653_gold", "Shared problem one", "1", "had-one"),
        _row("gsm8k_socratic", "Shared problem two", "2", "soc-two"),
        _row("had653_gold", "Retained problem three", "3", "had-three"),
        _row("gsm8k_socratic", "Retained problem four", "4", "soc-four"),
    ]


def _post_cutoff_problems(base: list[dict], holdout: int, count: int) -> list[str]:
    heldout, _ = swap.heldout_panel(base, holdout)
    cutoff = swap.identity_rank(heldout[-1])
    problems = []
    index = 0
    while len(problems) < count:
        problem = f"Novel A1 problem {index}"
        if swap.identity_rank(swap.sft.normalize_problem(problem)) > cutoff:
            problems.append(problem)
        index += 1
    return problems


def _candidates(base: list[dict], count: int = 4) -> list[dict]:
    return [
        _row("a1_deepmind", problem, str(100 + index), f"a1-{index}", 20)
        for index, problem in enumerate(_post_cutoff_problems(base, 2, count))
    ]


def _expected_counts() -> dict[str, int]:
    return {
        "a1_deepmind": 2,
        "openmath_gsm8k": 0,
        "had653_gold": 2,
        "gsm8k_socratic": 2,
    }


def test_swap_preserves_holdout_panel_and_selects_stably():
    base = _base_rows()
    candidates = _candidates(base)
    # A second trace for one identity must be reduced by stable hash, not
    # input order. Both carry the same canonical final.
    duplicate = {
        **candidates[0],
        "document": candidates[0]["document"] + " alternative",
        "doc_tokens": 21,
    }
    output, metadata = swap.assemble_swap(
        base,
        [*candidates, duplicate],
        swap_rows=2,
        holdout_problems=2,
        seed=7,
        expected_documents=6,
        expected_source_counts=_expected_counts(),
    )
    again, again_metadata = swap.assemble_swap(
        base,
        [duplicate, *reversed(candidates)],
        swap_rows=2,
        holdout_problems=2,
        seed=7,
        expected_documents=6,
        expected_source_counts=_expected_counts(),
    )

    assert [row["document"] for row in output] == [
        row["document"] for row in again
    ]
    assert metadata["selected_identities"] == again_metadata["selected_identities"]
    assert metadata["selection_stats"]["candidate_duplicate_traces"] == 1
    assert all(metadata["invariants"].values())
    assert not any(row["source"] == "openmath_gsm8k" for row in output)
    base_heldout, base_panel = swap.heldout_panel(base, 2)
    output_heldout, output_panel = swap.heldout_panel(output, 2)
    assert output_heldout == base_heldout
    assert swap.panel_signature(output_panel) == swap.panel_signature(base_panel)


def test_candidate_group_rejects_conflicting_canonical_finals():
    base = _base_rows()
    candidate = _candidates(base, count=1)[0]
    conflict = {
        **candidate,
        "document": candidate["document"] + " conflicting",
        "final_answer": "different",
    }
    with pytest.raises(ValueError, match="conflicting canonical finals"):
        swap.select_a1_replacements(
            base,
            [candidate, conflict],
            swap_rows=1,
            holdout_problems=2,
            seed=0,
        )


def test_removed_only_identity_is_allowed_strictly_after_holdout_cutoff():
    base = _base_rows()
    open_only_problem = _post_cutoff_problems(base, 2, 1)[0]
    base.append(
        _row("openmath_gsm8k", open_only_problem, "9", "open-only")
    )
    expected = {**_expected_counts(), "a1_deepmind": 3}
    output, metadata = swap.assemble_swap(
        base,
        _candidates(base, count=4),
        swap_rows=3,
        holdout_problems=2,
        seed=0,
        expected_documents=7,
        expected_source_counts=expected,
    )
    assert len(output) == 7
    assert metadata["dropped_nonheldout_identities"] == [
        swap.sft.normalize_problem(open_only_problem)
    ]
    assert metadata["invariants"][
        "removed_only_identities_strictly_after_holdout_cutoff"
    ]


def test_removed_only_identity_at_holdout_cutoff_fails_closed():
    base = _base_rows()
    heldout, _ = swap.heldout_panel(base, 3)
    shared = {
        swap.sft.normalize_problem("Shared problem one"),
        swap.sft.normalize_problem("Shared problem two"),
    }
    target = next(problem for problem in heldout if problem in shared)
    base = [
        row
        for row in base
        if row["source"] == "openmath_gsm8k" or swap.identity(row) != target
    ]
    with pytest.raises(ValueError, match="can affect the pinned holdout"):
        swap.assemble_swap(
            base,
            [],
            swap_rows=2,
            holdout_problems=3,
            seed=0,
            expected_documents=4,
            expected_source_counts={},
        )


def test_later_exact_panel_variant_is_promoted_before_nonmatching_variant():
    base = _base_rows()
    for row in base:
        if row["source"] == "had653_gold" and row["problem"] == "Shared problem one":
            row["problem"] = "Shared  problem one"
            row["document"] = "nonmatching retained variant"
    base.append(
        _row(
            "gsm8k_socratic",
            "Shared problem one",
            "1",
            "later-exact-variant",
        )
    )
    expected = {**_expected_counts(), "gsm8k_socratic": 3}
    candidates = [
        _row("a1_deepmind", problem, str(50 + index), f"panel-a1-{index}")
        for index, problem in enumerate(_post_cutoff_problems(base, 3, 4))
    ]
    output, _ = swap.assemble_swap(
        base,
        candidates,
        swap_rows=2,
        holdout_problems=3,
        seed=3,
        expected_documents=7,
        expected_source_counts=expected,
    )
    base_heldout, base_panel = swap.heldout_panel(base, 3)
    output_heldout, output_panel = swap.heldout_panel(output, 3)
    assert output_heldout == base_heldout
    assert swap.panel_signature(output_panel) == swap.panel_signature(base_panel)


def test_builder_writes_hashed_manifest_counts_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    base = _base_rows()
    base_path = tmp_path / "sft4.parquet"
    raw_path = tmp_path / "a1.parquet"
    output = tmp_path / "sft5.parquet"
    pq.write_table(pa.Table.from_pylist(base), base_path)
    pq.write_table(pa.Table.from_pylist([{"placeholder": "raw"}]), raw_path)
    candidates = _candidates(base)
    monkeypatch.setattr(swap, "GPT2BPETokenizer", _Tokenizer)
    monkeypatch.setattr(
        swap,
        "materialize_a1_candidates",
        lambda *_args, **_kwargs: (candidates, swap.Counter({"materialized": 4})),
    )
    monkeypatch.setattr(swap, "EXPECTED_DOCUMENTS", 6)
    monkeypatch.setattr(swap, "EXPECTED_SOURCE_COUNTS", _expected_counts())
    monkeypatch.setattr(swap, "decontamination_input_hashes", lambda: {})
    args = Namespace(
        base=str(base_path),
        base_sha256=swap.file_sha256(base_path),
        a1_source=str(raw_path),
        output=str(output),
        swap_rows=2,
        holdout_problems=2,
        seed=5,
    )

    manifest = swap.build(args)
    manifest_path = output.with_suffix(".manifest.json")
    assert output.exists() and manifest_path.exists()
    assert manifest["output_sha256"] == swap.file_sha256(output)
    assert manifest["parent"]["sha256"] == swap.file_sha256(base_path)
    assert manifest["raw_a1"]["sha256"] == swap.file_sha256(raw_path)
    assert manifest["source_counts"] == _expected_counts()
    assert manifest["documents"] == 6
    assert manifest["dropped_nonheldout_identity_count"] == 0
    assert manifest["dropped_nonheldout_identity_sha256"] == swap.sequence_sha256([])
    assert manifest["doc_tokens"]["total"] == sum(
        row["doc_tokens"] for row in pq.read_table(output).to_pylist()
    )
    assert all(manifest["invariants"].values())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        swap.build(args)


def test_atomic_publication_cleans_partial_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "sft5.parquet"
    manifest_path = output.with_suffix(".manifest.json")
    real_publish = swap._publish_no_replace
    calls = 0

    def fail_manifest(temporary: Path, final: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected manifest publication failure")
        real_publish(temporary, final)

    monkeypatch.setattr(swap, "_publish_no_replace", fail_manifest)
    with pytest.raises(OSError, match="injected manifest"):
        swap.write_outputs(
            [_row("a1_deepmind", "P", "1", "trace")],
            output,
            manifest_path,
            {"schema": "test"},
        )
    assert not output.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_configured_parent_sha_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    base_path = tmp_path / "sft4.parquet"
    raw_path = tmp_path / "a1.parquet"
    pq.write_table(pa.Table.from_pylist(_base_rows()), base_path)
    raw_path.write_bytes(b"raw")
    monkeypatch.setattr(swap, "GPT2BPETokenizer", _Tokenizer)
    args = Namespace(
        base=str(base_path),
        base_sha256="0" * 64,
        a1_source=str(raw_path),
        output=str(tmp_path / "out.parquet"),
        swap_rows=2,
        holdout_problems=2,
        seed=0,
    )
    with pytest.raises(ValueError, match="SFT4 SHA mismatch"):
        swap.build(args)

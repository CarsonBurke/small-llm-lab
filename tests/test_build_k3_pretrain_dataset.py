from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from postraining.decontaminate import ProblemGuard
from scripts import build_k3_pretrain_dataset as builder
from scripts.build_k3_pretrain_dataset import (
    DocumentDeduplicator,
    LoaderAlignedShardWriter,
    RawDocument,
    logical_interleave,
    paragraph_chunks,
    split_token_document,
    stable_digest,
)
from scripts.build_math_mix_dataset import GPT2BatchEncoder


def read_payload(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    assert header[0] == 20240520
    assert header[1] == 1
    return np.fromfile(path, dtype="<u2", offset=256 * 4, count=int(header[2]))


class _SourceDecoder:
    def decode(self, batches: list[list[int]]) -> list[str]:
        return ["".join(chr(96 + token) for token in batch) for batch in batches]


class _TargetEncoder:
    def encode(self, texts: list[str], out_type=int) -> list[list[int]]:
        assert out_type is int
        return [[100 + ord(character) - 96 for character in text] for text in texts]


def test_utf8_bytes_is_a_first_class_corpus_encoding() -> None:
    encoder, provenance = builder.load_corpus_tokenizer(
        "utf8_bytes", GPT2BatchEncoder()
    )

    assert encoder.encode(["A🙂"], out_type=int) == [list("A🙂".encode("utf-8"))]
    assert encoder.decode([list("A🙂".encode("utf-8"))]) == ["A🙂"]
    assert provenance["kind"] == "utf8_bytes"
    assert provenance["vocab_size"] == 261
    assert provenance["eot_id"] == 256
    assert len(provenance["spec_sha256"]) == 64


def test_custom_tokenizer_reencodes_challenge_validation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    shard = source / "fineweb_val_000000.bin"
    builder.write_shard(
        shard,
        np.asarray(
            [builder.GPT2_EOT_ID, 1, 2, builder.GPT2_EOT_ID, 3, 4],
            dtype=np.int32,
        ),
    )

    outputs, metadata = builder.materialize_challenge_validation(
        [shard], output, _SourceDecoder(), _TargetEncoder(), target_eot_id=7
    )

    assert [path.name for path in outputs] == ["fineweb_val_000000.bin"]
    np.testing.assert_array_equal(
        read_payload(outputs[0]), np.asarray([7, 101, 102, 7, 103, 104, 7])
    )
    assert metadata == {
        "reencoded": True,
        "documents": 2,
        "source_tokens": 6,
        "tokens": 7,
    }


def test_source_manifest_weights_match_domains() -> None:
    manifest = json.loads(Path("pretraining/k3_sources.json").read_text())
    assert sum(source["weight"] for source in manifest["sources"]) == 1.0
    for domain, expected in manifest["domains"].items():
        actual = sum(
            source["weight"]
            for source in manifest["sources"]
            if source["domain"] == domain
        )
        assert abs(actual - expected) < 1e-12
    assert len(manifest["sources"]) >= 10
    for source in manifest["sources"]:
        if "full_files" in source:
            assert set(source["files"]).issubset(source["full_files"])
            assert len(source["full_files"]) > len(source["files"])


def test_weight_profiles_cover_every_source_and_match_domains() -> None:
    manifest = json.loads(Path("pretraining/k3_sources.json").read_text())
    source_domains = {
        source["name"]: source["domain"] for source in manifest["sources"]
    }
    for path in Path("pretraining").glob("k3_weights_*.json"):
        profile = json.loads(path.read_text())
        assert set(profile["sources"]) == set(source_domains)
        assert abs(sum(profile["sources"].values()) - 1) < 1e-12
        for domain, expected in profile["domains"].items():
            actual = sum(
                weight
                for source, weight in profile["sources"].items()
                if source_domains[source] == domain
            )
            assert abs(actual - expected) < 1e-12


def test_loader_aligned_shards_preserve_one_logical_stream(tmp_path: Path) -> None:
    batch_tokens = 8
    total_steps = 5
    logical = np.arange(total_steps * batch_tokens + 1, dtype=np.uint16)
    writer = LoaderAlignedShardWriter(
        tmp_path,
        batch_tokens=batch_tokens,
        total_steps=total_steps,
        steps_per_shard=2,
    )
    writer.append(logical[:7], "a")
    writer.append(logical[7:31], "b")
    writer.append(logical[31:], "c")
    writer.finish()

    shards = [read_payload(path) for path in sorted(tmp_path.glob("*.bin"))]
    assert [len(shard) for shard in shards] == [17, 17, 9]
    assert all(
        sum(shard["source_tokens"].values()) == shard["tokens"]
        for shard in writer.shards
    )
    assert sum((len(shard) - 1) // batch_tokens for shard in shards) == total_steps
    assert all(left[-1] == right[0] for left, right in zip(shards, shards[1:]))
    reconstructed = np.concatenate(
        [shards[0], *(shard[1:] for shard in shards[1:])]
    )
    np.testing.assert_array_equal(reconstructed, logical)


def test_deduplicator_rejects_exact_formatting_and_registered_problems() -> None:
    protected = "Solve 2 + 2."
    dedup = DocumentDeduplicator(ProblemGuard.from_problems([protected]))
    registered = RawDocument("math", "math", (protected,), (protected,))
    assert not dedup.accept(registered)

    first = RawDocument("web", "web", ("Hello, WORLD!",), ("Hello, WORLD!",))
    formatting_duplicate = RawDocument(
        "knowledge", "knowledge", ("hello world",), ("hello world",)
    )
    assert dedup.accept(first)
    assert not dedup.accept(formatting_duplicate)
    assert dedup.counts["registry_exact"] == 1
    assert dedup.counts["formatting_duplicate"] == 1


def packed_document(*questions: str) -> RawDocument:
    """A DeepMind-shaped document: one segment and one key per QA pair."""
    return RawDocument(
        source="deepmind_math",
        domain="math",
        segments=tuple(f"{question}\nAnswer: 4" for question in questions),
        quality_keys=questions,
        qa=True,
    )


def test_a_packed_qa_document_loses_only_its_contaminated_items() -> None:
    """Fifteen clean pairs should not follow one held-out problem out."""
    guard = ProblemGuard.from_problems(["Solve 2 + 2."])
    counts = Counter()
    document = packed_document("What is 1 + 1?", "Solve 2 + 2.", "What is 3 + 3?")
    trimmed = builder.decontaminated(document, guard, counts)
    assert trimmed.segments == (
        "What is 1 + 1?\nAnswer: 4",
        "What is 3 + 3?\nAnswer: 4",
    )
    assert trimmed.quality_keys == ("What is 1 + 1?", "What is 3 + 3?")
    assert counts["deepmind_math:registry_items_dropped"] == 1
    assert "deepmind_math:registry_all_items_dropped" not in counts


def test_a_wholly_contaminated_packed_document_is_still_refused() -> None:
    guard = ProblemGuard.from_problems(["Solve 2 + 2.", "What is 1 + 1?"])
    counts = Counter()
    document = packed_document("What is 1 + 1?", "Solve 2 + 2.")
    assert builder.decontaminated(document, guard, counts) is None
    assert counts["deepmind_math:registry_items_dropped"] == 2
    assert counts["deepmind_math:registry_all_items_dropped"] == 1


def test_clean_and_unpackable_documents_pass_through_untouched() -> None:
    """Only item-aligned QA sources have anything to trim."""
    guard = ProblemGuard.from_problems(["Solve 2 + 2."])
    counts = Counter()
    clean = packed_document("What is 1 + 1?", "What is 3 + 3?")
    assert builder.decontaminated(clean, guard, counts) is clean
    web = RawDocument("fineweb", "web", ("Solve 2 + 2.", "and more"), ())
    assert builder.decontaminated(web, guard, counts) is web
    single = packed_document("Solve 2 + 2.")
    assert builder.decontaminated(single, guard, counts) is single
    assert not counts
    # Falling through is not admitting: whatever a source does not declare as
    # items, the whole-document test still weighs.
    assert DocumentDeduplicator(guard).rejection(single) == "registry_exact"
    page = RawDocument("fineweb", "web", ("Solve 2 + 2.",), ())
    assert DocumentDeduplicator(guard).rejection(page) == "registry_exact"


def test_code_and_math_operators_are_not_formatting_deduplicated() -> None:
    dedup = DocumentDeduplicator(ProblemGuard.from_problems([]))
    plus = RawDocument("code", "code", ("return x + y" * 20,), ("return x + y",))
    minus = RawDocument("code", "code", ("return x - y" * 20,), ("return x - y",))
    assert dedup.accept(plus)
    assert dedup.accept(minus)


def test_token_split_preserves_content_and_inserts_boundaries() -> None:
    tokens = np.arange(11, dtype=np.int32)
    tokens[0] = 50256
    chunks = list(split_token_document(tokens, max_document_tokens=5))
    assert [chunk.size for chunk in chunks] == [5, 5, 3]
    assert all(chunk[0] == 50256 for chunk in chunks)
    reconstructed = np.concatenate(
        [chunks[0], *(chunk[1:] for chunk in chunks[1:])]
    )
    np.testing.assert_array_equal(reconstructed, tokens)


@pytest.mark.parametrize(
    "character,prefix",
    [
        (character, 4 - consumed)
        for character in ("¢", "ह", "🙂")
        for consumed in range(1, len(character.encode("utf-8")))
    ],
)
def test_byte_token_split_never_bisects_utf8(
    character: str, prefix: int
) -> None:
    text = "a" * prefix + character + "z" * 12
    tokens = np.asarray([256, *text.encode("utf-8")], dtype=np.int32)

    chunks = list(
        split_token_document(
            tokens,
            max_document_tokens=5,
            eot_id=256,
            utf8_bytes=True,
        )
    )

    for chunk in chunks:
        chunk[1:].astype(np.uint8).tobytes().decode("utf-8", errors="strict")
    reconstructed = b"".join(
        chunk[1:].astype(np.uint8).tobytes() for chunk in chunks
    )
    assert reconstructed == text.encode("utf-8")


def test_paragraph_chunks_preserve_content_and_limit() -> None:
    text = "a" * 8 + "\n\n" + "b" * 8 + "\n\n" + "c" * 20
    chunks = list(paragraph_chunks(text, max_chars=10))
    assert chunks == ["a" * 8, "b" * 8, "c" * 10, "c" * 10]
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_logical_interleave_hits_exact_source_budgets() -> None:
    sources = {
        "a": iter([np.arange(7), np.arange(7, 14)]),
        "b": iter([np.arange(100, 106)]),
    }
    budgets = {"a": 10, "b": 4}
    emitted = list(logical_interleave(sources, budgets))
    totals = {
        name: sum(tokens.size for source, tokens in emitted if source == name)
        for name in sources
    }
    assert totals == budgets
    assert sum(tokens.size for _, tokens in emitted) == sum(budgets.values())


# -- staged (anneal) corpus assembly --------------------------------------


def _stream(name: str, count: int):
    return iter(np.full((1,), ord(name[0]), dtype=np.uint16) for _ in range(count))


def test_stage_budgets_without_an_anneal_is_a_single_stage() -> None:
    stages = builder.stage_budgets(100, {"a": 0.6, "b": 0.4}, None, 0.15)
    assert stages == [("bulk", {"a": 60, "b": 40})]


def test_stage_budgets_splits_the_total_exactly() -> None:
    bulk = {"a": 0.5, "b": 0.5}
    anneal = {"a": 0.1, "b": 0.9}
    stages = builder.stage_budgets(1000, bulk, anneal, 0.2)
    assert [name for name, _ in stages] == ["bulk", "anneal"]
    assert sum(sum(stage.values()) for _, stage in stages) == 1000
    assert sum(stages[1][1].values()) == 200
    assert stages[1][1] == {"a": 20, "b": 180}


def test_a_source_excluded_from_one_stage_keeps_an_explicit_zero() -> None:
    stages = builder.stage_budgets(
        1000, {"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, 0.25
    )
    assert stages == [("bulk", {"a": 750, "b": 0}), ("anneal", {"a": 0, "b": 250})]


def test_anneal_and_bulk_profiles_must_name_the_same_sources() -> None:
    with pytest.raises(ValueError, match="only one"):
        builder.stage_budgets(100, {"a": 1.0}, {"b": 1.0}, 0.1)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_anneal_fraction_outside_the_open_unit_interval_is_refused(fraction) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        builder.stage_budgets(100, {"a": 1.0}, {"a": 1.0}, fraction)


def test_staged_interleave_does_not_restart_its_sources() -> None:
    """The anneal continues each source, so no document is used twice.

    This is the property that makes a corpus-side anneal safe: the alternative
    -- rebuilding the stream from the top under a second profile -- would
    quietly turn the last stretch of a one-pass corpus into a second epoch of
    its most-weighted sources.
    """
    sources = {"a": _stream("a", 10), "b": _stream("b", 10)}
    stages = [("bulk", {"a": 4, "b": 4}), ("anneal", {"a": 1, "b": 5})]
    seen = list(builder.staged_interleave(sources, stages))
    assert [name for name, _, _ in seen].count("bulk") == 8
    assert sum(tokens.size for name, _, tokens in seen if name == "anneal") == 6
    # Ten documents existed per source and nine were consumed in total for
    # source "b"; a restart would have needed only five and left five unread.
    assert sum(1 for _ in sources["b"]) == 1


def test_staged_interleave_skips_sources_with_no_budget_in_a_stage() -> None:
    sources = {"a": _stream("a", 4), "b": _stream("b", 4)}
    stages = [("bulk", {"a": 2, "b": 0}), ("anneal", {"a": 0, "b": 3})]
    seen = [(stage, source) for stage, source, _ in builder.staged_interleave(sources, stages)]
    assert seen == [("bulk", "a"), ("bulk", "a")] + [("anneal", "b")] * 3


def test_the_math30_profile_pairs_with_its_anneal() -> None:
    bulk = json.loads(Path("pretraining/k3_weights_math30.json").read_text())
    anneal = json.loads(Path("pretraining/k3_weights_math30_anneal.json").read_text())
    assert set(bulk["sources"]) == set(anneal["sources"])
    assert anneal["domains"]["math"] > bulk["domains"]["math"]
    stages = builder.stage_budgets(
        1_000_000, bulk["sources"], anneal["sources"], 0.15
    )
    assert sum(sum(stage.values()) for _, stage in stages) == 1_000_000


# -- DeepMind module weighting --------------------------------------------


def _module_dir(tmp_path: Path, sizes: dict[str, int]) -> Path:
    for module, count in sizes.items():
        lines = []
        for index in range(count):
            lines += [f"{module} question {index}", f"{module} answer {index}"]
        (tmp_path / f"{module}.txt").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_weighted_deepmind_pairs_tracks_the_declared_ratio(tmp_path) -> None:
    directory = _module_dir(tmp_path, {"arith": 400, "poly": 400})
    pairs = list(
        builder.weighted_deepmind_pairs(
            directory, ["arith", "poly"], {"arith": 4.0, "poly": 1.0}
        )
    )
    counts = Counter(question.split()[0] for question, _ in pairs)
    assert counts["arith"] + counts["poly"] == 800
    # Both files run dry, so the ratio holds only while both are live; check
    # the weighted prefix rather than the tail the shorter module cannot fill.
    prefix = Counter(question.split()[0] for question, _ in pairs[:500])
    assert abs(prefix["arith"] / 500 - 0.8) < 0.01


def test_unweighted_deepmind_modules_default_to_one(tmp_path) -> None:
    directory = _module_dir(tmp_path, {"arith": 100, "poly": 100})
    pairs = list(
        builder.weighted_deepmind_pairs(directory, ["arith", "poly"], {"arith": 3.0})
    )
    prefix = Counter(question.split()[0] for question, _ in pairs[:100])
    assert abs(prefix["arith"] / 100 - 0.75) < 0.02


def test_a_weight_for_an_absent_module_is_an_error(tmp_path) -> None:
    directory = _module_dir(tmp_path, {"arith": 4})
    with pytest.raises(ValueError, match="not under"):
        list(builder.weighted_deepmind_pairs(directory, ["arith"], {"ghost": 2.0}))


def test_nonpositive_deepmind_weights_are_refused(tmp_path) -> None:
    directory = _module_dir(tmp_path, {"arith": 4, "poly": 4})
    with pytest.raises(ValueError, match="positive"):
        list(
            builder.weighted_deepmind_pairs(
                directory, ["arith", "poly"], {"poly": 0.0}
            )
        )


def test_declared_deepmind_module_weights_exist_on_disk() -> None:
    manifest = json.loads(Path("pretraining/k3_sources.json").read_text())
    source = next(s for s in manifest["sources"] if s["name"] == "deepmind_math")
    directory = Path(source["path"])
    if not directory.exists():
        pytest.skip(f"{directory} is not present")
    modules = {path.stem for path in directory.glob("*.txt")}
    assert set(source["module_weights"]) <= modules
    assert all(weight > 0 for weight in source["module_weights"].values())


def test_a_stage_boundary_does_not_destroy_the_partial_document() -> None:
    """The anneal must resume the document the bulk stage cut, not skip it.

    Sources are drawn to an exact token budget, so the document that overshoots
    a stage's budget gets truncated. If its tail were dropped, the next stage
    would be short by that much through no fault of the data, and the build
    would either fail at the finish line or quietly redistribute around a
    shortfall that does not exist.
    """
    documents = [np.arange(10, dtype=np.uint16), np.arange(10, 20, dtype=np.uint16)]
    sources = {"a": iter(documents)}
    stages = [("bulk", {"a": 7}), ("anneal", {"a": 13})]
    seen = list(builder.staged_interleave(sources, stages))
    assert sum(tokens.size for _, _, tokens in seen) == 20
    assert np.array_equal(
        np.concatenate([tokens for _, _, tokens in seen]), np.arange(20)
    )
    assert [name for name, _, _ in seen] == ["bulk", "anneal", "anneal"]


def test_byte_stage_seam_reprefixes_tail_and_transfers_utf8_shortfall() -> None:
    boundary = 256
    document = np.asarray(
        [boundary, *"a🙂b".encode("utf-8")], dtype=np.int32
    )
    stats = Counter()
    seen = list(
        builder.staged_interleave(
            {
                "a": iter((document,)),
                "b": iter((np.asarray([boundary, *b"12345"]),)),
            },
            [("bulk", {"a": 4, "b": 4}), ("anneal", {"a": 6, "b": 0})],
            boundary_id=boundary,
            utf8_bytes=True,
            split_stats=stats,
        )
    )

    assert [(stage, source, tokens.size) for stage, source, tokens in seen] == [
        ("bulk", "a", 2),
        ("bulk", "b", 6),
        ("anneal", "a", 6),
    ]
    assert seen[0][2].tolist() == [boundary, ord("a")]
    assert seen[2][2].tolist() == [boundary, *"🙂b".encode("utf-8")]
    assert stats == Counter(
        {
            "utf8_boundary_underfill": 2,
            "utf8_budget_transfers": 1,
            "reprefixed_tails": 1,
        }
    )


def test_source_stream_draws_the_pushed_back_tail_before_anything_new() -> None:
    stream = builder.SourceStream(
        iter([np.arange(3, dtype=np.uint16), np.arange(3, 6, dtype=np.uint16)])
    )
    assert stream.draw().tolist() == [0, 1, 2]
    stream.pushback(np.array([1, 2], dtype=np.uint16))
    with pytest.raises(RuntimeError, match="pending remainder"):
        stream.pushback(np.array([9], dtype=np.uint16))
    assert stream.draw().tolist() == [1, 2]
    assert stream.draw().tolist() == [3, 4, 5]
    # Exhaustion is a sentinel, not StopIteration: the interleaver has to be
    # able to tell a spent source from an empty draw and redistribute.
    assert stream.draw() is None
    assert stream.draw() is None


def test_a_pushed_back_tail_is_not_lost_by_a_single_stage_either() -> None:
    stream = builder.SourceStream(iter([np.arange(5, dtype=np.uint16)]))
    written = list(builder.logical_interleave({"a": stream}, {"a": 3}))
    assert written[0][1].tolist() == [0, 1, 2]
    assert stream.pending.tolist() == [3, 4]


def test_item_and_document_rejections_are_reported_in_separate_maps() -> None:
    """They count different things and a manifest must not invite adding them.

    At 8-24 items per DeepMind document, summing the two would overstate the
    documents refused by an order of magnitude for exactly one key.
    """
    counts = Counter(
        {
            "deepmind_math:registry_items_dropped": 228,
            "deepmind_math:registry_all_items_dropped": 0,
            "deepmind_math:registry_ngram": 3,
            "fineweb:too_short": 17,
        }
    )
    documents, items = builder.partition_rejection_counts(counts)
    assert items == {"deepmind_math:registry_items_dropped": 228}
    assert documents == {
        "deepmind_math:registry_all_items_dropped": 0,
        "deepmind_math:registry_ngram": 3,
        "fineweb:too_short": 17,
    }
    assert sum(documents.values()) + sum(items.values()) == sum(counts.values())

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.eval_generated_external_bpb import (
    RESULT_SCHEMA,
    continuation_cross_entropy,
    evaluator_source_provenance,
    generation_cohort,
    generated_rows,
    repetition_fraction,
    score_report,
    validate_comparison_contract,
)


def test_generated_rows_flattens_every_seed_ledger() -> None:
    payload = {
        "records": [
            {"rows": [{"raw_hex": "61"}, {"raw_hex": "62"}]},
            {"rows": [{"raw_hex": "63"}]},
        ]
    }
    assert tuple(row["raw_hex"] for row in generated_rows(payload)) == (
        "61",
        "62",
        "63",
    )


def test_repetition_fraction_counts_duplicate_fourgrams() -> None:
    assert repetition_fraction(b"abc") == 0.0
    assert repetition_fraction(b"abcdefgh") == 0.0
    assert repetition_fraction(b"aaaaa") == pytest.approx(0.5)


def test_generated_rows_rejects_missing_raw_bytes() -> None:
    with pytest.raises(ValueError, match="raw_hex"):
        generated_rows({"records": [{"rows": [{"text": "not authoritative"}]}]})


def test_generated_rows_normalizes_legacy_nanogpt_samples() -> None:
    rows = generated_rows(
        {
            "results": [
                {
                    "samples": [
                        {
                            "answer": "café",
                            "raw_generation": "café\nAnswer:",
                            "gold": "4",
                        }
                    ]
                }
            ]
        }
    )
    assert bytes.fromhex(str(rows[0]["raw_hex"])) == "café".encode()
    assert rows[0]["invalid_utf8"] is False


def _artifact(*, legacy: bool, seed: int = 0) -> dict[str, object]:
    common: dict[str, object] = {
        "gsm8k_train_sha256": "train",
        "gsm8k_test_sha256": "test",
        "prompt_format": "harness",
        "strip_calculator_annotations": True,
        "max_new_bytes": 512,
    }
    ledger: dict[str, object] = {
        "shots": 5,
        "seed": seed,
        "examples": 1_319,
        "exemplar_train_rows": [1, 2, 3, 4, 5],
    }
    if legacy:
        ledger["samples"] = [{"answer": "4", "raw_generation": "4\n\n"}]
        common["results"] = [ledger]
    else:
        ledger["serialized_sample_count"] = 1
        ledger["rows"] = [{"test_row": 0, "raw_hex": "34"}]
        common["records"] = [ledger]
    return common


def test_generation_comparison_authenticates_cross_format_cohort() -> None:
    duo, nano = _artifact(legacy=False), _artifact(legacy=True)
    validate_comparison_contract((duo, nano))
    assert generation_cohort(duo) == generation_cohort(nano)


def test_generation_comparison_rejects_mismatched_seed() -> None:
    with pytest.raises(ValueError, match="cohorts"):
        validate_comparison_contract(
            (_artifact(legacy=False), _artifact(legacy=True, seed=1))
        )


def test_continuation_cross_entropy_never_indexes_storage_pad() -> None:
    logits = torch.zeros((2, 2, 3))
    # ID 3 is deliberately outside the output support and must be ignored.
    targets = torch.tensor([[0, 3], [2, 3]])
    valid = torch.tensor([[True, False], [True, False]])
    assert continuation_cross_entropy(logits, targets, valid) == pytest.approx(
        2 * np.log(3)
    )


def test_score_report_counts_nonempty_responses_and_eot_normalization(
    tmp_path: Path,
) -> None:
    generation = tmp_path / "generation.json"
    generation.write_text("{}")
    rows = (
        {"raw_hex": "", "invalid_utf8": False},
        {"raw_hex": "61", "invalid_utf8": True},
    )
    report = score_report(
        generation,
        {"decode_mode": "duo", "diffusion_steps": 48},
        rows,
        (b"", b"a"),
        nll=2 * np.log(2),
        byte_count=1,
    )

    assert report["responses"] == 2
    assert report["nonempty_responses"] == 1
    assert report["empty_response_rate"] == pytest.approx(0.5)
    assert report["scored_terminal_eot_atoms"] == 2
    assert report["external_causal_byte_bpb"] == pytest.approx(2.0)


def test_evaluator_provenance_binds_current_source() -> None:
    provenance = evaluator_source_provenance()
    path = Path(str(provenance["path"]))

    assert RESULT_SCHEMA.endswith("/v2")
    assert provenance["schema"] == "external_causal_byte_generation_evaluator_source/v1"
    assert provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

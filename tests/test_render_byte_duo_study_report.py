from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pytest

from scripts.render_byte_duo_study_report import (
    matched_float_quantization_record,
    load_authenticated_evidence,
    parse_named_paths,
    parse_runs,
    response_preview,
    validate_artifact_identity,
    validate_external_report,
    validate_geometry_evidence,
    validate_trace,
)


def test_parse_runs_requires_named_unique_directories() -> None:
    assert parse_runs(["control=a", "phase=b"]) == [
        ("control", Path("a")),
        ("phase", Path("b")),
    ]
    with pytest.raises(ValueError, match="unique"):
        parse_runs(["control=a", "control=b"])


def test_parse_named_paths_reports_the_option_with_bad_input() -> None:
    assert parse_named_paths(["control=artifact.bdg"], "--artifact") == [
        ("control", Path("artifact.bdg"))
    ]
    with pytest.raises(ValueError, match="--topology"):
        parse_named_paths(["missing-separator"], "--topology")


def test_response_preview_escapes_byte_and_atomic_controls() -> None:
    assert response_preview([104, 105, 256, 10, 257]) == (
        "hi<|endoftext|>\\x0a<think>"
    )


def _complete_trace(checkpoint_sha256: str) -> dict[str, object]:
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "diffusion_steps": 2,
        "evaluator_source": {"sha256": "source"},
        "records": [
            {
                "rows": [
                    {
                        "diffusion_trace": {
                            "diffusion_blocks": [
                                {
                                    "prompt_atoms_noised": False,
                                    "initial_ids": [65, 9],
                                    "revisable_positions": [1],
                                    "non_revisable_positions": [0],
                                    "steps": [
                                        {
                                            "step": 1,
                                            "transition_kind": "reverse_grid_posterior",
                                            "time_t": 1.0,
                                            "time_s": 0.5,
                                            "input_ids": [65, 9],
                                            "output_ids": [65, 8],
                                        },
                                        {
                                            "step": 2,
                                            "transition_kind": "reverse_grid_posterior",
                                            "time_t": 0.5,
                                            "time_s": 0.1,
                                            "input_ids": [65, 8],
                                            "output_ids": [65, 7],
                                        },
                                        {
                                            "step": 3,
                                            "transition_kind": "exact_residual_noise_cleanup",
                                            "time_t": 0.1,
                                            "time_s": 0.0,
                                            "input_ids": [65, 7],
                                            "output_ids": [65, 6],
                                        },
                                    ],
                                }
                            ]
                        }
                    }
                ]
            }
        ],
    }


def test_trace_validation_checks_completeness_chaining_and_fixed_positions(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    trace = _complete_trace(digest)
    validate_trace(trace, tmp_path)

    trace["records"][0]["rows"][0]["diffusion_trace"]["diffusion_blocks"][0][
        "steps"
    ][1]["input_ids"] = [66, 8]
    with pytest.raises(ValueError, match="state-chained"):
        validate_trace(trace, tmp_path)


def test_external_score_binds_run_checkpoint_peer_generation_and_scorer(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    gsm_path = tmp_path / "duo.json"
    peer_path = tmp_path / "nano.json"
    scorer_path = tmp_path / "scorer.pt"
    gsm_path.write_text("{}")
    peer_path.write_text("{}")
    scorer_path.write_bytes(b"scorer")
    external = {
        "scorer_checkpoint": str(scorer_path),
        "scorer_checkpoint_sha256": hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
        "reports": [
            {
                "generation": str(gsm_path),
                "generation_sha256": hashlib.sha256(gsm_path.read_bytes()).hexdigest(),
            },
            {
                "generation": str(peer_path),
                "generation_sha256": hashlib.sha256(peer_path.read_bytes()).hexdigest(),
            },
        ],
    }

    report, scorer = validate_external_report(
        external, gsm_path, {"checkpoint_sha256": checkpoint_sha}, tmp_path, peer_path
    )
    assert report is external["reports"][0]
    assert scorer == hashlib.sha256(scorer_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="checkpoint"):
        validate_external_report(
            external, gsm_path, {"checkpoint_sha256": "wrong"}, tmp_path, peer_path
        )


def test_quantization_delta_uses_exact_target_matched_float_ledger() -> None:
    matched = {
        "ledger_seed": 11_337,
        "targets": 939_423,
        "changed_targets": 467_785,
        "conditional_canvas_nelbo_nats_per_atom": 1.8332374965803475,
    }
    nelbo = {
        "records": [
            matched,
            {
                "ledger_seed": 21_337,
                "targets": 937_877,
                "changed_targets": 466_726,
                "conditional_canvas_nelbo_nats_per_atom": 1.8451706620377726,
            },
        ]
    }
    postquant = {"targets": 939_423, "changed_targets": 467_785}

    assert matched_float_quantization_record(nelbo, postquant) is matched


def test_quantization_delta_rejects_unmatched_ledger() -> None:
    with pytest.raises(ValueError, match="unique matched"):
        matched_float_quantization_record(
            {"records": [{"targets": 1, "changed_targets": 1}]},
            {"targets": 2, "changed_targets": 1},
        )


def test_artifact_identity_is_bound_to_the_labeled_run(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    contract = {"dataset_payload_sha256": "data", "source": {"sha256": "source"}}
    metadata = {
        "provenance": {
            "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            "checkpoint_step": 2_000,
            "dataset_payload_sha256": "data",
            "source_sha256": "source",
        }
    }
    validate_artifact_identity(metadata, tmp_path, contract)
    metadata["provenance"]["checkpoint_step"] = 1_999
    with pytest.raises(ValueError, match="labeled run"):
        validate_artifact_identity(metadata, tmp_path, contract)


def test_readiness_evidence_is_hash_authenticated(tmp_path: Path) -> None:
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps({"eligible": True}))
    descriptor = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert load_authenticated_evidence(descriptor, "readiness") == {"eligible": True}
    descriptor["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        load_authenticated_evidence(descriptor, "readiness")


def test_geometry_evidence_is_bound_to_recipe_geometry_and_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"geometry checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source_sha = "source"
    data_sha = "data"
    train_geometry = {"canvas_length": 256, "branches": 15}
    canonical_geometry = {"canvas_length": 512, "branches": 8}
    dataset_geometry = {"required_branch_bytes": 4096, "branch_span_length": 512}
    model = {"schema_version": 5}
    contract = {
        "source": {"sha256": source_sha},
        "dataset_payload_sha256": data_sha,
        "model": model,
        "training": {
            "training_geometry": train_geometry,
            "canonical_validation_geometry": canonical_geometry,
            "dataset_geometry": dataset_geometry,
            "global_batch_size": 249,
            "microbatch_size": 16,
            "validation_batch_size": 64,
            "clean_ar_reduction": "clean reduction",
            "clean_ar_weight": 0.0,
            "combination": "duo objective",
            "diagnostic_cadence": {"validation_every": 20},
            "duo_nelbo_reduction": "atom reduction",
            "objective": "pure_nelbo",
            "schedule_eps": 0.001,
            "time_sampling": "striped",
            "readiness_evidence": {
                "training_validation": {"benchmark_harness_sha256": "harness"},
                "inference": {"benchmark_sha256": "harness", "diffusion_steps": 8},
            },
        },
    }
    result = {
        "script_args": ["--canvas-length", "256", "--branches", "15"],
        "overrides": {
            "BYTE_DUO_EXPECTED_SOURCE_SHA256": source_sha,
            "BYTE_DIFFUSION_EXPECTED_DATA_SHA256": data_sha,
            "BYTE_DUO_MICROBATCH": "16",
            "BYTE_DUO_VALIDATION_BATCH": "64",
        },
    }
    nelbo = {
        "source_sha256": source_sha,
        "dataset_payload_sha256": data_sha,
        "checkpoint_sha256": checkpoint_sha,
        "completed_steps": 2_000,
        "training_geometry": train_geometry,
        "canonical_validation_geometry": canonical_geometry,
        "evaluation_canvas_length": 512,
        "evaluation_branches": 8,
        "evaluation_geometry": "canonical_512x8_headline",
    }
    readiness = {
        "architecture": "duo",
        "duo_canvas_length": 256,
        "duo_branches": 15,
        "dataset_payload_sha256": data_sha,
        "model_config": model,
        "global_batch": 249,
        "selected_microbatch": 16,
        "selected_validation_batch_size": 64,
        "source": {"sha256": source_sha},
        "benchmark_harness_source": {"sha256": "harness"},
        "workload": {
            "canonical_validation_geometry": canonical_geometry,
            "clean_ar_reduction": "clean reduction",
            "clean_ar_weight": 0.0,
            "combination": "duo objective",
            "dataset_geometry": dataset_geometry,
            "diagnostic_cadence": {"validation_every": 20},
            "duo_nelbo_reduction": "atom reduction",
            "objective": "pure_nelbo",
            "schedule_eps": 0.001,
            "time_sampling": "striped",
            "training_geometry": train_geometry,
        },
        "results": {"16": {"eligible": True, "validation": {"eligible": True}}},
    }
    inference = {
        "canvas_length": 256,
        "branches": 15,
        "training_geometry": train_geometry,
        "canonical_validation_geometry": canonical_geometry,
        "dataset_geometry": dataset_geometry,
        "model_config": model,
        "requested_atoms_per_trajectory": 512,
        "eligible": True,
        "diffusion_steps": 8,
        "semantic_generation": False,
        "recipe_source": {"sha256": source_sha},
        "benchmark_source": {"sha256": "harness"},
    }

    validate_geometry_evidence(
        tmp_path, result, contract, nelbo, readiness, inference
    )
    inference["diffusion_steps"] = 7
    with pytest.raises(ValueError, match="geometry evidence identity mismatch"):
        validate_geometry_evidence(
            tmp_path, result, contract, nelbo, readiness, inference
        )

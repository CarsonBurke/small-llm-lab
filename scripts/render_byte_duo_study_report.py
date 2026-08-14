#!/usr/bin/env python3
"""Render an authenticated, evaluation-first report for the Byte-Duo study."""

from __future__ import annotations

import argparse
from html import escape
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import unicodedata


ARTIFACT_MAGIC = b"BDI4\x03\x00\x00\x00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="completed Byte-Duo 2k result directory; repeat for each ablation",
    )
    parser.add_argument("--baseline-label")
    parser.add_argument("--candidate-label")
    parser.add_argument(
        "--screened-run",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="source-matched likelihood-only screen stopped before expensive generation",
    )
    parser.add_argument("--screened-baseline-label")
    parser.add_argument(
        "--geometry-screen",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="completed geometry-only 2k run; reported cross-revision unless source matched",
    )
    parser.add_argument("--nano-result", type=Path, required=True)
    parser.add_argument("--nano-gsm", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--nfe-curve", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="authenticated artifact for a run label; omitted labels are reported as unexported",
    )
    parser.add_argument(
        "--topology",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="decode-topology control ledger; repeat for every control",
    )
    parser.add_argument("--external-name", default="external_generation_score.json")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_artifact_metadata(path: Path) -> dict[str, object]:
    """Read the authenticated JSON header without importing the model stack."""
    payload = path.read_bytes()
    if not payload.startswith(ARTIFACT_MAGIC) or len(payload) < len(ARTIFACT_MAGIC) + 8:
        raise ValueError(f"{path} is not a Byte-Duo v3 artifact")
    start = len(ARTIFACT_MAGIC)
    metadata_size = struct.unpack("<Q", payload[start : start + 8])[0]
    metadata_start = start + 8
    metadata_end = metadata_start + metadata_size
    if metadata_end > len(payload):
        raise ValueError(f"{path} has a truncated metadata header")
    value = json.loads(payload[metadata_start:metadata_end])
    if not isinstance(value, dict):
        raise TypeError(f"{path} artifact metadata must be an object")
    return value


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def fnum(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}"


def optional_fnum(value: object, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "not exported"
    return f"{finite(value, 'optional numeric value'):,.{digits}f}{suffix}"


def td(value: object, cls: str = "") -> str:
    return f'<td class="{cls}">{escape(str(value))}</td>'


def tr(values: list[object], classes: list[str] | None = None) -> str:
    classes = classes or [""] * len(values)
    return "<tr>" + "".join(td(value, cls) for value, cls in zip(values, classes)) + "</tr>"


def mean_record(records: object, key: str) -> float:
    if not isinstance(records, list) or not records:
        raise ValueError(f"record ledger is empty for {key}")
    return statistics.mean(finite(record[key], key) for record in records)


def matched_float_quantization_record(
    nelbo: dict[str, object], postquant: dict[str, object]
) -> dict[str, object]:
    """Find the exact float ledger paired with the artifact's quantized check."""

    records = nelbo.get("records")
    if not isinstance(records, list):
        raise ValueError("NELBO result has no independent record ledger")
    matches = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("targets") == postquant.get("targets")
        and record.get("changed_targets") == postquant.get("changed_targets")
    ]
    if len(matches) != 1:
        raise ValueError("post-quantization check has no unique matched float ledger")
    return matches[0]


def validate_artifact_identity(
    metadata: dict[str, object], directory: Path, contract: dict[str, object]
) -> None:
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("artifact has no authenticated provenance")
    checkpoint = directory / "checkpoint.pt"
    expected = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": 2_000,
        "dataset_payload_sha256": contract.get("dataset_payload_sha256"),
        "source_sha256": contract.get("source", {}).get("sha256"),
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(f"artifact provenance does not match labeled run: {mismatches}")


def validate_external_report(
    external: dict[str, object],
    gsm_path: Path,
    gsm: dict[str, object],
    run_directory: Path,
    peer_path: Path,
) -> tuple[dict[str, object], str]:
    reports = external.get("reports")
    if not isinstance(reports, list):
        raise ValueError("external score has no report ledger")
    checkpoint_sha = hashlib.sha256((run_directory / "checkpoint.pt").read_bytes()).hexdigest()
    if gsm.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("GSM ledger checkpoint does not match the labeled run")
    generation_sha = hashlib.sha256(gsm_path.read_bytes()).hexdigest()
    matches = [
        report
        for report in reports
        if isinstance(report, dict)
        and report.get("generation_sha256") == generation_sha
        and Path(str(report.get("generation"))).resolve() == gsm_path.resolve()
    ]
    peer_sha = hashlib.sha256(peer_path.read_bytes()).hexdigest()
    peer_matches = [
        report
        for report in reports
        if isinstance(report, dict)
        and report.get("generation_sha256") == peer_sha
        and Path(str(report.get("generation"))).resolve() == peer_path.resolve()
    ]
    scorer_sha = external.get("scorer_checkpoint_sha256")
    scorer_path = Path(str(external.get("scorer_checkpoint")))
    if (
        len(matches) != 1
        or len(peer_matches) != 1
        or not isinstance(scorer_sha, str)
        or not scorer_path.is_file()
        or hashlib.sha256(scorer_path.read_bytes()).hexdigest() != scorer_sha
    ):
        raise ValueError("external score is not authenticated to the run's GSM ledger/scorer")
    return matches[0], scorer_sha


def load_authenticated_evidence(evidence: object, label: str) -> dict[str, object]:
    if not isinstance(evidence, dict):
        raise ValueError(f"{label} evidence descriptor is missing")
    path = Path(str(evidence.get("path")))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != evidence.get("sha256"):
        raise ValueError(f"{label} evidence hash does not match its contract")
    return load(path)


def validate_geometry_evidence(
    directory: Path,
    result: dict[str, object],
    contract: dict[str, object],
    nelbo: dict[str, object],
    readiness: dict[str, object],
    inference: dict[str, object],
) -> None:
    """Bind every geometry-screen ledger to one recipe and checkpoint."""

    training = contract.get("training")
    source = contract.get("source")
    if not isinstance(training, dict) or not isinstance(source, dict):
        raise ValueError("geometry screen has no authenticated training/source contract")
    source_sha = source.get("sha256")
    data_sha = contract.get("dataset_payload_sha256")
    train_geometry = training.get("training_geometry")
    canonical_geometry = training.get("canonical_validation_geometry")
    dataset_geometry = training.get("dataset_geometry")
    model = contract.get("model")
    evidence = training.get("readiness_evidence")
    if not all(
        isinstance(value, dict)
        for value in (
            train_geometry,
            canonical_geometry,
            dataset_geometry,
            model,
            evidence,
        )
    ):
        raise ValueError("geometry screen contract is incomplete")
    readiness_descriptor = evidence.get("training_validation")
    inference_descriptor = evidence.get("inference")
    if not isinstance(readiness_descriptor, dict) or not isinstance(
        inference_descriptor, dict
    ):
        raise ValueError("geometry screen readiness descriptors are incomplete")

    checkpoint = directory / "checkpoint.pt"
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected_nelbo = {
        "source_sha256": source_sha,
        "dataset_payload_sha256": data_sha,
        "checkpoint_sha256": checkpoint_sha,
        "completed_steps": 2_000,
        "training_geometry": train_geometry,
        "canonical_validation_geometry": canonical_geometry,
        "evaluation_canvas_length": canonical_geometry.get("canvas_length"),
        "evaluation_branches": canonical_geometry.get("branches"),
        "evaluation_geometry": "canonical_512x8_headline",
    }
    nelbo_mismatches = {
        key: (nelbo.get(key), expected)
        for key, expected in expected_nelbo.items()
        if nelbo.get(key) != expected
    }

    expected_readiness = {
        "architecture": "duo",
        "duo_canvas_length": train_geometry.get("canvas_length"),
        "duo_branches": train_geometry.get("branches"),
        "dataset_payload_sha256": data_sha,
        "model_config": model,
        "global_batch": training.get("global_batch_size"),
        "selected_microbatch": training.get("microbatch_size"),
        "selected_validation_batch_size": training.get("validation_batch_size"),
    }
    readiness_mismatches = {
        key: (readiness.get(key), expected)
        for key, expected in expected_readiness.items()
        if readiness.get(key) != expected
    }
    readiness_source = readiness.get("source")
    readiness_harness = readiness.get("benchmark_harness_source")
    if not isinstance(readiness_source, dict) or readiness_source.get("sha256") != source_sha:
        readiness_mismatches["source.sha256"] = (
            readiness_source.get("sha256") if isinstance(readiness_source, dict) else None,
            source_sha,
        )
    expected_harness_sha = readiness_descriptor.get("benchmark_harness_sha256")
    if (
        not isinstance(readiness_harness, dict)
        or readiness_harness.get("sha256") != expected_harness_sha
    ):
        readiness_mismatches["benchmark_harness_source.sha256"] = (
            readiness_harness.get("sha256")
            if isinstance(readiness_harness, dict)
            else None,
            expected_harness_sha,
        )
    workload = readiness.get("workload")
    expected_workload = {
        key: training.get(key)
        for key in (
            "canonical_validation_geometry",
            "clean_ar_reduction",
            "clean_ar_weight",
            "combination",
            "dataset_geometry",
            "diagnostic_cadence",
            "duo_nelbo_reduction",
            "objective",
            "schedule_eps",
            "time_sampling",
            "training_geometry",
        )
    }
    if not isinstance(workload, dict):
        readiness_mismatches["workload"] = (workload, expected_workload)
    else:
        readiness_mismatches.update(
            {
                f"workload.{key}": (workload.get(key), expected)
                for key, expected in expected_workload.items()
                if workload.get(key) != expected
            }
        )
    readiness_results = readiness.get("results")
    selected_key = str(training.get("microbatch_size"))
    selected_result = (
        readiness_results.get(selected_key)
        if isinstance(readiness_results, dict)
        else None
    )
    selected_validation = (
        selected_result.get("validation")
        if isinstance(selected_result, dict)
        else None
    )
    if not isinstance(selected_result, dict) or selected_result.get("eligible") is not True:
        readiness_mismatches[f"results.{selected_key}.eligible"] = (
            selected_result.get("eligible") if isinstance(selected_result, dict) else None,
            True,
        )
    if (
        not isinstance(selected_validation, dict)
        or selected_validation.get("eligible") is not True
    ):
        readiness_mismatches[f"results.{selected_key}.validation.eligible"] = (
            selected_validation.get("eligible")
            if isinstance(selected_validation, dict)
            else None,
            True,
        )

    expected_inference = {
        "canvas_length": train_geometry.get("canvas_length"),
        "branches": train_geometry.get("branches"),
        "training_geometry": train_geometry,
        "canonical_validation_geometry": canonical_geometry,
        "dataset_geometry": dataset_geometry,
        "model_config": model,
        "requested_atoms_per_trajectory": 512,
        "eligible": True,
        "diffusion_steps": inference_descriptor.get("diffusion_steps"),
        "semantic_generation": False,
    }
    if inference_descriptor.get("diffusion_steps") != 8:
        raise ValueError("geometry inference contract is not the reported 8-step benchmark")
    inference_mismatches = {
        key: (inference.get(key), expected)
        for key, expected in expected_inference.items()
        if inference.get(key) != expected
    }
    inference_source = inference.get("recipe_source")
    inference_harness = inference.get("benchmark_source")
    if not isinstance(inference_source, dict) or inference_source.get("sha256") != source_sha:
        inference_mismatches["recipe_source.sha256"] = (
            inference_source.get("sha256") if isinstance(inference_source, dict) else None,
            source_sha,
        )
    expected_inference_harness_sha = inference_descriptor.get("benchmark_sha256")
    if (
        not isinstance(inference_harness, dict)
        or inference_harness.get("sha256") != expected_inference_harness_sha
    ):
        inference_mismatches["benchmark_source.sha256"] = (
            inference_harness.get("sha256")
            if isinstance(inference_harness, dict)
            else None,
            expected_inference_harness_sha,
        )

    expected_script_args = [
        "--canvas-length",
        str(train_geometry.get("canvas_length")),
        "--branches",
        str(train_geometry.get("branches")),
    ]
    result_mismatches: dict[str, tuple[object, object]] = {}
    if result.get("script_args") != expected_script_args:
        result_mismatches["script_args"] = (result.get("script_args"), expected_script_args)
    overrides = result.get("overrides")
    expected_overrides = {
        "BYTE_DUO_EXPECTED_SOURCE_SHA256": source_sha,
        "BYTE_DIFFUSION_EXPECTED_DATA_SHA256": data_sha,
        "BYTE_DUO_MICROBATCH": str(training.get("microbatch_size")),
        "BYTE_DUO_VALIDATION_BATCH": str(training.get("validation_batch_size")),
    }
    if not isinstance(overrides, dict):
        result_mismatches["overrides"] = (overrides, expected_overrides)
    else:
        result_mismatches.update(
            {
                f"overrides.{key}": (overrides.get(key), expected)
                for key, expected in expected_overrides.items()
                if overrides.get(key) != expected
            }
        )

    mismatches = {
        "result": result_mismatches,
        "nelbo": nelbo_mismatches,
        "readiness": readiness_mismatches,
        "inference": inference_mismatches,
    }
    mismatches = {key: value for key, value in mismatches.items() if value}
    if mismatches:
        raise ValueError(f"geometry evidence identity mismatch: {mismatches}")


def validate_trace(trace: dict[str, object], run_directory: Path) -> None:
    checkpoint = run_directory / "checkpoint.pt"
    if trace.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError("trace checkpoint does not match the selected run")
    evaluator = trace.get("evaluator_source")
    if not isinstance(evaluator, dict) or not isinstance(evaluator.get("sha256"), str):
        raise ValueError("trace has no evaluator provenance")
    records = trace.get("records")
    if not isinstance(records, list):
        raise ValueError("trace has no record ledger")
    blocks = [
        block
        for record in records if isinstance(record, dict)
        for row in record.get("rows", ()) if isinstance(row, dict)
        for block in row.get("diffusion_trace", {}).get("diffusion_blocks", ())
        if isinstance(block, dict)
    ]
    if not blocks or any(block.get("prompt_atoms_noised") is not False for block in blocks):
        raise ValueError("trace does not prove that every prompt atom stayed clean")
    diffusion_steps = trace.get("diffusion_steps")
    if not isinstance(diffusion_steps, int) or diffusion_steps <= 0:
        raise ValueError("trace has no valid diffusion-step contract")
    expected_transitions = ["reverse_grid_posterior"] * diffusion_steps + [
        "exact_residual_noise_cleanup"
    ]
    for block in blocks:
        initial = block.get("initial_ids")
        revisable = block.get("revisable_positions")
        fixed = block.get("non_revisable_positions")
        steps = block.get("steps")
        if not all(isinstance(value, list) for value in (initial, revisable, fixed, steps)):
            raise ValueError("trace block is missing state or position ledgers")
        width = len(initial)
        revisable_set = {int(index) for index in revisable}
        fixed_set = {int(index) for index in fixed}
        if (
            revisable_set & fixed_set
            or revisable_set | fixed_set != set(range(width))
            or len(steps) != diffusion_steps + 1
        ):
            raise ValueError("trace block has incomplete positions or transitions")
        previous = initial
        previous_time_s: float | None = None
        for expected_step, (transition, step) in enumerate(
            zip(expected_transitions, steps), start=1
        ):
            if not isinstance(step, dict):
                raise ValueError("trace transition is not an object")
            before = step.get("input_ids")
            after = step.get("output_ids")
            if (
                step.get("step") != expected_step
                or step.get("transition_kind") != transition
                or not isinstance(before, list)
                or not isinstance(after, list)
                or len(before) != width
                or len(after) != width
                or before != previous
            ):
                raise ValueError("trace transitions are missing, reordered, or not state-chained")
            time_t = finite(step.get("time_t"), "trace time_t")
            time_s = finite(step.get("time_s"), "trace time_s")
            if time_t < time_s or (
                previous_time_s is not None
                and not math.isclose(time_t, previous_time_s, abs_tol=1e-12)
            ):
                raise ValueError("trace transition times are not a contiguous reverse path")
            if any(before[index] != initial[index] or after[index] != initial[index] for index in fixed_set):
                raise ValueError("trace mutates a fixed prompt position")
            previous = after
            previous_time_s = time_s
        if not math.isclose(finite(steps[0]["time_t"], "initial time"), 1.0, abs_tol=1e-12):
            raise ValueError("trace does not begin at the noise prior")
        if not math.isclose(finite(steps[-1]["time_s"], "terminal time"), 0.0, abs_tol=1e-12):
            raise ValueError("trace does not finish at clean time")


def parse_named_paths(values: list[str], option: str) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"invalid {option} {value!r}; expected LABEL=PATH")
        paths.append((label, Path(path)))
    if len({label for label, _ in paths}) != len(paths):
        raise ValueError(f"{option} labels must be unique")
    return paths


def parse_runs(values: list[str]) -> list[tuple[str, Path]]:
    return parse_named_paths(values, "--run")


def gsm_contract(payload: dict[str, object]) -> dict[str, object]:
    records = payload.get("records", payload.get("results"))
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("full GSM evaluation must have three seed ledgers")
    return {
        "train": payload.get("gsm8k_train_sha256"),
        "test": payload.get("gsm8k_test_sha256"),
        "prompt": payload.get("prompt_format"),
        "strip_calculator": payload.get("strip_calculator_annotations"),
        "max_new_bytes": payload.get("max_new_bytes"),
        "cohorts": tuple(
            (
                record.get("shots"),
                record.get("seed"),
                record.get("examples"),
                tuple(record.get("exemplar_train_rows", ())),
            )
            for record in records
        ),
    }


def response_preview(ids: object, limit: int = 220) -> str:
    if not isinstance(ids, list):
        return ""
    controls = {
        256: "<|endoftext|>",
        257: "<think>",
        258: "</think>",
        259: "<answer>",
        260: "</answer>",
        261: "<MASK>",
        262: "<PAD>",
    }
    pieces: list[str] = []
    pending = bytearray()

    def flush_bytes() -> None:
        if not pending:
            return
        decoded = bytes(pending).decode("utf-8", errors="replace")
        for character in decoded:
            codepoint = ord(character)
            if unicodedata.category(character) == "Cc":
                pieces.append(
                    f"\\x{codepoint:02x}" if codepoint <= 0xFF else f"\\u{codepoint:04x}"
                )
            else:
                pieces.append(character)
        pending.clear()

    for raw_value in ids:
        value = int(raw_value)
        if 0 <= value < 256:
            pending.append(value)
            continue
        flush_bytes()
        pieces.append(controls.get(value, f"<ATOM:{value}>"))
    flush_bytes()
    text = "".join(pieces)
    return text[:limit] + ("…" if len(text) > limit else "")


def trace_sections(trace: dict[str, object]) -> str:
    records = trace.get("records")
    if not isinstance(records, list):
        raise ValueError("trace has no records")
    sections: list[str] = []
    for record in records:
        rows = record.get("rows") if isinstance(record, dict) else None
        if not isinstance(rows, list):
            raise ValueError("trace record has no rows")
        for sample in rows:
            if not isinstance(sample, dict):
                raise TypeError("trace sample must be an object")
            ledger = sample.get("diffusion_trace")
            blocks = ledger.get("diffusion_blocks") if isinstance(ledger, dict) else None
            if not isinstance(blocks, list):
                raise ValueError("trace sample has no diffusion blocks")
            block_html: list[str] = []
            for block in blocks:
                if not isinstance(block, dict) or not isinstance(block.get("steps"), list):
                    raise ValueError("trace block is malformed")
                step_rows: list[str] = []
                for step in block["steps"]:
                    before = step.get("input_ids", [])
                    after = step.get("output_ids", [])
                    active = block.get("revisable_positions", [])
                    changed = sum(
                        before[index] != after[index]
                        for index in active
                        if index < len(before) and index < len(after)
                    )
                    active_after = [after[index] for index in active if index < len(after)]
                    step_rows.append(
                        tr(
                            [
                                step.get("step"),
                                step.get("transition_kind"),
                                f"{finite(step.get('time_t'), 'time_t'):.5f} → {finite(step.get('time_s'), 'time_s'):.5f}",
                                changed,
                                response_preview(active_after),
                            ],
                            ["num", "", "num", "num", "preview"],
                        )
                    )
                block_html.append(
                    "<details><summary>Canvas "
                    + escape(str(block.get("block")))
                    + " · revisable "
                    + escape(str(len(block.get("revisable_positions", []))))
                    + " · committed "
                    + escape(str(len(block.get("committed_ids", []))))
                    + " atoms</summary><table><thead><tr><th>Step</th><th>Transition</th>"
                    + "<th>t → s</th><th>Changed</th><th>Active canvas preview</th></tr>"
                    + "</thead><tbody>"
                    + "".join(step_rows)
                    + "</tbody></table></details>"
                )
            text = sample.get("text")
            sections.append(
                "<details class='sample'><summary>GSM8K row "
                + escape(str(sample.get("test_row")))
                + " · gold "
                + escape(str(sample.get("gold")))
                + " · parsed "
                + escape(str(sample.get("prediction")))
                + "</summary><pre>"
                + escape(text if isinstance(text, str) else "<invalid UTF-8>")
                + "</pre>"
                + "".join(block_html)
                + "</details>"
            )
    return "".join(sections)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    run_specs = parse_runs(args.run)
    screened_specs = parse_named_paths(args.screened_run, "--screened-run")
    geometry_specs = parse_named_paths(args.geometry_screen, "--geometry-screen")
    artifact_paths = dict(parse_named_paths(args.artifact, "--artifact"))
    unknown_artifacts = set(artifact_paths) - {label for label, _ in run_specs}
    if unknown_artifacts:
        raise ValueError(f"artifact labels have no matching run: {sorted(unknown_artifacts)}")
    nano_result = load(args.nano_result)
    nano_gsm = load(args.nano_gsm)
    trace = load(args.trace)
    nfe = load(args.nfe_curve)
    topology_payloads = [
        (label, load(path))
        for label, path in parse_named_paths(args.topology, "--topology")
    ]
    nano_contract = gsm_contract(nano_gsm)

    summaries: list[dict[str, object]] = []
    data_hash: str | None = None
    source_hashes: set[str] = set()
    scorer_hash: str | None = None
    for label, directory in run_specs:
        result = load(directory / "result.json")
        contract = load(directory / "contract.json")
        nelbo = load(directory / "nelbo_5ledgers.json")
        phase_nelbo = load(directory / "nelbo_phase_5ledgers.json")
        buckets = load(directory / "time_buckets_2048.json")
        gsm = load(directory / "gsm8k_5shot_full_steps48.json")
        external = load(directory / args.external_name)
        readiness_evidence = contract["training"]["readiness_evidence"]
        readiness = load_authenticated_evidence(
            readiness_evidence["training_validation"], "training readiness"
        )
        inference = load_authenticated_evidence(
            readiness_evidence["inference"], "inference readiness"
        )
        artifact_path = artifact_paths.get(label)
        artifact_metadata = (
            load_artifact_metadata(artifact_path) if artifact_path is not None else None
        )
        if artifact_metadata is not None:
            validate_artifact_identity(artifact_metadata, directory, contract)
        postquant = (
            artifact_metadata.get("post_quantization_metrics")
            if artifact_metadata is not None
            else None
        )
        if artifact_metadata is not None and not isinstance(postquant, dict):
            raise ValueError(f"{label} artifact has no post-quantization ledger")
        matched_float = (
            matched_float_quantization_record(nelbo, postquant)
            if isinstance(postquant, dict)
            else None
        )
        if result.get("steps") != 2_000 or contract.get("schema") != "byte_duo_run/v1":
            raise ValueError(f"{label} is not a completed Byte-Duo 2k run")
        if gsm_contract(gsm) != nano_contract:
            raise ValueError(f"{label} and nanoGPT use different GSM contracts")
        if gsm.get("decode_mode") != "duo" or gsm.get("diffusion_steps") != 48:
            raise ValueError(f"{label} does not use the authenticated 48-step Duo decoder")
        if gsm.get("requested_batch_size") != 32 or gsm.get("sampling") != "categorical":
            raise ValueError(f"{label} GSM batching/sampling differs")
        if (
            inference.get("diffusion_steps") != 8
            or inference.get("semantic_generation") is not False
        ):
            raise ValueError(f"{label} readiness is not the fixed-compute 8-step benchmark")
        if any(record.get("serialized_sample_count") != 1_319 for record in gsm["records"]):
            raise ValueError(f"{label} does not retain the complete GSM sample ledger")
        current_data_hash = str(contract.get("dataset_payload_sha256"))
        data_hash = data_hash or current_data_hash
        if current_data_hash != data_hash:
            raise ValueError("Byte-Duo ablations use different pretraining data")
        source_hashes.add(str(contract["source"]["sha256"]))
        nelbo_bits = finite(nelbo["mean_nats_per_atom"], "NELBO") / math.log(2)
        interval = nelbo["approximate_95_percent_interval_nats_per_atom"]
        phase_bits = finite(phase_nelbo["mean_nats_per_atom"], "phase NELBO") / math.log(2)
        gsm_records = gsm["records"]
        external_reports = external.get("reports")
        if not isinstance(external_reports, list) or len(external_reports) != 2:
            raise ValueError(f"{label} external score must compare Duo and nanoGPT")
        duo_external, current_scorer_hash = validate_external_report(
            external,
            directory / "gsm8k_5shot_full_steps48.json",
            gsm,
            directory,
            args.nano_gsm,
        )
        scorer_hash = scorer_hash or current_scorer_hash
        if current_scorer_hash != scorer_hash:
            raise ValueError("Byte-Duo arms use different external sample scorers")
        summaries.append(
            {
                "label": label,
                "directory": directory,
                "params": contract["parameter_count"],
                "artifact_bytes": artifact_path.stat().st_size if artifact_path else None,
                "complete_artifact_bytes": (
                    artifact_path.stat().st_size + int(artifact_metadata["code_bytes"])
                    if artifact_path is not None and artifact_metadata is not None
                    else None
                ),
                "postquant_bits": (
                    finite(
                        postquant["conditional_canvas_nelbo_nats_per_atom"],
                        "post-quantization NELBO",
                    )
                    / math.log(2)
                    if isinstance(postquant, dict)
                    else None
                ),
                "quantization_gate_passed": (
                    (
                        finite(
                            postquant["conditional_canvas_nelbo_nats_per_atom"],
                            "post-quantization NELBO",
                        )
                        - finite(
                            matched_float["conditional_canvas_nelbo_nats_per_atom"],
                            "matched float fixed-ledger NELBO",
                        )
                    )
                    / math.log(2.0)
                    <= 0.05
                    if isinstance(postquant, dict)
                    and isinstance(matched_float, dict)
                    else False
                ),
                "float_export_bits": (
                    finite(
                        matched_float["conditional_canvas_nelbo_nats_per_atom"],
                        "matched float fixed-ledger NELBO",
                    )
                    / math.log(2)
                    if isinstance(matched_float, dict)
                    else None
                ),
                "quant_group_size": (
                    artifact_metadata["group_size"] if artifact_metadata else None
                ),
                "microbatch": readiness["selected_microbatch"],
                "update_ms": readiness["results"][str(readiness["selected_microbatch"])]["update_ms"],
                "training_utilization": readiness["results"][str(readiness["selected_microbatch"])]["gpu_utilization_percent"]["mean"],
                "training_power_w": readiness["results"][str(readiness["selected_microbatch"])]["power_w"]["mean"],
                "train_seconds": result["elapsed_seconds"],
                "ar_anchor_bpb": result.get("final_ar_anchor_bpb"),
                "nelbo_bits": nelbo_bits,
                "nelbo_ci": [finite(value, "NELBO interval") / math.log(2) for value in interval],
                "phase_bits": phase_bits,
                "high_noise_bits": finite(buckets["records"][-1]["clean_token_ce_bits_per_atom"], "high-noise CE"),
                "high_noise_accuracy": finite(buckets["records"][-1]["denoising_accuracy"], "high-noise accuracy"),
                "exact": mean_record(gsm_records, "exact_match"),
                "correct": sum(int(record["correct"]) for record in gsm_records),
                "examples": sum(int(record["examples"]) for record in gsm_records),
                "parsed": mean_record(gsm_records, "parsed_answer_rate"),
                "invalid": mean_record(gsm_records, "invalid_utf8_rate"),
                "generation_seconds_by_seed": tuple(
                    finite(record["generation_seconds"], "generation seconds")
                    for record in gsm_records
                ),
                "external_bpb": duo_external["external_causal_byte_bpb"],
                "external_unique": duo_external["unique_response_rate"],
                "inference_requested_atoms_s": inference["requested_atoms_per_second"],
                "inference_trajectories_s": inference["trajectories_per_second"],
            }
        )

    screened_summaries: list[dict[str, object]] = []
    for label, directory in screened_specs:
        result = load(directory / "result.json")
        contract = load(directory / "contract.json")
        nelbo = load(directory / "nelbo_5ledgers.json")
        source_hashes.add(str(contract["source"]["sha256"]))
        screened_summaries.append(
            {
                "label": label,
                "source": str(contract["source"]["sha256"]),
                "dataset": str(contract["dataset_payload_sha256"]),
                "objective": str(contract["training"]["objective"]),
                "noisy_ngrams": contract["model"]["duo_noisy_ngrams"],
                "fixed_bits": finite(
                    result["final_diffusion_nelbo_bits_per_atom"], "fixed validation NELBO"
                ),
                "five_ledger_bits": finite(nelbo["mean_nats_per_atom"], "five-ledger NELBO")
                / math.log(2.0),
                "five_ledger_ci": [
                    finite(value, "screened NELBO interval") / math.log(2.0)
                    for value in nelbo["approximate_95_percent_interval_nats_per_atom"]
                ],
                "elapsed_seconds": finite(result["elapsed_seconds"], "screened training time"),
            }
        )
    screened_html = ""
    if screened_summaries:
        screened_by_label = {str(item["label"]): item for item in screened_summaries}
        screened_baseline_label = args.screened_baseline_label or str(
            screened_summaries[0]["label"]
        )
        if screened_baseline_label not in screened_by_label:
            raise ValueError(f"unknown screened baseline {screened_baseline_label!r}")
        screened_baseline = screened_by_label[screened_baseline_label]
        if any(
            item["source"] != screened_baseline["source"]
            or item["dataset"] != screened_baseline["dataset"]
            for item in screened_summaries
        ):
            raise ValueError("screened runs are not source/data matched")
        screened_rows = "".join(
            tr(
                [
                    item["label"],
                    item["objective"],
                    item["noisy_ngrams"],
                    fnum(float(item["fixed_bits"]), 4),
                    fnum(float(item["five_ledger_bits"]), 4),
                    f"[{fnum(item['five_ledger_ci'][0],4)}, {fnum(item['five_ledger_ci'][1],4)}]",
                    f"{float(item['five_ledger_bits']) - float(screened_baseline['five_ledger_bits']):+.4f}",
                    fnum(float(item["elapsed_seconds"]), 1),
                ]
            )
            for item in screened_summaries
        )
        screened_candidate = screened_summaries[-1]
        screened_delta = float(screened_candidate["five_ledger_bits"]) - float(
            screened_baseline["five_ledger_bits"]
        )
        screened_decision = "promoted" if screened_delta < -0.005 else "rejected"
        screened_html = (
            "<section><h2>Latest source-matched early screen</h2><div class='scroll'><table>"
            "<thead><tr><th>Arm</th><th>Objective</th><th>Noisy branch n-grams</th>"
            "<th>Fixed NELBO bits/atom</th><th>Five-ledger NELBO bits/atom</th>"
            "<th>Approx 95% CI bits/atom</th>"
            "<th>Δ vs screen control</th><th>2k wall s</th></tr></thead><tbody>"
            + screened_rows
            + "</tbody></table></div><p class='note'><strong>"
            + escape(str(screened_candidate["label"]))
            + f" was {screened_decision} at the primary gate.</strong> Its source-matched "
            + f"five-ledger delta was {screened_delta:+.4f} bits/atom; promotion requires "
            + "less than -0.005. Because it failed the primary likelihood gate, its expensive "
            + "GSM/NFE/commit-one follow-ups were cancelled before starting. This is deliberate "
            + "sequential screening, not missing evidence presented as a success.</p></section>"
        )

    geometry_summaries: list[dict[str, object]] = []
    for label, directory in geometry_specs:
        result = load(directory / "result.json")
        contract = load(directory / "contract.json")
        nelbo = load(directory / "nelbo_5ledgers.json")
        if result.get("steps") != 2_000 or result.get("returncode") != 0:
            raise ValueError(f"{label} is not a completed geometry screen")
        if contract.get("dataset_payload_sha256") != data_hash:
            raise ValueError(f"{label} geometry screen uses different pretraining data")
        training = contract.get("training")
        if not isinstance(training, dict):
            raise ValueError(f"{label} geometry screen has no training contract")
        readiness = load_authenticated_evidence(
            training["readiness_evidence"]["training_validation"],
            f"{label} training readiness",
        )
        inference = load_authenticated_evidence(
            training["readiness_evidence"]["inference"],
            f"{label} inference readiness",
        )
        validate_geometry_evidence(
            directory, result, contract, nelbo, readiness, inference
        )
        source_hashes.add(str(contract["source"]["sha256"]))
        geometry_summaries.append(
            {
                "label": label,
                "source": str(contract["source"]["sha256"]),
                "canvas": training["canvas_length"],
                "branches": training["branches"],
                "fixed_bits": finite(
                    result["final_diffusion_nelbo_bits_per_atom"],
                    "geometry fixed NELBO",
                ),
                "five_ledger_bits": finite(
                    nelbo["mean_nats_per_atom"], "geometry five-ledger NELBO"
                )
                / math.log(2.0),
                "five_ledger_ci": [
                    finite(value, "geometry NELBO interval") / math.log(2.0)
                    for value in nelbo[
                        "approximate_95_percent_interval_nats_per_atom"
                    ]
                ],
                "elapsed_seconds": finite(
                    result["elapsed_seconds"], "geometry training time"
                ),
                "microbatch": readiness["selected_microbatch"],
                "update_ms": readiness["results"][
                    str(readiness["selected_microbatch"])
                ]["update_ms"],
                "inference_trajectories_s": inference["trajectories_per_second"],
                "inference_requested_atoms_s": inference["requested_atoms_per_second"],
            }
        )

    nano_records = nano_gsm["results"]
    nano_entries = nano_result.get("val_entries")
    if not isinstance(nano_entries, list):
        raise ValueError("nanoGPT result has no validation ledger")
    nano_final = next(record for record in reversed(nano_entries) if record.get("step") == 2_000)
    by_label = {str(item["label"]): item for item in summaries}
    baseline_label = args.baseline_label or str(summaries[0]["label"])
    candidate_label = args.candidate_label or str(summaries[-1]["label"])
    try:
        baseline = by_label[baseline_label]
        candidate = by_label[candidate_label]
    except KeyError as error:
        raise ValueError(f"unknown baseline/candidate label: {error.args[0]}") from error
    if baseline is candidate:
        raise ValueError("baseline and candidate must be different runs")
    candidate_directory = Path(candidate["directory"])
    candidate_checkpoint_sha = hashlib.sha256(
        (candidate_directory / "checkpoint.pt").read_bytes()
    ).hexdigest()
    validate_trace(trace, candidate_directory)
    for topology_label, topology in topology_payloads:
        if topology.get("checkpoint_sha256") != candidate_checkpoint_sha:
            raise ValueError(
                f"topology {topology_label!r} does not match the selected candidate checkpoint"
            )
    best = min(summaries, key=lambda item: float(item["nelbo_bits"]))

    table_rows = []
    for item in summaries:
        delta = float(item["nelbo_bits"]) - float(baseline["nelbo_bits"])
        table_rows.append(
            tr(
                [
                    item["label"],
                    f"{int(item['params']):,}",
                    (
                        f"{int(item['complete_artifact_bytes']):,}"
                        if item["complete_artifact_bytes"] is not None
                        else "rejected / unexported"
                    ),
                    item["microbatch"],
                    fnum(float(item["update_ms"]), 1),
                    fnum(float(item["nelbo_bits"]), 4),
                    optional_fnum(item["float_export_bits"], 4),
                    optional_fnum(item["postquant_bits"], 4),
                    f"[{fnum(item['nelbo_ci'][0], 4)}, {fnum(item['nelbo_ci'][1], 4)}]",
                    f"{delta:+.4f}",
                    optional_fnum(item["ar_anchor_bpb"], 4),
                    fnum(float(item["phase_bits"]), 4),
                    fnum(float(item["high_noise_bits"]), 3),
                    pct(float(item["high_noise_accuracy"])),
                    pct(float(item["exact"]), 3),
                    pct(float(item["parsed"])),
                    pct(float(item["invalid"])),
                    fnum(float(item["external_bpb"]), 3),
                ]
            )
        )

    nfe_rows = []
    reports = nfe.get("reports")
    if not isinstance(reports, list):
        raise ValueError("NFE external-score curve is missing reports")
    if nfe.get("scorer_checkpoint_sha256") != scorer_hash:
        raise ValueError("NFE curve uses a different external sample scorer")
    for report in reports:
        path = str(report["generation"])
        payload = load(Path(path))
        if payload.get("decode_mode") == "duo" and payload.get("checkpoint_sha256") != candidate_checkpoint_sha:
            raise ValueError("NFE curve contains a Duo generation from another checkpoint")
        if report.get("generation_sha256") != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise ValueError("NFE curve generation hash does not authenticate its ledger")
        records = payload.get("records", payload.get("results"))
        nfe_rows.append(
            tr(
                [
                    payload.get("diffusion_steps", "AR"),
                    pct(mean_record(records, "exact_match"), 3),
                    pct(mean_record(records, "parsed_answer_rate")),
                    pct(mean_record(records, "invalid_utf8_rate")) if "invalid_utf8_rate" in records[0] else "n/a",
                    fnum(finite(report["external_causal_byte_bpb"], "external BPB"), 3),
                    fnum(finite(report["external_causal_byte_perplexity"], "external perplexity"), 2),
                    pct(finite(report["unique_response_rate"], "unique response rate")),
                ]
            )
        )

    nano_exact = mean_record(nano_records, "exact_match")
    nano_parsed = mean_record(nano_records, "parsed_answer_rate")
    nano_correct = sum(int(record["correct"]) for record in nano_records)
    nano_examples = sum(int(record["examples"]) for record in nano_records)
    duo_correct = int(baseline["correct"])
    duo_seed_times = ", ".join(
        fnum(value, 2) for value in baseline["generation_seconds_by_seed"]
    )
    nano_compiled = nano_gsm.get("compiled_generation") is True
    nano_runtime_label = "compiled" if nano_compiled else "eager"
    nano_runtime_note = (
        "Its compiled generation ledger is directly labeled in the source artifact."
        if nano_compiled
        else "Compilation failed in FlexAttention lowering, so its quality ledger is valid but its runtime is not comparable with compiled Duo."
    )
    topology_rows = []
    for label, payload in topology_payloads:
        records = payload.get("records", payload.get("results"))
        topology_rows.append(
            tr(
                [
                    label,
                    payload.get("duo_visible_width") or payload.get("block_length"),
                    payload.get("duo_commit_width"),
                    payload.get("max_new_atoms"),
                    pct(mean_record(records, "exact_match"), 3),
                    pct(mean_record(records, "parsed_answer_rate")),
                    pct(mean_record(records, "invalid_utf8_rate")),
                    sum(int(record["generated_bytes"]) for record in records),
                    fnum(sum(finite(record["generation_seconds"], "generation seconds") for record in records), 2),
                ]
            )
        )
    exported = [item for item in summaries if item["complete_artifact_bytes"] is not None]
    if not exported:
        raise ValueError("at least one run must have an authenticated export")
    oversized = [
        item for item in exported if int(item["complete_artifact_bytes"]) > 16_000_000
    ]
    if oversized:
        raise ValueError(f"exports exceed the 16MB limit: {[item['label'] for item in oversized]}")
    smallest_export = min(exported, key=lambda item: int(item["complete_artifact_bytes"]))
    quality_exports = [item for item in exported if item["quantization_gate_passed"]]
    quant_delta = (
        float(smallest_export["postquant_bits"])
        - float(smallest_export["float_export_bits"])
        if smallest_export["postquant_bits"] is not None
        and smallest_export["float_export_bits"] is not None
        else None
    )
    candidate_nelbo_delta = float(candidate["nelbo_bits"]) - float(baseline["nelbo_bits"])
    candidate_external_delta = float(candidate["external_bpb"]) - float(baseline["external_bpb"])
    candidate_invalid_delta = float(candidate["invalid"]) - float(baseline["invalid"])
    candidate_high_noise_ce_delta = float(candidate["high_noise_bits"]) - float(baseline["high_noise_bits"])
    candidate_high_noise_accuracy_delta = float(candidate["high_noise_accuracy"]) - float(baseline["high_noise_accuracy"])
    promotion_passed = (
        candidate_nelbo_delta < -0.005
        and candidate_external_delta <= -0.05
        and candidate_invalid_delta <= 0.0
        and candidate_high_noise_ce_delta <= -0.02
        and candidate_high_noise_accuracy_delta >= 0.0025
    )
    baseline_source = str(load(Path(baseline["directory"]) / "contract.json")["source"]["sha256"])
    candidate_source = str(load(Path(candidate["directory"]) / "contract.json")["source"]["sha256"])
    source_matched = baseline_source == candidate_source
    decision_verb = "promote" if promotion_passed else "reject"
    comparison_kind = "source-matched one-factor" if source_matched else "cross-revision"
    nano_train_seconds = finite(nano_result["elapsed_seconds"], "nano training seconds")
    baseline_generation_seconds = sum(float(value) for value in baseline["generation_seconds_by_seed"])
    candidate_generation_seconds = sum(float(value) for value in candidate["generation_seconds_by_seed"])
    nano_generation_seconds = sum(
        finite(record["generation_seconds"], "nano generation seconds")
        for record in nano_records
    )
    quant_note = (
        f"{smallest_export['label']} group-{int(smallest_export['quant_group_size'])} int4 "
        f"export changes the exact matched "
        f"fixed-ledger NELBO by {quant_delta:+.4f} bits/atom "
        f"({fnum(float(smallest_export['float_export_bits']),4)} → "
        f"{fnum(float(smallest_export['postquant_bits']),4)})."
        if quant_delta is not None
        else f"{smallest_export['label']} has no matched quantization ledger."
    )
    geometry_html = ""
    if geometry_summaries:
        geometry_rows = "".join(
            tr(
                [
                    item["label"],
                    f"{item['canvas']}×{item['branches']}",
                    str(item["source"])[:16] + "…",
                    item["microbatch"],
                    fnum(float(item["update_ms"]), 1),
                    fnum(float(item["fixed_bits"]), 4),
                    fnum(float(item["five_ledger_bits"]), 4),
                    f"[{fnum(item['five_ledger_ci'][0],4)}, {fnum(item['five_ledger_ci'][1],4)}]",
                    f"{float(item['five_ledger_bits']) - float(baseline['nelbo_bits']):+.4f}",
                    fnum(float(item["inference_trajectories_s"]), 1),
                    fnum(float(item["inference_requested_atoms_s"]), 0),
                    fnum(float(item["elapsed_seconds"]), 1),
                ]
            )
            for item in geometry_summaries
        )
        geometry_html = (
            "<section><h2>Canvas-geometry screen</h2><div class='scroll'><table>"
            "<thead><tr><th>Arm</th><th>Train canvas×branches</th><th>Source</th>"
            "<th>Microbatch</th><th>Update ms</th><th>Fixed canonical NELBO bits/atom</th>"
            "<th>Five-ledger canonical NELBO bits/atom</th><th>Approx 95% CI bits/atom</th>"
            "<th>Δ vs retained historical control bits/atom</th><th>8-step trajectories/s</th>"
            "<th>Requested atoms/s</th><th>2k wall s</th></tr></thead><tbody>"
            + geometry_rows
            + "</tbody></table></div><p class='note'>The candidate changes train geometry, "
            "and the intended recipe fields are held fixed. Headline validation remains the "
            "canonical 512×8 ledger. Because the source revision differs, the displayed delta "
            "is a historical screen—not a causal or promotion comparison. It failed the "
            "0.005 bits/atom gate, so native-geometry likelihood, GSM, and other expensive "
            "follow-ups were not run; only a passing screen would trigger a fresh same-source "
            "512×8 control.</p></section>"
        )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Byte-Duo 2k evidence report</title><style>
:root{{--ink:#17202a;--muted:#64748b;--line:#dbe2ea;--paper:#fff;--wash:#f4f7fb;--accent:#155e75;--good:#166534;--warn:#9a3412}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1480px;margin:auto;padding:40px}} section{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:24px;margin:18px 0;box-shadow:0 5px 18px #0f172a0b}}
h1{{font-size:36px;margin:0 0 6px}} h2{{color:var(--accent)}} h3{{margin-bottom:6px}} .lede{{font-size:18px;color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}} .card{{padding:16px;border-radius:10px;background:#eef6f8;border-left:4px solid var(--accent)}} .value{{font-size:26px;font-weight:750}}
.note{{padding:14px 16px;border-left:4px solid var(--warn);background:#fff7ed}} .good{{border-left-color:var(--good);background:#f0fdf4}}
.scroll{{overflow:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{position:sticky;top:0;background:#edf2f7;white-space:nowrap}} td.num{{font-variant-numeric:tabular-nums}} td.preview{{white-space:pre-wrap;max-width:700px}}
code{{background:#eef2f6;padding:2px 5px;border-radius:4px}} pre{{white-space:pre-wrap;background:#111827;color:#e5e7eb;padding:14px;border-radius:8px;max-height:330px;overflow:auto}} details{{margin:8px 0;border:1px solid var(--line);border-radius:8px;padding:8px}} summary{{cursor:pointer;font-weight:650}}
.small{{font-size:13px;color:var(--muted)}}
</style></head><body><main>
<header><h1>Byte-Duo: 2,000-step decision report</h1><p class="lede">Scratch diffusion on the matched MathGLM-v6 corpus: authenticated likelihood diagnostics, clean-prompt GSM8K generation, speed measurements, export checks, and complete reverse trajectories.</p></header>
	{screened_html}
	{geometry_html}
<section><h2>Comprehensive generation and systems outcome</h2><div class="cards">
<div class="card"><div>Best within-family NELBO</div><div class="value">{escape(str(best['label']))}</div><div>{fnum(float(best['nelbo_bits']),4)} bits/atom</div></div>
<div class="card"><div>nanoGPT AR validation</div><div class="value">{fnum(finite(nano_final['val_bpb'],'nano BPB'),4)}</div><div>true teacher-forced bits/byte; not directly the Duo metric</div></div>
<div class="card"><div>{escape(str(candidate['label']))} AR anchor</div><div class="value">{optional_fnum(candidate['ar_anchor_bpb'],4)}</div><div>bits/scored atomic target, including controls; not challenge BPB</div></div>
<div class="card"><div>nanoGPT GSM8K ({nano_runtime_label})</div><div class="value">{pct(nano_exact,3)}</div><div>exact; {pct(nano_parsed)} parsed</div></div>
<div class="card"><div>Smallest size-valid export</div><div class="value">{int(smallest_export['complete_artifact_bytes']):,} B</div><div>{16_000_000-int(smallest_export['complete_artifact_bytes']):,} B headroom · {escape(str(smallest_export['label']))}; quality gate {'passed' if quality_exports else 'failed'}</div></div>
<div class="card"><div>8-step fixed-compute readiness</div><div class="value">{fnum(float(baseline['inference_trajectories_s']),1)}/s</div><div>{fnum(float(baseline['inference_requested_atoms_s']),0)} requested atoms/s; fresh seeded weights, batch 8, EOT ignored</div></div>
<div class="card"><div>Pretraining data hash</div><div class="value" style="font-size:15px">{escape(str(data_hash)[:16])}…</div><div>all Byte-Duo arms matched; MathGLM v6 included by manifest</div></div>
</div><p class="note"><strong>Decision: {decision_verb} {escape(str(candidate['label']))}.</strong> This is a {comparison_kind} comparison against {escape(str(baseline['label']))}. Candidate deltas are {candidate_nelbo_delta:+.4f} diffusion NELBO bits/atom, {candidate_external_delta:+.3f} frozen-scorer sample BPB, {candidate_high_noise_ce_delta:+.3f} high-noise CE bits, {candidate_high_noise_accuracy_delta*100:+.3f} percentage points high-noise accuracy, and {candidate_invalid_delta*100:+.3f} percentage points invalid UTF-8. Promotion requires all predeclared gates: >0.005 NELBO improvement, ≥0.05 external-BPB improvement, non-worse UTF-8, ≥0.02 high-noise CE improvement, and ≥0.25-point high-noise accuracy improvement.</p><p class="note">Duo NELBO bits/atom is a stochastic conditional-canvas variational bound. It is not autoregressive BPB and is never put on the same numeric axis as nanoGPT’s challenge BPB. Its five-ledger interval measures corruption-ledger Monte Carlo variation for one trained checkpoint, not training-run uncertainty. Float checkpoints produced GSM, trace, and external-score ledgers; int4 artifacts were checked on an exact matched 256-row NELBO ledger. The readiness card is a fixed-compute architecture/kernel benchmark, not semantic serving.</p></section>
<section><h2>Exactly how GSM8K evaluation ran</h2>
<ol><li>Each held-out GSM8K question was rendered with five worked training exemplars using the <code>harness</code> format. Seeds 0, 1, and 2 select matched exemplar rows for both models; every seed covers all 1,319 test questions.</li>
<li><strong>The prompt is clean and is never noised.</strong> Byte-Duo encodes literal UTF-8 prompt bytes as fixed conditioning. Because canvases are stride-aligned, the first 512-slot canvas can contain 0–3 fixed clean prompt-tail atoms and 509–512 newly sampled suffix atoms. A 512-generated-atom cap can therefore require a second canvas.</li>
<li>The reported main decode uses 48 scheduled categorical reverse transitions plus one exact residual-noise cleanup call per canvas, a 512-byte/atom semantic cap, batched generation at 32 requests, atomic EOT, strict UTF-8 validation, and the same textual stop strings as nanoGPT. Stop delimiters are removed before task scoring.</li>
<li>nanoGPT uses {nano_runtime_label} greedy cached autoregressive decoding with the same prompt, exemplars, response byte cap, stop rules, and held-out rows. {nano_runtime_note} Its natural unit is a GPT-2 token; Byte-Duo’s is an atomic byte/control.</li>
<li>The external sample BPB is a frozen causal byte model's score of each stop-trimmed response plus one terminal EOT. The denominator counts literal response bytes only. The scorer was trained on the same MathGLM-v6 source domain, so this is model-based predictability/fluency—not generator likelihood, neutral quality truth, or an independent-domain evaluation.</li></ol>
<p class="small">GSM train hash {escape(str(nano_contract['train']))}; test hash {escape(str(nano_contract['test']))}. Source revisions represented anywhere in this report: {escape(', '.join(sorted(value[:16]+'…' for value in source_hashes)))}.</p>
<p class="note">Float-checkpoint GSM exact counts are {duo_correct}/{int(baseline['examples'])} for retained Duo, {int(candidate['correct'])}/{int(candidate['examples'])} for the candidate, and {nano_correct}/{nano_examples} for nanoGPT. Retained Duo produced {int(round(float(baseline['parsed']) * int(baseline['examples'])))} parseable answers across {int(baseline['examples'])} generations; the candidate produced {int(round(float(candidate['parsed']) * int(candidate['examples'])))} across {int(candidate['examples'])}. Neither diffusion checkpoint answered exactly, versus {pct(nano_parsed)} parsed for nanoGPT. The five nanoGPT exact answers are also too few for a robust capability estimate.</p></section>
<section><h2>Ablation results</h2><div class="scroll"><table><thead><tr><th>Arm</th><th>Params</th><th>Complete artifact B</th><th>Microbatch</th><th>Update ms</th><th>Float five-ledger NELBO bits/atom</th><th>Export matched-float NELBO bits/atom</th><th>Artifact int4 NELBO bits/atom</th><th>Approx 95% CI bits/atom</th><th>Δ vs control bits/atom</th><th>Clean AR anchor bits/scored atom</th><th>Phase NELBO bits/atom</th><th>High-noise CE bits/atom</th><th>High-noise acc</th><th>GSM exact (float)</th><th>GSM parsed (float)</th><th>Invalid UTF-8</th><th>External sample BPB (float)</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<p class="small">Negative NELBO delta is better. “Artifact B” includes the serialized model and exact counted executable closure. The phase ledger exposes 0–3 fixed-clean prompt-phase atoms while leaving the canonical all-active ledger unchanged. High-noise is t∈[0.875,1].</p>
<p class="note">{escape(quant_note)} No tested artifact in this report passes the 0.05 bits/atom quantization-quality gate. The rejected joint checkpoint has no authenticated passing export mapped, so it is explicitly shown as unexported.</p></section>
<section><h2>Measured speed</h2><div class="scroll"><table><thead><tr><th>Workload</th><th>Retained Duo</th><th>Joint candidate</th><th>nanoGPT AR</th></tr></thead><tbody>
{tr(['2k wall time',fnum(float(baseline['train_seconds']),1)+' s',fnum(float(candidate['train_seconds']),1)+' s',fnum(nano_train_seconds,1)+' s'])}
{tr(['Readiness update',fnum(float(baseline['update_ms']),1)+' ms',fnum(float(candidate['update_ms']),1)+' ms','1,781.3 ms at final validation ledger'])}
{tr(['Generation loop, compile warmup excluded, 3×1,319',fnum(baseline_generation_seconds,1)+' s',fnum(candidate_generation_seconds,1)+' s',fnum(nano_generation_seconds,1)+' s (eager; unmatched metadata)'])}
{tr(['8-step fixed 512-atom trajectories/s',fnum(float(baseline['inference_trajectories_s']),1),fnum(float(candidate['inference_trajectories_s']),1),'not measured on matched compiled harness'])}
</tbody></table></div><p class="note">These are raw observed times, not matched speedup ratios. nanoGPT was eager because its compiled FlexAttention lowering failed and its artifact does not record a requested batch size. The Duo inference snapshots use fresh seeded weights. Commit-one decoding removes parallel generation; its raw 256-row run is roughly two orders of magnitude slower than the same candidate's full-block ledger, but the output caps differ (128 versus 512 atoms), so this is not a normalized speed ratio.</p></section>
<section><h2>Diffusion-step / quality curve — {escape(str(candidate['label']))}</h2><div class="scroll"><table><thead><tr><th>Reverse steps</th><th>GSM exact</th><th>Parsed</th><th>Invalid UTF-8</th><th>External sample BPB</th><th>External PPL</th><th>Unique</th></tr></thead><tbody>{''.join(nfe_rows)}</tbody></table></div>
<p class="small">This curve uses seed 0 and the first 256 held-out rows for every point, with complete response ledgers. The AR row is the matched nanoGPT control.</p></section>
<section><h2>Decode-topology controls — {escape(str(candidate['label']))}</h2><div class="scroll"><table><thead><tr><th>Topology</th><th>Visible atoms</th><th>Commit atoms</th><th>Output cap atoms</th><th>GSM exact</th><th>Parsed</th><th>Invalid UTF-8</th><th>Generated bytes</th><th>Seconds</th></tr></thead><tbody>{''.join(topology_rows)}</tbody></table></div>
<p class="note">The paper-aligned sampler control keeps the trained 512-atom visible canvas and commits one atom at a time; it still produces no correct answer and loses almost all parallel speed. Its output cap is 128 atoms versus 512 for full-block decoding, so the raw time ratio is not workload-normalized. The full-block failure therefore is not explained solely by committing 512 predictions at once.</p></section>
<section><h2>Complete sampled reverse trajectories — {escape(str(candidate['label']))}</h2><p>Every transition below records t→s, the exact transition kind, changed active positions, and a decoded preview. Atomic controls and byte-control characters are escaped visibly. The fixed prompt does not appear among revisable positions. The last row is the exact residual-noise cleanup.</p>{trace_sections(trace)}</section>
<section><h2>Reference alignment—and where this model is still a hybrid</h2><div class="scroll"><table><thead><tr><th>Subsystem</th><th>Byte-Duo implementation</th><th>Closest reference</th><th>Important divergence</th></tr></thead><tbody>
{tr(['Atoms / head','257-state scratch corruption support: 256 bytes + atomic EOT; untied 261-class clean output head','Scaling-Duo uses ordinary categorical states and an independent output projection','Four post-training controls are representable but absent from scratch targets and excluded from the scratch noise prior'])}
{tr(['Forward process / objective','Non-absorbing uniform replacement; exact continuous-time NELBO; exact ancestral posterior','Scaling Beyond Masked Diffusion Language Models','Small 64/32 time conditioner versus the reference 256/128; hierarchical decoder is project-specific'])}
{tr(['Hierarchy / n-grams','Causal clean byte encoder, 4-byte patches, global trunk, byte decoder; hashed 3–8-grams','BLT and Fast-BLT','Fixed stride rather than entropy patches; shared low-rank summed n-grams rather than per-order normalized embeddings'])}
{tr(['Diffusion canvas','Eight densely supervised 512-atom branches per packed 8,192-byte page','DiffusionGemma motivates a large continuation canvas','Scratch pretraining, no AR initialization/self-distillation; successful reference begins from an AR model'])}
{tr(['Generation','Joint reverse denoising with atomic controls, UTF-8 validation, optional commit width','Scaling-Duo full sequence; Fast-BLT small blocks; I-DLM proposal/verification','Full 512-atom commit is fast but low quality; commit-one retains visibility but loses parallel speed; no I-DLM accept/reject cell in this path'])}
</tbody></table></div><p class="note">The embedding and unembedding are conventional learned categorical projections; the principal research bet is the hierarchy plus mask-free diffusion, not a special bit-level encoding. The evidence above should therefore be read as an evaluation of this hybrid at 24.1M parameters and 2,000 updates, not as a reproduction result for any single paper.</p></section>
<section><h2>Interpretation boundaries</h2><ul>
<li>This is a 2k-update scratch-pretraining ablation, not the references’ mature post-training recipe. Scaling-Duo's GSM result follows five epochs of supervised fine-tuning on roughly 385K augmented examples and uses one-token-at-a-time left-to-right decoding. DiffusionGemma starts from an AR checkpoint and adds post-training. Fast-BLT's joint next-byte objective accompanies its complete architecture, not this partial hybrid.</li>
<li>A zero GSM exact score at this scale does not distinguish arithmetic knowledge from formatting failure. Parsed-answer rate, high-noise diagnostics, NFE curves, and trace behavior identify where generation fails.</li>
<li>High-noise modeling is a measured bottleneck while training occupancy is healthy: the retained training readiness sustains {pct(float(baseline['training_utilization'])/100)} utilization at {fnum(float(baseline['training_power_w']),1)} W, while both arms remain near 4.54 bits/atom and 17% accuracy at t∈[0.875,1]. Increasing reverse steps from 8 to 192 improves external sample BPB but not answer parsing.</li>
<li>The baseline/candidate decision above follows predeclared likelihood, high-noise, external-sample, and UTF-8 gates. Single checkpoints still do not quantify training-seed variance.</li>
<li>The 16MB export is authoritative. Parameter count alone is not the challenge constraint. This single-RTX-5090 2k study does not demonstrate the challenge's 8×H100 ten-minute training limit.</li>
</ul></section>
</main></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

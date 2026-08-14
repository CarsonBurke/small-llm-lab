#!/usr/bin/env python3
"""Render the final Byte-Duo training, quality, trace, and speed evidence."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.readiness import (
    diagnostic_cadence_contract,
    validate_architecture_readiness,
    validate_duo_inference_readiness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duo-result", type=Path, required=True)
    parser.add_argument("--duo-native-result", type=Path, required=True)
    parser.add_argument("--duo-contract", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--inference-readiness", type=Path, required=True)
    parser.add_argument("--nelbo", type=Path, required=True)
    parser.add_argument("--duo-gsm", type=Path, required=True)
    parser.add_argument("--duo-trace", type=Path, required=True)
    parser.add_argument("--nano-result", type=Path, required=True)
    parser.add_argument("--nano-gsm", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def nested(record: dict[str, object], *keys: str) -> object:
    value: object = record
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}"


def percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def atom(value: int) -> str:
    if value == 256:
        return "⟨EOT⟩"
    if value == 257:
        return "⟨THINK⟩"
    if value == 258:
        return "⟨/THINK⟩"
    if value == 259:
        return "⟨ANSWER⟩"
    if value == 260:
        return "⟨/ANSWER⟩"
    if value == 261:
        return "⟨MASK⟩"
    if value == 262:
        return "⟨PAD⟩"
    names = {9: "\\t", 10: "\\n", 13: "\\r", 32: "␠"}
    if value in names:
        return names[value]
    if 33 <= value <= 126:
        return chr(value)
    if 0 <= value < 256:
        return f"\\x{value:02x}"
    return f"⟨{value}⟩"


def atom_string(values: Iterable[int], *, limit: int | None = None) -> str:
    materialized = tuple(int(value) for value in values)
    shown = materialized if limit is None else materialized[:limit]
    result = " ".join(escape(atom(value)) for value in shown)
    if limit is not None and len(materialized) > limit:
        result += f" … ({len(materialized) - limit} more)"
    return result


def metric_card(label: str, value: str, note: str) -> str:
    return (
        '<div class="metric"><div class="metric-label">'
        + escape(label)
        + '</div><div class="metric-value">'
        + escape(value)
        + '</div><div class="metric-note">'
        + escape(note)
        + "</div></div>"
    )


def row(cells: Iterable[object]) -> str:
    return "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in cells) + "</tr>"


def readiness_candidate_table(readiness: dict[str, object]) -> str:
    results = nested(readiness, "results")
    require(isinstance(results, dict), "readiness results must be an object")
    selected = str(nested(readiness, "selected_microbatch"))
    body: list[str] = []
    for microbatch, value in sorted(results.items(), key=lambda item: int(item[0])):
        require(isinstance(value, dict), "readiness candidate must be an object")
        telemetry = value.get("telemetry", value)
        require(isinstance(telemetry, dict), "candidate telemetry must be an object")
        power = telemetry.get("power_w", value.get("power_w", {}))
        utilization = telemetry.get(
            "gpu_utilization_percent", value.get("gpu_utilization_percent", {})
        )
        power = power if isinstance(power, dict) else {}
        utilization = utilization if isinstance(utilization, dict) else {}
        status = "selected" if microbatch == selected else (
            "eligible" if value.get("eligible") else "rejected"
        )
        body.append(
            row(
                (
                    microbatch,
                    status,
                    fmt(finite(value.get("update_ms"), "candidate update_ms"), 2)
                    if value.get("update_ms") is not None
                    else "—",
                    fmt(finite(power.get("mean"), "candidate power"), 1)
                    if power.get("mean") is not None
                    else "—",
                    fmt(finite(utilization.get("mean"), "candidate utilization"), 1)
                    if utilization.get("mean") is not None
                    else "—",
                    fmt(
                        finite(
                            value.get("cuda_reserved_headroom_bytes", 0),
                            "candidate headroom",
                        )
                        / (1 << 30),
                        2,
                    ),
                )
            )
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Microbatch</th>'
        "<th>Status</th><th>Update ms</th><th>Mean W</th><th>Mean util %</th>"
        "<th>Reserved headroom GiB</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def gsm_summary_table(duo: dict[str, object], nano: dict[str, object]) -> str:
    rows_html: list[str] = []
    for name, result in (("Byte-Duo", duo), ("nanoGPT AR", nano)):
        records = result.get("records", result.get("results"))
        require(isinstance(records, list) and records, f"{name} GSM records missing")
        exact = tuple(finite(item["exact_match"], f"{name} exact match") for item in records)
        parsed = tuple(
            finite(item["parsed_answer_rate"], f"{name} parsed rate") for item in records
        )
        invalid_utf8 = (
            percent(
                statistics.mean(
                    finite(item["invalid_utf8_rate"], f"{name} invalid UTF-8 rate")
                    for item in records
                ),
                2,
            )
            if all("invalid_utf8_rate" in item for item in records)
            else "not recorded"
        )
        seconds = sum(finite(item["generation_seconds"], f"{name} seconds") for item in records)
        generated_bytes = sum(integer(item["generated_bytes"], f"{name} bytes") for item in records)
        decode = (
            f"categorical diffusion; batch {duo.get('requested_batch_size')}"
            if name == "Byte-Duo"
            else "greedy AR; batch not recorded in artifact"
        )
        rows_html.append(
            row(
                (
                    name,
                    decode,
                    ", ".join(percent(value, 3) for value in exact),
                    percent(statistics.mean(exact), 3),
                    percent(statistics.mean(parsed), 2),
                    invalid_utf8,
                    fmt(seconds, 2),
                    fmt(generated_bytes / seconds, 1),
                )
            )
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Model</th>'
        "<th>Native decoder</th><th>Exact match by seed</th><th>Mean exact match</th>"
        "<th>Mean parsed</th><th>Invalid UTF-8</th><th>Total generation s</th>"
        "<th>Literal bytes/s</th>"
        "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>"
    )


def validate_full_gsm_comparison(
    duo: dict[str, object], nano: dict[str, object]
) -> None:
    for key in (
        "gsm8k_train_sha256",
        "gsm8k_test_sha256",
        "prompt_format",
        "strip_calculator_annotations",
        "max_new_bytes",
    ):
        require(duo.get(key) == nano.get(key), f"GSM comparison differs on {key}")
    require(nano.get("seeds") == [0, 1, 2], "nano GSM seeds must be 0,1,2")
    duo_records = duo.get("records")
    nano_records = nano.get("results")
    require(
        isinstance(duo_records, list)
        and isinstance(nano_records, list)
        and len(duo_records) == len(nano_records) == 3,
        "full GSM comparison requires three paired records",
    )
    for duo_record, nano_record in zip(duo_records, nano_records, strict=True):
        require(
            isinstance(duo_record, dict) and isinstance(nano_record, dict),
            "GSM record must be an object",
        )
        for key in ("shots", "seed", "examples", "exemplar_train_rows"):
            require(
                duo_record.get(key) == nano_record.get(key),
                f"paired GSM records differ on {key}",
            )
        require(duo_record.get("shots") == 5, "GSM comparison must be five-shot")
        require(
            duo_record.get("examples") == 1_319,
            "GSM comparison must cover all 1,319 test examples",
        )
    require(
        [record.get("seed") for record in duo_records] == [0, 1, 2],
        "Duo GSM seeds must be 0,1,2",
    )


def trace_html(trace: dict[str, object]) -> str:
    require(trace.get("decode_mode") == "duo", "trace decode mode is not Duo")
    require(trace.get("sampling") == "categorical", "Duo trace must be categorical")
    diffusion_steps = integer(trace.get("diffusion_steps"), "trace diffusion steps")
    require(diffusion_steps > 0, "trace diffusion steps must be positive")
    schedule_eps = finite(trace.get("schedule_eps"), "trace schedule epsilon")
    require(0.0 < schedule_eps < 1.0, "trace schedule epsilon is invalid")
    records = trace.get("records")
    require(isinstance(records, list) and records, "trace contains no GSM records")
    sections: list[str] = []
    for record in records:
        require(isinstance(record, dict), "trace record must be an object")
        rows = record.get("rows")
        require(isinstance(rows, list), "trace result rows missing")
        for sample in rows:
            require(isinstance(sample, dict), "trace sample must be an object")
            diffusion_trace = sample.get("diffusion_trace")
            require(isinstance(diffusion_trace, dict), "sample diffusion trace missing")
            blocks = diffusion_trace.get("diffusion_blocks")
            require(isinstance(blocks, list) and blocks, "trace sample has no canvases")
            canvases: list[str] = []
            generated_bytes = 0
            committed_literal_values: list[int] = []
            for block_index, block in enumerate(blocks):
                require(isinstance(block, dict), "trace canvas must be an object")
                require(
                    block.get("block") == block_index
                    and block.get("prompt_atoms_noised") is False
                    and math.isclose(
                        finite(block.get("schedule_eps"), "block schedule epsilon"),
                        schedule_eps,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    ),
                    "trace claims the clean prompt was noised",
                )
                initial = tuple(int(value) for value in block["initial_ids"])
                active = tuple(int(value) for value in block["revisable_positions"])
                inactive = tuple(
                    int(value) for value in block["non_revisable_positions"]
                )
                width = len(initial)
                require(
                    len(set(active)) == len(active)
                    and len(set(inactive)) == len(inactive)
                    and set(active).isdisjoint(inactive)
                    and set(active) | set(inactive) == set(range(width)),
                    "trace active/inactive partition is invalid",
                )
                steps = block.get("steps")
                require(
                    isinstance(steps, list) and len(steps) == diffusion_steps + 1,
                    "trace must include every transition and final cleanup",
                )
                previous = initial
                step_rows: list[str] = []
                for expected_step, step in enumerate(steps, start=1):
                    require(isinstance(step, dict), "trace step must be an object")
                    require(
                        step.get("step") == expected_step
                        and step.get("all_active_suffix_positions_revisable") is True,
                        "trace step index/revisability is invalid",
                    )
                    final_cleanup = expected_step == diffusion_steps + 1
                    expected_t = (
                        schedule_eps
                        if final_cleanup
                        else 1.0
                        + (schedule_eps - 1.0)
                        * (expected_step - 1)
                        / diffusion_steps
                    )
                    expected_s = (
                        0.0
                        if final_cleanup
                        else 1.0
                        + (schedule_eps - 1.0)
                        * expected_step
                        / diffusion_steps
                    )
                    expected_alpha_t = 1.0 - (1.0 - schedule_eps) * expected_t
                    expected_alpha_s = (
                        1.0
                        if final_cleanup
                        else 1.0 - (1.0 - schedule_eps) * expected_s
                    )
                    require(
                        step.get("transition_kind")
                        == (
                            "exact_residual_noise_cleanup"
                            if final_cleanup
                            else "reverse_grid_posterior"
                        )
                        and math.isclose(
                            finite(step.get("time_t"), "trace time_t"),
                            expected_t,
                            rel_tol=1e-12,
                        )
                        and math.isclose(
                            finite(step.get("time_s"), "trace time_s"),
                            expected_s,
                            rel_tol=1e-12,
                        )
                        and math.isclose(
                            finite(step.get("alpha_t"), "trace alpha_t"),
                            expected_alpha_t,
                            rel_tol=1e-12,
                        )
                        and math.isclose(
                            finite(step.get("alpha_s"), "trace alpha_s"),
                            expected_alpha_s,
                            rel_tol=1e-12,
                        ),
                        "trace reverse-schedule ledger is invalid",
                    )
                    input_ids = tuple(int(value) for value in step["input_ids"])
                    output_ids = tuple(int(value) for value in step["output_ids"])
                    require(input_ids == previous, "trace transition input does not chain")
                    require(len(output_ids) == len(initial), "trace canvas width changed")
                    require(
                        all(output_ids[index] == input_ids[index] for index in inactive),
                        "trace changed a non-revisable clean-prefix slot",
                    )
                    changed = sum(
                        output_ids[index] != input_ids[index] for index in active
                    )
                    active_output = tuple(output_ids[index] for index in active)
                    step_rows.append(
                        "<tr>"
                        f"<td>{integer(step['step'], 'trace step')}</td>"
                        f"<td>{escape(str(step['transition_kind']))}<br>"
                        f"t {fmt(expected_t, 6)} → {fmt(expected_s, 6)}<br>"
                        f"α {fmt(expected_alpha_t, 6)} → {fmt(expected_alpha_s, 6)}</td>"
                        f"<td>{changed}</td>"
                        f'<td class="atoms">{atom_string(active_output, limit=96)}</td>'
                        "<td><details><summary>all active ids</summary>"
                        f'<div class="atoms full">{atom_string(active_output)}</div>'
                        "</details></td></tr>"
                    )
                    previous = output_ids
                initial_active = tuple(initial[index] for index in active)
                commit_positions = tuple(
                    int(value) for value in block["commit_positions"]
                )
                committed_ids = tuple(int(value) for value in block["committed_ids"])
                committed_bytes = tuple(
                    int(value) for value in block["committed_byte_values"]
                )
                require(
                    len(commit_positions) == len(committed_ids)
                    and tuple(sorted(commit_positions)) == commit_positions
                    and len(set(commit_positions)) == len(commit_positions)
                    and all(position in active for position in commit_positions),
                    "trace commit ledger is invalid",
                )
                require(
                    committed_ids
                    == tuple(previous[position] for position in commit_positions),
                    "trace commits do not match the final denoised state",
                )
                require(
                    committed_bytes
                    == tuple(value for value in committed_ids if value < 256),
                    "trace byte-retention ledger is invalid",
                )
                before = integer(
                    block["generated_bytes_before"], "trace bytes before commit"
                )
                after = integer(
                    block["generated_bytes_after"], "trace bytes after commit"
                )
                require(
                    before == generated_bytes
                    and after == before + len(committed_bytes),
                    "trace cumulative byte ledger is invalid",
                )
                generated_bytes = after
                committed_literal_values.extend(committed_bytes)
                transition = block.get("next_canvas_transition")
                termination = str(block["termination_after_block"])
                require(
                    (transition is None) == (termination != "continue"),
                    "trace continuation/termination ledger is inconsistent",
                )
                if transition is not None:
                    require(
                        isinstance(transition, dict)
                        and block_index + 1 < len(blocks)
                        and transition.get("absolute_atom_start")
                        == blocks[block_index + 1].get("absolute_atom_start")
                        and transition.get("clean_prefix_atoms_in_canvas")
                        == min(blocks[block_index + 1]["revisable_positions"]),
                        "trace next-canvas transition is invalid",
                    )
                    transition_note = (
                        "next canvas starts at atom "
                        + escape(str(transition["absolute_atom_start"]))
                        + " with "
                        + escape(str(transition["clean_prefix_atoms_in_canvas"]))
                        + " clean overlap atoms"
                    )
                else:
                    transition_note = "no next canvas"
                canvases.append(
                    "<details open><summary>Canvas "
                    + escape(str(block["block"]))
                    + " · absolute atom start "
                    + escape(str(block["absolute_atom_start"]))
                    + " · revisable atoms "
                    + str(len(active))
                    + "</summary>"
                    + f'<p class="atoms"><b>Uniform initial state:</b> '
                    + atom_string(initial_active, limit=96)
                    + "</p>"
                    + '<div class="table-wrap"><table><thead><tr><th>Reverse step</th>'
                    + "<th>Schedule transition</th><th>Changed active slots</th><th>Active-state preview</th>"
                    + "<th>Complete state</th></tr></thead><tbody>"
                    + "".join(step_rows)
                    + "</tbody></table></div>"
                    + '<p class="atoms"><b>Committed atoms:</b> '
                    + atom_string(committed_ids, limit=96)
                    + "</p><p><b>Commit:</b> "
                    + escape(str(before))
                    + " → "
                    + escape(str(after))
                    + " literal bytes · <b>after canvas:</b> "
                    + escape(termination)
                    + " · <b>transition:</b> "
                    + transition_note
                    + "</p></details>"
                )
            returned_count = integer(
                diffusion_trace["returned_byte_count_after_stop_trim"],
                "trace returned byte count",
            )
            returned_values = diffusion_trace.get(
                "returned_byte_values_after_stop_trim"
            )
            require(
                isinstance(returned_values, list)
                and returned_count == len(returned_values)
                and returned_count <= generated_bytes,
                "trace final retained-byte ledger is invalid",
            )
            raw_hex = sample.get("raw_hex")
            require(
                isinstance(raw_hex, str)
                and bytes(returned_values).hex() == raw_hex
                and tuple(returned_values)
                == tuple(committed_literal_values[:returned_count])
                and blocks[-1].get("termination_after_block")
                == sample.get("termination"),
                "trace returned bytes or final termination do not match the sample",
            )
            text = sample.get("text")
            output_text = text if isinstance(text, str) else "<invalid UTF-8>"
            sections.append(
                '<section class="sample"><h3>GSM8K row '
                + escape(str(sample["test_row"]))
                + "</h3><p><b>Gold:</b> "
                + escape(str(sample["gold"]))
                + " · <b>Parsed:</b> "
                + escape(str(sample.get("prediction")))
                + " · <b>Termination:</b> "
                + escape(str(sample["termination"]))
                + "</p><pre>"
                + escape(output_text)
                + "</pre><p class=\"atoms\"><b>Returned bytes after stop trim:</b> "
                + atom_string(returned_values, limit=128)
                + "</p>"
                + "".join(canvases)
                + "</section>"
            )
    return "".join(sections)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    duo_result = load_json(args.duo_result)
    native = load_json(args.duo_native_result)
    contract = load_json(args.duo_contract)
    readiness = load_json(args.readiness)
    inference = load_json(args.inference_readiness)
    nelbo = load_json(args.nelbo)
    duo_gsm = load_json(args.duo_gsm)
    duo_trace = load_json(args.duo_trace)
    nano_result = load_json(args.nano_result)
    nano_gsm = load_json(args.nano_gsm)
    data_manifest = load_json(args.data_manifest)

    require(
        duo_gsm.get("schema") == duo_trace.get("schema") == "byte_diffusion_gsm8k/v9",
        "Duo GSM evaluator schema mismatch",
    )
    require(
        duo_gsm.get("evaluator_source") == duo_trace.get("evaluator_source"),
        "full and traced Duo GSM use different evaluator revisions",
    )
    for label, artifact in (("full", duo_gsm), ("trace", duo_trace)):
        require(
            artifact.get("decode_mode") == "duo"
            and artifact.get("sampling") == "categorical"
            and artifact.get("block_length") == 512
            and artifact.get("diffusion_steps") == 8
            and artifact.get("max_new_atoms") == 512
            and artifact.get("max_new_bytes") == 512
            and artifact.get("context_bytes") == 8_192
            and artifact.get("requested_batch_size") == 8,
            f"{label} Duo GSM decoding geometry is not the authenticated recipe",
        )

    source = str(nested(contract, "source", "sha256"))
    data_sha = str(contract["dataset_payload_sha256"])
    checkpoint_sha = str(nelbo["checkpoint_sha256"])
    training = nested(contract, "training")
    require(isinstance(training, dict), "run training contract is malformed")
    trained_schedule_eps = finite(training.get("schedule_eps"), "trained schedule epsilon")
    require(
        math.isclose(
            finite(duo_gsm.get("schedule_eps"), "full GSM schedule epsilon"),
            trained_schedule_eps,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and math.isclose(
            finite(duo_trace.get("schedule_eps"), "trace schedule epsilon"),
            trained_schedule_eps,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "Duo GSM schedule differs from training",
    )
    training_evidence = validate_architecture_readiness(
        args.readiness,
        architecture="duo",
        source_sha256=source,
        dataset_payload_sha256=data_sha,
        global_batch_size=integer(training["global_batch_size"], "global batch"),
        microbatch_size=integer(training["microbatch_size"], "microbatch"),
        validation_batch_size=integer(
            training["validation_batch_size"], "validation batch"
        ),
        model_config=nested(contract, "model"),
        workload={
            "branches": training["branches"],
            "canvas_length": training["canvas_length"],
            "diagnostic_cadence": diagnostic_cadence_contract(
                log_every=integer(training["log_every"], "log cadence"),
                validation_every=integer(
                    training["validation_every"], "validation cadence"
                ),
            ),
            "objective": "pure_conditional_canvas_nelbo",
            "schedule_eps": training["schedule_eps"],
            "time_sampling": training["time_sampling"],
        },
        runtime={
            "compiled": training["compile"],
            "device_type": "cuda",
            "validation_rows": training["validation_rows"],
            "world_size": training["world_size"],
        },
    )
    inference_evidence = validate_duo_inference_readiness(
        args.inference_readiness,
        source_sha256=source,
        canvas_length=integer(training["canvas_length"], "canvas length"),
        model_config=nested(contract, "model"),
        parameter_count=integer(contract["parameter_count"], "parameter count"),
    )
    require(
        training.get("readiness_evidence")
        == {"training_validation": training_evidence, "inference": inference_evidence},
        "run contract does not bind the supplied readiness artifacts",
    )
    require(source == str(nelbo["source_sha256"]), "NELBO source drift")
    require(source == str(duo_gsm["training_source_sha256"]), "GSM source drift")
    require(source == str(duo_trace["training_source_sha256"]), "trace source drift")
    require(data_sha == str(readiness["dataset_payload_sha256"]), "readiness data drift")
    require(data_sha == str(nelbo["dataset_payload_sha256"]), "NELBO data drift")
    require(data_sha == str(data_manifest["payload_sha256"]), "manifest data drift")
    require(checkpoint_sha == str(duo_gsm["checkpoint_sha256"]), "GSM checkpoint drift")
    require(checkpoint_sha == str(duo_trace["checkpoint_sha256"]), "trace checkpoint drift")
    require(native.get("steps") == 2_000 and duo_result.get("steps") == 2_000, "run is not 2k")
    require(contract.get("run_name") == native.get("run_name") == duo_result.get("name"), "run identity drift")
    checkpoint_path = Path(str(native.get("checkpoint")))
    require(checkpoint_path.is_file(), "native result checkpoint does not exist")
    require(sha256(checkpoint_path) == checkpoint_sha, "evaluated checkpoint is not the trained run checkpoint")
    require(duo_trace.get("diffusion_trace_included") is True, "trace flag missing")
    validate_full_gsm_comparison(duo_gsm, nano_gsm)
    for label, artifact, expected_examples in (
        ("full", duo_gsm, 1_319),
        ("trace", duo_trace, 8),
    ):
        artifact_records = artifact.get("records")
        require(
            isinstance(artifact_records, list) and artifact_records,
            f"{label} Duo GSM records missing",
        )
        for record in artifact_records:
            require(
                isinstance(record, dict)
                and record.get("examples") == expected_examples
                and record.get("serialized_sample_count") == 8
                and isinstance(record.get("rows"), list)
                and len(record["rows"]) == 8,
                f"{label} Duo GSM coverage/sample ledger is invalid",
            )

    val_entries = duo_result.get("val_entries")
    require(isinstance(val_entries, list), "Duo result validation ledger missing")
    final_val = next(
        (
            value
            for value in reversed(val_entries)
            if isinstance(value, dict) and value.get("step") == 2_000
        ),
        None,
    )
    require(isinstance(final_val, dict), "Duo final validation missing")
    final_nelbo_bits = finite(
        final_val["val_diffusion_nelbo_bits_per_atom"], "final NELBO bits"
    )
    final_nelbo_nats = finite(final_val["val_loss"], "final NELBO nats")
    require(
        math.isclose(final_nelbo_bits, final_nelbo_nats / math.log(2), rel_tol=2e-6),
        "training NELBO bits/nats are inconsistent",
    )
    nano_entries = nano_result.get("val_entries")
    require(isinstance(nano_entries, list), "nano validation ledger missing")
    nano_final = next(
        (
            value
            for value in reversed(nano_entries)
            if isinstance(value, dict) and value.get("step") == 2_000
        ),
        None,
    )
    require(isinstance(nano_final, dict), "nano final validation missing")
    nano_bpb = finite(nano_final["val_bpb"], "nano final BPB")

    records = nelbo.get("records")
    require(
        nelbo.get("schema") == "byte_duo_nelbo_multiledger/v1"
        and nelbo.get("metric_semantics")
        == "conditional_canvas_duo_nelbo_not_ar_bpb"
        and nelbo.get("completed_steps") == 2_000
        and integer(nelbo.get("validation_rows"), "NELBO validation rows") >= 2_048,
        "NELBO artifact contract is incomplete",
    )
    require(isinstance(records, list) and len(records) == 5, "report requires five NELBO ledgers")
    nelbo_values = tuple(
        finite(value["conditional_canvas_nelbo_nats_per_atom"], "NELBO ledger")
        for value in records
    )
    ledger_seeds = nelbo.get("ledger_seeds")
    require(
        isinstance(ledger_seeds, list)
        and [record.get("ledger_seed") for record in records] == ledger_seeds
        and len(set(ledger_seeds)) == len(records),
        "NELBO seed ledger is inconsistent",
    )
    nelbo_mean = finite(nelbo["mean_nats_per_atom"], "NELBO mean")
    nelbo_sd = finite(nelbo["sample_sd_nats_per_atom"], "NELBO sample sd")
    require(
        math.isclose(nelbo_mean, statistics.mean(nelbo_values), rel_tol=1e-12)
        and math.isclose(nelbo_sd, statistics.stdev(nelbo_values), rel_tol=1e-12),
        "NELBO summary does not match its records",
    )

    gsm_records = duo_gsm.get("records")
    require(isinstance(gsm_records, list) and gsm_records, "Duo GSM records missing")
    duo_exact = statistics.mean(
        finite(value["exact_match"], "Duo GSM exact match") for value in gsm_records
    )
    duo_parsed = statistics.mean(
        finite(value["parsed_answer_rate"], "Duo GSM parsed rate")
        for value in gsm_records
    )
    duo_invalid_utf8 = statistics.mean(
        finite(value["invalid_utf8_rate"], "Duo GSM invalid UTF-8 rate")
        for value in gsm_records
    )
    duo_generation_seconds = tuple(
        finite(value["generation_seconds"], "Duo GSM generation seconds")
        for value in gsm_records
    )
    selected = str(readiness["selected_microbatch"])
    selected_result = nested(readiness, "results", selected)
    require(isinstance(selected_result, dict), "selected readiness record malformed")
    update_ms = finite(selected_result["update_ms"], "selected update ms")
    inference_atoms_s = finite(inference["requested_atoms_per_second"], "atoms/s")
    inference_traj_s = finite(inference["trajectories_per_second"], "trajectories/s")
    phases = inference.get("phases")
    require(
        isinstance(phases, list)
        and len(phases) == 4
        and all(isinstance(phase, dict) for phase in phases)
        and [phase.get("phase") for phase in phases] == [0, 1, 2, 3],
        "inference readiness phase ledger is invalid",
    )
    phase_zero_atoms_s = finite(
        phases[0]["requested_atoms_per_second"], "phase-zero requested atoms/s"
    )
    overflow_atoms_s = statistics.mean(
        finite(phase["requested_atoms_per_second"], "overflow requested atoms/s")
        for phase in phases[1:]
    )

    cards = "".join(
        (
            metric_card(
                "Duo NELBO",
                f"{fmt(final_nelbo_bits, 4)} bits/atom",
                f"{fmt(final_nelbo_nats, 4)} nats/atom; not AR BPB",
            ),
            metric_card(
                "Independent-ledger mean",
                f"{fmt(nelbo_mean, 4)} nats/atom",
                f"sample SD {fmt(nelbo_sd, 5)} across {len(nelbo_values)} ledgers",
            ),
            metric_card(
                "GSM8K exact match",
                percent(duo_exact, 3),
                f"parsed-answer rate {percent(duo_parsed, 2)}",
            ),
            metric_card(
                "nanoGPT AR BPB",
                fmt(nano_bpb, 4),
                "causal comparator only; not rank-comparable to Duo NELBO",
            ),
            metric_card(
                "Training update",
                f"{fmt(update_ms, 2)} ms",
                f"strict-ready microbatch {selected}, global batch 249",
            ),
            metric_card(
                "512-atom serving",
                f"{fmt(inference_atoms_s, 0)} atoms/s",
                f"{fmt(inference_traj_s, 2)} trajectories/s at batch 8",
            ),
        )
    )

    artifacts = (
        ("Duo ablation result", args.duo_result),
        ("Duo native result", args.duo_native_result),
        ("Duo run contract", args.duo_contract),
        ("Training readiness", args.readiness),
        ("Inference readiness", args.inference_readiness),
        ("Multi-ledger NELBO", args.nelbo),
        ("Full Duo GSM8K", args.duo_gsm),
        ("Traced Duo GSM8K", args.duo_trace),
        ("nanoGPT result", args.nano_result),
        ("nanoGPT GSM8K", args.nano_gsm),
        ("Byte dataset manifest", args.data_manifest),
    )
    artifact_rows = "".join(
        row((label, str(path), sha256(path), path.stat().st_size))
        for label, path in artifacts
    )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Byte-Duo 2k evidence report</title>
<style>
:root{{--bg:#091019;--panel:#111c28;--panel2:#172536;--text:#e8f0f7;--muted:#9eb0c2;--line:#2b4053;--cyan:#54d5e8;--green:#67d391;--amber:#f3bd63;--red:#ef7d7d}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(160deg,#071019,#0c1723 48%,#08121b);color:var(--text);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1380px;margin:auto;padding:42px 28px 80px}}h1{{font-size:42px;line-height:1.06;margin:0 0 12px}}h2{{margin-top:44px;border-bottom:1px solid var(--line);padding-bottom:8px}}h3{{margin-top:24px}}p,li{{max-width:1000px}}.lede{{font-size:18px;color:var(--muted)}}.badge{{display:inline-block;border:1px solid var(--green);color:var(--green);padding:4px 10px;border-radius:999px;font-weight:700;margin-bottom:16px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:28px 0}}.metric{{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:12px;padding:17px}}.metric-label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}}.metric-value{{font-size:25px;font-weight:750;color:var(--cyan);margin:5px 0}}.metric-note{{color:var(--muted);font-size:13px}}
.callout{{border-left:4px solid var(--amber);background:#201b13;padding:14px 18px;border-radius:6px;margin:20px 0;max-width:1050px}}.good{{border-left-color:var(--green);background:#102219}}code,.mono,.atoms{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}code{{color:#a9e8f1}}pre{{white-space:pre-wrap;background:#071019;border:1px solid var(--line);padding:14px;border-radius:8px;max-height:340px;overflow:auto}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:9px;margin:14px 0}}table{{border-collapse:collapse;width:100%;min-width:720px;background:var(--panel)}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}th{{position:sticky;top:0;background:var(--panel2);color:#cce6f0}}tr:last-child td{{border-bottom:0}}details{{border:1px solid var(--line);border-radius:8px;padding:9px 12px;margin:9px 0;background:#0d1823}}summary{{cursor:pointer;color:#cce6f0;font-weight:650}}.sample{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin:18px 0}}.atoms{{font-size:12px;line-height:1.45;word-break:break-word;color:#d3e7ee}}.atoms.full{{max-height:250px;overflow:auto;padding:9px;background:#08111a;border-radius:6px}}.small{{color:var(--muted);font-size:13px}}
</style></head><body><main>
<div class="badge">Authenticated 2,000-update scratch diffusion experiment</div>
<h1>Byte-Duo: quality, systems, and generation evidence</h1>
<p class="lede">Uniform-state byte diffusion trained from scratch with no absorbing MASK and no AR pretraining stage. Prompts stay clean; only fresh suffix atoms are initialized from the uniform prior and denoised.</p>
<div class="metrics">{cards}</div>

<div class="callout"><b>Metric boundary:</b> the Duo number is a conditional-canvas NELBO upper-bound proxy. It is not teacher-forced autoregressive bits-per-byte. The nanoGPT BPB is shown as an independent causal baseline, while GSM8K and generation behavior are the cross-family quality evidence.</div>

<h2>What was actually run</h2>
<ul>
<li>Source SHA-256 <code>{escape(source)}</code>; data payload <code>{escape(data_sha)}</code>.</li>
<li>24,094,791 parameters; 261 clean output states; PAD is inactive and MASK is forbidden in Byte-Duo.</li>
<li>Eight independent dense 512-atom document-contained canvases per 8,192-position page (up to 4,096 supervised atoms/page); exact global batch 249.</li>
<li>Continuous-time exact Duo NELBO, antithetic update-level time stratification, clean AR weight zero.</li>
<li>Eight requested reverse-grid transitions plus one exact final residual-noise-removal transition.</li>
<li>GSM prompts are strict UTF-8 clean prefixes. The evaluator does not noise the question or few-shot exemplars.</li>
</ul>

<h2>Strict training and validation readiness</h2>
<p>The same harness measures full optimizer updates and periodic held-row validation. Selection requires sustained power/utilization, at least 4 GiB and 15% reserved-memory headroom, and no measured graph breaks/recompiles/new graphs.</p>
{readiness_candidate_table(readiness)}

<h2>Diffusion metric reliability</h2>
<p>The final periodic training validation, evaluated on 256 rows, reported <b>{fmt(final_nelbo_nats, 6)} nats/atom</b>. Five independent counter-based ledgers evaluated 2,048 rows each and produced {', '.join(fmt(value, 6) for value in nelbo_values)} nats/atom. Their mean is <b>{fmt(nelbo_mean, 6)}</b> with sample SD <b>{fmt(nelbo_sd, 6)}</b>. This quantifies corruption-ledger variance; it does not remove the metric's cross-family limitation.</p>

<h2>GSM8K comparison</h2>
<p>Both models use the same local GSM8K snapshot, five-shot harness prompts, seeds 0/1/2, answer parser, and byte cap. Byte-Duo uses its native categorical reverse process at recorded batch size eight; nanoGPT uses greedy AR decoding. The prompts and scoring are matched, but the decoders are intentionally model-native rather than distribution-matched. Duo's current cohort RNG is reproducible for this fixed batch layout but is not row-invariant under rebatching, so the three seeds are reported separately. The bytes/s values are observed end-to-end evaluator throughput under those different native decoder configurations—not a matched serving benchmark. Token/atom counts and model-forward definitions remain model-specific.</p>
<p class="small">Duo's first full seed took {fmt(duo_generation_seconds[0], 2)} s, versus {', '.join(fmt(value, 2) for value in duo_generation_seconds[1:])} s for the later seeds. The explicit warmup excluded {fmt(finite(duo_gsm['compile_warmup_seconds_excluded_from_generation'], 'GSM compile warmup'), 2)} s, but live-row retirement still exposed additional compiled cohort shapes during seed 0. Therefore the aggregate bytes/s below includes cold shape-specialization cost and must not be read as steady-state kernel throughput; the separate authenticated 512-atom readiness benchmark is the systems number.</p>
{gsm_summary_table(duo_gsm, nano_gsm)}

<h2>Inference readiness and speed semantics</h2>
<p>Inference readiness covers every prompt phase modulo the four-byte compute patch. It measures a fixed 512 requested-atom budget at batch eight and ignores semantic EOT stopping, making workload size reproducible. Requested atoms/s is not GPT-2 tokens/s. The GSM table separately reports literal bytes/s under semantic stopping. The artifact's eligibility flag authenticates workload integrity, memory headroom, and stable compilation; it does not mean this latency-shaped workload met the embedded 425 W / 90% sustained-training saturation targets.</p>
<div class="table-wrap"><table><tbody>
{row(('Integrity/workload eligible', inference['eligible']))}
{row(('Posterior backend', inference['posterior_backend']))}
{row(('Diffusion steps', inference['diffusion_steps']))}
{row(('Trajectories/s', fmt(inference_traj_s, 3)))}
{row(('Requested atoms/s', fmt(inference_atoms_s, 1)))}
{row(('Denoised atom slots/s', fmt(finite(inference['denoised_atom_slots_per_second'], 'denoised slots/s'), 1)))}
{row(('Peak reserved GiB', fmt(finite(inference['peak_reserved_gib'], 'peak reserved'), 3)))}
{row(('Mean power W', fmt(finite(nested(inference, 'power_w', 'mean'), 'inference power'), 2)))}
{row(('Mean utilization %', fmt(finite(nested(inference, 'gpu_utilization_percent', 'mean'), 'inference util'), 2)))}
</tbody></table></div>

<h2>Known limitations and next measured ablations</h2>
<ul>
<li>The structured metrics and this report never call Duo NELBO “AR BPB.” The frozen training console still emits legacy <code>val_bpb</code>/<code>val_proxy_bpb</code> aliases for the same NELBO bits/atom; those aliases are invalid for challenge ranking and are deliberately ignored here.</li>
<li>Committed multi-canvas generation currently rebuilds the clean bank instead of incrementally extending clean KV state. The direct phase benchmark falls from {fmt(phase_zero_atoms_s, 1)} requested atoms/s for one canvas to a mean {fmt(overflow_atoms_s, 1)} for patch phases requiring two canvases. Incremental clean-cache extension is the clearest serving optimization, but it must be separately ablated against exact outputs.</li>
<li>Categorical GSM8K generations are invalid UTF-8 for <b>{percent(duo_invalid_utf8, 2)}</b> of rows on average across seeds ({', '.join(percent(finite(value['invalid_utf8_rate'], 'Duo GSM invalid UTF-8 rate'), 2) for value in gsm_records)}). This is a direct output-quality failure, not merely a rendering issue.</li>
<li>The no-mask uniform-state process is internally exact but is not a byte-for-byte reproduction of DUO_BASE's state space or sampling endpoints. Endpoint and prior variants remain future controlled ablations, not post-hoc changes to this run.</li>
<li>This is scratch pretraining only. No SFT, sampler distillation, or RL has been applied, so GSM8K measures base-model behavior rather than a post-trained math assistant.</li>
</ul>

<h2>Per-step GSM8K diffusion traces</h2>
<p>Every table begins with the uniform clean-state prior and includes every posterior state, including the final exact cleanup transition. “Changed” counts revisions among active suffix positions only; non-revisable prompt-phase atoms remain clean. Each canvas also records the exact committed atoms, literal-byte count before and after commit, termination decision, and transition into the next overlapping canvas.</p>
{trace_html(duo_trace)}

<h2>Artifact ledger</h2>
<p class="small">The report validates cross-artifact source, data, and checkpoint hashes before rendering. SHA-256 below covers each evidence file itself.</p>
<div class="table-wrap"><table><thead><tr><th>Artifact</th><th>Path</th><th>SHA-256</th><th>Bytes</th></tr></thead><tbody>{artifact_rows}</tbody></table></div>
</main></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(args.output)


if __name__ == "__main__":
    main()

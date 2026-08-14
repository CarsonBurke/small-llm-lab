#!/usr/bin/env python3
"""Score generated responses with one frozen external causal byte model.

This is a sampling-quality analogue of Gen PPL. It is not the evaluated
model's teacher-forced BPB: every generator is scored by the same separately
trained, hash-bound causal byte checkpoint. GPU execution must use ``mlq``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.inference import document_start_ar_metadata
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.training import CHECKPOINT_SCHEMA, model_config_from_dict


RESULT_SCHEMA = "external_causal_byte_generation_score/v2"


def evaluator_source_provenance() -> dict[str, object]:
    """Bind score artifacts to the exact evaluator implementation."""

    path = Path(__file__).resolve()
    return {
        "schema": "external_causal_byte_generation_evaluator_source/v1",
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def generated_rows(payload: dict[str, object]) -> tuple[dict[str, object], ...]:
    records = payload.get("records")
    rows: list[dict[str, object]] = []
    if isinstance(records, list) and records:
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("rows"), list):
                raise ValueError("generation record has no row ledger")
            for row in record["rows"]:
                if not isinstance(row, dict) or not isinstance(row.get("raw_hex"), str):
                    raise ValueError("generation row has no raw_hex")
                rows.append(row)
        return tuple(rows)

    # The matched nanoGPT evaluator predates the byte-diffusion row ledger. Its
    # `answer` field is the stop-trimmed response used for task scoring; using
    # `raw_generation` would unfairly retain delimiters that Duo physically
    # removes before writing raw_hex.
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("generation artifact has neither records nor results")
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("samples"), list):
            raise ValueError("generation result has no sample ledger")
        for sample in result["samples"]:
            if not isinstance(sample, dict) or not isinstance(sample.get("answer"), str):
                raise ValueError("generation sample has no stop-trimmed answer")
            rows.append(
                {
                    **sample,
                    "raw_hex": sample["answer"].encode("utf-8").hex(),
                    "invalid_utf8": False,
                }
            )
    return tuple(rows)


def generation_cohort(payload: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Authenticate the exact serialized GSM cohort used for sample scoring."""

    ledgers = payload.get("records", payload.get("results"))
    if not isinstance(ledgers, list) or not ledgers:
        raise ValueError("generation artifact has no evaluation ledgers")
    cohort: list[dict[str, object]] = []
    for ledger in ledgers:
        if not isinstance(ledger, dict):
            raise ValueError("generation ledger must be an object")
        rows = ledger.get("rows", ledger.get("samples"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("generation ledger has no serialized samples")
        declared = ledger.get("serialized_sample_count", len(rows))
        if declared != len(rows):
            raise ValueError("serialized sample count does not match the row ledger")
        test_rows = tuple(
            int(row.get("test_row", index))
            if isinstance(row, dict)
            else -1
            for index, row in enumerate(rows)
        )
        cohort.append(
            {
                "shots": ledger.get("shots"),
                "seed": ledger.get("seed"),
                "examples": ledger.get("examples"),
                "exemplar_train_rows": ledger.get("exemplar_train_rows"),
                "serialized_test_rows": test_rows,
            }
        )
    return tuple(cohort)


def validate_comparison_contract(payloads: tuple[dict[str, object], ...]) -> None:
    """Reject cross-generator comparisons with different GSM data or cohorts."""

    if not payloads:
        raise ValueError("at least one generation artifact is required")
    common_keys = (
        "gsm8k_train_sha256",
        "gsm8k_test_sha256",
        "prompt_format",
        "strip_calculator_annotations",
        "max_new_bytes",
    )
    reference = payloads[0]
    reference_cohort = generation_cohort(reference)
    for payload in payloads:
        if any(payload.get(key) != reference.get(key) for key in common_keys):
            raise ValueError("generation artifacts have different GSM contracts")
        if generation_cohort(payload) != reference_cohort:
            raise ValueError("generation artifacts have different serialized cohorts")


def repetition_fraction(data: bytes, order: int = 4) -> float:
    if len(data) < order:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(
        np.frombuffer(data, dtype=np.uint8), order
    )
    unique = np.unique(windows, axis=0).shape[0]
    return 1.0 - unique / windows.shape[0]


def continuation_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Sum next-byte NLL without ever presenting storage-only PAD to CE."""

    if logits.shape[:-1] != targets.shape or targets.shape != valid.shape:
        raise ValueError("continuation logits, targets, and validity must align")
    safe_targets = targets.masked_fill(~valid, -100)
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        safe_targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


def score_report(
    path: Path,
    payload: dict[str, object],
    rows: tuple[dict[str, object], ...],
    responses: tuple[bytes, ...],
    *,
    nll: float,
    byte_count: int,
) -> dict[str, object]:
    """Build one report while keeping response-count semantics testable."""

    if not responses or len(rows) != len(responses):
        raise ValueError("external score rows and responses must be nonempty and aligned")
    if byte_count <= 0:
        raise ValueError("external byte BPB is undefined for an all-empty corpus")
    bpb = nll / (byte_count * math.log(2.0))
    return {
        "generation": str(path),
        "generation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decode_mode": payload.get("decode_mode"),
        "diffusion_steps": payload.get("diffusion_steps"),
        "responses": len(responses),
        "nonempty_responses": sum(bool(data) for data in responses),
        "scored_bytes": byte_count,
        "scored_terminal_eot_atoms": len(responses),
        "empty_response_rate": sum(not data for data in responses) / len(responses),
        "external_causal_byte_bpb": bpb,
        "external_causal_byte_perplexity": 2.0**bpb,
        "invalid_utf8_rate": sum(bool(row.get("invalid_utf8")) for row in rows)
        / len(rows),
        "unique_response_rate": len(set(responses)) / len(responses),
        "mean_repeated_fourgram_fraction": sum(
            repetition_fraction(data) for data in responses
        )
        / len(responses),
    }


@torch.no_grad()
def score_responses(
    model: ByteDiffusionModel,
    responses: tuple[bytes, ...],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, int]:
    total_nll = 0.0
    total_bytes = 0
    stride = model.config.patch_stride
    pad_id = model.config.vocab.pad_id
    for start in range(0, len(responses), batch_size):
        batch = responses[start : start + batch_size]
        byte_lengths = torch.tensor(tuple(map(len, batch)), device=device)
        lengths = byte_lengths + 1  # one explicit EOT defines every sample end
        width = math.ceil(int(lengths.max()) / stride) * stride
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
        for row, data in enumerate(batch):
            byte_ids = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
            ids[row, : len(data)] = torch.from_numpy(byte_ids).to(device=device)
            ids[row, len(data)] = model.config.vocab.eot_id
        columns = torch.arange(width, device=device)
        valid = columns[None] < lengths[:, None]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.forward_ar_varlen(
                ids,
                valid,
                positions=columns[None].expand_as(ids),
                return_padded_logits=True,
                **document_start_ar_metadata(valid, stride),
            )
            if output.bos_patch_states is None:
                raise AssertionError("external scorer omitted virtual-BOS states")
            bos_logits = model.forward_bos_logits(output.bos_patch_states)
        first_nll = F.cross_entropy(
            bos_logits.float(), ids[:, 0], reduction="sum"
        )
        continuation_mask = columns[None, 1:] < lengths[:, None]
        continuation_nll = continuation_cross_entropy(
            output.logits[:, :-1], ids[:, 1:], continuation_mask
        )
        total_nll += float(first_nll + continuation_nll)
        total_bytes += int(byte_lengths.sum())
    return total_nll, total_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--generation", type=Path, action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("external generation scoring requires CUDA through mlq")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable result {args.output}")
    checkpoint = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("external scorer must be a causal byte checkpoint")
    if checkpoint.get("run_contract", {}).get("recipe") != "causal_only":
        raise ValueError("external scorer checkpoint is not causal-only")
    scorer_training_source = checkpoint.get("source_provenance")
    if (
        not isinstance(scorer_training_source, dict)
        or not scorer_training_source.get("schema")
        or not scorer_training_source.get("sha256")
    ):
        raise ValueError("external scorer checkpoint has no training-source provenance")
    config = model_config_from_dict(checkpoint["model_config"])
    model = ByteDiffusionModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda")
    model = model.to(device).eval()
    # Compile the two explicit scoring paths independently. Compiling the module
    # wrapper itself does not reliably preserve access to custom forward methods.
    model.forward_ar_varlen = torch.compile(  # type: ignore[method-assign]
        model.forward_ar_varlen, dynamic=True, fullgraph=False
    )
    model.forward_bos_logits = torch.compile(  # type: ignore[method-assign]
        model.forward_bos_logits, dynamic=True, fullgraph=False
    )

    payloads = tuple(json.loads(path.read_text()) for path in args.generation)
    if any(not isinstance(payload, dict) for payload in payloads):
        raise TypeError("generation artifacts must contain JSON objects")
    validate_comparison_contract(payloads)
    reports: list[dict[str, object]] = []
    for path, payload in zip(args.generation, payloads, strict=True):
        rows = generated_rows(payload)
        responses = tuple(bytes.fromhex(str(row["raw_hex"])) for row in rows)
        nll, byte_count = score_responses(
            model, responses, batch_size=args.batch_size, device=device
        )
        reports.append(
            score_report(
                path,
                payload,
                rows,
                responses,
                nll=nll,
                byte_count=byte_count,
            )
        )
    result = {
        "schema": RESULT_SCHEMA,
        "metric_semantics": (
            "frozen_external_causal_byte_score_of_samples_not_generator_ar_bpb"
        ),
        "normalization": (
            "sum_byte_and_terminal_eot_nll_divided_by_literal_response_bytes"
        ),
        "evaluator_source": evaluator_source_provenance(),
        "scorer_checkpoint": str(args.scorer_checkpoint),
        "scorer_checkpoint_sha256": hashlib.sha256(
            args.scorer_checkpoint.read_bytes()
        ).hexdigest(),
        "scorer_checkpoint_schema": checkpoint["schema"],
        "scorer_recipe": checkpoint["run_contract"]["recipe"],
        "scorer_training_source": scorer_training_source,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("external_generation_score " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

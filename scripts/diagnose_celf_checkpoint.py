"""Score any CELF checkpoint on the complete validation panel, completed run or not.

This is the diagnostic counterpart of ``train_celf.py evaluate``: it admits a
recovery checkpoint of a cancelled run and labels the report with the
checkpoint's update index. It lives outside the CELF source-hash contract so it
can be added without invalidating in-flight runs; it changes no training source.
Run through mlq.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pretraining.celf.config import ModelConfig  # noqa: E402
from pretraining.celf.data import ByteCorpus  # noqa: E402
from pretraining.celf.evaluation import generate_bytes  # noqa: E402
from pretraining.celf.model import CelfModel  # noqa: E402
from pretraining.celf.training import (  # noqa: E402
    TrainConfig,
    atomic_json,
    evaluate_nelbo,
    evaluate_samples,
    load_checkpoint,
    setup_device,
    sha256_file,
    source_hashes,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--probes", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; diagnostics are immutable")
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint["source_hashes"] != source_hashes():
        raise ValueError("CELF sources differ from the checkpoint's; refusing to score")
    config = TrainConfig(**checkpoint["train_config"])
    model_config = ModelConfig(**checkpoint["model_config"])
    device = setup_device(config.seed)
    corpus = ByteCorpus(ROOT / config.data_path, device)
    if corpus.metadata != checkpoint["data"]:
        raise ValueError("dataset differs from training provenance")
    model = CelfModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False).eval()
    model.compile_components()
    complete = corpus.validation.numel() // model_config.seq_bytes
    complete -= complete % config.microbatch
    validation = corpus.validation_prefix(complete * model_config.seq_bytes, model_config.seq_bytes)
    digest = sha256_file(args.checkpoint)
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_step": checkpoint["step"],
        "configured_steps": config.steps,
        "completed_run": checkpoint["step"] == config.steps,
        "run_name": config.name,
        "train_config": checkpoint["train_config"],
        "loss_config": checkpoint["loss_config"],
        "source_hashes": source_hashes(),
        "data": corpus.metadata,
        "scored_bytes": int(validation.numel()),
        "heldout_bytes": int(corpus.validation.numel()),
        "semantics": "Diagnostic scoring of a checkpoint at the recorded update index; one-sample fixed-anchor negative-ELBO estimates over complete validation contexts with Hutchinson divergence; not a completed-run result",
        "density": [],
        "samples": evaluate_samples(model, validation, config, sequences=validation.shape[0]),
        "generation": [],
    }
    started = time.perf_counter()
    for resolution in args.resolutions:
        record = evaluate_nelbo(model, validation, config, steps=resolution, probes=args.probes, sequences=validation.shape[0])
        record["steps"] = resolution
        record["probes"] = args.probes
        report["density"].append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
    for start in (0, 4096):
        prompt = bytes(corpus.validation[start : start + 2 * model_config.block_bytes].cpu().tolist())
        generated = generate_bytes(model, prompt, new_bytes=8 * model_config.block_bytes, steps=16, seed=1337)
        generated["prompt_text"] = prompt.decode("utf-8", errors="replace")
        report["generation"].append(generated)
    if sha256_file(args.checkpoint) != digest or source_hashes() != report["source_hashes"]:
        raise RuntimeError("inputs changed during execution")
    report["status"] = "complete"
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()

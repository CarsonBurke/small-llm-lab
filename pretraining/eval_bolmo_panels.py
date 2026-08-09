"""Score a Bolmo checkpoint on every evaluation panel of a dataset.

Reports, per split, the three bits-per-byte numbers that mean different things:

    codelength_bpb   the joint fused NLL. The tight *valid* codelength, and
                     the only number comparable against a subword model's BPB.
    marginalized_bpb the released evaluator's number, comparable against the
                     paper's Bolmo results and against nothing else.
    causal_bpb       predicted boundaries with strictly-prior routing. Also
                     valid, but loose by the train/eval mismatch it charges.

``marginalized_bpb`` marginalizes the fused boundary class away while routing
has already consumed that same boundary bit, which is derived from the byte
being scored. It does not normalize, so no decoder attains it. See
``BolmoModel.validation_statistics``.

The panel dataset need not be the one the checkpoint trained on -- that is the
point -- but it must be *compatible*, and compatibility here is not a
formality. ``expanded_ids`` indexes the source model's 50k-row embedding table
by longest-suffix match, so a panel built under a different source tokenizer
would silently index a different vocabulary and report numbers that look fine.
This driver therefore binds the checkpoint to its training dataset by manifest
hash, exactly as the trainer does, and then requires the panel's tokenizer and
atomic-vocabulary blocks to be byte-identical to that dataset's.

Read-only with respect to both datasets and the checkpoint. Run through mlq: it
executes a model on the GPU.

    mlq submit --name bolmo_panels --cwd "$PWD" --max-parallel-runs 1 -- \
      .venv/bin/python -m pretraining.eval_bolmo_panels \
        --checkpoint logs/bolmo_armC_srcopt_v2_final_model.pt \
        --train-data data/datasets/k3mix_v8_armC_bolmo_byte2048_epoch1 \
        --panel-data data/datasets/k3mix_v8_armC_bolmo_byte2048_panels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pretraining.bolmo import BolmoArchitecture, BolmoModel
from pretraining.train_bolmo import (
    ExampleStream,
    artifact_paths,
    evaluate,
    load_manifest,
    validate_paper_data_manifest,
)

# The blocks that must agree for a foreign panel to be meaningful. The
# tokenizer block pins ordered specials, merges, numeric vocabulary and the
# spec/n-gram hashes; the atomic block pins the byte-side ids and padding.
COMPATIBILITY_BLOCKS = ("source_tokenizer", "atomic_vocabulary")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--train-data",
        required=True,
        help="dataset the checkpoint was trained on; binds it by manifest hash",
    )
    parser.add_argument(
        "--panel-data",
        default=None,
        help="dataset holding the panels (default: --train-data)",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="comma-separated split names (default: every non-train split)",
    )
    parser.add_argument(
        "--patch-budget-scale",
        type=float,
        default=1.0,
        help=(
            "multiply the stored source width to get the pooled patch budget. "
            "1.0 is the training geometry and is correct on the canonical "
            "panel; out-of-distribution panels where the predictor "
            "over-segments need headroom, and exceeding the budget still "
            "raises rather than truncating"
        ),
    )
    parser.add_argument("--microbatch-examples", type=int, default=8)
    parser.add_argument("--backend", default="triton")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--autocast-kernel-dtype", default="float32")
    parser.add_argument("--output", default=None)
    return parser


def canonical_block(manifest: dict, name: str) -> str:
    return json.dumps(manifest[name], sort_keys=True, separators=(",", ":"))


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda")
    train_dir = Path(args.train_data)
    panel_dir = Path(args.panel_data or args.train_data)

    train_manifest = load_manifest(train_dir)
    panel_manifest = load_manifest(panel_dir)
    validate_paper_data_manifest(train_manifest)
    validate_paper_data_manifest(panel_manifest)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    recorded = payload.get("data_manifest_sha256")
    if not isinstance(recorded, str):
        raise ValueError("checkpoint does not bind a data manifest hash")
    if recorded != train_manifest["payload_sha256"]:
        raise ValueError(
            "--train-data is not the dataset this checkpoint trained on: "
            f"{recorded} != {train_manifest['payload_sha256']}"
        )
    for block in COMPATIBILITY_BLOCKS:
        if canonical_block(train_manifest, block) != canonical_block(
            panel_manifest, block
        ):
            raise ValueError(
                f"panel dataset {block!r} differs from the training dataset's; "
                "expanded_ids would index a different vocabulary"
            )

    config = dict(payload["model_config"])
    # Runtime kernel choices are not part of the architecture contract.
    config["architecture"] = dict(config["architecture"]) | {
        "backend": args.backend,
        "chunk_size": args.chunk_size,
        "autocast_kernel_dtype": args.autocast_kernel_dtype,
    }
    architecture = BolmoArchitecture(**config["architecture"])
    source_meta = panel_manifest["source_tokenizer"]
    atomic_meta = panel_manifest["atomic_vocabulary"]
    byte_pad_id = int(atomic_meta["pad_id"])
    source_pad_id = int(source_meta["source_pad_id"])
    if architecture.byte_pad_id != byte_pad_id:
        raise ValueError(
            "dataset atomic padding differs from the checkpoint: "
            f"{byte_pad_id} != {architecture.byte_pad_id}"
        )
    model = BolmoModel.from_config(config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)

    if args.splits:
        split_names = [name for name in args.splits.split(",") if name.strip()]
    else:
        split_names = [
            name for name in sorted(panel_manifest["splits"]) if name != "train"
        ]
    if not split_names:
        raise ValueError("no panels to score")

    results = []
    for split in split_names:
        declared = panel_manifest["splits"][split]
        context = int(declared["context_source_tokens"])
        stream = ExampleStream(
            artifact_paths(panel_dir, panel_manifest, split),
            seed=0,
            repeat=False,
            shuffle=False,
            source_vocab_size=int(source_meta["logical_vocab_size"]),
            source_pad_id=source_pad_id,
            atomic_vocab_size=architecture.atomic_vocab_size,
            context_source_tokens=context,
        )
        rows = stream.take(int(declared["examples"]))
        if len(rows) != int(declared["examples"]):
            raise ValueError(
                f"{split} is shorter than its manifest declares: "
                f"{len(rows)} != {declared['examples']}"
            )
        stored_width = int(
            panel_manifest["examples"]["validation_stored_source_width"]
            if split != "train"
            else panel_manifest["examples"]["training_stored_source_width"]
        )
        budget = max(stored_width, int(stored_width * args.patch_budget_scale))
        common = dict(
            patch_budget=budget,
            microbatch_examples=args.microbatch_examples,
            byte_pad_id=byte_pad_id,
            source_pad_id=source_pad_id,
            device=device,
        )
        predicted = evaluate(model, rows, **common)
        causal = evaluate(model, rows, causal_routing=True, **common)
        record = {
            "split": split,
            # The comparand. Same value as joint_bpb, named for what it is.
            "codelength_bpb": predicted.joint_bpb,
            "marginalized_bpb": predicted.byte_bpb,
            "causal_bpb": causal.byte_bpb,
            "boundary_accuracy": predicted.boundary_accuracy,
            "bytes_per_patch": predicted.predicted_bytes_per_patch,
            "scored_source_tokens": int(declared["scored_source_tokens"]),
            "atomic_tokens": int(
                declared["atomic_tokens_including_prepended_eot"]
            ),
            # Recorded because raising it lets the trunk see a longer sequence
            # than training did; a panel scored at >1.0 is not scored under the
            # training geometry.
            "patch_budget": budget,
            "stored_source_width": stored_width,
            "examples": len(rows),
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        results.append(record)

    summary = {
        "checkpoint": str(args.checkpoint),
        "train_data": str(train_dir),
        "train_data_manifest_sha256": train_manifest["payload_sha256"],
        "panel_data": str(panel_dir),
        "panel_data_manifest_sha256": panel_manifest["payload_sha256"],
        "results": results,
    }
    print(json.dumps(summary, indent=1, sort_keys=True), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

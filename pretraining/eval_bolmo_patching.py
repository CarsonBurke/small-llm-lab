"""Bracket what Bolmo's learned boundary predictor is worth.

Evaluates one trained checkpoint on the canonical validation span under
patchings that differ only in where patch boundaries fall:

    uniform       placement-blind floors, one matched to the oracle's patch
                  count and one to the predictor's own
    fixed stride  a boundary every N valid bytes, as coarser reference points
    predicted     the learned cosine boundary predictor (canonical rule)
    oracle        the source tokenizer's own boundaries (ceiling)

and one diagnostic that is not a patching at all:

    predicted_causal_routing
                  the same predicted boundaries, but a position may only read
                  a patch that closed before it

The boundary predictor is non-causal by design, so under every patching above
except the last, the score for byte ``t + 1`` is conditioned on a boundary bit
computed from byte ``t + 1``. Their ``byte_bpb`` is therefore not a codelength
and must not be compared against a subword model's bits-per-byte. Use
``predicted_joint_bpb``: it charges the boundary at ``t + 1`` from position
``t``, one step before routing reads it, so it is a causal codelength and a
tight one. ``causal_routing_bpb`` is also valid but is a *loose* bound — it
charges a train/eval mismatch of about 2.2 bpb on top of the leak — so it
brackets the comparison rather than being it. See
``BolmoModel.validation_statistics`` for the full argument.

The floor bounds below what *placement* buys the predictor, and the oracle
bounds above what any predictor could buy. A predicted BPB near the floor means
the boundary apparatus is not earning its cost; near the oracle means the
remaining gap lives in the byte model, not in patching.

The floor is placement-blind, not content-blind: it is handed a per-row patch
count, and every source of that count is content-derived. It therefore bounds
the value of deciding *where* boundaries fall, given how many the row deserves.
It says nothing about the value of knowing anything about the content.

The controlled floor is the *uniform* one, not a global stride. The oracle
spends exactly one patch per source token and so saturates the pooled patch
budget; any stride dense enough to match its mean density overruns that budget
on byte-rich rows, and the strides that do fit are coarser, which would confound
boundary placement with patch count and trunk sequence length. Uniform patching
holds the count fixed row by row and varies only placement.

Because the oracle and the predictor disagree on that count — the oracle spends
about 5% more patches on the canonical span — one floor cannot bracket both.
Both are reported, and ``learned_predictor_placement_gain_bpb`` compares the
predicted arm only against the floor carrying the predictor's own count.

Read-only with respect to the checkpoint and the dataset. Run through mlq: it
executes a model on the GPU.

    mlq submit --name bolmo_patching_bracket --cwd "$PWD" --max-parallel-runs 1 -- \
      .venv/bin/python -m pretraining.eval_bolmo_patching \
        --checkpoint logs/bolmo_armC_paper_v3_final_model.pt \
        --data data/datasets/k3mix_v8_armC_bolmo_byte2048_paper_v3
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--strides",
        default="6,8",
        help=(
            "comma-separated global strides, as coarser reference points. "
            "Strides below ~6 overrun the pooled patch budget on byte-rich "
            "rows; the controlled floor is the count-matched uniform patching."
        ),
    )
    parser.add_argument("--canonical-val-source-tokens", type=int, default=2**21)
    parser.add_argument("--microbatch-examples", type=int, default=8)
    parser.add_argument("--backend", default="triton")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--autocast-kernel-dtype", default="float32")
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda")
    data_dir = Path(args.data)
    manifest = load_manifest(data_dir)
    source_meta = manifest["source_tokenizer"]
    atomic_meta = manifest["atomic_vocabulary"]
    byte_pad_id = int(atomic_meta["pad_id"])
    source_pad_id = int(source_meta["source_pad_id"])

    validate_paper_data_manifest(manifest)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # The trainer binds the run to the dataset by manifest hash; an evaluator
    # that skips the binding will happily score a checkpoint against a dataset
    # it never saw and report the number as if it were canonical.
    recorded_manifest_sha = payload.get("data_manifest_sha256")
    if not isinstance(recorded_manifest_sha, str):
        raise ValueError("checkpoint does not bind a data manifest hash")
    if recorded_manifest_sha != manifest["payload_sha256"]:
        raise ValueError(
            "checkpoint was trained against a different dataset: "
            f"{recorded_manifest_sha} != {manifest['payload_sha256']}"
        )
    config = dict(payload["model_config"])
    # Runtime kernel choices are deliberately not part of the paper
    # architecture contract, so they are re-pinned here rather than inherited.
    config["architecture"] = dict(config["architecture"]) | {
        "backend": args.backend,
        "chunk_size": args.chunk_size,
        "autocast_kernel_dtype": args.autocast_kernel_dtype,
    }
    architecture = BolmoArchitecture(**config["architecture"])
    if architecture.byte_pad_id != byte_pad_id:
        raise ValueError(
            "dataset atomic padding differs from the checkpoint: "
            f"{byte_pad_id} != {architecture.byte_pad_id}"
        )
    model = BolmoModel.from_config(config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)

    source_tokens_per_example = int(manifest["examples"]["source_sequence_length"])
    if args.canonical_val_source_tokens % source_tokens_per_example:
        raise ValueError(
            "canonical validation tokens must divide by source tokens per example"
        )
    val_stream = ExampleStream(
        artifact_paths(data_dir, manifest, "validation"),
        seed=0,
        repeat=False,
        shuffle=False,
        source_vocab_size=int(source_meta["logical_vocab_size"]),
        source_pad_id=source_pad_id,
        atomic_vocab_size=architecture.atomic_vocab_size,
        context_source_tokens=1,
    )
    wanted = args.canonical_val_source_tokens // source_tokens_per_example
    rows = val_stream.take(wanted)
    if len(rows) != wanted:
        raise ValueError(
            f"validation split is too short: {len(rows)} != {wanted} examples"
        )

    def run(label: str, **kwargs) -> dict:
        metrics = evaluate(
            model,
            rows,
            microbatch_examples=args.microbatch_examples,
            byte_pad_id=byte_pad_id,
            source_pad_id=source_pad_id,
            device=device,
            **kwargs,
        )
        record = {
            "patching": label,
            "byte_bpb": metrics.byte_bpb,
            "joint_bpb": metrics.joint_bpb,
            "bytes_per_patch": metrics.predicted_bytes_per_patch,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        return record

    # One floor per arm it brackets. The oracle spends ~5% more patches than
    # the predictor on this span, so a single oracle-count floor would hand the
    # predicted arm's floor a finer patching than the arm itself gets.
    uniform_oracle = run("uniform_oracle_count", uniform_patching="oracle")
    uniform_predicted = run(
        "uniform_predicted_count", uniform_patching="predicted"
    )
    results = [uniform_oracle, uniform_predicted]
    results.extend(
        run(f"fixed_stride_{stride}", fixed_stride=int(stride))
        for stride in args.strides.split(",")
        if stride.strip()
    )
    predicted = run("predicted")
    oracle = run("oracle", oracle_boundaries=True)
    # Each floor must actually carry its arm's patch density. The counts are
    # recomputed in a separate pass, so this is the check that a floor silently
    # matched to the wrong arm would fail — which is the defect this bracket
    # shipped with until 2026-08-08.
    for floor, arm, label in (
        (uniform_oracle, oracle, "oracle"),
        (uniform_predicted, predicted, "predicted"),
    ):
        if floor["bytes_per_patch"] != arm["bytes_per_patch"]:
            raise ValueError(
                f"the {label} floor is not count-matched to the {label} arm: "
                f"{floor['bytes_per_patch']} != {arm['bytes_per_patch']}"
            )
    # The codelength diagnostic, not a patching: same predicted boundaries,
    # but a position may only read a patch that closed before it. See
    # ``BolmoModel.validation_statistics`` for why ``predicted``'s ``byte_bpb``
    # is not a codelength and this one is.
    causal = run("predicted_causal_routing", causal_routing=True)
    results.extend((predicted, oracle, causal))

    # Only the count-matched floors are controlled comparisons: a global stride
    # dense enough to match the oracle's mean density overruns the pooled patch
    # budget, so the surviving strides also change the patch count. Each arm is
    # measured against the floor carrying its own patch count.
    predicted_floor = uniform_predicted["byte_bpb"]
    oracle_floor = uniform_oracle["byte_bpb"]
    # The ratio is a rough guide, not a clean fraction: it divides a gap
    # measured at the predicted patch count by a span whose endpoints sit at
    # two different counts, and its three byte_bpb terms carry different
    # amounts of non-causality (one byte of boundary lookahead for
    # ``predicted``, whole-row quantities for the floors and the oracle).
    span = predicted_floor - oracle["byte_bpb"]
    summary = {
        "checkpoint": str(args.checkpoint),
        # Validating provenance without recording it leaves the artifact
        # unattributable after the fact, which is half a fix.
        "data": str(data_dir),
        "data_manifest_sha256": manifest["payload_sha256"],
        "scored_rows": len(rows),
        "canonical_val_source_tokens": args.canonical_val_source_tokens,
        "results": results,
        "uniform_predicted_count_floor_bpb": predicted_floor,
        "uniform_oracle_count_floor_bpb": oracle_floor,
        "predicted_bpb": predicted["byte_bpb"],
        "oracle_bpb": oracle["byte_bpb"],
        # What placement alone is worth at the predictor's own patch budget:
        # the controlled number, with both terms at the same patch count.
        "learned_predictor_placement_gain_bpb": (
            predicted_floor - predicted["byte_bpb"]
        ),
        # 0 means the learned predictor is worth nothing over blind placement;
        # 1 means it matches oracle patching. Values above 1 are expected and
        # observed (1.012 and 1.005 on the two measured arms): the oracle is
        # not a ceiling here, because ``byte_bpb`` is non-causal by a whole
        # row under oracle patching and by one byte under the predicted one.
        # Under the valid ``joint_bpb`` the oracle also loses outright, since
        # the joint charges for whichever boundary sequence is transmitted and
        # the model finds its own cheaper than the tokenizer's.
        "learned_predictor_recovery": (
            (predicted_floor - predicted["byte_bpb"]) / span
            if span > 0
            else float("nan")
        ),
        # ``predicted``'s marginal is conditioned on a boundary bit derived
        # from the byte it scores, so only these two are codelengths that a
        # subword model's BPB can be compared against.
        "causal_routing_bpb": causal["byte_bpb"],
        "predicted_joint_bpb": predicted["joint_bpb"],
        # How much the one byte of boundary lookahead is worth, inclusive of
        # the train/eval mismatch it charges the trained model.
        "noncausal_routing_credit_bpb": (
            causal["byte_bpb"] - predicted["byte_bpb"]
        ),
    }
    print(json.dumps(summary, indent=1, sort_keys=True), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

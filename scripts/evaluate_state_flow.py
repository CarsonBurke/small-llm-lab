"""Evaluate a completed dense pool checkpoint with cross-layer reads removed.

Run through mlq. This applies no optimizer updates and does not change weights.
Lattice-only removes higher-layer banks, which can only carry past-token
feedback to a lower reader. Own-bank-only retains the original query, router weights, gate parameters and self-bank
channel; all other source channels are zeroed immediately before the readout.
Later-layer gate values and hidden states recompute naturally. It measures
checkpoint dependence on cross-layer reads, including the effect
of removing their amplitude, rather than a retrained architecture comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn


def source_mask(width: int, sources: int, owner: int, device, *, include_lower: bool = False) -> torch.Tensor:
    if sources < 1 or width < 1 or width % sources or not 0 <= owner < sources:
        raise ValueError("source channels must divide the readout and include its owner")
    bank = torch.arange(width, device=device) // (width // sources)
    return bank <= owner if include_lower else bank == owner


class SourceMaskReadout(nn.Module):
    def __init__(self, readout: nn.Linear, sources: int, owner: int, *, include_lower: bool = False):
        super().__init__()
        self.readout = readout
        self.register_buffer("mask", source_mask(readout.in_features, sources, owner, readout.weight.device, include_lower=include_lower))

    def forward(self, read: torch.Tensor) -> torch.Tensor:
        return self.readout(read * self.mask)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--compare-query-shift", action="store_true",
                        help="also reproduce full-panel BPB with the isolated query-shift execution")
    args = parser.parse_args()
    for name in (args.run, args.name):
        if Path(name).name != name or name in (".", ".."):
            parser.error("run names must be single directory components")
    run = ROOT / "ablation_results" / args.run
    config = json.loads((run / "config.json").read_text())
    result = json.loads((run / "result.json").read_text())
    if (result["status"] != "completed" or result["completed_steps"] != 1000
            or result["final_val_step"] != 1000):
        raise ValueError("the intervention requires a completed 1000-update checkpoint")
    expected = dict(pool_writer="layer", pool_readout="concat")
    model_config = config["model_config"]
    if (any(model_config.get(k) != v for k, v in expected.items())
            or model_config["pool_reads"] != model_config["pool_banks"]):
        raise ValueError("the intervention requires dense concatenated layer-owned banks")
    for relative in ("gated_delta_pool.py", "gated_delta_model.py", "gated_delta_ops.py",
                     "gated_delta_runtime.py", "nanogpt_mini_model.py", "recurrent_slots_runtime.py"):
        current = ROOT / "pretraining/nanogpt_mini" / relative
        if digest(current) != digest(run / relative):
            raise ValueError(f"checkpoint source changed: {relative}")
    timing = Path(config["throughput_gate"]["path"])
    if digest(timing) != config["throughput_gate"]["sha256"]:
        raise ValueError("bound throughput report changed")
    evidence = json.loads(timing.read_text())
    for relative, expected_digest in evidence["source_sha256"].items():
        if digest(ROOT / relative) != expected_digest:
            raise ValueError(f"bound model/kernel source changed: {relative}")
    from scripts.train_recurrent_slots import PackedBatches, atomic_json
    from train_gpt import build_sentencepiece_luts
    from pretraining.nanogpt_mini.gated_delta_pool import SharedPoolGatedDeltaGPT
    from pretraining.nanogpt_mini.gated_delta_runtime import compiled_gated_delta_loss, gdn2_dependency_provenance, wait_for_exclusive_gpu
    from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphValidation
    import sentencepiece as spm

    if evidence["gdn2_dependency_provenance"] != gdn2_dependency_provenance():
        raise ValueError("installed FLA implementation changed")
    wait_for_exclusive_gpu()
    checkpoint = run / "model.pt"
    checkpoint_digest = digest(checkpoint)
    payload = torch.load(checkpoint, map_location="cuda", weights_only=True)
    if payload["completed_steps"] != 1000 or payload["model_config"] != model_config:
        raise ValueError("checkpoint and canonical metadata disagree")
    torch.manual_seed(config["seed"])
    model = SharedPoolGatedDeltaGPT(**model_config).cuda().eval()
    model.load_state_dict(payload["model"], strict=True)
    del payload
    data = ROOT / config["data_path"]
    loader = PackedBatches(str(data / "fineweb_val_*.bin"), config["val_tokens"], config["seq_len"])
    inputs, targets = loader.next()
    del loader
    tokenizer_path = ROOT / config["tokenizer"]
    if digest(tokenizer_path) != config["tokenizer_sha256"]:
        raise ValueError("checkpoint tokenizer changed")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    base, leading, boundary = build_sentencepiece_luts(tokenizer, vocab_size=model_config["vocab_size"], device="cuda")
    byte_counts = base[targets].long() + (leading[targets] & ~boundary[inputs.long()]).long()
    val_bytes = int(byte_counts.sum())
    if val_bytes != config["val_bytes"]:
        raise ValueError("validation byte panel differs from training")
    del base, leading, boundary, byte_counts
    out = ROOT / "ablation_results" / args.name
    out.mkdir(exist_ok=False)
    (out / Path(__file__).name).write_text(Path(__file__).read_text())
    atomic_json(out / "config.json", dict(run=args.run, checkpoint_sha256=checkpoint_digest,
                                          intervention="zero_off_diagonal_source_channels_no_renormalization",
                                          compare_query_shift=args.compare_query_shift,
                                          microbatch=config["microbatch"], val_tokens=config["val_tokens"],
                                          val_bytes=val_bytes, seq_len=config["seq_len"]))
    observations = []
    readouts = [block.pool_o_proj for block in model.blocks]
    modes = ["full", "lattice_only", "own_bank_only"]
    if args.compare_query_shift:
        modes.append("query_shift")
        source = ROOT / "pretraining/nanogpt_mini/gated_delta_dense_reads.py"
        (out / source.name).write_bytes(source.read_bytes())
    for mode in modes:
        if mode == "query_shift":
            from pretraining.nanogpt_mini.gated_delta_dense_reads import QueryShiftSharedPoolGPT
            for block, readout in zip(model.blocks, readouts):
                block.pool_o_proj = readout
            shifted = QueryShiftSharedPoolGPT(**model_config).cuda().eval()
            shifted.load_state_dict(model.state_dict(), strict=True)
            model = shifted
        elif mode != "full":
            for block, readout in zip(model.blocks, readouts):
                block.pool_o_proj = SourceMaskReadout(readout, model_config["pool_banks"], block.layer_index,
                                                     include_lower=mode == "lattice_only")
        loss = compiled_gated_delta_loss(model, config["segment_size"])
        graph = CUDAGraphValidation(loss, batch_size=config["microbatch"], seq_len=config["seq_len"])
        nll = torch.zeros((), device="cuda", dtype=torch.float64)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, y in zip(inputs.split(config["microbatch"]), targets.split(config["microbatch"])):
                nll += graph.replay(x, y)[0].double()
        loss.audit_graph_breaks()
        bpb = float(nll) / (math.log(2) * val_bytes)
        if not math.isfinite(bpb):
            raise FloatingPointError("nonfinite intervention BPB")
        if mode == "full" and abs(bpb - result["final_val_bpb"]) > 0.0005:
            raise ValueError("unmodified checkpoint does not reproduce canonical BPB")
        observation = dict(type="execution_comparison" if mode == "query_shift" else "state_intervention",
                           mode=mode, step=1000, val_bpb=bpb)
        observations.append(observation)
        with (out / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(observation, allow_nan=False) + "\n")
        print(json.dumps(observation), flush=True)
        del graph, loss
    summary = dict(status="completed", run=args.run, observations=observations,
                                          cross_layer_removal_cost_bpb=observations[2]["val_bpb"] - observations[0]["val_bpb"],
                                          feedback_removal_cost_bpb=observations[1]["val_bpb"] - observations[0]["val_bpb"],
                                          caveat="removal also changes readout amplitude; this is checkpoint dependence, not a retrained control")
    if args.compare_query_shift:
        summary["query_shift_bpb_delta"] = observations[3]["val_bpb"] - observations[0]["val_bpb"]
        summary["query_shift_bpb_reproduced"] = abs(summary["query_shift_bpb_delta"]) < .0005
    atomic_json(out / "result.json", summary)


if __name__ == "__main__":
    main()

"""Fixed-checkpoint memory interventions; execute through mlq, never train.

All three arms score the same 1,048,576-token FineWeb panel with 1024-token
reset windows. Slot averaging removes content diversity while retaining slot
identities. Zeroing removes history while retaining the learned static reader.
These interventions diagnose memory use; they are not quality improvements or
comparable training-speed measurements (the compiler can prune erased writes).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sentencepiece as spm
import torch

from pretraining.nanogpt_mini.recurrent_slots import RecurrentSlots
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphValidation, RecurrentLoss
from scripts.train_recurrent_slots import PackedBatches, atomic_json, build_sentencepiece_luts


class MemoryIntervention(RecurrentSlots):
    """Change only the memory made visible to the next token."""

    def __init__(self, *, intervention: str, **config):
        if intervention not in {"normal", "slotmean", "zero"}:
            raise ValueError(f"Unknown memory intervention: {intervention}")
        super().__init__(**config)
        self.intervention = intervention

    def step(self, tokens, memory):
        hidden, updated = super().step(tokens, memory)
        if self.intervention == "slotmean":
            updated = updated.mean(dim=1, keepdim=True).expand_as(updated)
        elif self.intervention == "zero":
            updated = torch.zeros_like(updated)
        return hidden, updated


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("ablation_results/nanomini_recurrent_slots_graph_1k/model.pt"))
    parser.add_argument("--data-path", type=Path, default=Path("data/datasets/fineweb10B_sp1024"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/fineweb_1024_bpe.model"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--segment-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or 1024 % args.batch_size:
        parser.error("--batch-size must divide the 1024 validation rows")
    if args.segment_size < 1 or 1024 % args.segment_size:
        parser.error("--segment-size must divide the full 1024-token context")
    return args


def evaluate_arm(arm, checkpoint, inputs, targets, byte_count, *, batch_size, segment_size):
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = MemoryIntervention(intervention=arm, **checkpoint["model_config"]).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    loss_fn = RecurrentLoss(model, segment_size=segment_size)
    print(json.dumps({"event": "prepare", "arm": arm, "batch_size": batch_size}), flush=True)
    graph = CUDAGraphValidation(loss_fn, batch_size=batch_size, seq_len=1024,
                               vocab_size=checkpoint["model_config"]["vocab_size"])
    try:
        torch.cuda.synchronize()
        evaluation_start = time.perf_counter()
        sums = torch.zeros(3, device="cuda", dtype=torch.float64)
        with torch.no_grad():
            for row in range(0, len(inputs), batch_size):
                loss, memory_square, slot_variance = graph.replay(
                    inputs[row:row + batch_size], targets[row:row + batch_size])
                # Consume graph-output aliases before the next replay.
                sums += torch.stack((loss, memory_square, slot_variance)).double()
        total_nll, memory_square_sum, slot_variance_sum = sums.tolist()
        if not all(math.isfinite(value) for value in (total_nll, memory_square_sum, slot_variance_sum)):
            raise FloatingPointError(f"Non-finite {arm} diagnostic")
        batches = len(inputs) // batch_size
        result = {
            "arm": arm, "val_loss": total_nll / targets.numel(),
            "val_bpb": total_nll / (math.log(2) * byte_count),
            "val_memory_rms": math.sqrt(memory_square_sum / batches),
            "val_memory_slot_std": math.sqrt(slot_variance_sum / batches),
            "evaluation_seconds": time.perf_counter() - evaluation_start,
            "graph_preparation_seconds": graph.preparation_seconds,
            "graph_capture_seconds": graph.capture_seconds,
            "arm_seconds": time.perf_counter() - start,
            "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        return result
    finally:
        torch.cuda.synchronize()
        graph.graph.reset()


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This diagnostic requires CUDA with bf16 support; submit through mlq")
    torch.cuda.set_device(0)
    checkpoint_path, data_path, tokenizer_path, output = (
        resolve(path) for path in (args.checkpoint, args.data_path, args.tokenizer, args.output))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "nanogpt_mini_recurrent_slots_v1":
        raise ValueError("Checkpoint is not the exact recurrent-slot architecture")
    if checkpoint.get("train_seq_len") != 1024 or checkpoint["model_config"]["vocab_size"] != 1024:
        raise ValueError("Diagnostic requires the trained 1024-token/SP1024 contract")
    output.mkdir(parents=True, exist_ok=False)
    wall_start = time.perf_counter()
    report = {
        "status": "running", "diagnostic": "fixed_checkpoint_post_write_memory_interventions",
        "optimizer_updates": 0, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_completed_steps": checkpoint.get("completed_steps"),
        "model_config": checkpoint["model_config"], "data_path": str(data_path),
        "tokenizer": str(tokenizer_path), "tokenizer_sha256": sha256(tokenizer_path),
        "batch_size": args.batch_size, "segment_size": args.segment_size,
        "seq_len": 1024, "val_tokens": 1048576, "memory_reset": "every_packed_row",
        "precision": "bf16_compute_and_embedding_fp32_other_parameters",
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "interventions": {
            "normal": "unchanged recurrent memory",
            "slotmean": "broadcast mean slot content after every write; retain distinct learned slot identities",
            "zero": "zero content after every write; retain static learned slot identities",
        },
        "interpretation": "Within-checkpoint intervention evidence, not training improvement; erasing writes can change compiled workload.",
        "arms": {},
    }
    sources = (Path(__file__), ROOT / "pretraining/nanogpt_mini/recurrent_slots.py",
               ROOT / "pretraining/nanogpt_mini/recurrent_slots_runtime.py",
               ROOT / "pretraining/nanogpt_mini/nanogpt_mini_model.py",
               ROOT / "scripts/train_recurrent_slots.py", ROOT / "train_gpt.py")
    report["source_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    (output / Path(__file__).name).write_text(Path(__file__).read_text())
    atomic_json(output / "report.json", report)
    try:
        loader = PackedBatches(str(data_path / "fineweb_val_*.bin"), 1048576, 1024)
        inputs, targets = loader.next()
        report["validation_shard"] = str(loader.files[loader.shard_index])
        del loader
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        if tokenizer.vocab_size() != 1024:
            raise ValueError("Expected the 1024-piece challenge tokenizer")
        base, leading, boundary = build_sentencepiece_luts(tokenizer, vocab_size=1024, device="cuda")
        counts = base[targets].long()
        counts += (leading[targets] & ~boundary[inputs.long()]).long()
        byte_count = int(counts.sum())
        if byte_count <= 0:
            raise ValueError("Validation panel contains no bytes")
        report["val_bytes"] = byte_count
        del base, leading, boundary, counts

        metrics_path = checkpoint_path.parent / "metrics.jsonl"
        if metrics_path.exists():
            logged = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
            matches = [entry for entry in logged if entry.get("type") == "val"
                       and entry.get("step") == checkpoint.get("completed_steps")]
            if matches:
                report["checkpoint_logged_val_bpb"] = matches[-1]["val_bpb"]
                report["checkpoint_logged_metric_source"] = str(metrics_path)

        for arm in ("normal", "slotmean", "zero"):
            result = evaluate_arm(arm, checkpoint, inputs, targets, byte_count,
                                  batch_size=args.batch_size, segment_size=args.segment_size)
            report["arms"][arm] = result
            result["delta_bpb_vs_normal"] = result["val_bpb"] - report["arms"]["normal"]["val_bpb"]
            if arm == "normal" and "checkpoint_logged_val_bpb" in report:
                report["normal_minus_checkpoint_logged_bpb"] = result["val_bpb"] - report["checkpoint_logged_val_bpb"]
            report["elapsed_seconds"] = time.perf_counter() - wall_start
            atomic_json(output / "report.json", report)
            print(json.dumps(result), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - wall_start
        atomic_json(output / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

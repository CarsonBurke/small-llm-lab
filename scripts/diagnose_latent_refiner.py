"""Completed-checkpoint source-use diagnostic; run only through mlq.

Scores the canonical FineWeb panel under normal and zero-gate execution.
This intervention changes a trained model's input distribution; its BPB delta
measures reliance on this path, not a retrained architecture's quality.
"""
from __future__ import annotations

import argparse
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
import torch.nn.functional as F

from pretraining.nanogpt_mini.latent_refiner import LatentRefiner
from pretraining.nanogpt_mini.latent_refiner_runtime import LatentRefinerLoss
from scripts.train_recurrent_slots import PackedBatches, atomic_json, build_sentencepiece_luts


SOURCES = (
    "scripts/diagnose_latent_refiner.py", "scripts/train_recurrent_slots.py",
    "pretraining/nanogpt_mini/latent_refiner.py",
    "pretraining/nanogpt_mini/latent_refiner_runtime.py",
    "pretraining/nanogpt_mini/recurrent_slots_runtime.py",
    "pretraining/nanogpt_mini/nanogpt_mini_model.py", "train_gpt.py",
)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("data/datasets/fineweb10B_sp1024"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/fineweb_1024_bpe.model"))
    return parser.parse_args(argv)


class DiagnosticStages:
    def __init__(self, model):
        self.model = model
        self.stage = torch.compile(self._stage, fullgraph=True, dynamic=False)
        self.head_loss = LatentRefinerLoss(model, segment_size=16).head_loss

    def _stage(self, inputs):
        u, gates = self.model.encode(inputs)
        normal, _ = self.model._parallel(u, gates)
        zero, _ = self.model._parallel(u, torch.zeros_like(gates))
        # Statistics exclude position zero, where all sources are explicitly zero.
        source = u[:, 1:] if self.model.config["source"] == "current" else u[:, :-1]
        current, selected_gates = u[:, 1:], gates[:, 1:]
        injection = selected_gates * source
        cosine = F.cosine_similarity(source, current, dim=-1, eps=1e-8)
        statistics = torch.stack((selected_gates.sum(), selected_gates.square().sum(),
                                  injection.square().sum(), current.square().sum(), cosine.sum()))
        return normal, zero, statistics

    def __call__(self, inputs, targets):
        normal, zero, stats = self.stage(inputs)
        normal_loss, zero_loss = [], []
        for h, z, y in zip(normal.flatten(0, 1).split(4096), zero.flatten(0, 1).split(4096),
                           targets.flatten().split(4096)):
            normal_loss.append(self.head_loss(h, y))
            zero_loss.append(self.head_loss(z, y))
        return torch.stack(normal_loss).sum(), torch.stack(zero_loss).sum(), stats


def main(argv=None):
    args = parse_args(argv)
    run, data, tokenizer_path = map(resolve, (args.run_dir, args.data_path, args.tokenizer))
    if not run.is_relative_to((ROOT / "ablation_results").resolve()):
        raise ValueError("--run-dir must be a canonical ablation_results directory")
    output = run / "diagnostic.json"
    if output.exists():
        raise FileExistsError(f"Diagnostic already exists: {output}")
    if data != resolve("data/datasets/fineweb10B_sp1024") or tokenizer_path != resolve("data/tokenizers/fineweb_1024_bpe.model"):
        raise ValueError("This diagnostic requires the canonical data and tokenizer paths")
    checkpoint_path = run / "model.pt"
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training = json.loads((run / "config.json").read_text())
    result = json.loads((run / "result.json").read_text())
    config = checkpoint["model_config"]
    source = config.get("source")
    if source not in {"current", "encoder"} or checkpoint.get("architecture") != f"refiner_{source}":
        raise ValueError("Only completed current/encoder latent-refiner checkpoints are supported")
    if checkpoint.get("completed_steps") != 1000 or checkpoint.get("train_seq_len") != 1024:
        raise ValueError("Checkpoint must contain exactly 1000 completed updates at context1024")
    if (result.get("status") != "completed" or result.get("completed_steps") != 1000
            or result.get("checkpoint_completed_steps") != 1000 or result.get("final_val_step") != 1000
            or result.get("architecture") != checkpoint["architecture"]):
        raise ValueError("Canonical training result does not confirm this completed checkpoint")
    expected = dict(vocab_size=1024, encoder_layers=4, model_dim=512, refiner_hidden=2048, gate_init=.05)
    if any(config.get(key) != value for key, value in expected.items()) or training.get("model_config") != config:
        raise ValueError("Model configuration differs from the intended source-control panel")
    if any(training.get(key) != value for key, value in dict(val_tokens=1048576, val_bytes=2524883,
                                                            seq_len=1024, validation_microbatch=64,
                                                            loss_reduction="sum").items()):
        raise ValueError("Training validation protocol differs from canonical panel")
    tokenizer_hash = sha256(tokenizer_path)
    if training.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Tokenizer differs from the trained checkpoint's tokenizer")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required; submit this workload through mlq")
    torch.cuda.set_device(0)
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    started = time.perf_counter()
    report = dict(status="running", diagnostic="latent_refiner_source_use", optimizer_updates=0,
                  checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                  checkpoint_completed_steps=1000, architecture=checkpoint["architecture"], model_config=config,
                  val_tokens=1048576, val_bytes=2524883, seq_len=1024, batch_size=64,
                  data_path=str(data), tokenizer=str(tokenizer_path), tokenizer_sha256=tokenizer_hash,
                  validation_window="first1048577_shard_tokens; shifted1048576targets;1024token_reset_rows",
                  statistics_window="exclude first position in every row",
                  gate_statistics="learned gates under normal execution; population std across token/channel entries",
                  precision="BF16 dense compute; FP32 encoder/refiner states and sigmoid",
                  intervention="set all gates to zero; encoder and refiner parameters unchanged",
                  interpretation="within-checkpoint distribution-shift diagnostic, not a quality improvement",
                  training_final_val_bpb=result["final_val_bpb"], normal_agreement_tolerance_bpb=1e-4,
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  source_sha256={name: sha256(ROOT / name) for name in SOURCES})
    report["training_source_hash_matches"] = {
        name: report["source_sha256"][name] == training.get("source_sha256", {}).get(name)
        for name in SOURCES if name in training.get("source_sha256", {})}
    snapshot = run / "diagnostic_source"
    snapshot.mkdir(exist_ok=False)
    for name in SOURCES:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    atomic_json(output, report)
    try:
        loader = PackedBatches(str(data / "fineweb_val_*.bin"), 1048576, 1024)
        inputs, targets = loader.next()
        shard = loader.files[loader.shard_index]
        report["validation_shard"] = str(shard)
        report["validation_shard_sha256"] = sha256(shard)
        del loader
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        if tokenizer.vocab_size() != 1024:
            raise ValueError("Expected 1024-piece tokenizer")
        base, leading, boundary = build_sentencepiece_luts(tokenizer, vocab_size=1024, device="cuda")
        byte_counts = base[targets].long() + (leading[targets] & ~boundary[inputs.long()]).long()
        if int(byte_counts.sum()) != 2524883:
            raise ValueError("Validation byte count differs from the canonical panel")
        del byte_counts, base, leading, boundary
        model = LatentRefiner(**config).cuda().eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        stages = DiagnosticStages(model)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            # Compilation preparation is excluded from evaluation timing/counts.
            stages(inputs[:64], targets[:64])
            torch.cuda.synchronize()
            prepared = time.perf_counter()
            sums = torch.zeros(7, device="cuda", dtype=torch.float64)
            for row in range(0, inputs.shape[0], 64):
                normal, zero, stats = stages(inputs[row:row + 64], targets[row:row + 64])
                sums += torch.cat((torch.stack((normal, zero)), stats)).double()
            totals = sums.tolist()
        if not all(math.isfinite(value) for value in totals):
            raise FloatingPointError("Nonfinite diagnostic output")
        normal, zero, gate_sum, gate_square, injection_square, u_square, cosine_sum = totals
        positions = 1024 * 1023
        elements = positions * config["model_dim"]
        gate_mean = gate_sum / elements
        normal_bpb, zero_bpb = (loss / (math.log(2) * 2524883) for loss in (normal, zero))
        difference = normal_bpb - result["final_val_bpb"]
        report.update(normal_bpb=normal_bpb, zero_source_bpb=zero_bpb,
                      zero_source_minus_normal_bpb=zero_bpb - normal_bpb,
                      gate_mean=gate_mean, gate_std=math.sqrt(max(0., gate_square / elements - gate_mean**2)),
                      injection_rms_over_encoder_rms=math.sqrt(injection_square / u_square),
                      mean_source_encoder_cosine=cosine_sum / positions,
                      normal_minus_training_bpb=difference,
                      normal_agrees_with_training=abs(difference) <= 1e-4,
                      preparation_seconds=prepared - started, evaluation_seconds=time.perf_counter() - prepared)
        if sha256(checkpoint_path) != checkpoint_hash:
            raise RuntimeError("Checkpoint changed during diagnostic execution")
        report["status"] = "completed" if report["normal_agrees_with_training"] else "validation_mismatch"
        print(json.dumps(report), flush=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(output, report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

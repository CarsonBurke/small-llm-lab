"""Matched 1000-update diagnostics for latent-state reuse and refinement.

Submit through mlq. Quality diagnostics are explicitly allowed even when their
throughput is lower than plain mini. Promotion separately requires a verified
5% throughput gain and >0.005 BPB gain against the recorded control.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts.train_recurrent_slots import (
    MetricsWriter, Muon, PackedBatches, Stagnation, atomic_json, atomic_save,
    build_sentencepiece_luts, matched_reference,
)
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch, CUDAGraphValidation

ARCHITECTURES = ("carry", "narrow", "refiner_current", "refiner_encoder", "refiner_refined")


def default_microbatch(architecture):
    return 512 if architecture == "carry" else 128 if architecture.startswith("refiner_") else 64


def validate_microbatch(architecture, microbatch):
    allowed = (64,) if architecture == "narrow" else (64, 128, 256) if architecture.startswith("refiner_") else (64, 512)
    if microbatch not in allowed:
        raise ValueError(f"{architecture} requires microbatch in {allowed}")


SOURCE_PATHS = (
    "scripts/train_latent_carry.py", "scripts/benchmark_latent_carry.py",
    "scripts/train_recurrent_slots.py", "scripts/benchmark_chunk_memory.py",
    "pretraining/nanogpt_mini/latent_carry.py",
    "pretraining/nanogpt_mini/latent_carry_runtime.py",
    "pretraining/nanogpt_mini/latent_refiner.py",
    "pretraining/nanogpt_mini/latent_refiner_runtime.py",
    "pretraining/nanogpt_mini/narrow_softmax_model.py",
    "pretraining/nanogpt_mini/nanogpt_mini_model.py",
    "pretraining/nanogpt_mini/chunk_memory_runtime.py",
    "pretraining/nanogpt_mini/recurrent_slots_runtime.py",
)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def make_model_and_loss(architecture):
    if architecture == "carry":
        from pretraining.nanogpt_mini.latent_carry import FinalStateDeltaCarry
        from pretraining.nanogpt_mini.latent_carry_runtime import LatentCarryLoss
        model = FinalStateDeltaCarry().cuda()
        loss = LatentCarryLoss(model, segment_size=16)
    elif architecture == "narrow":
        from pretraining.nanogpt_mini.narrow_softmax_model import NarrowSoftmaxGPT
        from pretraining.nanogpt_mini.latent_carry_runtime import CausalControlLoss
        model = NarrowSoftmaxGPT().cuda()
        loss = CausalControlLoss(model, segment_size=16)
    elif architecture in ARCHITECTURES and architecture.startswith("refiner_"):
        from pretraining.nanogpt_mini.latent_refiner import LatentRefiner
        from pretraining.nanogpt_mini.latent_refiner_runtime import LatentRefinerLoss
        source = architecture.removeprefix("refiner_")
        model = LatentRefiner(source=source).cuda()
        model.compile_segments()
        loss = LatentRefinerLoss(model, segment_size=16)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model, loss


def make_optimizers(model):
    special = {id(model.embed.weight), id(model.proj.weight)}
    scalars, matrices = [], []
    for name, parameter in model.named_parameters():
        if id(parameter) in special:
            continue
        if parameter.ndim < 2:
            scalars.append(parameter)
        elif parameter.ndim == 2:
            matrices.append(parameter)
        else:
            raise ValueError(f"Unclassified optimizer parameter: {name}")
    adam = torch.optim.AdamW([
        {"params": [model.embed.weight], "lr": 0.7},
        {"params": [model.proj.weight], "lr": 0.004},
        {"params": scalars, "lr": 0.015},
    ], betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
    result = [adam, Muon(matrices)]
    parameters = [p for optimizer in result for group in optimizer.param_groups for p in group["params"]]
    if len(parameters) != len(set(parameters)) or set(parameters) != set(model.parameters()):
        raise ValueError("Optimizers must cover all parameters exactly once")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--architecture", required=True, choices=ARCHITECTURES)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--val-every", default=20, type=int)
    parser.add_argument("--microbatch", type=int, choices=(64, 128, 256, 512))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--data-path", default="data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer", default="data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--throughput-report", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("--name must be one directory component")
    if args.steps != 1000 or args.val_every != 20:
        parser.error("Matched diagnostic requires --steps 1000 --val-every 20")
    if args.microbatch is None:
        args.microbatch = default_microbatch(args.architecture)
    try:
        validate_microbatch(args.architecture, args.microbatch)
    except ValueError as error:
        parser.error(str(error))
    return args


def throughput_evidence(args, model_config):
    """Require completed matching benchmark evidence; slower models remain diagnostic."""
    if args.throughput_report is None:
        raise ValueError("A completed throughput benchmark is required for diagnosis")
    import statistics
    path = args.throughput_report
    if not path.is_absolute():
        path = ROOT / path
    raw = path.read_bytes()
    report = json.loads(raw)
    required = {"status": "completed", "architecture": args.architecture,
                "batch_tokens": 524288, "seq_len": 1024, "optimizer_included": True,
                "compiled": True, "cuda_graph": True}
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError("Throughput evidence does not match complete diagnostic updates")
    if report.get("source_sha256") != source_hashes():
        raise ValueError("Throughput evidence source differs")
    if report.get("gpu") != torch.cuda.get_device_name() or report.get("torch") != str(torch.__version__):
        raise ValueError("Throughput evidence hardware/software differs")
    candidate, baseline = report["candidate"], report["baseline"]
    if report.get("repeats", 0) < 5 or candidate.get("model") != args.architecture:
        raise ValueError("Throughput evidence architecture or repetition count differs")
    if (candidate.get("model_config") != json.loads(json.dumps(model_config))
            or candidate.get("microbatch") != args.microbatch
            or baseline.get("microbatch") != 64 or baseline.get("model") != "nanogpt_mini"):
        raise ValueError("Throughput evidence model or microbatch differs")
    for arm in (candidate, baseline):
        times = arm.get("update_seconds", [])
        if (len(times) < 5 or arm.get("warmup_optimizer_updates", 0) < 5
                or arm.get("measured_optimizer_updates", 0) != len(times)
                or not all(isinstance(t, (int, float)) and math.isfinite(t) and t > 0 for t in times)):
            raise ValueError("Throughput evidence needs five finite complete update samples")
        expected = 524288 / statistics.median(times)
        if not math.isclose(arm["tokens_per_second"], expected, rel_tol=1e-6):
            raise ValueError("Throughput rate disagrees with measured update times")
    ratio = candidate["tokens_per_second"] / baseline["tokens_per_second"]
    passed = ratio >= 1.05 and max(candidate["update_seconds"]) < min(baseline["update_seconds"])
    return dict(available=True, speed_passed=passed, speedup=ratio, path=str(path),
                sha256=hashlib.sha256(raw).hexdigest(),
                candidate_tokens_per_second=candidate["tokens_per_second"],
                baseline_tokens_per_second=baseline["tokens_per_second"])


def promotion_status(status, completed_updates, bpb, reference_bpb, budget_matched, throughput):
    quality_passed = (status == "completed" and completed_updates == 1000 and bpb is not None
                      and budget_matched and reference_bpb - bpb > 0.005)
    return dict(quality_passed=quality_passed, speed_passed=throughput["speed_passed"],
                retained=bool(quality_passed and throughput["speed_passed"]))


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Diagnostics require CUDA bf16")
    import sentencepiece as spm
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    model, loss_fn = make_model_and_loss(args.architecture)
    optimizers = make_optimizers(model)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    speed = throughput_evidence(args, model.config)
    data_path, tokenizer_path = Path(args.data_path), Path(args.tokenizer)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not tokenizer_path.is_absolute():
        tokenizer_path = ROOT / tokenizer_path
    reference_bpb, same_execution, reference_metadata = matched_reference(args, data_path, tokenizer_path)
    fields = reference_metadata["matched_fields"]
    budget_matched = all(value for key, value in fields.items() if key != "microbatch")
    if not budget_matched:
        raise ValueError("Diagnostic data/tokenizer/seed/token budget differs from reference")
    out = ROOT / "ablation_results" / args.name
    if (ROOT / "tb_logs" / args.name).exists():
        raise FileExistsError(f"TensorBoard destination already exists: {args.name}")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    train_loader = PackedBatches(str(data_path / "fineweb_train_*.bin"), 524288, 1024)
    val_loader = PackedBatches(str(data_path / "fineweb_val_*.bin"), 1048576, 1024)
    val_inputs, val_targets = val_loader.next()
    del val_loader
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.vocab_size() != 1024:
        raise ValueError("Expected 1024-piece reference tokenizer")
    base, leading, boundary = build_sentencepiece_luts(tokenizer, vocab_size=1024, device="cuda")
    byte_counts = base[val_targets].long() + (leading[val_targets] & ~boundary[val_inputs.long()]).long()
    val_bytes = int(byte_counts.sum())
    if val_bytes <= 0:
        raise ValueError("Validation has no bytes")
    del base, leading, boundary, byte_counts
    config = {**vars(args), "throughput_report": str(args.throughput_report) if args.throughput_report else None,
              "purpose": "matched_causal_quality_diagnostic", "quality_allowed_without_speed_gain": True,
              "model_config": model.config, "batch_tokens": 524288, "seq_len": 1024,
              "val_tokens": 1048576, "val_bytes": val_bytes, "loss_reduction": "sum",
              "bptt": "full_1024_without_detach", "throughput_evidence": speed,
              "reference": "nanomini_fb_lam_first_1k", "reference_bpb": reference_bpb,
              "budget_matched": budget_matched, "same_execution_microbatch": same_execution,
              "microbatch_numeric_variant": not same_execution, "reference_metadata": reference_metadata,
              "optimizer": "control_12_step_cubic_muon_adamw", "compiled": True, "cuda_graph": True,
              "precision": "bf16_compute_embedding_fp32_other_parameters",
              "carry_state_dtype": "float32" if args.architecture != "narrow" else None,
              "validation_microbatch": 64,
              "head_execution": ("compiled_checkpointed_4096_token_chunks" if args.architecture != "narrow"
                                 else "compiled_full_sum_without_checkpoint"),
              "training_time_includes_initial_graph_preparation": True,
              "plain_mini_reference_bpb": 1.3433, "plain_mini_reference_precision": "reported_4dp",
              "parameters": sum(p.numel() for p in model.parameters()),
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
              "source_sha256": source_hashes()}
    atomic_json(out / "config.json", config)
    for relative in SOURCE_PATHS:
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    writer = MetricsWriter(out / "metrics.jsonl", args.name)
    cull = Stagnation()
    training_seconds, completed = 0.0, 0
    checkpoint_steps, last_val, status = None, None, "running"
    model_path = out / "model.pt"
    training_graph, validation_graph = None, None
    torch.cuda.reset_peak_memory_stats()

    def save_result(error=None):
        bpb = last_val["val_bpb"] if last_val else None
        result = dict(name=args.name, architecture=args.architecture,
                      purpose=config["purpose"], quality_allowed_without_speed_gain=True,
                      status=status, steps=1000, completed_steps=completed,
                      checkpoint_completed_steps=checkpoint_steps, val_every=20,
                      final_val_bpb=bpb, final_val_loss=last_val["val_loss"] if last_val else None,
                      final_val_step=last_val["step"] if last_val else None,
                      elapsed_seconds=time.perf_counter() - started, training_seconds=training_seconds,
                      train_time_ms=training_seconds * 1000,
                      training_ms_per_update=training_seconds * 1000 / completed if completed else None,
                      peak_vram_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                      peak_vram_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                      serialized_model_bytes=model_path.stat().st_size if model_path.exists() else None,
                      reference_bpb=reference_bpb, plain_mini_reference_bpb=1.3433,
                      improvement_vs_lam=reference_bpb - bpb if bpb is not None else None,
                      improvement_vs_plain_mini=1.3433 - bpb if bpb is not None else None,
                      budget_matched=budget_matched,
                      same_execution_microbatch=same_execution, throughput_evidence=speed,
                      autocull=asdict(cull), error=error,
                      **promotion_status(status, completed, bpb, reference_bpb, budget_matched, speed))
        atomic_json(out / "result.json", result)

    print(json.dumps(config), flush=True)
    save_result()
    try:
        for step in range(1001):
            if step % 20 == 0:
                model.eval()
                evaluation_start = time.perf_counter()
                if validation_graph is None:
                    validation_graph = CUDAGraphValidation(loss_fn, batch_size=64)
                total = torch.zeros((), device="cuda", dtype=torch.float64)
                with torch.no_grad():
                    for row in range(0, len(val_inputs), 64):
                        total += validation_graph.replay(val_inputs[row:row + 64],
                                                         val_targets[row:row + 64])[0].double()
                nll = float(total)
                if not math.isfinite(nll):
                    raise FloatingPointError(f"Nonfinite validation loss at update {step}")
                last_val = dict(type="val", step=step, val_loss=nll / 1048576,
                                val_bpb=nll / (math.log(2) * val_bytes),
                                train_time_ms=training_seconds * 1000,
                                eval_seconds=time.perf_counter() - evaluation_start)
                writer.write_entry(last_val)
                print(json.dumps(last_val), flush=True)
                pruned = cull.update(step, last_val["val_bpb"])
                atomic_save(model_path, dict(architecture=args.architecture, model=model.state_dict(),
                                            model_config=model.config, completed_steps=completed, train_seq_len=1024))
                atomic_save(out / "training_state.pt", dict(completed_steps=completed,
                            optimizers=[o.state_dict() for o in optimizers], autocull=asdict(cull),
                            data=train_loader.state_dict(), rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(), training_seconds=training_seconds))
                checkpoint_steps = completed
                status = "completed" if step == 1000 else "pruned" if pruned else "running"
                save_result()
                if status != "running":
                    if status == "pruned":
                        record = dict(type="AUTOCULL", step=step,
                                      reason="raw_and_ema_validation_stagnation", **asdict(cull))
                        writer.write_entry(record)
                        print(json.dumps(record), flush=True)
                    break
                model.train()
            torch.cuda.synchronize()
            update_start = time.perf_counter()
            inputs, targets = train_loader.next()
            if training_graph is None:
                training_graph = CUDAGraphMicrobatch(loss_fn, batch_size=args.microbatch)
            training_graph.zero_grad()
            total = torch.zeros((), device="cuda")
            for row in range(0, 512, args.microbatch):
                total += training_graph.replay(inputs[row:row + args.microbatch], targets[row:row + args.microbatch])
            gradients = [p.grad for p in model.parameters()]
            if any(gradient is None for gradient in gradients):
                raise RuntimeError("Disconnected model parameter")
            maximum = float(torch.stack(torch._foreach_norm(gradients, float("inf"))).amax())
            train_loss = float(total) / 524288
            if not math.isfinite(maximum) or not math.isfinite(train_loss):
                raise FloatingPointError(f"Nonfinite training arithmetic at update {step + 1}")
            eta = 1.0 if step / 1000 < 0.3 else (1 - step / 1000) / 0.7
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * eta
                optimizer.step()
            completed = step + 1
            training_graph.zero_grad()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - update_start
            training_seconds += seconds
            entry = dict(type="train", step=completed, train_loss=train_loss,
                         gradient_max_abs=maximum, step_avg_ms=seconds * 1000,
                         train_time_ms=training_seconds * 1000)
            writer.write_entry(entry)
            print(json.dumps(entry), flush=True)
    except BaseException as error:
        status = "failed"
        save_result(f"{type(error).__name__}: {error}")
        raise
    finally:
        writer.close()
    return 75 if status == "pruned" else 0


if __name__ == "__main__":
    raise SystemExit(main())

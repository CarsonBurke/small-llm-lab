"""Exact fixed-slot recurrent ablation; run exclusively through mlq.

The default data, 524288-token update, 1024-token reset windows, sum-CE,
optimizer arithmetic and validation panel match nanomini_fb_lam_first_1k.
Checkpointed segments retain the complete recurrent gradient across the window;
they change activation storage, never the model's temporal dependency.

Example: mlq submit --name slots_exact_1k --cwd "$PWD" --max-parallel-runs 1
  --priority 1 -- python3 scripts/train_recurrent_slots.py --name slots_exact_1k
  --steps 1000 --val-every 20

Exit 75 means conservative held-out stagnation pruning, not a completed ablation.
Use mlq --max-attempts 1 so pruning is not retried.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import glob
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

from scripts.ablation import MetricsWriter
from train_gpt import build_sentencepiece_luts
from pretraining.nanogpt_mini.recurrent_slots_runtime import (
    CUDAGraphMicrobatch, CUDAGraphValidation, RecurrentLoss,
)


@torch.compile(fullgraph=True, dynamic=False)
def muon_update(gradient, momentum):
    """The first-only control's 12-step cubic Muon, without rank sharding."""
    momentum.lerp_(gradient, 0.05)
    update = gradient.lerp(momentum, 0.95).bfloat16()
    transpose = gradient.size(-2) > gradient.size(-1)
    if transpose:
        update = update.mT
    update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(12):
        product = update @ update.mT
        polynomial = -1.5 * product + 0.5 * product @ product
        update = 2 * update + polynomial @ update
    if transpose:
        update = update.mT
    return update * max(1, gradient.size(-2) / gradient.size(-1)) ** 0.5


class Muon(torch.optim.Optimizer):
    def __init__(self, parameters):
        parameters = sorted(parameters, key=lambda p: p.size(), reverse=True)
        super().__init__(parameters, dict(lr=0.025, weight_decay=0.05))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    raise RuntimeError("Disconnected Muon parameter")
                state = self.state[parameter]
                if not state:
                    state["momentum"] = torch.zeros_like(parameter)
                update = muon_update(parameter.grad, state["momentum"])
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])


class PackedBatches:
    """Use exactly the reference's sorted shards, skipping short shard tails."""

    def __init__(self, pattern: str, tokens_per_batch: int, seq_len: int):
        self.files = [Path(path) for path in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No data shards match {pattern}")
        self.batch_tokens = tokens_per_batch
        self.seq_len = seq_len
        self.shard_index = 0
        self.position = 0
        self._load()

    def _load(self):
        path = self.files[self.shard_index]
        header = torch.from_file(str(path), shared=False, size=256, dtype=torch.int32)
        if header[0] != 20240520 or header[1] != 1:
            raise ValueError(f"Invalid shard header: {path}")
        count = int(header[2])
        if count <= self.batch_tokens + 1:
            raise ValueError(f"Shard is too short for one reference batch: {path}")
        with path.open("rb", buffering=0) as handle:
            handle.seek(1024)
            self.tokens = torch.empty(count, dtype=torch.uint16, pin_memory=True)
            if handle.readinto(self.tokens.numpy()) != 2 * count:
                raise ValueError(f"Truncated shard: {path}")

    def next(self):
        if self.position + self.batch_tokens + 1 >= len(self.tokens):
            self.shard_index = (self.shard_index + 1) % len(self.files)
            self.position = 0
            self._load()
        buffer = self.tokens[self.position:self.position + self.batch_tokens + 1]
        self.position += self.batch_tokens
        inputs = buffer[:-1].to("cuda", dtype=torch.int32, non_blocking=True)
        targets = buffer[1:].to("cuda", dtype=torch.int64, non_blocking=True)
        return inputs.view(-1, self.seq_len), targets.view(-1, self.seq_len)

    def state_dict(self):
        return {"files": [str(path) for path in self.files],
                "shard_index": self.shard_index, "position": self.position}


@dataclass
class Stagnation:
    """Prune only if both raw and smoothed held-out BPB stop improving."""

    warmup: int = 600
    patience: int = 300
    minimum_improvement: float = 0.001
    ema: float | None = None
    best: float | None = None
    best_ema: float | None = None
    last_improvement: int = 0

    def update(self, step: int, bpb: float) -> bool:
        self.ema = bpb if self.ema is None else 0.5 * bpb + 0.5 * self.ema
        improved = False
        if self.best is None or bpb < self.best - self.minimum_improvement:
            self.best = bpb
            improved = True
        if self.best_ema is None or self.ema < self.best_ema - self.minimum_improvement:
            self.best_ema = self.ema
            improved = True
        if improved:
            self.last_improvement = step
        return step >= self.warmup and step - self.last_improvement >= self.patience


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_save(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def matched_reference(args, data_path: Path, tokenizer_path: Path, reference_path: Path | None = None):
    """Bind promotion eligibility to the completed canonical reference contract."""
    if reference_path is None:
        reference_path = ROOT / "ablation_results/nanomini_fb_lam_first_1k/result.json"
    reference = json.loads(reference_path.read_text())
    overrides = reference["overrides"]
    reference_bpb = reference["final_val_bpb"]
    validations = reference.get("val_entries", [])
    validation_complete = (
        [entry.get("step") for entry in validations] == list(range(0, 1001, 20))
        and validations[-1].get("val_bpb") == reference_bpb
    )
    reference_complete = (reference.get("steps") == reference.get("completed_steps") == 1000
                          and reference.get("val_every") == 20
                          and reference.get("returncode") == 0
                          and reference.get("training_returncode") == 0
                          and reference.get("metric_integrity_errors") == []
                          and validation_complete
                          and math.isfinite(reference_bpb))
    matched_fields = {
        "completed_reference": reference_complete,
        "data_path": data_path.resolve() == (ROOT / overrides["DATA_PATH"]).resolve(),
        "tokenizer_path": tokenizer_path.resolve() == (ROOT / overrides["TOKENIZER_PATH"]).resolve(),
        "seed": args.seed == int(overrides["SEED"]),
        "microbatch": args.microbatch == int(overrides["MBS"]),
        "vocab_size": int(overrides["VOCAB_SIZE"]) == 1024,
        "seq_len": int(overrides["SEQ_LEN"]) == 1024,
        "val_tokens": int(overrides["VAL_TOKENS"]) == 1048576,
    }
    matched_control = all(matched_fields.values())
    reference_metadata = {"path": str(reference_path.resolve()),
                          "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                          "final_val_bpb": reference_bpb, "matched_fields": matched_fields,
                          "overrides": overrides,
                          "comparison_scope": "matched_update_token_and_validation_budget",
                          "reference_microbatch": int(overrides["MBS"]),
                          "run_microbatch": args.microbatch,
                          "same_execution_microbatch": args.microbatch == int(overrides["MBS"])}

    return reference_bpb, matched_control, reference_metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--architecture", choices=("exact", "chunk", "gdn1", "gdn2"), default="exact")
    parser.add_argument("--throughput-report", help="Required passing benchmark report for chunk training")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--val-every", type=int, default=20)
    parser.add_argument("--data-path", default="data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer", default="data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--segment-size", type=int, default=16)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--memory-head-dim", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--memory-width", type=int, choices=(128, 256, 512), default=512)
    parser.add_argument("--fused-projections", action="store_true")
    parser.add_argument("--value-expansion", type=float, choices=(1.0, 2.0), default=1.0)
    parser.add_argument("--gdn-backend", choices=("vendor", "fla"), default="vendor")
    parser.add_argument("--state-v-first", action="store_true")
    parser.add_argument("--disable-recompute", action="store_true")
    parser.add_argument("--allow-slow-diagnostic", action="store_true",
                        help="Allow a GDN2 quality diagnostic with completed slower timing evidence; never waive promotion gates")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("--name must be one directory component")
    if args.steps != 1000 or args.val_every != 20:
        parser.error("This matched ablation requires --steps 1000 --val-every 20")
    if args.microbatch <= 0 or 512 % args.microbatch:
        parser.error("--microbatch must divide 512 packed training rows")
    if args.segment_size <= 0 or 1024 % args.segment_size:
        parser.error("--segment-size must divide the 1024-token context")
    if args.slots <= 0:
        parser.error("--slots must be positive")
    if args.architecture in {"gdn1", "gdn2"} and args.segment_size != 64:
        parser.error("Gated Delta uses 64-token computational kernel chunks")
    if args.architecture in {"chunk", "gdn1", "gdn2"} and not args.throughput_report:
        parser.error("this architecture requires --throughput-report from a passing benchmark")
    if args.architecture not in {"gdn1", "gdn2"} and args.fused_projections:
        parser.error("--fused-projections is specific to Gated Delta models")
    if args.architecture not in {"gdn1", "gdn2"} and args.value_expansion != 1.0:
        parser.error("--value-expansion is specific to Gated Delta models")
    if args.architecture != "gdn2" and (args.gdn_backend != "vendor" or args.state_v_first or args.disable_recompute):
        parser.error("GDN2 execution options require --architecture gdn2")
    if args.allow_slow_diagnostic and args.architecture != "gdn2":
        parser.error("--allow-slow-diagnostic is specific to GDN2")
    return args


def verify_throughput_gate(args):
    """Require measured speed, matching execution, and unchanged model sources."""
    report_path = Path(args.throughput_report).resolve()
    report = json.loads(report_path.read_text())
    candidate, baseline = report["candidate"], report["baseline"]
    speeds = (candidate["tokens_per_second"], baseline["tokens_per_second"])
    if (report.get("batch_tokens") != 524288 or report.get("seq_len") != 1024
            or report.get("optimizer_included") is not True or report.get("repeats", 0) < 5
            or baseline.get("model") != "nanogpt_mini"
            or report.get("torch") != str(torch.__version__)):
        raise ValueError("Throughput benchmark does not match the complete-update protocol")
    if (report.get("status") != "completed" or report.get("gate_passed") is not True
            or not all(math.isfinite(x) and x > 0 for x in speeds)
            or speeds[0] < 1.05 * speeds[1]):
        raise ValueError("Chunk model has not cleared the nanoGPT throughput gate")
    for key, value in {"microbatch": args.microbatch, "chunk_size": args.segment_size,
                       "slots": args.slots}.items():
        if candidate.get(key) != value:
            raise ValueError(f"Throughput benchmark differs from training: {key}")
    expected = dict(vocab_size=1024, num_layers=6, model_dim=512, head_dim=128,
                    slots=args.slots, chunk_size=args.segment_size, checkpoint_chunks=False)
    if candidate.get("model_config") != expected:
        raise ValueError("Throughput benchmark model configuration differs")
    for relative in ("pretraining/nanogpt_mini/chunk_memory.py",
                     "pretraining/nanogpt_mini/chunk_memory_runtime.py",
                     "pretraining/nanogpt_mini/nanogpt_mini_model.py",
                     "pretraining/nanogpt_mini/recurrent_slots_runtime.py",
                     "scripts/train_recurrent_slots.py", "scripts/benchmark_chunk_memory.py"):
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if report.get("source_sha256", {}).get(relative) != digest:
            raise ValueError(f"Throughput benchmark source changed: {relative}")
    return {"path": str(report_path), "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "candidate_tokens_per_second": speeds[0], "baseline_tokens_per_second": speeds[1],
            "gpu": report.get("gpu")}


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This ablation requires a CUDA GPU with bf16 support")
    from pretraining.nanogpt_mini.recurrent_slots import RecurrentSlots
    if args.architecture in {"gdn1", "gdn2"}:
        from pretraining.nanogpt_mini.gated_delta_runtime import verify_throughput_report
        throughput_gate = verify_throughput_report(args)
    else:
        throughput_gate = verify_throughput_gate(args) if args.architecture == "chunk" else None
    if throughput_gate and throughput_gate["gpu"] != torch.cuda.get_device_name():
        raise ValueError("Throughput benchmark was measured on different hardware")
    import sentencepiece as spm

    out = ROOT / "ablation_results" / args.name
    out.mkdir(parents=True, exist_ok=False)
    if (ROOT / "tb_logs" / args.name).exists():
        raise FileExistsError(f"TensorBoard run already exists: {args.name}")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    wall_start = time.perf_counter()
    if args.architecture == "gdn1":
        from pretraining.nanogpt_mini.scalar_delta_model import ScalarDeltaGPT
        model = ScalarDeltaGPT(head_dim=args.memory_head_dim, mixer_dim=args.memory_width,
                               fused_projections=args.fused_projections, expand_v=args.value_expansion).cuda()
        model_source = ROOT / "pretraining/nanogpt_mini/scalar_delta_model.py"
        architecture = "nanogpt_mini_scalar_delta_v1"
    elif args.architecture == "gdn2":
        from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT
        model = GatedDeltaGPT(head_dim=args.memory_head_dim, mixer_dim=args.memory_width,
                             fused_projections=args.fused_projections, expand_v=args.value_expansion,
                             gdn_backend=args.gdn_backend, state_v_first=args.state_v_first,
                             disable_recompute=args.disable_recompute).cuda()
        model_source = ROOT / "pretraining/nanogpt_mini/gated_delta_model.py"
        architecture = "nanogpt_mini_gated_delta_v2"
    elif args.architecture == "chunk":
        from pretraining.nanogpt_mini.chunk_memory import ChunkMemory
        model = ChunkMemory(slots=args.slots, chunk_size=args.segment_size).cuda()
        model_source = ROOT / "pretraining/nanogpt_mini/chunk_memory.py"
        architecture = "nanogpt_mini_chunk_memory_v1"
    else:
        model = RecurrentSlots(vocab_size=1024, num_layers=6, model_dim=512,
                               head_dim=128, slots=args.slots).cuda()
        model_source = ROOT / "pretraining/nanogpt_mini/recurrent_slots.py"
        architecture = "nanogpt_mini_recurrent_slots_v1"
    if args.architecture == "exact":
        model.compile_segments()
    # bf16 embedding; fp32 remaining parameters and bf16 model arithmetic,
    # matching the reference's mixed parameter precision.
    if args.architecture in {"gdn1", "gdn2"}:
        from pretraining.nanogpt_mini.gated_delta_runtime import build_gated_delta_optimizers
        optimizers = build_gated_delta_optimizers(model)
    else:
        special_ids = {id(model.embed.weight), id(model.proj.weight), id(model.slot_identity)}
        scalars = [p for p in model.parameters() if p.ndim < 2 and id(p) not in special_ids]
        matrices = [p for p in model.parameters() if p.ndim >= 2 and id(p) not in special_ids]
        adam = torch.optim.AdamW([
            {"params": [model.embed.weight], "lr": 0.7},
            {"params": [model.proj.weight], "lr": 0.004},
            {"params": scalars, "lr": 0.015},
            {"params": [model.slot_identity], "lr": 0.002, "weight_decay": 0.0},
        ], betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
        optimizers = [adam, Muon(matrices)]
    grouped = [p for opt in optimizers for group in opt.param_groups for p in group["params"]]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(model.parameters()):
        raise RuntimeError("Optimizer groups must cover every parameter exactly once")
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

    if args.architecture in {"gdn1", "gdn2"}:
        from pretraining.nanogpt_mini.gated_delta_runtime import CompiledGatedDeltaLoss
        loss_fn = CompiledGatedDeltaLoss(model, segment_size=args.segment_size)
    elif args.architecture == "chunk":
        from pretraining.nanogpt_mini.chunk_memory_runtime import CompiledFullLoss
        loss_fn = CompiledFullLoss(model, segment_size=args.segment_size)
    else:
        loss_fn = RecurrentLoss(model, segment_size=args.segment_size)
    training_graph = None
    validation_graph = None

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    train_loader = PackedBatches(str(data_path / "fineweb_train_*.bin"), 524288, 1024)
    val_loader = PackedBatches(str(data_path / "fineweb_val_*.bin"), 1048576, 1024)
    val_inputs, val_targets = val_loader.next()
    del val_loader
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_absolute():
        tokenizer_path = ROOT / tokenizer_path
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.vocab_size() != 1024:
        raise ValueError("Reference protocol requires the 1024-piece tokenizer")
    base, leading, boundary = build_sentencepiece_luts(tokenizer, vocab_size=1024, device="cuda")
    byte_counts = base[val_targets].long()
    byte_counts += (leading[val_targets] & ~boundary[val_inputs.long()]).long()
    val_bytes = int(byte_counts.sum())
    if val_bytes <= 0:
        raise ValueError("Validation has no bytes")
    del byte_counts, base, leading, boundary

    reference_bpb, matched_control, reference_metadata = matched_reference(
        args, data_path, tokenizer_path)

    config = {**vars(args), "architecture": architecture, "throughput_gate": throughput_gate,
              "model_config": model.config, "batch_tokens": 524288, "seq_len": 1024,
              "val_tokens": 1048576, "val_bytes": val_bytes,
              "loss_reduction": "sum", "bptt": "full_1024_without_detach",
              "precision": "bf16_compute_and_embedding_fp32_other_parameters", "compiled": True,
              "execution": "compiled_regions_triton_cuda_graphs" if args.architecture in {"gdn1", "gdn2"} else "serial_microbatch_cuda_graphs",
              "reference": "nanomini_fb_lam_first_1k", "reference_bpb": reference_bpb,
              "matched_control": matched_control, "reference_metadata": reference_metadata,
              "parameters": sum(p.numel() for p in model.parameters()),
              "optimizer": "control_muon_adam_gdn_conv_decay_groups" if args.architecture in {"gdn1", "gdn2"} else "control_12_step_cubic_muon_adamw",
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()}
    atomic_json(out / "config.json", config)
    for source in (Path(__file__), model_source,
                   ROOT / "pretraining/nanogpt_mini/recurrent_slots_runtime.py",
                   ROOT / "pretraining/nanogpt_mini/nanogpt_mini_model.py"):
        (out / source.name).write_text(source.read_text())
    if args.architecture == "chunk":
        source = ROOT / "pretraining/nanogpt_mini/chunk_memory_runtime.py"
        (out / source.name).write_text(source.read_text())
    if args.architecture in {"gdn1", "gdn2"}:
        import shutil
        source = ROOT / "pretraining/nanogpt_mini/gated_delta_runtime.py"
        (out / source.name).write_text(source.read_text())
        source = ROOT / "pretraining/nanogpt_mini/gated_delta_model.py"
        (out / source.name).write_text(source.read_text())
        shutil.copytree(ROOT / "pretraining/gated_delta/vendor", out / "vendor",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    writer = MetricsWriter(out / "metrics.jsonl", args.name)
    cull = Stagnation()
    training_seconds = 0.0
    step = 0
    completed_updates = 0
    checkpoint_completed_steps = None
    last_val = None
    status = "running"
    model_path = out / "model.pt"
    torch.cuda.reset_peak_memory_stats()

    def save_checkpoint():
        nonlocal checkpoint_completed_steps
        atomic_save(model_path, {"architecture": config["architecture"],
                    "model": model.state_dict(), "model_config": model.config,
                    "train_seq_len": 1024, "completed_steps": step})
        atomic_save(out / "training_state.pt", {
            "completed_steps": step, "optimizers": [o.state_dict() for o in optimizers],
            "autocull": asdict(cull), "data": train_loader.state_dict(),
            "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
            "training_seconds": training_seconds,
        })
        checkpoint_completed_steps = step

    def save_result(error=None):
        result = {"name": args.name, "steps": args.steps, "completed_steps": completed_updates,
                  "checkpoint_completed_steps": checkpoint_completed_steps,
                  "status": status, "val_every": args.val_every,
                  "final_val_bpb": last_val["val_bpb"] if last_val else None,
                  "final_val_loss": last_val["val_loss"] if last_val else None,
                  "final_val_step": last_val["step"] if last_val else None,
                  "elapsed_seconds": time.perf_counter() - wall_start,
                  "training_seconds": training_seconds,
                  "train_time_ms": training_seconds * 1000,
                  "training_ms_per_update": training_seconds * 1000 / completed_updates if completed_updates else None,
                  "serialized_model_bytes": model_path.stat().st_size if model_path.exists() else None,
                  "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                  "autocull": asdict(cull), "promotion_metric": "challenge_bpb",
                  "matched_control": matched_control, "reference_bpb": reference_bpb,
                  "speed_passed": throughput_gate.get("speed_passed", True) if throughput_gate else None,
                  "retained": bool(status == "completed" and matched_control and last_val and
                                   (not throughput_gate or throughput_gate.get("speed_passed", True)) and
                                   reference_bpb - last_val["val_bpb"] > 0.005),
                  "error": error}
        atomic_json(out / "result.json", result)

    print(json.dumps(config), flush=True)
    save_result()
    try:
        for step in range(args.steps + 1):
            if step % args.val_every == 0:
                print(f"validation_start step={step}", flush=True)
                torch.cuda.synchronize()
                validation_start = time.perf_counter()
                model.eval()
                if validation_graph is None:
                    validation_graph = CUDAGraphValidation(loss_fn, batch_size=args.microbatch)
                    if args.architecture in {"gdn1", "gdn2"}:
                        config["compile_graph_breaks"] = loss_fn.audit_graph_breaks()
                        atomic_json(out / "config.json", config)
                val_sum = torch.zeros((), device="cuda", dtype=torch.float64)
                memory_stats = torch.zeros(2, device="cuda", dtype=torch.float64)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for row in range(0, len(val_inputs), args.microbatch):
                        outputs = validation_graph.replay(
                            val_inputs[row:row + args.microbatch],
                            val_targets[row:row + args.microbatch])
                        val_sum += outputs[0].double()
                        if args.architecture not in {"gdn1", "gdn2"}:
                            memory_stats += torch.stack(outputs[1:]).double()
                total_nll = float(val_sum)
                memory_rms, slot_std = (memory_stats / (len(val_inputs) // args.microbatch)).sqrt().tolist()
                if not math.isfinite(total_nll):
                    raise FloatingPointError(f"Non-finite validation loss at update {step}")
                last_val = {"type": "val", "step": step,
                            "val_loss": total_nll / val_targets.numel(),
                            "val_bpb": total_nll / (math.log(2) * val_bytes),
                            "train_time_ms": training_seconds * 1000,
                            "eval_seconds": time.perf_counter() - validation_start,
                            "val_memory_rms": memory_rms, "val_memory_slot_std": slot_std,
                            "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
                if args.architecture in {"gdn1", "gdn2"}:
                    last_val.pop("val_memory_rms")
                    last_val.pop("val_memory_slot_std")
                writer.write_entry(last_val)
                print(json.dumps(last_val), flush=True)
                pruned = cull.update(step, last_val["val_bpb"])
                save_checkpoint()
                status = "completed" if step == args.steps else "pruned" if pruned else "running"
                save_result()
                if status != "running":
                    if status == "pruned":
                        record = {"type": "AUTOCULL", "step": step,
                                  "reason": "raw_and_ema_validation_stagnation", **asdict(cull)}
                        writer.write_entry(record)
                        print(json.dumps(record), flush=True)
                    break
                model.train()

            torch.cuda.synchronize()
            update_start = time.perf_counter()
            inputs, targets = train_loader.next()
            if training_graph is None:
                training_graph = CUDAGraphMicrobatch(loss_fn, batch_size=args.microbatch)
                if args.architecture in {"gdn1", "gdn2"}:
                    config["compile_graph_breaks"] = loss_fn.audit_graph_breaks()
                    atomic_json(out / "config.json", config)
            train_sum = torch.zeros((), device="cuda")
            for row in range(0, len(inputs), args.microbatch):
                # Consume the aliased graph output before the next replay.
                # Gradient buffers accumulate in the original microbatch order.
                train_sum += training_graph.replay(
                    inputs[row:row + args.microbatch], targets[row:row + args.microbatch])
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise RuntimeError(f"Disconnected parameter: {name}")
            gradient_max = torch.stack(torch._foreach_norm(
                [parameter.grad for parameter in model.parameters()], float("inf"))).amax()
            gradient_max = float(gradient_max)
            if not math.isfinite(gradient_max):
                raise FloatingPointError(f"Non-finite gradient at update {step + 1}")
            # Sum CE is intentionally unnormalized to match the control.
            train_loss = float(train_sum) / targets.numel()
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"Non-finite training loss at update {step + 1}")
            progress = step / args.steps
            eta = 1.0 if progress < 0.3 else (1 - progress) / 0.7
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * eta
                optimizer.step()
            completed_updates = step + 1
            training_graph.zero_grad()
            torch.cuda.synchronize()
            duration = time.perf_counter() - update_start
            training_seconds += duration
            entry = {"type": "train", "step": step + 1, "train_loss": train_loss,
                     "gradient_max_abs": gradient_max,
                     "train_time_ms": training_seconds * 1000, "step_avg_ms": duration * 1000,
                     "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
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

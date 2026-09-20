"""CELF byte control trainer with canonical metrics and exact-resume checkpoints.

Run through scripts/train_celf.py and mlq. Optionally staged: ``codec_steps``
codec-only updates (the prior is not executed) before the prior joins with
fresh optimizer state and a stage-relative schedule; ``codec_steps=0`` trains
everything jointly from initialization. Validation carries the objective terms,
the latent effective rank, teacher-forced sampled-block diagnostics, and a
fixed-anchor negative-ELBO estimate on a fixed validation subset.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from checkpointing import RecoveryCheckpointPolicy, atomic_torch_save
from pretraining.celf.config import ARCHITECTURE, LossConfig, ModelConfig
from pretraining.celf.data import ByteCorpus
from pretraining.celf.evaluation import estimate_nelbo, generate_bytes, sample_blocks
from pretraining.celf.model import CelfModel
from pretraining.celf.objective import STAT_NAMES, Objective, summarize_statistics
from pretraining.nanogpt_mini.native_bits_data import sha256_file
from pretraining.source_muon import SOURCE_MUON_ALGORITHM, Muon
from scripts.ablation import MetricsWriter, metric_integrity_errors, read_metrics_jsonl

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrainConfig:
    name: str
    data_path: str = "data/datasets/native_bits_fineweb"
    steps: int = 2000
    codec_steps: int = 0
    batch_sequences: int = 32
    microbatch: int = 8
    val_sequences: int = 40
    val_every: int = 20
    log_every: int = 10
    sample_every: int = 100
    sample_sequences: int = 16
    sample_steps: int = 8
    nelbo_every: int = 200
    nelbo_sequences: int = 16
    nelbo_steps: int = 8
    seed: int = 1337
    resume: str | None = None
    warmup_steps: int = 40
    matrix_lr: float = 0.02
    other_lr: float = 0.0003
    scalar_lr: float = 0.0003
    embedding_lr: float = 0.003

    def validate(self, model: ModelConfig):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name):
            raise ValueError("name must be a simple run directory name")
        for key in (
            "steps",
            "batch_sequences",
            "microbatch",
            "val_sequences",
            "val_every",
            "log_every",
            "sample_every",
            "sample_sequences",
            "sample_steps",
            "nelbo_every",
            "nelbo_sequences",
            "nelbo_steps",
            "warmup_steps",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if (
            self.batch_sequences % self.microbatch
            or self.val_sequences % self.microbatch
            or self.sample_sequences % self.microbatch
            or self.nelbo_sequences % self.microbatch
        ):
            raise ValueError("every sequence budget must contain complete microbatches")
        if self.sample_sequences > self.val_sequences or self.nelbo_sequences > self.val_sequences:
            raise ValueError("diagnostic subsets must lie inside the validation prefix")
        if self.sample_every % self.val_every or self.nelbo_every % self.val_every:
            raise ValueError("diagnostic cadences must be multiples of val_every")
        if not 0 <= self.codec_steps < self.steps or self.codec_steps % self.val_every:
            raise ValueError(
                "codec_steps must leave joint updates and end on a validation boundary"
            )
        if self.codec_steps and (self.codec_steps < self.warmup_steps or self.codec_steps % self.log_every):
            raise ValueError("a codec stage must outlast warmup and end on a logging boundary")
        for key in ("matrix_lr", "other_lr", "scalar_lr", "embedding_lr"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"invalid {key}")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must fit uint32")
        del model


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".working")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [
        ROOT / name
        for name in (
            "pretraining/source_muon.py",
            "pretraining/nanogpt_mini/native_bits_data.py",
            "scripts/train_celf.py",
            "scripts/ablation.py",
            "checkpointing.py",
        )
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def setup_device(seed: int):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CELF requires a BF16-capable CUDA device; no fallback")
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    return torch.device("cuda")


def parameter_counts(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "decoder": sum(p.numel() for p in model.decoder.parameters()),
        "prior": sum(p.numel() for p in model.prior.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def component_parameters(model):
    """Codec (encoder + decoder) and prior parameter lists, each partitioned once."""
    muon_ids = {id(p) for p in model.muon_parameters()}
    codec = list(model.encoder.parameters()) + list(model.decoder.parameters())
    prior = list(model.prior.parameters())
    return {
        "codec": (codec, [p for p in codec if id(p) in muon_ids]),
        "prior": (prior, [p for p in prior if id(p) in muon_ids]),
    }


def make_optimizers(model, config, component):
    parameters, matrices = component_parameters(model)[component]
    matrices = [p for p in matrices if p.requires_grad]
    matrix_ids = {id(p) for p in matrices}
    embeddings = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding)
    }
    groups = {"embedding": [], "scalar": [], "other": []}
    for parameter in parameters:
        if not parameter.requires_grad or id(parameter) in matrix_ids:
            continue
        category = (
            "embedding"
            if id(parameter) in embeddings
            else "scalar"
            if parameter.ndim < 2
            else "other"
        )
        groups[category].append(parameter)
    adam_groups = [
        {
            "params": parameters,
            "lr": getattr(config, f"{name}_lr"),
            "weight_decay": 0.0 if name == "scalar" else 0.001,
        }
        for name, parameters in groups.items()
        if parameters
    ]
    optimizers = [
        Muon(matrices, lr=config.matrix_lr, weight_decay=0.05, mu=0.95),
        torch.optim.AdamW(adam_groups, betas=(0.8, 0.95), eps=1e-8, fused=True),
    ]
    assigned = [p for opt in optimizers for group in opt.param_groups for p in group["params"]]
    expected = {p for p in parameters if p.requires_grad}
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise RuntimeError("optimizers must partition every trainable parameter exactly once")
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers


def stage_of(step: int, config: TrainConfig) -> tuple[int, int, int]:
    """(stage id, step within stage, stage length); stage 1 is codec-only."""
    if step < config.codec_steps:
        return 1, step, config.codec_steps
    return 2, step - config.codec_steps, config.steps - config.codec_steps


def schedule_multiplier(step: int, config: TrainConfig):
    """Warmup then hold for the codec stage; warmup, hold, cooldown for the joint stage."""
    stage, stage_step, stage_steps = stage_of(step, config)
    warmup = min(1.0, (stage_step + 1) / config.warmup_steps)
    if stage == 1:
        return warmup
    progress = stage_step / stage_steps
    cooldown = 1.0 if progress < 0.3 else 1.0 - 0.95 * (progress - 0.3) / 0.7
    return min(warmup, cooldown)


@torch.compile(fullgraph=True, dynamic=False)
def finite_gradients(gradients):
    return torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()


def active_components(step: int, config: TrainConfig) -> tuple[str, ...]:
    return ("codec",) if stage_of(step, config)[0] == 1 else ("codec", "prior")


def optimizer_step(model, optimizers, config, step):
    """Step the components active at ``step``; every active parameter must have a gradient."""
    components = component_parameters(model)
    active = active_components(step, config)
    parameters = [p for name in active for p in components[name][0] if p.requires_grad]
    if any(p.grad is None for p in parameters):
        wanted = {id(p) for p in parameters}
        missing = [
            name for name, p in model.named_parameters() if id(p) in wanted and p.grad is None
        ]
        raise RuntimeError(f"trainable parameters disconnected from objective: {missing}")
    if not finite_gradients([p.grad for p in parameters]).item():
        raise FloatingPointError("non-finite CELF gradients; no clipping or silent repair")
    eta = schedule_multiplier(step, config)
    stage_step = stage_of(step, config)[1]
    for name in active:
        for optimizer in optimizers[name]:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * eta
                if "mu" in group:
                    group["mu"] = 0.85 + 0.1 * min(1.0, stage_step / 300)
            optimizer.step()


CHECKPOINT_VERSION = 2  # 2: per-component optimizer dict, codec_steps, attachment


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("architecture") != ARCHITECTURE or payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("not a compatible CELF checkpoint")
    return payload


def build_training_state(config, model_config, loss_config, device, data_metadata):
    model = CelfModel(model_config).to(device)
    previous = load_checkpoint(config.resume) if config.resume else None
    if previous is not None:
        if previous["model_config"] != asdict(model_config) or previous["loss_config"] != asdict(loss_config):
            raise ValueError("resume model/objective differs from checkpoint")
        if previous["source_hashes"] != source_hashes() or previous["data"] != data_metadata:
            raise ValueError("resume source or data fingerprint differs")
        for key, value in asdict(config).items():
            if key not in {"name", "resume"} and previous["train_config"][key] != value:
                raise ValueError(f"resume training setting differs: {key}")
        if (
            not 0 <= previous["step"] <= config.steps
            or previous["data_cursor"] != previous["step"] * config.batch_sequences * model_config.seq_bytes
        ):
            raise ValueError("invalid saved update or consumed-byte cursor")
        model.load_state_dict(previous["model"], strict=True)
    optimizers = {
        name: make_optimizers(model, config, name) for name in ("codec", "prior")
    }
    if previous is not None:
        for name, group in optimizers.items():
            for optimizer, state in zip(group, previous["optimizers"][name], strict=True):
                optimizer.load_state_dict(state)
    model.compile_components()
    if previous is not None:
        torch.set_rng_state(previous["rng_cpu"])
        torch.cuda.set_rng_state(previous["rng_cuda"])
    return model, optimizers, previous


def emit(writer, log, step, config, metrics, training_ms, validation, stage_id):
    """``stage_id`` is the stage that produced ``metrics``: for a train entry the
    stage of the update just applied (``step - 1``), for a val entry that of ``step``."""
    prefix = "val" if validation else "train"
    entry = {
        "type": prefix,
        "step": step,
        f"{prefix}_loss": metrics["objective"],
        "train_time_ms": training_ms,
        "stage_id": stage_id,
        "step_avg_ms": training_ms / max(1, step),
        **{f"{prefix}_{key}": value for key, value in metrics.items()},
    }
    writer.write_entry(entry)
    first = f"step:{step}/{config.steps} {prefix}_loss:{metrics['objective']:.9g}"
    if validation:
        first += f" val_objective:{metrics['objective']:.9g}"
    first += f" train_time:{training_ms:.3f}ms"
    extras = " ".join(
        f"{prefix}_{key}:{value:.9g}" for key, value in metrics.items() if key != "objective"
    )
    line = first + " " + extras
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()
    return entry


@torch.no_grad()
def evaluate_objective(model, objective, validation, config, *, flow=True):
    model.eval()
    generator = torch.Generator(device=validation.device).manual_seed(config.seed + 100000)
    sums = torch.zeros(len(STAT_NAMES), device=validation.device)
    for ids in validation.split(config.microbatch):
        _, stats = objective(ids, generator, flow=flow)
        sums.add_(stats)
    model.train()
    return summarize_statistics(sums.cpu().tolist())


def evaluate_samples(model, validation, config, *, sequences=None):
    subset = validation[: config.sample_sequences if sequences is None else sequences]
    totals = {"bytes": 0, "sampled_nll_nats": 0.0, "correct": 0.0, "posterior_nll": 0.0}
    for index, ids in enumerate(subset.split(config.microbatch)):
        record = sample_blocks(model, ids, steps=config.sample_steps, seed=config.seed + 200000 + index)
        totals["bytes"] += record["bytes"]
        totals["sampled_nll_nats"] += record["sampled_nll_nats"]
        totals["correct"] += record["sampled_byte_accuracy"] * record["bytes"]
        totals["posterior_nll"] += record["posterior_nll_per_byte"] * record["bytes"]
    return {
        "sampled_nll_per_byte": totals["sampled_nll_nats"] / totals["bytes"],
        "sampled_byte_accuracy": totals["correct"] / totals["bytes"],
        "posterior_decode_nats_per_byte": totals["posterior_nll"] / totals["bytes"],
        "sampled_bytes": totals["bytes"],
    }


def evaluate_nelbo(model, validation, config, *, steps=None, probes=1, sequences=None):
    subset = validation[: config.nelbo_sequences if sequences is None else sequences]
    keys = (
        "bytes",
        "reconstruction_nll_nats",
        "prior_logprob_nats",
        "posterior_logprob_nats",
        "negative_elbo_estimate_nats",
    )
    totals = {key: 0.0 for key in keys}
    for index, ids in enumerate(subset.split(config.microbatch)):
        record = estimate_nelbo(
            model,
            ids,
            steps=config.nelbo_steps if steps is None else steps,
            seed=config.seed + 300000 + index,
            probes=probes,
        )
        for key in keys:
            totals[key] += record[key]
    denominator = totals["bytes"] * math.log(2)
    return {
        "nelbo_bpb": totals["negative_elbo_estimate_nats"] / denominator,
        "nelbo_reconstruction_bpb": totals["reconstruction_nll_nats"] / denominator,
        "nelbo_rate_bpb": (totals["posterior_logprob_nats"] - totals["prior_logprob_nats"]) / denominator,
        "nelbo_bytes": totals["bytes"],
    }


def provenance_record(config, model_config, loss_config, sources, data_metadata, counts, device):
    return {
        "architecture": ARCHITECTURE,
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "train_config": asdict(config),
        "source_hashes": sources,
        "data": data_metadata,
        "parameters": counts,
        "anchor_capacity_bits_per_patch": model_config.anchor_capacity_bits_per_patch,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "precision": "FP32 master weights; explicit BF16 neural kernels; FP32 latent/reduction arithmetic; no autocast",
        "optimizer": SOURCE_MUON_ALGORITHM,
        "schedule": "stage-relative: codec stage warmup then hold; joint stage 40-update warmup, hold until 30%, linear cooldown to 5%; Muon momentum 0.85->0.95 over 300 stage updates",
        "staging": f"{config.codec_steps} codec-only updates (prior not executed), then {config.steps - config.codec_steps} joint updates with attachment '{loss_config.attachment}'",
        "latent_contract": "deterministic unit-power encoder; every consumer reads the flow-path state at decode_time; no learned variance, KL, entropy, reference encoder, EMA, or isotropy term",
        "gradient_contract": (
            "flow loss attached through history stream, noisy block, and target"
            if loss_config.attachment == "all"
            else "flow loss attached through the history stream only; noisy block and target detached from the encoder"
        )
        + "; masked branch feeds only the codec",
        "masking": "independent patch-level mask-token corruption on the encoder input; CE on masked bytes only",
        "scoring": "objective terms are not rates; nelbo_bpb is a one-sample fixed-anchor negative-ELBO estimate on a validation subset; final evaluation scores complete validation contexts",
    }


def train(config: TrainConfig, model_config: ModelConfig | None = None, loss_config: LossConfig | None = None):
    model_config = model_config if model_config is not None else ModelConfig()
    loss_config = loss_config if loss_config is not None else LossConfig()
    config.validate(model_config)
    destination = ROOT / "ablation_results" / config.name
    if destination.exists():
        raise FileExistsError("run directory exists; use a fresh name, including for checkpoint continuation")
    device = setup_device(config.seed)
    corpus = ByteCorpus(ROOT / config.data_path, device)
    validation = corpus.validation_prefix(config.val_sequences * model_config.seq_bytes, model_config.seq_bytes)
    model, optimizers, previous = build_training_state(config, model_config, loss_config, device, corpus.metadata)
    objective = Objective(model, loss_config)
    step = previous["step"] if previous else 0
    cursor = previous["data_cursor"] if previous else 0
    training_ms = previous["training_ms"] if previous else 0.0
    batch_bytes = config.batch_sequences * model_config.seq_bytes
    destination.mkdir(parents=True)
    sources = source_hashes()
    counts = parameter_counts(model)
    provenance = provenance_record(config, model_config, loss_config, sources, corpus.metadata, counts, device)
    atomic_json(destination / "provenance.json", provenance)
    print(json.dumps(provenance, allow_nan=False), flush=True)
    policy = RecoveryCheckpointPolicy()
    checkpoint_path = destination / "checkpoint.pt"
    latest_validation = previous["validation"] if previous else None

    def checkpoint():
        atomic_torch_save(
            {
                "architecture": ARCHITECTURE,
                "checkpoint_version": CHECKPOINT_VERSION,
                "model_config": asdict(model_config),
                "loss_config": asdict(loss_config),
                "train_config": asdict(config),
                "model": model.state_dict(),
                "optimizers": {
                    name: [optimizer.state_dict() for optimizer in group]
                    for name, group in optimizers.items()
                },
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
                "step": step,
                "data_cursor": cursor,
                "training_ms": training_ms,
                "validation": latest_validation,
                "source_hashes": sources,
                "data": corpus.metadata,
                "parameters": counts,
            },
            checkpoint_path,
        )
        policy.committed(step)

    writer = MetricsWriter(destination / "metrics.jsonl", config.name)
    started = time.perf_counter()
    try:
        with (destination / "train.log").open("w") as log:
            while True:
                stage = stage_of(step, config)[0]
                joint = stage == 2
                # The stage boundary and the final update always get the full diagnostics,
                # so the joint stage has an untrained-prior baseline whatever the cadences.
                landmark = step in (config.codec_steps, config.steps)
                if step % config.val_every == 0 or landmark:
                    latest_validation = evaluate_objective(
                        model, objective, validation, config, flow=joint
                    )
                    if joint and (step % config.sample_every == 0 or landmark):
                        latest_validation.update(evaluate_samples(model, validation, config))
                    if joint and (step % config.nelbo_every == 0 or landmark):
                        latest_validation.update(evaluate_nelbo(model, validation, config))
                    emit(writer, log, step, config, latest_validation, training_ms, True, stage)
                    if step == 0:
                        checkpoint()
                if step == config.steps:
                    break
                torch.cuda.synchronize()
                tick = time.perf_counter()
                model.train().zero_grad(set_to_none=True)
                batch = corpus.training_batch(cursor, batch_bytes, model_config.seq_bytes)
                sums = torch.zeros(len(STAT_NAMES), device=device)
                accumulation = batch.shape[0] // config.microbatch
                for ids in batch.split(config.microbatch):
                    loss, statistics = objective(ids, flow=joint)
                    (loss / accumulation).backward()
                    sums.add_(statistics)
                optimizer_step(model, optimizers, config, step)
                torch.cuda.synchronize()
                training_ms += 1000 * (time.perf_counter() - tick)
                step += 1
                cursor += batch_bytes
                if step % config.log_every == 0 or step == config.steps:
                    emit(writer, log, step, config, summarize_statistics(sums.cpu().tolist()), training_ms, False, stage)
                if policy.due():
                    checkpoint()
        if policy.terminal_due(step):
            checkpoint()
    finally:
        writer.close()
    integrity = metric_integrity_errors(read_metrics_jsonl(destination / "metrics.jsonl"), config.steps)
    if integrity:
        raise RuntimeError("; ".join(integrity))
    if source_hashes() != sources:
        raise RuntimeError("training source changed mid-run; refusing mixed-source completed result")
    if latest_validation is None:
        raise RuntimeError("completed run has no final validation")
    result = {
        "name": config.name,
        "steps": config.steps,
        "completed_steps": step,
        "returncode": 0,
        "promotion_metric": "objective_only",
        "final_val_bpb": None,
        "final_proxy_val_bpb": None,
        "final_val_loss": latest_validation["objective"],
        "final_val_nelbo_bpb": latest_validation.get("nelbo_bpb"),
        "validation": latest_validation,
        "elapsed_seconds": time.perf_counter() - started,
        "training_seconds": training_ms / 1000,
        "consumed_training_bytes": cursor,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameters": counts,
        "source_hashes": sources,
        "interpretation": "completed CELF control; the subset nelbo_bpb is a diagnostic; see the separate evaluation for the full-context estimate",
    }
    atomic_json(destination / "result.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


def qualify(output: Path, microbatch: int):
    """Full-shape compiled forward/backward/optimizer qualification, not quality evidence."""
    device = setup_device(1337)
    model_config = ModelConfig()
    corpus = ByteCorpus(ROOT / "data/datasets/native_bits_fineweb", device)
    model = CelfModel(model_config).to(device)
    model.compile_components()
    config = TrainConfig(name="celf_qualification", microbatch=microbatch, batch_sequences=microbatch, val_sequences=microbatch, sample_sequences=microbatch, nelbo_sequences=microbatch)
    optimizers = {name: make_optimizers(model, config, name) for name in ("codec", "prior")}
    objective = Objective(model, LossConfig())
    ids = corpus.training_batch(0, microbatch * model_config.seq_bytes, model_config.seq_bytes)
    rows = []
    # Repeats 0-1 exercise the codec-only stage (prior untouched), 2-4 the joint stage.
    staged = replace(config, codec_steps=2, steps=4, val_every=1, log_every=1, warmup_steps=2, sample_every=1, nelbo_every=1)
    for repeat in range(5):
        joint = stage_of(repeat, staged)[0] == 2
        model.train().zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.perf_counter()
        loss, stats = objective(ids, flow=joint)
        loss.backward()
        optimizer_step(model, optimizers, staged, repeat)
        torch.cuda.synchronize()
        rows.append(
            {
                "repeat": repeat,
                "stage_id": stage_of(repeat, staged)[0],
                "seconds": time.perf_counter() - tick,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "statistics": summarize_statistics(stats.cpu().tolist()),
            }
        )
        print(json.dumps(rows[-1], allow_nan=False), flush=True)
    validation = corpus.validation_prefix(microbatch * model_config.seq_bytes, model_config.seq_bytes)
    torch.cuda.synchronize()
    tick = time.perf_counter()
    samples = evaluate_samples(model, validation, config)
    torch.cuda.synchronize()
    samples["seconds"] = time.perf_counter() - tick
    tick = time.perf_counter()
    nelbo = evaluate_nelbo(model, validation, config)
    torch.cuda.synchronize()
    nelbo["seconds"] = time.perf_counter() - tick
    generation = generate_bytes(model, bytes(validation[0, : 2 * model_config.block_bytes].cpu().tolist()), new_bytes=2 * model_config.block_bytes, steps=4, seed=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output,
        {
            "status": "passed",
            "purpose": "compiled full-shape correctness/performance, not a learning ablation",
            "model_config": asdict(model_config),
            "parameters": parameter_counts(model),
            "microbatch": microbatch,
            "records": rows,
            "samples": samples,
            "nelbo": nelbo,
            "generation": generation,
            "source_hashes": source_hashes(),
        },
    )


def final_evaluation(checkpoint_path: Path, output: Path, resolutions: list[int], probes: int):
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["step"] != checkpoint["train_config"]["steps"]:
        raise ValueError("final evaluation requires a completed run")
    if checkpoint["source_hashes"] != source_hashes():
        raise ValueError("evaluation source differs from checkpoint")
    config = TrainConfig(**checkpoint["train_config"])
    model_config = ModelConfig(**checkpoint["model_config"])
    device = setup_device(config.seed)
    corpus = ByteCorpus(ROOT / config.data_path, device)
    if corpus.metadata != checkpoint["data"]:
        raise ValueError("evaluation dataset differs from training provenance")
    model = CelfModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False).eval()
    model.compile_components()
    complete = corpus.validation.numel() // model_config.seq_bytes
    complete -= complete % config.microbatch
    validation = corpus.validation_prefix(complete * model_config.seq_bytes, model_config.seq_bytes)
    digest = sha256_file(checkpoint_path)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "source_hashes": source_hashes(),
        "data": corpus.metadata,
        "scored_bytes": int(validation.numel()),
        "heldout_bytes": int(corpus.validation.numel()),
        "semantics": "One-sample fixed-anchor negative-ELBO estimates over complete validation contexts; Hutchinson divergence with the stated probe count; same seeds across solver resolutions; not exact marginal BPB",
        "density": [],
        "samples": evaluate_samples(model, validation, config, sequences=validation.shape[0]),
        "generation": [],
    }
    started = time.perf_counter()
    for resolution in resolutions:
        record = evaluate_nelbo(model, validation, config, steps=resolution, probes=probes, sequences=validation.shape[0])
        record["steps"] = resolution
        record["probes"] = probes
        report["density"].append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
    for start in (0, 4096):
        prompt = bytes(corpus.validation[start : start + 2 * model_config.block_bytes].cpu().tolist())
        generated = generate_bytes(model, prompt, new_bytes=8 * model_config.block_bytes, steps=16, seed=1337)
        generated["prompt_text"] = prompt.decode("utf-8", errors="replace")
        report["generation"].append(generated)
    if sha256_file(checkpoint_path) != digest or source_hashes() != report["source_hashes"]:
        raise RuntimeError("evaluation inputs changed during execution")
    report["status"] = "complete"
    report["elapsed_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("describe")
    training = commands.add_parser("train")
    training.add_argument("--name", required=True)
    training.add_argument("--data-path", default="data/datasets/native_bits_fineweb")
    training.add_argument("--steps", type=int, default=2000)
    training.add_argument("--codec-steps", type=int, default=0)
    training.add_argument("--attachment", choices=("all", "history"), default="all")
    training.add_argument("--batch-sequences", type=int, default=32)
    training.add_argument("--microbatch", type=int, default=8)
    training.add_argument("--val-sequences", type=int, default=40)
    training.add_argument("--val-every", type=int, default=20)
    training.add_argument("--resume")
    qualification = commands.add_parser("qualify")
    qualification.add_argument("--output", type=Path, required=True)
    qualification.add_argument("--microbatch", type=int, default=8)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--resolutions", type=int, nargs="+", default=[16, 32])
    evaluation.add_argument("--probes", type=int, default=2)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "describe":
        config = ModelConfig()
        with torch.device("meta"):
            model = CelfModel(config)
        print(json.dumps({"config": asdict(config), "parameters": parameter_counts(model), "anchor_capacity_bits_per_patch": config.anchor_capacity_bits_per_patch}, indent=2))
    elif command == "train":
        attachment = args.pop("attachment")
        train(TrainConfig(**args), loss_config=LossConfig(attachment=attachment))
    elif command == "qualify":
        qualify(**args)
    else:
        final_evaluation(args["checkpoint"], args["output"], args["resolutions"], args["probes"])


if __name__ == "__main__":
    main()

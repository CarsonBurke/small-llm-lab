"""Paired BPE/byte Cola controls with canonical metrics and exact-resume checkpoints.

Run through scripts/train_cola.py and mlq. VAE preparation and joint
training are separate named runs; a random or partially trained VAE cannot
silently substitute for the completed preparation checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from checkpointing import RecoveryCheckpointPolicy, atomic_torch_save
from pretraining.cola.config import ARCHITECTURE, ModelConfig, control_config
from pretraining.cola.data import ColaCorpus
from pretraining.cola.model import ColaModel
from pretraining.cola.objective import (
    STAT_NAMES,
    LossConfig,
    Objective,
    summarize_statistics,
)
from pretraining.nanogpt_mini.native_bits_data import sha256_file
from pretraining.source_muon import SOURCE_MUON_ALGORITHM, Muon
from scripts.ablation import MetricsWriter, metric_integrity_errors, read_metrics_jsonl

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrainConfig:
    name: str
    stage: str
    tokenization: str = "byte"
    bpe_cache_path: str = "data/datasets/cola_bpe_100278"
    data_path: str = "data/datasets/native_bits_fineweb"
    steps: int = 2000
    batch_sequences: int = 32
    microbatch: int = 4
    val_sequences: int = 40
    val_every: int = 20
    log_every: int = 10
    seed: int = 1337
    vae_checkpoint: str | None = None
    resume: str | None = None
    warmup_steps: int = 40
    matrix_lr: float = 0.002
    other_lr: float = 0.0001
    scalar_lr: float = 0.0003
    embedding_lr: float = 0.002

    def validate(self, model: ModelConfig):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name):
            raise ValueError("name must be a simple run directory name")
        if self.stage not in {"vae", "joint"}:
            raise ValueError("stage must be vae or joint")
        if model.vocab_size != control_config(self.tokenization).vocab_size:
            raise ValueError("model vocabulary differs from selected tokenization")
        if self.stage == "joint" and not (self.vae_checkpoint or self.resume):
            raise ValueError(
                "joint control requires a completed VAE preparation checkpoint"
            )
        for key in (
            "steps",
            "batch_sequences",
            "microbatch",
            "val_sequences",
            "val_every",
            "log_every",
            "warmup_steps",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if (
            self.batch_sequences % self.microbatch
            or self.val_sequences % self.microbatch
        ):
            raise ValueError("sequence budgets must contain complete microbatches")
        if model.seq_len % model.block_size:
            raise ValueError("training contexts must contain complete diffusion blocks")
        for key in ("matrix_lr", "other_lr", "scalar_lr", "embedding_lr"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"invalid {key}")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must fit uint32")


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
            "scripts/train_cola.py",
            "scripts/prepare_cola_bpe.py",
            "scripts/ablation.py",
            "checkpointing.py",
        )
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def setup_device(seed: int):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Cola control requires a BF16-capable CUDA device; no fallback"
        )
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    return torch.device("cuda")


def parameter_counts(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "vae": sum(p.numel() for p in model.vae.parameters()),
        "prior": sum(p.numel() for p in model.prior.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def make_optimizers(model, config):
    matrices = [p for p in model.muon_parameters() if p.requires_grad]
    matrix_ids = {id(p) for p in matrices}
    embeddings = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding)
    }
    groups = {"embedding": [], "scalar": [], "other": []}
    for parameter in model.parameters():
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
    assigned = [
        p for opt in optimizers for group in opt.param_groups for p in group["params"]
    ]
    expected = {p for p in model.parameters() if p.requires_grad}
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise RuntimeError(
            "optimizers must partition every trainable parameter exactly once"
        )
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers


def schedule_multiplier(step: int, config: TrainConfig):
    warmup = min(1.0, (step + 1) / config.warmup_steps)
    progress = step / config.steps
    cooldown = 1.0 if progress < 0.3 else 1.0 - 0.95 * (progress - 0.3) / 0.7
    return min(warmup, cooldown)


@torch.compile(fullgraph=True, dynamic=False)
def finite_gradients(gradients):
    return torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()


def optimizer_step(model, optimizers, config, step):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None for p in parameters):
        missing = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        raise RuntimeError(
            f"trainable parameters disconnected from objective: {missing}"
        )
    if not finite_gradients([p.grad for p in parameters]).item():
        raise FloatingPointError(
            "non-finite Cola gradients; no clipping or silent repair"
        )
    eta = schedule_multiplier(step, config)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * eta
            if "mu" in group:
                group["mu"] = 0.85 + 0.1 * min(1.0, step / 300)
        optimizer.step()


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("architecture") != ARCHITECTURE
        or payload.get("checkpoint_version") != 1
    ):
        raise ValueError("not a compatible byte Cola checkpoint")
    return payload


def build_training_state(config, model_config, loss_config, device, data_metadata):
    model = ColaModel(model_config).to(device)
    previous = load_checkpoint(config.resume) if config.resume else None
    parent = None
    reference = None
    if previous is not None:
        if previous["model_config"] != asdict(model_config) or previous[
            "loss_config"
        ] != asdict(loss_config):
            raise ValueError("resume model/objective differs from checkpoint")
        if (
            previous["source_hashes"] != source_hashes()
            or previous["data"] != data_metadata
        ):
            raise ValueError("resume source or data fingerprint differs")
        for key, value in asdict(config).items():
            if (
                key not in {"name", "resume", "vae_checkpoint"}
                and previous["train_config"][key] != value
            ):
                raise ValueError(f"resume training setting differs: {key}")
        if (
            not 0 <= previous["step"] <= config.steps
            or previous["data_cursor"]
            != previous["step"] * config.batch_sequences * model_config.seq_len
        ):
            raise ValueError("invalid saved update or consumed-position cursor")
        (model.vae if config.stage == "vae" else model).load_state_dict(
            previous["model"], strict=True
        )
        parent = previous.get("vae_parent")
    elif config.stage == "joint":
        initial = load_checkpoint(config.vae_checkpoint)
        if (
            initial["stage"] != "vae"
            or initial["step"] != initial["train_config"]["steps"]
        ):
            raise ValueError("VAE preparation is not complete")
        if initial["model_config"] != asdict(model_config) or initial[
            "loss_config"
        ] != asdict(loss_config):
            raise ValueError("VAE preparation architecture/objective differs")
        if (
            initial["data"] != data_metadata
            or initial["source_hashes"] != source_hashes()
        ):
            raise ValueError("VAE preparation data/source contract differs")
        if initial["validation"]["latent_low_clamp_fraction"] == 1.0:
            raise ValueError(
                "VAE posterior variance is entirely saturated; rejected preparation"
            )
        model.vae.load_state_dict(initial["model"], strict=True)
        parent = {
            "path": str(Path(config.vae_checkpoint).resolve()),
            "sha256": sha256_file(Path(config.vae_checkpoint)),
            "steps": initial["step"],
        }
    if config.stage == "joint":
        reference = deepcopy(model.vae.encoder).requires_grad_(False).eval()
        if previous is not None:
            reference.load_state_dict(previous["reference"], strict=True)
    else:
        model.prior.requires_grad_(False)
    optimizers = make_optimizers(model, config)
    if previous is not None:
        for optimizer, state in zip(optimizers, previous["optimizers"], strict=True):
            optimizer.load_state_dict(state)
    model.compile_components()
    if reference is not None:
        reference.compile_component()
    if previous is not None:
        torch.set_rng_state(previous["rng_cpu"])
        torch.cuda.set_rng_state(previous["rng_cuda"])
    return model, reference, optimizers, previous, parent


def emit(writer, log, step, config, metrics, training_ms, validation):
    prefix = "val" if validation else "train"
    entry = {
        "type": prefix,
        "step": step,
        f"{prefix}_loss": metrics["objective"],
        "train_time_ms": training_ms,
        "stage_id": 1 if config.stage == "vae" else 2,
        "step_avg_ms": training_ms / max(1, step),
        **{f"{prefix}_{key}": value for key, value in metrics.items()},
    }
    writer.write_entry(entry)
    first = f"step:{step}/{config.steps} {prefix}_loss:{metrics['objective']:.9g}"
    if validation:
        first += f" val_objective:{metrics['objective']:.9g}"
    first += f" train_time:{training_ms:.3f}ms"
    extras = " ".join(
        f"{prefix}_{key}:{value:.9g}"
        for key, value in metrics.items()
        if key != "objective"
    )
    line = first + " " + extras
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()
    return entry


@torch.no_grad()
def evaluate_objective(model, objective, validation, config):
    model.eval()
    generator = torch.Generator(device=validation.device).manual_seed(
        config.seed + 100000
    )
    sums = torch.zeros(len(STAT_NAMES), device=validation.device)
    for ids in validation.split(config.microbatch):
        _, stats = objective(ids, generator)
        sums.add_(stats)
    model.train()
    return summarize_statistics(sums.cpu().tolist())


def train(
    config: TrainConfig,
    model_config: ModelConfig | None = None,
    loss_config: LossConfig | None = None,
):
    model_config = (
        model_config
        if model_config is not None
        else control_config(config.tokenization)
    )
    loss_config = loss_config if loss_config is not None else LossConfig()
    config.validate(model_config)
    destination = ROOT / "ablation_results" / config.name
    if destination.exists():
        raise FileExistsError(
            "run directory exists; use a fresh name, including for checkpoint continuation"
        )
    device = setup_device(config.seed)
    corpus = ColaCorpus(
        ROOT / config.data_path,
        device,
        config.tokenization,
        ROOT / config.bpe_cache_path,
    )
    validation = corpus.validation_prefix(
        config.val_sequences * model_config.seq_len, model_config.seq_len
    )
    model, reference, optimizers, previous, parent = build_training_state(
        config, model_config, loss_config, device, corpus.metadata
    )
    objective = Objective(model, loss_config, corpus.byte_lengths, reference)
    step = previous["step"] if previous else 0
    cursor = previous["data_cursor"] if previous else 0
    training_ms = previous["training_ms"] if previous else 0.0
    consumed_source_bytes = previous["consumed_source_bytes"] if previous else 0
    batch_positions = config.batch_sequences * model_config.seq_len
    destination.mkdir(parents=True)
    sources = source_hashes()
    counts = parameter_counts(model)
    provenance = {
        "architecture": ARCHITECTURE,
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "train_config": asdict(config),
        "source_hashes": sources,
        "data": corpus.metadata,
        "parameters": counts,
        "vae_parent": parent,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "precision": "FP32 master weights; explicit BF16 neural kernels; FP32 Gaussian/reduction arithmetic; no autocast",
        "optimizer": SOURCE_MUON_ALGORITHM,
        "schedule": "40-update warmup; hold until30%; linear cooldown to5%; Muon momentum0.85→0.95 over300updates",
        "masking": "independent15% mask-only position corruption; auxiliary CE on masked positions; no80/10/10 replacement",
        "stage1_prior": "standard diagonal Gaussian, explicit scaled-control choice",
        "stage2": "joint VAE/DiT; negative posterior entropy; current||frozen stage1 encoder KL; detached clean history; live FM target",
        "weights_status": "beta/lambdas and masking details are explicit control choices; unreleased original coefficients are not claimed",
        "tokenizer_comparison": {
            "heldout_bytes_per_olmo2_bpe_token": 4.579409065883648,
            "context_positions": model_config.seq_len,
            "block_positions": model_config.block_size,
            "reference_bpe_context": 512,
            "reference_bpe_block": 16,
        },
        "comparison": "parameter-matched systems, not identical backbones;32contexts/update; source-byte exposure measured separately",
        "learning_rate_choice": "one tenth of initial nanoGPT-transfer rates; initial byte VAE hit a dead variance clamp; lower step-size stability hypothesis, not a claimed paper hyperparameter",
        "scoring": "objective/reconstruction/FM are separate; none is advertised as marginal BPB; final CNF negativeELBO estimate is a separate numerical evaluation",
    }
    atomic_json(destination / "provenance.json", provenance)
    print(json.dumps(provenance, allow_nan=False), flush=True)
    policy = RecoveryCheckpointPolicy()
    checkpoint_path = destination / "checkpoint.pt"
    latest_validation = previous["validation"] if previous else None

    def checkpoint():
        atomic_torch_save(
            {
                "architecture": ARCHITECTURE,
                "checkpoint_version": 1,
                "stage": config.stage,
                "model_config": asdict(model_config),
                "loss_config": asdict(loss_config),
                "train_config": asdict(config),
                "model": (model.vae if config.stage == "vae" else model).state_dict(),
                "reference": reference.state_dict() if reference is not None else None,
                "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
                "step": step,
                "data_cursor": cursor,
                "consumed_source_bytes": consumed_source_bytes,
                "training_ms": training_ms,
                "validation": latest_validation,
                "source_hashes": sources,
                "data": corpus.metadata,
                "vae_parent": parent,
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
                if step % config.val_every == 0 or step == config.steps:
                    latest_validation = evaluate_objective(
                        model, objective, validation, config
                    )
                    emit(
                        writer, log, step, config, latest_validation, training_ms, True
                    )
                    if step == 0:
                        checkpoint()
                if (
                    latest_validation is not None
                    and latest_validation["latent_low_clamp_fraction"] == 1.0
                ):
                    checkpoint()
                    raise FloatingPointError(
                        "posterior variance saturated at its floor across all coordinates; refusing a degenerate control"
                    )
                if step == config.steps:
                    break
                torch.cuda.synchronize()
                tick = time.perf_counter()
                model.train().zero_grad(set_to_none=True)
                batch = corpus.training_batch(
                    cursor, batch_positions, model_config.seq_len
                )
                sums = torch.zeros(len(STAT_NAMES), device=device)
                accumulation = batch.shape[0] // config.microbatch
                for ids in batch.split(config.microbatch):
                    loss, statistics = objective(ids)
                    (loss / accumulation).backward()
                    sums.add_(statistics)
                optimizer_step(model, optimizers, config, step)
                torch.cuda.synchronize()
                training_ms += 1000 * (time.perf_counter() - tick)
                step += 1
                cursor += batch_positions
                metrics = summarize_statistics(sums.cpu().tolist())
                consumed_source_bytes += int(metrics["source_bytes"])
                if step % config.log_every == 0 or step == config.steps:
                    emit(writer, log, step, config, metrics, training_ms, False)
                if policy.due():
                    checkpoint()
        if policy.terminal_due(step):
            checkpoint()
    finally:
        writer.close()
    integrity = metric_integrity_errors(
        read_metrics_jsonl(destination / "metrics.jsonl"), config.steps
    )
    if integrity:
        raise RuntimeError("; ".join(integrity))
    if source_hashes() != sources:
        raise RuntimeError(
            "training source changed mid-run; refusing mixed-source completed result"
        )
    if latest_validation is None:
        raise RuntimeError("completed stage has no final validation")
    result = {
        "name": config.name,
        "stage": config.stage,
        "steps": config.steps,
        "completed_steps": step,
        "returncode": 0,
        "promotion_metric": "objective_only",
        "final_val_bpb": None,
        "final_proxy_val_bpb": None,
        "final_val_loss": latest_validation["objective"],
        "validation": latest_validation,
        "elapsed_seconds": time.perf_counter() - started,
        "training_seconds": training_ms / 1000,
        "consumed_training_positions": cursor,
        "consumed_training_bytes": consumed_source_bytes,
        "tokenization": config.tokenization,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameters": counts,
        "vae_parent": parent,
        "source_hashes": sources,
        "interpretation": "completed control stage; no ablation/promotion claim; see separate CNF evaluation for numerical latent-model likelihood estimate",
    }
    atomic_json(destination / "result.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


def qualify(output: Path, microbatch: int, tokenization: str, bpe_cache_path: str):
    """Full architecture/context backward and optimizer qualification, not quality evidence."""
    device = setup_device(1337)
    model_config = control_config(tokenization)
    corpus = ColaCorpus(
        ROOT / "data/datasets/native_bits_fineweb",
        device,
        tokenization,
        ROOT / bpe_cache_path,
    )
    model = ColaModel(model_config).to(device)
    reference = deepcopy(model.vae.encoder).requires_grad_(False).eval()
    model.compile_components()
    reference.compile_component()
    rows = []
    ids = corpus.training_batch(
        0, microbatch * model_config.seq_len, model_config.seq_len
    )
    for stage in ("vae", "joint"):
        model.prior.requires_grad_(stage == "joint")
        config = TrainConfig(
            name="cola_qualification",
            stage=stage,
            tokenization=tokenization,
            microbatch=microbatch,
        )
        optimizers = make_optimizers(model, config)
        objective = Objective(
            model,
            LossConfig(),
            corpus.byte_lengths,
            reference if stage == "joint" else None,
        )
        for repeat in range(2):
            model.train().zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            tick = time.perf_counter()
            loss, stats = objective(ids)
            loss.backward()
            optimizer_step(model, optimizers, config, repeat)
            torch.cuda.synchronize()
            row = {
                "stage": stage,
                "repeat": repeat,
                "seconds": time.perf_counter() - tick,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "statistics": summarize_statistics(stats.cpu().tolist()),
            }
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
        del optimizers
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
            "source_hashes": source_hashes(),
        },
    )


def final_evaluation(checkpoint_path: Path, output: Path, resolutions: list[int]):
    from pretraining.cola.evaluation import estimate_negative_elbo, generate_tokens

    checkpoint = load_checkpoint(checkpoint_path)
    if (
        checkpoint["stage"] != "joint"
        or checkpoint["step"] != checkpoint["train_config"]["steps"]
    ):
        raise ValueError("final evaluation requires the completed joint control")
    if checkpoint["source_hashes"] != source_hashes():
        raise ValueError("evaluation source differs from checkpoint")
    config = TrainConfig(**checkpoint["train_config"])
    model_config = ModelConfig(**checkpoint["model_config"])
    device = setup_device(config.seed)
    corpus = ColaCorpus(
        ROOT / config.data_path,
        device,
        config.tokenization,
        ROOT / config.bpe_cache_path,
    )
    if corpus.metadata != checkpoint["data"]:
        raise ValueError("evaluation dataset differs from training provenance")
    models = {}

    def model_for_length(length):
        if length not in models:
            model = ColaModel(replace(model_config, seq_len=length)).to(device)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.requires_grad_(False).eval()
            model.compile_components()
            models[length] = model
        return models[length]

    digest = sha256_file(checkpoint_path)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "source_hashes": source_hashes(),
        "tokenization": config.tokenization,
        "data": corpus.metadata,
        "semantics": "Numerical one-posterior-sample CNF negativeELBO estimates over the entire same literal heldout file; no padding or dropped tails; same seeds across solver resolutions; not exact marginal or challenge BPB",
        "density": [],
        "generation": [],
    }
    started = time.perf_counter()
    for resolution in resolutions:
        records = []
        for index, row in enumerate(corpus.validation.split(model_config.seq_len)):
            batch = row.long().unsqueeze(0)
            record = estimate_negative_elbo(
                model_for_length(row.numel()),
                batch,
                steps=resolution,
                seed=1337 + index,
                probes=1,
                source_bytes=int(corpus.source_bytes(batch).item()),
            )
            records.append(record)
            print(
                json.dumps(
                    {"resolution": resolution, "batch": index, **record},
                    allow_nan=False,
                ),
                flush=True,
            )
        byte_count = sum(record["bytes"] for record in records)
        if byte_count != (ROOT / config.data_path / "val.txt").stat().st_size:
            raise RuntimeError(
                "evaluation did not score every heldout source byte exactly once"
            )
        totals = {
            key: sum(record[key] for record in records)
            for key in (
                "positions",
                "bytes",
                "reconstruction_nll_nats",
                "prior_logprob_nats",
                "posterior_logprob_nats",
                "negative_elbo_estimate_nats",
            )
        }
        totals["negative_elbo_estimate_bpb"] = totals["negative_elbo_estimate_nats"] / (
            byte_count * math.log(2)
        )
        report["density"].append(
            {"steps": resolution, "totals": totals, "records": records}
        )
    for prompt in ("The history of science is ", "To solve this problem, "):
        prompt_ids = corpus.encode_text(prompt)
        if corpus.decode_ids(prompt_ids) != prompt.encode("utf-8"):
            raise RuntimeError("prompt tokenization is not lossless")
        generated = generate_tokens(
            model_for_length(model_config.seq_len),
            prompt_ids,
            new_tokens=160 if config.tokenization == "byte" else 32,
            steps=16,
            seed=1337,
        )
        generated_bytes = corpus.decode_ids(generated["generated_ids"])
        output_bytes = corpus.decode_ids(generated["output_ids"])
        generated.update(
            prompt_text=prompt,
            generated_bytes=len(generated_bytes),
            generated_hex=generated_bytes.hex(),
            generated_text=generated_bytes.decode("utf-8", errors="replace"),
            output_hex=output_bytes.hex(),
            output_text=output_bytes.decode("utf-8", errors="replace"),
        )
        report["generation"].append(generated)
    if (
        sha256_file(checkpoint_path) != digest
        or source_hashes() != report["source_hashes"]
    ):
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
    training.add_argument("--stage", choices=("vae", "joint"), required=True)
    training.add_argument("--tokenization", choices=("bpe", "byte"), required=True)
    training.add_argument("--bpe-cache-path", default="data/datasets/cola_bpe_100278")
    training.add_argument("--data-path", default="data/datasets/native_bits_fineweb")
    training.add_argument("--steps", type=int, default=2000)
    training.add_argument("--batch-sequences", type=int, default=32)
    training.add_argument("--microbatch", type=int, default=4)
    training.add_argument("--val-sequences", type=int, default=40)
    training.add_argument("--val-every", type=int, default=20)
    training.add_argument("--vae-checkpoint")
    training.add_argument("--resume")
    qualification = commands.add_parser("qualify")
    qualification.add_argument("--output", type=Path, required=True)
    qualification.add_argument("--microbatch", type=int, default=4)
    qualification.add_argument("--tokenization", choices=("bpe", "byte"), required=True)
    qualification.add_argument(
        "--bpe-cache-path", default="data/datasets/cola_bpe_100278"
    )
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--resolutions", type=int, nargs="+", default=[16, 32])
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "describe":
        descriptions = {}
        for tokenization in ("bpe", "byte"):
            config = control_config(tokenization)
            with torch.device("meta"):
                model = ColaModel(config)
            descriptions[tokenization] = {
                "config": asdict(config),
                "parameters": parameter_counts(model),
            }
        print(json.dumps(descriptions, indent=2))
    elif command == "train":
        train(TrainConfig(**args))
    elif command == "qualify":
        qualify(**args)
    else:
        final_evaluation(args["checkpoint"], args["output"], args["resolutions"])


if __name__ == "__main__":
    main()

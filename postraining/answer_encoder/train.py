"""Train and inspect a faithful LeJEPA-style answer encoder."""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
import math
import os
import random
import time
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2TokenizerFast

from postraining.answer_encoder.data import (
    MaskedPredictionBatch,
    MaskedPredictionBatchSampler,
    MixedCorpus,
    SpanMaskConfig,
    TokenShardCorpus,
    ViewBatchSampler,
    ViewConfig,
    load_parquet_texts,
    tokenize_texts,
)
from postraining.answer_encoder.inference import (
    FrozenAnswerScorer,
    load_encoder_checkpoint,
    save_target_cache,
)
from postraining.answer_encoder.model import (
    ANSWER_ENCODER_SCHEMA,
    DEFAULT_REWARD_TEMPERATURE,
    AnswerEncoderConfig,
    LeJEPAObjective,
    LeJEPAObjectiveConfig,
    GlobalLatentObjective,
    GlobalLatentPredictor,
    TextAnswerEncoder,
    cosine_kernel_reward,
)
from postraining.answer_encoder.probes import run_behavioral_probes


DEFAULT_TRAIN_GLOB = "data/datasets/k3mix_v7_quality_gpt2_20k/fineweb_train_*.bin"
DEFAULT_VAL_GLOB = "data/datasets/k3mix_v7_quality_gpt2_20k/fineweb_val_*.bin"
DEFAULT_ANSWER_SOURCES = (
    "postraining/data/opsd_dapo17k_bare_train.parquet:solution",
    "data/pretraining_sources/github_code_clean/data/train-00000-of-00880.parquet:code",
)


@lru_cache(maxsize=1)
def _gpt2_tokenizer() -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    tokenizer.model_max_length = 1 << 30
    return tokenizer


def gpt2_encode(text: str) -> list[int]:
    return _gpt2_tokenizer().encode(text, add_special_tokens=False)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _serializable_args(args: argparse.Namespace) -> dict:
    return {key: value for key, value in vars(args).items() if key != "function"}


def _validate_resume_checkpoint_args(
    checkpoint_args: object,
    manifest_args: dict,
) -> None:
    if not isinstance(checkpoint_args, dict):
        raise ValueError("resume checkpoint is missing its training arguments")
    normalized_checkpoint_args = {"objective_space": "projection", **checkpoint_args}
    for key, value in manifest_args.items():
        if key in {"resume", "steps"}:
            continue
        if normalized_checkpoint_args.get(key) != value:
            raise ValueError(
                f"resume checkpoint argument {key!r} differs from the run manifest"
            )


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _atomic_torch_save(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _module_state_payload(
    model: TextAnswerEncoder,
    predictor: GlobalLatentPredictor | None,
    *,
    training: bool,
) -> dict:
    """Serialize the predictor only in resumable training state."""
    payload = {"model": model.state_dict()}
    if training:
        payload["predictor"] = None if predictor is None else predictor.state_dict()
    return payload


def _load_module_training_state(
    payload: dict,
    model: TextAnswerEncoder,
    predictor: GlobalLatentPredictor | None,
) -> None:
    model.load_state_dict(payload["model"], strict=True)
    predictor_state = payload.get("predictor")
    if predictor is None:
        if predictor_state is not None:
            raise ValueError("checkpoint contains a predictor for a multicrop run")
    else:
        if predictor_state is None:
            raise ValueError("global-latent checkpoint is missing predictor state")
        predictor.load_state_dict(predictor_state, strict=True)


def _effective_rank(samples: Tensor) -> float:
    values = samples.detach().float()
    values = values - values.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(values)
    probabilities = singular_values.square()
    probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def _resolved_projection_dimension(args: argparse.Namespace) -> int:
    """Keep v5's width while making the CLS-only objective 256-D by default."""
    if args.projection_dim is not None:
        if args.projection_dim < 1:
            raise ValueError("projection_dim must be positive")
        return args.projection_dim
    return 256 if args.training_objective == "global-latent" else 16


def _validate_training_geometry(args: argparse.Namespace) -> None:
    if args.training_objective != "global-latent":
        return
    if args.objective_space != "projection":
        raise ValueError("global-latent prediction operates in projection space")
    if args.reward_space != "projection":
        raise ValueError("global-latent deployment operates in projection space")
    if args.projector_normalization != "layer":
        raise ValueError("global-latent prediction requires per-item LayerNorm projection")


def _learning_rate(
    step: int,
    steps: int,
    warmup_steps: int,
    peak: float,
    minimum: float,
) -> float:
    if not 0.0 <= minimum <= peak:
        raise ValueError("minimum learning rate must be in [0, peak]")
    if step < warmup_steps:
        return peak * (0.01 + 0.99 * step / max(warmup_steps, 1))
    progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return minimum + (peak - minimum) * cosine


def _flatten_views(
    token_ids: Tensor,
    attention_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    batch, views, length = token_ids.shape
    return (
        token_ids.reshape(batch * views, length),
        attention_mask.reshape(batch * views, length),
    )


def _encode_view_batch(
    model: TextAnswerEncoder,
    token_ids: Tensor,
    attention_mask: Tensor,
    *,
    objective_space: str,
) -> Tensor:
    if objective_space == "backbone":
        batch, views = token_ids.shape[:2]
        flat_ids, flat_mask = _flatten_views(token_ids, attention_mask)
        backbones = model.encode_backbone(flat_ids, flat_mask)
        return backbones.reshape(batch, views, -1).transpose(0, 1)
    _, projections = _encode_view_batch_outputs(
        model, token_ids, attention_mask
    )
    if objective_space == "projection":
        return projections
    raise ValueError(f"unknown objective space {objective_space!r}")


def _encode_view_batch_outputs(
    model: TextAnswerEncoder,
    token_ids: Tensor,
    attention_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    batch, views = token_ids.shape[:2]
    flat_ids, flat_mask = _flatten_views(token_ids, attention_mask)
    backbone, projection = model(flat_ids, flat_mask)
    return (
        backbone.reshape(batch, views, -1).transpose(0, 1),
        projection.reshape(batch, views, -1).transpose(0, 1),
    )


def _encode_global_latent_batch(
    model: TextAnswerEncoder,
    predictor: GlobalLatentPredictor,
    batch: MaskedPredictionBatch,
) -> dict[str, Tensor]:
    """Predict the complete target CLS from a compacted visible context."""
    target_ids = batch.target_ids
    target_attention = batch.target_attention_mask
    prediction_mask = batch.prediction_mask
    context_ids = batch.context_ids
    context_attention = batch.context_attention_mask
    context_position_ids = batch.context_position_ids
    if target_ids.ndim != 2 or target_attention.shape != target_ids.shape:
        raise ValueError("target ids and attention must have shape [batch, target length]")
    if prediction_mask.shape != target_ids.shape:
        raise ValueError("prediction mask must match target ids")
    if context_ids.ndim != 2 or context_attention.shape != context_ids.shape:
        raise ValueError("context ids and attention must have shape [batch, context length]")
    if context_position_ids.shape != context_ids.shape:
        raise ValueError("context positions must match context ids")
    if context_ids.size(0) != target_ids.size(0):
        raise ValueError("target and context batches must match")
    if bool((prediction_mask & ~target_attention).any()):
        raise ValueError("padding positions cannot be latent prediction targets")
    if not bool(prediction_mask.any(dim=1).all()):
        raise ValueError("every sample must contain at least one prediction target")
    if bool((context_position_ids[context_attention] < 0).any()):
        raise ValueError("visible context positions must be nonnegative")

    target_hidden = model.encode_hidden(target_ids, target_attention)
    context_hidden = model.encode_hidden(
        context_ids,
        context_attention,
        position_ids=context_position_ids,
        allow_empty_tokens=True,
    )

    target_global = target_hidden[:, 0]
    batch_size = target_ids.size(0)
    context_full_attention = torch.cat(
        (
            torch.ones(batch_size, 1, dtype=torch.bool, device=target_ids.device),
            context_attention,
        ),
        dim=1,
    )
    predicted_global_hidden = predictor(
        context_hidden,
        context_full_attention,
    )
    projected = model.projector(
        torch.cat(
            (
                predicted_global_hidden,
                target_global,
            )
        )
    )
    predicted_global, target_global_projected = projected.split(batch_size)
    return {
        "predicted_global": predicted_global,
        "target_global": target_global_projected,
        "target_global_backbone": target_global,
        "masked_fraction": prediction_mask.sum() / target_attention.sum(),
        "visible_fraction": context_attention.sum() / target_attention.sum(),
        "visible_tokens_per_sample": context_attention.sum(dim=1).float().mean(),
    }


@contextmanager
def _projector_batch_statistics(model: TextAnswerEncoder):
    """Use live projector BN statistics without mutating buffers or leaking mode."""
    states = []
    for module in model.projector.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            states.append((module, module.training, module.track_running_stats))
            module.track_running_stats = False
            module.train()
    try:
        yield
    finally:
        for module, training, track_running_stats in states:
            module.track_running_stats = track_running_stats
            module.train(training)


@torch.inference_mode()
def evaluate_objective(
    model: TextAnswerEncoder,
    objective: LeJEPAObjective,
    sampler: ViewBatchSampler,
    *,
    batches: int,
    batch_size: int,
    device: torch.device,
    objective_space: str = "projection",
) -> dict[str, float]:
    """Evaluate the training objective with BN batch statistics.

    LeJEPA's BatchNorm projector is a training-only loss head. Stored running
    statistics are not used downstream by the reference and are not a valid
    basis for measuring the objective. The deterministic backbone is measured
    separately because it is the representation consumed at inference.
    """
    model.eval()
    totals: dict[str, float] = {}
    projection_centers = []
    backbone_centers = []
    objective_centers = []
    with _projector_batch_statistics(model):
        for _ in range(batches):
            token_ids, attention_mask = sampler.batch(batch_size)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                backbones, projections = _encode_view_batch_outputs(
                    model,
                    token_ids.to(device),
                    attention_mask.to(device),
                )
                objective_embeddings = (
                    backbones if objective_space == "backbone" else projections
                )
                loss, diagnostics = objective(objective_embeddings)
            values = {
                "val_loss": float(loss),
                **{key: float(value) for key, value in diagnostics.items()},
            }
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
            projection_centers.append(projections.mean(dim=0).cpu())
            backbone_centers.append(backbones.mean(dim=0).cpu())
            objective_centers.append(objective_embeddings.mean(dim=0).cpu())
    averaged = {key: value / batches for key, value in totals.items()}
    averaged["projection_effective_rank"] = _effective_rank(
        torch.cat(projection_centers)
    )
    averaged["backbone_effective_rank"] = _effective_rank(
        torch.cat(backbone_centers)
    )
    averaged["objective_effective_rank"] = _effective_rank(
        torch.cat(objective_centers)
    )
    averaged["rank_sample_count"] = batches * batch_size
    return averaged


@torch.inference_mode()
def evaluate_global_latent_objective(
    model: TextAnswerEncoder,
    predictor: GlobalLatentPredictor,
    objective: GlobalLatentObjective,
    sampler: MaskedPredictionBatchSampler,
    *,
    batches: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate global prediction and the deployed full-answer space."""
    model.eval()
    predictor.eval()
    totals: dict[str, float] = {}
    projection_samples = []
    backbone_samples = []
    with _projector_batch_statistics(model):
        for _ in range(batches):
            batch = sampler.batch(batch_size).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = _encode_global_latent_batch(
                    model, predictor, batch
                )
                loss, diagnostics = objective(
                    outputs["predicted_global"], outputs["target_global"]
                )
            values = {
                "val_loss": float(loss),
                "masked_fraction": float(outputs["masked_fraction"]),
                "visible_fraction": float(outputs["visible_fraction"]),
                "visible_tokens_per_sample": float(
                    outputs["visible_tokens_per_sample"]
                ),
                **{key: float(value) for key, value in diagnostics.items()},
            }
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
            projection_samples.append(outputs["target_global"].cpu())
            backbone_samples.append(outputs["target_global_backbone"].cpu())

    averaged = {key: value / batches for key, value in totals.items()}
    averaged["projection_effective_rank"] = _effective_rank(
        torch.cat(projection_samples)
    )
    averaged["backbone_effective_rank"] = _effective_rank(
        torch.cat(backbone_samples)
    )
    averaged["objective_effective_rank"] = averaged["projection_effective_rank"]
    averaged["projection_dimension"] = projection_samples[0].size(-1)
    averaged["rank_sample_count"] = batches * batch_size
    return averaged


@torch.inference_mode()
def _diagnose_batches(
    model: TextAnswerEncoder,
    objective: LeJEPAObjective,
    batches: list[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
    projector_batch_statistics: bool,
    objective_space: str = "projection",
) -> dict[str, float]:
    """Measure representations with either stored or current-batch BN statistics."""
    model.eval()

    totals: dict[str, float] = {}
    mode = _projector_batch_statistics(model) if projector_batch_statistics else nullcontext()
    with mode:
        for batch_index, (token_ids, attention_mask) in enumerate(batches):
            batch, views = token_ids.shape[:2]
            flat_ids, flat_mask = _flatten_views(
                token_ids.to(device),
                attention_mask.to(device),
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                backbone, projection = model(flat_ids, flat_mask)
                backbones = backbone.reshape(batch, views, -1).transpose(0, 1)
                projections = projection.reshape(batch, views, -1).transpose(0, 1)
                # Use identical slices for both BN modes so their SIGReg values are
                # directly comparable rather than confounded by Monte Carlo noise.
                torch.manual_seed(10_000 + batch_index)
                objective_embeddings = (
                    backbones
                    if objective_space == "backbone"
                    else projections
                )
                loss, diagnostics = objective(objective_embeddings)
            values = {
                "loss": float(loss),
                **{key: float(value) for key, value in diagnostics.items()},
                "projection_effective_rank": _effective_rank(projections.mean(dim=0)),
                "backbone_effective_rank": _effective_rank(backbones.mean(dim=0)),
            }
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
    return {key: value / len(batches) for key, value in totals.items()}


def _load_answer_examples(args: argparse.Namespace) -> list:
    sources = args.answer_source or list(DEFAULT_ANSWER_SOURCES)
    remaining = args.max_answer_examples
    examples = []
    seen_tokens: set[tuple[int, ...]] = set()
    for source in sources:
        if remaining == 0:
            break
        source_texts = load_parquet_texts(
            source,
            # Fetch modest headroom because distinct full strings can collapse
            # to the same truncated token sequence after cross-source dedup.
            max_texts=None if remaining is None else remaining * 2,
            deduplicate=args.deduplicate_answer_text,
            max_characters=args.max_source_tokens * 16,
        )
        source_examples = tokenize_texts(
            source_texts,
            gpt2_encode,
            block_separator_tokens=tuple(gpt2_encode("\n\n")),
            max_source_tokens=args.max_source_tokens,
            preserve_blocks=args.block_shuffle_probability > 0.0,
        )
        for example in source_examples:
            token_key = tuple(map(int, example.tokens.tolist()))
            if args.deduplicate_answer_text and token_key in seen_tokens:
                continue
            seen_tokens.add(token_key)
            examples.append(example)
            if len(examples) >= args.max_answer_examples:
                break
        if remaining is not None:
            remaining = max(args.max_answer_examples - len(examples), 0)
    return examples


def _split_answer_examples(examples: list, fraction: float, seed: int) -> tuple[list, list]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("answer validation fraction must be in (0, 1)")
    if len(examples) < 2:
        raise ValueError("at least two answer examples are required for a held-out split")
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    validation_count = min(max(round(len(order) * fraction), 1), len(order) - 1)
    validation_indices = set(order[:validation_count])
    train_examples = [example for index, example in enumerate(examples) if index not in validation_indices]
    validation_examples = [example for index, example in enumerate(examples) if index in validation_indices]
    return train_examples, validation_examples


def _manifest(args: argparse.Namespace, train_answer_count: int, val_answer_count: int) -> dict:
    return {
        "schema": ANSWER_ENCODER_SCHEMA,
        "args": _serializable_args(args),
        "train_answer_examples": train_answer_count,
        "validation_answer_examples": val_answer_count,
        "train_shards": sorted(glob.glob(args.train_glob)),
        "val_shards": sorted(glob.glob(args.val_glob)),
        "versions": {
            "python_torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
        },
    }


def train(args: argparse.Namespace) -> None:
    if args.steps < 1 or args.batch_size < 2 or args.eval_batches < 1:
        raise ValueError("steps/eval_batches must be positive and batch_size must be at least two")
    if args.eval_every < 1 or args.log_every < 1:
        raise ValueError("eval_every and log_every must be positive")
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("warmup_steps must be nonnegative and smaller than steps")
    if not 0.0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError("min_learning_rate must be in [0, learning_rate]")
    if not 0.0 <= args.answer_probability <= 1.0:
        raise ValueError("answer_probability must be in [0, 1]")
    if args.max_answer_examples < 1:
        raise ValueError("max_answer_examples must be positive")
    if args.max_source_tokens < 1:
        raise ValueError("max_source_tokens must be positive")
    if args.predictor_hidden_dim < 1 or args.predictor_layers < 1:
        raise ValueError("predictor dimensions must be positive")
    cosine_kernel_reward(torch.ones(1), temperature=args.reward_temperature)
    _validate_training_geometry(args)

    run_dir = Path("ablation_results") / args.name
    if args.resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
    else:
        if run_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing run directory {run_dir}")
        run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.jsonl"
    if not args.resume:
        metrics_path.write_text("")
    manifest_path = run_dir / "manifest.json"
    if args.resume:
        previous_args = json.loads(manifest_path.read_text())["args"]
        current_args = _serializable_args(args)
        for key, value in current_args.items():
            if key in {"resume", "steps"}:
                continue
            if previous_args.get(key) != value:
                raise ValueError(f"resume argument {key!r} differs from the original run")
    tensorboard = SummaryWriter(log_dir=str(Path("tb_logs") / args.name))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("answer-encoder training is a model workload and requires CUDA")

    answer_examples = _load_answer_examples(args)
    train_answer_examples, val_answer_examples = _split_answer_examples(
        answer_examples,
        args.answer_validation_fraction,
        args.seed,
    )
    train_corpus = MixedCorpus(
        TokenShardCorpus(args.train_glob),
        train_answer_examples,
        args.answer_probability,
    )
    val_corpus = MixedCorpus(
        TokenShardCorpus(args.val_glob),
        val_answer_examples,
        args.answer_probability,
    )
    if args.training_objective == "multicrop":
        view_config = ViewConfig(
            global_views=args.global_views,
            local_views=args.local_views,
            global_scale_min=args.global_scale_min,
            global_scale_max=args.global_scale_max,
            local_scale_min=args.local_scale_min,
            local_scale_max=args.local_scale_max,
            mask_probability=args.mask_probability,
            block_shuffle_probability=args.block_shuffle_probability,
        )
        separator = tuple(gpt2_encode("\n\n"))
        train_sampler = ViewBatchSampler(
            train_corpus,
            view_config,
            max_view_tokens=args.max_source_tokens,
            min_source_tokens=args.min_source_tokens,
            block_separator_tokens=separator,
            seed=args.seed,
        )
        val_sampler = ViewBatchSampler(
            val_corpus,
            view_config,
            max_view_tokens=args.max_source_tokens,
            min_source_tokens=args.min_source_tokens,
            block_separator_tokens=separator,
            seed=args.seed + 1,
        )
    else:
        mask_config = SpanMaskConfig(
            probability=args.span_mask_probability,
            mean_span_length=args.mean_mask_span_length,
        )
        train_sampler = MaskedPredictionBatchSampler(
            train_corpus,
            mask_config,
            max_tokens=args.max_source_tokens,
            min_source_tokens=args.min_source_tokens,
            seed=args.seed,
        )
        val_sampler = MaskedPredictionBatchSampler(
            val_corpus,
            mask_config,
            max_tokens=args.max_source_tokens,
            min_source_tokens=args.min_source_tokens,
            seed=args.seed + 1,
        )

    encoder_config = AnswerEncoderConfig(
        max_tokens=args.max_source_tokens,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dim=_resolved_projection_dimension(args),
        reference_projector=args.reference_projector,
        projector_normalization=args.projector_normalization,
        dropout=args.dropout,
    )
    objective_config = LeJEPAObjectiveConfig(
        sigreg_weight=args.sigreg_weight,
        sigreg_knots=args.sigreg_knots,
        sigreg_slices=args.sigreg_slices,
        sigreg_t_max=args.sigreg_t_max,
    )
    model = TextAnswerEncoder(encoder_config).to(device)
    predictor = None
    if args.training_objective == "global-latent":
        predictor = GlobalLatentPredictor(
            args.model_dim,
            args.predictor_hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.predictor_layers,
        ).to(device)
        objective = GlobalLatentObjective(objective_config).to(device)
    else:
        objective = LeJEPAObjective(objective_config).to(device)
    optimized_parameters = list(model.parameters())
    if predictor is not None:
        optimized_parameters.extend(predictor.parameters())
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    scaler = torch.amp.GradScaler("cuda")
    start_step = 0
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_payload.get("schema") != ANSWER_ENCODER_SCHEMA:
            raise ValueError("resume checkpoint has the wrong schema")
        if resume_payload.get("encoder_config") != encoder_config.to_dict():
            raise ValueError("resume checkpoint encoder config differs from current arguments")
        if resume_payload.get("objective_config") != objective_config.to_dict():
            raise ValueError("resume checkpoint objective config differs from current arguments")
        _validate_resume_checkpoint_args(resume_payload.get("args"), previous_args)
        _load_module_training_state(resume_payload, model, predictor)
        optimizer.load_state_dict(resume_payload["optimizer"])
        scaler.load_state_dict(resume_payload["scaler"])
        train_sampler.load_state_dict(resume_payload["train_sampler"])
        val_sampler.load_state_dict(resume_payload["val_sampler"])
        random.setstate(resume_payload["python_rng"])
        torch.set_rng_state(resume_payload["torch_rng"])
        torch.cuda.set_rng_state_all(resume_payload["cuda_rng"])
        start_step = int(resume_payload["step"])
        if not 0 <= start_step < args.steps:
            raise ValueError("resume step must be smaller than requested total steps")
    frozen_probe_model = model
    if not args.resume:
        _atomic_json(
            manifest_path,
            _manifest(args, len(train_answer_examples), len(val_answer_examples)),
        )
    started = time.perf_counter()

    def log_evaluation(step: int) -> dict:
        if args.training_objective == "global-latent":
            if predictor is None or not isinstance(objective, GlobalLatentObjective):
                raise AssertionError("global-latent modules were not initialized")
            validation = evaluate_global_latent_objective(
                model,
                predictor,
                objective,
                val_sampler,
                batches=args.eval_batches,
                batch_size=args.batch_size,
                device=device,
            )
        else:
            validation = evaluate_objective(
                model,
                objective,
                val_sampler,
                batches=args.eval_batches,
                batch_size=args.batch_size,
                device=device,
                objective_space=args.objective_space,
            )
        for parameter in frozen_probe_model.parameters():
            parameter.requires_grad_(False)
        scorer = FrozenAnswerScorer(
            frozen_probe_model,
            gpt2_encode,
            device=device,
            space=args.reward_space,
            reward_temperature=args.reward_temperature,
        )
        probes = run_behavioral_probes(scorer)
        for parameter in frozen_probe_model.parameters():
            parameter.requires_grad_(True)
        entry = {
            "type": "val",
            "step": step,
            "train_time_ms": 1000.0 * (time.perf_counter() - started),
            **validation,
            **probes["metrics"],
        }
        _append_jsonl(metrics_path, entry)
        for key, value in entry.items():
            if isinstance(value, (int, float)) and key not in {"step", "train_time_ms"}:
                tensorboard.add_scalar(f"val/{key}", value, step)
        print(
            f"step {step}: val={validation['val_loss']:.4f} "
            f"cos={validation['view_center_cosine']:.4f} "
            f"rank={validation['objective_effective_rank']:.1f} "
            f"numeric_margin={probes['metrics']['probe/numeric_exact_min_margin']:.4f}",
            flush=True,
        )
        return {"validation": validation, "probes": probes}

    def save_training_state(step: int) -> None:
        _atomic_torch_save(
            run_dir / "training_checkpoint.pt",
            {
                "schema": ANSWER_ENCODER_SCHEMA,
                **_module_state_payload(model, predictor, training=True),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "encoder_config": encoder_config.to_dict(),
                "objective_config": objective_config.to_dict(),
                "step": step,
                "args": _serializable_args(args),
                "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "train_sampler": train_sampler.state_dict(),
                "val_sampler": val_sampler.state_dict(),
            },
        )

    latest_evaluation = None
    if start_step == 0 and not args.resume:
        latest_evaluation = log_evaluation(0)
        save_training_state(0)
    model.train()
    if predictor is not None:
        predictor.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.steps + 1):
        optimizer.param_groups[0]["lr"] = _learning_rate(
            step - 1,
            args.steps,
            args.warmup_steps,
            args.learning_rate,
            args.min_learning_rate,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if args.training_objective == "global-latent":
                if predictor is None or not isinstance(objective, GlobalLatentObjective):
                    raise AssertionError("global-latent modules were not initialized")
                batch = train_sampler.batch(args.batch_size).to(device)
                global_outputs = _encode_global_latent_batch(
                    model, predictor, batch
                )
                loss, diagnostics = objective(
                    global_outputs["predicted_global"],
                    global_outputs["target_global"],
                )
                diagnostics["masked_fraction"] = global_outputs[
                    "masked_fraction"
                ].detach()
                diagnostics["visible_fraction"] = global_outputs[
                    "visible_fraction"
                ].detach()
                diagnostics["visible_tokens_per_sample"] = global_outputs[
                    "visible_tokens_per_sample"
                ].detach()
                objective_rank_samples = global_outputs["target_global"]
            else:
                batch = tuple(
                    value.to(device)
                    for value in train_sampler.batch(args.batch_size)
                )
                objective_embeddings = _encode_view_batch(
                    model,
                    *batch,
                    objective_space=args.objective_space,
                )
                loss, diagnostics = objective(objective_embeddings)
                objective_rank_samples = objective_embeddings.mean(dim=0)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step <= 10 or step % args.log_every == 0:
            entry = {
                "type": "train",
                "step": step,
                "train_loss": float(loss.detach()),
                "lr": optimizer.param_groups[0]["lr"],
                "train_time_ms": 1000.0 * (time.perf_counter() - started),
                **{key: float(value) for key, value in diagnostics.items()},
                "objective_effective_rank": _effective_rank(
                    objective_rank_samples
                ),
                "rank_sample_count": args.batch_size,
            }
            _append_jsonl(metrics_path, entry)
            for key, value in entry.items():
                if isinstance(value, (int, float)) and key not in {"step", "train_time_ms"}:
                    tensorboard.add_scalar(f"train/{key}", value, step)

        if step % args.eval_every == 0 or step == args.steps:
            latest_evaluation = log_evaluation(step)
            save_training_state(step)
            model.train()
            if predictor is not None:
                predictor.train()

    if latest_evaluation is None:
        raise AssertionError("training completed without a final evaluation")
    checkpoint_path = run_dir / "answer_encoder.pt"
    torch.save(
        {
            "schema": ANSWER_ENCODER_SCHEMA,
            **_module_state_payload(model, predictor, training=False),
            "encoder_config": encoder_config.to_dict(),
            "objective_config": objective_config.to_dict(),
            "step": args.steps,
            "args": _serializable_args(args),
            "behavioral_gate_passed": latest_evaluation["probes"]["gate"]["passed"],
            "behavioral_gate_space": args.reward_space,
            "behavioral_gate": latest_evaluation["probes"]["gate"],
        },
        checkpoint_path,
    )
    result = {
        "schema": ANSWER_ENCODER_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "steps": args.steps,
        "train_answer_examples": len(train_answer_examples),
        "validation_answer_examples": len(val_answer_examples),
        "answer_draws": train_corpus.answer_draws,
        "answer_epochs": train_corpus.answer_draws / max(len(train_answer_examples), 1),
        "behavioral_gate_passed": latest_evaluation["probes"]["gate"]["passed"],
        "behavioral_gate_space": args.reward_space,
        "final": latest_evaluation,
    }
    _atomic_json(run_dir / "result.json", result)
    tensorboard.close()
    print(f"saved {checkpoint_path}", flush=True)


def _frozen_scorer(
    args: argparse.Namespace,
    *,
    require_behavioral_gate: bool,
) -> FrozenAnswerScorer:
    device = torch.device(args.device)
    model, _ = load_encoder_checkpoint(
        args.checkpoint,
        device,
        require_behavioral_gate=require_behavioral_gate and not args.allow_failed_gate,
        required_embedding_space=args.reward_space,
    )
    return FrozenAnswerScorer(
        model,
        gpt2_encode,
        device=device,
        space=args.reward_space,
        reward_temperature=getattr(
            args, "reward_temperature", DEFAULT_REWARD_TEMPERATURE
        ),
    )


def probe(args: argparse.Namespace) -> None:
    report = run_behavioral_probes(
        _frozen_scorer(args, require_behavioral_gate=False)
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    print(rendered)


def diagnose(args: argparse.Namespace) -> None:
    """Separate projector BN mode effects from corpus-distribution effects."""
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("answer-encoder diagnostics are a model workload and require CUDA")
    model, payload = load_encoder_checkpoint(
        args.checkpoint,
        device,
        require_behavioral_gate=False,
    )
    train_args = argparse.Namespace(**payload["args"])
    if getattr(train_args, "training_objective", "multicrop") == "global-latent":
        raise ValueError(
            "diagnose compares historical multicrop projector statistics; "
            "global-latent checkpoints already record target/prediction diagnostics"
        )
    answer_examples = _load_answer_examples(train_args)
    train_answer_examples, val_answer_examples = _split_answer_examples(
        answer_examples,
        train_args.answer_validation_fraction,
        train_args.seed,
    )
    token_corpus = TokenShardCorpus(train_args.train_glob)
    train_corpus = MixedCorpus(
        token_corpus,
        train_answer_examples,
        train_args.answer_probability,
    )
    val_corpus = MixedCorpus(
        TokenShardCorpus(train_args.val_glob),
        val_answer_examples,
        train_args.answer_probability,
    )
    view_config = ViewConfig(
        global_views=train_args.global_views,
        local_views=train_args.local_views,
        global_scale_min=train_args.global_scale_min,
        global_scale_max=train_args.global_scale_max,
        local_scale_min=train_args.local_scale_min,
        local_scale_max=train_args.local_scale_max,
        mask_probability=train_args.mask_probability,
        block_shuffle_probability=train_args.block_shuffle_probability,
    )
    separator = tuple(gpt2_encode("\n\n"))

    def collect_batches(corpus, seed: int) -> list[tuple[Tensor, Tensor]]:
        sampler = ViewBatchSampler(
            corpus,
            view_config,
            max_view_tokens=train_args.max_source_tokens,
            min_source_tokens=train_args.min_source_tokens,
            block_separator_tokens=separator,
            seed=seed,
        )
        return [sampler.batch(args.batch_size) for _ in range(args.batches)]

    corpora = {
        "train_mixture": collect_batches(train_corpus, train_args.seed + 101),
        "validation_mixture": collect_batches(val_corpus, train_args.seed + 102),
    }
    objective = LeJEPAObjective(
        LeJEPAObjectiveConfig(**payload["objective_config"])
    ).to(device)
    report = {
        "schema": "text_lejepa_answer_encoder_diagnostic/v2",
        "checkpoint": str(args.checkpoint),
        "batch_size": args.batch_size,
        "batches": args.batches,
        "corpora": {},
    }
    for corpus_name, batches in corpora.items():
        report["corpora"][corpus_name] = {
            "stored_running_statistics": _diagnose_batches(
                model,
                objective,
                batches,
                device=device,
                projector_batch_statistics=False,
                objective_space=getattr(train_args, "objective_space", "projection"),
            ),
            "current_batch_statistics": _diagnose_batches(
                model,
                objective,
                batches,
                device=device,
                projector_batch_statistics=True,
                objective_space=getattr(train_args, "objective_space", "projection"),
            ),
        }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    print(rendered)


def score(args: argparse.Namespace) -> None:
    scorer = _frozen_scorer(args, require_behavioral_gate=True)
    target = scorer.preencode_target(args.target)
    cosine = scorer.cosine(args.answer, target)
    rewards = scorer.score(args.answer, target)
    print(
        json.dumps(
            {
                "target": args.target,
                "reward_temperature": args.reward_temperature,
                "scores": [
                    {
                        "answer": answer,
                        "cosine": float(similarity),
                        "reward": float(reward),
                    }
                    for answer, similarity, reward in zip(
                        args.answer, cosine, rewards, strict=True
                    )
                ],
            },
            indent=2,
        )
    )


def cache_targets(args: argparse.Namespace) -> None:
    scorer = _frozen_scorer(args, require_behavioral_gate=True)
    targets = {}
    for item in args.target:
        if "=" not in item:
            raise ValueError("cached targets must use ID=TEXT syntax")
        identifier, text = item.split("=", 1)
        if not identifier or identifier in targets:
            raise ValueError("target identifiers must be nonempty and unique")
        targets[identifier] = scorer.preencode_target(text)
    save_target_cache(
        args.output,
        checkpoint=args.checkpoint,
        space=args.reward_space,
        targets=targets,
    )
    print(f"cached {len(targets)} targets in {args.output}")


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reward-space", choices=("projection", "backbone"), default="projection")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--reward-temperature",
        type=float,
        default=DEFAULT_REWARD_TEMPERATURE,
        help="temperature for exp((cosine - 1) / temperature)",
    )
    parser.add_argument(
        "--allow-failed-gate",
        action="store_true",
        help="allow diagnostic scoring with an encoder that failed the behavioral gate",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train the self-supervised answer encoder")
    train_parser.set_defaults(function=train)
    train_parser.add_argument("--name", required=True)
    train_parser.add_argument("--resume")
    train_parser.add_argument("--train-glob", default=DEFAULT_TRAIN_GLOB)
    train_parser.add_argument("--val-glob", default=DEFAULT_VAL_GLOB)
    train_parser.add_argument("--answer-source", action="append")
    train_parser.add_argument("--max-answer-examples", type=int, default=25_000)
    train_parser.add_argument(
        "--deduplicate-answer-text",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_parser.add_argument("--answer-validation-fraction", type=float, default=0.05)
    # Keep broad, effectively unique pretraining records dominant so SIGReg
    # cannot be solved by arranging a small vocabulary of repeated answers.
    train_parser.add_argument("--answer-probability", type=float, default=0.2)
    train_parser.add_argument("--steps", type=int, default=2_000)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--max-source-tokens", type=int, default=256)
    train_parser.add_argument("--min-source-tokens", type=int, default=32)
    train_parser.add_argument("--model-dim", type=int, default=256)
    train_parser.add_argument("--num-layers", type=int, default=4)
    train_parser.add_argument("--num-heads", type=int, default=8)
    train_parser.add_argument("--mlp-ratio", type=int, default=4)
    train_parser.add_argument("--projection-hidden-dim", type=int, default=2048)
    train_parser.add_argument(
        "--projection-dim",
        type=int,
        default=None,
        help="projected CLS width (default: 256 for global-latent, 16 for multicrop)",
    )
    train_parser.add_argument(
        "--reference-projector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="match LeJEPA's torchvision MLP defaults (ReLU with linear biases)",
    )
    train_parser.add_argument(
        "--projector-normalization",
        choices=("batch", "layer"),
        default="layer",
        help="use per-example LayerNorm for a deployable projector",
    )
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument(
        "--training-objective",
        choices=("multicrop", "global-latent"),
        default="multicrop",
        help="use v5 multicrop agreement or CLS-only variable-cardinality prediction",
    )
    train_parser.add_argument("--predictor-hidden-dim", type=int, default=1024)
    train_parser.add_argument("--predictor-layers", type=int, default=2)
    train_parser.add_argument("--span-mask-probability", type=float, default=0.3)
    train_parser.add_argument("--mean-mask-span-length", type=float, default=3.0)
    train_parser.add_argument("--global-views", type=int, default=2)
    train_parser.add_argument("--local-views", type=int, default=6)
    train_parser.add_argument("--global-scale-min", type=float, default=0.3)
    train_parser.add_argument("--global-scale-max", type=float, default=1.0)
    train_parser.add_argument("--local-scale-min", type=float, default=0.05)
    train_parser.add_argument("--local-scale-max", type=float, default=0.3)
    train_parser.add_argument("--mask-probability", type=float, default=0.1)
    train_parser.add_argument("--block-shuffle-probability", type=float, default=0.0)
    train_parser.add_argument("--sigreg-weight", type=float, default=0.02)
    train_parser.add_argument("--sigreg-knots", type=int, default=17)
    train_parser.add_argument("--sigreg-slices", type=int, default=256)
    train_parser.add_argument("--sigreg-t-max", type=float, default=3.0)
    train_parser.add_argument(
        "--reward-temperature",
        type=float,
        default=DEFAULT_REWARD_TEMPERATURE,
        help="temperature used only when converting probe cosine to reward",
    )
    train_parser.add_argument("--learning-rate", type=float, default=2e-3)
    train_parser.add_argument("--min-learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=0.05)
    train_parser.add_argument("--warmup-steps", type=int, default=100)
    train_parser.add_argument("--eval-every", type=int, default=100)
    train_parser.add_argument("--eval-batches", type=int, default=4)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument(
        "--objective-space",
        choices=("projection", "backbone"),
        default="projection",
        help="representation receiving invariance and SIGReg",
    )
    train_parser.add_argument("--reward-space", choices=("projection", "backbone"), default="projection")
    train_parser.add_argument("--seed", type=int, default=1234)

    probe_parser = subparsers.add_parser("probe", help="run the frozen behavioral suite")
    probe_parser.set_defaults(function=probe)
    _add_inference_arguments(probe_parser)
    probe_parser.add_argument("--output")

    diagnose_parser = subparsers.add_parser(
        "diagnose", help="compare projector batch statistics across train and validation data"
    )
    diagnose_parser.set_defaults(function=diagnose)
    diagnose_parser.add_argument("--checkpoint", required=True)
    diagnose_parser.add_argument("--device", default="cuda")
    diagnose_parser.add_argument("--batch-size", type=int, default=32)
    diagnose_parser.add_argument("--batches", type=int, default=8)
    diagnose_parser.add_argument("--output")

    score_parser = subparsers.add_parser("score", help="score answers against one target")
    score_parser.set_defaults(function=score)
    _add_inference_arguments(score_parser)
    score_parser.add_argument("--target", required=True)
    score_parser.add_argument("--answer", action="append", required=True)

    cache_parser = subparsers.add_parser("cache-targets", help="pre-encode named targets")
    cache_parser.set_defaults(function=cache_targets)
    _add_inference_arguments(cache_parser)
    cache_parser.add_argument("--target", action="append", required=True, help="ID=TEXT")
    cache_parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

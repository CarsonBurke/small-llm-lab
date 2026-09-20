"""CUDA Mini native-bit, adaptive, dynamics and recurrent-embedder comparisons."""

from __future__ import annotations

import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from pretraining.nanogpt_mini.nanogpt_mini_adaptive_model import (
    AdaptiveConfig,
    AdaptiveGPT,
)
from pretraining.nanogpt_mini.nanogpt_mini_bitflow_model import (
    BitFlowConfig,
    BitFlowGPT,
)
from pretraining.nanogpt_mini.nanogpt_mini_character_model import (
    CharacterConfig,
    CharacterGPT,
)
from pretraining.nanogpt_mini.nanogpt_mini_dynamics_bounded_model import (
    BoundedDynamicsGPT,
)
from pretraining.nanogpt_mini.nanogpt_mini_dynamics_model import (
    DynamicsConfig,
    DynamicsGPT,
)
from pretraining.nanogpt_mini.nanogpt_mini_embedder_model import (
    EmbedderConfig,
    EmbedderGPT,
)
from pretraining.nanogpt_mini.nanogpt_mini_native_bits_model import (
    NativeBitsConfig,
    NativeBitsGPT,
)
from pretraining.nanogpt_mini.native_bits_data import (
    PreparedData,
    cyclic_ids,
    microbatches,
    sha256_file,
    validate_alphabet,
)

MODEL_TYPES = {
    "native": (NativeBitsConfig, NativeBitsGPT),
    "bitflow": (BitFlowConfig, BitFlowGPT),
    "softmax": (CharacterConfig, CharacterGPT),
    "adaptive": (AdaptiveConfig, AdaptiveGPT),
    "dynamics": (DynamicsConfig, DynamicsGPT),
    "dynamics_bounded": (DynamicsConfig, BoundedDynamicsGPT),
    "embedder": (EmbedderConfig, EmbedderGPT),
}
DYNAMICS_MODEL_KINDS = frozenset({"dynamics", "dynamics_bounded"})
ARCHITECTURES = {
    "native": "nanogpt_mini_native_bits_v1",
    "bitflow": "nanogpt_mini_bitflow_v1",
    "softmax": "nanogpt_mini_character_v1",
    "adaptive": "nanogpt_mini_adaptive_characters_v1",
    "dynamics": "nanogpt_mini_dynamics_characters_v1",
    "dynamics_bounded": BoundedDynamicsGPT.architecture,
    "embedder": "nanogpt_mini_autoregressive_embedder_v1",
}
RATE_LABELS = {
    "native": "two_part",
    "bitflow": "code_space",
    "softmax": "character",
    "adaptive": "code_space",
    "dynamics": "code_space",
    "dynamics_bounded": "code_space",
    "embedder": "character",
}
RATE_DESCRIPTIONS = {
    "native": "two_part_latent_and_identity_residual_nats; not exact marginal character likelihood",
    "bitflow": "code_space_nats; unused identity mass remains charged, not renormalized over valid characters",
    "softmax": "matched_character_softmax_nats; normalized over observed alphabet, not challenge heldout",
    "adaptive": "code_space_nats; deterministic causal refresh, full code-domain likelihood without valid-character renormalization; not challenge heldout",
    "dynamics": "code_space_nats; dense teacher likelihood during training and evaluation, preserved by verified speculative decoding; full code-domain likelihood without valid-character renormalization; auxiliaries excluded; not challenge heldout",
    "dynamics_bounded": "code_space_nats; dense teacher likelihood with RMS-normalized recurrent proposals; verified speculative decoding preserves the target; auxiliaries excluded; not challenge heldout",
    "embedder": "character_softmax_nats; observed-alphabet likelihood under deterministic causal emission decisions; policy, compute and critic auxiliaries excluded; not challenge heldout",
}
ModelConfig = (
    NativeBitsConfig
    | BitFlowConfig
    | CharacterConfig
    | AdaptiveConfig
    | DynamicsConfig
    | EmbedderConfig
)
Model = (
    NativeBitsGPT | BitFlowGPT | CharacterGPT | AdaptiveGPT | DynamicsGPT | EmbedderGPT
)


@dataclass(frozen=True)
class TrainConfig:
    data_path: str
    run_id: str = "native_bits"
    seq_len: int = 1024
    mbs: int = 8
    batch_characters: int = 8192
    val_characters: int = 65536
    iterations: int = 2000
    val_every: int = 20
    log_every: int = 10
    seed: int = 1337
    codec_weight: float = 1.0
    code_lr: float = 0.004
    resume: str | None = None
    model_kind: str = "native"

    def __post_init__(self):
        if self.model_kind not in MODEL_TYPES:
            raise ValueError(f"MODEL_KIND must be one of {', '.join(MODEL_TYPES)}")
        if not self.data_path:
            raise ValueError("NATIVE_DATA_PATH must name a prepared native-bit dataset")
        for name in (
            "seq_len",
            "mbs",
            "batch_characters",
            "val_characters",
            "iterations",
            "val_every",
            "log_every",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.codec_weight) or self.codec_weight < 0:
            raise ValueError("CODEC_WEIGHT must be finite and nonnegative")
        if not math.isfinite(self.code_lr) or self.code_lr <= 0:
            raise ValueError("CODE_LR must be finite and positive")
        if not 0 <= self.seed < 2**32:
            raise ValueError("SEED must fit an unsigned 32-bit integer")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("RUN_ID must be a simple directory name")

    @classmethod
    def from_env(cls):
        return cls(
            data_path=os.environ.get("NATIVE_DATA_PATH", ""),
            run_id=os.environ.get("RUN_ID", "native_bits"),
            seq_len=int(os.environ.get("SEQ_LEN", "1024")),
            mbs=int(os.environ.get("MBS", "8")),
            batch_characters=int(os.environ.get("BATCH_CHARACTERS", "8192")),
            val_characters=int(os.environ.get("VAL_CHARACTERS", "65536")),
            iterations=int(os.environ.get("ITERATIONS", "2000")),
            val_every=int(os.environ.get("VAL_LOSS_EVERY", "20")),
            log_every=int(os.environ.get("TRAIN_LOG_EVERY", "10")),
            seed=int(os.environ.get("SEED", "1337")),
            codec_weight=float(os.environ.get("CODEC_WEIGHT", "1")),
            code_lr=float(os.environ.get("CODE_LR", "0.004")),
            resume=os.environ.get("NATIVE_RESUME") or None,
            model_kind=os.environ.get("MODEL_KIND", "native"),
        )


def validate_model_environment(model_kind: str, settings: Mapping[str, str]) -> None:
    if model_kind not in MODEL_TYPES:
        raise ValueError(f"MODEL_KIND must be one of {', '.join(MODEL_TYPES)}")
    model_family = "dynamics" if model_kind in DYNAMICS_MODEL_KINDS else model_kind
    exclusive_options = {
        "adaptive": {"LOCAL_LAYERS", "LOCAL_DIM"},
        "embedder": {
            "EMBEDDER_DIM",
            "GATE_MODE",
            "FIXED_STRIDE",
            "EMISSION_COST",
            "BASELINE_LOSS_WEIGHT",
        },
        "dynamics": {
            "DYNAMICS_WIDTH",
            "ROLLOUT_HORIZON",
            "LATENT_WEIGHT",
            "ROLLOUT_WEIGHT",
            "GATE_WEIGHT",
        },
    }
    for owner, options in exclusive_options.items():
        if model_family != owner and (irrelevant := options & settings.keys()):
            raise ValueError(f"{owner}-only environment settings: {sorted(irrelevant)}")
    if model_family not in {"adaptive", "dynamics"} and "REFRESH_COST" in settings:
        raise ValueError(
            "REFRESH_COST environment settings require adaptive or dynamics"
        )
    if model_family in {"adaptive", "dynamics", "embedder"}:
        irrelevant = {
            "CODEC_DIM",
            "FLOW_LAYERS",
            "LATENT_BITS",
            "MIXTURE_COMPONENTS",
            "LEARN_CODES",
            "LEARN_CODEC",
            "MASK_SURROGATE",
            "DENSITY_HEAD",
            *(
                {"CODEC_WEIGHT", "CODE_LR"}
                if model_family in {"dynamics", "embedder"}
                else set()
            ),
            *({"CODE_BITS", "PREFIX_WIDTH"} if model_family == "embedder" else set()),
        } & settings.keys()
        if irrelevant:
            raise ValueError(
                f"environment settings not used by {model_kind}: {sorted(irrelevant)}"
            )


def model_config_from_env(vocab_size: int, model_kind: str) -> ModelConfig:
    validate_model_environment(model_kind, os.environ)
    common = {
        "vocab_size": vocab_size,
        "num_layers": int(os.environ.get("NUM_LAYERS", "6")),
        "model_dim": int(os.environ.get("MODEL_DIM", "512")),
    }
    if model_kind in DYNAMICS_MODEL_KINDS:
        return DynamicsConfig(
            **common,
            code_bits=int(os.environ.get("CODE_BITS", "0")),
            prefix_width=int(os.environ.get("PREFIX_WIDTH", "128")),
            dynamics_width=int(os.environ.get("DYNAMICS_WIDTH", "128")),
            rollout_horizon=int(os.environ.get("ROLLOUT_HORIZON", "2")),
            latent_weight=float(os.environ.get("LATENT_WEIGHT", "1")),
            rollout_weight=float(os.environ.get("ROLLOUT_WEIGHT", "1")),
            gate_weight=float(os.environ.get("GATE_WEIGHT", "1")),
            refresh_cost=float(os.environ.get("REFRESH_COST", "0.02")),
        )
    if model_kind == "adaptive":
        return AdaptiveConfig(
            **common,
            code_bits=int(os.environ.get("CODE_BITS", "0")),
            local_layers=int(os.environ.get("LOCAL_LAYERS", "2")),
            local_dim=int(os.environ.get("LOCAL_DIM", "128")),
            prefix_width=int(os.environ.get("PREFIX_WIDTH", "128")),
            refresh_cost=float(os.environ.get("REFRESH_COST", "0.02")),
        )
    if model_kind == "softmax":
        return CharacterConfig(**common)
    if model_kind == "embedder":
        return EmbedderConfig(
            **common,
            embedder_dim=int(os.environ.get("EMBEDDER_DIM", "128")),
            gate_mode=os.environ.get("GATE_MODE", "learned"),
            fixed_stride=int(os.environ.get("FIXED_STRIDE", "4")),
            emission_cost=float(os.environ.get("EMISSION_COST", "0.02")),
            baseline_loss_weight=float(os.environ.get("BASELINE_LOSS_WEIGHT", "0.01")),
        )
    codec_dim = int(os.environ.get("CODEC_DIM", "64"))
    if model_kind == "bitflow":
        learn_codec = os.environ.get("LEARN_CODEC", "1")
        if learn_codec not in {"0", "1"}:
            raise ValueError("LEARN_CODEC must be 0 or 1")
        density_head = os.environ.get("DENSITY_HEAD", "mixture")
        return BitFlowConfig(
            **common,
            code_bits=int(os.environ.get("CODE_BITS", "0")),
            codec_dim=codec_dim,
            flow_layers=int(os.environ.get("FLOW_LAYERS", "4")),
            mixture_components=int(
                os.environ.get(
                    "MIXTURE_COMPONENTS", "1" if density_head == "prefix" else "8"
                )
            ),
            learn_codec=learn_codec == "1",
            mask_surrogate=os.environ.get("MASK_SURROGATE", "sigmoid"),
            density_head=density_head,
            prefix_width=int(os.environ.get("PREFIX_WIDTH", "128")),
        )
    learn_codes = os.environ.get("LEARN_CODES", "1")
    if learn_codes not in {"0", "1"}:
        raise ValueError("LEARN_CODES must be 0 or 1")
    return NativeBitsConfig(
        **common,
        code_bits=int(os.environ.get("CODE_BITS", "32")),
        latent_bits=int(os.environ.get("LATENT_BITS", "8")),
        codec_dim=codec_dim,
        mixture_components=int(os.environ.get("MIXTURE_COMPONENTS", "1")),
        learn_codes=learn_codes == "1",
    )


def checkpoint_model_kind(checkpoint: dict) -> str:
    for model_kind, architecture in ARCHITECTURES.items():
        if checkpoint.get("architecture") == architecture:
            return model_kind
    raise ValueError("not a supported Mini character-comparison checkpoint")


def checkpoint_model_config(checkpoint: dict) -> ModelConfig:
    model_kind = checkpoint_model_kind(checkpoint)
    config_type, _ = MODEL_TYPES[model_kind]
    metadata = checkpoint["model_config"]
    if model_kind in DYNAMICS_MODEL_KINDS | {"adaptive", "embedder"}:
        if not isinstance(metadata, dict):
            raise ValueError(f"{model_kind} model_config must be a dictionary")
        expected = {field.name for field in fields(config_type)}
        if missing := expected - metadata.keys():
            raise ValueError(f"incomplete {model_kind} model_config: {sorted(missing)}")
        if unknown := metadata.keys() - expected:
            raise ValueError(f"unknown {model_kind} model_config: {sorted(unknown)}")
    return config_type(**metadata)


def checkpoint_transport_widths(checkpoint: dict) -> tuple[int, int]:
    """Derive framing from checkpoint metadata without constructing a CUDA model."""
    config = checkpoint_model_config(checkpoint)
    if isinstance(config, NativeBitsConfig):
        widths = config.latent_bits, max(1, (config.vocab_size - 1).bit_length())
    elif isinstance(config, (BitFlowConfig, AdaptiveConfig, DynamicsConfig)):
        widths = config.code_bits, 0
    else:
        widths = max(1, (config.vocab_size - 1).bit_length()), 0
    if not 1 <= widths[0] <= 256 or not 0 <= widths[1] <= 256:
        raise ValueError("checkpoint widths exceed the packet format")
    return widths


def cuda_device() -> torch.device:
    if (
        int(os.environ.get("WORLD_SIZE", "1")) != 1
        or int(os.environ.get("RANK", "0")) != 0
    ):
        raise ValueError("native bits supports one CUDA process only")
    if not torch.cuda.is_available():
        raise RuntimeError("native bits requires CUDA; no CPU model fallback")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("native bits requires CUDA bfloat16 support")
    return device


def load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a model metadata dictionary")
    checkpoint_model_kind(checkpoint)
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("unsupported Mini comparison checkpoint version")
    required = {
        "model",
        "model_config",
        "alphabet",
        "provenance",
        "optimizer",
        "rng",
        "step",
        "train_config",
        "data_cursor",
    }
    if missing := required - checkpoint.keys():
        raise ValueError(f"incomplete checkpoint: {sorted(missing)}")
    config = checkpoint_model_config(checkpoint)
    alphabet = validate_alphabet(checkpoint["alphabet"])
    if len(alphabet) != config.vocab_size:
        raise ValueError("checkpoint alphabet size differs from its model")
    return checkpoint


def load_model(checkpoint: dict) -> tuple[Model, torch.device]:
    model_config = checkpoint_model_config(checkpoint)
    _, model_type = MODEL_TYPES[checkpoint_model_kind(checkpoint)]
    device = cuda_device()
    model = model_type(model_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.compile_components()
    model.eval()
    return model, device


@torch.compile
def muon_update(grad, momentum, mu=0.95):
    # Exact original Mini twelve-iteration arithmetic, without DDP side effects.
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu).bfloat16()
    transposed = grad.size(-2) > grad.size(-1)
    if transposed:
        update = update.mT
    update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(12):
        gram = update @ update.mT
        update = 2 * update + (-1.5 * gram + 0.5 * gram @ gram) @ update
    if transposed:
        update = update.mT
    return update * max(1, grad.size(-2) / grad.size(-1)) ** 0.5


class Muon(torch.optim.Optimizer):
    def __init__(self, parameters):
        super().__init__(
            sorted(parameters, key=lambda p: p.size(), reverse=True),
            {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95},
        )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    raise RuntimeError("missing prior-block gradient")
                state = self.state[parameter]
                if not state:
                    state["momentum"] = torch.zeros_like(parameter)
                update = muon_update(parameter.grad, state["momentum"], group["mu"])
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])


def make_optimizers(model: Model, config: TrainConfig) -> list:
    matrices = [
        p for p in model.prior.blocks.parameters() if p.ndim >= 2 and p.requires_grad
    ]
    if isinstance(model, AdaptiveGPT):
        matrices.extend(
            p
            for p in model.local.blocks.parameters()
            if p.ndim >= 2 and p.requires_grad
        )
    matrix_ids = {id(p) for p in matrices}
    embedding = getattr(model, "table", None)
    table = embedding.weight if embedding is not None else None
    other_weights = [
        p
        for p in model.parameters()
        if p.requires_grad
        and p.ndim >= 2
        and id(p) not in matrix_ids
        and p is not table
    ]
    scalars = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [{"params": other_weights, "lr": 0.004}, {"params": scalars, "lr": 0.015}]
    if table is not None and table.requires_grad:
        groups.append(
            {"params": [table], "lr": getattr(model, "embedding_lr", config.code_lr)}
        )
    optimizers = [
        torch.optim.AdamW(
            groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True
        ),
        Muon(matrices),
    ]
    assigned = [
        p
        for optimizer in optimizers
        for group in optimizer.param_groups
        for p in group["params"]
    ]
    if len(assigned) != len(set(assigned)) or set(assigned) != {
        p for p in model.parameters() if p.requires_grad
    }:
        raise RuntimeError("optimizer groups do not partition trainable parameters")
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers


def to_cuda(ids: np.ndarray, device: torch.device) -> torch.Tensor:
    # Copy a bounded microbatch out of the read-only uint32 memmap.
    return (
        torch.from_numpy(ids.astype(np.int64))
        .pin_memory()
        .to(device, non_blocking=True)
    )


def stats_vector(
    model: Model, total: torch.Tensor, stats: dict, count: int
) -> torch.Tensor:
    return torch.stack(
        (
            total.detach(),
            *(stats[name].detach() for name in model.sum_stat_names),
            *(stats[name].detach() * count for name in model.mean_stat_names),
        )
    ).float()


def unpack_stats(
    model: Model, values: torch.Tensor, count: int, byte_count: int
) -> dict[str, float]:
    metrics = dict(
        zip(
            ("objective_nats", *model.sum_stat_names, *model.mean_stat_names),
            values.cpu().tolist(),
            strict=True,
        )
    )
    for name in model.mean_stat_names:
        metrics[name] /= count
    metrics["loss"] = metrics["rate_nats"] / count
    metrics["objective"] = metrics["objective_nats"] / count
    for name in model.sum_stat_names:
        metrics[f"{name}_per_character"] = metrics[name] / count
        if name == "codec_nats":
            metrics["codec_loss"] = metrics[name] / count
        elif name.endswith("_nats") and name != "compute_nats":
            stem = name.removesuffix("_nats")
            metrics["bpb" if stem == "rate" else f"{stem}_bpb"] = (
                metrics[name] / math.log(2) / byte_count
            )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("non-finite Mini comparison metrics")
    return metrics


@torch.no_grad()
def evaluate(
    model: Model, data: PreparedData, config: TrainConfig, device: torch.device
) -> dict:
    model.eval()
    sums = torch.zeros(
        1 + len(model.sum_stat_names) + len(model.mean_stat_names), device=device
    )
    seen = byte_count = 0
    for ids in microbatches(
        data.validation[: config.val_characters], config.seq_len, config.mbs
    ):
        total, stats = model(to_cuda(ids, device), config.codec_weight)
        sums += stats_vector(model, total, stats, ids.size)
        seen += ids.size
        byte_count += int(data.byte_lengths[ids].sum())
    if seen != config.val_characters:
        raise RuntimeError("validation failed to score the entire fixed prefix")
    metrics = unpack_stats(model, sums, seen, byte_count)
    metrics["proxy_bpb"] = metrics["bpb"]
    metrics[f"{RATE_LABELS[config.model_kind]}_bpb"] = metrics["bpb"]
    metrics.update(source_characters=seen, source_bytes=byte_count)
    model.train()
    return metrics


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".checkpoint-", suffix=".pt", delete=False
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_hashes(model_kind: str) -> dict[str, str]:
    _, model_type = MODEL_TYPES[model_kind]
    names = [
        Path(__file__),
        Path(__file__).with_name(f"{model_type.__module__.rsplit('.', 1)[1]}.py"),
        Path(__file__).with_name("nanogpt_mini_model.py"),
        Path(__file__).with_name("native_bits_data.py"),
        Path(__file__).with_name("native_bits_wire.py"),
        REPO_ROOT / "scripts/native_bits.py",
    ]
    if model_kind not in {"softmax", "embedder"}:
        names.append(Path(__file__).with_name("bit_density.py"))
    if model_kind == "adaptive":
        names.append(Path(__file__).with_name("adaptive_generation.py"))
    elif model_kind in {"embedder", "softmax"}:
        names.extend(
            Path(__file__).with_name(name)
            for name in (
                "mini_cached.py",
                "embedder_generation.py"
                if model_kind == "embedder"
                else "character_generation.py",
            )
        )
    elif model_kind in DYNAMICS_MODEL_KINDS:
        names.extend(
            Path(__file__).with_name(name)
            for name in (
                "dynamics_generation.py",
                "speculative_generation.py",
                "nanogpt_mini_bitflow_model.py",
            )
        )
        if model_kind == "dynamics_bounded":
            names.append(Path(__file__).with_name("nanogpt_mini_dynamics_model.py"))
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in names}


@torch.compile(fullgraph=True)
def gradients_are_finite(gradients: list[torch.Tensor]) -> torch.Tensor:
    # Checking a norm can overflow while every gradient element is finite.
    # Reduce booleans directly: no squaring, clipping, or gradient mutation.
    return torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()


def train() -> Path:
    config = TrainConfig.from_env()
    data = PreparedData(config.data_path)
    if config.val_characters > data.validation.size:
        raise ValueError(
            "VAL_CHARACTERS exceeds the heldout split; validation never wraps or drops characters"
        )
    model_config = model_config_from_env(len(data.alphabet), config.model_kind)
    destination = REPO_ROOT / "ablation_results" / config.run_id / "checkpoint.pt"
    if destination.exists() and not config.resume:
        raise FileExistsError(
            "checkpoint exists; choose another RUN_ID or set NATIVE_RESUME"
        )
    previous = load_checkpoint(config.resume) if config.resume else None
    if previous is not None:
        if (
            checkpoint_model_kind(previous) == "bitflow"
            and "mask_surrogate" not in previous["model_config"]
        ):
            raise ValueError(
                "historical bitflow checkpoint lacks mask_surrogate; annotate its "
                "verified original rule before resuming, rather than guessing"
            )
        if (
            checkpoint_model_kind(previous) != config.model_kind
            or asdict(checkpoint_model_config(previous)) != asdict(model_config)
            or previous["alphabet"] != data.alphabet
            or previous["provenance"]["data"] != data.metadata
        ):
            raise ValueError("resume model, alphabet or data provenance mismatch")
        previous_train_config = asdict(TrainConfig(**previous["train_config"]))
        for name, value in asdict(config).items():
            if (
                name
                not in {
                    "resume",
                    "run_id",
                    "data_path",
                    "val_every",
                    "log_every",
                    "iterations",
                }
                and previous_train_config[name] != value
            ):
                raise ValueError(f"resume training setting differs: {name}")
        # Allow a completed run to be extended by increasing the target step.
        # Do not permit shortening the schedule or resuming before the checkpoint.
        if config.iterations < previous["train_config"]["iterations"]:
            raise ValueError("resume target iterations cannot decrease")
        if (
            not 0 <= previous["step"] <= config.iterations
            or previous["data_cursor"] != previous["step"] * config.batch_characters
        ):
            raise ValueError("invalid resume step/cursor")
        if config.model_kind in DYNAMICS_MODEL_KINDS | {"embedder"} and previous[
            "provenance"
        ].get("source_hashes") != source_hashes(config.model_kind):
            raise ValueError(f"resume {config.model_kind} source provenance mismatch")
    device = cuda_device()
    torch.manual_seed(config.seed)
    _, model_type = MODEL_TYPES[config.model_kind]
    model = model_type(model_config).to(device)
    if previous is not None:
        model.load_state_dict(previous["model"], strict=True)
    optimizers = make_optimizers(model, config)
    if previous is not None:
        for optimizer, state in zip(optimizers, previous["optimizer"], strict=True):
            optimizer.load_state_dict(state)
    model.compile_components()
    if getattr(model, "compile_forward", True):
        model.compile(dynamic=False)
    step = previous["step"] if previous is not None else 0
    cursor = previous["data_cursor"] if previous is not None else 0
    training_ms = previous["training_ms"] if previous is not None else 0.0
    if previous is not None:
        torch.set_rng_state(previous["rng"]["cpu"])
        torch.cuda.set_rng_state(previous["rng"]["cuda"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    provenance = {
        "data": data.metadata,
        "source_hashes": source_hashes(config.model_kind),
        "model_kind": config.model_kind,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "precision": "bfloat16_internal",
        "rate": RATE_DESCRIPTIONS[config.model_kind],
        "evaluation": "matched prepared-character split; proxy, not challenge heldout",
        "transport": "raw packed latent/residual bits, not entropy compressed",
    }
    if isinstance(model, BitFlowGPT):
        provenance["codec_master_dtype"] = str(next(model.codec.parameters()).dtype)
    if config.model_kind in DYNAMICS_MODEL_KINDS:
        provenance.update(
            training_objective=(
                "one dense teacher backbone pass; teacher NLL plus weighted latent "
                "smooth-L1, rollout bitwise KL and causal error-predictor losses; "
                "all auxiliary teacher states and head parameters detached"
            ),
            evaluation=(
                "dense teacher likelihood on the matched prepared-character split; "
                "verified speculative decoding preserves this target distribution; "
                "auxiliaries and legacy approximate rollout are excluded; "
                "proxy, not challenge heldout"
            ),
            refresh=(
                "legacy learned gate is diagnostic only; exact speculation uses "
                "target verification and an inference-only proposal budget"
            ),
            recurrent_state=(
                "learned RMS-normalized manifold"
                if config.model_kind == "dynamics_bounded"
                else "unbounded residual"
            ),
            runtime_compute=(
                "one target suffix call verifies multiple proposals; actual work "
                "includes rejected suffixes and rollback; fewer calls do not imply "
                "fewer positions; prefill and decode are counted separately"
            ),
        )
    elif config.model_kind == "embedder":
        provenance.update(
            training_objective=(
                "one packed global event route plus persistent causal character encoder; "
                "observed-character NLL, expected event charge, suffix-REINFORCE and "
                "detached-input mean-future-NLL critic"
            ),
            evaluation=(
                "observed-alphabet softmax under deterministic causal emission decisions; "
                "every character scored; proxy, not challenge heldout"
            ),
            runtime_compute=(
                "only BOS and emitted summaries enter the global Transformer/event KV cache; "
                "useful and padded global positions, encoder work and scan compositions "
                "are reported separately; no periodic forced emission"
            ),
        )
    print(
        f"model_kind:{config.model_kind} model_config: {asdict(model_config)} train_config: {asdict(config)}"
        f" parameter_count:{parameter_count} trainable_parameter_count:{trainable_parameter_count}",
        flush=True,
    )
    print(
        f"modeled rate: {provenance['rate']}; auxiliaries excluded; not actual packet compression",
        flush=True,
    )
    print(
        "alphabet is observed train/heldout Unicode scalars shuffled into opaque IDs; no BPE or Unicode features",
        flush=True,
    )
    if config.model_kind in {"softmax", "embedder"}:
        provenance["embedding_lr"] = model.embedding_lr
        print(
            f"character embedding_lr:{model.embedding_lr:g} follows original Mini, not native CODE_LR",
            flush=True,
        )
    if data.metadata["byte_cache"]:
        print(
            "validation uses extracted byte-cache validation_web, NOT original sp1024 heldout text",
            flush=True,
        )
    while True:
        if step % config.val_every == 0 or step == config.iterations:
            metrics = evaluate(model, data, config, device)
            save_checkpoint(
                destination,
                {
                    "architecture": ARCHITECTURES[config.model_kind],
                    "checkpoint_version": 1,
                    "model": model.state_dict(),
                    "model_config": asdict(model_config),
                    "alphabet": data.alphabet,
                    "provenance": provenance,
                    "train_config": asdict(config),
                    "step": step,
                    "data_cursor": cursor,
                    "optimizer": [optimizer.state_dict() for optimizer in optimizers],
                    "rng": {
                        "cpu": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state(),
                    },
                    "training_ms": training_ms,
                    "validation": metrics,
                },
            )
            extras = " ".join(
                f"val_{name}:{value:.8g}"
                for name, value in metrics.items()
                if name not in {"loss", "bpb"} and not name.endswith("_nats")
            )
            print(
                f"step:{step}/{config.iterations} val_loss:{metrics['loss']:.8g} val_bpb:{metrics['bpb']:.8g}"
                f" train_time:{training_ms:.3f}ms step_avg:{training_ms / max(step, 1):.3f}ms {extras}",
                flush=True,
            )
        if step == config.iterations:
            break
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        model.zero_grad(set_to_none=True)
        sums = torch.zeros(
            1 + len(model.sum_stat_names) + len(model.mean_stat_names), device=device
        )
        batch = cyclic_ids(data.train, cursor, config.batch_characters)
        for ids in microbatches(batch, config.seq_len, config.mbs):
            total, stats = model(to_cuda(ids, device), config.codec_weight)
            sums += stats_vector(model, total, stats, ids.size)
            total.backward()
        metrics = unpack_stats(
            model, sums, batch.size, int(data.byte_lengths[batch].sum())
        )
        if config.model_kind in DYNAMICS_MODEL_KINDS:
            metrics["teacher_bpb"] = metrics["bpb"]
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is None:
                raise RuntimeError(f"missing gradient for {name}")
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if not gradients_are_finite(gradients).item():
            invalid = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and not torch.isfinite(parameter.grad).all().item()
            ]
            raise FloatingPointError(f"non-finite Mini comparison gradients: {invalid}")
        progress = step / config.iterations
        eta = 1.0 if progress < 0.3 else (1 - progress) / 0.7
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * eta
            optimizer.step()
        torch.cuda.synchronize(device)
        training_ms += (time.perf_counter() - started) * 1000
        step += 1
        cursor += config.batch_characters
        if step % config.log_every == 0 or step == config.iterations:
            extras = " ".join(
                f"train_{name}:{value:.8g}"
                for name, value in metrics.items()
                if name not in {"loss", "objective"} and not name.endswith("_nats")
            )
            print(
                f"step:{step}/{config.iterations} train_loss:{metrics['objective']:.8g}"
                f" train_time:{training_ms:.3f}ms step_avg:{training_ms / step:.3f}ms"
                f" {extras}",
                flush=True,
            )
    print(f"saved checkpoint: {destination}", flush=True)
    return destination


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise ValueError(
            "configure with environment variables or scripts/native_bits.py train"
        )
    train()

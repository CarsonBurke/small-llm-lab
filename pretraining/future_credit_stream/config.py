"""Streaming FFN recurrence with a future-bag carry over document streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE = "streaming_ffn_future_bag_carry_v1"

OBJECTIVES = ("ce", "future_bag", "tbptt")
"""``ce``: carry the hidden verbatim (the recurrent-CE control).
``future_bag``: gated carry writer trained so the carry's readout through the
frozen head is the discounted bag of the next ``horizon`` future tokens.
``tbptt``: CE recursion with true temporal gradients inside each optimizer
page; an upper reference, never a promotable recipe."""


@dataclass(frozen=True)
class Config:
    data_path: str = "data/datasets/fineweb_onepass_sp1024"
    tokenizer_path: str = "data/tokenizers/fineweb_1024_bpe.model"
    vocab_size: int = 1024
    model_dim: int = 512
    num_layers: int = 6
    mlp_hidden: int = 2048
    document_batch: int = 4096
    stream_steps: int = 8
    objective: str = "future_bag"
    horizon: int = 32
    discount: float = 0.9
    backbone_future_weight: float = 0.0
    iterations: int = 2000
    val_every: int = 20
    log_every: int = 10
    val_documents: int = 256
    val_stream_steps: int = 128
    checkpoint_every: int = 100
    seed: int = 1337
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    compile_mode: str = "reduce-overhead"
    cpu_threads: int = 8
    run_id: str = "ffn_bag_carry_sp1024_2k"

    @classmethod
    def from_env(cls) -> "Config":
        shared = {"iterations": "ITERATIONS", "val_every": "VAL_LOSS_EVERY",
                  "log_every": "TRAIN_LOG_EVERY", "run_id": "RUN_ID", "seed": "SEED"}
        values = {}
        for key, default in asdict(cls()).items():
            raw = os.environ.get("FUTURE_CREDIT_STREAM_" + key.upper())
            if raw is None and key in shared:
                raw = os.environ.get(shared[key])
            values[key] = type(default)(raw) if raw is not None else default
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        for key in ("vocab_size", "model_dim", "num_layers", "mlp_hidden", "document_batch",
                    "stream_steps", "horizon", "iterations", "val_every", "log_every",
                    "val_documents", "val_stream_steps", "checkpoint_every", "cpu_threads"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if not math.isfinite(self.discount) or not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be finite and in [0, 1]")
        weight = self.backbone_future_weight
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("backbone_future_weight must be finite and in [0, 1]")
        if weight > 0.0 and self.objective != "future_bag":
            raise ValueError("backbone_future_weight applies only to the future_bag objective")
        if not 0 <= self.warmup_steps < self.iterations:
            raise ValueError("warmup_steps must be in [0, iterations)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")
        if self.compile_mode not in ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
            raise ValueError("unsupported torch.compile mode; eager fallback is not supported")
        if Path(self.run_id).name != self.run_id or self.run_id in (".", ".."):
            raise ValueError("run_id must be a single directory name")

    @property
    def uses_writer(self) -> bool:
        return self.objective == "future_bag"

    @property
    def page_horizon(self) -> int:
        """Future tokens each training page must provide per position."""
        return self.horizon if self.uses_writer else 0

    @property
    def model_config(self) -> dict:
        values = {key: getattr(self, key) for key in
                  ("vocab_size", "model_dim", "num_layers", "mlp_hidden")}
        return {**values, "use_writer": self.uses_writer}

    @property
    def tokens_per_step(self) -> int:
        return self.document_batch * self.stream_steps

    @property
    def output_dir(self) -> Path:
        return REPO_ROOT / "ablation_results" / self.run_id

    def path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path

    def schedule_at(self, step: int) -> float:
        """Warmup then cosine to 10%, as a multiplier on the base rate."""
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.iterations - self.warmup_steps - 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    def learning_rate_at(self, step: int) -> float:
        return self.learning_rate * self.schedule_at(step)

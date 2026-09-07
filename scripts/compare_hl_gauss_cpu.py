"""CPU-only comparison of latent VAPO and cleanrl V-MPO v30 value supports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, cast

import torch
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPOSITORIES / "cleanrl"))

from cleanrl.shared.hl_gauss import (  # pyright: ignore[reportMissingImports]
    Dreamer3BucketHLGaussSupport,
)
from postraining.hl_gauss import HLGaussSupport, anchored_unit_geometry


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_supports() -> tuple[HLGaussSupport, Dreamer3BucketHLGaussSupport]:
    num_bins, v_min, v_max = anchored_unit_geometry(101, 4)
    local = HLGaussSupport(num_bins, v_min, v_max, sigma_ratio=1.0)
    symlog_limit = math.log1p(20_000.0)
    v30 = Dreamer3BucketHLGaussSupport(
        51,
        -symlog_limit,
        symlog_limit,
        0.75,
        torch.device("cpu"),
    )
    return local, v30


def project(
    local: HLGaussSupport,
    v30: Dreamer3BucketHLGaussSupport,
    kind: str,
    targets: Tensor,
) -> Tensor:
    if kind == "local":
        return local.project(targets)
    if kind == "v30":
        return v30.project_moment_matched(targets)
    raise ValueError(f"unknown support: {kind}")


def decode(
    local: HLGaussSupport,
    v30: Dreamer3BucketHLGaussSupport,
    kind: str,
    logits: Tensor,
) -> Tensor:
    if kind == "local":
        return local.to_expected_scalar(logits)
    if kind == "v30":
        return v30.to_scalar(logits)
    raise ValueError(f"unknown support: {kind}")


def projection_audit(
    local: HLGaussSupport, v30: Dreamer3BucketHLGaussSupport
) -> dict[str, Any]:
    targets = torch.linspace(0.0, 1.0, 1_001)
    results: dict[str, Any] = {}
    for kind in ("local", "v30"):
        probabilities = project(local, v30, kind, targets)
        if kind == "local":
            centers = cast(Tensor, local.centers)
            decoded = (probabilities * centers).sum(-1)
        else:
            decoded = v30.probs_to_scalar(probabilities)
            centers = cast(Tensor, v30.support)
        errors = (decoded - targets).abs()
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
        results[kind] = {
            "bins": int(probabilities.shape[-1]),
            "support_min": float(centers[0]),
            "support_max": float(centers[-1]),
            "nearest_positive_center": float(centers[centers > 0][0]),
            "mean_absolute_projection_bias": float(errors.mean()),
            "max_absolute_projection_bias": float(errors.max()),
            "mean_target_entropy": float(entropy.mean()),
            "mean_bins_above_1e-6": float(
                (probabilities > 1e-6).float().sum(-1).mean()
            ),
        }
    return results


def projection_speed(
    local: HLGaussSupport, v30: Dreamer3BucketHLGaussSupport
) -> dict[str, float]:
    targets = torch.rand(4_096, generator=torch.Generator().manual_seed(91))
    for _ in range(5):
        local.project(targets)
        v30.project_moment_matched(targets)

    def seconds_per_call(function, repetitions: int) -> float:
        started = time.perf_counter()
        for _ in range(repetitions):
            function()
        return (time.perf_counter() - started) / repetitions

    local_seconds = seconds_per_call(lambda: local.project(targets), 100)
    v30_seconds = seconds_per_call(
        lambda: v30.project_moment_matched(targets), 20
    )
    return {
        "local_seconds_per_4096_targets": local_seconds,
        "v30_seconds_per_4096_targets": v30_seconds,
        "v30_over_local_ratio": v30_seconds / local_seconds,
    }


def initial_logits(
    local: HLGaussSupport, kind: str, rows: int
) -> tuple[nn.Linear, Tensor]:
    if kind == "local":
        head = nn.Linear(64, local.num_bins)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(
                local.project_to_logprobs(torch.tensor(0.0), eps=1e-6)
            )
        return head, head(torch.zeros(rows, 64))
    head = nn.Linear(64, 51, bias=False)
    with torch.no_grad():
        head.weight.zero_()
    return head, head(torch.zeros(rows, 64))


def decoder_sensitivity(
    local: HLGaussSupport, v30: Dreamer3BucketHLGaussSupport
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for kind in ("local", "v30"):
        _, base = initial_logits(local, kind, 4_096)
        probe = base[0].detach().clone().requires_grad_()
        decode(local, v30, kind, probe).backward()
        if probe.grad is None:
            raise RuntimeError("decoder sensitivity gradient is missing")
        row: dict[str, Any] = {
            "decode_jacobian_l2": float(probe.grad.norm()),
            "decode_jacobian_abs_max": float(probe.grad.abs().max()),
        }
        generator = torch.Generator().manual_seed(12_345)
        for scale in (1e-4, 1e-3, 1e-2):
            noise = torch.randn(base.shape, generator=generator) * scale
            values = decode(local, v30, kind, base + noise).detach()
            row[f"logit_noise_{scale:g}_decoded_rms"] = float(
                values.square().mean().sqrt()
            )
            row[f"logit_noise_{scale:g}_decoded_abs_max"] = float(
                values.abs().max()
            )
        results[kind] = row
    return results


def make_regression_data(seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    count = 6_144
    inputs = torch.randn(count, 16, generator=generator)
    latent = (
        1.4 * inputs[:, 0]
        - inputs[:, 1]
        + 0.7 * inputs[:, 2] * inputs[:, 3]
        - 0.5 * inputs[:, 4].square()
        + 0.3 * torch.sin(2.0 * inputs[:, 5])
    )
    threshold = torch.quantile(latent[:4_096], 0.92)
    positive = latent > threshold
    magnitude = torch.sigmoid(0.8 * (latent - threshold) + 0.2)
    targets = torch.where(positive, magnitude, torch.zeros_like(latent))
    feature_weight = torch.randn(16, 64, generator=generator) / 4.0
    feature_bias = torch.randn(64, generator=generator) * 0.1
    features = torch.relu(inputs @ feature_weight + feature_bias)
    return (
        features[:4_096],
        targets[:4_096],
        features[4_096:],
        targets[4_096:],
    )


def train_trial(
    local: HLGaussSupport,
    v30: Dreamer3BucketHLGaussSupport,
    kind: str,
    seed: int,
    learning_rate: float,
    steps: int,
    batch_size: int,
) -> dict[str, float]:
    (
        train_features,
        train_targets,
        validation_features,
        validation_targets,
    ) = make_regression_data(seed)
    target_probabilities = project(local, v30, kind, train_targets)
    head, _ = initial_logits(local, kind, 1)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=0.0
    )
    generator = torch.Generator().manual_seed(seed + 1_000)

    with torch.no_grad():
        initial_predictions = decode(local, v30, kind, head(validation_features))
        initial_mse = (initial_predictions - validation_targets).square().mean()

    first_indices = torch.randint(
        0, train_features.shape[0], (batch_size,), generator=generator
    )
    first_logits = head(train_features[first_indices])
    first_loss = -(
        target_probabilities[first_indices] * first_logits.log_softmax(-1)
    ).sum(-1).mean()
    first_loss.backward()
    gradient_squares = [
        parameter.grad.double().square().sum()
        for parameter in head.parameters()
        if parameter.grad is not None
    ]
    initial_gradient_norm = torch.stack(gradient_squares).sum().sqrt()
    optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()
    for _ in range(steps):
        indices = torch.randint(
            0, train_features.shape[0], (batch_size,), generator=generator
        )
        logits = head(train_features[indices])
        loss = -(
            target_probabilities[indices] * logits.log_softmax(-1)
        ).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started

    with torch.no_grad():
        predictions = decode(local, v30, kind, head(validation_features))
        errors = predictions - validation_targets
        positive = validation_targets > 0
        return {
            "initial_mse": float(initial_mse),
            "initial_gradient_norm": float(initial_gradient_norm),
            "mse": float(errors.square().mean()),
            "mae": float(errors.abs().mean()),
            "positive_mae": float(errors[positive].abs().mean()),
            "zero_target_abs_prediction": float(
                predictions[~positive].abs().mean()
            ),
            "prediction_min": float(predictions.min()),
            "prediction_max": float(predictions.max()),
            "train_seconds": elapsed,
        }


def aggregate(
    trials: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(
                torch.tensor(
                    [trial[key] for trial in trials], dtype=torch.float64
                ).mean()
            ),
            "std": float(
                torch.tensor(
                    [trial[key] for trial in trials], dtype=torch.float64
                ).std(correction=0)
            ),
        }
        for key in trials[0]
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=(1e-3, 3e-3, 1e-2),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ablation_results/hl_gauss_cpu_comparison.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seeds < 1 or args.steps < 1 or args.batch_size < 1:
        raise ValueError("seeds, steps, and batch size must be positive")
    if any(
        rate <= 0 or not math.isfinite(rate)
        for rate in args.learning_rates
    ):
        raise ValueError("learning rates must be finite and positive")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.use_deterministic_algorithms(True)
    local, v30 = build_supports()

    learning: dict[str, Any] = {}
    for learning_rate in args.learning_rates:
        rate_results: dict[str, Any] = {}
        for kind in ("local", "v30"):
            trials = [
                train_trial(
                    local,
                    v30,
                    kind,
                    seed,
                    learning_rate,
                    args.steps,
                    args.batch_size,
                )
                for seed in range(args.seeds)
            ]
            rate_results[kind] = {
                "aggregate": aggregate(trials),
                "trials": trials,
            }
        learning[f"{learning_rate:g}"] = rate_results

    document = {
        "schema": "hl_gauss_cpu_comparison/v1",
        "device": "cpu",
        "model_workload": False,
        "scope": "bounded [0,1] sparse value regression",
        "sources": {
            "local": {
                "path": "postraining/hl_gauss.py",
                "sha256": source_sha256(
                    PROJECT_ROOT / "postraining/hl_gauss.py"
                ),
            },
            "v30": {
                "path": (
                    "../cleanrl/cleanrl/vmpo/"
                    "ppo_continuous_action_iterthink_v24_beta_vmpo_"
                    "v30_dreamer_bucket_moment_hlgauss_reward_norm.py"
                ),
                "sha256": source_sha256(
                    REPOSITORIES
                    / "cleanrl/cleanrl/vmpo/"
                    "ppo_continuous_action_iterthink_v24_beta_vmpo_"
                    "v30_dreamer_bucket_moment_hlgauss_reward_norm.py"
                ),
            },
            "v30_support": {
                "path": "../cleanrl/cleanrl/shared/hl_gauss.py",
                "sha256": source_sha256(
                    REPOSITORIES / "cleanrl/cleanrl/shared/hl_gauss.py"
                ),
            },
        },
        "settings": {
            "seeds": args.seeds,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rates": args.learning_rates,
            "train_examples": 4_096,
            "validation_examples": 2_048,
            "positive_fraction": 0.08,
        },
        "projection": projection_audit(local, v30),
        "projection_speed": projection_speed(local, v30),
        "decoder_sensitivity": decoder_sensitivity(local, v30),
        "learning": learning,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

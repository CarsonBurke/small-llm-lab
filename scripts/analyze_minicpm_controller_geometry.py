#!/usr/bin/env python3
"""Read-only geometry of one real, pre-update frozen-controller rollout batch.

Run only through mlq, e.g.:
  mlq submit --name controller_geometry --max-parallel-runs 1 --cwd REPO -- \
    .venv/bin/python scripts/analyze_minicpm_controller_geometry.py \
    --queued-run --input analysis.pt --output geometry.json

No actor, critic, optimizer, rollout, random action, or reward is created here.
Only the existing FP32 controller heads execute, on CUDA, on saved BF16 states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


SCHEMA = "minicpm-controller-analysis/v1"
SAVED_POINT = "after refresh_critic, before controller_update"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def distribution(torch, values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not values.numel():
        return {"count": 0}
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite diagnostic statistic")
    return {
        "count": values.numel(), "mean": values.mean().item(),
        "rms": values.square().mean().sqrt().item(),
        "min": values.min().item(), "max": values.max().item(),
        "quantiles": dict(zip(("p01", "p10", "p50", "p90", "p99"),
            torch.quantile(values, torch.tensor([.01, .1, .5, .9, .99],
                dtype=torch.float64, device=values.device)).tolist())),
    }


def validate_batch(torch, batch, captured_trainer_source=None, captured_rollout_source=None):
    if batch.get("schema") != SCHEMA or batch.get("saved_point") != SAVED_POINT:
        raise ValueError("requires the exact v1 pre-controller-update analysis batch")
    dimension = batch["model_dim"]
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("invalid model_dim")
    sigma = batch["component_std"]
    if not math.isfinite(sigma) or sigma <= 0 or not math.isclose(
            sigma * math.sqrt(dimension), batch["vector_sigma"], rel_tol=1e-12):
        raise ValueError("inconsistent fixed Gaussian scale")
    if not isinstance(batch["seed"], int) or not batch["records"]:
        raise ValueError("a seeded, nonempty actual rollout batch is required")
    required_sources = {"scripts/train_minicpm_latent_controller.py",
                        "postraining/latent_thought.py", "postraining/vapo/policy.py"}
    if not required_sources.issubset(batch["source_hashes"]):
        raise ValueError("missing scoring/head source hashes")
    current_hashes = {}
    overrides = {
        "scripts/train_minicpm_latent_controller.py": captured_trainer_source,
        "postraining/minicpm_latent_rollout.py": captured_rollout_source,
    }
    for name, source in overrides.items():
        if source is not None and name not in batch["source_hashes"]:
            raise ValueError(f"capture has no source hash to verify override: {name}")
    for name, expected in batch["source_hashes"].items():
        source = (overrides[name] if overrides.get(name) else ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file():
            raise ValueError(f"invalid source provenance path: {name}")
        current_hashes[name] = digest(source)
        if current_hashes[name] != expected:
            raise ValueError(f"source changed since capture: {name}")
    for record in batch["records"]:
        states, raw, kinds = (record[name] for name in ("observations", "raw", "kinds"))
        n = states.shape[0]
        for name, dtype in (("observations", torch.bfloat16), ("raw", torch.float32),
                            ("kinds", torch.int8), ("advantages", torch.float32),
                            ("old_logprobs", torch.float32)):
            tensor = record[name]
            if tensor.device.type != "cpu" or tensor.dtype != dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"invalid saved {name}: CPU exact dtype and finite values required")
        if states.shape != (n, dimension) or raw.ndim != 2 or raw.shape[1] != dimension:
            raise ValueError("invalid state/action shape")
        if n < 1 or kinds.shape != (n,) or any(record[key].shape != (n,)
                for key in ("advantages", "old_logprobs")):
            raise ValueError("misaligned actions, advantages or scores")
        m = raw.shape[0]
        if m < 1 or n not in (m, m + 1) or kinds[0].item() != 1:
            raise ValueError("mandatory FIRST and exact Gaussian-prefix alignment required")
        if not (kinds[1:m] == 2).all() or (n > m and kinds[-1].item() != 3):
            raise ValueError("only FIRST, CONTINUE and optional genuine STOP are allowed")
        if not isinstance(record["correct"], bool) or not isinstance(record["group_index"], int):
            raise ValueError("correct and prompt group provenance required")
        if record["group_index"] < 0:
            raise ValueError("negative prompt group")
    for key, shapes in (("transition_state", {"mean_head.weight": (dimension, dimension),
                                               "mean_head.bias": (dimension,)}),
                        ("gate_state", {"head.weight": (1, dimension), "head.bias": (1,)})):
        if set(batch[key]) != set(shapes):
            raise ValueError(f"unexpected state_dict keys in {key}")
        for name, shape in shapes.items():
            value = batch[key][name]
            if value.device.type != "cpu" or value.dtype != torch.float32 or value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid FP32 saved parameter: {key}/{name}")
    return current_hashes


def chunks(torch, record, size):
    for start in range(0, record["observations"].shape[0], size):
        stop = min(start + size, record["observations"].shape[0])
        yield {key: record[key][start:stop].to("cuda") for key in
               ("observations", "raw", "kinds", "advantages", "old_logprobs")}


def paired_variance_sums(left_sum, right_sum, individual_squares, pair_squares, pairs):
    """Unbiased conditional variance traces for sums of independent pairs."""
    if pairs < 2:
        raise ValueError("at least two independent pairs are required")
    marginal_centered = individual_squares - (
        left_sum.square().sum().item() + right_sum.square().sum().item()) / pairs
    paired_centered = pair_squares - (left_sum + right_sum).square().sum().item() / pairs
    multiplier = pairs / (pairs - 1)
    return {
        "pairs": pairs,
        "paired_sum_variance_trace": multiplier * paired_centered,
        "independent_marginal_sum_variance_trace": multiplier * marginal_centered,
        "within_pair_cross_covariance_trace": (
            paired_centered - marginal_centered) / (2 * (pairs - 1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queued-run", action="store_true", required=True,
                        help="Attest this foreground process is supervised by mlq; not a bypass.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--captured-trainer-source", type=Path,
                        help="Immutable captured trainer source; its SHA must equal the recorded original.")
    parser.add_argument("--captured-rollout-source", type=Path,
                        help="Immutable, nonexecuted rollout source; its SHA must equal the recorded original.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--fisher-rcond", type=float, default=1e-5,
                        help="Keep covariance eigenmodes lambda > rcond * lambda_max; report discarded modes.")
    parser.add_argument("--kl-budget", type=float, default=0.02)
    args = parser.parse_args()
    if args.batch_size <= 0 or not 0 < args.fisher_rcond < 1 or not 0 < args.kl_budget <= .02:
        parser.error("positive batch size, 0 < rcond < 1 and 0 < KL budget <= 0.02 required")
    if args.output.exists() or args.output.resolve() == args.input.resolve():
        parser.error("output must be a new path, distinct from the immutable input")

    import torch
    import torch.nn.functional as F
    from types import SimpleNamespace

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("queued CUDA with native BF16 is required; no CPU/model fallback")
    # Head likelihoods and matrix products use the accepted FP32/IEEE path, not TF32.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(ROOT))
    if args.captured_trainer_source:
        import importlib.util
        specification = importlib.util.spec_from_file_location("captured_controller", args.captured_trainer_source)
        controller_module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(controller_module)
        WhitenedGaussianController = controller_module.WhitenedGaussianController
        NormalizedStopGate = controller_module.NormalizedStopGate
        controller_scores = controller_module.controller_scores
    else:
        from scripts.train_minicpm_latent_controller import (
            WhitenedGaussianController, NormalizedStopGate, controller_scores,
        )

    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        batch = torch.load(args.input, map_location="cpu", weights_only=True)
    current_hashes = validate_batch(torch, batch, args.captured_trainer_source, args.captured_rollout_source)
    records = batch["records"]
    ntraj, dimension, sigma = len(records), batch["model_dim"], batch["component_std"]
    width = dimension + 1
    # Meta construction does not initialize a real random controller or consume CUDA RNG.
    with torch.device("meta"):
        transition = WhitenedGaussianController(dimension, batch["vector_sigma"])
        gate = NormalizedStopGate(dimension, .5)
    transition.load_state_dict({k: v.to("cuda") for k, v in batch["transition_state"].items()}, assign=True)
    gate.load_state_dict({k: v.to("cuda") for k, v in batch["gate_state"].items()}, assign=True)
    policy = SimpleNamespace(transition=transition, thinking_gate=gate)
    parameters = tuple(transition.parameters()) + tuple(gate.parameters())
    covariance_sum = torch.zeros((width, width), device="cuda", dtype=torch.float32)
    gradient = torch.zeros((dimension, width), device="cuda", dtype=torch.float32)
    gate_gradient = torch.zeros(width, device="cuda", dtype=torch.float32)
    ideal_gradient = torch.zeros_like(gradient)
    analytic_gradient = torch.zeros_like(gradient)
    analytic_gate_gradient = torch.zeros_like(gate_gradient)
    component_sum = torch.zeros(dimension, device="cuda")
    component_sq_sum = torch.zeros_like(component_sum)
    norm_parts, score_parts = [], []
    by_kind_scores = {1: [], 2: [], 3: []}
    trajectory_report = []
    trajectory_grad_sq = trajectory_gate_sq = 0.0
    group_grad_sq = group_gate_sq = 0.0
    weighted_group_sum = torch.zeros_like(gradient)
    weighted_group_gate_sum = torch.zeros_like(gate_gradient)
    group_sizes, group_report = [], []
    gaussian_count = state_count = 0
    objective = gaussian_objective = gate_objective = 0.0

    # Retain one covariance and Gaussian/gate gradient per prompt for honest cross-fitting.
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(record["group_index"], []).append((index, record))
    paired_run = batch.get("coupling") is not None
    paired_group_statistics = []
    if paired_run:
        if ntraj != 64 or len(groups) != 4 or any(len(members) != 16 for members in groups.values()):
            raise ValueError("paired diagnostic requires the captured four groups of sixteen")
        for index, record in enumerate(records):
            if record.get("logical_lane") != index or record.get("pair_id") != index // 2:
                raise ValueError("paired diagnostic lane provenance is inconsistent")
            if index % 2 and record["group_index"] != records[index - 1]["group_index"]:
                raise ValueError("a coupled pair crosses prompt groups")
    group_statistics = {}
    for group_index, members in sorted(groups.items()):
        group_gradient = torch.zeros_like(gradient)
        group_gate = torch.zeros_like(gate_gradient)
        group_covariance = torch.zeros_like(covariance_sum)
        group_gaussians = 0
        group_noise_sq = group_noise_sum = 0.0
        group_components = 0
        pair_moments = {
            name: {"left_sum": torch.zeros_like(prototype, dtype=torch.float64),
                   "right_sum": torch.zeros_like(prototype, dtype=torch.float64),
                   "individual_squares": 0.0, "pair_squares": 0.0}
            for name, prototype in (("mean_head", gradient), ("gate", gate_gradient))
        } if paired_run else {}
        pending_pair = None
        for index, record in members:
            trajectory_gradient = torch.zeros_like(gradient)
            trajectory_gate = torch.zeros_like(gate_gradient)
            trajectory_norms, trajectory_errors = [], []
            trajectory_component_sum = torch.zeros_like(component_sum)
            trajectory_component_sq = torch.zeros_like(component_sum)
            trajectory_gaussians = 0
            trajectory_objective = 0.0
            for part in chunks(torch, record, args.batch_size):
                states, raw, kinds, advantages, old = (part[key] for key in
                    ("observations", "raw", "kinds", "advantages", "old_logprobs"))
                count = raw.shape[0]
                scores = controller_scores(policy, states, kinds, raw)
                weighted_score = (scores * advantages.detach()).sum()
                grads = torch.autograd.grad(weighted_score, parameters, allow_unused=True)
                actual = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, grads)]
                trajectory_gradient[:, :-1].add_(actual[0])
                trajectory_gradient[:, -1].add_(actual[1])
                trajectory_gate[:-1].add_(actual[2].squeeze(0))
                trajectory_gate[-1].add_(actual[3].squeeze(0))
                with torch.no_grad():
                    features = F.rms_norm(states.float(), (dimension,))
                    x = torch.cat((features, torch.ones((states.shape[0], 1), device="cuda")), dim=1)
                    mean = transition.predict_mean(states[:count])
                    epsilon = (raw - mean) / sigma
                    log_sigma = transition.predict_log_sigma(states[:count])
                    # The exact implemented derivative includes FP32 exp/log rounding.
                    score_mean_derivative = (raw - mean) * (-log_sigma).exp().square() * sigma
                    analytic_gradient.add_(score_mean_derivative.T @ (advantages[:count, None] * x[:count]))
                    ideal_gradient.add_(epsilon.T @ (advantages[:count, None] * x[:count]))
                    logits = gate.stop_logit(states)
                    gate_factor = ((kinds == 3).float() - logits.sigmoid()) * (kinds != 1) * advantages
                    analytic_gate_gradient.add_((gate_factor[:, None] * x).sum(0))
                    feature_outer = x[:count].T @ x[:count]
                    covariance_sum.add_(feature_outer)
                    group_covariance.add_(feature_outer)
                    group_gaussians += count
                    component_sum.add_(epsilon.sum(0))
                    component_sq_sum.add_(epsilon.square().sum(0))
                    trajectory_component_sum.add_(epsilon.sum(0))
                    trajectory_component_sq.add_(epsilon.square().sum(0))
                    norms = epsilon.square().sum(-1).sqrt().cpu()
                    error = (scores.detach() - old).cpu()
                    norm_parts.append(norms)
                    score_parts.append(error)
                    trajectory_norms.append(norms)
                    trajectory_errors.append(error)
                    for kind in by_kind_scores:
                        by_kind_scores[kind].append(error[kinds.cpu() == kind])
                    gp = transition.log_prob(raw, mean, log_sigma)
                    gaussian_objective += (gp * advantages[:count]).sum().item() / ntraj
                    gate_objective += ((scores[:count].detach() - gp) * advantages[:count]).sum().item() / ntraj
                    gate_objective += (scores[count:].detach() * advantages[count:]).sum().item() / ntraj
                    trajectory_objective += weighted_score.detach().item()
                    trajectory_gaussians += count
                    gaussian_count += count
                    state_count += states.shape[0]
            gradient.add_(trajectory_gradient)
            gate_gradient.add_(trajectory_gate)
            group_gradient.add_(trajectory_gradient)
            group_gate.add_(trajectory_gate)
            gnorm = trajectory_gradient.square().sum(dtype=torch.float64).item()
            bnorm = trajectory_gate.square().sum(dtype=torch.float64).item()
            trajectory_grad_sq += gnorm
            trajectory_gate_sq += bnorm
            if paired_run:
                side = "left_sum" if index % 2 == 0 else "right_sum"
                for position, (name, value, square) in enumerate((
                        ("mean_head", trajectory_gradient, gnorm),
                        ("gate", trajectory_gate, bnorm))):
                    moment = pair_moments[name]
                    moment[side].add_(value)
                    moment["individual_squares"] += square
                    if index % 2:
                        moment["pair_squares"] += (pending_pair[position] + value).square().sum(
                            dtype=torch.float64).item()
                pending_pair = (trajectory_gradient, trajectory_gate) if index % 2 == 0 else None
            objective += trajectory_objective / ntraj
            group_noise_sq += trajectory_component_sq.sum().item()
            group_noise_sum += trajectory_component_sum.sum().item()
            group_components += trajectory_gaussians * dimension
            trajectory_report.append({
                "trajectory_index": index, "group_index": group_index, "correct": record["correct"],
                "gaussian_actions": trajectory_gaussians, "gate_actions": record["kinds"].numel() - 1,
                "genuine_stop": record["kinds"][-1].item() == 3,
                "summed_advantage_weighted_score": trajectory_objective,
                "mean_head_reward_gradient_norm": math.sqrt(gnorm),
                "gate_reward_gradient_norm": math.sqrt(bnorm),
                "epsilon_component_mean": trajectory_component_sum.sum().item() / (trajectory_gaussians * dimension),
                "epsilon_component_second_moment": trajectory_component_sq.sum().item() / (trajectory_gaussians * dimension),
                "epsilon_per_dimension_rms": distribution(torch, (trajectory_component_sq / trajectory_gaussians).sqrt().cpu()),
                "epsilon_norm": distribution(torch, torch.cat(trajectory_norms)),
                "old_score_error": distribution(torch, torch.cat(trajectory_errors)),
                "old_score_max_abs_error": torch.cat(trajectory_errors).abs().max().item(),
            })
        size = len(members)
        group_sizes.append(size)
        group_grad_sq += group_gradient.square().sum().item()
        group_gate_sq += group_gate.square().sum().item()
        weighted_group_sum.add_(group_gradient, alpha=size)
        weighted_group_gate_sum.add_(group_gate, alpha=size)
        group_statistics[group_index] = {
            "covariance_sum": group_covariance, "gradient_sum": group_gradient,
            "gate_gradient_sum": group_gate, "gaussian_actions": group_gaussians,
            "trajectories": size,
        }
        group_report.append({"group_index": group_index, "trajectories": size,
            "correct": sum(r["correct"] for _, r in members),
            "epsilon_component_mean": group_noise_sum / group_components,
            "epsilon_component_second_moment": group_noise_sq / group_components,
            "mean_head_summed_gradient_norm": group_gradient.norm().item(),
            "gate_summed_gradient_norm": group_gate.norm().item()})
        if paired_run:
            paired_group_statistics.append({
                "group_index": group_index,
                **{name: paired_variance_sums(**moment, pairs=size // 2)
                   for name, moment in pair_moments.items()},
            })

    gradient.div_(ntraj)
    gate_gradient.div_(ntraj)
    analytic_gradient.div_(ntraj)
    ideal_gradient.div_(ntraj)
    analytic_gate_gradient.div_(ntraj)
    covariance = covariance_sum / gaussian_count
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    cutoff = args.fisher_rcond * eigenvalues[-1].item()
    keep = eigenvalues > cutoff
    if not keep.any():
        raise RuntimeError("no resolved positive Fisher feature eigenmodes")
    retained = eigenvalues[keep]
    basis = eigenvectors[:, keep]
    projected_gradient = gradient @ basis
    fisher_direction = (projected_gradient / retained) @ basis.T
    discarded_gradient = gradient - projected_gradient @ basis.T

    def signal_summary(mean, individual_squares, cluster_squares, weighted_sum):
        signal = mean.square().sum().item()
        trajectory_centered_raw = individual_squares - ntraj * signal
        cluster_centered_raw = (cluster_squares - 2 * (weighted_sum * mean).sum().item()
                                + sum(size * size for size in group_sizes) * signal)
        # Do not clip cancellation or the bias-corrected signal: negative estimates matter.
        trajectory_noise = trajectory_centered_raw / (ntraj * (ntraj - 1)) if ntraj > 1 else None
        cluster_noise = (len(groups) / (len(groups) - 1) * cluster_centered_raw / ntraj**2
                         if len(groups) > 1 else None)
        corrected_signal = signal - cluster_noise if cluster_noise is not None else None
        return {"squared_mean_gradient_norm": signal,
                "trajectory_centered_sum_squares_raw": trajectory_centered_raw,
                "prompt_cluster_centered_sum_squares_raw": cluster_centered_raw,
                "iid_trajectory_variance_trace_of_mean": trajectory_noise,
                "prompt_cluster_robust_variance_trace_of_mean": cluster_noise,
                "signal_to_iid_trajectory_noise": signal / trajectory_noise if trajectory_noise else None,
                "signal_to_prompt_cluster_noise": signal / cluster_noise if cluster_noise else None,
                "legacy_ratio_interpretation": "Observed squared gradient / estimated noise; numerator includes noise, NOT corrected SNR.",
                "noise_bias_corrected_squared_signal": corrected_signal,
                "noise_bias_corrected_signal_to_prompt_cluster_noise": corrected_signal / cluster_noise if cluster_noise else None,
                "correction_formula": "estimated_squared_signal=||G||_F^2-V_cluster; corrected_SNR=(||G||_F^2-V_cluster)/V_cluster; neither estimate is clamped",
                "uncertainty": {"prompt_groups": len(groups),
                    "level": "high" if len(groups) <= 4 else "cluster estimate only",
                    "interpretation": "Only four prompt groups in the intended capture; a negative corrected estimate is possible. This plug-in cluster correction is not a confidence interval.",
                    "assumption": "Independent prompt clusters; exact unbiased correction for equal-size iid groups, approximate cluster-robust correction for unequal group sizes."}}

    def actual_kl(direction, scale, detailed=False, selected_records=None):
        selected_records = records if selected_records is None else selected_records
        selected_state_count = sum(record["observations"].shape[0] for record in selected_records)
        maxima, means, sums, joint_sums = [], [], [], []
        first, continuing, stopping = [], [], []
        with torch.no_grad():
            weight = transition.mean_head.weight + scale * direction[:, :-1]
            bias = transition.mean_head.bias + scale * direction[:, -1]
            for record in selected_records:
                pieces, joint = [], []
                for start in range(0, record["observations"].shape[0], args.batch_size):
                    stop = start + args.batch_size
                    states = record["observations"][start:stop].to("cuda")
                    features = F.rms_norm(states.float(), (dimension,))
                    before = transition.predict_mean(states)
                    after = states.float() + sigma * F.linear(features, weight, bias)
                    # Matches controller_kl: FP32 means, FP64 analytic KL reduction.
                    kl = .5 * ((after.double() - before.double()) / sigma).square().sum(-1)
                    if detailed:
                        kinds = record["kinds"][start:stop].to("cuda")
                        probability = gate.stop_logit(states).double().sigmoid()
                        joint_kl = torch.where(kinds == 1, kl, (1 - probability) * kl)
                        joint.append(joint_kl.cpu())
                    pieces.append(kl.cpu())
                    if detailed:
                        for target, kind in ((first, 1), (continuing, 2), (stopping, 3)):
                            target.append(kl[kinds == kind].cpu())
                values = torch.cat(pieces)
                maxima.append(values.max().item())
                means.append(values.sum().item())
                if detailed:
                    sums.append(values[:record["raw"].shape[0]].sum().item())
                    joint_sums.append(torch.cat(joint).sum().item())
        result = {"max_gaussian_kl_all_controller_states": max(maxima),
                  "mean_gaussian_kl_all_controller_states": sum(means) / selected_state_count}
        if detailed:
            result.update({"per_trajectory_max_gaussian_kl": maxima,
                "per_trajectory_realized_gaussian_kl_sum": sums,
                "per_trajectory_frozen_gate_mixture_kl_sum": joint_sums,
                "realized_gaussian_kl_sum": distribution(torch, sums),
                "frozen_gate_mixture_kl_sum": distribution(torch, joint_sums),
                "kl_by_kind": {name: distribution(torch, torch.cat(parts)) for name, parts in
                    (("first", first), ("continue", continuing), ("stop_counterfactual", stopping))}})
        return result

    def global_kl_bound(direction, scale):
        # Frobenius dominates the operator norm; no power-iteration lower estimate.
        with torch.no_grad():
            after_weight = transition.mean_head.weight + scale * direction[:, :-1]
            after_bias = transition.mean_head.bias + scale * direction[:, -1]
            delta = torch.cat((
                after_weight.double() - transition.mean_head.weight.double(),
                (after_bias.double() - transition.mean_head.bias.double())[:, None]), dim=1)
            squared_frobenius = delta.square().sum().item()
        # Positive FP64 square/reduction error bound, including subtraction and final rounding.
        roundoff_allowance = (delta.numel() + 4) * torch.finfo(torch.float64).eps
        squared_frobenius_upper = math.nextafter(squared_frobenius / (1 - roundoff_allowance), math.inf)
        bound = math.nextafter(.5 * width * squared_frobenius_upper, math.inf)
        return {
            "method": "conservative Frobenius upper bound on squared spectral norm",
            "feature_squared_norm_bound": width,
            "actual_rounded_parameter_step_squared_frobenius_norm": squared_frobenius,
            "squared_frobenius_upper_bound_with_roundoff": squared_frobenius_upper,
            "fp64_relative_roundoff_allowance": roundoff_allowance,
            "global_mean_head_kl_upper_bound": bound,
            "formula": "For every normalized feature x with ||x||^2<=d+1: KL=0.5||delta_A x||^2 <=0.5(d+1)||delta_A||_2^2 <=0.5(d+1)||delta_A||_F^2.",
            "scope": "Analytic affine mean-head KL for the actual rounded FP32 parameter step, accumulated in FP64; conservative, not a numerical certificate for FP32 RMS-normalization/residual-addition rounding. Empirical KL separately uses actual rounded FP32 means.",
            "empirical_budget": args.kl_budget,
            "bound_over_empirical_budget": bound / args.kl_budget,
            "interpretation": "A bound above budget fails to certify off-state safety; it does not prove a violating state exists.",
        }

    def select_scale(direction, selected_records, maximum):
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("nonfinite or zero sampled-state direction displacement")
        initial_scale = math.sqrt(args.kl_budget / maximum)
        scale = initial_scale
        for attempt in range(12):
            measured = actual_kl(direction, scale, selected_records=selected_records)
            value = measured["max_gaussian_kl_all_controller_states"]
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError("nonfinite or unresolvable actual Gaussian KL")
            if args.kl_budget * (1 - 1e-4) <= value <= args.kl_budget:
                return scale, initial_scale, attempt, measured
            scale *= math.sqrt(args.kl_budget / value) * (1 - 2e-5)
        raise RuntimeError("could not match actual FP32 analytic KL within 1e-4 below budget")

    directions = {"euclidean": gradient, "fisher_pseudoinverse": fisher_direction}
    comparison = {}
    coverage_parts, null_parts, ideal_kl = [], [], {name: 0.0 for name in directions}
    projected_ascent = {name: [] for name in directions}
    with torch.no_grad():
        for record in records:
            ascent = {name: 0.0 for name in directions}
            for part in chunks(torch, record, args.batch_size):
                states, raw, advantages = part["observations"], part["raw"], part["advantages"]
                features = F.rms_norm(states.float(), (dimension,))
                x = torch.cat((features, torch.ones((states.shape[0], 1), device="cuda")), 1)
                xp = x @ basis
                coverage_parts.append((xp.square() / retained).sum(-1).cpu())
                null_parts.append((x - xp @ basis.T).square().sum(-1).cpu())
                count = raw.shape[0]
                residual = raw - transition.predict_mean(states[:count])
                derivative = residual * (-transition.predict_log_sigma(states[:count])).exp().square() * sigma
                for name, direction in directions.items():
                    changes = x @ direction.T
                    ideal_kl[name] = max(ideal_kl[name], (.5 * changes.square().sum(-1)).max().item())
                    ascent[name] += (advantages[:count] * (derivative * changes[:count]).sum(-1)).sum().item()
            for name in directions:
                projected_ascent[name].append(ascent[name])

    for name, direction in directions.items():
        maximum = ideal_kl[name]
        if not math.isfinite(maximum):
            raise ValueError("nonfinite proposed direction")
        if maximum == 0:
            comparison[name] = {"status": "zero sampled-state displacement; cannot match positive KL", "scale": 0.0}
            continue
        try:
            scale, initial_scale, attempt, _ = select_scale(direction, records, maximum)
        except (RuntimeError, ValueError) as error:
            comparison[name] = {"status": "failed", "error": str(error)}
            continue
        measured = actual_kl(direction, scale, detailed=True)
        projections = torch.tensor(projected_ascent[name], dtype=torch.float64) * scale
        mean_projection = projections.mean().item()
        cluster_residuals = [sum(projections[index].item() - mean_projection for index, _ in members)
                             for members in groups.values()]
        cluster_se = (math.sqrt(len(groups) / (len(groups) - 1) * sum(v*v for v in cluster_residuals)) / ntraj
                      if len(groups) > 1 else None)
        predicted = scale * (gradient * direction).sum().item()
        comparison[name] = {"status": "matched", "gate_frozen": True, "scale": scale,
            "ideal_quadratic_scale": initial_scale, "rounding_scale_refinements": attempt,
            "unscaled_parameter_direction_norm": direction.norm().item(),
            "parameter_step_norm": scale * direction.norm().item(),
            "predicted_first_order_objective_ascent": predicted,
            "trajectory_projection_mean": mean_projection,
            "projection_vs_gradient_dot_abs_error": abs(mean_projection - predicted),
            "per_trajectory_predicted_ascent": projections.tolist(),
            "prompt_cluster_predicted_ascent_standard_error": cluster_se,
            "predicted_ascent_over_cluster_se": mean_projection / cluster_se if cluster_se else None,
            "projection_uncertainty_interpretation": "In-batch fitted-direction dispersion only; neither the standard error nor its ratio is independent confidence evidence.",
            "global_mean_head_kl": global_kl_bound(direction, scale),
            "ascent_per_sqrt_max_kl": predicted / math.sqrt(measured["max_gaussian_kl_all_controller_states"]),
            **measured}

    eigen_cpu = eigenvalues.cpu()
    positive = eigen_cpu[eigen_cpu > 0]
    spectral_mass = positive / positive.sum()
    errors = torch.cat(score_parts)
    per_dimension_mean = (component_sum / gaussian_count).cpu()
    per_dimension_rms = (component_sq_sum / gaussian_count).sqrt().cpu()
    per_dimension_variance = (component_sq_sum / gaussian_count).cpu() - per_dimension_mean.square()
    report = {
        "schema": "minicpm-controller-geometry/v1", "input": str(args.input.resolve()),
        "input_sha256": digest(args.input), "analysis_source_sha256": digest(Path(__file__).resolve()),
        "capture_source_hashes": batch["source_hashes"], "verified_source_hashes": current_hashes,
        "capture_source_overrides": {
            name: {"path": str(path.resolve()), "executed": executed,
                   "expected_sha256": batch["source_hashes"][name], "verified_sha256": current_hashes[name]}
            for name, path, executed in (
                ("scripts/train_minicpm_latent_controller.py", args.captured_trainer_source, True),
                ("postraining/minicpm_latent_rollout.py", args.captured_rollout_source, False))
            if path is not None},
        "saved_point": batch["saved_point"], "seed": batch["seed"],
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(), "observations": "BF16 exact saved",
                    "heads_and_geometry": "FP32 IEEE; TF32 disabled", "kl_reduction": "FP64, as controller_kl"},
        "counts": {"trajectories": ntraj, "prompt_groups": len(groups), "group_sizes": group_sizes,
                   "gaussian_actions": gaussian_count, "controller_states": state_count,
                   "gate_actions": state_count - ntraj, "model_dim": dimension},
        "component_std": sigma, "vector_sigma": batch["vector_sigma"],
        "formulas": {
            "features": "x=[rms_norm_FP32(saved_BF16_h);1], A=[W b], mu(h)=h+s A x; s fixed",
            "epsilon": "epsilon=(raw-mu)/s; raw is the exact saved FP32 action, never redrawn",
            "score_gradient": "d log N(raw;mu,s^2 I)/d A = epsilon x^T; implemented FP32 version uses (raw-mu)*exp(-log_s)^2*s*x^T",
            "gate_gradient": "d log Bernoulli(y;sigmoid(v*x))/dv=(y-p)x, y=1 only for genuine STOP; FIRST has no gate term",
            "objective": "J=(1/N_trajectories) sum_i sum_actual_actions_t detached_advantage_it * (Gaussian_score_it + eligible_gate_score_it); ascent gradient, negative of trainer loss gradient; no action or dimension normalization",
            "fisher": "C=(1/M_Gaussian_actions) sum x x^T; F=I_output tensor_product C in row-major vec(A). Exact conditional-action expected Gaussian score Fisher under the empirical uniform Gaussian-state measure, NOT realized epsilon outer-product empirical Fisher and NOT reward covariance",
            "preconditioner": "C=Q diag(lambda) Q^T; retain lambda>rcond*lambda_max; C_plus=Q_ret diag(1/lambda_ret) Q_ret^T; D_F=G C_plus; D_E=G. Discarded modes receive zero motion, not ridge damping",
            "kl": "KL(N(mu_old,s^2I)||N(mu_new,s^2I))=0.5*sum((mu_new_FP32-mu_old_FP32)/s)^2; ideal affine KL=0.5*alpha^2*||D x||^2",
            "cluster_uncertainty": "S_k=sum_{i in group k} g_i; G=sum_i g_i/N; Var_trace(G)=K/(K-1)/N^2 * sum_k ||S_k-n_k G||^2. Trajectory iid comparator is sum_i||g_i-G||^2/[N(N-1)]. Clusters are assumed independent; actions never count as reward replicates",
            "coverage": "leverage=x^T C_plus x; discarded_feature_energy=||x-Q_ret Q_ret^T x||^2, evaluated on ALL controller states including STOP",
        },
        "objective": {"actual": objective, "gaussian": gaussian_objective, "gate": gate_objective},
        "score_reconstruction": {"error": distribution(torch, errors), "max_abs_error": errors.abs().max().item(),
            "trainer_0_05_threshold_satisfied": errors.abs().max().item() <= .05,
            "by_kind": {str(kind): distribution(torch, torch.cat(parts)) for kind, parts in by_kind_scores.items()}},
        "gradient_reconstruction": {
            "mean_head_autograd_norm": gradient.norm().item(), "gate_autograd_norm": gate_gradient.norm().item(),
            "implemented_analytic_mean_max_abs_error": (gradient - analytic_gradient).abs().max().item(),
            "ideal_epsilon_mean_max_abs_error": (gradient - ideal_gradient).abs().max().item(),
            "implemented_analytic_mean_relative_error": (gradient - analytic_gradient).norm().item() / max(gradient.norm().item(), 1e-30),
            "analytic_gate_max_abs_error": (gate_gradient - analytic_gate_gradient).abs().max().item(),
            "mean_head_signal": signal_summary(gradient, trajectory_grad_sq, group_grad_sq, weighted_group_sum),
            "gate_signal": signal_summary(gate_gradient, trajectory_gate_sq, group_gate_sq, weighted_group_gate_sum)},
        "paired_gradient_variance": {
            "coupling": batch["coupling"],
            "scope": "Conditional on these four fixed prompts; eight independent pairs per prompt.",
            "normalization": "Sum each prompt's pair-sum variance, then divide by N_trajectories squared.",
            "marginal_comparator": "Separate left/right sample variances; estimates independent sampling of the same marginals, NOT another measured policy.",
            "legacy_iid_warning": "Per-trajectory IID estimates elsewhere are invalid for coupled arms; use this paired estimate or independent prompt clusters.",
            "uncertainty": "Small-sample descriptive estimate; no confidence interval or heldout learning claim. Negative estimates are not clamped.",
            "groups": paired_group_statistics,
            "aggregate": {
                name: {
                    "paired_variance_trace_of_mean": sum(
                        group[name]["paired_sum_variance_trace"] for group in paired_group_statistics) / ntraj**2,
                    "independent_marginal_variance_trace_of_mean": sum(
                        group[name]["independent_marginal_sum_variance_trace"]
                        for group in paired_group_statistics) / ntraj**2,
                } for name in ("mean_head", "gate")
            },
        } if paired_run else None,
        "epsilon": {"component_mean": per_dimension_mean.tolist(), "component_rms": per_dimension_rms.tolist(),
            "component_centered_variance": per_dimension_variance.tolist(),
            "component_centered_variance_distribution": distribution(torch, per_dimension_variance),
            "component_mean_distribution": distribution(torch, per_dimension_mean),
            "component_rms_distribution": distribution(torch, per_dimension_rms),
            "pooled_component_mean": per_dimension_mean.mean().item(),
            "pooled_component_second_moment": per_dimension_rms.square().mean().item(),
            "pooled_component_variance": per_dimension_rms.square().mean().item() - per_dimension_mean.mean().item()**2,
            "norm": distribution(torch, torch.cat(norm_parts)),
            "squared_norm": distribution(torch, torch.cat(norm_parts).square()),
            "isotropic_reference": {"component_mean": 0, "component_second_moment": 1,
                "expected_squared_norm": dimension, "squared_norm_standard_deviation": math.sqrt(2*dimension)},
            "variance_shift_by_correctness": {str(correct): distribution(torch,
                [r["epsilon_component_second_moment"] for r in trajectory_report if r["correct"] == correct])
                for correct in (False, True)}},
        "fisher_geometry": {"state_weighting": "uniform over actual Gaussian action states, each weight 1/M",
            "feature_dimension_including_bias": width, "full_parameter_dimension": dimension*width,
            "full_fisher_materialized": False, "trace": eigen_cpu.sum().item(),
            "eigenvalues_ascending": eigen_cpu.tolist(), "eigenspectrum": distribution(torch, eigen_cpu),
            "rcond": args.fisher_rcond, "absolute_cutoff": cutoff,
            "retained_rank": keep.sum().item(), "discarded_modes": (~keep).sum().item(),
            "full_fisher_retained_rank": dimension * keep.sum().item(),
            "full_fisher_discarded_modes": dimension * (~keep).sum().item(),
            "negative_numerical_eigenvalues": (eigen_cpu < 0).sum().item(),
            "most_negative_eigenvalue": min(0.0, eigen_cpu.min().item()),
            "retained_condition_number": (retained[-1] / retained[0]).item(),
            "unregularized_condition_number": (eigen_cpu[-1] / eigen_cpu[0]).item() if eigen_cpu[0] > 0 else None,
            "unregularized_condition_resolved_at_cutoff": bool(keep.all().item()),
            "effective_rank_entropy_positive_spectrum": (-spectral_mass * spectral_mass.log()).sum().exp().item(),
            "participation_rank_positive_spectrum": positive.sum().square().item() / positive.square().sum().item(),
            "discarded_positive_trace_fraction": eigen_cpu[(~keep).cpu()].clamp_min(0).sum().item() / positive.sum().item(),
            "discarded_gradient_energy_fraction": discarded_gradient.square().sum().item() / max(gradient.square().sum().item(), 1e-30),
            "coverage_leverage_all_states": distribution(torch, torch.cat(coverage_parts)),
            "discarded_feature_energy_all_states": distribution(torch, torch.cat(null_parts))},
        "matched_kl_direction_comparison": {"budget": args.kl_budget, "relative_matching_tolerance": 1e-4,
            "evidence_type": "interpolation; direction, preconditioner and scale fitted on all evaluated prompts",
            "directions": comparison, "cosine": (gradient * fisher_direction).sum().item() /
                max(gradient.norm().item() * fisher_direction.norm().item(), 1e-30)},
        "trajectories": sorted(trajectory_report, key=lambda r: r["trajectory_index"]),
        "prompt_groups": group_report,
        "caveats": [
            "This is an in-batch, first-order policy-score surrogate using saved real rewards/critic advantages, not actual reward improvement or an optimizer recommendation.",
            "The matched_kl_direction_comparison reuses all trajectories to fit and compare directions; its standard errors and ratios are descriptive interpolation statistics, never independent confidence evidence. The separate leave-one-prompt-group-out section uses training-only fits and scales.",
            "Prompt-cluster uncertainty is unavailable with one group and unstable with very few groups. Trajectory iid estimates ignore shared prompts. Scalar-coordinate Gaussian reference values are not reward sample sizes.",
            "Covariance rank cutoff is explicit truncation, not proof discarded modes are mathematically null; FP32 eigenspectrum can have negative roundoff modes. Full unregularized conditioning may be unresolved.",
            "Conditional expected Fisher integrates Gaussian epsilon analytically. It is not the realized empirical score Fisher, nor the joint gate/continuation Fisher, nor a full-trajectory reward covariance.",
            "If exact old scores differ by more than the trainer's 0.05 guard, the reconstructed derivative remains the saved objective derivative but cannot be described as verified on-policy; no old-score refresh is performed.",
            "Direction comparisons freeze the gate. STOP states are counterfactual Gaussian KL constraints matching controller_kl. FIRST is mandatory; forced closure and answers are absent.",
            "Per-action maximum KL<=0.02 does not bound the trajectory KL sum by 0.02. Reported sums are along sampled fixed states, not exact new-policy trajectory-distribution KL because future state visitation changes.",
            "Noise, raw actions, detached advantages and likelihood sums are unchanged. Small KL alone does not establish learnability, useful credit assignment, reward improvement or generalization.",
            "No backbone model or CPU model fallback is loaded. CUDA memory holds O(prompt_groups*d^2) sufficient statistics, never O(d^4) Fisher or all per-trajectory d^2 gradients.",
        ],
    }
    tensors = (gradient, gate_gradient, analytic_gradient, analytic_gate_gradient, fisher_direction,
               covariance, eigenvalues, per_dimension_mean, per_dimension_rms)
    report["finite_values"] = {"all_reported_head_statistics_finite": all(torch.isfinite(t).all().item() for t in tensors)}
    if not report["finite_values"]["all_reported_head_statistics_finite"]:
        raise ValueError("nonfinite analysis result")
    crossfit = {
        "status": "running" if len(groups) > 1 else "unavailable: fewer than two prompt groups",
        "split_unit": "entire prompt group, including every sampled trajectory from that prompt",
        "expected_folds": len(groups) if len(groups) > 1 else 0,
        "rcond": args.fisher_rcond, "kl_budget": args.kl_budget,
        "relative_matching_tolerance": 1e-4,
        "weighting": {
            "gradient": "Sum saved advantage-weighted action-score gradients / number of trajectories in that split; no action or dimension normalization.",
            "fisher": "Training Gaussian-state covariance sum / training Gaussian-action count; gate frozen, no gate Fisher.",
            "kl_constraint": "Maximum over all TRAIN controller states, including counterfactual STOP states; heldout states never select or revise scale.",
            "kl_mean": "Uniform over controller states in the indicated split.",
            "aggregate": "Report both equal-fold mean and heldout-trajectory-weighted mean; signs count whole prompt folds.",
        },
        "formulas": {
            "train_gradient": "G_-k=sum_{j!=k} S_j / N_-k; S_j is saved real-advantage score-gradient sum",
            "train_fisher": "C_-k=sum_{j!=k} C_sum_j/M_-k; keep lambda>rcond*lambda_max(C_-k), same relative cutoff as interpolation, fold-specific absolute cutoff",
            "directions": "D_E=G_-k; D_F=G_-k Q_ret diag(1/lambda_ret) Q_ret^T; gate delta=0",
            "scale": "Start alpha=sqrt(budget/max_TRAIN(0.5||D x||^2)); refine against actual FP32 train-head KL only, within unchanged relative tolerance below budget.",
            "heldout_signed_ascent": "alpha*<S_k/n_k,D>_F; S_k=sum advantage*((raw-mu)*exp(-log_s)^2*s) x^T, using existing autograd sufficient statistics; ideal derivative is advantage*epsilon*x^T",
            "actual_kl": "0.5*sum(((mu_new_FP32(h)-mu_old_FP32(h))/s)^2) with FP64 subtraction/reduction, separately on train and heldout states.",
        },
        "evidence_limits": [
            "Four prompt groups give high uncertainty. Folds overlap in training data, so fold signs are descriptive, not independent Bernoulli trials or a confidence interval.",
            "Heldout means held out from direction, Fisher and scale fitting only. The original policy, frozen critic and captured advantages are shared; this is not a new-prompt rollout or cross-fitted critic.",
            "Saved real advantages and exact raw FP32 epsilon are reused without new Gaussian draws. Signed projections are first-order score objectives, NOT measured reward changes.",
            "No heldout-based direction selection, scale adjustment, hyperparameter tuning or optimizer adoption occurs.",
        ],
        "folds": [], "aggregate": {},
    }
    report["leave_one_prompt_group_out"] = crossfit
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(initial=False):
        # Serialize before touching the previous checkpoint; failed strict JSON retains it.
        serialized = json.dumps(report, indent=2, allow_nan=False)
        if initial:
            with args.output.open("x") as stream:
                stream.write(serialized + "\n")
            return
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent,
                                         prefix=args.output.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(serialized + "\n")
                stream.flush()
                temporary.replace(args.output)
            finally:
                temporary.unlink(missing_ok=True)

    # Preserve every completed original diagnostic even if a later fold fails.
    checkpoint(initial=True)
    if len(groups) > 1:
        for heldout_index in sorted(groups):
            train_indices = [index for index in sorted(groups) if index != heldout_index]
            train_records = [record for index in train_indices for _, record in groups[index]]
            heldout_records = [record for _, record in groups[heldout_index]]
            train_stats = [group_statistics[index] for index in train_indices]
            heldout_stats = group_statistics[heldout_index]
            train_n = sum(stat["trajectories"] for stat in train_stats)
            train_m = sum(stat["gaussian_actions"] for stat in train_stats)
            # Explicitly sum only training groups; never subtract heldout data from a total.
            train_gradient = torch.zeros_like(gradient)
            train_gate = torch.zeros_like(gate_gradient)
            train_covariance = torch.zeros_like(covariance)
            for stat in train_stats:
                train_gradient.add_(stat["gradient_sum"])
                train_gate.add_(stat["gate_gradient_sum"])
                train_covariance.add_(stat["covariance_sum"])
            train_gradient.div_(train_n)
            train_gate.div_(train_n)
            train_covariance.div_(train_m)
            heldout_gradient = heldout_stats["gradient_sum"] / heldout_stats["trajectories"]
            fold = {
                "heldout_group_index": heldout_index, "training_group_indices": train_indices,
                "train_trajectories": train_n, "heldout_trajectories": heldout_stats["trajectories"],
                "train_gaussian_actions": train_m,
                "heldout_gaussian_actions": heldout_stats["gaussian_actions"],
                "train_gate_gradient_norm": train_gate.norm().item(),
                "heldout_gate_gradient_norm": (heldout_stats["gate_gradient_sum"] / heldout_stats["trajectories"]).norm().item(),
                "gate_frozen": True, "directions": {},
            }
            crossfit["folds"].append(fold)
            for name in ("euclidean", "fisher_pseudoinverse"):
                result = {"status": "running", "gate_frozen": True}
                fold["directions"][name] = result
                try:
                    with torch.no_grad():
                        direction = train_gradient
                        if name == "fisher_pseudoinverse":
                            fold_values, fold_vectors = torch.linalg.eigh(train_covariance)
                            fold_cutoff = args.fisher_rcond * fold_values[-1].item()
                            fold_keep = fold_values > fold_cutoff
                            if not fold_keep.any():
                                raise ValueError("no resolved positive training Fisher modes")
                            fold_retained = fold_values[fold_keep]
                            fold_basis = fold_vectors[:, fold_keep]
                            direction = ((train_gradient @ fold_basis) / fold_retained) @ fold_basis.T
                            result["fisher_fit"] = {
                                "rcond": args.fisher_rcond, "absolute_cutoff": fold_cutoff,
                                "retained_rank": fold_keep.sum().item(),
                                "discarded_modes": (~fold_keep).sum().item(),
                                "largest_eigenvalue": fold_values[-1].item(),
                                "smallest_eigenvalue": fold_values[0].item(),
                                "retained_condition_number": (fold_retained[-1] / fold_retained[0]).item(),
                            }
                        maximum = 0.0
                        for record in train_records:
                            for start in range(0, record["observations"].shape[0], args.batch_size):
                                states = record["observations"][start:start + args.batch_size].to("cuda")
                                features = F.rms_norm(states.float(), (dimension,))
                                displacement = F.linear(features, direction[:, :-1], direction[:, -1])
                                value = (.5 * displacement.square().sum(-1)).max().item()
                                if not math.isfinite(value):
                                    raise ValueError("nonfinite training displacement")
                                maximum = max(maximum, value)
                        scale, initial_scale, refinements, train_kl = select_scale(direction, train_records, maximum)
                        # Scale is frozen before any heldout KL or signed projection is evaluated.
                        heldout_kl = actual_kl(direction, scale, selected_records=heldout_records)
                        train_ascent = scale * (train_gradient.double() * direction.double()).sum().item()
                        heldout_ascent = scale * (heldout_gradient.double() * direction.double()).sum().item()
                        result.update({
                            "status": "matched", "scale": scale,
                            "ideal_quadratic_scale": initial_scale,
                            "rounding_scale_refinements": refinements,
                            "train_kl": train_kl, "heldout_kl": heldout_kl,
                            "heldout_max_kl_over_training_budget": heldout_kl["max_gaussian_kl_all_controller_states"] / args.kl_budget,
                            "train_signed_first_order_objective_ascent": train_ascent,
                            "heldout_signed_first_order_objective_ascent": heldout_ascent,
                            "heldout_ascent_sign": 1 if heldout_ascent > 0 else -1 if heldout_ascent < 0 else 0,
                            "train_minus_heldout_signed_ascent": train_ascent - heldout_ascent,
                            "parameter_step_norm": scale * direction.norm().item(),
                            "global_mean_head_kl": global_kl_bound(direction, scale),
                        })
                        # A nonfinite result is a failed fold, never clipped or nonstandard JSON.
                        json.dumps(result, allow_nan=False)
                except (RuntimeError, ValueError) as error:
                    fold["directions"][name] = {"status": "failed", "error": str(error)}
                checkpoint()
        for name in ("euclidean", "fisher_pseudoinverse"):
            completed = [(fold, fold["directions"][name]) for fold in crossfit["folds"]
                         if fold["directions"][name]["status"] == "matched"]
            if not completed:
                crossfit["aggregate"][name] = {"completed_folds": 0, "expected_folds": len(groups)}
                continue
            signed = [result["heldout_signed_first_order_objective_ascent"] for _, result in completed]
            weights = [fold["heldout_trajectories"] for fold, _ in completed]
            crossfit["aggregate"][name] = {
                "completed_folds": len(completed), "expected_folds": len(groups),
                "positive_folds": sum(value > 0 for value in signed),
                "negative_folds": sum(value < 0 for value in signed),
                "zero_folds": sum(value == 0 for value in signed),
                "positive_fraction_completed_folds": sum(value > 0 for value in signed) / len(signed),
                "all_expected_folds_positive": len(completed) == len(groups) and all(value > 0 for value in signed),
                "equal_fold_mean_signed_ascent": sum(signed) / len(signed),
                "trajectory_weighted_mean_signed_ascent": sum(value * weight for value, weight in zip(signed, weights)) / sum(weights),
                "per_fold_signed_ascent": signed,
                "completed_heldout_group_indices": [fold["heldout_group_index"] for fold, _ in completed],
                "heldout_kl_budget_exceeding_folds": sum(result["heldout_max_kl_over_training_budget"] > 1 for _, result in completed),
            }
        paired = [fold for fold in crossfit["folds"]
                  if all(result["status"] == "matched" for result in fold["directions"].values())]
        differences = [fold["directions"]["fisher_pseudoinverse"]["heldout_signed_first_order_objective_ascent"]
                       - fold["directions"]["euclidean"]["heldout_signed_first_order_objective_ascent"]
                       for fold in paired]
        crossfit["paired_fisher_minus_euclidean"] = {
            "heldout_group_indices": [fold["heldout_group_index"] for fold in paired],
            "signed_ascent_differences": differences,
            "equal_fold_mean_difference": sum(differences) / len(differences) if differences else None,
            "fisher_higher_folds": sum(value > 0 for value in differences),
            "interpretation": "Descriptive paired first-order comparison at TRAIN-matched KL, not necessarily matched heldout KL or actual reward.",
        }
        crossfit["status"] = "complete" if len(paired) == len(groups) else "partial_failure"
    checkpoint()
    print(json.dumps({"output": str(args.output), "trajectories": ntraj,
                      "prompt_groups": len(groups), "gaussian_actions": gaussian_count,
                      "old_score_max_abs_error": errors.abs().max().item(),
                      "retained_feature_rank": keep.sum().item()}), flush=True)
    if crossfit["status"] == "partial_failure":
        raise RuntimeError(f"cross-fit diagnostics partially failed; completed results retained at {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Queue-only frozen MiniCPM hidden-to-input ridge experiment, not RL training.

Submit through mlq (Main owns submission), with --queued-run. Dataset/model loading,
replay, regression and evaluation happen only inside main after that acknowledgement.
The vocabulary head is used for native teachers and visible answers, never thoughts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from postraining.core import load_unique_math_rows
from postraining.fast_inference import CapturedTrainingRolloutEngine
from postraining.minicpm_latent_rollout import MiniCPMLatentRolloutEngine
from postraining.vapo.policy import (
    VAPOPolicy,
    collate_replay_microbatch,
    plan_replay_microbatches,
)
from postraining.vapo.model.hf import (
    enable_packed_replay_attention,
    enable_replay_mlp_compilation,
    use_packed_replay_attention,
)
from postraining.vapo.model.lora import LoRAConfig
from postraining.train_minicpm_vapo import (
    _stop_ids,
    collect_rollouts,
    encode_math_prompt,
    file_sha256,
    resolve_thinking_end_token,
    resolve_thinking_start_token,
    rollout_diagnostics,
)


class RidgeThoughtInput(torch.nn.Module):
    """Apply the fitted fp32 linear map to untouched fp32 Gaussian actions."""

    def __init__(self, weight: torch.Tensor, feature_scale: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("feature_scale", feature_scale)

    def forward(self, base: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=raw.device.type, enabled=False):
            projected = F.linear(raw.float() / self.feature_scale, self.weight)
        return projected.to(dtype=base.dtype)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queued-run", action="store_true", required=True,
                        help="Caller attests this process is supervised by mlq; not a queue bypass.")
    parser.add_argument("--output", type=Path, required=True, help="New report JSON path.")
    parser.add_argument("--data", type=Path, default=Path("postraining/data/dapo-math-17k.parquet"))
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--calibration-prompts", type=int, default=32,
                        help="Total calibration prompts, including validation prompts.")
    parser.add_argument("--validation-prompts", type=int, default=8)
    parser.add_argument("--calibration-samples", type=int, default=2)
    parser.add_argument("--evaluation-prompts", type=int, default=32)
    parser.add_argument("--evaluation-samples", type=int, default=4)
    parser.add_argument("--group-prompts", type=int, default=8)
    parser.add_argument("--physical-batch-size", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument("--answer-reserve-tokens", type=int, default=1024)
    parser.add_argument("--gate-probabilities", type=float, nargs="+", default=[0.9, 1 / 64, 1 / 1024])
    parser.add_argument("--states-per-trajectory", type=int, default=512,
                        help="Uniform sample of native thinking transitions; replay itself is complete.")
    parser.add_argument("--replay-token-budget", type=int, default=22048)
    parser.add_argument("--replay-max-trajectories", type=int, default=4)
    parser.add_argument("--regression-batch-size", type=int, default=4096)
    parser.add_argument("--ridge", type=float, default=0.001,
                        help="Fixed regularization of mean normalized-feature Gram; no validation tuning.")
    parser.add_argument("--estimated-aggregate-tokens-per-second", type=float, default=250,
                        help="Unmeasured conservative planning assumption, not a benchmark or guarantee.")
    parser.add_argument("--estimated-overhead-seconds", type=float, default=3600)
    return parser


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def flush(path: Path, report: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_tensors(path: Path, payload: dict) -> dict:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def validate_args(args) -> None:
    positive = ("calibration_prompts", "validation_prompts", "calibration_samples",
                "evaluation_prompts", "evaluation_samples", "group_prompts", "physical_batch_size",
                "prompt_tokens", "max_new_tokens", "answer_reserve_tokens", "states_per_trajectory",
                "replay_token_budget", "replay_max_trajectories", "regression_batch_size")
    if any(getattr(args, name) < 1 for name in positive):
        raise ValueError("Counts and budgets must be positive")
    if args.validation_prompts >= args.calibration_prompts:
        raise ValueError("Calibration must include nonempty, distinct fit and validation splits")
    if any(count % args.group_prompts for count in (args.calibration_prompts, args.evaluation_prompts)):
        raise ValueError("Calibration and evaluation prompt counts must divide into complete groups")
    if args.max_new_tokens < args.answer_reserve_tokens + 2:
        raise ValueError("Generation budget must fit thoughts, close delimiter, and answer reserve")
    if args.replay_token_budget < args.prompt_tokens + args.max_new_tokens - 1:
        raise ValueError("Replay budget must accommodate a full native trajectory")
    if not args.gate_probabilities or not all(0 < p < 1 for p in args.gate_probabilities):
        raise ValueError("Gate probabilities must be strictly between zero and one")
    if len(set(args.gate_probabilities)) != len(args.gate_probabilities):
        raise ValueError("Duplicate evaluation arms are not useful")
    if not math.isfinite(args.ridge) or args.ridge <= 0:
        raise ValueError("Ridge must be finite and positive")
    if not math.isfinite(args.estimated_aggregate_tokens_per_second) or args.estimated_aggregate_tokens_per_second <= 0:
        raise ValueError("Runtime throughput assumption must be finite and positive")
    if not math.isfinite(args.estimated_overhead_seconds) or args.estimated_overhead_seconds < 0:
        raise ValueError("Runtime overhead assumption must be finite and nonnegative")


def collect_arm(policy, tokenizer, rows, hashes, args, report, artifacts, stops, opening, close,
                name, samples, seed_offset, *, latent=False, retain=False):
    policy.latent_thinking = latent
    kwargs = dict(stop_ids=stops, prompts_per_rollout=args.group_prompts,
                  samples_per_prompt=samples, physical_batch_size=min(args.physical_batch_size,
                  args.group_prompts * samples), cache_length=args.prompt_tokens + args.max_new_tokens,
                  temperature=0.9, top_k=20, top_p=0.95, thinking_end_token_id=close,
                  answer_reserve_tokens=args.answer_reserve_tokens, compile_decode=True)
    engine = (MiniCPMLatentRolloutEngine(policy, thinking_start_token_id=opening, **kwargs)
              if latent else CapturedTrainingRolloutEngine(policy, **kwargs))
    arm = {"groups": [], "responses": [], "artifacts": [], "compiled_decode": True,
           "arithmetic": engine.arithmetic, "samples_per_prompt": samples,
           "thinking_vocabulary_projection": False if latent else True}
    report["arms"][name] = arm
    retained = []
    try:
        for start in range(0, len(rows), args.group_prompts):
            group_seed = args.seed + seed_offset + start
            torch.manual_seed(group_seed)
            began = time.perf_counter()
            result = collect_rollouts(engine, tokenizer, rows[start:start + args.group_prompts],
                                      prompt_tokens=args.prompt_tokens, max_new_tokens=args.max_new_tokens,
                                      enable_thinking=True)
            metrics = rollout_diagnostics(result, samples_per_prompt=samples, stop_ids=stops)
            metrics.update(wall_seconds=time.perf_counter() - began, seed=group_seed)
            arm["groups"].append(metrics)
            payload = []
            for index, record in enumerate(result.records):
                prompt_index = start + index // samples
                response = record.token_ids[record.prompt_length:]
                closing = (response == close).nonzero(as_tuple=False).flatten()
                native_thoughts = int(closing[0]) if closing.numel() else record.response_length
                thoughts = int(record.latent_vectors.shape[0]) if latent else native_thoughts
                eos = bool(response.numel() and int(response[-1]) in stops)
                arm["responses"].append({
                    "prompt_index": prompt_index, "prompt_sha256": hashes[prompt_index],
                    "sample_index": index % samples, "correct": bool(record.correct),
                    "text": record.text, "stream_positions": record.response_length,
                    "thoughts": thoughts, "forced_close": record.forced_token_index >= 0,
                    "thinking_termination": ("forced_budget" if record.forced_token_index >= 0 else
                                             "learned_gate" if latent else
                                             "native_close" if closing.numel() else "no_close"),
                    "answer_termination": "eos" if eos else "token_budget",
                    "final_token_id": int(response[-1]),
                })
                payload.append({"prompt_sha256": hashes[prompt_index], "sample_index": index % samples,
                                "token_ids": record.token_ids.cpu(), "prompt_length": record.prompt_length,
                                "forced_token_index": record.forced_token_index,
                                "latent_vectors": record.latent_vectors.cpu() if latent else None,
                                "action_kinds": record.action_kinds.cpu() if latent else None})
            arm["artifacts"].append(save_tensors(artifacts / f"{name}-{start:04d}.pt", {"records": payload}))
            if retain:
                retained.extend(result.records)
            arm["accuracy"] = sum(r["correct"] for r in arm["responses"]) / len(arm["responses"])
            arm["mean_thought_length"] = sum(r["thoughts"] for r in arm["responses"]) / len(arm["responses"])
            arm["wall_seconds"] = sum(group["wall_seconds"] for group in arm["groups"])
            if latent:
                arm["sampler_telemetry"] = dict(engine.telemetry)
            flush(args.output, report)
            print(json.dumps({"arm": name, "group_start": start, **metrics}), flush=True)
            engine.release_cache()
            del result, payload
            gc.collect()
    finally:
        engine.release_cache()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    return retained


@torch.no_grad()
def replay_features(policy, records, fit_prompts, args, report, artifacts, close, excluded_tokens):
    """Select h[t-1] -> embedding(token[t]) only before the first native </think>."""
    selected = []
    rng = random.Random(args.seed + 100000)
    for record in records:
        response = record.token_ids[record.prompt_length:]
        closing = (response == close).nonzero(as_tuple=False).flatten()
        end = int(closing[0]) if closing.numel() else response.numel()
        eligible = [position for position, token in enumerate(response[:end].tolist())
                    if token not in excluded_tokens]
        indices = sorted(rng.sample(eligible, min(len(eligible), args.states_per_trajectory)))
        selected.append(torch.tensor(indices, dtype=torch.long))
    plan = plan_replay_microbatches(records, list(range(len(records))),
                                  token_budget=args.replay_token_budget,
                                  max_trajectories=args.replay_max_trajectories)
    enable_packed_replay_attention(policy.causal_lm)
    enable_replay_mlp_compilation(policy.causal_lm)
    features = {name: {"hidden": [], "target_ids": [], "record_indices": [], "response_positions": []}
                for name in ("fit", "validation")}
    began = time.perf_counter()
    try:
        for shard_number, indices in enumerate(plan):
            batch = collate_replay_microbatch(records, indices, pad_token_id=0, device=torch.device("cuda"))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = policy.replay_hidden(batch.input_ids, batch.attention_mask,
                    position_ids=batch.position_ids, cu_seqlens=batch.cu_seqlens,
                    sequence_boundaries=batch.sequence_boundaries,
                    max_sequence_length=batch.max_sequence_length)
            offset = 0
            for index in indices:
                record = records[index]
                positions = selected[index]
                if positions.numel():
                    state_positions = (positions + offset + record.prompt_length - 1).to("cuda")
                    split = "fit" if index // args.calibration_samples < fit_prompts else "validation"
                    features[split]["hidden"].append(hidden[0, state_positions].to(torch.bfloat16).cpu())
                    features[split]["target_ids"].append(record.token_ids[record.prompt_length + positions].long())
                    features[split]["record_indices"].append(torch.full_like(positions, index))
                    features[split]["response_positions"].append(positions)
                offset += record.input_length
            report["replay"] = {"completed_shards": shard_number + 1, "total_shards": len(plan),
                                "wall_seconds": time.perf_counter() - began,
                                "state_definition": "post-final-norm hidden at native token position t-1",
                                "target_definition": "pretrained input embedding of actual sampled native thinking token t",
                                "precision": "bf16 trunk and stored hidden, fp32 regression",
                                "sampling": "fixed uniform without replacement per native thinking trajectory"}
            flush(args.output, report)
            del hidden, batch
    finally:
        use_packed_replay_attention(policy.causal_lm, enabled=False)
    for split, columns in features.items():
        if not columns["hidden"]:
            raise RuntimeError(f"No native thinking states in {split} split")
        features[split] = {key: torch.cat(values) for key, values in columns.items()}
        report["replay"][split] = save_tensors(artifacts / f"{split}-native-features.pt", features[split])
        report["replay"][split]["states"] = features[split]["hidden"].shape[0]
    flush(args.output, report)
    return features


def feature_batches(features, policy, batch_size):
    for start in range(0, features["hidden"].shape[0], batch_size):
        hidden = features["hidden"][start:start + batch_size].to("cuda").float()
        target_ids = features["target_ids"][start:start + batch_size].to("cuda")
        yield hidden, policy.token_embeddings(target_ids).float()


@torch.no_grad()
def fit_bridge(policy, features, args):
    """Solve E[z z^T] W^T + lambda W^T = E[z y^T] on the GPU."""
    fit = features["fit"]
    n, dimension = fit["hidden"].shape
    scale_squared = torch.zeros((), device="cuda", dtype=torch.float32)
    for start in range(0, n, args.regression_batch_size):
        x = fit["hidden"][start:start + args.regression_batch_size].to("cuda").float()
        scale_squared.add_(x.square().sum())
    scale = (scale_squared / (n * dimension)).sqrt()
    if not torch.isfinite(scale).item() or scale.item() <= 0:
        raise RuntimeError("Native hidden features have invalid RMS")
    gram = torch.zeros((dimension, dimension), device="cuda", dtype=torch.float32)
    cross = torch.zeros_like(gram)
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        for x, y in feature_batches(fit, policy, args.regression_batch_size):
            z = x / scale
            gram.addmm_(z.T, z, alpha=1 / n)
            cross.addmm_(z.T, y, alpha=1 / n)
        eigenvalues = torch.linalg.eigvalsh(gram)
        gram.diagonal().add_(args.ridge)
        weight = torch.linalg.solve(gram, cross).T.contiguous()
        relative_residual = (gram @ weight.T - cross).norm() / cross.norm()
        if not torch.isfinite(weight).all().item():
            raise RuntimeError("Ridge solution is nonfinite")
        solver_report = {
            "fit_states": n, "dimension": dimension, "parameters": dimension * dimension,
            "ridge": args.ridge, "feature_rms": scale.item(), "intercept": False,
            "normalization": "one fit-only RMS scalar; no centering or output renormalization",
            "objective": "mean squared embedding error plus ridge times normalized-map Frobenius norm squared",
            "solver": "torch.linalg.solve on CUDA fp32, IEEE fp32 matmul (no TF32)",
            "gram_min_eigenvalue": eigenvalues[0].item(), "gram_max_eigenvalue": eigenvalues[-1].item(),
            "regularized_condition_number": ((eigenvalues[-1] + args.ridge) / (eigenvalues[0] + args.ridge)).item(),
            "relative_linear_system_residual": relative_residual.item(),
        }
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    return RidgeThoughtInput(weight, scale).eval(), solver_report


@torch.no_grad()
def representation_metrics(policy, bridge, features, args):
    totals = torch.zeros(12, dtype=torch.float64, device="cuda")
    per_dim_sum = torch.zeros(bridge.weight.shape[1], device="cuda")
    per_dim_square = torch.zeros_like(per_dim_sum)
    norm_min = torch.full((), torch.inf, device="cuda")
    norm_max = torch.zeros((), device="cuda")
    noise_rng = torch.Generator(device="cuda").manual_seed(args.seed + 200000)
    for x, y in feature_batches(features, policy, args.regression_batch_size):
        predicted = bridge(x.to(torch.bfloat16), x).float()
        noisy_raw = x + torch.randn(x.shape, device="cuda", generator=noise_rng) * policy.transition.component_std
        noisy = bridge(x.to(torch.bfloat16), noisy_raw).float()
        norms = x.norm(dim=-1)
        totals += torch.stack((
            (predicted - y).square().sum(), y.square().sum(),
            F.cosine_similarity(predicted, y).sum(), (noisy - y).square().sum(),
            F.cosine_similarity(noisy, y).sum(), predicted.norm(dim=-1).sum(),
            y.norm(dim=-1).sum(), norms.sum(), norms.square().sum(),
            (predicted - y).square().mean(dim=-1).sum(),
            (noisy - predicted).square().sum(), torch.tensor(x.shape[0], device="cuda"),
        )).double()
        per_dim_sum += x.sum(dim=0)
        per_dim_square += x.square().sum(dim=0)
        norm_min = torch.minimum(norm_min, norms.min())
        norm_max = torch.maximum(norm_max, norms.max())
    values = totals.cpu().tolist()
    n = int(values[11])
    variance = (per_dim_square / n - (per_dim_sum / n).square()).clamp_min(0)
    return {
        "states": n, "mse_per_component": values[9] / n,
        "relative_mse_vs_zero_embedding": values[0] / values[1], "cosine": values[2] / n,
        "gaussian_noisy_relative_mse": values[3] / values[1], "gaussian_noisy_cosine": values[4] / n,
        "gaussian_output_perturbation_relative_energy": values[10] / values[1],
        "prediction_mean_norm": values[5] / n, "target_mean_norm": values[6] / n,
        "hidden_native_state_features": {
            "mean_norm": values[7] / n, "norm_std": math.sqrt(max(0, values[8] / n - (values[7] / n) ** 2)),
            "min_norm": norm_min.item(), "max_norm": norm_max.item(),
            "mean_vector_norm": (per_dim_sum / n).norm().item(),
            "rms": math.sqrt(per_dim_square.sum().item() / (n * per_dim_sum.numel())),
            "dimension_variance_min": variance.min().item(), "dimension_variance_max": variance.max().item(),
        },
        "predictions": "fp32 map rounded to bf16 as deployed; independent fixed Gaussian diagnostic noise",
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.output.exists():
        raise FileExistsError(args.output)
    artifacts = args.output.with_suffix("").with_name(args.output.stem + "-artifacts")
    artifacts.mkdir(parents=True, exist_ok=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    maximum_positions = args.max_new_tokens * (
        args.calibration_prompts * args.calibration_samples +
        args.evaluation_prompts * args.evaluation_samples * (len(args.gate_probabilities) + 1))
    report = {
        "status": "initializing", "args": {key: str(value) if isinstance(value, Path) else value
                                            for key, value in vars(args).items()},
        "scope": "Standalone supervised representation diagnostic, followed by real-reward inference; no RL learning claim",
        "queue": {"caller_attestation": args.queued_run, "pid": os.getpid(), "submission_owner": "Main"},
        "runtime_estimate": {
            "label": "[INFERENCE] planning estimate, not measured and not a wall-time guarantee",
            "maximum_useful_generated_positions": maximum_positions,
            "assumed_aggregate_positions_per_second": args.estimated_aggregate_tokens_per_second,
            "assumed_model_loading_compilation_replay_regression_overhead_seconds": args.estimated_overhead_seconds,
            "estimated_max_runtime_seconds": maximum_positions / args.estimated_aggregate_tokens_per_second + args.estimated_overhead_seconds,
            "limitations": "Padding, graph compilation, prompt lengths, verification and filesystem stalls may exceed this estimate",
        },
        "limitations": [
            "A sampled next token is one sparse draw from an ambiguous lexical distribution; ridge estimates its conditional mean embedding, not a valid discrete token or guaranteed reasoning transition.",
            "Teacher-forced native-state fit does not establish stability on self-generated latent states; only held-out rollout outcomes test the deployed bridge.",
            "Evaluation is prompt-held-out from this experiment's fit and validation, but remains DAPO training-pool data; pretraining contamination and overlap with unrelated prior experiments are unknown.",
            "Replay and compiled rollout can differ numerically; this is not an exact replay/KL validation experiment.",
        ],
        "provenance": {"source_sha256": file_sha256(__file__), "torch_version": torch.__version__,
                       "source_dependencies": {path: file_sha256(path) for path in (
                           "postraining/vapo/policy.py", "postraining/minicpm_latent_rollout.py",
                           "postraining/fast_inference.py", "postraining/train_minicpm_vapo.py",
                           "postraining/latent_thought.py")}},
        "arms": {},
    }
    flush(args.output, report)
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Queued CUDA GPU required; there is no CPU/model fallback")
        torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")
        report["provenance"].update(data_sha256=file_sha256(args.data), gpu=torch.cuda.get_device_name())
        all_rows = load_unique_math_rows(str(args.data))
        random.Random(args.seed).shuffle(all_rows)
        needed = args.calibration_prompts + args.evaluation_prompts
        if len(all_rows) < needed:
            raise ValueError("Dataset has too few unique prompts")
        rows = all_rows[:needed]
        del all_rows
        prompt_hashes = [digest(row["prompt"]) for row in rows]
        if len(set(prompt_hashes)) != needed:
            raise ValueError("Prompt hash overlap violates fit/validation/evaluation exclusion")
        fit_prompts = args.calibration_prompts - args.validation_prompts
        boundaries = {"fit": (0, fit_prompts), "validation": (fit_prompts, args.calibration_prompts),
                      "evaluation": (args.calibration_prompts, needed)}
        report["splits"] = {name: {"prompt_sha256": prompt_hashes[start:end],
                                           "prompts": [row["prompt"] for row in rows[start:end]]}
                            for name, (start, end) in boundaries.items()}
        report["split_exclusion"] = {"fit_validation_overlap": 0, "fit_evaluation_overlap": 0,
                                     "validation_evaluation_overlap": 0,
                                     "evaluation_used_for_fit_or_hyperparameter_selection": False,
                                     "validation_used_for_hyperparameter_selection": False,
                                     "fit_selection": "fixed shuffled prompt partition; never correctness-filtered"}
        flush(args.output, report)
        policy, tokenizer = VAPOPolicy.from_family("minicpm5", device=torch.device("cuda"),
            lora_config=LoRAConfig(initialization="nora"), gradient_checkpointing=False,
            latent_thinking=True, thought_sigma=1.0)
        policy.eval().requires_grad_(False)
        stops = _stop_ids(policy, tokenizer)
        opening = resolve_thinking_start_token(tokenizer, stop_ids=stops)
        close = resolve_thinking_end_token(tokenizer, stop_ids=stops)
        encoded = [encode_math_prompt(tokenizer, row, prompt_tokens=args.prompt_tokens,
                                      enable_thinking=True).tolist() for row in rows]
        if len(set(digest(ids) for ids in encoded)) != needed:
            raise ValueError("Truncated encoded prompt overlap violates split exclusion")
        report["split_exclusion"]["encoded_prompt_overlap"] = 0
        for name, (start, end) in boundaries.items():
            report["splits"][name]["encoded_prompt_sha256"] = [digest(ids) for ids in encoded[start:end]]
        report["provenance"].update(model_id=policy.model_id, revision=policy.revision,
                                   hidden_size=int(policy.causal_lm.config.hidden_size),
                                   pretrained_model_dtype=str(next(policy.causal_lm.parameters()).dtype))
        report["sampler_contract"] = {"mean": "b (frozen zero residual mean head)",
            "vector_sigma": policy.transition.vector_sigma, "component_std": policy.transition.component_std,
            "raw_dtype": "torch.float32", "adapter_output_dtype": "torch.bfloat16",
            "raw_vectors": "saved exactly from real collector, never normalized, recentered, or replaced",
            "head_bypass": "stock MiniCPMLatentRolloutEngine _thought uses transition plus adapter only; vocabulary projection confined to admitted answer lanes"}
        report["status"] = "collecting_native_calibration"
        flush(args.output, report)
        records = collect_arm(policy, tokenizer, rows[:args.calibration_prompts],
            prompt_hashes[:args.calibration_prompts], args, report, artifacts, stops, opening, close,
            "native_calibration", args.calibration_samples, 10000, retain=True)
        report["status"] = "replaying_native_calibration"
        features = replay_features(policy, records, fit_prompts, args, report, artifacts, close,
                                   set(tokenizer.all_special_ids) | set(stops) | {opening, close})
        del records
        report["status"] = "fitting_bridge"
        flush(args.output, report)
        fit_started = time.perf_counter()
        bridge, report["ridge_fit"] = fit_bridge(policy, features, args)
        report["ridge_fit"]["wall_seconds"] = time.perf_counter() - fit_started
        report["representation_metrics"] = {
            split: representation_metrics(policy, bridge, values, args)
            for split, values in features.items()}
        report["bridge_artifact"] = save_tensors(artifacts / "bridge.pt", {
            "schema": "minicpm_hidden_to_input_ridge_experiment/v1", "state_dict":
                {key: value.cpu() for key, value in bridge.state_dict().items()},
            "model_id": policy.model_id, "revision": policy.revision, "ridge_fit": report["ridge_fit"],
            "sampler_contract": report["sampler_contract"], "splits": report["splits"],
            "source_sha256": report["provenance"]["source_sha256"],
        })
        policy.thought_adapter = bridge
        del features
        gc.collect()
        torch.cuda.empty_cache()
        evaluation_rows = rows[args.calibration_prompts:]
        evaluation_hashes = prompt_hashes[args.calibration_prompts:]
        report["status"] = "evaluating_native_control"
        flush(args.output, report)
        collect_arm(policy, tokenizer, evaluation_rows, evaluation_hashes, args, report, artifacts,
                    stops, opening, close, "native_control", args.evaluation_samples, 30000)
        for probability in args.gate_probabilities:
            with torch.no_grad():
                policy.thinking_gate.head.weight.zero_()
                policy.thinking_gate.head.bias.fill_(math.log(probability / (1 - probability)))
            report["status"] = f"evaluating_latent_stop_{probability:g}"
            flush(args.output, report)
            collect_arm(policy, tokenizer, evaluation_rows, evaluation_hashes, args, report, artifacts,
                        stops, opening, close, f"latent_stop_{probability:g}",
                        args.evaluation_samples, 30000, latent=True)
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        flush(args.output, report)


if __name__ == "__main__":
    main()

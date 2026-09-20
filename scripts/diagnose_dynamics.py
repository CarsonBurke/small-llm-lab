"""Mechanistic dynamics replay and frozen-batch probes; execute only through mlq.

No arguments: instrumentation around the existing full training loop; launch via
scripts/ablation.py with ordinary dynamics environment settings. `frozen-batches`
scores hard/easy training batches with fixed completed checkpoints.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pretraining.nanogpt_mini import nanogpt_mini_native_bits_train as training
from pretraining.nanogpt_mini.bit_density import binary_nll
from pretraining.nanogpt_mini.native_bits_data import (
    PreparedData,
    cyclic_ids,
    microbatches,
    sha256_file,
)
from scripts.native_bits import _load_pinned_checkpoint


@torch.compile(dynamic=True, fullgraph=True)
def trace_step(model, state, previous, age, teacher):
    candidate_age = age + 1
    candidate, estimate = model.advance(state, previous, candidate_age)
    route = estimate >= model.config.refresh_cost
    return (
        torch.where(route[:, None], teacher, candidate),
        torch.where(route, 0, candidate_age),
        candidate,
        candidate_age,
        estimate,
        route,
    )


@torch.compile(dynamic=True, fullgraph=True)
def score_trace(model, teacher, candidates, codes, ages, estimates, routes):
    p_logits = model.score(teacher, codes)
    q_logits = model.score(candidates, codes)
    p = p_logits.sigmoid()
    kl = (
        F.binary_cross_entropy_with_logits(q_logits, p, reduction="none")
        - F.binary_cross_entropy_with_logits(p_logits, p, reduction="none")
    ).sum(-1)
    teacher_rms = teacher.float().square().mean(-1).sqrt()
    candidate_rms = candidates.float().square().mean(-1).sqrt()
    features = torch.cat(
        (
            candidates.to(torch.bfloat16),
            ages.float().log1p()[..., None].to(torch.bfloat16),
        ),
        -1,
    )
    hidden = model.error_predictor.input(features).relu()
    raw = model.error_predictor.output(hidden.square()).float().squeeze(-1)
    age_derivative = (
        2
        * hidden.float()
        * model.error_predictor.output.weight[0].to(torch.bfloat16).float()
        * model.error_predictor.input.weight[:, -1].to(torch.bfloat16).float()
    ).sum(-1)
    # Diagnostic interventions only: never used to decide a route or train.
    age_two = torch.full_like(ages, 2)
    scale = teacher_rms / candidate_rms.clamp_min(torch.finfo(torch.float32).tiny)
    rescaled = (candidates.float() * scale[..., None]).to(torch.bfloat16)
    radius_logits = model.score(rescaled, codes)
    selected_rms = torch.where(routes, teacher_rms, candidate_rms)
    previous_rms = torch.cat((teacher_rms[:, :1], selected_rms[:, :-1]), dim=1)
    joint_mean_square = (
        model.config.model_dim * previous_rms.square() + model.config.code_bits
    ) / (model.config.model_dim + model.config.code_bits)
    return {
        "teacher_nll": binary_nll(p_logits, codes, 1),
        "rollout_nll": binary_nll(
            torch.where(routes[..., None], p_logits, q_logits), codes, 1
        ),
        "radius_repaired_rollout_nll": binary_nll(
            torch.where(routes[..., None], p_logits, radius_logits), codes, 1
        ),
        "candidate_all_bits_saturated_same_sign": (
            (q_logits.amin(-1) > 14) | (q_logits.amax(-1) < -14)
        ),
        "candidate_conditional_bit_entropy": F.binary_cross_entropy_with_logits(
            q_logits, q_logits.sigmoid(), reduction="none"
        ).sum(-1),
        "transition_character_inverse_rms": (
            joint_mean_square + torch.finfo(torch.bfloat16).eps
        ).rsqrt(),
        "candidate_kl": kl,
        "estimate": estimates,
        "age": ages,
        "route": routes,
        "teacher_rms": teacher_rms,
        "candidate_rms": candidate_rms,
        "state_mse": (candidates.float() - teacher.float()).square().mean(-1),
        "gate_raw": raw,
        "gate_age_derivative": age_derivative,
        "estimate_at_age_two": model.error_predictor(candidates, age_two),
        "estimate_at_teacher_radius": model.error_predictor(rescaled, ages),
        "estimate_at_both_interventions": model.error_predictor(rescaled, age_two),
    }


@torch.inference_mode()
def rollout_trace(model, ids):
    codes = model.identity_bits(ids)
    teacher = model.teacher_context(codes)
    state = teacher[:, 0]
    age = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
    candidates, ages = [state], [age]
    estimates = [torch.zeros(ids.shape[0], device=ids.device)]
    routes = [torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)]
    for index in range(1, ids.shape[1]):
        state, age, candidate, candidate_age, estimate, route = trace_step(
            model, state, codes[:, index - 1], age, teacher[:, index]
        )
        candidates.append(candidate)
        ages.append(candidate_age)
        estimates.append(estimate)
        routes.append(route)
    result = score_trace(
        model,
        teacher,
        torch.stack(candidates, 1),
        codes,
        torch.stack(ages, 1),
        torch.stack(estimates, 1),
        torch.stack(routes, 1),
    )
    return {key: value.detach().float().cpu().numpy() for key, value in result.items()}


def summarize_trace(arrays, byte_count, threshold, horizon):
    routes = arrays["route"].astype(bool)
    applicable = arrays["age"] > 0
    unsafe = applicable & (arrays["candidate_kl"] > threshold)
    accepted = applicable & ~routes
    false_safe = accepted & unsafe
    long = applicable & (arrays["age"] > horizon)
    excess = arrays["rollout_nll"] - arrays["teacher_nll"]
    bad_tail = false_safe & long
    positive_excess = np.maximum(excess, 0)
    teacher_bpb = float(
        arrays["teacher_nll"].sum(dtype=np.float64) / math.log(2) / byte_count
    )
    rollout_bpb = float(
        arrays["rollout_nll"].sum(dtype=np.float64) / math.log(2) / byte_count
    )
    return {
        "teacher_bpb": teacher_bpb,
        "rollout_bpb": rollout_bpb,
        "radius_repaired_rollout_bpb": float(
            arrays["radius_repaired_rollout_nll"].sum(dtype=np.float64)
            / math.log(2)
            / byte_count
        ),
        "false_safe_same_sign_saturated_positions": int(
            (
                false_safe
                & arrays["candidate_all_bits_saturated_same_sign"].astype(bool)
            ).sum()
        ),
        "minimum_transition_character_inverse_rms": float(
            arrays["transition_character_inverse_rms"][applicable].min()
        )
        if applicable.any()
        else 0.0,
        "rollout_excess_bpb": rollout_bpb - teacher_bpb,
        "refresh_rate": float(routes.mean()),
        "max_candidate_age": int(arrays["age"].max()),
        "beyond_training_horizon_positions": int(long.sum()),
        "false_safe_positions": int(false_safe.sum()),
        "false_safe_beyond_horizon_positions": int(bad_tail.sum()),
        "long_false_safe_positive_excess_fraction": float(
            positive_excess[bad_tail].sum() / max(float(positive_excess.sum()), 1e-30)
        ),
        "max_candidate_rms": float(arrays["candidate_rms"].max()),
        "max_teacher_rms": float(arrays["teacher_rms"].max()),
        "max_candidate_kl": float(arrays["candidate_kl"].max()),
        "minimum_gate_raw": float(arrays["gate_raw"][applicable].min())
        if applicable.any()
        else 0.0,
        "negative_age_derivative_false_safe": int(
            (false_safe & (arrays["gate_age_derivative"] < 0)).sum()
        ),
        "false_safe_recovered_by_age_two": int(
            (false_safe & (arrays["estimate_at_age_two"] >= threshold)).sum()
        ),
        "false_safe_recovered_by_teacher_radius": int(
            (false_safe & (arrays["estimate_at_teacher_radius"] >= threshold)).sum()
        ),
        "false_safe_recovered_by_both": int(
            (false_safe & (arrays["estimate_at_both_interventions"] >= threshold)).sum()
        ),
    }


def train_replay():
    config = training.TrainConfig.from_env()
    if config.model_kind not in training.DYNAMICS_MODEL_KINDS:
        raise ValueError("mechanistic replay requires a dynamics-family model")
    destination = ROOT / "ablation_results" / config.run_id / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    state = {"step": 0, "last_summary": None, "previous_step": None, "before": None}
    original_evaluate = training.evaluate
    original_save = training.save_checkpoint
    original_optimizers = training.make_optimizers
    original_hashes = training.source_hashes

    def source_hashes(kind):
        return {
            **original_hashes(kind),
            str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__)),
        }

    def evaluate(model, data, train_config, device):
        metrics = original_evaluate(model, data, train_config, device)
        model.eval()
        collected = {}
        try:
            for batch in microbatches(
                data.validation[: train_config.val_characters],
                train_config.seq_len,
                train_config.mbs,
            ):
                trace = rollout_trace(model, training.to_cuda(batch, device))
                for key, values in trace.items():
                    collected.setdefault(key, []).append(values.reshape(-1))
        finally:
            model.train()
        arrays = {
            key: np.concatenate(values, axis=0) for key, values in collected.items()
        }
        summary = summarize_trace(
            arrays,
            metrics["source_bytes"],
            model.config.refresh_cost,
            model.config.rollout_horizon,
        )
        summary["step"] = state["step"]
        summary["canonical_bpb"] = metrics["bpb"]
        if abs(summary["teacher_bpb"] - metrics["bpb"]) > 0.0001:
            raise RuntimeError(
                "diagnostic teacher likelihood differs from canonical evaluation"
            )
        state["last_summary"] = summary
        with (destination / "validation.jsonl").open("a") as handle:
            handle.write(json.dumps(summary, allow_nan=False) + "\n")
        np.savez_compressed(
            destination / f"step_{state['step']:04d}.npz", allow_pickle=False, **arrays
        )
        metrics.update(
            {
                f"diagnostic_{key}": value
                for key, value in summary.items()
                if key not in {"step", "canonical_bpb"}
            }
        )
        return metrics

    def save(path, payload):
        step = payload["step"]
        if step != state["step"]:
            raise RuntimeError("optimizer observer and checkpoint step differ")
        spike = state["last_summary"]["rollout_excess_bpb"] >= 0.25
        history = destination / "checkpoints"
        if spike and path.exists() and state["previous_step"] is not None:
            history.mkdir(exist_ok=True)
            previous = history / f"step_{state['previous_step']:04d}.pt"
            if not previous.exists():
                os.link(path, previous)
        original_save(path, payload)
        if spike:
            history.mkdir(exist_ok=True)
            current = history / f"step_{step:04d}.pt"
            if not current.exists():
                os.link(path, current)
        state["previous_step"] = step

    def make_optimizers(model, train_config):
        optimizers = original_optimizers(model, train_config)
        groups = {
            "teacher_head": tuple(model.head.parameters()),
            "transition": tuple(model.transition.parameters()),
            "error_predictor": tuple(model.error_predictor.parameters()),
        }

        def before_step(optimizer, args, kwargs):
            if (state["step"] + 1) % train_config.log_every == 0:
                state["before"] = {
                    name: tuple(p.detach().clone() for p in params)
                    for name, params in groups.items()
                }

        def after_step(optimizer, args, kwargs):
            state["step"] += 1
            if state["before"] is None:
                return
            record = {"step": state["step"]}
            for name, params in groups.items():
                count = sum(p.numel() for p in params)
                previous = state["before"][name]
                values = (
                    torch.stack(
                        (
                            torch.stack(
                                [p.detach().float().square().sum() for p in params]
                            ).sum(),
                            torch.stack(
                                [p.grad.float().square().sum() for p in params]
                            ).sum(),
                            torch.stack(
                                [
                                    (p.detach().float() - old.float()).square().sum()
                                    for p, old in zip(params, previous, strict=True)
                                ]
                            ).sum(),
                        )
                    )
                    .sqrt()
                    .div(math.sqrt(count))
                    .cpu()
                    .tolist()
                )
                record[name] = dict(
                    zip(
                        ("parameter_rms", "gradient_rms", "update_rms"),
                        values,
                        strict=True,
                    )
                )
                record[name]["update_to_parameter_rms"] = (
                    values[2] / values[0] if values[0] else None
                )
            with (destination / "updates.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            state["before"] = None

        optimizers[0].register_step_pre_hook(before_step)
        optimizers[-1].register_step_post_hook(after_step)
        return optimizers

    training.source_hashes = source_hashes
    training.evaluate = evaluate
    training.save_checkpoint = save
    training.make_optimizers = make_optimizers
    training.train()


@torch.inference_mode()
def frozen_batches(output):
    data = PreparedData(ROOT / "data/datasets/native_bits_fineweb")
    cases = [
        (
            "nanomini_dynamics_batch524288_static2k",
            [1010, 1020, 1140, 1150, 1660, 1670, 1910, 1920],
        ),
        (
            "nanomini_character_softmax_batch524288_2k",
            [1010, 1020, 1140, 1150, 1660, 1670, 1910, 1920],
        ),
        ("nanomini_dynamics_norm_h2_refresh002_2k", [1050, 1060]),
        ("nanomini_prefix128_dynamics_control_2k", [1050, 1060]),
    ]
    non_ascii = np.asarray([ord(char) > 127 for char in data.alphabet])
    results = []
    for name, steps in cases:
        path = ROOT / "ablation_results" / name / "checkpoint.pt"
        checkpoint, digest = _load_pinned_checkpoint(path)
        model, device = training.load_model(checkpoint)
        model.requires_grad_(False)
        config = checkpoint["train_config"]
        for step in steps:
            offset = ((step - 1) * config["batch_characters"]) % data.train.size
            batch = cyclic_ids(data.train, offset, config["batch_characters"])
            total = 0.0
            for ids in microbatches(batch, config["seq_len"], config["mbs"]):
                _, stats = model(training.to_cuda(ids, device))
                total += float(stats["rate_nats"])
            counts = np.bincount(batch, minlength=len(data.alphabet))
            frequencies = counts[counts > 0] / batch.size
            byte_count = int(data.byte_lengths[batch].sum())
            results.append(
                {
                    "run": name,
                    "checkpoint_step": checkpoint["step"],
                    "checkpoint_sha256": digest,
                    "training_batch_step": step,
                    "offset": int(offset),
                    "characters": int(batch.size),
                    "bytes": byte_count,
                    "frozen_bpb": total / math.log(2) / byte_count,
                    "unigram_character_entropy_bits": float(
                        -(frequencies * np.log2(frequencies)).sum()
                    ),
                    "non_ascii_fraction": float(non_ascii[batch].mean()),
                    "text_prefix": "".join(
                        data.alphabet[int(value)] for value in batch[:160]
                    ),
                }
            )
        del model
        torch.cuda.empty_cache()
    report = {
        "semantics": "Fixed step2000 weights score the exact historical batches; no optimizer update or shuffling. Differences therefore isolate batch content, not parameter motion.",
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


@torch.inference_mode()
def bounded_probe(checkpoint_paths, output):
    """A frozen-parameter intervention, not a training or quality ablation."""
    data = PreparedData(ROOT / "data/datasets/native_bits_fineweb")
    records = []
    for path in checkpoint_paths:
        checkpoint, digest = _load_pinned_checkpoint(path)
        if training.checkpoint_model_kind(checkpoint) != "dynamics":
            raise ValueError(
                "bounded-probe requires an original residual dynamics checkpoint"
            )
        if checkpoint["provenance"]["data"] != data.metadata:
            raise ValueError("probe data differs from checkpoint provenance")
        config = training.TrainConfig(**checkpoint["train_config"])
        byte_count = int(
            data.byte_lengths[data.validation[: config.val_characters]].sum()
        )
        reference_teacher = None
        for kind in ("dynamics", "dynamics_bounded"):
            payload = checkpoint
            if kind == "dynamics_bounded":
                payload = {
                    **checkpoint,
                    "architecture": training.ARCHITECTURES[kind],
                    "model": {
                        **checkpoint["model"],
                        "transition.state_norm.gains": torch.ones(
                            checkpoint["model_config"]["model_dim"]
                        ),
                    },
                }
            model, device = training.load_model(payload)
            model.requires_grad_(False)
            collected = {}
            for batch in microbatches(
                data.validation[: config.val_characters], config.seq_len, config.mbs
            ):
                for key, values in rollout_trace(
                    model, training.to_cuda(batch, device)
                ).items():
                    collected.setdefault(key, []).append(values.reshape(-1))
            arrays = {key: np.concatenate(values) for key, values in collected.items()}
            summary = summarize_trace(
                arrays,
                byte_count,
                model.config.refresh_cost,
                model.config.rollout_horizon,
            )
            if reference_teacher is None:
                reference_teacher = arrays["teacher_nll"]
            difference = float(
                np.max(np.abs(arrays["teacher_nll"] - reference_teacher))
            )
            if difference > 0.0001:
                raise RuntimeError(
                    "state intervention changed the frozen teacher likelihood"
                )
            records.append(
                {
                    "checkpoint": str(path),
                    "checkpoint_sha256": digest,
                    "checkpoint_step": checkpoint["step"],
                    "variant": kind,
                    "teacher_nll_max_absolute_difference": difference,
                    "shared_parameters_changed": False,
                    "source_characters": config.val_characters,
                    "source_bytes": byte_count,
                    **summary,
                }
            )
            print(json.dumps(records[-1], allow_nan=False), flush=True)
            del model
    report = {
        "semantics": (
            "Frozen shared parameters and unchanged learned gate. The only intervention "
            "is recurrent output RMSNorm with gains initialized to one. No training, "
            "no forced refresh, no age cap, and no inference-quality improvement claim. "
            "Legacy rollout is diagnostic; public sampling still verifies against the teacher."
        ),
        "source_hashes": training.source_hashes("dynamics_bounded"),
        "observer_sha256": sha256_file(Path(__file__)),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["train-replay", "frozen-batches", "bounded-probe"],
        nargs="?",
        default="train-replay",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, action="append")
    arguments = parser.parse_args()
    if arguments.mode == "frozen-batches":
        if arguments.output is None:
            parser.error("frozen-batches requires --output")
        frozen_batches(arguments.output)
    elif arguments.mode == "bounded-probe":
        if arguments.output is None or not arguments.checkpoint:
            parser.error(
                "bounded-probe requires --output and at least one --checkpoint"
            )
        bounded_probe(arguments.checkpoint, arguments.output)
    else:
        train_replay()

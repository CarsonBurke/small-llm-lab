#!/usr/bin/env python3
"""Probe frozen Q-DG gradients on production pools, with zero training updates.

Execute through mlq on CUDA. The frozen trainer supplies its actual loader,
sampler, graph decode, verifier and sharded replay. Instrumentation disables
native optimizer/checkpoint/event writes; optional optimizer steps operate on
restored shadow state only. This is not a training-benefit experiment.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys


class ProbeFinished(Exception):
    """Stop the native collector after the requested diagnostic pools."""


def scalar_coefficient(advantage: float, surprisal: float, mode: str) -> float:
    """Pure-math reference for the three coefficients used by the probe."""
    z = advantage * surprisal
    gate = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
    if mode == "dg":
        return advantage * gate
    if mode == "base":
        return .5 * advantage
    if mode == "tilt":
        return (gate - .5) * advantage
    raise ValueError(mode)


def geometry_from_moments(dg2: float, base2: float, tilt2: float, dot: float, residual2: float, difference2: float, tilt_difference_dot: float) -> dict:
    if not all(math.isfinite(value) for value in (dg2, base2, tilt2, dot, residual2, difference2, tilt_difference_dot)):
        raise ValueError("non-finite gradient or step moments")
    norm_dg, norm_base, norm_tilt = map(lambda value: math.sqrt(max(value, 0)), (dg2, base2, tilt2))
    return {
        "dg_norm": norm_dg, "constant_half_norm": norm_base, "tilt_norm": norm_tilt,
        "tilt_over_constant_half": norm_tilt / norm_base if norm_base else None,
        "dg_constant_half_cosine": dot / (norm_dg * norm_base) if norm_dg and norm_base else None,
        "algebra_residual_norm": math.sqrt(max(residual2, 0)),
        "algebra_residual_over_dg": math.sqrt(max(residual2, 0)) / norm_dg if norm_dg else None,
        "actual_dg_minus_constant_half_norm": math.sqrt(max(difference2, 0)),
        "actual_difference_over_constant_half": math.sqrt(max(difference2, 0)) / norm_base if norm_base else None,
        "algebra_residual_over_direct_tilt": math.sqrt(max(residual2, 0)) / norm_tilt if norm_tilt else None,
        "algebra_residual_over_actual_difference": math.sqrt(max(residual2, 0) / difference2) if difference2 > 0 else None,
        "direct_tilt_actual_difference_cosine": tilt_difference_dot / (norm_tilt * math.sqrt(difference2)) if norm_tilt and difference2 > 0 else None,
    }


def checked_write(write_json, path: Path, value: dict) -> None:
    # The generic orchestration writer permits NaN for compatibility; a
    # diagnostic must never publish invalid numeric evidence as success.
    json.dumps(value, allow_nan=False)
    write_json(path, value)


def input_equality_summary(reference_tokens: int, branch_tokens: int, order_matches: bool, advantage_max_abs_difference: float | None, surprisal_max_abs_difference: float | None) -> dict:
    differences = (advantage_max_abs_difference, surprisal_max_abs_difference)
    if any(value is not None and not math.isfinite(value) for value in differences):
        raise ValueError("non-finite replay input difference")
    return {
        "reference_tokens": reference_tokens, "branch_tokens": branch_tokens,
        "shard_shapes_and_active_mask_order_equal": order_matches,
        "advantage_max_abs_difference": advantage_max_abs_difference,
        "surprisal_max_abs_difference": surprisal_max_abs_difference,
        "exact_equal": reference_tokens == branch_tokens and order_matches and all(value == 0 for value in differences),
    }


def compare_inputs(reference: list, branch: list, torch) -> dict:
    reference_tokens = sum(chunk[0].numel() for chunk in reference)
    branch_tokens = sum(chunk[0].numel() for chunk in branch)
    shapes_match = len(reference) == len(branch)
    order_matches = shapes_match
    maxima = torch.zeros(2, device="cuda")
    for old, new in zip(reference, branch):
        if old[0].shape != new[0].shape or old[2].shape != new[2].shape:
            shapes_match = False
            order_matches = False
            continue
        order_matches &= torch.equal(old[2], new[2])
        if old[0].numel():
            maxima = torch.maximum(maxima, torch.stack(((old[0] - new[0]).abs().max(), (old[1] - new[1]).abs().max())))
    differences = maxima.cpu().tolist() if shapes_match else (None, None)
    return input_equality_summary(reference_tokens, branch_tokens, order_matches, *differences)


def driver_argv(initial: list[str], checkpoint: Path, output: Path) -> list[str]:
    result = []
    overrides = {"--output", "--actor-init", "--resume", "--max-train-hours", "--bpb-every", "--bench-every", "--aime-every", "--post-update-kl-every", "--snapshot-every", "--rollout-sample-every"}
    index = 0
    while index < len(initial):
        if initial[index] in overrides:
            index += 2
        elif initial[index] == "--fresh-critic":
            index += 1
        else:
            result.append(initial[index])
            index += 1
    return [*result, "--resume", str(checkpoint), "--output", str(output),
            "--bpb-every", "0", "--bench-every", "0", "--aime-every", "0",
            "--post-update-kl-every", "0", "--snapshot-every", "0", "--rollout-sample-every", "0"]


class NativeWritesDisabled:
    def __init__(self):
        self.suppressed_steps = 0
        self.suppressed_checkpoints = 0
        self.suppressed_events = 0

    def step(self, *args, **kwargs):
        self.suppressed_steps += 1

    def checkpoint(self, *args, **kwargs):
        self.suppressed_checkpoints += 1

    def logger(self, *args, **kwargs):
        owner = self

        class Logger:
            def log(self, *args, **kwargs):
                owner.suppressed_events += 1

            def purge_after(self, *args, **kwargs):
                return 0

        return Logger()

    def tensorboard(self, *args, **kwargs):
        owner = self

        class Writer:
            def add_scalar(self, *args, **kwargs):
                owner.suppressed_events += 1

            def close(self):
                pass

        return Writer()


@contextmanager
def instrument(module, replacements: dict):
    originals = {name: getattr(module, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


def cpu_state(value, torch):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item, torch) for item in value)
    return copy.deepcopy(value)


def tensor_geometry(dg: dict, base: dict, tilt: dict, groups: dict, torch) -> dict:
    moments = {group: [torch.zeros(7, device="cuda", dtype=torch.float64)] for group in ("all", *sorted(set(groups.values())))}
    for name in dg:
        a, b, c = dg[name].float(), base[name].float(), tilt[name].float()
        residual = a - b - c
        difference = a - b
        values = torch.stack(tuple(torch.sum(value, dtype=torch.float64) for value in
                                   (a * a, b * b, c * c, a * b, residual * residual, difference * difference, c * difference)))
        moments["all"][0] += values
        moments[groups[name]][0] += values
    result = {}
    for group, cells in moments.items():
        values = cells[0].cpu().tolist()
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite gradient or step geometry")
        result[group] = geometry_from_moments(*values)
    return result


def quantiles(values, weights, torch) -> dict:
    levels = torch.tensor([0, .01, .1, .5, .9, .99, 1], device=values.device)
    ordered, indices = values.sort()
    if weights is None:
        result = torch.quantile(values, levels).cpu().tolist()
    else:
        cumulative = weights[indices].cumsum(0)
        if float(cumulative[-1]) <= 0:
            return {}
        positions = torch.searchsorted(cumulative, levels * cumulative[-1]).clamp_max(values.numel() - 1)
        result = ordered[positions].cpu().tolist()
    return dict(zip(("0", ".01", ".1", ".5", ".9", ".99", "1"), result, strict=True))


def coefficient_statistics(chunks: list, torch) -> dict:
    advantage = torch.cat([chunk[0] for chunk in chunks])
    surprisal = torch.cat([chunk[1] for chunk in chunks])
    if not torch.isfinite(advantage).all() or not torch.isfinite(surprisal).all():
        raise ValueError("non-finite coefficient statistics")
    z = advantage * surprisal
    gate = z.sigmoid()
    dg, base, tilt = advantage * gate, .5 * advantage, advantage * (gate - .5)
    magnitude = advantage.abs()
    energy = advantage.square()
    magnitude_sum, energy_sum = float(magnitude.sum()), float(energy.sum())
    return {
        "tokens": advantage.numel(), "eta": 1.0,
        "gate_mean": float(gate.mean()), "gate_quantiles": quantiles(gate, None, torch),
        "gate_abs_advantage_weighted_mean": float((gate * magnitude).sum()) / magnitude_sum if magnitude_sum else None,
        "gate_abs_advantage_weighted_quantiles": quantiles(gate, magnitude, torch),
        "gate_advantage_energy_weighted_mean": float((gate * energy).sum()) / energy_sum if energy_sum else None,
        "delight_quantiles": quantiles(z, None, torch),
        "advantage_quantiles": quantiles(advantage, None, torch),
        "dg_coefficient_quantiles": quantiles(dg, None, torch),
        "constant_half_coefficient_quantiles": quantiles(base, None, torch),
        "tilt_coefficient_quantiles": quantiles(tilt, None, torch),
        "positive_gate_mean": float(gate[advantage > 0].mean()) if (advantage > 0).any() else None,
        "negative_gate_mean": float(gate[advantage < 0].mean()) if (advantage < 0).any() else None,
        "gate_below_point1_fraction": float((gate < .1).float().mean()),
        "gate_above_point9_fraction": float((gate > .9).float().mean()),
        "coefficient_tilt_over_constant_half_energy": float(tilt.square().sum().sqrt() / base.square().sum().sqrt()) if energy_sum else None,
        "dg_surrogate_mean": float((dg * surprisal).mean()),
        "constant_half_surrogate_mean": float((base * surprisal).mean()),
        "tilt_surrogate_mean": float((tilt * surprisal).mean()),
        "coefficient_algebra_max_abs": float((dg - base - tilt).abs().max()),
        "interpretation": "Coefficient statistics exclude score-vector cancellation and optimizer preconditioning. Advantage sign is not verifier outcome.",
    }


class QGateProbe:
    def __init__(self, module, torch, plan: dict, output: Path, write_json, *, shadow_steps: bool):
        self.module, self.torch, self.plan = module, torch, plan
        self.output, self.write_json, self.shadow_steps = output, write_json, shadow_steps
        self.original_update = module.update_minibatch
        self.original_loss = module.delightful_policy_loss
        self.original_step = module.step_optimizers
        self.rows = []
        self.report = {
            "schema": "q_dg_gate_gradient_probe/v1", "plan": plan,
            "persistent_training_updates": 0, "pools": self.rows,
            "production_gradient_clipping": "none in the verified frozen trainer",
            "interpretation": "Same fixed actor/critic and optimizer state across diagnostic pools. Gradient and shadow-step differences do not establish training benefit.",
        }

    def __call__(self, wrapper, critic, batch, optimizers, **kwargs):
        torch, module = self.torch, self.module
        if kwargs.get("value_only") or not kwargs.get("fused_behavior_statistics") or not critic.values_actions:
            raise ValueError("probe requires the production fused token-Q actor path")
        if kwargs.get("actor_step", True) or kwargs.get("critic_step", True):
            raise ValueError("native update attempted an optimizer step inside replay")
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), random.getstate())
        models = (cpu_state(wrapper.state_dict(), torch), cpu_state(critic.state_dict(), torch))
        optimizer_states = {name: cpu_state(opt.state_dict(), torch) for name, opt in optimizers.items()}
        parameters = {name: param for name, param in wrapper.named_parameters() if param.requires_grad}
        parameter_names = {id(param): name for name, param in parameters.items()}
        groups = {}
        for role, optimizer in optimizers.items():
            if role.startswith("actor"):
                for index, group in enumerate(optimizer.param_groups):
                    for parameter in group["params"]:
                        groups[parameter_names[id(parameter)]] = f"{role}/group_{index}"

        def restore():
            wrapper.load_state_dict(models[0], strict=True)
            critic.load_state_dict(models[1], strict=True)
            for name, optimizer in optimizers.items():
                # Non-capturable Adam step scalars can retain their CPU tensor
                # identities during load_state_dict and mutate on step().
                optimizer.load_state_dict(cpu_state(optimizer_states[name], torch))
                optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state_all(rng[1])
            random.setstate(rng[2])

        gradients, present_gradients, mode_inputs, dg_metrics = {}, {}, {}, None
        try:
            for mode in ("dg", "base", "tilt"):
                restore()
                mode_inputs[mode] = []

                def loss(logprobs, advantages, mask, temperature=1.0, denominator=None):
                    native_loss, diagnostics = self.original_loss(logprobs, advantages, mask, temperature, denominator)
                    if temperature != 1:
                        raise ValueError("unexpected DG gate temperature")
                    advantage, surprisal = advantages.detach(), -logprobs.detach()
                    active = mask.bool()
                    mode_inputs[mode].append((advantage[active].float().clone(), surprisal[active].float().clone(), active.clone()))
                    if mode == "dg":
                        return native_loss, diagnostics
                    gate = (advantage * surprisal).sigmoid()
                    coefficient = .5 * advantage if mode == "base" else (gate - .5) * advantage
                    denom = (mask.sum() if denominator is None else denominator.to(mask.device)).clamp_min(1)
                    return -(coefficient.detach() * logprobs * mask).sum() / denom, diagnostics

                with instrument(module, {"delightful_policy_loss": loss}):
                    metrics = self.original_update(wrapper, critic, batch, optimizers, **kwargs)
                if mode == "dg":
                    dg_metrics = metrics
                present_gradients[mode] = {name for name, parameter in parameters.items() if parameter.grad is not None}
                gradients[mode] = {name: parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros_like(parameter) for name, parameter in parameters.items()}
            input_comparability = {mode: compare_inputs(mode_inputs["dg"], mode_inputs[mode], torch) for mode in ("base", "tilt")}
            if not all(cell["exact_equal"] for cell in input_comparability.values()):
                self.report["input_comparability_failure"] = {
                    "pool": len(self.rows) + 1, "comparisons": input_comparability,
                    "interpretation": "Replay inputs differed; gradient differences cannot be attributed solely to the gate.",
                }
                self.write_json(self.output / "metrics.json", self.report)
                raise ValueError("DG/base/tilt replay inputs are not exactly equal")
            record = {
                "pool": len(self.rows) + 1,
                "sampler_cursor_start": self.plan["sampler_cursor"] + len(self.rows) * self.plan["production_pool_prompts"],
                "trajectories": batch.kind.size(0), "tokens": int(batch.emit_mask.sum()),
                "reward_mean": float(batch.reward_scalar.mean()),
                "production_would_freeze_actor": self.plan["zero_reward_actor_freeze"] and dg_metrics["reward_max"] == dg_metrics["reward_min"],
                "replay_input_comparability": input_comparability,
                "coefficient_statistics": coefficient_statistics(mode_inputs["dg"], torch),
                "raw_gradient_geometry": tensor_geometry(gradients["dg"], gradients["base"], gradients["tilt"], groups, torch),
                "parameter_groups": groups,
            }
            if self.shadow_steps:
                deltas = {}
                for mode in ("dg", "base"):
                    restore()
                    if not record["production_would_freeze_actor"]:
                        for name, parameter in parameters.items():
                            parameter.grad = gradients[mode][name].clone() if name in present_gradients[mode] else None
                        self.original_step(optimizers, "actor")
                    deltas[mode] = {name: parameter.detach() - models[0][name].to(parameter.device) for name, parameter in parameters.items()}
                difference = {name: deltas["dg"][name] - deltas["base"][name] for name in parameters}
                record["shadow_step_geometry"] = tensor_geometry(deltas["dg"], deltas["base"], difference, groups, torch)
                changed = torch.zeros(2, device="cuda", dtype=torch.float64)
                for name, parameter in parameters.items():
                    initial = models[0][name].to(parameter.device)
                    changed[0] += difference[name].count_nonzero()
                    changed[1] += ((initial + deltas["dg"][name]).to(torch.bfloat16) != (initial + deltas["base"][name]).to(torch.bfloat16)).count_nonzero()
                count = sum(parameter.numel() for parameter in parameters.values())
                fractions = (changed / count).cpu().tolist()
                record["shadow_step_geometry"]["all"]["master_parameter_difference_fraction"] = fractions[0]
                record["shadow_step_geometry"]["all"]["bf16_weight_projection_difference_fraction"] = fractions[1]
                record["shadow_weight_projection_note"] = "Casting restored master weights plus each delta to bf16 is a projection proxy, not a measured deployed-policy change."
                record["shadow_step_note"] = "tilt_norm denotes delta_DG−delta_constant_half here; every branch restores identical saved optimizer moments. No critic step."
            self.rows.append(record)
            self.write_json(self.output / "metrics.json", self.report)
            print(json.dumps({"diagnostic_pool": record["pool"], "raw_gradient": record["raw_gradient_geometry"]["all"]}), flush=True)
        finally:
            restore()
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state_all(rng[1])
            random.setstate(rng[2])
        if len(self.rows) >= self.plan["pools"]:
            raise ProbeFinished()
        return dg_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shadow-steps", action=argparse.BooleanOptionalAction, default=True)
    cli = parser.parse_args()
    # Import CPU-only orchestration helpers from their explicit live file.
    path = Path(__file__).with_name("run_dg_qtarget_study.py")
    spec = importlib.util.spec_from_file_location("qgate_orchestration", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    study, plan = json.loads(cli.study.read_text()), json.loads(cli.plan.read_text())
    helper.verify_source(study)
    if plan["source_root"] != study["source_root"] or plan["persistent_training_updates"] != 0 or plan["pools"] < 1:
        raise ValueError("invalid diagnostic plan")
    checkpoint = Path(plan["checkpoint"])
    if helper.sha256(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("archived diagnostic checkpoint changed")
    if cli.output.exists():
        raise FileExistsError("diagnostic output must be new")
    module = helper.frozen_trainer(study)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires CUDA; CPU model execution is forbidden")
    arm = study["arms"]["mc"]
    metadata = helper.checkpoint_metadata(study, arm, checkpoint)
    if metadata["step"] != plan["joint_step"] or metadata["value_warmup_step"] != plan["critic_warmup_step"]:
        raise ValueError("plan checkpoint counters changed")
    plan["sampler_cursor"] = metadata["sampler_cursor"]
    parsed = module.build_arg_parser().parse_args(helper.trainer_arguments(arm["argv"]))
    module.validate_args(module.build_arg_parser(), parsed)
    plan["zero_reward_actor_freeze"] = parsed.zero_reward_actor_freeze
    if (parsed.prompts_per_rollout, parsed.samples_per_prompt) != (plan["production_pool_prompts"], plan["samples_per_prompt"]):
        raise ValueError("diagnostic plan reduced the production pool")
    source = Path(module.__file__).read_text()
    if any(isinstance(node, ast.Attribute) and node.attr in {"clip_grad_norm_", "clip_grad_value_"} for node in ast.walk(ast.parse(source))):
        raise ValueError("gradient clipping requires an explicit pre-clip interception")
    cli.output.mkdir(parents=True)
    if helper.sha256(checkpoint.parent / "manifest.json") != plan["manifest_sha256"]:
        raise ValueError("archived input manifest changed")
    archived_manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
    reproduction = {
        "checkpoint_sha256_before": plan["checkpoint_sha256"],
        "archived_manifest_sha256": plan["manifest_sha256"],
        "frozen_source_root": study["source_root"],
        "frozen_source_sha256": study["source_sha256"],
        "underlying_source_provenance": archived_manifest["source_provenance"],
        "diagnostic_script_sha256": helper.sha256(Path(__file__)),
        "orchestration_helper_sha256": helper.sha256(path),
    }
    saved_source = cli.output / "source"
    saved_source.mkdir()
    for source_file in (Path(__file__), path):
        (saved_source / source_file.name).write_bytes(source_file.read_bytes())
    helper.atomic_save(cli.output / "diagnostic_manifest.json", {**plan, "driver": "frozen trainer with native writes/optimizer steps disabled", "script_sha256": helper.sha256(Path(__file__))})
    disabled = NativeWritesDisabled()
    write_json = lambda path, value: checked_write(helper.atomic_save, path, value)
    probe = QGateProbe(module, torch, plan, cli.output, write_json, shadow_steps=cli.shadow_steps)
    probe.report["reproducibility"] = reproduction
    original_argv = sys.argv
    driver_output = cli.output / "_driver"
    argv = driver_argv(arm["argv"], checkpoint, driver_output)
    driver_parser = module.build_arg_parser()
    driver_args = driver_parser.parse_args(helper.trainer_arguments(argv))
    module.validate_args(driver_parser, driver_args)
    helper.resolve_runtime_budget(module, driver_args, archived_manifest)
    module.validate_resume_arg_contract(archived_manifest["args"], driver_args)
    replacements = {"update_minibatch": probe, "step_optimizers": disabled.step,
                    "save_checkpoint": disabled.checkpoint, "JsonlLogger": disabled.logger,
                    "SummaryWriter": disabled.tensorboard}
    try:
        sys.argv = [str(module.__file__), *helper.trainer_arguments(argv)]
        with instrument(module, replacements):
            module.main()
        raise RuntimeError("native driver exited before collecting the diagnostic pools")
    except ProbeFinished:
        probe.report["native_writes_suppressed"] = vars(disabled)
        probe.report["checkpoint_sha256_after"] = helper.sha256(checkpoint)
        if probe.report["checkpoint_sha256_after"] != plan["checkpoint_sha256"]:
            raise RuntimeError("diagnostic modified its input checkpoint")
        write_json(cli.output / "metrics.json", probe.report)
        helper.atomic_save(driver_output / "diagnostic_only.json", {"persistent_training_updates": 0, "not_a_training_run": True})
        return 0
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())

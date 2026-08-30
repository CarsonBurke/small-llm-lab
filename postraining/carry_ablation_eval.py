"""Ablate raw stochastic thought content on a trained latent-VAPO policy.

The full arm reproduces the checkpoint policy, ``no_content`` disables the
combiner's learned residual over raw thought inputs, and ``token_only`` uses
``pin_emit=True`` to bypass gate, noise, and thought slots entirely.

Per-prompt correct counts from ``evaluate_latent_math`` give paired deltas;
significance comes from a sign-flip permutation test on the per-prompt mean
delta plus a bootstrap CI, both resampled over prompts (the independent
sampling unit — samples within a prompt are exchangeable but not
independent evidence about the panel).

Separately, mechanistic probes use the production replay path:
``refresh_old_statistics`` consumes the stored raw fp32 thought actions, then
the probe zeros ``batch.thoughts`` and recomputes actor/critic statistics. The
contrast measures dependence on action content without any redraw.
The probes deliberately roll at the TRAINING operating point (the checkpoint's
saved temperature/top-p) because they measure what PPO and GAE consumed
during training; the behavioral arms use the evaluation protocol
(temperature 1.0, top-p 0.7) like the trainer's bench/AIME evals. Both
operating points are recorded in ``results.json``.

The behavioral results and paired statistics are persisted BEFORE the
probes run, so a probe failure never discards the expensive arm sweeps.
Note the three pairwise comparisons are algebraically dependent
(full-vs-token_only = full-vs-no_content + no_content-vs-token_only); they
are three views of two degrees of freedom, not independent tests.

Reads the rolling training checkpoint (saved atomically via ``os.replace``),
so the checkpoint read is safe while training runs. Scheduling it
concurrently with the training job halves the card and was an explicit
operator decision for this analysis; the queue default remains
``--max-parallel-runs 1``:

    mlq submit --name carry_ablation --cwd "$PWD" --max-parallel-runs 2 -- \
        .venv/bin/python -m postraining.carry_ablation_eval \
        --wrapper-checkpoint postraining/runs/k3_latent_10h/latent_vapo_checkpoint.pt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

import train_gpt as baseline  # noqa: F401  (import order: patches must load first)
from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters
from postraining.core import (
    answer_style,
    deterministic_math_subset,
    encode_prompt,
    load_posttraining_tokenizer,
    load_unique_math_rows,
    modal_answer_baseline,
)
from postraining.hl_gauss import anchored_unit_geometry
from postraining.latent_eval import evaluate_latent_math
from postraining.latent_rollout import (
    THOUGHT_SLOT,
    refresh_old_statistics,
    rollout_continuations,
    trim_stream,
)
from postraining.latent_thought import (
    LatentThoughtModel,
    combiner_init_kwargs_from_checkpoint,
    rollout_policy_schema_for_mode,
    validate_renderer_checkpoint,
)
from postraining.math_prompt import require_answer_fence_prompt_schema
from postraining.model_io import fresh_trunk, load_model
from postraining.train_latent_vapo import (
    rewrite_prompts_for_answer_fence,
    score_math_rollout,
)
from postraining.train_vapo import prompt_text
from postraining.value_model import SeparateCritic

CARRY_ABLATION_SCHEMA = "carry_ablation_eval/v1"

ARMS = ("full", "no_content", "token_only")


@contextlib.contextmanager
def zeroed_carry(wrapper: LatentThoughtModel):
    """Temporarily zero the actor combiner's carry matrix, restoring exactly."""
    weight = wrapper.combiner.carry.weight
    saved = weight.detach().clone()
    with torch.no_grad():
        weight.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            weight.copy_(saved)


def paired_prompt_stats(
    counts_a: list[int],
    counts_b: list[int],
    samples: int,
    seed: int,
    resamples: int = 20000,
) -> dict[str, float | int]:
    """Paired per-prompt comparison of two same-panel evaluations.

    ``counts_*`` are per-prompt correct counts over ``samples`` attempts, in
    the same (original dataset) order for both arms. The prompt is the
    resampling unit for both the bootstrap CI and the sign-flip permutation
    test: within-prompt samples share the prompt's difficulty, so treating
    all attempts as independent would overstate significance.
    """
    if len(counts_a) != len(counts_b):
        raise ValueError("paired stats require identical prompt panels")
    if samples < 1:
        raise ValueError("samples must be positive")
    deltas = torch.tensor(
        [(a - b) / samples for a, b in zip(counts_a, counts_b, strict=True)],
        dtype=torch.float64,
    )
    prompts = deltas.numel()
    mean_delta = float(deltas.mean())
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        0, prompts, (resamples, prompts), generator=generator
    )
    boot_means = deltas[indices].mean(dim=1)
    lower, upper = (
        float(torch.quantile(boot_means, q)) for q in (0.025, 0.975)
    )
    signs = (
        torch.randint(
            0, 2, (resamples, prompts), generator=generator,
            dtype=torch.float64,
        )
        * 2.0
        - 1.0
    )
    permuted = (signs * deltas).mean(dim=1)
    extreme = int((permuted.abs() >= abs(mean_delta) - 1e-12).sum())
    return {
        "prompts": prompts,
        "samples_per_prompt": samples,
        "accuracy_a": float(sum(counts_a)) / (len(counts_a) * samples),
        "accuracy_b": float(sum(counts_b)) / (len(counts_b) * samples),
        "mean_delta": mean_delta,
        "bootstrap_ci_low": lower,
        "bootstrap_ci_high": upper,
        "permutation_p": (extreme + 1) / (resamples + 1),
        "prompts_improved": int((deltas > 0).sum()),
        "prompts_regressed": int((deltas < 0).sum()),
        "prompts_tied": int((deltas == 0).sum()),
    }


def run_arm(
    arm: str,
    wrapper: LatentThoughtModel,
    tokenizer,
    rows: list[dict],
    samples: int,
    max_new_tokens: int,
    max_stream_steps: int,
    seed: int,
    device: torch.device,
    prompt_tokens: int,
    answer_style_override: str | None,
    capture_problems: int,
    capture_samples: int,
    answer_fence_ids: tuple[int, int] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """One arm's full-panel evaluation; identical rows/seed across arms."""
    captured: list[dict[str, object]] = []
    context = (
        zeroed_carry(wrapper)
        if arm == "no_content"
        else contextlib.nullcontext()
    )
    with context:
        metrics = evaluate_latent_math(
            wrapper,
            tokenizer,
            rows,
            samples,
            max_new_tokens,
            max_stream_steps,
            samples,
            seed,
            device,
            prompt_tokens,
            captured_attempts=captured,
            answer_style_override=answer_style_override,
            capture_problem_count=min(capture_problems, len(rows)),
            capture_samples_per_problem=min(capture_samples, samples),
            pin_emit=arm == "token_only",
            answer_fence_ids=answer_fence_ids,
        )
    return metrics, captured


def hidden_probes(
    wrapper: LatentThoughtModel,
    critic: SeparateCritic,
    tokenizer,
    rows: list[dict],
    samples: int,
    saved_args: dict,
    seed: int,
    device: torch.device,
    stop_ids: tuple[int, ...],
) -> dict[str, object]:
    """Measure actor/critic dependence on stored raw thought actions.

    Rollout samples each fp32 raw action once. Refresh first uses those exact
    vectors, then zeroes the stored actions and refreshes again; no path redraws
    noise.
    """
    max_new_tokens = saved_args["resolved_train_max_new_tokens"]
    max_stream_steps = saved_args["resolved_train_max_stream_steps"]
    prompt_budget = saved_args["prompt_tokens"]
    nearby_reward_max = saved_args.get("nearby_reward_max", 0.1)
    # These numbers describe the training process, so the reward must be
    # the training reward: a gate-trained run scored ungated would show
    # value error on every bare-guess row.
    think_fence_ids = (
        (tokenizer.think_open_id, tokenizer.think_close_id)
        if saved_args.get("think_tokens")
        else None
    )
    answer_fence_ids = (
        (tokenizer.answer_open_id, tokenizer.answer_close_id)
        if saved_args.get("answer_fence")
        else None
    )
    generator = torch.Generator(device=device).manual_seed(seed)

    refresh_budgets = {
        "max_trajectories": saved_args.get("replay_max_trajectories", 32),
        "attention_budget": saved_args.get(
            "replay_attention_budget", 4 * 1024 * 1024
        ),
        "bucket_multiple": saved_args.get("replay_bucket", 1),
        "slot_budget": saved_args.get("replay_slot_budget"),
    }
    logprob_deltas: list[torch.Tensor] = []
    value_deltas: list[torch.Tensor] = []
    trajectory_rows: list[dict[str, float]] = []
    # Slot-weighted accumulators: a 500-token rollout should count 500 slots
    # toward the rms ratio, not one prompt's worth.
    injected_sq_sum = base_sq_sum = hidden_sq_sum = 0.0
    carried_elements = 0

    for row in rows:
        encoded = encode_prompt(tokenizer, prompt_text(row), prompt_budget)
        prompt_ids = torch.tensor(encoded, dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16
        ):
            batch = trim_stream(
                rollout_continuations(
                    wrapper,
                    prompt_ids[None],
                    max_new_tokens,
                    max_stream_steps,
                    saved_args.get("temperature", 1.0),
                    saved_args.get("top_p", 1.0),
                    generator=generator,
                    stop_ids=stop_ids or None,
                    cache_dtype=torch.bfloat16,
                    prompt_repeats=samples,
                )
            )
        score_math_rollout(
            batch,
            row["reward_model"]["ground_truth"],
            tokenizer,
            stop_ids,
            answer_style(row),
            nearby_reward_max,
            think_fence_ids=think_fence_ids,
            min_think_tokens=int(saved_args.get("think_min_tokens", 1)),
            answer_fence_ids=answer_fence_ids,
        )
        action = batch.action_mask.bool()
        with torch.no_grad():
            carried = batch.kind == THOUGHT_SLOT
            if bool(carried.any()):
                hiddens = batch.thoughts[carried].float()
                base = hiddens
                injected = F.linear(
                    hiddens, wrapper.combiner.carry.weight.float()
                )
                injected_sq_sum += float(injected.square().sum())
                base_sq_sum += float(base.square().sum())
                hidden_sq_sum += float(hiddens.square().sum())
                carried_elements += hiddens.numel()

        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16
        ):
            refresh_old_statistics(wrapper, critic, batch, **refresh_budgets)
            logp_full = batch.old_token_logprobs.clone()
            values_full = batch.old_values.clone()
            batch.thoughts.zero_()
            refresh_old_statistics(wrapper, critic, batch, **refresh_budgets)

        logp_delta = (logp_full - batch.old_token_logprobs)[action].float()
        value_delta = (values_full - batch.old_values)[action].float()
        logprob_deltas.append(logp_delta.cpu())
        value_deltas.append(value_delta.cpu())
        reward = batch.reward_scalar.float().cpu()
        per_row_logp = torch.where(
            action, logp_full - batch.old_token_logprobs, 0.0
        ).sum(dim=1).float().cpu()
        per_row_value = torch.where(
            action, values_full - batch.old_values, 0.0
        )
        action_counts = action.sum(dim=1).clamp_min(1)
        per_row_value_mean = (
            per_row_value.sum(dim=1) / action_counts
        ).float().cpu()
        for sample_index in range(reward.numel()):
            trajectory_rows.append(
                {
                    "reward": float(reward[sample_index]),
                    "sequence_logprob_delta": float(
                        per_row_logp[sample_index]
                    ),
                    "value_delta_mean": float(
                        per_row_value_mean[sample_index]
                    ),
                }
            )

    logp = torch.cat(logprob_deltas)
    values = torch.cat(value_deltas)

    def summarize(deltas: torch.Tensor, prefix: str) -> dict[str, float]:
        magnitude = deltas.abs()
        return {
            f"{prefix}_mean": float(deltas.mean()),
            f"{prefix}_abs_mean": float(magnitude.mean()),
            f"{prefix}_abs_p95": float(torch.quantile(magnitude, 0.95)),
            f"{prefix}_abs_max": float(magnitude.max()),
        }

    rewards = torch.tensor([r["reward"] for r in trajectory_rows])
    value_means = torch.tensor(
        [r["value_delta_mean"] for r in trajectory_rows]
    )
    if rewards.numel() > 2 and rewards.std() > 0 and value_means.std() > 0:
        stacked = torch.stack([rewards, value_means])
        reward_value_corr = float(torch.corrcoef(stacked)[0, 1])
    else:
        # None, not NaN: json.dumps emits bare NaN, which strict JSON
        # consumers reject.
        reward_value_corr = None
    return {
        "action_slots": int(logp.numel()),
        "trajectories": len(trajectory_rows),
        **summarize(logp, "token_logprob_delta"),
        **summarize(values, "critic_value_delta"),
        "reward_vs_value_delta_corr": reward_value_corr,
        "injection_to_embedding_rms_ratio": (
            math.sqrt(injected_sq_sum / max(base_sq_sum, 1e-24))
            if carried_elements
            else 0.0
        ),
        "carried_hidden_rms": (
            math.sqrt(hidden_sq_sum / carried_elements)
            if carried_elements
            else 0.0
        ),
        "carried_slots_elements": carried_elements,
        "probe_temperature": saved_args.get("temperature", 1.0),
        "probe_top_p": saved_args.get("top_p", 1.0),
        "trajectory_rows": trajectory_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrapper-checkpoint",
        default="postraining/runs/k3_latent_10h/latent_vapo_checkpoint.pt",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--math-data", default="postraining/data/dapo-math-17k.parquet"
    )
    parser.add_argument(
        "--aime-data", default="postraining/data/aime-2024.parquet"
    )
    parser.add_argument("--math-prompts", type=int, default=384)
    parser.add_argument("--math-samples", type=int, default=8)
    parser.add_argument("--aime-samples", type=int, default=16)
    parser.add_argument("--probe-prompts", type=int, default=32)
    parser.add_argument("--probe-samples", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    for name in (
        "math_prompts",
        "math_samples",
        "aime_samples",
        "probe_prompts",
        "probe_samples",
        "resamples",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    wrapper_path = Path(args.wrapper_checkpoint)
    manifest_path = wrapper_path.parent / "manifest.json"
    if args.checkpoint is None:
        if not manifest_path.exists():
            parser.error(
                f"cannot resolve the base checkpoint: {manifest_path} not "
                "found; pass --checkpoint explicitly"
            )
        manifest_payload = json.loads(manifest_path.read_text())
        args.checkpoint = manifest_payload["base"]["checkpoint"]

    device = torch.device("cuda")
    backbone = load_model(args.checkpoint, device)
    backbone.eval()
    payload = torch.load(wrapper_path, map_location="cpu", weights_only=False)
    saved_args = payload["args"]
    require_answer_fence_prompt_schema(
        payload,
        answer_fence=bool(saved_args.get("answer_fence")),
        source=str(wrapper_path),
    )
    reasoning_mode = saved_args.get("reasoning_mode", "latent")
    if reasoning_mode != "latent":
        raise SystemExit(
            "carry ablation only makes sense for a latent-mode checkpoint; "
            f"got reasoning mode {reasoning_mode!r}"
        )
    wrapper = LatentThoughtModel(
        backbone, **combiner_init_kwargs_from_checkpoint(payload)
    ).to(device)
    validate_renderer_checkpoint(
        payload,
        str(wrapper_path),
        expected_rollout_policy_schema=rollout_policy_schema_for_mode(
            reasoning_mode
        ),
    )
    wrapper.load_state_dict(payload["model"], strict=True)
    wrapper.eval()
    step = payload.get("step")
    carry_weight = wrapper.combiner.carry.weight
    carry_rms = float(carry_weight.detach().square().mean().sqrt())
    print(
        f"policy: {wrapper_path} (step {step}), "
        f"carry weight rms {carry_rms:.3e}"
    )

    if saved_args.get("value_anchored_support", False):
        value_num_bins, value_v_min, value_v_max = anchored_unit_geometry(
            saved_args.get("value_bins", 101),
            saved_args.get("value_margin_bins", 4),
        )
    else:
        value_num_bins = saved_args.get("value_bins", 101)
        value_v_min, value_v_max = 0.0, 1.0
    critic = SeparateCritic(
        fresh_trunk(backbone, device),
        num_bins=value_num_bins,
        sigma_ratio=saved_args.get("value_sigma_ratio", 2.0),
        v_min=value_v_min,
        v_max=value_v_max,
        prior_value=saved_args.get("value_prior", 0.05),
        **{
            key: value
            for key, value in combiner_init_kwargs_from_checkpoint(
                payload
            ).items()
            if key in {"mlp_hidden", "num_blocks"}
        },
    ).to(device)
    critic.load_state_dict(payload["critic"], strict=True)
    critic.eval()
    critic_carry_rms = float(
        critic.combiner.carry.weight.detach().square().mean().sqrt()
    )

    tokenizer = load_posttraining_tokenizer(
        backbone.architecture,
        FreshHyperparameters.tokenizer_path,
        think_tokens=bool(saved_args.get("think_tokens")),
        answer_tokens=bool(saved_args.get("answer_fence")),
        tokenizer_provenance=backbone.model_config.get("tokenizer_provenance"),
    )
    stop_ids = tuple(
        dict.fromkeys(
            t for t in (tokenizer.eos_id(), tokenizer.bos_id()) if t >= 0
        )
    )

    all_math_rows = load_unique_math_rows(args.math_data)
    aime_rows = load_unique_math_rows(args.aime_data)
    if saved_args.get("answer_fence"):
        # The run rolled out under fence-contract prompts; comparing arms
        # under the unrewritten Answer: instruction would measure an
        # off-distribution policy. Rewrite before subsetting, exactly as
        # the trainer does.
        all_math_rows = rewrite_prompts_for_answer_fence(all_math_rows)
        aime_rows = rewrite_prompts_for_answer_fence(aime_rows)
    math_rows = deterministic_math_subset(all_math_rows, args.math_prompts)
    probe_rows = random.Random(args.seed + 1).sample(
        all_math_rows, min(args.probe_prompts, len(all_math_rows))
    )

    panels = {
        "math": {
            "rows": math_rows,
            "samples": args.math_samples,
            "max_new_tokens": saved_args["resolved_train_max_new_tokens"],
            "max_stream_steps": saved_args["resolved_train_max_stream_steps"],
            "style_override": None,
        },
        "aime": {
            "rows": aime_rows,
            "samples": args.aime_samples,
            "max_new_tokens": saved_args["resolved_aime_max_new_tokens"],
            "max_stream_steps": saved_args["resolved_aime_max_stream_steps"],
            "style_override": "aime",
        },
    }

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else wrapper_path.parent / "carry_ablation" / f"step_{step:06d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_metrics: dict[str, dict[str, dict[str, object]]] = {}
    transcripts: dict[str, dict[str, list[dict[str, object]]]] = {}
    for panel_name, panel in panels.items():
        arm_metrics[panel_name] = {}
        transcripts[panel_name] = {}
        for arm in ARMS:
            metrics, captured = run_arm(
                arm,
                wrapper,
                tokenizer,
                panel["rows"],
                panel["samples"],
                panel["max_new_tokens"],
                panel["max_stream_steps"],
                args.seed,
                device,
                saved_args["prompt_tokens"],
                panel["style_override"],
                capture_problems=8,
                capture_samples=4,
                answer_fence_ids=(
                    (tokenizer.answer_open_id, tokenizer.answer_close_id)
                    if saved_args.get("answer_fence")
                    else None
                ),
            )
            arm_metrics[panel_name][arm] = metrics
            transcripts[panel_name][arm] = captured
            print(
                f"{panel_name}/{arm}: accuracy {metrics['accuracy']:.4f} "
                f"({metrics['samples']} samples), "
                f"emitted mean {metrics['emitted_tokens_mean']:.1f} "
                f"p95 {metrics['emitted_tokens_p95']}",
                flush=True,
            )

    paired: dict[str, dict[str, dict[str, float | int]]] = {}
    for panel_name, panel in panels.items():
        paired[panel_name] = {}
        for arm_a, arm_b in (
            ("full", "no_content"),
            ("full", "token_only"),
            ("no_content", "token_only"),
        ):
            paired[panel_name][f"{arm_a}_vs_{arm_b}"] = paired_prompt_stats(
                arm_metrics[panel_name][arm_a]["prompt_correct_counts"],
                arm_metrics[panel_name][arm_b]["prompt_correct_counts"],
                panel["samples"],
                args.seed,
                args.resamples,
            )

    result = {
        "schema": CARRY_ABLATION_SCHEMA,
        "step": step,
        "wrapper_checkpoint": str(wrapper_path),
        "base_checkpoint": args.checkpoint,
        "actor_carry_weight_rms": carry_rms,
        "critic_carry_weight_rms": critic_carry_rms,
        "aime_modal_baseline": modal_answer_baseline(aime_rows),
        "math_modal_baseline": modal_answer_baseline(math_rows),
        # The arms run the trainer's evaluation protocol; the probes run the
        # training operating point (recorded inside "probes").
        "arm_temperature": 1.0,
        "arm_top_p": 0.7,
        "arm_metrics": arm_metrics,
        "paired": paired,
        "eval_args": vars(args),
    }
    # Persist the behavioral sweep before the probes: a probe failure must
    # never discard the three-arm evaluation it follows.
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    (out_dir / "transcripts.json").write_text(
        json.dumps(transcripts, indent=2)
    )
    print(f"wrote {out_dir}/results.json (behavioral arms)", flush=True)

    print("probing critic/policy hidden readout...", flush=True)
    probes = hidden_probes(
        wrapper,
        critic,
        tokenizer,
        probe_rows,
        args.probe_samples,
        saved_args,
        args.seed + 2,
        device,
        stop_ids,
    )
    result["probes"] = {
        key: value
        for key, value in probes.items()
        if key != "trajectory_rows"
    }
    result["probe_trajectories"] = probes["trajectory_rows"]
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {out_dir}/results.json (with probes)")
    for panel_name in panels:
        for comparison, stats in paired[panel_name].items():
            direction = (
                "improves"
                if stats["mean_delta"] > 0
                else "hurts" if stats["mean_delta"] < 0 else "ties"
            )
            print(
                f"{panel_name} {comparison}: "
                f"{stats['accuracy_a']:.4f} vs {stats['accuracy_b']:.4f}, "
                f"delta {stats['mean_delta']:+.4f} "
                f"[{stats['bootstrap_ci_low']:+.4f}, "
                f"{stats['bootstrap_ci_high']:+.4f}], "
                f"p={stats['permutation_p']:.4f} ({direction})"
            )


if __name__ == "__main__":
    main()

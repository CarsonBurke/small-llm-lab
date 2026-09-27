"""Decompose the Delightful gate's entropy push by trajectory outcome.

Delightful PG weights each sampled token's score by
``c = sigmoid(U * l / eta) * U`` (U the advantage, l = -log pi of the sampled
token). How a per-token coefficient moves that position's entropy H depends
on the geometry of the step, so two first-order proxies are reported, each
as a one-sample estimate over the sampled token a:

- ``natural``: a natural-gradient step on that position's logits changes H by
  ``-Cov_{a~pi}(log pi(a), c(a))``, estimated by ``c * (l - H)``;
- ``softmax``: a plain gradient step on those logits moves along
  ``e_a - pi``, changing H by ``c * [pi_a (l - H) + sum_k pi_k^2 (log pi_k + H)]``,
  which down-weights low-probability tokens.

Neither is the shared-parameter effect exactly; agreement between them is
what makes a sign trustworthy. Positive sums push entropy up.

This loads a cot checkpoint (policy and critic), collects a pool through the
training rollout path on the run's own RL mixture, fills critic values and
sampled-token log-probs with the trainer's ``refresh_old_statistics``, and
computes GAE. It then reports both proxies, split by correct versus failed
trajectories, for these estimators:

- ``dg_critic``: DG on critic GAE at the run's actor lambda (what the run
  trains on);
- ``pg_critic``: 0.5 * PG on the same advantages, the gate-free control at
  DG's mean gate of 1/2;
- ``dg_critic_lambda_<x>``/``pg_critic_lambda_<x>``: the same at each fixed
  actor lambda in ``--lambdas``, on the same pool and critic values;
- ``dg_group``/``pg_group``: the same on a sequence-level group baseline,
  reward minus the prompt group's mean reward.

DG needs each token's advantage to be that token's own: with exact values,
lambda-0 GAE gives a non-causal token zero advantage, while longer horizons
mix in later tokens' luck, which DG's gate turns into an entropy push
(``postraining/sim_dg_sequence_credit.py``). The lambda sweep measures that
trade on the real critic. The pool's critic explained variance against the
lambda-1 return targets is reported beside it.

Uncertainty is a 95% bootstrap interval over prompt groups.

    python -m postraining.diagnose_dg_entropy \\
        --wrapper-checkpoint postraining/runs/<run>/latent_vapo_checkpoint.pt

Writes ``dg_entropy_diagnostic/step_XXXXXX.json`` beside the checkpoint.
It executes the model on the GPU, so it runs through ``mlq``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from postraining.core import (
    answer_style,
    encode_prompt,
    generalized_advantage_and_return_targets,
    actor_gae_lambdas,
)
from postraining.inspect_critic_values import load_run
from postraining.latent_rollout import (
    refresh_old_statistics,
    rollout_continuations,
    trim_stream,
)
from postraining.train_latent_vapo import (
    rewrite_prompts_for_answer_fence,
    score_math_rollout,
)
from postraining.train_vapo import prompt_text
from postraining.vapo.mixture import MixedPromptSampler, load_mixture_manifest

PROXIES = ("natural", "softmax")
OUTCOMES = ("correct", "failed")
# Trajectories per teacher-forced render: each holds a [stream, vocab] fp32
# log-softmax plus its exp, about 0.5 GB per row at GPT-2 vocab.
ENTROPY_ROWS_PER_CHUNK = 4
# bf16 render-versus-replay surprisal gap counted as a large mismatch.
MISMATCH_TOLERANCE = 0.05


def sampled_entropy_terms(
    logits: Tensor, token_ids: Tensor
) -> dict[str, Tensor]:
    """Per-slot statistics of the distribution that sampled the next token.

    Slot p's action emits ``token_ids[:, p + 1]`` from slot p's logits, the
    layout of ``old_token_logprobs``, so every output is indexed by the
    acting slot and right-padded with zeros to the stream length:
    ``surprisal`` l, ``entropy`` H, and ``collision`` =
    sum_k pi_k^2 (log pi_k + H), the softmax proxy's baseline term.
    """
    log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
    probs = log_probs.exp()
    surprisal = -log_probs.gather(-1, token_ids[:, 1:, None]).squeeze(-1)
    entropy = -(probs * log_probs).sum(-1)
    collision = (probs.square() * (log_probs + entropy[..., None])).sum(-1)
    return {
        name: F.pad(value, (0, 1))
        for name, value in (
            ("surprisal", surprisal),
            ("entropy", entropy),
            ("collision", collision),
        )
    }


def entropy_proxies(
    surprisal: Tensor, entropy: Tensor, collision: Tensor
) -> dict[str, Tensor]:
    """Per-unit-coefficient first-order entropy change of each proxy."""
    centered = surprisal - entropy
    return {
        "natural": centered,
        "softmax": torch.exp(-surprisal) * centered + collision,
    }


def delightful_coefficients(
    advantages: Tensor, surprisal: Tensor, eta: float
) -> Tensor:
    """The score weight ``delightful_policy_loss`` applies to each token."""
    return torch.sigmoid(advantages * surprisal / eta) * advantages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper-checkpoint", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--pools", type=int, default=1,
        help="rollout pools to collect; each draws the run's prompts_per_rollout "
        "in the mixture's exact-pass source proportions",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--lambdas", default="",
        help="comma-separated fixed actor GAE lambdas to evaluate beside the "
        "run's actor lambda",
    )
    args = parser.parse_args()
    if args.pools < 1 or args.bootstrap < 1:
        parser.error("--pools and --bootstrap must be positive")
    fixed_lambdas = [float(value) for value in args.lambdas.split(",") if value]
    if any(not 0.0 <= value <= 1.0 for value in fixed_lambdas):
        parser.error("--lambdas must lie in [0, 1]")
    advantage_names = (
        "critic",
        *(f"critic_lambda_{value:g}" for value in fixed_lambdas),
        "group",
    )
    estimators = [f"{kind}_{name}" for name in advantage_names for kind in ("dg", "pg")]

    wrapper_path = Path(args.wrapper_checkpoint)
    device = torch.device("cuda")
    run = load_run(wrapper_path, args.checkpoint, device)
    saved = run.saved_args
    if run.reasoning_mode != "cot":
        raise ValueError(
            "this diagnostic renders token-only teacher-forced logits; "
            f"got reasoning mode {run.reasoning_mode!r}"
        )
    if not saved.get("delightful_policy_gradient"):
        raise ValueError("the checkpoint's run did not train with DG")
    if saved.get("source_success_actor_gate"):
        raise ValueError(
            "the source success gate masks actor rows; this diagnostic "
            "reproduces only the ungated advantage"
        )
    if saved.get("exclude_modules"):
        raise ValueError(
            "the run excluded prompt modules; this diagnostic samples the "
            "unfiltered mixture"
        )
    if saved.get("temperature", 1.0) != 1.0 or saved.get("top_p", 1.0) != 1.0:
        raise ValueError(
            "full-softmax entropy is the sampling entropy only at "
            "temperature 1 and top_p 1"
        )
    eta = 1.0  # delightful_policy_loss's default; the trainer passes no other

    # The trainer's loader: it verifies every source's bytes and effective
    # corpus identity against the manifest the run bound.
    _, sources, _ = load_mixture_manifest(saved["rl_mixture_manifest"])
    samples = int(saved["samples_per_prompt"])
    rng = random.Random(args.seed)
    # The sampler's balanced schedule sets how many prompts of each source a
    # window of this size holds; the rows themselves are drawn at random.
    source_counts = MixedPromptSampler(
        sources, args.seed, dataset_identity="diagnostic"
    ).source_counts(0, int(saved["prompts_per_rollout"]) * args.pools)
    prompts = []
    for source in sources:
        if source.verifier != "math":
            raise ValueError(f"source {source.name} is not a math source")
        rows = list(source.rows)
        if run.answer_fence_ids is not None:
            rows = rewrite_prompts_for_answer_fence(rows)
        prompts += [
            (source.name, rows[index])
            for index in rng.sample(
                range(len(rows)), source_counts.get(source.name, 0)
            )
        ]

    generator = torch.Generator(device=device).manual_seed(args.seed)
    groups = []
    # Rendered minus replayed surprisal over emitted tokens: signed sum,
    # square sum, count above MISMATCH_TOLERANCE, max magnitude, tokens.
    mismatch = {"sum": 0.0, "square_sum": 0.0, "over": 0, "max": 0.0, "tokens": 0}
    # Critic values against lambda-1 return targets over action slots:
    # value/target first and second moments and the residual square sum.
    critic_fit = {"tokens": 0, "target_sum": 0.0, "target_square_sum": 0.0, "residual_square_sum": 0.0}
    for prompt_index, (source_name, row) in enumerate(prompts):
        encoded = encode_prompt(
            run.tokenizer,
            prompt_text(row),
            run.prompt_budget - len(run.solution_prefix_ids),
        ) + list(run.solution_prefix_ids)
        prompt_ids = torch.tensor(encoded, dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                batch = trim_stream(
                    rollout_continuations(
                        run.wrapper,
                        prompt_ids[None],
                        run.response_budget,
                        run.stream_budget,
                        1.0,
                        1.0,
                        generator=generator,
                        stop_ids=run.stop_ids or None,
                        pin_emit=run.pin_emit,
                        hidden_carry=run.hidden_carry,
                        record_likelihoods=True,
                        cache_dtype=torch.bfloat16,
                        prompt_repeats=samples,
                    )
                )
            score_math_rollout(
                batch,
                row["reward_model"]["ground_truth"],
                run.tokenizer,
                run.stop_ids,
                answer_style(row),
                saved.get("nearby_reward_max", 0.1),
                solution_prefix_ids=run.solution_prefix_ids,
                think_fence_ids=run.think_fence_ids,
                min_think_tokens=int(saved.get("think_min_tokens", 1)),
                answer_fence_ids=run.answer_fence_ids,
            )
            # The trainer's exact statistics path: critic values for GAE and
            # replayed sampled-token log-probs for the gate's surprisal.
            refresh_old_statistics(
                run.wrapper,
                run.critic,
                batch,
                max_trajectories=saved["replay_max_trajectories"],
                attention_budget=saved["replay_attention_budget"],
                bucket_multiple=saved["replay_bucket"],
                slot_budget=saved["replay_slot_budget"],
            )
            with torch.no_grad():
                chunks = [
                    sampled_entropy_terms(
                        run.wrapper.policy_logits(token_ids), token_ids
                    )
                    for token_ids in batch.token_ids.split(ENTROPY_ROWS_PER_CHUNK)
                ]
        rendered = {
            name: torch.cat([chunk[name] for chunk in chunks])
            for name in chunks[0]
        }
        mask = batch.emit_mask.bool()
        surprisal = -batch.old_token_logprobs.float()
        # The teacher-forced render must reproduce the replayed statistics
        # the gate uses, or its entropies describe a different distribution.
        difference = (rendered["surprisal"] - surprisal)[mask]
        mismatch["sum"] += float(difference.sum())
        mismatch["square_sum"] += float(difference.square().sum())
        mismatch["over"] += int((difference.abs() > MISMATCH_TOLERANCE).sum())
        mismatch["max"] = max(mismatch["max"], float(difference.abs().max()))
        mismatch["tokens"] += difference.numel()
        proxies = entropy_proxies(
            surprisal, rendered["entropy"], rendered["collision"]
        )
        lambdas = actor_gae_lambdas(
            batch.action_mask.sum(1),
            saved["gae_lambda_alpha"],
            saved.get("actor_gae_lambda"),
        )
        critic_advantages, _ = generalized_advantage_and_return_targets(
            batch.rewards, batch.old_values, batch.action_mask, lambdas
        )
        _, returns = generalized_advantage_and_return_targets(
            batch.rewards, batch.old_values, batch.action_mask,
            torch.ones_like(lambdas),
        )
        actions = batch.action_mask.bool()
        targets = returns[actions].float()
        critic_fit["tokens"] += targets.numel()
        critic_fit["target_sum"] += float(targets.sum())
        critic_fit["target_square_sum"] += float(targets.square().sum())
        critic_fit["residual_square_sum"] += float(
            (targets - batch.old_values[actions].float()).square().sum()
        )
        fixed_advantages = [
            generalized_advantage_and_return_targets(
                batch.rewards, batch.old_values, batch.action_mask,
                torch.full_like(lambdas, value),
            )[0].float()
            for value in fixed_lambdas
        ]
        rewards = batch.reward_scalar.float()
        group_advantages = (rewards - rewards.mean())[:, None].expand_as(surprisal)
        correct = rewards >= 1.0
        split = {"correct": mask & correct[:, None], "failed": mask & ~correct[:, None]}
        record = {
            "source": source_name,
            "correct": int(correct.sum()),
            "samples": samples,
            "tokens": {outcome: int(split[outcome].sum()) for outcome in OUTCOMES},
            "surprisal_sum": float(surprisal[mask].sum()),
            "entropy_sum": float(rendered["entropy"][mask].sum()),
        }
        for name, advantages in zip(
            advantage_names,
            (critic_advantages.float(), *fixed_advantages, group_advantages),
            strict=True,
        ):
            for estimator, coefficient in (
                (f"dg_{name}", delightful_coefficients(advantages, surprisal, eta)),
                (f"pg_{name}", 0.5 * advantages),
            ):
                record[estimator] = {
                    proxy: {
                        outcome: float((coefficient * change)[split[outcome]].sum())
                        for outcome in OUTCOMES
                    }
                    for proxy, change in proxies.items()
                }
            failed = advantages[split["failed"]]
            record[f"{name}_failed_advantage"] = {
                "sum": float(failed.sum()),
                "square_sum": float(failed.square().sum()),
                "positive_count": int((failed > 0).sum()),
            }
        groups.append(record)
        print(
            f"{prompt_index + 1:3d}/{len(prompts)} {source_name:15s} "
            f"correct {record['correct']:2d}/{samples} "
            f"tokens {sum(record['tokens'].values()):6d} "
            f"dg_critic natural failed {record['dg_critic']['natural']['failed']:+.3f} "
            f"correct {record['dg_critic']['natural']['correct']:+.3f}",
            flush=True,
        )

    def per_token(
        chosen: list[dict], estimator: str, proxy: str, outcome: str | None
    ) -> float:
        outcomes = OUTCOMES if outcome is None else (outcome,)
        tokens = sum(sum(g["tokens"].values()) for g in chosen)
        change = sum(g[estimator][proxy][o] for g in chosen for o in outcomes)
        return change / max(tokens, 1)

    # Per-token over ALL emitted tokens in every cell, so the failed and
    # correct cells add up to the total.
    bootstrap_rng = random.Random(args.seed + 1)
    resamples = [
        [bootstrap_rng.choice(groups) for _ in groups]
        for _ in range(args.bootstrap)
    ]
    summary = {}
    for estimator in estimators:
        summary[estimator] = {}
        for proxy in PROXIES:
            summary[estimator][proxy] = {}
            for outcome in (*OUTCOMES, None):
                draws = sorted(
                    per_token(chosen, estimator, proxy, outcome)
                    for chosen in resamples
                )
                summary[estimator][proxy][outcome or "total"] = {
                    "per_token": per_token(groups, estimator, proxy, outcome),
                    "ci95": [
                        draws[int(0.025 * len(draws))],
                        draws[int(0.975 * len(draws)) - 1],
                    ],
                }

    total_tokens = sum(sum(g["tokens"].values()) for g in groups)
    failed_tokens = sum(g["tokens"]["failed"] for g in groups)
    failed_advantage = {}
    for name in advantage_names:
        cells = [g[f"{name}_failed_advantage"] for g in groups]
        mean = sum(c["sum"] for c in cells) / max(failed_tokens, 1)
        failed_advantage[name] = {
            "mean": mean,
            "std": max(
                sum(c["square_sum"] for c in cells) / max(failed_tokens, 1)
                - mean * mean,
                0.0,
            ) ** 0.5,
            "positive_fraction": sum(c["positive_count"] for c in cells)
            / max(failed_tokens, 1),
        }
    report = {
        "schema": "dg_entropy_diagnostic/v3",
        "checkpoint": str(wrapper_path),
        "step": run.step,
        "eta": eta,
        "gae_lambda_alpha": saved["gae_lambda_alpha"],
        "actor_gae_lambda": saved.get("actor_gae_lambda"),
        "fixed_lambdas": fixed_lambdas,
        "critic_explained_variance": 1.0 - critic_fit["residual_square_sum"]
        / max(
            critic_fit["target_square_sum"]
            - critic_fit["target_sum"] ** 2 / max(critic_fit["tokens"], 1),
            1e-12,
        ),
        "seed": args.seed,
        "prompts": len(groups),
        "trajectories": len(groups) * samples,
        "correct_fraction": sum(g["correct"] for g in groups)
        / (len(groups) * samples),
        "tokens": total_tokens,
        "failed_token_fraction": failed_tokens / max(total_tokens, 1),
        "mean_surprisal": sum(g["surprisal_sum"] for g in groups) / total_tokens,
        "mean_entropy": sum(g["entropy_sum"] for g in groups) / total_tokens,
        "teacher_forced_surprisal_mismatch": {
            "mean": mismatch["sum"] / mismatch["tokens"],
            "rms": (mismatch["square_sum"] / mismatch["tokens"]) ** 0.5,
            "max": mismatch["max"],
            f"fraction_above_{MISMATCH_TOLERANCE}": mismatch["over"]
            / mismatch["tokens"],
        },
        "failed_token_advantage": failed_advantage,
        "entropy_change_per_token": summary,
        "groups": groups,
    }
    out_dir = wrapper_path.parent / "dg_entropy_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step_{run.step:06d}.json"
    out_path.write_text(json.dumps(report, indent=1))

    print(
        f"\nstep {run.step}: {report['trajectories']} trajectories, "
        f"correct {report['correct_fraction']:.4f}, tokens {total_tokens}, "
        f"mean surprisal {report['mean_surprisal']:.3f}, "
        f"mean entropy {report['mean_entropy']:.3f}, "
        f"teacher-forced surprisal mismatch mean "
        f"{report['teacher_forced_surprisal_mismatch']['mean']:+.2e} rms "
        f"{report['teacher_forced_surprisal_mismatch']['rms']:.2e} max "
        f"{mismatch['max']:.2e}"
    )
    print(f"critic explained variance (lambda-1 targets): {report['critic_explained_variance']:+.4f}")
    for name, cell in failed_advantage.items():
        print(
            f"failed-token {name} advantage: mean {cell['mean']:+.4f} "
            f"std {cell['std']:.4f} positive {cell['positive_fraction']:.3f}"
        )
    print("first-order entropy change per emitted token [95% CI]:")
    for proxy in PROXIES:
        print(f" {proxy}")
        for estimator in estimators:
            cells = []
            for key in ("failed", "correct", "total"):
                cell = summary[estimator][proxy][key]
                cells.append(
                    f"{key} {cell['per_token']:+.2e} "
                    f"[{cell['ci95'][0]:+.2e}, {cell['ci95'][1]:+.2e}]"
                )
            print(f"  {estimator:24s} " + " | ".join(cells))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

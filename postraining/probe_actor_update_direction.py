"""Measure what ONE actor step of each objective does to the token distribution.

DG and PG take different steps from the same pool only through the per-token
weight on ``grad log pi``: PG uses ``A``, DG ``sigmoid(A l / eta) A``. With
Muon (and Adam) the step size does not follow the gradient's scale, so DG's
roughly-1/2 gate cancels and its surprisal tilt is the whole difference.
This probe isolates that difference on the real model, critic, memory carry
and optimizer:

1. Load a state (actor, critic) and collect ``--pools`` training-sized pools
   at the training operating point (temperature 1, top-p 1), scored and
   refreshed as the trainer does.
2. Roll a separate fixed evaluation set from the same state.
3. For each pool and each estimator, restore the state, build fresh
   optimizers (no momentum history), run the trainer's ``update_minibatch``
   with the actor loss replaced by that estimator's weight, step, and
   re-score the fixed evaluation set teacher-forced.

The evaluation is deterministic (identical stored streams and beliefs), so
a per-step change resolves without sampling noise; ``noise_floor`` re-scores
the unchanged state to bound it. Reported per estimator, relative to the
unchanged state, over the evaluation set's emitted tokens:

- mean entropy and junk mass (probability on tokens below
  ``JUNK_PROBABILITY``);
- mean change of the sampled token's log-prob per base-surprisal bin (does
  the step lift tokens the state found unlikely?);
- the stop tokens' log-prob, and KL(base || stepped) estimated from the
  base-sampled evaluation tokens.

Estimators (``--estimators``):

- ``pg``: ``A``; ``dg``: ``sigmoid(A l) A``; ``tilt``: ``(sigmoid(A l) - 1/2) A``,
  DG's departure from 1/2 PG on its own;
- ``dg_eta<x>``: DG at temperature x; ``dg_cap<x>``: surprisal capped at x
  inside the gate;
- ``*_perm``: advantages permuted across the shard's tokens, keeping their
  distribution and destroying their alignment (signal-free control);
- ``*_td0``: actor GAE lambda 0, the one-step ``V(s') - V(s)`` advantage,
  which for deterministic token transitions is the critic's estimate of
  ``Q(s, a) - V(s)``;
- ``*_row``: every token takes its row's mean advantage, so no within-row
  (per-token critic) noise reaches the gate;
- ``dg_debias``: DG minus its noise rectification ``l sigma^2 / (4 eta)``,
  with ``sigma^2`` the shard's pooled within-row advantage variance (the
  second-order term DG adds to ``A/2`` for zero-mean noise).

It executes the model on the GPU, so it runs through ``mlq``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

import postraining.train_latent_vapo as tlv
from postraining.core import delightful_policy_loss, encode_prompt
from postraining.diagnose_tail_derailment import (
    JUNK_PROBABILITY,
    file_sha256,
    panel_prompts,
    score_group,
)
from postraining.inspect_critic_values import load_run
from postraining.latent_rollout import (
    pack_rollout_groups_for_replay,
    refresh_old_statistics,
    replay_beliefs,
    rollout_continuations,
    split_rollout_groups,
)
from postraining.train_vapo import prompt_text

SCHEMA = "actor_update_direction_probe/v1"
SURPRISAL_BINS = (0.0, 2.0, 6.0, 10.0, math.inf)
EVAL_POSITION_CHUNK = 256
ESTIMATOR = re.compile(
    r"^(pg|dg|tilt)(?:_eta(?P<eta>[0-9.]+))?(?:_cap(?P<cap>[0-9.]+))?"
    r"(?P<debias>_debias)?(?P<row>_row)?(?P<perm>_perm)?(?P<td0>_td0)?$"
)


class Estimator(NamedTuple):
    weight: Callable[[Tensor, Tensor, Tensor], Tensor]  # (A, l, sigma^2)
    permute: bool
    td0: bool
    row: bool


def estimator_weight(name: str) -> Estimator:
    """The per-token weight and advantage transforms of an estimator name."""
    match = ESTIMATOR.match(name)
    if match is None:
        raise ValueError(f"unknown estimator {name!r}")
    kind = match.group(1)
    eta = float(match.group("eta") or 1.0)
    cap = float(match.group("cap")) if match.group("cap") else math.inf
    if eta <= 0 or cap <= 0:
        raise ValueError(f"{name}: eta and cap must be positive")
    debias = bool(match.group("debias"))
    if kind == "pg" and (match.group("eta") or match.group("cap") or debias):
        raise ValueError(f"{name}: pg has no gate")

    def weight(advantages: Tensor, surprisal: Tensor, noise_variance: Tensor) -> Tensor:
        if kind == "pg":
            return advantages
        surprisal = surprisal.clamp(max=cap)
        gate = torch.sigmoid(advantages * surprisal / eta)
        value = (gate - 0.5) * advantages if kind == "tilt" else gate * advantages
        return value - surprisal * noise_variance / (4 * eta) if debias else value

    return Estimator(weight, bool(match.group("perm")), bool(match.group("td0")), bool(match.group("row")))


def within_row_variance(advantages: Tensor, mask: Tensor) -> Tensor:
    """Pooled within-row variance of the active advantages (a scalar)."""
    count = mask.sum(-1)
    mean = (advantages * mask).sum(-1) / count.clamp_min(1)
    squares = (((advantages - mean[:, None]) ** 2) * mask).sum()
    return squares / (count - (count > 0).to(count.dtype)).sum().clamp_min(1)


def patched_actor_loss(estimator: Estimator, generator: torch.Generator):
    """A drop-in ``delightful_policy_loss`` with another per-token weight.

    Diagnostics come from the real DG loss so the trainer's bookkeeping is
    unchanged; only the returned loss (and so the gradient) differs.
    """

    def loss_fn(logprobs, advantages, mask, temperature=1.0, denominator=None):
        _, diagnostics = delightful_policy_loss(logprobs, advantages, mask, temperature, denominator)
        advantages = advantages.detach()
        mask = mask.to(advantages.dtype)
        noise_variance = within_row_variance(advantages, mask)
        if estimator.row:
            row_mean = (advantages * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
            advantages = row_mean.expand_as(advantages) * mask
        if estimator.permute:
            active = mask.bool()
            values = advantages[active]
            order = torch.randperm(values.numel(), generator=generator, device="cpu").to(values.device)
            advantages = advantages.masked_scatter(active, values[order])
        surprisal = -logprobs.detach()
        coefficient = estimator.weight(advantages, surprisal, noise_variance).detach()
        denom = (mask.sum() if denominator is None else denominator.to(mask.device)).clamp_min(1)
        return -(coefficient * logprobs * mask).sum() / denom, diagnostics

    return loss_fn


def collect(run, panel, samples: int, chunk_prompts: int, seed: int, device) -> list:
    """Scored, refreshed groups (``source_id`` set) of one panel."""
    saved = run.saved_args
    groups = []
    for chunk_index, start in enumerate(range(0, len(panel), chunk_prompts)):
        chunk = panel[start : start + chunk_prompts]
        encoded = [encode_prompt(run.tokenizer, prompt_text(row), run.prompt_budget) for _, row in chunk]
        width = max(len(ids) for ids in encoded)
        prompt_ids = torch.zeros((len(chunk), width), dtype=torch.long)
        for index, ids in enumerate(encoded):
            prompt_ids[index, width - len(ids):] = torch.tensor(ids)
        lengths = torch.tensor([len(ids) for ids in encoded])
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            batched = rollout_continuations(
                run.wrapper,
                prompt_ids.to(device),
                run.response_budget,
                run.stream_budget,
                1.0,
                1.0,
                generator=torch.Generator(device=device).manual_seed(seed + chunk_index),
                stop_ids=run.stop_ids or None,
                prompt_lengths=lengths.to(device),
                pin_emit=True,
                hidden_carry=True,
                record_likelihoods=False,
                cache_dtype=torch.bfloat16,
                prompt_repeats=samples,
            )
        for group, (source, row) in zip(
            split_rollout_groups(batched, samples, lengths.repeat_interleave(samples).to(device)), chunk, strict=True
        ):
            score_group(group, row, run)
            group.source_id = torch.zeros(group.kind.size(0), dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                refresh_old_statistics(
                    run.wrapper,
                    run.critic,
                    group,
                    max_trajectories=saved["replay_max_trajectories"],
                    attention_budget=saved["replay_attention_budget"],
                    bucket_multiple=saved["replay_bucket"],
                    slot_budget=saved["replay_slot_budget"],
                )
            groups.append(group)
        del batched
    return groups


@torch.no_grad()
def evaluate(wrapper, groups, stop_ids: tuple[int, ...]) -> dict[str, Tensor]:
    """Per emitted token of the fixed set: full-distribution statistics.

    Teacher-forced through the replay path on the stored stream and beliefs,
    so the contexts are identical for every parameter state.
    """
    out = {name: [] for name in ("entropy", "junk", "sampled", "stop")}
    for group in groups:
        device_type = group.kind.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            stream_inputs, beliefs = replay_beliefs(wrapper, group)
            features = wrapper.renderer_features(stream_inputs, beliefs)
        targets = torch.zeros_like(group.token_ids)
        targets[:, :-1] = group.token_ids[:, 1:]
        emits = group.emit_mask.bool()
        for begin in range(0, features.size(1), EVAL_POSITION_CHUNK):
            span = slice(begin, begin + EVAL_POSITION_CHUNK)
            active = emits[:, span]
            if not active.any():
                continue
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
                logits = wrapper.backbone.logits_from_features(features[:, span][active])
            log_probs = logits.float().log_softmax(-1)
            probs = log_probs.exp()
            out["entropy"].append(-(probs * log_probs).sum(-1).cpu())
            out["junk"].append(probs.masked_fill(probs >= JUNK_PROBABILITY, 0.0).sum(-1).cpu())
            out["sampled"].append(log_probs.gather(-1, targets[:, span][active][:, None]).squeeze(-1).cpu())
            out["stop"].append(log_probs[:, list(stop_ids)].logsumexp(-1).cpu())
    return {name: torch.cat(values) for name, values in out.items()}


def compare(base: dict[str, Tensor], stepped: dict[str, Tensor]) -> dict:
    surprisal = -base["sampled"]
    bins = torch.bucketize(surprisal, torch.tensor(SURPRISAL_BINS[1:-1]), right=True)
    delta = stepped["sampled"] - base["sampled"]
    by_bin = {}
    for index in range(len(SURPRISAL_BINS) - 1):
        chosen = bins == index
        by_bin[f"[{SURPRISAL_BINS[index]}, {SURPRISAL_BINS[index + 1]})"] = {
            "tokens": int(chosen.sum()),
            "sampled_logprob_delta": float(delta[chosen].mean()) if chosen.any() else None,
        }
    return {
        "entropy_delta": float((stepped["entropy"] - base["entropy"]).mean()),
        "junk_mass_delta": float((stepped["junk"] - base["junk"]).mean()),
        "stop_logprob_delta": float((stepped["stop"] - base["stop"]).mean()),
        # The evaluation tokens were sampled from the base state, so the mean
        # log-ratio of the sampled tokens is an unbiased KL(base || stepped).
        "kl_base_to_stepped": float((-delta).mean()),
        "sampled_logprob_delta_by_base_surprisal": by_bin,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", action="append", required=True, help="label=checkpoint")
    parser.add_argument("--checkpoint", required=True, help="base SFT checkpoint the states share")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pools", type=int, default=3)
    parser.add_argument("--pool-prompts", type=int, default=64)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--eval-prompts", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--chunk-prompts", type=int, default=16)
    parser.add_argument("--lr-multipliers", default="1,8")
    parser.add_argument(
        "--estimators",
        default="pg,dg,tilt,dg_perm,pg_perm,dg_td0,pg_td0,dg_cap4,dg_eta4",
    )
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    estimators = [name for name in args.estimators.split(",") if name]
    for name in estimators:
        estimator_weight(name)
    multipliers = [float(value) for value in args.lr_multipliers.split(",") if value]
    if not multipliers or min(multipliers) <= 0:
        parser.error("--lr-multipliers must be positive")

    device = torch.device("cuda")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": SCHEMA, "args": vars(args), "junk_probability": JUNK_PROBABILITY, "states": {}}
    for spec in args.state:
        label, _, path = spec.partition("=")
        run = load_run(Path(path), args.checkpoint, device)
        # load_run freezes everything for inspection; the update needs the
        # trainer's exact trainable set, or the optimizer partition differs.
        tlv.make_actor_trainable(run.wrapper)
        saved = run.saved_args
        if not run.wrapper.memory_carry:
            raise ValueError(f"{label}: expected a memory-carry policy")
        fence = run.answer_fence_ids is not None
        mixture = saved["rl_mixture_manifest"]
        eval_groups = collect(
            run, panel_prompts(mixture, args.eval_prompts, args.seed - 1, fence),
            args.eval_samples, args.chunk_prompts, args.seed - 1, device,
        )
        base = evaluate(run.wrapper, eval_groups, run.stop_ids)
        repeat = evaluate(run.wrapper, eval_groups, run.stop_ids)
        state = {
            "path": path,
            "sha256": file_sha256(Path(path)),
            "step": run.step,
            "eval_tokens": int(base["entropy"].numel()),
            "eval_ended": float(sum(int(g.token_ids[r, g.emit_mask[r].bool().nonzero()[-1] + 1]) in run.stop_ids for g in eval_groups for r in range(g.kind.size(0))) / sum(g.kind.size(0) for g in eval_groups)),
            "base_entropy": float(base["entropy"].mean()),
            "base_junk_mass": float(base["junk"].mean()),
            "noise_floor": {
                key: float((repeat[key] - base[key]).abs().max()) for key in ("entropy", "junk", "sampled", "stop")
            },
            "pools": [],
        }
        print(f"{label}: eval tokens {state['eval_tokens']}, entropy {state['base_entropy']:.3f}, junk {state['base_junk_mass']:.4f}, noise {state['noise_floor']}", flush=True)
        pristine_actor = copy.deepcopy(run.wrapper.state_dict())
        original_loss = tlv.delightful_policy_loss
        for pool_index in range(args.pools):
            pool_seed = args.seed + 1000 * (pool_index + 1)
            groups = collect(
                run, panel_prompts(mixture, args.pool_prompts, pool_seed, fence),
                args.samples, args.chunk_prompts, pool_seed, device,
            )
            pool = pack_rollout_groups_for_replay(groups).to(device)
            del groups
            rewards = pool.reward_scalar.float()
            pool_record = {
                "seed": pool_seed,
                "trajectories": int(rewards.numel()),
                "correct_fraction": float((rewards >= 1.0).float().mean()),
                "reward_mean": float(rewards.mean()),
                "emitted_tokens": int(pool.emit_mask.sum()),
                "estimators": {},
            }
            for name in estimators:
                estimator = estimator_weight(name)
                for multiplier in multipliers:
                    run.wrapper.load_state_dict(pristine_actor)
                    optimizers = tlv.build_optimizers(
                        run.wrapper,
                        run.critic,
                        saved["learning_rate"] * multiplier,
                        critic_learning_rate=saved.get("critic_learning_rate"),
                        trunk_optimizer=saved["trunk_optimizer"],
                        muon_learning_rate=(saved.get("muon_learning_rate") or 0.0) * multiplier or None,
                        critic_muon_learning_rate=saved.get("critic_muon_learning_rate"),
                        # The fresh modules are actor parameters: their
                        # rates scale with the probed actor step too.
                        fresh_learning_rate=(saved.get("fresh_learning_rate") or 0.0) * multiplier or None,
                        fresh_muon_learning_rate=(saved.get("fresh_muon_learning_rate") or 0.0) * multiplier or None,
                    )
                    tlv.delightful_policy_loss = patched_actor_loss(
                        estimator, torch.Generator().manual_seed(pool_seed)
                    )
                    try:
                        metrics = tlv.update_minibatch(
                            run.wrapper,
                            run.critic,
                            pool,
                            optimizers,
                            actor_step=True,
                            critic_step=False,
                            policy_action_denominator=pool.emit_mask.sum(),
                            gae_lambda_alpha=saved["gae_lambda_alpha"],
                            actor_gae_lambda=0.0 if estimator.td0 else saved.get("actor_gae_lambda"),
                            replay_max_trajectories=saved["replay_max_trajectories"],
                            replay_attention_budget=saved["replay_attention_budget"],
                            replay_bucket=saved["replay_bucket"],
                            replay_slot_budget=saved["replay_slot_budget"],
                            delightful_policy_gradient=True,
                        )
                    finally:
                        tlv.delightful_policy_loss = original_loss
                    for parameter in run.critic.parameters():
                        parameter.grad = None
                    del optimizers
                    stepped = evaluate(run.wrapper, eval_groups, run.stop_ids)
                    record = compare(base, stepped)
                    record["update_metrics"] = {
                        key: float(value)
                        for key, value in metrics.items()
                        if isinstance(value, (int, float)) and ("advantage" in key or "delightful" in key or "grad" in key)
                    }
                    pool_record["estimators"][f"{name}@{multiplier:g}"] = record
                    print(
                        f"{label} pool {pool_index} {name}@{multiplier:g}: dH {record['entropy_delta']:+.2e} "
                        f"djunk {record['junk_mass_delta']:+.2e} dstop {record['stop_logprob_delta']:+.2e} "
                        f"KL {record['kl_base_to_stepped']:.2e}",
                        flush=True,
                    )
            run.wrapper.load_state_dict(pristine_actor)
            state["pools"].append(pool_record)
            del pool
            report["states"][label] = state
            out_path.write_text(json.dumps(report, indent=1))
        del run, pristine_actor, eval_groups, base, repeat
        torch.cuda.empty_cache()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

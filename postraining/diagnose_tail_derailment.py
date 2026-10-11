"""Locate where sampled trajectories derail, and what credit those tokens get.

Job 12674 (DG, memory carry) lost termination at the training operating
point (temperature 1, top-p 1) while its top-p 0.7 bench stayed terminated.
This diagnostic rolls one fixed mixture panel through several policies
("arms") and separates three questions:

- **Distribution.** Every decode step records statistics of the full
  next-token distribution: entropy, max probability, the "junk" mass on
  tokens below ``JUNK_PROBABILITY``, and log-probs of ``<think>`` and the stop
  tokens. Step 0 conditions only on the prompt, so it is a matched context
  across arms; later steps condition on each arm's own samples.
- **Derailment.** A row derails at the first emitted position where the
  trailing ``DERAIL_WINDOW``-token mean surprisal exceeds ``DERAIL_NATS``.
  The onset position, the onset token's surprisal, and whether the row ever
  opened ``<think>`` locate the failure.
- **Credit.** Each arm's own critic and the trainer's GAE give per-token
  advantages. Coefficients of Delightful PG (``sigmoid(A l) A``) and plain PG
  (``A``) are summed per sampled-token surprisal bin and outcome. With Muon
  normalizing the step, the two objectives differ in direction only through
  that surprisal tilt, so the table shows where DG and PG disagree.

Arms are ``label=path`` with optional ``,nomem`` (zero the actor's memory
read-out, leaving the trunk as trained) and ``,top_p=x``. All arms share the
panel (one pool in the mixture's source proportions, code included), the
generator seed per chunk, and the reward (no abstention penalty). Code rows
under ``test-fraction`` count as correct only when every test passes, so
partial-credit rows sit in the "failed" credit cells.

    python -m postraining.diagnose_tail_derailment \\
        --checkpoint <base sft_final_model.pt> --out-dir <dir> \\
        --arm sft=<r2>/critic_warmup_checkpoint.pt \\
        --arm dg240=<snapshot>.pt --arm dg240_nomem=<snapshot>.pt,nomem

It executes the model on the GPU, so it runs through ``mlq``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

import postraining.latent_rollout as latent_rollout
from postraining.core import (
    actor_gae_lambdas,
    answer_style,
    encode_prompt,
    generalized_advantage_and_return_targets,
)
from postraining.inspect_critic_values import load_run
from postraining.latent_rollout import (
    refresh_old_statistics,
    rollout_continuations,
    split_rollout_groups,
    trim_stream,
)
from postraining.train_latent_vapo import (
    rewrite_prompts_for_answer_fence,
    score_math_rollout,
    score_python_rollout,
)
from postraining.train_vapo import prompt_text
from postraining.vapo.mixture import MixedPromptSampler, load_mixture_manifest

SCHEMA = "tail_derailment_diagnostic/v1"
JUNK_PROBABILITY = 1e-4
DERAIL_WINDOW = 16
DERAIL_NATS = 5.0
SURPRISAL_EDGES = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, math.inf)
POSITION_EDGES = (0, 1, 16, 64, 256, 1024, math.inf)
# Per-step distribution statistics, in this column order.
STEP_FIELDS = ("entropy", "max_prob", "junk_mass", "think_logprob", "stop_logprob", "sampled_logprob")
OUTCOMES = ("correct", "failed")


@dataclass(frozen=True)
class Arm:
    label: str
    path: Path
    nomem: bool
    top_p: float


def parse_arm(spec: str) -> Arm:
    label, _, rest = spec.partition("=")
    if not label or not rest:
        raise ValueError(f"arm {spec!r} is not label=path[,nomem][,top_p=x]")
    path, *options = rest.split(",")
    nomem, top_p = False, 1.0
    for option in options:
        if option == "nomem":
            nomem = True
        elif option.startswith("top_p="):
            top_p = float(option.removeprefix("top_p="))
            if not 0.0 < top_p <= 1.0:
                raise ValueError(f"arm {label}: top_p must lie in (0, 1]")
        else:
            raise ValueError(f"arm {label}: unknown option {option!r}")
    return Arm(label, Path(path), nomem, top_p)


def step_statistics(
    logits: Tensor, token: Tensor, think_id: int, stop_ids: tuple[int, ...]
) -> Tensor:
    """``(rows, len(STEP_FIELDS))`` fp32 statistics of one decode step."""
    log_probs = logits.float().log_softmax(-1)
    probs = log_probs.exp()
    junk = probs.masked_fill(probs >= JUNK_PROBABILITY, 0.0).sum(-1)
    return torch.stack(
        (
            -(probs * log_probs).sum(-1),
            probs.amax(-1),
            junk,
            log_probs[:, think_id],
            log_probs[:, list(stop_ids)].logsumexp(-1),
            log_probs.gather(-1, token[:, None]).squeeze(-1),
        ),
        dim=-1,
    )


def derail_onset(surprisal: Tensor, window: int = DERAIL_WINDOW, nats: float = DERAIL_NATS) -> int | None:
    """First emitted index whose trailing ``window``-token mean exceeds ``nats``.

    The index is the window's END, the earliest position at which the
    criterion is observable; the window must be full.
    """
    if surprisal.numel() < window:
        return None
    means = surprisal.float().unfold(0, window, 1).mean(-1)
    hits = (means > nats).nonzero()
    return None if hits.numel() == 0 else int(hits[0]) + window - 1


def bin_index(values: Tensor, edges: tuple[float, ...]) -> Tensor:
    """Bin ``values`` into ``[edges[i], edges[i + 1])``."""
    inner = torch.tensor(edges[1:-1], dtype=values.dtype, device=values.device)
    return torch.bucketize(values, inner, right=True)


def credit_cells(
    surprisal: Tensor, advantages: Tensor, correct: bool
) -> dict[str, list[float]]:
    """Per-surprisal-bin sums of one row's DG and PG coefficients."""
    bins = bin_index(surprisal, SURPRISAL_EDGES)
    count = len(SURPRISAL_EDGES) - 1
    dg = torch.sigmoid(advantages * surprisal) * advantages
    sums = {
        "tokens": torch.ones_like(advantages),
        "advantage": advantages,
        "advantage_square": advantages.square(),
        "positive": (advantages > 0).float(),
        "dg": dg,
        "pg": advantages,
        "surprisal": surprisal,
    }
    return {
        name: torch.zeros(count, dtype=torch.float64).index_add_(0, bins.cpu(), value.double().cpu()).tolist()
        for name, value in sums.items()
    } | {"outcome": "correct" if correct else "failed"}


def aligned_row_steps(
    steps: Tensor, batch_row: int, emit_mask: Tensor, decode_logprobs: Tensor
) -> tuple[Tensor, Tensor]:
    """One trajectory's emit slots and its per-step statistics, verified.

    ``steps`` is the hook record ``(calls, rows, fields)`` of a rollout with
    ``compact_finished=False``, so call k is every row's k-th emitted token
    and ``batch_row`` indexes the batched (pre-split) row order. The hook and
    the rollout compute the sampled token's log-prob from the same fp32
    log-softmax, so any gap means the alignment is wrong.
    """
    slots = emit_mask.bool().nonzero().squeeze(-1)
    row_steps = steps[: slots.numel(), batch_row]
    if slots.numel():
        decode = decode_logprobs[slots].float().cpu()
        gap = float((row_steps[:, STEP_FIELDS.index("sampled_logprob")] - decode).abs().max())
        if gap > 1e-4:
            raise RuntimeError(f"step record misaligned with the rollout (max |gap| {gap:.3e})")
    return slots, row_steps


@contextlib.contextmanager
def recorded_sampling(think_id: int, stop_ids: tuple[int, ...], record: list[Tensor]):
    """Wrap the rollout's token sampler to log every step's distribution."""
    original = latent_rollout.top_p_sample

    def sampler(logits, temperature, top_p, **kwargs):
        token = original(logits, temperature, top_p, **kwargs)
        with torch.no_grad():
            record.append(step_statistics(logits, token, think_id, stop_ids))
        return token

    latent_rollout.top_p_sample = sampler
    try:
        yield
    finally:
        latent_rollout.top_p_sample = original


@contextlib.contextmanager
def memory_read_disabled(wrapper):
    """Zero the actor memory read-out, restoring it exactly."""
    if not wrapper.memory_carry:
        raise ValueError("nomem needs a memory-carry policy")
    weight = wrapper.combiner.reader.out.weight
    saved = weight.detach().clone()
    with torch.no_grad():
        weight.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            weight.copy_(saved)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_prompts(mixture_manifest: str, prompts: int, seed: int, answer_fence: bool) -> list[tuple[str, dict]]:
    """One pool's worth of prompts in the mixture's source proportions."""
    _, sources, _ = load_mixture_manifest(mixture_manifest)
    counts = MixedPromptSampler(sources, seed, dataset_identity="diagnostic").source_counts(0, prompts)
    rng = random.Random(seed)
    panel = []
    for source in sources:
        rows = list(source.rows)
        if answer_fence:
            rows = rewrite_prompts_for_answer_fence(rows)
        panel += [(source.name, rows[i]) for i in rng.sample(range(len(rows)), counts.get(source.name, 0))]
    return panel


def score_group(group, row: dict, run) -> None:
    """Score one group as the trainer's ``finish_group`` does, without abstention.

    Code rows keep the run's ``python_reward_mode``; under ``test-fraction``
    their reward lies in [0, 1], and only an all-tests pass counts as correct.
    """
    saved = run.saved_args
    verifier_kind = row.get("_verifier_kind", "math")
    if verifier_kind == "math":
        score_math_rollout(
            group,
            row["reward_model"]["ground_truth"],
            run.tokenizer,
            run.stop_ids,
            answer_style(row),
            saved.get("nearby_reward_max", 0.1),
            solution_prefix_ids=run.solution_prefix_ids,
            think_fence_ids=run.think_fence_ids,
            min_think_tokens=int(saved.get("think_min_tokens", 1)),
            answer_fence_ids=run.answer_fence_ids,
            abstention_penalty=None,
        )
    elif verifier_kind == "python_mbpp":
        if run.think_fence_ids is None or run.answer_fence_ids is None:
            raise ValueError("python rows need the think and answer fences")
        score_python_rollout(
            group,
            row["verification_info"],
            run.tokenizer,
            run.stop_ids,
            run.think_fence_ids,
            run.answer_fence_ids,
            int(saved["think_min_tokens"]),
            reward_mode=saved["python_reward_mode"],
            abstention_penalty=None,
        )
    else:
        raise ValueError(f"unsupported verifier kind {verifier_kind!r}")


def run_arm(arm: Arm, run, panel, args, device) -> tuple[dict, list[dict], list[dict], list[dict]]:
    saved = run.saved_args
    if not (run.pin_emit and run.hidden_carry and run.wrapper.memory_carry):
        raise ValueError(f"{arm.label}: expected a memory-carry policy, got {run.reasoning_mode!r}")
    tokenizer = run.tokenizer
    think_id = tokenizer.think_open_id
    samples = args.samples
    rows_out: list[dict] = []
    tokens_out: list[dict] = []
    credit: list[dict] = []
    mismatch = []
    context = memory_read_disabled(run.wrapper) if arm.nomem else contextlib.nullcontext()
    with context:
        for chunk_index, start in enumerate(range(0, len(panel), args.chunk_prompts)):
            chunk = panel[start : start + args.chunk_prompts]
            encoded = [
                encode_prompt(tokenizer, prompt_text(row), run.prompt_budget) for _, row in chunk
            ]
            width = max(len(ids) for ids in encoded)
            prompt_ids = torch.zeros((len(chunk), width), dtype=torch.long)
            for index, ids in enumerate(encoded):
                prompt_ids[index, width - len(ids):] = torch.tensor(ids)
            lengths = torch.tensor([len(ids) for ids in encoded])
            generator = torch.Generator(device=device).manual_seed(args.seed + chunk_index)
            record: list[Tensor] = []
            with recorded_sampling(think_id, run.stop_ids, record), torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                batched = rollout_continuations(
                    run.wrapper,
                    prompt_ids.to(device),
                    run.response_budget,
                    run.stream_budget,
                    1.0,
                    arm.top_p,
                    generator=generator,
                    stop_ids=run.stop_ids or None,
                    prompt_lengths=lengths.to(device),
                    pin_emit=True,
                    hidden_carry=True,
                    record_likelihoods=True,
                    compact_finished=False,
                    cache_dtype=torch.bfloat16,
                    prompt_repeats=samples,
                )
            # (steps, rows, fields): row order is the batched rollout's.
            steps = torch.stack(record).cpu()
            groups = split_rollout_groups(batched, samples, lengths.repeat_interleave(samples).to(device))
            del batched
            for group_index, (group, (source, row)) in enumerate(zip(groups, chunk, strict=True)):
                group = trim_stream(group)
                decode_logprobs = group.old_token_logprobs.float().clone()
                score_group(group, row, run)
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
                lambdas = actor_gae_lambdas(
                    group.action_mask.sum(1), saved["gae_lambda_alpha"], saved.get("actor_gae_lambda")
                )
                advantages, _ = generalized_advantage_and_return_targets(
                    group.rewards, group.old_values, group.action_mask, lambdas
                )
                for member in range(samples):
                    slots, row_steps = aligned_row_steps(
                        steps, group_index * samples + member, group.emit_mask[member], decode_logprobs[member]
                    )
                    emitted = int(slots.numel())
                    token_ids = group.token_ids[member, slots + 1].cpu()
                    decode = decode_logprobs[member, slots].cpu()
                    replay_surprisal = -group.old_token_logprobs[member, slots].float().cpu()
                    mismatch.append((replay_surprisal + decode).abs())
                    row_advantages = advantages[member, slots].float().cpu()
                    reward = float(group.reward_scalar[member])
                    correct = reward >= 1.0
                    decode_surprisal = -decode
                    onset = derail_onset(decode_surprisal)
                    ended = emitted > 0 and int(token_ids[-1]) in run.stop_ids
                    ids = token_ids.tolist()
                    rows_out.append(
                        {
                            "arm": arm.label,
                            "prompt": start + group_index,
                            "source": source,
                            "sample": member,
                            "reward": reward,
                            "correct": correct,
                            "ended": ended,
                            "emitted": emitted,
                            "first_is_think": bool(ids) and ids[0] == think_id,
                            "think_opened": think_id in ids,
                            "think_closed": tokenizer.think_close_id in ids,
                            "onset": onset,
                            "onset_surprisal": None if onset is None else float(decode_surprisal[onset - DERAIL_WINDOW + 1 : onset + 1].max()),
                            "text_head": tokenizer.decode(ids[:48]),
                        }
                    )
                    tokens_out.append(
                        {
                            "steps": row_steps.half(),
                            "replay_surprisal": replay_surprisal.half(),
                            "advantage": row_advantages,
                            "value": group.old_values[member, slots].float().cpu(),
                        }
                    )
                    if emitted:
                        credit.append(credit_cells(replay_surprisal, row_advantages, correct))
            print(f"{arm.label}: chunk {chunk_index + 1} done, {len(rows_out)} rows", flush=True)
    gaps = torch.cat(mismatch) if mismatch else torch.zeros(1)
    meta = {"replay_vs_decode_surprisal": {"mean_abs": float(gaps.mean()), "max_abs": float(gaps.max())}}
    return meta, rows_out, tokens_out, credit


def summarize(rows: list[dict], tokens: list[dict], credit: list[dict]) -> dict:
    total = len(rows)

    def fraction(key, subset=None):
        chosen = rows if subset is None else subset
        return sum(bool(r[key]) for r in chosen) / max(len(chosen), 1)

    derailed = [r for r in rows if r["onset"] is not None]
    onsets = sorted(r["onset"] for r in derailed)
    field = {name: index for index, name in enumerate(STEP_FIELDS)}

    def profile(selector) -> dict:
        out = {}
        for low, high in zip(POSITION_EDGES[:-1], POSITION_EDGES[1:], strict=True):
            chunks = [t["steps"][low : int(min(high, t["steps"].shape[0]))].float() for r, t in zip(rows, tokens) if selector(r)]
            chunks = [c for c in chunks if c.numel()]
            if not chunks:
                continue
            stacked = torch.cat(chunks)
            out[f"{low}-{high}"] = {
                "tokens": stacked.shape[0],
                "entropy": float(stacked[:, field["entropy"]].mean()),
                "surprisal": float(-stacked[:, field["sampled_logprob"]].mean()),
                "junk_mass": float(stacked[:, field["junk_mass"]].mean()),
                "max_prob": float(stacked[:, field["max_prob"]].mean()),
            }
        return out

    first = torch.stack([t["steps"][0].float() for t in tokens if t["steps"].shape[0]])
    table = {}
    for outcome in OUTCOMES:
        cells = [c for c in credit if c["outcome"] == outcome]
        if not cells:
            continue
        summed = {name: [sum(c[name][i] for c in cells) for i in range(len(SURPRISAL_EDGES) - 1)] for name in cells[0] if name != "outcome"}
        table[outcome] = [
            {
                "bin": f"[{SURPRISAL_EDGES[i]}, {SURPRISAL_EDGES[i + 1]})",
                "tokens": int(summed["tokens"][i]),
                "advantage_mean": summed["advantage"][i] / max(summed["tokens"][i], 1),
                "advantage_rms": (summed["advantage_square"][i] / max(summed["tokens"][i], 1)) ** 0.5,
                "positive_fraction": summed["positive"][i] / max(summed["tokens"][i], 1),
                "dg_sum": summed["dg"][i],
                "pg_sum": summed["pg"][i],
            }
            for i in range(len(SURPRISAL_EDGES) - 1)
        ]
    by_source = {}
    for source in sorted({r["source"] for r in rows}):
        chosen = [r for r in rows if r["source"] == source]
        by_source[source] = {
            "rows": len(chosen),
            "ended": fraction("ended", chosen),
            "derailed": sum(r["onset"] is not None for r in chosen) / len(chosen),
            "accuracy": fraction("correct", chosen),
        }
    return {
        "rows": total,
        "accuracy": fraction("correct"),
        "ended": fraction("ended"),
        "emitted_mean": sum(r["emitted"] for r in rows) / max(total, 1),
        "first_is_think": fraction("first_is_think"),
        "think_closed": fraction("think_closed"),
        "derailed": len(derailed) / max(total, 1),
        "derailed_ended": fraction("ended", derailed) if derailed else None,
        "derailed_first_is_think": fraction("first_is_think", derailed) if derailed else None,
        "ended_rows_that_derailed": sum(r["onset"] is not None for r in rows if r["ended"]) / max(sum(r["ended"] for r in rows), 1),
        "onset_quantiles": {q: onsets[min(int(q * len(onsets)), len(onsets) - 1)] for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if onsets else None,
        "onset_after_1024": sum(o >= 1024 for o in onsets) / max(len(onsets), 1),
        "step0": {name: float(first[:, i].mean()) for i, name in enumerate(STEP_FIELDS)},
        "step0_think_probability": float(first[:, field["think_logprob"]].exp().mean()),
        "profile_all": profile(lambda r: True),
        "profile_not_derailed": profile(lambda r: r["onset"] is None),
        "profile_derailed": profile(lambda r: r["onset"] is not None),
        "credit": table,
        "by_source": by_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True, type=parse_arm)
    parser.add_argument("--checkpoint", required=True, help="base pretraining/SFT checkpoint the arms share")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--chunk-prompts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    labels = [arm.label for arm in args.arm]
    if len(set(labels)) != len(labels):
        parser.error("arm labels must be unique")
    if min(args.prompts, args.samples, args.chunk_prompts) < 1:
        parser.error("--prompts, --samples and --chunk-prompts must be positive")

    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = None
    mixture = None
    report = {"schema": SCHEMA, "args": {k: str(v) for k, v in vars(args).items()}, "arms": {}}
    report["constants"] = {
        "junk_probability": JUNK_PROBABILITY,
        "derail_window": DERAIL_WINDOW,
        "derail_nats": DERAIL_NATS,
        "surprisal_edges": [str(e) for e in SURPRISAL_EDGES],
        "position_edges": [str(e) for e in POSITION_EDGES],
    }
    for arm in args.arm:
        run = load_run(arm.path, args.checkpoint, device)
        saved = run.saved_args
        if mixture is None:
            mixture = saved["rl_mixture_manifest"]
            panel = panel_prompts(mixture, args.prompts, args.seed, run.answer_fence_ids is not None)
        elif saved["rl_mixture_manifest"] != mixture:
            raise ValueError(f"{arm.label}: arms must share one RL mixture")
        meta, rows, tokens, credit = run_arm(arm, run, panel, args, device)
        summary = summarize(rows, tokens, credit)
        report["arms"][arm.label] = {
            "path": str(arm.path),
            "sha256": file_sha256(arm.path),
            "step": run.step,
            "nomem": arm.nomem,
            "top_p": arm.top_p,
            "source_delightful_policy_gradient": bool(saved.get("delightful_policy_gradient")),
            **meta,
            "summary": summary,
        }
        (out_dir / f"{arm.label}_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        torch.save(tokens, out_dir / f"{arm.label}_tokens.pt")
        (out_dir / "report.json").write_text(json.dumps(report, indent=1))
        print(
            f"{arm.label}: acc {summary['accuracy']:.4f} ended {summary['ended']:.3f} "
            f"derailed {summary['derailed']:.3f} first<think> {summary['first_is_think']:.3f} "
            f"step0 entropy {summary['step0']['entropy']:.3f} p(<think>) {summary['step0_think_probability']:.3f} "
            f"emitted {summary['emitted_mean']:.0f}",
            flush=True,
        )
        del run
        torch.cuda.empty_cache()
    print(f"wrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()

"""Branch rollouts: Monte-Carlo ground truth for token-Q action gaps.

A token-Q critic sees one sampled action per state, so the difference
Q(s, a) - Q(s, a') between two actions at the same state is learned only by
generalizing across states. This tool measures that difference directly on
a frozen policy's held-out trajectories (``rollout_dataset``):

1. pick one think-span position t per sampled trajectory, among positions
   where the policy left some probability off the token it took
   (pi(a_t) <= ``--max-taken-probability``), so a real alternative exists;
2. draw the alternative a' from the policy with a_t excluded and
   renormalized;
3. from prefix + a_t and from prefix + a', roll ``--continuations``
   continuations each with the same frozen policy, temperature and budget
   the dataset was collected under, through the trainer's own pinned-EMIT
   decoder (``PinnedDecodeArena``: compiled, CUDA-graphed, counter-Gumbel
   sampling seeded per trajectory), and score each completed trajectory
   with the run's exact reward (``score_math_rollout``).

The two branch means estimate Q(s_t, a_t) and Q(s_t, a') under the frozen
policy; their difference is the gap a critic's u . (x(a) - x(a')) predicts.
Only token policies are supported (pinned EMIT, no carry): their state at t
is the token prefix alone, so a prefix re-prompted from position zero is
the same state the trajectory was in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from postraining.critic_study import think_action_mask

BRANCH_SCHEMA = "token_q_branch_rollouts/v1"


@dataclass(frozen=True)
class BranchPair:
    """One selected (trajectory, position) and its two actions."""

    pool: int
    group: int
    sample: int
    position: int
    taken_token: int
    alternative_token: int
    taken_logprob: float
    alternative_logprob: float


def think_positions(tokens: Sequence[int], think_open: int, think_close: int) -> np.ndarray:
    """Completion indices of the think actions ``critic_study`` credits."""
    return np.flatnonzero(think_action_mask(tokens, think_open, think_close))


def sample_alternative(probabilities: np.ndarray, taken: int, rng: np.random.Generator) -> int:
    """A token from the policy with the taken token excluded, renormalized."""
    weights = np.asarray(probabilities, dtype=np.float64).copy()
    weights[taken] = 0.0
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("the policy put all its mass on the taken token")
    return int(rng.choice(weights.size, p=weights / total))


def continuation_budget(position: int, max_new_tokens: int) -> int:
    """New tokens a branch at completion index ``position`` may still emit:
    the trajectory's own budget minus the prefix and the branch token."""
    return max_new_tokens - position - 1


def truncate_continuation(tokens: Sequence[int], budget: int, stop_ids: frozenset[int]) -> list[int]:
    """A continuation as the original rollout would have ended it: at its
    first stop token, and at most ``budget`` tokens."""
    kept = []
    for token in tokens[:budget]:
        kept.append(int(token))
        if token in stop_ids:
            break
    return kept


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_manifest_for(dataset: Path, binding: dict) -> dict:
    """The manifest of the run that recorded ``dataset``, hash-checked."""
    path = dataset.parent / "manifest.json"
    if sha256_file(path) != binding["run_manifest_sha256"]:
        raise ValueError(f"{path} is not the run manifest the dataset was bound to")
    return json.loads(path.read_text(encoding="utf-8"))


def branch_arena_width(pairs, pairs_per_batch: int, max_new_tokens: int) -> int:
    """Capacity for the actual left-padded batches, including their slack.

    A batch shares its longest forced prefix and largest remaining budget.
    Those maxima can come from different pairs, so the original trajectory
    length does not bound the physical stream width. Each pair is still
    scored after truncation to its own remaining token budget.
    """
    if not pairs or pairs_per_batch < 1:
        raise ValueError("arena sizing needs pairs and a positive batch size")
    width = 0
    for start in range(0, len(pairs), pairs_per_batch):
        chunk = pairs[start : start + pairs_per_batch]
        prefix = max(len(group.prompt_tokens) + pair.position + 1 for pair, group in chunk)
        budget = max(continuation_budget(pair.position, max_new_tokens) for pair, _ in chunk)
        width = max(width, prefix + budget)
    return width


def roll_branches(arena, prompts: Sequence[Sequence[int]], budget: int, continuations: int, seed: int, first_request: int) -> list[list[int]]:
    """``continuations`` emitted continuations of each branch prompt, at
    most ``budget`` tokens each, prompt-major; the branch prompts are
    requests ``first_request`` onward of the counter-Gumbel seed space."""
    import torch

    from postraining import latent_rollout as lr

    if budget < 1:
        # Every branch here sits on the budget's last token: nothing follows.
        return [[] for _ in range(len(prompts) * continuations)]
    width = max(len(prompt) for prompt in prompts)
    prompt_ids = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        prompt_ids[row, width - len(prompt) :] = torch.tensor(prompt)
    with torch.no_grad():
        rolled = arena.rollout(
            prompt_ids,
            torch.tensor([len(prompt) for prompt in prompts]),
            prompt_repeats=continuations,
            max_new_tokens=budget,
            max_stream_steps=budget,
            token_seeds=lr.trajectory_token_seeds(
                seed, range(first_request, first_request + len(prompts)), continuations, arena.device,
            ),
        )
    return lr.emitted_token_rows(rolled)


def score_completions(group, completions, tokenizer, stop_ids, run_args, think_fence_ids, answer_fence_ids) -> list[float]:
    """The run's exact reward of each completion of ``group``'s prompt.

    The completions are rebuilt as one rollout group and scored by the
    trainer's ``score_math_rollout``, whose terminal reward per row is the
    trajectory's reward. ``stop_ids`` keep the trainer's order: the first is
    the EOS that abstention detection keys on.
    """
    from postraining.core import answer_style
    from postraining.rollout_dataset import group_batch
    from postraining.train_latent_vapo import score_math_rollout

    batch = group_batch(group.prompt_tokens, completions, [0.0] * len(completions))
    score_math_rollout(
        batch, group.ground_truth, tokenizer, tuple(stop_ids),
        answer_style({"reward_model": {"style": group.answer_style}}),
        float(run_args["nearby_reward_max"]),
        think_fence_ids=think_fence_ids,
        min_think_tokens=int(run_args["think_min_tokens"]),
        answer_fence_ids=answer_fence_ids,
        abstention_penalty=run_args["abstention_penalty"],
    )
    return batch.reward_scalar.tolist()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=512)
    parser.add_argument("--continuations", type=int, default=32)
    parser.add_argument("--max-taken-probability", type=float, default=0.98)
    # The trainer's rollout width: 64 prompts x 16 samples.
    parser.add_argument("--rows-per-rollout", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args(argv)
    if args.pairs < 1 or args.continuations < 2:
        parser.error("--pairs must be positive and --continuations at least 2")
    if not 0.0 < args.max_taken_probability < 1.0:
        parser.error("--max-taken-probability must lie in (0, 1)")
    if args.rows_per_rollout < 2 * args.continuations:
        parser.error("--rows-per-rollout must hold both branches of a pair")

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    import train_gpt as baseline  # noqa: F401
    from postraining import latent_rollout as lr
    from postraining.core import load_posttraining_tokenizer
    from postraining.graph_decode import PinnedDecodeArena
    from postraining.latent_thought import LatentThoughtModel
    from postraining.model_io import load_model
    from postraining.provenance import capture_source_provenance
    from postraining.rollout_dataset import (
        dataset_digest,
        group_batch,
        read_manifest,
        read_pool,
    )
    from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters

    manifest = read_manifest(args.rollout_dataset)
    binding = manifest["binding"]
    checkpoint_sha256 = sha256_file(args.checkpoint)
    # A collection from a --curriculum-init policy binds that snapshot;
    # branches continue --checkpoint itself.
    if (binding["checkpoint_sha256"] != checkpoint_sha256
            or binding.get("policy_state_sha256") is not None):
        raise ValueError("the dataset was not collected from --checkpoint")
    if binding["reasoning_mode"] != "cot":
        raise ValueError("branch rollouts need a token-policy dataset")
    run_args = run_manifest_for(args.rollout_dataset, binding)["args"]
    if float(run_args["top_p"]) != 1.0:
        raise ValueError("branches resample the run's sampler; only top_p 1 is supported")
    # The study tests branch gaps exactly on binary success counts.
    if run_args["abstention_penalty"] is not None:
        raise ValueError("branch rollouts need binary rewards; the run had an abstention penalty")
    if float(run_args["nearby_reward_max"]) != 0.0:
        raise ValueError("branch rollouts need binary rewards; the run gave nearby partial credit")
    temperature = float(run_args["temperature"])
    max_new_tokens = int(run_args["resolved_train_max_new_tokens"])
    # Trainer order: stop_ids[0] is the EOS abstention detection keys on.
    stop_order = tuple(int(token) for token in binding["stop_ids"])
    stop_ids = frozenset(stop_order)
    args.output.mkdir(parents=True, exist_ok=False)
    source_provenance = capture_source_provenance(
        args.output / "provenance", Path(__file__).resolve().parents[1]
    )

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone = load_model(args.checkpoint, device, payload=payload)
    sft_args = ((payload.get("sft") or {}).get("args") or {}) if isinstance(payload, dict) else {}
    del payload
    backbone.eval()
    wrapper = LatentThoughtModel(backbone).to(device).eval()
    tokenizer = load_posttraining_tokenizer(
        backbone.architecture,
        FreshHyperparameters.tokenizer_path,
        think_tokens=bool(run_args["think_tokens"]),
        answer_tokens=bool(run_args["answer_fence"]),
        tokenizer_provenance=backbone.model_config.get("tokenizer_provenance"),
    )
    if bool(sft_args.get("think_tokens")) != bool(run_args["think_tokens"]):
        raise ValueError("the checkpoint and the run disagree on think tokens")
    think_fence_ids = (tokenizer.think_open_id, tokenizer.think_close_id) if run_args["think_tokens"] else None
    if think_fence_ids is None:
        raise ValueError("branch positions live in the think span; the run had none")
    answer_fence_ids = (tokenizer.answer_open_id, tokenizer.answer_close_id) if run_args["answer_fence"] else None

    def rewards(group, completions: list[list[int]]) -> list[float]:
        return score_completions(
            group, completions, tokenizer, stop_order, run_args, think_fence_ids, answer_fence_ids,
        )

    groups = [group for entry in manifest["pools"] for group in read_pool(args.rollout_dataset, entry)]
    rng = np.random.default_rng(args.seed)
    # One candidate position per trajectory, trajectories in a seeded order:
    # pairs come from distinct trajectories, spread over prompts.
    order = [(index, sample) for index, group in enumerate(groups) for sample in range(group.samples)]
    rng.shuffle(order)
    pairs: list[tuple[BranchPair, object]] = []
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    for index, sample in order:
        if len(pairs) == args.pairs:
            break
        group = groups[index]
        tokens = group.completions[sample]
        positions = think_positions(tokens, *think_fence_ids)
        if positions.size == 0:
            continue
        batch = group_batch(group.prompt_tokens, [tokens], [group.rewards[sample]]).to(device)
        with torch.no_grad(), autocast:
            beliefs, inputs = lr.replay_head_inputs(wrapper, batch)
            emit_index = lr.slot_index(batch.emit_mask.bool())[torch.as_tensor(positions, device=device)]
            features = wrapper.renderer_features(lr.compact_slots(inputs, emit_index), lr.compact_slots(beliefs, emit_index))
            logits = backbone.logits_from_features(features).float() / temperature
        probabilities = logits.softmax(-1).double().cpu().numpy()
        taken = np.asarray(tokens, dtype=np.int64)[positions]
        taken_probability = probabilities[np.arange(positions.size), taken]
        eligible = np.flatnonzero(taken_probability <= args.max_taken_probability)
        if eligible.size == 0:
            continue
        choice = int(rng.choice(eligible))
        alternative = sample_alternative(probabilities[choice], int(taken[choice]), rng)
        pairs.append(
            (
                BranchPair(
                    pool=group.pool, group=group.group, sample=sample,
                    position=int(positions[choice]),
                    taken_token=int(taken[choice]), alternative_token=alternative,
                    taken_logprob=float(np.log(probabilities[choice, taken[choice]])),
                    alternative_logprob=float(np.log(probabilities[choice, alternative])),
                ),
                group,
            )
        )
    if len(pairs) < args.pairs:
        raise RuntimeError(f"only {len(pairs)} eligible pairs exist; asked for {args.pairs}")

    # Each rollout batch is sized for its longest prefix plus its largest
    # remaining budget; neighbouring positions keep both close. Pair order
    # carries no meaning downstream: records are keyed by trajectory.
    pairs.sort(key=lambda item: item[0].position)
    branch_rewards: list[tuple[list[float], list[float]]] = []
    per_rollout = args.rows_per_rollout // (2 * args.continuations)
    # Size the physical shared streams, not each trajectory's logical
    # budget: left-padding and the chunk's max budget add separate slack.
    arena = PinnedDecodeArena(
        wrapper,
        rows=per_rollout * 2 * args.continuations,
        kv_width=branch_arena_width(pairs, per_rollout, max_new_tokens),
        temperature=temperature,
        stop_ids=stop_order,
        device=device,
        hidden_carry=False,
    )
    for start in range(0, len(pairs), per_rollout):
        chunk = pairs[start : start + per_rollout]
        prompts = []
        for pair, group in chunk:
            prefix = list(group.prompt_tokens) + list(group.completions[pair.sample][: pair.position])
            prompts += [prefix + [pair.taken_token], prefix + [pair.alternative_token]]
        budget = max(continuation_budget(pair.position, max_new_tokens) for pair, _ in chunk)
        # A branch prompt's identity is its position in the sorted pair list
        # and its arm.
        continuations = roll_branches(arena, prompts, budget, args.continuations, args.seed, 2 * start)
        for offset, (pair, group) in enumerate(chunk):
            prefix = list(group.completions[pair.sample][: pair.position])
            own_budget = continuation_budget(pair.position, max_new_tokens)
            arms = []
            for arm, token in enumerate((pair.taken_token, pair.alternative_token)):
                rows = range((2 * offset + arm) * args.continuations, (2 * offset + arm + 1) * args.continuations)
                if token in stop_ids:
                    # The branch ended on its own token: one deterministic outcome.
                    (outcome,) = rewards(group, [prefix + [token]])
                    arms.append([outcome] * args.continuations)
                    continue
                arms.append(rewards(group, [
                    prefix + [token] + truncate_continuation(continuations[row], own_budget, stop_ids)
                    for row in rows
                ]))
            branch_rewards.append((arms[0], arms[1]))
        print(f"branches {min(start + per_rollout, len(pairs))}/{len(pairs)}", flush=True)

    table = pa.Table.from_pylist(
        [
            {
                "pool": pair.pool, "group": pair.group, "sample": pair.sample,
                "prompt_sha256": group.prompt_sha256, "position": pair.position,
                "taken_token": pair.taken_token, "alternative_token": pair.alternative_token,
                "taken_logprob": pair.taken_logprob, "alternative_logprob": pair.alternative_logprob,
                "continuations": args.continuations,
                "original_reward": group.rewards[pair.sample],
                "taken_rewards": taken, "alternative_rewards": alternative,
                "taken_reward_mean": float(np.mean(taken)),
                "alternative_reward_mean": float(np.mean(alternative)),
            }
            for (pair, group), (taken, alternative) in zip(pairs, branch_rewards, strict=True)
        ]
    )
    path = args.output / "branches.parquet"
    pq.write_table(table, path)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": BRANCH_SCHEMA,
                "rollout_dataset_digest": dataset_digest(manifest),
                "checkpoint_sha256": checkpoint_sha256,
                "file": path.name,
                "sha256": sha256_file(path),
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "source_provenance": source_provenance,
                "source_sha256": {
                    name: sha256_file(Path(__file__).resolve().parent / name)
                    for name in ("branch_rollouts.py", "graph_decode.py", "latent_rollout.py", "train_latent_vapo.py")
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    gaps = np.array([np.mean(t) - np.mean(a) for t, a in branch_rewards])
    print(json.dumps({"pairs": len(pairs), "gap_mean": float(gaps.mean()), "gap_abs_mean": float(np.abs(gaps).mean())}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-pool actor gradients of MAC's estimators on a frozen policy and critic.

Run through mlq. Each pool of a rollout dataset is one real update's data
(64 prompts x 16 samples). For each pool and estimator this computes the
actor gradient the trainer's own ``update_minibatch`` forms (no step), and
saves it in fp32, so ``scripts/analyze_estimator_alignment.py`` can estimate
each estimator's alignment with the true policy gradient from cross-pool
inner products: pools are independent draws, so for an unbiased estimator U
and pools k != l, E[g_E^k . g_U^l] = E[g_E] . grad J.

``--source-root`` selects the code the estimators run under: the repository,
or a run's extracted source snapshot (``provenance/source.*.tar.gz``), so a
critic schema retired from the repository is read by the code that trained
it. The rollout-dataset reader is loaded from this repository by path; its
batch type is the snapshot's own (``LatentRolloutBatch`` is unchanged).

``all_action_nogain`` is all-action with every token-Q head's belief gain g
zeroed: the actor's advantage is linear in u, so all_action minus it is the
gain term's own gradient, the direction (g / r)(z - zbar).

``delightful_shuffled_head_gate`` is the head gate's placebo: its gate reads
the head's A(a_t) permuted within surprisal-and-span strata, seeded per pool
by ``pool_shuffle_seed(--gate-shuffle-seed, pool)`` so pools stay independent
draws; config.json records the seeds and strata edges.

``--entropy-gradient`` also saves, per pool, ``entropy_bonus``: the gradient
of minus the mean policy entropy over the same emit states with the MAC
weights (1 / emit count), in the same sign convention as the estimators'
losses. For a saved series g, g . g_entropy_bonus is the first-order entropy
change of a unit descent step along g: positive raises the entropy.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ESTIMATORS = ("sampled_baseline", "all_action", "control_variate", "all_action_nogain",
              "delightful_sampled_baseline", "delightful_td_gate", "delightful_head_gate",
              "delightful_shuffled_head_gate")
SHUFFLED = "delightful_shuffled_head_gate"
ENTROPY_SERIES = "entropy_bonus"


def pool_shuffle_seed(seed: int, pool: int) -> int:
    """The shuffled gate's per-pool seed: distinct for every (seed, pool)
    with pool < 2**16."""
    if not 0 <= pool < 1 << 16 or seed < 0:
        raise ValueError("shuffle seeds need a nonnegative seed and a pool below 65536")
    return (seed << 16) | pool


def entropy_bonus_gradient(wrapper, batch, plan, vocab_chunk: int) -> None:
    """Accumulate into the actor's .grad the gradient of minus the mean
    policy entropy over ``batch``'s emit states: the MAC actor pass's
    replay, shards, compact emit slots, chunked readout and 1 / emit-count
    weights, with the entropy in place of the estimator's loss. Run under
    the update's autocast."""
    import torch

    from postraining import latent_rollout

    denominator = batch.emit_mask.sum().clamp_min(1)
    for microbatch, shard in latent_rollout.iter_planned_replay_microbatches(batch, plan):
        # Looked up at call time: main swaps in the compiled replay.
        beliefs, inputs = latent_rollout.replay_head_inputs(wrapper, microbatch)
        emit_beliefs = latent_rollout.compact_slots(beliefs, shard.emit_index)
        emit_inputs = latent_rollout.compact_slots(inputs, shard.emit_index).detach()
        targets = latent_rollout.compact_next_slots(microbatch.token_ids, shard.emit_index)
        leaves = []
        for start in range(0, targets.numel(), vocab_chunk):
            end = start + vocab_chunk
            leaf = emit_beliefs[start:end].detach().requires_grad_(True)
            _, entropy = latent_rollout.compact_emit_token_logprobs_and_entropy(
                wrapper, emit_inputs[start:end], leaf, targets[start:end])
            (-entropy.sum() / denominator).backward()
            leaves.append(leaf)
        emit_beliefs.backward(torch.cat([leaf.grad for leaf in leaves]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-state", type=Path, default=None,
                        help="a run snapshot whose 'model' replaces the checkpoint's policy")
    parser.add_argument("--critic", type=Path, required=True,
                        help="a study checkpoint or run snapshot holding 'critic'")
    parser.add_argument("--q-head-hidden", type=int, default=1024)
    parser.add_argument("--rollouts", type=Path, required=True, help="a rollout_dataset directory")
    parser.add_argument("--pools", type=int, nargs="+", required=True)
    parser.add_argument("--estimators", nargs="+", choices=ESTIMATORS, required=True)
    parser.add_argument("--gate-shuffle-seed", type=int, default=None,
                        help=f"required with, and only with, {SHUFFLED}")
    parser.add_argument("--entropy-gradient", action="store_true",
                        help=f"also save each pool's {ENTROPY_SERIES} gradient")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.gate_shuffle_seed is not None) != (SHUFFLED in args.estimators):
        parser.error(f"--gate-shuffle-seed comes with, and only with, {SHUFFLED}")
    if args.gate_shuffle_seed is not None:
        for pool in args.pools:
            try:
                pool_shuffle_seed(args.gate_shuffle_seed, pool)
            except ValueError as error:
                parser.error(str(error))
    if len(set(args.estimators)) != len(args.estimators):
        parser.error("--estimators repeats an estimator")
    return args


def main():
    args = parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch

    import train_gpt  # noqa: F401  (import order: its patches load first)
    import postraining.latent_rollout as latent_rollout
    import postraining.mean_actor_critic as mean_actor_critic
    import postraining.train_latent_vapo as trainer
    from postraining.latent_thought import LatentThoughtModel
    from postraining.mean_actor_critic import nano_mean_actor_critic
    from postraining.model_io import copy_trunk, load_model
    from postraining.value_model import SeparateCritic
    from postraining.vapo.mixture import file_sha256

    if not Path(latent_rollout.__file__).resolve().is_relative_to(args.source_root.resolve()):
        raise RuntimeError(f"postraining resolved outside {args.source_root}: {latent_rollout.__file__}")
    spec = importlib.util.spec_from_file_location(
        "rollout_dataset_reader", REPO / "postraining" / "rollout_dataset.py")
    rollout_dataset = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve their annotations through sys.modules.
    sys.modules[spec.name] = rollout_dataset
    spec.loader.exec_module(rollout_dataset)

    manifest = rollout_dataset.read_manifest(args.rollouts)
    binding = manifest["binding"]
    # Each pool's rewards and log-probs belong to the policy that sampled
    # it: --checkpoint, or the --curriculum-init snapshot the dataset binds.
    policy_sha256 = None if args.policy_state is None else file_sha256(args.policy_state)
    if (binding["checkpoint_sha256"] != file_sha256(args.checkpoint)
            or binding.get("policy_state_sha256") != policy_sha256):
        raise ValueError(f"{args.rollouts} was not sampled by this policy")

    device = torch.device("cuda")
    # The trainer's numerics.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.fx.experimental._config.use_duck_shape = False
    torch._dynamo.config.cache_size_limit = 64

    base = load_model(args.checkpoint, device)
    wrapper = LatentThoughtModel(base).to(device).eval()
    if args.policy_state is not None:
        wrapper.load_state_dict(
            torch.load(args.policy_state, map_location="cpu", weights_only=False)["model"], strict=True)
    trainer.make_actor_trainable(wrapper)
    critic_state = torch.load(args.critic, map_location="cpu", weights_only=False)["critic"]
    twin = any(key.startswith("q_twin.") for key in critic_state)
    critic = SeparateCritic(copy_trunk(wrapper.backbone, device), q_head_hidden=args.q_head_hidden,
                            **({"q_twin": True} if twin else {})).to(device).eval()
    critic.load_state_dict(critic_state, strict=True)
    heads = [critic.q_head] + ([critic.q_twin] if twin else [])
    gains = [getattr(head, "belief_gain", None) for head in heads]
    if "all_action_nogain" in args.estimators and any(gain is None for gain in gains):
        raise ValueError("all_action_nogain needs a critic with a belief gain")

    # The drill's --compile-replay artifacts: the trunk passes compile around
    # the eager KDA chunk kernel, the readout tails in full.
    fullgraph = "_kda_" not in base.architecture
    replay = torch.compile(trainer.replay_head_inputs, mode="default", fullgraph=fullgraph, dynamic=True)
    trainer.replay_head_inputs = replay
    latent_rollout.replay_head_inputs = replay
    trainer.nano_mean_actor_critic = torch.compile(
        nano_mean_actor_critic, mode="default", fullgraph=True, dynamic=True)
    entry = "token_q_parts" if hasattr(critic, "token_q_parts") else "token_values"
    setattr(critic, entry, torch.compile(getattr(critic, entry), mode="default",
                                         fullgraph=fullgraph, dynamic=True))

    parameters = inspect.signature(trainer.update_minibatch).parameters
    optimizers = trainer.build_optimizers(
        wrapper, critic, learning_rate=0.0, trunk_optimizer="muon",
        muon_learning_rate=0.0, critic_muon_learning_rate=0.0, fused=True,
    )
    actor = {name: parameter for name, parameter in wrapper.named_parameters() if parameter.requires_grad}
    # The trainer's own actor Muon partition.
    muon_ids = {id(p) for group in optimizers["actor_muon"].param_groups for p in group["params"]}
    output = args.output / args.label
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps({
        "argv": sys.argv, "source_root": str(args.source_root.resolve()),
        "trainer": trainer.__file__, "twin": twin,
        "gains": [None if gain is None else float(gain) for gain in gains],
        "muon_parameters": [name for name, p in actor.items() if id(p) in muon_ids],
        "actor_parameters": {name: list(p.shape) for name, p in actor.items()},
        **({} if args.gate_shuffle_seed is None else {"gate_shuffle": {
            "seed": args.gate_shuffle_seed,
            "pool_seeds": {pool: pool_shuffle_seed(args.gate_shuffle_seed, pool)
                           for pool in args.pools},
            "surprisal_edges": list(mean_actor_critic.GATE_SHUFFLE_SURPRISAL_EDGES),
            "strata": "surprisal bin x think/answer span, per replay shard",
        }}),
    }, indent=2) + "\n")

    entries = {entry["pool"]: entry for entry in manifest["pools"]}
    records = []
    for pool in args.pools:
        groups = rollout_dataset.read_pool(args.rollouts, entries[pool])
        packed = latent_rollout.pack_rollout_groups_for_replay([
            rollout_dataset.group_batch(group.prompt_tokens, group.completions, group.rewards, 64)
            for group in groups])
        plan = latent_rollout.build_replay_plan(packed, 128, 16 * 1024 * 1024, 64, 24576,
                                               include_action_indices=True).to(device)
        for estimator in args.estimators:
            # A fresh device copy per estimator: the update writes behavior
            # statistics into its batch, and ``to`` keeps the layout token
            # the replay plan is bound to (a deepcopy would not).
            batch = packed.to(device)
            saved = [None if gain is None else gain.detach().clone() for gain in gains]
            if estimator == "all_action_nogain":
                with torch.no_grad():
                    for gain in gains:
                        gain.zero_()
            wrapper.zero_grad(set_to_none=True)
            critic.zero_grad(set_to_none=True)
            start = time.perf_counter()
            kwargs = dict(
                actor_step=False, critic_step=False,
                replay_max_trajectories=128, replay_attention_budget=16 * 1024 * 1024,
                replay_bucket=64, replay_slot_budget=24576, replay_plan=plan,
                mean_actor_critic=True,
                mac_estimator="all_action" if estimator == "all_action_nogain" else estimator,
                mac_vocab_chunk=512, think_close_id=50258, fused_behavior_statistics=True,
                # Critic targets only shape the critic's own gradient, which
                # is discarded; MC targets keep that pass identical across
                # critics.
                critic_lambda=1.0,
            )
            if estimator == SHUFFLED:
                kwargs["mac_gate_shuffle_seed"] = pool_shuffle_seed(args.gate_shuffle_seed, pool)
            missing = set(kwargs) - set(parameters)
            if missing:
                raise TypeError(f"{trainer.__file__} update_minibatch lacks {sorted(missing)}")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                metrics = trainer.update_minibatch(
                    wrapper, critic, batch, optimizers, **kwargs)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            with torch.no_grad():
                for gain, value in zip(gains, saved):
                    if gain is not None:
                        gain.copy_(value)
            # fp32: a difference of saved gradients (the gain term) can be a
            # small fraction of either, below bf16 rounding.
            gradient = {name: p.grad.detach().float().cpu() for name, p in actor.items()
                        if p.grad is not None}
            torch.save(gradient, output / f"{estimator}__pool{pool:02d}.pt")
            norm = float(sum(float(p.grad.float().square().sum()) for p in actor.values()
                             if p.grad is not None) ** 0.5)
            record = {"pool": pool, "estimator": estimator, "seconds": seconds, "gradient_norm": norm,
                      "trajectories": int(batch.kind.size(0)),
                      **{key: float(value) for key, value in metrics.items()
                         if isinstance(value, (int, float)) or (torch.is_tensor(value) and value.numel() == 1)}}
            records.append(record)
            print(json.dumps({k: record[k] for k in ("pool", "estimator", "seconds", "gradient_norm")}),
                  flush=True)
        if args.entropy_gradient:
            batch = packed.to(device)
            wrapper.zero_grad(set_to_none=True)
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                entropy_bonus_gradient(wrapper, batch, plan, 512)
            torch.cuda.synchronize()
            gradient = {name: p.grad.detach().float().cpu() for name, p in actor.items()
                        if p.grad is not None}
            torch.save(gradient, output / f"{ENTROPY_SERIES}__pool{pool:02d}.pt")
            record = {"pool": pool, "estimator": ENTROPY_SERIES,
                      "seconds": time.perf_counter() - start,
                      "gradient_norm": float(sum(float(g.square().sum())
                                                 for g in gradient.values()) ** 0.5),
                      "trajectories": int(batch.kind.size(0))}
            records.append(record)
            print(json.dumps(record), flush=True)
    (output / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


if __name__ == "__main__":
    main()

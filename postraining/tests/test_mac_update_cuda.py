"""MAC's compiled CUDA update against eager on the production KDA8 trunk.

The CPU suite pins the MAC gradient to a brute-force full-vocabulary
expectation. This pins the trainer's compiled path on the real SFT
checkpoint and a real cot rollout: compiled replay, compiled critic token
values and the compiled MAC readout give the eager update's actor and critic
gradients and behavior statistics, for one head, for twin heads with
clipped double-Q targets, for twins under the compiled pessimistic Q prior,
and for a live memory policy whose critic reads the actor's memory.

Run only through mlq: .venv/bin/python -m pytest -q -s this_file.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch

import postraining.latent_rollout as latent_rollout
import postraining.train_latent_vapo as trainer
from postraining.graph_decode import PinnedDecodeArena
from postraining.latent_rollout import (
    assign_terminal_rewards,
    build_replay_plan,
    compact_emit_token_logprobs,
    iter_planned_replay_microbatches,
    compact_next_slots,
    compact_slots,
    pack_rollout_groups_for_replay,
    replay_head_inputs,
    split_rollout_groups,
    trim_stream,
)
from postraining.latent_thought import LatentThoughtModel
from postraining.mean_actor_critic import nano_mean_actor_critic
from postraining.memory_carry import DEFAULT_MEMORY_HEAD_DIM, DEFAULT_MEMORY_HEADS
from postraining.rollout_report import RowVerdict
from postraining.tests.test_memory_carry import _live_memory
from postraining.value_model import SeparateCritic

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires queued CUDA"
)

CHECKPOINT = Path(
    "postraining/runs/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt"
)
STUDY_CRITIC = Path(
    "postraining/runs/dg_full_critic_convergence_20261015/actor_trunk/checkpoint.pt"
)
SAMPLES = 4


@pytest.fixture(autouse=True)
def _training_numerics():
    saved = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.get_float32_matmul_precision(),
        torch.fx.experimental._config.use_duck_shape,
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.fx.experimental._config.use_duck_shape = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = saved[0]
    torch.set_float32_matmul_precision(saved[1])
    torch.fx.experimental._config.use_duck_shape = saved[2]


def _actor_and_study_critic(device, twin=False, memory=False):
    from postraining.model_io import copy_trunk, load_model

    geometry = ({"memory_heads": DEFAULT_MEMORY_HEADS, "memory_head_dim": DEFAULT_MEMORY_HEAD_DIM}
                if memory else {})
    wrapper = LatentThoughtModel(
        load_model(CHECKPOINT, device), hidden_carry=memory, memory_carry=memory, **geometry,
    ).to(device).eval()
    trainer.make_actor_trainable(wrapper)
    if memory:
        # Reads at a real scale, so the replay scan is load-bearing.
        _live_memory(wrapper.combiner, 103)
    critic = SeparateCritic(
        copy_trunk(wrapper.backbone, device), q_head_hidden=1024, q_twin=twin,
        actor_memory=wrapper.combiner if memory else None, q_head_reads_actor_belief=memory,
    ).to(device).eval()
    payload = torch.load(STUDY_CRITIC, map_location="cpu", weights_only=False)
    state = payload["critic"]
    if memory:
        # The cot study's combiner is its unused carry module.
        state = {name: value for name, value in state.items() if not name.startswith("combiner.")}
    missing, unexpected = critic.load_state_dict(state, strict=False)
    # Trained weights for a parity test, not a v7 critic: the study fit
    # predates the linear direction path L and the belief gain g, which a
    # fresh head zeroes. Nonzero L and g put both through the compiled
    # update too; g at the scale a toy fit reached (NOTES 2026-10-20).
    linear = {"q_head.belief_direction.weight", "q_head.belief_gain"}
    assert not unexpected and linear <= set(missing)
    assert all(key in linear or (twin and key.startswith("q_twin."))
               or (memory and (key.startswith("combiner.") or ".actor_belief." in key))
               for key in missing)
    if memory:
        with torch.no_grad():
            critic.combiner.out.weight.normal_(std=0.5 / critic.combiner.out.weight.size(1) ** 0.5)
            critic.q_head.actor_belief.weight.normal_(
                std=critic.q_head.belief.weight.std().item())
    with torch.no_grad():
        direction = critic.q_head.direction.weight
        noise = torch.randn(critic.q_head.belief_direction.weight.shape,
                            generator=torch.Generator().manual_seed(7))
        critic.q_head.belief_direction.weight.copy_(direction.std() * noise.to(direction))
        critic.q_head.belief_gain.fill_(0.01)
    if twin:
        # A trained second head that disagrees with the first: the study
        # head with its direction readout perturbed at its own scale.
        critic.q_twin.load_state_dict(critic.q_head.state_dict())
        with torch.no_grad():
            direction = critic.q_twin.direction.weight
            noise = torch.randn(direction.shape, generator=torch.Generator().manual_seed(6))
            direction.add_(0.5 * direction.std() * noise.to(direction))
    return wrapper, critic


def _scored_rollout(wrapper, device, prompts=16, width=64, new_tokens=256):
    memory = wrapper.memory_carry
    generator = torch.Generator(device=device).manual_seed(2)
    lengths = torch.randint(24, width + 1, (prompts,), device=device, generator=generator).cpu()
    prompt_ids = torch.zeros((prompts, width), dtype=torch.long, device=device)
    for row, length in enumerate(lengths.tolist()):
        prompt_ids[row, width - length:] = torch.randint(
            1, 50000, (length,), device=device, generator=generator)
    arena = PinnedDecodeArena(
        wrapper, rows=prompts * SAMPLES, kv_width=width + new_tokens,
        temperature=1.0, stop_ids=(50256,), device=device, hidden_carry=memory,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        batch = arena.rollout(
            prompt_ids, lengths, prompt_repeats=SAMPLES,
            max_new_tokens=new_tokens, max_stream_steps=new_tokens,
            token_seeds=torch.arange(prompts * SAMPLES, device=device, dtype=torch.int64),
        )
    packed = pack_rollout_groups_for_replay([
        trim_stream(group)
        for group in split_rollout_groups(batch, SAMPLES, lengths.repeat_interleave(SAMPLES))
    ])
    rewards = torch.rand(packed.kind.size(0), generator=torch.Generator().manual_seed(4))
    assign_terminal_rewards(packed, (rewards < 0.5).float().to(device))
    # The answer-level estimators' group terms, as the scorer stamps them
    # per group: correct rows share one answer, wrong rows split between
    # two, a fixed quarter of them ungraded.
    stamps = [_group_terms(group, [
        RowVerdict(True, "right") if reward else
        RowVerdict(draw >= 0.25, "wrong" if draw < 0.6 else "other")
        for reward, draw in zip(group.reward_scalar.tolist(), draws.tolist())])
        for group, draws in zip(
            _row_groups(packed, SAMPLES),
            torch.rand(packed.kind.size(0) // SAMPLES, SAMPLES,
                       generator=torch.Generator().manual_seed(5)))]
    packed.group_advantage = torch.cat([advantage for advantage, _ in stamps])
    packed.answer_surprisal = torch.cat([surprisal for _, surprisal in stamps])
    return packed


def _row_groups(batch, size):
    """Row views of a packed batch's consecutive groups, enough for the
    scorer's group terms (rewards and row count)."""
    from types import SimpleNamespace

    return [SimpleNamespace(kind=batch.kind[start:start + size],
                            reward_scalar=batch.reward_scalar[start:start + size])
            for start in range(0, batch.kind.size(0), size)]


def _group_terms(group, verdicts):
    trainer.stamp_group_terms(group, verdicts)
    return group.group_advantage, group.answer_surprisal


def _update(wrapper, critic, batch, device, estimator, max_trajectories=16, pessimism=0.0):
    twin = critic.q_twin is not None
    plan = build_replay_plan(batch, max_trajectories, 16 * 1024 * 1024, 64, 24576,
                             include_action_indices=True).to(device)
    optimizers = trainer.build_optimizers(
        wrapper, critic, learning_rate=0.0, trunk_optimizer="muon",
        muon_learning_rate=0.0, critic_muon_learning_rate=0.0, fused=True,
    )
    wrapper.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        metrics = trainer.update_minibatch(
            wrapper, critic, batch, optimizers, actor_step=False, critic_step=False,
            replay_max_trajectories=max_trajectories, replay_attention_budget=16 * 1024 * 1024,
            replay_bucket=64, replay_slot_budget=24576, replay_plan=plan,
            mean_actor_critic=True, mac_estimator=estimator, mac_vocab_chunk=512,
            think_close_id=50258,
            # Twin heads bootstrap their targets, so lambda below one.
            fused_behavior_statistics=True, critic_lambda=0.5 if twin else 1.0,
            critic_q_pessimism=pessimism,
        )
    # The prior holds each shard's critic graph through the actor pass.
    print({"pessimism": pessimism, "max_trajectories": max_trajectories,
           "peak_gib": torch.cuda.max_memory_allocated() / 2**30})
    # MAC's behavior log-probs are the shared readout's on the same planned
    # shards and readout chunks (other bf16 shapes round differently).
    recorded, reference = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for microbatch, shard in iter_planned_replay_microbatches(batch, plan):
            # The update's replay ran with grad, and a compiled replay with
            # and without grad are two Inductor graphs: under the memory
            # carry they round apart (job 14274: beliefs 0.011 mean apart,
            # each deterministic), under cot they happen not to.
            beliefs, inputs = (
                x.detach() for x in trainer.replay_head_inputs(wrapper, microbatch))
            with torch.no_grad():
                emit_inputs = compact_slots(inputs, shard.emit_index)
                emit_beliefs = compact_slots(beliefs, shard.emit_index)
                emit_targets = compact_next_slots(microbatch.token_ids, shard.emit_index)
                # MAC's readout GEMM runs per vocab chunk of rows; bf16 GEMM
                # rounding depends on its shape, one ulp is ~0.03 at |logit| 4-8.
                reference.append(torch.cat([
                    compact_emit_token_logprobs(
                        wrapper, emit_inputs[start:start + 512],
                        emit_beliefs[start:start + 512], emit_targets[start:start + 512])
                    for start in range(0, emit_targets.numel(), 512)
                ]).float())
                recorded.append(compact_slots(microbatch.old_token_logprobs, shard.emit_index))
    gap = (torch.cat(recorded) - torch.cat(reference)).abs()
    print({"mac_vs_readout_max": float(gap.max()), "mac_vs_readout_nonzero": int((gap > 0).sum()),
           "tokens": gap.numel()})
    assert float(gap.max()) < 1e-5
    gradients = {
        f"{prefix}.{name}": parameter.grad.detach().float().clone()
        for prefix, module in (("actor", wrapper), ("critic", critic))
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }
    return metrics, gradients, batch.old_token_logprobs.clone()


@pytest.mark.skipif(not (CHECKPOINT.exists() and STUDY_CRITIC.exists()),
                    reason="needs the SFT checkpoint and the study critic")
@pytest.mark.parametrize(("estimator", "twin", "pessimism", "memory"), [
    ("all_action", False, 0.0, False), ("control_variate", False, 0.0, False),
    ("sampled_baseline", False, 0.0, False), ("all_action", True, 0.0, False),
    ("all_action", True, 1e-3, False),
    ("sampled_baseline", True, 0.0, True), ("all_action", True, 0.0, True),
    ("delightful_sampled_baseline", True, 0.0, True), ("delightful_td_gate", True, 0.0, True),
    # The head gate on the cot twin the alignment measures, and under memory.
    # The shuffled control has no parity case: its permutation follows the
    # shards, so the resharded floors would be another draw of it.
    ("delightful_head_gate", True, 0.0, False), ("delightful_head_gate", True, 0.0, True),
    # The answer-level drills' arms: memory, the twin study critic.
    ("rloo_sequence", True, 0.0, True), ("delightful_answer_gate", True, 0.0, True),
])
def test_compiled_mac_update_matches_eager(monkeypatch, estimator, twin, pessimism, memory):
    device = torch.device("cuda")
    wrapper, critic = _actor_and_study_critic(device, twin, memory)
    batch = _scored_rollout(wrapper, device)
    eager = _update(wrapper, critic, copy.deepcopy(batch), device, estimator, pessimism=pessimism)
    # The same eager update on other shards: bf16 rounding alone. Two
    # shardings, since one draw of that rounding understates its scale on
    # the small-remainder tensors (job 14238: a block 7 bias at 3.4% when
    # compiled, 1.5% resharded, while the whole actor gradient was 1.7% and
    # 1.8%).
    resharded = _update(wrapper, critic, copy.deepcopy(batch), device, estimator, 8, pessimism)
    quartered = _update(wrapper, critic, copy.deepcopy(batch), device, estimator, 4, pessimism)

    replay = torch.compile(replay_head_inputs, mode="default", fullgraph=False, dynamic=True)
    for module in (trainer, latent_rollout):
        monkeypatch.setattr(module, "replay_head_inputs", replay)
    monkeypatch.setattr(trainer, "nano_mean_actor_critic", torch.compile(
        nano_mean_actor_critic, mode="default", fullgraph=True, dynamic=True))
    # The trainer's compiled critic entry point.
    critic.token_q_parts = torch.compile(critic.token_q_parts, mode="default",
                                         fullgraph=False, dynamic=True)
    if pessimism:
        monkeypatch.setattr(trainer, "nano_pessimistic_q_prior", torch.compile(
            trainer.nano_pessimistic_q_prior, mode="default", fullgraph=True, dynamic=True))
    compiled = _update(wrapper, critic, copy.deepcopy(batch), device, estimator,
                       pessimism=pessimism)

    assert set(eager[1]) == set(compiled[1])
    actor_norm = sum(g.square().sum() for k, g in eager[1].items() if k.startswith("actor"))
    assert actor_norm > 0
    def flat(gradients):
        return torch.cat([g.flatten() for k, g in sorted(gradients.items()) if k.startswith("actor")])
    for label, other in (("compiled", compiled[1]), ("resharded", resharded[1]),
                         ("quartered", quartered[1])):
        print({f"{label}_actor_cosine": float(torch.nn.functional.cosine_similarity(
                   flat(other), flat(eager[1]), dim=0)),
               f"{label}_actor_relative": float((flat(other) - flat(eager[1])).norm()
                                                / flat(eager[1]).norm())})
    def cosine(got, want):
        return (float(torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0))
                if want.numel() > 1 and want.norm() > 0 else 1.0)
    floor = {name: max(float((other[1][name] - want).norm() / want.norm().clamp_min(1e-30))
                       for other in (resharded, quartered))
             for name, want in eager[1].items()}
    cosine_floor = {name: min(cosine(other[1][name], want) for other in (resharded, quartered))
                    for name, want in eager[1].items()}
    errors = {}
    for name, want in eager[1].items():
        got = compiled[1][name]
        scale = want.norm().clamp_min(1e-30)
        errors[name] = (float((got - want).norm() / scale), cosine(got, want))
    worst = sorted(errors.items(), key=lambda item: -item[1][0])[:8]
    print({"worst_relative_error": [(name, value, floor[name]) for name, value in worst],
           "worst_over_floor": max((value[0] / max(floor[name], 1e-30), name)
                                   for name, value in errors.items()),
           "min_cosine": min(errors.items(), key=lambda item: item[1][1])})
    # The control variate's sampled term sums large per-token pieces of both
    # signs, so its parameter gradients are small remainders and bf16
    # rounding moves them by several percent; eager on other shards measures
    # that floor (job 14161: 5.8% on the worst gain, compiled 8.1%).
    assert float(torch.nn.functional.cosine_similarity(
        flat(compiled[1]), flat(eager[1]), dim=0)) > 0.9995
    # Compiled kernels round the block 7 MLP bias, a sum of large
    # cancelling per-token pieces, about twice as coarsely as resharding
    # does, in every run: 1.75x its floor in job 14236 (5.3% against 3.0%),
    # 2.2x in 14239 (3.4% against 1.5% and 1.6% from two shardings), while
    # the whole actor gradient matched its floor both times. Three times
    # the floor bounds that ratio.
    for name, (relative, similarity) in errors.items():
        allowed = max(2e-2, 3 * floor[name])
        assert relative < allowed, name
        # The same allowance in angle: a cosine deficit grows as the
        # relative error squared, so three times the floor's error is nine
        # times its deficit. The floor's own deficit understates its error
        # when that error lies along the gradient, so the angle an error of
        # the allowed size could make at most, 1 - sqrt(1 - allowed^2),
        # bounds it too (job 14318: a 4-element memory write-gate bias at
        # 10.9% against a 3.7% floor whose deficit was only 2.9e-4).
        assert 1 - similarity < max(
            1e-3, 9 * (1 - cosine_floor[name]), 1 - math.sqrt(1 - min(allowed, 1.0) ** 2)
        ), name
    emit = batch.emit_mask.bool()
    drift = (compiled[2][emit] - eager[2][emit]).abs()
    # Compiled and eager bf16 KDA replays round differently along the stream.
    assert float(drift.mean()) < 2e-2
    print({"logprob_drift_mean": float(drift.mean()), "logprob_drift_max": float(drift.max())})
    keys = ("policy_loss", "mc_explained_variance", "advantage_std", "policy_entropy_sum")
    if twin:
        assert eager[0]["twin/min_shift_sum"] > 0
        keys += ("twin/gap_square_sum", "twin/min_shift_sum")
    if memory:
        keys += ("memory_write_beta_mean", "carry_input_delta_rms_ratio")
        assert eager[0]["carry_input_delta_rms_ratio"] > 0
    critic_norm = sum(g.square().sum() for k, g in eager[1].items() if k.startswith("critic"))
    print({"critic_gradient_norm": float(critic_norm.sqrt())})
    if pessimism:
        keys += ("q_prior/penalty_sum", "q_prior/positive_sum")
        print({key: eager[0][key] for key in ("q_prior/penalty_sum", "q_prior/positive_sum", "q_prior/count")})
    for key in keys:
        assert compiled[0][key] == pytest.approx(eager[0][key], rel=2e-2, abs=1e-6), key
    print({key: eager[0][key] for key in (
        "mc_explained_variance", "advantage_std", "emit_action_count",
        "advantage_group/span_answer/count")})

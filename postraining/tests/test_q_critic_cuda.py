"""The token Q critic's readout and the trust region on the compiled CUDA path.

The CPU suite pins the advantage against a brute-force sum and the step KL's
zero and quadratic scaling in fp32 CPU arithmetic. This pins what it cannot
reach, on the real KDA8 SFT checkpoint under the memory carry the Q runs
use, after a real Muon step at 12892's rates: the trainer's fp32 trust-region
readout (eager replay, compiled no-grad readout, TF32 matmuls) reads a zero
step as exactly zero and a backtracked step as the square of its fraction;
and the TF32 advantage GEMM against a float64 reference.

Run only through mlq: .venv/bin/python -m pytest -q -s this_file.
"""

from __future__ import annotations

import time
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
    compact_emit_token_logprobs_entropy_and_advantages,
    compact_next_slots,
    compact_slots,
    pack_rollout_groups_for_replay,
    replay_head_inputs,
    slot_index,
    split_rollout_groups,
    trim_stream,
)
from postraining.latent_thought import LatentThoughtModel
from postraining.nano_backbone import policy_centered_token_values
from postraining.value_model import SeparateCritic
from postraining.vapo.reppo import backtrack_parameters, snapshot_parameters

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires queued CUDA"
)

CHECKPOINT = Path(
    "postraining/runs/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt"
)
SAMPLES = 4


@pytest.fixture(autouse=True)
def _training_numerics():
    """The trainer's TF32 settings and shape specialization."""
    saved = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
        torch.fx.experimental._config.use_duck_shape,
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.fx.experimental._config.use_duck_shape = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = saved[0]
    torch.backends.cudnn.allow_tf32 = saved[1]
    torch.set_float32_matmul_precision(saved[2])
    torch.fx.experimental._config.use_duck_shape = saved[3]


def _actor_and_critic(device):
    from postraining.model_io import copy_trunk, load_model

    wrapper = LatentThoughtModel(
        load_model(CHECKPOINT, device),
        hidden_carry=True,
        memory_carry=True,
        memory_heads=4,
        memory_head_dim=64,
    ).to(device).eval()
    # The trainer's trainable contract: load_model freezes, the trainer thaws.
    trainer.make_actor_trainable(wrapper)
    generator = torch.Generator(device=device).manual_seed(1)
    memory = wrapper.combiner
    with torch.no_grad():
        for parameter in (
            memory.reader.out.weight,
            memory.writer.write_gate.weight,
            memory.writer.decay_gate.weight,
        ):
            parameter.copy_(
                0.02 * torch.randn(
                    parameter.shape, device=device, generator=generator
                )
            )
    critic = SeparateCritic(
        copy_trunk(wrapper.backbone, device),
        actor_memory=wrapper.combiner,
        q_head_hidden=1024,
        q_head_reads_actor_belief=True,
    ).to(device).eval()
    with torch.no_grad():
        critic.combiner.out.weight.normal_(std=0.02, generator=generator)
        critic.q_head.direction.weight.normal_(std=0.02, generator=generator)
        critic.q_head.state.weight.normal_(std=0.02, generator=generator)
    return wrapper, critic


def _scored_rollout(wrapper, device, prompts=16, width=64, new_tokens=192):
    generator = torch.Generator(device=device).manual_seed(2)
    lengths = torch.randint(
        24, width + 1, (prompts,), device=device, generator=generator
    ).cpu()
    prompt_ids = torch.zeros((prompts, width), dtype=torch.long, device=device)
    for row, length in enumerate(lengths.tolist()):
        prompt_ids[row, width - length :] = torch.randint(
            1, 50000, (length,), device=device, generator=generator
        )
    arena = PinnedDecodeArena(
        wrapper, rows=prompts * SAMPLES, kv_width=width + new_tokens,
        temperature=1.0, stop_ids=(50256,), device=device, hidden_carry=True,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        batch = arena.rollout(
            prompt_ids, lengths, prompt_repeats=SAMPLES,
            max_new_tokens=new_tokens, max_stream_steps=new_tokens,
            token_seeds=torch.arange(
                prompts * SAMPLES, device=device, dtype=torch.int64
            ),
        )
    packed = pack_rollout_groups_for_replay(
        [
            trim_stream(group)
            for group in split_rollout_groups(
                batch, SAMPLES, lengths.repeat_interleave(SAMPLES)
            )
        ]
    )
    rewards = torch.rand(packed.kind.size(0), generator=torch.Generator().manual_seed(4))
    assign_terminal_rewards(packed, (rewards < 0.3).float().to(device))
    return packed


def _compile_like_the_trainer(monkeypatch):
    replay = torch.compile(
        replay_head_inputs, mode="default", fullgraph=False, dynamic=True
    )
    readout = torch.compile(
        compact_emit_token_logprobs_entropy_and_advantages,
        mode="default", fullgraph=True, dynamic=True,
    )
    for module in (trainer, latent_rollout):
        monkeypatch.setattr(module, "replay_head_inputs", replay)
    monkeypatch.setattr(
        trainer, "compact_emit_token_logprobs_entropy_and_advantages", readout
    )


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the SFT checkpoint")
def test_fp32_step_kl_is_exactly_zero_without_a_step_and_quadratic_in_it(
    monkeypatch,
):
    device = torch.device("cuda")
    _compile_like_the_trainer(monkeypatch)
    # The trainer's matmul precision: the readout's TF32.
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        _step_kl_scaling(device)
    finally:
        torch.set_float32_matmul_precision(precision)


def _step_kl_scaling(device):
    wrapper, critic = _actor_and_critic(device)
    batch = _scored_rollout(wrapper, device)
    settings = dict(
        replay_max_trajectories=16, replay_attention_budget=16 * 1024 * 1024,
        replay_bucket=1, replay_slot_budget=24576,
    )
    plan = build_replay_plan(
        batch, 16, settings["replay_attention_budget"], 1, 24576,
        include_action_indices=True,
    ).to(device)
    assert len(plan.specs) >= 4
    # Run 12892's rates.
    optimizers = trainer.build_optimizers(
        wrapper, critic, learning_rate=2e-5, trunk_optimizer="muon",
        muon_learning_rate=4.8333e-5, critic_muon_learning_rate=4.8333e-5,
        fused=True,
    )
    actor_parameters = [
        parameter
        for name, optimizer in optimizers.items()
        if name == "actor" or name.startswith("actor_")
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    # The trainer's trust-region artifacts: eager replay, compiled readout.
    readout = torch.compile(
        compact_emit_token_logprobs, mode="default", fullgraph=True,
        dynamic=True,
    )
    report = {}

    def read(stride):
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Under the training autocast, as the trainer calls it.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logprobs = trainer.trust_region_token_logprobs(
                wrapper, batch, plan, shard_stride=stride,
                replay_function=replay_head_inputs,
                emit_logprob_function=readout,
            )
        torch.cuda.synchronize()
        report[f"read_seconds_stride{stride}"] = time.perf_counter() - started
        return logprobs

    with torch.autocast("cuda", dtype=torch.bfloat16):
        trainer.update_minibatch(
            wrapper, critic, batch, optimizers,
            delightful_policy_gradient=True, fused_behavior_statistics=True,
            entropy_temperature=1e-3, replay_plan=plan,
            actor_step=False, critic_step=False, **settings,
        )
    read(1)  # compile
    references = {stride: read(stride) for stride in (1, 4)}
    torch.cuda.reset_peak_memory_stats()
    base_allocated = torch.cuda.memory_allocated()
    unmoved = {
        stride: trainer.step_token_kl(references[stride], read(stride))
        for stride in (1, 4)
    }
    report["read_peak_over_live_gib"] = (
        torch.cuda.max_memory_allocated() - base_allocated
    ) / 2**30
    before = snapshot_parameters(actor_parameters)
    trainer.step_optimizers(optimizers, "actor")
    curve = {}
    for fraction in (1.0, 0.5, 0.25):
        if fraction < 1.0:
            backtrack_parameters(actor_parameters, before, 0.5)
        curve[fraction] = trainer.step_token_kl(references[1], read(1))
    sampled = trainer.step_token_kl(references[4], read(4))
    full = curve[1.0]["kl/post_update_token_pre_step"]
    report.update(unmoved=unmoved, curve=curve, sampled_quarter_step=sampled)
    report["half_over_full"] = curve[0.5]["kl/post_update_token_pre_step"] / full
    report["quarter_over_full"] = (
        curve[0.25]["kl/post_update_token_pre_step"] / full
    )
    print(report)
    for stride in (1, 4):
        assert unmoved[stride]["kl/post_update_token_pre_step"] == 0.0, report
        assert (
            unmoved[stride]["ratio/post_update_token_pre_step_abs_log_max"]
            == 0.0
        ), report
    assert full > 1e-6, report
    # Quadratic: 1/4 and 1/16, plus a TF32 floor of a few 1e-6 that shows
    # at the quarter step. bf16 read 0.8-0.9 at both (NOTES 2026-10-12).
    assert 0.2 < report["half_over_full"] < 0.3, report
    assert 0.04 < report["quarter_over_full"] < 0.1, report
    assert (
        0 < sampled["trust/measured_tokens"]
        < curve[0.25]["trust/measured_tokens"]
    )


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the SFT checkpoint")
def test_tf32_policy_centered_advantage_matches_float64():
    device = torch.device("cuda")
    wrapper, critic = _actor_and_critic(device)
    batch = _scored_rollout(wrapper, device, prompts=4)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        beliefs, inputs = replay_head_inputs(wrapper, batch)
        # A reads only the direction: the centring row cancels in Q(a) - E_pi Q.
        _, member_directions = critic.token_q_parts(batch)
        directions = member_directions.mean(0)
        latents = critic.action_latents()
        emit_index = slot_index(batch.emit_mask.bool())
        targets = compact_next_slots(batch.token_ids, emit_index)
        emit_inputs = compact_slots(inputs, emit_index)
        emit_beliefs = compact_slots(beliefs, emit_index)
        emit_directions = compact_slots(directions, emit_index)
        _, _, advantages, _ = compact_emit_token_logprobs_entropy_and_advantages(
            wrapper, emit_inputs, emit_beliefs, targets, emit_directions, latents
        )
        logits = wrapper.backbone.logits_from_features(
            wrapper.renderer_features(emit_inputs, emit_beliefs)
        )
    q = emit_directions.double() @ latents.double().T
    reference = q.gather(-1, targets[:, None]).squeeze(-1) - (
        logits.double().softmax(-1) * q
    ).sum(-1)
    unfused = policy_centered_token_values(
        logits, targets, emit_directions, latents
    )
    scale = reference.square().mean().sqrt()
    report = {
        "advantage_rms": float(scale),
        "fused_error_rms": float(
            (advantages.double() - reference).square().mean().sqrt() / scale
        ),
        "unfused_error_rms": float(
            (unfused.double() - reference).square().mean().sqrt() / scale
        ),
    }
    print(report)
    assert report["fused_error_rms"] < 1e-2, report


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the SFT checkpoint")
def test_q_readout_peak_memory_at_the_production_slot_budget():
    """The token Q readout's forward adds a (slots, vocab) fp32 score GEMM
    beside the entropy readout's: measure both, compiled, forward and
    backward, at run 12892's replay slot budget."""
    from postraining.latent_rollout import compact_emit_token_logprobs_and_entropy

    device = torch.device("cuda")
    wrapper, critic = _actor_and_critic(device)
    slots, dim = 24576, wrapper.backbone.model_dim
    generator = torch.Generator(device=device).manual_seed(5)
    inputs = torch.randn(slots, dim, device=device, generator=generator)
    beliefs = torch.randn(slots, dim, device=device, generator=generator)
    targets = torch.randint(50257, (slots,), device=device, generator=generator)
    directions = 0.02 * torch.randn(slots, dim, device=device, generator=generator)
    with torch.no_grad():
        latents = critic.action_latents()
    readouts = {
        "entropy": (
            torch.compile(
                compact_emit_token_logprobs_and_entropy,
                fullgraph=True, dynamic=True,
            ),
            (),
        ),
        "token_q": (
            torch.compile(
                compact_emit_token_logprobs_entropy_and_advantages,
                fullgraph=True, dynamic=True,
            ),
            (directions, latents),
        ),
    }
    report = {}
    for name, (readout, extra) in readouts.items():
        for attempt in range(2):
            wrapper.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = readout(
                    wrapper, inputs, beliefs.requires_grad_(), targets, *extra
                )
            (outputs[0].sum() + outputs[1].sum()).backward()
            torch.cuda.synchronize()
            report[f"{name}_seconds"] = time.perf_counter() - started
            report[f"{name}_peak_gib"] = (
                torch.cuda.max_memory_allocated() - base
            ) / 2**30
            del outputs
            beliefs.grad = None
    print(report)
    assert report["token_q_peak_gib"] < report["entropy_peak_gib"] + 6.0, report

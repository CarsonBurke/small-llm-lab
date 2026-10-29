"""VAPO over forced-initial stochastic latent-thought trajectories.

Every latent row samples one mandatory raw fp32 Gaussian THINK action before
its first emitted token. A Bernoulli continuation gate is active only while
thinking; STOP_AND_EMIT is irreversible. The raw action is stored exactly and
shared by live decode, parallel actor replay, and the separate critic.

``--thought-sigma`` is a vector-level isotropic magnitude: runtime width d
uses component std sigma/sqrt(d). The default DG and optional TPO token
estimators remain selected exactly as before; gate and Gaussian factors are
trained by the clipped VAPO path. ``--no-delightful-policy-gradient`` selects
the standard joint clipped VAPO actor.

``--reasoning-mode`` selects the policy family: ``latent`` (above), the
token-only controls ``cot`` and ``none``, and ``carry``, the deterministic
hidden carry. Carry is a token-only policy whose generated-token inputs add,
through a zero-initialized combiner, the detached post-final-norm belief that
produced that token; prompt tokens and the first generation input carry
nothing, there is no gate, Gaussian, or thought slot, and the stored beliefs
are replay constants (no BPTT). It starts from the SFT ``--checkpoint``
exactly like cot, and a zero-init carry rollout is bitwise the cot rollout.
- The critic is a SEPARATE from-scratch model (same architecture class,
  fresh weights, fully trainable, no SIGReg or latent prediction) trained
  purely by unclipped squared error on its scalar value head
  (``CRITIC_SCHEMA``). It re-derives combined embeddings with its own
  combiner weights, so credit lands on the same raw-thought or
  carried-belief inputs the actor conditions on. No parameter or gradient
  clip applies anywhere.
- No pretraining anchor: SIGReg and the latent target-prediction objective
  are dropped at RL time. The combiner trains purely on its ability to make
  its thought or carry inputs useful; the teacher-forced val-BPB guard is
  the drift detector.

Deferred by design: test-time-read-compute (carrying hiddens for read/prompt
tokens) is out of scope for the carry policy schema.

The default broad-v5 prompt cycle is stratified across DAPO-Math-17K,
DeepMind Mathematics, GSM8K train, and MBPP train. Rewards are binary exact
and grade only a structurally valid ``<answer>...</answer>`` span after at
least 33 tokens inside ``<think>...</think>``. AIME 2024 avg@k and held-out
DeepMind Mathematics are the evaluations. Prompt groups roll out and replay
separately because their lengths differ, then accumulate into the shared
actor/critic optimizer minibatch. ``--rollout-only`` reports whether rewards
vary within prompt groups before any update is attempted.

    python3 -m postraining.train_latent_vapo \
        --checkpoint ablation_results/<run>/pretraining_checkpoint.pt \
        --output postraining/runs/<name> [--rollout-only]
"""

from __future__ import annotations

import argparse
import faulthandler
import signal
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from checkpointing import RecoveryCheckpointPolicy, atomic_link_or_copy

import train_gpt as baseline
from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters
from pretraining.latent_moe_training import (
    apply_accumulated_quantile_balance,
    enable_quantile_balance_collection,
    reset_quantile_balance_accumulators,
)
import postraining.latent_rollout
from postraining.latent_eval import evaluate_latent_math
from postraining.benchmark_report import write_benchmark_report
from postraining.gate_transcripts import GateTranscriptWriter
from postraining.rollout_dataset import RolloutDatasetWriter
from postraining.rollout_report import (
    RolloutSampleRecorder,
    RowVerdict,
    capture_due,
    purge_rollout_samples_from,
    write_rollout_samples,
)
from postraining.core import (
    POSTTRAIN_REWARD_SCHEMA,
    JsonlLogger,
    POSTTRAIN_CONTEXT_TOKENS,
    ANSWER_CLOSE,
    ANSWER_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    answer_style,
    clipped_policy_loss,
    delightful_policy_loss,
    target_policy_loss,
    encode_prompt,
    extract_final_answer,
    deterministic_math_subset,
    fenced_answer_text,
    single_fence_span,
    structural_abstention,
    structural_format_ok,
    generalized_advantage_and_return_targets,
    generalized_value_estimate_targets,
    action_value_advantages,
    actor_gae_lambdas,
    temporal_difference_residuals,
    load_posttraining_tokenizer,
    load_unique_math_rows,
    math_corpus_identity,
    modal_answer_baseline,
    module_answer_baselines,
    nearby_numeric_reward,
    parse_numeric_answer,
    validate_posttraining_context_budget,
    verify_answer,
)
from postraining.vapo.reppo import (
    EntropyTemperature,
    enforce_kl_bound,
    snapshot_parameters,
)
from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    LatentRolloutBatch,
    ReplayPlan,
    WriteNoisePlan,
    assign_terminal_rewards,
    build_replay_plan,
    compact_emit_token_log_odds,
    compact_emit_token_logprobs,
    compact_emit_token_logprobs_and_entropy,
    compact_emit_token_logprobs_entropy_and_advantages,
    compact_next_slots,
    compact_slots,
    compact_stream_to_device,
    compact_thought_actions,
    emitted_token_rows,
    generated_slot_mask,
    pack_rollout_groups_for_replay,
    iter_planned_replay_microbatches,
    recorded_write_noise,
    refresh_old_statistics,
    replay_head_inputs,
    replay_policy_action_offsets,
    scatter_replay_statistics,
    select_trajectory_rows,
    scatter_slots,
    slot_index,
    rollout_continuations,
    split_rollout_groups,
    think_slot_mask,
    trajectory_token_seeds,
    trajectory_used_lengths,
    trim_stream,
    validate_carry_policy,
    validate_hidden_carry_rollout,
)
from postraining.graph_decode import PinnedDecodeArena
from postraining.memory_eval_ablation import (
    FOREIGN_READ_ABLATIONS,
    MEMORY_EVAL_ABLATION_SCHEMA,
    memory_eval_ablation,
)
from postraining.perturbation_sweep import (
    PERTURBATION_SWEEP_SCHEMA,
    WRITE_NOISE_SWEEP_SCHEMA,
    SweepConfig,
    SweepWriter,
    WriteNoiseSweepConfig,
    run_sweep,
    run_write_noise_sweep,
    target_parameters,
    tensor_rms,
)
from postraining.latent_thought import (
    EMIT,
    THINK,
    RENDERER_FEATURES_SCHEMA,
    DecodeRangeMask,
    LatentThoughtModel,
    combiner_init_kwargs_from_checkpoint,
    rollout_policy_schema_for_mode,
    validate_renderer_checkpoint,
)
from postraining.math_prompt import (
    ANSWER_FENCE_PROMPT_SCHEMA,
    canonicalize_answer_fence_rows,
)
from postraining.rollout_scheduler import (
    ContinuousScheduleStats,
    rollout_continuous_refill_groups,
    warmup_decode_width_buckets,
)
from postraining.model_io import copy_trunk, fresh_trunk, load_model
from postraining.memory_carry import MemoryReader, memory_write_mask
from postraining.muon import Muon
from postraining.reasoning_modes import (
    mode_carries_hidden,
    mode_carries_memory,
    mode_pins_emit,
    mode_rollout_budget,
    training_rollout_budget,
)
from postraining.provenance import capture_source_provenance
from postraining.runtime.profiling import (
    PROFILE_SCHEMA,
    DisabledProfiler,
    RunProfiler,
    device_timed,
    elapsed_seconds,
)
from postraining.train_vapo import prompt_text
from postraining.choice_prompt import RENDERED_LABELS
from postraining.binary_q import (
    binary_q_delightful_policy_loss, exact_prefix_success_constraints,
)
from postraining.value_model import SeparateCritic, next_slot_tokens, policy_centered_q
from postraining.mean_actor_critic import (
    CRITIC_GATED_ESTIMATORS, DELIGHTFUL_MAC_ESTIMATORS, SAMPLED_ONLY_ESTIMATORS,
    SEQUENCE_ESTIMATORS, SUPPORT_CENSUS_FIELDS, answer_cluster_labels,
    answer_cluster_surprisal, answer_gate, delight_gate, delight_inputs,
    gate_shuffle_strata, leave_one_out_advantages, nano_mean_actor_critic,
    nano_target_logprobs_and_advantages, sampled_coefficients, sequence_coefficients,
    shuffle_within_strata, validate_mac_estimator, validate_softcap,
)
from postraining.pessimistic_q import nano_pessimistic_q_prior
from postraining.vapo.config import (
    build_arg_parser,
    normalized_abstention_reward,
    resolve_rollout_defaults,
    validate_args,
)
from postraining.vapo.code_reward import (
    PYTHON_REWARD_SCHEMA,
    PYTHON_PARTIAL_REWARD_SCHEMA,
    PYTHON_RESULT_CODES,
    batch_python_test_results,
    batch_python_test_scores,
)
from postraining.vapo.mixture import (
    MixedPromptSampler,
    file_sha256,
    load_mixture_manifest,
    mixture_identity,
)
from postraining.vapo.schemas import (
    CRITIC_ACTION_HEAD_SCHEMA,
    CRITIC_SCHEMA,
    critic_schema_for,
    PROMPT_ORDER_SCHEMA,
    actor_objective_schema,
    execution_schema_for,
    optimizer_schema_for_trunk_optimizer,
    resume_execution_schema_compatible,
    replay_numerics_schema_for,
    resume_replay_schema_compatible,
    resume_critic_schema_compatible,
)


# Refreshed device minibatches held for their update instead of the
# scatter-to-CPU/repack/re-upload round trip. Bounded because packed batch
# bytes track the pool's longest stream and, under the hidden carry, each
# slot stores a full-width carried belief; past the budget the refresh loop
# falls back to the scatter path for the remaining minibatches.
# VRAM-resident only between a pool's refresh and its last update — never
# across a collection.
RETAINED_MINIBATCH_BUDGET_BYTES = 8 << 30
# The critic initializes from its own RNG stream, --seed plus this offset,
# so every reasoning mode starts from the same critic at one seed (main).
CRITIC_INIT_SEED_OFFSET = 0x5EED_C817
# v2: the grading style follows each row's reward_model.style (Minerva for
# DAPO/AIME lineage data, official exact match for mathematics_dataset rows)
# instead of Minerva-normalizing everything.
REWARD_SCHEMA = POSTTRAIN_REWARD_SCHEMA
ABSTENTION_REWARD_SCHEMA = f"{REWARD_SCHEMA}+normalized_eos_abstention/v4"


def reward_schema_for_abstention(penalty: float | None) -> str:
    return REWARD_SCHEMA if penalty is None else ABSTENTION_REWARD_SCHEMA


SOURCE_SIGNAL_MASK_SCHEMA = "per_source_zero_reward_actor_mask/v1"
RESUME_ARG_CONTRACT_SCHEMA = "vapo_exact_environment_objective_args/v3"
RESUME_EXACT_ARG_FIELDS = (
    "reasoning_mode",
    "answer_tokens",
    "prompt_tokens",
    "continuation_tokens",
    "samples_per_prompt",
    "ppo_epochs",
    "temperature",
    "top_p",
    "combined_mlp_blocks",
    "combined_mlp_hidden",
    "memory_heads",
    "memory_head_dim",
    "thought_sigma",
    "init_stop_thinking_probability",
    "gae_lambda_alpha",
    "actor_gae_lambda",
    "nearby_reward_max",
    "abstention_penalty",
    "think_tokens",
    "think_min_tokens",
    "answer_fence",
    "source_success_actor_gate",
    "zero_reward_actor_freeze",
    "value_warmup_steps",
    "rollout_tail_batch",
    "rollout_compile",
    "rollout_tail_graph",
    "rollout_flex_decode",
    "rollout_graph_decode",
    "compile_replay",
    "duck_shape",
    "replay_bucket",
    "replay_max_trajectories",
    "replay_attention_budget",
    "replay_slot_budget",
    "rollout_sync_every",
    "rollout_compact_dead_ratio",
    "rollout_groups",
    "rollout_scheduler",
    "consume_all_prompts",
    "seed",
)
TPO_RESUME_EXACT_ARG_FIELDS = (
    "target_policy_optimization",
    "tpo_eta",
)
# Checked only when either side has an action head, so runs from before the
# head resume unchanged.
ACTION_HEAD_RESUME_EXACT_ARG_FIELDS = (
    "critic_action_head_hidden",
    "value_warmup_scope",
)
# Checked only when either side uses one, likewise. The temperature's own
# state resumes from the checkpoint; its target and rate are the contract.
POST_ACTION_Q_RESUME_EXACT_ARG_FIELDS = (
    "critic_post_action_q_hidden", "critic_post_action_q_prior",
    "q_critic_lambda", "actor_gae_lambda",
)
Q_CRITIC_RESUME_EXACT_ARG_FIELDS = (
    "critic_q_head_hidden",
    "q_critic_lambda",
)
ENTROPY_TEMPERATURE_RESUME_EXACT_ARG_FIELDS = (
    "entropy_target",
    "entropy_temperature_init",
    "entropy_temperature_learning_rate",
)
KL_TRUST_REGION_RESUME_EXACT_ARG_FIELDS = (
    "actor_kl_bound",
    "actor_kl_backtracks",
    "actor_kl_shard_stride",
)


def combiner_weight_telemetry(
    wrapper: LatentThoughtModel, critic: SeparateCritic | None = None
) -> dict:
    """RMS of the fresh input module's zero-initialized outputs.

    Leaving zero is the first sign the carried information is being used:
    the carry map and type bias of the hidden carry, the read projection of
    the memory carry (with its query, write and gate projections for scale,
    and the critic's own memory reader and Q-head belief inputs).
    """
    rms = lambda tensor: tensor.detach().square().mean().sqrt()
    combiner = wrapper.combiner
    if wrapper.memory_carry:
        values = {
            "memory/read_out_weight_rms": rms(combiner.reader.out.weight),
            "memory/query_weight_rms": rms(combiner.reader.query.weight),
            "memory/key_weight_rms": rms(combiner.writer.key.weight),
            "memory/value_weight_rms": rms(combiner.writer.value.weight),
            "memory/write_gate_weight_rms": rms(combiner.writer.write_gate.weight),
            "memory/decay_gate_weight_rms": rms(combiner.writer.decay_gate.weight),
        }
        if critic is not None and isinstance(critic.combiner, MemoryReader):
            values["memory/critic_read_out_weight_rms"] = rms(critic.combiner.out.weight)
            values["memory/critic_query_weight_rms"] = rms(critic.combiner.query.weight)
        beliefs = [] if critic is None or critic.q_head is None else [
            head.actor_belief.weight for head in critic.q_heads
            if head.actor_belief is not None
        ]
        if beliefs:
            values["memory/critic_q_actor_belief_weight_rms"] = torch.stack(
                [rms(weight) for weight in beliefs]
            ).mean()
        return values
    return {
        "combiner/carry_weight_rms": rms(combiner.carry.weight),
        "combiner/type_bias_rms": rms(combiner.type_bias),
    }


def validate_resume_arg_contract(saved: dict, current: argparse.Namespace) -> None:
    saved_exclusions = set(filter(None, saved.get("exclude_rl_sources", "").split(",")))
    current_exclusions = set(filter(None, getattr(current, "exclude_rl_sources", "").split(",")))
    if not saved_exclusions <= current_exclusions:
        raise ValueError("resume cannot re-enable excluded RL sources")
    saved_rollout = int(saved.get("prompts_per_rollout", -1))
    saved_minibatch = int(saved.get("prompts_per_minibatch", -1))
    current_rollout = int(current.prompts_per_rollout)
    current_minibatch = int(current.prompts_per_minibatch)
    topology_changed = (saved_rollout, saved_minibatch) != (
        current_rollout,
        current_minibatch,
    )
    if topology_changed:
        if not (
            (
                bool(saved.get("delightful_policy_gradient"))
                or bool(saved.get("target_policy_optimization"))
            )
            and (
                bool(getattr(current, "delightful_policy_gradient", False))
                or bool(getattr(current, "target_policy_optimization", False))
            )
            and saved_rollout == saved_minibatch
            and current_rollout == current_minibatch
        ):
            raise ValueError(
                "exact resume changed rollout/minibatch topology outside the "
                "one-fresh-batch Delightful regime: checkpoint="
                f"{saved_rollout}/{saved_minibatch}, current="
                f"{current_rollout}/{current_minibatch}"
            )
        if not bool(getattr(current, "allow_dg_topology_migration", False)):
            raise ValueError(
                "exact Delightful resume changed rollout/minibatch topology "
                f"from {saved_rollout}/{saved_minibatch} to "
                f"{current_rollout}/{current_minibatch}; pass "
                "--allow-dg-topology-migration to acknowledge the change"
            )
    exact_fields = RESUME_EXACT_ARG_FIELDS
    if getattr(current, "allow_rollout_groups_migration", False):
        # validate_args admits the flag only for counter-keyed draws, whose
        # sampling law does not depend on chunking.
        exact_fields = tuple(
            name for name in exact_fields if name != "rollout_groups"
        )
    if (
        bool(saved.get("target_policy_optimization"))
        or bool(getattr(current, "target_policy_optimization", False))
    ):
        exact_fields += TPO_RESUME_EXACT_ARG_FIELDS
    if (
        saved.get("critic_action_head_hidden") is not None
        or getattr(current, "critic_action_head_hidden", None) is not None
    ):
        exact_fields += ACTION_HEAD_RESUME_EXACT_ARG_FIELDS
    if saved.get("value_warmup_rewind") or getattr(
        current, "value_warmup_rewind", False
    ):
        # Where the first actor step starts in the prompt stream.
        exact_fields += ("value_warmup_rewind",)
    if saved.get("mean_actor_critic") or getattr(current, "mean_actor_critic", False):
        exact_fields += ("mean_actor_critic", "mac_estimator")
    if saved.get("critic_q_twin") or getattr(current, "critic_q_twin", False):
        exact_fields += ("critic_q_twin",)
    if saved.get("critic_q_pessimism") or getattr(current, "critic_q_pessimism", 0.0):
        exact_fields += ("critic_q_pessimism",)
    if "eval_top_p" in saved:
        # The periodic evaluation protocol, which a bench stop also compares;
        # runs from before the flags evaluated at the then-constant defaults.
        exact_fields += ("eval_temperature", "eval_top_p")
    for trigger, fields in (
        ("critic_q_direction_learning_rate", ("critic_q_direction_learning_rate",)),
        ("critic_q_head_hidden", Q_CRITIC_RESUME_EXACT_ARG_FIELDS),
        ("critic_post_action_q_hidden", POST_ACTION_Q_RESUME_EXACT_ARG_FIELDS),
        ("entropy_target", ENTROPY_TEMPERATURE_RESUME_EXACT_ARG_FIELDS),
        ("actor_kl_bound", KL_TRUST_REGION_RESUME_EXACT_ARG_FIELDS),
    ):
        if (
            saved.get(trigger) is not None
            or getattr(current, trigger, None) is not None
        ):
            exact_fields += fields
    mismatches = [
        (name, saved.get(name), getattr(current, name))
        for name in exact_fields
        if saved.get(name) != getattr(current, name)
    ]
    for name in ("resolved_train_max_new_tokens", "resolved_train_max_stream_steps"):
        if name in saved and saved[name] != getattr(current, name, None):
            mismatches.append((name, saved[name], getattr(current, name, None)))
    if mismatches:
        details = ", ".join(
            f"{name}: checkpoint={before!r}, current={after!r}"
            for name, before, after in mismatches
        )
        raise ValueError(
            "exact resume changed environment/objective arguments: " + details
        )


CRITIC_INIT_PROVENANCE_FIELDS = (
    "init", "checkpoint", "source_execution_schema", "checkpoint_sha256",
    "source_actor_step", "source_value_warmup_step", "optimizer_state",
    "source_reward_schema", "value_contract_changes",
    "value_change_resolution", "action_head_init",
)


STUDY_CRITIC_SCHEMA = "token_q_critic_study_checkpoint/v2"


STUDY_COLLECTION_CONTRACT = (
    "reasoning_mode", "temperature", "top_p", "python_reward_mode",
    "nearby_reward_max", "abstention_penalty", "think_min_tokens",
    "answer_fence", "prompt_tokens", "continuation_tokens",
)


def validate_study_critic_init(payload: dict, critic: SeparateCritic,
                               args: argparse.Namespace,
                               data_identity: str | None = None) -> None:
    """Accept a token Q critic fit by ``critic_study`` against this base actor.

    The study fits a token-weighted, prompt-context, actor-initialized critic
    whose direction readout steps at the head rate over its width
    (``inverse_dim``); this run continues those rates. With
    ``data_identity`` the study's rollout collection must also share this
    run's prompts, sampler and reward contract, so its Q values the same
    returns.
    """
    if payload.get("schema") != STUDY_CRITIC_SCHEMA:
        raise ValueError(f"--study-critic-init needs a {STUDY_CRITIC_SCHEMA} checkpoint")
    if payload.get("checkpoint_sha256") != args.base_checkpoint_sha256:
        raise ValueError("study critic was fit against another base checkpoint")
    arm = payload["arm"]
    if (arm.get("context"), arm.get("critic_init"), arm.get("direction_scale"),
            arm.get("weighting")) != ("prompt", "actor", "inverse_dim", "token"):
        raise ValueError(
            "study critic must be a token-weighted, prompt-context, "
            "actor-initialized, inverse_dim arm"
        )
    if bool(arm.get("q_twin", False)) != args.critic_q_twin:
        raise ValueError(
            "a twin study critic initializes exactly a --critic-q-twin run, "
            "and a single-head one exactly a single-head run"
        )
    if payload.get("critic_schema") != critic_schema_for(True, twin_q=args.critic_q_twin):
        raise ValueError(
            "study critic was fit under another token Q contract, such as the "
            "input-latent action table; its directions do not transfer"
        )
    if arm["train_belief_gain"] is not True:
        raise ValueError(
            "study critic froze the token Q's belief gain, which this run trains"
        )
    if args.critic_init != "actor":
        raise ValueError("an actor-initialized study critic requires --critic-init actor")
    if payload["config"]["q_head_hidden"] != args.critic_q_head_hidden:
        raise ValueError("study critic head width differs from --critic-q-head-hidden")
    if int(payload["step"]) < 1:
        raise ValueError("study checkpoint holds an untrained critic")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in payload["critic"].values()
               if tensor.is_floating_point()):
        raise ValueError("study critic has non-finite parameters")
    head_rate = (args.critic_action_head_learning_rate
                 if args.critic_action_head_learning_rate is not None
                 else args.critic_learning_rate)
    direction_rate = arm["head_learning_rate"] / critic.q_head.direction.out_features
    if head_rate != arm["head_learning_rate"] or (
        args.critic_q_direction_learning_rate != direction_rate
    ) or args.critic_learning_rate != arm["trunk_learning_rate"] or (
        args.critic_muon_learning_rate != arm["muon_learning_rate"]
    ):
        raise ValueError(
            "a study critic continues at its study's rates: head "
            f"{arm['head_learning_rate']}, direction {direction_rate}, trunk "
            f"{arm['trunk_learning_rate']}, muon {arm['muon_learning_rate']}"
        )
    if data_identity is None:
        return
    collection = json.loads(
        (Path(payload["config"]["train_dataset"]).parent / "manifest.json").read_text()
    )
    if collection.get("math_data_identity") != data_identity:
        raise ValueError("study critic was fit on rollouts of other prompts")
    # A memory policy's read is zero-initialized, so at step 0 it is bitwise
    # the cot policy whose rollouts a cot study critic valued.
    expected = {name: getattr(args, name, None) for name in STUDY_COLLECTION_CONTRACT}
    if mode_carries_memory(args.reasoning_mode):
        expected["reasoning_mode"] = "cot"
    changed = [
        name for name in STUDY_COLLECTION_CONTRACT
        if collection["args"].get(name) != expected[name]
    ]
    if changed:
        raise ValueError(f"study critic's rollouts used another contract: {changed}")


def load_study_critic(critic: SeparateCritic, state: dict[str, torch.Tensor]) -> None:
    """Load a study critic's weights, which a study fits on cot rollouts.

    A memory run's critic adds two inputs the cot critic lacks: its own
    reader of the actor's memory, and each Q head's projection of the
    actor's belief. The reader's output starts at zero by construction and
    the projections are zeroed here, so the loaded critic values every cot
    stream exactly as the study's did, which a memory policy is at step 0.
    The study's combiner is the cot critic's unused carry module and is
    dropped.
    """
    if critic.actor_memory_writer is None:
        critic.load_state_dict(state, strict=True)
        return
    memory_only = {name for name in critic.state_dict()
                   if name.startswith("combiner.") or ".actor_belief." in name}
    missing, unexpected = critic.load_state_dict(
        {name: value for name, value in state.items() if not name.startswith("combiner.")},
        strict=False,
    )
    if unexpected or set(missing) != memory_only:
        raise ValueError(
            "a study critic initializes a memory critic outside its memory inputs: "
            f"missing {sorted(set(missing) - memory_only)}, unexpected {sorted(unexpected)}"
        )
    with torch.no_grad():
        for head in critic.q_heads:
            if head.actor_belief is not None:
                head.actor_belief.weight.zero_()
    if bool(critic.combiner.out.weight.any()):
        raise ValueError("the memory critic's reader must start at zero output")


def validate_critic_only_init(payload: dict, args: argparse.Namespace,
                              data_identity: str) -> None:
    """Accept critic weights with the same value task, independent of actor loss."""
    if args.value_warmup_steps != 0:
        raise ValueError("--critic-only-init requires --value-warmup-steps 0")
    if (float(payload.get("args", {}).get("critic_q_pessimism", 0.0))
            > getattr(args, "critic_q_pessimism", 0.0)):
        raise ValueError(
            "a critic fit under a Q prior initializes only a run with at "
            "least that --critic-q-pessimism"
        )
    if mode_carries_memory(args.reasoning_mode) and not uses_post_action_q(args):
        # The critic's reader is fit to the memory its source run's actor
        # writer built; this run's writer is a fresh draw, so the loaded
        # reader would read an unrelated basis at the same scale.
        raise ValueError(
            "--critic-only-init cannot carry a memory critic: its reader is "
            "bound to the source run's actor writer"
        )
    if not resume_critic_schema_compatible(
        payload, uses_q_critic(args), post_action_q=uses_post_action_q(args),
        twin_q=getattr(args, "critic_q_twin", False),
    ):
        raise ValueError("critic-only checkpoint uses an incompatible critic schema")
    expected = {
        "base_checkpoint_sha256": args.base_checkpoint_sha256,
        "reward_schema": reward_schema_for_abstention(
            getattr(args, "abstention_penalty", None)
        ),
        "python_reward_schema": getattr(args, "python_reward_schema", None),
        "math_data_identity": data_identity,
        "answer_fence_prompt_schema": (
            ANSWER_FENCE_PROMPT_SCHEMA if args.answer_fence else None
        ),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"critic-only checkpoint changed {name}")
    saved = payload.get("args", {})
    for name in ("reasoning_mode", "think_tokens", "answer_fence",
                 "think_min_tokens", "nearby_reward_max", "abstention_penalty",
                 "combined_mlp_hidden",
                 "combined_mlp_blocks", "prompt_tokens", "continuation_tokens",
                 "resolved_train_max_new_tokens", "resolved_train_max_stream_steps"):
        if saved.get(name) != getattr(args, name, None):
            raise ValueError(f"critic-only checkpoint changed {name}")
    if int(payload.get("value_warmup_step", 0)) < 1:
        raise ValueError("critic-only checkpoint has no completed critic warmup")
    if "critic" not in payload:
        raise ValueError("critic-only checkpoint has no critic weights")



def carries_init_critic(args: argparse.Namespace) -> bool:
    """Whether this run carries its --actor-init checkpoint's critic.

    Branching from an RL checkpoint keeps the critic by default: a fresh
    one spends its warmup relearning values the source critic already fit,
    and advantages computed against it are noise until it has.
    """
    return bool(getattr(args, "actor_init", None)) and not getattr(
        args, "fresh_critic", False
    )


# Arguments that, with the checkpoint's reward schema, define a carried
# critic's value target: the lambda-one return of the sampling policy under
# the terminal reward, truncated at the response horizon. The actor
# objective (DG/TPO, GAE lambda for the actor, source gates) moves no target.
CARRIED_CRITIC_VALUE_ARGS = (
    "abstention_penalty",
    # The pessimistic prior shrinks the fit at rarely taken tokens.
    "critic_q_pessimism",
    "nearby_reward_max",
    "temperature",
    "top_p",
    "prompt_tokens",
    "continuation_tokens",
    "answer_tokens",
    "resolved_train_max_new_tokens",
    "resolved_train_max_stream_steps",
)


def source_trained_critic_without_warmup(source_args: dict) -> bool:
    """Whether a source run's critic started trained, so it needed no
    warmup of its own: it was carried or transferred into that run."""
    return bool(
        source_args.get("critic_only_init")
        or source_args.get("actor_critic_init")
        or (source_args.get("actor_init") and not source_args.get("fresh_critic"))
    )


def critic_action_head_schema(critic: SeparateCritic) -> str | None:
    return None if critic.action_head is None else CRITIC_ACTION_HEAD_SCHEMA


def uses_post_action_q(args: argparse.Namespace) -> bool:
    return getattr(args, "critic_post_action_q_hidden", None) is not None


def mixture_python_reward_schema(args, sources) -> str | None:
    """Bind the new binary-Q mode to verifiers its sampler actually uses.

    Historical modes retain their manifest-wide checkpoint contract. New
    post-action Q excludes disabled sources before defining the reward task.
    """
    excluded = (
        set(filter(None, args.exclude_rl_sources.split(",")))
        if uses_post_action_q(args) else set()
    )
    if not any(source.verifier == "python_mbpp" and source.name not in excluded
               for source in sources):
        return None
    return (PYTHON_PARTIAL_REWARD_SCHEMA if args.python_reward_mode == "test-fraction"
            else PYTHON_REWARD_SCHEMA)


def validate_actor_init_python_reward(payload, expected, *, carries_critic: bool) -> None:
    """A fresh math-only critic inherits no Python reward objective.

    A carried critic still requires its exact value task; active Python
    rewards retain the existing objective compatibility check.
    """
    if payload is None or (expected is None and not carries_critic):
        return
    if payload.get("python_reward_schema") != expected:
        raise ValueError(
            "initialization checkpoint was trained under a different "
            "Python reward objective than the selected code reward mode"
        )


def critic_run_schema(critic: SeparateCritic) -> str:
    return critic_schema_for(
        critic.values_actions,
        post_action_q=getattr(critic, "post_action_q_head", None) is not None,
        twin_q=getattr(critic, "q_twin", None) is not None,
    )


def uses_q_critic(args: argparse.Namespace) -> bool:
    return getattr(args, "critic_q_head_hidden", None) is not None


def run_actor_objective_schema(args: argparse.Namespace) -> str:
    """This run's actor objective schema, every component included."""
    return actor_objective_schema(
        getattr(args, "reasoning_mode", "latent"),
        getattr(args, "delightful_policy_gradient", False),
        getattr(args, "target_policy_optimization", False),
        q_advantage=uses_q_critic(args),
        post_action_binary_q=uses_post_action_q(args),
        entropy_temperature=getattr(args, "entropy_target", None) is not None,
        kl_trust_region=getattr(args, "actor_kl_bound", None) is not None,
        mac_estimator=(
            args.mac_estimator if getattr(args, "mean_actor_critic", False) else None
        ),
    )


def refits_critic_values(args: argparse.Namespace) -> bool:
    """Whether this run's warmup refits the critic's values before the first
    actor step; an action-head-only warmup leaves them as carried."""
    return args.value_warmup_steps > 0 and (
        getattr(args, "value_warmup_scope", "critic") == "critic"
    )


def rewinds_warmup_prompts(args: argparse.Namespace) -> bool:
    """Whether the warmup returns the prompt cursor to where it began.

    An action-head warmup changes neither the actor nor the carried values,
    so rewinding leaves the head as the run's only difference from one that
    starts acting at the same checkpoint: the first actor step draws the same
    prompts under the same rollout keys, which derive from the cursor. A
    whole-critic warmup leaves the actor untouched too, and rewinds under
    --value-warmup-rewind: a fresh critic warmed from an --actor-init
    checkpoint then starts acting on the prompts that checkpoint's own
    continuation would.
    """
    return args.value_warmup_steps > 0 and (
        getattr(args, "value_warmup_scope", "critic") == "action-head"
        or bool(getattr(args, "value_warmup_rewind", False))
    )


def validate_carried_critic(payload: dict, args: argparse.Namespace) -> list[str]:
    """Refuse an --actor-init critic that cannot be carried; name the value
    contract fields this run changes from the one it fit.

    The critic's value task is the return under the terminal reward, policy
    temperature and horizon it fit. A change leaves it a biased start
    exactly where the two contracts value a trajectory differently, so one
    is accepted only together with a critic refit (``refits_critic_values``)
    before the first actor step, or an explicit --accept-critic-value-change
    (for instance when the changed reward never applied to a row the critic
    was fit on, which this check cannot see). The changed fields and how
    they were resolved are recorded in its provenance. A field an older checkpoint never recorded reads as its
    parser default. The reasoning mode, fences, think floor, data identity
    and Python reward schema are held fixed on every initialization path.
    """
    if "critic" not in payload:
        raise ValueError(
            "--actor-init checkpoint has no critic to carry; pass --fresh-critic"
        )
    if not resume_critic_schema_compatible(payload):
        raise ValueError(
            f"--actor-init critic schema is {payload.get('critic_schema')!r}, "
            f"expected {CRITIC_SCHEMA!r}; pass --fresh-critic"
        )
    expected_optimizer_schema = optimizer_schema_for_trunk_optimizer(
        args.trunk_optimizer,
        fresh_muon=args.fresh_learning_rate is not None,
    )
    if payload.get("optimizer_schema") != expected_optimizer_schema:
        raise ValueError(
            "--actor-init optimizer schema is "
            f"{payload.get('optimizer_schema')!r}, expected "
            f"{expected_optimizer_schema!r}; pass the source run's "
            "--trunk-optimizer and --fresh-learning-rate, or --fresh-critic"
        )
    source_args = payload.get("args", {})
    if hasattr(source_args, "__dict__"):
        source_args = vars(source_args)
    unfit = (
        int(payload.get("step", 0)) == 0
        and int(payload.get("value_warmup_step", 0))
        < int(source_args.get("value_warmup_steps") or 0)
    ) or (
        int(payload.get("step", 0)) == 0
        and int(payload.get("value_warmup_step", 0)) == 0
        and not source_trained_critic_without_warmup(source_args)
    )
    if unfit and not refits_critic_values(args):
        raise ValueError(
            "--actor-init checkpoint's critic stopped before its warmup "
            "finished; finish it with --value-warmup-steps > 0 or pass "
            "--fresh-critic"
        )
    changes = []
    if payload.get("reward_schema") != reward_schema_for_abstention(
        args.abstention_penalty
    ):
        changes.append("reward_schema")
    defaults = build_arg_parser()
    changes.extend(
        name
        for name in CARRIED_CRITIC_VALUE_ARGS
        if source_args.get(name, defaults.get_default(name))
        != getattr(args, name)
    )
    if (
        changes
        and not refits_critic_values(args)
        and not getattr(args, "accept_critic_value_change", False)
    ):
        raise ValueError(
            "--actor-init critic fit a different value target "
            f"({', '.join(changes)} changed); refit it with "
            "--value-warmup-steps > 0 under --value-warmup-scope critic, "
            "carry it as-is with "
            "--accept-critic-value-change, or pass --fresh-critic"
        )
    return changes


def carry_init_critic(
    critic: SeparateCritic,
    optimizers: dict[str, torch.optim.Optimizer],
    payload: dict,
    args: argparse.Namespace,
) -> None:
    """Load an --actor-init checkpoint's critic and its optimizer state.

    The critic keeps its optimizer state with its weights. The actor's
    optimizers are left as built, empty: a branch should not inherit
    momentum along the objective it departs from. An action head the source
    critic never had starts fresh -- zero output, empty optimizer -- beside
    the carried values; one it had must be this run's head.
    """
    source_head = payload.get("critic_action_head_schema")
    fresh_head = source_head is None and critic.action_head is not None
    if source_head is not None and source_head != critic_action_head_schema(
        critic
    ):
        raise ValueError(
            f"--actor-init critic has action head {source_head!r}; this run's "
            f"is {critic_action_head_schema(critic)!r}"
        )
    if fresh_head:
        missing, unexpected = critic.load_state_dict(
            payload["critic"], strict=False
        )
        head_keys = {
            f"action_head.{name}" for name in critic.action_head.state_dict()
        }
        if unexpected or set(missing) != head_keys:
            raise ValueError(
                "--actor-init critic does not match this critic outside its "
                f"action head: missing {sorted(set(missing) - head_keys)}, "
                f"unexpected {sorted(unexpected)}"
            )
    else:
        critic.load_state_dict(payload["critic"], strict=True)
    source_optimizers = payload["optimizers"]
    critic_names = [
        name
        for name in optimizers
        if name.startswith("critic")
        and not (fresh_head and name == "critic_action")
    ]
    source_critic_names = sorted(
        name for name in source_optimizers if name.startswith("critic")
    )
    if sorted(critic_names) != source_critic_names:
        raise ValueError(
            "--actor-init checkpoint critic optimizer layout "
            f"{source_critic_names} does not match the --trunk-optimizer "
            f"{args.trunk_optimizer} layout {sorted(critic_names)}; pass the "
            "matching --trunk-optimizer and --fresh-learning-rate, or "
            "--fresh-critic"
        )
    for name in critic_names:
        optimizers[name].load_state_dict(source_optimizers[name])
    reassert_learning_rates({name: optimizers[name] for name in critic_names}, args)

def resolve_critic_init_provenance(
    args: argparse.Namespace,
    actor_init_payload: dict | None,
    resumed_manifest: dict | None,
) -> dict:
    """How the critic's weights started, for the run manifest.

    A resume loads the critic from its checkpoint, so its start is whatever
    the resumed run recorded (``resumed_manifest``: the manifest of the run
    that wrote the checkpoint, whether resuming in place or into a new
    directory). That record is carried forward, and a --critic-init that
    contradicts it is refused.
    """
    if args.resume:
        if resumed_manifest is None:
            raise ValueError(
                "resume needs the resumed run's manifest to carry its critic "
                "init forward"
            )
        recorded = resumed_manifest.get("critic") or {}
        init = recorded.get("init")
        if init not in ("scratch", "actor_copy", "warm_checkpoint", "critic_only_checkpoint",
                        "study_checkpoint"):
            raise ValueError(
                f"resume manifest records an unknown critic init {init!r}"
            )
        if (init in ("actor_copy", "study_checkpoint")) != (args.critic_init == "actor"):
            raise ValueError(
                f"resume run's critic started as {init!r}; resume it with "
                f"--critic-init {'actor' if init in ('actor_copy', 'study_checkpoint') else 'scratch'}"
            )
        return {name: recorded.get(name) for name in CRITIC_INIT_PROVENANCE_FIELDS
                if name in recorded}
    if getattr(args, "critic_only_init", None):
        return dict(args.critic_only_init_provenance)
    if getattr(args, "study_critic_init", None):
        return dict(args.study_critic_init_provenance)
    if carries_init_critic(args):
        return {
            "init": "warm_checkpoint",
            "checkpoint": str(args.actor_init),
            "checkpoint_sha256": args.initialization_checkpoint_sha256,
            "source_execution_schema": actor_init_payload.get("execution_schema"),
            "source_actor_step": int(actor_init_payload.get("step", 0)),
            "source_value_warmup_step": int(
                actor_init_payload.get("value_warmup_step", 0)
            ),
            "optimizer_state": "carried",
            "source_reward_schema": actor_init_payload.get("reward_schema"),
            "value_contract_changes": list(args.carried_critic_value_changes),
            # How a changed contract was answered: a refit before the first
            # actor step, or an explicit acceptance of the critic as-is.
            "value_change_resolution": (
                None
                if not args.carried_critic_value_changes
                else "refit"
                if refits_critic_values(args)
                else "accepted"
            ),
            "action_head_init": (
                None
                if getattr(args, "critic_action_head_hidden", None) is None
                else "carried"
                if actor_init_payload.get("critic_action_head_schema")
                else "fresh"
            ),
        }
    return {
        "init": "actor_copy" if args.critic_init == "actor" else "scratch",
        "checkpoint": None,
        "source_execution_schema": None,
    }


def resume_topology_history(
    existing_manifest: dict | None,
    saved_args: dict | None,
    current: argparse.Namespace,
    *,
    checkpoint: str | None,
    checkpoint_sha256: str | None,
    step: int,
    sampler_cursor: int,
) -> list[dict]:
    """Preserve and extend the auditable rollout-topology history."""
    history = list((existing_manifest or {}).get("topology_history") or [])
    if not saved_args or not checkpoint:
        return history
    before = {
        "prompts_per_rollout": int(saved_args.get("prompts_per_rollout", -1)),
        "prompts_per_minibatch": int(
            saved_args.get("prompts_per_minibatch", -1)
        ),
        "rollout_groups": int(saved_args.get("rollout_groups", -1)),
    }
    after = {
        "prompts_per_rollout": int(current.prompts_per_rollout),
        "prompts_per_minibatch": int(current.prompts_per_minibatch),
        "rollout_groups": int(current.rollout_groups),
    }
    if before == after:
        return history
    transition = {
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_step": int(step),
        "source_sampler_cursor": int(sampler_cursor),
        "before": before,
        "after": after,
    }
    transition_identity = {
        key: transition[key]
        for key in (
            "source_checkpoint_sha256",
            "source_step",
            "source_sampler_cursor",
            "before",
            "after",
        )
    }
    already_recorded = any(
        all(entry.get(key) == value for key, value in transition_identity.items())
        for entry in history
    )
    if not already_recorded:
        history.append(transition)
    return history


class _TileLangCompileCounter(logging.Handler):
    """Count TileLang JIT kernel compiles via their announcement log line.

    FLA's KDA ops JIT TileLang kernels outside every torch.compile counter,
    so the only in-process signal is ``tilelang.jit.kernel`` logging
    "TileLang begins to compile kernel". Attached by logger NAME at import,
    before tilelang itself is imported — logger objects are process-global
    singletons, so the handler survives tilelang's own logging setup.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "begins to compile" in record.getMessage():
            self.count += 1


tilelang_compile_counter = _TileLangCompileCounter()
logging.getLogger("tilelang.jit.kernel").addHandler(tilelang_compile_counter)


def math_dataset_identity(
    path: str | Path, exclude_modules: str, rows: list[dict]
) -> str:
    """Content identity that makes a sequential cursor safe to resume."""
    digest = hashlib.sha256()
    digest.update(PROMPT_ORDER_SCHEMA.encode())
    digest.update(b"\0")
    digest.update(exclude_modules.encode())
    digest.update(b"\0")
    digest.update(math_corpus_identity(rows).encode())
    digest.update(b"\0")
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sample_prompt_batch(
    loader: "baseline.DistributedTokenLoader",
    prompt_tokens: int,
    continuation_tokens: int,
    prompts: int,
    samples_per_prompt: int,
    seq_len: int,
    grad_accum: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carve (prompt, reference-continuation) pairs from the token stream.

    Training no longer uses this (RL data is DAPO-Math only); it remains for
    ``sample_latent --fineweb`` inspection.
    """
    needed = prompt_tokens + continuation_tokens
    if needed > seq_len:
        raise ValueError("prompt + continuation must fit in one training sequence")
    inputs, _ = loader.next_batch(prompts * seq_len * grad_accum, seq_len, grad_accum)
    rows = inputs[:prompts]
    prompt_ids = rows[:, :prompt_tokens]
    reference_ids = rows[:, prompt_tokens:needed]
    return (
        prompt_ids.repeat_interleave(samples_per_prompt, dim=0),
        reference_ids.repeat_interleave(samples_per_prompt, dim=0),
    )


class MathPromptSampler:
    """Sequential prompt stream with a resumable monotonic cursor.

    The first epoch walks ``rows`` in their given order, so existing one-pass
    runs, resumes, and curriculum cursors keep their exact historical stream.
    When a run's step budget outlasts the dataset, later epochs reuse every
    prompt exactly once per epoch in a deterministic per-epoch shuffle derived
    from ``seed`` — the cursor keeps counting total prompts consumed, so
    checkpoints resume mid-epoch without replaying or skipping prompts.
    """

    def __init__(
        self,
        rows: list[dict],
        seed: int,
        dataset_identity: str | None = None,
    ):
        if not rows:
            raise ValueError("no math prompts loaded")
        self.rows = rows
        self.seed = seed
        self.cursor = 0
        self.dataset_identity = dataset_identity
        self._order_epoch: int | None = None
        self._order: list[int] | None = None

    @property
    def epoch(self) -> int:
        return self.cursor // len(self.rows)

    def _epoch_order(self, epoch: int) -> list[int]:
        if epoch != self._order_epoch:
            order = list(range(len(self.rows)))
            if epoch > 0:
                random.Random(self.seed * 1_000_003 + epoch).shuffle(order)
            self._order_epoch, self._order = epoch, order
        assert self._order is not None
        return self._order

    def next_rows(self, count: int) -> list[dict]:
        if count < 0:
            raise ValueError("prompt count must be nonnegative")
        picked: list[dict] = []
        while count:
            epoch, offset = divmod(self.cursor, len(self.rows))
            take = min(count, len(self.rows) - offset)
            order = self._epoch_order(epoch)
            picked.extend(self.rows[i] for i in order[offset:offset + take])
            self.cursor += take
            count -= take
        return picked


def answer_prefix_token_ids(tokenizer) -> tuple[int, ...]:
    """The ``"\\nAnswer:"`` ids as they tokenize inside a QA document.

    SentencePiece is boundary-sensitive: the sp1024 model encodes standalone
    ``"\\nAnswer:"`` as ``A n s w er :`` with the newline dropped, while the
    same substring inside the mathmix documents the backbone pretrained on
    (``"{question}\\nAnswer: {answer}"``) tokenizes as ``▁An s w er :``.
    Teacher-forcing the standalone ids would place the policy in a prompt
    state it never saw.  Deriving the ids from document-shaped encodings —
    and requiring two different question endings to agree — keeps the forced
    prefix on the pretraining distribution.
    """
    tails = []
    for context_text in ("?", "."):
        context = tokenizer.encode(context_text)
        combined = tokenizer.encode(context_text + "\nAnswer:")
        if combined[: len(context)] != context:
            raise RuntimeError(
                "tokenizer merged across the Answer: prefix boundary; "
                "cannot derive stable in-context prefix ids"
            )
        tails.append(tuple(combined[len(context):]))
    if tails[0] != tails[1]:
        raise RuntimeError(
            "the in-context Answer: prefix tokenizes differently after "
            f"different question endings: {tails[0]} vs {tails[1]}"
        )
    return tails[0]


def think_span_tokens(
    tokens: Sequence[int],
    think_fence_ids: tuple[int, int],
) -> int | None:
    """Token count inside the single ``<think>...</think>`` pair, else None.

    None means the fence structure itself is broken (missing, duplicated,
    or reversed fences), so no inner length exists to measure.
    """
    span = single_fence_span(tokens, think_fence_ids)
    if span is None:
        return None
    open_index, close_index = span
    return close_index - open_index - 1


# The verifier's field pattern (core.extract_final_answer): matching
# anything narrower here reopens the gate — the old literal "Answer:"
# substring test let a case-variant ``answer: 42`` guess sit in front
# of an all-filler fence and still collect full reward.
ANSWER_FIELD_PATTERN = re.compile(r"(?i)answer\s*:")


def think_format_ok(
    tokens: Sequence[int],
    think_fence_ids: tuple[int, int],
    tokenizer,
    min_think_tokens: int = 1,
    decoded_text: str | None = None,
) -> bool:
    """One ``<think>...</think>`` of >= ``min_think_tokens`` tokens, with
    the GRADED answer field after the close.

    The gate exists to make the bare-guess collapse attractor worth zero,
    so the degenerate satisfactions matter as much as the honest one: an
    empty fence (``<think></think>`` appended as ritual) fails, and a
    fence whose graded answer precedes the close (guess first, fence
    later) fails. The verifier grades the LAST ``Answer:`` field
    (``extract_final_answer``), so that match — found with the
    verifier's own case-insensitive pattern — is the one whose position
    is checked; earlier matches (e.g. an instruction echo inside the
    fence) are not graded and do not fail the gate.
    ``min_think_tokens`` raises the floor from "non-empty" to a compute
    budget: round 3 showed a 1-token floor collapses to a ~15-token
    minimal compliant skeleton, so the floor is the lever that forces
    sequential latent compute to actually happen before the answer.
    ``decoded_text`` lets a caller that already decoded ``tokens`` skip
    the second decode (it must be the decode of exactly ``tokens``).
    """
    inner = think_span_tokens(tokens, think_fence_ids)
    if inner is None or inner < max(min_think_tokens, 1):
        return False
    _, close_id = think_fence_ids
    close = next(
        index for index, token in enumerate(tokens) if token == close_id
    )
    text = (
        tokenizer.decode(list(tokens)) if decoded_text is None
        else decoded_text
    )
    matches = list(ANSWER_FIELD_PATTERN.finditer(text))
    if not matches:
        # No answer field anywhere: nothing graded precedes the fence.
        return True
    # decode skips the fence specials, so the pre-close prefix length
    # locates the close inside the decoded text. Byte-level BPE can
    # split a multi-byte character at the boundary; two characters of
    # slack absorb the replacement-char wobble without readmitting a
    # real pre-close answer.
    prefix_length = len(tokenizer.decode(list(tokens[:close])))
    return matches[-1].start() >= prefix_length - 2


def rewrite_prompts_for_answer_fence(rows: list[dict]) -> list[dict]:
    """Canonicalize source framing to one shared think/answer contract."""

    return canonicalize_answer_fence_rows(rows)


def check_think_floor_against_corpus(
    sft_provenance: dict, min_think_tokens: int
) -> None:
    """Fail loudly when the think floor exceeds what the SFT corpus taught.

    The corpus-shape guard verifies the fence SHAPE, not span length: a
    corpus of anchored-but-terse traces measures a 1.0 fence fraction yet
    cannot reach an RL floor above its span lengths, so every
    structurally-gated reward is zero from step 0 (red-team round 2,
    finding D; the round-3/4 collapses both started from exactly such a
    zero-reward desert). A floor above the corpus median is refused; one
    above the 1st percentile warns.
    """
    if min_think_tokens <= 1:
        return
    percentiles = sft_provenance.get("think_span_token_percentiles")
    if not percentiles:
        print(
            f"WARNING: --think-min-tokens {min_think_tokens} cannot be "
            "checked against the SFT corpus (no "
            "think_span_token_percentiles in provenance; pre-round-5 SFT "
            "checkpoint?)",
            flush=True,
        )
        return
    p50 = float(percentiles["p50"])
    p1 = float(percentiles["p1"])
    if min_think_tokens > p50:
        raise RuntimeError(
            f"--think-min-tokens {min_think_tokens} exceeds the SFT "
            f"corpus's median think-span length ({p50:.0f} tokens): the "
            "majority of the format prior falls below the floor, so most "
            "structurally-gated rewards would be zero"
        )
    if min_think_tokens > p1:
        print(
            f"WARNING: --think-min-tokens {min_think_tokens} exceeds the "
            f"SFT corpus's 1st-percentile think-span length ({p1:.0f} "
            "tokens); the shortest taught traces fall below the floor",
            flush=True,
        )


def relaxed_fenced_answer_text(
    tokens: Sequence[int],
    tokenizer,
    answer_fence_ids: tuple[int, int],
) -> str | None:
    """First ``<answer>``...``</answer>`` span, tolerant of breakage.

    NEVER used for reward. The gate-zeroed-correct alarm asks "was the
    value right even though the structure was not?", and a fence-native
    policy expresses its value inside (possibly duplicated, unanchored,
    or think-less) answer fences that ``decode`` strips — so the strict
    extractor and the plain-text parse are both blind exactly when the
    alarm matters most (red-team finding: every fence-native collapse
    mode read as a clean zero).

    Read the resulting alarm as approximate, not a count: taking the
    FIRST span means a scratch ``<answer>`` inside the think span wins
    over the policy's actual answer (over-count), and under duplicated
    spans a wrong first answer hides a right second one (under-count).
    """
    open_id, close_id = answer_fence_ids
    open_index = next(
        (index for index, token in enumerate(tokens) if token == open_id),
        None,
    )
    if open_index is None:
        return None
    close_index = next(
        (
            index
            for index, token in enumerate(
                tokens[open_index + 1:], open_index + 1
            )
            if token == close_id
        ),
        None,
    )
    if close_index is None or close_index - open_index < 2:
        return None
    return tokenizer.decode(list(tokens[open_index + 1:close_index]))


def abstention_terminal_scores(
    scores: list[float], abstained: list[bool], penalty: float | None
) -> list[float]:
    """Terminal rewards with explicit abstention (``--abstention-penalty``).

    A verifier score ``s`` retains its [0, 1] value; an explicit abstention
    scores ``penalty / (1 + penalty)``. This is the positive affine transform
    of the signed objective (correct 1, abstain 0, wrong -penalty). It keeps
    the same optimal decision threshold while putting common wrong answers
    at the zero baseline of the freshly initialized critic. ``None``
    leaves the scores unchanged, abstentions included (they already score
    0 as format failures).
    """
    if len(scores) != len(abstained):
        raise ValueError("one abstention flag per score is required")
    if penalty is None:
        return list(scores)
    abstention_reward = normalized_abstention_reward(penalty)
    return [
        abstention_reward if abstain else score
        for score, abstain in zip(scores, abstained, strict=True)
    ]


def score_math_rollout(
    batch: LatentRolloutBatch,
    truth: str,
    tokenizer,
    stop_ids: tuple[int, ...],
    style: str = "minerva",
    nearby_reward_max: float = 0.1,
    solution_prefix_ids: tuple[int, ...] = (),
    think_fence_ids: tuple[int, int] | None = None,
    min_think_tokens: int = 1,
    answer_fence_ids: tuple[int, int] | None = None,
    abstention_penalty: float | None = None,
) -> list[RowVerdict]:
    """Exact verifier reward plus bounded final-answer numeric proximity.

    Returns each row's gate verdict and graded prediction for transcripts.

    ``solution_prefix_ids`` are teacher-forced solution tokens that live at
    the end of the prompt (the none-mode ``Answer:`` prefix): the emitted
    continuation alone never contains them, so they rejoin the decode before
    the verifier parses a final answer.

    ``think_fence_ids`` gates ALL reward on a non-empty think fence closed
    before the answer (see ``think_format_ok``): a bare guess scores zero
    even when the final answer is right, so the reward-optimal policy
    cannot drop the thinking channel. Gate-zeroed-but-correct rows are
    counted on ``batch.think_gate_zeroed_correct`` so telemetry can tell a
    signal-destroying gate apart from a policy that got worse.

    ``answer_fence_ids`` (requires ``think_fence_ids``) upgrades the gate
    to ``structural_format_ok`` and grades ONLY the decoded content of the
    single ``<answer>`` span — no regex over full decoded text remains, so
    the position-check bypass class is gone. Within the span the field is
    still located by the verifier's last-``Answer:``-match rule on the
    reframed string, so a span containing its own ``Answer: x`` line
    grades ``x`` — a fixed single graded field either way, never extra
    reward. The gate-zeroed-correct counterfactual prefers a RELAXED span
    scan over the plain-text parse: "the value was right but the
    structure was not" must see the value where a fence-native policy
    puts it, or every fence-native collapse reads as a clean zero.
    """
    if answer_fence_ids is not None and think_fence_ids is None:
        raise ValueError("answer_fence_ids requires think_fence_ids")
    if abstention_penalty is not None and answer_fence_ids is None:
        raise ValueError("abstention_penalty requires answer_fence_ids")
    scores = []
    verdicts = []
    abstained: list[bool] = []
    gate_zeroed_correct = 0
    for emitted in emitted_token_rows(batch):
        abstained.append(False)
        stop_cut = next(
            (index for index, token in enumerate(emitted) if token in stop_ids),
            None,
        )
        if stop_cut is None:
            scores.append(0.0)
            verdicts.append(RowVerdict(format_ok=False, parsed_answer=None))
            continue
        visible = emitted[: stop_cut + 1]
        solution = tokenizer.decode(list(solution_prefix_ids) + visible)
        correct, prediction = verify_answer(solution, truth, style)
        if answer_fence_ids is not None:
            if structural_abstention(
                visible, think_fence_ids, answer_fence_ids,
                stop_ids[0], min_think_tokens,
            ):
                # Reported in verifier_status; the verdict's format_ok is
                # structural_format_ok, which no abstention meets.
                abstained[-1] = True
                scores.append(0.0)
                verdicts.append(RowVerdict(format_ok=False, parsed_answer=None))
                continue
            if not structural_format_ok(
                visible, think_fence_ids, answer_fence_ids, min_think_tokens
            ):
                # The alarm's counterfactual must see the value where a
                # fence-native policy actually puts it: a relaxed span
                # scan first (decode strips broken fences, so the plain
                # parse alone is blind to them), then the plain parse.
                relaxed = relaxed_fenced_answer_text(
                    visible, tokenizer, answer_fence_ids
                )
                if relaxed is not None:
                    correct, prediction = verify_answer(
                        "Answer: " + relaxed, truth, style, window=None
                    )
                gate_zeroed_correct += bool(correct)
                scores.append(0.0)
                verdicts.append(RowVerdict(format_ok=False, parsed_answer=prediction))
                continue
            # The gate guarantees a single non-empty anchored span, so
            # the graded field is the fenced value re-framed for the
            # verifier; window=None because the reframe IS the field —
            # a long value must not push its own prefix out of the
            # verifier's tail window and grade [INVALID].
            solution = "Answer: " + fenced_answer_text(
                visible, tokenizer, answer_fence_ids
            )
            correct, prediction = verify_answer(
                solution, truth, style, window=None
            )
        elif think_fence_ids is not None and not think_format_ok(
            visible, think_fence_ids, tokenizer,
            min_think_tokens,
            # The solution decode IS the row decode when no prefix is
            # teacher-forced (always true under the gate: none-mode is
            # rejected at argument validation).
            decoded_text=solution if not solution_prefix_ids else None,
        ):
            gate_zeroed_correct += bool(correct)
            scores.append(0.0)
            verdicts.append(RowVerdict(format_ok=False, parsed_answer=prediction))
            continue
        verdicts.append(RowVerdict(format_ok=True, parsed_answer=prediction))
        raw_final_answer = extract_final_answer(
            solution,
            window=None if answer_fence_ids is not None else 300,
        )
        strict_numeric = (
            parse_numeric_answer(raw_final_answer)
            if raw_final_answer is not None
            else None
        )
        if correct:
            scores.append(1.0)
        elif strict_numeric is None:
            scores.append(0.0)
        else:
            # Only the raw final Answer: field enters proximity scoring.
            # Earlier candidates and multi-number final fields cannot add or
            # manufacture reward.
            scores.append(
                nearby_numeric_reward(
                    raw_final_answer, truth, nearby_reward_max
                )
            )
    # Unconditional assignment: rescoring without the gate must clear a
    # stale count rather than leave the old alarm value behind.
    batch.think_gate_zeroed_correct = (
        gate_zeroed_correct if think_fence_ids is not None else None
    )
    batch.verifier_status = torch.tensor(
        [
            PYTHON_RESULT_CODES["abstained" if abstain else "not_applicable"]
            for abstain in abstained
        ],
        dtype=torch.long,
        device=batch.rewards.device,
    )
    assign_terminal_rewards(
        batch,
        torch.tensor(
            abstention_terminal_scores(scores, abstained, abstention_penalty),
            dtype=torch.float32,
            device=batch.rewards.device,
        ),
    )
    return verdicts


def graded_answer_keys(
    verdicts: Sequence[RowVerdict], rewards: Sequence[float]
) -> list[tuple[str, float] | None]:
    """Each row's answer-cluster key: the verifier's own canonical prediction
    for the field it graded paired with the reward that grading earned, or
    None where the reward gate graded nothing (unterminated, malformed,
    abstaining, or no ``Answer:`` field). A gate-zeroed row's verdict carries
    a counterfactual prediction that earned no reward; it is not that
    answer, so it keys None too.

    The reward is part of the key because canonicalization is coarser than
    grading: ``verify_answer`` normalizes "3 4" and "34" to one string yet
    grades them differently, so the string alone would let a wrong row join
    the correct cluster. ``latent_eval.majority_vote_score`` votes on the
    same (answer, correct) pairs. The sequence estimators refuse shaped
    rewards (``vapo.config``), so the reward here is the verifier's grade.
    """
    if len(verdicts) != len(rewards):
        raise ValueError("one verdict per group row is required")
    return [
        (verdict.parsed_answer, float(reward))
        if verdict.format_ok and verdict.parsed_answer not in (None, "[INVALID]")
        else None
        for verdict, reward in zip(verdicts, rewards)
    ]


def stamp_group_terms(batch: LatentRolloutBatch, verdicts: Sequence[RowVerdict]) -> None:
    """Stamp one scored prompt group's answer-level estimator terms
    (``LatentRolloutBatch.group_advantage``/``answer_surprisal``): the
    leave-one-out advantage of each terminal reward, and the surprisal of
    each row's graded (answer, reward) cluster, rows without a graded answer
    forming one shared cluster (``mean_actor_critic.answer_cluster_labels``)."""
    if len(verdicts) != batch.kind.size(0):
        raise ValueError("one verdict per group row is required")
    rewards = batch.reward_scalar.float()
    labels = torch.tensor(
        answer_cluster_labels(graded_answer_keys(verdicts, rewards.tolist())),
        device=batch.kind.device,
    )
    batch.group_advantage = leave_one_out_advantages(rewards)
    batch.answer_surprisal = answer_cluster_surprisal(labels).float()


def score_python_rollout(
    batch: LatentRolloutBatch,
    verification_info: dict,
    tokenizer,
    stop_ids: tuple[int, ...],
    think_fence_ids: tuple[int, int],
    answer_fence_ids: tuple[int, int],
    min_think_tokens: int,
    reward_mode: str = "binary",
    abstention_penalty: float | None = None,
) -> list[RowVerdict]:
    """Test-based reward over the strict fenced code span.

    Returns each row's gate verdict, with the test harness's result as the
    graded prediction of an eligible row.
    """
    if reward_mode not in {"binary", "test-fraction"}:
        raise ValueError(f"unsupported Python reward mode {reward_mode!r}")
    if verification_info.get("schema") != PYTHON_REWARD_SCHEMA:
        raise ValueError("Python row uses an incompatible verifier schema")
    answers: list[str] = []
    eligible_indices: list[int] = []
    emitted_rows = emitted_token_rows(batch)
    scores = [0.0] * len(emitted_rows)
    statuses = [PYTHON_RESULT_CODES["format_ineligible"]] * len(emitted_rows)
    verdicts = [
        RowVerdict(format_ok=False, parsed_answer=None) for _ in emitted_rows
    ]
    abstained = [False] * len(emitted_rows)
    for index, emitted in enumerate(emitted_rows):
        stop_cut = next(
            (position for position, token in enumerate(emitted) if token in stop_ids),
            None,
        )
        if stop_cut is None:
            continue
        visible = emitted[: stop_cut + 1]
        if structural_abstention(
            visible, think_fence_ids, answer_fence_ids,
            stop_ids[0], min_think_tokens,
        ):
            abstained[index] = True
            statuses[index] = PYTHON_RESULT_CODES["abstained"]
            continue
        if not structural_format_ok(
            visible,
            think_fence_ids,
            answer_fence_ids,
            min_think_tokens,
        ):
            continue
        answers.append(fenced_answer_text(visible, tokenizer, answer_fence_ids))
        eligible_indices.append(index)
    if answers:
        if reward_mode == "test-fraction":
            results = batch_python_test_scores(answers, verification_info)
            for index, result in zip(eligible_indices, results, strict=True):
                scores[index] = result.reward
                statuses[index] = PYTHON_RESULT_CODES[result.status]
                verdicts[index] = RowVerdict(
                    format_ok=True,
                    parsed_answer=f"{result.status}: {result.passed}/{result.total} tests passed",
                )
        else:
            results = batch_python_test_results(answers, verification_info)
            for index, result in zip(eligible_indices, results, strict=True):
                scores[index] = float(result == "pass")
                statuses[index] = PYTHON_RESULT_CODES[result]
                verdicts[index] = RowVerdict(format_ok=True, parsed_answer=result)
    batch.verifier_status = torch.tensor(
        statuses, dtype=torch.long, device=batch.rewards.device
    )
    batch.think_gate_zeroed_correct = 0
    assign_terminal_rewards(
        batch,
        torch.tensor(
            abstention_terminal_scores(scores, abstained, abstention_penalty),
            dtype=torch.float32,
            device=batch.rewards.device,
        ),
    )
    return verdicts


evaluate_aime_latent = evaluate_latent_math


def rollout_diagnostics(
    batch: LatentRolloutBatch,
    samples_per_prompt: int,
    stop_ids: tuple[int, ...] = (),
    *,
    refreshed_statistics: bool = True,
    think_fence_ids: tuple[int, int] | None = None,
    tokenizer=None,
    min_think_tokens: int = 1,
    answer_fence_ids: tuple[int, int] | None = None,
    reward_floor: float = 0.0,
) -> dict[str, float | int]:
    """Pool-level rollout metrics.

    ``reward_floor`` is the terminal reward of a zero verifier score (0).
    A partial reward lies strictly between it and 1.

    ``refreshed_statistics`` says whether ``refresh_old_statistics`` filled
    ``batch.old_values``. It has no default meaning worth guessing: an
    unrefreshed batch carries zeros there, and reporting their mean as
    ``old_value_mean`` would publish a hard 0.0 that reads exactly like a
    critic collapsed to zero. The key is omitted instead.
    """
    stop_set = set(stop_ids)
    emitted_rows = list(emitted_token_rows(batch))
    # Fraction of rows that terminated themselves (emitted BOS or EOS)
    # rather than exhausting the token budget.
    ended = [
        float(any(token in stop_set for token in row))
        for row in emitted_rows
    ]
    format_ok: list[float] = []
    think_lengths: list[int] = []
    if think_fence_ids is not None:
        if tokenizer is None:
            raise ValueError("think_fence_ids requires the tokenizer")
        for row in emitted_rows:
            stop_cut = next(
                (index for index, token in enumerate(row) if token in stop_set),
                None,
            )
            visible = row if stop_cut is None else row[: stop_cut + 1]
            inner = think_span_tokens(visible, think_fence_ids)
            if inner is not None:
                think_lengths.append(inner)
            # Mirror the scorer: an unterminated row earns no reward
            # regardless of its fence, so counting it compliant would
            # let this fraction read high while every reward is zero
            # (truncation itself is visible in ended_fraction).
            if answer_fence_ids is not None:
                compliant = stop_cut is not None and structural_format_ok(
                    visible, think_fence_ids, answer_fence_ids,
                    min_think_tokens,
                )
            else:
                compliant = stop_cut is not None and think_format_ok(
                    visible, think_fence_ids, tokenizer, min_think_tokens
                )
            format_ok.append(float(compliant))
    generated = batch.action_mask.bool()
    grouped = batch.reward_scalar.reshape(-1, samples_per_prompt)
    grouped_lengths = batch.action_mask.sum(1).to(grouped.dtype).reshape(
        -1, samples_per_prompt
    )
    exact = batch.reward_scalar == 1.0
    grouped_exact = exact.reshape(-1, samples_per_prompt).float()
    abstained = (
        batch.verifier_status == PYTHON_RESULT_CODES["abstained"]
        if batch.verifier_status is not None
        else torch.zeros_like(exact)
    )
    abstained_lengths = [
        len(row)
        for row, is_abstained in zip(
            emitted_rows, abstained.to(device="cpu").tolist(), strict=True
        )
        if is_abstained
    ]
    thought_actions = think_slot_mask(batch)
    continued_thoughts = thought_actions & batch.stop_mask.bool()
    forced_thoughts = thought_actions & ~batch.stop_mask.bool()
    gate_decisions = batch.stop_mask.sum()
    stop_decisions = (
        (batch.actions == EMIT).to(batch.stop_mask.dtype) * batch.stop_mask
    ).sum()
    continue_decisions = (
        (batch.actions == THINK).to(batch.stop_mask.dtype) * batch.stop_mask
    ).sum()
    thought_count = thought_actions.sum()
    thought_metrics: dict[str, float] = {
        "total_thoughts": float(thought_count),
        "thoughts_per_trajectory": float(
            thought_count / max(batch.reward_scalar.numel(), 1)
        ),
        "continued_thoughts": float(continued_thoughts.sum()),
        "forced_first_thought_fraction": float(
            forced_thoughts.sum() / max(batch.reward_scalar.numel(), 1)
        ),
        "continuation_fraction": float(
            continue_decisions / gate_decisions.clamp_min(1)
        ),
        "gate_decisions": float(gate_decisions),
        "gate_stop_fraction": float(
            stop_decisions / gate_decisions.clamp_min(1)
        ),
    }
    if batch.thoughts.size(-1) and bool(thought_actions.any()):
        action_mask_no_tail = thought_actions[:, :-1]
        acted = batch.thoughts[:, 1:][action_mask_no_tail]
        means = batch.old_thought_means[:, :-1][action_mask_no_tail]
        log_sigmas = batch.old_thought_log_sigmas[:, :-1][action_mask_no_tail]
        component_std = log_sigmas.exp().mean()
        thought_metrics.update(
            thought_sigma=float(
                component_std * math.sqrt(batch.thoughts.size(-1))
            ),
            thought_component_std=float(component_std),
            sampled_noise_norm=float((acted - means).norm(dim=-1).mean()),
        )
    return {
        "trajectories": batch.reward_scalar.numel(),
        "stream_length": batch.stream_length,
        "reward_mean": float(batch.reward_scalar.mean()),
        "reward_max": float(batch.reward_scalar.max()),
        "reward_min": float(batch.reward_scalar.min()),
        "reward_std": float(batch.reward_scalar.std(unbiased=False)),
        "within_group_reward_std": float(grouped.std(dim=1, unbiased=False).mean()),
        "exact_accuracy": float(exact.float().mean()),
        "exact_within_group_reward_std": float(
            grouped_exact.std(dim=1, unbiased=False).mean()
        ),
        "partial_reward_fraction": float(
            ((batch.reward_scalar > reward_floor) & ~exact & ~abstained)
            .float()
            .mean()
        ),
        "actions_per_trajectory": float(batch.action_mask.sum(1).mean()),
        # Within-prompt covariance of reward and action count, averaged
        # over prompt groups: the direction in which the reward itself
        # pulls length under a group-mean baseline. Its sign separates a
        # reward-driven length trend from one the critic or optimizer
        # introduces.
        "length_reward_covariance": float(
            (
                (grouped - grouped.mean(1, keepdim=True))
                * (grouped_lengths - grouped_lengths.mean(1, keepdim=True))
            ).mean()
        ),
        "abstained_emitted_tokens_sum": float(sum(abstained_lengths)),
        "abstained_emitted_tokens_count": float(len(abstained_lengths)),
        "abstained_emitted_tokens_max": float(max(abstained_lengths, default=0)),
        **thought_metrics,
        "ended_fraction": sum(ended) / max(len(ended), 1),
        **(
            {
                f"verifier_{name}_fraction": float(
                    (batch.verifier_status == code).float().mean()
                )
                for name, code in PYTHON_RESULT_CODES.items()
            }
            if batch.verifier_status is not None
            else {}
        ),
        **(
            {
                "think_format_fraction": (
                    sum(format_ok) / max(len(format_ok), 1)
                ),
                # Inner-token sums over structurally intact fences; the
                # aggregate divides ONCE across groups (a mean-of-means
                # over varying fence counts would bias low exactly when
                # many groups have no intact fence). The derived mean —
                # pinned at the floor, it says the policy pays exactly
                # the mandated compute — is emitted only when a fence
                # exists to measure: a published 0.0 would read like
                # zero-length thinks (same convention as
                # ``old_value_mean`` below).
                "think_tokens_sum": float(sum(think_lengths)),
                "think_tokens_count": float(len(think_lengths)),
                **(
                    {
                        "think_tokens_mean": (
                            sum(think_lengths) / len(think_lengths)
                        )
                    }
                    if think_lengths
                    else {}
                ),
                # Verifier-correct rows the gate zeroed: nonzero here
                # means the gate, not the policy, is eating reward
                # signal. Omitted when no gated scoring stamped the
                # batch — absence is not a healthy zero.
                **(
                    {
                        "think_gate_zeroed_correct_fraction": (
                            batch.think_gate_zeroed_correct
                            / max(batch.reward_scalar.numel(), 1)
                        )
                    }
                    if batch.think_gate_zeroed_correct is not None
                    else {}
                ),
            }
            if think_fence_ids is not None
            else {}
        ),
        **(
            {
                "old_value_mean": (
                    float(batch.old_values[generated].mean())
                    if generated.any()
                    else 0.0
                )
            }
            if refreshed_statistics
            else {}
        ),
    }


def aggregate_diagnostics(
    groups: list[LatentRolloutBatch],
    samples_per_prompt: int,
    stop_ids: tuple[int, ...] = (),
    *,
    refreshed_statistics: bool = True,
    think_fence_ids: tuple[int, int] | None = None,
    tokenizer=None,
    min_think_tokens: int = 1,
    answer_fence_ids: tuple[int, int] | None = None,
    reward_floor: float = 0.0,
) -> dict[str, float | int]:
    """Mean of per-group rollout diagnostics; trajectory counts are summed
    and reward extrema pooled."""
    per_group = [
        rollout_diagnostics(
            group,
            samples_per_prompt,
            stop_ids,
            refreshed_statistics=refreshed_statistics,
            think_fence_ids=think_fence_ids,
            tokenizer=tokenizer,
            min_think_tokens=min_think_tokens,
            answer_fence_ids=answer_fence_ids,
            reward_floor=reward_floor,
        )
        for group in groups
    ]
    aggregated: dict[str, float | int] = {}
    # Derived per-group keys (currently the conditional think mean) can
    # be present in some groups and absent in others; they are rebuilt
    # from their pooled numerators below, never averaged group-wise.
    derived = {"think_tokens_mean", "abstained_emitted_tokens_mean"}
    # Intersection, not per_group[0]'s keys: conditional keys (the
    # gate-zeroed fraction depends on per-batch scoring state) may be
    # present in some groups only, and indexing them into every group
    # would KeyError — or silently vanish — depending on group order.
    shared_keys = set(per_group[0])
    for metrics in per_group[1:]:
        shared_keys &= set(metrics)
    for key in per_group[0]:
        if key in derived or key not in shared_keys:
            continue
        values = [metrics[key] for metrics in per_group]
        if key == "trajectories":
            aggregated[key] = int(sum(values))
        elif key in (
            "think_tokens_sum", "think_tokens_count",
            "abstained_emitted_tokens_sum", "abstained_emitted_tokens_count",
        ):
            aggregated[key] = float(sum(values))
        elif key == "abstained_emitted_tokens_max":
            aggregated[key] = float(max(values))
        elif key == "reward_max":
            # A mean of per-group extrema reads max == min whenever every
            # group is internally constant, though groups differ.
            aggregated[key] = float(max(values))
        elif key == "reward_min":
            aggregated[key] = float(min(values))
        else:
            aggregated[key] = float(sum(values) / len(values))
    if aggregated.get("think_tokens_count"):
        aggregated["think_tokens_mean"] = float(
            aggregated["think_tokens_sum"] / aggregated["think_tokens_count"]
        )
    if aggregated.get("abstained_emitted_tokens_count"):
        aggregated["abstained_emitted_tokens_mean"] = float(
            aggregated["abstained_emitted_tokens_sum"]
            / aggregated["abstained_emitted_tokens_count"]
        )
    return aggregated


def source_group_id(group: LatentRolloutBatch) -> int:
    """Return the one source represented by a prompt group, fail closed."""
    if group.source_id is None or group.source_id.numel() != group.kind.size(0):
        raise ValueError("multi-source rollout group has missing source labels")
    labels = group.source_id.unique()
    if labels.numel() != 1:
        raise ValueError("one prompt group cannot contain multiple sources")
    return int(labels.item())


def source_diagnostics(
    groups: list[LatentRolloutBatch],
    source_names: list[str],
    samples_per_prompt: int,
    stop_ids: tuple[int, ...] = (),
    *,
    refreshed_statistics: bool = True,
    think_fence_ids: tuple[int, int] | None = None,
    tokenizer=None,
    min_think_tokens: int = 1,
    answer_fence_ids: tuple[int, int] | None = None,
    reward_floor: float = 0.0,
) -> dict[str, dict[str, float | int]]:
    """Compute the same rollout diagnostics independently for every source.

    An exact-pass mixture gives a small source well under one prompt per
    pool, so a source absent from this pool is omitted rather than reported
    as a zero it never earned.
    """
    by_source: dict[int, list[LatentRolloutBatch]] = {}
    for group in groups:
        source_id = source_group_id(group)
        if not 0 <= source_id < len(source_names):
            raise ValueError(f"rollout carries unknown source id {source_id}")
        by_source.setdefault(source_id, []).append(group)
    return {
        source_names[source_id]: aggregate_diagnostics(
            source_groups,
            samples_per_prompt,
            stop_ids,
            refreshed_statistics=refreshed_statistics,
            think_fence_ids=think_fence_ids,
            tokenizer=tokenizer,
            min_think_tokens=min_think_tokens,
            answer_fence_ids=answer_fence_ids,
            reward_floor=reward_floor,
        )
        for source_id, source_groups in sorted(by_source.items())
    }


def lockstep_decode_metrics(
    groups: list[LatentRolloutBatch],
    chunk_groups: int,
) -> dict[str, float]:
    """How many decode steps a chunk paid for the actions it actually kept.

    A chunk's rows step together until its LAST row finishes, so the chunk
    costs one decode step per action of its longest trajectory while the mean
    trajectory stops far earlier. ``groups`` arrives in chunk order,
    ``chunk_groups`` per chunk, so the per-chunk maximum is the step count and
    the ratio against the mean action count is the share of decode work that
    produced nothing. Generation dominates pool wall time, so this ratio, not
    the batch width, is what bounds collection.

    Row length comes from ``action_mask``, NOT from ``stream_length``: groups
    reach this point through ``trim_stream``, which rounds the kept length UP
    to ``--replay-bucket`` and pads past the original stream to bound the set
    of compiled replay shapes. That bucketing would add up to a full bucket of
    pool-to-pool jitter to a metric whose whole job is to detect a change.
    The count still lands on the loop's SYNC_EVERY boundary, so it measures
    productive steps and omits the <=15 no-op steps after the last row ends.
    """
    if not groups:
        raise ValueError("at least one rollout group is required")
    # Both reductions cross the bus in ONE transfer. They used to be two
    # ``.tolist()``/``float()`` calls, and this runs inside the pool's own
    # timers at maximum queue depth, so each one drained the pipeline and was
    # billed to the collection it was measuring -- enough to swamp the effect
    # of anything being measured. Stacking them costs one sync instead.
    per_group = torch.stack(
        [
            torch.stack((group.action_mask.sum(1).max(), group.action_mask.sum()))
            for group in groups
        ]
    ).tolist()
    row_actions = [longest for longest, _ in per_group]
    chunk_steps = [
        max(row_actions[start : start + chunk_groups])
        for start in range(0, len(row_actions), chunk_groups)
    ]
    steps_mean = sum(chunk_steps) / len(chunk_steps)
    actions = sum(total for _, total in per_group)
    rows = sum(group.kind.size(0) for group in groups)
    return {
        "decode_steps_per_chunk_mean": steps_mean,
        "decode_steps_per_chunk_max": float(max(chunk_steps)),
        "decode_steps_total": float(sum(chunk_steps)),
        "decode_step_utilization": (
            (actions / rows) / steps_mean if steps_mean else 0.0
        ),
    }


def _weighted_metric_mean(
    metrics: list[dict[str, float]], key: str, weight_key: str
) -> float:
    total_weight = sum(metric[weight_key] for metric in metrics)
    if total_weight <= 0:
        return 0.0
    return sum(
        metric[key] * metric[weight_key] for metric in metrics
    ) / total_weight


def aggregate_value_diagnostics(
    metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Pool token moments across an online sequence of critic states."""
    if not metrics:
        raise ValueError("at least one metric group is required")
    target_mean = _weighted_metric_mean(
        metrics, "value_target_mean", "action_count"
    )
    residual_mean = _weighted_metric_mean(
        metrics, "value_residual_mean", "action_count"
    )
    action_count = sum(metric["action_count"] for metric in metrics)
    if action_count <= 0:
        raise ValueError("value diagnostics require at least one action")
    target_second_moment = sum(
        (
            metric["value_target_variance"]
            + metric["value_target_mean"] ** 2
        )
        * metric["action_count"]
        for metric in metrics
    ) / action_count
    residual_second_moment = sum(
        (
            metric["value_residual_variance"]
            + metric["value_residual_mean"] ** 2
        )
        * metric["action_count"]
        for metric in metrics
    ) / action_count
    target_variance = max(0.0, target_second_moment - target_mean**2)
    residual_variance = max(0.0, residual_second_moment - residual_mean**2)
    explained_variance = (
        1.0 - residual_variance / target_variance
        if target_variance > 1e-12
        else 0.0
    )
    return {
        "prediction_mean": _weighted_metric_mean(
            metrics, "value_mean", "action_count"
        ),
        "target_mean": target_mean,
        "residual_mean": residual_mean,
        "target_variance": target_variance,
        "residual_variance": residual_variance,
        "explained_variance": explained_variance,
        "loss": _weighted_metric_mean(metrics, "value_loss", "action_count"),
    }


def aggregate_action_value_diagnostics(
    metrics: list[dict[str, float]],
) -> dict[str, float]:
    """``aggregate_value_diagnostics`` for the action head's Q, which fits
    the same targets."""
    return aggregate_value_diagnostics(
        [
            {
                **metric,
                "value_residual_mean": metric["action_value_residual_mean"],
                "value_residual_variance": metric[
                    "action_value_residual_variance"
                ],
                "value_loss": metric["action_value_loss"],
            }
            for metric in metrics
        ]
    )


def aggregate_mc_value_diagnostics(
    metrics: list[dict[str, float]],
) -> dict[str, float]:
    """``aggregate_value_diagnostics`` for a Q critic's fit against the
    Monte-Carlo return rather than its own bootstrapped targets."""
    return aggregate_value_diagnostics(
        [
            {
                **metric,
                "value_target_mean": metric["mc_target_mean"],
                "value_target_variance": metric["mc_target_variance"],
                "value_residual_mean": metric["mc_residual_mean"],
                "value_residual_variance": metric["mc_residual_variance"],
            }
            for metric in metrics
        ]
    )


def post_think_state_mask(
    token_ids: torch.Tensor, emit_mask: torch.Tensor, think_close_id: int
) -> torch.Tensor:
    """Emit slots whose state has consumed a completion's closing think fence.

    The token at a slot is already consumed, so the slot that emits the
    fence is still a think state and the one holding it is the first answer
    state. Only fences after the first emit slot count: the prompt is not
    the completion.
    """
    emits = emit_mask.bool()
    in_completion = torch.zeros_like(emits)
    in_completion[:, 1:] = emits[:, :-1].cumsum(1) > 0
    closed = ((token_ids == think_close_id) & in_completion).cumsum(1) > 0
    return closed & emits


def credit_assignment_moments(
    batch: LatentRolloutBatch,
    advantages: torch.Tensor,
    value_targets: torch.Tensor,
    actor_signal_rows: torch.Tensor,
    reward_floor: float = 0.0,
    action_values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Raw token moments for the credit-assignment health checks.

    - Calibration of the critic's one-step differences: regressing the
      Monte-Carlo advantage (lambda-1 return minus value) on the TD(0)
      advantage gives slope 1 for a calibrated critic, about 1/2 when critic
      noise dominates the differences, and above 1 when the critic
      under-reacts. Lambda-0 actor advantages are trustworthy only near 1.
      Only non-terminal slots enter: at a row's last action both residuals
      are R - V exactly, and those identical points would pull a pure-noise
      critic's slope from 1/2 toward 1.
    - Advantage versus sampled-token surprisal on failed rows the actor
      trains on (rows a source gate zeroed carry no pseudo-reward; failed
      means the zero-score reward ``reward_floor``, never an abstention):
      a critic whose differences reward rare tokens regardless of outcome
      is a pseudo-reward that Delightful PG amplifies into entropy growth.
    - With ``action_values``, the same calibration for the action head's
      residual Q - V, over every action slot: its terminal residual is a
      prediction, not R - V, so no slot is degenerate.

    Sums, not ratios, so groups pool exactly in the dashboard. Masked
    products rather than boolean indexing keep the update free of host syncs.
    """
    # An action slot followed by another action slot is non-terminal.
    actions = batch.action_mask.double()
    actions = torch.cat((actions[:, :-1] * actions[:, 1:], actions[:, -1:] * 0), dim=1)
    td0 = temporal_difference_residuals(
        batch.rewards, batch.old_values, batch.action_mask
    ).double() * actions
    mc = (value_targets - batch.old_values).double() * actions
    failed_rows = batch.reward_scalar <= reward_floor
    if batch.verifier_status is not None:
        failed_rows &= batch.verifier_status != PYTHON_RESULT_CODES["abstained"]
    failed = batch.emit_mask.double() * (
        failed_rows * actor_signal_rows
    )[:, None].double()
    failed_advantage = advantages.double() * failed
    failed_surprisal = -batch.old_token_logprobs.double() * failed
    action_moments = {}
    if action_values is not None:
        every_action = (
            batch.action_mask * actor_signal_rows[:, None]
        ).double()
        action_residual = (
            action_values - batch.old_values
        ).double() * every_action
        every_mc = (value_targets - batch.old_values).double() * every_action
        action_moments = {
            "credit_action_count": every_action.sum(),
            "credit_action_residual_sum": action_residual.sum(),
            "credit_action_residual_square_sum": action_residual.square().sum(),
            "credit_action_mc_sum": every_mc.sum(),
            "credit_action_mc_square_sum": every_mc.square().sum(),
            "credit_action_residual_mc_sum": (action_residual * every_mc).sum(),
        }
    return {
        **action_moments,
        "credit_count": actions.sum(),
        "credit_td0_sum": td0.sum(),
        "credit_mc_sum": mc.sum(),
        "credit_td0_square_sum": td0.square().sum(),
        "credit_td0_mc_sum": (td0 * mc).sum(),
        "failed_count": failed.sum(),
        "failed_advantage_sum": failed_advantage.sum(),
        "failed_surprisal_sum": failed_surprisal.sum(),
        "failed_advantage_square_sum": failed_advantage.square().sum(),
        "failed_surprisal_square_sum": failed_surprisal.square().sum(),
        "failed_advantage_surprisal_sum": (
            failed_advantage * failed_surprisal
        ).sum(),
    }


def credit_assignment_dashboard(
    metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Pool ``credit_assignment_moments`` across groups into the two checks."""
    total = {
        key: sum(metric[key] for metric in metrics)
        for key in metrics[0]
        if key.startswith(("credit_", "failed_"))
    }

    def covariance(first: str, second: str, cross: str, count: str) -> float:
        n = max(total[count], 1.0)
        return total[cross] / n - (total[first] / n) * (total[second] / n)

    td0_variance = covariance(
        "credit_td0_sum", "credit_td0_sum", "credit_td0_square_sum", "credit_count"
    )
    advantage_variance = covariance(
        "failed_advantage_sum", "failed_advantage_sum",
        "failed_advantage_square_sum", "failed_count",
    )
    surprisal_variance = covariance(
        "failed_surprisal_sum", "failed_surprisal_sum",
        "failed_surprisal_square_sum", "failed_count",
    )
    scale = math.sqrt(max(advantage_variance, 0.0) * max(surprisal_variance, 0.0))
    # A statistic without variance to normalize by is undefined, not zero;
    # omit it rather than log a value indistinguishable from a real result.
    dashboard = {}
    if td0_variance > 1e-18:
        dashboard["credit/td0_calibration_slope"] = covariance(
            "credit_td0_sum", "credit_mc_sum", "credit_td0_mc_sum", "credit_count"
        ) / td0_variance
    if "credit_action_count" in total:
        residual_variance = covariance(
            "credit_action_residual_sum", "credit_action_residual_sum",
            "credit_action_residual_square_sum", "credit_action_count",
        )
        mc_variance = covariance(
            "credit_action_mc_sum", "credit_action_mc_sum",
            "credit_action_mc_square_sum", "credit_action_count",
        )
        residual_mc = covariance(
            "credit_action_residual_sum", "credit_action_mc_sum",
            "credit_action_residual_mc_sum", "credit_action_count",
        )
        dashboard["credit/action_residual_std"] = math.sqrt(
            max(residual_variance, 0.0)
        )
        if residual_variance > 1e-18:
            dashboard["credit/action_residual_calibration_slope"] = (
                residual_mc / residual_variance
            )
        if residual_variance > 1e-18 and mc_variance > 1e-18:
            dashboard["credit/action_residual_mc_corr"] = residual_mc / math.sqrt(
                residual_variance * mc_variance
            )
    if scale > 1e-18:
        dashboard["credit/failed_advantage_surprisal_corr"] = covariance(
            "failed_advantage_sum", "failed_surprisal_sum",
            "failed_advantage_surprisal_sum", "failed_count",
        ) / scale
    return dashboard


def advantage_group_dashboard(
    metrics: list[dict[str, float]], source_names: Sequence[str]
) -> dict[str, float]:
    """Mean and RMS advantage per ``update_minibatch`` advantage group,
    pooled over prompt groups, with source ids named."""
    totals: dict[str, dict[str, float]] = {}
    for metric in metrics:
        for key, value in metric.items():
            if key.startswith("advantage_group/"):
                _, name, moment = key.split("/")
                group = totals.setdefault(name, {})
                group[moment] = group.get(moment, 0.0) + value
    dashboard = {}
    for name, group in totals.items():
        if name.startswith("source_"):
            label = f"advantage/source/{source_names[int(name[7:])]}"
        else:
            label = f"advantage/token/{name}"
        count = group["count"]
        dashboard[f"{label}/tokens"] = count
        if count:
            dashboard[f"{label}/mean"] = group["sum"] / count
            dashboard[f"{label}/rms"] = math.sqrt(group["square_sum"] / count)
    return dashboard


ANSWER_GATE_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def sequence_estimator_moments(estimator: str, advantages: torch.Tensor,
                               surprisal: torch.Tensor, coefficients: torch.Tensor,
                               tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """Row sums of a sequence estimator's group terms, as device scalars:
    rows; the answer-cluster surprisal l, whose row mean is the mean
    per-group plug-in answer entropy (every group has G rows); rows of a
    unanimous group (l = 0); the squares of U and of the coefficient; and
    both weighted by the row's emitted tokens |.| T, the share of the
    token-normalized update each row carries. The answer gate adds its sum
    and count over rows with U != 0, split by the sign of U, and its
    quantiles there (``ANSWER_GATE_QUANTILES``, NaN if no row has U != 0).
    """
    moments = {
        "mac/sequence/rows": torch.tensor(float(advantages.numel()), device=advantages.device),
        "mac/sequence/surprisal_sum": surprisal.sum(),
        "mac/sequence/unanimous_rows": (surprisal == 0).sum(),
        "mac/sequence/advantage_square_sum": advantages.square().sum(),
        "mac/sequence/coefficient_square_sum": coefficients.square().sum(),
        "mac/sequence/advantage_token_abs_sum": (advantages.abs() * tokens).sum(),
        "mac/sequence/coefficient_token_abs_sum": (coefficients.abs() * tokens).sum(),
    }
    if estimator == "delightful_answer_gate":
        gate = answer_gate(advantages, surprisal)
        for sign, mask in (("positive", advantages > 0), ("negative", advantages < 0)):
            moments[f"mac/answer_gate_{sign}_sum"] = (gate * mask).sum()
            moments[f"mac/answer_gate_{sign}_count"] = mask.sum()
        active = torch.where(advantages != 0, gate, torch.full_like(gate, math.nan))
        for quantile, value in zip(ANSWER_GATE_QUANTILES, torch.nanquantile(
                active, torch.tensor(ANSWER_GATE_QUANTILES, device=gate.device))):
            moments[f"mac/answer_gate_q{round(quantile * 100):02d}"] = value
    return moments


def mac_dashboard(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Per span: MAC's policy variance of Q, sum_a pi(a) A(a)^2 per state,
    the critic's action contrast on the policy's support (a gain only for a
    logit step along A, not for MAC's update along pi*A); and the mean
    and RMS of the sampled log-probability's return coefficient, the scale
    of the estimator's sampled term. Spans are counted by
    ``advantage_group/span_*``. Under the delightful sampled baseline, also
    its delight gate's mean by span and by the sign of G - c."""
    dashboard = {}
    for span in ("think", "answer"):
        key = f"mac/{span}/q_variance_sum"
        if not any(key in metric for metric in metrics):
            continue
        count = sum(metric[f"advantage_group/span_{span}/count"] for metric in metrics)
        dashboard[f"mac/{span}_states"] = count
        if count:
            def total(name: str, span: str = span) -> float:
                return sum(metric[f"mac/{span}/{name}_sum"] for metric in metrics)

            dashboard[f"mac/{span}_policy_q_variance"] = total("q_variance") / count
            dashboard[f"mac/{span}_coefficient_mean"] = total("coefficient") / count
            dashboard[f"mac/{span}_coefficient_rms"] = math.sqrt(
                total("coefficient_square") / count)
            if any(f"mac/{span}/gate_sum" in metric for metric in metrics):
                dashboard[f"mac/{span}_delight_gate_mean"] = total("gate") / count
    for sign in ("positive", "negative"):
        if any(f"mac/gate_{sign}_count" in metric for metric in metrics):
            count = sum(metric[f"mac/gate_{sign}_count"] for metric in metrics)
            if count:
                # The gate where G - c has this sign: DG's asymmetry between
                # reinforcing and suppressing a surprising token.
                dashboard[f"mac/delight_gate_{sign}_mean"] = sum(
                    metric[f"mac/gate_{sign}_sum"] for metric in metrics) / count
            mass = sum(metric[f"mac/gate_{sign}_mass_sum"] for metric in metrics)
            if mass:
                # The same gate weighted by |r| (1 - pi(a_t)), the size of
                # the term it scales, and that mass's share with
                # |r * surprisal| > 1.
                dashboard[f"mac/delight_gate_{sign}_mass_mean"] = sum(
                    metric[f"mac/gate_{sign}_weighted_sum"] for metric in metrics) / mass
                dashboard[f"mac/delight_{sign}_active_mass_share"] = sum(
                    metric[f"mac/gate_{sign}_active_mass_sum"] for metric in metrics) / mass

    def total(name: str) -> float:
        return sum(metric[f"mac/{name}_sum"] for metric in metrics)

    # A critic gate's advantage over non-terminal generated states: its
    # RMS; its correlation with r = G - c, how much of the outcome the
    # critic attributes to the tokens it gates; and the slope of r on A,
    # 1 for a calibrated critic. For the TD residual (r_t - A_t = r_{t+1}
    # has zero mean given s_{t+1}) it is about 1/2 when critic noise
    # dominates A. ``head`` is the action head's A(a_t); ``shuffled_head``
    # is it after the shuffled gate's within-stratum permutation, whose
    # slope keeps only what the strata carry.
    for source, label in (("gate", "td"), ("head", "head"), ("shuffled", "shuffled_head")):
        if not any(f"mac/{source}_advantage_square_sum" in metric for metric in metrics):
            continue
        states = total(f"{source}_advantage_states")
        square, residual = total(f"{source}_advantage_square"), total("residual_square")
        if states:
            dashboard[f"mac/{label}_advantage_rms"] = math.sqrt(square / states)
        if square and residual:
            dashboard[f"mac/{label}_advantage_residual_correlation"] = total(
                f"{source}_advantage_residual") / math.sqrt(square * residual)
        if square:
            dashboard[f"mac/{label}_advantage_calibration_slope"] = total(
                f"{source}_advantage_residual") / square
    if any("mac/shuffle_head_cross_sum" in metric for metric in metrics):
        head, shuffled = total("head_advantage_square"), total("shuffled_advantage_square")
        if head and shuffled:
            dashboard["mac/shuffle_head_correlation"] = total(
                "shuffle_head_cross") / math.sqrt(head * shuffled)
        mass = total("shuffle_mass")
        if mass:
            dashboard["mac/shuffle_fixed_mass_share"] = total("shuffle_fixed_mass") / mass
    if any("mac/gate_nonterminal_mass_sum" in metric for metric in metrics):
        mass = total("gate_nonterminal_mass")
        if mass:
            dashboard["mac/delight_gate_nonterminal_mass_mean"] = total(
                "gate_nonterminal_weighted") / mass
            dashboard["mac/delight_nonterminal_active_mass_share"] = total(
                "gate_nonterminal_active_mass") / mass
    for span in ("think", "answer"):
        # The sampled baseline's G - c on the same states, beside a sequence
        # estimator's coefficient RMS: the two arms' scales, step for step.
        key = f"mac/{span}/sb_coefficient_square_sum"
        count = dashboard.get(f"mac/{span}_states")
        if count and any(key in metric for metric in metrics):
            dashboard[f"mac/{span}_sb_coefficient_rms"] = math.sqrt(
                sum(metric[key] for metric in metrics) / count)
    if any("mac/sequence/rows" in metric for metric in metrics):
        def rows_total(name: str) -> float:
            return sum(metric[f"mac/sequence/{name}"] for metric in metrics)

        rows = rows_total("rows")
        dashboard["mac/answer_cluster_entropy_mean"] = rows_total("surprisal_sum") / rows
        dashboard["mac/answer_unanimous_share"] = rows_total("unanimous_rows") / rows
        dashboard["mac/sequence_advantage_rms"] = math.sqrt(
            rows_total("advantage_square_sum") / rows)
        dashboard["mac/sequence_coefficient_rms"] = math.sqrt(
            rows_total("coefficient_square_sum") / rows)
        mass = rows_total("advantage_token_abs_sum")
        if mass:
            # |omega| over |U| where the token-normalized update puts its
            # weight: the gate's mean weighted by |U| T (1 for RLOO).
            dashboard["mac/answer_gate_token_mass_mean"] = rows_total(
                "coefficient_token_abs_sum") / mass
    for sign in ("positive", "negative"):
        if any(f"mac/answer_gate_{sign}_count" in metric for metric in metrics):
            count = sum(metric[f"mac/answer_gate_{sign}_count"] for metric in metrics)
            if count:
                dashboard[f"mac/answer_gate_{sign}_mean"] = sum(
                    metric[f"mac/answer_gate_{sign}_sum"] for metric in metrics) / count
    for quantile in ANSWER_GATE_QUANTILES:
        key = f"mac/answer_gate_q{round(quantile * 100):02d}"
        values = [metric[key] for metric in metrics if math.isfinite(metric.get(key, math.nan))]
        if values:
            # One optimizer minibatch per step in the drills; a mean of
            # minibatch quantiles otherwise.
            dashboard[key] = sum(values) / len(values)
    return dashboard


def support_dashboard(metrics: list[dict[str, float]]) -> dict[str, float]:
    """The critic's policy-centered Q that MAC read this update, by the
    policy's support: its mean and positive fraction at tokens with
    pi < 1e-4 (no regression data), at ranks 2-8, and its mean at the top
    token (``mean_actor_critic.SUPPORT_CENSUS_FIELDS``)."""
    if not any("support/top_count_sum" in metric for metric in metrics):
        return {}
    total = {name: sum(metric[f"support/{name}_sum"] for metric in metrics)
             for name in SUPPORT_CENSUS_FIELDS}
    dashboard = {}
    for span in ("offsupport", "rank2to8"):
        count = total[f"{span}_count"]
        if count:
            dashboard[f"support/{span}_advantage_mean"] = total[f"{span}_advantage"] / count
            dashboard[f"support/{span}_positive_fraction"] = total[f"{span}_positive"] / count
    if total["top_count"]:
        dashboard["support/offsupport_tokens_per_state"] = total["offsupport_count"] / total["top_count"]
        dashboard["support/top_advantage_mean"] = total["top_advantage"] / total["top_count"]
    return dashboard


def q_prior_dashboard(metrics: list[dict[str, float]]) -> dict[str, float]:
    """The pessimistic Q prior per head and state: its unweighted term,
    sum_a relu(A_h)^2 under the current policy, and how many tokens read
    better than the policy's value (``postraining.pessimistic_q``)."""
    count = sum(metric.get("q_prior/count", 0.0) for metric in metrics)
    if not count:
        return {}
    return {
        "q_prior/penalty_per_state": sum(metric["q_prior/penalty_sum"] for metric in metrics) / count,
        "q_prior/positive_tokens_per_state": (
            sum(metric["q_prior/positive_sum"] for metric in metrics) / count),
    }


def twin_dashboard(metrics: list[dict[str, float]]) -> dict[str, float]:
    """The twin Q heads' disagreement on the taken token: the RMS of
    Q1 - Q2, and the mean of their mean minus their min, by how much the
    clipped double-Q bootstrap sits below the Q the actor reads."""
    if not any("twin/action_count" in metric for metric in metrics):
        return {}
    count = sum(metric["twin/action_count"] for metric in metrics)
    if not count:
        return {}
    return {
        "twin/gap_rms": math.sqrt(
            sum(metric["twin/gap_square_sum"] for metric in metrics) / count),
        "twin/min_shift_mean": sum(
            metric["twin/min_shift_sum"] for metric in metrics) / count,
    }


def belief_gain_dashboard(critic) -> dict[str, float]:
    """Each token Q head's belief gain g before the step: u . (x - xbar)
    holds (g / r)(z - zbar), so g is how much Q follows the critic trunk's
    own centred logit and how hard MAC sharpens the actor toward it."""
    if getattr(critic, "q_head", None) is None:
        return {}
    names = ("critic_q/belief_gain", "critic_q/belief_gain_twin")
    return {name: float(head.belief_gain.detach())
            for name, head in zip(names, critic.q_heads, strict=False)}


def aggregate_actor_tensorboard_metrics(
    metrics: list[dict[str, float]],
) -> dict[str, float]:
    """One compact, correctly weighted dashboard row per actor step."""
    if not metrics:
        raise ValueError("at least one actor metric group is required")

    value = aggregate_value_diagnostics(metrics)
    advantage_mean = _weighted_metric_mean(
        metrics, "advantage_mean", "action_count"
    )
    advantage_second_moment = _weighted_metric_mean(
        [
            {
                **metric,
                "advantage_second_moment": (
                    metric["advantage_std"] ** 2
                    + metric["advantage_mean"] ** 2
                ),
            }
            for metric in metrics
        ],
        "advantage_second_moment",
        "action_count",
    )
    advantage_std = math.sqrt(
        max(0.0, advantage_second_moment - advantage_mean**2)
    )
    last = metrics[-1]
    # Each group is already divided by the denominator of the complete actor
    # optimizer minibatch, so its loss is a contribution to be summed rather
    # than another independently normalized minibatch mean.
    policy_contribution = sum(metric["policy_loss"] for metric in metrics)
    dashboard = {
        "loss/policy": policy_contribution,
        "loss/actor_total": policy_contribution,
        ("value/token_weighted_bce" if "binary_success_bce" in last
         else "value/token_weighted_mse"): value["loss"],
        "value/prediction_mean": value["prediction_mean"],
        "value/target_mean": value["target_mean"],
        "value/residual_mean": value["residual_mean"],
        "value/online_epoch_explained_variance": value["explained_variance"],
        "advantage/mean": advantage_mean,
        "advantage/std": advantage_std,
        "kl/token_behavior": _weighted_metric_mean(
            metrics, "token_behavior_kl", "action_count"
        ),
        "kl/policy_behavior_per_action": sum(
            metric["policy_behavior_kl_per_action"] for metric in metrics
        ),
        "clip/policy": sum(
            metric["policy_clip_fraction"] for metric in metrics
        ),
        "loss/stochastic_policy": sum(
            metric.get("stochastic_policy_loss", 0.0) for metric in metrics
        ),
        "clip/stochastic_policy": sum(
            metric.get("stochastic_policy_clip_fraction", 0.0)
            for metric in metrics
        ),
        "ratio/token_abs_log_max": max(
            metric["token_abs_log_ratio_max"] for metric in metrics
        ),
        "ratio/harmful_positive_log_max": max(
            metric["harmful_positive_log_ratio_max"] for metric in metrics
        ),
        # Both optimizers accumulate across groups; only the final group has
        # the complete pre-step gradient norm.
        "grad/trunk": last["trunk_grad_norm"],
        "grad/renderer": last["renderer_grad_norm"],
        "grad/combiner": last["combiner_grad_norm"],
        "grad/critic": last["critic_grad_norm"],
    }
    if "credit_count" in last:
        dashboard.update(credit_assignment_dashboard(metrics))
    if "mc_explained_variance" in last:
        # A Q critic's ``value/*`` rows fit its generalized value estimates;
        # these are the same predictions against the Monte-Carlo return.
        mc_value = aggregate_mc_value_diagnostics(metrics)
        dashboard["value/mc_target_mean"] = mc_value["target_mean"]
        dashboard["value/mc_residual_mean"] = mc_value["residual_mean"]
        dashboard["value/mc_explained_variance"] = mc_value[
            "explained_variance"
        ]
    if "entropy_loss" in last:
        entropy_loss = sum(metric["entropy_loss"] for metric in metrics)
        dashboard["loss/entropy"] = entropy_loss
        dashboard["loss/actor_total"] += entropy_loss
    if "policy_entropy_sum" in last:
        entropy_count = sum(metric["policy_entropy_count"] for metric in metrics)
        # Undefined without an active emit slot, so omitted rather than 0.
        if entropy_count > 0:
            dashboard["entropy/policy_mean"] = sum(
                metric["policy_entropy_sum"] for metric in metrics
            ) / entropy_count
    if "binary_success_bce" in last:
        dashboard["value/token_weighted_brier"] = _weighted_metric_mean(
            metrics, "binary_success_brier", "action_count"
        )
        dashboard["value/binary_success_r_squared"] = 1.0 - (
            _weighted_metric_mean(metrics, "binary_success_brier", "action_count")
            / max(value["target_variance"], 1e-12)
        ) if value["target_variance"] > 1e-12 else 0.0
        dashboard["q/actor_success_probability_mean"] = advantage_mean
        dashboard["q/actor_success_probability_std"] = advantage_std
        dashboard["q/binary_dg_coefficient_mean"] = _weighted_metric_mean(
            metrics, "binary_dg_coefficient_mean", "action_count"
        )
    if "action_value_loss" in last:
        action_value = aggregate_action_value_diagnostics(metrics)
        dashboard["value/action_value_mse"] = action_value["loss"]
        dashboard["value/action_value_residual_mean"] = action_value[
            "residual_mean"
        ]
        dashboard["value/action_value_online_explained_variance"] = (
            action_value["explained_variance"]
        )
        dashboard["grad/action_head"] = last["action_head_grad_norm"]
    if "carry_input_delta_rms_ratio" in last:
        # The hidden carry has no stochastic factor or head to report.
        dashboard.pop("loss/stochastic_policy")
        dashboard.pop("clip/stochastic_policy")
        dashboard["carry/input_delta_rms_ratio"] = _weighted_metric_mean(
            metrics, "carry_input_delta_rms_ratio", "action_count"
        )
        if "memory_write_beta_mean" in last:
            dashboard["memory/write_beta_mean"] = _weighted_metric_mean(
                metrics, "memory_write_beta_mean", "action_count"
            )
            dashboard["memory/decay_log_timescale_mean"] = (
                _weighted_metric_mean(
                    metrics, "memory_decay_log_timescale_mean", "action_count"
                )
            )
    else:
        dashboard["grad/gate"] = last["gate_grad_norm"]
        dashboard["grad/thought_mean"] = last["thought_mean_grad_norm"]
    if "delightful_gate_mean" in last:
        dashboard.pop("clip/policy")
        dashboard.pop("ratio/harmful_positive_log_max")
        dashboard.update(
            {
                "delightful/gate_mean": _weighted_metric_mean(
                    metrics, "delightful_gate_mean", "action_count"
                ),
                "delightful/positive_gate_mean": _weighted_metric_mean(
                    metrics,
                    "delightful_positive_gate_mean",
                    "delightful_positive_count",
                ),
                "delightful/negative_gate_mean": _weighted_metric_mean(
                    metrics,
                    "delightful_negative_gate_mean",
                    "delightful_negative_count",
                ),
                "delightful/delight_mean": _weighted_metric_mean(
                    metrics, "delightful_delight_mean", "action_count"
                ),
                "delightful/surprisal_mean": _weighted_metric_mean(
                    metrics, "delightful_surprisal_mean", "action_count"
                ),
                "delightful/reinforce_positive_advantage": sum(
                    metric["delightful_positive_loss_contribution"]
                    for metric in metrics
                ),
                "delightful/suppress_negative_advantage": sum(
                    metric["delightful_negative_loss_contribution"]
                    for metric in metrics
                ),
            }
        )
    if "tpo_target_probability_mean" in last:
        dashboard.pop("clip/policy")
        dashboard.pop("ratio/harmful_positive_log_max")
        target_log_odds_shift_rms = math.sqrt(
            max(
                0.0,
                _weighted_metric_mean(
                    metrics,
                    "tpo_target_log_odds_shift_square_mean",
                    "tpo_active_count",
                ),
            )
        )
        pre_update_residual_rms = math.sqrt(
            max(
                0.0,
                _weighted_metric_mean(
                    metrics,
                    "tpo_pre_update_probability_residual_square_mean",
                    "tpo_active_count",
                ),
            )
        )
        dashboard.update(
            {
                "tpo/loss": _weighted_metric_mean(
                    metrics, "tpo_loss", "tpo_active_count"
                ),
                "tpo/pre_update_fit_kl": _weighted_metric_mean(
                    metrics, "tpo_pre_update_fit_kl", "tpo_active_count"
                ),
                "tpo/target_behavior_kl": _weighted_metric_mean(
                    metrics, "tpo_target_behavior_kl", "tpo_active_count"
                ),
                "tpo/old_probability_mean": _weighted_metric_mean(
                    metrics, "tpo_old_probability_mean", "tpo_active_count"
                ),
                "tpo/pre_update_probability_mean": _weighted_metric_mean(
                    metrics,
                    "tpo_pre_update_probability_mean",
                    "tpo_active_count",
                ),
                "tpo/target_probability_mean": _weighted_metric_mean(
                    metrics,
                    "tpo_target_probability_mean",
                    "tpo_active_count",
                ),
                "tpo/target_move_abs_mean": _weighted_metric_mean(
                    metrics, "tpo_target_move_abs_mean", "tpo_active_count"
                ),
                "tpo/target_log_odds_shift_abs_mean": _weighted_metric_mean(
                    metrics,
                    "tpo_target_log_odds_shift_abs_mean",
                    "tpo_active_count",
                ),
                "tpo/target_log_odds_shift_rms": target_log_odds_shift_rms,
                "tpo/pre_update_probability_residual_mean": (
                    _weighted_metric_mean(
                        metrics,
                        "tpo_pre_update_probability_residual_mean",
                        "tpo_active_count",
                    )
                ),
                "tpo/pre_update_probability_residual_abs_mean": _weighted_metric_mean(
                    metrics,
                    "tpo_pre_update_probability_residual_abs_mean",
                    "tpo_active_count",
                ),
                "tpo/pre_update_probability_residual_rms": (
                    pre_update_residual_rms
                ),
                "tpo/positive_target_probability_mean": _weighted_metric_mean(
                    metrics,
                    "tpo_positive_target_probability_mean",
                    "tpo_positive_count",
                ),
                "tpo/negative_target_probability_mean": _weighted_metric_mean(
                    metrics,
                    "tpo_negative_target_probability_mean",
                    "tpo_negative_count",
                ),
            }
        )
    return dashboard


def rollout_tensorboard_metrics(metrics: dict[str, float | int]) -> dict[str, float]:
    """Select semantic rollout signals while leaving rich telemetry in JSON."""
    return {
        "reward/mean": float(metrics["reward_mean"]),
        "reward/within_group_std": float(metrics["within_group_reward_std"]),
        "reward/exact_accuracy": float(metrics["exact_accuracy"]),
        "reward/exact_within_group_std": float(
            metrics["exact_within_group_reward_std"]
        ),
        "reward/partial_fraction": float(metrics["partial_reward_fraction"]),
        "reward/ended_fraction": float(metrics["ended_fraction"]),
        **(
            {
                "reward/abstained_fraction": float(
                    metrics["verifier_abstained_fraction"]
                ),
            }
            if "verifier_abstained_fraction" in metrics
            else {}
        ),
        **(
            {
                "behavior/abstained_emitted_tokens_mean": float(
                    metrics["abstained_emitted_tokens_mean"]
                ),
                "behavior/abstained_emitted_tokens_max": float(
                    metrics["abstained_emitted_tokens_max"]
                ),
            }
            if "abstained_emitted_tokens_mean" in metrics
            else {}
        ),
        "behavior/actions_per_trajectory": float(
            metrics["actions_per_trajectory"]
        ),
        **(
            {
                "behavior/length_reward_covariance": float(
                    metrics["length_reward_covariance"]
                )
            }
            if "length_reward_covariance" in metrics
            else {}
        ),
        "behavior/thoughts_per_trajectory": float(
            metrics.get("thoughts_per_trajectory", 0.0)
        ),
        "behavior/continued_thoughts": float(
            metrics.get("continued_thoughts", 0.0)
        ),
        "behavior/forced_first_thought_fraction": float(
            metrics.get("forced_first_thought_fraction", 0.0)
        ),
        "behavior/continuation_fraction": float(
            metrics.get("continuation_fraction", 0.0)
        ),
        "behavior/gate_stop_fraction": float(
            metrics.get("gate_stop_fraction", 0.0)
        ),
        **(
            {
                "behavior/thought_sigma_vector": float(
                    metrics["thought_sigma"]
                ),
                "behavior/thought_component_std": float(
                    metrics["thought_component_std"]
                ),
                "behavior/sampled_noise_norm": float(
                    metrics["sampled_noise_norm"]
                ),
            }
            if "thought_sigma" in metrics
            else {}
        ),
        **(
            {
                "reward/think_format_fraction": float(
                    metrics["think_format_fraction"]
                ),
            }
            if "think_format_fraction" in metrics
            else {}
        ),
        # Conditionally present (their absence means "nothing to
        # measure", not zero) — publishing a stand-in 0.0 would fake
        # the very readings these exist to give.
        **(
            {
                "reward/think_gate_zeroed_correct_fraction": float(
                    metrics["think_gate_zeroed_correct_fraction"]
                ),
            }
            if "think_gate_zeroed_correct_fraction" in metrics
            else {}
        ),
        **(
            {
                "behavior/think_tokens_mean": float(
                    metrics["think_tokens_mean"]
                ),
            }
            if "think_tokens_mean" in metrics
            else {}
        ),
    }


def optimizer_minibatch_orders(
    group_count: int,
    groups_per_minibatch: int,
    generator: torch.Generator | None = None,
    allow_partial_final: bool = False,
) -> list[list[int]]:
    """Shuffle and partition one frozen-policy pool without reuse.

    PPO keeps every stored old log-probability fixed to the behavior policy
    that sampled the pool.  As in CleanRL, one fresh permutation is drawn at
    the start of the (single) epoch; later minibatches therefore compare the
    updated policy to that same behavior snapshot rather than refreshing old
    statistics after each optimizer step.
    """
    if group_count < 1:
        raise ValueError("group_count must be positive")
    if groups_per_minibatch < 1:
        raise ValueError("groups_per_minibatch must be positive")
    if group_count % groups_per_minibatch and not allow_partial_final:
        raise ValueError("group_count must divide into complete minibatches")
    order = torch.randperm(group_count, generator=generator).tolist()
    minibatches = [
        order[start : start + groups_per_minibatch]
        for start in range(0, group_count, groups_per_minibatch)
    ]
    if any(not minibatch for minibatch in minibatches):
        raise RuntimeError("optimizer partition produced an empty minibatch")
    return minibatches


def stratified_optimizer_minibatch_orders(
    groups: list[LatentRolloutBatch],
    groups_per_minibatch: int,
    source_count: int,
    generator: torch.Generator | None = None,
    allow_partial_final: bool = False,
) -> list[list[int]]:
    """Partition one mixture pool so every minibatch mirrors its composition.

    Each source's groups are shuffled and laid end to end, then dealt across
    the minibatches by a largest-deficit schedule over their sizes. With
    complete minibatches the deal is round-robin, so a source's count differs
    by at most one between optimizer steps; an unlucky shuffle cannot turn one
    domain into an all-zero minibatch while another supplies all of the
    policy signal. The minibatch order is then shuffled so no step
    systematically receives the remainders.
    """
    group_count = len(groups)
    if group_count < 1:
        raise ValueError("group_count must be positive")
    if groups_per_minibatch < 1:
        raise ValueError("groups_per_minibatch must be positive")
    if group_count % groups_per_minibatch and not allow_partial_final:
        raise ValueError("group_count must divide into complete minibatches")
    indices_by_source: list[list[int]] = [[] for _ in range(source_count)]
    for index, group in enumerate(groups):
        source_id = source_group_id(group)
        if not 0 <= source_id < source_count:
            raise ValueError(f"rollout carries unknown source id {source_id}")
        indices_by_source[source_id].append(index)
    stratified = []
    for indices in indices_by_source:
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        stratified.extend(indices[position] for position in permutation)

    capacities = [
        min(groups_per_minibatch, group_count - start)
        for start in range(0, group_count, groups_per_minibatch)
    ]
    minibatches: list[list[int]] = [[] for _ in capacities]
    for position, index in enumerate(stratified, start=1):
        # Ties go to the lowest minibatch, which makes equal capacities an
        # exact round-robin deal.
        target = max(
            range(len(capacities)),
            key=lambda slot: (
                position * capacities[slot] / group_count
                - len(minibatches[slot]),
                -slot,
            ),
        )
        minibatches[target].append(index)
    if [len(minibatch) for minibatch in minibatches] != capacities:
        raise AssertionError("stratified optimizer minibatch has wrong size")
    for minibatch in minibatches:
        permutation = torch.randperm(
            len(minibatch), generator=generator
        ).tolist()
        minibatch[:] = [minibatch[position] for position in permutation]
    order = torch.randperm(len(minibatches), generator=generator).tolist()
    return [minibatches[position] for position in order]


def plan_one_pass_training(
    *,
    dataset_rows: int,
    sampler_cursor: int,
    warmup_updates: int,
    remaining_actor_steps: int,
    prompts_per_minibatch: int,
    actor_start_cursor: int | None = None,
) -> tuple[int, int]:
    """Validate exact one-pass consumption and return prompts and actor steps.

    Warmup consumes one complete prompt minibatch per critic update. Actor
    training then consumes every remaining row, with at most one terminal
    partial minibatch. A rewinding warmup (``rewinds_warmup_prompts``) reads
    the actor's first prompts instead of preceding them: the actor then
    starts at ``actor_start_cursor``, at or before a mid-warmup resume's
    cursor, and consumes every row from there. This helper is deliberately pure so fresh
    curriculum starts and checkpoint resumes share testable no-reuse
    arithmetic.
    """
    if not 0 <= sampler_cursor <= dataset_rows:
        raise ValueError("sampler cursor must lie within the dataset")
    if warmup_updates < 0 or remaining_actor_steps < 0:
        raise ValueError("update counts must be nonnegative")
    if prompts_per_minibatch < 1:
        raise ValueError("prompts_per_minibatch must be positive")
    available = dataset_rows - sampler_cursor
    warmup_prompts = prompts_per_minibatch * warmup_updates
    if warmup_prompts > available:
        raise ValueError("critic warmup alone exceeds the unused target dataset")
    if actor_start_cursor is None:
        actor_prompts = available - warmup_prompts
    elif not 0 <= actor_start_cursor <= sampler_cursor:
        raise ValueError("a rewound actor cannot start past the warmup cursor")
    else:
        actor_prompts = dataset_rows - actor_start_cursor
    required_actor_steps = math.ceil(actor_prompts / prompts_per_minibatch)
    if remaining_actor_steps != required_actor_steps:
        raise ValueError(
            "--consume-all-prompts requires exactly "
            f"{required_actor_steps} remaining actor steps for "
            f"{actor_prompts} post-warmup prompts; got "
            f"{remaining_actor_steps}"
        )
    return available, required_actor_steps


def actor_minibatch_action_denominator(
    groups: list[LatentRolloutBatch],
    indices: list[int],
) -> torch.Tensor:
    """Global VAPO stream-action denominator for one actor step."""
    if not indices:
        raise ValueError("at least one prompt group is required")
    return torch.stack(
        [groups[index].action_mask.sum() for index in indices]
    ).sum()


def source_actor_signal_mask(
    source_ids: torch.Tensor | None,
    reward_scalar: torch.Tensor,
    *,
    enabled: bool = False,
) -> torch.Tensor:
    """Optionally activate rows only when their source has a positive reward.

    Disabled is the default: all rows retain their critic-GAE actor signal,
    regardless of source reward. The opt-in legacy gate protects against
    baseline error from an entirely zero-reward source, but can silently skew
    a mixed-source objective toward whichever sources already succeed. It is
    deliberately not a paper-prescribed VAPO/DAPO component.

    When enabled, this remains source-level rather than per-prompt filtering:
    failed prompt groups inside a source with any success retain dense
    critic-GAE credit. The pairwise formulation avoids a device-to-host sync
    for source counts.
    """
    if not enabled or source_ids is None:
        return torch.ones_like(reward_scalar, dtype=torch.bool)
    if source_ids.shape != reward_scalar.shape:
        raise ValueError("source ids and scalar rewards must have identical shape")
    same_source = source_ids[:, None] == source_ids[None, :]
    source_max_reward = torch.where(
        same_source,
        reward_scalar[None, :],
        reward_scalar.new_full((), float("-inf")),
    ).amax(1)
    return source_max_reward > 0


def write_actor_tensorboard_metrics(
    tensorboard, dashboard: dict[str, float], behavior_age: int, step: int
) -> None:
    """Write one actor-step row, eliding exact fresh-behavior zeros."""
    fresh_behavior_tags = (
        "kl/token_behavior",
        "kl/policy_behavior_per_action",
        "clip/policy",
        "clip/stochastic_policy",
        "ratio/token_abs_log_max",
        "ratio/harmful_positive_log_max",
    )
    if behavior_age == 0:
        refresh_drift = max(
            (
                dashboard[tag]
                for tag in fresh_behavior_tags
                if tag in dashboard
            ),
            default=0.0,
        )
        tensorboard.add_scalar(
            "debug/behavior_refresh_max_drift", refresh_drift, step
        )
    for tag, value in dashboard.items():
        if behavior_age == 0 and tag in fresh_behavior_tags:
            continue
        tensorboard.add_scalar(tag, value, step)


def gradient_norm_tensor(parameters) -> torch.Tensor:
    parameters = list(parameters)
    grads = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not grads:
        if not parameters:
            return torch.zeros((), dtype=torch.float32)
        return parameters[0].new_zeros((), dtype=torch.float32)
    # The old loop converted every parameter's sum to float separately,
    # forcing hundreds of device synchronizations. Foreach norms enqueue the
    # whole collection and synchronize once for the final scalar.
    norms = torch._foreach_norm(grads, 2.0)
    return torch.linalg.vector_norm(torch.stack(norms)).float()


def gradient_norm(parameters) -> float:
    """Public scalar form used by standalone diagnostics and tests."""
    return float(gradient_norm_tensor(parameters))


def scalar_tensors_to_floats(
    values: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Resolve same-device scalar telemetry with one device synchronization.

    Stacked in float64 so raw moment sums keep their precision for pooling.
    """
    if not values:
        return {}
    keys = tuple(values)
    packed = torch.stack(
        [values[key].detach().reshape(()).double() for key in keys]
    ).cpu().tolist()
    return dict(zip(keys, packed, strict=True))


def renderer_parameters(backbone) -> list[torch.nn.Parameter]:
    """The trainable vocabulary-readout parameters, across backbone families.

    The fresh lineage renders through its pretrained ``policy_probe``; the
    nano backbones expose their native readout (untied projection, or the
    tied-dot scale/bias) via their own ``renderer_parameters`` method.
    """
    if hasattr(backbone, "policy_probe"):
        return list(backbone.policy_probe.parameters())
    return list(backbone.renderer_parameters())


def non_trunk_parameter_ids(backbone) -> set[int]:
    """Backbone parameters excluded from the trunk optimizer group and norm.

    The renderer readout is always its own semantic group. The fresh critic
    probe additionally exists but stays frozen and unused; nano backbones
    carry no probes at all.
    """
    excluded = {id(parameter) for parameter in renderer_parameters(backbone)}
    critic_probe = getattr(backbone, "critic_probe", None)
    if critic_probe is not None:
        excluded |= {id(parameter) for parameter in critic_probe.parameters()}
    return excluded


def validate_combiner_geometry(
    payload: dict, args, memory_carry: bool, source: object
) -> None:
    """Reject a checkpoint whose combiner shapes differ from the CLI's.

    The shapes are fixed by the saved CLI arguments; a mismatch would
    otherwise surface as an opaque strict-load shape error (or, for the
    loaded gain scalar, not at all). A memory-carry combiner has memory
    geometry and no MLP; every other combiner has an MLP and no memory.
    """
    saved = combiner_init_kwargs_from_checkpoint(payload)
    if saved.get("memory_carry", False) != memory_carry:
        raise ValueError(
            f"{source} memory_carry={saved.get('memory_carry', False)} "
            f"does not match --reasoning-mode {args.reasoning_mode}"
        )
    if memory_carry:
        saved_memory = (saved["memory_heads"], saved["memory_head_dim"])
        if saved_memory != (args.memory_heads, args.memory_head_dim):
            raise ValueError(
                f"{source} was trained with memory geometry "
                f"heads/head_dim={saved_memory}, not "
                f"{(args.memory_heads, args.memory_head_dim)}"
            )
        return
    if (saved["mlp_hidden"], saved["num_blocks"]) != (
        args.combined_mlp_hidden,
        args.combined_mlp_blocks,
    ):
        raise ValueError(
            f"{source} was trained with combiner geometry "
            f"mlp_hidden={saved['mlp_hidden']}/blocks={saved['num_blocks']}, "
            f"not {args.combined_mlp_hidden}/{args.combined_mlp_blocks}; pass "
            "the checkpoint's --combined-mlp-hidden and --combined-mlp-blocks"
        )


def make_actor_trainable(wrapper: LatentThoughtModel) -> int:
    """Set the actor's trainable-parameter contract; return frozen router count.

    Every wrapper parameter trains except the MoE routing (frozen by the
    backbone's own rule) and the fresh lineage's critic probe, which stays
    checkpointed but has no graph edge. Nano backbones carry no probes.
    """
    backbone = wrapper.backbone
    for parameter in wrapper.parameters():
        parameter.requires_grad_(True)
    frozen_moe_router_parameters = (
        backbone.freeze_moe_routing_()
        if hasattr(backbone, "freeze_moe_routing_")
        else 0
    )
    if hasattr(backbone, "critic_probe"):
        for parameter in backbone.critic_probe.parameters():
            parameter.requires_grad_(False)
    return frozen_moe_router_parameters


def muon_matrix_parameters(
    blocks: torch.nn.Module, excluded_parameter_ids: set[int]
) -> list[torch.nn.Parameter]:
    """The pretraining Muon partition: block matrices with ndim >= 2.

    Mirrors ``pretraining/nanogpt_mini/nanogpt_mini_gpt2vocab_train.py`` exactly — embeddings, the
    readout, biases, and norm gains stay under AdamW. Fresh-lineage probes
    live under ``blocks[-1]``, so the exclusion set must be honored here too.

    KDA blocks add the one family of ndim>=2 tensors pretraining kept OUT of
    Muon: the ``[D, 1, W]`` depthwise conv windows (their own AdamW group,
    ``KDA_CONV_LR``). Orthogonalizing per-channel windows is not a Muon
    update, so they stay under AdamW here too — 1-D ``A_log``/``dt_bias``
    already fall through on ndim.
    """
    conv_parameter_ids = {
        id(conv.weight)
        for block in blocks
        if getattr(block, "use_kda", False)
        for conv in (
            block.attn.q_conv1d,
            block.attn.k_conv1d,
            block.attn.v_conv1d,
        )
    }
    return [
        parameter
        for parameter in blocks.parameters()
        if parameter.ndim >= 2
        and parameter.requires_grad
        and id(parameter) not in excluded_parameter_ids
        and id(parameter) not in conv_parameter_ids
    ]


def step_optimizers(
    optimizers: dict[str, torch.optim.Optimizer], role: str
) -> None:
    """Step every optimizer belonging to ``role`` ("actor" or "critic").

    The Muon split registers trunk matrices under ``<role>_muon``; iterating
    by role prefix keeps every step/zero seam covering both optimizers.
    """
    for name in sorted(optimizers):
        if name == role or name.startswith(role + "_"):
            optimizers[name].step()


def zero_optimizers(
    optimizers: dict[str, torch.optim.Optimizer], role: str
) -> None:
    """zero_grad(set_to_none=True) for every optimizer of ``role``."""
    for name in sorted(optimizers):
        if name == role or name.startswith(role + "_"):
            optimizers[name].zero_grad(set_to_none=True)


def optimizer_learning_rates(args) -> dict[str, float]:
    """The per-optimizer learning rates the CLI owns across resumes."""
    return {
        "actor": args.learning_rate,
        "critic": args.critic_learning_rate,
        "actor_muon": args.muon_learning_rate,
        "critic_muon": args.critic_muon_learning_rate,
        "critic_action": getattr(args, "critic_action_head_learning_rate", None),
        "actor_fresh": getattr(args, "fresh_learning_rate", None),
        "actor_fresh_muon": getattr(args, "fresh_muon_learning_rate", None),
        "critic_fresh": getattr(args, "fresh_learning_rate", None),
        "critic_fresh_muon": getattr(args, "fresh_muon_learning_rate", None),
    }


def reassert_learning_rates(
    optimizers: dict[str, torch.optim.Optimizer], args
) -> None:
    """Reapply the CLI learning rates after loading optimizer state."""
    rates = optimizer_learning_rates(args)
    for name, optimizer in optimizers.items():
        for group in optimizer.param_groups:
            group["lr"] = (
                args.critic_q_direction_learning_rate
                if group.get("token_q_direction", False)
                else rates[name]
            )


def build_optimizers(
    wrapper: LatentThoughtModel,
    critic: SeparateCritic,
    learning_rate: float,
    critic_learning_rate: float | None = None,
    trunk_optimizer: str = "adamw",
    muon_learning_rate: float | None = None,
    critic_muon_learning_rate: float | None = None,
    fused: bool = True,
    critic_action_learning_rate: float | None = None,
    critic_q_direction_learning_rate: float | None = None,
    fresh_learning_rate: float | None = None,
    fresh_muon_learning_rate: float | None = None,
) -> dict[str, torch.optim.Optimizer]:
    """The actor/critic optimizer layout.

    One actor AdamW holds semantic groups for trunk, all fresh latent-policy
    modules, and renderer. Every trainable policy component uses the same actor
    learning rate; strict execution schemas guard the changed group layout.
    The critic may use its own constant rate.

    ``trunk_optimizer="muon"`` restores the pretraining update geometry:
    block matrices (ndim >= 2) move from the AdamW trunk group into separate
    ``actor_muon``/``critic_muon`` Muon optimizers. Polar Express supplies
    the orthogonalized momentum update, while embeddings, readout, gains,
    and the combiners stay under AdamW (the combiner is not under
    ``backbone.blocks``, so Muon routing never sees it). Weight decay stays
    0 everywhere: pretraining's Muon decay (0.05) regularizes a from-scratch
    run, but over a long RL schedule it would only shrink the pretrained
    weights.

    A critic action head has its own ``critic_action`` AdamW, so a carried
    critic's optimizer state loads unchanged beside a fresh head and an
    action-head warmup can step the head alone. A standalone Q critic's
    token head takes the same optimizer and rate: a relu^2 layer and
    zero-initialized readouts like the action head, whose measured fit
    wanted the head rate, not the trunk's. ``critic_q_direction_learning_rate``
    moves that head's direction readout u(s) into its own group: u is dotted
    with every action latent, so its effective step scales with their width.

    ``fresh_learning_rate`` gives the from-scratch modules -- the actor's
    ``new_parameters()`` and the critic's ``fresh_parameters()`` -- their own
    ``<role>_fresh`` AdamW (ndim < 2) and ``<role>_fresh_muon`` Muon
    (ndim >= 2), pretraining's partition, instead of the fine-tuning rate
    of the role's AdamW. The optimizer names are the resume layout, so a
    run cannot resume across the change.
    """
    if trunk_optimizer not in ("adamw", "muon"):
        raise ValueError(f"unknown trunk optimizer {trunk_optimizer!r}")
    if critic_learning_rate is None:
        critic_learning_rate = learning_rate
    backbone = wrapper.backbone
    excluded_parameter_ids = non_trunk_parameter_ids(backbone)
    actor_muon_parameters: list[torch.nn.Parameter] = []
    critic_muon_parameters: list[torch.nn.Parameter] = []
    if trunk_optimizer == "muon":
        actor_muon_parameters = muon_matrix_parameters(
            backbone.blocks, excluded_parameter_ids
        )
        critic_muon_parameters = muon_matrix_parameters(critic.trunk.blocks, set())
    actor_muon_ids = {id(parameter) for parameter in actor_muon_parameters}
    critic_muon_ids = {id(parameter) for parameter in critic_muon_parameters}
    if (fresh_learning_rate is None) != (fresh_muon_learning_rate is None):
        raise ValueError("fresh modules need both their AdamW and Muon rates")
    fresh = {} if fresh_learning_rate is None else {
        "actor": [p for p in wrapper.new_parameters() if p.requires_grad],
        "critic": [p for p in critic.fresh_parameters() if p.requires_grad],
    }
    critic_fresh_ids = {id(parameter) for parameter in fresh.get("critic", ())}
    critic_action_parameters = [
        parameter
        for head in (critic.action_head, critic.q_head,
                     getattr(critic, "q_twin", None),
                     getattr(critic, "post_action_q_head", None))
        if head is not None
        for parameter in head.parameters()
        if id(parameter) not in critic_fresh_ids
    ]
    critic_action_ids = {id(parameter) for parameter in critic_action_parameters}
    trunk_parameters = [
        parameter
        for parameter in backbone.parameters()
        if parameter.requires_grad
        if id(parameter) not in excluded_parameter_ids
        and id(parameter) not in actor_muon_ids
    ]
    critic_adamw_parameters = [
        parameter
        for parameter in critic.parameters()
        if parameter.requires_grad
        if id(parameter) not in critic_muon_ids
        and id(parameter) not in critic_action_ids
        and id(parameter) not in critic_fresh_ids
    ]
    optimizers = {
        "actor": torch.optim.AdamW(
            [
                {"params": trunk_parameters, "lr": learning_rate},
                *(
                    [{"params": list(wrapper.new_parameters()), "lr": learning_rate}]
                    if fresh_learning_rate is None
                    else []
                ),
                {"params": renderer_parameters(backbone), "lr": learning_rate},
            ],
            weight_decay=0.0,
            fused=fused,
        ),
        "critic": torch.optim.AdamW(
            critic_adamw_parameters,
            lr=critic_learning_rate,
            weight_decay=0.0,
            fused=fused,
        ),
    }
    if critic_action_parameters:
        head_rate = (
            critic_learning_rate
            if critic_action_learning_rate is None
            else critic_action_learning_rate
        )
        action_groups = [{"params": critic_action_parameters, "lr": head_rate}]
        if critic_q_direction_learning_rate is not None:
            if critic.q_head is None:
                raise ValueError("a direction learning rate needs a token Q critic")
            directions = [weight for head in critic.q_heads for weight in head.direction_weights]
            direction_ids = {id(direction) for direction in directions}
            action_groups = [
                {"params": [p for p in critic_action_parameters if id(p) not in direction_ids],
                 "lr": head_rate},
                {"params": directions, "lr": critic_q_direction_learning_rate,
                 "token_q_direction": True},
            ]
        optimizers["critic_action"] = torch.optim.AdamW(
            action_groups, lr=head_rate, weight_decay=0.0, fused=fused,
        )
    if trunk_optimizer == "muon":
        if muon_learning_rate is None or critic_muon_learning_rate is None:
            raise ValueError("the muon trunk optimizer requires its learning rates")
        optimizers["actor_muon"] = Muon(
            actor_muon_parameters, lr=muon_learning_rate, weight_decay=0.0
        )
        optimizers["critic_muon"] = Muon(
            critic_muon_parameters, lr=critic_muon_learning_rate, weight_decay=0.0
        )
    for role, parameters in fresh.items():
        vectors = [parameter for parameter in parameters if parameter.ndim < 2]
        matrices = [parameter for parameter in parameters if parameter.ndim >= 2]
        if vectors:
            optimizers[f"{role}_fresh"] = torch.optim.AdamW(
                vectors, lr=fresh_learning_rate, weight_decay=0.0, fused=fused
            )
        if matrices:
            optimizers[f"{role}_fresh_muon"] = Muon(
                matrices, lr=fresh_muon_learning_rate, weight_decay=0.0
            )
    # Exact-partition guard (pretraining's optimizer assertion): every critic
    # parameter in exactly one optimizer, and the Muon split disjoint from
    # the actor's semantic groups.
    actor_registered = [
        parameter
        for name, optimizer in optimizers.items()
        if name.startswith("actor")
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(actor_registered) != len({id(p) for p in actor_registered}):
        raise AssertionError("actor parameter registered in two optimizers")
    actor_registered_ids = {id(parameter) for parameter in actor_registered}
    actor_expected_ids = {
        id(parameter)
        for parameter in wrapper.parameters()
        if parameter.requires_grad
    }
    if actor_registered_ids != actor_expected_ids:
        raise AssertionError(
            "every trainable actor parameter must belong to exactly one "
            "optimizer"
        )
    trunk_coverage = {id(p) for p in trunk_parameters} | actor_muon_ids
    trunk_expected = {
        id(parameter)
        for parameter in backbone.parameters()
        if parameter.requires_grad
        if id(parameter) not in excluded_parameter_ids
    }
    if trunk_coverage != trunk_expected:
        raise AssertionError(
            "the Muon split must exactly cover the AdamW trunk group it "
            "replaced"
        )
    critic_registered = [
        parameter
        for name, optimizer in optimizers.items()
        if name.startswith("critic")
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if {id(p) for p in critic_registered} != {
        id(p) for p in critic.parameters() if p.requires_grad
    } or len(critic_registered) != len({id(p) for p in critic_registered}):
        raise AssertionError("critic parameters must partition across optimizers")
    # Actor and critic share nothing: not a Parameter object and not the
    # memory behind one, so a tied view cannot leak value gradients into the
    # policy (or the reverse) through either optimizer.
    # Zero-size parameters own no memory (data_ptr 0) and cannot alias.
    def storages(module: torch.nn.Module) -> set[tuple[torch.device, int]]:
        return {
            (p.device, p.untyped_storage().data_ptr())
            for p in module.parameters()
            if p.numel()
        }

    actor_storages = storages(wrapper)
    critic_storages = storages(critic)
    if actor_storages & critic_storages or {
        id(p) for p in wrapper.parameters()
    } & {id(p) for p in critic.parameters()}:
        raise AssertionError("actor and critic must not share parameters")
    return optimizers


def post_action_q_loss_and_predictions(
    critic: SeparateCritic,
    batch: LatentRolloutBatch,
    mc_targets: torch.Tensor,
    contract: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Proper binary fit and exact-prefix actor credit, with separate units.

    Critic diagnostics use the raw sigmoid prediction. Only the actor's
    detached success probability receives deterministic prefix overrides.
    No preterminal realized continuation outcome enters that probability.
    """
    success_logits = critic.action_success_logits(batch).float()
    raw_probabilities = success_logits.sigmoid()
    known, exact_probabilities = exact_prefix_success_constraints(batch, **contract)
    actor_probabilities = torch.where(known, exact_probabilities, raw_probabilities).detach()
    errors = F.binary_cross_entropy_with_logits(
        success_logits, mc_targets, reduction="none"
    )
    return raw_probabilities, actor_probabilities, errors


def update_minibatch(
    wrapper: LatentThoughtModel,
    critic: SeparateCritic,
    batch: LatentRolloutBatch,
    optimizers: dict[str, torch.optim.Optimizer],
    value_only: bool = False,
    actor_step: bool = True,
    critic_step: bool = True,
    policy_action_denominator: torch.Tensor | None = None,
    value_action_denominator: torch.Tensor | None = None,
    gae_lambda_alpha: float = 0.05,
    actor_gae_lambda: float | None = None,
    replay_max_trajectories: int = 32,
    replay_attention_budget: int = 4 * 1024 * 1024,
    replay_bucket: int = 1,
    replay_slot_budget: int | None = None,
    replay_plan: ReplayPlan | None = None,
    delightful_policy_gradient: bool = False,
    target_policy_optimization: bool = False,
    tpo_eta: float = 2.0,
    source_success_actor_gate: bool = False,
    fused_behavior_statistics: bool = False,
    reward_floor: float = 0.0,
    value_scope: str = "critic",
    critic_lambda: float | None = None,
    post_action_q_contract: dict | None = None,
    entropy_temperature: float | None = None,
    advantage_source_count: int = 0,
    advantage_token_classes: (
        dict[str, tuple[torch.Tensor, torch.Tensor | None]] | None
    ) = None,
    mean_actor_critic: bool = False,
    mac_estimator: str | None = None,
    mac_vocab_chunk: int = 512,
    mac_gate_shuffle_seed: int | None = None,
    think_close_id: int | None = None,
    critic_q_pessimism: float = 0.0,
    policy_batch: LatentRolloutBatch | None = None,
) -> dict[str, float]:
    """One minibatch update.

    Actor and critic gradients can be accumulated across prompt groups with
    ``actor_step=False`` and ``critic_step=False``. The trainer uses the same
    effective trajectory minibatch and one optimizer step for both; standalone
    callers keep the self-contained single-step defaults.

    The group is replayed in stable length-sorted shards because causal
    attention is quadratic in stream length. The planner bounds B*L^2, B*L
    (the vocabulary head's linear memory term), and trims every shard
    independently. The actor objectives keep denominators from the COMPLETE
    optimizer minibatch, which can span prompt groups, so sharding changes
    only floating-point reduction order.

    ``fused_behavior_statistics`` derives the behavior statistics from this
    update's own forwards instead of a prior ``refresh_old_statistics``
    pass. It is valid only at behavior age 0 -- the pool's sole optimizer
    minibatch, before any step -- where the refresh would replay the same
    parameters through the same compiled artifacts and return exactly these
    values. Each shard's critic forward runs first and fills the GAE
    baseline; the actor forward then takes its own detached log-probs as
    the old policy, so every ratio is one and the gradient is the refreshed
    update's. The lambda-one value targets come from rewards alone, which
    they equal in exact arithmetic without the telescoped baseline terms.
    Latent thoughts and TPO log-odds have no fused form.

    A critic with an action head also fits Q(s_t, a_t) to the same
    lambda-one targets, and the actor's advantages bootstrap from it
    (``action_value_advantages``). ``value_scope="action_head"`` is a
    value-only update of that head alone, for warming it against a critic
    whose values are already fit; it steps only the ``critic_action``
    optimizer.

    A standalone Q critic (``critic.values_actions``) is REPPO's: its
    values are Q(s_t, a_t), regressed on generalized value estimates
    (``generalized_value_estimate_targets``) with lambda
    ``critic_lambda`` -- length-adaptive when None, one in a value-only
    update -- bootstrapped from this update's own Q. It values every token
    at once, so the actor's advantage is Q centered exactly under the
    policy, A_t = Q(s_t, a_t) - sum_a pi(a | s_t) Q(s_t, a), formed in the
    actor's own readout from the critic's direction u_t at the same slot
    (``compact_emit_token_logprobs_entropy_and_advantages``). Both come
    from the update's critic pass, so it needs fused behavior statistics:
    the targets REPPO forms at collection, from the same parameters. Each
    shard's critic pass runs just before its actor pass, so u lives for
    one shard; A_t needs nothing from other rows.

    ``entropy_temperature`` adds REPPO's entropy term to the Delightful
    objective, -temperature times the policy entropy summed over the
    actor's active emit slots and divided by ``policy_action_denominator``,
    the policy loss's own normalizer, so the temperature trades against the
    per-token Delightful loss. It uses the exact vocabulary entropy rather
    than a sampled log-probability, and Q stays the plain task value
    (``postraining.vapo.reppo``).

    ``advantage_source_count`` and ``advantage_token_classes`` break the
    actor's advantages down over emit slots: by the row's source id, and by
    the class of the token taken -- name to (token ids, and the ids the
    previous token must be, or None for any) -- with every slot no class
    claims in ``other``. Classes should not overlap. They report as
    ``advantage_group/<name>/{sum,square_sum,count}``, sums that pool
    across groups (``advantage_group_dashboard``).

    ``mean_actor_critic`` replaces the actor estimator with a token Q
    estimator (``postraining.mean_actor_critic``) at every emit slot, each
    weighted by one over ``policy_action_denominator``, read from this
    pass's critic before its step. ``mac_estimator`` selects it: all-action
    MAC, the full-vocabulary expectation of the critic's detached
    policy-centered Q; MAC plus the sampled token's Monte Carlo residual
    G - Q(s_t, a_t) as a score-function control variate; or that sampled
    term alone with G - E_pi Q. Its readout chunks of ``mac_vocab_chunk``
    states also return the behavior log-probs, entropy and sampled-token
    advantages. ``think_close_id`` splits the advantage report into think
    and post-think spans. The ``SEQUENCE_ESTIMATORS`` replace G - c with one
    coefficient per trajectory from the group terms its scorer stamped
    (``stamp_group_terms``): RLOO's U, or U under the answer gate. They keep
    the same token weights 1 / ``policy_action_denominator`` as the sampled
    baseline, so the two differ only in the coefficient; the critic still
    trains on the same pass, for telemetry.

    ``mac_gate_shuffle_seed`` seeds, and is required by, the measurement
    control ``delightful_shuffled_head_gate``: each shard first reads every
    emit state's head advantage A(a_t) without a loss, permutes it within
    surprisal-and-span strata (``mean_actor_critic.shuffle_within_strata``)
    from one generator seeded per call, and gates on the result. Shards run
    in plan order, so the permutation is a function of the seed, the batch
    and the plan; give each call its own seed.

    ``critic_q_pessimism`` beta > 0 adds a pessimistic prior to every token
    Q head's loss (``postraining.pessimistic_q``): beta times the sum over
    the vocabulary of relu(A_h)^2 at each generated state, A_h the head's Q
    centered under the current policy, normalized as the squared error is.
    MAC reads Q at tokens the critic never regressed; the prior keeps those
    from reading better than the policy's value. It needs the policy MAC
    reads, so it runs in MAC actor updates only: each shard's critic
    backward waits for the actor pass, whose readout chunks also form the
    prior's gradients.

    A token Q critic's Q is centred under the policy that sampled the
    batch, Q(s, a_t) = c + u . (x(a_t) - sum_a pi(a) x(a))
    (``value_model.policy_centered_q``), so each shard's critic loss waits
    for the taken tokens' offsets from the policy's mean action rows: from
    the actor pass's own readout in an actor update, from a replay of the
    policy in a value-only one.
    ``policy_batch``, value-only, is the policy's view of the same
    trajectories when the critic's context differs, row for row and action
    for action; the default is ``batch``.
    """
    backbone = wrapper.backbone
    action_head = critic.action_head is not None
    q_critic = critic.values_actions
    post_action_q = getattr(critic, "post_action_q_head", None) is not None
    bilinear_q = q_critic and not post_action_q
    twin_q = getattr(critic, "q_twin", None) is not None
    if critic_q_pessimism < 0 or not math.isfinite(critic_q_pessimism):
        raise ValueError("the Q prior weight must be finite and nonnegative")
    if critic_q_pessimism and (value_only or not mean_actor_critic):
        raise ValueError("the Q prior reads the policy from a MAC actor update")
    if critic_q_pessimism and value_scope != "critic":
        raise ValueError("the Q prior trains the critic's token Q heads")
    if policy_batch is not None and not (value_only and bilinear_q):
        raise ValueError("a separate policy view centres a token Q critic's value-only fit")
    if policy_batch is not None and policy_batch.kind.size(0) != batch.kind.size(0):
        raise ValueError("the policy view must hold the critic batch's trajectories")
    if mean_actor_critic and not value_only:
        if (not bilinear_q or (wrapper.hidden_carry and not wrapper.memory_carry)
                or think_close_id is None):
            raise ValueError(
                "MAC needs a token Q critic, a cot or memory policy and the think fence")
        if (delightful_policy_gradient or target_policy_optimization
                or source_success_actor_gate or entropy_temperature is not None):
            raise ValueError("MAC is a standalone actor objective without outcome masks")
        if mac_vocab_chunk < 1:
            raise ValueError("MAC needs a positive vocabulary chunk")
        validate_mac_estimator(mac_estimator, controls=True)
        if mac_estimator in SEQUENCE_ESTIMATORS and (
                batch.group_advantage is None or batch.answer_surprisal is None):
            raise ValueError(
                f"{mac_estimator} needs every row's group terms, stamped when its "
                "prompt group is scored (stamp_group_terms)")
    if (mac_gate_shuffle_seed is not None) != (
            mean_actor_critic and not value_only
            and mac_estimator == "delightful_shuffled_head_gate"):
        raise ValueError("a gate shuffle seed comes with, and only with, the shuffled head gate")
    if post_action_q:
        if actor_gae_lambda not in (None, 0.0) or critic_lambda not in (None, 1.0):
            raise ValueError("post-action binary Q requires actor lambda 0 and critic lambda 1")
        if post_action_q_contract is None:
            raise ValueError("post-action binary Q requires its exact prefix reward contract")
        if source_success_actor_gate:
            raise ValueError("post-action Q forbids sampled-future source masking")
        if entropy_temperature is not None or target_policy_optimization or reward_floor != 0:
            raise ValueError("post-action binary Q forbids non-binary rewards or extra actor objectives")
    if q_critic and not value_only and not fused_behavior_statistics:
        raise ValueError(
            "a standalone Q critic's targets and advantages come from the "
            "update's own critic pass; it needs fused behavior statistics"
        )
    if q_critic and not value_only and not (delightful_policy_gradient or mean_actor_critic):
        raise ValueError("a standalone Q critic drives Delightful PG only")
    if entropy_temperature is not None and not delightful_policy_gradient:
        raise ValueError("the entropy temperature is a Delightful PG term")
    if value_scope not in ("critic", "action_head"):
        raise ValueError(f"unknown value scope {value_scope!r}")
    if value_scope == "action_head" and not (value_only and action_head):
        raise ValueError(
            "an action-head-only update is value-only on a critic with one"
        )
    if action_head and target_policy_optimization:
        raise ValueError("TPO does not consume action-value advantages")
    critic_role = "critic_action" if value_scope == "action_head" else "critic"
    if critic_step:
        zero_optimizers(optimizers, critic_role)
    if actor_step and "actor" in optimizers:
        zero_optimizers(optimizers, "actor")
    if replay_max_trajectories < 1:
        raise ValueError("replay max trajectories must be positive")
    if replay_attention_budget < 1:
        raise ValueError("replay attention budget must be positive")
    if delightful_policy_gradient and target_policy_optimization:
        raise ValueError("actor objectives are mutually exclusive")
    hidden_carry = wrapper.hidden_carry
    # A carry policy must be optimized on its own carried streams, and a
    # stochastic or token-only policy never on them.
    validate_carry_policy(batch, wrapper)
    if fused_behavior_statistics and (
        value_only or target_policy_optimization
    ):
        raise ValueError(
            "fused behavior statistics need an actor update without TPO"
        )
    if (
        not value_only
        and not fused_behavior_statistics
        and not batch.statistics_refreshed
    ):
        # An unrefreshed batch carries zeros in ``old_token_logprobs`` and
        # ``old_values``; nothing about the tensor shapes can express "not
        # yet refreshed", so the explicit flag is the only guard.
        raise RuntimeError(
            "actor update requires refresh_old_statistics after rollout"
        )
    if target_policy_optimization and not batch.tpo_statistics_refreshed:
        raise RuntimeError(
            "TPO update requires a log-odds refresh of old statistics"
        )

    denominators = {
        "action": batch.action_mask.sum().clamp_min(1),
    }
    if value_action_denominator is None:
        value_action_denominator = batch.action_mask.sum()
    if policy_action_denominator is None:
        policy_action_denominator = (
            batch.emit_mask.sum()
            if delightful_policy_gradient or target_policy_optimization or mean_actor_critic
            else batch.action_mask.sum()
        )
    zero = batch.action_mask.new_zeros(())
    totals = {
        key: zero.clone()
        for key in (
            "value_loss", "value_sum", "value_target_sum", "target_square_sum",
            "advantage_sum", "advantage_square_sum",
            "policy_loss", "policy_clip", "token_kl_sum",
            "stochastic_policy_loss", "stochastic_policy_clip",
            "residual_sum", "residual_square_sum",
            "token_abs_log_ratio_max",
            "harmful_positive_log_ratio_max",
            "delightful_gate_sum", "delightful_positive_gate_sum",
            "delightful_positive_count", "delightful_negative_gate_sum",
            "delightful_negative_count", "delightful_delight_sum",
            "delightful_surprisal_sum", "delightful_positive_loss_sum",
            "delightful_negative_loss_sum", "binary_q_coefficient_sum",
            "tpo_active_count", "tpo_loss_sum", "tpo_fit_kl_sum",
            "tpo_target_behavior_kl_sum", "tpo_old_probability_sum",
            "tpo_current_probability_sum", "tpo_target_probability_sum",
            "tpo_target_move_abs_sum",
            "tpo_target_log_odds_shift_abs_sum",
            "tpo_target_log_odds_shift_square_sum",
            "tpo_probability_residual_sum",
            "tpo_probability_residual_abs_sum",
            "tpo_probability_residual_square_sum", "tpo_positive_count",
            "tpo_negative_count", "tpo_neutral_count",
            "tpo_positive_target_probability_sum",
            "tpo_negative_target_probability_sum",
            "carry_input_delta_square_sum", "carry_token_square_sum",
            "memory_write_beta_sum", "memory_decay_log_timescale_sum",
            "memory_write_count",
            "action_value_loss", "action_residual_sum",
            "action_residual_square_sum",
            "mc_target_sum", "mc_target_square_sum", "mc_residual_sum",
            "mc_residual_square_sum", "entropy_sum", "entropy_count",
            "entropy_loss", "mac_q_variance_think_sum",
            "mac_q_variance_answer_sum", "mac_coefficient_think_sum",
            "mac_coefficient_answer_sum", "mac_coefficient_square_think_sum",
            "mac_coefficient_square_answer_sum", "mac_gate_think_sum",
            "mac_gate_answer_sum", "mac_gate_positive_sum", "mac_gate_positive_count",
            "mac_gate_negative_sum", "mac_gate_negative_count",
            "mac_gate_positive_mass_sum", "mac_gate_positive_weighted_sum",
            "mac_gate_positive_active_mass_sum", "mac_gate_negative_mass_sum",
            "mac_gate_negative_weighted_sum", "mac_gate_negative_active_mass_sum",
            "mac_gate_advantage_states_sum", "mac_gate_advantage_square_sum",
            "mac_gate_advantage_residual_sum", "mac_residual_square_sum",
            "mac_head_advantage_states_sum", "mac_head_advantage_square_sum",
            "mac_head_advantage_residual_sum", "mac_shuffled_advantage_states_sum",
            "mac_shuffled_advantage_square_sum", "mac_shuffled_advantage_residual_sum",
            "mac_shuffle_head_cross_sum", "mac_shuffle_fixed_mass_sum", "mac_shuffle_mass_sum",
            "mac_gate_nonterminal_mass_sum", "mac_gate_nonterminal_weighted_sum",
            "mac_gate_nonterminal_active_mass_sum",
            "mac_sb_coefficient_square_think_sum", "mac_sb_coefficient_square_answer_sum",
            "twin_gap_square_sum",
            "twin_min_shift_sum", "q_prior_penalty_sum", "q_prior_positive_sum",
            "q_prior_count",
        )
    }
    totals["mac_support_census"] = torch.zeros(
        len(SUPPORT_CENSUS_FIELDS), dtype=torch.float64, device=zero.device)
    shuffle_generator = None
    if mac_gate_shuffle_seed is not None:
        # One stream for the whole call, drawn shard by shard in plan order.
        shuffle_generator = torch.Generator(device=zero.device).manual_seed(
            mac_gate_shuffle_seed)
    # A sequence estimator's coefficient, one per trajectory, from the group
    # terms alone: no critic value enters it.
    sequence_rows = (
        sequence_coefficients(mac_estimator, batch.group_advantage.float(),
                              batch.answer_surprisal.float())
        if mean_actor_critic and not value_only and mac_estimator in SEQUENCE_ESTIMATORS
        else None
    )

    action_counts = batch.action_mask.sum(1)
    lambdas = (
        torch.ones_like(action_counts)
        if value_only
        else actor_gae_lambdas(
            action_counts,
            gae_lambda_alpha,
            (1.0 if post_action_q else critic_lambda) if q_critic else actor_gae_lambda,
        )
    )
    actor_signal_rows = source_actor_signal_mask(
        batch.source_id,
        batch.reward_scalar,
        enabled=source_success_actor_gate,
    )

    def actor_advantages() -> torch.Tensor:
        if action_head:
            advantages = action_value_advantages(
                batch.rewards,
                batch.old_values,
                batch.old_action_values,
                batch.action_mask,
                lambdas,
            )
        else:
            advantages, _ = generalized_advantage_and_return_targets(
                batch.rewards,
                batch.old_values,
                batch.action_mask,
                lambdas,
            )
        return advantages.detach() * actor_signal_rows[:, None]

    mc_targets = None
    if q_critic:
        # The Monte-Carlo return, against which Q's fit is also reported, so
        # its explained variance compares with a V critic's. Each shard
        # regresses on its own GVE targets, formed once its Q exists; at
        # lambda one they are this return exactly.
        _, mc_targets = generalized_advantage_and_return_targets(
            batch.rewards,
            torch.zeros_like(batch.old_values),
            batch.action_mask,
            torch.ones_like(lambdas),
        )
        mc_targets = mc_targets.detach()
        if post_action_q:
            torch._assert_async(
                (~batch.action_mask.bool() | ((mc_targets == 0) | (mc_targets == 1))).all(),
                "post-action Q requires exact binary Monte Carlo labels",
            )
        value_targets = mc_targets.clone()
        # Filled shard by shard from the actor's readout.
        advantages = torch.zeros_like(batch.old_values)
        if bilinear_q:
            with torch.no_grad():
                # One action table serves every shard before the critic step.
                action_latents = critic.action_latents()
    elif fused_behavior_statistics:
        # The baseline does not exist until the critic pass below; the
        # lambda-one targets never needed it.
        _, value_targets = generalized_advantage_and_return_targets(
            batch.rewards,
            torch.zeros_like(batch.old_values),
            batch.action_mask,
            lambdas,
        )
        advantages = None
    else:
        advantages, value_targets = generalized_advantage_and_return_targets(
            batch.rewards,
            batch.old_values,
            batch.action_mask,
            lambdas,
        )
        if value_only:
            advantages = advantages.detach()
        elif action_head:
            advantages = actor_advantages()
        else:
            advantages = advantages.detach() * actor_signal_rows[:, None]
    value_targets = value_targets.detach()
    actor_active_mask = (
        batch.action_mask * actor_signal_rows[:, None]
        if not value_only
        else batch.action_mask
    )
    if replay_plan is None:
        replay_plan = build_replay_plan(
            batch,
            replay_max_trajectories,
            replay_attention_budget,
            replay_bucket,
            replay_slot_budget,
            include_action_indices=not value_only,
        ).to(batch.kind.device)
    replay_plan.validate_settings(
        replay_max_trajectories,
        replay_attention_budget,
        replay_bucket,
        replay_slot_budget,
        require_action_indices=not value_only,
    )

    # Non-finite losses must abort the run, but ``torch.isfinite`` on a live
    # loss is a host sync at the worst possible moment: maximum queue depth,
    # immediately before ``backward``, so the GPU drains and then idles
    # through the entire backward launch -- twice per shard, roughly a dozen
    # shards per minibatch. The tests are accumulated on device and resolved
    # in one sync after the shard loop instead. Every ``step_optimizers`` call
    # in this function comes later, so a poisoned shard still aborts before it
    # can reach the weights; only the wasted backward passes are new.
    finite_guards: list[
        tuple[int, str, torch.Tensor, dict[str, torch.Tensor]]
    ] = []

    if value_only and bilinear_q and policy_batch is not None:
        # One host read of the policy view's lengths serves every shard.
        policy_lengths = trajectory_used_lengths(policy_batch).cpu().tolist()
        # The views hold the same trajectories action for action: equal
        # per-row counts keep each policy row on its critic slot.
        torch._assert_async(
            (policy_batch.emit_mask.sum(1) == batch.action_mask.sum(1)).all(),
            "the policy and critic views disagree on a trajectory's action count")
        torch._assert_async(
            (compact_next_slots(policy_batch.token_ids, slot_index(policy_batch.emit_mask.bool()))
             == compact_next_slots(batch.token_ids, slot_index(batch.action_mask.bool()))).all(),
            "the policy and critic views took different tokens")

    def policy_view(shard, microbatch: LatentRolloutBatch) -> LatentRolloutBatch:
        """The policy's own view of a value-only shard's trajectories."""
        if policy_batch is None:
            return microbatch
        return select_trajectory_rows(
            policy_batch, shard.rows, max(policy_lengths[row] for row in shard.host_rows))

    def critic_shard(
        shard_index: int, microbatch: LatentRolloutBatch, shard, actor=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Critic forward/backward for one shard; returns detached (V, Q)
        and, for a token Q critic, its direction u at every slot.

        ``actor``, in a token Q critic's actor update, runs the shard's
        actor pass between the critic's forward and backward: given the
        detached mean directions, mean state values and per-head
        directions, it returns the Q prior's gradients for each head's
        compact emit directions (None without the prior), which join the
        value loss's single backward, and the taken token's offset from the
        policy's mean action row at every compact emit slot, which centres Q
        before its loss forms.
        """
        rows = shard.rows
        stream_length = shard.stream_length
        directions = None
        prior_directions = None
        binary_errors = None
        if post_action_q:
            values, constrained_probabilities, binary_errors = post_action_q_loss_and_predictions(
                critic, microbatch, mc_targets[rows, :stream_length],
                post_action_q_contract,
            )
            action_values = None
        elif bilinear_q:
            states, member_directions = critic.token_q_parts(microbatch)
            # The actor reads the twins' mean, TD7's actor objective; Q is
            # linear in (c, u), so that is the mean head's.
            directions = member_directions.mean(0).detach()
            # Every action is an EMIT, so the actor's compact emit slots are
            # the critic's action slots, in the same order.
            torch._assert_async(
                (microbatch.action_mask.bool() == microbatch.emit_mask.bool()).all(),
                "a token Q critic's actions are the policy's emits",
            )
            if actor is not None:
                prior_directions, emit_offsets = actor(
                    directions, states.mean(0).detach(), member_directions.detach())
                emit_index = shard.emit_index
            else:
                if not value_only:
                    raise ValueError("a token Q actor update centres Q in its actor pass")
                emit_index = slot_index(microbatch.emit_mask.bool())
                emit_offsets = replay_policy_action_offsets(
                    wrapper, policy_view(shard, microbatch), action_latents, mac_vocab_chunk)
            offsets = scatter_slots(directions.new_zeros(directions.shape), emit_index, emit_offsets)
            member_values = policy_centered_q(states, member_directions, offsets)
            values = member_values.mean(0)
            action_values = None
        else:
            values, action_values = critic.estimates(microbatch)
        if q_critic:
            # Bootstrapped from this pass's own Q, before any step: the
            # parameters that would have produced it at collection. Twin
            # heads bootstrap from the smaller of their two Q values of the
            # next taken token, clipped double Q.
            micro_value_targets = (
                mc_targets[rows, :stream_length]
                if post_action_q else generalized_value_estimate_targets(
                    microbatch.rewards,
                    member_values.amin(0) if twin_q else values,
                    microbatch.action_mask, lambdas[rows],
                )
            )
            with torch.no_grad():
                micro_mc_targets = mc_targets[rows, :stream_length]
                mc_residuals = micro_mc_targets - values.detach()
                totals["mc_target_sum"] += (
                    micro_mc_targets * microbatch.action_mask
                ).sum()
                totals["mc_target_square_sum"] += (
                    micro_mc_targets.square() * microbatch.action_mask
                ).sum()
                totals["mc_residual_sum"] += (
                    mc_residuals * microbatch.action_mask
                ).sum()
                totals["mc_residual_square_sum"] += (
                    mc_residuals.square() * microbatch.action_mask
                ).sum()
        else:
            micro_value_targets = value_targets[rows, :stream_length]
        # Unclipped squared error: the regression target is the lambda-one
        # return, independent of the old values, so there is no old-value
        # trust region to enforce.
        if post_action_q:
            # Probability-domain metrics stay comparable; optimize the strictly
            # proper binary log loss, using logits for numerical stability.
            value_error = binary_errors
        elif twin_q:
            # Each twin regresses the shared target on its own; their losses
            # add, as TD7's do.
            value_error = (member_values - micro_value_targets).square().sum(0)
            with torch.no_grad():
                gap = member_values[0].detach() - member_values[1].detach()
                totals["twin_gap_square_sum"] += (gap.square() * microbatch.action_mask).sum()
                totals["twin_min_shift_sum"] += (
                    (values.detach() - member_values.detach().amin(0)) * microbatch.action_mask
                ).sum()
        else:
            value_error = (values - micro_value_targets).square()
        local_value_numerator = (value_error * microbatch.action_mask).sum()
        weighted_critic_loss = (
            local_value_numerator / value_action_denominator.clamp_min(1)
            if value_scope == "critic"
            # V is fit already; only the head's loss reaches a gradient.
            else local_value_numerator.new_zeros(())
        )
        if actor is not None and prior_directions is not None:
            compact = torch.stack([
                compact_slots(head, shard.emit_index) for head in member_directions])
            weighted_critic_loss = weighted_critic_loss + (compact * prior_directions).sum()
        if action_values is not None:
            # Q regresses on the same lambda-one return: the head sees V
            # detached, so it fits the residual R - V from the action alone.
            local_action_numerator = (
                (action_values - micro_value_targets).square()
                * microbatch.action_mask
            ).sum()
            weighted_critic_loss = weighted_critic_loss + (
                local_action_numerator / value_action_denominator.clamp_min(1)
            )
            with torch.no_grad():
                action_residuals = micro_value_targets - action_values.detach()
                totals["action_value_loss"] += local_action_numerator.detach()
                totals["action_residual_sum"] += (
                    action_residuals * microbatch.action_mask
                ).sum()
                totals["action_residual_square_sum"] += (
                    action_residuals.square() * microbatch.action_mask
                ).sum()
            action_values = action_values.detach()
        finite_guards.append(
            (
                shard_index,
                "value",
                weighted_critic_loss.detach(),
                {},
            )
        )
        weighted_critic_loss.backward()
        with torch.no_grad():
            values = values.detach()
            totals["value_loss"] += local_value_numerator.detach()
            totals["value_sum"] += (values * microbatch.action_mask).sum()
            totals["value_target_sum"] += (
                micro_value_targets * microbatch.action_mask
            ).sum()
            totals["target_square_sum"] += (
                micro_value_targets.square() * microbatch.action_mask
            ).sum()
            residuals = micro_value_targets - values
            totals["residual_sum"] += (
                residuals * microbatch.action_mask
            ).sum()
            totals["residual_square_sum"] += (
                residuals.square() * microbatch.action_mask
            ).sum()
            if value_only:
                # During critic pretraining there is no actor objective, but
                # the diagnostics must still measure calibration against the
                # CURRENT critic. The MC training target is independent of
                # the baseline at lambda=1; reporting the earlier
                # zero-baseline GAE made every successful action look
                # positively advantageous by construction.
                totals["advantage_sum"] += (
                    residuals * microbatch.action_mask
                ).sum()
                totals["advantage_square_sum"] += (
                    residuals.square() * microbatch.action_mask
                ).sum()
        if post_action_q:
            # Advanced indexing makes shard copies: write current Q directly
            # to the parent array, never reuse stale microbatch.old_values.
            advantages[rows, :stream_length] = (
                constrained_probabilities.detach() * actor_signal_rows[rows, None]
            )
        return values, action_values, directions

    def carry_telemetry(microbatch: LatentRolloutBatch, stream_inputs: torch.Tensor) -> None:
        """Accumulate a carry policy's input and memory-write diagnostics."""
        with torch.no_grad():
            # The combiner's whole input delta (carry projection, type
            # bias and MLP stack) against the token embedding it rides
            # on, over generated slots only: zero at initialization, and
            # the first sign the carry is being used.
            # Compacted to the generated slots first, so the diagnostic
            # costs (slots, dim) rather than two dense stream temporaries.
            generated_index = slot_index(generated_slot_mask(microbatch))
            token_latent = wrapper.embed_tokens(
                compact_slots(microbatch.token_ids, generated_index)
            ).float()
            injected = (
                compact_slots(stream_inputs, generated_index).float()
                - token_latent
            )
            totals["carry_input_delta_square_sum"] += injected.square().sum()
            totals["carry_token_square_sum"] += token_latent.square().sum()
            if wrapper.memory_carry:
                # What the writer does with the response's beliefs: mean
                # write strength and mean log decay timescale
                # log 1/(1 - alpha) over heads, so every head counts
                # alike (the arithmetic timescale is the slowest head's)
                # and a saturated gate stays finite. Dense and masked:
                # no host sync for a telemetry gather. Explored writes are
                # reported as made, noise included.
                _, _, log_decay, beta = wrapper.combiner.writer(
                    microbatch.hiddens,
                    recorded_write_noise(microbatch, wrapper.combiner.writer),
                )
                response = (
                    memory_write_mask(microbatch.kind, PAD_SLOT)
                    & generated_slot_mask(microbatch)
                ).to(beta.dtype)[..., None]
                log_timescale = -torch.log(
                    (-torch.expm1(log_decay)).clamp_min(1e-12)
                )
                totals["memory_write_beta_sum"] += (
                    beta * response
                ).sum()
                totals["memory_decay_log_timescale_sum"] += (
                    log_timescale * response
                ).sum()
                totals["memory_write_count"] += (
                    response.sum() * beta.size(-1)
                )

    def actor_shard(
        shard_index: int,
        microbatch: LatentRolloutBatch,
        shard,
        directions: torch.Tensor | None = None,
        state_values: torch.Tensor | None = None,
        member_directions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor] | None:
        """Actor forward/backward for one shard at the current advantages;
        a token Q critic's are formed here from its ``directions`` u and
        its detached ``state_values`` c(s_t). Given each head's detached
        ``member_directions``, MAC's readout chunks also form the Q prior's
        gradients for each head's compact emit directions. A token Q
        critic's pass returns those gradients (None without them) and the
        taken token's offset from the policy's mean action row at every
        compact emit slot."""
        rows = shard.rows
        stream_length = shard.stream_length
        emit_index = shard.emit_index
        if (directions is None) == bilinear_q:
            raise ValueError("token Q directions come with a token Q critic")
        micro_advantages = None if bilinear_q else advantages[rows, :stream_length]
        # Each stream position is an MDP action. THINK is a stored raw
        # Gaussian vector (plus an optional continuation gate); EMIT is a
        # token (plus the one-way STOP gate only while thinking is active).
        beliefs, stream_inputs = replay_head_inputs(
            wrapper, microbatch
        )

        # The CPU replay plan has already found every compact action slot.
        # Every compaction below therefore has a host-known output shape and
        # uses index_select/index_add without a data-dependent CUDA sync.
        assert emit_index is not None
        # Shared with refresh_old_statistics: one compiled artifact for both
        # keeps the two forwards bit-identical (the age-0 zero-clip canary).
        emit_inputs = compact_slots(stream_inputs, emit_index)
        emit_beliefs = compact_slots(beliefs, emit_index)
        emit_targets = compact_next_slots(microbatch.token_ids, emit_index)
        compact_entropy = None
        if mean_actor_critic:
            # Each chunk's vocabulary readout backpropagates into its own
            # detached belief leaf at once, then the temporal graph runs
            # backward a single time; the tied readout's parameter gradients
            # add to it.
            compact_directions = compact_slots(directions, emit_index)
            weights = torch.ones_like(emit_targets, dtype=torch.float32) / (
                policy_action_denominator.clamp_min(1)
            )
            answer = compact_slots(
                post_think_state_mask(microbatch.token_ids, microbatch.emit_mask, think_close_id),
                emit_index,
            ).float()
            # The sampled continuation's return against the critic's state
            # value; the readout subtracts A(a_t) for the control variate's
            # G - Q(a_t).
            dense_residuals = mc_targets[rows, :stream_length] - state_values
            residuals = compact_slots(dense_residuals, emit_index).float()
            baseline_residuals = None
            if sequence_rows is not None:
                # The trajectory's constant coefficient on each of its
                # tokens. The critic's G - c is kept only to report SB's
                # coefficient scale beside it.
                baseline_residuals = residuals
                residuals = sequence_rows[rows][
                    torch.div(emit_index, stream_length, rounding_mode="floor")]
            gate_advantages = None
            # Telemetry of a critic gate reads non-terminal slots only.
            nonterminal = compact_next_slots(microbatch.action_mask, emit_index).float()
            if mac_estimator == "delightful_td_gate":
                # The critic's advantage of the sampled token. Transitions
                # are deterministic, so Q(s_t, a_t) = V(s_{t+1}), and with
                # undiscounted returns r_t + c(s_{t+1}) - c(s_t) is
                # (G_t - c_t) - (G_{t+1} - c_{t+1}): the residual minus the
                # next state's, zero past the last action (the outcome R
                # itself is the terminal value).
                next_residuals = compact_next_slots(dense_residuals, emit_index) * nonterminal
                gate_advantages = residuals - next_residuals.float()
            head_advantages = None
            if mac_estimator == "delightful_shuffled_head_gate":
                # Every state's head advantage before any loss, from the
                # MAC readout's own arithmetic, then permuted within its
                # surprisal-and-span stratum.
                with torch.no_grad():
                    head_logprobs, head_advantages = (
                        torch.cat(parts) for parts in zip(*(
                            nano_target_logprobs_and_advantages(
                                backbone, emit_beliefs[start:start + mac_vocab_chunk],
                                emit_targets[start:start + mac_vocab_chunk],
                                compact_directions[start:start + mac_vocab_chunk],
                                action_latents)
                            for start in range(0, emit_targets.numel(), mac_vocab_chunk))))
                    gate_advantages = shuffle_within_strata(
                        head_advantages, gate_shuffle_strata(head_logprobs, answer),
                        shuffle_generator)
            prior = member_directions is not None
            if prior:
                compact_members = torch.stack([
                    compact_slots(head, emit_index) for head in member_directions])
                # Normalized as the critic's squared error is.
                prior_weights = torch.full_like(
                    weights, critic_q_pessimism) / value_action_denominator.clamp_min(1)
                prior_directions = []
            leaves, statistics, losses, offsets = [], [], [], []
            for start in range(0, emit_targets.numel(), mac_vocab_chunk):
                end = start + mac_vocab_chunk
                leaf = emit_beliefs[start:end].detach().requires_grad_(True)
                if prior:
                    chunk_directions, penalty, positive = nano_pessimistic_q_prior(
                        backbone, leaf.detach(), compact_members[:, start:end], action_latents,
                        prior_weights[start:end],
                    )
                    prior_directions.append(chunk_directions)
                    with torch.no_grad():
                        totals["q_prior_penalty_sum"] += penalty.sum()
                        totals["q_prior_positive_sum"] += positive.sum()
                        totals["q_prior_count"] += penalty.numel() * (leaf.size(0))
                loss, *chunk_statistics = nano_mean_actor_critic(
                    backbone, leaf, emit_targets[start:end],
                    compact_directions[start:end], action_latents, weights[start:end],
                    residuals[start:end], mac_estimator,
                    None if gate_advantages is None else gate_advantages[start:end],
                )
                loss.backward()
                leaves.append(leaf)
                losses.append(loss.detach())
                *chunk_statistics, chunk_census, chunk_offsets = chunk_statistics
                with torch.no_grad():
                    totals["mac_support_census"] += chunk_census
                statistics.append(chunk_statistics)
                offsets.append(chunk_offsets)
            emit_beliefs.backward(torch.cat([leaf.grad for leaf in leaves]))
            compact_token_logprobs, compact_entropy, compact_advantages, q_variance = (
                torch.cat(values) for values in zip(*statistics)
            )
            shard_loss = torch.stack(losses).sum()
            finite_guards.append((shard_index, "mac_actor", shard_loss, {}))
            with torch.no_grad():
                totals["policy_loss"] += shard_loss
                local = torch.zeros_like(microbatch.old_token_logprobs)
                scatter_slots(local, emit_index, compact_token_logprobs)
                batch.old_token_logprobs[rows, :stream_length] = local
                # The sampled log-probability's return coefficient, as the
                # estimator formed it from G - c: G - Q(a_t), G - E_pi Q, or
                # that times its delight gate.
                coefficients = sampled_coefficients(
                    mac_estimator, residuals, compact_advantages.float(),
                    compact_token_logprobs, gate_advantages)
                # The reported advantage is the one the policy gradient
                # weights: the sampled coefficient for the sampled-only
                # estimators, the critic's centered Q of the taken token
                # otherwise.
                local = torch.zeros_like(microbatch.old_values)
                scatter_slots(
                    local, emit_index,
                    coefficients if mac_estimator in SAMPLED_ONLY_ESTIMATORS
                    else compact_advantages,
                )
                if mac_estimator in DELIGHTFUL_MAC_ESTIMATORS:
                    delight_advantages = delight_inputs(
                        mac_estimator, residuals, compact_advantages.float(), gate_advantages)
                    gate = delight_gate(delight_advantages, compact_token_logprobs)
                    for span, mask in (("answer", answer), ("think", 1 - answer)):
                        totals[f"mac_gate_{span}_sum"] += (gate * mask).sum()
                    # Most sampled tokens are near-certain, so their gate is
                    # 1/2 and their score e_a - pi is near zero. Weighting by
                    # the policy-gradient term's sampled-logit magnitude
                    # |r| (1 - pi(a_t)) reads the gate where the gradient is;
                    # |U * surprisal| > 1 puts it outside [0.27, 0.73].
                    mass = residuals.abs() * -torch.expm1(compact_token_logprobs.float())
                    active = (delight_advantages * compact_token_logprobs).abs() > 1
                    if mac_estimator in CRITIC_GATED_ESTIMATORS:
                        # How much of the outcome a critic advantage tracks:
                        # its scale, and its correlation with r = G - c.
                        # Non-terminal slots only: at a row's last action
                        # the TD residual is r exactly, and those points
                        # would pull a pure-noise critic toward perfect
                        # tracking. ``gate``: the TD residual; ``head``:
                        # the action head's A(a_t), the shuffled gate's
                        # before its permutation; ``shuffled``: what that
                        # gate reads.
                        moments = {
                            "delightful_td_gate": {"gate": gate_advantages},
                            "delightful_head_gate": {"head": delight_advantages},
                            "delightful_shuffled_head_gate": {
                                "head": head_advantages, "shuffled": gate_advantages},
                        }[mac_estimator]
                        for source, values in moments.items():
                            totals[f"mac_{source}_advantage_states_sum"] += nonterminal.sum()
                            totals[f"mac_{source}_advantage_square_sum"] += (
                                values.square() * nonterminal).sum()
                            totals[f"mac_{source}_advantage_residual_sum"] += (
                                values * residuals * nonterminal).sum()
                        totals["mac_residual_square_sum"] += (
                            residuals.square() * nonterminal).sum()
                        if head_advantages is not None:
                            # What the permutation leaves of A: its overlap
                            # with the original, and the gradient mass
                            # |r| (1 - pi) at states it did not move
                            # (singleton strata, or fixed points); by mass,
                            # since near-certain states fill the strata
                            # that matter least.
                            totals["mac_shuffle_head_cross_sum"] += (
                                head_advantages * gate_advantages * nonterminal).sum()
                            totals["mac_shuffle_fixed_mass_sum"] += (
                                mass * (head_advantages == gate_advantages)).sum()
                            totals["mac_shuffle_mass_sum"] += mass.sum()
                        # The gate where A is the critic's alone, not R - c.
                        totals["mac_gate_nonterminal_mass_sum"] += (mass * nonterminal).sum()
                        totals["mac_gate_nonterminal_weighted_sum"] += (
                            gate * mass * nonterminal).sum()
                        totals["mac_gate_nonterminal_active_mass_sum"] += (
                            mass * active * nonterminal).sum()
                    for sign, mask in (("positive", residuals > 0), ("negative", residuals < 0)):
                        totals[f"mac_gate_{sign}_sum"] += (gate * mask).sum()
                        totals[f"mac_gate_{sign}_count"] += mask.sum()
                        totals[f"mac_gate_{sign}_mass_sum"] += (mass * mask).sum()
                        totals[f"mac_gate_{sign}_weighted_sum"] += (gate * mass * mask).sum()
                        totals[f"mac_gate_{sign}_active_mass_sum"] += (
                            mass * (mask & active)).sum()
                advantages[rows, :stream_length] = local
                totals["entropy_sum"] += compact_entropy.sum()
                totals["entropy_count"] += compact_entropy.numel()
                totals["mac_q_variance_answer_sum"] += (q_variance * answer).sum()
                totals["mac_q_variance_think_sum"] += (q_variance * (1 - answer)).sum()
                for span, mask in (("answer", answer), ("think", 1 - answer)):
                    totals[f"mac_coefficient_{span}_sum"] += (coefficients * mask).sum()
                    totals[f"mac_coefficient_square_{span}_sum"] += (
                        coefficients.square() * mask).sum()
                    if baseline_residuals is not None:
                        totals[f"mac_sb_coefficient_square_{span}_sum"] += (
                            baseline_residuals.square() * mask).sum()
            if hidden_carry:
                carry_telemetry(microbatch, stream_inputs)
            return (torch.cat(prior_directions, dim=1) if prior else None,
                    torch.cat(offsets))
        if bilinear_q:
            compact_token_logprobs, compact_entropy, compact_advantages, emit_offsets = (
                compact_emit_token_logprobs_entropy_and_advantages(
                    wrapper, emit_inputs, emit_beliefs, emit_targets,
                    compact_slots(directions, emit_index), action_latents,
                )
            )
            micro_advantages = torch.zeros_like(microbatch.old_values)
            scatter_slots(micro_advantages, emit_index, compact_advantages)
            micro_advantages = micro_advantages * actor_signal_rows[rows, None]
            advantages[rows, :stream_length] = micro_advantages
        elif target_policy_optimization:
            compact_token_log_odds = compact_emit_token_log_odds(
                wrapper, emit_inputs, emit_beliefs, emit_targets
            )
            compact_token_logprobs = F.logsigmoid(compact_token_log_odds)
        elif entropy_temperature is not None:
            compact_token_logprobs, compact_entropy = (
                compact_emit_token_logprobs_and_entropy(
                    wrapper, emit_inputs, emit_beliefs, emit_targets
                )
            )
        else:
            compact_token_logprobs = compact_emit_token_logprobs(
                wrapper, emit_inputs, emit_beliefs, emit_targets
            )
        new_token_logprobs = torch.zeros_like(microbatch.old_token_logprobs)
        scatter_slots(new_token_logprobs, emit_index, compact_token_logprobs)
        if fused_behavior_statistics:
            old_token_logprobs = new_token_logprobs.detach()
            with torch.no_grad():
                # The post-update drift and credit diagnostics read the
                # behavior log-probs back from the batch.
                batch.old_token_logprobs[rows, :stream_length] = (
                    old_token_logprobs
                )
        else:
            old_token_logprobs = microbatch.old_token_logprobs
        active_emit_mask = (
            microbatch.emit_mask * actor_signal_rows[rows, None]
        )

        if hidden_carry:
            # Tokens are the carry policy's only actions: there is no gate or
            # Gaussian factor to clip, so the joint VAPO ratio is the token
            # ratio itself. The carry's credit reaches the combiner through
            # the replayed stream inputs of the token objective.
            new_stochastic_logprobs = None
            old_stochastic_logprobs = None
            weighted_stochastic_loss = new_token_logprobs.new_zeros(())
            weighted_stochastic_clip = new_token_logprobs.new_zeros(())
            carry_telemetry(microbatch, stream_inputs)
        else:
            new_stop_logprobs = -F.binary_cross_entropy_with_logits(
                wrapper.gate.stop_logit(beliefs),
                microbatch.actions.float(),
                reduction="none",
            )
            new_thought_joint = torch.zeros_like(new_token_logprobs)
            old_thought_joint = torch.zeros_like(new_token_logprobs)
            think_index = shard.think_index
            if think_index.numel() and fused_behavior_statistics:
                raise ValueError(
                    "fused behavior statistics do not cover latent thoughts"
                )
            if think_index.numel():
                thought_means, thought_targets = compact_thought_actions(
                    wrapper, microbatch, beliefs, think_index
                )
                thought_log_sigma = wrapper.transition.predict_log_sigma(
                    compact_slots(beliefs, think_index)
                )
                compact_new_thought = wrapper.transition.per_dim_log_prob(
                    thought_targets, thought_means, thought_log_sigma
                )
                compact_old_thought = compact_slots(
                    microbatch.old_thought_logprobs, think_index
                )
                scatter_slots(
                    new_thought_joint,
                    think_index,
                    compact_new_thought.sum(-1),
                )
                scatter_slots(
                    old_thought_joint,
                    think_index,
                    compact_old_thought.sum(-1),
                )
            new_stochastic_logprobs = (
                new_stop_logprobs * microbatch.stop_mask + new_thought_joint
            )
            if fused_behavior_statistics:
                old_stochastic_logprobs = new_stochastic_logprobs.detach()
                with torch.no_grad():
                    batch.old_stop_logprobs[rows, :stream_length] = (
                        new_stop_logprobs.detach().float()
                        * microbatch.stop_mask
                    )
            else:
                old_stochastic_logprobs = (
                    microbatch.old_stop_logprobs * microbatch.stop_mask
                    + old_thought_joint
                )
            stochastic_mask = (
                (microbatch.stop_mask.bool() | think_slot_mask(microbatch))
                .to(microbatch.action_mask.dtype)
                * actor_signal_rows[rows, None]
            )
            weighted_stochastic_loss, weighted_stochastic_clip, _ = (
                clipped_policy_loss(
                    new_stochastic_logprobs,
                    old_stochastic_logprobs,
                    micro_advantages,
                    stochastic_mask,
                    denominator=value_action_denominator,
                    estimate_kl=False,
                )
            )

        if target_policy_optimization:
            new_token_log_odds = torch.zeros_like(
                microbatch.old_token_log_odds
            )
            scatter_slots(
                new_token_log_odds, emit_index, compact_token_log_odds
            )
            weighted_policy_loss, tpo = target_policy_loss(
                new_token_log_odds,
                microbatch.old_token_log_odds,
                micro_advantages,
                active_emit_mask,
                eta=tpo_eta,
                denominator=policy_action_denominator,
            )
            weighted_policy_clip = weighted_policy_loss.detach().new_zeros(())
            for source_name, total_name in (
                ("active_count", "tpo_active_count"),
                ("loss_sum", "tpo_loss_sum"),
                ("fit_kl_sum", "tpo_fit_kl_sum"),
                ("target_behavior_kl_sum", "tpo_target_behavior_kl_sum"),
                ("old_probability_sum", "tpo_old_probability_sum"),
                ("current_probability_sum", "tpo_current_probability_sum"),
                ("target_probability_sum", "tpo_target_probability_sum"),
                ("target_move_abs_sum", "tpo_target_move_abs_sum"),
                (
                    "target_log_odds_shift_abs_sum",
                    "tpo_target_log_odds_shift_abs_sum",
                ),
                (
                    "target_log_odds_shift_square_sum",
                    "tpo_target_log_odds_shift_square_sum",
                ),
                ("residual_sum", "tpo_probability_residual_sum"),
                ("residual_abs_sum", "tpo_probability_residual_abs_sum"),
                (
                    "residual_square_sum",
                    "tpo_probability_residual_square_sum",
                ),
                ("positive_count", "tpo_positive_count"),
                ("negative_count", "tpo_negative_count"),
                ("neutral_count", "tpo_neutral_count"),
                (
                    "positive_target_probability_sum",
                    "tpo_positive_target_probability_sum",
                ),
                (
                    "negative_target_probability_sum",
                    "tpo_negative_target_probability_sum",
                ),
            ):
                totals[total_name] += tpo[source_name]
        elif delightful_policy_gradient:
            policy_loss_function = (
                binary_q_delightful_policy_loss if post_action_q
                else delightful_policy_loss
            )
            weighted_policy_loss, delightful = policy_loss_function(
                new_token_logprobs,
                micro_advantages,
                active_emit_mask,
                denominator=policy_action_denominator,
            )
            weighted_policy_clip = weighted_policy_loss.detach().new_zeros(())
            if post_action_q:
                totals["binary_q_coefficient_sum"] += delightful["coefficient_sum"]
            totals["delightful_gate_sum"] += delightful["gate_sum"]
            totals["delightful_positive_gate_sum"] += delightful[
                "positive_gate_sum"
            ]
            totals["delightful_positive_count"] += delightful[
                "positive_count"
            ]
            totals["delightful_negative_gate_sum"] += delightful[
                "negative_gate_sum"
            ]
            totals["delightful_negative_count"] += delightful[
                "negative_count"
            ]
            totals["delightful_delight_sum"] += delightful["delight_sum"]
            totals["delightful_surprisal_sum"] += delightful[
                "surprisal_sum"
            ]
            totals["delightful_positive_loss_sum"] += delightful[
                "positive_loss_sum"
            ]
            totals["delightful_negative_loss_sum"] += delightful[
                "negative_loss_sum"
            ]
        else:
            joint_new_logprobs = (
                new_token_logprobs
                if new_stochastic_logprobs is None
                else new_token_logprobs + new_stochastic_logprobs
            )
            joint_old_logprobs = (
                old_token_logprobs
                if old_stochastic_logprobs is None
                else old_token_logprobs + old_stochastic_logprobs
            )
            weighted_policy_loss, weighted_policy_clip, _ = clipped_policy_loss(
                joint_new_logprobs,
                joint_old_logprobs,
                micro_advantages,
                actor_active_mask[rows, :stream_length],
                denominator=policy_action_denominator,
                estimate_kl=False,
            )
        if target_policy_optimization or delightful_policy_gradient:
            weighted_policy_loss = (
                weighted_policy_loss + weighted_stochastic_loss
            )
            weighted_policy_clip = (
                weighted_policy_clip + weighted_stochastic_clip
            )
        totals["stochastic_policy_loss"] += weighted_stochastic_loss.detach()
        totals["stochastic_policy_clip"] += weighted_stochastic_clip.detach()
        actor_total = weighted_policy_loss
        if compact_entropy is not None:
            # The token Q's fused readout yields the exact entropy at no
            # extra cost, so it is reported whether or not an entropy term
            # consumes it; without the term it stays a diagnostic.
            compact_active = compact_slots(active_emit_mask, emit_index).to(
                compact_entropy.dtype
            )
            active_entropy = compact_entropy * compact_active
            if entropy_temperature is not None:
                # REPPO's temperature * log pi, in expectation: the entropy
                # term over the same emit slots and denominator as the
                # policy loss.
                entropy_loss = -(
                    entropy_temperature
                    * active_entropy.sum()
                    / policy_action_denominator.clamp_min(1)
                )
                actor_total = actor_total + entropy_loss
                totals["entropy_loss"] += entropy_loss.detach()
            with torch.no_grad():
                totals["entropy_sum"] += active_entropy.detach().sum()
                totals["entropy_count"] += compact_active.sum()
        if post_action_q:
            domain_valid = delightful["valid_logprobs"] & delightful["valid_q"]
            finite_guards.append((
                shard_index, "binary_q_input_domain",
                torch.where(domain_valid, actor_total.detach().new_zeros(()),
                            actor_total.detach().new_full((), float("nan"))),
                {name: delightful[name] for name in (
                    "positive_logprob_count", "nonfinite_logprob_count", "positive_logprob_max",
                )},
            ))
        finite_guards.append(
            (
                shard_index,
                "actor",
                actor_total.detach(),
                {"policy": weighted_policy_loss.detach()},
            )
        )
        actor_total.backward()
        with torch.no_grad():
            token_log_ratio = new_token_logprobs - old_token_logprobs
            active_log_ratio = token_log_ratio * microbatch.action_mask
            totals["token_abs_log_ratio_max"] = torch.maximum(
                totals["token_abs_log_ratio_max"],
                active_log_ratio.abs().max(),
            )
            # PPO intentionally leaves A<0, ratio>1+epsilon unclipped. This
            # tail statistic directly exposes the branch that overflowed in
            # the first frozen-pool run without changing the objective.
            harmful_positive_log_ratio = torch.where(
                (micro_advantages < 0) & microbatch.action_mask.bool(),
                token_log_ratio.clamp_min(0),
                torch.zeros_like(token_log_ratio),
            )
            totals["harmful_positive_log_ratio_max"] = torch.maximum(
                totals["harmful_positive_log_ratio_max"],
                harmful_positive_log_ratio.max(),
            )
            totals["policy_loss"] += weighted_policy_loss.detach()
            totals["policy_clip"] += weighted_policy_clip
            # k3 estimator of the token behavior KL over action slots.
            totals["token_kl_sum"] += (
                (torch.expm1(token_log_ratio) - token_log_ratio)
                * microbatch.action_mask
            ).sum()
        if bilinear_q:
            return None, emit_offsets
        return None

    if fused_behavior_statistics and q_critic:
        for shard_index, (microbatch, shard) in enumerate(
            iter_planned_replay_microbatches(batch, replay_plan)
        ):
            if bilinear_q:
                # The critic's loss waits for the actor pass: Q is centred
                # under the policy its readout forms.
                def actor(directions, states, members, microbatch=microbatch, shard=shard,
                          shard_index=shard_index):
                    return actor_shard(shard_index, microbatch, shard, directions, states,
                                       members if critic_q_pessimism else None)
                values, _, directions = critic_shard(shard_index, microbatch, shard, actor)
            else:
                values, _, directions = critic_shard(shard_index, microbatch, shard)
            with torch.no_grad():
                batch.old_values[shard.rows, : shard.stream_length] = values
            if not bilinear_q:
                actor_shard(shard_index, microbatch, shard)
            del directions
        batch.statistics_refreshed = True
    elif fused_behavior_statistics:
        for shard_index, (microbatch, shard) in enumerate(
            iter_planned_replay_microbatches(batch, replay_plan)
        ):
            values, action_values, _ = critic_shard(
                shard_index, microbatch, shard
            )
            with torch.no_grad():
                # Advanced row indexing materializes a copy, so assignment
                # targets the parent, as in refresh_old_statistics.
                batch.old_values[shard.rows, : shard.stream_length] = (
                    values.float()
                )
                if action_values is not None:
                    batch.old_action_values[
                        shard.rows, : shard.stream_length
                    ] = action_values
        advantages = actor_advantages()
        for shard_index, (microbatch, shard) in enumerate(
            iter_planned_replay_microbatches(batch, replay_plan)
        ):
            actor_shard(shard_index, microbatch, shard)
        batch.statistics_refreshed = True
    else:
        for shard_index, (microbatch, shard) in enumerate(
            iter_planned_replay_microbatches(batch, replay_plan)
        ):
            critic_shard(shard_index, microbatch, shard)
            if not value_only:
                actor_shard(shard_index, microbatch, shard)
    advantage_groups: dict[str, torch.Tensor] = {}
    if not value_only:
        with torch.no_grad():
            totals["advantage_sum"] = (advantages * batch.action_mask).sum()
            totals["advantage_square_sum"] = (
                advantages.square() * batch.action_mask
            ).sum()
            emit = batch.emit_mask.to(advantages.dtype)
            if advantage_source_count and batch.source_id is None:
                raise ValueError("a source breakdown needs the batch's source ids")
            for source_id in range(advantage_source_count):
                advantage_groups[f"source_{source_id}"] = emit * (
                    batch.source_id == source_id
                ).to(emit.dtype)[:, None]
            if advantage_token_classes:
                taken = next_slot_tokens(batch.token_ids)
                unlisted = emit
                for name, (token_ids, previous_ids) in (
                    advantage_token_classes.items()
                ):
                    member = torch.isin(taken, token_ids)
                    if previous_ids is not None:
                        # The token at a slot precedes the one taken there.
                        member &= torch.isin(batch.token_ids, previous_ids)
                    advantage_groups[name] = emit * member.to(emit.dtype)
                    unlisted = unlisted - advantage_groups[name]
                advantage_groups["other"] = unlisted
            if think_close_id is not None:
                answer = post_think_state_mask(
                    batch.token_ids, batch.emit_mask, think_close_id
                ).to(emit.dtype)
                advantage_groups["span_think"] = emit - answer
                advantage_groups["span_answer"] = answer

    if finite_guards:
        # One host sync resolves every shard's loss guard, after all backward
        # passes are queued. This must stay ABOVE every ``step_optimizers``
        # call in this function -- including the ``value_only`` return path --
        # because ``actor_step``/``critic_step`` default to True and standalone
        # callers do step here.
        guarded = torch.stack(
            [value.reshape(()).float() for _, _, value, _ in finite_guards]
        )
        if not torch.isfinite(guarded).all():
            offenders = []
            for (shard, label, _, components), finite in zip(
                finite_guards, torch.isfinite(guarded).tolist(), strict=True
            ):
                if finite:
                    continue
                detail = " ".join(
                    f"{name}={float(component)}"
                    for name, component in components.items()
                )
                offenders.append(
                    f"shard={shard} {label}" + (f" {detail}" if detail else "")
                )
            raise RuntimeError(
                "non-finite loss before optimizer step: "
                + "; ".join(offenders)
            )

    action_denom = denominators["action"]
    advantage_mean = totals["advantage_sum"] / action_denom
    advantage_variance = (
        totals["advantage_square_sum"] / action_denom
        - advantage_mean.square()
    ).clamp_min(0)
    value_target_mean = totals["value_target_sum"] / action_denom
    target_variance = (
        totals["target_square_sum"] / action_denom
        - value_target_mean.square()
    ).clamp_min(0)
    residual_mean = totals["residual_sum"] / action_denom
    residual_variance = (
        totals["residual_square_sum"] / action_denom
        - residual_mean.square()
    ).clamp_min(0)
    explained_variance = torch.where(
        target_variance > 1e-12,
        1.0 - residual_variance / target_variance.clamp_min(1e-12),
        torch.zeros_like(target_variance),
    )
    metric_tensors = {
        "value_loss": totals["value_loss"] / action_denom,
        "value_mean": totals["value_sum"] / action_denom,
        "value_target_mean": value_target_mean,
        "value_target_variance": target_variance,
        "value_residual_mean": residual_mean,
        "value_residual_variance": residual_variance,
        "explained_variance": explained_variance,
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_variance.sqrt(),
        "action_count": batch.action_mask.sum(),
        "emit_action_count": batch.emit_mask.sum(),
        "actor_active_trajectory_fraction": actor_signal_rows.float().mean(),
    }
    if post_action_q:
        metric_tensors["binary_success_bce"] = totals["value_loss"] / action_denom
        metric_tensors["binary_success_brier"] = totals["residual_square_sum"] / action_denom
        metric_tensors["binary_success_r_squared"] = torch.where(
            target_variance > 1e-12,
            1.0 - metric_tensors["binary_success_brier"] / target_variance.clamp_min(1e-12),
            torch.zeros_like(target_variance),
        )
        if not value_only:
            metric_tensors["actor_success_probability_mean"] = advantage_mean
            metric_tensors["actor_success_probability_std"] = advantage_variance.sqrt()
            metric_tensors["binary_dg_coefficient_mean"] = totals["binary_q_coefficient_sum"] / action_denom
    for name, mask in advantage_groups.items():
        metric_tensors[f"advantage_group/{name}/sum"] = (
            advantages * mask
        ).sum()
        metric_tensors[f"advantage_group/{name}/square_sum"] = (
            advantages.square() * mask
        ).sum()
        metric_tensors[f"advantage_group/{name}/count"] = mask.sum()
    if q_critic:
        # Q's fit against the Monte-Carlo return: the explained variance a
        # V critic reports, beside ``explained_variance`` against Q's own
        # generalized value estimates.
        mc_target_mean = totals["mc_target_sum"] / action_denom
        mc_target_variance = (
            totals["mc_target_square_sum"] / action_denom
            - mc_target_mean.square()
        ).clamp_min(0)
        mc_residual_mean = totals["mc_residual_sum"] / action_denom
        mc_residual_variance = (
            totals["mc_residual_square_sum"] / action_denom
            - mc_residual_mean.square()
        ).clamp_min(0)
        metric_tensors.update(
            mc_target_mean=mc_target_mean,
            mc_target_variance=mc_target_variance,
            mc_residual_mean=mc_residual_mean,
            mc_residual_variance=mc_residual_variance,
            mc_explained_variance=torch.where(
                mc_target_variance > 1e-12,
                1.0 - mc_residual_variance / mc_target_variance.clamp_min(1e-12),
                torch.zeros_like(mc_target_variance),
            ),
        )
    if critic_q_pessimism:
        metric_tensors.update({
            "q_prior/penalty_sum": totals["q_prior_penalty_sum"],
            "q_prior/positive_sum": totals["q_prior_positive_sum"],
            "q_prior/count": totals["q_prior_count"],
        })
    if twin_q:
        metric_tensors.update({
            "twin/gap_square_sum": totals["twin_gap_square_sum"],
            "twin/min_shift_sum": totals["twin_min_shift_sum"],
            "twin/action_count": batch.action_mask.sum(),
        })
    if (entropy_temperature is not None or q_critic) and not value_only:
        metric_tensors.update(
            policy_entropy_sum=totals["entropy_sum"],
            policy_entropy_count=totals["entropy_count"],
        )
    if mean_actor_critic and not value_only:
        metric_tensors.update({
            f"support/{name}_sum": totals["mac_support_census"][index]
            for index, name in enumerate(SUPPORT_CENSUS_FIELDS)
        })
        metric_tensors.update({
            f"mac/{span}/{name}_sum": totals[f"mac_{name}_{span}_sum"]
            for span in ("think", "answer")
            for name in ("q_variance", "coefficient", "coefficient_square")
        })
        if mac_estimator in CRITIC_GATED_ESTIMATORS:
            sources = {"delightful_td_gate": ("gate",), "delightful_head_gate": ("head",),
                       "delightful_shuffled_head_gate": ("head", "shuffled")}[mac_estimator]
            metric_tensors.update({
                f"mac/{name}_sum": totals[f"mac_{name}_sum"]
                for name in (
                    *(f"{source}_advantage_{moment}" for source in sources
                      for moment in ("states", "square", "residual")),
                    "residual_square", "gate_nonterminal_mass", "gate_nonterminal_weighted",
                    "gate_nonterminal_active_mass",
                    *(("shuffle_head_cross", "shuffle_fixed_mass", "shuffle_mass")
                      if "shuffled" in sources else ()))
            })
        if sequence_rows is not None:
            metric_tensors.update({
                f"mac/{span}/sb_coefficient_square_sum":
                    totals[f"mac_sb_coefficient_square_{span}_sum"]
                for span in ("think", "answer")
            })
            metric_tensors.update(sequence_estimator_moments(
                mac_estimator, batch.group_advantage.float(), batch.answer_surprisal.float(),
                sequence_rows, batch.emit_mask.sum(1).float()))
        if mac_estimator in DELIGHTFUL_MAC_ESTIMATORS:
            metric_tensors.update({
                f"mac/{span}/gate_sum": totals[f"mac_gate_{span}_sum"]
                for span in ("think", "answer")
            })
            metric_tensors.update({
                f"mac/gate_{sign}_{name}": totals[f"mac_gate_{sign}_{name}"]
                for sign in ("positive", "negative")
                for name in ("sum", "count", "mass_sum", "weighted_sum", "active_mass_sum")
            })
    if entropy_temperature is not None and not value_only:
        # Already over the complete minibatch's policy denominator, a
        # contribution to sum across groups like ``policy_loss``.
        metric_tensors.update(entropy_loss=totals["entropy_loss"])
    if action_head:
        # Same moments as V's, against the same targets: the head's own fit,
        # and how much of V's residual the sampled action explains.
        action_residual_mean = totals["action_residual_sum"] / action_denom
        action_residual_variance = (
            totals["action_residual_square_sum"] / action_denom
            - action_residual_mean.square()
        ).clamp_min(0)
        metric_tensors.update(
            action_value_loss=totals["action_value_loss"] / action_denom,
            action_value_residual_mean=action_residual_mean,
            action_value_residual_variance=action_residual_variance,
            action_value_explained_variance=torch.where(
                target_variance > 1e-12,
                1.0 - action_residual_variance / target_variance.clamp_min(1e-12),
                torch.zeros_like(target_variance),
            ),
            action_head_grad_norm=gradient_norm_tensor(
                critic.action_head.parameters()
            ),
        )
    if value_only:
        metric_tensors["critic_grad_norm"] = gradient_norm_tensor(
            critic.parameters()
        )
        if critic_step:
            step_optimizers(optimizers, critic_role)
        return scalar_tensors_to_floats(metric_tensors)
    if not post_action_q:
        metric_tensors.update(
            credit_assignment_moments(
                batch,
                advantages,
                # A Q critic's TD(0) and Monte-Carlo residuals are Q's own:
                # r_t + Q_{t+1} - Q_t against R - Q_t.
                mc_targets if q_critic else value_targets,
                actor_signal_rows,
                reward_floor,
                action_values=batch.old_action_values if action_head else None,
            )
        )

    # Norms are cumulative across prompt groups; the final group reports the
    # complete pre-step norm. Fresh probes are registered
    # under blocks[-1], so exclude them from the trunk norm by identity.
    excluded_parameter_ids = non_trunk_parameter_ids(backbone)
    grad_norms = {
        "trunk_grad_norm": gradient_norm_tensor(
            parameter
            for parameter in backbone.parameters()
            if id(parameter) not in excluded_parameter_ids
        ),
        "renderer_grad_norm": gradient_norm_tensor(
            renderer_parameters(backbone)
        ),
        "combiner_grad_norm": gradient_norm_tensor(
            wrapper.combiner.parameters()
        ),
        "critic_grad_norm": gradient_norm_tensor(critic.parameters()),
    }
    if not hidden_carry:
        grad_norms.update(
            gate_grad_norm=gradient_norm_tensor(wrapper.gate.parameters()),
            thought_mean_grad_norm=gradient_norm_tensor(
                wrapper.transition.parameters()
            ),
        )
    if critic_step:
        step_optimizers(optimizers, "critic")
    if actor_step and "actor" in optimizers:
        step_optimizers(optimizers, "actor")
    metric_tensors.update(
        policy_loss=totals["policy_loss"],
        # Each shard's clip fraction already carries the complete-minibatch
        # denominator, so summing shards (and, in the trainer, groups) yields
        # the exact clipped-action fraction of the optimizer minibatch.
        policy_clip_fraction=totals["policy_clip"],
        token_behavior_kl=totals["token_kl_sum"] / denominators["action"],
        policy_behavior_kl_per_action=(
            totals["token_kl_sum"] / policy_action_denominator.clamp_min(1)
        ),
        token_abs_log_ratio_max=totals["token_abs_log_ratio_max"],
        harmful_positive_log_ratio_max=(
            totals["harmful_positive_log_ratio_max"]
        ),
        reward=batch.reward_scalar.mean(),
        reward_max=batch.reward_scalar.amax(),
        reward_min=batch.reward_scalar.amin(),
        **grad_norms,
    )
    if hidden_carry:
        # Per-minibatch RMS ratio over generated slots, summed across shards
        # before the one division.
        metric_tensors.update(
            carry_input_delta_rms_ratio=(
                totals["carry_input_delta_square_sum"]
                / totals["carry_token_square_sum"].clamp_min(1e-30)
            ).sqrt(),
        )
        if wrapper.memory_carry:
            # Means over every head of every response write in the minibatch.
            write_count = totals["memory_write_count"].clamp_min(1)
            metric_tensors.update(
                memory_write_beta_mean=(
                    totals["memory_write_beta_sum"] / write_count
                ),
                memory_decay_log_timescale_mean=(
                    totals["memory_decay_log_timescale_sum"] / write_count
                ),
            )
    else:
        metric_tensors.update(
            stochastic_policy_loss=totals["stochastic_policy_loss"],
            stochastic_policy_clip_fraction=totals["stochastic_policy_clip"],
        )
    if target_policy_optimization:
        positive_count = totals["tpo_positive_count"]
        negative_count = totals["tpo_negative_count"]
        tpo_active_count = totals["tpo_active_count"]
        tpo_denom = tpo_active_count.clamp_min(1)
        metric_tensors.update(
            tpo_active_count=tpo_active_count,
            tpo_loss=totals["tpo_loss_sum"] / tpo_denom,
            tpo_pre_update_fit_kl=totals["tpo_fit_kl_sum"] / tpo_denom,
            tpo_target_behavior_kl=(
                totals["tpo_target_behavior_kl_sum"] / tpo_denom
            ),
            tpo_old_probability_mean=(
                totals["tpo_old_probability_sum"] / tpo_denom
            ),
            tpo_pre_update_probability_mean=(
                totals["tpo_current_probability_sum"] / tpo_denom
            ),
            tpo_target_probability_mean=(
                totals["tpo_target_probability_sum"] / tpo_denom
            ),
            tpo_target_move_abs_mean=(
                totals["tpo_target_move_abs_sum"] / tpo_denom
            ),
            tpo_target_log_odds_shift_abs_mean=(
                totals["tpo_target_log_odds_shift_abs_sum"] / tpo_denom
            ),
            tpo_target_log_odds_shift_square_mean=(
                totals["tpo_target_log_odds_shift_square_sum"] / tpo_denom
            ),
            tpo_pre_update_probability_residual_mean=(
                totals["tpo_probability_residual_sum"] / tpo_denom
            ),
            tpo_pre_update_probability_residual_abs_mean=(
                totals["tpo_probability_residual_abs_sum"] / tpo_denom
            ),
            tpo_pre_update_probability_residual_square_mean=(
                totals["tpo_probability_residual_square_sum"] / tpo_denom
            ),
            tpo_positive_target_probability_mean=torch.where(
                positive_count > 0,
                totals["tpo_positive_target_probability_sum"]
                / positive_count.clamp_min(1),
                torch.zeros_like(positive_count),
            ),
            tpo_negative_target_probability_mean=torch.where(
                negative_count > 0,
                totals["tpo_negative_target_probability_sum"]
                / negative_count.clamp_min(1),
                torch.zeros_like(negative_count),
            ),
            tpo_positive_count=positive_count,
            tpo_negative_count=negative_count,
            tpo_neutral_count=totals["tpo_neutral_count"],
        )
    elif delightful_policy_gradient:
        positive_count = totals["delightful_positive_count"]
        negative_count = totals["delightful_negative_count"]
        metric_tensors.update(
            delightful_gate_mean=(
                totals["delightful_gate_sum"] / action_denom
            ),
            delightful_positive_gate_mean=torch.where(
                positive_count > 0,
                totals["delightful_positive_gate_sum"]
                / positive_count.clamp_min(1),
                torch.zeros_like(positive_count),
            ),
            delightful_negative_gate_mean=torch.where(
                negative_count > 0,
                totals["delightful_negative_gate_sum"]
                / negative_count.clamp_min(1),
                torch.zeros_like(negative_count),
            ),
            delightful_delight_mean=(
                totals["delightful_delight_sum"] / action_denom
            ),
            delightful_surprisal_mean=(
                totals["delightful_surprisal_sum"] / action_denom
            ),
            delightful_positive_loss_contribution=(
                totals["delightful_positive_loss_sum"] / action_denom
            ),
            delightful_negative_loss_contribution=(
                totals["delightful_negative_loss_sum"] / action_denom
            ),
            delightful_positive_count=positive_count,
            delightful_negative_count=negative_count,
        )
    return scalar_tensors_to_floats(metric_tensors)


@torch.no_grad()
def trust_region_token_logprobs(
    wrapper: LatentThoughtModel,
    batch: LatentRolloutBatch,
    replay_plan: ReplayPlan,
    *,
    shard_stride: int,
    replay_function: Callable = replay_head_inputs,
    emit_logprob_function: Callable = compact_emit_token_logprobs,
) -> Tensor:
    """Emitted-token log-probs on every ``shard_stride``-th shard, in fp32.

    The trust region's view of the policy. Under bf16 autocast any parameter
    change -- one weight element moved by 1/16 of its step -- re-rolls the
    model's rounding through the recurrence to a token KL of 2-3e-4 against
    the unmoved policy, while the systematic step at 12892's rates is about
    6e-5 (NOTES 2026-10-12). Autocast off, the matmuls run in the trainer's
    TF32, whose floor is about 3e-6, and KL is quadratic in the step. Both
    ends of a measured step must come from here: the update's bf16 behavior
    log-probs differ from these by the bf16 floor. Whole shards keep their
    planned shapes; the length-sorted plan makes a stride a sample across
    lengths. Returns the compact log-probs of the sampled shards, in plan
    order.
    """
    if shard_stride < 1:
        raise ValueError("shard stride must be positive")
    compact_logprobs = []
    with torch.autocast(batch.kind.device.type, enabled=False):
        for shard_index, (microbatch, shard) in enumerate(
            iter_planned_replay_microbatches(batch, replay_plan)
        ):
            if shard_index % shard_stride:
                continue
            beliefs, stream_inputs = replay_function(wrapper, microbatch)
            emit_index = shard.emit_index
            compact_logprobs.append(
                emit_logprob_function(
                    wrapper,
                    compact_slots(stream_inputs, emit_index),
                    compact_slots(beliefs, emit_index),
                    compact_next_slots(microbatch.token_ids, emit_index),
                ).float()
            )
            del beliefs, stream_inputs
    return torch.cat(compact_logprobs)


@torch.no_grad()
def step_token_kl(reference: Tensor, current: Tensor) -> dict[str, float]:
    """The k3 token KL(reference || current) of two trust-region readouts."""
    if reference.shape != current.shape:
        raise ValueError("step KL compares readouts of the same tokens")
    log_ratio = current - reference
    return scalar_tensors_to_floats(
        {
            "kl/post_update_token_pre_step": (
                torch.expm1(log_ratio) - log_ratio
            ).mean(),
            "ratio/post_update_token_pre_step_abs_log_max": (
                log_ratio.abs().max()
            ),
            "trust/measured_tokens": log_ratio.new_tensor(log_ratio.numel()),
        }
    )


@torch.no_grad()
def measure_post_update_policy_drift(
    wrapper: LatentThoughtModel,
    batches: list[LatentRolloutBatch],
    *,
    replay_max_trajectories: int,
    replay_attention_budget: int,
    replay_bucket: int,
    replay_slot_budget: int | None = None,
    replay_function: Callable = replay_head_inputs,
    emit_logprob_function: Callable = compact_emit_token_logprobs,
    emit_log_odds_function: Callable = compact_emit_token_log_odds,
    replay_plans: list[ReplayPlan] | None = None,
    target_policy_optimization: bool = False,
    tpo_eta: float = 2.0,
    gae_lambda_alpha: float = 0.05,
    actor_gae_lambda: float | None = None,
    source_success_actor_gate: bool = False,
) -> dict[str, float]:
    """Evaluate the just-updated policy on its behavior trajectories.

    On behavior-age 0, optimized ratios are exactly one before the first actor
    step; clipping therefore cannot reveal how far that step moved the deployed
    policy. This read-only replay measures the actual post-step drift
    without reusing trajectories for a gradient. It runs only at the explicit
    diagnostic cadence because it costs one additional policy forward.
    The trainer passes an eager ``replay_function`` and a separately compiled
    no-grad ``emit_logprob_function``: enabling gradients here would retain
    a full replay graph and can exceed peak memory. Both are parameters
    rather than module lookups precisely because the trainer rebinds those
    globals to its grad-enabled training artifacts.

    The KL to the behavior log-probs mixes the step's drift with bf16
    rounding, which any parameter change re-rolls to a token KL of 2-3e-4;
    ``trust_region_token_logprobs`` measures the step alone.
    """
    if not batches:
        raise ValueError("post-update drift requires at least one rollout batch")
    if replay_plans is not None and len(replay_plans) != len(batches):
        raise ValueError("post-update drift plan count must match batches")
    device = next(wrapper.parameters()).device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    totals = {
        "token_kl": zero.clone(),
        "token_abs_log_ratio_max": zero.clone(),
        "action_count": zero.clone(),
        "tpo_active_count": zero.clone(),
        "tpo_fit_kl": zero.clone(),
        "tpo_probability_residual": zero.clone(),
        "tpo_probability_residual_abs": zero.clone(),
        "tpo_probability_residual_square": zero.clone(),
    }
    for batch_index, stored_batch in enumerate(batches):
        # Actor behavior pools live in ordinary CPU memory so a 64-prompt
        # frozen-policy pool does not retain many GiB of dense fp32 carried
        # hiddens on the GPU. Stream one prompt group at a time for this
        # infrequent diagnostic, exactly as the gradient-bearing update path
        # does below.
        batch = (
            stored_batch
            if stored_batch.kind.device == device
            else stored_batch.to(device)
        )
        replay_plan = (
            replay_plans[batch_index]
            if replay_plans is not None
            else build_replay_plan(
                stored_batch,
                replay_max_trajectories,
                replay_attention_budget,
                replay_bucket,
                replay_slot_budget,
            ).to(device)
        )
        replay_plan.validate_settings(
            replay_max_trajectories,
            replay_attention_budget,
            replay_bucket,
            replay_slot_budget,
            require_action_indices=True,
        )
        if target_policy_optimization:
            action_counts = batch.action_mask.sum(1)
            advantages, _ = generalized_advantage_and_return_targets(
                batch.rewards,
                batch.old_values,
                batch.action_mask,
                actor_gae_lambdas(action_counts, gae_lambda_alpha, actor_gae_lambda),
            )
            actor_signal_rows = source_actor_signal_mask(
                batch.source_id,
                batch.reward_scalar,
                enabled=source_success_actor_gate,
            )
            actor_active_mask = (
                batch.action_mask * actor_signal_rows[:, None]
            )
        for microbatch, shard in iter_planned_replay_microbatches(
            batch, replay_plan
        ):
            beliefs, stream_inputs = replay_function(
                wrapper, microbatch
            )
            token_logprobs = torch.zeros_like(microbatch.old_token_logprobs).float()
            if target_policy_optimization:
                token_log_odds = torch.zeros_like(
                    microbatch.old_token_log_odds
                ).float()
            if shard.emit_index.numel():
                emit_index = shard.emit_index
                emit_inputs = compact_slots(stream_inputs, emit_index)
                emit_beliefs = compact_slots(beliefs, emit_index)
                emit_targets = compact_next_slots(
                    microbatch.token_ids, emit_index
                )
                if target_policy_optimization:
                    compact_token_log_odds = emit_log_odds_function(
                        wrapper, emit_inputs, emit_beliefs, emit_targets
                    )
                    compact_token_logprobs = F.logsigmoid(
                        compact_token_log_odds
                    )
                    scatter_slots(
                        token_log_odds, emit_index, compact_token_log_odds
                    )
                else:
                    compact_token_logprobs = emit_logprob_function(
                        wrapper, emit_inputs, emit_beliefs, emit_targets
                    )
                scatter_slots(token_logprobs, emit_index, compact_token_logprobs)
            token_log_ratio = (
                token_logprobs - microbatch.old_token_logprobs.float()
            )
            action_mask = microbatch.action_mask.float()
            totals["token_abs_log_ratio_max"] = torch.maximum(
                totals["token_abs_log_ratio_max"],
                (token_log_ratio * action_mask).abs().max(),
            )
            totals["token_kl"] += (
                (torch.expm1(token_log_ratio) - token_log_ratio)
                * action_mask
            ).sum()
            totals["action_count"] += action_mask.sum()
            if target_policy_optimization:
                rows = shard.rows
                stream_length = shard.stream_length
                _, tpo = target_policy_loss(
                    token_log_odds,
                    microbatch.old_token_log_odds,
                    advantages[rows, :stream_length],
                    actor_active_mask[rows, :stream_length],
                    eta=tpo_eta,
                )
                totals["tpo_active_count"] += tpo["active_count"]
                totals["tpo_fit_kl"] += tpo["fit_kl_sum"]
                totals["tpo_probability_residual"] += tpo["residual_sum"]
                totals["tpo_probability_residual_abs"] += tpo[
                    "residual_abs_sum"
                ]
                totals["tpo_probability_residual_square"] += tpo[
                    "residual_square_sum"
                ]
        del batch

    diagnostics = {
        "kl/post_update_token_behavior": (
            totals["token_kl"] / totals["action_count"].clamp_min(1)
        ),
        "ratio/post_update_token_abs_log_max": totals[
            "token_abs_log_ratio_max"
        ],
    }
    if target_policy_optimization:
        tpo_denom = totals["tpo_active_count"].clamp_min(1)
        diagnostics.update(
            {
                "tpo/post_update_fit_kl": totals["tpo_fit_kl"] / tpo_denom,
                "tpo/post_update_probability_residual_mean": (
                    totals["tpo_probability_residual"] / tpo_denom
                ),
                "tpo/post_update_probability_residual_abs_mean": (
                    totals["tpo_probability_residual_abs"] / tpo_denom
                ),
                "tpo/post_update_probability_residual_rms": (
                    totals["tpo_probability_residual_square"] / tpo_denom
                ).sqrt(),
            }
        )
    return scalar_tensors_to_floats(diagnostics)


def save_checkpoint(
    path: Path,
    wrapper: LatentThoughtModel,
    critic: SeparateCritic,
    optimizers: dict[str, torch.optim.Optimizer],
    step: int,
    args: argparse.Namespace,
    sampler: MathPromptSampler | MixedPromptSampler,
    warmup_step: int,
    actor_init_provenance: dict | None = None,
    zero_reward_frozen_updates: int = 0,
    entropy_temperature: EntropyTemperature | None = None,
) -> None:
    reasoning_mode = getattr(args, "reasoning_mode", "latent")
    # The schema tags below are derived from the mode; the parameters come
    # from the wrapper. A mismatch would label one policy with another's
    # contract, and `combiner_init_kwargs_from_checkpoint` would rebuild it
    # with the wrong heads.
    if (wrapper.hidden_carry, wrapper.memory_carry) != (
        mode_carries_hidden(reasoning_mode),
        mode_carries_memory(reasoning_mode),
    ):
        raise ValueError(
            f"reasoning mode {reasoning_mode!r} does not match a wrapper with "
            f"hidden_carry={wrapper.hidden_carry}, "
            f"memory_carry={wrapper.memory_carry}"
        )
    payload = {
        "step": step,
        "value_warmup_step": warmup_step,
        "execution_schema": execution_schema_for(
            reasoning_mode, getattr(args, "rollout_scheduler", "lockstep")
        ),
        "actor_objective_schema": run_actor_objective_schema(args),
        "resume_arg_contract_schema": RESUME_ARG_CONTRACT_SCHEMA,
        "critic_schema": critic_run_schema(critic),
        "critic_action_head_schema": critic_action_head_schema(critic),
        "source_signal_mask_schema": (
            SOURCE_SIGNAL_MASK_SCHEMA
            if (
                getattr(args, "rl_mixture_manifest", None)
                and getattr(args, "source_success_actor_gate", False)
            )
            else None
        ),
        "python_reward_schema": getattr(args, "python_reward_schema", None),
        "replay_numerics_schema": replay_numerics_schema_for(reasoning_mode),
        "source_provenance": getattr(args, "source_provenance", None),
        "prompt_order_schema": PROMPT_ORDER_SCHEMA,
        "answer_fence_prompt_schema": (
            ANSWER_FENCE_PROMPT_SCHEMA
            if getattr(args, "answer_fence", False)
            else None
        ),
        "math_data_identity": sampler.dataset_identity,
        "base_checkpoint_sha256": getattr(
            args, "base_checkpoint_sha256", None
        ),
        "initialization_checkpoint_sha256": getattr(
            args, "initialization_checkpoint_sha256", None
        ),
        "reward_schema": reward_schema_for_abstention(
            getattr(args, "abstention_penalty", None)
        ),
        "reasoning_mode": reasoning_mode,
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "rollout_policy_schema": rollout_policy_schema_for_mode(reasoning_mode),
        "thought_input_schema": wrapper.thought_input_schema,
        "thought_distribution_schema": wrapper.thought_distribution_schema,
        "optimizer_schema": optimizer_schema_for_trunk_optimizer(
            getattr(args, "trunk_optimizer", "adamw"),
            fresh_muon=getattr(args, "fresh_learning_rate", None) is not None,
        ),
        "model": wrapper.state_dict(),
        "critic": critic.state_dict(),
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "args": vars(args),
        "sampler_cursor": sampler.cursor,
        "zero_reward_frozen_updates": int(zero_reward_frozen_updates),
        "entropy_temperature_state": (
            None
            if entropy_temperature is None
            else entropy_temperature.state_dict()
        ),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
        "actor_init_provenance": actor_init_provenance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def logged_zero_reward_frozen_updates(
    metrics_path: str | Path, through_step: int
) -> int:
    """Recover the cumulative freeze count from retained per-step telemetry.

    Older checkpoints predate the persisted counter. Keeping the last record
    for each actor step also makes this robust to a previously interrupted
    append followed by an exact resume.
    """
    path = Path(metrics_path)
    if not path.is_file():
        return 0
    frozen_by_step: dict[int, bool] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") != "train" or "step" not in record:
                continue
            step = int(record["step"])
            if step > through_step:
                continue
            dashboard = record.get("dashboard") or {}
            if "guard/zero_reward_actor_frozen" in dashboard:
                frozen_by_step[step] = bool(
                    dashboard["guard/zero_reward_actor_frozen"]
                )
    return sum(frozen_by_step.values())


def purge_benchmark_reports_after(output: Path, step: int) -> int:
    """Remove stale answer histories and restore latest to checkpoint time."""
    history_dir = output / "bench_answers"
    removed = 0
    if history_dir.exists():
        for path in history_dir.glob("step_*.json"):
            try:
                history_step = int(path.stem.removeprefix("step_"))
            except ValueError:
                continue
            if history_step > step:
                path.unlink()
                removed += 1
    remaining = sorted(history_dir.glob("step_*.json")) if history_dir.exists() else []
    if remaining:
        payload = json.loads(remaining[-1].read_text())
        write_benchmark_report(
            output,
            int(payload["step"]),
            payload["metrics"],
            payload["attempts"],
            reward_schema=str(payload.get("reward_schema", REWARD_SCHEMA)),
            # Keep the source payload's own schema tag: restamping a
            # pre-segments history as the current version would break
            # "answers/v4 implies segments" for artifacts on disk.
            schema=str(payload["schema"]) if payload.get("schema") else None,
        )
    else:
        for path in (history_dir / "latest.json", output / "bench_answers.html"):
            if path.exists():
                path.unlink()
    return removed


def main() -> None:
    # Read-only live stack diagnostics without ptrace privileges or stopping RL.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    run_started = time.monotonic()
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    args.base_checkpoint_sha256 = file_sha256(args.checkpoint)
    args.resume_checkpoint_sha256 = (
        file_sha256(args.resume) if args.resume else None
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    checkpoint_payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    backbone = load_model(args.checkpoint, device, payload=checkpoint_payload)
    checkpoint_trained_seq_len = (
        checkpoint_payload.get("trained_seq_len")
        if isinstance(checkpoint_payload, dict) else None
    )
    sft_provenance = (
        (checkpoint_payload.get("sft") or {})
        if isinstance(checkpoint_payload, dict)
        else {}
    )
    del checkpoint_payload  # drop the CPU weight copy; keep only sft metadata
    sft_thinks = bool((sft_provenance.get("args") or {}).get("think_tokens"))
    sft_answers = bool((sft_provenance.get("args") or {}).get("answer_fence"))
    # Fail before any GPU/optimizer construction: both mismatches produce
    # runs that LOOK healthy. An untrained fence means every format-gated
    # reward is zero; a think-SFT base without the flag silently drops the
    # fence ids in decode and trains with no gate at all.
    if args.think_tokens and not sft_thinks:
        raise RuntimeError(
            "--think-tokens requires a base checkpoint whose SFT stage "
            "trained the <think>/</think> fence rows (sft.args."
            "think_tokens); this checkpoint's fence rows are untrained, "
            "so every format-gated reward would be zero"
        )
    if sft_thinks and not args.think_tokens:
        print(
            "WARNING: base checkpoint was SFT-trained with think fences "
            "but --think-tokens is off — the policy will emit fence ids "
            "the tokenizer silently drops, and no format gate applies. "
            "Pass --think-tokens unless this is a deliberate ablation.",
            flush=True,
        )
    if args.answer_fence and not sft_answers:
        raise RuntimeError(
            "--answer-fence requires a base checkpoint whose SFT stage "
            "trained the <answer>/</answer> fence rows (sft.args."
            "answer_fence); this checkpoint's fence rows are untrained, "
            "so every structurally-gated reward would be zero"
        )
    if args.answer_fence:
        # The flag alone is an operator claim; the stored fraction is a
        # measurement over the SFT corpus's token ids (red-team round 2:
        # an --answer-fence SFT run over a fence-less corpus would stamp
        # the flag while the anchored shape was never a CE target).
        fence_fraction = sft_provenance.get("answer_fence_document_fraction")
        if fence_fraction is None or float(fence_fraction) < 0.99:
            raise RuntimeError(
                "--answer-fence requires SFT provenance measuring >=99% "
                "anchored-fence documents (answer_fence_document_fraction "
                f"= {fence_fraction!r}); this checkpoint cannot have "
                "learned the structural gate's shape"
            )
        prompt_schema = sft_provenance.get("answer_fence_prompt_schema")
        if prompt_schema != ANSWER_FENCE_PROMPT_SCHEMA:
            raise RuntimeError(
                "--answer-fence requires an SFT checkpoint trained with "
                f"prompt schema {ANSWER_FENCE_PROMPT_SCHEMA!r}; got "
                f"{prompt_schema!r}. Regenerate the canonical SFT corpus "
                "and retrain SFT before starting RL."
            )
    if sft_answers and not args.answer_fence:
        print(
            "WARNING: base checkpoint was SFT-trained with answer fences "
            "but --answer-fence is off — the policy will emit fence ids "
            "the tokenizer silently drops, and the decoded-text gate "
            "will grade text the policy framed for the structural gate. "
            "Pass --answer-fence unless this is a deliberate ablation.",
            flush=True,
        )
    check_think_floor_against_corpus(sft_provenance, args.think_min_tokens)
    is_nano = backbone.architecture.startswith("nanogpt_mini")
    if args.mean_actor_critic:
        # MAC renders the policy through nano's raw readout and softcap.
        if not is_nano:
            raise ValueError("--mean-actor-critic needs the nano rational-softcap readout")
        validate_softcap(backbone.logit_softcap)
    if not is_nano and not backbone.architecture.endswith(
        "probes_pope_belief_attached_ce_onepass_2k"
    ):
        raise ValueError(
            "belief-renderer VAPO requires a nanogpt_mini checkpoint or one "
            "pretrained with [token_latent, raw_belief] CE; old "
            "predicted-latent probe "
            f"architecture {backbone.architecture!r} is incompatible"
        )
    # KDA mixers wrap FLA's chunk kernel, which deliberately graph-breaks
    # under torch.compile: the teacher-forced replay/valuation surfaces can
    # still be compiled, but not under fullgraph=True. The decode step stays
    # a pure-PyTorch recurrence, so the rollout step artifacts keep fullgraph
    # — including paged_step_core, whose recurrent layers are lane-indexed
    # gather/step/scatter (see kda_backbone._block_paged_step).
    # The memory carry's replay scan is the same fenced FLA kernel, on any
    # trunk.
    is_recurrent_trunk = "_kda_" in backbone.architecture
    trunk_fullgraph = not (
        is_recurrent_trunk or mode_carries_memory(args.reasoning_mode)
    )

    # Runtime rollout capacity follows the requested response budget; record
    # checkpoint/training windows separately so extension is never mislabeled.
    checkpoint_context_limit = (
        getattr(backbone, "train_context_tokens", 1024)
        if is_nano else POSTTRAIN_CONTEXT_TOKENS
    )
    trained_context_tokens = (
        int(checkpoint_trained_seq_len or checkpoint_context_limit) if is_nano else None
    )
    try:
        context_tokens = resolve_rollout_defaults(
            args, is_nano=is_nano, checkpoint_context_tokens=checkpoint_context_limit,
            trained_context_tokens=trained_context_tokens,
        )
    except ValueError as error:
        parser.error(str(error))
    context_extrapolation = (
        trained_context_tokens is not None and context_tokens > trained_context_tokens
    )
    if context_extrapolation:
        print(f"Rollout context: {context_tokens} tokens; checkpoint trained window: "
              f"{trained_context_tokens} tokens (context extension)", flush=True)
    if args.answer_tokens < 2:
        parser.error("--answer-tokens must fit an answer plus its terminator")

    pin_emit = mode_pins_emit(args.reasoning_mode)
    hidden_carry = mode_carries_hidden(args.reasoning_mode)
    rollout_policy_schema = rollout_policy_schema_for_mode(args.reasoning_mode)

    def mode_budgets(max_tokens: int) -> tuple[int, int]:
        return mode_rollout_budget(
            args.reasoning_mode,
            max_tokens,
            answer_tokens=args.answer_tokens,
            prompt_tokens=args.prompt_tokens,
            context_tokens=context_tokens,
        )

    train_max_new_tokens, max_stream_steps = training_rollout_budget(
        args.reasoning_mode,
        args.continuation_tokens,
        answer_tokens=args.answer_tokens,
        prompt_tokens=args.prompt_tokens,
        context_tokens=context_tokens,
    )
    aime_max_new_tokens, aime_stream_steps = mode_budgets(args.aime_max_tokens)
    bench_max_new_tokens, bench_stream_steps = mode_budgets(
        args.bench_max_tokens
    )
    # Persist effective values as well as the user's raw override. Readers
    # should not have to reconstruct backbone-dependent defaults from a later
    # checkout merely to reproduce a checkpoint's rollout policy.
    args.resolved_train_max_new_tokens = train_max_new_tokens
    args.resolved_train_max_stream_steps = max_stream_steps
    if args.think_tokens:
        # The gated reward needs the floor, both think fences, room for
        # the answer (an Answer: line, or the fenced value under
        # --answer-fence, budgeted by --answer-tokens as a conservative
        # reserve), and a stop token INSIDE the training budget; a floor
        # that leaves no such room zeroes every reward while the
        # truncation early-return keeps think_gate_zeroed_correct at a
        # healthy-looking 0 (rows die before the gate is consulted).
        # The answer fence adds two more structural tokens.
        floor_overhead = (
            2 + args.answer_tokens + (2 if args.answer_fence else 0)
        )
        if args.think_min_tokens + floor_overhead > train_max_new_tokens:
            parser.error(
                f"--think-min-tokens {args.think_min_tokens} plus fence "
                f"and answer overhead ({floor_overhead}) exceeds the "
                f"training rollout budget ({train_max_new_tokens} new "
                "tokens): every reward would be zero by truncation"
            )
    args.resolved_aime_max_new_tokens = aime_max_new_tokens
    args.resolved_aime_max_stream_steps = aime_stream_steps
    args.resolved_bench_max_new_tokens = bench_max_new_tokens
    args.resolved_bench_max_stream_steps = bench_stream_steps
    validate_posttraining_context_budget(
        args.prompt_tokens, max_stream_steps, context_tokens
    )
    validate_posttraining_context_budget(
        args.prompt_tokens, aime_stream_steps, context_tokens
    )
    validate_posttraining_context_budget(
        args.prompt_tokens, bench_stream_steps, context_tokens
    )

    backbone.eval()
    # Full-model RL: trunk, renderer, combiner, Gaussian mean, and one-way
    # continuation gate all train. Noise scale is a strict policy hyperparameter.
    # The hidden-carry policy builds only the combiner: it has no Gaussian or
    # gate to train, and its sigma/stop arguments are None.
    memory_carry = mode_carries_memory(args.reasoning_mode)
    combined_mlp = (
        {}
        if memory_carry
        else {
            "mlp_hidden": args.combined_mlp_hidden,
            "num_blocks": args.combined_mlp_blocks,
        }
    )
    wrapper = LatentThoughtModel(
        backbone,
        **combined_mlp,
        thought_sigma=args.thought_sigma,
        init_stop_thinking_probability=(
            args.init_stop_thinking_probability
        ),
        hidden_carry=hidden_carry,
        memory_carry=memory_carry,
        memory_heads=args.memory_heads,
        memory_head_dim=args.memory_head_dim,
    ).to(device)
    # No module here behaves differently under train(): pin eval mode once so
    # the training flag (a dynamo guard) never flips between the step-0 evals
    # and the training loop and re-specializes the compiled step.
    wrapper.eval()
    actor_init_payload = None
    actor_init_provenance = None
    initialization_path = args.actor_init or args.curriculum_init
    carry_critic = carries_init_critic(args)
    args.initialization_checkpoint_sha256 = (
        file_sha256(initialization_path) if initialization_path else None
    )
    if initialization_path:
        actor_init_payload = torch.load(
            initialization_path, map_location="cpu", weights_only=False
        )
        if actor_init_payload.get("prompt_order_schema") != PROMPT_ORDER_SCHEMA:
            raise ValueError(
                "initialization checkpoint predates deterministic sequential "
                "prompt traversal; its cursor cannot prove no prompt reuse"
            )
        if (
            args.answer_fence
            and actor_init_payload.get("answer_fence_prompt_schema")
            != ANSWER_FENCE_PROMPT_SCHEMA
        ):
            raise ValueError(
                "initialization checkpoint uses a different answer-fence "
                "prompt schema; start from a checkpoint trained with "
                f"{ANSWER_FENCE_PROMPT_SCHEMA!r}"
            )
        if (
            args.actor_init
            and actor_init_payload.get("base_checkpoint_sha256")
            != args.base_checkpoint_sha256
        ):
            # A branch continues its source's lineage; the manifest's base
            # block must name the checkpoint the branch actually descends from.
            raise ValueError(
                "--actor-init checkpoint descends from a different base "
                "checkpoint than --checkpoint"
            )
        if carry_critic:
            args.carried_critic_value_changes = validate_carried_critic(
                actor_init_payload, args
            )
        validate_combiner_geometry(
            actor_init_payload, args, wrapper.memory_carry, initialization_path
        )
        validate_renderer_checkpoint(
            actor_init_payload,
            initialization_path,
            expected_rollout_policy_schema=rollout_policy_schema,
        )
        wrapper.load_state_dict(actor_init_payload["model"], strict=True)
        actor_init_provenance = {
            "checkpoint": str(initialization_path),
            "checkpoint_sha256": args.initialization_checkpoint_sha256,
            "critic_initialized": carry_critic,
            "curriculum_transition": bool(args.curriculum_init),
            "source_step": actor_init_payload.get("step"),
            "source_reward_schema": actor_init_payload.get("reward_schema"),
            "source_execution_schema": actor_init_payload.get("execution_schema"),
            "source_actor_objective_schema": actor_init_payload.get(
                "actor_objective_schema"
            ),
            "source_value_warmup_step": actor_init_payload.get(
                "value_warmup_step"
            ),
            "source_sampler_cursor": int(actor_init_payload["sampler_cursor"]),
            "target_sampler_cursor": 0 if args.curriculum_init else int(
                actor_init_payload["sampler_cursor"]
            ),
            "combined_mlp_hidden": args.combined_mlp_hidden,
            "combined_mlp_blocks": args.combined_mlp_blocks,
        }
    frozen_moe_router_parameters = make_actor_trainable(wrapper)
    if frozen_moe_router_parameters:
        # This trainer deliberately stays in eval mode to keep one compiled
        # graph family. Collection is enabled for those eval-mode replay
        # forwards; each optimizer minibatch resets away rollout/refresh
        # statistics before its teacher-forced update.
        enable_quantile_balance_collection(
            backbone, num_bins=1000, collect_in_eval=True
        )

    critic_uses_dense_trunk = bool(
        getattr(backbone, "model_config", {}).get("moe_num_experts", 0)
    )
    # The critic's random initialization draws from a stream of its own,
    # seeded by --seed alone, not from whatever the actor wrapper consumed
    # before it (cot, carry and memory construct different heads): one
    # seed starts every reasoning mode from the same critic, so a mode
    # comparison is paired rather than confounded by critic init.
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(args.seed + CRITIC_INIT_SEED_OFFSET)
        if args.critic_init == "actor":
            if critic_uses_dense_trunk:
                raise ValueError(
                    "--critic-init actor copies the actor trunk, but a LatentMoE "
                    "actor's critic is a dense trunk with no matching weights"
                )
            # The actor as loaded at this point (base checkpoint, or the
            # --actor-init/--curriculum-init weights), in storage of its own:
            # pretrained features without sharing a single tensor. The critic
            # head stays zero-initialized, so the copy starts at value 0 and gets
            # no trunk gradient until the head has learned a readout.
            critic_trunk = copy_trunk(backbone, device)
        else:
            critic_trunk = fresh_trunk(
                backbone,
                device,
                model_config_overrides=(
                    {"moe_num_experts": 0, "moe_layer_indices": []}
                    if critic_uses_dense_trunk
                    else None
                ),
                architecture_override=(
                    backbone.architecture.replace("_stable_latentmoe_v1", "")
                    + "_dense_critic"
                    if critic_uses_dense_trunk
                    else None
                ),
            )
        critic = SeparateCritic(
            critic_trunk,
            mlp_hidden=args.combined_mlp_hidden,
            num_blocks=args.combined_mlp_blocks,
            actor_memory=(wrapper.combiner
                          if wrapper.memory_carry and not uses_post_action_q(args)
                          else None),
            action_head_hidden=args.critic_action_head_hidden,
            q_head_hidden=args.critic_q_head_hidden,
            post_action_q_hidden=args.critic_post_action_q_hidden,
            post_action_q_prior=args.critic_post_action_q_prior,
            q_head_reads_actor_belief=(
                args.critic_q_head_hidden is not None and wrapper.hidden_carry
            ),
            q_twin=args.critic_q_twin,
        ).to(device)
    critic.eval()  # no dropout in this architecture; keep norms deterministic
    if actor_init_payload is not None:
        # These three apply on EVERY initialization path (--actor-init and
        # --curriculum-init load the same actor weights): a fence-setting
        # mismatch means the loaded policy either never trained the fence
        # rows (all-zero structurally-gated reward) or emits fence ids the
        # tokenizer silently drops (no gate at all). Red-team round 2: the
        # guards once covered only the retired warm-critic init, whose own
        # error text recommended --actor-init — then the unguarded path.
        init_args = actor_init_payload.get("args", {})
        if bool(init_args.get("think_tokens")) != bool(args.think_tokens):
            raise ValueError(
                "the initialization checkpoint's policy was trained under "
                "a different --think-tokens setting (checkpoint "
                f"{bool(init_args.get('think_tokens'))}, got "
                f"{bool(args.think_tokens)}); the fence gate and the "
                "policy's emission format must agree"
            )
        # The floor moves the same reward distribution (default 1 in old
        # manifests = the pre-floor gate) — otherwise forgetting the flag
        # silently drops the floor to 1.
        if int(init_args.get("think_min_tokens", 1)) != args.think_min_tokens:
            raise ValueError(
                "the initialization checkpoint's policy was trained under "
                "a different --think-min-tokens floor (checkpoint "
                f"{int(init_args.get('think_min_tokens', 1))}, got "
                f"{args.think_min_tokens})"
            )
        # The structural gate is a different reward function from the
        # decoded-text gate (default False in old manifests).
        if bool(init_args.get("answer_fence")) != bool(args.answer_fence):
            raise ValueError(
                "the initialization checkpoint's policy was trained under "
                "a different --answer-fence setting (checkpoint "
                f"{bool(init_args.get('answer_fence'))}, got "
                f"{bool(args.answer_fence)})"
            )
        if (
            args.fresh_learning_rate is not None
            and init_args.get("fresh_learning_rate") != args.fresh_learning_rate
        ):
            # The loaded fresh modules are trained, so a from-scratch rate
            # would restart them -- unless it is the rate they trained at.
            raise ValueError(
                "--fresh-learning-rate trains modules from scratch; the "
                "initialization checkpoint's were trained at "
                f"--fresh-learning-rate {init_args.get('fresh_learning_rate')}"
            )
    optimizers = build_optimizers(
        wrapper,
        critic,
        learning_rate=args.learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        trunk_optimizer=args.trunk_optimizer,
        muon_learning_rate=args.muon_learning_rate,
        critic_muon_learning_rate=args.critic_muon_learning_rate,
        critic_action_learning_rate=args.critic_action_head_learning_rate,
        critic_q_direction_learning_rate=args.critic_q_direction_learning_rate,
        fresh_learning_rate=args.fresh_learning_rate,
        fresh_muon_learning_rate=args.fresh_muon_learning_rate,
    )
    if carry_critic:
        carry_init_critic(critic, optimizers, actor_init_payload, args)
    if args.perturbation_gradient_direction:
        # The sweep never steps: the snapshot's own critic is only the
        # sampled-baseline estimator's state value for the policy it fit.
        if actor_init_payload.get("critic_schema") != critic_run_schema(critic):
            raise ValueError(
                f"--curriculum-init critic schema {actor_init_payload.get('critic_schema')!r} "
                f"is not this run's {critic_run_schema(critic)!r}; it cannot be the "
                "sweep's baseline"
            )
        critic.load_state_dict(actor_init_payload["critic"], strict=True)
        print(f"perturbation sweep: baseline critic from {initialization_path} "
              f"(step {actor_init_payload.get('step')})", flush=True)
    # Every parameter an actor step moves, for the KL trust region.
    actor_step_parameters = [
        parameter
        for name, optimizer in optimizers.items()
        if name == "actor" or name.startswith("actor_")
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    entropy_temperature = (
        None
        if args.entropy_target is None
        else EntropyTemperature(
            args.entropy_target,
            args.entropy_temperature_init,
            args.entropy_temperature_learning_rate,
        )
    )

    tokenizer = load_posttraining_tokenizer(
        backbone.architecture,
        FreshHyperparameters.tokenizer_path,
        think_tokens=args.think_tokens,
        answer_tokens=args.answer_fence,
        tokenizer_provenance=backbone.model_config.get("tokenizer_provenance"),
    )
    think_fence_ids: tuple[int, int] | None = None
    answer_fence_ids: tuple[int, int] | None = None
    if args.think_tokens:
        # Provenance already verified at checkpoint load.
        think_fence_ids = (tokenizer.think_open_id, tokenizer.think_close_id)
    if args.answer_fence:
        answer_fence_ids = (
            tokenizer.answer_open_id, tokenizer.answer_close_id
        )
    # The terminal reward of a zero verifier score.
    reward_floor = abstention_terminal_scores([0.0], [False], args.abstention_penalty)[0]
    # dict.fromkeys dedupes while keeping order: GPT-2's single
    # <|endoftext|> token reports as both EOS and BOS.
    stop_ids = tuple(
        dict.fromkeys(
            t for t in (tokenizer.eos_id(), tokenizer.bos_id()) if t >= 0
        )
    )
    if not stop_ids:
        raise RuntimeError(
            "posttraining requires a valid BOS or EOS token for explicit "
            "trajectory termination"
        )
    # The token classes the actor's advantages are broken down over: the
    # option label that opens an answer span is where a multiple-choice prior
    # would leak into other sources' rows, and fences and stops carry the
    # format.
    def token_id_tensor(ids):
        return torch.tensor(sorted(ids), dtype=torch.long, device=device)

    label_ids = {
        encoded[0]
        for encoded in (tokenizer.encode(label) for label in RENDERED_LABELS)
        if len(encoded) == 1
    }
    fence_ids = {
        token
        for fence in (think_fence_ids, answer_fence_ids)
        if fence is not None
        for token in fence
    }
    advantage_token_classes = {"stop": (token_id_tensor(stop_ids), None)}
    if fence_ids:
        advantage_token_classes["fence"] = (token_id_tensor(fence_ids), None)
    if answer_fence_ids is not None:
        advantage_token_classes["answer_label"] = (
            token_id_tensor(label_ids),
            token_id_tensor([answer_fence_ids[0]]),
        )
    answer_prefix_ids: tuple[int, ...] = ()
    if args.reasoning_mode == "none":
        # Mathmix QA documents close with "\nAnswer: <x>" before EOS.
        # Teacher-forcing the prefix onto the prompt leaves the policy only
        # the answer value and its terminator to emit; scoring and decoding
        # prepend the same ids before parsing.
        answer_prefix_ids = answer_prefix_token_ids(tokenizer)
        if not answer_prefix_ids:
            raise RuntimeError(
                "the none-mode Answer: prefix encoded to zero tokens"
            )
        if len(answer_prefix_ids) >= args.prompt_tokens:
            raise RuntimeError(
                "the encoded Answer: prefix must leave room in the prompt "
                "budget"
            )
    aime_rows = (
        load_unique_math_rows(args.aime_data)
        if args.aime_every > 0 and not args.rollout_only
        else []
    )
    if args.answer_fence:
        aime_rows = rewrite_prompts_for_answer_fence(aime_rows)
    aime_modal_baseline = modal_answer_baseline(aime_rows)
    all_bench_rows = (
        load_unique_math_rows(args.bench_data)
        if (args.bench_every > 0 or args.bench_only) and not args.rollout_only
        else []
    )
    if args.answer_fence:
        all_bench_rows = rewrite_prompts_for_answer_fence(all_bench_rows)
    bench_rows = deterministic_math_subset(all_bench_rows, args.bench_max_rows)
    bench_dataset_baseline = modal_answer_baseline(all_bench_rows)
    bench_subset_baseline = modal_answer_baseline(bench_rows)
    mixture_manifest = None
    mixture_sources = None
    corpus_audit = {}
    if args.rl_mixture_manifest:
        math_rows, mixture_sources, mixture_manifest = load_mixture_manifest(
            args.rl_mixture_manifest
        )
        if (
            mixture_manifest.get("answer_fence_prompt_schema")
            != ANSWER_FENCE_PROMPT_SCHEMA
        ):
            raise ValueError("mixture uses a different answer-fence prompt schema")
        # Only a mixture that executes Python rewards is bound to the Python
        # verifier; a math-only mixture's rewards never touch it.
        args.python_reward_schema = mixture_python_reward_schema(args, mixture_sources)
        if (
            args.python_reward_schema is not None
            and mixture_manifest.get("python_reward_schema") != PYTHON_REWARD_SCHEMA
        ):
            raise ValueError("mixture uses a different Python reward schema")
        validate_actor_init_python_reward(
            actor_init_payload, args.python_reward_schema, carries_critic=carry_critic,
        )
        if (
            sft_provenance.get("traces_sha256")
            != mixture_manifest.get("sft_corpus_sha256")
        ):
            raise ValueError(
                "base SFT checkpoint trace bytes do not match the mixture's "
                "bound SFT corpus"
            )
        if any(
            source.verifier == "python_mbpp" for source in mixture_sources
        ) and not (args.think_tokens and args.answer_fence):
            raise ValueError(
                "Python rewards require --think-tokens and --answer-fence"
            )
    else:
        math_rows = load_unique_math_rows(args.math_data, audit=corpus_audit)
    if args.answer_fence:
        math_rows = rewrite_prompts_for_answer_fence(math_rows)
        if mixture_sources is not None:
            by_source: dict[str, list[dict]] = {
                source.name: [] for source in mixture_sources
            }
            for row in math_rows:
                by_source[row["_rl_source"]].append(row)
            mixture_sources = [
                replace(source, rows=tuple(by_source[source.name]))
                for source in mixture_sources
            ]
    excluded_modules = {
        name for name in args.exclude_modules.split(",") if name
    }
    if excluded_modules:
        present_modules = {
            (row.get("extra_info") or {}).get("module") for row in math_rows
        }
        missing = sorted(excluded_modules - present_modules)
        if missing:
            raise ValueError(
                f"--exclude-modules names absent from {args.math_data}: {missing}"
            )
        kept_rows = [
            row
            for row in math_rows
            if (row.get("extra_info") or {}).get("module") not in excluded_modules
        ]
        print(
            f"excluded modules {sorted(excluded_modules)}: "
            f"{len(math_rows)} -> {len(kept_rows)} RL rows",
            flush=True,
        )
        math_rows = kept_rows
    numeric_reward_rows = [
        row
        for row in math_rows
        if row.get("_verifier_kind", "math") == "math"
    ]
    math_modal_baseline = modal_answer_baseline(numeric_reward_rows)
    modal_solution = f"Answer: {math_modal_baseline['answer']}"
    modal_shaped_rewards = []
    for row in numeric_reward_rows:
        truth = str(row["reward_model"]["ground_truth"])
        correct, _ = verify_answer(
            modal_solution, truth, answer_style(row)
        )
        modal_shaped_rewards.append(
            1.0
            if correct
            else nearby_numeric_reward(
                str(math_modal_baseline["answer"]),
                truth,
                args.nearby_reward_max,
            )
        )
    # On the trained reward scale, beside reward/mean.
    modal_shaped_rewards = abstention_terminal_scores(
        modal_shaped_rewards,
        [False] * len(modal_shaped_rewards),
        args.abstention_penalty,
    )
    math_modal_baseline["shaped_reward"] = (
        sum(modal_shaped_rewards) / len(modal_shaped_rewards)
        if modal_shaped_rewards
        else 0.0
    )
    if mixture_manifest is None:
        data_identity = math_dataset_identity(
            args.math_data, args.exclude_modules, math_rows
        )
        sampler = MathPromptSampler(
            math_rows, args.seed, dataset_identity=data_identity
        )
    else:
        data_identity = mixture_identity(
            args.rl_mixture_manifest, mixture_manifest,
            randomize_choice_options=args.randomize_choice_options,
        )
        assert mixture_sources is not None
        sampler = MixedPromptSampler(
            mixture_sources,
            args.seed,
            dataset_identity=data_identity,
            randomize_choice_options=args.randomize_choice_options,
            excluded_sources=tuple(filter(None, args.exclude_rl_sources.split(","))),
        )
    if args.exclude_rl_sources:
        print(
            f"RL source exclusions: {args.exclude_rl_sources}; original manifest, "
            "source IDs, cursor coordinates and retained prompt RNG are preserved",
            flush=True,
        )
    if getattr(args, "critic_only_init", None):
        critic_source = Path(args.critic_only_init)
        if Path(args.output).exists():
            raise ValueError("--critic-only-init requires a new output directory")
        critic_payload = torch.load(critic_source, map_location="cpu", weights_only=False)
        validate_critic_only_init(critic_payload, args, data_identity)
        critic.load_state_dict(critic_payload["critic"], strict=True)
        args.critic_only_init_provenance = {
            "init": "critic_only_checkpoint",
            "checkpoint": str(critic_source),
            "checkpoint_sha256": file_sha256(critic_source),
            "source_execution_schema": critic_payload.get("execution_schema"),
            "source_actor_step": int(critic_payload.get("step", 0)),
            "source_value_warmup_step": int(critic_payload["value_warmup_step"]),
            "optimizer_state": "fresh",
        }
        del critic_payload
        print("Loaded critic-only weights; actor remains base SFT, with fresh "
              "optimizers, RNG, sampler, and actor step", flush=True)
    if getattr(args, "study_critic_init", None):
        study_source = Path(args.study_critic_init)
        study_payload = torch.load(study_source, map_location="cpu", weights_only=False)
        validate_study_critic_init(study_payload, critic, args, data_identity)
        load_study_critic(critic, study_payload["critic"])
        args.study_critic_init_provenance = {
            "init": "study_checkpoint",
            "checkpoint": str(study_source),
            "checkpoint_sha256": file_sha256(study_source),
            "source_execution_schema": STUDY_CRITIC_SCHEMA,
            "source_actor_step": 0,
            "source_value_warmup_step": int(study_payload["step"]),
            "optimizer_state": "fresh",
        }
        print(
            f"Loaded study critic ({study_payload['step']} updates, final EV "
            f"{study_payload['final_evaluation']['ev_all_actions']:.3f}) from "
            f"{study_source}; fresh critic optimizer state",
            flush=True,
        )
        del study_payload
    if actor_init_payload is not None:
        if args.curriculum_init:
            print(
                f"actor curriculum-initialized from {initialization_path} at "
                f"source step {actor_init_payload.get('step')}; fresh target "
                "critic, optimizers, step, and sampler cursor",
                flush=True,
            )
        else:
            if actor_init_payload.get("math_data_identity") != data_identity:
                raise ValueError(
                    "initialization checkpoint's prompt cursor belongs to different "
                    "dataset bytes, effective corpus policy/targets, exclusions, or ordering; "
                    "start a new run instead of reusing the cursor"
                )
            source_args = actor_init_payload.get("args", {})
            if hasattr(source_args, "__dict__"):
                source_args = vars(source_args)
            # The content hash above, not a cwd-relative filename, proves that
            # the cursor belongs to these exact dataset bytes. This also keeps
            # a checkpoint relocatable into a reproducible detached worktree.
            if int(source_args.get("seed", -1)) != args.seed:
                raise ValueError(
                    "initialization checkpoint must use the source seed so its "
                    "deterministic prompt order can continue without reuse"
                )
            source_exclusions = source_args.get("exclude_modules") or ""
            if source_exclusions != args.exclude_modules:
                raise ValueError(
                    "initialization checkpoint must use the same "
                    "--exclude-modules setting"
                )
            actor_cursor = int(actor_init_payload.get("sampler_cursor", -1))
            if actor_cursor < 0:
                raise ValueError(
                    "initialization source has no valid prompt cursor: "
                    f"{actor_cursor}"
                )
            sampler.cursor = actor_cursor
            print(
                f"actor{' and critic' if carry_critic else ''} initialized "
                f"from {initialization_path} at source step "
                f"{actor_init_payload.get('step')} and sampler cursor "
                f"{actor_cursor}; {args.value_warmup_steps} critic warmup "
                "updates before the first actor step",
                flush=True,
            )
    seq_len = FreshHyperparameters.train_seq_len
    tokenizer_provenance = backbone.model_config.get("tokenizer_provenance")
    is_bound_custom_tokenizer = (
        tokenizer_provenance is not None
        and tokenizer_provenance.get("kind") == "toast_tst"
    )
    is_gpt2_vocab = (
        is_nano
        and "gpt2vocab" in backbone.architecture
        and not is_bound_custom_tokenizer
    )
    if is_gpt2_vocab:
        # GPT-2 tokens map to fixed byte strings, so the guard reduces to a
        # direct LUT sum: zero leading-space/boundary tables make eval_val's
        # SentencePiece correction term vanish.
        gpt2_bytes = torch.load(
            "data/tokenizers/gpt2_byte_lut.pt", weights_only=True
        ).to(device)
        no_correction = torch.zeros(
            gpt2_bytes.size(0), dtype=torch.bool, device=device
        )
        luts = (gpt2_bytes, no_correction, no_correction)
    elif not is_bound_custom_tokenizer:
        luts = baseline.build_sentencepiece_luts(
            tokenizer, FreshHyperparameters.vocab_size, device
        )
    else:
        # TST token byte lengths depend on adjacent magnitude tokens, so no
        # per-token byte LUT exists. eval_val still supplies the token CE; an
        # all-one LUT makes its temporary BPB equal bits/token, which is
        # replaced below by the exact decode-based byte denominator.
        vocab_size = int(backbone.model_config["vocab_size"])
        one_byte = torch.ones(vocab_size, dtype=torch.int16, device=device)
        no_correction = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        luts = (one_byte, no_correction, no_correction)
    # The BPB guard shares the checkpoint's own tokenizer family, and nano
    # pretrains against its own shard family; keep the guard on the same
    # validation bytes as nano's own pretraining val_bpb (DATA_PATH overrides).
    bpb_val_files = FreshHyperparameters.val_files
    if is_nano:
        default_val_dataset = backbone.model_config.get(
            "pretraining_data_path",
            (
                "data/datasets/fineweb10B_gpt2"
                if is_gpt2_vocab
                else "data/datasets/fineweb_onepass_sp1024"
            ),
        )
        bpb_val_files = os.path.join(
            os.environ.get("DATA_PATH", default_val_dataset),
            "fineweb_val_*.bin",
        )
    val_tokens = baseline.load_validation_tokens(bpb_val_files, seq_len)
    if args.bpb_val_tokens > 0:
        usable = (args.bpb_val_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(
                f"--bpb-val-tokens must cover at least one {seq_len}-token sequence"
            )
        val_tokens = val_tokens[: usable + 1]
        print(
            f"BPB guard subsampled to {usable} validation tokens", flush=True
        )
    exact_val_byte_count = None
    if is_bound_custom_tokenizer:
        from pretraining.byte_accounting import ByteCounter

        exact_val_byte_count = ByteCounter(
            {"tokenizer_provenance": tokenizer_provenance}
        ).count(val_tokens[1:])
        if exact_val_byte_count <= 0:
            raise ValueError("custom-tokenizer validation has no bytes")

    start_step = 0
    warmup_step = 0
    resume_args = None
    resumed_zero_reward_frozen_updates = None
    if args.resume:
        target_execution_schema = execution_schema_for(
            args.reasoning_mode, args.rollout_scheduler
        )
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_args = payload.get("args", {})
        if payload.get("resume_arg_contract_schema") != RESUME_ARG_CONTRACT_SCHEMA:
            raise ValueError(
                "resume checkpoint predates the exact environment/objective "
                "argument contract"
            )
        validate_resume_arg_contract(resume_args, args)
        if payload.get("base_checkpoint_sha256") != args.base_checkpoint_sha256:
            raise ValueError(
                "resume uses different base checkpoint bytes than the saved run"
            )
        validate_combiner_geometry(
            payload,
            args,
            mode_carries_memory(args.reasoning_mode),
            "resume checkpoint",
        )
        validate_renderer_checkpoint(
            payload,
            args.resume,
            expected_rollout_policy_schema=rollout_policy_schema,
        )
        if not resume_execution_schema_compatible(
            payload,
            expected_execution_schema=target_execution_schema,
            allow_token_rng_migration=args.allow_token_rng_migration,
        ):
            raise ValueError(
                "resume checkpoint execution schema must be "
                f"{target_execution_schema!r}; got "
                f"{payload.get('execution_schema')!r}. v29 is the latent "
                "stream, Gaussian scale, gate, and request-RNG contract; v32 "
                "draws cot/none/carry tokens by the counter-keyed graph "
                "decode race on each row's unpadded slot. The only migration "
                "is lockstep cot/none v31 -> v32, acknowledged with "
                "--allow-token-rng-migration."
            )
        # Lands in the manifest's args: a migrated continuation names the
        # contract its restored state was trained under.
        args.resumed_execution_schema = payload.get("execution_schema")
        if not resume_replay_schema_compatible(payload, args.reasoning_mode):
            raise ValueError(
                "resume checkpoint replay numerics schema must be "
                f"{replay_numerics_schema_for(args.reasoning_mode)!r}; got "
                f"{payload.get('replay_numerics_schema')!r}; old replay "
                "layouts are never migrated"
            )
        expected_reward_schema = reward_schema_for_abstention(
            args.abstention_penalty
        )
        if payload.get("reward_schema") != expected_reward_schema:
            raise ValueError(
                f"resume checkpoint reward schema must be {expected_reward_schema!r}; "
                f"got {payload.get('reward_schema')!r}"
            )
        expected_actor_objective_schema = run_actor_objective_schema(args)
        if (
            payload.get("actor_objective_schema")
            != expected_actor_objective_schema
        ):
            raise ValueError(
                "resume checkpoint actor objective schema must be "
                f"{expected_actor_objective_schema!r}; got "
                f"{payload.get('actor_objective_schema')!r}. Use "
                "--actor-init for an initialization restart under the "
                "current objective."
            )
        expected_source_mask_schema = (
            SOURCE_SIGNAL_MASK_SCHEMA
            if args.rl_mixture_manifest and args.source_success_actor_gate
            else None
        )
        if (
            payload.get("source_signal_mask_schema")
            != expected_source_mask_schema
        ):
            raise ValueError(
                "resume checkpoint source-signal mask schema must be "
                f"{expected_source_mask_schema!r}; got "
                f"{payload.get('source_signal_mask_schema')!r}"
            )
        expected_python_reward_schema = getattr(args, "python_reward_schema", None)
        if payload.get("python_reward_schema") != expected_python_reward_schema:
            raise ValueError(
                "resume checkpoint Python reward schema must be "
                f"{expected_python_reward_schema!r}; got "
                f"{payload.get('python_reward_schema')!r}"
            )
        if payload.get("math_data_identity") != data_identity:
            raise ValueError(
                "resume checkpoint's prompt cursor belongs to different dataset "
                "bytes, effective corpus policy/targets, exclusions, or ordering; "
                "start a new run instead of reusing the cursor"
            )
        if bool(resume_args.get("think_tokens")) != bool(args.think_tokens):
            raise ValueError(
                "resume requires the checkpoint's --think-tokens setting: "
                "the fence gate changes the return distribution the critic "
                f"was fit to (checkpoint {bool(resume_args.get('think_tokens'))}, "
                f"got {bool(args.think_tokens)})"
            )
        if int(resume_args.get("think_min_tokens", 1)) != args.think_min_tokens:
            raise ValueError(
                "resume requires the checkpoint's --think-min-tokens: the "
                "floor is part of the reward the critic was fit to "
                f"(checkpoint {int(resume_args.get('think_min_tokens', 1))}, "
                f"got {args.think_min_tokens})"
            )
        if bool(resume_args.get("answer_fence")) != bool(args.answer_fence):
            raise ValueError(
                "resume requires the checkpoint's --answer-fence setting: "
                "the structural gate is part of the reward the critic was "
                f"fit to (checkpoint {bool(resume_args.get('answer_fence'))}, "
                f"got {bool(args.answer_fence)})"
            )
        if (
            args.answer_fence
            and payload.get("answer_fence_prompt_schema")
            != ANSWER_FENCE_PROMPT_SCHEMA
        ):
            raise ValueError(
                "resume checkpoint answer-fence prompt schema must be "
                f"{ANSWER_FENCE_PROMPT_SCHEMA!r}; got "
                f"{payload.get('answer_fence_prompt_schema')!r}. Prompt "
                "wording is part of the policy environment and cannot "
                "change across an exact resume."
            )
        resume_seed = resume_args.get("seed")
        if resume_seed is not None and int(resume_seed) != args.seed:
            raise ValueError(
                "resume requires the checkpoint's --seed: prompt epochs after "
                f"the first reshuffle from it (checkpoint seed {resume_seed}, "
                f"got {args.seed})"
            )
        wrapper.load_state_dict(payload["model"], strict=True)
        if not resume_critic_schema_compatible(
            payload, critic.values_actions, post_action_q=uses_post_action_q(args),
            twin_q=critic.q_twin is not None,
        ):
            raise ValueError(
                "resume checkpoint critic schema must be "
                f"{critic_run_schema(critic)!r}; got "
                f"{payload.get('critic_schema')!r}; old critic heads are "
                "never migrated"
            )
        if payload.get("critic_action_head_schema") != (
            critic_action_head_schema(critic)
        ):
            raise ValueError(
                "resume checkpoint critic action head schema is "
                f"{payload.get('critic_action_head_schema')!r}, expected "
                f"{critic_action_head_schema(critic)!r}"
            )
        critic.load_state_dict(payload["critic"], strict=True)
        expected_optimizer_schema = optimizer_schema_for_trunk_optimizer(
            args.trunk_optimizer,
            fresh_muon=args.fresh_learning_rate is not None,
        )
        if (
            not args.reset_optimizers_on_resume
            and payload.get("optimizer_schema") != expected_optimizer_schema
        ):
            raise ValueError(
                "resume checkpoint optimizer schema is "
                f"{payload.get('optimizer_schema')!r}, expected "
                f"{expected_optimizer_schema!r}; pass "
                "--reset-optimizers-on-resume to keep model/cursor state "
                "while discarding incompatible optimizer state"
            )
        if args.reset_optimizers_on_resume:
            # Trunk-optimizer migration: keep model, critic, step, and prompt
            # cursor; every optimizer starts with empty state under the
            # freshly built layout. Adam second moments rebuild over the next
            # few hundred updates — a transient, not a schedule change.
            print(
                "resume with reset optimizers: discarding "
                f"{sorted(payload['optimizers'])} state for the "
                f"--trunk-optimizer {args.trunk_optimizer} layout "
                f"{sorted(optimizers)}",
                flush=True,
            )
        elif set(payload["optimizers"]) != set(optimizers):
            raise ValueError(
                "resume checkpoint optimizer layout "
                f"{sorted(payload['optimizers'])} does not match the "
                f"--trunk-optimizer {args.trunk_optimizer} layout "
                f"{sorted(optimizers)}; migrate with "
                "--reset-optimizers-on-resume"
            )
        else:
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(payload["optimizers"][name])
        # Learning rates remain the documented cross-run knobs and are
        # reasserted over the loaded optimizer state.
        reassert_learning_rates(optimizers, args)
        start_step = int(payload["step"])
        if entropy_temperature is not None:
            # The actor objective schema already bound the entropy term, so
            # a checkpoint without its state is not this run's.
            if payload.get("entropy_temperature_state") is None:
                raise ValueError(
                    "resume checkpoint has no entropy temperature state"
                )
            entropy_temperature.load_state_dict(
                payload["entropy_temperature_state"]
            )
        resumed_zero_reward_frozen_updates = payload.get(
            "zero_reward_frozen_updates"
        )
        warmup_step = int(
            payload.get(
                "value_warmup_step",
                args.value_warmup_steps if start_step > 0 else 0,
            )
        )
        torch.set_rng_state(payload["cpu_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        sampler.cursor = int(payload["sampler_cursor"])
        actor_init_provenance = payload.get("actor_init_provenance")
        if (
            start_step == 0
            and warmup_step < args.value_warmup_steps
            and rewinds_warmup_prompts(args)
            and "target_sampler_cursor" not in (actor_init_provenance or {})
        ):
            raise RuntimeError(
                "a resumed warmup rewinds to its --actor-init cursor, which "
                "this checkpoint does not record"
            )

    if args.bpb_only or args.bench_only:
        planned_prompt_count = 0
    elif args.rollout_only:
        planned_prompt_count = (
            args.prompts_per_rollout * args.rollout_only_repeats
        )
    else:
        warmup_updates = (
            max(args.value_warmup_steps - warmup_step, 0) if start_step == 0 else 0
        )
        remaining_actor_steps = max(args.steps - start_step, 0)
        warmup_prompt_count = args.prompts_per_minibatch * warmup_updates
        # An action-head warmup rewinds the cursor (see the warmup loop), so
        # its prompts overlap the actor's instead of preceding them.
        actor_start_cursor = (
            actor_init_provenance["target_sampler_cursor"]
            if warmup_updates and rewinds_warmup_prompts(args)
            else None
        )
        if args.consume_all_prompts:
            planned_prompt_count, _ = plan_one_pass_training(
                dataset_rows=len(math_rows),
                sampler_cursor=sampler.cursor,
                warmup_updates=warmup_updates,
                remaining_actor_steps=remaining_actor_steps,
                prompts_per_minibatch=args.prompts_per_minibatch,
                actor_start_cursor=actor_start_cursor,
            )
        elif actor_start_cursor is not None:
            planned_prompt_count = max(
                warmup_prompt_count,
                actor_start_cursor
                + args.prompts_per_minibatch * remaining_actor_steps
                - sampler.cursor,
            )
        else:
            planned_prompt_count = args.prompts_per_minibatch * (
                warmup_updates + remaining_actor_steps
            )
    projected_end_cursor = (
        sampler.cursor_after(sampler.cursor, planned_prompt_count)
        if isinstance(sampler, MixedPromptSampler)
        else sampler.cursor + planned_prompt_count
    )
    if projected_end_cursor > len(math_rows):
        # A recorded rollout dataset treats every group as a distinct
        # prompt (held-out clusters, train/held-out disjointness), so it
        # refuses a second epoch as a one-pass run does.
        if args.consume_all_prompts or args.rollout_dataset:
            raise ValueError(
                "one-pass run would cross the dataset's first epoch and "
                "reuse prompts: cursor "
                f"{sampler.cursor} + planned {planned_prompt_count} > "
                f"{len(math_rows)} rows"
            )
        final_epoch = (projected_end_cursor - 1) // len(
            math_rows
        )
        print(
            f"prompt plan spans {final_epoch + 1} epochs of the "
            f"{len(math_rows)}-row dataset ({planned_prompt_count} prompts); "
            "epochs after the first reshuffle deterministically from --seed",
            flush=True,
        )

    # Built before the first torch.compile so compile accounting starts from
    # an empty metrics deque, and so every compiled artifact below can be
    # wrapped in a call counter. When --profile is absent this is the
    # do-nothing profiler and every profiling call site below is inert.
    profiler = (
        RunProfiler(Path(args.output), device, args)
        if args.profile
        else DisabledProfiler()
    )

    # Every compiled graph in this process reaches a fp32 region —
    # CombinedEmbedding's inject or SeparateCritic.values both disable
    # the ambient autocast — so all of them carry _enter_autocast
    # nodes, and AOTAutogradCache refuses to key on an unrecognized
    # call_function target. It therefore BYPASSED on every process start
    # (measured: "Bypassing autograd cache due to: Unsupported call_function
    # target _enter_autocast", 8 bypasses in an 8-step smoke). Inductor's FX
    # graph cache was already hitting, which is why cold compilation was
    # ~93% AOTAutograd and only 0.28 s of codegen per graph. Declaring the
    # two autocast markers cacheable lets the AOT artifact persist across
    # processes: warm pool-1 replay refresh 18.1 s -> 11.0 s, per-shard
    # steady state unchanged, and the one-off single-row shard
    # specialization 7.2 s -> 1.7 s. Sound because the autocast arguments
    # are constants baked into the graph nodes and hashed with them, and
    # because bf16 is the only ambient autocast dtype here — the ambient
    # DTYPE is not part of the cache key, only whether autocast is on.
    # Declared before the first torch.compile and outside every flag, since
    # this value is serialized into the on-disk key of EVERY graph: gating
    # it would make an artifact's key depend on an unrelated flag. Bump the
    # version string to invalidate every cached artifact.
    for autocast_marker in (
        "torch.amp.autocast_mode._enter_autocast",
        "torch.amp.autocast_mode._exit_autocast",
    ):
        torch._inductor.config.unsafe_marked_cacheable_functions[
            autocast_marker
        ] = "v1"

    # See the --duck-shape help. Set before the first compile because it
    # decides how the FIRST trace allocates symbols, which is the only trace
    # that matters here.
    torch.fx.experimental._config.use_duck_shape = args.duck_shape
    # Note for whoever meets a recompile-limit failure here: cache_size_limit
    # (raised to 64 below) is not the only ceiling. exceeds_recompile_limit
    # checks accumulated_recompile_limit (default 256) FIRST, and then again
    # as a backstop on the frame's compile id. Under fullgraph=True either
    # raises rather than falling back to eager, so it would kill a long run
    # mid-flight. Deliberately NOT raised: compute_cache_size walks the live
    # entry list, so invalidated entries do not accumulate there.
    #
    # The frame that gets close is step_core, and only under
    # --rollout-flex-decode: eval's dynamic artifact, the tail graph, and the
    # dynamic=False rollout artifact all compile that one code object, and the
    # last of them specializes per (bucketed row count, kv width). That is why
    # decode_width_grid below caps the width axis at two -- at ~9 row counts
    # it keeps the total near 20 rather than the ~36 a block-fine width grid
    # would produce. Without the flag the measured maximum for any frame is 3.

    # Rollout and evaluation use the identical dynamic narrow-prefix step.
    # Compile it once: separate wrappers paid the same large cold compilation
    # cost at step-0 eval and again at the first training collect. If eval ever
    # latches an eager fallback, the eval closure below also disables this
    # shared artifact for training before it can be called again.
    # Pinned-EMIT lockstep modes (cot/none/carry) decode through the graph
    # arena below instead; only the latent lockstep rollout steps this one.
    eager_lockstep_rollout = (
        args.rollout_compile
        and args.rollout_scheduler == "lockstep"
        and not pin_emit
    )
    compiled_generation_step = None
    if args.eval_compile or eager_lockstep_rollout:
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64
        )
        compiled_generation_step = profiler.register_artifact(
            "generation_step",
            torch.compile(
                wrapper.step_core,
                mode="max-autotune-no-cudagraphs",
                fullgraph=True,
                dynamic=True,
            ),
        )
    rollout_step_core = (
        compiled_generation_step if eager_lockstep_rollout else None
    )
    if rollout_step_core is not None and args.rollout_flex_decode:
        # Flex decoding needs its OWN artifact: the shared one above is
        # dynamic=True for eval's arbitrary batches, and under symbolic shapes
        # Inductor has no flex decode choice to pick at all (measured -- the
        # lowering fails outright with "no choices exist for backend", and a
        # dynamic batch alone is enough to do it). The rollout can afford
        # dynamic=False because it now compacts to bucketed row counts, so
        # the row count takes a bounded number of values and the KV width is
        # pinned by the mask. Separate from the paged artifact for the same
        # reason that one exists.
        # NOT max-autotune: each bucket specialization compiles lazily, the
        # first time that row count appears, which is mid-rollout with the
        # full KV cache set already resident. Exhaustive benchmarking there
        # allocates candidate workspaces on top of it and spikes the peak --
        # measured at 24.68 GiB against a 15.0 GiB steady state on the same
        # config, which is what OOMs the production shape.
        rollout_step_core = profiler.register_artifact(
            "flex_generation_step",
            torch.compile(
                wrapper.step_core,
                fullgraph=True,
                dynamic=False,
                # Under --rollout-graph-decode the loop holds one row count and
                # one KV width for the whole run, which is the precondition
                # capture needs: a single recording, replayed every step. The
                # caches are marked static and the mask's closed-over buffers
                # are refilled in place, so no address the recording captured
                # ever moves.
                **(
                    {"mode": "reduce-overhead"}
                    if args.rollout_graph_decode
                    else {}
                ),
            ),
        )
    eval_step_core = compiled_generation_step if args.eval_compile else None
    rollout_paged_step_core = None
    if args.rollout_scheduler == "continuous_refill":
        # One compiled entry per declared width bucket; the default limit
        # of 8 would silently drop the small tail widths to eager. Raised
        # here rather than only in the eval/lockstep branch above so the
        # paged path never depends on --eval-compile being on.
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64
        )
        if args.rollout_graph_decode:
            # Declared shapes, hard errors: a cudagraph skip after the
            # warmup captured every width is a bug, not a fallback —
            # surface it as a RuntimeError instead of a silently eager
            # A/B arm.
            torch._inductor.config.triton.cudagraph_or_error = True
        rollout_paged_step_core = profiler.register_artifact(
            "rollout_paged_step",
            torch.compile(
                wrapper.paged_step_core,
                fullgraph=True,
                # FlexDecoding requires a concrete batch dimension. The
                # scheduler supplies one full-capacity main bucket plus
                # power-of-two tail buckets, avoiding a specialization for
                # every possible survivor count.
                dynamic=False,
                # Under --rollout-graph-decode the bucket set doubles as a
                # vLLM-style capture list: one CUDA graph per declared
                # width, replayed as ~one launch per decode step. The mode
                # is max-autotune (== max-autotune-no-cudagraphs plus
                # triton.cudagraphs) rather than reduce-overhead, which
                # would silently drop autotuning and confound the A/B with
                # a kernel-quality change. The paged arena is
                # mark_static_address'd at allocation (mutated graph
                # inputs must hold fixed addresses or inductor skips
                # capture), and warmup_decode_width_buckets captures every
                # width before the first real token.
                mode=(
                    "max-autotune"
                    if args.rollout_graph_decode
                    else "max-autotune-no-cudagraphs"
                ),
            ),
        )

    # Static-tail CUDA graph for the fixed-size compacted training tail.
    # The caches are allocated ONCE and live for the whole run: a fresh
    # allocation would move the static addresses and force a graph
    # re-record. The artifact wraps the ORIGINAL eager step_core (collect's
    # temporary rollout_step_core patch never reaches it), and it only ever
    # runs under the training autocast at one shape, so it stays a single
    # cudagraph specialization.
    # Flex decoding wants a block-aligned KV width. The boolean tail mask does
    # not, and rounding for it would only widen the range it scores, so the
    # alignment follows the flag that needs it.
    decode_block_size = DecodeRangeMask.DEFAULT_BLOCK_SIZE
    rollout_cache_width = args.prompt_tokens + max_stream_steps
    if args.rollout_flex_decode:
        rollout_cache_width = (
            -(-rollout_cache_width // decode_block_size) * decode_block_size
        )
    rollout_tail_caches = None
    rollout_tail_step_core = None
    if args.rollout_tail_graph:
        rollout_tail_caches = wrapper.make_static_generation_cache(
            args.rollout_tail_batch,
            rollout_cache_width,
            device,
            dtype=torch.bfloat16,
        )
        rollout_tail_step_core = profiler.register_artifact(
            "rollout_tail_step",
            torch.compile(
                wrapper.step_core,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,
            ),
        )

    # Flex decoding. A boolean attn_mask disqualifies every fused SDPA backend,
    # and the v25 kernel profile put the resulting memory-efficient kernel at
    # 32% of pool device time with 12672 of its 12684 calls coming from this
    # step. The target is the lockstep MAIN loop, not the tail: at
    # --rollout-groups 32 the direct call count in NOTES.md:1285 is 2080 main
    # vs 32 tail, so a tail-only change reaches 1.52% of the decode iterations
    # and the tail does not even engage until 96.9% of rows have ended.
    # Inductor lowers flex decoding for fully static shapes alone, so the main
    # loop rounds its live row count up to a bucket multiple and gives filler
    # rows an EMPTY key range -- measured to return exactly 0, finite, reading
    # no KV -- which makes the padding free.
    #
    # Masks are owned by the caller, not rebuilt inside the rollout, because
    # they hold persistent buffers whose addresses a cudagraph would capture.
    # (They do NOT cost extra compiles: rebuilding per step was measured at
    # zero additional specializations, so the mask_mod closure is not guarded
    # on by identity.)
    rollout_tail_decode_mask = None
    if args.rollout_flex_decode and not is_nano:
        # validate_args cannot see the checkpoint. The other accepted family is
        # PoPE, whose _attention_step scores a complex inner product over a
        # k_real/k_imag pair rather than one QK product and raises on a block
        # mask -- at the first tail switch, which is many pools in. Fail here.
        raise ValueError(
            f"--rollout-flex-decode needs a single-QK decode step; "
            f"architecture {backbone.architecture!r} does not have one"
        )
    # The KV width is what makes flex decoding expensive in memory. The
    # boolean path narrows the cache to the written prefix every step; the flex
    # step reads the whole allocated width, so the width is fixed at
    # allocation. Pinning it to the args upper bound (prompt_tokens +
    # max_stream_steps = 2560) instead of the pool's actual padded prompt costs
    # ~2.2 GiB at the production shape, which is more than a 32 GB card has
    # spare once two rollout groups' cache sets briefly coexist -- measured as
    # an OOM at make_generation_cache with 27.09 GiB already allocated. So the
    # mask is built per chunk width instead, and a pool of short prompts skips
    # the widest cache entirely.
    #
    # The grid is deliberately much coarser than the KV block size. Every
    # distinct (row count, kv width) pair is its own dynamic=False
    # specialization of step_core, and eval's artifact and the tail graph
    # compile the SAME code object, so all three share one cache_size_limit.
    # Rounding to the 128-wide block would give four widths against the ~9
    # bucketed row counts -- 36 entries against a limit of 64, and an overflow
    # under fullgraph=True RAISES rather than falling back to eager, killing a
    # long run mid-flight. Half the prompt bound caps the width axis at two
    # while keeping the adaptivity that pays.
    decode_width_grid = max(
        decode_block_size,
        -(-args.prompt_tokens // (2 * decode_block_size)) * decode_block_size,
    )
    rollout_decode_masks: dict[int, DecodeRangeMask] = {}
    # No pool holds more than prompts_per_rollout prompts, so a chunk never
    # needs more rows than that: a smaller pool must not pay for (and
    # capture every bucket of) a full --rollout-groups workspace.
    rollout_chunk_groups = min(
        max(args.rollout_groups, 1), args.prompts_per_rollout
    )
    rollout_graph_rows = rollout_chunk_groups * args.samples_per_prompt

    def decode_mask_for(prompt_width: int) -> DecodeRangeMask | None:
        """Main-loop mask sized for a chunk left-padded to ``prompt_width``."""
        if not args.rollout_flex_decode or rollout_step_core is None:
            return None
        if args.rollout_graph_decode:
            # One recording for the run means one width for the run, so the
            # per-chunk adaptivity below is not available: the mask must match
            # the static cache exactly.
            width = rollout_cache_width
        else:
            width = prompt_width + max_stream_steps
            width = -(-width // decode_width_grid) * decode_width_grid
        mask = rollout_decode_masks.get(width)
        if mask is None:
            mask = DecodeRangeMask(
                # --rollout-groups 0 selects the sequential single-prompt
                # branch, which still rolls out samples_per_prompt rows.
                rollout_graph_rows,
                width,
                device,
                block_size=decode_block_size,
            )
            rollout_decode_masks[width] = mask
        return mask

    # The captured main loop's arena. Allocated ONCE for the whole run, at the
    # full row count and the full KV width, because a re-allocation moves the
    # addresses the recording captured and forces a re-record. It is also the
    # reason the per-chunk cache churn disappears under this flag: the old path
    # allocated a fresh cache per chunk and relied on the cyclic collector to
    # reclaim the previous one before the next expanded.
    #
    # That permanence is also its cost: unlike the per-chunk cache it replaces,
    # it stays resident through the update, so it is charged against peak VRAM
    # rather than overlapping it. ``planned_prompt_count`` is zero exactly when
    # no rollout will run (--bpb-only, --bench-only), and those modes must not
    # pay multiple GiB for an arena they never touch.
    rollout_graph_caches = None
    if args.rollout_graph_decode and planned_prompt_count > 0:
        rollout_graph_caches = wrapper.make_static_generation_cache(
            rollout_graph_rows,
            rollout_cache_width,
            device,
            dtype=torch.bfloat16,
        )

    # Lockstep cot/none/carry decode: one CUDA graph per live-row bucket,
    # replayed once per token; see graph_decode.py. The compiled tick lives
    # for the run; its caches and captures only for each rollout, so they do
    # not stay resident through the update.
    decode_arena = None
    if (
        pin_emit
        and args.rollout_scheduler == "lockstep"
        and planned_prompt_count > 0
    ):
        decode_arena = PinnedDecodeArena(
            wrapper,
            rows=rollout_graph_rows,
            kv_width=args.prompt_tokens + max_stream_steps,
            temperature=args.temperature,
            stop_ids=stop_ids,
            device=device,
            hidden_carry=hidden_carry,
            sync_every=args.rollout_sync_every,
            compile=args.rollout_compile,
            cuda_graphs=args.rollout_compile and device.type == "cuda",
            write_noise=args.write_noise_sweep is not None,
        )
    # Pinned-EMIT evaluation decodes through its own arena: the eval
    # protocol's nucleus is a compile-time constant of the tick, which the
    # untruncated training arena must never carry. It is sized to evaluate
    # a whole panel in as few rollouts as possible -- each rollout pays for
    # its longest row, so AIME's 30x32 samples fit one -- and holds its
    # caches only while an evaluation runs.
    eval_arena = None
    eval_stream_steps = max(
        [aime_stream_steps] * bool(aime_rows)
        + [bench_stream_steps] * bool(bench_rows)
        or [0]
    )
    if (
        pin_emit
        and args.eval_compile
        and device.type == "cuda"
        and eval_stream_steps > 0
    ):
        eval_arena = PinnedDecodeArena(
            wrapper,
            rows=args.eval_batch_trajectories,
            kv_width=args.prompt_tokens + eval_stream_steps,
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            stop_ids=stop_ids,
            device=device,
            hidden_carry=hidden_carry,
            sync_every=args.rollout_sync_every,
            replay_storage=False,
            memory_read_ablation=(
                args.memory_eval_ablation
                if args.memory_eval_ablation in FOREIGN_READ_ABLATIONS
                else "none"
            ),
        )

    if args.rollout_flex_decode and rollout_tail_caches is not None:
        rollout_tail_decode_mask = DecodeRangeMask(
            args.rollout_tail_batch,
            rollout_cache_width,
            device,
            block_size=decode_block_size,
        )

    # The no-grad post-update drift diagnostic keeps its own functions: the
    # training artifacts below are rebound over these names, and calling a
    # grad-enabled artifact under no-grad adds a guarded specialization to it.
    # Its replay head stays eager — once per 100 steps it is not worth a
    # second no-grad specialization per shard shape, and without autograd its
    # memory is only the live activations. Its readout tail is compiled as a
    # separate no-grad artifact: eagerly it materializes several
    # (slots, vocab) fp32 temporaries, which made this diagnostic the run's
    # peak-memory step (~19 GiB against a ~7 GiB update).
    diagnostic_replay_head_inputs = replay_head_inputs
    diagnostic_emit_token_logprobs = compact_emit_token_logprobs
    diagnostic_emit_token_log_odds = compact_emit_token_log_odds
    if args.compile_replay:
        diagnostic_emit_token_logprobs = profiler.register_artifact(
            "diagnostic_emit_token_logprobs",
            torch.compile(
                compact_emit_token_logprobs,
                mode="default",
                fullgraph=True,
                dynamic=True,
            ),
        )
        if args.target_policy_optimization:
            diagnostic_emit_token_log_odds = profiler.register_artifact(
                "diagnostic_emit_token_log_odds",
                torch.compile(
                    compact_emit_token_log_odds,
                    mode="default",
                    fullgraph=True,
                    dynamic=True,
                ),
            )

    # Replay compilation is independent of rollout. Dynamic B/L plus bounded
    # 64-token buckets lets one artifact cover the length-aware shard plan;
    # no CUDA graph owns outputs that remain live through eager losses/backward.
    # Measured, not assumed: one artifact each covers 23 distinct (B, L) shard
    # shapes and every unseen shape costs 3-26 ms, so the bucket exists to keep
    # the packed batches compact, NOT to limit compilations. The only shape
    # that can still recompile is a single-row shard — the framework's 0/1
    # specialization, guard "2 <= batch.token_ids.size()[0]" — which the
    # planner emits whenever the row remainder is one. That is one extra
    # compile per process and torch offers no way to avoid it here:
    # mark_unbacked makes Inductor's constant folder raise
    # GuardOnDataDependentSymNode on the row dimension.
    trim_multiple = args.replay_bucket if args.compile_replay else 1
    if args.compile_replay:
        torch._dynamo.config.cache_size_limit = 64
        # ``values`` reads the instance's ``estimates``, so rebinding that one
        # method compiles both the value and the action-value paths.
        if uses_post_action_q(args):
            critic.action_success_logits = profiler.register_artifact(
                "critic_action_success_logits",
                torch.compile(
                    critic.action_success_logits, mode="default",
                    fullgraph=trunk_fullgraph, dynamic=True,
                ),
            )
        critic.estimates = profiler.register_artifact(
            "critic_estimates",
            torch.compile(
                critic.estimates,
                # Length buckets still span many GEMM shapes. Max-autotune
                # repeatedly stalls training to benchmark each new regime;
                # default Inductor dispatches them to stable cuBLAS kernels.
                mode="default",
                # trunk_fullgraph: KDA replay traverses the eager FLA chunk
                # kernel, so the artifact compiles as subgraphs around it.
                # Age-0 exactness needs one artifact shared by refresh and
                # update, not zero graph breaks — the rebinding below is
                # unchanged.
                fullgraph=trunk_fullgraph,
                dynamic=True,
            ),
        )
        compiled_replay = profiler.register_artifact(
            "replay_head_inputs",
            torch.compile(
                replay_head_inputs,
                mode="default",
                fullgraph=trunk_fullgraph,
                dynamic=True,
            ),
        )
        # Both consumers bound the function by name at import time:
        # update_minibatch through this module's global,
        # refresh_old_statistics through latent_rollout's.  Rebinding both
        # to the SAME compiled object keeps refresh and update on one
        # compiled artifact — the property that makes behavior-age-0 PPO ratios
        # exactly one (see refresh_old_statistics on why it also runs
        # grad-enabled).
        globals()["replay_head_inputs"] = compiled_replay
        postraining.latent_rollout.replay_head_inputs = compiled_replay
        # The vocabulary readout tail, for the same reason and by the same
        # rebinding. Eagerly it is the readout GEMM followed by ~7 separate
        # passes over a (slots, 50257) fp32 tensor — softcap pow/add/rsqrt/
        # mul/mul, then log-softmax — to produce one scalar per slot. Fused,
        # the vocabulary axis stays in registers and only the (slots,) result
        # is written. Both consumers must land on THIS object: two artifacts
        # would be free to pick different reduction orders, and refresh minus
        # update is precisely the age-0 canary.
        compiled_emit_logprobs = profiler.register_artifact(
            "compact_emit_token_logprobs",
            torch.compile(
                compact_emit_token_logprobs,
                mode="default",
                fullgraph=True,
                dynamic=True,
            ),
        )
        globals()["compact_emit_token_logprobs"] = compiled_emit_logprobs
        postraining.latent_rollout.compact_emit_token_logprobs = (
            compiled_emit_logprobs
        )
        if args.entropy_target is not None:
            # The entropy objective's readout tail, compiled for the same
            # reason: eagerly its forward holds (slots, vocab) fp32
            # temporaries. Only the update consumes it; under fused
            # statistics its log-probs are also the behavior log-probs.
            globals()["compact_emit_token_logprobs_and_entropy"] = (
                profiler.register_artifact(
                    "compact_emit_token_logprobs_and_entropy",
                    torch.compile(
                        compact_emit_token_logprobs_and_entropy,
                        mode="default",
                        fullgraph=True,
                        dynamic=True,
                    ),
                )
            )
        if args.critic_q_head_hidden is not None:
            # The token Q critic's readout tail: the same fused log-probs
            # and entropy, plus the policy-centered Q from the same raw
            # logits. It is the update's only readout under that critic.
            globals()["compact_emit_token_logprobs_entropy_and_advantages"] = (
                profiler.register_artifact(
                    "compact_emit_token_logprobs_entropy_and_advantages",
                    torch.compile(
                        compact_emit_token_logprobs_entropy_and_advantages,
                        mode="default",
                        fullgraph=True,
                        dynamic=True,
                    ),
                )
            )
        if args.critic_q_head_hidden is not None:
            # critic_shard reads a token Q critic's trunk pass directly, not
            # through the rebound ``estimates``; centring Q under the policy
            # is elementwise and follows the actor pass.
            critic.token_q_parts = profiler.register_artifact(
                "critic_token_q_parts",
                torch.compile(
                    critic.token_q_parts, mode="default",
                    fullgraph=trunk_fullgraph, dynamic=True,
                ),
            )
        if args.mean_actor_critic:
            # One artifact per readout chunk shape class; the custom function
            # keeps the vocabulary axis out of autograd either way.
            globals()["nano_mean_actor_critic"] = profiler.register_artifact(
                "nano_mean_actor_critic",
                torch.compile(
                    nano_mean_actor_critic, mode="default",
                    fullgraph=True, dynamic=True,
                ),
            )
        if args.critic_q_pessimism:
            globals()["nano_pessimistic_q_prior"] = profiler.register_artifact(
                "nano_pessimistic_q_prior",
                torch.compile(
                    nano_pessimistic_q_prior, mode="default",
                    fullgraph=True, dynamic=True,
                ),
            )
        if args.target_policy_optimization:
            compiled_emit_log_odds = profiler.register_artifact(
                "compact_emit_token_log_odds",
                torch.compile(
                    compact_emit_token_log_odds,
                    mode="default",
                    fullgraph=True,
                    dynamic=True,
                ),
            )
            globals()["compact_emit_token_log_odds"] = compiled_emit_log_odds
            postraining.latent_rollout.compact_emit_token_log_odds = (
                compiled_emit_log_odds
            )

    output = Path(args.output)
    existing_manifest = None
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise ValueError(
                f"refusing to start a fresh run in nonempty output {output}; "
                "use a new directory or pass a compatible --resume checkpoint"
            )
        if Path(args.resume).resolve().parent != output.resolve():
            raise ValueError(
                "a nonempty resume output must be the resume checkpoint's "
                "own run directory"
            )
        existing_manifest_path = output / "manifest.json"
        if not existing_manifest_path.is_file():
            raise ValueError("nonempty resume output has no run manifest")
        existing_manifest = json.loads(existing_manifest_path.read_text())
        if (
            existing_manifest.get("math_data_identity") != data_identity
            or (existing_manifest.get("base") or {}).get("checkpoint_sha256")
            != args.base_checkpoint_sha256
        ):
            raise ValueError(
                "nonempty resume output manifest belongs to a different run"
            )
    output.mkdir(parents=True, exist_ok=True)
    source_names = (
        [str(entry["name"]) for entry in mixture_manifest["sources"]]
        if mixture_manifest is not None
        else ["math"]
    )
    source_ids = {name: index for index, name in enumerate(source_names)}
    topology_history = resume_topology_history(
        existing_manifest,
        resume_args,
        args,
        checkpoint=args.resume,
        checkpoint_sha256=args.resume_checkpoint_sha256,
        step=start_step,
        sampler_cursor=sampler.cursor,
    )
    resumed_manifest = existing_manifest
    if args.resume and resumed_manifest is None:
        # Resuming into a new directory: the lineage lives beside the
        # checkpoint, in the run directory that wrote it.
        resumed_manifest_path = Path(args.resume).parent / "manifest.json"
        if not resumed_manifest_path.is_file():
            raise ValueError(
                f"resume checkpoint {args.resume} has no run manifest beside "
                "it to carry its critic init forward"
            )
        resumed_manifest = json.loads(resumed_manifest_path.read_text())
    critic_init_provenance = resolve_critic_init_provenance(
        args, actor_init_payload, resumed_manifest
    )
    source_provenance = capture_source_provenance(
        output / "provenance", Path(__file__).resolve().parents[1]
    )
    args.source_provenance = source_provenance
    logger = JsonlLogger(output / "metrics.jsonl")
    tensorboard_purge_step = None
    if args.resume:
        removed = logger.purge_after(start_step, warmup_step)
        removed_reports = purge_benchmark_reports_after(output, start_step)
        # The pool at start_step is collected again, so its capture goes too.
        removed_captures = purge_rollout_samples_from(output, start_step)
        if removed_captures:
            print(
                f"purged {removed_captures} stale rollout sample captures "
                f"from actor step {start_step}",
                flush=True,
            )
        if removed:
            print(
                f"purged {removed} stale metrics after actor step {start_step} "
                f"and warmup step {warmup_step}",
                flush=True,
            )
        if removed_reports:
            print(
                f"purged {removed_reports} stale benchmark answer reports",
                flush=True,
            )
        tensorboard_purge_step = (
            warmup_step + 1 if start_step == 0 else start_step + 1
        )
    tensorboard = SummaryWriter(
        output / "tensorboard", purge_step=tensorboard_purge_step
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "phase": (
                    "latent_vapo_verifiable_mixture"
                    if mixture_manifest is not None
                    else "latent_vapo_math"
                ),
                # Top level, not buried in args: a profiled run's timings are
                # perturbed by the profiler and must not be quoted as this
                # configuration's cost.
                "profiled": bool(args.profile),
                "profile_schema": PROFILE_SCHEMA if args.profile else None,
                # Top level too: an ablated bench is not the policy's bench.
                "memory_eval_ablation": {
                    "mode": args.memory_eval_ablation,
                    "schema": MEMORY_EVAL_ABLATION_SCHEMA,
                },
                "source_provenance": source_provenance,
                "topology_history": topology_history,
                "execution_schema": execution_schema_for(
                    args.reasoning_mode, args.rollout_scheduler
                ),
                "actor_objective_schema": run_actor_objective_schema(args),
                "source_signal_mask_schema": (
                    SOURCE_SIGNAL_MASK_SCHEMA
                    if args.rl_mixture_manifest
                    and args.source_success_actor_gate
                    else None
                ),
                "resume_arg_contract_schema": RESUME_ARG_CONTRACT_SCHEMA,
                "replay_numerics_schema": replay_numerics_schema_for(
                    args.reasoning_mode
                ),
                "prompt_order_schema": PROMPT_ORDER_SCHEMA,
                "answer_fence_prompt_schema": (
                    ANSWER_FENCE_PROMPT_SCHEMA if args.answer_fence else None
                ),
                "math_data_identity": data_identity,
                "math_corpus_identity": math_corpus_identity(math_rows),
                "math_corpus_audit": corpus_audit,
                "rl_mixture_manifest": mixture_manifest,
                "excluded_rl_sources": list(filter(None, args.exclude_rl_sources.split(","))),
                "source_exclusion_history": (
                    list((resumed_manifest or {}).get("source_exclusion_history", []))
                    + ([{
                        "checkpoint": args.resume,
                        "checkpoint_sha256": args.resume_checkpoint_sha256,
                        "step": start_step,
                        "original_sampler_cursor": sampler.cursor,
                        "before": (resume_args or {}).get("exclude_rl_sources", ""),
                        "after": args.exclude_rl_sources,
                        "schema": "original_mixture_position_skip/v1",
                    }] if args.resume and (resume_args or {}).get(
                        "exclude_rl_sources", ""
                    ) != args.exclude_rl_sources else [])
                ),
                "rl_source_ids": source_ids,
                "python_reward_schema": getattr(args, "python_reward_schema", None),
                "reward_schema": reward_schema_for_abstention(
                    args.abstention_penalty
                ),
                "reasoning_mode": args.reasoning_mode,
                "context_tokens": context_tokens,
                "checkpoint_context_limit": checkpoint_context_limit,
                "trained_context_tokens": trained_context_tokens,
                "context_extrapolation": context_extrapolation,
                "eval_context_extrapolation": bool(
                    context_extrapolation and (args.rollout_only or args.bench_only)
                ),
                "math_modal_answer_baseline": math_modal_baseline,
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "rollout_policy_schema": rollout_policy_schema,
                "thought_input_schema": wrapper.thought_input_schema,
                "thought_distribution_schema": (
                    wrapper.thought_distribution_schema
                ),
                "optimizer_schema": optimizer_schema_for_trunk_optimizer(
                    args.trunk_optimizer,
                    fresh_muon=args.fresh_learning_rate is not None,
                ),
                "args": vars(args),
                "base": {
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": args.base_checkpoint_sha256,
                    "architecture": backbone.architecture,
                    # SFT lineage when the base came out of sft_trace_train
                    # (plain scalars: schema, traces, args, steps) — the
                    # think_tokens flag in here is what the startup guards
                    # key off.
                    "sft": sft_provenance or None,
                },
                "actor_init": actor_init_provenance,
                "critic": {
                    **critic_init_provenance,
                    "architecture": critic.trunk.architecture,
                    "model_config": critic.trunk.model_config,
                    "dense_for_latent_moe": critic_uses_dense_trunk,
                    "frozen_actor_moe_router_parameters": (
                        frozen_moe_router_parameters
                    ),
                    "critic_schema": critic_run_schema(critic),
                    "action_head_schema": critic_action_head_schema(critic),
                    "action_head_hidden": args.critic_action_head_hidden,
                    "q_head_hidden": args.critic_q_head_hidden,
                    "q_twin": args.critic_q_twin,
                    "q_pessimism": args.critic_q_pessimism,
                    "post_action_q_hidden": args.critic_post_action_q_hidden,
                    "post_action_q_prior": args.critic_post_action_q_prior,
                    "q_metrics": ("raw_sigmoid_predictions_before_exact_prefix_override"
                                  if uses_post_action_q(args) else None),
                    "q_head_reads_actor_belief": (
                        critic.q_head is not None
                        and critic.q_head.actor_belief is not None
                    ),
                    "q_lambda": (
                        None
                        if args.critic_q_head_hidden is None and not uses_post_action_q(args)
                        else "length_adaptive"
                        if args.q_critic_lambda is None
                        else args.q_critic_lambda
                    ),
                    "combined_mlp_hidden": args.combined_mlp_hidden,
                    "combined_mlp_blocks": args.combined_mlp_blocks,
                    "parameters": sum(p.numel() for p in critic.parameters()),
                },
            },
            indent=2,
        )
        + "\n"
    )
    # Guess-calibration record: each module's modal-answer share is the
    # accuracy of answering its most common ground truth every time, so any
    # module accuracy is evidence of solving only above this line. Retain it
    # once per dataset in structured JSON for drill-down without creating
    # dozens of low-frequency TensorBoard series.
    for name, dataset_rows in (("bench", bench_rows), ("math", sampler.rows)):
        baselines = module_answer_baselines(dataset_rows)
        if baselines and not args.resume:
            # One nested kwarg: module names must never collide with the
            # logger's fixed "type" key.
            logger.log(type=f"{name}_guess_baseline", baselines=baselines)
    if not args.resume:
        logger.log(type="math_modal_answer_baseline", **math_modal_baseline)

    rollout_recorder = RolloutSampleRecorder(
        tokenizer, stop_ids, source_names, answer_prefix_ids
    )
    gate_transcripts = (
        GateTranscriptWriter(output / "gate_transcripts.jsonl", tokenizer, stop_ids)
        if args.rollout_only and args.gate_transcripts else None
    )
    rollout_dataset = (
        RolloutDatasetWriter(
            output / "rollout_dataset",
            {
                # The run manifest pins everything a reward depended on:
                # checkpoint, data, prompt and reward schemas, sampling.
                "run_manifest_sha256": file_sha256(output / "manifest.json"),
                "checkpoint_sha256": args.base_checkpoint_sha256,
                # The --curriculum-init snapshot whose actor sampled the
                # trajectories, or None for the --checkpoint policy itself.
                # Readers that replay --checkpoint refuse any other policy.
                "policy_state_sha256": args.initialization_checkpoint_sha256,
                "math_data_identity": data_identity,
                "reasoning_mode": args.reasoning_mode,
                "reward_schema": reward_schema_for_abstention(
                    args.abstention_penalty
                ),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "samples_per_prompt": args.samples_per_prompt,
                "seed": args.seed,
                "max_new_tokens": args.resolved_train_max_new_tokens,
                "think_min_tokens": args.think_min_tokens,
                "nearby_reward_max": args.nearby_reward_max,
                # Trainer order: stop_ids[0] is the EOS that abstention
                # detection keys on.
                "stop_ids": [int(token) for token in stop_ids],
                "trim_multiple": trim_multiple,
                "prompts_per_rollout": args.prompts_per_rollout,
            },
            stop_ids,
            planned_pools=args.rollout_only_repeats,
        )
        if args.rollout_only and args.rollout_dataset else None
    )

    def finish_group(
        batch: LatentRolloutBatch,
        row: dict,
        refresh_statistics: bool,
    ) -> LatentRolloutBatch:
        batch = trim_stream(batch, multiple=trim_multiple)
        if hidden_carry:
            validate_hidden_carry_rollout(batch)
        verifier_kind = row.get("_verifier_kind", "math")
        if verifier_kind == "math":
            verdicts = score_math_rollout(
                batch, row["reward_model"]["ground_truth"], tokenizer, stop_ids,
                answer_style(row),
                args.nearby_reward_max,
                solution_prefix_ids=answer_prefix_ids,
                think_fence_ids=think_fence_ids,
                min_think_tokens=args.think_min_tokens,
                answer_fence_ids=answer_fence_ids,
                abstention_penalty=args.abstention_penalty,
            )
        elif verifier_kind == "python_mbpp":
            assert think_fence_ids is not None and answer_fence_ids is not None
            verdicts = score_python_rollout(
                batch,
                row["verification_info"],
                tokenizer,
                stop_ids,
                think_fence_ids,
                answer_fence_ids,
                args.think_min_tokens,
                reward_mode=args.python_reward_mode,
                abstention_penalty=args.abstention_penalty,
            )
        else:
            raise ValueError(f"unsupported verifier kind {verifier_kind!r}")
        source_name = str(row.get("_rl_source", "math"))
        if gate_transcripts is not None:
            gate_transcripts.write(batch, verdicts, row)
        if rollout_dataset is not None:
            rollout_dataset.write(batch, verdicts, row, prompt_text(row))
        if rollout_recorder.armed:
            rollout_recorder.offer(
                batch,
                verdicts,
                prompt_text(row),
                (
                    str(row["reward_model"]["ground_truth"])
                    if verifier_kind == "math"
                    else json.dumps(row["verification_info"], ensure_ascii=False)
                ),
                source_name,
            )
        batch.source_id = torch.full(
            (batch.kind.size(0),),
            source_ids[source_name],
            dtype=torch.long,
            device=batch.kind.device,
        )
        if args.mean_actor_critic and args.mac_estimator in SEQUENCE_ESTIMATORS:
            if verifier_kind != "math":
                # Test results are not answers: two programs passing the
                # same tests are not one cluster, and their source text
                # has no canonical form.
                raise ValueError(
                    f"--mac-estimator {args.mac_estimator} clusters graded math "
                    f"answers; source {source_name!r} is verified by {verifier_kind}"
                )
            stamp_group_terms(batch, verdicts)
        # Stepwise rollout and parallel replay disagree numerically at
        # bf16 scale; recompute the stored PPO statistics through the
        # update-step replay path so fresh-behavior ratios are exactly one.
        # This also fills old_values from the separate critic (the
        # rollout never values).
        if refresh_statistics:
            refresh_old_statistics(
                wrapper,
                critic,
                batch,
                max_trajectories=args.replay_max_trajectories,
                attention_budget=args.replay_attention_budget,
                bucket_multiple=args.replay_bucket,
                slot_budget=args.replay_slot_budget,
                target_policy_optimization=args.target_policy_optimization,
            )
        return batch

    last_decode_schedule_metrics: dict[str, float] | None = None
    rollout_paged_cache = None

    def _collect(
        refresh_statistics: bool,
        prompt_count: int,
        offload_to_cpu: bool,
        scoring_pool: ThreadPoolExecutor | None,
        write_noise: WriteNoisePlan | None,
    ) -> list[LatentRolloutBatch]:
        """One rollout: a scored prompt group per sampled DAPO problem.

        ``write_noise`` explores the memory writes of every trajectory.
        """
        nonlocal last_decode_schedule_metrics, rollout_paged_cache
        last_decode_schedule_metrics = None
        pool_start_cursor = sampler.cursor
        rollout_rows = sampler.next_rows(prompt_count)
        prompt_budget = args.prompt_tokens - len(answer_prefix_ids)
        with profiler.phase("prompt_encode"):
            encoded_rows = [
                (
                    row,
                    encode_prompt(tokenizer, prompt_text(row), prompt_budget)
                    + list(answer_prefix_ids),
                )
                for row in rollout_rows
            ]
        # The sequential sampler has already consumed these rows. Stable
        # length sorting only reduces left-padding inside this pool; it does
        # not alter dataset coverage, reuse, or use RNG to order data.
        encoded_rows.sort(key=lambda item: len(item[1]))
        groups = []
        cpu = torch.device("cpu")

        def retain_group(
            batch: LatentRolloutBatch,
            row: dict,
        ) -> LatentRolloutBatch:
            return finish_group(batch, row, refresh_statistics)

        # The sampler cursor is monotonic and checkpointed, so mixed with the
        # user seed it gives each pool a resume-stable key namespace; a
        # trajectory's key is then (pool, position in the sorted pool,
        # sample), whatever the chunking.
        pool_seed = (args.seed << 32) ^ pool_start_cursor

        def lockstep_rollout(
            prompt_ids: torch.Tensor,
            prompt_lengths_cpu: torch.Tensor | None,
            first_request: int,
        ) -> LatentRolloutBatch:
            """Roll out every sample of a prompt chunk.

            ``prompt_lengths_cpu`` is None for an unpadded single prompt.
            """
            request_ids = range(first_request, first_request + prompt_ids.size(0))
            noise = {}
            if write_noise is not None:
                noise["write_noise_seeds"], noise["write_noise_scale"] = (
                    write_noise.inputs(
                        pool_seed, request_ids, args.samples_per_prompt, device
                    )
                )
            if decode_arena is not None:
                return decode_arena.rollout(
                    prompt_ids,
                    (
                        torch.full((1,), prompt_ids.size(1))
                        if prompt_lengths_cpu is None
                        else prompt_lengths_cpu
                    ),
                    prompt_repeats=args.samples_per_prompt,
                    max_new_tokens=train_max_new_tokens,
                    max_stream_steps=max_stream_steps,
                    token_seeds=trajectory_token_seeds(
                        pool_seed,
                        request_ids,
                        args.samples_per_prompt,
                        device,
                    ),
                    # Straight from the arena to the host: a device copy of
                    # the memory carry's replay record does not fit beside
                    # the arena on a 32 GB card.
                    offload_device=cpu if offload_to_cpu else None,
                    **noise,
                )
            return rollout_continuations(
                wrapper, prompt_ids, train_max_new_tokens, max_stream_steps,
                args.temperature, args.top_p, stop_ids=stop_ids or None,
                prompt_lengths=(
                    None
                    if prompt_lengths_cpu is None
                    else prompt_lengths_cpu.to(device)
                ),
                pin_emit=pin_emit,
                hidden_carry=hidden_carry,
                record_likelihoods=False,
                cache_dtype=torch.bfloat16,
                sync_every=args.rollout_sync_every,
                compact_dead_ratio=args.rollout_compact_dead_ratio,
                tensor_positions=rollout_step_core is not None,
                compact_finished=(
                    rollout_step_core is None
                    or args.rollout_tail_batch > 0
                ),
                finished_batch_size=(
                    args.rollout_tail_batch
                    if rollout_step_core is not None
                    and args.rollout_tail_batch > 0
                    else None
                ),
                prompt_repeats=args.samples_per_prompt,
                caches=(
                    rollout_graph_caches
                    if rollout_step_core is not None
                    else None
                ),
                tail_caches=rollout_tail_caches,
                tail_step_core=rollout_tail_step_core,
                decode_mask=decode_mask_for(prompt_ids.size(1)),
                tail_decode_mask=rollout_tail_decode_mask,
                **noise,
            )

        if args.rollout_groups <= 1:
            for request_id, (row, encoded) in enumerate(encoded_rows):
                prompt_ids = torch.tensor(
                    encoded,
                    dtype=torch.long, device=device,
                )
                with profiler.phase("decode"):
                    batch = lockstep_rollout(prompt_ids[None], None, request_id)
                if offload_to_cpu and decode_arena is None:
                    with profiler.phase("stream_d2h"):
                        batch = compact_stream_to_device(batch, cpu)
                with profiler.phase("score_inline"):
                    groups.append(retain_group(batch, row))
                del batch
            return groups
        # Left-padded batched rollout: all chunk groups step together, so
        # each launch carries chunk*samples rows instead of samples — the
        # sequential per-group loop is launch-bound, not compute-bound.
        samples = args.samples_per_prompt
        pending_scored: list = []

        def drain_scored() -> None:
            # Each future carries one chunk's worth of retained groups, in
            # chunk order. Timed as a phase because this is where a scoring
            # worker that has fallen behind the GPU shows up: the wait is
            # host time inside collect that no rollout kernel covers.
            with profiler.phase("score_wait"):
                for future in pending_scored:
                    groups.extend(future.result())
                pending_scored.clear()

        def split_and_retain_chunk(batched, lengths, chunk_rows) -> list:
            """Split one host-resident chunk into groups and retain each.

            Runs on the scoring worker so the whole CPU tail of a chunk
            overlaps the next chunk's rollout instead of stalling the GPU.
            """
            # Timed as a worker span, not a phase: this runs concurrently
            # with the next chunk's decode, and decode is launch-bound, so
            # worker CPU held here is time the host cannot spend launching.
            with profiler.worker_span("score"):
                split_groups = split_rollout_groups(batched, samples, lengths)
                retained = []
                for index, (group, row) in enumerate(
                    zip(split_groups, chunk_rows, strict=True)
                ):
                    retained.append(retain_group(group, row))
                    # Release each source as soon as it is retained so the
                    # chunk's cloned storage does not stay live to the end.
                    split_groups[index] = None
                    del group
            return retained

        def complete_chunk(
            batched: LatentRolloutBatch,
            prompt_lengths_cpu: torch.Tensor,
            chunk: list[dict],
            *,
            already_offloaded: bool = False,
        ) -> None:
            if offload_to_cpu and not already_offloaded:
                # One chunk-level D2H transfer, then all variable-length
                # splitting, trimming, decoding, and scoring stay on CPU.
                with profiler.phase("stream_d2h"):
                    batched = compact_stream_to_device(batched, cpu)
            expanded_prompt_lengths = (
                prompt_lengths_cpu.repeat_interleave(samples)
            )
            if not offload_to_cpu:
                expanded_prompt_lengths = expanded_prompt_lengths.to(device)
            if scoring_pool is None:
                with profiler.phase("score_inline"):
                    split_groups = split_rollout_groups(
                        batched, samples, expanded_prompt_lengths
                    )
                    del batched
                    for group_index, (group, row) in enumerate(
                        zip(split_groups, chunk, strict=True)
                    ):
                        groups.append(retain_group(group, row))
                        split_groups[group_index] = None
                        del group
                    del split_groups
                return

            # Draining the previous chunk before submitting this one preserves
            # chunk order and bounds host storage at one pending chunk.
            drain_scored()
            pending_scored.append(
                scoring_pool.submit(
                    split_and_retain_chunk,
                    batched,
                    expanded_prompt_lengths,
                    chunk,
                )
            )

        encoded_chunks = [
            encoded_rows[start : start + rollout_chunk_groups]
            for start in range(0, len(encoded_rows), rollout_chunk_groups)
        ]

        def upload_chunk(
            encoded_chunk: list[tuple[dict, list[int]]],
            width: int,
        ) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
            chunk = [row for row, _ in encoded_chunk]
            encoded = [ids for _, ids in encoded_chunk]
            prompt_ids = torch.zeros((len(chunk), width), dtype=torch.long)
            for index, ids in enumerate(encoded):
                prompt_ids[index, width - len(ids):] = torch.tensor(
                    ids, dtype=torch.long
                )
            prompt_ids = prompt_ids.to(device)
            prompt_lengths_cpu = torch.tensor(
                [len(ids) for ids in encoded], dtype=torch.long
            )
            return chunk, prompt_ids, prompt_lengths_cpu

        if args.rollout_scheduler == "continuous_refill":
            # One physical lane pool remains full while later prompt groups
            # are pending. Every admitted request keeps its own logical
            # position, request-local RNG stream, and paged live KV prefix.
            # A common prompt width retains the ordinary rollout's left-pad
            # position convention. One pool-wide prefill bank traverses each
            # unique prompt once; refill only fans its cached KV/state into
            # sample lanes.
            common_width = max(len(ids) for _, ids in encoded_rows)
            with profiler.phase("prompt_upload"):
                uploaded = [
                    upload_chunk(encoded_chunk, common_width)
                    for encoded_chunk in encoded_chunks
                ]
            schedule_stats = ContinuousScheduleStats()
            if rollout_paged_cache is None:
                # Static-batch FlexDecoding specializations retain their
                # first cache inputs. Own that arena explicitly and reuse it
                # across pools instead of attempting another multi-GiB
                # allocation. Stale suffixes are unreachable through each
                # request's live-page mask; every admitted prefix and future
                # write overwrites the reachable locations.
                rollout_paged_cache = wrapper.make_paged_generation_cache(
                    args.rollout_groups * samples,
                    args.prompt_tokens + max_stream_steps,
                    device,
                    dtype=torch.bfloat16,
                )
                if args.rollout_graph_decode:
                    # Graph capture bakes tensor addresses. The arena is
                    # mutated inside the compiled step (dense K/V writes),
                    # and inductor refuses to capture graphs whose mutated
                    # inputs are not statically addressed — without the
                    # marks it falls back to eager silently. page_home is
                    # read-only but constant for the run; marking it too
                    # saves its per-replay copy into the placeholder.
                    for cache_layer in rollout_paged_cache.layers:
                        for cache_tensor in cache_layer:
                            torch._dynamo.mark_static_address(cache_tensor)
                    torch._dynamo.mark_static_address(
                        rollout_paged_cache.page_home
                    )
                    with profiler.phase("decode_graph_warmup"):
                        warmed = warmup_decode_width_buckets(
                            wrapper,
                            rollout_paged_cache,
                            args.rollout_groups * samples,
                        )
                    print(
                        "rollout graph decode: warmed width buckets "
                        f"{warmed}; cudagraph_skips="
                        f"{torch._dynamo.utils.counters['inductor']['cudagraph_skips']}",
                        flush=True,
                    )
            with profiler.phase("decode"):
                scheduled_batches = rollout_continuous_refill_groups(
                    wrapper,
                    [item[1] for item in uploaded],
                    [item[2] for item in uploaded],
                    prompt_repeats=samples,
                    capacity_rows=args.rollout_groups * samples,
                    max_new_tokens=train_max_new_tokens,
                    max_stream_steps=max_stream_steps,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    # The sampler cursor is monotonic and checkpointed. Mixed
                    # with the user seed it gives each pool a reproducible,
                    # resume-stable request-key namespace.
                    seed=(args.seed << 32) ^ pool_start_cursor,
                    stop_ids=stop_ids or None,
                    cache_dtype=torch.bfloat16,
                    pin_emit=pin_emit,
                    offload_device=cpu if offload_to_cpu else device,
                    schedule_stats=schedule_stats,
                    paged_cache=rollout_paged_cache,
                )
            last_decode_schedule_metrics = schedule_stats.metrics()
            for batch_index, (chunk, _, prompt_lengths_cpu) in enumerate(
                uploaded
            ):
                batched = scheduled_batches[batch_index]
                complete_chunk(
                    batched,
                    prompt_lengths_cpu,
                    chunk,
                    already_offloaded=offload_to_cpu,
                )
                # complete_chunk's scoring future owns the batch until it has
                # split and retained its groups. Drop the scheduler list's
                # second reference now, otherwise every dense source chunk
                # survives alongside all split pool groups.
                scheduled_batches[batch_index] = None
                del batched
            drain_scored()
            return groups

        for chunk_index, encoded_chunk in enumerate(encoded_chunks):
            with profiler.phase("prompt_upload"):
                chunk_width = max(len(ids) for _, ids in encoded_chunk)
                chunk, prompt_ids, prompt_lengths_cpu = upload_chunk(
                    encoded_chunk, chunk_width
                )
            with profiler.phase("decode"):
                batched = lockstep_rollout(
                    prompt_ids,
                    prompt_lengths_cpu,
                    chunk_index * rollout_chunk_groups,
                )
            complete_chunk(
                batched,
                prompt_lengths_cpu,
                chunk,
                already_offloaded=offload_to_cpu and decode_arena is not None,
            )
            del batched
        drain_scored()
        return groups

    def training_autocast():
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        )

    def collect(
        refresh_statistics: bool = True,
        prompt_count: int | None = None,
        offload_to_cpu: bool = False,
        write_noise: WriteNoisePlan | None = None,
    ) -> list[LatentRolloutBatch]:
        if offload_to_cpu and refresh_statistics:
            raise ValueError(
                "CPU-offloaded collection must defer behavior refresh until "
                "optimizer minibatches are assembled"
            )
        original_step_core = wrapper.step_core
        original_paged_step_core = wrapper.paged_step_core
        if rollout_step_core is not None:
            wrapper.step_core = rollout_step_core
        if rollout_paged_step_core is not None:
            wrapper.paged_step_core = rollout_paged_step_core
        # Offloaded chunks are host tensors after compact_stream_to_device's
        # barrier, so one worker thread can trim and score chunk N (the
        # tokenizer's decode releases the GIL) while the launch-bound rollout
        # loop feeds chunk N+1 to the GPU. On-device chunks stay sequential:
        # worker tensor ops would enqueue D2H syncs into the rollout stream.
        scoring_pool = (
            ThreadPoolExecutor(max_workers=1) if offload_to_cpu else None
        )
        try:
            # rollout_continuations itself is no-grad, while the optional
            # refresh below must remain grad-enabled so refresh/update share
            # one compiled replay specialization and identical PPO numerics.
            with profiler.phase("collect"), training_autocast():
                return _collect(
                    refresh_statistics,
                    args.prompts_per_rollout
                    if prompt_count is None
                    else prompt_count,
                    offload_to_cpu,
                    scoring_pool,
                    write_noise,
                )
        finally:
            if scoring_pool is not None:
                scoring_pool.shutdown(wait=True)
            wrapper.step_core = original_step_core
            wrapper.paged_step_core = original_paged_step_core

    def training_update(*update_args, **update_kwargs):
        if args.mean_actor_critic:
            update_kwargs.update(
                mean_actor_critic=True,
                mac_estimator=args.mac_estimator,
                mac_vocab_chunk=args.mac_vocab_chunk,
                think_close_id=think_fence_ids[1],
            )
        if args.critic_q_pessimism and not update_kwargs.get("value_only", False):
            # The prior reads the policy from the actor's MAC pass, which a
            # value-only update does not run.
            update_kwargs["critic_q_pessimism"] = args.critic_q_pessimism
        if uses_post_action_q(args):
            update_kwargs["post_action_q_contract"] = {
                "think_fence_ids": think_fence_ids,
                "answer_fence_ids": answer_fence_ids,
                "eos_id": tokenizer.eos_id(),
                "stop_ids": stop_ids,
                "min_think_tokens": args.think_min_tokens,
                "max_new_tokens": train_max_new_tokens,
            }
        with training_autocast():
            return update_minibatch(*update_args, **update_kwargs)

    def teacher_forced_bpb() -> float:
        """Teacher-forced BPB through the deployed belief renderer."""
        wrapper.eval()
        val_loss, bpb = baseline.eval_val(
            FreshHyperparameters, wrapper, 0, 1, device, 8, val_tokens, *luts
        )
        if exact_val_byte_count is not None:
            bpb = (
                val_loss
                / math.log(2.0)
                * ((val_tokens.numel() - 1) / exact_val_byte_count)
            )
        wrapper.eval()
        return bpb

    def aime_eval(step: int) -> None:
        nonlocal rollout_step_core
        wrapper.eval()
        captured_attempts: list[dict[str, object]] = []
        eval_started = time.perf_counter()
        metrics = evaluate_aime_latent(
            wrapper, tokenizer, aime_rows, args.aime_samples,
            aime_max_new_tokens,
            aime_stream_steps, args.aime_chunk, args.seed, device,
            prompt_tokens=args.prompt_tokens,
            batch_trajectories=args.eval_batch_trajectories,
            compiled_step_core=None if eval_arena is not None else eval_step_core,
            compiled_tail_batch=args.eval_tail_batch or None,
            decode_arena=eval_arena,
            answer_style_override="aime",
            captured_attempts=captured_attempts,
            pin_emit=pin_emit,
            hidden_carry=hidden_carry,
            prompt_suffix_ids=answer_prefix_ids,
            think_fence_ids=think_fence_ids,
            answer_fence_ids=answer_fence_ids,
            min_think_tokens=args.think_min_tokens,
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
        )
        metrics["dataset_modal_answer"] = aime_modal_baseline["answer"]
        metrics["dataset_modal_answer_style"] = "aime"
        metrics["dataset_modal_answer_accuracy"] = aime_modal_baseline[
            "accuracy"
        ]
        if metrics["compile_fallback"] and eval_step_core is rollout_step_core:
            rollout_step_core = None
        eval_seconds = time.perf_counter() - eval_started
        tensorboard.add_scalar("perf/aime_eval_seconds", eval_seconds, step)
        logger.log(type="aime", step=step, seconds=eval_seconds, **metrics)
        write_benchmark_report(
            output / "aime_answers",
            step,
            metrics,
            captured_attempts,
            reward_schema=reward_schema_for_abstention(args.abstention_penalty),
        )
        tensorboard.add_scalar("aime/accuracy", metrics["accuracy"], step)
        tensorboard.add_scalar(
            "aime/policy_accuracy", metrics["policy_accuracy"], step
        )
        for key in ("abstained_fraction", "answered_accuracy"):
            if key in metrics:
                tensorboard.add_scalar(f"aime/{key}", metrics[key], step)
        print(
            f"step:{step} "
            f"aime_policy_avg@{metrics['policy_samples']}:"
            f"{metrics['policy_accuracy']:.4f}",
            flush=True,
        )

    def bench_eval(
        step: int,
        *,
        eval_seed: int | None = None,
        repeat: int | None = None,
    ) -> dict:
        nonlocal rollout_step_core
        wrapper.eval()
        if eval_seed is None:
            eval_seed = args.seed
        captured_attempts: list[dict[str, object]] = []
        eval_started = time.perf_counter()
        metrics = evaluate_aime_latent(
            wrapper, tokenizer, bench_rows, args.bench_samples,
            bench_max_new_tokens, bench_stream_steps, args.bench_samples,
            eval_seed, device, prompt_tokens=args.prompt_tokens,
            batch_trajectories=args.eval_batch_trajectories,
            compiled_step_core=None if eval_arena is not None else eval_step_core,
            compiled_tail_batch=args.eval_tail_batch or None,
            decode_arena=eval_arena,
            foreign_memory_reads=(
                args.memory_eval_ablation
                if args.memory_eval_ablation in FOREIGN_READ_ABLATIONS
                else None
            ),
            captured_attempts=captured_attempts,
            capture_problem_count=args.bench_capture_problems,
            capture_samples_per_problem=args.bench_capture_samples,
            pin_emit=pin_emit,
            hidden_carry=hidden_carry,
            prompt_suffix_ids=answer_prefix_ids,
            think_fence_ids=think_fence_ids,
            answer_fence_ids=answer_fence_ids,
            min_think_tokens=args.think_min_tokens,
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
        )
        metrics["dataset_modal_answer"] = bench_dataset_baseline["answer"]
        metrics["dataset_modal_answer_style"] = bench_dataset_baseline["style"]
        metrics["dataset_modal_answer_accuracy"] = bench_dataset_baseline[
            "accuracy"
        ]
        metrics["subset_modal_answer_accuracy"] = bench_subset_baseline[
            "accuracy"
        ]
        if metrics["compile_fallback"] and eval_step_core is rollout_step_core:
            rollout_step_core = None
        eval_seconds = time.perf_counter() - eval_started
        tensorboard.add_scalar("perf/bench_eval_seconds", eval_seconds, step)
        write_benchmark_report(
            output,
            step,
            metrics,
            captured_attempts,
            reward_schema=reward_schema_for_abstention(args.abstention_penalty),
            problem_count=args.bench_capture_problems,
            samples_per_problem=args.bench_capture_samples,
        )
        logger.log(
            type="bench",
            step=step,
            seconds=eval_seconds,
            eval_seed=eval_seed,
            repeat=repeat,
            memory_eval_ablation=args.memory_eval_ablation,
            **metrics,
        )
        tensorboard.add_scalar("bench/accuracy", metrics["accuracy"], step)
        tensorboard.add_scalar(
            "bench/policy_accuracy", metrics["policy_accuracy"], step
        )
        for key in ("abstained_fraction", "answered_accuracy"):
            if key in metrics:
                tensorboard.add_scalar(f"bench/{key}", metrics[key], step)
        for key, tag in (("contract_majority_accuracy", "majority_accuracy"),
                         ("contract_pass_at_samples", "pass_at_samples"),
                         ("contract_mean_log_kt_success", "mean_log_kt_success")):
            tensorboard.add_scalar(f"bench/{tag}", metrics[key], step)
        tensorboard.add_scalar(
            "bench/answer_cluster_entropy_mean",
            sum(metrics["prompt_answer_cluster_entropy"])
            / max(len(metrics["prompt_answer_cluster_entropy"]), 1),
            step,
        )
        print(
            f"step:{step} "
            f"bench_policy_avg@{metrics['policy_samples']}:"
            f"{metrics['policy_accuracy']:.4f}",
            flush=True,
        )
        return metrics

    if args.bench_only:
        wrapper.eval()
        # Restored bitwise on exit; nothing trains or saves in this branch.
        with memory_eval_ablation(wrapper, args.memory_eval_ablation):
            for repeat in range(args.bench_only_repeats):
                # Repeat zero warms the compiled decoder. Subsequent repeats
                # use matched but distinct workloads across evaluator modes,
                # avoiding a single stochastic tail being mistaken for
                # expected speed.
                bench_eval(
                    start_step,
                    eval_seed=args.seed + repeat,
                    repeat=repeat,
                )
        tensorboard.close()
        return

    if args.bpb_only:
        eval_started = time.perf_counter()
        bpb = teacher_forced_bpb()
        eval_seconds = time.perf_counter() - eval_started
        print(
            json.dumps(
                {
                    "type": "bpb",
                    "step": start_step,
                    "val_bpb": bpb,
                    "seconds": eval_seconds,
                    "val_tokens": val_tokens.numel() - 1,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        tensorboard.close()
        return

    if args.rollout_only and args.perturbation_sweep:
        sweep_config = SweepConfig(
            targets=tuple(args.perturbation_sweep),
            scales=tuple(args.perturbation_scales),
            random_directions=args.perturbation_random_directions,
            gradient_direction=args.perturbation_gradient_direction,
            radial_direction=args.perturbation_radial_direction,
            seed=args.seed,
            gradient_warmup_pools=args.perturbation_gradient_warmup_pools,
        )
        named_parameters = dict(wrapper.named_parameters())
        sweep_targets = {
            target: target_parameters(named_parameters, target)
            for target in sweep_config.targets
        }
        gradient_names = [
            name for parameters in sweep_targets.values() for name in parameters
        ]

        def collect_at(cursor: int) -> list[LatentRolloutBatch]:
            # Rewinding the cursor replays the pool's prompts and its
            # counter-keyed token noise (pool_seed is seed ^ cursor).
            sampler.cursor = cursor
            groups = collect(refresh_statistics=False, offload_to_cpu=True)
            torch.cuda.synchronize()
            return groups

        def token_gradient(
            groups: list[LatentRolloutBatch],
        ) -> tuple[dict[str, torch.Tensor], dict]:
            """The trainer's sampled-baseline actor gradient on one fresh
            pool, as a per-trajectory estimate of grad J at the targets.

            The actor loss is sum over emit slots of -(G - c) log pi divided
            by the emit-slot count, so grad J's REINFORCE estimate (mean over
            trajectories of sum over slots) is -emits / trajectories times
            the loss gradient. No optimizer steps.
            """
            packed = pack_rollout_groups_for_replay(groups, pin_memory=True)
            plan = build_replay_plan(
                packed,
                args.replay_max_trajectories,
                args.replay_attention_budget,
                args.replay_bucket,
                args.replay_slot_budget,
            )
            batch = packed.to(device, non_blocking=True)
            device_plan = plan.to(device, non_blocking=True)
            del packed, plan
            zero_optimizers(optimizers, "actor")
            zero_optimizers(optimizers, "critic")
            policy_denominator = batch.emit_mask.sum()
            metrics = training_update(
                wrapper, critic, batch, optimizers,
                actor_step=False,
                critic_step=False,
                policy_action_denominator=policy_denominator,
                value_action_denominator=actor_minibatch_action_denominator(
                    [batch], [0]
                ),
                gae_lambda_alpha=args.gae_lambda_alpha,
                actor_gae_lambda=args.actor_gae_lambda,
                replay_max_trajectories=args.replay_max_trajectories,
                replay_attention_budget=args.replay_attention_budget,
                replay_bucket=args.replay_bucket,
                replay_slot_budget=args.replay_slot_budget,
                replay_plan=device_plan,
                fused_behavior_statistics=True,
                reward_floor=reward_floor,
                critic_lambda=args.q_critic_lambda,
            )
            trajectories = batch.kind.size(0)
            emits = int(policy_denominator)
            scale = -emits / trajectories
            gradient = {}
            for name in gradient_names:
                grad = named_parameters[name].grad
                if grad is None:
                    raise RuntimeError(f"the actor update left no gradient at {name}")
                gradient[name] = grad.detach().double().cpu() * scale
            zero_optimizers(optimizers, "actor")
            zero_optimizers(optimizers, "critic")
            del batch, device_plan
            stats = {
                "trajectories": trajectories,
                "emit_slots": emits,
                "loss_to_grad_j_scale": scale,
                "gradient_norms": {
                    name: float(value.norm()) for name, value in gradient.items()
                },
                **{
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                    or (torch.is_tensor(value) and value.numel() == 1)
                },
            }
            return gradient, stats

        def sweep_log(values: dict) -> None:
            logger.log(**values)
            print(json.dumps(values, sort_keys=True), flush=True)

        sweep_writer = SweepWriter(output / "perturbation_sweep", {
            "schema": PERTURBATION_SWEEP_SCHEMA,
            "run_manifest_sha256": file_sha256(output / "manifest.json"),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.base_checkpoint_sha256,
            "policy_state": args.curriculum_init,
            "policy_state_sha256": args.initialization_checkpoint_sha256,
            "math_data_identity": data_identity,
            "reasoning_mode": args.reasoning_mode,
            "seed": args.seed,
            "start_cursor": sampler.cursor,
            "pools_planned": args.rollout_only_repeats,
            "prompts_per_rollout": args.prompts_per_rollout,
            "samples_per_prompt": args.samples_per_prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "reward_schema": reward_schema_for_abstention(args.abstention_penalty),
            "config": {
                "targets": list(sweep_config.targets),
                "scales": list(sweep_config.scales),
                "random_directions": sweep_config.random_directions,
                "gradient_direction": sweep_config.gradient_direction,
                "radial_direction": sweep_config.radial_direction,
                "gradient_warmup_pools": sweep_config.gradient_warmup_pools,
            },
            "token_gradient_estimator": (
                f"mac_{args.mac_estimator}" if sweep_config.gradient_direction else None
            ),
            "targets": {
                target: {
                    name: {
                        "shape": list(value.shape),
                        "rms": rms,
                    }
                    for (name, value), rms in zip(
                        parameters.items(), tensor_rms(parameters).values()
                    )
                }
                for target, parameters in sweep_targets.items()
            },
        })
        # --max-train-hours bounds the whole process: the sweep stops at the
        # last pool boundary its predicted next pool still fits before.
        deadline = (
            None if args.max_train_hours is None else args.max_train_hours * 3600
        )
        completed = run_sweep(
            sweep_config,
            sweep_targets,
            args.rollout_only_repeats,
            cursor=lambda: sampler.cursor,
            collect_at=collect_at,
            token_gradient=(
                token_gradient if sweep_config.gradient_direction else None
            ),
            writer=sweep_writer,
            out_of_time=lambda estimate: deadline is not None
            and time.monotonic() - run_started + estimate > deadline,
            log=sweep_log,
        )
        sweep_log({"type": "perturbation_sweep_done", "pools": completed})
        tensorboard.close()
        raise SystemExit(0 if completed else 2)

    if args.write_noise_sweep is not None:
        noise_config = WriteNoiseSweepConfig(scales=tuple(args.write_noise_sweep))

        def collect_with_noise(
            cursor: int, plan: WriteNoisePlan
        ) -> list[LatentRolloutBatch]:
            # Rewinding the cursor replays the pool's prompts, its token
            # noise and its write noise, all keyed on (seed, cursor).
            sampler.cursor = cursor
            groups = collect(
                refresh_statistics=False, offload_to_cpu=True, write_noise=plan
            )
            torch.cuda.synchronize()
            return groups

        def noise_log(values: dict) -> None:
            logger.log(**values)
            print(json.dumps(values, sort_keys=True), flush=True)

        noise_writer = SweepWriter(output / "write_noise_sweep", {
            "schema": WRITE_NOISE_SWEEP_SCHEMA,
            "run_manifest_sha256": file_sha256(output / "manifest.json"),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.base_checkpoint_sha256,
            "policy_state": args.curriculum_init,
            "policy_state_sha256": args.initialization_checkpoint_sha256,
            "math_data_identity": data_identity,
            "reasoning_mode": args.reasoning_mode,
            "seed": args.seed,
            "start_cursor": sampler.cursor,
            "pools_planned": args.rollout_only_repeats,
            "prompts_per_rollout": args.prompts_per_rollout,
            "samples_per_prompt": args.samples_per_prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "reward_schema": reward_schema_for_abstention(args.abstention_penalty),
            "scales": list(noise_config.scales),
            "noise": {
                "space": "absolute on writer pre-activations: pre-norm key, "
                "value, decay logit, write logit",
                "shared_per_prompt": True,
                "memory_heads": wrapper.combiner.heads,
                "memory_head_dim": wrapper.combiner.head_dim,
            },
        })
        deadline = (
            None if args.max_train_hours is None else args.max_train_hours * 3600
        )
        completed = run_write_noise_sweep(
            noise_config,
            args.rollout_only_repeats,
            cursor=lambda: sampler.cursor,
            collect_at=collect_with_noise,
            writer=noise_writer,
            out_of_time=lambda estimate: deadline is not None
            and time.monotonic() - run_started + estimate > deadline,
            log=noise_log,
        )
        noise_log({"type": "write_noise_sweep_done", "pools": completed})
        tensorboard.close()
        raise SystemExit(0 if completed else 2)

    if args.rollout_only:
        all_passed = True
        for repeat in range(args.rollout_only_repeats):
            if gate_transcripts is not None:
                gate_transcripts.repeat = repeat
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            groups = collect(
                refresh_statistics=False,
                offload_to_cpu=True,
            )
            torch.cuda.synchronize()
            if rollout_dataset is not None:
                rollout_dataset.finish_pool()
            # collect(refresh_statistics=False) above: no old_values exist.
            metrics = aggregate_diagnostics(
                groups,
                args.samples_per_prompt,
                stop_ids,
                refreshed_statistics=False,
                think_fence_ids=think_fence_ids,
                tokenizer=tokenizer,
                min_think_tokens=args.think_min_tokens,
                answer_fence_ids=answer_fence_ids,
                reward_floor=reward_floor,
            )
            if last_decode_schedule_metrics is None:
                metrics.update(
                    lockstep_decode_metrics(
                        groups, rollout_chunk_groups
                    )
                )
            else:
                metrics.update(last_decode_schedule_metrics)
            metrics["collect_seconds"] = time.perf_counter() - started
            metrics["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
            metrics["useful_actions_per_second"] = (
                metrics["trajectories"]
                * metrics["actions_per_trajectory"]
                / metrics["collect_seconds"]
            )
            per_source = (
                source_diagnostics(
                    groups,
                    source_names,
                    args.samples_per_prompt,
                    stop_ids,
                    refreshed_statistics=False,
                    think_fence_ids=think_fence_ids,
                    tokenizer=tokenizer,
                    min_think_tokens=args.think_min_tokens,
                    answer_fence_ids=answer_fence_ids,
                    reward_floor=reward_floor,
                )
                if mixture_manifest is not None
                else None
            )
            passed = (
                metrics["within_group_reward_std"]
                >= args.gate_min_within_group_reward_std
            )
            all_passed &= passed
            logger.log(
                type="rollout_gate",
                repeat=repeat,
                passed=passed,
                source_metrics=per_source,
                **metrics,
            )
            print(
                json.dumps(
                    {
                        "type": "rollout_gate",
                        "repeat": repeat,
                        "passed": bool(passed),
                        "source_metrics": per_source,
                        **metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del groups
        tensorboard.close()
        raise SystemExit(0 if all_passed else 2)

    if start_step == 0 and warmup_step == 0:
        bpb = teacher_forced_bpb()
        logger.log(type="bpb", step=0, val_bpb=bpb)
        tensorboard.add_scalar("guard/val_bpb", bpb, 0)
        print(f"step:0 teacher-forced val_bpb:{bpb:.4f}", flush=True)
        if aime_rows:
            aime_eval(0)
        if bench_rows:
            bench_eval(0)
    checkpoint_policy = RecoveryCheckpointPolicy(
        args.checkpoint_interval_seconds
    )
    # An action-head warmup fits only the head, against the carried values.
    warmup_scope = args.value_warmup_scope.replace("-", "_")
    warmup_role = "critic_action" if warmup_scope == "action_head" else "critic"
    if start_step == 0 and warmup_step < args.value_warmup_steps:
        for warmup in range(warmup_step + 1, args.value_warmup_steps + 1):
            warmup_started = time.perf_counter()
            collect_started = warmup_started
            # The warmup pool takes the actor update's path: offloaded at
            # collection, packed into one minibatch with one replay plan,
            # and fit in a single sharded pass. Updating prompt group by
            # prompt group instead ran one small replay per group, at more
            # than twice the cost of a whole actor-and-critic update.
            groups = collect(
                refresh_statistics=False,
                prompt_count=args.prompts_per_minibatch,
                offload_to_cpu=True,
            )
            torch.cuda.synchronize()
            collect_seconds = time.perf_counter() - collect_started
            update_started = time.perf_counter()
            packed = pack_rollout_groups_for_replay(groups, pin_memory=True)
            del groups
            replay_plan = build_replay_plan(
                packed,
                args.replay_max_trajectories,
                args.replay_attention_budget,
                args.replay_bucket,
                args.replay_slot_budget,
                include_action_indices=False,
            )
            device_batch = packed.to(device, non_blocking=True)
            device_replay_plan = replay_plan.to(device, non_blocking=True)
            del packed, replay_plan
            zero_optimizers(optimizers, warmup_role)
            metrics = training_update(
                wrapper,
                critic,
                device_batch,
                optimizers,
                value_only=True,
                critic_step=False,
                replay_max_trajectories=args.replay_max_trajectories,
                replay_attention_budget=args.replay_attention_budget,
                replay_bucket=args.replay_bucket,
                replay_slot_budget=args.replay_slot_budget,
                replay_plan=device_replay_plan,
                value_scope=warmup_scope,
            )
            step_optimizers(optimizers, warmup_role)
            del device_batch, device_replay_plan
            update_seconds = time.perf_counter() - update_started
            value_metrics = aggregate_value_diagnostics([metrics])
            metrics.update(
                seconds=time.perf_counter() - warmup_started,
                collect_seconds=collect_seconds,
                update_seconds=update_seconds,
                value_mean=value_metrics["prediction_mean"],
                value_target_mean=value_metrics["target_mean"],
                value_target_variance=value_metrics["target_variance"],
                value_residual_mean=value_metrics["residual_mean"],
                value_residual_variance=value_metrics["residual_variance"],
                explained_variance=value_metrics["explained_variance"],
                value_loss=value_metrics["loss"],
            )
            logger.log(
                type="value_warmup", step=warmup, scope=warmup_scope, **metrics
            )
            print(
                f"critic warmup:{warmup}/{args.value_warmup_steps} "
                f"collect:{collect_seconds:.2f}s update:{update_seconds:.2f}s "
                f"value_loss:{metrics['value_loss']:.6f}"
                + (
                    ""
                    if critic.action_head is None
                    else f" action_value_loss:{metrics['action_value_loss']:.6f}"
                    f" action_ev:{metrics['action_value_explained_variance']:.4f}"
                ),
                flush=True,
            )
            warmup_dashboard = {
                ("value_warmup/token_weighted_bce" if uses_post_action_q(args)
                 else "value_warmup/token_weighted_mse"): metrics["value_loss"],
                "value_warmup/prediction_mean": metrics["value_mean"],
                "value_warmup/target_mean": metrics["value_target_mean"],
                "value_warmup/residual_mean": metrics["value_residual_mean"],
                "value_warmup/online_explained_variance": metrics[
                    "explained_variance"
                ],
                "value_warmup/residual/action_mean": metrics[
                    "advantage_mean"
                ],
                "value_warmup/critic_grad_norm": metrics[
                    "critic_grad_norm"
                ],
            }
            warmup_dashboard.update({
                f"value_warmup/{tag}": value
                for tag, value in twin_dashboard([metrics]).items()
            })
            if critic.action_head is not None:
                warmup_dashboard.update(
                    {
                        "value_warmup/action_value_mse": metrics[
                            "action_value_loss"
                        ],
                        "value_warmup/action_value_online_explained_variance": (
                            metrics["action_value_explained_variance"]
                        ),
                        "value_warmup/action_head_grad_norm": metrics[
                            "action_head_grad_norm"
                        ],
                    }
                )
            for tag, value in warmup_dashboard.items():
                tensorboard.add_scalar(tag, value, warmup)
            # The optimizer state is sufficient for the next update and for
            # checkpoints. Keeping the accumulated gradient buffers alive
            # overlaps them with the next rollout's KV cache for no benefit.
            zero_optimizers(optimizers, warmup_role)
            warmup_step = warmup
            if warmup == args.value_warmup_steps and rewinds_warmup_prompts(args):
                # Recovery checkpoints inside the warmup keep the advanced
                # cursor so a resumed warmup draws fresh prompts; only the
                # boundary hands the actor the stream it started from.
                sampler.cursor = actor_init_provenance["target_sampler_cursor"]
                print(
                    f"critic warmup: prompt cursor rewound to {sampler.cursor} "
                    "for the first actor step",
                    flush=True,
                )
            checkpoint_state = (start_step, warmup_step, sampler.cursor)
            checkpoint_due = checkpoint_policy.due()
            if warmup == args.value_warmup_steps:
                # Preserve the clean actor-initial/critic-warm checkpoint as
                # a semantic boundary artifact. If recovery is also due, link
                # the identical serialized payload into the rolling pathname.
                critic_warm_path = output / "critic_warmup_checkpoint.pt"
                save_checkpoint(
                    critic_warm_path,
                    wrapper,
                    critic,
                    optimizers,
                    start_step,
                    args,
                    sampler,
                    warmup_step,
                    actor_init_provenance,
                    entropy_temperature=entropy_temperature,
                )
                atomic_link_or_copy(
                    critic_warm_path,
                    output / "latent_vapo_checkpoint.pt",
                )
                checkpoint_policy.committed(checkpoint_state)
            elif checkpoint_due:
                save_checkpoint(
                    output / "latent_vapo_checkpoint.pt",
                    wrapper,
                    critic,
                    optimizers,
                    start_step,
                    args,
                    sampler,
                    warmup_step,
                    actor_init_provenance,
                    entropy_temperature=entropy_temperature,
                )
                checkpoint_policy.committed(checkpoint_state)

    def crossed_interval(previous: int, current: int, interval: int) -> bool:
        return interval > 0 and current // interval > previous // interval

    step = start_step
    # Session-local by design: resuming after a desert stop is a deliberate
    # operator act, so the resumed session gets a fresh
    # --zero-reward-stop-pools budget rather than stopping immediately.
    zero_reward_pool_streak = 0
    if resumed_zero_reward_frozen_updates is None:
        zero_reward_frozen_updates = logged_zero_reward_frozen_updates(
            output / "metrics.jsonl", start_step
        )
    else:
        zero_reward_frozen_updates = int(resumed_zero_reward_frozen_updates)
        if not 0 <= zero_reward_frozen_updates <= start_step:
            raise ValueError(
                "resume checkpoint has invalid zero-reward freeze counter: "
                f"{zero_reward_frozen_updates} at actor step {start_step}"
            )
    stopped_at_pool_boundary = False
    # Session-local, like the constant-reward streak.
    bench_below_floor_evals = 0
    while step < args.steps:
        if (
            args.bench_stop_floor is not None
            and bench_below_floor_evals >= args.bench_stop_evals
        ):
            print(
                f"{bench_below_floor_evals} consecutive benchmark evaluations "
                f"below contract accuracy {args.bench_stop_floor} at step "
                f"{step}/{args.steps}; stopping at the pool boundary",
                flush=True,
            )
            logger.log(type="bench_stop", step=step, floor=args.bench_stop_floor,
                       evaluations=bench_below_floor_evals)
            stopped_at_pool_boundary = True
            break
        if (
            args.zero_reward_stop_pools > 0
            and zero_reward_pool_streak >= args.zero_reward_stop_pools
        ):
            # A sustained constant-reward streak is a dead run: with the
            # actor frozen it cannot recover, and with the freeze disabled
            # it is re-entering the round-4 spiral. Pool-boundary stop —
            # the loop tail below saves the final checkpoint, so the run
            # truncates resumably instead of burning the remaining budget
            # on signal-free rollouts.
            print(
                f"{zero_reward_pool_streak} consecutive constant-reward "
                f"pools at step {step}/{args.steps}; stopping at the pool "
                "boundary",
                flush=True,
            )
            stopped_at_pool_boundary = True
            break
        if (
            args.max_train_hours is not None
            and time.monotonic() - run_started > args.max_train_hours * 3600
        ):
            # Pool-boundary stop: the loop tail below already saves the
            # final checkpoint, so a deadline exit is an ordinary truncation
            # of a constant-learning-rate run, resumable like any other.
            print(
                f"--max-train-hours {args.max_train_hours} reached at step "
                f"{step}/{args.steps}; stopping at the pool boundary",
                flush=True,
            )
            stopped_at_pool_boundary = True
            break
        previous_step = step
        # Before the clock starts: attaching the profiler's own machinery
        # would otherwise land inside this pool's wall time and inside its
        # unaccounted remainder, making traced and untraced pools
        # incomparable on the one row that matters.
        profiler.pool_started(step)
        started = time.perf_counter()
        collect_started = started
        requested_pool_updates = min(
            args.prompts_per_rollout // args.prompts_per_minibatch,
            args.steps - step,
        )
        pool_prompt_count = (
            requested_pool_updates * args.prompts_per_minibatch
        )
        if args.consume_all_prompts:
            pool_prompt_count = min(
                pool_prompt_count, len(math_rows) - sampler.cursor
            )
        if pool_prompt_count < 1:
            raise RuntimeError("actor steps remain but no unused prompts are available")
        pool_updates = math.ceil(
            pool_prompt_count / args.prompts_per_minibatch
        )
        torch.cuda.reset_peak_memory_stats()
        capture_rollouts = capture_due(
            step, pool_updates, args.rollout_sample_every
        )
        if capture_rollouts:
            rollout_recorder.begin(step, metrics_step=step + pool_updates)
        groups = collect(
            refresh_statistics=False,
            prompt_count=pool_prompt_count,
            offload_to_cpu=True,
        )
        if capture_rollouts:
            # collect() has joined its scoring worker: every group is in.
            write_rollout_samples(output, rollout_recorder.finish())
        with profiler.phase("rollout_diagnostics"):
            rollout_metrics = aggregate_diagnostics(
                groups,
                args.samples_per_prompt,
                stop_ids,
                refreshed_statistics=False,
                think_fence_ids=think_fence_ids,
                tokenizer=tokenizer,
                min_think_tokens=args.think_min_tokens,
                answer_fence_ids=answer_fence_ids,
                reward_floor=reward_floor,
            )
            if last_decode_schedule_metrics is None:
                rollout_metrics.update(
                    lockstep_decode_metrics(groups, rollout_chunk_groups)
                )
            else:
                rollout_metrics.update(last_decode_schedule_metrics)
            # The profile's launch counts need this denominator to price the
            # decode loop per step without a join against the metrics stream.
            profiler.note_counter(
                "decode_steps", rollout_metrics.get("decode_steps_total", 0.0)
            )
            source_rollout_metrics = (
                source_diagnostics(
                    groups,
                    source_names,
                    args.samples_per_prompt,
                    stop_ids,
                    refreshed_statistics=False,
                    think_fence_ids=think_fence_ids,
                    tokenizer=tokenizer,
                    min_think_tokens=args.think_min_tokens,
                    answer_fence_ids=answer_fence_ids,
                    reward_floor=reward_floor,
                )
                if mixture_manifest is not None
                else None
            )
            allow_partial_final = (
                args.consume_all_prompts and sampler.cursor == len(math_rows)
            )
            if mixture_sources is not None:
                minibatch_orders = stratified_optimizer_minibatch_orders(
                    groups,
                    args.prompts_per_minibatch,
                    len(mixture_sources),
                    allow_partial_final=allow_partial_final,
                )
            else:
                minibatch_orders = optimizer_minibatch_orders(
                    len(groups),
                    args.prompts_per_minibatch,
                    allow_partial_final=allow_partial_final,
                )
        if len(minibatch_orders) != pool_updates:
            raise RuntimeError("rollout pool did not produce the planned updates")
        # A pool whose every trajectory scored the same reward carries no
        # reward signal: every advantage is critic error. All-wrong and
        # all-abstain pools are constant too. The actual freeze decision is
        # made per optimizer
        # MINIBATCH below (an update over a constant-reward minibatch is the
        # harmful unit — a single differently rewarded trajectory elsewhere
        # in the pool must not unfreeze it); the pool-level statistic here
        # feeds the rollout log and the consecutive-desert stop. NaN fails
        # closed: a non-finite mean is a broken pool, not a licence to update.
        if not math.isfinite(rollout_metrics["reward_mean"]):
            raise RuntimeError(
                f"non-finite pool reward_mean "
                f"{rollout_metrics['reward_mean']} at step {step}"
            )
        pool_reward_zero = (
            rollout_metrics["reward_max"] == rollout_metrics["reward_min"]
        )
        rollout_metrics["zero_reward_actor_frozen"] = float(
            args.zero_reward_actor_freeze and pool_reward_zero
        )
        if pool_reward_zero:
            zero_reward_pool_streak += 1
            frozen_note = (
                f"actor optimizer frozen for its {pool_updates} updates "
                "(critic continues)"
                if args.zero_reward_actor_freeze
                else "freeze disabled by --no-zero-reward-actor-freeze"
            )
            print(
                f"steps {step + 1}..{step + pool_updates}: constant-reward "
                f"pool — {frozen_note}; streak {zero_reward_pool_streak}",
                flush=True,
            )
        else:
            zero_reward_pool_streak = 0

        # Assemble each shuffled optimizer minibatch on CPU with RIGHT tail
        # padding, then refresh all old statistics under the still-frozen
        # behavior policy. The refreshed DEVICE batch is retained and consumed
        # directly by the matching update below — repacking and re-uploading
        # the identical order and width, plus the d2h stat scatter, was pure
        # round-trip overhead. Retention is bounded: past the budget (streams
        # lengthen when thinking grows) the pre-retention path takes over —
        # scatter stats to the CPU groups and let update repack.
        #
        # A pool consumed by one optimizer minibatch is updated at behavior
        # age 0, where the refresh forward would repeat the update's own
        # forward on the same parameters. The update then derives the
        # behavior statistics itself (``fused_behavior_statistics``) and
        # this loop only packs and uploads. Latent thoughts and TPO log-odds
        # have no fused form and keep the refresh.
        fused_behavior_statistics = (
            len(minibatch_orders) == 1
            and args.reasoning_mode != "latent"
            and not args.target_policy_optimization
        )
        retained_device_batches: dict[int, LatentRolloutBatch] = {}
        device_replay_plans: dict[int, ReplayPlan] = {}
        replay_layout_tokens = {
            tuple(minibatch_order): object()
            for minibatch_order in minibatch_orders
        }
        retained_bytes_total = 0
        packed_batch_bytes_max = 0
        packed_real_slot_counts = []
        packed_capacity_slots = 0
        replay_computed_slots = 0
        replay_shards = 0
        pool_cpu_pack_seconds = 0.0
        # Transfer and replay are asynchronous, so these are event pairs read
        # after the single barrier at the end of collection. Wall clocks here
        # used to need a synchronize per minibatch, which drained the pipeline
        # eight times a pool purely to read a clock.
        h2d_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        refresh_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        d2h_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        old_value_sums = []
        old_value_counts = []
        def pack_minibatch(
            minibatch_order: list[int],
        ) -> tuple[
            LatentRolloutBatch,
            ReplayPlan,
            list[LatentRolloutBatch],
            float,
        ]:
            selected = [groups[index] for index in minibatch_order]
            pack_started = time.perf_counter()
            with profiler.worker_span("pack"):
                packed = pack_rollout_groups_for_replay(
                    selected, pin_memory=True
                )
                packed.replay_layout_token = replay_layout_tokens[
                    tuple(minibatch_order)
                ]
                replay_plan = build_replay_plan(
                    packed,
                    args.replay_max_trajectories,
                    args.replay_attention_budget,
                    args.replay_bucket,
                    args.replay_slot_budget,
                )
            return (
                packed,
                replay_plan,
                selected,
                time.perf_counter() - pack_started,
            )

        # Minibatch orders are a disjoint partition of the pool, so a single
        # worker can pack minibatch N+1's pinned host batch while N's H2D
        # and refresh occupy the GPU — the worker only reads groups this
        # loop's scatter fallback never writes in the same iteration.
        pack_pool = ThreadPoolExecutor(max_workers=1)
        try:
            with profiler.phase("refresh_pipeline"):
                pack_future = pack_pool.submit(
                    pack_minibatch, minibatch_orders[0]
                )
                for age, minibatch_order in enumerate(minibatch_orders):
                    with profiler.phase("pack_wait"):
                        (
                            cpu_batch,
                            cpu_replay_plan,
                            selected_groups,
                            pack_elapsed,
                        ) = pack_future.result()
                    pool_cpu_pack_seconds += pack_elapsed
                    if age + 1 < len(minibatch_orders):
                        pack_future = pack_pool.submit(
                            pack_minibatch, minibatch_orders[age + 1]
                        )
                    with profiler.phase("h2d"), device_timed(h2d_events):
                        device_batch = cpu_batch.to(device, non_blocking=True)
                        device_replay_plan = cpu_replay_plan.to(
                            device, non_blocking=True
                        )
                    # Safe without a barrier: the source is pinned, so it came
                    # from the caching host allocator, which records a CUDA
                    # event when the block is freed and withholds it from
                    # reuse until the copy retires.
                    del cpu_batch
                    del cpu_replay_plan
                    device_replay_plans[age] = device_replay_plan
                    if not fused_behavior_statistics:
                        with (
                            profiler.phase("refresh"),
                            device_timed(refresh_events),
                            training_autocast(),
                        ):
                            refresh_old_statistics(
                                wrapper,
                                critic,
                                device_batch,
                                max_trajectories=args.replay_max_trajectories,
                                attention_budget=args.replay_attention_budget,
                                bucket_multiple=args.replay_bucket,
                                slot_budget=args.replay_slot_budget,
                                replay_plan=device_replay_plan,
                                target_policy_optimization=(
                                    args.target_policy_optimization
                                ),
                            )
                    packed_batch_bytes = sum(
                        value.numel() * value.element_size()
                        for field in fields(device_batch)
                        if isinstance(
                            (value := getattr(device_batch, field.name)),
                            torch.Tensor,
                        )
                    )
                    packed_batch_bytes_max = max(
                        packed_batch_bytes_max, packed_batch_bytes
                    )
                    # Reduced on device and read once after the pool's single
                    # barrier; int() here was a blocking sync per minibatch.
                    packed_real_slot_counts.append(
                        (device_batch.kind != PAD_SLOT).sum()
                    )
                    packed_capacity_slots += device_batch.kind.numel()
                    # What each replay pass actually computes: shards are
                    # length-sorted and trimmed to their own bucketed width,
                    # so this, not the packed width, is the update's waste.
                    replay_computed_slots += sum(
                        spec.row_count * spec.stream_length
                        for spec in device_replay_plan.specs
                    )
                    replay_shards += len(device_replay_plan.specs)
                    if not fused_behavior_statistics:
                        generated = device_batch.action_mask.bool()
                        # masked_fill, not boolean indexing: the latter lowers
                        # to masked_select, whose output size is data
                        # dependent, so it blocks the host on every minibatch.
                        # masked_fill also keeps a non-finite PAD slot out of
                        # the sum, which multiplying by the mask would not.
                        old_value_sums.append(
                            device_batch.old_values.masked_fill(
                                ~generated, 0.0
                            ).sum()
                        )
                        old_value_counts.append(generated.sum())
                    # A fused pool is one minibatch that the update holds on
                    # device regardless, so a repack would only upload it
                    # twice.
                    if (
                        fused_behavior_statistics
                        or retained_bytes_total + packed_batch_bytes
                        <= RETAINED_MINIBATCH_BUDGET_BYTES
                    ):
                        retained_device_batches[age] = device_batch
                        retained_bytes_total += packed_batch_bytes
                    elif not fused_behavior_statistics:
                        with (
                            profiler.phase("stat_scatter_d2h"),
                            device_timed(d2h_events),
                        ):
                            scatter_replay_statistics(
                                device_batch, selected_groups
                            )
                    del device_batch
                    del device_replay_plan
                    del selected_groups
        finally:
            pack_pool.shutdown(wait=True)
        # A phase of its own because this is where every kernel the pool
        # enqueued and has not retired actually drains. Folded into the
        # remainder it would be the largest anonymous entry in the table.
        with profiler.phase("pool_barrier"):
            # The only barrier this loop takes on its own account. Every
            # device reduction and event pair below is read after it, so no
            # per-minibatch metric drains the pipeline. Whether refresh
            # itself still blocks inside the shard planner is a separate
            # question, and one --profile answers rather than assumes.
            torch.cuda.synchronize()
        collect_seconds = time.perf_counter() - collect_started
        with profiler.phase("pool_reductions"):
            if not fused_behavior_statistics:
                # Fused pools have no pre-update value pass; the update's
                # train/value_mean is the same quantity.
                old_value_sum = float(torch.stack(old_value_sums).sum())
                old_value_count = int(torch.stack(old_value_counts).sum())
                rollout_metrics["old_value_mean"] = (
                    old_value_sum / old_value_count if old_value_count else 0.0
                )
            rollout_metrics["fused_behavior_statistics"] = float(
                fused_behavior_statistics
            )
            rollout_metrics["retained_minibatches"] = len(
                retained_device_batches
            )
            rollout_metrics["packed_batch_gib_max"] = (
                packed_batch_bytes_max / 2**30
            )
            packed_real_slots = int(torch.stack(packed_real_slot_counts).sum())
            rollout_metrics["packed_padding_utilization"] = (
                packed_real_slots / packed_capacity_slots
                if packed_capacity_slots
                else 0.0
            )
            rollout_metrics["replay_slot_utilization"] = (
                packed_real_slots / replay_computed_slots
                if replay_computed_slots
                else 0.0
            )
            rollout_metrics["replay_shards"] = replay_shards
            rollout_metrics["pool_cpu_pack_seconds"] = pool_cpu_pack_seconds
            # Device time on the transfer and replay streams, not host wall
            # time.
            rollout_metrics["pool_h2d_seconds"] = elapsed_seconds(h2d_events)
            if not fused_behavior_statistics:
                rollout_metrics["pool_refresh_seconds"] = elapsed_seconds(
                    refresh_events
                )
            rollout_metrics["pool_d2h_seconds"] = elapsed_seconds(d2h_events)
        rollout_peak_vram_bytes = torch.cuda.max_memory_allocated()
        # Restart the peak counter here so the train rows below report the
        # UPDATE-phase peak (packed minibatch + replay-shard activations),
        # not the collection-phase KV-cache peak that always dominates it.
        # Replay memory knobs (--replay-attention-budget and retained device
        # batches) are tuned against this number.
        torch.cuda.reset_peak_memory_stats()
        rollout_step = step + pool_updates
        with profiler.phase("pool_logging"):
            logger.log(
                type="rollout", step=rollout_step,
                collect_seconds=collect_seconds,
                pool_updates=pool_updates,
                pool_trajectories=pool_prompt_count * args.samples_per_prompt,
                peak_vram_bytes=rollout_peak_vram_bytes,
                source_metrics=source_rollout_metrics,
                **rollout_metrics,
            )
            tensorboard.add_scalar(
                "perf/collect_seconds", collect_seconds, rollout_step
            )
            tensorboard.add_scalar(
                "perf/rollout_peak_vram_gib",
                rollout_peak_vram_bytes / 2**30,
                rollout_step,
            )
            tensorboard.add_scalar(
                "perf/pool_cpu_pack_seconds",
                pool_cpu_pack_seconds,
                rollout_step,
            )
            for tag in (
                "packed_batch_gib_max",
                "packed_padding_utilization",
                "replay_slot_utilization",
                "replay_shards",
                "decode_steps_per_chunk_mean",
                "decode_step_utilization",
                "decode_step_savings_fraction",
                "pool_h2d_seconds",
                "pool_refresh_seconds",
                "pool_d2h_seconds",
            ):
                if tag in rollout_metrics:
                    tensorboard.add_scalar(
                        f"perf/{tag}", rollout_metrics[tag], rollout_step
                    )
            rollout_dashboard = rollout_tensorboard_metrics(rollout_metrics)
            # Horizontal anti-collapse calibration: on DAPO, a terminated
            # constant modal answer already earns nontrivial exact and shaped
            # reward. On-policy improvement is meaningful only relative to
            # both.
            rollout_dashboard["reward/modal_constant_exact_baseline"] = float(
                math_modal_baseline["accuracy"]
            )
            rollout_dashboard["reward/modal_constant_shaped_baseline"] = float(
                math_modal_baseline["shaped_reward"]
            )
            for tag, value in rollout_dashboard.items():
                tensorboard.add_scalar(tag, value, rollout_step)
            if source_rollout_metrics is not None:
                for source_name, source_metrics in source_rollout_metrics.items():
                    for metric_name in (
                        "reward_mean",
                        "exact_accuracy",
                        "within_group_reward_std",
                        "ended_fraction",
                        "verifier_abstained_fraction",
                        "think_format_fraction",
                        "actions_per_trajectory",
                        "length_reward_covariance",
                        "think_gate_zeroed_correct_fraction",
                    ):
                        if metric_name in source_metrics:
                            tensorboard.add_scalar(
                                f"source/{source_name}/{metric_name}",
                                source_metrics[metric_name],
                                rollout_step,
                            )

        for behavior_age, minibatch_order in enumerate(minibatch_orders):
            # Actor and critic each take one optimizer step over the same
            # effective trajectory minibatch. Prompt groups and replay shards
            # are memory partitions, not optimizer minibatches.
            with profiler.phase("update"):
                update_started = time.perf_counter()
                next_step = step + 1
                zero_optimizers(optimizers, "actor")
                zero_optimizers(optimizers, "critic")
                combiner_parameters = list(wrapper.combiner.parameters())
                combiner_before = [
                    parameter.detach().clone()
                    for parameter in combiner_parameters
                ]
                selected_groups = [groups[index] for index in minibatch_order]
                # Popped straight into the one name the update releases: an
                # alias would keep the packed device batch (4.25 GiB on the
                # memory drill) alive through the next pool's rollout.
                device_minibatch = retained_device_batches.pop(
                    behavior_age, None
                )
                device_replay_plan = device_replay_plans.pop(behavior_age)
                minibatch_h2d_events = []
                if device_minibatch is not None:
                    # Refreshed on-device in the pool loop above; identical to
                    # what repacking selected_groups would rebuild.
                    minibatch_pack_seconds = 0.0
                else:
                    with profiler.phase("minibatch_pack"):
                        minibatch_pack_started = time.perf_counter()
                        cpu_minibatch = pack_rollout_groups_for_replay(
                            selected_groups, pin_memory=True
                        )
                        cpu_minibatch.replay_layout_token = (
                            replay_layout_tokens[tuple(minibatch_order)]
                        )
                        minibatch_pack_seconds = (
                            time.perf_counter() - minibatch_pack_started
                        )
                    with (
                        profiler.phase("minibatch_h2d"),
                        device_timed(minibatch_h2d_events),
                    ):
                        device_minibatch = cpu_minibatch.to(
                            device, non_blocking=True
                        )
                    # See the pool-loop transfer: freeing pinned host storage
                    # with the copy in flight is safe, the caching host
                    # allocator defers reuse behind a recorded event.
                    del cpu_minibatch
                value_action_denominator = (
                    actor_minibatch_action_denominator(
                        [device_minibatch], [0]
                    )
                )
                policy_action_denominator = (
                    device_minibatch.emit_mask.sum()
                    if (
                        args.delightful_policy_gradient
                        or args.target_policy_optimization
                        or args.mean_actor_critic
                    )
                    else value_action_denominator
                )
                if frozen_moe_router_parameters:
                    reset_quantile_balance_accumulators(backbone)
                with profiler.phase("forward_backward"):
                    metrics = training_update(
                        wrapper, critic, device_minibatch, optimizers,
                        actor_step=False,
                        critic_step=False,
                        policy_action_denominator=policy_action_denominator,
                        value_action_denominator=value_action_denominator,
                        gae_lambda_alpha=args.gae_lambda_alpha,
                        actor_gae_lambda=args.actor_gae_lambda,
                        replay_max_trajectories=args.replay_max_trajectories,
                        replay_attention_budget=args.replay_attention_budget,
                        replay_bucket=args.replay_bucket,
                        replay_slot_budget=args.replay_slot_budget,
                        replay_plan=device_replay_plan,
                        delightful_policy_gradient=(
                            args.delightful_policy_gradient
                        ),
                        target_policy_optimization=(
                            args.target_policy_optimization
                        ),
                        tpo_eta=args.tpo_eta,
                        source_success_actor_gate=(
                            args.source_success_actor_gate
                        ),
                        fused_behavior_statistics=fused_behavior_statistics,
                        reward_floor=reward_floor,
                        critic_lambda=args.q_critic_lambda,
                        entropy_temperature=(
                            None
                            if entropy_temperature is None
                            else entropy_temperature.temperature
                        ),
                        advantage_source_count=len(source_names),
                        advantage_token_classes=advantage_token_classes,
                    )
                # The first minibatch runs against refresh-computed behavior
                # statistics with the actor untouched. Later disjoint minibatches
                # intentionally have behavior age 1..N.
                if behavior_age == 0:
                    guard = (
                        "token_abs_log_ratio_max"
                        if (
                            args.target_policy_optimization
                            or args.delightful_policy_gradient
                        )
                        else "policy_clip_fraction"
                    )
                    if metrics[guard] > 1e-6:
                        if args.target_policy_optimization:
                            raise RuntimeError(
                                f"step {next_step}: behavior-age-0 {guard}="
                                f"{metrics[guard]:.3e}; TPO refresh and "
                                "update replay paths diverged"
                            )
                        print(
                            f"WARNING step {next_step}: behavior-age-0 {guard}="
                            f"{metrics[guard]:.3e} (expected exactly 0; "
                            "refresh/update code paths diverged)",
                            flush=True,
                        )
                # An update whose every trajectory scored the same reward
                # (all wrong, all abstaining, or all correct) is the
                # harmful unit from the round-4 postmortem: its advantages
                # are pure critic error whose only coherent direction is
                # anti-termination. The decision is per MINIBATCH — a
                # rewarded trajectory elsewhere in the pool must not
                # unfreeze an all-zero update. Only the actor OPTIMIZER
                # step is skipped: the forward/backward above already ran,
                # so behavior-age canaries, gradient telemetry, and the
                # non-finite checks stay uniform, and skipping the step is
                # required because Adam momentum from earlier updates moves
                # weights even on a zero-signal gradient. The critic always
                # steps so value predictions catch up to the reward
                # targets. (Rewards are finite here: the pool-level
                # isfinite raise above covers the same tensors.)
                minibatch_actor_frozen = (
                    args.zero_reward_actor_freeze
                    and metrics["reward_max"] == metrics["reward_min"]
                )
                if minibatch_actor_frozen:
                    zero_reward_frozen_updates += 1
                minibatch_metrics = [metrics]
                # Reading the loss and gradient scalars is where the host
                # first waits on the backward, so this phase carries the
                # whole minibatch's device time that forward_backward only
                # enqueued.
                with profiler.phase("grad_telemetry"):
                    actor_dashboard = aggregate_actor_tensorboard_metrics(
                        minibatch_metrics
                    )
                    actor_dashboard.update(
                        advantage_group_dashboard(
                            minibatch_metrics, source_names
                        )
                    )
                    actor_dashboard.update(mac_dashboard(minibatch_metrics))
                    actor_dashboard.update(twin_dashboard(minibatch_metrics))
                    actor_dashboard.update(belief_gain_dashboard(critic))
                    actor_dashboard.update(q_prior_dashboard(minibatch_metrics))
                    actor_dashboard.update(support_dashboard(minibatch_metrics))
                    nonfinite_gradients = {
                        name: actor_dashboard[name]
                        for name in (
                            "grad/trunk", "grad/renderer", "grad/combiner",
                            "grad/critic",
                            *(
                                ()
                                if hidden_carry
                                else ("grad/gate", "grad/thought_mean")
                            ),
                        )
                        if not math.isfinite(actor_dashboard[name])
                    }
                if nonfinite_gradients:
                    raise RuntimeError(
                        "non-finite gradients before optimizer step: "
                        f"{nonfinite_gradients}"
                    )

                def measure_drift() -> dict[str, float]:
                    with profiler.phase("post_update_drift"), training_autocast():
                        drift = measure_post_update_policy_drift(
                            wrapper,
                            [device_minibatch],
                            replay_max_trajectories=args.replay_max_trajectories,
                            replay_attention_budget=args.replay_attention_budget,
                            replay_bucket=args.replay_bucket,
                            replay_slot_budget=args.replay_slot_budget,
                            replay_function=diagnostic_replay_head_inputs,
                            emit_logprob_function=diagnostic_emit_token_logprobs,
                            emit_log_odds_function=diagnostic_emit_token_log_odds,
                            replay_plans=[device_replay_plan],
                            target_policy_optimization=(
                                args.target_policy_optimization
                            ),
                            tpo_eta=args.tpo_eta,
                            gae_lambda_alpha=args.gae_lambda_alpha,
                            actor_gae_lambda=args.actor_gae_lambda,
                            source_success_actor_gate=(
                                args.source_success_actor_gate
                            ),
                        )
                    if not all(math.isfinite(value) for value in drift.values()):
                        raise RuntimeError(
                            "non-finite post-update policy drift at step "
                            f"{next_step}: {drift}"
                        )
                    return drift

                def trust_region_readout() -> torch.Tensor:
                    with profiler.phase("trust_region_kl"):
                        return trust_region_token_logprobs(
                            wrapper,
                            device_minibatch,
                            device_replay_plan,
                            shard_stride=args.actor_kl_shard_stride,
                            replay_function=diagnostic_replay_head_inputs,
                            emit_logprob_function=diagnostic_emit_token_logprobs,
                        )

                def measure_step_kl() -> dict[str, float]:
                    step_kl = step_token_kl(
                        pre_step_logprobs, trust_region_readout()
                    )
                    if not all(math.isfinite(value) for value in step_kl.values()):
                        raise RuntimeError(
                            f"non-finite step KL at step {next_step}: {step_kl}"
                        )
                    return step_kl

                actor_parameters_before = None
                pre_step_logprobs = None
                with profiler.phase("optimizer_step"):
                    # The critic steps first: the trust region's reference
                    # read below then runs over the actor's gradients alone.
                    # The two own disjoint parameters, so order is free.
                    step_optimizers(optimizers, "critic")
                    zero_optimizers(optimizers, "critic")
                    if not minibatch_actor_frozen:
                        if frozen_moe_router_parameters:
                            moe_load_cv, moe_max_load = (
                                apply_accumulated_quantile_balance(backbone)
                            )
                            actor_dashboard.update(
                                {
                                    "moe/load_cv2": float(moe_load_cv),
                                    "moe/max_load": float(moe_max_load),
                                }
                            )
                        if args.actor_kl_bound is not None:
                            # The point a KL-overshooting step backtracks
                            # to, and the policy there, read in the same
                            # fp32 as every post-step readout.
                            actor_parameters_before = snapshot_parameters(
                                actor_step_parameters
                            )
                            trust_started = time.perf_counter()
                            pre_step_logprobs = trust_region_readout()
                            torch.cuda.synchronize(device)
                            trust_seconds = time.perf_counter() - trust_started
                        step_optimizers(optimizers, "actor")
                    # Gradient buffers have already been reduced to scalar
                    # telemetry. Release them before the optional second
                    # replay so this read-only diagnostic cannot stack an
                    # inference forward on top of the training step's peak
                    # allocation.
                    zero_optimizers(optimizers, "actor")
                with torch.no_grad():
                    combiner_deltas = [
                        parameter.detach() - before
                        for parameter, before in zip(
                            combiner_parameters, combiner_before, strict=True
                        )
                    ]
                    # The combiner is the one fresh policy component: its
                    # carry matrix leaving zero is the first sign the carried
                    # belief is being used, and a non-finite update here must
                    # abort before the next rollout deploys it.
                    combiner_updates = scalar_tensors_to_floats(
                        {
                            "combiner/update_rms": (
                                torch.stack(
                                    [
                                        delta.square().sum()
                                        for delta in combiner_deltas
                                    ]
                                ).sum()
                                / sum(
                                    delta.numel() for delta in combiner_deltas
                                )
                            ).sqrt(),
                            "combiner/update_abs_max": torch.stack(
                                [delta.abs().max() for delta in combiner_deltas]
                            ).max(),
                            **combiner_weight_telemetry(wrapper, critic),
                        }
                    )
                if not all(
                    math.isfinite(value) for value in combiner_updates.values()
                ):
                    raise RuntimeError(
                        f"non-finite combiner after optimizer step {next_step}: "
                        f"{combiner_updates}"
                    )
                actor_dashboard.update(combiner_updates)
                actor_dashboard["perf/minibatch_h2d_seconds"] = (
                    elapsed_seconds(minibatch_h2d_events)
                )
                actor_dashboard["perf/minibatch_cpu_pack_seconds"] = (
                    minibatch_pack_seconds
                )
                # Monotonic process-global gauges, not per-step. Three
                # separate compile populations can stall a step, and no
                # single counter sees them all: Dynamo forward graphs
                # (unique_graphs), AOT backward compiles (aot_autograd
                # total — invisible to unique_graphs), and TileLang JIT
                # kernels from FLA's KDA ops (invisible to every dynamo
                # counter; counted via a logging hook on tilelang's
                # "begins to compile" line — k3_latent_10h burned 999 s
                # there while unique_graphs stayed flat). A mid-run delta
                # in any of them is a compile; the gauges cannot say WHICH
                # artifact, only that step-time forensics are warranted.
                actor_dashboard["perf/dynamo_unique_graphs"] = float(
                    torch._dynamo.utils.counters["stats"]["unique_graphs"]
                )
                actor_dashboard["perf/aot_autograd_compiles"] = float(
                    torch._dynamo.utils.counters["aot_autograd"]["total"]
                )
                actor_dashboard["perf/tilelang_kernel_compiles"] = float(
                    tilelang_compile_counter.count
                )
                # A cudagraph skip means a graph-decode arm silently ran
                # eager; none of the compile gauges above can see it.
                actor_dashboard["perf/cudagraph_skips"] = float(
                    torch._dynamo.utils.counters["inductor"][
                        "cudagraph_skips"
                    ]
                )
                actor_dashboard["guard/zero_reward_actor_frozen"] = float(
                    minibatch_actor_frozen
                )
                actor_dashboard["guard/zero_reward_frozen_updates_total"] = (
                    float(zero_reward_frozen_updates)
                )
                if actor_parameters_before is not None:
                    # The trust region on the realized token KL to the
                    # pre-step policy. combiner/update_* above measured the
                    # whole step; the kept update is trust/step_fraction of
                    # it (a lerp).
                    trust_started = time.perf_counter()
                    step_kl, trust = enforce_kl_bound(
                        actor_step_parameters,
                        actor_parameters_before,
                        measure_step_kl(),
                        measure_step_kl,
                        bound=args.actor_kl_bound,
                        max_backtracks=args.actor_kl_backtracks,
                        kl_key="kl/post_update_token_pre_step",
                    )
                    del actor_parameters_before, pre_step_logprobs
                    actor_dashboard.update(step_kl)
                    actor_dashboard.update(trust)
                    actor_dashboard["perf/trust_region_seconds"] = (
                        trust_seconds + time.perf_counter() - trust_started
                    )
                if next_step == 1 or (
                    args.post_update_kl_every > 0
                    and next_step % args.post_update_kl_every == 0
                ):
                    drift_started = time.perf_counter()
                    actor_dashboard.update(measure_drift())
                    actor_dashboard["perf/post_update_kl_seconds"] = (
                        time.perf_counter() - drift_started
                    )
                if entropy_temperature is not None:
                    # The temperature this step's objective used, then one
                    # dual step on the entropy it saw -- only when an actor
                    # step consumed it and some emit slot was active.
                    actor_dashboard["entropy/temperature"] = (
                        entropy_temperature.temperature
                    )
                    actor_dashboard["entropy/target"] = entropy_temperature.target
                    if (
                        not minibatch_actor_frozen
                        and "entropy/policy_mean" in actor_dashboard
                    ):
                        entropy_temperature.update(
                            actor_dashboard["entropy/policy_mean"]
                        )
                # One PPO epoch uses every behavior group exactly once. Release
                # the compact host sources and packed device batch after its
                # update and optional drift.
                del device_minibatch
                del device_replay_plan
                del selected_groups
                for index in minibatch_order:
                    groups[index] = None
                step = next_step
                update_seconds = time.perf_counter() - update_started
                logger.log(
                    type="train",
                    step=step,
                    behavior_age=behavior_age,
                    trajectories=(
                        len(minibatch_order) * args.samples_per_prompt
                    ),
                    seconds=update_seconds,
                    pool_seconds=time.perf_counter() - started,
                    peak_vram_bytes=torch.cuda.max_memory_allocated(),
                    dashboard=actor_dashboard,
                    group_metrics=minibatch_metrics,
                )
                tensorboard.add_scalar("perf/update_seconds", update_seconds, step)
                write_actor_tensorboard_metrics(
                    tensorboard, actor_dashboard, behavior_age, step
                )

        # No operation below consumes behavior groups. Release the emptied
        # container before
        # BPB/evaluation allocates its own full-context activations and KV
        # caches, and before the next collect evaluates its RHS.
        # Freeing a pool's host storage is tens of thousands of refcount
        # drops and allocator returns, so it gets a name rather than
        # inflating the remainder.
        with profiler.phase("pool_teardown"):
            zero_optimizers(optimizers, "actor")
            zero_optimizers(optimizers, "critic")
            del groups

        if crossed_interval(previous_step, step, args.bpb_every):
            with profiler.phase("bpb_eval"):
                eval_started = time.perf_counter()
                bpb = teacher_forced_bpb()
                eval_seconds = time.perf_counter() - eval_started
                logger.log(
                    type="bpb", step=step, val_bpb=bpb, seconds=eval_seconds
                )
                tensorboard.add_scalar("guard/val_bpb", bpb, step)
                tensorboard.add_scalar(
                    "perf/bpb_eval_seconds", eval_seconds, step
                )
        if aime_rows and crossed_interval(previous_step, step, args.aime_every):
            with profiler.phase("aime_eval"):
                aime_eval(step)
        if bench_rows and crossed_interval(previous_step, step, args.bench_every):
            with profiler.phase("bench_eval"):
                bench_metrics = bench_eval(step)
            if args.bench_stop_floor is not None:
                bench_below_floor_evals = (
                    bench_below_floor_evals + 1
                    if bench_metrics["contract_accuracy"] < args.bench_stop_floor
                    else 0
                )
        checkpoint_state = (step, warmup_step, sampler.cursor)
        if checkpoint_policy.due():
            with profiler.phase("checkpoint_save"):
                save_started = time.perf_counter()
                save_checkpoint(
                    output / "latent_vapo_checkpoint.pt", wrapper, critic,
                    optimizers, step, args, sampler, warmup_step,
                    actor_init_provenance, zero_reward_frozen_updates,
                    entropy_temperature,
                )
                checkpoint_policy.committed(checkpoint_state)
                save_seconds = time.perf_counter() - save_started
                logger.log(type="checkpoint", step=step, seconds=save_seconds)
                tensorboard.add_scalar(
                    "perf/checkpoint_seconds", save_seconds, step
                )
        if crossed_interval(previous_step, step, args.snapshot_every):
            with profiler.phase("checkpoint_save"):
                (output / "snapshots").mkdir(exist_ok=True)
                save_checkpoint(
                    output / "snapshots" / f"step_{step:06d}.pt", wrapper,
                    critic, optimizers, step, args, sampler, warmup_step,
                    actor_init_provenance, zero_reward_frozen_updates,
                    entropy_temperature,
                )
        profiler.pool_finished(step, time.perf_counter() - started)
    if args.consume_all_prompts and sampler.cursor != len(math_rows):
        # A pool-boundary stop (deadline or consecutive-desert) is a
        # deliberate truncation, not a scheduling bug — save the terminal
        # checkpoint instead of dying on the exhaustion invariant.
        if stopped_at_pool_boundary:
            print(
                "--consume-all-prompts truncated by a pool-boundary stop: "
                f"cursor {sampler.cursor}/{len(math_rows)}",
                flush=True,
            )
        else:
            raise RuntimeError(
                "--consume-all-prompts completed without exhausting the "
                f"target dataset: cursor {sampler.cursor}/{len(math_rows)}"
            )
    terminal_checkpoint_state = (step, warmup_step, sampler.cursor)
    if checkpoint_policy.terminal_due(terminal_checkpoint_state):
        save_checkpoint(
            output / "latent_vapo_checkpoint.pt", wrapper, critic,
            optimizers, step, args, sampler, warmup_step,
            actor_init_provenance, zero_reward_frozen_updates,
            entropy_temperature,
        )
        checkpoint_policy.committed(terminal_checkpoint_state)
    if (
        aime_rows
        and args.aime_every > 0
        and step % args.aime_every != 0
    ):
        # Persist terminal weights before the optional expensive eval. The
        # periodic cadence measures 100, 200, ...; this additionally measures
        # a one-pass dataset ending between cadence boundaries.
        aime_eval(step)
    tensorboard.close()
    profiler.close()


if __name__ == "__main__":
    main()

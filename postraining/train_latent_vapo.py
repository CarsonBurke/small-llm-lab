"""Latent-thought VAPO: full-model PPO over THINK/EMIT gates, tokens, thoughts.

The WHOLE deployed policy path trains at RL time — no frozen trunk. A fresh
linear head owns the Gaussian thought mean instead of reusing pretraining's
next-token latent predictor, so reward can shape a scratch representation
without inheriting that discrete-token target.

- PPO trains everything through one differentiable teacher-forced replay.
  Gate and content log probabilities form one joint action probability at
  each stream position. EMIT clips its joint gate+token ratio. THINK (v24)
  trains under a Dreamer4-style sampled reverse-KL penalty on the thought
  policy — ``KL(behavior || current)`` via the k3 estimator, weighted by
  ``--thought-reverse-kl-coef`` — and NO surrogate-side trust region. That
  is deliberate: the v23 projection bounded only the projected mean inside
  the surrogate, while the raw acting policy drifted through the
  Muon-owned trunk under the LM/gate/renderer losses (job 363: median raw
  KL 0.0424 nats, 68% of updates above 0.03). The penalty is the only term
  that prices that channel directly.

  EXACTLY ONE THINK trust mechanism is active per run. The surrogate-side
  modes (``--thought-clip-mode projected``, the v23 TRPL-style Mahalanobis
  mean projection; ``joint`` and ``per_dim``, the v21/v22 ratio clips) and
  the reverse-KL penalty are alternative SOLUTIONS to the same problem,
  never layers of one — mixing them makes neither term's contribution
  attributable. The rule is a biconditional enforced in both
  ``run_latent_vapo`` and argv parsing: nonzero coefficient <=> clip mode
  ``none``. The surrogate-side modes remain available as ablation arms by
  pairing them with ``--thought-reverse-kl-coef 0``.

  Known gap either way: sigma is not trust-regioned beyond the
  tanh-bounded log-sigma range and the +/-2 ratio guard (a numerical bound
  on the 512-D Gaussian tail that bounds by zeroing a saturated action's
  gradient). Acceptable while sigma is effectively pinned.
  Optional THINK clips its shared gate once as a separate 1-D factor.
- The critic is a SEPARATE from-scratch model (same architecture class,
  fresh weights, fully trainable, no SIGReg or latent prediction) trained
  purely by HL-Gauss cross-entropy on [0, 1] value targets.  It values
  every stream position — vocab AND latent — which is what gives thinking
  its training signal. The support (v22) anchors bin CENTERS at exactly 0
  and 1 with margin bins beyond each — Dreamer3's exact-zero bucket applied
  to both ends of the unit range — so the dominant exact-0/exact-1 verifier
  targets project symmetrically instead of decoding a truncation bias
  inward; labels smooth at sigma_ratio 1.0.
- An optional Bernoulli gate-entropy bonus can preserve THINK exploration;
  there is no continuous-policy entropy bonus or beta-NLL. A sampled reverse
  KL constrains aggregate drift of the 512-D Gaussian against the frozen
  rollout behavior policy, complementing factorwise PPO clipping.
  A zero-initialized belief-conditioned head learns diagonal per-dimension
  thought log-sigma, starting at -3 in every dimension.
- No pretraining anchor: SIGReg and the latent target-prediction objective are
  dropped at RL time. The fresh mean and recurrent policy train purely on
  their ability to think; the teacher-forced val-BPB guard is the drift
  detector.

Prompts are DAPO-Math-17K. An EOS-terminated, verifier-correct final Answer:
field receives reward 1; a wrong but strictly numeric final field receives
bounded distance shaping of at most 0.1. Malformed or unterminated responses
receive zero. The auxiliary positive-example LM loss remains exact-only via
its reward threshold, and AIME 2024 avg@k is the eval. Prompt groups roll out
and replay separately because their lengths differ, then accumulate into the
shared actor/critic optimizer minibatch. ``--rollout-only`` reports whether
rewards vary within prompt groups at the initial 50/50 gate before any update
is attempted.

    python3 -m postraining.train_latent_vapo \
        --checkpoint ablation_results/<run>/pretraining_checkpoint.pt \
        --output postraining/runs/<name> [--rollout-only]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.tensorboard import SummaryWriter

import train_gpt as baseline
from fresh_lejepa_train import FreshHyperparameters
import postraining.latent_rollout
from postraining.latent_eval import (
    evaluate_latent_math,
    verify_terminated_answer,
)
from postraining.benchmark_report import (
    CAPTURE_PROBLEMS,
    CAPTURE_SAMPLES_PER_PROBLEM,
    write_benchmark_report,
)
from postraining.core import (
    POSTTRAIN_REWARD_SCHEMA,
    JsonlLogger,
    POSTTRAIN_CONTEXT_TOKENS,
    POSTTRAIN_PROMPT_TOKENS,
    POSTTRAIN_RESPONSE_TOKENS,
    POSTTRAIN_STREAM_TOKENS,
    answer_style,
    clipped_policy_loss,
    encode_prompt,
    extract_final_answer,
    deterministic_math_subset,
    generalized_advantage_and_return_targets,
    length_adaptive_lambda,
    load_posttraining_tokenizer,
    load_unique_math_rows,
    modal_answer_baseline,
    module_answer_baselines,
    nearby_numeric_reward,
    parse_numeric_answer,
    positive_example_lm_loss,
    validate_posttraining_context_budget,
    verify_answer,
)
from postraining.hl_gauss import anchored_unit_geometry
from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    LatentRolloutBatch,
    assign_terminal_rewards,
    compact_emit_token_logprobs,
    compact_stream_to_device,
    pack_rollout_groups_for_replay,
    emitted_token_rows,
    half_forced_group_members,
    iter_length_aware_microbatches,
    refresh_old_statistics,
    replay_head_inputs,
    scatter_replay_statistics,
    select_thought_actions,
    rollout_continuations,
    split_rollout_groups,
    trim_stream,
)
from postraining.latent_thought import (
    EMIT,
    THINK,
    RENDERER_FEATURES_SCHEMA,
    THOUGHT_DISTRIBUTION_SCHEMA,
    THOUGHT_INPUT_SCHEMA,
    THOUGHT_MEAN_SCHEMA,
    LatentThoughtModel,
    migrate_legacy_wrapper_checkpoint,
    rollout_policy_schema_for_mode,
    validate_renderer_checkpoint,
)
from postraining.model_io import fresh_trunk, load_model
from postraining.muon import Muon
from postraining.train_vapo import prompt_text
from postraining.value_model import SeparateCritic


EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_reverse_kl_thought_trust_anchored_value_zero_affine_general_lr_sequential_data/v24"
)
NO_THOUGHT_KL_EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_projected_thought_trust_no_kl_anchored_value_zero_affine_general_lr_sequential_data/v23"
)
JOINT_CLIP_EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_joint_thought_clip_no_kl_anchored_value_zero_affine_general_lr_sequential_data/v22"
)
UNANCHORED_VALUE_EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_joint_thought_clip_no_kl_zero_affine_general_lr_sequential_data/v21"
)
PER_DIM_REVERSE_KL_EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_per_dim_clip_reverse_kl_zero_affine_general_lr_sequential_data/v20"
)
PERFORMANCE_COMPATIBLE_EXECUTION_SCHEMA = (
    "shuffled_pool1024_disjoint_b256_per_dim_clip_reverse_kl_zero_affine_general_lr_sequential_data/v19"
)
PREVIOUS_EXECUTION_SCHEMA = (
    "shuffled_pool1024_disjoint_b256_per_dim_thought_clip_zero_affine_general_lr_sequential_data/v18"
)
GAIN_SCALED_EXECUTION_SCHEMA = (
    "disjoint_b512_gain_scaled_gaussian_adapter_general_lr_sequential_data/v15"
)
GAIN_SCALED_THOUGHT_INPUT_SCHEMA = (
    "fresh_learned_scalar_identity_affine_s1e-4/v4"
)
PROMPT_ORDER_SCHEMA = "sequential_one_pass/v1"

# Refreshed device minibatches held for their update instead of the
# scatter-to-CPU/repack/re-upload round trip. Bounded because packed batch
# bytes track the pool's longest stream (a think-heavy pool can quadruple
# them); past the budget the refresh loop falls back to the scatter path
# for the remaining minibatches. VRAM-resident only between a pool's
# refresh and its last update — never across a collection.
RETAINED_MINIBATCH_BUDGET_BYTES = 8 << 30
DEFAULT_BPB_GUARD_TOKENS = 2 * 1024 * 1024
DEFAULT_PERIODIC_EVAL_EVERY = 150
# v2: the grading style follows each row's reward_model.style (Minerva for
# DAPO/AIME lineage data, official exact match for mathematics_dataset rows)
# instead of Minerva-normalizing everything.
REWARD_SCHEMA = POSTTRAIN_REWARD_SCHEMA


def math_dataset_identity(path: str | Path, exclude_modules: str) -> str:
    """Content identity that makes a sequential cursor safe to resume."""
    digest = hashlib.sha256()
    digest.update(PROMPT_ORDER_SCHEMA.encode())
    digest.update(b"\0")
    digest.update(exclude_modules.encode())
    digest.update(b"\0")
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def resume_execution_schema_compatible(
    payload: dict,
    *,
    allow_reverse_kl_migration: bool = False,
    allow_performance_migration: bool = False,
    allow_joint_clip_migration: bool = False,
    allow_anchored_value_migration: bool = False,
    allow_projected_thought_migration: bool = False,
    allow_thought_reverse_kl_migration: bool = False,
) -> bool:
    """Resume compatible policy state at a complete rollout-pool boundary."""
    execution_schema = payload.get("execution_schema")
    if execution_schema == EXECUTION_SCHEMA:
        return not (
            allow_reverse_kl_migration
            or allow_performance_migration
            or allow_joint_clip_migration
            or allow_anchored_value_migration
            or allow_projected_thought_migration
            or allow_thought_reverse_kl_migration
        )
    # v24 restores the Dreamer4 reverse-KL term the v21 objective retired, so
    # a v23 checkpoint resumes under an objective it never trained: the
    # sampled k3 penalty now bounds the aggregate drift the mean projection
    # leaves untouched. No parameters or execution change and the pool is
    # rebuilt on resume, so the objective opt-in alone suffices.
    if execution_schema == NO_THOUGHT_KL_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and not allow_projected_thought_migration
            and not allow_reverse_kl_migration
            and not allow_performance_migration
            and not allow_joint_clip_migration
            and not allow_anchored_value_migration
        )
    # v23 changes only the THINK objective (Mahalanobis trust-region
    # projection replaces the joint ratio clip); no parameters or execution
    # change, and the pool is rebuilt on resume, so a v22 checkpoint resumes
    # by objective opt-in alone.
    if execution_schema == JOINT_CLIP_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and allow_projected_thought_migration
            and not allow_reverse_kl_migration
            and not allow_performance_migration
            and not allow_joint_clip_migration
            and not allow_anchored_value_migration
        )
    # v22 changes only the critic support (anchored 0/1 bin centers, sharper
    # sigma): actor state and rollout execution are untouched, but the value
    # head belongs to the old grid and must be rebuilt, so the resume is
    # never silent.
    if execution_schema == UNANCHORED_VALUE_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and allow_projected_thought_migration
            and allow_anchored_value_migration
            and not allow_reverse_kl_migration
            and not allow_performance_migration
            and not allow_joint_clip_migration
        )
    # v21 changes only the THINK objective (joint clip-higher on the Gaussian,
    # reverse KL retired to an ablation knob); sampling, RNG-to-row
    # attribution, and floating-point execution are identical to v20, so a
    # pool-boundary v20 checkpoint resumes by objective opt-in alone.
    if execution_schema == PER_DIM_REVERSE_KL_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and allow_projected_thought_migration
            and allow_anchored_value_migration
            and allow_joint_clip_migration
            and not allow_reverse_kl_migration
            and not allow_performance_migration
        )
    # v20 changed only execution relative to v19: identical prompts and
    # policies are sampled, but deterministic prefixes are shared and finished
    # rows are compacted. A pool-boundary v19 checkpoint therefore needs the
    # execution acknowledgement on top of the objective one, because future
    # RNG-to-row attribution and floating-point execution are not preserved.
    if execution_schema == PERFORMANCE_COMPATIBLE_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and allow_projected_thought_migration
            and allow_anchored_value_migration
            and allow_joint_clip_migration
            and allow_performance_migration
            and not allow_reverse_kl_migration
        )
    # v18 needs every acknowledgement: v19 added reverse KL (an objective its
    # checkpoints never trained under), v20 changed stochastic execution, v21
    # changed the THINK objective again, v22 rebuilt the value support, v23
    # replaced the THINK clip with the projected trust region, and v24
    # restored the reverse-KL term alongside it.
    return (
        allow_reverse_kl_migration
        and allow_performance_migration
        and allow_joint_clip_migration
        and allow_anchored_value_migration
        and allow_projected_thought_migration
        and allow_thought_reverse_kl_migration
        and execution_schema == PREVIOUS_EXECUTION_SCHEMA
    )


def migrate_zero_adapter_resume(
    payload: dict,
    wrapper: LatentThoughtModel,
) -> dict[str, object]:
    """Explicitly replace v15's gain-scaled adapter while preserving resume.

    The recurrent policy semantics change, so this migration is never silent:
    the caller must opt in. All actor state and Adam moments are retained
    except the three old adapter parameters. The new weight/bias start at exact
    zero with fresh optimizer state; the removed scalar has no successor.
    """
    actual_execution = payload.get("execution_schema")
    actual_input = payload.get("thought_input_schema")
    if actual_execution != GAIN_SCALED_EXECUTION_SCHEMA:
        raise ValueError(
            "zero-adapter migration requires execution schema "
            f"{GAIN_SCALED_EXECUTION_SCHEMA!r}; got {actual_execution!r}"
        )
    if actual_input != GAIN_SCALED_THOUGHT_INPUT_SCHEMA:
        raise ValueError(
            "zero-adapter migration requires thought-input schema "
            f"{GAIN_SCALED_THOUGHT_INPUT_SCHEMA!r}; got {actual_input!r}"
        )

    state = payload["model"]
    scalar_key = "adapter.interpolation_strength"
    if scalar_key not in state:
        raise ValueError("gain-scaled checkpoint is missing its adapter scalar")
    scalar = state[scalar_key]
    weight_key = "adapter.projection.weight"
    bias_key = "adapter.projection.bias"
    expected_shapes = {
        weight_key: tuple(wrapper.adapter.projection.weight.shape),
        bias_key: tuple(wrapper.adapter.projection.bias.shape),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in state or tuple(state[key].shape) != expected_shape:
            actual_shape = tuple(state[key].shape) if key in state else None
            raise ValueError(
                f"gain-scaled checkpoint {key} shape must be "
                f"{expected_shape}; got {actual_shape}"
            )
    if scalar.numel() != 1 or not torch.isfinite(scalar).all():
        raise ValueError("gain-scaled checkpoint adapter scalar must be finite")

    actor_optimizer = payload["optimizers"]["actor"]
    groups = actor_optimizer["param_groups"]
    if len(groups) != 6 or len(groups[2]["params"]) != 3:
        raise ValueError(
            "gain-scaled actor optimizer does not have the expected six-group "
            "layout with scalar/weight/bias adapter parameters"
        )

    source_strength = float(scalar)
    scalar_id, weight_id, bias_id = groups[2]["params"]
    if len({scalar_id, weight_id, bias_id}) != 3:
        raise ValueError(
            "gain-scaled actor optimizer adapter parameter IDs must be distinct"
        )

    state.pop(scalar_key)
    state[weight_key] = torch.zeros_like(state[weight_key])
    state[bias_key] = torch.zeros_like(state[bias_key])
    old_state = actor_optimizer["state"]
    for parameter_id in (scalar_id, weight_id, bias_id):
        old_state.pop(parameter_id, None)
    # Optimizer state_dict loading maps saved parameter IDs to live parameters
    # positionally. Retain the old weight/bias IDs in their original order so
    # later groups and every unrelated moment remain aligned.
    groups[2]["params"] = [weight_id, bias_id]

    # Adapter migration changes only v15's recurrent input semantics. Land on
    # the last no-KL schema so entering v19's objective remains a second,
    # explicit migration rather than an accidental side effect of this flag.
    payload["execution_schema"] = PREVIOUS_EXECUTION_SCHEMA
    payload["thought_input_schema"] = THOUGHT_INPUT_SCHEMA
    return {
        "source_execution_schema": actual_execution,
        "source_thought_input_schema": actual_input,
        "source_adapter_strength": source_strength,
        "reset_parameters": [
            "adapter.projection.weight",
            "adapter.projection.bias",
        ],
        "reset_optimizer_state": True,
    }


def value_support_geometry_matches(saved_args: dict, args) -> bool:
    """Whether a checkpoint's critic support geometry matches the CLI's.

    Distinct (bins, margin) pairs can collide on the same total head width
    (e.g. anchored 101/4 and 103/3 both build 110 bins), in which case a
    strict state-dict load succeeds while sigma and the projection width
    silently change — so geometry must be compared as ARGS, not shapes.
    Margin only shapes the grid when the support is anchored.
    """
    return (
        bool(saved_args.get("value_anchored_support", False))
        == args.value_anchored_support
        and saved_args.get("value_bins") == args.value_bins
        and (
            not args.value_anchored_support
            or saved_args.get("value_margin_bins") == args.value_margin_bins
        )
    )


def migrate_anchored_value_resume(
    payload: dict, critic: SeparateCritic
) -> dict[str, object]:
    """Load a pre-anchored critic into the anchored-support geometry.

    The trunk and adapter transfer verbatim — they are the critic's learned
    capacity and are grid-agnostic. The value head and support buffers belong
    to the source's [0, 1]-edge grid (a different bin count), so they keep
    the freshly constructed state: zero head weights with the prior projected
    into the bias. Decoded values collapse to the prior until the head
    relearns from the transferred trunk features; the caller must also start
    the critic AdamW optimizer fresh, because its saved moments include the
    old head parameters.
    """
    source = payload["critic"]
    rebuilt_prefixes = ("head.", "support.")
    transferred = {
        key: value
        for key, value in source.items()
        if not key.startswith(rebuilt_prefixes)
    }
    result = critic.load_state_dict(transferred, strict=False)
    expected_missing = {
        key
        for key in critic.state_dict()
        if key.startswith(rebuilt_prefixes)
    }
    if result.unexpected_keys or set(result.missing_keys) != expected_missing:
        raise ValueError(
            "anchored-value migration expects the source critic to differ "
            "only in head/support state; got unexpected keys "
            f"{sorted(result.unexpected_keys)} and missing keys "
            f"{sorted(result.missing_keys)}"
        )
    return {
        "transferred_parameters": len(transferred),
        "rebuilt_state": sorted(expected_missing),
    }


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


def score_math_rollout(
    batch: LatentRolloutBatch,
    truth: str,
    tokenizer,
    stop_ids: tuple[int, ...],
    style: str = "minerva",
    nearby_reward_max: float = 0.1,
    solution_prefix_ids: tuple[int, ...] = (),
) -> None:
    """Exact verifier reward plus bounded final-answer numeric proximity.

    ``solution_prefix_ids`` are teacher-forced solution tokens that live at
    the end of the prompt (the none-mode ``Answer:`` prefix): the emitted
    continuation alone never contains them, so they rejoin the decode before
    the verifier parses a final answer.
    """
    scores = []
    for emitted in emitted_token_rows(batch):
        stop_cut = next(
            (index for index, token in enumerate(emitted) if token in stop_ids),
            None,
        )
        if stop_cut is None:
            scores.append(0.0)
            continue
        solution = tokenizer.decode(
            list(solution_prefix_ids) + emitted[: stop_cut + 1]
        )
        raw_final_answer = extract_final_answer(solution)
        strict_numeric = (
            parse_numeric_answer(raw_final_answer)
            if raw_final_answer is not None
            else None
        )
        correct, _ = verify_answer(solution, truth, style)
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
    assign_terminal_rewards(
        batch, torch.tensor(scores, dtype=torch.float32, device=batch.rewards.device)
    )


evaluate_aime_latent = evaluate_latent_math


def think_run_lengths(kind: torch.Tensor) -> torch.Tensor:
    """Lengths of every consecutive-THOUGHT run in a (batch, stream) kind map.

    This is total latent-compute telemetry: it includes any forced initial
    thought and merges it with an immediately following optional thought.
    Runs are per-row (prompts themselves are token slots), so flattening
    start/end indices row-major keeps them paired.
    """
    thinks = kind == THOUGHT_SLOT
    previous = torch.zeros_like(thinks)
    previous[:, 1:] = thinks[:, :-1]
    following = torch.zeros_like(thinks)
    following[:, :-1] = thinks[:, 1:]
    starts = (thinks & ~previous).flatten().nonzero().squeeze(-1)
    ends = (thinks & ~following).flatten().nonzero().squeeze(-1)
    return (ends - starts + 1).float()


def rollout_diagnostics(
    batch: LatentRolloutBatch,
    samples_per_prompt: int,
    stop_ids: tuple[int, ...] = (),
) -> dict[str, float | int]:
    gate_actions = batch.gate_mask.sum().clamp_min(1)
    stop_set = set(stop_ids)
    # Fraction of rows that terminated themselves (emitted BOS or EOS)
    # rather than exhausting the token/stream budget.
    ended = [
        float(any(token in stop_set for token in row))
        for row in emitted_token_rows(batch)
    ]
    generated = batch.action_mask.bool()
    think_fraction = float(
        ((batch.gate_actions == THINK).float() * batch.gate_mask).sum()
        / gate_actions
    )
    runs = think_run_lengths(batch.kind)
    grouped = batch.reward_scalar.reshape(-1, samples_per_prompt)
    exact = batch.reward_scalar == 1.0
    grouped_exact = exact.reshape(-1, samples_per_prompt).float()
    # Did thinking pay off THIS group?  Within-group Pearson correlation
    # between per-trajectory think counts and rewards (0.0 when either side
    # has no variance — undefined groups dilute the aggregate toward zero,
    # which is the honest prior for "no evidence either way").
    # Reward comparisons here isolate OPTIONAL Bernoulli-selected thoughts;
    # the paired forced/unforced prefix comparison is reported separately.
    think_counts = (
        (batch.gate_actions == THINK).float() * batch.gate_mask
    ).sum(1)
    # Forced actions are THINKs recorded without a gate decision. The THINK
    # conjunction matters for pinned-EMIT modes, where NO action carries a
    # gate decision yet nothing was forced.
    forced_initial = (
        (batch.gate_actions == THINK).float()
        * (batch.action_mask - batch.gate_mask)
    ).sum(1) > 0
    thinkers = think_counts > 0

    def think_reward_correlation(rows: torch.Tensor) -> float:
        counts = think_counts[rows]
        rewards = batch.reward_scalar[rows]
        if counts.numel() == 0:
            return 0.0
        centered_counts = counts - counts.mean()
        centered_rewards = rewards - rewards.mean()
        scale = (
            centered_counts.square().mean().sqrt()
            * centered_rewards.square().mean().sqrt()
        )
        return (
            float((centered_counts * centered_rewards).mean() / scale)
            if float(scale) > 0
            else 0.0
        )
    return {
        "trajectories": batch.reward_scalar.numel(),
        "stream_length": batch.stream_length,
        "reward_mean": float(batch.reward_scalar.mean()),
        "reward_std": float(batch.reward_scalar.std(unbiased=False)),
        "within_group_reward_std": float(grouped.std(dim=1, unbiased=False).mean()),
        "exact_accuracy": float(exact.float().mean()),
        "exact_within_group_reward_std": float(
            grouped_exact.std(dim=1, unbiased=False).mean()
        ),
        "partial_reward_fraction": float(
            ((batch.reward_scalar > 0.0) & ~exact).float().mean()
        ),
        "think_fraction": think_fraction,
        "forced_initial_thinks_per_trajectory": float(
            (((batch.gate_actions == THINK).float() * batch.action_mask).sum(1)
             - think_counts).mean()
        ),
        "forced_initial_trajectory_fraction": float(
            forced_initial.float().mean()
        ),
        "reward_mean_forced_initial": (
            float(batch.reward_scalar[forced_initial].mean())
            if bool(forced_initial.any())
            else 0.0
        ),
        "reward_mean_unforced_initial": (
            float(batch.reward_scalar[~forced_initial].mean())
            if bool((~forced_initial).any())
            else 0.0
        ),
        "thoughts_per_trajectory": float(
            ((batch.gate_actions == THINK).float() * batch.action_mask).sum(1).mean()
        ),
        "think_run_mean": float(runs.mean()) if runs.numel() else 0.0,
        "think_run_std": float(runs.std(unbiased=False)) if runs.numel() else 0.0,
        "think_runs_per_trajectory": runs.numel() / batch.reward_scalar.numel(),
        "emits_per_trajectory": float(batch.emit_mask.sum(1).mean()),
        "actions_per_trajectory": float(batch.action_mask.sum(1).mean()),
        "old_value_mean": float(batch.old_values[generated].mean()) if generated.any() else 0.0,
        "optional_thinking_trajectory_fraction": float(thinkers.float().mean()),
        "reward_mean_optional_thinking": (
            float(batch.reward_scalar[thinkers].mean()) if bool(thinkers.any()) else 0.0
        ),
        "reward_mean_no_optional_thinking": (
            float(batch.reward_scalar[~thinkers].mean())
            if bool((~thinkers).any())
            else 0.0
        ),
        "think_reward_correlation": think_reward_correlation(
            torch.ones_like(forced_initial)
        ),
        "think_reward_correlation_forced_initial": think_reward_correlation(
            forced_initial
        ),
        "think_reward_correlation_unforced_initial": think_reward_correlation(
            ~forced_initial
        ),
        "ended_fraction": sum(ended) / max(len(ended), 1),
    }


def aggregate_diagnostics(
    groups: list[LatentRolloutBatch],
    samples_per_prompt: int,
    stop_ids: tuple[int, ...] = (),
) -> dict[str, float | int]:
    """Mean of per-group rollout diagnostics; trajectory counts are summed."""
    per_group = [
        rollout_diagnostics(group, samples_per_prompt, stop_ids) for group in groups
    ]
    aggregated: dict[str, float | int] = {}
    for key in per_group[0]:
        values = [metrics[key] for metrics in per_group]
        if key == "trajectories":
            aggregated[key] = int(sum(values))
        else:
            aggregated[key] = float(sum(values) / len(values))
    runs = [think_run_lengths(group.kind) for group in groups]
    nonempty_runs = [run for run in runs if run.numel()]
    if nonempty_runs:
        all_runs = torch.cat(nonempty_runs)
        aggregated["think_run_p95"] = float(
            torch.quantile(all_runs, 0.95, interpolation="higher")
        )
        aggregated["think_run_max"] = float(all_runs.max())
    else:
        aggregated["think_run_p95"] = 0.0
        aggregated["think_run_max"] = 0.0
    return aggregated


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
        "excess_ce": _weighted_metric_mean(
            metrics, "value_excess_ce", "action_count"
        ),
    }


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
    log_sigma_mean = _weighted_metric_mean(
        metrics, "thought_log_sigma_mean", "thought_action_count"
    )
    log_sigma_second_moment = _weighted_metric_mean(
        [
            {
                **metric,
                "thought_log_sigma_second_moment": (
                    metric["thought_log_sigma_std"] ** 2
                    + metric["thought_log_sigma_mean"] ** 2
                ),
            }
            for metric in metrics
        ],
        "thought_log_sigma_second_moment",
        "thought_action_count",
    )
    last = metrics[-1]
    # Each group is already divided by the denominator of the complete actor
    # optimizer minibatch, so its loss is a contribution to be summed rather
    # than another independently normalized minibatch mean.
    policy_contribution = sum(metric["policy_loss"] for metric in metrics)
    positive_lm_contribution = sum(
        metric["positive_lm_loss"] * metric["positive_lm_weight"]
        for metric in metrics
    )
    thought_reverse_kl_contribution = sum(
        metric["thought_reverse_kl_penalty"] for metric in metrics
    )
    gate_entropy_bonus = sum(
        metric["gate_entropy_bonus"] for metric in metrics
    )
    return {
        "loss/policy": policy_contribution,
        "loss/positive_lm_weighted": positive_lm_contribution,
        "kl/thought_reverse_weighted": thought_reverse_kl_contribution,
        "bonus/gate_entropy_weighted": gate_entropy_bonus,
        "loss/actor_total": (
            policy_contribution
            + positive_lm_contribution
            + thought_reverse_kl_contribution
            - gate_entropy_bonus
        ),
        "value/token_weighted_excess_ce": value["excess_ce"],
        "value/prediction_mean": value["prediction_mean"],
        "value/target_mean": value["target_mean"],
        "value/residual_mean": value["residual_mean"],
        "value/online_epoch_explained_variance": value["explained_variance"],
        "advantage/mean": advantage_mean,
        "advantage/std": advantage_std,
        "advantage/optional_think_mean": _weighted_metric_mean(
            metrics, "think_advantage_mean", "think_action_count"
        ),
        "advantage/forced_initial_think_mean": _weighted_metric_mean(
            metrics,
            "forced_initial_think_advantage_mean",
            "forced_initial_think_action_count",
        ),
        "advantage/all_think_mean": _weighted_metric_mean(
            metrics, "thought_advantage_mean", "thought_action_count"
        ),
        "advantage/emit_mean": _weighted_metric_mean(
            metrics, "emit_advantage_mean", "emit_action_count"
        ),
        "behavior/optional_think_probability": 1.0 - _weighted_metric_mean(
            metrics, "emit_probability", "gate_action_count"
        ),
        "behavior/thought_adapter_weight_rms": last[
            "thought_adapter_weight_rms"
        ],
        "behavior/thought_adapter_bias_rms": last[
            "thought_adapter_bias_rms"
        ],
        "behavior/gate_entropy": _weighted_metric_mean(
            metrics, "gate_entropy", "gate_action_count"
        ),
        "sigma/log_std_mean": log_sigma_mean,
        "sigma/log_std_std": math.sqrt(
            max(0.0, log_sigma_second_moment - log_sigma_mean**2)
        ),
        "sigma/log_std_min": min(
            metric["thought_log_sigma_min"] for metric in metrics
        ),
        "sigma/log_std_max": max(
            metric["thought_log_sigma_max"] for metric in metrics
        ),
        "sigma/std_mean": _weighted_metric_mean(
            metrics, "thought_sigma_mean", "thought_action_count"
        ),
        "sigma/expected_noise_norm": _weighted_metric_mean(
            metrics, "thought_expected_noise_norm", "thought_action_count"
        ),
        "sigma/realized_noise_norm": _weighted_metric_mean(
            metrics, "thought_realized_noise_norm", "thought_action_count"
        ),
        "sigma/normalized_noise_rms": _weighted_metric_mean(
            metrics, "thought_normalized_noise_rms", "thought_action_count"
        ),
        "sigma/head_raw_bias_mean": last["thought_log_sigma_raw_bias_mean"],
        "sigma/head_weight_rms": last["thought_log_sigma_weight_rms"],
        "sigma/state_residual_gain": last["thought_log_sigma_residual_gain"],
        "sigma/mean_norm": _weighted_metric_mean(
            metrics, "thought_mean_norm", "thought_action_count"
        ),
        "sigma/noise_mean_norm_ratio": (
            _weighted_metric_mean(
                metrics, "thought_mean_norm", "thought_action_count"
            )
            / max(
                _weighted_metric_mean(
                    metrics, "thought_expected_noise_norm", "thought_action_count"
                ),
                1e-12,
            )
        ),
        "sigma/mean_head_weight_rms": last["thought_mean_weight_rms"],
        "sigma/mean_head_bias_rms": last["thought_mean_bias_rms"],
        "sigma/mean_output_gain": last["thought_mean_output_gain"],
        "kl/gate_behavior": _weighted_metric_mean(
            metrics, "gate_behavior_kl", "gate_action_count"
        ),
        "kl/renderer_behavior": _weighted_metric_mean(
            metrics, "renderer_behavior_kl", "emit_action_count"
        ),
        "kl/thought_behavior_joint": _weighted_metric_mean(
            metrics, "thought_behavior_kl_joint", "thought_action_count"
        ),
        # Closed-form squared-Mahalanobis movement of the THINK mean from
        # stored behavior (the projected objective's trust statistic; the
        # sampled-ratio KL above stays as a cross-check).
        "trust/thought_d_mean": _weighted_metric_mean(
            metrics, "thought_trust_d_mean", "thought_action_count"
        ),
        "trust/thought_d_max": max(
            metric["thought_trust_d_max"] for metric in metrics
        ),
        "trust/projection_penalty": sum(
            metric["thought_projection_penalty"] for metric in metrics
        ),
        "kl/policy_behavior_per_action": sum(
            metric["policy_behavior_kl_per_action"] for metric in metrics
        ),
        "clip/policy": sum(
            metric["policy_clip_fraction"] for metric in metrics
        ),
        "ratio/joint_abs_log_max": max(
            metric["joint_abs_log_ratio_max"] for metric in metrics
        ),
        "ratio/thought_dim_abs_log_max": max(
            metric["thought_dim_abs_log_ratio_max"] for metric in metrics
        ),
        "ratio/thought_joint_abs_log_max": max(
            metric["thought_joint_abs_log_ratio_max"] for metric in metrics
        ),
        "ratio/harmful_positive_log_max": max(
            metric["harmful_positive_log_ratio_max"] for metric in metrics
        ),
        # Share of THINK actions whose mean the trust region actually moved.
        # Computed since v23 but never surfaced, which left the projected
        # arm's headline diagnostic invisible; it reads identically 0 under
        # --thought-clip-mode none, so it also separates the two arms.
        "clip/thought_projection": sum(
            metric["thought_policy_clip_fraction"] for metric in metrics
        ),
        # Both optimizers accumulate across groups; only the final group has
        # the complete pre-step gradient norm.
        "grad/trunk": last["trunk_grad_norm"],
        "grad/renderer": last["renderer_grad_norm"],
        "grad/adapter": last["adapter_grad_norm"],
        "grad/gate": last["gate_grad_norm"],
        "grad/sigma": last["sigma_grad_norm"],
        "grad/thought_mean": last["thought_mean_grad_norm"],
        "grad/critic": last["critic_grad_norm"],
    }


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
        "reward/forced_initial_mean": float(
            metrics["reward_mean_forced_initial"]
        ),
        "reward/unforced_initial_mean": float(
            metrics["reward_mean_unforced_initial"]
        ),
        "reward/forced_initial_delta": float(
            metrics["reward_mean_forced_initial"]
            - metrics["reward_mean_unforced_initial"]
        ),
        "reward/ended_fraction": float(metrics["ended_fraction"]),
        "behavior/optional_think_fraction": float(metrics["think_fraction"]),
        "behavior/thoughts_per_trajectory": float(
            metrics["thoughts_per_trajectory"]
        ),
        "behavior/emits_per_trajectory": float(
            metrics["emits_per_trajectory"]
        ),
        "behavior/think_run_p95": float(metrics["think_run_p95"]),
        "behavior/think_run_max": float(metrics["think_run_max"]),
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


def plan_one_pass_training(
    *,
    dataset_rows: int,
    sampler_cursor: int,
    warmup_updates: int,
    remaining_actor_steps: int,
    prompts_per_minibatch: int,
) -> tuple[int, int]:
    """Validate exact one-pass consumption and return prompts and actor steps.

    Warmup consumes one complete prompt minibatch per critic update. Actor
    training then consumes every remaining row, with at most one terminal
    partial minibatch. This helper is deliberately pure so fresh curriculum
    starts and checkpoint resumes share testable no-reuse arithmetic.
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
    actor_prompts = available - warmup_prompts
    required_actor_steps = math.ceil(actor_prompts / prompts_per_minibatch)
    if remaining_actor_steps != required_actor_steps:
        raise ValueError(
            "--consume-all-prompts requires exactly "
            f"{required_actor_steps} remaining actor steps for "
            f"{actor_prompts} post-warmup prompts; got "
            f"{remaining_actor_steps}"
        )
    return available, required_actor_steps


def actor_minibatch_denominators(
    groups: list[LatentRolloutBatch],
    indices: list[int],
    positive_reward_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Global action/gate/positive-token denominators for one actor step."""
    if not indices:
        raise ValueError("at least one prompt group is required")
    action_count = torch.stack(
        [groups[index].action_mask.sum() for index in indices]
    ).sum()
    gate_action_count = torch.stack(
        [groups[index].gate_mask.sum() for index in indices]
    ).sum()
    positive_token_count = torch.stack(
        [
            (
                groups[index].emit_mask
                * (
                    groups[index].reward_scalar >= positive_reward_threshold
                )[:, None]
            ).sum()
            for index in indices
        ]
    ).sum()
    return action_count, gate_action_count, positive_token_count


def write_actor_tensorboard_metrics(
    tensorboard, dashboard: dict[str, float], behavior_age: int, step: int
) -> None:
    """Write one actor-step row, eliding exact fresh-behavior zeros."""
    if behavior_age == 0:
        refresh_drift = max(
            dashboard[tag]
            for tag in (
                "kl/gate_behavior",
                "kl/renderer_behavior",
                "kl/thought_behavior_joint",
                "kl/policy_behavior_per_action",
                "kl/thought_reverse_weighted",
                "clip/policy",
                "clip/thought_projection",
                "ratio/joint_abs_log_max",
                "ratio/thought_dim_abs_log_max",
                "ratio/thought_joint_abs_log_max",
                "ratio/harmful_positive_log_max",
                "trust/thought_d_mean",
                "trust/thought_d_max",
                "trust/projection_penalty",
            )
        )
        tensorboard.add_scalar(
            "debug/behavior_refresh_max_drift", refresh_drift, step
        )
    for tag, value in dashboard.items():
        if behavior_age == 0 and tag in {
            "kl/gate_behavior",
            "kl/renderer_behavior",
            "kl/thought_behavior_joint",
            "kl/policy_behavior_per_action",
            "kl/thought_reverse_weighted",
            "clip/policy",
            "clip/thought_projection",
            "ratio/joint_abs_log_max",
            "ratio/thought_dim_abs_log_max",
            "ratio/thought_joint_abs_log_max",
            "ratio/harmful_positive_log_max",
            "trust/thought_d_mean",
            "trust/thought_d_max",
            "trust/projection_penalty",
        }:
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
    """Resolve same-device scalar telemetry with one device synchronization."""
    if not values:
        return {}
    keys = tuple(values)
    packed = torch.stack(
        [values[key].detach().reshape(()).float() for key in keys]
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


def muon_matrix_parameters(
    blocks: torch.nn.Module, excluded_parameter_ids: set[int]
) -> list[torch.nn.Parameter]:
    """The pretraining Muon partition: block matrices with ndim >= 2.

    Mirrors ``nanogpt_mini_gpt2vocab_train.py`` exactly — embeddings, the
    readout, biases, and norm gains stay under AdamW. Fresh-lineage probes
    live under ``blocks[-1]``, so the exclusion set must be honored here too.
    """
    return [
        parameter
        for parameter in blocks.parameters()
        if parameter.ndim >= 2 and id(parameter) not in excluded_parameter_ids
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
        "critic": args.learning_rate,
        "actor_muon": args.muon_learning_rate,
        "critic_muon": args.critic_muon_learning_rate,
    }


def reassert_learning_rates(
    optimizers: dict[str, torch.optim.Optimizer], args
) -> None:
    """Reapply the CLI learning rates after loading optimizer state."""
    rates = optimizer_learning_rates(args)
    for name, optimizer in optimizers.items():
        for group in optimizer.param_groups:
            group["lr"] = rates[name]


def build_optimizers(
    wrapper: LatentThoughtModel,
    critic: SeparateCritic,
    learning_rate: float,
    trunk_optimizer: str = "adamw",
    muon_learning_rate: float | None = None,
    critic_muon_learning_rate: float | None = None,
    fused: bool = True,
) -> dict[str, torch.optim.Optimizer]:
    """The actor/critic optimizer layout.

    One actor AdamW retains six semantic groups for exact telemetry and
    checkpoint validation, but every trainable policy component uses the same
    general learning rate as the critic: trunk, Bernoulli gate, recurrent
    adapter, renderer, state-dependent log-sigma, and fresh mean. This removes
    the previous hand-tuned head-specific rates from the fresh-policy test.
    The fresh probes live under
    ``blocks[-1]`` (so pretraining's optimizer saw them), which makes
    name-prefix filtering wrong — exclude them from the trunk by identity.
    Nano backbones keep the identical six-group layout with their native
    readout as the renderer group.

    ``trunk_optimizer="muon"`` restores the pretraining update geometry:
    block matrices (ndim >= 2) move from the AdamW trunk group into separate
    ``actor_muon``/``critic_muon`` Muon optimizers — same NewtonSchulz5
    orthogonalization, momentum, and rectangular scaling the trunk was
    pretrained under — while embeddings, readout, gains, and every RL-only
    head stay under AdamW. Weight decay stays 0 everywhere: pretraining's
    Muon decay (0.05) regularizes a from-scratch run, but over a long RL
    schedule it would only shrink the pretrained weights.
    """
    if trunk_optimizer not in ("adamw", "muon"):
        raise ValueError(f"unknown trunk optimizer {trunk_optimizer!r}")
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
    trunk_parameters = [
        parameter
        for parameter in backbone.parameters()
        if id(parameter) not in excluded_parameter_ids
        and id(parameter) not in actor_muon_ids
    ]
    critic_adamw_parameters = [
        parameter
        for parameter in critic.parameters()
        if id(parameter) not in critic_muon_ids
    ]
    optimizers = {
        "actor": torch.optim.AdamW(
            [
                {"params": trunk_parameters, "lr": learning_rate},
                {"params": list(wrapper.gate.parameters()), "lr": learning_rate},
                {"params": list(wrapper.adapter.parameters()), "lr": learning_rate},
                {"params": renderer_parameters(backbone), "lr": learning_rate},
                {
                    "params": list(wrapper.transition.log_sigma_head.parameters()),
                    "lr": learning_rate,
                },
                {
                    "params": list(wrapper.transition.mean_head.parameters()),
                    "lr": learning_rate,
                },
            ],
            weight_decay=0.0,
            fused=fused,
        ),
        "critic": torch.optim.AdamW(
            critic_adamw_parameters,
            lr=learning_rate,
            weight_decay=0.0,
            fused=fused,
        ),
    }
    if trunk_optimizer == "muon":
        if muon_learning_rate is None or critic_muon_learning_rate is None:
            raise ValueError("the muon trunk optimizer requires its learning rates")
        optimizers["actor_muon"] = Muon(
            actor_muon_parameters, lr=muon_learning_rate, weight_decay=0.0
        )
        optimizers["critic_muon"] = Muon(
            critic_muon_parameters, lr=critic_muon_learning_rate, weight_decay=0.0
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
    trunk_coverage = {id(p) for p in trunk_parameters} | actor_muon_ids
    trunk_expected = {
        id(parameter)
        for parameter in backbone.parameters()
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
        id(p) for p in critic.parameters()
    } or len(critic_registered) != len({id(p) for p in critic_registered}):
        raise AssertionError("critic parameters must partition across optimizers")
    return optimizers


def joint_action_logprobs(
    new_gate_logprobs: torch.Tensor,
    old_gate_logprobs: torch.Tensor,
    new_token_logprobs: torch.Tensor,
    old_token_logprobs: torch.Tensor,
    new_thought_logprobs: torch.Tensor,
    old_thought_logprobs: torch.Tensor,
    gate_mask: torch.Tensor,
    emit_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine conditional policy factors into one log-probability/action."""
    return (
        new_gate_logprobs * gate_mask
        + new_token_logprobs * emit_mask
        + new_thought_logprobs,
        old_gate_logprobs * gate_mask
        + old_token_logprobs * emit_mask
        + old_thought_logprobs,
    )


def per_dimension_thought_policy_loss(
    new_gate_logprobs: torch.Tensor,
    old_gate_logprobs: torch.Tensor,
    new_thought_logprobs: torch.Tensor,
    old_thought_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    gate_mask: torch.Tensor,
    action_denominator: torch.Tensor,
    epsilon_low: float = 0.20,
    epsilon_high: float = 0.28,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip a diagonal-Gaussian THINK action one latent dimension at a time.

    Factorwise clipping must retain the score-function gradient of the joint
    diagonal Gaussian: ``A * sum_d grad(log p_d)``. Merely averaging the
    dimensional losses would weaken mean/sigma learning by ``latent_dim``.
    The straight-through rescaling below reports their mean surrogate value
    while preserving their summed gradient.

    The optional Bernoulli gate is clipped exactly once as its own factor. A
    detached baseline correction makes an unchanged optional THINK report one
    action's loss rather than two without altering either factor's gradient.
    A forced THINK has ``gate_mask == 0`` and trains only Gaussian factors.

    The returned clip fractions are action-normalized. A thought whose every
    dimension clips contributes one to the first; a clipped optional gate
    contributes one to the second.
    """
    if new_thought_logprobs.ndim != 2:
        raise ValueError("thought log-probabilities must have shape [N, D]")
    if new_thought_logprobs.shape != old_thought_logprobs.shape:
        raise ValueError("new and old thought log-probabilities must match")
    action_count, latent_dim = new_thought_logprobs.shape
    if latent_dim < 1:
        raise ValueError("thought log-probabilities need at least one dimension")
    expected_vector_shape = (action_count,)
    for name, value in (
        ("new gate log-probabilities", new_gate_logprobs),
        ("old gate log-probabilities", old_gate_logprobs),
        ("advantages", advantages),
        ("gate mask", gate_mask),
    ):
        if tuple(value.shape) != expected_vector_shape:
            raise ValueError(f"{name} must have shape {expected_vector_shape}")

    dimension_log_ratio = new_thought_logprobs - old_thought_logprobs
    log_lower = torch.log(
        dimension_log_ratio.new_tensor(1.0 - epsilon_low)
    )
    log_upper = torch.log(
        dimension_log_ratio.new_tensor(1.0 + epsilon_high)
    )
    dimension_advantages = advantages[:, None]
    effective_dimension_log_ratio = torch.where(
        dimension_advantages >= 0,
        torch.minimum(dimension_log_ratio, log_upper),
        torch.maximum(dimension_log_ratio, log_lower),
    )
    dimension_denominator = (
        action_denominator.to(dimension_log_ratio.device) * latent_dim
    ).clamp_min(1)
    dimension_mean_loss = -(
        effective_dimension_log_ratio.exp() * dimension_advantages
    ).sum() / dimension_denominator
    dimension_clip_fraction = (
        (dimension_log_ratio < log_lower)
        | (dimension_log_ratio > log_upper)
    ).sum() / dimension_denominator
    dimension_loss = (
        dimension_mean_loss.detach()
        + latent_dim * (dimension_mean_loss - dimension_mean_loss.detach())
    )
    gate_loss, gate_clip_fraction, _ = clipped_policy_loss(
        new_gate_logprobs,
        old_gate_logprobs,
        advantages,
        gate_mask,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        denominator=action_denominator,
        estimate_kl=False,
    )
    optional_gate_baseline = (
        advantages.detach() * gate_mask
    ).sum() / action_denominator.clamp_min(1)
    return (
        dimension_loss + gate_loss + optional_gate_baseline,
        dimension_clip_fraction,
        gate_clip_fraction,
    )


def joint_thought_policy_loss(
    new_gate_logprobs: torch.Tensor,
    old_gate_logprobs: torch.Tensor,
    new_thought_logprobs: torch.Tensor,
    old_thought_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    gate_mask: torch.Tensor,
    action_denominator: torch.Tensor,
    epsilon_low: float = 0.20,
    epsilon_high: float = 0.28,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip a diagonal-Gaussian THINK action's JOINT ratio exactly once.

    The thought is one action: its per-dimension log-ratios sum into a single
    joint ratio and the clip-higher band applies to that ratio, exactly like
    EMIT's joint gate+token clip. Unlike the per-dimension objective this
    imposes the trust region on the actual sampled action, at the cost of an
    all-or-nothing gradient per thought — one clipped joint ratio silences
    every dimension of that action's favorable-direction gradient, and the
    harmful direction carries the raw ``exp(sum)`` ratio (watch
    ``harmful_positive_log_ratio_max``; the log-space clip inside
    ``clipped_policy_loss`` bounds only the favorable side).

    The optional Bernoulli gate is clipped exactly once as its own factor. A
    detached baseline correction makes an unchanged optional THINK report one
    action's loss rather than two, matching the per-dimension objective's
    reporting. A forced THINK has ``gate_mask == 0`` and trains only the
    Gaussian factor. Clip fractions are action-normalized, like the
    per-dimension objective's.
    """
    if new_thought_logprobs.ndim != 2:
        raise ValueError("thought log-probabilities must have shape [N, D]")
    if new_thought_logprobs.shape != old_thought_logprobs.shape:
        raise ValueError("new and old thought log-probabilities must match")
    if new_thought_logprobs.size(1) < 1:
        raise ValueError("thought log-probabilities need at least one dimension")
    expected_vector_shape = (new_thought_logprobs.size(0),)
    for name, value in (
        ("new gate log-probabilities", new_gate_logprobs),
        ("old gate log-probabilities", old_gate_logprobs),
        ("advantages", advantages),
        ("gate mask", gate_mask),
    ):
        if tuple(value.shape) != expected_vector_shape:
            raise ValueError(f"{name} must have shape {expected_vector_shape}")

    thought_loss, thought_clip_fraction, _ = clipped_policy_loss(
        new_thought_logprobs.sum(-1),
        old_thought_logprobs.sum(-1),
        advantages,
        torch.ones_like(advantages),
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        denominator=action_denominator,
        estimate_kl=False,
    )
    gate_loss, gate_clip_fraction, _ = clipped_policy_loss(
        new_gate_logprobs,
        old_gate_logprobs,
        advantages,
        gate_mask,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        denominator=action_denominator,
        estimate_kl=False,
    )
    optional_gate_baseline = (
        advantages.detach() * gate_mask
    ).sum() / action_denominator.clamp_min(1)
    return (
        thought_loss + gate_loss + optional_gate_baseline,
        thought_clip_fraction,
        gate_clip_fraction,
    )


def project_thought_means(
    new_means: torch.Tensor,
    old_means: torch.Tensor,
    old_log_sigmas: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TRPL-style mean projection onto the behavior trust region.

    Distances are squared Mahalanobis in behavior-sigma units (= twice the
    Gaussian KL while sigma is frozen), computed in closed form from the
    stored behavior means — no sampled ratio enters the trust decision, so
    unlike the joint clip it cannot fire on rollout noise. Inside the region
    the mean passes through untouched. Outside, the excess is scaled back
    onto the boundary WITH the gradient flowing through the scale: the
    Jacobian ``s * (I - u u^T)`` annihilates the radial component (nothing
    keeps pushing outward) while tangential learning survives.

    Returns ``(projected_means, scale, mahalanobis_sq)`` with ``scale == 1``
    exactly where no projection occurred.
    """
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("trust epsilon must be finite and positive")
    if new_means.shape != old_means.shape:
        raise ValueError("new and old thought means must match")
    if old_log_sigmas.shape != old_means.shape:
        raise ValueError("behavior log sigmas must match the means")
    normalized_shift = (new_means - old_means) * (-old_log_sigmas).exp()
    mahalanobis_sq = normalized_shift.square().sum(-1)
    # Both torch.where branches evaluate; the clamp keeps the unselected
    # rsqrt finite at zero shift (age-0 rows) so its zeroed gradient cannot
    # poison the backward pass with 0 * inf.
    scale = torch.where(
        mahalanobis_sq > epsilon,
        (epsilon / mahalanobis_sq.clamp_min(1e-12)).sqrt(),
        torch.ones_like(mahalanobis_sq),
    )
    projected = old_means + scale[:, None] * (new_means - old_means)
    return projected, scale, mahalanobis_sq


def projected_thought_policy_loss(
    new_gate_logprobs: torch.Tensor,
    old_gate_logprobs: torch.Tensor,
    projected_thought_logprobs: torch.Tensor,
    old_thought_logprobs: torch.Tensor,
    trust_scale: torch.Tensor,
    advantages: torch.Tensor,
    gate_mask: torch.Tensor,
    action_denominator: torch.Tensor,
    epsilon_low: float = 0.20,
    epsilon_high: float = 0.28,
    ratio_guard: float = 2.0,
    gate_advantages: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Surrogate for THINK actions scored under the PROJECTED Gaussian.

    ``exp(log pi_proj - log pi_behavior) * A`` with the gradient through the
    ratio, exactly the unclipped PPO surrogate — the trust region lives in
    the projection, not in a ratio band. With the mean projected, the
    MEAN-driven part of the joint log-ratio concentrates near zero
    (its sampling noise has standard deviation ~sqrt(2 * d_M) <=
    sqrt(2 * epsilon)); the sigma-driven part is NOT constrained by the
    projection (see the module docstring on the sigma gap), so
    ``ratio_guard`` — a symmetric log-space clamp — bounds the exponential
    against Gaussian-tail and sigma-drift excursions. It is a numerical
    guard, not a trust mechanism: a saturated clamp zeroes that action's
    gradient rather than steering it.

    The optional Bernoulli gate keeps its own 1-D textbook clip, and the
    detached baseline makes an unchanged optional THINK report one action's
    loss, both exactly as in the joint/per-dimension objectives. The gate
    factor uses ``gate_advantages`` (raw scale, matching the EMIT gate)
    while ``advantages`` are the caller's unit-normalized values that
    calibrate the thought surrogate. The middle return value is the
    action-normalized PROJECTION fraction (reported through the
    thought-clip-fraction channel).
    """
    if projected_thought_logprobs.ndim != 2:
        raise ValueError("thought log-probabilities must have shape [N, D]")
    if projected_thought_logprobs.shape != old_thought_logprobs.shape:
        raise ValueError("new and old thought log-probabilities must match")
    if ratio_guard <= 0.0:
        raise ValueError("ratio guard must be positive")
    if gate_advantages is None:
        gate_advantages = advantages
    expected_vector_shape = (projected_thought_logprobs.size(0),)
    for name, value in (
        ("new gate log-probabilities", new_gate_logprobs),
        ("old gate log-probabilities", old_gate_logprobs),
        ("trust scale", trust_scale),
        ("advantages", advantages),
        ("gate advantages", gate_advantages),
        ("gate mask", gate_mask),
    ):
        if tuple(value.shape) != expected_vector_shape:
            raise ValueError(f"{name} must have shape {expected_vector_shape}")

    joint_log_ratio = (
        projected_thought_logprobs.sum(-1) - old_thought_logprobs.sum(-1)
    )
    guarded_log_ratio = joint_log_ratio.clamp(-ratio_guard, ratio_guard)
    denominator = action_denominator.to(
        joint_log_ratio.device
    ).clamp_min(1)
    thought_loss = -(guarded_log_ratio.exp() * advantages).sum() / denominator
    projection_fraction = (trust_scale < 1.0).sum() / denominator
    gate_loss, gate_clip_fraction, _ = clipped_policy_loss(
        new_gate_logprobs,
        old_gate_logprobs,
        gate_advantages,
        gate_mask,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        denominator=action_denominator,
        estimate_kl=False,
    )
    optional_gate_baseline = (
        gate_advantages.detach() * gate_mask
    ).sum() / action_denominator.clamp_min(1)
    return (
        thought_loss + gate_loss + optional_gate_baseline,
        projection_fraction,
        gate_clip_fraction,
    )


def sampled_reverse_kl(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Per-factor k3 estimate of ``KL(old behavior || current policy)``.

    Rollout actions are sampled from the old policy. For
    ``log_ratio = log(new) - log(old)``, the expectation under those actions
    of ``exp(log_ratio) - 1 - log_ratio`` is the reverse KL. Keeping the
    factors separate until after k3 avoids exponentiating the potentially
    enormous joint ratio of the 512-D diagonal Gaussian.
    """
    if new_logprobs.shape != old_logprobs.shape:
        raise ValueError("new and old log-probabilities must match")
    log_ratio = new_logprobs - old_logprobs
    return torch.expm1(log_ratio) - log_ratio


def update_minibatch(
    wrapper: LatentThoughtModel,
    critic: SeparateCritic,
    batch: LatentRolloutBatch,
    optimizers: dict[str, torch.optim.Optimizer],
    value_only: bool = False,
    positive_lm_weight: float = 0.0,
    positive_reward_threshold: float = 0.5,
    thought_pg_coef: float = 1.0,
    # Kept in lockstep with the CLI defaults so callers that omit them
    # exercise the SHIPPED arm; the pair must satisfy the biconditional
    # enforced below (nonzero coefficient <=> clip mode 'none').
    thought_reverse_kl_coef: float = 0.5,
    thought_clip_mode: str = "none",
    thought_trust_epsilon: float = 0.03,
    thought_projection_penalty_coef: float = 1.0,
    gate_pg_coef: float = 1.0,
    gate_entropy_coef: float = 0.0,
    actor_step: bool = True,
    critic_step: bool = True,
    policy_action_denominator: torch.Tensor | None = None,
    gate_action_denominator: torch.Tensor | None = None,
    positive_token_denominator: torch.Tensor | None = None,
    value_action_denominator: torch.Tensor | None = None,
    gae_lambda_alpha: float = 0.05,
    replay_max_trajectories: int = 32,
    replay_attention_budget: int = 4 * 1024 * 1024,
    replay_bucket: int = 1,
    replay_slot_budget: int | None = None,
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
    """
    backbone = wrapper.backbone
    if critic_step:
        zero_optimizers(optimizers, "critic")
    if actor_step and "actor" in optimizers:
        zero_optimizers(optimizers, "actor")
    if replay_max_trajectories < 1:
        raise ValueError("replay max trajectories must be positive")
    if replay_attention_budget < 1:
        raise ValueError("replay attention budget must be positive")
    if (
        not math.isfinite(thought_reverse_kl_coef)
        or thought_reverse_kl_coef < 0
    ):
        raise ValueError(
            "thought reverse KL coefficient must be finite and nonnegative"
        )
    if thought_clip_mode not in ("projected", "joint", "per_dim", "none"):
        raise ValueError(
            f"unknown thought clip mode {thought_clip_mode!r}; "
            "expected 'projected', 'joint', 'per_dim', or 'none'"
        )
    # Exactly one THINK trust mechanism per run. The surrogate-side modes
    # (projection / ratio clip) and the reverse-KL penalty are alternative
    # solutions to the same problem; mixing them makes neither term's
    # contribution attributable, so the two conditions below are the two
    # halves of one biconditional: coefficient nonzero <=> mode 'none'.
    if thought_clip_mode == "none" and thought_reverse_kl_coef == 0.0:
        raise ValueError(
            "thought clip mode 'none' removes the surrogate trust region, so "
            "--thought-reverse-kl-coef must be nonzero; an unconstrained "
            "512-D Gaussian surrogate is what the v22 joint-clip run showed "
            "cannot bound aggregate drift"
        )
    if thought_clip_mode != "none" and thought_reverse_kl_coef != 0.0:
        raise ValueError(
            f"thought clip mode {thought_clip_mode!r} and a nonzero "
            "--thought-reverse-kl-coef are alternative trust mechanisms, not "
            "layers of one, and must never be combined: the clip modes bound "
            "movement inside the surrogate while the penalty prices realized "
            "aggregate divergence of the acting policy. Use "
            f"--thought-clip-mode none with the penalty, or {thought_clip_mode!r} "
            "with --thought-reverse-kl-coef 0"
        )
    if (
        not math.isfinite(thought_projection_penalty_coef)
        or thought_projection_penalty_coef < 0.0
    ):
        raise ValueError(
            "thought projection penalty coefficient must be finite and "
            "nonnegative"
        )
    if not value_only and not batch.statistics_refreshed:
        # The explicit flag also covers pinned-EMIT batches, whose zero-width
        # thought tensors cannot express "not yet refreshed" as a width
        # mismatch.
        raise RuntimeError(
            "actor update requires refresh_old_statistics after rollout"
        )

    positive = batch.reward_scalar >= positive_reward_threshold
    thought_actions = (batch.gate_actions == THINK).float() * batch.action_mask
    optional_think_actions = (
        (batch.gate_actions == THINK).float() * batch.gate_mask
    )
    forced_initial_think_actions = thought_actions - optional_think_actions
    emit_actions = (batch.gate_actions == EMIT).float() * batch.action_mask
    denominators = {
        "action": batch.action_mask.sum().clamp_min(1),
        "gate": batch.gate_mask.sum().clamp_min(1),
        "emit": batch.emit_mask.sum().clamp_min(1),
        "thought": thought_actions.sum().clamp_min(1),
    }
    if policy_action_denominator is None:
        policy_action_denominator = batch.action_mask.sum()
    if gate_action_denominator is None:
        gate_action_denominator = batch.gate_mask.sum()
    if value_action_denominator is None:
        value_action_denominator = batch.action_mask.sum()
    if positive_token_denominator is None:
        positive_token_denominator = (
            batch.emit_mask * positive[:, None]
        ).sum()
    zero = batch.action_mask.new_zeros(())
    totals = {
        key: zero.clone()
        for key in (
            "value_loss", "value_sum", "value_target_sum", "target_entropy_sum",
            "policy_loss", "thought_reverse_kl_penalty",
            "policy_clip", "emit_policy_clip",
            "thought_policy_clip", "thought_gate_policy_clip", "gate_kl_sum",
            "gate_entropy_sum", "gate_entropy_bonus",
            "emit_probability_sum",
            "positive_lm", "renderer_kl_sum", "thought_kl_sum",
            "advantage_sum", "advantage_square_sum",
            "optional_think_advantage_sum", "forced_initial_think_advantage_sum",
            "thought_advantage_sum", "emit_advantage_sum", "target_square_sum",
            "residual_sum", "residual_square_sum",
            "joint_abs_log_ratio_max", "thought_dim_abs_log_ratio_max",
            "thought_joint_abs_log_ratio_max",
            "harmful_positive_log_ratio_max",
            "thought_log_sigma_sum", "thought_log_sigma_square_sum",
            "thought_sigma_sum", "thought_expected_noise_norm_sum",
            "thought_realized_noise_norm_sum", "thought_normalized_noise_square_sum",
            "thought_mean_norm_sum",
            "thought_trust_sum", "thought_trust_max",
            "thought_projection_penalty",
        )
    }
    totals["thought_log_sigma_min"] = zero.new_full((), float("inf"))
    totals["thought_log_sigma_max"] = zero.new_full((), float("-inf"))

    action_counts = batch.action_mask.sum(1)
    lambdas = (
        torch.ones_like(action_counts)
        if value_only
        else length_adaptive_lambda(action_counts, gae_lambda_alpha)
    )
    advantages, value_targets = generalized_advantage_and_return_targets(
        batch.rewards,
        batch.old_values,
        batch.action_mask,
        lambdas,
    )
    advantages = advantages.detach()
    value_targets = value_targets.detach()
    # Standardize THINK advantages (projected mode): the projected surrogate
    # has no ratio band making it scale-invariant, so the trust and penalty
    # coefficients are calibrated against unit-scale advantages. Centering
    # matters as much as scaling — without it a low-variance pool (all
    # trajectories sharing an outcome) turns mean/std into an unbounded
    # gradient amplifier; the subtracted per-minibatch constant is a
    # standard score-function baseline (the projected Gaussian normalizes
    # to one, so E[b * grad ratio] = 0). Computed over the complete
    # minibatch's actions; EMIT and the gate keep raw advantages (their
    # clip decisions never depended on this scale).
    advantage_action_count = batch.action_mask.sum().clamp_min(1)
    advantage_scale_mean = (
        advantages * batch.action_mask
    ).sum() / advantage_action_count
    thought_advantage_normalizer = (
        (advantages.square() * batch.action_mask).sum()
        / advantage_action_count
        - advantage_scale_mean.square()
    ).clamp_min(0).sqrt().clamp_min(1e-6)

    # One host transfer for the whole minibatch replaces the blocking
    # ``bool(think_mask.any())`` device round-trip previously inside every
    # replay shard, each of which stalled the CPU's launch-ahead on all
    # queued GPU work before it.
    row_has_think = (
        ((batch.gate_actions == THINK) & batch.action_mask.bool())
        .any(dim=1)
        .cpu()
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

    for shard_index, (
        microbatch,
        rows,
        stream_length,
        host_rows,
    ) in enumerate(iter_length_aware_microbatches(
        batch,
        replay_max_trajectories,
        replay_attention_budget,
        replay_bucket,
        slot_budget=replay_slot_budget,
    )):
        micro_advantages = advantages[rows, :stream_length]
        micro_value_targets = value_targets[rows, :stream_length]
        local_action_count = microbatch.action_mask.sum()
        value_logits = critic.value_logits(microbatch)
        value_ce = critic.support.cross_entropy(value_logits, micro_value_targets)
        local_value_numerator = (value_ce * microbatch.action_mask).sum()
        weighted_value_loss = (
            local_value_numerator / value_action_denominator.clamp_min(1)
        )
        finite_guards.append(
            (
                shard_index,
                "value",
                weighted_value_loss.detach(),
                {},
            )
        )
        weighted_value_loss.backward()
        with torch.no_grad():
            values = critic.support.to_expected_scalar(value_logits)
            # During critic pretraining there is no actor objective, but the
            # diagnostics must still measure calibration against the CURRENT
            # critic.  The MC training target is independent of the baseline
            # at lambda=1; reporting the earlier zero-baseline GAE made every
            # successful action look positively advantageous by construction.
            diagnostic_advantages = (
                micro_value_targets - values if value_only else micro_advantages
            )
            target_probs = critic.support.project(micro_value_targets)
            target_entropy = -(
                target_probs * target_probs.clamp_min(1e-20).log()
            ).sum(-1)
            micro_thought_actions = (
                (microbatch.gate_actions == THINK).float()
                * microbatch.action_mask
            )
            micro_optional_thinks = (
                (microbatch.gate_actions == THINK).float()
                * microbatch.gate_mask
            )
            micro_forced_initial_thinks = (
                micro_thought_actions - micro_optional_thinks
            )
            micro_emits = (
                (microbatch.gate_actions == EMIT).float()
                * microbatch.action_mask
            )
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
            totals["target_entropy_sum"] += (
                target_entropy * microbatch.action_mask
            ).sum()
            totals["advantage_sum"] += (
                diagnostic_advantages * microbatch.action_mask
            ).sum()
            totals["advantage_square_sum"] += (
                diagnostic_advantages.square() * microbatch.action_mask
            ).sum()
            totals["optional_think_advantage_sum"] += (
                diagnostic_advantages * micro_optional_thinks
            ).sum()
            totals["forced_initial_think_advantage_sum"] += (
                diagnostic_advantages * micro_forced_initial_thinks
            ).sum()
            totals["thought_advantage_sum"] += (
                diagnostic_advantages * micro_thought_actions
            ).sum()
            totals["emit_advantage_sum"] += (
                diagnostic_advantages * micro_emits
            ).sum()
        if value_only:
            continue

        # A stream position is one MDP action. EMIT clips the joint gate+token
        # ratio. THINK clips one ratio per Gaussian dimension while retaining
        # their summed score gradient; its optional gate is clipped once.
        beliefs, predicted, stream_inputs, token_targets = replay_head_inputs(
            wrapper, microbatch
        )
        new_gate_logprobs = wrapper.gate.log_prob(
            microbatch.gate_actions.float(), beliefs
        )

        emit_mask = microbatch.emit_mask.bool()
        emit_features = wrapper.renderer_features(
            stream_inputs[emit_mask], beliefs[emit_mask]
        )
        # Shared with refresh_old_statistics: identical chunk boundaries keep
        # the two eager forwards bit-identical (the age-0 zero-clip canary).
        compact_token_logprobs = compact_emit_token_logprobs(
            backbone, emit_features, token_targets[emit_mask]
        )
        new_token_logprobs = torch.zeros_like(microbatch.old_token_logprobs)
        new_token_logprobs[emit_mask] = compact_token_logprobs

        micro_positive = microbatch.reward_scalar >= positive_reward_threshold
        weighted_positive_lm = positive_example_lm_loss(
            new_token_logprobs,
            microbatch.emit_mask,
            micro_positive,
            denominator=positive_token_denominator,
        )

        think_mask = (
            (microbatch.gate_actions == THINK)
            & microbatch.action_mask.bool()
        )
        new_thought_joint = torch.zeros_like(new_token_logprobs)
        old_thought_joint = torch.zeros_like(new_token_logprobs)
        policy_thought_logprobs = None
        compact_old_thought_logprobs = None
        thought_reverse_kl_factors = None
        weighted_thought_reverse_kl = zero
        projected_policy_thought_logprobs = None
        thought_trust_scale = None
        weighted_projection_penalty = zero
        shard_has_think = bool(row_has_think[host_rows].any())
        if shard_has_think:
            thought_means, thought_targets, _ = select_thought_actions(
                microbatch, predicted
            )
            thought_log_sigma = wrapper.transition.predict_log_sigma(
                beliefs[think_mask]
            )
            if thought_pg_coef != 0.0 or thought_reverse_kl_coef != 0.0:
                new_thought_logprobs = wrapper.transition.per_dim_log_prob(
                    thought_targets,
                    thought_means,
                    thought_log_sigma,
                )
            else:
                with torch.no_grad():
                    new_thought_logprobs = wrapper.transition.per_dim_log_prob(
                        thought_targets,
                        thought_means.detach(),
                        thought_log_sigma.detach(),
                    )
            new_thought_logprobs = new_thought_logprobs.float()
            compact_old_thought_logprobs = (
                microbatch.old_thought_logprobs[think_mask].float()
            )
            if thought_reverse_kl_coef != 0.0:
                thought_reverse_kl_factors = sampled_reverse_kl(
                    new_thought_logprobs,
                    compact_old_thought_logprobs,
                )
                weighted_thought_reverse_kl = (
                    thought_reverse_kl_coef
                    * thought_reverse_kl_factors.sum()
                    / policy_action_denominator.clamp_min(1)
                )
            else:
                # Preserve behavior-KL telemetry for the coefficient-zero
                # ablation without retaining k3 activations or traversing its
                # backward graph.
                with torch.no_grad():
                    thought_reverse_kl_factors = sampled_reverse_kl(
                        new_thought_logprobs.detach(),
                        compact_old_thought_logprobs,
                    )
            if thought_pg_coef == 0.0:
                policy_thought_logprobs = new_thought_logprobs.detach()
            else:
                policy_thought_logprobs = (
                    new_thought_logprobs.detach()
                    + thought_pg_coef
                    * (new_thought_logprobs - new_thought_logprobs.detach())
                )
            new_thought_joint = new_thought_joint.masked_scatter(
                think_mask, policy_thought_logprobs.sum(-1)
            )
            old_thought_joint = old_thought_joint.masked_scatter(
                think_mask, compact_old_thought_logprobs.sum(-1)
            )
            if thought_clip_mode == "projected":
                behavior_thought_means = (
                    microbatch.old_thought_means[think_mask].float()
                )
                behavior_thought_log_sigmas = (
                    microbatch.old_thought_log_sigmas[think_mask].float()
                )
                if thought_pg_coef == 0.0:
                    with torch.no_grad():
                        (
                            projected_thought_means,
                            thought_trust_scale,
                            thought_trust_sq,
                        ) = project_thought_means(
                            thought_means.detach().float(),
                            behavior_thought_means,
                            behavior_thought_log_sigmas,
                            thought_trust_epsilon,
                        )
                        projected_policy_thought_logprobs = (
                            wrapper.transition.per_dim_log_prob(
                                thought_targets,
                                projected_thought_means,
                                thought_log_sigma.detach(),
                            ).float()
                        )
                else:
                    (
                        projected_thought_means,
                        thought_trust_scale,
                        thought_trust_sq,
                    ) = project_thought_means(
                        thought_means.float(),
                        behavior_thought_means,
                        behavior_thought_log_sigmas,
                        thought_trust_epsilon,
                    )
                    raw_projected_logprobs = (
                        wrapper.transition.per_dim_log_prob(
                            thought_targets,
                            projected_thought_means,
                            thought_log_sigma,
                        ).float()
                    )
                    projected_policy_thought_logprobs = (
                        raw_projected_logprobs.detach()
                        + thought_pg_coef
                        * (
                            raw_projected_logprobs
                            - raw_projected_logprobs.detach()
                        )
                    )
                    # TRPL projection penalty: pull the raw mean toward its
                    # detached projection so the rollout policy tracks the
                    # trained one. The gap is zero wherever no projection
                    # occurred, so this term never acts inside the region.
                    projection_gap = (
                        thought_means.float()
                        - projected_thought_means.detach()
                    ) * (-behavior_thought_log_sigmas).exp()
                    weighted_projection_penalty = (
                        thought_projection_penalty_coef
                        * thought_pg_coef
                        * projection_gap.square().sum()
                        / policy_action_denominator.clamp_min(1)
                    )
                with torch.no_grad():
                    totals["thought_trust_sum"] += thought_trust_sq.sum()
                    totals["thought_trust_max"] = torch.maximum(
                        totals["thought_trust_max"], thought_trust_sq.max()
                    )
                    totals["thought_projection_penalty"] += (
                        weighted_projection_penalty.detach()
                    )
            elif thought_clip_mode == "none":
                # The reverse-KL penalty is the whole trust mechanism here, so
                # the surrogate scores the RAW Gaussian: no projection, an
                # all-ones trust scale (nothing is ever projected, so the
                # reported projection fraction is identically 0), and no
                # tracking penalty, which exists only to close a raw/projected
                # gap that cannot open.
                projected_policy_thought_logprobs = policy_thought_logprobs
                thought_trust_scale = torch.ones_like(
                    policy_thought_logprobs[..., 0]
                )
                with torch.no_grad():
                    # Closed-form drift telemetry stays live and comparable
                    # with the projected arm; only its use as a constraint is
                    # dropped. thought_trust_epsilon is unused in this mode.
                    _, _, thought_trust_sq = project_thought_means(
                        thought_means.detach().float(),
                        microbatch.old_thought_means[think_mask].float(),
                        microbatch.old_thought_log_sigmas[think_mask].float(),
                        thought_trust_epsilon,
                    )
                    totals["thought_trust_sum"] += thought_trust_sq.sum()
                    totals["thought_trust_max"] = torch.maximum(
                        totals["thought_trust_max"], thought_trust_sq.max()
                    )

        if gate_pg_coef == 0.0:
            policy_gate_logprobs = new_gate_logprobs.detach()
        else:
            policy_gate_logprobs = (
                new_gate_logprobs.detach()
                + gate_pg_coef
                * (new_gate_logprobs - new_gate_logprobs.detach())
            )
        new_joint_logprobs, old_joint_logprobs = joint_action_logprobs(
            policy_gate_logprobs,
            microbatch.old_gate_logprobs,
            new_token_logprobs,
            microbatch.old_token_logprobs,
            new_thought_joint,
            old_thought_joint,
            microbatch.gate_mask,
            microbatch.emit_mask,
        )
        with torch.no_grad():
            joint_log_ratio = new_joint_logprobs - old_joint_logprobs
            active_joint_log_ratio = joint_log_ratio * microbatch.action_mask
            totals["joint_abs_log_ratio_max"] = torch.maximum(
                totals["joint_abs_log_ratio_max"],
                active_joint_log_ratio.abs().max(),
            )
            # PPO intentionally leaves A<0, ratio>1+epsilon unclipped. This
            # tail statistic directly exposes the branch that overflowed in
            # the first frozen-pool run without changing the objective.
            harmful_positive_log_ratio = torch.where(
                (micro_advantages < 0) & microbatch.action_mask.bool(),
                joint_log_ratio.clamp_min(0),
                torch.zeros_like(joint_log_ratio),
            )
            totals["harmful_positive_log_ratio_max"] = torch.maximum(
                totals["harmful_positive_log_ratio_max"],
                harmful_positive_log_ratio.max(),
            )
        weighted_emit_policy_loss, weighted_emit_policy_clip, _ = (
            clipped_policy_loss(
                policy_gate_logprobs * microbatch.gate_mask
                + new_token_logprobs,
                microbatch.old_gate_logprobs * microbatch.gate_mask
                + microbatch.old_token_logprobs,
                micro_advantages,
                microbatch.emit_mask,
                denominator=policy_action_denominator,
                estimate_kl=False,
            )
        )
        weighted_policy_loss = weighted_emit_policy_loss
        weighted_policy_clip = weighted_emit_policy_clip
        if policy_thought_logprobs is not None:
            # 'none' shares this surrogate: with an identity projection and an
            # all-ones trust scale it reduces to the same unclipped IS form
            # under the +/-2 numerical guard, which is exactly the arm we want
            # against 'projected' -- one term differs, not the objective shape.
            if thought_clip_mode in ("projected", "none"):
                (
                    weighted_thought_policy_loss,
                    weighted_thought_policy_clip,
                    weighted_thought_gate_clip,
                ) = projected_thought_policy_loss(
                    policy_gate_logprobs[think_mask],
                    microbatch.old_gate_logprobs[think_mask],
                    projected_policy_thought_logprobs,
                    compact_old_thought_logprobs,
                    thought_trust_scale,
                    (micro_advantages[think_mask] - advantage_scale_mean)
                    / thought_advantage_normalizer,
                    microbatch.gate_mask[think_mask],
                    policy_action_denominator,
                    gate_advantages=micro_advantages[think_mask],
                )
            else:
                thought_loss_function = (
                    joint_thought_policy_loss
                    if thought_clip_mode == "joint"
                    else per_dimension_thought_policy_loss
                )
                (
                    weighted_thought_policy_loss,
                    weighted_thought_policy_clip,
                    weighted_thought_gate_clip,
                ) = thought_loss_function(
                    policy_gate_logprobs[think_mask],
                    microbatch.old_gate_logprobs[think_mask],
                    policy_thought_logprobs,
                    compact_old_thought_logprobs,
                    micro_advantages[think_mask],
                    microbatch.gate_mask[think_mask],
                    policy_action_denominator,
                )
            weighted_policy_loss = (
                weighted_policy_loss + weighted_thought_policy_loss
            )
            weighted_policy_clip = (
                weighted_policy_clip
                + weighted_thought_policy_clip
                + weighted_thought_gate_clip
            )
            totals["thought_policy_clip"] += weighted_thought_policy_clip
            totals["thought_gate_policy_clip"] += weighted_thought_gate_clip
        totals["emit_policy_clip"] += weighted_emit_policy_clip

        # Normalize over every optional gate decision in the complete
        # optimizer minibatch, matching the policy surrogate's global
        # normalization. The auxiliary objective deliberately updates only
        # the Bernoulli head. Sending a high-coefficient entropy gradient into
        # the shared belief trunk could erase useful state features instead of
        # making the gate use those features less deterministically.
        # gate_pg_coef=0 remains a true gate freeze, including this term.
        weighted_gate_entropy_bonus = (
            gate_entropy_coef
            * gate_pg_coef
            * (
                wrapper.gate.entropy(beliefs.detach())
                * microbatch.gate_mask
            ).sum()
            / gate_action_denominator.clamp_min(1)
        )

        actor_total = (
            weighted_policy_loss
            + weighted_thought_reverse_kl
            + weighted_projection_penalty
            + positive_lm_weight * weighted_positive_lm
            - weighted_gate_entropy_bonus
        )
        finite_guards.append(
            (
                shard_index,
                "actor",
                actor_total.detach(),
                {
                    "policy": weighted_policy_loss.detach(),
                    "thought_reverse_kl": weighted_thought_reverse_kl.detach(),
                    "projection_penalty": (
                        weighted_projection_penalty.detach()
                    ),
                    "positive_lm": weighted_positive_lm.detach(),
                    "gate_entropy_bonus": (
                        weighted_gate_entropy_bonus.detach()
                    ),
                },
            )
        )
        actor_total.backward()
        with torch.no_grad():
            gate_log_ratio = (
                new_gate_logprobs - microbatch.old_gate_logprobs
            )
            renderer_log_ratio = (
                new_token_logprobs - microbatch.old_token_logprobs
            )
            totals["policy_loss"] += weighted_policy_loss.detach()
            totals["thought_reverse_kl_penalty"] += (
                weighted_thought_reverse_kl.detach()
            )
            totals["policy_clip"] += weighted_policy_clip
            totals["gate_entropy_bonus"] += weighted_gate_entropy_bonus.detach()
            totals["gate_kl_sum"] += (
                (torch.expm1(gate_log_ratio) - gate_log_ratio)
                * microbatch.gate_mask
            ).sum()
            totals["gate_entropy_sum"] += (
                wrapper.gate.entropy(beliefs) * microbatch.gate_mask
            ).sum()
            totals["emit_probability_sum"] += (
                wrapper.gate.emit_logit(beliefs).sigmoid()
                * microbatch.gate_mask
            ).sum()
            totals["renderer_kl_sum"] += (
                (torch.expm1(renderer_log_ratio) - renderer_log_ratio)
                * microbatch.emit_mask
            ).sum()
            totals["positive_lm"] += weighted_positive_lm.detach()
            if shard_has_think:
                detached_log_sigma = thought_log_sigma.detach().float()
                residual = thought_targets.float() - thought_means.detach().float()
                normalized_residual = residual * (-detached_log_sigma).exp()
                totals["thought_mean_norm_sum"] += (
                    thought_means.detach().float().norm(dim=-1).sum()
                )
                totals["thought_log_sigma_sum"] += detached_log_sigma.sum()
                totals["thought_log_sigma_square_sum"] += (
                    detached_log_sigma.square().sum()
                )
                totals["thought_log_sigma_min"] = torch.minimum(
                    totals["thought_log_sigma_min"], detached_log_sigma.min()
                )
                totals["thought_log_sigma_max"] = torch.maximum(
                    totals["thought_log_sigma_max"], detached_log_sigma.max()
                )
                totals["thought_sigma_sum"] += detached_log_sigma.exp().sum()
                totals["thought_expected_noise_norm_sum"] += (
                    (2.0 * detached_log_sigma).exp().sum(-1).sqrt().sum()
                )
                totals["thought_realized_noise_norm_sum"] += (
                    residual.square().sum(-1).sqrt().sum()
                )
                totals["thought_normalized_noise_square_sum"] += (
                    normalized_residual.square().sum()
                )
                thought_log_ratio = (
                    new_thought_logprobs
                    - microbatch.old_thought_logprobs[think_mask]
                )
                totals["thought_dim_abs_log_ratio_max"] = torch.maximum(
                    totals["thought_dim_abs_log_ratio_max"],
                    thought_log_ratio.abs().max(),
                )
                totals["thought_joint_abs_log_ratio_max"] = torch.maximum(
                    totals["thought_joint_abs_log_ratio_max"],
                    thought_log_ratio.sum(-1).abs().max(),
                )
                totals["thought_kl_sum"] += (
                    thought_reverse_kl_factors.detach().sum()
                )

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
    # Pinned-EMIT batches store zero-width thoughts: keep the per-dimension
    # thought telemetry finite (zero) instead of dividing by a zero width or
    # reporting the untouched +/-inf extrema.
    thought_dim = max(batch.old_thought_logprobs.size(-1), 1)
    for extremum in ("thought_log_sigma_min", "thought_log_sigma_max"):
        totals[extremum] = torch.where(
            torch.isfinite(totals[extremum]),
            totals[extremum],
            torch.zeros_like(totals[extremum]),
        )
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
        "value_excess_ce": (
            totals["value_loss"] / action_denom
            - totals["target_entropy_sum"] / action_denom
        ),
        "think_advantage_mean": (
            totals["optional_think_advantage_sum"]
            / optional_think_actions.sum().clamp_min(1)
        ),
        "forced_initial_think_advantage_mean": (
            totals["forced_initial_think_advantage_sum"]
            / forced_initial_think_actions.sum().clamp_min(1)
        ),
        "thought_advantage_mean": (
            totals["thought_advantage_sum"] / denominators["thought"]
        ),
        "emit_advantage_mean": (
            totals["emit_advantage_sum"] / emit_actions.sum().clamp_min(1)
        ),
        "think_action_count": optional_think_actions.sum(),
        "thought_action_count": thought_actions.sum(),
        "action_count": batch.action_mask.sum(),
        "gate_action_count": batch.gate_mask.sum(),
        "emit_action_count": emit_actions.sum(),
        "forced_initial_think_action_count": forced_initial_think_actions.sum(),
    }
    if value_only:
        metric_tensors["critic_grad_norm"] = gradient_norm_tensor(
            critic.parameters()
        )
        if critic_step:
            step_optimizers(optimizers, "critic")
        return scalar_tensors_to_floats(metric_tensors)

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
        "adapter_grad_norm": gradient_norm_tensor(wrapper.adapter.parameters()),
        "critic_grad_norm": gradient_norm_tensor(critic.parameters()),
        "gate_grad_norm": gradient_norm_tensor(wrapper.gate.parameters()),
        "sigma_grad_norm": gradient_norm_tensor(
            wrapper.transition.log_sigma_head.parameters()
        ),
        "thought_mean_grad_norm": gradient_norm_tensor(
            wrapper.transition.mean_head.parameters()
        ),
    }
    if critic_step:
        step_optimizers(optimizers, "critic")
    if actor_step and "actor" in optimizers:
        step_optimizers(optimizers, "actor")
    metric_tensors.update(
        policy_loss=totals["policy_loss"],
        thought_reverse_kl_penalty=totals["thought_reverse_kl_penalty"],
        policy_clip_fraction=(
            totals["policy_clip"]
            * policy_action_denominator.clamp_min(1)
            / (
                policy_action_denominator
                + optional_think_actions.sum()
            ).clamp_min(1)
        ),
        emit_policy_clip_fraction=(
            totals["emit_policy_clip"]
            * policy_action_denominator.clamp_min(1)
            / denominators["emit"]
        ),
        thought_policy_clip_fraction=(
            totals["thought_policy_clip"]
            * policy_action_denominator.clamp_min(1)
            / denominators["thought"]
        ),
        thought_gate_policy_clip_fraction=(
            totals["thought_gate_policy_clip"]
            * policy_action_denominator.clamp_min(1)
            / optional_think_actions.sum().clamp_min(1)
        ),
        gate_entropy=totals["gate_entropy_sum"] / denominators["gate"],
        gate_entropy_bonus=totals["gate_entropy_bonus"],
        emit_probability=totals["emit_probability_sum"] / denominators["gate"],
        positive_lm_loss=totals["positive_lm"],
        positive_fraction=positive.float().mean(),
        gate_behavior_kl=totals["gate_kl_sum"] / denominators["gate"],
        renderer_behavior_kl=totals["renderer_kl_sum"] / denominators["emit"],
        thought_behavior_kl_joint=(
            totals["thought_kl_sum"] / denominators["thought"]
        ),
        thought_behavior_kl_per_dim=(
            totals["thought_kl_sum"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_trust_d_mean=(
            totals["thought_trust_sum"] / denominators["thought"]
        ),
        thought_trust_d_max=totals["thought_trust_max"],
        thought_projection_penalty=totals["thought_projection_penalty"],
        policy_behavior_kl_per_action=(
            totals["gate_kl_sum"]
            + totals["renderer_kl_sum"]
            + totals["thought_kl_sum"]
        ) / policy_action_denominator.clamp_min(1),
        joint_abs_log_ratio_max=totals["joint_abs_log_ratio_max"],
        thought_dim_abs_log_ratio_max=(
            totals["thought_dim_abs_log_ratio_max"]
        ),
        thought_joint_abs_log_ratio_max=(
            totals["thought_joint_abs_log_ratio_max"]
        ),
        harmful_positive_log_ratio_max=(
            totals["harmful_positive_log_ratio_max"]
        ),
        advantage_mean=advantage_mean,
        advantage_std=advantage_variance.sqrt(),
        reward=batch.reward_scalar.mean(),
        thought_log_sigma_mean=(
            totals["thought_log_sigma_sum"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_log_sigma_std=(
            totals["thought_log_sigma_square_sum"]
            / (denominators["thought"] * thought_dim)
            - (
                totals["thought_log_sigma_sum"]
                / (denominators["thought"] * thought_dim)
            ).square()
        ).clamp_min(0).sqrt(),
        thought_log_sigma_min=totals["thought_log_sigma_min"],
        thought_log_sigma_max=totals["thought_log_sigma_max"],
        thought_sigma_mean=(
            totals["thought_sigma_sum"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_expected_noise_norm=(
            totals["thought_expected_noise_norm_sum"] / denominators["thought"]
        ),
        thought_realized_noise_norm=(
            totals["thought_realized_noise_norm_sum"] / denominators["thought"]
        ),
        thought_normalized_noise_rms=(
            totals["thought_normalized_noise_square_sum"]
            / (denominators["thought"] * thought_dim)
        ).sqrt(),
        thought_mean_norm=(
            totals["thought_mean_norm_sum"] / denominators["thought"]
        ),
        thought_mean_weight_rms=(
            wrapper.transition.mean_head.output_gain.detach().abs()
            * wrapper.transition.mean_head.weight.detach().square().mean().sqrt()
        ),
        thought_mean_bias_rms=(
            wrapper.transition.mean_head.bias.detach().square().mean().sqrt()
        ),
        thought_mean_output_gain=(
            wrapper.transition.mean_head.output_gain.detach()
        ),
        thought_adapter_weight_rms=(
            wrapper.adapter.projection.weight.detach().square().mean().sqrt()
        ),
        thought_adapter_bias_rms=(
            wrapper.adapter.projection.bias.detach().square().mean().sqrt()
        ),
        thought_log_sigma_raw_bias_mean=(
            wrapper.transition.log_sigma_head.bias.detach().mean()
        ),
        thought_log_sigma_weight_rms=(
            wrapper.transition.log_sigma_head.residual_gain.detach().abs()
            * wrapper.transition.log_sigma_head.weight.detach().square().mean().sqrt()
        ),
        thought_log_sigma_residual_gain=(
            wrapper.transition.log_sigma_head.residual_gain.detach()
        ),
        **grad_norms,
    )
    metrics = scalar_tensors_to_floats(metric_tensors)
    metrics.update(
        gate_pg_coef=float(gate_pg_coef),
        thought_pg_coef=float(thought_pg_coef),
        thought_reverse_kl_coef=float(thought_reverse_kl_coef),
        positive_lm_weight=float(positive_lm_weight),
        gate_entropy_coef=float(gate_entropy_coef),
    )
    return metrics


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
) -> dict[str, float]:
    """Evaluate the just-updated policy on its behavior trajectories.

    On behavior-age 0, optimized ratios are exactly one before the first actor
    step; clipping therefore cannot reveal how far that step moved the deployed
    policy. This read-only replay measures the actual post-step drift
    without reusing trajectories for a gradient. It runs only at the explicit
    diagnostic cadence because it costs one additional policy forward.
    ``replay_function`` deliberately stays eager in the trainer: invoking the
    grad-enabled training artifact under this function's no-grad context would
    force Dynamo/AOTAutograd to compile a second guarded graph, while enabling
    gradients here retains a full replay graph and can exceed peak memory.
    """
    if not batches:
        raise ValueError("post-update drift requires at least one rollout batch")
    device = next(wrapper.parameters()).device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    totals = {
        "gate_kl": zero.clone(),
        "renderer_kl": zero.clone(),
        "thought_kl": zero.clone(),
        "joint_abs_log_ratio_max": zero.clone(),
        "thought_joint_abs_log_ratio_max": zero.clone(),
        "gate_count": zero.clone(),
        "emit_count": zero.clone(),
        "thought_count": zero.clone(),
        "action_count": zero.clone(),
    }
    for stored_batch in batches:
        # Actor behavior pools live in ordinary CPU memory so a 64-prompt
        # frozen-policy pool does not retain many GiB of dense fp32 thoughts
        # and old per-dimension Gaussian log-probabilities on the GPU. Stream
        # one prompt group at a time for this infrequent diagnostic, exactly
        # as the gradient-bearing update path does below.
        batch = (
            stored_batch
            if stored_batch.kind.device == device
            else stored_batch.to(device)
        )
        # Per-shard branch tables, one host transfer per stored batch (free
        # for CPU pools) instead of two blocking device syncs per shard.
        row_has_emit = stored_batch.emit_mask.bool().any(dim=1).cpu()
        row_has_think = (
            (
                (stored_batch.gate_actions == THINK)
                & stored_batch.action_mask.bool()
            )
            .any(dim=1)
            .cpu()
        )
        for microbatch, _, _, host_rows in iter_length_aware_microbatches(
            batch,
            replay_max_trajectories,
            replay_attention_budget,
            replay_bucket,
            slot_budget=replay_slot_budget,
        ):
            beliefs, predicted, stream_inputs, token_targets = replay_function(
                wrapper, microbatch
            )
            gate_logprobs = wrapper.gate.log_prob(
                microbatch.gate_actions.float(), beliefs
            ).float()
            gate_log_ratio = gate_logprobs - microbatch.old_gate_logprobs.float()

            emit_mask = microbatch.emit_mask.bool()
            token_logprobs = torch.zeros_like(microbatch.old_token_logprobs).float()
            if bool(row_has_emit[host_rows].any()):
                emit_logits = wrapper.backbone.logits_from_features(
                    wrapper.renderer_features(
                        stream_inputs[emit_mask], beliefs[emit_mask]
                    )
                )
                compact_token_logprobs = (
                    emit_logits.float()
                    .log_softmax(-1)
                    .gather(-1, token_targets[emit_mask][..., None])
                    .squeeze(-1)
                )
                token_logprobs[emit_mask] = compact_token_logprobs
            token_log_ratio = (
                token_logprobs - microbatch.old_token_logprobs.float()
            )

            think_mask = (
                (microbatch.gate_actions == THINK)
                & microbatch.action_mask.bool()
            )
            thought_joint = torch.zeros_like(token_logprobs)
            old_thought_joint = torch.zeros_like(token_logprobs)
            thought_log_ratio = None
            if bool(row_has_think[host_rows].any()):
                thought_means, thought_targets, _ = select_thought_actions(
                    microbatch, predicted
                )
                thought_log_sigma = wrapper.transition.predict_log_sigma(
                    beliefs[think_mask]
                )
                thought_logprobs = wrapper.transition.per_dim_log_prob(
                    thought_targets, thought_means, thought_log_sigma
                ).float()
                old_thought_logprobs = microbatch.old_thought_logprobs[
                    think_mask
                ].float()
                thought_log_ratio = thought_logprobs - old_thought_logprobs
                thought_joint[think_mask] = thought_logprobs.sum(-1)
                old_thought_joint[think_mask] = old_thought_logprobs.sum(-1)

            new_joint, old_joint = joint_action_logprobs(
                gate_logprobs,
                microbatch.old_gate_logprobs.float(),
                token_logprobs,
                microbatch.old_token_logprobs.float(),
                thought_joint,
                old_thought_joint,
                microbatch.gate_mask,
                microbatch.emit_mask,
            )
            joint_log_ratio = new_joint - old_joint
            action_mask = microbatch.action_mask.float()
            totals["joint_abs_log_ratio_max"] = torch.maximum(
                totals["joint_abs_log_ratio_max"],
                (joint_log_ratio * action_mask).abs().max(),
            )
            totals["gate_kl"] += (
                (torch.expm1(gate_log_ratio) - gate_log_ratio)
                * microbatch.gate_mask
            ).sum()
            totals["renderer_kl"] += (
                (torch.expm1(token_log_ratio) - token_log_ratio)
                * microbatch.emit_mask
            ).sum()
            if thought_log_ratio is not None:
                totals["thought_kl"] += (
                    torch.expm1(thought_log_ratio) - thought_log_ratio
                ).sum()
                totals["thought_joint_abs_log_ratio_max"] = torch.maximum(
                    totals["thought_joint_abs_log_ratio_max"],
                    thought_log_ratio.sum(-1).abs().max(),
                )
            totals["gate_count"] += microbatch.gate_mask.sum()
            totals["emit_count"] += microbatch.emit_mask.sum()
            totals["thought_count"] += think_mask.sum()
            totals["action_count"] += action_mask.sum()
        del batch

    return scalar_tensors_to_floats(
        {
            "kl/post_update_gate_behavior": (
                totals["gate_kl"] / totals["gate_count"].clamp_min(1)
            ),
            "kl/post_update_renderer_behavior": (
                totals["renderer_kl"] / totals["emit_count"].clamp_min(1)
            ),
            "kl/post_update_thought_behavior_joint": (
                totals["thought_kl"] / totals["thought_count"].clamp_min(1)
            ),
            "kl/post_update_policy_behavior_per_action": (
                (
                    totals["gate_kl"]
                    + totals["renderer_kl"]
                    + totals["thought_kl"]
                )
                / totals["action_count"].clamp_min(1)
            ),
            "ratio/post_update_joint_abs_log_max": totals[
                "joint_abs_log_ratio_max"
            ],
            "ratio/post_update_thought_joint_abs_log_max": totals[
                "thought_joint_abs_log_ratio_max"
            ],
        }
    )


def save_checkpoint(
    path: Path,
    wrapper: LatentThoughtModel,
    critic: SeparateCritic,
    optimizers: dict[str, torch.optim.Optimizer],
    step: int,
    args: argparse.Namespace,
    sampler: MathPromptSampler,
    warmup_step: int,
    actor_init_provenance: dict | None = None,
) -> None:
    reasoning_mode = getattr(args, "reasoning_mode", "latent")
    payload = {
        "step": step,
        "value_warmup_step": warmup_step,
        "execution_schema": EXECUTION_SCHEMA,
        "prompt_order_schema": PROMPT_ORDER_SCHEMA,
        "math_data_identity": sampler.dataset_identity,
        "reward_schema": REWARD_SCHEMA,
        "reasoning_mode": reasoning_mode,
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "rollout_policy_schema": rollout_policy_schema_for_mode(reasoning_mode),
        "thought_input_schema": THOUGHT_INPUT_SCHEMA,
        "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
        "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
        "model": wrapper.state_dict(),
        "critic": critic.state_dict(),
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "args": vars(args),
        "sampler_cursor": sampler.cursor,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
        "actor_init_provenance": actor_init_provenance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


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
        )
    else:
        for path in (history_dir / "latest.json", output / "bench_answers.html"):
            if path.exists():
                path.unlink()
    return removed


def build_arg_parser() -> argparse.ArgumentParser:
    """The CLI surface, separate from main() so the shipped defaults
    and the argv-level guards are testable without running training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--math-data", default="postraining/data/dapo-math-17k.parquet")
    parser.add_argument(
        "--exclude-modules", default="",
        help="comma-separated extra_info.module names to drop from --math-data "
        "(module-tagged datasets only); names absent from the data are an error",
    )
    # The reasoning mode fixes the rollout policy family for the whole run:
    # "latent" samples the THINK/EMIT gate and Gaussian thoughts (current
    # behavior); "cot" pins every gate decision to EMIT with the full token
    # budget (token chain of thought); "none" pins EMIT, teacher-forces an
    # "Answer:" prefix onto the prompt, and budgets only the answer itself.
    parser.add_argument(
        "--reasoning-mode",
        choices=("latent", "cot", "none"),
        default="latent",
    )
    # none-mode emission budget: the final answer value plus its terminator.
    parser.add_argument("--answer-tokens", type=int, default=24)
    # Prompt budget: DAPO prompts longer than this keep their TAIL (the
    # question and answer-format instruction sit at the end). None derives a
    # backbone default after the checkpoint loads: fresh PoPE 1024, nano 512.
    parser.add_argument("--prompt-tokens", type=int, default=None)
    parser.add_argument("--continuation-tokens", type=int, default=None)
    # Compute-scaled VAPO topology: sample n=16 responses for 64 prompts under
    # one frozen behavior policy, then shuffle prompt groups once and take four
    # disjoint B256 optimizer minibatches. This activates PPO's behavior-policy
    # clipping without trajectory reuse. VAPO uses the same n=16 but a larger
    # 512-prompt / 8192-trajectory pool and B512 minibatches.
    parser.add_argument("--prompts-per-rollout", type=int, default=64)
    parser.add_argument("--prompts-per-minibatch", type=int, default=16)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    # One pass only: repeated PPO epochs reuse the same generated trajectories.
    parser.add_argument("--ppo-epochs", type=int, default=1)
    # Total generated-slot budget per trajectory (thinks + emits).  Thinking
    # is never forcibly interrupted; overthinking costs emitted tokens and
    # therefore reward. 0 means 4x the emit cap capped to the backbone
    # context; None derives the backbone default (fresh: the explicit
    # 4096-slot side of the 1024-prompt + 4096-stream contract; nano: 512).
    # Pinned-EMIT modes ignore this — every slot is a token, so the stream
    # budget equals the emit cap.
    parser.add_argument("--max-stream-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    # One general rate for actor and critic. The fresh-policy experiment starts
    # every actor-side optimizer state empty from the critic-warm checkpoint;
    # using the critic's 3e-4 rate also removes the prior hand-tuned split
    # between trunk, renderer, gate, adapter, and continuous-policy heads.
    # 5e-5 is the empirically stable RL rate across the latent-VAPO runs;
    # the old 3e-4 default (a pretraining-scale rate) caused behavior-KL
    # spikes and policy collapse when a job omitted --learning-rate.
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    # Trunk update geometry. "muon" mirrors pretraining: block matrices step
    # under NewtonSchulz5-orthogonalized momentum while embeddings, readout,
    # gains, and every RL-only head stay under AdamW. Old checkpoints
    # (pre-Muon-split) resume with "adamw".
    parser.add_argument(
        "--trunk-optimizer", choices=("muon", "adamw"), default="muon"
    )
    # Default: --learning-rate scaled by pretraining's Muon:generic-Adam
    # ratio (0.025 / 0.015). NOTE: at equal nominal LR a Muon step moves each
    # element ~1/sqrt(model_dim) as far as AdamW, so this default under-moves
    # the trunk relative to the AdamW baseline; it is the conservative anchor
    # for the planned LR sweep, not a tuned optimum.
    parser.add_argument("--muon-learning-rate", type=float, default=None)
    # Defaults to --muon-learning-rate. The critic trunk is from-scratch
    # (genuine pretraining regime), so it may tolerate a higher rate.
    parser.add_argument("--critic-muon-learning-rate", type=float, default=None)
    # Trunk-optimizer migration: resume model/critic/step/prompt-cursor from
    # a checkpoint whose optimizer layout differs, starting all optimizer
    # state empty instead of loading it.
    parser.add_argument(
        "--reset-optimizers-on-resume", action="store_true"
    )
    # With the anchored support (default): interior divisions of [0, 1], so
    # bin width is 1/value_bins and the total head width is
    # value_bins + 1 + 2 * value_margin_bins. With --no-value-anchored-support:
    # the legacy total bin count over [0, 1] edges.
    parser.add_argument("--value-bins", type=int, default=101)
    # Dreamer3's exact-zero bucket generalized to both ends of the unit
    # target range: 0 and 1 become bin CENTERS with margin bins beyond each,
    # so the dominant exact-0/exact-1 verifier targets project symmetrically
    # instead of decoding a truncation bias ~0.8*sigma inward.
    parser.add_argument(
        "--value-anchored-support",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    # Bins beyond each anchor. margin + 0.5 half-widths must cover >= 3 sigma
    # of the label Gaussian or the truncation bias the anchors exist to
    # remove comes back through the support edge.
    parser.add_argument("--value-margin-bins", type=int, default=4)
    # HL-Gauss projection sigma as a fraction of bin width. cleanrl v215 /
    # Dreamer4 used 2.0; the dg_v25 critic ablations walked it down (0.75,
    # then 0.5 on a much coarser grid), and sharper labels also shrink what
    # remains of any boundary bias proportionally.
    parser.add_argument("--value-sigma-ratio", type=float, default=1.0)
    # Head bias starts at the projected prior. The CE loss is a divergence in
    # DISTRIBUTION space, so the prior belongs on the target distribution's
    # MODE, not its mean: the optimal constant output is the mixture of
    # projected targets (~92% of targets are exactly 0), which no single
    # projected scalar can match, so the best one sits on the dominant mode.
    # 0 is an exact bin center under the anchored support, so project(0) is
    # symmetric and untruncated. Measured against the warmup target moments
    # (mean 0.0136, var 0.0068), KL(optimum || project(prior)) is 0.68 nats at
    # 0.0, 1.65 at 0.015 (the target MEAN), and 9.84 at the old 0.05 --
    # which the near-frozen bias (AdamW at 5e-5) then takes ~1e3 steps to
    # unwind through head.weight alone.
    parser.add_argument("--value-prior", type=float, default=0.0)
    # Initialization only: the bounded output of the state-dependent
    # log-sigma head. Zero-init weights make noise state-independent at step 0;
    # -3 gives std 0.050 and expected 512-D noise norm 1.13. The fresh
    # adapter starts at exact zero, so neither exploration nor the fresh mean
    # enters the recurrent trunk until replay gradients open its affine map.
    # exp(-3.0) ~= 0.050 per-dim noise: expected noise norm at 512 dims is
    # 0.050*sqrt(512) ~= 1.13, i.e. ~42% of a 2.7 thought mean norm (~22%
    # at the observed 5.2 drift ceiling). -2.5 was considered and rejected
    # (noise norm 1.86 ~= 69% of mean: too much); raising the MEAN head's
    # init instead is the open alternative if exploration needs more range.
    parser.add_argument("--thought-log-sigma-init", type=float, default=-3.0)
    # Initialization only. The orthogonal map gives an RMS-normalized belief
    # an exactly controlled mean RMS without weakening matrix optimization.
    parser.add_argument("--thought-mean-gain-init", type=float, default=0.1)
    # Gradient multiplier for the thought factor inside the joint action log
    # probability. The forward ratio stays exact; 0 detaches only that factor
    # as a control arm while token/gate gradients still train the trunk.
    parser.add_argument("--thought-pg-coef", type=float, default=1.0)
    # Dreamer4-style reverse KL from the frozen rollout behavior policy to the
    # current diagonal-Gaussian thought policy. The factorwise k3 estimator is
    # summed over latent dimensions but divided by ALL policy actions, so 0.3
    # has action-level scale. 0.3 is Dreamer4's own weight
    # (dreamer4.py pmpo_kl_div_loss_weight, pmpo_reverse_kl=True): its
    # kl_div is KL(behavior || current) summed over action dimensions and
    # masked-meaned over positions, the same direction and normalization
    # convention as sampled_reverse_kl here.
    #
    # THE default trust mechanism as of v24 (it was retired to an ablation
    # knob in v21). The v23 measurement is why: the mean PROJECTION bounds
    # only the projected mean inside the surrogate, while the raw acting
    # policy drifts through the Muon-owned trunk under the LM/gate/renderer
    # losses. Job 363 ran at a median raw KL of 0.0424 nats with 68% of
    # updates above 0.03 -- the projection cannot see that channel, and this
    # penalty is the only term that acts on it directly.
    #
    # MUTUALLY EXCLUSIVE with every surrogate-side trust mode. The penalty
    # and the projection/ratio clip are two different SOLUTIONS to the same
    # problem, not two layers of one: the penalty prices realized aggregate
    # divergence of the acting policy, the projection hard-constrains
    # closed-form mean movement inside the surrogate. Running both makes the
    # measurement uninterpretable -- neither term's contribution can be
    # attributed -- so a nonzero coefficient REQUIRES
    # --thought-clip-mode none, and any other clip mode requires 0 here.
    parser.add_argument("--thought-reverse-kl-coef", type=float, default=0.5)
    # v23 default: TRPL-style Mahalanobis mean projection onto the behavior
    # trust region. The sampled joint ratio of a 512-D Gaussian is
    # noise-dominated (log-ratio ~ N(-KL, 2*KL), so at any real drift the
    # v21/v22 'joint' clip decision fired on sampled noise, not on policy
    # movement, and its unclipped harmful side carried e^3+ importance
    # weights). The projection measures movement in closed form from stored
    # behavior means instead. 'joint' and 'per_dim' remain ablation arms.
    #
    # 'none' is THE v24 default: it removes the surrogate-side trust
    # mechanism entirely -- no projection, no ratio band, no tracking
    # penalty -- leaving --thought-reverse-kl-coef as the only constraint on
    # THINK drift. The two families bound different things (closed-form mean
    # movement inside the surrogate vs realized aggregate divergence of the
    # acting policy) and are alternative solutions, so exactly one is active
    # in any run; see --thought-reverse-kl-coef for the exclusion rule. The
    # +/-2 log-ratio guard still applies in 'none': it is a numerical bound
    # on the 512-D Gaussian tail, not a trust region.
    parser.add_argument(
        "--thought-clip-mode",
        choices=("projected", "joint", "per_dim", "none"),
        default="none",
    )
    # Squared-Mahalanobis trust radius per THINK action, in behavior-sigma
    # units (= twice the Gaussian KL at frozen sigma; 0.03 ~= 0.015 nats).
    # The default is the TRPL reference BaseProjectionLayer mean bound
    # (mean 0.03; its cov bound 1e-3 becomes relevant only once sigma is
    # unpinned and the covariance projection lands). Movement beyond the
    # radius is projected back: radial gradients vanish, tangential ones
    # survive.
    parser.add_argument("--thought-trust-epsilon", type=float, default=0.03)
    # TRPL projection penalty: pulls the raw mean toward its (detached)
    # projection so the rollout policy tracks the trained one. Identically
    # zero while no projection occurs — this is a constraint hinge, not an
    # always-on KL penalty.
    parser.add_argument(
        "--thought-projection-penalty-coef", type=float, default=1.0
    )
    # Head-only Bernoulli entropy bonus, averaged over optional gate
    # decisions. Kept small enough that it cannot outweigh the gate's
    # advantage signal: at 1e-2 (and even 4e-3-scale) the bonus dragged the
    # gate toward 50/50 while think advantage stayed pinned negative on a
    # zero-reward base policy, compounding THINK-step derailment.
    parser.add_argument("--gate-entropy-coef", type=float, default=1e-4)
    # Freeze the gate for the first N steps: the gate learns "don't think"
    # from a clean binary signal far faster than the 512-dim thought content
    # can learn to be useful, so exploration dies before content training
    # bites (v3: think fraction 0.094 -> 0.04 by step 700 with think
    # advantage pinned negative).  Holding the gate at its init probability
    # gives the thought/adapter channels a head start; gate PPO switches on
    # at step N against whatever thought quality that warmup earned.
    parser.add_argument("--gate-freeze-steps", type=int, default=0)
    # Length-adaptive GAE lambda (core.length_adaptive_lambda): VAPO's
    # horizon alpha*l with a floor of min(l, 1/alpha).  The raw alpha=0.05
    # formula clamps lambda to 0 at this run's 12-23-action trajectories
    # (TD(0): terminal reward credits nothing more than one step back —
    # audited as the think-fraction ratchet); the floor restores
    # whole-trajectory credit for short answers while keeping VAPO's
    # variance control for long ones.
    parser.add_argument("--gae-lambda-alpha", type=float, default=0.05)
    # Initial THINK probability via the gate head's bias (weights stay zero,
    # so the gate is still belief-independent at init).  The uniform 50/50
    # start is reward-starved at cold start (measured Jul 18): interleaving
    # untrained noisy thoughts into half the stream collapses bench accuracy
    # from ~14% emit-only to ~1e-4, leaving every PPO advantage at critic
    # noise.  Starting emit-heavy keeps rollouts in the competent regime
    # (abundant within-group reward variance) while the gate stays free to
    # think more wherever it pays.
    parser.add_argument("--init-think-probability", type=float, default=0.1)
    parser.add_argument("--positive-lm-weight", type=float, default=0.1)
    parser.add_argument("--positive-reward-threshold", type=float, default=0.5)
    parser.add_argument(
        "--nearby-reward-max",
        type=float,
        default=0.1,
        help="maximum reward for a wrong, terminated numeric final answer",
    )
    # VAPO paper: 50 value-pretraining steps before policy updates.
    parser.add_argument("--value-warmup-steps", type=int, default=50)
    # The BPB guard is a do-no-harm regression check, not an optimization
    # target. A deterministic 2M-token prefix takes ~2.5s on the 5090 versus
    # ~75s for all 62M validation tokens, while still giving the guard ample
    # precision to detect renderer drift. Use 0 explicitly for a final,
    # challenge-comparable full-validation measurement.
    parser.add_argument(
        "--bpb-every", type=int, default=DEFAULT_PERIODIC_EVAL_EVERY
    )
    parser.add_argument(
        "--bpb-val-tokens", type=int, default=DEFAULT_BPB_GUARD_TOKENS,
        help="deterministic validation-prefix size for the BPB guard "
        f"(default: {DEFAULT_BPB_GUARD_TOKENS}; 0 = full set); guard values "
        "remain comparable only within one setting",
    )
    parser.add_argument(
        "--bpb-only", action="store_true",
        help="evaluate the BPB guard once and exit (for timing/config checks)",
    )
    parser.add_argument(
        "--bench-only", action="store_true",
        help="evaluate the full held-out benchmark and exit; requires a "
        "dedicated --output when used with --resume",
    )
    parser.add_argument("--bench-only-repeats", type=int, default=1)
    # The full 1024-answer / 4096-stream AIME eval is intentionally final-only
    # by default; periodic copies would dominate the posttraining workload.
    parser.add_argument("--aime-every", type=int, default=0)
    parser.add_argument("--aime-data", default="postraining/data/aime-2024.parquet")
    parser.add_argument("--aime-samples", type=int, default=32)
    parser.add_argument("--aime-max-tokens", type=int, default=None)
    # Rollout positions are sequential, so batching all samples of a problem
    # into one rollout is nearly free parallelism; lower this only if VRAM
    # becomes the constraint.
    parser.add_argument("--aime-chunk", type=int, default=32)
    # The easier benchmark of record alongside AIME: held-out DeepMind
    # interpolate problems at exactly the mathmix QA training difficulty,
    # where a 27M model can show a real accuracy curve.
    parser.add_argument(
        "--bench-data", default="postraining/data/deepmind-interpolate-easy.parquet"
    )
    parser.add_argument(
        "--bench-every", type=int, default=DEFAULT_PERIODIC_EVAL_EVERY
    )
    parser.add_argument("--bench-samples", type=int, default=8)
    parser.add_argument(
        "--bench-max-rows",
        type=int,
        default=0,
        help="fixed hash-selected benchmark prompt count (0 evaluates all)",
    )
    parser.add_argument("--bench-max-tokens", type=int, default=None)
    # Batch multiple problem groups into the same left-padded GPU rollout.
    # This is a trajectory rather than prompt count so avg@8 and avg@32 use
    # comparable memory; replay-free eval storage makes 128 rows practical.
    parser.add_argument("--eval-batch-trajectories", type=int, default=128)
    parser.add_argument(
        "--eval-tail-batch", type=int, default=16,
        help="single compiled survivor-batch size (0 disables compaction)",
    )
    parser.add_argument(
        "--rollout-tail-batch", type=int, default=16,
        help="single compiled training survivor-batch size "
        "(0 disables compaction)",
    )
    # Compile the dynamic-prefix one-token model step used only by eval.
    # Static full-cache CUDA graphs are intentionally avoided: measured
    # attention over all 5K cache slots was 2.6x slower than eager narrowing.
    parser.add_argument(
        "--eval-compile", action=argparse.BooleanOptionalAction, default=True
    )
    # Compile the narrow-cache one-token training rollout exactly like eval.
    # Tensor-valued positions keep one graph valid across the whole stream;
    # stable batch shapes avoid recompilation at termination boundaries.
    parser.add_argument(
        "--rollout-compile", action=argparse.BooleanOptionalAction, default=True
    )
    # Route the fixed-size compacted training tail (--rollout-tail-batch
    # survivors, ~85% of decode iterations) through persistent static caches
    # and a CUDA-graph-compiled step. The eval-compile note above measured
    # full-cache attention 2.6x slower than narrowing at 512-row eval widths;
    # at the 16-row tail the step is launch-bound, so the graph should win —
    # off by default until bench_step_compile confirms it at the production
    # shape. Costs a persistent tail-batch x full-stream bf16 cache
    # (~0.7 GiB at 16 x 3584 for the 6-layer nano).
    parser.add_argument(
        "--rollout-tail-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    # Checkpoint in true actor/critic optimizer-update units.
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument("--warmup-save-every", type=int, default=10)
    # Compile the compute-bound parallel replay surfaces independently of the
    # narrow-cache rollout. The rejected old path coupled compilation to
    # full-5K static attention, which measured 2.6x slower.
    parser.add_argument(
        "--compile-replay", action=argparse.BooleanOptionalAction, default=True
    )
    # Stable length-sorted shards are bounded by B*L^2 attention area rather
    # than a fixed row count. This admits all 32 normal ~150-token rows and
    # automatically isolates rare 1K-4K outliers.
    parser.add_argument("--replay-bucket", type=int, default=64)
    parser.add_argument("--replay-max-trajectories", type=int, default=32)
    parser.add_argument(
        "--replay-attention-budget", type=int, default=4 * 1024 * 1024
    )
    # Bounds the LINEAR per-shard memory term: slots x 50257-wide emit
    # logits (plus their autograd-retained log-softmax, ~6 bytes/element in
    # the update path). 8192 slots ~= 2.5 GiB retained per shard. Without
    # this, raising --replay-attention-budget lets short-L shards grow their
    # slot count unboundedly and the vocabulary head OOMs before attention.
    parser.add_argument("--replay-slot-budget", type=int, default=8192)
    # A chunk decodes at its longest row's length, so rows that finished
    # early keep stepping until the batch is narrowed. Compaction can only
    # fire on a sync boundary, which makes these two the knobs that set how
    # much of the decode is spent on already-finished rows.
    parser.add_argument(
        "--rollout-sync-every", type=int, default=16,
        help="decode steps between the live-row check that also gates "
        "compaction (lower = narrower batches sooner, more scalar syncs)",
    )
    parser.add_argument(
        "--rollout-compact-dead-ratio", type=float, default=0.25,
        help="compact once this fraction of the current rollout width has "
        "finished (lower = compact sooner, more KV cache copies)",
    )
    parser.add_argument(
        "--post-update-kl-every",
        type=int,
        default=100,
        help="read-only same-batch policy-drift replay cadence; step 1 is "
        "always measured, 0 disables later measurements",
    )
    # Prompt groups rolled out together as one left-padded batch (measured:
    # the sequential per-group rollout is launch-bound at ~140 W, so stepping
    # groups*samples rows per launch is the utilization lever.
    parser.add_argument("--rollout-groups", type=int, default=16)
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--gate-min-within-group-reward-std", type=float, default=0.01)
    parser.add_argument(
        "--actor-init", default=None,
        help="initialize only the actor from a latent-VAPO checkpoint, while "
        "resetting critic/optimizers and continuing its unused prompt stream",
    )
    parser.add_argument(
        "--actor-critic-init", default=None,
        help="initialize actor and pretrained critic weights from a warmup "
        "checkpoint, preserve optimizer state, and continue its unused prompt stream",
    )
    parser.add_argument(
        "--curriculum-init",
        default=None,
        help="initialize trained actor weights from another reward/dataset, "
        "while resetting critic, optimizers, step, and target prompt cursor",
    )
    parser.add_argument(
        "--consume-all-prompts",
        action="store_true",
        help="consume the target dataset exactly once, allowing one final "
        "undersized optimizer minibatch rather than dropping remainder rows",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--migrate-reverse-kl-resume",
        action="store_true",
        help="explicitly resume a v18 per-dimension-clip checkpoint under "
        "the reverse-KL objective while preserving all training state; a "
        "direct v18-to-v20 resume also requires the v20 execution flag",
    )
    parser.add_argument(
        "--migrate-thought-reverse-kl-resume",
        action="store_true",
        help="explicitly resume a v23 projection-only checkpoint under the "
        "v24 objective, which adds the Dreamer4 reverse-KL penalty on the "
        "raw thought policy; all training state transfers verbatim",
    )
    parser.add_argument(
        "--migrate-v20-execution-resume",
        action="store_true",
        help="explicitly resume v19 learning state/cursor under v20 dense "
        "prefill and survivor-compaction execution; future RNG attribution "
        "is intentionally not bit-exact",
    )
    parser.add_argument(
        "--migrate-joint-clip-resume",
        action="store_true",
        help="explicitly resume a v20 per-dimension-clip/reverse-KL "
        "checkpoint under the v21 joint thought clip (no KL) objective "
        "while preserving all training state; execution is unchanged",
    )
    parser.add_argument(
        "--migrate-anchored-value-resume",
        action="store_true",
        help="explicitly resume a v21 unanchored-value checkpoint under the "
        "v22 anchored support: trunk, adapter, actor, and cursor state "
        "transfer verbatim, while the value head restarts at the projected "
        "prior with fresh critic AdamW state (the old head belongs to a "
        "different grid and cannot be carried over)",
    )
    parser.add_argument(
        "--migrate-zero-adapter-resume",
        action="store_true",
        help="explicitly resume a v15 gain-scaled checkpoint while replacing "
        "only its adapter with the zero-initialized affine and fresh adapter "
        "Adam state; entering v20 also requires both migration flags",
    )
    parser.add_argument(
        "--migrate-projected-thought-resume",
        action="store_true",
        help="explicitly resume a v22 joint-clip checkpoint under the v23 "
        "projected THINK trust region; all training state transfers "
        "verbatim (the pool and its stored behavior statistics are rebuilt "
        "on resume)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Argv-level guards that need no checkpoint. Backbone-dependent
    checks stay in main() because they need the loaded model."""
    if args.replay_max_trajectories < 1:
        parser.error("--replay-max-trajectories must be positive")
    if not math.isfinite(args.gate_entropy_coef) or args.gate_entropy_coef < 0.0:
        parser.error("--gate-entropy-coef must be finite and nonnegative")
    if (
        not math.isfinite(args.thought_reverse_kl_coef)
        or args.thought_reverse_kl_coef < 0.0
    ):
        parser.error(
            "--thought-reverse-kl-coef must be finite and nonnegative"
        )
    # Exactly one THINK trust mechanism per run (biconditional; mirrored in
    # run_latent_vapo so library callers get the same rule).
    if args.thought_clip_mode == "none" and args.thought_reverse_kl_coef == 0.0:
        parser.error(
            "--thought-clip-mode none removes the surrogate trust region, so "
            "--thought-reverse-kl-coef must be nonzero"
        )
    if args.thought_clip_mode != "none" and args.thought_reverse_kl_coef != 0.0:
        parser.error(
            f"--thought-clip-mode {args.thought_clip_mode} and a nonzero "
            "--thought-reverse-kl-coef are alternative trust mechanisms and "
            "must never be combined; pass --thought-clip-mode none to use the "
            "reverse-KL penalty, or --thought-reverse-kl-coef 0 to use the "
            "surrogate-side mode"
        )
    if (
        not math.isfinite(args.thought_trust_epsilon)
        or args.thought_trust_epsilon <= 0.0
    ):
        parser.error("--thought-trust-epsilon must be finite and positive")
    if (
        not math.isfinite(args.thought_projection_penalty_coef)
        or args.thought_projection_penalty_coef < 0.0
    ):
        parser.error(
            "--thought-projection-penalty-coef must be finite and nonnegative"
        )
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("--learning-rate must be finite and positive")
    if args.muon_learning_rate is None:
        # Pretraining ran Muon at 0.025 beside the generic AdamW groups at
        # 0.015; carrying that ratio onto the RL rate is the "proportionate"
        # translation of the pretraining recipe.
        args.muon_learning_rate = args.learning_rate * (0.025 / 0.015)
    if args.critic_muon_learning_rate is None:
        args.critic_muon_learning_rate = args.muon_learning_rate
    for name, value in (
        ("--muon-learning-rate", args.muon_learning_rate),
        ("--critic-muon-learning-rate", args.critic_muon_learning_rate),
    ):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")
    if args.reset_optimizers_on_resume and not args.resume:
        parser.error("--reset-optimizers-on-resume requires --resume")
    if (
        not math.isfinite(args.nearby_reward_max)
        or args.nearby_reward_max < 0.0
        or args.nearby_reward_max >= args.positive_reward_threshold
    ):
        parser.error(
            "--nearby-reward-max must be finite, nonnegative, and below "
            "--positive-reward-threshold"
        )
    if not -5.0 < args.thought_log_sigma_init < 2.0:
        parser.error("--thought-log-sigma-init must be strictly inside (-5, 2)")
    if args.replay_attention_budget < 1:
        parser.error("--replay-attention-budget must be positive")
    if args.replay_slot_budget < 1:
        parser.error("--replay-slot-budget must be positive")
    if args.post_update_kl_every < 0:
        parser.error("--post-update-kl-every must be nonnegative")
    if args.replay_bucket < 1:
        parser.error("--replay-bucket must be positive")
    if args.warmup_save_every < 1:
        parser.error("--warmup-save-every must be positive")
    if args.eval_batch_trajectories < 1:
        parser.error("--eval-batch-trajectories must be positive")
    if args.eval_tail_batch < 0:
        parser.error("--eval-tail-batch must be nonnegative")
    if args.rollout_tail_batch < 0:
        parser.error("--rollout-tail-batch must be nonnegative")
    if args.rollout_tail_graph and args.rollout_tail_batch < 1:
        parser.error("--rollout-tail-graph requires --rollout-tail-batch >= 1")
    if args.rollout_tail_graph and not args.rollout_compile:
        # The tail switch rides the compiled rollout's tensor positions and
        # fixed-size compaction; an eager rollout never engages it.
        parser.error("--rollout-tail-graph requires --rollout-compile")
    if args.bench_only_repeats < 1:
        parser.error("--bench-only-repeats must be positive")
    if args.bench_max_rows < 0:
        parser.error("--bench-max-rows must be nonnegative")
    if args.migrate_zero_adapter_resume and not args.resume:
        parser.error("--migrate-zero-adapter-resume requires --resume")
    if args.migrate_reverse_kl_resume and not args.resume:
        parser.error("--migrate-reverse-kl-resume requires --resume")
    if args.migrate_v20_execution_resume and not args.resume:
        parser.error("--migrate-v20-execution-resume requires --resume")
    if args.migrate_joint_clip_resume and not args.resume:
        parser.error("--migrate-joint-clip-resume requires --resume")
    if args.migrate_anchored_value_resume and not args.resume:
        parser.error("--migrate-anchored-value-resume requires --resume")
    if args.migrate_projected_thought_resume and not args.resume:
        parser.error("--migrate-projected-thought-resume requires --resume")
    if args.migrate_thought_reverse_kl_resume and not args.resume:
        parser.error("--migrate-thought-reverse-kl-resume requires --resume")
    if (
        args.migrate_thought_reverse_kl_resume
        and args.thought_reverse_kl_coef == 0.0
    ):
        parser.error(
            "--migrate-thought-reverse-kl-resume migrates INTO the v24 "
            "reverse-KL objective; it cannot combine with "
            "--thought-reverse-kl-coef 0"
        )
    if args.migrate_anchored_value_resume and not args.value_anchored_support:
        parser.error(
            "--migrate-anchored-value-resume migrates INTO the anchored "
            "support; it cannot combine with --no-value-anchored-support"
        )
    if args.value_bins < 1:
        parser.error("--value-bins must be positive")
    if args.value_margin_bins < 0:
        parser.error("--value-margin-bins must be nonnegative")
    if not math.isfinite(args.value_sigma_ratio) or args.value_sigma_ratio <= 0.0:
        parser.error("--value-sigma-ratio must be finite and positive")
    if (
        args.value_anchored_support
        and args.value_margin_bins + 0.5 < 3.0 * args.value_sigma_ratio
    ):
        parser.error(
            "--value-margin-bins must give the anchors >= 3 sigma of slack "
            "(margin + 0.5 >= 3 * sigma_ratio), or boundary targets decode "
            "the truncation bias the anchored support exists to remove"
        )
    if args.prompts_per_rollout < 1:
        parser.error("--prompts-per-rollout must be positive")
    if args.prompts_per_minibatch < 1:
        parser.error("--prompts-per-minibatch must be positive")
    if args.prompts_per_rollout % args.prompts_per_minibatch:
        parser.error(
            "--prompts-per-rollout must be divisible by --prompts-per-minibatch"
        )
    if args.ppo_epochs != 1:
        parser.error("--ppo-epochs must be 1; trajectory reuse is disabled")
    if args.bpb_val_tokens < 0:
        parser.error("--bpb-val-tokens must be nonnegative")
    exclusive_modes = sum(
        (args.bpb_only, args.bench_only, args.rollout_only)
    )
    if exclusive_modes > 1:
        parser.error(
            "--bpb-only, --bench-only, and --rollout-only are mutually exclusive"
        )
    if (
        args.bench_only
        and args.resume
        and Path(args.output).resolve() == Path(args.resume).resolve().parent
    ):
        parser.error(
            "--bench-only with --resume requires a dedicated --output so "
            "evaluation cannot purge or overwrite the training run"
        )
    initialization_modes = sum(
        option is not None
        for option in (
            args.actor_init,
            args.actor_critic_init,
            args.curriculum_init,
            args.resume,
        )
    )
    if initialization_modes > 1:
        parser.error(
            "--actor-init, --actor-critic-init, --curriculum-init, and "
            "--resume are mutually exclusive"
        )
    if args.samples_per_prompt < 2 or args.samples_per_prompt % 2:
        parser.error("--samples-per-prompt must be even for the 50/50 forced split")
    if args.prompts_per_minibatch * args.samples_per_prompt != 256:
        parser.error(
            "optimizer minibatches must contain exactly 256 trajectories"
        )
    for name, samples in (
        ("--aime-samples", args.aime_samples),
        ("--bench-samples", args.bench_samples),
    ):
        if samples < 2 or samples % 2:
            parser.error(f"{name} must be even for the 50/50 forced split")
    if (
        (args.bench_every > 0 or args.bench_only)
        and args.bench_samples < CAPTURE_SAMPLES_PER_PROBLEM
    ):
        parser.error(
            "--bench-samples must be at least "
            f"{CAPTURE_SAMPLES_PER_PROBLEM} for automatic answer capture"
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    backbone = load_model(args.checkpoint, device)
    is_nano = backbone.architecture.startswith("nanogpt_mini")
    if not is_nano and not backbone.architecture.endswith(
        "probes_pope_belief_attached_ce_onepass_2k"
    ):
        raise ValueError(
            "belief-renderer VAPO requires a nanogpt_mini checkpoint or one "
            "pretrained with [token_latent, raw_belief] CE; old "
            "predicted-latent probe "
            f"architecture {backbone.architecture!r} is incompatible"
        )

    # Backbone-derived context contract: fresh PoPE executes 5x its pretrained
    # window, so RL keeps the full 1024-token prompt plus a 4096-slot stream.
    # Nano's half-truncate RoPE (base 1024) has no extrapolation story, so RL
    # stays strictly inside the pretraining window the checkpoint records
    # (train_seq_len; 1024 for checkpoints predating the field). The response
    # budget targets the standard ~1k tokens for DAPO math as soon as the
    # window affords it after the prompt.
    if is_nano:
        context_tokens = getattr(backbone, "train_context_tokens", 1024)
        default_prompt_tokens = 512
        default_response_tokens = min(
            1024, (context_tokens - default_prompt_tokens) // 2
        )
    else:
        context_tokens = POSTTRAIN_CONTEXT_TOKENS
        default_prompt_tokens = POSTTRAIN_PROMPT_TOKENS
        default_response_tokens = POSTTRAIN_RESPONSE_TOKENS
    if args.prompt_tokens is None:
        args.prompt_tokens = default_prompt_tokens
    if args.continuation_tokens is None:
        args.continuation_tokens = default_response_tokens
    if args.aime_max_tokens is None:
        args.aime_max_tokens = default_response_tokens
    if args.bench_max_tokens is None:
        args.bench_max_tokens = default_response_tokens
    if args.answer_tokens < 2:
        parser.error("--answer-tokens must fit an answer plus its terminator")

    pin_emit = args.reasoning_mode != "latent"
    rollout_policy_schema = rollout_policy_schema_for_mode(args.reasoning_mode)
    if pin_emit and args.max_stream_steps is not None:
        parser.error(
            "--max-stream-steps is a latent-mode knob; pinned-EMIT modes "
            "always use exactly one stream slot per emitted token"
        )

    def mode_budgets(max_tokens: int) -> tuple[int, int]:
        """(max_new_tokens, max_stream_steps) for one rollout/eval budget."""
        if args.reasoning_mode == "none":
            return args.answer_tokens, args.answer_tokens
        if args.reasoning_mode == "cot":
            return max_tokens, max_tokens
        return max_tokens, min(
            4 * max_tokens, context_tokens - args.prompt_tokens
        )

    if args.reasoning_mode == "latent":
        train_max_new_tokens = args.continuation_tokens
        if args.max_stream_steps is None:
            max_stream_steps = min(
                POSTTRAIN_STREAM_TOKENS, context_tokens - args.prompt_tokens
            )
        elif args.max_stream_steps == 0:
            _, max_stream_steps = mode_budgets(args.continuation_tokens)
        else:
            max_stream_steps = args.max_stream_steps
    else:
        train_max_new_tokens, max_stream_steps = mode_budgets(
            args.continuation_tokens
        )
    # The eval budget always scales with its own emit cap; an explicit
    # --max-stream-steps is a training-rollout knob.
    aime_max_new_tokens, aime_stream_steps = mode_budgets(args.aime_max_tokens)
    bench_max_new_tokens, bench_stream_steps = mode_budgets(
        args.bench_max_tokens
    )
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
    # Full-model RL: every parameter on a deployed policy path trains — trunk,
    # embeddings, fresh thought mean/sigma, renderer, gate, and adapter. The
    # retired pretrained prediction projector remains checkpointed but has no
    # graph edge; the backbone critic probe is frozen and unused.
    wrapper = LatentThoughtModel(backbone).to(device)
    # No module here behaves differently under train(): pin eval mode once so
    # the training flag (a dynamo guard) never flips between the step-0 evals
    # and the training loop and re-specializes the compiled step.
    wrapper.eval()
    if not 0.0 < args.init_think_probability < 1.0:
        raise SystemExit("--init-think-probability must be strictly inside (0, 1)")
    if not math.isfinite(args.thought_mean_gain_init) or (
        args.thought_mean_gain_init <= 0.0
    ):
        raise SystemExit("--thought-mean-gain-init must be finite and positive")
    # The CLI owns both fresh continuous-policy scales. Ordinary resume loads
    # learned values over these; explicit actor restart copies them into the
    # critic-warm payload before loading it.
    wrapper.transition.mean_head.reset_output_gain(
        args.thought_mean_gain_init
    )
    wrapper.transition.reset_noise(args.thought_log_sigma_init)
    with torch.no_grad():
        # P(EMIT) = sigmoid(bias) while the zero-init weights ignore the belief.
        wrapper.gate.head.bias.fill_(
            math.log((1.0 - args.init_think_probability) / args.init_think_probability)
        )
    actor_init_payload = None
    actor_init_provenance = None
    initialization_path = (
        args.actor_critic_init or args.actor_init or args.curriculum_init
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
        if args.actor_critic_init:
            if int(actor_init_payload.get("step", -1)) != 0:
                raise ValueError(
                    "--actor-critic-init requires a pre-policy critic-warmup checkpoint"
                )
            if (
                int(actor_init_payload.get("value_warmup_step", -1))
                < args.value_warmup_steps
            ):
                raise ValueError(
                    "--actor-critic-init checkpoint has incomplete critic warmup"
                )
            if actor_init_payload.get("reward_schema") != REWARD_SCHEMA:
                raise ValueError(
                    "--actor-critic-init checkpoint uses an incompatible reward schema"
                )
        migrate_legacy_wrapper_checkpoint(
            actor_init_payload,
            wrapper,
            initialize_fresh_mean=bool(args.actor_critic_init),
            initialize_fresh_adapter=bool(args.actor_critic_init),
        )
        validate_renderer_checkpoint(
            actor_init_payload,
            initialization_path,
            # A critic-warmup checkpoint has not updated the actor, and its
            # transition head is reset below before the policy is ever used.
            allow_transition_reset=bool(args.actor_critic_init),
            expected_rollout_policy_schema=rollout_policy_schema,
        )
        wrapper.load_state_dict(actor_init_payload["model"], strict=True)
        if args.actor_critic_init:
            # A critic-warmup checkpoint has never trained its actor, so this
            # run owns the initial exploration level. A trained --actor-init
            # source instead preserves its learned state-dependent noise.
            wrapper.transition.reset_noise(args.thought_log_sigma_init)
        actor_init_provenance = {
            "checkpoint": str(initialization_path),
            "critic_initialized": bool(args.actor_critic_init),
            "curriculum_transition": bool(args.curriculum_init),
            "source_step": actor_init_payload.get("step"),
            "source_reward_schema": actor_init_payload.get("reward_schema"),
            "source_execution_schema": actor_init_payload.get("execution_schema"),
            "source_value_warmup_step": actor_init_payload.get(
                "value_warmup_step"
            ),
            "source_sampler_cursor": int(actor_init_payload["sampler_cursor"]),
            "target_sampler_cursor": 0 if args.curriculum_init else int(
                actor_init_payload["sampler_cursor"]
            ),
            "fresh_mean_initialized": bool(args.actor_critic_init),
            "fresh_mean_output_gain": (
                args.thought_mean_gain_init if args.actor_critic_init else None
            ),
            "fresh_log_sigma": (
                args.thought_log_sigma_init if args.actor_critic_init else None
            ),
            "fresh_adapter_initialized": bool(args.actor_critic_init),
            "fresh_adapter_zero_initialized": bool(args.actor_critic_init),
        }
    for parameter in wrapper.parameters():
        parameter.requires_grad_(True)
    if hasattr(backbone, "critic_probe"):
        # Fresh lineage only: the pretrained critic probe stays checkpointed
        # but has no graph edge. Nano backbones carry no probes.
        for parameter in backbone.critic_probe.parameters():
            parameter.requires_grad_(False)

    if args.value_anchored_support:
        value_num_bins, value_v_min, value_v_max = anchored_unit_geometry(
            args.value_bins, args.value_margin_bins
        )
    else:
        value_num_bins, value_v_min, value_v_max = args.value_bins, 0.0, 1.0
    critic = SeparateCritic(
        fresh_trunk(backbone, device),
        num_bins=value_num_bins,
        sigma_ratio=args.value_sigma_ratio,
        v_min=value_v_min,
        v_max=value_v_max,
        prior_value=args.value_prior,
    ).to(device)
    critic.eval()  # no dropout in this architecture; keep norms deterministic
    if args.actor_critic_init:
        init_args = actor_init_payload.get("args", {})
        if not value_support_geometry_matches(init_args, args):
            raise ValueError(
                "--actor-critic-init warm critic was trained on a different "
                "value support geometry (anchored/bins/margin "
                f"{init_args.get('value_anchored_support', False)}/"
                f"{init_args.get('value_bins')}/"
                f"{init_args.get('value_margin_bins')}); rerun critic warmup "
                "under the current flags or start with --actor-init"
            )
        critic.load_state_dict(actor_init_payload["critic"], strict=True)

    optimizers = build_optimizers(
        wrapper,
        critic,
        learning_rate=args.learning_rate,
        trunk_optimizer=args.trunk_optimizer,
        muon_learning_rate=args.muon_learning_rate,
        critic_muon_learning_rate=args.critic_muon_learning_rate,
    )
    if args.actor_critic_init:
        # A step-0 critic-warm checkpoint has never stepped its actor. Its
        # optimizer state is therefore empty and carries no momentum to
        # preserve; loading its obsolete three-group layout would only couple
        # the gate and recurrent adapter again. Preserve the trained critic's
        # optimizer state and start the untouched actor optimizer in the
        # current six-group layout.
        source_optimizers = actor_init_payload["optimizers"]
        for name, source in source_optimizers.items():
            if name.startswith("actor") and source["state"]:
                raise ValueError(
                    "--actor-critic-init requires a pristine actor optimizer"
                )
        critic_names = [name for name in optimizers if name.startswith("critic")]
        missing = [name for name in critic_names if name not in source_optimizers]
        extra = [
            name
            for name in source_optimizers
            if name.startswith("critic") and name not in optimizers
        ]
        if missing or extra:
            raise ValueError(
                "--actor-critic-init checkpoint optimizer layout "
                f"{sorted(n for n in source_optimizers if n.startswith('critic'))} "
                f"does not match the --trunk-optimizer {args.trunk_optimizer} "
                f"layout {sorted(critic_names)}; rerun the critic warmup or "
                "pass the matching --trunk-optimizer"
            )
        for name in critic_names:
            optimizers[name].load_state_dict(source_optimizers[name])
        reassert_learning_rates(
            {name: optimizers[name] for name in critic_names}, args
        )

    tokenizer = load_posttraining_tokenizer(
        backbone.architecture, FreshHyperparameters.tokenizer_path
    )
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
    aime_modal_baseline = modal_answer_baseline(aime_rows)
    all_bench_rows = (
        load_unique_math_rows(args.bench_data)
        if (args.bench_every > 0 or args.bench_only) and not args.rollout_only
        else []
    )
    bench_rows = deterministic_math_subset(all_bench_rows, args.bench_max_rows)
    bench_dataset_baseline = modal_answer_baseline(all_bench_rows)
    bench_subset_baseline = modal_answer_baseline(bench_rows)
    math_rows = load_unique_math_rows(args.math_data)
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
    math_modal_baseline = modal_answer_baseline(math_rows)
    modal_solution = f"Answer: {math_modal_baseline['answer']}"
    modal_shaped_rewards = []
    for row in math_rows:
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
    math_modal_baseline["shaped_reward"] = (
        sum(modal_shaped_rewards) / len(modal_shaped_rewards)
        if modal_shaped_rewards
        else 0.0
    )
    data_identity = math_dataset_identity(args.math_data, args.exclude_modules)
    sampler = MathPromptSampler(
        math_rows, args.seed, dataset_identity=data_identity
    )
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
                    "dataset bytes, exclusions, or ordering"
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
                f"actor{' and critic' if args.actor_critic_init else ''} initialized "
                f"from {initialization_path} at source step "
                f"{actor_init_payload.get('step')} and sampler cursor {actor_cursor}",
                flush=True,
            )
    seq_len = FreshHyperparameters.train_seq_len
    is_gpt2_vocab = is_nano and "gpt2vocab" in backbone.architecture
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
    else:
        luts = baseline.build_sentencepiece_luts(
            tokenizer, FreshHyperparameters.vocab_size, device
        )
    # The BPB guard shares the checkpoint's own tokenizer family, and nano
    # pretrains against its own shard family; keep the guard on the same
    # validation bytes as nano's own pretraining val_bpb (DATA_PATH overrides).
    bpb_val_files = FreshHyperparameters.val_files
    if is_nano:
        default_val_dataset = (
            "data/datasets/fineweb10B_gpt2"
            if is_gpt2_vocab
            else "data/datasets/fineweb_onepass_sp1024"
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

    start_step = 0
    warmup_step = args.value_warmup_steps if args.actor_critic_init else 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        adapter_migration = None
        if args.migrate_zero_adapter_resume:
            adapter_migration = migrate_zero_adapter_resume(payload, wrapper)
        migrate_legacy_wrapper_checkpoint(payload, wrapper)
        validate_renderer_checkpoint(
            payload,
            args.resume,
            expected_rollout_policy_schema=rollout_policy_schema,
        )
        if not resume_execution_schema_compatible(
            payload,
            allow_reverse_kl_migration=args.migrate_reverse_kl_resume,
            allow_performance_migration=args.migrate_v20_execution_resume,
            allow_joint_clip_migration=args.migrate_joint_clip_resume,
            allow_anchored_value_migration=args.migrate_anchored_value_resume,
            allow_projected_thought_migration=(
                args.migrate_projected_thought_resume
            ),
            allow_thought_reverse_kl_migration=(
                args.migrate_thought_reverse_kl_resume
            ),
        ):
            raise ValueError(
                "resume checkpoint execution schema must be "
                f"{EXECUTION_SCHEMA!r}; "
                f"got {payload.get('execution_schema')!r}. Use --actor-init or "
                "--actor-critic-init for an initialization restart, or "
                "the migration flags matching the source: v23 requires "
                "--migrate-thought-reverse-kl-resume; v22 requires that plus "
                "--migrate-projected-thought-resume; v21 additionally requires "
                "--migrate-anchored-value-resume; v20 additionally requires "
                "--migrate-joint-clip-resume; v19 additionally requires "
                "--migrate-v20-execution-resume; v18 requires all of those "
                "plus --migrate-reverse-kl-resume."
            )
        if payload.get("reward_schema") != REWARD_SCHEMA:
            raise ValueError(
                f"resume checkpoint reward schema must be {REWARD_SCHEMA!r}; "
                f"got {payload.get('reward_schema')!r}"
            )
        if payload.get("math_data_identity") != data_identity:
            raise ValueError(
                "resume checkpoint's prompt cursor belongs to different dataset "
                "bytes, exclusions, or ordering"
            )
        resume_seed = payload.get("args", {}).get("seed")
        if resume_seed is not None and int(resume_seed) != args.seed:
            raise ValueError(
                "resume requires the checkpoint's --seed: prompt epochs after "
                f"the first reshuffle from it (checkpoint seed {resume_seed}, "
                f"got {args.seed})"
            )
        wrapper.load_state_dict(payload["model"], strict=True)
        anchored_value_migration = None
        if args.migrate_anchored_value_resume:
            anchored_value_migration = migrate_anchored_value_resume(
                payload, critic
            )
        else:
            resume_args = payload.get("args", {})
            if not value_support_geometry_matches(resume_args, args):
                raise ValueError(
                    "resume checkpoint critic support geometry "
                    "(anchored/bins/margin "
                    f"{resume_args.get('value_anchored_support', False)}/"
                    f"{resume_args.get('value_bins')}/"
                    f"{resume_args.get('value_margin_bins')}) does not match "
                    "the current flags; resume with the checkpoint's value "
                    "support flags"
                )
            critic.load_state_dict(payload["critic"], strict=True)
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
                f"{sorted(optimizers)}; checkpoints from before the Muon "
                "trunk split resume with --trunk-optimizer adamw, or migrate "
                "with --reset-optimizers-on-resume"
            )
        else:
            for name, optimizer in optimizers.items():
                if anchored_value_migration is not None and name == "critic":
                    # The saved critic AdamW moments are positionally keyed
                    # over param groups that include the OLD grid's head
                    # parameters; loading them against the rebuilt head would
                    # misalign shapes. Trunk-matrix Muon state (critic_muon)
                    # is grid-agnostic and loads normally.
                    print(
                        "anchored-value migration: fresh critic AdamW state "
                        "(saved moments belong to the old value head)",
                        flush=True,
                    )
                    continue
                optimizer.load_state_dict(payload["optimizers"][name])
        # The sigma head is learned: the model load above restored it, and
        # (unlike the old fixed-buffer scheme) the CLI must NOT reassert it on
        # resume — --thought-log-sigma-init is an initialization, not a
        # schedule.  Learning rates remain the documented cross-run knobs and
        # are reasserted below.
        reassert_learning_rates(optimizers, args)
        start_step = int(payload["step"])
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
        source_execution_schema = payload.get("execution_schema")
        if source_execution_schema == PREVIOUS_EXECUTION_SCHEMA:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["reverse_kl_resume_migration"] = {
                "source_execution_schema": PREVIOUS_EXECUTION_SCHEMA,
                "thought_reverse_kl_coef": args.thought_reverse_kl_coef,
            }
        if source_execution_schema in {
            PREVIOUS_EXECUTION_SCHEMA,
            PERFORMANCE_COMPATIBLE_EXECUTION_SCHEMA,
        }:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["performance_resume_migration"] = {
                "source_execution_schema": source_execution_schema,
                "target_execution_schema": EXECUTION_SCHEMA,
            }
        if adapter_migration is not None:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["zero_adapter_resume_migration"] = (
                adapter_migration
            )
        if anchored_value_migration is not None:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["anchored_value_resume_migration"] = {
                "source_execution_schema": source_execution_schema,
                "target_execution_schema": EXECUTION_SCHEMA,
                **anchored_value_migration,
            }
        if args.migrate_projected_thought_resume:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["projected_thought_resume_migration"] = {
                "source_execution_schema": source_execution_schema,
                "target_execution_schema": EXECUTION_SCHEMA,
                "thought_trust_epsilon": args.thought_trust_epsilon,
                "thought_projection_penalty_coef": (
                    args.thought_projection_penalty_coef
                ),
            }
        if args.migrate_thought_reverse_kl_resume:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["thought_reverse_kl_resume_migration"] = {
                "source_execution_schema": source_execution_schema,
                "target_execution_schema": EXECUTION_SCHEMA,
                "thought_reverse_kl_coef": args.thought_reverse_kl_coef,
                "thought_clip_mode": args.thought_clip_mode,
            }

    if args.actor_critic_init:
        torch.set_rng_state(actor_init_payload["cpu_rng"])
        torch.cuda.set_rng_state_all(actor_init_payload["cuda_rng"])
        random.setstate(actor_init_payload["python_rng"])

    if args.bpb_only or args.bench_only:
        planned_prompt_count = 0
    elif args.rollout_only:
        planned_prompt_count = args.prompts_per_rollout
    else:
        warmup_updates = (
            max(args.value_warmup_steps - warmup_step, 0) if start_step == 0 else 0
        )
        remaining_actor_steps = max(args.steps - start_step, 0)
        warmup_prompt_count = args.prompts_per_minibatch * warmup_updates
        if args.consume_all_prompts:
            planned_prompt_count, _ = plan_one_pass_training(
                dataset_rows=len(math_rows),
                sampler_cursor=sampler.cursor,
                warmup_updates=warmup_updates,
                remaining_actor_steps=remaining_actor_steps,
                prompts_per_minibatch=args.prompts_per_minibatch,
            )
        else:
            planned_prompt_count = args.prompts_per_minibatch * (
                warmup_updates + remaining_actor_steps
            )
    if sampler.cursor + planned_prompt_count > len(math_rows):
        if args.consume_all_prompts:
            raise ValueError(
                "one-pass run would cross the dataset's first epoch and "
                "reuse prompts: cursor "
                f"{sampler.cursor} + planned {planned_prompt_count} > "
                f"{len(math_rows)} rows"
            )
        final_epoch = (sampler.cursor + planned_prompt_count - 1) // len(
            math_rows
        )
        print(
            f"prompt plan spans {final_epoch + 1} epochs of the "
            f"{len(math_rows)}-row dataset ({planned_prompt_count} prompts); "
            "epochs after the first reshuffle deterministically from --seed",
            flush=True,
        )

    # Rollout and evaluation use the identical dynamic narrow-prefix step.
    # Compile it once: separate wrappers paid the same large cold compilation
    # cost at step-0 eval and again at the first training collect. If eval ever
    # latches an eager fallback, the eval closure below also disables this
    # shared artifact for training before it can be called again.
    compiled_generation_step = None
    if args.rollout_compile or args.eval_compile:
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64
        )
        compiled_generation_step = torch.compile(
            wrapper.step_core,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
            dynamic=True,
        )
    rollout_step_core = (
        compiled_generation_step if args.rollout_compile else None
    )
    eval_step_core = compiled_generation_step if args.eval_compile else None

    # Static-tail CUDA graph for the fixed-size compacted training tail.
    # The caches are allocated ONCE and live for the whole run: a fresh
    # allocation would move the static addresses and force a graph
    # re-record. The artifact wraps the ORIGINAL eager step_core (collect's
    # temporary rollout_step_core patch never reaches it), and it only ever
    # runs under the training autocast at one shape, so it stays a single
    # cudagraph specialization.
    rollout_tail_caches = None
    rollout_tail_step_core = None
    if args.rollout_tail_graph:
        rollout_tail_caches = wrapper.make_static_generation_cache(
            args.rollout_tail_batch,
            args.prompt_tokens + max_stream_steps,
            device,
            dtype=torch.bfloat16,
        )
        rollout_tail_step_core = torch.compile(
            wrapper.step_core,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False,
        )

    # Preserve the eager function for infrequent no-grad diagnostics. Calling
    # the compiled training artifact under no-grad would create a distinct
    # AOTAutograd specialization solely because grad mode is a Dynamo guard.
    diagnostic_replay_head_inputs = replay_head_inputs

    # Replay compilation is independent of rollout. Dynamic B/L plus bounded
    # 64-token buckets lets one artifact cover the length-aware shard plan;
    # no CUDA graph owns outputs that remain live through eager losses/backward.
    trim_multiple = args.replay_bucket if args.compile_replay else 1
    if args.compile_replay:
        torch._dynamo.config.cache_size_limit = 64
        critic.value_logits = torch.compile(
            critic.value_logits,
            # Length buckets still span many GEMM shapes. Max-autotune
            # repeatedly stalls training to benchmark each new regime;
            # default Inductor dispatches them to stable cuBLAS kernels.
            mode="default",
            fullgraph=True,
            dynamic=True,
        )
        compiled_replay = torch.compile(
            replay_head_inputs,
            mode="default",
            fullgraph=True,
            dynamic=True,
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

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output / "metrics.jsonl")
    tensorboard_purge_step = None
    if args.resume:
        removed = logger.purge_after(start_step, warmup_step)
        removed_reports = purge_benchmark_reports_after(output, start_step)
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
                "phase": "latent_vapo_dapo",
                "execution_schema": EXECUTION_SCHEMA,
                "prompt_order_schema": PROMPT_ORDER_SCHEMA,
                "math_data_identity": data_identity,
                "reward_schema": REWARD_SCHEMA,
                "reasoning_mode": args.reasoning_mode,
                "context_tokens": context_tokens,
                "math_modal_answer_baseline": math_modal_baseline,
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "rollout_policy_schema": rollout_policy_schema,
                "thought_input_schema": THOUGHT_INPUT_SCHEMA,
                "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
                "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
                "args": vars(args),
                "base": {
                    "checkpoint": str(args.checkpoint),
                    "architecture": backbone.architecture,
                },
                "actor_init": actor_init_provenance,
                "critic": {
                    "init": (
                        "warm_checkpoint"
                        if args.actor_critic_init
                        else "scratch"
                    ),
                    "checkpoint": (
                        str(args.actor_critic_init)
                        if args.actor_critic_init
                        else None
                    ),
                    "source_execution_schema": (
                        actor_init_payload.get("execution_schema")
                        if args.actor_critic_init
                        else None
                    ),
                    "source_warmup_optimizer_semantics": (
                        "legacy_per_prompt_group"
                        if args.actor_critic_init
                        else None
                    ),
                    "architecture": backbone.architecture,
                    "value_bins": args.value_bins,
                    "value_anchored_support": args.value_anchored_support,
                    "value_margin_bins": args.value_margin_bins,
                    "value_num_bins_total": value_num_bins,
                    "value_v_min": value_v_min,
                    "value_v_max": value_v_max,
                    "value_sigma_ratio": args.value_sigma_ratio,
                    "value_prior": args.value_prior,
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

    def rollout_force_members(groups_count: int, samples: int) -> torch.Tensor:
        """Latent mode's 50/50 forced-THINK split; inert in pinned modes."""
        if pin_emit:
            return torch.zeros(
                groups_count * samples, dtype=torch.bool, device=device
            )
        return half_forced_group_members(groups_count, samples, device)

    def finish_group(
        batch: LatentRolloutBatch,
        row: dict,
        refresh_statistics: bool,
    ) -> LatentRolloutBatch:
        batch = trim_stream(batch, multiple=trim_multiple)
        score_math_rollout(
            batch, row["reward_model"]["ground_truth"], tokenizer, stop_ids,
            answer_style(row),
            args.nearby_reward_max,
            solution_prefix_ids=answer_prefix_ids,
        )
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
            )
        return batch

    def _collect(
        refresh_statistics: bool,
        prompt_count: int,
        offload_to_cpu: bool,
        scoring_pool: ThreadPoolExecutor | None,
    ) -> list[LatentRolloutBatch]:
        """One rollout: a scored prompt group per sampled DAPO problem."""
        rollout_rows = sampler.next_rows(prompt_count)
        prompt_budget = args.prompt_tokens - len(answer_prefix_ids)
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

        if args.rollout_groups <= 1:
            for row, encoded in encoded_rows:
                prompt_ids = torch.tensor(
                    encoded,
                    dtype=torch.long, device=device,
                )
                batch = rollout_continuations(
                    wrapper, prompt_ids[None],
                    train_max_new_tokens, max_stream_steps,
                    args.temperature, args.top_p, stop_ids=stop_ids or None,
                    force_initial_think=rollout_force_members(
                        1, args.samples_per_prompt
                    ),
                    pin_emit=pin_emit,
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
                    tail_caches=rollout_tail_caches,
                    tail_step_core=rollout_tail_step_core,
                )
                if offload_to_cpu:
                    batch = compact_stream_to_device(batch, cpu)
                groups.append(retain_group(batch, row))
                del batch
            return groups
        # Left-padded batched rollout: all chunk groups step together, so
        # each launch carries chunk*samples rows instead of samples — the
        # sequential per-group loop is launch-bound, not compute-bound.
        samples = args.samples_per_prompt
        pending_scored: list = []

        def drain_scored() -> None:
            for future in pending_scored:
                groups.append(future.result())
            pending_scored.clear()

        for chunk_start in range(0, len(encoded_rows), args.rollout_groups):
            encoded_chunk = encoded_rows[
                chunk_start : chunk_start + args.rollout_groups
            ]
            chunk = [row for row, _ in encoded_chunk]
            encoded = [ids for _, ids in encoded_chunk]
            width = max(len(ids) for ids in encoded)
            prompt_ids = torch.zeros(
                (len(chunk), width), dtype=torch.long, device=device
            )
            for index, ids in enumerate(encoded):
                prompt_ids[index, width - len(ids):] = torch.tensor(
                    ids, dtype=torch.long, device=device
                )
            prompt_lengths_cpu = torch.tensor(
                [len(ids) for ids in encoded], dtype=torch.long
            )
            prompt_lengths = prompt_lengths_cpu.to(device)
            batched = rollout_continuations(
                wrapper, prompt_ids, train_max_new_tokens, max_stream_steps,
                args.temperature, args.top_p, stop_ids=stop_ids or None,
                prompt_lengths=prompt_lengths,
                force_initial_think=rollout_force_members(len(chunk), samples),
                pin_emit=pin_emit,
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
                prompt_repeats=samples,
                tail_caches=rollout_tail_caches,
                tail_step_core=rollout_tail_step_core,
            )
            if offload_to_cpu:
                # One chunk-level D2H transfer, then all variable-length
                # splitting, trimming, decoding, and scoring stay on CPU.
                # This avoids one synchronization/transfer per prompt group.
                batched = compact_stream_to_device(batched, cpu)
            expanded_prompt_lengths = prompt_lengths_cpu.repeat_interleave(
                samples
            )
            if not offload_to_cpu:
                expanded_prompt_lengths = expanded_prompt_lengths.to(device)
            split_groups = split_rollout_groups(
                batched, samples, expanded_prompt_lengths
            )
            # Every split owns cloned storage; drop the much larger
            # groups*samples rollout before the first full-stream replay.
            del batched
            if scoring_pool is None:
                for group_index, (group, row) in enumerate(
                    zip(split_groups, chunk, strict=True)
                ):
                    groups.append(retain_group(group, row))
                    # Once retained, remove the source from the temporary
                    # chunk list immediately. This bounds GPU storage in
                    # ordinary collection.
                    split_groups[group_index] = None
                    del group
            else:
                # The previous chunk's futures ran during this chunk's
                # rollout; draining before submitting keeps `groups` in
                # chunk order. Host storage stays bounded at one pending
                # chunk beyond what `groups` retains anyway.
                drain_scored()
                for group, row in zip(split_groups, chunk, strict=True):
                    pending_scored.append(
                        scoring_pool.submit(retain_group, group, row)
                    )
            del split_groups
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
    ) -> list[LatentRolloutBatch]:
        if offload_to_cpu and refresh_statistics:
            raise ValueError(
                "CPU-offloaded collection must defer behavior refresh until "
                "optimizer minibatches are assembled"
            )
        original_step_core = wrapper.step_core
        if rollout_step_core is not None:
            wrapper.step_core = rollout_step_core
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
            with training_autocast():
                return _collect(
                    refresh_statistics,
                    args.prompts_per_rollout
                    if prompt_count is None
                    else prompt_count,
                    offload_to_cpu,
                    scoring_pool,
                )
        finally:
            if scoring_pool is not None:
                scoring_pool.shutdown(wait=True)
            wrapper.step_core = original_step_core

    def training_update(*update_args, **update_kwargs):
        with training_autocast():
            return update_minibatch(*update_args, **update_kwargs)

    def teacher_forced_bpb() -> float:
        """Teacher-forced BPB through the deployed belief renderer."""
        wrapper.eval()
        _, bpb = baseline.eval_val(
            FreshHyperparameters, wrapper, 0, 1, device, 8, val_tokens, *luts
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
            compiled_step_core=eval_step_core,
            compiled_tail_batch=args.eval_tail_batch or None,
            answer_style_override="aime",
            captured_attempts=captured_attempts,
            pin_emit=pin_emit,
            prompt_suffix_ids=answer_prefix_ids,
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
            reward_schema=REWARD_SCHEMA,
        )
        tensorboard.add_scalar("aime/accuracy", metrics["accuracy"], step)
        tensorboard.add_scalar(
            "aime/forced_initial_accuracy",
            metrics["forced_initial_accuracy"],
            step,
        )
        tensorboard.add_scalar(
            "aime/unforced_initial_accuracy",
            metrics["unforced_initial_accuracy"],
            step,
        )
        tensorboard.add_scalar(
            "aime/optional_think_fraction", metrics["think_fraction"], step
        )
        print(f"step:{step} aime_avg@{args.aime_samples}:{metrics['accuracy']:.4f}", flush=True)

    def bench_eval(
        step: int,
        *,
        eval_seed: int | None = None,
        repeat: int | None = None,
    ) -> None:
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
            compiled_step_core=eval_step_core,
            compiled_tail_batch=args.eval_tail_batch or None,
            captured_attempts=captured_attempts,
            pin_emit=pin_emit,
            prompt_suffix_ids=answer_prefix_ids,
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
            reward_schema=REWARD_SCHEMA,
        )
        logger.log(
            type="bench",
            step=step,
            seconds=eval_seconds,
            eval_seed=eval_seed,
            repeat=repeat,
            **metrics,
        )
        tensorboard.add_scalar("bench/accuracy", metrics["accuracy"], step)
        tensorboard.add_scalar(
            "bench/forced_initial_accuracy",
            metrics["forced_initial_accuracy"],
            step,
        )
        tensorboard.add_scalar(
            "bench/unforced_initial_accuracy",
            metrics["unforced_initial_accuracy"],
            step,
        )
        tensorboard.add_scalar(
            "bench/optional_think_fraction", metrics["think_fraction"], step
        )
        print(f"step:{step} bench_avg@{args.bench_samples}:{metrics['accuracy']:.4f}", flush=True)

    if args.bench_only:
        for repeat in range(args.bench_only_repeats):
            # Repeat zero warms the compiled decoder. Subsequent repeats use
            # matched but distinct workloads across evaluator modes, avoiding
            # a single stochastic tail being mistaken for expected speed.
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

    if args.rollout_only:
        metrics = aggregate_diagnostics(collect(), args.samples_per_prompt, stop_ids)
        passed = metrics["within_group_reward_std"] >= args.gate_min_within_group_reward_std
        logger.log(type="rollout_gate", passed=passed, **metrics)
        print(json.dumps({"passed": bool(passed), **metrics}, sort_keys=True))
        tensorboard.close()
        raise SystemExit(0 if passed else 2)

    if start_step == 0 and (warmup_step == 0 or args.actor_critic_init):
        bpb = teacher_forced_bpb()
        logger.log(type="bpb", step=0, val_bpb=bpb)
        tensorboard.add_scalar("guard/val_bpb", bpb, 0)
        print(f"step:0 teacher-forced val_bpb:{bpb:.4f}", flush=True)
        if aime_rows:
            aime_eval(0)
        if bench_rows:
            bench_eval(0)
    if start_step == 0 and warmup_step < args.value_warmup_steps:
        for warmup in range(warmup_step + 1, args.value_warmup_steps + 1):
            warmup_started = time.perf_counter()
            collect_started = warmup_started
            groups = collect(
                refresh_statistics=False,
                prompt_count=args.prompts_per_minibatch,
            )
            torch.cuda.synchronize()
            collect_seconds = time.perf_counter() - collect_started
            update_started = time.perf_counter()
            value_action_denominator = torch.stack(
                [group.action_mask.sum() for group in groups]
            ).sum()
            zero_optimizers(optimizers, "critic")
            group_metrics = [
                training_update(
                    wrapper,
                    critic,
                    group,
                    optimizers,
                    value_only=True,
                    critic_step=False,
                    value_action_denominator=value_action_denominator,
                    replay_max_trajectories=args.replay_max_trajectories,
                    replay_attention_budget=args.replay_attention_budget,
                    replay_bucket=args.replay_bucket,
                    replay_slot_budget=args.replay_slot_budget,
                )
                for group in groups
            ]
            step_optimizers(optimizers, "critic")
            del groups
            update_seconds = time.perf_counter() - update_started
            metrics = {
                key: float(sum(m[key] for m in group_metrics) / len(group_metrics))
                for key in group_metrics[0]
            }
            value_metrics = aggregate_value_diagnostics(group_metrics)
            metrics.update(
                seconds=time.perf_counter() - warmup_started,
                collect_seconds=collect_seconds,
                update_seconds=update_seconds,
                action_count=sum(m["action_count"] for m in group_metrics),
                value_mean=value_metrics["prediction_mean"],
                value_target_mean=value_metrics["target_mean"],
                value_target_variance=value_metrics["target_variance"],
                value_residual_mean=value_metrics["residual_mean"],
                value_residual_variance=value_metrics["residual_variance"],
                explained_variance=value_metrics["explained_variance"],
                value_excess_ce=value_metrics["excess_ce"],
                value_loss=_weighted_metric_mean(
                    group_metrics, "value_loss", "action_count"
                ),
                critic_grad_norm=group_metrics[-1]["critic_grad_norm"],
                think_advantage_mean=_weighted_metric_mean(
                    group_metrics, "think_advantage_mean", "think_action_count"
                ),
                forced_initial_think_advantage_mean=_weighted_metric_mean(
                    group_metrics,
                    "forced_initial_think_advantage_mean",
                    "forced_initial_think_action_count",
                ),
                thought_advantage_mean=_weighted_metric_mean(
                    group_metrics,
                    "thought_advantage_mean",
                    "thought_action_count",
                ),
                emit_advantage_mean=_weighted_metric_mean(
                    group_metrics, "emit_advantage_mean", "emit_action_count"
                ),
            )
            logger.log(type="value_warmup", step=warmup, **metrics)
            warmup_dashboard = {
                "value_warmup/token_weighted_excess_ce": metrics[
                    "value_excess_ce"
                ],
                "value_warmup/prediction_mean": metrics["value_mean"],
                "value_warmup/target_mean": metrics["value_target_mean"],
                "value_warmup/residual_mean": metrics["value_residual_mean"],
                "value_warmup/online_explained_variance": metrics[
                    "explained_variance"
                ],
                "value_warmup/residual/optional_think_mean": metrics[
                    "think_advantage_mean"
                ],
                "value_warmup/residual/forced_initial_think_mean": metrics[
                    "forced_initial_think_advantage_mean"
                ],
                "value_warmup/residual/all_think_mean": metrics[
                    "thought_advantage_mean"
                ],
                "value_warmup/residual/emit_mean": metrics[
                    "emit_advantage_mean"
                ],
                "value_warmup/critic_grad_norm": metrics[
                    "critic_grad_norm"
                ],
            }
            for tag, value in warmup_dashboard.items():
                tensorboard.add_scalar(tag, value, warmup)
            # The optimizer state is sufficient for the next update and for
            # checkpoints. Keeping the accumulated gradient buffers alive
            # overlaps them with the next rollout's KV cache for no benefit.
            zero_optimizers(optimizers, "critic")
            warmup_step = warmup
            if (
                warmup % args.warmup_save_every == 0
                or warmup == args.value_warmup_steps
            ):
                # Preserve the clean actor-initial/critic-warm checkpoint.
                # The rolling checkpoint is overwritten by PPO updates, so
                # it cannot serve as a trustworthy actor restart point.
                checkpoint_name = (
                    "critic_warmup_checkpoint.pt"
                    if warmup == args.value_warmup_steps
                    else "latent_vapo_checkpoint.pt"
                )
                save_checkpoint(
                    output / checkpoint_name,
                    wrapper,
                    critic,
                    optimizers,
                    start_step,
                    args,
                    sampler,
                    warmup_step,
                    actor_init_provenance,
                )

    def crossed_interval(previous: int, current: int, interval: int) -> bool:
        return interval > 0 and current // interval > previous // interval

    step = start_step
    while step < args.steps:
        previous_step = step
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
        groups = collect(
            refresh_statistics=False,
            prompt_count=pool_prompt_count,
            offload_to_cpu=True,
        )
        rollout_metrics = aggregate_diagnostics(
            groups, args.samples_per_prompt, stop_ids
        )
        rollout_metrics.update(
            lockstep_decode_metrics(groups, max(args.rollout_groups, 1))
        )
        minibatch_orders = optimizer_minibatch_orders(
            len(groups),
            args.prompts_per_minibatch,
            allow_partial_final=(
                args.consume_all_prompts and sampler.cursor == len(math_rows)
            ),
        )
        if len(minibatch_orders) != pool_updates:
            raise RuntimeError("rollout pool did not produce the planned updates")

        # Assemble each shuffled optimizer minibatch on CPU with RIGHT tail
        # padding, then refresh all old statistics under the still-frozen
        # behavior policy. The refreshed DEVICE batch is retained and consumed
        # directly by the matching update below — repacking and re-uploading
        # the identical order and width, plus the d2h stat scatter, was pure
        # round-trip overhead. Retention is bounded: past the budget (streams
        # lengthen when thinking grows) the pre-retention path takes over —
        # scatter stats to the CPU groups and let update repack.
        retained_device_batches: dict[int, LatentRolloutBatch] = {}
        retained_bytes_total = 0
        packed_batch_bytes_max = 0
        packed_real_slots = 0
        packed_capacity_slots = 0
        pool_cpu_pack_seconds = 0.0
        pool_h2d_seconds = 0.0
        pool_refresh_seconds = 0.0
        pool_d2h_seconds = 0.0
        old_value_sums = []
        old_value_counts = []
        def pack_minibatch(
            minibatch_order: list[int],
        ) -> tuple[LatentRolloutBatch, list[LatentRolloutBatch], float]:
            selected = [groups[index] for index in minibatch_order]
            pack_started = time.perf_counter()
            packed = pack_rollout_groups_for_replay(selected, pin_memory=True)
            return packed, selected, time.perf_counter() - pack_started

        # Minibatch orders are a disjoint partition of the pool, so a single
        # worker can pack minibatch N+1's pinned host batch while N's H2D
        # and refresh occupy the GPU — the worker only reads groups this
        # loop's scatter fallback never writes in the same iteration.
        pack_pool = ThreadPoolExecutor(max_workers=1)
        try:
            pack_future = pack_pool.submit(
                pack_minibatch, minibatch_orders[0]
            )
            for age, minibatch_order in enumerate(minibatch_orders):
                cpu_batch, selected_groups, pack_elapsed = (
                    pack_future.result()
                )
                pool_cpu_pack_seconds += pack_elapsed
                if age + 1 < len(minibatch_orders):
                    pack_future = pack_pool.submit(
                        pack_minibatch, minibatch_orders[age + 1]
                    )
                transfer_started = time.perf_counter()
                device_batch = cpu_batch.to(device, non_blocking=True)
                torch.cuda.synchronize()
                pool_h2d_seconds += time.perf_counter() - transfer_started
                del cpu_batch
                refresh_started = time.perf_counter()
                with training_autocast():
                    refresh_old_statistics(
                        wrapper,
                        critic,
                        device_batch,
                        max_trajectories=args.replay_max_trajectories,
                        attention_budget=args.replay_attention_budget,
                        bucket_multiple=args.replay_bucket,
                        slot_budget=args.replay_slot_budget,
                    )
                torch.cuda.synchronize()
                pool_refresh_seconds += time.perf_counter() - refresh_started
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
                packed_real_slots += int(
                    (device_batch.kind != PAD_SLOT).sum()
                )
                packed_capacity_slots += device_batch.kind.numel()
                generated = device_batch.action_mask.bool()
                old_value_sums.append(
                    device_batch.old_values[generated].sum()
                )
                old_value_counts.append(generated.sum())
                if (
                    retained_bytes_total + packed_batch_bytes
                    <= RETAINED_MINIBATCH_BUDGET_BYTES
                ):
                    retained_device_batches[age] = device_batch
                    retained_bytes_total += packed_batch_bytes
                else:
                    transfer_started = time.perf_counter()
                    scatter_replay_statistics(device_batch, selected_groups)
                    pool_d2h_seconds += (
                        time.perf_counter() - transfer_started
                    )
                del device_batch
                del selected_groups
        finally:
            pack_pool.shutdown(wait=True)
        old_value_sum = float(torch.stack(old_value_sums).sum())
        old_value_count = int(torch.stack(old_value_counts).sum())
        rollout_metrics["old_value_mean"] = (
            old_value_sum / old_value_count if old_value_count else 0.0
        )
        rollout_metrics["retained_minibatches"] = len(retained_device_batches)
        rollout_metrics["packed_batch_gib_max"] = packed_batch_bytes_max / 2**30
        rollout_metrics["packed_padding_utilization"] = (
            packed_real_slots / packed_capacity_slots
            if packed_capacity_slots
            else 0.0
        )
        rollout_metrics["pool_cpu_pack_seconds"] = pool_cpu_pack_seconds
        rollout_metrics["pool_h2d_seconds"] = pool_h2d_seconds
        rollout_metrics["pool_refresh_seconds"] = pool_refresh_seconds
        rollout_metrics["pool_d2h_seconds"] = pool_d2h_seconds
        torch.cuda.synchronize()
        collect_seconds = time.perf_counter() - collect_started
        rollout_peak_vram_bytes = torch.cuda.max_memory_allocated()
        # Restart the peak counter here so the train rows below report the
        # UPDATE-phase peak (packed minibatch + replay-shard activations),
        # not the collection-phase KV-cache peak that always dominates it.
        # Replay memory knobs (--replay-attention-budget and retained device
        # batches) are tuned against this number.
        torch.cuda.reset_peak_memory_stats()
        rollout_step = step + pool_updates
        logger.log(
            type="rollout", step=rollout_step,
            collect_seconds=collect_seconds,
            pool_updates=pool_updates,
            pool_trajectories=pool_prompt_count * args.samples_per_prompt,
            peak_vram_bytes=rollout_peak_vram_bytes,
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
            "perf/packed_batch_gib_max",
            rollout_metrics["packed_batch_gib_max"],
            rollout_step,
        )
        tensorboard.add_scalar(
            "perf/packed_padding_utilization",
            rollout_metrics["packed_padding_utilization"],
            rollout_step,
        )
        tensorboard.add_scalar(
            "perf/decode_steps_per_chunk_mean",
            rollout_metrics["decode_steps_per_chunk_mean"],
            rollout_step,
        )
        tensorboard.add_scalar(
            "perf/decode_step_utilization",
            rollout_metrics["decode_step_utilization"],
            rollout_step,
        )
        tensorboard.add_scalar(
            "perf/pool_cpu_pack_seconds", pool_cpu_pack_seconds, rollout_step
        )
        tensorboard.add_scalar(
            "perf/pool_h2d_seconds", pool_h2d_seconds, rollout_step
        )
        tensorboard.add_scalar(
            "perf/pool_refresh_seconds", pool_refresh_seconds, rollout_step
        )
        tensorboard.add_scalar(
            "perf/pool_d2h_seconds", pool_d2h_seconds, rollout_step
        )
        rollout_dashboard = rollout_tensorboard_metrics(rollout_metrics)
        # Horizontal anti-collapse calibration: on DAPO, a terminated
        # constant modal answer already earns nontrivial exact and shaped
        # reward. On-policy improvement is meaningful only relative to both.
        rollout_dashboard["reward/modal_constant_exact_baseline"] = float(
            math_modal_baseline["accuracy"]
        )
        rollout_dashboard["reward/modal_constant_shaped_baseline"] = float(
            math_modal_baseline["shaped_reward"]
        )
        for tag, value in rollout_dashboard.items():
            tensorboard.add_scalar(tag, value, rollout_step)

        for behavior_age, minibatch_order in enumerate(minibatch_orders):
            # Actor and critic each take one optimizer step over the same
            # effective trajectory minibatch. Prompt groups and replay shards
            # are memory partitions, not optimizer minibatches.
            update_started = time.perf_counter()
            next_step = step + 1
            zero_optimizers(optimizers, "actor")
            zero_optimizers(optimizers, "critic")
            sigma_parameters = list(
                wrapper.transition.log_sigma_head.parameters()
            )
            sigma_before = [
                parameter.detach().clone() for parameter in sigma_parameters
            ]
            mean_parameters = list(wrapper.transition.mean_head.parameters())
            mean_before = [
                parameter.detach().clone() for parameter in mean_parameters
            ]
            selected_groups = [groups[index] for index in minibatch_order]
            retained_minibatch = retained_device_batches.pop(
                behavior_age, None
            )
            if retained_minibatch is not None:
                # Refreshed on-device in the pool loop above; identical to
                # what repacking selected_groups would rebuild.
                device_minibatch = retained_minibatch
                minibatch_pack_seconds = 0.0
                minibatch_h2d_seconds = 0.0
            else:
                minibatch_pack_started = time.perf_counter()
                cpu_minibatch = pack_rollout_groups_for_replay(
                    selected_groups, pin_memory=True
                )
                minibatch_pack_seconds = (
                    time.perf_counter() - minibatch_pack_started
                )
                minibatch_h2d_started = time.perf_counter()
                device_minibatch = cpu_minibatch.to(device, non_blocking=True)
                torch.cuda.synchronize()
                minibatch_h2d_seconds = (
                    time.perf_counter() - minibatch_h2d_started
                )
                del cpu_minibatch
            (
                policy_action_denominator,
                gate_action_denominator,
                positive_token_denominator,
            ) = actor_minibatch_denominators(
                [device_minibatch], [0], args.positive_reward_threshold
            )
            metrics = training_update(
                wrapper, critic, device_minibatch, optimizers,
                positive_lm_weight=args.positive_lm_weight,
                positive_reward_threshold=args.positive_reward_threshold,
                thought_pg_coef=args.thought_pg_coef,
                thought_reverse_kl_coef=args.thought_reverse_kl_coef,
                thought_clip_mode=args.thought_clip_mode,
                thought_trust_epsilon=args.thought_trust_epsilon,
                thought_projection_penalty_coef=(
                    args.thought_projection_penalty_coef
                ),
                gate_entropy_coef=args.gate_entropy_coef,
                gate_pg_coef=(
                    0.0 if next_step <= args.gate_freeze_steps else 1.0
                ),
                actor_step=False,
                critic_step=False,
                policy_action_denominator=policy_action_denominator,
                gate_action_denominator=gate_action_denominator,
                positive_token_denominator=positive_token_denominator,
                value_action_denominator=policy_action_denominator,
                gae_lambda_alpha=args.gae_lambda_alpha,
                replay_max_trajectories=args.replay_max_trajectories,
                replay_attention_budget=args.replay_attention_budget,
                replay_bucket=args.replay_bucket,
                replay_slot_budget=args.replay_slot_budget,
            )
            # The first minibatch runs against refresh-computed behavior
            # statistics with the actor untouched. Later disjoint minibatches
            # intentionally have behavior age 1..N.
            if behavior_age == 0:
                guard = "policy_clip_fraction"
                if metrics[guard] > 1e-6:
                    print(
                        f"WARNING step {next_step}: behavior-age-0 {guard}="
                        f"{metrics[guard]:.3e} (expected exactly 0; "
                        "refresh/update code paths diverged)",
                        flush=True,
                    )
            minibatch_metrics = [metrics]
            actor_dashboard = aggregate_actor_tensorboard_metrics(
                minibatch_metrics
            )
            nonfinite_gradients = {
                name: actor_dashboard[name]
                for name in (
                    "grad/trunk", "grad/gate", "grad/adapter", "grad/renderer",
                    "grad/sigma", "grad/thought_mean", "grad/critic",
                )
                if not math.isfinite(actor_dashboard[name])
            }
            if nonfinite_gradients:
                raise RuntimeError(
                    "non-finite gradients before optimizer step: "
                    f"{nonfinite_gradients}"
                )
            step_optimizers(optimizers, "actor")
            step_optimizers(optimizers, "critic")
            # Gradient buffers have already been reduced to scalar telemetry.
            # Release them before the optional second replay so this read-only
            # diagnostic cannot stack an inference forward on top of the
            # training step's peak allocation.
            zero_optimizers(optimizers, "actor")
            zero_optimizers(optimizers, "critic")
            with torch.no_grad():
                sigma_deltas = [
                    parameter.detach() - before
                    for parameter, before in zip(
                        sigma_parameters, sigma_before, strict=True
                    )
                ]
                mean_deltas = [
                    parameter.detach() - before
                    for parameter, before in zip(
                        mean_parameters, mean_before, strict=True
                    )
                ]
                head_updates = scalar_tensors_to_floats(
                    {
                        "sigma/head_update_rms": (
                            torch.stack(
                                [delta.square().sum() for delta in sigma_deltas]
                            ).sum()
                            / sum(delta.numel() for delta in sigma_deltas)
                        ).sqrt(),
                        "sigma/head_update_abs_max": torch.stack(
                            [delta.abs().max() for delta in sigma_deltas]
                        ).max(),
                        "sigma/mean_head_update_rms": (
                            torch.stack(
                                [delta.square().sum() for delta in mean_deltas]
                            ).sum()
                            / sum(delta.numel() for delta in mean_deltas)
                        ).sqrt(),
                        "sigma/mean_head_update_abs_max": torch.stack(
                            [delta.abs().max() for delta in mean_deltas]
                        ).max(),
                        "sigma/mean_head_parameter_abs_max": torch.stack(
                            [
                                wrapper.transition.mean_head.output_gain.detach().abs()
                                * wrapper.transition.mean_head.weight.detach().abs().max(),
                                wrapper.transition.mean_head.bias.detach().abs().max(),
                            ]
                        ).max(),
                        # Overwrite the pre-update minibatch snapshot with the
                        # adapter magnitude the next rollout will deploy.
                        "behavior/thought_adapter_weight_rms": (
                            wrapper.adapter.projection.weight.detach()
                            .square().mean().sqrt()
                        ),
                        "behavior/thought_adapter_bias_rms": (
                            wrapper.adapter.projection.bias.detach()
                            .square().mean().sqrt()
                        ),
                        "sigma/mean_output_gain": (
                            wrapper.transition.mean_head.output_gain.detach()
                        ),
                        "sigma/state_residual_gain": (
                            wrapper.transition.log_sigma_head.residual_gain.detach()
                        ),
                    }
                )
            if not all(math.isfinite(value) for value in head_updates.values()):
                raise RuntimeError(
                    f"non-finite policy head after optimizer step {next_step}: "
                    f"{head_updates}"
                )
            actor_dashboard.update(head_updates)
            actor_dashboard["perf/minibatch_h2d_seconds"] = (
                minibatch_h2d_seconds
            )
            actor_dashboard["perf/minibatch_cpu_pack_seconds"] = (
                minibatch_pack_seconds
            )
            if next_step == 1 or (
                args.post_update_kl_every > 0
                and next_step % args.post_update_kl_every == 0
            ):
                drift_started = time.perf_counter()
                with training_autocast():
                    post_update_drift = measure_post_update_policy_drift(
                        wrapper,
                        [device_minibatch],
                        replay_max_trajectories=args.replay_max_trajectories,
                        replay_attention_budget=args.replay_attention_budget,
                        replay_bucket=args.replay_bucket,
                        replay_slot_budget=args.replay_slot_budget,
                        replay_function=diagnostic_replay_head_inputs,
                    )
                if not all(
                    math.isfinite(value) for value in post_update_drift.values()
                ):
                    raise RuntimeError(
                        f"non-finite post-update policy drift at step {next_step}: "
                        f"{post_update_drift}"
                    )
                actor_dashboard.update(post_update_drift)
                actor_dashboard["perf/post_update_kl_seconds"] = (
                    time.perf_counter() - drift_started
                )
            # One PPO epoch uses every behavior group exactly once. Release
            # the compact host sources and packed device batch after its
            # update and optional drift.
            del device_minibatch
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
        zero_optimizers(optimizers, "actor")
        zero_optimizers(optimizers, "critic")
        del groups

        if crossed_interval(previous_step, step, args.bpb_every):
            eval_started = time.perf_counter()
            bpb = teacher_forced_bpb()
            eval_seconds = time.perf_counter() - eval_started
            logger.log(type="bpb", step=step, val_bpb=bpb, seconds=eval_seconds)
            tensorboard.add_scalar("guard/val_bpb", bpb, step)
            tensorboard.add_scalar("perf/bpb_eval_seconds", eval_seconds, step)
        if aime_rows and crossed_interval(previous_step, step, args.aime_every):
            aime_eval(step)
        if bench_rows and crossed_interval(previous_step, step, args.bench_every):
            bench_eval(step)
        if crossed_interval(previous_step, step, args.save_every):
            save_started = time.perf_counter()
            save_checkpoint(
                output / "latent_vapo_checkpoint.pt", wrapper, critic,
                optimizers, step, args, sampler, warmup_step,
                actor_init_provenance,
            )
            save_seconds = time.perf_counter() - save_started
            logger.log(type="checkpoint", step=step, seconds=save_seconds)
            tensorboard.add_scalar("perf/checkpoint_seconds", save_seconds, step)
    if args.consume_all_prompts and sampler.cursor != len(math_rows):
        raise RuntimeError(
            "--consume-all-prompts completed without exhausting the target "
            f"dataset: cursor {sampler.cursor}/{len(math_rows)}"
        )
    save_checkpoint(
        output / "latent_vapo_checkpoint.pt", wrapper, critic,
        optimizers, step, args, sampler, warmup_step,
        actor_init_provenance,
    )
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


if __name__ == "__main__":
    main()

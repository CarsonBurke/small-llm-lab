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
  A weak orthogonal belief-conditioned head learns diagonal per-dimension
  thought log-sigma around its configured initial level.
- No pretraining anchor: SIGReg and the latent target-prediction objective are
  dropped at RL time. The fresh mean and recurrent policy train purely on
  their ability to think; the teacher-forced val-BPB guard is the drift
  detector.

Prompts are DAPO-Math-17K. An EOS-terminated, verifier-correct final Answer:
field receives reward 1; a wrong but strictly numeric final field receives
bounded distance shaping of at most 0.1. Malformed or unterminated responses
receive zero. AIME 2024 avg@k is the eval. Prompt groups roll out and replay
separately because their lengths differ, then accumulate into the shared
actor/critic optimizer minibatch. ``--rollout-only`` reports whether rewards
vary within prompt groups at the initial 50/50 gate before any update is
attempted.

    python3 -m postraining.train_latent_vapo \
        --checkpoint ablation_results/<run>/pretraining_checkpoint.pt \
        --output postraining/runs/<name> [--rollout-only]
"""

from __future__ import annotations

import argparse
import atexit
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import fields
import hashlib
import json
import math
import os
import random
import subprocess
import threading
import time
import warnings
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
    compact_slots,
    compact_stream_to_device,
    pack_rollout_groups_for_replay,
    emitted_token_rows,
    half_forced_group_members,
    iter_length_aware_microbatches,
    refresh_old_statistics,
    replay_head_inputs,
    scatter_replay_statistics,
    scatter_slots,
    select_thought_actions,
    slot_index,
    think_slot_mask,
    rollout_continuations,
    split_rollout_groups,
    trim_stream,
)
from postraining.latent_thought import (
    EMIT,
    THINK,
    CRITIC_ADAPTER_INIT_KINDS,
    RENDERER_FEATURES_SCHEMA,
    SIGMA_STATE_INIT_KINDS,
    THOUGHT_ACTION_TRANSFORM_KINDS,
    THOUGHT_ADAPTER_KINDS,
    THOUGHT_DISTRIBUTION_SCHEMA,
    THOUGHT_MEAN_SCHEMA,
    LatentThoughtModel,
    migrate_legacy_wrapper_checkpoint,
    rollout_policy_schema_for_mode,
    transform_thought_action,
    validate_renderer_checkpoint,
)
from postraining.model_io import fresh_trunk, load_model
from postraining.muon import MUON_ALGORITHM_SCHEMA, Muon
from postraining.train_vapo import prompt_text
from postraining.value_model import SeparateCritic


EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_reverse_kl_thought_trust_anchored_value_orthogonal_silu_adapter_general_lr_sequential_data/v26"
)
IDENTITY_AFFINE_EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_reverse_kl_thought_trust_anchored_value_identity_affine_general_lr_sequential_data/v25"
)
TANH_ACTION_EXECUTION_SCHEMA_SUFFIX = "+tanh_raw_gaussian_recurrent_input/v1"
ZERO_AFFINE_EXECUTION_SCHEMA = (
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
PROMPT_ORDER_SCHEMA = "sequential_one_pass/v1"
ACTOR_OBJECTIVE_SCHEMA = "vapo_policy_no_positive_example_lm/v1"
ADAMW_ALGORITHM_SCHEMA = (
    "torch_adamw_betas0.9_0.999_eps1e-8_amsgrad_false_weight_decay0/v1"
)


def execution_schema_for_adapter(
    kind: str, thought_action_transform: str = "identity"
) -> str:
    """Execution schema for the selected deployed recurrent thought path."""
    if kind == "orthogonal_silu":
        schema = EXECUTION_SCHEMA
    elif kind == "identity_affine":
        schema = IDENTITY_AFFINE_EXECUTION_SCHEMA
    else:
        raise ValueError(f"unknown thought adapter kind {kind!r}")
    if thought_action_transform == "identity":
        return schema
    if thought_action_transform == "tanh":
        return schema + TANH_ACTION_EXECUTION_SCHEMA_SUFFIX
    raise ValueError(
        f"unknown thought action transform {thought_action_transform!r}"
    )


def optimizer_schema_for_trunk_optimizer(kind: str) -> str:
    """Exact update algorithm whose state a checkpoint may restore."""
    if kind == "adamw":
        return ADAMW_ALGORITHM_SCHEMA
    if kind == "muon":
        return f"{MUON_ALGORITHM_SCHEMA}+{ADAMW_ALGORITHM_SCHEMA}"
    raise ValueError(f"unknown trunk optimizer {kind!r}")


# Refreshed device minibatches held for their update instead of the
# scatter-to-CPU/repack/re-upload round trip. Bounded because packed batch
# bytes track the pool's longest stream (a think-heavy pool can quadruple
# them); past the budget the refresh loop falls back to the scatter path
# for the remaining minibatches. VRAM-resident only between a pool's
# refresh and its last update — never across a collection.
RETAINED_MINIBATCH_BUDGET_BYTES = 8 << 30
DEFAULT_BPB_GUARD_TOKENS = 2 * 1024 * 1024
DEFAULT_PERIODIC_EVAL_EVERY = 150
# Restores the trunk step size that the Muon:AdamW rate ratio was chosen for,
# after ``postraining.muon`` moved to Polar Express. It is a property of that
# orthogonalizer, not of the RL objective, and it multiplies only the DERIVED
# --muon-learning-rate default: an explicit rate on the command line is taken
# literally. See the derivation in validate_args.
#
# MEASURED, not guessed (job 442, 624 non-degenerate real gradients from live
# post-training, every Muon parameter over 8 steps of both optimizers). The
# move to Polar Express shrank the step by two independent mechanisms: Polar
# Express at 5 iterations leaves a wider singular-value ripple than the 12
# Newton-Schulz iterations it replaced, and the old ``max(1, rows/cols)**0.5``
# rectangular scale was dropped. ``new/old`` step norm by shape:
#
#     (512, 512)   n=414   0.823      <- 24 matrices, no rectangular term
#     (512, 2048)  n=108   0.570      <- 6 matrices, no rectangular term
#     (2048, 512)  n=102   0.494      <- 6 matrices, lost a 2.0x rect scale
#
# Geometric mean 0.690, so the compensation is 1/0.690. The first value used
# here was 2.4, which is what a synthetic-gaussian bracket suggested; against
# real gradients that overshoots badly, because real trunk gradients are very
# low rank (stable rank 1.0-2.6) and that is the regime where Polar Express
# and Newton-Schulz diverge most. At 2.4 the 414 (512, 512) matrices -- two
# thirds of the trunk -- would step 1.97x the tuned configuration. At 1.45 the
# spread is 1.19 / 0.83 / 0.72, centred on 1.
#
# A single scalar cannot correct all three shapes at once; that needs the
# per-parameter rectangular term back, which needs param groups, which is
# blocked on the momentum-aliasing bug in ``Muon._momenta`` (keyed by
# (shape, device) but bucketed per param group). Left as future work.
POLAR_EXPRESS_STEP_COMPENSATION = 1.45
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
    expected_execution_schema: str = EXECUTION_SCHEMA,
    allow_reverse_kl_migration: bool = False,
    allow_performance_migration: bool = False,
    allow_joint_clip_migration: bool = False,
    allow_anchored_value_migration: bool = False,
    allow_projected_thought_migration: bool = False,
    allow_thought_reverse_kl_migration: bool = False,
) -> bool:
    """Resume compatible policy state at a complete rollout-pool boundary."""
    execution_schema = payload.get("execution_schema")
    if execution_schema == expected_execution_schema:
        return not (
            allow_reverse_kl_migration
            or allow_performance_migration
            or allow_joint_clip_migration
            or allow_anchored_value_migration
            or allow_projected_thought_migration
            or allow_thought_reverse_kl_migration
        )
    # Every older policy used an affine thought interface. Its saved matrices
    # have the same shapes as v26 but acquire different semantics under
    # 2*SiLU, so no objective flag can make a nonlinear resume sound.
    if expected_execution_schema != IDENTITY_AFFINE_EXECUTION_SCHEMA:
        return False
    # v25 changes only the fresh adapter initialization. A v24 resume restores
    # its learned affine and optimizer state exactly.
    if execution_schema == ZERO_AFFINE_EXECUTION_SCHEMA:
        return not (
            allow_reverse_kl_migration
            or allow_performance_migration
            or allow_joint_clip_migration
            or allow_anchored_value_migration
            or allow_projected_thought_migration
            or allow_thought_reverse_kl_migration
        )
    if execution_schema == NO_THOUGHT_KL_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and not allow_projected_thought_migration
            and not allow_reverse_kl_migration
            and not allow_performance_migration
            and not allow_joint_clip_migration
            and not allow_anchored_value_migration
        )
    if execution_schema == JOINT_CLIP_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and allow_projected_thought_migration
            and not allow_reverse_kl_migration
            and not allow_performance_migration
            and not allow_joint_clip_migration
            and not allow_anchored_value_migration
        )
    if execution_schema == UNANCHORED_VALUE_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and allow_projected_thought_migration
            and allow_anchored_value_migration
            and not allow_reverse_kl_migration
            and not allow_performance_migration
            and not allow_joint_clip_migration
        )
    if execution_schema == PER_DIM_REVERSE_KL_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and allow_projected_thought_migration
            and allow_anchored_value_migration
            and allow_joint_clip_migration
            and not allow_reverse_kl_migration
            and not allow_performance_migration
        )
    if execution_schema == PERFORMANCE_COMPATIBLE_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and allow_projected_thought_migration
            and allow_anchored_value_migration
            and allow_joint_clip_migration
            and allow_performance_migration
            and not allow_reverse_kl_migration
        )
    return (
        allow_reverse_kl_migration
        and allow_performance_migration
        and allow_joint_clip_migration
        and allow_anchored_value_migration
        and allow_projected_thought_migration
        and allow_thought_reverse_kl_migration
        and execution_schema == PREVIOUS_EXECUTION_SCHEMA
    )


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
    thought_reverse_kl_contribution = sum(
        metric["thought_reverse_kl_penalty"] for metric in metrics
    )
    gate_entropy_bonus = sum(
        metric["gate_entropy_bonus"] for metric in metrics
    )
    return {
        "loss/policy": policy_contribution,
        "kl/thought_reverse_weighted": thought_reverse_kl_contribution,
        "bonus/gate_entropy_weighted": gate_entropy_bonus,
        "loss/actor_total": (
            policy_contribution
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
        "thought_input/raw_abs_max": max(
            metric["thought_raw_abs_max"] for metric in metrics
        ),
        "thought_input/raw_abs_gt_0_5_fraction": _weighted_metric_mean(
            metrics, "thought_raw_abs_gt_0_5_fraction", "thought_action_count"
        ),
        "thought_input/raw_abs_gt_0_8_fraction": _weighted_metric_mean(
            metrics, "thought_raw_abs_gt_0_8_fraction", "thought_action_count"
        ),
        "thought_input/raw_abs_gt_1_fraction": _weighted_metric_mean(
            metrics, "thought_raw_abs_gt_1_fraction", "thought_action_count"
        ),
        "thought_input/raw_abs_gt_2_fraction": _weighted_metric_mean(
            metrics, "thought_raw_abs_gt_2_fraction", "thought_action_count"
        ),
        "thought_input/transform_distortion_rms": math.sqrt(
            _weighted_metric_mean(
                [
                    {
                        **metric,
                        "thought_transform_distortion_square_mean": (
                            metric["thought_transform_distortion_rms"] ** 2
                        ),
                    }
                    for metric in metrics
                ],
                "thought_transform_distortion_square_mean",
                "thought_action_count",
            )
        ),
        "thought_input/squashed_raw_norm": _weighted_metric_mean(
            metrics, "thought_squashed_raw_norm", "thought_action_count"
        ),
        "thought_input/adapter_output_norm": _weighted_metric_mean(
            metrics, "thought_adapter_output_norm", "thought_action_count"
        ),
        "thought_input/post_thought_belief_norm": _weighted_metric_mean(
            metrics, "post_thought_belief_norm", "thought_action_count"
        ),
        "thought_input/mean_abs_max": max(
            metric["thought_mean_abs_max"] for metric in metrics
        ),
        "thought_input/mean_abs_gt_0_8_fraction": _weighted_metric_mean(
            metrics,
            "thought_mean_abs_gt_0_8_fraction",
            "thought_action_count",
        ),
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global action and gate denominators for one actor step."""
    if not indices:
        raise ValueError("at least one prompt group is required")
    action_count = torch.stack(
        [groups[index].action_mask.sum() for index in indices]
    ).sum()
    gate_action_count = torch.stack(
        [groups[index].gate_mask.sum() for index in indices]
    ).sum()
    return action_count, gate_action_count


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
    ``actor_muon``/``critic_muon`` Muon optimizers. Polar Express supplies
    the orthogonalized momentum update, while embeddings, readout, gains,
    and every RL-only head stay under AdamW. Weight decay stays 0 everywhere:
    pretraining's Muon decay (0.05) regularizes a from-scratch run, but over
    a long RL schedule it would only shrink the pretrained weights.
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
    # new_full, not new_tensor: see clipped_policy_loss -- the host-built
    # constant costs a blocking copy every call.
    log_lower = torch.log(
        dimension_log_ratio.new_full((), 1.0 - epsilon_low)
    )
    log_upper = torch.log(
        dimension_log_ratio.new_full((), 1.0 + epsilon_high)
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
            "renderer_kl_sum", "thought_kl_sum",
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
            "thought_raw_abs_max", "thought_raw_abs_gt_0_5_count",
            "thought_raw_abs_gt_0_8_count",
            "thought_raw_abs_gt_1_count", "thought_raw_abs_gt_2_count",
            "thought_transform_distortion_square_sum",
            "thought_squashed_raw_norm_sum",
            "thought_adapter_output_norm_sum",
            "post_thought_belief_norm_sum",
            "thought_mean_abs_max", "thought_mean_abs_gt_0_8_count",
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

        # One nonzero per mask per shard. Every compaction below then goes
        # through index_select, whose output shape is known on the host, so
        # the eager tail stops stalling once per indexing expression --
        # forward AND backward, since index_select's gradient is index_add
        # while boolean indexing's re-derives the index from the mask.
        emit_index = slot_index(microbatch.emit_mask.bool())
        emit_features = wrapper.renderer_features(
            compact_slots(stream_inputs, emit_index),
            compact_slots(beliefs, emit_index),
        )
        # Shared with refresh_old_statistics: identical chunk boundaries keep
        # the two eager forwards bit-identical (the age-0 zero-clip canary).
        compact_token_logprobs = compact_emit_token_logprobs(
            backbone, emit_features, compact_slots(token_targets, emit_index)
        )
        new_token_logprobs = torch.zeros_like(microbatch.old_token_logprobs)
        scatter_slots(new_token_logprobs, emit_index, compact_token_logprobs)

        new_thought_joint = torch.zeros_like(new_token_logprobs)
        old_thought_joint = torch.zeros_like(new_token_logprobs)
        # Bound only on the has-THINK path, and read again further down
        # under ``policy_thought_logprobs is not None``, which is set in the
        # same branch. Reset here with the rest so a previous shard's index
        # can never survive into this one.
        think_index = None
        policy_thought_logprobs = None
        compact_old_thought_logprobs = None
        thought_reverse_kl_factors = None
        weighted_thought_reverse_kl = zero
        projected_policy_thought_logprobs = None
        thought_trust_scale = None
        weighted_projection_penalty = zero
        shard_has_think = bool(row_has_think[host_rows].any())
        if shard_has_think:
            think_index = slot_index(think_slot_mask(microbatch))
            thought_means, thought_targets = select_thought_actions(
                microbatch, predicted, think_index
            )
            thought_log_sigma = wrapper.transition.predict_log_sigma(
                compact_slots(beliefs, think_index)
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
            compact_old_thought_logprobs = compact_slots(
                microbatch.old_thought_logprobs, think_index
            ).float()
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
            # index_copy_ rather than masked_scatter: both place the compact
            # values at the mask's true slots in order, but masked_scatter's
            # gradient is masked_select, whose output shape is data
            # dependent, so the backward pass stalls the host.
            scatter_slots(
                new_thought_joint, think_index, policy_thought_logprobs.sum(-1)
            )
            scatter_slots(
                old_thought_joint,
                think_index,
                compact_old_thought_logprobs.sum(-1),
            )
            if thought_clip_mode == "projected":
                behavior_thought_means = compact_slots(
                    microbatch.old_thought_means, think_index
                ).float()
                behavior_thought_log_sigmas = compact_slots(
                    microbatch.old_thought_log_sigmas, think_index
                ).float()
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
                        compact_slots(
                            microbatch.old_thought_means, think_index
                        ).float(),
                        compact_slots(
                            microbatch.old_thought_log_sigmas, think_index
                        ).float(),
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
            compact_gate_logprobs = compact_slots(
                policy_gate_logprobs, think_index
            )
            compact_old_gate_logprobs = compact_slots(
                microbatch.old_gate_logprobs, think_index
            )
            compact_thought_advantages = compact_slots(
                micro_advantages, think_index
            )
            compact_thought_gate_mask = compact_slots(
                microbatch.gate_mask, think_index
            )
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
                    compact_gate_logprobs,
                    compact_old_gate_logprobs,
                    projected_policy_thought_logprobs,
                    compact_old_thought_logprobs,
                    thought_trust_scale,
                    (compact_thought_advantages - advantage_scale_mean)
                    / thought_advantage_normalizer,
                    compact_thought_gate_mask,
                    policy_action_denominator,
                    gate_advantages=compact_thought_advantages,
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
                    compact_gate_logprobs,
                    compact_old_gate_logprobs,
                    policy_thought_logprobs,
                    compact_old_thought_logprobs,
                    compact_thought_advantages,
                    compact_thought_gate_mask,
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
                raw_thought = thought_targets.detach().float()
                transformed_thought = transform_thought_action(
                    raw_thought, wrapper.thought_action_transform
                )
                raw_abs = raw_thought.abs()
                totals["thought_raw_abs_max"] = torch.maximum(
                    totals["thought_raw_abs_max"], raw_abs.max()
                )
                totals["thought_raw_abs_gt_0_5_count"] += (
                    raw_abs > 0.5
                ).sum()
                totals["thought_raw_abs_gt_0_8_count"] += (
                    raw_abs > 0.8
                ).sum()
                totals["thought_raw_abs_gt_1_count"] += (raw_abs > 1.0).sum()
                totals["thought_raw_abs_gt_2_count"] += (raw_abs > 2.0).sum()
                totals["thought_transform_distortion_square_sum"] += (
                    transformed_thought - raw_thought
                ).square().sum()
                totals["thought_squashed_raw_norm_sum"] += (
                    transformed_thought.norm(dim=-1).sum()
                )
                thought_input_index = think_index + 1
                totals["thought_adapter_output_norm_sum"] += compact_slots(
                    stream_inputs.detach(), thought_input_index
                ).float().norm(dim=-1).sum()
                totals["post_thought_belief_norm_sum"] += compact_slots(
                    beliefs.detach(), thought_input_index
                ).float().norm(dim=-1).sum()
                mean_abs = thought_means.detach().float().abs()
                totals["thought_mean_abs_max"] = torch.maximum(
                    totals["thought_mean_abs_max"], mean_abs.max()
                )
                totals["thought_mean_abs_gt_0_8_count"] += (
                    mean_abs > 0.8
                ).sum()
                thought_log_ratio = (
                    new_thought_logprobs
                    - compact_slots(
                        microbatch.old_thought_logprobs, think_index
                    )
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
        thought_raw_abs_max=totals["thought_raw_abs_max"],
        thought_raw_abs_gt_0_5_fraction=(
            totals["thought_raw_abs_gt_0_5_count"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_raw_abs_gt_0_8_fraction=(
            totals["thought_raw_abs_gt_0_8_count"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_raw_abs_gt_1_fraction=(
            totals["thought_raw_abs_gt_1_count"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_raw_abs_gt_2_fraction=(
            totals["thought_raw_abs_gt_2_count"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_transform_distortion_rms=(
            totals["thought_transform_distortion_square_sum"]
            / (denominators["thought"] * thought_dim)
        ).sqrt(),
        thought_squashed_raw_norm=(
            totals["thought_squashed_raw_norm_sum"] / denominators["thought"]
        ),
        thought_adapter_output_norm=(
            totals["thought_adapter_output_norm_sum"]
            / denominators["thought"]
        ),
        post_thought_belief_norm=(
            totals["post_thought_belief_norm_sum"]
            / denominators["thought"]
        ),
        thought_mean_abs_max=totals["thought_mean_abs_max"],
        thought_mean_abs_gt_0_8_fraction=(
            totals["thought_mean_abs_gt_0_8_count"]
            / (denominators["thought"] * thought_dim)
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

            token_logprobs = torch.zeros_like(microbatch.old_token_logprobs).float()
            if bool(row_has_emit[host_rows].any()):
                emit_index = slot_index(microbatch.emit_mask.bool())
                emit_logits = wrapper.backbone.logits_from_features(
                    wrapper.renderer_features(
                        compact_slots(stream_inputs, emit_index),
                        compact_slots(beliefs, emit_index),
                    )
                )
                compact_token_logprobs = (
                    emit_logits.float()
                    .log_softmax(-1)
                    .gather(
                        -1, compact_slots(token_targets, emit_index)[..., None]
                    )
                    .squeeze(-1)
                )
                scatter_slots(token_logprobs, emit_index, compact_token_logprobs)
            token_log_ratio = (
                token_logprobs - microbatch.old_token_logprobs.float()
            )

            think_mask = think_slot_mask(microbatch)
            thought_joint = torch.zeros_like(token_logprobs)
            old_thought_joint = torch.zeros_like(token_logprobs)
            thought_log_ratio = None
            if bool(row_has_think[host_rows].any()):
                think_index = slot_index(think_mask)
                thought_means, thought_targets = select_thought_actions(
                    microbatch, predicted, think_index
                )
                thought_log_sigma = wrapper.transition.predict_log_sigma(
                    compact_slots(beliefs, think_index)
                )
                thought_logprobs = wrapper.transition.per_dim_log_prob(
                    thought_targets, thought_means, thought_log_sigma
                ).float()
                old_thought_logprobs = compact_slots(
                    microbatch.old_thought_logprobs, think_index
                ).float()
                thought_log_ratio = thought_logprobs - old_thought_logprobs
                scatter_slots(
                    thought_joint, think_index, thought_logprobs.sum(-1)
                )
                scatter_slots(
                    old_thought_joint,
                    think_index,
                    old_thought_logprobs.sum(-1),
                )

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
        "execution_schema": execution_schema_for_adapter(
            wrapper.thought_adapter_kind,
            wrapper.thought_action_transform,
        ),
        "actor_objective_schema": ACTOR_OBJECTIVE_SCHEMA,
        "prompt_order_schema": PROMPT_ORDER_SCHEMA,
        "math_data_identity": sampler.dataset_identity,
        "reward_schema": REWARD_SCHEMA,
        "reasoning_mode": reasoning_mode,
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "rollout_policy_schema": rollout_policy_schema_for_mode(reasoning_mode),
        "thought_input_schema": wrapper.thought_input_schema,
        "thought_action_transform_schema": (
            wrapper.thought_action_transform_schema
        ),
        "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
        "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
        "thought_sigma_init_schema": wrapper.sigma_state_init_schema,
        "critic_adapter_init_schema": critic.adapter_init_schema,
        "optimizer_schema": optimizer_schema_for_trunk_optimizer(
            getattr(args, "trunk_optimizer", "adamw")
        ),
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


PROFILE_SCHEMA = "latent_vapo_profile_v1"

# nvidia-smi query order. Power leads because a power collapse is the
# user-visible symptom; the SM clock separates an idle GPU from a throttled
# one at the same reported utilization.
DEVICE_SAMPLE_FIELDS = (
    "utilization_gpu_percent",
    "utilization_memory_percent",
    "power_draw_watts",
    "clocks_sm_mhz",
    "memory_used_mib",
)

# Host-side launch calls, as kineto names the CUPTI runtime and driver
# callbacks. Triton launches every compiled kernel through the DRIVER entry
# point cuLaunchKernelEx, so matching only the cudaLaunchKernel family would
# miss most of an Inductor-compiled decode loop's dispatches — which is the
# one number the launch-bound decode loop is diagnosed by.
LAUNCH_EVENT_PREFIXES = (
    "cudaLaunchKernel",
    "cudaLaunchCooperativeKernel",
    "cuLaunchKernel",
    "cuLaunchCooperativeKernel",
)


def compilation_record(metric) -> dict:
    """Turn one Dynamo CompilationMetrics row into a reportable record.

    Three different things append to that stream and only one of them is a
    frame compilation:

    * a forward compile, from ``_dynamo/convert_frame.py``, which is the
      only writer that fills ``co_name``, and hardcodes ``is_forward=True``;
    * a lazy backward compile, from ``_aot_autograd/runtime_wrappers.py``,
      which carries no code object and sets ``is_forward=False``;
    * a RUNTIME row, from ``_dynamo/utils.py``, billing Triton autotuning or
      a cudagraph re-record against the compile id that caused it. It also
      has no code object and sets ``is_forward = not is_backward``.

    So ``co_name is None`` does not mean "backward" — the pair
    ``(is_runtime, is_forward)`` is what separates them, and conflating the
    last two inflates compile time by whatever autotuning cost. Fields the
    writer left unset stay ``None`` here rather than becoming ``0.0``: an
    absent column and a measured zero are different claims.
    """

    def seconds(micros: int | None) -> float | None:
        return None if micros is None else micros / 1e6

    if metric.is_runtime:
        kind = "runtime"
        name = f"runtime of {metric.compile_id}"
    elif metric.is_forward is False:
        kind = "backward"
        name = f"backward of {metric.compile_id}"
    else:
        kind = "forward"
        name = metric.co_name or f"compile {metric.compile_id}"
    return {
        "compile_id": str(metric.compile_id),
        "kind": kind,
        "function": name,
        "file": metric.co_filename,
        "line": metric.co_firstlineno,
        "cache_size": str(metric.cache_size),
        "is_forward": metric.is_forward,
        "is_runtime": bool(metric.is_runtime),
        "recompile_reason": metric.recompile_reason,
        "seconds": (metric.duration_us or 0) / 1e6,
        "dynamo_seconds": seconds(metric.dynamo_cumulative_compile_time_us),
        "aot_seconds": seconds(metric.aot_autograd_cumulative_compile_time_us),
        "inductor_seconds": seconds(
            metric.inductor_cumulative_compile_time_us
        ),
        "backward_seconds": seconds(metric.backward_cumulative_compile_time_us),
        "runtime_autotune_seconds": seconds(
            metric.runtime_triton_autotune_time_us
        ),
        "runtime_cudagraph_seconds": seconds(metric.runtime_cudagraphify_time_us),
    }


class InertPhase:
    """The disabled profiler's phase object.

    One shared instance with empty ``__enter__``/``__exit__`` is what makes
    ``--profile`` free when it is off: a phase costs one attribute lookup
    and two empty calls, and nothing is timed, recorded, or synchronized.
    """

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exception) -> bool:
        return False


INERT_PHASE = InertPhase()


class DisabledProfiler:
    """Every profiler entry point, doing nothing.

    Call sites stay unconditional rather than wrapped in ``if profiling:``
    so the profiled and unprofiled control flow cannot drift apart.
    """

    enabled = False

    def phase(self, name: str) -> InertPhase:
        return INERT_PHASE

    def worker_span(self, name: str) -> InertPhase:
        return INERT_PHASE

    def register_artifact(self, name: str, function):
        return function

    def pool_started(self, step: int) -> None:
        pass

    def pool_finished(self, step: int, wall_seconds: float) -> dict:
        return {}

    def close(self) -> dict:
        return {}


class DeviceSampler:
    """Background ``nvidia-smi`` poller stamped on the profiler's clock.

    Samples carry a ``perf_counter`` stamp taken at READ time rather than
    nvidia-smi's own wall clock, so they share one monotonic timeline with
    the phase records and can be sliced per phase without clock conversion.

    Polling is not measuring. nvidia-smi returns whatever NVML last
    latched, and each field latches on its own schedule: on this device
    power refreshes about every 500 ms against the default 250 ms poll, so
    half the readings are copies of the one before. A phase shorter than
    one refresh can therefore contain nothing but a value formed before it
    started. ``window`` measures each field's refresh period from the data
    and reports a field only when enough readings were formed entirely
    inside the phase; otherwise it withholds that field and says so. A
    withheld number is better than a stale one attributed to the wrong
    phase.
    """

    # Below this many attributable readings a field's distribution is not a
    # distribution. Three is not principled, it is the smallest count for
    # which a minimum and a mean say different things.
    MINIMUM_READINGS = 3

    # A field changing on fewer than this share of polls is quiet, not
    # slow, and its median gap measures silence instead of a refresh rate.
    # The live fields here change on 20-50% of polls, and the fallback is
    # the poll interval, so the boundary is nowhere near either case.
    QUIET_FRACTION = 0.05

    # The most a field's refresh may be believed to lag the poller. This
    # device refreshes at 2x the default poll and the bound is 4x, so it
    # binds only where the median has stopped measuring a refresh rate.
    MAXIMUM_OVERSAMPLE = 4

    def __init__(self, interval_ms: int, device: torch.device):
        self.interval_ms = interval_ms
        self.device = device
        self.samples: list[tuple[float, tuple[float, ...]]] = []
        self.error: str | None = None
        self._process = None
        self._thread = None
        self._periods: list[float] | None = None
        self._periods_at = -1

    def _selector(self) -> list[str]:
        """Pin nvidia-smi to the training GPU, by UUID.

        Without this every visible GPU emits its own line each tick and idle
        devices are averaged into the phase summary. The UUID is used rather
        than an index because nvidia-smi numbers devices physically while
        CUDA_VISIBLE_DEVICES renumbers them.
        """
        if self.device.type != "cuda":
            return []
        uuid = getattr(
            torch.cuda.get_device_properties(self.device), "uuid", None
        )
        return ["-i", f"GPU-{uuid}"] if uuid is not None else []

    def start(self) -> None:
        if self.interval_ms <= 0:
            self.error = "sampling disabled"
            return
        try:
            self._process = subprocess.Popen(
                [
                    "nvidia-smi",
                    *self._selector(),
                    "--query-gpu=utilization.gpu,utilization.memory,"
                    "power.draw,clocks.sm,memory.used",
                    "--format=csv,noheader,nounits",
                    f"-lms={self.interval_ms}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as failure:
            self.error = f"nvidia-smi unavailable: {failure}"
            return
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        for line in self._process.stdout:
            columns = line.strip().split(", ")
            if len(columns) != len(DEVICE_SAMPLE_FIELDS):
                continue
            try:
                values = tuple(float(column) for column in columns)
            except ValueError:
                continue
            self.samples.append((time.perf_counter(), values))
        # nvidia-smi rejecting the selector looks exactly like a quiet GPU
        # otherwise: stderr goes nowhere and the report shows no samples.
        if not self.samples and self._process.poll():
            self.error = (
                f"nvidia-smi exited {self._process.returncode} without "
                "producing a sample"
            )

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def period_ms(self, field: str) -> float:
        """How often this field actually changes underneath the poller.

        Measured, not assumed, because the fields do not share a schedule.
        The estimate is the median gap between CHANGES. On this device that
        reads 500 ms for power, clocks and both utilizations against a
        250 ms poll, and the gap histogram is a single sharp mode there, so
        the median is measuring the driver's refresh rather than how often
        the quantity happens to move.

        A field that changes rarely falls back to the delivered sample
        spacing instead of taking a median of long quiet stretches. That is
        not a concession: aliasing is the risk of reporting a value formed
        outside the phase as the phase's, and a field that barely moves
        holds the same value inside and out.

        The estimate is clamped into ``[spacing, MAXIMUM_OVERSAMPLE *
        spacing]``. Below the spacing it would claim a resolution the
        poller never delivered; far above it, the median has stopped
        measuring a refresh and started measuring a quiet stretch, and
        there is no reading of the data that makes an unbounded answer
        safer than a bounded wrong one. ``spacing`` is the delivered gap,
        not the requested interval: ``-lms`` is a request, and a starved
        reader thread that delivers a line a second apart would otherwise
        have every reading credited with a quarter-second of coverage it
        does not have.

        Recomputed whenever the series has grown. The first caller is the
        end of pool 0, whose samples are the least representative in the
        run -- cold compilation holds the GPU at a flat idle, which reads
        as a quiet field -- and freezing that answer would set the run's
        resolution from its worst window.
        """
        if self._periods is None or self._periods_at != len(self.samples):
            self._periods_at = len(self.samples)
            self._periods = [
                self._measure_period(position)
                for position in range(len(DEVICE_SAMPLE_FIELDS))
            ]
        return self._periods[DEVICE_SAMPLE_FIELDS.index(field)]

    def spacing_ms(self) -> float:
        """The gap the poller actually delivered, in milliseconds."""
        gaps = sorted(
            later - earlier
            for (earlier, _), (later, _) in zip(self.samples, self.samples[1:])
        )
        if not gaps:
            return float(self.interval_ms)
        return max(float(self.interval_ms), gaps[len(gaps) // 2] * 1e3)

    def _measure_period(self, index: int) -> float:
        spacing = self.spacing_ms()
        changes = [
            stamp
            for position, (stamp, values) in enumerate(self.samples)
            if position > 0
            and values[index] != self.samples[position - 1][1][index]
        ]
        gaps = sorted(
            later - earlier for earlier, later in zip(changes, changes[1:])
        )
        if len(gaps) < 4 or len(changes) < self.QUIET_FRACTION * len(
            self.samples
        ):
            return spacing
        return min(
            self.MAXIMUM_OVERSAMPLE * spacing,
            max(spacing, gaps[len(gaps) // 2] * 1e3),
        )

    def _readings(
        self, field: str, windows: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """One field's readings that belong to this phase and no other.

        The driver refreshes the field every ``period``, so a reading
        stamped ``t`` was latched somewhere in ``[t - period, t]``. It is
        admitted only when that whole interval falls inside one occurrence,
        which is what keeps a value formed before the phase from being
        reported as the phase's.

        Admitted samples are then thinned to one per period, since polling
        faster than the refresh re-reads the same latch. A changed value is
        always kept: a change proves a new latch regardless of the clock.

        Returns ``(seconds_covered, value)`` pairs rather than bare values.
        A reading covers one refresh period, EXCEPT where a change let it
        through early, and only the caller that turns readings into seconds
        can tell the difference. Charging every reading a whole period once
        reported more seconds below the power floor than the phase lasted.
        """
        index = DEVICE_SAMPLE_FIELDS.index(field)
        period = self.period_ms(field) / 1e3
        readings: list[tuple[float, float]] = []
        previous: float | None = None
        admitted = float("-inf")
        for stamp, values in self.samples:
            value = values[index]
            changed = value != previous
            previous = value
            formed = any(
                started <= stamp - period and stamp <= ended
                for started, ended in windows
            )
            if formed and (changed or stamp - admitted >= period):
                readings.append((min(period, stamp - admitted), value))
                admitted = stamp
        return readings

    def window(
        self, windows: list[tuple[float, float]], power_floor: float
    ) -> dict:
        """Power and clock statistics pooled over one phase's occurrences.

        Deliberately not a mean of per-occurrence means. A power collapse
        lasting a few hundred milliseconds inside a ten-second decode phase
        moves the mean by a couple of watts and disappears; the minimum
        and the seconds spent under the floor are what make it a line
        item. There is no percentile column: at a 500 ms refresh even a
        ten-second phase yields about twenty readings, and a twentieth of
        twenty is the minimum under another name. Readings are pooled across occurrences so
        four decode chunks are one distribution, not four averages.

        Fields are reported independently. A short phase usually keeps its
        utilization numbers and loses its power numbers, because those two
        refresh at different rates, and reporting the pair as though they
        were equally well measured is what made a one-sample phase look
        like a measurement.
        """
        # Nothing sampled at all is not the same claim as sampled and
        # withheld, and a run on a device with no sampler should not report
        # every field as suppressed.
        if not self.samples:
            return {}
        drawn = self._readings("power_draw_watts", windows)
        clocked = self._readings("clocks_sm_mhz", windows)
        used = self._readings("utilization_gpu_percent", windows)
        power = sorted(watts for _, watts in drawn)
        clocks = sorted(mhz for _, mhz in clocked)
        utilization = [percent for _, percent in used]
        stats: dict = {}
        withheld = []
        for field, readings in (
            ("power_draw_watts", power),
            ("clocks_sm_mhz", clocks),
            ("utilization_gpu_percent", utilization),
        ):
            if len(readings) < self.MINIMUM_READINGS:
                withheld.append(field)
            else:
                stats[f"{field}_period_ms"] = self.period_ms(field)
                stats[f"{field}_readings"] = len(readings)
        if "power_draw_watts_readings" in stats:
            stats.update(
                {
                    "power_draw_watts_mean": sum(power) / len(power),
                    "power_draw_watts_min": power[0],
                    "power_draw_watts_max": power[-1],
                    # Seconds, not a count: comparable across phases of
                    # different lengths and answerable against the phase's
                    # own wall, which it can no longer exceed because each
                    # reading is charged only the span it covers.
                    "seconds_below_power_floor": sum(
                        covered
                        for covered, watts in drawn
                        if watts < power_floor
                    ),
                }
            )
        if "clocks_sm_mhz_readings" in stats:
            stats["clocks_sm_mhz_mean"] = sum(clocks) / len(clocks)
            stats["clocks_sm_mhz_min"] = clocks[0]
        if "utilization_gpu_percent_readings" in stats:
            stats["utilization_gpu_percent_mean"] = sum(utilization) / len(
                utilization
            )
        if withheld:
            stats["withheld"] = withheld
        return stats


class SyncDetector:
    """Blocking host syncs, believed only after positive controls pass.

    ``set_sync_debug_mode("warn")`` reports each blocking synchronization as
    a Python warning located at the calling line. The detector first trips
    three syncs it knows must fire and refuses to report anything unless it
    caught all three: a broken detector and a sync-free window look
    identical otherwise, which is how an earlier investigation came to
    report a host sync from a process that was running on CPU.

    One known blind spot, reported alongside the counts rather than left for
    someone to trip over: torch installs its warning handler per thread, so
    a sync raised on an autograd backward worker never reaches this hook.
    Absence of a site here is not proof that the thread is sync-free.
    """

    # c10 emits "called a synchronizing CUDA operation" through PyErr_WarnEx
    # with stacklevel 1, so the warning lands on the caller's own line.
    SYNC_MESSAGE = "called a synchronizing CUDA operation"

    def __init__(self, device: torch.device):
        self.device = device
        self.sites: dict[str, int] = {}
        self.controls_passed = False
        self.control_detail: dict[str, bool] = {}
        self._active = False
        self._previous_showwarning = None
        self._previous_filters = None

    def _install(self) -> None:
        self._previous_showwarning = warnings.showwarning
        self._previous_filters = warnings.filters[:]
        fallback = self._previous_showwarning

        def record(message, category, filename, lineno, file=None, line=None):
            if self.SYNC_MESSAGE in str(message):
                key = f"{filename}:{lineno}"
                self.sites[key] = self.sites.get(key, 0) + 1
            else:
                fallback(message, category, filename, lineno, file, line)

        warnings.showwarning = record
        # The sync warning repeats from one location every call, and the
        # default once-per-location filter would collapse a hot loop's
        # thousands of syncs into a single report. Restored on the way out:
        # the filter list is process-global state this must not leak.
        warnings.simplefilter("always")

    def _restore(self) -> None:
        if self._previous_showwarning is not None:
            warnings.showwarning = self._previous_showwarning
            self._previous_showwarning = None
        if self._previous_filters is not None:
            warnings.filters[:] = self._previous_filters
            warnings._filters_mutated()
            self._previous_filters = None

    def run_controls(self) -> None:
        if self.device.type != "cuda":
            self.control_detail = {"cuda_device": False}
            return
        probe = torch.ones(4, device=self.device)
        self._install()
        torch.cuda.set_sync_debug_mode("warn")
        try:
            for name, call in (
                ("item", lambda: probe.sum().item()),
                ("cpu", lambda: probe.cpu()),
                ("bool_any", lambda: bool(probe.any())),
            ):
                before = sum(self.sites.values())
                call()
                self.control_detail[name] = sum(self.sites.values()) > before
        finally:
            torch.cuda.set_sync_debug_mode("default")
            self._restore()
        self.sites.clear()
        self.controls_passed = all(self.control_detail.values())

    def enable(self) -> None:
        if not self.controls_passed or self._active:
            return
        self._install()
        torch.cuda.set_sync_debug_mode("warn")
        self._active = True

    def disable(self) -> None:
        if not self._active:
            return
        torch.cuda.set_sync_debug_mode("default")
        self._restore()
        self._active = False

    def report(self) -> dict:
        if not self.controls_passed:
            return {
                "trusted": False,
                "reason": "positive controls did not all trip, so an empty "
                "result would be indistinguishable from a broken detector",
                "controls": self.control_detail,
                "sites": [],
                "total": 0,
            }
        ranked = sorted(self.sites.items(), key=lambda entry: -entry[1])
        return {
            "trusted": True,
            "controls": self.control_detail,
            "blind_spot": "torch's warning handler is per thread, so syncs "
            "on autograd backward workers are not counted here",
            "sites": [
                {"location": location, "count": count}
                for location, count in ranked
            ],
            "total": sum(self.sites.values()),
        }


class RunProfiler:
    """Per-phase accounting for one profiling run.

    Three properties this exists to provide, none of which the hand-rolled
    ``pool_*_seconds`` counters it supplements had:

    - Phases nest, and each one reports SELF time (its wall minus its
      children's). Root self times plus one explicit ``unaccounted`` row
      equal the pool wall time by construction, so work that no phase covers
      shows up as a remainder instead of hiding inside a plausible number.
    - Phase timing takes no barrier. Host time comes from ``perf_counter``
      and device time from a CUDA event pair resolved ONCE per pool, so
      measuring a phase does not drain the pipeline into it.
    - A blocking host sync is a line item attributed to a source location
      rather than silently folded into whatever phase it lands in.
    """

    enabled = True

    def __init__(
        self,
        output: Path,
        device: torch.device,
        args: argparse.Namespace,
    ):
        self.directory = output / "profile"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.args = args
        self._path: list[str] = []
        self._records: list[dict] = []
        self._events: list[tuple[tuple[str, ...], object, object]] = []
        self._worker_spans: list[tuple[str, float, float]] = []
        self._call_counts: dict[str, int] = {}
        self._calls_before_pool: dict[str, int] = {}
        self._compile_seconds: dict[str, float] = {}
        self._runtime_seconds: dict[str, float] = {}
        self._seen_compilations: set[tuple] = set()
        self._before_pool: list[dict] = []
        self._memory_before: dict = {}
        self._kernels: dict | None = None
        self._torch_profile = None
        self._closed = False
        self.pools: list[dict] = []
        self.pool_index = 0

        # The compilation metrics deque is bounded (64 by default) and evicts
        # silently, so without raising it a profiled run would drop the
        # earliest compilations, which are the ones worth seeing.
        torch._dynamo.utils.set_compilation_metrics_limit(
            max(args.profile_compile_records, 64)
        )
        # Anything already compiled before this object existed belongs to
        # whoever compiled it, not to this run's accounting.
        self._new_compilations()
        self.sampler = DeviceSampler(args.profile_device_interval_ms, device)
        self.sampler.start()
        self.sync_detector = SyncDetector(device)
        self.sync_detector.run_controls()
        # The sampler owns a child process and sync debug mode is global
        # state; a run that ends by raising must still release both, and a
        # partial profile is worth printing.
        atexit.register(self.close)

    @contextmanager
    def phase(self, name: str):
        self._path.append(name)
        path = tuple(self._path)
        start_event = None
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        started = time.perf_counter()
        try:
            yield
        finally:
            ended = time.perf_counter()
            if start_event is not None:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                self._events.append((path, start_event, end_event))
            self._records.append(
                {"path": path, "started": started, "ended": ended}
            )
            self._path.pop()

    @contextmanager
    def worker_span(self, name: str):
        """Time a region running on a background worker thread.

        Kept out of the phase tree: a worker span overlaps whatever
        main-thread phase is open, so summing it in would break the
        reconciliation. It is reported per phase as overlap instead.
        ``list.append`` is the only shared mutation, which the GIL makes
        atomic, so this needs no lock.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self._worker_spans.append(
                (name, started, time.perf_counter())
            )

    def register_artifact(self, name: str, function):
        """Count calls into a compiled artifact so its compile time can be
        priced against its use. An artifact that is compiled and never
        called is pure cost, and this run has had one."""
        if function is None:
            return None
        self._call_counts[name] = 0

        def counted(*call_args, **call_kwargs):
            self._call_counts[name] += 1
            return function(*call_args, **call_kwargs)

        return counted

    def pool_started(self, step: int) -> None:
        self._records.clear()
        self._events.clear()
        self._worker_spans.clear()
        self._kernels = None
        # Allocator counters, read as a per-pool delta. An allocation retry
        # flushes the cache and synchronizes every stream, which presents
        # exactly as a power collapse and costs nothing to rule in or out.
        self._memory_before = (
            torch.cuda.memory_stats() if self.device.type == "cuda" else {}
        )
        # Drained here as well as at pool end so cold compilation — step-0
        # eval, the value-warmup loop, everything before the first pool — is
        # reported in its own bucket instead of being billed to pool 0, whose
        # own wall time it can exceed.
        self._before_pool = self._new_compilations()
        # Snapshot rather than zero: close() reports run totals from the same
        # counters, and a pool's figure is the delta against this.
        self._calls_before_pool = dict(self._call_counts)
        if self._path:
            raise RuntimeError(
                f"profile phase stack left open across pools: {self._path}"
            )
        # Each instrument gets its own pool. Sync detection raises a Python
        # warning per blocking call, which would both distort a kineto trace
        # and be distorted by one, so it runs on the pools AFTER the traced
        # window rather than sharing them.
        traced_from = self.args.profile_skip_pools
        sync_from = traced_from + self.args.profile_pools
        if sync_from <= self.pool_index < sync_from + self.args.profile_sync_pools:
            self.sync_detector.enable()
        if traced_from <= self.pool_index < sync_from:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            # A fresh profile per pool rather than one scheduled window:
            # kernel launch counts are only meaningful attributed to a single
            # pool, and torch's schedule() spends a whole pool on warmup
            # between active windows.
            self._torch_profile = torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_stack=self.args.profile_stack,
            )
            self._torch_profile.start()

    def _device_seconds(self) -> dict[tuple[str, ...], float]:
        if not self._events:
            return {}
        # The one barrier this profiler takes, once per pool and only under
        # the flag: elapsed_time is undefined until both events complete.
        torch.cuda.synchronize()
        totals: dict[tuple[str, ...], float] = {}
        for path, start_event, end_event in self._events:
            totals[path] = totals.get(path, 0.0) + (
                start_event.elapsed_time(end_event) / 1e3
            )
        return totals

    def _summarize_kernels(self) -> dict:
        """Top kernels by device time, and the launch count two ways.

        Device-side kernel invocations and host-side ``cudaLaunchKernel``
        calls answer different questions and may disagree; a launch-bound
        loop is diagnosed by the host-side number.
        """
        profile, self._torch_profile = self._torch_profile, None
        if profile is None:
            return {}
        profile.stop()
        # Measured on this trainer: one pool is 1.03M device launches, and
        # its Chrome trace is 2.3 GB — past what Perfetto will open, and the
        # launch counts and kernel ranking come from key_averages anyway.
        # So the export is opt-in and the counts are not.
        trace = None
        exported = None
        if self.args.profile_trace:
            trace = self.directory / f"trace_pool_{self.pool_index}.json"
            try:
                profile.export_chrome_trace(str(trace))
                exported = trace.stat().st_size
            except (OSError, MemoryError, RuntimeError) as failure:
                return {"trace": str(trace), "error": f"export: {failure}"}
        try:
            averages = profile.key_averages()
        except (AssertionError, MemoryError, RuntimeError) as failure:
            return {
                "trace": str(trace) if trace else None,
                "trace_bytes": exported,
                "error": f"key_averages: {failure}",
            }
        kernels = []
        device_launches = 0
        host_launches = 0
        for entry in averages:
            if getattr(entry, "device_type", None) == (
                torch.autograd.DeviceType.CUDA
            ):
                device_launches += entry.count
                kernels.append(
                    {
                        "name": entry.key,
                        "count": entry.count,
                        "device_seconds": (
                            getattr(entry, "self_device_time_total", 0) or 0
                        )
                        / 1e6,
                    }
                )
            elif entry.key.startswith(LAUNCH_EVENT_PREFIXES):
                host_launches += entry.count
        kernels.sort(key=lambda kernel: -kernel["device_seconds"])
        return {
            "trace": str(trace) if trace else None,
            "trace_bytes": exported,
            "device_kernel_launches": device_launches,
            "host_cuda_launch_calls": host_launches,
            "total_device_seconds": sum(
                kernel["device_seconds"] for kernel in kernels
            ),
            "top_kernels": kernels[: self.args.profile_top_kernels],
        }

    def _new_compilations(self) -> list[dict]:
        fresh = []
        for metric in torch._dynamo.utils.get_compilation_metrics():
            key = (
                str(metric.compile_id),
                metric.co_name,
                metric.is_forward,
                metric.is_runtime,
                metric.start_time_us,
            )
            if key in self._seen_compilations:
                continue
            self._seen_compilations.add(key)
            fresh.append(compilation_record(metric))
        return fresh

    def _bill_compilations(self, entries: list[dict]) -> None:
        """Add records to the run totals, keeping the two kinds apart.

        Compile time is charged to the frame, which is what a fix targets.
        Runtime autotuning and cudagraph re-records have no frame, so they
        are charged to the compile id that provoked them.
        """
        for entry in entries:
            if entry["kind"] == "runtime":
                self._runtime_seconds[entry["compile_id"]] = (
                    self._runtime_seconds.get(entry["compile_id"], 0.0)
                    + entry["seconds"]
                )
            else:
                self._compile_seconds[entry["function"]] = (
                    self._compile_seconds.get(entry["function"], 0.0)
                    + entry["seconds"]
                )

    def _allocator_delta(self) -> dict:
        """Allocator events this pool caused.

        ``num_alloc_retries`` is the one to watch: a retry empties the
        caching allocator and synchronizes every stream, which is a
        device-wide stall that looks like nothing else in the phase table.
        """
        if self.device.type != "cuda":
            return {}
        after = torch.cuda.memory_stats()
        return {
            counter: after.get(counter, 0)
            - self._memory_before.get(counter, 0)
            for counter in (
                "num_alloc_retries",
                "num_ooms",
                "num_device_alloc",
                "num_device_free",
                "num_sync_all_streams",
            )
        }

    def pool_finished(self, step: int, wall_seconds: float) -> dict:
        if self._path:
            raise RuntimeError(
                f"profile phase stack still open at pool end: {self._path}"
            )
        self.sync_detector.disable()
        kernels = self._summarize_kernels()
        device_seconds = self._device_seconds()
        walls: dict[tuple[str, ...], float] = {}
        counts: dict[tuple[str, ...], int] = {}
        windows: dict[tuple[str, ...], list[tuple[float, float]]] = {}
        for record in self._records:
            path = record["path"]
            walls[path] = walls.get(path, 0.0) + (
                record["ended"] - record["started"]
            )
            counts[path] = counts.get(path, 0) + 1
            windows.setdefault(path, []).append(
                (record["started"], record["ended"])
            )
        # Worker spans overlap main-thread phases by construction, so they
        # are reported alongside a phase rather than summed into the tree.
        # The quantity that matters is worker CPU running INSIDE decode: the
        # decode loop is launch-bound, and a worker holding the GIL there
        # stops the host from launching and drains the GPU.
        worker_seconds: dict[tuple[str, ...], float] = {}
        for path, spans in windows.items():
            worker_seconds[path] = sum(
                max(0.0, min(ended, worker_ended) - max(started, worker_started))
                for _, worker_started, worker_ended in self._worker_spans
                for started, ended in spans
            )
        # Depth-first, siblings by descending wall time. The report indents by
        # depth, so any order that separates a phase from its parent prints a
        # tree that lies about who owns what.
        def tree_order(path: tuple[str, ...]) -> tuple:
            return tuple(
                (-walls[path[: depth + 1]], path[depth])
                for depth in range(len(path))
            )

        phases = []
        for path in sorted(walls, key=tree_order):
            children = sum(
                wall
                for other, wall in walls.items()
                if len(other) == len(path) + 1 and other[: len(path)] == path
            )
            phases.append(
                {
                    "path": list(path),
                    "name": path[-1],
                    "depth": len(path) - 1,
                    "count": counts[path],
                    "wall_seconds": walls[path],
                    "self_seconds": walls[path] - children,
                    "device_seconds": device_seconds.get(path),
                    "worker_seconds": worker_seconds.get(path, 0.0),
                    # On the same perf_counter clock as device_samples.json,
                    # so the raw power series can be sliced by phase offline.
                    "windows": windows[path],
                    "device": self.sampler.window(
                        windows[path], self.args.profile_power_floor
                    ),
                }
            )
        roots = sum(wall for path, wall in walls.items() if len(path) == 1)
        unaccounted = wall_seconds - roots
        if unaccounted < -1e-6:
            raise RuntimeError(
                "profile phase tree is broken: root phases total "
                f"{roots:.3f} s inside a {wall_seconds:.3f} s pool"
            )
        before_pool, self._before_pool = self._before_pool, []
        compilations = self._new_compilations()
        self._bill_compilations(before_pool + compilations)
        pool = {
            "schema": PROFILE_SCHEMA,
            "pool_index": self.pool_index,
            "step": step,
            "wall_seconds": wall_seconds,
            "phases": phases,
            "unaccounted_seconds": unaccounted,
            "unaccounted_fraction": (
                unaccounted / wall_seconds if wall_seconds else 0.0
            ),
            "reconciled": unaccounted
            <= self.args.profile_reconcile_tolerance * max(wall_seconds, 1e-9),
            # Split so "did this pool compile anything?" has an answer that
            # cold startup cannot contaminate. Steady state means both lists
            # hold no record of kind "forward" or "backward" from some pool
            # onwards. Runtime records are NOT part of that test: a
            # reduce-overhead artifact re-records its cudagraph whenever the
            # pool it captured against is invalidated, which can happen at
            # any point in a perfectly converged run.
            "compilations_before_pool": before_pool,
            "compilations": compilations,
            "compile_seconds": sum(
                entry["seconds"]
                for entry in before_pool + compilations
                if entry["kind"] != "runtime"
            ),
            "runtime_seconds": sum(
                entry["seconds"]
                for entry in before_pool + compilations
                if entry["kind"] == "runtime"
            ),
            "artifact_calls": {
                name: count - self._calls_before_pool.get(name, 0)
                for name, count in self._call_counts.items()
            },
            "kernels": kernels,
            "allocator": self._allocator_delta(),
            "worker_seconds_total": sum(
                ended - started for _, started, ended in self._worker_spans
            ),
        }
        self.pools.append(pool)
        (self.directory / f"pool_{self.pool_index}.json").write_text(
            json.dumps(pool, indent=2)
        )
        if not pool["reconciled"]:
            print(
                f"WARNING profile pool {self.pool_index}: "
                f"{unaccounted:.3f} s of {wall_seconds:.3f} s "
                f"({100.0 * pool['unaccounted_fraction']:.1f}%) is covered by "
                "no phase; the accounting is not closed",
                flush=True,
            )
        self.pool_index += 1
        return pool

    def close(self) -> dict:
        if self._closed:
            return {}
        self._closed = True
        if self._torch_profile is not None:
            self._torch_profile.stop()
            self._torch_profile = None
        self.sampler.stop()
        self.sync_detector.disable()
        # A run whose modes exit before the pool loop (--bench-only,
        # --bpb-only, --rollout-only) still compiles, and so does anything
        # after the last pool. Without this drain those compilations are
        # simply never reported.
        outside_pools = self._before_pool + self._new_compilations()
        self._bill_compilations(outside_pools)
        summary = {
            "schema": PROFILE_SCHEMA,
            "pools": self.pools,
            "compilations_outside_pools": outside_pools,
            "syncs": self.sync_detector.report(),
            "artifact_calls": dict(self._call_counts),
            "compile_seconds_by_function": dict(self._compile_seconds),
            "runtime_seconds_by_compile_id": dict(self._runtime_seconds),
            "dead_artifacts": [
                name for name, calls in self._call_counts.items() if not calls
            ],
            "device_sampler_error": self.sampler.error,
            "device_samples": len(self.sampler.samples),
            # The poll interval is a request; these are what the driver
            # actually delivered, and they are the reason a short phase
            # reports no power.
            "device_sample_period_ms": {
                field: self.sampler.period_ms(field)
                for field in DEVICE_SAMPLE_FIELDS
            },
        }
        (self.directory / "profile_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        # The raw series, not just its per-phase digest. A periodic collapse
        # is defined by its period, and no summary statistic carries that;
        # at a quarter-second interval a whole run is a few thousand rows.
        (self.directory / "device_samples.json").write_text(
            json.dumps(
                {
                    "fields": ["perf_counter", *DEVICE_SAMPLE_FIELDS],
                    "samples": [
                        [stamp, *values] for stamp, values in self.sampler.samples
                    ],
                }
            )
        )
        print(format_profile_summary(summary), flush=True)
        return summary


@contextmanager
def device_timed(sink: list):
    """Time a device region with a CUDA event pair instead of a barrier.

    A host wall clock around an asynchronous region measures launch cost,
    and making it measure device time needs a ``synchronize`` that drains
    the pipeline into the very region being timed. The pair is recorded on
    the stream and read later, after a barrier the caller already takes.
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    try:
        yield
    finally:
        end_event.record()
        sink.append((start_event, end_event))


def elapsed_seconds(events: list) -> float:
    """Total device seconds over recorded pairs.

    Waits on each end event rather than requiring the caller to have taken a
    barrier: elapsed_time raises on an event that has not completed, and
    where a barrier has already happened this returns immediately.
    """
    total = 0.0
    for start_event, end_event in events:
        end_event.synchronize()
        total += start_event.elapsed_time(end_event)
    return total / 1e3


def format_compile_split(entry: dict) -> str:
    """Render only the sub-timings the writer actually filled in.

    A runtime record has no dynamo or inductor column and a forward has no
    autotune column; printing the absent ones as ``0.00`` reads as a
    measurement rather than as silence.
    """
    parts = [
        (label, entry[field])
        for label, field in (
            ("dynamo", "dynamo_seconds"),
            ("aot", "aot_seconds"),
            ("inductor", "inductor_seconds"),
            ("backward", "backward_seconds"),
            ("autotune", "runtime_autotune_seconds"),
            ("cudagraph", "runtime_cudagraph_seconds"),
        )
        if entry.get(field) is not None
    ]
    return (
        " / ".join(f"{label} {value:.2f}" for label, value in parts)
        if parts
        else "no sub-timings reported"
    )


def format_profile_summary(summary: dict) -> str:
    """End-of-run table. Plain text so it survives the job log."""
    lines = ["", "=" * 96, "PROFILE SUMMARY", "=" * 96]
    for pool in summary["pools"]:
        records = pool["compilations_before_pool"] + pool["compilations"]
        compiles = [entry for entry in records if entry["kind"] != "runtime"]
        lines.append(
            f"\npool {pool['pool_index']} (actor step {pool['step']}): "
            f"{pool['wall_seconds']:.3f} s wall, "
            f"{pool['compile_seconds']:.3f} s compiling in "
            f"{len(compiles)} compilations, "
            f"{pool['runtime_seconds']:.3f} s in "
            f"{len(records) - len(compiles)} runtime records"
        )
        lines.append(
            f"  {'phase':<30}{'n':>5}{'wall s':>9}{'self s':>9}{'% pool':>8}"
            f"{'gpu s':>9}{'wrkr s':>8}{'W mean':>8}{'W min':>7}"
            f"{'s<flr':>7}{'MHz':>6}"
        )
        for phase in pool["phases"]:
            label = "  " * phase["depth"] + phase["name"]
            share = 100.0 * phase["self_seconds"] / max(pool["wall_seconds"], 1e-9)
            device = phase["device_seconds"]
            sampled = phase["device"]

            def column(name: str, digits: int = 0, width: int = 7) -> str:
                value = sampled.get(name)
                text = "-" if value is None else f"{value:.{digits}f}"
                return f"{text:>{width}}"

            lines.append(
                f"  {label:<30}{phase['count']:>5}"
                f"{phase['wall_seconds']:>9.3f}{phase['self_seconds']:>9.3f}"
                f"{share:>8.1f}"
                f"{(f'{device:.3f}' if device is not None else '-'):>9}"
                f"{phase['worker_seconds']:>8.2f}"
                + column("power_draw_watts_mean", width=8)
                + column("power_draw_watts_min")
                + column("seconds_below_power_floor", digits=1)
                + column("clocks_sm_mhz_min", width=6)
            )
        flag = "" if pool["reconciled"] else "  <-- NOT CLOSED"
        lines.append(
            f"  {'unaccounted':<30}{'':>5}{'':>9}"
            f"{pool['unaccounted_seconds']:>9.3f}"
            f"{100.0 * pool['unaccounted_fraction']:>8.1f}{flag}"
        )
        # A phase total hides a stall. Four refreshes averaging 4.5 s and
        # three refreshes at 0.7 s plus one at 16 s are the same row above,
        # and only the second is worth chasing. Printed with the start
        # stamp because device_samples.json shares this clock, so an
        # outlier can be looked up against power and clocks directly.
        for phase in pool["phases"]:
            spans = [ended - started for started, ended in phase["windows"]]
            if len(spans) < 2:
                continue
            longest = max(spans)
            # Twice the mean of the OTHER occurrences. Comparing against the
            # mean of all of them buries the outlier in its own average: at
            # two occurrences no span can reach twice a mean it is half of,
            # so the one case this is for -- 0.7 s and then 16 s -- would
            # never print.
            if longest < 1.0 or longest * (len(spans) - 1) < 2.0 * (
                sum(spans) - longest
            ):
                continue
            started = max(phase["windows"], key=lambda pair: pair[1] - pair[0])[0]
            lines.append(
                f"    uneven: {'.'.join(phase['path'])} longest occurrence "
                f"{longest:.3f} s of {phase['wall_seconds']:.3f} s over "
                f"{len(spans)}, started {started:.3f}"
            )
        allocator = pool["allocator"]
        if any(allocator.values()):
            lines.append(
                "    allocator this pool: "
                + ", ".join(
                    f"{counter.removeprefix('num_')}={value}"
                    for counter, value in allocator.items()
                    if value
                )
            )
        for label, entries in (
            ("before pool", pool["compilations_before_pool"]),
            ("in pool", pool["compilations"]),
        ):
            lines.extend(
                f"    {entry['kind']} ({label}) {entry['function']}:"
                f"{entry['line']} {entry['seconds']:.2f} s "
                f"({format_compile_split(entry)}) cache_size="
                f"{entry['cache_size']} reason="
                f"{entry['recompile_reason'] or 'first compile'}"
                for entry in entries
            )
        kernels = pool["kernels"]
        if kernels and "error" not in kernels:
            lines.append(
                f"    kernel launches: {kernels['device_kernel_launches']} "
                f"device-side, {kernels['host_cuda_launch_calls']} host "
                f"launch calls; {kernels['total_device_seconds']:.3f} s "
                "total device time"
                + (
                    f"; trace {kernels['trace_bytes'] / 2**20:.0f} MiB"
                    if kernels.get("trace_bytes")
                    else ""
                )
            )
            for kernel in kernels["top_kernels"]:
                lines.append(
                    f"      {kernel['device_seconds']:8.3f} s "
                    f"x{kernel['count']:<9} {kernel['name'][:56]}"
                )
        elif kernels:
            lines.append(f"    kernel summary failed: {kernels['error']}")
    outside = summary["compilations_outside_pools"]
    if outside:
        lines.append(
            f"\n{len(outside)} compilations outside any pool "
            "(startup, evaluation, or after the last pool):"
        )
        for entry in outside:
            lines.append(
                f"  {entry['kind']} {entry['function']}:{entry['line']} "
                f"{entry['seconds']:.2f} s reason="
                f"{entry['recompile_reason'] or 'first compile'}"
            )
    calls = summary["artifact_calls"]
    if calls:
        lines.append("\ncompiled artifact calls over the run:")
        for name, count in sorted(calls.items()):
            note = "  <-- NEVER CALLED; its compile time is waste" if not count else ""
            lines.append(f"  {name:<40}{count:>12}{note}")
    compiles = summary["compile_seconds_by_function"]
    if compiles:
        lines.append("\ncompile seconds by frame:")
        for name, seconds in sorted(compiles.items(), key=lambda item: -item[1]):
            lines.append(f"  {name:<40}{seconds:>12.2f}")
    runtimes = summary["runtime_seconds_by_compile_id"]
    if runtimes:
        lines.append(
            "\nruntime seconds (Triton autotuning and cudagraph re-records,"
            " charged to the compile id that caused them):"
        )
        for name, seconds in sorted(runtimes.items(), key=lambda item: -item[1]):
            lines.append(f"  {name:<40}{seconds:>12.2f}")
    syncs = summary["syncs"]
    lines.append("\nblocking host syncs:")
    if not syncs["trusted"]:
        lines.append(f"  NOT REPORTED: {syncs['reason']}")
        lines.append(f"  positive controls: {syncs['controls']}")
    else:
        lines.append(f"  positive controls all tripped: {syncs['controls']}")
        lines.append(
            f"  {syncs['total']} blocking syncs over the sampled pools"
        )
        for site in syncs["sites"][:20]:
            lines.append(f"    {site['count']:>9}  {site['location']}")
        lines.append(f"  blind spot: {syncs['blind_spot']}")
    if summary["device_sampler_error"]:
        lines.append(f"\ndevice sampler: {summary['device_sampler_error']}")
    else:
        lines.append(f"\ndevice samples: {summary['device_samples']}")
        # With no samples every period is the fallback, and printing those
        # under "measured" would be the report inventing its own evidence.
        periods = summary.get("device_sample_period_ms") or {}
        if periods and summary["device_samples"]:
            lines.append(
                "  measured refresh: "
                + ", ".join(
                    f"{field} {period:.0f} ms"
                    for field, period in periods.items()
                )
            )
            lines.append(
                "  a phase shorter than a field's refresh reports '-' for it "
                "rather than a value formed before the phase began"
            )
    lines.append("=" * 96)
    return "\n".join(lines)


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
    # under Polar-Express-orthogonalized momentum while embeddings, readout,
    # gains, and every RL-only head stay under AdamW. Old checkpoints
    # (pre-Muon-split) resume with "adamw".
    parser.add_argument(
        "--trunk-optimizer", choices=("muon", "adamw"), default="muon"
    )
    # Default: --learning-rate scaled by pretraining's Muon:generic-Adam
    # ratio (0.025 / 0.015), then by POLAR_EXPRESS_STEP_COMPENSATION (see the
    # derivation in validate_args). NOTE: at equal nominal LR a Muon step
    # moves each element ~1/sqrt(model_dim) as far as AdamW, so this default
    # under-moves the trunk relative to the AdamW baseline; it is the
    # conservative anchor for the planned LR sweep, not a tuned optimum.
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
    # Initialization only: the central bounded output of the state-dependent
    # log-sigma head. -2 gives std ~0.135 and expected 512-D noise norm ~3.06.
    # Its inverse raw bias is only -0.144 in the [-5, 2] tanh map, retaining
    # 98% of the midpoint's local sensitivity. The orthogonal state map starts
    # behind a 0.01 gain, so it adds only mild statewise variation while
    # retaining every input direction.
    parser.add_argument("--thought-log-sigma-init", type=float, default=-2.0)
    parser.add_argument(
        "--thought-sigma-state-init",
        choices=SIGMA_STATE_INIT_KINDS,
        default="orthogonal",
        help=(
            "fresh log-sigma state map; constant is the zero-weight control, "
            "orthogonal is the full-rank 0.01-gain treatment"
        ),
    )
    # Initialization only. The orthogonal map gives an RMS-normalized belief
    # an exactly controlled mean RMS without weakening matrix optimization.
    parser.add_argument("--thought-mean-gain-init", type=float, default=0.1)
    parser.add_argument(
        "--thought-adapter",
        choices=THOUGHT_ADAPTER_KINDS,
        default="orthogonal_silu",
        help=(
            "deployed thought input map; identity_affine is the v25 control, "
            "orthogonal_silu adds full-width random mixing and 2*SiLU"
        ),
    )
    parser.add_argument(
        "--thought-action-transform",
        choices=THOUGHT_ACTION_TRANSFORM_KINDS,
        default="identity",
        help=(
            "recurrent input transform applied after raw Gaussian sampling; "
            "likelihoods and replay storage remain in raw action space"
        ),
    )
    parser.add_argument(
        "--critic-adapter-init",
        choices=CRITIC_ADAPTER_INIT_KINDS,
        default="orthogonal",
        help="fresh critic thought-affine initialization",
    )
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
    # Duck shaping gives every input dimension that happens to share a VALUE
    # on the first trace the same symbol. This model is 512 wide and the
    # rollout batch is rollout_groups * samples_per_prompt = 512, so the
    # first rollout trace unifies the batch dim with model_dim, the first
    # rms_norm against a 512-wide parameter then emits Eq(symbol, 512), and
    # the batch dim is static from then on: the first compaction recompiles.
    # Eval carries the same hazard at 128 rows against head_dim 128. Turning
    # duck shaping off costs some extra symbols and guards and removes the
    # whole class.
    #
    # Off by default on the measurement, not the theory: warm 16-step runs
    # put pool 0 at 37.5 s with it off against 38.4 s on, and the extra
    # step_core specialization it causes cost 6.4 s of compile plus 3.6 s
    # of autotune in the run where the cache was cold for it. Steady state
    # is unchanged rather than better — pool 1 spans 19.9-20.6 s with it on
    # across three runs and 20.3-20.9 s with it off, pool 2 spans 17.1-17.2
    # against 17.2-17.6, and those ranges overlap. Kept as a flag because
    # that is a startup argument, not a throughput one, and a wider model
    # or a different rollout batch moves which dimensions collide.
    parser.add_argument(
        "--duck-shape", action=argparse.BooleanOptionalAction, default=False
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
        "--migrate-projected-thought-resume",
        action="store_true",
        help="explicitly resume a v22 joint-clip checkpoint under the v23 "
        "projected THINK trust region; all training state transfers "
        "verbatim (the pool and its stored behavior statistics are rebuilt "
        "on resume)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    # Profiling. Off by default and costing nothing when off: every call site
    # runs unconditionally against a disabled profiler whose phase object has
    # empty enter/exit. A profiled run is not a training run — it takes a
    # per-pool barrier to resolve CUDA events, runs torch.profiler over whole
    # pools, and turns on sync debug mode — so --profile is recorded in the
    # manifest and refuses a long --steps without --profile-force.
    parser.add_argument(
        "--profile",
        action="store_true",
        help="write a closed per-phase timing breakdown, kernel launch "
        "counts, compile accounting, host-sync attribution and a GPU power "
        "trace to <output>/profile; for diagnosis only, never for a run "
        "whose metrics matter",
    )
    parser.add_argument(
        "--profile-force",
        action="store_true",
        help="allow --profile with more than --profile-max-steps steps",
    )
    parser.add_argument(
        "--profile-max-steps",
        type=int,
        default=40,
        help="largest --steps that --profile accepts without --profile-force",
    )
    parser.add_argument(
        "--profile-pools",
        type=int,
        default=1,
        help="pools that get kernel-level accounting: launch counts and the "
        "top kernels by device time. Measured cost is roughly an eighth of "
        "the profiled pool, so this is a window, not the whole run",
    )
    parser.add_argument(
        "--profile-trace",
        action="store_true",
        help="also write a Chrome trace for each pool in that window; "
        "measured at 2.3 GB per pool, which is past what Perfetto opens, so "
        "the launch counts come without it by default",
    )
    parser.add_argument(
        "--profile-skip-pools",
        type=int,
        default=1,
        help="pools to leave untraced first; pool 0 carries compilation and "
        "allocator growth that no later pool repeats",
    )
    parser.add_argument(
        "--profile-stack",
        action="store_true",
        help="record Python stacks in the Chrome trace; multiplies trace size",
    )
    parser.add_argument(
        "--profile-sync-pools",
        type=int,
        default=1,
        help="pools over which to attribute blocking host syncs to a source "
        "line; the warning path itself is slow, so keep this small. These "
        "run after the traced pools, so reaching them needs at least "
        "--profile-skip-pools + --profile-pools + 1 pools",
    )
    parser.add_argument(
        "--profile-top-kernels",
        type=int,
        default=20,
        help="kernels to report per pool, ranked by total device time",
    )
    parser.add_argument(
        "--profile-device-interval-ms",
        type=int,
        default=250,
        help="nvidia-smi sampling interval for the power and utilization "
        "timeline; 0 disables the sampler",
    )
    parser.add_argument(
        "--profile-power-floor",
        type=float,
        default=150.0,
        help="watts below which the GPU counts as starved; each phase "
        "reports the seconds it spent there, which a mean hides",
    )
    parser.add_argument(
        "--profile-compile-records",
        type=int,
        default=4096,
        help="compilation metrics to retain; torch's deque holds 64 and "
        "evicts the earliest silently",
    )
    parser.add_argument(
        "--profile-reconcile-tolerance",
        type=float,
        default=0.02,
        help="fraction of pool wall time allowed to fall outside every phase "
        "before the run warns that its accounting is not closed",
    )
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
        #
        # POLAR_EXPRESS_STEP_COMPENSATION restores the step size that ratio
        # was chosen for. Muon now orthogonalizes with modded-nanogpt's Polar
        # Express, which deliberately stops in a ripple band around the polar
        # factor instead of converging onto it -- worth about half the step on
        # a low-rank trunk gradient -- and, following that reference, no
        # longer scales rectangular updates by max(1, rows/cols)**0.5, costing
        # the twelve (2048, 512) and (512, 2048) matrices a further 2**0.5.
        # Without this the orthogonalization swap would silently shrink the
        # trunk step by ~2.4x, and the ratio above already under-moves it.
        args.muon_learning_rate = (
            args.learning_rate
            * (0.025 / 0.015)
            * POLAR_EXPRESS_STEP_COMPENSATION
        )
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
        or args.nearby_reward_max >= 1.0
    ):
        parser.error(
            "--nearby-reward-max must be finite, nonnegative, and below "
            "the exact-answer reward of 1"
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
    # A profiled run perturbs what it measures, so it must never be mistaken
    # for a training run whose numbers are quoted.
    if args.profile and not args.profile_force:
        if args.steps > args.profile_max_steps:
            parser.error(
                f"--profile with --steps {args.steps} exceeds "
                f"--profile-max-steps {args.profile_max_steps}; profile a "
                "short run, or pass --profile-force to accept that this "
                "run's timings are not comparable to an unprofiled one"
            )
    for name, value in (
        ("--profile-pools", args.profile_pools),
        ("--profile-skip-pools", args.profile_skip_pools),
        ("--profile-sync-pools", args.profile_sync_pools),
        ("--profile-top-kernels", args.profile_top_kernels),
    ):
        if value < 0:
            parser.error(f"{name} must be nonnegative")
    if args.profile_device_interval_ms < 0:
        parser.error("--profile-device-interval-ms must be nonnegative")
    if not 0.0 <= args.profile_reconcile_tolerance <= 1.0:
        parser.error(
            "--profile-reconcile-tolerance is a fraction of pool wall time"
        )
    if not math.isfinite(args.profile_power_floor) or args.profile_power_floor < 0.0:
        parser.error("--profile-power-floor must be finite and nonnegative")


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
    wrapper = LatentThoughtModel(
        backbone,
        thought_adapter=args.thought_adapter,
        sigma_state_init=args.thought_sigma_state_init,
        thought_action_transform=args.thought_action_transform,
    ).to(device)
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
    wrapper.transition.set_noise_level(args.thought_log_sigma_init)
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
        else:
            init_args = actor_init_payload.get("args", {})
            source_sigma_state_init = init_args.get(
                "thought_sigma_state_init", "constant"
            )
            if source_sigma_state_init != args.thought_sigma_state_init:
                raise ValueError(
                    f"{initialization_path} used --thought-sigma-state-init "
                    f"{source_sigma_state_init!r}, not "
                    f"{args.thought_sigma_state_init!r}; a trained actor must "
                    "retain its initialization lineage"
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
            # A critic-warmup checkpoint has not updated the actor; the
            # explicit migration above installed this run's fresh transition.
            allow_transition_reset=bool(args.actor_critic_init),
            expected_rollout_policy_schema=rollout_policy_schema,
            expected_thought_input_schema=wrapper.thought_input_schema,
            expected_thought_action_transform_schema=(
                wrapper.thought_action_transform_schema
            ),
        )
        wrapper.load_state_dict(actor_init_payload["model"], strict=True)
        actor_init_provenance = {
            "checkpoint": str(initialization_path),
            "critic_initialized": bool(args.actor_critic_init),
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
            "fresh_mean_initialized": bool(args.actor_critic_init),
            "fresh_mean_output_gain": (
                args.thought_mean_gain_init if args.actor_critic_init else None
            ),
            "fresh_log_sigma": (
                args.thought_log_sigma_init if args.actor_critic_init else None
            ),
            "fresh_sigma_state_init": (
                args.thought_sigma_state_init if args.actor_critic_init else None
            ),
            "fresh_adapter_initialized": bool(args.actor_critic_init),
            "fresh_adapter_kind": (
                args.thought_adapter if args.actor_critic_init else None
            ),
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
        adapter_init=args.critic_adapter_init,
        thought_action_transform=args.thought_action_transform,
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
        source_critic_adapter_init = init_args.get(
            "critic_adapter_init", "identity"
        )
        if source_critic_adapter_init != args.critic_adapter_init:
            raise ValueError(
                "--actor-critic-init warm critic used "
                f"--critic-adapter-init {source_critic_adapter_init!r}, not "
                f"{args.critic_adapter_init!r}; rerun critic warmup under the "
                "current adapter initialization"
            )
        expected_optimizer_schema = optimizer_schema_for_trunk_optimizer(
            args.trunk_optimizer
        )
        if (
            actor_init_payload.get("optimizer_schema")
            != expected_optimizer_schema
        ):
            raise ValueError(
                "--actor-critic-init optimizer schema is "
                f"{actor_init_payload.get('optimizer_schema')!r}, expected "
                f"{expected_optimizer_schema!r}; rerun critic warmup under "
                "the current optimizer implementation"
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
        target_execution_schema = execution_schema_for_adapter(
            args.thought_adapter,
            args.thought_action_transform,
        )
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_args = payload.get("args", {})
        source_sigma_state_init = resume_args.get(
            "thought_sigma_state_init", "constant"
        )
        if source_sigma_state_init != args.thought_sigma_state_init:
            raise ValueError(
                "resume checkpoint used --thought-sigma-state-init "
                f"{source_sigma_state_init!r}, not "
                f"{args.thought_sigma_state_init!r}"
            )
        source_critic_adapter_init = resume_args.get(
            "critic_adapter_init", "identity"
        )
        if source_critic_adapter_init != args.critic_adapter_init:
            raise ValueError(
                "resume checkpoint used --critic-adapter-init "
                f"{source_critic_adapter_init!r}, not "
                f"{args.critic_adapter_init!r}"
            )
        migrate_legacy_wrapper_checkpoint(payload, wrapper)
        validate_renderer_checkpoint(
            payload,
            args.resume,
            expected_rollout_policy_schema=rollout_policy_schema,
            expected_thought_input_schema=wrapper.thought_input_schema,
            expected_thought_action_transform_schema=(
                wrapper.thought_action_transform_schema
            ),
        )
        if not resume_execution_schema_compatible(
            payload,
            expected_execution_schema=target_execution_schema,
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
                f"{target_execution_schema!r}; "
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
        if payload.get("actor_objective_schema") != ACTOR_OBJECTIVE_SCHEMA:
            raise ValueError(
                "resume checkpoint actor objective schema must be "
                f"{ACTOR_OBJECTIVE_SCHEMA!r}; got "
                f"{payload.get('actor_objective_schema')!r}. Use "
                "--actor-init for an initialization restart under the "
                "current objective."
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
        expected_optimizer_schema = optimizer_schema_for_trunk_optimizer(
            args.trunk_optimizer
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
        if source_execution_schema == ZERO_AFFINE_EXECUTION_SCHEMA:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["identity_affine_execution_relabel"] = {
                "source_execution_schema": source_execution_schema,
                "target_execution_schema": target_execution_schema,
                "adapter_state": "preserved",
            }
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
                "target_execution_schema": target_execution_schema,
            }
        if anchored_value_migration is not None:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["anchored_value_resume_migration"] = {
                "source_execution_schema": source_execution_schema,
                "target_execution_schema": target_execution_schema,
                **anchored_value_migration,
            }
        if args.migrate_projected_thought_resume:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["projected_thought_resume_migration"] = {
                "source_execution_schema": source_execution_schema,
                "target_execution_schema": target_execution_schema,
                "thought_trust_epsilon": args.thought_trust_epsilon,
                "thought_projection_penalty_coef": (
                    args.thought_projection_penalty_coef
                ),
            }
        if args.migrate_thought_reverse_kl_resume:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["thought_reverse_kl_resume_migration"] = {
                "source_execution_schema": source_execution_schema,
                "target_execution_schema": target_execution_schema,
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

    # Built before the first torch.compile so compile accounting starts from
    # an empty metrics deque, and so every compiled artifact below can be
    # wrapped in a call counter. When --profile is absent this is the
    # do-nothing profiler and every profiling call site below is inert.
    profiler = (
        RunProfiler(Path(args.output), device, args)
        if args.profile
        else DisabledProfiler()
    )

    # Every compiled graph in this process reaches a fp32 head region —
    # FreshThoughtMeanHead, StateDependentLogSigmaHead, or
    # SeparateCritic.value_logits — so all of them carry _enter_autocast
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
    # entry list, so invalidated entries do not accumulate there, and the
    # measured maximum for any frame in this process is 3.

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
        rollout_tail_step_core = profiler.register_artifact(
            "rollout_tail_step",
            torch.compile(
                wrapper.step_core,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,
            ),
        )

    # Preserve the eager function for infrequent no-grad diagnostics. Calling
    # the compiled training artifact under no-grad would create a distinct
    # AOTAutograd specialization solely because grad mode is a Dynamo guard.
    diagnostic_replay_head_inputs = replay_head_inputs

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
        critic.value_logits = profiler.register_artifact(
            "critic_value_logits",
            torch.compile(
                critic.value_logits,
                # Length buckets still span many GEMM shapes. Max-autotune
                # repeatedly stalls training to benchmark each new regime;
                # default Inductor dispatches them to stable cuBLAS kernels.
                mode="default",
                fullgraph=True,
                dynamic=True,
            ),
        )
        compiled_replay = profiler.register_artifact(
            "replay_head_inputs",
            torch.compile(
                replay_head_inputs,
                mode="default",
                fullgraph=True,
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
                # Top level, not buried in args: a profiled run's timings are
                # perturbed by the profiler and must not be quoted as this
                # configuration's cost.
                "profiled": bool(args.profile),
                "profile_schema": PROFILE_SCHEMA if args.profile else None,
                "execution_schema": execution_schema_for_adapter(
                    args.thought_adapter,
                    args.thought_action_transform,
                ),
                "actor_objective_schema": ACTOR_OBJECTIVE_SCHEMA,
                "prompt_order_schema": PROMPT_ORDER_SCHEMA,
                "math_data_identity": data_identity,
                "reward_schema": REWARD_SCHEMA,
                "reasoning_mode": args.reasoning_mode,
                "context_tokens": context_tokens,
                "math_modal_answer_baseline": math_modal_baseline,
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "rollout_policy_schema": rollout_policy_schema,
                "thought_input_schema": wrapper.thought_input_schema,
                "thought_action_transform_schema": (
                    wrapper.thought_action_transform_schema
                ),
                "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
                "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
                "thought_sigma_init_schema": wrapper.sigma_state_init_schema,
                "optimizer_schema": optimizer_schema_for_trunk_optimizer(
                    args.trunk_optimizer
                ),
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
                    "adapter_init_schema": critic.adapter_init_schema,
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

        if args.rollout_groups <= 1:
            for row, encoded in encoded_rows:
                prompt_ids = torch.tensor(
                    encoded,
                    dtype=torch.long, device=device,
                )
                with profiler.phase("decode"):
                    batch = rollout_continuations(
                        wrapper, prompt_ids[None],
                        train_max_new_tokens, max_stream_steps,
                        args.temperature, args.top_p,
                        stop_ids=stop_ids or None,
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

        for chunk_start in range(0, len(encoded_rows), args.rollout_groups):
            encoded_chunk = encoded_rows[
                chunk_start : chunk_start + args.rollout_groups
            ]
            chunk = [row for row, _ in encoded_chunk]
            encoded = [ids for _, ids in encoded_chunk]
            with profiler.phase("prompt_upload"):
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
            with profiler.phase("decode"):
                batched = rollout_continuations(
                    wrapper, prompt_ids, train_max_new_tokens, max_stream_steps,
                    args.temperature, args.top_p, stop_ids=stop_ids or None,
                    prompt_lengths=prompt_lengths,
                    force_initial_think=rollout_force_members(
                        len(chunk), samples
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
                    prompt_repeats=samples,
                    tail_caches=rollout_tail_caches,
                    tail_step_core=rollout_tail_step_core,
                )
            if offload_to_cpu:
                # One chunk-level D2H transfer, then all variable-length
                # splitting, trimming, decoding, and scoring stay on CPU.
                # This avoids one synchronization/transfer per prompt group.
                with profiler.phase("stream_d2h"):
                    batched = compact_stream_to_device(batched, cpu)
            expanded_prompt_lengths = prompt_lengths_cpu.repeat_interleave(
                samples
            )
            if not offload_to_cpu:
                expanded_prompt_lengths = expanded_prompt_lengths.to(device)
            if scoring_pool is None:
                with profiler.phase("score_inline"):
                    split_groups = split_rollout_groups(
                        batched, samples, expanded_prompt_lengths
                    )
                    # Every split owns cloned storage; drop the much larger
                    # groups*samples rollout before the first full-stream
                    # replay.
                    del batched
                    for group_index, (group, row) in enumerate(
                        zip(split_groups, chunk, strict=True)
                    ):
                        groups.append(retain_group(group, row))
                        # Once retained, remove the source from the temporary
                        # chunk list immediately. This bounds GPU storage in
                        # ordinary collection.
                        split_groups[group_index] = None
                        del group
                    del split_groups
            else:
                # The split is 16 groups x 14 tensors of CLONED host storage
                # (~2.4 GiB per chunk); running it here left the GPU with
                # nothing queued between chunk N's last decode step and chunk
                # N+1's first, a measured 0.10-0.18 s hole per chunk. It is
                # pure CPU work on host tensors after
                # compact_stream_to_device's barrier, so it belongs on the
                # worker alongside the trim/decode/score it feeds.
                # Draining the PREVIOUS chunk before submitting this one
                # keeps `groups` in chunk order and still bounds host storage
                # at one pending chunk.
                drain_scored()
                pending_scored.append(
                    scoring_pool.submit(
                        split_and_retain_chunk,
                        batched, expanded_prompt_lengths, chunk,
                    )
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
            with profiler.phase("collect"), training_autocast():
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
        groups = collect(
            refresh_statistics=False,
            prompt_count=pool_prompt_count,
            offload_to_cpu=True,
        )
        with profiler.phase("rollout_diagnostics"):
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
                    args.consume_all_prompts
                    and sampler.cursor == len(math_rows)
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
        packed_real_slot_counts = []
        packed_capacity_slots = 0
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
        ) -> tuple[LatentRolloutBatch, list[LatentRolloutBatch], float]:
            selected = [groups[index] for index in minibatch_order]
            pack_started = time.perf_counter()
            with profiler.worker_span("pack"):
                packed = pack_rollout_groups_for_replay(
                    selected, pin_memory=True
                )
            return packed, selected, time.perf_counter() - pack_started

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
                        cpu_batch, selected_groups, pack_elapsed = (
                            pack_future.result()
                        )
                    pool_cpu_pack_seconds += pack_elapsed
                    if age + 1 < len(minibatch_orders):
                        pack_future = pack_pool.submit(
                            pack_minibatch, minibatch_orders[age + 1]
                        )
                    with profiler.phase("h2d"), device_timed(h2d_events):
                        device_batch = cpu_batch.to(device, non_blocking=True)
                    # Safe without a barrier: the source is pinned, so it came
                    # from the caching host allocator, which records a CUDA
                    # event when the block is freed and withholds it from
                    # reuse until the copy retires.
                    del cpu_batch
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
                    generated = device_batch.action_mask.bool()
                    # masked_fill, not boolean indexing: the latter lowers to
                    # masked_select, whose output size is data dependent, so
                    # it blocks the host on every minibatch. masked_fill also
                    # keeps a non-finite PAD slot out of the sum, which
                    # multiplying by the mask would not.
                    old_value_sums.append(
                        device_batch.old_values.masked_fill(
                            ~generated, 0.0
                        ).sum()
                    )
                    old_value_counts.append(generated.sum())
                    if (
                        retained_bytes_total + packed_batch_bytes
                        <= RETAINED_MINIBATCH_BUDGET_BYTES
                    ):
                        retained_device_batches[age] = device_batch
                        retained_bytes_total += packed_batch_bytes
                    else:
                        with (
                            profiler.phase("stat_scatter_d2h"),
                            device_timed(d2h_events),
                        ):
                            scatter_replay_statistics(
                                device_batch, selected_groups
                            )
                    del device_batch
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
            old_value_sum = float(torch.stack(old_value_sums).sum())
            old_value_count = int(torch.stack(old_value_counts).sum())
            rollout_metrics["old_value_mean"] = (
                old_value_sum / old_value_count if old_value_count else 0.0
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
            rollout_metrics["pool_cpu_pack_seconds"] = pool_cpu_pack_seconds
            # Device time on the transfer and replay streams, not host wall
            # time.
            rollout_metrics["pool_h2d_seconds"] = elapsed_seconds(h2d_events)
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
                "decode_steps_per_chunk_mean",
                "decode_step_utilization",
                "pool_h2d_seconds",
                "pool_refresh_seconds",
                "pool_d2h_seconds",
            ):
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

        for behavior_age, minibatch_order in enumerate(minibatch_orders):
            # Actor and critic each take one optimizer step over the same
            # effective trajectory minibatch. Prompt groups and replay shards
            # are memory partitions, not optimizer minibatches.
            with profiler.phase("update"):
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
                minibatch_h2d_events = []
                if retained_minibatch is not None:
                    # Refreshed on-device in the pool loop above; identical to
                    # what repacking selected_groups would rebuild.
                    device_minibatch = retained_minibatch
                    minibatch_pack_seconds = 0.0
                else:
                    with profiler.phase("minibatch_pack"):
                        minibatch_pack_started = time.perf_counter()
                        cpu_minibatch = pack_rollout_groups_for_replay(
                            selected_groups, pin_memory=True
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
                (
                    policy_action_denominator,
                    gate_action_denominator,
                ) = actor_minibatch_denominators(
                    [device_minibatch], [0]
                )
                with profiler.phase("forward_backward"):
                    metrics = training_update(
                        wrapper, critic, device_minibatch, optimizers,
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
                # Reading the loss and gradient scalars is where the host
                # first waits on the backward, so this phase carries the
                # whole minibatch's device time that forward_backward only
                # enqueued.
                with profiler.phase("grad_telemetry"):
                    actor_dashboard = aggregate_actor_tensorboard_metrics(
                        minibatch_metrics
                    )
                    nonfinite_gradients = {
                        name: actor_dashboard[name]
                        for name in (
                            "grad/trunk", "grad/gate", "grad/adapter",
                            "grad/renderer", "grad/sigma",
                            "grad/thought_mean", "grad/critic",
                        )
                        if not math.isfinite(actor_dashboard[name])
                    }
                if nonfinite_gradients:
                    raise RuntimeError(
                        "non-finite gradients before optimizer step: "
                        f"{nonfinite_gradients}"
                    )
                with profiler.phase("optimizer_step"):
                    step_optimizers(optimizers, "actor")
                    step_optimizers(optimizers, "critic")
                    # Gradient buffers have already been reduced to scalar
                    # telemetry. Release them before the optional second
                    # replay so this read-only diagnostic cannot stack an
                    # inference forward on top of the training step's peak
                    # allocation.
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
                    elapsed_seconds(minibatch_h2d_events)
                )
                actor_dashboard["perf/minibatch_cpu_pack_seconds"] = (
                    minibatch_pack_seconds
                )
                if next_step == 1 or (
                    args.post_update_kl_every > 0
                    and next_step % args.post_update_kl_every == 0
                ):
                    drift_started = time.perf_counter()
                    with profiler.phase("post_update_drift"), training_autocast():
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
                bench_eval(step)
        if crossed_interval(previous_step, step, args.save_every):
            with profiler.phase("checkpoint_save"):
                save_started = time.perf_counter()
                save_checkpoint(
                    output / "latent_vapo_checkpoint.pt", wrapper, critic,
                    optimizers, step, args, sampler, warmup_step,
                    actor_init_provenance,
                )
                save_seconds = time.perf_counter() - save_started
                logger.log(type="checkpoint", step=step, seconds=save_seconds)
                tensorboard.add_scalar(
                    "perf/checkpoint_seconds", save_seconds, step
                )
        profiler.pool_finished(step, time.perf_counter() - started)
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
    profiler.close()


if __name__ == "__main__":
    main()

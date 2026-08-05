"""Latent-thought policy gradients over a deterministic hidden carry.

The WHOLE deployed policy path trains at RL time — no frozen trunk. The only
actions are tokens; latent thinking is deterministic function structure, not
a stochastic action channel.

- A "thought" is the post-final-norm belief that produced a generated token.
  When that token feeds back as input, its producing belief rides along as a
  gated residual on the token embedding through the wrapper's
  ``CombinedEmbedding`` (gain*W(h) + type bias, then prenorm-residual relu^2
  MLP blocks). Prompt tokens and the first generation step carry no hidden.
  The carried belief is detached replay data — gradients reach the trunk,
  embeddings, and combiner through the teacher-forced pass, never backward
  through time.
- The default actor is Delightful Policy Gradient: one differentiable
  teacher-forced replay applies Osband's sigmoid gate to advantage times
  current-token surprisal, with eta=1 and no importance ratio or clipping.
  ``--no-delightful-policy-gradient`` selects the historical token-level VAPO
  control with DAPO's asymmetric token-ratio clip.
  ``--target-policy-optimization`` instead uses raw critic GAE at every
  visited prefix to shift behavior-anchored executed-token-versus-rest odds.
  It has no PG auxiliary, comparison actions, or action-Q critic. In every
  mode, tokens are the only actions.
- The critic is a SEPARATE from-scratch model (same architecture class,
  fresh weights, fully trainable, no SIGReg or latent prediction) trained
  purely by HL-Gauss cross-entropy on [0, 1] value targets. It re-derives
  combined embeddings with its own combiner weights, so credit lands on the
  same carried-belief inputs the actor conditions on. The support anchors
  bin CENTERS at exactly 0 and 1 with margin bins beyond each — Dreamer3's
  exact-zero bucket applied to both ends of the unit range — so the dominant
  exact-0/exact-1 verifier targets project symmetrically instead of decoding
  a truncation bias inward; labels smooth at sigma_ratio 1.0.
- No pretraining anchor: SIGReg and the latent target-prediction objective
  are dropped at RL time. The combiner trains purely on its ability to make
  carried beliefs useful; the teacher-forced val-BPB guard is the drift
  detector.

Deferred by design: test-time-read-compute (carrying hiddens for read/prompt
tokens) is out of scope for this policy schema.

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

import train_gpt as baseline
from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters
import postraining.latent_rollout
from postraining.latent_eval import evaluate_latent_math
from postraining.benchmark_report import write_benchmark_report
from postraining.core import (
    POSTTRAIN_REWARD_SCHEMA,
    JsonlLogger,
    POSTTRAIN_CONTEXT_TOKENS,
    ANSWER_CLOSE,
    ANSWER_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    POSTTRAIN_PROMPT_TOKENS,
    POSTTRAIN_RESPONSE_TOKENS,
    answer_style,
    clipped_policy_loss,
    delightful_policy_loss,
    target_policy_loss,
    encode_prompt,
    extract_final_answer,
    deterministic_math_subset,
    fenced_answer_text,
    single_fence_span,
    structural_format_ok,
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
    LatentRolloutBatch,
    ReplayPlan,
    assign_terminal_rewards,
    build_replay_plan,
    compact_emit_token_log_odds,
    compact_emit_token_logprobs,
    compact_next_slots,
    compact_slots,
    compact_stream_to_device,
    emitted_token_rows,
    pack_rollout_groups_for_replay,
    iter_planned_replay_microbatches,
    refresh_old_statistics,
    replay_head_inputs,
    scatter_replay_statistics,
    scatter_slots,
    rollout_continuations,
    split_rollout_groups,
    trim_stream,
)
from postraining.latent_thought import (
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
from postraining.model_io import fresh_trunk, load_model
from postraining.muon import Muon
from postraining.reasoning_modes import (
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
from postraining.value_model import SeparateCritic
from postraining.vapo.config import build_arg_parser, validate_args
from postraining.vapo.code_reward import (
    PYTHON_REWARD_SCHEMA,
    PYTHON_RESULT_CODES,
    batch_python_test_results,
)
from postraining.vapo.mixture import (
    MixedPromptSampler,
    file_sha256,
    load_mixture_manifest,
    mixture_identity,
    rollout_window_source_quotas,
)
from postraining.vapo.schemas import (
    PROMPT_ORDER_SCHEMA,
    REPLAY_NUMERICS_SCHEMA,
    actor_objective_schema,
    execution_schema_for_rollout_scheduler,
    optimizer_schema_for_trunk_optimizer,
    resume_execution_schema_compatible,
    resume_replay_schema_compatible,
    value_support_geometry_matches,
)


# Refreshed device minibatches held for their update instead of the
# scatter-to-CPU/repack/re-upload round trip. Bounded because packed batch
# bytes track the pool's longest stream and each slot stores an fp32
# carried hidden; past the budget the refresh loop falls back to the
# scatter path for the remaining minibatches. VRAM-resident only between a
# pool's refresh and its last update — never across a collection.
RETAINED_MINIBATCH_BUDGET_BYTES = 8 << 30
# v2: the grading style follows each row's reward_model.style (Minerva for
# DAPO/AIME lineage data, official exact match for mathematics_dataset rows)
# instead of Minerva-normalizing everything.
REWARD_SCHEMA = POSTTRAIN_REWARD_SCHEMA
SOURCE_SIGNAL_MASK_SCHEMA = "per_source_zero_reward_actor_mask/v1"
RESUME_ARG_CONTRACT_SCHEMA = "vapo_exact_environment_objective_args/v1"
RESUME_EXACT_ARG_FIELDS = (
    "reasoning_mode",
    "answer_tokens",
    "prompt_tokens",
    "continuation_tokens",
    "samples_per_prompt",
    "ppo_epochs",
    "temperature",
    "top_p",
    "value_bins",
    "value_anchored_support",
    "value_margin_bins",
    "value_sigma_ratio",
    "value_prior",
    "combined_mlp_blocks",
    "combined_mlp_hidden",
    "gae_lambda_alpha",
    "nearby_reward_max",
    "think_tokens",
    "think_min_tokens",
    "answer_fence",
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


def validate_resume_arg_contract(saved: dict, current: argparse.Namespace) -> None:
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
    if (
        bool(saved.get("target_policy_optimization"))
        or bool(getattr(current, "target_policy_optimization", False))
    ):
        exact_fields += TPO_RESUME_EXACT_ARG_FIELDS
    mismatches = [
        (name, saved.get(name), getattr(current, name))
        for name in exact_fields
        if saved.get(name) != getattr(current, name)
    ]
    if mismatches:
        details = ", ".join(
            f"{name}: checkpoint={before!r}, current={after!r}"
            for name, before, after in mismatches
        )
        raise ValueError(
            "exact resume changed environment/objective arguments: " + details
        )


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
    }
    after = {
        "prompts_per_rollout": int(current.prompts_per_rollout),
        "prompts_per_minibatch": int(current.prompts_per_minibatch),
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
) -> None:
    """Exact verifier reward plus bounded final-answer numeric proximity.

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
    scores = []
    gate_zeroed_correct = 0
    for emitted in emitted_token_rows(batch):
        stop_cut = next(
            (index for index, token in enumerate(emitted) if token in stop_ids),
            None,
        )
        if stop_cut is None:
            scores.append(0.0)
            continue
        visible = emitted[: stop_cut + 1]
        solution = tokenizer.decode(list(solution_prefix_ids) + visible)
        correct, _ = verify_answer(solution, truth, style)
        if answer_fence_ids is not None:
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
                    correct, _ = verify_answer(
                        "Answer: " + relaxed, truth, style, window=None
                    )
                gate_zeroed_correct += bool(correct)
                scores.append(0.0)
                continue
            # The gate guarantees a single non-empty anchored span, so
            # the graded field is the fenced value re-framed for the
            # verifier; window=None because the reframe IS the field —
            # a long value must not push its own prefix out of the
            # verifier's tail window and grade [INVALID].
            solution = "Answer: " + fenced_answer_text(
                visible, tokenizer, answer_fence_ids
            )
            correct, _ = verify_answer(solution, truth, style, window=None)
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
            continue
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
    batch.verifier_status = torch.full(
        (batch.kind.size(0),),
        PYTHON_RESULT_CODES["not_applicable"],
        dtype=torch.long,
        device=batch.rewards.device,
    )
    assign_terminal_rewards(
        batch, torch.tensor(scores, dtype=torch.float32, device=batch.rewards.device)
    )


def score_python_rollout(
    batch: LatentRolloutBatch,
    verification_info: dict,
    tokenizer,
    stop_ids: tuple[int, ...],
    think_fence_ids: tuple[int, int],
    answer_fence_ids: tuple[int, int],
    min_think_tokens: int,
) -> None:
    """Binary all-tests-pass reward over the strict fenced code span."""
    if verification_info.get("schema") != PYTHON_REWARD_SCHEMA:
        raise ValueError("Python row uses an incompatible verifier schema")
    answers: list[str] = []
    eligible_indices: list[int] = []
    emitted_rows = emitted_token_rows(batch)
    scores = [0.0] * len(emitted_rows)
    statuses = [PYTHON_RESULT_CODES["format_ineligible"]] * len(emitted_rows)
    for index, emitted in enumerate(emitted_rows):
        stop_cut = next(
            (position for position, token in enumerate(emitted) if token in stop_ids),
            None,
        )
        if stop_cut is None:
            continue
        visible = emitted[: stop_cut + 1]
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
        results = batch_python_test_results(answers, verification_info)
        for index, result in zip(eligible_indices, results, strict=True):
            scores[index] = float(result == "pass")
            statuses[index] = PYTHON_RESULT_CODES[result]
    batch.verifier_status = torch.tensor(
        statuses, dtype=torch.long, device=batch.rewards.device
    )
    batch.think_gate_zeroed_correct = 0
    assign_terminal_rewards(
        batch,
        torch.tensor(scores, dtype=torch.float32, device=batch.rewards.device),
    )


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
) -> dict[str, float | int]:
    """Pool-level rollout metrics.

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
    exact = batch.reward_scalar == 1.0
    grouped_exact = exact.reshape(-1, samples_per_prompt).float()
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
        "actions_per_trajectory": float(batch.action_mask.sum(1).mean()),
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
) -> dict[str, float | int]:
    """Mean of per-group rollout diagnostics; trajectory counts are summed."""
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
        )
        for group in groups
    ]
    aggregated: dict[str, float | int] = {}
    # Derived per-group keys (currently the conditional think mean) can
    # be present in some groups and absent in others; they are rebuilt
    # from their pooled numerators below, never averaged group-wise.
    derived = {"think_tokens_mean"}
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
        elif key in ("think_tokens_sum", "think_tokens_count"):
            aggregated[key] = float(sum(values))
        else:
            aggregated[key] = float(sum(values) / len(values))
    if aggregated.get("think_tokens_count"):
        aggregated["think_tokens_mean"] = float(
            aggregated["think_tokens_sum"] / aggregated["think_tokens_count"]
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
) -> dict[str, dict[str, float | int]]:
    """Compute the same rollout diagnostics independently for every source."""
    by_source: dict[int, list[LatentRolloutBatch]] = {
        source_id: [] for source_id in range(len(source_names))
    }
    for group in groups:
        source_id = source_group_id(group)
        if source_id not in by_source:
            raise ValueError(f"rollout carries unknown source id {source_id}")
        by_source[source_id].append(group)
    missing = [source_names[index] for index, rows in by_source.items() if not rows]
    if missing:
        raise ValueError(f"rollout pool omitted configured sources: {missing}")
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
        )
        for source_id, source_groups in by_source.items()
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
    last = metrics[-1]
    # Each group is already divided by the denominator of the complete actor
    # optimizer minibatch, so its loss is a contribution to be summed rather
    # than another independently normalized minibatch mean.
    policy_contribution = sum(metric["policy_loss"] for metric in metrics)
    dashboard = {
        "loss/policy": policy_contribution,
        "loss/actor_total": policy_contribution,
        "value/token_weighted_excess_ce": value["excess_ce"],
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
        "behavior/actions_per_trajectory": float(
            metrics["actions_per_trajectory"]
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
    source_quotas: list[int],
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Partition one complete mixture cycle with identical source shares.

    Every optimizer step sees the same source composition, preventing an
    unlucky shuffle from turning one domain into an all-zero minibatch while
    another domain supplies all of the policy signal.
    """
    group_count = len(groups)
    if group_count < 1 or group_count % groups_per_minibatch:
        raise ValueError("a stratified pool must contain complete minibatches")
    if sum(source_quotas) != group_count:
        raise ValueError("source quotas must exactly describe the rollout pool")
    minibatch_count = group_count // groups_per_minibatch
    if any(quota % minibatch_count for quota in source_quotas):
        raise ValueError(
            "each source quota must divide across all optimizer minibatches"
        )
    indices_by_source = [[] for _ in source_quotas]
    for index, group in enumerate(groups):
        source_id = source_group_id(group)
        if not 0 <= source_id < len(source_quotas):
            raise ValueError(f"rollout carries unknown source id {source_id}")
        indices_by_source[source_id].append(index)
    observed = [len(indices) for indices in indices_by_source]
    if observed != source_quotas:
        raise ValueError(
            f"rollout source counts {observed} do not match quotas {source_quotas}"
        )
    for indices in indices_by_source:
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        indices[:] = [indices[position] for position in permutation]

    minibatches = [[] for _ in range(minibatch_count)]
    for source_id, quota in enumerate(source_quotas):
        per_minibatch = quota // minibatch_count
        for minibatch_index in range(minibatch_count):
            start = minibatch_index * per_minibatch
            minibatches[minibatch_index].extend(
                indices_by_source[source_id][start : start + per_minibatch]
            )
    for minibatch in minibatches:
        if len(minibatch) != groups_per_minibatch:
            raise AssertionError("stratified optimizer minibatch has wrong size")
        permutation = torch.randperm(
            len(minibatch), generator=generator
        ).tolist()
        minibatch[:] = [minibatch[position] for position in permutation]
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
) -> torch.Tensor:
    """Activate actor rows only when their source has an exact success.

    In a mixed minibatch, aggregate reward can hide an entirely zero-reward
    source. Its critic targets remain useful calibration data, but its actor
    advantages are pure baseline error and push away arbitrary sampled text.
    This is deliberately not per-prompt reward-variance filtering: failed
    prompt groups inside a source with successes retain dense critic-GAE
    credit. The pairwise formulation avoids a device-to-host sync for source
    counts.
    """
    if source_ids is None:
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
    critic_learning_rate: float | None = None,
    trunk_optimizer: str = "adamw",
    muon_learning_rate: float | None = None,
    critic_muon_learning_rate: float | None = None,
    fused: bool = True,
) -> dict[str, torch.optim.Optimizer]:
    """The actor/critic optimizer layout.

    One actor AdamW holds three semantic groups for exact telemetry and
    checkpoint validation — trunk, combiner, renderer — and every trainable
    policy component uses one general actor learning rate. The critic may use
    its own constant rate so actor-rate ablations do not alter value warmup.
    The fresh probes live under
    ``blocks[-1]`` (so pretraining's optimizer saw them), which makes
    name-prefix filtering wrong — exclude them from the trunk by identity.
    Nano backbones keep the identical three-group layout with their native
    readout as the renderer group.

    ``trunk_optimizer="muon"`` restores the pretraining update geometry:
    block matrices (ndim >= 2) move from the AdamW trunk group into separate
    ``actor_muon``/``critic_muon`` Muon optimizers. Polar Express supplies
    the orthogonalized momentum update, while embeddings, readout, gains,
    and the combiners stay under AdamW (the combiner is not under
    ``backbone.blocks``, so Muon routing never sees it). Weight decay stays
    0 everywhere: pretraining's Muon decay (0.05) regularizes a from-scratch
    run, but over a long RL schedule it would only shrink the pretrained
    weights.
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
                {"params": list(wrapper.combiner.parameters()), "lr": learning_rate},
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
    replay_max_trajectories: int = 32,
    replay_attention_budget: int = 4 * 1024 * 1024,
    replay_bucket: int = 1,
    replay_slot_budget: int | None = None,
    replay_plan: ReplayPlan | None = None,
    delightful_policy_gradient: bool = False,
    target_policy_optimization: bool = False,
    tpo_eta: float = 2.0,
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
    if delightful_policy_gradient and target_policy_optimization:
        raise ValueError("actor objectives are mutually exclusive")
    if not value_only and not batch.statistics_refreshed:
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
    if policy_action_denominator is None:
        policy_action_denominator = batch.action_mask.sum()
    if value_action_denominator is None:
        value_action_denominator = batch.action_mask.sum()
    zero = batch.action_mask.new_zeros(())
    totals = {
        key: zero.clone()
        for key in (
            "value_loss", "value_sum", "value_target_sum", "target_entropy_sum",
            "policy_loss", "policy_clip", "token_kl_sum",
            "advantage_sum", "advantage_square_sum", "target_square_sum",
            "residual_sum", "residual_square_sum",
            "token_abs_log_ratio_max",
            "harmful_positive_log_ratio_max",
            "delightful_gate_sum", "delightful_positive_gate_sum",
            "delightful_positive_count", "delightful_negative_gate_sum",
            "delightful_negative_count", "delightful_delight_sum",
            "delightful_surprisal_sum", "delightful_positive_loss_sum",
            "delightful_negative_loss_sum",
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
        )
    }

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
    actor_signal_rows = source_actor_signal_mask(
        batch.source_id, batch.reward_scalar
    )
    if not value_only:
        advantages = advantages * actor_signal_rows[:, None]
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

    for shard_index, (microbatch, shard) in enumerate(
        iter_planned_replay_microbatches(batch, replay_plan)
    ):
        rows = shard.rows
        stream_length = shard.stream_length
        micro_advantages = advantages[rows, :stream_length]
        micro_value_targets = value_targets[rows, :stream_length]
        local_action_count = microbatch.action_mask.sum()
        value_logits = critic.value_logits(microbatch)
        value_ce = critic.support.cross_entropy(value_logits, micro_value_targets)
        local_value_numerator = (value_ce * microbatch.action_mask).sum()
        emit_index = shard.emit_index if not value_only else None
        weighted_critic_loss = (
            local_value_numerator / value_action_denominator.clamp_min(1)
        )
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
        if value_only:
            continue

        # A stream position is one MDP action: emitting one token from the
        # combined (embedding + carried belief) input. The stored hiddens are
        # replay constants, so this teacher-forced pass reaches the trunk,
        # embeddings, and combiner without any gradient through time.
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
        if target_policy_optimization:
            compact_token_log_odds = compact_emit_token_log_odds(
                wrapper, emit_inputs, emit_beliefs, emit_targets
            )
            compact_token_logprobs = F.logsigmoid(compact_token_log_odds)
        else:
            compact_token_logprobs = compact_emit_token_logprobs(
                wrapper, emit_inputs, emit_beliefs, emit_targets
            )
        new_token_logprobs = torch.zeros_like(microbatch.old_token_logprobs)
        scatter_slots(new_token_logprobs, emit_index, compact_token_logprobs)

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
                actor_active_mask[rows, :stream_length],
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
            weighted_policy_loss, delightful = delightful_policy_loss(
                new_token_logprobs,
                micro_advantages,
                microbatch.action_mask,
                denominator=policy_action_denominator,
            )
            weighted_policy_clip = weighted_policy_loss.detach().new_zeros(())
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
            weighted_policy_loss, weighted_policy_clip, _ = clipped_policy_loss(
                new_token_logprobs,
                microbatch.old_token_logprobs,
                micro_advantages,
                microbatch.action_mask,
                denominator=policy_action_denominator,
                estimate_kl=False,
            )
        actor_total = weighted_policy_loss
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
            token_log_ratio = (
                new_token_logprobs - microbatch.old_token_logprobs
            )
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
        "value_excess_ce": (
            totals["value_loss"] / action_denom
            - totals["target_entropy_sum"] / action_denom
        ),
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_variance.sqrt(),
        "action_count": batch.action_mask.sum(),
        "actor_active_trajectory_fraction": actor_signal_rows.float().mean(),
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
        "combiner_grad_norm": gradient_norm_tensor(
            wrapper.combiner.parameters()
        ),
        "critic_grad_norm": gradient_norm_tensor(critic.parameters()),
    }
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
        **grad_norms,
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
) -> dict[str, float]:
    """Evaluate the just-updated policy on its behavior trajectories.

    On behavior-age 0, optimized ratios are exactly one before the first actor
    step; clipping therefore cannot reveal how far that step moved the deployed
    policy. This read-only replay measures the actual post-step drift
    without reusing trajectories for a gradient. It runs only at the explicit
    diagnostic cadence because it costs one additional policy forward.
    ``replay_function`` and ``emit_logprob_function`` deliberately stay eager
    in the trainer: invoking a grad-enabled training artifact under this
    function's no-grad context would force Dynamo/AOTAutograd to compile a
    second guarded graph, while enabling gradients here retains a full replay
    graph and can exceed peak memory. Both are parameters rather than module
    lookups precisely because the trainer rebinds those globals to compiled
    artifacts.
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
                length_adaptive_lambda(action_counts, gae_lambda_alpha),
            )
            actor_signal_rows = source_actor_signal_mask(
                batch.source_id, batch.reward_scalar
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
) -> None:
    reasoning_mode = getattr(args, "reasoning_mode", "latent")
    payload = {
        "step": step,
        "value_warmup_step": warmup_step,
        "execution_schema": execution_schema_for_rollout_scheduler(
            getattr(args, "rollout_scheduler", "lockstep")
        ),
        "actor_objective_schema": actor_objective_schema(
            getattr(args, "delightful_policy_gradient", False),
            getattr(args, "target_policy_optimization", False),
        ),
        "resume_arg_contract_schema": RESUME_ARG_CONTRACT_SCHEMA,
        "source_signal_mask_schema": (
            SOURCE_SIGNAL_MASK_SCHEMA
            if getattr(args, "rl_mixture_manifest", None)
            else None
        ),
        "python_reward_schema": (
            PYTHON_REWARD_SCHEMA
            if getattr(args, "rl_mixture_manifest", None)
            else None
        ),
        "replay_numerics_schema": REPLAY_NUMERICS_SCHEMA,
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
        "reward_schema": REWARD_SCHEMA,
        "reasoning_mode": reasoning_mode,
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "rollout_policy_schema": rollout_policy_schema_for_mode(reasoning_mode),
        "thought_input_schema": wrapper.thought_input_schema,
        "optimizer_schema": optimizer_schema_for_trunk_optimizer(
            getattr(args, "trunk_optimizer", "adamw")
        ),
        "model": wrapper.state_dict(),
        "critic": critic.state_dict(),
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "args": vars(args),
        "sampler_cursor": sampler.cursor,
        "zero_reward_frozen_updates": int(zero_reward_frozen_updates),
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
    is_recurrent_trunk = "_kda_" in backbone.architecture
    trunk_fullgraph = not is_recurrent_trunk

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

    def mode_budgets(max_tokens: int) -> tuple[int, int]:
        return mode_rollout_budget(
            args.reasoning_mode,
            max_tokens,
            answer_tokens=args.answer_tokens,
        )

    train_max_new_tokens, max_stream_steps = training_rollout_budget(
        args.reasoning_mode,
        args.continuation_tokens,
        answer_tokens=args.answer_tokens,
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
    # Full-model RL: every parameter on a deployed policy path trains — trunk,
    # embeddings, renderer, and the combined-embedding stack. The retired
    # pretrained prediction projector remains checkpointed but has no graph
    # edge; the backbone critic probe is frozen and unused.
    wrapper = LatentThoughtModel(
        backbone,
        mlp_hidden=args.combined_mlp_hidden,
        num_blocks=args.combined_mlp_blocks,
    ).to(device)
    # No module here behaves differently under train(): pin eval mode once so
    # the training flag (a dynamo guard) never flips between the step-0 evals
    # and the training loop and re-specializes the compiled step.
    wrapper.eval()
    actor_init_payload = None
    actor_init_provenance = None
    initialization_path = (
        args.actor_critic_init or args.actor_init or args.curriculum_init
    )
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
        # The combiner's tensor shapes are fixed by the saved CLI arguments;
        # a geometry mismatch would otherwise surface as an opaque strict-load
        # shape error (or, for the loaded gain scalar, not at all).
        saved_combiner = combiner_init_kwargs_from_checkpoint(actor_init_payload)
        if (
            saved_combiner["mlp_hidden"],
            saved_combiner["num_blocks"],
        ) != (args.combined_mlp_hidden, args.combined_mlp_blocks):
            raise ValueError(
                f"{initialization_path} was trained with combiner geometry "
                f"mlp_hidden={saved_combiner['mlp_hidden']}/"
                f"blocks={saved_combiner['num_blocks']}, not "
                f"{args.combined_mlp_hidden}/{args.combined_mlp_blocks}; "
                "pass the checkpoint's --combined-mlp-hidden and "
                "--combined-mlp-blocks"
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
            "combined_mlp_hidden": args.combined_mlp_hidden,
            "combined_mlp_blocks": args.combined_mlp_blocks,
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
        mlp_hidden=args.combined_mlp_hidden,
        num_blocks=args.combined_mlp_blocks,
    ).to(device)
    critic.eval()  # no dropout in this architecture; keep norms deterministic
    if actor_init_payload is not None:
        # These three apply on EVERY initialization path (--actor-init and
        # --curriculum-init load the same actor weights): a fence-setting
        # mismatch means the loaded policy either never trained the fence
        # rows (all-zero structurally-gated reward) or emits fence ids the
        # tokenizer silently drops (no gate at all). Red-team round 2: the
        # guards originally lived under --actor-critic-init only, whose
        # own error text recommended --actor-init — the unguarded path.
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
    if args.actor_critic_init:
        init_args = actor_init_payload.get("args", {})
        if not value_support_geometry_matches(init_args, args):
            raise ValueError(
                "--actor-critic-init warm critic was trained on a different "
                "value support geometry (anchored/bins/margin/sigma_ratio "
                f"{init_args.get('value_anchored_support', False)}/"
                f"{init_args.get('value_bins')}/"
                f"{init_args.get('value_margin_bins')}/"
                f"{init_args.get('value_sigma_ratio')}); rerun critic warmup "
                "under the current flags or start with --actor-init"
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
        critic_learning_rate=args.critic_learning_rate,
        trunk_optimizer=args.trunk_optimizer,
        muon_learning_rate=args.muon_learning_rate,
        critic_muon_learning_rate=args.critic_muon_learning_rate,
    )
    if args.actor_critic_init:
        # A step-0 critic-warm checkpoint has never stepped its actor. Its
        # optimizer state is therefore empty and carries no momentum to
        # preserve. Preserve the trained critic's optimizer state and start
        # the untouched actor optimizer in the current three-group layout.
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
        backbone.architecture,
        FreshHyperparameters.tokenizer_path,
        think_tokens=args.think_tokens,
        answer_tokens=args.answer_fence,
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
    mixture_rollout_source_quotas = None
    if args.rl_mixture_manifest:
        math_rows, mixture_sources, mixture_manifest = load_mixture_manifest(
            args.rl_mixture_manifest
        )
        if (
            mixture_manifest.get("answer_fence_prompt_schema")
            != ANSWER_FENCE_PROMPT_SCHEMA
        ):
            raise ValueError("mixture uses a different answer-fence prompt schema")
        if mixture_manifest.get("python_reward_schema") != PYTHON_REWARD_SCHEMA:
            raise ValueError("mixture uses a different Python reward schema")
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
        math_rows = load_unique_math_rows(args.math_data)
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
    math_modal_baseline["shaped_reward"] = (
        sum(modal_shaped_rewards) / len(modal_shaped_rewards)
        if modal_shaped_rewards
        else 0.0
    )
    if mixture_manifest is None:
        data_identity = math_dataset_identity(args.math_data, args.exclude_modules)
        sampler = MathPromptSampler(
            math_rows, args.seed, dataset_identity=data_identity
        )
    else:
        data_identity = mixture_identity(
            args.rl_mixture_manifest, mixture_manifest
        )
        assert mixture_sources is not None
        sampler = MixedPromptSampler(
            mixture_sources,
            args.seed,
            dataset_identity=data_identity,
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
    resume_args = None
    resumed_zero_reward_frozen_updates = None
    if args.resume:
        target_execution_schema = execution_schema_for_rollout_scheduler(
            args.rollout_scheduler
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
        saved_combiner = combiner_init_kwargs_from_checkpoint(payload)
        if (
            saved_combiner["mlp_hidden"],
            saved_combiner["num_blocks"],
        ) != (args.combined_mlp_hidden, args.combined_mlp_blocks):
            raise ValueError(
                "resume checkpoint was trained with combiner geometry "
                f"mlp_hidden={saved_combiner['mlp_hidden']}/"
                f"blocks={saved_combiner['num_blocks']}, not "
                f"{args.combined_mlp_hidden}/{args.combined_mlp_blocks}; "
                "resume with the checkpoint's combiner flags"
            )
        validate_renderer_checkpoint(
            payload,
            args.resume,
            expected_rollout_policy_schema=rollout_policy_schema,
        )
        if not resume_execution_schema_compatible(
            payload,
            expected_execution_schema=target_execution_schema,
        ):
            raise ValueError(
                "resume checkpoint execution schema must be "
                f"{target_execution_schema!r}; "
                f"got {payload.get('execution_schema')!r}. v28 replaced the "
                "stochastic thought/gate policy with the deterministic "
                "hidden carry; no saved state from an earlier schema has a "
                "sound interpretation under it, so there are no migrations. "
                "Start a fresh run, or initialize from a v28 checkpoint via "
                "--actor-init/--actor-critic-init."
            )
        if not resume_replay_schema_compatible(payload):
            raise ValueError(
                "resume checkpoint replay numerics schema must be "
                f"{REPLAY_NUMERICS_SCHEMA!r}; got "
                f"{payload.get('replay_numerics_schema')!r}; pre-v28 replay "
                "layouts are never migrated"
            )
        if payload.get("reward_schema") != REWARD_SCHEMA:
            raise ValueError(
                f"resume checkpoint reward schema must be {REWARD_SCHEMA!r}; "
                f"got {payload.get('reward_schema')!r}"
            )
        expected_actor_objective_schema = actor_objective_schema(
            args.delightful_policy_gradient,
            args.target_policy_optimization,
        )
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
            SOURCE_SIGNAL_MASK_SCHEMA if args.rl_mixture_manifest else None
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
        expected_python_reward_schema = (
            PYTHON_REWARD_SCHEMA if args.rl_mixture_manifest else None
        )
        if payload.get("python_reward_schema") != expected_python_reward_schema:
            raise ValueError(
                "resume checkpoint Python reward schema must be "
                f"{expected_python_reward_schema!r}; got "
                f"{payload.get('python_reward_schema')!r}"
            )
        if payload.get("math_data_identity") != data_identity:
            raise ValueError(
                "resume checkpoint's prompt cursor belongs to different dataset "
                "bytes, exclusions, or ordering"
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
        if not value_support_geometry_matches(resume_args, args):
            raise ValueError(
                "resume checkpoint critic support geometry "
                "(anchored/bins/margin/sigma_ratio "
                f"{resume_args.get('value_anchored_support', False)}/"
                f"{resume_args.get('value_bins')}/"
                f"{resume_args.get('value_margin_bins')}/"
                f"{resume_args.get('value_sigma_ratio')}) does not match "
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

    if args.actor_critic_init:
        torch.set_rng_state(actor_init_payload["cpu_rng"])
        torch.cuda.set_rng_state_all(actor_init_payload["cuda_rng"])
        random.setstate(actor_init_payload["python_rng"])

    if mixture_sources is not None:
        mixture_rollout_source_quotas = rollout_window_source_quotas(
            mixture_sources,
            args.prompts_per_rollout,
            allow_balanced_rotation=(
                args.delightful_policy_gradient
                or args.target_policy_optimization
            ),
            start_cursor=sampler.cursor,
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

    # Every compiled graph in this process reaches a fp32 region —
    # CombinedEmbedding's inject or SeparateCritic.value_logits both disable
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
    compiled_generation_step = None
    if (
        args.eval_compile
        or (
            args.rollout_compile
            and args.rollout_scheduler == "lockstep"
        )
    ):
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
        compiled_generation_step
        if args.rollout_compile and args.rollout_scheduler == "lockstep"
        else None
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
    rollout_graph_rows = max(args.rollout_groups, 1) * args.samples_per_prompt

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

    if args.rollout_flex_decode and rollout_tail_caches is not None:
        rollout_tail_decode_mask = DecodeRangeMask(
            args.rollout_tail_batch,
            rollout_cache_width,
            device,
            block_size=decode_block_size,
        )

    # Preserve the eager functions for infrequent no-grad diagnostics. Calling
    # a compiled training artifact under no-grad would create a distinct
    # AOTAutograd specialization solely because grad mode is a Dynamo guard.
    # Both the replay head and the readout tail get rebound below, so both
    # need capturing here.
    diagnostic_replay_head_inputs = replay_head_inputs
    diagnostic_emit_token_logprobs = compact_emit_token_logprobs
    diagnostic_emit_token_log_odds = compact_emit_token_log_odds

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
    # A DG rollout is consumed in one optimizer update, so it needs no
    # within-pool source stratification. This also permits the deterministic
    # 24-prompt rotation (10/8/3/3, then 11/7/3/3) while preserving the exact
    # broad-v5 ratio over its eight-update cursor phase cycle.
    mixture_source_quotas = (
        mixture_rollout_source_quotas
        if mixture_manifest is not None
        and not (
            args.delightful_policy_gradient
            or args.target_policy_optimization
        )
        else None
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
                "source_provenance": source_provenance,
                "topology_history": topology_history,
                "execution_schema": execution_schema_for_rollout_scheduler(
                    args.rollout_scheduler
                ),
                "actor_objective_schema": actor_objective_schema(
                    args.delightful_policy_gradient,
                    args.target_policy_optimization,
                ),
                "source_signal_mask_schema": SOURCE_SIGNAL_MASK_SCHEMA,
                "resume_arg_contract_schema": RESUME_ARG_CONTRACT_SCHEMA,
                "replay_numerics_schema": REPLAY_NUMERICS_SCHEMA,
                "prompt_order_schema": PROMPT_ORDER_SCHEMA,
                "answer_fence_prompt_schema": (
                    ANSWER_FENCE_PROMPT_SCHEMA if args.answer_fence else None
                ),
                "math_data_identity": data_identity,
                "rl_mixture_manifest": mixture_manifest,
                "rl_source_ids": source_ids,
                "python_reward_schema": (
                    PYTHON_REWARD_SCHEMA
                    if mixture_manifest is not None
                    and any(
                        entry["verifier"] == "python_mbpp"
                        for entry in mixture_manifest["sources"]
                    )
                    else None
                ),
                "reward_schema": REWARD_SCHEMA,
                "reasoning_mode": args.reasoning_mode,
                "context_tokens": context_tokens,
                "math_modal_answer_baseline": math_modal_baseline,
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "rollout_policy_schema": rollout_policy_schema,
                "thought_input_schema": wrapper.thought_input_schema,
                "optimizer_schema": optimizer_schema_for_trunk_optimizer(
                    args.trunk_optimizer
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
                    "architecture": backbone.architecture,
                    "value_bins": args.value_bins,
                    "value_anchored_support": args.value_anchored_support,
                    "value_margin_bins": args.value_margin_bins,
                    "value_num_bins_total": value_num_bins,
                    "value_v_min": value_v_min,
                    "value_v_max": value_v_max,
                    "value_sigma_ratio": args.value_sigma_ratio,
                    "value_prior": args.value_prior,
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

    def finish_group(
        batch: LatentRolloutBatch,
        row: dict,
        refresh_statistics: bool,
    ) -> LatentRolloutBatch:
        batch = trim_stream(batch, multiple=trim_multiple)
        verifier_kind = row.get("_verifier_kind", "math")
        if verifier_kind == "math":
            score_math_rollout(
                batch, row["reward_model"]["ground_truth"], tokenizer, stop_ids,
                answer_style(row),
                args.nearby_reward_max,
                solution_prefix_ids=answer_prefix_ids,
                think_fence_ids=think_fence_ids,
                min_think_tokens=args.think_min_tokens,
                answer_fence_ids=answer_fence_ids,
            )
        elif verifier_kind == "python_mbpp":
            assert think_fence_ids is not None and answer_fence_ids is not None
            score_python_rollout(
                batch,
                row["verification_info"],
                tokenizer,
                stop_ids,
                think_fence_ids,
                answer_fence_ids,
                args.think_min_tokens,
            )
        else:
            raise ValueError(f"unsupported verifier kind {verifier_kind!r}")
        source_name = str(row.get("_rl_source", "math"))
        batch.source_id = torch.full(
            (batch.kind.size(0),),
            source_ids[source_name],
            dtype=torch.long,
            device=batch.kind.device,
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
    ) -> list[LatentRolloutBatch]:
        """One rollout: a scored prompt group per sampled DAPO problem."""
        nonlocal last_decode_schedule_metrics, rollout_paged_cache
        last_decode_schedule_metrics = None
        pool_start_cursor = sampler.cursor
        if (
            mixture_source_quotas is not None
            and prompt_count == args.prompts_per_rollout
        ):
            assert isinstance(sampler, MixedPromptSampler)
            sampler.validate_next_source_quotas(mixture_source_quotas)
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
                        caches=(
                            rollout_graph_caches
                            if rollout_step_core is not None
                            else None
                        ),
                        tail_caches=rollout_tail_caches,
                        tail_step_core=rollout_tail_step_core,
                        decode_mask=decode_mask_for(len(encoded)),
                        tail_decode_mask=rollout_tail_decode_mask,
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
            encoded_rows[start : start + args.rollout_groups]
            for start in range(0, len(encoded_rows), args.rollout_groups)
        ]

        def upload_chunk(
            encoded_chunk: list[tuple[dict, list[int]]],
            width: int,
        ) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
            chunk = [row for row, _ in encoded_chunk]
            encoded = [ids for _, ids in encoded_chunk]
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

        for encoded_chunk in encoded_chunks:
            with profiler.phase("prompt_upload"):
                chunk_width = max(len(ids) for _, ids in encoded_chunk)
                chunk, prompt_ids, prompt_lengths_cpu = upload_chunk(
                    encoded_chunk, chunk_width
                )
                prompt_lengths = prompt_lengths_cpu.to(device)
            with profiler.phase("decode"):
                batched = rollout_continuations(
                    wrapper, prompt_ids, train_max_new_tokens, max_stream_steps,
                    args.temperature, args.top_p, stop_ids=stop_ids or None,
                    prompt_lengths=prompt_lengths,
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
                    caches=(
                        rollout_graph_caches
                        if rollout_step_core is not None
                        else None
                    ),
                    tail_caches=rollout_tail_caches,
                    tail_step_core=rollout_tail_step_core,
                    decode_mask=decode_mask_for(chunk_width),
                    tail_decode_mask=rollout_tail_decode_mask,
                )
            complete_chunk(batched, prompt_lengths_cpu, chunk)
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
                )
        finally:
            if scoring_pool is not None:
                scoring_pool.shutdown(wait=True)
            wrapper.step_core = original_step_core
            wrapper.paged_step_core = original_paged_step_core

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
            answer_fence_ids=answer_fence_ids,
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
            "aime/policy_accuracy", metrics["policy_accuracy"], step
        )
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
            answer_fence_ids=answer_fence_ids,
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
            "bench/policy_accuracy", metrics["policy_accuracy"], step
        )
        print(
            f"step:{step} "
            f"bench_policy_avg@{metrics['policy_samples']}:"
            f"{metrics['policy_accuracy']:.4f}",
            flush=True,
        )

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
        all_passed = True
        for repeat in range(args.rollout_only_repeats):
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            groups = collect(
                refresh_statistics=False,
                offload_to_cpu=True,
            )
            torch.cuda.synchronize()
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
            )
            if last_decode_schedule_metrics is None:
                metrics.update(
                    lockstep_decode_metrics(
                        groups, max(args.rollout_groups, 1)
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
                advantage_mean=_weighted_metric_mean(
                    group_metrics, "advantage_mean", "action_count"
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
                "value_warmup/residual/action_mean": metrics[
                    "advantage_mean"
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
    while step < args.steps:
        if (
            args.zero_reward_stop_pools > 0
            and zero_reward_pool_streak >= args.zero_reward_stop_pools
        ):
            # A sustained all-zero-reward streak is a dead run: with the
            # actor frozen it cannot recover, and with the freeze disabled
            # it is re-entering the round-4 spiral. Pool-boundary stop —
            # the loop tail below saves the final checkpoint, so the run
            # truncates resumably instead of burning the remaining budget
            # on signal-free rollouts.
            print(
                f"{zero_reward_pool_streak} consecutive all-zero-reward "
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
        groups = collect(
            refresh_statistics=False,
            prompt_count=pool_prompt_count,
            offload_to_cpu=True,
        )
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
            )
            if last_decode_schedule_metrics is None:
                rollout_metrics.update(
                    lockstep_decode_metrics(groups, max(args.rollout_groups, 1))
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
                )
                if mixture_manifest is not None
                else None
            )
            if (
                mixture_source_quotas is not None
                and len(groups) == sum(mixture_source_quotas)
            ):
                minibatch_orders = stratified_optimizer_minibatch_orders(
                    groups,
                    args.prompts_per_minibatch,
                    mixture_source_quotas,
                )
            else:
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
        # Rewards are non-negative (exact 1.0, nearby-numeric partial, else
        # 0), so a zero reward mean means every trajectory in the window
        # scored zero. The actual freeze decision is made per optimizer
        # MINIBATCH below (an update over an all-zero minibatch is the
        # harmful unit — a single rewarded trajectory elsewhere in the pool
        # must not unfreeze it); the pool-level statistic here feeds the
        # rollout log and the consecutive-desert stop. NaN fails closed:
        # a non-finite mean is a broken pool, not a licence to update.
        if not math.isfinite(rollout_metrics["reward_mean"]):
            raise RuntimeError(
                f"non-finite pool reward_mean "
                f"{rollout_metrics['reward_mean']} at step {step}"
            )
        pool_reward_zero = rollout_metrics["reward_mean"] == 0.0
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
                f"steps {step + 1}..{step + pool_updates}: all-zero-reward "
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
                        "think_format_fraction",
                        "actions_per_trajectory",
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
                retained_minibatch = retained_device_batches.pop(
                    behavior_age, None
                )
                device_replay_plan = device_replay_plans.pop(behavior_age)
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
                policy_action_denominator = actor_minibatch_action_denominator(
                    [device_minibatch], [0]
                )
                with profiler.phase("forward_backward"):
                    metrics = training_update(
                        wrapper, critic, device_minibatch, optimizers,
                        actor_step=False,
                        critic_step=False,
                        policy_action_denominator=policy_action_denominator,
                        value_action_denominator=policy_action_denominator,
                        gae_lambda_alpha=args.gae_lambda_alpha,
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
                # An update whose every trajectory scored zero is the
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
                # steps so value predictions catch down to the zero
                # targets. (Rewards are finite here: the pool-level
                # isfinite raise above covers the same tensors.)
                minibatch_actor_frozen = (
                    args.zero_reward_actor_freeze
                    and metrics["reward"] == 0.0
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
                    nonfinite_gradients = {
                        name: actor_dashboard[name]
                        for name in (
                            "grad/trunk", "grad/renderer",
                            "grad/combiner", "grad/critic",
                        )
                        if not math.isfinite(actor_dashboard[name])
                    }
                if nonfinite_gradients:
                    raise RuntimeError(
                        "non-finite gradients before optimizer step: "
                        f"{nonfinite_gradients}"
                    )
                with profiler.phase("optimizer_step"):
                    if not minibatch_actor_frozen:
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
                            "combiner/carry_weight_rms": (
                                wrapper.combiner.carry.weight.detach()
                                .square().mean().sqrt()
                            ),
                            "combiner/type_bias_rms": (
                                wrapper.combiner.type_bias.detach()
                                .square().mean().sqrt()
                            ),
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
                            emit_logprob_function=diagnostic_emit_token_logprobs,
                            emit_log_odds_function=(
                                diagnostic_emit_token_log_odds
                            ),
                            replay_plans=[device_replay_plan],
                            target_policy_optimization=(
                                args.target_policy_optimization
                            ),
                            tpo_eta=args.tpo_eta,
                            gae_lambda_alpha=args.gae_lambda_alpha,
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
                bench_eval(step)
        if crossed_interval(previous_step, step, args.save_every):
            with profiler.phase("checkpoint_save"):
                save_started = time.perf_counter()
                save_checkpoint(
                    output / "latent_vapo_checkpoint.pt", wrapper, critic,
                    optimizers, step, args, sampler, warmup_step,
                    actor_init_provenance, zero_reward_frozen_updates,
                )
                save_seconds = time.perf_counter() - save_started
                logger.log(type="checkpoint", step=step, seconds=save_seconds)
                tensorboard.add_scalar(
                    "perf/checkpoint_seconds", save_seconds, step
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
    save_checkpoint(
        output / "latent_vapo_checkpoint.pt", wrapper, critic,
        optimizers, step, args, sampler, warmup_step,
        actor_init_provenance, zero_reward_frozen_updates,
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

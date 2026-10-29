"""Command-line configuration and argv-level validation for latent VAPO."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

from postraining.benchmark_report import CAPTURE_PROBLEMS, CAPTURE_SAMPLES_PER_PROBLEM
from postraining.core import POSTTRAIN_PROMPT_TOKENS, POSTTRAIN_STREAM_TOKENS
from postraining.latent_eval import EVAL_TEMPERATURE, EVAL_TOP_P
from postraining.mean_actor_critic import MAC_ESTIMATORS, SEQUENCE_ESTIMATORS
from postraining.memory_eval_ablation import MEMORY_EVAL_ABLATIONS
from postraining.latent_thought import (
    DEFAULT_INIT_STOP_THINKING_PROBABILITY,
    DEFAULT_THOUGHT_SIGMA,
)
from postraining.memory_carry import (
    DEFAULT_MEMORY_HEAD_DIM,
    DEFAULT_MEMORY_HEADS,
)
from postraining.reasoning_modes import (
    REASONING_MODES,
    mode_carries_hidden,
    mode_carries_memory,
    mode_pins_emit,
)


# One 8-row eval batch at the 1024-token guard sequence length. The guard is
# a catastrophic-drift canary on a fixed deterministic prefix, and drift is
# paired against the same tokens every eval, so a small preset detects an LM
# collapse at ~1/256 the old 2M-token cost (which was 42% of a run's wall
# time at the pre-v28 cadence). Absolute-BPB comparisons against recorded
# pretraining values (the init-identity gate) must pass an explicit
# --bpb-val-tokens 2097152 to reproduce the reference measurement.
DEFAULT_BPB_GUARD_TOKENS = 8 * 1024
DEFAULT_BPB_EVAL_EVERY = 150
DEFAULT_MATH_EVAL_EVERY = 250
DEFAULT_RL_RESPONSE_TOKENS = 2048


def normalized_abstention_reward(penalty: float) -> float:
    """The abstention threshold on the nonnegative verifier reward scale."""
    return (
        penalty / (1.0 + penalty)
        if penalty <= 1.0
        else 1.0 - 1.0 / (1.0 + penalty)
    )

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
# --fresh-learning-rate under --reasoning-mode memory when unset: a fifth of
# pretraining's from-scratch AdamW rate. The fresh-rate memory drill led the
# shared-AdamW memory drill and 14165 by +0.05 to +0.08 bench accuracy from
# step 20 (NOTES.md 2026-10-21).
MEMORY_FRESH_LEARNING_RATE = 3e-3


def resolve_rollout_defaults(args, *, is_nano: bool, checkpoint_context_tokens: int,
                             trained_context_tokens: int | None) -> int:
    """Resolve response budgets and a runtime context that can contain them.

    Runtime capacity is distinct from the checkpoint's trained window. Growing
    the former does not rewrite training metadata or claim context training.
    Explicit frozen-evaluation overrides remain hard limits checked downstream.
    """
    if args.prompt_tokens is None:
        args.prompt_tokens = 512 if is_nano else POSTTRAIN_PROMPT_TOKENS
    if args.continuation_tokens is None:
        args.continuation_tokens = DEFAULT_RL_RESPONSE_TOKENS
    if args.bench_max_tokens is None:
        args.bench_max_tokens = args.continuation_tokens
    if args.aime_max_tokens is None:
        args.aime_max_tokens = args.continuation_tokens
    if args.eval_context_tokens is not None:
        if not is_nano:
            raise ValueError("--eval-context-tokens is only supported for nano backbones")
        return args.eval_context_tokens
    context = max(checkpoint_context_tokens, trained_context_tokens or 0)
    response = (args.answer_tokens if args.reasoning_mode == "none" else
                max(args.continuation_tokens, args.bench_max_tokens, args.aime_max_tokens))
    required_stream = response + (args.reasoning_mode == "latent")
    if context < args.prompt_tokens + required_stream:
        # Preserve room for latent actions rather than reducing a latent run
        # to its mandatory first thought. CoT/carry need only emitted slots.
        if args.reasoning_mode == "latent":
            required_stream = max(required_stream, POSTTRAIN_STREAM_TOKENS)
        context = args.prompt_tokens + required_stream
    return context


def build_arg_parser() -> argparse.ArgumentParser:
    """The CLI surface, separate from main() so the shipped defaults
    and the argv-level guards are testable without running training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    # 950 joint updates + the default 50 critic warmup updates = 1000 total.
    parser.add_argument("--steps", type=int, default=950,
                        help="joint updates, excluding critic warmup (default: 950 + 50 warmup)")
    parser.add_argument(
        "--max-train-hours", type=float, default=None,
        help="stop training at the first pool boundary after this many hours "
        "of process wall time and save the final checkpoint; --steps stays "
        "the step ceiling (learning rates are constant, so an early stop is "
        "a truncation, not a schedule change)",
    )
    parser.add_argument("--math-data", default="postraining/data/dapo-math-17k.parquet")
    parser.add_argument(
        "--rl-mixture-manifest",
        default="postraining/data/vapo_broad_v12_exact.manifest.json",
        help="immutable multi-source verifier manifest, served exact-pass "
        "(every row once per cycle), that replaces --math-data (pass an "
        "empty string for a single --math-data source)",
    )
    parser.add_argument(
        "--exclude-rl-sources", default="",
        help="comma-separated mixture source names to skip while preserving the "
        "original stream cursor and retained row order; resume may only add exclusions",
    )
    parser.add_argument(
        "--randomize-choice-options", action=argparse.BooleanOptionalAction,
        default=None,
        help="fresh deterministic option order for screened MC rows on each "
        "mixture presentation (default on in training, off in rollout-only "
        "gates to preserve frozen prompt comparisons)",
    )
    parser.add_argument(
        "--exclude-modules", default="",
        help="comma-separated extra_info.module names to drop from --math-data "
        "(module-tagged datasets only); names absent from the data are an error",
    )
    # The reasoning mode fixes the rollout policy family for the whole run:
    # "latent" takes one mandatory raw Gaussian THINK action, then samples a
    # one-way continuation gate until STOP_AND_EMIT; "cot" and "none" are
    # token-only controls that bypass the entire gate/noise/thought path;
    # "carry" is the deterministic hidden carry: a token-only policy whose
    # generated-token inputs add, through a zero-init combiner, the detached
    # belief that produced each token; "memory" is the gated-delta memory
    # carry: the same token-only policy whose carried beliefs (prompt beliefs
    # included) are written into a bounded per-trajectory memory that each
    # generated token's input reads (postraining/memory_carry.py).
    parser.add_argument(
        "--reasoning-mode",
        choices=REASONING_MODES,
        default="latent",
    )
    # none-mode emission budget: the final answer value plus its terminator.
    parser.add_argument("--answer-tokens", type=int, default=24)
    # Prompt budget: DAPO prompts longer than this keep their TAIL (the
    # question and answer-format instruction sit at the end). None derives a
    # backbone default after the checkpoint loads: fresh PoPE 1024, nano 512.
    parser.add_argument("--prompt-tokens", type=int, default=None)
    parser.add_argument(
        "--python-reward-mode", choices=("binary", "test-fraction"),
        default="test-fraction",
        help="code reward: fraction of test cases passed, or all-tests-pass binary",
    )
    parser.add_argument("--continuation-tokens", type=int, default=None,
                        help="generated response tokens (default 2048); benchmark and AIME budgets inherit this unless overridden")
    parser.add_argument(
        "--eval-context-tokens", type=int, default=None,
        help="Explicit nano context extrapolation for frozen rollout/benchmark evaluation only; does not change checkpoint training metadata.",
    )
    # Default on-policy topology: collect one fresh 24-prompt x 16-sample batch from
    # the current policy and consume it in exactly one update. The equality of
    # rollout and minibatch prompt counts is an objective constraint for the
    # on-policy estimator, not merely a throughput setting. The opt-out VAPO
    # control can still request a wider frozen pool explicitly.
    parser.add_argument("--prompts-per-rollout", type=int, default=24)
    parser.add_argument("--prompts-per-minibatch", type=int, default=24)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    # One pass only: repeated PPO epochs reuse the same generated trajectories.
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument(
        "--delightful-policy-gradient",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="replace the clipped VAPO/PPO actor objective with Delightful "
        "Policy Gradient (Osband, 2026), using eta=1 current-token "
        "surprisal and no importance ratios",
    )
    parser.add_argument(
        "--mean-actor-critic",
        action="store_true",
        help="replace the actor objective with all-action Mean Actor-Critic: "
        "at every generated state, the policy expectation of the token Q "
        "critic's detached policy-centered Q over the full vocabulary",
    )
    parser.add_argument(
        "--mac-estimator",
        choices=MAC_ESTIMATORS,
        default=None,
        help="the --mean-actor-critic estimator (required with it): all_action "
        "MAC; control_variate, MAC plus the sampled token's Monte Carlo "
        "residual G - Q(s,a_t), unbiased for any critic; sampled_baseline, "
        "that sampled term alone with G - E_pi Q; delightful_sampled_baseline, "
        "that term under the DG gate sigmoid((G - E_pi Q) * surprisal), eta=1; "
        "delightful_td_gate, that term under the DG gate of the critic's "
        "TD advantage sigmoid((r_t + c(s_t+1) - c(s_t)) * surprisal); or "
        "delightful_head_gate, under the DG gate of the action head's advantage "
        "of the sampled token sigmoid(A(a_t) * surprisal); or, with no critic "
        "in the coefficient, one per trajectory: rloo_sequence, the "
        "leave-one-out group advantage U = R - mean of the group's other "
        "rewards; delightful_answer_gate, U under the answer-level DG gate "
        "sigmoid(U * -log(n_k / G)), n_k the group's count of the trajectory's "
        "graded-answer cluster (math sources with --answer-fence only)",
    )
    parser.add_argument(
        "--mac-vocab-chunk",
        type=int,
        default=512,
        help="states per full-vocabulary MAC readout chunk (memory bound only)",
    )
    parser.add_argument(
        "--target-policy-optimization",
        action="store_true",
        help="replace the actor estimator with intra-trajectory token TPO: "
        "raw critic GAE shifts executed-token-versus-rest odds from the "
        "rollout policy, with no PG auxiliary",
    )
    parser.add_argument(
        "--source-success-actor-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="zero every actor advantage from a mixture source unless that "
        "source has at least one positive-reward trajectory in the optimizer "
        "minibatch; disabled by default because this custom mask can turn a "
        "nominally broad mixture into training on only its easiest sources",
    )
    parser.add_argument(
        "--tpo-eta",
        type=float,
        default=2.0,
        help="old-policy target temperature applied directly to raw GAE",
    )
    parser.add_argument(
        "--allow-dg-topology-migration",
        action="store_true",
        help="explicitly allow an exact Delightful resume to change the "
        "equal prompts-per-rollout/prompts-per-minibatch topology; model, "
        "optimizer, RNG, step, and sampler state are still restored",
    )
    parser.add_argument(
        "--allow-rollout-groups-migration",
        action="store_true",
        help="explicitly allow a resume of a counter-keyed lockstep "
        "cot/none/carry run to change --rollout-groups; every draw is keyed "
        "on its own trajectory, so chunking changes floating-point rounding "
        "but not the sampling law, and the resume is no longer bit-exact",
    )
    parser.add_argument(
        "--allow-token-rng-migration",
        action="store_true",
        help="explicitly allow a resume to continue a lockstep cot/none run "
        "saved under the v31 padded-slot token draw with the v32 unpadded-slot "
        "draw; model, optimizer, RNG, step, and sampler state are still "
        "restored, and later draws differ from what v31 would have drawn",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    # One general rate for actor and critic. A branch starts every
    # actor-side optimizer state empty, even when it carries the critic's;
    # using the critic's 3e-4 rate also removes the prior hand-tuned split
    # between trunk, renderer, and combiner.
    # 2e-5 is the lower-drift rate selected after round 5: 5e-5 drove the
    # weak SFT policy from ~120 to ~47 think tokens within 100 actor updates
    # and ultimately concentrated it on a cross-prompt response template.
    # The old 3e-4 default (a pretraining-scale rate) was still more unstable.
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    # Constant AdamW rate for the critic's non-Muon parameters. Keeping this
    # independently selectable avoids turning an actor-rate ablation into an
    # accidental critic-warmup ablation. None preserves the shared-rate
    # behavior used by existing runs.
    parser.add_argument("--critic-learning-rate", type=float, default=None)
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
    # Derived from --critic-learning-rate with the same Muon:AdamW geometry
    # as the actor. The critic trunk is from scratch, so its constant rate can
    # be ablated independently without changing the actor.
    parser.add_argument("--critic-muon-learning-rate", type=float, default=None)
    # Fresh modules -- every parameter absent from the pretrained checkpoint:
    # the actor's combiner/transition/gate and the critic's input combiner
    # (the memory reader under --reasoning-mode memory) -- train from
    # scratch, not as a fine-tune. Set, they get their own optimizers in
    # pretraining's partition: matrices (ndim >= 2) under Muon at
    # --fresh-muon-learning-rate, the rest under AdamW at this rate, for
    # both roles. Unset, --reasoning-mode memory uses
    # MEMORY_FRESH_LEARNING_RATE; other modes share their role's AdamW at
    # --learning-rate / --critic-learning-rate, which a zero-initialized
    # module cannot leave in a few hundred steps (the memory drill's writer
    # moved 3% of its init by step 340).
    parser.add_argument("--fresh-learning-rate", type=float, default=None)
    # Default: --fresh-learning-rate through the same Muon:AdamW geometry
    # as --muon-learning-rate.
    parser.add_argument("--fresh-muon-learning-rate", type=float, default=None)
    # The critic trunk's starting weights. "scratch" is a random instance of
    # the actor's architecture; "actor" is a storage-independent copy of the
    # actor's weights at run start (VAPO initializes its critic from a
    # pretrained LM-based reward model, never randomly). Either way the value
    # head is zero-initialized and nothing is shared with the actor.
    parser.add_argument(
        "--critic-init", choices=("scratch", "actor"), default="scratch"
    )
    # Trunk-optimizer migration: resume model/critic/step/prompt-cursor from
    # a checkpoint whose optimizer layout differs, starting all optimizer
    # state empty instead of loading it.
    parser.add_argument(
        "--reset-optimizers-on-resume", action="store_true"
    )
    # Latent policy noise is specified at vector scale. At runtime width d,
    # components use std = sigma/sqrt(d), hence E||noise||² = sigma².
    # Both stochastic-policy arguments default to None so an explicit value
    # outside latent mode can be refused; validate_args fills the recorded
    # 1.0/0.9 priors for every mode that builds the stochastic heads, which
    # keeps their resume contract byte-identical. Carry records None.
    parser.add_argument(
        "--thought-sigma",
        type=float,
        default=None,
        help="expected L2 magnitude scale of isotropic latent-action noise "
        "(sampled only in latent mode; cot and none keep it as a resume field; "
        "carry refuses it; default 1.0)",
    )
    parser.add_argument(
        "--init-stop-thinking-probability",
        type=float,
        default=None,
        help="initial Bernoulli STOP probability after the forced first "
        "thought (sampled only in latent mode; cot and none keep it as a "
        "resume field; carry refuses it; default 0.9)",
    )
    # Combined-embedding geometry used to adapt raw Gaussian thought slots.
    parser.add_argument("--combined-mlp-blocks", type=int, default=1)
    parser.add_argument("--combined-mlp-hidden", type=int, default=2048)
    # Gated-delta memory geometry (--reasoning-mode memory only): heads of
    # head_dim x head_dim fp32 state per trajectory. Defaults 4 x 64.
    parser.add_argument("--memory-heads", type=int, default=None)
    parser.add_argument("--memory-head-dim", type=int, default=None)
    # Length-adaptive GAE lambda (core.length_adaptive_lambda): VAPO's
    # horizon alpha*l with a floor of min(l, 1/alpha).  The raw alpha=0.05
    # formula clamps lambda to 0 at this run's 12-23-action trajectories
    # (TD(0): terminal reward credits nothing more than one step back —
    # audited as the think-fraction ratchet); the floor restores
    # whole-trajectory credit for short answers while keeping VAPO's
    # variance control for long ones.
    parser.add_argument("--gae-lambda-alpha", type=float, default=0.05)
    # A fixed actor lambda replaces the length-adaptive rule for the actor's
    # advantages; critic targets stay lambda-1 returns. Token-level
    # Delightful PG needs each token's advantage to be its own, which over a
    # calibrated critic is lambda 0 (core.actor_gae_lambdas).
    parser.add_argument("--actor-gae-lambda", type=float, default=None)
    parser.add_argument(
        "--critic-action-head-hidden",
        type=int,
        default=None,
        help="width of a dueling Q(s_t, a_t) action head on the critic; the "
        "actor's advantages then bootstrap from Q instead of V(s_{t+1}) "
        "(core.action_value_advantages). Default: no action head",
    )
    parser.add_argument(
        "--critic-q-head-hidden",
        type=int,
        default=None,
        help="hidden width of a standalone token Q critic, REPPO's critic "
        "with no V: Q(s, a) = c + u . x(a) for every token at once. It fits "
        "generalized value estimates (--q-critic-lambda) and the Delightful "
        "actor's advantage is Q centered exactly under the policy, "
        "A_t = Q(s_t, a_t) - sum_a pi(a | s_t) Q(s_t, a). Default: a V critic",
    )
    parser.add_argument(
        "--critic-q-twin", action="store_true",
        help="TD3/TD7 twin critics on the token Q critic: a second, "
        "independently initialized Q head on the same trunk. Both regress "
        "the same generalized value estimates, bootstrapped from the "
        "smaller of their Q of the next taken token (clipped double Q); "
        "the actor reads their mean",
    )
    parser.add_argument(
        "--critic-q-pessimism", type=float, default=0.0,
        help="Pessimistic prior weight beta on every token Q head: beta "
        "times sum_a relu(A(s, a))^2 at each generated state, A the head's "
        "Q centered under the current policy, so a token the policy all "
        "but never takes cannot read better than its value; a positive "
        "advantage shrinks by pi/(pi + beta) at the tabular fixed point "
        "(postraining.pessimistic_q). Applied in MAC actor updates. "
        "0 disables it",
    )
    parser.add_argument(
        "--critic-post-action-q-hidden", type=int, default=None,
        help="opt-in binary success Q from the independent critic's causal "
        "post-action prefix; no state baseline, policy mean, or actor memory",
    )
    parser.add_argument(
        "--critic-post-action-q-prior", type=float, default=0.02,
        help="initial binary-success probability for the post-action Q head; "
        "an explicit initialization choice, not an empirically optimal prior",
    )
    parser.add_argument(
        "--q-critic-lambda",
        type=float,
        default=None,
        help="lambda of a Q critic's generalized value estimates; 1 is the "
        "Monte-Carlo return. Default: VAPO's length-adaptive lambda "
        "(--gae-lambda-alpha). Value warmup always fits lambda one",
    )
    parser.add_argument(
        "--critic-q-direction-learning-rate",
        type=float,
        default=None,
        help="separate AdamW rate for the token Q head's direction readout "
        "u(s); default the head's rate. A --study-critic-init critic "
        "requires its study's rate, head rate over the direction width",
    )
    parser.add_argument(
        "--critic-action-head-learning-rate",
        type=float,
        default=None,
        help="AdamW rate of the action-conditioned head, dueling or "
        "standalone Q; default --critic-learning-rate",
    )
    parser.add_argument(
        "--entropy-target",
        type=float,
        default=None,
        help="REPPO's entropy constraint: a temperature, tuned toward this "
        "mean policy entropy (nats per emitted token), weights an exact "
        "entropy bonus in the Delightful objective. It only raises entropy "
        "below the target; above it the temperature decays toward zero. "
        "Default: no entropy term",
    )
    parser.add_argument(
        "--entropy-temperature-init",
        type=float,
        default=None,
        help="initial entropy temperature; required with --entropy-target. "
        "It trades entropy against Q, so it is set relative to Q's scale: "
        "REPPO's ent_start 0.01 against MuJoCo Q of order 100 is a ratio of "
        "about 1e-4",
    )
    parser.add_argument(
        "--entropy-temperature-learning-rate",
        type=float,
        default=None,
        help="Adam rate of the log temperature: about the largest change in "
        "log temperature per actor step; default 0.05",
    )
    parser.add_argument(
        "--actor-kl-bound",
        type=float,
        default=None,
        help="per-step trust region: the mean token KL(old || new) of each "
        "actor step, read in fp32 on its own minibatch before and after the "
        "step, is held at or below this bound by backtracking the step along "
        "its own direction. bf16 rounding alone puts 2-3e-4 between any two "
        "nearby policies, so only the fp32 readout resolves bounds near it. "
        "Default: unbounded",
    )
    parser.add_argument(
        "--actor-kl-backtracks",
        type=int,
        default=None,
        help="most backtracking passes per step under --actor-kl-bound; "
        "default 3",
    )
    parser.add_argument(
        "--actor-kl-shard-stride",
        type=int,
        default=None,
        help="under --actor-kl-bound, measure the step's KL on every Nth "
        "planned replay shard of its minibatch (shards are length-sorted, "
        "so this samples across lengths); default 4",
    )
    parser.add_argument(
        "--nearby-reward-max",
        type=float,
        default=0.0,
        help="maximum reward for a wrong, terminated numeric final answer",
    )
    # EOS after a valid think with no answer content earns penalty / (1 + penalty).
    # Verifier scores remain in [0, 1], so answering pays only above that
    # confidence. This is an affine normalization of signed [−penalty, 1]
    # rewards that puts wrong answers at the critic's initial zero baseline.
    parser.add_argument(
        "--abstention-penalty",
        type=float,
        default=None,
        help="enable abstention: EOS after a valid think with no answer scores "
        "PENALTY/(1+PENALTY), while wrong or malformed answers score 0",
    )
    # Requires a base checkpoint whose SFT stage trained the fence rows
    # (sft_trace_train --think-tokens); enforced at startup. Gating ALL
    # reward on a completed <think>...</think> makes the bare-guess policy
    # (the 1024/1025 collapse attractor) worth zero even when correct.
    parser.add_argument(
        "--think-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="decode <think>/</think> as special tokens and gate math "
        "reward on a well-formed, closed think fence",
    )
    # Round 3 (sft2_rl_think_gsm8k_4k) showed the 1-token floor collapses
    # to a ~15-token minimal compliant skeleton: the fence survives but
    # carries no compute. The floor mandates a sequential-compute budget
    # inside the fence; reward is zero below it.
    parser.add_argument(
        "--think-min-tokens",
        type=int,
        default=None,
        help="minimum token count inside the think fence for any reward "
        "(default: 33 with --think-tokens, 1 otherwise)",
    )
    # Requires a base checkpoint whose SFT stage trained the answer fence
    # rows (sft_trace_train --answer-fence); enforced at startup. With the
    # answer a token-delimited span, the reward gate and extraction are
    # purely structural on token ids — the decoded-text regex position
    # check (and its case/spacing/boundary bypass class) is retired.
    parser.add_argument(
        "--answer-fence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="decode <answer>/</answer> as special tokens; gate reward on "
        "<think>...</think><answer>...</answer> structure and grade only "
        "the fenced answer span (requires --think-tokens)",
    )
    # Round-4 postmortem (NOTES.md): over an all-zero-reward pool a stale
    # critic leaves value predictions slightly positive, so every advantage
    # is uniformly negative and the only coherent policy gradient is
    # anti-termination — expected length ~1/p_stop explodes into an
    # absorbing zero-reward desert. Such a pool carries no policy signal
    # (any nonzero advantage in it is pure critic error), so the actor
    # optimizer is skipped for its updates while the critic keeps stepping
    # toward the zero targets that end the spiral.
    parser.add_argument(
        "--zero-reward-actor-freeze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip actor optimizer steps for optimizer minibatches whose "
        "every trajectory earned the same reward (including all wrong or "
        "all abstaining); the critic still updates toward those targets",
    )
    # The actor freeze is the actual safety intervention. A stop based only on
    # consecutive zero-reward prompt windows is unsound for a heterogeneous
    # sparse-reward mixture: the frozen actor is unchanged, but later prompts
    # can be easier. Keep the optional operational circuit breaker available
    # without making it part of the selected production regime.
    parser.add_argument(
        "--bench-stop-floor",
        type=float,
        default=None,
        help="stop at the pool boundary (saving the final checkpoint) once "
        "--bench-stop-evals consecutive periodic benchmark evaluations score "
        "contract accuracy below this floor; default disabled",
    )
    parser.add_argument(
        "--bench-stop-evals",
        type=int,
        default=2,
        help="consecutive below-floor benchmark evaluations for --bench-stop-floor",
    )
    parser.add_argument(
        "--zero-reward-stop-pools",
        type=int,
        default=0,
        help="stop at the pool boundary (saving the final checkpoint) "
        "after this many consecutive constant-reward rollout pools; "
        "applies regardless of --zero-reward-actor-freeze; 0 disables",
    )
    # VAPO paper: 50 value-pretraining steps before policy updates.
    parser.add_argument(
        "--value-warmup-steps",
        type=int,
        default=None,
        help="critic-only updates this run makes before its first actor "
        "step; default 50, or 0 when the run loads a trained critic "
        "(--actor-init without --fresh-critic, --critic-only-init). A "
        "carried critic whose value contract changed must refit with a "
        "positive count, or be carried as-is with "
        "--accept-critic-value-change",
    )
    parser.add_argument(
        "--value-warmup-scope",
        choices=("critic", "action-head"),
        default="critic",
        help="what --value-warmup-steps fit: the whole critic, or only a "
        "fresh action head on a carried critic whose values are fit; an "
        "action-head warmup then rewinds the prompt cursor, so the first "
        "actor step matches a run that starts acting at the same checkpoint",
    )
    parser.add_argument(
        "--value-warmup-rewind",
        action="store_true",
        help="rewind the prompt cursor after a whole-critic warmup too, so a "
        "fresh critic warmed from an --actor-init checkpoint leaves the first "
        "actor step on the prompts a run acting at that checkpoint draws",
    )
    # The BPB guard is a do-no-harm regression check, not an optimization
    # target. A deterministic 2M-token prefix takes ~2.5s on the 5090 versus
    # ~75s for all 62M validation tokens, while still giving the guard ample
    # precision to detect renderer drift. Use 0 explicitly for a final,
    # challenge-comparable full-validation measurement.
    parser.add_argument(
        "--bpb-every", type=int, default=DEFAULT_BPB_EVAL_EVERY
    )
    parser.add_argument(
        "--bpb-val-tokens", type=int, default=DEFAULT_BPB_GUARD_TOKENS,
        help="deterministic validation-prefix size for the BPB guard "
        f"(default: {DEFAULT_BPB_GUARD_TOKENS} = one eval batch; 0 = full "
        "set); guard values remain comparable only within one setting — "
        "identity checks against recorded pretraining val_bpb need 2097152",
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
    parser.add_argument(
        "--memory-eval-ablation", choices=MEMORY_EVAL_ABLATIONS, default="none",
        help="--bench-only memory-carry ablation: zero_read zeroes the "
        "actor's memory read-out; foreign_state (foreign_sample) makes every "
        "read see another prompt's (another sample's) memory state at the "
        "same tick (postraining.memory_eval_ablation)",
    )
    parser.add_argument(
        "--bench-capture-problems", type=int, default=CAPTURE_PROBLEMS,
        help="leading benchmark problems whose answers are saved",
    )
    parser.add_argument(
        "--bench-capture-samples", type=int, default=CAPTURE_SAMPLES_PER_PROBLEM,
        help="leading samples saved per captured benchmark problem",
    )
    # Keep the hard and easy held-out math evaluations on the same cadence so
    # their learning curves remain directly aligned.
    parser.add_argument(
        "--aime-every", type=int, default=DEFAULT_MATH_EVAL_EVERY
    )
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
        "--bench-every", type=int, default=DEFAULT_MATH_EVAL_EVERY
    )
    parser.add_argument("--bench-samples", type=int, default=8)
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=EVAL_TEMPERATURE,
        help="sampling temperature of the periodic benchmark and AIME evaluations",
    )
    parser.add_argument(
        "--eval-top-p",
        type=float,
        default=EVAL_TOP_P,
        help="nucleus of the periodic benchmark and AIME evaluations; 1 "
        "evaluates the untruncated policy",
    )
    # Training-rollout transcripts (postraining/rollout_report.py): every N
    # actor steps, the first correct and first incorrect trajectory of each
    # RL source in that pool. Display only, so not a resume invariant; 0
    # disables capture.
    parser.add_argument("--rollout-sample-every", type=int, default=25)
    parser.add_argument(
        "--bench-max-rows",
        type=int,
        default=0,
        help="fixed hash-selected benchmark prompt count (0 evaluates all)",
    )
    parser.add_argument("--bench-max-tokens", type=int, default=None)
    # Batch multiple problem groups into the same left-padded GPU rollout.
    # This is a trajectory rather than prompt count so avg@8 and avg@32 use
    # comparable memory. Under pinned EMIT the panels decode through a graph
    # arena of this many rows whose workspace exists only during an eval
    # rollout, so a whole 960-sample AIME panel is one rollout: its ~2k-step
    # tail is paid once instead of once per 128-row chunk.
    parser.add_argument("--eval-batch-trajectories", type=int, default=1024)
    parser.add_argument(
        "--eval-tail-batch", type=int, default=16,
        help="single compiled survivor-batch size (0 disables compaction)",
    )
    parser.add_argument(
        "--rollout-tail-batch", type=int, default=16,
        help="single compiled training survivor-batch size "
        "(0 disables compaction); latent lockstep only",
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
    # Give the compiled lockstep decode step a flex-decoding block table
    # instead of a boolean attn_mask. Handing SDPA any mask disqualifies its
    # fused backends and lands the step on the memory-efficient kernel, which
    # the v25 profile measured at 32% of pool device time with 98.5% of those
    # calls coming from the MAIN loop, not the tail (NOTES.md:1285).
    #
    # Reaching that main loop is the whole difficulty. Flex decoding lowers
    # for fully static shapes alone -- measured, and a dynamic batch by itself
    # is enough to make the lowering fail -- while the survivor count moves
    # with compaction. The flag therefore also switches compaction to round UP
    # to a DecodeRangeMask.DEFAULT_ROW_BUCKET multiple, which is affordable
    # only because a block table can give the surplus rows an empty range:
    # they read no KV and come back exactly zero. They do still occupy KV
    # cache, so the grid is linear rather than powers of two -- see
    # DecodeRangeMask. Microbenchmarked at the production shape (6 layers, 4x128,
    # 2560-key cache), bucketed flex against the dynamic SDPA step it
    # replaces: 0.72x at position 128 falling to 0.29x at 2047, and within
    # 0-19% of an exact-count flex batch throughout.
    #
    # Off until the end-to-end A/B confirms it.
    parser.add_argument(
        "--rollout-flex-decode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    # Capture the main decode loop as a CUDA graph instead of launching it.
    # The v34 kernel profile measured ~348 host launches per decode step for a
    # six-layer model -- 157k of them on kernels averaging 1.9 us, at or below
    # launch cost -- and 32% of pool wall with no kernel resident at all. A
    # replay issues one launch for the whole step.
    #
    # The price is a fixed batch and a fixed KV width: capture pins both, so
    # the loop stops compacting finished rows away and stops sizing its cache
    # to the chunk. Dead rows keep costing their share of the step's GEMMs
    # while costing no attention (the empty range is what makes bucket padding
    # free), so this trades device work for launch overhead and only pays if
    # the launch bubble is the larger of the two. Off until the A/B says so.
    # Two capture targets share the flag: with the lockstep scheduler it
    # rides --rollout-flex-decode's static arena; with continuous_refill it
    # captures the paged step per declared execution-width bucket (full
    # capacity + power-of-two tails), all recorded up front by
    # warmup_decode_width_buckets before the first real token.
    parser.add_argument(
        "--rollout-graph-decode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--checkpoint-interval-seconds",
        type=float,
        default=480.0,
        help="elapsed wall time between rolling exact-recovery checkpoints",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help=(
            "also keep an exact checkpoint every N steps under snapshots/, "
            "for analysing the policy at fixed points of a run (0 = off)"
        ),
    )
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
    # than a fixed row count, which automatically isolates rare long
    # outliers into small shards. The v28 hidden-carry streams are 1x
    # (prompt + response, ~1.2K slots worst case) where the stochastic
    # design ran 4x, so the ceilings below are sized for that regime: the
    # 40-step profiled smoke measured ~27 rows/shard against the old
    # 32/4M/8192 ceilings with a 5.6 GiB train peak on a 32 GiB card —
    # pure accumulation overhead with no memory pressure to justify it.
    parser.add_argument("--replay-bucket", type=int, default=64)
    parser.add_argument("--replay-max-trajectories", type=int, default=128)
    parser.add_argument(
        "--replay-attention-budget", type=int, default=16 * 1024 * 1024
    )
    # Bounds the LINEAR per-shard memory term: trunk activations plus the
    # fused readout's retained bf16 emit logits (2 bytes/element). Measured
    # on KDA8 (NOTES 2026-09-25): the update peaks at ~7.1 GiB here and
    # ~18 GiB at 3x with the attention budget lifted, with no update-time
    # gain from the fewer, larger shards. Without this, raising
    # --replay-attention-budget lets short-L shards grow their slot count
    # unboundedly.
    parser.add_argument("--replay-slot-budget", type=int, default=24576)
    # A chunk decodes at its longest row's length, so rows that finished
    # early keep stepping until the batch is narrowed. Compaction can only
    # fire on a sync boundary, which makes these two the knobs that set how
    # much of the decode is spent on already-finished rows.
    parser.add_argument(
        "--rollout-sync-every", type=int, default=16,
        help="lockstep-only decode steps between live-row checks that gate "
        "compaction; continuous_refill retires and admits every iteration",
    )
    parser.add_argument(
        "--rollout-compact-dead-ratio", type=float, default=0.25,
        help="compact once this fraction of the current rollout width has "
        "finished (lower = compact sooner, more KV cache copies); latent "
        "lockstep only -- the cot/none/carry graph arena compacts whenever "
        "survivors fit a smaller row bucket",
    )
    parser.add_argument(
        "--post-update-kl-every",
        type=int,
        default=100,
        help="read-only same-batch policy-drift replay cadence; step 1 is "
        "always measured, 0 disables later measurements",
    )
    # Prompt groups rolled out together as one left-padded batch. Under the
    # CUDA-graph decode arena a chunk's wall time is set by its longest
    # trajectory: a trained 2048-token-cap policy ends 99.7% of trajectories
    # in ~100 tokens, yet each chunk still runs ~2048 ticks, most of them on
    # 8-16 live rows. One 1024-row chunk (64 groups x 16 samples) pays that
    # tail once instead of twice: collect 2.0 s vs 2.9 s per pool, update
    # unchanged, 13.5 vs 7.6 GiB peak (perf_trained_g64/g32_20260926). The
    # v28 eager-rollout measurement that chose 32 (jobs 969/978/979: width
    # saturated at 32 groups) predates the arena. continuous_refill measured
    # 6.4k actions/s (job 980): 0.92 step utilization cannot buy back its
    # per-step paged overhead at this model scale.
    parser.add_argument("--rollout-groups", type=int, default=64)
    parser.add_argument(
        "--rollout-scheduler",
        choices=("lockstep", "continuous_refill"),
        default="lockstep",
        help="lockstep completes each prompt chunk independently; "
        "continuous_refill recycles completed physical lanes into later "
        "prompt groups using request-stable sampling and paged KV attention",
    )
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument(
        "--gate-transcripts", action="store_true",
        help="save every frozen-policy gate response and verifier verdict as JSONL",
    )
    parser.add_argument(
        "--rollout-dataset", action="store_true",
        help="record every frozen-policy trajectory as a hash-bound "
        "rollout dataset (one parquet per pool) for offline critic studies",
    )
    parser.add_argument(
        "--rollout-only-repeats",
        type=int,
        default=1,
        help="production-shape rollout pools to collect in one process; "
        "repeat zero includes compile/cold-start cost and later repeats "
        "measure steady state",
    )
    parser.add_argument(
        "--perturbation-sweep", nargs="+", default=None,
        choices=("writer", "reader_out"),
        help="--rollout-only memory policies: per pool, re-collect the same "
        "prompts and token noise with these parameter sets moved by +/- each "
        "--perturbation-scales along each direction "
        "(postraining.perturbation_sweep); --rollout-only-repeats is the pool count",
    )
    parser.add_argument(
        "--perturbation-scales", nargs="+", type=float, default=None,
        help="per-element move relative to each tensor's RMS",
    )
    parser.add_argument(
        "--perturbation-random-directions", type=int, default=0,
        help="fresh Gaussian directions per target and pool",
    )
    parser.add_argument(
        "--perturbation-gradient-direction", action="store_true",
        help="also test the token-path gradient direction, the mean of the "
        "warmup pools' estimates; needs the --curriculum-init snapshot's "
        "critic as baseline",
    )
    parser.add_argument(
        "--perturbation-gradient-warmup-pools", type=int, default=0,
        help="leading pools that only estimate the token-path gradient for "
        "--perturbation-gradient-direction; counted in --rollout-only-repeats",
    )
    parser.add_argument(
        "--perturbation-radial-direction", action="store_true",
        help="also test theta -> (1 +/- scale) theta",
    )
    parser.add_argument(
        "--write-noise-sweep", nargs="+", type=float, default=None,
        metavar="SCALE",
        help="--rollout-only memory policies: per pool, re-collect the same "
        "prompts and token noise with the memory writes explored at +/- each "
        "absolute pre-activation noise scale, one noise pattern per prompt "
        "(postraining.perturbation_sweep.run_write_noise_sweep); "
        "--rollout-only-repeats is the pool count",
    )
    parser.add_argument("--gate-min-within-group-reward-std", type=float, default=0.01)
    parser.add_argument(
        "--actor-init", default=None,
        help="branch from a latent-VAPO checkpoint: load its actor, carry its "
        "critic and critic optimizer state (unless --fresh-critic), start a "
        "fresh actor optimizer, and continue its unused prompt stream",
    )
    parser.add_argument(
        "--fresh-critic", action="store_true",
        help="with --actor-init, start the critic and its optimizer fresh "
        "(per --critic-init) instead of carrying the checkpoint's",
    )
    parser.add_argument(
        "--accept-critic-value-change", action="store_true",
        help="carry an --actor-init critic whose value contract changed "
        "without refitting it, e.g. when the changed reward never applied "
        "to a row the critic saw; the changed fields and this acceptance "
        "are recorded in the manifest",
    )
    parser.add_argument(
        "--study-critic-init",
        default=None,
        help="initialize the token Q critic (trunk and head) from a "
        "token_q_critic_study_checkpoint/v2 fit against this --checkpoint; "
        "the critic's optimizer starts fresh",
    )
    parser.add_argument(
        "--critic-only-init", default=None,
        help="load only trained critic weights; actor comes from --checkpoint, "
        "with fresh optimizers, RNG, and prompt stream (requires zero warmup)",
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


def validate_perturbation_sweep(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """The antithetic memory sweep's flag contract."""
    options = (
        args.perturbation_scales is not None,
        args.perturbation_random_directions != 0,
        args.perturbation_gradient_direction,
        args.perturbation_radial_direction,
        args.perturbation_gradient_warmup_pools != 0,
    )
    if args.perturbation_sweep is None:
        if any(options):
            parser.error("--perturbation-* options need --perturbation-sweep")
        return
    if not args.rollout_only:
        parser.error("--perturbation-sweep requires --rollout-only")
    if not mode_carries_memory(args.reasoning_mode):
        parser.error("--perturbation-sweep perturbs the memory carry: --reasoning-mode memory")
    if len(set(args.perturbation_sweep)) != len(args.perturbation_sweep):
        parser.error("--perturbation-sweep targets must be distinct")
    if args.gate_transcripts or args.rollout_dataset:
        parser.error(
            "--perturbation-sweep writes its own artifact; perturbed pools must "
            "not enter --gate-transcripts or --rollout-dataset"
        )
    scales = args.perturbation_scales
    if not scales:
        parser.error("--perturbation-sweep needs --perturbation-scales")
    if any(not (math.isfinite(scale) and scale > 0) for scale in scales) or len(set(scales)) != len(scales):
        parser.error("--perturbation-scales must be distinct, finite and positive")
    if args.perturbation_random_directions < 0:
        parser.error("--perturbation-random-directions must be nonnegative")
    if not any(options[1:4]):
        parser.error(
            "--perturbation-sweep needs a direction: --perturbation-random-directions, "
            "--perturbation-gradient-direction or --perturbation-radial-direction"
        )
    if args.perturbation_gradient_direction:
        # The token path measured is the trainer's sampled-baseline estimator,
        # REINFORCE with the snapshot critic's state value: unbiased for any
        # state-only baseline, and the critic only lowers its variance.
        if not (args.mean_actor_critic and args.mac_estimator == "sampled_baseline"):
            parser.error(
                "--perturbation-gradient-direction measures the sampled-baseline "
                "token path: --mean-actor-critic --mac-estimator sampled_baseline"
            )
        if args.curriculum_init is None:
            parser.error(
                "--perturbation-gradient-direction takes its baseline critic from "
                "the --curriculum-init snapshot"
            )
        warmup = args.perturbation_gradient_warmup_pools
        if warmup < 1 or warmup >= args.rollout_only_repeats:
            parser.error(
                "--perturbation-gradient-direction needs --perturbation-gradient-warmup-pools "
                "of at least 1 and below --rollout-only-repeats"
            )
    elif args.perturbation_gradient_warmup_pools:
        parser.error("--perturbation-gradient-warmup-pools needs --perturbation-gradient-direction")


def validate_write_noise_sweep(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """The write-space antithetic sweep's flag contract."""
    scales = args.write_noise_sweep
    if scales is None:
        return
    if args.perturbation_sweep is not None:
        parser.error("--write-noise-sweep and --perturbation-sweep are separate sweeps")
    if not args.rollout_only:
        parser.error("--write-noise-sweep requires --rollout-only")
    if not mode_carries_memory(args.reasoning_mode):
        parser.error("--write-noise-sweep explores the memory carry: --reasoning-mode memory")
    if args.rollout_scheduler != "lockstep":
        parser.error("--write-noise-sweep decodes through the lockstep arena")
    if args.gate_transcripts or args.rollout_dataset:
        parser.error(
            "--write-noise-sweep writes its own artifact; explored pools must "
            "not enter --gate-transcripts or --rollout-dataset"
        )
    if not scales or any(not (math.isfinite(s) and s > 0) for s in scales) or len(set(scales)) != len(scales):
        parser.error("--write-noise-sweep scales must be distinct, finite and positive")
    if args.samples_per_prompt < 2 or args.prompts_per_rollout < 2:
        # The signal estimate pairs distinct samples of one prompt; its
        # control pairs distinct prompts.
        parser.error(
            "--write-noise-sweep needs at least two samples per prompt and "
            "two prompts per pool"
        )


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Argv-level guards that need no checkpoint. Backbone-dependent
    checks stay in main() because they need the loaded model."""
    if args.value_warmup_steps is None:
        # A loaded critic is already fit: warming it again by default would
        # spend 50 pools of prompts refitting values it has. A resume
        # restores its run's count through the exact-resume check, which
        # names this field when the two disagree.
        loads_trained_critic = (
            bool(args.critic_only_init) or bool(args.study_critic_init)
            or bool(args.actor_init and not args.fresh_critic)
        )
        args.value_warmup_steps = 0 if loads_trained_critic else 50
    if not (math.isfinite(args.eval_temperature) and args.eval_temperature > 0):
        parser.error("--eval-temperature must be positive and finite")
    if not 0 < args.eval_top_p <= 1:
        parser.error("--eval-top-p must lie in (0, 1]")
    if args.bench_stop_floor is not None and not (
        0 <= args.bench_stop_floor <= 1 and args.bench_stop_evals >= 1
        and args.bench_data and args.bench_every >= 1
    ):
        parser.error(
            "--bench-stop-floor needs a [0, 1] floor, --bench-stop-evals >= 1 "
            "and a periodic benchmark (--bench-data, --bench-every)"
        )
    if args.critic_q_direction_learning_rate is not None and not (
        args.critic_q_head_hidden is not None
        and math.isfinite(args.critic_q_direction_learning_rate)
        and args.critic_q_direction_learning_rate > 0
    ):
        parser.error("--critic-q-direction-learning-rate needs a token Q critic and a positive rate")
    if args.critic_q_twin and args.critic_q_head_hidden is None:
        parser.error("--critic-q-twin pairs a second head with --critic-q-head-hidden's")
    if not (math.isfinite(args.critic_q_pessimism) and args.critic_q_pessimism >= 0):
        parser.error("--critic-q-pessimism must be finite and nonnegative")
    if args.critic_q_pessimism and (
        args.critic_q_head_hidden is None or args.critic_post_action_q_hidden is not None
    ):
        parser.error("--critic-q-pessimism regularizes the bilinear token Q critic (--critic-q-head-hidden)")
    if args.critic_q_pessimism and not args.mean_actor_critic:
        parser.error("--critic-q-pessimism reads the policy from --mean-actor-critic's readout")
    if args.study_critic_init is not None:
        if args.critic_q_head_hidden is None:
            parser.error("--study-critic-init loads a token Q critic; pass --critic-q-head-hidden")
        if args.resume or args.actor_init or args.curriculum_init or args.critic_only_init or args.fresh_critic:
            parser.error(
                "--study-critic-init starts a run at --checkpoint; it excludes "
                "--resume, --actor-init, --curriculum-init, --critic-only-init "
                "and --fresh-critic"
            )
    if args.mean_actor_critic:
        # Token-only policies whose replay rebuilds every input from stored
        # data: cot, and the memory carry, whose reads replay through one
        # scan over the recorded beliefs.
        if args.reasoning_mode not in ("cot", "memory"):
            parser.error("--mean-actor-critic supports --reasoning-mode cot or memory")
        if args.delightful_policy_gradient or args.target_policy_optimization:
            parser.error("--mean-actor-critic replaces the DG and TPO actor objectives")
        if args.critic_q_head_hidden is None:
            parser.error("--mean-actor-critic needs a token Q critic (--critic-q-head-hidden)")
        if args.source_success_actor_gate or args.zero_reward_actor_freeze:
            parser.error(
                "MAC's objective uses no sampled outcome; pass "
                "--no-source-success-actor-gate --no-zero-reward-actor-freeze"
            )
        if args.entropy_target is not None or args.actor_kl_bound is not None:
            parser.error("--mean-actor-critic has no entropy or KL trust-region term")
        if args.mac_vocab_chunk < 1:
            parser.error("--mac-vocab-chunk must be positive")
        if args.mac_estimator is None:
            parser.error("--mean-actor-critic needs --mac-estimator")
        if args.mac_estimator in SEQUENCE_ESTIMATORS:
            # The group is the baseline and the answer the action: a
            # leave-one-out mean needs a second sample, and the clusters
            # need the verifier's single graded answer span.
            if args.samples_per_prompt < 2:
                parser.error(f"--mac-estimator {args.mac_estimator} needs "
                             "--samples-per-prompt of at least 2")
            if not args.answer_fence:
                parser.error(f"--mac-estimator {args.mac_estimator} clusters the graded "
                             "<answer> span: pass --answer-fence")
            # U is the verifier's grade and clusters key (answer, reward):
            # partial credit for a wrong answer or an abstention payout
            # would move U off the grade and split one graded answer.
            if args.nearby_reward_max != 0 or args.abstention_penalty is not None:
                parser.error(f"--mac-estimator {args.mac_estimator} needs the pure "
                             "verifier reward: drop --nearby-reward-max and "
                             "--abstention-penalty")
    elif args.mac_estimator is not None:
        parser.error("--mac-estimator selects a --mean-actor-critic estimator")
    if args.eval_context_tokens is not None:
        if args.eval_context_tokens < 1:
            parser.error("--eval-context-tokens must be positive")
        if not (args.rollout_only or args.bench_only):
            parser.error("--eval-context-tokens requires --rollout-only or --bench-only")
    # TPO replaces the actor objective outright. Resolve that precedence once
    # so every downstream objective/schema check sees exactly one actor mode,
    # even when --delightful-policy-gradient was also passed.
    if args.target_policy_optimization:
        args.delightful_policy_gradient = False
    if not math.isfinite(args.tpo_eta) or args.tpo_eta <= 0.0:
        parser.error("--tpo-eta must be finite and positive")
    if args.replay_max_trajectories < 1:
        parser.error("--replay-max-trajectories must be positive")
    if args.max_train_hours is not None and args.max_train_hours <= 0:
        parser.error("--max-train-hours must be positive")
    if args.max_train_hours is not None and args.consume_all_prompts:
        parser.error(
            "--max-train-hours truncates the run at a wall-clock deadline "
            "and cannot guarantee --consume-all-prompts' one-pass contract"
        )
    if args.combined_mlp_blocks < 0:
        parser.error("--combined-mlp-blocks must be nonnegative")
    if args.combined_mlp_hidden < 1:
        parser.error("--combined-mlp-hidden must be positive")
    carries_hidden = mode_carries_hidden(args.reasoning_mode)
    if mode_carries_memory(args.reasoning_mode) and (
        args.combined_mlp_hidden != parser.get_default("combined_mlp_hidden")
        or args.combined_mlp_blocks != parser.get_default("combined_mlp_blocks")
    ):
        parser.error(
            "--combined-mlp-hidden/--combined-mlp-blocks configure the "
            "hidden carry's combiner; the memory carry has none"
        )
    if mode_carries_memory(args.reasoning_mode):
        if args.memory_heads is None:
            args.memory_heads = DEFAULT_MEMORY_HEADS
        if args.memory_head_dim is None:
            args.memory_head_dim = DEFAULT_MEMORY_HEAD_DIM
        if args.memory_heads < 1 or args.memory_head_dim < 1:
            parser.error("--memory-heads and --memory-head-dim must be positive")
    else:
        for flag, value in (
            ("--memory-heads", args.memory_heads),
            ("--memory-head-dim", args.memory_head_dim),
        ):
            if value is not None:
                parser.error(f"{flag} configures --reasoning-mode memory only")
    # cot/none build (unused) Gaussian and gate heads whose priors are resume
    # fields, so they keep accepting the flags exactly as before; only the
    # carries, which build no such head, refuse them.
    if carries_hidden:
        for flag, value in (
            ("--thought-sigma", args.thought_sigma),
            (
                "--init-stop-thinking-probability",
                args.init_stop_thinking_probability,
            ),
        ):
            if value is not None:
                parser.error(
                    f"{flag} configures the latent-thought policy; "
                    f"--reasoning-mode {args.reasoning_mode} samples no "
                    "thought or stop gate"
                )
    if not carries_hidden:
        if args.thought_sigma is None:
            args.thought_sigma = DEFAULT_THOUGHT_SIGMA
        if args.init_stop_thinking_probability is None:
            args.init_stop_thinking_probability = (
                DEFAULT_INIT_STOP_THINKING_PROBABILITY
            )
    if carries_hidden and args.rollout_scheduler != "lockstep":
        parser.error(
            f"--reasoning-mode {args.reasoning_mode} requires "
            "--rollout-scheduler lockstep: continuous refill does not "
            "implement the hidden carry"
        )
    if not carries_hidden:
        if (
            not math.isfinite(args.thought_sigma)
            or args.thought_sigma <= 0.0
        ):
            parser.error("--thought-sigma must be finite and positive")
        if (
            not math.isfinite(args.init_stop_thinking_probability)
            or not 0.0 < args.init_stop_thinking_probability < 1.0
        ):
            parser.error(
                "--init-stop-thinking-probability must be strictly in (0, 1)"
            )
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("--learning-rate must be finite and positive")
    if args.critic_learning_rate is None:
        args.critic_learning_rate = args.learning_rate
    if (
        not math.isfinite(args.critic_learning_rate)
        or args.critic_learning_rate <= 0.0
    ):
        parser.error("--critic-learning-rate must be finite and positive")
    if (args.critic_action_head_hidden is None and args.critic_q_head_hidden is None
            and args.critic_post_action_q_hidden is None):
        if args.critic_action_head_learning_rate is not None:
            parser.error(
                "--critic-action-head-learning-rate needs "
                "--critic-action-head-hidden, --critic-q-head-hidden, or "
                "--critic-post-action-q-hidden"
            )
    else:
        if args.critic_action_head_learning_rate is None:
            args.critic_action_head_learning_rate = args.critic_learning_rate
        if (
            not math.isfinite(args.critic_action_head_learning_rate)
            or args.critic_action_head_learning_rate <= 0.0
        ):
            parser.error(
                "--critic-action-head-learning-rate must be finite and positive"
            )
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
        args.critic_muon_learning_rate = (
            args.critic_learning_rate
            * (0.025 / 0.015)
            * POLAR_EXPRESS_STEP_COMPENSATION
        )
    for name, value in (
        ("--muon-learning-rate", args.muon_learning_rate),
        ("--critic-muon-learning-rate", args.critic_muon_learning_rate),
    ):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")
    if args.fresh_learning_rate is None and args.reasoning_mode == "memory":
        args.fresh_learning_rate = MEMORY_FRESH_LEARNING_RATE
    if args.fresh_learning_rate is None:
        if args.fresh_muon_learning_rate is not None:
            parser.error("--fresh-muon-learning-rate requires --fresh-learning-rate")
    else:
        if args.reasoning_mode in ("cot", "none"):
            parser.error(
                "--fresh-learning-rate needs fresh modules the policy uses: "
                f"{args.reasoning_mode} has none"
            )
        if args.fresh_muon_learning_rate is None:
            args.fresh_muon_learning_rate = (
                args.fresh_learning_rate
                * (0.025 / 0.015)
                * POLAR_EXPRESS_STEP_COMPENSATION
            )
        for name, value in (
            ("--fresh-learning-rate", args.fresh_learning_rate),
            ("--fresh-muon-learning-rate", args.fresh_muon_learning_rate),
        ):
            if not math.isfinite(value) or value <= 0.0:
                parser.error(f"{name} must be finite and positive")
    if args.reset_optimizers_on_resume and not args.resume:
        parser.error("--reset-optimizers-on-resume requires --resume")
    if args.allow_dg_topology_migration and not args.resume:
        parser.error("--allow-dg-topology-migration requires --resume")
    if args.allow_token_rng_migration and not args.resume:
        parser.error("--allow-token-rng-migration requires --resume")
    if args.allow_rollout_groups_migration:
        if not args.resume:
            parser.error("--allow-rollout-groups-migration requires --resume")
        if (
            args.reasoning_mode not in ("cot", "none", "carry", "memory")
            or args.rollout_scheduler != "lockstep"
        ):
            parser.error(
                "--allow-rollout-groups-migration needs the counter-keyed "
                "lockstep cot/none/carry/memory draw; other schedulers and latent "
                "mode sample with chunk-dependent state"
            )
    if (
        not math.isfinite(args.nearby_reward_max)
        or args.nearby_reward_max < 0.0
        or args.nearby_reward_max >= 1.0
    ):
        parser.error(
            "--nearby-reward-max must be finite, nonnegative, and below "
            "the exact-answer reward of 1"
        )
    if args.think_tokens and args.reasoning_mode == "none":
        # none mode teacher-forces "Answer:" and budgets only the answer
        # value: no fence can ever be emitted, so every format-gated
        # reward would be zero and the run trains on nothing.
        parser.error(
            "--think-tokens requires a reasoning mode that emits its own "
            "reasoning; none-mode budgets only the answer value"
        )
    if args.think_min_tokens is None:
        args.think_min_tokens = 33 if args.think_tokens else 1
    if args.think_min_tokens < 1:
        parser.error("--think-min-tokens must be at least 1")
    if args.think_min_tokens > 1 and not args.think_tokens:
        parser.error("--think-min-tokens above 1 requires --think-tokens")
    if args.answer_fence and not args.think_tokens:
        parser.error("--answer-fence requires --think-tokens")
    if args.abstention_penalty is not None:
        if not math.isfinite(args.abstention_penalty) or args.abstention_penalty <= 0.0:
            parser.error("--abstention-penalty must be finite and positive")
        threshold = normalized_abstention_reward(args.abstention_penalty)
        threshold_f32 = struct.unpack("f", struct.pack("f", threshold))[0]
        if not 0.0 < threshold_f32 < 1.0:
            parser.error(
                "--abstention-penalty must give a float32 reward strictly "
                "between wrong (0) and correct (1)"
            )
        if not args.answer_fence:
            # Code and math answers still use the trained answer fence.
            parser.error("--abstention-penalty requires --answer-fence")
        if args.source_success_actor_gate:
            # The gate silences a source until some row scores above 0, but
            # a source of wrong answers and abstentions is exactly where
            # the penalty's contrast teaches when to abstain.
            parser.error(
                "--abstention-penalty conflicts with --source-success-actor-gate"
            )
    if args.zero_reward_stop_pools < 0:
        parser.error("--zero-reward-stop-pools must be nonnegative")
    if args.replay_attention_budget < 1:
        parser.error("--replay-attention-budget must be positive")
    if args.replay_slot_budget < 1:
        parser.error("--replay-slot-budget must be positive")
    if args.post_update_kl_every < 0:
        parser.error("--post-update-kl-every must be nonnegative")
    if args.replay_bucket < 1:
        parser.error("--replay-bucket must be positive")
    if not 300 <= args.checkpoint_interval_seconds <= 600:
        parser.error(
            "--checkpoint-interval-seconds must be between 300 and 600"
        )
    if args.snapshot_every < 0:
        parser.error("--snapshot-every must be nonnegative")
    if args.eval_batch_trajectories < 1:
        parser.error("--eval-batch-trajectories must be positive")
    if args.eval_tail_batch < 0:
        parser.error("--eval-tail-batch must be nonnegative")
    if args.rollout_tail_batch < 0:
        parser.error("--rollout-tail-batch must be nonnegative")
    if mode_pins_emit(args.reasoning_mode) and args.rollout_scheduler == "lockstep":
        # Token-only lockstep modes decode through graph_decode's arena,
        # which buckets rows and KV width itself; these flags configure the
        # eager loop that only latent still runs, and would be silently inert.
        for flag, value in (
            ("--rollout-flex-decode", args.rollout_flex_decode),
            ("--rollout-graph-decode", args.rollout_graph_decode),
            ("--rollout-tail-graph", args.rollout_tail_graph),
        ):
            if value:
                parser.error(
                    f"{flag} configures the latent lockstep decode loop; "
                    f"--reasoning-mode {args.reasoning_mode} decodes through "
                    "the CUDA-graph arena"
                )
    if args.rollout_tail_graph and args.rollout_tail_batch < 1:
        parser.error("--rollout-tail-graph requires --rollout-tail-batch >= 1")
    if args.rollout_flex_decode and not args.rollout_compile:
        # The block table replaces the boolean mask of the tensor-position
        # step, which only the compiled lockstep rollout takes.
        parser.error("--rollout-flex-decode requires --rollout-compile")
    if args.rollout_flex_decode and args.rollout_scheduler != "lockstep":
        # continuous_refill already decodes through flex over paged lanes.
        parser.error("--rollout-flex-decode applies to the lockstep scheduler")
    if (
        args.rollout_graph_decode
        and args.rollout_scheduler == "lockstep"
        and not args.rollout_flex_decode
    ):
        # Capture needs one static row count, and the empty KV range is what
        # makes holding one affordable when rows finish early. The boolean
        # mask path has no equivalent: a fully masked SDPA row is NaN. The
        # continuous_refill scheduler needs no extra flag: its paged step
        # already decodes through flex over bucketed static widths, so
        # graph capture applies to it directly.
        parser.error(
            "--rollout-graph-decode with --rollout-scheduler lockstep "
            "requires --rollout-flex-decode"
        )
    if args.rollout_graph_decode and args.rollout_tail_graph:
        # The tail graph exists to give the compacted remnant a static shape.
        # Under capture the main loop never compacts, so there is no remnant
        # and the tail artifact would only compile and never run.
        parser.error(
            "--rollout-graph-decode already holds a static shape; "
            "--rollout-tail-graph has nothing left to snap to"
        )
    if args.rollout_tail_graph and not args.rollout_compile:
        # The tail switch rides the compiled rollout's tensor positions and
        # fixed-size compaction; an eager rollout never engages it.
        parser.error("--rollout-tail-graph requires --rollout-compile")
    if args.rollout_scheduler == "continuous_refill":
        if args.rollout_groups <= 1:
            parser.error(
                "--rollout-scheduler continuous_refill requires "
                "--rollout-groups > 1"
            )
        if args.rollout_groups >= args.prompts_per_rollout:
            parser.error(
                "--rollout-scheduler continuous_refill requires more than "
                "one rollout chunk"
            )
        if not args.rollout_compile:
            parser.error(
                "--rollout-scheduler continuous_refill requires "
                "--rollout-compile"
            )
        if args.rollout_tail_graph:
            parser.error(
                "--rollout-tail-graph is incompatible with "
                "--rollout-scheduler continuous_refill"
            )
    # PPO refresh/update score the backbone's categorical distribution
    # directly. Temperature and nucleus transforms define a different policy;
    # accepting them here would sample under q while optimizing log p.
    if args.temperature != 1.0:
        parser.error(
            "--temperature must be 1 for latent VAPO; transformed sampling "
            "is not part of the scored PPO policy"
        )
    if args.top_p != 1.0:
        parser.error(
            "--top-p must be 1 for latent VAPO; nucleus sampling is not part "
            "of the scored PPO policy"
        )
    if args.bench_only_repeats < 1:
        parser.error("--bench-only-repeats must be positive")
    if args.bench_max_rows < 0:
        parser.error("--bench-max-rows must be nonnegative")
    if args.rollout_sample_every < 0:
        parser.error("--rollout-sample-every must be nonnegative")
    if args.prompts_per_rollout < 1:
        parser.error("--prompts-per-rollout must be positive")
    args.exclude_rl_sources = ",".join(sorted(set(
        name.strip() for name in args.exclude_rl_sources.split(",") if name.strip()
    )))
    if args.exclude_rl_sources and not args.rl_mixture_manifest:
        parser.error("--exclude-rl-sources requires --rl-mixture-manifest")
    if args.exclude_rl_sources and args.consume_all_prompts:
        parser.error("source exclusions currently require an explicit --steps budget")
    if args.rl_mixture_manifest and args.exclude_modules:
        parser.error("--exclude-modules is not supported with an RL mixture")
    if args.prompts_per_minibatch < 1:
        parser.error("--prompts-per-minibatch must be positive")
    if args.prompts_per_rollout % args.prompts_per_minibatch:
        parser.error(
            "--prompts-per-rollout must be divisible by --prompts-per-minibatch"
        )
    if (
        (args.delightful_policy_gradient or args.target_policy_optimization)
        and args.prompts_per_rollout != args.prompts_per_minibatch
    ):
        parser.error(
            "the selected on-policy actor estimator "
            "requires --prompts-per-minibatch to equal --prompts-per-rollout "
            "so a frozen rollout pool produces exactly one actor update"
        )
    if args.ppo_epochs != 1:
        parser.error("--ppo-epochs must be 1; trajectory reuse is disabled")
    if args.bpb_val_tokens < 0:
        parser.error("--bpb-val-tokens must be nonnegative")
    if args.randomize_choice_options is None:
        args.randomize_choice_options = not args.rollout_only
    if args.rollout_only_repeats < 1:
        parser.error("--rollout-only-repeats must be positive")
    if args.gate_transcripts and not args.rollout_only:
        parser.error("--gate-transcripts requires --rollout-only")
    if args.rollout_dataset and not args.rollout_only:
        parser.error("--rollout-dataset requires --rollout-only")
    if args.rollout_dataset and args.reasoning_mode != "cot":
        # Only a token policy's stream is a function of its token ids, which
        # is what lets a recorded trajectory be replayed exactly.
        parser.error("--rollout-dataset requires --reasoning-mode cot")
    if args.rollout_dataset and (args.actor_init or args.resume):
        # The dataset binds the policy it sampled: --checkpoint, or the
        # --curriculum-init snapshot whose actor replaces it (its hash is
        # bound too). An --actor-init or resumed run would sample one policy
        # under another's cursor and critic.
        parser.error(
            "--rollout-dataset samples --checkpoint or a --curriculum-init "
            "policy; it refuses --actor-init and --resume"
        )
    if args.rollout_only_repeats != 1 and not args.rollout_only:
        parser.error("--rollout-only-repeats requires --rollout-only")
    validate_perturbation_sweep(parser, args)
    validate_write_noise_sweep(parser, args)
    exclusive_modes = sum(
        (args.bpb_only, args.bench_only, args.rollout_only)
    )
    if exclusive_modes > 1:
        parser.error(
            "--bpb-only, --bench-only, and --rollout-only are mutually exclusive"
        )
    if args.memory_eval_ablation != "none":
        if not args.bench_only:
            parser.error("--memory-eval-ablation is an evaluation ablation: it requires --bench-only")
        if not mode_carries_memory(args.reasoning_mode):
            parser.error("--memory-eval-ablation ablates the memory carry: --reasoning-mode memory")
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
            args.critic_only_init,
            args.curriculum_init,
            args.resume,
        )
    )
    if initialization_modes > 1:
        parser.error(
            "--actor-init, --critic-only-init, --curriculum-init, and "
            "--resume are mutually exclusive"
        )
    if args.critic_only_init:
        if args.value_warmup_steps != 0:
            parser.error("--critic-only-init requires --value-warmup-steps 0")
        if args.critic_init == "actor":
            parser.error("--critic-init actor conflicts with --critic-only-init")
    if args.fresh_critic and not args.actor_init:
        parser.error(
            "--fresh-critic applies only to --actor-init, the one "
            "initialization that otherwise carries a trained critic"
        )
    if args.accept_critic_value_change and not (
        args.actor_init and not args.fresh_critic
    ):
        parser.error(
            "--accept-critic-value-change applies only to a critic carried "
            "by --actor-init"
        )
    if args.critic_init == "actor" and args.actor_init and not args.fresh_critic:
        parser.error(
            "--critic-init actor conflicts with --actor-init, which carries "
            "the checkpoint's trained critic; add --fresh-critic to start "
            "one from the actor instead"
        )
    if args.samples_per_prompt < 1:
        parser.error("--samples-per-prompt must be positive")
    if args.actor_gae_lambda is not None and not 0.0 <= args.actor_gae_lambda <= 1.0:
        parser.error("--actor-gae-lambda must lie in [0, 1]")
    if args.critic_action_head_hidden is not None:
        if args.critic_action_head_hidden < 1:
            parser.error("--critic-action-head-hidden must be positive")
        if args.target_policy_optimization:
            parser.error(
                "--critic-action-head-hidden needs an actor objective that "
                "consumes its advantages; TPO does not"
            )
    if args.critic_post_action_q_hidden is not None:
        if args.critic_post_action_q_hidden < 1:
            parser.error("--critic-post-action-q-hidden must be positive")
        if not math.isfinite(args.critic_post_action_q_prior) or not (
            0.0 < args.critic_post_action_q_prior < 1.0
        ):
            parser.error("--critic-post-action-q-prior must lie strictly in (0, 1)")
        if args.critic_action_head_hidden is not None or args.critic_q_head_hidden is not None:
            parser.error("post-action Q cannot be combined with another critic head")
        if not args.delightful_policy_gradient or args.target_policy_optimization:
            parser.error("post-action Q requires --delightful-policy-gradient")
        if args.reasoning_mode == "latent":
            parser.error("post-action binary Q supports token actions only")
        if args.prompts_per_rollout != args.prompts_per_minibatch:
            parser.error("post-action Q requires one fresh rollout minibatch")
        if args.actor_gae_lambda not in (None, 0.0):
            parser.error("post-action Q requires --actor-gae-lambda 0")
        if args.q_critic_lambda not in (None, 1.0):
            parser.error("post-action binary Q requires --q-critic-lambda 1")
        args.actor_gae_lambda = 0.0
        args.q_critic_lambda = 1.0
        if args.nearby_reward_max != 0 or args.abstention_penalty is not None:
            parser.error("post-action binary Q forbids fractional or shaped rewards")
        if args.python_reward_mode != "binary":
            parser.error("post-action binary Q requires --python-reward-mode binary")
        if not args.think_tokens or not args.answer_fence:
            parser.error("post-action Q requires --think-tokens and --answer-fence")
        if args.source_success_actor_gate or args.zero_reward_actor_freeze:
            parser.error("post-action Q forbids sampled-outcome actor masking; use "
                         "--no-source-success-actor-gate --no-zero-reward-actor-freeze")
        if args.entropy_target is not None or args.actor_kl_bound is not None:
            parser.error("post-action binary Q uses only its conditional binary-DG objective")
        if args.actor_init and not args.fresh_critic and not args.resume:
            parser.error("post-action Q needs --fresh-critic with --actor-init")
        if not args.resume and not args.critic_only_init and args.critic_init != "actor":
            parser.error("post-action Q requires --critic-init actor for an informed independent trunk")
    if args.critic_q_head_hidden is not None:
        if args.critic_q_head_hidden < 1:
            parser.error("--critic-q-head-hidden must be positive")
        if args.critic_action_head_hidden is not None:
            parser.error(
                "a standalone Q critic has no V for --critic-action-head-hidden "
                "to add to"
            )
        if not (args.delightful_policy_gradient or args.mean_actor_critic):
            parser.error(
                "--critic-q-head-hidden drives the Delightful or MAC actor "
                "with policy-centered Q; pass --delightful-policy-gradient "
                "or --mean-actor-critic"
            )
        if args.reasoning_mode == "latent":
            parser.error(
                "a Q critic needs fused behavior statistics, which latent "
                "thoughts do not have"
            )
        if args.prompts_per_rollout != args.prompts_per_minibatch:
            parser.error(
                "a Q critic forms its targets and advantages in the one fresh "
                "minibatch's critic pass; --prompts-per-minibatch must equal "
                "--prompts-per-rollout"
            )
        if args.actor_gae_lambda is not None:
            parser.error(
                "a Q critic's actor advantage has no lambda; it belongs to its "
                "targets: pass --q-critic-lambda instead of --actor-gae-lambda"
            )
        if args.q_critic_lambda is not None and not (
            0.0 <= args.q_critic_lambda <= 1.0
        ):
            parser.error("--q-critic-lambda must lie in [0, 1]")
        if args.actor_init and not args.fresh_critic and not args.resume:
            parser.error(
                "a Q critic starts fresh from an --actor-init checkpoint; pass "
                "--fresh-critic"
            )
        if not (args.resume or args.critic_only_init) and args.critic_init != "actor":
            # The action table is the trunk's readout, zero in a fresh trunk,
            # where Q has no token contrast and no gradient that leaves it.
            parser.error("a token Q critic requires --critic-init actor: its action "
                         "table is the trunk's readout, zero in a fresh trunk")
    elif args.q_critic_lambda is not None and args.critic_post_action_q_hidden is None:
        parser.error("--q-critic-lambda needs a standalone or post-action Q critic")
    if args.entropy_target is None:
        for name in (
            "entropy_temperature_init", "entropy_temperature_learning_rate"
        ):
            if getattr(args, name) is not None:
                parser.error(
                    f"--{name.replace('_', '-')} needs --entropy-target"
                )
    else:
        if args.entropy_temperature_init is None:
            parser.error(
                "--entropy-target needs --entropy-temperature-init, set "
                "relative to the scale of Q"
            )
        if args.entropy_temperature_learning_rate is None:
            args.entropy_temperature_learning_rate = 0.05
        if not math.isfinite(args.entropy_target) or args.entropy_target <= 0.0:
            parser.error("--entropy-target must be finite and positive")
        if not args.delightful_policy_gradient:
            parser.error(
                "--entropy-target is a term of the Delightful objective"
            )
        for name, value in (
            ("--entropy-temperature-init", args.entropy_temperature_init),
            (
                "--entropy-temperature-learning-rate",
                args.entropy_temperature_learning_rate,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                parser.error(f"{name} must be finite and positive")
    if args.actor_kl_bound is None:
        if args.actor_kl_backtracks is not None:
            parser.error("--actor-kl-backtracks needs --actor-kl-bound")
        if args.actor_kl_shard_stride is not None:
            parser.error("--actor-kl-shard-stride needs --actor-kl-bound")
    else:
        if args.actor_kl_backtracks is None:
            args.actor_kl_backtracks = 3
        if args.actor_kl_shard_stride is None:
            args.actor_kl_shard_stride = 4
        if args.actor_kl_shard_stride < 1:
            parser.error("--actor-kl-shard-stride must be positive")
        if not math.isfinite(args.actor_kl_bound) or args.actor_kl_bound <= 0.0:
            parser.error("--actor-kl-bound must be finite and positive")
        if args.actor_kl_backtracks < 1:
            parser.error("--actor-kl-backtracks must be positive")
        if args.reasoning_mode == "latent":
            parser.error(
                "--actor-kl-bound measures token KL only, not latent thoughts"
            )
    if args.value_warmup_rewind:
        if args.value_warmup_steps < 1 or args.value_warmup_scope != "critic":
            parser.error(
                "--value-warmup-rewind applies to a whole-critic warmup with "
                "--value-warmup-steps > 0; an action-head warmup always rewinds"
            )
        # A resume carries the --actor-init cursor in its checkpoint.
        if not (args.actor_init or args.resume):
            parser.error(
                "--value-warmup-rewind returns to an --actor-init checkpoint's "
                "prompt cursor"
            )
    if args.value_warmup_scope == "action-head":
        if args.critic_action_head_hidden is None:
            parser.error(
                "--value-warmup-scope action-head needs "
                "--critic-action-head-hidden"
            )
        if args.value_warmup_steps < 1:
            parser.error(
                "--value-warmup-scope action-head needs --value-warmup-steps > 0"
            )
        if not (args.resume or (args.actor_init and not args.fresh_critic)):
            parser.error(
                "--value-warmup-scope action-head warms a head against "
                "values already fit: carry a critic with --actor-init"
            )
    # Replay is token-normalized and memory-sharded, so trajectory batch size
    # is an actual optimization choice rather than a shape invariant. The
    # selected DG regime uses 24 x 16 = 384 fresh trajectories per update.
    for name, samples in (
        ("--aime-samples", args.aime_samples),
        ("--bench-samples", args.bench_samples),
    ):
        if samples < 1:
            parser.error(f"{name} must be positive")
    if args.bench_capture_problems < 1 or args.bench_capture_samples < 1:
        parser.error("--bench-capture-problems and --bench-capture-samples must be positive")
    if 0 < args.bench_max_rows < args.bench_capture_problems:
        parser.error("--bench-capture-problems exceeds --bench-max-rows")
    if (
        (args.bench_every > 0 or args.bench_only)
        and args.bench_samples < args.bench_capture_samples
    ):
        parser.error(
            "--bench-samples must be at least --bench-capture-samples "
            f"({args.bench_capture_samples}) for answer capture"
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

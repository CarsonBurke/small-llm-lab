"""Command-line configuration and argv-level validation for latent VAPO."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from postraining.benchmark_report import CAPTURE_SAMPLES_PER_PROBLEM


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


def build_arg_parser() -> argparse.ArgumentParser:
    """The CLI surface, separate from main() so the shipped defaults
    and the argv-level guards are testable without running training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    # The selected production regime is a 40K-update Delightful Policy
    # Gradient campaign. Shorter runs remain explicit ablations rather than
    # the behavior of an otherwise production-shaped invocation.
    parser.add_argument("--steps", type=int, default=40_000)
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
        default="postraining/data/vapo_broad_v5.manifest.json",
        help="immutable multi-source verifier manifest whose exact source "
        "quotas replace --math-data (pass an empty string for a single "
        "--math-data source)",
    )
    parser.add_argument(
        "--exclude-modules", default="",
        help="comma-separated extra_info.module names to drop from --math-data "
        "(module-tagged datasets only); names absent from the data are an error",
    )
    # The reasoning mode fixes the rollout policy family for the whole run:
    # "latent" carries each generated token's producing belief back into its
    # input through the gated combined embedding; "cot" is the token-only
    # control (no hidden carry) with the full token budget; "none" is
    # token-only, teacher-forces an "Answer:" prefix onto the prompt, and
    # budgets only the answer itself.
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
    # Selected DG topology: collect one fresh 24-prompt x 16-sample batch from
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
        default=True,
        help="replace the clipped VAPO/PPO actor objective with Delightful "
        "Policy Gradient (Osband, 2026), using eta=1 current-token "
        "surprisal and no importance ratios",
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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    # One general rate for actor and critic. The fresh-policy experiment starts
    # every actor-side optimizer state empty from the critic-warm checkpoint;
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
    # which the near-frozen bias (AdamW at 2e-5) then takes many steps to
    # unwind through head.weight alone.
    parser.add_argument("--value-prior", type=float, default=0.0)
    # Combined-embedding geometry (reasoning mode "latent"). The carry matrix
    # is zero-initialized, so the injected carry vanishes and step 0
    # reproduces the pretrained token path bit-for-bit while the matrix gets
    # a full-rank first-step gradient (loss direction outer hidden). No
    # scalar gate: the v1 gain*W factorization was a multiplicative saddle
    # neither factor escaped.
    # Prenorm-residual relu^2 MLP blocks applied to the combined embedding at
    # hasThought positions, identity at init (zeroed proj). 0 is the
    # pure-gated-residual ablation arm.
    parser.add_argument("--combined-mlp-blocks", type=int, default=1)
    # Hidden width of each combiner MLP block; 2048 matches the trunk's own
    # 4x blocks.
    parser.add_argument("--combined-mlp-hidden", type=int, default=2048)
    # Length-adaptive GAE lambda (core.length_adaptive_lambda): VAPO's
    # horizon alpha*l with a floor of min(l, 1/alpha).  The raw alpha=0.05
    # formula clamps lambda to 0 at this run's 12-23-action trajectories
    # (TD(0): terminal reward credits nothing more than one step back —
    # audited as the think-fraction ratchet); the floor restores
    # whole-trajectory credit for short answers while keeping VAPO's
    # variance control for long ones.
    parser.add_argument("--gae-lambda-alpha", type=float, default=0.05)
    parser.add_argument(
        "--nearby-reward-max",
        type=float,
        default=0.0,
        help="maximum reward for a wrong, terminated numeric final answer",
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
        "every trajectory earned zero reward; the critic still updates so "
        "value predictions catch down to the zero targets",
    )
    # The actor freeze is the actual safety intervention. A stop based only on
    # consecutive zero-reward prompt windows is unsound for a heterogeneous
    # sparse-reward mixture: the frozen actor is unchanged, but later prompts
    # can be easier. Keep the optional operational circuit breaker available
    # without making it part of the selected production regime.
    parser.add_argument(
        "--zero-reward-stop-pools",
        type=int,
        default=0,
        help="stop at the pool boundary (saving the final checkpoint) "
        "after this many consecutive all-zero-reward rollout pools; "
        "applies regardless of --zero-reward-actor-freeze; 0 disables",
    )
    # VAPO paper: 50 value-pretraining steps before policy updates.
    parser.add_argument("--value-warmup-steps", type=int, default=50)
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
    # Bounds the LINEAR per-shard memory term: slots x 50257-wide emit
    # logits (plus their autograd-retained log-softmax, ~6 bytes/element in
    # the update path). 24576 slots ~= 7.4 GiB retained per shard worst
    # case (every slot an action); realistic emit fractions retain ~half.
    # Without this, raising --replay-attention-budget lets short-L shards
    # grow their slot count unboundedly and the vocabulary head OOMs
    # before attention.
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
    # groups*samples rows per launch is the utilization lever. Measured on
    # the v28 rollout gate (jobs 969/978/979, 1024 trajectories each):
    # 16 groups 13.2k useful actions/s at 5.5 GiB, 32 groups 42.4k at
    # 10.4 GiB, 64 groups 41.1k at 20.0 GiB — width saturates at 32.
    # continuous_refill measured 6.4k/s (job 980): 0.92 step utilization
    # cannot buy back its per-step paged overhead at this model scale.
    parser.add_argument("--rollout-groups", type=int, default=32)
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
        "--rollout-only-repeats",
        type=int,
        default=1,
        help="production-shape rollout pools to collect in one process; "
        "repeat zero includes compile/cold-start cost and later repeats "
        "measure steady state",
    )
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
    # TPO is an explicit replacement for the default DG estimator. Resolve
    # that precedence once so every downstream objective/schema check sees
    # exactly one actor mode, while ordinary invocations retain the shipped
    # DG default.
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
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("--learning-rate must be finite and positive")
    if args.critic_learning_rate is None:
        args.critic_learning_rate = args.learning_rate
    if (
        not math.isfinite(args.critic_learning_rate)
        or args.critic_learning_rate <= 0.0
    ):
        parser.error("--critic-learning-rate must be finite and positive")
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
    if args.reset_optimizers_on_resume and not args.resume:
        parser.error("--reset-optimizers-on-resume requires --resume")
    if args.allow_dg_topology_migration and not args.resume:
        parser.error("--allow-dg-topology-migration requires --resume")
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
    if args.rl_mixture_manifest and args.exclude_modules:
        parser.error("--exclude-modules is not supported with an RL mixture")
    if args.rl_mixture_manifest and args.consume_all_prompts:
        parser.error("--consume-all-prompts is not defined for cyclic mixtures")
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
    if args.rollout_only_repeats < 1:
        parser.error("--rollout-only-repeats must be positive")
    if args.rollout_only_repeats != 1 and not args.rollout_only:
        parser.error("--rollout-only-repeats requires --rollout-only")
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
    if args.samples_per_prompt < 1:
        parser.error("--samples-per-prompt must be positive")
    # Replay is token-normalized and memory-sharded, so trajectory batch size
    # is an actual optimization choice rather than a shape invariant. The
    # selected DG regime uses 24 x 16 = 384 fresh trajectories per update.
    for name, samples in (
        ("--aime-samples", args.aime_samples),
        ("--bench-samples", args.bench_samples),
    ):
        if samples < 1:
            parser.error(f"{name} must be positive")
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

#!/usr/bin/env bash
# Post-training pipeline for the best pretrained KDA checkpoint.
#
# Base: logs/nanogpt_gpt2_kda8_kkkdkkkd_triton_mbs32_optimized_2k_final_model.pt
#   Lowest val_bpb (1.1824) of the 63 KDA pretraining runs in logs/, full
#   2000 steps, validated on data/datasets/fineweb10B_gpt2 like its peers so
#   the BPB comparison is like-for-like. 64.0M params, 8 layers x 512 dim,
#   GPT-2 padded vocab 50304, KDA mixers on layers 0-2/4-6 with full
#   attention on 3 and 7, train_seq_len 1024.
#
# Stage 1 runs at --seq-len 5120 as of 2026-09-21. Stage 2 now requests
# 256 prompt + 2048 response tokens; its older base therefore runs beyond
# its 1024-token trained window, recorded in the run manifest. SFT extension is deliberate
# and measured, not a default: UltraData's think split is reasoning traces,
# and at 1024 only 36.6% of its math traces and 3.3% of its code traces fit,
# against 71.0% and 33.4% at 5120. Layers 3 and 7 carry half-truncate RoPE
# computed dynamically from torch.arange(T), so a longer window runs without
# a code change but is extrapolation for those two layers until trained at
# length -- which is what this run does. Whole-document packing keeps
# positions 1024-5120 densely supervised by packed short documents rather
# than only by the rare long trace, so the extension is trained, not assumed.
# Whether it held is a measurement, not an assumption: compare the
# arithmetic probe against job 9040 before trusting stage 2 at length. The
# probe is the matched panel; the sampling gate is not, because the corpus
# and --holdout-problems both changed, so its panel is different problems.
#
# Corpus and epochs changed on 2026-09-21. The previous stage 1 distilled
# ~36k self-generated traces for 3 epochs: holdout completion CE fell
# 3.9449 -> 0.9009, and the resulting policy still scored 0.00 on every
# DeepMind interpolate module, 0.20% on dapo-math-17k, and -- decisively --
# 0.000 on all 15 families of the arithmetic capability probe at every digit
# count, 3-digit integer addition included, and 0.000 leniently too. A
# near-zero training loss beside an absent primitive is memorisation of a
# small trace set. Those traces and both 3-epoch checkpoints are deleted.
#
# Stage 1 makes one pass over a large corpus. The v2 5120-token mix is
# 887,458 documents: 334,058 OpenMathInstruct-2 worked solutions, 253,400
# think-split reasoning traces from UltraData-SFT-2605 (191,720 of them its
# OpenMathInstruct-2 subset, 58,384 Nemotron-Cascade math, 3,296 code), and
# 300,000 worked arithmetic drills from data/math_drills/v4, which are the
# only rows here that show carrying, borrowing, place value and
# digit-by-digit long division. GSM8K-derived rows are excluded by
# problem_source, because GSM8K is an evaluation set for this project, and
# every problem additionally runs through prepare_sft_traces'
# decontamination index (exact + word 8-grams over the GSM8K test
# questions, the DeepMind interpolate bench panel and AIME 2024/2025/2026)
# plus a containment check against the KodCode and GSM8K RL prompt pools.
# That removed 225,740 contaminated and 2,746 containing documents the
# provenance labels alone would have admitted: OpenMathInstruct-2 is
# MATH-derived and MATH contains AIME problems. UltraData rows carry no
# problem_source, so rewritten GSM8K-train variants may survive; GSM8K test
# is unaffected (see the corpus manifest's gsm8k_train_derivation_caveat).
#
# --allow-unverified is required and deliberate: these solutions are
# model-generated and this repository does not re-verify them, so the corpus
# stores verified=False and the trainer refuses it unless the operator says
# so. It is not a verification the corpus claims and the flag waives.
#
# Stage 1 (sft) and stage 2 (rl) are separate mlq jobs. Stage 2 names its
# base explicitly (RL_BASE_RUN below), chosen from matched evaluations of
# the candidate stage 1 checkpoints rather than assumed to be the newest.
#
# Reproducibility: the RL mixture manifest binds sft_corpus_sha256 to the
# exact SFT corpus bytes, and train_latent_vapo refuses a base checkpoint
# whose recorded traces_sha256 differs. Each stage 2 run is therefore
# hash-bound to its base's corpus; a mismatch fails the run instead of
# drifting it.
#
# The mixture is v12_exact (schema vapo_verifiable_mixture/v2, 2026-09-23):
# deepmind_easy + the full UltraData Math pool (v5, 19,765) + DAPO deduplicated
# against it (8,408), 99,917 prompts with no quotas. One mixture cycle serves
# every prompt exactly once, sources interleaved by size, so no source loops
# faster than another. The earlier v8/v9 quotas (48/8/8 per 64) reran the
# 2,306-prompt ultradata pool ~31 times per pass over deepmind_easy.
# The manifest is bound to the job 9040 corpus (sft_mix_omi2_drills_v2); a
# stage 1 checkpoint from any other corpus is refused at startup and needs a
# manifest rebuilt against its corpus.
#
# v8 leads with deepmind_easy: the train-easy tier of the same 18 modules,
# a disjoint training split at markedly easier surface difficulty (compare
# train-easy "What is -5 - 110911?" against interpolate "What is the
# difference between -221017 and -1429.06?"). The 144 problems of the
# interpolate bench panel are excluded from it by question text, so that
# panel stays genuinely held out. ultradata_math is the MiniCPM RL corpus
# (openbmb/UltraData-RL-2609, Math domain only: 95% of its prompts fit a
# 256-token budget, against 16% for its Code and 0% for Long_Context), and
# dapo stays as a small hard tail so the policy cannot narrow onto one
# templated prompt shape. gsm8k is retired from the mixture entirely -- it
# is an eval here, and prepare_vapo_mixture now refuses to select it.
#
# Usage: scripts/launch_kda8_posttrain.sh {sft|sft-resume|rl|rl-carry} [run-name-suffix]
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 {sft|sft-resume|rl|rl-carry} [run-name-suffix]" >&2
  exit 2
fi
STAGE="$1"
SUFFIX="${2:-}"
BASE=logs/nanogpt_gpt2_kda8_kkkdkkkd_triton_mbs32_optimized_2k_final_model.pt
TRACES=postraining/data/sft_mix_ud2605_omi2_drills_v2.parquet
# Stage 2 starts from job 9040's checkpoint, not this script's stage 1, with
# the v12_exact mixture that is hash-bound to its corpus. The 5120-token stage 1
# lost on every matched panel (2026-09-22): arithmetic probe 0.368 vs 0.516
# over 1920 items (lower on 13 of 15 families, format and termination
# unchanged), DeepMind interpolate 0.005 vs 0.018 with 550 vs 375 mean
# emitted tokens. Its corpus cut drills from 28% to 6.8% of tokens, and two
# thirds of it is 1.8k-2.5k-token think traces that the policy imitates
# into a 768-token RL budget.
RL_BASE_RUN=kda8_sft_omi2_drills_e1
MIXTURE=postraining/data/vapo_broad_v12_exact.manifest.json
# _r2: the first attempt (kda8_sft_ud2605_5k_v2_e1, job 9132) was cancelled
# at step 1,161 to optimize the step; its CANCELLED.md explains why.
SFT_RUN="kda8_sft_ud2605_5k_v2_e1_r2${SUFFIX}"
RL_RUN="kda8_vapo_cot_v9${SUFFIX}"
RL_CARRY_RUN="kda8_vapo_carry_v9${SUFFIX}"
# AFTER_SUCCESS=<job id> holds the submission until that mlq job succeeds,
# e.g. the CUDA tests gating a trainer change.
QUEUE_ARGS=(--cwd "$PWD" --max-parallel-runs 1 --priority 1)
if [ -n "${AFTER_SUCCESS:-}" ]; then
  QUEUE_ARGS+=(--after-success "$AFTER_SUCCESS")
fi

# Stage 1 arguments, shared by sft and sft-resume: a resume must rerun the
# identical command, and the trainer refuses any difference.
SFT_ARGS=(
    --name "$SFT_RUN"
    --checkpoint "$BASE"
    --traces "$TRACES"
    --think-tokens
    --answer-fence
    --allow-unverified
    --epochs 1
    --seq-len 5120
    --allow-context-extension
    --rows-per-micro-batch 8
    --grad-accum 1
    --lr-scale 0.1
    --warmup-steps 20
    --holdout-problems 512
    --eval-every 100
    --gate-prompts 128
    --gate-samples 8
    --gate-prompt-tokens 448
    --gate-max-new-tokens 768
    --gate-think-min-tokens 1
    --seed 1234
)

case "$STAGE" in
sft)
  # 8 rows x 5120 tokens = 40960 tokens/step, close to the 32768 of job 9040
  # so its lr-scale and warmup carry over and the step count stays
  # comparable. All 8 rows run as ONE micro-batch: the step normalizes by its
  # own supervised count, so the micro-batch split is a pure memory knob, and
  # measured on this card (scripts/benchmark_sft_step.py) 8x1 is the fastest
  # split at 110 ms/step and 9.4 GiB peak (job 9155), against 130 ms for 2x4
  # (job 9152). Flash attention keeps layers 3 and 7 linear in memory at 5120.
  #
  # --eval-every 100, not 25: the compiled step is ~110 ms, so an estimated
  # ~0.4 s holdout pass every 25 steps would be ~14% of the run; every 100
  # steps still gives ~220 points on the CE curve.
  #
  # --holdout-problems 512, not 256: the corpus is mixed-domain and code rows
  # are ungradeable by the math verifier, so they are excluded from the gate
  # panel. 512 held-out problems keep the panel comfortably above the 128
  # the gate samples; the trainer errors rather than silently shrinking it.
  # --gate-prompt-tokens 448, not 256: 3 of the 128 gate prompts in this
  # corpus's panel run past 256 tokens (longest 366), and encode_prompt
  # front-truncates, so the gate would grade a question missing its
  # opening. Measured on the real panel before launch (the trainer also
  # refuses at startup). 448+768 still fits the 5120 window.
  #
  # Checkpoints every 5 minutes (the trainer default) and on SIGTERM, so
  # `mlq cancel` stops at a step boundary with nothing lost; continue with
  # the sft-resume stage, which reruns exactly these arguments.
  mlq submit --name "$SFT_RUN" "${QUEUE_ARGS[@]}" -- \
    .venv/bin/python -u -m postraining.sft_trace_train "${SFT_ARGS[@]}"
  ;;
sft-resume)
  mlq submit --name "${SFT_RUN}_resume" "${QUEUE_ARGS[@]}" -- \
    .venv/bin/python -u -m postraining.sft_trace_train "${SFT_ARGS[@]}" --resume
  ;;
rl | rl-carry)
  # --checkpoint-interval-seconds 300: the rolling exact-recovery checkpoint
  # at the trainer's floor, so a stop loses at most five minutes. Resume
  # with --resume on the run's latent_vapo_checkpoint.pt.
  # All generated-response budgets are 2048 tokens. Benchmark/AIME defaults
  # now inherit the rollout budget; explicit values here make launches legible.
  # The runtime context expands to 256 + 2048 = 2304 tokens while preserving
  # the checkpoint's actual 1024-token training metadata.
  #
  # 64 prompts, not the default 24: only mixed groups carry within-group
  # advantage signal, and the SFT gate measured 39.84% mixed prompts.
  # --prompts-per-minibatch equals --prompts-per-rollout, so one pool
  # produces exactly one update at behavior age 0, which lets the update
  # derive its behavior statistics from its own forward instead of a
  # separate refresh pass.
  #
  # Throughput/memory measurements below used the historical 768-token cap;
  # the new 2048-token training configuration has not been benchmarked.
  # --rollout-groups 64 (default 32) makes that pool ONE 1024-row decode wave
  # instead of two 512-row waves. Measured on the RTX 5090 at 4 repeats per
  # config with the warmup repeat discarded (job 9086): 512 rows ~88k tok/s,
  # 1024 rows ~113k, and 2048/4096 rows also ~111k for 2-4x the VRAM. 1024
  # rows is the knee -- 6.3 GB peak, leaving room for actor, critic and
  # optimizer state.
  #
  # cot and carry decode through graph_decode's CUDA-graph arena (fused
  # Triton KDA step, live-range decode attention); the --rollout-flex-decode,
  # --rollout-graph-decode and --rollout-tail-graph flags configure only the
  # latent loop and are refused here. The arena measured ~160k useful tok/s
  # against the eager loop's ~92k on this pool shape (NOTES 2026-09-25). Its
  # full-width caches (5.2 GiB at 1024 rows) raise the rollout peak to
  # ~7.7 GB and are freed before the update.
  # --rollout-scheduler continuous_refill raises lane utilization to 0.83 from
  # 0.45 but collects in 258 s against ~3 s, so it is not viable here.
  # Single-repeat rollout benchmarks are warmup-dominated and misleading: the
  # same config reads 14k tok/s on repeat 0 and 113k on repeat 1.
  #
  # cot runs first as the token-only control -- the right first measurement
  # on a new backbone. rl-carry is the matched treatment: identical arguments
  # and the same SFT checkpoint, differing only in --reasoning-mode carry
  # (deterministic hidden carry). Its combiner is zero-initialized, so carry
  # starts as exactly this cot policy and any divergence is learned.
  # Not latent: that is the stochastic v29 thought policy, and it needs
  # stream_steps >= emitted+1, so a 1024-token window would cost it a third
  # of the response budget.
  if [ "$STAGE" = rl ]; then
    MODE=cot
    RUN="$RL_RUN"
  else
    MODE=carry
    RUN="$RL_CARRY_RUN"
  fi
  mlq submit --name "$RUN" "${QUEUE_ARGS[@]}" -- \
    .venv/bin/python -u -m postraining.train_latent_vapo \
    --checkpoint "postraining/runs/${RL_BASE_RUN}/sft_final_model.pt" \
    --output "postraining/runs/${RUN}" \
    --checkpoint-interval-seconds 300 \
    --rl-mixture-manifest "$MIXTURE" \
    --reasoning-mode "$MODE" \
    --no-delightful-policy-gradient \
    --think-tokens \
    --answer-fence \
    --prompt-tokens 256 \
    --continuation-tokens 2048 \
    --bench-max-tokens 2048 \
    --aime-max-tokens 2048 \
    --answer-tokens 24 \
    --steps 950 \
    --value-warmup-steps 50 \
    --prompts-per-rollout 64 \
    --prompts-per-minibatch 64 \
    --samples-per-prompt 16 \
    --rollout-groups 64 \
    --seed 1337
  ;;
*)
  echo "unknown stage ${STAGE}; expected sft, sft-resume, rl or rl-carry" >&2
  exit 2
  ;;
esac

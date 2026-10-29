#!/usr/bin/env bash
# Antithetic common-random-number perturbation sweep of the gated-delta
# memory at the memory drill's final snapshot (step 1 of the revised order
# in docs/memory_write_critic_plan_20261020.md). No optimizer steps.
#
#   $1 = check   2 pools: the null control (bitwise re-collect) and the
#                flip-rate calibration of the scales; not evidence
#   $1 = main    the measurement; stops at a pool boundary by
#                --max-train-hours inside mlq's 45-minute limit
#
# Policy: --curriculum-init loads the snapshot's actor; the sweep loads the
# snapshot's critic as the sampled-baseline estimator's state value
# (--perturbation-gradient-direction). Everything else is the drill's
# environment (scripts/submit_mac_sampled_baseline_memory_drill_20261020.sh)
# so the pools are the drill's pools: memory mode, fences, budgets,
# 64 prompts x 16 samples, temperature 1, compiled decode and replay.
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
RUNS="$REPO/postraining/runs"
MODE=${1:?check or main}
SNAPSHOT="$RUNS/mac_sampled_baseline_memory_drill_20261020/snapshots/step_000340.pt"
case "$MODE" in
  check)
    NAME=memory_perturbation_check_20261020
    SWEEP=(--rollout-only-repeats 2 --perturbation-scales 0.25 1.0
           --perturbation-random-directions 1 --perturbation-gradient-direction
           --perturbation-gradient-warmup-pools 1)
    ;;
  main)
    NAME=memory_perturbation_sweep_20261020
    # Scales 0.25 and 1 (RMS-relative metric): calibration (job 14335) put
    # writer gradient-direction flip rates at 30-50% and the measured D within
    # the linear range at 0.25, with visible curvature at 1. Sixteen warmup
    # pools fix the gradient direction (base collect + token gradient only);
    # every later pool is a test pool. K=1 random direction per pool.
    SWEEP=(--rollout-only-repeats 64 --perturbation-scales 0.25 1.0
           --perturbation-random-directions 1
           --perturbation-gradient-direction --perturbation-gradient-warmup-pools 16
           --perturbation-radial-direction
           --max-train-hours 0.68)
    ;;
  *) echo "unknown mode $MODE" >&2; exit 64 ;;
esac
exec mlq submit --name "$NAME" --cwd "$REPO" --max-parallel-runs 1 --priority 11 \
  --time-limit 45m -- \
  "$REPO/.venv/bin/python" -u -m postraining.train_latent_vapo \
  --checkpoint "$RUNS/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt" \
  --curriculum-init "$SNAPSHOT" \
  --math-data "$REPO/postraining/data/drill_rl_study_train_20261014_v1.parquet" \
  --rl-mixture-manifest '' \
  --aime-every 0 --bpb-every 0 \
  --reasoning-mode memory --think-tokens --answer-fence --think-min-tokens 33 --answer-tokens 24 \
  --python-reward-mode test-fraction --randomize-choice-options \
  --prompt-tokens 512 --continuation-tokens 2048 \
  --prompts-per-rollout 64 --prompts-per-minibatch 64 --samples-per-prompt 16 --rollout-groups 64 \
  --mean-actor-critic --mac-estimator sampled_baseline --mac-vocab-chunk 512 \
  --critic-q-head-hidden 1024 --critic-q-twin --critic-init actor \
  --trunk-optimizer muon --learning-rate 2e-5 --critic-learning-rate 2e-5 \
  --muon-learning-rate 4.8333333333333334e-05 --critic-muon-learning-rate 4.8333333333333334e-05 \
  --critic-action-head-learning-rate 1e-4 --critic-q-direction-learning-rate 1.953125e-07 \
  --q-critic-lambda 1 --gae-lambda-alpha 0.05 --value-warmup-steps 0 \
  --no-source-success-actor-gate --no-zero-reward-actor-freeze \
  --temperature 1 --top-p 1 --eval-temperature 1 --eval-top-p 1 \
  --compile-replay --replay-max-trajectories 128 --replay-attention-budget 16777216 \
  --replay-bucket 64 --replay-slot-budget 24576 \
  --eval-batch-trajectories 1024 --eval-tail-batch 16 --rollout-tail-batch 16 \
  --seed 1337 \
  --rollout-only --perturbation-sweep writer reader_out "${SWEEP[@]}" \
  --output "$RUNS/$NAME"

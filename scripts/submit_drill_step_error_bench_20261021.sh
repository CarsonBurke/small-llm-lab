#!/usr/bin/env bash
# Full held-out bench capture (512 problems x 16 samples, every answer saved)
# for the step-level error anatomy (scripts/analyze_drill_step_errors.py).
# $1 = snapshot step of the fresh-rate memory drill, or 0 for its start
# policy (the SFT checkpoint; the memory read is zero there). Optional
# AFTER=<job id>: start only after that job succeeds.
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
RUNS="$REPO/postraining/runs"
DRILL=mac_sampled_baseline_memory_fresh_lr_drill_20261021_r2
STEP=${1:?snapshot step, or 0 for the start policy}
INIT=()
if [[ "$STEP" != 0 ]]; then
  INIT=(--curriculum-init "$RUNS/$DRILL/snapshots/step_$(printf %06d "$STEP").pt")
fi
NAME="drill_step_error_bench_s$(printf %03d "$STEP")_20261021"
exec mlq submit --name "$NAME" --cwd "$REPO" --max-parallel-runs 1 --priority 11 \
  --time-limit 30m ${AFTER:+--after-success "$AFTER"} -- \
  "$REPO/.venv/bin/python" -u -m postraining.train_latent_vapo \
  --checkpoint "$RUNS/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt" \
  "${INIT[@]}" \
  --math-data "$REPO/postraining/data/drill_rl_study_train_20261014_v1.parquet" \
  --rl-mixture-manifest '' \
  --bench-data "$REPO/postraining/data/drill_rl_study_heldout_20261014_v1.parquet" \
  --bench-max-rows 512 --bench-samples 16 --bench-max-tokens 2048 \
  --bench-capture-problems 512 --bench-capture-samples 16 \
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
  --fresh-learning-rate 3e-3 \
  --q-critic-lambda 1 --gae-lambda-alpha 0.05 --value-warmup-steps 0 --no-zero-reward-actor-freeze \
  --temperature 1 --top-p 1 --eval-temperature 1 --eval-top-p 1 \
  --eval-batch-trajectories 1024 --eval-tail-batch 16 --rollout-tail-batch 16 \
  --seed 1337 \
  --bench-only \
  --output "$RUNS/$NAME"

#!/usr/bin/env bash
# Antithetic write-noise sweep of the gated-delta memory at the memory
# drill's final snapshot (Stage B of docs/memory_write_critic_plan_20261020.md):
# the measurement gate for a write critic Q_w = b + A(eps). No optimizer steps.
#
# Per pool: the noiseless base, a bitwise null re-collect, then +s and -s for
# every scale, with one noise pattern per prompt shared by its samples (token
# noise independent). scripts/analyze_write_noise_sweep.py turns it into the
# signal bounds. Stops at a pool boundary by --max-train-hours inside mlq's
# 45-minute limit.
#
# Scales are absolute on the writer pre-activations. At step 340 the key and
# value pre-activations are ~1 RMS per coordinate, the write logit ~0 and the
# decay logits 2.7-6.9, so 0.03-1 spans "far inside" to "comparable to the
# signal"; the flip rate per scale is measured, not assumed.
#
# Optional: AFTER=<job id> to start only after that job succeeds (the CUDA
# parity of the noise path).
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
RUNS="$REPO/postraining/runs"
NAME=memory_write_noise_sweep_20261021
SNAPSHOT="$RUNS/mac_sampled_baseline_memory_drill_20261020/snapshots/step_000340.pt"
exec mlq submit --name "$NAME" --cwd "$REPO" --max-parallel-runs 1 --priority 11 \
  --time-limit 45m ${AFTER:+--after-success "$AFTER"} --idempotency-key "${NAME}_r1" -- \
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
  --rollout-only --rollout-only-repeats 256 --write-noise-sweep 0.03 0.1 0.3 1.0 \
  --max-train-hours 0.68 \
  --output "$RUNS/$NAME"

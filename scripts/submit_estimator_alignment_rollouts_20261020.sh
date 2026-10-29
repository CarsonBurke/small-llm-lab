#!/usr/bin/env bash
# Held-out rollout pools from in-run drill policies, for the estimator
# alignment measurement at the policies the two drills actually reached.
# Same contract as dg_critic_study_heldout_rollouts_20261014 (the start
# policy's pools): held-out parquet from row 0, 8 x 64 prompts x 16 samples,
# T=1, top-p 1. --curriculum-init loads the snapshot's actor and resets the
# prompt cursor to 0, so every policy is scored on the same 512 prompts.
#   sb   mac_sampled_baseline_allstates_drill_20261019 (the leader)
#   aa   mac_twin_v7_gain_allaction_drill_20261020 (culled at step 260)
# at step 240, where the gap is clear (bench 0.768 against 0.787).
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
RUNS="$REPO/postraining/runs"
declare -A DRILL=([sb]=mac_sampled_baseline_allstates_drill_20261019 [aa]=mac_twin_v7_gain_allaction_drill_20261020)
for arm in sb aa; do
  for step in 240; do
    name=estimator_alignment_rollouts_20261020_${arm}_s${step}
    snapshot=$(printf "%s/%s/snapshots/step_%06d.pt" "$RUNS" "${DRILL[$arm]}" "$step")
    mlq submit --name "$name" --cwd "$REPO" --max-parallel-runs 1 --priority 10 \
      --time-limit 45m --idempotency-key "${name}_r4" -- \
      "$REPO/.venv/bin/python" -u -m postraining.train_latent_vapo \
      --checkpoint "$RUNS/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt" \
      --curriculum-init "$snapshot" \
      --rl-mixture-manifest '' --reasoning-mode cot --think-tokens --answer-fence \
      --think-min-tokens 33 --nearby-reward-max 0.0 --temperature 1.0 --top-p 1.0 \
      --prompt-tokens 512 --continuation-tokens 2048 --samples-per-prompt 16 \
      --rollout-only --rollout-dataset --gate-min-within-group-reward-std 0 --gate-transcripts \
      --seed 1337 --math-data "$REPO/postraining/data/drill_rl_study_heldout_20261014_v1.parquet" \
      --prompts-per-rollout 64 --prompts-per-minibatch 64 --rollout-only-repeats 8 \
      --output "$RUNS/$name"
  done
done

#!/usr/bin/env bash
# The ablation arm of twin_critic_study_v7_20261020 (cancelled at step 540
# in its trained-gain arm once agreement had peaked at 40-60 and EV had
# flattened): frozen_belief_gain_twin, g held at zero, exactly schema v6.
# Same inputs and data order; the rate is constant, so its first 200
# updates match the trained arm's first 200, the span over which that arm's
# branch agreement rose, peaked and fell (NOTES 2026-10-20).
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
NAME=frozen_gain_study_v7_20261020
RUNS="$REPO/postraining/runs"
exec mlq submit --name "$NAME" --cwd "$REPO" --max-parallel-runs 1 --priority 10 \
  --time-limit 2h --idempotency-key "$NAME" -- \
  "$REPO/.venv/bin/python" -u -m postraining.critic_study \
  --checkpoint "$RUNS/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt" \
  --train-dataset "$RUNS/dg_critic_study_train_rollouts_20261014/rollout_dataset" \
  --heldout-dataset "$RUNS/dg_critic_study_heldout_rollouts_20261014/rollout_dataset" \
  --branch-dataset "$RUNS/dg_critic_study_branches_20261014_v2" \
  --epochs 20 --eval-every 20 --arms frozen_belief_gain_twin \
  --family-alpha 0.008333333333333333 --max-updates 200 \
  --output "$RUNS/dg_frozen_gain_study_v7_20261020"

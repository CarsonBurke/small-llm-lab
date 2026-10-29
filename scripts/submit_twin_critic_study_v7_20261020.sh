#!/usr/bin/env bash
# The twin token-Q critic fit of twin_critic_study_20261019 refit under
# critic schema v7: Q centred under the frozen policy, the critic trunk's
# fixed readout as the action table, and u = D f + L b + g b with the belief
# gain g at the head rate. Two arms, one factor apart:
#   actor_trunk_twin         g trains (the configuration the RL drill uses)
#   frozen_belief_gain_twin  g held at its zero init, exactly schema v6
# A toy at this regime left L's diagonal at 5e-6 after 640 updates at the
# direction rate and put g's top-8 ranking at determined states at 0.83
# against 0.37 without it (NOTES 2026-10-20); this measures it on the real
# critic. Same checkpoint, rollouts, branches, budget, evaluation cadence and
# family alpha as the 20261019 fit. The trained-gain arm's checkpoint
# warm-starts the next --critic-q-twin drill (--study-critic-init).
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
NAME=twin_critic_study_v7_20261020
RUNS="$REPO/postraining/runs"
exec mlq submit --name "$NAME" --cwd "$REPO" --max-parallel-runs 1 --priority 10 \
  --time-limit 3h --idempotency-key "$NAME" -- \
  "$REPO/.venv/bin/python" -u -m postraining.critic_study \
  --checkpoint "$RUNS/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt" \
  --train-dataset "$RUNS/dg_critic_study_train_rollouts_20261014/rollout_dataset" \
  --heldout-dataset "$RUNS/dg_critic_study_heldout_rollouts_20261014/rollout_dataset" \
  --branch-dataset "$RUNS/dg_critic_study_branches_20261014_v2" \
  --epochs 20 --eval-every 20 --arms actor_trunk_twin,frozen_belief_gain_twin \
  --family-alpha 0.008333333333333333 --max-updates 640 \
  --output "$RUNS/dg_twin_critic_study_v7_20261020"

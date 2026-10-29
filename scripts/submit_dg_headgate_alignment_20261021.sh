#!/usr/bin/env bash
# The DG retirement test (NOTES 2026-10-21, "decision tokens carry large Q
# gaps ..."): per-pool actor gradients at the start policy on the v7 s520
# twin critic and the eight held-out pools job 14345 read, for
#   sampled_baseline, control_variate  the unbiased references
#   delightful_head_gate               DG gated on the action head's A(a_t)
#   delightful_shuffled_head_gate      the same gate with A permuted within
#                                      surprisal-and-span strata (seeded)
# plus each pool's entropy-bonus gradient for the entropy projections.
# Analyze with scripts/analyze_estimator_alignment.py --entropy.
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
RUNS="$REPO/postraining/runs"
label=v7_s520_headgate
mlq submit --name "estimator_alignment_$label" --cwd "$REPO" --max-parallel-runs 1 --priority 11 \
  --time-limit 45m --idempotency-key "estimator_alignment_20261021_r1_$label" -- \
  "$REPO/.venv/bin/python" -u "$REPO/scripts/estimator_gradient_alignment.py" \
  --checkpoint "$RUNS/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt" \
  --rollouts "$RUNS/dg_critic_study_heldout_rollouts_20261014/rollout_dataset" \
  --pools 0 1 2 3 4 5 6 7 --output "$RUNS/estimator_alignment_20261020" --label "$label" \
  --critic "$RUNS/dg_twin_critic_study_v7_20261020/actor_trunk_twin/checkpoints/step_000520.pt" \
  --estimators sampled_baseline control_variate delightful_head_gate delightful_shuffled_head_gate \
  --gate-shuffle-seed 20261021 --entropy-gradient

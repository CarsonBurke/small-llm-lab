#!/usr/bin/env bash
# Phase 2 of estimator_alignment_20261020: per-pool actor gradients at the
# step-240 policies of the two drills, on their own fresh held-out pools
# (scripts/submit_estimator_alignment_rollouts_20261020.sh), with the
# critics those drills held at step 240.
#   sb240_v4      sampled-baseline drill policy, its online v4 critic (its
#                 own source snapshot): sampled_baseline, control_variate,
#                 all_action
#   aa240_v7      all-action drill policy, its online v7 twin critic:
#                 sampled_baseline, control_variate, all_action,
#                 all_action_nogain
#   aa240_v7s60   the same policy with the critic it started from, which
#                 separates the policy's drift from the critic's
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
RUNS="$REPO/postraining/runs"
OUT="$RUNS/estimator_alignment_20261020"
SB="$RUNS/mac_sampled_baseline_allstates_drill_20261019/snapshots/step_000240.pt"
AA="$RUNS/mac_twin_v7_gain_allaction_drill_20261020/snapshots/step_000240.pt"
submit() {
  local label=$1 rollouts=$2; shift 2
  mlq submit --name "estimator_alignment_$label" --cwd "$REPO" --max-parallel-runs 1 --priority 10 \
    --time-limit 45m --idempotency-key "estimator_alignment_20261020_r3_$label" -- \
    "$REPO/.venv/bin/python" -u "$REPO/scripts/estimator_gradient_alignment.py" \
    --checkpoint "$RUNS/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt" \
    --rollouts "$RUNS/$rollouts/rollout_dataset" --pools 0 1 2 3 4 5 6 7 --output "$OUT" \
    --label "$label" "$@"
}
submit sb240_v4 estimator_alignment_rollouts_20261020_sb_s240 \
  --source-root "$REPO/.worktrees/src_baseline_20261019" --policy-state "$SB" --critic "$SB" \
  --estimators sampled_baseline control_variate all_action
submit aa240_v7 estimator_alignment_rollouts_20261020_aa_s240 --policy-state "$AA" --critic "$AA" \
  --estimators sampled_baseline control_variate all_action all_action_nogain
submit aa240_v7s60 estimator_alignment_rollouts_20261020_aa_s240 --policy-state "$AA" \
  --critic "$RUNS/dg_twin_critic_study_v7_20261020/actor_trunk_twin/checkpoints/step_000060.pt" \
  --estimators control_variate all_action

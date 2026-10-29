#!/usr/bin/env bash
# Per-pool actor gradients of MAC's estimators at the start policy, for
# scripts/analyze_estimator_alignment.py: the eight held-out start-policy
# pools are eight independent updates' data. Three critics:
#   v4      the sampled_baseline drill's warm start (dg_full_critic_convergence
#           actor_trunk), under that drill's own source snapshot, which still
#           reads critic schema v4
#   v7_s60  the v7 twin study at update 60, the all-action drill's warm start
#   v7_s520 the same arm at update 520, its last checkpoint before the cancel
# Under each: sampled_baseline and control_variate (unbiased references) and
# all_action; under v7 also all_action_nogain (the belief gain zeroed).
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
RUNS="$REPO/postraining/runs"
OUT="$RUNS/estimator_alignment_20261020"
COMMON=(--checkpoint "$RUNS/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt"
        --rollouts "$RUNS/dg_critic_study_heldout_rollouts_20261014/rollout_dataset"
        --pools 0 1 2 3 4 5 6 7 --output "$OUT")
STUDY="$RUNS/dg_twin_critic_study_v7_20261020/actor_trunk_twin/checkpoints"
submit() {
  local label=$1; shift
  mlq submit --name "estimator_alignment_$label" --cwd "$REPO" --max-parallel-runs 1 --priority 10 \
    --time-limit 1h --idempotency-key "estimator_alignment_20261020_r2_$label" -- \
    "$REPO/.venv/bin/python" -u "$REPO/scripts/estimator_gradient_alignment.py" "${COMMON[@]}" \
    --label "$label" "$@"
}
submit v4 --source-root "$REPO/.worktrees/src_baseline_20261019" \
  --critic "$RUNS/dg_full_critic_convergence_20261015/actor_trunk/checkpoint.pt" \
  --estimators sampled_baseline control_variate all_action
submit v7_s60 --critic "$STUDY/step_000060.pt" \
  --estimators sampled_baseline control_variate all_action all_action_nogain
submit v7_s520 --critic "$STUDY/step_000520.pt" \
  --estimators sampled_baseline control_variate all_action all_action_nogain

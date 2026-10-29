#!/usr/bin/env bash
# Answer-level DG against its RLOO control (NOTES.md, "Answer-level DG:
# pre-registration", 2026-10-21). The fresh-rate memory SB drill's argv
# (scripts/submit_mac_sampled_baseline_memory_fresh_lr_drill_20261021.sh,
# run mac_sampled_baseline_memory_fresh_lr_drill_20261021_r2) with one
# change, the estimator:
#   ARM=rloo_sequence           one coefficient per trajectory, the group's
#                               leave-one-out reward advantage U;
#   ARM=delightful_answer_gate  U under the answer-level delight gate
#                               sigmoid(U * -log(n_k / 16)).
# Neither reads the critic; it trains beside them for telemetry exactly as
# under SB. Session plumbing differs, not the optimization: sessions of 75
# minutes (host CPU is congested; --max-train-hours 1.05 leaves 12 minutes
# for startup, the last pool, its bench and the checkpoints), priority 10,
# and snapshots every 60 steps (the greedy benches read steps 60/120/.../360;
# 20-step snapshots would be 4 x 18 x 1.2 GB).
# Usage: submit_answer_gate_drill_20261021.sh ARM SEED SESSION
#   SESSION 1 starts; 2+ resume. Optional AFTER=<job id>: start only after
#   that job succeeds. Idempotency keys are _r2: the _r1 jobs (14453-14460)
#   were skipped behind the failed smoke 14452 before any started.
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
RUNS="$REPO/postraining/runs"
ARM=${1:?rloo_sequence or delightful_answer_gate}
SEED=${2:?seed}
SESSION=${3:?session}
case "$ARM" in
  rloo_sequence|delightful_answer_gate) ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
NAME="mac_${ARM}_memory_drill_20261021_s${SEED}"
RESUME=()
INIT=(--study-critic-init "$RUNS/dg_twin_critic_study_v7_20261020/actor_trunk_twin/checkpoints/step_000520.pt")
if [[ "$SESSION" != 1 ]]; then
  RESUME=(--resume "$RUNS/$NAME/latent_vapo_checkpoint.pt")
  INIT=()
fi
exec mlq submit --name "${NAME}_s$SESSION" --cwd "$REPO" --max-parallel-runs 1 --priority 10 \
  --time-limit 75m ${AFTER:+--after-success "$AFTER"} --idempotency-key "${NAME}_s${SESSION}_r2" -- \
  "$REPO/.venv/bin/python" -u -m postraining.train_latent_vapo \
  --checkpoint "$RUNS/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt" \
  --math-data "$REPO/postraining/data/drill_rl_study_train_20261014_v1.parquet" \
  --rl-mixture-manifest '' \
  --bench-data "$REPO/postraining/data/drill_rl_study_heldout_20261014_v1.parquet" \
  --bench-every 20 --bench-max-rows 512 --bench-samples 16 --bench-max-tokens 2048 \
  --aime-every 0 --aime-max-tokens 2048 --bpb-every 0 \
  --reasoning-mode memory --think-tokens --answer-fence --think-min-tokens 33 --answer-tokens 24 \
  --python-reward-mode test-fraction --randomize-choice-options \
  --prompt-tokens 512 --continuation-tokens 2048 --steps 360 \
  --prompts-per-rollout 64 --prompts-per-minibatch 64 --samples-per-prompt 16 --rollout-groups 64 \
  --mean-actor-critic --mac-estimator "$ARM" --mac-vocab-chunk 512 \
  --critic-q-head-hidden 1024 --critic-q-twin --critic-init actor "${INIT[@]}" \
  --trunk-optimizer muon --learning-rate 2e-5 --critic-learning-rate 2e-5 \
  --muon-learning-rate 4.8333333333333334e-05 --critic-muon-learning-rate 4.8333333333333334e-05 \
  --critic-action-head-learning-rate 1e-4 --critic-q-direction-learning-rate 1.953125e-07 \
  --fresh-learning-rate 3e-3 \
  --q-critic-lambda 1 --gae-lambda-alpha 0.05 --value-warmup-steps 0 \
  --no-zero-reward-actor-freeze --zero-reward-stop-pools 3 \
  --temperature 1 --top-p 1 --eval-temperature 1 --eval-top-p 1 \
  --compile-replay --replay-max-trajectories 128 --replay-attention-budget 16777216 \
  --replay-bucket 64 --replay-slot-budget 24576 \
  --eval-batch-trajectories 1024 --eval-tail-batch 16 --rollout-tail-batch 16 \
  --checkpoint-interval-seconds 300 --snapshot-every 60 --rollout-sample-every 25 \
  --max-train-hours 1.05 --seed "$SEED" --bench-stop-floor 0.55 --bench-stop-evals 2 \
  "${RESUME[@]}" --output "$RUNS/$NAME"

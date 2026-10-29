#!/usr/bin/env bash
# The stage-1 run (mac_online_allstates_drill_20261019) with only the token
# Q estimator changed: ESTIMATOR=control_variate adds the sampled token's
# Monte Carlo residual G - Q(s,a_t) to all-action MAC (unbiased for any
# critic); ESTIMATOR=sampled_baseline is its control, that sampled term
# alone with G - E_pi Q. Same SFT actor, study critic, pools, rates, bench
# and stop rule as stage 1.
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
ESTIMATOR=${ESTIMATOR:?set ESTIMATOR to control_variate or sampled_baseline}
case "$ESTIMATOR" in
  control_variate) NAME=mac_cv_allstates_drill_20261019 ;;
  sampled_baseline) NAME=mac_sampled_baseline_allstates_drill_20261019 ;;
  *) echo "unknown ESTIMATOR $ESTIMATOR" >&2; exit 2 ;;
esac
exec mlq submit --name "$NAME" --cwd "$REPO" --max-parallel-runs 1 --priority 10 \
  --time-limit 10h --idempotency-key "$NAME" -- \
  "$REPO/.venv/bin/python" -u -m postraining.train_latent_vapo \
  --checkpoint "$REPO/postraining/runs/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt" \
  --math-data "$REPO/postraining/data/drill_rl_study_train_20261014_v1.parquet" \
  --rl-mixture-manifest "" \
  --bench-data "$REPO/postraining/data/drill_rl_study_heldout_20261014_v1.parquet" \
  --bench-every 20 --bench-max-rows 512 --bench-samples 16 --bench-max-tokens 2048 \
  --aime-every 0 --aime-max-tokens 2048 --bpb-every 0 \
  --reasoning-mode cot --think-tokens --answer-fence --think-min-tokens 33 \
  --answer-tokens 24 --python-reward-mode test-fraction --randomize-choice-options \
  --prompt-tokens 512 --continuation-tokens 2048 \
  --steps 360 --prompts-per-rollout 64 --prompts-per-minibatch 64 \
  --samples-per-prompt 16 --rollout-groups 64 \
  --mean-actor-critic --mac-estimator "$ESTIMATOR" --mac-vocab-chunk 512 \
  --critic-q-head-hidden 1024 --critic-init actor \
  --study-critic-init "$REPO/postraining/runs/dg_full_critic_convergence_20261015/actor_trunk/checkpoint.pt" \
  --trunk-optimizer muon --learning-rate 2e-5 --critic-learning-rate 2e-5 \
  --muon-learning-rate 4.8333333333333334e-05 --critic-muon-learning-rate 4.8333333333333334e-05 \
  --critic-action-head-learning-rate 1e-4 --critic-q-direction-learning-rate 1.953125e-07 \
  --q-critic-lambda 1 --gae-lambda-alpha 0.05 \
  --no-zero-reward-actor-freeze --zero-reward-stop-pools 3 \
  --temperature 1 --top-p 1 --eval-temperature 1 --eval-top-p 1 \
  --compile-replay --replay-max-trajectories 128 --replay-attention-budget 16777216 \
  --replay-bucket 64 --replay-slot-budget 24576 \
  --eval-batch-trajectories 1024 --eval-tail-batch 16 --rollout-tail-batch 16 \
  --checkpoint-interval-seconds 300 --snapshot-every 20 --rollout-sample-every 25 \
  --max-train-hours 8 --seed 1337 \
  --bench-stop-floor 0.55 --bench-stop-evals 2 \
  --output "$REPO/postraining/runs/$NAME"

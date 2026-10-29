#!/usr/bin/env bash
# All-action MAC with twin token-Q heads plus the pessimistic Q prior,
# beta 1e-4, on the drill pool. Matched to mac_twin_allaction_drill_20261019
# (job 14177) except --critic-q-pessimism 1e-4: each head also minimizes
# beta * sum_a relu(A(s, a))^2 under the current policy, so a token the
# policy all but never takes cannot read better than its value (a positive
# advantage keeps pi/(pi + beta) of itself; beta equals the census's
# off-support threshold). Replaces CQL(H), which inflated the critic's
# score scale (job 14195, NOTES 2026-10-20). Same twin study critic as warm
# start, no warmup; support/* and q_prior/* log the effect in-run.
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
NAME=mac_twin_qprior_allaction_drill_20261020
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
  --mean-actor-critic --mac-estimator all_action --mac-vocab-chunk 512 \
  --critic-q-head-hidden 1024 --critic-q-twin --critic-q-pessimism 1e-4 --critic-init actor \
  --study-critic-init "$REPO/postraining/runs/dg_twin_critic_study_20261019/actor_trunk_twin/checkpoint.pt" \
  --trunk-optimizer muon --learning-rate 2e-5 --critic-learning-rate 2e-5 \
  --muon-learning-rate 4.8333333333333334e-05 --critic-muon-learning-rate 4.8333333333333334e-05 \
  --critic-action-head-learning-rate 1e-4 --critic-q-direction-learning-rate 1.953125e-07 \
  --gae-lambda-alpha 0.05 \
  --no-zero-reward-actor-freeze --zero-reward-stop-pools 3 \
  --temperature 1 --top-p 1 --eval-temperature 1 --eval-top-p 1 \
  --compile-replay --replay-max-trajectories 128 --replay-attention-budget 16777216 \
  --replay-bucket 64 --replay-slot-budget 24576 \
  --eval-batch-trajectories 1024 --eval-tail-batch 16 --rollout-tail-batch 16 \
  --checkpoint-interval-seconds 300 --snapshot-every 20 --rollout-sample-every 25 \
  --max-train-hours 8 --seed 1337 \
  --bench-stop-floor 0.55 --bench-stop-evals 2 \
  --output "$REPO/postraining/runs/$NAME"

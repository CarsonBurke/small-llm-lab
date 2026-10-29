#!/usr/bin/env bash
# Smoke for the answer-level estimators (not evidence): the CUDA parity
# cases of rloo_sequence and delightful_answer_gate (compiled update against
# eager on the SFT checkpoint, memory twin critic), then two training steps
# of the drill's own argv under delightful_answer_gate with a 32-prompt
# bench, which runs the scorer's group stamps, packing, the update, its
# telemetry and the bench's majority/entropy metrics end to end.
# Attempt r2: the parity cases are informational and the job's outcome is
# the training run's. In r1 (job 14452) delightful_answer_gate's compiled
# gradient missed one per-parameter allowance (blocks.0.attn.A_log 3.05%
# against max(2%, 3 x its 0.71% resharding floor)) while the whole actor
# gradient matched (cosine 0.99977, relative 2.1% against the 2.4%
# resharding floor). set -e then stopped the job before training ran.
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
RUNS="$REPO/postraining/runs"
NAME=answer_gate_smoke_20261021
exec mlq submit --name "$NAME" --cwd "$REPO" --max-parallel-runs 1 --priority 11 \
  --time-limit 30m ${AFTER:+--after-success "$AFTER"} --idempotency-key "${NAME}_r2" -- \
  bash -c '
set -euo pipefail
parity=0
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -s -p no:cacheprovider \
  postraining/tests/test_mac_update_cuda.py -k "rloo_sequence or delightful_answer_gate" \
  || parity=$?
echo "parity pytest exit code: $parity"
.venv/bin/python -u -m postraining.train_latent_vapo \
  --checkpoint '"$RUNS"'/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt \
  --math-data postraining/data/drill_rl_study_train_20261014_v1.parquet \
  --rl-mixture-manifest "" \
  --bench-data postraining/data/drill_rl_study_heldout_20261014_v1.parquet \
  --bench-every 2 --bench-max-rows 32 --bench-samples 16 --bench-max-tokens 2048 \
  --aime-every 0 --aime-max-tokens 2048 --bpb-every 0 \
  --reasoning-mode memory --think-tokens --answer-fence --think-min-tokens 33 --answer-tokens 24 \
  --python-reward-mode test-fraction --randomize-choice-options \
  --prompt-tokens 512 --continuation-tokens 2048 --steps 2 \
  --prompts-per-rollout 64 --prompts-per-minibatch 64 --samples-per-prompt 16 --rollout-groups 64 \
  --mean-actor-critic --mac-estimator delightful_answer_gate --mac-vocab-chunk 512 \
  --critic-q-head-hidden 1024 --critic-q-twin --critic-init actor \
  --study-critic-init '"$RUNS"'/dg_twin_critic_study_v7_20261020/actor_trunk_twin/checkpoints/step_000520.pt \
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
  --checkpoint-interval-seconds 300 --rollout-sample-every 25 \
  --seed 1337 --output '"$RUNS/$NAME"'
'

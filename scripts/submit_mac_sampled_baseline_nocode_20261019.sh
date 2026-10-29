#!/usr/bin/env bash
# Stage 2 of docs/mac_online_critic_plan_20261019.md: the sampled_baseline
# token-Q estimator that held on the drill pool
# (mac_sampled_baseline_allstates_drill_20261019), on the reviewed RL
# mixture without code or science MC. Science MC is 4-option and sat at
# chance (0.21-0.24) for all 950 steps of the 10-13 nocode run while
# carrying 5-10x the other sources' within-group reward spread; under a
# sampled estimator it would supply most of the gradient as guessing noise.
# Corpus, evaluations and prompt budget follow the 10-13 nocode runs; the
# actor is the SFT checkpoint in cot, the critic a fresh token-Q head on an
# actor-copy trunk warmed for 50 value-only pools, rates as the drill run.
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
NAME=mac_sampled_baseline_nocode_nosci_20261019_v2
exec mlq submit --name "$NAME" --cwd "$REPO" --max-parallel-runs 1 --priority 10 \
  --time-limit 16h --idempotency-key "$NAME" -- \
  "$REPO/.venv/bin/python" -u -m postraining.train_latent_vapo \
  --checkpoint "$REPO/postraining/runs/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt" \
  --rl-mixture-manifest "$REPO/postraining/data/reviewed_candidate_20260926_v2/mixture.manifest.json" \
  --exclude-rl-sources ultradata_code_l3,science_mc \
  --math-data "$REPO/postraining/data/dapo-math-17k.parquet" \
  --aime-data "$REPO/postraining/data/aime-2024.parquet" \
  --bench-data "$REPO/postraining/data/deepmind-interpolate-easy.parquet" \
  --bench-every 20 --aime-every 20 --bpb-every 20 \
  --bench-max-tokens 2048 --aime-max-tokens 2048 \
  --reasoning-mode cot --think-tokens --answer-fence --answer-tokens 24 \
  --python-reward-mode test-fraction --randomize-choice-options \
  --prompt-tokens 256 --continuation-tokens 2048 \
  --steps 950 --prompts-per-rollout 64 --prompts-per-minibatch 64 \
  --samples-per-prompt 16 --rollout-groups 64 \
  --mean-actor-critic --mac-estimator sampled_baseline --mac-vocab-chunk 512 \
  --critic-q-head-hidden 1024 --critic-init actor \
  --value-warmup-scope critic --value-warmup-steps 50 \
  --trunk-optimizer muon --learning-rate 2e-5 --critic-learning-rate 2e-5 \
  --muon-learning-rate 4.8333333333333334e-05 --critic-muon-learning-rate 4.8333333333333334e-05 \
  --critic-action-head-learning-rate 1e-4 --critic-q-direction-learning-rate 1.953125e-07 \
  --q-critic-lambda 1 --gae-lambda-alpha 0.05 \
  --no-zero-reward-actor-freeze --zero-reward-stop-pools 32 \
  --temperature 1 --top-p 1 --eval-temperature 1 --eval-top-p 1 \
  --compile-replay --replay-max-trajectories 128 --replay-attention-budget 16777216 \
  --replay-bucket 64 --replay-slot-budget 24576 \
  --eval-batch-trajectories 1024 --eval-tail-batch 16 --rollout-tail-batch 16 \
  --checkpoint-interval-seconds 300 --snapshot-every 50 --rollout-sample-every 25 \
  --max-train-hours 14 --seed 1337 \
  --output "$REPO/postraining/runs/$NAME"

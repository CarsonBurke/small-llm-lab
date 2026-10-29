#!/usr/bin/env bash
# mac_sampled_baseline_memory_drill_20261020 (jobs 14311/14316) with the
# memory trained as a fresh module instead of a fine-tune. The one change:
#   --fresh-learning-rate 3e-3  the memory's writer/reader (and the critic's
#                               memory reader) get their own optimizers in
#                               pretraining's partition: matrices under Muon
#                               at 3e-3 * (0.025/0.015) * 1.45 = 7.25e-3,
#                               gate biases under AdamW at 3e-3 -- a fifth of
#                               pretraining's from-scratch rates. Under the
#                               shared 2e-5 AdamW the writer moved 3% of its
#                               init in 340 steps and the read reached 2% of
#                               the input RMS.
# Everything else, including the v7 twin study critic start, is the memory
# drill's, so the bench curve compares step for step against it and 14165.
# Sessions: $1 = 1 starts, 2+ resume (as the memory drill). Optional
# AFTER=<job id>: start only after that job succeeds.
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
RUNS="$REPO/postraining/runs"
# _r2: the first attempt (job 14391) crashed at its first train step on a
# missing MemoryReader import in the weight telemetry; its directory is kept.
NAME=mac_sampled_baseline_memory_fresh_lr_drill_20261021_r2
SESSION=${1:-1}
RESUME=()
if [[ "$SESSION" != 1 ]]; then
  RESUME=(--resume "$RUNS/$NAME/latent_vapo_checkpoint.pt")
fi
INIT=(--study-critic-init "$RUNS/dg_twin_critic_study_v7_20261020/actor_trunk_twin/checkpoints/step_000520.pt")
if [[ "$SESSION" != 1 ]]; then
  INIT=()
fi
exec mlq submit --name "${NAME}_s$SESSION" --cwd "$REPO" --max-parallel-runs 1 --priority 11 \
  --time-limit 45m ${AFTER:+--after-success "$AFTER"} --idempotency-key "${NAME}_s${SESSION}_r1" -- \
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
  --mean-actor-critic --mac-estimator sampled_baseline --mac-vocab-chunk 512 \
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
  --checkpoint-interval-seconds 300 --snapshot-every 20 --rollout-sample-every 25 \
  --max-train-hours 0.62 --seed 1337 --bench-stop-floor 0.55 --bench-stop-evals 2 \
  "${RESUME[@]}" --output "$RUNS/$NAME"

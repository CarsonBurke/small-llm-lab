#!/usr/bin/env bash
# Reviewed STEM/code/QA RL. Fresh runs: 950 joint + 50 critic warmup updates.
# Usage: launch_reviewed_rl.sh [run-name] [critic-checkpoint] [joint-steps]
# Code earns the fraction of sandboxed test cases passed; exact success remains
# separately reported. Choice options are randomized on each presentation.
set -euo pipefail
cd "$(dirname "$0")/.."
run=${1:-kda8_pg_reviewed_v2_partial_2048}
critic_checkpoint=${2:-}
steps=${3:-950}
init_args=(--value-warmup-steps 50)
if [[ -n "$critic_checkpoint" ]]; then
  init_args=(--critic-only-init "$critic_checkpoint" --value-warmup-steps 0)
fi
if [[ -e "postraining/runs/$run" ]]; then
  echo "Refusing to overwrite existing run: $run" >&2
  exit 1
fi
queue_args=(--name "$run" --cwd "$PWD" --max-parallel-runs 1 --priority 1 --time-limit 12h)
if [[ -n "${AFTER_SUCCESS:-}" ]]; then
  queue_args+=(--after-success "$AFTER_SUCCESS")
fi
mlq submit "${queue_args[@]}" -- \
  .venv/bin/python -u -m postraining.train_latent_vapo \
  --checkpoint postraining/runs/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt \
  --output "postraining/runs/$run" \
  --rl-mixture-manifest postraining/data/reviewed_candidate_20260926_v2/mixture.manifest.json \
  --reasoning-mode cot --think-tokens --answer-fence \
  --no-delightful-policy-gradient \
  --python-reward-mode test-fraction --randomize-choice-options \
  --prompt-tokens 256 --continuation-tokens 2048 \
  --bench-max-tokens 2048 --aime-max-tokens 2048 --answer-tokens 24 \
  --steps "$steps" "${init_args[@]}" \
  --prompts-per-rollout 64 --prompts-per-minibatch 64 \
  --samples-per-prompt 16 --rollout-groups 32 \
  --bench-every 20 --aime-every 20 --bpb-every 20 \
  --rollout-sample-every 25 --zero-reward-stop-pools 32 \
  --checkpoint-interval-seconds 300 --seed 1337

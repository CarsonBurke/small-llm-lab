#!/usr/bin/env bash
# The frozen-policy token-Q critic fit of dg_full_critic_convergence_20261015
# (arm actor_trunk, 640 updates) with TD3/TD7 twin heads: arm
# actor_trunk_twin, two independently initialized token-Q heads on one
# actor-copy trunk, each regressing the same Monte Carlo target, losses
# summed. Gates score the heads' mean; per-head EV and the heads'
# disagreement are logged beside them. Same rollouts, branches, budget,
# evaluation cadence and family alpha as the single-head fit. The SFT
# checkpoint is the main-checkout copy of the one that run read
# (.worktrees/dg-critic-screen-inputs-20261014/sft_final_model.pt; both
# SHA256 c882966296b9109a828af67cc2ad4a3851a1233411727bb0ec0479b2a04b2742),
# which the study checks against the datasets' collection binding.
# Its checkpoint warm-starts a --critic-q-twin RL run (--study-critic-init).
set -euo pipefail
REPO=/home/marvin/Documents/repositories/parameter-golf
NAME=twin_critic_study_20261019
RUNS="$REPO/postraining/runs"
exec mlq submit --name "$NAME" --cwd "$REPO" --max-parallel-runs 1 --priority 10 \
  --time-limit 3h --idempotency-key "$NAME" -- \
  "$REPO/.venv/bin/python" -u -m postraining.critic_study \
  --checkpoint "$RUNS/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt" \
  --train-dataset "$RUNS/dg_critic_study_train_rollouts_20261014/rollout_dataset" \
  --heldout-dataset "$RUNS/dg_critic_study_heldout_rollouts_20261014/rollout_dataset" \
  --branch-dataset "$RUNS/dg_critic_study_branches_20261014_v2" \
  --epochs 20 --eval-every 20 --arms actor_trunk_twin \
  --family-alpha 0.008333333333333333 --max-updates 640 \
  --output "$RUNS/dg_twin_critic_study_20261019"

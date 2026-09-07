#!/usr/bin/env bash
# Rerun of ablation_results/minicpm_gate10_nora_mb4_v3_4k at 10k generation length.
# Every value pinned explicitly (recovered from the 4k checkpoint args); nothing
# rides argparse defaults, so future default changes cannot silently alter this.
# Deliberate changes vs the 4k run: --output, --max-new-tokens 10000.
# Carried as-is (explicitly pinned): --lora-initialization nora, --replay-token-budget 8192,
# --checkpoint-interval-seconds 300, --steps 10, --min-rollout-tokens-per-second 1.0
# (default floor is 4000; 1.0 effectively disables the throughput gate).
# Note: trainer has no --thinking flag; thinking is always on (set_defaults).
set -euo pipefail
exec .venv/bin/python -u -m postraining.train_minicpm_vapo \
  --model openbmb/MiniCPM5-1B \
  --revision 87179e5c1f455ef22e6223592d2d61351b525bfc \
  --data postraining/data/dapo-math-17k.parquet \
  --output ablation_results/minicpm_gate10_nora_mb4_v3_10k \
  --steps 10 \
  --value-warmup-steps 10 \
  --prompts-per-rollout 4 \
  --samples-per-prompt 16 \
  --rollout-physical-batch-size 0 \
  --prompt-tokens 1024 \
  --max-new-tokens 10000 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 20 \
  --lora-rank 16 \
  --lora-alpha 32.0 \
  --lora-initialization nora \
  --critic-width 256 \
  --actor-lr 1e-6 \
  --critic-lr 1e-5 \
  --nextlat-horizon 2 \
  --nextlat-projection-factor 1.6 \
  --nextlat-samples 64 \
  --nextlat-kl-chunk-tokens 16 \
  --nextlat-mse-coefficient 1.0 \
  --nextlat-kl-coefficient 1.0 \
  --train-nextlat \
  --nextlat-trunk-balance parameter \
  --gradient-clip-norm 1.0 \
  --optimizer-minibatches 4 \
  --post-update-kl-interval 10 \
  --ppo-epochs 1 \
  --clip-low 0.2 \
  --clip-high 0.28 \
  --value-coefficient 1.0 \
  --replay-token-budget 8192 \
  --replay-max-trajectories 16 \
  --logit-chunk-tokens 128 \
  --compile-rollout \
  --replay-checkpoint-interval 4 \
  --replay-attention-backend sdpa \
  --fast-rollout \
  --min-rollout-tokens-per-second 1.0 \
  --device-telemetry \
  --device-telemetry-interval-ms 250 \
  --device-power-floor 400.0 \
  --checkpoint-interval-seconds 300 \
  --resume postraining/runs/minicpm_gate10_nora_warmup_v3_4k/vapo_adapter_checkpoint.pt \
  --gate-min-positive-trajectories 1 \
  --gate-min-positive-groups 1 \
  --gate-max-truncation-fraction 0.5 \
  --seed 1337

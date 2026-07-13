# Fresh LeJEPA + VAPO

This directory contains the isolated post-training pipeline and the VAPO v3
paper. It uses the official DAPO-Math-17k prompts and DAPO AIME 2024 file.

```bash
python3 ablation.py --steps 2000 --name fresh_lejepa_2k --script fresh_lejepa_train.py

# Required learnability gate before spending optimizer compute. This exits 2
# unless the frozen policy produces a positive answer group and at least half
# of sampled responses reach EOS within the gate context.
python3 -m postraining.train_vapo \
  --checkpoint ablation_results/fresh_lejepa_2k/pretraining_checkpoint.pt \
  --rollout-only --gate-prompts 128 --max-new-tokens 768

python3 -m postraining.train_vapo \
  --checkpoint ablation_results/fresh_lejepa_2k/pretraining_checkpoint.pt \
  --steps 5000

python3 -m postraining.eval_aime \
  --checkpoint postraining/runs/fresh_lejepa_vapo/checkpoints/step_00128.pt \
  --step 128

python3 -m postraining.plot_accuracy
```

The pretraining run archives its raw/quantized states, optimizer and RNG state,
resolved metadata, tokenizer hash, and source under its named ablation folder.

The VAPO defaults are 50 critic-only value-pretraining updates, 512 distinct
prompts with 16 responses each, two PPO epochs with true 512-trajectory
optimizer minibatches, length-adaptive
policy GAE, lambda-one critic targets, token-level PPO, asymmetric clipping
`(0.20, 0.28)`, and positive-response NLL weight 0.1. Training samples with
temperature 1/top-p 1; scheduled AIME evaluation uses temperature 1/top-p 0.7.
Rollouts are accumulated in CPU memory in bounded `--rollout-prompt-chunk`
GPU shards. Each optimizer minibatch is likewise accumulated from bounded
`--microbatch-trajectories` GPU shards; these controls change memory use, not
the algorithmic batch sizes. Checkpoints occur only after a complete rollout
and all of its PPO epochs, making every checkpoint an exact resume boundary.
AIME itself is measured at exact requested optimizer steps
(100, 200, ...) and preserves training RNG, so charts retain the paper's x-axis
even though the nearest resumable checkpoint is later. Consequently, the final
checkpoint step can slightly exceed `--steps`. AIME step 0 and every exact
100-step boundary are logged to JSONL and TensorBoard by default.

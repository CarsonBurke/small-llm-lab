# Fresh LeJEPA + VAPO

This directory contains the isolated post-training pipeline and the VAPO v3
paper. It uses the official DAPO-Math-17k prompts and DAPO AIME 2024 file.

```bash
python3 ablation.py --steps 2000 --name fresh_lejepa_2k --script fresh_lejepa_train.py

python3 -m postraining.train_vapo \
  --checkpoint ablation_results/fresh_lejepa_2k/pretraining_checkpoint.pt \
  --steps 5000

python3 -m postraining.eval_aime \
  --checkpoint postraining/runs/fresh_lejepa_vapo/checkpoints/step_00100.pt \
  --step 100

python3 -m postraining.plot_accuracy
```

The pretraining run archives its raw/quantized states, optimizer and RNG state,
resolved metadata, tokenizer hash, and source under its named ablation folder.

The VAPO defaults are 50 critic-only value-pretraining updates, 16 responses
per prompt, two PPO epochs with 512-trajectory minibatches, length-adaptive
policy GAE, lambda-one critic targets, token-level PPO, asymmetric clipping
`(0.20, 0.28)`, and positive-response NLL weight 0.1. Training samples with
temperature 1/top-p 1; scheduled AIME evaluation uses temperature 1/top-p 0.7.
Automatic prompt calibration picks the largest tested prompt count below 95%
of physical VRAM at the configured maximum response length, without gradient
accumulation. AIME step 0 and every 100 update steps are logged to JSONL and
TensorBoard by default.

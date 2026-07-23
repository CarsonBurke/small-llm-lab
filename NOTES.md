# Parameter Golf - Working Notes

## Hardware
- RTX 5090 (32GB GDDR7), single GPU
- Throughput: **588ms/step** (vs 43.5ms/step on 8×H100)
- Peak VRAM: 10.9GB (baseline model)
- Ratio: 8×H100 is ~13.5× faster

## Baseline Reference (official log, 8×H100)
| Step | BPB | Wall time |
|------|------|-----------|
| 0 | 4.0978 | 0s |
| 1000 | 1.3805 | 43s |
| 2000 | 1.3213 | 87s |
| 5000 | 1.2719 | 218s |
| 10000 | 1.2477 | 435s |
| 13200 | 1.2281 | 575s |
| 13780 | 1.2172 | 600s (cap) |

Post-quantization submission score: **1.2244 BPB**

## Current SOTA
**1.0810 BPB** (SP8192 + 3-Layer Recurrence + Parallel Residuals + Legal TTT)

## Ablation Setup
- Default ablation: **2000 steps** (~20 min on 5090)
- Default warmdown: **1200 steps** (starts at step 800 in a 2000-step run)
- Expected BPB at 2000 steps: ~1.30 (`baseline_2k`: 1.2967)
- Val every 20 steps by default for ablations
- Metrics JSONL: `ablation_results/<run>/metrics.jsonl`
- Tensorboard: `http://localhost:6006` (logs in `tb_logs/`)
- Results JSON: `ablation_results/<run>/result.json`
- Plot: `python3 plot_ablations.py`

## Time Estimates (RTX 5090)
| Steps | Time |
|-------|------|
| 500 | ~5 min |
| 2000 | ~20 min |
| 5000 | ~49 min |
| 13780 | ~135 min |

## Commands
```bash
# Quick ablation (2000 steps, default)
python3 ablation.py

# Custom step count
python3 ablation.py --steps 2000 --name baseline_2k

# Sweep learning rates
python3 ablation.py --sweep lr --steps 2000

# Compare results
python3 ablation.py --compare

# Plot
python3 plot_ablations.py

# Full baseline (match official)
python3 ablation.py --steps 13780 --name baseline_full

# Manual single run with env overrides
python3 ablation.py --steps 2000 --name my_test --env MODEL_DIM=640 NUM_HEADS=10
```

## Post-training: Nano GPT backbones + reasoning modes (2026-07-22)

`postraining/train_latent_vapo.py` now supports nano backbones and three reasoning modes:

- `--reasoning-mode latent` (default) — THINK/EMIT latent VAPO, unchanged behavior.
- `--reasoning-mode cot` — gate pinned to EMIT, full token budget, token-only PPO.
- `--reasoning-mode none` — pinned EMIT, answer-only budget (`--answer-tokens`, default 24),
  `"\nAnswer:"` teacher-forced onto the prompt tail and rejoined before reward parsing.

Backbones: fresh PoPE (unchanged) plus `nanogpt_mini_v1` / `nanogpt_mini_tieddot_v1`
(via `postraining/nano_backbone.py`; belief = post-norm final hidden state; critic gets a
fresh nano trunk with nano-native init). Context budgets are backbone-derived:
fresh 5120/1024/1024/4096 (ctx/prompt/response/stream), nano 1024/512/256/512
(no RoPE extrapolation on nano). Checkpoints and manifests record `reasoning_mode`
via mode-tagged rollout policy schemas, so cross-mode resume fails loudly.

Nano pretraining scripts save `logs/<RUN_ID>_final_model.pt` (model classes extracted
to `nanogpt_mini_model.py`). RL prerequisite: pretrain on `mathmix_v4_sp1024`
(`DATA_PATH` env), then gate with `python3 -m postraining.dapo_hit_rate_probe
--checkpoint <ckpt>` (needs ≥1 positive group + within-group reward variance).

Validation (2026-07-22, all via mlq 271-290; tests 280 passed / 5 skipped):
- Mathmix 2k RL bases: `logs/nanomini_mathmix_2k_final_model.pt` (val_bpb 1.3589),
  `logs/nanomini_tieddot_mathmix_2k_final_model.pt` (val_bpb 1.3574). Trainer
  `--bpb-only` reproduces both (1.3609 / 1.3596; fp32 masters + 2M-token guard window).
- Hit-rate probes: 2/256 hits, 2/32 variance groups both variants → formal DAPO
  go, but marginal; Answer-line compliance only 10-14% under free generation.
- `--rollout-only` gate (min within-group reward std 0.01, exit 2 on fail):
  latent 8.8e-5 FAIL, cot 0.0083 FAIL, **none 0.041 PASS** (teacher-forced
  Answer: prefix → 42% termination, 1.5% exact). Real RL from these nano
  checkpoints should start with none-mode or improve the base first.
- 4-step training + save→resume roundtrip per mode: all finite, age-0
  clip/policy exactly 0.0 (also post-resume), peak VRAM ≤5.7 GiB.

### Long-context + GPT-2 vocab extension (2026-07-22, later)

Nano pretraining scripts take `SEQ_LEN` (and `MBS`; halve MBS when doubling
SEQ_LEN) and record `train_seq_len` in the checkpoint; the trainer/probe derive
budgets from it: prompt 512, response `min(1024, (ctx-512)//2)`, latent stream
`min(4*response, ctx-512)`. At SEQ_LEN=4096 that is 512/1024 with a full-budget
cot stream — jobs 296-302 rebuild the sp1024 mathmix bases at 4096 and re-gate.

`nanogpt_mini_gpt2vocab_v1` is now a supported RL backbone:
- `postraining.core.load_posttraining_tokenizer` returns a `GPT2BPETokenizer`
  (transformers GPT2TokenizerFast) for `*gpt2vocab*` architectures; the single
  `<|endoftext|>`=50256 is both BOS and EOS, so stop ids dedupe to (50256,).
- BPB guard: `data/tokenizers/gpt2_byte_lut.pt` + zeroed correction tables fed
  to `eval_val` = direct LUT byte sum; val default `data/datasets/fineweb10B_gpt2`.
- `build_math_mix_dataset.py --tokenizer gpt2` builds the same mix under GPT-2
  BPE (QA docs drop their final EOS: the next doc's leading token terminates
  the last answer; keeping both would pretrain a doubled stop token).

Active chain (2026-07-22, trimmed for compute — jobs 296-303 + 306 of the
sp1024 4k track and extra probes cancelled as superseded): 304 `mathmix_v4_gpt2`
build → 305 `nanomini_gpt2vocab_mathmix4k_2k` (SEQ_LEN=4096 MBS=2) → 307 cot
`--rollout-only` variance gate (the single smoke run; exit 2 blocks the RL) →
308 `rl_gpt2vocab_cot_20k` (DAPO-Math-17K, `--reasoning-mode cot --steps 20000
--aime-every 250`, output `logs/rl_gpt2vocab_cot`).

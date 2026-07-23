# LeWM Reference — Exact Mechanics (../le-wm)

Read in full on 2026-07-21 from the local checkout. Every claim cites
file:line in that repo. Purpose: pin the reference implementation precisely
so our deviations are deliberate, enumerated, and individually testable.

## What LeWM trains

A continuous visual world model for control (PushT et al.): encode frames,
predict the next frame's *embedding* from context embeddings + actions, plan
at eval by rolling the predictor forward and scoring candidate action
sequences by latent MSE to a goal embedding. **There is no discrete emission
anywhere** — no CE, no vocabulary, no sampling. Our energy head is an
extension beyond the reference, not a transplant from it.

## Exact component inventory

| Component | LeWM | File:line |
|---|---|---|
| Encoder | ViT-tiny, patch 14, 224px, from scratch (`pretrained: false`), CLS token only → 192-d per frame | config/train/model/lewm.yaml; jepa.py:37-38 |
| Projector | MLP 192→2048→192 = Linear → **BatchNorm1d** → GELU → Linear (config overrides the module's LayerNorm default) | model/lewm.yaml `projector`; module.py:218-240 |
| `emb` (the regularized/target space) | `projector(CLS)` — SIGReg, prediction targets, and predictor *inputs* all live here | jepa.py:39-40 |
| Action encoder | Conv1d(k=1) + Linear→SiLU→Linear → 192 | module.py:189-214 |
| Predictor | ARPredictor: learned pos-embedding (`randn`, 4 frames) + Transformer depth 6, heads 16, dim_head 64, mlp 2048, dropout 0.1, **AdaLN-zero** blocks conditioned on action embeddings, non-affine LayerNorms, causal SDPA, final LayerNorm | module.py:244-285, 89-127; model/lewm.yaml |
| pred_proj | second **BatchNorm1d** MLP 192→2048→192 on predictor output | model/lewm.yaml `pred_proj`; jepa.py:51-54 |
| Losses | `MSE(pred_emb, tgt_emb).mean()` + `0.09 · sigreg(emb)` — nothing else | train.py:39-41 |
| Targets | slices of the same `emb` tensor, **fully attached** (no detach / stop-grad / EMA anywhere in training) | train.py:32-36 |
| SIGReg | knots 17 on t∈[0,3], trapezoid weights doubled interior, Gaussian CF window, 1024 fresh random unit projections per call, per-timestep batch statistic ×B, `.mean()` over (T, projections) | module.py:10-36 |
| SIGReg input | the **entire** encoded sequence incl. the purely-future target position (history 3 + 1 pred; `emb[:, n_preds:]` targets ⊂ sigreg input) | train.py:30-40; lewm.yaml wm |
| Optimizer | **one** AdamW for everything: lr 5e-5, weight_decay 1e-3, LinearWarmupCosineAnnealing (epoch interval), **grad clip 1.0**, bf16, batch 128, 100 epochs | lewm.yaml; train.py:86-93 |
| Rollout (eval) | predictor output (`pred_proj` space) is appended and **fed back as predictor input** — emb-space/pred-space interchangeability is assumed, never re-grounded | jepa.py:88-103 |
| Goal cost | last-step latent MSE vs `goal_emb.detach()` | jepa.py:115-130 |

## Where we match exactly

- SIGReg math: statistic, knots, window, weights, ×B scaling, mean over
  (T, P), num_proj 1024, weight 0.09 — bit-for-bit semantics (ours chunks
  projections/positions and checkpoints, preserving the exact mean).
- SIGReg placement: encoder-side projected embeddings, all positions
  including target-only ones; never on predictor/pred_proj output.
- Attached targets: no stop-grad/EMA anywhere in either training path.
- Trajectory-encoded-once with shared statistics (our
  `training_latents_with_belief` ≙ their single `encode` of all frames).
- Projector shape 512→2048→512 ≙ their 192→2048→192, incl. BN in the
  `bnproj` arm (job 242); prediction projector ≙ `pred_proj`.
- Per-position batch of 128 for the statistic (their whole batch = 128; our
  paired microbatches = 128).

## Enumerated deviations (each deliberate or testable)

1. **No discrete head in the reference.** Their latents are never decoded to
   symbols; generation-side design (energy head, bias, temperature) has no
   reference analog and must be earned by ablation.
2. **Statistic structure:** they compute ONE statistic per optimizer step on
   the whole batch; we compute 4 paired-128 statistics per step. Full
   alignment = one pooled B=512 statistic per step (registered as the
   stronger-signal direction; also fixes the 8×H100 pairing incompatibility).
3. **Optimizer:** uniform AdamW 5e-5 + wd 1e-3 + grad-clip 1.0 + cosine vs
   our Muon/Adam split, per-module LRs, no weight decay, no clipping,
   trapezoid warmdown. Inherited from the challenge baseline harness —
   defensible (it is heavily tuned for the 10-min budget) but it means
   *nothing* about our LR structure is validated by the reference.
4. **Dropout:** predictor dropout 0.1 (attention + FFN + AdaLN path) vs our
   0.0 everywhere. At 1-epoch-equivalent data scale ours is defensible;
   theirs runs 100 epochs and needs it.
5. **Positional encoding:** learned absolute pos-embedding over 4 frames vs
   PoPE over 1024 positions.
6. **Conditioning:** AdaLN-zero on action embeddings; the LM has no action
   stream — our trunk conditions on history alone. Structural, not a choice.
7. **Encoder input norm:** they project raw CLS; we `rms_norm` the embedding
   rows before the projector. Ours is a discrete-table artifact (embedding
   norms are optimizer state, not signal); harmless but unreferenced.
8. **Rollout re-grounding:** their rollout feeds predictor output straight
   back as input (space drift unchecked). Our generation samples a token and
   re-embeds through the codebook every step — strictly better grounded;
   noted because it means their rollout-stability evidence does not
   constrain our generation loop.
9. **Weight decay reaches their BN affines and biases** (single AdamW group,
   no exclusions); our harness applies none. If we ever add wd, the
   reference precedent is "decay everything," not the usual no-decay-on-norm
   convention.

## Sharpest transferable facts

- BN-in-projector is the reference configuration in *both* LeJEPA and LeWM;
  our RMS swap was the deviation (now under test, job 242 / H4).
- The reference never regularizes, decays, clips, or constrains the
  predictor output side beyond the MSE itself — all shaping pressure is on
  the encoder side. Our energy head keeps that property (SIGReg never
  touches ẑ); any future "regularize ẑ" idea is un-referenced territory.
- Their entire anti-collapse story is SIGReg weight 0.09 on a batch of 128
  at every step — the same constants we run. No warmup on the weight, no
  schedule, no EMA. Matches our canary evidence that 0.09 holds the
  geometry open.

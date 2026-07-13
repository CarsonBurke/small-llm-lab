# Latent World Model and VAPO Post-Training Plan

## Goal

Start from `fresh_lejepa_shared_rms_v1_probes_1k`, preserve the successful
shared-RMS transformer and large residual probes, and move all multi-step
reasoning into a stochastic latent world model. A lightweight renderer turns
selected latent states into vocabulary tokens, but rendered tokens never enter
the world-model transition path.

At each rollout step the model:

1. samples the next latent belief with a JEDI-style transition;
2. samples a Bernoulli THINK/EMIT action;
3. on EMIT, renders exactly one token; and
4. continues autoregressing in latent space after either action.

The model has a 20,480-position active attention/KV context and a separate
20,480 emitted-token response limit. There is deliberately no hidden-step,
total-latent-step, or wall-clock limit: a longer rollout uses a rolling window
over its most recent 20,480 world-model positions.

## Model

### Checkpoint migration

- Load every shape-compatible shared-RMS V1 weight from
  `ablation_results/fresh_lejepa_shared_rms_v1_probes_1k/pretraining_checkpoint.pt`.
- Allow missing keys only for the new stochastic transition, gate, action
  embeddings, positional-extension state, and new critic state. Any other
  missing, unexpected, or mismatched key is an error.
- Preserve the transformer, shared token encoder, final RMSNorm, large
  `ResidualProbe` actor/critic shapes, and FP32 SIGReg path.
- This checkpoint is selected because it is the only completed shared-RMS V1
  run with a full `pretraining_checkpoint.pt`; the nominal V1 2K run stopped at
  step 490 and contains metrics only.
- The checkpoint contains no diffusion denoiser. Port the V5-to-V9
  `BeliefTransitionEDM` and recursive-loss implementation into the shared-RMS
  V1 architecture as new modules; do not claim or assume inherited transition
  weights. The V9 belief gain of roughly 0.028 at step 1K is weak evidence, so
  transition usefulness is an explicit gate before RL.
- Treat long-context training as weight adaptation with a fresh optimizer, not
  an optimizer/RNG continuation.

### Latent state and stochastic transition

- The state is the contextual, post-transformer RMS-normalized belief.
- The predictive target is the next contextual belief from the same live
  encoder, detached at the target boundary; no EMA or target network is used.
- Replace the deterministic next-belief predictor with JEDI-style EDM
  preconditioning, its log-normal noise distribution, three denoising steps,
  teacher/predicted-conditioning switching, and the paper's bounded latent
  parameterization.
- The transition attends to the full cached latent history and the previous
  THINK/EMIT action. Zero-initialized action embeddings let RL learn distinct
  private-thought and emitted-token dynamics without perturbing pretraining.
- Keep SIGReg in FP32 on contextual beliefs and differentiable denoised
  predictions. Use effective SIGReg microbatch 128.
- Define alignment exactly. Given tokens `x[0..T]`, the encoder produces
  contextual beliefs `b_t` from inputs through `x_t`; the transition consumes
  state through `b_t` and predicts `b_hat_(t+1)` against
  `stopgrad(b_(t+1))`. The renderer consumes projected `embedding(x_t)` and
  `b_hat_(t+1)` and predicts `x_(t+1)`. It never receives the teacher
  `b_(t+1)`, preventing next-token leakage.

### THINK/EMIT gate

- Add a zero-initialized Bernoulli head over the current belief. It therefore
  begins post-training at exactly 50% EMIT.
- Mask and freeze the gate during predictive pretraining; every pretraining
  position is rendered and no hidden thinking is taught.
- Sample the gate in training and evaluation. Do not use class-index argmax:
  equal zero logits can otherwise select THINK forever before training. Clamp
  only numerical underflow so both Bernoulli outcomes retain nonzero sampling
  probability; report gate probabilities as well as sampled behavior.
- EMIT produces one token. THINK produces none. Both append a latent state and
  action to the world-model history.

### Renderer and critics

- Reuse the winning V1 large residual actor instead of introducing an MoE or a
  compressed probe.
- Preserve its successful input semantics:
  `concat(detach(project(current_token_embedding)),
  detach(predicted_next_latent))`.
- Initially, `current_token` is the final prompt token. THINK leaves it
  unchanged; EMIT replaces it with the newly emitted token. This gives the
  renderer one-token lexical continuity without feeding tokens into the world
  model or giving the renderer an independent temporal reasoning stack.
- Renderer CE and PPO cannot backpropagate through either detached input.
- During adaptation, log renderer CE with the predicted latent intact, zeroed,
  and batch-shuffled. Require a measurable gain from the aligned predicted
  latent over both controls before beginning RL. This tests decodability
  without letting token CE reshape the world model.
- Keep the vocabulary output learned and untied from the input embedding.
- Use two critics: a latent critic for plan-level return and the V1-shaped
  detached probe critic for emitted-token return.

## Pre-RL Adaptation and Context

### Predictive adaptation

Run 2,000 steps after checkpoint migration:

- train future-belief diffusion prediction, renderer CE, critics, and FP32
  SIGReg;
- keep the gate masked and frozen;
- use effective microbatch 128 and the parameter-golf optimizer conventions;
- use a 0.3x learning-rate multiplier for the reused encoder/backbone relative
  to the newly initialized denoiser; and
- save full resumable checkpoints at 1K and 2K.

Report two different likelihood metrics:

- teacher-forced BPB uses the real `x_t` in the V1 renderer input and remains
  comparable with pretraining; and
- latent-rollout BPB/NLL feeds back only the most recently emitted token while
  beliefs advance open-loop. It is a new metric and is not compared directly
  with the teacher-forced baseline.

Reject adaptation for collapse, non-finite gradients, negligible transition
belief gain, or no renderer gain from the predicted latent. Treat short-context
teacher-forced BPB as a regression diagnostic, not the Parameter Golf
competition's 0.005 keep/discard rule: this is an intentionally larger research
fork outside the 16 MB/10-minute competition target.

### PoPE context ablation

- PoPE replaces YaRN as the sole proposed long-context mechanism.
- PoPE is a separate from-scratch shared-RMS V1 experiment; never convert the
  existing RoPE checkpoint because PoPE changes Q/K content geometry.
- Faithfully implement softplus Q/K magnitudes, real/imaginary components, and
  learned phase offsets while adapting complex attention and caching to 8Q/4KV
  GQA.
- At the original 1,024 context and effective B128, compare a matched
  from-scratch RoPE 2K control with PoPE 2K. Continue the PoPE winner through
  low-learning-rate stages at 4,096, 8,192, and finally 20,480. The last stage
  supplies genuine target-length training because 20,480 is beyond the roughly
  10x extrapolation demonstrated by PoPE, while avoiding the roughly 20x
  attention cost during bulk pretraining. Validate at 2K, 4K, 8K, 10K, 16K,
  and the required 20,480 context after every continuation stage.
- Implement incremental complex K caching with absolute positions and a rolling
  20,480-position window. Tokens older than the active window are evicted; the
  rollout itself may continue.
- Keep PoPE only for material short- or long-context likelihood/retrieval gain
  that justifies its measured memory and throughput cost. Record the result as
  a research ablation rather than a Parameter Golf competition submission.

## Latent VAPO

### Rollout topology

For each prompt, sample four independent latent plans and four renderer samples
per plan, yielding 16 responses. The four renderings share their plan's latent
and gate trajectory but sample tokens independently.

- Stop a rendering on EOS or 20,480 emitted tokens.
- Stop a plan when all four renderings stop.
- A sampled Bernoulli with a nonzero numerical EMIT probability terminates
  almost surely while preserving the requested lack of a hidden-step policy
  cap. The harness must still detect process failure/OOM externally and mark
  the rollout failed rather than fabricating a truncated reward.

### PPO-compatible diffusion

- Use stochastic Gaussian reverse transitions around the three-step JEDI/EDM
  sampler so the world model has a tractable policy log-probability.
- Store every sampled reverse-chain state/action, its conditioning state,
  timestep/sigma, old Gaussian log-density, final latent, and gate action/log-
  probability. Recompute the new policy's density of those same stored states;
  never re-inject or re-sample stored noise under the new parameters.
- Apply a reverse-step sigma floor and bounded log-ratio before exponentiation
  so the near-zero final EDM step cannot create infinite PPO ratios.
- Before full integration, require a standalone diffusion-PPO toy latent
  bandit to improve reward reliably and to reproduce ratio 1 before updates.

### Hierarchical credit assignment

- A plan's terminal world-model/gate reward is the mean reward of its four
  renderings. A rendering's terminal residual reward is its reward minus that
  plan mean, isolating wording quality from latent-plan quality.
- Feed these two terminal rewards into their respective latent and renderer
  critics and length-adaptive GAE. Do not also normalize over the four plans as
  a second advantage estimator. Normalize final GAE advantages over the full
  valid update batch, with an epsilon/std-zero guard. Keep four-plan and four-
  renderer group statistics as diagnostics.
- World-model PPO updates only stochastic latent transitions. Gate PPO updates
  the Bernoulli policy and action-conditioned transition. Renderer PPO updates
  only the detached actor probe.
- Apply length-adaptive GAE separately to latent/gate and emitted-token
  sequences. Keep policy updates disabled until the rollout has both positive
  and negative rewards and each critic beats its constant-prediction baseline;
  50 updates is an upper warmup default, not an unconditional switch point.
- During RL use PPO plus FP32 SIGReg for the world model: no predictive replay,
  EMA, frozen-reference KL, or supervised future-belief loss.
- Retain VAPO's correct-response renderer NLL auxiliary with weight 0.1.
- Preserve VAPO's target configuration of 512 prompts, 16 responses, two PPO
  epochs, mini-batch 512, actor/world-model LR `1e-6`, and critic LR `2e-6`,
  but do not start there on one RTX 5090. Scale through explicit rollout gates:
  16 prompts/256 emitted tokens, 64/1,024, 128/4,096, then 512/20,480. Advance
  only after measuring wall time, CPU-spill size, peak VRAM, reward health, and
  update stability at the prior stage.

## TensorBoard and Run Artifacts

Keep the main dashboard small enough to diagnose runs at a glance.

### Outcomes

- train reward, exact-match accuracy, and positive-group fraction;
- AIME 2024 pass@1 and sampled pass@k; and
- reward/accuracy versus emitted response length.

### Rollouts

- emitted tokens mean/p95/max and truncation fraction;
- hidden steps mean/p95/max and hidden-to-emitted ratio;
- EMIT fraction, gate entropy, and EOS fraction;
- within-plan renderer reward variance; and
- across-plan mean-reward variance.

### Learning health

- world-model, gate, and renderer PPO loss, approximate KL, clip fraction,
  entropy, ratio statistics, and gradient norm;
- both critic losses and explained variance;
- renderer correct-response NLL;
- SIGReg loss and effective-rank ratio; and
- during predictive adaptation only: diffusion loss by noise level,
  teacher/predicted-conditioning fraction, renderer CE, and BPB.

### Performance and provenance

- latent steps/s, emitted tokens/s, rollout/update time, peak VRAM, KV-cache
  memory, and stored-trajectory memory;
- write the same metrics to JSONL; and
- store architecture, exact checkpoint lineage, positional method, sampler
  settings, loss weights, git state, and commands in each run manifest.

## Verification and Execution Order

1. Test strict checkpoint migration, detached gradient boundaries, gate masking,
   absolute cached positions, full/cached equivalence, diffusion log-probability
   recomputation, and 4x4 advantage grouping.
2. Run the matched from-scratch RoPE/PoPE CUDA smokes and 2K ablation.
3. Continue PoPE to longer sequences and verify the full 20,480 active context.
4. Port V9 JEDI modules into shared-RMS V1 and run the 2K predictive adaptation.
5. Require transition belief gain plus aligned-latent renderer gain; report
   teacher-forced and open-loop likelihood separately.
6. Prove diffusion PPO on the toy latent bandit, including ratio/sigma tests.
7. Run a tiny latent-RL gradient smoke proving each loss updates only its
   intended subsystem.
8. Scale rollout size/cap through the staged resource gates, then run VAPO with
   reward-health-gated critic warmup and evaluate AIME 2024 on the paper-aligned
   training-step-versus-accuracy schedule.
9. Stop variants with collapse, unstable diffusion ratios, renderer-to-world-
   model gradient leakage, no latent decodability gain, or infeasible measured
   rollout cost.

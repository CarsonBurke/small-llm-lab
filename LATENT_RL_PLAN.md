# Latent-Thought VAPO Post-Training Plan

Research fork on top of the parameter-golf pretraining stack. Not bound by
the 16 MB / 10-minute competition budget; teacher-forced FineWeb val BPB is
kept only as a do-no-harm regression guard.

## What is built (postraining/)

### Backbone and wrapper

- Base model: `FreshLeJEPASharedRMSV1PoPE` (9-layer U-Net GPT, dim 512,
  8Q/4KV GQA, 1024-piece SentencePiece BPE, PoPE attention, fp32 masters).
  Current checkpoint lineage is recorded in each run's `manifest.json`.
- `LatentThoughtModel` (latent_thought.py) wraps the frozen backbone with
  three small heads:
  - `GaussianTransitionHead`: diagonal heteroscedastic Gaussian over the next
    projected token latent. The mean is the frozen pretrained prediction
    path; only a state-dependent log-std is new. (The earlier JEDI/EDM
    diffusion-transition design was dropped: a Gaussian around the pretrained
    predictor gives a tractable PPO log-density with none of the
    reverse-chain ratio machinery.)
  - `ThinkEmitGate`: zero-init Bernoulli head — exactly 50/50 at start.
  - `ThoughtAdapter`: zero-init residual correction for injected thoughts, so
    an untrained thought is exactly the sampled imagined next-token latent.
- With the gate forced to EMIT, the wrapper is step-for-step identical to the
  backbone's `generation_step` (pinned by tests).

### Rollouts (latent_rollout.py)

- THINK samples a latent from the transition head and feeds it back through
  the adapter (occupies a stream position, renders nothing); EMIT samples a
  token from the renderer (the pretrained `policy_probe` LM head) and feeds
  it back through the embedding.
- Thinking is **unlimited**: no consecutive-think watchdog, no forced EMITs.
  The only bound is `max_stream_steps` generated slots (thinks + emits);
  overthinking costs emitted tokens and therefore reward. (The old 4-think
  watchdog and its forced-action PPO exclusions were removed.)
- Everything PPO needs is stored as replayable data; `replay_beliefs` /
  `refresh_old_statistics` recompute statistics through the exact update-step
  code path so epoch-0 PPO ratios are exactly 1.

### Critic (value_model.py, hl_gauss.py)

- `SeparateCritic`: a **from-scratch** trunk of the same architecture class
  (fully trainable, ~28.9M params, training-only scaffolding) with its own
  thought adapter and an HL-Gauss categorical value head (cleanrl v215
  recipe: 101 bins on [0,1], sigma_ratio 2.0, zero-weight head with
  projected-prior bias, softmax-CE to truncated-Gaussian two-hot targets, no
  value clipping, no advantage normalization).
- The policy's stepwise path never computes values; the critic scores stored
  streams in parallel. Rationale: thought content receives no policy
  gradient by design (D4), so a critic reading the policy trunk read
  latents that nothing was training — hence a separate model.

### Training (train_latent_vapo.py)

- VAPO-aligned: length-adaptive GAE (λ from |trajectory|), clip-higher PPO,
  50-iteration value warmup (MC returns) before any policy update,
  positive-example LM loss on correct trajectories, transition head trained
  only on grounded transitions via β-NLL (world model, never policy
  gradient).
- Diagnostics: think-run length stats, per-loss grad norms (gate, renderer,
  transition, critic), emit probability, emits/actions per trajectory,
  teacher-forced val BPB guard, AIME24 avg@k eval through the latent policy.
- `sample_latent.py` inspects the trained policy; its decoded output marks
  every think run inline as `{n}🪙`.

## Pretraining: math-mix corpus (open data only)

RL needs a base model that can actually score on verifiable math; a
FineWeb-only 27M model earns exactly zero verifier reward (no gradient).
`build_math_mix_dataset.py` builds `data/datasets/mathmix_v3_sp1024`
(500M tokens, challenge shard format, same tokenizer):

| slice | fraction | source |
|---|---|---|
| FineWeb | 45% | existing tokenized shards (BOS-split, stitched across files) |
| FineMath-4+ | 25% | HuggingFaceTB/finemath (math web prose) |
| DeepMind math train-easy | 21% | mathematics_dataset v1.0 tarball, 18 verifier-compatible modules, worksheet docs |
| OpenMathInstruct-2 | 9% | gsm8k/augmented_gsm8k rows: problem + short solution |

- Every QA answer is formatted `Answer: <x>` (exactly what
  `verify_answer` extracts) and **closed with EOS** — the tokenizer
  normalizes newlines away, so EOS is the model's only learnable stop
  signal. Rollouts/evals truncate at the first EOS.
- **Half of all QA docs are wrapped in the verbatim DAPO-Math-17K prompt
  template** (preamble + problem + `Remember to put your answer on its own
  line after "Answer:".` + response). The v2 corpus lacked this, and the
  v2-pretrained model produced an `Answer:` line in 0/1024 emit-only DAPO
  rollouts — it treated template-wrapped problems as prose to continue.
  Template compliance, not math difficulty, was the reward-variance
  blocker; DAPO ground truths are mostly small integers, so a compliant
  model earns occasional lucky hits, which is all RL needs to start.
- No homemade/synthetic data: all sources are published open datasets
  (Apache-2.0 / CC-BY-4.0 / ODC-By). The template wrapper is the RL
  prompt's own text, not synthetic content.
- Val shard is pure FineWeb, copied unchanged, so BPB stays comparable.
  (v2 reference: final FineWeb val BPB 1.5849 vs 1.4966 for FineWeb-only —
  expected, 55% of tokens are no longer FineWeb.)
- Run: `ablation.py --script fresh_lejepa_train_v1_probe_shared_rms_pope_zero.py
  --env DATA_PATH=data/datasets/mathmix_v3_sp1024` (2k steps, b128).

## RL: exclusively the VAPO paper's setup

Post-training uses **only** what the paper uses — no FineWeb continuation
rewards, no synthetic RL tasks:

- Training prompts: DAPO-Math-17K (`postraining/data/dapo-math-17k.parquet`,
  ~17k unique problems) with the binary Minerva-style verifier as terminal
  reward. (The paper never names its RL dataset; "identical experimental
  settings" to DAPO pins it to DAPO-Math-17K.)
- Eval: AIME 2024 avg@32 at temperature 1.0 / top-p 0.7.
- Paper alignment kept: value warmup, length-adaptive GAE, clip-higher,
  positive-example LM loss, token-level (here: action-level) loss; critic is
  HL-Gauss instead of MSE (deliberate deviation, documented above).

### Postmortem: why Tier 0 died

Tier-0 (FineWeb continuation reward = prefix match + char F1) was RL
reproducing the pretraining objective through a worse optimizer: reward rose
by gaming F1 while BPB drifted up, and the gate correctly learned that
thinking never helps next-token prediction on web text. Runs
`latent_vapo_tier0_v2` (probe critic) and `_v3` (separate HL-Gauss critic,
killed at step ~240) are kept as artifacts; v3's calibrated critic held the
gate near 50/50 where v2's miscalibrated one collapsed it — the
critic works, the task was wrong.

## Status / order of execution

1. ~~Separate HL-Gauss critic + tests~~ — done, reviewed clean.
2. ~~Unlimited thinking (watchdog removed)~~ — done.
3. ~~Math-mix corpus from open datasets~~ — v2 built, superseded by
   `mathmix_v3_sp1024` (DAPO-templated QA docs; manifest in the dataset dir).
4. ~~DAPO rollout plumbing~~ — done, reviewed (one real finding: AIME eval
   didn't thread `eos_id` into rollouts; fixed + regression test). Group =
   per-prompt rollout batch = PPO minibatch; verifier-scored terminal
   rewards; EOS-truncated generations; resumable `MathPromptSampler`.
5. ~~v2 math-mix pretraining~~ — `fresh_lejepa_srms_pope_zero_b128_mathmix_2k`,
   final FineWeb val BPB 1.5849. Emit-only sampling: instant
   `Answer: <n> <eos>` on bare worksheet prompts, but 0/1024 `Answer:` lines
   on real DAPO prompts → zero reward variance, RL gate correctly refused.
   Root cause: the DAPO instruction template never appeared in pretraining.
6. **v3 math-mix pretraining** —
   `fresh_lejepa_srms_pope_zero_b128_mathmix_v3_2k` (in flight). Gate:
   emit-only DAPO hit-rate probe must show nonzero within-group reward
   variance (`postraining/dapo_hit_rate_probe.py`, 64 prompts x 16 samples).
7. **Smoke RL run** on DAPO-Math with the v3 checkpoint;
   `--rollout-only` gate first (`gate_min_within_group_reward_std`).
8. Full run + AIME curve. Honest ceiling: a 27M model will not meaningfully
   solve DAPO/AIME problems; the research question is whether latent
   thinking earns reward above the emit-only baseline under a real verifier,
   not leaderboard accuracy.

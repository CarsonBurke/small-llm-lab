# Latent-Thought VAPO Post-Training Plan

Research fork on top of the parameter-golf pretraining stack. Not bound by
the 16 MB / 10-minute competition budget; teacher-forced FineWeb val BPB is
kept only as a do-no-harm regression guard.

## What is built (postraining/)

### Backbone and wrapper

- Base model: `FreshLeJEPASharedRMSV1PoPE` (9-layer U-Net GPT, dim 512,
  8Q/4KV GQA, 1024-piece SentencePiece BPE, PoPE attention, fp32 masters).
  Current checkpoint lineage is recorded in each run's `manifest.json`.
- **Current RL base (user call, Jul 18): the belief-attached CE variant**
  (`FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE`,
  `fresh_lejepa_train_v1_probe_shared_rms_pope_belief_attached.py`). Its
  vocabulary CE renderer reads `[token latent, raw belief]`; CE trains the
  embedding/projector and temporal trunk, but has no graph edge into the
  next-latent prediction projector. The latent MSE remains the pretraining
  objective that trains that projector. It has a distinct architecture ID,
  and `model_io.load_model` reconstructs that exact class.
- `LatentThoughtModel` (latent_thought.py) wraps the backbone with:
  - `GaussianTransitionHead`: diagonal Gaussian over the next projected
    token latent with **fixed sigma** (log-sigma -0.5, matched to the
    backbone's pretraining prediction error; a buffer, not a parameter).
    The mean IS the backbone's prediction path — at RL time the policy
    gradient flows through it into the whole trunk. (The earlier JEDI/EDM
    diffusion-transition design was dropped: a Gaussian around the
    predictor gives a tractable PPO log-density with none of the
    reverse-chain ratio machinery. The v1 delta-offset/learned-log-std
    heads were removed with the frozen-trunk design, Jul 18.)
  - `ThinkEmitGate`: zero-init Bernoulli head — exactly 50/50 at start.
  - `ThoughtAdapter`: zero-init residual correction for injected thoughts, so
    an untrained thought is exactly the sampled imagined next-token latent.
  - Renderer: `[current input latent, raw belief] -> policy_probe -> vocab`.
    It deliberately bypasses `prediction_latent`, which is reserved for the
    continuous thought policy. Cached and parallel belief-renderer paths are
    pinned against each other by tests.
- Pretraining, teacher-forced evaluation, cached generation, and VAPO all use
  the same belief renderer. Old predicted-renderer checkpoints are rejected,
  even though their tensors are shape-compatible.

### Rollouts (latent_rollout.py)

- THINK samples a latent from the transition head and feeds it back through
  the adapter (occupies a stream position, renders nothing); EMIT samples a
  token from the belief-reading `policy_probe` renderer and feeds it back
  through the embedding. `prediction_latent` still runs densely on every
  stream step to preserve the launch-efficient batched path, but the renderer
  never consumes it. Only the THINK-masked thought surrogate gives the
  projector gradients; EMIT token CE/PPO does not.
- Thinking is **unlimited**: no consecutive-think watchdog, no forced EMITs.
  The only bound is `max_stream_steps` generated slots (thinks + emits);
  overthinking costs emitted tokens and therefore reward. (The old 4-think
  watchdog and its forced-action PPO exclusions were removed.)
- Everything PPO needs is stored as replayable data; `replay_beliefs` /
  `refresh_old_statistics` recompute statistics through the exact update-step
  code path so epoch-0 PPO ratios are exactly 1.

### The latent-thinking contract (restated, Jul 18)

At every generated stream position the policy outputs a THINK/EMIT Bernoulli
from the belief. THINK: a latent is sampled from the transition head and
autoregressed back into the stream (the le-wm imagination-rollout pattern) —
it occupies a slot in the output chain as a latent "token" but renders no
vocab token. EMIT: the CE probe (renderer) runs and a vocab token is sampled
and fed back through the embedding. The critic must evaluate **both vocab and
latent positions** so thinking gets training signal: `SeparateCritic` scores
every stored stream slot in parallel and GAE runs over the full action mask
(gate decisions at token AND thought slots).

**v2 (user prescription, Jul 18 — supersedes the frozen-trunk v1): the
WHOLE policy model trains.** Freezing the world model was rejected
("it should always be learning"). At RL time the world model's objective
changes from predicting what the next token/latent WILL be to predicting
what it SHOULD be: the thought policy's mean is the model's own predicted
latent, and the Gaussian-vector score backprops through the prediction path
into the entire trunk. Gate and content factors form one action probability:
gate+token for EMIT, gate+summed Gaussian density for optional THINK, and the
Gaussian density alone for a forced THINK. VAPO's clipped surrogate is then
applied once per joint action. Sigma is a FIXED
constant (log-sigma -1.5, σ ≈ 0.22 —
red-teamed down from the error-matched -0.5, whose per-step offset norm
~13.7 against thought norms ~22.6 corrupts long think runs with no way to
self-shrink): no beta-NLL, no learned uncertainty, no continuous-policy
entropy bonus, and no KL penalty — the trust region is the continuous-policy
constraint. The separate optional gate-entropy ablation is documented below.
The v1 delta/log-std heads and the frozen-trunk adaptation trainer
(`train_adaptation.py`, `adaptation_core.py`) were deleted.
`--thought-pg-coef 0` now disables only the thought-content surrogate
(the trunk still trains through the token surrogate) — it is a control
arm for thought-reward specifically, no longer a bitwise gate-only
reproduction.

**The pretraining/post-training split, stated as the design thesis (user,
Jul 18):** just as post-training turns a next-token predictor into a
*useful*-token predictor, RL turns the next-latent world model into a
useful-latent predictor — directly, by retraining the prediction path
with reward, not through a residual offset. **No pretraining objective
survives into RL** (user, Jul 18): SIGReg and the latent
target-prediction loss are dropped entirely — keeping either would pit
"predict what the next latent WILL be" against the policy gradient's
"predict what it SHOULD be" on the same prediction path. (A PPO-ptx-style
LeJEPA anchor was designed and implemented as the answer to the "is
SIGReg/LeJEPA still a thing?" concern, then rejected for exactly that
objective conflict; the code is gone, not flagged off.) At RL time the
model trains purely on its ability to think; SIGReg/LeJEPA's role ends at
producing the pretrained latent space RL starts from. The teacher-forced
val-BPB guard runs through the deployed belief renderer, not the backbone's
old predicted-latent renderer — if the latent space or renderer degrades
enough to hurt language modeling, it shows there.

### Design position vs Coconut (chain of continuous thought)

Recorded from the Jul 18 design discussion — how our latent thinking relates
to Coconut (Hao et al.), and why the differences are forced, not stylistic.

**Training economics.** Normal token CoT has no BPTT: discreteness severs
the graph between steps and teacher forcing gives one parallel forward.
Coconut restores a differentiable chain by feeding the *exact* last hidden
state forward as the next input — which costs c serial forwards per thought
segment and backprop depth ~L·c, with no teacher forcing possible (each
thought input depends on the previous live forward). We are the third
corner: thoughts are *sampled* actions, so they are replayable data — one
parallel teacher-forced replay (`replay_beliefs`) recomputes everything PPO
needs, and credit assignment flows through the critic + GAE instead of
BPTT. Training cost is a token-CoT-shaped parallel pass; the price is paid
in estimator variance (score-function gradients), not compute.

**Deterministic-exact vs sampled thought — the forced pairing.** Coconut's
thought MUST be the exact deterministic hidden state: its gradient
mechanism backprops through the thought, so the thought must be a
differentiable function of the previous pass. Ours MUST be stochastic: the
score-function gradient needs a density to evaluate, and exploration in
thought space is the only channel through which reward can discover better
thoughts. Neither design gets to choose the other's thought type without
also taking its training mechanism. The ledger:

- *Costs of sampling:* (1) noisy channel — each THINK step injects
  N(0, σ) perturbation and long think runs accumulate it, where Coconut
  carries computation exactly (v2 note: σ is a fixed constant, so this
  cost is flat per step and does not anneal on its own); (2) information
  bottleneck — Coconut feeds the full belief (everything the trunk
  computed), while our thought is the belief pushed through
  `prediction_latent`, a vector trained to mean one thing (expected
  next-token latent), so non-next-token content is projected away. In v2
  the escape hatch is the prediction path itself: with no surviving
  prediction objective, reward-training the trunk can freely repurpose
  what thoughts encode; (3) estimator variance vs exact gradients
  (above).
- *Benefits of sampling:* (1) it is what makes the whole replay/PPO
  economics exist — a density to importance-correct and a score-function
  gradient to train thought content; (2) exploration in thought space at
  a scale matched to the pretrained predictor's own per-dim error
  (σ ≈ 0.22; the error-matched 0.61 ≈ sqrt of the 0.36 latent MSE proved
  too hot — see above); (3) the superposition
  property Coconut credits for its search-like behavior mostly survives:
  our Gaussian's *mean* is the trained expectation over continuations —
  itself a blend of candidate futures — and we sample around it rather
  than collapsing to a token; (4) projecting into token-latent space
  means an untrained thought is already in-distribution for the trunk —
  no Coconut-style multi-stage curriculum teaching the model to consume
  its own raw hidden states (and in v2 the trunk keeps adapting to its
  own thoughts as both train).

**Empirical adjudication (open).** Whether noise accumulation hurts long
think runs is answerable from existing diagnostics: think-run length stats
vs reward, thought clip fractions and `trunk_grad_norm` (whether thought
reward actually moves the prediction path), the `--thought-pg-coef 0`
control arm, and `sample_latent.py` stream inspection. If fixed noise
proves too blunt (long runs degrading), `--thought-log-sigma` is the
single knob — including annealing it externally across runs.

### Critic (value_model.py, hl_gauss.py)

- `SeparateCritic`: a **from-scratch** trunk of the same architecture class
  (fully trainable, ~28.9M params, training-only scaffolding) with its own
  thought adapter and an HL-Gauss categorical value head (cleanrl v215
  recipe: 101 bins on [0,1], sigma_ratio 2.0, zero-weight head with
  projected-prior bias, softmax-CE to truncated-Gaussian two-hot targets, no
  value clipping, no advantage normalization).
- The policy's stepwise path never computes values; the critic scores stored
  streams in parallel. v2 rationale: with the policy trunk now
  reward-trained, a shared-trunk critic would couple value estimation to a
  representation being actively bent by the policy gradient — the separate
  model decouples them entirely (and its from-scratch training
  distribution has always included thought slots, so latents are native
  inputs, not out-of-distribution ones).

### Training (train_latent_vapo.py)

- VAPO-aligned objective: length-adaptive GAE (λ from |trajectory|), one
  clip-higher PPO ratio (0.20/0.28) per joint action, and positive-example
  LM loss token-normalized across correct trajectories.
- v6 optimizer layout: one actor AdamW with param groups — pretrained
  trunk at the VAPO paper's `--actor-lr` (1e-6), scalar gate at `--gate-lr`
  (1e-4), recurrent thought adapter at `--adapter-lr` (1e-6), and renderer
  probe at `--renderer-lr` (1e-6) — plus the critic AdamW (3e-4,
  from-scratch scale). The adapter is not treated as an isolated fresh head:
  it changes every later belief and therefore every factor of the recurrent
  512-D thought policy. Old v1
  checkpoints cannot `--resume` across this change (optimizer keys and
  head shapes differ); none are worth keeping. Likewise, VAPO checkpoints
  from before the `input_latent+belief/v1` renderer schema are rejected on
  resume/eval even though tensor shapes happen to match.
- Actor and critic gradients accumulate across length-varying prompt groups
  and each optimizer steps once on the same effective trajectory minibatch.
  Actor policy/NLL and critic value CE use their global token denominators;
  replay groups and length-aware shards are memory partitions only. PPO is
  restricted to one pass so generated trajectories are never reused. The
  default behavior pool is one fresh 16-prompt × 32-sample B512 minibatch per
  actor+critic update. The tested 2048-pool/four-disjoint-B512 variant was
  rejected: with an exact 512-D Gaussian joint ratio, rare negative-advantage
  tails overflowed stale updates at both 1e-5 and VAPO's 1e-6 actor LR. Fresh
  B512 completed 500 updates at 1e-6 and improved held-out avg@8 from 17.53%
  to a 20.49% peak (20.23% at the final scheduled step-480 evaluation), with
  only +0.00013 BPB drift at step 320. A 1e-7 frozen-pool diagnostic did not
  rescue the design: it still developed catastrophic joint-ratio tails before
  cancellation. Fresh B512 has behavior age zero on every update, so the
  formal clipped objective is preserved but ratios are one and PPO clipping/KL
  are inactive (on-policy policy gradient). Optional thinking collapsed from
  2.39% to 0.068% on the held-out bench; at step 480, forced-initial accuracy
  was 19.97% versus 20.49% unforced. The gain therefore demonstrates stable
  policy/renderer post-training, not learned useful optional thinking.
  This run initialized from the requested post-trained critic checkpoint,
  whose optimizer provenance is `legacy_per_prompt_group`; new warmups use
  aligned B512 critic updates instead.
- Diagnostics: think-run length stats, per-loss grad norms (gate, renderer,
  transition, critic), emit probability, emits/actions per trajectory,
  teacher-forced belief-renderer val BPB guard, AIME24 avg@k eval through
  the latent policy.
- `sample_latent.py` inspects the trained policy; its decoded output marks
  every think run inline as `{n}🪙`.

## Pretraining: math-mix corpus (open data only)

RL needs a base model that can actually score on verifiable math; a
FineWeb-only 27M model earns exactly zero verifier reward (no gradient).
`build_math_mix_dataset.py` builds `data/datasets/mathmix_v4_sp1024`
(1,048,592,000 tokens, challenge shard format, same tokenizer):

| slice | fraction | source |
|---|---|---|
| FineWeb | 45% | existing tokenized shards (BOS-split, stitched across files) |
| FineMath-4+ | 25% | HuggingFaceTB/finemath (math web prose) |
| DeepMind math train-easy | 21% | mathematics_dataset v1.0 tarball, 18 verifier-compatible modules, worksheet docs |
| OpenMathInstruct-2 | 9% | all four published sources: problem + generated solution |

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
- The corpus is exactly one 2,000-step stream: each step consumes 524,288
  training tokens plus eight shifted-target boundary tokens, for
  1,048,592,000 total. Construction and source ordering use no RNG, and each
  source iterator is consumed monotonically without reuse. The manifest
  records `one_pass=true` and `rng=none`.
- The strict pretraining entrypoint requires the shard payload count to equal
  that exact total, validates the manifest, disables loader-reset compile
  warmup, and uses a loader that raises at exhaustion instead of wrapping.
- The v3 step-900 event was a corpus-size/construction failure, not an
  optimizer instability: its 500,009,831-token stream was exhausted inside
  step 954 and then reused from the start. Before the wrap, the remaining-
  budget random source chooser had concentrated the tail into short QA docs,
  producing the multi-step distribution shift; the wrap reset it abruptly.
- Run (only after explicit queue permission): `ablation.py --script
  fresh_lejepa_train_v1_probe_shared_rms_pope_belief_attached.py --env
  DATA_PATH=data/datasets/mathmix_v4_sp1024` (2k steps, b128).

## RL: exclusively the VAPO paper's setup

**The algorithm is VAPO** (value-model-augmented PPO: trained critic, GAE,
value warmup). Everywhere "DAPO" appears in this project it names the
**DAPO-Math-17K prompt dataset and its verifier**, never the DAPO algorithm
(which is value-free and is not what we run).

Post-training uses **only** what the paper uses — no FineWeb continuation
rewards, no synthetic RL tasks:

- Training prompts: DAPO-Math-17K (`postraining/data/dapo-math-17k.parquet`,
  ~17k unique problems) with the binary Minerva-style verifier as terminal
  reward. (The paper never names its RL dataset; "identical experimental
  settings" to DAPO pins it to DAPO-Math-17K.)
- Eval of record: AIME 2024 avg@32 at temperature 1.0 / top-p 0.7 (user,
  Jul 18: even very marginal AIME improvement counts as a sign). Companion
  benchmark where a 27M model can show an actual curve:
  `postraining/data/deepmind-interpolate-easy.parquet` — 144 held-out
  DeepMind `interpolate` problems from the same 18 modules as the training
  mix, DAPO-templated, verifier round-trip validated
  (`build_deepmind_eval_set.py`). The RL trainer evaluates it every
  `--bench-every` steps (`bench/accuracy`), and `eval_aime.py --test-file`
  probes it emit-only between pretraining stages.
- Paper alignment kept: value warmup, length-adaptive GAE, clip-higher,
  positive-example LM loss, token-level (here: action-level) loss; critic is
  HL-Gauss instead of MSE (deliberate deviation, documented above).
- **Gate-entropy ablation** (user call, Jul 19): VAPO's objective has no
  entropy term — the paper only monitors entropy, with clip-higher as the
  exploration mechanism — but the optional THINK gate collapsed before its
  much slower 512-D content policy could learn. `--gate-entropy-coef` now
  enables an explicit, head-only Bernoulli entropy bonus, globally averaged
  over optional gate actions. The paper-faithful default remains zero; the
  intervention is logged separately as `bonus/gate_entropy_weighted`.
  No KL penalty either, confirmed against the
  full paper (Jul 18 full-text read): KL appears only in VAPO's Sec. 2.2
  theoretical preliminaries (Eq. 1); the loss actually optimized is
  L_PPO + mu*L_NLL (Eqs. 7-10) with no KL term, no beta in the
  hyperparameter list, and no KL ablation — clipping is the policy
  constraint. Independently, no KL is coherent here anyway: there is no
  reference policy for heads that don't exist in the pretrained model.

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
3. ~~Math-mix corpus from open datasets~~ — v2/v3 built; v3 diagnosed as
   undersized and superseded. Exact one-pass `mathmix_v4_sp1024` is ready to
   build once queue permission is given.
4. ~~DAPO rollout plumbing~~ — done, reviewed (one real finding: AIME eval
   didn't thread `eos_id` into rollouts; fixed + regression test). Group =
   per-prompt rollout batch = PPO minibatch; verifier-scored terminal
   rewards; EOS-truncated generations; resumable `MathPromptSampler`.
5. ~~v2 math-mix pretraining~~ — `fresh_lejepa_srms_pope_zero_b128_mathmix_2k`,
   final FineWeb val BPB 1.5849. Emit-only sampling: instant
   `Answer: <n> <eos>` on bare worksheet prompts, but 0/1024 `Answer:` lines
   on real DAPO prompts → zero reward variance, RL gate correctly refused.
   Root cause: the DAPO instruction template never appeared in pretraining.
6. ~~v3 math-mix pretraining~~ — superseded after diagnosing its undersized
   500M-token corpus, tail shift, and second-pass wrap.
7. **Next:** build mathmix v4 and pretrain the belief-attached CE architecture
   under the exact-one-pass contract below. No workload is queued pending
   explicit permission.
8. Then smoke RL + full run + AIME curve. Honest ceiling: a 27M model will not meaningfully
   solve DAPO/AIME problems; the research question is whether latent
   thinking earns reward above the emit-only baseline under a real verifier,
   not leaderboard accuracy.

## Belief-attached CE lineage (current, Jul 18)

The renderer architecture changed from `[token, predicted next latent]` to
`[token, raw belief]`, so the old checkpoint cannot be reused. The next base
must be pretrained from scratch on mathmix v4. This is also required because
v3 was shorter than the configured training stream and silently began a
second epoch; its step-900/954 disruption makes that lineage unsuitable as a
clean base.

Pipeline (mlq chain, each stage gated on the previous one's success):

1. Build `data/datasets/mathmix_v4_sp1024`: exactly 1,048,592,000 training
   tokens, deterministic construction/order, no RNG, no source reuse.
2. `fresh_lejepa_srms_pope_belief_attached_ce_b128_mathmix_v4_2k` —
   from-scratch belief-attached CE pretraining with strict exact-one-pass
   preflight (`ablation.py --script
   fresh_lejepa_train_v1_probe_shared_rms_pope_belief_attached.py --env
   DATA_PATH=...`).
3. Emit-only DAPO hit-rate probe (`postraining/dapo_hit_rate_probe.py`) —
   answer-line compliance, hit rate, within-group variance (informational).
4. Emit-only AIME avg@32 probe (`postraining/eval_aime.py`,
   `--max-tokens 512`; max prompt 439 tok + 512 < 1024 context) —
   **informational**, run after every pretraining stage per the user's
   periodic-AIME requirement. NOT a mechanical exit gate: at 27M scale AIME
   pass/fail over 960 samples is lottery noise (answers are ints 0-999;
   expected hits ~0-2 by luck), so a hard threshold would stop or pass the
   chain essentially at random. The user's "no AIME signal -> no RL" bar is
   applied by reading this number alongside the DAPO probe — the *math
   distribution the RL gradient actually comes from* — and by the binding
   gate below. (`eval_aime.py --min-accuracy` exists as an opt-in exit-2
   gate if a hard AIME stop is ever wanted.) If signal is absent across
   probes, the lever is more/longer math pretraining (4k+ steps, higher QA
   fraction), then probe again.
4b. Emit-only easy-benchmark probe (`eval_aime.py --test-file
   deepmind-interpolate-easy.parquet --max-tokens 128`) — the same-difficulty
   held-out set; this is where pretraining signal should actually register.
5. Latent VAPO `--rollout-only` gate — **binding go/no-go** (exit 2 stops
   the chain): within-group reward variance under the true 50/50 zero-init
   gate, the actual nonzero-policy-gradient condition on the DAPO-Math
   prompt set. Note step 3
   measures best-case emit-only variance (gate pinned EMIT); this one
   measures the condition RL actually starts from.
6. Full latent VAPO run (`postraining/runs/latent_vapo_belief_v2_ctx5k`),
   teacher-forced FineWeb BPB as the do-no-harm guard, resumable checkpoints
   every 50 steps, and one full AIME avg@32 through the final latent policy.
   Posttraining and eval share a 5120-position contract: up to 1024 prompt
   tokens, 1024 emitted answer tokens, and 4096 total generated THINK+EMIT
   slots. The current AIME and DeepMind prompts max out at 440 and 186 tokens,
   respectively, so none are truncated. PoPE executes this mechanically, but
   quality beyond its 1024-position pretraining window remains an extrapolation
   to measure rather than an assumed capability. The 5K cache requires
   `--rollout-groups 4` on the 32GB development GPU; 32 samples per prompt and
   all optimizer batch semantics remain unchanged.

Belief-attached CE note for RL, updated for v2: the replay pass is now the
differentiable forward — `update_minibatch` no longer detaches anything on
the policy side. Gate and token surrogates backprop through raw beliefs into
the trunk; only THINK-masked joint-action factors consume projected
thought means and trains the prediction projector. The projector executes
dense batched work in rollout and replay, avoiding a per-step GPU-to-CPU
branch; unused EMIT positions contribute exactly zero projector gradient.
The old D4/frozen-trunk separation is gone by prescription, and no
pretraining objective runs at RL time. The teacher-forced FineWeb BPB guard
therefore uses the deployed belief renderer and is load-bearing: with the
trunk moving and nothing anchoring it, it is the sole drift alarm.

## torch.compile (reduce-overhead / CUDA graphs), Jul 18

User prescription: compile "almost exactly like pretraining. For both
models." Pretraining's recipe is `torch.compile(model, dynamic=False,
fullgraph=True)`; the RL trainer can mirror it with `mode="reduce-overhead"`
on top, behind `--compile` (default off; tests always exercise eager paths).
Profiling motivated it: the stepwise rollout is launch-bound (~3.8 ms per
stream step at batch 32 vs ~0.3 ms of compute), and collect() is ~75% of
iteration wall time.

Three compiled surfaces:

1. **Stepwise policy path** — `wrapper.step_core` (pure tensors in,
   `(belief, predicted, logits)` out; no dataclass construction or cache
   aliasing crosses the compile boundary — red-teamed) compiled with
   `mode="reduce-overhead", fullgraph=True, dynamic=False`. Static shapes
   come from a new `key_mask` mode in `_attention_step`/`forward_step`:
   K/V written by `index_copy_` at a 0-dim tensor position, attention
   over the WHOLE preallocated cache under a boolean mask instead of
   `narrow(:pos+1)`. Caches are allocated once per (batch, cache_length)
   shape (`make_static_generation_cache`: zero-filled — masked garbage
   would reach softmax as NaN — and `mark_static_address`ed so the CUDA
   graph may mutate them in place), and reused forever including across
   the aime/bench evals; reallocation would force a graph re-record.
   Sampling stays outside the graph. The rollout loop's `bool(any())`
   finish check — a D2H sync that would serialize the launch-bound loop
   regardless of graphs (red-teamed) — now fires every 16 steps only;
   the overshoot steps are no-op writes by construction.
2. **Policy replay** — `replay_head_inputs` compiled once and rebound in
   BOTH consumer modules (train_latent_vapo global for update_minibatch,
   latent_rollout global for refresh_old_statistics): one artifact for
   refresh and update is what preserves exact epoch-0 PPO ratios. For the
   same reason `refresh_old_statistics` now runs its forward GRAD-ENABLED
   (grad mode is a dynamo guard; a no-grad trace would be a different
   artifact with different bf16 reduction order) and detaches on store.
3. **Critic** — `critic.value_logits` compiled (values() rides through it).

Surfaces 2-3 use PLAIN compile (no reduce-overhead): their outputs stay
live across the whole eager loss region and its backward — the classic
CUDA-graph output-overwritten hazard — and the batched minibatches are
compute-bound, so graphs would buy little there (red-team call). The
user's cudagraph prescription lands where it pays. The measured compiled
path is nevertheless 2.6x slower at this model size, so eager generation
batches four prompt groups (128 trajectories at 32 samples/problem) into
each left-padded rollout. Four groups keep the 5K fp32 PoPE caches inside
the 32GB development GPU.

Stream-length variability is bounded by `trim_stream(multiple=64)`
(`--replay-bucket`): replay shapes bucket to 64-step boundaries, each
bucket is one compiled graph (padding-invariant: PAD inputs zeroed,
causal attention, all losses masked; pinned by test). The current contract is
`--prompt-tokens 1024 + 4096` generated slots, with at most 1024 emitted
tokens. AIME runs once from the final checkpoint; the easier benchmark stays
periodic and normally terminates far short of its cap.

Post-mortem (Jul 18, jobs 136/137): the first bucketing capped the
rounded length at the ORIGINAL stream length, so any group whose
content reached into the last partial bucket leaked one arbitrary
stream shape — an unbounded compile set, which is why collect never
converged (47-123 s vs eager 14-16 s; each leaked shape costs a silent
~15-25 s inductor forward compile inside refresh plus ~10 s backward at
the first update minibatch). The bucket boundary is now strict:
trim_stream PADS BEYOND the original stream when needed (pad columns
are the natural all-PAD state). With `--replay-bucket 128` the whole
run sees at most ~7 replay shapes (multiples of 128 up to 896).

Final verdict (job 138, the fixed probe — iterations 2-4 ran with zero
compile spikes, so these are true steady-state numbers): compiled
collect 36-40 s vs eager 11-16 s; minibatch 0.13 s vs 0.11 s;
iteration 42-57 s vs 14-19 s.  The static-cache cudagraph step is
structurally slower here: it runs masked SDPA over the FULL ~900-slot
preallocated cache at every step (~4x the attention FLOPs of eager's
prefix-only step), which swamps the ~3.5 ms/step launch savings at
this model size — the 459 W / 100% utilization was masked no-op
attention, not extra throughput.  Fresh bucket shapes also kept
appearing as late as iteration 5 (streams lengthen as the policy
shifts), re-paying ~30 s compiles.  Decision: the 2k run resumed EAGER
(job 139, --no-compile); eager still keeps two wins from this effort —
the top_p>=1 multinomial fast path (no per-step full-vocab sort) and
the SYNC_EVERY=16 finish check (no per-step D2H sync).  The compile
plumbing stays behind --compile; it becomes attractive again if the
model grows (launch overhead fixed, attention cost scales) or once
rollouts are batched, and the batch-all-16-groups lever now helps
eager directly (~6500 -> ~900 sequential step iterations).

Rollout numerics under compile differ from eager at bf16 scale; this is
ABSORBED by design: refresh_old_statistics recomputes every stored PPO
statistic through the same replay path the update consumes.

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

### Muon trunk optimizer for RL (2026-07-23)

`postraining/muon.py` + `--trunk-optimizer {muon,adamw}` (default muon): block
matrices (ndim>=2, both actor backbone and from-scratch critic trunk) step
under the exact pretraining Muon (NS5 x12 bf16, nesterov mu 0.95,
max(1,rows/cols)**0.5 scale); embed/readout/gains and all RL heads stay AdamW.
Weight decay 0 everywhere (post-training rule). `--muon-learning-rate`
defaults to lr*(0.025/0.015)=8.33e-5 at lr 5e-5 — the pretraining Muon rate
scaled by the same ~300x factor as Adam 0.015→5e-5. Caveat: at equal nominal
lr a Muon step moves each element ~sqrt(512)=23x less than AdamW, so the
default under-moves the trunk; if learning does not speed up, sweep
`--muon-learning-rate` {8.3e-5, 2.5e-4, 5e-4, 1.1e-3(RMS-match)}.
`--critic-muon-learning-rate` defaults to the actor value. Optimizer
checkpoint keys become {actor, actor_muon, critic, critic_muon}; pre-split
checkpoints (job 316 lineage) resume with `--trunk-optimizer adamw` (clear
error otherwise). Reviewed + red-teamed; tests in
postraining/tests/test_muon.py.

Bench/AIME diagnosis (job 316 run, cot 5e-5): NOT answer-format mismatch —
step-0 model passed exact-match on letters/bools/lists/decimals (closest 0.78,
sort 0.58, pair 0.55). DAPO's tiny-integer ground truths (median 2 chars)
mode-collapsed the policy onto a ~12-token `boxedboxed{N Answer:N` small-int
emitter: bench 24.5%→~2% (modal floor) by step 450, CoT gone, AIME pinned 0
(single-digit guesses vs 3-digit answers). Same story in guard/val_bpb
1.24→~1.9. User decision 2026-07-23: keep current run as-is.

### Latent-from-base cold start diagnosis (2026-07-23)

Latent RL from the same base checkpoint as the cot run (manifests confirm
identical `logs/nanomini_gpt2vocab_mathmix4k_2k_final_model.pt`) started at
exactly 0% accuracy / reward ~1e-4 vs cot's 1-2% / 0.023-0.044 at the same
steps. Cause is NOT the 50/50 forced first-thought: `--init-think-probability`
is a PER-DECISION gate probability, and at init-time DAPO stream lengths
(mean ~1322 actions, smoke job 318) P(zero THINK slots) = 0.9^N ~ 1e-14, so
every trajectory carried dozens of null-input slots the pretrained trunk has
never seen (the zero adapter zeroes thought content, not the slot itself) →
ended_fraction ~0.55, universal derailment, zero within-group reward
variance, advantages ~0, positive-LM starved. The 0.1 default came from a
Jul 18 measurement on the short-answer DeepMind task and does not transfer
to long streams. Compounding: `--gate-entropy-coef 0.01` overpowered the
(tiny, nearby-credit-only) negative think advantage — think fraction climbed
0.100→0.125 in 60 steps while reward stayed flat (job 319, cancelled).

Changes: `--gate-entropy-coef` default 4e-3 → 1e-4 (bonus must not outweigh
gate advantage on a zero-reward policy). Job 321 (0.1 init-think) cancelled
minutes in; job 322 `rl_gpt2vocab_latent_20k_ge1e4_itp01` relaunched with
`--init-think-probability 0.01` (user choice; ~5% think-free free-half
trajectories at N~300 — weak but nonzero signal; forced half still supplies
gate contrast). Aborted run dirs staged: job 319 (60 steps) at
postraining/runs/rl_gpt2vocab_latent_ge01_step60.

### Replay perf pass 1 + OOM root cause (2026-07-23)

Steady-state cycle measured on job 322: 11.1s collection (57%) + 4x1.7s
updates (35%) + ~1.7s pool transfer/refresh. First A1 attempt (attention
budget 4M->16M, max-traj 32->128, job 326) OOM'd in refresh at
logits_from_features: the shard planner bounded only the QUADRATIC
attention term (B*L^2) and row count, so fatter short-L shards grew the
LINEAR vocab term (slots x 50257 logits + autograd-retained log-softmax,
~6 B/elem) unboundedly. Also ~7 GiB of the card now belongs to parallel
workloads (cleanrl jobs, tts daemon), ceiling ~24 GiB.

Fixes (job 331, queued behind cleanrl runs): planner gained
--replay-slot-budget (default 8192; bounds B*L) — one mechanism sizes
shards for BOTH memory terms, refresh/update share it so the age-0
zero-clip canary is preserved by construction (a nested chunking helper
was considered and rejected: wrong layer, and per-shard fwd->bwd retains
the graph regardless of inner chunking). refresh's eager emit-logprob tail
now runs no_grad (compile-guard argument only covers the compiled
replay_head_inputs). Shared compact_emit_token_logprobs helper keeps
refresh/update forwards bit-identical. Refreshed device minibatches are
retained for their update (8 GiB budget, falls back to scatter+repack)
killing the d2h scatter + re-pack + re-upload (~0.2s/step). Train rows now
log UPDATE-phase peak VRAM (counter reset after collection). Also
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on the run.

Still open for the 2x target: the collection tail — ~85% of decode
iterations run 16-wide launch-bound (~5% never-ending think-heavy rows x
4 sequential chunks). Options: static-cache CUDA-graph tail (bench 93:
compiled-static 2.95 vs eager 5.10 ms/step at b32/c896; needs re-bench at
b16/c4096) or --max-stream-steps cap (semantics change, user call).

### Static-tail CUDA graph (--rollout-tail-graph, 2026-07-23, REJECTED by bench)

VERDICT (job 344, bench_step_compile b16/c3584, start 640): compiled static
row-mask (the tail-graph path) 2.119 ms/step vs compiled dynamic row-mask
(current production tail) 0.986 ms/step — 2.15x SLOWER. All full-width
variants cluster together (manual cuda graph 2.307, compiled static 2.076)
while narrowing paths win (eager narrow 1.502, dynamic 0.986): at 3584-key
cache length even a 16-row tail is KV-read-bound, not launch-bound, so the
fixed-shape graph pays full-width attention every step and loses more than
it saves. The eval-compile caution generalizes down to b16. Flag stays
default-OFF permanently; no smoke A/B. Code retained as a tested, inert
opt-in (correctness pinned by CPU tests) — delete if it becomes a
maintenance burden.

Implemented the load-bearing collection optimization: ~85% of decode
iterations run in the 16-wide compacted tail at ~0.5 ms/step launch-bound
overhead. New opt-in flag --rollout-tail-graph (requires --rollout-compile,
default OFF) switches that tail, at the fixed-size compaction, onto
persistent trainer-owned static caches (make_static_generation_cache,
16 x (prompt_tokens + max_stream_steps), bf16, ~0.7 GiB for the 6-layer
nano) and a reduce-overhead/cudagraph compile of step_core. The backbone
already had the needed fixed-shape branch (_attention_step: tensor
position + 2-D (batch, keys) key_mask narrows to the mask width); the
rollout now builds a full-width per-row mask combining causality with
left-pad validity, grown one column per step in place, so shapes are
constant and one graph replays per step. Stale cache values from earlier
rollouts stay masked by construction. Prior caution (eval-compile comment:
full-cache attention 2.6x slower than narrowing) was measured at 512-row
eval widths where attention compute dominates; the 16-row tail is
launch-bound, hence bench before enabling: job 333 reruns
bench_step_compile at b16/c3584 with two new variants (compiled dynamic
row-mask = production tail baseline; compiled static row-mask = the new
path; bench 93 at b32/c896 measured 2.95 vs 5.10 ms/step). CPU tests pin
trajectory equivalence vs the dynamic tail under mixed prompt lengths
(argmax policy — pad-mask corruption would flip tokens), dirty-cache
reuse, fixed dispatch shapes, and misfit-cache validation. Job 331 was
resubmitted as 334 (identical command) behind the bench; the flag stays
off until the bench and a short smoke A/B justify flipping it.

### v21: joint thought clip, reverse KL off (2026-07-24, user-directed)

User call (over my objection; prediction on record below): THINK actions now
clip their JOINT 512-dim Gaussian ratio exactly once per action (clip-higher
band 0.20/0.28, same as EMIT's joint gate+token clip); the optional gate
stays a separately clipped factor. --thought-reverse-kl-coef default
0.3 -> 0.0 and --thought-clip-mode {joint,per_dim} defaults to joint, so the
v20 factorwise-clip + reverse-KL objective survives only as an ablation arm.
EXECUTION_SCHEMA bumped to v21 (objective-only change; execution/RNG
identical to v20) with --migrate-joint-clip-resume gating v20-checkpoint
resumes, mirroring the v18->v19 reverse-KL migration precedent.

Run lineage: job 334 (per-dim/KL, killed at user request ~step 990) ->
job 345 resumes the same checkpoint/output under the new objective.

Measured basis for my prediction (steps 930-960): thought joint |log ratio|
max mean 3.6 (spikes 37) vs band 0.25; joint KL 0.075-0.18 nats/thought =>
typical thought sits at/past the band edge on aged rows. Watch metrics on
345: thought_policy_clip_fraction (predicted ~1e-4 -> 0.3-0.6 on aged rows),
harmful_positive_log_ratio_max (unclipped pessimistic side; the non-finite
actor-loss guard raises if exp overflows), trunk_grad_norm spikes,
think_fraction flattening (was rising 0.005->0.03), and the gate-vs-content
learning-rate imbalance ratcheting think probability down. If instead
reward/accuracy keep climbing with thought_clip moderate, my objection was
wrong at this sigma (0.05 floor keeps per-dim ratios tiny; the joint drift
is dominated by mean movement).

### v22: anchored value support, sigma 1.0 (2026-07-23, user-directed)

Critic support discussion (dg_v25_d3_s05_b51_exp donor) landed on: the
Dreamer3 symexp grid itself is mis-scaled for [0,1] verifier targets (all
targets would fall in ~3 of 51 buckets) and the expected-scalar decode was
already ours; the transferable ideas are the exact-zero bucket and sharper
labels. User picked: sigma_ratio 1.0 (was 2.0) + exact-zero bucket.

Implementation: `anchored_unit_geometry(interior, margin)` in
postraining/hl_gauss.py lays bin CENTERS exactly on 0 and 1 (Dreamer3's
zero bucket applied to both unit-range ends) with `--value-margin-bins 4`
bins beyond each anchor, so exact-0/exact-1 targets project symmetrically
instead of decoding a truncated half-Gaussian ~0.8*sigma inward (old grid:
+0.016 at 0; anchored: <1e-4, test-pinned). Defaults now
`--value-anchored-support` on, `--value-bins 101` = interior divisions
(width 1/101, total head 110 bins), `--value-sigma-ratio 1.0`; validation
enforces margin+0.5 >= 3*sigma_ratio. GAE bootstrap targets slightly
outside [0,1] now get real bins instead of clamping.

EXECUTION_SCHEMA v22; `--migrate-anchored-value-resume` resumes a v21
checkpoint by transferring critic trunk+adapter verbatim, rebuilding
head+support at the projected prior, keeping critic_muon (trunk matrices)
and all actor optimizer state, and dropping only the critic AdamW moments
(they include the old head). Decoded values collapse to the prior until the
head relearns from the transferred trunk features — expect a value-loss
spike and a few hundred steps of critic re-fit after migrating (no warmup
rerun; the skip is soft — nothing asserts on critic quality). Plain resume
now validates geometry ARGS, not head shapes (anchored 101/4 vs 103/3 both
build 110 bins but different sigma). Old grid reproducible via
--no-value-anchored-support --value-sigma-ratio 2.0.
Job 345 (joint clip, still queued — never started) would have failed the
v22 schema check with only --migrate-joint-clip-resume; cancelled and
resubmitted as job 348 (rl_latent_jointclip_v22_anchoredval) with
--migrate-anchored-value-resume added, so the joint-clip experiment starts
directly on the anchored critic. Both v21->v22 deltas (joint clip AND the
critic support) land in one run relative to job 334's v20 baseline — keep
that in mind when attributing metric changes.

### v23: projected THINK trust region (2026-07-23)

Job 348 (v22 joint clip, no KL) confirmed the joint-ratio clip cannot
govern a 512-D Gaussian: under behavior sampling the joint log-ratio is
~N(-kappa, 2*kappa), so at the measured drift (kappa~0.15) the +/-0.22
band clips ~68% of actions on SAMPLED NOISE (noise/signal ~ sqrt(2/kappa)
~= 3.6), and the clip is asymmetric in magnitude — favorable updates cap
at 1.28x while harmful ones run unbounded (observed |log ratio| up to
~3.6 => 36x). v20's stability was entirely the reverse-KL penalty (its
per-dim clip at 3e-5 nats/dim was inert). Thinking collapsed on 348
(optional think prob 0.034 -> 5e-4, think fraction 0.93 -> 0.006); user
killed it ~step 8900.

v23 (`EXECUTION_SCHEMA .../v23`) replaces the ratio-band trust region
with a TRPL-style MEAN PROJECTION measured in closed form — no sampled
ratio enters the trust decision:

- Pool stores behavior Gaussians: `old_thought_means` /
  `old_thought_log_sigmas`, fp32 dense, captured inside
  `refresh_old_statistics`'s parallel replay path (same forward as
  old_thought_logprobs), so at behavior age 0 the policy sits at
  Mahalanobis distance 0 BIT-EXACTLY (fp16 rejected: ulp ~17% of the
  per-dim signal at mean norm ~4.6 sigma).
- `project_thought_means`: d = ||(mu-mu0)/sigma0||^2; if d > eps_mu:
  mu~ = mu0 + sqrt(eps_mu/d)*(mu-mu0) (user's Eq. 6), gradient THROUGH
  the scale — Jacobian s*(I - uu^T) annihilates the radial component
  (nothing keeps pushing outward) and preserves tangential learning.
  A straight-through detach of s would keep the radial push; rejected.
- Surrogate: -exp(clamp(logpi~ - logpi0, +/-2)) * A with gradient
  through the ratio (unclipped PPO form; the trust region lives in the
  projection). The +/-2 clamp is a numerical guard vs the Gaussian
  tail, NOT a trust mechanism: with the mean projected, the joint
  log-ratio concentrates within ~sqrt(2*eps) of 0 at every age.
- TRPL tracking penalty: alpha * ||(mu - mu~.detach())/sigma0||^2 per
  THINK action, exactly 0 inside the region (`trust/projection_penalty`).
- Thought advantages are standardized — centered on the masked minibatch
  mean AND scaled by the masked std (floor 1e-6) — for the thought
  surrogate ONLY; the gate factor and EMIT keep raw advantages. The
  centering is load-bearing (review finding): scale-only normalization
  turns a low-variance pool (all trajectories sharing an outcome) into a
  mean/std gradient amplifier the isfinite guard cannot catch.
- Gate keeps its 1-D textbook clip; forced thinks (gate_mask=0)
  contribute only the Gaussian factor, as before.

Defaults: `--thought-clip-mode projected` (joint/per_dim kept as
ablation arms), `--thought-trust-epsilon 0.03` (user 2026-07-24: match
the TRPL reference BaseProjectionLayer mean bound 0.03; ~0.015 nats KL
at frozen sigma; the reference cov bound 1e-3 applies once sigma unpins
and the cov projection lands), `--thought-projection-penalty-coef 1.0`.
`--thought-log-sigma-init` stays -3.0: -2.5 was evaluated (sigma 0.082,
noise norm 1.86 = ~69% of a 2.7 thought-mean norm, ~36% at the 5.2
ceiling, vs ~42%/~22% at -3.0) and rejected by user 2026-07-24 as too
noisy; raising the mean-head init is the alternative if exploration
needs more range. Sigma remains unconstrained by the projection
(tanh-bounded head is the backstop; documented gap). Cov projection
deferred while sigma stays pinned. Telemetry: `trust/thought_d_mean`,
`trust/thought_d_max`, `trust/projection_penalty` — all three joined the
age-0 canary lists (must be exactly 0 on fresh behavior). Resume: v22
checkpoints need `--migrate-projected-thought-resume` (state transfers
verbatim; cumulative lattice back through v18 unchanged). Tests: 6 new
(projection identity, boundary Jacobian vs analytic s(I-uu^T),
d == 2*KL closed form, age-0 gradient == joint mode at ratio 1, ratio
guard saturation, refresh storage + age-0 exactness); 311 postraining +
83 repo tests green.

TRPL paper audit (2026-07-24, vs arXiv 2101.09207 + boschresearch/
trust-region-layers): mean projection algebraically EXACT vs Eq. 6
(mu0+sqrt(eps/d)(mu-mu0) == (mu+w*mu0)/(1+w), w=sqrt(d/eps)-1); distance
is the un-halved dimension-summed squared Mahalanobis under OLD cov —
apples-to-apples with the reference mean_bound 0.03; penalty form
matches Eq. 12 (projection detached); IS surrogate with projected
numerator and NO ratio clip matches the paper (their clip is "n.a." for
projections; our +/-2 guard is ~11 sigma out at eps=0.03, inert);
per-THINK-action granularity == paper's per-state; KL/W2/Frobenius
share this mean projection (they differ only in cov, moot while sigma
pinned). Deliberate deltas: act-with-raw + per-collection anchoring
(coherent — behavior stores the RAW acted means, age-0 ratio exactly 1;
surrogate has zero radial gradient outside the region so the penalty
only holds the boundary), and effective alpha=1.0 vs paper's tuned 8.0
(ablation candidate, not a bug — our unit-std advantages also shift the
balance vs theirs). Dimension caveat: 0.03 was tuned at d~=17; at d=512
it is ~24x tighter per dim (RMS 0.0077 sigma/dim), predicts ~75% of
inner updates projecting and ~3x mean-movement throttle vs the natural
~0.3/pool drift. Loosening does NOT reintroduce ratio-noise domination
(trust decision is closed-form on means); IS-variance ceiling ~eps 0.3
(log-ratio std sqrt(eps)). Audit suggests sweeping eps {0.03, 0.1, 0.3}
(0.1 expected sweet spot); dimension-proportional 0.9 would match the
natural step and effectively unconstrain. Starting value stays 0.03
(user call, conservative side).

Run plan (user: no fresh-v22 pair; judge against 348's telemetry): one
fresh 20k latent run from `logs/nanomini_gpt2vocab_mathmix4k_2k_final_model.pt`
with v23 defaults + `--init-think-probability 0.01`, replay budgets as
job 348. Success signals: trust/thought_d_mean bounded ~eps, projection
fraction a minority, optional think prob NOT collapsing, forced-vs-free
reward delta emerging.

20k run history (2026-07-23/24): smoke2 (job 358) passed clean; 20k
started as job 361 resuming the smoke checkpoint at step 40 (age-0
canaries exactly zero on the first post-refresh update — bit-exact
refresh confirmed in production), then restarted as job 363 at step 72
with `--thought-projection-penalty-coef 8.0` (paper Table 2 alpha).

TRPL re-audit no.2 (agent, pdftotext of the actual paper): NO new
misalignment. Corrections/confirmations: (1) act-with-raw is
paper-FAITHFUL — Algorithm 2 collects with the raw network policy and
anchors the region to the raw old forward pass; "acts with projected"
was a wrong mental model, the regression penalty is how the paper
itself closes raw~projected, not a mitigation for a delta. (2) Penalty
normalization verified: surrogate and penalty divide by the SAME action
denominator and the paper's d is also a dim-sum, so alpha=8 is
calibrated right (no hidden 512x); think-fraction dilution cancels
between the two terms. (3) Table 2 confirms mean bound 0.03, alpha 8.0,
cov bound 1e-3, importance clip n.a., entropy penalty 0 — our lack of a
thought-entropy bonus matches the paper base. (4) Latent risk: the
projected surrogate feeds UNPROJECTED sigma gradients into the log-sigma
head (paper's W2 projection bounds both moments); harmless while
log-std is pinned ~-3 (head update rms 1e-5) but must land together
with the cov projection if sigma ever unpins — treat sigma as
explicitly frozen until then.

alpha=8 empirical result (job 363, ~100 steps): NO effect on drift —
end-of-pool d_mean ~0.135 and joint KL ~0.068 nats, identical to
alpha=1; only the reported penalty scaled 8x (0.0004->0.0032).
Attribution: mean_head weights are static (rms 0.004403->0.004397) —
the drift enters through the Muon-owned TRUNK, whose belief updates
come from LM/gate/renderer losses over ALL actions. The penalty cannot
counter that channel: its trunk gradient share is diluted ~100x by the
think fraction and Muon renormalizes momentum to fixed magnitude
regardless of loss scale. Interpretation: the RL surrogate itself is
fully trust-region-bounded (exact projection, radial annihilation,
ratios <=1.8 inside the +-2 guard); the measured raw KL ~0.068/pool is
representation drift from the auxiliary objectives — a channel
single-objective TRPL never had — re-anchored every pool refresh.
Options if it must shrink: raise eps to 0.1 (aligns budget with
realized drift, audit's predicted sweet spot) or constrain the trunk
(rejected: fights the LM objective). alpha stays 8 (paper-faithful,
free).

### v24: Dreamer4 reverse KL REPLACES TRL (2026-07-24, user-directed)

TRPL audit no.3 (agent, vs the paper-faithful reimplementation in
`../cleanrl/cleanrl/shared/trl_projection.py`) found a FACTOR-OF-2
misalignment that audits no.1 and no.2 both missed. The reference
`kl_mean_part` returns `0.5 * maha` and compares THAT against
`mean_bound = 0.03`, so the paper's bound is 0.03 nats of mean KL.
`project_thought_means` compares the UN-HALVED `mahalanobis_sq`
against the same 0.03, i.e. 0.015 nats — the region is 2x tighter than
the paper's, and audit no.1's "apples-to-apples with the reference
mean_bound 0.03" (line ~380 above) is wrong. Verified numerically:
identical shifts give distances differing by exactly 2.000 and
post-projection boundary radii by exactly sqrt(2)
(0.008438 vs 0.011934 at sigma = e^-3.02). The same un-halving is in
the tracking penalty (`projection_gap.square().sum()`, which also
normalizes by behavior sigma where the reference uses the PROJECTED
sigma), so alpha 8 is effectively 16 in paper units — moot, since
alpha measurably does nothing. Per-dimension the bound is 7.75x
tighter than the paper's setting at d=17 (0.00766 vs 0.0594 sigma/dim);
sqrt(2) of that is the bug, 5.5x is d=512 vs d=17.

Decisive finding: the 0.03 that matters is NOT the one the projection
bounds. `new_thought_logprobs` is built from the RAW `thought_means`, so
`kl/thought_behavior_joint` and `trust/thought_d_mean` both measure the
raw ACTING policy. Job 363 over 17,499 updates: median raw KL 0.0424
nats, p90 0.0671, p99 0.0799, max 0.5087, **68.4% of updates above
0.03** and 36% above 0.05; `d_max` reached 179.67 (~90 nats on one
action). `d_mean` exceeded eps on 75.0% of updates, matching audit
no.1's ~75% projection-rate prediction exactly. The projection bounds
only the projected mean inside the surrogate; the raw drift arrives
through the Muon-owned trunk, which no projection or tracking penalty
can reach (see the alpha=8 result above). There is NO KL-triggered
early stop anywhere — `--post-update-kl-every` is pure telemetry.

v24 therefore turns the Dreamer4 reverse-KL penalty back on as THE
default trust mechanism, and REMOVES the projection from the default
path entirely: `--thought-reverse-kl-coef` 0.0 -> **0.5** (user call;
dreamer4 parity would be 0.3), `--thought-clip-mode` projected ->
**none**. The two are now MUTUALLY EXCLUSIVE by construction -- see the
biconditional below. Dreamer4 reference for the coefficient's
`pmpo_kl_div_loss_weight` (`../dreamer4/dreamer4.py:5247`, with
`pmpo_reverse_kl = True`; its HalfCheetah script uses 0.3, cartpole
0.05). Its `kl_div` is `KL(behavior || current)` summed over action
dims and masked-meaned over positions — same direction and same
normalization convention as `sampled_reverse_kl` here, so the weight
transfers. Caveat: dreamer4 balances it against a PMPO loss
(`advantage.tanh().abs()` weighting), not a PPO surrogate on
unit-normalized advantages, so 0.3 is parity-by-construction, not a
tuned value for this objective.

- New `--thought-clip-mode none`: no projection, no ratio band, no
  tracking penalty, leaving reverse KL as the sole constraint. It reuses
  `projected_thought_policy_loss` with an identity projection and an
  all-ones trust scale, so the objective SHAPE is identical to
  'projected' and exactly one term differs — the clean A/B. The +/-2
  log-ratio guard still applies (numerical, not a trust region).
  Rejected at coefficient 0: that is the unconstrained 512-D surrogate
  job 348 already showed cannot bound drift.
- EXACTLY ONE trust mechanism per run, enforced as a biconditional
  (nonzero coefficient <=> clip mode 'none') in BOTH `update_minibatch`
  and argv (`validate_args`). The surrogate-side modes and the penalty
  are alternative SOLUTIONS -- the clip modes bound movement inside the
  surrogate, the penalty prices realized aggregate divergence of the
  acting policy -- so mixing them makes neither term's contribution
  attributable. Arms: reverse-KL only (`--thought-clip-mode none`, the
  DEFAULT) or TRL only (`--thought-clip-mode projected
  --thought-reverse-kl-coef 0`). The mixed configuration that v24
  originally shipped is now REJECTED, not merely discouraged.
- `clip/thought_projection` finally surfaced. `thought_policy_clip_fraction`
  has been computed since v23 but never reached the dashboard dict, so
  the projected arm's headline diagnostic ("projection fraction a
  minority", the stated v23 success signal) was invisible for all of
  job 363. Joined the age-0 canary list (exactly 0 on fresh behavior).
- `EXECUTION_SCHEMA` -> v24 (`..._reverse_kl_thought_trust_...`; the
  first-cut name encoded the now-forbidden mixture and was replaced
  before any checkpoint carried it);
  v23 becomes `NO_THOUGHT_KL_EXECUTION_SCHEMA`; resume needs
  `--migrate-thought-reverse-kl-resume`, which is rejected together with
  `--thought-reverse-kl-coef 0`. Every older schema now additionally
  requires the new flag. State transfers verbatim.
- `main()` split into `build_arg_parser()` + `validate_args(parser,
  args)` so the shipped defaults and the argv guards are testable
  without running training; backbone-dependent checks stay in `main`.
  `update_minibatch`'s signature defaults now track the CLI defaults,
  so a caller that omits them exercises the SHIPPED arm.
- Tests: none-vs-projected once the region binds; none rejects
  coefficient 0; every surrogate mode rejects a nonzero coefficient
  (parametrized); shipped defaults pinned in both the signature and the
  parser; argv guards exercised through `validate_args`. Migration
  lattice extended. 326 postraining + 83 repo green (409 total).

DELIBERATELY NOT CHANGED: `--thought-trust-epsilon` stays 0.03. In this
convention that enforces 0.015 nats, inside the user's stated 0.03
ceiling; "fixing" the halving to reach paper-0.03 would LOOSEN the only
constraint currently being enforced while the raw policy is the thing
out of bounds. Revisit once the raw KL is actually under 0.03. The
paper-unit conversion is: current eps 0.03 -> 0.015 nats, 0.06 ->
0.030 nats (paper Table 2), 0.10 -> 0.050 nats (audit's predicted sweet
spot), 0.30 -> 0.150 nats.

STILL OPEN: a target-KL early stop on the inner epochs is what would
make 0.03 a hard ceiling rather than a soft penalty. Not implemented.

### GPU-utilization audit (2026-07-24, read-only, job-348-era telemetry)

Collection is CPU/latency-serialized, not compute-bound. Steady-state pool
cycle ~2.66s: rollout generation 1.29s (48%), 4x PPO update 1.00s (37%),
refresh 0.25s (9%), pack+H2D 0.12s (5%). collect_seconds is bimodal
(median 1.37, p90 5.74, max 38) — set by the longest survivor per
256-row chunk; the 82%-busy / 30%-mem-controller / 315W signature is tiny
decode kernels (16-wide tail) + GPU-idle gaps during inline CPU scoring.

Ranked findings (impact/effort):
1. F1 inline CPU scoring serializes chunks (train_latent_vapo.py:4101-4110
   -> score_math_rollout): tokenizer.decode + regex for 256 rows runs on
   the main thread BEFORE the next chunk's rollout launches; GPU idle.
   Est 0.2-0.5 s/cycle. Fix: thread-pool/double-buffer scoring.
2. F2 progressive compaction unreachable (latent_rollout.py:532-547):
   with finished_batch_size=16, batch keeps full 256 width until <=16
   survive (~90-120 steps of ~240 dead rows); compact_finished is coupled
   to rollout_tail_batch (:4021/:4071) though rollout_step_core compiles
   dynamic=True and could take intermediate widths. Est 0.2-0.4 s.
3. F3+F4 16-wide tail under-occupies (30% mem-controller, paid 4x
   sequentially) + max_stream_steps uncapped (core.py:28-31; the 5-38s
   outliers). Mitigations: cap --max-stream-steps ~256-512 (objective
   tradeoff: truncates longest, mostly-failing answers), widen/merge tails.
4. F5 replay H2D pageable + blocking (:4522/:4676): pin + non_blocking +
   overlap pack with refresh. ~0.05-0.12 s.
5. F6 bool(think_mask.any()) / emit_mask.any() host syncs per replay shard
   (:1869,:2072,:2408,:2432). Drop guards. ~0.02-0.1 s.
6. F7 Muon NS5 per-matrix Python loop (muon.py:66-73): batch same-shape
   matrices into bmm. ~0.02-0.05 s/step.

F1+F2 alone plausibly reclaim ~15-35% of the cycle.

Implemented 2026-07-24 (user call: v23 run had not started, so land them
before it): F1 (1-worker ThreadPoolExecutor scores offloaded chunk N
while chunk N+1's rollout launches; executor owned by collect()'s
try/finally; on-device/diagnostic paths stay sequential), F2
(progressive compaction above the fixed tail under the existing
>=25%-dead hysteresis; intermediate widths never equal the tail size so
the static tail graph cannot engage early; dynamic-shape compiled step
absorbs the widths; NOTE same-seed rollouts differ from pre-change code
since noise draw shapes change), F5 (pack_rollout_groups_for_replay
pin_memory + minibatch N+1 packed on a worker during N's H2D+refresh +
non_blocking uploads, also in the update-loop repack fallback), F6
(iter_length_aware_microbatches yields host row lists; per-shard
bool(mask.any()) syncs in update_minibatch and the drift diagnostic
replaced by one CPU table per minibatch/batch). Skipped: F7 batched-Muon
(perturbs verified optimizer numerics for ~1-2% cycle), F3/F4
(run-config and objective tradeoffs, not code: --max-stream-steps cap
remains available per-run).

Smoke job 356 (40 steps, TORCH_LOGS=recompiles) verdicts:
- Recompiles: step_core recompiled exactly ONCE (static->dynamic on the
  first width change 128->70) and never again across all progressive
  widths — the dynamic=True two-artifact story holds; CUDA-graph /
  reduce-overhead alternatives remain rejected on job-344 data (static
  full-capacity attention 2.1-2.3x slower than dynamic narrow).
  Pre-existing bounded extras: muon per matrix shape, one
  0/1-specialization each for value_logits/replay_head_inputs from a
  1-trajectory shard.
- OOM caught: the realloc compaction branch (full-capacity target per
  cache tensor) OOMed at width 70 during peak KV pressure — the old
  code only ever ran it at <=16 rows late in the stream. Fix: the
  progressive above-tail case now compacts IN PLACE (survivor rows
  gathered to the front of existing storage, narrowed contiguous
  views); storage is freed at the tail snap exactly as before, so the
  peak memory profile matches job 348 while dead-row compute still
  drops. Tail-snap and tail-free (None) paths keep the realloc
  behavior unchanged. Re-smoked as job 358; 20k run queued as 359.

## Post-training perf pass, 2026-07-24 (v23 run in flight as job 363)

CORRECTION, same day, before trusting anything below: the numbers I first
wrote here as "steady state (last 100 pools)" were actually a MID-RUN
window (~step 2000 of 4248). Job 363 is not stationary -- the policy is
learning to answer with fewer actions, and every collection cost tracks
that. Do not quote a window from this run without its step range.

  window (steps)     collect  actions/traj
  44-440              13.20 s   417.4
  1944-2340            8.93 s   154.1   <- what I originally called steady
  3048-3444            8.38 s   104.4
  3852-4248            6.25 s    83.4   <- current

Current pool (last 100 pools, steps 3852-4248): collect 6.25 s, refresh
0.541, d2h 0.198, cpu pack 0.198, h2d 0.065; update 0.566 s/step.
actions_per_trajectory 83.3, thoughts_per_trajectory 2.17, packed padding
utilization 0.255, packed_batch_gib_max 2.59, retained_minibatches 3.59,
ended_fraction 0.998.

The pool split, taken from pool_seconds (cumulative from pool start, so
the last train row of each pool IS the pool wall time) rather than
reconstructed:

  window        pool wall   collect          update
  first 100      18.88 s    13.17 (69.7%)    5.91 (31.3%)
  middle         12.25 s     9.23 (75.4%)    3.12 (25.5%)
  last 100        8.55 s     6.30 (73.7%)    2.43 (28.5%)

**Generation is 70-75% of pool wall time and that share is STABLE across
the whole run.** There are 4 updates per pool (prompts_per_rollout 64 /
prompts_per_minibatch 16), not 16 -- I briefly published a "generation is
only ~41%, the update path is now the larger half" correction here that
was built on 16 updates/pool. That was wrong; it inverted the ranking on
a bad divisor. Generation was and remains the right target. The original
"~60%" was a mild underestimate, not an overstatement.

Within collect, generation proper is 64.6% of pool wall; the remaining
~9 points are refresh 6.6, cpu pack 2.5, d2h 1.8, h2d 0.8. (collect_seconds
brackets all of those, which is why it reads 73.7%. Both numbers are right
-- quote which one you mean.) The decomposition closes to a 0.013 s
residual in every window, so this accounting is exact, not approximate.

RETRACTED, and this is the important one: I wrote here that
ended_fraction 0.998 meant the "~2% of rows run to max_new_tokens=1024"
premise had "largely dissolved". WRONG, and backwards. 0.2% of 1024 rows
is ~2 unfinished rows per pool, and under lockstep ONE such row in a
256-row chunk forces the entire chunk to the full budget. Direct evidence
from the bench rows, which is not subtle:

  step   emitted_tokens          recurrent_steps_per_rollout
         mean   p95    max       mean    p95    max
  3752   108.4  1024   1024      1048.7  1055   1055
  4052    87.6  1024   1024      1042.6  1047   1047
  4352    68.3   150   1024      1044.0  1050   1050
  4500    68.0   125   1024      1039.8  1045   1045

Every chunk runs ~1044 steps to serve a mean of 68 emitted tokens -- a
15x waste factor that is FLAT while emitted_tokens_mean fell 108 -> 68.
Note p95 == max on the step count: that IS the lockstep signature, every
chunk pays its slowest row. Bench ended_fraction is 0.92-0.96, not the
0.998 of the training rollout -- different populations, don't mix them.

Corroborated independently from the training side: packed_batch_gib_max
inverts exactly to a packed stream width (8244 B/slot at B=256 -- four
(B,L,512) fp32 fields = 8192 B plus ten scalar-per-slot fields = 52 B).
Over the last 120 pools that gives mean 1260, median 1280, max 1600, and
every single value lands on an exact multiple of 64, which is what
trim_stream(multiple=replay_bucket=64) produces. Meanwhile
rollout_diagnostics' stream_length reads 407 because aggregate_diagnostics
takes the MEAN over the 64 groups, not the max. So one group per minibatch
is still running out past 1200 slots in essentially every pool.

So the runaway rows are the #1 item and are relatively BIGGER than at
mid-run, not smaller. ~2 rows in 1024 set the step count for the whole
pool.

What the drift does change, and these do hold:
- d2h fell 0.732 -> 0.198 and retained_minibatches rose 2.74 -> 3.59
  (shorter streams fit the 8 GiB budget), so both the scatter win and the
  RETAINED_MINIBATCH_BUDGET_BYTES bump are worth proportionally less than
  the mid-run numbers suggested.
- Absolute headroom everywhere is smaller: the whole pool is now 8.55 s
  against 18.88 s early. Percentages are the honest unit here, not
  seconds.

Method note, since I got this wrong once: do NOT reconstruct pool wall
time as collect + N x update_seconds. Read it off the last train row's
pool_seconds, and get N from prompts_per_rollout // prompts_per_minibatch
in the manifest, never from a ratio of record-slice lengths.

What survives unchanged: per-step decode cost is 1.94 ms, measured
independently from the bench rows and the aime rows and corroborated by
the training regression slope (1.8-2.9 ms/step), and it is nearly FLAT in
batch width -- so narrowing the batch (the prior audit's F2) does not
touch the price. Chunks still step in lockstep until the last row
finishes. The changes below are all overhead removal that scales with
STEP COUNT, which is why they hold up as the run's action counts fall.

Landed (all bit-exact against the prior implementation unless noted):

- latent_rollout.py decode loop: the six per-step boolean-row-mask stream
  writes became torch.where over the dense (live_rows, position) index.
  A boolean index has a data-dependent output shape, so each one copied
  its count to the host and drained this launch-bound loop -- six syncs
  per step, which made the SYNC_EVERY guard pointless. Verified bit-exact
  over 320 randomized trials (pin_emit x record_likelihoods x
  replay_storage, including compaction permutations) and, independently,
  field-by-field against HEAD over 14 rollout configurations.
- latent_rollout.py scatter_replay_statistics: narrow each group's slice
  ON DEVICE before it crosses the bus. The packed batch is padded to its
  longest group, so the whole-batch transfer moved ~2.4x the bytes any
  group keeps. Destinations stay PAGEABLE on purpose: an earlier pinned +
  async version was reverted after review because these buffers become
  the pool's long-lived behavior statistics (order of GBs) and this box
  is 60 GB with 31 GB already in swap -- pinning that much is worse than
  the bandwidth is worth. The pinned variant remains available if the
  host memory picture changes.
- core.py generalized_advantage_and_return_targets: hoisted the residual,
  the shifted validity, and gamma*lambdas out of the reverse recurrence;
  15 -> 8 eager launches per column. Bit-exact across dtypes, gammas,
  scalar/per-row lambdas, gappy masks, and length 0/1.
- train_latent_vapo.py update_minibatch: the two per-shard
  torch.isfinite(loss) guards were host syncs at maximum queue depth,
  immediately before backward -- roughly two dozen per minibatch. They
  now accumulate on device and resolve in ONE sync after the shard loop,
  still above every step_optimizers call in the function (including the
  value_only return path). Tests assert weights are unchanged after the
  raise. This is the one deliberate behavior change: a poisoned shard now
  wastes its backward pass before aborting.

New telemetry: perf/decode_steps_per_chunk_mean, decode_steps_per_chunk_max,
decode_step_utilization (lockstep_decode_metrics). A chunk pays one decode
step per action of its LONGEST row; the utilization ratio against mean
actions is the share of decode work that produced nothing. This is the
number that decides whether the max_stream_steps / rollout_groups change
below is worth an execution-schema bump. Row length is read from
action_mask, NOT from stream_length: groups reach the metric through
trim_stream, which rounds the kept length UP to --replay-bucket (64) and
pads past the original stream to bound compiled replay shapes. The first
version of this metric measured stream_length and so carried up to a full
bucket of pool-to-pool jitter -- fatal for a metric whose entire job is
detecting a change. Caught in review; the unit test now pins that
bucketing moves nothing.

Operational, and the single largest wall-clock item found: the machine
auto-suspended 00:00:00 -> 08:57:22 (8 h 57 m, ~half the run's wall
clock) and will do so again nightly. Confirmed via journalctl, not
inferred from a telemetry gap (telemetry completeness checked at 0.3%
over a live 21-minute window). Fix is to wrap the runner:
`systemd-inhibit --what=sleep:idle mlq daemon ...` (or the submit itself).

Open, deliberately NOT done -- each needs the GPU, which job 363 holds:
- max_stream_steps 3584 -> 1536 or 2048, and --rollout-groups 16 -> 32.
  KV is allocated at full num_heads (make_generation_cache uses
  attention.num_heads, NOT num_kv_heads -- GQA does not shrink it) for the
  whole prompt+3584 window while stream_actions_max is ~1050. Confirmed by
  a second pass that max_stream_steps gates ONLY validation, allocation
  shapes, and the loop bound: no RNG draw shape depends on it (gate.sample,
  sample_latent, and top_p_sample all draw at the live row width), the step
  count is bound by max_new_tokens=1024 and never reaches either budget, and
  resume validates only execution/prompt-order/data schemas, not this arg.

  Footprint is B x max_stream, and this is the bit I had WRONG:

    config                          B x L        vs today
    today      (256, 512+3584)   1,048,576         --
    2048 steps, groups 32 (512, 2560)  1,310,720   +25%  (+3.5 GiB)
    1536 steps, groups 32 (512, 2048)  1,048,576   exactly neutral

  Doubling the chunk width doubles B, so 2048 + groups 32 is NOT the
  memory-neutral trade I implied -- 1536 is. Also: my "2.15 GiB thoughts"
  was a GB/GiB slip; B*L*512*4 = 2^31 B = 2.000 GiB exactly, and the freed
  amount at 2048/groups-16 is 5.27 GiB, not ~5.5.

  Measured headroom (last 100 pools, verified directly from metrics.jsonl):
  rollout peak_vram_bytes mean 17.10, p90 18.68, max 18.93 GiB; train peak
  mean 13.44, max 14.22. The (512, 2560) projection lands ~20-22 GiB on a
  32 GiB card, so 2048 fits -- but benchmark at 1536 FIRST, because it is
  the single-variable test: any per-step change is attributable to width
  alone rather than to a 25% larger footprint picking different autotune
  kernels. Raising 1536 -> 2048 later needs no second schema bump.

  Sequencing: max_stream_steps alone needs NO execution-schema bump (it
  changes no draw shape and no step count); --rollout-groups 32 does,
  because it changes which RNG draws map to which row. Land them as two
  steps. A v24 bump is five mechanical edits around EXECUTION_SCHEMA
  (train_latent_vapo.py:156) plus the resume cascade at :210-289 and the
  test matrix at test_latent_rollout.py:2985-3020; 363's checkpoints CAN
  resume across it behind a --migrate flag (the v19->v20 precedent:
  identical prompts/policy/parameters, only RNG-to-row attribution moves).
  Behavior-age-0 bit-exactness does not interact -- it is a property of the
  replay path, which never sees chunk width, cache length, or rollout RNG.

  One caveat that is a HYPOTHESIS, not a finding: attention narrows the KV
  cache with torch.narrow on dim 2, so strides are cache_length-dependent
  and Inductor bakes that length as a literal under
  max-autotune-no-cudagraphs. A different constant could select a different
  kernel and hence a different bf16 reduction order -- mathematically
  identical, not bitwise. Directly testable by running groups=16 +
  max_stream_steps=2048 at a fixed seed and diffing.

  Break-even, against the pool-aligned decomposition (only the 5.38 s of
  generation moves -- refresh, pack, d2h and h2d are per-minibatch work on
  a fixed 4x256 partition and do not scale with --rollout-groups). Halving
  the chunk count also makes each chunk likelier to contain a row that runs
  to the cap: P(chunk holds one) goes 1-0.998^256 = 0.40 to 1-0.998^512 =
  0.64, so total steps fall by ~0.58, not 0.50.

    r = c(512)/c(256)   generation   pool    change
    1.00                   3.12      6.07    -27%
    1.15                   3.59      6.54    -21%
    1.20                   3.74      6.70    -20%
    1.40                   4.37      7.32    -12%
    1.60                   4.99      7.94     -5%
    1.72                   5.38      8.33      0%   <- true break-even

  Book -20%, not the -29% I first estimated off collect_seconds. Kill line
  r > 1.4. Expected r is 1.15-1.30.

  BENCHMARK BLOCKER -- addressed in code, NOT VERIFIED ON HARDWARE.
  bench_step_compile.py held five live KV cache sets in one --batch-sizes
  iteration (12288*B*L each = 6.375 GiB at B=512/L=1088, so 31.9 GiB) and
  OOMed before reaching the variant we care about. It now takes --variant,
  and each block drops its caches, closures and captured graph before the
  next allocates. Two things that made this more than a del: a CUDAGraph
  owns a private pool that only returns to the allocator when the GRAPH
  object is collected, so it must go before its caches; and position_index
  was defined inside the compiled-static block but consumed by dynamic-row
  and static-row, so a naive selector would have failed with NameError. It
  is hoisted.

  What is actually verified: an AST pass confirms position_index is bound
  in shared setup ahead of the first guard, that no variant reads a name
  bound by another, and that each frees every name it binds; argparse is
  exercised for the no-arg, single, multi and invalid cases; the default
  runs all five in the original order. What is NOT verified: that the peak
  is really one cache set. That is a claim about allocator and cudagraph-
  pool behaviour and it needs a run with torch.cuda.max_memory_allocated
  to confirm -- job 363 holds the GPU. Treat "~6.4 GiB peak" as the design
  intent, not a measurement. Run --variant dynamic-row at B=256 first as a
  cheap smoke test before spending a slot on the B=512 comparison.

  Bias note for whoever runs it: running a subset skips the eager block,
  which is currently the first CUDA work in the process and incidentally
  warms cuBLAS handles and SDPA backend selection. That work moves into
  the selected variant's own compile plus the 64 warmup steps, which are
  timed separately and discarded. Whatever residual bias remains applies
  identically at B=256 and B=512, so the RATIO -- the decision metric -- is
  unaffected. Do not compare absolute ms/step across different --variant
  sets.
- RETAINED_MINIBATCH_BUDGET_BYTES 8 -> 13 GiB -- deferred. Worth less than
  the ~0.96 s/pool I estimated now that retained_minibatches is 3.59. Do
  NOT change it in the same job as a width change: different phases, but
  they surface on the same peak_vram_bytes field, so you would not know
  which one moved.
- Capping the stream budget truncates ~3-6% of rows by the eval length
  percentiles, so it is a real objective change and needs an ablation,
  not a free win. The MOTIVATION is now the strongest item on this list
  (see the retraction above): chunks run ~1044 steps for a mean of 68
  emitted tokens. If chunks stopped near the p99 row instead of the max,
  generation goes ~5.38 -> ~2.4 s, i.e. **-36% of pool wall** -- larger
  than everything else here combined, and it collapses the packed width
  (2.59 -> ~0.9 GiB) which fixes retention and ~75% of the GAE scan for
  free. Two separate costs to accept, and they are the whole difficulty:
  changing when the loop stops changes the RNG stream for the rest of the
  chunk (schema bump), and truncating a row changes its reward (objective
  change, for ~2 rows/pool that are emitting garbage anyway). The
  objective-NEUTRAL variant is to merge the four chunks' tails so tail
  rows decode together. USER DECISION, not mine to make: this trades a
  small objective change for the largest measured win in the run.

  TAIL MERGE, specced (objective-neutral alternative to the above). Per-row
  `position` is the enabling change and is SMALL inside the compiled step
  -- three sites, all verified by reading nano_backbone.py:

    (a) :142-144 RoPE. theta = position * angular_freq then
        .cos()[None,None,None,:]. A (B,) position makes theta (B, 64), so
        the reshape becomes [:, None, None, :]. Two lines.
    (b) :182-183 KV write. cache[0].index_copy_(2, index, k) writes ONE
        slot for every row; per-row needs scatter_ with the index
        broadcast to (B,H,1,D). Flattening to a linear index does not work
        -- element (b,h,l,d) sits at b*HLD + h*LD + l*D + d, so a per-row l
        is not a uniform stride.
    (c) :186-188 mask. Add arange(W)[None,:] <= position[:,None], else a
        row at position 40 attends slots 41..W holding stale KV.

  The piece of luck, confirmed at :181: position_length = key_mask.shape[-1].
  The attention extent already comes from the MASK WIDTH, not from
  position, so torch.narrow at :184-185 needs no change at all.

  Cost: one extra compile at startup (dynamo guards on rank, () -> (B,) is
  a new graph; dynamic=True already covers B and mask width, no new dynamic
  dim). The KV store stops being a contiguous slice store and becomes a
  real scatter -- same bytes, still coalesced within a row since D=128 is
  contiguous, but B separate 256-B runs per head. That is the one place to
  expect a regression.

  Per-row position is NOT a general win: SDPA extent becomes max(position)
  over the live set, so a row at 40 batched with a row at 1000 pays 1000
  columns. It converts "every row pays the slowest row's STEP COUNT" into
  "every row pays the slowest row's ATTENTION WIDTH". It pays only when
  merged rows have similar positions -- which the tail case satisfies,
  since parked survivors all ran near the cap.

  It also is not sufficient. Chunks are generated SEQUENTIALLY
  (train_latent_vapo.py:4550), so when chunk 1 is in its tail, chunks 2-4
  do not exist yet. Merging needs PARKING: run each chunk to its tail, park
  that tail's KV, finish all four together. A parked 16-row tail is
  12288*16*4096 = 0.75 GiB, so three parked tails ~2.25 GiB against the
  measured 18.93 GiB peak -- affordable. Merging is a ~3 GiB D2D copy,
  ~2 ms. Chunks have different prompt widths, so the merged buffer needs a
  per-row pad offset, which valid_slots already expresses.

  VALUE -- re-derived, and the estimate I was handed (tail ~60% of steps,
  from the mid-run window) is stale in BOTH directions. The tail fraction
  is volatile and tracks the policy's answer-length distribution:

    step   emit mean/p95      chunk steps   head   tail   tail%
    3900   108.1 / 1024          1044       1026     18     2%
    4052    87.6 / 1024          1043       1026     17     2%
    4200    76.6 /  423          1041        425    616    59%
    4352    68.3 /  150          1044        152    892    85%
    4652    63.7 /  130          1038        132    906    87%

  At step 3900 this change was worth almost NOTHING (tail 2%); it is now
  worth 85-88%. Any estimate here must be dated.

  THE VOLATILITY IS A STRONGER ARGUMENT THAN EITHER ESTIMATE. Going 2% ->
  88% in ~750 steps means the prize is a function of the policy's current
  answer-length distribution, not of the code. A schema bump justified by
  today's spot measurement can be worth nothing by the time it lands. That
  cuts against implementing either variant on a single reading, and it
  applies equally to the -31% and -42% figures. What it argues FOR is
  landing decode_step_utilization first (already done, costs nothing) and
  watching the ratio across several hundred steps before committing.

  Direction of the error: both caveats below bias the SAME way -- p95
  approximates p93.75 at 16 of 256 rows, which reads the head high and the
  tail low -- so 85-88% is if anything a slight underestimate.

  At the current 87%:
  4 x 1040 = 4160 steps becomes 4 x 132 + 906 = 1434, saving ~66% of
  generation, i.e. 5.38 -> ~1.85 s and pool 8.33 -> ~4.8 s, about -42%.
  Caveat: this uses BENCH row-length percentiles as a proxy for the
  training rollout's distribution, and they are different populations
  (bench ended_fraction 0.92-0.96 vs training 0.998). decode_step_utilization
  from the new telemetry settles it directly on the next run.

  SUBSTITUTES, NOT ADDITIVE: the tail merge and the runaway-row truncation
  above target the SAME steps. Truncating near p99 removes the tail
  outright (~87% of generation); merging makes the tail be paid once
  instead of four times (~66%). Do not add them. The merge also overlaps
  --rollout-groups 32 -- with two chunks instead of four it saves 1T not
  3T -- so if the width change lands first, re-estimate the merge against
  the post-change baseline, not this one.

  Cross-check on the whole picture: 4 chunks x 1040 steps at the derived
  1.14 ms/step reproduces the measured generation time, so the "generation
  is chunk_count x chunk_steps x per_step" model is sound.

- GAE runs over the PADDED rectangle. generalized_advantage_and_return_
  targets is called once per minibatch on (256, ~1285) while
  packed_padding_utilization is 0.256, so ~75% of the scan is pure
  padding: ~10.3k eager launches per update, ~41k per pool. Estimated
  0.22-0.30 s/pool (~12-15% of the update half) on a 5.5 us/dispatch
  figure carried over from the decode-loop work -- NOT measured, and a
  torch.profiler range around train_latent_vapo.py:1970 would settle it
  in one pool. It is the cheapest measurement on this list. The fix is
  bit-exact if wanted: the recurrence is row-separable and padding
  columns contribute exactly 0 (mask=0), so it can move inside the
  length-aware shard loop, which already narrows to each shard's used
  length. Not done -- unmeasured, and this project does not land
  speculative changes.
Rejected on risk/reward: top_p_sample fp32 fusion (RNG path, schema
bump), the 18 boolean->integer index conversions in update_minibatch
(~0.02 s/pool), replay_bucket and bpb_val_tokens default changes, and the
compact-THINK-indexed old_thought_* refactor.


## v24 launch config + perf pass (2026-07-24, job 385)

OBJECTIVE (user-directed): reverse-KL ONLY, never mixed with TRL.
`--thought-clip-mode none` + `--thought-reverse-kl-coef 0.5` are the
defaults; the biconditional (nonzero coef <=> mode 'none') is enforced
in `update_minibatch` and in `validate_args`. `--value-prior` 0.05 ->
0.0 (critic-warmup agent: KL(optimum || project(prior)) is 0.68 nats at
0.0 vs 9.84 at 0.05, and the near-frozen AdamW bias took ~1e3 steps to
unwind that). Takes effect on FRESH warmup only -- `value_prior` is not
in `execution_schema`, so a resume silently keeps the old bias.

PERF, measured (steps 6, --value-warmup-steps 0, 2nd pool = steady):

| config                          | collect_s | pool_s | peak VRAM |
|---------------------------------|-----------|--------|-----------|
| g16 baseline (256-wide x 4)     |  7.03     | 9.01   | 18.76 GB  |
| g32 + --max-stream-steps 2048   |  6.45     | 8.46   | 18.99 GB  |
| + --rollout-tail-graph          |  5.61     | 7.34   | 19.32 GB  |

= 20% off collect, 19% off the pool cycle, ~+0.6 GiB VRAM.
`--rollout-tail-graph` had been off "until bench_step_compile confirms
it at the production shape" (:3426 comment). CONFIRMED: 52% of decode
steps run at the 16-row tail where the step is launch-bound (60 eager
dispatches/step, 0.31 ms of work inside a 1.94 ms step), so the CUDA
graph wins exactly as that comment predicted.

NEGATIVE RESULT: naive widening does NOT help by itself. An earlier
rollout-only A/B suggested g32 was SLOWER; that measurement was
confounded (power-trace window included post-decode CPU). Instrumented
`collect_seconds` shows g32 is 8% FASTER. Widening beyond that is
pointless: attention decode has arithmetic intensity exactly 1.0
FLOP/byte against a 5090 ridge of 117, and KV streaming is >=70% of
decode bytes, so a wider batch is MORE memory-bound, not less. 1024-wide
is structurally impossible (48 GiB of KV at cache length 4096).

POWER: 500W is NOT reachable during decode on this model -- roofline,
not syncs and not occupancy. Measured mem-controller util 44-47%, SM
util 96%, power 341W: a dispatch-bound loop punctuated by
bandwidth-bound kernels. Average power rises only by shrinking decode's
67-75% share of pool wall time so the compute-dense replay/update
occupies more of it. Pure rollout peaked at 365W; full training at 396W.

REJECTED: replacing `torch.multinomial(probs,1)` with an inline
gumbel/exponential argmax to "remove two host syncs". Verified on this
box (torch 2.12.1, RTX 5090) with `set_sync_debug_mode('warn')` and
three positive controls: multinomial does NOT sync on CUDA, and the
replacement is 20% SLOWER (94.2 vs 78.0 us/call). The claim came from a
CPU profile, where the validation path differs.

STILL OPEN (not applied before launch):
- `split_rollout_groups` runs on the main thread with nothing queued
  (`train_latent_vapo.py:4805-4807`, 4x/pool): 16 groups x 14 tensors
  cloned on CPU, measured 0.10-0.18 s/chunk = 0.4-0.75 s/pool of TRUE
  GPU idle. Fix: submit the split->retain_group chain to the existing
  scoring pool instead of only retain_group. Costs ~2.4 GiB host mem.
- Per-chunk 2.36 GiB pinned D2H behind a full-stream barrier
  (`latent_rollout.py:127-142`), ~0.2-0.3 s/pool, not covered by any
  existing timer. Needs a side stream; high risk.
- Checkpoint save on the main thread, measured 0.63 s every 32 steps
  (0.4% amortized).

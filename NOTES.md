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

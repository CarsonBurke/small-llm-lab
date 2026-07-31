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
Superseded by user decision 2026-07-25: `--thought-log-sigma-init` is
-2.5 (sigma 0.082, expected 512-D noise norm 1.86 = ~69% of a 2.7
thought-mean norm), and the policy adapter now shares the critic's exact
identity-weight, zero-bias initialization. Job 468
(`rl_latent_identity_sigma25_2k`) is the 2,000-step validation arm,
matched to the canceled v25 command. Job 469
(`rl_latent_identity_sigma30_2k`, after job 468) holds sigma at -3.0:
job 469 versus job 448's first 2,000 steps isolates identity versus zero
adapter initialization, while job 468 versus job 469 isolates sigma
-2.5 versus -3.0. Do not scale either to 20,000 steps unless reward/bench,
forced-vs-unforced thought payoff, termination, and BPB improve together.
Sigma remains unconstrained by the projection
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

Superseded again on 2026-07-25, pending separate 2,000-step ablations:

- Job 470 (`rl_latent_identity_sigma20_2k`) tested central log-sigma `-2.0`,
  per the user call. It was canceled by request at step 1556,
  so it is not a completed ablation. The user judged it clearly better than
  `-2.5`; at step 1500 the recorded evidence was still mixed: it had worse BPB
  (1.25769 versus 1.25504) and bench accuracy (0.138 versus 0.216) than
  job 468 at the same step. It kept more optional thinking but did not
  improve the late rollout reward mean.
- A line-by-line CleanRL SAC review rejected `-1.5` as a transferred default.
  It is only the raw-zero midpoint of CleanRL's `[-5, 2]` safety map, not an
  explicit log-std initialization. Our selected default returns to `-2.0`
  (std 0.135; expected 512-D noise norm 3.06). Its inverse raw bias is
  `-0.14384`, retains 98% of the midpoint's local tanh sensitivity, and has
  no decay toward zero because the RL AdamW weight decay is zero.
- The sigma treatment uses a unit-orthogonal state map behind a learned
  outer gain of `0.01`. This equals an orthogonal gain-0.01 map at
  initialization, while the explicit gain keeps Adam's matrix updates
  behind the same scale. The old zero-weight, state-independent head remains
  the control.
- The actor treatment is one unit-orthogonal full-width affine followed by
  `2*SiLU`; the v25 identity affine remains the control. The critic treatment
  keeps one affine and changes only its fresh initialization from identity to
  unit-orthogonal.
- Run the sigma-state, actor-adapter, and critic-adapter arms separately.
  The critic arm needs a matched critic warmup. Combine only winners. No new
  run was queued while this change was prepared.

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

RETRACTED, do not cite the table above. `--steps 6` truncates pool 2 to
512 trajectories and 2 updates, so every row is a half pool compared
against a full one. Re-measured at `--steps 8` below.

RETRACTED likewise: "52% of decode steps run at the 16-row tail". That
number had no source. `ended_fraction` measures 0.735-0.82 across every
run, i.e. 18-27% of rows are still alive at the 1024-token cap, and the
16-row tail graph needs >=96.9% ended before it engages. It never fires
at the production shape. Measured directly: `cs_notail2` (tail graph
OFF) gives refresh 2.38 s and pool 19.27 s against a control at 2.35 s
and 19.32 s. **`--rollout-tail-graph` is NEUTRAL** and costs ~1.2 s of
compile plus 0.8 GiB; defaulting it off is free. An earlier -19% claim
for it came from the truncated-pool table and was wrong.

CORRECTED perf pass (steps 8, steady pool). Raw generation seconds
confound trajectory length, which drifts 554-591 actions/traj run to
run, so the comparable column is normalized decode cost,
ms per row-step = 1000 * gen / (util * steps * 2):

| run                        | gen  | util  | ms/row-step | pool  |
|----------------------------|------|-------|-------------|-------|
| cs_F_ctrl (baseline)       | 8.29 | 0.560 | 7.0898      | 19.32 |
| cs_notail2 (baseline rpt)  | 8.27 | 0.550 | 7.1974      | 19.27 |
| cs_E_split                 | 7.64 | 0.529 | 6.9051      | 17.94 |
| cs_H_bundle                | 7.64 | 0.566 | 6.4582      | 18.53 |
| cs_I_profoff               | 7.41 | 0.552 | 6.4363      | 17.99 |

Control-to-control noise floor is 1.5% (the two baselines above).
Cumulative landed: **-9.2% normalized generation, -6.9% pool wall**,
from splitting/retaining rollout groups on the scoring worker, the RoPE
step table, and skipping the temperature divide at T=1.0.

SUPERSEDED IN PART: the RoPE step table was later REMOVED -- it
perturbed stepped-decode numerics, and the age-0 canary could not have
caught that because the canary runs the prefill path
(`replay_head_inputs`) while the table lived in `_attention_step`. See
the 2026-07-25 section. The group split and the T=1.0 divide skip both
stand; the -9.2% figure now overstates what is shipped by whatever the
table was worth, which was never isolated on its own.

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

CORRECTION (2026-07-24, `--profile`): the last sentence had the phases
backwards. Decode is the BEST phase, not the worst. Pool 3 of
`runs/cr_prof_final`, 17.56 s wall, accounting reconciled to 0.07%:

| phase                            | wall   | power | SM util |
|----------------------------------|--------|-------|---------|
| collect / decode                 | 6.82 s | 396 W | 97%     |
| update / forward_backward        | 7.26 s | 324 W | 83%     |
| refresh_pipeline / refresh       | 2.40 s | 291 W | 79%     |
| refresh_pipeline / stat_scatter  | 0.29 s | 330 W | 10%     |
| update / optimizer_step          | 0.010 s| --    | --      |

SECOND CORRECTION -- the power/util columns above are ALIASED, do not
cite them for any phase under ~1 s. NVML refreshes every 500 ms and the
sampler polls at 250 ms: 489 duplicate sample pairs against 29
singletons across 1007 samples. Every reading describes a ~0.5 s window
ENDING at its timestamp, so a short phase draws its stats from time
spent outside itself. Consequences:

* `stat_scatter_d2h` "330 W / 10%" is ONE sample, the byte-identical
  tuple that also opens `forward_backward`. It describes neither phase.
  Discard the row.
* `refresh` "291 W / 79%" is 2.40 s cut into four ~0.6 s windows, every
  sample straddling a boundary. Lag-trimmed it reads 81.2% util and
  **355 W power-while-busy -- statistically identical to
  forward_backward's 353 W.** Refresh is NOT a distinct low-power phase;
  it is the same replay kernels without a backward.
* `forward_backward` "83%" is real but biased low; lag-trimming
  converges to **88.6-89.6%**. The ~5.6 lost points are the
  inter-minibatch boundary leaking into the phase head.
* `device_seconds ~= wall_seconds` (99.4%) carries NO information -- it
  is a CUDA-event pair around the phase, device-timeline wall, not
  kernel-busy time.

So there is NO low-power replay phase to fix. Total idle inside
`forward_backward` is at most 0.83 s of 7.26 s = 4.7% of pool wall, and
that is the ceiling on any "the GPU drains" optimization.

The decisive datum, using the CORRECTED per-pool shard count (185
update + 185 refresh; `artifact_calls` was cumulative in that run, so
1586 was 4x the truth):

| phase                     | wall    | shards | ms/shard |
|---------------------------|---------|--------|----------|
| update / forward_backward | 7.263 s | 185    | 39.26    |
| refresh_pipeline / refresh| 2.400 s | 185    | 12.97    |

Ratio **3.03**, the textbook forward:forward+backward ratio to 1%. Both
sit at ~30 TFLOP/s ~= 14% MFU. Refresh runs the same compiled artifact
with no backward, no surrogate, no telemetry and ~6 boolean-mask indexes
per shard against update's ~15 -- so if mask syncs were dragging update
down, its cost per FLOP would be worse. It is identical.

`optimizer_step` at 0.010 s settles the Polar Express question anyway:
speed-neutral at this scale, its value is step geometry, not wall clock.

500 W IS PHYSICALLY OUT OF REACH ON THIS WORKLOAD, and this supersedes
the "shrink decode's share" advice above. Peak power over the entire
251 s trace is **420.0 W**; exactly 2 samples of 1007 exceed 420 W and
ZERO exceed 450 W. Board limit is 575 W (default, unmodified), clocks
stay pinned at 2820-2857 MHz, and the card is neither power- nor
thermally throttled. It is not being asked to switch enough transistors.

The intuition inverts: power tracks DRAM/L2 traffic, not tensor cores.
Decode at AI ~= 1.0 against a ridge of 117 -- as memory-bound as this
workload gets -- is the HIGHEST-power phase at 407 W while busy. The
replay update, ~50x more arithmetically intense, runs at 353 W while
busy. Removing every nanosecond of idle from `forward_backward` moves it
from 324 W to 353 W, and 353 is the arithmetic ceiling. Reaching 500 W
needs a wider model, not better scheduling.

WATTS IS THE WRONG OBJECTIVE and actively misleads: the highest-power
phase is decode, so maximizing watts rewards moving work INTO the phase
that is ~47% wasted on dead rows. Track wall time per unit of learning
-- pool seconds per gradient step, or seconds per unit of BPB/reward
movement. For a non-wall-clock proxy use MFU (update path: 14%) or
achieved GB/s.

RANKED, where the update path's 7.26 s actually goes (>=6.43 s busy,
<=0.83 s idle):

1. 50257-wide readout + its log_softmax chain, **2.0-3.0 s (28-41%),
   UNMEASURED**. ~182k emit slots/pass. The GEMM (M~182k, K=512,
   N=50257) is compute-bound at AI ~400 and is 53% of all FLOPs, but the
   elementwise chain is pure bandwidth: bf16 logits plus an
   autograd-retained fp32 log-softmax (NOTES:181 already records
   "~6 B/elem" for this tensor), written, re-read, gathered, traversed
   again in backward. Order 0.8-1.5 TB/pool.
2. Skinny trunk GEMMs, 0.8-1.5 s. d=512 means K=512 in every
   projection; cuBLAS bf16 gets 40-60% of peak on those shapes and
   backward doubles the count. Structural to a 6x512 model.
3. Eager loss-body elementwise + autograd, ~0.16 s.
4. Host syncs from boolean masking. Forward side 0.02-0.15 s as priced
   below; BACKWARD side was never counted and is where the cost sits.
   Fixed, -5.58% on this phase. See the correction below.
5. Allocator: EXONERATED by direct measurement -- pool 3 has
   `num_alloc_retries 0, num_ooms 0, num_device_alloc 1,
   num_device_free 0, num_sync_all_streams 0`.
6. Varying (B,L) shard sizes: negligible. Shards average ~4000 slots;
   M is not the problem, K is.

THE BOOLEAN-MASK SYNC FIX IS NOT WORTH FUNDING -- **WRONG ON SCOPE, and
the fix shipped at -5.58%.** The arithmetic below is correct for every
sync it counted, and I am leaving it intact because the reasoning is
sound and reusable. It counted the wrong half of the graph.

The original claim: there are 15 boolean-index sites per shard on the
shipped path, but only the FIRST drains a deep queue and that drain is
unavoidable (the forward's result is the loss body's input). Syncs 2-15
hit a near-empty queue at ~10-20 us: 15 x 20 us x 185 shards =
**0.055 s/pool**, 0.14 s pessimistic -- 0.3-0.8% of pool wall, the same
order as the 1.1% that already refuted the decode-sync claim.
NOTES:998 priced and rejected this once already.

Every number in that paragraph held up. Measured forward syncs are
<=33 us, at the fast end of the 10-20 us estimate, and killing them all
moved `refresh` -- which is forward-only -- by 0.15%, total CI overlap.
The forward audit was right and the refresh arm proves it.

What it never looked at: **`autograd/graph.py:882`, 1074 syncs per
pool, >=0.39 ms each -- ~12x a forward sync.** Boolean-mask indexing
lowers to `masked_select`, and `masked_select` syncs AGAIN in backward,
where it stalls the autograd engine at maximum queue depth instead of a
near-empty one. Total syncs per pool 7152 -> 1079. The
`update/forward_backward` phase, normalized by `decode` in the same
pool, went **1.0086 -> 0.9523, -5.58%, non-overlapping CIs at n=4.**

Two methodology lessons worth more than the fix:

* A phase audit that walks forward call sites is not a sync audit. Ask
  what each data-dependent op costs in BACKWARD before pricing it.
* Raw phase wall cannot resolve 5% at n=4 here: `decode` alone varies
  **8.1% within a single control arm**. Normalize the phase under test
  by another phase in the same pool. This does NOT contradict the 1.5%
  floor at NOTES:1049 -- that one is already a normalized metric
  (ms per row-step). The two together are the point: raw phase wall is
  ~8%, the same quantity normalized is ~1.5%, so never A/B on raw wall.

The bit-exactness risk was real and was discharged, not dodged:
`masked_fill(~mask, 0)` replaces boolean indexing where a sum follows,
which also keeps a non-finite PAD slot out of the total (multiplying by
the mask would not: NaN x 0 = NaN). Refresh and update stay on one
kernel, and the behaviour-age-0 canary reads exactly 0.0 in both arms.

WHERE THE EFFORT SHOULD GO, ranked by measured payoff:

1. **CONTINUOUS BATCHING / ROW REFILL -- 3.21 s of 17.56 s (18.3% of
   pool wall). Never attempted; 4-60x the sync fix.**
   `decode_step_utilization` is 0.5295 on pool 3 (0.52-0.59 across all
   four pools -- direct telemetry, not the bench-percentile proxy
   NOTES:962 warns about). `ended_fraction` 0.806, and
   `decode_steps_per_chunk` mean 1044.5 == max 1045: the lockstep
   signature. Since NOTES:716 establishes per-step decode cost is FLAT
   in batch width, compaction alone saves NOTHING -- refill does,
   because each of the ~1044 steps then yields 512 useful actions
   instead of 271, collecting the pool in ~0.53x the steps. The enabling
   primitive is already specced: the per-row `position` change at
   NOTES:895-926 (RoPE reshape, KV `scatter_`, mask `arange <=
   position`). Costs an execution-schema bump because RNG-to-row
   attribution moves; v19->v20 is the precedent.

2. **Measure the readout/log_softmax chain BEFORE writing code for it.**
   One pool with `--profile-trace` fills `top_kernels` (the path exists;
   `kernels: {}` above only because `profile_trace` was false). If
   log_softmax + its backward + gather + the readout GEMMs exceed
   ~1.5 s/pool, a fused CE that never materializes the fp32 log-softmax
   is worth 1-2 s/pool with NO schema bump and NO objective change --
   and `compact_emit_token_logprobs` already exists to keep refresh and
   update on one kernel, so bit-exactness is preserved by construction.
   If it comes back under 0.5 s, drop it. Cheapest decisive measurement
   on the list.

3. **Default `--rollout-tail-graph` off.** Confirmed dead by direct
   count: pool 3 ran **32 `rollout_tail_step` calls against 2080
   `generation_step` calls -- 1.52% of decode steps**. It needs >=96.9%
   ended to engage and `ended_fraction` is 0.735-0.82. The ~1.2 s
   compile and 0.8 GiB are pure cost.

4. The sync fix, as hygiene only. 0.02-0.15 s/pool.

UNSETTLED without hardware: the split between the log_softmax chain and
the skinny trunk GEMMs inside the 6.43 s busy time. One profiled pool
with `--profile-trace` decides it -- if `_log_softmax` +
`_log_softmax_backward_data` + `gather` + readout GEMMs exceed ~4 s of
device time, item 2 outranks item 1; under ~2 s, item 1 stands alone.

COMPILE CONVERGENCE (same run, answering "compile once per training
run"): pool 0 = 12.9 s compiling in 8 records, plus 4.7 s of Triton
autotune / cudagraph recording in 4 more. Pools 1, 2, 3 = 0 records.
`compilations_outside_pools` 0, `dead_artifacts` none.

Three writers append to the Dynamo compilation-metrics stream and only
one is a frame compile: forward (`convert_frame.py`, the only writer
that fills `co_name`), lazy backward (`runtime_wrappers.py`, no code
object, `is_forward=False`), and RUNTIME (`_dynamo/utils.py`, no code
object, `is_forward = not is_backward`), which bills Triton autotuning
or a cudagraph re-record. Classifying on `co_name is None` alone
conflates the last two and inflated the pool-0 headline by 27% -- an
earlier note here said "12 compilations / 17.57 s". Records are now
classified by `(is_runtime, is_forward)` and the two are reported
separately.

"Zero records after pool 0" is also the WRONG convergence test: a
`reduce-overhead` artifact re-records its cudagraph whenever the pool it
captured against is invalidated, which can happen in a perfectly
converged run. The test is zero FORWARD and BACKWARD records per pool,
with runtime rows reported separately.

"Compile exactly once" is NOT reachable, by construction: two
`torch.compile` wrappers on one code object never share a cache entry
(`_TorchCompileInductorWrapper.__eq__` compares `(config, dynamic,
name)`), so `step_core` under max-autotune/dynamic and under
reduce-overhead/static are two artifacts. Floor is ~8-10 records:
those two, `replay_head_inputs` and `value_logits` with their backwards,
and three `_polar_express` shapes that `dynamic=False` in `muon.py`
makes unavoidable (0.13 s total). Target: all of them before pool 0
ends, zero forward/backward records in every pool after.

UNVERIFIED at length: 4 pools cannot show whether new length buckets
appear at step 5000+.

DUCK SHAPES -- the rollout batch dim was never dynamic. `dynamic=True`
only sets `assume_static_by_default=False`; unmarked dims still get DUCK
sizing, so every input dim sharing a value on the first trace gets ONE
symbol. The model is 512 wide and `--rollout-groups 32 x
--samples-per-prompt 16` = 512, so the batch dim was unified with
`model_dim`, the first `rms_norm` against a 512-wide parameter emitted
`Eq(s, 512)`, and the first compaction recompiled. Eval carries the same
hazard at 128 rows against `head_dim` 128. Switchable with
`torch.fx.experimental._config.use_duck_shape = False` via
`--duck-shape`.

KEEP DUCK SHAPING ON. A/B at 16 steps, no instrument, steady-state pool
3: duck-on 17.68 s vs duck-off 18.49 s, i.e. duck-off is 4.6% SLOWER,
above the 1.5% noise floor. Duck-off buys one 6.41 s recompile back and
then loses 0.81 s every pool -- 67 minutes over 5000 pools. The reason
is that duck-on does not stay specialized: automatic dynamic makes the
batch dim symbolic on the SECOND shape, so duck-on converges to the same
dynamic batch dim after paying once, while keeping every other dim
duck-specialized. Duck-off de-specializes dims that never needed it.
n=1 per arm; repeat before betting anything large on 4.6%.

The eval hazard at 128 rows vs `head_dim` 128 is real but bounded: one
extra recompile the first time eval generation runs, then automatic
dynamic takes over. Startup cost, not per-pool, so it does not bite at
any step count.

`accumulated_recompile_limit` is NOT a long-run hazard -- an earlier
note here claimed it was and raised it to 512. Retracted. Two counters
gate `exceeds_recompile_limit`: `compute_cache_size` walks the LIVE
entry list, so an invalidated cudagraph entry LEAVES the list and its
churn does not accumulate; the `frame_compile_id` backstop is
historical but increments once per compile of that frame. Both are
bounded by distinct shape/guard classes, not by step count. Measured
maximum across every frame: **3**, against a default of 256. Reverted
to the default.

CONVERGENCE, measured both arms, 16 steps / 4 pools: every compilation
in the run happens in pool 0 (8 compiles under duck-on, 7 under
duck-off), and pools 1, 2, 3 have zero forward, zero backward AND zero
runtime records. This held BEFORE any of the recompile work, so it is a
property of the code, not of a fix. Of the 8: `step_core` 0/0 first
trace (6.88 s), `step_core` 0/1 the duck-sizing recompile (6.41 s,
avoidable), `step_core` 0/2 the `--rollout-tail-graph` artifact
(0.38 s), `replay_head_inputs` (4.66 s), `value_logits` (4.25 s), and
three `_polar_express` grad shapes (24,512,512) / (6,2048,512) /
(6,512,2048) at 0.13 s total.

BEHAVIOUR-AGE-0 EXACTNESS holds as a measured invariant, not an
argument: `ratio/joint_abs_log_max` and `ratio/thought_joint_abs_log_max`
are exactly 0.000e+00 at every age-0 step (1, 5, 9, 13) in all three
runs including the duck-off arm, which was the one most likely to split
refresh and update onto different artifacts.

OPEN, and the biggest risk to a 20k run: one control run showed
`refresh 18.219 s` against a 2.1-2.7 s steady state, intermittent and
not reproduced in either profiled run.

Ruled out: the duck-shape recompile, which fires at the first compaction
inside pool 1 and costs 6.4 s, not 16. Allocator retry is also out --
`num_alloc_retries` and `num_ooms` are 0 on every profiled pool.

STILL UNEXPLAINED. A one-row-shard attribution was proposed and then
RETRACTED -- see the retraction below the evidence.

The observation (jobs 421 pool 2 and 422 pool 3, both after pool 0, in
refresh) is a LATE Dynamo recompile from torch's 0/1 specialization:

```
replay_head_inputs:1137  reason=1/0: 2 <= batch.token_ids.size()[0]
value_logits:82          reason=2/0: 2 <= batch.token_ids.size()[0]
```

The first trace assumed row count >= 2, so the guard can only fail on a
shard of 0 or 1 rows. Replay shards always hold >= 1 row, so it is
exactly a **one-row replay shard**. `plan_replay_shards`
(`latent_rollout.py:1279-1314`) packs greedily over rows sorted by
descending length and yields the remainder, so a trailing one-row shard
is DATA-dependent -- some pools, not others. That is the intermittency.
Reproduces under both flags tested: 421 is duck-ON/tail-off, 422 is
duck-OFF/tail-on. Absent from 414, 417, 418.

Warm cost 1.59 s (421) and 1.38 s (422), both with an FX cache hit. Cold
is the story: the `step_core` NoneType variant cost 6.39 s cold with
`inductor 3.85` against 0.38 s warm, a 17x swing on ONE frame. Two
frames plus autotune at that ratio is the right order for 16 s, and
`cr_prof_ctrl` ran before anything had ever compiled the one-row
variant. STRONGLY INDICATED, not proven -- the cold number for these two
frames is not yet measured directly.

RETRACTION -- this does NOT explain the 18.2 s stall, and the
specialization was already known. It is documented in-file at
`train_latent_vapo.py:5849-5857`, naming the same guard, the same
planner behaviour, and recording that `mark_unbacked` was already tried
and rejected (Inductor's constant folder raises
`GuardOnDataDependentSymNode` on the row dim). A prior measurement at
`:5756-5758` puts the same event at **7.2 s -> 1.7 s** after the
`unsafe_marked_cacheable_functions` autocast fix.

So the 1.59 s (421) and 1.38 s (422) figures CONFIRM the documented
1.7 s rather than revealing anything. The pre-fix cost was 7.2 s with
the FX cache already hitting, so even a genuinely cold variant lands
near 7-8 s, not 16. And `cr_prof_ctrl`'s pool-0 refresh of 10.82 s is
itself a post-autocast-fix number, so that run already had the fix in.
The story was fitted to a magnitude that was never measured.

WHAT SURVIVES: the specialization fires LATE -- pool 2 in 421, pool 3 in
422 -- so it is the only known compile after pool 0.

SUPERSEDED: "and it costs ~1.5 s per process, not 16." That 1.5 s is a
cache-HIT cost. The BUILD cost, measured in job 445, is
`replay_head_inputs` 6.52 s plus `value_logits` 10.53 s = 17.05 s. So
the magnitude does fit the stall after all; see the 2026-07-25 section.

Planner names, twice corrected and now checked against the file: the
plan is built by `plan_length_aware_shards`
(`latent_rollout.py:1330`) and `iter_length_aware_microbatches`
(`:1288`) is only the iterator over it. The split is new -- the file was
refactored mid-session by the host-sync work, which is why earlier
entries name only the iterator. "plan_replay_shards" never existed.

CONTRADICTING THE IN-FILE COMMENT -- verified on a toy model, NOT
against the production guard: a red-team pass
reproduced both frames on CPU and found `mark_unbacked` on the row dim
DOES work for `value_logits`, and for `replay_head_inputs` too once the
`torch.zeros_like` plus slice-assign at `latent_rollout.py:1152-1153`
becomes `F.pad(batch.token_ids[:, 1:], (0, 1))`. The blocker is
Inductor's `constant_fold_uniform_value` on that specific `zeros_like`,
not the row dim as such. That would delete the specialization outright.

The probe was then re-run against the repo's real `NanoGPTBackbone`,
`LatentThoughtModel` and `select_trajectory_rows`, forward and
backward: baseline recompiles at one row; `mark_unbacked` alone raises
`GuardOnDataDependentSymNode` exactly as the in-file comment says; and
`mark_unbacked` plus the `F.pad` rewrite serves both row counts with
one graph. So the comment's DIAGNOSIS is right and its CONCLUSION is
wrong -- the blocker is that foldable `zeros_like`, not the row dim.

What remains unverified is that this removes PRODUCTION's guard. The
toy raised a different one (`128*rows*stream > 4096`, from Inductor's
codecache) so it reproduces the recompile but not the exact guard, and
the production graph may hold other foldable uniform-value nodes. The
honest label is "verified on a toy model; not verified against the
production guard".

NOT LANDED, deliberately. It is worth a one-off compile, and against a
multi-hour run that is noise. It edits the one path refresh and update
share and the one the age-0 canary rides. Wrong risk-to-payoff ratio
until something else justifies touching that path.

TWO LIVE LEADS for the actual stall, in rated order:

1. **Allocator retry.** `num_alloc_retries` empties the caching
   allocator and synchronizes every stream -- a device-wide stall that
   appears in no phase's own time. Sampled per pool at
   `train_latent_vapo.py:3840-3860` and read ZERO in every profiled
   pool -- but no profiled run has reproduced the stall either, so this
   is untested, not excluded. A long shard from an unusually long
   trajectory is exactly the data-dependent trigger that fits.
2. **A device clock event.** Job 422 pool 2 shows a collect window at
   **195 MHz and 35 W for 0.8 s** against 2820 MHz / 340 W steady state
   -- the GPU dropping to idle clocks mid-phase. Nothing rules out a
   longer instance, and it would never reproduce under a profiler.

BEST INSTRUMENT: the 20k run itself. It is ~5000 pools against the ~20
that have been profiled, and `pool_refresh_seconds` lands in
`metrics.jsonl` every pool. A 40-step smoke gives 10 pools and is
underpowered for an event seen once in twenty.

Superseded cold-compile hypothesis, kept because the data behind it
stands:
the coldest-cache run of the group, and the measured cold-vs-warm
refresh delta on pool 1 is 23.50 s vs 11.65 s, the right order for a
16 s excursion. Refresh by pool, showing warmth is the whole pool-1
story and duck shaping changes nothing in steady state:

| run                        | pool 1 | pool 2 | pool 3 | pool 4 |
|----------------------------|--------|--------|--------|--------|
| cr_prof_final (on, warm)   | 10.99  | 2.59   | 2.24   | 2.31   |
| cr_duck_on 417 (on, warm)  | 11.65  | 2.65   | 2.32   | --     |
| cr_duck_off 416 (off, COLD)| 23.50  | 2.68   | 2.35   | 2.43   |

The duck-off row is NOT a comparable arm: turning duck shaping off
changes the traced graphs, so every FX and AOT cache key is new and
pool 1 pays a full cold compile. Test in flight: a profiled 10-pool run
with `TORCHINDUCTOR_CACHE_DIR` in a scratch dir (cold cache without
disturbing the shared one), reading which pool compiles and why. If
nothing after pool 1 compiles even cold, compilation is not the cause.

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


## bf16 cache dtype fix + RoPE step table, verified on GPU (2026-07-25)

THE BUG (real, reproduced on hardware for both `NanoGPTBackbone` and
`NanoTiedDotBackbone`): under `autocast("cuda", bf16)` with bf16 caches,
`F.rms_norm` returns **fp32** (it is on `AT_FORALL_FP32`) while `v`,
never normed, stays **bf16**. Both `index_copy_` into same-dtype caches,
so eager raises

    RuntimeError: index_copy_(): self and source expected to have the
    same dtype, but got (self) BFloat16 and (source) Float

Compile hides it; eager does not. Live on `--no-rollout-compile`, any
Dynamo bail-out, `sample_latent.py`, `inspect_critic_values.py`.

Casting only the `index_copy_` writes is a HALF FIX: `q` is fp32 too and
never reconciled, so eager SDPA then raises `Expected query, key and
value to have the same dtype`. It is invisible under CUDA autocast only
because SDPA is on the lower-precision list and silently down-casts `q`.
The fix casts all three. `.copy_()` slice-stores never raised because
`copy_` converts silently -- only `index_copy_` is strict.

`rms_norm` has NO AutocastCPU kernel (fallthrough), so CPU
reproductions of this are misleading. Verify on CUDA.

ATTRIBUTION (job 441, `mode="default"` so Inductor kernel choice is
deterministic; 128 steps x batch 16, bf16 caches, real CUDA autocast,
each variant a fresh process):

    CONTROL baseline vs itself : bit-identical   <- noise floor zero
    CONTROL fixed vs itself    : bit-identical
    dtype casts alone (q,k,v)  : DIFFERS
    RoPE table alone           : BIT-IDENTICAL
    both together              : DIFFERS
    k+v casts, no q cast       : DIFFERS
    k cast only                : DIFFERS

The two self-comparison controls are load-bearing -- an earlier bisect
WITHOUT them reported "RoPE table alone: DIFFERS" and was wrong,
because autotuner variance was being read as attribution.

CORRECTION -- "the RoPE step table is bit-identical" is TRUE ONLY IN
THAT ARTIFACT. Job 435 also ran the TAIL artifact (`reduce-overhead`,
`dynamic=False` -- the CUDA-graph decode tail training actually uses):

    tail CONTROL baseline vs itself : bit-identical  <- regime is
    tail CONTROL fixed vs itself    : bit-identical     deterministic
    tail dtype casts alone          : DIFFERS
    tail RoPE table alone           : DIFFERS   <- 30 elements, 6 layers

Both controls clean, so it is not noise. **The table IS implicated.**
It was DROPPED -- a perf change with no compiled measurement behind it
does not get to move bytes. The dtype cast was kept.

How provably-identical table CONTENTS still move bits: the eager proof
was of the table's values. Under compile the old path computes
`cos`/`sin` inline in the fused kernel via Triton's libdevice, and the
table path replaces that with a memory load -- different fusion,
different FMA contraction, possibly a different sin/cos implementation
than eager ATen. Values equal, arithmetic re-associated. **An eager
value-equality proof does NOT imply compiled bit-equality.**

**The dtype cast owns the remaining drift and it is unavoidable** --
the k-only cast is the smallest change that makes `index_copy_` legal
and already shows the full effect.

WHAT DIFFERS: `logits`, greedy tokens, sampled tokens and cache V are
IDENTICAL in every differing pair over all 128 steps. Only cache K
moves: 26 elements across ~6.3M written slots (~4e-6 density), every
max|d| a power of two equal to exactly 1 bf16 ulp at that magnitude --
values landing the other side of a rounding boundary, not error
accumulation. Cause: an explicit `.to(bf16)` is a separate rounding
point, so Inductor contracts the producing FMAs differently.

NONDETERMINISM, and this is the part to remember: under
`max-autotune-no-cudagraphs` (the ACTUAL rollout compile mode) the
fixed variant differs run-to-run at the same 1-ulp magnitude between two
identical processes, while the baseline does not. The casts give the
autotuner a second viable schedule. **Cross-process bit-reproducibility
is therefore not achievable in the rollout artifact.** Do not go hunting
for a regression when you see this.

Autotune selection happens ONCE PER PROCESS at compile time, so it
cannot vary within a run. It affects reproducibility across restarts and
resumes, not intra-run consistency. (n=1 pair per cell; a replication at
n=4 against casts-only was queued, since the table was a plausible cause
of the autotune tie.)

THE CANARY DOES NOT COVER THIS, and an earlier note here claimed it did.
`refresh_old_statistics` (`latent_rollout.py:1411`) and
`update_minibatch` both compute through `replay_head_inputs`, the
parallel replay/prefill path. The dtype change is in `_attention_step`,
which runs ONLY in stepped decode (`step_core`); the prefill path
already cast to the cache dtype and was untouched. So the 0.000e+00
age-0 result is real and reassuring about refresh/update agreement, but
it is STRUCTURALLY INSENSITIVE to a decode-path numerics change and is
not evidence either way about this one.

The honest argument is the simpler one: decode drift changes WHICH
TOKENS get sampled into the replay. That is data, not the ratio
identity -- a different draw from the same distribution, not a
correctness failure. The crash it prevents is a correctness failure.

DECISION: took the fix. A 1-ulp perturbation that moved zero tokens in
2048 sampled steps beats shipping a latent hard error into a 20k-step
run. "Bit-exact vs HEAD" was the WRONG launch criterion in the first
place -- the age-0 canary is the criterion, and it holds.

RESOLVED, and it turned up something bigger. `eval_open_loop.py:192`
carried "the stepwise cache path does not support autocast: fp32 caches
reject bf16 values". Both halves of that comment are false:

* The cache absorbs bf16 now, via the q/k/v casts.
* **Open-loop eval runs fp32 while rollout runs CUDA bf16 autocast**
  (`training_autocast()`, train_latent_vapo.py:6419). The comment
  claimed the fp32 choice MATCHED the rollout regime. It never did.

So every open-loop number on record measures a precision the trainer
does not use. Left fp32 on purpose: switching it moves the whole
historical series, which is a deliberate call rather than a cleanup.
The comment now says so instead of asserting the opposite.

## Muon step compensation + batching (2026-07-25)

`POLAR_EXPRESS_STEP_COMPENSATION` **2.4 -> 1.45**, from job 442: 624
non-degenerate real gradients out of live post-training, every Muon
parameter over 8 steps of both optimizers. Ratio of new to old update
norm, by shape:

| shape       | n   | ratio |                                  |
|-------------|-----|-------|----------------------------------|
| (512, 512)  | 414 | 0.823 | no rectangular term              |
| (512, 2048) | 108 | 0.570 | no rectangular term              |
| (2048, 512) | 102 | 0.494 | lost a 2.0x rectangular scale    |

Geometric mean 0.690, so compensation is 1/0.690 = 1.45. At the
previously chosen 2.4 the 414 (512,512) matrices -- the bulk of the
trunk -- step **1.97x** harder than tuned. Stable ranks measured
1.0-2.6. This overrides an earlier user-directed 2e-4; at
`--learning-rate 5e-5` the derived Muon LR is now 1.2083e-4.

BATCHING IS NOT BIT-EXACT, and the test that asserted it was is fixed
rather than the kernel. cuBLAS selects its strided-batched GEMM by
batch count, so a 3-row `bmm` and a 1-row `bmm` accumulate in different
orders; five iterations of a cubic amplify that bf16 rounding to a few
percent on the small elements of a near-orthogonal matrix. This is not
a defect: `bmm` cannot mix batch entries, Muon's output is not on the
age-0 canary path, and shape buckets are fixed within a run so runs
stay reproducible. `test_cuda_kernels_match_the_per_tensor_reference`
now pins the property that actually matters -- each batched row tracks
its own unbatched result >20x more closely than a sibling's -- which
still catches a real row-mixing bug while tolerating rounding.

## v25 20k launch: job 448 (2026-07-25)

Config as v24 plus the Muon compensation at 1.45, `--duck-shape` off,
`--rollout-tail-graph` off, `--replay-max-trajectories 128`,
`--replay-attention-budget 16777216`, `--rollout-groups 32`,
`--max-stream-steps 2048`. Step 0 baselines: teacher-forced val_bpb
1.2415, aime_avg@32 0.0000, bench_avg@8 0.1189.

The trainer records NO provenance -- no git hash, no source snapshot.
With a working tree this far from HEAD that makes a run
unreconstructable, so 448's diff, HEAD and untracked tests are copied
into `runs/rl_gpt2vocab_latent_v25/provenance/`. Worth making the
trainer do this itself.

STEADY STATE, first 8 pools:

| metric                    | pool 0 | pools 1-7          |
|---------------------------|--------|--------------------|
| `pool_refresh_seconds`    | 6.71   | 2.01-2.77          |
| `collect_seconds`         | 15.68  | 9.95-11.33         |
| `decode_step_utilization` | 0.578  | 0.535-0.583        |

Pool 0 is compile-inflated and not comparable. Do NOT read the ~2.1 s
refresh against the 2.40 s at NOTES:1115 as an improvement: that
baseline ran a different shard count and budget. Different measurement,
not a better one.

500 W IS CONFIRMED UNREACHABLE, second independent measurement, this
time on the shipped config and sampled at 1 Hz against NVML's measured
500 ms refresh so no reading is a duplicate.

An early n=150 window gave median 335 / mean 320 W. SUPERSEDED: that
window opened during startup and understated steady state by ~13%. Over
n=764 (12.7 min of healthy pools): min 158, p25 330, **median 381**,
p75 400, p95 420, **max 429**, mean 361 W. Zero samples >=450 W, zero
>=500 W. Median util 97%, SM clock 2797-2842 MHz with ZERO samples
below 1000 MHz, board limit 575 W, not throttled. Agrees with the
1007-sample trace (peak 420 W, zero above 450).

The lesson is the same one that produced the aliasing retraction: a
short window is not a small version of a long one when the run has
phases. Take the baseline over pool interiors, not from process start.

Steady state at ~361 W mean is where this workload already sat before
any of this session's work. Power did not move, and was never the thing
to move.

The target should be retired, not pursued. Power tracks DRAM/L2
traffic, so the highest-power phase is decode -- the phase running at
0.56 utilization, i.e. the one wasting the most work. Maximizing watts
rewards moving work INTO it. Every real win so far LOWERED mean power
while cutting wall time. Track pool seconds per gradient step.

CONTINUOUS BATCHING is now measured across 8 independent pools rather
than one: `decode_step_utilization` 0.535-0.583. Still the largest
unclaimed item on the list, still never attempted, and the 8-pool
spread makes it a stable target rather than a single-pool artifact.

### Continuous refill implementation (2026-07-26, OPT-IN / REJECTED FOR 5090 TRAINING)

`--rollout-scheduler continuous_refill` now implements the systems path
behind that measurement: fixed physical decode lanes, per-iteration
whole-group refill, per-request positions, request-stable Philox streams, and
block-sparse paged FlexAttention over only each row's live KV pages. Every
unique prompt in the pool is prefetched in one dense prefix-bank pass; refill
only scatters cached KV/state into its sample lanes. FlexDecoding retains the
first cache inputs in its compiled artifact, so the trainer owns and reuses
one finite-initialized paged arena across pools. Masked partial pages must
never contain NaN garbage; stale suffixes remain unreachable. No trajectory
is shortened; the existing token/context limits remain safety semantics.

Production-shape CUDA results:

* g32 rollout-only, repeats 2-7: 5.396 s mean collection vs 8.329 s
  lockstep, **35.2% lower wall** and **27.4% higher useful-action/s**.
  Decode utilization rose to ~0.91. However, its ~23.2 GiB persistent arena
  leaves too little memory for replay: the end-to-end run OOMed.
* g24 fits (~23.2 GiB update peak) but requires three admission waves. Its
  warm collection was 19.22 s vs 10.92 s for matched g32 lockstep, and its
  final pool took 27.12 s vs 19.88 s. Reject.
* g16 fits but similarly loses to lockstep (20.24 s final pool vs the older
  18.29 s control).

The implementation stays as a tested opt-in/reference path, but lockstep
remains the production default on a 32 GiB 5090. The scheduler has real
decode payoff only at g32; making that trainable needs a materially smaller
cache representation or freeing the compiled arena before replay, not a
smaller refill batch.

A streamed 8192-token selected-logprob readout was also tested as a memory
escape hatch. It let g16 train and reduced refresh, but checkpointed vocab
recomputation made the matched lockstep pool 19.88 s vs 18.29 s (+8.7%).
It was removed. The existing slot-budgeted dense readout remains faster.

The resulting dense lockstep v34 run is modestly faster than v33: step-8
pool wall is 17.47 s vs 18.29 s (-4.5%) while processing 0.5% more actions;
step 12 is 16.88 s vs 17.44 s (-3.2%) while processing 2.7% more actions.
That is roughly a 5-6% work-normalized throughput uplift. Job 517 is the
selected 20,000-step run; AIME and easy-benchmark evaluation cadence is 250.

This replaces the scalar tail-parking experiment. Whole-group admission may
leave at most 15 lanes unused; one completion/position transfer is the
per-iteration host boundary. Async/versioned RL and speculative decoding are
not part of this change: the former adds policy staleness on one shared GPU,
while the latter has no lossless draft/verifier construction for the joint
discrete gate/token plus continuous latent action.

### Compile floor: REACHED (job 445, ten pools)

Zero compilations and zero runtime records in pools 1-9. All seven
compiles and all three runtime records land in pool 0:

1. `step_core` first compile.
2. `step_core` second entry -- the frame is called with and without a
   `key_mask`, so it is two graphs. Removing it costs a masked SDPA on
   the common path.
3. `replay_head_inputs` first compile.
4. `value_logits` first compile.
5-7. `_polar_express`, one per Muon shape class: (24,512,512),
   (6,2048,512), (6,512,2048). 0.13 s for all three.

"Compile exactly once" was never reachable: two `torch.compile`
wrappers on one code object cannot share a cache entry.

The only post-pool-0 record ever seen is `replay_head_inputs` +
`value_logits` recompiling for a ONE-ROW replay shard. It did not fire
in ten pools, so it is data-dependent, not periodic. 448's settings do
not make it immune -- at a 2048 stream the attention budget caps a
shard at four rows -- but they do remove the solo-shard-by-length case,
so only remainders can be one row.

### The 18.2 s refresh stall: candidate found, magnitude fits

The retraction of the original attribution was itself too hasty. The
"1.7 s" at `train_latent_vapo.py:5757` is a cache-HIT cost, not a build
cost. Job 445 measured the build: `replay_head_inputs` **6.52 s** and
`value_logits` **10.53 s**, 17.05 s for the pair, and that is with the
FX graph cache still hitting. A 16-18 s excess in ONE refresh
occurrence is exactly that size.

Job 445 pool 0 reported `uneven: refresh_pipeline.refresh longest
occurrence 18.028 s of 20.559 s over 4` -- one occurrence of four. That
distinction is why the stall stayed unexplained: a phase mean cannot
tell one 18 s occurrence from four slow ones.

Open question is now narrow: can the on-disk cache miss for the one-row
variant? Job 449 answers it with a scratch `TORCHINDUCTOR_CACHE_DIR`.
448's own pool 0 spent only 6.71 s in refresh total, so no 17 s
artifact build happened there -- consistent with a warm cache.

Falsifiable prediction from this hypothesis: across 448's ~5000 pools,
ONE early outlier and none after. A recurring cadence kills it and puts
an allocator or device event back in play.

CAUTION on job 445's wall times: its steady pools are ~35% slower than
422's because CPU work was running on the box alongside it
(`score_wait` reaching 3 s is the scoring worker starved of CPU). Its
compile accounting is unaffected, but do not use its wall times as a
baseline for anyone's change. Related: the shared FX cache is
invalidated by ordinary source edits, so "warm" is not a stable
baseline while several people are editing.

### Task #3 ANSWERED: the 18.2 s refresh stall is a cold-window compile

First, the discriminator I circulated was WRONG and the correction is the
whole method. "A compile inflates only refresh; a clock drop inflates the
whole pool, so compare against `collect_seconds`" does not work, because
`collect_seconds` SPANS the refresh pipeline: `collect_started = started`
at `train_latent_vapo.py:6841`, `collect_seconds` computed at `:7014`
after the refresh loop and after `torch.cuda.synchronize()` in
`pool_barrier`. It inflates whenever refresh does and can never separate
them. The quantity that separates them is the RESIDUAL,
`collect_seconds - pool_refresh_seconds`, against its own median.

With the residual, outliers split into two non-overlapping classes.
Across all 96 historical runs, threshold refresh > max(3 s, 4x run median):

**Class A -- refresh-confined, residual at its median. Three events in the
entire recorded history**, all in the first four pools of their run:

| run                        | pool | of  | refresh | residual x |
|----------------------------|------|-----|---------|------------|
| `cr_prof_ctrl`             | 3    | 4   | 18.219  | **1.00**   |
| `..._v21_perdim_16m_...`   | 1    | 268 | 3.064   | 1.52       |
| `..._v21_perdim_16m_...`   | 2    | 268 | 15.546  | 1.51       |

v21 ran 268 pools and never produced another. Its residual is flat
through the event (83.3, 83.1, 79.6, 72.6, 80.4 s at pools 1-5), so
decode was untouched while refresh took 15.5 s.

**This is the stall.** Magnitude matches the 17.05 s first build of
`replay_head_inputs` + `value_logits` measured in job 445. Which pool it
lands in varies because the one-row shard is data-dependent. Warm hits at
~1.7 s fall under the 3 s floor and are invisible to the detector, which
is consistent with the story rather than a gap in it. Cost: 15.9 s once
per run, 0.55% of v21's total collect wall.

**Class B -- pool-wide, residual inflated too. 20 events, and most are
not anomalies at all**: at 11 of them `stream_length` is 4-6x the run
median (1008-1317 against medians of 197-240). Longer trajectories, more
work, everything slower. v23's cluster at pools 155-180 is entirely this.
The remaining ~9 are real: normal stream length, 15-25 s of ADDITIVE
excess in refresh and in the non-refresh remainder alike. Something takes
the whole pool, not one phase of it. **The 195 MHz / 35 W device-clock
lead belongs here, not with the refresh stall.** Untested: no run on
record carries device samples through one of these events. Job 448 now
does -- see `runs/rl_gpt2vocab_latent_v25/device_trace.csv`.

Aggregate for both classes: under 1.5% of collect wall in every long run
(0.55%, 0.92%, 0.94%, 1.48%). Not a throughput problem beside the 18.3%
of pool wall that dead-row decode costs.

CHECKPOINT HYPOTHESIS IS DEAD, and the way it died is worth keeping. A
40-row lookback put a checkpoint before nearly every event BY
CONSTRUCTION. Counted properly: 2 of 337 checkpoints in one run precede a
stall, and `cr_prof_ctrl` -- the original 18.2 s case -- has no
checkpoint rows at all. A lookback window wide enough to catch a frequent
event will always "explain" a rare one.

JOB 448, 32 pools: zero outliers of either class. Refresh median 1.99 s,
max 2.769 s (pool 0). Residual median 8.25 s, so decode is 4x refresh --
pointing at continuous batching again. Pool 0 refresh of only 6.7 s means
the on-disk FX cache was already warm, so class A may never fire in this
run. Observing it needs a COLD-CACHE start, not a longer run.

### The anchored-value test flake: closed, cause UNEXPLAINED (2026-07-25)

`test_anchored_value_migration_transfers_trunk_and_rebuilds_head` failed
ONCE, in the live main tree, and never again. Closed as test-only.

DO NOT record the cause as a torn read. I proposed that and it does not
survive checking: `git log -L 470,510` shows the filter last touched by
`b193e2f`, three commits back; neither the working-tree diff nor the
provenance snapshot taken at 00:01:29 -- inside the failure window -- has
a hunk overlapping those lines; and `value_model.py` was not written that
day at all. There is also a mechanical objection: `test_latent_rollout.py`
imports `migrate_anchored_value_resume` at module scope, so a truncated
file fails collection for the WHOLE file and a half-written one raises
SyntaxError. A silently-wrong filter needs the edit to land inside the
filter, and none did. "Cause unexplained" is the honest label; a
plausible story recorded as settled would misdirect the next person.

What DID cause agents to disagree about the suite, and this is
established: provenance copies of two test files were written into
`runs/<name>/provenance/` with real `.py` names, colliding by basename
with the modules in `postraining/tests/`. From 00:01:29 to 00:20:18
repo-root `pytest` ERRORED at collection and collected ZERO tests. Two
agents running "the full suite" either side of that window ran different
sets. Fixed by `pytest.ini` (`testpaths`, `norecursedirs`) so the suite
is one stable set, and `postraining/tests/__init__.py` so each test
module is imported once rather than existing twice in `sys.modules`.

THE MIGRATION FILTER IS SOUND, verified against a real `SeparateCritic`
rather than by reading. 60 state-dict keys; top-level children exactly
`trunk`, `adapter`, `support`, `head`; exactly 4 keys match the
prefixes; NO key containing "head" or "support" escapes the filter, and
`startswith` is anchored so it cannot over-match. The grid scalars
(`num_bins`, `v_min`, `v_max`, `bin_width`, `sigma`, `eps`) are plain
Python attributes on `HLGaussSupport`, NOT buffers, so they are absent
from the state dict and can never leak -- the target's fresh geometry
always wins. `strict=False` hides nothing: the checks at `:497-503` are
TIGHTER than `strict=True`, because they pin which keys may be absent.
Pathological case tested -- source and target both 17 bins, so a leaked
head would be silently copyable rather than shape-rejected -- and the
head stayed exactly 0.0.

Two real defects came out of the audit anyway; see tasks #11 and #12.
Neither affects job 448.

## v34 post-training perf: block-table fix, readout fusion, flex decode (2026-07-27)

### Control variance, measured (three replicates, lockstep g32, 44-48 steps)

Arms `bench_ctrl_1`/`bench_ctrl_2`/`ab_ctrl`. Warm pools only (first four
discarded). This is the yardstick every claim below is held to:

| metric | spread |
|---|---|
| `pool_seconds` | 12.02 / 12.85 / 14.63 (+-10%) |
| `decode_ms_per_action` | 16.10 / 16.35 / 17.72 (+-5%) |
| `refresh_ms_per_emit` | 3.584 / 3.600 / 3.604 (+-0.3%) |
| `actions_per_trajectory` | 356.9 / 391.0 / 400.4 (+-6%) |

Two lessons. Raw phase wall is worthless at this spread -- normalize per unit
of work. And the refresh phase is FAR quieter than decode, which is why the
readout result below is trustworthy at a fraction of the replicate count.
Arms must also match on `--steps`: longer runs reach longer streams, and
`bench_ctrl_1` (48 steps, 10 warm pools) is the slowest of the three for that
reason alone.

### `kv_range_blocks` emitted a mis-strided partial table (SHIPPED BUG, fixed)

`partial_indices` was 2 wide while `full_indices` was `blocks_per_row` wide.
The Triton decode kernel derives both from one descriptor -- it offsets
FULL_KV_IDX by `stride("KV_IDX")` and bounds it by `size("KV_IDX", -1)`
(`torch/_inductor/kernel/flex/templates/flex_decode.py.jinja:97,135,185`) --
so a narrower partial table makes it read the full table at the wrong row
stride. Full blocks skip `mask_mod`, so nothing downstream corrects it: every
row but row 0 silently attended the wrong keys.

On-device, compiled, against masked SDPA (job 554):

| builder | shipped | fixed | signal |
|---|---|---|---|
| `DecodeRangeMask` B16 L2560 | 0.2872 | 4.5e-07 | 0.254 |
| `PagedGenerationCache` C8 L1024 | - | 1.5e-06 | 0.662 |

The paged row is the one that matters: this affected the shipped
`continuous_refill` scheduler, not just new code. **Why every test missed it:
eager `flex_attention` builds its mask from `mask_mod` alone and never reads
the block tables** (`torch/_higher_order_ops/flex_attention.py:205-215`), so
CPU tests agree with SDPA no matter how wrong the tables are. Any future
block-table test must run under `torch.compile` on CUDA to mean anything.

### Readout tail fused into one compiled artifact: KEEP (-52% refresh)

`compact_emit_token_logprobs` now covers renderer features -> readout GEMM ->
softcap -> fp32 log-softmax -> target gather, compiled once and shared by
refresh and update. Eagerly that is ~8 passes over a (slots, 50257) fp32
tensor to produce one scalar per slot.

Two arms vs `bench_ctrl_2`, all at 48 steps:

| metric | ctrl | readout_1 | readout_2 |
|---|---|---|---|
| `refresh_ms_per_emit` | 3.584 | 1.727 (-52%) | 1.735 (-52%) |
| `update_total` | 4.442 | 2.528 (-43%) | 2.510 (-44%) |
| `pool_seconds` | 12.02 | 9.341 (-22%) | 9.772 (-19%) |
| `decode_ms_per_action` (control phase) | 16.35 | +0.8% | -0.6% |
| `peak_vram_bytes` | 2.043e10 | -3.8% | -3.8% |

Decode is flat within +-0.8%, which is what rules out drift. Age-0 zero-clip
canary is exactly 0.0 across all 12 age-0 rows in both arms, so refresh and
update stay bit-identical through the shared artifact -- that is why BOTH must
resolve the same object, and why `measure_post_update_policy_drift` takes the
eager function as a parameter instead of the rebound global (a no-grad call
would compile a second artifact, grad mode being a Dynamo guard).

Peak VRAM went DOWN 0.78 GiB, which refutes the prediction that tracing
refresh grad-enabled would put the vocabulary-wide activations on the peak.

### Flex decoding for the lockstep decode step: NEGATIVE so far, default OFF

Motivation: a boolean `attn_mask` disqualifies every fused SDPA backend and
lands the step on the memory-efficient cutlass kernel, 32% of pool device time
in the v25 profile.

Three targeting attempts, two of them wrong, recorded so they are not redone:

1. **Main loop, dynamic shapes.** Fails outright. Inductor lowers flex decode
   for fully static shapes ONLY -- `NoValidChoicesError: no choices exist for
   backend`. Job 545: `dynamic=False` lowers; `dynamic=True`, `dynamic=None`,
   `dynamic=False + mark_dynamic(batch)` and `dynamic=True + mark_static(kv)`
   all fail. A dynamic BATCH alone is enough to kill it.
2. **Static tail only.** Lowers, but is INERT. NOTES.md:1285 already had the
   count: 32 `rollout_tail_step` calls against 2080 `generation_step` calls,
   1.52% of decode steps. The tail needs >=96.9% ended to engage and
   `ended_fraction` topped out at 0.948 over 11 pools (job 558). The "~85% of
   decode iterations" figure at NOTES.md:202 predates `--rollout-groups 32`
   and does NOT survive it -- do not reuse it.
3. **Main loop, static power-of-two buckets.** Lowers and is fast in isolation.
   Rests on one measured fact: an empty KV range costs no read AND returns
   exactly zero (job 564 -- a zero softmax denominator could as easily have
   given NaN, which 0-weighted masking would then spread). So surplus rows in
   a bucket are free, which is what makes rounding survivors UP affordable.

Microbenchmark, production shape (6 layers, 4x128, 2560-key cache, job 562),
bucketed flex vs the dynamic SDPA step it replaces:

| pos | 128 | 512 | 1024 | 1280 | 1536 | 2047 |
|---|---|---|---|---|---|---|
| bucket/sdpa | 0.72x | 0.70x | 0.63x | 0.59x | 0.54x | 0.29x |
| bucket/exact | 1.02x | 1.00x | 1.01x | 1.06x | 1.01x | 1.19x |

**But it does not show up end to end.** At `--rollout-groups 16`
(jobs 570/571): `decode_ms_per_action` 19.94 -> 19.48, **-2.3%**, inside the
+-5% control noise. A 30-70% kernel win producing ~0% at pool level means
attention is a far smaller share of the decode STEP than 32% of device time
suggests -- the step is substantially launch-bound. Anyone reviving this
should profile the step's kernel mix FIRST and confirm the attention share at
the target row count, before optimizing the attention.

### Two KV cache sets coexist across rollout groups (PRE-EXISTING, unfixed)

Instrumented `new_caches` (job 569), g32:

```
rows=32  length=134   resident=0.83GiB   <- group 1 prefix
rows=512 length=2560  resident=3.16GiB   <- group 1 expanded (15.0GiB)
rows=32  length=255   resident=15.91GiB  <- group 2 prefix: group 1 STILL RESIDENT
rows=512 length=2560  resident=18.34GiB  <- OOM on the second 15GiB
```

`del batched` is in place and `LatentRolloutBatch` holds no cache reference,
so this is a cycle the collector has not run on. The shipping config survives
only because its cache follows the ACTUAL padded prompt width (2182 here, two
sets = 25.6GiB) rather than a pinned `prompt_tokens + max_stream_steps` (2560,
two sets = 30GiB). Control peak already reached 22.11GiB of 31.36GiB -- the
headroom is luck, not design. Worth fixing on its own merits; it is what
blocks any change that widens the rollout cache.

### Decode is launch-bound, measured (job 606, boolean control path, pool 5)

The suspicion in the flex-decode section above ("the step is substantially
launch-bound") is now a measurement rather than an inference.

| | |
|---|---|
| pool wall | 14.408 s |
| total device time, all kernels | 9.832 s (**32% of wall has no kernel resident**) |
| host launch calls | 724,341 per pool |
| decode steps per pool | 2,080 |
| **launches per decode step** | **~348** (a SIX-layer model: ~57 per layer) |
| decode phase | 8.844 s wall = 4.25 ms/step |
| decode attention (`fmha_cutlassF`) | 4.093 s / 12,492 calls = 328 us each, 6/step |

The tail is mostly work smaller than its own launch:

| kernel | calls/pool | per step | device s | mean |
|---|---|---|---|---|
| `triton_poi_fused__to_copy_2` | 73,614 | 35 | 0.087 | **1.2 us** |
| `triton_poi_fused__to_copy_1` | 58,998 | 28 | 0.123 | 2.1 us |
| `triton_tem_fused__rms_norm_addmm_view_3` | 49,920 | 24 | 0.210 | 4.2 us |
| `triton_poi_fused__to_copy_8` | 24,960 | 12 | 0.095 | 3.8 us |

157k launches for 0.305 s of device time. Note the run profiled the BOOLEAN
path (`fmha_cutlassF` is the memory-efficient SDPA kernel), so the launch
count is the control's, not flex's.

Blocking host syncs: 336 over the sampled pools (~48/pool), concentrated at
`latent_rollout.py:808` (132, the compaction active-row count), `:1323` (96),
and `train_latent_vapo.py:4433` (64). Real but the smaller half; the launch
bubble is the larger one.

Also from this profile: `rollout_tail_step` compiled and was called **zero**
times under `--rollout-tail-graph`, so its compile time is pure waste in that
configuration. Worth understanding before trusting the tail path.

### Flex decode at production shape: NEGATIVE, confirms the g16 result

Job 592, `ab9b_flex` vs `ab9_ctrl` truncated to a matched 250-pool slice:
`decode_ms_per_action` 27.57 -> 27.32, **-0.9%**, inside control variance. The
earlier reduced-shape -19.5% badly overstated it. This is the third
measurement agreeing (g16 -2.3%, production -0.9%) and the profile above says
why: attention is 46% of decode wall and the step cannot go faster than its
launch rate.

### The paged tests could not fail (2026-07-27)

`attn.proj.weight` is zero-initialised in every backbone here, so a freshly
constructed test model's attention branch contributes EXACTLY zero.
`test_paged_rope/pope_gqa_matches_masked_dense_decode` and
`test_paged_refill_hides_a_stale_longer_suffix` were therefore passing on a
model where the KV cache is unreachable -- they asserted parity of a quantity
neither arm read. Added `_wake_attention()`; with attention awake they
immediately caught a live ownership bug in the new page-table `mask_mod`.

This compounds the eager/block-table blind spot already recorded above. A
paged decode test is only meaningful if BOTH hold: attention is awake, and
either the block tables are walked explicitly or the test runs compiled on
CUDA.

### Page pool: a free list alone saves nothing (scoping, 2026-07-27)

`capacity_rows * pages_per_lane` remains the worst-case bound however pages
are handed out, so an allocator only helps if one of two things gives.
Truncation is out ("no trajectory is shortened"). That leaves DYNAMIC
ADMISSION: admit while free pages cover every live row's worst-case
remainder, so concurrency throttles when rows run long and rises when they do
not. The median row uses ~1/5 of `pages_per_lane`, so the same arena should
hold roughly 4x the rows on average -- which is the right lever for a
launch-bound loop, since it multiplies useful work per launch rather than
reducing launches.

Stage (a) (addressing decoupled from reservation, identity mapping, no
behaviour change) is landed. Stage (b) is allocator + admission policy, not
just a free list.

### Static decode arena: short chunks pad up, they do not run narrow (2026-07-27)

`--rollout-graph-decode` hands `rollout_continuations` a caller-owned KV
arena of `max(rollout_groups,1) * samples_per_prompt` rows so the main decode
loop holds one shape and `mode="reduce-overhead"` can capture it. The first
shape of that flag REJECTED any chunk that did not fill the arena exactly,
which kills every fresh run at its first value-warmup rollout
(`prompts_per_minibatch` prompts, not `rollout_groups`), plus the final short
pool and `--consume-all-prompts`.

Short chunks now pad the PROMPT batch up to the arena with copies of the last
prompt, mark those rows `ended` before step zero (empty key range, `record`
never writes them), and drop them from the returned batch. The alternative --
run a row-prefix of the arena -- is compute-optimal but records one graph per
distinct row count, which is exactly the cost the arena exists to remove.

Known cost, NOT yet measured: at the bench config the warmup rolls 16 prompts
into a 32-prompt arena, so those steps do ~2x the decode work. Read the A/B on
steady-state steps, not the aggregate. If capture wins, the fix is to pad to a
`row_bucket` multiple and slice the arena instead -- 256 and 512 are both
multiples of 64, so the bench would land on two shapes and zero waste.

Second known cost: the arena is resident through the update, where the
per-chunk cache it replaces was freed. At the bench config (512 rows x 2560
width x 6 layers x 2 tensors x 4 heads x 128 head_dim, bf16) that is 15.0 GiB
held against a 32 GB card for the whole run. Padding adds two more transients
on a short chunk: the stream tensors are sized at ARENA rows, so `thoughts`
((rows, stream, 512) fp32) doubles to 2.5 GiB at the warmup shape, and
`_drop_filler_rows` clones the real prefix while the padded original is still
live. Peak VRAM is the thing to watch in the A/B, not just decode time.

The bench DOES exercise all of this: `bench.sh` sets `--value-warmup-steps 50`
and is not `--rollout-only`, so 50 collects of `prompts_per_minibatch = 16`
prompts run into the 32-prompt arena before the first actor step. Launches per
step are unaffected by the row count -- same kernels, wider shapes -- so the
launch-count question the A/B exists to answer is not confounded.
`decode_ms_per_action` on those pools IS confounded, roughly 2x, because the
control runs 256 rows where the arena arm runs 512.

RNG is not paired between the two arms: with no explicit generator the trainer
samples at the padded row count, so a graph run and a non-graph run diverge in
draws from the first warmup collect. Same class as the existing compaction
caveat -- the arms are independent samples, not a paired comparison.

Stale KV in a reused arena stays unreachable for the same reason it does
per-chunk: every slot inside a row's live range `[decode_starts, head]` is
written this chunk before it is read. That argument covers a non-finite value
as much as an ordinary stale one, so the arena is deliberately never
re-zeroed.

### Graph decode A/B round 1: control 631 clean, graph 632 OOM (2026-07-27)

Both arms from one frozen tree, `--steps 40 --rollout-compile
--rollout-flex-decode`, profiling actor pools 5-6.

`ab10_ctrl` (631) succeeded. Its pools are the flex control this flag has to
beat, and they are worse than the boolean job-606 profile in the way that
matters:

| | pool 4 (actor 20) | pool 5 (actor 24) |
|---|---|---|
| pool wall | 11.876 s | 30.011 s |
| total device time | 7.796 s | 7.851 s |
| host launch calls | 803,493 | 815,404 |
| decode steps (flex kernel calls / 6 layers) | 2,080 | 2,080 |
| **launches per decode step** | **386** | **392** |
| idle wall | 34% | **74%** |

Pool 5 spends 25.3 s of 30.0 s in decode for the same 7.85 s of device work
as pool 4's 11.9 s pool -- `uneven: collect.decode longest occurrence 21.404 s
of 25.289 s over 2`. A pool whose chunks are ragged runs its long tail nearly
empty. Launch rate is the ceiling, and it is 386-392 per step here versus the
348 measured on the boolean path.

Blocking syncs, 343 over the two pools: 132 at `latent_rollout.py:908` (the
`int(active.sum())` compaction count), 96 at `:1458`, 64 at
`train_latent_vapo.py:4477`.

`ab10_graph` (632) OOMed. It got further than the flag's previous shape ever
did -- `flex_generation_step` ran 2,080 times, one whole padded warmup chunk
decoded correctly, and cudagraphs captured (248 MiB in private pools) -- then
died in the FIRST value-warmup update, at `refresh_old_statistics` ->
`compact_emit_token_logprobs`, needing 590 MiB with 109 MiB free of 31.36 GiB.

That is the predicted failure at the predicted place: 15.0 GiB arena resident
through the update, plus the warmup's padded stream tensors (16 prompts into
a 32-prompt arena doubles `thoughts` to 2.5 GiB) plus `_drop_filler_rows`'
clone. Caveat: the card was shared -- ~1.5 GiB belonged to three of the user's
concurrent jobs -- so the exact threshold is not reproducible, but the margin
was ~500 MiB and the padding waste is ~2.5 GiB, so the ordering of the
conclusion does not depend on the confound.

Next: re-run the pair with `--value-warmup-steps 0`. Every chunk is then
exactly 32 prompts = 512 rows = the arena, so there is NO padding, no clone,
and the question reduces to whether the resident arena alone fits. That also
removes the ~2x confound from `decode_ms_per_action` on warmup pools. If it
still OOMs, drop to `--rollout-groups 16` (7.5 GiB arena) to get the
launches-per-step answer at a shape that fits, remembering that a smaller
batch flatters capture: less work per launch means more of the win is
available.

### Round 2 (`--value-warmup-steps 0`): control reproduces, graph still OOMs

Jobs 636/637. Control 636 reproduces 631 almost exactly -- pool 4
382 launches/step at 34% idle, pool 5 389 at 76% idle, against 631's 386/392
and 34%/74%. The metric is stable run to run, which is what makes it the one
worth deciding on.

New detail from 636: on the two starved pools the card reports 107-111 W at
255-690 MHz, against 291-307 W at 2835 MHz on the fast ones. It is not merely
idle, it is DOWNCLOCKING because it is starved. Direct confirmation of a
launch-bound decode.

Graph 637 OOMed again, at the first actor update, needing 50 MiB with 137 MiB
free. Removing the padding recovered ~2.5 GiB and moved the shortfall from
590 MiB to 50 MiB -- so the padding really was most of the round-1 excess, and
the resident 15.0 GiB arena alone is still slightly over budget at g32.

Caveat that matters here: ~1.55 GiB belonged to three of the user's
concurrent jobs both times. On an EMPTY card 637 would very likely have
passed. The flag is not intrinsically 15 GiB over -- it is marginal, and
marginal on a shared card means unusable.

Round 3 (jobs 638/639) drops to `--rollout-groups 16`: 256 rows x 2560 = 7.5
GiB arena, comfortably inside budget. Read the launches-per-step delta there
and remember it FLATTERS capture -- a smaller batch does less work per launch,
so a larger share of the step is launch overhead available to remove.

### CUDA-graph decode at g16: LARGE WIN (jobs 638 vs 639, 2026-07-27)

Same frozen tree, `--steps 40 --value-warmup-steps 0 --rollout-groups 16
--rollout-compile --rollout-flex-decode`, +/- `--rollout-graph-decode`.

| | ctrl pool 4 | graph pool 4 | ctrl pool 5 | graph pool 5 |
|---|---|---|---|---|
| **decode phase** | 15.344 s | **7.211 s (-53%)** | 74.453 s | **7.942 s (-89%)** |
| pool wall | 20.14 s | **11.75 s (-42%)** | 80.47 s | **13.06 s (-84%)** |
| total device time | 8.63 s | 9.37 s | 9.55 s | 9.78 s |
| idle wall | 57% | **20%** | 88% | **25%** |
| host launches | 1,389,139 | 718,858 | 1,395,416 | 732,795 |
| decode steps | 4,160 | 4,160 | 4,160 | 4,160 |
| **launches / step** | 334 | **173 (-48%)** | 335 | **176 (-47%)** |
| decode W mean / min | 176 / 129 | **291 / 253** | 80 / 38 | **262 / 142** |

Device time is UNCHANGED to within 8% -- the same kernels do the same work.
Everything above is bubble removal. Capture halves the launches per decode
step and the card stops downclocking: pool 5's decode ran at 80 W mean / 38 W
min in the control and 262 / 142 under capture.

Pool 5 is the shape of the win. It is the ragged pool whose longest chunk
outlives the others, and the control ran it at 88% idle for 74 s. Under
capture it costs 7.9 s. Launch rate was not merely A ceiling on that pool, it
was ~7x the real work.

Caveats, both real:

1. g16 FLATTERS this. A 256-row step does less work per launch than a 512-row
   step, so a larger share of it is overhead available to remove. The g32
   control sits at 382-389 launches/step and 34-76% idle against g16's
   334-335 and 57-88%, so the g32 win will be smaller. How much smaller is
   unmeasured.
2. g32 does not RUN yet: the 15.0 GiB arena OOMs (round 2 above, short by
   50 MiB on a card also holding ~1.55 GiB of other jobs).

The syncs did NOT move: 487 -> 467 over two pools, still 264 at
`latent_rollout.py:908` (`int(active.sum())`, the SYNC_EVERY liveness check,
which runs whether or not compaction is enabled) and 96 at `:1458`. Capture
removed the launch bubble, not the sync bubble. That is the next lever, and it
is now the larger remaining one.

**The unblock for g32 is task #4, not more graph work.** The round-2 shortfall
was 50 MiB. `thoughts` is `(rows, stream, 512)` fp32 = 2.5 GiB per g32 chunk
with two chunks retained, so storing THINK state sparsely frees GiBs where
tens of MiB are needed. #4 was scoped as a memory tidy-up; it is now the
dependency that makes the biggest measured perf win in this line of work
usable at the production shape.

### KDA 3:1 means mixers, not 24 full transformer blocks (2026-07-29)

The literal 18-KDA + 6-dense full-block implementation was the wrong cost
model. It has 24 MLPs, 126,688,438 parameters, and measured 3,553 ms/step on
the 5090. It did fit and complete 20 training steps at MBS 8, but it is 3.2x
the six-dense baseline's wall time before any quality evidence.

The corrected default keeps the original six complete dense blocks and adds
18 attention-only KDA residual mixers in `KKKD` x6. It preserves vocab 50,304,
dimension 512, sequence length 1,024, MBS 8, and global batch 524,288. Static
facts:

| architecture | parameters | raw params | params + grads + optimizer state |
|---|---:|---:|---:|
| 6 dense | 70,470,784 | 219.7 MiB | 806.8 MiB |
| 6 dense + 18 KDA mixers | 88,884,406 | 289.9 MiB | 1,017.9 MiB |
| 6 dense + 18 full KDA blocks | 126,688,438 | 434.2 MiB | 1,450.8 MiB |

Thus the mixer design adds 18,413,622 parameters, 70.2 MiB of raw weights,
and about 211.1 MiB of persistent training state over baseline. A CPU
construction/initialization check increased process peak RSS by 475 MiB versus
403 MiB for baseline, only +72 MiB. All 377 trainable tensors have exactly one
optimizer owner. The six retained dense blocks and global tensors are
bit-exact with the six-layer baseline at initialization (89/89 comparisons);
KDA insertions use isolated local RNG streams.

Approximate forward math, including the large vocabulary projection, is 1.41x
baseline for the mixer design and 2.16x for full KDA blocks. Current kernels
are much less efficient than that arithmetic suggests. From measured
1,104 ms dense, 1,407 ms five-KDA replacement, and 3,553 ms 18-full-KDA
timings, the mixer design is expected around 2.7-3.1 s/step until the KDA
training path is improved. The exact queued benchmark is job 874.

The full-block model already trained at the production microbatch shape on the
32 GB card. The mixer removes 37.8M parameters and 18 saved 2,048-wide MLP
activation paths, so it is safely below that observed bound. Expected peak is
roughly 8-12 GiB in default compile mode and potentially 10-16 GiB with CUDA
graph pools; job 874 now logs allocated/reserved VRAM and process RSS to
replace this estimate.

Job 872 is the corrected CUDA-graph KDA five-layer replacement run to 1,000
steps. It retains the original full-block replacement architecture explicitly
for continuity with the prior KDA curve. Job 874 benchmarks the new mixer
architecture after 872 completes.

## v28 hidden-carry throughput defaults (2026-07-31, jobs 969-981)

Rollout-gate A/B on the mathmix4k checkpoint, 1024 trajectories, latent mode
(fresh combiner = identity, so numbers hold for all modes):
- lockstep 16 groups: 13.2k useful actions/s, 5.5 GiB (job 969)
- lockstep 32 groups: 42.4k, 10.4 GiB (978)  <- new default
- lockstep 64 groups: 41.1k, 20.0 GiB (979) — width saturates
- continuous_refill: 6.4k, 7.5 GiB (980) — 0.92 step utilization but the
  paged per-step overhead loses 6.5x to wide lockstep at this scale
Replay shard ceilings raised for the 1x-stream regime (32/4M/8192 ->
128/16M/24576): shards per 4-step pool 76 -> 28, age-0 clip stays exactly 0
(refresh and update share one shard plan and one compiled artifact).
Profiled 40-step smokes (jobs 974, 981): steady pool wall 5.78 -> 3.63 s
(decode 4.51 -> 2.29 s at 201 -> 224 W; still launch-bound, combiner adds
~12 eager launches/step — fold into step_core if it ever matters). Train
peak 7.6 GiB, rollout peak ~14.5 GiB on the 32 GiB card. BPB guard default
is now one 8192-token eval batch (~free); identity gates against recorded
pretraining val_bpb need --bpb-val-tokens 2097152.

## KDA post-training adapter (2026-07-31, jobs 993-995)

The k3 campaign's checkpoints (`*_kda_kkkdkkkd_mixers_v3`) could not load into
post-training at all: `model_io` refused `_kda_` architectures. Built the
adapter while the v8 data build (989) runs:

- `nanogpt_mini_kda_model.py` — import-safe extraction of the KDA training
  script (byte-matched classes; env config -> constructor args carried by the
  checkpoint's `model_config`). Carries a pure-PyTorch reference recurrence
  matching FLA `chunk_kda` under the training flags (l2norm eps INSIDE sqrt,
  safe gate `-5*sigmoid(exp(A_log)*(g+dt_bias))`, sigmoid beta, v-major
  fp32 state) plus its own depthwise conv and gated RMSNorm with
  key-identical parameters.
- `postraining/kda_backbone.py` — `NanoKDABackbone`: dense layers keep the
  nano KV step/prefill; KDA layers carry `(conv_q, conv_k, conv_v, state)`
  caches. Decode step is pure PyTorch (compiles into the fullgraph step
  artifact); prefill/replay use `chunk_kda` (mixer is an eager region;
  replay artifacts drop to fullgraph=False for KDA only). Left padding:
  conv inputs zeroed at pads -> k=0 -> delta-rule writes vanish -> state
  exactly zero through the pad prefix (asserted, not assumed).
- `latent_rollout.py` learned that recurrent cache layers (arity 4) have no
  length axis: fan-out expansion, all three compaction branches, and every
  cache-width check now branch on it. The old code would have sliced the
  [B,H,128,128] state's Dv axis by `prompt_length` — silently wrong whenever
  a prompt was shorter than 128 tokens.
- Muon partition: conv windows (ndim 3) excluded, matching pretraining's
  KDA_CONV_LR AdamW group; A_log/dt_bias fall through to AdamW on ndim.
- continuous_refill is refused for KDA (paged KV addressing has no
  recurrent analogue); lockstep is the production path anyway.

CPU: 383 tests pass incl. new `test_kda_backbone.py` (stepwise decode ==
teacher-forced within fp32 tolerance, left-pad invariance exact, age-0
rollout->replay logprob parity, strict-load via model_io round-trip).
GPU gates queued through mlq: 993 `kda_gpu_parity` (kernel-vs-reference,
dense-vs-decode logits on the 986-layout trunk, left-pad through CUDA
kernels), 994 `kda_train_smoke` (4-step end-to-end on a synthetic random
`logs/kda_synthetic_random_dev.pt` in the exact 990 payload format), then
995 `k3_base_rollout_gate` chained --after-success on the real 20k
checkpoint `logs/k3_quality_20k_ctx8k_final_model.pt`.

Red-team review (independent session) found no correctness bugs; verified
the recurrence derivation by executing it against FLA's naive_recurrent_kda
(1.5e-8) and naive_chunk_kda (3.1e-8), traced all four recurrent
compaction/expansion branches to actual line hits, and confirmed the
compile posture (step_core fullgraph OK; replay needs graph breaks exactly
because the mixer is compiler-disabled). Fixes applied from its findings:
- `nanogpt_mini_kda_model.py` now calls chunk_kda with
  disable_recompute=False (pretraining ships True for backward speed on
  8xH100; in grad-enabled replay it would retain per-layer w/u/qg/kg/v_new/h
  at replay-shard width — roughly a GiB extra across 6 mixers at 24576
  slots, an OOM risk on the 32 GiB card). Forward values identical.
- `prefill_belief` cache copies are now zip(strict=True) + shape-checked.
- `kda_gpu_parity.py` left-pad check compares all four cache tensors, not
  just the delta state.
- Pretraining Muon/PerHeadMuon: `state["mu"]` is recreated when missing so
  resuming a checkpoint that predates tensor-valued mu no longer KeyErrors.
Known and accepted (documented, not fixed): recurrent-layer detection is
tuple arity (4), fine for dense pair/PoPE triple/KDA quad but a future
4-tuple length-addressed cache would need a per-layer flag; the delta state
assumes Dv == Dk (true for every KDA config here); post-training Muon does
not reproduce pretraining's PerHeadMuon on q/k/v (deliberate fine-tuning
choice); left-padded replay differs from rollout by ~0.07 nats on dense AND
KDA trunks alike (pre-existing: replay's causal attention has no pad mask
and dense QKV biases emit nonzero K/V at zeroed pads) — PPO age-0 is
unaffected since refresh and update both price through replay.

Gate results (2026-07-31): 993 kda_gpu_parity PASSED all bounds with wide
margins (kernel-vs-reference 1.4e-4 out / 1.4e-3 state vs 5e-3 bound; fp32
prefill/decode vs dense 2.0e-4 / 6.5e-4 vs 2e-3; left-pad 2.3e-4 logits /
3.7e-4 over all four cache tensors vs 1e-3; bf16 decode 1.6e-2 vs 0.5
rail). 994 kda_train_smoke PASSED end to end on the synthetic checkpoint:
rollout + replay + PPO updates + checkpoint save, 0 blocking syncs over the
profiled pools, steady state roughly 15 s rollout (2048 max-length decode
steps, random model never EOSes — worst case) + 1.7 s per update.

10-hour run queued as 997 k3_latent_10h (--after-success 995): latent mode
on logs/k3_quality_20k_ctx8k_final_model.pt, --steps 40000 as an
unreachable ceiling, stop by the new --max-train-hours 10 flag (pool-
boundary wall-clock truncation; LRs are constant so early stop is a
truncation, not a schedule change; refused with --consume-all-prompts).
Chain: 996 kda_resume_smoke (running) -> 990 pretraining -> 995 rollout
gate -> 997.

## 2026-07-31: mid-run compile stalls — root cause and fix (varlen chunk_kda)

k3_latent_10h showed sporadic ~12 s full-GPU-idle stalls (95% of early
wall time, decaying but never gone). First attribution — Dynamo
specializing on replay-shard row counts — was WRONG, established by
red-team review:

- Under the default --no-duck-shape, DUCK and DYNAMIC dims both get fresh
  symbols, so shard dims were already dynamic; the added dim-0
  maybe_mark_dynamic in select_trajectory_rows is a no-op under defaults
  (kept as hygiene for the duck-on config, where it also removes the
  512-collision).
- Real cause: TileLang JIT compiles of FLA's KDA fwd/bwd kernels. The
  kernel builders bake batch size B into the JIT cache key (T and seq
  count are T.dynamic). Every distinct shard row count reaching a KDA
  replay = fresh ~12 s compile: 85 compiles / 999 s in the first 40 min
  of job 997 (attempt 813 log). Aggravated by --replay-max-trajectories
  32 -> 128 (4x wider B space). torch.compile machinery can't see it:
  KimiDeltaAttention.forward is compiler-disabled, and
  counters["stats"]["unique_graphs"] counts only Dynamo forward graphs.

Fix (nanogpt_mini_kda_model.py, single chunk_kda call site): varlen form.
Flatten [B, T] -> [1, B*T] with uniform cu_seqlens (+ cu_seqlens_cpu twin,
no H2D copy; device arange + CPU arange). Same per-row math — chunking
and state resets are per sequence — but B == 1 for every shard, so each
kernel has exactly ONE JIT key for the whole run. No row bucketing, no
filler rows, no padding waste. final_state comes back [N=B, H, Dv, Dk],
identical to batched.

Telemetry (train_latent_vapo.py): three monotonic gauges now cover the
three compile populations — perf/dynamo_unique_graphs (Dynamo fwd),
perf/aot_autograd_compiles (AOT bwd), perf/tilelang_kernel_compiles
(logging hook on tilelang.jit.kernel "begins to compile"; the population
unique_graphs is blind to).

Verification: CPU suite 384 pass (varlen branch is CUDA-only). GPU gates
queued behind 997: job 1002 kda_parity_varlen (fp32 prefill/decode vs
dense bounds exercise varlen through the wrapper) and job 1000
dynamic_rows_gate (60 steps on the real checkpoint; gate criterion is
perf/tilelang_kernel_compiles plateauing at the per-kernel-type count —
NOT unique_graphs, which stays flat regardless). 997 left running: its
key space had saturated (~1 compile/100 steps); fix lands on resume/next
run.

TRT/vLLM reference takeaways recorded for the graph-decode design: shapes
declared never discovered; out-of-range = hard error at config time (TRT
demo refuses cudagraph+dynamic at argparse); persistent buffers copied
into, never rebound; warmup enqueue before capture; pad up to the bucket
rather than adding buckets; make "zero compiles this run" assertable
(error-on-cache-miss analog: the three compile gauges above).

## 2026-07-31: vLLM-style CUDA-graph decode for the continuous scheduler

Reference reports (vllm + TensorRT clones) confirmed the paged path was
already structurally graph-ready: fixed physical lanes = capture sizes,
empty-range/scratch-page padding = PAD_SLOT_ID discipline, caller-owned
arena = bind_kv_cache, slot-indexed lanes = "index rather than move".
The only gap was the compile mode and capture-before-serve.

Implemented (flag-gated, off by default):
- --rollout-graph-decode now legal with continuous_refill (config.py);
  lockstep keeps its flex-decode requirement, TRT-style config-time error.
- rollout_paged_step_core compiles mode="reduce-overhead" under the flag
  (one cudagraph per declared execution-width bucket, inductor cudagraph
  trees share one memory pool).
- Paged arena tensors mark_static_address'd at allocation (mutated graph
  inputs must be statically addressed or inductor silently skips capture).
- warmup_decode_width_buckets (rollout_scheduler.py): drives every
  declared width (capacity + pow2 tails, largest first, 3 passes, all
  rows dead -> scratch-page writes only) through the compiled step at
  arena creation, inside the patch window under no_grad + autocast — all
  graphs captured before the first real token, shapes declared never
  discovered.
- Tests: config gating; warmup covers the exact _decode_execution_width
  range and a warmed arena reproduces a fresh arena's rollouts
  bit-identically. Suite: 386 pass.

Queued A/B behind parity gate 1002: jobs 1003 (graph_decode_ab_off) /
1004 (graph_decode_ab_on), latent rollout-only x3 repeats on the k3
checkpoint. Decision metric: collect_seconds / decode_step_utilization;
also check tilelang/aot/dynamo compile gauges stay flat after pool 0.
Red-team review of the change in flight (cudagraph mutation semantics,
output lifetime across replays, warmup inertness on the KDA cache, flex
BlockMask under capture, memory pinned by the capture set).

Red-team of the graph-decode change (torch-source-verified) found 4 real
issues, all fixed before the A/B:
1. Warmup captured bf16 graphs but the decode loop feeds fp32
   (embed_tokens is fp32 under autocast) — every captured graph was dead
   and the real widths would have compiled+recorded lazily mid-rollout.
   Fix: warmup builds its input through model.embed_tokens, guard-identical
   to the loop (dtype AND stride); helper's dtype param removed.
2. reduce-overhead silently drops max_autotune + coordinate descent vs
   the control arm -> confounded A/B. Fix: mode="max-autotune" (==
   max-autotune-no-cudagraphs + triton.cudagraphs) under the flag.
3. dynamo cache_size_limit (8) < declared width count; only raised in the
   eval/lockstep branch. Fix: raised at paged-artifact creation.
4. No signal for a silent cudagraph skip. Fix: perf/cudagraph_skips gauge
   + torch._inductor.config.triton.cudagraph_or_error=True under the flag
   (skip -> RuntimeError, declared-shapes doctrine).
Verified-safe by the review: mutation/static-marks chain (marks are
load-bearing and sufficient; only cache K/V mutated in-graph), output
lifetime (scatter_prefix copies immediately; nothing aliases graph
memory), eager-mutation-between-replays semantics (INFERENCE-mode
artifact + per-call generation bump; @no_grad on warmup AND rollout is
load-bearing — documented in the helper docstring), warmup inertness
(scratch pages only; arena is dense-KV only — KDA + continuous_refill is
hard-refused upstream), BlockMask (eager per-step inputs, no capture
conflict), memory (capture set ~0.2-0.4 GiB vs 12.9 GiB arena).
Also: page_home now static-marked (saves a per-replay copy).

A/B rescheduled on the DENSE checkpoint (jobs 1005/1006 replace
1003/1004, which would have hard-errored: k3 is a KDA trunk and the
paged path refuses KDA). Launch-overhead conclusions transfer; wiring
graph decode into a KDA production run needs either the lockstep+flex
arm or KDA paged-cache support — future work, gated on the A/B result.

## KDA continuous-refill support (2026-07-31)

Motivation: k3_latent_10h runs lockstep at ~7.7% decode-lane utilization with
collect = 76% of pool wall time; the dense A/B showed continuous refill reaches
92.5% step utilization (but ran 6.4x slower eager — hence graph decode, jobs
1010/1011). KDA trunks were hard-refused on the continuous path. Port done:

- `make_paged_generation_cache` allocates 4-tuple lane arenas for recurrent
  layers (3 conv windows cache-dtype, fp32 state), 2*capacity rows; rows
  [capacity, 2*capacity) are per-row scratch sinks — the recurrent analogue of
  scratch_addresses, needed because the scheduler pads decode batches with
  slot 0 (a live lane) and KDA step writes are unconditional (index_copy_ with
  duplicate indices is UB).
- `paged_step` computes `lane_rows = where(live, slot_ids, capacity+arange)`
  and threads it through `paged_step_core` to `_block_paged_step`; the KDA
  implementation is gather -> attn.step (mutates the gathered copies) ->
  index_copy_ back. Dense/fresh backbones take-and-ignore the new arg.
- `admit_prompt_prefixes` fans conv/state rows per lane (full-row overwrite is
  what stands in for kv_starts stale-suffix masking) and reads
  `bank.prompt_width` instead of a layer-0 axis (layer 0 may be recurrent).
- Trainer refusal removed; lockstep-only compaction unaffected (continuous
  scheduler never moves live lanes; assign_pages has no callers).

Found while testing, worth remembering: raw per-chunk scheduler batches with
left-padded rows are NOT replayable — rollout excludes pads structurally,
replay only zeroes their inputs, and residual biases re-inflate them from
block 0 (live dense trunk diverges 0.39, KDA 0.34). Production is unaffected
because the trainer only replays `split_rollout_groups` output (per-group,
pad-trimmed). The old dense scheduler test replayed the raw chunk batch and
passed only because the fresh-init trunk is an identity residual stream; it
now livens the trunk and splits first, as does the new KDA variant.

Tests: 4 model-level (paged parity vs per-row dense decode incl. 2 steps,
padding-row lane inertness, lane recycling exactness, fullgraph one-graph) +
2 scheduler-level (lifecycle, split-group carry-vs-replay parity). 392 CPU
tests green. GPU validation: queue a k3-checkpoint rollout-only A/B
(lockstep vs continuous_refill x3) after 1011; graph-decode-on-KDA waits for
the 1010/1011 verdict.

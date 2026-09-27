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
- Plot: `python3 scripts/plot_ablations.py`

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
python3 scripts/ablation.py

# Custom step count
python3 scripts/ablation.py --steps 2000 --name baseline_2k

# Sweep learning rates
python3 scripts/ablation.py --sweep lr --steps 2000

# Compare results
python3 scripts/ablation.py --compare

# Plot
python3 scripts/plot_ablations.py

# Full baseline (match official)
python3 scripts/ablation.py --steps 13780 --name baseline_full

# Manual single run with env overrides
python3 scripts/ablation.py --steps 2000 --name my_test --env MODEL_DIM=640 NUM_HEADS=10
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
to `pretraining/nanogpt_mini/nanogpt_mini_model.py`). RL prerequisite: pretrain on `mathmix_v4_sp1024`
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
- `scripts/build_math_mix_dataset.py --tokenizer gpt2` builds the same mix under GPT-2
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

- `pretraining/nanogpt_mini/nanogpt_mini_kda_model.py` — import-safe extraction of the KDA training
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
- `pretraining/nanogpt_mini/nanogpt_mini_kda_model.py` now calls chunk_kda with
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

Fix (pretraining/nanogpt_mini/nanogpt_mini_kda_model.py, single chunk_kda call site): varlen form.
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

Red-team follow-up (2026-07-31, second reviewer pass — verdict: no confirmed
bug, with active verification: inductor-vs-eager parity 4.8e-7, 20-step
dirty-scratch stress, pad on/off scheduler bit-identity, and confirmation the
scratch redirect is load-bearing). Actions taken:
- B3: `paged_step` now validates lane_rows duplicate-free on CPU (sync-free
  path only, mirroring admission's prefix-address check) + a test that
  duplicate slot_ids without a live mask raise.
- B2/C4 comment corrections: index_copy_ with duplicates is NOT UB — it picks
  a winner nondeterministically (documented PyTorch semantics). Per-row
  scratch sinks are kept anyway: determinism-mode compliance, and scratch
  rows ARE re-read (a row dead for several steps gathers its own stale
  scratch each step — safe because dead outputs are never consumed and
  admission rewrites lanes wholesale, but a shared sink would make that
  reread racy). Cost at production shape (~1.77 GiB extra) accepted.
- B5 tests added: (1) 20-step dirty-scratch soak — same padded schedule run
  with and without scrubbing scratch rows to zero each step must be
  bit-identical in live logits and lane storage (same batch width both runs,
  so bit-equality is legitimate); (2) real-KDA scheduler pad_decode_width
  on/off — tokens+logprobs bit-identical, stored carries within 1e-5 (B4:
  cross-width BLAS numerics move beliefs ~7e-7 measured; lane corruption
  moves them ~0.3 — three orders of separation, so the bound discriminates);
  (3) KDA warmup_decode_width_buckets preserves arenas bit-exactly.
- B4 disposition: width-bucket numerics do NOT threaten the age-0 canary —
  refresh_old_statistics recomputes behavior stats through the replay path,
  so the canary compares replay-vs-replay; rollout-side wobble is an off-policy
  perturbation the refresh discipline already absorbs. No code change.
- B1: kda_gpu_parity.py gained section 4 — paged continuous-refill decode
  (8-lane pool, shuffled ragged admission, 32 steps with 4 dead padding rows
  per step) vs per-row dense decode (bound 2e-3 fp32), plus compiled
  paged_step_core (fullgraph, dynamic=False — the production shadowing
  pattern) vs eager (bound 2e-3). Queued as job 1012.
- 396 CPU tests green. Queue: 1012 kda_paged_parity -> 1013
  kda_sched_ab_lockstep -> 1014 kda_sched_ab_continuous (k3 checkpoint,
  --rollout-only x3), priority 0 so the user's sffactor jobs (999/1001) go
  first. Graph-decode-on-KDA still waits for the 1010/1011 verdict AND 1012.
Deferred: C1 (tuple-arity dispatch hardening) and C2 (skip prefix_addresses
on fully-recurrent trunks) — no fully-recurrent trunk exists in the lineup;
revisit if one does.

## Combiner reparameterization: drop the scalar gate (2026-07-31)

Diagnosis (from k3_latent_10h telemetry, user-confirmed direction): the v1
combiner `base + gain*W(h) + type_bias` (gain zero-init, W orthogonal) is a
multiplicative saddle. Actor gain oscillated at ~1e-4 for 8k+ steps with W
frozen at init rms 0.0442; only type_bias trained. Checkpoint forensics
sharpened it: the CRITIC's gain trained to -0.063 (value CE has no ratio
gate), so the saddle is starvation of the actor path specifically, and the
mechanism works when gradient reaches it — evidence the reparameterization,
not the idea, was the blocker. v2: `base + W(h) + type_bias`, W zero-init
(ControlNet zero-conv pattern; dL/dW = loss direction outer hidden,
full-rank from step 1). Schema: zero_init_hidden_residual_prenorm_mlp/v2.
"Concat after its own linear" (considered): a linear over a concat
decomposes into a sum of two linears, so under the identity-at-init
constraint it IS the zero-init additive form; only nonlinear cross-terms in
the combiner MLP would differ, and nothing implicates those.

Mid-run surgery (postraining/fold_combiner_gain.py): W' := gain*W is
function-preserving, so the 1008 run's ~2h of trunk/critic training
carries over. Optimizer remap: drop gain's global index (actor: combiner
group position 0 — Module.parameters() yields direct Parameters before
children, so v1 order is [gain, type_bias, carry, ...]), shift higher
indices, drop carry's Adam moments (accumulated under the gain-scaled
parameterization; lazily reinitialized). Muon optimizers untouched (no
combiner params). Verified on a live checkpoint copy: strict model+critic
loads, all 4 optimizer loads through build_optimizers, fold error 1e-10
(actor) / 1.2e-7 (critic). 396 CPU tests green after the reparameterization.

Plan: after diff review — cancel 1008 at its rolling checkpoint, fold, and
resume with --max-train-hours = 7.5 minus 1008's elapsed (same deadline).
Watch combiner/carry_weight_rms (now starts at the folded near-zero value
and should GROW if the carry is useful) and combiner/update_rms (was ~1e-5,
should rise with the ungated gradient).

Surgery executed (2026-07-31 14:17 PDT). Diff review: APPROVED, no blockers
(reviewer independently caught the same two converter traps fixed during
development — param ordering and the critic's None-expected-index path — and
verified: optimizer group sizes, fused-AdamW lazy state init for the
moment-dropped carry, fold parity 7e-10 on real weights, no stale gain refs
repo-wide, no compile-cache hazard). Review follow-ups landed: Muon guard
scans all tensor values (Muon keys "momentum", not "exp_avg"), converter
refuses source==target, synthetic round-trip test added
(test_fold_combiner_gain.py — the script is NOT one-shot: the v1
critic_warmup_checkpoint.pt converts the same way if ever needed for
--actor-critic-init), doc drift fixed. 1008 cancelled at step ~12152
(2.49h elapsed); fold: actor gain -4.90e-4 -> carry rms 2.2e-5, critic gain
-6.42e-2 -> carry rms 3.2e-3. Job 1015 (k3_latent_10h_v2combiner) resumes
from latent_vapo_checkpoint_v2.pt with --max-train-hours 5.0 (v1 rolling
checkpoint left untouched). Cancelling 1008 auto-skipped dependents
1009-1011; resubmitted as 1016 (dynamic_rows_gate) -> 1017/1018
(graph-decode A/B) behind 1015. The user's 999/1001 took the freed lease
(they had been waiting); 1015 is the protected next-up job.
Success criteria for v2: combiner/carry_weight_rms grows off 2.2e-5 and
combiner/update_rms rises above the saddle-era ~1e-5.

## Carry-ablation deep analysis (2026-07-31, job 1020)

Question (user): is the recurrent hidden actually helping — what does the
critic read from it, is token use more coherent, why is AIME near zero?

Design (postraining/carry_ablation_eval.py, runs concurrently with 1015
against the atomic rolling checkpoint; scheduler granted a backfill bypass
of protected 1012 after bumping 1015's declaration to 2):

- Three behavioral arms on identical panels/seed: full; no_content
  (carry.weight zeroed — content channel off, type_bias+MLP alive);
  token_only (pin_emit — whole combiner bypassed, plain-token policy).
  Panels: 384 DAPO math prompts x 8 samples (the distribution with
  statistical power; accuracy ~0.25) and AIME-2024 x 16 samples
  (descriptive only — 30 prompts can't power a paired test).
- Paired per-prompt stats via new prompt_correct_counts in
  evaluate_latent_math (schema v4): bootstrap CI + sign-flip permutation
  test over prompts. The three pairwise comparisons share two degrees of
  freedom (full-vs-token_only = sum of the other two deltas).
- Mechanistic probes at the training operating point: roll 32x8
  trajectories, refresh_old_statistics with stored vs zeroed batch.hiddens
  (production replay path, no_grad — compile-guard rationale doesn't apply
  eagerly), diff old_token_logprobs/old_values at action slots; plus
  rms(W h)/rms(embed) injection scale.

Review (subagent): arm isolation, pairing validity, and stats confirmed
sound; hard action_mask dtype crash and missing no_grad caught and fixed
pre-submission; behavioral results now persist before probes so a probe
OOM can't discard the sweep.

First readout: checkpoint step 23648, actor carry rms 3.91e-3 (still
growing), math/full accuracy 0.278 avg@8, emitted mean 18.7 tokens.

### Carry-ablation results (job 1020, checkpoint step 23648)

Math panel (384 prompts x 8 avg@k, paired): full 0.2780, no_content 0.2744,
token_only 0.2754 — no significant accuracy contribution from the carry
(best delta +0.0036, p=0.15; full-vs-token_only +0.0026, p=0.066). 378/384
prompts are exact ties across arms; within-group reward std ~0.005 —
the policy is near-deterministic per prompt (entropy collapse), so GRPO
advantages are zero on ~98% of prompts and learning has starved.

What the carry DOES do:
- Termination control: emitted mean 18.7 (full) -> 26.2 (no_content) ->
  111.9 with p95 1024 and 9.3% never terminating (token_only). Long
  token_only tails are degenerate repetition loops.
- Self-consistency: boxed==Answer agreement 28/26/23 of 32 (math) and
  22/20/17 (aime) across full/no_content/token_only — monotonic.
- Policy sharpening: zeroing hiddens at replay drops taken-token logprob
  by 0.073 nats/slot mean (+1.35/sequence); injection rms is 0.75x the
  token embedding rms — the channel is large and live.
- Critic reads it weakly: |dV| mean 0.0099, p95 0.037; reward correlation
  of the critic's hidden-read is -0.16 (not reward-predictive yet).

AIME: 0/480 (full and no_content); token_only 1/480. Degenerate
16-24-token answer-only responses cannot solve AIME; the RL-entrenched
brevity actively removes any chance (the rambling token_only arm was the
only one to score). Root problem is the entropy/mode collapse plus
partial-credit-dominated reward, not the carry mechanism.

## SFT-from-modern-traces stage (v29 restart plan)

Decision after the carry ablation + R1-Zero analysis: the base policy has
no samplable multi-step reasoning repertoire, so RL-from-base starves
(mode collapse, ~zero within-group variance). Instill the repertoire by
teacher-forced SFT on modern-teacher traces, then restart RL from the
SFT checkpoint. Teacher constraint (operator): modern models only (Claude
Opus 4.6/4.7-era); NO R1-family corpora, NO Sonnet 4.6.

### Corpus: postraining/data/sft_traces_v1.parquet

Built by postraining/prepare_sft_traces.py from three HF community sets
downloaded to postraining/data/modern_traces/. GSM8K-provenance sources
verified by matching normalized problem text (160-char key) to GSM8K
canonical answers AND requiring the trace to numerically reach the truth;
final answer forced to ground truth. Exact-duplicate documents dropped in
the prep pass.

- 14,029 documents, 13,985 verified (44 unverified opus4647 kept flagged)
- opus46_10k 7,187 / opus46_ti9k 6,798 / opus4647_8k7 44 (that source is
  2/3 exact duplicates; 132 pre-dedup)
- ~7,400 unique problems, ~6,600 with 2+ stylistically distinct traces
  (useful for sampling diversity); doc tokens p50 167 / p90 433 / max
  3058, 3.2M total
- Known gap: corpus is GSM8K difficulty; RL trains on DAPO competition
  difficulty. API self-distillation over DAPO-17k is flagged as a pending
  operator decision (costs real money).

### Trainer: postraining/sft_trace_train.py

RL-exact framing (encode_prompt prompt + separately encoded completion),
whole-document packing into 4096-token rows with single-50256 boundaries
(pretraining convention) and a trained stop target per document; loss on
completions only (~2.1M supervised tokens/epoch, 66% of row tokens).
Pretraining optimizer geometry (Muon matrices + AdamW groups) scaled by
--lr-scale (default 0.1), wd 0, linear warmup+decay. Holdout split by
normalized problem identity (256 problems, 485 docs, no variant leakage).
Success gate = sampling diversity at temp 1.0 / top-p 1.0 through
evaluate_latent_math with pin_emit=True (pure token policy): accuracy,
mixed-prompt fraction, within-group reward std, on the held-out panel.
3 epochs = 291 steps at 8x4096 tokens/step. Checkpoint saved in
load_model payload shape so the RL trainer can init from it directly.

### SFT trainer review + submission (job 1023)

Red-team review (subagent) found one blocker and two calibration issues,
all fixed pre-submission:
- OOM (high): 4x4096 full-vocab grad-enabled logits = 2x the slot budget
  the replay path OOM'd at on this card. Fixed twice over: readout now
  gathers supervised positions only (exact — the readout is positionwise;
  cuts vocab memory 34%) and defaults moved to 2 rows x 4 accum (identical
  update: loss is normalized by the step-wide supervised count).
- Muon geometry (medium): postraining Muon (Polar Express, no rectangular
  scale) realizes a 2-4x smaller trunk step than nominal; the Muon rate now
  carries POLAR_EXPRESS_STEP_COMPENSATION (1.45) and warms momentum
  0.85->0.95 over 500 steps as pretraining did. Without this an --lr-scale
  ablation would mis-attribute its result.
- Gate contamination (medium-low): 512 gate tokens truncate ~5% of
  reference-length solutions; truncation registers as "mixed" groups —
  the exact pass signal. Default now 768 (panel max 674).
Clean: packing off-by-ones (verified targets==tokens[i+1] over real rows),
eval call, optimizer partition (65 params exact), split leakage (160-char
key injective over GSM8K), checkpoint round-trip, tb_watcher schema.

Submitted: job 1023 sft_traces_v1 (queued behind 1015 + KDA chain).
Gate readout = accuracy / mixed-prompt fraction / within-group reward std
at temp 1.0 top-p 1.0, avg@8 over 128 held-out problems. If samples skew
degenerate-short, queue a --min-completion-tokens 30 arm (34% of the
opus46_10k source is <30 completion tokens).

## KDA chain results (jobs 1012-1018, post-1015)

- 1012 kda_paged_parity: ALL BOUNDS PASSED. Notables: fp32_paged_vs_dense
  1.4e-4 (bound 2e-3), fp32_compiled_paged_vs_eager 6.0e-7 (bound 2e-3 —
  compile is numerically exact), bf16_decode_vs_dense 1.6e-2 (bound 0.5),
  leftpad state 4.0e-4 (bound 1e-3, tightest margin at 2.5x).
- 1016 dynamic_rows_gate: PASSED. 60 steps from base, exit 0, checkpoint
  written, step ~0.45s, val_bpb 1.2226 at step 0 (base readiness, 8192-token
  subsample). Fresh base policy within-group reward std 0.022 (vs 0.003 at
  1015's collapsed end — the base has usable variance the RL run destroyed).
- 1013 kda_sched_ab_lockstep: PASSED (3/3 repeats). Warm collect ~7.0s per
  1024-trajectory rollout (~93k useful actions/s; repeat 0 cold at 45.5s =
  compile), decode utilization 0.61-0.64, peak VRAM ~6.0GB, stream 1188-1200.
  Awaiting 1014 (continuous arm) for the A/B; 1017 backfilled ahead of it.
- 1017 graph_decode_ab_off (continuous_refill, graphs off, mathmix base):
  PASSED 3/3. Warm collect 6.7-7.4s, ~71-75k useful actions/s, decode
  utilization 0.87, peak VRAM 12.7GB. Verdict vs 1018 (graphs on) pending.
- 1014 kda_sched_ab_continuous: PASSED 3/3. Warm collect 7.9-9.0s (~72-80k
  useful actions/s), decode utilization 0.89-0.90, peak VRAM 9.3GB, cold
  start 268s (vs lockstep's 45.5s).
- SCHEDULER A/B VERDICT (1013 vs 1014, k3 base, 1024-trajectory rollout
  gates): lockstep is 15-25% faster end-to-end (93k vs 72-80k useful
  actions/s) and uses 35% less VRAM (6.0 vs 9.3GB) despite lower decode
  utilization (0.62 vs 0.90) — continuous refill's admission machinery and
  6x worse compile cold-start eat its utilization advantage at this
  scale. Keep lockstep as the default for k3-base training; revisit only
  if trajectory lengths get long/ragged enough to starve lockstep chunks.
- 1018 graph_decode_ab_on: PASSED 3/3. Warm collect 6.4-6.8s (~78k useful
  actions/s), utilization 0.87-0.88, peak VRAM 21.7GB.
- GRAPH-DECODE A/B VERDICT (1017 vs 1018, mathmix base, continuous
  scheduler): graphs on buys only ~5-9% throughput (78k vs 71-75k ua/s)
  for +9GB VRAM (21.7 vs 12.7). Not worth it — training needs that
  headroom (1015 training alone peaked 8.5GB). Keep graph decode OFF.

### SFT run progress (job 1023)

Started cleanly: 13,500 train / 485 held-out docs, 291 steps. Holdout
completion CE: 1.5695 (step 0) -> 1.0180 (25) -> 0.8986 (50).

### SFT results (job 1023) — GATE PASSED on diversity

Holdout completion CE 1.5695 -> 0.7969 over 291 steps. Sampling gate
(avg@8, temp 1.0, top-p 1.0, 128 held-out problems):
- accuracy 0.0166 (vs 1015 bench 0.013-0.019 at top-p 0.7)
- mixed-prompt fraction 0.125 (16/128) — 10x the collapsed RL run's ~1%
- within-group reward std 0.042 (vs 0.003-0.006 collapsed)
- all-wrong 112/128, all-correct 0; emitted mean 251 tokens (p95 667)
Transcripts: both teacher styles sampled on the SAME prompt (terse
one-liner AND structured numbered derivation, including correct ones);
termination clean; failures are arithmetic/comprehension (64M model),
not format. RL restart now has GRPO signal on ~12.5% of prompts.

Next: sft_rl_probe_4k — RL from the SFT checkpoint, 4000 steps, watching
whether reward climbs and mixed fraction SUSTAINS (vs from-base collapse).

### Think-token pipeline (operator directive, for future runs)

Directive: add think tokens to the vocab (no re-pretraining), stop
stripping them from traces. Implemented:
- Free vocab slack: pretraining pads GPT-2's 50257 to 50304, so rows
  50257-50303 exist untrained in embed+readout. <think>=50257,
  </think>=50258 — zero architecture change (~0 bytes of budget).
- TWO distinct tokens, not one parity fence: the </think> logit is the
  stop-thinking policy (directly measurable/biasable/rewardable — the
  carry ablation showed termination is THE high-leverage action), and a
  pair can't desync the parse. <think> opens every completion; only
  </think> is a load-bearing generated action.
- core.GPT2BPETokenizer(think_tokens=True) registers the pair (opt-in,
  default unchanged for the running stack).
- prepare_sft_traces --think-tags -> sft_traces_v2_think.parquet (built:
  same 14,029 docs / 13,985 verified, +4 tokens/doc, Answer: line stays
  OUTSIDE the fence so grading starts where thinking ends).
- sft_trace_train --think-tokens: registers the pair and zeroes the two
  readout rows (anti-trained during pretraining: never a CE target, only
  softmax-denominator pressure). Embedding rows keep pretrained init
  (RMS-normalized on input anyway).
Next SFT round should use: --traces postraining/data/sft_traces_v2_think.parquet --think-tokens

### RL probe 1024 diagnosis (mid-run) + fix arms 1025/1026

Operator observation confirmed in metrics: exact accuracy CLIMBS (0.012 ->
0.06-0.12 on training rollouts — RL is learning 3-6x faster from the SFT
base) but diversity drains fast (wg reward std 0.044 -> ~0.015-0.02) and
answers re-shorten. Cause is the same as 1015's post-mortem, now moving
faster because gradients are stronger: 87-97% of reward mass is PARTIAL
credit (nearby_reward_max 0.1 dense component), which is optimizable by
fast answer-guessing without reasoning — RL amplifies the terse SFT style.
No token-entropy knob exists in the trainer, and none is wanted: fix the
objective, not the symptom.

Fix arms (single-change ablations, queued behind 1024's finish):
- 1025 sft_rl_binary_4k: --nearby-reward-max 0 (binary exact reward,
  R1-style). Isolates partial credit as the collapse driver.
- 1026 sft_rl_binary_gsm8k_4k: binary AND --math-data
  postraining/data/gsm8k_rl_prompts.parquet (7,217 GSM8K train prompts in
  the SFT episode framing, SFT-holdout excluded — built by
  prepare_gsm8k_rl_prompts.py). Matches the prompt distribution to the
  band where exact success actually varies (the SFT gate's 12.5% mixed),
  so binary reward has within-group signal.
Readout: does wg std SUSTAIN while exact accuracy climbs; emitted length
should stay in the reasoning regime (100-400 tokens), not re-pin at ~20.

## 2026-07-31: 1024 sft_rl_probe_4k final — baseline arm numbers

Job succeeded at 4000 steps. This run IS the baseline arm for the
1025/1026 comparison. Full collapse trajectory, quantified:

- Emitted length (actions_per_trajectory, training rollouts):
  391 (step 4, the SFT policy) -> 13 (step 404) -> 8.1 (step 804) ->
  pinned at ~8 for the remaining 3200 steps. Collapse is complete in
  under 800 steps and never recovers.
- Partial reward fraction: 0.88-0.99 throughout — reward mass is almost
  entirely the dense nearby component, confirming the diagnosis.
- Exact accuracy (rollouts): noisy 0.01-0.12 band, no sustained trend
  once length pins; final 0.078.
- Within-group reward std: 0.044 (start) -> 0.011-0.020 (mid) -> 0.001
  at step 4000; exact-reward wg std literally 0.0 at the end (every
  group fully agrees -> zero GRPO-style signal).
- Bench avg@1152: 0.007 (step 0) -> peak 0.031 (step 2500) -> DECLINING
  tail 0.0295 (3500) -> 0.0252 (3752) -> 0.0234 (4000). AIME 0.0000
  throughout. Bench emitted mean re-pins from 399 tokens at step 0 to
  ~8 by step 1000.
- Net vs 1015 (RL from pretrained base): bench roughly doubled
  (1.3-1.9% -> 2.3-3.1% band), so the SFT base is worth keeping, but
  the partial-credit objective still eats the reasoning style within
  ~800 steps.

Pass bar for 1025/1026 stays as pre-registered above; the sharpest
single signal is actions_per_trajectory at steps 400-1000 (baseline:
13 -> 8).

## 2026-07-31: 1025 refutes binary reward as the fix; round 3 design

1025 (binary exact reward, DAPO prompts) at step 1100 of 4000:
- Collapse happened ANYWAY, faster: 382 tokens -> 15.8 by step 112,
  pinned ~5-8 by step 436. Partial credit was an accelerant, not the
  root cause.
- The decisive number: at ~5 emitted tokens exact acc reached 0.081;
  at 382 tokens (the SFT reasoning style) it was 0.015. At temp 1.0
  the policy's own long reasoning DERAILS more than it helps —
  terse guessing is genuinely reward-optimal, so any pure-accuracy
  objective will prune reasoning. RL is working; the objective is
  mis-specified for what we want.
- Then destabilized: step ~1084 length blew up to 519 -> 777, bench
  acc 0.0 with emitted mean 917 and ended_fraction 0.11 (policy
  stopped terminating). Cancelled 1025 at 1100; cancelled 1026
  (same v1 base, superseded).

Root causes now on the SFT side too: sft_traces_v1 contains a strong
terse mode (opus46_10k half, p50 36 completion tokens, 34% < 30) —
RL collapses INTO a mode SFT deliberately taught. And nothing marks
thinking as structurally distinct, so there is no way to reward its
presence (operator: "isn't thinking").

Round 3 (queued):
- 1027 sft_think_v2: SFT on sft_traces_v2_think.parquet with
  --think-tokens (fence rows 50257/50258 registered + trained) and
  --min-completion-tokens 30 (drops the degenerate-terse tail; the
  attractor is removed from the base policy, holdout CE still
  measured on the unfiltered corpus distribution).
- 1028 sft2_rl_think_gsm8k_4k: RL from that base with binary reward
  (--nearby-reward-max 0), GSM8K prompts (distribution matching,
  kept from the 1026 design), and NEW --think-tokens reward gating:
  score_math_rollout now zeroes ALL reward unless the emitted stream
  contains exactly one <think> then one </think> before the stop
  token. The bare-guess attractor is worth 0 even when correct, so
  the reward-optimal policy must keep the thinking channel; what it
  does inside the fence is then shaped by accuracy alone.
- Trainer support added: --think-tokens flag (vapo/config.py),
  tokenizer plumbed via load_posttraining_tokenizer(think_tokens=),
  startup check that the base checkpoint's SFT metadata has
  think_tokens=True (else zero-reward run), think_format_fraction
  rollout telemetry, think_format_ok gating in score_math_rollout.
  Tests: test_latent_rollout.py (fence gating, diagnostics fraction,
  tokenizer loader).

1027 sft_think_v2 gate (2026-07-31): PASSED. accuracy 0.0146 (v1
0.0166), mixed 10.9% (v1 12.5%), wg std 0.037 (v1 0.042), holdout CE
0.782 (v1 0.797), emitted mean 276 / p95 768 (at the cap — fence
overhead + terse filter lengthen completions; a few truncations).
Fence compliance from gate_transcripts token ids: 61/64 well-formed
(<think> then </think> before stop), 64/64 opened; the 3 failures are
budget truncations. The RL format gate therefore starts ~95%
satisfied — no zero-reward desert.

Red-team review of the think-gate diff (findings, all fixed):
- HIGH: think_format_ok accepted an EMPTY <think></think> and a fence
  emitted AFTER the answer — the bare-guess attractor survived at a
  2-token cost. Now: exactly one open+close, >=1 token between them,
  and no "Answer:" field decoded before the close.
- MED: --think-tokens + --reasoning-mode none would silently zero all
  reward (none mode budgets only the answer) — parser error now.
- MED: --resume / --actor-critic-init could flip --think-tokens and
  keep a critic fit to the other return distribution — both refused.
- LOW: think-SFT base run WITHOUT --think-tokens silently dropped
  fence ids in decode — loud startup warning (legit as ablation).
- LOW: think_format_fraction never reached TensorBoard — whitelisted,
  plus new think_gate_zeroed_correct_fraction (verifier-correct rows
  the gate zeroed; scorer stashes the count on the batch) to tell a
  signal-destroying gate from a worsening policy.
- Provenance guard moved to checkpoint load (fails in seconds), and
  the RL manifest now records the SFT lineage.

## 2026-07-31: 1028 crash — step-0 bench eval NaN (compile-dependent)

1028 died at bench_eval(0): torch.multinomial device assert
"probability tensor contains inf/nan". Investigation (scratchpad
repro_bench_nan.py, jobs 1029-1036):
- Checkpoint weights all finite; fence embed rows ~24 norm (near
  init), readout fence rows ~1.2, untrained band ~35 (anti-trained).
- Think tokenizer encodes all 144 bench prompts byte-identically to
  the plain tokenizer; no prompt contains a fence string.
- EAGER full bench eval passes on BOTH v2 (acc 0.0122, emitted 422)
  and v1 (acc 0.0113, emitted 402) checkpoints under the exact
  trainer eval parameters (samples 8, 1024/1024, top_p 0.7, seed
  1337, tail batch 16). The only remaining delta vs the trainer is
  the compiled step core (max-autotune-no-cudagraphs, dynamic=True).
- COMPILED full bench eval ALSO passes on both checkpoints in
  isolation (1035 v2: acc 0.0087, emitted 432; 1036 v1: acc 0.0069,
  emitted 394). All four {v1,v2} x {eager,compiled} arms are clean.
Remaining un-reproduced delta: in-process history — in 1028 the same
dynamic compiled artifact specialized on AIME shapes and made its
max-autotune benchmark choices before bench ran; autotune picks are
benchmark-noise-dependent. Operator directive: eager eval is NOT
acceptable (maximum performance) — fix at the compile level, never
by degrading eval.
- 1037 (identical rerun): NaN RECURRED at the same site (bench after
  AIME passes) — deterministic in the trainer, so not an autotune
  lottery.
- 1038 (exact-state repro: AIME 960 rollouts then bench through ONE
  shared compiled artifact, trainer parity on requires_grad/eval):
  STILL PASSES. Eval sequence alone is not sufficient.
- Realization: every repro ran under CUDA_LAUNCH_BLOCKING=1 (set for
  stack fidelity); the trainer runs without it. If this is a launch-
  order RACE (async kernel reading a buffer mid-write), blocking
  hides it — explaining every pass/fail split observed so far.
- Race test results: 1039 (repro, aime-first, compiled, NO launch
  blocking) passes. 1040 (THE TRAINER, identical failing config,
  WITH launch blocking, 60 diag steps) PASSES step-0 bench (0.0078)
  and completes — the deterministic crash disappears under
  serialized launches. RACE CONFIRMED: an async launch-order hazard
  in the compiled eval path; the assert's reported stack (scatter at
  latent_rollout.py:1110) is a sync point, not the source — the bad
  values reach torch.multinomial's probs, i.e. NaN logits out of the
  step path at execution time.
- v2-sensitivity note: same trainer+kernels pass AIME (960 traj,
  chunk 32) and crash only in bench (144x8, chunk 8, different
  compaction/tail geometry), and only with the v2 checkpoint —
  consistent with a narrow timing window that generation/termination
  patterns steer into.
- Standalone repro escalation, ALL PASS (no NaN): 1041 full step-0
  sequence (BPB guard fwd + AIME + bench, shared artifact, no CLB;
  bpb 1.4566 matches trainer bit-for-bit); 1042 + transcript capture
  (the last call-level delta); 1043 + TF32/matmul-precision parity
  (this DID change kernels — aime 0.0000->0.0010, emitted 401->428 —
  the earlier repros were exercising different compiled code, now
  matched). Standalone reproduction exhausted.
- Bisection moved INTO the trainer (deterministic 2/2 crash there),
  60-step no-CLB diag runs: 1044 --aime-every 0 (does the crash need
  AIME-first?), 1045 --bench-max-rows 32 (small-bench crash would
  give a fast platform for compute-sanitizer initcheck/racecheck).
  Trainer-only ingredients left: resident critic+optimizers
  (allocator layout), TensorBoard writer thread, profiler artifact
  wrapper/CUDA-event phases, logger.
- Trainer bisection results: 1044 (--aime-every 0) PASSES bench
  (0.0095); 1045 (--bench-max-rows 32, with AIME) PASSES (0.0117).
  The window needs AIME-first AND the full 144-row bench AND
  trainer-resident context AND async launches. Crash matrix so far:
  only {trainer, AIME, full bench, no CLB} fails — 2/2 determinism
  there, 0/9 anywhere else.
- Switched tools: compute-sanitizer initcheck (1046) and memcheck
  (1047) over the standalone eval (16 bench rows, compiled) —
  uninitialized/OOB reads are detected value-independently, so a
  passing run can still expose the bad read the race consumes.

Honest uncertainty: the fence gate forces the CHANNEL to exist, not
useful content in it — the policy can learn 2-token thinks. If GSM8K
accuracy from short thinks beats long thinks, length will still
shrink inside the fence; making thinking PAY is the hidden-carry
research bet, and think_format_fraction + emitted length + wg std
will show which way it goes. Watch: think_format_fraction (should
start ~1.0 from SFT and stay), actions_per_trajectory (reasoning
regime ~100-400 vs re-pin), within_group_reward_std sustainment.

## 2026-08-01: NaN investigation closed (unreproduced); round 3 verdict; round 4 design

NaN closure:
- initcheck (1046, 16 bench rows compiled) flagged exactly one site:
  chunk_gla_fwd_kernel_o at fla/ops/gla/chunk.py:429 reading the
  uninitialized upper-triangle blocks of Aqk (torch.empty at
  fla/ops/kda/chunk_intra.py:817; the intra kernels only write the
  diagonal + lower blocks). VALUE-SAFE: line 430 tl.where(m_s, b_A, 0)
  selects the garbage away before the dot — a select, not a multiply,
  so NaN cannot propagate. Known FLA design pattern, not the bug.
- Found a real per-process nondeterminism source while auditing: this
  FLA install ships NO config dir (fla/configs missing), so every
  autotuned kernel (fwd_o 9 configs, fwd_A 8-12, kda intra 3-12)
  falls back to LIVE Triton autotune — winners picked by noisy timing
  per process. The trainer can run different kernel variants than any
  standalone repro, and CUDA_LAUNCH_BLOCKING perturbs the choice.
  Fits the whole crash matrix, but unproven: the discriminating
  experiment (TRITON_PRINT_AUTOTUNING diff between failing trainer
  and passing repro) never got a failing side — job 1048, the exact
  1037 command, ran the FULL 4000 steps cleanly. Crash record now
  2 fail / 1 pass on identical configs; sanitizer jobs cancelled.
- Status: dormant, mechanism unresolved. Mitigation ready if it
  recurs: pin FLA kernel configs via FLA_CONFIG_DIR (JSON with
  default_config per kernel) — deterministic kernel choice at full
  performance, no eager anywhere. Eval stayed compiled throughout.

Round 3 verdict (sft2_rl_think_gsm8k_4k, job 1048, 4000 steps):
- The format gate did NOT prevent collapse; it changed the target.
  actions/trajectory 295 -> 24 by step 444 -> ~15 for the rest;
  emitted p95 = 15 tokens; think_format_fraction 0.83 -> ~1.0.
  The policy learned a minimal well-formed <think>X</think> skeleton
  around the same terse guess. gate_zeroed_correct ~0 all run — the
  gate ate nothing because compliance is 2 tokens cheap.
- Train-prompt exact_accuracy 0.159 at 4000, but bench DECLINED
  0.0095 -> ~0.005 and wg reward std fell to 0.006: overfit terse
  guessing on the train distribution, negative transfer.
- Conclusion matches rounds 1-2: any reward the policy can reach
  without paying compute, it will reach without paying compute.

Round 4: think-length floor (--think-min-tokens).
- Gate now requires >= K tokens INSIDE the fence (default 1 = old
  behavior; validation refuses K>1 without --think-tokens). Even
  low-quality filler tokens run sequential latent compute through the
  hidden-carry channel — the floor mandates the compute budget and
  RL decides how to spend it. This is the cleanest direct test of
  the hidden-carry bet: does forced sequential compute beat the
  15-token attractor at equal steps? Control = round-3 run.
- New telemetry: behavior/think_tokens_mean (inner tokens over
  structurally intact fences, stop-cut visible slice) — a mean
  pinned at K says the policy pays exactly the mandated budget.
- Arm: --think-min-tokens 64 (SFT think mean 276, p95 768, so the
  floor is comfortably satisfied at init; 64 forces ~4x the collapsed
  budget without demanding SFT-length essays).

Red-team pass on the floor gate (all fixed, 418 tests green):
- HIGH: gate's pre-close check was a case-sensitive literal "Answer:"
  while the verifier greps (?i)answer\s*: — "answer: 42" before an
  all-filler fence collected full reward (confirmed live against the
  real tokenizer+verifier). Gate now uses the verifier's own pattern
  and checks the position of the LAST match (the graded one) against
  the close; this also stops zeroing correct rows over instruction
  echoes inside the fence.
- Resume/--actor-critic-init now refuse a changed --think-min-tokens
  (same critic-return-distribution argument as the think_tokens guard).
- Startup validation: floor + fences + answer must fit the training
  rollout budget (an oversized floor zeroes every reward by truncation
  while the gate-zeroed alarm stays at 0).
- think_tokens_mean now pooled sum/count across groups (mean-of-means
  biased low when groups lack intact fences) and omitted when there is
  nothing to measure; think_gate_zeroed_correct promoted to a real
  batch field (None = never scored) so device moves can't silently
  reset the alarm; think_format_fraction now mirrors the scorer
  (unterminated rows are not compliant).
- inspect_critic_values + carry_ablation_eval reconstruct the gate
  from saved args (they compared gate-trained critics against ungated
  rewards and decoded fence ids away silently).

## 2026-08-01: Round-5 direction (user decisions)

- NO decoder-side thinking constraints (budget forcing rejected): the
  model must OUTPUT its thinking; enforcement stays on the training
  side (SFT + reward), not the sampler.
- Distillation is the big lever, sourced from Hugging Face trace
  datasets — NOT API self-distillation. Teacher quality bar: nothing
  below Claude Opus 4.6 (Kimi K3 traces explicitly welcome). Many HF
  trace sets are slop/mislabeled — provenance vetting required; all
  R1/QwQ/gpt-oss-era corpora (OpenR1-Math, AM-1.4M, Synthetic-1,
  OpenMathReasoning) are below the bar and excluded.
- Add <answer></answer> special tokens (GPT-2 slack 50259/50260)
  alongside the think fence; requires a fresh SFT pass. Makes the
  reward gate purely structural on token ids (deletes the decoded-text
  regex position check and its bypass class).
- Focus: dense token-level supervision from good reasoning traces
  across varied tasks including math (K3 recipe insight: dense
  per-token signal beats sparse outcome reward for weak policies).

## 2026-08-01: HF trace-corpus survey verdict (scout + independent spot-checks)

- The frontier-teacher (Opus-4.6+) trace pool on HF is ~38k rows TOTAL,
  of which <3k is math and only 810 rows are ground-truth-verified
  (bevangelista/AIME_2000_2026_Kimi_K3 — genuine K3, separated
  gen_reasoning/gen_answer, 100% answer-verified vs official AIME key,
  but median 1,619 reasoning tokens and AIME difficulty: wrong band for
  the 64M student and over the 1,024-token budget). The 50k-500k corpus
  the round-5 plan assumed DOES NOT EXIST on HF today.
- Usable-if-supplementary pool (~10-15k rows, mostly non-math):
  TeichAI/lordx64-claude-opus-4.7-max-cleaned (4,807; real Batch-API
  extended thinking, separated fields), Crownelius GPT-5.6 Sol/Luna
  (15,353; agentic coding, tool-ID-fingerprint verified),
  greghavens/kimi-k3-coding-and-debugging-traces (3,956; verified K3,
  coding only), TeichAI small sets (~5.5k). Borderline:
  Jackrong/DeepSeek-V4-Distill-8000x (7,716; teacher is V4-FLASH).
- CRITICAL: our OWN current SFT corpus (prepare_sft_traces.py sources
  adapt_opus4647 = angrygiraffe, adapt_trace_inversion = Jackrong
  TraceInversion, adapt_opus46_10k = Roman1111111) is built from three
  sets the survey demolished: angrygiraffe's CoT is admitted-synthetic
  (written into the response, not real thinking) and 4x-duplicated;
  TraceInversion traces are generated by a Qwen3-4B answer-conditioned
  rationalizer; Roman1111111 ships NO reasoning traces at all (137-188
  char answers; ~12% of advertised token volume). Final answers were
  verified by our pipeline, but the reasoning styles the sft2 models
  learned are fake/sub-bar CoT. Spot-checked independently via HF
  datasets-server (row counts + schemas confirmed scout's claims).
- No teacher top-k logit datasets exist (structural: frontier APIs
  don't expose logprobs). True token-level vocab distillation would
  require serving an open-weight teacher (K3 = 2.8T/104B active).
- Contamination canary: truncated prompt "Your solution must read
  input from standard input (input())..." with no problem body marks
  the broken lordx64 lineage; grep any candidate corpus for it.
- Decision pending (user): strict bar => insufficient math volume.
  Options: (a) strict + supplement-only, (b) selectively relax bar
  (V4-Flash), (c) self-generate math core via cheap K3 API using the
  bevangelista recipe (effort escalation, stop-on-first-correct,
  verify vs ground truth, <=512-token traces on GSM8K/MATH L1-3) —
  previously rejected by user, resurfaced with cost evidence (~$25-60
  for ~20k short verified traces at K3 API rates).
- sft_trace_train.py: register_think_tokens generalized to
  register_special_tokens (zeroes readout rows for all registered
  fences); new --answer-tokens flag (requires --think-tokens).

## 2026-08-01: Round-4 verdict (run 1050, think-floor 64) — catastrophic late collapse

- Through step ~2940 the floor did its job: fence compliance ~0.9+,
  think ~100 tokens, reward oscillating 0.01-0.11, bench policy
  accuracy peaked 0.0182 (steps 1500/3000) vs 0.0061 at step 0.
- Steps 2948-2996: format compliance crashed 0.60 -> 0.00 in ~50 steps
  during a reward drought; the fence habit broke structurally, so the
  gate zeroed ALL reward from ~step 3036 onward.
- Mechanism of the death spiral (from train dashboard): the critic was
  stale — V_pred +0.02 while targets had already gone to 0 — so every
  trajectory got a uniformly negative advantage (mean -4.5e-3),
  punishing the policy's CURRENT behavior including termination.
  Expected length ~ 1/p_stop, so a small downward nudge on the stop
  logit exploded actions 100 -> 1000 within ~90 steps (3060-3150);
  ended_fraction hit 0.00 and reward became identically zero — an
  absorbing desert with no recovery gradient for the final ~950 steps.
- Endpoint: bench and AIME policy accuracy 0.0000; teacher-forced
  val_bpb degraded 1.4566 -> 1.6943; final policy is a degenerate
  repetition loop ("The dog will *not* be used for the dog." to the
  1024-token budget) on arithmetic prompts.
- Verdict vs round 3: WORSE. Round 3 (gate only) collapsed to a
  working terse guesser (acc ~0.08 train-dist); round 4 (gate + floor)
  destroyed the policy outright. Together they bracket the diagnosis:
  pure-accuracy RL prunes the thinking channel; format-gated RL on a
  weak SFT prior is unstable because reward droughts + stale critic
  yield anti-termination gradients. RL cannot conjure reasoning this
  prior does not contain — consistent with the corpus finding that the
  sft2 prior was trained on fake/sub-bar CoT.
- Reinforces round-5 plan: dense SFT distillation from genuine traces
  FIRST; RL only from a strong prior, with the structural answer-fence
  gate. A critic-staleness/zero-reward guard (freeze policy updates
  when a rollout window's reward is all-zero) is worth considering in
  the round-5 trainer, as a diagnosis-backed fix, not a patch.
- Run preserved at postraining/runs/sft2_rl_think64_gsm8k_4k (final
  checkpoint only; the 0.0182 mid-run policy was not separately saved).

## 2026-08-01: Structural answer-fence gate (round-5 machinery) + red-team round 2

- Implemented --answer-fence across the stack: purely token-id reward
  gate (structural_format_ok), fenced-span grading, eval preference for
  the fenced span with plain-text fallback, SFT/prepare corpus support
  (--answer-tags => sft_traces_v3_answer.parquet), provenance/resume/
  init guards, and fence-aware diagnostics tools.
- Red-team round 2 found 3 HIGH (all fixed): (1) order-only structure
  admitted a plain-text guess BEFORE <think> — gate now ANCHORED:
  <think> must open the completion and </answer> must sit immediately
  before the stop token; SFT compose emits exactly that shape. (2) the
  flag-trusting provenance guard — SFT now MEASURES the anchored-shape
  fraction over corpus token ids (answer_fence_document_fraction, >=99%
  enforced at SFT arg-time and RL load-time), and RL prompts are
  rewritten from the "Answer:" template sentences to the fence contract
  (byte-identical to the SFT suffix, fail-closed if absent). (3) the
  gate-zeroed-correct alarm was blind to fence-native collapse (decode
  strips broken fences) — counterfactual now uses a relaxed span scan
  before the plain parse. Also fixed: verify window cliff (>292-char
  fenced values graded [INVALID]; verify_answer gained window=None for
  self-constructed reframes), init guards now cover --actor-init and
  --curriculum-init (not just --actor-critic-init), aggregate key
  intersection, sample_latent + dapo_hit_rate_probe fence-awareness.
- 426 tests pass (new: anchored-gate violations incl. the two red-team
  layouts, relaxed-extraction, window cliff, corpus-shape measurement,
  prompt rewrite, heterogeneous aggregation, config validation).

## 2026-08-01: Red-team round-2 closure — 8/9 confirmed fixed, 6 new findings, all addressed

Verification pass reproduced the suite and probed the fixes executably.
Verdicts: 8 CONFIRMED-FIXED, 1 PARTIAL (sample_latent's --math-data path
still graded plain-text). New findings and what was done:

- A (MED, fixed): the prompt rewrite counted replacements, so 20 unique
  dapo-math-17k rows kept an unlisted Chinese instruction demanding
  "Answer: \boxed{...}" — contradicting the fence contract on prompts the
  trainer believed it had framed. The Chinese template is now in the
  rewrite table (ANSWER_INSTRUCTION_REWRITES), and the rewrite is
  genuinely fail-closed: any surviving "Answer:" in a rewritten prompt
  raises. A new test sweeps ALL real RL/eval parquets through the rewrite
  and asserts nothing survives (the synthetic fixture was built from the
  table itself, so it could only ever pass).
- B (LOW-MED, fixed): SFT suffix and RL rewrite reminder are duplicated
  literals in two modules with nothing binding them — a one-line parity
  test now pins INSTRUCTION_SUFFIX_ANSWER == "\n\n" +
  ANSWER_FENCE_INSTRUCTIONS[1] (and the plain-text pair).
- C (MED, fixed): all four inspection tools (sample_latent,
  dapo_hit_rate_probe, inspect_critic_values, carry_ablation_eval) graded
  fence-aware but SAMPLED under the unrewritten plain-text prompt —
  off-distribution policy, systematically wrong reconstructed rewards.
  Each now applies rewrite_prompts_for_answer_fence gated on the saved
  answer_fence flag. The PARTIAL (sample_latent math-path grading) fixed
  in the same pass.
- D (LOW, fixed): the corpus measurement validated shape, never span
  length — an anchored-but-terse corpus measures 1.0 yet earns all-zero
  reward at a floor above its spans. SFT now records
  think_span_token_percentiles {min, p1, p50} in provenance; RL startup
  refuses a --think-min-tokens floor above the corpus p50, warns above
  p1, warns (not blocks) on pre-round-5 provenance without the stats.
- E (LOW, accepted as monitoring note): the floor is a decode-step
  budget, not an information budget — unregistered padded-vocab ids
  50261-50303 can pay it while decoding to replacement characters.
  Signature to watch in round 5: think_tokens_mean pinned exactly at the
  floor + unreadable think spans in captured transcripts while the gate
  reports full compliance.
- F (LOW, fixed): the fence instruction described only the answer half of
  the anchored contract. Both instruction sentences (and the SFT suffix,
  in lockstep) now state "start with <think>" as well.
- Relaxed-counterfactual caveat (documented + pinned by test): first-span
  extraction means a scratch <answer> inside the think span wins over the
  real answer (over-count) and a wrong first span hides a right second
  one (under-count) — read the gate-zeroed-correct alarm as approximate.
- Training-dynamics note (from the round-2 pass): the instruction suffix
  itself contains the literal "<answer></answer>", so every prompt
  carries an EMPTY fence pair as special ids in-context — a pattern the
  gate rejects. Doesn't reach the gate (prompt slice excluded) but worth
  remembering when reading early-round format-compliance curves.
- 430 tests pass. Separator emulation in the corpus measurement was
  verified faithful against real pack_rows output (every completion-final
  </answer> is followed by a CE-targeted stop), and fence tokens cannot
  pay the think floor (single-pair rule + ordering reject every
  arrangement).

## 2026-08-01: Red-team round 3 — CLOSED. Gate machinery done; two tightenings applied

Round-3 verification: every round-2 item CONFIRMED-FIXED, nothing
STILL-OPEN. The red-team swept NINE real parquets (incl.
deepmind-interpolate-rl-full's 179,856 unique rows) through the rewrite:
zero rows retain any case-insensitive Answer:-field match, no problem
statement quotes "Answer:" (so the fail-closed post-condition cannot
false-positive at startup), the Chinese replacement introduces no
plain-text demand, and no init/resume/rollout path reaches rollouts with
the think floor unchecked. Applied its two cheap residual tightenings
immediately:

- Post-condition now uses the verifier's own case-insensitive pattern
  (ANSWER_FIELD_DEMAND, mirroring core.extract_final_answer) instead of
  a case-sensitive substring — an unlisted "answer:" template would have
  slipped a case-sensitive scan yet still parsed as a field at grading.
- The real-data sweep test now covers all nine parquets, notably
  deepmind-interpolate-easy (the DEFAULT --bench-data — a template
  landing there hits every run), and asserts with the same pattern.

Accepted as recorded notes (no code change):

- Prompt-side fence ids are now deliberate: finding F's instruction
  wording tokenizes to an unpaired <think> plus an EMPTY
  <answer></answer> pair inside every prompt (single_fence_span over the
  prompt alone returns a span). Harmless today — every consumer slices
  ids[prompt_length:] or uses emitted_token_rows — but any FUTURE code
  running fence extraction over prompt+completion will silently pick up
  that contaminating pair. Slice first.
- Floor-vs-policy gap: --actor-init from a run whose POLICY collapsed to
  short think spans passes the corpus check (it validates the base
  checkpoint's SFT distribution, not the loaded policy's behavior); the
  think_min_tokens equality guard on the init path narrows this to
  near-zero. Practical detector: think_tokens_mean in the first steps.
- Preexisting, not introduced here: resume does not pin base-checkpoint
  identity, so resuming with a MORE permissive base (longer-span corpus)
  passes silently; both restrictive directions fail closed.

430 tests pass. The answer-fence gate stack is closed out; next
actionable work is the corpus decision (user) and then the round-5
SFT -> RL sequence on a rebuilt trace corpus.

## 2026-08-01: Corpus decision (user) — source existing K3 traces from HF, no API generation

User: "there's definitely quality k3 traces out there for you to use. No
need for us to generate our own." So: no API spend, no self-generation
pipeline. [SUPERSEDED same day — see next section: the K3-exhaustive
hunt came back negative for math, and the user then approved BOTH
relaxed-bar HF sourcing AND self-generation.]
The trace-corpus-scout is re-tasked with a K3-EXHAUSTIVE hunt
(the first survey was breadth-first across teachers): all K3 naming
permutations incl. Chinese-community names, recency-sorted (K3 distill
sets are recent), mixed-teacher datasets with per-row generator tags,
and README-only teacher attribution. Quality bar unchanged — genuine K3
provenance (fraud canaries from the first survey still apply), math at
GSM8K-to-easy-competition difficulty, separable reasoning/final fields,
verifiable ground truth, reasoning spans mostly under ~1024 GPT-2
tokens, target 10k+ usable rows. bevangelista's 810 verified AIME K3
rows stay as a supplement. When the scout reports: rebuild
prepare_sft_traces adapters (delete the three demolished-source
adapters), verify + decontaminate, compose with --think-tags
--answer-tags, SFT with --think-tokens --answer-fence, then round-5 RL
with --answer-fence.

## 2026-08-01: K3-exhaustive hunt NEGATIVE for math; user approves relaxed bar + self-generation; generation pipeline built

- Scout's K3-exhaustive sweep (run twice, independently): the complete
  inventory of genuine K3 math on HF is bevangelista's 810 AIME rows
  (all answer-verified, but olympiad difficulty and only ~330-360 under
  1,024 GPT-2 tokens). Everything else K3 is coding (greghavens'
  moonshiner 3,956 rows, endlessly mirrored — Siddh07ETH 15.7k,
  Accretion, j0no12 are all rehosts/mislabels of it), creative writing,
  TikZ, or quantization logits. The kimi-k3 tag page is a closed set of
  11 datasets (independently confirmed via HF API). K3 is ~3 months old;
  the community math-distillation wave hasn't reached it. The two orgs
  that DID regenerate GSM8K/MATH-500 from K3 (Inferact/RadixArk DSpark)
  never published the data.
- User decision (AskUserQuestion): options 1 AND 3 — relax the teacher
  bar for the HF math core (K2.5/K2.6/GLM-5.x-tier acceptable; trace
  correctness/length/difficulty now weigh more than teacher tier) AND
  self-generation via K3 API is back ON (~$25-60). Scout re-tasked with
  the relaxed-bar hunt (report pending).
- Built postraining/generate_k3_traces.py: verified-trace generation
  over OUR OWN RL problem sources (7,217 GSM8K train + 13,000
  decontaminated deepmind-interpolate-rl = 20,217 problems), so the SFT
  prior lands exactly on the RL distribution. Every trace graded by the
  RL verifier itself (core.verify_answer, per-row answer_style, default
  window — byte-for-byte RL semantics). bevangelista protocol adapted:
  effort escalation low,low,high, stop on first verified; truncation
  retries the same effort with doubled max_tokens (not more effort).
  Stores BOTH the native reasoning channel and the visible solution;
  which becomes the <think> span is a compose-time decision
  (telegraphic-register caveat from the survey).
- Money-safety (red-team: 3 HIGH, 4 MED, 3 LOW — all fixed, closure
  verification pending): budget meter fails CLOSED on usage-less
  responses (ContractError + estimate charge); timeouts/dropped
  connections charge a conservative estimate (provider may have billed);
  429/5xx rejections charge nothing; per-style YieldCanary halts a style
  whose verified yield drops below 20% after 50 resolved (catches the
  deepmind exact-style canonical-form mismatch class: 37% of ground
  truths are non-integer — fractions, option letters, comma lists,
  booleans — so a value-blind EXACT_STYLE_HINT rides the system prompt
  for style=exact); pool fails fast with cancel_futures; resume
  tolerates torn trailing lines; --limit pilots interleave sources.
- Red-team round 2 on the generator: all 10 round-1 findings
  CONFIRMED-FIXED; 5 new findings, all addressed — (NEW-1) exhausted
  records now tagged {style, exhausted: "schedule"} and
  --retry-exhausted regenerates the canary's detection sample after a
  contract fix instead of burning those rows forever; (NEW-2) zero-yield
  early trigger fires the canary at half the sample size (deterministic
  mismatch has exactly zero hits), cutting detection cost ~60%; (NEW-3)
  canary aborts list their styles in the final summary and exit 2 so
  queued runs cannot look clean; (NEW-4) null/junk usage values coerce
  to 0 and land in the charged fail-closed branch instead of crashing
  uncharged; (NEW-5) pilots below canary coverage print a warning.
- Reviewer's final pass: everything CONFIRMED-FIXED (verified
  end-to-end), one dry-run nit applied (--dry-run now forwards
  --retry-exhausted into its skip-count preview). Operating note: the
  zero-yield canary trigger has ~1.7% odds of spuriously halting a
  GENUINELY hard style with real ~15% yield (zero hits in the first
  min_resolved//2 draws) — recoverable via --retry-exhausted, but
  remember it if a legitimately hard source ever halts unexpectedly.
- NOT YET LAUNCHED: needs the user's API key (MOONSHOT_API_KEY, or
  --base-url/--effort-key for OpenRouter) and provider prices
  (--price-in-per-mtok/--price-out-per-mtok are deliberately required
  flags). Plan: --limit ~100 pilot first (validates effort-key surface,
  usage reporting, exact-style yield), inspect transcripts + register,
  then full 20k run under --budget-usd.
- 445 tests pass (15 for the generator).

## 2026-08-02: Relaxed-bar corpus report (scout final) — blend chosen

Scout's ranked finalists, all length/provenance/GT numbers computed from
datasets-server statistics over full populations (not cards). Its own
earlier "use the K2.5/GLM million-row math corpora" suggestion was
SELF-CORRECTED: Jackrong GLM-5.1 Math median 82,868 chars/output, K2.5
General-Math median 9,616 tokens — the entire long-form class is out
(same failure as marin OpenThoughts).

Chosen blend (~28k HF rows + K3 generation + gold anchor):
- nvidia/OpenMathInstruct-2, cc-by-4.0, teacher Llama-3.1-405B:
  RESTRICT problem_source to augmented_gsm8k + gsm8k (153,311 rows,
  median 897 chars, 99.7% under 4,082) — the augmented_math 83% majority
  has ~22% ill-posed problems whose expected_answer matches the boxed
  value by CONSTRUCTION (internal consistency, not correctness; will
  pass any boxed-match check). Target ~10k.
- mlfoundations-dev/a1_math_deepmind, NO LICENSE, teacher deepseek-
  reasoner over deepmind/math_dataset TRAIN splits only (YAML-pinned;
  interpolate = TEST is never touched, so no bench contamination). Use
  the deepseek_solution field (median 814 chars), NOT reasoning/
  final_reasoning_trace (5-6k). Repairs needed: bytes-literal unwrap
  (b'...\n' on question/answer in 100/100 rows) + answer verification
  (23.5% of solutions disagree with the gold answer column — teacher
  errors; filter on agreement). ~23.9k usable; take ~8k. ONLY source on
  our RL/bench distribution. License flagged to user.
- HAD653/GSM8K-OpenMath-MathReason-13k, license placeholder, teacher
  gpt-oss-120B: median 349 chars, fixed Problem/Reasoning/Answer
  template, but final_answer is TEACHER-derived. Restrict to rows
  joinable to GSM8K train gold (~57% verbatim train questions) so GT is
  independent. ~6k.
- sxiong/synthetic-math, MIT, GPT-4o problems cross-verified by R1
  answer agreement: filter L1-L3, use solution field (median 741). ~4k
  MATH-style coverage.
- openai/gsm8k socratic train (7,473, MIT, human gold) as style anchor.
- Swap option (not chosen): codelion/gsm8k-synth — mechanically-executed
  GT and 0% overlap vs BOTH gsm8k splits, but calculator-annotation
  register (weak think-span style) + confirmed template redundancy.
- Rejected with measurements: OpenMathReasoning (median 19k chars, 51%
  olympiad), Mixture-of-Thoughts math (median 4,936 tok), orca-math (no
  extractable GT), MetaMathQA (27.8% answer-conditioned FOBAR/SV),
  NuminaMath-CoT (21% olympiad, no answer column), MathInstruct (40%
  programs, 34% multiple-choice), tulu-3 math (no GT), whynlp/gsm8k-aug
  (equation-only), cm00cm K2.7 perfectblend (no GT field, unlabelled
  ~40-50% math, inherits MetaMathQA answer-conditioning, no decontam),
  Accretion reasoning rows (lordx64 K2.6 lineage, unverified, count
  mismatch vs card).
- Method warning: datasets-server /search is token-based (stopwords
  dropped) — overstates phrase counts ~100x; never use it for phrase
  membership.

Next: fetch the five sources locally (postraining/data/relaxed_bar/),
rebuild prepare_sft_traces (delete the three demolished adapters; new
adapters with per-source repair + OUR verifier on every row + 8-gram
decontamination vs GSM8K test / bench sets), then compose --think-tags
--answer-tags alongside the K3-generated core (pipeline ready, awaiting
user API key + prices).

## 2026-08-02: prepare_sft_traces rebuilt on the relaxed-bar blend

Deleted the three fraudulent adapters (opus4647_8k7, opus46_ti9k,
opus46_10k) and their GSM8K-answer-bank verification path. New pipeline:
six adapters over the local relaxed_bar parquets + the (pending) K3
JSONL, every kept row verified with core.verify_answer semantics, then
exact-text + word-8-gram decontamination vs GSM8K test, deepmind-
interpolate-easy, AIME 2024/2026 (exact match matters: bench problems
like "Work out 64339656 - 0." are too short for any n-gram).

Built sft_traces_v3_answer.parquet: 36,286 docs, 9.2M GPT-2 tokens,
p50 219 p99 693 max 3,574 (cap 4,096). Per-source kept: openmath 10,000
(cap-sampled from ~151k verified; 44 wrong, 1,617 contaminated —
augmented rewrites of test problems), a1_deepmind 6,000 (sampled from
16,977 survivors), had653_gold 6,950 (6,815 unjoinable dropped),
sxiong_l13 5,974, socratic 7,362 (94 contaminated). K3 core absent
until generation runs; rebuild with --k3-jsonl after.

Measured corrections to the scout's survey:
- a1_math_deepmind teacher-error rate is ~46%, not 23.5% (hand-checked
  sample: r(1) for r(h)=h^3-h^2+h answered -18; 70/143 probability
  answered 9/20). Survivors 16,977/31,600. Composed final = CANONICAL
  answer string (exact-style RL grading), never the teacher rendering.
- sxiong has 2 degenerate rows (empty answer + empty \boxed{}) that
  PASS the Minerva grader empty-vs-empty (regex captures a trailing
  space); the central empty-field guard drops them (an empty final
  would compose the zero-width <answer></answer> the anchored gate
  rejects).
- deepseek_solution finals need: bytes unwrap, "**Answer:**" statement
  extraction, \( \) / \[ \] delimiter strip, \dfrac->\frac, "x = v"
  split, and a symbolic bridge (latex_to_python -> sympy) for
  Python-syntax truths like -96*a**2 vs LaTeX -96a^2.

Perf lesson (two stalled builds): sympy.simplify AND Expr.equals both
take unbounded rewriting paths on w**(-3058)-scale exponents; with
~14.6k genuinely-wrong rows paying that cost before being dropped the
build ran 26+ min without finishing a1. Replaced with bounded numeric
probing: evalf(50) at two fixed rational points, relative tolerance
1e-30, everything non-Expr/undefined/inconclusive dropped. Full a1
filter: 10s; whole build ~25s.

Tests: postraining/tests/test_prepare_sft_traces.py (15) — synthetic
adapter contracts + real-parquet slices (schema-drift canaries) +
decontamination index. Suite 459 passed / 17 skipped. Red-team review
of the rebuild spawned per standing practice.

## 2026-08-02: prepare rebuild red-team round 1 — FIX-FIRST, all fixed

Findings (agent-verified against the real build) and resolutions:
1. MAJOR: "verified" was a structural no-op for 60.6% of rows —
   openmath-augmented boxed==expected_answer BY CONSTRUCTION (0 drops
   possible), sxiong answer column byte-identical to its own boxed in
   5,995/5,995, had653 claim identical to gold in all joined rows. FIX:
   augmented_gsm8k EXCLUDED outright (138,547 rows — same rationale as
   augmented_math; red-team measured 13.9% decimal finals vs 0.0% in
   genuine rows + hand-confirmed ill-posed problems with integer
   labels). openmath now = problem_source "gsm8k" only, verified
   against OUR local GSM8K gold via the had653 join (61 teacher errors
   dropped, 100% joinable). sxiong kept, documented honestly (its
   independent check is upstream GPT-4o x R1 agreement; ours is only
   an extraction canary).
2. MAJOR (latent): no fence-string guard — a K3 trace containing a
   literal "<think>" would compose a gate-failing document undetected.
   FIX: central screen_candidates() with FENCE_STRINGS guard + a test
   that composed docs pass structural_format_ok.
3. MINOR: empty-truth Minerva pathology now fails in graded_correct
   itself, not just the downstream empty-field guard.
4. MINOR: deepmind-family SFT framing diverged from RL (missing DAPO
   header + FENCE0 prefix). FIX: DEEPMIND_PROMPT_PREFIX prepended to
   the problem column for a1 (and future K3 deepmind keys); byte
   parity with the RL rewrite pinned by test reconstructing real
   rewritten prompts. Trainer needs no change (prompt boundary is
   still problem+suffix).
5. MINOR: Answer:-field blocks inside think spans (260 docs) — strip
   now cuts the trailing marker BLOCK (colon required, 4-line scan,
   loop until stable) across ALL sources: 260 -> 7 residual.
6. No-change finding: 21% of sxiong/a1 finals are non-numeric — that
   IS the canonical register for exact-style deepmind RL grading
   (a1) / deliberate MATH coverage (sxiong); documented.
Clean surfaces confirmed by the agent: anchored gate 36,286/36,286,
byte-identical rebuild determinism, decontamination 0 misses under
stricter-than-pipeline comparison, symbolic_agree 0 false positives
vs a 6-point 80-digit oracle, no gold-join collisions from the
160-char normalize_problem truncation.

Rebuilt corpus: same 36,286 total (openmath cap still met from the
14.5k genuine pool). Suite 464 passed / 17 skipped. Verification pass
by the red-team pending.

## 2026-08-02: red-team verification pass — SHIP; SFT queued

Verification pass results on the rebuilt corpus (sha 26adfebb...):
- F1/F2 live: all 10k openmath rows problem_source=gsm8k, finals equal
  OUR gold 10,000/10,000, filter provably grading (160 rows pass with
  boxed != gold byte-form; 61 wrong dropped independently recomputed).
- Anchored gate 36,286/36,286 including deepmind framing; F5 framing
  verified byte-exact against 4,000 real rewritten deepmind-rl-full
  prompts (not just the 3 pinned rows).
- Strip cascade audit: 5,534 a1 rows changed, hand-read worst cases all
  correct (LaTeX answer-display blocks + post-answer commentary); 0
  rows lost derivation content, 0 emptied. Residual V3 (cascade
  bounded by data not construction) fixed post-verdict with
  MAX_ANSWER_BLOCK_LINES=12 cap + single central application (a1
  in-adapter strip removed); rebuild remains byte-identical (no real
  cut approaches the cap), suite 464 passed.
- Observation V5 (watch): GSM8K-family sources now supply 67% of rows
  over 7,375 distinct train problems (~3.3 traces/problem) —
  augmented_gsm8k was the only GSM8K-band problem-diversity source;
  memorization risk noted, K3 core + a1 + sxiong carry the diversity.
- Observation V6: 2,565 non-numeric finals retained by design (a1
  symbolic = canonical exact-style register, now with matching RL
  framing).

Queued mlq job 1054 sft_v3_answer_hfonly: sft_trace_train --think-tokens
--answer-fence on sft_traces_v3_answer.parquet (HF-only corpus, K3 core
still blocked on user API key + prices). Serves as the HF-only reference
for measuring K3's marginal value when the core lands; check
answer_fence_document_fraction >= 0.99 and think-span percentiles vs the
planned RL floor when it finishes.

## 2026-08-02: SFT on v3 corpus (HF-only) — job 1054 done, e2 ablation queued

sft_v3_answer_hfonly (3 epochs, 872 steps, defaults otherwise):
- answer_fence_document_fraction 1.0; think-span percentiles
  min 10 / p1 33 / p50 131 recorded in checkpoint metadata (RL floor
  guard binds against p50=131).
- Holdout completion CE: trough 0.7479 at step 575 (end of epoch 2),
  jump to ~0.78 at the epoch-3 boundary, final 0.7684 — epoch 3 looks
  net-harmful (V5 concentration: ~3.3 traces/problem). Trainer saves
  FINAL only, no best-holdout tracking.
- Sampling gate (128 prompts x 8 samples, anchored-gate reward):
  accuracy 0.0273, mixed prompts 0.172, within-group reward std 0.0616.
- Queued job 1055 sft_v3_answer_hfonly_e2 (--epochs 2) to A/B the
  overfit question on gate metrics; pick the better checkpoint for
  round-5 RL. Both runs are the HF-only reference for measuring the
  K3 core's marginal value when it lands.

## 2026-08-01: e2 ablation verdict + zero-reward actor-freeze guard

e2 ablation (job 1055, sft_v3_answer_hfonly_e2, --epochs 2): gate
metrics WORSE than the 3-epoch run — accuracy 0.0254 vs 0.0273, mixed
prompts 0.133 vs 0.172. Differences are ~1 sigma, so no strong signal
either way, but nothing supports switching. Verdict: keep
postraining/runs/sft_v3_answer_hfonly/sft_final_model.pt (3 epochs,
job 1054) as the round-5 RL base. The epoch-3 holdout-CE rise did not
translate into worse sampling-gate behavior — the gate metrics are the
RL-relevant criterion.

Zero-reward actor-freeze guard (round-4 postmortem prescription,
implemented before any round-5 launch):
- Mechanism being guarded against: all-zero-reward pool + stale critic
  (V_pred ~ +0.02 vs zero targets) -> uniformly negative advantages ->
  anti-termination gradient -> expected length ~ 1/p_stop explosion ->
  absorbing zero-reward desert (see round-4 postmortem above).
- Fix (revised after red-team FIX-FIRST): the freeze decision is per
  optimizer MINIBATCH, not per pool — the harmful unit is an update
  whose every trajectory scored zero (rewards are non-negative: 1.0
  exact, [0, 0.1] nearby-numeric partial, else 0 — so mean==0 iff all
  zero), and during the descent into the desert the pool mean is
  barely-nonzero while up to 3 of 4 minibatches are already pure
  desert; a pool-level gate would only engage after full absorption.
  Gate: training_update's metrics["reward"] == 0.0 skips
  step_optimizers("actor") for that minibatch. The critic still steps,
  so value predictions catch down to the zero targets — that is what
  kills the stale-critic advantage bias. Actor forward/backward still
  runs (behavior-age-0 clip canary, grad norms, non-finite checks stay
  uniform); grads are zeroed as usual. Skipping the optimizer STEP is
  required rather than relying on small grads: AdamW (weight decay 0
  per ADAMW_ALGORITHM_SCHEMA) still moves weights from stale momentum
  on a zero-signal step. This is ~unbiased when the critic is
  calibrated (V ~ 0 on desert prompts -> advantages ~ 0 -> the skipped
  update was ~0 anyway) and protective exactly when the critic is
  stale — zero-reward trajectories in MIXED minibatches still
  contribute their contrastive negative-advantage signal.
- Consecutive-desert stop (red-team major 2): the freeze can latch —
  with all reward structurally gated, a collapsed policy produces
  all-zero pools forever and the run would burn its whole budget on
  signal-free rollouts. A session-local consecutive all-zero-POOL
  streak (reset on any rewarded pool) triggers a pool-boundary stop
  (final checkpoint saved by the loop tail) after
  --zero-reward-stop-pools pools (default 8 = 8192 consecutive
  zero-reward trajectories; the SFT base's ~2.7% gate accuracy makes
  one all-zero 1024-trajectory pool a ~e^-28 event). Applies
  regardless of the freeze flag; 0 disables.
- Flags: --zero-reward-actor-freeze (BooleanOptionalAction, default
  ON), --zero-reward-stop-pools (int, default 8, >=0 validated). No
  EXECUTION_SCHEMA rev: optimizer stepping policy only, not
  scheduler-visible rollout/replay semantics; resume across the flags
  is safe in both directions.
- NaN fails closed: a non-finite pool reward_mean raises instead of
  silently disabling the guard (reward_mean == 0.0 is False for NaN).
- Telemetry: rollout log field zero_reward_actor_frozen (pool-level),
  per-pool print naming the affected step range (rollout-step
  convention, red-team nit 6), per-train-step tensorboard series
  guard/zero_reward_actor_frozen and cumulative
  guard/zero_reward_frozen_updates_total (distinguishes requested
  steps from actual actor updates in the next postmortem, red-team
  minor 4).
- Declined (red-team minor 5): eliding age>0 clip/KL rows on frozen
  steps — the guard flag series disambiguates the zero readings, and
  a frozen step genuinely producing zero drift is itself the canary
  that the freeze works; a gap would hide that confirmation.
- Tests: flag defaults/negation + stop-pools validation + reward
  non-negativity pin (test_latent_rollout.py
  test_zero_reward_actor_freeze_flag_defaults_on,
  test_nearby_numeric_reward_is_nonnegative).
- Verification-pass fix (red-team round 2 NEW-ISSUE, MAJOR): the
  desert stop made a previously-unreachable tail bug reachable —
  under --consume-all-prompts any early break hit the
  dataset-exhaustion RuntimeError before the terminal save_checkpoint
  (the --max-train-hours break was shielded only by validate_args
  refusing that flag combo). Fixed with a stopped_at_pool_boundary
  flag set on both break paths; the exhaustion mismatch downgrades to
  a printed truncation warning and the tail (final checkpoint, aime
  eval, closers) runs normally. Session-local streak intentionally
  grants a fresh stop budget on resume (documented at the
  initializer).

## 2026-08-01: Round-5 RL launched (job 1056 sft3_rl_answer33_gsm8k_4k)

Guard shipped (red-team round 3: SHIP, all findings closed). Launch is
a minimal-diff arm against round-4 (job 1050) as control:
- SAME: --reasoning-mode latent, --steps 4000, --nearby-reward-max 0
  (binary exact; partial credit proven the round-2 collapse
  accelerant), --math-data postraining/data/gsm8k_rl_prompts.parquet.
- CHANGED: base = postraining/runs/sft_v3_answer_hfonly/
  sft_final_model.pt (v3 verified-trace corpus, 3 epochs, gate acc
  0.0273 / mixed 0.172); --answer-fence (structural token-id gate,
  byte-exact SFT<->RL framing via rewrite_prompts_for_answer_fence);
  --think-min-tokens 33 (= corpus p1, no floor-guard warning; round 4
  used 64 against an SFT think mean of 276 — 33 sits inside 99% of
  the new prior's span support while still pricing the ~15-token
  skeleton attractor out); zero-reward guards default ON
  (minibatch actor freeze + 8-pool desert stop).
- Watch: guard/zero_reward_actor_frozen and the desert-stop print;
  behavior/think_tokens_mean pinning at 33 = paying only the floor;
  fence compliance around reward droughts (round-4 died at
  steps 2948-2996); bench avg vs round-4 peak 0.0182.

## 2026-08-02: Round-5 verdict (job 1056 sft3_rl_answer33_gsm8k_4k) — first non-collapsing RL round

Run completed all 4000 steps. Headline: NO collapse — the first RL arm
to finish with structure intact and bench above its SFT start.
- Bench avg@1152: 0.0130 (step 0) -> terminal 0.0208 (1.6x base),
  late-run highs 0.0278 (2752) and 0.0330 (3252, best bench of any RL
  arm in project history incl. the partial-credit run's 0.031).
  Noisy band 0.005-0.033 throughout; AIME 0.0000 always.
- Trajectory: think span compressed from ~57-67 tokens to PINNED at
  the 33 floor by ~step 3000 (identical attractor to rounds 3/4), but
  fence compliance held 0.99+, ended_fraction ~1.0, and train reward
  KEPT CLIMBING (terminal pool 0.075) — the floor+fence priced out
  the bare-guess and skeleton attractors, and accuracy improved even
  at the floor. Whether the 33 forced think tokens carry real
  hidden-carry compute or just habit is unmeasured here (carry
  ablation eval would answer it).
- Diversity drain confirmed but non-fatal: opener concentration
  2/14 -> 10/16 by step 1000; wg reward std thirds 0.066/0.055/0.034.
  Entropy collapse is the round-6 problem, not a death mode here.
- Guard postmortem vindication: zero_reward_frozen_updates_total 766
  (19% of 4000 updates were all-zero minibatches, actor step
  skipped) vs exactly ONE all-zero pool (steps 3837-3840, streak 1,
  reset immediately, no stop). The red-team's minibatch-granularity
  fix did essentially all the work: a pool-level gate would have
  engaged 4/766 times. No death spiral, no length explosion, no
  desert absorption — round-4's mechanism arrived (drought windows)
  and the guard ate it.
- Teacher-forced val_bpb 1.646 -> 1.842 (+0.196): RL sharpening away
  from the LM distribution — round 4 paid +0.24 for a destroyed
  policy; this pays similar drift for a working one. Watch if the
  guard's do-no-harm framing needs a tighter bound in round 6.
- Verdict vs round-4 control (peak 0.0182, terminal 0.0000): KEEP
  every round-5 change (v3 corpus base, answer fence, floor 33,
  zero-reward guards). Round-6 target: entropy/diversity preservation
  (KL-to-SFT-prior anchor; DAPO-style dynamic sampling of
  zero-variance groups under review from paper survey) + K3 core for
  prior diversity (still user-blocked on API key).

## 2026-08-02: Sampling survey — VAPO / DAPO / GRPO / Dr. GRPO (user-requested)

Question: is rollout sampling (temperature etc.) the diversity
bottleneck? Survey answer: NO paper touches the sampler to fight
diversity loss; all manage it in the objective.
- Training rollout sampling: GRPO "naive nucleus" (params
  unspecified); DAPO unspecified (eval temp 1.0 / top-p 0.7); VAPO
  unspecified (eval same as DAPO); Dr. GRPO SPECIFIED: temp 1.0,
  top-p 1.0, top-k off — byte-identical to our enforced settings.
- Samples per prompt: GRPO 64, DAPO 16, VAPO 16 (ours: 16), Dr. GRPO 8.
- KL-to-reference: GRPO YES (0.04, k3 estimator, ref=SFT); DAPO
  REMOVED (long-CoT divergence is desired); Dr. GRPO REMOVED (beta=0,
  verifier reward eliminates distribution-shift concern); VAPO
  formulated but never given a coefficient (effectively absent). The
  literature majority drops KL for reasoning RL — our KL-to-prior
  idea is contrarian; their entropy tool is Clip-Higher instead.
- Entropy: DAPO Clip-Higher eps 0.20/0.28 (we already match);
  DAPO monitors token entropy and wants a SLOW UPWARD trend — we do
  not log token entropy at all (gap; only HL-Gauss target entropy).
- Zero/low-variance groups: DAPO dynamic sampling drops all-correct
  AND all-wrong groups and oversamples until the batch is full of
  variance-bearing groups — their single largest ablation gain
  (42->50 AIME). VAPO (value-based, like us) deliberately does NOT
  filter: the critic extracts signal from all-wrong groups; instead
  it adds Positive-Example LM Loss (NLL on correct rollouts, weight
  0.1) for the low-accuracy regime. Dr. GRPO: GRPO's group-std
  division UPWEIGHTS near-zero-variance groups (difficulty bias);
  removing std makes all-wrong groups contribute zero gradient
  naturally. Our minibatch zero-reward freeze is a middle position;
  766/4000 frozen updates in run 1056 = 19% of update budget spent
  on signal-free minibatches that dynamic sampling would have
  replaced with variance-bearing prompts.
- Us vs papers: advantages are critic-GAE, no group-std division
  (Dr. GRPO bias absent); token-level loss semantics already match
  (denominator covers minibatch); value pretraining 50 steps matches
  VAPO's; decoupled/length-adaptive GAE present.
- Round-6 candidates from survey, ranked: (1) token-entropy
  telemetry (free, diagnosis-grade); (2) VAPO Positive-Example LM
  Loss weight 0.1 — built for "remarkably low accuracy" tasks like
  our 3% regime, amplifies rare successes densely; (3) DAPO-style
  dynamic sampling replacing/augmenting the freeze (recycles the 19%
  wasted budget; needs care — our all-zero windows also train the
  critic toward zero, which the freeze design values); (4) KL-to-SFT
  -prior anchor (contrarian to DAPO/Dr. GRPO, but our failure mode
  is a WEAK prior collapsing to degenerate phrasing, not a strong
  model needing room to diverge).

## 2026-08-01: Round-5 semantic-collapse review — lower both model LRs

The structurally intact round-5 policy still collapsed semantically:
captured cross-problem responses converged on the same unrelated
"rate / age" skeleton, benchmark coverage contracted from 10 to 4
prompt groups and 7 to 2 nonzero modules, and aggregate accuracy moved
only 15/1152 -> 24/1152. Response sampling is the exact categorical
policy (temperature 1, top-p 1); the failure is not duplicated RNG.

Operator decision: lower the shared actor/critic AdamW default from
5e-5 to 2e-5. Both derived Muon defaults therefore move from
1.2083e-4 to 4.8333e-5. This makes the next run a both-model LR arm;
no training job was launched as part of the config change. Round-5's
first actor update measured post-update token KL 0.00743 and max token
log-ratio 4.23, while think length fell ~120 -> ~47 within 100 updates.
An older policy-schema LR arm at 2e-5 retained materially more benchmark
length/diversity at 2k, but remains only supporting evidence; the new
deterministic hidden-carry configuration still needs its own 2k ablation.

## 2026-08-02: Both-model LR 2e-5 ablation — KEEP; semantic prior still weak

Job 1068 `sft3_rl_answer33_gsm8k_lr2e5_2k` completed 2,000 actor
updates from the same selected v3 three-epoch SFT checkpoint as round 5.
Only both models' rates changed: actor/critic AdamW 5e-5 -> 2e-5 and
actor/critic Muon 1.2083e-4 -> 4.8333e-5. Reward, GSM8K data/seed,
sampling, fence, and 33-token floor were unchanged.

Aligned against round 5 at step 2,000:
- FineWeb BPB 1.6833 vs 1.7744 (0.0911 less drift).
- Bench accuracy 0.0148 vs 0.0122 (small/noisy), but prompt coverage
  11/144 vs 4/144 and nonzero module coverage 5/18 vs 3/18.
- Bench within-group reward std 0.0292 vs 0.0117; emitted mean 93.1 vs
  41.6 tokens. The 16 saved transcripts stayed 16/16 unique; normalized
  cross-problem similarity was 0.397 vs 0.646.
- Training reward 0.0527 vs 0.0508, within-group std 0.0897 vs 0.0255,
  think mean 53.4 vs 37.4. Zero-reward actor freezes fell 206 -> 16.
- New arm's held-out peak was 0.0278 at step 1,752 with 16/144 prompt
  coverage; old arm's best through 2k was 0.0208 at step 500 with
  10/144 coverage. AIME remained zero in both.

Verdict: KEEP the lower defaults. This is a clear diversity, coverage,
and retention win without sacrificing train reward. It does not solve
semantic reasoning: terminal transcripts remain arithmetically
nonsensical, just less template-collapsed. LR controlled collapse rate
and severity; the weak generative prior / terminal-only credit remains.

Launch note: jobs 1066 and 1067 failed before training (wrong entry-point
import, then system Python missing FLA). Their partial startup artifacts
were preserved at
`postraining/runs/sft3_rl_answer33_gsm8k_lr2e5_2k.failed_start_1067`;
1068 used the same `.venv/bin/python -m ...` entry point as round 5.

## 2026-08-02: Canonical single-contract math prompts

Prompt audit found that the answer-fence rewrite preserved DAPO's duplicated
formatting mechanically: DAPO/DeepMind/AIME rows carried two legacy Answer:
directives (20 DAPO rows carried a third Chinese directive), and each became a
complete think/answer fence instruction. All 6,000 DeepMind-family v3 SFT
documents therefore taught the contract twice; GSM8K taught it once.

Worse, the 173-character DeepMind SFT prefix exceeded normalize_problem's
160-character identity window. Every one of the 6,000 DeepMind documents
collided to one split identity; all landed in training and zero entered the
256-problem SFT holdout/gate panel.

Fix: one shared `math_prompt` contract now strips every known DAPO/DeepMind/
AIME/GSM8K wrapper (including old fence rewrites and the Chinese reasoning
directive) and emits exactly `{bare problem}` plus one full suffix:
`Start your response with <think> ... <answer></answer>.` SFT stores bare
problem identities for every family; runtime RL/eval, trace generation,
future GSM8K parquet generation, and OPSD share the same constants.
`answer_fence_prompt_schema` is stamped and checked across SFT, RL manifests/
checkpoints, OPSD, initialization, and resume so wording cannot change
silently. Existing v3 SFT and the live 40k RL run are legacy-schema artifacts;
regenerate/retrain SFT before a new canonical-schema RL run. The already-
running process continues with its in-memory legacy code, but its checkpoint
must not be resumed under the new prompt implementation.

Verification: all 487 postraining CPU tests passed (17 skipped), including
real-parquet checks that every canonicalized DAPO, GSM8K, AIME, and DeepMind
prompt contains exactly one of each fence token and no surviving Answer:
demand. Source parquets and the live run were not rewritten in place.

## 2026-08-02: Live hidden-carry ablation (job 1072, step 13,536)

Matched evaluation while job 1071 continued: 768 GSM8K-train prompts x 4
samples per arm, same prompt panel/seed; full policy, carry-content zeroed,
and complete combiner bypass (`token_only`). This is in-distribution after
many GSM8K epochs, so it measures whether the learned policy uses hidden
carry, not held-out reasoning generalization.

- Full 0.3337 vs no-content 0.2233: +0.1104, 95% bootstrap CI
  [+0.0905, +0.1315], permutation p=0.00005.
- No-content 0.2233 vs token-only 0.1012: +0.1221, CI
  [+0.1012, +0.1439], p=0.00005.
- Full vs token-only: +0.2324, CI [+0.2051, +0.2604], p=0.00005.
- Full/no-content lengths were identical (41.9/41.8 mean), so the carry-
  content accuracy gain is not a termination-length artifact. Token-only
  length rose to 59.4 and its occasional tail reached 1024.
- Mechanistic probe: hidden injection RMS = 0.245x token-embedding RMS;
  zeroing hidden changed taken-token log probability by 0.112 nats/action
  absolute mean and critic value by 0.102 absolute mean. The critic hidden-
  read delta correlated 0.690 with reward.
- AIME stayed 0/120 in all arms.

Verdict: the current combiner is decisively live and both components matter
on the memorized GSM8K training distribution: hidden content contributes
~11 points and the type-bias/MLP pathway another ~12. This overturns the old
collapsed-policy ablation's accuracy-neutral result, but does not establish
transfer; a held-out matched arm is still required.

## 2026-08-02: Canonical-v4 SFT selected + DAPO OPSD data built

The immutable HF-only canonical corpus contains 36,286 verified documents
(9.3M GPT-2 tokens) with the expected source mix. Full CPU tokenization
validated 35,812 train / 474 held-out documents over 256 held-out problems;
all completion fences are structurally valid. The corrected held-out set now
contains 79 DeepMind traces rather than zero. Corpus SHA256:
`ac398fc38e4db4d5d53cd03594850b96d3be79425fc80f963f326e23f99e8007`.

The first e2/e3 submissions (jobs 1074/1075) failed before training and
exposed 76 rows whose stored problem retained boundary whitespace while the
composed document used stripped bytes. The builder now stores the exact
canonical bare problem; the failed empty run directories and invalid corpus
artifacts were removed and regenerated. Successful matched arms:

- 1079 e2, 586 steps: holdout completion CE 0.6253.
- 1080 e3, 879 steps: holdout completion CE 0.6359.

The training-time gates accidentally graded all sources as Minerva and used a
one-token think minimum. Immutable-checkpoint reruns 1085/1086 corrected the
79 DeepMind rows to exact style and enforced the corpus p1 think-span floor of
33 tokens. Corrected e2/e3 metrics respectively were: strict contract
accuracy 0.0127/0.0332, mixed prompts 0.0859/0.2031, within-group reward std
0.0300/0.0728, structural format 0.9414/0.9658, and termination
0.9521/0.9785.

Verdict: select e3. Its slightly worse teacher-forced CE is outweighed by a
large win on every rollout-learnability and structural metric.

The immutable DAPO OPSD adapter validated 1,791,700 physical rows as 17,917
conflict-free logical examples, dropped 10 prompts over the 1,024-token
student cap, and created 17,651 train plus 256 SFT-decontaminated gate rows.
The wrong-answer control is a deterministic, split-local,
multiset-preserving answer derangement with no numerically equivalent or self
donors. Thus train and gate both preserve their own exact answer-frequency
distribution without borrowing donors across the split. The OPSD loader fully
audited all 17,651 correct-reference rows with zero runtime rejections. OPSD
now refuses an explicit answer arm unless its training bytes, held-out gate,
split schema, and source SFT hash match the immutable DAPO build manifest.

## 2026-08-02: Answer-only self-rationalized OPSD development gate

Paper/source audit used arXiv 2601.18734v3 and the authors' repository at
commit `7448751f307a9cdbcc1246dd1565a1a605b443df`. Algorithm 1 conditions the
frozen teacher on a worked reference solution and does not sample a teacher
trace. The released `reason_first` option does sample one, but is disabled in
the main reproduction and expects a worked reference. DAPO has only verified
final answers, so the experiment below is explicitly a same-model,
answer-only extension—not a paper reproduction.

The development gate reused the already inspected 512-prompt panel and is
therefore ineligible for authorization. For each of 4,096 frozen
question-only responses, the rationale source used sample `(j + 1) mod 8` so
the scored response never conditioned its own teacher. Question, correct-
answer, and permuted-answer rationales came only from the frozen v4 SFT
checkpoint, were stripped to think content, symmetrically filtered, and cut
to exactly 96 tokens. Correct/permuted/direct teacher response positions were
identical. The direct control paired the correct answer with the independent
question-only rationale. Primary scoring excluded answer-equivalent spans and
the last think quartile.

Job 1171 result (`opsd_dapo_self_rationalized_dev96_v3`): underpowered and no
evidence of a correctness-directed OPSD update. It retained 2,540/4,096
rationale triplets (62.0%), leaving 27 correct and 2,316 incorrect structured
responses across 23 mixed prompts. Correct-rationale AUC uplift was +0.0371
versus both direct and permuted controls, with both confidence intervals
crossing zero and permutation p >= 0.31. Exact clipped-update correctness
contrast was +0.00029 versus direct (CI crosses zero, p=0.813) and -0.00045
versus permuted (CI crosses zero, p=0.692). Answer-token sanity was also not
robustly positive.

The contexts changed the gradient substantially—correct/control gradient
delta was roughly 1.25-1.38x the correct-teacher gradient norm—but that change
did not align with response correctness. Mean pre-conclusion forward KL was
0.154 for correct and permuted rationales versus 0.141 for the direct control;
correct/permuted gradient cosine was 0.567. This is dense context/style signal,
not demonstrated mathematical supervision. Verdict: do not create a sealed
authorization panel and do not train this answer-only self-rationalized OPSD
variant. A fresh panel would only be justified by a new mechanism that first
shows a material correctness-directed update on development data.

## 2026-08-02: Broad verifier-mixed VAPO launched from canonical-v4 SFT

The immutable v5 mixture cycles 64 prompt groups as 28 deduplicated DAPO,
20 module-stratified DeepMind Mathematics, 8 GSM8K-train, and 8 official MBPP
train tasks. Every 16-group optimizer step is stratified 7/5/2/2. Rewards are
binary exact only, structurally grade only the registered answer span, and
use no nearby/partial math reward. An all-zero source contributes critic
targets but its actor rows are masked until that source produces an exact
success. Source-level reward/format/termination and typed Python verifier
outcomes are logged to JSONL and TensorBoard.

The Python verifier is `bwrap_python_positive_ast_all_tests_binary/v5`: a
disclosed deterministic subset, bubblewrap network/filesystem isolation,
CPU/address-space/file/FD limits, global eight-sandbox concurrency, and a
completion sentinel emitted only after every test. Its positive AST policy
blocks process, reflection, frame/code access, dynamic execution, and magic
comparison hooks. All 374 official reference solutions pass a mandatory
build-time preflight. Earlier v1-v4 attempts are invalid preserved artifacts:
red-team review found early-exit/sentinel/frame bypasses, a fixture-order bug,
mixed-status packing failure, and an unreliable per-user process limit. No
actor update from those attempts is part of the v5 lineage.

The v5 frozen-policy gate passed at aggregate exact accuracy 0.0205: DAPO
0.0179, DeepMind 0.0281, GSM8K 0.0312, MBPP 0.0000. MBPP therefore starts
actor-masked, not as negative policy evidence. The first production pools
completed without verifier infrastructure failures; at rollout step 64 the
source accuracies were DAPO 0.0201, DeepMind 0.0688, GSM8K 0.0391, MBPP 0,
with 0 Python timeouts. Job 1189 targets 40,000 actor steps in
`postraining/runs/sft4_vapo_broad_v5` from the byte-hashed canonical-v4 SFT.

Scope qualification: 7,123/7,217 GSM8K RL rows exactly overlap the v4 SFT
corpus (mostly `gsm8k_socratic`), so GSM8K reward is not novel-domain
evidence. DAPO has one exact SFT overlap and DeepMind none. The RL sources have
zero exact overlap with AIME-2024; DeepMind RL-full has zero exact overlap with
the 144-row DeepMind-easy benchmark.

## 2026-08-02: LeJEPA answer-encoder v1 failed; reference-regime rerun queued

Job 1196 (`answer_lejepa_v1_2k`) completed 2,000 steps and failed the
behavioral reward gate. Deterministic validation projection rank fell from
16.92 at initialization to 1.23, validation SIGReg rose to 26.09, and the
mean view cosine reached 0.9933. The learned geometry was not semantic:
`airplane` scored above `kitten` against `cat`, a negated statement scored
above its valid paraphrase, and reordered versus buggy functions differed by
only 0.00057 cosine. Backbone-space probes also failed.

Direct numerical comparison against `../lejepa` showed exactly equal SIGReg
values for identical inputs and random slices (1.01120496 in both
implementations). The Epps-Pulley statistic and tensor axes were therefore
not the bug. The recipe was misapplied: v1 used the BatchNorm projection head
as a deployable reward even though LeJEPA discards it downstream, used only
32 samples for a 128-dimensional distribution, and forced invariance across
four often-disjoint 5-30% text crops. Train-mode projection rank was only
10.24 at step 2,000; an isotropic 32x128 Gaussian sample is expected near
27.4 effective rank.

Job 1201 will quantify live-batch versus stored BatchNorm statistics on the
preserved v1 checkpoint. Job 1202 (`answer_lejepa_reference_2k`) is the
corrected baseline: backbone reward, exact reference MLP defaults, batch 256,
projection dimension 16, four full-text weakly masked views, no encoder
dropout, and learning rate 0.002. Both are queued behind the active v5 VAPO
workload; no success claim is made before their measured results.

## 2026-08-02: Direct-CLS LeJEPA answer encoder fixes SIGReg, fails semantics

The v1 diagnostic (job 1201) confirmed a severe stored-BatchNorm mismatch:
FineWeb projection effective rank was 1.28 with stored statistics versus 5.98
with current-batch statistics. The reference-regime v2 run (job 1202) fixed
the projected distribution (15.6/16 train and 11.82/16 validation effective
rank) but still failed the semantic gate. A source audit then found that
14,839 of 25,000 answer rows were one token, only 9,779 token sequences were
unique, and the chunk-mean encoder could not express the intended within-chunk
function-order invariance test.

Job 1209 (`answer_lejepa_cls_v3_2k`) removed those confounds. Each token is a
ViT-style patch in one bidirectional Transformer; a learned CLS token attends
the full sequence and its normalized final state is the answer embedding.
There is no token or chunk mean. Dynamic sinusoidal positions retain token
order, while invariance must be learned from views. Transformer blocks are
initialized independently. The corpus contains 25,000 unique truncated token
sequences (23,750 train, 1,250 held out, zero overlap; median 256 tokens), and
uses four full-length 30%-masked views, batch 256, exact reference projector
and SIGReg, mixed-corpus validation, gradient scaling, and the reference
nonzero learning-rate floor.

The 2,000-step run completed in 283 seconds. Optimization and SIGReg worked:
held-out loss was 0.03329, projected effective rank was 14.92/16, projected
dimension standard deviation was 1.001, and view-center cosine was 0.99836.
The deployable backbone cosine nevertheless failed all four semantic checks.
Backbone effective rank was only 5.05/256; function reorder minus semantic bug
was -0.000010, numeric equivalents minus non-equivalents was -0.004873,
paraphrase minus negation was -0.000436, and taxonomy-related minus unrelated
was -0.000119. Verdict: the projected LeJEPA objective no longer mechanically
collapses, but same-document token masking did not produce a usable raw-cosine
backbone reward. This run used no block-shuffle views, so it does not test
explicitly learned function-order invariance; projected-space behavior and a
targeted reorder augmentation remain separate diagnostics. Do not integrate
this checkpoint into RL.

Reviewer red-team narrowed the next causal test to job 1212
(`answer_lejepa_cls_direct_sigreg_v4_2k`), which kept v3's seed, data, masking,
and disabled block shuffling but applied invariance and SIGReg directly to the
256-dimensional CLS backbone used for reward. It completed 2,000 steps with
held-out loss 0.12238 and raised deployed effective rank from 5.05 to 23.62.
This confirms that v3's nonlinear projector had absorbed much of SIGReg.

The higher-rank representation still failed three of four semantic checks.
Numeric equivalents minus non-equivalents was -0.007447, paraphrase minus
negation was -0.005271, and function reorder minus semantic bug was -0.000504;
only taxonomy-related minus unrelated was positive (+0.003476). The v3
projected-space diagnostic (job 1211) also failed all checks, including a
-0.689 numeric-equivalence margin because `10.0` was far below `100`.
Verdict: direct reward-space regularization improves diversity but does not
recover semantic correctness. Missing cross-answer equivalence views are now
the leading explanation. Do not integrate v4 into RL or spend another run on
the same masked-view objective. Block shuffling can test function order only;
it cannot repair the independently observed numeric and negation failures.

## 2026-08-03: Multi-crop projected answer encoder queued

The v3/v4 manifests exposed the central adaptation error: both used four
full-length views, no local views, no block shuffling, and only independent
token masking. They therefore did not test the analogue of LeJEPA's strongly
different patch grids. V5 changes the view orbit to two global 30-100% crops
and six local 5-30% crops with 10% masking. A 256-sample FineWeb audit found
eight unique views for 255 samples (mean 7.996); global crops averaged about
93 tokens and local crops about 26.

The standard projected invariance plus SIGReg objective is restored, and the
same projector is the deployed target/answer embedding. Because that head is
retained rather than discarded, its hidden normalization is per-example
LayerNorm; reference BatchNorm would train on crop-batch statistics but deploy
on single full answers with stored statistics. Model, corpus, batch, optimizer,
and seed otherwise remain fixed. The focused suite passes 28 tests and an
independent review found no fresh-run blocker. Job 1215
(`answer_lejepa_multicrop_v5_2k`) is queued at concurrency 1 behind the active
shared-GPU workloads; no result claim is made before its full 2,000 steps.

Job 1215 subsequently completed successfully. Held-out view-center cosine was
0.78821, projected effective rank was 11.76/16, SIGReg was 16.47, and total
validation loss was 0.51664. The much lower view cosine confirms that v5 no
longer solved invariance with nearly identical inputs. It also produced the
first material positive taxonomy relation: the minimum of `cat`/`kitten` and
`cat`/`dog` exceeded `cat`/unrelated by 0.05882 cosine.

The overall behavioral gate still failed. Numeric equivalence margin was
-0.38541 (`10.0` and `10.1` both near 0.614 cosine while `50` and `100` were
above 0.999), paraphrase minus negation was -0.29811, and function reorder
minus semantic bug was -0.00683. Verdict: the multi-crop fix materially changes
and improves the representation, but the 2,000-step v5 checkpoint is not an
RL reward model. Do not integrate it.

## 2026-08-03: Masked latent-prediction LeJEPA learns patches, still fails semantics

Job 1219 (`answer_lejepa_masked_latent_v6_2k`) tested the faithful
missing-patch formulation without CE. A shared bidirectional encoder processes
one complete sequence and one position-aligned context with 30% contiguous
span masking. A training-only 256 -> 1024 -> 256 predictor maps masked-context
token and CLS states toward the corresponding complete-view states; both sides
then use the same attached 256 -> 2048 -> 2048 -> 16 LayerNorm projector.
There is no EMA target or stop-gradient. Patch and CLS populations each use
the exact two-view LeJEPA center MSE plus SIGReg and are averaged, so token
count cannot drown out the deployed CLS objective. A one-token answer is fully
masked in the context while its original token remains attached on the target
side. The focused suite passes 54 tests, including original-token gradients,
alignment/padding, exact MSE scaling, sampler resume, and reference diagnostic
cosine. Independent review found no queue blocker.

The 2,000-step run completed in 190 seconds. Held-out masked fraction was
0.30023 (45.53 predicted patches per sample). Target/predicted patch effective
ranks reached 15.93/16 and 15.92/16 with 0.99034 direct cosine. The deployed
target CLS projection reached rank 15.50/16; global prediction cosine was
0.99860 and backbone rank was 19.94/256. The intended latent-prediction problem
therefore trained successfully without representation collapse.

The reward gate nevertheless passed only taxonomy. Numeric equivalence margin
was -0.11033: against `10`, `10.0` scored 0.88805 and `ten` 0.99770, while
`9`, `50`, and `100` scored 0.99791, 0.99838, and 0.99788. Paraphrase minus
negation was -0.01179 (0.98192 versus 0.99371), and function reorder minus
semantic bug was -0.000018 (0.999981 versus 1.000000). Taxonomy passed by
+0.00182. These are large improvements over v5's -0.38541, -0.29811, and
-0.00683 failing margins, but the signs remain wrong. Moreover, numeric
equivalence worsened during training from -0.00522 at step 100 to -0.11033 at
step 2,000 while patch prediction steadily improved. This is evidence that
objective convergence alone does not produce the requested cosine ordering in
the present setup. Do not integrate v6 into RL.

## 2026-08-03: Variable-cardinality masked JEPA queued

V7 removes masked patches from the context entirely. The compacted visible
tokens retain their original positions, and a two-layer Transformer predictor
uses learned position-conditioned patch queries plus a global query to predict
the complete target patch latents and CLS. Query self-attention is
bidirectional; cross-attention sees only the compacted attached context
encoder states. Predictions and attached complete-target states then pass
through the same LayerNorm projector before the unchanged two-view center MSE
and separate 50/50 patch/global SIGReg objectives. Padded query outputs never
enter either loss population. The inference checkpoint retains only the
shared encoder and projector.

The focused suite passes 61 tests, including original-position preservation,
unequal missing-patch counts, fully deleted single-token contexts, isolated
context gradients, padding/batch-neighbor invariance, independent predictor
layer initialization, exact MSE scaling, and valid-only SIGReg populations.
Independent review found no queue blocker.

Job 1221 (`answer_lejepa_variable_cardinality_v7_2k`) completed the full 2,000
steps in 196 seconds. Held-out visible/masked fractions were 0.69977/0.30023.
Target and predicted patch ranks were 15.91/16 and 15.92/16 with 0.99039 patch
cosine. The projected target CLS rank was 15.30/16; global prediction cosine
was 0.96451, versus 0.99860 for the same-length v6 predictor. This confirms
that deletion creates a genuinely harder complete-state prediction problem
without representation collapse.

The reward gate failed 0/4. Numeric equivalence margin improved from v6's
-0.11033 to -0.05279, but `10.0` remained only 0.94599 cosine from `10` while
`9`, `50`, and `100` were all about 0.999. Paraphrase minus negation was
-0.01401, function reorder minus semantic bug was -0.000013, and taxonomy
related minus unrelated was -0.00226. At step 100 the numeric equivalence
margin was already negative (-0.00420) and it worsened to -0.05279 while the
objective and rank improved. V7 is a faithful variable-cardinality JEPA, but
its projected CLS remains unusable as a correctness reward. Do not integrate
v7 into RL.

## 2026-08-04: V7 projected patch-set matching rejected

Job 1223 (`answer_lejepa_v7_patch_set_probe`) evaluated exact Hungarian
matching over v7's normalized projected contextual token patches. The report
includes the same-checkpoint CLS baseline, matched-only normalization, and a
cardinality-aware variant that assigns zero reward to unmatched patches. The
first attempt failed before model execution because the standalone script did
not add the repository root to `sys.path`; the standard entry-point bootstrap
and an outside-repository subprocess regression test fixed it. Attempt two
completed successfully; the focused suite passes 65 tests.

Matched-only patch scoring improved numeric equivalence from the CLS margin
of -0.05279 to -0.02507, taxonomy from -0.00226 to +0.00400, and paraphrase
versus negation from -0.01401 to -0.01083. All except taxonomy still fail.
Function reorder versus semantic bug worsened from -0.000013 to -0.000068:
reordered cosine was 0.999926 while the bug was 0.999993. Cardinality-aware
matching produced -1.34098 numeric and -0.32874 paraphrase margins because it
mostly measured GPT-2 token-count differences. Verdict: CLS pooling is not
the cause of the missing function-bug distinction, and existing token patches
do not contain a useful localized program-semantic signal. Do not replace the
CLS reward with this patch-set readout.

## 2026-08-04: Pretrained Qwen baseline and CLS-only v8

Job 1228 (`qwen3_embedding_8b_answer_reward_probe_v2`) evaluated the frozen
Qwen3-Embedding-8B checkpoint at 16, 256, 1024, and 4096 dimensions using
plain symmetric, symmetrically instructed, and documented retrieval-style
target-instructed scoring. The 16-dimensional result is explicitly an
unsupported extrapolation below Qwen's advertised 32-dimensional MRL minimum.
The audit contains the original four probes, five graded semantic/code/math
cases, and 41 systematic numeric targets; semantic and numeric families are
reported separately so the numeric sweep cannot dominate the conclusion.

The intended plain symmetric 256-dimensional readout passed all four small
behavioral checks: numeric equivalence +0.02593, taxonomy +0.16650,
paraphrase versus negation +0.13591, and function reorder versus bug +0.04981.
Across the five curated semantic/code/math cases it achieved 0.7808 mean
Spearman and 0.8654 pairwise ordering accuracy. Numerical sweeps achieved
0.7795 mean Spearman and 0.8628 pairwise accuracy. Dimensionality was not the
main limitation: 1024-D and 4096-D plain scoring were similar.

The strict correct-versus-incorrect margin remained negative. For the linear
equation, the correct answer-only `5` scored 0.36450 while a lexically similar
arithmetic slip ending in `x = 4` scored 0.88509. For factorial, an off-by-one
loop scored 0.98139 while an equivalent `math.prod` implementation scored
0.83686. Thus Qwen provides a substantially better coarse, often monotonic
semantic reward than any trained answer JEPA here, but raw cosine can strongly
prefer a near-copy bug over a behaviorally correct rewrite. Use it as a frozen
baseline or shaping component, not as the sole correctness reward. The
canonical report is
`ablation_results/qwen3_embedding_8b_answer_reward_probe/result.json`.

Job 1229 (`answer_lejepa_global_cls_v8_2k`) then tested one deployed
256-dimensional projected CLS with no patch queries, losses, SIGReg samples,
or inference readout. A training-only global Transformer query predicts the
complete-answer CLS from compact, original-position contexts; predicted and
target CLS states use the same attached LayerNorm projector and exact
two-view center MSE plus SIGReg.

The full 2,000-step run completed successfully but failed all four behavioral
checks. Projected effective rank was only 8.55/256 and backbone rank was
3.45/256; global prediction cosine was 0.95557. Numeric equivalence improved
from v7's -0.05279 to -0.04306 and paraphrase/negation from -0.01401 to
-0.00863, while taxonomy worsened from -0.00226 to -0.01053 and function
reorder/bug worsened from -0.000013 to -0.000087. `10`, `50`, and `100`
remain nearly identical (1.00000, 0.99509, and 0.99914). Verdict: changing to
one larger CLS is architecturally cleaner but does not solve the missing
cross-answer correctness geometry. Do not integrate v8 into RL.

## 2026-08-04: Intra-trajectory token TPO replaces candidate TPO

The K=8 candidate construction collapsed while Adam/Muon turned tiny target
gradients into policy moves millions of times larger than the nominal target
KL. Candidate diversity also decayed toward one token. That implementation did
not have a learned action-Q head: it assigned the executed candidate critic GAE
and zero utility to the other candidates. All candidate rollout, storage, and
replay machinery has been removed rather than retained as a dormant option.

`--target-policy-optimization` now ports the successful local CleanRL
HalfCheetah intra-trajectory objective. Every executed token receives raw
detached critic GAE `A`. For categorical tokens, the continuous-density ratio
formula is replaced by its normalized executed-token-versus-rest counterpart:
`target_p = sigmoid(logit(old_p) + A / eta)`. Binary cross entropy fits the
current executed-token marginal to that locally feasible target and
self-extinguishes there in isolation. Repeated identical prefixes can request
incompatible marginals for different sampled tokens, so the shared categorical
policy may fit a compromise. Stable target-versus-rest log odds are computed
directly from full-vocabulary logits rather than reconstructed from rounded
softmax probabilities. There is no PG auxiliary, counterfactual score,
comparison token, or PPO clip.

Unlike the HalfCheetah run, advantages are not whitened: binary verifier
returns and unit-range value targets already define a meaningful scale. With
gamma=1, GAE remains on approximately [-1, 1], apart from the critic support's
narrow margin bins; default eta=2 therefore shifts old-policy odds by roughly
[exp(-0.5), exp(0.5)] for advantage magnitude at most one. This eta-controlled
old-policy anchor is the target-space trust control used here. It does not
mathematically bound the realized optimizer step, and no hard KL guard or
adaptive KL controller changes the update.

The actor objective uses the ordinary transition occupancy measure: every
active token contributes once and the complete optimizer minibatch's action-
token count is the denominator. This matches the CleanRL reference's flattened
fixed-step rollout and the existing VAPO path. There is no trajectory-length
reweighting; replay shards are memory partitions only.

The source-success actor mask remains. An entirely unsuccessful source is
critic-only even when an older behavior batch has become stale; active sources
retain dense intra-trajectory critic credit. Telemetry now reports target and
old probabilities, target log-odds shift, target move and target KL. The
read-only post-update replay reports actual target-fit KL and probability
residuals alongside achieved behavior KL. The actor objective schema is v6,
preventing silent resume from candidate TPO or the infeasible density-ratio
port.

## 2026-08-06: Corpus v8 — decontamination registry, math to 30%, worked-step drills, ToaST+TST

An audit of how much mathematics the pretraining corpus actually contains, and
of what post-training was silently training on, produced four connected
changes. None of them has been trained on yet; everything below is
construction and measurement, not a result.

### The problem registry (`postraining/problem_registry.py`)

There was no global answer to "which split owns this problem". The corpus
builders carried a `--heldout` list that named DAPO and AIME and not GSM8K,
and for web sources the exclusion was structurally inert: `quality_keys` is
the whole document chunk, so it fired only when an entire page equalled a bare
problem statement. Measured consequence: 97.5% of `openmath_gsm8k`, 97.0% of
`gsm8k_socratic`, 96.7% of `had653_gold` and 96.9% of the GSM8K RL pool were
problems the base model had already pretrained on.

`postraining/problem_sources.json` now declares 38 sources across
`eval > rl > sft > pretrain`, and `scripts/build_problem_registry.py` resolves
them to one assignment: **344,309 distinct problems** (eval 2,472, rl 204,430,
sft 137,407). Identity is the same `pgolf-holdout` blake2b over the
framing-stripped, NFKC-normalized, casefolded statement the builder already
used, so registry keys and the retired held-out keys agree bit for bit.

### The n-gram index (`postraining/decontaminate.py`)

Exact key matching cannot see a problem quoted inside a larger page, so a
13-word-gram index runs alongside it, the GPT-3/Llama convention. Words are
alphanumeric runs after NFKC and casefold; hashes are a cached blake2b per
word folded into a numpy polynomial rolling hash (0.64 ms per 3,000-word
document, 28 MB/s), stored as a sorted `uint64` array searched with one
vectorized `searchsorted`. **4,045,060 unique 13-grams** over the eval, RL and
SFT splits.

Three defects a red-team pass found and confirmed by execution, all now fixed:

- The index was built from **raw prompts**, so the 31-word DAPO preamble
  contributed 19 pure-boilerplate 13-grams. Every QA document the corpus
  builder rendered with that same template was then refused as contaminated:
  measured **20.2% of `deepmind_math` and 16.9% of `openmath_instruct`
  rejected for reproducing an instruction wrapper**. `--min-ngram-hits` could
  not separate them, since a real GSM8K test problem also contributes exactly
  19 distinct n-grams. The index now applies `strip_framing`, the same
  canonical view `problem_key` hashes, and `deepmind_math` fell to 5.18%.
- `--index-splits` defaulted to `eval` alone, leaving **341,837 of 344,309
  protected problems (99.3%) on exact matching only** — including the GSM8K
  train pool the whole mechanism exists for. Five admitted documents were
  verified to contain a GSM8K-train 13-gram. The default is now
  `eval rl sft`, and `ProblemGuard.__init__` **fails closed** when the index
  does not cover every non-empty protected split rather than silently
  degrading.
- `scripts/audit_problem_overlap.py` could not measure the builder: it read
  only `document.text` and dropped `quality_keys`, reporting
  `openmath_instruct exact 0.00%` where the builder rejected 16.8%, and it
  skipped `token_shards` entirely so fineweb was never audited. It now runs
  the builder's own `ProblemGuard` over whole `RawDocument`s and decodes the
  fineweb shards.

Windows one word dominates (over half the slots) are dropped from the index —
`a_1 a_2 a_3 ...` reduces to "a 1 a 2 a 3 a 4 a 5 a 6 a" and would flag every
LaTeX subscript run. 244,236 windows were dropped this way. The filter bounds
the pathology rather than eliminating it; an alternating pattern still lands
half its windows, because `index_words` discards the operators that would
distinguish it.

**Measured admission rates** on a 25 MB-per-source sample against the final
index: `fineweb` 0.00%, `fineweb_edu_dedup` 0.06%, `github_code_clean` 0.22%,
`sci_code` 0.40%, `open_web_math` 1.58%, `finemath_4plus` 2.45%,
`deepmind_math` 5.18% n-gram, `openmath_instruct` 16.45% exact + 15.98%
n-gram. The last two are genuine overlap with the RL and SFT pools, not false
positives — OpenMathInstruct's GSM8K band is 97% GSM8K, and
`deepmind_interpolate_rl_full` comes from the same generator as
`deepmind_math`. The false-positive cost of the 100x wider index is therefore
approximately zero on pure web text, which is what made widening it safe.

### Worked-step drills and a held-out arithmetic probe

The corpus shortfall is not word problems; it is text that carries out an
arithmetic procedure step by step. `postraining/math_drills.py` generates
deterministic drills across 15 families with declared digit curricula:
column addition/subtraction with explicit carries and borrows, partial
products, long division, decimals, fractions, percentages, rounding,
comparison, and unit conversion. Two thirds are written in expanded form and
one third least-significant-digit-first.

A design flaw found by reading the generated output rather than the code: the
`forward` order was originally produced by *reversing* the LSB-first step
list, which printed carries before the lines that produce them — a non-causal
trace, exactly the thing this corpus exists to avoid teaching. `forward` is
now genuinely the expanded-form algorithm, and `FAMILY_ORDERS` declares which
orders each family can causally support. 9,884 of 10,000 generated drills were
re-solved independently with `Fraction` arithmetic: 0 wrong.

`postraining/arithmetic_probe.py` draws a held-out panel from a reserved seed,
excluding the training key set and the registry, and `assert_disjoint`
re-derives the training keys rather than trusting a manifest. Below three
digits a family's problem space is exhausted by any real training set — there
are a hundred one-digit additions — so the probe measures the multi-digit
regime and says so instead of reporting memorized items as held-out accuracy.
Built: 2,000,000 drills (703.2 MB, 352 characters each) and a 1,920-item
panel; the drill corpus audits at 0.00% exact and 0.00% n-gram against the
registry.

### Corpus v8 profiles

`k3_weights_math30.json` takes math from 15% to 30%, paid out of web
(50% to 42%), keeping `math_drills` at 6% and `deepmind_math` at 5% so
templated synthetic text stays a minority of the math budget. DeepMind's 56
`train-easy` modules are no longer round-robined: `module_weights` gives
arithmetic 5-6x, place value and rounding 4x, measurement 3-4x.

`--anneal-weights` draws the final `--anneal-fraction` of the budget under a
second profile (`k3_weights_math30_anneal.json`, math 50%, no raw web). The
stages **share their source iterators**, so the anneal continues each source
rather than restarting it — otherwise the tail of a one-pass corpus would
quietly become a second epoch of its heaviest sources. This is distinct from
the existing `--cooldown-data`, which swaps in a whole second dataset.

Embedding tying needed no work: `TIE_EMBEDDINGS` already defaults on, and
51.5M of 64.0M parameters are embeddings.

### ToaST + TST tokenizer

`tokenization/` combines vocabulary-independent binary split trees (ToaST,
arXiv 2605.22705v1) with triadic digit grouping and magnitude suffixes (TST,
arXiv 2604.11582v3). The shipped 50,257 artifact
(`data/tokenizers/toast_tst_n1_50k`) was trained on a 189.8 MB sample of the
math-30 mixture and measured on a 4.0 MB held-out slice of it: GPT-2 3.4673
bytes/token, ToaST+TST **3.7288 (7.54% better)**, zero round-trip failures,
and an exactly integral LP (zero fractional variables, zero relative gap).
The 4.0001-at-50,257 and 3.4131-at-16,384 figures quoted in earlier revisions
of this entry came from a different 95.3 MB exploratory sample and a 16,384
artifact that was never built; `tokenization/README.md` retired them and this
entry now agrees.

Two deliberate deviations: numbers stay lossless (variable-length boundary
groups instead of zero padding, so `0.1`/`0.10`/`0.100` do not collide), and a
numeric token's identity is `(digits, power)` so its value is exactly
`int(digits) * 10**power`.

**Compression is not capability.** A tokenizer that packs more bytes per token
also gives the model fewer forward passes per byte. Nothing here should be
adopted on the bytes-per-token table alone, which is what the four-arm
ablation in `pretraining/README.md` is for: GPT-2/quality, GPT-2/math30,
ToaST+TST-50k/math30, ToaST+TST-16k/math30. Bits-per-byte is
tokenizer-independent, so all four stay comparable to every run already in
`ablation_results/`.

The tokenizer artifacts are not on disk; the user runs the training commands.

### Corpus-v8 review round: what the red team changed

Eight confirmed findings, all fixed and covered by tests. The three that
would have silently corrupted a result:

**The contamination index was built from raw prompts.** Nineteen 13-grams of
DAPO instruction boilerplate ("Let's think step by step and output the final
answer within \boxed{}") entered the protected set, so any document quoting
the template was refused: 20.2% of `deepmind_math` and 16.9% of
`openmath_instruct`. `ContaminationIndex.build` now strips framing first, and
`deepmind_math` re-measures at 5.18%. A guard that rejects the right documents
for the wrong reason looks identical to a correct one in aggregate; only
reading the rejected documents showed it.

**Bits per byte was computed from a GPT-2 byte table regardless of tokenizer.**
The one metric the four-arm ablation is compared on would have been wrong for
arms C and D — and wrong in the flattering direction, since a denser tokenizer
would have had its bytes undercounted. `pretraining/byte_accounting.py` keeps
the table for GPT-2 corpora and decodes runs for everything else; a TST token
carries `(digits, power)` and renders to bytes that depend on its neighbours,
so no per-token table can exist for it.

**Vocabulary identity was compared on padded size.** GPT-2's 50,257 and a
trained 50,257 both pad to 50,304, so the cooldown-vs-main guard compared
equal for two tokenizers that agree on nothing. It now compares
`(kind, vocab_size, eot_id, spec_sha256, ngrams_sha256)`. Relatedly,
`SplitTreeNumericTokenizer.from_directory` did not verify `ngrams.bin`
against the digest in `tokenizer.json` — and since split trees are rebuilt
from those counts at encode time, `spec_sha256` alone never pinned the
encoding.

Also fixed: a stage-boundary truncation that destroyed the tail of the
document it split (found before review, confirmed reproduced by the reviewer);
zero-weight sources breaking budget allocation in three places in the
checkpointed builder; a resume-divergent rejection counter living on the
stateless guard; and `--index-splits` defaulting to `eval` alone, which left
99.3% of protected problems on exact matching — now `eval rl sft`, with
`ProblemGuard` refusing an index that does not cover the splits the registry
actually populates.

Two changes went beyond the findings. Packed QA documents are now
decontaminated per item: a DeepMind document holds 8-24 independent pairs, so
refusing the whole document for one held-out problem was discarding roughly
fifteen clean pairs each time. And `sample_tokenizer_corpus.py` applies the
guard, which it never did — a tokenizer fitted to the evaluation problems buys
shorter encodings of exactly the text it is judged on.

One bug the review request itself surfaced: `run_arithmetic_probe.py`
defaulted to `--temperature 0.0` for greedy decoding, and `top_p_sample`
implemented temperature by division -- `logits / 0.0` is +inf for every
positive logit and NaN for a zero one, which `torch.multinomial` refuses. The
probe would have crashed on its first step. `top_p_sample` now treats
temperature 0 as greedy, expressed as a one-hot distribution rather than an
argmax shortcut so the draw order stays one multinomial per step; a test
asserts the generator ends in the same state as a sampled draw, because that
order is part of the rollout execution schema every actor objective shares.

Two more probe defects found while writing that fix. Scoring paired captured
attempts with panel items positionally without checking `problem_index` --
`evaluate_latent_math` does sort the capture back into dataset order, so the
pairing was correct, but a permuted capture would have produced a per-family
table that was a permutation of the truth, which no aggregate would reveal;
the probe now asserts the alignment. And the headline accuracy credited a
`parsed_answer` scraped from the tail of an unterminated generation. The
headline score now requires `terminated` and `structural_format_ok`, with the
lenient number reported beside it rather than dropped.

One test was deleted rather than fixed: `test_the_trainer_refuses_an_unpadded_
vocab_size` asserted two source substrings existed in the trainer file and
would have passed if the guard body were `pass`. Its replacements in
`tests/test_byte_accounting.py` call the real functions.

Round three found no blocking defect and confirmed three things empirically
that had been argued rather than measured. Cross-boundary n-grams: joining a
document's segments does create windows spanning a boundary, and none of them
matched — 0 of 205 refusals came from a synthetic boundary. Item-level
decontamination over 4,000 DeepMind documents: 0.356% of items contaminated,
5.13% of documents holding at least one, 3,424 clean siblings released (5.35%
of the stream), no document losing every item. And the shipped index's split
declaration verified externally at 2,229/2,229 sampled problems detected.

The four nits are fixed. Probe transcripts are written before the scoring
guards, so a guard failure keeps its own evidence. Item and document rejection
counts now live in separate manifest maps (`item_rejection_counts` vs
`rejection_counts`) — summing them would have overstated documents refused by
~16x on the DeepMind key. The byte counter gained a test for decode runs
flushed around multi-byte neighbours, since the specials-are-one-byte
convention is only defensible if it is neutral across arms rather than
inherited from GPT-2's table. And `build_problem_registry.py` now ends with a
positive control that re-queries sampled protected problems against the
finished index and fails the build if an indexable one is not detected — as a
build-time gate rather than a load-time one, so the already-audited
`data/problem_registry/v1` stays valid.

Two operational items came out of that round and are recorded rather than
fixed. Build **arm C first**: at a fixed token budget the ToaST-50k arm reads
~13% more source text than the GPT-2 arms, so it is the arm most likely to
exhaust a thin source, and `build_source_caches` fails closed rather than
cycling. Discovering that after three other corpora exist is expensive.
Second, **518,560 of 2,861,011 protected rows yield no indexable 13-gram** —
255 of 300 sampled `deepmind_interpolate_rl_full` rows — and are therefore
protected by exact key alone, which for web text catches only a page that is
nothing but the problem. Those are short generated RL templates so verbatim
web reproduction is unlikely, but this is the decontamination's real residual
exposure. It is not what the index does wrong; it is what no n-gram index can
do.

Round four attacked the probe and the greedy branch itself and found one real
hole in a fix from round three. Greedy decoding accepts logits that sampling
refuses: `argmax` ranks NaN above every real logit and returns index 0 for a
wholly `-inf` row, so the `torch.multinomial` error that this codebase relies
on to notice a corrupted forward or an over-aggressive mask simply does not
fire at temperature 0 — which is the arithmetic probe's default. The failure
would have been a complete, correctly ordered, contract-passing per-family
accuracy table computed from argmax-of-NaN, with nothing aggregate to reveal
it. `top_p_sample` now requires at least one finite logit per row and no NaN or
`+inf` before taking the greedy path, and validates `top_p` in [0, 1] at both
temperatures, since a negative `top_p` masked every rank and arrived as the
same opaque multinomial error.

The first version of that guard tested only "at least one finite entry per
row and no NaN", which a row holding one `+inf` beside 31 finite logits
satisfies -- and argmax then returns the `+inf` position as a confident
answer. A stable softmax subtracts the row max, so `exp(inf - inf)` is NaN
and the sampled path refuses the same row: the parity claim was false for
exactly that case. `-inf` stays legal, because it is the mask value and a row
masked to one candidate is a request rather than a corruption. Parity now
verified case by case: clean, all-`-inf` row, one-`-inf` row among clean ones,
NaN, `+inf`, `-inf` alone, masked-to-one, finite-but-1e38, and `-inf` with
`+inf` all agree between temperature 0 and temperature 1 on refuse-or-accept.

The probe's latent-checkpoint refusal was reading the weakest of three
available signals. It tested `args["reasoning_mode"]`, but
`train_latent_vapo.py` reads its own mode with a `latent` default, so a
payload can carry a trained combiner and record no mode at all. Worse, the
guard was unreachable for its own target: a latent VAPO checkpoint has no
top-level `sft` key, so the prompt-schema check rejected it first and named
the wrong reason. It is now `carries_trained_combiner(payload)` — decided on
`combiner.*` parameter keys, with top-level `reasoning_mode` as a second
line — and it runs before the schema check. The schema rejection goes through
`parser.error` rather than a bare traceback, like every other rejection there.

Verified rather than changed, in the same round: the temperature-0 branch is
bit-identical to `argmax` across 1,440 truncation/seed combinations including
exact ties, and leaves the generator in the same state as a sampled draw in
all four truncation branches. The zero-init `CombinedEmbedding` is an exact
identity, not an approximate one, so the probe's arrangement is faithful
independently of `pin_emit`.

Round five found the positive control from round three to be wrong, and wrong
in the way that matters: run against the shipped `data/problem_registry/v1`
it raised `SystemExit` at 2,875/2,879. The four undetected rows were
`Simplify (d*d*((d*(d*d**3)/d)/d*d)/d)/(d/d**6) assuming d is positive.` and
its siblings — 18 words of which 12 are `d`. The control decided a row was
indexable by word count, but `ContaminationIndex.build` indexes a row only if
`informative_ngram_hashes(...).size > 0`, and `informative` rejects any window
one token dominates. So the control counted a row the builder correctly
declined to index and then blamed the index for not detecting it.

The provenance had been right all along: `covered_rows + short_rows =
protected_rows` exactly, so `short_rows` already meant "contributed no
informative n-gram". The control inherited that field's *name* and re-derived
a length test from it. `short_rows` is now documented as the narrower name it
is — a `v2` registry should call it `uncoverable_rows` — with the instruction
that anything deciding coverage must call the builder's predicate rather than
infer one. The control does that now and passes on `v1` at 2,875/2,875
(eval 602, rl 814, sft 1,459; 685 uncoverable).

All three round-three control tests passed throughout, and that is the lesson
worth keeping: they used diverse prose fixtures, and the only failing case is
a row that is *long but repetitive*. A green suite said nothing about whether
the next registry build would abort. Two regression tests now use the verbatim
failing row, one asserting it is not blamed on the index and one asserting
`control["indexable"] == index.covered_rows` and
`control["uncoverable"] == index.short_rows`, which is the drift itself rather
than a symptom of it.

Also fixed in the same path: `registry.write` ran before the control, so a
failed gate left `registry.parquet` in the output directory and the
immutability guard at the top of `main` then refused the retry — a control
failure would have burned a version number. Nothing is written now until the
control passes. And the failure message no longer claims to prove the index's
declared splits: one `splits` value feeds both the build and the check, so
what it proves is that the indexing pipeline covered what it was handed.

## Bolmo: source-aligned optimizer, and why its BPB was not a codelength (2026-08-08)

Four things landed together: a Bolmo arm trained under the source model's own
optimizer, an annealed source baseline, an exact accounting of the batch-size
asymmetry, and — the one that governs how any of it may be read — the finding
that Bolmo's reported `byte_bpb` is not a bits-per-byte a subword model can be
compared against.

### The source-aligned arm

The epoch-matched Bolmo arm trailed its source model by 0.0805 BPB under the
paper's AdamW recipe. That comparison was confounded: the paper's Stage-2 rates
(2.6e-5 global, 5.2e-5 local) are Table-8 values for a 1B model on a much longer
schedule, and at 50M parameters over 2,400 updates they simply starve it.
`pretraining/source_muon.py` is a single-process port of the source trainer's
`Muon`/`PerHeadMuon` — the quintic Newton-Schulz iteration, the
`max(1, rows/cols) ** 0.5` rectangular scaling, Nesterov momentum and decoupled
decay, unchanged — and `BOLMO_OPTIMIZER_RECIPE=source` runs Bolmo under it with
the source's learning rates, cosine schedule, 1% warmup, momentum warmup and no
gradient clipping. `SourceOptimizerRecipe` binds all of it, plus
`SOURCE_MUON_ALGORITHM`, into the training contract.

Endpoints on the same one epoch, same data, same 2,400 updates:

| arm | stage 1 (800) | final (2,400) byte | final joint |
|---|---|---|---|
| paper AdamW | 1.6414 | 1.4816 | 1.6276 |
| source Muon | 1.4778 | 1.2369 | 1.3629 |

The recipe is worth more than anything else measured on this model. Note the
stage-1 endpoint alone (1.4778, frozen trunk) already beats the paper arm's
full 2,400-step result. Boundary accuracy improved 0.9793 to 0.9839, still under
the 0.99 gate we have been overriding.

This is *not* clean evidence that Muon is a better byteification optimizer. The
paper arm's Stage-2 trunk LR is 2.6e-5 AdamW against 0.025 Muon here, so the
paper arm's trunk is effectively frozen while the source arm's is genuinely
being pretrained for another 419M tokens. Most of that 0.265 gap is probably
"training the trunk beats not training it". Isolating the optimizer needs a
paper-recipe arm at matched trunk-update magnitude.

### The schedule was not matched, and the source baseline was mid-anneal

The source run was `ITERATIONS=2000 STOP_AFTER_STEP=1000`: half of one cosine,
killed at multiplier 0.5087, never annealed. Its val BPB was flat at 1.397-1.403
from step 840 on — a plateau at half peak LR, not a converged number. Bolmo, by
contrast, restarts its cosine at the stage boundary (`train_bolmo.py` rebuilds
the optimizer with `planned_steps=planned_stage2` and
`stage_step = global_step - planned_stage1`), so it gets two complete anneals to
zero. That asymmetry runs entirely in Bolmo's favour.

`k3_v8_armC_kda8_cosine_nextlat_nope_anneal1k` is the control: identical command
with `--steps 1000` and no `STOP_AFTER_STEP`, so the cosine completes over the
same 524,288,000 tokens. **1.4011 to 1.3777.** The anneal is worth 0.023 BPB,
close to the 0.009 measured on the source arm's own stage-1 tail. Per domain:
web 1.2236 to 1.2043, code 0.8494 to 0.8302, knowledge 1.0447 to 1.0307, math
0.8232 to 0.8248 (math alone did not move).

Every future Bolmo-versus-source comparison uses 1.3777, not 1.4011.

### The batch asymmetry is exact, and it is not extra data

Source `global_batch_tokens` is 524,288 (16 microbatch sequences x 16
accumulations x 2048). Bolmo's examples are also 2048 source tokens, at 64 per
step in stage 1 and 128 in stage 2 — a quarter and a half of the source batch.
So `800*64 + 1600*128 = 256,000` examples = 524,288,000 source tokens, exactly
the source's `1000 * 524,288`. The 2.4x update count is purely the batch ratio;
there is no extra data. But it does mean the "source-aligned" arm is aligned on
the LR *numbers* and not on LR-relative-to-batch, which is what actually
transfers. `bolmo_srcopt_bs256_*` fixes that: 1,000 total updates, 333/667
stage split, 256 examples in both stages, which consumes the same 256,000
examples exactly and needs no dataset rebuild.

### `byte_bpb` is not a codelength

`NonCausalBoundaryPredictor.forward` computes `log_p[t]` from `hidden[t]` and
`hidden[t+1]`. The local encoder is causal, so `boundaries[t]` is a function of
byte `t+1`. `prepare_hidden` routes position `t` to patch
`cumsum(boundaries)[t] - 1`, which consumes `boundaries[t]`. Position `t` scores
byte `t+1`. So every byte's score is conditioned on a bit derived from that
byte. The marginal does not normalize: substituting all 257 atomic values for
the target and summing the assigned marginal gives 1.0218, and the boundary
flips for 126 of 257 candidates. Frozen to a single causal routing tensor it is
1.000000.

This is inherited, not a porting bug. The paper is explicit — the predictor
"has access to one byte of future context", and Section 3.1.1 accounts for it as
"the single bit of information leaked by discrete boundary predictions", the
argument being that end-to-end training of the same predictor would leak 16
bits and collapse. `test_boundary_predictor_uses_next_byte_and_forces_bos` has
asserted the lookahead all along; what was missing was the consequence for the
metric.

What is *not* inherited is comparing that number to a subword model's BPB. The
paper's own prefill/decode split says which regime is which: the non-causal
predictor tokenizes the prefill, and during decoding boundaries are emitted as
the fused `<b>` symbol. `joint_bpb` is that decoding regime — the routing bit at
`t` was transmitted at `t-1`, `x -> (x, b(x))` is injective, so it is a valid
codelength, deliberately loose. `byte_bpb` is the prefill regime applied to
every scored position, which is teacher forcing with lookahead.

`validation_statistics(causal_routing=True)` is the new diagnostic. It keeps the
predicted boundaries and the pooled patch contents exactly as they are — `pool`
still receives the unshifted mask, and every patch end it selects is causal —
and shifts only the *routing* mask right by one, so a position reads the most
recent patch that closed strictly before it. Everything scoring byte `t+1` is
then a function of bytes `0..t`, the predictor is a deterministic function of
those bytes so a decoder can recompute it, and marginalizing the output bit is
legitimate. It charges a train/eval mismatch on top of the leak — the model was
trained with the non-causal routing — so it is an upper bound on what a
causally-routed model would cost, and `noncausal_routing_credit_bpb` in
`eval_bolmo_patching.py` reports the difference.

Restated on the numbers that are codelengths, against the annealed source:

| | byte (invalid) | joint (valid, loose) | source |
|---|---|---|---|
| paper AdamW | 1.4816 | 1.6276 | 1.3777 |
| source Muon | 1.2369 | 1.3629 | 1.3777 |

The source arm's margin over the source model collapses from 0.141 to 0.015,
and the paper arm goes from 0.104 behind to 0.250 behind. Nothing here supports
a byteification-beats-tokenization claim yet. (Both left-hand figures are
against the annealed 1.3777, per the rule set above; an earlier revision of
this paragraph quoted 0.164 and 0.08, which are against the superseded
un-annealed 1.4011.)

### The remaining confound, and the control the paper itself runs

Bolmo's trunk was pretrained on these 524,288,000 tokens and Bolmo then trains
on the same 524,288,000 tokens. The trunk sees the corpus twice; the source saw
it once. Appendix A of the paper runs exactly the right control — Bolmo against
the source model given continued training on the same data under the same
settings — so `k3_v8_armC_kda8_cosine_nextlat_nope_contd2k` resumes the source
run and completes steps 1001-2000 of its original cosine. That is the same
second epoch, fully annealed, at token level. Until it lands the defensible
claim is "the source trunk plus a second annealed epoch, in byte space", not
anything about byteification.

Also unresolved and pointing the same way: nextlat. The source's auxiliary was
about 12% of its training objective (`nextlat_total` 0.3155 of `train_loss`
2.7063) and does not transport — its KL term runs through the 50,304-way
unembedding Bolmo deletes. The trunk Bolmo inherits was shaped by it and both
source baselines had it active throughout.

Audited and clean: the validation span is byte-identical across all three
artifacts (9,081,780 scored bytes, recomputed independently from the byte
dataset and from the token shard, and both Bolmo arms report the same step-0
BPB of 8.13146); no train/val contamination, since the split is document-level
by hash after dedup and the val shard is the held-out FineWeb shard; both Bolmo
arms consumed identical data; `evaluate` sets `eval()`/`train()` correctly with
no cached batches; and the source's nextlat/MTP auxiliaries are `self.training`
gated, so its own BPB is pure CE. All of it is n=1 per arm with no seeds.

Review of the diagnostic found the routing fix itself correct — flipping the
final byte of a row moves the scored logits by 1.48 under the old rule and by
exactly 0.0 under causal routing, verified through every path: the
suffix-matched encoder inputs, the mLSTM recurrences, `pool`'s `argsort`
left-packing (prefix-stable, so the first patches are bit-identical across
masks agreeing only on a prefix), and the KDA trunk. `_fused_targets` keeping
the *unshifted* boundaries is also right: the fused head's label is the true
boundary bit of byte `t+1`, not the bit used for routing, and substituting the
routing mask there would be the bug.

Three defects were real and are fixed. The end-to-end test was vacuous: the
predictor's identity-initialized projections leave an untrained tiny encoder's
cosine saturated on one side of `log 0.5`, so its thresholded mask never moved
for any byte or seed and the assertion passed with `causal_routing` doing
nothing — for three of four seeds the mask was a single patch, where the two
rules cannot differ even in principle. It now randomizes those projections to
make the decision live, taps the real decoder rather than a fake one, and
asserts both directions. Nothing had exercised `prepare_hidden` under the
shifted mask, where `clamp(min=0)` became load-bearing — the unshifted mask's
forced leading boundary never produced a negative index, so deleting the clamp
would have broken only the causal path, on GPU, silently. And
`causal_routing` combined with the oracle, uniform or fixed-stride patchings
now raises: those masks are whole-row quantities, non-causal by many bytes
rather than one, and shifting their routing would not make them codelengths.

Left as-is, pre-existing and unchanged by this work:
`_force_well_formed_boundaries` never marks a row's last valid byte a boundary
because its lookahead score is `-100_000`, so the trailing partial patch is
never pooled and tail positions read the last complete patch. The causal run
inherits that.

## Bolmo does not beat its source: the margin was update count, and the comparison was never capacity-matched (2026-08-08)

The earlier reading — Bolmo's 1.3629 joint BPB against the source's annealed
1.3777 — does not survive its controls. Three independent findings remove it,
and two of them cannot be fixed by re-running anything.

### The margin was optimizer steps

All arms consume exactly 524,288,000 source tokens of the same corpus in the
same order.

| run | updates x global batch | valid codelength (joint bpb) | marginalized byte bpb |
| --- | --- | --- | --- |
| source, annealed, 1 epoch | 1000 x 524,288 tok | **1.3777** | - |
| Bolmo `srcopt_bs256`, 1 epoch | 1000 x 524,288 tok | **1.3927** | 1.2640 |
| Bolmo `srcopt_v2`, 1 epoch | 2400 x 131,072/262,144 tok | **1.3629** | 1.2369 |
| Bolmo `epoch1` (paper AdamW), 1 epoch | 2400 x 131,072/262,144 tok | 1.6276 | 1.4816 |
| source, 2 epochs | 2000 x 524,288 tok | **1.3304** | - |

At the source's exact batch and step count, Bolmo's codelength **loses by
0.0150**. At 2.4x the updates on identical data it wins by 0.0148. The win and
the loss are the same size; the effect is update count, not byteification.
`bs256` still carries the per-stage LR restart, which favours Bolmo, and loses
anyway.

The source is also nowhere near converged: a second epoch takes it to 1.3304,
below every Bolmo arm. Job 1499 queues the remaining control — the source at one
epoch and 2000 updates (half batch), via the new `GLOBAL_BATCH_TOKENS` override,
which is the first time this trainer's global batch has been anything but a
hardcoded `8 * 64 * 1024`.

### The comparison was never capacity-matched

Counted from the two checkpoints:

| | source | Bolmo `srcopt_v2` |
| --- | --- | --- |
| total parameters | 64,048,446 | 50,433,886 |
| vocabulary I/O | 51,511,296 (80.4%) | 26,157,568 (51.9%) |
| **modelling (non-vocab)** | **12,537,150** | **24,276,318** |

`embed.weight` and `proj.weight` are untied — separate tensors, not `allclose`,
different data pointers — and this trainer has no tying support at all
(`TIE_EMBEDDINGS` lives only in `train_gpt.py`). Bolmo's `global_blocks` is
12,485,822 parameters, the source trunk byte for byte; on top of it Bolmo adds
2,935,328 encoder and 8,855,168 decoder parameters and deletes a 25.8M-parameter
output projection that does no modelling. **Bolmo is the source's entire trunk
plus 94% more compute-bearing capacity**, at 21% fewer total parameters, and
those new parameters run at byte resolution (~4.56 bytes/patch), so the FLOP
increase is larger still and is unmeasured.

The paper holds this fixed: Bolmo 1B is -0.7% total parameters against OLMo 2
1B, Bolmo 7B +4.5%. Here it is -21.3% total and +94.4% modelling. At
`model_dim=512` with a 50,304-way untied head the head is 2.07x the modelling
stack; the paper notes the softmax only begins to dominate a 1B model somewhere
between 200k and 400k vocabulary. We are outside their regime, and
"byteification at matched capacity" is not what this measures.

### The source pays an unmeasured segmentation tax

A subword model's reported BPB charges `-log P(canonical tokenization | text)`,
not `-log P(text)`. That inflation is never measured. Structural bound over the
validation span: ~3.4 vocabulary tokens prefix the upcoming bytes at each token
start (only 11.4% of starts unambiguous), giving a ceiling of 0.3862 bits/byte.
The bound is loose — a trained model concentrates on the canonical segmentation
— but the direction is fixed: the source's true codelength is *below* 1.3777, so
correcting for it widens the source's win. Bolmo's analogous tax is measured and
paid (`joint - byte` = 0.1287 on `bs256`). The same asymmetry exists in the
paper's own comparisons, where Bolmo still trails; it flatters byte models
generally and does not explain why ours had looked different.

### `joint_bpb`, not `canonical_bpb`, is the comparand

Traced through the code rather than inferred. `pool` selects the **last** byte of
each patch, so `pooled[k]` depends only on bytes up to that patch's end;
`prepare_hidden` routes position `t` to `cumsum(boundaries)[t] - 1 <= t`. Neither
leaks content. Position `t`'s logits are scored against
`(byte[t+1], boundary[t+1])`, so `boundary[t]` — the bit routing consumes at `t`
— was charged at `t-1`. `joint_bpb` is therefore a genuinely causal codelength,
which is exactly the paper's Boundary Symbol Fusion, and it is tight. Only
`byte_bpb` is invalid: marginalizing the boundary means never paying for a bit
routing still reads.

`causal_routing_bpb` is 3.4159 (`srcopt_v2`) and 3.6611 (`epoch1`) — about 2.2
bpb above the marginalized number on both arms. That is almost entirely
train/eval mismatch, not the leak: forcing routing one patch stale changes what
every position reads. It is a valid upper bound and a useless one. The earlier
framing of it as the primary codelength was wrong, and the driver docstrings and
`pretraining/README.md` are corrected accordingly.

Oracle patching is *worse* than predicted under the valid metric (joint 1.4464
vs 1.3629 on `srcopt_v2`), because the joint charges for whichever boundary
sequence is transmitted and the model finds its own cheap. The oracle is not an
achievable ceiling for a causal predictor.

### Baseline handicaps found, and what was ruled out

The baseline is untuned, not sabotaged. `NUM_LAYERS=8` and `MLP_HIDDEN=2070` are
hardcoded literals in `run_k3_context_curriculum.py:365,369` inherited from the
16MB/10-min FineWeb lineage; every LR, WD and momentum is a trainer default; no
mlq job in this repo has ever used `--sweep`; arm C was chosen for build-ordering
risk and arms A/B/D were never built. Its two distinguishing flags were never
validated together — on the v7 factorial (byte-identical configs, same step-0
val_bpb 3.5345) NoPE alone cost +0.0014, NextLat alone cost **+0.0225 and 2.2x
wall clock**, and the stacked cell was killed at step 10 and never measured. The
source stacks both. Different dataset, so the sign is not guaranteed to carry.

The tokenizer's compression advantage inverts on the evaluation distribution.
Measured on the identical 9,081,775-byte window: GPT-2 needs 2,053,772 tokens
(4.4220 bytes/token), ToaST+TST 50k needs 2,097,152 (4.3305) — ToaST is **2.1%
worse** at the same 50,257 vocabulary. The recorded +13% was measured on a
domain-weighted k3 sample; the validation shard is raw FineWeb and ToaST's
`numeric.group_size: 1` was tuned for a math-30% mixture. BPB is designed to be
tokenizer-independent so the BPB impact is *not* established — but the source
pays whatever misfit exists and the byte model is structurally immune, and the
GPT-2 control (arm B) was never built.

Ruled out, with citations: every other `train_steps` coupling (weight decay is
constructor-only, momentum warmup is absolute-step, `seq_len` is read once and
never reassigned, no batch ramp, no EMA, auxiliary weights are module constants,
`MTP_NUM_HEADS=0`); `WARMDOWN_ITERS` is inert in this trainer; data ordering
(both runs consumed identical tokens in identical order, and the corpus is
exactly one 1000-step epoch); validation-loop defects (NextLat and MTP both gate
on `self.training`, so `eval()` is pure CE; the domain path uses the identical
formula); NoPE extrapolation (train and val both 2048, no ramp); dtype (KDA
kernel flags identical across both trainers, corroborated by Bolmo's stage-1
oracle 1.3799 reproducing the source's 1.3777); vocabulary under-utilization
(1,654 of 50,304 ids unseen); validation-set OOD-ness (symmetric — Bolmo is
scored on the identical shard). Adversarial tuning is ruled out chronologically:
the source finished 2026-08-06 19:47, the first Bolmo run started 21:09, and both
post-hoc reruns move the baseline *down*. One asymmetry runs the other way —
Bolmo's trunk is initialized from the weaker 1.4011 weights while being compared
against 1.3777.

No second conditioning leak exists. Row geometry is bit-exact against the raw
shard for all 1024 rows, the BPB denominator is byte-identical at 9,081,780 on
both sides, and the whole-row and row-length channels are closed.

### Defects fixed in this pass

The uniform floor was count-matched to the **oracle** while being used to
bracket the **predicted** arm, which spends about 5% fewer patches — so the
floor was systematically finer than the arm it bounded. `uniform_patching` now
names the arm whose count it matches (`"oracle"` or `"predicted"`), the driver
reports both floors, and `learned_predictor_placement_gain_bpb` compares the
predicted arm only against its own count. The floor was also documented as
"content-blind"; it is not, since the per-row patch count is content-derived
from either source. It is placement-blind, and both docstrings now say so. The
recovery ratio remains a rough guide: its endpoints sit at two different patch
counts and its three terms carry different amounts of non-causality.

`eval_bolmo_patching.py` had weaker provenance gates than the trainer — no
`validate_paper_data_manifest` and no check of the dataset hash against the
checkpoint's `data_manifest_sha256`, so a mismatched pair produced numbers
silently. Both are now enforced.

### Remaining exposure

Not closed: `fineweb_edu_dedup` is ~37% of training and is a filtered FineWeb
subset, and the validation shard never passes the deduplicator. An empirical
64-token n-gram scan found 18/124,997 collisions (1.4e-4), all web boilerplate.
This would inflate Bolmo and the source equally so the head-to-head is safe, but
the absolute BPB is not a clean held-out number. Separately, the canonical
validation span is 100% web while training is 42/30/14/14 web/math/code/knowledge;
the `domainval_*` shards exist and are unused.

The Stage-1 go/no-go gate reads `canonical_val_rows`, which is model selection on
the evaluation set. Exposure is one binary decision with no best-checkpoint
selection, and the 0.99 gate was overridden on both arms, so it was not binding.
It should still be re-specified against a held-out slice or retired.

Boundary bits over the unscored context prefix are supplied free: 5,453 bits over
the corpus, 0.0006 bpb. Negligible but real.

### What survives

Byteification transfers the trunk faithfully — Bolmo's stage-1 oracle BPB of
1.3799 reproduces the source's 1.3777 — and reaches a comparable codelength while
spending its parameters very differently. "Bolmo produces a shorter code for this
text than its source" is false at matched updates. "Bolmo models this text better
than its source" was never supportable at this scale, because the two models do
not have comparable modelling capacity and the source's BPB is inflated by an
unmeasured tax. This is consistent with the paper, which reports Bolmo
*approaching* its source and attributes the residual gap to boundary-predictor
error (S6.1).

### Review addendum to the entry above (2026-08-08)

An adversarial review of the fixes found nine further defects. All are corrected.

In `_uniform_boundaries`, the claim that a count match leaves "only placement"
as the variable is false: the floor always closes the last valid byte, and
`_force_well_formed_boundaries` never does, so the floor tiles a row the
predicted arm leaves with an unpooled tail patch. Under 0.1% of a row, but the
count-match framing made it newly load-bearing, so the docstring now says
placement *and* tail-closure convention. The precondition
`remaining_patches >= 1` is also newly reachable: predicted counts bottom out
at 1 for a checkpoint whose predictor never fires, which oracle counts cannot
do, so the error now explains that. `uniform_patching not in (\n X\n)` was bare
grouping that a trailing comma would silently turn into a one-tuple rejecting
every valid value; rewritten. `eval_bolmo_patching.py` now asserts each floor's
`bytes_per_patch` equals its arm's — the check that would have caught this bug
in the first place — and records `data`, `data_manifest_sha256`, `scored_rows`
and the token budget in the summary, since validating provenance without
recording it leaves the artifact unattributable.

The new test survived a mutation replacing `_force_well_formed_boundaries`
with a bare threshold, because its fixture had no padding and already fired at
position 0 — the two things that function exists for. It now also asserts on a
padded row whose predictor is silent at position 0 and active inside the pad
region, where the well-formed count is 3 and the raw count 4. That mutant now
fails.

`GLOBAL_BATCH_TOKENS` had three holes. Resume never checked it, although the
trainer has always written `global_batch_tokens` into the checkpoint: since
`SEQ_LEN` and `MBS` deliberately vary across a curriculum resume, the global
batch is the load-bearing invariant, and resuming under a different one
re-indexes the data stream at `start_step * batch_size` and rescales the
schedule. Guarded. `run_k3_context_curriculum.py` copied the ambient
environment and popped only `RESUME_CHECKPOINT`, so an exported override would
have halved a campaign whose own `expected_batch_tokens = 524_288` assert
still claimed it had not; it now pops `GLOBAL_BATCH_TOKENS` too. And `0` and
negatives passed `batch_size % (world_size * mbs * seq_len) == 0`, which the
lazy data generator would have turned into a bare `ZeroDivisionError` after
model init, compile warmup and step-0 validation; rejected up front.

`train_bolmo.py` still described the causal-routing pass as "the number to
compare against a subword model's bits-per-byte", the framing this entry calls
wrong, and `eval_bolmo_patching.py` still offered `causal_routing_bpb` as a
co-equal comparand. Both now say `joint_bpb` is the tight comparand and causal
routing is a loose bracket. `learned_predictor_recovery` is already 1.012 and
1.005 on the two measured arms — above the 1.0 its comment called full recovery
— because the oracle is not a ceiling here; the comment now says so.

Documentation errors corrected: the paragraph restating margins "against the
annealed source" quoted 0.164 and 0.08, which are against the superseded
un-annealed 1.4011; they are 0.141 and 0.104. `pretraining/README.md` named
`val/joint_bpb`, which is the 256-example proxy's joint, where it meant
`val/canonical_joint_bpb`. The math-30 profile's web share drops from 50% to
42%, not 60% — `k3_weights_quality.json` is `0.5`. The 4.0001-bytes-per-token
and 13%-compression figures came from a superseded 95.3 MB exploratory sample;
the shipped artifact measures 3.7288 against GPT-2's 3.4673, i.e. 7.54%, which
`tokenization/README.md` had already retired and the other two documents had
not. No 16,384 artifact exists, so arm D has no measured compression at all.
`k3_weights_math30.json` repeats the 60% error in its `description`, but it is
bound by `weight_profile_sha256` in every built dataset, so it is left alone
and wrong rather than silently re-hashed.

Not fixed, recorded instead: `pretraining/README.md` prescribes
`--max-train-examples 240000` (`..._paper_v3`) while every checkpoint behind
the tables above carries `..._byte2048_epoch1` at 256,000, and `ExampleStream`
uses `take_exact` with `repeat=False`, so the documented command does not
reproduce the reported runs. `data/problem_registry/v1/overlap.json` is absent,
so the admission-rate table in this file rests on no artifact on disk.

### The learning-rate reset was not the confound (2026-08-08)

`bolmo_armC_srcopt_cont` repeats the matched-batch arm (256 examples both
stages, 1000 updates, 524,288 tokens per update, one 524,288,000-token epoch)
under a single continuous cosine across the stage boundary instead of a
per-stage restart. Only `BOLMO_SOURCE_SCHEDULE_SCOPE` differs.

| matched arm | canonical byte bpb | canonical joint bpb |
| --- | --- | --- |
| per-stage restart (`bs256`) | 1.2640 | **1.3927** |
| continuous (`cont`) | 1.2672 | **1.3960** |
| source, annealed | - | **1.3777** |

The continuous schedule is *worse* by 0.0033. It led at every intermediate
step — proxy joint delta -0.1242 at step 340, narrowing monotonically to
-0.0047 at 860 — and gave the lead back over the final anneal, because the
per-stage arm runs a complete cosine inside stage 2 while the continuous one
is already in its tail. Stage-1 canonical was 1.5162 against 1.4913, as
expected when stage 1 ends at a 0.76 multiplier rather than annealed.

So the schedule question that motivated this arm is closed in the opposite
direction from the concern: the restart was worth 0.003 bpb and favoured
Bolmo's competitor arm, not Bolmo. Both matched arms lose to the source, by
0.0150 and 0.0183. The `srcopt_v2` margin of 0.0148 was update count, not
schedule.

Everything else agrees across the two arms — boundary accuracy 0.9836 both,
bytes per patch 4.5659 against 4.5675, causal 3.4618 against 3.4555 — which is
the check that the schedule was the only variable.

## 2026-08-08: MathGLM added, drills rewritten, and a validator that did not validate

Multiplication scored 0% at every digit width in the arithmetic probe. That was
a data defect, not a capacity limit. The generated drill corpus under schema v2
stated every product as a single jump -- `2933 x 3 (ones) = 8799`, one line, no
algorithm -- and `mul_integer` capped the multiplier strictly below the
multiplicand, so no equal-width product (2x2, 3x3, 4x4) existed anywhere in two
million drills. `digit_order` was inert for multiplication: 62,988 `forward`
and 62,895 `reversed` rows rendered identical bytes. Measured over 20,000
multiplication drills:

| drill schema | times-table facts | distinct digit pairs |
| --- | --- | --- |
| v2 | 0 | 0/100 |
| v3 | 72,926 | 100/100 |

Schema v3 decomposes each partial product into single-digit column facts with
carries and combines the partials through `column_addition`, keeps a genuinely
different expanded/distributive form for `forward`, lets the multiplier reach
the multiplicand's width, and drops `div_integer` to one digit. Each working
method draws its lead sentence from a phrasing pool keyed on the operands, so
the corpus no longer opens a quarter of a million drills with one identical
line (`mul_integer` 1 -> 1,477 distinct leads, `div_integer` 1 -> 2,958).

### The probe panel was scoring trained items

`run_arithmetic_probe.py` defaulted to `data/math_drills/v1/probe.jsonl` long
after the generator had changed. 84 of its 1,920 items (4.4%) appear verbatim
in the current training stream. The panel recorded `disjoint_from_training:
true` without saying *from what*, so the existing check passed. Panels now
carry `drill_schema` and `drills_sha256`; the probe refuses a panel that does
not name its corpus, or whose digest disagrees with the manifest beside it.
`data/math_drills/v4` exists only for that: its `drills.parquet` is byte-
identical to v3 (`e9c9c476e4cd68cf`), which also confirms the generator is
reproducible.

### MathGLM as a pretraining corpus

`jonathanasdf/MathGLM-dataset-5M` supplies what the drills lack: 1 to 26
chained operations under precedence (95.8% at 1-9), multiplication that is
66.9% four-digit by four-digit, and operators past the four basics (`[]`
grouping, `^` powers, postfix `%` as hundredths, so `4.0/1%` is 400). Every
upstream row fails the corpus quality gate on its own -- 61% `too_short`, 39%
`low_alpha_fraction`, and *not one alphabetic character in the entire file* --
so `scripts/build_mathglm_corpus.py` packs chains under rotating natural
language framing (10 headers, 8 question forms, an optional step-count note)
and re-checks each document against that same gate.

### Negative result: four validators in a row that did not validate

The upstream arithmetic is not trustworthy, and neither were my first four
attempts to check it. Each defect was found by measuring the *output* of the
previous build, never by reading the code:

1. `MAX_EXPONENT = 64` rejected `1^2864=1` -- 37,204 valid rows discarded. The
   premise came from a regex that matched only a bare literal after `^`, which
   covered 39k of 162,403 occurrences; real exponents reach four digits and go
   negative.
2. Bounding result magnitude symmetrically rejected `46^-94` -- 64,275 rows,
   and it hid which of them were genuinely wrong. Only a large *positive*
   exponent can materialise an unbounded integer.
3. `abs_tol=1e-12` made every value below that floor compare equal to every
   other, so `46^-94 = 1/<121-digit denominator>` -- two numbers 36 orders of
   magnitude apart -- passed as a true statement.
4. Comparing through floats admitted 7,571 wide products wrong past the
   sixteenth significant digit, e.g. `385924542305736*3405=1314073066551031040`
   whose true value ends `031080`. That is precisely the multi-digit
   multiplication family the corpus was added to teach.

The rule that survives: every `=`-separated segment must evaluate to the same
value, whole integers compared exactly and everything else on relative
tolerance with no absolute floor. Over 5,000,000 rows it admits 4,918,912
(98.378%) and rejects 1.622% -- 57,796 self-contradicting chains, 23,290
unevaluable, 2 unparsed. Two upstream defect families account for the
disagreements: MathGLM's step generator mangles scientific notation, reducing
`0.018706333107955823-1.923635517236736e-06` to `0.018706333107955823-06` and
landing on `-5.98` where the truth is `0.0187`; and its fraction chains step
aside into scratch work, so `=` does not always join equal values.

Verified independently against the built corpus with exact integer arithmetic
sharing no code with the builder: 136,264 integer chains, 3,995 of them
carrying a value past 2^53, **zero** false equalities.

### Shipped

`data/mathglm/v6` -- 2,922,446 documents, 1,686,167,867 characters, 577 chars
each, `e74d7993cd1ae184`. `data/math_drills/v4` -- 2,000,000 drills, 1028.7 MB,
514 chars each, 1,920-item bound panel. Both registered in `k3_sources.json`.
`k3_weights_math30_mathglm.json` splits the arithmetic budget mathglm 6% /
drills 4%, paid out of `deepmind_math` (5% -> 2%) and `open_web_math`, holding
`openmath_instruct` at 8% as the only source carrying worked word problems.

At 1.375 chars/token mathglm is a 1,226M-token corpus and 6% of a 524M-token
run is 31.5M tokens, i.e. 2.6% coverage; drills at 2.092 chars/token are 492M
tokens and 4% is 21.0M, i.e. 4.3%. Arithmetic therefore takes 10% of the run
while both corpora stay largely unread -- the binding constraint is the run's
token budget, not the weights.

Superseded and not to be used: `data/mathglm/v1` and `v2` contain the false
arithmetic outright; `v3` and `v4` are lossy from defects 1 and 2; `v5` admits
the wide-product errors of defect 4. `data/math_drills/v1` through `v3` predate
the panel binding.

## 2026-08-18: DiffusionBlocks conversion of the KDA hybrid — design and pre-registration

DiffusionBlocks (Shing et al., ICLR 2026, arXiv:2506.14202; reference code in
`../DiffusionBlocks`) reinterprets residual connections as Euler steps of a
reverse diffusion over the hidden state and trains each block of layers as an
independent denoiser over its own equal-probability-mass slice of the EDM
lognormal noise schedule (P_mean -1.2, P_std 1.2, sigma in [0.002, 80],
sigma_data 0.5, gamma 0.1 log-overlap). Exactly one block receives gradients
per optimizer step, so gradient/activation/optimizer memory scales with
num_layers/num_blocks; the paper's LM adaptation noises the unit-L2-normalized
embedding of the *next* token and conditions on the clean prefix, reporting
LM1B PPL 14.58 -> 12.32 on a 12-layer Llama-2-style model with B=4.

New trainer: `pretraining/nanogpt_mini/nanogpt_mini_gpt2vocab_kda_dblock_train.py`
with importable math in `pretraining/nanogpt_mini/nanogpt_mini_dblock.py`.
Block partition follows the architecture: every maximal run of KDA mixers plus
the dense attention layer that closes it is one diffusion block (default 24L
3:1 schedule -> 6 blocks of KDA,KDA,KDA,dense). The paper's
sequence-concatenation trick assumes dense attention, so the recurrent mixers
required a new construction: the residual stream is [clean(T), noisy(T)];
dense layers use a flex_attention mask where the noisy slot for target i
(acting at RoPE position i+1) attends clean keys j<=i plus itself; KDA layers
interleave [c_0, n_0, c_1, n_1, ...] through one chunk_kda call with the
noisy slots' decay and beta logits forced to -30000, which the in-kernel
sigmoids turn into decay 1 / write 0. The clean state trajectory is therefore
bit-identical to a clean-only pass and each noisy slot reads the state after
its inclusive clean prefix (CPU-proved against the pure-PyTorch KDA oracle in
`pretraining/tests/test_nanogpt_mini_dblock.py`, 20 tests; CUDA kernel parity
in `pretraining/tests/test_nanogpt_mini_dblock_gpu.py`). Short convs give the
noisy branch the window [c_{i-2}, c_{i-1}, c_i, n_i]. Zero-init AdaLN
shift/scale from a DiT sigma embedder modulates only the noisy half, so clean
computation is sigma-independent and the trunk stays exactly paired with the
baseline's init RNG. Known deliberate deviation: read-only KDA slots drop the
delta-rule self-write a committed step would add before its own read; dense
keeps the self key; both are identical at train and inference time.

Validation is not training reward: it is the honest generative chain —
num_blocks Euler steps from pure per-row-seeded noise, each level routed to
the block owning its sigma, softmax-expectation denoising over the normalized
embedding table, and the final level's teacher-forced CE reported as
val_loss/val_bpb, directly comparable to the baseline trainer's numbers.

Hypothesis: at matched total FLOPs the blockwise regime trades some quality
for a large cut in training-step VRAM (roughly num_layers/num_blocks of the
grad-live stack, minus the 2x sequence width) and converts the freed memory
into larger microbatches and higher throughput. Success criterion: training
peak VRAM materially below baseline (>=2x) with chain val_bpb on the shared
fineweb panel within striking distance of the baseline's teacher-forced
val_bpb at matched FLOPs; failure modes pre-registered: chain CE dominated by
error accumulation across levels (visible in per-level CE telemetry),
low-sigma blocks starved by the EDM weight profile, or KDA read-only slots
providing too little context mixing for late blocks. FLOPs accounting: one
dblock step touches ~2T tokens x L/B layers ~ 1/3 of a baseline step at B=6,
so matched-FLOPs arms run 3x the steps; per-block update count is then still
half the baseline's per-layer count.

Arms (all on `data/datasets/fineweb10B_gpt2`, SEED 1337, global batch 524288
tokens, seq 1024, VAL_TOKENS 4194304, VAL_LOSS_EVERY 100, stable_linear):
A baseline trainer 24L 1000 steps (MBS 8); B dblock 24L B=6 3000 steps
(MBS 8, ~matched FLOPs); C dblock as B but MBS raised to exploit freed VRAM
(throughput arm). A GPU parity suite and a 12-step engineering shakeout
(shapes/compile only, not evidence) gate the arms; mlq jobs 3070/3071.

modded-nanogpt caveat, recorded up front: their records require the fixed
teacher-forced eval; the dblock chain changes evaluation semantics, so any
upstream submission is an experimental/discussion contribution, not a record
claim.

### Amendment (same day): schedule-matched arms

The original A(1000) vs B(3000) pairing confounded blockwise training with a
different LR-schedule horizon and 3x the consumed tokens (stable_linear
anneals over each run's own step count). Corrected design compares only
within matched-schedule pairs, all else identical (SEED 1337, data stream,
global batch 524288, seq 1024, panel, val cadence):

- 1000-step pair — A1000 baseline (mlq 3080) vs B1000 dblock (3083): the
  controlled test of blockwise training itself at matched steps, data, and
  schedule; dblock spends ~1/3 the FLOPs and trains each block ~167 times.
- 3000-step trio — A3000 baseline control (3086), B3000 dblock (3084;
  FLOPs-matched to A1000 but quality claims stay within this trio), C3000
  dblock MBS 32 (3085; throughput arm).

Jobs 3081/3082 were cancelled before start and resubmitted as 3084/3085.

### Amendment 2 (same day): microbatch shape and run naming

The first launcher used MBS=8 at seq 1024 (64 microbatches per 524288-token
step): 4.3 s/step at 54% GPU / 305 W — launch-bound, far off the machine's
known envelope (previous 524288-token runs at MBS=32 sustain ~1.25 s/step
near 500 W). All arms were cancelled (baseline had reached step 580; its
partial log logs/dblockA_base24L_1k.txt is retained but is not evidence) and
resubmitted at MBS=32, throughput arm at MBS=64. Names now separate the pure
AR control from DiffusionBlocks arms: ar_base24L_1k_mbs32 (mlq 3088),
dblock_db6_1k_mbs32 (3089), dblock_db6_3k_mbs32 (3090), dblock_db6_3k_mbs64
(3091), ar_base24L_3k_mbs32 (3092). MBS is a gradient-accumulation detail:
losses are token sums and chain-validation noise is per-absolute-row, so
results are MBS-invariant; only throughput and peak VRAM depend on it.

### Results: 1000-step matched pair, and diagnosis of the dblock chain

Completed runs: ar_base24L_1k_mbs32 (mlq 3088) and dblock_db6_1k_mbs32
(3089), identical seed/data/schedule/panel.

- Baseline: val_loss 3.59192 (1.1701 bpb) at step 1000; 2.38 s/step; peak
  14355 MiB allocated.
- Dblock: chain val_loss 8.327 at its step-300 minimum, then monotone
  worsening to 9.36522 (3.0508 bpb) at step 1000. Efficiency claims held:
  1.30 s/step (1.83x faster) and 8652 MiB training-window peak (~40% less
  allocated VRAM); the freed memory is real but currently buys nothing.

The failure is not distributed uniformly. Per-block training CE is healthy
to excellent (blocks 0/1 ~5.0-5.7 at high sigma; block 3 -> 0.86; blocks
4/5 -> ~0.05-0.07), and chain level 0 — whose input is pure noise, so its
CE measures context-only prediction by a 4-layer block — improves steadily
to 5.53. But mid/low-sigma chain levels sit at or far above the uniform CE
of 10.83 (level3 13.68, level4 16.88 at step 1000): those blocks are
confidently wrong on chain inputs while near-perfect on training inputs.

Working diagnosis — train/inference input mismatch, not mis-wiring. In
training, every block receives z = y + sigma*eps with y the true unit-norm
target embedding; at sigma <= 0.2 nearest-neighbour decoding over ~50k
near-orthogonal embeddings is trivial, so the low-sigma half of the network
learns to read the target out of its own noisy input (train CE 0.05)
instead of modelling context. At inference the Euler chain instead delivers
the previous level's softmax-expectation embedding; with per-token posterior
perplexity in the tens-to-hundreds that expectation is a near-zero-norm
mixture, and leak-trained blocks decode a wrong nearest token from it with
high confidence — CE above uniform, worsening as upstream blocks sharpen.

Literature check: the reference implementation's diffusion_step matches our
chain exactly (equi-mass grid, steps = blocks by default, expectation
denoising, final denoise at sigma_min), but the paper evaluates its AR
text models only with MAUVE and teacher-scored generative perplexity of
sampled text, stating that traditional perplexity "is not derived from
ELBO" and is non-trivial for the framework. Teacher-forced chain CE — our
pre-registered success metric — is a question the paper deliberately does
not answer, and its released code covers only image classification, where
sharp posteriors hide the mechanism.

Actions: queued 3k arms 3090/3091 and control 3092 cancelled before start
(no point burning ~5 GPU-h on a diverging recipe). Added an env-gated
DBLOCK_DIAG evaluation mode to the dblock trainer (subagent-reviewed; the
expectation-mode probe is a line-for-line replica of chain_validation_loss
and must reproduce val_loss as a self-check) and queued mlq 3102 on the
final checkpoint with four probes: per-level oracle inputs y + sigma*eps
(clears or convicts wiring), renormalized-expectation chain (isolates the
norm collapse), sampled-token-embedding chain (stays on the training
manifold; candidate cheap fix), and a 24-step fine chain (tests Euler
discretization coarseness). Verdict between "implementation bug" and
"method property under likelihood evaluation" is deferred to those probes;
the modded-nanogpt submission idea is off the table either way unless the
chain metric is repaired. Partial logs of the cancelled arms and the
superseded MBS=8 attempt remain non-evidence.

### Amendment 3: reference-code audit found two fidelity gaps; DiT arm queued

Line-by-line comparison of the trainer against the DiffusionBlocks reference
(vit.py / model.py) and the paper's Appendix E.4 ("augmented with time
conditioning as in DiT") found the port faithful on the chain protocol, EDM
coefficients, equal-mass bands, gamma overlap, CE-for-L2 swap, normalized
embeddings, and pre-combine layernorm — but unfaithful in two places:

1. The reference sigma-conditions the OUTPUT HEAD: forward_output_embeddings
   applies a per-sigma shift/scale (zero-init) to the EDM denoised estimate
   before the classifier. Our shared vocab head saw raw denoised vectors
   whose scale varies roughly 10x across blocks (c_out*normed-hidden at
   sigma 80 vs c_skip*z at sigma 0.002) with no adaptation mechanism.
2. The reference modulates EVERY token per layer (DiT adaLN on context and
   noisy alike, plus zero-init residual gates); ours modulated only the
   noisy stream, keeping context features sigma-independent.

Honest prior: per-block training CE was already excellent, so the head and
context conditioning evidently suffice in-distribution; these gaps are
unlikely to be the whole explanation for the chain divergence, whose
mechanism (expectation-embedding norm collapse feeding leak-trained blocks)
is orthogonal. But they are the two concrete infidelities a "faithful
implementation" claim must not carry, so they get an arm.

New env flag DBLOCK_DIT_FIDELITY=1 (default off; recorded in checkpoints,
resume-validated): per-layer AdaLN emits clean+noisy shift/scale pairs
(4*dim) and modulates both halves, and a zero-init head_ada
(BiasFreeLinear cond -> 2*dim, shared, Muon) modulates the denoised
estimate before proj. Zero init preserves the paired baseline init at step
0. Residual gates are deliberately not adopted: zero-init gates would zero
every layer output at init and destroy init pairing; noted as a remaining
deviation. Queued as dblock_db6dit_1k_mbs32 (mlq 3108), identical recipe to
dblock_db6_1k_mbs32 otherwise; runs after the diagnostic eval (mlq 3102).
Success criterion: materially better chain val trajectory than 9.365 at
step 1000; failure keeps the negative-result conclusion with the fidelity
objection removed.

### Diagnostic results (mlq 3102): wiring definitively cleared; chain is the failure

Self-check passed: the expectation-mode probe chain reproduced the run's
val_loss exactly (9.3652). Probe results on the dblock_db6_1k final
checkpoint, val panel, per-level CE (chain sigma levels 0..5 route to
blocks 0..5):

- Oracle inputs (z = y + sigma*eps at the exact chain sigmas): 5.53, 4.84,
  3.03, 0.54, 0.0057, 0.83. Every block is near-perfect on its training
  distribution at its inference sigma — no routing, mask, conditioning, or
  head defect. The chain CEs at the same sigmas are 5.53, 6.54, 8.81,
  13.68, 16.88, 9.37.
- Expectation telemetry: the chain estimate's cosine to the true target
  embedding starts at 0.293 after block 0 and DECREASES monotonically to
  0.123 by level 4 while its norm rises 0.25 -> 0.89. Every level after the
  first destroys alignment while gaining confidence; chain level-0 CE
  (5.53) is the best any level achieves. The diffusion refinement is
  strictly anti-productive for teacher-forced prediction.
- Renormalized-expectation chain: worse (final 9.41; level4 19.0). Norm
  collapse is not the binding constraint; direction is.
- Sampled-token chain: worse (final 9.55; level4 20.0; sampled-token cos
  0.096 at level 0 — the sample is nearly always wrong at high sigma, and
  low-sigma blocks lock onto it).
- 24-step fine chain: no better (final 9.53); cos telemetry plateaus at
  0.124 through all low-sigma levels — finer Euler discretization adds
  nothing because the blocks do not implement the true posterior mean on
  chain-distributed inputs.

Verdict: the implementation is correct end-to-end; the failure is the
method's training distribution. Blocks trained only on y + sigma*eps never
learn to refine realistic uncertain estimates, and the low-sigma half of
the network learns a nearly deterministic read of the leaked target that
is confidently wrong on anything else. Consistent with the paper avoiding
held-out perplexity for its AR models. The DBLOCK_DIT_FIDELITY arm (3108)
stays queued to close the conditioning-fidelity objection; given oracle
CE of 0.0057 in-distribution, extra conditioning capacity is not expected
to change the conclusion.

### Closure: DiT-fidelity arm cancelled before start

mlq 3108 (dblock_db6dit_1k_mbs32) was cancelled before start from outside
this session (0/1 attempts; the queue is shared and actively managed). Not
resubmitted. The negative-result verdict rests on the mlq 3102 diagnostics
alone, which already cleared the implementation and localized the failure
to the method's training distribution; the conditioning-fidelity objection
remains formally untested but is bounded by the oracle result (CE 0.0057
in-distribution — capacity is demonstrably not the constraint). The
DBLOCK_DIT_FIDELITY flag stays in the trainer, checkpoint-recorded and
resume-validated, should the arm ever be wanted.

### Chain-consistent training: the repair arm (mlq 3111)

The mlq-3102 diagnosis localizes the failure to the training input
distribution, so the repair changes exactly that and nothing else.
DBLOCK_CHAIN_TRAIN=1: for the step's sampled block b, each microbatch
Euler-propagates fresh noise (global training RNG) through inference
levels 0..b-1 with the current weights under no_grad, and block b trains
with plain CE on that exact chain state at its fixed inference sigma.
Properties:

- No leak by construction: the training z never contains the target
  embedding, only predecessor beliefs plus scheduled noise, identical to
  what validation hands the block. Train and eval objectives coincide up
  to noise seeding (validation keeps per-row panel seeds).
- Memory claim preserved: the prefix runs gradient-free; gradients and
  optimizer state still exist for one block plus shared modules.
- EDM weighting dropped in this mode (weight = 1): each block sees one
  fixed sigma, making the weight a per-block constant absorbed by the
  per-block optimizers. Band sampling and gamma overlap are unused.
- Startup invariants: DBLOCK_INFER_STEPS == NUM_BLOCKS and a verified 1:1
  level-to-block routing, since the per-step block sample doubles as the
  trained chain level.
- DBLOCK_COND_HEAD=1 accompanies it: the sigma-conditioned head shift/scale
  (0.5M params, ~0.5%) without DIT_FIDELITY's per-layer widening, which is
  degenerate under fixed per-block sigmas. Zero-init, shared, Muon-owned.
- Known risk, accepted: nonstationary inputs (block b's input distribution
  shifts as predecessors improve). Watch per-level val CEs for oscillation.
- Interpretation shift, recorded honestly: with chain-consistent inputs the
  model is a blockwise-trained 6-stage iterative refiner whose stages pass
  a 512-dim belief vector; the diffusion machinery is its parameterization.
  This departs from the paper (which never trains on chain states) and is
  our extension, not a reproduction.

Queued as dblock_chain_1k_mbs32 (mlq 3111), otherwise identical recipe to
dblock_db6_1k_mbs32 (SEED 1337, 1000 steps, batch 524288, seq 1024,
MBS 32, same panels and cadence). Hypothesis: chain val_loss now tracks
training and improves monotonically; success = materially closing the gap
toward the AR baseline's 3.592 (the 9.365 of the leak-trained arm is the
floor to beat by a wide margin). Expected cost: prefix adds ~2.5 no-grad
block forwards plus head/expectation per step; step time should stay under
the AR baseline's 2.38 s.

### Fused-cascade chain training (DBLOCK_CHAIN_TRAIN=all, mlq 3112)

Interim 3111 verdict at step 800: the mechanism works — chain val_loss
tracks training and falls monotonically (7.696 @100 -> 6.499 @800 vs the
leak arm's 9.82 @100 diverging), already far below the leak arm's
best-ever 8.33. Two problems, both structural to one-block-per-step:

- Cost estimate was wrong: step_avg ~3.9-4.2 s (vs 1.30 s leak arm,
  2.38 s AR baseline). The prefix is not "cheap no-grad forwards" — each
  prefix level pays a 50k-vocab head projection plus fp32 softmax and
  expectation GEMM (~40 head+expectation evaluations per step at MBS 32),
  and it buys gradient signal for only one block per step.
- Final-block lag: levels 0-4 improve fast (CE ~4.6-4.8 @800) but block 5
  — trained on ~1/6 of steps against predecessors that keep moving — makes
  the level-4 estimate *worse* (4.76 -> 6.50). The graded output is
  bottlenecked on the least-trained, most-nonstationary block.

DBLOCK_CHAIN_TRAIN=all fixes both with one rollout per microbatch that
trains every block in level order: block b's logits are computed with
grad, graded with plain CE, backpropagated immediately (one block's graph
alive at a time — the activation-memory claim survives), then reused
detached for the Euler step producing block b+1's input. The prefix is
never recomputed; every block sees every batch (baseline update density);
all shared+block optimizers step each step. denoised_expectation now does
the softmax in fp32 and the vocab contraction on bf16 tensor cores
(bounded ~2^-9 relative error on unit-norm expectations; train and val
share the operator, so chain consistency is preserved by construction —
val numbers before/after this change differ imperceptibly but are not
bit-identical). Config records dblock_chain_train as the string
"0"/"1"/"all"; resume rejects bool-era checkpoints loudly (3111's
checkpoint is a historical control, not a resume source).

Adversarial review (subagent, full checklist): no critical/moderate
findings; gradient isolation across levels, level-to-block invariant,
all-param grad coverage, parser-safe logging (ce_l0..ce_l5), and loud
resume rejection all confirmed. Flagged transient-VRAM risk of the
uncompiled CE (three fp32 logits-sized buffers per level) is why the
cascade runs MBS=16 (identical gradients and global batch; ~14 GB peak
instead of ~27 GB on the 32 GB card).

Queued as dblock_chainall_1k_mbs16 (mlq 3112) behind 3111 (which runs to
completion as the one-block-per-step control), otherwise identical recipe
(SEED 1337, 1000 steps, batch 524288, seq 1024, DBLOCK_COND_HEAD=1, same
panels/cadence). Hypothesis: per-step val improvement at least matches
3111 with level 5 no longer degrading level 4 (watch val_level4 vs
val_level5); step time expected between 3111's ~4 s and ~2x that (six
grad blocks + six heads per microbatch, minus the redundant prefix) — to
be measured, not assumed. train_loss in this mode is the unweighted
level-mean CE (not comparable to mode-0's EDM-weighted train_loss);
train_ce remains the final-level CE, unit-comparable to val_loss.

### Result: dblock_chain_1k_mbs32 (mlq 3111, one-block-per-step control)

Completed 1000/1000, checkpoint logs/dblock_chain_1k_mbs32_final_model.pt.
Final chain val_loss 6.16985 (bpb 2.0099), step_avg 3880 ms, peak alloc
17.3 GiB. Monotone val trajectory: 7.696 @100, 7.281 @500, 6.499 @800,
6.170 @1000 — chain-consistent training decisively repairs the leak-arm
divergence (9.365 final, never below 8.33). Still 2.58 above the AR
baseline's 3.592, and the signature bottleneck held to the end: levels 0-4
plateau near CE 4.55-4.71 while the final graded block *degrades* its
input estimate (level4 4.650 -> level5 6.170), consistent with block 5
receiving ~1/6 of the updates against a nonstationary predecessor chain.
This is the control the fused cascade (mlq 3112) must beat: same recipe,
all blocks updated every step.

### Closure 3112, relaunch as fused-tail cascade (mlq 3113)

dblock_chainall_1k_mbs16 (3112) measured 13.5 s/step marginal (steps
20->40) — the eager fp32 cross-entropy plus the separate eager expectation
pass over the 50304 vocab, run 6x per microbatch, cost ~6 s/step on their
own. Its 60 steps already showed the cascade's qualitative win: per-level
train CEs are ordered and healthy with no final-block cliff (ce_l4 6.95 ->
ce_l5 7.67 at step 60, versus 3111's persistent +1.5 val gap). Killed at
~step 65 as an engineering iteration, not a scientific arm.

Fix: `loss_tail_chain` — a compiled tail (rebound next to loss_tail)
computing the level's plain-CE sum and, except at the final level, the
detached posterior expectation (fp32 softmax -> bf16 probs @ bf16 table ->
fp32) in one fused softmax pipeline; `dblock_cascade_level` orchestrates
_run_block + tail per level. The normalized bf16 table is hoisted to
once per step. Numerics identical to eager `denoised_expectation` up to
GEMM tiling order (both fp32-accumulated); validation and diagnostics
keep the eager operator unchanged. Focused adversarial review (subagent):
no findings — gradient isolation, two-bool compile specialization,
level->block invariant, and train/val operator parity all confirmed;
fused-tail peak memory at or below the eager path.

Relaunched as dblock_chainallf_1k_mbs16 (mlq 3113), recipe identical to
3112. Expectation: ~4-6 s/step saved vs 3112; the remaining cost (six
2-layer two-stream grad passes plus six vocab heads per microbatch) is
the honest price of full-coverage cascade training at this vocab size.

### Result: dblock_chainallf_1k_mbs16 (mlq 3113, fused cascade)

Completed 1000/1000, checkpoint logs/dblock_chainallf_1k_mbs16_final_model.pt.
Final chain val_loss 4.16706 (bpb 1.3574), step_avg 8797 ms (marginal
~8.8 s from step 20 on), train-phase peak alloc 6.65 GiB. Trajectory:
7.587 @100, 5.693 @300 (already past 3111's final 6.170), 4.757 @500,
4.167 @1000 — monotone throughout. Per-level CEs converged to a tight,
ordered band (3.949-3.974 for levels 0-4) with the final graded level at
4.167; the final-level gap shrank monotonically after step 200
(1.90 -> 0.21), consistent with the mixture-flattening explanation (at
sigma 0.002 the EDM head has c_skip ~ 1, c_out ~ 0.002, so the last block
mostly re-projects the incoming posterior expectation; as predecessor
posteriors sharpen the projection loses less).

Cross-arm ledger (identical data, seed, schedule, panels):
- leak-trained dblock (mode 0):   val 9.365, 1.30 s/step, diverging
- chain single-block (3111):      val 6.170, 3.88 s/step, final-block cliff
- fused cascade (3113):           val 4.167, 8.80 s/step, no cliff
- AR baseline (arbase_1k):        val 3.592, 2.38 s/step

Honest verdict: chain-consistent cascade training makes DiffusionBlocks
*work* — val tracks train, scales with steps, and the one-block-graph
memory claim is real (6.65 GiB train peak vs ~17-22 GiB for the other
arms/baseline). It does not yet beat the AR baseline: 0.575 nats worse at
3.7x the step time on this 1k-step budget. Remaining known levers, in
expected-value order: (1) the final-level readout parameterization — the
residual 0.21 gap between level-4 and the graded level is structural
(c_out = 0.002 strangles the last block's correction; a sigma-free or
learnable-scale final readout is the principled fix and could recover
most of it); (2) MBS 32-64 (~10 GiB-25 GiB peak, est. ~10% step-time
saving from fewer launches); (3) the per-signal economics already favor
the cascade (1.47 s per block-update vs 3.88), so longer budgets close
quality gaps faster than the single-block mode ever could.

### Pre-registration: dblock_chainall_cpg_1k_mbs32 (clean propagation + readout gain)

Context: the user's requirement is DiffusionBlocks faster than the AR
baseline at equal-or-better bpb. Reference-code audit (this session)
settled the paper question: the released DiffusionBlocks repo is image
classification only (ViT/CIFAR, B=3) — the LM variant was never released.
The released code is exactly our leak arm (z = y + sigma*eps per sampled
block, raw clean context re-embedded per block, EDM-weighted CE through a
shared head, vocab softmax->expectation round-trip each sampler step), so
the paper's training-speed claim comes from the rule mlq-3102 proved
divergent under likelihood grading. Per-step chain-consistent training has
a FLOP floor of ~2x baseline trunk + per-level head work, so the route to
"faster overall" is a cheap-mode/cascade mixture — which is only worth
building if the cascade's quality ceiling rises first. This arm is that
gate.

Hypothesis: two defects account for much of 3113's remaining 0.575-nat
gap to the AR baseline. (1) Every block restarts clean context from
norm1(embed(inputs)), so context is only ever processed by 4 layers
(vs 24 in the baseline) and cross-block information flows solely through
the model_dim noisy bottleneck. (2) At the final sigma (0.002) the EDM
readout gives the last block's hidden state a fixed 0.002 weight against
c_skip ~ 1 on z (the measured 0.21 residual final-level gap).

Change (nanogpt_mini_gpt2vocab_kda_dblock_train.py, new config axes,
resume/eval-validated, checkpoint-recorded):
- DBLOCK_CLEAN_PROP=1: _run_block accepts/returns the clean-half residual;
  in-order sweeps (cascade training, val chain, oracle) thread block b's
  clean output into block b+1, detached between levels so gradients stay
  blockwise. Clean context thus accumulates the baseline's full 24-layer
  trunk depth (the clean trajectory is z- and sigma-independent: read-only
  KDA noisy slots, clean-only flex attention for clean queries,
  noisy-half-only AdaLN; DIT_FIDELITY+CLEAN_PROP rejected at config
  time). diagnostic_chain_loss uses harvest_clean_inputs (one in-order
  zero-noisy sweep) because fine grids revisit blocks. Known tradeoff:
  the embedding now receives context gradients only through block 0's
  pass (deeper clean states are detached) — watch for embed undertraining.
- DBLOCK_READOUT_GAIN=1: per-level learnable scalar g_b (zero-init) in
  the head: denoised = c_skip*z + (c_out + g_b)*hidden. Indexed outside
  the compiled head (one graph, gradient via indexing); shared Adam scalar
  group (lr 0.015); logged per level on val lines as rgain_l{k}.
- Eval-only loads now validate the dblock config axes like resume does
  (evaluating a cleanprop checkpoint without the flag would silently score
  a different chain). Pre-axis checkpoints (3111/3113) are loudly rejected
  by both paths; their embedded log source remains the way to re-eval them.

Run: identical recipe to 3113 except MBS=32 (3113 peaks: 6.65 GiB train
/ 8.55 GiB val at MBS=16 -> ~13/17 GiB expected; fewer launches, est.
~10% step time) plus the two new flags. Cost estimate ~8 s/step, ~2.4 h.

Success criteria (matched panels, same seed/schedule/data):
- Primary: final chain val_loss meaningfully below 3113's 4.167; the
  final-level gap (graded level minus level-4 CE) collapses toward 0 and
  rgain_l5 moves materially off 0.
- Directional target: closing toward the AR baseline's 3.592. If the arm
  lands near the baseline, build the leak/cascade mixture for wall-clock;
  if it barely moves, the context-depth hypothesis is wrong and the
  honest negative verdict stands.
Failure modes to watch: embed undertraining (val plateau with healthy
per-level CEs), gain instability at high-sigma levels (rgain_l0 drifting
large), later-block input-distribution churn from the now-evolving
propagated clean stream (per-level CE oscillation).

## 2026-09-02: MiniCPM VAPO v6 — canonical NextLat with shared-resident replay

The stopped step-203 MiniCPM5 VAPO checkpoint now has a consumable v6
continuation at
`postraining/runs/minicpm5_dapo_vapo_criticpretrain_fa4_batchedkv_8k_production/vapo_adapter_checkpoint_v6.pt`.
The migration preserves the actor adapter, actor NextLat head, value head,
completed-boundary cursor/RNG state, and every existing optimizer moment. It
initializes a separate critic adapter and separate critic NextLat head; the
writer remaps nonempty legacy NextLat optimizer moments into the actor's
appended optimizer group.

NextLat now matches `../NextLat`'s 1B configuration and objective: factor-1.6
2D-to-4864 residual MLP, bias-free LayerNorm/Linear layers, GELU, horizon 2,
SmoothL1 weight 1, and exact teacher-to-student categorical KL weight 1 through
the detached LM head. Actor and critic each own an independent 46,074,880
parameter dynamics head. Training samples 64 valid response states per
optimizer minibatch, recursively evaluates both horizons without crossing
packed-sequence boundaries, and computes the exact 130,560-way KL in
recomputed row chunks rather than retaining vocabulary logits.

Memory/residency changes:
- Actor and critic LoRA/value/NextLat trainables are disjoint, while every
  immutable critic backbone parameter aliases the actor's storage.
- The fused rollout replica remains GPU-resident. Phase transitions release
  only its 6+ GiB KV/CUDAGraph state; no policy is parked on CPU.
- Rollout KV is an explicit compiled-decode input and FA4 consumes the passed
  key/value tensors, so compiled code no longer owns evicted cache backing.
- Replay is unpadded FA4 varlen packing under a sum-of-real-tokens budget,
  with per-trajectory position resets and cumulative sequence lengths.
- Exact actor behavior log-probabilities are captured during rollout. Replay
  refresh therefore runs only the independent critic.
- NextLat sampling selects one capacity-weighted replay shard and performs a
  dense 64-row head call, avoiding 46M-parameter MLP launches on one or two
  rows per shard.

Production-shape proof is mlq job 4641, two full 4-prompt x 16-sample x
4096-token cycles from step 203, with the 12,288-real-token replay budget.
It completed step 205 and wrote a clean v6 checkpoint with no pending replay,
340 actor optimizer states, 345 critic optimizer states, and independent actor
and critic NextLat storage. At the matched 199,844-action first cycle:
- phase sum: 94.93 s versus the prior 100.7 s total, 5.73% lower;
- critic refresh: 7.86 -> 5.32 s and 44 -> 18 shards;
- update: 30.62 -> 24.86 s and 45 -> 20 shards, despite now training both
  canonical NextLat heads;
- rollout: 60.23 -> 64.76 s, the explicit cost of capturing exact behavior
  log-probabilities instead of performing a second actor replay.

The second cycle completed in 76.87 s phase-sum (57.09 rollout, 2.63 critic
refresh, 17.15 update) on 152,189 actions. Allocated VRAM after update was
identical on both cycles at 14,304,929,792 bytes; second-cycle update peak was
24,858,605,568 bytes. This is the cache-lifetime regression check: residency
does not grow across cycles. Focused CPU coverage is 69 passing tests, including
dense value/gradient parity for chunked KL, packed attention layout and replay
subsets, exact rollout log-probabilities, shared-storage/disjoint-trainable
invariants, checkpoint migration, and slow-rollout SDPA switching. Pyright is
clean on the model, trainer, and migrator. An independent follow-up review
found no remaining concrete defect.

## 2026-09-04: MiniCPM serving-runtime transfer audit

The measured B64 continuous-rollout smoke exposed two serving-path losses.
At 4,096 generated positions it spent 7.594 s in prefill and 50.614 s in
decode. Static prefill took 0.831 s. The continuous prefix-bank builder copied
logits, values, and every layer's K/V to pageable host storage with dozens of
synchronizing `.to("cpu")` calls, then copied the same four prompts back to the
device. Decode productive utilization was 71.63%; active rows fell from 64 to
48 by position 1,536 and to 36 by position 3,584, but completed rows retained
their full visible FA4 KV lengths on every later captured step.

Exact serving-runtime changes staged without a model run:

- Continuous generation now retains its unique-prompt prefix bank on the GPU,
  removing the immediate D2H/H2D round trip. The public reusable host-bank path
  uses pinned storage, nonblocking copies, and one final synchronization rather
  than one synchronization per tensor.
- Completed static lanes set their FA4 `seqused_k` length to one. The fixed B64
  CUDA graph and all active-row outputs are unchanged, while dead lanes stop
  scanning stale KV history.
- The existing 10,000-token, B64, BF16 policy stays unchanged. These are
  execution-only changes; no sampling, reward, optimizer, or objective setting
  moved.

Transfer decisions from vLLM/SGLang and local evidence:

- Keep CUDA graphs, continuous refill, fused projections, device-resident
  sequence metadata, pinned host staging, and weight/KV phase separation.
- Do not add paged/shared-prefix KV yet: the installed FA4 SM120 path explicitly
  rejects `page_table`, split-KV, block sparsity, and FP8 KV. A custom paged
  kernel would be a separate benchmarked project, not a flag change.
- Do not enable NextLat speculative decoding: measured position-2 acceptance is
  0.90% at step 370 and 0.55% at step 203, far below break-even.
- Do not enable W8A16 output-head quantization: its B1 speedup was about 7%, but
  the categorical quality gate failed (KL 0.00693, top-1 match 0.8846).
- Keep replay logit chunks at 128 and checkpoint interval 4. The step-370 sweep
  measured interval 4 at 1,867.87 tok/s versus 1,838.54 for interval 2; larger
  10,240/12,288-token replay shards were slower and used more VRAM.

Queued evidence gates use a hard maximum of ten actor steps or 30 minutes,
whichever arrives first. Critic warmup is a separate prerequisite and is not
charged to either limit.

1. Run matched B64 and B48 10,000-token rollout-only gates from the same
   step-370 checkpoint. B64 must move prefill toward the static 0.831 s
   reference without a generated-token or quality regression. Keep B48 only if
   its end-to-end rollout tok/s beats B64 on the same 64 logical trajectories;
   high truncation can make delayed admissions lose despite higher occupancy.
2. Run matched standard-LoRA and NoRA arms for at most ten actor steps after
   each arm completes ten critic-only warmup steps. Ten steps cannot satisfy
   the repository's normal 2,000-step/0.005-BPB adoption rule, so this gate may
   reject NoRA early but cannot promote it to the default.
3. Branch the warmed standard-LoRA state into four-minibatch and one-minibatch
   arms for at most ten actor steps. Changing the minibatch count changes AdamW
   step count, behavior-policy age, and NextLat sampling; this remains a
   learning experiment rather than a speed refactor.

Every gate, including critic warmup, has an `mlq --time-limit 30m`; all jobs
use `--max-parallel-runs 1`.

Queue history:

- 4849 and 4850: successful B64/B48 10,000-token rollout gates. B64 took
  260.04 s for 335,049 tokens (1,288.45 useful tok/s; 2,578.61 scheduled
  decode tok/s); B48 took 283.94 s for 343,495 tokens (1,209.75 useful tok/s;
  2,179.69 scheduled decode tok/s). B48 is rejected: it was 9.2% slower
  end-to-end despite 2.5% more sampled tokens.
- 4851 and 4852: failed before model loading because the requested 60-second
  checkpoint interval violated the repository's validated 300–600-second
  range. Dependent jobs 4853–4855 were skipped.
- 4859: replacement standard-LoRA critic warmup, 10 warmup steps, 30 minutes.
- 4860: replacement NoRA critic warmup, 10 warmup steps, 30 minutes.
- 4861: standard LoRA / four minibatches, after 4859, 10 actor steps or
  30 minutes.
- 4862: standard LoRA / one minibatch, after 4859, 10 actor steps or
  30 minutes.
- 4863: NoRA / four minibatches, after 4860, 10 actor steps or 30 minutes.
- 4859 and 4860 reached critic replay but OOMed with the 10,000-token response
  and 11,024-token replay budgets; dependents 4861–4863 were skipped.
- 4868 and 4869 are memory-safe replacements using the previously validated
  4,096-token response and 8,192-token replay budgets. Each critic warmup still
  has 10 steps and a 30-minute hard limit.
- 4870 and 4871 are the standard-LoRA four/one-minibatch gates after 4868;
  4872 is the NoRA four-minibatch gate after 4869. Each has 10 actor steps and
  a 30-minute hard limit.

## 2026-09-06: Uno / Ψ-Spec (diffusion-augmented LLM) applicability review

Paper filed at `papers/uno_diffusion_augmented_2609.04010v1.pdf`
(arXiv:2609.04010v1). Full record and assessment:
`docs/uno_diffusion_augmented_assessment.md`. **The 2026-09-07 review was
documentation-only; subsequent authorized opt-in implementation is recorded below.**

The method trains gated diffusion LoRA with one-step blockwise distillation,
then drafts a block in parallel and verifies against the AR target. Losslessness
comes from correct rejection sampling against the current target distribution,
not freezing weights alone. The paper reports 1.47x system throughput on a
**single H200** in a synthetic 1K/8K test and up to 40% end-to-end DAPO speedup
(detailed RL timings deferred). Table 8 mean TPF falls 2.25→2.10 (6.7%),
but GSM8K falls 2.66→1.95 (26.7%); the average is not a drift bound.

Evidence-backed conclusions:

- `maximal_coupling_verify` is a reusable per-position accept/residual primitive.
  NextLat's ragged cache, pending-token loop and bonus handling are references,
  **not a verbatim production integration**. Uno needs AR-only committed KV,
  diffusion-suffix invalidation, correct pending-token indexing, block rollback,
  EOS/budget handling and continuous lane refill under graph capture.
- NextLat's `dense_top_p_probabilities` has no top-k argument, whereas production
  `top_k_top_p_sample` applies top-k=20 before nucleus filtering. The free token
  and verifier must preserve the live actor's exact transformed distribution;
  proposal q must describe what was actually sampled.
- NextLat's measured 0.55–0.90% position-2 acceptance does not disprove Uno,
  but TV is not a demonstrated repair. KL also constrains TV/acceptance.
  NextLat has an exact first token after initial prefill, not every later cycle;
  Uno draws a new AR-only first token each ordinary cycle. The two-token floor
  excludes EOS, output-budget and capacity truncation and is not a speed guarantee.
- LoRA sizing is exact: 29,184r per layer ×24 = 700,416r, matching 11,206,656
  at r=16. Rank48 gives 33,619,968 parameters, 3.11% of the frozen base.
  Rank48/alpha3072 is an unvalidated candidate, not a proven optimum.
  The paper is internally inconsistent: main text gives rank128/alpha256,
  Appendix C.6.4 selects alpha/r=64 for Uno and 16 for UnoQwen.
- Paper Table17a/18 pairs give effective fitted delta=0.081–0.094, not direct
  marginal kernel timings. The old delta~=0.11 calculation mixes a historical
  4k-position decode denominator with assumed 10k-rollout KV lengths and omits
  attention arithmetic, adapter, sampler and cache overhead. KV capacity is
  17.34GB/16.15GiB at B64×11,024 slots; capacity does not prove HBM traffic.
  At assumed delta=.11 and paper tau=3.89, B4 gives **1.46x decode**, with an
  optimistic whole-step **1.27–1.31x** only if the entire rollout sped up equally.
  Neither this delta nor Uno acceptance is measured for MiniCPM.
- `_fixed_varlen_fa4_attention` specializes q_len=1, but more importantly
  `_CompactStaticLayer.update:333-348` treats width>1 as prefill, overwrites
  prefix slots and returns only new K/V. E0 needs correct cached append semantics,
  not just a new `max_seqlen_q`. The installed FA4 signature is only an API lead.
- Training has a separate blocker: SM120 backward rejects `mask_mod` and block
  sparsity (`flash_attn/cute/interface.py:1891-1915`), while Uno's concatenated
  clean/noisy-block mask is not ordinary causal attention. Efficient forward/
  backward and peak-memory feasibility must be established before training.
  Forward custom masks are exposed, so block-sparsity rejection does not prove
  tree inference impossible; linear-first is a scope choice, not a hard limit.
- A separate unmerged gated diffusion adapter must leave the actor enabled.
  Preserve base projection fusion; compare grouped/fused low-rank updates rather
  than assume dense block-diagonal B GEMMs are negligible. With top-k20 p and q,
  exact coupling can use fixed-size supports (union ≤40 IDs), avoiding dense
  full-vocabulary residual buffers. Full head/top-k work still remains.
- A fresh actor equals its selected base under both standard and current NoRA
  initialization because B is zero. Resumed actors need not. Rank does not bound
  update magnitude, output KL/TV or acceptance drift. Base-trained adapters can
  remain lossless with the current actor verifier while losing all useful speedup.
  Narrower sampling support likewise does not guarantee greater acceptance.

At the review boundary these were proposed gates, not implemented or queued.
Every local GPU/model workload must use `mlq submit --max-parallel-runs 1`:

1. E0a: correctness-first attention probes at n={1,2,4,8,16}, B64, representative
   ragged lengths and near-capacity states. Separate kernel timing from compile/
   capture startup. Establish supported masked training backward separately.
2. E0b: after append-correct cache plumbing, measure full block forwards, gated
   draft, verifier, sampling and commit overhead at matched occupancy/length.
   Define `C_B=t_cycle/t_AR`; decode speedup is `tau/C_B`, ceiling `(B+1)/C_B`.
   At B4/delta=.25, 1.11x uses paper tau; the actual token-count ceiling is
   **1.43x**, not 1.11x. At delta=.11, a 1.25x decode target needs tau≥3.325
   before omitted overhead, not merely tau≥3.0.
3. E1: only after inference/training feasibility, consider rank48/alpha3072,
   TV-only, lr1e-5/2% warmup, L2048, position-chunked full-vocabulary TV with
   bounded backward memory. Proposed 100M@B2 +300M@B4 is **400M tokens**.
   Retokenize/count the candidate corpus, pin initialization and teacher, and
   measure actual training throughput. The paper's best 1ep/3ep ratio compares
   different configurations; it is not proof of a 1B learning curve or pilot
   sufficiency. No 15-hour pilot or one-GPU-day go/no-go is established.
4. E2: matched contemporary AR/candidate runs at B64/10,000 tokens, same actor,
   prompt pool, sampling, logical work, cache and refill policy. Historical
   reference: 2,578.61 scheduled decode tok/s and **1,288.45 useful rollout
   tok/s**, 260.04s (`NOTES.md:5962-5966`). A 4,096-token gate needs its own
   matched baseline. Require ≥1.25x useful rollout tok/s, report full-cycle wall
   time, memory, occupancy, truncation and checkpoint-dependent acceptance too.

Correctness requires the actual production p/q law, same-prefix block/serial
verifier checks through rollback/refill/EOS/budgets, and analytical plus
statistical sampler checks. Loose fusion-logit tolerances cannot prove
distributional equivalence. Same-seed generated-ID hashes need not agree across
AR/speculative algorithms; keep them as replay provenance, not a losslessness gate.

Verdict: plausible research candidate, not production-ready or measured faster.
The no-speculation decision and repository adoption rules remain unchanged.

## 2026-09-07: Opt-in MiniCPM Uno implementation before RL

User authorized making the reviewed method implementation-ready behind a flag.
Default AR is unchanged; no 400M-token distillation pilot or RL campaign launched.

- `postraining/uno.py`: separate fp32-master standard LoRA (rank48/alpha3072,
  all seven projections), checkpoint-stable gates, compiled native FlexAttention
  paired clean/noise mask, duplicated RoPE, full-vocabulary uniform noise and
  detached **same-output-position** teacher NTP distributions. Position-chunked
  full-vocabulary L1 recomputes the output head during backward; no vocabulary-
  chunk normalization approximation and no FA4 custom-mask backward.
- `scripts/train_minicpm_uno.py`: explicitly supplied offline reasoning JSONL/text,
  document-heldout split, pinned source/tokenizer/teacher identity, bf16 compute,
  L2048, microbatch1×accumulation64, lr1e-5 with2% warmup,
  100M tokens@B2 +300M@B4. These are candidate transfer defaults, not demonstrated
  optimums. Atomic latest.pt recovery saves optimizer/RNG/cursor at completed
  updates; adapter.pt exports the finished curriculum. Metrics.jsonl and
  TensorBoard track training/heldout L1/TV, gradient norm and useful token throughput.
- `postraining/uno_speculative.py`: captured B-query draft+verify, native FA4
  bottom-right causal suffix attention, clean-only KV commit/rollback, pending
  correction/bonus token, EOS/budget handling and inherited continuous refill.
  B4 is the default. Exact sparse top-k→nucleus coupling uses target-only residual
  support (≤20 IDs), not a dense vocabulary residual or padded union.
- RL enables only with `--uno-rollout --uno-checkpoint ... --uno-block-size 4`.
  The actor stays merged into the fused replica, the diffusion adapter stays
  unmerged/frozen, and synchronization preserves both roles. PPO likelihoods
  remain untempered replay actor likelihoods. Adapter bytes are SHA-256 pinned
  in RL recovery; mode changes require a completed rollout boundary.
- `scripts/benchmark_minicpm_uno.py` compares separate-process AR/Uno arms on
  matching actor/data/prompts/sampling/physical lanes/output cap, with warmup
  excluded. Default128 logical requests over64 lanes exercises refill.
  ≥1.25x useful rollout tok/s is an explicit candidate gate, not a measured result.
  Reports include wall time, useful outputs, truncation, occupancy, memory, tau
  and conditional per-position acceptance. Uno's production throughput floor
  is in useful rollout tok/s; the legacy AR scheduled-token floor is not comparable.
- Reference release `ifm-ai/uno@46fbdb66f026bae9c68a1e5a3f97a17c7805c778`
  confirms native FlexAttention and UnoQwen r128/alpha2048 TV-only training.
  The release also has a distinct1B checkpoint, not MiniCPM performance evidence.
- Independent training/runtime-performance reviews found and corrected CPU
  adapter masters before CUDA teacher merge, and AR-sized completion polling
  that wasted whole block cycles near output limits. Both reviewers report
  no unresolved source findings. CPU suite: **94 passed**.
- Full MiniCPM CUDA numerical qualification is queued as **mlq5268**,
  max-parallel-runs1,45-minute execution limit. It tests native FA4 width1/2/4/8/16,
  full-shape bf16 masked backward/head gradients/teacher isolation, B64 captured
  ragged-prefix/refill/EOS/budget/cache rollback and nonzero actor resynchronization.
  Its temporary one-update adapter is a numerical fixture only, not learning
  or speedup evidence. Qualification is not yet claimed passed.

Operational commands and limitations are documented in `postraining/README.md`.
Train the real adapter and demonstrate matched useful speedup/quality before
enabling an RL campaign; acceptance drift and compute payback remain unmeasured.

## 2026-09-08: Uno width-invariant numerical target

User requested further investigation of the failed native one-token AR numerical
gate, then selected a **shared invariant target**, keeping legacy AR as the default
and reporting its drift separately. The paper (§5.2.1) explicitly acknowledges
numerical nondeterminism; its ordinary shape-dependent GEMMs do not solve this for us.

- Full-model job5517: gate-zero first logits and clean-KV reconstruction became
  exact with local `emulate_precision_casts=True`; native width-one comparison still
  reached max raw TV0.06131. Identical-QKV FA4 is exact; output-projection GEMM
  M64/M256 arithmetic differs by up to0.015625 bf16. Eviction now clears Cache
  payload and collects unreachable Dynamo cycles before restoring the actor.
  All old backing references died; allocated memory returned to4,781,344,256 bytes.
- Job5518: four contiguous M64 projection calls plus cast emulation in both paths
  yield bit-identical full-model block/serial logits. Default compiler fusion
  cannot be matched merely by preserving GEMM shapes.
- Job5520: fixed-tile bf16 Triton GEMM gives exact full-model serial/block logits
  and gated first token at B64, prefixes1/65/513/8192. Probe medians: native15.09ms,
  verify15.81ms, draft18.19ms; stock verify14.08ms/draft16.43ms. These are
  **unqualified provisional timings**: later tests exposed per-layer eager fallback.
  cuBLASLt disabling reduced reduction AND split-K still had maxTV0.01763.
- Implemented `postraining/invariant_linear.py`, contract
  `bf16-lane64-n128-k32-casts/v1`. Uno and opt-in invariant AR share it.
  Default AR is untouched numerically. A pending-first AR schedule also aligns
  final-prompt KV/logits on admission/refill; the fixed-batch statistics-free
  handoff no longer consumes that token twice.
- Recovery pins the Uno arithmetic version for pending replay. The benchmark
  compares invariant AR, Uno and legacy AR; both AR speedup gates must pass.
- Waited for the external CleanRL workers; none were terminated. Job5646 passed
  all3 full-model CUDA tests: exact selected-target logits, clean KV, transformed
  supports/probabilities, actor refresh, refill/capacity and cache release.
  Legacy raw maxTV was0.03000 before and0.04575 after the fixture actor update.
  The real2048-token masked backward peaked at3,046,521,856 allocated bytes.
- Root cause in jobs5630–5634: disabled FA4 callbacks caused per-layer graph
  breaks and eight-variant Dynamo exhaustion. Projection hooks now capture modules,
  not layer-name strings. FA4 has a compiler-visible custom-op/fake contract;
  invariant forwards require fullgraph compilation and fail on recompile exhaustion.
  Legacy AR's existing compiler behavior is deliberately unchanged.
- The performance review found48 KV clone/copyback pairs per fullgraph forward:
  48.22GiB logical read/write traffic at cache8230. Deriving views removed synthetic
  aliases but not copies; `scatter_` has no registered in-place lowering. Shared
  indexed suffix updates use `index_put_`, which does. Job5646 and independent
  generated-code inspection verify all96 fullcopies perforward disappeared.
  Identical33-cycle fixture:3.00618s→0.76872s, or91.10→23.29ms/cycle, a3.91×
  schedule-only speedup. Acceptance counts unchanged; not an AR-relative result.
- CPU regression suite:96 passed; Ruff undefined-symbol checks passed.
- Three-arm job5647 completed:8 prompts×16 responses,64 lanes,B4,4096-token cap,
  three same-seed timing repetitions. Median useful tokens/s: legacy4811.94,
  invariant7712.27, Uno7185.02; Unoτ=2.132. Thus0.9316× matched AR and1.4932×
  legacy AR; the1.25× dual gate correctly failed. Truncation96.9–99.2%.
  This used a one-update numerical fixture, not a properly distilled adapter.
- Canonical metrics/repetition data/source hashes saved under
  `ablation_results/uno_invariant_runtime_20260908/`. Temporary fixture checkpoints
  and diagnostic harnesses removed.

No full adapter-learning or RL campaign launched. Remaining gates are real offline
adapter learning, matched useful speedup with that adapter, task quality, acceptance
drift under RL and amortized compute payback.

## 2026-09-08: Optimized ordinary AR promoted; invariant GEMMs excluded

User vetoed Uno training until better hardware, then separately authorized an
AR runtime default change, a short GEMM comparison and a ten-minute-capped check.

- Job 5706 completed in 141.80 seconds (180-second limit): optimized native
  cuBLAS 15,463.29 useful tokens/s, invariant GEMMs 9,460.03. Retain ordinary
  GEMMs and ordinary AR scheduling; use indexed KV writes, compiler-visible FA4
  and fullgraph preserved-cast compilation by default.
- Job 5711 completed its measurement harness in 463.89 seconds (600-second
  job limit), using the step-370 `minicpm5_vapo_balanced_lr_cont200_v15` actor,
  four DAPO prompts × sixteen responses, 64 lanes and a 10,000-token output cap.
  Legacy → optimized: 1,251.95 → 4,164.31 useful tokens/s (3.33×);
  logical-pool time 253.53 → 89.78 seconds (2.82×). The optimized arm emitted
  373,859 tokens versus 317,407 legacy tokens; throughput is not fewer-token work.
  Peak allocation remained about 20.48 GiB.
- Quality warning: optimized 22/64 correct and 44/64 completed, versus legacy
  26/64 correct and 47/64 completed. Four distinct problems are insufficient to
  establish systematic regression or non-regression. No quality-neutral claim.
- Same-prefix sampler TV: mean 0.001830, maximum 0.034688; top-1 agreement 100%.
  After a synthetic in-memory actor update and cache release/recreation, reused
  and fresh optimized replicas had exactly equal logits. An 80-row logical pool
  refilled 64 physical lanes and emitted all 480 in-budget tokens.
- `DEFAULT_OPTIMIZED_DECODE=True`; ordinary GEMMs remain native. Identity is
  `bf16-cublas-fa4-fullgraph-casts/v1`. Pending legacy checkpoint records fail
  closed on arithmetic mismatch; completed-rollout checkpoint boundaries work.
  Explicit legacy benchmark behavior and Uno's invariant target remain available.
- Future Uno adoption gates include the new optimized production AR baseline;
  exceeding only the slower legacy/invariant controls is insufficient.

Canonical evidence: `ablation_results/minicpm_ar_gemm_choice_20260908/` and
`ablation_results/minicpm_ar_production_20260908/`. No full RL training run or Uno
adapter training was launched. Remaining quality and end-to-end training checks
are tracked in `postraining/TODO_MINICPM5.md`.

## 2026-09-16: Future-bag carry for the streaming FFN recurrence (pre-registration)

Lineage: the streaming FFN line in `pretraining/future_credit_stream/`. The
best result there is plain recurrent CE with the actual hidden as the detached
carry (2.139849 / 2.139481 / 2.139952 proxy BPB across three matched CE runs,
memory gain ~1.165 BPB). Every attempt to shape the carry with a future proxy
lost: delayed NextLat predicted carry +0.458, NextLat actual carry +0.257,
vector synthetic credit worse at step 660 (cancelled), scalar TD value into
the hidden +0.586 with memory gain falling to 0.884. The TD arm's logs show the
critic never fit (prediction ~0.2 vs target ~4.7).

Two designs were built and dropped before any arm ran. (1) A hindsight
advantage critic: a scalar head over the carry direction fitted by a Bellman
regression to `ce(zero carry) - ce(actual carry)` and differentiated into a
gated writer. Dropped by request: a learned value is an indirect, hard-to-fit
stand-in, and it needed a second zero-carry forward per token. (2) An
imagined rollout: run the reader on the actor's own belief-weighted embedding
plus the proposed carry for k future steps, score each on the real future
token, and differentiate into the writer. Dropped for cost: k extra reader
passes per token exceed TBPTT itself, which is off the table.

Hypothesis. The carry must be chosen without gradients from the future,
without future tokens as inputs, without a critic, and without running the
model again. CE recursion carries `h_t`, the vector whose readout through the
head is the belief about `x_{t+1}`. The next reader will know `x_{t+1}`; what
it needs from the carry is what the past says about the tokens after it. So a
gated writer (`c = g * unit(write(h)) + (1-g) * unit(c_prev)`, initialized
to CE recursion) is trained so that the carry's readout through the *same
frozen final norm and head* is the discounted bag of the target and the
upcoming tokens: `q_t ~ sum_{j<=32} 0.9^j onehot(x_{t+1+j})`, truncated at
the document's closing BOS (included), soft-target cross-entropy. The target
is the exact discounted return of future one-hots from the corpus, so the
horizon comes without a bootstrap, a learned value, or a vector residual to
regress. The backbone stays pure CE by default (`backbone_future_weight=0`).

Two amendments from the pre-run review, before any arm started. (a) The bag
begins at `x_{t+1}` rather than `x_{t+2}`: with a full-rank head the bag pins
the carry's whole direction, and a bag that skipped the current target left
nothing anchoring the carry to what the hidden already encodes (recent-token
identity the next reader needs). Starting at `x_{t+1}` makes `discount=0`
exactly the hidden's own job, so the identity-initialized writer starts near
its optimum and every larger discount adds the horizon on top; it also
leaves no empty rows. (b) Both mixed terms are RMS-normalized inside the
writer: otherwise retention could be expressed through the relative norms of
`write(h)` and `c_prev` instead of the gate, and `gate_mean` would not
measure retention. Only the carry's direction reaches the reader and the
judge, so the normalization costs nothing.

Cost and parameters. Per token: two head matmuls (one gradient-free for the
hidden's reference bag loss), a scatter, two softmaxes, and the writer's three
`512 x 512` matmuls; expected 10 to 15% of step time. The writer's 787,456
parameters (about 5.8% of the 13.65 M backbone) run at inference too, so the
arm is not parameter-matched to `ce`; the results table reports parameter
counts and the step-time ratio. The head is zero-initialized, so the writer's
gradient is zero at step 0 and small during warmup; Adam normalizes it, and
the 100-step warmup from lr 1e-6 is what bounds the early writer steps.

Verifier: matched 2,000-update proxy BPB on the fixed 256-document validation
panel, plus `val_memory_gain_bpb`, plus training diagnostics `bag_loss`
against `hidden_bag_loss` (the same bag cross-entropy for the hidden itself,
what CE recursion would carry: the matched reference the writer must beat;
`bag_entropy` is not a floor, the bag is a sample of an unseen future),
`gate_mean`, `carry_cosine`, and the second-half median `step_avg_ms` ratio
to CE. Same seed, data order,
tokenizer, and backbone initialization across arms.

Success criterion: `future_bag` (discount 0.9) beats the matched `ce` control
by more than 0.005 proxy BPB. Interpretation: `discount=0` (carry judged on
`x_{t+1}` only, the hidden's own job) versus 0.5 versus 0.9 shows whether the
horizon, rather than the writer, matters; `tbptt` (true temporal gradients inside each 8-tick page,
reference only) bounds what any local method could gain; the
`backbone_future_weight=1` arm tests whether letting the bag gradient shape
the backbone helps or, as with every earlier future-proxy gradient into the
backbone, hurts.

Likely failure modes: (1) the reader cannot exploit a bag-shaped carry:
`bag_loss` falls clearly below `hidden_bag_loss` while proxy BPB and memory
gain do not move or worsen (the carry drifts away from the hidden the reader
was trained to read; watch `carry_cosine` fall with no BPB gain); (2) a linear-gated writer
over a CE-shaped hidden is too weak to change anything (`carry_cosine` ~1,
`gate_mean` ~0.95, `bag_loss` flat); (3) with discount 0.9 the bag is diffuse
(entropy several nats) and the gradient mostly asks for a topic direction the
head already exposes; (4) TBPTT-8 itself not beating CE, which would say the
FFN reader cannot exploit a better carry at this scale.

Arms (all `--after-success` on the GPU contract tests, job 7747):
7748 `ffn_bag_ce_sp1024_2k` (ce control), 7749 `ffn_bag_g90_sp1024_2k`
(future_bag, discount 0.9), 7750 `ffn_bag_g50_sp1024_2k` (discount 0.5),
7751 `ffn_bag_g0_sp1024_2k` (discount 0), 7752 `ffn_bag_g90_bb1_sp1024_2k`
(discount 0.9, `backbone_future_weight=1.0`), 7753
`ffn_bag_tbptt8_sp1024_2k` (tbptt reference). Horizon 32 everywhere.
Results are appended below when the runs complete.

Results (2026-09-17, all six arms complete). Same seed, data, and backbone
init; 2,000 updates;
`bag_loss` / `hidden_bag_loss` / `gate_mean` / `carry_cosine` are the last
training log (step 2000); step ms is the second-half median.

| arm | params | proxy BPB | vs ce | memory gain | bag_loss / hidden_bag_loss | gate_mean | carry_cosine | step ms | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ce` | 13,652,480 | 2.139688 | - | 1.1667 | - | - | - | 21.37 | 1.00x |
| `g0` | 14,439,936 | 2.141280 | +0.001591 | 1.1369 | 3.448 / 3.450 | 0.992 | 0.997 | 26.63 | 1.25x |
| `g50` | 14,439,936 | 2.146800 | +0.007111 | 1.1546 | 5.010 / 5.813 | 0.927 | 0.660 | 26.25 | 1.23x |
| `g90` | 14,439,936 | 2.168735 | +0.029047 | 1.1119 | 5.842 / 7.848 | 0.784 | 0.365 | 26.30 | 1.23x |
| `g90_bb1` | 14,439,936 | 2.183606 | +0.043918 | 1.0931 | 5.802 / 7.656 | 0.795 | 0.411 | 27.37 | 1.28x |
| `tbptt8` | 13,652,480 | 2.175066 | +0.035378 | 1.2650 | - | - | - | 20.46 | 0.96x |

Reading. This is pre-registered failure mode 1, and it is monotone in the
discount. The writer wins its own judge decisively (the carry's bag
cross-entropy ends 0.8 nats below the hidden's at discount 0.5 and 2.0 nats
below at 0.9), so failure mode 2, a writer too weak to move, is ruled out.
Proxy BPB and memory gain worsen in proportion to how far the carry leaves
the hidden: cosine 0.998 -> 0.66 -> 0.37 maps to +0.002 -> +0.007 -> +0.029
BPB. The horizon is not what failed; the judge is. The head reads one
direction of the carry, while the reader is six FFN blocks that consume
everything, including whatever the head ignores. The writer is free to
discard what the head cannot see and is rewarded for doing so whenever that
buys bag mass. `hidden_bag_loss` rising through training (7.15 at step 510
to 7.85 at step 2000, discount 0.9) shows the CE-shaped hidden sharpening
away from the bag direction as it learns; the writer follows the bag instead.
Letting the bag gradient shape the backbone (`backbone_future_weight=1`) is
worse again (+0.044), the fifth future-proxy gradient into the backbone to
lose. The discount-0 arm is within 0.002 of CE, so the 787,456 writer
parameters and the normalized gate mixing are harmless on their own but earn
nothing. The step-time ratio is 1.23x, above the 10 to 15% estimate: the
two `[4096, 1024]` softmaxes and the scatter are not small at this model size.

Lesson. Four reader-free judges of the carry have now lost in this line
(NextLat regression, synthetic vector credit, scalar TD value, head-readout
bag). A judge the writer can satisfy without the reader gets satisfied
without the reader. The one judge that cannot be gamed is the reader's own
loss: `dCE_{t+1}/dc_t`, which the next tick already computes down to
`latent_norm`'s input and currently discards. But the `tbptt8` reference
answers whether a better carry can help at all here, and the answer is no at
this budget (failure mode 4): true temporal gradients through the backbone
over every 8-tick page end at 2.175066, +0.035 worse than detached CE, with
a larger memory gain (1.265 vs 1.167) bought by a worse reset-state BPB
(3.440 vs 3.306). The gap to CE closes over training (+0.072 at step 800,
+0.035 at step 2000), so this is a 2,000-update statement, not a limit; at
this budget the model learns faster with each tick as its own graph. TBPTT's
one backward per page also makes it the fastest arm (0.96x). Since the
upper reference for temporal credit loses, no local carry objective, reader-
judged or otherwise, should be expected to gain at this scale, and the
reader-gradient writer is not built. The line stops here: nothing is
promotable; CE recursion stands.

## 2026-09-17: Stationary buffer read for the streaming FFN (pre-registration)

Lineage: the streaming FFN line in `pretraining/future_credit_stream/`. CE
recursion with the detached actual hidden as the carry stands at 2.139688
proxy BPB (`ffn_bag_ce_sp1024_2k`, memory gain 1.167). Every attempt to
choose that carry by a reader-free proxy lost, and the TBPTT-8 reference
(true temporal gradients through the backbone inside each 8-tick page) lost
too, at 2.175066. The recursive carry is one vector rewritten every step;
information from `K` steps back reaches the reader only by surviving `K`
rewrites, none of which is asked to keep it, and gradient pressure through
that chain does not fix it at this budget.

Hypothesis. Make the recurrence stationary: keep the last `K` post-final-norm
hiddens of the document verbatim and detached, each written once and never
rewritten while readable, and let every step read all of them directly. This
is causal attention's structural trick (each position's representation is
built once and consumed by many later positions) and Transformer-XL's
stop-gradient memory at a segment length of one token. Information from
`x_{t-k}`, `k <= K`, then reaches step `t` in one hop, produced by a network
that actually had `x_{t-k}` as input. The producer of a slot is still trained
only by its own CE; long-horizon utility reaches it only through the reader
adapting. What is new is that far less has to be carried on purpose, because
the reader reaches back directly and can choose *which* past step to read
from the current token (content addressing), which a recursive carry cannot
do at all.

Design (`pretraining/stationary_stream/`). Backbone, data pipeline, seed,
optimizer, schedule, validation panel, and checkpoint contract are those of
`future_credit_stream`; the backbone initialization is bitwise identical to
the CE-recursion control (tested), so `ffn_bag_ce_sp1024_2k` is the matched
control and no new control arm is run. State per lane: a buffer of the last
`K` detached BF16 hiddens `[K, B, D]` and a validity mask `[K, B]`, kept as
a ring: each tick the host overwrites the oldest slot in place with the
tick's hidden (one `[B, D]` write, no shifting), and a per-tick age vector
(one row of a precomputed `[K, K]` table indexed by the newest slot) tells
the read how many steps old each slot is. The mask is cleared on the lane's
BOS, so slots written before the document are never read. The read, over
ages `a = 1..K`:

```text
q      = W_q norm(embed x_t)                       (H heads x d_k)
k_a    = rotate(age a)( W_k h_{t-a} )              the query is not rotated
w      = softmax_a ( [ b_null , q . k_a / sqrt(d_k) ] )   masked slots -> -inf; a null slot is always available
read   = W_v ( concat_h sum_a w_{h,a} slice_h(h_{t-a}) )   value projection after mixing: exact by linearity
h_t    = FFN stack( norm(embed x_t) + read )
```

RoPE on the keys carries time: the rotation for every age `1..K` is
precomputed and gathered by the slot's current age, so every score depends
on relative age only and the ring layout is invisible to the read (tested:
permuting slots together with their ages leaves the output unchanged).
`W_v` starts at zero, so the untrained model is the context-free FFN and
grows the read from the CE gradient. No gradient crosses a tick; nothing
judges the buffer.

Cost. `H=4`, `d_k=32`, `K=10`: read parameters 393,988 (2.9% of the
13.65 M backbone; the model has 14,045,956 parameters, the control
13,652,480), per-token MACs about 0.98 M against the backbone's 13.1 M
(7.5%), buffer 40 MiB over 4096 lanes. `K=32`: 2.4 M MACs (18%), 128 MiB.
The user's requirement is that this stays cheap; the results table reports
the second-half median step-time ratio to the control (21.37 ms).

Verifier: matched 2,000-update proxy BPB on the fixed 256-document panel,
`val_memory_gain_bpb` (against an always-empty buffer, realized as an
all-true reset mask over the same ring), and the read diagnostics
`null_mass` (attention on the null slot) and `read_age` (mean age of the
non-null attention), both over lanes with a readable slot, and `read_rms`
(RMS of the read without the value bias, against the unit-RMS observation;
exactly zero while the value map is still zero), plus the step-time ratio.

Success criterion: `K=10` beats the CE-recursion control by more than 0.005
proxy BPB at no more than about 1.1x its step time. Interpretation: `K=1`
isolates the read from the depth (it can only reach `h_{t-1}`, what CE
recursion adds directly), though it is not literally CE recursion: the
control's carry enters live from step 0 at unit RMS through `latent_norm`,
while here the zero-initialized value map starts context-free and must grow
the read; `K=32` shows whether more reachable history keeps paying at this
budget.

Likely failure modes: (1) the reader ignores the buffer (`null_mass` -> 1,
`read_rms` -> 0, memory gain near zero) because the zero-initialized value
map never grows; (2) `K=1 ~ K=10 ~ ce`: an FFN reader cannot exploit reachable
depth at this scale, which would also explain TBPTT's loss; (3) `K=10` below
`ce`: a convex mixture of past hiddens entering at learned scale is worse than
one hidden entering at unit RMS through `latent_norm`, in which case the fix
is the entry path, not the buffer; (4) step time above 1.15x at `K=10`, which
fails the performance requirement regardless of BPB.

Arms (all `--after-success` on the GPU contract tests): `stat_k1_sp1024_2k`,
`stat_k10_sp1024_2k`, `stat_k32_sp1024_2k`. First submission 2026-09-17
(mlq 7777-7780) was skipped: the GPU contract tests passed 9 of 10, the one
miss being the two-element null-bias gradient at 2.4% against a 2% bound
after two *independent* optimizer steps of the compiled model and the eager
reference (Adam on BF16 sign noise); the test now resyncs the reference to
the compiled parameters after each step. Resubmitted at priority 2: mlq 7783
(`stat_gpu_tests`), 7784 (`stat_k10_sp1024_2k`), 7785 (`stat_k1_sp1024_2k`),
7786 (`stat_k32_sp1024_2k`).

Results (value_map entry; control `ffn_bag_ce_sp1024_2k` 2.139688, memory
gain 1.1667, 21.37 ms):

| arm | proxy BPB | vs control | reset-latent BPB | memory gain | null_mass | read_age | read_rms | step ms (ratio) |
|---|---|---|---|---|---|---|---|---|
| `stat_k1_sp1024_2k` (7785) | 2.203452 | -0.0638 | 2.9751 | 0.7716 | 0.346 | 1.00 | 0.285 | 23.31 (1.09x) |
| `stat_k10_sp1024_2k` (7784) | 2.228332 | -0.0886 | 2.9345 | 0.7061 | 0.086 | 3.16 | 0.227 | 25.75 (1.20x) |
| `stat_k32_sp1024_2k` (7786) | cancelled by request after K=1 and K=10 | | | | | | | |

`scripts/compare_stationary_stream.py --candidate stat_k10_sp1024_2k
--control ffn_bag_ce_sp1024_2k --reference stat_k1_sp1024_2k`:
improvement -0.0886, memory-gain delta -0.4606, gate not met.

Reading. The buffer contents are the control's carry and more (the age-1
slot is exactly the previous post-final-norm hidden; the ring tests pin
it), and the reader used it: at `K=10` the null mass fell to 0.09 and the
attention's mean age was 3.2 steps, so it reached well past `h_{t-1}`. Yet
both arms recovered far less from context than the control (memory gain
0.71-0.77 against 1.17) and both were *better than the control with
context removed* (reset-latent 2.93-2.98 against the control's 3.306). That
is failure mode (3), the entry path, and `K=1` settles it against the
history: the read enters through a zero-initialized value map with no
normalization, so the backbone first learns the context-free solution and
the read then has to grow against it. Its RMS ended at 0.23-0.29 against
the unit-RMS observation, roughly a fifth of the control's carry, which
enters through `latent_norm` at unit RMS from step 0. More history cannot
fix a weak entry: `K=10` was 0.025 *worse* than `K=1`, and at 1.20x the
control's step time it also failed the performance requirement, while
`K=1` at 1.09x met it. Two smaller contributors: a convex mixture cannot
give two slots full weight at once, and with `W_v = 0` the query, key, and
null bias receive no gradient until the value map has grown.

Follow-up, pre-registered 2026-09-18: the `latent_norm` read entry. Same
ring and RoPE-by-age keys, but no null slot and no value map: the softmax
runs over the readable slots only (a lane with none reads zero, exactly the
control's zeroed carry), the mixed vector passes through the control's own
`latent_norm` (RMSNorm with learned gains) and enters at unit RMS from step
0, the query is zero-initialized so content addressing starts neutral, and
a learned per-head recency bias `-(age - 1)` makes the newest slot take
most of the mass at init (63% at `K=10`). With `K=1` this *is* the control's
forward (tested bitwise against `StreamingFFNModel(use_writer=False)` over
a page with resets), so `K=1` should reproduce 2.1397 up to run-to-run
noise and `K=10` can only add reachable slots. Read parameters drop to
131,368 (query, key, recency bias); the model has 13,783,848 parameters.
Success criterion unchanged: `K=10` beats the control by more than 0.005 at
no more than about 1.1x step time; `K=1` must land within 0.01 of the
control or the parity claim is wrong somewhere in the compiled path.
Likely failure modes: (1) `K=10 ~ K=1 ~ ce`: the FFN reader gains nothing
from selectable history at this scale even with the control's entry path;
(2) `K=10` below `K=1`: the recency-biased mixture blurs `h_{t-1}` at init
and the query never sharpens it back (would show as read_age well above 1
with no BPB gain); (3) step time above 1.15x at `K=10`, given the value
map's 262k MACs are gone but the `K` key projections remain. The weight
decay rule was restated as "weight matrices only" so the 2-D recency bias
table is not decayed toward zero; for the control and the value_map arms
the partition is unchanged. First submission at priority 2 (mlq 7789-7791)
was skipped: the GPU contract tests passed 15 of 16, with the K=1 bitwise
parity against the control confirmed, and the one miss was the 16-element
key-bias gradient on the first page at 2.02% against the 2% bound. That
gradient is cancellation-dominated by construction (softmax logit gradients
sum to zero across slots, so the shared bias keeps only the rotation
differences), so the bound now carries an absolute BF16-accumulation floor
of 2e-5. Resubmitted: mlq 7794 (`stat_norm_gpu_tests`), 7795
(`stat_norm_k1_sp1024_2k`), 7796 (`stat_norm_k10_sp1024_2k`); `K=32` is not run (the user cancelled the
value_map `K=32` arm and the `K=1` vs `K=10` comparison carries the
question).

Results (latent_norm entry, recency slope 1):

| arm | proxy BPB | vs control | reset-latent BPB | memory gain | read_age | read_rms (pre-norm) | step ms (ratio) |
|---|---|---|---|---|---|---|---|
| `stat_norm_k1_sp1024_2k` (7795) | 2.138618 | -0.0011 | 3.2898 | 1.1511 | 1.00 | 1.104 | 21.72 (1.02x) |
| `stat_norm_k10_sp1024_2k` (7796) | 2.151391 | +0.0117 | 3.2725 | 1.1211 | 1.53 | 0.984 | 23.29 (1.09x) |

`K=1` reproduces the control (0.001 better, run noise), so the whole
value_map loss was the entry path, and the compiled ring, age rotation, and
norm placement are validated end to end at run scale. `K=10` is failure
mode (2): its read age stayed at 1.53-1.59 from step 100 to step 2000,
which is the recency prior's own expectation (`softmax(-(a-1))` over ten
ages gives 1.58), so the query never sharpened the mixture back onto
`h_{t-1}` and never learned to address anything else. The blurred read
cost 0.099 BPB at step 200 against the control and the gap closed
monotonically to 0.012 by step 2000 without crossing. The pre-norm mixed
RMS of 0.98 says consecutive hiddens are nearly parallel, so the mixture
loses little magnitude, just precision. Speed is met: 1.09x at `K=10`.

Next arm (pre-registered): the same `K=10` with a steep recency prior
(slope 4: the newest slot takes 98% of the mass at init), so the model
starts as the control and the extra slots can only add. Outcomes: worse
than `K=1` means the slots hurt through the query path; equal means the
reader never finds a use for them at this budget (failure mode 1 for this
entry); better than the control by more than 0.005 is the first positive
result of the line.
Jobs: 7801 (GPU tests after adding `read_recency_slope`, 16 passed; the slope only
scales the `age_bias` init, so slope 1 is unchanged and the compare script
treats it as a read knob), 7802 (skipped: its `--after-success` was bound to the wrong
job id by a shell parse), 7803 (`stat_norm_k10_s4_sp1024_2k`).

Result (`stat_norm_k10_s4_sp1024_2k`, K=10, slope 4):

| proxy BPB | vs control | reset-latent BPB | memory gain | read_age (step 20 -> 2000) | read_rms | step ms (ratio) |
|---|---|---|---|---|---|---|
| 2.139679 | -0.00001 | 3.3031 | 1.1634 | 1.019 -> 1.033 | 1.085 | 23.31 (1.09x) |

Outcome "equal": the arm is the control to five decimals along the whole
trajectory (largest gap either way 0.009 BPB, at step 200), and the read
age moved from the prior's expectation 1.019 to 1.033 in 2,000 updates,
i.e. the query put 1.4% more of the mass on older slots than it started
with. The reader had ten exact copies of `h_{t-1}..h_{t-10}` at 1.09x
step time and never found a use for any of them. Combined with the
slope-1 arm this is a clean pair: mass off `h_{t-1}` costs (0.012 BPB at
63% newest), mass on `h_{t-1}` reproduces the control, and the query's
gradient does not move it in either direction. `comparison.json` in the
arm's directory carries the control and both slope-1 arms as references;
`K=1` at slope 1 is numerically the same model as `K=1` at any slope.

Review of the slope knob (review-diff subagent, after the run): five
findings, all applied. (1) The compare script refused same-line references
whose stationary source hashes differ, which every pre-knob arm now does;
references are now matched on the shared backbone and data sources like
the control, a missing knob compares at its default, and stationary hash
drift is reported under `config_differences[<arm>]["sources"]`. (2) The
`BufferRead` guard accepted `inf` (NaN bias) and skipped `value_map`; it
now matches the config contract on both entries. (3) A non-default slope
on `value_map` was inert but recorded; `validate` rejects it. (4) The
`-1e4` softmax fill was coupled to the slope's range; it is `-1e30` (fp32
logits, identical softmax for any reachable logit, uniform on a fully
masked lane). (5) Tests: `inf`/`nan`/`0.0` slopes, the model-side guard,
the `value_map` rejection, and a new `test_compare_stationary_stream.py`
over fabricated artifacts. The steep arm ran before (2)-(4); none changes
its arithmetic (its slope is finite, its entry is latent_norm, and every
masked logit underflows to zero mass under either fill). GPU tests
rerun as job 7806.

Reading. At this width, depth, and budget, direct access to
`h_{t-2}..h_{t-10}` adds nothing over the recursion through `h_{t-1}`,
which already carries what the backbone can use of them. This is the
third result of the same shape on this backbone (TBPTT through the carry
did not help, value_map and latent_norm reads did not help), and it is
the pretraining counterpart of the MiniCPM token-carry finding: the carry
is learnable and cheap, but the model does not extract a loss reduction
from it. The stationary line as specified (single read at the model
input, CE only, 2k updates) is closed as a negative result. The remaining
untested directions, none of which this line's budget justifies without a
new hypothesis: a read inside each block rather than at the input, a
larger recursion-free context (attention over the observations
themselves, i.e. a transformer), or an objective that rewards using the
history (the future-credit line, which also lost at 2k).

## 2026-09-17: Slot memory over token carry (pre-registration, no run yet)

An alternative to the direct hidden carry: the actor decides *where* its
belief is stored and *what* it reads back. Implemented as an opt-in extension
of MiniCPM `--token-carry` (`--token-carry --slot-memory`, schema
`minicpm5_vapo_slot_memory/v1`; `postraining/slot_memory.py`). Nothing in
the existing native, latent, or plain token-carry paths changes; slot
checkpoints and pending records are rejected by every non-slot loader,
by continuation preflight when the mode or geometry differs, and by stock
native-only evaluation.

**Hypothesis.** With `M=64` addressable slots the model can keep a small,
persistent working set of producer beliefs alive across thousands of tokens
instead of only the previous step's, and the joint (token, slot) policy
gradient can learn a write discipline (overwrite the least useful slot,
skip writes) that the fixed `h_{t-1}` carry cannot express. **Verifier.**
The same verifiable-math reward as the token-carry campaign, on
`verifiable_mixed_10k_20260917` with the sealed held-out panels; matched
starting checkpoint and identical rollout settings to `minicpm_carry_verifiable_10k_20260917`
(mlq 7792) except the two slot flags. **Success criterion.** Held-out
accuracy at the matched step count at or above the token-carry control with
non-overlapping bootstrap intervals, together with a read that is actually
used: `carry_actor_probe_null_mass` below 0.5 and `carry_actor_probe_read_age_mean`
above 1 (a memory that only reads the previous step reproduces the control).
**Likely failure modes.** (1) Null collapse: `slot_null_fraction` near 1 and
null mass near 1, the read is switched off and the run is the control with
extra RNG consumption; (2) write entropy collapse to one slot
(`slot_distinct_slots_per_trajectory` near 1), a one-slot ring identical to
the carry; (3) the joint log-prob adds `log(M+1)`-scale variance to the
ratio at init because the slot prior is uniform, so PPO clipping engages
on slot choices that have no effect on the read yet; (4) fused decode versus
packed replay drift larger than in plain carry because the read is an
attention over stored BF16 hiddens.

**Contract.** Step `t` consumes an input embedding, produces the post-final-norm
hidden `m_t` (same producer as token carry), samples `x_t` from the LM head and
`σ_t ∈ {0..M-1, ∅}` from `SlotChoiceHead(m_t)` (zero-init `Linear(H, M+1)`,
uniform prior, temperature 1, independent of `x_t` given `m_t`), then writes
`m_t` into slot `σ_t` (pure overwrite; `∅` writes nothing). The next input is
`E(x_t) + scale * (Wd E(x_t) + Wo read_t)` where `read_t` is a softmax over the
alive entries after write `t` plus a learned null key with zero value; the
query is `rope(Wq_tok E(x_t) + Wq_car m_t, t)`, keys are `rope(Wk m_i, i)`
applied at write time so age `t - i` is native, values are `Wv m_i`. There is
no direct `Wh m_t` term: the read is the only latent channel. `log π = log π_tok
+ log π_σ` with one shared advantage; the rollout engine stores the sum as the
behavior log-prob and behavior refresh recomputes both from replay.
Per-token records: `carry_hiddens` (BF16 `[T, H]`, unchanged) and
`slot_choices` (int16 `[T]`, `-1` = `∅`), `slot_count`. `alive[t, s] =
max{i ≤ t : σ_i = s}`; key `i` is visible to query `t` iff
`alive[t, σ_i] == i`, so at most `M` keys per query. Replay builds that
`(Q, M)` table per trajectory with global key offsets, packs it, and runs
compiled FlexAttention with a block mask derived from the table (no dense
`(Q, K)` scan); a dense reference backend exists for tests and the probe only.
Forced `</think>` never writes (`σ = ∅`, slot log-prob 0) and stays
excluded by `policy_mask`. The terminal slot choice `σ_{T-1}` is never read
(a cap-ended lane does not even sample it and records `∅`), so replay drops
its log-prob term (`slot_action_mask`): it has no causal effect and must not
collect the trajectory advantage. Rollout stores the full joint sum; behavior refresh
(unconditional after every rollout) makes the replay likelihood
authoritative. Episode boundaries clear the lane's alive table,
history, and pending-forced flag at admission; a whole rollout clears every
lane. At init `Wd = Wo = 0`, so step 0 and every later step reproduce the
base model exactly; the null key is zero so an empty memory reads a
well-defined zero vector rather than a NaN softmax.

**Null read.** Judged necessary, not optional: without it the read over an
empty memory is undefined at step 0 and after any prefix of `∅` writes, and
the policy has no way to express "read nothing" except by writing garbage.
The learned null key (zero-init, one per head) costs `heads * head_dim`
parameters. The alternative of reading zero when no slot is alive and
forcing a read otherwise was rejected because it makes the read
non-differentiable in the empty/non-empty boundary.

Verification so far: 24 CPU tests in `postraining/tests/test_slot_memory.py`
(alive table and mask against loop references, rollout-vs-replay visibility
and read agreement across an episode boundary, joint log-prob composition,
`∅` handling, forced tokens, packed replay against a per-trajectory oracle
with gradients, checkpoint/schema refusals); GPU numerical checks in
`postraining/tests/test_slot_memory_cuda.py`: mlq 7793 ran the pre-review
file; the compiled FlexAttention replay matched the dense reference (read
max abs 0.0039, gradient relative errors below 0.007) and the rollout-vs-replay
check failed only at the two terminal positions (the un-sampled terminal `∅`
under a peaked slot head), which is exactly the credit the review removed.
Resubmitted after the fixes as mlq 7798.
`scripts/evaluate_minicpm_vapo.py` accepts the slot schema (the actor's
slot geometry, head, and combiner keys are validated against saved args);
stock generate in `eval_hf_math` refuses it. A subagent review found and I
fixed: an unconditional forced-flag reset that broke the non-slot continuous
engine tests, the missing evaluation path, BF16 quantization of the null
mixing factor in the FlexAttention read, an unbounded slot count versus the
int16 record storage, stale joint log-probs on lane re-admission, and the
terminal-slot credit above. No training run has been launched; results are
appended below when one completes.

## 2026-09-17: Latent feedback ("temporal residual") on the nanogpt-mini transformer (pre-registration)

Context. The full-bandwidth transformer (arXiv:2608.08888) feeds the previous
position's top-layer state back into the input through a gated linear unit
and trains the recurrence with Jacobi passes (pass k carries pass k-1's
shifted top states; every pass is scored with next-token loss; no detach).
It reports validation-loss and downstream gains equivalent to roughly 1.5x
the tokens at 1B scale. Our MiniCPM token carry (additive, detached,
LayerScale 0.01, trained only by 1,000 RL steps) showed no benefit, and the
FFN-only streaming lines showed a huge memory gain (1.17 BPB) because the
carry was the model's only memory. The question here is what in the paper's
result is confounded and what is necessary, on a transformer that has
attention to compete with the carry, and whether a linear-attention memory
over the fully processed past makes more of the channel than a single
vector.

Confounds in the paper, each with an arm that isolates it:

1. Compute: feedback passes cost k forwards per step; the paper compares at
   equal tokens (1.28-1.5x compute). Arms: the baseline at 2,000 steps
   (matched tokens) and at 4,000 steps (matched compute for two-pass arms).
2. Auxiliary-objective effect: the pass-2 loss backpropagates into pass-1
   top states, shaping them to be useful inputs regardless of decoding. The
   paper reports gains with standard decoding too. Arm: `glu` with the
   carried state detached (`FB_DETACH=1`), which keeps the runtime channel
   and removes the representation-shaping gradient. Every arm also reports
   pass-1 loss (standard decoding of the same weights).
3. Fusion form: the paper asserts the gated fusion is necessary because an
   additive path lets the model ignore the state (shortcut). No ablation
   is shown. Arm: `add` (additive, zero-initialised map, RMS-normed).
4. Evaluation regime: the paper's validation losses use k parallel feedback
   passes; only generation runs the true recurrence. Every arm here reports
   Jacobi passes 1, 2, 3, 8 on the full panel and the true sequential
   recurrence (`val_bpb_seq`, KV cache, token by token) on the first
   262,144 validation tokens alongside pass 1 on the same subset.
5. Training signal: dense next-token supervision over many tokens versus
   our RL-only carry. Not a separate arm; the comparison with the MiniCPM
   line is the whole design.

New hypothesis (the "more use" design). The channel's worth should scale
with what attention cannot reach: every layer of every position already
sees the layer-matched past through the KV cache, so a fully processed
summary of the previous position adds little unless attention is limited.
Arms with a sliding attention window of the 64 previous tokens plus the
current one (`FB_WINDOW=65`, exactly the sibling slot line's `fifo` arm
with 64 slots, which is the windowed control) for the `glu` fusion and
the memory below test this. If the feedback gain is
larger under the window than at full attention, the temporal residual is
a substitute for attention reach, which is what a slot-limited KV cache
(the sibling line) would need.

`lam`: linear-attention memory over all earlier top states, read at every
layer. Keys and values are projections of the (jittered) carried top
states; each layer queries with its own projection of its attention input;
weights are `phi(q_t).phi(k_i) * lambda_h^(t-i)` for `i < t`, `phi = elu+1`
with a learned per-head temperature, normalised by their sum; the per-head
decays are learned from `(0.5, 0.9, 0.98, 0.999)` (half-lives 1, 7, 34, 700
tokens), so a head can be the paper's `h_{t-1}` or a long summary. At
decode time the memory is a fixed-size fast-weight state per head; in
training it is a masked matmul. The read enters through zero-initialised
projections, so the arm starts as the baseline. Compared with the paper's
channel, one Jacobi pass already gives every layer of position t the fully
processed states of all earlier positions rather than the previous one.
Extra parameters: 3.7M (18%) over the 20.0M baseline; reported, not
matched.

Common protocol. nanogpt-mini (6L/512d, 4 heads, seq 1024, 524,288-token
steps, Muon), `DATA_PATH=data/datasets/fineweb10B_sp1024`,
`VAL_TOKENS=1048576`, 2,000 steps, seed 1337, two passes every step from
step 0 (the paper introduces passes late and mixes 75/22/3; at 2k steps a
fixed two-pass schedule is the cleaner comparison), jitter 0.02, no prefix
mixin, loss = pass-1 + pass-2. Headline `val_bpb` is pass 2 (pass 1 for the
baseline). Trainer `pretraining/nanogpt_mini/nanogpt_mini_feedback_train.py`,
model `nanogpt_mini_feedback_model.py`, tests
`pretraining/tests/test_nanogpt_mini_feedback{,_gpu}.py`.

Arms (run names):
`nanomini_base_2k`, `nanomini_fb_glu_2k`, `nanomini_fb_lam_2k`,
`nanomini_base_w64_2k`, `nanomini_fb_glu_w64_2k`, `nanomini_fb_lam_w64_2k`,
`nanomini_fb_add_2k`, `nanomini_fb_glu_detach_2k`, `nanomini_base_4k`.

Success criteria and readings, fixed before the runs:
- The temporal residual "works" in an arm iff `val_bpb_seq` beats the
  baseline's `val_bpb_seq` (same subset) by more than 0.005 AND is within
  0.01 of that arm's pass-8 number (the recurrence reaches what the Jacobi
  passes promise). Pass-2 gains that vanish under the sequential eval are
  Jacobi artefacts.
- Necessary-ingredient readings: `glu` vs `add` (gating), `glu` vs
  `glu_detach` (gradient through the state), `glu` at 2k vs `base_4k`
  (compute), pass-1 of `glu` vs `base` (representation-shaping alone).
- Window readings: gain(glu_w64 over base_w64) vs gain(glu over base), same
  for `lam`. A larger windowed gain supports the substitute-for-reach
  hypothesis.
- `lam` vs `glu`: the memory earns its parameters if it beats `glu` by
  more than 0.005 on `val_bpb_seq`; the learned decays and temperatures are
  logged (`mem_decay{h}`, `mem_temp{h}`) to read what horizon it uses.

Review fixes before the runs (subagent review of the model/trainer): the
memory read is computed in fp32 with the [B,H,T,T] scores recomputed in
backward (checkpointed) so the training path matches the fp32 fast-weight
state of sequential decoding; the sequential recurrence is scored at step 0
as well as at the end; the checkpoint is written before the final
validation; the GPU tests hold the trainer's own gate tolerance (0.005 nats
per token) at T=1024 and pin the memory kernel over 1024 positions with the
production decays; a two-step trainer exercise covers every validation
branch (a path test, not evidence). Validation runs the model eagerly in
every arm (the baseline trainer validates through the compiled forward;
same math, so cross-arm comparisons are unaffected but a val line will not
reproduce an older nanogpt-mini log bitwise).

Failure modes: (a) Jacobi non-contraction: pass 8 or the sequential loss
worse than pass 2 (the paper's 3-pass mixin exists for this); (b) the
sequential evaluator disagreeing with pass 1 in `none` mode (a runtime
check raises); (c) `lam` decays drifting to 1 with reads averaging the
whole past (no addressing; `mem_temp` stuck at 1); (d) the 2x step time of
two passes making the compute-matched control the real winner.

Queue (2026-09-17): GPU contracts job 7814 (13 passed on the pre-review
files: lam two-pass step 250 ms per microbatch, peak 22.2 GiB). The first
chain (7815-7823) was cancelled at the baseline by request: the sibling
slot line's `full` arm is the base trainer bitwise and its `fifo` arm with
64 slots is the 64-token window, so `nanomini_slot_full_2k` and
`nanomini_slot_fifo_2k` serve as `base` and `base_w64` here; `base_4k`
(compute-matched control) is deferred. Post-review GPU contracts and the six
feedback arms (glu, lam, glu_w64, lam_w64, add, glu_detach) were requeued
chained at priority 2: tests 7825, then 7826, 7827, 7833-7836 (the two
windowed arms use `FB_WINDOW=65` = 64 past tokens + self, the slot fifo
visibility; their `_w64` names refer to the 64 slots). Job 7825 failed
(15 passed, 4 failed): the trainer path test could not get the device
because the test process still held the compiled models from the
training-step tests (now runs first), and the glu sequential-vs-prefix
check at real width failed the 0.005-nats gate bound in bf16 by 0.018-0.023
nats per token on random weights (add, lam and none passed). That is the
bf16 noise floor of a multiplicative recurrent path, not an algorithmic
difference (the fp32 CPU checks agree to 1e-4 and the test now pins the
identity in fp32 at real width too); it means `val_bpb_seq` versus the
Jacobi passes for `glu` carries a floor of roughly 0.005 bpb, which the
0.01-bpb criterion must be read against. Requeued: tests 7838, arms
7839-7844 in the same order. Job 7838: 25 passed, the lam trainer path
test failed with an out-of-memory in the compiled backward on the fp32
[64, 4, 1024, 1024] score buffer (the isolated step fit; with the val
cache and optimizer resident it did not). The memory read is now the
chunkwise form of decayed linear attention (chunk 128: intra-chunk [C, C]
decay mask, earlier chunks through the fast-weight state carried across
chunks), O(T C) memory in fp32 with no checkpointing; the dense form stays
as the test reference (`linear_memory_read_dense`) and the CPU tests pin
chunked = dense = recurrent state including padding of a non-multiple T.
Requeued: tests 7846, arms 7847-7852. Job 7846: 26 passed. Per-microbatch
(64 x 1024) compiled step and peak memory: none 71 ms / 7.3 GiB; glu 129 ms
/ 14.6 GiB (two passes); add 126 ms / 14.6 GiB; lam 179 ms / 19.5 GiB (was
250 ms / 22.2 GiB with the dense bf16 read); lam with the 65-token window
247 ms (the explicit-mask SDPA path is slower than the causal kernel).
bf16 sequential-vs-prefix gaps on random weights (nats per token): none
0.003, add 0.0007, lam 0.0005-0.002, glu 0.018-0.023; all modes exact in
fp32.

## 2026-09-17: Slot-limited KV cache with a learned write policy (nanogpt-mini)

Pretraining experiment on the nanogpt-mini testbed (6L/512d, sp1024
FineWeb, 2,000 updates of 524,288 tokens). Code:
`pretraining/nanogpt_mini/nanogpt_mini_slot_model.py` (alive table, mask_mod
and table-derived FlexAttention block mask, slot attention, slot head,
advantage, Jacobi passes, sequential evaluator) and
`pretraining/nanogpt_mini/nanogpt_mini_slot_train.py` (fork of the base
trainer; env knobs `SLOT_MODE`, `SLOT_M`, `SLOT_PASSES`, `SLOT_GAMMA`,
`SLOT_HORIZON`, `SLOT_ENTROPY`, `SLOT_DETACH`, `SLOT_BASELINE_DECAY`,
`SLOT_SEQ_TOKENS`, `SLOT_BACKEND`, all documented in its docstring).

**Hypothesis.** A transformer whose attention at every layer sees only the
token itself plus `M = 64` memory entries can learn *which* entries to keep
through a sampled per-token write action, and that learned write discipline
recovers a useful part of what a fixed 64-token sliding window loses
against full causal attention, at the same parameter count (plus a
`512 x 65` head) and the same data.

**Mechanism.** The memory entry of token `t` is its per-layer rotary'd
key/value pair (6 layers x (k, v) x 512 bf16 = 12 KB; 64 slots x 64 lanes
is about 50 MB). After the post-final-norm state of position `t` is
computed, a zero-initialised fp32 head samples `σ_t ∈ {0..63, ∅}`
(uniform at init); slot `σ_t` is overwritten with the entry of `t` (`∅`
writes nothing). The replay rule is `alive[t, s] = max{i < t : σ_i = s}`
(or -1) and key `i` is visible to query `t` iff `i == t` or
`alive[t, σ_i] == i`, so at most 65 keys per query; position 0 attends
to itself only. RoPE is the ordinary one (keys at their own position,
queries at theirs), so age is native to the score. The `(T, M)` table is
built per sequence by a cummax; the FlexAttention `BlockMask` is derived
from the table (`O(T M)` scatter into a `[B, QB, KB]` presence grid plus
the diagonal, `BlockMask.from_kv_blocks`, `mask_mod` inside every kept
block) rather than by scanning `T x T`; the compiled kernel runs outside
the outer compiled graph as the pointer lines do. A dense `[B, T, T]` SDPA
backend exists for tests only (`SLOT_BACKEND=dense`).

**Training.** Jacobi passes with shared parameters (the full-bandwidth
transformer's schedule, arXiv:2608.08888): pass 1 is a plain full-causal
forward (the reference, `CE^(1)`), its head samples `σ^(1)`; pass 2 attends
under the alive table of `σ^(1)` and yields `CE^(2)`; with `SLOT_PASSES>2`,
pass `k` uses `σ^(k-1)`. The token term is teacher-forced CE. The slot
term is a policy gradient with a discounted future-credit advantage:

```text
A_t   = sum_{u=t+1}^{t+H} γ^(u-t) (CE_u^(1) − CE_u^(k)) − b       γ = 0.9, H = 32
loss  = CE^(1) + mean_{k≥2} CE^(k)
        − sum_t stopgrad(A_t) · log π_σ(σ_t^(k-1) | s_t^(k-1))
        − SLOT_ENTROPY · sum_t H(π_σ(· | s_t^(k-1)))                    (default 0)
```

Sum-CE convention throughout (Muon/Adam are scale-invariant). `b` is an
EMA (decay 0.99 per microbatch, seeded by the first microbatch mean) of
the raw credit, one per later pass. `σ_{T-1}` is never consumed and gets
no policy gradient. `SLOT_DETACH=1` cuts the slot term's gradient at the
head; the default lets it reach the trunk through the top state.
`slot_head.weight` is a Muon parameter (matrix), its bias AdamW. All
passes share `x0 = norm(embed)`; the trunk math of pass 1 is bitwise the
base model.

**Three modes of the same trainer.** `SLOT_MODE=policy` (above, two
passes); `SLOT_MODE=fifo` (`σ_t = t mod 64`, no head, one restricted
pass: a 64-token sliding window in slot form, no `∅`); `SLOT_MODE=full`
(no restriction, one pass; tested bitwise against the base `GPT` forward
and state dict, so it is the line's own base control). Init reproduces the
baseline trunk draws under the seed (the head is registered last and
zeroed).

**Evaluation.** Every validation reports `val_bpb` (headline: the last
slot-restricted pass; the only pass in fifo/full), `val_bpb_full` (pass 1),
`val_slot_null` (fraction of `∅`), `val_slot_entropy`, `val_slot_age`
(softmax-mass-weighted mean age of attended non-self entries, over layers;
computed with a value-channel flex pass). At the final step the true
sequential regime is run token by token with per-layer slot banks on the
first 262,144 validation tokens with `σ_t` sampled from the sequential top
state (`val_bpb_seq`) and argmax (`val_bpb_seq_greedy`), plus
`val_seq_null`. Full mode skips it (causal SDPA is its own sequential
regime).

**Success criterion.** At 2,000 updates, policy beats fifo by more than
0.01 proxy BPB on `val_bpb`, and closes at least half of fifo's gap to
full (`val_bpb_full` of the full arm), i.e.
`(fifo − policy) ≥ 0.5 · (fifo − full)`. The sequential number must
track the parallel one (`val_bpb_seq − val_bpb` small compared with the
policy–fifo margin) for the parallel headline to count.

**Failure modes to watch.** (1) `∅` collapse: `val_slot_null → 1`, the
memory empties and the arm degrades toward a context-free model; (2) a
uniform policy that never sharpens (`val_slot_entropy` stays at
`log 65 = 4.17`), in which case the arm is a random-eviction cache and any
margin over fifo is the random-vs-window difference, not a learned
discipline; (3) advantage variance: `train_adv_std` large relative to the
signal, with Muon turning a noisy head gradient into a fixed-size
orthogonalised step (a known risk of putting a policy head under Muon;
the detach and entropy knobs are the levers); (4) Jacobi non-contraction:
`val_bpb_seq` far above pass-2 `val_bpb` because the slots the policy picks
from full-context states differ from what it picks from restricted states.

**Cost.** Policy mode runs two trunk passes plus the block-mask build and
a slot-restricted flex attention per layer: about 2x the base step
(base is about 580 ms/step at MBS 64 on the 5090). Fifo runs one
restricted pass; full runs the base. The measured numbers are in the GPU
test output and the run logs (`step_avg`).

**Verification.** 18 CPU tests in `pretraining/tests/test_nanogpt_mini_slot.py`
(alive table and visibility against loops with overwrite and `∅`; mask_mod
and table-derived block presence against the dense reference; fifo as a
sliding window; full mode bitwise the base; advantage against a loop;
policy-gradient reach with and without detach; entropy and baseline knobs;
no gradient for the terminal slot; sequential vs parallel and prefix
recompute with injected slots; three passes). GPU tests in
`pretraining/tests/test_nanogpt_mini_slot_gpu.py`: compiled flex vs dense
forward and every gradient at real dims, flex vs dense age statistic,
sequential vs parallel/prefix recompute at real dims, one compiled MBS 64 x
1024 policy step with peak memory asserted under 28 GB, and (added after
the first run) the compiled policy graph against the eager one on an
injected slot trajectory (loss, telemetry, EMA baseline buffer, every
gradient).

GPU test job mlq 7824 (first version): flex vs dense at 6L/512d, B=2,
T=1024, M=64 with 20% null writes agreed to fp32 per-token CE max abs
1.4e-6 (worst parameter-gradient relative error 1.9e-6) and bf16 CE max abs
1.9e-2 (worst gradient rel 2.5e-2, flex accumulates dK/dV in bf16);
sequential vs parallel per-token CE max abs 1.4e-6 (fp32) and 1.5e-2 (bf16,
mean 3.2e-3); prefix recompute agreed at every tested prefix; the
table-derived block mask builds in 0.63 ms (31 ms on the first call) and
skips 57.8% of the 128-blocks at that null rate. The MBS 64 training step
OOMed: the earlier tests had exercised more than Dynamo's default 8
distinct flex shapes, the compiled `flex_attention` silently fell back to
eager flex (the dense math reference, a 1 GB score matrix per layer), and
the backward ran out of memory. Fix: `compiled_flex_attention` raises
`recompile_limit` to 64 and sets `fail_on_recompile_limit_hit`, so a
fallback is an error and never a silent slow path. Resubmitted with the
parity test as mlq 7832; its numbers and the run job ids are recorded
below.

Chunkwise read review (subagent, after 7846): algebra, padding, broadcasts,
strict causality and the backward all verified against the dense form and
the recurrent state at production shapes (forward 8e-7, gradients 6e-7
relative); fp32 denormals at lambda 0.5, C 128 provably immaterial
(bitwise identical under simulated flush-to-zero); one compiled graph, no
breaks. Items applied: T = 0 and chunk <= 0 guards, the positivity
precondition and the 1e-6 * (1 - lambda) normaliser asymmetry documented,
a CPU test at the default chunk with T = 260 and the production decays
against both references. Follow-ups, not applied under the live chain
(math-neutral but the arms must share one source): saved-for-backward
memory is about 1.6 GiB per layer call rather than 128 MiB because the
[B, H, n, C, C] scores are kept twice (the decay mask is differentiable);
a custom autograd function saving only the masked scores would recover
it; a step-0 runtime gate binding the chunked read to the state in lam
mode (currently pinned by the GPU test only).

2026-09-18: `nanomini_fb_glu_2k` (job 7847) was killed by the user at step
900 for poor performance. Against the July baseline `nanogpt_mini_fullval_2k`
(identical training data for the first 1,000 steps, same schedule, 62M-token
validation set instead of the 1M-token subset): step 800 base 1.3948 vs
glu pass-2 1.3920 and pass-1 1.4092; step 900 glu 1.3735 / 1.3908 (base not
logged at 900; 1.3692 at 1,000). Reading: two-pass gated feedback is level
with the baseline at twice the step time (1,160 ms vs 535 ms per step) and
standard decoding of the same weights is 0.014 bpb worse; the pass-2 minus
pass-1 gap of 0.017 shows the channel is used, but only to recover what
feedback training cost. The remaining arms were not requeued; only
`nanomini_fb_lam_2k` was resubmitted (cull rule: pass-2 not ahead of glu's
1.4402 at step 500 kills it).

Speed follow-up (during the lam run): the memory read now dispatches to the
fused chunked kernel `fla.ops.simple_gla.chunk_simple_gla` on CUDA
(`FB_MEMORY_KERNEL=0` restores the fp32 chunked torch form, which stays the
CPU reference). The kernel is inclusive, so keys and values are shifted
right by one position and the numerator is multiplied by lambda to give
exactly the strict `lambda^(t-i)` read; the per-head decay enters as a
per-token gate because the kernel returns no gradient for its
data-independent `g_gamma`; the normaliser is a separate fp32 decayed key
sum (`decayed_key_sum`, chunked, no d x d state). Operands are bf16 with the
kernel's fp32 state, so the read carries bf16 rounding like the rest of
the network (the sequential evaluator keeps its fp32 state; the GPU test
prints the kernel-vs-chunked gap). GPU contracts queued as job 7860 behind
the lam run; the lam run itself (job 7854) uses the torch chunked form.
Baseline job 7856 was queued behind lam and cancelled at the user's
request ("don't waste time on base").

`nanomini_fb_lam_2k` (job 7854) was killed by the user at step 300 for
speed (1,655 ms/step, 2.8x the base). Its record: step 100 pass-2 1.7259 /
pass-1 1.7327 (same-subset base `nanomini_base_100`: 1.7922); step 200
1.5679 / 1.5779; step 300 1.4955 / 1.5060 (old 1k baseline, larger val set:
1.7458 / 1.5856 / 1.5159; glu: 1.7897 / 1.5920 / 1.5118). Decays stayed at
their inits (0.52, 0.91, 0.99, 0.999), head-0 temperature rose to 1.27.
Reading: the runtime read was worth 0.007-0.010 bpb (pass 2 minus pass 1)
for 1.45x glu's step; nearly the whole lead over glu and the base sits in
pass-1 decoding, i.e. in the trunk, and came through the second pass's
gradient into the earlier top states.

Kernel path (job 7860): kernel vs fp32 chunked at production geometry,
relative gaps out 0.4%, dq 1.5%, dk 0.6%, dv 0.4%, d(decay) 0.6%; 24
passed, three failures from the environment and a stale assert (the
trainer path tests collided on the default rendezvous port with another
process, now a free port per test; the dispatch-vs-dense 1e-4 assert was
left from the fp32 era and now compares the chunked form). Timings in that
job ran alongside another process and are not usable; a dedicated
read-timing test was added.

**Single-pass design (`FB_MODE=top`, pre-registered before its run).**
The killed two-pass arms showed the gain lives in the trunk and arrives
through the gradient the future positions send into the head-facing top
state via the memory keys and values. That signal does not need a second
pass: a strictly causal decayed linear-attention read over earlier
positions' *final* states is computable inside the ordinary parallel
forward (it needs only earlier positions, which the pass has), so the
memory is read once at the top, `s_t = norm2(x_t + W_o r_t)` with `W_o`
zero-initialised, keys and values from `h_t = norm2(x_t)`. Cost: one pass
plus one read and four 512 x 512 projections (about 3% over the base,
inference too). Arms: `nanomini_fb_top_2k` and `nanomini_fb_top_detach_2k`
(keys and values from `h.detach()`: the read stays, the gradient into the
earlier top states is cut). Every validation logs `val_bpb` (with the
read), `val_bpb_noread` (same weights, read removed), and at the deep
evals `val_bpb_seq` (true recurrence; the trainer gates it against the
parallel loss at 0.005 bpb since there is no Jacobi gap in this mode).
Readings: `val_bpb_noread` of `top` against the base curve isolates the
trunk-shaping effect; `top` minus `top_detach` isolates the gradient
ingredient; `val_bpb` minus `val_bpb_noread` is the read's inference value
at its true cost. Success: `top` beats the old 1k baseline curve by more
than the 0.02 lam showed at steps 100-300, and holds a lead at 2,000
against `nanogpt_mini_fullval_2k`'s 1.2856 (different validation subset;
the 1M-token subset read about 0.046 harder at step 100). Queue: GPU
contracts and the two arms chained at priority 1 (the user's cap).

`nanomini_fb_top_2k` (job 7870) was killed by the user at step 300 as
unimpressive; the detach arm (7871) was skipped with it. Record (val_bpb
with the read / `val_bpb_noread`): step 100 1.7638 / 1.7862, step 200
1.5912 / 1.6137, step 300 1.5120 / 1.5340, at 600-670 ms/step (1.05-1.15x
the base). Decays drifted to (0.45, 0.89, 0.98, 0.999); temperatures rose
to (1.44, 1.43, 1.26, 1.10), i.e. the short-horizon heads were addressing.
Same-step comparison (old 1k baseline, larger val set: 1.7458 / 1.5856 /
1.5159; same-subset base at step 100: 1.7922): top with the read is 0.028
better than the same-subset base at step 100 and level with glu and the
old baseline at 300 (-0.004), where lam pass 2 was 0.020 better and lam
pass 1 0.010 better. `noread` is not a baseline for this mode (the head is
trained only with the read present), so the trunk-shaping hypothesis is
not tested by it; what the arm shows is that a single read at the top,
feeding the head, is worth about what the paper's input fusion is worth
per token (nothing beyond noise at step 300) while lam's every-layer reads
of fully processed states were worth 0.02.

Reading across the line (all arms killed between steps 300 and 900; no
2,000-step number and no matched 1M-subset baseline beyond step 100):
- Confounded in the paper: compute. The gated input feedback (`glu`) with
  two Jacobi passes is level with the baseline per token at 2x the step
  time and 0.014 worse under standard decoding (step 800).
- Necessary for a per-token gain here: reads of the fully processed earlier
  states at every layer (`lam`), which is the Feedback-Transformer
  mechanism; that is exactly what needs the second pass, since in one
  parallel pass no layer can see earlier positions' final states. The
  paper's mechanism (one vector, fused at the input) and the cheap
  single-pass read at the top (`top`) do not carry it.
- The runtime read in `lam` was worth 0.007-0.010 (pass 2 minus pass 1);
  the rest of its 0.02-0.06 lead was in the trunk (pass 1), which only the
  two-pass training produced. `top` has the gradient into earlier top
  states and did not reproduce that, so the trunk gain is tied to the
  every-layer reads in pass 2, not to the gradient alone.
- Compute-matched, none of the arms beats the baseline given more steps
  at this scale: the base gains about 0.045 per 100 steps at step 300 and
  about 0.02 for 30% more steps at 2,000, against lam's 0.02 (other
  subset) per-token lead at 2.8x.
Not run: the paper's 75/22/3 pass mix (1.28x) on lam, the partial second
pass (re-run only the top k layers with reads, 1 + k/6), and the chunked
single-pass form (exact cross-chunk reads through the fast-weight state,
no within-chunk reads). The last is the only one at 1x FLOPs and loses the
short horizons the run was using.

## 2026-09-18: MiniCPM RL cycle — exact replay paths, per-step split-KV plan, open decisions

Cycle model at production shape (4×16 rollouts, 10k cap, 64 lanes): rollout
decode about 47s, cache release about 1s, behavior refresh 15–21s, update
40–57s, KL probe about 1.2s, checkpoint 1–2s. Decode is KV-bandwidth bound at
long contexts (64 lanes × 5k mean length × 24 layers ≈ 8 GB of KV per step),
so kernel-count trims in decode are worth a few percent at most; the large
levers (FP8 KV, rollout-replica behavior log-probs, hidden trunk balance,
tail/refresh overlap) all change numerics or learning and need ablations.

Implemented now, exact by construction (queued parity job 8088 must confirm
candidate parameter error at the unchanged-path repeat control, about 7e-9):

- LoRA power-of-two scale fold: scaling alpha/rank = 2.0 multiplies `lora_b`
  (1536×16 fp32) before the GEMM and the update is added in place. Scaling by
  2 commutes with bf16/fp32 rounding, FMA accumulation and the autocast cast,
  so forward values and all gradients are bit-identical (CPU test asserts
  `torch.equal` on values and the three gradients). Non-power-of-two scales
  keep the unfolded path.
- Packed SDPA concatenates token-major views; the HF head merge becomes a view
  instead of a second [tokens, 2048] copy per layer. Job 5784 rejected a
  layout candidate for 0.3–0.4%; this is that gain as a two-line change and is
  re-measured inside the bundled job, not claimed beyond it.
- Action-row gathers use `index_select` on the single packed row; backward is
  one `index_add` over unique positions instead of sorted `index_put_`.
- `_FrozenParameterStash.restore` copies from pinned backing non-blocking.
- Hidden-space trunk balancing records finiteness on the deferred device flags
  (two host syncs per shard removed on that opt-in path).
- Rollout scoring time is now telemetry (`rollout_performance/rollout_scoring_seconds`).
- Split-KV decode plans partitions once per captured step from
  `flash_sequence_lengths` (`postraining/split_kv_plan.py`), and the FA4
  custom op takes the shared plan as optional tensors; per-launch planning
  remains the fallback and the CUDA qualification test asserts bitwise
  equality of both paths across retirement/refill lengths. Expected saving:
  roughly 50–70 tiny kernels per step, on the order of 1–2s per 47s rollout.
  Microbenchmark job 8089 reports kernels per step and replay time.

Deferred, needing sign-off or ablation (documented in `postraining/TODO_MINICPM5.md`):
behavior log-probs from the merged-bf16 rollout replica (about 10s per refresh,
changes the ratio reference; opt-in flag exists), hidden trunk balance (2.6%
faster update, 2000-step ablation required), logit chunk 1024 (bitwise
qualification against 128 queued), compiled RMSNorm/RoPE regions and saved
log-sum-exp in the head backward (rounding changes; the compiled-SiLU backward
rejection at 5.5e-7 shows Adam amplifies any rounding change to lr-scale
parameter differences), refresh shard budget, KL-probe subsampling,
tail/refresh overlap on a second stream (design project; same-GPU memory
conflict with released KV), FP8 KV (kernel project), shared frozen bottom
stack for the critic (learning-affecting design change).

Results (jobs 8089 and 8090, both under two minutes):

- Parity: candidate parameter maximum error 7.45e-9 equals the legacy repeat
  control 7.45e-9 (exact-path repeat 6.68e-9). Update 17.85/18.13s → 16.62/16.63s
  (8.2% warm update speedup), peak allocation unchanged at 20.22 GiB. This is
  an isolated update measurement on the compile-experiment fixture; the full
  cycle gain is roughly 3–4s of the 40–57s update phase.
- Logit chunk 1024 vs 128: behavior log-probs and advantages bitwise
  identical; refresh 6.54s → 6.31s for 160k actions. Update-path gradients at
  1024 not compared.
- SDPA backends for GQA causal replay: flash available (no hard fail under
  `sdpa_kernel(FLASH_ATTENTION)`), efficient unavailable, cuDNN available.
  4096-token forward+backward probe: default 1.50ms, cuDNN 1.90ms, forced
  flash 3.52ms. The default dispatch is therefore not flash on this torch
  build; which kernel production replay runs needs a profiler confirmation
  before any backend pin (a pin changes rounding and needs parity).
- Split-KV plan hoist: 312 → 76 kernels per decode step, bitwise identical
  across full/mixed/half-retired/short length patterns. Replay time 11.69 →
  11.78ms at full 11,264-token length (KV-bandwidth bound, within noise),
  6.50 → 6.14ms mixed, 0.44 → 0.15ms short. The saving lives in the
  prompt-length and draining phases, consistent with the bandwidth model.

Not changed: the remaining per-layer `cumulative_length` reduction in the
compact cache (about 24 tiny kernels per step) could share one buffer across
layers; it touches the transformers cache contract and was left for a
separate measured change.

## 2026-09-18: CELF — CE-anchored latent flow over byte patches (pre-registration)

**Context.** The Cola controls (`pretraining/cola`) reproduce the Cola-DLM
structure at 120M: a VAE (separate causal encoder and decoder, per-position
16-dim Gaussian posterior, LayerNormed mean) plus a block-causal DiT flow
prior. Measured facts from the completed BPE controls (2k steps, 32 x 2560
positions per update):
- `cola_bpe_120m_v3_joint_2k` final validation: reconstruction 0.317 bpb,
  masked accuracy 0.144, flow MSE 1.36 per coordinate on unit-power latents,
  reference-KL 1.57 nats/pos, arithmetic-mean variance 0.133 (geometric
  0.038 from the entropy term), effective log-SNR 2.0.
- Fixed-anchor NELBO (job 8079, v3): 4.77 bpb at 16 and 32 Heun steps; v2:
  4.72 = reconstruction 0.33 + prior NLL 3.40 + posterior log q 0.99. The
  latent channel alone costs about 4.4 bpb. The 25M nanomini AR control
  reads 1.52 bpb at 400 steps on a different validation subset (not matched;
  order-of-magnitude context only).
- The variational terms are decorative at the used coefficients: beta x KL is
  0.027 nats/pos (stage 1), beta x negative entropy 0.003, reference-KL
  0.157, against reconstruction 1.0 and flow 1.36. The stage-1 variance
  (0.30) is a drift under lr 1e-4 (the first byte VAE saturated its clamp);
  stage 2 pulled it to 0.13 in 200 updates while the reference-KL spiked to
  13, then everything flatlined. The rate is the cost of an unoptimized noise
  level, not a learned tradeoff.
- Attachment: the flow gradient reaches the encoder through the noisy block
  and the velocity target; only the history stream is detached
  (`objective.py`, `model.py::_forward`). The frozen-encoder KL exists to hold
  the resulting predictability pressure back.
- Masked CE with a causal decoder and independent masking is next-token CE at
  15% of positions (the masked position decodes from a clean prefix); it does
  not force contextual content into the latent. Latents are token lookups.
- The byte Cola joint control (job 8081) was cancelled by request; there is no
  byte Cola NELBO. The BPE v3 figure (4.77) is the Cola reference.

**Hypothesis.** A deterministic encoder whose only anti-collapse term is the
cross-entropy of a decoder that reads latents at one fixed noise level, with
the flow loss attached on every side (history, noisy block, target) and no
KL / entropy / reference / EMA / isotropy term, learns latents whose
innovation is only what the context cannot predict. Concretely: its
fixed-anchor NELBO on the same literal byte stream is well below the Cola
reference at matched parameters and bytes per update, its rate term is the
majority of the gap, and its latent effective rank stays far from collapse.

**Design (`pretraining/celf`, architecture `celf_byte_v1`, 125.9M params vs
119.6M for byte Cola; parameter-matched, not FLOP-matched — patching makes
the step cheaper).**
- Bytes are patched 4 per latent; 32-dim latent per patch; blocks of 8
  patches (32 bytes); 2560-byte contexts = 640 latents = 80 blocks.
- Encoder: byte embeddings concatenated per patch, 4 pre-norm SwiGLU layers
  at 512, attention bidirectional inside a block and causal across blocks;
  output normalized to unit RMS per position (the only geometric constraint;
  it pins scale so MSE means something — CE cannot pin scale).
- The latent every consumer reads is the flow-path state at
  `decode_time` = 0.25: w = 0.75 z + 0.25 eps. The decoder (same body,
  head to 4 x 256 logits, reads only w, block-bidirectional) is trained at
  that noise level; the prior's history stream is w; the evaluation posterior
  is N(0.75 z, 0.0625 I) by construction. Gaussian capacity of one anchored
  latent: 53 bits per 4-byte patch (a rate ceiling, chosen above the 32-bit
  identity so context can be carried).
- Prior: 13-layer 640-wide AdaLN DiT (same as the byte Cola prior), two-stream
  training (history stream at time 0.25, flow stream at sampled t), mask:
  history queries see history in blocks <= own; flow queries see history in
  blocks < own and flow keys in their own block. Times are logit-normal
  rescaled to [0.25, 1]; the path below 0.25 is never used.
- Losses: reconstruction CE per byte (clean branch) + masked CE per masked
  byte (15% patch-level mask-token corruption of the encoder input; the
  masked branch feeds only the codec) + flow MSE per latent (coordinate
  mean). Weights 1/1/1 — a stated choice, not a tuned one.
- Optimizer: source Muon on attention/FFN matrices (lr 0.02, wd 0.05),
  AdamW elsewhere (3e-4; byte embedding 3e-3; scalars no decay), 40-update
  warmup, hold to 30%, linear cooldown to 5%. 2,000 updates of 32 x 2560
  bytes (81,920 bytes/update, as byte Cola), microbatch 8.

**Verifier and metrics.** Every 20 updates on 40 validation contexts:
objective terms, reconstruction bpb/accuracy at the anchor noise, masked
accuracy, flow MSE, `latent_effective_rank` (covariance-entropy rank of the
32 coordinates). Every 100: teacher-forced sampled blocks (8 Heun steps from
noise to 0.25 under true anchored history, 16 contexts): greedy byte accuracy
and decoder NLL at the sampled latents, plus decoder NLL at the posterior
sample. Every 200: `val_nelbo_bpb` = reconstruction + (log q - log p) with
the conditional flow marginal at 0.25 by Hutchinson divergence (8 Heun
steps, 1 probe, 16 contexts). Final evaluation: 16 and 32 Heun steps, 2
probes, over the 48 complete 2560-byte validation contexts (122,880 of
124,766 bytes), plus two 256-byte generations from 64-byte validation
prompts.

**Success criterion.** Final `nelbo_bpb` (32 steps) below 3.0 bpb — i.e.
more than 1.7 bpb under the Cola BPE reference — with reconstruction bpb
under 0.2 and effective rank above 16 of 32 at the end. Anything between 3.0
and 4.77 is "cheaper than Cola, still not competitive with AR" and does not
justify a second arm before an AR byte control exists. Above 4.77 is a
negative result for the CE-only anchor at this budget.

**Failure modes and their signatures.**
1. Token-lookup latents (semantic collapse): reconstruction bpb near zero
   early, flow MSE flat near its init (about 1.0 per coordinate), rate term
   dominating the NELBO, masked accuracy stuck near the unigram rate.
2. Predictability pressure eroding decodability (the both-sides-attached
   risk): `posterior_decode_nats_per_byte` rising over training while flow
   MSE falls; sampled byte accuracy not tracking reconstruction accuracy.
   This is the trigger for the second arm (identical, target detached).
3. Dimensional collapse under unit power: effective rank falling toward the
   32-bit identity floor (about 4-8) — the only case where an isotropy term
   would be justified.
4. Anchor too noisy for the identity: reconstruction accuracy plateauing well
   below 1 at 0.25 with capacity 53 bits — would call for `decode_time` 0.15.
5. Divergence-estimate noise: the 16-vs-32-step and probe deltas in the final
   evaluation bound the numerical part; they are not an error bar.

**Cull rule.** At update 600, kill if `val_nelbo_bpb` > 6 or
`latent_effective_rank` < 4 or reconstruction accuracy < 0.5.

**Queue.** GPU contracts 8108 -> full-shape qualification 8109 -> run 8110
(`celf_byte_126m_v1_2k`) -> evaluation 8111, chained on success, priority 0,
max-parallel 1, one attempt each, time limits 45m/45m/4h/2h.

### Amendment 1 (2026-09-18): v1 collapsed to a constant latent; staged arms v2

**Result of `celf_byte_126m_v1_2k` (job 8110, cancelled by request at
update ~580; no final evaluation, 8111 skipped).** Failure mode 3 in its
most extreme form, and much earlier than the cull point. Validation, 40
contexts:

| update | recon bpb | recon acc | masked acc | flow MSE | eff. rank | nelbo bpb (rate) |
|---|---|---|---|---|---|---|
| 0 | 8.00 | 0.000 | 0.000 | 1.995 | 15.2 | 21.85 (13.85) |
| 20 | 6.32 | 0.161 | 0.157 | 0.991 | 14.3 | |
| 40 | 4.83 | 0.161 | 0.157 | 0.795 | 18.2 | |
| 60 | 4.73 | 0.161 | 0.157 | 0.102 | 16.8 | |
| 100 | 4.70 | 0.161 | 0.157 | 0.051 | 23.2 | |
| 200 | 4.70 | 0.161 | 0.157 | 0.013 | 19.6 | 4.92 (0.17) |
| 400 | 4.70 | 0.161 | 0.157 | 0.004 | 17.1 | 4.75 (-0.00) |
| 500 | 4.70 | 0.161 | 0.157 | 0.003 | 14.5 | |
| 580 | 4.70 | 0.161 | 0.157 | 0.003 | 13.9 | |

Reconstruction accuracy froze at 0.1609 from update 20 onward — the
validation unigram rate (the decoder learned the byte marginal and nothing
else); 4.70 bpb is the unigram entropy. Sampled-block accuracy 0.1615 at
every diagnostic. Train flow MSE: 1.83 (10) -> 1.01 (20) -> 0.84 (40) ->
0.28 (50) -> 0.11 (60) -> 0.044 (100) -> 0.005 (500). The reported
effective rank stays 14-23 because per-patch unit power keeps the
covariance non-degenerate even when the latents carry no byte information;
the rank canary is therefore *not* a collapse detector for this mode and the
reconstruction-accuracy plateau is.

**Mechanism.** The encoder found the latent that is trivially predictable:
a constant (or byte-independent) z makes the flow target u = eps - z a
function of the noise alone, so the prior fits it exactly and the flow loss
goes to zero. The decoder head is zero-initialised, so at update 0 the CE
terms put no gradient into z at all, while the attached-target flow loss
(both-sides-attached) pulled on z at unit scale from the first update. By
update ~50 the encoder had collapsed and the CE gradient into z at that
point — through a decoder that had only learned the marginal — was too weak
to pull it back against the flow restoring force. The global optimum of the
objective is unchanged (a decodable latent saves ~13 nats per patch of CE
against at most ~2 units of flow penalty), so this is a race at
initialisation, not a statement about the objective. It is exactly the
"CE must be useful early enough" condition the pre-registration assumed
and did not enforce.

**What v2 changes (implementation, same package, same 125.9M model).**
- `TrainConfig.codec_steps`: a codec-only stage in which the prior is not
  executed (`Objective(..., flow=False)`) and only the encoder/decoder
  optimizers step. The prior joins at update `codec_steps` with fresh
  optimizer state. Optimizers are partitioned per component
  (`make_optimizers(model, config, component)`); the schedule is
  stage-relative (codec stage: 40-update warmup then hold; joint stage:
  warmup, hold to 30%, cooldown to 5%; Muon momentum ramp per stage).
  `stage_id` is logged on every metrics entry; sample/NELBO diagnostics run
  only in the joint stage. Resume invariants cover `codec_steps`.
- `LossConfig.attachment`: `all` (as v1: flow gradient reaches the encoder
  through history, noisy block, and target) or `history` (noisy block and
  target built from `z.detach()`; the encoder is attached only through the
  conditioning stream). Recorded in provenance `gradient_contract`.
- CPU contracts for the stage schedule and config validation; GPU contracts
  that the codec stage never touches the prior, that `history` changes only
  the encoder gradient, and that the component optimizers partition every
  parameter. Qualification now exercises both stages.

**Arms (pre-registered).**
- Arm A `celf_byte_126m_v2_staged_2k`: 300 codec-only updates + 2000 joint
  updates, `attachment=all`. Tests the user's claim on its intended terms:
  both sides attached, CE the only anti-collapse anchor, but from a latent
  that CE has already made informative.
- Arm B `celf_byte_126m_v2_staged_history_2k`: identical, `attachment=history`.
  Removes the endpoint/target gradient so the flow loss cannot pull the
  encoder toward predictability at all; the encoder still shapes the
  latents through the conditioning stream and CE. This is the control that
  separates "staging suffices" from "target attachment is the problem".
- Prediction: Arm A holds reconstruction accuracy above 0.9 through the
  joint stage if CE is a sufficient anchor; if it decays toward the unigram
  rate after update 300 while flow MSE falls, failure mode 2/3 is confirmed
  for both-sides attachment at this scale and Arm B is the surviving design.
  Arm B's flow MSE should stay higher than Arm A's for the same recon
  (nothing simplifies the target), so the NELBO comparison at 2300 is the
  decision.

**Cull rules.** At update 300 (end of codec stage): reconstruction
accuracy < 0.5 kills the arm (the codec alone cannot be at fault of the
prior). At update 900 (600 joint updates): reconstruction accuracy < 0.5,
or `val_nelbo_bpb` > 6, or flow MSE < 0.05 with reconstruction accuracy
falling, kills the arm. Success criterion as pre-registered (final
`nelbo_bpb` < 3.0 with recon < 0.2 bpb and effective rank > 16), now at
update 2300.

**Queue (v2).** GPU contracts 8126 -> qualification 8127 -> Arm A train 8128
-> Arm A evaluation 8129; Arm B train 8132 -> evaluation 8133 chained on the
qualification only, so a culled Arm A does not skip its control. Priority 0,
max-parallel 1, one attempt each, time limits 45m/45m/4h/2h. Jobs 8130/8131
were the same Arm B mis-chained behind Arm A's evaluation and were cancelled
before starting. Job 8111 (v1 evaluation) was skipped by mlq when 8110 was
cancelled.

### Amendment 2 (2026-09-18): Arm A result — staging fixes the collapse, the rate does not close

**`celf_byte_126m_v2_staged_2k` (job 8128), cancelled by request at update
~1970 of 2300; final evaluation 8129 skipped.** Validation, 40 contexts
(NELBO: 16 contexts, 8 Heun steps, 1 probe):

| update | stage | recon bpb | recon acc | masked acc | flow MSE | eff. rank | nelbo bpb (rate) | sampled acc |
|---|---|---|---|---|---|---|---|---|
| 100 | codec | 4.54 | 0.159 | 0.162 | - | 1.7 | | |
| 200 | codec | 3.06 | 0.401 | 0.162 | - | 8.3 | | |
| 300 | boundary | 1.22 | 0.812 | 0.236 | 2.00 | 16.1 | 15.10 (13.84) | 0.075 |
| 340 | joint | 1.25 | 0.785 | 0.245 | 0.99 | 9.9 | | |
| 400 | joint | 1.04 | 0.805 | 0.275 | 0.46 | 15.3 | 5.64 (4.56) | 0.068 |
| 600 | joint | 0.47 | 0.910 | 0.324 | 0.47 | 17.9 | 4.68 (4.19) | 0.065 |
| 800 | joint | 0.34 | 0.931 | 0.362 | 0.44 | 19.0 | 4.15 (3.79) | 0.066 |
| 1000 | joint | 0.27 | 0.945 | 0.389 | 0.44 | 19.6 | 3.96 (3.67) | 0.066 |
| 1200 | joint | 0.24 | 0.950 | 0.413 | 0.42 | 20.3 | 3.74 (3.48) | 0.066 |
| 1400 | joint | 0.20 | 0.957 | 0.434 | 0.42 | 20.9 | 3.61 (3.39) | 0.066 |
| 1600 | joint | 0.19 | 0.959 | 0.447 | 0.40 | 21.4 | 3.47 (3.26) | 0.069 |
| 1800 | joint | 0.17 | 0.966 | 0.458 | 0.41 | 21.6 | 3.34 (3.16) | 0.069 |
| 1960 | joint | 0.16 | 0.966 | 0.471 | 0.40 | 21.7 | | |

Train at 1970: recon 0.136 bpb, masked accuracy 0.50, flow MSE 0.375.
Recovery checkpoint at update 1516; diagnostic scoring of it at 16/32
steps and 2 probes over the full 48-context panel queued as job 8172
(`scripts/diagnose_celf_checkpoint.py`, outside the hash contract, output
`diagnostic_step1516.json`).

**What it shows.**
- The v1 collapse was an initialisation race, as diagnosed: with 300
  codec-only updates first, both-sides-attached joint training never
  collapsed. The transient dip at updates 320-360 (recon acc 0.834 -> 0.748,
  rank 14.5 -> 9.9) reversed by 400 and every CE term improved monotonically
  thereafter. CE as the sole anti-collapse anchor holds at this scale once
  the latent is informative before the flow gradient arrives. Both cull
  rules passed.
- The codec stage itself sat on the unigram plateau through update 100
  (rank 1.7) before escaping; the decoder's zero-init head plus unit-power
  latents make the first ~100 CE updates slow. Not a problem for staging,
  but it is the same plateau the flow exploited in v1.
- The result is nonetheless poor: NELBO 3.34 bpb at 1800 and still falling
  ~0.13 bpb per 200 updates. The 25M nanomini AR control reached 1.52 bpb at
  400 updates on the same bytes. Reconstruction is 0.17 bpb; the rate term
  is 3.16 bpb — 95% of the total. The latents are decodable and not
  predictable enough, i.e. failure mode 1 in a mild form: not token-lookup
  (masked accuracy rose to 0.47, so the encoder does carry context), but the
  prior cannot place enough density on the anchored code of the true patch.
- Teacher-forced sampled-block accuracy stayed at 0.065-0.069 throughout,
  which is the byte unigram collision probability. This diagnostic is the
  collision rate of the sampler with the data; a sampler whose 32-byte block
  conditionals were sharp would exceed the Renyi-2 floor (>= 2^-H, about
  0.35 for 1.5 bits/byte) on at least the first bytes of each block. Flat at
  the unigram collision rate means the prior's samples are marginal-like,
  so the high rate is a weak prior, not (only) a loose density estimator.
  Sampled NLL per byte rose 4.8 -> 9.9 as the decoder sharpened, consistent
  with samples landing on other patches' codes.
- Flow MSE plateaued at 0.40 per coordinate from update 400 while the rate
  kept falling; coordinate MSE is a poor proxy for the rate, which is what
  the encoder is implicitly trading against reconstruction. The objective
  the encoder optimises (CE + coordinate MSE) is not the bound we report.

**Candidate causes for the rate, to be separated before any redesign.**
1. Joint modelling of 8 patches (32 bytes) per flow block: one velocity
   field must resolve a 256-dimensional mixture with ~2^48 plausible modes
   per block. Isolated by `block_patches=1` (patch-causal prior; the codec
   becomes strictly causal), no other change.
2. Estimator resolution: 8 Heun steps and 1 probe on a flow that must be
   sharp near t=0.25. Job 8172 (16/32 steps, 2 probes) bounds this for Arm A;
   Arm B's final evaluation 8133 does the same for B.
3. Attachment: Arm B (`attachment=history`) separates whether the encoder's
   target-side gradient helped or hurt predictability.
4. Inherent continuous-density cost of scoring discrete bytes through a
   Gaussian posterior at fixed sigma (the dequantisation-style gap). Not
   testable by ablation here; it is the ceiling on this family if 1-3 come
   back negative.

**Arm C (pre-registered, queued after 8133 and 8172 complete because the
CLI change touches hashed sources): `celf_byte_126m_v2_staged_p1_2k`,
identical to Arm A with `block_patches=1`.** Prediction: if joint-block
modelling is the bottleneck, rate at update 1800 falls by at least 1 bpb
against Arm A's 3.16; if it stays within 0.3 bpb, the block size is not the
cause and cause 4 dominates. Cull as Arm A. Lesson recorded: no edit to any
file under the CELF hash contract while a CELF job is queued or running;
diagnostics that must land mid-flight go in unhashed scripts.

### Amendment 3 (2026-09-18): Arm B result — target attachment buys rate, not recon

**`celf_byte_126m_v2_staged_history_2k` (job 8132), cancelled by request at
update ~2100 of 2300; evaluation 8133 skipped.** Matched-update comparison
with Arm A (same validation subsets, seeds, and estimator settings):

| update | A recon | A rate | A nelbo | A flow MSE | B recon | B rate | B nelbo | B flow MSE |
|---|---|---|---|---|---|---|---|---|
| 300 | 1.22 | 13.84 | 15.10 | 2.00 | 0.86 | 13.85 | 14.74 | 2.00 |
| 400 | 1.04 | 4.56 | 5.64 | 0.46 | 0.36 | 7.40 | 7.79 | 1.37 |
| 600 | 0.47 | 4.19 | 4.68 | 0.47 | 0.13 | 5.62 | 5.76 | 1.32 |
| 800 | 0.34 | 3.79 | 4.15 | 0.44 | 0.07 | 5.00 | 5.08 | 1.27 |
| 1000 | 0.27 | 3.67 | 3.96 | 0.44 | 0.05 | 4.88 | 4.95 | 1.26 |
| 1200 | 0.24 | 3.48 | 3.74 | 0.42 | 0.04 | 4.44 | 4.48 | 1.23 |
| 1400 | 0.20 | 3.39 | 3.61 | 0.42 | 0.03 | 4.32 | 4.36 | 1.22 |
| 1600 | 0.19 | 3.26 | 3.47 | 0.40 | 0.03 | 4.03 | 4.06 | 1.21 |
| 1800 | 0.17 | 3.16 | 3.34 | 0.41 | 0.02 | 3.84 | 3.87 | 1.20 |
| 2000 | | | | | 0.02 | 3.58 | 3.60 | 1.17 |

(bpb; flow MSE per coordinate; effective rank A 21.6 / B 25.1 at 1800;
masked accuracy A 0.458 / B 0.476; sampled-block accuracy A 0.069 / B 0.076.)

**Reading.**
- Both-sides attachment (A) beats history-only (B) on the bound at every
  matched update after the prior joins: 0.53 bpb better at 1800. The
  target-side gradient made the latents about three times more predictable
  in coordinate MSE (0.41 vs 1.20) and cut the rate by 0.68 bpb, at a cost
  of 0.15 bpb reconstruction. The user's claim — attach both sides, let CE
  anchor — is supported as a rate/distortion trade at this scale, and the
  predictability pressure did not erode decodability (failure mode 2 did
  not occur in either arm).
- With the target detached, the encoder converges to an almost lossless
  code (0.018 bpb) that the prior cannot predict (rate 3.58 at 2000 and
  falling more slowly than A's). So the rate is not a matter of decoder
  noise or codec capacity; the prior fails to model the code either way,
  and attachment only shifts where on the rate/distortion curve the model
  sits. Neither arm approaches the 1.52 bpb AR control.
- Both runs' codec stages are configured identically yet diverged from
  update 100 (recon at 200: A 3.06, B 2.55; at 300: 1.22 vs 0.86). The
  codec-stage plateau escape is sensitive to GPU nondeterminism, so single
  runs of this family carry at least a few tenths of a bpb of run-to-run
  spread at the boundary; the joint-stage comparison above is nevertheless
  consistent in sign at every point.
- Cancelled by the user before completion; no final panel evaluation
  exists for either arm. Diagnostic job 8172 (Arm A recovery checkpoint,
  update 1516, 16/32 steps, 2 probes) remains queued at priority 0 behind
  other users' priority-1 work; Arm C was not queued.

**Status of the CELF hypothesis.** Confirmed: CE alone prevents collapse
once the latent is informative before the flow gradient arrives, and
target-side attachment improves the bound. Not confirmed: that a
CE-anchored latent flow is competitive with byte AR at 126M/2k updates —
it is about 2.2x worse in bpb, with the entire gap in the prior's rate
term. The open causes are the 8-patch joint block (Arm C, `block_patches=1`,
ready to queue), estimator resolution (job 8172), and the inherent
continuous-density cost of scoring bytes through a fixed-sigma Gaussian
posterior.

## Compiled LAM: performance-first 1,000-update ablation

Retain the ordinary nanoGPT-mini backbone (1024 vocabulary, 6 layers, width
512) and two-pass LAM. Parameter counts remain 19,958,784 for `none` and
23,635,976 for LAM. No detach, window change, backbone replacement, or loss
change was bundled into the optimization.

**Performance diagnosis and retained change.** The installed
`fla.ops.simple_gla.chunk_simple_gla` is decorated with
`torch.compiler.disable`, breaking the graph at each memory read. The retained
`torch.library.custom_op` adapter calls the pinned FLA forward/backward kernels
with their original gate scans and gradient conventions. FLA's forward chunk
states use bf16 storage and backward recomputes states in fp32; the normaliser
remains fp32. The adapter does not change these precision choices. The trainer
now requires `fullgraph=True`, and missing FLA on CUDA raises rather than
silently selecting Torch.

Exclusive RTX 5090 benchmark: B64/T1024/H4/d128, full 6L/512 model, five
warmups and twenty measurements of eight accumulated forward/backward
microbatches. Median per-microbatch times (optimizer excluded):

| Path | ms/microbatch | Peak allocated GiB |
|---|---:|---:|
| Baseline, first comparison | 61.00 | 7.40 |
| Torch chunked memory | 176.56 | 19.65 |
| Legacy FLA with graph breaks | 172.90 | 20.20 |
| Full-graph FLA, retained final source | 159.49 | 18.28 |

The final source benchmark agrees with the initial full-graph measurement
(159.36 ms). This is approximately 9.7% lower latency than Torch and 7.8%
lower than legacy FLA, not removal of the two-pass cost. Canonical artifacts:
`ablation_results/nanomini_lam_performance/{benchmark,final_benchmark}.json`.

Rejected candidates:
- Sharing query-independent memory preparation across six layers reached
  154.95 ms, but whole-model temperature-gradient relative maximum error was
  0.07057 against a predeclared 0.05 tolerance. Removed the candidate from the
  model rather than loosening tolerance.
- CUDA graphs initially failed gradient accumulation because graph-owned
  gradient buffers were overwritten. External stable buffers fixed correctness,
  but timing was 159.43 ms versus 159.33 ms without CUDA graphs: no win.
  Graph-private allocations also make its allocated-memory counter unsuitable
  for comparison; reserved memory remained about 18.4 GiB.
- Candidate implementations are archived in `candidates.zip` under the same
  result directory. The retained benchmark excludes these rejected candidates.

**Verification.** Final numerical-contract job 8432: 53 passed, 8 deselected
(short trainer runs and redundant real-shape timing cases not selected).
The retained regression compares the compiled adapter's outputs and all
q/k/v/gate gradients directly with FLA, including a non-chunk-aligned sequence.
Final benchmark job 8434 passed. Removed the old single-sample timing assertion;
performance evidence now comes from the isolated repeated benchmark.

**Matched quality result.** Jobs 8435 (LAM) and 8436 (baseline) both completed
1,000 optimizer updates and 51 validations, one every 20 updates including
step zero. Both used seed 1337, FineWeb10B-SP1024, 524,288 training tokens per
update, microbatch 64, sequence length 1024, the same 1,048,576-token validation
subset, and the same 70% cooldown schedule scaled to 1,000 updates.

| Run | Final BPB | Mean training ms/update | Whole-run seconds |
|---|---:|---:|---:|
| `nanomini_fb_base_perf_1k` | 1.3433 | 529.495 | 577.32 |
| `nanomini_fb_lam_compiled_1k` | 1.3243 | 1343.868 | 1525.58 |

Mean training time includes compilation but excludes validation/checkpoint
time. LAM's steady median is 1349.14 ms versus 528.48 ms for baseline.
Compared with the killed historical run's approximately 1655 ms/update, the
new mean is about 19% lower; that historical comparison is not a fresh paired
kernel benchmark.

**Keep decision:** LAM clears the >0.005 BPB threshold with a 0.0190 improvement
at matched updates/tokens, but costs 2.538x the training time. This does not
establish a matched-compute advantage. At 1,000 updates, pass one is 1.3426
(only 0.0007 better than baseline), and the second pass buys 0.0183 BPB. The
early hypothesis that most gain lives in the plain trunk is not supported at
the final budget. Passes 3 and 8 both score 1.3236, with no degradation through
eight passes. Sequential BPB 1.3028 is on a smaller 262,144-token subset and
must not be compared directly against the 1M-token headline.

Metrics/results remain in the two canonical run directories; checkpoints:
`logs/nanomini_fb_{base_perf,lam_compiled}_1k_final_model.pt`.
`ablation_results/nanomini_lam_performance/comparison.json` verifies completion,
validation cadence, matching environment, quality margin, and timing.

**Paper-inspired next hypothesis, not run:** arXiv:2608.08888v1 uses aggregate
75% one-pass / 22% two-pass / 3% three-pass training for its larger runs,
equivalent to 1.28x pass compute, not a cheap two-pass update. Its input-only
GLU is also cheaper than LAM's layerwise reads. An isolated 1,000-update LAM
schedule ablation could reuse `PassSchedule` from the separate full-bandwidth
trainer: 750/220/30 updates, progressively introducing feedback. Stage boundaries
in that utility are a local interpretation, not specified by the paper.
Using the measured roughly 2.6x two-pass cost gives an estimated 1.45x average
forward/backward cost for that mixture; this is a cost model, not a measured
LAM result. Do not alter the completed always-two-pass reference.

## LAM iteration: precision control and dropping layerwise memory reads

The user reported worse intermediate BPB for the optimized path and asked for
the speed/quality cost of removing each-layer insertion. Kept all training
runs at 1,000 updates with validation every 20 and reused the preceding matched
baseline and all-layer FLA run. Added `FB_MEMORY_LAYERS` to select zero-based
LAM insertion sites without removing any transformer blocks. Only selected
query/output projections are instantiated; parallel and sequential evaluation
use the same fixed mapping, serialized in checkpoint config. Default `all`
preserves existing full-layer checkpoints.

### Precision diagnosis

Original `nanomini_fb_lam_2k` and the optimized 1k run have real intermediate
differences, e.g. 1.5346 vs 1.5432 BPB at step 240. They are not a matched
end-budget comparison: the original schedule starts cooling at 600, the new
one at 300, and the original has no completed 1,000-step result. The schedule
difference cannot explain gaps before 300.

Matched reference job 8450, `nanomini_fb_lam_torch_1k`, completed all 1,000
updates using the fp32 Torch memory, with the same data/seed/batch/1k schedule:
**1.3248 BPB**, versus optimized FLA **1.3243**. The 0.0005 difference is below
the 0.005 material-improvement threshold; no sustained quality penalty was
demonstrated. The new fp32 control also scores 1.5432 at step 240, so that
historical gap is not unique to the FLA memory math. Exact transient causes
are not established by these single-seed controls; do not attribute all
historical differences to either precision or the schedule.

Same-checkpoint job 8454 evaluated the entire canonical 1,048,576-token window:

| FLA-trained checkpoint evaluator | Pass-one BPB | Pass-two BPB |
|---|---:|---:|
| Torch fp32 memory | 1.3426302185 | 1.3242711448 |
| Compiled FLA memory | 1.3426302185 | 1.3242790014 |

Identical weights/window/bytes; p2 difference is just **0.0000078566 BPB**.
This rules out a meaningful scoring discrepancy on that checkpoint, not
arbitrary training-trajectory effects. The fresh matched training timing is
1,465.981 ms/update for fp32 versus 1,343.868 for FLA: **8.33% lower latency**,
more reliable than the earlier historical ~1,655 ms comparison.
Queued reciprocal-checkpoint evaluator 8455 was cancelled by Main before
starting as redundant; no reciprocal measurement is claimed.

### Insertion ablation

Exclusive benchmark 8449, B64/T1024/6L/512d, five warmups and twenty samples
of eight accumulated microbatches, excluding optimizer:

| Variant | Parameters | ms/microbatch | Reduction vs six reads |
|---|---:|---:|---:|
| Plain mini | 19,958,784 | 60.95 | — |
| LAM, all six sites | 23,635,976 | 159.22 | — |
| LAM, sites 0/2/4 | 22,060,040 | 141.22 | 11.3% |
| LAM, site 0 only | 21,009,416 | 128.91 | 19.0% |
| Input-only GLU | 20,484,096 | 124.44 | 21.8% |

First-only quality job 8452, `nanomini_fb_lam_first_1k`, completed all
1,000 updates and 51 validations:

| Variant | Final headline BPB | Mean training ms/update |
|---|---:|---:|
| Plain mini (existing matched control) | 1.3433 | 529.495 |
| Six-site LAM, compiled FLA | 1.3243 | 1343.868 |
| First-site LAM, compiled FLA | 1.3265 | 1096.485 |

**Retain first-layer-only as the next control.** It costs just **0.0022 BPB**
versus six reads while saving **18.4% training latency** and **2,626,560
parameters**. It beats the plain backbone by **0.0168 BPB**, clearing the
>0.005 keep rule, but still costs **2.071x** baseline training time because
both transformer passes remain. Do not claim a matched-compute advantage.
First-only p1/p2/p3/p8: 1.3426 / 1.3265 / 1.3258 / 1.3257.
Sequential BPB 1.3058 uses a smaller 262,144-token window and is not directly
comparable to those full-panel values.

GLU job 8451 was cancelled by request; do not restart it or claim a 1k quality
result. Three-site LAM was benchmarked but not trained: first-only's small
quality cost made a middle-site run unnecessary for this iteration.

Verification job 8448: **70 passed, 8 deselected**. Added behavior coverage
for sparse causality, sequential/prefix agreement with live projections,
strict checkpoint roundtrip, and unchanged default/full-layer behavior.
Production compiled benchmark and both completed 1k training runs passed.
Artifacts: `ablation_results/nanomini_lam_iteration/`; checkpoint:
`logs/nanomini_fb_lam_first_1k_final_model.pt`; metrics/TensorBoard retain
their canonical run names.

### AdaRMSNorm proposal — not implemented or launched

The user suggested adaptive normalization at every layer instead of repeated
memory reads. The proposed controlled next arm keeps first-layer LAM, reuses
its strictly causal read as conditioning, and adds rank-32 scale-only
modulation to each block's attention/MLP RMSNorm inputs. Estimated additional
projection weights: roughly 0.21M, versus ~9.4M for dense six-output DiT-style
modulation across six width-512 blocks. No BPB improvement is presumed.
Initialize modulation to identity; zero residual gates combined with the
backbone's zero output projections can block learning. Both-pass training
cost remains; this idea is separate from progressive pass scheduling.

## Model-agnostic VAPO library, and KDA post-training

### The RL stack is no longer bound to MiniCPM5

The VAPO/RL post-training capability built against MiniCPM5 reached into
`LlamaForCausalLM` internals and a pinned model id/revision/vocab, so no
custom backbone could use it. It is now a library with declared contracts in
`postraining/vapo/model/`:

- `protocols.py` — `Capability`, `TrunkGeometry`, the `Readout` protocol, the
  `TrunkAdapter` ABC, the `RolloutEngine` protocol, and the `ModelFamily` ABC.
  A trunk *declares* what it can do (LoRA adapters, packed replay attention,
  static vs paged KV cache, FA4 decode, ...) instead of being recognized by
  class name.
- `hf.py` / `nano.py` — the two families. `MINICPM5_SPEC` keeps the exact
  pinned id, revision and 130,560 vocab; `NanoFamily` covers the nanoGPT-mini
  and KDA backbones through `model_io`.
- `lora.py`, `readout.py` — the shared primitives, moved verbatim so the
  qualified MiniCPM arithmetic is unchanged.

`minicpm_vapo.py` became `vapo/policy.py`; `MiniCPMVAPOPolicy`/`Critic` became
`VAPOPolicy`/`VAPOCritic`, and `.from_pretrained()` became
`.from_family(key, ...)`. `TrajectoryRecord` and `ContinuousTrainingGeneration`
were already model-agnostic and are the shared replay/rollout currency; the two
rollout backends (the measured HF engine in `fast_inference.py`, and
`vapo/rollout/nano_engine.py`) now sit behind one interface.

Deliberately *not* merged: the MiniCPM rollout engine's optimizations
(continuous lane refill, FA4 varlen replay, fused projections, CUDA-graph
decode) are heavily measured against Llama internals. `NanoRolloutEngine` is a
correct lockstep decoder with no measured throughput claim. Do not describe it
as fast until someone benchmarks it.

Family-specific escape hatches stay explicit: `TrunkAdapter.causal_lm` raises a
family-mismatch error rather than an `AttributeError` from inside a rollout
engine, and every packed-replay/static-cache call site lives in the Hugging
Face entrypoints.

### fla's TileLang KDA backward cannot run here — it hangs training

Any KDA training run through the post-training stack died at its first
backward, then **deadlocked while unwinding the exception**: all threads in
`futex_wait`, GPU at 0-8% and 25W, no progress and no crash. `mlq` reports such
a job as running forever.

The error, recoverable only via `PYTHONFAULTHANDLER=1` plus `SIGABRT`:

```
RuntimeError: TypeAttr `__ffi_repr__` is already registered for type index 132.
```

`fla` dispatches the KDA chunked backward to `fla.ops.kda.backends.tilelang`.
TileLang vendors its own TVM while `tvm_ffi` is also installed standalone, so
both register the same FFI TypeAttr for the same TVM type index. Reproduced in
about thirty lines with a bare `chunk_kda` forward/backward and no
post-training code, and `KDATileLangBackend.can_use()` is `True` by default in
this environment — so this was never specific to one entrypoint. Forward-only
paths never touch the kernel, which is why it surfaced only once KDA reached
training.

`postraining/kda_backbone.py` now defaults `FLA_TILELANG=0` at import, the same
pin the KDA GPU tests and ablation scripts already carry, which selects fla's
reference Triton KDA backward. Verified finite gradients for q/k/v/g/beta on
well-conditioned inputs (an early probe's `nan` came from unscaled synthetic
inputs blowing up the delta-rule recurrence, not from the kernel). Because fla
caches its per-class capability check when the backend module is imported, the
environment default only works if it lands first: `NanoKDABackbone.__init__`
asserts the resolved backend is not TileLang and fails with the diagnosis
otherwise, rather than hanging.

### Post-training the best pretrained KDA checkpoint

Base: `logs/nanogpt_gpt2_kda8_kkkdkkkd_triton_mbs32_optimized_2k_final_model.pt`
— the lowest `val_bpb` (1.1824) of the 63 KDA pretraining runs in `logs/`, full
2000 steps, validated on `data/datasets/fineweb10B_gpt2` like its peers so the
comparison is like-for-like. 64.0M parameters, 8 layers x 512 dim, GPT-2 padded
vocab 50304, KDA mixers on layers 0-2/4-6 with full attention on 3 and 7,
`train_seq_len` 1024. Runners-up: `kda5_outer_reduce_fixed_1k` (1.2068, 1000
steps) and `k3mix_v5_kda8_ref_2k` (1.2371).

The 1024-token pretraining window governs every budget. Layers 3 and 7 carry
half-truncate RoPE with no extrapolation story, so nothing may pack or sample
past it. The shipped defaults did exactly that — SFT packed at `--seq-len 4096`
and its sampling gate ran `512 + 768` — because they were tuned for the
8k-context K3 lineage. `sft_trace_train.py` now derives all three from
`backbone.train_context_tokens` and refuses a gate budget that exceeds it.
Measured on the corpus: prompts are p50 43 / p99 136 / max 941 tokens, so a
256-token prompt covers 99.7% of them, and 35,894 of 36,286 documents (98.9%)
fit the packed 1024 window.

Pipeline: `scripts/launch_kda8_posttrain.sh {sft|rl}`, every value pinned
explicitly. SFT runs the canonical `sft_traces_v6_answer_bare_a1swap10k`
corpus for three epochs at 8 rows x 1024 x 4 accumulation — 32,768 tokens per
step, matching the K3 reference recipe's 2 x 4096 x 4 so its `lr-scale` and
warmup carry over. RL then runs `vapo_broad_v6_bare`, which binds
`sft_corpus_sha256` to those exact corpus bytes; `train_latent_vapo` rejects a
base checkpoint whose recorded `traces_sha256` differs, so the chain is
hash-bound end to end.

RL uses `cot`, not `latent`: latent requires `stream_steps >= emitted + 1` and
a 1024-token window leaves only 768 stream slots after a 256-token prompt, so
latent would cost a third of the response budget. `cot` is also the token-only
control, which is the right first measurement on a new backbone. Latent at
`--continuation-tokens 512` is the follow-up ablation.

### UltraData SFT is gated, so it is not in the pipeline

`openbmb/UltraData-SFT-2605` is the natural SFT companion to the pinned
`UltraData-RL-2609` — a `think` split across Math/Code/Knowledge/IF, with 74 GB
of Math traces over 250 shards, which the existing bounded byte-window
acquisition in `ultradata_data.py` could sample the same way. It is **gated**:
`Access to dataset openbmb/UltraData-SFT-2605 is restricted... Please log in`,
and there is no HF token in the environment or any auth path in
`data_acquisition.py`. `openbmb/UltraData-Math` is public but is pretraining
web math, not SFT traces; `UltraData-SFT-Agent-2609` is tool-use/agent data.
Adding the SFT source needs accepted dataset terms plus a token; until then the
pipeline uses the repository's verified-trace corpus.

## 2026-09-21 — Arithmetic is absent, not weak; SFT corpus and RL mixture rebuilt

### The measurement that reframed everything

The arithmetic capability probe (`postraining/arithmetic_probe.py`,
`data/math_drills/v4/probe.jsonl`, 1,920 items disjoint from drill training)
run against the 3-epoch v6 SFT policy:

    overall 0.000 over 1,920 items
    add_integer      0.000  3d=0.00 4d=0.00 5d=0.00 6d=0.00
    sub_integer      0.000  3d=0.00 4d=0.00 5d=0.00 6d=0.00
    ... all 15 families 0.000 at every digit count ...
    ungradable 1,497  unterminated 1,385  malformed 1,407
    lenient 0.000 (scraped tail answers credited)

Zero at three-digit integer addition, and zero again under lenient grading,
so this is not the completion contract costing us credit. The primitive is
absent. That is consistent with the 2026-08-06 audit recorded in
`postraining/math_drills.py`: 0.115% of the pretraining mix was core
arithmetic drill and none of it showed working.

Everything else follows from this. The earlier RL result — 0.00 on every
DeepMind interpolate module, 0.20% on dapo-math-17k, 2.73% on gsm8k — was
never a tuning problem.

### The 3-epoch SFT was memorisation

Holdout completion CE fell 3.9449 -> 0.9009 over three epochs on ~36k
self-generated traces, beside the all-zero probe above. A near-zero training
loss next to an absent primitive is memorisation of a small trace set.
`postraining/data/sft_traces_v6_answer_bare_a1swap10k.parquet` and both
3-epoch checkpoints (`kda8_sft_v6_bare_e3`, `sft6_bare_a1swap10k_e3`,
245 MB each) are deleted; `metrics.jsonl`, `result.json` and
`gate_transcripts.json` are kept in each run directory beside a `RETIRED.md`
so the negative result stays auditable. `--epochs` now defaults to 1.

### UltraData: which repo is actually usable

- `openbmb/UltraData-SFT-Agent-2609` is public (54 GB, ~500k samples) but is
  the wrong corpus for a 1024-token, tool-less, 30M-param policy. Measured
  over 1,870 real records with GPT-2 BPE: **100% of samples exceed the 1024
  context** in every one of its four domains. Median tokens per sample —
  Code_Agent 73,966 (median 108 messages), General_Agent 39,095,
  Search_Agent 37,090, Tool_Use 8,656. It targets MiniCPM5-2B and trains
  function-calling against 6-25 tool schemas.
- `openbmb/UltraData-SFT-2605` is the core-domain SFT corpus that would fit
  (319 GB; `think/Math` 74 GB, `think/Code` 177 GB, `no_think/Math` 30 GB,
  and its `think` split maps straight onto our `<think>`/`<answer>`
  contract). It is `gated: auto` and returns **HTTP 401 for both its data
  files and its README** with no token in the environment, so its record
  schema cannot even be inspected. `prepare_sft_corpus.py` declares it in
  `CREDENTIALED_ADAPTERS` and refuses with that explanation rather than
  reporting an unknown-source typo.
- `openbmb/UltraData-RL-2609` is public and already adapted at a pinned
  revision by `postraining/ultradata_data.py`. This is the MiniCPM RL corpus.

### New SFT corpus: `postraining/prepare_sft_corpus.py`

Streams published instruction corpora into the schema `sft_trace_train`
already consumes (`source`, `problem`, `document`, `final_answer`,
`verified`, `doc_tokens`), with `document = problem + <think>\n{solution}\n
</think>\n<answer>{answer}</answer>` because the answer-fence contract
appends no instruction suffix. Adapters declare a corpus instead of sniffing
it; rows are dropped, never reshaped.

- `openmathinstruct2` (nvidia/OpenMathInstruct-2, CC-BY-4.0, 12.6 GB / 32
  shards / ~14M rows). Worked step-by-step solutions; measured 98.5% of
  documents fit 1024 tokens, median 313, p90 676.
- `math_drills` (local `data/math_drills/v4/drills.parquet`, 2M rows). The
  only corpus here that shows carrying, borrowing, place value and
  digit-by-digit long division. Its own `document` column carries the retired
  plain `Answer:` framing and is ignored; the document is rebuilt under the
  canonical contract.

Evaluation protection is part of the corpus contract, not a later filter:
`problem_source` values in `excluded_provenance` are dropped by the adapter
and the exclusion is recorded in the manifest. For OpenMathInstruct-2 that is
`gsm8k` and `augmented_gsm8k` — 17.5% of rows — because GSM8K is an
evaluation set here. A corpus that cannot report per-row provenance cannot be
admitted.

### RL mixture v8: weighted by measured learnability

The v7 mixture gave dapo 28/64 of every pool and DeepMind interpolate 20/64,
i.e. it weighted most heavily the two sources a frozen-policy gate scored at
0.20% and 0.00%, and gave gsm8k (2.73%, the only source with signal) 8/64.
VAPO's advantage is group-relative, so a uniformly wrong group contributes
exactly no policy gradient: 87% of every rollout pool could not teach
anything.

| source | v7 | v8 | gate accuracy |
|---|---:|---:|---:|
| deepmind_easy (train-easy) | — | 48 | not yet measured |
| ultradata_math | — | 8 | not yet measured |
| dapo | 28 | 8 | 0.0020 |
| deepmind (interpolate) | 20 | 0 | 0.0000 |
| mbpp | 8 | 0 | 100% format-ineligible |
| gsm8k | 8 | retired | 0.0273 (eval set) |

- **`deepmind_easy` is the `train-easy` tier**, not the bench panel. All
  three `deepmind-*` parquets draw the same 18 modules from the `interpolate`
  (test) split; "easy" in `deepmind-interpolate-easy` names the *module*
  selection, not a difficulty tier, which is why the bench panel and the
  `deepmind` training source score identically at 0.00%. `train-easy` is a
  disjoint training split at markedly easier surface difficulty — compare
  train-easy `What is -5 - 110911?` against interpolate `What is the
  difference between -221017 and -1429.06?` — and the 144 bench problems are
  excluded from it by question text. 72,000 prompts, 4,000 per module.
  `scripts/build_deepmind_rl_prompts.py` now takes `--source-dir`/`--split`
  and refuses a `--split` that does not name the directory, so the manifest
  cannot misreport which tier the rows came from.
- **`gsm8k` is retired from the mixture.** `RETIRED_SOURCES` names the reason
  (training RL on it would make every reported gsm8k number a training-set
  score) instead of surfacing as an unknown source.
- **`mbpp` and `deepmind` ship at quota 0**: available, off by default, and
  they require an explicit `--quota SOURCE=N` to contribute.
- **`--sft-corpus` is now required**, not defaulted to a deleted file.

### UltraData-RL-2609 generalized for this policy

`postraining/task_data.py` is a pinned MiniCPM campaign: it hard-errors
unless `--context-tokens` is exactly 10000, tokenizes with MiniCPM5-1B and
bundles CodeContests. Rather than re-acquire 4 GiB to change a tokenizer,
`scripts/build_ultradata_math_rl_prompts.py` promotes the Math rows of the
existing hash-bound extraction into a standalone pool: 2,973 unique prompts,
deduplicated by `original_query_sha256`, ordered by content hash, bound to
revision `e6ecfa73`. Math only — measured against GPT-2 BPE, its math prompts
are median 132 tokens with 95% inside a 256-token budget, against a 502-token
median (16% fit) for Code and 4,036 (0% fit) for Long_Context.

Its prompts are rendered with the MiniCPM5 chat template, so
`postraining/math_prompt.py` now registers those clauses:
`ULTRADATA_REASONING_INSTRUCTION`, `ULTRADATA_FINAL_ANSWER_INSTRUCTION`,
`ULTRADATA_BUDGET_SUFFIX` and the three code-agent clauses. The budget
sentence matters beyond redundancy: it states 9k tokens, which is false for
every model in this repository, so leaving it in a prompt would train the
policy against a context it does not have. Because a budget clause contains
no `Answer:`, the existing fail-closed check could not see it, so
`CONTEXT_BUDGET_DEMAND` now matches the shape rather than the one registered
literal — a corpus rebuilt at a different budget fails closed instead of
drifting. 3,599 of 3,600 rows canonicalize; 4 rows across both extractions
carry a literal `Answer:` inside the problem body and are quarantined with
their reason in the manifest rather than reshaped.

### Still blocked

`UltraData-SFT-2605` needs a Hugging Face token that has accepted its terms.
`hf auth whoami` reports "Not logged in" and `hf env` reports
`Has saved token ?: False`. Everything else in this entry is done.

## 2026-09-21 — One-epoch SFT restores arithmetic; multiplication is the bottleneck

### The measurement that reframed everything

The retired three-epoch v6 lineage scored **0.000 on all 15 arithmetic-probe
families at every digit count**, over 1,920 items, and 0.000 leniently as
well. That is not a weak skill; the primitive was absent. It made the 0.00%
RL results a capability gap rather than a tuning problem.

Job 9040 (`kda8_sft_omi2_drills_e1`) is the reply: **one** epoch over 931,057
decontaminated documents (293.8M tokens) — OpenMathInstruct-2 plus worked
arithmetic drills — instead of three epochs over a few tens of thousands of
traces.

### Held-out sampling gate (job 9040)

The panel comes from `split_holdout`, a deterministic problem-level split, so
these are held-out problems, not training reward. They are nonetheless
*in-distribution*: this is the corpus's own held-out split, not transfer.

| metric | v6 (3 epochs) | 9040 (1 epoch) |
| --- | --- | --- |
| accuracy | 0.00% | 26.17% |
| structural format | — | 84.86% |
| terminated | — | 85.94% |
| mixed-prompt fraction | 12.5% | 39.84% |
| holdout completion CE | 0.9009 | 0.4660 |

The mixed-prompt fraction is the one that governs what comes next. A
uniformly-wrong group produces zero policy gradient under group-relative
advantage, so at 12.5% the RL stage had almost nothing to learn from; at
39.84% it does. Falling CE was deliberately *not* read as success — that was
the signal that misled the v6 lineage, which reached 0.9009 CE while scoring
zero everywhere.

### Arithmetic capability probe (job 9041)

Construction-disjoint panel, 1,920 items, greedy decoding. Overall **0.516**,
from 0.000.

```
add_integer     0.977   (6d=0.94)    mul_integer     0.422
compare_decimal 0.945                div_integer     0.281
percent_of      0.875                fraction_mul    0.195
sub_integer     0.859   (6d=0.75)    div_decimal     0.117
round_decimal   0.844                fraction_add    0.023
add_decimal     0.812                mul_decimal     0.000
unit_convert    0.742                percent_change  0.000
sub_decimal     0.648
```

All 15 families had 12,542-58,094 drills in the training corpus, so the zeros
are not data gaps. `mul_decimal` scoring exactly 0.000 while `mul_integer`
scores 0.422 looked like a grading artifact, so the transcripts were read
rather than trusted. It is not an artifact. The model has learned every
*procedure* and fails on one *primitive*:

- `fraction_add`: LCD of 65 and 61 computed correctly as 3195, method
  correct, but `43/65 = 2187/3195` (2115 is right) — the cross-multiplication
  is wrong.
- `percent_change`: method correct, but `29380 x 50 = 2945000`
  (1469000 is right), then `29380 - 29450 = 29450` — it emitted the
  subtrahend.
- `mul_decimal`: correct long-multiplication scaffold and correct
  decimal-place rule ("1 decimal place and 1, so the product has 2"), but the
  partial-product addition chain drifts (`865200 + 360 = 865160`).

So multi-digit **multiplication** is the bottleneck, and every family
downstream of it inherits its error rate; addition, subtraction, comparison
and rounding are solid at 0.81-0.98. One `mul_decimal` trace also hit the
512-token cap mid-multiplication, which a longer context fixes directly.

The ordering does not track data volume — `mul_decimal` had 37,190 drills and
scores 0.000 while `percent_of` had 16,986 and scores 0.875 — so it tracks
compositional depth instead.

### Consequences

- The arithmetic drills work and stay in every subsequent corpus.
- One epoch over a large corpus beats three over a small one, decisively.
- Reasoning traces need room: the next corpus is built at 5,120 tokens.

## 2026-09-21 — Rollout throughput is not a bottleneck; single-repeat benchmarks are

Measured generation throughput on the job 9040 SFT checkpoint
(`postraining/runs/kda8_sft_omi2_drills_e1/sft_final_model.pt`) through
`train_latent_vapo --rollout-only` against the v9 mixture, cot mode, 256
prompt + 768 continuation tokens. Reported metric is the trainer's own
`useful_actions_per_second` (generated tokens/s) with `peak_vram_bytes`.

**The headline result is a measurement lesson.** With
`--rollout-only-repeats 1` (jobs 9082, 9083) the width curve looked like a
clean bottleneck: 512 rows/wave 13.5k tok/s, 1024 rows 15.6k, 2048 rows
25.3k, 4096 rows 33.9k. All four numbers were `torch.compile` warmup. The
same 2048-row config re-read 18.7k tok/s in a second single-repeat job at
identical flags and seed — a 35% swing I initially mistook for GPU clock
noise. At 4 repeats per config, discarding repeat 0:

| rows/wave | groups x samples | steady tok/s (reps 1-3) | peak VRAM |
| --------- | ---------------- | ----------------------- | --------- |
| 512       | 32 x 16          | 82-93k                  | 3.5 GB    |
| 1024      | 64 x 16          | 111-115k                | 6.3 GB    |
| 2048      | 64 x 32          | 100-119k                | 12.1 GB   |
| 4096      | 64 x 64          | 81-129k                 | 23.6 GB   |

Repeat 0 reads 14-16k tok/s at every width (23-44 s collect) against 3-7 s
once warm. Throughput saturates at ~113k tok/s from 1024 rows onward, so the
extra width buys nothing and costs 2-4x the VRAM. `--rollout-groups 64` with
`--prompts-per-rollout 64` is the production setting (job 9086); 64 is also
the v9 mixture's cycle length, which `rollout_window_source_quotas` refuses
to exceed.

`act/traj` (~355), `trajectories`, `ended_fraction` (~0.85) and
`think_format_fraction` (~0.80) were identical across every config and flag
combination, which is what makes these throughput comparisons and not
behaviour changes.

### The three default-off decode flags, now A/B'd

Their code comments in `postraining/vapo/config.py` each said "off until the
A/B says so". It now says so, and the answer is no for all three:

- `--rollout-flex-decode`: ~93k tok/s against the default ~113k, and a 93 s
  first-repeat compile against 23 s. A regression.
- `--rollout-flex-decode --rollout-graph-decode`: ~65k tok/s. A larger
  regression.
- `--rollout-tail-graph`: 117-125k against the default's 100-119k at the same
  width, i.e. indistinguishable. Its warmup repeat matched the default's to
  within 1% (16,554 vs 16,664 tok/s, 43.7 vs 43.5 s), which is what exposed
  the warmup artifact in the first place.
- `--rollout-scheduler continuous_refill`: raises `decode_step_utilization` to
  0.83 from lockstep's 0.45 exactly as designed, but collects 2048 rows in
  258 s against lockstep's ~6 s. The per-step scheduling and paged-attention
  overhead swamps the utilization win by ~40x. Not viable at this model size.

Note that `decode_step_utilization` is mean actions over the chunk's LONGEST
trajectory, so 0.45 is an upper bound on wasted decode work rather than a
measured loss — dead-row compaction claws some of it back, and `should_compact`
disables itself only under a preallocated arena, which is what the flex and
graph flags introduce.

**Conclusion: generation is not the RL bottleneck.** At ~113k tok/s a
1024-row pool of 768-token continuations collects in ~3.3 s. Any future
optimization effort belongs on the replay/update half of the step, which
`--rollout-only` does not exercise.

### Full-step attribution (job 9087, production widths)

`--rollout-only` measures generation alone. A real 60-step run at the chosen
widths (64 prompts x 16 samples, `--rollout-groups 64`, 1024 rows/wave) gives
the whole step. The two phases report under different keys, and they disagree
about which half dominates:

| phase | n | pool/total | rollout | update | rollout share |
| ----- | - | ---------- | ------- | ------ | ------------- |
| `value_warmup` | 50 | 7.14 s | 3.25 s (`collect_seconds`) | 3.80 s | 46% |
| `train` (actor+critic) | 38 | 6.02 s (`pool_seconds`) | 3.69 s | 2.33 s (`seconds`) | 61% |

Critic warmup is update-heavy; the real actor phase is rollout-heavy at 61%.
The 3.69 s rollout is 1024 rows x ~355 actions / 3.69 s = ~99k tok/s, i.e. at
the measured width ceiling already, so there is no width change left to make.
A 40,000-step run is therefore ~67 h, and roughly 60% of that is generation
running at its throughput limit rather than at a fixable inefficiency.

Note the early `train` steps carry compile cost and are excluded above (first
5 dropped); including them inflates the median pool time.

## 2026-09-22 — SFT step 3.5x faster; v10 mixture; the latent RL mode is not what the docs say

### Throughput

The first 5,120-token SFT attempt (job 9132, `kda8_sft_ud2605_5k_v2_e1`) ran
at ~380 ms/step and ~108k tok/s. That is a quarter of pretraining's ~454k
tok/s on this card, so it was cancelled at step 1,161 of 22,099. Its
`metrics.jsonl` stays as the eager reference loss curve (`CANCELLED.md`).

Measured with `scripts/benchmark_sft_step.py`. It drives the trainer's own
`SupervisedCE`, `accumulate_step_gradients` and optimizers on real packed
rows of `sft_mix_ud2605_omi2_drills_v2.parquet`: 40,960 tokens per step, 10
warmup steps discarded, 40 timed. Final losses agree across configurations
to 2.2587-2.2595.

| configuration                                     | ms/step | tok/s | peak alloc |
| ------------------------------------------------- | ------- | ----- | ---------- |
| old trainer (job 9132)                            | ~380    | ~108k | -          |
| gathered readout, eager, 2x4 micro-batches        | 330     | 124k  | 15.8 GiB   |
| compiled blocks, old whole-`forward` KDA fence    | 150     | 273k  | 3.8 GiB    |
| compiled blocks, fence on the FLA kernel only     | 130     | 314k  | 3.3 GiB    |
| same, 4x2                                         | 124     | 331k  | 6.3 GiB    |
| same, 8x1 (job 9155; production)                  | 110     | 374k  | 9.4 GiB    |

What each row changed:

- **Readout.** The vocab head renders only supervised positions, gathered
  by `index_select` on host-computed indices. It no longer renders
  full-sequence logits masked after the fact. This drops the ~1/3 of packed
  slots that carry no loss off the largest matmul, and removes the
  boolean-mask sync and the dynamic shape.
- **Compilation.** Each residual block compiles on its own, as pretraining
  does. Readout, softcap and CE compile as one dynamic-shape region.
- **KDA fence.** `torch.compiler.disable` used to cover all of
  `KimiDeltaAttention.forward`, so its projections, short convolutions and
  gated norm ran eagerly. It now covers only `_delta_rule`, the `chunk_kda`
  call. That is worth 13% on its own.
- **Micro-batch split.** The loss is normalised by the step-wide supervised
  count, so the split is a pure memory knob. It is exact in fp32 (1e-7);
  under bf16 the difference is ~2e-3 relative, from rounding. One micro-batch
  is fastest.
- **Host side.** Uploads are pinned and non-blocking. The loss is logged one
  step late through a pinned async copy plus event, because `.item()` is
  stream-ordered and would drain the queue.
- **Tokenisation.** Encoding is batched through the HF fast tokenizer. It is
  identical to per-document encoding on 40k real rows. Tokenising the
  corpus takes ~3 min, where per-document preparation took ~13 min in total.

Profile at 8x1 (job 9155): host/device wall ratio 1.00, so the GPU is
saturated. Time splits as matmuls ~45% (the head GEMM runs at ~205 TFLOPS),
FLA KDA ~25%, flash attention ~8%, CE ~7%, and the eager depthwise short
conv ~5%. The conv is what is left. Fusing it changes shared numerics, so it
is deferred.

FLA's `disable_recompute=True` bought nothing here: 130.4 vs 130.5 ms at
2x4, and 115.8 ms at 8x1 against 109.5 ms with recompute on. So the knob was
not added; the call keeps `disable_recompute=False`.

Review follow-ups:

- All blocks share `Block.forward`'s code object. A single epoch already
  needs exactly Dynamo's default 8 graphs ({KDA, attention} x {grad, no-grad}
  x {full, tail micro-batch}). `SupervisedCE` now raises the budget to 64
  inside a scoped `torch._dynamo.config.patch`, with
  `fail_on_recompile_limit_hit`, so an unanticipated shape fails the run
  instead of silently running the block eagerly.
- `postraining/tests/test_sft_compiled_step_cuda.py` checks compiled against
  eager per parameter (each < 5e-2 relative), a tail micro-batch, and the
  no-grad holdout path.
- The fence change also means Inductor now traces the KDA projections and
  convolutions for the RL replay/critic compile, bolmo, and the few-shot
  prefill. `kda_gpu_parity.py` re-ran to a new file,
  `postraining/runs/kda_gpu_parity/result_kernel_fence_20260922.json`. The
  script now takes `--output` and refuses to overwrite an existing result.

### v10 RL mixture

`vapo_broad_v9_bare` binds `sft_corpus_sha256` to the job 9040 corpus. The
RL trainer requires that hash to equal the parent checkpoint's
`traces_sha256`, so v9 cannot follow a v2-corpus SFT. `vapo_broad_v10_bare`
is v9 rebuilt with `--sft-corpus
postraining/data/sft_mix_ud2605_omi2_drills_v2.parquet`. The rows are the
same; only the corpus binding differs. v9 is unchanged.

### Latent mode discrepancy

CLAUDE.md, the README and the launch script describe `latent` as
deterministic hidden carry. That was v28 (ac2d67c). v29 (0969a9d) replaced
it in `train_latent_vapo` with a stochastic policy: a forced Gaussian THINK
action plus a Bernoulli stop gate. Deterministic carry now exists only on
the MiniCPM path (`train_minicpm_vapo --token-carry`). v29 also needs
`stream >= continuation + 1`, so at the checkpoint's 1,024-token RL context,
256 prompt + 768 continuation does not fit. A KDA latent run needs one of
two things: v28 carry restored as a versioned mode, or v29 with a shorter
continuation or a longer RL context.

### Stage 1 result: `kda8_sft_ud2605_5k_v2_e1_r2` (job 9182)

- **Training:** 22,099 steps in 44.8 min (121.6 ms/step including 222 holdout passes), at ~517 W and 99% utilisation.
- **Parity with the cancelled eager run:** holdout CE matches it to four decimals at every shared eval through step 1,100. The one exception is step 700, where eager reads 1.505 and the compiled run 1.459; the two agree again from step 800.
- **Final holdout completion CE:** 0.9446. This is on this corpus's own holdout, so it is not comparable to job 9040's 0.466.

Sampling gate on its own panel. 128 prompts x 8 samples, 448 prompt + 768 response tokens, T=1.

| metric                  | this run | job 9040 (own panel) |
| ----------------------- | -------- | -------------------- |
| accuracy                | 0.086    | 0.262                |
| mixed-prompt fraction   | 0.219    | 0.398                |
| prompts all wrong       | 100/128  | 70/128               |
| structural format       | 0.453    | 0.849                |
| ended within 768 tokens | 0.458    | 0.859                |
| emitted tokens (mean)   | 539      | 344                  |

The panels differ, and they cannot be crossed. The new corpus contains OMI2, so job 9040's held-out problems may sit in its training split, and `sft_gate_eval` refuses cross-corpus panels for exactly that reason. The gate therefore does not say this checkpoint is worse. What it does show is behavioural: over half the samples never close their answer within 768 tokens.

The 35 unterminated transcripts in the capture are not repetition loops (median zlib ratio 0.43, against 0.56 for terminated ones). They are long, unfocused imitations of the UltraData/Nemotron think style ("Actually... Let's check..."), cut off by the budget. That is the likely cost of training on traces up to 5,120 tokens and then gating and RL-ing at 768.

The matched comparisons are:

- the arithmetic probe (`data/math_drills/v4/probe.jsonl`, same settings as job 9040's `arith_probe_kda8_sft_e1`);
- `train_latent_vapo --bench-only` on the DeepMind interpolate panel. It runs only that panel; AIME is not in it. Both corpora are decontaminated against the panel.

Read those before choosing a stage 2 base.

### `--reasoning-mode carry`: deterministic hidden carry restored

`carry` is the v28 contract under new schemas: execution
`..._deterministic_hidden_carry_token_only_generator_token_rng/v30`, replay
`compact_token_logprob_next_slot_hidden_carry_exact_replay/v5`, rollout policy
`deterministic_hidden_carry/v2`, input `zero_init_hidden_residual_prenorm_mlp/v2`.
The `cot`, `latent` and `none` schema bytes are unchanged, and a test pins
them. The belief that produced token t is stored detached at t's slot and fed
through the zero-init combiner when t is consumed. Replay reads the stored
beliefs, never recomputed ones. The critic has its own combiner. Carry is
lockstep-only and refuses the thought-sigma and stop-probability flags. At
zero init its rollout is bitwise the `cot` rollout.

Verification:

- **CPU tests:** `test_hidden_carry.py` (26 tests) plus the affected suites give 414 passed. Covered: nonzero-combiner rollout vs replay on nano and KDA, where zeroing the carry moves log-probs by more than 100x the tolerance; left padding, split/pack, compaction and the static tail; per-row isolation for actor and critic; resume refusal of v1 and cross-mode checkpoints.
- **Red-team (independent subagent):** no defect found in alignment, replay, gradient flow, resume or mode plumbing. Its one real gap was that nothing ran a whole carry rollout through the compiled decode step, which is the production path. Parity section 7 now does, with and without the CUDA-graph tail.
- **GPU parity, job 9192, sections 1-6:** carry rollout-vs-replay log-prob error 0.0080, beside cot's 0.0077 on the same trunk; stored carry vs replayed belief 0.025; compiled vs eager carry step 0.015 (logits) and 0.017 (belief). Section 7 is job 9208.
- **Section 5 fails** (Stable LatentMoE, not used by this lineage; the SFT checkpoint has no expert weights). `moe_bf16_decode_vs_dense` reads 0.540 against a 0.5 bound, and `moe_bf16_compiled_paged_vs_eager` reads 0.459 against 0.02. No recorded run of section 5 has passed: the only earlier result (July) predates it, and HEAD's script died in section 2 before reaching it. The kernel-fence change does not touch this: section 5 runs eager `forward` and compiles only `paged_step_core`, which uses `step`. A logit gap of that size is consistent with top-2 routing flips under bf16 rounding. That diagnosis is not verified, and the bound was not loosened.

The launch script's `rl-carry` stage is `rl` with `--reasoning-mode carry`
and its own run name: same SFT checkpoint, mixture, arguments and seed.

### SFT exact resume, end to end on GPU (job 9194)

The same 766-step run was trained straight through, and separately SIGTERMed
at step 38 and resumed.

- **The resume works:** the first leg exited 75 with the step-38 checkpoint, and the metric streams have identical keys (784 entries each).
- **It is not bitwise identical on GPU, and resume is not the cause:** the two runs already differ at step 2 (about 1e-4 train loss), 36 steps before the resume point, so the compiled GPU step is nondeterministic from run to run. The gap around step 38 stays at that noise level. By step 766 chaos has amplified it to a 0.004 difference in holdout CE.
- **Division of evidence:** bitwise exactness is established by the CPU test (`test_resume_state_continues_training_exactly`, which fails with the optimizer-state restore removed); the GPU run establishes the SIGTERM/exit-75/resume plumbing.
- **Practical noise floor:** about 0.004 holdout CE separates identical SFT runs at this scale.

### Matched comparison: the 5120-token stage 1 is worse; stage 2 stays on job 9040

Same panels, settings and seed for both checkpoints.

| metric                                  | job 9040 | `_r2` (job 9182) |
| --------------------------------------- | -------- | ---------------- |
| arithmetic probe, 1,920 items (T=0)     | 0.516    | 0.368            |
| probe structural format                 | 0.920    | 0.931            |
| probe unterminated at 512 tokens        | 152      | 132              |
| DeepMind interpolate avg@8 (job 9203/9204) | 0.018 | 0.005            |
| interpolate mixed-prompt fraction       | 0.083    | 0.042            |
| interpolate mean emitted tokens (of 768) | 375     | 550              |

- **Probe:** the gap is about 10 standard errors. `_r2` is lower on 13 of 15 families and ties the other two at 0.000 (`mul_decimal`, `percent_change`). The biggest losses are `sub_decimal` 0.648 -> 0.133 and `sub_integer` 0.859 -> 0.398. Format and termination are unchanged, so this is lost arithmetic, not truncation.
- **Interpolate:** both checkpoints sit below the modal-answer baseline (0.021), so this panel only supports the direction.
- **Diagnosis, measured from the corpora:** drills fell from 83M tokens (28.2% of 293M) to 50M (6.8% of 738M). Two thirds of the new corpus is UltraData think traces averaging 1.8k-2.5k tokens (OMI2 subset 45.9%, Nemotron-Cascade 19.9%), so drill steps are both fewer and a smaller share of each step. The longer, unfocused responses on both panels are what imitating those traces looks like inside a 768-token budget.
- **Decision:** stage 2 (cot, then carry) starts from `kda8_sft_omi2_drills_e1` with the v9 mixture. That checkpoint was trained in the same 1,024-token window RL uses, and the v9 hash binding was exercised by job 9203. The `_r2` checkpoint is kept as the context-extension negative result. A later 5,120-token attempt would need drill share restored and a response budget that matches the traces it imitates.

### Stage 2 cot launched: `kda8_vapo_cot_v9` (job 9223)

Launched with `scripts/launch_kda8_posttrain.sh rl`: job 9040 base, v9
mixture, 64 prompts x 16 samples, 768 response tokens.

- **Matched setup:** the step-0 bench, 0.0182 on DeepMind interpolate, reproduces job 9203 exactly.
- **Step time:** 9.1 s (about 6.5 s collect plus 2.3 s update). 40,000 steps is about 101 h, not the 67 h estimated earlier.
- **The learnability gate was missing:** the "39.84% mixed prompts" the launch comment cites is job 9040's SFT gate on OMI2 holdout problems, not a frozen-policy gate on this mixture. The first 18 pools stand in for that gate:

| source         | groups/pool | reward | within-group std | ended | mean actions |
| -------------- | ----------- | ------ | ---------------- | ----- | ------------ |
| deepmind_easy  | 48          | 0.021  | 0.055            | 0.93  | 264          |
| dapo           | 8           | 0.010  | 0.032            | 0.74  | 540          |
| ultradata_math | 8           | 0.008  | 0.025            | 0.63  | 590          |

- **Reading the table:** a 1-of-16 group has std 0.24, so roughly a fifth of groups carry advantage signal. That is sparse but not absent. dapo and ultradata_math also run out of budget on a quarter to a third of samples.
- **Cull rule, fixed before looking:** at step 1,000, and again at 2,000, compare held-out interpolate policy accuracy and deepmind_easy pool reward against step 0. Stop the run if neither has risen beyond noise, rather than spend four days of the card.
- **Carry arm:** `rl-carry` follows only if cot shows learning. A carry-vs-cot comparison on a policy that is not learning measures nothing.


### Parity section 7: carry through the compiled decode step (job 9208 retry)

The unmasked tensor-position `torch.narrow` path cannot trace fullgraph
(`Could not guard on data-dependent expression Eq(u0, 0)`), so section 7 feeds
left-padded ragged prompts with `prompt_lengths`, as `upload_chunk` does.
Only the sequential `--rollout-groups 0` branch reaches the narrow path; that
is a latent problem for non-default configs, not for production.
`result_compiled_carry_rollout_20260922_v2.json`: compiled rollout vs replay
log-prob error 0.0275 (cot) and 0.0342 (carry) against a 0.5 bound; carry
stored hidden vs replayed belief 0.114 against 0.25. The CUDA-graph-tail
variant gives identical maxima.

### `kda8_vapo_cot_v9` collapsed because its critic never left its prior; cancelled at ~step 274

- **Symptoms by step 250:** mean think tokens fell from 276 to 58 and the think-format fraction from 0.84 to 0.67. Gate-zeroed correct samples rose from 0 to 1.2%. Bench answers degenerated toward `\boxed{0}`, and mean emitted tokens on interpolate fell from 375 to 64. Bench 0.0182 -> 0.0252 is inside noise at 64 tokens and is not learning evidence. AIME stayed at 0 -> 0. Pool reward went from about 2.0% to 2.2%.
- **Root cause:** the HL-Gauss head bias was initialized to the log-probs of a point prior at 0 with a 1e-6 floor, so every bin outside roughly [-0.02, 0.02] started at logit -13.8. AdamW at 2e-5 moves a bias by about 2e-5 per step.
  - The decoded value mean was 6e-5, constant over all 50 warmup steps and 270 RL steps. Explained variance was 0.000, and excess CE was 0.1-0.2 nats throughout.
  - The critic parameters did train, but slowly: all 142 moved, by at most 0.008. Head bias went from -0.961 to -0.966, and head weight RMS from 8e-4 to 5e-3.
- **Mechanism:** with V ≈ 0 the advantage is ≈ reward, so the update only reinforces correct samples. The positive contribution was about 2.5e-3 against about 4e-5 negative. Correct samples are short, so the policy learned their length rather than their correctness.
- **Kept:** the run directory and its checkpoints, as the negative result.

### Scalar critic replaces HL-Gauss (`CRITIC_SCHEMA scalar_zero_init_linear_unclipped_mse/v1`)

- **Head and loss:** `SeparateCritic.head` is `Linear(model_dim, 1)` with zero weight and zero bias, read in fp32. It trains by unclipped, token-weighted squared error against the lambda-one (Monte Carlo) return; the actor's advantages still use length-adaptive lambda.
  - Under Adam a zero head's output moves by about lr x ||belief||_1 per step through its weight. That is about 0.008 per step at 2e-5 for 512 post-norm features with a consistent-sign component, so the ~2% marginal is within a few updates rather than hundreds. That lr x ||belief||_1 figure is an upper bound (it assumes the mean belief is sign-coherent across tokens). Measured on CPU with a whole 512-wide fresh critic at 2e-5 AdamW against a natural 2.5% Bernoulli marginal: 0.0029 after one step, 0.0196 after 5 and 0.0294 after 8, then settling to 0.0228 by step 50 (`test_scalar_head_fits_the_success_marginal_at_the_production_rate`). That settles the marginal only. A from-scratch trunk at 2e-5 moves its embeddings about one init std only after about 1,000 steps, so state-dependent value comes from a linear probe of near-random features. Warmup explained variance on the real pool is the measurement that decides whether the critic rate has to rise.
- **Removed:** `hl_gauss.py`, its test, `scripts/compare_hl_gauss_cpu.py`, and the five `--value-*` support flags.
- **Resume:** refuses any checkpoint without the critic schema, so categorical critics are never migrated.
- **Dashboards:** `value/token_weighted_mse` replaces `value/token_weighted_excess_ce`.
- **Audited alongside:** there is no gradient-norm clip and no value clip anywhere on this path. `build_optimizers` now asserts that actor and critic share no Parameter and no storage; the critic trunk is a fresh random instance from `fresh_trunk`.
- **If this does not fix it:** the next suspects are the critic learning rate (default = actor 2e-5, AdamW for head, combiner and embeddings, Muon for trunk matrices) and the fresh trunk's init. The decision gate is warmup explained variance and prediction mean against the target mean.

### Training-rollout transcripts and `scripts/` cleanup

- **Transcripts:** `postraining/rollout_report.py` replaces `scripts/render_minicpm_responses.py`. It renders both MiniCPM TensorBoard text samples and the KDA trainer's own captures (`<run>/rollout_samples/step_XXXXXX.json`, `training_rollout_samples/v1`, re-rendered into `<run>/rollout_samples.html`). Every `--rollout-sample-every` steps (default 25, step 0 included) the trainer keeps one correct and one incorrect trajectory per source.
  - **Picks:** a hash of (step, prompt, row), not first match. The collector scores groups shortest prompt first, so first-match picks showed only the easiest problems.
  - **Verdicts:** `score_math_rollout` and `score_python_rollout` now return a `RowVerdict` per row, and the transcript stores that verdict and the verifier's own prediction rather than re-parsing, so it shows exactly what was graded.
  - **Steps:** each capture also records `metrics_step` (= step + pool updates), the step at which that pool's rollout metrics are logged.
  - **Resume:** captures at or after the resume step are purged.
- **`scripts/` cleanup:** no directory restructure, because manifests hash script paths and tests and package code import `scripts.<name>`. `scripts/README.md` is now a domain index.
  - **Deleted with `git rm`, as superseded or dead:** `compare_hl_gauss_cpu`, `render_minicpm_responses` (and its test), `rbf_bench`, `benchmark_components`, `run_kda18_dense6_ablation`, `run_optimal_gdn2_ablation`, `benchmark_qwen35_sglang` and `benchmark_qwen35_vllm`.
  - **Path fixes:** ten MiniCPM scripts and two tests still referenced the pre-move `postraining/minicpm_vapo.py`; they now use `postraining/vapo/policy.py`. Old controller-geometry artifacts recorded under the old path key may be refused.
- **Relaunch:** `kda8_vapo_cot_v9_scalar` (job 9240) started after the scalar-critic CUDA tests passed (job 9239, 274 tests).
- **Cancelled and resumed (2026-09-22):** job 9240 was cancelled at RL step ~10 on warmup explained variance ~0 (prediction mean 0.0097 against target 0.0176 at warmup step 50; EV -0.005 to +0.001 through RL step 9). That was premature: think tokens (263-287), think-format (0.81-0.84) and emitted length (~770) showed no collapse, and a from-scratch trunk is expected to need many steps before it resolves state-dependent value on a ~2% reward. Job 9245 resumes it from the post-warmup step-0 checkpoint with identical arguments and the default critic rate.
- **VAPO on critic init (`papers/vapo_2504.05118v3.pdf`, sections 4.1 and 5.1):** the critic is never random. It is initialized from a reward model (a full pretrained LM), then value-pretrained on a fixed policy (pi_sft) with Monte Carlo returns "until key training metrics, including value loss and explained variance, attain sufficiently low values", saved, and loaded for RL; the reported config is a 50-step warmup. Critic lr is 2x the actor's (2e-6 vs 1e-6). Without value pretraining VAPO collapses like vanilla PPO (shorter responses, answering without reasoning; AIME24 11 vs 60). The init bias VC-PPO identified comes from the reward model's EOS-scoring objective, which a zero-init head on a copied trunk would avoid. Our 50-step warmup is fixed-count rather than EV-gated, and our trunk is random, so both depart from the paper; the paper says nothing about embedding-only init.
- **`--critic-init {scratch,actor}` (2026-09-22):** `actor` starts the critic trunk from `model_io.copy_trunk`, a storage-independent copy of the actor as loaded at run start (base checkpoint or `--actor-init`/`--curriculum-init` weights), behind the same zero head. Default stays `scratch`, so job 9245 is unchanged. The manifest's `critic.init` records `scratch`, `actor_copy` or `warm_checkpoint`; a resume now carries the recorded block forward from the resumed run's manifest, in place or from the checkpoint's own directory when resuming into a new one (previously a resume of an `--actor-critic-init` run relabelled it `scratch`), and refuses a contradicting `--critic-init`. Refused with `--actor-critic-init` and for LatentMoE actors. Not yet run.

### `kda8_vapo_cot_v9_scalar` degradation: DG turns critic noise into an entropy bonus (2026-09-22, job 9245, step ~500)

- **Trajectory (50-step means):**
  - Think tokens fell from 263 to 74 by step 350; the length collapse recurred despite the scalar critic.
  - Ended rose from 0.875 to 0.998, then fell to 0.975. Think-format peaked at 0.915 (step 76) and fell to 0.834.
  - Actions per trajectory are rising again, while think length stays flat.
  - ultradata_math accuracy peaked at 0.045 (step 300) and fell to 0.027.
  - Bench at step 250: 0.0174, below the dataset modal-answer baseline of 0.0208, at 101 mean emitted tokens (382 at step 0).
- **Transcripts:**
  - Correct samples are mostly lucky small-integer guesses (0, 1) after incoherent think text.
  - The unterminated rows are 768-token word salad: mixed scripts, mojibake, random numerals.
- **Entropy:** DG surprisal mean (mean -log pi of the sampled tokens, a sampled-entropy proxy) went 0.88 (step 150) -> 1.01 (300) -> 1.56 (400) -> 2.55 (500), and is accelerating.
- **The DG gate is inert as designed:**
  - Gate means are 0.495-0.512 throughout, at eta = 1 on raw GAE advantages (std 0.03-0.06).
  - 24-38% of policy tokens carry positive advantage, while only 2-3.5% of trajectories are correct. With explained variance ~0, per-token advantage signs come from the critic's V_{t+1} - V_t differences, which is noise.
- **Mechanism:** a token with surprisal l and zero-mean advantage noise +-u gets expected DG coefficient
  - 1/2 [sigmoid(u l) u - sigmoid(-u l) u], which is approximately u^2 l / 4 for small u l, and positive.
  - E_a[l(a) grad log pi(a)] = grad H exactly. So DG over zero-mean advantage noise is, in expectation, entropy ascent with strength about Var(u) / (4 eta).
  - At the observed advantage std, that is about 1e-3, a typical LLM entropy-bonus coefficient, applied with no counterweight.
  - Plain PG has no such term: it is linear in U, so zero-mean noise averages out.
  - The DG paper's token-reversal runs used a sequence-level empirical-mean baseline, with no per-token critic noise. None of Osband's three DG papers (2603.14608, 2603.20521, 2603.20526) has LLM experiments.
  - TPO (2604.06159, Appendix E) notes that DG has no trust region, and shows DG's cross-context coefficient vanishing as beta ~ p_n on hard contexts.
- **Implication:** with an EV ~0 critic, DG's token-level gating needs either a trustworthy advantage or a sequence-level one. For an all-fail group, a group-mean baseline gives exactly zero advantage and so no noise.
- **Superseded:** the step-683 measurement below refutes the sequence-level fix; the entropy push is intrinsic to DG, not specific to critic noise.

### DG entropy diagnostic at step 683: the group-baseline fix is refuted (2026-09-22, job 9246)

- **Tool:** `postraining/diagnose_dg_entropy.py` (CPU tests in `postraining/tests/test_diagnose_dg_entropy.py`).
  - It collects one mixture pool (48/8/8 prompts x 16 samples) from the checkpoint through the training rollout, `score_math_rollout`, `refresh_old_statistics` and length-adaptive GAE.
  - It reports two first-order entropy-change proxies per emitted token, each proven exact for a tabular softmax by the tests:
    - natural, c (l - H);
    - softmax, c [pi_a (l - H) + sum_k pi_k^2 (log pi_k + H)].
  - The coefficients c are DG and 0.5 * PG, each on critic GAE and on a group baseline (R - group mean R). CIs are a 95% bootstrap over prompt groups.
  - Output: `postraining/runs/kda8_vapo_cot_v9_scalar/dg_entropy_diagnostic/step_000683.json`.
- **Pool:** 1,024 trajectories, 2.44% correct, 132k emitted tokens.
  - Mean surprisal 3.786 and mean entropy 3.796: on-policy consistency holds, and entropy is up from a surprisal of 2.55 at step 500.
  - Failed-token critic advantage: mean -0.0040, std 0.0105, 29.2% positive.
- **Per emitted token, natural proxy [95% CI]:**
  - dg_critic: failed +1.06e-4 [+5.2e-5, +1.6e-4]; total -3.5e-6 [-1.5e-4, +1.1e-4].
  - pg_critic: failed -2.4e-5 [-8.0e-5, +3.2e-5]; total -3.59e-4 [-6.7e-4, -1.2e-4].
  - dg_group: failed +3.23e-3 [+5.4e-4, +7.4e-3]; total +4.11e-3 [+6.0e-4, +9.0e-3].
  - pg_group: total -6.5e-4 [-1.4e-3, -7.9e-5].
  - The softmax proxy agrees in sign: dg_critic +2.1e-5, pg_critic -2.2e-5, dg_group +5.1e-4, pg_group -4.6e-5 (totals).
- **What this shows:**
  - The noise term exists as predicted. DG on failed-trajectory critic noise pushes entropy up, and the matched 0.5 * PG control does not.
  - The general cause: DG's coefficient c(l) = U sigmoid(U l / eta) has dc/dl = U^2 sigmoid'(U l / eta) / eta >= 0 for either sign of U. It always weights surprising tokens more than PG does: more credit on success, less blame on failure.
  - Relative to 0.5 * PG, that is an entropy bonus of about E[U^2 Var_pi(l | s)] / (4 eta) for small U l, with per-state varentropy, whether U is signal or noise.
  - At first order, DG does not create a rise here. Its excess over 0.5 * PG (natural +3.55e-4) cancels PG's sharpening (-3.59e-4), leaving a dg_critic total of about 0, and the two proxies disagree on its sign.
    - PG's sharpening sits mostly in the correct cell, which rests on a handful of groups at 2.44% correct.
    - So DG removes the sharpening, and whatever drives the observed rise lies outside these local proxies: shared-parameter spillover, optimizer geometry, state drift.
  - **Group baseline:** it is not a fix at any eta.
    - Scaling U and eta together scales c and 0.5 * PG alike, so the fair comparison is at matched gate strength |U| / eta.
    - At eta = 1, the group-baseline DG excess is 7.3x (natural) and 12x (softmax) PG's sharpening, against about 1.0x and 1.9x for critic GAE.
    - At eta ~ 6, which matches |U| / eta to critic GAE (group |U| is about 6x larger), the linearised ratio is about 1.2x, no better than critic GAE.
    - The sequence-level fix therefore moves none of this mechanism, and **it was not implemented.**
  - These are sums per group, so an eta sweep needs per-token tensors, which the v1 output does not keep.
  - The render-versus-replay surprisal mismatch was 0.395 at its maximum, and only the maximum was recorded (v2 records signed mean and RMS). Near-zero totals like dg_critic's are within that uncertainty; the large group effects are not.
- **Open, and not measured here:**
  - Under Muon, whose update size is set by the orthogonalization and not by gradient magnitude, a pool gradient that is mostly noise (EV ~ 0, 2.4% correct) still becomes a full-size step.
  - That weight-space random walk may drive the entropy rise more than the first-order bias does, which is about 0 in total for dg_critic here.
  - The discriminating measurement is the cosine between actor gradients from two independent pools at the same checkpoint.
- **Action:** the run was resumed unchanged from step 683 as job 9247.
  - The alternatives for a follow-up are:
    - the clipped VAPO actor objective (`--no-delightful-policy-gradient`), whose PG control is entropy-decreasing here;
    - a gradient-SNR measurement before choosing.


### DG under sequence rewards: the reference code, the author's own failure mode, and a simulation (2026-09-22)

- **Question:** for the restart of `kda8_vapo_cot_v9_scalar` from job 9040 with `--critic-init actor`, keep DG on critic GAE (option 1), switch DG to the author's group baseline (option 2), or drop DG.
- **Reference code** (google-deepmind/egg, `egg/losses/dg.py`):
  - The default is `use_grouped_baseline=True`: the advantage is R minus the group mean, broadcast to every answer token.
  - The gate is sigmoid((chi - lambda)/eta) on the stop-grad current-policy surprisal, and the loss is token-mean.
  - Its DPG task is token reversal over vocab 2, with a **sequence-scalar** reward (fraction correct, optionally to the first error). Every token is causal. The paper uses eta = 1.
- **The author's own failure mode** (Osband, 2603.20526 section 4.2, Prop 3):
  - When an action has sigma/Delta >> 1, lucky draws look like breakthroughs, and delight amplifies them. DG collapses sharply where PG degrades gracefully.
  - "No per-sample statistic computed from (R, pi) can distinguish a genuine breakthrough from a lucky draw."
  - A sequence reward broadcast to tokens puts every weakly causal token in exactly this regime. For a non-causal token in a group with success rate p, PG's expected coefficient is 0, while DG's is p(1-p)[sigmoid((1-p) l) - sigmoid(-p l)] > 0 (checked by Monte Carlo). At high p, DG's negative gate also shuts off the blame for junk tokens that cause the remaining failures.
- **Simulation:** `postraining/sim_dg_sequence_credit.py`, CPU only (it hides the GPU; Adam's graph-capture health check otherwise opens a CUDA context).
  - Setup: 64 prompts x 16 samples, Adam, 1500 steps, 3 seeds. Outputs are in the session scratchpad and not kept.
  - Tabular policy: DG never beat PG on any task, including egg's.
  - A red-team review found no bugs, but pointed out that a tabular policy cannot show DG's claimed shared-budget benefit. So a **residual MLP** policy was added, with the same initial distribution and every context sharing parameters.
  - With the MLP, the paper's claim reproduces. Final accuracy, lr 1e-3, eta 1:

    | task | pg_group | dg_group | pg_gae | dg_gae |
    |---|---|---|---|---|
    | sequential (egg, first error) | 0.717 | **0.877** | 0.671 | **0.889** |
    | graded (egg) | 0.962 | 0.983 | 0.963 | 0.993 |
    | lottery (filler non-causal) | 0.986 | 0.970 | 0.969 | 0.984 |
    | chain_rare (1 rare causal token) | 0.979 | 0.974 | 0.931 | 0.864 |
    | chain (3 causal tokens) | 0.942 | 0.885 | 0.882 | 0.708 |
    | coherence (junk mildly harmful) | **0.980** | 0.378 | **0.965** | 0.562 |
    | chain_coherence | **0.955** | 0.336 | **0.866** | 0.422 |

  - DG wins where PG collapses prematurely onto wrong tokens and every token is causal. Filler entropy is 0.05 for both there, but PG locks in errors.
  - DG loses catastrophically once junk tokens carry any small cost. Filler entropy goes from 1.0 to 2.4-2.9 and coherent mass falls from 0.96 to 0.41. This is the shape seen in the real runs (v9 surprisal 0.88 -> 3.8; v5 9.2).
  - **Learning rate does not change the picture** (coherence / sequential final accuracy):
    - lr 3e-4: DG group 0.53 vs PG 0.97 / 0.85 vs 0.69.
    - lr 3e-3: 0.33 vs 0.99 / 0.87 vs 0.71.
    - Tabular Adam at twice DG's lr made DG worse. Tabular SGD tuned per estimator also kept PG ahead (0.965 vs 0.83-0.89).
  - **Raising eta trades the benefit away** rather than finding a sweet spot (MLP, lr 1e-3):

    | eta | coherence DG group / gae (PG 0.980 / 0.965) | sequential DG group / gae (PG 0.717 / 0.671) |
    |---|---|---|
    | 1 | 0.378 / 0.562 | 0.877 / 0.889 |
    | 4 | 0.590 / 0.907 | 0.767 / 0.787 |
    | 16 | 0.959 / 0.962 | 0.737 / 0.722 |

  - **Unstable lr (1e-2, MLP):** PG itself collapses onto junk (coherence 0.166, filler entropy 0.06), and only DG at eta 16 survives (0.986). DG at large eta is a robustness hedge against an over-aggressive step, not a credit-assignment improvement.
- **Option 1 vs option 2:**
  - Critic-GAE DG beat group-baseline DG on every junk-token task, in both parameterizations and at every eta. This agrees with the step-683 diagnostic (group excess 7-12x PG sharpening vs 1-2x).
  - The mechanism is lambda decay (0.95^k): it shrinks |U| on early tokens and so acts as a larger, position-dependent eta. The real trainer has the same decay (lambda 0.95-0.974).
  - Option 1's own cost, zero-mean critic noise becoming positive entropy drift of about l sigma^2 / (4 eta), grew only slowly with value noise in the toy (tabular 0.752 -> 0.690 at noise 0.01 -> 0.3).
  - **Option 2 is dominated. Neither DG option is recommended at eta = 1.**
- **Superseded:** the entry below traces the junk-token failure to non-token-specific advantages, not to DG, and the restart keeps DG with a lambda-0 actor.
- **Verdict (confidence about 75%; the toy is not the LM):**
  - Restart with the clipped VAPO actor (`--no-delightful-policy-gradient`) plus `--critic-init actor`.
  - PG's known risk in this toy is premature collapse onto wrong tokens, so watch sampled-token entropy/surprisal for a collapse, not a rise.
  - If DG is kept, use option 1, not option 2.
  - No matched DG vs VAPO comparison in this lineage justified making DG the default (commit 0288aed).

### DG needs token-specific advantages: oracle critic, lambda 0, and the `kda8_vapo_cot_v10_dg_td0` restart (2026-09-22, jobs 9268, 9277)

- **Question:** the verdict above ("neither DG option at eta = 1") rested on broadcast group advantages and long-lambda GAE. Is the junk-token failure intrinsic to DG, or to the advantage it is given?
- **Audit:** a subagent compared the trainer's DG loss with egg's `dg.py` line by line (gate, surprisal, slots, mask, denominator, on-policy, no extra actor terms). They match. There is no DG code bug.
- **Mechanism:** DG's coefficient U sigmoid(U l / eta) is nonlinear in U. Luck that is not caused by the token (independent of it) enters U as zero-mean spread, and DG turns that into a bias of about l Var(U) / 4 toward rare tokens, i.e. entropy ascent. PG is linear in U, so the same luck is only variance. That is why the luck hurts DG and not PG.
  - The papers' DG has no critic. egg broadcasts a group-mean baseline, but its tasks are vocab 2 with every token causal, so there is no non-causal luck to amplify. Our per-token GAE critic is inherited from VAPO.
  - Under an exact critic, a lambda-0 (TD(0)) advantage is exactly zero for a non-causal token. Later TD errors have zero conditional mean (martingale), so the "leak" from lambda > 0 carries information only to the extent the critic is wrong. That is the lambda bias/variance trade.
- **Simulation** (`postraining/sim_dg_sequence_credit.py`, now with `{pg,dg}_oracle` estimators on exact E[R | prefix]; MLP policy, Adam lr 1e-3, eta 1, 3 seeds, final accuracy / filler entropy):

  | oracle critic | coherence PG | coherence DG | sequential PG | sequential DG |
  |---|---|---|---|---|
  | exact, lambda 0 | 0.999 / 0.11 | **0.996 / 0.09** | 0.699 | **0.837** |
  | exact, lambda 0.5 | 0.999 | 0.995 | 0.697 | 0.817 |
  | exact, lambda 0.8 | 0.998 | 0.983 / 0.43 | 0.704 | 0.830 |
  | exact, lambda 0.95 | 0.993 | **0.709 / 2.08** | 0.693 | 0.829 |
  | shrink 0.1 + noise 0.01, lambda 0 | 0.983 | 0.965 / 0.90 | 0.679 | 0.733 |
  | shrink 0.3 + noise 0.01, lambda 0 | 0.994 | 0.985 | 0.705 | 0.760 |
  | shrink 0.3 + noise 0.01, lambda 0.8 | 0.989 | 0.929 | 0.707 | 0.823 |

  - With token-specific advantages, DG matches PG on the junk-token tasks and keeps its sequential win (chain_coherence at lambda 0: 0.991 vs 0.998; lottery 0.997 vs 0.995).
  - The same exact values at lambda 0.95, the trainer's length-adaptive range, reproduce the entropy blow-up. Lambda, not DG, is the failure.
  - DG tolerates iid value noise of 0.01 per state (coherence 0.994). With a partial (shrunk) critic, a longer lambda buys sequential credit at a coherence cost, as the trade predicts.
- **Choosing the actor lambda:** minimize the MSE of A_t(lambda) as an estimate of the token's own effect. With v the per-step luck variance and s^2 the iid critic error variance, the optimum solves lambda / (1 - lambda)^2 = s^2 / v (r 0.1 -> 0.08; r 1 -> 0.38; r 10 -> 0.73).
  - The ratio is not reliably measurable: it moves as the policy and critic change, and critic shrinkage is invisible to the TD-error autocovariance.
  - DG's cost of variance is bias (entropy), while PG's is only noise, so DG wants the low-variance end. The chosen setting is **actor lambda 0**, with critic targets kept as lambda-1 returns. With an EV ~0 critic no lambda helps; the critic has to learn.
  - Historical risk: TD(0) over a bad critic produced the think-fraction ratchet. That is why the run is monitored for calibration (below).
- **Implementation:**
  - `--actor-gae-lambda` (fixed actor lambda; `core.actor_gae_lambdas`), which is resume-exact. Critic targets are unchanged.
  - Telemetry `credit/td0_calibration_slope`: MC advantage regressed on the TD(0) advantage. It is about 1 for a calibrated critic, about 1/2 when critic noise dominates, and above 1 when the critic under-reacts.
  - Telemetry `credit/failed_advantage_surprisal_corr`: advantage vs. sampled-token surprisal on the emit slots of trained failed rows. Positive means the critic is paying a rarity pseudo-reward that DG amplifies.
  - Both are pooled from float64 sums and omitted when undefined.
  - `diagnose_dg_entropy.py` schema v3 adds `--lambdas` and critic EV.
  - A red-team review found no blocking bugs. Its three low findings were fixed: rows zeroed by the source gate are excluded, sums are resolved in float64, and undefined statistics are omitted rather than logged as 0.
- **Critic gate (job 9268):** `--critic-init actor`, 300 online warmup steps, `--steps 0`. EV by 50-step block: 0.0013, 0.0060, 0.0069, 0.0080, 0.0063, 0.0084.
  - The critic learns the mean (token-weighted target mean ~0.013, about 2% of rollouts correct) and little else.
  - For scale: prompt identity explained ~12% of reward variance in-sample in the step-683 pool, inflated by 16-sample noise, so a critic that fully learned problem difficulty would reach roughly 0.05. The warm critic is far below that.
- **Restart (job 9277, `postraining/runs/kda8_vapo_cot_v10_dg_td0`):**
  - It starts from `--actor-critic-init postraining/runs/kda8_critic_gate_actor_w300/critic_warmup_checkpoint.pt` (SFT job-9040 actor, warm critic, cursor 19200) with `--actor-gae-lambda 0`, DG at eta 1, and otherwise the v9_scalar arguments. It replaces `kda8_vapo_cot_v9_scalar`.
  - Watch: EV rising above ~0.01, the calibration slope toward 1, the surprisal correlation near 0, and sampled entropy / surprisal against v9's 0.88 -> 3.8 rise.
  - A weak critic at lambda 0 means little credit for think tokens. The expected failure is slow learning, not entropy growth.
  - The lambda diagnostic on the warm checkpoint (job 9276) is held so it does not delay the run.


## 2026-09-22 — v10 degraded; DG's only difference from PG is a sign-blind rare-token push

Scripts and outputs: session scratchpad `dgref/` (`ref_egg.py`, `port_torch.py`, `port_batched.py`, `sim_think_length.py`; red-team scripts in `dgref/redteam/`). CPU only; no GPU time used.

- **v10 (job 9277) outcome: degraded; cancelled by request at step 491.**

  | steps | think tokens (intact fences) | think-format | ended | accuracy | surprisal | EV |
  |---|---|---|---|---|---|---|
  | 1–50 | 257 | 0.833 | 0.886 | 0.020 | 1.21 | 0.011 |
  | 301–350 | 139 | 0.751 | 0.891 | 0.028 | 0.78 | 0.026 |
  | 401–437 | 123 | 0.599 | 0.860 | 0.018 | 0.84 | 0.011 |

  - Bench: 0.0095 at step 0, 0.0139 at step 250. AIME stayed at 0.
  - Late samples are repetition loops ("-5k-5k-5k…", "Therefore, Δ < 5 or Δ < 5…") and a shared phrase ("Using algebra 5") across unrelated prompts.
  - `think_tokens_mean` counts intact fences only. From step 350 on, the decline is format and termination breakdown rather than early `</think>`.
  - Gate means were 0.4978 (negative advantage) and 0.5022 (positive) throughout. With |A| ~0.01 at eta 1, that is automatic and not diagnostic. Surprisal *fell*, so v10 does not show DG's aggregate entropy fingerprint.
  - v9_scalar (lambda ~0.95) does show it: surprisal 1.0 -> 3.9 while the positive gate mean rose 0.503 -> 0.527. Its think length fell 194 -> 74 *before* surprisal rose.
  - The calibration slope logged by 9277 (~0.72) predates the terminal-slot exclusion fix.
- **Reference check: our DG loss reproduces the paper.** Token reversal (M=2, H=10, reward to first error), 10x10 batch, Adam 1e-4, 1000 steps, median sampled sequence error:

  | friction | egg DG | egg REINFORCE | port DG (prod loss) | port PG |
  |---|---|---|---|---|
  | none (5 seeds) | 0.005 | 0.951 | 0.007 | 0.560 |
  | bugs pE=3e-3 (15 seeds) | 0.010 | 0.747 | 0.023 | 0.675 |
  | corruption pR=0.01 (5 seeds) | 0.006 | 0.671 | 0.008 | 0.522 |

  - egg's REINFORCE fails by committing to deterministic wrong answers (entropy ~0.001).
  - One residual port gap is unexplained: on bugs, 9/15 port DG seeds reach error < 0.05, against 15/15 for egg.
- **Identity:** U·σ(U·s/η) = U/2 + (U/2)·tanh(U·s/(2η)).
  - DG is half of PG plus a term that is even in U: it pushes the sampled token up whatever the sign of the outcome, in proportion to |U| and to the token's surprisal. This is the only way DG differs from PG.
  - It is not baseline-invariant. An error d common to all actions at a state pushes rare tokens up for either sign of d (red-team derivation).
- **That term is DG's benefit on reversal** (port, 3–5 seeds):
  - DG at eta=100 (gate linear) loses the win: none 0.357, bugs 0.987.
  - dgsym, which gates on |U|·s so the coefficient is odd in U, is PG-level on none (0.53–0.75).
  - PG with the importance ratio plus only the sampled U²·s/(2η) term ("taylr", red team) recovers DG's greedy win (bugs and none, greedy error ~0.00–0.01).
  - Reversal scores every answer token causally, so the rows with large |U| are rows whose rare tokens caused it.
- **The same term is the proposed think-length mechanism** (`sim_think_length.py`, a tabular stop logit, early `</think>` rare at s ≈ 7, success q(L) rising to 0.04 at L*=48). Early-stop probability, starting from 0.041:

  | credit | PG | DG | dgsym |
  |---|---|---|---|
  | group baseline (sequence) | 0.019 | 0.070 | 0.032 |
  | oracle TD(0) | 0.003 | 0.004 | 0.004 |
  | oracle + critic noise sd 0.1 | 0.021 | 0.999 (length 48 -> 8) | 0.021 |

  - With exact group accounting (red team; G=16, p=0.03, s=7), DG reinforces a rare action once its success rate exceeds about 0.36p, while PG requires more than p.
  - Once stopping at t is common, stopping at t-1 becomes the new rare lucky action, so the collapse cascades.
  - The red team reproduced the collapse under per-state fixed, per-step, and pure-offset critic errors, so the noise model does not build the result in.
- **Not established:** the production magnitude.
  - Full collapse in the toy needs TD noise about 10x v10's scaled level. The sparse group-credit toy only drifts.
  - There is no matched PG arm: every kda8 cot run had DG on.
  - Pre-DG PG runs 1024/1025 also collapsed length (382 -> 16), but from a base with a taught terse mode. Truncation (11–17% unterminated rows) favours shorter output under PG too.
- **Retracted:**
  - "Decoupled DG" (critic-TD gate x leave-one-out group magnitude) is not a fix. Its expected sign follows Q − V0, not Q − V(h), so it is biased whenever V(h) ≠ V0, and it lost DG's reversal win (none 0.53, bugs 0.998).
  - PG plus a U²-weighted exact-entropy bonus ("pgent") failing on bugs was a confound: it had no importance ratio.
  - Port critic runs at lambda 0/0.8 trained the critic on lambda-matched targets instead of the production lambda-1 targets. They are quarantined in `dgref/out/invalid_critic_lambda_targets/`; the fixed rerun is pending.
- **Next (GPU):**
  1. Step-0 direction test on one pool. Project g_DG, g_PG and g_DG − g_dgsym onto ∇ Σ log π(`</think>` | h) at think positions, bootstrapped over groups. Split the DG − PG part into |delight| > 1 tokens and the rest, and compare E[A | `</think>`] with continue at matched positions.
  2. A matched PG / DG / dgsym trio from the same start for ~300 steps, measuring think length over all rows.
  3. `port_batched.py`, the whole reversal grid in one ensemble process. It is FLOP-bound on CPU (~3 s/step for 165 members).

### Correction: the actor-lambda-0 prediction for v10 was wrong (2026-09-22)

The entry "DG needs token-specific advantages" above concluded "Lambda, not DG, is the failure". It chose actor lambda 0 and predicted that v10's failure would be "slow learning, not entropy growth". **Both claims are refuted.** v10 degraded: think length fell, format and termination broke, it fell into repetition loops, and it produced a shared phrase across unrelated prompts.

v10 telemetry by 50-step block (`kda8_vapo_cot_v10_dg_td0/metrics.jsonl`):

| blocks | EV | td0 calibration slope | advantage std | positive gate mean |
|---|---|---|---|---|
| all 10 blocks | 0.006–0.025, never rose | 0.69–0.74 | 0.011–0.016 | 0.5014–0.5027 |

- **Calibration slope:** 0.5 means critic noise dominates and 1 means calibrated. The failed-advantage/surprisal correlation stayed between −0.03 and 0.02.

Why the prediction failed:

- **It rested on an oracle toy.** The toy's lambda-0 win used exact E[R | prefix]. Its own partial-critic row (shrink 0.1, lambda 0) already showed DG degrading, and the production critic (EV ~0.01) was worse than any toy row. The entry even said "with an EV ~0 critic no lambda helps", then predicted a benign failure anyway.
- **Lambda 0 swaps reward luck for critic error; it does not remove non-causal credit.** A think token's advantage becomes V(s') − V(s), so the actor ascends the critic's own function.
  - A learned critic's errors are systematic: the same function scores similar states the same way across prompts. They accumulate instead of averaging out as iid noise would. A pattern shared across prompts is what that predicts.
  - This is the same failure as the earlier TD(0) think-fraction ratchet, which was listed as a risk and underweighted.
  - Critic exploitation is a hypothesis. It has not been measured; that needs critic values on the looping and shared-phrase states.
- **v10 barely exercised DG.** With |A| ~0.01, A·s/η is small for nearly every token, and the positive gate mean was 0.502. v10 was effectively PG/2 on TD(0) credit from a near-useless critic.
  - Its failure is evidence about TD(0) over a bad critic. It is not evidence about DG.
  - Its surprisal falling does not clear DG either.
- **Lambda 0 could not have fixed DG in production anyway.** DG's nonlinearity engages only where |A|·s is order 1. In this pool (about 2% success) that is essentially the rare successful rows under lambda ≈ 1 credit: A ≈ +1 there, so the gate rises toward 1 on their rare tokens. This is the same terminal-reward, rare-success regime as TPO Table 3.
  - Lambda 0 shrinks those rows to |A| ~0.01, which removes DG's effect together with the think tokens' only real credit. It removes credit; it does not repair DG.
  - Inference to verify: v9's positive gate mean rose only 0.503 -> 0.527, so any DG effect there must concentrate in a small set of tokens. The step-0 direction test, split by |delight| > 1, is the check.

### Scale-aware DG gate temperature (RMS(A) or baseline b): wins where every token is causal, not a production fix (2026-09-23, jobs 9306–9307)

- **Hypothesis:** with eta fixed at 1 and sparse reward, |A| is ~0.01–0.02 on failures, so the failure gate sits at ~0.5 and only the success-row gate is active (DG paper Prop 1: w- <= pi(a)^(b/eta)). Setting the gate temperature from the batch's own advantage scale should activate the gate and recover DG's benefit.
  - `dgrms`: eta_eff = eta * RMS(A) over the batch's tokens, zero advantages of all-equal groups included.
  - `dgbase`: eta_eff = eta * b. For group advantages, b is the prompt's group-mean reward, so a binary-reward failure has A / eta_eff = -1 / eta (Prop 1's regime). For critic advantages, b is the critic's value level.
  - The magnitude stays the raw A; only the gate temperature changes.
- **Analytic expectation at success rate p << 1 (batch baseline):**
  - dgrms gives failures |A| / eta_eff = sqrt(p / (1 - p)) ≈ 0.14 at p = 0.02, so their gate stays near 0.5; mostly the success gate saturates.
  - dgbase is the variant that activates the failure gate (σ(-s)).
  - Reversal gate telemetry agrees. Failure-token gate mean ± std, episodes 1–100, H=7: dg@3 0.500±0.001, dgrms 0.487±0.018, dgbase 0.377±0.091.
- **TPO Table 3 reversal proxy** (reverse-copy, V=2, terminal all-correct reward, 20 seeds x 2000 episodes, Muon; scratchpad `da621980…/scratchpad/tporepro/`, `out/r1`, `out/r1scale`). Last-100 error, paired-by-seed differences ± SE:
  - **dg reproduces the paper:** H=7 final 35.5% vs 33.8%; H=8 60.0% vs 58.8%.
  - **The gate beats matched PG at one epoch.** The epoch-matched REINFORCE control pg1 fails (97.2%, 99.2%); dg - pg1 = -61 ± 9 pp (H=7) and -39 ± 8 pp (H=8).
  - **Grouped DG:** gdg reaches 1.8% / 1.2% against gpg1 31% / 66%. Every grouped scale variant sits at 0–2.4%, so the grouped arms cannot separate the variants.
  - **Single-sample:**

    | H | dg eta 1 | dg eta 0.03 | dgbase eta 1 | dgbase eta 3 | dgrms | dgrms eta 0.3 |
    |---|---|---|---|---|---|---|
    | 7 | 36.7 | 3.6 | 25.2 | 5.4 | 40.8 | 36.0 |
    | 8 | 60.2 | 5.6 | 43.6 | 4.8 | 50.2 | 47.9 |

    - dgbase at eta 1 is not significant against dg: -11 ± 11 pp (H=7), -17 ± 10 pp (H=8).
    - dgbase at eta 3 is: -31 ± 10 pp and -55 ± 8 pp.
    - **dgbase at eta 3 cannot be told apart from a fixed eta of 0.03** (+1.9 ± 3.1 pp, -0.8 ± 3.2 pp). On reversal, the gain comes from a sharper gate, and adapting it to the batch adds nothing.
  - Operational: `tpo.runtime.bootstrap_runtime()` crashes on the CUDA-13 `nvidia` namespace-package layout, so the runner now sets only its XLA autotune and compilation-cache variables. The full grid runs in minutes on the GPU, not hours.
- **Junk-token toy** (`postraining/sim_dg_sequence_credit.py`, new `dgrms_*` / `dgbase_*` estimators plus gate telemetry; MLP policy, Adam 1e-3, 1500 steps):
  - **eta 1, 3 seeds, with a pool-mean b for gae** (superseded by the critic-level b below):
    - sequential: pg 0.71, dg 0.88, dgrms 0.99, dgbase 0.99 (group).
    - coherence: PG 0.98 (group) / 0.97 (gae); every DG variant 0.38–0.56, filler entropy 2.4–2.9.
    - The scaled gates are more active (group failure gate 0.27 -> ~0.10; gae 0.37 -> 0.18–0.30). They lift the group collapse only slightly (coherence 0.38 -> 0.45–0.47).
    - dgrms normalizes critic noise up to O(1) in the gate, so a larger eta does not rescue it: coherence stays at 0.39–0.65 through eta 8. It remains the best sequential estimator at every eta. It is dropped for critic advantages, where noise dominates.
  - **dgbase with b = the critic's running mean, 6 seeds, partial-critic oracle** (shrink 0.5, noise 0.01, lambda 0.95). Final accuracy / filler entropy:

    | task | pg gae | dg gae 16 | dgbase gae 16 | dgbase gae 32 | pg oracle | dgbase oracle 16 |
    |---|---|---|---|---|---|---|
    | sequential | 0.722 | 0.726 | **0.827** | 0.799 | 0.720 | **0.807** |
    | coherence | **0.970** / 0.73 | 0.963 / 0.91 | 0.954 / 1.05 | 0.961 / 0.93 | **0.987** / 0.49 | 0.977 / 0.75 |
    | chain_coherence | **0.878** / 0.72 | 0.869 / 1.00 | 0.853 / 1.21 | 0.863 / 1.03 | 0.955 / 0.46 | 0.961 / 0.76 |

    - At eta 16–32, dgbase is Pareto-better than fixed-eta DG. Against PG it gains 8–10 points on sequential and loses about 1 point of junk-task accuracy, with 0.2–0.5 nats more filler entropy.
    - Large eta partly just interpolates toward PG: under Adam, σ -> 0.5 is a constant factor.
- **Why this does not transfer to production:**
  - The filler-entropy excess is created early. dgbase_gae coherence at eta 16 goes 1.00 -> 1.29 by step 250, then decays in parallel with PG, while success is low and eta_eff = eta·b is small.
  - In the toy, success climbs within a few hundred steps, so b grows and the gate cools. Production success stays near 2% for the whole run (v9, v10). So dgbase there is effectively a fixed eta of about 16–32 x 0.02 ≈ 0.3–0.6, held permanently in the entropy-raising phase.
  - That is a sharper gate than v9's eta 1, which already showed the entropy blow-up (surprisal 1.0 -> 3.9).
  - The reversal proxy cannot test this risk because every token is causal there. Its win is the sharp gate's win, which is exactly what makes non-causal luck expensive.
- **Verdict:** not qualified for a full training run. Neither the RMS nor the baseline temperature separates signal from luck, since both only rescale the gate.
  - dgbase at large eta is the best DG form measured in the toy. A production test would need a regime where b rises, or a matched short PG/DG/dgbase trio from job 9040 with sampled-token entropy as the stop criterion.
  - These are toy results, and the mechanism paragraph is an argument, not a measurement.
- **Decision (2026-09-23): PG.** The restart uses the clipped VAPO actor (`--no-delightful-policy-gradient`), launched as job 9314 in `postraining/runs/kda8_vapo_cot_v11_pg`.
  - The command is v10's with `--actor-gae-lambda 0` dropped, so the trainer's length-adaptive default applies. Lambda 0 was a DG-specific fix, and it failed.
  - The production critic's EV is about 0.01, which matches the toy's mean-only `gae` arm. There PG was best on every junk-token task, and every DG variant raised filler entropy.
  - Reward-mean DG becomes competitive only with a critic that knows state values (the partial-oracle arm). Revisit it if critic EV rises well above 0.01.
  - The full reversal grid (job 9308) was cancelled at H=8 with these partial results:
    - dgsym H=7 80.3% vs dg 35.5%, which refutes the failure-blame hypothesis.
    - gdgclip 29.3% vs gdg 1.4%, so multi-epoch clipped DG hurts.
    - gdgz 1.1%, so GRPO normalization makes no difference to grouped DG.
    - TPO 0.0% (paper 6.9%), GRPO 15.3% (14.5%), PPO 21.1% (12.0%; the gap is unexplained).
  - Watch sampled-token entropy and surprisal for premature collapse (PG's toy failure mode), think length over all rows, EV, and the bench panel against step 0.

## 2026-09-23 — Full UltraData Math RL pool (v3), DAPO rebuilt to yield (v2), mixture `vapo_broad_v11_exact`

Hypothesis: the 2,306-row `ultradata-math-rl-v2` pool used 10% of the
UltraData-RL-2609 Math domain. The whole domain can replace it, provided the
UltraData and DAPO pools do not serve the same problem under two names. No
GPU work. The artifacts:

- `postraining/data/ultradata-math-rl-v3.parquet` (19,865 rows, sha256 `2d2468a5…`)
- `postraining/data/dapo-math-17k-v2-ud3dedup.parquet` (8,408 rows, sha256 `70ce9d37…`)
- `postraining/data/vapo_broad_v11_exact.manifest.json`, which passed the builder's exact-overlap refusal

### Source

`verifiable_mixed_10k_20260917` already converted all 32,412 published Math
records (four shards whose hashes match `PUBLISHED_COUNTS`). It kept 23,635
source-eligible rows and dropped:

- 6,884 rows with an unverifiable text target
- 1,879 rows with an unverifiable question structure
- 14 non-atomic rows

After 263 duplicate prompts and 11 conflicting-prompt quarantines, 23,361 rows
remained across both splits, so no new download was needed. The v2 builder
deduplicated only by `original_query_sha256` and missed 143
whitespace-identical rows.

### UD vs DAPO overlap by matcher

This is `postraining/problem_overlap.py` on canonical problems, UD 23,355
canonicalizable vs DAPO 17,390, before any filter:

| matcher | DAPO rows matched | answer agreement |
|---|---|---|
| raw prompt (pre-canonicalization) | 0 | – |
| whitespace/case on canonical prompt | 5,193 | 0.961 |
| LaTeX skeleton, not already whitespace-matched | +300 | 0.983 |
| 8-token shingle containment (c >= 0.3, k >= 6, digit guard), not already matched | +809 | 0.670 |
| total | 6,302 of DAPO (36%), 6,110 UD rows | |

Sensitivity of the total to the thresholds:

- c >= 0.5: 6,185
- c >= 0.2: 6,356
- k >= 4: 6,315

The total is flat because most mass sits in the exact and skeleton tiers.

Shingle-tier "disagreement" is mostly not label noise. DAPO rewrites answers
into integers ("... in the form \frac{m}{n} ... provide m + n"; 1/4 against
5). So agreement measures the answer transform, not match precision.

### Calibration

Manual inspection of matched pairs, bucketed by containment:

- c >= 0.5: about 95% are the same problem.
- 0.3–0.4: 18 of 20 are the same problem.
- 0.2–0.3: about 13 of 20. The false positives share template fragments at k = 5–7.

Other calibration findings:

- Pairs with k of 4–7 are about half false positives (BALLOONIST/TARTAR, "Gugu napkin"), hence k >= 6.
- The digit-multiset guard separates sibling variants: the same sentence with 100 instead of 200. At the chosen thresholds without the guard there are 259 UD / 289 DAPO such pairs. Only 9% of them agree on the answer, so they are different problems and are kept.
- Fixes made during calibration:
  - Template clauses shared by more than 10 problems in a pool are ignored: DAPO's m+n clause is on about 640 problems and UD's Chinese "原始的答案是…" on about 310. This cut 0.5–0.6-band pairs from 19,955 to a handful.
  - CJK text is tokenized per character. The ASCII rule matched unrelated Chinese problems on their subscripts.
  - The skeleton keeps `+ - = < >`. `(x+y)^2dx - …` and `… + …` had collided.
- Known false negatives: English/Chinese translations of one problem, and heavy paraphrases below c = 0.3.
- About 220 whitespace-identical cross-source pairs carry different labels. Where checked, UD had the corrected label: the 1996 AIME locker problem is 342 in UD against 1024 in DAPO, and "visionary integers" is 88 against 2020. That is why UD owns the overlap.

Within UD, the chosen rule puts about 2,656 rows in near-duplicate pairs:

- 143 whitespace duplicates
- conflicting labels, such as 380 vs 382, and Bob's binary string at 30 vs 19

A first build used union-find components and quarantined any component whose
targets disagreed. A red-team review of it found that this cost 595 UD rows
and 242 DAPO rows while catching almost no label errors:

- 219 of the 242 conflicting UD components were joined only by shingle edges.
- 86 of the components were answer-form rewrites, where both labels are right for their own question.
- A sample of 12 shingle-only conflicts held 7 rewrites, 5 symbolic siblings, and 0 label errors.
- 25 components were chains, not cliques. One shingle edge fused a clean 4-copy "∫ sin x/x^a converges" cluster with a different e^{-ax}/(x²+1) problem, and the whole component was lost.

The revised policy, which is the one built here, works in two tiers:

- **Same text** (equal skeleton): agreeing targets collapse to the lowest identity; disagreeing targets quarantine the group.
- **Shingle**: a row collapses only into an earlier kept row it matches directly with an agreeing target. No chaining. A different target is kept as a distinct question.

The same review found:

- 430+ DAPO rows with a form-feed byte: `\f`-escape corruption ("\x0crac{m}{n}") mixed with ligature loss ("\x0crst"). The two cannot be separated by rule, so these rows are dropped, never repaired.
- The skeleton erased `^`, `/` and decimal points, so it now keeps them.

The first build's artifacts were never consumed. They are staged outside the
repository, not deleted.

### Ownership

UD owns every shared problem. DAPO yields to all UD Math *candidates*, not
only to the screened pool, so a problem that UD quarantined for a label
conflict does not return under DAPO's label. What DAPO keeps after yielding
(8,619) is not negligible, so DAPO stays as a residual source.

The final pools overlap as follows:

- UD v3 vs DAPO v2:
  - whitespace 0, skeleton 0
  - shingle 6 with pairwise templates, all different problems on inspection. Examples: two regions enclosed by different equations; Chinese problems sharing only a "答案应为" answer clause.
- Either pool vs `deepmind_easy`: 0 under every matcher, with or without template filtering.

### Screens

Both pools pass the screens in `postraining/math_rl_pool.py`, in this order.

| screen | UD v3 | DAPO v2 |
|---|---|---|
| input | 23,361 | 17,390 |
| uncanonicalizable (literal `Answer:`) | 6 | 0 |
| C0 control byte | 72 | 643 |
| target fails self-verification | 0 | 0 |
| eval 8-gram / exact contamination | 859 | 1,317 |
| containment of GSM8K-RL / KodCode | 1 | 1 |
| prompt > 256 GPT-2 tokens | 458 (2.0% of measured) | 568 (3.7%) |
| yielded to UD | – | 5,268 (4,344 whitespace, 248 skeleton, 676 shingle) |
| same-text copies collapsed | 633 | 490 |
| same-text label conflicts quarantined | 8 rows | 10 rows |
| shingle near-duplicates collapsed (agreeing target) | 1,459 | 685 |
| shingle neighbours kept (distinct target) | 260 pairs | 103 pairs |
| kept | 19,865 | 8,408 |

Prompt tokens are measured with BOS on problems that reach the budget screen:

- UD: median 77, p90 161, p99 304
- DAPO: median 85, p90 187, p99 371

The contamination screen is the SFT corpora's rule unchanged: exact match, or
any shared word 8-gram with AIME 2024/2025/2026, GSM8K test, or the DeepMind
bench. It over-rejects competition math:

- Of UD's 866 hits (before the control-byte screen), none exceeds 0.38 containment of an evaluation problem. All 20 sampled hits are false positives on AIME boilerplate such as "m and n are relatively prime positive integers. Find m+n" and "is not divisible by the square of any prime".
- DAPO has exactly one true hit: containment 1.0 of AIME 2026's "Patrick started walking …".

The rule was kept for consistency with the project's single eval-contamination
definition. The rows it costs are AIME-style problems this policy cannot yet
solve.

The mixture cycle is 100,017 prompts:

| source | rows | share |
|---|---|---|
| deepmind_easy | 71,744 | 71.7% |
| ultradata_math | 19,865 | 19.9% |
| dapo | 8,408 | 8.4% |

v10 was 71,744 / 2,306 / 17,390.

### Open issue: verifier brittleness on UD targets

8,538 of the UD v3 targets are not plain integers or decimals. For 1,857 of
them (9.3% of the pool) a trivially equivalent rendering scores zero under the
trainer's Minerva normalization:

- `\dfrac{5}{6}` against `\frac{5}{6}` or `5/6`
- `\[ 2047 \]` against `2047`

`normalize_final_answer` has no `\dfrac` fold. Those rows reward only the
target's exact rendering. All DAPO v2 targets are integers. The fix belongs in
the verifier, as a versioned reward change, not in the pool. Not done here.

## 2026-09-23 — UltraData Knowledge (STEM) for SFT and RL: adapter, 2,048-token cap, benchmark-question screen

This supersedes "UltraData SFT is gated, so it is not in the pipeline" (under
"Model-agnostic VAPO library, and KDA post-training"):
the Code/Math think split already had an adapter, and now the Knowledge
think split does too. No GPU time was used. The built artifacts, with the
answer balancing, GPQA and science pools that followed, are under "Final
artifacts" at the end of this entry.

**Data on disk.** All 50 shards of `data/think/Knowledge` (13.9 GB, 499,667
two-turn records) at revision `affda6ac` are in
`instruction_corpus_shards/ultradata_sft_2605_hf`. Every shard matches the
hub's `lfs_sha256` (`knowledge_think.sha256.json`). MMLU `all`
test/validation/dev (`c30699e8`) and MMLU-Pro test/validation (`b189ec76`) are
in `postraining/data/stem_eval_decontam` with a verified `manifest.json`, and
they are for screening only. GPQA was missing at first (gated); it was
added later the same day, see "Final artifacts". Before this, the only
verifier-ready STEM data on disk was UltraData-RL-2609 Knowledge, and there
was no local MMLU, MMLU-Pro, GPQA, ARC, SciQ or OBQA.

**What the Knowledge split is.** It is keyword-classified as 45% chemistry,
22% math, 16% physics, 4% biology and 3% engineering. About 0.07% of it
contains CJK, and every record's `source` is the dataset name, so there is no
upstream provenance. The records fall into these categories:

| Category | Share | How the adapter treats it |
|---|---|---|
| Untemplated MC | 35.2% | Dropped: it concludes in dozens of free styles |
| Templated MC | 30.8% (154,111 rows) | Admitted |
| Free-form | 22.8% | Dropped: no gradeable answer |
| MC with options missing | 10.8% | Dropped |

A templated MC row has a two-line header that declares the labels and a
`'<prefix> $LETTER'` conclusion (8 prefixes, used equally). Its presented
answer ends by filling that template in. Among the templated rows:

- Label styles are split evenly: upper 52.9k, lower 51.0k, digit 50.2k.
- Separators are split evenly too: `:` `)` `.` `-` each about 38k.
- Most rows have 8 options (94.9k) or 10 (47.1k).
- The 160-character identity duplicates 1.5% of them.

**The traces are long.** The median templated document is 5,070 tokens (p10
1,932, p90 11,617), which is longer than the traces that diluted the 5,120 run
(2026-09-22). Here is what each document cap keeps:

| Cap (tokens) | Documents kept | Share | Tokens |
|---:|---:|---:|---:|
| 1,024 | 1,035 | 0.7% | 0.9M |
| 2,048 | 18,061 | 11.7% | 28.6M |
| 3,072 | 39,958 | 25.9% | 84.3M |
| 5,120 | 77,893 | 50.5% | 238.8M |

At 5,120 this one source would match the whole 293.8M-token canonical mix.
The adapter therefore carries a per-source `max_doc_tokens=2048`, which binds
under any `--seq-len`, and the manifest records it.

**Most templated traces talk about the header this contract removes.** The
header is stripped from the prompt, but the teacher's reasoning still quotes
it, for example "Final line: 'The correct answer is $LETTER'" or "Respond with
a single symbol from A-H". Training on those traces would teach the model to
answer an instruction it never sees. A first scratch build (17,958 documents,
28.5M tokens) found that 75.6% of admitted traces contained `$LETTER`, and a
review caught it. `choice_prompt.cites_answer_format` now drops any row whose
reasoning cites `$LETTER`, a "last/final line", "without quotes", a
"required/answer/output format", or "the instructions". It leaves "The user
wants ..." restatements alone. With the screen in place, the end-to-end scratch
build (`--seq-len 5120`, GPQA patched out) produced **2,803 documents and 4.35M
tokens** (median 1,592) from 499,667 rows. The drops were:

- 343,992 not templated;
- 99,720 whose reasoning cites the answer format;
- 51,460 over the cap (26,463 over length, 24,670 by the prefix probe, 327 by the character bound);
- 1,182 with no single-label conclusion;
- 382 with malformed framing;
- 63 caught by the math-eval index;
- 28 with a fence literal;
- 27 with a completion that was too short;
- 4 duplicates;
- 3 caught by the benchmark-question screen;
- 2 caught by the KodCode/GSM8K containment.

**This is small, and it is skewed.** Answer positions are b 32%, c 28%,
d 15% and a 13%, while chance is about 12%, so SFT would teach a B/C prior.
Options can be permuted only where the trace's label references are provably
complete; the balancing that follows is under "Final artifacts". The answer
span is the label exactly as the prompt shows it (`b`, `2`, `B`). Before the
format screen, a 2,048 cap kept 43% chemistry, 24% math and 17% physics.

A larger clean source would be the 35% of *untemplated* MC rows. They never
saw a header, so their traces cannot cite one, but they conclude in dozens of
free styles, and a label extractor for them would need its own verified
conclusion grammar. That is not built.

**The benchmark-question screen is a new rule, not the 0.30 containment.**
Exam stems are short and formulaic. At 0.30, the containment rule flagged 86
RL Knowledge queries, and every inspected one shared only phrasing ("which of
the following statements best describes the") with a short MMLU question.
Distinctive-gram filtering alone does not fix it at small n. With 5-grams at
0.4–0.5, UltraData's own boilerplate ("statements accurately describes a") is
rare in MMLU and still matched.

`contains_benchmark_question` works in two steps. First it checks for an exact
normalized match of the problem or of its question stem. Then, over 8-grams
that occur in at most 3 benchmark questions, it flags a candidate that holds
≥60% of a question's distinctive grams, for questions with at least 4 of them.
A verbatim-substring test for shorter questions was tried and removed: "Which
of the following is not a true statement?" (an MMLU question with 2
distinctive grams) rejected fluorescence problems.

Results against MMLU and MMLU-Pro:

| Measurement | Result |
|---|---|
| RL queries flagged | 0 of 11,872 (the highest scored 0.5, on phrasing only) |
| SFT templated rows flagged | 10 of 154,111, all genuine (quartic k=86, the variable-force integral, Larmor frequency, Bob's die) plus one generic exact stem |
| Recall, stem rendered with options | 2,000/2,000 |
| Recall, embedded after a prefix sentence | 83% (the misses are the 21% of questions with <4 distinctive grams) |
| Recall, one mid-sentence word changed | 47% |

Paraphrase is not caught. That is the limit of a text screen.

**The RL pool (`scripts/build_ultradata_knowledge_rl_prompts.py`, dry run,
GPQA unscreened) has 2,413 rows.** 2,320 are single-choice: 1,980 with 10
options, 321 with 8, 15 with 7 and 4 with 9. The other 93 are numeric. The
builder:

- parses options with `choice_prompt.split_options`;
- resolves the target by label or by unique option text, and resolves it to nothing when both name different options, which avoids 14 wrong letters from digit labels over numeric options, such as truth "3" with an option "4: 3";
- drops duplicate option texts (4), cross-references in options or the question (262, e.g. "none of the above"), and repeated question stems (35);
- permutes options with a content-seeded RNG, so the answer position is uniform (modal letter 11.3% against a source skew of B 818 to J 241);
- renders options as `A. option` lines;
- grades one exact upper-case letter (`rule` → `exact`).

The other drops were 7,127 free-text targets, 2,007 rows over the 256-token
prompt budget and 20 unresolved targets. Chance accuracy is 10.4%, reported
per `choice_{k}` module. Forty-four raw RL queries overlap SFT Knowledge stems
from the pre-screen build. That is train/train overlap, not evaluation
contamination, but a stage-2 run that uses both sources should exclude one
from the other. The mixture
source `ultradata_knowledge` is **off by default**. MiniCPM5-1B scored 0.39%
on this slice, and a 64M model is likely at chance on graduate STEM MC, so run
a frozen gate that shows accuracy above chance before any RL. Otherwise mixed
groups come from lucky letters and carry no signal.

### Final artifacts: answer balancing, GPQA screen, science pools

**GPQA is a screen target only.** `Idavidrein__gpqa` at revision `83022cef`
is in `stem_eval_decontam`: extended 546, main 448 and diamond 198 rows (main
and diamond are subsets of extended), each file matching the hub's git blob
hash. Its terms were accepted; it is CC BY 4.0 and must not be republished.
Everything under `postraining/data/stem_eval_decontam` is evaluation-only:
`shard_paths` and `prepare_vapo_mixture` refuse any path under it, and
`tests/test_benchmark_isolation.py` fails if:

- a declared input lies under it;
- a GPQA question, bare or rendered with its options, passes the production
  screen;
- a built Knowledge or science artifact holds a GPQA question.

The SciQ, ARC and OpenBookQA validation and test splits are in the same
directory and were added to `STEM_QUESTION_TARGETS`, which now holds 15
targets and 36,469 texts.

**SFT answer balancing, first build (superseded).** `sft_ultradata_knowledge_v1`
(630 documents, 109 relabelled) was built under looser proof rules. A
subagent red team then checked all 109 relabelled pairs against their
originals, and **27 were contradictory**. The trace's label and the
option it named no longer agreed after the permutation. The final rules close
these classes of error:

- a label-shaped token that was not an option: an article, a variable or a
  quantity (`B = 0.5 T`);
- `(X)` and `**X**` taken as proofs of a reference;
- a keyword on the previous line vouching for a label on the next;
- quoted, possessive, `$`-, brace- or underscore-adjacent labels (`'D'`,
  `__D__`) that the tokenizer did not see and so left unrenamed;
- keyed labels outside the row's label set, which silently stayed put;
- positional wording ("the last two options", "the former").

The byte-exact inverse round trip could not catch any of these. It only
proves the tokenization is stable, not that the renamed token meant an
option. v1 was never used. It and the matching RL v1 files are staged in the
session scratchpad, not deleted.

**SFT answer balancing, final rules.** A row is relabelled only when every
standalone label token in its trace, including quoted, possessive, `$`- or
brace-adjacent ones, is provably an option reference. A token that is an
article ("a force") or is followed by `=`, `<`, "stands for" and similar
never counts. The only proofs are:

- a keyword directly before the label, on the same line ("option B", "answer
  is B");
- `\boxed{B}` for letter labels;
- a line that restates the option as label, separator and that option's
  text;
- a list continuing a proven reference, with no clause following.

A row is refused when any of these holds:

- the trace names a label the row does not have;
- the trace, question or options use positional wording ("the last option",
  "the former");
- the question or an option refers to options itself (cross-reference
  phrases, bare label lists);
- an option text holds any non-article label token;
- digit-labelled options contain label numbers.

Only **26 of 1,526** rows that pass the format and trace screens qualify
(1.7%). The main refusals are unprovable "a"/"A" (373 + 309) and "1" (98).
`balance_choice_answers` is unchanged: exact per-option-count balance, with
unrelabelable rows at their position and relabelable rows filling the
deficits. The coordinator kept exact balance over yield, because the tokens
are negligible next to the 293.8M-token base mix.

**Additional screens in the final build:**

- `cites_answer_format` also catches quoted answer phrases ("'The answer
  is'"), "format the conclusion as", "concluding sentence", "a specific
  format", "end with 'The correct option is'", and a capitalised `LETTER`
  placeholder. It drops 103,407 rows.
- A new `defective_trace` drops traces:
  - that never state the label in their last 300 characters (2,972);
  - whose text degenerates into a 20+ repeat of a short unit (6);
  - that end in sentence debris such as ".ted" (10).
- Deduplication for this adapter uses `choice_identity`: the casefolded
  question plus the sorted option texts. The same question under A–H and 1–8
  labels, or with reordered options, is one problem. The manifest's
  `deduplicated_by` now records the identity per corpus.

Result: **235 documents, 379,353 tokens, 26 relabelled.** Answer positions
went from a 195, b 462, c 421, d 237, e 93, f 42, g 37, h 15, i 11, j 16 to
a–h 25–27 each, i 13, j 13 (i and j exist only with 10 options). By option
count:

| Options | Rows before | Relabelable | Documents |
|---:|---:|---:|---:|
| 10 | 368 | 3 | 131 |
| 8 | 1,022 | 18 | 89 |
| 7 | 91 | 5 | 15 |
| 5, 6, 9 | 25, 2, 21 | 0 | 0 |

With no relabelable rows, the balanced size of the 5-, 6- and 9-option groups
is zero, because some position in each has no source rows. I read all 26
relabelled pairs against their originals. Every renamed token names the
same option text as before, and every conclusion and `<answer>` follows. One
cosmetic effect remains: a trace that walked the options in order now walks
them in shuffled label order.

**RL option-text rule.** The first Knowledge RL v1 build let options such as
"All of A, C, D, E, G, and H" through, and the shuffle breaks them. It was
never used and was replaced the same day. `choice_rl_pool.choice_problem`
now:

- drops bare label references (label lists, or an option that is just a
  label) against both the rendered letters and the source labels;
- drops positional references ("the previous statement", "any former");
- drops a rendering that does not parse back to the same options (SciQ's
  "E. coli ..." questions read as an option line).

The narrower `POSITIONAL_REFERENCE` is used rather than the SFT's
cross-reference phrase, because "all of these" survives any shuffle. The
tokenizer fixes above made the label-list rule stricter. Rebuilt as
`ultradata-knowledge-rl-v2`, it drops 14 more rows than v1, and every other
row is byte-identical:

- 2 are true positional references;
- 12 are false positives: math that collides with labels, such as `$G/H$`,
  `D/H`, `{1, 1, 10}` or `Q^{1/2}` against digit source labels.

Together with "A/B testing" and the like, the rule costs about 6% of the
choice rows. It fails closed.

**Science MC pool.** `scripts/build_science_mc_rl_prompts.py` takes the ARC-Challenge,
ARC-Easy, OpenBookQA and SciQ **train** splits only, each pinned by revision
and sha256. It uses the shared single-choice contract with modules
`{source}_choice_{k}` and chance mostly 1/4. Screens:

- MMLU, MMLU-Pro, GPQA and each source's own validation and test splits; a
  generic stem shared with an evaluation question drops the row even when its
  options differ;
- owners: the Knowledge RL v2 pool, the Knowledge SFT v2 corpus and every
  earlier source, in ARC-Challenge, ARC-Easy, OpenBookQA, SciQ order.

No math pool is an owner. The math pools are free-form, not single-choice.
SciQ lists its answer first (13,722 A in source order), and the shuffle
brings every letter to about 25%. SciQ is CC BY-NC 3.0 (non-commercial);
OpenBookQA's hub card says "unknown" (Apache-2.0 upstream). The mixture
source `science_mc` is off by default.

Built pool, 19,763 rows. `science-mc-rl-v2` is byte-identical to the v1
build: the new owner files change nothing, and the new RL rules drop no
science row. Only its manifest's owner hashes differ.

| Source | Seen | Kept | Evaluation screen | Other drops |
|---|---:|---:|---:|---|
| ARC-Challenge | 1,119 | 1,095 | 13 | 8 cross-reference, 2 duplicate options, 1 duplicate |
| ARC-Easy | 2,251 | 2,224 | 19 | 4 cross-reference, 4 owned |
| OpenBookQA | 4,957 | 4,851 | 53 | 42 duplicates, 6 cross-reference, 3 duplicate options, 2 owned |
| SciQ | 11,679 | 11,593 | 42 | 36 duplicate options, 4 cross-reference, 3 ambiguous rendering, 1 owned |

Letters: A 4,980, B 4,795, C 5,023, D 4,964, E 1. The modal share is 25.4%
against 25.0% chance. Prompts have a median of 37 tokens and a max of 189.
No science question matched the Knowledge pools.

| Artifact | Rows | sha256 |
|---|---:|---|
| `postraining/data/sft_ultradata_knowledge_v2.parquet` | 235 docs | `e5a0505b908b07d100b6b553b93c245e79d5761a790f941c9ed385f763304c5d` |
| `postraining/data/ultradata-knowledge-rl-v2.parquet` | 2,337 | `aee5610b4a8ef70598467ce971911a2a79b2669d6341c11c6cf389cfeea41728` |
| `postraining/data/science-mc-rl-v2.parquet` | 19,763 | `77c6f99be840d5219b7b5a5e2785da2d0eaa96c634fe50d37f56c6a80e2a0538` |

The Knowledge RL v2 pool keeps 2,337 rows: 1,926 with 10 options, 300 with
8, 14 with 7, 4 with 9 and 93 numeric. Its letters run from 186 to 252
(modal 11.2% against 10.4% chance); in source order they ran from 142 to 343.
`prepare_vapo_mixture` and the isolation test point at the v2 files. The
isolation tests pass on all three, and the CPU suite passes except for
failures unrelated to this work: CUDA-only tests, a missing ablation
checkpoint, and `UnoTrainingRolloutEngine.logprobs`.

**Risks.**

- The balanced Knowledge SFT corpus is tiny: 235 documents and 0.38M tokens,
  0.13% of the base mix. It will teach the choice format and little
  knowledge. 98% of its traces cannot be relabelled provably, and the 5-, 6-
  and 9-option groups are empty.
- A relabelled row that exceeds the token cap is dropped rather than kept, so
  that position ends one short (T-1). The cap is applied before balancing, so
  this is rare.
- The Knowledge source has label noise: some rows have two defensible
  answers.
- A 64M model is probably at chance on the Knowledge pool (10.4% chance), so
  run a frozen gate before selecting either choice source.
- The text screen does not catch paraphrases.

## 2026-09-23 — Exact-pass RL mixtures, fused behavior statistics, rollout decode flags

Job 9314 (`kda8_vapo_cot_v11_pg`, PG from the job 9040 SFT base) was healthy
when cancelled at step ~950: bench accuracy rose from 0.0095 to 0.0547 by
step 750. It was cancelled to adopt two changes it could not pick up mid-run.
Its `CANCELLED.md` records this.

### Exact-pass mixtures (`vapo_verifiable_mixture/v2`)

The v8/v9 manifests fixed per-rollout quotas (48/8/8 of 64). This replayed
the 2,306-prompt UltraData pool about 31 times for every pass over
`deepmind_easy`. A v2 manifest carries no quotas:

- One cycle serves every row of every source exactly once. The manifest records `prompts_per_cycle`, which the loader checks against the rows it reads.
- Sources are interleaved by a largest-deficit schedule over row counts, so every window of the cycle holds each source at its share to within about one prompt.
- Cycle 0 takes each source in file order; later cycles take a seeded reshuffle. The sampler cursor is the only resume state.
- v1 manifests and `"quota"` entries are rejected.

Two related changes:

- Optimizer minibatches deal each source's groups by the same largest-deficit rule, so no minibatch is single-source. This now holds even when a pool is split into several minibatches.
- `source_diagnostics` omits a source that is absent from a pool instead of raising. With exact-pass scheduling, a small source can be missing from a pool.

A run meets "one corpus iteration = one pass of every problem" when
(value-warmup steps + policy steps) x 64 is a whole number of cycles, because
warmup draws from the same cursor. For v12 (99,917 prompts, odd), no step
count is exact below 64 cycles. 38,730 policy steps after 300 warmup steps
end 5 prompts short of 25 cycles: every problem is served 25 times except 5,
which are served 24.

### Fused behavior statistics

When a pool is a single optimizer minibatch, the behavior policy is the
current policy (behavior age 0). The separate refresh forward that
recomputed old log-probs and values was pure overhead: about 0.52 s/step
(`pool_refresh_seconds`, job 9314). `update_minibatch` with
`fused_behavior_statistics=True` now:

1. runs the critic pass and writes `batch.old_values`;
2. computes GAE once;
3. runs the actor pass with old = new.detach().

Actor gradients are bit-identical to refresh-then-update. The critic differs
by at most 6e-8 (CPU tests, delightful and source-gate variants, cot and
carry). The fused path is off for latent mode, TPO and multi-minibatch pools,
and it refuses latent thoughts. Consequences for reading metrics:

- The behavior-age-0 canary is vacuous in fused runs: old equals new by construction.
- `rollout_metrics.fused_behavior_statistics` marks such steps.
- `pool_refresh_seconds` and `old_value_mean` are absent, so the old `collect_seconds - pool_refresh_seconds` recipe does not apply.

GPU suite job 9348 ran 1301 passed and 2 failed, both unrelated:

- `fixed_yarn`: missing file
- `uno_speculative`

### Rollout decode flags (tp_v10_rollout_*, v10 prompts, warm repeats 1-5)

| flags | collect_seconds, repeats 1-5 | trajectories |
|---|---|---|
| default | 3.28 3.13 3.93 4.97 4.56 | reference |
| `--rollout-tail-graph` 16 | 3.26 3.24 3.06 3.18 3.15 | identical to default |
| `--rollout-tail-graph` 64 | 3.26 3.19 3.13 3.47 3.38 | identical to default |
| sync every 8 | 3.15 3.20 3.41 3.22 3.32 | differ |

Agents were running CPU-bound work during these jobs, which explains the
default's spread from 3.1 to 5.0 s. The best repeats are within noise of
each other. Decode defaults are unchanged; tail16 is worth re-measuring on a
quiet host. Collection was 65% of the 4.06 s pre-fusion step. The fused step
should be about 3.5 s (4.06 - 0.52). That figure is an estimate, not a measurement.

### Math verifier v4 (`terminated_final_answer_exact1_numeric_log1p_max01/v4`)

This resolves the "verifier brittleness" open issue of the UltraData v3 pool
entry above. The policy, trained with SFT on OpenMathInstruct-2 and drills,
writes fractions as `\frac{a}{b}` (77k SFT answers) or `a/b` (the drills).
1,795 UltraData targets are `\dfrac`, and under v3 a correct answer scored
zero on every one. v4 changes `postraining/core.py` as follows:

- **Notation that changes typesetting but not value:**
  - `\dfrac`/`\tfrac` fold to `\frac`.
  - `\left`/`\right` sizing is dropped, with `\rightarrow` guarded.
  - A whole-answer `$…$`, `\[…\]`, `\(…\)` or `\boxed{…}` is unwrapped before and after the `=` split.
  - A bare `p/q` becomes `\frac{p}{q}`.
- **The `=` split** happens only at an `=` outside braces, so `\sum_{k=1}^{n}` survives.
- **Commas** collapse only as thousands grouping. Under v3, `2, 5` graded as `25`, and `7, 6, 2, 2, 2, 1, 5` graded as `762, 22, 15`.
- **The single-number gate** reads `graded_answer_field` of both the truth and the prediction: the value after an optional single `head =`, where head is a variable, subscript or `f(x)`. `n = 25` therefore gates like `25`, while `a = 7, b = 3` and `19,7` state no single number. Digits separated only by whitespace never grade as one number (`4 0` against `40^\circ`).
- **`parse_numeric_answer`** accepts the same wrappers, a spaced leading sign (and rejects a second sign), spaces inside `\frac{ }{ }` and around `/`, and `\frac12`.
- **Kept strict:** unreduced (`2/16`) and decimal (`0.125`, `5.00`) forms, and leading zeros. The zeros are digits of the answer to questions like "last four digits": `0352`, the binary string `0111`.

Measured on the pools:

| check | v3 | v4 |
|---|---|---|
| `\dfrac` → `\frac` variants accepted | 1 of 1,795 | 1,795 |
| `\frac` → `p/q` variants accepted | 1 of 1,954 | 1,835 |
| `\[…\]`-wrapped variants accepted | 22 of 19,865 | 19,851 |

Other results:

- Every target still verifies against itself.
- Nothing changed on DAPO.
- 317 saved RL answer spans and 500 SFT gate-transcript spans grade identically under v3 and v4, so matched comparisons against job 9040 and 9314 stand.

Two rounds of adversarial review found no value-changing hack. `verifiable_reward_identity` now hashes the new helpers.

**Equation targets.** Minerva keeps only the text after the last `=`. A
target like `x + 2y - 5 = 0` therefore paid a bare `0` and zeroed the
equation actually asked for. `math_rl_pool` now quarantines Minerva targets
with no graded field as `equation_target` (102 rows): an equation between
expressions, several equations, or `a = b = c`. Pool manifests record
`policy.reward_schema`.

The pool is rebuilt as `ultradata-math-rl-v5` (19,765 rows, sha256
`output_sha256` in its manifest), and it is the `ultradata_math` default. An
intermediate v4 build, never consumed, is staged outside the repository.
Remaining known gaps:

- 63 degree targets such as `40^\circ` fall outside the numeric gate. They are covered only by the digit-space rule.
- English/Chinese translations are still missed by the overlap matchers.

The reward-schema bump means an `--actor-critic-init` checkpoint from v3,
such as job 9268's critic warmup, is refused. The next PG run needs a fresh
warmup.

## 2026-09-23 — UltraData-Code L3/py: sandbox-verified SFT corpus and RL pool

**Hypothesis.** UltraData-Code L3 exercises (`task` / `analysis` / `solution`
/ `test`) can supply both code SFT and code RL for the 64M model, provided
a row is admitted only when this repository's own Python reward agrees with
it. The dataset card calls the tests "test candidates", and nothing upstream
ran them. No GPU time was used. Nothing is in a mixture yet.

**Mirror.** 19 of 147 L3/py shards at revision `85182d82`, evenly spaced
(1, 9, …, 147; upstream shard order is not documented as random): 2,681,998
rows, 20.57 GB, each sha256 equal to the hub LFS hash
(`instruction_corpus_shards/ultradata_code_hf/l3_py.download.manifest.json`).
The hub's xet transfer stalled at 67 MB, so the tool streams the pinned
`resolve` URLs over plain HTTP with retries.

**Verification is whole-module, and the pass rate is about half.** On a
2,000-row pass, 51% verified. Of the 978 rows that failed:

| Asserts failing | Rows |
|---|---:|
| 0 (a non-assert statement raised) | 16 |
| 1 | 358 |
| 2 | 232 |
| 3 | 146 |
| 4 or more | 211 |
| Crashed or timed out | 15 |

The median failing row still passed 80% of its asserts. Per-assert salvage
was rejected. Data is abundant, and a partial pass means solution and tests
disagree somewhere, so the "passing" asserts only certify agreement with a
solution that is wrong elsewhere.

In v2, 140,817 of 378,768 sandboxed rows verified (37.2%; 50–52% before
repeat-skipping). The remaining verdicts:

| Verdict | Rows |
|---|---:|
| `reference_tests_failed` | 237,737 |
| `reference_too_slow` | 160 |
| `reference_nondeterministic` | 20 |
| `verifier_error_RuntimeError` | 18 |
| `stub_passes` | 16 |

The 18 verifier errors survived three attempts, so they are row-specific
rather than transient. The suspected cause is a program too large for one
argv argument (E2BIG), and it is unconfirmed. These rows are rejected,
since they would raise inside RL too. For verified rows the wall time was
p50 28 ms and p99 86 ms.

**Screening drops (of 2,681,998).** The largest single filter is the
reward's own AST policy: 32% of reference solutions would be rejected as
policy output (`.startswith`, `typing`, and so on).

| Drop | Rows | Reason |
|---|---:|---|
| `solution_policy_rejected` | 858,598 | Fails the reward's AST policy |
| `meta_reference` | 442,730 | Cites "the snippet" or "original code", context the prompt never shows |
| `entry_point_not_in_task` | 110,814 | Task never names the function it grades |
| `test_import_unsupported` | 44,015 | Test imports an unsupported module |
| `contains_reference_problem` | 30,496 | Evaluation containment |
| `test_syntax_error` | 21,214 | Test does not parse |
| `test_unbound_name` / `test_reads_solution_name` | 10,164 / 7,426 | Tests read solution-only imports, classes or constants |
| `contaminated_math_index` | 5,411 | Math decontamination index |

That leaves 1,147,903 screened rows. Deduplication then removed 186,128
exact duplicates and 2,130 MinHash near-duplicates (126 permutations, 42×3
bands, estimated Jaccard ≥ 0.5; exhaustive checks on a sample found no pair
above 0.5 that the LSH missed).

**Containment is conservative and mostly false positives.** A 16,949-row
row group produced 318 raw containment hits. Almost all were KodCode, via
boilerplate ("Create a function that takes a list of integers and
returns…"). There was 1 MBPP hit (also a false positive) and 0 HumanEval hits. The rule
therefore costs about 1–2% of rows for little real protection. This
contradicts the README's earlier "loses about 1 problem in 800" for
UltraData-SFT code prompts, which are a different distribution.

**Paraphrase duplicates dominate, so problem identity is the entry-point
name.** After exact and MinHash dedupe, 34% of 3,726 verified sample rows
shared their entry-point name with another row (`factorial` 30×,
`is_palindrome_number` 19×, `max_profit` 17×). Those pairs are the same
problem reworded, at a median estimated Jaccard of only 0.056, and 13% reach
0.15. Word shingles cannot separate them. Therefore:

- An identity (the sorted entry-point names) belongs to one pool only.
- RL takes 1 row per identity and SFT at most 3.
- Saturated identities are skipped before sandboxing: 492,603 rows. Without
  the skip, about 60% of verified rows were repeats.

Given the same verdicts, the pools are identical for any worker count. The
verdicts are wall-clock bound, so heavy load can flip a borderline row. A
second review found about 290 near-twins across the pools under different
names, at estimated Jaccard 0.3–0.5 (`cigar_party` / `party_success`). The
build now drops SFT rows at estimated Jaccard ≥ 0.3 to any RL task
(63-band LSH): 294 rows in v2. In v2, 20,000 RL identities and 90,794 SFT
identities share nothing.

**RL pool** (`ultradata-code-l3-v2-rl.parquet`, 20,000 rows, MBPP row
schema, `SOURCE_SPECS` entry off by default):

- Prompts are the bare task plus 2 example asserts (18,038 rows) or 1
  (1,962). Prompt tokens have mean 195 and max 255, within the 256 budget.
- Grading covers every kept statement: median 10, p99 27.
- A row is RL-eligible only if the unshown statements call an entry point
  with arguments no shown example uses.
- A scratch `prepare_vapo_mixture --sources ultradata_code_l3` manifest
  loads.
- 200 of 200 sampled reference solutions pass through
  `normalize_python_answer` / `python_tests_pass` on the RL rows, and 0 of
  200 empty answers pass.

**SFT corpus** (`sft_ultradata_code_l3_v2.parquet`): 116,525 documents,
102.5M tokens, every row `verified = True`.

- Document tokens: mean 879, p10 543, p50 800, p90 1,343, p99 1,899.
- The adapter's 2,048-token ceiling dropped 3,668 of 120,206 rows (3%).
- 74% of documents fit 1,024 tokens.

The v1 artifacts (`ultradata-code-l3-v1*`, `sft_ultradata_code_l3_v1*`)
predate the cross-pool screen and the SFT-target fix. They are superseded
and were never consumed.

The 5,120 run (2026-09-22) lost because long traces diluted drills, and
this corpus alone is a third of the 293.8M-token canonical mix. So mix it
with a `--source ultradata_code_l3=N` cap sized against the drill share, and
prefer 1,024-token packing when extending job 9040's lineage. Code rows are
`gradeable = False`, so they stay out of the math sampling-gate panel.
`load_documents` now refuses a partly verified corpus without
`--allow-unverified`.

**Known risks.**

1. **Reward hack, pre-existing in `code_reward.py`.** Candidate code and the
   tests share one namespace, and the AST policy allows module-level
   rebinding of builtins (`abs = lambda x: 0`) and stores to safe attributes
   of imported modules (`math.sqrt = …`). A near-miss answer can therefore
   neutralise tolerance asserts. Among screened rows, 5% use `abs(`/`isclose`
   tolerance asserts and 16% name a builtin in an assert. A generic
   "stub + shadow builtins" candidate passed 0 of 370 verified rows, so this
   is not a blanket exploit. Fixing it changes the reward and needs a
   `PYTHON_REWARD_SCHEMA` bump, which also touches MBPP.
2. **SFT and RL prompts differ in shape.** SFT prompts are the bare task and
   show no examples; RL prompts append `Example(s):` asserts.
3. **A pass certifies consistency, not correctness.** The tests are
   synthetic, and a verified pair can agree on a wrong spec.
4. **Licensing.** Apache-2.0 plus each source repository's license, with no
   unchanged redistribution.

### Choice sources fail the frozen gate; math-only v12 launched

Job 9466 (`kda8_gate_choice_9040`) ran a frozen rollout gate of the job 9040
SFT policy on `vapo_gate_choice_v2`:

- `science_mc` v2: 19,763 rows, chance mostly 1/4
- `ultradata_knowledge` v2: 2,337 rows, chance about 1/10

It ran 8 repeats of 64 prompts x 16 samples at the production 768-token
budget:

| metric | science_mc | ultradata_knowledge |
|---|---|---|
| exact accuracy per repeat | 0.000–0.005 | 0.000–0.009 |
| fenced-format fraction | 0.17–0.24 | 0.13–0.36 |
| terminated | about 0.38 | about 0.38 |

Within-group reward std was at or near 0, so the gate failed (exit 2). The
policy was never taught to answer with a letter. It is far *below* chance,
not at it, so RL from 9040 has no signal on these sources. They need a base
whose SFT includes letter-answer traces. The 235-document Knowledge SFT
source alone is too small for that, so a choice-trace SFT source is needed
first.

The RL mixture for the 9040 base is therefore math-only:

- `postraining/data/vapo_broad_v12_exact.manifest.json`: deepmind_easy 71,744 + `ultradata-math-rl-v5` 19,765 + dapo v2 8,408 = 99,917 per cycle. It is the default.
- Critic warmup job 9467 (`kda8_critic_warm_v12`, 300 steps, `--critic-init actor`) runs under reward schema v4. The v3 warmup from job 9268 is refused.
- PG job 9468 (`kda8_vapo_cot_v12_pg`) is gated on 9467's success: 38,730 steps, the same recipe as the cancelled job 9314.

## 2026-09-23 — Python reward v6: isolated test namespace; code pools v3; v12 PG stopped

**Math-only PG on v12 (job 9468, from critic warmup 9467).** Plain PG
(`--no-delightful-policy-gradient`) learned: `bench_policy_avg@1152` 0.0095
(step 0) -> 0.0226 (250) -> 0.0295 (500) -> 0.0573 (750), against job 9314's
0.0547 at step 750; AIME stayed 0. The critic warmup ended at explained
variance 0.018 (mean target 0.017; sparse reward). Cancelled at step 912 by
decision: every RL step on the 9040 base is redone after the next SFT stage
(math + code + letter-answer data), so the GPU goes to that. The last
exact-resume checkpoint is 19:25; it cannot resume under the new
Python-schema binding rule below (its payload records v5).

**Reward hack closed (v5 -> v6, `bwrap_python_positive_ast_isolated_tests_binary/v6`).**
v5 executed candidate and tests in one namespace. Five hacks passed v5 with a
wrong answer (`abs = lambda x: 0`; a deferred `global abs` rebind inside the
entry point; `sorted = ...`; `m = math; m.sqrt = ...`, eagerly or at call
time; `C = Counter; C.most_common = ...`). In the 20,000 v2 RL rows, tests
reference `abs` 1,608 times, `sorted` 742, `sum` 337. v6:

- tests run in a separate namespace over pristine builtins and receive only
  `verification_info.entry_points` (required; UltraData-Code: the solution
  functions the tests call, already the only solution names screening
  admits; MBPP: `mbpp_entry_points`, reference module-scope bindings the
  tests read; KodCode: the one `from solution import`). Declared entry points
  may shadow builtins (2 of 20,000 tasks ask for `sum`/`pow`);
- candidate imports return private module copies; classes of imported
  modules are snapshotted and must be identical (keys and value identity)
  after the candidate and after each test statement;
- the AST policy rejects attribute/subscript stores rooted at an
  import-bound name;
- two pre-existing harness bugs found by the red-team: candidate `print`
  output was flushed after the sentinel, so any printing solution failed
  (402 of 120,206 v2 pool solutions contain `print(`); and `-I` implies
  `-E`, so `PYTHONHASHSEED=0` never applied and set-order verdicts were
  random (one v2 pool row passed 37/64 runs). stdout now goes to /dev/null;
  bwrap `--clearenv` plus `-s -P -S` replaces `-I`.

Checks: all 374 MBPP train references pass; the AST rule changes no verdict
on the 120,206 v2 SFT-pool solutions; 26 benign import/identity probes
(datetime, re.Pattern, Counter identity, heapq, namedtuple, module
identity) match v5; on 2,500 sampled v2 pool rows, 6 v5 passes fail v6, 5
of them tasks whose *tests* define a helper the solution uses ("helper is
provided") — unsolvable under isolation, dropped by rebuilding. Known
residual: in-place mutation of a mutable container held by a library class
is not detected by the identity snapshot. The red-team's exploit-search
goal was blocked by a safety classifier, so the exploit search is the
author's own review, not an independent one.

The trainer now binds `python_reward_schema` (mixture check, checkpoint
payload, resume, run manifest) only when the mixture contains a
`python_mbpp` source, so math-only mixtures such as `vapo_broad_v12_exact`
remain launchable across Python-verifier revisions.

**UltraData-Code pools v3** (`ultradata-code-l3-v3`, build schema v2, same
arguments as v2): 374,792 sandboxed, 140,814 verified (v2: 140,817);
RL 20,000 rows (sha `4b5a1ec7…`), SFT pool 120,211 rows (sha `c116d127…`).
`stub_passes` rose from 16 to 1,154 — most likely references that print
(always failing under v5) paired with tests that constrain nothing; the stub
screen drops them. Mean verified wall time 31 -> 47 ms (p99 86 -> 191 ms).
v2 artifacts are staged out of `postraining/data/` (never consumed by RL).

## 2026-09-24 — Science multiple-choice teacher traces v1

Teacher generation job 9506 succeeded: Qwen3.8-27B produced two samples for
each of 10,020 questions (20,040 completions). Selection required **both**
samples to give the gold letter (`--min-correct-share 1.0`); 9,426 questions
met that rule, and 9,329 retained one screened trace. This is 93.10% of all
complete questions. Of the others, 274 had one correct sample and 320 had
none; 97 passed the two-answer rule but had no trace survive the content and
length screens. The teacher's letter accuracy is an outcome check, not a
proof that every reasoning sentence is sound.

The immutable pool is `postraining/data/science-mc-traces-v1-pool.parquet`
(sha256 `8ff03b177440f432b6a9c0be3866579df037457844d6e863169a85f485098631`),
selected from generation JSONL sha256
`f72185a0a5a40ed0f9b88b4c2ccdb5aba3514ebfc7fa23bdc0d2692689d7a2b4`.
The science-only SFT corpus is
`postraining/data/sft_science_mc_traces_v1.parquet` (9,329 verified documents;
sha256 `aa31a3b80400432893b1f4d2c3be7264164b004daad07eeeddbbcabfad136282`).
The corpus builder rechecked the pool against the complete SFT question
partition and its RL counterpart; all 9,329 pool rows survived the adapter.
They set `gradeable = False` because the SFT trainer's sampling gate uses a
math verifier that cannot grade a letter; science accuracy needs a separate
exact-letter evaluation after SFT.
Pool document lengths total 1,827,405 GPT-2 tokens (median 182, p90 270,
maximum 765). Kept letters are A/B/C/D = 2,346/2,280/2,372/2,331.

The red-team read the real kept traces and found two that explicitly called
their correct option incorrect. The answer-aware mechanical screen now rejects
that direct contradiction and selection uses the other, coherent teacher
sample for both questions. It rejected four candidate completions in all;
the question and document yields did not change. The review found 28 groups
of repeated question stems with the same answer (58 rows), mostly alternate
distractors; no exact question/choice overlap with science RL v5 or Knowledge
v5 and no confirmed answer-key or teacher-instruction leak. Correct final
letters still do not prove that every reasoning sentence is sound. One kept
trace (`6bdbca9b0a36…`) briefly rejects a fraction of a week before correctly
equating one seventh of a week with one day; the reviewer judged that local
inconsistency too small to block this corpus.

| SFT question source | Complete | Kept | Yield |
|---|---:|---:|---:|
| ARC-Challenge | 529 | 514 | 97.16% |
| ARC-Easy | 1,112 | 1,086 | 97.66% |
| OpenBookQA | 2,489 | 2,163 | 86.90% |
| SciQ | 5,890 | 5,566 | 94.50% |

The Knowledge rows are excluded from this science corpus. No Knowledge v6
corpus was built; `sft_ultradata_knowledge_v5.parquet` remains the reference
against which the science question split was built (sha256
`f55c56adabd969f0449f5820e54a40ff76184ac81dba4224298330fa729b3967`).
At that point, no new multiple-choice GPU job was authorized; the combined
run was authorized in the next exchange.

### Combined 1,024-token SFT, authorized 2026-09-24

The user authorized the combined SFT at a 1,024-token window. CPU build job
9525 produced `postraining/data/sft_mix_omi2_drills_code30k_science_v1.parquet`
(sha256 `56a069ff217d6abe764c26e59ff0d3fcd33dc05759c91f3873341feb1aeaad4b`):
eight OpenMathInstruct-2 shards to match job 9040, 500,000 arithmetic drills,
the first 30,000 verified code documents that pass the 1,024-token screen,
and all 9,329 verified science traces. The extra code/science benchmark
screens leave 430,611 OMI2 rows, 446 fewer than the 9040-only build. The
combined corpus has 969,940 documents and 316,405,222 document tokens:

| Source | Documents | Document tokens | Token share |
|---|---:|---:|---:|
| OpenMathInstruct-2 | 430,611 | 209,807,664 | 66.31% |
| Arithmetic drills | 500,000 | 83,038,228 | 26.24% |
| UltraData-Code L3 | 30,000 | 21,731,925 | 6.87% |
| Science MC traces | 9,329 | 1,827,405 | 0.58% |

Science and code rows train and enter holdout CE, but have `gradeable = False`
for the trainer's math-only sampling gate. The 256-problem held-out split has
248 math-gradeable panel rows; two of the first 128 prompts exceed 256 tokens
(maximum 265). Job 9529 uses a 272-token prompt plus 752-token response gate,
within the pretrained 1,024-token window. It starts from the pretrained KDA8
checkpoint, takes one epoch (12,470 scheduled updates, 32 packed 1,024-token
rows per update), and otherwise follows job 9040's SFT recipe. A separate
science exact-letter gate against RL v5 is queued for the new checkpoint (job
9530) and job 9040 (9531), with the same rollout settings, followed by the
matched arithmetic probe (9532). No Knowledge rows are in the combined SFT.

#### Combined SFT outcome

Job 9529 completed one epoch and saved
`postraining/runs/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt`
(sha256 `c882966296b9109a828af67cc2ad4a3851a1233411727bb0ec0479b2a04b2742`).
It ran 12,470 updates in 1,767.5 training-clock seconds. Its own holdout
completion CE was 0.5271; the holdout differs from job 9040's and those CE
values are not a matched comparison. The math-only SFT sampling gate scored
25.39% exact accuracy on 128 prompts × 8 samples, with 34.38% mixed prompts.

The matched frozen science RL-v5 rollout gates used 64 prompts × 16 samples ×
8 repeats and a 256-token prompt plus 768-token continuation budget:

| Checkpoint | Job | Exact science accuracy | Mean within-group reward std | Think format | Ended |
|---|---:|---:|---:|---:|---:|
| Combined SFT | 9530 | 14.60% | 0.3085 | 78.06% | 81.08% |
| Job 9040 SFT | 9531 | 0.085% | 0.0033 | 21.22% | 39.59% |

Job 9530 passed the rollout learnability gate. Job 9531 exited 2 because its
gate failed. **Scope correction (2026-09-24):** the sampler takes file order in
its first cycle, and science RL v5 is grouped by source. All 8 × 64 prompts
above are distinct ARC-Challenge questions (the first 512 rows), not a
science-wide sample. The logged 25.56% modal-answer baseline describes all
9,743 RL questions and is not a matched comparator. The 512 ARC-Challenge
questions have a 26.56% modal-label share, but a constant-letter answer would
not satisfy the rollout's required think/answer structure; the 14.60% figure
is a strict reward rate, not a measure of raw answer accuracy. Thus this gate
establishes a strong improvement over 9040 on ARC-Challenge and usable reward
variation there. It does not yet establish whether the checkpoint is strong or
weak across SciQ, OpenBookQA, and ARC-Easy; use a source-stratified evaluation
with saved responses before deciding on science RL.

The matched greedy arithmetic probe (`data/math_drills/v4/probe.jsonl`,
256-token prompt, 512-token continuation) scored 972/1,920 = 50.63% for
combined SFT (job 9532) versus 991/1,920 = 51.61% for job 9040. Keep 9040
as the math RL reference. This is a 19-question regression on this panel;
other ability changes require matched evaluation before claiming them.

#### Source-balanced science diagnostic (job 9629)

The user authorized one further multiple-choice GPU diagnostic after the
source-order limitation above was found. A deterministic 256-question panel,
`postraining/data/science-mc-rl-v5-balanced-diagnostic-v1.parquet` (sha256
`a66de30f3a5915faa6d276462612b0e21a7f3566036fee4e10edecbe7779d46d`),
hash-selects 64 disjoint RL-v5 questions per source and interleaves the four
sources. The largest stored prompt is 189 tokens, below the 256-token budget.
Job 9629 evaluated the combined SFT checkpoint with eight samples per prompt,
768 continuation tokens, temperature 1.0, and the benchmark evaluator's
top-p 0.7. Its answer check is raw exact-letter accuracy after termination,
with a fallback for responses without an answer fence. This differs from the
strict rollout gate (top-p 1.0 plus think/answer structure requirement).

| Source | Correct / 512 samples | Raw exact accuracy | Prompts with any correct / 64 |
|---|---:|---:|---:|
| ARC-Challenge | 97 | 18.95% | 44 |
| ARC-Easy | 99 | 19.34% | 50 |
| OpenBookQA | 100 | 19.53% | 51 |
| SciQ | 100 | 19.53% | 50 |
| **Balanced total** | **396 / 2,048** | **19.34%** | **195 / 256** |

The run ended 85.25% of completions. Its benchmark report's
`structural_format_fraction = 1.0` is not an assessment of think/answer
structure: the job-9629 evaluator call omitted think-fence IDs, so that field
was always 1.0. The AIME and benchmark call sites now pass think-fence IDs
and the configured think-token minimum; evaluation metric schema v7 separates
future reports from this artifact. Raw accuracy is unaffected. Only the first
four prompts × four samples are saved
in `postraining/runs/kda8_science_balanced_probe_v1/bench_answers/step_000000.json`.
Their text is often incoherent and imports math reasoning into science. One
captured response with gold `D` wrote `<answer>\\text{D}</answer>` and was
marked wrong by exact-letter grading; this shows some formatting undercount,
but the 16 saved examples cannot quantify its rate across all 2,048 samples.
Independent inspection found that 15/16 captured responses satisfy the
33-token think/answer structure, and the three exact-correct captured
responses also satisfy it. This small sample does not suggest a large
raw-versus-contract gap; the full-panel gap remains unmeasured.
The matched 256-question panel's modal-letter share is 27.34%, though a
constant letter that does not terminate or meet the strict answer contract is
not a valid rollout-policy baseline. The diagnostic supports weak sampled
science MC answer performance on all four sources, while the earlier 14.60%
number specifically measures strict RL reward on ARC-Challenge.

## 2026-09-25 — KDA8 VAPO step throughput: fused decode, per-rollout arena workspace, fused readout

Scope: the production cot pool (64 prompts x 16 samples = 1,024 rows, prompt
<= 256, 768 generated tokens, `kda8_sft_omi2_drills_e1` actor, v12 exact
mixture). The code does not change what a step computes; replay numerics
change only by fp32/bf16 rounding (schemas bumped, below). Every measurement
is an mlq job on the shared RTX 5090 with an 8-step `--profile` run (pools
4-7 steady) or a 6-repeat `--rollout-only` run. GPU condition varies by up to
~10% between jobs: back-to-back A/B decode benches of identical tick code
differed by 5%, so single-pool differences below ~0.3 s are not evidence.

### Result

| Measure | Baseline (9992 profile, rollout-only) | Now (v9 profile 10086) |
|---|---:|---:|
| Steady pool wall | 6.0-6.6 s | 4.3-4.7 s |
| Decode | 3.3-3.7 s | 1.8-2.1 s |
| Update (forward_backward) | 2.4-3.0 s | 2.2-2.8 s |
| Useful tok/s (rollout-only) | 84-101k | 146-155k (v6) |
| First rollout (compile + capture) | 56 s | 19 s |
| Update peak VRAM | 10.4-11.1 GiB | 7.1-7.2 GiB |
| Step-1 update peak (drift diagnostic) | 16.5 GiB | 6.7 GiB |
| Rollout peak VRAM | 6.4-7.1 GiB | 8.2 GiB |

### Decode: what bound it and what changed

The per-rollout loop was already one compiled, CUDA-graphed tick per step
after the arena work (graph per live-row bucket), but it gave only ~8%:
decode is bandwidth-bound, not launch-bound. A 1024-row tick moves about
2.4 GB of fp32 KDA state (6 layers x [1024, 3, 128, 128]) plus live
attention KV; the floor is ~3 ms against ~7-10.5 ms measured.

- **KDA state update:** the compiled pure-PyTorch recurrence made several
  passes over the state per layer, and its two einsums ran as bf16 cutlass
  bmm under autocast — the documented fp32 recurrence was contracting in
  bf16 (a numerics bug; CPU test error 0.05 before the fix). Now one Triton
  kernel (`pretraining/nanogpt_mini/kda_decode_kernel.py`, a `triton_op`
  inside the compiled tick) reads and writes the state once per step, all in
  fp32; `kda_recurrent_step` is the fp32 reference with autocast disabled.
- **Decode attention:** prompts are short (mean 35.6 tokens, median 17, p90
  92) but a pool is left-padded to ~190, so most prompt KV reads were
  padding. `postraining/decode_attention.py` (split-K flash-decoding over
  each row's live key range, `triton_op`) reads only live keys.
- 1024-row tick at positions 256/512/1022: 7.1/8.0/10.5 ms -> 3.97/4.82/6.21
  ms (v5 bench 10019). Full-budget rollout (no stops): 8.0 s -> 4.4-4.9 s.
- Parity (`kda_gpu_parity`, v5/v6/v8): all decode, arena, race and carry
  bounds pass. The only failures are MoE section 5 (`moe_bf16_decode_vs_dense`
  0.51 > 0.5, `moe_bf16_compiled_paged_vs_eager` 0.60 > 0.02), which fail
  identically on the untouched baseline tree (job 9962).

### Arena memory and a capture bug

Holding the arena's full-width caches (5.21 GiB at 1024 rows) for the run
kept them resident through the update and OOM'd the step-1 drift diagnostic
(job 10021). `PinnedDecodeArena.workspace()` now allocates caches, belief
and hidden buffers and captures the bucket graphs per rollout, then frees
them: re-opening costs 35-37 ms (memset + 17 captures; the side-stream
warm-up runs once), and closing returns the allocator to +0 MiB. Separately,
the compiled tick guards on grad and autocast state, so a capture opened
under a different autocast than warm-up recompiled inside the capture
(`cudaErrorStreamCaptureInvalidated`, job 10029). `_call_tick` now pins
no-grad and autocast-off around every tick; the CPU regression test fails
without it.

### Update: where its time goes

Kernel trace of a steady update (v6): 2.39 s device time — GEMM 0.80, FLA
KDA 0.50, other Inductor 0.45, other aten 0.25, log-softmax 0.23, conv1d
0.15. Replay packing is already good: new `replay_slot_utilization`
telemetry (real tokens / computed shard slots) reads 0.87-0.90 over 21
shards, far above what `packed_padding_utilization` (~0.4) suggests, so
varlen repacking would buy
at most ~12% of replay compute and was not pursued. A 2x slot budget
(`--replay-slot-budget 49152 --replay-max-trajectories 256`, job 10087) only
cuts shards 21 -> 18 because the attention budget binds for long rows,
lowers utilization to 0.85, raises the update peak to 11.7 GiB, and shows
no measurable update-time change. Replay attention is flash SDPA (memory
linear in B*L), so the B*L^2 budget does not bind memory for this backbone;
lifting it (`--replay-attention-budget 1073741824 --replay-max-trajectories
1024`) with 2x / 3x slot budgets (jobs 10093 / 10094) cuts shards 21 -> 10 ->
7 and kernel launches 211k -> 172k per pool, but device time per pool stays
3.85-4.06 s and the update 2.2-3.0 s in all three: slot utilization falls
0.88 -> 0.83 -> 0.80, so the extra padded compute cancels the per-shard
overhead saved, while the update peak rises 7.1 -> 12.8 -> 18.1 GiB. The
update is device-bound (GEMM + FLA), not shard-overhead-bound; the defaults
stay.

### Fused softcapped readout

The emit tail (readout GEMM -> rational softcap -> fp32 log-softmax ->
target gather) was the update's largest memory-bound piece: its compiled
backward was zeros + scatter_add + log-softmax backward + a separate bias
sum over fp32 (slots, 50304) tensors, ~0.27 s per pool, and it set the
update's peak memory. `SoftcappedTargetLogprobs` (`nano_backbone.py`,
exposed as `target_logprobs_from_features`) saves only the bf16 GEMM output
and a per-slot logsumexp; its backward is one elementwise pass writing the
bf16 raw-logit gradient. It serves VAPO scoring
(`compact_emit_token_logprobs`), SFT's CE and `TrunkRenderReadout`.

- Readout bench (`postraining/bench_emit_readout.py`, job 10075, fwd+bwd
  compiled): 4,096 / 16,384 / 24,576 slots take 7.2 / 27.5 / 43.4 ms
  composed vs 5.5 / 22.2 / 31.4 ms fused; peak 1.09 / 4.51 / 6.79 GiB vs
  0.33 / 1.46 / 2.21 GiB; values agree to 2e-6. At 24,576 slots the rough
  floor (three 1.27-TFLOP GEMMs plus ~6 passes over the bf16 logits) is
  ~27 ms.
- In the trainer the readout kernels fell from ~0.27 s to ~0.13-0.15 s per
  pool and the update peak from ~11 GiB to ~7.1 GiB. The forward logsumexp
  is a two-pass reduction (0.04-0.06 s) where the old log-softmax used
  Inductor's online softmax (0.025 s); not worth an indirect formulation.
- Compiled, Inductor fuses the bias-gradient sum into the backward without
  emulating the bf16 rounding of the per-row gradient, so the compiled bias
  gradient is closer to fp64 than eager's; the CUDA test pins "no less exact
  than the eager composed readout", not bit equality.
- The post-update drift diagnostic (step 1 and every 100 steps) ran the tail
  eagerly and was the run's peak-memory step (17-19 GiB; its 4.97 GiB fp32
  temporary OOM'd the 2x-budget run, job 10077). It now uses its own
  compiled no-grad artifact: step-1 update peak 6.7 GiB, drift 1.6 -> 1.1 s.
- `REPLAY_NUMERICS_SCHEMA` (v5) and `CARRY_REPLAY_NUMERICS_SCHEMA` (v6) are
  bumped: behavior log-probs round differently, so VAPO resume across this
  change is refused. SFT resume is refused through its module hashes.

### Other

- `RunProfiler` created its output directory in `__init__`, so every fresh
  `--profile` run refused its own nonempty output; it now creates it lazily.
- An ~8.5 GB GPU allocation from another tenant persists on the device and
  is not listed by nvidia-smi; account for it when sizing budgets.
- Remaining decode opportunities: small-bucket ticks (8-128 rows) are
  ~0.4-1.0 ms, dominated by per-kernel latency across ~8 layers; the update
  remains GEMM + FLA bound after this.

## 2026-09-26 — STEM/code/QA readiness audit and missing frozen-policy gates

The combined SFT is **finished**, not awaiting a restart:
`postraining/runs/kda8_sft_omi2_drills_code30k_science_v1_e1/result.json`
records 12,470 updates, completion CE 0.527120, and a 25.39% math-only
sampling gate. Its corpus contains 30,000 code documents and 9,329 science
traces, but no UltraData Knowledge SFT documents. The science traces were
only 0.58% of document tokens. Completed SFT is not evidence that all domains
are RL-ready: the source-balanced science diagnostic measured 19.34% raw
exact-letter accuracy, and the earlier passing strict gate covered only
ARC-Challenge. A rollout gate's exit status checks reward variation, not
above-chance accuracy.

Existing effective RL pools: math 99,917 (71,744 DeepMind easy + 19,765
UltraData math + 8,408 DAPO), code 20,000, science 9,743, Knowledge 2,337.
Both the existing math and combined-SFT science manifests load under the
current corpus contracts. Knowledge here is verifiable multiple-choice and
numeric QA, not an open-ended QA corpus.

Queued through mlq, priority 1, max parallel runs 1, one attempt each:

- 10175: `scripts.prepare_posttraining_gates`, 20-minute limit. Produces
  immutable panels under `postraining/data/readiness_20260926_v1/`, bound to
  the completed combined SFT's exact corpus bytes.
- 10176 / 10177 / 10178: code / science / Knowledge frozen-policy gates,
  respectively; depend on 10175 success, one-hour limit each. Each covers
  512 distinct prompts x 16 samples, 256 prompt + 768 continuation tokens,
  production CoT think/answer contract, seed 1337. Panels hash-select and
  interleave strata to avoid the old file-order sampling error. Science
  balances datasets; Knowledge balances modules until a small module is
  exhausted, so its aggregate must not be called population accuracy.
- 10179: immutable full math+code+science candidate manifest,
  `postraining/data/vapo_combined_candidate_20260926.manifest.json`, bound
  to combined SFT. Twenty-minute limit. A candidate manifest does not
  authorize or qualify a training mixture. Knowledge is excluded pending
  its independent gate.
- 10180: after all three gates terminate, summarize full evidence into
  `postraining/runs/readiness_20260926_v1/result.json`; ten-minute limit.
  Reports per-module contract accuracy, format, termination, mixed groups,
  and terminated parsed-letter accuracy/chance for choice questions. Missing
  or repeated prompt/sample coverage fails rather than producing a success.

`--gate-transcripts` now optionally saves **every** frozen gate response,
including failures, token IDs, prompt/module, grading contract and verdict,
to each run's `gate_transcripts.jsonl`. It leaves ordinary training and
throughput probes unchanged. Existing transcript files are refused. Future
analysis can diagnose code verifier failures and formatting loss without
re-running the model. No optimizer updates or further SFT were submitted;
the remaining training decision depends on these results. At submission,
an unrelated running GPU job occupied the queue; it was not interrupted.

Validation: 47 focused panel/capture/summary/mixture/reasoning tests and all
22 benchmark-isolation tests passed. An independent reviewer checked panel
selection, CoT budgets, thread-safe transcript coverage and queue dependencies.

## 2026-09-26 — Corpus quality review and completed six-source SFT evaluation

All six frozen-policy evaluations completed their full 512 distinct prompts x
16 samples (49,152 completions total). They used the completed combined SFT,
temperature 1, top-p 1, 256 prompt + 768 continuation tokens, production
think/answer grading, and no optimizer updates. These are stratified
training-pool learnability diagnostics, not held-out benchmark estimates.

| Source | Correct / 8,192 | Strict accuracy | Mixed prompts / 512 |
|---|---:|---:|---:|
| DeepMind easy | 158 | 1.93% | 98 |
| UltraData math | 135 | 1.65% | 75 |
| DAPO | 95 | 1.16% | 60 |
| Code | 1 | 0.012% | 1 |
| Science MC | 1,346 | 16.43% | 475 |
| Knowledge/QA | 140 | 1.71% | 93 |

Canonical evidence: `postraining/runs/corpus_quality_20260926/final_result.json`
and `model_gate_summary_v2.json`, with complete transcripts in each
`kda8_readiness_<source>_20260926/` run. The first full summary (job 10185)
failed on math rows without module metadata; the source-name fallback was
fixed, regression-tested, and the read-only summarizer rerun successfully.
An exit 2 from a completed frozen gate indicates insufficient reward
variation in at least one batch, not a crashed or incomplete evaluation.

Semantic review found 11 primary flags among 64 deterministic code tasks
(contract/test contradictions, invalid-input requirements or overly strict
outputs); at least three ambiguous/nonunique keys among 16 Knowledge rows;
and one confirmed wrong science key among 32 science questions. Evidence,
row IDs, reviewer samples and proposed quarantines are saved alongside the
result. These are sampled findings, not estimates that certify every other
row. Original immutable corpora were not edited. The 30,000 actual trained
code problems map to the current SFT sibling pool and have zero entry-point
identity overlap with RL. Twelve reviewed science SFT traces had no confirmed
serious defect. Math review checked 12 questions/source, substantiating all
DeepMind targets, 11 DAPO targets and seven UltraData targets; unproved and
ambiguous examples are explicitly recorded.

Code audit job 10184 tested ten input-blind trivial programs on 128 sampled
tasks through the production sandbox: all 1,280 failed. Thirty-one of 20,000
code rows contain a provably vacuous assertion, although other assertions
can still constrain those tasks. Passing a reference and rejecting trivial
programs do not prove agreement with the task's prose.

Canonical prompt-budget audit 10186 found all sources fit except 16 Knowledge
rows at 257 tokens including BOS. `choice_rl_pool.screen_problem` omitted
BOS; it now uses `encode_prompt`, with an exact-boundary regression test.
Existing corpora remain unchanged and must be rebuilt into new outputs to
apply that fix. Two affected prompts were in the Knowledge gate; excluding
them changes its strict accuracy only from 1.709% to 1.716%. Excluding the
one reviewed flagged code task in the code panel does not change the verdict.
The initial census counted raw math source wrappers; its mutation results
are valid, but its budget section is superseded by `canonical_census.json`.

Readiness interpretation:

- Code: 4,879 unterminated, 2,458 terminated but invalid structure, 673 policy
  rejections, 181 test failures, one pass. At least 625 policy rejections
  contain malformed Python; this is not just an import-policy problem.
- Science: valid-option accuracy is 22.44–27.15% across sources. Prompt-cluster
  label-permutation tests found no compelling above-chance signal; all 16
  sampled explanations (including eight rewarded) were incoherent. Reward
  variance alone would admit a guessing policy.
- Knowledge: ten-option questions emit a valid letter in only 606/3,216
  responses, and 94.2% of those letters are A–D. The combined SFT contained
  no Knowledge data. Dataset ambiguity independently blocks promotion.
- Math: some genuine workings exist, but sampled successes often have invalid
  reasoning. 99/135 UltraData successes target 0 or 1. Constant-answer gold
  prevalence is 4.49% for DeepMind answer 1 and 8.01% for UltraData answer 0;
  these are answer-frequency controls, not measured model policies.

Do not launch broad RL from this evidence. The user's proposed curriculum
should gate hard sources on improved held-out easy-task performance and a
new source-specific probe, not on reward variance or a source name alone.
There are no calibrated item-difficulty labels: DeepMind has 18 modules,
science has ARC-Easy/Challenge/source labels, and code/UltraData/DAPO have no
useful calibrated item labels. DeepMind's place-value, GCD and multi-add/sub
modules have the strongest sampled mixed-group rates, but even those need
answer-frequency controls before being treated as mastery. Keep DAPO/code/
Knowledge locked; establish coherent elementary capability before promotion.
Science's high mixed-group rate is not sufficient to unlock it.

Validation for this turn: 102 choice/budget tests and six focused
panel/capture/summary/audit tests passed; `git diff --check` clean. No further
SFT or RL optimizer run was launched.


### Saved-response attribution and MC reward interpretation (2026-09-26)

Queued offline attribution jobs 10187 and 10188 completed. Evidence lives in
`postraining/runs/corpus_quality_20260926/response_format_attribution.json`
and `code_format_regrade.json`. Each source contains 8192 saved responses.
Truncated / terminated-invalid-format counts: DeepMind 767/508; UltraData
math 3221/200; DAPO 2358/339; science 1303/228; Knowledge 3568/496; code
4879/2458. Every truncated response emitted the full 768-token allowance.
Gold-independent extraction plus wrapper normalization changes correct counts
158->170, 135->140, 95->102, 1346->1368, and 140->163 respectively. This is
counterfactual recovery, not validated deployment accuracy: provisional
answers can be selected, and oversized fields were left ungraded. Of 612
recovered AST-valid code candidates, 602 failed tests and 10 were rejected by
sandbox policy; none passed. No strict correct answer disagreed with selected
raw grading. Focused normalization/first-answer-selection checks passed.
These results establish little recoverable correctness in existing text;
they do not measure the causal benefit of a larger generation budget.

MC recommendation: retain correctness rewards and the learned critic baseline;
do not treat subtracting 1/K as a way to identify guesses. An action-independent
policy-gradient baseline reduces variance without changing the expected
objective (OpenAI Spinning Up, rl_intro3). Constant subtraction cancels exactly
under group mean centering; our VAPO/GAE stack is not interchangeable with GRPO,
so no terminal-reward offset was patched. Use above-chance held-out accuracy
with answer-position controls for curriculum admission. Current option
permutations are fixed at corpus construction. Fresh option-order randomization
and a correctness-plus-permutation-consistency diagnostic are useful next
experiments; consistency alone does not prove reasoning. ACRE
(https://arxiv.org/abs/2510.10104) studies this idea in larger multimodal models,
not this checkpoint. Pure independent four-way guessing produces mixed groups
on about 99% of questions with 16 samples, so group reward variance is not a
science readiness criterion. No new model generation, SFT, RL, or reward change
was performed for this follow-up.


### Long-budget follow-up and MC controls (2026-09-26)

Recent KDA8 CoT RL v7/v9/v10/v11/v12 used 1,024 total tokens: 256 prompt +
768 response. The launcher selected the older 1,024-token checkpoint for RL,
separately from its 5,120-token SFT stage. Older latent runs had larger context
windows but separate emitted-token and latent-stream caps. See
`corpus_quality_20260926/historical_rollout_budgets.json` for exact manifests.

The current code/science SFT checkpoint trained at 1,024 tokens. All six frozen
panels now completed 512 questions x 16 samples at 4,096 response tokens plus
256 prompt tokens (jobs 10189-10194). This explicitly extrapolates that model's
trained window. The new `--eval-context-tokens` override is restricted to frozen
evaluation; checkpoint metadata and training limits are preserved. Manifests
record actual SFT training length separately from the checkpoint context limit.

Source | Accuracy at 768 | Accuracy at 4096 | Capped at 4096 | Late correct
--- | ---: | ---: | ---: | ---:
deepmind_easy | 1.929% | 1.965% | 1.72% | 10
ultradata_math | 1.648% | 3.027% | 2.37% | 90
dapo | 1.160% | 1.660% | 1.89% | 32
ultradata_code_l3 | 0.012% | 0.012% | 3.70% | 0
science_mc | 16.431% | 16.565% | 1.48% | 6
ultradata_knowledge | 1.709% | 2.014% | 1.87% | 21

Late correct means this successful saved trajectory ended beyond 768 tokens;
it does not establish failure of an independently generated shorter response.
The comparison records realized prefix equality and prompt-cluster uncertainty.
Independent hash-selected review found no coherent derivation in 12 sampled
late-correct UDmath/DAPO responses. Of all late-correct UDmath responses, 72/90
had literal gold 0 or 1. Score gains alone do not establish reasoning mastery.

The older 5,120-token SFT lineage was also tested with 4,096 response tokens.
It has a different corpus, so this is checkpoint selection, not an isolated
training-context ablation. Accuracies: DeepMind 2.136%, UDmath 1.221%, DAPO
0.745%, code 0%. It does not establish a stronger broad RL starting point.
An initial UDmath attempt ran out of memory during cache compaction; its partial
evidence is excluded. Recovery jobs 10207-10209 use 32 concurrent prompt groups,
retaining all 512 questions x 16 samples. Summary job 10210 completed. Code
variance-gate exits were nonzero but all required transcript coverage completed.

Fresh option shuffling is now the training default for screened MC rows, seeded
by source-qualified identity and sampler cursor. Gold and source-option IDs
remap immutably. Dataset identity binds the presentation policy; legacy resumes
use `--no-randomize-choice-options` to preserve their previous policy. Frozen
gates stay fixed unless explicitly enabled. Correctness rewards and the critic
are unchanged. Shuffled science (10195) scored 1,271/8,192 versus 1,357 fixed;
four-option valid-letter accuracy ranged from 22.0% to 24.9%, without positive
above-chance evidence. Shared sampling seeds mean independent-guess 1/K^2 is
not the null for paired joint correctness.

Long code format recovery (10204): 988 AST-valid excluded candidates yielded
980 test failures, six policy rejections, and two passes. Both extra passes
ended before token 768. Strict plus recovered accuracy is only 3/8,192. The
older lineage's recovery (10212) yielded zero passes from 730 candidates.

The immutable reviewed candidate contains 131,292 rows across all six sources,
excluding 16 reviewed defects and 16 canonical over-budget prompts. It reserves
673 science question-family rows around the 512-question validation panel.
No question/component overlap was detected against the actual 969,940-row SFT
corpus. This panel was already used for development, and lexical screens do not
certify absence of semantic paraphrases or pretraining exposure. Candidate prep
10206 completed after correcting its input to explicitly include all six sources;
original pools and launcher defaults were not changed. Candidate path:
`postraining/data/reviewed_candidate_20260926_v1/mixture.manifest.json`.

Broad RL is not promoted by these results. No optimizer updates were run.
Validation: 44 focused tests passed, conservative code-extraction checks passed,
and independent review covered MC remapping/resume identity, budget attribution,
and corpus exclusions. Full evidence and final decision live under
`postraining/runs/corpus_quality_20260926/`, including `followup_result.json`.


## RL response defaults and math overlap review (2026-09-26)

Future `train_latent_vapo.py` runs now resolve the response cap to 2048 tokens;
benchmark/AIME caps inherit it unless overridden. The KDA launcher explicitly
uses 256 prompt + 2048 response tokens, with runtime context 2304. The actual
checkpoint training window is preserved and extrapolation is recorded in the
manifest. Latent modes reserve additional stream capacity; answer-only mode
keeps its answer budget. Exact resume also checks saved effective emitted and
stream budgets. Default training is 950 joint updates + 50 critic warmup =
1000 total. Historical explicit token budgets remain usable for exact resumes.
The 2048 training configuration has not been benchmarked; old launcher memory
and throughput measurements were at 768 tokens.

The existing DAPO builder removed 5,268 rows against raw UltraData candidates,
but broader retrieval still found residual paraphrase overlap. The first review
covered 148 pairs (top 110 cosine, all 42 numeric skeleton and all 6 production
matcher hits). Independent full-text adjudication of 49 proposed exclusions
confirmed 47 duplicates and 1 underspecified DAPO rewrite; one ambiguous decimal
variant was retained. Exact integer enumeration independently supports 90 for
the double-dipped recurrence problem, contradicting DAPO's 48. Existing matcher
failures include short paraphrases below its six-eight-token-shingle threshold
and raw numeric spelling guards (percent/decimal, fraction markup). All 6 current
production shingle hits were false positives. Do not lower global thresholds
or delete similarity candidates without adjudication.

Evidence lives in `postraining/runs/corpus_quality_20260926/`:
`math_source_overlap_review_v2.json` reconciles the initial review;
`math_source_overlap_independent_review.json` records independent adjudication.
This audit concerns cross-source RL overlap, not SFT exposure or evaluation
contamination. Confirmed overlaps are a lower bound.

Extended review through cosine rank 219 increased coverage to 256 distinct pairs.
After independent adjudication, the cleanup removes 65 unique duplicate DAPO
questions (66 cross-source pairs), two underspecified DAPO rewrites, and one
invalid UltraData probability question (Math_04930). This is 68 additional rows
relative to candidate v1. See `math_source_overlap_final.json` for consolidated
counts. UltraData Math_04163 is retained after deduplication, but its 1/8 key has
not received a complete global-minimum proof; duplicate identity is confirmed.
No global overlap thresholds or canonical source pools were changed.
Validation: 125 focused tests pass, launcher shell syntax and diff whitespace
checks pass. No new model training or evaluation was launched for these edits.

Candidate build mlq10217 succeeded. `postraining/data/reviewed_candidate_20260926_v2/mixture.manifest.json` has 131,224 rows: DAPO8,340 and UltraData math19,764; other source counts unchanged. The actual SFT/science holdout audit still passes its recorded matching rules. This is a prepared candidate, not an activated training mixture.


## Partial code reward RL launch (2026-09-26)

User authorized training with passed/total test feedback despite low all-tests
accuracy. Added `--python-reward-mode test-fraction` (default); binary remains
explicitly selectable. Reward schema `bwrap_python_isolated_test_fraction/v1`
is distinct from the unchanged v6 corpus verifier contract. Ordered setup
statements are grouped with the next assertion-bearing case; failed cases do
not skip later cases. Initialization, timeout, syntax/policy, and integrity
failures receive zero. Harness integrity is checked after each original test
statement, including grouped fixtures. Full-suite accuracy stays separate from
partial reward. Resume and actor/critic initialization reject objective changes.

80 focused tests pass. Final saved-rollout audit mlq10223 succeeded: among
8192 code responses at the previous 4096 cap, 33 earn partial credit and 1
passes all tests; 24/512 questions have mixed fractional rewards versus 1/512
under binary rewards. Mean test fraction is 0.00087246. Constant-return
controls can earn substantial partial credit; do not interpret average test
fraction alone as solved-task accuracy. See `code_partial_reward_final_audit.json`
and `code_partial_reward_semantic_review.json` in corpus_quality_20260926.

Launched mlq10224, `kda8_rl_reviewed_v2_partial_2048_20260926_r2`, via
`scripts/launch_reviewed_rl.sh`: reviewed candidate v2 (all six sources),
math/code/science SFT checkpoint, 256 prompt + 2048 response, 64 prompts x16
samples per update in 32-group waves, 50 critic warmup + 950 joint updates,
held-out math/BPB every20, checkpoints every300seconds, choice permutations,
priority1/maxparallel1/time limit12h. Sustained32all-zero pools stop the run;
no short-plateau cutoff. Initial10222 was cancelled before optimizer updates
because the final verifier change landed during startup;10224 starts fresh
with the final-harness audit prerequisite. No binary critic state is reused.


Final launch supersedes the startup records above: **mlq10228**, run
`kda8_rl_reviewed_v2_partial_2048_20260926_r3`, queued automatically after
final scorer-v2 replay audit10227. Schema is now
`bwrap_python_isolated_test_fraction/v2`, scorer SHA256
`8c580aeb8e52773a1345ed2a8a3d1044a0cfdaba5d0ee29c5ba1271d14cdfe7e`.
Uncalled helper definitions no longer earn credit; invoked assertion-bearing
helpers count as cases. All19,989code rows statically group without errors;
all11helper-definition rows independently reviewed;84focused tests passed.
10224 was also cancelled before optimizer updates; the shared queue now owns
final audit/training startup. Configuration and1000total-update budget remain
as above. `corpus_quality_20260926/partial_rl_launch.json` is the launch record.


### GPU-idle stall fixed (2026-09-26)

User reported idleGPU on10228. Nine critic warmup updates completed at roughly
11–20s/update afterinitialcompilation; warmup10 then spent minutes on oneCPU
core withnoGPUtraining. SIGINT traceback proved the stall was
`parse_numeric_answer -> Fraction -> numerator *= 10**exp`: a generated
scientific exponent requested an enormous exact integer. Interrupted before
anyactorupdates; no checkpoint existed yet. This was a reward-parser resource
bug, not insufficientGPUcapacity or lack ofreward signal.

Bounded scientific exponent magnitude to4096 before everyFraction conversion,
including numerator/denominator forms; existing128character input limit stays.
No gold answer among131224reviewedrows exceeds this bound.160focused tests
passed, with independentreview. Reward schema bumpedv5. SIGUSR1 now dumpslive
Pythonstacks withoutptrace and warmup timings printto stdout. SamefullRLrun
requeuedas10242 / `kda8_rl_reviewed_v2_partial_2048_20260926_r4`, pri1,max1,12h;
the9criticwarmupupdates repeat because no recoverycheckpoint wasavailable.


## Standard PG default; SFT actor with retained critic (2026-09-26)

User requested standardPG insteadofDG and explicitly resetactor toSFT while
retainingcritic. ConfigdefaultDGFalse, launcherspin--no-delightful-policy-gradient.
Added --critic-only-init withsource/reward/corpus/fence/budget/schema guards;
onlycriticweightsload, actorremainsSFT, optimizers/RNG/samplerfresh. Critictransfer
provenance persists through resumes.130focused tests pass; actualdonor passes
compatibilitycheck. Independentimplementation review complete beforelaunch.

DGjob10242cancelled atcompletedactorstep420. Latestcheckpoint containsstep397
critic plus50warmupupdates; actorweightsfromit areignored. PGjob10265queued as
`kda8_pg_reviewed_v2_partial_2048_20260926`, 530PGupdates, nowarmup, same2048
responses/partialcoderewards/all6sources. Budget50+420+530=1000. Initial estimate
553usedsavedstep397; correctedto530usingactualcompletedstep420 beforestart
(queued10264cancelledwithoutstarting). Queuepriority1,maxparallel1,12h limit.
See corpus_quality_20260926/pg_critic_transfer_launch.json.

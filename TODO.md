# TODO

## Posttraining wall-clock (measured Jul 19, v6 run TB wall times)
67 min through step ~1760: BPB guard 28.4 min (42%, 22 full-val runs at ~77 s,
value flat 1.402x throughout), collect 20.9 min (median 16.5 s/iteration,
p90 52 s — heavy tail unexplained, now instrumented), value warmup 7.8 min
(one-time), updates 5.7 min, bench only ~4 min. Fixes: `--bpb-every` default
80 -> 320 (guard cadence, not an optimization target), optional
`--bpb-val-tokens` cap, and first-class timing (`perf/collect_seconds`,
`perf/bpb_eval_seconds`, `perf/bench_eval_seconds`, `perf/aime_eval_seconds`,
`perf/checkpoint_seconds` + `seconds` in the JSONL records). Next candidates,
decide from the new telemetry: checkpoint save cost (593 MB every 32 steps)
and the collect p90 tail (suspect stray inductor recompiles or
non-terminating rollouts).

## Reward grading v2 (user-approved Jul 19)
Reward/eval verdict review: the Minerva matcher is verl `math_dapo`-standard for
DAPO/AIME data, but on mathematics_dataset rows its rewrites only widen the match
(leading-"a " strip collapses letter candidates, `split("=")` grabs the RHS,
first-$...$-pair extraction, comma-digit concatenation) and several modules have
small enough answer spaces that per-module accuracy must be read against each
module's modal-answer share, not zero. Changes (REWARD_SCHEMA
`terminated_bos_or_eos_style_by_row/v2`; resume from v1 checkpoints is refused):
grading style now follows each row's `reward_model.style` — `rule` (mathematics
_dataset) = official exact string match over the whole emission, lighteval =
Minerva unchanged; AIME evals additionally require an integer in [0, 999]; bench
logs per-module accuracy + data-derived modal-share baselines (`bench_module/`,
`bench_module_baseline/` in TB); `--exclude-modules` can drop RL modules (default
keeps all — mix changes need ablation evidence, and guessable ≠ unlearnable:
comparison modules are real easy math whose progress is now measurable against
their baselines). DAPO-ability probe on the v6 checkpoint queued as job 180
(sample_latent --math-rows 32 --samples 16 on dapo-math-17k; history: job 129
gate 0/128, job 131 probe 0/512+ pre-RL).

## Latent RL — attached-CE lineage (user, Jul 18; see LATENT_RL_PLAN.md "Attached-CE lineage")
mlq chain, each stage `--after-success` the previous:
CHAIN HISTORY: job 97 cancelled by request @1540 (no ckpt). Rerun job 111 lost @~1860
(mlq runner died); its step-1500 rolling checkpoint survived. Mid-train probes (bench
avg@32): 7.1%@500 → 14.4%@1000 → 15.1%@1500 — PLATEAU after step 1000; AIME ~0 throughout
(3/960 @1000, noise); answer-line compliance 100% everywhere; transcripts recorded
(eval_aime now always writes *_transcripts_step*.jsonl). Sufficiency call: pretraining
sufficient, RL is the lever with slope; user approved post-training Jul 18.
GATE RESULT (job 129, Jul 18): FAILED exit 2 on DAPO-Math — 128/128 trajectories reward
0.0, within_group_reward_std 0.0; DAPO probe (131) confirmed emit-only 0/512+ hits →
task difficulty, not thought noise. Latent mechanics nominal (think_fraction 0.501).
COLD-START FIX (user-approved Jul 18): 50/50 zero-init gate is reward-starved — measured
~1e-4 hit rate vs 13.9% emit-only at IDENTICAL sampling (temp 1.0/top-p 1.0), so untrained
noisy thoughts (not sampling config) collapse accuracy; value warmup "instant convergence"
was the tell (critic hit the exact HL-Gauss CE floor 1.4292 predicting constant 0).
New `--init-think-probability 0.1` sets the gate head BIAS (weights stay zero); sigma
used the then-current -1.5 (the current default is -2). Jobs 133 (b8 samples)
and 134 (zero-init, starved; dir staged as
`latent_vapo_dm_v1_zeroinit_starved`) superseded by job 135 (`latent_vapo_dm_v1`,
samples-per-prompt 32, init think 10%).
COMPILE (user prescription Jul 18, in progress): torch.compile reduce-overhead/CUDA
graphs "almost exactly like pretraining, for both models" — implemented behind
`--compile` (default on) + `--replay-bucket 64`; static key_mask stepping path
(full-cache masked SDPA, 0-dim tensor positions, persistent zeroed
mark_static_address caches), compiled replay_head_inputs rebound in both consumer
modules + compiled critic.value_logits; refresh_old_statistics now grad-enabled so
refresh/update share one compiled artifact (epoch-0 ratio exactness). See
LATENT_RL_PLAN.md "torch.compile" section. Eager job 135 timings for comparison:
collect ~14-16 s, iteration ~20 s (32 minibatches). Red-team verdict applied:
stepwise-only cudagraphs (replay/critic get plain compile — output-lifetime hazard),
compiled surface = step_core (pure tensors, no dataclass/cache aliasing in graph),
rollout finish-check sync relaxed to every 16 steps (was a per-step D2H serializer);
diff review APPROVED after: remainder eval chunks now PAD to cache width (dynamic
fallback through the compiled step could LRU-evict training graphs), epoch-0
clip-fraction==0 runtime guard added (prints WARNING if refresh/update artifacts
diverge). Deferred lever if collect still dominates: batch all 16 groups into one
left-padded rollout (~6500->~900 step iterations, amortizes eager sampling 16x).
OPS: job 135 cancelled @~step 736 (rolling ckpt @700); job 136 = compiled benchmark
(--steps 160, warmup 2); next = resume 135's run with --compile from
postraining/runs/latent_vapo_dm_v1/latent_vapo_checkpoint.pt after verifying
benchmark correctness (epoch-0 guard silent, sane val_bpb/rewards) + speedup.
BUCKET CAP LEAK (found via jobs 136/137 never converging — collect stuck 47-123 s
from persistent silent inductor compiles): trim_stream capped the bucketed length at
the ORIGINAL stream length, so any group whose content reached the last partial
bucket leaked an arbitrary stream shape -> unbounded compile set. Fixed: trim now
PADS BEYOND the original stream to the strict bucket boundary (pad columns are the
natural all-PAD state; padding-invariance test extended with a forced pad-beyond
case at multiple=64). Job 136 correctness evidence unaffected (epoch-0 guard
exactly 0, val_bpb flat 1.4297-1.4303, step-0 evals match eager, GPU 100%/459 W in
steady regions vs 220 W eager). Job 138 = fixed steady-state probe (192 steps,
--replay-bucket 128); decision: steady collect < eager 14-16 s -> resume compiled,
else --no-compile (top_p>=1 multinomial fast path helps eager too).
BATCHED ROLLOUT (Jul 18, user-approved): all 16 prompt groups roll out as ONE
left-padded batch (512 rows; per-row 2-D key_mask threaded through all three
attention steps; pad-region queries attend all-True — REQUIRED, a fully-masked
SDPA row is NaN and would poison real queries via deeper-layer K/V;
split_rollout_groups clones each group minus its pad columns so downstream is
unchanged). Position shift exact (PoPE/RoPE/YaRN relative — red-team verified
the math). --rollout-groups (default 16), eager-only. Red-team: SOUND; diff
review: APPROVED. --compile default flipped to FALSE (measured 2.6x slower +
blocks the batched path). New think-credit metrics: think/emit_advantage_mean,
think_action_count (update), think_reward_correlation, reward_mean_thinking vs
_pure_emit, thinking_trajectory_fraction (rollout). Job 140 = batched timing
probe (queued behind 139); job 141 = sample dump (sample_latent --math-rows
--json-out) for the HTML report. Untested corner (low risk): RoPE/yarn 2-D
key_mask branches (equivalence test covers PoPE only).
GATE AUDIT VERDICT (think fraction 0.10 -> 0.02): mechanisms all clean EXCEPT
lambda-clamp credit starvation — length_adaptive_lambda alpha=0.05 (VAPO's
thousand-token calibration) clamps lambda to EXACTLY 0 at this run's 12-23
action trajectories -> GAE = TD(0) -> terminal reward credits nothing >1 step
back (lambda^m = 0) -> THINK decisions structurally never receive reward
credit; decline direction likely genuine (zero-init adapter thoughts noisy;
bench rose 4.8->15.2% while think fell) but irreversibility is mechanical.
FIX (shipped, commit 9931036 Jul 19): floored horizon inside
core.length_adaptive_lambda — max(alpha*l, min(l, 1/alpha)); alpha stays 0.05.
Short trajectories get lambda = 1 - 1/l (12-23 actions -> 0.92-0.95, credit at
first action of l=20 now 0.38 vs 0.00), mid lengths the 0.95 baseline, long ones
VAPO's alpha*l unchanged; continuous at both boundaries, tested in
test_core.py::test_length_adaptive_lambda_floors_the_credit_horizon. Live in
belief v3+ runs (launched after 10:38); the Jul-18 v1 lineage incl. the gate
audit ran starved. (Earlier prescription here — "--gae-lambda-alpha default
1.0" — superseded: it fixes short lengths identically but degrades long ones
toward Monte Carlo.) Ruled out: budget confound (thinks don't
consume emit cap; 13-24 of 512 slots), BCE signs, position alignment,
positive-LM leak, warmup bias, refresh divergence (gate clip exactly 0).
Job 142 = step microbenchmark (eager narrow vs reduce-overhead vs MANUAL
cuda-graph capture, batch 32+512, profiler cudaGraphLaunch counts) — tests
whether compiled slowness is dynamo per-call overhead, not attention FLOPs
(the FLOP math says full-cache SDPA is ~us-level; 2.6 ms/step unexplained).
VERDICT (job 138, clean iters 2-4 with ZERO compile spikes): compiled collect
36-40 s vs eager 11-16 s, minibatch 0.13 s vs 0.11 s, iteration 42-57 s vs
14-19 s. The static-cache cudagraph step pays masked SDPA over the FULL ~900-slot
cache every step (~4x the attention FLOPs of eager's prefix-only step) — swamps
the ~3.5 ms/step launch savings at this model size; 459 W was busy-work, not
throughput. New bucket shapes also still appeared at iter 5 (streams lengthen as
policy shifts) re-paying ~30 s compiles. DECISION: resumed eager. Job 139 =
latent_vapo_dm_v1_resume (--resume from ckpt step ~700, --no-compile, otherwise
job-135 args verbatim); eager still gains the top_p>=1 multinomial fast path
(no per-step full-vocab sort) and SYNC_EVERY=16 finish check vs original 135.
Compile plumbing stays in tree behind --compile for a future larger model /
batched-rollout revisit; next perf lever remains batch-all-16-groups (helps
eager directly: ~6500 -> ~900 sequential steps).
CURRENT CHAIN (curriculum lever per plan/red-team): RL prompts switch to
`postraining/data/deepmind-interpolate-rl.parquet` (18k problems, built by
build_deepmind_rl_prompts.py from interpolate splits, 144 eval problems excluded,
zero overlap verified; model scores 15.1% avg@32 there = mixed-success regime VAPO
needs). 132 gate (BINDING) → 133 full latent VAPO v2 (`postraining/runs/latent_vapo_dm_v1`),
both on `.../checkpoint_step1500_snapshot.pt` with `--math-data` override. AIME + bench
evals unchanged (bench stays held out).
1. `pope_attached_mathmix_v3_2k` — from-scratch attached-CE pretraining on mathmix_v3
   (FineWeb-only attached-CE ckpt can't be continued: PoPE pretraining hard-rejects
   checkpoint init by design, and train_adaptation's detached-belief CE can't teach
   the trunk math/template compliance)
2. `pope_attached_dapo_probe` — emit-only DAPO hit-rate/compliance probe (informational)
3. `pope_attached_aime_probe` — emit-only AIME avg@32, informational (hard AIME gate is
   lottery noise at 27M; user's periodic-AIME requirement satisfied by probing after
   every pretraining stage + every 80 RL steps; on no signal anywhere, extend math
   pretraining and re-probe). AIME = eval of record; marginal improvement counts (user)
3b. `pope_attached_bench_probe` — emit-only deepmind-interpolate-easy avg@32 (144
   held-out same-difficulty problems, build_deepmind_eval_set.py; user-approved easier
   benchmark). RL trainer also evals it every 80 steps (bench/accuracy)
NOTE: algorithm is VAPO everywhere; "DAPO" = the DAPO-Math-17K prompt dataset/verifier
4. `latent_vapo_attached_gate` — `train_latent_vapo --rollout-only` reward-variance gate
   (BINDING: exit 2 stops the chain)
5. `latent_vapo_attached_v1` — historical Jul 18 latent VAPO **v2**
   configuration, since superseded by `LATENT_RL_PLAN.md`: FULL-MODEL
   training (no frozen trunk — world model retrained from "will be" to "should be" by
   per-dim thought PPO through the prediction path + token PPO), fixed thought sigma
   (log -1.5; no beta-NLL, no entropy, no KL), NO pretraining objective at RL time
   (user, Jul 18: SIGReg + latent target-prediction dropped — the PPO-ptx anchor was
   built then rejected as a "will be"-vs-"should be" objective conflict; val-BPB guard
   is the sole drift alarm); actor accumulates over each PPO epoch (1 trunk step/epoch,
   red-teamed), renderer probe pinned at --renderer-lr 1e-6; ckpt every 50.
   Jul 19 context correction: rerun from the clean belief-attached checkpoint
   with prompt cap 1024, emitted-answer cap 1024, total THINK+EMIT budget 4096
   (5120 max context), and rollout-groups 4 for 32GB cache fit. Run AIME24
   avg@32 once from the final latent checkpoint rather than paying for it every
   80 steps.
   Jobs 100/101 HELD during the v2 rework — release after review completes.
   train_adaptation.py/adaptation_core.py deleted (frozen-trunk artifacts).

## GPU queue (v4 runner — detached, gated on the user's trading_bot_0 job exiting)
Runner: scratchpad/queue4.sh, log queue4.log. User stopped sparse_churn_2k at step 1140
(pre-TB-stats launch; staged `*_prestats_killed_step1140`; trajectory: 1.6284@300 1.5683@400
1.5330@500 1.5012@600 — steady ~+0.05 BPB behind the scan arm; 1641ms/step measured) to
prioritize the cross-layer arm. queue3 history: `sparse_fast_smoke_v2` DONE (2738.79ms/step —
radix-select bought ~nothing over 2712; BPB 2.5443@40); fast-2k rerun killed for the perf push.
1. `xlayer_smoke` — 40 steps; first run of the cross-layer arm (step time + sanity);
   gated on XLAYER_REVIEWED sentinel (written; two adversarial reviews clean)
2. `sparse_xlayer_2k` — cross-layer quality gate
3. `sparse_churn_2k` — RERUN with churn/* TB stats (rewire/support/hnorm per layer at val
   cadence via metrics.jsonl; deterministic reproduction of the killed run)
4. `baseline_sparse_entmax_fast_2k` — scan-arm quality gate rerun (killed run was pacing dense:
   1.4112@800 vs 1.4116, 1.3567@1300)
5. `nextlat_aux_off_800` — lambdas=0 probe (NEXTLAT_LAMBDA_MSE=0 NEXTLAT_LAMBDA_KL=0)

## Old queue (dead, for the record)
1. ~~`baseline_pope_zero_2k`~~ — killed at ~step 850 (bpb 1.4324@800 vs baseline 1.4116; with-gain PoPE underperforms)
2. ~~`baseline_nextlat_2k`~~ — DONE: 1.3576@2000 vs baseline 1.2967 (+0.061). Verdict below
3. ~~`baseline_pope_zero_nogain_2k`~~ — DONE: 1.3443@2000 vs baseline 1.2967 (+0.048). No-gain slightly WORSE
   than with-gain at matched steps (1.5656/1.4410 vs 1.5539/1.4324 @400/800) — q_gain hypothesis falsified;
   PoPE itself underperforms in this regime. Next suspect if pursued: GQA (NUM_KV_HEADS=8)
4. ~~`baseline_pope_zero_fast_nogain_2k`~~ — DONE: 1.3447@2000 vs slow no-gain 1.3443 — cached-table bf16
   numerics confirmed BPB-neutral (Δ0.0004). 894 vs 932 ms/step (~4% faster; doubled-width attention
   dominates PoPE cost). PoPE line closed: both variants ~+0.048 over baseline
5. ~~`baseline_sparse_entmax_2k`~~ — killed at step 500 (user: perf unacceptable). Reference impl is
   dense-cost + checkpointing: 4.59s/step train (7.7x baseline) and ~165s/val. BPB was ON PACE with dense
   baseline (1.4883@480 vs ~1.49 interp) — quality signal good, cost not. Artifacts staged in
   `*_crashed_step500`. Superseded by the fused run below
6. `baseline_sparse_entmax_fast_2k` — fused Triton kernel (`sparse_entmax_kernel.py` +
   `sparse_entmax_fast_train_gpt.py`): selection at coarse width only (no dense T^2 fine scores),
   per-query gather + in-register entmax-1.5 (tau bisection) + closed-form JVP backward, fp32 atomic
   dk/dv, no activation checkpointing. Parity vs reference: fwd 6e-7, grads 8e-7 (fp32); eager one-layer
   fwd+bwd 47ms vs reference 159ms vs dense SDPA 7ms. val every 100 (milestones preserved)
7. `nextlat_aux_off_800` — queued (tail queue v2); reviewer's decisive probe: lambdas=0 should reproduce
   baseline within RNG noise, proving the NextLat trunk path clean (gap = objective, not bug). NOTE: its
   03:47 result.json in `nextlat_aux_off_800_oomrace/` is a stale substring-race OOM artifact, not data
8. `sparse_fast_smoke_v2` — gated smoke (40 steps) after queue drains: radix-select + broadcast-dedup
   step time (background runner bsh09aleu)
9. `sparse_churn_2k` — queue after smoke: churn-wired arm (`sparse_churn_train_gpt.py`) vs the scan
   control; expect ~1.0-1.1s/step (no selection at all: no coarse matmul, no topk, no scan)

## After the queue drains
- [ ] Cross-layer attention arm (user, Jul 13: "when we're fully done here with perf and such"):
      all-previous-layers-are-candidates — attention candidates drawn from the outputs of every
      earlier layer, not just the current layer's token stream. Design TBD with user before building
- [ ] `../trading_bot_0`: pretraining, then RL (user-approved; scope out that repo first)

## Analysis / follow-ups
- [x] NextLat verdict (reviewer, high confidence): faithful port, no bug; the ~+0.03-0.05 BPB gap is the
      aux objective competing for trunk capacity at 1-10% of the paper's smallest validated duration
      (10B tokens, 100M params vs our ~1B tokens, ~30M). Tied embeddings + softcap are golf-specific
      interactions the paper never ran. Shelve NextLat unless scaling up; aux-off probe queued as confirmation
- [ ] PoPE: if no-gain still lags, next probe is `NUM_KV_HEADS=8` (paper is MHA; GQA+PoPE untested)
- [x] Sparse entmax fused Triton kernel — built early (user call: perf first). Remaining perf lever:
      block-pooled coarse pass (Quest/NSA-style) to kill the T^2 coarse matmul + topk for long context;
      topk (9ms) and dedup sort are the selection floor at T=1024
- [x] Fused selection kernel (`sparse_select_kernel.py`): one warp/query-row, coarse scores in registers,
      radix-select partition (26-iter bitwise binary search for strong/weak rank thresholds + cumsum
      compaction — no tl.sort; sort cost ~4x the rest). 11.25ms @ B=64 vs 17.6 (sort) vs ~25 (torch
      pipeline). Emits top-(k-1-r) strong SET + weakest-r SET, position order — exploration replaces the
      last r slots i.i.d., so distribution-identical to reference (not sample-path-identical under matched
      seed). Dead ends measured: warps=4 rows 2x slower (shared-mem barriers), QT=16 tl.dot tile 1.6x
      slower (Triton 2D-reduction layout conversions), tl.sort 1.6x slower
- [x] Dedup broadcast-compare (`_dedup_valid`): explored slots vs earlier valid slots replaces the
      128-slot sentinel sort in the fused path; earliest-survivor deterministic; 50-trial parity vs
      reference dedup green
- [x] Sub-reviews (radix kernel adversarial + module integration): both CLEAN. Kernel: bit-map
      monotonicity, search exactness, compaction disjointness, short-row fillers all verified; NS=0 case
      added to suite; boundary-tie exploration-protection effect documented. Integration: weak-set
      equivalence proven, dedup 20k-trial parity, RNG stream unchanged, compile-safe (dedup 5-D
      intermediate fuses under Inductor; only an uncompiled train forward would spike ~1GB)
- [ ] Fine-kernel micro-opts NOT taken (marginal vs risk): tl.sort-based exact tau, bf16 p save.
      Ascending-order gathers came free with the partition layout
- [x] Churn-wired arm built (`sparse_churn_train_gpt.py`): NO selection stage — wiring (per-query
      candidate sets) threads across layers within a forward; layer 0 structural (self + 64 local +
      31 strided + 32 random), then entropy/null-gated random rewiring of slots at p <= 1/(m+1);
      q_rewire = clamp(max(H_norm, p_null), 0.05, 1), never scheduled. Eval randomness from integer
      hash (no RNG state consumed). Per-layer rewire/support/H_norm logged at val cadence.
      Eager suite green (34 checks). Compile smoke caught a real fullgraph bug: affine int64 hash key
      folded into Triton int32 index math -> compile error; hash rewritten to 31-bit state
      (mod-2^31 exact under int32 wraparound, all literals < 2^31)
- [x] Perf campaign (user: ">=4x faster" than 2712ms/step). Measured churn arm at production shape
      (B=64 T=1024, 9 layers, compiled fullgraph): 194.6ms/micro baseline. Landed, in order:
      support-masked bwd gathers + fwd V-gather skip (exact: entmax zeros; parity = atomic noise
      floor), bwd num_warps 2->8, Newton tau solve replacing 30-round bisection (8 iters, fp32-exact,
      identical support sets; fwd 4.43->2.77ms/layer), dedicated churn-tail kernel
      (`sparse_churn_kernel.py::churn_rewire`, warps=1, Philox seed-tensor randomness, in-register
      KxK dedup, stats as [5] sums; ~2.5 vs torch tail 3.35ms/layer).
      FINAL: 168.8ms/micro -> ~1400ms/step est (1.94x vs scan arm's 2712). A one-off 341ms reading
      was measurement contamination (compile smoke sharing the GPU session); clean rerun matches the
      profiler CUDA sum (kernels/layer: bwd 5.69, churn 3.44, fwd 2.81). Two adversarial reviews
      (kernel math re-derivation + compile/integration semantics): NO correctness bugs; Newton
      docstring's false "one-step exact" claim fixed (BITER=8 floor documented), tl.rand<1.0
      causality dependence documented. Post-review micro-opts (UNTESTED until GPU frees — rerun
      test_churn.py + test_churn_compile.py + profile_churn.py time after queue drains):
      stats atomics banked x64 (were 2.6M same-address adds/layer), last-layer churn skipped in
      eval (never consumed; kept in training for the _STATS[8] diagnostics row)
- [x] Fusing churn INTO the attention fwd kernel: measured +5.5ms/layer (register pressure + 2D
      layout conversions on the KxK dedup) - REJECTED, kept as dedicated kernel. Chunked-bitmap
      dedup penciled out negative (more layout conversions than saved ops). tl bench floor for the
      KxK dedup ~1.9ms/layer; Philox ~free; base tail loads/stores 0.6ms/layer
- [ ] Perf levers NOT taken (need user/ablation): SPARSE_K=64/32 sweep (halves/quarters all three
      kernel terms; K sweep was already a planned arm), bf16 dk/dv atomics (quality risk), GQA-group
      shared wiring (design change). Note the structural wall: trunk-only micro ~62ms is shared with
      the dense baseline -> at T=1024 total speedup is bounded ~4.7x even with FREE attention;
      the sparse arm's asymptotic win is longer context (dense pays T^2)
- [ ] Compile smoke with fused selection + new dedup once GPU frees; then decide whether the running
      fast_2k (torch-pipeline selection, ~2713ms/step) result stands as the quality gate or relaunch
- [ ] Sparse entmax ablation arms (only if fast 2k shows signal): oracle selection (`SPARSE_COARSE_DIM=0`),
      k sweep (64/128/256). Softmax-over-candidates arm DROPPED (user call: entmax only — clean rejection of
      explored edges is core to the design; block-structured entmax kernel projects ~equal speed anyway)
- [ ] If fast 2k BPB holds: TWO-STAGE token-level selection (user rejected block attention + query-tile
      sharing as learnability risks — correct at k=128/T=1024 granularity: 2 blocks/query is not top-k).
      Stage 1: max-pooled block scores as recall-only prefilter (Quest upper-bound trick), top ~16 regions;
      stage 2: exact coarse scores over shortlist -> per-query token-level top-k, exploration/dedup/entmax
      unchanged. Selection 25 -> ~3-5ms/layer, stays cheap at 20k+. Fine kernel unchanged (13ms flat in T).
      Accepts: no dense-beating at T=1024 (~16-18 vs 7ms/layer); ~7x win vs dense at 20k. No re-ablation of
      routing semantics needed — only stage-1 recall to verify (oracle-recall diagnostic vs full coarse)

## Submission-fork reminders (not for ablations)
- [ ] Strip NextLat dynamics weights from int8 export if NextLat is ever kept (training-only organ, ~2.6MB over budget)
- [ ] ROPE_BASE sweep belongs at the long-context stage

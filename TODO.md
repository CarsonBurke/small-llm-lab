# TODO

## GPU queue (v3 runner — detached, gated on the user's trading_bot_0 job exiting)
Runner: scratchpad/queue3.sh, log queue3.log. Relaunched 14:12 Jul 13 with churn first;
sparse_churn_2k started 14:20 (user's train-planner finished). Earlier context:
`sparse_fast_smoke_v2` DONE (2738.79ms/step — radix-select path bought ~nothing over 2712;
BPB 2.5443@40); fast-2k rerun deliberately killed for the perf push (`*_rerun_killed_early`).
1. `sparse_churn_2k` — RUNNING; churn arm quality gate vs baseline 1.2967@2k (>0.005 threshold);
   also reports real step time (est ~1400ms/step from 168.8ms/micro)
2. `baseline_sparse_entmax_fast_2k` — scan-arm quality gate rerun (killed run was pacing dense:
   1.4112@800 vs 1.4116, 1.3567@1300)
3. `nextlat_aux_off_800` — lambdas=0 probe (NEXTLAT_LAMBDA_MSE=0 NEXTLAT_LAMBDA_KL=0);
   13:17 attempt was rc=2 (stray positional arg, no data) — rerun is the real one

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

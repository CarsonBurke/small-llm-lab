# MiniCPM5 TODO

## Uno diffusion-assisted rollouts — vetoed pending better hardware

Decision: **2026-09-08, user veto.** The proposed qualification takes too long on the current RTX 5090. Defer until better hardware is available and the user explicitly reauthorizes it. Do not queue adapter training, further Uno qualification runs, or an Uno RL campaign in the meantime. Keep production on default AR.

- [ ] Once reauthorized on better hardware, train a real offline reasoning-corpus adapter using the proposed 400M-token curriculum (100M at B2, then 300M at B4). Measure actual training throughput and held-out learning; this budget does not guarantee convergence.
- [ ] Benchmark the trained adapter against invariant AR, current optimized production AR and legacy AR at matched production settings. Require at least 1.25× useful-token throughput against all three.
- [ ] Before production adoption, verify trained-adapter numerical correctness, math quality, acceptance under actor drift, full RL-cycle speedup, and amortized training-cost payback.

### Preserved evidence

Implementation and numerical qualification are complete; trained benefit is unproven. The one-update fixture reached 0.932× invariant AR and 1.493× legacy AR, failing the dual performance gate. No full adapter-learning or RL campaign was launched.

- Assessment: [Uno assessment](../docs/uno_diffusion_augmented_assessment.md)
- Operational instructions: [MiniCPM Uno workflow](README.md#opt-in-uno-diffusion-assisted-rollouts)
- Measurements: [runtime qualification](../ablation_results/uno_invariant_runtime_20260908/result.json)

## Standalone AR runtime — optimized default enabled

User authorized the runtime fixes as the default, a short GEMM comparison and a
ten-minute-capped longer check. Uno training remains vetoed.

- [x] Initial job 5698: invariant AR beat legacy AR on the short workload, but
  bundled fixed arithmetic with compiler/cache fixes. Historical evidence:
  [initial comparison](../ablation_results/minicpm_invariant_ar_20260908/result.json).
- [x] Isolated GEMM comparison, job 5706, 141.80 seconds under a 180-second cap:
  optimized ordinary cuBLAS 15,463 tokens/s versus invariant GEMMs 9,460.
  **Do not include invariant GEMMs in default AR.**
- [x] Enabled fullgraph compilation, compiler-visible FA4 and indexed in-place
  KV writes by default, retaining ordinary GEMMs and the ordinary AR scheduler.
  Explicit eager mode remains legacy; Uno retains its separate invariant target.
- [x] Production-length job 5711, 463.89 seconds under a 600-second cap:
  step-370 actor, four DAPO prompts × sixteen responses, 64 lanes, 11,024-slot
  cache and 10,000-token response cap. Legacy → optimized: 1,251.95 → 4,164.31
  useful tokens/s (3.33×); logical-pool wall time 253.53 → 89.78 seconds (2.82×).
  Peak allocated memory stayed approximately 20.48 GiB.
- [x] After cache release and an in-memory actor update, the reused optimized
  replica matched a fresh replica exactly. Refill check emitted all 80 logical
  responses over 64 physical lanes with two admission events.
- [x] Pending-record resumes now pin rollout arithmetic. Old checkpoints with
  pending legacy records fail closed rather than silently switching targets;
  checkpoints at completed-rollout boundaries can adopt the new default.
- [ ] Establish broader math-quality non-regression. The longer sample scored
  **22/64 optimized versus 26/64 legacy**, with 44 versus 47 completed responses.
  Only four distinct problems were sampled: this neither establishes a systematic
  regression nor proves quality neutrality.
- [x] Ran full-cycle-path profiler job 5748 at production 64-response/10,000-token
  dimensions, capped at 180 seconds, no retries. It stopped during update; this is
  a censored measurement, **not** a completed-step time.
  Startup 25.53s; rollout 102.11s (including 12.61s first compile/capture);
  cache release 0.79s; behavior refresh 21.42s; update >30.21s before termination.
  One actor/critic optimizer pair completed. Different starting actor from job5711:
  this NoRA warmup checkpoint generated 539,335 tokens, so times are not an A/B.
- [x] Bounded replay experiments: rejected the attention-layout change (job5784,
  32.44s; only 0.3–0.4% forward/backward improvement) and restored the original
  SDPA implementation. Retained stronger single-/multi-segment gradient tests.
- [x] Disabled activation checkpointing by default after job5787 (72.67s under
  a 180-second cap). Full 16-trajectory actor/critic optimizer minibatch with
  160k response tokens and auxiliary losses: 19.07→17.60s, 8.4% higher throughput;
  peak allocation 16.77→20.22 GiB. Policy/value losses identical. Maximum final
  parameter difference 6.52e-9, below unchanged-path repeat difference 8.38e-9.
  `--replay-checkpoint-interval 4` remains the explicit lower-memory option.
- [x] Compiled replay MLPs by default after integrated job5820 (116.43s under
  a 180-second cap): matched 16-trajectory/160k-action update 17.73→16.87s,
  5.1% higher warm throughput; peak allocation 20.22→18.16 GiB.
  Policy/value/auxiliary losses identical; maximum parameter difference 7.45e-9
  equals the unchanged-path repeat maximum. One warm candidate timing, not a
  full-cycle or cold-start result; compiler disk-cache state was not controlled.
  `--no-compile-replay` selects the explicit eager comparison.
- [x] Rejected ordinary compiled SiLU backward despite its 6.3% speedup: it
  changed gradients beyond the unchanged-path control. Native SiLU backward
  matched all 24 actor MLPs' output/input/LoRA gradients on actual activations.
  The qualified implementation keeps that derivative native.
- [x] Verified actual helper through exact compiled behavior refresh, full
  parameter-balanced retained-graph updates, checkpoint interval 4, unchanged
  checkpoint keys, and fused-replica/source offload roundtrips (10,130 tokens).
  Independent review caught a source-bound closure in deepcopy; the fix uses
  PyTorch's deepcopy-excluded compilation slot. Runtime testing also required
  disabling AOT buffer donation across the entire update, not only forward.
  Two new CPU regressions added; 110 focused tests and static checks passed.
- [ ] Measure the resulting complete RL-cycle time in a later authorized run.
  The isolated update comparison excludes resident rollout-replica weights;
  reserve memory for them. No full-step or learning-quality improvement claimed.
- Physical-lane-count sweep (64/48/32): **rejected by user**; do not queue it.
- Async rollout/update overlap: discussion only, not an implementation plan.
  Requires a frozen behavior snapshot, policy-lag handling, and simultaneous
  inference/training memory and compute capacity. Do not assume same-GPU overlap
  improves throughput.
  Current code intentionally releases rollout KV before allocating training
  activations; same-GPU overlap must resolve that memory conflict as well.

Benchmark modes: `optimized-ar` is the production runtime, `legacy-ar` the
historical control, and `ar` the fixed-arithmetic Uno reference. AR-only modes
need no Uno checkpoint. `--cache-length` fixes allocation independently of the
output cap; `--export-responses` preserves sampled text and token IDs. All GPU
workloads must use `mlq`; no further workloads are queued for this task.

Evidence:
- [GEMM choice](../ablation_results/minicpm_ar_gemm_choice_20260908/result.json)
- [Production-length comparison and lifecycle checks](../ablation_results/minicpm_ar_production_20260908/result.json)
- [Bounded full-cycle-path profile](../ablation_results/minicpm_rl_phase_profile_20260908/result.json)
- [Rejected attention-layout experiment](../ablation_results/minicpm_replay_layout_20260908/result.json)
- [Accepted checkpoint-recomputation tradeoff](../ablation_results/minicpm_replay_checkpoint_20260908/result.json)
- [Rejected compiler backward and all-layer diagnosis](../ablation_results/minicpm_replay_compile_20260908/result.json)
- [Integrated compiled replay qualification](../ablation_results/minicpm_replay_compile_integrated_20260908/result.json)

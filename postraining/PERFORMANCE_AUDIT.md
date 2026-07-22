# Post-training performance audit

Date: 2026-07-21

Scope: latent VAPO rollout generation, evaluation, replay refresh, actor/critic
updates, data movement, compilation, and diagnostics. This audit is based on
the current source plus the recorded v21 run; it does not disturb the live
training process.

## Measured phase decomposition

The current lineage
`latent_vapo_dapo_nearby_v21_perdim_16m_from_v19_step8064` records:

| Phase | Total recorded time | Share of train + collect |
|---|---:|---:|
| Rollout collection, scoring, and behavior refresh | 13,410 s | 89.1% |
| Actor/critic updates | 1,636 s | 10.9% |
| AIME evaluation | 182 s | amortized, secondary |
| BPB evaluation | 7.8 s | negligible |

In the last 25 rollout pools, collection averaged 17.52 seconds per four
updates while the four updates together averaged about 2.79 seconds. Measured
refresh, pool H2D/D2H, and CPU packing account for about 1.12 seconds of that
pool. Generation, rollout offload, and scoring are therefore the first-order
bottleneck.

## Findings and resolutions

### P0: prompt ingestion is repeated autoregressively for every sample

`rollout_continuations` walks the whole prompt through one-token decoding.
Training repeats each prompt 16 times and AIME repeats it 32 times. Every one
of those prefix calls also runs the fresh Gaussian heads and the four-layer
vocabulary renderer even though only the final prompt belief needs policy
heads.

This combines three avoidable costs:

1. duplicate deterministic prefix work for every stochastic continuation;
2. hundreds of launch-bound one-token model calls;
3. intermediate policy-head and full-vocabulary work whose outputs are unused.

Resolution:

- Add a cache-producing full-sequence prefill over unique prompts.
- Left-pad and length-bucket the already-consumed prompt pool, without changing
  the one-pass dataset cursor.
- Populate each layer's PoPE K-real/K-imag/V prefix cache once per unique
  prompt, then repeat the final output and caches across samples before any RNG
  is consumed.
- Run mean, sigma, gate-facing belief, and renderer only at the final prompt
  position.

The expected prefix work changes from approximately `samples * prompt` to
`prompt` per problem: 16x less duplicated training prefix work and 32x less in
AIME, plus the launch-efficiency gain from dense causal prefill.

### P0: compiled training retains finished rows until the longest trajectory

Compiled rollout explicitly disables finished-row compaction. Recent pools
terminate essentially every trajectory and average roughly 51 actions, yet a
rare long sample keeps all 128 chunk rows stepping. Evaluation already proves
the bounded-shape design with a fixed B16 survivor tail; directional evaluator
measurements reduced warm time by about 36%.

Resolution:

- Enable bounded compiled survivor compaction during training.
- Start with the already-supported fixed B16 tail, shared with evaluation, so
  no arbitrary survivor shapes are introduced.
- Record active/capacity row utilization and compaction-copy time. A measured
  power-of-two ladder can follow if B16 leaves material tail waste.
- Version the rollout sampling schema because compaction changes which global
  RNG draws map to later rows, while leaving the policy distribution intact.

### P0: maximum-context thought buffers are transferred before trimming

Each B128 rollout chunk allocates dense fp32 thoughts at
`[batch, prompt + 4096, 512]`, at least 1 GiB per chunk. The whole maximum-size
batch is synchronously moved to CPU before its actual used stream tail is
known. A 64-prompt pool uses eight such chunks even though recent trajectories
average tens, not thousands, of generated slots.

Resolution:

- Determine the chunk's used width once on device.
- Transfer only sliced stream views to compact owning CPU tensors; never clone
  a second full GPU thought buffer.
- Preserve fp32 thought samples and exact slot alignment.
- Treat compact/sparse thought-action storage as the next schema-level step:
  raw thoughts and old 512-D log-probabilities should ultimately exist only at
  THINK positions, not prompt/token/pad positions.

### P0: GAE and Monte Carlo returns are recomputed twice per replay shard

The eager Python reverse loop launches several pointwise CUDA kernels per
stream position. It runs twice per shard, and a B256 update normally has at
least eight B32 shards. A 400-position update therefore executes about 6,400
Python recurrence iterations; longer streams scale proportionally.

Resolution:

- Compute row-separable advantages and value targets once for the complete
  optimizer minibatch, before replay sharding.
- Slice those precomputed tensors with each replay shard's row plan.
- Fuse the lambda-GAE and lambda-one return recurrences into one traversal.
- Replace the eager GPU loop with a dedicated reverse-recurrence kernel after
  equivalence validation. The compute-once refactor alone removes about 15/16
  of recurrence traversals in the ordinary eight-shard case.

### P1: replay state and planning are dense and repeatedly materialized

Raw thought actions and old per-dimension Gaussian log-probabilities are both
dense fp32 `[batch, stream, 512]` tensors, although recent rows contain only
about 13–22 thoughts in 400–650 stream positions. The replay planner then
copies these tensors through advanced indexing for every shard, rebuilds the
same length plan for refresh and update, scatters refreshed statistics to CPU,
repacks them, and retransfers them.

Resolution design:

- Build one immutable CPU `ReplayPlan` while the packed batch is still on CPU:
  one stable row permutation, contiguous length-bucketed slices, and compact
  integer EMIT/THINK indices.
- Reuse the identical plan for refresh and update, removing GPU-to-CPU length
  synchronization and repeated boolean compaction.
- Store fp32 thoughts and old Gaussian factors compactly with both decision
  and input stream positions, preserving the `p` decision to `p+1` input
  shift.
- Once compact batches fit safely, retain refreshed device minibatches through
  their updates instead of D2H scatter plus repack/H2D. Current measured
  redundant movement is about 1.11 seconds per pool.

This is a schema-level refactor and must not be approximated with bf16 action
storage: that would change behavior-age-zero likelihoods.

### P1: replay has avoidable synchronization and a stale B32 cap

Per-shard `bool(tensor.any())`, finite-loss Python guards, GPU-to-CPU length
copies, and dynamic boolean indexing serialize the launch stream. Meanwhile
the current 16M attention budget permits about B64 at 500 slots, but the hard
B32 cap forces eight or more shards.

Resolution:

- Let the replay plan provide known integer action indices and contiguous row
  slices.
- Accumulate finite flags on device and check once before optimizer stepping.
- After compact state and fused returns reduce memory pressure, choose shard
  rows from measured activation bytes plus `B * L^2`, rather than a fixed B32
  ceiling.

### P1 evaluation: top-p sampling full-sorts the vocabulary

AIME uses top-p 0.7, which currently sorts the full vocabulary at every
recurrent step for every row, including THINK and inert filler rows. Training
uses top-p 1 and already avoids the sort, so this is evaluation-only.

Resolution after the rollout P0 work:

- Implement exact adaptive top-p: compute full-vocabulary normalization,
  select a growing top-k candidate set, and fall back to a full sort only when
  the candidate mass does not reach p.
- Do not approximate or otherwise change the evaluation distribution.

## Deprioritized hypotheses

- BPB: about 2.6 seconds every 300 steps; under 0.1% of recorded wall time.
- Checkpointing: about 0.287 seconds every 32 steps.
- Post-update KL replay: about 0.08–0.20 seconds every 100 steps after startup.
- TensorBoard and JSON output: asynchronous/small, with no evidence of a stall.
- Persistent compile fallback: steady logs report compiled execution without
  fallback. Cold compilation is visible but does not explain ongoing pools.
- The old full-5K masked CUDA-graph decoder: measured roughly 2.6x slower than
  narrow-prefix decoding and should not be restored.

## Implementation order

1. Add pool-boundary phase and utilization telemetry without per-step syncs.
2. Unique-prompt dense cache prefill and stable prompt-length bucketing.
3. Fixed-tail training compaction and pre-D2H stream trimming.
4. Compute advantages/returns once per optimizer minibatch.
5. Immutable replay plan, compact thought state, and retained device batches.
6. Re-profile after the current run; then tune replay rows and adaptive top-p.

## Resolution status

Staged in the working tree, without disturbing the current run:

- A unique-prompt dense causal prefill now computes each problem once, writes
  RoPE/PoPE caches directly, evaluates policy heads only at the final prompt
  position, and expands the cache/state across stochastic group members before
  any RNG is consumed.
- Consumed prompt pools are stable-sorted by encoded length before grouping;
  the sequential one-pass dataset cursor and no-reuse guarantee are unchanged.
- Compiled training now uses the same bounded B16 survivor-tail strategy as
  evaluation, controlled independently by `--rollout-tail-batch`.
- Rollout batches are sliced to their actually used prefix before D2H transfer,
  then all fields are copied into pinned host storage asynchronously with one
  stream synchronization; unused maximum-context thought storage is never
  copied to host.
- Lambda-GAE and lambda-one return targets are computed together once over the
  complete optimizer minibatch, then sliced by replay shards. This replaces
  two reverse Python traversals per shard with one traversal per update.
- Execution schema v20 records the changed RNG-to-row execution mapping. An
  objective-identical pool-boundary resume from v19 requires the explicit
  `--migrate-v20-execution-resume` acknowledgment because future execution is
  not bit-exact.

Deliberately deferred until the staged P0 changes can be profiled:

- sparse THINK-only replay storage and an immutable shared replay plan;
- retaining refreshed device minibatches through update;
- a custom reverse-recurrence kernel;
- replay-row retuning and exact adaptive top-p.

Those changes alter replay storage/lifetime or are secondary in measured wall
time. Implementing them before measuring the P0 pass would combine independent
variables and make regressions harder to localize.

Targeted equivalence and regression tests have been authored for the combined
return recurrence, dense RoPE/PoPE prefill, left padding, repeated-prompt cache
expansion, fixed-tail sizing, and explicit schema migration. They have not been
run. Validation must also confirm that padded dense prefill selects an efficient
SDPA backend; if the arbitrary padding mask falls back to math SDPA, the next
implementation should use varlen Flash prefill rather than accepting a hidden
quadratic materialization.

No tests or model workloads were run while the live job was active, as
requested. No performance claim is considered validated until an equivalent
full post-training workload is measured after that run finishes.

## One-step validation

After the live run finished, mlq job `238` executed one real B256 optimizer
step from the same preserved DAPO critic-warm checkpoint and cursor used by the
previous run. It consumed 16 unused prompts and generated 256 full trajectories;
periodic AIME/BPB/benchmark evaluation was disabled. The new targeted CPU tests
also passed (7 tests).

The closest comparison is the previous run's first pool, which began from the
same actor/critic state and cursor but combined four B256 updates. Prompt subsets
and realized trajectory lengths differ, so normalized figures are more useful
than raw pool wall time:

| Measurement | Previous path | Staged path | Change |
|---|---:|---:|---:|
| Optimizer update | 10.30 s mean, steps 2–4 | 3.14 s | -69.5% |
| Pre-D2H/score/rollout excluding refresh, per trajectory | 110.2 ms | 89.4 ms | -18.9% |
| Same non-refresh time, per sampled action | 81.0 us | 82.4 us | +1.6% |
| D2H, per trajectory | 1.68 ms | 1.03 ms | -39.0% |
| CPU pack, per trajectory | 0.505 ms | 0.366 ms | -27.5% |
| H2D, per trajectory | 0.312 ms | 0.209 ms | -33.1% |
| Peak allocated VRAM | 11.46 GiB | 10.91 GiB | -4.7% |

The staged batch generated fewer actions per trajectory (1,085 versus 1,360),
so the lower per-trajectory rollout time is not by itself evidence of faster
recurrent decoding. Per-action non-refresh time is effectively flat within this
single sample: no material decode regression or uplift is established.

Cold replay compilation dominated collection: 36.33 of 59.21 seconds in the
one-step job. The previous first four-update pool spent 43.56 seconds on the
same cold refresh phase and amortized it across four updates. Consequently the
one-step end-to-end wall time is not a steady-state comparison. The important
measured wins are the update recurrence/sharding path, transfers, packing, and
VRAM; a later normal four-update pool is still required to measure steady
collection and confirm masked-SDPA backend selection.

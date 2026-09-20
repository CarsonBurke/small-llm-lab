# Exact recurrent slot memory

Hypothesis: learned mutable memory with iterative contextual reads can replace
ordinary token attention. This is an experimental architecture, not a retained
quality improvement. The existing `nanogpt_mini_slot_model.py` implements KV-cache
eviction and does not test this hypothesis.

The model lives in `pretraining/nanogpt_mini/recurrent_slots.py`; the operational
entry point is `scripts/train_recurrent_slots.py`.

## State contract

Each independent 1,024-token training/validation row starts with zero contents
in 32 slots of width 512. Learned slot identities are persistent parameters.
For token t:

1. Normalize the current token embedding.
2. Encode the pre-token slot snapshot into shared keys and values.
3. Apply six blocks, each with its own query/output projections and ordinary
   mini MLP. Each block reads the same snapshot; later queries incorporate
   earlier reads. There is no token attention, token cache, or positional FIFO.
4. Predict the next token from the final normalized representation.
5. A shared writer combines normalized old contents, final token representation,
   and slot identity through a 128-wide nonlinear bottleneck. It produces a
   candidate vector and scalar sigmoid gate per slot. Update simultaneously:
   `new = old + gate * (tanh(candidate) - old)`.

The initial gate is 0.1; learned gates may later differ by content and identity.
No functional specialization is presumed. Multi-depth writes, heterogeneous
initial persistence times, and dedicated write-only token representations are
deferred until this first architecture demonstrates value.

Training and incremental decoding use the same `step()` transition. Every
token's write becomes visible only to subsequent tokens. Activation checkpoint
segments recompute work but do not detach the state or shorten credit assignment.
State resets at row boundaries match the reference's context windows.

Shared memory K/V projections reduce repeated slot encoding; this is an explicit
architectural choice, so parameter counts are reported rather than assumed to
match LAM. Fixed slot storage is not a guarantee of cheap training: full BPTT
retains checkpoint boundaries, and exact token recurrence limits parallelism.

## Decision protocol

Compare with `ablation_results/nanomini_fb_lam_first_1k/`: seed 1337,
FineWeb10B-SP1024, 524,288 training tokens/update, sequence length 1,024,
microbatch 64, 1,048,576 validation tokens, 1,000 optimizer updates,
validation every 20 updates, and the control's sum-CE/70% cooldown convention.
The recorded first-only control is 1.3265 BPB. A retained quality extension
must improve by more than 0.005 BPB at the completed matched budget.

Timing, serialized model size, peak training allocation, and memory-state size
are separate measurements. This local validation subset cannot establish the
challenge's full-validation SOTA or the 8xH100 time constraint. An incomplete,
failed, or pruned run cannot support a matched-budget keep decision.

Run GPU numerical contracts through `mlq` before dependent training. The tests
cover prefix causality, snapshot immutability, incremental/sequence agreement,
future-loss writer credit, checkpoint gradients, and compiled numerical parity.
They are correctness checks, not reduced training evidence.

All workloads use `--max-parallel-runs 1 --priority 1`; no existing job is
preempted. Canonical metrics and summaries belong under `ablation_results/`,
with pretraining TensorBoard events under `tb_logs/`.

## First experiment

Correctness job **8464** passed all seven CUDA/bf16 tests in 185.71 seconds,
within its 45-minute limit including compilation. The CPU protocol tests
also passed. Independent model/trainer review found no blocking issue.
Dependent training job **8465**, `nanomini_recurrent_slots_exact_1k`, started
after those GPU contracts passed, then was cancelled at the user's request
after seven updates because GPU utilization was poor. It provides no quality
conclusion. Its last saved checkpoint is initialization (update zero).

```sh
.venv/bin/python scripts/train_recurrent_slots.py \
  --name nanomini_recurrent_slots_exact_1k --steps 1000 --val-every 20 \
  --microbatch 64 --segment-size 16 --slots 32
```

The training workload deliberately has no wall-clock timeout: its authorized
budget is 1,000 updates, and steady training timing is reported separately from compilation.
Conservative pruning begins no earlier than update 600 and requires 300 updates
without a 0.001 BPB improvement in either raw or EMA validation BPB. Pruned runs
exit 75, retain their latest checkpoint, and are not retried automatically.

`result.json` records completion, reference matching, BPB, timing, allocation,
serialized checkpoint size, and the quality keep decision. `config.json` records
the architecture and control provenance. Source copies and checkpoint progress
are stored with the run. The trainer saves optimizer/data/RNG state for recovery,
but does not yet expose a resume command; existing run directories are protected
against overwriting.

Static protocol tests can run directly because they execute no model:

```sh
.venv/bin/python -m pytest -q pretraining/tests/test_recurrent_slots_protocol.py
```

## Execution repair

`recurrent_slots_runtime.py` captures complete forward/backward microbatches in
CUDA graphs, including checkpoint recomputation. Stable parameter-gradient
buffers accumulate across replays and are zeroed outside the graph. Validation
has a separate forward-only graph. Captured outputs alias reusable storage and
must be consumed before another replay.

Five-sample benchmark medians on the RTX 5090:

| Execution | Rows per microbatch | Forward/backward time | Equivalent 524,288-token update |
|---|---:|---:|---:|
| Original compiled segments | 64 | 2.675 s | 21.400 s |
| Full microbatch CUDA graph | 64 | 1.509 s | 12.071 s |
| Full microbatch CUDA graph (rejected: gradient mismatch) | 512 | 4.897 s | 4.897 s |
| Eight concurrent B64 graphs (no additional speed gain) | 64 | — | 12.124 s |

These exclude optimizer execution and are performance checks, not training
evidence. The original actual trainer spent roughly 24–26 seconds per update
after startup. The profile recorded 326,563 kernel events per B64 microbatch;
small matrix operations and host dispatch both matter.

Jobs 8466/8467/8469 verified graph replay against ordinary compiled execution
with distinct inputs, accumulated gradients, changed parameters, and validation
diagnostics. The final check treated the stale-AccumulateGrad stream warning as
an error and passed after releasing old autograd graphs. Reports live in
`ablation_results/recurrent_slots_performance{,_b512,_final}/`.

**B512 failed numerical equivalence; training quality is untested.** Packing verifier
8468 found materially different writer/identity gradients between eight B64
microbatches and one B512 batch, despite a 7.7e-8 relative loss difference.
Do not infer training equivalence from graph replay parity: that comparison
keeps batch shape fixed. Job 8473 found small forward differences beginning before the first memory
write and larger backward differences over longer recurrence horizons. Job 8474
also failed with reduced-precision BF16 reductions disabled: slot-identity
gradient relative error was 0.470, and writer-identity Muon direction error was
1.031. Stricter GEMM accumulation did not establish equivalence. Failed evidence remains in
`ablation_results/recurrent_slots_packing/verification.json`.

Changing execution microbatch preserves the mathematical update/token budget
because this model has no operation coupling different rows. Numerical
trajectories can still change; reference metadata records both microbatch sizes
explicitly rather than claiming identical execution.

The selected training path retains B64, its original accumulation order, and
its precision settings. CUDA graph replay removed host dispatch gaps and passed
all-gradient comparisons, including changed parameters. The eight-stream
experiment (job 8475) also reproduced gradients exactly but required about
15 GiB of additional device footprint without improving throughput over serial
graph replay. Its implementation remains available solely for diagnostic
reproducibility; the trainer does not use it. Reports are under
`ablation_results/recurrent_slots_concurrent/`.

Restart job **8479**, `nanomini_recurrent_slots_graph_1k`, uses the selected
serial B64 graph path and the original 1,000-update / validation-every-20 budget.
It was submitted with parallel limit 1, priority 1, one attempt, and no wall-clock
limit; the conservative stagnation rule above remains enabled. Final integration
review found no blockers, and 44 pure metadata tests passed. Check its canonical
metrics/result files for progress; submission does not establish a quality gain.

## Stopped graph run: update 100

Job 8479 was cancelled at update 100. The checkpoint and full validation panel
remain available. BPB was 1.973663, compared with 1.7226 for first-only LAM and
1.7407 for plain mini at the same update. Training consumed 1,213.29 seconds,
versus 111.76 seconds for LAM. Steady updates were about 12 seconds. This is
a poor intermediate quality/compute result, not a completed 1,000-update result.

Terminal memory RMS was 0.9381 and slot standard deviation 0.0270. These metrics
suggest similar slot contents but do not alone prove slots are redundant.
A checkpoint intervention compares normal memory, slot-mean contents after every
write, and zero contents after every write on the full validation panel.

The earlier B512 gate was too strong as an experiment-selection rule: it
rejects numerical equivalence, not the mathematical validity or potential
quality of a separately labelled B512 training arm. B512 remains unproven for
quality. Its measured speedup must not be described as a trajectory-preserving
optimization. CUDA graphs fixed launch overhead but did not solve small-matrix
throughput or the architecture's 1,024-token serial dependency. Power draw alone
is not a throughput metric, and unused memory does not establish spare compute.

Checkpoint diagnostic job **8481** completed all three full-panel arms:

| Post-write intervention | BPB | Difference from normal |
|---|---:|---:|
| Normal | 1.97365799 | 0 |
| Broadcast slot mean | 1.97367656 | +0.00001857 |
| Zero contents | 2.70562938 | +0.73197139 |

Distinct dynamic slot contents have negligible measured value under this
intervention at update 100. Averaging changes both future reads and writes; it
is not a pure retrieval intervention. Zeroing also creates distribution shift,
so its degradation cannot alone establish useful prefix information. This
checkpoint provides evidence for investigating selective writes, not proof
that a particular replacement will improve BPB. No new training was launched.

Normal B512 validation took 2.505 seconds and differed from logged B64 BPB by
-0.000005. This supports B512 evaluation at this checkpoint; training quality
still requires its own ablation. Canonical diagnostic evidence lives at
`ablation_results/recurrent_slots_checkpoint_memory_use/report.json`.

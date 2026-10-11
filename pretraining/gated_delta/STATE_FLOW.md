# Dense cross-layer state reads

Hypothesis: layer-owned banks with dense differentiable reads and independent
source channels can expose more useful recurrent information than a top-two
mixture, without the sequential execution cost of exact interactive sharing.
This is an experiment, not a demonstrated BPB improvement.

`SharedPoolGatedDeltaGPT` supports `pool_reads` and `pool_readout`. The candidate
uses six single-writer banks, one 128-wide head each, all six reads per layer,
and a concatenated 768-to-512 output map per layer. Each bank retains its own
key, value, query and output adapters; the input projections are shared across
layers. Full-softmax source probabilities multiply each normalized read by
`banks * probability` before concatenation. At uniform routing each source
has unit weight. No source is discarded and the learned output map can use
each independently. The readout starts at zero, preserving the private model
at initialization. The Switch balancing coefficient is zero when every bank
is read; the existing z-loss remains.

Dense layer routing uses fixed source order. Bank j's events for token t are
reads by layers 0 through j-1, write j, then reads by j through L-1. Packing
uses exact static offsets and no filler, capacity padding, sorting, or dynamic
segment search. The recurrence still runs through the compiled custom FLA
operators and training/validation CUDA graphs. An existing packed-gather shape
bug was fixed: the gather now receives `[rows, heads, head_dim]`, rather than
`[1, rows, heads, head_dim]`.

There are two Jacobi passes. Pool writes and queries come from the private
first pass; the second pass reads that memory. The second pass's enriched
states do not write back into the pool. Current-token higher-layer writes are
invisible to a lower-layer read, but their previous-token writes are visible.
There is full gradient flow through the pool and first-pass writers.
Incremental decoding remains unimplemented.

The isolated control uses the same six 128-wide layer-owned banks and two
reads with the existing summed readout. Comparing these arms tests broad
access plus independent source channels and the removal of the now-constant
balance penalty (0 versus the control's 0.01 coefficient). It does not isolate access from
readout capacity or balance regularization; `pool_reads=6, pool_readout=sum` is the further separating
ablation if the combined candidate earns a follow-up. Both arms train for
exactly 1000 updates, validate every 20, and use seed 1337, the same 524288-token
updates, 1024-token sequences, tokenizer, data stream, optimizer and schedule.
The historical backbone reference is `gdn2_custom_ops_pinned_1k` (1.358566 BPB).
Its microbatch is 64, versus 16 for these arms. The historical first pool result
predates the present common projections and bank adapters and cannot isolate
this change. The feedback reference is `nanomini_fb_lam_first_1k` (about 1.3265).

Run artifacts:

- `ablation_results/gdn2_dense_flow128_b16/benchmark.json`
- `ablation_results/gdn2_layer_top2_128_b16_v2/benchmark.json`
- `ablation_results/gdn2_dense_flow128_1k/metrics.jsonl` and `result.json`
- `ablation_results/gdn2_layer_top2_128_1k/metrics.jsonl` and `result.json`
- TensorBoard: corresponding names under `tb_logs/` (pretraining only).

Fresh in-process kernel tuning is used identically in benchmarking and
training (`FLA_CACHE_RESULTS=0`, `FLA_CACHE_MODE=disabled`,
`TRITON_CACHE_AUTOTUNING=0`, `FLA_DISABLE_TENSOR_CACHE=1`). Each training arm
requires its current-source throughput report and explicitly records slow
diagnostic status if it misses the existing plain-mini speed gate. A 1000-step
BPB improvement greater than 0.005 is required for promotion; parameter count
alone does not establish a valid quantized 16 MB submission.

## Qualification so far

131 static/run-protocol checks passed. GPU job 11917 passed four contracts:
literal bank recurrence for routed and layer writers, packed gradients, and
dense-concat causality/whole-graph gradient flow. Job 11927 passed the separately parameterized dense packed backward against
per-bank kernels.

Production dense benchmark job 11922 completed: 2.7403 seconds per
524288-token update (191326 tokens/s), 20022 MiB reserved with co-resident
training and validation graphs, no graph breaks, 28384153 parameters. Plain
mini measured 0.5539 seconds in the same benchmark. The dense arm failed the
plain-mini speed gate (exit 75); its quality run is explicitly diagnostic.
The graph-pool allocation accounting makes the report's allocated-memory
counter less informative than total reserved memory here.

Both zero-initialized readout constructors preserve the RNG state, so changing
readout width leaves every other initialized parameter bit-identical across
candidate and control at the same seed. CPU tests enforce this.

Dense quality job 11924 completed all 1000 updates: **1.337375 BPB**,
versus its own first pass's **1.357507 BPB** (0.020132 functional read benefit).
Training-update time totaled 2804.5 seconds on the RTX 5090. This is a single
seed and does not yet establish an improvement over the queued matched
top-two control. The result's `retained=false` remains correct because the
plain-mini speed gate was not passed. The separately qualified control timing
is 2.2765 seconds/update, 13746 MiB reserved and 26418073 parameters: the dense
candidate costs about 20.4% more update time and 1966080 additional parameters.
Static checks now total 140, including the source-intervention mask contracts.

The matched control's startup was delayed: desktop graphics
sustained 13–16% whole-device utilization, above the existing startup guard's
10% limit for three consecutive polls. Its first idle wait was cancelled
without training and resubmitted as job 11947. The guard remains unchanged.
Job 11954 follows with source interventions and an optional full-panel
query-shift inference comparison. Job 11955 automatically writes
`ablation_results/gdn2_state_flow_comparison/comparison.json` and
`learning_curves.png` after a successful control and terminal evaluation.
`pending.json` records the current unresolved comparison. No quality
promotion or matched-control gain is claimed until that report completes.

Follow-up inspection confirmed the guard eventually cleared without code or
threshold changes. The control reached update 20 with validation 2.384150 BPB
and 100% device utilization; its initial validation matched the dense arm
exactly at 4.152969 BPB. The earlier stall consumed no optimizer updates.

## Isolated execution experiment

`gated_delta_dense_reads.py` implements query-shift execution without changing
the current trainer. Each bank's writer recurrence is replicated across reader
heads. Readers at or above the writer query its inclusive state. For a lower
reader, query t+1 is evaluated against state t, then the output shifts right
with an initial zero. That reconstructs the previous-token state read without
read-only events in the recurrence. Adapters act on the original latent width
before head replication; writer gradients sum through differentiable copies.

For six readers, kernel elements fall from seven events per token to six head
replicas per token, not by sevenfold. Shorter state-passing chains and more
parallel heads may help occupancy. BF16 chunk boundaries also change from
roughly nine writes per chunk to 64, so algebraic equivalence does not establish
numerical or quality equivalence. The queued full-context tests compare outputs
and gradients for both head geometries, check prefix causality and sequence
independence, and use learned six-bank gates from the final checkpoint.
`benchmark_state_flow_execution.py` measures complete optimizer updates and
co-resident graphs, including materialization overhead. No execution switch is
enabled in production pending those results and checkpoint BPB reproduction.

Job 11944 passed both synthetic full-context/causality/sequence-isolation
cases, but failed the learned checkpoint's preset 2% gradient tolerance:
`pool.f_proj.0.weight` differed by 2.1535% in relative norm. Learned output
error was 0.4346%; most other parameter gradients differed by under 1%, with
decay-related gradients higher. This is not proof of a structural recurrence
error, but it prevents treating the BF16 execution as a drop-in training
replacement. The tolerance is unchanged; production remains event-based.

`evaluate_state_flow.py` separately measures final-checkpoint dependence on
cross-layer reads: lattice-only keeps own/lower banks; own-bank-only removes
all other source channels. These interventions preserve query/router values
and gate parameters, but later hidden states and gate values recompute. They
also remove readout amplitude. They are not retrained architecture controls.

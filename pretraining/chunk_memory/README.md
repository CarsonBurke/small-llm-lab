# Chunkwise memory throughput experiment

Target: exceed plain nanoGPT-mini's complete pretraining optimizer-update
throughput, then test quality at 1,000 updates. This is an unproven candidate.

The six-layer, width-512 backbone uses causal attention inside 256-token chunks.
Only learned slot memory connects chunks. After the first block's local attention,
a contextual query reads the previous chunk's memory. The remaining blocks process
the chunk normally. Once all predictions are computed, 32 slot queries attend to
the completed chunk's final states and produce gated memory updates. Writes are
visible to the next chunk; gradients traverse all four chunks of a 1,024-token
window. Memory resets between packed rows. These are independent training streams,
not necessarily natural-document boundaries in the packed dataset.

This explicitly relaxes the original no-token-attention proposal. There is no
within-chunk deep-state feedback. The initial implementation accepts complete
chunks; online decoding would need a separate partial-chunk buffering/cache API.
It does not expose an inaccurate token-by-token decoder.

The benchmark compares both models under compiled CUDA-graph execution, identical
524,288-token updates, 1,024-token rows, mixed bf16/FP32 parameter precision, and
the same Muon/Adam optimizer arithmetic. It includes backward, gradient checks,
optimizer execution, and zeroing. Inputs are already on the GPU for both arms;
transfer and compilation are reported/excluded rather than silently charged to
only one model. No activation checkpoint recomputation is added to the baseline.

A candidate must exceed baseline median tokens/sec by at least 5%, with repeat
separation, before training. The trainer checks the report, model configuration,
source hashes, execution environment, and optimizer-inclusive update protocol.
A rejected speed candidate does not trigger a quality run. Benchmark optimizer
steps measure throughput only; they are not reduced quality evidence.

Quality training retains the existing explicit 1,000-update budget, validation
every 20 updates, full validation panel, and first-only LAM quality reference.
A throughput win alone does not establish a BPB gain. No candidate is retained
as a quality improvement before the matched full-budget result clears 0.005 BPB.

## Queued experiment

- **8485**: five GPU correctness contracts (45-minute execution limit).
- **8486**: C256/B64 complete-update benchmark, dependent on 8485 succeeding
  (30-minute execution limit). Exit 75 means the speed gate failed.
- **8487**: `nanomini_chunk_memory_c256_1k`, dependent on 8486 succeeding,
  with explicit 1,000 updates and validation every 20. No wall-clock limit;
  existing conservative stagnation cull starts no earlier than update 600.

All jobs have priority 1, parallel limit 1, and one attempt. Other running work
is not preempted. At submission there is no measured candidate speed or BPB.
The trainer also independently rejects failed/stale benchmark reports, so a
manually launched training command does not bypass the speed gate accidentally.

Canonical benchmark: `ablation_results/chunk_memory_throughput_c256/benchmark.json`.
Canonical training: `ablation_results/nanomini_chunk_memory_c256_1k/metrics.jsonl`.

```sh
.venv/bin/python scripts/train_recurrent_slots.py \
  --name nanomini_chunk_memory_c256_1k --architecture chunk \
  --throughput-report ablation_results/chunk_memory_throughput_c256/benchmark.json \
  --steps 1000 --val-every 20 --microbatch 64 --segment-size 256 --slots 32
```

The command above must run through mlq. Source/configuration drift invalidates
its throughput report and requires rebenchmarking. Static protocol checks passed
52 tests before submission; GPU checks are pending.

## Learning-design reconsideration

Training job **8487 is held** following the user's request to reconsider quality.
The correctness and throughput jobs remain queued as bounded diagnostic evidence;
a speed win alone will not release training. No chunk-model BPB has been measured.
The exact recurrent predecessor's step-100 score (1.9737 versus plain mini1.7407)
is evidence against that predecessor, not a measured score for this candidate.

The chunk candidate simultaneously changes the temporal write schedule, exact
context visibility, number of reads, and writer architecture. Near a chunk start,
only the current chunk's input tokens remain available exactly: even the preceding
chunk's most recent tokens are compressed into slots. It retains full BPTT, but
that does not restore lost input information or allow within-chunk deep feedback.
The checkpoint intervention on the predecessor supports negligible measured value
of distinct slot contents there; it does not identify why they became redundant
or establish that attention-based chunk writing fixes the issue.

A cleaner direction to assess is a single-pass, layer-by-layer hybrid: preserve
exact recent-token attention across arbitrary chunk boundaries, compress only
older context, and construct memory from representations already available at
that depth. Restricting writes to a parallel/scan-compatible form sacrifices
recursively computed final-state feedback but avoids making all six layers
sequential across time. This is a proposed research direction, not an implemented
or retained quality improvement. Keep both throughput and matched-budget BPB as
separate acceptance conditions; report learning at equal elapsed time too.

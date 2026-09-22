# Layerwise Gated DeltaNet-2 control

Research hypothesis: a conventional, optimized gated-delta memory can replace
nanoGPT-mini token attention without the redundant slot-bank projection work or
full-network token recurrence of the earlier experiment. No BPB or speed gain
is established before queued measurements finish.

The model uses six width-512 blocks, the mini MLP/backbone/head, and official
GDN2 mixers with four 128-wide heads, value expansion 1, and causal short
convolutions of width 4. Each layer has its own evolving associative state.
Channel-wise learned decay, erase, and write controls govern retention; there
is no hardcoded recent-versus-old storage split or eviction age. No ordinary
token attention is used. This is conventional depth-ordered computation, not
feedback from the preceding token's final layer into the current first layer.

The official 64-token WY training chunks preserve the token-level recurrence,
including writes inside a chunk. They are not the delayed-write 256-token
architecture. The mathematical state update reads the current layer input,
updates its state and produces its output; predicting the next token remains
causal. Independent packed 1,024-token rows reset all recurrent/convolution state.

Official source: https://github.com/NVlabs/GatedDeltaNet-2
Paper: https://arxiv.org/abs/2605.22791
Pinned revision and local compatibility changes are recorded in vendor provenance.
The vendored code is under NVIDIA's noncommercial source license, retained with
its notices; this is a research control, not an unrestricted challenge submission
artifact. No claim of submission eligibility is made.

The project environment uses fla-core 0.5.2 and flash-linear-attention 0.5.2. The matching frontend and project-local torchvision 0.28.0 were added with
--no-deps; the latter avoids an incompatible user-site torchvision import.
Torch was not replaced. Installed versions are recorded in environment.json. Official
Triton kernels remain opaque optimized operations, surrounded by compiled tensor
regions and captured together in CUDA graphs. A full-graph compiler assertion
would reject the upstream explicitly-disabled kernel entry point; the runtime
records the actual compilation contract instead of silently substituting a
Python recurrent implementation.

The optimizer is an explicit adapted-mini recipe, not a reproduction of the
paper: existing dense Muon plus embedding/head/scalar Adam groups, with separate
Adam groups for depthwise convolutions (lr 0.002) and no-decay rate/step parameters
(lr 0.002). Official mixer initialization is preserved. In particular, the mini
'zero projection' reset must never recurse through GDN2's q/k/v/gate projections.
These rates are starting choices, not tuned or demonstrated optimal values.

Correctness checks precede dependent workloads. They compare optimized kernels
against a direct GPU recurrence and cover causal visibility, continuation/reset,
and gradients including erase/write controls. Benchmark steps measure throughput
only. The throughput benchmark compares complete 524,288-token optimizer updates
against optimized plain mini, with five measured repeats after preparation.
A passing speed report is required before the 1,000-update quality ablation;
validation remains every 20 updates on the canonical panel. Quality and speed
are separate retention conditions. Training results and source snapshots go to
ablation_results; pretraining TensorBoard stays under tb_logs.

## Queued control

- **8498**: five GPU correctness contracts; 45-minute execution limit.
- **8499**: throughput benchmark, after 8498 succeeds; 30-minute limit.
- **8500**: `nanomini_gated_delta_v2_1k`, after 8499 succeeds; explicit
  1,000 optimizer updates, validation every 20, and no wall-clock limit.

All use priority 1, parallel limit 1, and one attempt. The benchmark exits 75
when the candidate fails the >=5% throughput gain with repeat separation;
dependent training is skipped. Failed correctness also prevents downstream work.
The training stagnation cull remains no earlier than step 600, requiring 300
updates without material raw or EMA validation improvement. At queue submission,
71 pure metadata tests and independent static review passed; GPU evidence is
pending. The retired chunk experiment jobs 8485/8486/8487 are terminal
(cancelled/skipped) and will not train.

Canonical throughput: `ablation_results/gated_delta_v2_throughput/benchmark.json`.
Canonical quality: `ablation_results/nanomini_gated_delta_v2_1k/metrics.jsonl`.
The model enforces the Triton convolution backend; FLA_CONV_BACKEND must be
unset or `triton`, and alternate backends are rejected.

```sh
.venv/bin/python scripts/train_recurrent_slots.py \
  --name nanomini_gated_delta_v2_1k --architecture gdn2 \
  --throughput-report ablation_results/gated_delta_v2_throughput/benchmark.json \
  --steps 1000 --val-every 20 --microbatch 64 --segment-size 64
```

Run the command through mlq; the trainer independently verifies benchmark
source/dependency hashes, hardware, model configuration, and timing protocol.

Correctness job 8498 passed four tests (including the recurrence-gradient oracle)
but stopped at the compiler-boundary audit in the production test. FLA's causal
convolution and gated-normalization dispatchers also intentionally disable
Dynamo tracing. After verifying their installed source, the audit now permits
those two exact function names alongside chunk_gdn2; unrelated breaks remain
errors. Seven metadata tests cover the allowlist/rejection behavior (26 GDN
protocol tests pass). No model or kernel arithmetic changed.

Replacement chain: **8506** correctness -> **8507** throughput -> **8508** quality,
with the same budgets, limits, priority, and gates above. Original dependent
jobs 8499/8500 were skipped after 8498 failed. Results remain at the canonical
paths because those dependent workloads never started.

## Measured outcome

Job **8506 passed all five GPU contracts** in 80.19 seconds. Job **8507**
completed its benchmark and exited 75 because the speed gate was not met:

| Model | Parameters | Median complete update | Tokens/sec |
|---|---:|---:|---:|
| Plain nanoGPT-mini | 19,958,784 | 0.574863 s | 912,023 |
| Layerwise GDN2 control | 24,708,888 | 1.273456 s | 411,705 |

The GDN2 control achieved 45.14% of baseline throughput (2.22x longer updates).
Five timing samples were tightly clustered in each arm. Both used B64/T1024,
524,288-token complete updates, optimized kernels, compiled surrounding tensor
operations and CUDA graphs. Graph pool reserved memory was about 15,306 MiB
for GDN2 versus 8,822 MiB for baseline; post-capture max-allocated counters
understate graph pool footprint and should not be interpreted as total VRAM.

Job **8508 was skipped**. No quality-training BPB result exists for this model.
This rejects the current configuration against the pretraining-throughput
requirement; it does not establish that all gated-delta configurations are
slower or that GDN2 has inferior BPB. The canonical benchmark JSON contains
all samples, dependency/source provenance, and exact model configuration.

## Throughput iteration

The corrected CUDA profile attributes more than 96% of complete-update time
to model forward/backward, so optimizer tuning cannot close the observed gap.
The default GDN2 replay launches 1,093 kernels versus mini's 525. Profile
aggregation uses Chrome trace kernel events only; the original aggregation
double-counted annotation spans and is preserved with a correction record in
ablation_results/gated_delta_profile.

Eight GPU contracts passed for the default and narrower configurations (job
8517), including the head-64 recurrence gradient oracle and graph execution.
Full-update measurements from jobs 8507, 8518, and 8519:

| Head / mixer width | Parameters | GDN2 tok/s | Matched mini tok/s | Ratio |
|---|---:|---:|---:|---:|
| 128 / 512 | 24,708,888 | 411,705 | 912,023 | 0.451x |
| 64 / 512 | 23,922,096 | 445,767 | 909,180 | 0.490x |
| 64 / 256 | 18,985,368 | 626,070 | 911,281 | 0.687x |

All three fail the throughput requirement. Head size and mixer width alter
capacity; these are architectural ablations, not equivalent implementations.
They preserve immediate per-token updates and learned retention without an
age-based storage split. No quality claim follows from these timing runs.
The two narrowed reports retain exact executed source snapshots in their
respective sources/ directories.

The next implementation ablation packs seven independent input projections
into one matrix multiplication. Original parameters, initialization, optimizer
groups and recurrence remain unchanged. Wider GEMM reduction order can change
BF16 rounding; numerical contracts precede measurement. This option is explicit
in model configuration, CLI and benchmark-to-training provenance checks.

### Packed projections and larger document batches

Job 8525 passed all 13 GPU contracts, including nonzero-weight fused/unfused
loss and every-parameter gradient parity, cache causality/continuation, and
captured graph execution after mutating every packed projection source.

| Execution | GDN2 tok/s | Matched mini tok/s | GDN2 reserved VRAM |
|---|---:|---:|---:|
| M256/K64, packed projections, B64 (8526) | 717,171 | 917,701 | 10,970 MiB |
| M256/K64, packed projections, B128 (8527) | 671,545 | 924,864 | 21,562 MiB |

Both fail the speed gate. Larger document batches use more VRAM but make this
GDN2 configuration slower; occupancy or reserved memory alone is not the
objective. Packed B64 improves on the earlier 626,070 tok/s unfused result,
but a fresh same-source unfused arm was not run, so the comparison also includes
the projection hook's scheduling change and separate benchmark sessions.
Exact executed sources are retained alongside both reports.

A scalar-gated FLA control is prepared next at the same M256/K64 state size.
It replaces channel-wise decay and separate erase/write gates with scalar
per-head decay and a shared delta strength. It also changes the output-gate
projection and parameter count. This is an architecture ablation, not an
equivalent implementation or a demonstrated quality improvement.

### Scalar gates and state shape

Scalar control GPU contracts passed (8528: four tests), and additional head-128
recurrence/production-graph contracts passed (8531: two tests). Installed FLA
0.5.2 sources match their wheel RECORDs; exact source hashes accompany reports.

| Scalar configuration | State values/layer | Parameters | Candidate tok/s | Mini tok/s |
|---|---:|---:|---:|---:|
| M256/K64/H4, B64 (8529) | 16,384 | 17,614,256 | 805,218 | 950,802 |
| M256/K64/H4, B128 (8530) | 16,384 | 17,614,256 | 803,645 | 929,177 |
| M128/K128/H1, B64 (8532) | 16,384 | 15,637,260 | 1,007,664 | 960,947 |

Equal state counts do not imply equal learning capacity. The single-head
variant halves input/output projection width, increases each head's key/value
width, and shares scalar gates across the entire layer's state. It is also
smaller in parameter count. The measured 4.8615% throughput gain has separated
timing samples, but misses the predeclared 5% margin. Job 8533 quality training
was therefore skipped, not run for fewer updates. No scalar BPB result exists.

All three executed source trees are archived alongside their benchmark reports.
Next is projection fusion at M128/K128, preserving that architecture and its
original parameter identities rather than reducing capacity again.

### First completed speed-qualified quality run

Packed scalar M128/K128/H1 passed nine GPU contracts (8534). Its complete-update
benchmark (8535) measured 1,089,741 tok/s against 926,871 for mini: 17.57% faster,
with separated repeats. Graph reservation was 7,126 MiB versus 8,790 MiB.
The benchmark's exact source tree is archived under its sources/ directory.

Quality job 8536 completed all 1,000 updates, validating every 20 updates on
the matched canonical panel. It reached **1.439372 BPB**, versus plain mini
1.3433 and first-only LAM 1.3265. Training took 490.275 seconds (518.411 seconds
total); the raw model artifact is 61,543,822 bytes and peak reserved VRAM was
8,294 MiB. This is an unquantized research artifact, not a 16 MB submission.

The run is **not retained**: it loses 0.096072 BPB against mini and 0.112872
against first-only LAM. The speed result demonstrates that this tested
memory-only configuration can outperform mini throughput, not that its learning
tradeoff is acceptable.

The next diagnostic uses M128/K64/H2/expand_v2: the same 16,384 state scalars
per layer, but two independent heads and a 256-wide read instead of 128.
Each head has a narrower key. Padded dense arithmetic rises about 7.8% across
the model; recurrence, convolution and measured latency need separate checking.
Jobs 8537 -> 8538 -> 8539 cover nonsquare-state correctness, throughput and a
conditional matched 1,000-update ablation. No quality result is assumed.

### Interpretation and architectural boundary

The expanded-value arm passed five nonsquare-state GPU tests (8537).
Benchmark 8538 measured a 4.07% median gain (955,819 versus 918,399 tok/s),
but candidate samples ranged from 0.524 to 0.584 seconds and overlapped mini.
It failed both the 5% margin and repeat-separation conditions. Dependent quality
job 8539 was skipped. It provides no BPB evidence about widening the read.

The completed 1.4394-BPB result does not isolate a cause: the speed-qualified
model changes temporal retrieval, read width, number of heads, gate granularity,
parameter count, and the feedback dependency simultaneously. It must not be
treated as a verdict on full-depth latent carry or on full-width GDN2.

The deficit also appears in training loss. At the eleven jointly logged updates
900, 910, ..., 1000, scalar GDN averages 2.42019 nats/token versus mini 2.25605;
every paired difference is positive (0.1584–0.1735). At update 1000 their
validation losses are 2.40237 and 2.242. This rules out a purely held-out
generalization explanation, but does not distinguish capacity, retrieval, or
optimization limitations at the finite update budget.

Layerwise GDN is an execution control. It writes each layer's input to that
layer's own memory; previous tokens' final-layer states do not feed the next
token's first layer. Immediate writes and learned retention survive, but the
original cross-depth feedback contract does not.

A faithful delta-memory extension would retain one pre-token memory snapshot
for successive contextual reads, then write from the final processed token
state for the next token. Replacing the earlier slot bank with already-projected
associative state targets real arithmetic: the slot bank's shared 32x512-to1024
KV projection alone costs 16,777,216 MACs per token before its writer. Such a
replacement need not inherit that cost. Exact final-state feedback still has a
within-document sequential dependency; document batching and fused execution
must be benchmarked rather than assuming the layerwise scan throughput applies.

A narrow softmax control would help separate the current model's reduced read
width/parameterization from its compressed-memory mechanism. Neither that
control nor the faithful delta-carry architecture has been measured here.

### Final-state carry and narrow-attention diagnostics

The authorized follow-up implements both controls in
`pretraining/nanogpt_mini/latent_carry.py` and `narrow_softmax_model.py`.
Carry has six D512 blocks reading one unchanged FP32 H4/K32/V128 matrix,
then writes a learned delta correction from the normalized final hidden state.
The next token can read that write in its first block. There is no ordinary
token attention, delayed chunk write, memory detachment, or prescribed eviction.
Checkpoint segments preserve full 1,024-token BPTT. Gate biases begin at several
retention rates and remain learned; initial memory itself is fixed zeros.
This does not preserve an exact cache of the latest hidden state: its projected
value is written into associative memory with learned strength.

Narrow softmax keeps mini's exact causal token attention, Q/K normalization,
RoPE, MLP and residual width, with one 128-wide attention head. GPU contracts
passed: carry four tests (8540), narrow two tests (8541). Independent static
review found no model or training-contract blockers; 17 protocol tests passed.

Standalone `scripts/benchmark_latent_carry.py` measures five warmup and five
timed complete 524,288-token optimizer updates. `scripts/train_latent_carry.py`
requires matching completed timing evidence but permits an explicitly diagnostic
1,000-update quality run below the speed gate. Promotion still requires both
at least 5% faster separated timings and >0.005 BPB improvement over first-only
LAM. Benchmarks exclude preparation/data transfer; reported training time
includes them. Carry B512 versus reference B64 is a recorded numerical variant.

These experiments triangulate causes, not isolate them. Narrow attention doing
well would show that width128 alone does not force the observed GDN quality loss.
Carry changes source depth, read width, head layout, gating, and aggregate memory:
one shared 16,384-scalar state versus six independent states of that size.
Any gain supports that complete design, not a causal claim about source depth.
An otherwise identical first-block versus final-block writer would isolate source
depth more closely if the complete carry design warrants that follow-up.

Completed timing and narrow-control results:

| Architecture / batch rows | Candidate tok/s | Paired mini tok/s | BPB at 1,000 |
|---|---:|---:|---:|
| Narrow softmax / B64 | 1,240,226 | 920,297 | 1.384314 |
| Final-state carry / B64 | 62,963 | 920,852 | not run at B64 |
| Final-state carry / B512 | 226,900 | 863,891 | cancelled; no 1,000-update result |

B512 improves carry throughput 3.60× over B64, but remains 3.81× slower than
its paired mini timing. Peak allocated VRAM in the carry benchmark was 5,358 MiB
at B512 versus 1,020 MiB at B64; reserved peaks were 7,966 and 1,316 MiB.
These allocator measurements exclude other processes and driver/context memory.
Narrow attention passed the speed gate (+34.76%) with separated repetitions.

Narrow quality job 8550 completed all 1,000 updates and the canonical validation
panel, reaching 1.384314 BPB. It loses 0.041014 versus mini and 0.057814 versus
first-only LAM, while beating the scalar GDN by 0.055058. It is not retained.
Training took 403.789 seconds, total elapsed 457.300 seconds; its raw model is
59,919,467 bytes. Narrowing is therefore a measured quality compromise, but
not a sufficient explanation for the scalar GDN deficit. This comparison does
not isolate compression from the other architectural and optimizer differences.

The full B512 profile (8567) recorded 289,893 kernel events and 2.074 seconds
of summed CUDA kernel durations. Name-based categories were 47.5% GEMMs,
31.4% reductions/normalizations, and 21.0% pointwise work. These are instrumented
kernel sums, not utilization percentages or an exact checkpoint-cost breakdown.
An uninstrumented update in that process took 2.147 seconds.

Trace review identified a concrete avoidable cost in `RecurrentLoss`: 128
independent slices of the full FP32 hidden tensor caused 128 full 1-GiB
gradient fills and 127 full-gradient additions. Those groups took 81.768 and
261.240 ms, respectively. Counts, launch dimensions and neighboring head
backward/copy kernels identify slice-gradient expansion rather than recurrent
state or parameter-gradient accumulation. The implementation now splits hidden
and target tensors once, letting backward concatenate disjoint chunk gradients.
Head shapes, loss convention, memory writes and full BPTT are unchanged.
The carry quality run used its already captured original graph, with executed
sources archived. The user stopped it at 264 completed updates to reconsider
the architecture; its latest checkpoint/validation is update 260, 1.618608 BPB.
This is not a matched 1,000-update quality result. Its queued runtime-only
follow-ups were cancelled/skipped with that change in direction. The measured
avoidable cost alone cannot close a 3.8× gap. The split-head correctness check
passed both GPU tests with the replacement model's initial qualification.

### Rethinking the mechanism: contextual encoder and exact latent refinement

The strongest matched positive evidence remains first-only LAM. Its two-pass
score reads ordinary first-pass transformer states; it does not establish that
exact final-output recurrence is necessary. Ordinary attention remains intact,
and the retrieved information receives substantial subsequent nonlinear
processing. The Full-bandwidth Transformer paper also preserves token attention
and the KV cache; its claimed benefit is improved access to earlier computation,
not replacing all history with mutable compressed memory. Local MiniCPM carry
and GLU results do not establish a general carry benefit.

The replacement experiment is `LatentRefiner` in
`pretraining/nanogpt_mini/latent_refiner.py`: four ordinary full-width D512
encoder blocks with four 128-wide attention heads, followed by one gated
512→2048→512 residual MLP refiner. It retains exact token attention and the
complete latest 512-vector. Only the refiner recurs across tokens; current
encoder states and gate projections are computed in parallel. There is no
associative bank, slot selection, decay schedule, or deferred chunk write.

Three arms share parameters, initialization and final-output summed CE:
`refiner_current` supplies the current encoder state, `refiner_encoder` supplies
the immediately previous encoder state, and `refiner_refined` supplies the
immediately previous finalized refiner state. All use zero source at position
zero and retain future-loss gradients into their sources. The first controls
additional computation; the second tests deep-state reuse; the third tests
recursive refinement. Current/previous-encoder controls execute in parallel.

The bargain is explicit: the refined state cannot influence the four encoder
blocks without making them sequential again. The model also has fewer encoder
blocks than mini. Its approximate 21% reduction in dense trunk MACs is a cost
allowance, not a throughput result; small recurrent GEMMs and checkpoint
recomputation may consume it. Full-shape timing precedes quality qualification.
No quality benefit or speed improvement is assumed for this new family.

The completed follow-up is **rejected**: previous-encoder source reaches
1.3612666 BPB and current-encoder source 1.3623702 at 1,000 updates, with
22.1% and 23.2% paired throughput gains. The 0.00110 source gain misses the
0.005 threshold. Removing either trained gated branch worsens BPB by about
0.043, so branch reliance does not establish a special previous-state benefit.
The exact recursive arm failed combined speed/capacity qualification and has
no matched quality result. See [full contracts, measurements, and interpretation](../nanogpt_mini/LATENT_REFINER.md).

### Standard GDN v2 full-width execution

**Rejected at the matched 1,000-update budget.** Standard full-width GDN2
reached **1.36051193 BPB**, versus plain mini **1.3433** and first-only LAM
**1.3265**. The trained execution configuration also failed the speed gate.
This is a completed negative experiment, not a promoted architecture.

| Model | BPB at update 1,000 | Reported training seconds* |
|---|---:|---:|
| Plain mini | 1.3433 | 529.495 |
| First-only LAM | 1.3265 | 1096.485 |
| Standard full-width GDN2 | 1.36051193 | 1096.596 |

*Accumulated training timers differ: GDN2 includes initial training-graph
preparation/compilation and data transfer; evaluation is excluded. Use the
paired complete-update benchmarks below for execution speed comparisons.

All use the canonical 1,048,576-token / 2,524,883-byte validation panel,
sequence length 1,024, seed 1337 and 524,288 tokens per optimizer update.
The GDN2 run completed all 1,000 updates with finite training and validation;
its serialized, unquantized model is 97,839,478 bytes. This is not a 16MB
submission artifact. Canonical results are in
[the run result](../../ablation_results/gdn2_fullwidth_fla_kfirst_saved_1k/result.json),
with [quality curves](../../ablation_results/gdn2_fullwidth_comparison/quality.png)
and [comparison data](../../ablation_results/gdn2_fullwidth_comparison/comparison.json).

The active architecture is six D512 standard GDN v2 blocks, each with four
K128/V128 heads, channel-wise erase/write/decay controls, standard width-4
causal convolution caches, and its residual MLP. Aggregate value bandwidth
is 512. Current BPE tokens enter the residual stream; there is no historical
token attention, direct previous-final-latent bypass, separate writer,
final-to-bottom loop, or fixed-point solver. Each layer maintains its own
matrix memory. Chunk backward preserves temporal derivatives through all
1,024 positions; training does not detach at each token.

Execution options select vendor or installed FLA kernels, state layout, and
saved versus recomputed backward intermediates. Packed input projections
retain the original parameters and initialization. Sequence and cached
execution use the same selected layout. Source hashes bind local code,
installed FLA Python sources and execution configuration to timing evidence.

The trained configuration is FLA K-first, saved intermediates, packed input
projections, B64. Its complete-update benchmark used five optimizer warmups
and five timed updates, fixed mini B64 and co-resident training/validation
graphs. Both resident validation checks passed:

| Model | Median complete update | Tokens/sec | Reserved MiB |
|---|---:|---:|---:|
| Plain mini | 0.578187 s | 906,779 | 9,784 |
| Full-width GDN2, trained configuration | 1.125012 s | 466,029 | 21,618 |

The speed ratio is 0.513939, or 1.946x update time. Post-reset allocated
counters omit some private-graph pool accounting, so reserved memory is
reported here. A saved training snapshot showed 100% GPU utilization,
510.56W and 25,915MiB device memory; that is a point observation, not an
average utilization measurement. See the [paired benchmark](../../ablation_results/gdn2_fullwidth_fla_kfirst_saved_b64/benchmark.json).

The GDN2-only `--allow-slow-diagnostic` admitted this one quality run despite
the completed slow benchmark. It does not waive source/configuration,
complete-timing or resident-memory checks, and never waives the speed gate
for retention. The final result records `speed_passed=false` and
`retained=false`. The experiment is worse by 0.01721193 BPB than mini and
0.03401193 than first-only LAM, instead of the required >0.005 improvement.

Validation: 101 pure protocol tests passed. Nine GPU contracts passed in
job 8628; production B64/T1024/D512 gradient and CUDA-graph freshness parity
passed in job 8650. Initial production-test attempts exposed test memory
lifecycle and rounded scalar-sensitivity problems, corrected by computing
reference gradients before capture and requiring >5% gradient separation
under test-only weight perturbation. All parity tolerances stayed unchanged.
Independent static review found no blockers. The three additional K128
layout/recomputation literal-gradient cases passed in job 8657 (8.72s),
bringing the qualified GPU execution suite to 13 cases across these jobs.

Job 8652 completed the quality diagnostic. Profile 8653 and execution-only
benchmarks 8654–8656 also completed. All workloads used mlq priority 1 and
parallel limit 1; training had a 30-minute limit and execution measurements
20 minutes. All four complete-update benchmarks failed the speed gate:

| Execution schedule | GDN2 tokens/sec | Paired mini tokens/sec | Throughput ratio |
|---|---:|---:|---:|
| K-first, saved (trained) | 466,029 | 906,779 | 0.51394 |
| K-first, recomputed | 479,766 | 945,760 | 0.50728 |
| V-first, saved | 492,229 | 956,078 | 0.51484 |
| V-first, recomputed | 478,896 | 939,403 | 0.50979 |

Compare paired ratios: absolute rates also vary between baseline samples.
There is no evidence here of a meaningful layout/recomputation rescue.
Only the first schedule has a trained checkpoint; the others are execution
measurements of mathematically equivalent configurations, not extra quality
runs. Exit code 75 denotes a completed benchmark rejected by the speed gate,
not an incomplete timing run.

Interpretation is deliberately limited. Full-width values do not remove
history compression: each head stores superposed associations, not an exact
bank of previous final-layer states. GDN2 has 24,708,888 parameters versus
mini's 19,958,784, different mixer initialization and dedicated optimizer
groups for convolutions/decay. LAM retains ordinary attention and reads
previous-pass final states. These comparisons therefore cannot isolate
compression, forgetting, source depth or write quality as the cause of the
gap. Missing within-window temporal gradients are not the explanation.
A bank of rich final-layer latents remains a separate, untested hypothesis.

The subsequent [final-to-first GDN2 feedback experiment](FEEDBACK.md) retained
the transformer and replaced first-layer LAM's carry with attached GDN2
memory. It reached 1.3228 BPB at 1,000 updates, improving first-layer LAM by
0.0037 at 9.66% more training time. This missed the >0.005 retention rule;
source and checkpoint are archived, and the active integration was removed.
A three-pass follow-up was subsequently stopped at the user's update-400
gate: 1.4566 BPB versus two-pass training's 1.4510, with 48.33% more training
time. It is archived with its checkpoint and has no 1,000-update result.

Profile 8653 completed. Its paired complete-update medians were 549.09ms
for mini and 1038.91ms for GDN2 (954,837 versus 504,652 tokens/sec). This
separate timing sample is not substituted for the training admission
benchmark. Actual CUDA kernel durations for one B64 microbatch total
67.085ms and 127.513ms respectively. GDN2 temporal kernels cost 35.352ms,
short convolutions 6.058ms, and dense GEMM/reduction kernels 46.460ms;
mini's FlashAttention kernels cost 9.683ms and dense kernels 37.786ms.
Selected copy/concatenation kernels add 10.524ms in GDN2 versus 1.387ms
in mini. Other normalization, elementwise and reduction work accounts for
the remainder. Kernel categories are exclusive; annotation spans are not
counted as kernel execution.

The instrumented full-update replay stage costs 1011.154ms versus 536.450ms;
optimizer steps cost 25.893ms versus 14.956ms. Optimizer work is therefore
only about 2.5% of GDN2 update time. CUDA kernels occupy about 99.8% of
the profiled kernel timeline, which indicates little inter-kernel idle time,
not necessarily high SM or tensor-core occupancy. This is not primarily a
CPU-launch starvation problem. Fusion/copy cleanup can offer modest gains,
but these measurements do not support treating such cleanup as a credible
standalone remedy for the near-twofold throughput gap. The quality failure
is independent. See [profile evidence](../../ablation_results/gdn2_fullwidth_fla_kfirst_saved_profile/profile.json).

Follow-up evidence is in [inference and learning diagnostics](DIAGNOSTICS.md)
and [execution optimization](EXECUTION_OPTIMIZATION.md), whose whole-graph
custom operators under a pinned kernel profile now train the production
configuration at 972 ms per update (from 1,097) with final BPB 1.35857 against
1.36051 over the matched 1,000 updates, with the
[4K context protocol](CONTEXT_ABLATION.md) recorded separately. These separate
measured outcomes from unresolved learning and hardware hypotheses. The
[shared pool of routed state banks](STATE_POOL.md) adds twelve GDN2 banks per
sequence that all six layers write to and read from through top-1 routers,
in two Jacobi passes alongside the unchanged private recurrences.

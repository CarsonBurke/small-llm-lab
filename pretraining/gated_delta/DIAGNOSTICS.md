# Full-width GDN2: inference and learning diagnosis

The 1,000-update checkpoint misses the current retention gates, but that is
not evidence that this architecture has reached its best attainable quality.
Its disadvantage to mini shrinks from 0.10523 BPB at update 200 to 0.01721 at
update 1,000. Optimization, capacity, retention and positional addressing
remain distinct hypotheses.

## Cached inference

[Canonical benchmark](../../ablation_results/gdn2_inference_fixed_prefix_v4/benchmark.json)
and [compact comparison](../../ablation_results/gdn2_inference_fixed_prefix_v4/comparison.json).
Job 8665 completed all 12 cases on the RTX 5090. Each uses five warmups and
ten measured CUDA-graph replays, frozen BF16 dense weights, FP32 GDN decay
parameters, and prepacked projections for both models. Mini stores real BF16
keys and values. GDN uses the trained checkpoint; mini uses nonzero random
weights because the matched reference checkpoint is unavailable. Inputs are
synthetic, and this measures execution rather than generation quality.

| Batch | Prefix tokens | Mini GPU microseconds | GDN2 GPU microseconds | GDN2 throughput / mini |
|---:|---:|---:|---:|---:|
| 1 | 128 | 142.6 | 169.8 | 0.840x |
| 1 | 1,024 | 151.2 | 172.5 | 0.876x |
| 1 | 4,096 | 191.3 | 182.5 | 1.048x |
| 32 | 128 | 365.5 | 442.9 | 0.825x |
| 32 | 1,024 | 603.0 | 443.1 | 1.361x |
| 32 | 4,096 | 1,346.5 | 448.3 | 3.004x |

Each replay processes one next token per document from a restored fixed
prefix. Cache restoration, host token transfer and sampling are excluded;
full-vocabulary logits are included. These are stationary-prefix latencies,
not measured growing-stream serving throughput. Prefill is slower for GDN2:
0.50–0.72x mini across the measured cases. The 4,096-token results describe
execution scaling, not quality beyond the 1,024-token training horizon.

Full-sequence/cached parity, original/deployment-precision parity and
captured-cache mutation checks passed before timing. The mutation check
compares the context-induced logit difference itself, so a weak memory
contribution cannot hide stale cache inputs behind an overall-logit tolerance.
An earlier partial benchmark stored mini keys in FP32; it was cancelled and
superseded by the BF16-cache run. Earlier compilation and sensitivity-check
iterations are not used as final comparative evidence.

GDN's measured cache is 1,646,592 bytes/document: 1.5 MiB of FP32 recurrent
matrices plus 72 KiB of BF16 convolution state. Mini needs 12 KiB per cached
token/document, including the next-token slot in these measurements: about
12.01 MiB at prefix 1,024 and 48.01 MiB at prefix 4,096. Packed weights are
shared across documents and are separate from these cache figures.

## What costs time in training

The existing paired profile measures 127.513ms of CUDA kernels per B64
microbatch for GDN2 versus 67.085ms for mini. GDN temporal kernels cost
35.352ms and short convolutions 6.058ms, versus mini FlashAttention's
9.683ms. Dense GEMM/reduction work costs 46.460ms versus 37.786ms, and
selected copying/concatenation costs 10.524ms versus 1.387ms. Other
normalization and elementwise work accounts for the remainder.

The optimizer is only about 2.5% of GDN's full-update time. Little idle space
between kernels does not imply each kernel saturates SMs or tensor cores.
Training and prefill benefit less from fixed-size memory than long-context
cached decoding: mini's FlashAttention is highly efficient at these short
training windows, while GDN pays for gate projections, convolutions and the
chunkwise forward/backward construction. Frozen inference can prepack
weights once; training must preserve fresh weights and gradients.

## Learning observations

[Checkpoint diagnostic](../../ablation_results/gdn2_fullwidth_learning_diagnostic_v2/diagnostic.json),
job 8664, sampled eight evenly spaced canonical validation rows (8,192 tokens).
No optimizer updates were performed. This sample is not a replacement for
the canonical 1.36051193 BPB evaluation. Instrumented versus canonical
logits differ by 0.001756 relative RMS; reported position losses use the
uninstrumented compiled forward. Instrumentation changes compiled
materialization/fusion, and exact bitwise equality is not assumed.

- Trained memory-branch RMS is 0.375–0.788 times residual-input RMS, versus
  roughly 0.0005–0.0007 at initialization. The branch is not globally dormant.
  This includes current-token processing and does not establish distant use.
- Erase/write saturation below 0.01 or above 0.99 affects less than 0.65% of
  channels at each layer on this sample. Wholesale sigmoid lockup is absent.
- Uncentered key effective rank falls from roughly 100–106 to 9–58 out of
  128. Concentrated addressing could be useful specialization or interference;
  centered spectra and mean-direction energy are needed before calling this
  harmful collapse.
- Eleven of 24 heads have median decay-only stationary channel half-lives
  below one token. Top-layer heads 0 and 2 have medians near 194 and 107.
  These statistics ignore erase operations and input-conditioned changes;
  they are not measured information lifetimes.
- GDN's square mixer matrices start at RMS about 0.00781, versus mini's
  nonzero square initialization near 0.02539. Both use Muon at peak 0.025.
  Thus equal nominal LR does not mean equal relative update scale. Measured
  raw gradient norms are not normalized Muon update magnitudes.

The upstream model's reported training uses AdamW, whereas this experiment
uses mini's Muon/Adam hybrid. The official absolute learning rates should
not be copied blindly across model sizes and budgets. See the
[official training recipe](https://arxiv.org/html/2605.22791v1#A5.SS1).
A momentum-aware update-to-weight diagnostic and a controlled mixer/gate
optimizer ablation are appropriate; the current results do not identify the
correct replacement LR.

## Capacity and position

With L layers, B documents, H heads, key width K and value width V:

- Persistent matrix state: O(L B H K V).
- Per-token matrix update/read: O(L B H K V), plus projections and MLPs.
- T-token recurrence work: O(L B T H K V), plus chunk-processing overhead.
- Short-convolution state: O(L B conv_width H (2K + V)).

Doubling K alone doubles matrix capacity and its update/read work; doubling
both K and V quadruples them. Total latency has other costs, so these are
not whole-model speed multipliers. Saved-intermediate training also retains
BF16 chunk-boundary states of size O(L B ceil(T/C) H K V), with C=64, plus
tokenwise auxiliaries, intra-chunk matrices and backbone activations.
Current B64/T1024 boundary states alone occupy about 768 MiB across six layers.

K256/V128/H4 is a useful larger-addressing experiment that preserves total
value bandwidth 512. It corresponds to mixer_dim=1024, head_dim=256,
expand_v=0.5. It also changes projection widths, key normalization/read
scaling and kernel tiling, so it is not a storage-only change. It must be
benchmarked and ablated rather than assumed to improve quality.

There is no RoPE or explicit position embedding in this GDN model. Causal
width-4 convolutions and ordered, generally noncommuting state transitions
make it order-sensitive. Nevertheless, a query cannot directly select a
historical record by its stored position. More capacity may reduce
interference without necessarily improving positional discrimination.

## Pretraining choices to revisit

The 1,024-token independent-row reset limits trained temporal dependencies.
An isolated 4,096-token training ablation, holding total tokens/update and
update budget fixed, would better reward long retention. Kernel chunks do
not reset state or detach gradients; full within-sequence temporal credit
should be preserved. Longer-context quality needs a separate evaluation,
while the canonical panel remains the matched score.

Keep this separate from optimizer, state-capacity and positional changes.
The current checkpoint's rejection is a selection decision under the tested
recipe, not a reason to abandon mechanistic investigation of the architecture.

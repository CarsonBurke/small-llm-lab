# Exact GDN2 execution optimization

The target is higher training throughput without changing state width, temporal
credit, sequence length, learning hyperparameters, or precision policy. These
are execution experiments, not BPB ablations. The approved 4,096-token training
experiment follows this work.

## Measured comparisons

[Full-update comparison](../../ablation_results/gdn2_exact_execution_inplace/benchmark.json),
job 8673, uses B64/T1024, 524,288 tokens/update, five optimizer warmups and five
timed updates per case, with training and validation CUDA graphs co-resident.
Controls bracket candidates. Source snapshots and installed dependency hashes
are checked before and after execution. Synthetic data measures execution only.

| Execution | Tokens/s | Speedup versus faster control |
|---|---:|---:|
| Original packed projections, control before | 479,979 | 1.000x |
| Independent identical reference | 479,284 | 0.999x |
| Separate projections | 484,594 | 1.010x |
| In-place erase-gradient accumulation | 462,597 | 0.964x |
| Original packed projections, control after | 472,941 | 0.985x |

Separate projections were consistently faster in this comparison, but the gain
was below the predeclared 5% material-improvement gate. The packed default stays.
Eliminating a large backward temporary was slower despite reducing nominal
memory traffic. That custom kernel and its production switches were removed;
exact source snapshots remain with the results for reproducibility.

## Correctness qualification

Before timing, production-shape compiled forward/backward comparisons used
identical initialization with nonzero readout and MLP output weights, exercising
every parameter gradient. The reference repeat and in-place kernel matched all
gradients exactly. Separate projections had maximum gradient relative error
0.00721, below the predeclared 0.015 bound. In-place operator tests also passed
all 16 cases, including layouts, tails, variable lengths and recomputation modes.

Two other candidates failed production qualification and were not timed:

- [Global compiler autotuning](../../ablation_results/gdn2_exact_execution_comparison/benchmark.json)
  changed the worst parameter gradient by 0.02510, despite loss relative error
  only 4.11e-7.
- [Fused decay activation](../../ablation_results/gdn2_exact_execution_fusions/benchmark.json)
  changed the worst parameter gradient by 0.02729, despite loss relative error
  only 6.85e-8. Eleven smaller operator/model/cache tests had passed, showing why
  production-shape qualification matters. Its production integration was removed.

Both worst gradients were the top block's four-component `A_log`. Different
FP32 reduction order and cancellation are plausible explanations, not proven
causes. These failures do not establish a BPB regression; they fail the tighter
execution-equivalence requirement. No tolerance was relaxed. Global autotuning
also changes compiled Muon kernels, which model-gradient comparison alone does
not qualify. It must not be promoted without separate optimizer-update checks.

## What the profile rules out

The original model already compiles dense regions, runs specialized Triton
recurrence kernels and captures forward/backward in CUDA graphs. A low power
reading alone does not identify missing compilation or host dispatch overhead.

The historical B64 replay profile assigns only 0.041ms to packing model weights.
The much larger 2.857ms concatenation assembles activation gradients. Caching
packed training weights would target the wrong cost and introduce freshness
complexity. Naively fusing three convolutions through concatenated activations
would introduce new copies because downstream kernels require contiguous inputs.

The largest temporal backward kernel, WY backward, costs 12.276ms per replay
(9.63% of kernel time). Its hardware resource use and tile selection are the
next targeted investigation. Even halving that kernel would yield only about
5% replay speedup; it cannot by itself close the roughly twofold gap to mini.

Nsight job 8675 could execute the operator but could not read GPU performance
counters (`ERR_NVGPUCTRPERM`). No occupancy or pipeline-utilization claim can be
made from that attempt. The selected upstream tile was BK32/BV32, four warps,
two stages; all input gradients were finite. Compiler resource metadata and
bounded tile comparisons can still be inspected without changing system policy.

## Confirmed autotuning cache problem

The [cache audit](../../ablation_results/gdn2_autotune_cache_audit/audit.json)
reconstructs the current SM120 disk-cache fingerprint and preserves the exact
cached JSON. WY's tuning key contains chunk size, state layout and tensor dtypes,
but omits batch, sequence length, heads, key width and value width. Its current
32x32/four-warp choice was cached with a 0.036864ms timing, from a substantially
smaller workload than this production operator. Disabling FLA's JSON preset
registry does not disable Triton's separate persistent autotuning cache.

[Tile probes](../../ablation_results/gdn2_wy_tiles/benchmark.json) found 255
registers/thread and compiler-reported spills for the cached kernel. Larger
tiles increased spills or exceeded shared memory. A subsequent
[thread-layout comparison](../../ablation_results/gdn2_wy_warps/benchmark.json)
found 32x32/eight-warps reduced the spill count from 246 to 144 and kernel time
from roughly 2.62ms to 2.02ms (1.297x). All seven output gradients passed the
unchanged 1e-4 relative-norm bound, and all 20 candidate timings were faster than
both bracketing controls. These are isolated-operator measurements, not a 30%
whole-model speedup. Compiler spill counts are not measured memory traffic or
hardware occupancy.

Job 8682 tests the supported fresh-process workaround: `FLA_CACHE_RESULTS=0`,
`FLA_CACHE_MODE=disabled`, and `TRITON_CACHE_AUTOTUNING=0` before imports, with the
production batch as the first invocation. It brackets that policy with cached
controls, checks every production model gradient, and measures complete
optimizer updates. No kernel source, arithmetic precision or optimizer setting
changes. In-memory autotune keys still omit geometry, so changing production
shapes requires separate processes. Upstream autotuning also uses a different
timing method from CUDA-graph replay; fresh tuning must be measured rather than
assumed optimal.
The throughput comparison measures warmed execution. Fresh autotuning has a
startup cost; any later training-wall-time claim must include that separately.

The [full-update comparison](../../ablation_results/gdn2_production_retune_policy/benchmark.json)
completed its three arms but failed qualification: cached controls measured
474,847 and 482,193 tokens/s, versus 494,549 for fresh tuning. The worst parameter
gradient differed by 0.02244 (`blocks.5.attn.A_log`, reference norm 0.0055943,
difference norm 0.00012554). The repeated cached control matched exactly.
The apparent 2.56% gain versus the faster control is therefore unqualified,
and broad retuning is not adopted. The dependent 4K benchmark did not run.
The next bounded test changes only the independently qualified WY thread
layout while preserving all other cached kernel selections.

## KDA8 comparison

The user's KDA8 pointer led to the completed
[eight-block optimized run](../../ablation_results/nanogpt_gpt2_kda8_kkkdkkkd_triton_mbs32_optimized_2k/result.json).
Its recorded layout is
six KDA-only mixers and two dense-attention/MLP blocks, three 128-wide memory
heads, saved intermediates, V-first state, separate projections, and Triton
execution. That particular run did not enable CUDA graphs or fused linear CE.
Related KDA runs explore those features; the run name alone is not evidence
that every optimization was enabled.

Its approximately 1.15-second updates process 524,288 GPT-2-vocabulary tokens.
Its 1.1824 final BPB uses 2,000 updates and a 33,554,432-token validation panel;
neither quality nor token throughput is a matched comparison with this 1,024-
vocabulary, 1,000-update experiment. The archived log includes its executed
source and effective settings in its [archived log](../../logs/nanogpt_gpt2_kda8_kkkdkkkd_triton_mbs32_optimized_2k.txt), rather than relying on the subsequently edited
trainer defaults.

Transferable execution mechanisms include saved intermediates, state layout,
shape-correct autotuning, and fused vocabulary loss. The first two were already
tested here; fused vocabulary loss has a much larger target in KDA8's roughly
50K vocabulary than in the current 1K vocabulary. Variable-length flattening
helped KDA avoid shape-specific TileLang compilation in variable-batch workloads;
this fixed-shape graph workload does not have that same problem.

KDA's tied scalar erase/write gate and bounded sigmoid decay differ from GDN2's
independent channelwise erase/write and negative-softplus decay. Its six missing
MLPs and 25% narrower memory projection also change capacity allocation. These
may be useful architectural choices, but are separate quality ablations rather
than execution-only changes. No quality penalty is inferred merely from their
being less expressive in those particular dimensions.

The installed TileLang backend specializes KDA's WY backward and enables
`TL_ENABLE_FAST_MATH=True`. Adapting it to GDN2 requires different derivatives
and explicit numerical qualification; it is not a drop-in kernel swap. The
current GDN2 pipeline already shares several lower-level KDA helpers.

## Whole-graph custom operators and pinned kernel selections

The strict bit-level gate above rejected every candidate whose only effect was
a different floating-point reduction order, including retuned Triton tiles.
That gate is stricter than learning can resolve. This stage relaxes it to a
learning-relevant one and removes the compiled-graph boundaries that FLA's
`torch.compiler.disable` kernels imposed.

Mechanism. `pretraining/nanogpt_mini/gated_delta_ops.py` registers the
installed FLA kernels as `torch.library` custom operators with fake kernels:
the chunk recurrence forward returns the saved-intermediate tuple that the
released `disable_recompute=True` path keeps, its backward calls the released
`chunk_gdn2_bwd` plus in-kernel L2-norm backward, and the width-4 SiLU short
convolution and the swish-gated RMS norm wrap FLA's Triton forward/backward
pairs. `GatedDeltaGPT(custom_ops=True)` swaps the layer's opaque FLA modules for
subclasses that dispatch to those operators for full uncached sequences,
sharing the existing parameters. The loss then compiles with
`fullgraph=True`, so the audit permits no graph break at all. Inductor now owns
the packed-projection split, the residual stream, the erase/write gate casts
and the fp32 residual-gradient accumulation that used to run as eager kernels
between graph segments. Cached and single-token evaluation keep FLA's own
dispatch. The operators declare `needs_exact_strides`, so Inductor hands the
Triton launchers exactly the strides the fake kernels traced; the launchers'
layout checks turn any mismatch into an error rather than silent corruption.
Whole-graph execution also lowered peak reserved memory from 21,614 MiB to
18,428 MiB in the first bracket below.

Kernel selections. `scripts/pin_gdn2_autotune.py` tunes every FLA kernel fresh
at the production shape inside the production CUDA-graph executor with
`FLA_CACHE_RESULTS=0`, `FLA_CACHE_MODE=disabled` and `TRITON_CACHE_AUTOTUNING=0`,
then writes FLA strict-mode config files plus a manifest to
`pretraining/gated_delta/autotune/<profile>/`. Runs select the profile with
`FLA_CACHE_MODE=strict FLA_CONFIG_DIR=<profile>` (and the two cache variables
off); the runtime autotuning policy records the profile's path and a digest of
every JSON file in it, manifest included, so a throughput report only qualifies
a run with identical kernel selections and provenance. FLA's gated-norm and
global-cumsum kernels use plain Triton autotuners that config files cannot pin
(FLA's `configure_fla_cache_autotune` hook would have to run before the package
imports those modules, which its own `__init__` does eagerly); they retune
in-process at the production shape, and the benchmark records their selections
without gating on them. The pinned
profile `rtx5090_gdn2_b64_t1024_custom_ops` covers 14 kernels and 19 keys,
including the 32x32/eight-warp WY layout qualified earlier.

Learning-relevant qualification. `scripts/benchmark_gdn2_execution.py` now
qualifies one complete production update: eight B64 x T1024 microbatches with
the same nonzero readout/MLP initialization, summed in float64. For each
parameter it also estimates the sampling noise of that summed gradient from the
same eight microbatch gradients, sqrt(8/7 x sum_i ||G_i - mean||^2). A
candidate parameter passes when its relative gradient error is at most 0.015,
or when it is at most 0.10 and the difference is at most a quarter of the
reference arm's sampling-noise standard deviation. The loss must agree to 1e-3.
`scripts/benchmark_gdn2_custom_ops.py` runs four fresh-process arms because
FLA and Triton read their policy at import: production-policy control,
custom operators under the production policy, custom operators under the
pinned strict profile, and a trailing control. Every arm saves its CPU
gradient artifact and kernel-selection snapshots; the parent compares
artifacts before reading any timing. The matched 1,000-update run is the
final verifier of learning.

First bracket (`ablation_results/gdn2_custom_ops_bracket`). All four arms
completed; the parent then crashed on its own configuration comparison (the
`custom_ops` key was not yet excluded from the learning-config equality), so
the gradient comparison below was re-run from the saved CPU artifacts with the
corrected gate and is not an in-protocol qualification. Both custom-operator
arms produced bit-identical gradients to each other and passed the
learning-relevant gate against the leading control: loss relative error
6.9e-8, 142 parameters within the strict bound and one noise-qualified
(`blocks.4.attn.A_log`, relative error 0.02317 at 1.1% of its sampling noise;
reference signal-to-noise 0.48), minimum cosine similarity 0.99993. The
trailing control reproduced the leading control exactly. Median seconds per
update over five timed updates: controls 1.1288 and 1.1184, custom operators
1.0316, custom operators pinned 1.0052; every candidate sample was faster than
every control sample. Against the faster control that is 1.084x for the
operators alone and 1.113x with the pinned profile, and peak reserved memory
fell from 21,614 MiB to 18,428 MiB. In-process tuning under the production
policy had picked different tiles from the pinned profile for eight kernels
(among them the short-convolution backward block, the `dhu` value block, the
state-kernel stage count and the L2-norm backward block), which is where the
extra 2.6% came from.

Shared-device collisions (negative result). Four attempts at the qualification
bracket died before or during an arm because a process launched outside the
queue took 22 to 27 GiB of the device: `gdn2_custom_ops_bracket` (parent crash
after all four arms completed), `_v2` and `_v3` (arm out of memory), and
`_v4` (out of memory in the first arm even though an idle-device guard job had
passed moments earlier, because the queue admitted a higher-priority job
between the guard and the bracket). A separate guard job cannot close that
race. Every GDN2 GPU workload now checks the precondition itself.
`wait_for_exclusive_gpu` blocks until `nvidia-smi pmon` and the whole-device
utilization counter agree, on three consecutive 10-second polls, that other
processes hold at most 8,192 MiB, no other compute process uses more than 5%
of the streaming multiprocessors, no graphics-capable process (compositor,
terminals, browsers, whose redraws every measurement here shares) more than
25%, and device utilization (entirely foreign before this process computes)
is at most 10%. `ForeignGpuSampler` then samples other
processes' memory and SM share once a second on a daemon thread through every
timed arm; the report records the peaks, and a benchmark refuses to qualify an
arm whose peak crossed a limit or that could not be observed at all. Point
checks before and after an arm were the first version of this guard and
cannot see a process that arrives and leaves in between, which is what the
bracket below showed had happened to the first bracket's controls. The
trainer waits before creating its immutable run directory and records the
sampled summary in its result. The first sampled profile run
(`ablation_results/gdn2_custom_ops_pinned_profile`, kept as a failed artifact)
completed its measurement and then refused to qualify it because one of 21
samples showed the Wayland compositor at 10% SM; that is a desktop redraw
every measurement here shares, which is why graphics-capable rows now have
their own recorded peak and looser limit while compute rows keep the 5% gate.
A browser compute workload would be a graphics-capable process too, so the
sampler also records every sample and burst in which graphics SM exceeded the
compute limit and fails an arm when a burst lasts more than four consecutive
samples, or when more than four samples and more than a quarter of all samples
exceeded it (a workload pulsing four seconds on and one off would pass the
run-length rule alone). A sample the observer could not take extends the
current burst. The allowance comes from the desktop: the browser's GPU
process held 11% for two consecutive samples during a throughput run
(`ablation_results/gdn2_custom_ops_pinned_b64_v4`, failed closed under the
earlier one-sample rule and kept), and a repeat under the four-sample rule
(`..._b64_v5`) saw the same process for two of eight baseline samples. The failed directories are kept as immutable
failed artifacts; the completed arms of the first bracket are the source of
the numbers above.

Qualified bracket (`ablation_results/gdn2_custom_ops_bracket_v5`). With every
arm seeing an idle device before and after its measurement, the parent
qualified in protocol: both custom-operator arms passed the learning-relevant
gate with the same statistics as the first bracket (loss relative error
6.9e-8, 142 strict and one noise-qualified parameter, worst noise ratio 0.0116,
minimum cosine similarity 0.99993) and the trailing control reproduced the
leading control bit for bit. Median seconds per update over five timed
updates, each arm in a fresh process:

| arm | median s/update | samples | peak reserved MiB |
| --- | --- | --- | --- |
| control before | 1.0722 | 1.0703 to 1.0844 | 21,614 |
| custom operators | 0.9734 | 0.9693 to 0.9745 | 18,428 |
| custom operators, pinned profile | 0.9675 | 0.9644 to 0.9696 | 18,428 |
| control after | 1.0696 | 1.0673 to 1.0800 | 21,614 |

Against the faster control that is 1.099x for the operators alone and 1.106x
with the pinned profile, with every candidate sample faster than every control
sample and a control drift ratio of 1.0024. The pinned profile itself is worth
only 0.6% over in-process tuning here, and its slowest sample overlaps the
fastest in-process sample, so the whole-graph operators carry the gain; the
first bracket's 2.6% profile gap was measured under a slower device. The
controls in this bracket ran about 5% faster than in the first bracket
(1.07 versus 1.13 s per update), which is the strongest evidence that the first
bracket's timings were taken under foreign load that the point checks of the
time could not see.

Matched 1,000-update verifier (`ablation_results/gdn2_custom_ops_pinned_1k`).
The production configuration with whole-graph custom operators under the
pinned strict profile, otherwise identical to `gdn2_fullwidth_fla_kfirst_saved_1k`
(same data order, seed, optimizer, B64 x T1024, eight microbatches per update),
qualified against the pinned throughput report
`ablation_results/gdn2_custom_ops_pinned_b64` and completed with no graph break:

| run | final BPB | ms per update | peak reserved MiB |
| --- | --- | --- | --- |
| reference (`gdn2_fullwidth_fla_kfirst_saved_1k`) | 1.36051 | 1096.6 | 21,234 |
| custom operators, pinned profile | 1.35857 | 972.0 | 18,404 |

Validation BPB tracked the reference within a few thousandths at every matched
step from the first evaluation onward, and the final value is 0.0019 lower,
inside step-to-step scatter and far below the 0.005 keep rule, so learning is
not compromised. Training time per update fell 1.128x. A repeat of the throughput report with
the sampler active through both arms (`ablation_results/gdn2_custom_ops_pinned_b64_v2`)
observed no foreign compute in any sample and reproduced the candidate at
0.9640 s per update (samples 0.9626 to 0.9646) against plain mini at 0.5444.
The report that binds the final runtime sources (after the exclusivity guard
settled, including the duty-share rule) is
`ablation_results/gdn2_custom_ops_pinned_b64_v6`: both arms sampled exclusive
with no sample above the compute limit, candidate 0.9620 s per update
(samples 0.9601 to 0.9643) against plain mini at 0.5449; a future trainer must
qualify against it. `..._b64_v5` (0.9629 s, both arms exclusive) bound the
sources one guard revision earlier and is kept.
Both runs still fail
the 5% speed gate against plain mini and carry `--allow-slow-diagnostic`; this
stage reduces the gap to the transformer baseline (0.564x of its throughput
in the pinned report versus 0.514x in the reference run's report) without
closing it. Whole-graph
custom operators under the pinned profile are now the production execution
path for full-width GDN2.

Profile (`ablation_results/gdn2_custom_ops_pinned_profile_v2`, sampled
exclusive throughout, no graph break, all eighteen short convolutions on the
Triton path). Against the reference profile
`gdn2_fullwidth_fla_kfirst_saved_profile`, the full update fell from 1038.9 ms
to 963.0 ms and steady-state reserved memory from 21,586 MiB to 18,404 MiB.
Per microbatch the replayed CUDA time fell from 127.5 ms to 120.7 ms and the
kernel launches from 835 to 734: Inductor no longer emits the boundary copies,
the seven-way packed-projection gradient concatenation or the eager fp32
residual-gradient accumulation that the three graph segments per layer used
to need, while the GDN2 kernels themselves (the WY backward, the intra-chunk
backward, the state kernels) take the same time as before. The remaining
budget is dominated by the projection GEMMs and those FLA kernels, which is
where any further gain would have to come from.

## Installed-kernel options of the custom operators

FLA 0.5.2 (the latest release; upstream main adds no GDN2 kernels beyond it)
ships two chunk-path options the production execution did not use.

**`safe_gate` is not an execution option for GDN2.** Its sub-chunk intra
kernels (`chunk_gdn2_fwd_kernel_intra_sub_chunk` and the `SAFE_GATE` backward
branch) build each 16-token diagonal block with one tensor-core dot by
exponentiating the cumulative log-decay against the sub-chunk midpoint in both
directions, `exp2(g - g_mid)` and `exp2(g_mid - g)`, up to eight tokens away.
The production kernels only ever exponentiate non-positive differences, which
underflow to the correct limit. FLA's frontend therefore requires per-token
log-decay within [-5, 0) for the sub-chunk kernels (its bounded activation
`lower_bound * sigmoid(exp(A_log) * g)` guarantees this for KDA-style gates) and
refuses `safe_gate` with `use_gate_in_kernel` unless that bound is supplied.
GDN2's activation `-exp(A_log) * softplus(g + dt_bias)` is unbounded, and the
trained full-width checkpoint violates the range by a wide margin
(`ablation_results/gdn2_fullwidth_learning_diagnostic_v2`, the
`gdn2_fullwidth_fla_kfirst_saved_1k` model): per-token log-decay at the first
percentile is -18.8 (layer 0), -7.2 (layer 1) and -23.1 (layer 2) nats, the
tenth percentile of layer 0 is -5.3, and layers 3 and 4 hold channels whose
stationary half-life is 0.06 to 0.13 tokens, about -11 nats on every token.
Eight consecutive tokens of such a channel put 2^127 or more into the fp32
exponentials, which overflow to infinity and then to NaN in the score
matrices. At initialization the same statistic is -1.3 nats, so the one-update
fp64 gradient gate would have qualified the option and the failure would have
appeared only after the decay parameters moved. The option was removed from
the custom operators, the model configuration and the scripts rather than
guarded, because bounding the activation is a model change outside exact
execution optimization; the protocol tests assert that it stays absent.

**`gate_in_kernel`** moves the decay activation into FLA's chunk cumsum kernel
(`kda_gate_chunk_cumsum` forward, `kda_gate_bwd` backward, both pinnable): the
kernels take the raw bf16 projection plus `A_log` and `dt_bias`, compute the
same fp32 `-exp(A_log) * softplus(g + dt_bias)` and return the two parameter
gradients, so the surrounding graph loses the softplus forward and backward,
the fp32 decay materialization and its gradient (about 2.6 ms of Inductor glue
per microbatch in the profile). Its `dt_bias` gradient is reduced from the
bf16 `dg` the kernel returns instead of the frontend's fp32 `dg`, and the
cumsum accumulates in a different order, so it is a kernel-order variant rather
than a bit-identical one; the GPU contract tests hold it to a 4e-3
norm-relative tolerance against the released operator and the fp64 gradient
gate qualifies it like any other execution variant. The released chunk
operator with cached state takes the same raw projection through its own
in-kernel gate, the short-sequence recurrent kernel keeps the frontend's fp32
decay, and grouped value heads are refused because the frontend repeats the
projection across value-head groups while the kernels index the parameters
per key head.

Bracket (`ablation_results/gdn2_gate_in_kernel_bracket`, job 8806, after the
GPU contract job 8805 passed all kernel-option cases). Four fresh-process arms
under the production policy (in-process tuning, no pinned profile): control,
custom operators, custom operators with `gate_in_kernel`, control. The
`gate_in_kernel` arm passed the learning-relevant gate against the leading
control: loss relative error 3.8e-7, 142 parameters strict and one
noise-qualified (`blocks.4.attn.A_log`, relative error 0.0190 at 1.3% of its
sampling noise), minimum cosine similarity 0.99992; the plain custom-operator
arm reproduced its earlier statistics and the trailing control matched the
leading control bit for bit. Median seconds per update over five timed
updates: controls 1.1000 and 1.0641, custom operators 1.0721 (samples 1.0698
to 1.0907), `gate_in_kernel` 1.0531 (samples 1.0518 to 1.0547). Every
`gate_in_kernel` sample was faster than every custom-operator sample (1.018x)
and than every control sample (1.010x against the faster control), but the
control drift ratio was 1.034 and both candidate arms ran well behind the
qualified bracket above (custom operators 1.072 versus 0.973 s), with no
foreign compute in any exclusivity sample; in-process tuning under a busy
desktop (graphics up to 12% of the streaming multiprocessors in five to seven
samples per arm) is the likely difference, not the operators. The
comparison that matters for promotion is therefore the pinned one: profile
`rtx5090_gdn2_b64_t1024_gate_in_kernel` (job 8832) and its throughput report
(`ablation_results/gdn2_gate_in_kernel_pinned_b64`, job 8833) against the
pinned custom-operator report's 0.9620 s per update.

Pinned comparison, negative result. The `gate_in_kernel` profile was tuned
fresh at the production shape (job 8832, profile
`rtx5090_gdn2_b64_t1024_gate_in_kernel`) and its strict-mode throughput report
(`ablation_results/gdn2_gate_in_kernel_pinned_b64`, job 8833) measured
1.0045 s per update (samples 1.0007 to 1.0379) against the pinned
custom-operator report's 0.9620 s (samples 0.9601 to 0.9643). The plain-mini
arm of the same session was also slower than in the custom-operator session
(0.5671 versus 0.5449 s, peak foreign graphics 21% versus a quieter desktop),
so the session drifted by about 4%; relative to its own mini arm the
`gate_in_kernel` candidate cost 1.772x, the pinned custom-operator candidate
1.766x. The in-process bracket's 1.8% advantage does not survive pinned
kernel selections, so `gate_in_kernel` is not promoted: the production policy
remains the custom-operator execution with profile
`rtx5090_gdn2_b64_t1024_custom_ops` and its 1,000-update run
`gdn2_custom_ops_pinned_1k` (1.35857 BPB, 972 ms per update). The option and
its profile stay available for later brackets; no further `gate_in_kernel`
runs are planned.

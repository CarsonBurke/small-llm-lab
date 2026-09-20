# GDN v2 with processed-latent feedback: feasibility decision

Status: investigated design, not an implemented or retained improvement.
No new training result or throughput estimate is established here.

Subsequent scope clarification: the user accepted standard layerwise GDN v2
and its structured temporal backward. The active implementation therefore
uses a persistent state per block, without the final-to-bottom nonlinear loop
or solver below. This document preserves the analysis of that distinct design;
it is not the specification of the active GDN v2 experiment. See
[the experiment record](README.md#standard-gdn-v2-full-width-execution).

The target is a bounded temporal memory written from rich final processing
states, read early enough to change the next token's computation, and trained
by future prediction losses without retaining the token-unrolled processor
graph. The latest final latent remains directly available at full width.
Ordinary historical-token attention is absent from this target.

## What GDN v2 supplies

[Gated DeltaNet-2](https://arxiv.org/html/2605.22791v1) independently controls
key-channel erasure, value-channel writing, and key-channel decay. With state
orientation S[K,V], its update is:

```text
Sbar = diag(decay) S_previous
erase_direction = erase_gate * key
write_value = write_gate * value
S_next = Sbar + outer(key, write_value - transpose(Sbar) erase_direction)
```

For supplied controls this is affine in S_previous. Its specialized forward
and backward can transport information and actual loss derivatives across
time without a token-unrolled graph of a nonlinear processor. This still
computes temporal derivatives; it is not stop-gradient training.

The edit residual is not a measure of future usefulness. Future CE must train
the value, address, erase, write, and decay projections. Reconstructing the
incoming association alone would not ensure that association is useful.

A full-width value is not an independently recoverable full-width record.
Associations share a finite matrix and can interfere. Using GDN v2 therefore
accepts learned associative compression of the broader history. It does not
impose an age cutoff or a recent-versus-old compression schedule. The separate
latest-latent path preserves that one vector without superposition.

## The proposed causal architecture

For each token, all processor reads use the same pre-token memory. A write
becomes visible only to subsequent tokens:

```text
h_t = Processor_theta(token_t, h_(t-1), S_(t-1))
logits_t = Head_theta(h_t)
controls_t = Writer_theta(h_t)
S_t = GDN2_update(S_(t-1), controls_t)
```

Processor has several residual MLP stages and contextual memory reads.
It receives only the current token embedding, the previous final latent,
and the associative memory. Writer projections are shared across time and
are functions of h_t; any extra input must be causally available and specified.
There is no separate four-block token-attention writer.

This is genuinely final-to-bottom feedback. It is also nonlinear in the
previous memory: the processor determines the write controls. Applying a
GDN2 operator to the last line does not make the entire model scan-parallel.

## Eliminate the affine memory; solve the remaining latent feedback

Let Z[B,T,D] be a proposed sequence of final latents. For fixed Z, compute
Writer_theta(Z) in parallel. The GDN2 scan then determines the entire memory
evolution conditional on those writes, including all intervening erasures
and decay. Define:

```text
Memory(Z) = GDN2_prefix_scan(Writer_theta(Z))
F_theta(Z)_t = Processor_theta(token_t, Z_(t-1), Memory(Z)_(t-1))
Z = F_theta(Z)
```

The solver variables are final latents, not a separate full matrix state at
every token. A scan transports an old write across any number of intervening
updates within one evaluation. Outer iterations resolve the remaining
nonlinear feedback through processors and changed write decisions.

The exclusive prefix is essential. The existing layerwise GDN2 frontend reads
after its current write; directly substituting it would change this model.
An initial implementation can align a zero/no-op write followed by writes
0..T-2 with queries 0..T-1. Document starts must reset both the associative
state and the latest-latent path. Sequence continuation requires the actual
incoming state, not an invented zero reset.

Use label-free causal forward iteration or a causality-preserving solver:

```text
Z_next = F_theta(Z)
```

Future targets do not optimize or initialize Z. This differs from the earlier
proposal to optimize latent states jointly with prediction loss under a
consistency penalty, which could encode targets in infeasible latent states.
Pure causal iterations from a causal initialization remain causal even before
convergence; their problem is mismatch with the intended recurrent model.

Strict causality gives a unique finite-sequence solution and a strictly
lower-triangular Jacobian. Exact-arithmetic Jacobi propagation reaches the
solution in at most T rounds, but this is not a useful throughput bound.
Few-round convergence must be measured. Small consistency residuals alone
do not ensure small streaming-prediction errors.

## Future credit without an iteration graph

At a consistent Z, define J = derivative of F with respect to Z and ell as
the derivative of the summed next-token CE with respect to Z. Solve:

```text
adjoint = ell + transpose(J) adjoint
parameter_gradient = explicit_head_gradient
                   + transpose(dF/dtheta) adjoint
```

The last derivative holds Z fixed; it includes processor and writer weights.
Shared weights receive contributions from all positions. No learned credit
predictor, score-function estimator, or RL objective is involved.

Each J-transpose product uses ordinary local derivatives plus GDN2's temporal
backward. It accounts for all direct memory transport at that operating point,
including how an edit affects later reads under the other supplied controls.
Repeated adjoint evaluations account for changes mediated through later
processors and their writes. One evaluation must not be described as complete
credit for all such paths.

Do not retain or differentiate through solver-iteration history. Construct
a fixed local graph at the solved Z, use detached adjoint iterates, and release it
after the parameter gradient. Reusing this graph saves repeated forward work,
but it still retains ordinary layer activations and GDN2 backward intermediates.
If those are recomputed instead, their time belongs in the training cost.

At a feasible primal solution and solved adjoint this gives the same
derivatives as the specified recurrent model, up to numerical precision.
Finite primal or adjoint solves are approximations. Nilpotence does not imply
good conditioning: the inverse of I-J can amplify residual errors strongly.
Temporal weight sharing does not correct stale states or unsolved adjoints.

## Kernel and storage feasibility

The repository has two related operator implementations. The vendored public
entry is `vendor/gdn2_ops/chunk_gdn2.py::chunk_gdn2`; installed FLA exposes its
own GDN2 entry. Both support nonsquare key/value state. Their layout argument
names differ: vendor `transpose_state_layout`, installed FLA `state_v_first`.
The vendor supports K up to 256 and optional `disable_recompute`; saved
intermediates trade memory for backward time, not gradient truncation.

The current frontend computes all projections from a layer's input and keeps
separate layer memories. It does not implement this final-state source or a
shared, repeatedly queried memory snapshot.

Later contextual queries depend on earlier reads. The ordinary fused operator
does not provide a ready-made reusable differentiable memory-read interface.
Calling the whole scan again for every query stage duplicates work. Exporting
its intermediate states is inference-only in the current vendor interface.
A useful implementation needs shared write factorizations/chunk boundary states
with separate query/read stages and accumulated backward contributions.
That is kernel work, not a frontend flag.

Materializing every pre-token matrix is too expensive: at B64/T1024/H4/K128/V128,
FP32 matrices alone occupy 16 GiB. Final Z at D512 occupies 64 MiB in BF16 or
128 MiB in FP32. This reduction concerns solver variables, not all working
memory: projections, factorization buffers, chunk states, activations and
gradients still exist. A matrix state per 64-token chunk is approximately
256 MiB at this geometry, excluding extra boundaries and other buffers.

## Compute evidence and a cost screen

Existing complete-update measurements are recorded in [README.md](README.md):

| Existing model | Model tok/s | Paired mini tok/s | Interpretation |
|---|---:|---:|---|
| Six-layer GDN v2, width 512 | 411,705 | 912,023 | Too slow; no 1,000-update BPB |
| Packed GDN v2, mixer width 256, B64 | 717,171 | 917,701 | Still too slow; reduced width |
| Scalar delta, mixer width 128 | 1,089,741 | 926,871 | Different operator; BPB 1.439372, rejected |

These are not measurements of the proposed shared-bank architecture. In
particular, the scalar quality failure is not a GDN v2 quality result.
The historical `run_optimal_gdn2_ablation.py` is a different vocabulary/hybrid
experiment, requests 2,000 updates, and is not a suitable launcher. Its
isolated-kernel win did not establish a whole-model win.

A dense-MAC screen shows why solver cost is decisive. For D512, six MLPs
with hidden width 2048 cost 12,582,912 MAC/token. Mini's six sets of four
attention projections add 6,291,456, totaling 18,874,368 before attention
mixing and the head. A candidate with two D512 query/output pairs and five
D512 writer projections totals 14,942,208 dense MAC/token, about 79.2% of
that subtotal. Six query/output pairs raise it to 90.3%.

This screen omits GDN2 operations, nonlinearities, head, activation backward,
solver passes, initialization/collection, and memory traffic. It is neither
a latency estimate nor a lower bound. It suggests limited room for repeated
full processor sweeps; eliminating token attention does not fund arbitrary
solver iterations. No defensible expected tok/s is available yet.

Record all costs in a complete update:

```text
primal evaluations, including the final local-graph evaluation
+ state-adjoint VJPs
+ final joint state/parameter VJP
+ loss head and optimizer
+ any initialization, rollout collection, replay, or refresh
```

Fresh training sequences cannot obtain free converged latents from a cache.
If a separate initializer or repeated-data cache is introduced, its cost,
storage, and change to the training contract must be explicit.

## Decision and experimental acceptance criteria

The preferred target remains final-to-bottom latent feedback. GDN v2 gives
it a more structured forward/credit computation, but does not yet make it
competitive. Do not replace it silently with another layerwise model.

A depth-staggered alternative, reading h[t-1,depth] and a memory written from
h[t,depth] at the next depth, is exactly parallelizable. It preserves an
exact latest lower-depth latent and real temporal gradients. Its explicit
compromise is losing previous-final-state access at the bottom. It is an
alternative architecture, not an implementation of this target.

Before a quality run, an implementation of the target must demonstrate:

1. Exclusive-prefix GDN2 forward and all write/control gradients match a
   direct GPU recurrence, including resets and nonzero incoming state.
2. Shared multi-read execution and its accumulated gradients match independent
   reads of the same memory; no current write becomes visible early.
3. Parallel primal states/predictions and implicit parameter gradients match
   an exact short-sequence recurrent GPU oracle at nontrivial weights. Test
   direct long-range influence and influence mediated through another write.
   These are correctness tests, not reduced-run quality evidence.
4. At production B/T/D, report primal/adjoint iterations, streaming prediction
   discrepancy, residuals, full memory use, and complete-update throughput.
   Numerical tolerances must be declared before qualification, relative to the
   selected floating-point oracle; residual size alone is insufficient.
   Nearly zero feedback can make initialization deceptively easy. Continue
   these measurements at training checkpoints as the model learns to use the
   memory; an initialization-only solver benchmark cannot qualify that cost.
5. Only a speed-qualified candidate advances to the canonical matched
   1,000-update experiment: 524,288 tokens/update, validation every 20 updates,
   same validation window/byte denominator and loss convention. Retention
   requires >0.005 BPB gain against first-only LAM (1.3265) and the existing
   separated >=5% full-update speed qualification against mini.

Every GPU workload uses mlq, including correctness and performance probes.
No new workload was launched for this feasibility investigation. The result
is a concrete model and learning formulation, with the remaining kernel and
convergence risks identified, rather than another unqualified training run.

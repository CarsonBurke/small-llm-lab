# TileRT assessment for Parameter Golf

Status: **do not integrate TileRT**  
Assessed: 2026-08-11  
Upstream snapshot: TileRT `v0.1.5.post2`, repository commit
[`43d130d`](https://github.com/tile-ai/TileRT/commit/43d130d6007cc3e9db961d506d1d79e677ec7175)

## Decision

TileRT is not a viable pretraining, post-training rollout, or final inference
backend for the repository's models at the assessed snapshot. It is an
inference-only, model-specific binary runtime for DeepSeek-V3.2 and GLM-5/5.1
on exactly eight B200 GPUs. It cannot execute on the challenge's H100s or the
local RTX 5090, and its public repository does not provide the runtime/compiler
source needed to port a Parameter Golf architecture.

Do not spend an ablation slot or engineering cycle on a TileRT integration.
The adjacent open-source TileLang project is worth evaluating independently:
the repository already has a production-shape KDA operator measurement in
which TileLang is 12.74% to 14.37% faster than the matching Triton backend on
the RTX 5090. That is operator evidence, not an end-to-end training result.

## Scope

This assessment distinguishes four questions that should not be conflated:

| Question | Conclusion at the assessed snapshot |
|---|---|
| Can TileRT accelerate 8xH100 challenge pretraining? | No. TileRT has no training path and cannot run on Hopper. |
| Can TileRT accelerate local RTX 5090 experiments? | No. The distributed wheel contains SM100 code without an SM120 or PTX fallback. |
| Can TileRT accelerate this repository's post-training rollouts or serving? | No direct path. The custom dense, KDA, and byte-native architectures have no TileRT backend. |
| Are TileRT's low-latency ideas relevant? | Yes. Kernel fusion, launch reduction, cache-aware decoding, and schedule overlap target measured local bottlenecks, but must be implemented through supported tools and validated end to end. |

Faster inference does not directly improve FineWeb BPB. It can matter
indirectly if the saved time is reinvested in legal test-time training or
longer-context evaluation while remaining under the evaluation limit. It also
matters to post-training when faster rollout collection permits more useful
optimization under a fixed compute budget.

## Hard compatibility blockers

### Hardware

The official TileRT environment is pinned to eight B200 GPUs, Python 3.12,
PyTorch 2.11 with CUDA 13.0, and a CUDA 13.2 runtime. The upstream
[installation table](https://github.com/tile-ai/TileRT/blob/43d130d6007cc3e9db961d506d1d79e677ec7175/README.md#L66-L84)
treats these versions as hard requirements.

The official `v0.1.5.post2` wheel was also inspected directly. Its DeepSeek
backend contained 52 SM100 cubins and its GLM backend contained 40 SM100
cubins; neither library contained embedded PTX. There is therefore no binary
or JIT fallback for H100 SM90 or RTX 5090 SM120.

This is a design constraint, not merely an untested combination. A TileRT
maintainer states in [issue #17](https://github.com/tile-ai/TileRT/issues/17)
that the implementation uses Blackwell-specific instructions and cannot
currently be rebuilt or run on Hopper. Broader home-lab GPU support, including
the RTX 5090, has no announced timeline in
[issue #33](https://github.com/tile-ai/TileRT/issues/33).

### Training

TileRT exposes inference operators only. It has no backward pass, autograd
contract, optimizer integration, distributed training design, or training
numerics validation. The maintainers explicitly explain in
[issue #31](https://github.com/tile-ai/TileRT/issues/31) that training has
different objectives and is not on the near-term roadmap.

This repository's challenge pretraining path already uses full-graph
`torch.compile`, DDP, BF16 computation, and fused scaled-dot-product attention
in [`train_gpt.py`](../train_gpt.py). TileRT's batch-one decode scheduler does
not replace any of those training components.

### Model architecture

The released runtime loads one of two binary libraries:
`libtilert_dsv32.so` or `libtilert_glm5.so`. The supported mapping is hardcoded
in upstream
[`tilert/__init__.py`](https://github.com/tile-ai/TileRT/blob/43d130d6007cc3e9db961d506d1d79e677ec7175/tilert/__init__.py#L43-L61).
The DeepSeek configuration alone assumes a 129,280-token vocabulary, width
7,168, 61 layers, 128 heads, and 256 routed experts
([`model_args.py`](https://github.com/tile-ai/TileRT/blob/43d130d6007cc3e9db961d506d1d79e677ec7175/tilert/models/deepseek_v3_2/model_args.py#L58-L78)).

Those shapes, FP8 formats, sparse attention, MoE routing, weight sharding, and
multi-token-prediction state do not match the small dense/GQA challenge model,
the KDA recurrent lineage, or the byte-native diffusion lineages. Adapting one
would require a new model-specific runtime backend rather than configuration.

### Public implementation surface

The Python repository is not the source tree used to build TileRT. Its own
[`pyproject.toml`](https://github.com/tile-ai/TileRT/blob/43d130d6007cc3e9db961d506d1d79e677ec7175/pyproject.toml#L68-L71)
calls the repository a presentation copy, says the wheel is built in a private
development repository, and deliberately omits a build system. Most optimized
operators dispatch into closed shared libraries. The public TileLang reference
kernels are useful examples, but they are not the TileRT scheduler/compiler or
a portable backend implementation.

## Where the underlying ideas could help

### Pretraining: TileLang rather than TileRT

The strongest positive evidence is the existing KDA forward-plus-backward
operator benchmark at batch 64, sequence length 1,024, three heads, and head
dimension 128 on the RTX 5090:

| KDA backend | Recompute | Median time | Relative result |
|---|---:|---:|---:|
| TileLang | disabled | 4.245968 ms | 14.37% faster than Triton |
| Triton | disabled | 4.958480 ms | reference |
| TileLang | enabled | 4.825248 ms | 12.74% faster than Triton |
| Triton | enabled | 5.529808 ms | reference |

The source artifact is
[`ablation_results/kda_training_benchmark_b64/operator_benchmark.json`](../ablation_results/kda_training_benchmark_b64/operator_benchmark.json).
It records the device, CUDA, PyTorch, FLA, TileLang, and Triton versions as
well as numerical comparisons. The post-training stack already observes FLA's
TileLang JIT dispatch in
[`postraining/train_latent_vapo.py`](../postraining/train_latent_vapo.py).

Do not extrapolate the operator percentages to an entire update or to H100.
TileLang is intended to be numerically equivalent to the Triton backend, so it
is not expected to improve fixed-step BPB. Under the current project protocol,
a retained change must improve BPB by more than 0.005 at 2,000 steps. Preserve
this operator result as exploratory systems evidence, but do not promote or
keep a TileLang substitution on speed alone. If the project explicitly adopts
a systems-only exception in the future, an end-to-end backend decision would
still require identical-seed, production-shape runs measuring update time,
peak memory, numerical parity, loss trajectory, and matched-wall-clock BPB on
both the RTX 5090 and H100.

### Post-training: measured room, current alternatives rejected

The recorded performance audit attributes 89.1% of train-plus-collect time to
rollout collection, scoring, and behavior refresh, versus 10.9% to actor and
critic updates. See
[`postraining/PERFORMANCE_AUDIT.md`](../postraining/PERFORMANCE_AUDIT.md).
Inference acceleration can therefore materially affect post-training wall
time.

The current rollout implementation already pursues the relevant mechanisms:

- dense prefill and incremental caches;
- survivor compaction and continuous refill;
- compiled one-token steps;
- FlexAttention decoding; and
- optional CUDA-graph capture over declared decode shapes.

Profiling recorded about 348 host launches per six-layer decode step, 157,000
kernels averaging 1.9 microseconds, and 32% of pool wall time with no resident
kernel. If all of that idle fraction disappeared without adding work, the
ideal ceiling would be about 1.47x for the affected pool and about 1.40x
overall when combined with the 89.1% collection share. This is an upper bound,
not a forecast.

The tradeoffs were subsequently resolved by full-workload A/Bs recorded in
[`NOTES.md`](../NOTES.md):

- production-shape Flex decode changed decode time per action by only -0.9%,
  inside control variance and consistent with an earlier -2.3% result;
- lockstep scheduling delivered about 93,000 useful actions per second versus
  72,000 to 80,000 for continuous refill, making lockstep 15% to 25% faster
  while using 35% less VRAM; and
- CUDA-graph decode improved the tested continuous scheduler by only 5% to 9%
  while consuming another 9 GB of VRAM, so it was rejected.

Keep Flex and graph decode off and retain lockstep as the KDA default. Revisit
continuous refill only if production trajectories become sufficiently long
and ragged to starve lockstep chunks. Revisit graph decode only if its memory
cost falls materially or a new model shape changes the launch-versus-dead-work
tradeoff. Do not rerun the same A/Bs without one of those changed conditions.

Multi-token prediction is not a free TileRT feature that can be attached to
these models. It needs a trained draft/MTP path, verified acceptance behavior,
sampling-equivalence work, and artifact-space accounting. Treat it as a
separate model and inference ablation if it ever becomes competitive with the
simpler cache/fusion work.

## Recommended experiment order

1. Do not install or port TileRT.
2. Preserve the existing TileLang operator result, but do not promote the
   backend under the current protocol unless it improves 2,000-step BPB by
   more than 0.005.
3. Keep KDA post-training on lockstep with Flex and graph decode disabled. Do
   not repeat the completed scheduler/backend A/Bs unless their recorded
   revisit conditions become true.
4. Profile the current production workload before proposing another systems
   experiment; historical hotspots may no longer dominate the active lineage.
5. Write a custom TileLang or Triton kernel only after current profiling
   identifies a remaining operation responsible for at least roughly 10% of
   end-to-end time. Component microbenchmarks alone are insufficient.

All GPU experiments must be submitted through `mlq` with
`--max-parallel-runs 1`.

## Revisit criteria

Reassess this decision only when at least one of the following changes:

- TileRT publishes an H100/Hopper backend that is demonstrated on 8xH100;
- TileRT publishes an RTX 5090/SM120 backend or PTX-capable portable wheel;
- TileRT opens the runtime/compiler source and documents custom-model backend
  authoring;
- TileRT adds backward/autograd and distributed training support;
- a supported TileRT architecture becomes the actual deployment target; or
- an independent benchmark demonstrates a transferable benefit on small,
  high-batch models rather than batch-one, trillion-scale MoE decode.

On reassessment, repeat binary architecture inspection and check upstream
issues rather than assuming a newer version removed these constraints.

This assessment used read-only upstream inspection and existing local
artifacts. It did not launch a new GPU workload.

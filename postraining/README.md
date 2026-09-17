# Post-training

The historical round-5 base is
`postraining/runs/sft_v3_answer_hfonly/sft_final_model.pt` (the three-epoch
round-5 trace-SFT run selected by its sampling gate in `NOTES.md`). The
underlying pretrained checkpoint is
`logs/k3_quality_20k_ctx8k_final_model.pt`: the measured `KKKDKKKD` KDA
skeleton trained through the 2K -> 4K -> 8K context curriculum. The selected
SFT checkpoint retains that architecture/context metadata and records its
trained `<think>`/`<answer>` tokenizer contract. It predates the canonical
single-prompt schema and is valid only for already-running legacy-schema
jobs. The next canonical replacement must be trained at
`postraining/runs/sft6_bare_a1swap10k_e3/sft_final_model.pt` from the immutable
SFT6 corpus; prompt-schema guards deliberately reject old checkpoints under
the current code.

The older 6-layer dense `mathmix_v4` checkpoint was superseded because its
four-source corpus was narrow, its 2,000-step run was too short, and it did
not provide KDA's long-context memory scaling. Selection of any replacement
base still requires:

1. the 2,000-step data/architecture gates complete;
2. the winning KDA recipe completes the 8,000-step curriculum;
3. the KDA post-training adapter strict-loads the checkpoint and matches dense
   prefill/decode logits;
4. FineWeb plus web/code/math/knowledge validation and the frozen-policy
   rollout gate pass.

The KDA adapter for gate 3 exists: `postraining/kda_backbone.py`
(`NanoKDABackbone` over the import-safe `pretraining/nanogpt_mini/nanogpt_mini_kda_model.py`) loads
`*_kda_*` checkpoints through `model_io.load_model` with zero key mapping.
Dense MHA layers keep KV caches; KDA mixers carry
`(conv_q, conv_k, conv_v, state)` decode caches — the delta-rule state is
always fp32. Rollout decode is a pure-PyTorch recurrence (stays inside the
fullgraph-compiled step artifact); prefill and teacher-forced replay dispatch
to FLA's `chunk_kda` on CUDA (the mixer is an eager region, so the replay
artifacts compile with graph breaks — set automatically). The
continuous-refill scheduler is KV-address machinery and is refused for KDA;
use lockstep. CPU parity/integration tests live in
`postraining/tests/test_kda_backbone.py`; the CUDA parity gate
(kernel-vs-reference, dense-vs-decode logits, left-pad invariance) is
`postraining/kda_gpu_parity.py`, run through mlq.

## MiniCPM5 native-token VAPO

`train_minicpm_vapo` defaults to standard-token VAPO, with opt-in continuous latent
thinking described below. It does not change the nano, latent-thinking, or OPSD
trainers. It defaults to the pinned final
`openbmb/MiniCPM5-1B` RL+OPD checkpoint because its existing math success rate
provides useful sparse-reward variation and its post-training reduces overlong
responses. The checkpoint remains a native `LlamaForCausalLM`: its tokenizer,
thinking chat template, attention blocks, GQA layout, and Transformers KV cache
are used directly.

Actor and critic use separate rank-16 LoRA adapters and trainable heads over
shared immutable MiniCPM embedding and transformer storage. The critic has its
own 256-wide scalar value head and never consumes actor hidden states. Its
unused 130,560-way output projection is removed. Only adapters, value head, and
auxiliary heads are serialized.

Fresh MiniCPM VAPO runs default to NoRA initialization: normalize each column of
the FP32 LoRA A matrix once at construction, leave B zero, and retain alpha/rank
scaling. This preserves the initial base-model outputs; trained adapters are never
renormalized on forward or checkpoint load. Both actor and critic use this
initialization with independent trainable storage. A **fresh run has no `--resume`**:
it starts new adapters, optimizers and critic warmup from the pinned native base.
Continuation preserves its checkpoint's initialization; missing legacy metadata
means standard LoRA, not NoRA. Use the continuation preflight rather than changing
that label.

NoRA is the experimental fresh-run default, not a proven quality winner. The
existing ten-step gate has finite optimization metrics but an incomplete matched
standard-LoRA control and severe 4K-response truncation; the named 2K comparison
did not complete. The implementation/resume review and focused tests qualify a
fresh experiment, not empirical default adoption under the longer-run criteria.
Review evidence and the fresh 10K, v3-attention run:
[NoRA readiness and fresh run](../ablation_results/minicpm5_vapo_nora_fresh_20m_20260910/result.json).
That fresh 20-minute run completed ten critic-warmup cycles and saved actor step
one (four actor and 44 critic optimizer steps) with finite checkpoint tensors.
The two actor-phase rollouts measured 8,703 and 9,038 useful tokens/s, but 59.4%
and 92.2% of responses reached the 10K cap. The second update was interrupted by
the time limit. This confirms the integrated path runs; it does not establish
learning improvement or acceptable task-quality/truncation.

Fresh runs reserve **1,000 answer tokens inside the existing 10,000-token response
cap** (`--answer-reserve-tokens 1000`). If the response has not naturally emitted
the native `</think>` token, rollout inserts it as response token 9,000, leaving
up to 1,000 subsequent tokens for the answer. Earlier natural thinking closure
or EOS is unchanged. The delimiter counts toward the total cap; no additional
KV capacity is required. Set the reserve to zero to disable budget forcing.

The forced delimiter stays in replay context and value/advantage computation,
but is excluded from PPO likelihood, policy-KL diagnostics, and sampled-action
normalization. It is an environment intervention, not a policy action or an
episode termination. `rollout_quality/forced_thinking_trajectories` reports its
use; truncation still means the response failed to emit EOS before the total
cap. Speculative decoders discard the verified suffix after an injected
delimiter and continue from the corrected context.

Changing the answer reserve on continuation requires a completed rollout
boundary with no pending replay records. Continuation preflight preserves the
saved reserve, treating missing legacy metadata as zero; it never silently
relabels pending unforced generations. Budget forcing is a compute-budget
contract, not a demonstrated accuracy improvement.

**Opt-in sampled-token hidden carry:** use
`--token-carry --no-train-nextlat --answer-reserve-tokens 0` on a fresh
`postraining.train_minicpm_vapo` run. Native token-only remains the default;
top-k 20, temperature 0.9, top-p 0.95, and the optimized BF16 rollout backend
are unchanged.

After sampling token `x` from the vocabulary head, the next input is
`E(x) + sigmoid(g) * (Wd E(x) + Wh stopgrad(h))`, where `h` is the previous
final normalized hidden state, immediately before the LM head. The two
bias-free projections are implemented without allocating a concatenation.
Both matrices start at zero; the learnable scalar gate starts at **0.01**.
The pretrained embedding bypass remains intact, and the gate attenuates both
new residual branches (4,718,593 parameters per combiner). The gate initially
has zero gradient while the residual is zero; it learns once a residual develops.
Prompts use plain embeddings. Every generated token uses the carry path,
including ordinary sampled thinking delimiters and EOS. There are no Gaussian
actions, latent slots, additional noise, or learned stopping gates.

Rollout stores the actor's **behavior-time producer hidden** alongside each
sampled token as detached BF16 data. Actor and critic consume that identical
observed stream through **independent gated combiners over their token embeddings**. Detach
is before each trainable combiner, not after the combined embedding. The critic
does not generate its own carry trajectory, consume an actor-combined embedding,
or backpropagate its value loss into the actor. Its state-value objective is
unchanged; prompt embeddings remain plain.

Behavior refresh, PPO updates, and post-update KL each use the ordinary parallel
packed forward. Updating either model changes its projections and predictions,
not the stored observations. There is no sequential teacher-forced reconstruction,
extra replay KV cache, new attention mask, or special attention backend. This is
the deterministic-carry contract already established in commit `ac2d67c`.

Fixed-batch and continuous/refilled captured rollout retain exact producer
histories in logical response order. CPU exports own their storage and survive
refill/cache release. This mode requires compiled fast rollout and compiled
replay. Gaussian latent mode, Uno/NextLat proposals, auxiliary NextLat training,
and forced delimiters remain rejected for this isolated experiment.
Checkpoints use `minicpm5_vapo_token_carry/v3`; both residual projections and
scalar gates, optimizers, pending tokens **and stored carries**, and RNG state
are saved. Old v1/v2 checkpoints are rejected: v1 lacks the stored observations,
and v2 uses the ungated identity-initialized token projection. Continuation
preflight supports v3; stock native-only evaluation rejects carry checkpoints
instead of dropping their inputs.

Bounded correctness verification (not a quality evaluation):

```bash
.venv/bin/python scripts/diagnose_minicpm_token_carry.py \
  --output ablation_results/minicpm_token_carry_check/correctness.json
```

The entrypoint submits through `mlq` with parallel limit 1, normal priority,
a five-minute cap, and one attempt. It checks identity initialization, stored
producer persistence/refill, independent critic gradients, fixed replay inputs,
and checkpoint restoration. Earlier v1 diagnostic artifacts concern the
superseded reconstruction design, not the stored-carry training contract.

Historical ungated v2 run **7642**, now cancelled, used 64 trajectories per
rollout with the full 10,000-token response cap, ten critic-warmup cycles, and
native packed replay with activation checkpointing disabled. The first actor-phase
rollout produced 534,081 tokens at **8,490 useful tokens/s**; its complete
actor/critic update took **53.4 s**. A steady critic-warmup update processed
640,000 actions in **29.9 s**. The real replay profile confirms
`aten::_scaled_dot_product_flash_attention` rather than masked non-Flash SDPA.
After actor step one, both token and carry matrices had moved independently
in the saved actor and critic; all four matrices were finite.

Historical evidence:
`ablation_results/minicpm_token_carry_stored_train_20260916/result.json`,
`metrics.jsonl`, `packed_replay_kernels.txt`, and `vapo_adapter_checkpoint.pt`.

That run collapsed: its last five recorded rollout accuracies were zero.
Finite parameters, improving value loss, and throughput did not establish
successful learning.

Behavior refresh averaged **24.06 s**, or **15.04%** of rollout + refresh +
update time across its first nine actor iterations. The continuous rollout
exports placeholder log-probabilities, so refresh must materialize actor
likelihoods as well as independent critic values for GAE. The candidate
refactor retains both passes and transfers the two scalar statistics per
action to CPU once per rollout rather than twice per trajectory. GPU buffers
cost eight bytes per response action; likelihood and advantage semantics are
unchanged. Speedup is pending a paired production-data benchmark.

Fresh gated run **7679** is queued at normal priority, parallel limit one,
with a two-hour cap and one attempt. It uses the same full response geometry
and performs an alternating original/candidate refresh benchmark on its first
real actor rollout, checking bit-identical likelihoods and advantages.
Evidence is written to
`ablation_results/minicpm_token_carry_gated_train_20260916/`.
Gated quality and GPU throughput are not yet established. Host coverage passes
192 focused tests; the real-tiny-Llama CPU model test was excluded.

When training starts, the linked TensorBoard run is available at
`http://127.0.0.1:6101/?runFilter=minicpm_token_carry_gated_train_20260916#timeseries`.
The server on port 6106 lists the same run with `/tensorboard` appended.
Both servers rescan every 180 seconds. `carry/actor_gate` and
`carry/critic_gate` report the learned gates. Behavior refresh also reports
`carry/{actor,critic}_probe_carry_to_token_rms` and
`carry/{actor,critic}_probe_residual_to_token_rms`, using at most 256 evenly
spaced carry inputs from the first nonempty packed shard, not the whole rollout.

**Opt-in latent thinking:** add `--latent-thinking` to a fresh MiniCPM run.
The default is false; `--no-latent-thinking` explicitly retains native token
generation. This mode uses the existing stochastic latent policy:

```
native <think> prefix → first latent → continue latents → </think> → answer tokens
```

The first continuous thought is mandatory. Thereafter a separate Bernoulli
head chooses continue or stop. Inside the block, a Gaussian transition head
produces raw fp32 vectors centered on the current hidden state plus a learned
residual. `--thought-sigma` defaults to 1.0 and is the vector-level noise scale:
component standard deviation is `thought_sigma / sqrt(hidden_size)`.
`--init-stop-thinking-probability` defaults to 0.9; this initializes the learned
gate and cannot bypass the first thought. An identity-initialized combined
embedding adapter feeds each vector into the shared MiniCPM transformer.

Thought and close steps never execute the vocabulary head. Stopping consumes
the native close token before ordinary answer generation; answer sampling
excludes the two thinking delimiters, so the block cannot reopen. The total
`--max-new-tokens` budget counts latent slots, the close, and answer tokens.
The answer reserve retains its meaning; with reserve zero, latent mode still
forces closure early enough to leave one answer slot.

**The critic sees every exact sampled latent vector**, through its own
independent trainable thought adapter and transformer adapters. It predicts
values before each action, including states reached after thoughts, and all
latent transitions participate in advantage computation and value training.
Replay never redraws noise or substitutes actor hidden states for critic
states. PPO scores first-thought Gaussian, continue gate plus Gaussian, learned
stop gate, and answer-token actions; forced stops have no actor likelihood.
NextLat's lexical auxiliary trains only on answer-to-answer transitions in
this mode. `latent_thinking/*` TensorBoard metrics separate thought counts,
answer-token counts, and learned stops from total stream length.

Latent checkpoints use `minicpm5_vapo_latent/v1` and store both thought adapters,
actor Gaussian/gate heads, exact pending raw actions, optimizer state, and RNG
state. Native v6 checkpoints remain unchanged; switching reasoning modes on
resume is rejected. Continuation preflight supports latent checkpoints.
The standalone native HF evaluator rejects them rather than silently dropping
latent heads; `--rollout-only --latent-thinking` uses the actual latent runtime.
Token-only Uno proposals are incompatible with latent mode.

**Latent training is not numerically qualified.** Short-stream execution checks pass.
CUDA generation runs for host-driven latent, graph-chunk latent, and native
decoding. Actor and critic replay pass exact raw-action, independent state
reconstruction, and checkpoint-gradient parity checks under deterministic
controls. Default nondeterministic backward varies by about 1.4% in relative L2
even between identical runs. Separately, bf16 rollout and packed replay produce
different Gaussian means: one measured first-thought state has mean-distance
4.53 at component sigma 0.0255, or approximately 15,774 nats of Gaussian KL.
Batch shape and projection fusion affect this difference; toggling replay MLP
compilation did not. Disabling reduced-precision bf16 GEMM reductions also failed
to resolve it (approximately 15,700 nats); that setting has not been adopted.
The 17-position PPO/NextLat update completes with finite trainable parameters,
but latent ratio mean/std and approximate KL still overflow fp64 after updates.
At 2,177 stream positions, actor and critic replay retain exact checkpoint parity,
but the integrated PPO update fails with a non-finite primary gradient.
The saved-trajectory diagnostic starts with ratios exactly one, then reaches
unclipped negative-advantage objectives around `exp(995)` after one optimizer
update. Reordering the existing PPO clipping cannot resolve this divergence.
Do not treat this mode as validated for training or interpret refreshed age-zero
PPO ratios as proof of sampling/replay distribution agreement. The benchmark's
checkpoint-gradient checks use a scoped deterministic CUDA workspace and restore
the original settings before timing. A passed execution benchmark does not
establish Gaussian distribution agreement or training validity.
Evidence and numerical controls are in
`ablation_results/minicpm_latent_performance_20260911/`.

PPO applies its existing sign-dependent clipping before exponentiation, avoiding
spurious NaN gradients on already-clipped branches. Unfavorable unclipped ratios
remain unbounded. Ratio moments are accumulated in log space; KL diagnostics use
fp64 `expm1` to avoid fp32 overflow and near-zero cancellation.

The latent decoder uses a fused bf16 replica, compiled embedding-only trunk,
separate phase heads, and continuous physical-lane refill. It requires CUDA/FA4.
Phase and position state stays on-device inside adaptive chunks of up to eight
steps. Each graph has fixed thinking/answer membership, so thinking rows never
enter the vocabulary projection. A stopped row consumes the close token, then
pauses with its KV prefix intact until answer admission at the next boundary.
Exact fp32 actions and scores use bounded GPU/pinned-host staging and transfer
once per chunk, not once per thought. Prefix KV remains host-backed.

The graph cache is bounded; graphs are recaptured for each public rollout
because KV and source-weight residency changes invalidate their addresses.
Public-generation benchmark timing includes that capture cost. Replay collation
precomputes token/gate indices and contiguous NextLat ranges on the CPU; packed
replay avoids device-side index discovery and a full embedding-buffer copy.

`scripts/benchmark_minicpm_latent.py` compares fixed stream-position workloads
for optimized latent and native decoding. `--mode all` also measures actor/critic
replay and warmed rollout → behavior refresh → optimizer cycles, including the
NextLat auxiliary. These synthetic cycles exclude data loading, math verification,
logging, and checkpoint I/O; they are not accuracy or learning evidence.
An optional `--engines host optimized native --host-reference PATH` adds a
preserved pre-chunking runtime. `--expected-sources PATH` rejects source drift
while queued. Reports include exact raw-action sidecars, phase counts,
original-versus-replayed likelihood drift, graph/staging telemetry, wall time,
CUDA time, and peak memory. Integrated-update and core-cycle failures are retained
without discarding independent arms; the overall status and exit code remain
failed. Core cycles are blocked when their update qualification fails.
Run every benchmark through `mlq`, for example:

```bash
mlq submit --name minicpm-latent-qualification --max-parallel-runs 1 -- \
  .venv/bin/python scripts/benchmark_minicpm_latent.py --mode qualify \
    --prompts 2 --samples-per-prompt 2 --physical-batch-size 3 \
    --context-tokens 256 --thought-steps 8 --answer-tokens 8 \
    --output ablation_results/minicpm_latent_qualification.json
```

CPU regressions establish phase, replay, gradient, and checkpoint contracts;
they do not establish real CUDA capture compatibility, speedup, or math quality.
Native AR throughput thresholds are not evidence of latent throughput: qualify
the configured workload and set its minimum-throughput threshold from that
measurement. Matched moderate- and long-context workloads are needed to separate
head-bypass gains from attention/KV bandwidth limits.

**Measured RTX 5090 costs (2026-09-11).** Medians in seconds for 64
trajectories, 64 physical rollout lanes, and replay batch size 1. Latent rollouts
add one close token and 128 answer positions; native performs lexical work at
every matched stream position.

| Workload | Native | Previous host latent | Graph-chunk latent |
| --- | ---: | ---: | ---: |
| Rollout: 1,024 context + 2,048 thoughts | 9.68 | 13.74 | 11.49 |
| Rollout: 8,192 context + 1,024 thoughts | 12.87 | 17.76 | 16.14 |
| Actor likelihood replay: 2,177 action positions | 8.36 | — | 7.15 |
| Critic value replay: 2,177 action positions | 6.82 | — | 7.05 |
| Core cycle: 2,177 action positions | 32.89 | Blocked | Blocked |

The 2k rollout control used four warmups and seven measurements with reversed
engine order; the other rows used two warmups and three measurements. Replay rows
isolate forward/backward and include CPU collation/H2D, not optimizer or NextLat
work. The core row includes actual PPO/NextLat updates and inference-weight
refresh. Initial engine construction is measured separately and excluded.
One graph-chunk rollout outlier reached 23.47 seconds (22.31 in decoding) despite
unchanged work/graph counters. All samples are retained: the 2k median throughput
gain over host latent
is 19.6%, but aggregate throughput over all seven measurements improves only 4.0%.
Native remains faster, and no latent full-cycle throughput is claimed.
Canonical results, raw records, numerical diagnostics, and source manifests:
`ablation_results/minicpm_latent_performance_20260911/result.json`.

Rollout remains actor-only. After generation, length-bucketed teacher-forced
actor and critic passes materialize replay-consistent behavior-policy
log-probabilities, fixed behavior values, and advantages. Host replay stores
int32 token ids plus fp32 selected-token log-probabilities and advantages,
never full logits. Replay reconstructs selected-token probabilities in
128-token output-head chunks under a padded-token budget.

Each rollout collects four distinct prompts with sixteen responses apiece. The
default gives all 64 logical trajectories one physical GPU lane in one KV
cache. `--rollout-physical-batch-size` can benchmark fewer continuously
refilled lanes without changing the logical rollout. With four optimizer
minibatches, each actor and critic step consumes a disjoint 16-trajectory
quarter of the rollout.

One PPO epoch uses four true optimizer minibatches. Each contains a disjoint
quarter of the rollout; length-bucketed replay batches are only memory shards
whose gradients accumulate inside that optimizer minibatch. Actor and critic
therefore each take four AdamW steps per rollout. Their default learning rates
are 1e-6 and 2e-6 respectively, matching VAPO's actor/critic scale. Fixed
behavior log-probabilities make KL, ratios, and clipping meaningful after the
first minibatch. Exact post-update behavior KL is measured every ten rollouts.

Actor and critic each own a canonical residual NextLat dynamics MLP. It consumes
the current hidden state and next-token embedding, predicts the next hidden
state, and uses Smooth L1 plus categorical KL against detached targets. Up to 64
response transitions per optimizer minibatch are distributed across its memory
shards, keeping auxiliary cost fixed when replay sharding changes. On each side,
the NextLat loss is downscaled to at most the policy or weighted-value loss
magnitude. Its transformer gradient is measured separately and capped to the
primary objective's parameter-gradient norm before the two are accumulated, so
NextLat cannot dominate the LoRA update even when the losses have different
conditioning. Actor and critic parameter gradients are then clipped
independently to norm 1.0 before each step. Speculative decoding is not part of
this training path.

The default rollout path keeps a fused inference-only actor replica resident on
the GPU. QKV and gate/up projections stay fused, and one decode step—including
top-k sampling and replay-buffer writes—is captured as a CUDA graph.
Left-padded prefill K/V is compacted into per-row contiguous prefixes. The four
unique prompt prefixes remain in a device-resident bank and expand into their
sample lanes without a host round trip. Fixed-shape FA4 varlen decode reads the
persistent sequence-major cache using device-resident sequence lengths; once a
lane finishes, its visible KV length drops to one so later captured steps do not
scan dead history. No critic model or critic KV cache runs during autoregressive
decoding. The actor replica weights remain resident across rollout and replay,
while phase-local KV state is released before replay. Replay uses an
11,024-token packed budget and stable segmented SDPA. Activation checkpointing
defaults to off (`--replay-checkpoint-interval 0`): retaining activations avoids
recomputing decoder layers during backward. A 16-trajectory, 160k-response-token
actor/critic optimizer-minibatch comparison measured 19.07→17.60 seconds (8.4%
higher throughput), with peak allocation 16.77→20.22 GiB. This benchmark did not
include a resident rollout replica or an entire RL cycle; reserve additional
memory for those persistent weights and other workloads. Use
`--replay-checkpoint-interval 4` when memory is tighter. Commands explicitly
pinning an interval keep that setting; changing this option is resume-compatible.
Replay MLP compilation is also enabled by default (`--compile-replay`).
It preserves bf16 casts and keeps SiLU backward native: ordinary compiler
decomposition changed replay gradients despite identical forward losses.
The integrated long-trajectory update fixture measured 17.73→16.87 seconds
(5.1% higher warm throughput) and 20.22→18.16 GiB peak allocated memory.
These are isolated update measurements, not cold-start or full-cycle numbers.
Compilation preserves parameter names and replica ownership, and supports the
separate retained-graph primary/auxiliary backward passes. Use
`--no-compile-replay` for an explicit eager comparison; this option is
resume-compatible. Attention remains segmented SDPA.
The `--replay-attention-backend fa4` path remains available only for profiling;
SM120 FA4 varlen backward produced NaNs in production.

MiniCPM VAPO uses the checkpoint's native thinking template, temperature 0.9,
top-k 20, and top-p 0.95. The default 10,000-token response budget uses the
64-row static cache after offloading the frozen training backbone.
`--top-k 0 --no-fast-rollout` retains the slower exact full-vocabulary nucleus
sampler. Production aborts below `--min-rollout-tokens-per-second`: the AR path
uses steady scheduled decode tok/s; opt-in Uno uses useful end-to-end rollout tok/s.
Choose the Uno floor from a matched benchmark, not the AR scheduled-token value.

Compiled MiniCPM AR now defaults to compiler-visible FA4, in-place indexed KV
writes and fullgraph compilation with preserved bf16 casts. It retains ordinary
cuBLAS GEMMs and the ordinary AR scheduler; fixed invariant GEMMs are **not**
enabled. The runtime identity is `bf16-cublas-fa4-split4-m16n32-fp32-fullgraph-casts/v3`.
Explicit noncompiled ordinary decode retains the legacy path.

Ordinary optimized decode on SM120 uses packed-GQA FA4 `16x32` single-warp tiles
with four-way split-KV for single-query BF16 MiniCPM attention (16 query heads,
2 KV heads, head dimension 128, contiguous sequence-major KV). Invariant AR/Uno,
prefill and other attention geometries retain their existing paths. The adapter
reuses the pinned FA4 main loop; its output epilogue preserves FP32 partials
until FA4's FP32 merge writes the final BF16 output. Partition lengths and offsets
are recomputed on-device during every graph replay; no KV replication or host
length synchronization is needed.

Before split-KV, tuning the unsplit tile alone retained arithmetic identity:
The matched 10,000-token-cap comparison improved 4,041→6,506 useful tokens/s
(92.51→57.46 seconds), with all 64 response token sequences identical.
Integrated production qualification reproduced 6,497 useful tokens/s and those
same responses, plus bit-exact checked logits through continuation and refill.

Compact split-4 adds a measured throughput benefit in the draining tail, at the
cost of overhead at full occupancy. Earlier identical recorded-token replay
improved throughput 13.7% (56.13→49.36s). The larger native split-16 prototype
reached 16.5%, but the compact adapter avoids maintaining a fork of FA4's main
loop. Sources and original measurements:
[FA4 split experiments](../ablation_results/minicpm_fa4_split_20260909/result.json).

BF16 probability operands with FP32 accumulation are accepted mixed precision.
Split-KV changes the reduction order and can change sampled responses; it is
not bit-identical to unsplit FA4. Acceptance requires FP32-reference accuracy,
correct masking and cache/graph lifecycle, not old-kernel token identity.
Statistical math-quality neutrality is **not established** by the small sampled
comparisons. Integrated adoption evidence:
[split-4 qualification](../ablation_results/minicpm_fa4_split_adoption_20260909/result.json).

The original `64x64` split-4 adoption replayed identical 373,859 tokens and 640,000
scheduled lane-positions: tuned unsplit FA64 took 56.57s versus split-4 48.52s,
or 6,609→7,706 useful tokens/s (**16.6% higher throughput**). Natural split-4
generation emitted 310,203 tokens in 43.39s (7,149 useful tokens/s); its different
token workload is not a kernel-only speed comparison.
Five GPU reference/graph cases and 103 focused CPU tests passed. Full-decoder
checks covered continuation and retirement/refill; after cache release, an
actor update and recapture, the reused replica's checked logits matched a fresh
replica exactly. This does not claim equivalence to unsplit sampled responses.

Further attention scheduling experiments retained `16x32` tiles with 32 threads,
one pipeline stage and four splits. Smaller tiles reduce padding for eight packed
GQA query rows; split starts align to the 32-token K tile. On identical recorded
373,859-token work, integrated production improved 49.14→46.64s, or
7,608→8,016 useful tokens/s (**5.35% higher throughput**). The prototype rerun
measured 46.68s. Eight splits tied while doubling partial-output scratch; a direct
scheduler fork was slower, and fused preparation added only about 0.6% in single
runs, insufficient to justify another kernel. Existing preparation is retained.
No physical-lane, context-capacity, sampling or precision reduction was used.

Natural v3 generation emitted 320,124 tokens in 42.05s (7,613 useful tokens/s);
this different workload is not a matched speed comparison. Six GPU reference/graph
cases and 104 focused CPU tests passed, including strided queries and an FP32
partial-output cancellation regression. Full-decoder continuation and retirement/
refill checks stayed finite; maximum checked transformed-sampling TV versus v2
was 0.0690. After cache release, actor update and recapture, checked logits matched
a fresh replica exactly. This is accepted mixed-precision drift, not a statistical
math-quality guarantee. Sources, bounded jobs and measurements:
[attention scheduling qualification](../ablation_results/minicpm_attention_scheduling_20260909/result.json).

Qualification job 5711 used a step-370 actor and the 10,000-token cap: useful
throughput improved 1,251.95→4,164.31 tokens/s (3.33×), and the 64-response pool
fell from 253.53→89.78 seconds (2.82×). Peak allocation stayed about 20.48 GiB.
The whole comparison and cache/actor-refresh checks took 463.89 seconds.
Quality is **not established as unchanged**: optimized scored 22/64 versus
legacy 26/64 on only four distinct problems. See
[MiniCPM TODO and evidence](TODO_MINICPM5.md#standalone-ar-runtime--optimized-default-enabled).

Checkpoint metadata pins `rollout_arithmetic`. Pending legacy, unsplit-v1 or
split-v2 records cannot resume under v3; use a checkpoint at a completed-rollout
boundary to change arithmetic. Existing running processes do not switch modes.

TensorBoard is the only live metric stream. Semantic categories cover rollout
quality, rollout performance, refill efficiency, sampling, replay, actor,
critic, advantages, KL, ratios, clipping, gradients, auxiliary NextLat,
optimization time, and system telemetry. Every category is capped at twelve
charts. Configuration and correct/incorrect response samples are text
summaries. Scalar writers rely on TensorBoard's asynchronous flush interval
instead of synchronously flushing every progress callback.

Establish the pinned native checkpoint baseline before running the integrated
learnability gate:

```bash
mlq submit --name minicpm5_native_aime_baseline --cwd "$PWD" \
  --max-parallel-runs 1 --time-limit 4h -- \
  python3 -m postraining.eval_hf_math \
    --model openbmb/MiniCPM5-1B --suite aime_2024 --thinking \
    --samples-per-problem 4 --max-problems 30 \
    --prompt-tokens 1024 --max-new-tokens 4096 \
    --batch-trajectories 4 --no-autotune-batch-trajectories \
    --temperature 0.9 --top-p 0.95 \
    --output postraining/runs/minicpm5_native_baseline

mlq submit --name minicpm5_vapo_gate --cwd "$PWD" \
  --max-parallel-runs 1 --time-limit 4h -- \
  python3 -m postraining.train_minicpm_vapo \
    --rollout-only \
    --output postraining/runs/minicpm5_vapo_gate
```

Before resuming, run the CPU-only preflight. It validates the v6 schema,
dataset bytes, output ownership, and target step without loading either model
or reserving the GPU. It prints the complete `mlq submit` command; `--steps`
in the trainer is an absolute actor-step target, while the preflight also
accepts the less error-prone additional-step form:

```bash
python3 scripts/preflight_minicpm_vapo.py \
  --resume postraining/runs/source/vapo_adapter_checkpoint.pt \
  --output postraining/runs/minicpm5_vapo_continuation \
  --additional-steps 200
```

### Opt-in Uno diffusion-assisted rollouts

**Deferred by user veto until better hardware and explicit reauthorization.**
The standalone AR optimization above is independent of this deferred training.

Train the diffusion adapter **before RL**, then freeze it. The current actor—not
the frozen distillation teacher—remains the verifier at every rollout. Default
generation stays AR; enabling Uno requires all three explicit settings:
`--uno-rollout --uno-checkpoint RUN/adapter.pt --uno-block-size 4`.
Fast rollout and CUDA-graph capture must remain enabled.

Uno uses the explicit `bf16-lane64-n128-k32-casts/v1` numerical target. Fixed-tile
Triton projections retain bf16 inputs/outputs and fp32 accumulation; compiler casts
are preserved, and the FA4 custom operator keeps decoding in one complete graph.
Compilation failure is fatal rather than silently changing arithmetic.
Its matched AR reference uses the same arithmetic and recomputes
the last prompt token at the prefill/decode boundary. This is **not** a claim of
finite-precision equality to legacy AR or the new ordinary-GEMM optimized default.
Qualification requires exact serial/block logits, clean KV and transformed sampling
laws on the selected target, with legacy drift reported separately.

Current evidence (2026-09-08): 96 CPU tests and full-model CUDA numerical
qualification job5646 pass, including exact logits/KV/sampling laws before and
after actor refresh. Legacy raw-vocabulary maximum TV was0.03000/0.04575 on those
regression prefixes. Inspect **both stdout and stderr** when qualifying; earlier
probes hid Dynamo fallback warnings in stderr and are not valid runtime evidence.
Do not treat the temporary one-update adapter as learning or trained-speedup evidence.
Replacing functionalized KV scatters with in-place indexed suffix writes reduced
the identical33-cycle fixture from3.006s to0.769s (3.91×). This is a runtime-code
optimization, not evidence of a trained adapter beating AR.
The three-arm benchmark (job5647; B64/B4,4096-token cap,three timing repetitions)
measured4812/7712/7185 useful tokens/s for legacy AR/invariant AR/Uno respectively.
The one-update fixture achieved0.932× matched AR and1.493× legacy AR, so the
≥1.25× dual performance gate **failed**. Between96.9% and99.2% of responses were
cap-truncated; this is not a task-quality result. Keep production RL on default AR
until a properly distilled adapter passes the learning, quality and speed gates.
Canonical metrics and source fingerprints:
`ablation_results/uno_invariant_runtime_20260908/{metrics.jsonl,result.json}`.

`scripts/train_minicpm_uno.py` implements one paired clean/noisy backbone forward,
duplicated logical RoPE positions, the Uno block mask with compiled native
FlexAttention, and detached same-position teacher next-token distributions.
Noise is uniform over the **entire** vocabulary. The objective is full-vocabulary
probability L1 (2×TV), position-chunked with recomputation backward. Only independent
fp32-master diffusion LoRA trains; base/actor weights and the output head are frozen.
FA4 is used for causal rollout suffixes, **not** unsupported masked training backward.

Candidate MiniCPM defaults are rank48/alpha3072, all seven projections,
standard random-A/zero-B initialization, bf16 compute, sequence2048,
microbatch1 × accumulation64, lr1e-5, 2% token warmup, and
100M supervised tokens at block2 followed by 300M at block4.
These transfer hyperparameters are not a demonstrated MiniCPM optimum.
Input is an explicitly supplied offline reasoning corpus: local JSONL
`messages`, OpenThoughts `conversations`, `text`, or question/response pairs;
each `.txt` file is one document. DAPO questions alone are not a reasoning corpus.
The default document-hash heldout split is 1%; explicit heldout documents must
be disjoint from training. Prepared streams, tokenizer, source bytes, teacher,
optimizer/RNG/cursor, and curriculum identity are pinned for exact recovery.

```bash
# Point UNO_DATA at a real offline reasoning JSONL corpus.
mlq submit --name minicpm-uno-prepare --cwd "$PWD" --max-parallel-runs 1 -- \
  .venv/bin/python scripts/train_minicpm_uno.py \
  --data "$UNO_DATA" --output postraining/runs/minicpm_uno --prepare-only

# Start only after numerical/runtime feasibility checks; this is a 400M-token pilot.
mlq submit --name minicpm-uno-distill --cwd "$PWD" --max-parallel-runs 1 -- \
  .venv/bin/python scripts/train_minicpm_uno.py \
  --data "$UNO_DATA" --output postraining/runs/minicpm_uno

# Same immutable arguments plus --resume postraining/runs/minicpm_uno/latest.pt
# recover training. A completed curriculum also exports adapter.pt.
```

For an already-trained RL actor, add `--teacher-checkpoint ACTOR_CHECKPOINT`
to distillation and retain that same teacher on recovery. No teacher checkpoint
means the pinned native MiniCPM base, which also equals a fresh zero-B actor.
Recovery checkpoints are atomic at completed optimizer boundaries; SIGINT/SIGTERM
finish the current accumulated update before saving. Distillation writes
`metrics.jsonl` and `tensorboard/` beneath its output directory, including heldout
L1/TV, token throughput and gradient norms.

Before enabling RL, compare Uno against all three AR controls on the same trained adapter and actor:

```bash
mlq submit --name minicpm-uno-compare --cwd "$PWD" --max-parallel-runs 1 -- \
  .venv/bin/python scripts/benchmark_minicpm_uno.py \
  --uno-checkpoint postraining/runs/minicpm_uno/adapter.pt \
  --output postraining/runs/minicpm_uno/comparison.json
```

The benchmark defaults to eight prompts × sixteen responses over 64 physical lanes,
10,000 response tokens, and the production temperature/top-k/top-p. Add
`--actor-checkpoint` for a resumed actor; use `--prompts 4` for a matched 64-logical-row
comparison without backlog. The default `--engine all` runs separate-process
`legacy-ar`, `optimized-ar` (production), `ar` (invariant) and `uno` arms,
reclaiming graph/compile memory between methods. Warmup is excluded and useful
emitted tokens are the numerator. Uno must exceed all three AR controls by
≥1.25×; a slower numerical reference cannot inflate the qualification claim.
The JSON also records logical-pool wall time, truncation, memory, occupancy,
per-position acceptance and checkpoint/data/prompt identity. It does **not**
prove distributional correctness, math quality, full RL-cycle speedup or compute payback.
For AR-only work use `--engine optimized-ar`, `--engine legacy-ar` or `--engine ar`
without `--uno-checkpoint`. `--cache-length` specifies the cache capacity separately
from the output cap. Answer grading uses the trainer's existing non-symbolic grader
outside the generation timer; `--export-responses` also saves text and token IDs.

Append the Uno settings to the normal VAPO command only after those gates pass:
Set `UNO_USEFUL_TPS_FLOOR` to a measured useful-token floor; do not copy the default
4,000 scheduled-token AR floor into an Uno campaign.

```bash
mlq submit --name minicpm-vapo-uno --cwd "$PWD" --max-parallel-runs 1 -- \
  .venv/bin/python -m postraining.train_minicpm_vapo \
  --uno-rollout --uno-checkpoint postraining/runs/minicpm_uno/adapter.pt \
  --uno-block-size 4 --min-rollout-tokens-per-second "$UNO_USEFUL_TPS_FLOOR" \
  --output postraining/runs/minicpm_vapo_uno
```

The inherited continuous scheduler commits only clean verified KV, keeps the final
correction/bonus token pending, and truncates at the first EOS/output limit.
Sparse coupling uses the exact production top-k-then-nucleus law; PPO likelihoods
still come from untempered actor replay, never draft/residual probabilities.
Uno checkpoints are separately stored and SHA-256-pinned in RL recovery.
Pending Uno replay also pins the arithmetic version; changing it requires a
completed rollout boundary. AR↔Uno/block-size changes likewise require a completed
rollout boundary. Relocating identical adapter bytes at a completed
rollout boundary is allowed;
silently replacing them on an Uno resume is not. Acceptance can deteriorate as
RL changes the actor; rebenchmark later checkpoints instead of assuming rank bounds drift.

Opt-in full-model numerical regression (not a learning or speedup trial):

```bash
mlq submit --name minicpm-uno-contracts --cwd "$PWD" --max-parallel-runs 1 \
  --time-limit 45m --env RUN_UNO_CUDA_VALIDATION=1 -- \
  .venv/bin/python -m pytest postraining/tests/test_uno_cuda.py -x -v -s
```

See `docs/uno_diffusion_augmented_assessment.md` for paper/release provenance,
the evidence gates, and measured versus still-unmeasured claims.


Run every GPU workload through `mlq`. Before a new training campaign, run the
frozen-policy learnability gate:

```bash
.venv/bin/python -m postraining.prepare_sft6_bare

mlq submit \
  --name sft6_bare_a1swap10k_e3 \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  .venv/bin/python -m postraining.sft_trace_train \
    --name sft6_bare_a1swap10k_e3 \
    --checkpoint logs/k3_quality_20k_ctx8k_final_model.pt \
    --traces postraining/data/sft_traces_v6_answer_bare_a1swap10k.parquet \
    --epochs 3 \
    --think-tokens \
    --answer-fence
```

The migration removes only SFT5's legacy prompt suffix. It preserves every
completion byte, row order, source assignment, and label, and publishes a new
hash-bound artifact rather than editing SFT5 in place. Run the following gate
only after that SFT6 job completes successfully:

```bash
mlq submit \
  --name posttrain_base_gate \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  python3 -m postraining.train_latent_vapo \
    --checkpoint postraining/runs/sft6_bare_a1swap10k_e3/sft_final_model.pt \
    --output postraining/runs/posttrain_base_gate \
    --reasoning-mode cot \
    --think-tokens \
    --answer-fence \
    --rollout-groups 4 \
    --rollout-only
```

Then start the selected 40K DG broad-mixture regime with an explicit output
directory. Its objective, data mixture, exact-only reward, fenced thinking,
and 24 x 16 fresh-batch topology are defaults:

```bash
mlq submit \
  --name posttrain_dg_broad \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  python3 -m postraining.train_latent_vapo \
    --checkpoint postraining/runs/sft6_bare_a1swap10k_e3/sft_final_model.pt \
    --output postraining/runs/posttrain_dg_broad
```

With `--answer-fence`, every math family is canonicalized to the same episode
prompt before SFT, RL, or evaluation:

```text
{bare problem}
```

DAPO, DeepMind, AIME, and GSM8K source wrappers are removed rather than
rewritten one-for-one. No prose reminder replaces them: `<think>` and
`<answer>` are completion tokens learned by SFT and enforced structurally by
reward/evaluation. The prompt schema is recorded in SFT and RL checkpoints and
enforced on initialization and exact resume; regenerate the canonical SFT
corpus and retrain SFT when this schema changes.

## Latent thinking: forced-initial stochastic policy

`--reasoning-mode latent` uses execution schema v29. Every trajectory takes
one mandatory continuous `THINK` action before its first emitted token. After
that action, a learned Bernoulli gate chooses `CONTINUE_THINK` or
`STOP_AND_EMIT`; stopping is irreversible, so all later actions are emitted
tokens. `cot`, `none`, and OPSD `pin_emit` rollouts bypass the gate, noise, and
thought slots completely.

The Gaussian transition is centered on the deterministic latent policy's
current belief plus a learned zero-initialized residual. It samples one raw
fp32 action, so exploration does not replace the established latent direction
with a fresh random projection. That exact vector is stored once and reused
by live decode, actor replay, and the separate critic; replay never redraws
noise. `--thought-sigma` is the isotropic vector-magnitude scale, not a
per-component standard deviation. For runtime embedding width `d`:

```
component_std = thought_sigma / sqrt(d)
E[||noise||^2] = thought_sigma^2
```

Raw actions are adapted through the current combined-embedding stack before
entering the trunk. `--combined-mlp-hidden` (default 2048) and
`--combined-mlp-blocks` (default 1; 0 keeps only the residual adapter) retain
the current combiner geometry. `--init-stop-thinking-probability` explicitly
sets the fresh gate's STOP bias; it never removes the mandatory first thought.

The standard clipped VAPO arm is selected with
`--no-delightful-policy-gradient`. DG and TPO retain their current token
objectives and add the scored gate/Gaussian factors without changing their
selection flags. Checkpoint, manifest, replay, evaluation, and exact-resume
schemas are strict; deterministic-carry and older stochastic checkpoints have
no migration into v29.

The campaign uses GPT-2 BPE with token 50256 as both BOS and EOS. The final
checkpoint records its exact architecture, corpus manifest, and maximum
pretraining context.

The BPB guard evaluates one 8-row eval batch (8192 tokens) of a fixed
FineWeb validation prefix by default — a cheap catastrophic-drift canary,
paired against the same tokens every eval. Guard values are comparable only
within one `--bpb-val-tokens` setting; identity checks against a recorded
pretraining val_bpb (the init gate) need `--bpb-val-tokens 2097152`.

`train_latent_vapo` writes a manifest, source snapshot, JSONL metrics,
TensorBoard events, and exact-resume checkpoints beneath the output directory.
The base checkpoint file is never overwritten; the trainable actor copy and
the separate critic state live in the run directory.

### Delightful Policy Gradient

`train_latent_vapo` defaults to the discrete-action [Delightful Policy
Gradient](../papers/delightful_policy_gradient_2603.14608v1.pdf) from Osband
(2026). A new production run therefore needs only its checkpoint and output
paths:

```bash
python3 -m postraining.train_latent_vapo \
  --checkpoint <base-sft-checkpoint> \
  --output postraining/runs/<name>
```

The default immutable broad-v6 bare-prompt manifest and MBPP verifier corpus
are rebuilt with `python3 -m postraining.prepare_vapo_mixture`; the builder
defaults to the same `postraining/data/vapo_broad_v6_bare` prefix consumed by
training.

Each emitted token is one action. Its actor score term is gated by
`sigmoid(advantage * -current_token_log_probability)` with the paper's fixed
temperature eta=1. The gate is stop-gradient, and this mode uses neither PPO
importance ratios nor clipping. The critic, tokenwise GAE, verifier rewards,
and hidden-carry replay are unchanged.

DG is an on-policy estimator. The trainer therefore rejects configurations
where one frozen rollout pool would feed multiple sequential actor updates;
`--prompts-per-minibatch` must equal `--prompts-per-rollout`. The selected
24-prompt windows alternate between 10/8/3/3 and 11/7/3/3 groups from
DAPO/DeepMind/GSM8K/MBPP. This is the nearest integer rotation to broad-v5's
7/5/2/2 ratio and recovers that ratio exactly over the eight-update cursor
phase cycle. Each fresh optimizer batch therefore contains 24 prompts x 16
samples = 384 trajectories. The historical VAPO control remains
available explicitly with `--no-delightful-policy-gradient`; wider frozen
rollout pools must likewise be requested explicitly. `--no-think-tokens`,
`--no-answer-fence`, and an empty `--rl-mixture-manifest` preserve explicit
control configurations. Checkpoints and run manifests bind the selected
actor-objective schema, so exact resume cannot silently switch between VAPO
and DG. A DG exact resume may deliberately change only the equal rollout and
minibatch prompt counts while preserving actor, critic, optimizer, RNG, and
prompt cursor state. That change requires the explicit
`--allow-dg-topology-migration` acknowledgement. The run manifest records the
checkpoint step, sampler cursor, and before/after topology so the earlier
segment cannot be mistaken for the resumed segment. The zero-reward actor
freeze remains enabled. The separate custom per-source success gate is off by
default and available only through `--source-success-actor-gate`, while the
heterogeneous-mixture desert stop defaults off because hard prompt windows do
not prove that a frozen policy cannot succeed on later prompts.

### Intra-trajectory Target Policy Optimization

`--target-policy-optimization` selects the intra-trajectory target-matching
actor used by the local CleanRL HalfCheetah experiment while retaining the
existing state critic and dense GAE credit. It overrides the default DG flag;
there is no policy-gradient auxiliary, importance ratio, PPO clip, sampled
comparison action, or action-Q head.

Every visited prefix is its own target-fitting problem. For the executed token,
raw detached critic GAE `A` shifts the rollout policy's log odds by `A / eta`.
If the executed token had rollout probability `p_old`, its target is
`sigmoid(logit(p_old) + A / eta)`. This normalized target is feasible as a
local executed-token-versus-rest marginal. Binary cross entropy over that
partition has zero gradient when the current executed-token probability reaches
the target. If repeated trajectories visit an identical prefix and request
incompatible targets for different tokens, the shared categorical policy fits
a compromise; zero aggregate residual is not guaranteed. The default `eta=2`
controls the size of the old-policy-anchored target move. Log odds are computed
directly from finite vocabulary logits, so probabilities rounded to zero or one
cannot create endpoint NaNs.

Advantages are not whitened, centered, RMS-scaled, or transformed. Rewards and
critic targets are in [0, 1]; with the deployed gamma=1 recurrence, lambda-GAE
therefore stays on that meaningful native scale, apart from the critic
support's narrow margin bins. At eta=2, an advantage magnitude of one changes
the target odds by a factor of `exp(0.5)` or `exp(-0.5)`. The frozen rollout
policy is the anchor. This is the paper-style target trust control, not a hard
bound on the optimizer's realized KL; no hard KL guard or controller is added.

Every active token is one transition and the actor loss is divided by the
complete minibatch's action-token count. This matches both the CleanRL intra-
TPO reference, which flattens its fixed rollout into a transition batch, and
the existing VAPO reduction. There is no separate trajectory-length weighting.

The custom source-success actor mask is disabled by default. Every source keeps
its critic-GAE actor signal even when its current minibatch has no successful
trajectory; the whole-minibatch zero-reward actor freeze remains a separate
safety mechanism. `--source-success-actor-gate` restores the legacy behavior
that zeroes an entire source until it has one positive-reward trajectory. This
is an opt-in local heuristic, not a VAPO or DAPO paper component. An age-zero
guard checks the executed-token replay probability and aborts before an
optimizer step if the behavior anchor is stale.

```bash
python3 -m postraining.train_latent_vapo \
  --checkpoint <base-sft-checkpoint> \
  --output postraining/runs/<name> \
  --target-policy-optimization
```

TensorBoard exposes old and target probabilities, requested odds shift, target
KL, and pre-update fit diagnostics. At the post-update replay cadence it also
reports the actual target-fit KL and signed/absolute/RMS probability residuals,
beside achieved behavior KL. TPO checkpoints use a distinct v6 actor-objective
schema and cannot silently resume as DG, VAPO, candidate TPO, or an earlier
executed-action objective.

## On-policy self-distillation (OPSD)

`train_opsd` is an additional post-training method; it does not change the
SFT or latent-VAPO objectives or checkpoints. It implements
[Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734v3) with:

- one on-policy response sampled from the question-conditioned student;
- the same initialization checkpoint as a frozen step-0 teacher, conditioned
  on the verified reference solution and the student's response prefix;
- full-vocabulary forward KL at every response position;
- pointwise clipping of each vocabulary entry's KL contribution before the
  vocabulary sum; and
- gradients through student logits only.

The paper's main 100-step configuration is the CLI default: effective batch
32, 1024 completion tokens, temperature 1.1, top-p 0.95, top-k 20, AdamW at
5e-6, gradient norm 0.1, and pointwise clip 0.05. The selected SFT base's
think/answer fence settings and source trace parquet are inferred from its
checkpoint provenance. Recurrent rollout decoding uses the same dynamic
compiled step as the current VAPO production path by default; the eager path
remains available as `--no-rollout-compile` for compiler diagnosis.

The paper's main Qwen experiments additionally pair a thinking-mode-off
student with a thinking-mode-on teacher. This KDA backbone has no Qwen-style
chat-template mode switch, and the selected SFT checkpoint was explicitly
trained to emit a structural `<think>` span. OPSD therefore preserves that
checkpoint contract instead of giving the student a contradictory format
instruction; only the privileged teacher receives the paper's independent
reasoning transition. Starting from the pretrained checkpoint uses the
non-fenced student prompt. Loss reduction follows paper Algorithm 1 exactly:
mean over each response's tokens, then mean over examples (the authors'
released trainer flattens valid tokens into a global mean when lengths vary).

Static validation can be run without CUDA:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m postraining.train_opsd \
  --name opsd_v1 \
  --validate-only
```

For DAPO, build the deduplicated answer-privilege data and run the frozen
teacher-uplift gate before authorizing any OPSD updates:

```bash
.venv/bin/python -m postraining.opsd.prepare_dapo \
  --sft-corpus postraining/data/sft_traces_v6_answer_bare_a1swap10k.parquet

mlq submit \
  --name opsd_dapo_teacher_uplift_bare_256_v1 \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  .venv/bin/python -m postraining.opsd.teacher_uplift \
    --name opsd_dapo_teacher_uplift_bare_256_v1 \
    --checkpoint postraining/runs/sft6_bare_a1swap10k_e3/sft_final_model.pt \
    --gate-data postraining/data/opsd_dapo17k_bare_gate.parquet \
    --data-manifest postraining/data/opsd_dapo17k_bare.manifest.json \
    --rows 256
```

The easier broad answer-only curriculum combines DeepMind Mathematics,
GSM8K, and DAPO with an exact 24/18/6 trajectory schedule in every batch of
48. This intentionally upweights the two easier sources and reduces DAPO to
12.5% of updates. The immutable builder globally deduplicates normalized
problems, rejects conflicting truths, removes SFT-overlapping gate questions,
and constructs a source-stratified 96/72/24 held-out gate with split-local,
token-position-matched answer derangements:

```bash
mlq submit \
  --name opsd_math_mixture_prepare \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  .venv/bin/python -m postraining.opsd.prepare_math_mixture
```

Mixture training requires `--effective-batch-size 48`; use
`--rollout-batch-size 48` to generate the whole update in parallel when it
fits. The sampler enforces exact source counts without runtime skips, stores
redundant per-source cursors for exact resume, and emits per-source trajectory
fractions and verifier accuracy to TensorBoard. MBPP is deliberately excluded:
an answer-only teacher cannot use executable tests as privileged information
without a separate code-specific teacher contract and authorization gate.

The paper's main method conditions the frozen teacher on a worked reference
solution; its teacher does not generate another trace. DAPO contains only
verified final answers, so applying OPSD to DAPO is an explicit answer-only
extension rather than a reproduction of Algorithm 1. The development gate
tests the released code's optional explicit-rationalization idea without
importing traces from another model: the frozen SFT checkpoint generates
separate question-only, correct-answer, and permuted-answer think prefixes,
and only those fixed prefixes condition teacher scoring. They are never SFT
targets. Each prefix is mechanically filtered, stripped to think content,
and cut to one fixed token budget; the rationale sample is different from
the frozen student response being scored.

A deterministic answer-derangement arm controls for generic self-distillation
and prompt/style effects. Correct and permuted donors are matched by encoded
answer length, and every rationale arm uses the same token budget, so all
compared teacher logits begin at identical positions. The direct incremental
control combines the correct answer with an independent question-only
self-rationale. This distinguishes useful answer-conditioned derivation from
the effect of merely revealing the final answer.

OPSD authorization requires two frozen gates. The generation gate checks
strict fenced correctness, format, termination, repetition, and loops. The
paired-logit gate scores exactly the same question-only response tokens under
all teacher contexts. Its primary endpoints require the correct-rationale
teacher to outperform both the permuted-rationale control and the direct
correct-answer control on answer-masked, pre-conclusion think tokens. The
advantage must survive the exact pointwise-clipped OPSD update and comprise a
material fraction of its gradient. Final-answer likelihood alone is only a
copying sanity check and cannot authorize training. Panels already inspected
during design are development-only; only a fresh sealed panel can produce an
authorization artifact.

The trainer remains fail-closed by default. A deliberately experimental run
requested despite a failed gate must supply both the immutable failed artifact
and `--allow-failed-authorization`. The run contract records the failed
decision and that the override was applied; this flag never converts a failure
into a pass and must not be used as evidence that the privileged teacher is
effective.

Training is a model workload and must go through `mlq`:

```bash
mlq submit \
  --name opsd_v1 \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  .venv/bin/python -m postraining.train_opsd \
    --name opsd_v1 \
    --checkpoint postraining/runs/sft6_bare_a1swap10k_e3/sft_final_model.pt \
    --dataset postraining/data/opsd_dapo17k_bare_train.parquet \
    --reference-column solution \
    --data-manifest postraining/data/opsd_dapo17k_bare.manifest.json \
    --authorization \
      postraining/runs/opsd_dapo_bare_256_authorization_v1/results.json
```

For the broad 24/18/6 curriculum, substitute the math-mixture train parquet,
manifest, and its freshly generated dual-gate authorization, and set both
batch sizes to 48. A DAPO-only authorization cannot authorize this mixture.

The run writes versioned step exports, an exact-resume
`opsd_checkpoint.pt`, the load-model-compatible `opsd_final_model.pt`,
canonical `metrics.jsonl`, live TensorBoard events under `tensorboard/`,
sampled `generations.jsonl`, a manifest, and a source snapshot beneath
`postraining/runs/<name>/`. The dashboard separates raw forward KL from the
potentially negative clipped objective and includes clipping, rollout,
throughput, gradient, and GPU-memory diagnostics. For final-answer DAPO runs,
every already-generated on-policy response is graded with the same verifier
and structural contract as VAPO. TensorBoard reports
`reward/exact_accuracy`, raw verifier accuracy, structural validity, and the
CPU verifier time on every update. These diagnostics do not enter the OPSD
loss and add no model generation or forward pass.

The full 144-row, avg@8 DeepMind interpolate-easy benchmark is available with
`--eval-every 20`, but is disabled during training by default because it adds
1,152 extra trajectories per evaluation. That panel is explicitly excluded
from SFT construction and has no exact or eight-word overlap with the OPSD
train split. Its TensorBoard series reports raw and contract accuracy, change
from step 0, format, termination, and per-prompt success coverage. Evaluation
uses 128-way trajectory batches by default. An OPSD export preserves the input
checkpoint's nested `sft` metadata, so it can initialize the current VAPO
trainer with the same fence-token reconstruction and provenance gates.

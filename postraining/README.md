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

`train_minicpm_vapo` is an additional standard-token VAPO path; it does not change
the nano, latent-thinking, or OPSD trainers. It defaults to the pinned final
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
11,024-token packed budget, stable segmented SDPA, and checkpoints every fourth
actor and critic decoder layer. The `--replay-attention-backend fa4` path
remains available only for profiling; SM120 FA4 varlen backward produced NaNs
in production. Retaining the other activations cuts recomputation without
exceeding 32 GiB.

MiniCPM VAPO uses the checkpoint's native thinking template, temperature 0.9,
top-k 20, and top-p 0.95. The default 10,000-token response budget uses the
64-row static cache after offloading the frozen training backbone.
`--top-k 0 --no-fast-rollout` retains the slower exact full-vocabulary nucleus
sampler. Production aborts if steady scheduled decode throughput falls below
`--min-rollout-tokens-per-second`.

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

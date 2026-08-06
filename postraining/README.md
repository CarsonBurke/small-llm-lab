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
jobs. The selected canonical replacement is the three-epoch checkpoint at
`postraining/runs/sft_v4_answer_canonical_hfonly_e3/sft_final_model.pt`;
prompt-schema guards deliberately reject the old checkpoint under the current
code. Its matched two-epoch arm had materially weaker sampling-gate results.

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

Run every GPU workload through `mlq`. Before a new training campaign, run the
frozen-policy learnability gate:

```bash
mlq submit \
  --name posttrain_base_gate \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  python3 -m postraining.train_latent_vapo \
    --checkpoint postraining/runs/sft_v4_answer_canonical_hfonly_e3/sft_final_model.pt \
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
    --checkpoint postraining/runs/sft_v4_answer_canonical_hfonly_e3/sft_final_model.pt \
    --output postraining/runs/posttrain_dg_broad
```

With `--answer-fence`, every math family is canonicalized to the same episode
prompt before SFT, RL, or evaluation:

```text
{bare problem}

Start your response with <think> and reason until </think>, then end it with only the final answer inside <answer></answer>.
```

DAPO, DeepMind, AIME, and GSM8K source wrappers are removed rather than
rewritten one-for-one, so duplicated legacy `Answer:` reminders cannot create
duplicated fence contracts. The prompt schema is recorded in SFT and RL
checkpoints and enforced on initialization and exact resume; regenerate the
canonical SFT corpus and retrain SFT when this schema changes.

## Latent thinking: deterministic hidden carry

`--reasoning-mode latent` trains the hidden-carry policy (execution schema
v28). A "thought" is the post-final-norm belief that produced a generated
token; when that token feeds back as input, its producing belief rides along
as a residual on the token embedding:

```
combined = embed(token) + hasThought * (W(belief) + type_bias)
input    = combined + MLP(RMSNorm(combined))   # per block, relu^2, identity at init
```

`hasThought` is 1 exactly where the input token was model-generated (prompt
tokens and the first generation step carry no hidden). `W` is
zero-initialized, so a fresh run is bit-exact with the pretrained token
policy while `W` gets a full-rank gradient from the first step. (The v1
form gated an orthogonal `W` behind one zero-init scalar gain — a
multiplicative saddle neither factor escaped in practice.) There is no
stochastic thought channel: the only actions are tokens, training defaults to
token-level DG with an HL-Gauss critic, and the carried belief is
detached replay data — no BPTT. The separate critic re-derives combined
embeddings with its own combiner weights. `cot` and `none` remain token-only
control modes with zero-width hidden storage. Combiner geometry is set by
`--combined-mlp-hidden` (default 2048) and `--combined-mlp-blocks`
(default 1; 0 ablates to the pure residual).

Checkpoints from the pre-v28 stochastic thought/gate policy cannot be
resumed or evaluated; there are no migrations. Test-time-read-compute
(carrying hiddens for read tokens) is deliberately out of scope for this
schema.

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

The default immutable broad-v5 manifest and MBPP verifier corpus are rebuilt
with `python3 -m postraining.prepare_vapo_mixture`; the builder defaults to the
same `postraining/data/vapo_broad_v5` prefix consumed by training.

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
  --sft-corpus postraining/data/sft_traces_v4_answer_canonical_hfonly.parquet

mlq submit \
  --name opsd_dapo_teacher_uplift_contractlast_512_v3 \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  .venv/bin/python -m postraining.opsd.teacher_uplift \
    --name opsd_dapo_teacher_uplift_contractlast_512_v3 \
    --checkpoint postraining/runs/sft_v4_answer_canonical_hfonly_e3/sft_final_model.pt \
    --gate-data postraining/data/opsd_dapo17k_contractlast_gate.parquet \
    --data-manifest postraining/data/opsd_dapo17k_contractlast.manifest.json \
    --rows 512
```

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
    --checkpoint postraining/runs/sft_v4_answer_canonical_hfonly_e3/sft_final_model.pt \
    --dataset postraining/data/opsd_dapo17k_contractlast_train.parquet \
    --reference-column solution \
    --data-manifest postraining/data/opsd_dapo17k_contractlast.manifest.json \
    --authorization \
      postraining/runs/opsd_dapo_contractlast_512_authorization_v3/results.json
```

The run writes versioned step exports, an exact-resume
`opsd_checkpoint.pt`, the load-model-compatible `opsd_final_model.pt`,
canonical `metrics.jsonl`, sampled `generations.jsonl`, a manifest, and a
source snapshot beneath `postraining/runs/<name>/`. An OPSD export preserves
the input checkpoint's nested `sft` metadata, so it can initialize the current
VAPO trainer with the same fence-token reconstruction and provenance gates.

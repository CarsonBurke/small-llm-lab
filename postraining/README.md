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

Then start a run with an explicit output directory:

```bash
mlq submit \
  --name posttrain_cot \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  python3 -m postraining.train_latent_vapo \
    --checkpoint postraining/runs/sft_v4_answer_canonical_hfonly_e3/sft_final_model.pt \
    --output postraining/runs/posttrain_cot \
    --reasoning-mode cot \
    --think-tokens \
    --answer-fence \
    --steps 2000
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
stochastic thought channel: the only actions are tokens, training is standard
token-level VAPO (DAPO clip + HL-Gauss critic), and the carried belief is
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

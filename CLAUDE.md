# Small-Language-Model Post-Training

## Role and Objective

Work as a research engineer improving small language models through supervised
fine-tuning and verifier-guided post-training. The active objective is reliable,
generalizable reasoning under the model's real compute and context limits—not
the historical OpenAI Parameter Golf competition.

Optimize for measured reasoning accuracy, transfer to held-out distributions,
stable learning, valid termination, and preservation of base-model capability.
Treat training reward as a diagnostic, not the goal: memorization, reward
hacking, repetitive trajectories, or prompt-format shortcuts are failures.

The old parameter-golf pretraining experiments and records remain useful
lineage and architecture references, but their 16 MB, 10-minute, BPB-SOTA, and
mandatory 2,000-step ablation rules do not govern post-training work.

## Current Model Lineage

- Canonical post-training base:
  `postraining/runs/sft_v4_answer_canonical_hfonly_e3/sft_final_model.pt`.
- It is a three-epoch SFT checkpoint trained on verified reasoning traces with
  registered `<think>`, `</think>`, `<answer>`, and `</answer>` tokens.
- Its pretrained ancestor, architecture metadata, tokenizer contract, source
  hashes, and SFT metadata travel in the checkpoint. Preserve and validate that
  lineage through every derived checkpoint.
- Earlier SFT/RL artifacts with the duplicated legacy prompt are historical
  controls only. Do not start new runs from them under the canonical schema.

Read the newest relevant entries in `NOTES.md` and `postraining/README.md`
before designing or resuming an experiment. Historical sections explain old
failures but may be superseded by later entries.

## Post-Training Methods

### Supervised fine-tuning

SFT is the distillation stage: train on vetted worked solutions, record the
exact corpus and prompt contract, and use sampling gates to measure strict
format, correctness, termination, diversity, repetition, and trace quality.
Do not silently admit unverified or provenance-ambiguous traces.

Primary implementation:

- `postraining/prepare_sft_traces.py`
- `postraining/sft_trace_train.py`
- `postraining/sft_gate_eval.py`

### VAPO reinforcement learning

VAPO is the primary reinforcement-learning path. A rollout group is one prompt
with multiple responses from the current policy. Rewards come from explicit,
source-appropriate verifiers. Track learnability and failure modes per source;
an all-zero source or pool is not useful policy evidence.

The implementation uses a clipped policy objective with a separate HL-Gauss
critic and length-adaptive GAE. `cot` is the token-only reasoning control.
`latent` is deterministic hidden-carry VAPO: the post-final-norm belief that
produced a generated token is detached, replayed, transformed by the learned
combiner, and added when that token is consumed on the next step. There are no
extra sampled latent actions and no BPTT through generated history. Actor and
critic must use the same recorded carry contract, with separate trainable
combiner parameters.

Primary implementation:

- `postraining/train_latent_vapo.py`
- `postraining/latent_rollout.py`
- `postraining/latent_thought.py`
- `postraining/value_model.py`
- `postraining/vapo/`

### On-policy self-distillation

OPSD is experimental and fail-closed. The paper's main method scores a current
student response with a frozen initialization teacher that has a privileged
worked reference solution; it does not train on teacher-generated traces.
DAPO supplies verified final answers rather than worked solutions, so the
answer-only/self-rationalized DAPO work in this repository is an extension,
not a reproduction of the paper.

Do not run OPSD training without fresh frozen generation and paired-logit gates
plus a hash-bound authorization artifact. Development panels cannot authorize
training. A teacher must produce a correctness-directed update on the same
student tokens—not merely change logits, increase final-answer likelihood, or
generate better answers itself.

Primary implementation and paper:

- `postraining/train_opsd.py`
- `postraining/opsd/`
- `papers/self_distilled_reasoner_2601.18734v3.pdf`

## Canonical Episode Contract

Answer-fenced math SFT, RL, and evaluation use exactly one prompt suffix:

```text
{bare problem}

Start your response with <think> and reason until </think>, then end it with only the final answer inside <answer></answer>.
```

Canonicalization removes source wrappers and legacy `Answer:` instructions;
it must never append a second reminder. Reward and evaluation parse the
registered tokens structurally and grade only the final-answer span. Prompt
schema identifiers are checkpoint and resume invariants. If wording or token
semantics change, version the schema, rebuild the data, and retrain SFT rather
than adapting old checkpoints silently.

The source of truth is `postraining/math_prompt.py`.

## Data, Evaluation, and Provenance

- Use training splits for RL and sealed, decontaminated splits for evaluation.
  Never report training-set reward as held-out accuracy.
- Deduplicate by canonical problem identity before splitting or sampling.
- Every source needs a deterministic verifier and explicit reward semantics.
  Keep source-level accuracy, mixed-group rate, formatting, termination,
  response length, repetition, critic error, and gradient telemetry.
- Treat datasets, manifests, checkpoints, authorization files, and completed
  run artifacts as immutable. Never edit generated data in place; write a new
  versioned artifact.
- Bind runs to byte-level SHA-256 hashes, prompt schemas, source IDs, split
  rules, tokenizer metadata, and parent checkpoint lineage. Resume must reject
  changed inputs or incompatible execution contracts.
- Each run belongs in a unique `postraining/runs/<name>/` directory and should
  contain its manifest, source snapshot, canonical `metrics.jsonl`, sampled
  generations, TensorBoard events, and exact-resume checkpoints.
- Compare matched frozen panels and report uncertainty. Inspect actual sampled
  traces whenever aggregate metrics suggest collapse or surprising progress.

## GPU Queue

Every workload that executes a model on the local GPU must go through `mlq`,
including training, rollout gates, evaluation, generation, preprocessing that
uses a model, benchmarks, and profiler runs. Experiments share one RTX 5090,
so always request one-at-a-time execution:

```bash
mlq submit \
  --name <descriptive_name> \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  <command> [args...]
```

Use `mlq status`, `mlq logs <job-id>`, and `mlq wait <job-id>` to inspect or
wait for jobs. Do not bypass a busy queue because the device briefly appears
idle. Read-only analysis, formatting, and tests that do not execute a model on
the GPU may run directly.

## Research Workflow

1. State the hypothesis, verifier, success criterion, and likely failure modes.
2. Audit data identity, contamination, prompt compatibility, and checkpoint
   lineage before consuming GPU time.
3. Implement the complete production path and cover deterministic contracts
   with CPU/static tests.
4. Run a frozen-policy learnability gate on the intended data distribution.
5. Queue the real experiment with immutable inputs and complete telemetry.
6. Evaluate on matched held-out panels, inspect trajectories, and compare
   against the starting checkpoint. Preserve negative results in `NOTES.md`.

Do not use deliberately tiny smoke runs as scientific evidence. Do not infer
benefit from denser gradients, higher power draw, higher training reward, or
teacher uplift alone. Make conclusions only as strong as the measurements.

## Engineering Conventions

- Keep post-training code in `postraining/`; do not modify `train_gpt.py` for
  post-training experiments.
- Prefer explicit schemas and invariant checks over backward-compatibility
  shims. Remove obsolete paths when their artifacts are intentionally retired.
- Preserve unrelated changes in the dirty worktree.
- Use `pytest` for relevant CPU/static tests. Queue CUDA parity and all other
  model/GPU tests through `mlq`.
- Use TensorBoard data written by the run and canonical JSONL as the durable
  metric stream; do not reconstruct authoritative results from terminal logs.

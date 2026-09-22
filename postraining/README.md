# Post-training

## Reviewed math corpus and prompt traversal

`core.load_unique_math_rows()` deduplicates by the **exact full ordered chat
messages**, not `extra_info.index`. Roles, case, whitespace, and message order
remain significant. Identical prompts with identical effective reward/test
contracts keep their first occurrence; conflicting contracts quarantine the
whole prompt group. Source indices remain provenance, not question identity.
An empty effective corpus is an error.

The 2026-09-17 DAPO audit found 1,791,700 physical rows, 17,917 source indices,
and 17,398 exact distinct prompts: 519 extra copies across 495 duplicate
groups. Seven prompt groups had conflicting targets. Together with one
reviewed invalid-target quarantine, future loads yield **17,390 questions**.
Audit evidence is in `runs/math_corpus_audit_20260917/{before,after}.json`.

`data/math_target_reviews.json` records exact prompt fingerprints, expected
source targets, and mathematical evidence:

- Nested divisor count: **12 → 6**; the outer divisor-count operation was missed.
- Count of `n ∈ [1, 2015]` with `5 | n³ + 3ⁿ`: **403 → 404**; verified by
  modular and direct integer enumeration.
- Subset-sum tolerance target **d=10 quarantined**: nineteen copies of 96 total
  1824, but no subset lies in `[1800, 1820]`. No replacement optimum is asserted.

Reviews apply to matching source copies at load time without rewriting parquet
bytes. Unexpected targets on a reviewed prompt quarantine it rather than
silently overwrite a new label. Correction receipts certify only the terminal
answer, **not any retained source solution trace**. This is a targeted repair,
not certification of every remaining answer.

MiniCPM's seeded sequential cursor now traverses the content-unique corpus
before wrapping. Generating 16 attempts together for one prompt is intentional;
it does not select that prompt again later in the same corpus pass. Warmup
consumes the same cursor, and exact resume preserves it. Existing weighted
mixture samplers retain their independent per-source passes and quotas; they
do not promise a single global corpus epoch.

New training checkpoints bind the ordered effective prompts, grading contracts,
and review-policy identity in addition to their existing dataset guards.
Missing or changed identities reject exact resume: **do not reuse an old cursor
against a deduplicated/relabelled corpus**. Start a new run instead. Actor-only
evaluation of existing checkpoints remains supported; live processes and saved
artifacts are not retroactively changed.

VAPO mixture and OPSD preparation use the same reviewed loader and record corpus
provenance. Existing prepared manifests without the current policy/effective
source identities must be rebuilt into **new immutable outputs** before future
training; never overwrite them beneath an existing run.

## Run locations and dashboards

Post-training runs belong in `postraining/runs/<run_name>/`, with TensorBoard
events in its `tensorboard/` subdirectory. The **parameter-golf-postraining**
dashboard at `http://127.0.0.1:6106/` watches this tree directly; no dashboard
symlink is needed for new runs. Put metrics, summaries, and checkpoints in the
same run directory.

The **parameter-golf** dashboard at `http://127.0.0.1:6101/` watches `tb_logs/`
and is for **pretraining only**. Never link post-training runs there.
MiniCPM training (including `scripts/run_minicpm_vapo.py`) and its resume
preflight reject output directories outside `postraining/runs/`, including
symlinks that escape the tree, before loading a model or writing run artifacts.
Historical external runs remain accessible through existing post-training
dashboard links; active run files are not moved. Resume historical checkpoints
into a new canonical run directory rather than an external legacy destination.

**Carry-RL final benchmarks:** AIME 2025 and AIME 2026 are optional evaluations
after training, not training rewards or automatic per-update evaluations.
Each local `postraining/data/aime-<year>.parquet` contains all 30 problems,
with 15-problem `aime-<year>-i.parquet` and `aime-<year>-ii.parquet` splits.
The user-selected published reference is MiniCPM5-1B's **40.42% Avg@16** on
both years on its [model card](https://huggingface.co/openbmb/MiniCPM5-1B):
30 problems × 16 independent attempts, averaged per attempt, not pass@16.
Before claiming a matched comparison, establish the publisher's prompt,
thinking/token budget, sampling, and grading protocol; also evaluate the
untouched pretrained checkpoint with the same local protocol as the carry
checkpoint. Per-year dataset manifests record provenance and transcription
limitations. AIME 2026 has three reviewed source corrections; AIME 2025 uses
pinned OpenCompass transcriptions with an integer answer-key cross-check
against Math-AI. The publisher's exact dataset revision remains unverified.

`scripts/evaluate_minicpm_vapo.py` evaluates carry-v4 or native-v6 checkpoints
through `CapturedTrainingRolloutEngine`, restoring the actor adapter and trained
carry combiner rather than using stock `generate`. It does not construct a
critic or optimizer, and disables replay-history storage without disabling
recurrence. The ordinary `eval_hf_math.py` entry point still refuses carry
checkpoints and points to this evaluator; it also accepts optional
`--suite aime_2025` or `aime_2026`, plus their `_i` and `_ii` splits, for stock
HF models. These optional HF suites default to Avg@16; the carry evaluator
continues to inherit its checkpoint's sample count.

The checkpoint owns evaluation defaults: thinking mode, prompt suffix,
prompt/response budgets, answer reserve, sampling, samples per problem,
batch geometry, and seed. For the current full-softmax carry run this means
30 problems × 16 attempts, 10,000 response tokens, a 1,000-token answer reserve,
the 9K-budget suffix, and temperature 1 / top-k -1 / top-p 1.
Use explicit overrides for a different comparison protocol;
`--prompt-suffix "" --answer-reserve-tokens 0` removes both budget interventions.
These local defaults are **not** claimed to reproduce the publisher's protocol.

```bash
# Set CHECKPOINT to the selected trained checkpoint; metadata inspection runs
# directly because --dry-run never loads a model or writes run artifacts.
.venv/bin/python scripts/evaluate_minicpm_vapo.py \
  --checkpoint "$CHECKPOINT" --output postraining/runs/minicpm_carry_aime26 \
  --dry-run

# Run only when evaluation is wanted; neither command is part of training.
mlq submit --name minicpm-carry-aime26 --cwd "$PWD" \
  --max-parallel-runs 1 --time-limit 4h --max-attempts 1 -- \
  .venv/bin/python scripts/evaluate_minicpm_vapo.py \
  --checkpoint "$CHECKPOINT" --output postraining/runs/minicpm_carry_aime26

# Same saved protocol, untouched base checkpoint, no learned adapter or carry.
mlq submit --name minicpm-stock-aime26 --cwd "$PWD" \
  --max-parallel-runs 1 --time-limit 4h --max-attempts 1 -- \
  .venv/bin/python scripts/evaluate_minicpm_vapo.py \
  --checkpoint "$CHECKPOINT" --stock --output postraining/runs/minicpm_stock_aime26
```
Select AIME25 by adding `--suite aime_2025` to the carry or stock command and
choosing a separate output, such as `postraining/runs/minicpm_carry_aime25`.
AIME26 remains the default. Both full suites record the 40.42% Avg@16 reference;
individual sittings do not inherit that combined-benchmark score.
Acquire/rebuild the pinned AIME25 text files with
`.venv/bin/python scripts/build_aime_2025_eval_set.py` (no model/GPU execution).
AIME25 is declared eval-only in `problem_sources.json` and included in the SFT
decontamination targets. Existing immutable registry snapshots and previously
prepared training corpora are not retroactively rewritten.


Outputs are `result.json` (resolved protocol, hashes, provenance and metrics),
`attempts.jsonl` (every generated response), `metrics.jsonl` (cumulative batch
metrics), and `tensorboard/` on **parameter-golf-postraining** only. Output
directories must be new or empty. The primary `contract_content_accuracy` is
mean per-attempt correctness, not pass@k, and accepts correct capped answers
like training. EOS-required and relaxed/boxed-answer metrics are also reported
separately so grading differences remain visible. All problems are evaluated,
including the incomplete final prompt batch. Gaussian-latent and Uno
checkpoints are rejected rather than silently evaluated as native tokens.

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

Fresh trainer runs default to reserving **1,000 answer tokens inside the existing
10,000-token response cap** (`--answer-reserve-tokens 1000`). If the response has
not naturally emitted the native `</think>` token, rollout inserts it as response
token 9,000, leaving up to 1,000 subsequent tokens for the answer. Earlier natural
thinking closure or EOS is unchanged. The delimiter counts toward the total cap;
no additional KV capacity is required. Set the reserve to zero to disable budget
forcing.

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
`--token-carry --no-train-nextlat --answer-reserve-tokens 1000` on a fresh
`postraining.train_minicpm_vapo` run. Native token-only remains the default;
top-k 20, temperature 0.9, top-p 0.95, and the optimized BF16 rollout backend
are unchanged.

After sampling token `x` from the vocabulary head, the next input is
`E(x) + scale * (Wd E(x) + Wh stopgrad(h))`, where `h` is the previous
final normalized hidden state, immediately before the LM head. The two
bias-free projections are implemented without allocating a concatenation.
Both matrices start at zero; each channel of the independent FP32 LayerScale
vectors starts at **0.01**, without a sigmoid or tanh. Multiplication of the
residual by the scale is FP32, then the scaled residual is cast to the token
embedding dtype before addition. Scales may be negative or exceed one.
The pretrained embedding bypass remains intact (4,720,128 parameters per
combiner). Scale gradients are initially zero while the residual is zero;
the matrices learn first, then the scales. This parameterization is an
optimization hypothesis, not a demonstrated quality improvement.
Prompts use plain embeddings. Every generated token uses the carry path,
including sampled or imposed native thinking delimiters and EOS. There are no Gaussian
actions, latent slots, additional noise, or learned stopping gates.

Rollout stores the actor's **behavior-time producer hidden** alongside each
sampled token as detached BF16 data. Actor and critic consume that identical
observed stream through **independent LayerScale combiners over their token embeddings**. Detach
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
replay. Gaussian latent mode, Uno/NextLat proposals, and auxiliary NextLat
training remain incompatible with this experiment. The normal answer reserve
applies: with a 10,000-token response cap and 1,000-token reserve, force
`</think>` at response position 8,999 (zero-based) if thinking is still open.
The following 1,000 slots remain available for the answer; early natural closure
is untouched. An imposed delimiter stays in the carry/context and critic targets,
but is excluded from the sampled policy objective and KL. This reserves space,
not a guarantee of a complete or correct answer.
Checkpoints use `minicpm5_vapo_token_carry/v4`; both residual projections and
per-channel scales, optimizers, pending tokens **and stored carries**, and RNG
state are saved. Old v1/v2/v3 checkpoints are rejected: v1 lacks the stored
observations, v2 uses the ungated identity-initialized token projection, and v3
uses a scalar sigmoid gate. Continuation preflight requires a finite FP32 scale
vector matching the projection width; stock native-only evaluation rejects
carry checkpoints instead of dropping their inputs.

**Opt-in slot memory (extension of token carry):** add `--slot-memory`
(`--slot-memory-slots 64 --slot-memory-heads 1 --slot-memory-head-dim 128`)
to a fresh `--token-carry --no-train-nextlat` run. The direct `Wh h` term is
replaced by an attention read over up to `M` stored producer hiddens: every
generated step also samples a slot action `σ ∈ {0..M-1, ∅}` from a
zero-initialized head over the producer hidden, writes that hidden into the
slot (pure overwrite; `∅` writes nothing), and the next input is
`E(x) + scale * (Wd E(x) + Wo read)` with RoPE-by-position keys applied at
write time plus a learned null key. `log π = log π_tok + log π_σ` shares one
advantage; forced `</think>` never writes and stays excluded from the policy
objective; the terminal slot choice is never read and carries no credit;
admission clears the lane. Records carry `slot_choices` beside
`carry_hiddens`; replay rebuilds the alive table per trajectory and runs
compiled FlexAttention with a block-sparse mask (at most `M` keys per query).
Checkpoints use `minicpm5_vapo_slot_memory/v1`; plain token-carry loaders,
continuation with a different mode or geometry, and stock generate reject
them; `scripts/evaluate_minicpm_vapo.py` evaluates them through the
slot-aware rollout engine. Design note: `NOTES.md` 2026-09-17 "Slot memory over token carry".
CPU contracts: `postraining/tests/test_slot_memory.py`; GPU checks:

```bash
mlq submit --name slot-memory-cuda-validation --cwd "$PWD" --max-parallel-runs 1 \
  --time-limit 45m --env RUN_SLOT_MEMORY_CUDA_VALIDATION=1 -- \
  .venv/bin/python -m pytest postraining/tests/test_slot_memory_cuda.py -x -v -s
```

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
likelihoods as well as independent critic values for GAE. The
refactor retains both passes and transfers the two scalar statistics per
action to CPU once per rollout rather than twice per trajectory. GPU buffers
cost eight bytes per response action; likelihood and advantage semantics are
unchanged. A paired benchmark on 64 trajectories / 534,081 actions produced
bit-identical likelihoods and advantages. Median refresh times were 26.49 s
original versus 24.92 s batched transfer (6.3% speedup), but trial timings varied
substantially; this is not a stable general throughput estimate.

Gated run **7679** was stopped after the user reported 100% truncation with
the answer reserve disabled. Its rolling checkpoint did not retain the critic
warmup boundary; training/benchmark evidence remains in
`ablation_results/minicpm_token_carry_gated_train_20260916/`.

Replacement job **7718** restored the 1,000-token answer reserve, then was
externally cancelled during warmup. It retained actor step zero / critic warmup
step three with no pending replay. Historical job **7746** resumed that checkpoint and
added `--prompt-suffix "You have a budget of 9k tokens"` without changing the
10,000-token response cap, reserve, or reward. Normal priority, parallel limit
one, two-hour cap, one attempt remain in force.
The warmup boundary is now saved separately as `critic_warmup_checkpoint.pt`
before actor updates, using the existing immutable checkpoint publication helper.
The suffix is appended to each task before native chat templating, is reflected
in TensorBoard sample text, and is saved in checkpoint arguments. It may change
on continuation only when no replay records are pending. It survives the
1,024-token prompt truncation with the pinned MiniCPM tokenizer. 135 focused
prompt/checkpoint/evaluator host tests pass; the earlier carry/forcing checks
passed separately. This prompt communicates a ceiling, not an exact counter.

Evidence is written to
`ablation_results/minicpm_token_carry_gated_budget9k_20260916/`.
Its linked TensorBoard run remains available at
`http://127.0.0.1:6106/?runFilter=minicpm_token_carry_gated_budget9k_20260916#timeseries`.
The run name has `/tensorboard` appended. The server rescans every 180 seconds.
The historical sigmoid run's
`carry/actor_gate` and `carry/critic_gate` report its scalar gates. Behavior refresh reports
`carry/{actor,critic}_probe_carry_to_token_rms` and
`carry/{actor,critic}_probe_residual_to_token_rms`, using at most 256 evenly
spaced carry inputs from the first nonempty packed shard, not the whole rollout.

On user request, **7746 was cancelled and replaced by LayerScale job 7758**.
The replacement is queued at normal priority with parallel limit one, a
two-hour cap, one attempt, and no additional autocull policy. It starts fresh
with ten critic-warmup cycles rather than converting the v3 optimizer state.
The 1,000-step target, 64 trajectories per rollout, budget prompt, response cap,
answer reserve, and replay configuration are unchanged. This is not an exactly
matched warmup comparison: 7746 resumed three earlier warmup cycles without
the budget prompt, whereas 7758 uses that prompt from the start.

LayerScale reports `carry/{actor,critic}_scale_{mean,min,max,rms}`, retaining
the carry/token and residual/token RMS probes above. The focused host suite
passed 172 tests; a separate BF16 boundary check confirmed that FP32 scaling
produces distinct outputs where rounding the scales first would erase their
difference, with scale gradients preserved and carry-producer gradients absent.
GPU/compiled execution is pending queue admission; no learning benefit is
claimed. Artifacts: `ablation_results/minicpm_token_carry_layerscale_budget9k_20260916/`.
TensorBoard:
`http://127.0.0.1:6106/?runFilter=minicpm_token_carry_layerscale_budget9k_20260916#timeseries`.

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
refilled lanes without changing the logical rollout.

One PPO epoch now defaults to one actor and one critic optimizer update over
all 64 trajectories (`--optimizer-minibatches 1`). Length-bucketed replay
microbatches only bound activation memory: their loss sums use the full
optimizer batch's action counts, gradients accumulate, and clipping and AdamW
run once after all shards. Forced actions are excluded from the policy count.
Use `--replay-token-budget` and `--replay-max-trajectories` to control VRAM.
This lets sparse successful trajectories contribute to the same update as
failures instead of making several sequential, potentially all-failure updates.
It cannot supply positive reward to an entirely unsuccessful rollout.

`--optimizer-minibatches 4` retains the old four disjoint 16-trajectory updates
for comparisons. One full-rollout update is not equivalent to those four AdamW
steps; learning rates are unchanged, not multiplied by four. Fixed behavior
log-probabilities and advantages are retained across replay. Exact post-update
behavior KL is measured every ten PPO epochs. Existing checkpoints can change
optimizer minibatch count only between rollouts, with no pending records;
historical launch scripts that explicitly pin four retain that setting.

Token-carry training disables NextLat, so its primary objective is invariant
to memory partitioning apart from numerical roundoff. Native/Gaussian runs
with NextLat still sample and balance auxiliary gradients per memory shard;
that auxiliary objective is not partition-invariant. With one optimizer batch,
its sample budget is 64 per PPO epoch rather than four budgets of 64. These
are implementation semantics, not evidence of improved training accuracy.

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

Three replay execution paths changed on 2026-09-18 without a numerical
qualification flag because each is exact by construction: a power-of-two LoRA
scale (alpha/rank, 2.0 in production) is folded into the rank-16 `lora_b`
operand before its GEMM and added in place, so the token-sized multiply
disappears in both directions; segmented SDPA concatenates token-major views
so the head merge is a view instead of a second activation copy; action-row
gathers use `index_select` on the single packed row, whose backward is one
`index_add` over unique positions instead of the sorted accumulate path.
Hidden-space trunk balancing records finiteness on the deferred device flags
like the parameter-space path. The parity job
(`scripts/ablate_minicpm_exact_replay_paths.py`, job 8090) measured the same
16-trajectory/160k-action optimizer minibatch at 17.85/18.13s on the legacy
paths and 16.62/16.63s on the exact paths (8.2% higher warm update throughput,
identical peak allocation) with a maximum parameter difference of 7.45e-9,
equal to the unchanged-path repeat control. Behavior refresh with logit chunk
1024 is bitwise identical to chunk 128 and 3.6% faster; the update path at
1024 has not been compared. Flash SDPA serves GQA causal replay on SM120, but a
4096-token probe showed the default dispatch (1.50ms) beating forced flash
(3.52ms), so which kernel production replay runs still needs a profiler check.

Rollout decode plans the split-KV partitions once per captured step
(`postraining/split_kv_plan.py`) from the same device lengths every layer
consumes; the attention operator accepts that shared plan or derives its own,
with identical output. The microbenchmark (`scripts/benchmark_split_kv_plan.py`,
job 8089) cut kernels per decode step from 312 to 76 with bitwise-identical
output; replay time is unchanged at full 11,264-token length (KV-bandwidth
bound) and drops 6.50→6.14ms at mixed lengths and 0.44→0.15ms at short
lengths. `--behavior-logprobs rollout` reuses the captured merged-bf16
replica log-probabilities instead of the replay actor pass; it changes the
importance ratio's reference distribution and stays opt-in until a learning
ablation and explicit sign-off.

MiniCPM VAPO uses the checkpoint's native thinking template, temperature 0.9,
top-k 20, and top-p 0.95. The default 10,000-token response budget uses the
64-row static cache after offloading the frozen training backbone.
`--top-k 0 --no-fast-rollout` retains the slower exact full-vocabulary nucleus
sampler. Production aborts below `--min-rollout-tokens-per-second`: the AR path
uses steady scheduled decode tok/s; opt-in Uno uses useful end-to-end rollout tok/s.
Choose the Uno floor from a matched benchmark, not the AR scheduled-token value.

Native token rollouts, including token carry, also support
`--top-k -1 --top-p 1 --temperature 1` for full-vocabulary categorical sampling
without leaving the compiled fast decoder. This setting matches the untempered
full-softmax PPO scorer. `top-k=-1` requires `top-p=1` and is not supported by
Gaussian latent or Uno rollouts; a non-unit temperature still differs from the
current PPO scorer. Existing sampling defaults are unchanged.

`scripts/run_minicpm_vapo.py` accepts the trainer's arguments and preserves the
carry experiments' phase timings and canonical `<output>/metrics.jsonl` stream
alongside TensorBoard. Use `--output postraining/runs/<run_name>` and submit this
entry point through `mlq`, as with the trainer.

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

TensorBoard and run-local `metrics.jsonl` record training metrics. Semantic
categories cover rollout quality, rollout performance, refill efficiency, sampling, replay, actor,
critic, advantages, KL, ratios, clipping, gradients, auxiliary NextLat,
optimization time, and system telemetry. Every category is capped at twelve
charts. Configuration and correct/incorrect response samples are text
summaries. Scalar writers rely on TensorBoard's asynchronous flush interval
instead of synchronously flushing every progress callback.

Render saved rollout responses as a standalone HTML transcript:

```bash
.venv/bin/python -m postraining.rollout_report \
  --run postraining/runs/<run_name> --last 4 --compare-steps 44,64-84
```

The same command serves the KDA trainer, which writes its own captures
(below); for a MiniCPM run it reads the TensorBoard text samples. The default
destination is `<run>/rollout_samples.html`; `--output <path>.html` selects
another. Re-run it to atomically refresh the latest saved steps; omit
`--compare-steps` for only the latest examples. Questions and reference answers
accompany the original correct/incorrect response text, and different questions
are not paired as if they were the same problem. Original TensorBoard text
omissions remain marked. These are selected examples, not a random sample, and
a live writer may not have flushed its latest text yet. The command reads saved
files only (no model, checkpoint, or GPU workload), so it can run directly
without `mlq`.

KDA training transcripts: every `--rollout-sample-every` actor steps (default
25, always including step 0; 0 disables) `train_latent_vapo` keeps, for each RL
source, one correct (exact verifier reward 1) and one incorrect trajectory from
that pool. The pick is uniform over the pool's matching trajectories, by the
smallest hash of (step, prompt, row), so it is deterministic and does not
follow the collector's shortest-prompt-first scoring order. Each capture
records the prompt, ground truth, full emitted token stream with its fence and
EOS tokens (plus any teacher-forced `Answer:` prefix, marked as such), reward,
termination, and the scorer's own row verdict: its format-gate decision and
the verifier's prediction for the graded field (for a gate-zeroed row, the
counterfactual the gate-zeroed-correct alarm scored). It also records the
group's correct count. A capture is labelled with the number of updates the
policy had taken when it collected the pool; that pool's rollout metrics are
logged at that step plus the pool's update count, which the report shows
beside each step. Captures land in `<run>/rollout_samples/step_XXXXXX.json`
(`training_rollout_samples/v1`), and `<run>/rollout_samples.html` is
re-rendered from the latest eight after each one. Resume deletes captures at or
after the resume step, because those pools are collected again. The option is
display-only and not a resume invariant.

### Generic verifiable tasks

`verifiable_tasks.py` owns the dataset-independent `verifiable_task/v1` contract.
Rows contain a complete `prompt` chat, `reward_model.style` and
`verification_info.schema` set to that version, a string ground truth, and an
arbitrary nonempty `extra_info.domain` for reporting. Verification dispatches on
`verification_info.kind`, **not** the dataset name or reporting domain:

- `math`: exact numeric equivalence, otherwise existing normalized final-answer
  comparison.
- `text`: case-sensitive, whitespace-normalized final-answer equality.
- `python_stdio`: `call_type="std"`, `fn_name=null`, and nonempty, equal-length
  string `inputs`/`outputs` lists. Every hidden case must pass.

Only the final response after closed thinking is scored. `Answer:` owns its
entire remaining tail; one enclosing `\boxed{...}` is accepted. Without an
answer field, the final balanced box is used. Unfinished thinking cannot earn
reward. Python code executes in fresh bubblewrap namespaces with a minimal
read-only system Python runtime, seccomp process/network restrictions, bounded
CPU/memory/output/wall time, and no host credentials. Expected outputs stay
outside the candidate process. Missing isolation infrastructure raises rather
than silently assigning incorrect rewards.

MiniCPM uses one seeded content-unique corpus pass without replacement; grouped
attempts remain intentional. Canonical prompts already own their instructions:
the trainer does not append `--prompt-suffix` again and rejects overlong prompts
instead of truncating them. Checkpoints bind grading implementation, ordered
effective corpus and reporting domains. JSONL/TensorBoard record per-domain
accuracy, mixed groups, token lengths, caps and verifier outcomes.

With `--context-tokens 10000`, **prompt plus response** must fit within 10,000
tokens. Each rollout lane gets at most `min(max_new_tokens, context_tokens -
actual_chat_prompt_tokens)` response tokens; final-answer reservation applies
inside that allowance. No prompt is truncated. This explicit-context mode uses
the native compiled rollout path, not latent/Uno/eager rollout modes.
Changing the context or generation policy is rejected on exact resume.

Rollout/evaluation logs also include duplicate overlapping word 3-gram and
16-gram fractions and the longest identical-word run. These are observational:
they neither change reward nor stop generation. The duplicate n-gram statistic
follows common [Open-R1](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py)
and [TRL](https://huggingface.co/docs/trl/main/en/rewards#get_repetition_penalty_reward)
practice, but this campaign does **not** apply their repetition penalty.
Calibration on saved outputs is in
`runs/minicpm_carry_analysis_20260917/repetition_calibration.json`; even passing
code can be highly repetitive, so a universal cutoff is not justified.

`scripts/evaluate_minicpm_tasks.py` evaluates the same contract and reward path
with the compiled BF16 actor. Supply `--data`, a preserved `--checkpoint` and a
new `--output`; `--stock` uses only checkpoint configuration, not trained adapter
weights. Defaults are up to 32 questions per represented domain and eight
attempts each. `--require-domain-success` fails if a domain has no successful
attempt after the complete evaluation. These local adaptation holdouts are
**not claimed pretraining-clean**. Queue model evaluation through `mlq`.

### Bounded multi-source preparation

`task_data.py` owns prompt construction, global uniqueness, conflict quarantine,
uncapped splitting, and immutable publication. Dataset adapters emit explicit
verification kinds; neither the trainer nor the verifier depends on UltraData.

- `ultradata_data.py` pins `openbmb/UltraData-RL-2609` at
  `e6ecfa733708a4c54b5a98c3ca0fd16fc6923790`. Math/STEM and supplementary code
  reuse bounded raw windows. QA reads all eight indexed Parquet files at
  conversion revision `a9fcbd481b7f80a884c3727102be394329b1f4cc`: 18,045 source
  rows, one fewer than the advertised 18,046. Missing data is not fabricated.
- `codecontests_data.py` pins `deepmind/code_contests` at
  `802411c3010cb00d1b05bad57ca77365a3c699d6`. The default reads all 39 **training**
  shards (13,328 source rows), projecting questions, metadata and complete
  public/private/generated test arrays. Reference/incorrect solutions and the
  upstream validation/test splits are not fetched. The source is CC-BY-4.0.

Those are source inventories, **not** promises of retained counts. Unsupported
test formats, interactive/file-I/O/custom-checker tasks, incomplete or oversized
test suites, and over-budget prompts are rejected, never shortened. UltraData
also excludes source-specific multiple-choice/proof/multipart tasks, reviewed
bad math labels and free-form targets that cannot be safely scored by exact
answer comparison. QA structure checks target the actual question, not lists
inside its supplied context.

```bash
mlq submit --name prepare-verifiable-mixed --cwd "$PWD" \
  --max-parallel-runs 1 --priority 1 --max-attempts 1 --time-limit 45m -- \
  .venv/bin/python -u scripts/prepare_verifiable_tasks.py \
    --output postraining/data/<new_corpus> --context-tokens 10000 \
    --code-shards 39 --validation-fraction 0.10 --seed 1337 \
    --acquisition-bytes 4294967296
```

Preparation is CPU/network work, not model training; negligible GPU utilization
is expected. It still goes through the repository's exclusive `mlq` policy.
The **4 GiB lifetime response-body cap** includes prior cache acquisitions and
conservatively charged failed/interrupted requests. `data_acquisition.py`
verifies cached digests and response ranges, reads bounded Parquet metadata,
and batches adjacent selected columns without crossing unused-column gaps.
No server-side whole-file fallback is accepted.

All usable unique examples are retained: there is no example-count cap,
downsampling, oversampling or raw-population weighting. Actual training
proportions follow retained unique counts after a seeded 10% per-domain holdout.
Raw-window code coverage remains prefix/source-order biased; context, record
and verifier bounds also affect the mix. This is not uniform full-release
sampling.

Current full-chat prompt caps are Math 2,048, Knowledge 2,048, Code 4,096 and
Long_Context 6,144 tokens. Prompts contain no fixed “9k thinking” instruction.
Use the following shared flags for training and base evaluation:

```text
--context-tokens 10000 --prompt-tokens 6144 --max-new-tokens 10000
--answer-reserve-tokens 1000 --prompt-suffix ''
```

Training additionally needs a replay budget covering the total sequence:
`--replay-token-budget 10000`. Evaluate the untouched base with
`scripts/evaluate_minicpm_tasks.py --stock --require-domain-success` before
launching a fresh run; do not resume the collapsed carry checkpoint.

Preparation publishes a new immutable directory with `train.parquet`,
`validation.parquet` and `manifest.json`. The manifest records source ranges,
byte charges, filter reasons, actual domain counts/proportions,
source/tokenizer/template/reward identities, split hashes and globally
content-disjoint membership. Local holdouts are not claimed pretraining-clean.
Format checks do not certify every upstream label or hidden-test suite.
UltraData's upstream licensing/redistribution restrictions and CodeContests'
CC-BY-4.0 terms still apply; this workflow does not rehost either release.

Evaluate the untouched MiniCPM5-1B base on a reproducible KodCode coding sample:

```bash
mlq submit --name minicpm-base-kodcode --cwd "$PWD" \
  --max-parallel-runs 1 --priority 1 --max-attempts 1 --time-limit 4h -- \
  .venv/bin/python -u scripts/evaluate_minicpm_kodcode.py \
    --checkpoint postraining/runs/<preserved_run>/vapo_adapter_checkpoint.pt \
    --dataset-revision dcf78a8bbba9a613b596ce993c4921a38687dfcc \
    --output postraining/runs/<new_coding_eval>
```

The checkpoint supplies validated base identity/configuration only: `stock=True`
does not load its trained adapter, carry, or NextLat weights. Defaults select
256 reference-verified, sandbox-compatible KodCode-Light-RL-10K problems with
seed 42, then generate eight attempts each using compiled BF16, full-softmax
sampling (`temperature=1`, `top_p=1`, `top_k=-1`), a 10,000-token response cap,
and a 1,000-token final-answer reserve. Coding prompts contain the question and
required signature, never reference solutions or hidden tests. Prompts exceeding
2,048 tokens are explicitly excluded rather than truncated or replaced.

The adapter retains plain zero-argument assertion tests and excludes unsupported
pytest features, fixtures, imports, and reference failures with recorded counts.
Selection never consults base-model success. This evaluates the supported subset,
not all 10K problems; reference execution does not certify test completeness or
specification correctness. KodCode is **CC BY-NC 4.0**.

`--dry-run` checks local metadata without model work or output files.
`--prepare-only` materializes the selection and reference preflights without a
model; queue this dataset-preparation workload too. `--prepared-dir <directory>`
reuses a digest-verified prepared selection. Outputs include `manifest.json`,
`prepared_rows.jsonl`, per-attempt evidence, `metrics.jsonl`, `result.json`,
TensorBoard events, and an automatically refreshed `responses.html`.

For training suitability, inspect per-attempt correctness alongside complete
all-fail/mixed/all-pass problem fractions, difficulty/subset breakdowns, truncation,
and format/policy rejections. Any-success across eight attempts is not Avg@8;
partial groups and missing problems are reported separately. No automatic
training launch or quality threshold is imposed. Preserve sampled IDs as a held-out
set when constructing a later training mixture; a useful base success distribution
justifies an ablation, not a claim of improved learning or generalization.

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

### Deterministic hidden carry

`--reasoning-mode carry` (execution schema v30) is a token-only policy like
`cot`, except that each generated token's input adds the detached
post-final-norm belief that sampled it:

```
input(x_t) = combiner(E(x_t), stopgrad(h_{t-1}))
```

Prompt tokens, including the last one that seeds the first decode step, enter
as plain embeddings. There is no gate, Gaussian, or thought slot: the stream
is prompt plus emitted tokens, tokens are the only actions, and nothing
backpropagates through generated history. The combiner is the same
zero-initialized `CombinedEmbedding` stack (`--combined-mlp-hidden`,
`--combined-mlp-blocks`), an exact identity at initialization, and carry draws
tokens from the same generator lane as `cot`, so a fresh carry rollout is
bitwise the `cot` rollout of the same checkpoint and seed.

Rollout stores each belief, in its live dtype, at the slot of the token it
produced. Actor replay and the separate critic read those identical stored
carries, each through its own combiner; value loss never reaches the actor's.
DG, TPO, and clipped VAPO keep their token objectives with no stochastic
factor, under carry-specific objective schemas. `carry/input_delta_rms_ratio`
replaces the stochastic loss/clip and gate/thought-mean gradient tags: at
generated slots, the RMS of the stream input minus the plain token embedding,
over the token-embedding RMS. It is the combiner's whole effect (carry
projection, type bias and MLP stack), so it is not the same number as
`carry_ablation_eval`'s `injection_to_embedding_rms_ratio`, which measures the
linear carry projection `W h` alone.

The separate critic (every mode) is a trunk of the actor's architecture read
by a zero-initialized fp32 scalar head (`CRITIC_SCHEMA`), trained by
unclipped MSE against the lambda-one return, with no parameter shared with
the actor and no gradient-norm clip. `--critic-init scratch` (default) starts
the trunk from a random instance; `--critic-init actor` starts it from a
storage-independent copy of the actor's weights as loaded at run start,
matching VAPO's pretrained-LM critic init without its reward-model bias. The
manifest's `critic.init` records `scratch`, `actor_copy`, or
`warm_checkpoint` (`--actor-critic-init`). A resume carries that record forward
from the resumed run's manifest (its own directory, or the checkpoint's when
resuming into a new one) and refuses a contradicting `--critic-init`.

Carry refuses `--thought-sigma`, `--init-stop-thinking-probability`, and
`--rollout-scheduler continuous_refill`. Its checkpoints record rollout policy
`deterministic_hidden_carry/v2`, input schema
`zero_init_hidden_residual_prenorm_mlp/v2`, and no thought distribution; the
v28 `deterministic_hidden_carry/v1` checkpoints and every cot/latent checkpoint
are refused on resume. `sample_latent`, `run_arithmetic_probe
--wrapper-checkpoint` (which also accepts cot wrappers and refuses latent and
none), and `carry_ablation_eval` read the policy from the checkpoint's
rollout-policy schema and decode it with the carry on. This is
the MiniCPM token carry's idea on the nano/KDA trainer, not its LayerScale
parameterization. Whether the carry improves reasoning is unmeasured.

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

The default immutable bare-prompt manifest is rebuilt with `python3 -m
postraining.prepare_vapo_mixture`; the builder defaults to the same
`postraining/data/vapo_broad_v8_bare` prefix consumed by training and requires
`--sft-corpus`. See "RL mixture quotas follow measured learnability" below for
the v8 sources; the MBPP verifier corpus is now built only when `mbpp` is
explicitly selected with a `--quota`.

Each emitted token is one action. Its actor score term is gated by
`sigmoid(advantage * -current_token_log_probability)` with the paper's fixed
temperature eta=1. The gate is stop-gradient, and this mode uses neither PPO
importance ratios nor clipping. The critic, tokenwise GAE, verifier rewards,
and stored-stream replay are unchanged.

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

## Post-training any model: the VAPO model library

`postraining/vapo/model/` declares what the RL algorithm needs from a model, so
a backbone plugs in by satisfying a contract rather than by being a particular
class. Nothing in the algorithm branches on a model id.

- `protocols.py` — `Capability` (LoRA adapters, frozen output head, gradient
  checkpointing, packed replay attention, compiled replay MLP, static vs paged
  KV cache, fused projections, FA4/split-KV decode, W8A16 head, chat
  template), `TrunkGeometry`, the `Readout` protocol, the `TrunkAdapter` ABC,
  the `RolloutEngine` protocol and the `ModelFamily` ABC.
- `registry.py` — `register_family` / `get_family` / `list_families`.
- `hf.py` — `HFCausalFamily` plus `MINICPM5_SPEC` (pinned id, revision and
  vocab). `HFCausalFamily.resolved_spec(model_id=..., revision=...)` records an
  override instead of mutating the pin.
- `nano.py` — `NanoFamily` for nanoGPT-mini and KDA backbones via `model_io`.
- `lora.py`, `readout.py` — shared LoRA and chunked-frozen-head primitives.

A trunk answers `family`, `hidden_size`, `vocab_size`, `geometry`,
`capabilities`, `identity()`, `readout`, `embed_tokens`, `hidden_states`,
`cached_hidden_states`, `new_kv_cache` and `layers()`. Ask before you act:
`trunk.supports(Capability.X)` branches, `trunk.require(Capability.X)` fails
with the family name and the missing capability. `identity()` is what lands in
a checkpoint payload under `"trunk"`, so lineage travels with the weights
rather than being reconstructed from a model id.

Build a policy with `VAPOPolicy.from_family("minicpm5", device=..., ...)` or
`from_family("nano", checkpoint=...)`; `VAPOCritic.from_family` mirrors it.

Adding a family means implementing `TrunkAdapter` and `ModelFamily` and calling
`register_family`. If your family exposes no LoRA capability, `VAPOPolicy`
trains the whole trunk instead of injecting adapters.

Family-specific code stays honest: `TrunkAdapter.causal_lm` raises a
family-mismatch error rather than an `AttributeError` from deep inside a
rollout engine, and the packed-replay/static-cache/fused-projection paths live
in the Hugging Face entrypoints only.

Two rollout backends sit behind the `RolloutEngine` protocol and share
`ContinuousTrainingGeneration` as their currency. `fast_inference.py` is the
measured Hugging Face engine (continuous lane refill, FA4 varlen replay, fused
projections, graph decode). `vapo/rollout/nano_engine.py` is a correct
lockstep decoder for nano/KDA trunks: fixed physical lanes, one paged cache,
left-padded batched prefill, per-row limits and stop tokens, and exact
untruncated log-probabilities for sampled tokens — the same convention as the
Hugging Face engine's `selected_token_logprobs`. It does **not** do continuous
lane refill and its throughput has never been measured; do not treat it as the
fast path.

### KDA backbones

`postraining/kda_backbone.py` defaults `FLA_TILELANG=0` at import. fla's
TileLang KDA backward cannot run in this environment — it collides with the
standalone `tvm_ffi` over the same TVM FFI TypeAttr, and a KDA training run
otherwise dies at its first backward and then deadlocks unwinding, looking like
a hang rather than a failure. The Triton KDA backward is fla's reference path
and what these checkpoints were pretrained with. `NanoKDABackbone.__init__`
asserts the resolved backend is not TileLang, because fla caches that decision
when its backend module is first imported.

### Reproducing the KDA post-training pipeline

`scripts/launch_kda8_posttrain.sh {sft|rl}` queues each stage through `mlq`
with every value pinned explicitly and the reason for each non-default
recorded in the script header. Stage `rl` consumes the checkpoint stage `sft`
writes, so run `sft`, read its gate, then run `rl`.

Both stages are budget-bound to the base checkpoint's own context:
`sft_trace_train.py` derives `--seq-len` and its sampling-gate budgets from
`backbone.train_context_tokens` and refuses a gate budget larger than that
window, so a short-context checkpoint can no longer pack or sample into RoPE
positions it never saw. They are also hash-bound: the RL mixture manifest
binds `sft_corpus_sha256`, and the RL trainer rejects a base checkpoint whose
recorded `traces_sha256` differs.

## Large single-epoch SFT corpora

`prepare_sft_corpus.py` builds an SFT corpus from published instruction data
in the same schema the curated trace corpora use, so `sft_trace_train.py`
consumes it unchanged:

| column | meaning |
|---|---|
| `source` | `{adapter}:{provenance}`, e.g. `openmathinstruct2:augmented_math` |
| `problem` | canonical bare problem, all source wrappers stripped |
| `document` | `problem` + `<think>\n{solution}\n</think>\n<answer>{answer}</answer>` |
| `final_answer` | the graded answer span |
| `verified` | `"False"` for model-generated solutions that were not re-verified |
| `doc_tokens` | GPT-2 BPE length of `document` |

`document` begins with `problem` because the answer-fence prompt contract
appends no instruction suffix; the trainer recovers the completion as
`document[len(problem):]` and errors loudly if that does not hold.

Adapters declare a corpus rather than sniffing it. Each names its columns,
its provenance column, and the provenance values to exclude:

```bash
.venv/bin/python -m postraining.prepare_sft_corpus \
    --source openmathinstruct2 --source math_drills=500000 \
    --shards 8 --output postraining/data/sft_mix_omi2_drills_v1.parquet
```

`--source NAME[=MAX]` is repeatable and `MAX` caps that corpus alone;
`--max-documents` caps the whole build. Remote adapters stream shards into
`postraining/data/instruction_corpus_shards/` and hash every one into the
manifest; local adapters read a materialized parquet directly.

Two rules are part of the corpus contract rather than later filters:

- **Evaluation protection.** `excluded_provenance` rows are dropped by the
  adapter and the exclusion is recorded in the manifest. OpenMathInstruct-2
  excludes `gsm8k` and `augmented_gsm8k` (17.5% of rows) because GSM8K is an
  evaluation set for this project. A corpus that cannot report per-row
  provenance cannot be admitted.
- **Drop, never reshape.** Documents over `--seq-len`, problems the
  canonicalizer rejects, multi-line answers and too-short completions are
  counted in the manifest and discarded.

`--epochs` defaults to **1**. Repeated passes over a small trace set drove
holdout completion CE from 3.9449 to 0.9009 while the derived policy scored
0.000 on every arithmetic-probe family at every digit count; that gap is
memorisation, not reasoning.

A corpus that is gated is declared in `CREDENTIALED_ADAPTERS` and refuses
with the reason, so it reads as a missing credential and not a typo.

## RL mixture quotas follow measured learnability

`prepare_vapo_mixture.py` ships per-source prompts-per-pool quotas set by
frozen-policy gate accuracy, not by source size. VAPO's advantage is
group-relative, so a uniformly wrong rollout group contributes no policy
gradient — weighting an all-zero source heavily spends the pool on prompts
that cannot teach anything.

- `deepmind_easy` (48) — `mathematics_dataset-v1.0` **train-easy** tier,
  18 verifier-compatible modules, 4,000 prompts each. Build it with
  `scripts/build_deepmind_rl_prompts.py --source-dir .../train-easy --split
  train-easy`; `--split` must name the directory or the build refuses, so a
  manifest cannot misreport its tier. Note that `deepmind-interpolate-easy`
  is the **bench panel** and "easy" there names the module selection, not a
  difficulty tier.
- `ultradata_math` (8) — `openbmb/UltraData-RL-2609` Math rows, promoted from
  the pinned extraction by `scripts/build_ultradata_math_rl_prompts.py`.
  Math only: 95% of its math prompts fit a 256-token budget, against 16% for
  Code and 0% for Long_Context.
- `dapo` (8) — small hard tail, so the policy cannot narrow onto one
  templated prompt shape.
- `deepmind` (interpolate) and `mbpp` ship at **quota 0**: selectable, off by
  default, and each needs an explicit `--quota SOURCE=N` to contribute.
- `gsm8k` is in `RETIRED_SOURCES` and cannot be selected at all.

`--sft-corpus` is required and is bound into the manifest by sha256, so RL
cannot resume against a base distilled on different bytes.

## `prepare_sft_corpus.py` — large single-epoch SFT corpora

Builds a corpus in the canonical answer-fence schema
(`instruction_corpus_answer_fence/v2`) from published instruction data, for
the one-epoch-over-a-lot regime that replaced three-epochs-over-a-little.

```bash
.venv/bin/python -m postraining.prepare_sft_corpus \
  --source ultradata_sft_2605 --source openmathinstruct2 --source math_drills \
  --shards 8 --seq-len 5120 --workers 16 \
  --output postraining/data/sft_mix_ud2605_omi2_drills_v1.parquet
```

Build every corpus in **one invocation**. Deduplication spans adapters, so
overlapping sources resolve there — UltraData's Math half is largely
OpenMathInstruct-2 derived, and listing UltraData first keeps its reasoning
trace and drops the short duplicate.

### Adapters

Two record shapes, declared per adapter rather than sniffed:

- `parquet_columns` — flat columnar corpora (`openmathinstruct2`,
  `math_drills`).
- `ultradata_chat` — openbmb JSONL chat records. **Only the `think` split is
  admitted.** `reasoning_content` becomes the `<think>` body and the
  presented answer supplies `<answer>`. The `no_think` split has no reasoning
  field at all, so admitting it would mean synthesising a `<think>` body.
  Math answers come from the balanced `\boxed{}` span; Code answers are the
  final fenced program.

### Rows are dropped, never reshaped

Everything here fails closed. A row whose problem carries an unregistered
instruction clause, whose provenance is missing or unrecognised, whose answer
span would be empty, or which restates an evaluation problem is dropped and
counted in the manifest.

Provenance is an **allowlist** (`known_sources`). A denylist fails open: a
renamed or newly added evaluation-derived source in an unmirrored shard would
train by default.

### Decontamination is two rules, not one

Math rows use the shared exact-plus-any-8-gram index from
`prepare_sft_traces` (GSM8K test, DeepMind interpolate, AIME 2024/2025/2026).

Code rows need their own rule, because KodCode is an evaluation set here and
programming statements share heavy boilerplate — "the first line of the input
contains two" — plus digit-run grams out of example I/O blocks. Folding
KodCode into the binary index rejected **466 of 800** distinct UltraData code
problems while catching **zero** actual duplicates. `code_contaminated` uses
an overlap *fraction* instead: measured, real KodCode problems score exactly
1.000 (200/200 sampled) against median 0.004 / p99 0.092 / max 0.308 for
distinct problems, so the 0.30 threshold catches every duplicate with a 3.3x
margin and loses about 1 problem in 800.

### Mixed-domain corpora and the sampling gate

Code rows carry `gradeable = False`. The trainer's sampling gate scores every
panel row with the MATH verifier, so a program in `final_answer` would be
wrong by construction and would deflate gate accuracy and the mixed-group
rate the RL stage is sized from. `split_holdout` keeps ungradeable rows out
of the panel; they still train and still count toward holdout CE. Raise
`--holdout-problems` for a mixed corpus — the trainer errors rather than
silently gating on a short panel.

### `--workers`

Screening and tokenizing are pure and order-preserving; deduplication and the
caps stay in the parent, so the corpus is byte-identical for any worker count
(verified for 1 vs 8 and 1 vs 16). The pool uses an explicit `fork` context:
Python 3.14 defaults to `forkserver` on Linux, which re-imports the module and
would leave the shared decontamination index unset in the workers.

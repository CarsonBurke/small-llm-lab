# Causal Workspace Machine (CWM): architecture and test plan

Status: plan only (v2, 2026-10-01). No code or GPU jobs yet. v1 was
red-teamed by a subagent. This version fixes:
- false init-equivalence;
- a weakened lattice control;
- task shortcuts;
- optimistic throughput;
- switches that did not isolate their claims.

This plan turns `synthesis.md` §1–3 into one pretrainable architecture. Each
claim maps to a *read mask* or a switch in a single implementation, so every
comparison shares code, parameters, data, and memory capacity.

## 0. Hypotheses

| Claim (synthesis.md) | Operational hypothesis | Decisive comparison |
|---|---|---|
| 1. Fixed-depth lattices cap per-token serial depth. The maximal causal DAG (any event reads any earlier event) turns depth into time. | Letting events read *higher* rounds at earlier tokens gives categorical gains on tasks that need state-dependent sequential computation beyond what log-depth doubling allows, at matched parameters, FLOPs, and memory. | CWM-full vs CWM-lattice. Same entries, same banks, same parameters; **only the read mask differs.** |
| 2. x -> x+4 is an addressing and bandwidth problem. | Source-addressed reads of earlier rounds beat an additive residual bus, and source-tagged memory beats untagged memory, at matched state bytes. | Depth read on/off; tag on/off; single-writer vs shared-writer at equal state bytes. |
| 3. The optimum reads any earlier event, uses an adaptive event count, keeps two memory kinds, and commits once per token. | (a) Storing all events beats storing a subset. (b) Learned halting beats fixed and random R at matched mean R. (c) Exact plus compressed memory dominates either one on the state-bytes Pareto front. (d) Commit-once is a semantics guarantee, enforced by contracts and not tested as a quality arm. | §4 switches |
| 4. Composition principle. | The parallel perception tier costs no quality against an all-sequential core at matched FLOPs. beta in (0, 2) raises state tracking in both arms. | P = 2 vs P = 0; beta range |

## 1. Architecture

Notation: d = 512, the tokenizer is the GPT-2 padded vocabulary of 50304,
and training uses T = 1024. An event is (t, r), where t is the observed token
and r the round, with r = 1..R_max = 6 untied round blocks. Round R_max + 1
denotes the final output h_{t,R_t}.

### 1.1 One uniform memory interface

Every event (t, r) writes from its **input** state x_{t,r}, the round's state
after its depth read and before any memory read. This is how a transformer
computes K/V from the layer input. It writes:
- **one exact entry** into the store E: key, value, plus a source tag
  embedding for round r. Entries are written by rounds in the writer set
  W = {1, 3, 5, out}, where `out` writes from the final h_{t,R_t};
- **one delta-rule write** into its own associative bank M_r: 4 heads of
  64 x 64 in fp32, per-key-channel decay, beta in (0, 2).

These writes are **committed after token t is scored** (commit-once). Within
a token, rounds communicate only through the workspace h and the round log
Lambda_t.

Event (t, r) reads with three separate mechanisms, each with its own
normalization and output projection:

1. **Depth read.** Softmax over Lambda_t = [u_t, h_{t,1}, ..., h_{t,r-1}],
   with round-identity key embeddings.
2. **Exact read** over committed entries of E, through sparse block
   selection (§1.4).
3. **Bank read** from M_j.

**Legality classes.** Every committed entry and bank write has a source
round s and a token t' < t. Current-token perception entries count as source
0 with t' <= t.
- **Lattice-legal:** s <= r. Reading the input of round s <= r, as a
  transformer layer r reads K/V computed from inputs of layer r.
- **Feedback:** s > r, including `out`. A higher round at an earlier token:
  the missing edge.

Legal and feedback sources are read by **separate** attention and bank-mix
paths, each with its own softmax and normalization. They are summed through
separate output projections, and **O_fb is zero-initialized.**

**Placebo control.** A third configuration feeds the feedback path entries
and banks from a *different stream in the batch*: same shapes, parameters,
and capacity, no information. This separates "feedback information" from
"extra parameters and read capacity".

### 1.2 Perception tier (parallel, P = 2)

- Layer 0 is KDA and layer 1 is causal softmax attention, both run with
  chunked kernels over the whole sequence.
- Output: u_t, plus source-0 (token) entries in E.
- It never sees feedback.
- P = 0 is the claim-4 ablation.

### 1.3 Round block r

1. Depth read.
2. **Write proposal** from x_{t,r}: E entry (if r ∈ W) and bank delta,
   buffered.
3. Exact read, legal path + feedback path.
4. Bank read, legal path + feedback path.
5. MLP, hidden size 2048.
6. Halting head (§3, Stage 3).

After R_t rounds:
- logits = Head(norm h_{t,R_t}), with the embedding tied;
- commit buffered writes for rounds 1..R_t (rounds not run write nothing),
  plus the `out` entry.

### 1.4 Sparse exact reads (the "sparse, content-addressed reads" of the optimum)

E holds up to |W| + 1 = 5 entries per token, so dense decode-attention per
round is unaffordable at T = 1024. Each read path instead uses:
- a dense local window of the last 64 tokens' entries;
- **top-k = 8 blocks** of 64 entries, chosen per head by scoring
  mean-pooled block keys (NSA/MoBA-style selection). Selection is computed
  separately per legality class, so the lattice arm selects from legal
  blocks only.

Stage 2 runs at short T with **dense** reads, so selection cannot confound
the mechanism tests. Stage 3 runs selection in both CWM arms.

### 1.5 Exact properties (Stage 0 contracts)

- **Init equivalence:** with O_fb = 0 at step 0, CWM-full ≡ CWM-lattice
  ≡ CWM-placebo **bitwise** at the same seed, because the feedback paths
  have separate softmax and normalization.
- **Lattice parallelizability:** with feedback masked, every write
  depends only on rounds ≤ r and the current token. Each bank is then a
  KDA-style scan with exclusive reads (writer fields shifted right, inclusive
  kernel), and each entry set is computable layer by layer. CWM-lattice
  therefore trains in **parallel mode**. Sequential mode must equal parallel
  mode.
- **Causality and commit-once:** perturbing tokens > t leaves logits ≤ t
  unchanged, and writes from token t have zero influence on events at t.
- **Halting:** commit-once with variable R_t is well defined. Skipped rounds
  write nothing; `out` is written from h_{t,R_t}.

## 2. Execution: exact sequential training

There is **no Jacobi approximation**: training, evaluation, and decoding all
run the same per-token recurrence (synthesis §4: finite-pass schemes are
depth recurrence, not this).

**Schedule per microbatch of B streams:**
1. Perception tier, in parallel.
2. Lock-step core sweep over t.
3. Head and CE in parallel over stored final states.

**Backward:** full BPTT over T. Bank states are checkpointed every C = 16
tokens (6 x 64 KiB per stream, so 64 checkpoints x 96 MiB ≈ 6 GiB at
B = 256). The chunk is recomputed under a custom autograd Function.

**Store-attention backward** (the dominant cost, per the red-team):
- Each read saves its output and log-sum-exp (LSE) in the forward, so
  recompute never re-reads the store for the forward values.
- Backward runs **one flash-style backward per chunk** over all C x
  (reads per token) queries against the selected and window blocks. Each K/V
  block is read once per chunk, not once per token, and dK/dV accumulate
  into a single fp32 buffer.
- Stock autograd would materialize a [B, entries, d] gradient per read, about
  28 GiB per chunk, which is infeasible.

**Kernels** (Triton, hand-written backward):
- sparse-plus-window decode attention with device-side lengths
  (CUDA-graph-safe);
- multi-bank read with legal/feedback split;
- fused delta commit with beta in (0, 2);
- a fused per-round kernel or persistent kernel to remove the 200–400
  launches per token step (an unfused graph would mean about 600k autograd
  ops per update, seconds of CPU time).

**Realistic budget:** 8–15 s/update at 524,288 tokens after kernel work,
7–13x KDA8's 1.153 s.

- **Stage 0** times a naive PyTorch prototype first.
- **Stage 1 gate:** ≤ 15 s/update and ≤ 28 GiB reserved.
- **Fallback:** shrink d and T *for every arm*. The recurrence is never
  approximated.

CWM-lattice in parallel mode should be ≤ 2 s/update. That is also why the
**doubled-depth lattice** control below is affordable.

## 3. Stages

### Stage 0: reference, contracts, prototype timing (CPU, then mlq GPU)

`pretraining/cwm/` contains `config.py` (architecture id `cwm_v1`),
`reference.py` (literal per-event fp32 loop), `model.py`, `executor.py`
(sequential, chunk checkpointing, parallel-lattice mode), `kernels/`, and
`train.py`. `train.py` is forked from the KDA8 trainer, so the data stream,
packing, reset policy, and validation panel are identical. The audit reads
these from the KDA8 run.

**Contracts:**
1. executor = reference, forward and gradients, with variable R_t across
   chunk boundaries;
2. causality;
3. commit-once;
4. bitwise init equivalence of full, lattice, and placebo;
5. sequential lattice = parallel lattice;
6. streaming parity of the incremental decoder against the training
   forward (0.005 nats/token in bf16, exact in fp32);
7. sparse selection = dense attention when k covers all blocks;
8. CUDA-graph capture with no host syncs;
9. parameter, FLOP, state-byte, and traffic accounting against KDA8;
10. lineage metadata compatible with the SFT trainer;
11. FLA chunk kernel with beta in (0, 2) (sigmoid off) matches the
    reference.

### Stage 1: kernels and throughput gate (§2)

### Stage 2: mechanism suite from scratch (small, dense reads, categorical)

**Model:** d = 256, P = 1, R_max = 4, about 5M parameters, dense exact reads.
T = 512–1024 directly (affordable at this size). Length generalization is
a secondary readout only, because it confounds RoPE and decay extrapolation.
Three seeds per arm and task. Curriculum over chain length, applied
identically to every arm.

**Arms:**
- CWM-full;
- CWM-lattice;
- CWM-placebo;
- CWM-lattice with doubled R (parallel);
- KDA-style lattice and transformer at matched parameters;
- **Feedback Transformer** (Fan et al. 2020; the established form of claim 1);
- **GRU + attention** (positive control for learnability by BPTT).

**Tasks.** Inputs never contain intermediate values: those are targets only,
so CoT-by-teacher-forcing is impossible. Generators are deterministic, with
sealed test seeds.

| Task | Claim | Prediction |
|---|---|---|
| **Iterated permutation lookup.** One n-cycle permutation (n ≥ 128, resampled per sequence) is given in context, then bare `apply` tokens. Target x_k = f(x_{k-1}), with k up to 512. A long single cycle defeats the periodicity (k mod period) shortcut. | 1 | The lattice's accuracy cliff moves with depth (≈ 2^depth hops via doubling, and the doubled-R lattice's cliff moves right). CWM-full stays flat in k. |
| **Branching register machine.** Program in context. Instructions read register values to choose jumps (data-dependent control flow). Bare `step` tokens; target is the register state. | 1 | CWM-full ≫ every lattice; the gap grows with executed steps. |
| Straight-line register programs; S5 word problem | 4 | Input-driven automata. beta (0, 2) and more Householders help the lattice. CWM-full ≫ the single-Householder lattice; a deep enough lattice can close the gap. |
| Parity, mod-3 | 4 | beta (0, 2) ≫ (0, 1) in every arm. |
| **Variable hops per token.** Each query token needs h ∈ [1, 12] chained lookups, with h varying per token. | 3b | A weight-tied looped core with R_max = 16 and learned halting beats fixed R and random R at matched mean R. |
| Multi-step inference that needs an *intermediate* (non-final) round result from an earlier token | 3a | The all-events store (every round writes entries) beats W = {1, 3, 5, out}. |
| Associative recall vs long-gap overwrite retention, swept over state bytes | 3c | Exact + banks dominates the Pareto front over windowed-exact-only and banks-only. |
| Variable-binding chains | 2 | Depth read on > off; tagged > untagged at equal state bytes. |

**Kill rule for claim 1:**
- **Fails** if CWM-full does not beat CWM-lattice and the doubled-R lattice
  on iterated lookup and the branching machine, *while a positive control*
  (Feedback Transformer or GRU+attention) does.
- **Inconclusive** if every positive control also fails. That is a
  learnability/curriculum problem, to be fixed before Stage 3.
- **CWM-placebo must not match CWM-full.** If it does, the gain is capacity,
  not feedback.

### Stage 3: LM pretraining at KDA8 scale

Same data, tokenizer, T = 1024, 2,000 updates x 524,288 tokens, schedule, and
validation panel as KDA8.

**Arms:**
- CWM-full, 2 seeds;
- CWM-lattice (parallel), 2 seeds;
- CWM-placebo;
- CWM-lattice with doubled R;
- KDA8 (existing, 1.1824). Its inclusive reads make it slightly stronger
  than the exclusive-read CWM lattice, which is stated, not hidden.

**Ablations at 1,000 updates** (only the ones that survive Stage 2 with a
signal):
- depth read;
- tagging and writer split at matched state bytes;
- writer set W vs all-events;
- P = 0;
- beta range;
- halting (fixed vs learned, at matched mean R).

**Diagnostics** (canonical metrics.jsonl and TensorBoard):
- eval-time feedback ablation (zero O_fb, scored causally);
- hop profile: cut feedback from tokens more than Δ ago, for Δ ∈ {1, 4,
  16, 64, 256};
- read mass by source and legality per round;
- bank and block-selection entropy;
- halting distribution;
- state-Jacobian spectral estimate;
- per-chunk gradient norms.

**Pre-registered readings:**
- Feedback is *used* if eval ablation costs > 0.01 BPB and the hop profile
  shows dependence beyond Δ = 4.
- It *helps LM* if CWM-full beats both CWM-lattice seeds and the placebo by
  > 0.005 BPB.
- Wall-clock-matched BPB is reported but does not gate. Synthesis §4
  predicts a small BPB effect.

### Stage 4: reasoning after SFT

The SFT trainer and sampling gate get a CWM model-registry adapter (decode
equals the training step; streaming parity is a contract). Each of CWM-full,
CWM-lattice, and CWM-placebo gets one epoch of
`sft_mix_omi2_drills_v2.parquet`, with job 9040's recipe at T = 1024.

**Headline:** an **answer-only (no-CoT) reasoning panel**, plus the Stage 2
categorical tasks, re-run on the SFT checkpoints. This needs a new
versioned eval schema; the canonical think-fenced contract is left
unchanged. With CoT, the lattice already gets serial depth that grows with T
through tokens, so CWM's advantage there can only be bandwidth.
Pre-registered: "bandwidth, not reachability" for CoT panels.

**Secondary:** arithmetic probe (against 9040's 0.516) and the sampling gate.
Synthesis §6 says arithmetic is not a categorical separator.

CWM becomes a candidate post-training base only if it beats job 9040 on held
out panels. Until then, 9040 stays canonical.

## 4. Switches (one implementation)

| Switch | Values | Claim |
|---|---|---|
| `read_mask` | full / lattice / placebo | 1 |
| `lattice_depth` | R / 2R (lattice only) | 1 (control) |
| `depth_read` | attn / none | 2 |
| `memory_tags` | on / off | 2 |
| `bank_writers` | single / shared (state bytes matched) | 2 |
| `writer_set` | {1,3,5,out} / all rounds | 3a |
| `halting` | fixed / random / learned (matched mean R); `core` untied R=6 / tied R=16 | 3b |
| `memory` | exact+banks / exact-window / banks, swept over state bytes | 3c |
| `beta_range` | (0, 1) / (0, 2) | 4 |
| `perception_layers` | 2 / 0 | 4 |

Commit-once is not a switch. It is the semantics, verified by contracts 3
and 6.

## 5. Risks

- **BPTT through ~6,000 serial steps.** Mitigations: zero-init feedback,
  bounded beta and alpha, normalized reads, curriculum. The Jacobian
  estimate and gradient norms are monitored. Any clip becomes an arm.
- **Unused feedback** (STATE_POOL; the Jacobi lines' ~2 hops). The eval
  ablation, hop profile, and placebo detect it. Stage 2 gates Stage 3.
- **Throughput.** The Stage 0 prototype timing and Stage 1 gate catch it.
  Fallback: shrink every arm.
- **Distribution shift.** Learned halting at evaluation after
  stochastic-depth training is a shift: train with the learned policy's own
  samples once it exists.
- **Hidden capacity asymmetry.** Report active parameters per arm; the
  placebo carries this check.

## 6. Compute budget (re-estimated after Stage 1)

| Stage | Estimate |
|---|---|
| Stage 2 | ~8 tasks x ~9 arms x 3 seeds of 5M-parameter models; ~20–30 GPU-h, sized after the first task |
| Stage 3 | CWM-full x2 and placebo at 2k (≈ 3 x 5–8 h); lattice arms in parallel mode (≈ 4 x 1 h); ablations at 1k (≈ 6 x 3–4 h). ≈ 45–70 GPU-h |
| Stage 4 | 3 SFT runs plus evaluations |

# Synthesis: what the optimum for causal communication actually is

Written 2026-10-01. It builds on `theory.md`, `architecture_literature.md`
and `local_architecture.md` in this folder, and an adversarial subagent
red-teamed it. The earlier notes are hedged surveys; this note commits to a
position. It separates what is provable, what local measurements show, and
what is conjecture.

## 1. The right object: the event DAG

Index every computation event by (t, l), meaning observed token t and layer l.
An architecture is a choice of which earlier events each event may read, plus
how each read is addressed.

- **Transformer.** (t, l) reads (t', l-1) for all t' <= t. Every edge goes up
  one layer, so the longest dependency path has length L whatever T is. That
  one fact gives parallel teacher forcing, and it also bounds the work any
  single token can do. Transformers sit in uniform TC0 (Merrill and
  Sabharwal 2023, "The Parallelism Tradeoff").
- **Maximal causal DAG.** (t, l) reads any (t', l') that comes earlier in
  lexicographic order. This graph equals one processor running over a
  flattened stream of T*L events, with attention to every earlier event. It
  is the Feedback Transformer taken to its limit: depth turned into time.
  Chain-of-thought already reaches this graph's serial depth, about O(T*L),
  through sampled tokens (Merrill and Sabharwal 2024). Latent flattening adds
  only one thing: it removes the log2 |V| bits-per-step bottleneck of a
  discrete token.

The user's two complaints map onto different missing structure.

1. **x -> x+4 within a token.** The identity residual already delivers an
   exact copy of x to x+4. What is missing is separability and selection:
   - everything is superposed in one d-wide additive bus;
   - no reader can address a contribution by its source;
   - PreNorm progressively dilutes early writes.

   This is a bandwidth and addressing problem, not a reachability problem.
2. **(t, x+4) -> (t+1, x+1).** This is a genuinely missing edge. In
   isolation it is just one more read. Its real significance is that
   feedback edges let serial depth grow with T, which turns a constant-depth
   circuit into an unbounded-depth one.

## 2. The abstract optimum

"Optimal" has no meaning without priced resources. The statistical target is
the predictive sufficient state; `theory.md` covers this, and nothing below
changes it. The architectural optimum is the solution of:

    min  E[log loss]  +  price(serial depth)  +  price(work)  +  price(state bytes)

That problem has no single solution, only a Pareto frontier. Still, the
frontier has a definite shape, and the optimum has definite properties:

1. **Any event may read any earlier event (the maximal DAG), with sparse,
   content-addressed reads.** Do not broadcast everything everywhere. Every
   read carries source identity: which layer or role wrote it, and how old it
   is.
2. **Adaptive event count per token.** Serial depth is spent where the input
   needs it, such as hard reasoning steps or state updates, and not on easy
   tokens. A fixed L is a training-parallelism artifact.
3. **Two memory types, because counting forces it.**
   - Exact episodic records (KV-like) for arbitrary detail, which needs
     growing state: an M-bit state cannot recall N > M arbitrary bits.
   - Compressed, editable associative state for beliefs and tracked
     variables.
4. **One commit per token.** Writes from token t's internal events become
   persistent only once token t is scored. Internal rounds use a scratchpad.
   This makes causality and train/inference equivalence well defined.

All four properties raise serial depth or growing state, and both cost
training throughput. So the real question is which subset to buy, and in
which regime.

## 3. The governing constraint: composition, not linearity

Parallel training time is lower-bounded by the longest dependency chain,
unless the per-step maps compose with a representation of bounded size. If
they do, a chain of T steps can be reduced by a scan in O(log T) depth.

- **Examples of such maps:**
  - affine maps with input-dependent, state-independent coefficients (linear
    attention, Mamba, DeltaNet, KDA, GDN2);
  - finite-automaton transition functions;
  - max-plus maps;
  - set/reset gates.
- **Not a theorem.** Without such structure, believing that a nonlinear chain
  cannot be parallelized rests on TC0 != NC1-style conjectures, not on a
  proof.
- **Parallel is not constant-depth.** Scans reach NC1. A delta rule with
  beta in [0, 2] (eigenvalues in [-1, 1]), or DeltaProduct, solves S5 word
  problems that transformers and positive-eigenvalue SSMs cannot (Grazzi et
  al., ICLR 2025; Siems et al. 2025).
- **Newton or DEER-type solvers for genuinely nonlinear cells now work at
  scale when the cell is contractive.** ParaRNN (Danieli et al., ICLR 2026)
  trains 7B nonlinear RNNs. Contractivity is the price, and it conflicts with
  long memory.

The design principle follows: **route long temporal paths through
composable operators with expressive transitions, keep nonlinearity on short
paths, and pay sequential cost only where it is already being paid.**

The cross-time, cross-depth edge (t, top) -> (t+1, bottom) closes a loop
through the nonlinear stack. It therefore falls on the wrong side of this
principle in teacher-forced training. Section 5 covers where it does not.

## 4. What finite-pass feedback really is (deflationary, with evidence)

Under K-pass Jacobi feedback, pass k+1 reads pass k's top states at earlier
tokens. That dependency graph is a subgraph of a K*L-layer stack. In a 2L
stack, layer L+j at token t already reads layer L at every t' <= t,
including the current token. Relative to that stack, 2-pass feedback adds
only constraints and regularizers:

- weight tying;
- re-injecting the input embedding;
- an auxiliary loss at mid depth (loss = pass 1 + pass 2);
- a strict one-token shift;
- optionally, dense reads from mid depth into every deep layer.

In short, finite-pass feedback is depth recurrence (Universal Transformer or
Huginn style), not recurrence over time. True T-scaling feedback exists only
when the model runs sequentially.

The local data agree:

| Run (nanogpt-mini 6L/512d) | BPB | Training time |
|---|---:|---:|
| plain, 1k updates | 1.3433 | ~529 s |
| first-layer LAM feedback, 1k (p1/p2/p8 = 1.3426 / 1.3265 / 1.3257) | 1.3265 | ~1,096 s |
| six-site LAM feedback, 1k | 1.3243 | ~1,344 s |
| plain, 2k updates (`nanogpt_mini_fullval_2k`, 62M-token val panel) | 1.2856 | ~1,071 s |

Three readings follow.

- **The read is genuinely used.** Pass 2 beats pass 1 by 0.016 on the same
  weights, and pass 1 equals plain, so the trunk did not change.
- **Matched wall clock, plain wins by about 0.04 BPB.** The two runs use
  different validation panels, but that gap dwarfs the panel offset of about
  0.004. Feedback buys less than the extra tokens it displaces.
- **The learned feedback map converges in about 2 hops.** Pass 2, pass 3 and
  pass 8 are nearly equal, and 3-pass training lost to 2-pass at update 400.
  Feedback chains longer than about 2 carry almost nothing. The T-scaling
  serial depth that feedback promises goes unused under next-token
  pretraining on web text. Web text rarely needs it: most BPB comes from
  lexical and recall structure that a constant-depth lattice already handles.

The routed state pool beat its GDN2 control, 1.3495 against 1.3586, with
routers that never specialized. That is a capacity or regularization effect,
not evidence about communication topology.

The compute-matched controls planned on 2026-09-17 were never run: base_4k,
a looped 6L, and an untied 12L. They are the cheapest way to confirm or kill
this section's claim directly.

## 5. Where the missing edge is worth buying: the sequential regime

Generation and RL rollouts are already token-sequential. In that regime a
(t, top) -> (t+1, bottom) edge adds no per-token serial latency. The cost
moves entirely into training credit assignment. This is also where serial
depth matters (reasoning), and where the discrete token channel's few bits
per step is the binding bottleneck. The repo's `carry` mode is exactly this
edge.

The current carry objective is a semi-gradient. Stored carries are detached,
so no gradient crosses the carry. The objective is the score-function
estimator with token log-probs, and it is unbiased only if the gradient of
log-pi is the total derivative through the carry chain. What is computed now
is the 0-hop truncation, which is not the gradient of any objective.

A cheap, principled improvement, given that the rollout recorded the true
carries:

1. **Pass A** replays with the recorded carries as constants. Its forward is
   exact at theta_old, up to bf16 drift between decode and replay.
2. **Pass B** uses pass A's recomputed h_{t-1}, not detached, as its carry
   input. Its forward values equal pass A's on-policy, but its gradient
   includes every path with at most one carry hop.
3. **K passes give at most K hops.** Put the loss on the last pass only, to
   avoid K-fold reweighting of the direct terms. The exact gradient needs
   K = T, because the carry Jacobian is nilpotent. Truncation is biased
   unless the combiner is contractive. The 2-hop convergence in section 4
   suggests that small K captures most of it.

Caveats:

- After the first PPO epoch, pass A prices tokens under old carries and new
  theta. This is R2D2's stale-stored-state problem, so refresh or burn in.
- Cost grows by a factor of K in replay compute and activation memory.
- Carries break speculative-decode verification. Prompt prefill is
  unaffected because prompts carry nothing.

## 6. Prescription

**Pretraining** (teacher forced, so stay parallel):

- **Temporal mixers.** Use composable recurrences with expressive
  transitions, plus a little softmax attention for exact recall. KDA
  currently uses beta = sigmoid, so its eigenvalues are in [0, 1]. Allowing
  beta in [0, 2] is a one-line, low-risk test of state tracking: parity,
  mod-k counting, and S5.
- **Depth communication.** Use dynamic weighted reads over earlier layer
  outputs (DenseFormer, MUDDFormer, attention residuals). It is cheap and
  parallel, and it fixes addressing. Expect little at 8 layers, since
  reported gains concentrate in deep stacks. Hyper-connections widen the
  residual stream, which costs memory traffic and stability.
- **Recurrence.** Spend extra compute on depth recurrence (looped blocks)
  only if it beats an untied deeper model at matched wall clock.
- **Feedback channels.** Include them zero-initialized, so the pretrained
  model is exactly the parallel model.

**Post-training and inference** (sequential anyway):

- Activate nonlinear cross-time feedback here: carry, then an addressed
  memory over past top states rather than only h_{t-1}.
- Train it with the at-most-K-hop replay above.

**Measurement.** BPB is the wrong yardstick for this edge.

- **Categorical separators:**
  - S5/A5 word problems (NC1-complete);
  - parity and mod-3 counting, which probe eigenvalue sign;
  - long-horizon variable tracking.
- **Multi-digit arithmetic is not a separator.** Addition is in AC0 and
  iterated multiplication is in TC0, and their carries are aperiodic. The
  `mul_decimal` drift happens with CoT. It is a learnability and
  length-generalization problem, not missing communication.
- **Report alongside:**
  - matched wall clock;
  - a feedback-ablated evaluation of the same weights (permute or zero the
    channel);
  - the sequential-vs-Jacobi gap.

## Gaps

- The compute-matched pretraining controls do not exist.
- The plain-2k comparison uses a different validation panel.
- None of the section 5 or 6 proposals have been run.
- Whether nonlinear feedback helps reasoning accuracy, rather than BPB, at
  64M parameters is unknown. It is the experiment that matters.

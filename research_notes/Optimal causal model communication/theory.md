# The abstract optimum for causal model communication

## What does optimal communication actually mean?

### Takeaway
The statistically ideal model maintains a recursively updateable state sufficient for the entire conditional future. The engineering ideal additionally computes and communicates that state cheaply enough to learn it under the actual budget. These are different optimization problems; there is no established architecture that solves both optimally for language.

### Cited Findings
- Computational mechanics defines causal states by equivalence of histories' conditional future distributions and establishes the causal-state representation as minimal among representations consistent with accurate prediction. This is a statement about the stochastic process and predictive representations, not a proof that a particular neural architecture trains optimally. — [Shalizi and Crutchfield, 2001](https://csc.ucdavis.edu/~cmg/compmech/pubs/cmppss.htm)
- Predictive state representations encode dynamical state through predictions of future observations. The original paper establishes a linear predictive representation with no more predictions than the states of a minimal POMDP representation. — [Littman, Sutton and Singh, 2001](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html)
- Optimal causal filtering extends rate-distortion ideas to a hierarchy of approximate causal representations; relaxing model complexity recovers the exact causal-state partition under the ideal known-distribution setting. — [Still, Crutchfield and Ellison](https://arxiv.org/abs/0708.1580)

### Inferences
All mathematical statements below are direct reasoning, not empirical claims about Parameter Golf.

Let H_t be the observed token history through token t. Define S_t by grouping two histories if they imply the same distribution over every possible future continuation. Then an ideal predictor can use S_t instead of the entire history, and S_t can be updated by S_(t+1) = U(S_t, token_(t+1)). This is the right abstract target. It combines remembering, inference and belief revision. A posterior over latent world states is one construction if the generative model is known; a predictive state is another, which does not require a chosen interpretation of the latent world.

The important qualification is **entire future**, rather than current next-token logits. Two histories may have exactly the same next-token distribution but different distributions two tokens later. For example, a bit disclosed earlier is irrelevant until a later query marker arrives. A next-token-only state can forget the bit prematurely. Any recursively used compression must preserve information for later updates as well as for the current readout.

For a deterministic compression S = f(H) and an unrestricted next-token decoder, the best expected log loss using S is H(Y|S). The best using H is H(Y|H). Their difference is I(Y; H | S). In words: the irreducible penalty of the representation is precisely the useful predictive information it discarded. This does not include decoder approximation, parameter estimation, optimization failure or finite data effects. It also does not justify estimating this mutual information naively in a neural experiment.

A corresponding objective for an engineered model is to minimize predictive loss plus priced memory, compute, communication and sequential latency, with parameter bytes and training wall clock as hard constraints when appropriate. One compact description is:

`min E[-log2 q(Y|state)] + a * memory + b * operations + c * communication + d * serial_latency`

subject to the actual archive, hardware and training constraints. Weighting resources differently changes the optimum. In this repository, an expressive graph that reduces loss at equal FLOPs but processes far fewer examples in ten minutes may be worse for the task.

The communication principle is **preserve and expose useful predictive information at the cheapest point where it can affect a decision**. It is not 'send all information everywhere.' Unrestricted communication can lower graph distance while increasing bandwidth, interference and the difficulty of learning which messages matter. An optimum can suppress nearly all edges for an easy input and enable additional communication for an ambiguous one.

An M-bit, finite-precision persistent state cannot distinguish more than 2^M possible histories. For the task 'read an arbitrary N-bit string, then answer any later index query exactly,' all 2^N strings must remain distinguishable; therefore M >= N. A fixed-size state cannot be an exact replacement for growing history across all tasks. This is an elementary counting argument. Infinite-precision real-valued states evade the premise mathematically but do not describe hardware. The implication is not that finite state fails on natural language; the implication is that exact episodic recall and compressed inference need distinct resource policies.

### Gaps
- We do not know the minimal causal-state complexity of real language under the challenge's distribution. It could be extremely large or not admit a useful finite exact representation.
- No paper here establishes that conditional-future equivalence classes are cheaply learnable from a finite corpus. Minimal information is not necessarily minimal update compute or easiest gradient optimization.
- There is no experimentally established optimum for the repository's parameter and time budget. The proposed design below is a reasoned candidate, not an assertion of superiority.

## What is the transformer's structural limitation, and what would remove it?

### Takeaway
The user's missing edge is real when t denotes token position: conventional causal attention routes earlier-token representations forward in depth but does not feed their completed late-layer states into current early-layer computation. Adding that edge is causally legal, but creates a sequential dependency that conventional teacher-forced training avoids.

### Cited Findings
- The original Transformer replaces recurrence with attention and obtains training parallelism from this design. — [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- The Feedback Transformer directly targets the missing connection: future low-level states can read earlier high-level states. It stores a learned weighted sum of the layers at each previous token and shares key/value projections across depth. The authors explicitly describe token-by-token sequential computation as the tradeoff. These experiments are precedents, not evidence for this repository. — [Fan et al., 2021](https://arxiv.org/html/2002.09402v3)
- Universal Transformers use attention with repeated refinement in depth, including a per-position adaptive stopping mechanism. Depth recurrence alone does not imply completed-token feedback into a future token's first layer. — [Dehghani et al., 2018](https://arxiv.org/abs/1807.03819)
- Mamba uses input-dependent state-space parameters and a hardware-aware parallel algorithm. Its example demonstrates that structured recurrence and efficient training can coexist; it does not establish efficient scanning for arbitrary nonlinear state-dependent feedback. — [Gu and Dao, 2023](https://arxiv.org/abs/2312.00752)

### Inferences
Use three distinct indices: t = observed token; r = internal refinement round; u = optimizer update. The user's edge is about t. If t instead means u, the weights and optimizer state do carry information across training updates, while arbitrary examples do not have a causal sequence relationship. Reusing example activations across updates is a separate problem involving stale weights and data boundaries.

A causal transformer computes h_(t,l) from the current lower-depth state and lower-depth states at positions <= t. Its causal graph has monotone depth. The identity residual offers a straightforward forward path from l to l+4, but residual addition superposes contributions into a fixed-width vector; it does not give them independently addressable persistent identities. Calling forward transmission intrinsically 'difficult' requires evidence. The sharper, provable graph point is that h_(t,l+4) cannot influence h_(t+1,l+1) inside the usual computational graph.

This matters most when newly integrated information should affect how the next observation is parsed, queried or interpreted. The model can recompute a similar conclusion at higher depth for the next token, but this is repeated inference rather than reuse of the completed inference. The transformer can also use a generated token as an indirect channel from the top at t to the bottom at t+1. That channel is lossy, constrained by required language output, and absent as a learned activation route under teacher forcing. A generated word is not a satisfactory persistent representation of all inferred state.

Causality forbids using an unseen token, not using previous-token conclusions. A legal design may make h_(t,L) directly readable by h_(t+1,1). But computing all low-layer states simultaneously is no longer valid, because h_(t+1,1) must wait for h_(t,L). For a simple L-round token-by-token graph, the straightforward critical path through a length-T sequence is proportional to T * L rather than just L. Batching independent documents helps utilization without removing each document's dependency chain. Sharing weights saves parameter bytes without automatically saving sequential operations.

Structured linear/affine updates can often be composed with associative scans. An arbitrary learned function that reads prior state, retrieves memory and chooses nonlinear writes generally lacks a compact cheap associative summary. Composition is mathematically associative, but representing a growing composed function may cost more than the direct recurrence. Therefore 'just parallel scan it' is not an adequate answer.

There are intermediate schedules: feedback only across chunk boundaries, delayed feedback, fixed rounds of parallel causal refinement, or a fast scan state augmented by a slower nonlinear persistent workspace. Each removes some edges or delays them in exchange for throughput. They approximate the desired semantics and should be evaluated as such.

### Gaps
- The fraction of language-model loss caused specifically by absent late-to-early feedback, rather than finite width, limited training, attention policy or optimization, is unknown here.
- This agent did not inspect gdn2_pool_pinned_1k; another researcher owns repository-specific analysis. No claims about its precise state or mask semantics are made here.

## What detailed candidate best approximates the ideal, and why might it fail?

### Takeaway
A reasonable candidate is a persistent, typed predictive workspace with multiple memory forms, queried and updated by a small recurrent processor. It should support immediate causal feedback and optional iterative revision, while using hardware-friendly schedules wherever their restricted semantics are sufficient. This is a proposed family, not the unique optimum.

### Cited Findings
- ACT learns how many internal computational steps to use before emitting an output. Its original character-level language modeling experiment did not obtain large performance gains, a useful caution against assuming adaptive computation improves language compression automatically. — [Graves, 2016](https://arxiv.org/abs/1603.08983)
- Deep equilibrium models express a fixed point of a weight-tied transformation and train it by implicit differentiation. Constant memory with respect to effective depth does not mean zero-cost iterations or bounded sequence memory. — [Bai, Kolter and Koltun, 2019](https://arxiv.org/abs/1909.01377)
- Subsequent DEQ work explicitly identifies slower execution, architectural brittleness and instability, and proposes Jacobian regularization. — [Bai, Koltun and Kolter, 2021](https://proceedings.mlr.press/v139/bai21b.html)

### Inferences
Suggested conceptual state:

These are capability distinctions, not an asserted optimal fixed partition by age. Retention should depend on future predictive utility, reversibility of the compression, and read/update costs. An old precise identifier can deserve exact storage while a recent redundant paragraph can be compressed immediately. A learned hierarchy may allocate capacity dynamically across these forms. Keeping the entire raw history as a fallback changes the runtime memory budget and cannot be treated as free.

1. A short exact episodic memory of recent inputs and selected inferred records, preserving arbitrary details that compressed state cannot reliably reconstruct.
2. A compact persistent belief/workspace, partitioned into multiple slots with keys, values, age and confidence metadata rather than only one pooled summary.
3. A slower compressed memory for broader context whose retention policy is learned or otherwise constrained by predictive usefulness.
4. A temporary scratchpad for the current token's inference. Scratchpad capacity and persistent storage capacity are separate choices.

A concrete tensor interface could use W_t in R^(K*d_w) for K workspace slots, A_t in R^(N_t*d_a) for addressable episodic records, C_t in R^(J*d_c) for compressed contextual records, and Z_(t,r) in R^(P*d_z) for P scratchpad slots. K, J, P and widths are capacity knobs; N_t must have a resource limit or an explicitly charged growth policy. Address keys and retention metadata are additional state, not free labels. Small read adapters map their heterogeneous widths into the processor space. The ambitious target is a processor whose operations are retrieve, compare, infer, revise, retain and emit; the execution graph follows the inference problem rather than a fixed semantic hierarchy of layers.

Four things should remain conceptually distinct: memory content (what is retained), addressing/control (which record is read and why), recurrent compute workspace (where conclusions are integrated and revised), and learning future utility (which information is worth preserving or computing). Improving only the first can still leave a model unable to find relevant state; improving only connectivity can still fail because the state was overwritten. A next-token objective can in principle train retention through later losses, but long credit paths make this hard. Auxiliary multi-horizon prediction or distillation can shorten those paths, while introducing their own objective mismatch and compute cost. They are hypotheses, not mandatory architecture components.

On receiving token x_t, initialize the scratchpad from the token and the prior persistent state. For each permitted refinement round:

`queries = Q(scratchpad, prior_workspace)`

`reads = Retrieve(queries, episodic_memory, workspace, compressed_memory)`

`proposals = Processor(scratchpad, reads, x_t)`

`scratchpad = GatedUpdate(scratchpad, proposals)`

`workspace = GatedWrite(workspace, proposals)`

Produce the next-token distribution after sufficient refinement, then commit the predictive state for token t+1. At least one read happens before early lexical/contextual processing, which provides the desired previous-late-state-to-current-early-state edge. Every module can query the workspace; specialized modules can use different adapters so that the same state does not force identical interpretations across depth.

Teacher forcing provides x_(t+1) in advance, but does not provide W_t: the learned inference still has to compute W_t before the dependent early computation at t+1. Knowing all observed tokens does not remove this serial edge. Predicting approximate W_t in parallel would be a separate state-prediction/distillation scheme with an approximation error to measure, rather than exact parallel execution of the feedback model.

The read/write order must be explicit. All synchronous participants read a snapshot and their proposals are combined into the next snapshot. An asynchronous implementation specifies which writes are visible to later readers. Calling an all-to-all graph 'instantaneous' hides a fixed-point or scheduling problem. Feedback from a later module at the same token can reach an earlier module only in a subsequent round; same-round circular dependencies are not a valid feedforward computation.

Confidence should describe uncertainty in the belief, not merely the next-token entropy. A high-entropy output can be irreducible randomness and offer no value from extra thinking. An appropriate adaptive controller predicts the marginal reduction in expected future loss from another computation, priced against its cost. Even that value is expensive to estimate; fixed refinement rounds are the better first ablation before introducing a learned halting controller.

Persistent hypotheses can be revised. If an earlier sentence ambiguously identifies an entity, a later observation may update a workspace entry and invalidate a dependent summary. A append-only KV cache cannot revise old embeddings directly. However, writing a new correction record plus retrieval that prefers the correction can sometimes be cheaper and easier than in-place dependency tracking. Optimality does not mandate mutable memory everywhere.

Separate message identity from additive interference. Multi-slot keyed memory allows two useful pieces of evidence to coexist rather than average together. It can still suffer collisions, slot competition, duplicated records and retrieval mistakes. Concatenating all layer states preserves more information than a weighted sum but increases bytes and read cost; learned pooling only helps if it retains what later queries need. The compression policy must be tested with later predictive consequences, not just reconstructing one selected activation.

For Parameter Golf, start with the smallest missing edge: one gated summary from completed chunk n entering early computation in chunk n+1; retain the existing token memory. It tests whether reusable prior inference is valuable without combining every proposed feature. Then test multiple independently addressable summary slots, earlier read placement and one extra recurrent refinement separately. Full token feedback is the semantic reference but might be unsuitable for the ten-minute training budget. These are suggested ablation questions; no workload was launched and no implementation was changed.

Red-team critique:

- Immediate full feedback converts parallel training into a long sequential dependency chain. It may reduce expressivity costs while losing the wall-clock contest badly.
- A small workspace can become a harsher information bottleneck than separate layer caches. Distinct slots do not establish sufficient predictive state.
- Recurrence can propagate an error indefinitely; useful closed-loop correction and destabilizing feedback share the same edge. Gates, normalization and bounded rounds mitigate but do not prove safety or fidelity.
- Repeated refinement does not necessarily converge, and forcing contraction for convergence can suppress long-term memory or useful sharp transitions. Conversely, algorithms need not converge to a static equilibrium to be useful; imposing a fixed point can be the wrong inductive bias.
- Cross-token gradients can become longer and less stable. Detached memory avoids backward cost but changes credit assignment. Backpropagating further costs memory and training time.
- Specialized memory can memorize poorly, leak across document boundaries, or infer a retention rule appropriate only to the training context distribution. Reset behavior is part of model semantics.
- Conditional computation and sparse routing may save abstract FLOPs but worsen GPU utilization. Dynamic halting can also make later state construction inconsistent unless the training and inference schedules agree.
- A richer graph increases the solution space but can make the desired solution harder to learn. Strict graph inclusion does not imply lower validation loss after a fixed training budget.
- The best statistical compression of history may be hard to compute. Sometimes redundant storage is faster and more learnable than minimal state. Therefore minimal-state theory is a target for predictive sufficiency, not a command to remove every redundancy.

### Gaps
- No evidence establishes the optimal slot count, chunk size, number of rounds, retention policy, confidence representation or update schedule. Those must be selected by matched-budget ablations.
- No universal architecture is best independently of task distribution and priced resources. An independent-token source needs no memory; arbitrary random-index recall needs growing distinguishable memory; nonlinear world-state updates require a different update capability. These constructed examples already make a universal practical optimum ill-posed.

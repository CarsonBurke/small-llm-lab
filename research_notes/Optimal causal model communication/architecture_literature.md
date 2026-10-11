# Ideal causal communication: architecture and literature

## What is the ideal causal computation graph, and what already addresses it?

### Takeaway
The defensible optimum is a predictive sufficient state with unrestricted useful access to past computations, plus a computational budget: it is not a particular finite architecture. My proposed engineering target is a recurrent, addressed workspace with preserved episodic history, selective reads/writes, and an explicitly causal compute schedule. Its advantage over this repository's baseline remains hypothetical.

### Cited Findings
- Feedback Transformer directly identifies the user's concern: higher-layer states from earlier tokens are inaccessible to lower layers at later tokens. It stores a learned weighted sum of all depth states per token, which every subsequent layer can attend to. Its exact forward computation is token sequential. Shared K/V projections reduce cache redundancy; the paper does not establish that arbitrary feedback retains ordinary parallel teacher-forcing cost. [Fan et al., 2020/2021](https://arxiv.org/html/2002.09402)
- Universal Transformer repeats a shared self-attention/transition block across depth and can halt adaptively per position; it preserves parallelism over sequence positions. Weight sharing and iterative depth are its relevant contributions, but they do not by themselves make completed deep states at token t available to shallow states at token t+1. [Dehghani et al., 2018](https://arxiv.org/abs/1807.03819)
- Hyper-Connections uses multiple residual streams with learnable depth and width mixing. This lets multiple patterns of previous-depth features survive without being forced into a single accumulating sum. [Zhu et al., 2024](https://arxiv.org/html/2409.19606)
- Attention Residuals uses input-dependent softmax attention over previous depth outputs; Block AttnRes trades full depth-history retention for block summaries and lower communication/memory cost. This provides direct depth retrieval but remains a feed-forward depth graph. [Kimi Team, 2026](https://arxiv.org/abs/2603.15031)
- Huginn applies a shared recurrent-depth block repeatedly and trains a large proof-of-concept model to benefit from additional latent computation at evaluation. This establishes feasibility in its setting, not benefit at Parameter Golf's size and training duration. [Geiping et al., 2025](https://arxiv.org/abs/2502.05171)

### Inferences
These are original architectural proposals, not established results:

1. Define time t as the index of an observed input token. State after x_t predicts x_(t+1). Time is not optimizer-update number; activation persistence across shuffled training batches is not coherent document state.
2. For ideal prediction, two histories belong to the same sufficient-state equivalence class when they induce the same conditional distribution over every future sequence. The ideal state identifies that class. There is no guarantee it has a bounded finite representation. Exact history is a universal sufficient representation; compression helps only when it preserves task-relevant distinctions.
3. The unconstrained statistical optimum is the actual data conditional. The constrained engineering optimum minimizes expected held-out log loss over artifact size, training time, evaluation time, device memory, and learnability. Giving the graph additional communication cannot establish improved optimization or score. A Pareto frontier is the correct claim; a single provably optimal architecture is unavailable.
4. A residual is not incapable of delivering depth-x information to x+4. Identity paths are strong transport. The concern is selective access to independently preserved representations, competing updates, normalization, and finite bandwidth. Do not explain residuals as necessarily destructive addition: a network can allocate subspaces and maintain features.
5. Keep three objects distinct: current token working state W_t, persistent editable state S_t, and episodic archive E_<t. Editable memory compresses facts and updates beliefs; archive retains exact previous observations/computations for details that compression may omit. The archive can store raw tokens and selected representations, trading recomputation against cache bytes. Storing all intermediate states is an unrestricted ideal, not the first implementation.

An exact sequential specification:

```text
Given x_t, S_(t-1), E_<t:
  W_t^0 = Embed(x_t)
  S_t^0 = S_(t-1)
  for microstep r = 1 ... R_t:
    each active specialist reads W_t^(r-1), S_t^(r-1), E_<t
    specialists emit candidate updates and proposed writes
    route/combine those proposals into W_t^r and S_t^r
  score x_(t+1) from W_t^R and S_t^R
  commit S_t = S_t^R
  append selected completed representations to E
```

All modules in one microstep read the same previous microstep version; writes become visible at the next microstep. This avoids undefined instantaneous cycles. Alternatively, modules execute serially and immediately commit writes, providing Gauss–Seidel semantics; that is a different schedule and must be trained/evaluated as such. Adaptive R_t must depend only on the observed prefix and current internal state. Same-token top-down feedback needs another microstep; previous-token top-down feedback is already available at step zero. Updating a past belief inside S is causal; retroactively changing scores already assigned is not.

A concrete associative bank for H heads, key width d_k, value width d_v, and K banks:

```text
M_j has shape d_v by d_k
q = unit_norm(Wq_j norm(h))
read_j = M_j q
bank weights = softmax(query(h) dot bank_key_j + type/age biases)
read = sum_j bank_weight_j * read_j

k = unit_norm(Wk_j norm(h_writer))
v = Wv_j norm(h_writer)
a in (0,1], b in [0,1] from writer state
M_decay = a * M_j
error = v - M_decay k
M_j_new = M_decay + b * error * transpose(k)
```

This updates a selected association toward a new value instead of simply accumulating outer products. Parameter tying can share Wq/Wk/Wv across depth while retaining cheap per-depth adapters. Multiple writers need ordered microsteps or a defined simultaneous merge; applying all errors against the old bank and summing is not equivalent to sequential delta writes. A few exact slots for exceptionally important states can complement the associative banks. Give layers separate query/read gates and let depth outputs write with source-depth and age metadata. Do not force everything through a single pooled vector.

The state feedback channel should have a gated residual path, e.g. h_next = h + gate(h, read) * Project(read), with normed inputs and a small nonzero initialized gate. Preserve a raw episodic route so an overly contractive memory need not erase essential evidence. Global contraction is not an unconditional objective: for arbitrary sequences it conflicts with preserving distinguishable histories indefinitely at finite precision.

### Gaps
- No source proves this architecture optimal or shows a Parameter Golf BPB improvement. Choosing dimensions, schedule, routing sparsity, or halting policy requires measurements.
- Full per-layer archive access has state-storage, routing, and attention costs; no finite representation can promise lossless storage of arbitrary long histories at fixed precision.
- Better depth access, recurrence, and editable state are separate hypotheses; combining all at once would obscure which is useful.

## Can unrestricted feedback be made parallel and train/inference consistent?

### Takeaway
Exact arbitrary nonlinear feedback changes the sequential dependency graph. Few parallel refinement passes can be a useful causal architecture, but they neither remove unrestricted temporal recurrence nor guarantee that training matches indefinite sequential inference.

### Cited Findings
- Full-bandwidth Transformer fuses the previous top-layer hidden state with the next token embedding through a gated linear unit. Its exact token recurrence is sequential; training uses shifted hidden states from previous whole-sequence passes as a multi-pass approximation. The authors retain losses on all passes and allow gradients through carried states. Two-pass-only training is unstable under deeper compositions in their experiments; mixing a small fraction of three-pass batches stabilizes observed rollouts. Prefix mixing addresses plain-prompt versus fused-generation boundaries. They describe observed contraction-like behavior, not a general convergence theorem. [Wang et al., August 2026](https://arxiv.org/html/2608.08888)
- Deep Equilibrium Models solve a weight-tied network's fixed-point equation and differentiate implicitly, with activation memory independent of effective depth. Solvers, fixed-point existence, and backward conditioning remain substantive requirements. [Bai et al., 2019](https://arxiv.org/abs/1909.01377)
- DEER parallelizes nonlinear sequential models using Newton linearization and associative linear solves. Its stated dense-Jacobian costs are O(n² T P) memory and O(n³ T P) time, where n is state width and P the number of shifted arguments. Convergence can fail from a poor initial guess. Their strongest acceleration claims cannot be extrapolated to wide transformer states for free. [Lim et al., 2023/2024](https://arxiv.org/html/2309.12252)
- Gated DeltaNet supplies an efficient structured parallel algorithm for memory updates that combine global decay and targeted delta corrections. The important structural property is recurrence linear/affine in the memory when keys, gates, and values are supplied independently of that recurrent memory. [Yang et al., 2024/2025](https://arxiv.org/html/2412.06464v3)

### Inferences
Consider exact recurrence s_t = F(x_t, s_(t-1)). A Jacobi iterate s_t^(r+1) = F(x_t, s_(t-1)^r) moves a feedback dependency across at most one adjacent-token recurrence edge per pass. From a zero initial sequence, exact dependence on s_0 reaches position t only after t iterations in general. A transformer F can also attend far backwards, so a few passes already have long-range *attention* access; they still do not execute arbitrarily many chains of nonlinear feedback transitions. Do not equate those two distances. A nilpotent causal Jacobian can have spectral radius zero and still large transient sensitivity: causality does not prove a few-pass approximation is accurate.

There are three honest contracts:

- **Exact sequential:** train and evaluate the token-sequential recurrence. Parallelize batches/documents and operations within a token. Gradient truncation can approximate credit assignment while leaving forward state semantics exact, but efficiency may be unacceptable.
- **Exact finite-pass:** define K causal refinement passes as the model itself, and use the same passes at training and evaluation. Every pass rebuilds its own cache from its own states. Do not reuse baseline K/V as if fused inputs were identical. Incremental generation can maintain K streams of caches; there is no justification for switching silently to a single unlimited feedback stream.
- **Approximate recurrent:** train a small number of passes or short unrolls and evaluate a longer exact recurrence. Measure sequence-level state discrepancy, loss discrepancy, and stability versus an exact sequential reference. Empirical robustness is the criterion, not a claim of architectural equivalence.

An exact scan-friendly compromise is s_t = A_t s_(t-1) + b_t, with A_t and b_t calculated from a parallel causal backbone that does not depend on s_(t-1). Associative scan composes affine transforms. However, if that backbone's top-layer output reads this same state before producing A_t/b_t, the overall loop becomes nonlinear again; a mathematically affine local update does not rescue the globally circular computation graph.

A useful two-sweep construction is: sweep A computes all token representations from a conventional causal backbone; writes from sweep A form a scanned shared memory; sweep B queries that memory from every layer. This gives deep-A-token-t to shallow-B-token-(t+1) access and is fully parallel and consistent. Sweep-B writes cannot feed back into sweep-A without another sweep or token-sequential scheduling. It should be described as a bounded communication circuit, not complete feedback.

### Gaps
- Stable fixed-point iterates do not guarantee useful representation, favorable backward conditioning, or preserved long-term memory.
- No evidence guarantees DEQ/DEER wins at 16MB and ten-minute training; dense state Jacobians are particularly unattractive here.
- Small-pass stability may be obtained by suppressing feedback, so stable losses and state changes alone do not prove the channel is used.

## What is a credible constrained experiment and how should it be red-teamed?

### Takeaway
First test whether selectively preserved depth states help; then test a single, inexpensive path from completed representations into later shallow computation. Parameter Golf's time constraint rewards scan-friendly or bounded two-sweep graphs before unrestricted token recurrence.

### Cited Findings
- The official current README counts code plus compressed model weights toward 16,000,000 bytes; evaluation has a separate ten-minute limit on 8×H100. It allows arbitrary evaluation sequence length and test-time learning only on validation tokens already scored. No separate fixed scratch-state byte cap is stated there; runtime still must fit available hardware and time. [OpenAI Parameter Golf README](https://raw.githubusercontent.com/openai/parameter-golf/main/README.md)

### Inferences
Suggested sequence of hypotheses, each with its own matched 1000-update run:

1. Content-dependent depth retrieval: keep 2–4 block-level states and give downstream blocks scalar or low-rank input-dependent reads. This addresses interference/selection cheaply and preserves full sequence parallelism. Start without adaptive depth, episodic bank changes, or cross-token nonlinear feedback.
2. Two-sweep shared-memory feedback: first sweep's selected deep states produce keys/values; second sweep's shallow blocks query a strictly earlier-token prefix memory. Tie the backbone if parameters require it; low-rank read/write adapters add about O(d r) weights rather than O(d²) per depth. Compare with an equal-cost two-sweep control carrying only embeddings/low-depth states, so extra compute alone cannot explain gains.
3. Chunk-recurrent state: divide each document into chunks of B=32 or 64 tokens; process tokens within a chunk in parallel, while all layers read the previous completed chunk's deep-memory summary. Commit once per chunk. This gives exact feedback with lag up to B tokens; time depth becomes T/B rather than T. Training and evaluation must retain the same boundaries and reset policy. Measure boundary effects and sequence-length sensitivity.
4. Only after a promising result consider exact token recurrent feedback on a reduced-width, fused kernel or DEQ solver. This is a separate research branch and can easily fail the wall-clock budget.

For every experiment record BPB, compression round-trip BPB, parameter artifact bytes, step time, tokens seen, measured evaluation time, peak memory, and multi-seed variability for finalists. The local >0.005 BPB gate at 1000 optimizer updates is necessary for retention but does not establish a ten-minute win. Equal-update comparisons give slower variants more wall clock; add matched-training-time comparisons before scaling. No GPU runs were started for these notes.

Red-team controls:

- Block, permute, or zero the feedback path at validation to test whether it carries useful information, keeping evaluation causal and scoring the intervention honestly.
- Compare deep-state write versus embedding-state write under identical memory bytes/projections; this isolates abstraction depth from additional state capacity.
- Compare genuine addressable slots versus a pooled vector with identical total state width; this isolates selective access from raw extra width.
- Compare shared-depth parameters plus more iterations against untied shallower parameters at identical artifact size and measured wall clock; parameter reuse is not compute reuse.
- Validate full-sequence causal evaluation against incremental evaluation of the declared schedule. A tiny CPU recurrence fixture can catch scheduling or mask mistakes without model workloads.
- Perturb unseen future tokens and verify earlier logits do not change. Exclude x_(t+1) from memory committed before its score; prefix summaries, pooling, or recurrent chunk carries can silently leak it.
- Reset at document boundaries and match reset/burn-in policies between ablation, training, evaluation, and generation. Otherwise state improvements may arise from unintended document history or from a different forward model.
- For approximate recurrent training, compare K=1,2,3,4... passes with exact sequential forward on a short prefix and track discrepancy, rather than interpreting stable refinement as equivalence.

### Gaps
- The optimal state capacity, slot layout, chunk size, number of specialists, and communication budget are unknown. No proposal here meets the repository's measured-improvement criterion until run.
- A wall-clock estimate from existing kernels does not establish H100 throughput for new routing, recurrent, or two-sweep kernels. Systems engineering may dominate the outcome.
- A rich global workspace can become a shared choke point. Its benefit depends on separation of roles, addressability, useful read gates, and preserving independent evidence; denser connectivity alone can make noise and credit assignment worse.

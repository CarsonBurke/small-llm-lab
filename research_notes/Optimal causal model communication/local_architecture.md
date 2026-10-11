# Local architecture: measured communication, limitations, and a proposed successor

## What does gdn2_pool_pinned_1k actually communicate, and what did it establish?

### Takeaway
The measured model gives a second private-model pass access to a pool built entirely from first-pass computations. This establishes a useful causal cross-layer communication path, but does not implement a self-consistent recurrent workspace. Its gain cannot be attributed entirely to pool reads.

### Cited Findings
- The historical snapshot has six private GDN2 blocks and twelve shared four-head K128/V128 banks, with event order `(token, layer)`, inclusive reads, top-1 routes, and private-layer write fields copied into the selected shared bank. All pool queries/routes/writes come from each first-pass normalized block input. — [Snapshot port construction](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/gated_delta_pool.py:260), [ordered pool execution](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/gated_delta_pool.py:338), [run configuration](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/config.json:71).
- Pass two rebuilds the private model from embeddings, recomputes its private recurrence, and adds precomputed pool reads. Its pool query, route, and write controls are not recomputed from feedback-informed hidden states. Only its readout gate is contextual to pass two (except block zero, whose private computation can be reused). — [Second pass](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/gated_delta_pool.py:279), [complete passes](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/gated_delta_pool.py:361).
- Hard argmax has no derivative through the selected assignment. Task loss can train selected-bank confidence; balance loss trains probabilities. The read is multiplied by the selected probability, around 0.1 in the documented run, after scale-invariant normalization. — [Routing and confidence](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/gated_delta_pool.py:270), [read scale](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/gated_delta_pool.py:283), [diagnosis](/home/marvin/Documents/repositories/parameter-golf/pretraining/gated_delta/STATE_POOL.md:197).
- Canonical final BPB is 1.3494630174, versus the decision-gate control's final 1.3585655147: improvement 0.0091024973. The same checkpoint's no-pool first pass scores 1.3520259252, so adding reads at evaluation improves it by only 0.0025629078. Final summed balance loss is 5.999111, largest per-sequence bank share 0.103917. — [Canonical result](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/result.json:8), [control score](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/config.json:65), [canonical final metrics](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/metrics.jsonl).
- This was one seed, 1,000 updates, 524,288 training tokens/update, full attached 1,024-token BPTT, and 1,048,576 validation tokens. The pool has 27,904,536 parameters and required 2,433.78 training ms/update. Throughput report speedup against plain mini was 0.231897; the actual model file was 110,633,176 bytes. There is no demonstrated 16 MB artifact or 8-H100 ten-minute result. — [Configuration](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/config.json:94), [timing/artifact](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/result.json:11), [throughput](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/config.json:42).
- Incremental decoding is explicitly unsupported in this snapshot. — [Cache rejection](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/gated_delta_pool.py:389).
- **Current source is different from measured source**: it now uses shared pool projections, per-bank full-rank adapters, top-2 reads with weights renormalized to one, optional single-layer writers, and narrower default pool width. These improvements must not be retroactively credited to the historical run. — [Shared projections](/home/marvin/Documents/repositories/parameter-golf/pretraining/nanogpt_mini/gated_delta_pool.py:265), [adapters](/home/marvin/Documents/repositories/parameter-golf/pretraining/nanogpt_mini/gated_delta_pool.py:311), [routing](/home/marvin/Documents/repositories/parameter-golf/pretraining/nanogpt_mini/gated_delta_pool.py:356), [defaults](/home/marvin/Documents/repositories/parameter-golf/pretraining/nanogpt_mini/gated_delta_pool.py:397).
- Nearby final-to-first feedback achieved 1.3228 BPB versus first-layer LAM 1.3265, below the >0.005 retention threshold. Adding a third training pass was pruned at update 400: 1.4566 versus two-pass 1.4510, 48.33% more training time. These results argue against assuming more iterations automatically help. — [Feedback record](/home/marvin/Documents/repositories/parameter-golf/pretraining/gated_delta/FEEDBACK.md:3), [third-pass decision](/home/marvin/Documents/repositories/parameter-golf/pretraining/gated_delta/FEEDBACK.md:113).

### Inferences
- Write `A_l,t` for first-pass state and `B_l,t` for second-pass state. The actual graph permits `A_l,t1 -> pool -> B_j,t2`; it never permits `B_l,t1 -> pool -> B_j,t2`. Thus memory can transport advanced first-pass features to early second-pass computation, but cannot accumulate conclusions already improved by its own feedback. This follows directly from the complete-passes code above.
- A fixed two-pass augmentation is a coherent causal model in its own right, not inherently erroneous. It could be streamed exactly by retaining separate first- and second-pass private caches, pool state, and first-pass short-convolution caches, processing each token first through A then through B. That would preserve the measured graph; feeding B writes back would define a different model. Cache support is absent, so this is a proposed implementation semantics, not demonstrated parity.
- The 0.00654 BPB remaining gap between pool-model pass one and the standalone control plausibly comes from changed loss/gradient/training dynamics. It does not prove the pool is useless: feedback learning can improve the backbone even when the final inference read has a small marginal effect. Neither pass comparison isolates extra parameter count, auxiliary loss, or compute.
- Uniform balance/load is evidence of balanced aggregate routing, not proof of no token-dependent specialization. The record's stronger diagnosis is plausible but would require route mutual information, content-stratified bank ablations, and bank-swap tests to establish.

### Gaps
- No multi-seed uncertainty, matched-compute control, causal intervention separating pool training from readout benefit, or streaming-parity evidence is present for this run.
- The current adapters and top-2 policy were inspected as source, not established as measured quality improvements here.

## What are the actual communication constraints, and how should a recurrent workspace be red-teamed?

### Takeaway
The central tradeoff is feedback-dependent computation versus sequence-parallel execution, with memory capacity and address compatibility as additional constraints. A universal shared matrix can move the bottleneck rather than remove it.

### Cited Findings
- The repository's own feasibility analysis states that GDN2 is affine in preceding state **conditional on supplied controls**, while final-state-dependent controls make the entire processor/memory loop nonlinear and not scan-parallel. — [Affine update](/home/marvin/Documents/repositories/parameter-golf/pretraining/gated_delta/LATENT_FEEDBACK_DESIGN.md:19), [nonlinear complete model](/home/marvin/Documents/repositories/parameter-golf/pretraining/gated_delta/LATENT_FEEDBACK_DESIGN.md:47).
- It explicitly warns that full-width values are superposed associations, not separately recoverable full-width records, and proposes a separate latest-latent path. — [Compression warning](/home/marvin/Documents/repositories/parameter-golf/pretraining/gated_delta/LATENT_FEEDBACK_DESIGN.md:41).
- Exact token-layer event execution was documented at roughly 156 s/update versus the packed pool's roughly 2.4 s/update. This measures one implementation, not a hardware lower bound. — [Sequential experiment](/home/marvin/Documents/repositories/parameter-golf/pretraining/gated_delta/STATE_POOL.md:36), [actual run timing](/home/marvin/Documents/repositories/parameter-golf/ablation_results/gdn2_pool_pinned_1k/result.json:14).
- The proposed causal latent fixed point has a strictly lower-triangular sequence Jacobian and a unique finite-horizon solution, but may require up to T Jacobi rounds; nilpotence does not imply good conditioning or few-round convergence. Finite primal/adjoint solves give approximate states/gradients. — [Solver caveats](/home/marvin/Documents/repositories/parameter-golf/pretraining/gated_delta/LATENT_FEEDBACK_DESIGN.md:94), [adjoint caveats](/home/marvin/Documents/repositories/parameter-golf/pretraining/gated_delta/LATENT_FEEDBACK_DESIGN.md:112).
- Repeated contextual reads cannot simply reuse the existing fused scan for free; reusable write factorization plus separately differentiable reads requires kernel work. Materializing every B64/T1024/H4/K128/V128 FP32 memory costs 16 GiB. — [Kernel/storage analysis](/home/marvin/Documents/repositories/parameter-golf/pretraining/gated_delta/LATENT_FEEDBACK_DESIGN.md:159).

### Inferences
- Residual paths are already short forward channels: the deeper layer receives a direct identity-carried sum. Their real limitations include feature superposition, shared finite width, and lack of protected source identity—not a guaranteed inability to convey information over four layers. Dense source-aware reads would address separability, but cost memory/bandwidth and can overwhelm readers.
- All-to-all communication is not automatically optimal: it incurs message production, selection and interference. A shared recurrent matrix is particularly vulnerable to incompatible writer key/value spaces, destructive erase/decay from unrelated layers, high-frequency writers overwriting long-lived facts, queries reading their own redundant writes, and soft mixtures losing distinctions. Single-writer or typed-bank interfaces reduce these risks without asserting they improve BPB.
- A useful semantic distinction is token time versus internal computation round versus optimizer update. Previous-token deep feedback is causally legal; same-token high-to-low feedback requires an additional internal round or a specified fixed point. Current graph's same-token lower-to-higher visibility does not constitute same-token upper-to-lower visibility.
- More nonlinear feedback usually increases the training dependency span. Choosing affine input-dependent controls preserves efficient scans precisely by restricting which earlier conclusions can affect later write decisions. This is a tradeoff, not a free operator substitution.
- A fixed point is not a necessary ideal. A finite number of explicit internal rounds defines an exact trainable model and avoids solver mismatch. Convergence/halting still costs compute, and large loop gains can damage temporal credit or amplify quantization errors.
- Symmetry breaking alone is insufficient. Bank parameters can differ without meaningful functional diversity; apparently different adapters can be changes of coordinates. The current orthogonal initialization is additionally modified by per-head structure and coordinate-wise gates, so it should not be described as exact function-preserving global basis invariance. Reader/writer diagnostics matter more than a bank-identity slogan.

### Gaps
- Local runs do not determine whether memory capacity, key alignment, finite-round stale reads, training allocation, or optimizer dynamics are the dominant quality bottleneck.
- No argument here establishes a globally optimum architecture or universal lower bound for specialized hardware.

## What concrete successor best matches the user's abstract objective?

### Takeaway
My proposed communication target is a causal, source-aware workspace with separate persistent memory and temporary token reasoning, explicit commit semantics, and protected high-bandwidth access to recent deep conclusions. This is a research hypothesis, not a recommended retained experiment or predicted BPB gain.

### Cited Findings
- Local proposed semantics already specify pre-token snapshot reads, a direct previous-final-latent path, and a final commit that becomes visible only on subsequent tokens. — [Causal proposal](/home/marvin/Documents/repositories/parameter-golf/pretraining/gated_delta/LATENT_FEEDBACK_DESIGN.md:47).
- Historical diagnosis proposes single-writer banks, shared key projections, bank adapters, full-scale reads and top-2 read-only events as candidate repairs. — [Pool diagnosis](/home/marvin/Documents/repositories/parameter-golf/pretraining/gated_delta/STATE_POOL.md:197).

### Inferences
Define persistent state `P_(t-1)` as (recent full-width conclusion records, typed associative banks, optional compact task/belief state). Define token workspace `W_t,0` from the current embedding plus direct previous-final conclusions. For explicit internal round r:

```text
requests_i = Request_i(token workspace, read-only P_(t-1))
messages_i = Read_i(P_(t-1), W_t,r, requests_i)
proposal_i = Specialist_i(local state_i, messages_i)
W_t,r+1 = Merge(W_t,r, source-tagged proposals)
```

After R rounds:

```text
logits_t = Head(W_t,R)
P_t = Commit(P_(t-1), selected source-tagged conclusions from W_t,R)
```

Exact serial implementation semantics (one token commit, no repeated aging):

```text
P = true incoming persistent state
for token x_t:
    snapshot = P                          # immutable view for this token
    W = Initialize(x_t, snapshot.latest)   # speculative scratch, not P
    for round r in 0..R-1:
        reads = Read(snapshot, W)          # may query differently each round
        proposals = Process(W, reads)
        W = MergeScratch(W, proposals)     # only scratch changes
    logits_t = Head(W)
    controls = Writer(W)
    P = Commit(snapshot, controls)         # exactly one decay/erase/write
    P.latest = FinalConclusion(W)
```

If a parallel solver instead proposes a whole sequence of final latents Z,
every solver iteration must recompute `Memory(Z)` from the **same true incoming
persistent state**, applying each candidate token update once in candidate
token order. It must not apply a new candidate update to the previous
iteration's already-advanced memory. Solver iterations refine speculative
trajectories; only the solved/final trajectory represents token progression.
That distinction prevents repeated decay/writes from changing the model just
because the solver needed more iterations.

- Each specialist has a preserved addressable result slot; readers choose which sources and fields to retrieve. A forward residual stream remains useful locally, but no longer carries every communication channel in one undifferentiated sum. Separate source identity from content; learn adapters into common addresses while protecting write ownership where appropriate.
- Keep recent records exact and older history compressed when resources permit. Associative memories store summaries/beliefs; exact records preserve lexical detail, uncertainty distinctions and rare facts. No fixed-size bounded state can losslessly preserve an arbitrarily long arbitrary history at finite precision. This architectural partition follows the documented superposition limitation, not a measured best allocation.
- Freeze persistent state within a token; write temporary proposals to workspace and commit once. This avoids order-dependent persistent overwrites and defines strict causality cleanly. Within-token rounds let early processing use deep conclusions from previous rounds; token-to-token commit lets early processing use all previous-token conclusions. If exact `(token, layer)` event semantics are preferred, specify them instead and acknowledge layer-order bias.
- Allow multiple contextual reads at different processing stages. A query formed before any feedback cannot ask the question suggested by that feedback. Read routing should be distinct from write routing: information useful to several readers should not need duplication into every reader's selected bank.
- Gate message acceptance and write/erase independently. A routing probability should select/weight retrieval, not automatically shrink every message to one twelfth of its intended scale. Learn expected future predictive utility of writes through future CE; a reconstruction objective alone does not establish usefulness.
- Preserve training/inference semantics. Either train the explicit finite-round model and stream it exactly, or solve and differentiate a specified recurrence with quantified primal/adjoint error. Do not advertise a two-pass approximation as an exact recurrent solution. Ordinary BPTT/checkpointing is the correctness reference; alternative parallel solvers earn their place through measured speed and streaming-quality parity.
- In a 16 MB/<10-minute setting, the likely practical search direction is much smaller: one protected previous-final latent path, a few single-writer banks, reader-specific adapters, and one feedback round. Search equal-token and equal-wall-time fronts separately. The unconstrained architectural ideal and the challenge's feasible optimum can differ sharply.

### Gaps
- No dimension, bank count, number of rounds, routing policy, compression schedule or BPB payoff is inferable as optimal from current local data. All numerical allocation choices require ablations.
- A minimally decisive next comparison would hold backbone initialization, token budget, loss weighting and added parameter budget constant across: two private passes; protected final-latent carry; one shared memory; typed single-writer memory; feedback-informed requery/rewrite. Also compare equal training seconds and check streaming parity. This is proposed experimental design only; no jobs were submitted.

# Four Delightful papers: mechanisms, guarantees, and FineWeb guard implications

## What do the four papers actually establish?

### Takeaway
The papers support two useful mechanisms: suppress unlikely negative updates and redistribute learning toward useful surprises. They do not establish improvement of a different corpus's cross-entropy. Every task-CE theorem concerns the rewarded task's correct target, and the neural shared-parameter counterexample explicitly blocks a universal extension.

### Cited Findings

All four are already present and indexed in `../reading-list/INDEX.md` lines 35–42, with complete exact-version HTML editions, independent original-PDF page editions, manifests and audits. No additions, renames, conversions or index edits were necessary. Metadata was verified against primary arXiv pages. Figures and appendices were inspected, including figure contact sheets saved alongside these notes. — [Library index](../../../reading-list/INDEX.md)

#### 1. Delightful Policy Gradient — Ian Osband, 15 March 2026, 2603.14608v1

Source: [full paper](https://arxiv.org/html/2603.14608v1), [metadata](https://arxiv.org/abs/2603.14608v1), [local edition](../../../reading-list/markdown/2026-osband-delightful-policy-gradient/paper.md).

- Exact rule: action surprisal ell = -log pi(a|history); advantage A; delight A*ell; gate w = sigmoid(A*ell/eta); update sums w*A*grad log pi. Eta = 1 in experiments. The gate is a weight on the score term, not an extra differentiable factor in a naive weighted surrogate. It is deliberately biased relative to PG. For positive surprise it approaches 1; negative surprise approaches 0; very common actions approach half weight. “Amplification” means relative weighting: individual raw PG terms are never made larger by a gate bounded by 1. — [Section 2](https://arxiv.org/html/2603.14608v1#S2)
- Local variational interpretation: maximize delight*w + eta*binary_entropy(w), giving a sigmoid and softplus optimal value. This is entropy regularization over whether to accept an update; it is not a KL penalty that anchors model parameters or policy to pretraining. Optimal policies are stationary if true on-policy advantages vanish on their support. Baseline independence of expected PG does not automatically survive action-dependent gating. — [Appendix A](https://arxiv.org/html/2603.14608v1#A1)
- Single-context variance theorem: deterministic one-correct-action bandit, K >= 3, symmetric incorrect probabilities, baseline b in (0,1). DG preserves expected logit-gradient direction; perpendicular variance is multiplied by the square of the negative gate. Approximate cosine-gap improvement uses concentrated batch gradients and normalized steps. Without symmetry, expected direction changes too; tail suppression remains but the collinearity statement does not. — [Section 4.1](https://arxiv.org/html/2603.14608v1#S4.SS1), [Appendix C.2–C.3](https://arxiv.org/html/2603.14608v1#A3.SS2)
- Multi-context CE theorem: independent contexts with orthogonal score vectors, zero baseline to isolate positive-success reweighting, normalized steps, eta > 1/2. CE oracle = sum v_n; PG oracle = sum p_n*v_n; DG expected update = sum p_n*sigmoid(-log(p_n)/eta)*v_n. At eta=1 this weight is p/(1+p), not 1. The gate compresses the ratio of easy/hard context weights and makes cosine with TASK CE larger, including infinite samples. It does not equal supervised CE or eliminate the rare-success bottleneck. Appendix C.6 extends two-context result to N contexts under these conditions. — [Section 4.2](https://arxiv.org/html/2603.14608v1#S4.SS2), [Appendix C.6](https://arxiv.org/html/2603.14608v1#A3.SS6)
- Empirical tests: MNIST as reward-only classification, synthetic token reversal/copy/flip variants, and 28 Control Suite tasks. MNIST gains persist at 100 samples per context, supporting expected-direction change beyond variance reduction. Neural architecture and optimizer are matched in those diagnostics. Control suite has 3 seeds per environment, 10M environment steps; discrete transformer experiments are very small models, not pretrained internet LMs. Held-out MNIST test improvement is within-task generalization. — [Sections 3, 5, 6 and appendices](https://arxiv.org/html/2603.14608v1#S3)
- Control implementation clips density-based surprisal to [-10,10]; density surprisal is scale dependent and can be negative. This limitation is irrelevant to discrete token surprisal but matters to claims of one universal gate across spaces. Original paper's conclusion leaves transfer to large-scale transformers/RLHF open. — [Section 2.2](https://arxiv.org/html/2603.14608v1#S2.SS2), [Conclusion](https://arxiv.org/html/2603.14608v1#S8)

#### 2. Delightful Distributed Policy Gradient — Ian Osband, 2603.20521v2, revised 13 May 2026

Source: [full paper](https://arxiv.org/html/2603.20521v2), [metadata](https://arxiv.org/abs/2603.20521v2), [local edition](../../../reading-list/markdown/2026-osband-delightful-distributed-policy-gradient/paper.md).

- Same sigmoid update, with learner-relative surprisal rather than actor-relative probability. It is not exact off-policy correction. High-surprisal failures from stale/buggy actors are suppressed; surprising successes remain usable without behavior probabilities. Current learner probability is essential. — [Section 2](https://arxiv.org/html/2603.20521v2#S2)
- Exact algebra for a fixed advantage magnitude c: positive gate = 1/(1+p**(c/eta)); negative gate = p**(c/eta)/(1+p**(c/eta)). Thus negative gate <= p**(c/eta), while positive gate -> 1. — [Section 4](https://arxiv.org/html/2603.20521v2#S4)
- The theory bounds the failure-only perpendicular second moment in a logit-space one-correct-action bandit with baseline 1/2 and contaminated sampler mu = (1-rho)*pi + rho*nu. With eta=1 the squared gate contributes a learner-overlap factor pi(a); contamination noise shrinks as wrong-action probabilities vanish. It does NOT say all negative updates are harmful or establish lower FineWeb loss. A negative update on an incorrect action can have useful component toward the task optimum. Exact importance weighting recovers population PG; finite-batch variance/direction can nevertheless be worse than DG. — [Section 4 and Appendix B](https://arxiv.org/html/2603.20521v2#A2)
- A sign-blind learner-probability gate cannot suppress rare failures without suppressing rare successes with the same probability. Advantage sign is essential to the asserted separation. — [Proposition 2](https://arxiv.org/html/2603.20521v2#S4)
- Positive zero-baseline success terms at eta=1 integrate to a concave success utility log(1+p), compared with PG's p and CE's log(p). This is a modest balanced-success objective, not an implicit FineWeb likelihood objective. — [Appendix B](https://arxiv.org/html/2603.20521v2#A2)
- MNIST uses stale actors and an oracle true-label-dependent expected-reward baseline as a diagnostic control. DG gets about 2% error versus exact-IW PG about 8% at large delay. Zero-baseline uncorrected REINFORCE closely tracks DG, so positive-only learning is a useful comparator but does not explain every DG gain. — [Section 3](https://arxiv.org/html/2603.20521v2#S3), [Appendix A.3](https://arxiv.org/html/2603.20521v2#A1.SS3)
- Token reversal tests staleness, all-zero actor bugs, random Bernoulli reward corruption, injected oracle successes, and their combination. Models are roughly 50K parameters. Actor bugs/oracle intervention probabilities are external mixture probabilities; PPO there is a practical stored-actor-ratio baseline, not exact mixture correction. Combined scaling evaluation removes training frictions and oracle injection, then uses fresh prompts with greedy exact match. — [Section 5](https://arxiv.org/html/2603.20521v2#S5), [Appendix C](https://arxiv.org/html/2603.20521v2#A3)
- False delight is a stated main failure mode: noise, reward hacking, or verifier errors can turn unlikely bad actions into apparent rare successes. Conclusions explicitly describe small diagnostics rather than large distributed systems validation. — [Conclusion](https://arxiv.org/html/2603.20521v2#S7)

#### 3. Delightful Gradients Accelerate Corner Escape — Jincheng Mei and Ian Osband, 12 May 2026, 2605.11908v1

Source: [full paper](https://arxiv.org/html/2605.11908v1), [metadata](https://arxiv.org/abs/2605.11908v1), [local edition](../../../reading-list/markdown/2026-mei-delightful-gradients-corner-escape/paper.md).

- PG can self-trap near a confidently wrong corner: low-probability worse actions depress expected reward, granting the already-common wrong favorite positive advantage; its logit can rise faster than the optimal action's logit. EG is the eta -> 0 positive-advantage-only limit. Dropping negative contributions creates population drift that helps better actions escape. — [Sections 2–4](https://arxiv.org/html/2605.11908v1#S4)
- Local corner guarantee is a sector bound, not unrestricted instant escape: corner mass = 1-epsilon, optimal action probability >= epsilon/K, sufficiently small epsilon depending on gaps/K/temperature. EG optimal-vs-corner logit gap grows at least gap*epsilon/(4K); finite-temperature DG at least gap*epsilon/(8K). Negative contributions are polynomially suppressed with exponent |A|/eta. Better-than-corner actions are allies locally in the tabular geometry. — [Theorems 1–2](https://arxiv.org/html/2605.11908v1#S4.SS2)
- EG first-exit bound is (4K/(gap*initial_epsilon))*log(initial_corner_probability/initial_opt_probability). It applies until parity OR exit from the stated sector. Calling this simply “logarithmic escape in initialization” hides the inverse initial off-corner-mass factor. Finite-temperature paper experiments are faster, but the explicit first-exit theorem is EG. — [Theorem 3](https://arxiv.org/html/2605.11908v1#S4.SS3)
- Main text claims global tabular EG convergence and asymptotic O(1/t); final Appendix D explicitly extends claims to finite-temperature DG via Theorems 7–8/Corollaries 3–4. Do not incorrectly say finite-temperature global convergence is absent. These use distinct reward/unique optimum and, for MDPs, full reachability assumptions. They are not neural LM, Adam, minibatch, inaccurate-advantage or out-of-distribution CE guarantees. — [Main Section 4.4 and MDP section](https://arxiv.org/html/2605.11908v1#S4.SS4), [Appendix D](https://arxiv.org/html/2605.11908v1#A4)
- Exact shared-parameter counterexample: two equally likely states share p = sigmoid(theta). Action 1 gives +10 in state 1 and -100 in state 2; alternative gives zero. PG drift = -45*p*(1-p), so p -> 0, best in class. EG drift = 5*p*(1-p)*(1-11p), giving stable p=1/11. Finite DG eta=1 has an interior root near 0.116. The same action's opposite advantages across shared states interact with suppressed larger negative correction. Rewards here differ from the earlier bounded theoretical assumptions; this remains a meaningful concrete shared-parameter failure example. — [Appendix A](https://arxiv.org/html/2605.11908v1#A1)
- Empirical MNIST recovery from wrong-class CE initialization is much better with DG than PG at 10K recovery steps. This demonstrates robustness to a deliberately bad neural initialization in one task, not universal escape with shared representations. — [Section 6 and Appendix B](https://arxiv.org/html/2605.11908v1#S6)
- Independent reading caveat, not needed to explain local observation: Algorithm 2/eq(3) define a score update whose tabular component is pi(a)*(w(a)*A(a) - sum_b pi(b)*w(b)*A(b)). Appendix D.1 Lemma 8 instead explicitly assumes theta'(a)=theta(a)+alpha*w(a)*A(a), and C.8/C.12 likewise use multiplicative policy tilts. Those are different updates unless specially preconditioned. Also C.7/C.11/D.1 call a thin inequality region “measure-zero”; at finite epsilon such inequalities need not have zero volume. Accordingly these are paper-claimed global results, not independently certified proofs for the implemented score rule. Local tail algebra and the explicit shared-parameter counterexample are stronger foundations for this task. — [Algorithm 2 and equation (3)](https://arxiv.org/html/2605.11908v1#S2), [Appendix C and D](https://arxiv.org/html/2605.11908v1#A3)

#### 4. Delightful Exploration — Ian Osband, 13 May 2026, 2605.13287v1

Source: [full paper](https://arxiv.org/html/2605.13287v1), [metadata](https://arxiv.org/abs/2605.13287v1), [local edition](../../../reading-list/markdown/2026-osband-delightful-exploration/paper.md).

- This is an ACTING algorithm (DE), not the DG gradient estimator. Host chooses greedy/near-greedy action; an epsilon override samples only actions whose expected positive improvement over host baseline times capped host-relative surprisal exceeds price lambda. If gate empty, follow host. Default schedule M/(M+t). Surprisal subtracts the minimum surprisal, making preferred host action zero, then caps at L. — [Section 2](https://arxiv.org/html/2605.13287v1#S2)
- In revealed-value search the gate eligibility is exactly Pandora reservation eligibility with effective inspection cost lambda/surprisal. Pandora ordering additionally needs unthrottled selection by largest reservation index; generic weighted DE sampling does not inherit the full optimality theorem. Noisy bandit EI is a value-of-perfect-information proxy; independent-arm one-step knowledge gradient <= EI. Correlations invalidate that simple proxy inequality. — [Section 2.2](https://arxiv.org/html/2605.13287v1#S2.SS2), [Appendix F](https://arxiv.org/html/2605.13287v1#A6)
- Uniform fresh-arm prior gives pointwise shutoff baseline 1-sqrt(2*lambda/L); gate may reopen when current host baseline drops. Structural results bound post-separation selected overrides and linear information expenditure, not greedy host's overall regret. Fixed-price version can prematurely satisfice. Large-world prior-tail regret-optimal pricing is for a different revealed-value search regime, and its required inspections outgrow default O(M log T) overrides. — [Section 4](https://arxiv.org/html/2605.13287v1#S4), [Limitations](https://arxiv.org/html/2605.13287v1#S5)
- Experiments are Bernoulli/linear bandits and DeepSea tabular MDPs, transferred unit-scale hyperparameters M=100, lambda=.1, L=10. DeepSea EI is a mean-based stale-plan surrogate, not a full posterior expected improvement. Full worst-case regret of greedy-host noisy algorithm remains open. — [Section 3 and Appendix E](https://arxiv.org/html/2605.13287v1#S3)

### Inferences

- Direct relevance ranks: DG original and distributed papers strongest for gradient preservation/reweighting; corner paper supports task confidence/corner mechanism and offers a crucial neural counterexample; exploration paper chiefly offers conceptual host preservation, not causal evidence for our DG-only trainer.
- All four support “beneficial surprise versus redundant surprise”; none supplies language-distribution targets to a training loop that does not observe them.
- Improvements in guard CE require a bridge: shared gradients/representations, altered calibration, correction of pretraining artifacts, or evaluation/configuration effects. Claiming “DG is closer to CE, therefore FineWeb improves” skips that bridge.

### Gaps

- No paper tests FineWeb BPB retention/improvement after a pretrained LM learns unrelated tasks.
- No paper establishes that gate suppresses catastrophic forgetting in neural parameter space, or that it protects the prior policy with a KL-like guarantee.
- Guarantees use exact/local tabular geometry; arbitrary state sharing, sampled reward noise, optimizer moments and coupled output embeddings are unmodeled.

## Which theories best connect these mechanisms to guard BPB?

### Takeaway
The most defensible general explanation is useful cross-task updates becoming visible when destructive/noisy updates are removed. However calibration change and altered effective training scale can mimic representation transfer and deserve explicit controls.

### Cited Findings

- Negative learner-surprise suppression is directly documented and algebraically exact; the empirical comparison with positive-only zero-baseline learning helps isolate it. — [Distributed DG Section 4 and Appendix A.3](https://arxiv.org/html/2603.20521v2#S4)
- Exact task-CE direction rebalancing persists without sampling noise, so larger batch size cannot reproduce every DG mechanism. — [Original DG Section 4.2](https://arxiv.org/html/2603.14608v1#S4.SS2)
- Shared parameters can create either useful cross-context generalization or conflicting corrections; counterexample shows sign-aware selection has no universal external-target preference. — [Corner paper Appendix A](https://arxiv.org/html/2605.11908v1#A1)

### Inferences

1. **Preserved prior plus real useful transfer.** Generated failures contain many context-appropriate tokens whose negative episode advantage penalizes them all. PG may damage broadly reusable language representations while assigning credit too coarsely; DG preferentially drops high-surprisal negative terms. Positive task-relevant updates can still train attention, copying, compositional structure, syntax or calibration shared with held-out text. Removing a negative component can reveal an existing positive component, which explains actual BPB improvement rather than only slower degradation. This is a hypothesis until inspected gradients and local configuration support the bridge.
2. **Curriculum/geometry rather than preservation alone.** DG can change expected direction across task contexts and avoid confident low-value task corners. A different task trajectory might reuse pretraining features more efficiently and specialize less. Predicted signature: advantage survives matched actual parameter movement, not just matched raw learning rate or update count. Stronger claims that task CE equals FineWeb CE are unsupported.
3. **Calibration change without broad new capability.** Write DG coefficient as `A/2 + A*tanh(A*ell/(2*eta))/2`. The added term is nonnegative; for small gate arguments it is approximately `A*A*ell/(4*eta)`. With action-independent zero-mean reward noise and roughly action-independent variance, expected correction is proportional to `E_pi[ell*grad log pi]`, the entropy gradient. Thus noisy evaluative feedback can induce an entropy-like drift under DG even when PG's expected score term is zero. If the pretrained LM is overconfident on FineWeb, softening can lower BPB without new task-general competence. The small-argument/action-independent assumptions need validation, and asymmetric reward variance can reverse useful behavior. Compare best temperature-calibrated guard loss and entropy before/after.
4. **Effective learning-rate/path length confound.** Gates shrink every raw sample term, but aggregate norm need not simply shrink because cancellation changes; Adam often neutralizes global gradient scaling. Therefore matched learning rate is insufficient and “DG is just half learning rate” is also unjustified. Compare actual update norms, parameter movement, clipping activity and reward-matched checkpoints.
5. **Off-policy/state mismatch suppression.** If our sampler carries latent state or differs from learner scoring, learner-relative surprise can identify mismatched failures. If latent carry changes generation behavior but not broad negative credit, it need not solve the failure mechanism. This is conditional on trainer behavior, which another investigator is auditing.

Suggested discrimination before more training: gradient alignment of positive/negative score contributions with a fixed held-out FineWeb loss, before and after gating; per-module interference; guard loss by token surprisal/frequency and natural language vs formatting; temperature-calibrated BPB vs native BPB; actual update norms and clipping; plots at matched validation token sets and starting checkpoints. These are proposals, not newly run evidence. Any model evaluation/gradient workload must enter mlq.

### Gaps

- The papers alone cannot rank transfer versus calibration for this repository.
- Guard dataset/latent-state behavior, frozen modules, KL/entropy terms, optimizer and LR confounds require local implementation/data audit.

## What caveats and implications should shape the conclusion?

### Takeaway
DG's selective plasticity can improve unrelated metrics only when accepted updates happen to help shared functionality or calibration. It is not preservation by construction, and the observed beneficial window need not last indefinitely.

### Cited Findings

- False delightful successes can be caused by reward variance, corrupted verifiers, or reward hacking. — [Distributed DG Conclusion](https://arxiv.org/html/2603.20521v2#S7)
- Supplemental companion (not one of the four indexed/requested papers): **Does This Gradient Spark Joy?**, Ian Osband, arXiv 2603.20526, introduces a hard Kondo gate for backward selection. Section 4.2 studies heteroskedastic gambling: rare suboptimal high-variance actions occasionally look like breakthroughs; delight favors lucky positive observations while suppressing the negative correction. No per-sample reward/probability statistic can resolve lucky draw versus genuine discovery. MNIST action-specific noise creates sharp DG collapse, while homogeneous reward noise degrades more smoothly. This companion was consulted for this specific failure-mode caveat; no fifth library paper was added. — [Supplemental full text](https://arxiv.org/html/2603.20526#S4.SS2)
- The shared-state counterexample explicitly exhibits loss of negative correction under parameter sharing even with exact deterministic advantages. — [Corner Appendix A](https://arxiv.org/html/2605.11908v1#A1)

### Inferences

- Slower forgetting is credible as a gate mechanism; BPB improvement is stronger evidence that some accepted updates/calibration changes help held-out likelihood, provided metrics are fixed and comparable.
- A favorable window could end when transferable correction saturates and task specialization continues, or when reward noise starts dominating selected surprises. Hence task reward, native guard BPB, calibrated BPB, drift and high-delight verifier reliability should be monitored together.
- This observation does not establish that unrelated RL can replace continued FineWeb CE pretraining. It could suggest a useful warm-start correction/regularization regime if matched causal controls confirm it.
- Do not reflexively add KL, clamps or stronger gating: distinguish actual interference, false delight and calibration before proposing fixes. No experiments, code changes or workloads were authorized by this paper-reading task.

### Gaps

- No experimental proof that the user's observed intervals are not subset noise, temperature effects, reset artifacts, different initial checkpoints, or configuration changes.
- The exact locally implemented gate may differ from papers in surprisal normalization, token/sequence aggregation, gate detachment, group advantages or interaction with entropy/KL. Local code evidence is essential.

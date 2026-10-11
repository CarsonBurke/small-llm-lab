# Delight Gate, general-language preservation, and the symmetric-noise mechanism

## What do the four papers establish, and where do their assumptions stop?

### Takeaway

The papers support a mechanism for filtering harmful surprising negative updates and preserving surprising positive updates. They do not establish that unrelated reasoning RL must lower FineWeb cross-entropy, and their cross-entropy oracle is the task's supervised oracle, not FineWeb's oracle.

### Cited Findings

- DG uses f(U, l) = U sigmoid(U l / T), where l = -log p. Its original bandit analysis separates perpendicular-noise suppression from cross-context reweighting toward the task's CE oracle; the latter theorem uses orthogonal context scores, zero baseline, and T > 1/2. The appendix's entropy derivation concerns entropy of the scalar gate, not policy entropy. — [Delightful Policy Gradient, sections 2–4 and appendix A](https://arxiv.org/html/2603.14608v1)
- The distributed paper proves rare negative terms receive gate at most p^(|U|/T), while rare positive terms approach full strength. Its finite-batch perpendicular second-moment guarantee is in a contaminated tabular classifier; it warns that verifier errors and reward hacking can produce false delight. — [Delightful Distributed Policy Gradient, sections 4 and 7](https://arxiv.org/html/2603.20521v2)
- The corner paper gives a local sector result with p_opt >= epsilon/K and exact advantages. Its first-exit bound has a 1/epsilon_initial prefactor and stops at parity or sector exit. A shared-parameter example gives a suboptimal DG fixed point around 0.116 while PG converges to best-in-class. — [Delightful Gradients Accelerate Corner Escape, sections 4 and appendix A](https://arxiv.org/html/2605.11908v1)
- Delightful Exploration modifies acting through posterior expected improvement times host-relative surprisal and a priced override. This is not the same operation as DG's update gate. Its Pandora equivalence assumes revealed-value inspection; worst-case greedy-host regret remains open. — [Delightful Exploration, sections 2 and 5](https://arxiv.org/html/2605.13287v1)

### Inferences

- Mechanism transfer is plausible because language and reasoning share embeddings, attention, and the rendering distribution. Distribution identity is unnecessary for an update on one domain to improve another: the necessary condition is favorable gradient alignment, with tolerable curvature.
- The original paper's task-CE alignment cannot be substituted for FineWeb-CE alignment. Hard reasoning contexts can have gradients that help, hurt, or leave general prose unchanged.
- Low-probability token does not mean semantically bad token. A failed rollout can contain good grammar, correct partial steps, names, and numbers. Assigning the whole sequence a negative advantage punishes these indiscriminately; DG attenuates that punishment more strongly for already unlikely tokens. This is a credible preservation mechanism, but requires empirical gradient attribution.
- No universal entropy bonus follows merely from the gate's variational entropy derivation. That entropy is Bernoulli gate entropy. A separate conditional-noise argument below yields policy smoothing under stronger assumptions.
- Red-team caveat on the corner paper: appendix D's monotonicity identity assumes logit update += alpha*w*U, whereas Algorithm 2 and equation 3 use the expected score-gradient update p*(w*U - E_p[w*U]). These are different geometries. Further, the assertion that p_opt <= O(epsilon² + sum p_bad^(1+delta)) defines a measure-zero set is not justified by the inequality: a thin region can have positive measure in every fixed neighborhood. These are direct reading observations, not a complete disproof audit. Do not use its global guarantees as evidence for neural Adam training.

### Gaps

- The paper agent confirmed the indexed quartet as policy gradient, distributed policy gradient, corner escape, and exploration. They additionally cite Does this gradient spark joy?, arXiv:2603.20526, which is outside that indexed quartet.
- Neither the papers nor this note prove FineWeb preservation under sequence-level noisy advantages, latent carry, shared neural parameters, and adaptive optimizers.

## What is the precise asymmetric-noise effect, and can it explain BPB improvement?

### Takeaway

DG contains an exact positive surprisal-dependent coefficient in addition to half of PG. Under action-independent symmetric advantage noise, this yields a conditional generalized-entropy gradient; with weak gating it reduces to a variance-scaled Shannon-entropy gradient. This is a plausible route from unrelated noisy RL to improved calibration, but it is conditional on the noise model and on initial overconfidence.

### Cited Findings

- The defining update is a gated score-gradient term and deliberately changes the expected direction, rather than being an unbiased PG estimator. — [DG definition](https://arxiv.org/html/2603.14608v1#S2)
- The distributed paper explicitly distinguishes true discoveries from high-surprisal false-positive advantage samples. — [Distributed DG limitations](https://arxiv.org/html/2603.20521v2#S7)

### Inferences

The following is our derivation from the defining gate, not a theorem claimed by the papers. Use T for temperature in the denominator; if code uses inverse temperature eta multiplying U*l, substitute 1/T = eta.

Exact scalar decomposition:

    f(U,l) = U/2 + (|U|/2) tanh(|U| l / (2T)).

For discrete tokens l >= 0, the second coefficient is nonnegative for either advantage sign. It rewards surprising sampled actions relative to the half-PG component. This statement is about decomposition relative to half-PG: DG does not exceed ordinary PG's positive per-sample magnitude, since its gate is at most 1.

Consider a fixed context and an advantage random variable whose conditional distribution is identical for every sampled action and symmetric around zero. Then:

    E[f(U,l) | action] = b(l)
    b(l) = E[(|U|/2) tanh(|U| l / (2T))].

b(l) is positive and increases with l. If U is equally likely +c or -c, b(l) = (c/2) tanh(c l/(2T)). The expected update is E_action[b(-log p) score]. Define a separable potential G(p) = sum_action F(p_action) with F'(x) = b(-log x). Then:

    expected DG update = gradient_theta G(p).
    F''(x) = -b'(-log x)/x <= 0.

For nondegenerate finite-temperature noise, F is strictly concave, so the unique full-support simplex maximizer is uniform. Thus this special-case direction is exactly a gradient of a concave generalized entropy. It need not be the gradient of Shannon entropy at finite gate strength. The stationary uniform distribution is conditional on the context, not a global uniform vocabulary histogram.

Weak-gate expansion:

    f(U,l) approximately U/2 + U² l/(4T).
    E[f | action] approximately Var(U) l/(4T).
    expected update approximately Var(U)/(4T) * gradient_theta H(p).

The last identity follows from E[score] = 0 and gradient H(p) = E[(-log p) score]. This approximation needs |U|*l/T small, finite suitable moments, on-policy sampling, identical action-independent noise, and detached advantages/gates. Our late-reward sequence advantage need not satisfy any of these conditions at individual tokens.

Limits and counterarguments:

- If all gates saturate in the hard-gate limit, b(l) approaches the same constant E|U|/2 for every supported action. E[constant*score] = 0: the independent-noise bias vanishes. In the infinite-temperature limit DG becomes half-PG, so it also vanishes. The noise-induced smoothing need not increase monotonically with gate sharpness; intermediate strength matters.
- With action-dependent noise variance, the weak-gate extra term is E[U²|action]*l/(4T), not a common entropy coefficient. It can favor particular tokens or reward hacks. With nonzero real advantages, DG adds this term to half the genuine PG field.
- A constant positive score coefficient is not self-rehearsal in expectation: on-policy score expectation is zero. Positive-only learning from genuinely successful samples is reward-conditioned imitation, but does not reconstruct FineWeb targets.
- More entropy does not automatically lower NLL. If p already matches FineWeb's conditional target q, moving toward uniform worsens KL(q||p). It helps BPB only if it corrects overconfidence or another distribution mismatch in the right direction.
- Even with conditional smoothing on reasoning prompts, shared parameters can sharpen some FineWeb contexts. The sign of the FineWeb loss derivative must be measured.

### Gaps

- Need saved gate/advantage/surprisal distributions to see whether the weak-gate or saturation regime applies, and whether noise variance depends strongly on token identity, source, or action.
- Need checkpoint logits or queued evaluation to separate temperature recalibration from improved conditional features. No model workloads were executed for this note.

## Which explanations rank highest, and what minimally distinguishes them?

### Takeaway

Best provisional explanation is that DG changes which language-model updates survive noisy sequence credit assignment, reducing destructive drift and sometimes improving calibration. Genuine language transfer is possible but currently weaker than calibration/preservation and comparison confounds as explanations. Latent carry is not the sole bottleneck if the guard evaluates the plain backbone and renderer.

### Cited Findings

- DG's tail filtering is sign-aware, and a probability-only filter cannot preserve rare successes while suppressing rare failures. — [Distributed DG proposition 2](https://arxiv.org/html/2603.20521v2#S4)
- On the original task, DG changes both sampling noise and population direction. — [DG tabular analysis](https://arxiv.org/html/2603.14608v1#S4)

### Inferences

The coordinator relayed these local observations: guard uses the plain backbone+renderer and bypasses latent carry; DG v5 improves from 1.6524 to 1.5945 around update 2400; the no-source-gate run reaches 1.5999 around 1500 but rebounds to 1.6512 around 8250; older PG comparisons confound rollout topology and initial freeze; a current first-step comparison has approximately half DG trunk gradient norm but essentially equal rollout KL (0.000146 versus 0.000149). These are inputs from the evidence agent, not independently verified in this note.

1. **Reduced destructive negative learning plus conditional calibration correction.** Highest mechanistic plausibility. Failure advantages applied across complete reasoning trajectories penalize otherwise useful language tokens and encourage common-policy modes through softmax coupling. DG suppresses rare negative terms, reduces familiar positive reinforcement toward half-strength, and gives surviving surprising successes more relative influence. The noise decomposition can supply a smoothing component. Prediction: improvements occur disproportionately among baseline overconfident contexts, high-loss tail tokens, and shared prose scaffolding; negative/high-surprisal gradient components are most harmful to guard. Counterargument: genuinely wrong rare actions also need corrections, and softmax mass from their punishment can help other correct tokens. Diagnostic: temperature calibration plus sign/surprisal gradient attribution at an improvement checkpoint.

2. **Lower effective drift and better optimizer direction.** High plausibility for slower degradation, inadequate by itself for sustained improvement. Smaller raw gradients do not imply smaller Adam updates: rescaling both moments often cancels. Near-equal first-step KL despite half raw trunk norm argues against a simplistic halved-learning-rate explanation, but rollout KL is on RL contexts rather than FineWeb. Prediction: PG degradation may collapse toward DG when plotted versus measured guard-policy KL or layerwise parameter movement. Diagnostic: compare actual update norm and FineWeb KL, not just gradient norm or optimizer steps. Retain a matched-scale PG control only if this remains unresolved.

3. **Task-conditioned learning improves shared language features.** Moderate plausibility; potentially most useful implication if confirmed. Reasoning rollouts still use language syntax, compositional dependencies, numbers, and copying. Successful rare trajectories can teach shared representations, even though source corpora differ. Prediction: lower BPB persists after per-checkpoint temperature fitting, and is concentrated on meaningful contexts rather than exclusively confidence bins. Diagnostic: temperature-adjusted guard loss on a separate fit split, per-document and token-type loss changes, and first-order FineWeb loss projection of the actual update. Simple CE-oracle theorem citations are insufficient evidence.

4. **Source gating/curriculum selects a more transferable update mixture.** Moderate and highly plausible modifier. Early no-source-gate improvement followed by rebound suggests DG alone is not permanent protection. Later source composition, advantage noise, or over-specialization could change the alignment. Prediction: guard derivative flips with source-level reward, gate-open fraction, or entropy/advantage variance. Diagnostic: source-tagged gradient contribution and gate statistics across the improvement and rebound windows, compared at matched data/update budgets.

5. **Evaluation artifact or incomparable history.** Must resolve before interpreting causal superiority. Changing guard shards, byte counts, context length, token normalization, reconstruction path, checkpoint choice, or baseline selection can mimic improvement. Older PG confounds prevent strong general statements about DG versus PG. Prediction: deterministic replay on a fixed held-out guard slice reproduces real improvement, with exact metadata equality. Diagnostic: inspect existing manifests and evaluation code before allocating runs; replay via mlq only if needed. A fixed shard does not create improvement by itself, but it can support over-selection if repeatedly used to choose training variants.

Minimum high-information diagnostic order:

1. Static audit of comparability: common checkpoint, guard IDs and byte counts, rendering path, optimizer groups, latent/topology/freeze settings, source mixture, gate detach and sign, reward normalization, and actual update schedule.
2. Fit scalar logit temperature on a distinct FineWeb calibration subset and evaluate raw plus temperature-adjusted BPB on untouched guard tokens for initial, best, and later checkpoints. If improvements disappear after equal recalibration, calibration explains much of the effect. If gains persist, conditional-distribution changes remain. This is explanatory analysis only; do not replace challenge scoring with tuned guard evaluation.
3. Inspect actual layerwise update and guard-policy KL. Nearly identical RL rollout KL is not proof of equal FineWeb drift.
4. At the same checkpoint and identical saved RL batch, decompose PG and DG into positive/negative and surprisal bins. Compute FineWeb gradient projection using actual preconditioned optimizer steps or direct finite-step loss, preserving identical optimizer state. First-order delta L_guard = grad L_guard dot delta_theta; curvature can make finite updates worse despite favorable projection. Pair this with entropy and variance terms from the derivation.
5. Only then choose a 1000-update matched ablation: comparable PG, DG, or one discriminating gate/entropy variant. Do not introduce many simultaneous changes or interpret reduced-budget smoke diagnostics as training evidence.

Implications if hypothesis 1 dominates: preserve the sign-aware gate, improve local credit assignment, measure calibration and false delight, and avoid treating latent carry as a substitute for protection of shared language parameters. If hypothesis 2 dominates: compare at matched real update/drift rather than step count. If hypothesis 3 dominates: target source and token classes with demonstrated positive transfer, while keeping the guard held out from training and tuning decisions.

### Gaps

- No matched causal comparison or new model evaluations were run. Ranking is provisional pending coordinator evidence and diagnostics.
- Guard improvement in one run cannot establish universal DG generalization or challenge-level BPB benefit. Rebound is affirmative evidence against permanent implicit rehearsal.

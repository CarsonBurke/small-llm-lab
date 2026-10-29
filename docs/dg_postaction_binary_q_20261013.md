# Post-action binary Q and conditional DG: findings and implementation contract

Date: 2026-10-13. Status: analytical candidate implemented as an opt-in trainer mode; no evidence of a globally optimal training configuration. No training run is authorized by this document.

## Constraints

- No V, learned state-value baseline, policy-average Q baseline, grouped reward mean, or substitute serving that role.
- Reduce avoidable continuation noise while keeping the critic informed about the chosen action.
- Preserve the existing binary terminal reward, rather than quietly introducing a signed utility mapping.
- Keep mathematical identities, measured findings, proposed changes, and unknowns separate.

## Measured collapse findings

The existing artifacts establish the following limited findings about job13044:

| Finding | Evidence | Limit |
| --- | --- | --- |
| Actor lambda zero and critic Monte Carlo targets were implemented as intended. | [Team forensics](../postraining/runs/dg_qmean_lam0_dirlr1e6_study_20261013/team_forensics.json). | This does not validate every other implementation path. |
| Format rejection increased before relaxed answer correctness clearly deteriorated. | Accepted/relaxed-correct fractions: 0.0542/0.0585 at updates201–220, 0.0497/0.0582 at221–240, 0.0271/0.0409 at241–260. Same artifact. | Counters do not separate all rejection causes; the thinking floor alone is not established as the cause. |
| Memory Adam update RMS at250 was approximately twice the healthy control: 3.006e-5 versus1.518e-5. | Same artifact, `memory_update_comparison`. | Update magnitude is not a measure of whether a direction helps reward. |
| Memory-only checkpoint substitutions caused substantially larger functional movement on two identical tested histories. | [Native audit summary](../postraining/runs/dg_qmean_collapse_audit_20261013/summary.json): exact mean vocabulary KL0.47947/0.27959 versus healthy0.03212/0.04771. | Two histories and78 matched positions do not identify a population threshold or establish collapse causality. |
| A known zero-return early-close candidate had positive learned centered Q. | [Checkpoint250 result](../postraining/runs/dg_qmean_collapse_audit_20261013/checkpoint250/result.json), captured row0/action slot39: think count22, probability0.0084759, Q0.0227825, policy meanQ0.0222801, contrast+0.00050243. | Establishes this critic mispreference, not its prevalence or gradient contribution. |
| Reduced-precision centering violates the deterministic-policy identity on controlled inputs. | Team forensics, precision audit job13053: false advantage RMS1.59339e-6 versus0 in highest precision. | Actual checkpoint embeddings with synthetic policies/features; not an observed compiled training-gradient effect. Common errors can cancel in expected ordinary PG. |

The audits do not establish why Q-derived gradients caused different memory movement, whether decode/replay discrepancies caused harm, or whether freezing memory would prevent collapse. Tiny chosen-token centered advantages at nearly deterministic actions are not evidence of missing credit.

## Reference scope

The verified local DG implementation is [`../deepmind-egg/egg/losses/dg.py`](../../deepmind-egg/egg/losses/dg.py), revision `d005eac307a0cb6ccb9c63ad03aee39a3e3c30d4`.

It computes a reward baseline, advantage times surprisal, a sigmoid gate, stops the gate gradient, and averages the gated score loss over valid answer tokens. Its `use_grouped_baseline=False` branch still uses a global reward mean. Therefore the no-baseline estimator proposed here is **not an unchanged reproduction of that reference loss**. It retains the DG gating kernel under the user's different baseline constraint. Eta1 is a tested reference setting, not proof that every associated critic, optimizer, or utility configuration is optimal here.

## Conditional estimator identity

Assume terminal reward R is binary, with no discounting. Let H contain the actor-visible prefix and the selected action a, but no sampled future. Define:

```text
Q(H) = P(R=1 | H)
s = -log pi(a | prefix)
F(R,s) = R * sigmoid(R*s/eta)
```

For eta>0:

```text
E[F(R,s) | H] = Q(H) * sigmoid(s/eta)
```

At eta1, since exp(-s)=pi(a):

```text
w(H) = Q(H) / (1+pi(a))
```

This is the conditional expectation of the **entire** binary-outcome DG coefficient. It is not obtained by substituting expected reward inside the nonlinear sigmoid:

```text
Q * sigmoid(Q*s/eta)             # different estimator
Q * sigmoid(s/eta)               # conditional binary DG estimator
```

The actor score is measurable from H. With exact conditional Q, multiplying the conditioned coefficient by that score preserves the expected sampled binary-DG gradient and removes its conditional future-outcome variance. The conditional coefficient variance removed is:

```text
Q*(1-Q) * sigmoid(s/eta)^2
```

This is a Rao–Blackwell variance guarantee for that specified estimator. It is not a guarantee for learned Q, a guarantee of reduced parameter-update variance under Adam, or a theorem about benchmark improvement. It does not eliminate sampled-action variance. Critic training labels still contain Bernoulli continuation noise.

## Local objective and limits

At a fixed prefix, with Q frozen and independent policy logits, eta1 gives the expected score gradient of:

```text
J(pi; Q) = sum_a Q_a * log(1+pi_a)
```

This objective is concave on the probability simplex. Its KKT solution is:

```text
pi_a = max(Q_a/nu - 1, 0)
nu = sum_active Q / (number_active + 1)
```

For a zero-Q action, the expected derivative with respect to its own logit is:

```text
pi_a * (0 - sum_b pi_b*w_b) <= 0
```

If all Q are zero, all coefficients are zero. If positive Q are equal, the local optimum is uniform. Actions with Q below half the largest Q cannot enter the optimum's support.

These statements concern fixed-prefix expected logit gradients, not a guarantee that a finite shared-neural-parameter update decreases every invalid token probability. The critic can be wrong; other prefixes share parameters; optimizers transform gradients; and changing policy also changes visited prefixes and continuation returns.

The local optimum is not the greedy maximum-return policy. For Q=(0.09,0.10), it assigns pi=(0.42105,0.57895), with expected return0.095789. A policy assigning0.99 to the best action has return0.0999. This is reward-weighted diversity and can sacrifice immediate return. The local objective must not be presented as a proven global sequential training objective.

The original binary utility also limits what gate activity can mean: success gates lie between0.5 and1; failure weights are zero. A broad gate distribution covering both sides of0.5 is incompatible with these choices. Inflating Q to obtain more saturated gates does not test this estimator.

## Rejected alternatives and retained tradeoffs

### Signed utility mapping

For U=2R-1, conditioning the entire DG coefficient gives:

```text
E[U*sigmoid(U*s/eta) | H] = Q - sigmoid(-s/eta)
```

At eta1 this is `Q-pi/(1+pi)`. Its Q-dependent term is linear, so a common additive Q error cancels in the expected ordinary score gradient. The remaining term is unconditional exploration. With pi=(0.9,0.1) and Q=(0.2,0), its expected bad-action logit gradient is+0.016450; the binary estimator gives-0.009474. Signed utility changes the reward mapping and can favor known invalid actions through exploration. It is not the proposed implementation.

### Applying DG to expected signed utility

At uniform probability1/50304, naive coefficients for Q0.02 and0.10 are approximately-0.0000294 and-0.0001386. The worse action gets the larger coefficient. Conditioning the whole coefficient avoids this reversal. This counterexample concerns a candidate signed mapping, not a demonstrated mechanism in the historical centered-Q run.

### Common Q errors in the binary estimator

Binary conditioning does not remove all effects of common critic error. Adding the same error c to every Q changes coefficients by `c/(1+pi)`, which creates probability-dependent exploration pressure. In the scalar pi=(0.9,0.1) example, adding0.03 changes the rare-action logit gradient by+0.00103349. Calibration of absolute success probabilities remains necessary. No hidden policy-average subtraction is proposed to repair this.

## Opt-in implementation contract

The opt-in implementation has these requirements. This list describes the candidate's scope, not validation of a completed training experiment:

- An independent pretrained critic trunk consumes the normal prompt and generated prefix including the selected token. A scalar logit head predicts Bernoulli success probability Q through sigmoid.
- Critic parameters, embeddings, and trunk features do not borrow live actor writer weights or actor beliefs in this initial implementation. Gradients remain separate.
- The selected token at actor action slot t must be included in the critic representation used for Q_t. Prompt and padding slots are excluded from losses and actor credit.
- Critic targets are undiscounted terminal binary outcomes, with Monte Carlo lambda1 and stable binary cross-entropy-with-logits. Actor lambda0 is required; recursive corrections are excluded.
- The detached actor coefficient is `Q*sigmoid(s/eta)`. It replaces the entire previous gated-credit product; it must not be gated a second time. No policy-average Q computation belongs in this mode.
- Irreversible invalid-prefix and terminal-known-outcome overrides require an exact contract oracle consistent with the existing verifier. A zero override must mean every permitted continuation has return0, not merely that the sampled continuation failed. A correct terminal outcome may be known exactly; nonterminal unknown outcomes remain predictions.
- The first run explicitly used success prior0.5 (zero logits). The default is now0.02 at user direction after observed severe initial overprediction; neither is a proven optimal prior. Calibrating a critic prior from observed rewards would be calibration, not an actor value baseline; neither initialization has been validated for this candidate.
- Initialization, learning rates, critic capacity, and warmup adequacy are unresolved. Previous `1e-6` direction-readout evidence does not transfer directly to a new scalar Bernoulli head.

Full untruncated gold-task metadata, privileged answer information, a joint correctness/format/termination outcome model, ensembles, and counterfactual continuation supervision remain proposals. They must not be described as already implemented or tested. The initial critic receives normal prompt context only. A larger future context is not itself an actor baseline, but availability and target-policy conditioning would need validation.

## Validation and unresolved decisions

[Scalar simulation](../scripts/sim_critic_design_math.py) and [results](../postraining/runs/critic_design_math_20261013/results.json) verify the estimator identities, explicit counterexamples, and300 random finite-action KKT cases. Maximum observed simplex error is8.88e-16 and active stationarity error1.11e-16. These are scalar mathematical checks, not model-performance evidence.

An independent review of [`postraining/binary_q.py`](../postraining/binary_q.py) found no blocker in its detached whole-coefficient calculation or exact-prefix oracle. Its 18 focused tests passed. A separate tensor-only check preserved every live prefix of 2,744 valid grammar completions and assigned exact terminal success at EOS; it also compared the expected actor gradient against the local objective gradient, with maximum absolute difference 1.86e-9. [Independent review artifact](../postraining/runs/critic_design_math_20261013/independent_review.json). These checks do not validate trainer integration, a compiled GPU path, or the verifier contract for every dataset mode.

Before a training run, implementation checks should establish post-action indexing, detach boundaries, binary-target enforcement, no hidden baseline, exactly one DG multiplier, oracle soundness, and exclusive optimizer parameter ownership. A critic classifier's calibration on available labels and its errors on provably invalid branches are more relevant than aggregate EV alone, but cannot completely validate preferences for unsampled counterfactual actions.

Still unknown:

- Whether Q on ordinary informative actions is accurate enough for critic-only actor credit.
- Whether independent post-action conditioning improves those predictions within the available budget.
- Whether objective-driven diversity helps this benchmark or amplifies wrong predictions.
- Which critic initialization and optimizer rates produce a suitable function-space update.
- Whether the earlier collapse requires harmful memory updates, bad critic preferences, or their interaction.
- Whether privileged task metadata or multi-outcome conditioning is worth its complexity and cost.

There is no claim that these unknowns have been solved by algebra, scalar simulations, or the opt-in code path. No model or training run was launched for this findings document.

## Code and configuration

Implemented in the main checkout, preserving the historical modes:

| Responsibility | Code |
| --- | --- |
| Whole conditioned DG coefficient and prefix oracle | [binary_q.py](../postraining/binary_q.py) |
| Independent successor-conditioned critic and scalar probability head | [value_model.py](../postraining/value_model.py) |
| Monte Carlo BCE, actor integration, optimizer ownership and metrics | [train_latent_vapo.py](../postraining/train_latent_vapo.py) |
| Mutually exclusive modes and binary-objective constraints | [config.py](../postraining/vapo/config.py) |
| Distinct critic/actor contracts and strict resume checks | [schemas.py](../postraining/vapo/schemas.py) |

The mode-specific argument set below is a configuration example, not a command to launch training. Hidden size and prior are exposed choices, not measured optima. The ordinary checkpoint, dataset, budget, and learning-rate arguments are still required by the trainer.

```text
--critic-post-action-q-hidden 1024
--critic-post-action-q-prior 0.02
--critic-init actor
--reasoning-mode memory
--delightful-policy-gradient
--think-tokens
--answer-fence
--nearby-reward-max 0
--python-reward-mode binary
--actor-gae-lambda 0
--q-critic-lambda 1
--no-source-success-actor-gate
--no-zero-reward-actor-freeze
```

Use `--fresh-critic` when initializing the actor from an existing RL checkpoint. The new critic is initialized from the actor's pretrained trunk, copied without parameter sharing; existing V/bilinear-Q critic checkpoints cannot resume as this critic. The rollout and minibatch prompt counts must match, giving one fresh actor update per pool. Entropy auxiliaries and actor KL backtracking are excluded from this particular objective contract.

Source-success masking and freezing an all-zero-reward pool are rejected because they depend on sampled future outcomes. They would change the conditioned estimator even if the Q prediction and gate were implemented correctly. This restriction is specific to this estimator, not a general claim that those controls are always harmful.

The prefix oracle accepts all actual scorer stop IDs, including a distinct BOS when the tokenizer has one. It counts only emitted response tokens, preserves ordinary gaps allowed by the existing strict format verifier, and propagates irreversible violations. It uses the realized verifier label only at a terminating token. Reaching the known response budget without a terminator gives exact zero.

Diagnostics distinguish raw predicted probabilities from actor probabilities after exact overrides. BCE and Brier score describe the raw critic fit; R² includes prediction bias, while explained variance subtracts residual mean. The actor reports its success-probability mean and the whole DG-coefficient mean. Legacy signed TD/GAE calibration diagnostics are skipped in this mode. Metrics retain the existing token weighting, so repeated labels within one trajectory are correlated; they are not independent calibration samples.

No performance claim is attached to eliminating the vocabulary-wide policy-average Q readout or borrowed memory scan. The successor-conditioned critic uses one causal replay pass; measured wall time and compiled GPU behavior remain unvalidated for this mode.

## Completed code checks

- 58 focused tensor/stub tests passed across [estimator/oracle tests](../postraining/tests/test_binary_q.py), [critic tests](../postraining/tests/test_post_action_q_critic.py), and [trainer integration tests](../postraining/tests/test_post_action_q_integration.py). No actual policy or critic model workload was executed.
- The actual `update_minibatch` warmup and fused actor branches were exercised with stored scalar logits and replay stubs. Changing a sampled terminal label between zero and one left all preceding live-prefix actor gradients identical; only the terminating action used the exact label. Critic BCE gradients changed with the label as intended.
- Independent review checked every live prefix of 2,744 valid grammar completions and the expected local-objective gradient. [Review artifact](../postraining/runs/critic_design_math_20261013/independent_review.json).
- Python compilation and `git diff --check` passed. Configuration rejects incompatible heads, shaped rewards, sampled-outcome actor masks and old critic contracts.

These checks establish the tested algebra, indexing, masking and integration routes. They do not establish GPU compilation correctness, runtime performance, predictive calibration, collapse prevention, or a preferred learning rate. No new training job was queued or started.

## Prior correction (2026-10-13)

The initial prior0.5 gave mean Q0.5 against token-weighted target0.01591 at warmup1. At warmup46, mean Q was still0.31278 against target0.02179. This establishes substantial marginal overprediction and slow correction under the selected optimizer; it does not establish the best prior or learning rate.

Fresh heads now default to prior0.02, with zero output weights and bias `log(0.02/0.98) = -3.89182`. Existing checkpoints retain learned predictions when resumed; a CLI prior change is not a recalibration of a saved critic. The held follow-up therefore starts a fresh critic from the original pretrained actor trunk, rather than claiming to change the warmed critic through its resume arguments.

## Native actor readout failure and correction (2026-10-14)

Continuation13408 aborted with CUDA assertion `log probabilities must be nonpositive` before any recorded actor optimizer update. The assertion also rejects NaNs; native offending values were not captured. The later CUDA RNG-restore traceback is an asynchronous report, not evidence that RNG restoration caused the failure.

The original readout separately softcapped gathered targets and the full vocabulary row. Compiled GPU tensor audits13455 and13460 reproduced705 finite positive log probabilities, up to9.89e-7 across tested caps15/30. At the production cap15, the largest positive value was5.40e-7. Inputs were controlled bf16 peaked logits with vocabulary50304, rows1/8/128/512; these were not actual replay activations.

The correction derives target score and normalization from the same max-shifted capped row. Backward reconstructs that normalization, and entropy uses `-sum(p log p)`. The corrected audit produced zero positive scores, finite compiled gradients, and maximum selected-score error1.20e-7 against a float64 reference. [Forward/backward audit](../postraining/runs/dg_readout_assert_audit_20261014/results_with_backward.json).

Invalid active scores and probabilities are now checked by the existing host-side guard before any optimizer step, with positive/nonfinite counts and maximum positive score reported. Invalid values are neither clipped nor replaced. This avoids poisoning the CUDA context before diagnostic values can be read.101 CPU tensor/stub tests pass, including rejection of positive,NaN and infinite scores before optimizer steps. These tests do not establish complete production actor-update success.

## Verification after session upgrade (2026-10-14)

Job13471 was observed running with 60 completed production actor updates. The frozen source hashes matched its plan; the rolling checkpoint and TensorBoard scalars were present. This supersedes the earlier limitation that no complete actor optimizer update had been verified. [Runtime check](../postraining/runs/dg_postaction_binary_rb_prior002_study_20261013/runtime_verification_after_upgrade_20261014.json).

The latest attempt emits repeated CUDA allocator mapping warnings. Updates continued after those warnings; this does not establish why each allocation failed or eliminate the possibility of a future resource failure. No live configuration or source was changed. These execution checks do not establish benchmark improvement, prevention of the historical collapse, or an optimal critic configuration.

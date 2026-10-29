# Memory write critic: DDPG-style writes (2026-10-20)

Stage 3 of `docs/mac_online_critic_plan_20261019.md`, "write noise with a
write-time critic", read as DDPG/TD3: the writer stays deterministic,
rollout writes carry exploration noise, and the writer is trained by the
critic's gradient with respect to its write. Chosen over a stochastic
Gaussian write policy with its own score-function term (a
heads x head_dim continuous action per token, the v29 latent regime).

## What the writer learns from today

In `--reasoning-mode memory` the writer's only signal is pathwise through
the token loss: replay's scan carries d log pi(a_j) / d w_t for every read
j >= t, weighted by the estimator's coefficient (G - c for the sampled
baseline). That is an unbiased gradient of the token objective with
respect to the writes, with REINFORCE's variance. In the memory drill the
read reached 1.8% of the input RMS by step 247 (NOTES 2026-10-20).

## The action and its critic

- **Action.** At slot t the write w_t = (k_t, v_t, log alpha_t, beta_t)
  per head, a function of the belief h_{t-1} that produced token t.
- **State before the action.** Everything up to slot t except w_t: the
  tokens, and the memory S_{t-1}.
- **Write critic.** Q_w(s_t^-, w_t), regressed on the Monte Carlo return G
  at slot t, the same target as the token critic at lambda 1.

The critic must depend on w_t only at slot t. The critic trunk is causal
attention over its inputs, so if w_t entered the trunk's input at slot t,
every later value V(s_j) would depend on it with the tokens between t and
j held fixed. Those tokens were sampled under w_t, so that dependence is
not w_t's effect on the return; it is a bias. So:

- **Trunk input at slot t reads S_{t-1}, not S_t.** Each write reaches the
  trunk one slot later than today. All of these reads are constants for
  the actor.
- **A write head reads the latest write at its own slot only.** With
  S_t = alpha S_{t-1} + beta k (v - alpha S_{t-1}^T k)^T, the read of S_t
  by a query q is

      S_t^T q = alpha P + beta (k . q) (v - alpha K),
      P = S_{t-1}^T q,  K = S_{t-1}^T k.

  P and K are reads of the previous state, which a scan gives at every slot
  in parallel by shifting its queries one slot left (the scan writes then
  reads). With P and K detached, this read is exactly today's value in the
  forward pass and depends on w_t alone in the backward pass. The head is
  g(z_t, r_t): z_t is the trunk feature at slot t and r_t this isolated
  read through the critic's reader. Its gradient dQ_w / dw_t is the DPG
  term, and nothing crosses slots.

## Exploration noise

- Rollout writes are mu(h_{t-1}) + epsilon_t, with Gaussian noise on v and
  on the pre-normalization key, the decay logit and the write logit. Each
  term's noise scale is relative to that term's RMS.
- Noise is counter-keyed like the token draw (slot and row), so replay
  regenerates it exactly and nothing new is stored. The token log-probs
  stay exact because the actor's replay consumes the same noisy writes.
- The critic fits Q_w at the noisy writes. That gives it data around mu,
  where the actor reads its slope. Without noise its slope is pure
  extrapolation, the off-support failure that collapsed all-action MAC
  (NOTES 2026-10-19).
- The actor's DPG term is read at mu(h_{t-1}), with S_{t-1} from the
  behavior trajectory.

## Writer update: one signal, not two

The token path and the DPG term estimate the same quantity: how a write
changes the expected return, including through the tokens it makes more
likely. Summing them would count that effect twice and mix an unbiased
term with a critic-biased one. The arm therefore detaches the write terms
from the token loss. The reader keeps its token-path gradient, because it
needs no gradient through the writes. The writer then learns only from
-dQ_w/dw. The existing memory drill is the control, with the token path
only.

What this can and cannot buy: with Monte Carlo targets, Q_w's slope is a
regression of G on the noise. By the compatible-features argument a linear
Q_w in epsilon carries the same information as a score-function estimate
on the writes. Any gain comes from Q_w pooling across states and pools,
and from its nonlinearity. That is the hypothesis.

## Gates before an actor run

1. **CPU.** Exactness of the isolated read: forward equal to the full read,
   and a backward that reaches w_t only through slot t. A brute-force check
   of the shifted-query P and K against `reference_delta_scan`. A
   counter-keyed noise replay that matches decode.
2. **Frozen-policy learnability (GPU, 45 min or less).** Collect noisy-write
   pools from the step-247 memory drill policy and fit Q_w on them. On
   held-out pools, test whether the predicted noise effect,
   epsilon . dQ_w/dw, explains the return residual G - Q_w(mu), against
   shuffled-noise controls. If it does not, the actor arm has nothing to
   follow. Then choose the noise scale from the smallest one that passes.
3. **Drill.** The memory drill's config, with noise, write critic and DPG
   writer, against the memory drill and 14165, step for step.

## Telemetry

- Q_w's explained variance on the fresh pool before its fit.
- Q_w's slope norm and the noise-effect correlation per pool.
- The writer's DPG gradient norm against the norm the token path would
  have given (computed, not applied).
- Read share of the input RMS, write beta, decay timescales.

## Open questions

- Should the DPG term use TD3's delayed and smoothed actor updates?
  Smoothing (target noise) only matters for bootstrapped targets, and these
  are Monte Carlo. Delay is cheap to add if Q_w's slope is unstable.
- Which states get the write head: every generated slot, or think slots
  only?

## Red team (2026-10-20): revisions before any build

Two findings block the design as written.

1. **The slope is probably not identifiable.**
   - Let rho^2 be the share of the return's variance that the write noise
     explains. Slots in a trajectory share one G, so the sample size is
     trajectories (about 1,000 per pool), not slots.
   - Fitting even a state-independent slope (520 or more parameters) to
     half fidelity takes about 51 pools at rho^2 = 1e-2, 508 at 1e-3 and
     5,080 at 1e-4.
   - The whole learned read moves accuracy by at most 1-2 points (the
     memory drill matched 14165). So rho^2 of 1e-3 or less is plausible,
     which is borderline to hopeless within a 45-minute gate (about 600
     pools).
   - The token path's rival estimate uses the exact derivative of the
     log-probs with respect to the writes, while the noise estimate must
     rediscover it at variance growing as 1/sigma^2.
2. **Confounding.**
   - The write is mu(h_{t-1}) + eps, and mu varies across states far more
     than eps does. A regression of G on w therefore learns which writes
     co-occur with success, and the actor would chase that association.
   - Any write critic must separate the two: Q_w = b(s, h_{t-1}) +
     A(s, h_{t-1}, eps) with A(., 0) = 0, so the slope is learned from eps
     alone.

Corrections to the design above:

- **The token path is truncated, not unbiased.** Replay cuts
  write -> trunk -> later belief -> later write (`hiddens.detach()`).
  Monte Carlo returns see that loop; that is the real case for a
  return-based write signal.
  - *Corrected 2026-10-22:* the cut is the memory's credit contract, the
    same one a sampled token has (its effect through later beliefs is never
    differentiated, and the token policy learns). Closing the loop is
    truncated BPTT (`memory_belief_adjoint`), excluded for cost and as
    unnecessary; its one-hop drill was a null. The case for a return-based
    write signal is to credit a write as an action, as a token is credited,
    not to see the loop. See NOTES 2026-10-22, "Per-step write credit".
- **Leave the critic trunk on S_t.** Later values depending on w_t is
  correct state conditioning. The reads are constants for the actor, so
  differentiating through them is not the issue. Moving the trunk to
  S_{t-1} would cost the token critic information the policy used. Instead
  read Q_w from z_{t-1} (causal, has not seen w_t), e(x_t) and the write's
  features, as the existing next-slot heads do.
- **Feed the head the write itself**: (k, v, gate logits) or eps, plus
  K = S_{t-1}^T k as a forward-only feature. A single critic-query read
  hides what a write is worth to future queries.
- **Do not detach the token path.** Combine the two signals instead. Under
  detach, prompt writes (including the first generated slot's, made at
  prefill) get no gradient at all.
- **Noise.** Use absolute noise in logit or log-timescale space, not
  noise relative to RMS. Key it on (trajectory seed, unpadded slot) with
  its own domain tag. The prompt state needs to be per row when noisy. Turn
  noise off at evaluation.

Revised order:

1. Measure rho^2 directly, with no critic. Run common-random-number
   antithetic rollouts at the final memory policy: the same prompts and
   token seeds, with the writer weights (and separately the reader's
   output) perturbed by +delta and -delta. Policy entropy is 0.025 nats
   per token, so coupled trajectories differ only where a perturbation
   flips a token, and G(+) - G(-) is a low-variance directional derivative
   of the return. This needs no new decode noise.
2. Measure the token path's writer gradient SNR across pools
   (`scripts/estimator_gradient_alignment.py`, restricted to writer
   parameters).
3. Build a write critic only if step 1 shows a usable signal. If it does,
   antithetic evolution strategies on the writer are the simpler first
   arm.

# Per-step TPO for latent-thought VAPO

Handoff document. Status: **idea, not started**. Read-only analysis done 2026-07-21/22;
no code written. Companion repo: `../tpo` (sibling checkout of
https://github.com/JeanKaddour/tpo, "Target Policy Optimization", arXiv:2604.06159).

## One-paragraph summary

Replace the factorized PPO-clip surrogate in `postraining/train_latent_vapo.py` with
per-step TPO-style target matching, keeping the HL-Gauss critic + length-adaptive GAE
exactly as they are. Sequence-level TPO (the paper's critic-free headline form) is
**inapplicable** to our action space — the group softmax over full-sequence
log-likelihoods is destroyed by 512-D Gaussian thought densities (analysis below). The
viable form is per-step anchored cross-entropy with GAE advantages as scores, which is
the MPO/V-MPO corner of the same family (paper Table 4). Carson has direct evidence this
per-step form beats plain PPO in continuous-control environments. In this form nothing
fundamental about post-training changes: it swaps the update rule, not the credit
machinery.

## Background: what TPO is

For K candidates y_i ~ π_old(·|x), form the group policy p_i = softmax_i(log π(y_i|x)),
tilt it by within-group z-scored scores u_i to get the frozen target
q_i ∝ p_i^old · exp(u_i/η), and minimize CE(q, p^θ). Gradient on the group logits is
p^θ − q: bounded in [−1, 1], self-extinguishing at the target, no importance ratios, no
clipping, no critic (in the published form; scores are raw task rewards).
Reference implementation: `../tpo/src/tpo/algorithms.py:287-322` (`tpo_skill`,
`tpo_target`, `anchored_tpo_cross_entropy_loss`); sequence-level and token-level
experiment wiring in `../tpo/src/tpo/experiments/_trial.py:329-445`.
Paper (agent-friendly markdown): `../tpo/paper_for_agents/tpo_for_agents.md`.

Its empirical wins concentrate in exactly our regime: terminal near-binary reward,
K-sample prompt groups, multi-step reuse of a frozen rollout pool (paper §3.6–3.8,
Fig. 14–15).

## Why sequence-level TPO fails here (do not re-litigate; arithmetic verified twice)

Our trajectory log-likelihood is gate log-probs + token log-probs + a **512-D diagonal
Gaussian log-density per THINK step** (`postraining/latent_thought.py:288-305`). With
init log σ = −3:

- Per-dim log-density = 2.081 − ε²/2 nats ⇒ per THINK step: mean ≈ **+810 nats**,
  SD ≈ **16 nats** (= √(512·0.5), from the ε² term of the reparameterization — this
  noise floor is *invariant to the learned σ*).
- The reward tilt is z-scored: bounded to ±√(K−1) ≈ **±4 nats** at K=16.
- So candidates differing by one think step differ by ~810 nats in log p; even
  equal-think-count candidates differ by tens of nats of latent-sampling luck. The group
  softmax p_old collapses to ~one-hot on the luckiest/most-think-heavy trajectory,
  q ≈ p_old, gradient ≈ 0. The bounded gate/token factors are yoked into the same
  softmax and get starved too.
- No η fixes it: large η flattens the tilt further; small η is the no-anchor variant the
  paper's own ablation (Fig. 8) shows is consistently harmful.
- The paper never tests this regime: all transformer experiments are fixed-length
  (zero heterogeneity); the verl LLM runs use raw log-prob sums, no length
  normalization (`../tpo/src/tpo/experiments/_trial.py:340`), and ordinary token
  completions sit at the benign end (~10–40 nats spread, so the ±4-nat tilt can still
  reorder near-tied candidates).
- Note the codebase already hit this wall from the PPO side: the per-dim thought clip +
  straight-through summed-gradient rescale (`train_latent_vapo.py:1076-1163`, "the
  enormous joint ratio of the 512-D diagonal Gaussian") exists precisely because the
  joint 512-D quantity is unmanageable.

Corollary answered along the way: the 16 samples do **not** need equal length per se —
the real requirement is (log-likelihood spread within a group) ≲ (reward tilt). Lengths
only matter insofar as they move log-likelihood. Per-step grouping removes the length
axis entirely.

## The proposal: per-step anchored target matching

At each action position, replace the clipped surrogate with anchored CE:

- **Scores** u = within-group z-scored **GAE advantages** (already computed per
  position, `train_latent_vapo.py:1287-1305`). This departs from critic-free paper-TPO
  and is deliberately MPO-flavored; per-step there is no alternative score source.
- **Anchor** = softmax over the group of the *old* per-step action log-probs (stored:
  gate B,S; token B,S; thought B,S,512 per-dim; see `refresh_old_statistics`,
  `latent_rollout.py:894-975`). q frozen at pool refresh; the current policy's log-probs
  come from the same teacher-forced replay the PPO path uses.
- **Group per factor, never mixed** (THINK densities ~+810 nats vs EMIT token log-probs
  ~−1 nat would otherwise hog the softmax):
  - **EMIT groups**: joint gate+token log-prob. Well-behaved; η ≈ 1.
  - **THINK groups**: summed 512-D density. Residual ~16-nat ε²-noise spread vs ±4-nat
    tilt ⇒ ~4× noise-dominated at η=1. **Use η ≈ 0.25** (inside the paper's robust
    range, Appendix D) to bring the tilt to ±16 nats; alternatively per-dim or chunked
    grouping so anchor spread ~ tilt (design knob, untested).
- **Group membership**: positions of the same factor within the optimizer minibatch (or
  per prompt-group). Caveat to keep in mind: per-step groups across the 16 trajectories
  are *cross-context* (different states at step t), so the anchor compares
  log π(a_i|s_i) across states — weaker theory than the paper's same-context group, but
  it is exactly what worked for Carson in continuous control.

What this deletes and keeps:

- **Keeps**: separate HL-Gauss critic, value warmup, length-adaptive GAE γ=1, packed
  shuffled pool with 4 disjoint age-0..3 minibatches, gate entropy bonus
  (initially), reverse-KL on thoughts (initially — TPO's frozen-q anchoring may
  subsume it; ablate its removal second).
- **Deletes**: PPO ratios and clip-higher, and the entire per-dim thought clip +
  straight-through rescale machinery — TPO's per-candidate gradient weight is p−q,
  bounded in [−1,1], with no ratios to explode. Frozen q structurally covers the
  staleness of minibatches 1–3 that clip+KL currently patch.

## Implementation sketch

1. New module `postraining/tpo/` (this folder): loss functions
   `emit_tpo_loss(old_logp, new_logp, advantages, group_index, eta)` and
   `think_tpo_loss(...)` mirroring `anchored_tpo_cross_entropy_loss` but taking
   precomputed old log-probs for the anchor (multi-minibatch: q from old, log p from
   current — paper §2 "if rollouts are reused, q stays frozen").
2. Fork the trainer entry (do not touch the v19 path): flag `--policy-objective tpo`
   in a forked script or a guarded branch inside `update_minibatch`
   (`train_latent_vapo.py:1171-1762`), replacing `clipped_policy_loss` (EMIT,
   `core.py:376-424`) and `per_dimension_thought_policy_loss` (THINK,
   `train_latent_vapo.py:1061-1150`).
3. Unit tests analogous to `postraining/tests/test_latent_thought.py`: gradient equals
   p−q on group logits; q ≈ p_old under zero-variance advantages (neutral groups); η
   scaling; frozen-q invariance across replay microbatch sharding (denominator
   convention must match `iter_length_aware_microbatches`).

## Ablation / evaluation plan

Same protocol as the current VAPO runs (mlq, one RTX 5090, `--max-parallel-runs 1`):

- Baseline: current v19 objective on the same pools/seed.
- Arms: (1) per-step TPO both factors (η_emit=1, η_think=0.25); (2) EMIT-only TPO,
  THINK stays PPO (isolates the risky factor); (3) η_think sweep {0.1, 0.25, 0.5, 1}.
- Metrics: existing prompt-group success, AIME avg@k, teacher-forced val-BPB guard,
  plus **anchor-health diagnostics** (log every update):
  - within-group entropy of p_old and of q (per factor) — near-zero entropy ⇒
    degenerate anchor;
  - fraction of groups where argmax q ≠ argmax p_old (tilt actually moving mass);
  - mean |p−q| (effective gradient weight; compare against PPO clip-frac);
  - THINK-factor: correlation of anchor rank with ‖ε‖² (should be high; confirms the
    noise mechanism) and with reward (should grow if learning).
- Kill criteria: q-entropy ≈ 0 on THINK groups at all tested η ⇒ fall back to arm (2)
  or per-dim chunked grouping.

## Open questions

- Cross-context grouping theory: is batch-standardized anchoring over different states
  principled enough, or should THINK groups be restricted to aligned early positions
  (the half-forced initial thinks are same-context by construction —
  `half_forced_group_members`, `latent_rollout.py:85-99` — and make a clean same-context
  subgroup to compare against).
- Does frozen-q allow raising off-policy reuse (real ppo-epochs > 1) that PPO-clip
  currently forbids? Paper Fig. 14–15 suggests yes; would amortize rollout cost.
- Can reverse-KL (coef 0.3) and/or gate entropy bonus be removed once TPO anchoring is
  in? Ablate after the main arm wins/loses.
- σ dynamics under TPO: no per-dim clip means the sigma head is disciplined only by the
  anchor + reverse-KL; watch log-sigma drift against its tanh bounds [−5, 2].

## Provenance

- VAPO code map and TPO feasibility notes: subagent reports in session
  2026-07-21/22 (vapo-mapper, tpo-redteam). Key verified numbers: +810 nats/THINK-step
  mean, 16-nat SD, ±4-nat tilt at K=16, σ-invariance of the noise floor.
- Carson: per-step TPO on top of PPO benefited continuous-control training (external
  experiments; motivates arm (1) directly).

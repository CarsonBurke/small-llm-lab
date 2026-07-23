# Post-Training the Energy-Readout Model

Settled understanding from design discussion, 2026-07-21. Scope: what changes
when the objective moves from *what the next token will be* (pretraining on
data) to *what the next token should be* (SFT / RL on preference or reward).
Nothing here is ablated yet; this is the design record, in the same spirit as
IDEA.md's registered hypotheses.

## The structural fact everything follows from

The policy lives in a continuous, SIGReg-regularized latent space; tokens are
emitted by an energy head over a tied codebook
(`c_k = latent_projector(rms_norm(tok_emb))`,
`logit_k = b_k − ½·Σ_d s_d(ẑ_d − c_{k,d})²`). The CE gradient through this
head is already an attract/repel policy update: attraction of ẑ toward the
target code ∝ (1 − p_y), repulsion from every other code ∝ p_k, self-balancing.

## 1. Energy-CE is the RL carrier — no objective surgery needed

Advantage-weighted CE (GRPO/DAPO-style, as in `postraining/`) has a direct
geometric meaning here: positive-advantage tokens pull the belief latent
toward their codes, negative-advantage tokens push it away. "Will be" → 
"should be" is just replacing the empirical next-token target with
advantage-weighted targets inside the same energy-CE. DPO maps the same way:
chosen/rejected pairs are paired attract/repel forces on the same ẑ.

## 2. Train everything; detach the codebook's *target* role only

Freezing the codebook (tok_emb + projector) during RL was considered and
rejected: it is non-standard, and in a 16MB artifact it parks ~4M parameters
that could still be improving. The trust region (below) already bounds
semantic drift, which is the only thing a freeze would buy.

The right cut is `ENERGY_DETACH_CODEBOOK=1`: detach the codebook only in its
**head** role, keep it fully trainable through its **input** role. Effects:

- The dictionary keeps improving, but only through *usage* — gradients from
  the trunk consuming context latents — never through the advantage-weighted
  attract/repel on the emission side.
- It removes the ugliest gradient channel in GRPO-style RL: negative-advantage
  tokens push their target representation down, and with tied embeddings one
  bad rollout would degrade that token everywhere it appears as input.
  Detaching the target side keeps destabilizing push-downs off the shared
  table entirely.
- Head and input cannot desynchronize: it is the same tensor, merely detached
  in the head computation, so the dictionary reflects input-path updates
  instantly. No frozen-copy staleness.
- The emission side still has full degrees of freedom: ẑ (trunk + prediction
  projector), the bias b_k, and the (per-dim) scale.

Freeze the per-dim scale s_d during RL regardless: RL pressure sharpens
distributions, and a learnable inverse temperature silently absorbs that as
entropy collapse. Control entropy explicitly (entropy bonus / KL), not through
the head's metric.

## 3. SIGReg during post-training: exactly where the encoder path trains

SIGReg touches only the encoder-side token latents (`embed_tokens` output);
its gradients reach exactly tok_emb + latent_projector and never ẑ. Hence:

- If the embedding path were frozen, SIGReg would be a constant — vacuous,
  drop it. (This is why the rejected freeze design had no SIGReg.)
- In the adopted design the embedding path trains, and with the head-side
  gradient detached, SIGReg + the input-path gradient are the **only** forces
  shaping the dictionary's geometry. SIGReg therefore stays on: it is what
  holds the dictionary spread open while narrow, non-i.i.d. RL data drifts it.
  The weight likely wants to come down from 0.09 (smaller, correlated
  batches); it still costs the pooled re-encode per step.
- The collapse people fear in RL — policy sharpening / entropy collapse — was
  never SIGReg's job (it never constrained ẑ). That belongs to the entropy
  bonus and the trust region.

## 4. Trust region in latent space, not just logits

Standard RLHF anchors with KL(π‖π_ref) on logits. This model adds a cheaper,
better-conditioned anchor: ‖ẑ − ẑ_ref‖² in the SIGReg-shaped space, where
distances are meaningful because marginals are held near N(0, I). Use both: a
small logit-KL (emission-distribution correctness) plus the latent L2
(geometry drift protection for the codebook alignment).

## 5. Cheap affordances of the regularized latent

- **Value / reward heads on the belief latent.** A linear or small-MLP head
  suffices precisely because the space is regularized — far cheaper than
  logit-level reward modeling.
- **Latent-space search → distill.** The head is an EBM: "should" can be
  computed at test time by gradient descent on ẑ under energy + reward, then
  sampled through the codebook. Expert-iteration post-training (search in
  latent, distill the improved emission via plain energy-CE) is unusually
  natural here, and composes with the latent-thought VAPO work — thought
  vectors live in the same well-conditioned space the readout consumes.

## The recommended stack

1. **SFT** on "should" data with unchanged energy-CE; SIGReg on (reduced
   weight) since the embedding path trains.
2. **RL** with the DAPO/VAPO harness: advantage-weighted energy-CE;
   `ENERGY_DETACH_CODEBOOK=1`; per-dim scale frozen; latent-L2 + small
   logit-KL trust region; value head on the belief latent; SIGReg on at
   reduced weight.
3. Optionally **latent-search distillation** on top.

## Open direction

Latent thinking (continuous thought steps in the SIGReg-shaped space, avoiding
Coconut-style backprop-through-time) is under discussion and not yet recorded
as settled design.

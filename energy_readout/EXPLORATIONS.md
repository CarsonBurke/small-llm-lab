# Explorations — InfoNCE and Geometry-Exploiting Decoders

Companion to `IDEA.md`. This document answers two questions: (1) is InfoNCE a
better way to turn LeJEPA predictions into discrete tokens, and (2) what other
approaches exploit the latent geometry LeJEPA builds? External literature
only; per project owner direction, no other experiment in this repo is used as
a reference.

## Bottom line

**The energy readout already *is* InfoNCE — the exact, full-negative-set,
zero-variance version of it.** `logit_k = −s·‖ẑ−c_k‖²/2 + b_k` with CE over
the whole codebook contrasts ẑ against every target latent exactly once.
Every batch-negative or sampled InfoNCE scheme (CPC, wav2vec 2.0, VL-JEPA) is
a *sub-sample* of that contrast — and at V=1024 a duplicate-heavy,
false-negative-ridden one. The right way to "use InfoNCE" here is to run the
energy readout with contrastive-literature hygiene (temperature in log-space,
clamped; frequency-informed bias; collapse monitoring), not to add batch
negatives.

## Part 1 — InfoNCE

### The subsumption argument

```
CE_energy = s·‖ẑ − c_y‖²/2                          ← attractive term (the latent MSE)
          + log Σ_k exp(−s·‖ẑ − c_k‖²/2 + b_k)      ← repulsion vs ALL codebook latents
```

Our target latent is the projected tied embedding of the actual next token —
a codebook entry. So any "other position's target latent" used as an in-batch
negative is some `c_k` the logsumexp already covers exactly. At V=1024 batch
negatives are dominated by duplicate frequent tokens (space, "the", comma),
which act as **false negatives**: they repel a token from its own code,
injecting gradient noise on exactly the tokens that dominate BPB. The
InfoNCE mutual-information bound grows as log(#distinct negatives); the full
codebook maxes it at V with zero duplication, at the cost of one `[d×V]`
matmul we already pay.

Per-source verdicts:

| Source | What it does | Verdict here |
|---|---|---|
| CPC (van den Oord 2018) | InfoNCE over marginal-sampled negatives, per-step bilinear scores, no temperature | Subsumed per step; multi-step *targets* are the one residual idea (below) |
| wav2vec 2.0 (Baevski 2020) | Cosine InfoNCE, fixed τ=0.1, 100 hard same-utterance distractors + codebook diversity loss | Their effort to get hard non-duplicate negatives is exactly what full-vocab CE gives for free |
| CLIP (Radford 2021) | Batch InfoNCE; temperature learned in log-space and **clamped** | Import the temperature hygiene, nothing else |
| Sampled softmax / NCE (Mnih & Teh 2012; Jozefowicz 2016) | Estimators of the full-softmax gradient for huge V | Pure variance at V=1024; reject |
| VL-JEPA / LLM-JEPA / NEPA (2025-26) | Embedding-prediction JEPA with contrastive or regression anti-collapse | Their targets are unquantized encoder outputs that need batch negatives; ours are SIGReg-regularized codebook entries — the need doesn't transfer. None reports a BPB/perplexity win; generation stays on a CE head |

### Honest exceptions (not subsumed by full-vocab CE)

1. **Multi-step future-latent prediction (CPC-style):** per-horizon projectors
   `P_k(h_t)` energy-decoded against `c_{y_{t+k}}`, k=1..K. Each step is
   individually a full-vocab CE, but the aggregate is denser future
   supervision than single-step CE can express. The only contrastive idea
   with residual signal; unproven for BPB, costs params and step time.
2. **Negatives outside the codebook** (other positions' *predicted* latents,
   perturbed latents): coherent but evidence-free for generative BPB, and
   reintroduces the collapse/false-negative problems. Not planned.

## Part 2 — Geometry-exploiting decoders

The organizing algebra: a Gaussian class-conditional model
`p(z|k) = N(c_k, σ²I)` with prior `π_k` has posterior
`log p(k|z) ∝ −‖z−c_k‖²/2σ² + log π_k` — and since `‖z‖²` cancels per
position, that is a **linear softmax** with `w_k = c_k/σ²`,
`b_k = −‖c_k‖²/2σ² + log π_k` (the classic GDA⇔softmax equivalence). Under
SIGReg's isotropic-Gaussian pressure on the latents, the energy readout is
therefore the (approximately) Bayes-matched parameterization whose weights
*are* the codebook — an inductive-bias and parameter-economy argument, not a
capacity argument. Caveat, stated honestly: SIGReg enforces the *marginal*
≈ N(0,I), not equal-covariance *class-conditionals*, so Bayes-optimality is
heuristic; and no published work yet shows a JEPA latent objective lowering
autoregressive BPB. A win here would be novel and must be earned by ablation.

| Candidate | Verdict |
|---|---|
| RBF-Softmax / Mahalanobis-GDA heads | Same linear softmax as ours; adopt the parameterization, expect no capacity gain |
| Logit adjustment (Menon 2020) | `b_k = log π_k` is our bias at convergence; valuable as **initialization** at 2k-step scale |
| vMF continuous head (Kumar & Tsvetkov 2019) | Exists to dodge big softmaxes; documented quality loss; strictly weaker at V=1024 — reject |
| kNN-LM (Khandelwal 2020) | Real nonparametric gains but the datastore alone dwarfs 16MB — reject |
| HL-Gauss / two-hot (Farebrother 2024) | Tokens have no ordinal axis — category error; the transferable residue is geometry-aware label smoothing (below) |
| Mixture-density / CALM / LatentLM heads | Multimodality argument applies to *continuous* targets; softmax over V is already fully multimodal. Only relevant if we ever predict continuous chunk latents — different project |
| Per-class covariance (full Mahalanobis) | O(V·d²) params — budget-fatal, reject |

## Part 3 — What we adopt

Ranked by expected BPB-per-engineering-hour; status maps to the
implementation in `energy_readout/fresh_lejepa_train_energy_readout.py`.

1. **Energy readout with contrastive hygiene** — log-space temperature with a
   straight-through **clamp** on the scale (`ENERGY_SCALE_MIN`/
   `ENERGY_SCALE_MAX`, linear-scale bounds; gradient stays alive at the
   bounds so the parameter can re-enter), codebook undetached, SIGReg
   load-bearing. *Implemented.*
2. **No duplicated attractive term** — the standalone attached MSE
   double-counts the CE numerator and re-adds unimodal point-prediction
   pressure; default `ENERGY_LATENT_MSE_WEIGHT=0`, swept in the ablation
   plan. *Implemented (validates the IDEA.md default).*
3. **Unigram-initialized bias** — `b_k ← log p̂(k)` warm-start
   (`ENERGY_BIAS_INIT_COUNTS=<json>`), learnable thereafter; pays off
   precisely in early-step BPB, which the 2k ablation measures.
   *Implemented; counts file generated offline from the training shards.*
4. **Geometry-aware soft targets** — CE against
   `q_k ∝ exp(−s′·‖c_y−c_k‖²/2)` instead of one-hot; speculative (must beat
   plain label smoothing). *Documented only; future fork.*
5. **Multi-step future-latent auxiliary CE** — the one non-subsumed
   contrastive idea; params and step time make it a post-win follow-up.
   *Documented only.*

Rejected outright: batch/sampled InfoNCE negatives, sampled softmax/NCE, vMF
head, kNN-LM, mixture-density heads, per-class covariances (reasons above).

## Collapse caution

With the codebook undetached, SIGReg is **load-bearing**: CE is
scale-invariant, so nothing else prevents joint ẑ–codebook contraction
(distances → 0, `s` → ∞ compensating), nor "prototypes chasing the encoder"
(the VQ-VAE degeneracy). Canaries logged every validation: learned scale
`s`, mean codebook norm, mean pairwise codebook distance. Escalation path if
they trend degenerate: raise SIGReg weight → constrain codebook norms →
freeze `s` → detach the codebook in the repulsion term only.

## Flagged uncertainties

- The Bayes-decoder view is contingent on class-conditional Gaussianity,
  which SIGReg does not directly enforce.
- VL-JEPA's temperature handling and LLM-JEPA's exact negative mechanism
  could not be verified from primary text.
- No primary source demonstrates a JEPA latent objective improving
  autoregressive BPB; that claim is unproven and ours to test.

## Sources

- CPC: https://arxiv.org/abs/1807.03748
- wav2vec 2.0: https://arxiv.org/abs/2006.11477
- CLIP: https://arxiv.org/pdf/2103.00020 (temperature clamp: https://github.com/openai/CLIP/issues/46)
- Sampled softmax / LM scaling: https://arxiv.org/abs/1602.02410 · https://arxiv.org/abs/1410.8251
- VL-JEPA: https://arxiv.org/abs/2512.10942 · LLM-JEPA: https://arxiv.org/abs/2509.14252 · NEPA: https://arxiv.org/html/2512.16922v2
- vMF output head: https://arxiv.org/abs/1812.04616 · follow-up: https://arxiv.org/pdf/2310.20620
- RBF-Softmax: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5351_ECCV_2020_paper.php
- Mahalanobis/GDA: https://arxiv.org/abs/1807.03888 · GDA⇔softmax: https://kuleshov-group.github.io/aml-book/contents/lecture7-gaussian-discriminant-analysis.html
- HL-Gauss: https://arxiv.org/abs/2402.13425 · Stop Regressing: https://arxiv.org/abs/2403.03950
- kNN-LM: https://arxiv.org/abs/1911.00172 · analysis: https://arxiv.org/pdf/2301.02828
- Logit adjustment: https://arxiv.org/pdf/2007.07314
- CALM: https://arxiv.org/pdf/2510.27688 · LatentLM: https://arxiv.org/pdf/2412.08635
- LeJEPA: https://arxiv.org/abs/2511.08544 · LeWorldModel: https://arxiv.org/abs/2603.19312 · I-JEPA: https://arxiv.org/abs/2301.08243
- LeVLJEPA (the paper that prompted this family): https://arxiv.org/abs/2607.00784

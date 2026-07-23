# Energy Readout — Tied-Codebook Token Emission for LeJEPA

Companion docs: `EXPLORATIONS.md` (InfoNCE verdict and the survey of other
geometry-exploiting decoders; motivates the temperature clamp, unigram bias
init, and collapse canaries adopted below).

## The Idea

The lejepa-ce lineage trains two decoders that learn overlapping structure:

1. **Latent MSE** pulls `predicted = prediction_projector(belief)` toward the
   projected embedding of the actual next token (`target_latent`), building a
   metric structure where the prediction sits near the correct token's code.
2. **A 4.2M-parameter ResidualProbe** maps `cat(token_latent, belief)` to vocab
   logits with CE, learning token discrimination from scratch as a separate
   parametric classifier.

At decode time only (2) is used; the metric structure that (1) spent gradient
budget building is thrown away. The probe is also the single largest non-trunk
artifact expense, and the untrained critic probe ships another ~3.15M random
parameters.

This family deletes the probes and reads token logits **directly from the
JEPA geometry**: an energy (negative squared distance) between the predicted
latent and the projected tied codebook. LeVLJEPA (arXiv 2607.00784) decodes its
discrete decisions by similarity-argmax against candidate embeddings, but only
as an *untrained* zero-shot readout — and loses exactly there to InfoNCE
baselines. This experiment tests the trained-and-generative version of that
readout: CE through the codebook energy, so the discriminative signal and the
JEPA prediction signal flow through one shared geometry.

## The Readout

For each position `t` with belief `h_t`:

```
ẑ_t      = prediction_projector(h_t)                       # existing JEPA prediction path
c_k      = latent_projector(rms_norm(tok_emb))_k           # projected tied codebook, NOT detached
logit_tk = -s · ||ẑ_t - c_k||² / 2 + b_k
```

- `s = exp(log_s)`: learned global inverse-temperature, `log_s` initialized to
  `-½·ln(model_dim)` (i.e. `s = 1/√d`). Red-team calibration note: the token
  projector has no output norm, so at init `‖c‖² ≈ 72` (not `d`) and the logit
  spread is ~0.15 — *under*-dispersed, starting CE ≈ ln(V) like the parent's
  zero-init probe; safe. As SIGReg drives code marginals toward `N(0, I)`,
  `E‖c‖² → d` and `s` must adapt downward — `s` is logged every val step.
- `b_k`: learned per-token bias, zero-initialized. Absorbs unigram frequency.
- **No softcap on this head.** Distance logits are one-sided (bounded above by
  `b_k`, unbounded below); `softcap·tanh(raw/softcap)` saturates the *target's*
  logit precisely when its code is far — early training, when the attractive
  gradient matters most.
- Computed as `-s/2·(||ẑ||² - 2·ẑCᵀ + ||c||²) + b`: one `(B,T,d)×(d,V)` matmul,
  the same cost as a standard tied LM head. Logits in fp32 for the CE.

## Why CE Through Energy Unifies the Two Losses

```
CE_t = s/2·||ẑ_t - c_y||² - b_y + log Σ_k exp(-s/2·||ẑ_t - c_k||² + b_k)
```

The numerator term equals the attached latent MSE (scaled by `s`) **at the
loss-value level**. The gradient story is different and must be stated
honestly: `∂CE/∂ẑ = s·(c̄_p − c_y)` where `c̄_p` is the probability-weighted
mean code — the logsumexp exactly cancels the displacement-proportional pull
of the numerator. So with `λ_lat = 0`, `ẑ` is **not** anchored to the latent
manifold as a regression target; it is fixed only up to softmax-equivalence
(a gauge freedom against `s` and `b`). For BPB this is exactly a log-linear
LM and is fine; but `predicted` stops being a latent-space regression, which
is what downstream latent-rollout consumers assume. **Decision:** under the
default `λ_lat = 0`, `predicted` is declared a readout-only feature; restore
`ENERGY_LATENT_MSE_WEIGHT > 0` when world-model semantics are needed.

What the CE does add over pure MSE is the repulsive force — MSE's
mode-averaging pathology (an MSE-optimal ẑ is the probability-weighted mean
of plausible codes, near none of them) is replaced by a proper log-linear
distribution over codes. Because
`-s/2·||ẑ-c_k||² = s·ẑ·c_k - s/2·||c_k||² - const_t`, and per-position
constants cancel exactly in softmax CE, this head has the same expressiveness
as a standard tied-softmax LM head plus a per-token norm bias — multimodal
next-token distributions are represented exactly as in any log-linear LM.

This is also the full-negative-set InfoNCE: CE over the vocabulary with
similarity logits is InfoNCE whose negative set is the entire codebook, exact
and computed for free at V=1024. In-batch-negative InfoNCE would be a noisy
sampled approximation of this same objective and is deliberately not used.

## Loss Composition

```
L = CE(energy_logits, y)
  + λ_lat · MSE(ẑ, c_y)          # default 0 — subsumed by the CE numerator
  + λ_sig · SIGReg(trajectory)   # unchanged from parent (0.09, paired B128, deferred)
```

`λ_lat` stays as an env knob (`ENERGY_LATENT_MSE_WEIGHT`, default `0.0`) so the
explicit-MSE anchor can be restored as a fallback ablation without a new fork.

## What Is Deliberately Different From the Parent

Parent: `fresh_lejepa_train_v1_probe_shared_rms_pope_belief_attached.py` — the
latest lejepa-ce variant (1.4009 BPB @ 2k on `mathmix_v4_sp1024`). This is the
**only** in-repo reference for this family; no other experiment in the repo is
used as a basis or comparison.

| Aspect | Parent | This family |
|---|---|---|
| Emission head | 4.2M ResidualProbe on `cat(token_latent, belief)` | tied energy readout, ~1K new params (`log_s`, `b`) |
| Critic probe | 3.15M untrained params in artifact | not installed |
| Codebook gradient | probe CE reaches embeddings via input features | CE reaches embeddings *as the classifier weights* (non-detached codebook) |
| Prediction projector role | latent MSE only, "exclusively the world-model path" | carries the CE signal — deliberately reverses the parent's separation, because routing discrimination through the JEPA prediction *is* the hypothesis |
| Latent MSE | weight 1.0, separate loss | default 0 (inside CE numerator); knob to restore |
| Softcap | 30·tanh on probe logits | none on energy head |
| Current-token input to head | yes (`token_latent` feature) | no — optional bigram logit table (below) |

Optional flag `ENERGY_BIGRAM_TABLE=1`: adds a zero-initialized `V×V` additive
logit table indexed by the current token (~1.05M params). This is the cheapest
replacement for the current-token shortcut channel the probe had via its
`token_latent` input; default off to keep the primary ablation pure.

## Scope

Per project owner direction, this family is derived **only** from the current
lejepa-ce model (`fresh_lejepa_train_v1_probe_shared_rms_pope_belief_attached.py`
and the modules it inherits from). No other experiment in this repo is used as
a reference, basis, or evidence source. If the pure energy readout loses at
2k, the first fallback is `ENERGY_BIGRAM_TABLE=1` (restores the cheapest form
of the current-token channel the probe had) before abandoning the family.

## Risks

- **Losing the nonlinear `cat(token_latent, belief)` channel** may cost real
  BPB (the strongest objection from the internal red-team; the bigram flag is
  the mitigation, a full probe restore the fallback).
- **Embedding table serves three geometries** (trunk input, world-model target,
  classifier weights) at `tied_embed_lr=0.05` tuned for the baseline; may need
  LR retune. Watch `emb` cosine structure and effective rank.
- **Scale dynamics**: a single learned `s` couples the CE sharpness to the MSE
  scale; if `s` grows large the attractive term dominates and the loss reverts
  toward hard-assignment MSE. Log `s` every val step.
- Artifact remains over the 16MB cap even after deleting probes (~21M params
  pre-fold); the serialization-side fixes (fold `latent_projector`, strip
  training-only modules) are a separate workstream and orthogonal to this
  ablation.

## Hypothesis (registered before results, run `energy_readout_2k`, job 239)

**H1:** Routing the emission CE through the codebook-energy geometry holds BPB
within +0.02 of the parent (1.4009 @ 2k, mathmix_v4) — or beats it — despite
deleting 7.35M readout parameters, *because* the probe's capacity was largely
re-learning the metric structure the latent losses already build, and
full-vocabulary CE through that geometry supervises the codebook directly as
classifier weights instead of indirectly through probe inputs.

**H2 (secondary):** Dropping the duplicated attractive term (λ_lat = 0)
improves calibration on multimodal next-token positions relative to
CE + full-weight MSE, visible as BPB gain concentrated after ~step 500 once
the codebook has structure.

Registered predictions:

- `val_bpb` @ step 0 ≈ 4.10 (near-uniform start, matching the parent).
- `val_bpb` @ 2k central estimate **1.40–1.44**; win if < 1.3959 (protocol
  threshold); 1.396–1.421 counts as favorable anyway — equal BPB from ~7.35M
  fewer artifact params is a parameter-golf win; > 1.44 falsifies H1.
- `step_avg_ms` 5–10% below the parent's ~1978ms (probe+critic FLOPs removed,
  minus the full-vocab codebook projection cost).
- `energy_scale` rises from 0.044 and settles in ~0.1–2 without touching
  either clamp; `codebook_pairdist_mean` grows as SIGReg inflates code norms.
  A falling pairdist with `s` racing upward is the joint-collapse signature.
- The diagnostic latent-MSE component drifts *upward* vs the parent — at
  λ_lat = 0 nothing anchors ẑ as a regression target (the gauge freedom
  documented above); this is expected, not a failure.

Attribution rules, decided in advance: if H1 fails (> 1.44), the probe's
current-token channel was load-bearing → run `energy_readout_bigram_2k`
before abandoning. If H1 holds, run the `dot` and detached-codebook controls
to attribute the result before claiming the geometry mattered.

In-flight canary calibration note (step ~200, recorded before completion):
the raw `energy_scale` band above assumed a near-unit-norm codebook; SIGReg
instead inflated the codebook to ~N(0,I) scale (norm mean ≈ 28 ≈ √512), so
`s` compensated *downward* to ~0.019. The substantive prediction (bounded
effective temperature `s·pairdist²/2`, distances growing — the anti-collapse
direction) holds; the raw 0.1–2 band was miscalibrated.

## FineWeb-Matched Comparison (registered before results, jobs 240/241)

To compare against `baseline_2k` without confounds the training data must be
FineWeb-only. Trunk shape is already matched by construction —
`FreshHyperparameters` subclasses the baseline `Hyperparameters`, inheriting
9 layers × 512 dim × 8 heads (4 KV) × mlp_mult 2, seq 1024, 524,288 batch
tokens, vocab 1024 (same SP tokenizer), same `fineweb_val` shard. Dataset
`fineweb_onepass_sp1024` (built by `build_math_mix_dataset.py` with
`--fineweb-fraction 1.0`, zero-fraction sources now skippable) reshapes the
baseline's own FineWeb tokens into the strict deterministic one-pass layout
the belief-attached family requires.

Residual differences that are the *design under test*, not confounds: latent
projectors + SIGReg + PoPE + energy head (vs. softcapped tied dot head),
warmup 0 (one-pass contract) vs. baseline warmup 20, and one-pass ordering
vs. the baseline loader's ordering over the same tokens.

References on this data: `baseline_2k` = 1.2967; pope-attached-CE
(`..._scratch_2k`) = 1.3406. No belief-attached FineWeb reference existed,
so job 241 runs one.

**H3:** `energy_readout_fineweb_2k` lands within ±0.02 of
`belief_attached_fineweb_2k` (head parity transposes across data), with a
central estimate of 1.33–1.37 by analogy to the pope-attached 1.3406.
Beating `baseline_2k` (1.2967) at 2k is *not* predicted — the family
currently pays a BPB premium for the latent machinery; the family's case
remains parameter/artifact economy plus any closing of that gap.
(Job 241 was cancelled before start by project direction; H3's parent
reference does not exist, so the FineWeb result is scored against the
baseline references only.)

## BN-Projector Arm (registered before results, job 242)

The lineage's original projector was `TokenProjector`
(Linear→FP32BatchNorm1d→GELU→Linear); the shared-RMS fork swapped BN for
RMSNorm. The recorded head-to-head at 1k steps (probe family, default data)
favored **BN**: `fresh_lejepa_shared_bn_v1_probes_1k_retry` = 1.5719 vs
`fresh_lejepa_shared_rms_v1_probes_1k` = 1.5817 (+0.0098 for BN, ~2× the
keep threshold, equal step time ~1537ms). The first BN attempt crashed at
startup (rc=1, zero entries) and was retried successfully; the lineage kept
RMS anyway — no recorded justification found. LeJEPA and LeWM both use BN
MLP projectors (le-wm: `config/train/model/lewm.yaml`, BatchNorm1d in both
`projector` and `pred_proj`).

`energy_readout_bnproj_fineweb_2k` restores BN in both projectors under the
energy head. The codebook uses `TokenProjector.inference` — running-stats
BN, no stat update, gradients attached through the linears and BN affine —
because the vocab table is not a training batch (mechanism already present
in the lineage's `policy_codebook`).

**H4:** BN projectors beat the RMS energy run (job 240) at 2k by
0.005–0.015 BPB if the 1k probe-family effect transfers; a null or negative
result at equal step time keeps RMS and closes the question. Known risk,
accepted: during training the latents are batch-stat normalized while the
codebook is running-stat normalized; the mismatch shrinks as statistics
converge, and would show up as elevated early policy loss relative to
job 240, converging by mid-run. Single-GPU BN statistics; 8×H100 would need
SyncBN or per-rank stats validation before any full-scale run.
(Job 240 was cancelled at step 760 — val_bpb 1.4267 — to free the GPU for
this arm; the H4 comparison is scored at matched steps ≤760 plus the final
BN value against the registered band. Early evidence: BN ahead at every
eval through step 160, e.g. 2.181 vs 2.394 @ 40 — and the predicted early
BN penalty did not materialize.)
(Job 242 was in turn cancelled at step 400 per the planned switch to the
per-dim arm — last val 1.4958 BPB @ 380, still ahead of the RMS run at every
matched eval. H4 is scored on the matched-step trajectory ≤400.)

## Per-Dim Scale Arm (registered before results, job 243)

`energy_readout_perdim_bn_fineweb_2k`
(`energy_readout/fresh_lejepa_train_energy_readout_perdim.py`): on top of
the BN arm, the scalar inverse-temperature becomes a per-dimension vector —
`logit_k = b_k − ½·Σ_d s_d·(ẑ_d − c_{k,d})²`, i.e. diagonal-covariance GDA.
512 params, Adam scalar group, elementwise straight-through clamp, init
identical to the scalar head (all dims at 1/√d). Canaries:
`energy_scale_mean/min/max` + the codebook stats.

**H5:** per-dim scale beats the scalar BN arm (job 242) by ≥0.005 BPB at 2k
*if* the latent dimensions carry usefully heterogeneous signal/noise; the
mechanism is the head down-weighting noisy axes instead of one global
temperature compromising. Null result (|Δ| < 0.005) is the expected outcome
under SIGReg's isotropy pressure (marginals pushed toward N(0,I) leave
little per-axis structure to exploit) and would close the question in favor
of the scalar head. Falsifying signature for the mechanism: if
`energy_scale_max/energy_scale_min` stays ≈1 all run, the head found no
axis structure and any BPB delta is noise, not mechanism. Step-time cost
should be nil (one extra elementwise multiply).
(Sequencing: job 243 was cancelled at step 270 — ahead of the planned
step-400 switch, per user direction — to hand the GPU to the pooled-B512
arm. Last val 1.5497 BPB @ 260, slightly ahead of scalar BN at matched
steps (e.g. 1.6351 vs 1.6425 @ 160). Mechanism canary fired: the per-dim
scale spread reached ~9× (max 0.0671 / min 0.0074 @ 260), so the head does
find axis structure despite SIGReg isotropy pressure. H5 is scored on the
matched-step trajectory ≤260 against job 242.)

## Pooled-B512 SIGReg Arm (registered before results, job 244)

`energy_readout_perdim_b512sigreg_fineweb_2k`
(`energy_readout/fresh_lejepa_train_energy_readout_perdim_b512sigreg.py`):
on top of the per-dim arm, the paired SIGReg scheme (four B=128 statistics
per optimizer step from 8×B=64 microbatches) becomes le-wm's exact
structure: **one** pooled B=512 statistic per step, computed by re-encoding
all eight microbatches in a single forward. Because the Epps–Pulley
statistic is ×B-scaled, the null floor is B-independent while deviation
signal grows ∝B — pooling quadruples test power over paired-128 at the same
weight 0.09, so no re-tune. It also removes the even-local-accumulation
requirement that breaks the paired scheme at 8×H100 world_size=8
(grad_accum_steps=1); this module still requires the whole 512-sequence
step batch on one rank (cross-rank gather deliberately not implemented).
Gradient scale is identical to the paired totals for any grad_scale
definition (`stat·w·grad_accum_steps·grad_scale` once ≙
`stat·w·2·grad_scale` per pair × accum/2 pairs); the logged
train_loss/components contributions are ×grad_accum_steps so the later
÷grad_accum_steps recovers the per-step statistic. The pooled encode runs
BN in train mode over the full 512×1025 batch — one batch statistic per
step, also the le-wm configuration.

**H6:** the pooled statistic's 4× power tightens the latent geometry's
adherence to N(0,I) at equal weight; the honest BPB prediction is
small-positive-to-null at 2k (|Δ| vs job 243 within ±0.005 would not be
surprising — SIGReg was not visibly underpowered at B=128). The run's
primary value is validating the 8×H100-compatible SIGReg design before any
full-scale run; a BPB *regression* >0.005 would falsify "more test power is
free" and keep paired-128. Costs to watch: one extra pooled encode per step
(~+12% forward FLOPs, unchecked activations ~5–8 GB transient on the 5090)
— step_avg_ms vs job 243 is part of the verdict. Canaries unchanged from
the per-dim arm; sigreg_loss is now a single-statistic estimate (expect
lower variance in its train trace).
(Job 244 OOM'd at startup: 2 GiB request failed with 10.56 GiB
reserved-but-unallocated — allocator fragmentation from the pooled-encode
transient, not true exhaustion (the 5090 also shares ~7 GB with desktop
processes). Retried as job 245 with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — no numerics change.
If 245 still OOMs, the principled fallback is a memory-leaner pooled
encode, not smaller B.)

**H6 correction (registered mid-run, before the verdict):** the "no
weight re-tune needed" claim was a null-floor argument and only holds AT
the null. The measured geometry is far off the null (pooled statistic
~2.55 vs paired ~1.35 at matched steps), and off-null both the statistic
and its gradient scale ∝B — so weight 0.09 on pooled-512 exerts ~4× the
restoring force of 0.09 on paired-128. Early evidence this is costing
BPB: 245 trails the per-dim paired run by ~0.02 at matched steps
(2.0344 vs 2.0154 @ 60), the per-dim scale spread is suppressed (2.4× @
60 vs the paired run's 7.5× @ 160 — isotropy enforced harder), and
codebook pairdist runs lower. Equal-pressure alignment would be
`FRESH_LEJEPA_SIGREG_WEIGHT≈0.0225` on the pooled script; together with
job 247 (weight 0) the family brackets sigreg pressure at {0, ~1×, ~4×}.
Step-time cost confirmed: ~2245ms vs ~1990ms paired (+11%, the pooled
re-encode).
(Job 245 cancelled at step 120 per user direction — last val 1.7607 vs
paired 1.719 @ 120; the +0.04 gap never closed. H6 verdict on the ≤120
trajectory: pooled-512 at unadjusted weight 0.09 is WORSE than
paired-128, consistent with the ×4-pressure correction above. The
structural design remains validated — the run trained correctly, +11%
step time — only the weight transfer was wrong.)

## Equal-Pressure Pooled Arm (registered before results, job 248)

`energy_readout_b512_sigreg0225_fineweb_2k`: the pooled-B512 script with
`FRESH_LEJEPA_SIGREG_WEIGHT=0.0225` (= 0.09/4), matching the per-deviation
restoring force of paired-128 @ 0.09 while keeping le-wm's
one-statistic-per-step structure and the 8×H100-compatible loop. Runs
after job 247, completing the pressure bracket {0, ~1× (this), ~4× (245)}
with paired-~1× (243) as the structural control.

**H9:** at matched pressure, pooled-512 matches or slightly beats
paired-128 at matched steps (|Δ| ≤ 0.005 expected; the pooled statistic
is lower-variance, which if anything helps). Signatures: per-dim scale
spread and codebook pairdist should recover to the paired run's
trajectory (spread ≳7× by step ~160, pairdist ~42+ by 160). If it still
trails paired by >0.005 with canaries matched, the deficit is not
pressure but something structural in pooling (e.g. the B=512 BN
batch-stat regime in the sigreg encode) and paired-128 stays the recipe
on single-GPU while 8×H100 needs its own solution.
(Verdict @ step 400, cancelled per cadence: H9 CONFIRMED — pooled@0.0225
tracks paired@0.09 at matched steps (1.5429 vs 1.5497 @ 260, the last
paired point), so the ×4 pressure correction and the pooled structure
are both validated. But the no-sigreg run beats both in the late window
(1.4850 vs 1.4996 @ 400, +0.015): the pressure bracket is monotone —
0 > ~1× > ~4×. SIGReg is a mild drag at every tested weight in this
family; the stripped base is the winning configuration. Last val
1.4996 @ 400; canaries: scale spread widened to 12× under weak sigreg —
axis structure grows as pressure drops, consistent across the bracket.)

## Checkpoint Geometry Analysis (step-200, sigreg 243 vs no-sigreg 247)

CPU analysis of the projected codebook (1024 codes, 512 dims) from both
runs' step-200 checkpoints, testing two hypotheses about what SIGReg
does for the geometry. Both were refuted:

- "SIGReg keeps dims from getting too large" (user hypothesis): per-dim
  std spread is essentially identical (max/median 1.90 no-sigreg vs 1.86
  sigreg; max/min 2.61 vs 2.85), and the largest single-dim std is
  SMALLER without sigreg (1.83 vs 2.41). Median per-dim std sits at
  ~0.96 ≈ 1 with no regularizer at all — CE + the BN-hidden projector
  self-normalizes.
- "No-sigreg risks dimensional collapse" (my hypothesis): backwards —
  effective rank (participation ratio) is HIGHER without sigreg (21.4 vs
  12.5 of 512); the sigreg'd run concentrates 27.6% of variance in one
  direction vs 19.4% without.
- Mean offset (gauge drift) also comparable: |mean| 8.2 vs 9.3.

Conclusion: sigreg's snapshot effect is a modest rescale (code norm 31 vs
23); the shape CE builds unregularized is nearly the shape sigreg
enforces — consistent with the BPB dead heat (±0.006, lead trading, 247
pairdist recovering 27.6→29.0 by step 140). The proposed output-BN
projector arm was dropped as unmotivated: the property it would pin
(per-dim variance ≈1) already holds. Remaining unexplained: the ~43
unweighted EP statistic; per-dim snapshot stats don't show what it sees
(likely frequency-weighted train-latent structure); not currently
BPB-relevant.

## Stripped No-SIGReg Arm (registered before results, job 249)

`energy_readout_perdim_nosigreg_stripped_fineweb_2k`
(`energy_readout/fresh_lejepa_train_energy_readout_perdim_nosigreg.py`):
per-dim BN arm with the sigreg apparatus REMOVED rather than
zero-weighted — no re-encode, no statistic, no RNG projection draws, no
batch-shape guards (grad accumulation constrained only by divisibility →
8×H100-compatible). Component bookkeeping and logging schema unchanged
(sigreg slot logs 0). Runs after job 248.

**H10:** BPB trajectory statistically identical to job 247 (|Δ| ≤ 0.005
at matched steps; only differences are 8 vs 9 BN stat updates/step and
the RNG stream) at ~1990 ms/step vs 247's ~2245 (−11%). This is the
engineering-validation run for the 10k candidate: if it matches 247, the
stripped module is the scale-up base — at full scale the reclaimed 11%
step time converts to ~10% more training steps in the 10-minute budget,
worth far more than any measured sigreg effect.
(Verdict @ step 440, cancelled per cadence: H10 CONFIRMED and the
step-time win was radically underestimated — 915 ms/step vs 2245 for
the weight-0 run (−59%, not −11%: the dominant sigreg cost was the
1024-projection ECF statistic + checkpointed recompute, not the pooled
encode). BPB statistically identical to job 247 (−0.004 @ 200,
+0.000 @ 300, +0.003 @ 400). Last val 1.4719 @ 440 — family best.
The stripped module is the certified scale-up base.)

## Head-Extension Arms on the Stripped Base (registered before results, jobs 250–252)

All three build on the stripped no-sigreg module (job 249's base): per-dim
scale, BN projector, no sigreg apparatus. Each is single-factor vs 249 and
starts numerically identical to it at step 0 (zero/symmetric init, trunk
RNG parity verified by test). Standard 400-step cadence applies.

### Low-Rank Current-Token Residual (job 250)

`energy_readout_lowrank_token_fineweb_2k`: restores the probe's deleted
current-token channel inside the code geometry —
`Δlogit_k = (U e_x)ᵀ(V c_k)`, rank 32 (~49k params, both factors flat →
fused-Adam). LoRA init: U ~ N(0,0.02), V = 0 → exact base head at step 0,
V gets gradient immediately. Chosen over the 1M-param bigram table: tied
to the live codebook, 20× cheaper, and rank-32 suffices for the
"emission depends on where you are, not just where you're going" channel.

**H11:** the strongest of the three (the red-team called the deleted
token channel the family's top BPB risk). Win = >0.005 BPB vs 249 at
matched steps; a bigram-like tilt should show as fast early gains
(current-token statistics are learnable within tens of steps).

**H11 verdict (2026-07-22, cancelled @ 500): NOT SUPPORTED — discard.**
Trails 249 at every matched step: 1.5033 vs 1.4878 @ 400, 1.4879 vs
1.4736 @ 420, 1.4770 vs 1.4719 @ 440 (−0.005 to −0.015). No fast early
gains ever materialized, so the "deleted token channel" risk was
overstated: the trunk already carries current-token information into ẑ.
The rank-32 residual only added optimizer noise at this scale.

### Mixture of Prototype Energies, M=4 (job 251)

`energy_readout_mixture4_fineweb_2k`: breaks the softmax bottleneck with
p = Σ_m π_m(ẑ)·softmax(exp(δ_m)·L + b_m); shared codebook and per-dim
metric, per-component temperature+bias, linear gate on ẑ (~6.7k params,
flat → Adam). At init all components identical ⇒ output exactly the base
distribution regardless of gate noise; gate noise breaks symmetry via
component posteriors after step 1. Val loss = mixture NLL (this is a
capacity change; the mixture IS the model's likelihood).

**H12:** modest win (0.005–0.01) if the log-linear rank bound d+1=513 is
binding at V=1024; null if not binding at this scale. Canary: effective
gate entropy — if π collapses to one component, capacity was declined
and any delta is noise.

**H12 verdict (2026-07-22, cancelled @ 400): NULL — discard.** Matched
steps vs 249: −0.002 @ 340, +0.003 @ 360, +0.003 @ 380, +0.002 @ 400 —
a consistent whisker ahead late but under the 0.005 bar, at +9% step
time (995 vs 915 ms). The rank bound isn't binding at V=1024/d=512; the
mixture's extra likelihood capacity buys noise-level gains here. (No
memory blowup — the 3GiB/microbatch review concern didn't materialize.)

### Geometry-Aware Soft Targets (job 252)

`energy_readout_softgeo_fineweb_2k`: train-only target
(1−ε)·onehot + ε·q with ε=0.1 and q = the head's own emission
distribution at the true code (detached; live per-dim metric + bias — no
new temperature hyper). Eval stays plain CE (comparable BPB).

**H13:** weakest prior of the three — soft targets usually COST val
likelihood (why plain smoothing was rejected); win condition is that
geometry-aware leakage regularizes better than nothing at 2k. Any win is
provisional until it also beats plain label smoothing at equal ε
(control not yet queued). Expect early val_bpb slightly worse, verdict
from the mid-run trend.

**H13 verdict (2026-07-22, cancelled @ 400): NOT SUPPORTED — discard.**
Matched steps vs 249: −0.003 @ 340, +0.002 @ 360, −0.002 @ 380,
−0.004 @ 400. Slightly worse to even, exactly the classic
label-smoothing likelihood tax; the geometry-aware leakage did not
regularize better than nothing at this scale. The plain-smoothing
control is moot. Prior confirmed.

### Scaled Tied-Dot Head (job 253)

`energy_readout_dothead_fineweb_2k`: the stripped no-sigreg script run
with `ENERGY_HEAD_FORM=dot` (existing knob, zero new code) —
`logit_k = b_k + Σ_d s_d ẑ_d c_{k,d}`. Algebra: the distance form is
exactly dot + b_k − ½‖c_k‖²_S (the ẑ² term is k-independent and cancels
in softmax), so this arm deletes ONLY the live norm penalty "prefer
small codes." Since b_k is a learned per-token bias, the two forms are
equivalent up to reparameterization for a STATIC codebook; they differ
in dynamics: the distance form re-derives the penalty from the live
codebook every step (norm growth is instantly self-penalizing), the dot
form asks the bias to learn it (one slow scalar per token) and otherwise
lets code norm act as a free confidence/frequency dial — the standard
tied-softmax behavior.

**H14:** distance form's automatic norm accounting is worth a small
amount (dot ends 0.000–0.010 WORSE at 2k) because the codebook moves
fast in this family and b_k lags ‖c_k‖²_S for rare tokens. Falsified if
dot matches or wins — which would say the norm term was a leash, codes
want norm-as-confidence, and the simpler head is preferable (also
slightly cheaper: no c_sq/z_sq terms). Canaries: expect codebook_norm to
grow larger and spread wider under dot (norm now encodes frequency);
watch rare-token bias drift.

**H14 verdict (2026-07-22, cancelled @ 400): SUPPORTED (weakly) —
distance form retained.** Matched steps vs 249: −0.007 @ 340,
+0.001 @ 360, +0.002 @ 380, −0.001 @ 400. Worse early, then a tie
inside noise — within H14's predicted band, never better by the keep
threshold. The live norm penalty costs nothing and helps early; the
family keeps the distance head, and the nanogpt-mini port (job 255)
runs ENERGY_HEAD_FORM=distance.

## Family scoreboard (2k arms, final)

249 stripped no-sigreg per-dim distance head is the family
configuration: 250 lowrank LOST, 251 mixture TIE (under threshold,
+9% cost), 252 softgeo LOST, 253 dot TIE (worse early). Every
extension was refuted; the base is the ported architecture.

## Scale-Up Plan (SUPERSEDED 2026-07-22)

~~Take the best performer at 2k and run it for a full 10k steps.~~
Superseded by user directive: skip the 10k run. Instead, after the
250–253 queue drains:

1. **nanogpt-mini baseline**: fork `../modded-nanogpt/train_gpt.py`
   (current speedrun record) into a single-RTX-5090 variant at ~20M
   params (reduce layers + MLP width as needed; SDPA for FA3, drop
   FP8/distributed as required). Run 1k steps.
2. **nanogpt-mini + energy head**: port the no-sigreg energy readout
   (or whichever 250–253 arm wins) onto that baseline. Run 1k steps.
   Hypothesis: the energy head transfers — it should beat the mini
   baseline's tied softmax if implemented correctly.

### nanogpt-mini baseline (`nanogpt_mini_train.py`)

Fork of `modded-nanogpt/records/track_3_optimization/train_gpt_simple.py`
(the fixed-arch 124M-class optimization-track baseline, per user), NOT the
495M-param speedrun HEAD. Already SDPA/no-FP8/single-GPU-capable; kept
verbatim except: 12L/768d → 6L/512d (19,958,784 params at vocab 1024,
head_dim 128 → 4 heads, MLP 4×d), our sp1024 shards + DATA_PATH, env
knobs for the ablation harness, val_bpb via train_gpt.py's byte-LUT,
fixed seed for init parity with the coming energy variant. Upstream's
untied embed/head, std-1 embed init, zero-init projections, AdamW
(0.7/0.004/0.015) + Muon (0.025, NS-12), cooldown_frac 0.7, sum-CE
gradients, softcap 15, SDPA scale 0.12, 10.5M-token preloaded val window
all unchanged.

**H15 (reference):** 1k-step val_bpb of this baseline is the mark the
energy port must beat. No keep/discard criterion; it's the control.

**H15 result (2026-07-22, job 254 SUCCEEDED):** final val_bpb
**1.3477 @ 1000** (curve: 1.7458 @ 100, 1.5856 @ 200, 1.5159 @ 300,
1.4781 @ 400, 1.4410 @ 500, 1.4158 @ 600, 1.3958 @ 700, 1.3778 @ 800,
1.3584 @ 900). ~515 ms/step val-inclusive avg; run ~11 min. Note: not
comparable to the family's 2k-protocol numbers (different val window,
schedule, batch — see the fork docstring); the ONLY valid comparison is
job 255 on this same protocol.

### nanogpt-mini + energy head (`nanogpt_mini_energy_train.py`)

Self-contained fork of the mini baseline (files stay standalone, upstream
convention). Trunk/loop/optimizer/data/val verbatim; the swap: input path
becomes `latent_projector(rms_norm(embed(ids)))` (train-mode BN — this is
also what keeps the codebook's running stats alive), head becomes the
family's per-dim distance energy over
`latent_projector.inference(rms_norm(embed.weight))` with ẑ =
`prediction_projector(norm2(trunk))`, straight-through-clamped s_d, b_k,
no softcap. `proj` deleted (head tied to embed by construction).
TokenProjector/FP32BatchNorm1d imported from the family, run under a
local bf16 autocast to reproduce the family's global-autocast numerics.
Projector matrices → Muon; all new ndim<2 → the AdamW scalar group;
embed keeps upstream lr 0.7 (registered risk vs family's 0.05 tied lr).
Params 23.6M vs baseline 20.0M — the +3.6M IS the energy architecture
(two projectors, minus the deleted 0.53M head); capacity and mechanism
are confounded by design here, as in the family. ENERGY_HEAD_FORM
switches distance|dot pending H14's verdict.

**H16:** the energy head transfers to a conventional trunk — the port
beats the mini baseline's softcapped tied-dot… rather, untied-softmax
head at 1k steps ("expect it will perform better if we implement it
correctly" — user). Win = val_bpb below H15's curve at matched steps.
Failure modes to watch: embed lr 0.7 thrashing the codebook (canary:
codebook_norm/scale_max in val lines), BN cold-start in the first ~50
steps.

**H16 verdict (2026-07-22, job 255 SUCCEEDED): TIE at 1k — not a win,
with a strongly convergent trend.** Final 1.3475 vs baseline 1.3477
(−0.0002). Trajectory: −0.005 ahead @ 100, then behind through the
middle (+0.0115 @ 200, +0.0123 @ 300 — the BN/codebook warm-up cost),
then a monotone catch-up (+0.0096 @ 400 → +0.0055 @ 600 → +0.0006 @ 900
→ −0.0002 @ 1000) — it crossed the baseline exactly at the finish.
Canaries clean the whole run: scale_mean drifted 0.044→0.029,
codebook_norm 19.5→36.8, no thrash, no collapse. Costs: +3.6M params
and +21% step time (625 vs 515 ms) — so on equal wall-clock or equal
params the baseline wins at this horizon.

Open questions (would each need a run): (1) does the late slope hold —
i.e., does energy WIN at 2k on this protocol? The delta shrank
monotonically for 700 straight steps, but LR cooldown ends at 1000 and
extrapolating through a schedule change is speculation. (2) embed lr
0.7 vs the family's 0.05 for the tied table — untested here; the first
knob if the port is pushed further. (3) The 2k-protocol family showed
the same shape (energy family competitive-not-dominant vs its own
baselines) — consistent story: the energy head matches softmax heads
but has not yet shown a transfer-scale BPB payoff.

**Energy-port intricacy checklist (user-flagged), to handle in the
variant:** (a) logit softcap — upstream caps at 15; the energy head has
NO softcap (distance logits are shaped by BN geometry + per-dim scale);
the head is replaced wholesale, cap and all. (b) untied embed vs proj —
the energy head deletes `proj` and derives the codebook from `embed` via
the BN projector (tied by construction). (c) init — upstream embeds are
std-1 (vs family's tied_embed_init_std 0.005); BN absorbs input scale,
but the projector sees std-1 inputs at step 0. (d) precision — upstream
trunk runs implicit bf16 through weight casts; BN projector and energy
logits must stay fp32 as in the family. (e) optimizer routing — new
projector matrices → Muon vs Adam decision; flat scalars (per-dim s,
biases) → the ndim<2 AdamW group (lr 0.015) vs family's fused-Adam
group; codebook-side embed keeps lr 0.7?  Register the choices in the
variant's docstring before running. (f) loss reduction — upstream
returns sum-CE (unnormalized grads); the family's forward returns mean —
the port must keep sum semantics or retune nothing.

## Family-Regime Arm on the Port (registered before results)

`nanogpt_mini_energy_famreg_1k`: identical to job 255 except the head
and codebook training regime is restored to the family's — env
`EMBED_LR=0.05 EMBED_WD=0 EMBED_INIT_STD=0.005 HEAD_SCALAR_LR=0.04
HEAD_WD=0` (new knobs in `nanogpt_mini_energy_train.py`; defaults
reproduce job 255 bit-for-bit; `scale_min` added to the val canaries).

Motivation (post-H16 dual review): the parity audit found the head CODE
faithful (10 clean checks) but the training REGIME anchored to the
baseline optimizer: head scalars at lr 0.015 + wd 0.001 (family: 0.04,
wd 0), and embed — which IS the codebook basis — at lr 0.7 (family:
0.05, init std 0.005). Smoking gun: job 249's per-dim scales specialized
to a stable ~2.1× max/min spread by step 120 with scale_mean falling
0.044→0.018, while job 255's stayed near-isotropic (max/mean ~1.24, no
min logged) and its scale_mean REBOUNDED during cooldown — the head we
tested never became the head that won the family arms. The red-team
review counter-argues a structural ceiling: the head is a reparametrized
tied softmax facing an untied+bias baseline, so parity may be the best
possible outcome regardless of regime. This run discriminates the two.

**H17:** with the family regime, the per-dim scales specialize (max/min
≥ 1.8 by step 400, scale_mean falling through cooldown) AND val BPB
separates from the H16 curve. If BPB beats the BASELINE by >0.005 at
1000, the H16 tie was a regime artifact (audit right, red-team wrong) —
then decompose which factor carried it. If scales specialize but BPB
still ties (±0.005), the red-team's structural-ceiling story is
confirmed and the energy question CLOSES at this scale: non-adoption,
budget back to the trunk. If scales still fail to specialize, the
regime hypothesis is refuted and something unmodeled gates
specialization (trunk geometry, batch scale) — B1 territory. Bundled
five-knob change, deliberately: max power on "was the regime the mask";
attribution only if it wins. Risk registered: embed lr 0.05 + init std
0.005 changes INPUT-path embedding dynamics too, not just the codebook
— a loss here is ambiguous between "family regime bad for the trunk"
and "structural ceiling", so only the scale canaries + a WIN are cleanly
interpretable; a big early-loss would mark the bundle too aggressive
for the untied-trunk setting, not close the question.

**H17 verdict (2026-07-22, job 256 SUCCEEDED): TIE by the registered
rule — famreg 1.3457 vs baseline 1.3477 at 1000, a −0.0020 edge, inside
the ±0.005 tie band → the red-team's structural-ceiling reading stands;
the regime was NOT masking a >0.005 win.** But both H17 canary
predictions confirmed: the scales specialized immediately (2.6× max/min
by step 100 vs 1.24 max/mean after 1000 steps in job 255 — the regime
WAS what gated specialization) and the curve shape flipped — famreg
paid its tax EARLY (+0.020 @ 100, the slow 0.05 embed warming up),
matched the old energy run by 200, and finished as the strongest of
the three arms, crossing the baseline at ~900 and ending −0.0020 with
the steepest late slope. scale_mean rebounded during cooldown
(0.019→0.023) as in 255 but from a specialized state; codebook_norm
plateaued ~39. Reading: specialization is worth roughly the +0.002
that separates famreg from 255's tie, not the family-transfer win.
Attribution of the five bundled knobs is moot at this effect size.
The energy question at 20M/V=1024 closes per the registered rule:
non-adoption (the head costs +3.6M params/+21% step time for ≤0.002).
Only open thread: all three arms' late slopes differ, and famreg's is
steepest at the horizon boundary — a 1900-step famreg-vs-baseline pair
(data ceiling ~1992) is the one run that could still overturn
non-adoption; not queued, needs a fresh registration if wanted.

`nanogpt_mini_tieddot_1k`, script `nanogpt_mini_tieddot_train.py`: the
baseline with ONLY the attachment factor changed — untied `proj` head
replaced by a tied attached dot readout, codebook =
`rms_norm(embed.weight)` (no stop-grad), logits = softcap_15(s·(z·c_k)
+ b_k) with scalar s ZERO-init (reproduces the baseline's zero-init-proj
logits-start-at-0 trick) and b_k zero-init. No projectors, no BN, no
distance form, no per-dim scales; softcap and full baseline optimizer
regime KEPT (embed lr 0.7 default; s and b_k fall in the ndim<2 group).
19.43M params — 0.52M CHEAPER than the baseline. User-requested probe:
"purely the attached CE dot, no energy or bn". This is decomposition
probe #2/#1 merged from the post-H16 discussion: H16's tie only bounds
the SUM of the port's factors at ~0; this isolates whether target-side
gradients into the embedding (weight tying through a normalized
codebook) carry value on their own in an untied-baseline setting.

**H18:** classic weight tying at V=1024/d=512 with a hot (0.7) embed lr
is predicted to LOSE to the untied baseline by 0.005–0.02 at 1k: the
table must serve two roles (input geometry at std-1 scale, readout
geometry) with no adapter between them, and the family+famreg evidence
says codebook-serving tables want a gentler regime. A TIE or WIN at
19.43M params would be the interesting outcome — it would say the
baseline's 0.53M untied head buys nothing, freeing that budget for the
trunk under the 16MB cap. Canaries: readout_scale (sign/magnitude
trajectory; |s| stabilizing without the softcap saturating) and
bias_absmax (should track log-unigram range, ~2-4). If it loses at
0.7, a follow-up at EMBED_LR=0.05 EMBED_INIT_STD=0.005 discriminates
"tying is bad here" from "tying needs the family regime" — register
before running.

**H18 verdict (2026-07-22, job 257 SUCCEEDED): REFUTED in the
interesting direction — tieddot finished 1.3451, the BEST of all four
arms (baseline 1.3477, energy 1.3475, famreg 1.3457), at 0.52M FEWER
params than the baseline and ~equal step time (519 vs 511 ms; the
energy arms cost 626-638 ms).** The predicted hot-lr loss appeared only
as a warm-up tax: +0.058 @ 100 (largest of any arm), repaid by 400
(dead even), ahead from 500 on with the steepest late slope, final
−0.0026 — still inside the ±0.005 tie band, so formally a TIE by the
keep rule, not a win. Canaries clean: readout_scale rose smoothly
0→0.097 (≈2·d^-0.5), bias_absmax 3.45 (log-unigram range as predicted).
Reading, combining H16-H18: the entire energy apparatus (projectors,
BN, distance form, per-dim scales) nets ≤0.002 over NAKED attached
tying, which itself matches-or-slightly-beats the untied baseline at
negative param cost — attachment (dense target-side gradients into the
table) is the only component of the family design that survives
decomposition in this setting, and everything else is an (expensive)
adapter that mainly buys back the warm-up tax it doesn't need at a
gentle regime. Open follow-ups (not queued, register before running):
(1) tieddot at the family embed regime (0.05/0.005) — can the warm-up
tax be removed and the tie turned into a >0.005 keep? (2) the freed
0.52M params spent on trunk — the real 16MB-cap question; (3) 1900-step
tieddot-vs-baseline pair — all arms' late slopes differ and tieddot's
is steepest.

## 4k Horizon Pair (registered before results)

`nanogpt_mini_4k` and `nanogpt_mini_tieddot_4k`: the H18 pair rerun at
ITERATIONS=4000, everything else identical (default regimes, val every
20). User-directed horizon extension of follow-up (3). Data caveat,
registered: the one-pass slice holds ~1992 batches, so both runs cycle
the shards and see each training token ~2.0x — a two-epoch regime, not
one-pass; identical for both arms, so the comparison is internally
valid but neither number is comparable to the 1k arms (different
schedule: constant LR to 1200, decay to 4000) nor to one-pass runs.

**H19:** tieddot's steeper late slope is horizon-real, not a cooldown
artifact: tieddot beats the baseline at 4000 by >0.005 (a formal keep
at negative param cost). Mechanism: the warm-up tax (~400 steps) is
fixed-cost and amortizes, while the dense target-side gradient
advantage and the freed capacity compound. Refuted if the baseline
wins or |Δ| ≤ 0.005; the repeated-data regime is the registered risk —
if the untied head's extra 0.53M params turn into a second-epoch
memorization edge, that shows up as the baseline gaining specifically
after step ~2000 (first shard revisit), which the curves can localize.

## Tieddot Family-Regime Arm (registered before results)

`nanogpt_mini_tieddot_famreg_1k`: job 257's tieddot with the embed
regime swapped to the family's — `EMBED_LR=0.05 EMBED_WD=0
EMBED_INIT_STD=0.005` (knobs already in the script; head scalars stay
in the 0.015 group — tieddot has no HEAD_SCALAR_LR knob and its head
is 1 scalar + 1024 biases, so the embed table is the only lever).
Follow-up (1) from the H18 verdict. Rationale: tieddot's only deficit
vs the baseline was the 400-step warm-up tax (+0.058 @ 100), the exact
signature famreg (job 256) showed the gentle regime addresses in the
attached-table setting; its post-warm-up slope was the best of all
arms.

**H20:** the gentle regime removes most of the warm-up tax without
blunting the late slope: ahead of or even with the baseline by step
200 (vs 400 for job 257), final ≤ 1.3427 (i.e., beats the baseline
1.3477 by >0.005 — a formal keep at negative param cost). Registered
risk (famreg's lesson cuts both ways): famreg paid its OWN early tax
(+0.020 @ 100) from the slow input-path embedding, so the tax may
MOVE rather than vanish; refuted if final delta vs baseline stays
inside ±0.005, in which case tieddot's value remains "free params,
no BPB cost" and the respend arm (follow-up 2) becomes the whole
story.

**H20 verdict (2026-07-22, job 260 SUCCEEDED): REFUTED — the gentle
regime made tieddot strictly worse at every milestone.** Final 1.3474
(vs default tieddot 1.3451, baseline 1.3477): the warm-up tax got
DEEPER, not smaller (+0.073 vs baseline @ 100 against +0.058 at lr
0.7), the deficit vs the default sibling held ~0.002-0.005 through the
whole run, and the finish is a plain tie with the baseline instead of
the default arm's −0.0026. Reading: the family regime's value in the
energy port came from letting the per-dim scales specialize against a
slow-moving codebook; tieddot has no per-dim scales to protect, and
the naked table simply wants the baseline's hot lr to reorganize
quickly. Default (0.7/0.001/std-1) is tieddot's regime. The respend
arm (H21) therefore runs at the DEFAULT regime.

## Tieddot Respend Arm (registered before results)

`nanogpt_mini_tieddot_wide_1k`: default-regime tieddot with
MLP_HDIM=2112 (new env knob, default 2048 preserves all prior runs) —
respends the 0.52M params freed by deleting the untied head into trunk
MLP width (6 blocks x 2x512x64 = +0.39M -> 19.83M total, still 0.13M
under the untied baseline's 19.96M). This is follow-up (2), the 16MB-
cap question: does tied-head-plus-bigger-trunk beat the untied
baseline at equal-or-fewer params?

**H21:** yes — tieddot alone already edged the baseline (−0.0026), and
+2.4M FLOPs-worth of MLP width per token has never been free-tested in
this lineage; predict final delta vs baseline in −0.004 to −0.010,
i.e., at least holding tieddot's edge and plausibly crossing the
formal 0.005 keep bar. Refuted if the wide arm does not beat plain
tieddot (would say the freed params don't convert to trunk value at
this scale/horizon; hdim 2112 breaks the clean 4x ratio and 64-mult
alignment is the only concession to kernels). Step time is part of
the verdict: expect ~+2-3% vs tieddot's 519 ms.

**H21 verdict (2026-07-22, job 261 SUCCEEDED): SUPPORTED, just under
the formal bar.** Final 1.3434 — beats the baseline by 0.0043 (bottom
of the predicted −0.004..−0.010 band, short of the 0.005 keep bar) and
plain tieddot by 0.0017, at 19.83M params (0.13M under baseline) and
524 ms/step (+1% vs tieddot, +2.5% vs baseline). Best arm of the
six-run mini program. The freed head params DO convert to trunk value
(wide > plain tieddot at every milestone from 200 on, bar a step-400
noise blip). Mini-program final standings @1k: wide-tieddot 1.3434 <
tieddot 1.3451 < famreg-energy 1.3457 < famreg-tieddot 1.3474 <
energy 1.3475 < baseline 1.3477. Caveats: single seed each, effects
2-4x the ~0.001-0.002 val jitter but below the keep bar; upstream
warns 1k-step reads correlate weakly with long-horizon finals. Next:
transfer the tied attached dot head to the challenge lineage
(train_gpt.py fork) and test on the standard 2k protocol vs
baseline_2k — the mini sandbox has done its job.

## Challenge-Lineage: PowerCool Tail (registered before results)

`powercool_2k`, script `powercool_train_gpt.py` (full-file fork of
train_gpt.py; `diff train_gpt.py powercool_train_gpt.py` shows exactly
the schedule hunk + knob + docstring): the linear warmdown
`scale = remaining/warmdown` becomes `scale = (remaining/warmdown)^p`,
p = POWERCOOL_P (default 1.2), applied in both the step-based and
wallclock-based branches. Source: modded-nanogpt track-3 record #46
("PowerCool", lr ∝ (end−step)^1.2), a wallclock-free technique — zero
per-step cost, transfers cleanly to the 10-min-capped challenge.
Everything else identical to baseline; no RNG or init change (schedule
only). Reference: baseline_2k = 1.2967 @ 2000; protocol iterations
2000, warmdown 1200 (decay from step 800).

**H23:** the concave-down power tail (lower LR through the whole
decay phase, much lower late) improves final BPB by spending more of
the tail in a low-noise regime near the endpoint; predict −0.003 to
−0.010 at 2000; keep if ≥0.005. Counterargument: #46's gain came as
part of a package (SOAP-Muon, radial constraints, EMA readout) at a
3.5× longer relative tail; alone at warmdown 1200/2000 the lower
mid-tail LR may simply mean less progress, showing as +0.002..+0.005.
Mid-run reads will look WORSE by construction (lower LR earlier in the
tail → less hot progress at matched steps); only step 2000 decides.
Single seed, standard 2k protocol, val every 20.

**H23 outcome (2026-07-22): final 1.3034; verdict PENDING the
baseline_2k_fresh anchor (job 265), but trending REFUTED.** Against
tied_bias (same environment, bias ≈ null once re-anchored) it is
+0.0022 worse. Curve shape inverts the naive read: powercool LED at
matched steps mid-tail (1.3232 vs 1.3250 @1600, 1.3104 vs 1.3109
@1800 — cooler LR sits closer to the descent path), then the linear
run's hotter final 200 steps closed harder (−0.0097 vs −0.0070). The
registered counterargument (power tail alone = less area under the LR
curve = less progress at this short horizon) is the operative story;
#46 ran it with SOAP-Muon + EMA readout at a much longer relative
tail. Final verdict number vs job 265 when it lands.

## Mini-vs-Challenge Head-to-Head at 2k (registered before results)

User direction: stop polishing the challenge baseline; make the
nanogpt-mini lineage BEAT it on the 2k protocol ("we have all the
components; get the learning rate/schedule right"). Blocker found and
fixed first: mini's val_bpb evaluated a 10.5M-token val PREFIX vs the
challenge's full 62.02M-token split — cross-lineage comparisons were
invalid (mini's own docstring warns this). Both mini scripts gain a
VAL_TOKENS env knob (default unchanged); runs below use
VAL_TOKENS=61,997,056 = the challenge window rounded down to mini's
64×1024 microbatch (99.96% identical, ≤~0.0002 BPB skew), same val
shard (md5-identical across datasets), VAL_LOSS_EVERY=200.

`nanogpt_mini_fullval_2k` (job TBD): mini baseline, ITERATIONS=2000,
native schedule (cooldown_frac 0.7 → constant to 600, decay to 2000),
native LRs (embed .7 / muon .025 / scalars .015, wd .001), 524,288
tokens/step (SAME as challenge — matched-step is matched-data).
`nanogpt_mini_tieddot_fullval_2k` (job TBD): same, tieddot head.
Anchor: challenge tied_bias_2k = 1.3012 @ 2000 (full-val window).
Caveat: mini's onepass train slice wraps at ~1992 steps (~8 repeated
batches at the very end — negligible). Mini 1k full-val number does
NOT exist; the old 1.3477 was window-val and must not be quoted
against these.

**H27:** the mini recipe at 2k with full-val eval beats the challenge
anchor by >0.005 (prediction: mini lands 1.28-1.30; the track-3 tuned
recipe is stronger per token than the challenge baseline). If it wins,
it becomes the base lineage for the challenge submission and the
attached-target line builds on it (see Program Note below). If it
LOSES, the gap is schedule/LR-horizon mistuning (mini LRs were tuned
for ~3k-step horizons at staged batch sizes) — LR sweep on the mini
side is the registered follow-up, per user intent.

**H27 VERDICT (job 269, 2026-07-22): CONFIRMED, but narrowly.**
`nanogpt_mini_fullval_2k` = **1.2856 @ 2000** (returncode 0). Margin
−0.0156 vs tied_bias_2k 1.3012 (current-code anchor), −0.0111 vs the
stale April baseline_2k 1.2967. Within predicted range and above the
0.005 bar → the mini lineage is the challenge-submission base. BUT the
curve shape is a red flag (user observation): mini led by +0.15 @200
and +0.02 @1000, then surrendered most of it in the last 600 steps —
the challenge baseline's hard linear tail out-converges the mini's
cooldown_frac-0.7 trapezoid. The mini's ENDGAME is mistuned for the 2k
horizon; per-step advantage is real, schedule tail is the weakness.
This is exactly the phase the June-19 record system targets (per-group
power tails + mu cooldown + tail-EMA) → H28 (job 275) is the direct
answer; if it disappoints, sweep cooldown_frac/embed-LR on the mini.
NEW mini reference for all H28+ comparisons: **1.2856 @ 2000 full-val**.

**Pair completion (job 270, 2026-07-22):** `nanogpt_mini_tieddot_fullval_2k`
= **1.2817 @ 2000** (returncode 0) — tieddot beats the plain mini by
0.0039 at 2k full-val (same direction as the 1k window-val result,
+0.0026, now at a longer horizon and clean protocol; still under the
0.005 keep bar on its own). ATTACHED-TARGET 2k REFERENCE: 1.2817 —
supersedes the 1.3451 @ 1k window-val number in the Program Note for
all future head ablations. Also beats the challenge anchor 1.3012 by
0.0195, so the tieddot mini is currently the best challenge-protocol
result in the repo.

## Mini-Lineage: June-19 Record Port (registered before results)

`nanogpt_mini_cwd_2k`, script `nanogpt_mini_cwd_train.py`: per user
direction ("fully aligned with the June 19th version, except for our
parameter changes"), the mini's tuned-baseline training system is
replaced WHOLESALE by record #46's
(`20260619_cwd_rowfloor_tailema/train_gpt_cwd_SOTA.py`): record LRs
(embed 0.3, proj 1/320, aux 0.01 in three beta groups (0.8,.99)/
(0.8,.997)/(0.8,.9965), Muon 0.0375), bias-correction-free aux Adam,
full record Muon (SOAP-f1 on all hidden matrices β2 .90 power .5,
attn.proj trust gate floor .45/cap .85, radial outward .5, per-row u/w
floor .3825, radius pin, cautious WD .025, gram-normalized 12-iter NS),
EMA-Nesterov (0.3 scheduled / ema .99), tail-EMA readout (λ .6, embed
excluded; `val_bpb` = blend once active, `val_raw_bpb` alongside),
record init (torch defaults + proj zero + depth-scaled fc α.30 + CGI
gains split α.125, pair-from scaled 6/12→3/6), NaN guard. All step
constants re-expressed as FRACTIONS of the record's t_end 2900 and
resolved against ITERATIONS=2000: Adam power-tail crossover ≈1025,
Muon ≈354 (power 1.2, power_c = lr/(t_end−start)^1.2 — same crossover
semantics as the record's hardcoded constants), mu warmup 207/cooldown
138, trust floor end 948/fade 1121, Nesterov prefill 207/rest 1345,
tail-EMA tau 103 start 1655 end 2000. Kept from mini: 6L×512d vocab
1024, sp1024 data + cycle loader, byte-LUT BPB val with VAL_TOKENS,
524,288 tokens/step, single-GPU shim. NOTE: this port supersedes the
"no expensive per-step machinery" constraint for SOAP — the user's
June-19-alignment directive explicitly includes the full system.

**H28:** the record's training system on the mini model beats the
mini's native recipe at matched 2000 steps (compare vs
`nanogpt_mini_fullval_2k`, full-val window) — prediction: −0.01 to
−0.03 BPB; the record's system is worth ~0.006 at 2890 steps on the
12-layer model AND its LR/schedule structure (per-group power tails,
higher Muon lr) is closer to a tuned 2k-horizon recipe than the mini's
0.7-cooldown trapezoid. Counterarguments: (1) every constant was tuned
at 12L×768d vocab 50k — embed lr 0.3 vs mini's tuned 0.7 on a 1024-row
table may be badly undertrained; (2) horizon fraction-scaling of
schedule constants is an assumption, not a result; (3) SOAP eigh/QR on
≤2048² matrices adds step time (acceptable per user: matched-STEP BPB
is the primary metric, but wallclock is logged and still matters).
Step time will be watched at the first val; if the run wins, per-lever
decomposition and an embed-LR probe are the registered follow-ups.

**H28 VERDICT (job 275, 2026-07-22): CONFIRMED.**
`nanogpt_mini_cwd_2k` = **1.2692 @ 2000** (returncode 0, ~730ms/step =
+24% vs mini's 588ms; final val is the tail-EMA blend readout). Margins:
−0.0164 vs mini 1.2856, −0.0125 vs tieddot 1.2817, −0.0320 vs challenge
anchor 1.3012 — new best challenge-protocol result; in predicted range
(−0.01 to −0.03). The record's endgame (per-group power tails + mu
cooldown + tail-EMA) fixed exactly the mini's weak phase (it was 1.2735
@1800 → 1.2692 @2000, still converging hard at the end). The
horizon-fraction rescale assumption survived contact. NEW BASE for the
mini lineage: **1.2692**. Registered follow-ups now queued: H31 (SOAP=0
decomposition), H32 (EMBED_LR 0.7), H33 (tieddot head on the port).

## Mini-Lineage: GPT-2 Vocab Probe (registered before results)

`nanogpt_mini_gpt2vocab_2k`, script `nanogpt_mini_gpt2vocab_train.py`
(user request: "a run with Nano GPT's vocab and embedder"): the mini
recipe VERBATIM (same trunk 6L×512d, same optimizers/LRs/schedule/init)
with vocab 50,304 GPT-2 BPE, trained on modded-nanogpt's fineweb10B
GPT-2 shards (symlinked as data/datasets/fineweb10B_gpt2; 11 chunks +
val downloaded). BPB via data/tokenizers/gpt2_byte_lut.pt (exact
per-token byte lengths; EOT counted 1 byte, <0.1% of denominator).
VAL_TOKENS=33,554,432 ≈ 149M bytes at 4.44 bytes/token — byte-matched
to the sp1024 full-val window. MBS=8 (50k-logit memory). DIAGNOSTIC
ONLY: 51.5M embed+head params is over the 16MB budget by itself; the
actionable follow-up on a win is intermediate vocab (sp8192 shards
exist; challenge SOTA uses 8192). Caveats: matched-step = ~1.7× more
text BYTES per step (inherent to the vocab change); embed lr 0.7 was
tuned for dense 1024-row updates, 50k rows are ~50× sparser (EMBED_LR/
PROJ_LR knobs registered for the follow-up probe); different val text
window (same FineWeb val distribution, byte-matched size).

**H29:** with nanogpt's native vocab the mini recipe's BPB at 2000
steps improves substantially over the sp1024 mini (1.2856) — vocab
1024 forces ~1.7× more predictions per byte and each carries lower
per-token entropy leverage; prediction 1.15-1.25 if vocab is a main
bottleneck, ≈1.28+ if not. Either way this calibrates how much of the
gap to nanogpt-repo numbers is tokenization rather than recipe.

**H29 VERDICT (job 276, 2026-07-22): CONFIRMED — vocab is a major
bottleneck.** `nanogpt_mini_gpt2vocab_2k` = **1.1734 @ 2000**
(returncode 0), −0.112 vs sp1024 mini 1.2856. Matched-BYTES correction
(sp1024 bytes/token measured 2.4076, GPT-2 4.444 → sp1024's 2k-step
byte total ≈ GPT-2 step 1084; curve interp 1000→1200) gives ≈**1.233**,
still −0.053 — and conservative, since the GPT-2 run is mid-decay at
1084. Remaining confounds: 71M vs 19.4M params, ~1.85× context bytes.
Conclusion: tokenization+capacity worth ≈0.05-0.11 BPB; the 16MB-
compatible question is INTERMEDIATE vocab → sp8192 probe (H34).

## Mini-Lineage: sp8192 Vocab Probe (registered before results)

`nanogpt_mini_sp8192_2k`: mini recipe verbatim via new env knobs on
`nanogpt_mini_train.py` (VOCAB_SIZE, defaults preserve 1024 semantics)
with vocab 8192, fineweb10B_sp8192 shards (2B train tokens, 40.5M val)
+ fineweb_8192_bpe tokenizer. VAL_TOKENS=40,501,248 (largest
65536-multiple in the val shard; ~134M bytes at ~3.3 bytes/token —
near the ~150M byte windows of the other runs). Untied 8192 head:
params ≈ 19.96M − 2·0.52M + 2·4.19M ≈ 27.3M → over 16MB at int8 as-is;
this probe is still DIAGNOSTIC (calibrates the vocab axis at a size the
challenge SOTA (8192) proved submittable with rebalancing — tied head
and/or trunk shrink is the design work if it wins).
**H34:** sp8192 lands between the sp1024 mini (1.2856) and the
matched-bytes GPT-2 number (~1.23): prediction 1.24-1.27 raw @ 2000.
If ≤1.26, the vocab axis dominates every recipe lever measured so far
and param rebalancing for 8192 becomes the top design priority.

## Mini-Lineage: Machinery-Free Energy Head (registered before results)

`nanogpt_mini_tiedenergy_fullval_2k`, script
`nanogpt_mini_tiedenergy_train.py` (user direction: "energy only based
on NanoGPT Mini" — the energy score WITHOUT the lejepa machinery whose
cost/regularization, not the energy form itself, plausibly sank the old
line): TiedEnergyGPT = logits_k = s·(−½‖norm2(trunk) − e_k‖²) + b_k,
RAW attached codebook (no rms_norm — normalized distance collapses to
tieddot's dot product up to per-row constants; raw rows give a free
learned norm prior), capless (distance logits one-sided; capless CE is
per-position shift-invariant so the ‖z‖² term is harmless), fp32
scoring (energies O(dim); bf16 quantum at that magnitude destroys
cross-token differences), scalar zero-init scale + zero per-token bias
(baseline's uniform-start trick; at s=0 nothing flows until s moves —
verified on CPU: loss=ln V exactly, scale grad ≈ −235, embed grad 0).
Everything else = tieddot script verbatim (seed/RNG parity, optimizer
groups, schedule, data, val). Cost ≈ +3% trunk FLOPs (one fp32 matmul).
Fills the untested cell: attachment ✓ (tieddot won), machinery ✗ (old
port nulled at 1.3475 ≈ baseline 1.3477 @1k), score form = energy.

**H30:** the energy form adds nothing over the dot form once the
machinery is gone — prediction: within ±0.003 of tieddot (2k full-val
1.2817); the free per-row norm prior is redundant with b_k. Worth
running because (1) user directive, (2) cheap, (3) if it BEATS tieddot
by >0.005 the norm-prior/metric interpretation matters and per-dim
scales (diagonal Mahalanobis) become the registered follow-up. Failure
modes to watch: scale sign flip, capless logit blowup with embed lr
0.7 (telemetry: readout_scale + bias_absmax printed at each val).

**H30 VERDICT (job 291, 2026-07-22): REFUTED — energy form is WORSE.**
`nanogpt_mini_tiedenergy_fullval_2k` = **1.2961 @ 2000** (returncode 0):
+0.0144 vs tieddot 1.2817, +0.0105 vs even the plain untied mini
1.2856. Outside the predicted ±0.003 band, in the bad direction. The
machinery-free distance head is the worst of the three head forms; the
old projector-energy port's null (≈baseline @1k) now looks like the
projectors COMPENSATING for a bad score form rather than dead weight.
Plausible mechanisms (undecomposed): free per-row codebook norms under
embed lr 0.7/wd 0.001 make the effective per-token slope drift
(‖e_k‖² enters BOTH slope and bias of the equivalent dot form); capless
logits lose the softcap's regularization. Energy-head line CLOSED for
the challenge unless a specific mechanism hypothesis emerges; the
attached-head lineage continues on the DOT form (tieddot, H33).

## Mini-Lineage: Record-Port Decomposition Probes (registered before results)

Base for both: `nanogpt_mini_cwd_train.py`, anchor = cwd port **1.2692**
@ 2000 full-val (job 275). Env knobs default to record values, so the
reference semantics are unchanged.

`nanogpt_mini_cwd_nosoap_2k` (SOAP=0): everything in the record system
EXCEPT SOAP-f1 (and the attn trust gate, which only gates SOAP output) —
plain nesterov momentum into NS, keeping radial/rowfloor/pin/CWD,
per-group power tails, mu schedule, EMA-Nesterov, tail-EMA, record LRs
and init. Directly answers "is SOAP-f1 necessary" (user question):
SOAP is ~all of the +24% step-time cost.
**H31:** SOAP-f1 carries a minority of the win — no-SOAP lands within
0.008 of 1.2692 (i.e. ≥half the 0.0164 margin survives at ~0% overhead).
Counterargument: the record found trust-gating necessary for attn.proj
early — without SOAP the geometry constants (TARGET_UW, CWD) are
off-tune (H26 caveats apply). If no-SOAP holds ≥1.278, the cheap system
becomes the default base for head ablations (faster iteration).

**H31 VERDICT (job 292, 2026-07-22): NARROWLY REFUTED — SOAP carries
~half the win.** `nanogpt_mini_cwd_nosoap_2k` = **1.2780 @ 2000**
(returncode 0, 541ms/step vs full port's 730ms — SOAP was all the
overhead and then some; no-SOAP is even faster than the mini's 588ms).
Gap to full port: +0.0088 (bar was 0.008). Decomposition: of the
port's 0.0164 win over the mini recipe, the cheap levers (geometry +
power tails + mu schedule + EMA-Nesterov + tail-EMA + record LRs)
retain 0.0076 (46%); SOAP-f1 contributes 0.0088 (54%) for +35% step
time. Matched-STEP verdict: keep SOAP (primary metric per user). OPEN
QUESTION for the submission (wallclock-scored, <10min on 8xH100):
no-SOAP affords ~1.35x the steps at equal time on the 5090 — whether
no-SOAP @ ~2700 steps beats full @ 2000 is a matched-TIME follow-up,
and the eager-QR overhead ratio on H100s differs from the 5090. Also:
no-SOAP @ 1.2780 still beats every non-port run (mini 1.2856, tieddot
1.2817) — the cheap system is a valid fast-iteration base.

`nanogpt_mini_cwd_embedlr07_2k` (EMBED_LR=0.7): record embed lr 0.3 was
tuned on a 50k-row table (sparse per-row updates); our 1024-row table
gets dense updates and the mini's separately-tuned optimum was 0.7.
power_c derives from initial_lr at runtime, so the power tail scales
consistently.
**H32:** embed lr 0.7 improves the port by >0.005. Counterargument: the
record's EMA-Nesterov lookahead is scheduled by the EMBED group's lr
ratio — raising embed lr changes the lookahead schedule only via ratio
(shape identical), but tail interactions are untested; and 0.7 was tuned
WITH wd 0.001 + betas (0.8,0.95), not the record's (0.8,0.99) wd 0.

**H32 VERDICT (job 293, 2026-07-22): DIRECTION CONFIRMED, BELOW BAR.**
`nanogpt_mini_cwd_embedlr07_2k` = **1.2670 @ 2000** (returncode 0):
−0.0022 vs the port's 1.2692. Right direction (denser 1024-row table
does want more embed LR under the record system) but under both the
0.005 hypothesis bar and the keep bar. The slope is shallow (+0.4 lr →
−0.002), suggesting a near-flat optimum between 0.3 and 0.7 — a finer
sweep is LOW priority vs the remaining axes. Standing note: any future
"best" config should carry EMBED_LR=0.7 only if it survives a paired
confirmation at that config; treat 1.2670 as suggestive, not a new
base (sub-bar deltas are within run-to-run noise until replicated).

## Mini-Lineage: Tieddot Head on the Record Port (registered before results)

`nanogpt_mini_cwd_tieddot_2k`, script `nanogpt_mini_cwd_tieddot_train.py`
(fork of the cwd port; head-local deltas only — see docstring). Combines
the two winners: record training system (1.2692) x attached tied dot
readout (+0.0039 on the mini recipe). Readout scale/bias route to the
other-aux Adam group (lr .01, betas (0.8,.997)); embed excluded from
tail-EMA as before, readout params included; params DROP by 0.53M (proj
deleted) — also relieves the 16MB artifact budget.
**H33:** the improvements compose — prediction 1.262-1.267 (tieddot's
delta carries, possibly attenuated: the record system already optimizes
the proj head harder than the mini recipe did, and embed lr 0.3 (vs
mini 0.7) may weaken the attached codebook's adaptation). Keep if
>0.005 vs 1.2692; even null is informative (head choice decouples from
training system). Watch readout_scale telemetry for softcap saturation
under the record's aux betas.

## Program Note: Attached-Target Line Continues (user direction, 2026-07-22)

The attached-target (tieddot) direction is to be REVISITED, not closed:
future head iterations are built on top of `nanogpt_mini_tieddot_train.py`
and ablated against `nanogpt_mini_tieddot_1k` = 1.3451 @ 1k as the mini
reference (the best param-lean arm; wide-tieddot excluded per user —
no param increase). I.e., the question shifts from "does attachment
beat the baseline head" (answered: yes, slightly) to "what improves the
attached head itself." Candidate axes for registration when this
resumes: bias/scale init and LR, codebook normalization variants,
softcap value, embed-LR interaction (mini optimum 0.7), and whether
the challenge transfer (tieddot_2k, H25) corroborates at 2k.

## Challenge-Lineage: Muon Geometry Pack (registered before results)

`muon_geom_2k`, script `muon_geom_train_gpt.py` (full-file fork): the
CHEAP tier of record #46's Muon levers, grafted into the challenge
Muon's apply loop in the record's exact order — radial split-scale
(outward 0.5), per-row u/w floor (target 0.3825, rho 1.0), first-order
radius pin, cautious WD 0.025 (mask pre-update, applied post-pin).
Constants verbatim from the record (u/w-ratio-based → scale-free;
their Muon lr 0.0375/mu 0.95 ≈ ours 0.04/0.95). SOAP-f1 deliberately
EXCLUDED per user constraint (no expensive per-step machinery: its
freq-1 eigh/QR on 1536² and 1024² matrices ≈ +30-50% step time).
Registered as a BUNDLE with per-lever env knobs (MUON_TARGET_UW,
MUON_ROWFLOOR, MUON_CWD, MUON_RADIAL_OUTWARD/INWARD) so a win can be
decomposed without new code. Anchor: tied_bias_2k = 1.3012.

**H26:** the norm-control system (damped outward drift + row-level
update floor + radius pin + cautious decay) improves matched-step BPB
by −0.003 to −0.010 at 2000; keep if ≥0.005 vs anchor. Counterarguments:
(1) all three levers were validated ON TOP of SOAP-f1 + EMA-Nesterov at
a 2890-step horizon — without that base the geometry may not transfer;
(2) TARGET_UW 0.3825 was tuned against their NS variant (12-iter
triple-coeff) vs ours (5-step quintic) — update norms post-NS differ,
so the floor may bind differently; (3) CWD adds decay where the
challenge had none. A bundle loss kills all levers at these constants;
a bundle win triggers knob decomposition. Single seed, 2k protocol.
Review (clean; order/constants verified verbatim vs record, lr=0 tail
is a proven no-op, DDP-equivalent) adds two caveats: (i) the challenge
fuses qkv into one [1536,512] matrix, so the radial split-scale and
radius pin treat q/k/v as ONE radial object with one pinned radius —
the record ran them per-projection; attention-side effects reflect the
fused variant; (ii) CWD strength and the radius delta scale with lr —
at MATRIX_LR 0.04 vs the record's 0.0375 those two levers run ~6.7%
hotter than where they were tuned (the floor and radial direction are
genuinely scale-free).

## Challenge-Lineage: Full Tieddot Head (registered before results)

`tieddot_2k`, script `tieddot_train_gpt.py` (import-and-patch fork like
tied_bias): the full mini winning head transferred, not just its bias —
`logits = softcap·tanh((s·(x @ rms_norm(tok_emb).T) + b)/softcap)`.
Codebook rows rms-normalized (norms stop carrying frequency), zero-init
scalar gain s, zero-init per-token bias b (the frequency carrier).
+1025 params; logits exactly 0 at init; no RNG consumed; s and b under
blocks[-1] → scalar Adam group. User direction: pursue the plain
tieddot head (param-lean), not the wide respend. Judged vs
baseline_2k_fresh (job 265), NOT the stale 1.2967.

**H25:** the tieddot parametrization (unit-norm codebook + learned
gain + bias) transfers its mini advantage (−0.0026 vs mini baseline
@1k) to the challenge lineage; predict −0.002 to −0.006 at 2000; keep
if ≥0.005. Counterarguments: (1) H22's bias-only arm showed no benefit
over the (properly re-anchored) baseline trajectory — the remaining
delta here is the normalized-codebook + gain geometry, worth at most
~0.002 in mini decompositions; (2) the challenge's TIED_EMBED_LR=0.05
was tuned for the raw-table head; with a normalized codebook the
embedding may want a hotter LR (mini's optimum was 0.7) — if the
result is null, a TIED_EMBED_LR probe is the registered follow-up
before discarding. Step-0 val will differ from baseline (logits start
at 0 → uniform), by design. Single seed, standard 2k protocol.

## Challenge-Lineage: Tail-EMA Readout (registered before results)

`ema_2k`, script `ema_train_gpt.py` (full-file fork; diff = 4 hunks):
from warmdown onset (step 800 on the 2k protocol; EMA_START knob) an
fp32 shadow tracks the weights with decay EMA_BETA (default 0.99, ~100
step horizon); every val swaps the EMA in and back out; the FINAL step
leaves it in, so final_model.pt, the int8+zlib artifact, and the
round-trip eval all ship the averaged model. Training dynamics
untouched — raw weights get all updates; the change is readout-only.
Source: track-3 record #46 ("tail-EMA readout"). Per-step cost ~1-2ms
in the tail only (weighted fp32 add over 19.4M params). Reference:
baseline_2k = 1.2967 @ 2000.

**H24:** Polyak-style tail averaging removes SGD noise the linear
decay hasn't yet annealed, predict −0.002 to −0.008 at 2000; keep if
≥0.005. Counterargument: a linear-to-ZERO tail already implicitly
averages (late steps are tiny), so the marginal gain of explicit
averaging may be <0.005 — #46 ran EMA on top of PowerCool, whose
faster-early tail leaves more noise for the EMA to absorb; if both
H23 and H24 land near-null alone, the combination is the registered
follow-up. Mid-tail matched-step reads are apples-to-oranges (EMA lags
the live weights by ~100 steps early in the tail); only 2000 decides.
Single seed, standard 2k protocol, val every 20. Review (clean, no
invalidating bugs; training weights verified bit-identical to baseline
— swap fully undone before each opt.step, shadow never reads its own
readout) adds two caveats: (i) every tail val logs the EMA readout, so
this run's tail curve has no raw-weight reading — deltas come from
comparing against baseline/powercool raw curves; (ii) the EMA update
runs inside the timed region — negligible at 2k with the cap off, but
on the wallclock-capped 8xH100 final it would shave a few steps; flag
before scaling.

**H24 outcome (2026-07-22): NOT RUN TO VERDICT — user-cancelled at
~step 1500 to advance the queue.** Partial reads: EMA readout led the
raw-weight sibling trajectories at every matched mid-tail step —
1.3606 vs ~1.384 @1000 (+0.024 lead), 1.3401 vs ~1.358 @1200 (+0.018),
1.3246 vs ~1.339 @1400 (+0.014) — with the lead shrinking as LR decay
converged the raw weights toward their own average, exactly the
registered dynamic. Whether any lead survived to 2000 is UNKNOWN.
H24 is unanswered, not refuted; the mid-tail evidence is promising
enough that a rerun (or a mini-lineage port) is a reasonable revisit
if readout tricks come back into scope.

## Challenge-Lineage Transfer: Tied-Head Bias (registered before results)

`tied_bias_2k`, script `tied_bias_train_gpt.py` (import-and-patch fork;
train_gpt.py untouched): the challenge baseline's tied readout —
`softcap·tanh(x @ tok_emb.T / softcap)`, which already HAS the
attachment property the mini program isolated — gains the one component
it lacks from the winning mini head: a per-token bias inside the cap
(1024 params, zero-init, registered under blocks[-1] so the optimizer
split routes it to the scalar Adam group with an fp32 master; parent
params init bit-identically). Reference: `baseline_2k` = 1.2967 @ 2000
(tb_logs/baseline_2k).

**H22:** small positive — the bias gives unigram log-odds a direct
additive path instead of coupling them to embedding row norms, paying
off mostly in early steps; predict −0.002 to −0.008 at 2000, keep only
if ≥0.005. Registered counterargument (why this may be null here when
it mattered in mini): the mini codebook was rms-normalized, so the
bias was the ONLY frequency channel; the challenge head's raw tied
table has free row norms — and the input path rms-normalizes per
token, so norm-encoded frequency has no role conflict. Null (±0.005)
would confirm norm-freedom already covers it; a win says the additive
path is cheaper for Adam at lr 0.05 than norm growth. Single seed,
standard 2k protocol, val every 20. Review note (pre-result): the bias
trains under SCALAR_LR — tuned for small control scalars, not a
unigram-scale (|b|~3.4) table — so a NULL here is ambiguous between
"norm freedom covers it" and "bias underfit at this lr"; one bias-lr
probe adjudicates before discarding. Fork reviewed clean (routing,
fp32 master, RNG parity, compile/DDP, quant round-trip all confirmed).

**H22 AMENDMENT (2026-07-22, supersedes the interpretation below):**
the powercool_2k run (bit-identical to baseline until step 800) read
1.5405 @ 400 vs the baseline_2k reference's 1.5159 — a +0.025 offset
at a step where the fork cannot differ. tied_bias read +0.024 at the
same step. The reference is STALE: baseline_2k was recorded 2026-04-12,
train_gpt.py's Q/K/V fusion landed 2026-04-15 (e5b90a7; different Muon
geometry → different trajectory), and until 2026-07-19 ablation.py
forced WARMDOWN_ITERS=0 (flat LR) while the current protocol decays
from step 800 (the half-finished baseline_matched_2k of 2026-04-18 was
evidently a re-anchor attempt after the fusion). Consequences: (1) the
"early transient" narrative below is WRONG — the +0.024 @ 400 was
reference offset, not bias adaptation (proven by powercool showing the
identical offset with a schedule-only fork); (2) the −0.0045 verdict
is UNANCHORED. Job 265 (baseline_2k_fresh) was queued to re-anchor but
CANCELLED per user direction ("You don't need to redo April's").
Working anchor for current-code runs is therefore tied_bias_2k =
1.3012 @ 2000 — a baseline-plus-bias trajectory whose pre-tail curve
matched schedule-identical siblings to ~0.0005, used on the assumption
the 1025-param zero-init bias is ≈null; any verdict within ~0.002 of
this anchor is ambiguous by construction. The April 1.2967 remains
useful only as a cross-era curiosity (pre-fusion code, flat LR); the
QKV-fusion-regression question stays open and, if ever tested, should
be tested as an UNFUSED fork vs a current-code run, not by rerunning
April. Original (misanchored) verdict kept below for the record.

**H22 outcome (2026-07-22): REFUTED — counterargument confirmed.** Job
262 final 1.3012 vs baseline_2k 1.2967: +0.0045 WORSE. Matched-step
deficit was a monotonically decaying transient: +0.024 @ 400, +0.0097
@ 800, +0.0067 @ 1200, +0.0051 @ 1600, +0.0045 @ 2000 (step-0 val
4.1077 matched baseline exactly, confirming init parity). The review's
bias-LR-artifact reading (B1) is rejected without a sweep, on two
grounds: (1) capacity — scalar_lr 0.04 is 2.7× the lr (0.015) under
which the mini bias reached |b|~3.4, gradients on the bias are dense
(every position, full vocab), and 2000 Adam steps at 0.04 bound drift
at ~80 ≫ 3.4, so the bias cannot have been underfit; (2) shape — an
underfit-bias world shows a small flat effect, not a large EARLY
deficit that decays; the observed shape is an early-transient COST
(the bias double-counts frequency while row norms already carry it,
and the trunk must re-equilibrate) that never finishes paying back.
Conclusion: the challenge's raw tied table with free row norms already
owns the frequency channel; the additive bias path is redundant there
and its adaptation transient is a pure tax. Discarded — no bias-LR
sweep. This closes the head-transfer branch: attachment (the one
component the mini program isolated as load-bearing) is ALREADY native
to the challenge baseline, and the remaining mini-head deltas
(rms-normed codebook + scalar scale + bias) were at best ~neutral in
mini and require the bias as frequency carrier. Next experiments should
come from the optimizer/schedule menu (track-3 record #46 techniques:
tail-EMA readout, PowerCool, Muon momentum schedule), not head
parametrization.

**H19 outcome (2026-07-22): NOT RUN TO VERDICT — user-cancelled.** Job
258 (baseline 4k) cancelled at ~step 2000 (1.3432, already below the 1k
final with half the decay remaining — on-schedule, not underperforming;
the mid-run gap vs the 1k curve was verified to be pure
schedule-position: the two runs are bit-identical through step 300
where the 1k cooldown starts). Job 259 (tieddot 4k) cancelled before
start. H19 is UNANSWERED, not refuted. Upstream's own guidance
(track-3 README: "val loss at step 1000 does not strongly correlate
with the final loss") is worth keeping in mind for all matched-step
mid-run reads in this lineage.

## Tied-Embed LR Arm (registered before results, job 246)

`energy_readout_b512_tiedlr05_fineweb_2k`: identical to the B512 arm except
`TIED_EMBED_LR=0.5` (native `train_gpt.py:76` env knob; default 0.05).
Rationale: 0.05 was tuned for the baseline's *direct* tied path, where
`tok_emb` rows are read as logit weights themselves. Here the embedding
sits behind a 2M-param BN-MLP projector in both its input role and its
codebook role, so the effective LR of the geometry the head actually uses
is governed by a path the harness never tuned. Single-factor comparison
against job 245 at matched steps.

**H7:** raising tied_embed_lr 10× improves BPB at 2k by >0.005 if the
embedding table is currently the bottleneck resource (under-trained
relative to the projector that consumes it). Signature if true: faster
early policy-loss descent and larger codebook_pairdist growth vs 245.
Risks: BN downstream of a fast-moving table can chase shifting input
statistics (watch for elevated val noise), and Adam on a 10× LR can
destabilize rare rows that get sparse gradient — a *worse* result is fully
plausible; that too closes the question. If 0.5 wins or loses clearly, the
bracket {0.15, 1.0} decides direction next.
(Verdict: H7 not supported on the observed window. Job 246 cancelled at
step 140 per user direction — behind its 0.05 control at every matched
eval (+0.046 @ 20 narrowing to ~+0.008 @ 100; last val 1.7151 @ 140).
The predicted faster-early-descent signature never appeared; the 10× LR
cost a transient and at best reached parity. Canaries ≈ control. The
{0.15, 1.0} bracket is deprioritized; if revisited, bracket DOWN toward
0.15 rather than up, since the transient penalty scaled with LR.)

## No-SIGReg Arm (registered before results, job 247)

`energy_readout_b512_nosigreg_fineweb_2k`: identical to the B512 arm
(same script) except `FRESH_LEJEPA_SIGREG_WEIGHT=0` (native knob,
fresh_lejepa_train.py:165, propagates to `sigreg_loss_weight` at import).
Deliberately weight-0 rather than encode-removed: the pooled encode, its
BN running-stat updates, and the RNG projection draws all still execute
identically to job 245, so the ONLY difference is the sigreg gradient —
the cleanest single-factor cut available (at the cost of ~12% wasted
forward compute, irrelevant to step-matched BPB). Free diagnostic:
`train_components[2]` logs the *unweighted* Epps–Pulley statistic, so the
run measures how non-Gaussian the geometry drifts once nothing pins it.

**H8:** under energy-CE, SIGReg is not load-bearing for collapse (CE's
attract/logsumexp-repel structure excludes the trivial solution, unlike
attached-target MSE where collapse is a global minimum); with the weight
at 0, CE shapes the embedding geometry freely. Registered prediction:
no catastrophic collapse (val_bpb stays on a normal descent curve);
direction of the BPB delta vs 245 genuinely uncertain — small win if the
freed anisotropy pays (per-dim head exploits axis structure SIGReg was
suppressing; expect energy_scale_max/min spread ≫ the ~9× seen under
SIGReg), small loss if conditioning or rare-code placement degrades
(watch codebook_pairdist_mean and the unweighted sigreg statistic
climbing without bound). If |Δ| ≤ 0.005 at matched steps, sigreg weight
becomes a free axis and the pooled-B machinery can be dropped at full
scale for ~12% step-time savings; if 247 clearly wins, the LeJEPA
regularizer exits the recipe and the family becomes
"BN-projected energy-CE." Either outcome also decides whether the
post-training latent-manifold argument (SIGReg-pinned N(0,I)) must be
paid for in pretraining BPB or comes free.
(In-flight evidence, registered at step 100: strong early lead over
paired-243 (−0.095 @ 20, −0.023 @ 60, −0.033 @ 80) that CROSSED OVER at
step 100 (1.8010 vs 1.7949, +0.006 behind). Unweighted EP statistic blew
out to ~43 (vs ~2.5 pinned); codebook contracted: pairdist 27.6 / norm
20.5 vs ~39 / ~28 under sigreg, with the temperature floor compensating
upward (scale_min 0.024 vs 0.010). Shape matches "unconstrained CE
sprints early, regularized geometry compounds late." Verdict deferred to
the 200–400 window and the 2k final.)
(Verdict on the full comparable window, steps 20–260: DEAD HEAT — deltas
oscillate ±0.006 with zero trend (e.g. −0.004 @ 180, +0.005 @ 200,
−0.004 @ 260); the step-100 crossover was noise. H8 confirmed on the tie
branch: sigreg costs 11% step time and buys no BPB. Canary convergence:
the no-sigreg codebook contracted early (pairdist 27.6 @ 100) then
climbed back to 36.3 / norm 27.1 by step 300 — CE rebuilds approximately
the sigreg equilibrium (~39/~28) unaided. Cancelled at step 420 per the 400-step
switch cadence — last val 1.4784 @ 420, with the codebook fully
converged to the sigreg-equilibrium geometry unaided (pairdist 43.5,
norm 32.7). H8 scored on the ≤260 matched window (tie) plus this
trajectory; 248 remains the test of whether small-but-nonzero beats
zero.)

## Ablation Plan

Run 1 executed on the parent's data (`mathmix_v4_sp1024`) against the parent
(`fresh_lejepa_srms_pope_belief_attached_ce_b128_mathmix_v4_2k`: 1.4009 BPB).
Per project direction the baseline comparison track uses
`fineweb_onepass_sp1024` (jobs 240/241 above); follow-up arms 2–7 should run
on the FineWeb dataset so they compare against `baseline_2k` unconfounded
(swap the unigram counts file for a FineWeb-derived one in run 6).

| # | Run | Question |
|---|---|---|
| 1 | `energy_readout_2k` | Does the pure energy readout beat the probe at equal steps? |
| 2 | `energy_readout_bigram_2k` | Is the current-token channel load-bearing? |
| 3 | `energy_readout_mse1_2k` (`ENERGY_LATENT_MSE_WEIGHT=1`) | Does the explicit MSE anchor still add anything once CE carries the attractive term? |
| 4 | `energy_readout_dot_2k` (`ENERGY_HEAD_FORM=dot`) | Attribution control: does the `-s/2·‖c_k‖²` norm term matter, or is this just a tied dot head? |
| 5 | `energy_readout_sgcb_2k` (`ENERGY_DETACH_CODEBOOK=1`) | Attribution control: is the classifier-weight gradient into the codebook the active ingredient? |
| 6 | `energy_readout_unibias_2k` (`ENERGY_BIAS_INIT_COUNTS=energy_readout/unigram_counts_mathmix_v4.json`) | Does the log-unigram warm start pay off at 2k-step scale? |
| 7 | equal-wallclock rerun of the winner | Probe FLOPs and param savings only matter if BPB-per-second improves. |

Runs 4 and 5 exist because the head is provably a reparametrized tied-softmax
head: the experiment is really about *gradient dynamics*, and without these
controls a win or loss cannot be attributed to the distance form, the norm
bias, or the codebook gradient.

Keep threshold per protocol: >0.005 BPB at step 2000, plus no step-time
regression.

## Implementation Notes (from red-team review)

Script: `fresh_lejepa_train_energy_readout.py`. Tests:
`test_energy_readout.py` (CPU-only, run directly).

- **Parameter registration.** `energy_log_scale` (0-d) and `energy_bias` (1-d)
  are registered under `blocks[-1]`: `train_gpt.py`'s optimizer split only
  sees `blocks.named_parameters()` plus the embedding/skip/lm-head specials,
  so a top-level parameter would receive gradients and **no optimizer step**
  (a silent freeze). `ndim < 2` routes both to the fused-Adam scalar group
  (`SCALAR_LR`) with fp32 masters via `restore_low_dim_params_to_fp32`.
- **Probe deletion.** The parent lineage direct-assigns both `ResidualProbe`s
  in `__init__` (not via the `install_probes` hook), so overriding the hook
  would silently keep 7.35M probe params. The fork instead deletes the
  modules after `super().__init__()` — RNG-safe, so every surviving parameter
  initializes bit-identically to the parent — and asserts no `probe`
  parameter survives.
- **λ_lat single-sourcing.** The patched accumulation loop *recomposes* the
  logged train loss as `comp[0] + latent_loss_weight·comp[1] +
  sigreg_loss_weight·comp[2]` from the model class attribute, while gradients
  come from the forward. Both read `type(self).latent_loss_weight`, set from
  `ENERGY_LATENT_MSE_WEIGHT`, so logs and gradients cannot diverge. The MSE
  is still computed (`no_grad`) as component 1 at weight 0 — a free
  diagnostic of how far the CE-shaped prediction drifts from latent
  regression.
- **Bigram routing.** `energy_bigram` is deliberately a flat 1-D parameter: a
  `(V, V)` matrix under `blocks` would be routed to Muon (orthogonalized
  updates are wrong for a frequency-indexed lookup table) and kept in bf16
  masters; 1-D lands in Adam with fp32 masters and serializes with a single
  per-tensor int8 scale.
- **Scale telemetry.** `energy_scale:<s>` is printed at every validation via
  an `eval_val` wrapper (composes under `v1.main`'s own eval wrapper).
- **Post-training incompatibility.** This family deletes the critic and
  changes readout features from `cat(token_latent, belief)` (2d-wide) to
  `predicted` (d-wide). Value/generation paths used by post-training raise
  a clear `RuntimeError`; this family is pretraining-only until a critic
  variant is defined.

## Red-Team Findings (design review, pre-implementation)

An adversarial review of this design was run before implementation; verdict
"implement with these changes" — all changes are incorporated above. High
findings: (1) new head params would have been silently frozen without
`blocks[-1]` registration; (2) probe deletion via the `install_probes` hook
would have silently kept all 7.35M probe params due to direct assignment in
the parent `__init__`; (3) the logged train loss is recomposed from
`latent_loss_weight` independently of the forward, so an unwired knob would
have made logs diverge from the optimized objective. Medium: bigram table
mis-routed to Muon/bf16 as naturally written; attribution controls (dot-form,
detached-codebook) required to interpret any outcome; `predicted` loses
latent-regression semantics at `λ_lat = 0` (gradient of CE cancels the
displacement pull — documented above, now an explicit decision). Low: init
calibration is under-dispersed but safe (starting CE ≈ ln V); `‖ẑ‖²`/`‖c‖²`
norms computed fp32; deleting the gradient-less critic also removes a latent
DDP unused-parameter hazard for the eventual 8×H100 validation. Strategic
caveat retained: the lejepa-ce line currently trails the plain-CE baseline at
equal steps and runs ~3.4× its step time, so a win here must eventually be
judged at equal wallclock, not just equal steps.

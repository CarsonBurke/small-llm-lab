# Uno / Ψ-Spec (Diffusion-Augmented LLMs) — Paper Record and MiniCPM Applicability Assessment

Paper: *Unlocking Lossless Speedups in LLMs via Discrete Diffusion*, Sahoo et al.,
arXiv:2609.04010v1 (3 Sep 2026), IFM / UIUC / Cornell Tech / Harvard / Cerebras.
Filed at `papers/uno_diffusion_augmented_2609.04010v1.pdf`. Paper:
https://arxiv.org/abs/2609.04010. Code/checkpoints: https://s-sahoo.com/uno

Status, **2026-09-08 numerical investigation**: opt-in distillation and captured
Ψ-Spec rollout are implemented; ordinary AR remains the default. The user selected
a shared, width-invariant bf16 target for Uno and its matched AR reference, rather
than requiring the legacy cuBLAS/Inductor rounding schedule. CPU contracts pass
(96 tests). Full-model CUDA qualification job5646 passed bit-identical serial/block
logits, clean KV and transformed sampling laws, including after actor-weight refresh.
Invariant decoding now requires complete compilation; the old eager-fallback probes
are not qualified runtime evidence. Three-arm fixture benchmark job5647 measured
0.932× invariant AR and1.493× legacy AR: the ≥1.25× dual gate correctly failed.
No full adapter-learning or RL campaign has started; trained benefit remains unproven.

**Subsequent decision:** Uno training/qualification is vetoed until better hardware
and explicit reauthorization. Standalone AR now defaults to optimized compilation
and KV writes with ordinary cuBLAS GEMMs, not the invariant Uno arithmetic.
Job 5706 found invariant GEMMs slower; job 5711 measured 3.33× useful rollout
throughput versus legacy at a 10,000-token cap. Its small quality sample was
22/64 optimized versus 26/64 legacy: quality neutrality is not established.
See [current MiniCPM decisions](../postraining/TODO_MINICPM5.md).
Future Uno adoption must also beat the new optimized production AR baseline,
not merely the historical legacy and invariant controls below.

Operational workflow: [MiniCPM Uno instructions](../postraining/README.md#opt-in-uno-diffusion-assisted-rollouts).
Implementation is in `postraining/uno.py`, `postraining/uno_speculative.py`,
`postraining/invariant_linear.py`, `postraining/invariant_attention.py`,
`scripts/train_minicpm_uno.py`, and `scripts/benchmark_minicpm_uno.py`.
Enable only with `--uno-rollout --uno-checkpoint ... --uno-block-size 4` after
qualification. The verifier uses the current actor's weights with an explicit
numerical contract; diffusion weights remain separate and frozen. Replay likelihoods
remain unchanged. Recovery pins both checkpoint bytes and the Uno arithmetic version.

Reference implementation: [`ifm-ai/uno` at
`46fbdb66f026bae9c68a1e5a3f97a17c7805c778`](https://github.com/ifm-ai/uno/tree/46fbdb66f026bae9c68a1e5a3f97a17c7805c778).
Its release also advertises a distinct 1B checkpoint; that is not a MiniCPM adapter
or evidence for this rank48 transfer recipe. The implementation uses the released
paired-mask/gated-LoRA/TV insights, while retaining this repository's fused actor,
production sampling law and continuous scheduler.

---

# Part 1 — Organized paper record

## 1.1 Core idea

Define a high-quality **AR** distribution, then learn to draw multiple tokens in parallel
*from that same distribution*. One architecture, two decoupled weight sets per layer:

| Weights | Symbol | Role | Training |
|---|---|---|---|
| AR weights | θ_AR | response quality; the verifier | standard NTP pretrain → SFT → RL |
| Diffusion weights | θ_Δ | generation speed; the drafter | LoRA adapters, one *Diffusion Distillation* phase, θ_AR frozen |

The draft pathway is `θ_AR + θ_Δ`; the verify pathway is `θ_AR` alone.
**Losslessness comes from correct rejection sampling against the current target distribution**,
not from freezing weights alone. Freezing θ_AR during distillation preserves the original
target; RL may subsequently change it. Correct proposal probabilities, target sampling
transforms, clean KV state, and verifier numerics are still required. “Lossless” means equality
in distribution in exact arithmetic, not identical sampled IDs for the same RNG seed.

The paper explicitly acknowledges “sampling randomness and numerical nondeterminism”
in §5.2.1. Its pinned runtime uses ordinary `F.linear` with separate width-one and
width-B graphs; it does not provide a shape-invariant GEMM solution. Its mathematical
guarantee is not evidence of equality to a different native bf16 implementation.

Distinguishing properties vs. speculative decoding: no separate draft model, **one shared KV
cache** (EAGLE-3 and DFlash each keep two), fewer added parameters, lower peak memory, and
training context length `2·L` independent of block size `B` (DFlash needs `B·L`).

## 1.2 Diffusion formulation

- Forward process (interpolating, Sahoo et al. 2024a): `z_ℓt ~ Cat(· ; α_t x_ℓ + (1−α_t) π)`,
  `α_0 = 1`, `α_1 = 0`.
- Prior `π = 1/K` (**uniform-state**, USDM), not `π = m` (masked). Chosen for native
  self-correction, few-step generation, and better inference-time scaling.
- **NTP parameterization retained**: logits at position ℓ predict token ℓ+1, unlike
  conventional d-LLMs which predict the clean token at the same position. This is what makes
  the same weights usable as both drafter and AR verifier.
- Ψ-samplers (Deschenaux et al. 2026) are predictor–corrector:
  `Ψ_{s|t} = κ_t q_{s|t} + (1−κ_t)[α_s q_{0|t} + (1−α_s)π]`, `κ_t = 1` ⇒ ancestral.
  Critically, at `s=0, t=1` the whole thing **collapses to `Cat(·; x_θ(z_1))`** for both MDM
  (Eq. 9) and USDM (Eq. 10) — so single-step generation needs none of the Ψ machinery at
  inference time. The Ψ apparatus only matters for the multi-step / inference-time-scaling
  variant (§4.3, left as future work).

## 1.3 Diffusion Distillation (training θ_Δ)

Objective (Eq. 3): `L(θ_Δ) = α·L_DCD + β·L_TV`.

- **One-step blockwise DCD.** Materializing PF-ODE trajectories is prohibitive at LLM scale,
  so the entire trajectory is distilled into a single step mapping fully corrupt `z_1 ~ π^L`
  to clean `x`, done blockwise over `N` blocks of size `B`.
  `L_DCD = Σ_b Σ_ℓ KL( x_{θ_Δ,θ_AR}^{(N+b,ℓ)}([x,z_1]) ‖ x_{θ_AR}^{(b,ℓ)}([x,z_1]) )`.
- **Total-variation loss:** the paper sums vocabulary L1 distances, equal to twice conventional
  TV. For a fixed conditioning context, maximal-coupling acceptance is exactly
  `Σ_v min(p_v,q_v) = 1 − TV(p,q)` ([Leviathan et al., §3.2](https://proceedings.mlr.press/v202/leviathan23a.html)).
  This directly aligns the local objective with acceptance. It does **not** prove that minimizing
  an unweighted, teacher-forced block loss globally maximizes on-policy accepted-prefix length.
  That expectation depends on all preceding accept events and the inference context distribution.
- **Single forward pass for teacher and student.** Concatenate `[x, z_1]` (hence context
  `2·L`), use a **block-causal mask** (causal within `x`; causal within each noisy block
  `z_1^{(b)}`; `z_1^{(b)}` additionally attends to all preceding clean blocks `x^{(<b)}`), and
  use **gated LoRA** (Samragh et al. 2025) to *disable* the adapters at clean positions and
  enable them at noisy positions. θ_AR alone therefore produces the teacher logits, and
  `θ_AR + θ_Δ` produce the student logits, in the same pass.
- Curriculum: block size ramped upward during training; θ_AR frozen throughout.

## 1.4 Ψ-Spec sampler

Single-step draft, AR verify. Algorithm 1 (Suppl. B.3), differences vs. Leviathan in brackets:

```
1: z ~ ∏_{1}^{B-1} U[V]                     # [init block of B-1 uniform-random tokens]
2: [q0, q] ← x_{θAR,θΔ}([x_L, z]; x_<L)     # [θAR on clean x_L; θAR+θΔ on z — gated LoRA]
3: x_{L+1} ~ q0                             # [free clean token, drawn from AR weights only]
4: x̃ ~ q                                    # [B-1 draft tokens sampled in parallel]
5: p ← x_{θAR}([x_{L+1}, x̃]; x)             # AR verifier pass
6: r ~ ∏_{i=1}^{B-1} U[0,1]
7: n ← min({ i ∈ 1..B-1 : r_i > min(1, p_i[x̃_i]/q_i[x̃_i]) } ∪ {B})
8-12: if n = B: z̃ ~ p_B  else  z̃ ~ Norm([p_n − q_n]_+)
13: return [x, x_{L+1}, x̃_{1..n-1}, z̃]
```

Key structural facts:
- **Position 1 is drawn directly from the current AR target**, with diffusion adapters disabled
  and only clean causal KV visible. It needs no accept/reject test. In an unterminated cycle,
  immediate rejection then emits **2** tokens; full acceptance emits `B+1`.
  Hence `1 ≤ TPF ≤ (B+1)/2`, and `τ = 2·TPF` for two forwards per cycle.
  EOS, remaining output budget, or context capacity can truncate the emitted block below this
  floor. TPF counts forwards, not their wall-clock cost; TPF ≥ 1 does not guarantee speedup.
- The clean/noisy logit split within the draft pass is required to avoid distribution shift
  (θ_Δ is trained only on corrupted positions) and is realized by gated LoRA.
- **Linear sampler**: one candidate, sampled from the per-position marginals. Best for *system*
  throughput (compute-bound at high batch).
- **Tree sampler**: top-`K` per position, Medusa-style tree attention, EAGLE-3-style log-prob
  pruning to the top-`V` prefixes. `(B,K,V)`. Best for *per-request* throughput at batch 1
  (memory-bound, spare compute).
- Implemented in Nano-vLLM (used for all reported results) and SGLang.

## 1.5 Results that matter for us

Two settings: from-scratch **Uno** (8B: 36L, d=4096, MLP 12288, 32Q/8KV, head 128, vocab
250,624, 6.95B body + 2.05B untied embeddings; AR weights on ~23T tokens) and **UnoQwen**
(θ_AR = frozen Qwen3-8B).

**θ_Δ configuration and cost**

| | Uno (from scratch) | UnoQwen |
|---|---|---|
| LoRA rank / α, main text | 128 / 256 (α/r = 2) | 128 / 256 (α/r = 2) |
| Targets | **every** projection (Q,K,V,O,gate,up,down) | same |
| Trainable | — | 0.35B (4.4% of 8B) |
| Tokens | 7B (SFT-distribution) | 14.7B = 3 epochs OpenThoughts3-1.2M |
| Seq len | 16,384 → 65,536 | 4,096 |
| Block curriculum | 2→4→8 (600M each), then 8 | 2,4,6,8,12,16 (½ epoch each) |
| Batch / LR | 128 / WSD peak 5e-5, 200 warmup | 64 / constant 1e-5, 2% warmup |
| Loss | α=0.01, β=1 | α=0, β=1 |
| Wall clock | ~60 h × 64 H200 | ~32 h × 32 H200 |

**V1 recipe inconsistency, with a released reference now available:** §§5.1.2 and
5.2.2 specify α=256 at r=128, while Appendix C.6.4 selects α/r=64 for Uno and 16 for
UnoQwen. The pinned release's UnoQwen reproduction uses r128/α2048, TV-only,
lr1e-5, global batch128 and 562 warmup steps, consistent with the appendix's
UnoQwen scaling—not the main-text α256. The shorter MiniCPM pilot's r48/α3072,
effective batch64 and 2% token warmup remain explicit candidate choices.

**Throughput (Table 7, 1K/8K test, H200)** — Uno 5255 sys / 383 per-req vs base AR 3577 / 176.
So **1.47× system throughput at max batch, 2.18× per-request**. DiffusionGemma-26B-A4B is
faster at batch 1 (836) but collapses in aggregate (1136) and loses badly on accuracy.

**Batch-size scaling (Table 18, Qwen3-8B, concurrency 64)** — Uno B4 5733 / B8 5600 / B16 4112
vs AR 3662 system tok/s. Among these linear settings, B=4 wins at C=64 and B=16 at C≤4.
Including trees, B16,V32 wins at C=1 and C=2; linear B16 wins at C=4.

These are the paper's **synthetic 1K/8K timing tests**: random inputs and a prescribed average
acceptance measured separately on benchmarks (§5). They are not end-to-end continuous-refill
rollouts. Table 18 contains both system and per-user rows despite its per-request caption.
Table 17's caption calls its values “unfiltered,” whereas §5.2.1/C.1.2 specify top-p=.95,
top-k=50 at temperature 1; retain that source ambiguity rather than infer a filtering benefit.

**Acceptance (Table 17a, temp = 1, linear sampler)** — τ: B4 3.89, B8 4.94, B16 5.42.
Tree B16,V32 5.97; B16,V60 6.11. At temp = 0: B4 4.27, B16 7.11, tree B16/V60 8.37.
Math and coding are the highest-τ domains (MATH500 B16 τ=6.26; MBPP B16 τ=5.20).

**Ablations (§5.2.4, Tables 12–15)** — these are the actionable ones:
1. **Loss**: 0.01·KL+TV 2.40 ≈ TV-only 2.39 > KL+TV 2.23 ≈ KL-only 2.22 (Table 12;
   the body rounds/reports KL-only as 2.23). KL is ~10× larger in magnitude. This supports
   TV as a promising objective in that experiment, not a proven repair for NextLat.
2. **Curriculum**: both arms first train ½ epoch at B=2 and ½ epoch at B=4. Ramping the
   remaining two epochs through B=6,8,12,16 gives 2.71 versus 2.65 for two epochs at B=16.
3. **LoRA placement**: all projections 2.39 > QKVO 2.38 ≈ O-only 2.38 > QV 2.26 > Q-only 2.14,
   at matched 349M parameters. Spread beats concentration, weakly.
4. **Rank**: r 128→256 gives 2.39→2.47 for 2× adapter parameters and adapter matmul FLOPs,
   **not** 2× total draft cost.
5. **α/r**: at 1 epoch, 2→64 lifts TPF 2.39→2.63 (+10%).
   The optimum differs with horizon: 64 at 1 epoch, 16 at 3 epochs.
6. **Best one-epoch TPF / best three-epoch TPF ≈ 97%** (2.63/2.71). This compares
   different scaling and training curricula, not a single run's convergence curve.
   The shared two-forward floor also inflates this ratio: `(2.63−1)/(2.71−1) ≈ 95%`
   of the excess TPF. No sub-epoch final-model comparison is reported.

**RL post-training (§5.1.3, Table 8, Suppl. C.2)** — DAPO from the SFT checkpoint, four
experts (math, code, tool use, browse). θ_Δ trained on the SFT checkpoint, **frozen**, used
only to accelerate rollouts; only θ_AR updated. Math expert: 2,560 steps, 28.9B response
tokens, lr 5e-7, 16 groups × 32 prompts = 512 sequences/step. The authors report
**up to 40% end-to-end training speedup**, with detailed timing deferred to a future revision.
Table 8 compares SFT with the consolidated post-RL model: mean TPF 2.25→2.10
(6.7% reduction; called 6% in the paper). This is not a maximum drift bound:
GSM8K drops 2.66→1.95 (26.7%), and every listed math task loses acceptance.

## 1.6 Honest limitations stated by the authors

- Every step is 2 forward passes. Quadratic/single-pass sampling (Samragh et al.) would fix
  this but needs kernels they did not write.
- MTP heads are *complementary* and might raise acceptance; not combined.
- Inference-time scaling by `T > B` denoising steps is proposed but unexplored.
- Tree verification cost can dominate (V=64 loses to V=32 at batch 1).
- I-DLM's released "lossless" LoRA + sampler could not be reproduced as lossless.

---

# Part 2 — Applicability to our MiniCPM VAPO RL stack

## 2.1 What we actually have

Verified against the pinned config
`~/.cache/huggingface/hub/models--openbmb--MiniCPM5-1B/snapshots/87179e5c.../config.json`:

`openbmb/MiniCPM5-1B` @ `87179e5c1f455ef22e6223592d2d61351b525bfc` — `LlamaForCausalLM`,
24 layers, hidden 1536, intermediate 4608, 16 Q heads × head_dim 128 (q_proj out = **2048**),
2 KV heads (k/v out = 256), vocab 130,560, **untied** embeddings, rope_theta 5e6,
max_position 131,072, bf16. 1,091,839,488 total / 1,080,632,832 frozen.

- Trainer: `postraining/train_minicpm_vapo.py` — single-GPU VAPO (length-adaptive GAE,
  `_precompute_advantages` `minicpm_vapo.py:965`) with DAPO clip-higher (0.20 / 0.28), on
  DAPO-Math-17k. Actor and critic own **disjoint rank-16 / α-32 LoRA** over one shared frozen
  bf16 backbone. Thinking template on by default.
- Rollout: bespoke CUDA-graph-captured continuous-refill decoder,
  `CapturedTrainingRolloutEngine.generate_prompt_pool` (`postraining/fast_inference.py`),
  64 lanes (4 prompts × 16 samples), sequence-major static KV (`_CompactStaticLayer`),
  fixed-shape FA4 varlen (`_fixed_varlen_fa4_attention`), fused QKV + gate/up. No vLLM, no SGLang, no
  HF `generate`. Sampling temp 0.9 / top-p 0.95 / top-k 20.
- Weight sync: in-process. `build_fused_rollout_replica` (`:418`) holds a LoRA-merged,
  projection-fused replica; `synchronize_fused_lora_policy_` (`:196`) re-merges the live actor
  adapters into it each rollout while the training backbone is offloaded to pinned host memory.
- **Rollout is the bottleneck**: 64.76 s of a 94.93 s cycle (68%) and 57.09 s of 76.87 s (74%)
  — `NOTES.md:5880-5891`. Decode productive utilization 71.63%, lanes decaying 64→48 by
  position 1,536 and →36 by 3,584 (`NOTES.md:5902-5910`). Production reference:
  **2,578.61 scheduled decode tok/s** at B64/10,000 tokens (`NOTES.md:5962-5964`).

**Reusable primitives exist; the production verifier does not transfer verbatim.**
`maximal_coupling_verify` (`postraining/minicpm_vapo.py:923-963`) implements per-position
acceptance and residual correction, not the whole block algorithm or bonus-token transition.
`postraining/nextlat_speculative.py` supplies `RaggedStaticCache`, `ragged_causal_mask`,
accept/commit/rollback, bonus-token handling, and `NextLatDecodeStats`. Existing tests cover
these building blocks; they do not exercise an Uno drafter in the production refill engine.

In particular, `dense_top_p_probabilities` accepts temperature and top-p **but no top-k**.
Production `top_k_top_p_sample` uses top-k=20 before nucleus filtering. Reusing the NextLat
probability path unchanged would preserve a different target distribution. Both the free
token and verifier must use the live actor's exact production sampling transform; each draft
probability must describe the proposal actually sampled. The existing benchmark's approximate
fusion-logit gate is useful diagnostics, not a distributional-equivalence proof (§2.8).

## 2.2 What the NextLat rejection does and does not establish

`NOTES.md:5931-5932` records position-2 acceptance of 0.90% at step 370 and 0.55% at step 203.
The NextLat benchmark reports B64 d=2 4,032 tok/s and d=6 2,544 tok/s versus its own AR
8,456 tok/s (2.10× and 3.32× slower). These are not the production 10k-token baseline.

Uno changes several things together: a full-backbone noisy-block drafter, TV distillation,
parallel rather than sequential drafting, and a first token sampled directly from the target.
That makes it a distinct hypothesis worth measuring, **not an established fix**:

- KL is also an agreement objective: Pinsker's inequality bounds TV by `sqrt(KL/2)`.
  The paper's 2.22→2.39 TPF ablation cannot attribute NextLat's near-zero acceptance solely
  to its loss or predict that TV will repair it.
- Parallel drafting removes sequential latent-MLP/LM-head calls, but adds an entire adapted
  backbone block pass. The original timing does not isolate which cost dominates.
- NextLat's first post-prefill proposal uses the target distribution (`nextlat_speculative.py`
  initial `has_pending=False` branch), but subsequent pending-token cycles use its latent
  approximation. Uno restores the AR-only first-token property each ordinary cycle.
  This improves the forward-count floor, not necessarily tokens/second.

The latent-action objection (`NOTES.md:1795-1799`) is separate from native token sampling.
The existing no-speculation production decision remains in force until correctness and
matched throughput evidence justify revisiting it. This assessment is not an adoption.

## 2.3 Adapter sizing (exact)

Per-layer LoRA parameters at rank `r` over all 7 projections, from the real config:

```
q: r(1536+2048)=3584r   k: r(1536+256)=1792r   v: 1792r   o: r(2048+1536)=3584r
gate: r(1536+4608)=6144r   up: 6144r   down: r(4608+1536)=6144r
per layer = 29,184 r      × 24 layers = 700,416 r
```

Validation: `r=16 → 11,206,656` — exactly the recorded trainable count in
`postraining/benchmarks/minicpm_lora_vapo_training.json:39-46`. Formula confirmed.

| r | θ_Δ params | % of 1.0806B frozen |
|---|---|---|
| 32 | 22.4 M | 2.07% |
| **48** | **33.6 M** | **3.11%** |
| 64 | 44.8 M | 4.15% |
| 128 | 89.7 M | 8.30% |

Scaling `r/hidden = 128/4096 = 1/32` gives **r=48**; α/r=64 gives **α=3072**.
These are candidate hyperparameters, not an optimal transplant established by data.
The paper's 0.35B/4.4% count is for **UnoQwen**, not the from-scratch Uno architecture.
Table 12's r=128→256 gain does not predict a MiniCPM r=48→128 gain. Adapter parameter
ratios also do not equal wall-time overhead: the frozen input embedding is not a dense
matmul, the LM head is, and small unmerged GEMMs can be launch- or bandwidth-limited.

## 2.4 Throughput model: useful sensitivity analysis, not a forecast

At fixed live-row occupancy and cache lengths, define:

```
C_B = (t_draft(B) + t_verify(B) + t_sampling/cache(B)) / t_AR
S_decode = τ / C_B
S_decode,max = (B+1) / C_B
```

Use the full one-token AR step, including sampling/cache overhead, as the denominator.
Measure the cycle components without double-counting. For a hypothetical equal-cost pair
of forwards with negligible added overhead and linear query scaling:

```
δ_B = (t(B)/t(1) − 1) / (B−1)
C_B ≈ 2[1 + (B−1)δ_B]
```

`δ_2` need not predict `δ_4`, `δ_8`, or `δ_16`. Adapter work, attention arithmetic,
softmax/top-k, RNG, residual sampling, rollback, refill, and graph scheduling can change `C_B`.

Back-solving this simplified model from Tables 17a/18 at concurrency 64:

| B | τ | paper system tok/s | speedup over 3662 | effective δ_B |
|---|---|---|---|---|
| 4 | 3.89 | 5733 | 1.566 | 0.081 |
| 8 | 4.94 | 5600 | 1.529 | 0.088 |
| 16 | 5.42 | 4112 | 1.123 | 0.094 |

The arithmetic is correct, but these are **effective fitted costs** from the paper's synthetic
throughput procedure (§1.5), not independently measured marginal kernel timings. Their
agreement does not validate an RTX 5090 forecast.

**Memory/compute inventory, not a measured roofline:**

- Frozen weights occupy ~2.16 GB bf16; not all are streamed by each decode matmul.
- KV capacity is `24 × 2 KV heads × 128 × 2(K,V) × 2 bytes = 24,576 bytes`
  per cached token per row. At 64 rows and 6,000 visible tokens this is 9.44 GB;
  at all 11,024 slots it is **17.34 GB / 16.15 GiB**, not ~6 GiB.
  A 6,000-token mean cursor is assumed, not measured for the proposed workload.
  Capacity is not measured HBM traffic: kernels, GQA reuse, tiling, occupancy, and actual
  per-row lengths determine bytes read and whether KV bandwidth dominates.
- Body+head linear work is ~112.6 GFLOP per additional query across 64 rows.
  At 6,000 visible tokens, attention QK/AV adds approximately
  `4 × 64 × 24 × 2048 × 6000 = 75.5 GFLOP` per query, before adapters and sampling.
- The original estimate divides **only the linear work** by an assumed, not achieved,
  84 TFLOP/s to obtain 1.34 ms.
  Dividing again by the historical `50.614/4096 = 12.36 ms` gives ~0.11, but that old
  4,096-position run had declining occupancy and predates serving fixes. Combining it with
  10k-token cache assumptions does not estimate current production `δ_B`.

Keep δ=.11 only as a labeled **conditional scenario**:

| B | paper τ | assumed C_B | decode speedup | optimistic whole-step speedup |
|---|---|---|---|---|
| 4 | 3.89 | 2.66 | 1.46× | 1.27–1.31× |
| 8 | 4.94 | 3.54 | 1.40× | 1.24–1.27× |
| 16 | 5.42 | 5.30 | 1.02× | ~1.02× |

The final column uses Amdahl's law `1/[(1−f)+f/S]`, f=.68–.74, **as if the entire
rollout sped up by the decode factor**. Prefill, synchronization, scheduling and other
unaccelerated rollout work make that optimistic. Measure the decode fraction separately.
Neither τ nor δ has been measured for this 1B setup; B=4 is a starting candidate, not a
locally established optimum.

Stronger top-k/temperature truncation is **not guaranteed to improve acceptance**.
For example, p=(.51,.49), q=(.49,.51) overlap by .98, but top-1 truncation makes their supports
disjoint and acceptance zero. Likewise, model size alone does not establish next-token
entropy or draftability on our prompts. Measure post-transform agreement in the actual domain.

## 2.5 Implementation and performance prerequisites

**1. Multi-query cached append is implemented separately from prefill.**
The original `_CompactStaticLayer.update` classified width>1 as prefill and would
overwrite the prefix; a `max_seqlen_q` change alone was invalid. Prefill/append
mode is now explicit, including single-token prefill. `UnoTrainingRolloutEngine`
uses `_UnoStaticLayer` for per-row suffix append and native bottom-right-causal
FA4 with a fixed B-query draft/verification schedule and reserved scratch capacity.

Installed `flash_attn_varlen_func` accepts fixed-shape query/KV tensors, `max_seqlen_q`,
`seqused_q`, `seqused_k`, and causal mode without requiring a page table. That establishes an
API candidate; the queued numerical checks qualify SM120 behavior rather than
assuming it from the signature. Validate bottom-right causal alignment, per-row
append positions, active lengths and full-prefix visibility before timing.
Attention-only support cannot establish full-backbone `C_B`.

**2. Shared KV needs a clean-state invariant, not just a layout conversion.**
Retain only AR-path KV for committed clean tokens. For Algorithm 1, the draft processes
`[x_L,z]` over clean `x_<L`; only x_L's KV can become committed. Discard/overwrite the
noisy suffix before verification. Verify `[x_{L+1},x̃]` over clean x, then retain the
accepted-prefix KV while leaving the correction/bonus token pending for the next draft.
No rejected proposal or diffusion-path KV may become visible to the next clean token.
Per-row cursors, RoPE positions, EOS, near-capacity blocks, inactive rows and lane refill
must obey this invariant inside graph replay. `RaggedStaticCache` and the NextLat loop are
references, not drop-in replacements for the production sequence-major cache/scheduler.

**3. Add a separate gated diffusion adapter without disabling the actor.**
During RL, θ_AR means the **current actor**, including its rank-16 update. It can remain
merged in the fused replica. Only θ_Δ is gated: off on the clean first token and all verifier
positions; on at noisy draft positions. Training also needs a gate, block mask, aligned
position IDs and **same-output-position** teacher/student next-token targets.
Neither an extra teacher shift nor reconstructing the current noisy input is the objective.
Ensure actor synchronization cannot overwrite or accidentally merge the diffusion adapter.

For fused projections, stacked A factors are a candidate; a dense block-diagonal B is not
automatically optimal. At r=48, QKV uses A `(144,1536)` and B `(2560,144)`, while gate/up
uses A `(96,1536)` and B `(9216,96)`. The dense B multiplies structural zeros: 2/3 of QKV
and 1/2 of gate/up B entries. Compare independent/grouped or fused low-rank updates rather
than prescribe padded GEMMs as “negligible.” Preserve the base QKV/gate-up fusion and
avoid diffusion work altogether on the verifier pass; benchmark the complete draft path.

**4. Training uses compiled FlexAttention, not FA4 custom-mask backward.**
The concatenated teacher/noisy-block mask is not ordinary triangular attention.
Installed SM120 FA4 backward rejects `mask_mod` and `block_sparse_tensors`.
The released Uno implementation uses native FlexAttention for this formulation;
`postraining/uno.py` now constructs a compiled BlockMask and uses HF's compiled
FlexAttention forward/backward with checkpoint-stable gates. Clean rows are causal;
noisy rows see earlier clean blocks and causal positions within their own noisy block.
Full-model numerical/backward qualification and measured throughput remain distinct.

Tree inference is **deferred**, not proven impossible: block-sparsity rejection alone does
not exclude other mask implementations; the SM120 forward path exposes `mask_mod`.
Linear B=4 is the paper-supported high-concurrency starting point. No MiniCPM τ ceiling
can be inferred from the paper's measured linear or tree acceptance.

**5. Use exact small-support coupling where possible; preserve graph shapes.**
Dense fp32 distributions at 64×B×130,560 cost 134/267/535 MB each for B=4/8/16.
Three such tensors are already 0.40/0.80/1.60 GB; logits, sorting indices and temporary
softmax/residual buffers add more. The old NextLat helper materializes dense residuals
and synchronizes during adaptive nucleus search; it is not the captured Uno verifier.

The new verifier keeps exact bounded top-k/nucleus supports. It looks up p at proposed
IDs and q at target IDs with sort/searchsorted, then samples `max(p-q,0)` only on
the target support (at most 20 IDs): outside that support the residual is exactly zero.
This handles partial/disjoint support without dense full-vocabulary probabilities
or a padded 40-ID union. Full LM-head/top-k work still remains. Fixed-shape device
buffers avoid dynamic rejected-row compaction and per-position host synchronization.
If proposal truncation changes, use its actual normalized support, not this bound.

Keep these transformed behavior probabilities separate from the untempered current-actor
logprobs stored for RL replay (`selected_token_logprobs`); do not record proposal/residual
logprobs as the actor likelihood.

**6. Distillation peak memory requires an actual backward measurement.**
At L=2048, `[x,z]` has 4096 positions: full logits occupy 2.139 GB fp32 across both halves;
each L-position teacher or student distribution is 1.070 GB. A 128-position full-vocabulary
chunk is 66.85 MB **per tensor**, not the whole loss footprint.
Chunk over positions while normalizing over the full vocabulary. Independent vocabulary-
chunk softmaxes would change TV. Merely summing chunk losses before backward can retain
every chunk's autograd buffers; use a verified recomputation/backward design that bounds
retained activations. Existing frozen-head logprob chunking is not proof of this TV design.
Include mask/attention temporaries, optimizer, checkpoint recomputation, teacher/student
probabilities and other resident models when measuring peak memory. 32 GB fit is unproven.

## 2.6 Teacher, drift, and corpus

**A fresh actor's base is a reasonable candidate teacher, not a universal optimum.**
`LoRALinear.__init__` initializes B to zero under both standard and current NoRA
initialization (NoRA normalizes A only). Thus a fresh, unloaded actor equals the frozen base
mathematically. A resumed actor checkpoint is generally different. If accelerating a resumed
run is the objective, evaluate or distill against that actual actor checkpoint.
Pin teacher weights, model revision, tokenizer, chat template, sampling settings, and corpus
preprocessing. A base-trained adapter can be reused with a new current-actor verifier without
invalidating rejection sampling; **its speedup is not guaranteed**.

**Low rank does not bound distribution drift.** Learned A and B do not define a fixed
rank-16 linear subspace of output distributions, nor constrain update magnitude or logit KL.
Even a rank-one update can reverse the dominant token and destroy draft overlap.
The paper's 6.7% mean TPF decline hides 26.7% loss on GSM8K (§1.5). Measure τ and cycle
throughput at relevant actor checkpoints and rollout lengths rather than assume less drift
than full-parameter DAPO.

**OpenThoughts3-1.2M is a candidate offline corpus.** The paper demonstrates successful
cross-distribution distillation for Qwen3-8B, not for MiniCPM. Check dataset licensing,
thinking-template alignment, truncation/packing, and held-out prompt separation; retokenize
and count tokens with MiniCPM's tokenizer. The paper's 4.9B tokens/epoch does not transfer
unchanged across tokenizers. Existing on-policy rollout traces are another candidate.
Generating a new 500M-token corpus at the recorded 1,288.45 useful tok/s would cost
~108 hours; that is expensive, not proof that reusing or generating a smaller corpus is
nonviable.

## 2.7 Training cost: illustrative arithmetic only

The original `8N` approximation with N=1.0806B gives 8.645 GFLOP per supervised token.
At an **assumed** sustained 84 TFLOP/s it yields:

| Budget | Nominal FLOPs | Arithmetic time, not a job estimate |
|---|---|---|
| 100M tokens | 8.645e17 | 2.86 h |
| 400M tokens (proposed pilot) | 3.458e18 | 11.44 h |
| 500M tokens | 4.323e18 | 14.29 h |
| 4.9B tokens (paper's tokenizer) | 4.236e19 | 5.84 days |
| 14.7B tokens (paper's tokenizer) | 1.271e20 | 17.51 days |

This is neither a lower nor upper bound. `8N` counts embeddings like dense linear work,
ignores teacher stop-gradient savings, and omits attention, low-rank backward, checkpoint
recomputation, loss, optimizer and data overhead. The implementation now selects compiled
FlexAttention explicitly; this arithmetic still does not measure its full update cost.
Measure full forward/loss/backward/optimizer throughput and peak memory at the intended
precision, compiled path, length and batch before setting a time budget. Sub-epoch quality,
adapter convergence at 1B, and tokens to useful τ remain unmeasured. No “one GPU-day
go/no-go” or “six-day full adapter” commitment follows from this arithmetic.

## 2.8 Evidence gates — implementation available, production adoption unproven

All local model/GPU workloads, including kernel probes, benchmarks and backward checks, must
use `mlq submit --max-parallel-runs 1`. Do not promote a technique from reduced-run quality
evidence or bypass the repository's ablation/adoption rules. The following is a conditional
research plan, not authorization to change production defaults.

**E0a — kernel feasibility and correctness, no training.**
Probe query widths `{1,2,4,8,16}` with sequence-major KV, 64 rows, homogeneous and ragged
visible lengths spanning short, middle and near-capacity contexts in an 11,024-slot cache.
Reserve n writable slots; do not append past capacity. Check multi-query outputs against
the same-prefix reference and validate causal alignment before timing. Separately evaluate
the supported training-attention/backward formulation; a fast causal forward is insufficient.
Record unsupported shapes as failures, not silent fallback measurements. Compilation,
warmup and integration time are separate from steady-state GPU timing; no ten-minute
completion guarantee is justified.

**E0b — full-model cost gate, after append-correct cache plumbing.**
Measure complete base block forwards against the one-token production path, with matching
weights, cache lengths, dtype, graph mode, logits and sampling. Compare FA4 and a labeled
SDPA reference only where both implement the same full-prefix semantics. Then measure an
actual unmerged gated draft path and sampling/cache cycle overhead before using `C_B` as
an adoption forecast. A no-adapter measurement only screens backbone cost.

For a required speedup R, require `τ ≥ R·C_B`; rule out a measured configuration if even
`(B+1)/C_B < R`. In the simplified B=4 model:

- δ=.15 and paper τ=3.89 predict **1.34×**, before omitted overhead.
- δ=.25 and paper τ predict **1.11×**; the true token-count ceiling is **5/3.5=1.43×**,
  not 1.11×. This may fail a chosen paper-acceptance business case, not prove impossibility.
- At δ=.11, τ=3.0 predicts only **1.13×**; **1.25× requires τ≥3.325** before overhead.
  A bare δ cutoff or τ≥3.0 gate cannot approve a 1.25× rollout claim.

**E1 — distillation only after feasible training and inference paths.**
Candidate pilot: r=48, α=3072, all seven projections, TV-only, lr=1e-5 with 2% warmup,
L=2048, position-chunked full-vocabulary loss, 100M tokens at B=2 then 300M at B=4
(**400M total**, not 500M). Specify diffusion initialization independently of the actor:
standard random-A/zero-B is a candidate, not the repository's implicit NoRA default.
This is a hypothesis, not a reproduced paper recipe.
Resolve the paper's scaling inconsistency (§1.5), memory/backward path and actual token
budget first. Evaluate held-out DAPO-math at temp=.9/top-p=.95/top-k=20, reporting τ,
conditional per-position acceptance, completion/truncation and measured `C_B` together.
Use sufficient learning evidence under the repository protocol; do not equate “97% of
best paper TPF at one epoch” with a convergence guarantee for this pilot.

**E2 — matched production rollout and correctness gates.**
The historical reference is **B64 / 10,000 response tokens**, not 4,096:
2,578.61 scheduled decode tok/s and **1,288.45 useful rollout tok/s** over 260.04 s
(`NOTES.md:5962-5966`). Rerun AR and candidate on the same pinned actor, prompt pool,
64 logical trajectories, output cap, cache budget, sampler and refill policy. A 4,096-token
experiment needs its own matched AR reference. Hold physical/logical concurrency fixed
and report any resident-capacity penalty.

Require ≥1.25× **useful rollout tok/s** for the proposed performance gate; also report total
rollout and full RL cycle wall time, prefill/decode timing, productive occupancy, τ, peak
VRAM and scheduled decode tok/s. Count only emitted in-budget tokens, never rejected or
post-EOS drafts. Scheduled throughput alone can reward idle/wasted work. Recheck at
representative later actor checkpoints; do not infer drift tolerance from LoRA rank.
Also amortize adapter training and setup: a measured S-fold rollout speedup saves
`T_rollout·(1−1/S)` GPU time over the remaining baseline rollout budget. That must exceed
the adapter's measured GPU cost for compute payback; engineering cost is additional.

Correctness needs multiple complementary checks:

- Compare serial/current-actor versus block verifier logits on the **same prefixes** across
  ragged lengths, clean first tokens, accepted/rejected suffixes, rollback, EOS, capacity
  boundaries, inactive lanes and refill. Check transformed supports/probabilities, not just
  mean cosine/top-1. Specify tolerances for the chosen bf16/fused implementation and report
  approximation rather than claim exact arithmetic from loose aggregate thresholds.
- The selected MiniCPM contract is `bf16-lane64-n128-k32-casts/v1`: fixed token/lane
  GEMM tiles, fp32 dot-product accumulation with bf16 inputs/outputs, and explicit
  compiler rounding. Tests require **bit-identical** logits, clean KV and compiled
  transformed supports/probabilities against this one-token reference. Drift from
  legacy AR is reported separately, not hidden by a wider numerical tolerance.
  Prefill is shared; its final token is recomputed through the invariant target
  before sampling, including on lane admission/refill.
- Validate the coupling against analytical small-vocabulary distributions (identical and
  disjoint support, partial overlap, ties, and every rejection position), plus empirical
  sequence-distribution comparisons with uncertainty. Include bonus/correction-token and
  output-budget transitions. These supplement, not replace, the algorithm/state invariant.
- A generated-ID SHA-256 is a replay diagnostic for the **same** algorithm and RNG schedule.
  AR and speculative decoding consume randomness differently; same-seed IDs need not
  match despite identical distributions. Hash equality is neither required nor sufficient.

## 2.9 Verdict

**Opt-in implementation available; production benefit not established.**
The code implements TV-only gated distillation before RL and linear B4 Ψ-Spec
against the changing actor, without enabling speculation by default.
Independent reviews corrected teacher-adapter device migration, multi-token
completion polling, cyclic retention of evicted KV and a pending-token prefill
handoff error. CPU contracts, full-shape training backward and complete-graph
captured numerical qualification pass. The invariant target forbids silent eager
fallback; the explicit legacy control remains unchanged, but is no longer the default.

The remaining research unknowns are trained MiniCPM draft acceptance, acceptance
drift under RL, and amortized payoff. Neither rank confinement,
narrow top-k, a released 1B model nor fitted paper timings resolves them.
Losslessness, learning quality and profitability remain separate gates.
After explicit reauthorization, train/evaluate the real offline-corpus adapter
and compare against the current production AR before enabling Uno for an RL campaign.

### Numerical experiment record

- Job5517 confirmed local precision-cast emulation makes the gated first token and
  independent clean-KV rebuild exact at fixed width. Native width-one versus block
  still reached 0.06131 maximum raw-vocabulary TV. FA4 on identical Q/K/V was exact;
  changing the output projection from M64 to M256 produced a 0.015625 bf16 difference.
  Its cache-release regression freed all old backing references; full-model
  backward peaked at 3,046,521,856 allocated bytes.
- Job5518 isolated projection geometry: four contiguous M64 GEMMs per block
  projection, with cast emulation in both paths, gave zero serial/block logit error.
  This costs four launches per projection and was not the selected implementation.
- Job5520 tested the fixed-tile Triton implementation on the full MiniCPM model,
  B64 and ragged prefixes 1/65/513/8192: zero serial/block logit error and zero
  gated-first-token error. CUDA-graph probe timings were 15.09ms one-token target,
  15.81ms verifier and 18.19ms draft, versus 14.08ms/16.43ms stock verifier/draft.
  These timings are **provisional, not qualified performance evidence**: those
  probes did not enforce complete compilation, and later tests exposed eager
  fallback. The invariant target differs numerically from stock bf16 arithmetic.
- Disabling bf16 reduced-precision reduction alone did not solve the discrepancy.
  cuBLASLt with both reduced-precision reduction and split-K disabled reduced
  same-backend serial/block maximum TV to 0.01763, but was still not exact.
  A separate default-compiler probe showed that matching GEMM shapes alone also
  does not reproduce shape-dependent compiler fusion/rounding.
- Jobs5630–5634 exposed the compiler failure: graph breaks at the native FA4
  wrapper specialized `LlamaAttention.forward` by layer index and exhausted
  Dynamo's eight-variant limit. Layer-qualified hook defaults added another
  specialization source. Hooks now capture projection modules directly; FA4 is
  a compiler-visible custom operator with checked mutation/fake-layout contracts.
  Invariant model forwards require `fullgraph=True`, preserved precision casts,
  and hard failure on recompilation-limit exhaustion. No limit was raised.
- Job5646 passed the full CUDA suite, including ragged prefixes1/65/513/8192,
  B64/B4, refill, EOS/output limits, cache eviction/rebuild, gated first token,
  exact selected-target sampling laws, and actor refresh. Legacy AR maximum raw
  TV was0.03000 before and0.04575 after the fixture actor update. Legacy retains
  its original compiler behavior, including its pre-existing fallback; the matched
  invariant AR reference does not use that fallback.
- Generated-code review found a distinct performance defect: functionalized
  `scatter_` cloned and copied back all48 KV backings each forward. At cache8230,
  that is48.22GiB of logical read/write traffic per forward, not a measured DRAM
  counter. Fresh returned views alone did not fix it. Indexed suffix updates use
  Inductor's registered in-place lowering. Job5646 passed every numerical check;
  generated-code inspection confirmed all96 full copies per forward disappeared.
  The identical33-cycle fixture fell from3.00618s (job5641) to0.76872s (job5646):
  **3.91× faster decoding schedule**, not an AR-relative or trained-adapter speedup.
- The benchmark now has three separate-process arms: invariant AR, Uno on the
  same target, and unchanged legacy AR. The ≥1.25× useful-throughput gate must pass
  against **both** AR arms; slowing the reference cannot manufacture qualification.

### Three-arm runtime fixture (job5647)

RTX5090;8 prompts×16 responses,64 physical lanes, Uno B4,4096-token output cap;
three same-seed timing repetitions after warmup in separate processes.
The adapter received only one2048-token numerical-fixture update.

| Engine | Median useful tokens/s | Median pool seconds | Peak allocated GiB |
|---|---:|---:|---:|
| Legacy AR, unchanged | 4812 | 108.58 | 11.82 |
| Shared invariant AR | 7712 | 67.75 | 11.79 |
| Uno fixture | 7185 | 72.90 | 11.96 |

Uno reachedτ=2.132 tokens per active row-cycle: **0.932× matched invariant AR**
and **1.493× legacy AR**. The performance gate correctly **failed** because both
comparisons must reach1.25×. This establishes runtime mechanics, not the speedup
of a properly distilled adapter. Between96.9% and99.2% of responses hit the
output cap; no task-quality conclusion is supported.

Canonical evidence, including raw repetition metrics and source fingerprints:
`ablation_results/uno_invariant_runtime_20260908/metrics.jsonl` and `result.json`.
The benchmark fixture checkpoint and throwaway diagnostic scripts were removed.

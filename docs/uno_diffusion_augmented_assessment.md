# Uno / Ψ-Spec (Diffusion-Augmented LLMs) — Paper Record and MiniCPM Applicability Assessment

Paper: *Unlocking Lossless Speedups in LLMs via Discrete Diffusion*, Sahoo et al.,
arXiv:2609.04010v1 (3 Sep 2026), IFM / UIUC / Cornell Tech / Harvard / Cerebras.
Filed at `papers/uno_diffusion_augmented_2609.04010v1.pdf`. Code/checkpoints: https://s-sahoo.com/uno

Status: **read-only investigation. No code changed. Nothing queued.**

---

# Part 1 — Organized paper record

## 1.1 Core idea

Define a high-quality **AR** distribution, then learn to draw multiple tokens in parallel
*from that same distribution*. One architecture, two decoupled weight sets per layer:

| Weights | Symbol | Role | Training |
|---|---|---|---|
| AR weights | θ_AR | response quality; the verifier | standard NTP pretrain → SFT → RL |
| Diffusion weights | θ_Δ | generation speed; the drafter | LoRA adapters, one *Diffusion Distillation* phase, θ_AR frozen |

The draft pathway is `θ_AR + θ_Δ`; the verify pathway is `θ_AR` alone. Because θ_AR is never
modified, rejection sampling against it is **exactly lossless** — unlike d-LLMs
(DiffusionGemma, Nemotron-Labs-Diffusion) and self-speculative conversions (TiDAR), which
mutate base weights and are therefore lossy.

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
- **Total-variation loss** — the term that actually matters:
  `L_TV = Σ_b Σ_ℓ | x_{θ_Δ,θ_AR}^{(N+b,ℓ)} − x_{θ_AR}^{(b,ℓ)} |`.
  Justified by Leviathan et al. Cor. 3.6: minimizing blockwise TV distance between draft and
  target maximizes the expected accepted-prefix length. This is a *direct* optimization of
  acceptance, not a proxy.
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
6: r ~ ∏ U[0,1]
7: n ← min({ i : r_i > min(1, p_i/q_i) } ∪ {B})
8-12: if n = B: z̃ ~ p_B  else  z̃ ~ Norm([p_n − q_n]_+)
13: return [x, x_{L+1}, x̃_{1..n-1}, z̃]
```

Key structural facts:
- **Position 1 is drafted with base AR weights only**, so it matches the verifier exactly and
  is *always accepted*. Combined with the residual resample, even total rejection yields **2**
  output tokens; full acceptance yields `B+1`. Hence `1 ≤ TPF ≤ (B+1)/2`, and
  `τ = 2·TPF` tokens per step (a step = 1 draft forward + 1 verify forward).
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
| LoRA rank / α | 128 / 256 (α/r = 64 chosen for Uno) | 128 / 2048 (α/r = 16 chosen) |
| Targets | **every** projection (Q,K,V,O,gate,up,down) | same |
| Trainable | — | 0.35B (4.4% of 8B) |
| Tokens | 7B (SFT-distribution) | 14.7B = 3 epochs OpenThoughts3-1.2M |
| Seq len | 16,384 → 65,536 | 4,096 |
| Block curriculum | 2→4→8 (600M each), then 8 | 2,4,6,8,12,16 (½ epoch each) |
| Batch / LR | 128 / WSD peak 5e-5, 200 warmup | 64 / constant 1e-5, 2% warmup |
| Loss | α=0.01, β=1 | α=0, β=1 |
| Wall clock | ~60 h × 64 H200 | ~32 h × 32 H200 |

**Throughput (Table 7, 1K/8K test, H200)** — Uno 5255 sys / 383 per-req vs base AR 3577 / 176.
So **1.47× system throughput at max batch, 2.18× per-request**. DiffusionGemma-26B-A4B is
faster at batch 1 (836) but collapses in aggregate (1136) and loses badly on accuracy.

**Batch-size scaling (Table 18, Qwen3-8B, concurrency 64)** — Uno B4 5733 / B8 5600 / B16 4112
vs AR 3662. Optimum is **small B at high concurrency**: B=4 wins at C=64, B=16 wins at C≤4.

**Acceptance (Table 17a, temp = 1, linear sampler)** — τ: B4 3.89, B8 4.94, B16 5.42.
Tree B16,V32 5.97; B16,V60 6.11. At temp = 0: B4 4.27, B16 7.11, tree B16/V60 8.37.
Math and coding are the highest-τ domains (MATH500 B16 τ=6.26; MBPP B16 τ=5.20).

**Ablations (§5.2.4, Tables 12–15)** — these are the actionable ones:
1. **Loss**: TV-only 2.39 > 0.01·KL+TV 2.40 ≈ TV-only, ≫ KL-only 2.22 = KL+TV 2.23. KL is
   naturally ~10× larger in magnitude and *swamps* TV unless downweighted. **TV is the term
   that buys acceptance.**
2. **Curriculum**: ramping B 2→16 gives 2.71 vs 2.65 for fixed B=16. Small effect.
3. **LoRA placement**: all projections 2.39 > QKVO 2.38 ≈ O-only 2.38 > QV 2.26 > Q-only 2.14,
   at matched 349M parameters. Spread beats concentration, weakly.
4. **Rank**: r 128→256 gives 2.39→2.47 for 2× the parameters and 2× the draft cost.
5. **α/r ratio matters far more than rank**: at 1 epoch, α/r = 2→64 lifts TPF 2.39→2.63
   (+10%). Optimum shifts with horizon: α/r = 64 at 1 epoch, 16 at 3 epochs.
6. **1 epoch reaches ~97% of 3-epoch TPF** (2.63 vs 2.71). Nothing below 1 epoch is reported.

**RL post-training (§5.1.3, Table 8, Suppl. C.2)** — DAPO from the SFT checkpoint, four
experts (math, code, tool use, browse). θ_Δ trained on the SFT checkpoint, **frozen**, used
only to accelerate rollouts; only θ_AR updated. Math expert: 2,560 steps, 28.9B response
tokens, lr 5e-7, 16 groups × 32 prompts = 512 sequences/step. Result: **up to 40% end-to-end
training speedup** (math and code; smaller for tool use, where tool calls dominate), and the
SFT-trained adapters retained their speedup after RL with only a **6% TPF drop** (2.25 → 2.10).
Detailed results deferred to a future revision.

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
  `CapturedTrainingRolloutEngine.generate_prompt_pool` (`fast_inference.py:1791`), 64 lanes
  (4 prompts × 16 samples), sequence-major static KV (`_CompactStaticLayer` `:273`), fixed-shape
  FA4 varlen (`_fixed_varlen_fa4_attention` `:377`), fused QKV + gate/up. No vLLM, no SGLang, no
  HF `generate`. Sampling temp 0.9 / top-p 0.95 / top-k 20.
- Weight sync: in-process. `build_fused_rollout_replica` (`:418`) holds a LoRA-merged,
  projection-fused replica; `synchronize_fused_lora_policy_` (`:196`) re-merges the live actor
  adapters into it each rollout while the training backbone is offloaded to pinned host memory.
- **Rollout is the bottleneck**: 64.76 s of a 94.93 s cycle (68%) and 57.09 s of 76.87 s (74%)
  — `NOTES.md:5880-5891`. Decode productive utilization 71.63%, lanes decaying 64→48 by
  position 1,536 and →36 by 3,584 (`NOTES.md:5902-5910`). Production reference:
  **2,578.61 scheduled decode tok/s** at B64/10,000 tokens (`NOTES.md:5962-5964`).

**The verification half of Ψ-Spec already exists in this repo and is unit-tested.**
`postraining/nextlat_speculative.py` (680 lines) implements exactly Algorithm 1 lines 5–13:
`maximal_coupling_verify` (`minicpm_vapo.py:923-963`) is `min(1, p/q)` accept with
`Norm([p−q]_+)` residual resample; `dense_top_p_probabilities` (`:858-920`) materializes the
nucleus densely; `RaggedStaticCache` / `ragged_causal_mask` (`nextlat_speculative.py:37-150`)
gives per-row independent write cursors inside a fixed-shape compiled graph; the accept /
commit / rollback loop (`:480-610`), bonus token (`:548-598`), and per-position acceptance
telemetry (`NextLatDecodeStats` `:20-35`) are all there;
`scripts/benchmark_minicpm_nextlat.py` provides a logit-fidelity gate (cosine ≥ 0.999,
top-1 ≥ 0.90, top-20 ≥ 0.95, KL ≤ 0.01, `:197-213`) and acceptance reporting (`:364-424`).

**What is missing is only the drafter and the block-wide forward.**

## 2.2 Why the existing rejection of speculative decoding does not transfer

`NOTES.md:5931-5932` — "Do not enable NextLat speculative decoding: measured position-2
acceptance is 0.90% at step 370 and 0.55% at step 203, far below break-even." Measured
throughput: B64 d=2 4,032 tok/s and d=6 2,544 tok/s vs AR 8,456 tok/s — 2.10× and 3.32×
**slower**.

That verdict is about a *drafter*, not the harness, and NextLat's failure mode is precisely
what Uno's L_TV exists to fix:

- NextLat is trained with SmoothL1 on hidden states + categorical KL through a detached LM head
  (`NOTES.md:5853-5860`). **It was never trained against a total-variation objective, and
  never on a noisy-block input.** Uno's own ablation shows KL-only gets TPF 2.22 vs TV-only
  2.39 — and that gap is with a *good* drafter; with a bad one, KL-only training has no reason
  to produce agreement on the sampled token at all.
- NextLat drafts **sequentially** (`nextlat_speculative.py:404-421`): `draft_length` iterations
  of latent-dynamics MLP + full 130,560-way LM head. Uno drafts the **entire block in one
  forward pass**. Sequential drafting is why d=6 was 3.3× slower even at ~0 acceptance.
- NextLat has no free always-accepted first token. Uno's position-1-from-θ_AR construction
  guarantees TPF ≥ 1 and ≥ 2 output tokens per cycle even on total rejection.

Also flagged: the repo-wide "does not land speculative changes" note (`NOTES.md:1030-1031`) and
`postraining/README.md:92-93`. The other written rejection (`NOTES.md:1795-1799`) is explicitly
scoped to the **latent** VAPO action space ("no lossless draft/verifier construction for the
joint discrete gate/token plus continuous latent action") — irrelevant to native MiniCPM tokens,
where a lossless construction demonstrably exists and is already implemented.

## 2.3 Adapter sizing (exact)

Per-layer LoRA parameters at rank `r` over all 7 projections, from the real config:

```
q: r(1536+2048)=3584r   k: r(1536+256)=1792r   v: 1792r   o: r(2048+1536)=3584r
gate: r(1536+4608)=6144r   up: 6144r   down: r(4608+1536)=6144r
per layer = 29,184 r      × 24 layers = 700,416 r
```

Validation: `r=16 → 11,206,656` — exactly the recorded trainable count in
`postraining/benchmarks/minicpm_lora_vapo_training.json:47`. Formula confirmed.

| r | θ_Δ params | % of 1.0806B frozen |
|---|---|---|
| 32 | 22.4 M | 2.07% |
| **48** | **33.6 M** | **3.11%** |
| 64 | 44.8 M | 4.15% |
| 128 | 89.7 M | 8.30% |

Uno used `r/hidden = 128/4096 = 1/32`. The **proportional transplant is r = 48**, giving 3.11%
of base (Uno's own adapter was 4.4% of 8B). With α/r = 64 (Uno's from-scratch choice, and the
1-epoch optimum in Table 15a) that is **α = 3072**. Using r = 128 here would make the adapter
8.3% of the model — the draft pass would carry ~8% extra FLOPs and 90M extra resident
parameters for the +0.08 TPF that Table 12 attributes to doubling rank. Not worth it.

## 2.4 The throughput model, calibrated on the paper's own numbers

A Ψ-Spec cycle is `t_draft(B) + t_verify(B)`. Let `t(n)` be one forward with `n` query
positions per row at 64 rows, and `δ = [t(2) − t(1)] / t(1)` the marginal cost of an extra
position. Then

```
speedup vs AR  =  τ / ( 2 · [1 + (B−1)·δ] )
```

Calibrating δ from Table 17a (τ) against Table 18 (throughput at concurrency 64, Qwen3-8B/H200,
AR = 3662 tok/s):

| B | τ | measured tok/s | implied speedup | implied δ |
|---|---|---|---|---|
| 4 | 3.89 | 5733 | 1.566 | 0.081 |
| 8 | 4.94 | 5600 | 1.529 | 0.088 |
| 16 | 5.42 | 4112 | 1.123 | 0.094 |

δ ≈ 0.081–0.094, stable across B. **The model is sound and δ is the single decisive quantity.**

Roofline estimate of δ for MiniCPM5-1B on one 5090 at 64 rows [INFERENCE]:

- Weights per forward: 1.0806e9 × 2 B = **2.16 GB**.
- KV per forward: 24 L × 2 kv × 128 × 2 (K,V) × 2 B = 24,576 B per position per row; at an
  average live cursor of ~6,000 over a 10k-token rollout × 64 rows = **9.4 GB** (17.3 GB at the
  full 11,024-slot cache). **KV read dominates the base step** — the regime speculation likes.
- Marginal compute per extra query position: body 679.5M + head 200.6M = 880.1M params
  ⇒ 2 × 64 × 880.1e6 = **112.7 GFLOP**. At ~84 TFLOPS achieved (40% of ~209 TFLOPS bf16 dense)
  ⇒ **1.34 ms**.
- Base step measured: 50.614 s decode / 4,096 positions = **12.35 ms** (`NOTES.md:5904-5906`).
- ⇒ **δ ≈ 1.34 / 12.35 ≈ 0.11**.

Projected speedup if the paper's τ transplants to a 1B model:

| B | τ (paper, temp 1) | 1 + (B−1)·0.11 | rollout speedup | end-to-end (rollout 68–74%) |
|---|---|---|---|---|
| 4 | 3.89 | 1.33 | **1.47×** | 1.28–1.31× |
| 8 | 4.94 | 1.77 | 1.40× | 1.24–1.27× |
| 16 | 5.42 | 2.65 | 1.03× | ~1.02× |

Optimum at **B = 4**, consistent with the paper's own finding that B=4 maximizes system
throughput at high concurrency. Projected end-to-end **1.28–1.31×**, bracketing the paper's
claimed "up to 40%" RL speedup.

Two effects push in opposite directions, and neither is measured:
- **In our favor:** our sampling is temp 0.9 / top-p 0.95 / **top-k 20**, versus the paper's
  temp 1 / top-k 50. Harder truncation shrinks the effective support and should *raise* τ
  materially — Table 17a's temp-0 column is ~10% higher than temp-1 at B=4 and ~31% higher at
  B=16. Our domain (DAPO-math with thinking traces) is also the paper's highest-τ domain.
- **Against us:** τ is unmeasured below 8B. A 1B model has higher next-token entropy and a
  proportionally smaller adapter. **This is the primary risk and there is no evidence either
  way in the paper or the repo.**

## 2.5 Concrete blockers, in order of engineering weight

**1. Multi-position decode is hard-disabled in the production path.** ⚠️ largest item.
`_fixed_varlen_fa4_attention` (`fast_inference.py:377-416`) hardcodes `max_seqlen_q=1` and
falls back to `sdpa_attention_forward` with a materialized mask whenever
`query.shape[2] > 1` (`:388-402`); `_hybrid_fa4_mask` (`:369-374`) returns `None` only at
`q_length == 1`. Ψ-Spec needs `q_len = B` in **both** passes, so as written every Ψ-Spec
forward abandons FA4 and runs SDPA against a sequence-major cache laid out for FA4.

*Positive finding:* the installed FA4 build supports this natively. Read from
`.venv/lib/python3.14/site-packages/flash_attn/cute/interface.py`,
`flash_attn_varlen_func` accepts `max_seqlen_q`, `seqused_q`, `seqused_k` with
`(batch, seqlen_q, nheads, hdim)` inputs and `causal=True`; varlen causal aligns the diagonal
to the end of each row's k-range, which is exactly right when
`seqused_k[row] = cursor[row] + B`. It needs **no `page_table`**, so it does not touch the
SM120 rejections recorded at `NOTES.md:5926-5929`. **The `q_len == 1` specialization is a code
choice, not a kernel limitation.** Unverified on hardware — this is experiment E0 below.

**2. Gated LoRA does not exist.** `LoRALinear.forward` (`minicpm_vapo.py:89-99`) applies
`update * self.scaling` unconditionally. Uno needs a per-position 0/1 gate so θ_AR alone
produces teacher logits at clean positions while `θ_AR + θ_Δ` produce student logits at noisy
positions — in one pass, at both training and draft time. This is a one-line change
(`update * scaling * gate`) that must default to gate ≡ 1 so the actor path is bit-identical.

**3. The rollout replica merges LoRA; a gated adapter is unmergeable by construction.**
`synchronize_fused_lora_policy_` (`fast_inference.py:196-270`) folds
`base.weight + scaling·B@A` into the fused replica weights via `_copy_merged_lora_weight_`
(`:187-193`), and hard-fails if any actor projection is not a `LoRALinear` (`:224`, `:252`).
θ_Δ must stay **unmerged**. `_FusedFirstProjection` (`:88-139`) therefore needs an unmerged
gated path with stacked-A / block-diagonal-B. Concrete shapes at r = 48:
QKV → A `(144 × 1536)`, B `(2560 × 144)`; gate/up → A `(96 × 1536)`, B `(9216 × 96)`;
`o_proj` and `down_proj` stay plain gated `LoRALinear`. Those B GEMMs are trivial and
CUDA-graph friendly; block-diagonal zeros waste 2/3 of a negligible GEMM.

**4. Two incompatible KV cache classes.** The production path uses sequence-major
`_CompactStaticLayer` (`fast_inference.py:273`) with CUDA-graph capture of a fixed-shape
1-position step (`_capture_continuous_decode_schedule` `:1770`). Ψ-Spec commits a
data-dependent number of tokens per row per cycle. `RaggedStaticCache`
(`nextlat_speculative.py:37-136`) already solves exactly that with scatter writes at per-row
cursors and `mark_static_address` annotations — but it is a *different* class with a different
layout. **Unifying them is the single largest piece of work** and is what currently makes the
NextLat benchmark numbers non-comparable to the production engine (319 tok/s B1 AR vs the
production 2,578 scheduled tok/s at B64).

**5. Tree sampler is out of reach; linear only.** Tree attention needs an arbitrary DAG mask.
FA4 exposes `mask_mod` / `block_sparse_tensors`, but `NOTES.md:5926-5929` records block
sparsity as rejected on this SM120 build. This forfeits the tree column (τ 5.97–6.11) and caps
us at linear (τ 3.89–5.42). **Acceptable**: linear B=4 is the paper's own
system-throughput-optimal configuration, and we run 64 lanes, never batch 1.

**6. Dense maximal coupling at vocab 130,560.** 64 rows × B positions × 130,560 × 4 B:
B=4 → 134 MB/tensor, B=8 → 267 MB, B=16 → 535 MB; ×3 tensors (q, p, residual) ⇒ 0.40 / 0.80 /
1.6 GB on top of the ~6 GiB KV that `release_cache` frees (`fast_inference.py:1271-1282`).
Fine at B ≤ 8 on 32 GB. Available optimization: the accept test only needs gathered scalars
`p[x]`, `q[x]`; the dense residual is needed **only for rejected rows**, so it can be
materialized lazily. `dense_top_p_probabilities` is dense-always today.

**7. Distillation memory.** `[x, z_1]` at L = 2048 is 4,096 positions; logits are
4,096 × 130,560 = 2.14 GB fp32 per tensor, and TV needs both softmaxes. Must be chunked over
positions — precedent exists: the trainer already computes frozen-head logprobs in 128-token
chunks (`train_minicpm_vapo.py:1035-1041`), which brings this to 67 MB/chunk. With existing
non-reentrant checkpointing, 32 GB is sufficient.

## 2.6 Corpus and teacher — both resolve cleanly

**Teacher = the pure frozen base, not a checkpoint.** `LoRALinear.lora_b` is zero-initialized
(`minicpm_vapo.py:74-76`), so at RL step 0 `θ_AR ≡ base` **exactly**. Distill θ_Δ against the
frozen base once and it is valid for every RL run and every variant branched off that base.

**Drift during RL is structurally bounded.** Our θ_AR moves only inside an 11.2M-parameter
rank-16 subspace of a 1.08B model. The paper's Table 8 lost only 6% TPF across 2,560 steps of
**full-parameter** DAPO on an 8B model. Rank-confined updates should drift strictly less.
This is an advantage of our setup, not a risk. [INFERENCE, but the mechanism is clear]

**Corpus: OpenThoughts3-1.2M, retokenized.** The paper's §5.2 result is precisely that θ_Δ
trained on a *different* data distribution than θ_AR still yields lossless speedups — the same
OpenThoughts corpus that *degrades* Qwen3-8B when used for SFT (Table 9) nonetheless trains a
working adapter. Reasoning traces match our math/thinking domain.
The alternative — a self-generated on-policy corpus, for which precedent exists in
`scripts/generate_minicpm_critic_corpus.py` — is **not viable**: at the measured 1,288 useful
tok/s, generating 500M tokens costs ~108 h, roughly 7× the cost of training on it.

## 2.7 Cost to a trained adapter

LoRA-only training: forward 2N + backward-through-activations 2N ≈ 4N per position, and 2
positions per supervised token (the `[x, z_1]` concatenation) ⇒ ~8N = **8.64 GFLOP per
supervised token** at N = 1.0806e9. At ~84 TFLOPS achieved (40% MFU; likely optimistic for a
1536-hidden model at seq 4096 on one 5090 — at 30% these become ~1.33× longer):

| Budget | FLOPs | Wall clock, 1× 5090 |
|---|---|---|
| 100M tokens (smoke) | 8.6e17 | **~3 h** |
| 500M tokens (pilot) | 4.3e18 | **~14 h** |
| 4.9B tokens (= UnoQwen 1 epoch) | 4.2e19 | **~6 days** |
| 14.7B tokens (= UnoQwen 3 epochs) | 1.3e20 | ~17 days |

The 1-epoch point is the sensible ceiling: Table 15 shows 1 epoch reaches 2.63 vs 3 epochs 2.71
(97%). Whether 100–500M tokens suffices for a model 8× smaller — with 10× fewer adapter
parameters to fit — is **unmeasured in the paper** (nothing below 1 epoch is reported) and is
the second decisive unknown.

## 2.8 Recommended gate sequence

All GPU work goes through `mlq submit --max-parallel-runs 1` per `AGENTS.md`. **Nothing queued;
this is a proposal.**

**E0 — δ measurement. No training. ~10 GPU-minutes. Kills or greenlights the whole idea.**
Measure `t(n)` for `n ∈ {1,2,4,8,16}` at 64 rows on the production fused replica at the real
11,024-slot cache, two arms: (a) FA4 varlen with `max_seqlen_q=n` + per-row `seqused_k` +
`causal=True`, (b) the current SDPA fallback. This also answers blocker 1 empirically.
**Gate: δ ≤ 0.15 at n = 4.** If δ > 0.25, the best achievable speedup is
`3.89 / (2 × 1.75) = 1.11×` and no τ the paper reports can pay for the work — **stop there.**

**E1 — pilot adapter. ~15–20 GPU-hours.** r = 48, α = 3072 (α/r = 64), all 7 projections,
TV-only (α_loss = 0, β = 1), lr 1e-5 constant with 2% warmup, gated LoRA, block-causal mask,
chunked-vocab TV loss, L = 2048, OpenThoughts3 retokenized, block curriculum 100M @ B=2 then
300M @ B=4. **Gate: linear-B4 τ ≥ 3.0** at temp 0.9 / top-p 0.95 / top-k 20 on held-out
DAPO-math prompts, against the theoretical floor τ = 2.0 (zero acceptance). τ ≥ 3.0 at δ = 0.11
already implies a 1.13× rollout win and validates that the recipe transfers to 1B.

**E2 — production shape.** Rollout-only gate at 64 rows / 4,096 tokens against the recorded
2,578.61 scheduled decode tok/s reference (`NOTES.md:5962-5964`). **Gate: ≥ 1.25× scheduled
decode tok/s** with distributional equivalence confirmed by the existing logit-fidelity gate
(`scripts/benchmark_minicpm_nextlat.py:197-213`) and generated-id SHA-256 comparison.

Total to a go/no-go: **~1 GPU-day**, and E0 alone (10 minutes) can end it.

## 2.9 Verdict

**Technically applicable, and unusually well-matched — but gated on one 10-minute measurement.**

In favor:
- The **entire verification half already exists and is unit-tested** (`maximal_coupling_verify`
  is literally Algorithm 1 lines 6–12; ragged per-row KV, accept/commit/rollback, bonus token,
  per-position telemetry, fidelity gate, benchmark harness).
- LoRA is hand-rolled on **exactly** the 7 projections Uno's placement ablation prescribes, and
  merging adapters into the rollout replica is already a first-class operation.
- **Rollout is 68–74% of RL step time** — the payoff multiplier is real, not hypothetical.
- Our decode step is **KV-read dominated** (9.4–17.3 GB KV vs 2.16 GB weights), which is
  precisely the regime where extra query positions are nearly free; roofline δ ≈ 0.11 is close
  to the paper's own δ ≈ 0.081–0.094 on much larger hardware.
- Distilling against the **zero-initialized-LoRA base** means one adapter serves every RL run,
  and RL drift is confined to a rank-16 subspace — strictly less drift than the 6% the paper
  measured under full-parameter DAPO.
- Our top-k 20 / temp 0.9 sampling should raise τ above the paper's temp-1 / top-k-50 figures.

Against:
- **δ is unverified on hardware**, and the production decode path is hard-specialized to
  `q_len == 1` with an SDPA fallback that would negate the FA4 win. FA4's signature says the
  needed path exists; that must be measured, not assumed.
- **τ is unmeasured below 8B.** No evidence a 1B model's draft adapter reaches τ ≈ 3.9 at B=4.
- Tree sampling is unavailable on this SM120 FA4 build → linear only, forfeiting the paper's
  best acceptance numbers.
- Unifying `RaggedStaticCache` with the sequence-major CUDA-graph-captured
  `_CompactStaticLayer` is real, non-trivial engineering.
- ~6 GPU-days for a full-recipe adapter on a single 5090.

Expected payoff if both unknowns land where the roofline says: **1.4–1.5× rollout, 1.28–1.31×
end-to-end RL step.** That clears the repo's adoption bar by a wide margin — but the repo's
current written policy against speculative decoding must be revisited explicitly, on the
grounds that it rejected a *drafter* (NextLat, KL-trained, sequential, 0.55–0.90% position-2
acceptance) and not the harness, and that L_TV is the specific mechanism that addresses that
failure.

**Next concrete action: E0.** It is 10 GPU-minutes, needs no training, no new adapter, and no
change to the trainer — only a benchmark that calls the existing fused replica with
`max_seqlen_q > 1`.

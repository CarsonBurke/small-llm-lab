# Byte Diffusion Family

## Status

The active experiment is **Byte-Duo**: a scratch-trained, byte-native,
uniform-state diffusion model with no absorbing MASK state and no preliminary
autoregressive stage. Byte-Duo is a coherent research hybrid, not an
architecture claimed by any one reference: BLT supplies its hierarchy,
Scaling-dLLMs its forward process/objective/posterior/time conditioning, and
DiffusionGemma motivates the large bidirectional continuation canvas.
Paper-driven replacements remain ablations, not promoted defaults.

Current experiment identity (recomputed and pinned again after every source
change):

- byte dataset: `data/byte_diffusion_aligned_v5`;
- dataset payload SHA-256:
  `a220aa2d8964301bcb60fc5f23c686b7a353e77baaa4e27450734f470761ddee`;
- completed checkpoints bind their exact source SHA-256 in `contract.json`;
  stale hard-coded hashes in documentation are intentionally avoided;
- current ablation code SHA-256 is emitted by
  `python3 scripts/train_byte_duo.py --print-provenance` and pinned into every
  readiness and 2,000-update job; it intentionally changes after any imported
  trainer or sampler edit;
- model parameters: 24,094,791;
- objective: exact continuous-time Duo NELBO on eight evenly dispersed
  512-atom canvases per packed page, with clean AR weight exactly zero;
- inference: exact ancestral uniform-state posterior with configurable
  reverse-grid transitions plus one exact residual-noise-removal transition;
  sampling terminal time `1e-5` is independent of schedule epsilon `1e-3`;
- current source-matched B8 readiness selects M16 at about 2.00 s/update;
  mean utilization is 99.3%, mean power is 461 W, and there are zero measured
  graph breaks or recompiles. M24 is slightly slower and M32 exhausts memory;
- latest B8 validation evidence at batch 64: 98% mean utilization and the same
  authenticated NELBO ledger as batch 24 to within 1e-6 nats/atom.

The current source-matched control's five-ledger held-out NELBO is `1.84081`
nats per atom (`2.65573` bits/atom; approximate 95% interval
`[1.83536, 1.84626]` nats/atom). Its step-2,000 fixed validation ledger is
`2.64446` bits/atom. This is a conditional-canvas likelihood
upper bound, not AR BPB. Readiness separately records a code-provenance hash that changes
with any imported training or sampling source; evidence from another code hash
is rejected.

## References

The checked-in PDFs are the immutable design references. The adjacent
`../scaling-dllms` checkout at revision
`9e09467d738cdfee44a1063a4af022c15feb9353` is executable cross-check evidence,
not a substitute for the paper.

| Reference | Local copy | SHA-256 |
|---|---|---|
| Byte Latent Transformer | `papers/byte_latent_transformer_2412.09871.pdf` | `23e0cc90e9e2ef0c11291043ca7ad4d48d5384328f31007b8b40230f370756ff` |
| Fast Byte Latent Transformer | `papers/fast_byte_latent_transformer_2605.08044.pdf` | `7957dd70eb7175a86bdcb1f8e8fca2f1444a348c0058dc924633f892d7efc8a2` |
| DiffusionGemma | `papers/diffusion_gemma_2608.00146.pdf` | `9c8e0cef08271873ff0ce560e0c6149912ce5ce2fa056394de4feec205bb2064` |
| Introspective Diffusion Language Models | `papers/introspective_diffusion_language_models_2604.11035v1.pdf` | `8b07e696da127ae41842b9b2515d7ea3a0b709448c927a04806831a6377465b2` |
| Fast-dLLM v1 | `papers/fast_dllm_v1_2505.22618.pdf` | `8dcf7152cbfb042bcd00cd74dea56fdb324a8cfe655e9182f06fa2e97e269398` |
| Fast-dLLM v2 | `papers/fast_dllm_v2_2509.26328.pdf` | `9e3dcb349e0269fee8ac8c525131284a6408e506900c69940bb898f41274d49f` |
| Scaling Beyond Masked Diffusion Language Models | `papers/scaling_beyond_masked_diffusion_language_models_2602.15014.pdf` | `c08d5d7d00a9f67b6989fdf6305fb4d729a58e5c6f77bf7e3aae9908d350f3e1` |

The papers answer different questions:

- BLT and Fast BLT define the byte hierarchy, causal entropy patching, and a
  practical small-block absorbing-diffusion reference.
- DiffusionGemma motivates one large noisy canvas per row, self-conditioning,
  and low-NFE generation, but its successful recipe is an AR conversion rather
  than scratch diffusion pretraining.
- I-DLM defines a complete strict-causal proposal/verification cell; it is not
  a large-canvas model.
- Fast-dLLM supplies block packing, complementary corruption, caching, and
  serving ideas for absorbing masked diffusion.
- Scaling-dLLMs supplies the promoted mask-free forward process, exact
  lower-variance NELBO, exact reverse posterior, and explicit time
  conditioning.

## Representation

### Atomic ids, not bits

The model head has 261 predictable categorical states:

- `0..255`: literal octets;
- `256..260`: five typed controls, including atomic EOT;
- `261`: absorbing MASK, used only by the Fast-BLT and I-DLM cells;
- `262`: inactive PAD storage, never predicted.

A categorical byte head is preferable to eight independent bit heads. It can
assign an arbitrary distribution over the 256 octets, gives registered control
events one atomic position, and avoids either an independence assumption or an
extra autoregressive network over the eight bits. A bit-factorized categorical
head would only be attractive if its measured parameter or kernel savings beat
the current 261-way head.

UTF-8 code points occupy one to four bytes. The model operates on byte
positions, not characters: `🙂` is four byte positions in a clean sequence,
but all four positions can be revised in parallel within the same diffusion
canvas. “Four bytes” therefore does not imply four serial inference forwards.
The fixed patch stride of four is compute routing and has no semantic relation
to UTF-8 width.

Newline is literal byte `0x0A`. EOT and the thinking/answer delimiters are
typed atomic controls, not byte strings searched inside user text. The
tokenizer maps literal text to strict UTF-8 without normalization and maps a
trusted control event to its registered id. Incremental decoding buffers an
incomplete UTF-8 code point and never replaces invalid bytes silently.

### Embedding and unembedding

The hierarchy uses a small learned atomic embedding at local width 256. PAD is
fixed inactive storage. Fast-BLT/DiffusionGemma cells admit MASK as an input;
Byte-Duo's `CleanAtomEmbedding` rejects MASK and accepts only the 261 clean
states plus PAD. There is no separate “mode embedding.”

The output head is an untied 261-way linear classifier over clean atoms.
Byte-Duo gives the final head a learnable class bias; both its weight and bias
are zero-initialized together with the final conditional projection. MASK and
PAD are not output classes. This matches the uniform-state reference more
closely than inheriting the older fused byte/boundary output.

The current scratch corpus has positive targets only for bytes and EOT: every
shard's special-atom count exactly equals its EOT count. Scratch Duo therefore
uses `K=257` for corruption, NELBO normalization, and sampling while retaining
the four typed-control rows in the embedding/head. Those rows are activated by
post-training only after its tokenizer inserts real typed-control targets; a
named `K=261` pretraining ablation measures the prior-mass cost of including
never-positive controls. This avoids wasting `4/261` of the full-noise prior
without sacrificing atomic controls later.

N-gram byte features remain compact and causal. They are not a tokenizer and
do not change the output normalization. Causal entropy patching uses a separate
authenticated compact entropy model, never lookahead from the target byte.
The baseline also hashes corrupted branch bytes. Because none of the references
validates random-byte n-grams, a clean-bank-only n-gram cell is an explicit
2,000-update ablation.

The current compressed n-gram design is not an exact BLT transcription. BLT
uses distinct per-order hash embeddings for orders 3 through 8 and averages
the seven embedding contributions (the byte embedding plus six n-grams).
Byte-Duo shares a low-rank table across orders and currently sums its features.
Clean-bank-only n-grams are tested before a separately named mean-aggregation
ablation, so support under heavy corruption and scale normalization are not
confounded.

## Architecture cells

The four complete cells are intentionally separate; partial hybrids are not
promoted.

| Cell | Noise/process | Attention and supervision | Role |
|---|---|---|---|
| Byte-Duo | nonabsorbing uniform-state diffusion; no MASK | causal clean prefix, bidirectional 512-atom branch, explicit time AdaLN, exact NELBO | primary scratch diffusion model |
| DiffusionGemma | uniform replacement, self-conditioning | one 512-atom branch plus a clean AR anchor | complete large-canvas conversion-style control |
| I-DLM | all-mask proposal | strict causal proposal and clean paths, exact proposal/anchor balance, fused acceptance replay | complete introspective small-stride control |
| Fast-BLT | absorbing MASK | BLT hierarchy, same-position denoising, causal AR anchor, fixed or entropy patches | paper-faithful byte reference |

The initial cells use the BLT-shaped parameter allocation: a light local encoder,
heavy 512-wide global trunk, and light byte decoder. The main model has one
local encoder layer, nine global blocks, and two decoder blocks. The clean
prefix is document-isolated and causal; a noisy branch sees that prefix and
its own branch bidirectionally, never another branch or future clean targets.

Parameter counts for the default cells are:

| Cell | Parameters |
|---|---:|
| Byte-Duo | 24,094,791 |
| DiffusionGemma | 23,043,330 |
| I-DLM | 23,017,984 |
| Fast-BLT base | 23,010,306 |

### Byte-Duo forward process

For a scratch-pretraining atom `x`, active support `K = 257`, and time `t`, the forward
distribution is:

`q_t(z | x) = alpha(t) * one_hot(x) + (1 - alpha(t)) / K`.

The schedule is `alpha(t) = 1 - (1 - eps) * t`, with `eps = 0.001` and sampled
times uniformly stratified over `[0, 1)`. Only an exact zero endpoint is lifted
to the dtype's machine epsilon to keep the continuous-time integrand finite.
A replaced state is another ordinary clean atom; it can
coincidentally equal the target. Positions can change repeatedly during the
reverse process, which preserves the reference's self-correction behavior.

Training implements the paper's exact continuous-time NELBO
integrand rather than sparse importance-weighted masked CE. The exact analytic
posterior is used at inference. Vocabulary-dense NELBO arithmetic is compiled,
and the reverse posterior has a Triton fused sample kernel. CPU property tests
compare the fused path to the transparent formula.

### Time conditioning

Byte-Duo has an explicit sinusoidal time embedding followed by a small MLP.
One packed zero-initialized projection produces independent per-layer AdaLN
shift, scale, and gate slices for the branch path. Packing preserves the
parameterization while replacing dozens of small projection launches with one
GEMM. The clean bank is time-independent and cacheable. This is
one of the principal differences from Fast BLT, whose absorbing corruption
does not require the same explicit uniform-state time parameterization.
The completed baseline uses 64 Fourier features and condition width 32. The
executable scaling-dLLMs reference uses 256 and 128 respectively. That
conditioner alone no longer fits once executable code is counted: it is about
630 KB over the 16,000,000-byte cap including the 128-KiB metadata reserve.
The shippable paper-sized ablation therefore uses an 8-global/4-decoder
reallocation and global FFN width 640. It has 25,943,305 parameters and a
current exact complete-size preflight of about 15.92 MB. A matched reallocated
cell without the large conditioner isolates its effect; an 80-wide shippable
conditioner on the default backbone supplies a second time-conditioning point.

## Training

### Eight canvases per row

Each 8,192-position packed page contributes eight systematically spaced,
document-contained 512-atom continuation canvases. Every valid position in a
canvas is supervised, so a row supplies up to 4,096 dense NELBO terms. The
clean prefix is context, not a second AR objective. At the production global
batch of 249, the current corpus geometry yields about 861,000 active diffusion
targets per update.

This retains DiffusionGemma's large-canvas compute pattern while increasing
target exposure eightfold over the original B1 prototype. B1 was too
sample-inefficient; B16 exceeded the useful memory/throughput point on the
development GPU. The B8 cell is the ablated production point, not an implicit
mix of objectives.

The global update has exactly 249 distinct pages. On multi-GPU runs the
remainder rotates between ranks without duplicate examples, and every rank
executes the same number of backward collectives. The 5090 development run
authenticates microbatches 16, 24, and 32 on the exact full update and selects
the fastest candidate retaining at least 15% (and at least 4 GiB) reserved
memory headroom. The 2,000-update gate binds that measured winner rather than
hard-coding unnecessary accumulation. The 8×H100 production run uses
microbatch 32, so each rank's 31 or 32
rows execute as one local backward pass with no gradient accumulation. The
2,000-update control
therefore sees the same selected byte corpus and page order contract as its
causal control, but not the same number or semantics of supervised targets as
the GPT-2 AR model.

### Noise-learning tradeoff

Antithetic stratification samples one distinct time for every page-branch pair
across the full optimizer update, then slices that ledger across local batches.
Strata are laid out branch-major: each B8 page receives one time from each
eighth of the unit interval instead of eight adjacent, highly correlated
times. This also prevents a local-batch-size change from changing the time
distribution.
The exact NELBO uses unchanged and
changed states correctly; it does not discard easy unchanged positions, and it
does not pretend dense reconstruction CE is a likelihood bound.

Very low noise teaches local correction but carries little generative-prior
signal. Very high noise teaches generation from near-uniform states but is
harder and higher variance. Uniform stratification across time plus the exact
lower-variance integrand is the reference solution used here. Validation fixes
row ids, origins, times, replacement draws, and targets counter-
deterministically, then repeats the final NELBO on five independent ledgers to
estimate seed variance.

Following the Scaling-dLLMs large-model recipe, 1% of training branches sample
a fresh width uniformly from 1 through 512. This preserves essentially all
dense supervision while teaching the exact short-suffix and single-fresh-slot
geometries used by variable-length and left-to-right evaluation. It is an
isolated control against an otherwise identical fixed-width run.

### No pretraining stages

There is one scratch pretraining run. It does not first learn an AR model, does
not byteify BOLMo, and does not perform a DiffusionGemma-style conversion.
Sampler distillation, supervised post-training, or RL would be later,
independently measured phases. They are not called “stage 2 pretraining.”

## Inference

Prompts are always clean. Evaluation never corrupts or noises prompt atoms.
Only newly allocated suffix positions start from the uniform clean-state prior
and undergo reverse diffusion.

The number of diffusion steps is a call-time setting, not baked into the
checkpoint. Eight steps was only a throughput readiness point, not a justified
quality setting. Scaling-dLLMs defaults to 1,000 steps; pre-distillation
DiffusionGemma evaluates up to 192. The
sampler performs `steps + 1` denoiser evaluations because the configured lower
endpoint retains a small residual noise level; the final exact posterior moves
to `alpha = 1`. The current schedule is fixed per request rather than adaptive
per row. Adaptive stopping requires its own quality/speed ablation.

Every suffix position stays revisable on every transition. Optional traces
retain the initial uniform canvas and every posterior state, including the
final residual-noise-removal step. They also record the exact `(t, s)` and
`(alpha_t, alpha_s)` endpoints, committed atom positions and IDs, literal bytes
retained by each canvas, stop-trimmed returned bytes, termination, and the
clean overlap carried into the next canvas. This makes a GSM8K failure
auditable through both denoising and canvas commitment.

The clean prefix encoder/global/decoder bank and static canvas attention
layouts are prepared once per canvas. Reverse steps run only the branch against
that cache. Committing a canvas changes clean context and requires a new clean
bank for the next canvas.

Generation also exposes a semi-autoregressive commit-width control inspired by
the block/LTR schedules in Scaling-dLLMs. The model still denoises its trained
512-atom canvas, but only the earliest `k` new atoms are committed; discarded
lookahead is regenerated after those atoms join the clean prefix. This tests
whether joint 512-atom commitment is the quality bottleneck without changing
training or treating an out-of-distribution narrow canvas as equivalent. It
trades additional canvases and NFEs for a shorter dependency horizon.
Separately, `visible_width` limits the actually visible noisy suffix. Setting
both visible and commit width to one gives a true clean-prefix plus one-fresh-
atom geometry with no discarded noisy lookahead; the 1% variable-width
training exposure keeps that path in distribution.

### What currently limits generation

Independent code/paper audits found no error in the exact posterior, time
direction, prompt isolation, or NELBO formula. The material mismatches are:

- the baseline is scratch-trained for only 2,000 updates, while the scaling
  study's IsoFLOP compute-optimal analysis reports roughly 23 times the AR
  compute for Duo to match AR likelihood; this is context, not a prescription
  to train this fixed model exactly 23 times longer;
- the original GSM run used eight transitions over a jointly revised
  512-byte suffix. On the fresh seed-0/256-row curve, 8 to 192 transitions
  improves frozen-external sample BPB from `4.225` to `3.736`, but every point
  has zero exact and zero parsed GSM answers; invalid UTF-8 is non-monotonic;
- paper GSM numbers use five-epoch supervised tuning on 385k augmented GSM
  examples and left-to-right generation, so they are not comparable to this
  base-model five-shot test;
- high-noise denoising is the clear bottleneck: at `t in [0.875,1]`, clean
  target CE is `4.537` bits/atom, target probability is `7.08%`, and accuracy
  is `17.26%`;
- the global trunk owns about 80% of parameters while the local decoder owns
  about 6.5%, and inference can place 0--3 fixed clean phase bytes inside a
  noisy branch although the baseline training topology never does;
- FP32 fused posterior sampling is faster, but scaling-dLLMs uses FP64. A
  matched FP32/FP64 oracle sweep measures whether precision affects quality.

Candidate isolated 2,000-update ablations are: reference-sized time conditioning
with required capacity compensation; the named `joint_nelbo_clean_ar`
objective; matched 9-global/2-decoder to
8-global/4-decoder reallocation controls; random
0--3-byte fixed-prefix phase exposure; clean-bank-only n-grams; `K=257` versus
`K=261` prior support; and 1% variable-width versus fixed-width training. Each has
its own VRAM, utilization, source, dataset, and inference-readiness evidence.
The completed fixed-stride, gated-projection Fast-BLT-shaped control
underperformed the matched causal control at 2,000 updates. It is not the
paper-faithful entropy-patched split-cross-attention result; that complete cell
still requires its own block-size, confidence, and entropy-bounded ablations.
Complete I-DLM and dense-x0/self-conditioning cells remain separate controls
rather than being spliced into Duo without an ablation.

The decoder-reallocation control is now fixed as 8 global blocks, 4 decoder
blocks, and decoder FFN width 688, with every other `ByteDiffusionConfig` field
unchanged. It has 24,078,409 parameters versus 24,094,791 for the retained
9-global/2-decoder control, a difference of -16,382 (-0.068%). This is the
closest tensor-core-aligned under-budget decoder FFN width and keeps the
comparison focused on where capacity is spent. Its hypothesis is that the
decoder's roughly 6.5% share of the retained parameter budget is insufficient
for the measured high-noise denoising bottleneck. It uses the same canonical
512×8 validation, pure NELBO objective, data, optimizer, schedule, and sampler.
Cross-run promotion requires a five-ledger improvement greater than 0.005
bits/atom; failed likelihood arms do not receive GSM/NFE follow-ups.

The joint-objective cell changes no architecture, data, corruption, schedule,
or sampler setting. It optimizes the fixed equal-weight sum of two separately
global-normalized terms: exact Duo NELBO over active corrupted atoms, plus
clean causal next-atom and document-BOS NLL over the existing 8,192-atom clean
bank. The pure-Duo default remains `pure_nelbo` with weight zero and retains
its no-clean-unembedding training path. Both the CLI flag
`--objective joint_nelbo_clean_ar` and environment setting
`BYTE_DUO_OBJECTIVE=joint_nelbo_clean_ar` select the ablation; no arbitrary
loss coefficient is exposed. Training and validation telemetry report the
NELBO, clean-AR NLL/BPB, and their joint optimizer value separately. Readiness
evidence binds the objective name, fixed weight, and both normalization rules,
so a pure-NELBO benchmark cannot authorize the joint 2,000-update job.

During training, a persistent single-worker CPU producer prepares whole
updates, pinned host buffers and one persistent CUDA transfer stream overlap
their H2D transfer with GPU execution, and `record_stream` protects their
lifetime. At yield `N`, H2D for update `N+1` is already enqueued and CPU
preparation for `N+2` is submitted. Checkpoints commit only the consumed
update's explicit cursor and corruption-generator snapshot, so speculative
prefetch cannot advance resume state.

### Tail, EOT, and overflow

Packed training pages preserve physical document boundaries and add only
attention-excluded PAD. A selected canvas never crosses a document. At serving
time a prompt ending mid-patch keeps its already committed phase clean and
diffuses only the fresh suffix of the aligned canvas.

EOT is one predicted atom. If it appears inside a canvas, later sampled slots
are discarded semantically. A literal byte limit and UTF-8 display limit are
tracked separately. Raw evaluation may end on an incomplete code point and
reports invalid UTF-8; a text adapter may hide that incomplete display suffix
but must report the discarded bytes.

Causal-entropy Fast-BLT serving recomputes the exact clean patch topology, then
generates and commits a whole fixed-size diffusion block. The block may cross
several entropy patch boundaries; only EOT truncates its semantic suffix. The
authenticated patcher's `max_patch_size` bounds physical patch pooling and is
independent of the diffusion horizon. B4, B8, and B16 recipes therefore use
2,048, 1,024, and 512 branches respectively to preserve 8,192 branch byte-slots
per page. Serving requires B to match the checkpoint's trained canvas, not the
patcher's maximum size.

## Data and tokenization

The canonical 2,000-update source mix is selected in literal UTF-8 bytes and
includes MathGLM v6. Its weights include 6% MathGLM, 4% math drills, 8%
OpenMathInstruct, 6% FineMath 4+, 4% OpenWebMath, and 2% DeepMind Math, along
with web, code, Wikipedia, arXiv, and synthetic textbook data.

`data/byte_diffusion_aligned_v5` is a packed byte-native view of that selected
stream. The nanoGPT comparator uses
`data/datasets/bd_mathglm_v6_gpt2_2k_v4`, a GPT-2 token view of the same
selected documents. Its manifest explicitly records
`document_selection_changed=false` and
`selection_unit=utf8_bytes`. Thus “2k corpus” means enough deterministic source
for 2,000 matched updates; it is tokenized separately for each model only
after document selection.

## Metrics

No single BPB number is valid for every cell.

- **AR challenge BPB** is normalized causal next-token codelength divided by
  literal source bytes. It is the primary Parameter Golf metric for an AR
  model. The latest nanoGPT control reaches 1.3301 at update 2,000 on its fixed
  proxy ledger.
- **Byte-Duo conditional-canvas NELBO** is a likelihood upper-bound proxy in
  nats or bits per supervised atom. It is valid for controlled comparisons
  within this exact forward process. It is not teacher-forced AR BPB and is not
  directly rank-comparable to nanoGPT's 1.3301.
- **Dense denoising CE** for DiffusionGemma is an optimization diagnostic, not
  BPB and not an ELBO.
- **Fast-BLT absorbing ELBO proxy** is only comparable within the exact
  absorbing estimator and patch policy.
- **Generation quality** is GSM8K exact match plus parsed-answer rate, UTF-8
  validity, EOT/stop behavior, repetition, and sampled outputs.
- **Generation speed** reports model forwards, NFE, trajectories/s, literal
  bytes/s, requested atom slots/s, batch size, prompt length, and stopping
  policy. Byte atoms and GPT-2 tokens are never relabeled as the same unit.

The NELBO proxy is reliable enough to compare checkpoints of this Byte-Duo
recipe when evaluated on a frozen large ledger and repeated independent ledger
seeds. It is not reliable as an absolute cross-family language-quality score:
the references show likelihood can mis-rank different diffusion processes,
and the conditional canvas is not a whole-sequence normalized AR factorization.
GSM8K and sampled-text audits therefore remain promotion evidence.

The console and TensorBoard paths now use only
`val_diffusion_nelbo_bits_per_atom`; they do not publish the value under
`val_bpb` or `val_proxy_bpb`. Structured Duo metrics additionally set
`nelbo_is_ar_bpb=false`, and the derived result leaves AR BPB null.

## Existing controls

The exact current-data causal byte control is
`ablation_results/bd_causal_c0_mathglm_v6_2k_v3`:

- 2,000 updates;
- proxy BPB 1.805144;
- measured training time 2,270.55 seconds;
- total wall time 2,583.60 seconds.

The latest nanoGPT current-data control is
`ablation_results/nanogpt_latest_mathglm_v6_matched_2k_v7`:

- final fixed-ledger BPB 1.3301;
- average update time 1,781.27 ms;
- measured training time 3,690.05 seconds;
- total wall time 3,843.68 seconds;
- fresh 5-shot greedy GSM8K exact match over seeds 0/1/2: `1/1319`, `2/1319`,
  and `2/1319`, for mean `0.1264%`;
- parsed-answer rates: `4.62%`, `14.71%`, and `11.14%` (mean `10.16%`).

The source-`923680b6…` pure Byte-Duo 48-step run is `0/1319` exact for all
three seeds and only `1/3957` responses parse at all. Under the same frozen
causal-byte sample scorer, Duo is `3.8052` bits/byte versus nanoGPT `0.9182`;
this is an external
GenPPL-style sample-quality measurement, not either generator's own BPB.

Historical topology controls from `byte_duo_control_current_2k_v3` (source
`c1c5f667…`) found that visible-1/commit-1 decoding emits EOT almost
immediately, while commit widths 32 and 128 still produce zero parsed answers.
These are useful diagnostic observations, not source-matched controls for the
`923680b6…` pure run. They suggest that the failure is not explained by
committing a 512-atom canvas alone: strict causal continuation exposure and
near-prior denoising remain plausible bottlenecks.

Two source-matched 2,000-update ablations are rejected:

- equal-weight clean-AR supervision improves its auxiliary atomic-target
  anchor but regresses five-ledger NELBO by `0.0945` bits/atom, external sample
  BPB by `0.1480`, invalid UTF-8 by `6.015` percentage points, and remains
  `0/3957` exact;
- removing n-grams only from corrupted branches regresses five-ledger NELBO
  from `2.6557` to `2.7349` bits/atom (`+0.0791`). It failed the primary gate,
  so its expensive GSM/NFE/commit-one follow-ups were cancelled before start.

The completed next isolated cell is a 256-atom native diffusion canvas with 15
branches per page. This follows DiffusionGemma's actual 256-token block more
closely and keeps gross dense supervision within 6.25% of 512×8. Headline
likelihood stays on the frozen canonical 512×8 ledger; native-256 likelihood
and generation are available diagnostics but were not run after this candidate
failed the canonical promotion gate. Entropy patching, schedule changes, and
objective changes are not combined with this cell. Its fixed canonical
step-2,000 NELBO is `2.645456` bits/atom, only `0.000995` worse than the
historical source-`923680b6…` 512×8 control (`2.644461`). Its independent
five-ledger mean is `2.657177` bits/atom (approximate 95% corruption-ledger
interval `[2.648744, 2.665610]`), `0.001451` worse than the historical pure
control. This does not meet the `>0.005` improvement needed to justify a new
same-source 512×8 control, so the geometry is not promoted. Training readiness selected microbatch 16 at
`2182.4` ms/update, `99.38%` mean utilization, and `441.1 W`; microbatch 24 was
slightly slower and 32 OOMed. Fixed-compute inference reached `85.17`
trajectories/s and `43,606` requested atoms/s, slower than the historical
512×8 readiness (`120.02` trajectories/s and `61,449` requested atoms/s)
because a 512-atom continuation needs two or three 256-atom canvases.

This cell uses a predeclared futility screen because failed 2,000-update runs
need not consume the full queue. No decision is made before update 800: both
rejected source-`923680b6…` arms looked transiently better at update 400. The
candidate is stopped at update 800 if its canonical fixed-ledger NELBO is at
least `0.025` bits/atom worse than the `923680b6…` pure-control curve, or at
update 1,200 if it is at least `0.015` worse. Those cross-revision comparisons
can only terminate a clearly futile run; they cannot promote one. A surviving
candidate must finish 2,000 updates and improve by more than `0.005` bits/atom.
If it does, a fresh 512×8 control is trained under the candidate's exact source
hash before any causal attribution or promotion claim.

The nanoGPT GSM result is poor but real: this is a 2,000-update base model, not
a post-trained math model. Byte-Duo must use the same GSM snapshot, prompts,
shots, seeds, output budget, and answer parser. Its native reverse process is
categorical while nanoGPT's is greedy, so this is a task-and-scoring-matched
native-decoder comparison rather than a distribution-matched decode. The
current Duo RNG stream is fixed-batch reproducible but not row-invariant under
rebatching; requested batch size 32 and all three generation seeds are therefore
part of the full-GSM evidence contract. Batch size eight applies to the sampled
trace and fixed-compute readiness ledgers, not the full GSM evaluation.

The AR systems reference processes 524,288 GPT-2 tokens/update in 662.94 ms,
790,853 tokens/s, at 505.44 W mean power and 99.27% mean utilization. It is a
training-throughput reference, not a generation benchmark.

### Next complete paper cells

The next Fast-BLT control is capacity-matched rather than assembled from the
individually tested partial features. It uses one encoder block, eight global
blocks, two decoder blocks, FFN widths `512/600/512`, distinct rank-16 hash
tables for orders 3 through 8 with `/7` mean aggregation, split decoder
cross-attention, causal entropy patches, Bernoulli absorbing corruption, and
the paper-sum AR plus denoising objective. This configuration has 23,010,304
parameters, two fewer than the 23,010,306-parameter Fast-BLT base. Removing
one 704-wide global block saves 2,130,944 parameters and narrowing the eight
remaining global SwiGLUs from 704 to 600 saves 1,277,952, exactly compensating
for the 3,408,894 parameters added by per-order n-grams and split attention.
The diffusion block length is independent of the entropy patcher's maximum
patch size; B4, B8, and B16 use 2,048, 1,024, and 512 branches respectively so
each row retains an 8,192-byte branch budget. Uniform group-32 int4 is the
preferred export preflight: the current complete-size estimate is 15,246,952
bytes, leaving 753,048 bytes under the 16 MB cap. This remains a compressed
paper-faithful control because its n-gram tables and count-based entropy model
are much smaller than the reference systems. It requires a dedicated preset,
real-topology readiness support, and cached split-attention conditions before
its 2,000-update ablation.

DiffusionGemma's next implementation is inference-only persistent hierarchical
prefix caching, not another training-objective hybrid. For prompt length `L`,
prefill through `floor(L / 4) * 4`; carry the remaining 0--3 prompt bytes as
fixed atoms at the start of the first canvas. Cache rotated clean K/V for the
local encoder, global transformer, and local decoder, plus normalized patch
states. Each denoising transition then computes only the dynamic canvas, and a
full finalized canvas is appended once only when another canvas is required.
For canvas 512 and 12 transitions, the transformer-compute upper-bound ratio
is about 3.75x at a 2,048-byte prefix, 5.4x at 4,096, and 7.65x at 8,192;
attention reads and weight traffic make realized speedups smaller. This is an
exact BLT-hierarchy adaptation of DiffusionGemma's cache semantics, not a
literal Gemma reproduction. The generic BDI4 container can be reused, but the
cell needs a separate exporter, provenance contract, self-conditioning cache
parity test, and post-quantization denoising/AR-anchor diagnostics. Denoising
CE must never be relabeled as BPB.

## Memory, export, and readiness

The completed group-32 int4 Byte-Duo artifact is `13,584,549` model bytes and
`15,576,571` bytes with its exact counted executable closure, leaving
`423,429` bytes. It improves on group 64 by `0.00768` nats/atom, but its
post-quantization NELBO is still `1.87978` nats/atom versus `1.83321` for the
matched float ledger, a material regression that must not be hidden. The entropy Fast-BLT artifact embeds
its authenticated 1,069,713-byte patcher; measured production accounting is
14,445,439 total bytes including 1,132,603 code bytes, leaving 1,554,561 bytes.
The parser authenticates patcher SHA/configuration and payload contiguity.

Training is not allowed to start merely because it fits memory. The readiness
report must authenticate source, data, model, objective, global batch,
microbatch, validation batch, cadence, compilation, and world size. Steady
training updates must meet:

- mean power at least 425 W and p10 power at least 400 W;
- mean GPU utilization at least 90% and p10 at least 80%;
- at least three telemetry samples;
- reserved-memory headroom of at least 15% of VRAM and at least 4 GiB;
- no measured graph breaks, recompiles, or new graphs.

Periodic validation uses the same utilization, telemetry-count, headroom, and
graph-stability gates, but treats watts as diagnostic. Its measurement window
is short enough that power ramp-up makes an absolute watt floor noisy even
while every sample reports 100% GPU utilization.

The power floor is an idle/host-bound regression guard, not the optimization
target: update latency and useful targets/second are primary. The source-matched
5090 run is attention/memory-bound at about 99% utilization and 440--470 W, so
the old absolute 500 W rule rejected faster execution and even a fully batched
validation pass. The AR cell's 505 W remains a useful comparative signal.

Retained 512×8 inference readiness separately covers all four prompt phases at
batch eight, prompt length about 2,048 bytes, canvas 512, eight diffusion steps,
and twelve repetitions. Geometry screens use their native canvas while retaining
a fixed 512-atom requested continuation. Readiness requires the fused Triton posterior, telemetry coverage,
headroom, and stable compiled graphs. It deliberately does not require a
saturated GPU: this is an online-latency geometry, where enforcing high power
or utilization would reward a larger batch instead of a faster trajectory.

## Experiment sequence

1. Pass strict Byte-Duo training/validation readiness and inference readiness.
2. Run exactly 2,000 updates through the standard ablation runner with frozen
   source/data hashes: the readiness-selected microbatch (currently M16) on the 32 GiB development
   GPU and a single M32 local batch on each 8×H100 rank.
3. Evaluate conditional-canvas NELBO on at least five independent 2,048-row
   ledgers.
4. Run 5-shot GSM8K over seeds 0, 1, and 2 with categorical Duo sampling and a
   separate small traced subset retaining every reverse state.
5. Benchmark a 512-atom budget and a 200-atom trajectory against the latest
   nanoGPT incremental decoder at matched batch/concurrency and prompt length.
6. Only then consider extra canvases per row, alternate NFE, adaptive stopping,
   distillation, SFT, or RL as isolated ablations.

Broad CPU tests and focused mathematical oracles gate source changes. CUDA
tests, readiness, training, and evaluation are always submitted through
`mlq`.

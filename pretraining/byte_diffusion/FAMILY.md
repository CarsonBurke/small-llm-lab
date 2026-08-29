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

Executable reference checkouts are pinned beside this repository. They are
read-only design evidence and are not imported by the training package:

| Reference implementation | Local checkout | Revision | Scope |
|---|---|---|---|
| Meta BLT | `../blt` | `9774ed4fcc78313f9f218295f3d7e4decdadf2ae` | Official entropy patcher, byte/patch hierarchy, pooling, and local cross-attention |
| I-DLM | `../I-DLM` | `a23c1a12ef997c7f3ad616b25bcfb62db39ded68` | Official all-masked causal training and introspective serving implementation |
| Fast-dLLM v1/v2 | `../Fast-dLLM` | `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b` | Official prefix/dual cache, confidence-parallel decoding, and hierarchical block cache |
| Scaling-dLLMs | `../scaling-dllms` | `9e09467d738cdfee44a1063a4af022c15feb9353` | Official masked, uniform-state Duo, and interpolating training/sampling code |
| DiffusionGemma Transformers | `../transformers-diffusiongemma` | `0cdd8a1908949037cf7718769ffc03dbdf9d9fd6` | Paper-designated reference model and generation implementation |
| DiffusionGemma vLLM | `../vllm-diffusiongemma` | `ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9` | Paper-designated optimized serving implementation |

As of 2026-08-13, Fast-BLT does not publish BLT-D/BLT-DV source. The
official Meta BLT checkout predates the Fast-BLT paper and contains no BLT-D
implementation. Fast-BLT fidelity must therefore be checked against the PDF
plus executable BLT primitives rather than claimed from unavailable code.

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

### Executable-reference decision

The checked-out implementations rule out treating one hybrid as if it copied
all references. The scratch primary cell uses the following explicit
copy/adapt/reject policy.

Copy from Scaling-dLLMs/Duo:

- one categorical state per mutable byte or typed control;
- uniform-state corruption, continuous time, exact lower-variance NELBO, and
  the exact reverse posterior;
- per-block and final zero-initialized AdaLN modulation; and
- an untied, zero-initialized categorical output head.

Adapt from BLT and Fast-BLT:

- entropy-patch only immutable clean bytes;
- keep the clean local encoder continuous and causal across patch boundaries;
- pool each clean variable-length patch once and use sequential latent
  positions in the global transformer;
- keep each noisy future byte at full resolution, with its original byte
  position, instead of sending noisy bytes through the patch pool/global
  hierarchy;
- let the noisy block attend bidirectionally within itself, causally to clean
  prefix state, and by split cross-attention to the preceding clean latent;
  and
- incrementally cache closed clean patches while carrying and recomputing only
  the final open clean patch across commits.

Keep as separate complete controls:

- Fast-BLT's absorbing MASK, clean causal next-byte loss, and `1/t`-weighted
  masked reconstruction;
- DiffusionGemma's dense clean-target CE, 50% self-conditioning, uniform
  replacement sampler, and conversion/post-training pipeline; and
- I-DLM/Fast-dLLM's causal proposal, introspective verification, and
  AR-compatible caches.

Reject in the primary cell:

- fixed or overlapping latent patches on mutable bytes;
- target-derived entropy boundaries on noisy bytes;
- boundaries recomputed from corrupted bytes;
- causal n-gram features over corrupted garbage;
- a serving-only prompt-phase workaround that training never sees; and
- sampler mechanisms borrowed from a different training objective without
  their matching training procedure.

This is an informed adaptation, not an exact reproduction. Scaling-dLLMs and
DiffusionGemma use flat full-resolution backbones; Fast-BLT keeps the noisy
block only in its local decoder and validates block sizes 4, 8, and 16. The
hybrid retains BLT's efficient clean hierarchy but gives a large mutable byte
canvas a capacity-matched full-resolution decoder. Block width remains an
ablation variable rather than assuming that 512 is optimal.

“Byte carry” must not conflate two mechanisms. BLT's official implementation
keeps causal local-byte context continuous across patch boundaries and seeds
its local decoder with local-encoder states. Fast-BLT explicitly changes the
diffusion decoder input to a fresh embedding lookup over the concatenated clean
and corrupted byte sequences. The faithful Fast-BLT control therefore uses
fresh decoder embeddings; an encoder-to-decoder skip is a separately named
BLT/U-Net ablation, not silently mixed into it.

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

Executable-source inspection sharpens an important limitation of this layout.
No reference groups a large mutable diffusion canvas into disjoint fixed
four-byte patches:

- BLT entropy-patches clean bytes and pools only bytes assigned to each
  variable patch;
- Fast-BLT keeps its B4/B8/B16 corrupted future block in the local decoder;
  its unavailable BLT-D source cannot justify a different implementation;
- DiffusionGemma recomputes every canvas token independently with the full
  shared backbone against a read-only causal prefix cache;
- Scaling-dLLMs applies a flat bidirectional DiT directly to tokenizer-token
  positions; and
- I-DLM avoids the issue with strict token-causal proposal topology.

Byte-Duo's fixed stride four is therefore an efficient large-canvas adaptation,
not a paper mechanism and not a UTF-8 boundary. It does not cut local byte
communication at four-byte edges: the branch local encoder and decoder each
attend bidirectionally across the complete 512-byte branch. The stride affects
only how those already contextualized byte states are compressed into the
global route and assigned back to decoder positions.

An eight-byte-overlap/stride-four proposal was rejected before integration.
It would have changed compression and unpooling without restoring any missing
local byte continuity, and no reference uses that construction. The two
reference-supported alternatives are structurally different: BLT uses causal
dynamic patches for the clean sequence, while Fast-BLT keeps its consecutive
noisy future block out of the local encoder and global patch transformer and
runs it directly through the full-resolution local decoder conditioned on the
preceding clean latent. DiffusionGemma and Scaling-dLLMs instead use a flat
full-resolution backbone with no patch lattice. These alternatives must be
tested as complete cells rather than approximated by overlapping fixed groups.

The ablation order isolates these claims. First, keep the authenticated v5
corpus, fixed clean-prefix hierarchy, corruption ledger, optimizer, and
8-global/4-decoder allocation, but bypass pooling/global processing for the
mutable branch. This tests only mutable topology. Next, if it passes, replace
the clean fixed lattice with authenticated causal entropy patches and
document-local patch-ordinal RoPE. Only then reallocate additional global
capacity into the full-resolution decoder and sweep block width. The final
design is not allowed to retain the fixed clean lattice merely because the
first topology control does.

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
shift, scale, and gate slices for the mutable branch path. The legacy topology
modulates encoder, global, and decoder blocks because mutable states traverse
all three. The full-resolution topology emits only decoder and final-norm
slices: its clean encoder/global hierarchy is time independent, so allocating
those modulation rows would create about 0.86M dead parameters. Packing the
live slices replaces small projection launches with one GEMM. The clean bank
is time-independent and cacheable. This is
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

The promoted full-resolution mutable-topology control uses one clean encoder,
six clean-global blocks, eight byte-decoder blocks, decoder FFN width 880, and
24,025,677 parameters. Mutable bytes remain at all 512 byte positions through
all eight decoder blocks; they are never pooled into a 128-state stride-four
lattice. Only the immutable clean bank is pooled. Branch-constant preceding
latents are projected once per branch and decoder layer, then broadcast over
the 512 bytes; inference caches those projected conditions across denoising
transitions. Random-phase training and a separate exact-byte-origin serving
ledger close the former aligned-origin train/serve mismatch.

This control completed 2,000 updates with an online final NELBO of `2.510111`
bits/atom in 6,483.73 seconds. Its five-ledger canonical mean is `2.533413`
bits/atom with approximate 95% interval `[2.528974, 2.537853]`; the exact-byte-
origin serving mean is `2.534914` with interval `[2.530762, 2.539066]`. The
previous pooled 8-global/4-decoder result was `2.569715` on its five-ledger
canonical evaluation, so the improvement is about `0.0363` bits/atom and is
well beyond ledger noise. The tradeoff is measured training time: 6,483.73
seconds versus 5,082.60 seconds for that pooled predecessor, about 27.6% slower.
Eight-step categorical GSM8K remains `0/3957` exact with only two parsed
answers, so better conditional likelihood has not yet produced mathematical
reasoning at this pretraining scale.

The next isolated ablation changes only the immutable clean-bank patch policy
from fixed groups of four to authenticated causal-entropy patches of length
one through eight, targeting mean length four. It retains the exact same
24,025,677 trainable parameters and full-resolution mutable canvas. Patches
use document-local ordinal positions, add no virtual BOS latent, and expose a
patch latent to a clean byte only when that byte closes the patch; otherwise
the byte sees the preceding closed patch or the exact zero prior. An arbitrary
mutable origin is conditioned on its preceding closed patch. A prompt ending
inside an open patch retains those clean bytes as direct decoder K/V but
withholds the unfinished patch latent. Thus entropy patching changes clean
compression boundaries without reintroducing mutable-byte pooling or target
lookahead.

The fresh same-source 9-global/2-decoder pure-Duo control was terminated at
step 630 rather than completed. At the last shared checkpoint, step 600, its
fixed validation ledger was `3.162384` bits/atom versus `3.061337` for the
8-global/4-decoder candidate, a candidate delta of `-0.101047`. This is strong
early rejection evidence under the run-killing policy, but it is not a
2,000-step or five-ledger comparison and must not be reported as one.

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

The clean-entropy ablation uses `data/byte_diffusion_entropy_v6_b4`, payload
SHA-256 `c9f57d27f0910f0751bc9c5b0989203ee6a668f80ce72bd09febdd15ff45772b`,
and patcher SHA-256
`1409df540343409c9186d8f4ec674636d159f74e3a72af214e6d9b991cb2472c`.
It contains the same selected 4,194,303,930 training atoms and 3,263,762
documents, including MathGLM v6, but repacks physical pages so an entropy patch
does not cross an 8,192-byte page cut. That removes artificial continuation
padding and is semantically preferable, but it means a v6-versus-v5 training
delta is not a perfectly isolated causal estimate. A marginal result requires
a paired logical-coordinate ledger or a fresh fixed-patch control on the v6
packing before attribution.

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
individually tested partial features. The closed
`fast_blt_entropy_b4_complete_g6d8_v1` cell uses one encoder block, six global
blocks, eight decoder blocks, FFN widths `512/512/680`, distinct rank-16 hash
tables for orders 3 through 8 with BLT-prime hashing and `/7` mean aggregation,
and split decoder cross-attention before every decoder layer. It has 24,013,824
parameters, 11,853 fewer than the retained 24,025,677-parameter G6/D8 Duo.
The two compressed departures from BLT are explicit: 8,192-entry rank-16
per-order tables instead of full-width 500k tables, and the authenticated
count-based causal entropy model instead of BLT's large neural entropy model.

The training population is ragged, not a rectangular `B4×2048` bank. For each
clean row, every physical entropy patch start with a local preceding latent is
one four-byte block. The first physical patch of a document is valid because
its preceding latent is the local virtual-BOS patch; only a continuation-page
start whose prior latent lives on the previous page is omitted. A block may
cross any number of entropy patches and may split a UTF-8 code point. If it
extends past the finite 8,192-byte training sequence, the available
same-document prefix is supervised and the rest is PAD, matching Fast-BLT's
finite-sequence rule. Clean bytes remain one continuous causal stream across
patch boundaries. Every row draws one `t`; all block bytes use independent
Bernoulli masks at that `t`; the objective is the paper's summed clean CE plus
`1/t` masked reconstruction sum, followed by a row mean. There is no origin
sampling or importance weight.

The implementation uses compact segmented Perceiver pooling, one shared clean
byte K/V bank, exhaustive flat branch storage, and cached two-key split
conditions. Training and validation group rows by measured physical work
`8192 + 4M`, not by a fictional fixed branch count. Exact cumulative grouping
and real-group splitting keep DDP backward-call counts equal without dummy
supervision. Readiness must authenticate the actual per-update origin counts,
branch atoms, microsteps, physical work, mask-construction cost, graph reuse,
VRAM, utilization, and power before the 2,000-update ablation.

That complete D4 result is the decision point for further patching work. The
next patch ablations, each separately named and holding the complete cell
fixed, are: (1) project entropy cuts onto valid UTF-8 code-point starts, using
the last complete-codepoint boundary before max size and a raw-byte fallback
for invalid streams; (2) BLT monotonic/jump entropy boundaries; (3) the
official BLT BPE and whitespace/static patchers. UTF-safe cuts are not a
reference mechanism and must not be mixed into the first Fast-BLT result.
Likewise, a 512-byte decoder-prefix window is a separately named systems
adaptation if the paper-faithful unbounded prefix is too slow; it must not be
silently substituted into the complete cell. B8 and B16 follow only after D4
establishes whether the full mechanism is useful.

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

The completed historical group-32 int4 Byte-Duo artifact is `13,584,549`
model bytes and `15,576,571` bytes with its then-current executable closure,
leaving `423,429` bytes. It improves on group 64 by `0.00768` nats/atom, but
its post-quantization NELBO is still `1.87978` nats/atom versus `1.83321` for
the matched float ledger, a material regression that must not be hidden.

The current 6-global/8-decoder clean-entropy preflight is `14,768,783` complete
bytes including the exact executable closure and a 128 KiB reserve, leaving
`1,231,217` bytes below the 16 MB cap. The authenticated 1,069,713-byte entropy
patcher is stored losslessly as a 492,660-byte zlib stream. Loading is bounded
and rejects corrupt, truncated, trailing, or oversized streams; the raw SHA,
configuration, schedule epsilon, patch policy, and atomic manifest are all
bound. A dedicated Duo serving/GSM entry point keeps independent I-DLM,
DiffusionGemma, JEPA, post-training, and nanoGPT stacks out of the counted
deployment closure rather than hiding them with accounting exclusions.

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
   source/data hashes: the readiness-selected microbatch on the 32 GiB
   development GPU and an independently qualified local batch on each 8×H100
   rank.
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

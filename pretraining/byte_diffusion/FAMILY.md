# Byte Diffusion

## Status

Implemented, correctness-gated family. No BPB or generation-quality result is
claimed yet: the required 2,000-update ablations have not run. Engineering
measurements below are queued RTX 5090 benchmarks, not paper results or quality
evidence. The model is initialized from scratch and pretrained natively with
diffusion masks; it is not a conversion of a BOLMo checkpoint. The completed
`bolmo_srcopt_cont_stage2` run is retained only as a byte-accounting, quality,
and hardware calibration reference:

- checkpoint: `logs/bolmo_armC_srcopt_cont_final_model.pt`
- SHA-256: `8df83ed93485e0b4fd118ff687ba49cd3c8ba2127714d8e41c48627e19c75cb7`
- reported canonical marginalized byte score: 1.267235; this uses
  one-byte-lookahead routing and is not a normalized causal codelength
- canonical valid joint byte-and-routing codelength: 1.395972 BPB
- strict-prior causal-routing diagnostic: 3.461784 BPB; this is a severe
  train/eval mismatch, not a quality target
- atomic vocabulary: UTF-8 bytes `0..255`, five atomic source specials
- current size: 50,433,758 parameters, of which 25,756,160 are the retained
  source-token embedding

BOLMo supplies the byte representation and evaluation contract, not weights or
architecture. The local mLSTMs, KDA trunk, retained source-token embedding,
learned patching, fused byte/boundary head, losses, and two-stage byteification
procedure are absent from the new model.

The current implementation closes the architecture, data, training, inference,
and export loop:

- exact default size: 23,011,584 parameters;
- complete signed-int4 artifact: 12,590,075 bytes including 348,797 counted
  code bytes, leaving 3,409,925 bytes below the 16,000,000-byte cap;
- canonical v3 builder/loader artifacts are SHA-verified, lazily shard-cached,
  isolate one document segment per physical row, and retain short documents
  and final tails for AR/BPB while diffusion uses PAD-excluded short blocks;
- the virtual leading EOT/BOS predicts every document's first byte without
  shifting its fixed four-byte patch phase;
- validation reports normalized total codelength per scorer-accounted byte
  (including each registered special/EOT atom as one byte), reduces CE in FP32,
  separately reports atomic CE, includes partial batches, shards across DDP
  ranks with rank-invariant corruption, and uses BF16 model execution;
- clean/global/decoder K/V caches are typed and reused across canvas NFE, while
  mid-patch prompts take up to three AR atoms and commits discard scratch K/V
  before PAD-masked causal replay of accepted ids;
- checkpoints and exported artifacts bind the exact byte/control manifest;
  final export requires a completed checkpoint, dataset provenance, and
  post-dequantization evaluation.

The CPU gate currently passes 212 focused tests, plus a real two-rank
Gloo update/validation/exact-resume test. Queued CUDA parity passes
native varlen Flash forward/backward, Flex branch forward/backward, Triton
categorical/reveal parity, and production branch backpropagation. Compiled
steady-state measurements are recorded in the compute section.

## References

The repository keeps immutable local copies so the implementation can be
audited against the exact paper versions used for the design.

| Paper | Local copy | Source | SHA-256 |
|---|---|---|---|
| Byte Latent Transformer | [PDF](papers/byte_latent_transformer_2412.09871.pdf) | https://arxiv.org/pdf/2412.09871 | `23e0cc90e9e2ef0c11291043ca7ad4d48d5384328f31007b8b40230f370756ff` |
| Fast Byte Latent Transformer | [PDF](papers/fast_byte_latent_transformer_2605.08044.pdf) | https://arxiv.org/pdf/2605.08044 | `7957dd70eb7175a86bdcb1f8e8fca2f1444a348c0058dc924633f892d7efc8a2` |
| DiffusionGemma Technical Report | [PDF](papers/diffusion_gemma_2608.00146.pdf) | https://arxiv.org/pdf/2608.00146 | `9c8e0cef08271873ff0ce560e0c6149912ce5ce2fa056394de4feec205bb2064` |
| Introspective Diffusion Language Models | [PDF](papers/introspective_diffusion_language_models_2604.11035v1.pdf) | https://arxiv.org/pdf/2604.11035v1 | `8b07e696da127ae41842b9b2515d7ea3a0b709448c927a04806831a6377465b2` |
| Fast-dLLM v1 | [PDF](papers/fast_dllm_v1_2505.22618.pdf) | https://arxiv.org/pdf/2505.22618 | `8dcf7152cbfb042bcd00cd74dea56fdb324a8cfe655e9182f06fa2e97e269398` |
| Fast-dLLM v2 | [PDF](papers/fast_dllm_v2_2509.26328.pdf) | https://arxiv.org/pdf/2509.26328 | `9e3dcb349e0269fee8ac8c525131284a6408e506900c69940bb898f41274d49f` |

The digests in the table are the integrity contract for the checked-in PDFs.
BLT and Fast BLT are the architectural references. DiffusionGemma and I-DLM
are the large-canvas and exact-proposal references. Fast-dLLM is supporting
evidence for block-aligned packing, complementary masks, shifted conversion
objectives, and hierarchical caching; it is not the architectural base.

## Objective

Pretrain a byte-native diffusion model from random initialization which can
generate several bytes per expensive forward pass. The same weights retain a
causal attention mode because the challenge BPB and exact draft verification
require a normalized next-byte distribution; this is a training/evaluation
property, not compatibility with BOLMo. The family has three serving modes
sharing one checkpoint:

1. **AR anchor:** exact next-byte generation and BPB evaluation.
2. **Introspective stride:** lossless or near-lossless self-speculative byte
   generation for high-concurrency rollouts.
3. **Canvas diffusion:** iterative generation of a fixed 512-byte canvas for
   low-concurrency latency and long parallel proposals.

The intended final system routes between the modes. Canvas diffusion is the
primary pretraining objective. The causal prefix loss is an auxiliary applied
inside the same mixed-mask forward, not a separate AR pretraining stage.

Production batching is document-count exact. Each optimizer update consumes
256 rows. On the 32GiB RTX 5090, cropped clean-plus-canvas banks use ten
24-row microbatches plus one 16-row tail at worst, with a 208,896-position
cap. The trainer normalizes AR targets and diffusion canvases over the whole
update, so the tail changes neither sample exposure nor objective weighting.
On 8xH100 the same global batch may use one 32-row microbatch per rank.

### Evaluation contract

The three losses/readouts are not interchangeable:

- **Challenge BPB** is the exact causal codelength: sum next-atom NLL for all
  literal bytes and registered control atoms, then divide by the repository
  byte count and `ln(2)`. Each registered special, including EOT, contributes
  one denominator byte, matching the challenge LUT. This is the only headline directly comparable with the AR
  challenge scorer. Periodic training evaluation uses a deterministic
  2,048-row proxy and must be logged as `val_proxy_bpb`; a completed checkpoint
  receives `val_challenge_bpb` only after the full bound FineWeb validation
  split is scored.
- **Diffusion ELBO-BPB** is an upper bound on NLL obtained from the complete
  absorbing-diffusion variational objective, divided by the same challenge-byte
  denominator. It requires the `1/t` weighting, complete block inclusion
  accounting, and Monte Carlo uncertainty. Ordinary mean masked-byte CE is an
  optimization diagnostic and must never be renamed BPB.
- **Sampler quality** is empirical: task accuracy/pass rate, UTF-8 validity,
  EOT behavior, repetition, NFE, latency, and throughput. A good ELBO does not
  guarantee that a low-NFE approximate sampler preserves quality.

This matches how the references avoid a false equivalence. Fast BLT uses its
jointly trained causal next-byte path for likelihood-based evaluation.
DiffusionGemma and I-DLM primarily report downstream quality and serving speed
while retaining causal paths/anchors. Fast-dLLM derives the absorbing MDM
objective from an ELBO, but its acceleration results are evaluated by sampled
quality and throughput rather than treating raw denoising CE as perplexity.

## What the papers establish

### Byte Latent Transformer

The original BLT paper defines the architecture that Fast BLT reuses. Three
details are easy to lose in a simplified implementation:

- the local encoder is a causal byte Transformer with a local window that may
  cross patch boundaries, not independent attention inside each patch;
- patch representations are pooled with Perceiver-style cross-attention from
  patch queries to the bytes belonging to that patch;
- every local decoder layer first cross-attends from byte queries to global
  patch latents and then applies a byte Transformer layer.

BLT also finds causal byte n-gram hash features and decoder cross-attention
materially useful. Its full hash tables are far too large for this artifact,
so a compact factorized version is an ablation rather than a presumed free
win. Across the published scales, most parameters live in the global model:
the Fast BLT 1B configuration reports 1.28B global, 19M encoder, and 160M
decoder parameters. The small model here should preserve that allocation
principle even though the exact optimum may move at 25M parameters.

### Fast Byte Latent Transformer

Fast BLT shows that absorbing discrete diffusion works directly over a small
byte vocabulary and can coexist with an AR byte loss. Its useful design
elements are:

- a dedicated atomic `[MASK]` state;
- independent Bernoulli masking of byte positions at a shared sampled `t`;
- clean-prefix causal attention and bidirectional attention inside each noisy
  block;
- AR next-byte loss plus masked-byte reconstruction loss;
- entropy-bounded or confidence-based progressive unmasking;
- optional AR verification of a diffusion draft.

Fast BLT's clean and corrupted predictions have different alignments. Clean
hidden position `i` predicts the next byte `i + 1`; a corrupted block position
reconstructs its own clean byte `i`. It initializes decoder states from an
embedding lookup for the actual clean/corrupted input ids and preserves each
block byte's original absolute position for RoPE. It does not embed the noise
level explicitly.

Its training expansion is not suitable for a 512-byte target here. It creates
one fixed-length block at every eligible patch start, so its diffusion tensor
has approximately `number_of_patches * block_length` positions. Fast BLT only
tests block lengths 4, 8, and 16. We must not extrapolate that construction to
512.

### DiffusionGemma

DiffusionGemma shows that a large iterative canvas is practical when conversion
training samples one canvas rather than expanding every possible anchor. It
uses a 256-subword-token canvas, at most 48 denoising steps, and about 12 steps
on average with adaptive stopping. Its decisive lessons are:

- warm-start from a capable AR checkpoint;
- preserve an AR encoder/clean loss while teaching a noisy decoder path;
- select one canvas per training sequence;
- train with continuous noise levels and self-conditioning;
- treat sampler distillation plus RL as a distinct second conversion stage;
- measure throughput by concurrency because the low-batch advantage can
  disappear when generation becomes compute-bound.

Its corruption unit is a tokenizer token, not a byte, and the terminal canvas
contains independently uniform random vocabulary ids rather than `[MASK]`.

The paper does not establish that 512 bytes are emitted correctly in one pass,
that diffusion improves BPB, or that its conversion is cheap in absolute
compute. It establishes that a large canvas can settle in far fewer forwards
than its number of positions.

### Introspective Diffusion Language Models

I-DLM identifies a quality and serving failure in ordinary block diffusion:
the proposal distribution may not agree with the model's own causal
distribution after proposed tokens become clean context. Its useful design
elements are:

- use the same strict causal attention pattern for proposal and clean paths;
- preserve the AR logit shift, so hidden position `i` predicts position
  `i + 1`;
- densely supervise an all-masked proposal region and a clean causal region;
- balance masked and clean CE magnitudes without differentiating through the
  balancing ratio;
- accept or reject each proposal against the causal `p/q` ratio;
- optionally gate proposal-only LoRA residuals off at causal-anchor positions,
  making the anchor exactly the base AR model;
- optimize for high concurrency, not only batch-one latency.

I-DLM uses small token strides, not a 512-position canvas. It is a complementary
proposal-and-verification path and the best first target for post-training
rollouts with hundreds of concurrent trajectories.
It trains the proposal copy with every token position masked, so it supplies
hard all-mask supervision but no evidence about partial byte corruption.

### Fast-dLLM

Fast-dLLM v2 converts pretrained token models using block size 32, block-
aligned packing, a learned `[MASK]` embedding, masked-only CE, complementary
mask views, and a shifted target: the hidden state immediately before a
masked token predicts it. The shift is evidence for retaining the geometry of
an existing AR checkpoint; it is not evidence that a scratch byte denoiser
should replace Fast BLT's same-position reconstruction. The useful ideas here
are therefore isolated ablations:

- align and pad each document independently before packing so a diffusion
  block never crosses a document boundary;
- distribute complementary masks across rows when it improves device-time
  efficiency, rather than unconditionally duplicating every row;
- separate the training block size from a smaller inference sub-block only
  after measuring the quality loss;
- explore its hierarchical/DualCache scheme only after an exact no-cache
  reference works, because mutable diffusion caches are approximate.

Fast-dLLM v1 is an inference-side caching and confidence-parallelism reference.
Neither version is scratch pretraining evidence and neither is byte-level.

## Family hypotheses

Every hypothesis needs a controlled run before it becomes a family default.

1. A causal introspective byte proposer can preserve the scratch model's own
   base-only AR distribution while reducing serial forwards during batched
   rollouts.
2. A small fixed number of 512-byte canvases is computationally viable when
   clean-prefix work is shared; Fast BLT's block-per-patch expansion is not.
3. The 261-way atomic output space makes byte-canvas sampling substantially
   cheaper than DiffusionGemma's 262k-way sampling, but backbone computation,
   not the output head, remains the dominant scaling term.
4. Sampler distillation and RL are necessary for useful few-step 512-byte
   decoding. Plain diffusion SFT should not be judged only at a low fixed NFE.
5. A later learned-patching variant will lower introspective acceptance unless
   routing is strictly causal and identical between proposal and anchor paths.
6. A shared AR anchor with two proposal mechanisms will dominate a
   canvas-only model because our workloads span batch one through roughly 384
   concurrent trajectories.

## Representation contract

### Atomic ids

Keep one categorical id per atomic symbol:

- `0..255`: literal octet values used by UTF-8 text;
- `256..260`: the current five atomic specials, including EOT;
- `261`: diffusion `[MASK]`;
- `262`: padding, never a prediction target.

The exact control ordering is a versioned byte-model manifest shared by data,
training, post-training, inference, checkpoints, and export. It is not derived
from a source tokenizer. Newline remains byte `0x0A`; it is not a control.

### Tokenization and corpus identity

The byte tokenizer is the production tokenizer, not a preprocessing fallback:

- literal text maps exactly to strict UTF-8 octets with no normalization;
- control tokens are typed events, never magic substrings scanned out of
  untrusted text, so literal `<think>` remains seven bytes;
- prompts contain only their literal/control atoms; the model supplies its
  leading EOT/BOS virtually, so serving never embeds EOT inside a live prefix;
- SFT/RL composes `<think>`, `</think>`, `<answer>`, and `</answer>` from
  separate trusted fields as atomic controls;
- incremental decode buffers incomplete UTF-8 code points and rejects a
  control/EOT inside one instead of emitting replacement text.

Corpus selection also uses `utf8_bytes`. The deterministic K3 builder budgets,
deduplicates, interleaves, validates, and shards in the same byte ids that the
model consumes. ToaST is not present in the canonical byte build. A GPT-2 AR
comparison must be materialized from that selected byte-bounded document
stream, rather than independently selecting a GPT-2-token-bounded corpus; this
holds raw documents and bytes fixed while exposing the expected difference in
sequence lengths.

The model predicts categorical byte ids, not eight independent bits. The byte
distribution can express arbitrary dependencies between bits and preserves one
atomic position for each special. UTF-8 code points may occupy one to four byte
positions; the diffusion objective operates on byte positions and learns those
dependencies jointly inside a canvas.

Noise is applied to atomic byte positions, not Unicode characters. The bytes
of a multi-byte code point may be masked independently. Whole-code-point masks
are not the baseline: deriving their boundaries from the clean target changes
the corruption prior and the mask-run length reveals encoded width; no byte-
level reference here requires it. Specials are noised only when valid for the
selected objective; PAD is never noised, attended to, scored, or emitted.

The fixed width four is a compute-patch stride, not a UTF-8 width. A patch may
contain four ASCII characters, one four-byte code point, fragments of several
code points, or a code point split across two patches. Encoder and decoder byte
windows cross patch boundaries, so a continuation byte in the next patch can
still use its preceding bytes. Patch offset and UTF-8 decoder state are
independent pieces of incremental state.

### Boundaries

Patch boundaries are routing decisions, not text symbols. Remove the current
fused `2 * atomic_vocab_size` output vocabulary from the new family. The AR
head predicts only atomic ids. If learned patching survives its ablation, a
separate deterministic boundary head decides whether a committed byte closes a
patch.

The AR anchor must be a normalized causal distribution. The current BOLMo
boundary predictor reads one byte of lookahead and its standard routing is
therefore not a valid anchor for exact `p/q` acceptance. Before implementing
lossless introspection, choose one of:

1. replace it with a strictly causal boundary predictor whose decision affects
   only later positions; or
2. use content-independent fixed patching.

The initial implementation uses fixed patching for proposal and anchor parity.
Learned causal patching is a later, isolated ablation. No boundary is predicted
inside an unresolved canvas: its fixed patch lattice is known from position,
and noisy patch latents are scratch state until commit.

### Tail and overflow contract

BOLMo pads tensors to a multiple of 128 but, under predicted routing, can leave
the bytes after its final predicted boundary unpooled because the last
lookahead boundary is undefined. Its local decoder still scores that small tail
from the last completed patch state. The new family does not inherit this
ambiguous row-end behavior.

Follow Fast BLT's PAD-excluded short-block handling and Fast-dLLM's document-
local block-aligned packing. DiffusionGemma's fixed canvas/EOS behavior and
I-DLM's accepted-prefix-only commit provide the inference-side precedent.

- Pad each document independently to a multiple of four before block-aligned
  packing. Up to three PAD slots are never attention keys, targets, corrupted
  positions, or outputs.
- EOT closes a short final patch early; pool only its valid bytes and reset the
  fixed-stride phase for the next document.
- Align artificial training chunks to patch boundaries and carry one extra
  target-byte halo for shifted AR labels. If no halo exists at a true stream
  end, exclude only that undefined final label.
- Incremental AR/ISD state carries zero to three committed bytes in an
  `incomplete_patch_buffer`. They decode from the previous complete global
  latent. On byte four or EOT, pool the patch and append its global state.
- Begin the simple canvas path only at a patch boundary and use a length
  divisible by four. A prompt ending mid-patch first takes at most three AR
  steps. Mixed committed/noisy first patches are a later latency ablation.
- If EOT is produced inside a canvas, ignore all later canvas positions and
  commit the EOT-ending partial patch. Those later predictions are neither
  overflow output nor training targets.
- A raw maximum-byte limit may cut a UTF-8 code point even when block handling
  is correct. Raw-byte evaluation records that outcome. A text-serving adapter
  may roll display back to the last accepting UTF-8 DFA state, but must report
  discarded bytes and must not count them as generated text or silently alter
  BPB.

## Target architecture

### Shared clean-prefix backbone

Build a new model from explicit components rather than adding diffusion
branches inside `BolmoModel`:

```text
atomic embedding
    -> patch-local encoder
    -> fixed or strictly-causal patch latents
    -> dual-mode global latent backbone
    -> aligned byte proposal/anchor decoder
    -> atomic-id head
```

There is no state-dict, API, tensor-shape, or architectural compatibility
requirement. The final path contains no legacy shim and loads no BOLMo weights.
The 25.8M retained source-token embedding is removed. A prefix-only learned
embedding table is an explicit parameter-matched ablation, not a compatibility
path.

### Paper-to-design decisions

| Question | Primary evidence | Family decision |
|---|---|---|
| byte representation | BLT / Fast BLT | categorical byte/special ids, never independent bits |
| local encoder | BLT | one causal sliding-window byte block followed by patch cross-attention pooling; not four-byte-only self-attention |
| capacity allocation | BLT / Fast BLT | heavy global trunk, light encoder and decoder |
| clean prediction | all references | shifted next-byte CE |
| corrupted prediction | Fast BLT | same-position masked-byte CE; Fast-dLLM's shifted variant is an ablation |
| decoder input | Fast BLT | fresh embedding lookup of clean/corrupted input ids |
| decoder/global link | BLT / Fast BLT | aligned global-to-byte conditioning at every decoder layer |
| explicit timestep embedding | Fast BLT / DiffusionGemma | absent in both documented architectures; optional uniform-noise ablation, not a reference requirement |
| self-conditioning | DiffusionGemma | absent initially; add a budgeted `FFW(stopgrad(p) @ E)` arm |
| training expansion | Fast BLT / DiffusionGemma | 512 sampled corrupted positions per row: several isolated BLT-D blocks or one canvas; never all patch starts at length 512 |
| patching | BLT | fixed stride four for the correctness baseline; causal entropy patching is a later BPB ablation |
| large canvas | DiffusionGemma | 512 bytes remains the target, but the latent-canvas path is explicitly novel and must earn promotion over BLT-D at 16/32 bytes |
| exact fast generation | I-DLM | separate strict-causal proposal mode after the base denoiser works |
| packing/cache | Fast-dLLM | block-aligned documents now; approximate mutable caches only after parity baselines |

### Input embedding and local encoder

Use local width 256 and global width 512. A single learned input table
`E_in[263, 256]` is shared by the encoder and fresh decoder input lookups.
Rows `0..260` are clean symbols, row 261 is learned `[MASK]`, and row 262 is
fixed to zero for padding. The output matrix is separate and has exactly 261
rows. Direct tying to `E_in[0:261]` is dimensionally valid and saves 66,816
parameters, but imposes an unsupported input/output geometry constraint. Use
untied output in the named baseline and compare tying as a parameter-matched
ablation.

The clean local encoder follows BLT rather than the earlier intra-patch
simplification:

1. embed bytes and preserve their absolute document positions;
2. run one 256-wide, 4-head causal Transformer block with a 512-byte sliding
   window;
3. initialize each fixed-stride-four patch query by max-pooling its four byte
   states and projecting to width 512;
4. pool the patch with 8-head Perceiver-style cross-attention whose keys/values
   are only the bytes in that patch.

A compact causal n-gram feature arm is implemented behind a switch: one shared
`32768 x 16` hash table for ending n-grams of lengths 3 through 8 plus six
small `16 -> 256` projections. Hashes are computed from the actual visible or
`[MASK]` ids, never from hidden clean targets. This is only a parameter-
constrained analogue of BLT's much larger tables and requires a 2,000-step
on/off ablation.

For a noisy latent canvas, the same encoder weights use a different verified
mask: noisy bytes read the committed clean prefix and all bytes in their own
canvas, but no other branch or clean target. Patch pooling then yields one
noisy latent per four bytes. This is the first material extension beyond Fast
BLT and is tested against the decoder-mechanics reference described below.

### Heavy dual-mode global backbone

Use nine 512-wide Transformer blocks, 8 heads, and SwiGLU width 704. Clean
latents are causal and cacheable. Noisy canvas latents read cached clean-prefix
K/V and attend bidirectionally within only their own branch. They preserve
absolute patch indices; concatenating branches must not change their RoPE
positions or permit cross-branch attention.

The local and global self-attention choices both give head dimension 64, a
kernel-friendly baseline. Head count is not copied from the billion-parameter
papers when doing so would create 16- or 32-wide heads at this scale.

The two denoising paths are named so results cannot be accidentally conflated:

- **BLT-D reference:** the encoder and global trunk process only the clean
  row. Isolated 16-byte, then 32-byte, corrupted blocks are handled by the
  decoder and every block reads the clean global latent immediately before its
  start. Training samples enough nonoverlapping blocks to total 512 corrupted
  decoder positions per clean row: 32 blocks at length 16 or 16 at length 32.
  This preserves Fast BLT's decoder mechanics, masks, positions, alignment,
  and loss while scaling its decoder to two layers, lowering one-key cross-
  attention to a projection, fixing routing, and replacing its block-at-every-
  patch expansion with sampled compute matching.
- **latent canvas:** corrupted bytes are encoded into patch latents and the
  global trunk is rerun over those latents with cached clean-prefix K/V. This
  makes a 512-byte canvas 128 global positions rather than asking one prefix
  latent to describe 512 future bytes. It is inspired by BLT's hierarchy and
  DiffusionGemma's full-backbone canvas, but no cited paper establishes it.

The inherited KDA blocks are not used. A causal KDA control is reasonable only
after the Transformer reference is correct; KDA is not silently treated as
bidirectional.

### Light dual-mode byte decoder

Use two 256-wide local decoder blocks, each with 4-head RoPE self-attention,
a 512-byte sliding window, SwiGLU width 512, and aligned global conditioning
before the Transformer block. This moves the parameter ratio toward BLT and
reduces the repeated per-byte work that dominates both training and sampling.

With exactly one permitted global latent per byte, BLT cross-attention's
softmax is over one key and therefore reduces algebraically to a learned
projection of that latent plus a residual. The optimized baseline uses a
`512 -> 256` projected addition at every decoder layer. A literal cross-
attention module remains a parity/quality ablation; if a byte is ever allowed
to read multiple latents, real cross-attention becomes mandatory.

Decoder states start from `E_in[input_id]`, not bidirectional encoder hidden
states. Clean decoder attention is causal. Corrupted BLT-D or canvas positions
attend bidirectionally inside only their own block/canvas and causally to the
committed prefix. In clean mode, position `i` predicts `x[i + 1]`. In the
primary absorbing-denoising mode, corrupted position `i` predicts `x[i]`.
These alignments share one untied `256 -> 261` output matrix but use different
label indexing.

Never instantiate dense attention over the complete expanded byte sequence.
Long-range work belongs to patch latents; byte attention is windowed and every
Flash/Flex implementation must match a dense small-tensor oracle.

### Parameter and export budget

The revised provisional count follows the BLT allocation rather than
subtracting BOLMo's old embedding. All attention, MLP, pooling, conditioning,
and output projections are bias-free in this count; blocks use RMSNorm. The
implementation must reproduce the named tensor shapes and emit its exact count
before any GPU workload.

| Component | Configuration | Estimated parameters |
|---|---:|---:|
| input embedding | `263 x 256`, PAD row fixed zero | 67,328 |
| factorized n-grams | `32768 x 16` plus six `16 x 256` projections | 548,864 |
| local encoder Transformer | 1 block, width 256, SwiGLU 512 | 655,872 |
| patch pooling | max-pool projection plus local-to-global cross-attention | 918,272 |
| global latent backbone | 9 blocks, width 512, SwiGLU 704 | 19,178,496 |
| local decoder | 2 blocks, width 256, SwiGLU 512, per-layer `512 x 256` conditioning | 1,574,400 |
| untied output head | bias-free `256 x 261` | 66,816 |
| final norms and three local-width mode embeddings | no gates/timestep/self-conditioning | 1,536 |
| provisional base total | factorized n-grams on | 23,011,584 |
| hard parameter cap | | 24.8M |

About 83% of the provisional parameters are in the global trunk, versus roughly
54% in the previous draft; the four-layer local decoder is removed. A raw
4-bit payload for 23,011,584 values is about 11.51MB. This does not prove artifact
eligibility: quantization scales, exceptions, metadata, and executable code
must all fit in 16,000,000 decimal bytes. The remaining parameter headroom is
not pre-spent on LoRA, timestep projections, or self-conditioning.

Proposal-only adapters are a later I-DLM ablation and must compete against
putting the same serialized bytes into the base model. Local-decoder-only LoRA
does not reproduce I-DLM's all-linear masked-position residuals and must not be
called lossless merely because its gates are zero on anchor positions.

### Initial configurations

The implementation begins with two named configurations rather than pretending
that the 512-byte extrapolation is already established:

| Field | `bd_blt16_ref` | `bd_canvas512` target |
|---|---:|---:|
| local/global widths | 256 / 512 | 256 / 512 |
| encoder / global / decoder blocks | 1 / 9 / 2 | 1 / 9 / 2 |
| self-attention heads | 4 local / 8 global | 4 local / 8 global |
| fixed patch stride | 4 | 4 |
| factorized n-grams | on; ablate off | on; inherit winning reference arm |
| corrupted global latents | none | 128 |
| corrupted length | 32 x 16-byte blocks, then 16 x 32 | 512; benchmark 128/256 first |
| corrupted positions per clean row | 512 | 512 at `M=1`; ablate `M=2,4` |
| corruption | independent-byte absorbing `[MASK]` | `absorbing_rb`; then `allmask_50`, then whole-patch if needed |
| corrupted target alignment | same position | same position |
| input/output tables | shared encoder/decoder input; untied bias-free output | same |
| final normalization | RMSNorm at global and decoder outputs | same |
| noise embedding / self-conditioning | off / off | off / off |
| high-NFE evaluation | 16 | 48 |

Both models train from random initialization with clean AR and denoising losses
in the same pretraining run. The reference is an engineering and scientific
control, not an earlier pretraining stage or a warm-start checkpoint.

### Compute, supervision, and memory estimates

Quality and sample-efficiency statements remain planning estimates. Systems
figures in the following table are measured queued runs on the RTX 5090 with
BF16, compiled static shapes, fused AdamW, native varlen Flash, and compiled
FlexAttention. They time forward, backward, and optimizer for one microstep;
validation and data construction are excluded.

| `B_row` | `L` | `C` | `M` | step ms | positions/s | peak allocated / reserved | average / peak power |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,192 | 512 | 1 | 18.82 | 435K | 1.11 / 1.15 GiB | 179 / 179 W, one sample |
| 8 | 8,192 | 512 | 1 | 42.51 | 1.54M | 5.85 / 6.50 GiB | 323 / 484 W |
| 16 | 8,192 | 512 | 1 | 85.38 | 1.54M | 11.19 / 12.10 GiB | 423 / 499 W |
| 32 | 8,192 | 512 | 1 | 176.41 | 1.486M | 21.49 / 23.09 GiB | 455 / 491 W |
| 16 | 8,192 | 512 | 2 | 118.16 | 1.11M clean positions/s | 11.72 / 13.49 GiB | 333 / 491 W |

The post-audit canvas batch-32 result is stored with hardware/software, dirty-
tree, warmup, and raw power metadata in
[`benchmarks/rtx5090_canvas512_b32.json`](benchmarks/rtx5090_canvas512_b32.json).
The corresponding `B=8`, 32 x 16-byte BLT-D branch bank measures 85.37 ms,
768K clean positions/s, 5.33/6.08 GiB allocated/reserved, and 231/265 W
average/peak; its record is
[`benchmarks/rtx5090_blt16x32_b8.json`](benchmarks/rtx5090_blt16x32_b8.json).
This lower BLT utilization is direct evidence for benchmarking its branch-bank
mask/projection path, not a reason to inflate microbatch memory blindly.

Real packed-corpus update benchmarks supersede the synthetic capacity points
for launch selection. Batch 16 is warning-free at 1.168s/update, 13.50/15.06
GiB allocated/reserved, 302W, and 68% utilization. The final gated batch-24
run with its 16-row tail is warning-free at 1.111s/update, 19.74/21.58GiB,
343W, and 80.7% average utilization (99% peak), with no post-warmup graph
breaks or recompiles. Batch 32 reaches 1.097s/update but repeatedly leaves less than
200MiB physical headroom and emits allocator mapping-OOM warnings. Batch 24
therefore gives up only 1.3% wall throughput while recovering about 5.7GiB of
allocated headroom, and is the single-5090 production choice. At 1.111s/update,
2,000 training updates are about 37.0 minutes before periodic validation.

A fresh-process CUDA resume preflight restores the cursor and every Python,
Torch, CUDA, and corruption RNG state exactly, reproduces the next update's
reported metrics exactly, and keeps one-step floating-state drift bounded to
5.82e-5 for model weights and 5.83e-7 for optimizer state. Native varlen Flash
backward uses parallel accumulation, so separate processes are not falsely
claimed to be bit-identical; the observed mean model drift is 6.55e-9.

Do not feed a long clean row and supervise only one 512-byte suffix. Process
the clean row once, obtain its causal prefix states and AR loss, and attach
`M` independently corrupted non-overlapping canvas branches at sampled patch
boundaries. Each branch may read only the clean prefix strictly before its own
start. Begin with `M = 1`; `M in {2, 4}` is a device-time-matched ablation,
not a default. This shares clean-prefix work without Fast BLT's block at every
eligible patch start.

Define batch quantities precisely:

- `B_row`: clean documents per optimizer microstep;
- `M`: independently corrupted branches attached to each clean row;
- `B_canvas = B_row * M`: canvas branches per microstep;
- `B_global`: clean documents per optimizer update after accumulation.

For a clean row of length `L`, the activation token-state count scales
approximately with `L + M*C`, not `M*(L+C)`, only if prefix activations and K/V
are genuinely shared. At `L = 8,192` and `C = 512`, moving from `M = 1` to
`M = 4` adds about 18% token states relative to `M = 1`. It is neither a compute
nor a VRAM prediction: canvas queries also attend prefix/window keys, and
global/local widths, ragged gathers, saved tensors, workspaces, and compiler
pools differ.

Supervision counts provide a useful variance estimate, not a claim that
reconstruction targets equal independent language-model tokens. At 2,000
updates, `B_global = 256`, `L_max = 8,192`, `C = 512`, and `M = 1`, the current
document-aligned corpus supplies 651.7M clean AR targets. Short documents make
the masked count lower than the 131M full-canvas ceiling: the manifest's
157,803,345 eligible canvas atoms imply 79,157,672.5 expected active masked
targets under uniform `K in 1..U`. `M = 4` is not assumed to provide exactly
four times the useful signal. The 4.19B figure obtained from `updates *
B_global * L_max` is only the padded-position envelope and must not be reported
as supervision. About 4,667 of the 512,000 `M = 1` canvases are expected to be
exactly all-mask because short documents raise `P(K=U)`. Those endpoints still
contribute only 512K expected target labels, 0.647% of masked-label mass, so
endpoint quality may be variance-limited even when mean masked CE is stable.

The compute-matched BLT-D reference also processes 512 corrupted positions per
row and has about the same expected active-target count. In contrast, Fast
BLT's literal length-16 block at every stride-four patch start would create
about 32,752 corrupted positions for an 8,192-byte row, roughly 64 times this
budget. That expansion supplies much more overlapping supervision and is one
reason its sample efficiency cannot be copied without its compute cost. We
therefore make no numerical BPB-efficiency promise: compare clean BPB,
all-mask continuation, and quality per device-second. More masked labels alone
are not “more signal” because their errors are correlated and reconstruction
with visible neighbors is easier than continuation.

The previous microbatch capacities were not justified. The implemented sweep
now supports these operating points:

| Workload | first `B_row` | subsequent probes |
|---|---:|---:|
| joint clean + BLT-D blocks totaling 512 positions | 8 | 16, 32, 64 |
| joint clean + canvas-512, `M=1` | 24 selected | 16 safe control; 32 capacity-only |
| joint clean + canvas-512, `M=4` | 4 | 8, 16, 32 |
| canvas-only kernel characterization | 32 | 64, 128, 256 |

Unmeasured rows remain probe coordinates, not expected maxima. For
`B_global = 256`, accumulation is `ceil(256 / B_row)` and the final microbatch
is truncated to the exact remaining row count. With no batch-dependent
normalization this preserves the optimization batch; a one-update causal
gradient-parity test covers the non-divisible tail path.

The benchmark grid is:

- clean-length buckets `4096, 6144, 8192, 10240, 12288`;
- `C in {16, 32, 128, 256, 512}` as applicable;
- `M in {1, 2, 4}` and `B_row in {4, 8, 16, 32, 64}`;
- checkpointing `none`, `global-only`, and `full-block`;
- a complete BF16 forward, separate raw losses, backward, optimizer step, and
  compiled steady state after warm-up.

Record peak allocated and reserved memory, compiler/CUDA-graph pools, saved-
tensor bytes, recompilations, retries, step time, and useful clean/noisy
positions per second. Saved-tensor hooks must demonstrate that branch memory
grows with `M*C`, not `M*L`; a backend assertion must reject dense attention
fallback. Choose the largest stable point with 15--20% physical VRAM headroom,
not merely the largest allocation that completes once.

The selected batch-24 path takes 1.111 seconds per exact global-256 optimizer
update on the single 5090, or about 37.0 minutes for 2,000 updates before
validation/data overhead. Applying the repository's measured 13.5x
5090-to-8xH100 calibration gives about 2.7 minutes. This is an estimate, not an
8xH100 measurement; it leaves useful room under ten minutes but must be
replaced by the official run. The run exposes 651.7M clean AR targets in this
document-aligned corpus (the exact count is data-dependent, not the 4.19B
padded-position envelope) and 79.2M expected active diffusion targets. The
completed run's logged count remains the operational record. That is
substantial correlated reconstruction supervision, not
evidence of a specific BPB improvement.
Larger `M`, sampler replay, distillation, and RL all count against wall time;
they are not free sample-efficiency gains.

Parameter storage is not the VRAM bottleneck. The provisional 23,011,584-parameter model
uses under 0.7GiB for BF16/FP32 weights, gradients, and optimizer state in the
expected setup. Activations, attention workspaces, and compiler pools dominate.
Bucket lengths and tune `B_row` per bucket rather than padding everything to
the worst case. Sampler trajectories are always no-grad with selected states
replayed for gradients; never retain an NFE-length activation graph.

### Kernel and utilization plan

Do not make 500W board power a promotion target. Existing repository `d=512`
workloads reached about 97% SM utilization while peaking below 430W; watts can
be increased by memory traffic without increasing useful work. Optimize
steady-state step time, achieved Tensor Core FLOP/s, useful byte positions/s,
DRAM bandwidth, kernel-gap fraction, and quality-adjusted generated bytes/s.
Record power as telemetry. A dense, well-batched canvas workload may exceed
500W, but neither compilation nor a custom kernel can guarantee that it should.

The first optimized implementation uses the local PyTorch 2.13/CUDA 13.3
stack:

- force Flash SDPA, with a failing assertion rather than silent fallback, for
  ordinary causal global attention and pure 128-position bidirectional canvas
  attention;
- use a measured FlexAttention `BlockMask` or verified variable-length Flash
  path for branch-to-prefix attention, because sampled branch starts have
  different prefix lengths and a rectangular mask-free call would leak later
  clean targets; bucket starts by prefix length if that is faster;
- use FlexAttention for the 512-byte sliding window and I-DLM masks; never
  materialize an `8K x 8K` dense local mask;
- pack QKV, fuse RoPE, compile the complete static microstep and optimizer, and
  keep corruption, exact-`K` sampling, and canvas-start RNG on device;
- create separate `fullgraph`, `dynamic=False` compiled buckets for clean
  length, canvas length, `M`, and microbatch; add CUDA graphs only after those
  shapes and kernels are stable;
- use BF16 first. FP8 projections/MLPs are a later quality-controlled ablation,
  with norms and logits retained in BF16/FP32.

Reasonable fused work is implemented up front rather than deferred: packed
bias-free QKV projections are compiled with RoPE and surrounding pointwise
operations; native varlen Flash handles causal/window banks; FlexAttention
lowers branch masks to Triton; exact corruption remains on device; and custom
Triton kernels fuse the 261-way categorical sample/entropy/argmax/confidence
pass plus the stable 512-position reveal/EOT update. Torch reference kernels
remain executable parity oracles and CUDA paths fail closed rather than falling
back to dense masks. A fused corruption-plus-loss gather and transactional K/V
page trim are still reasonable next kernels. Fused RMSNorm/SwiGLU is not yet
justified because compiled batch sweeps plateau in useful positions/s rather
than exposing a measured launch-gap breakdown; this is a specific remaining
profile, not a policy to wait on obvious fusion.

The production profile does expose one remaining material packing cost:
expanded-mask `masked_select`/`masked_scatter` around each clean Q/K/V varlen
bank. This was acted on, not deferred. Exact CPU-built ragged indices, fused
QKV gather, opaque custom gather/scatter autograd, and fixed-capacity
dummy-suffix variants were each implemented and CUDA-tested. On the installed
PyTorch 2.13 compiler all four fail production dynamic backward lowering with
an Inductor `CantSplit` error when packed-clean and branch gradients meet.
Those rejected paths are not retained. The warning-free 1.111s/update path is
the launch baseline; revisiting vector-granularity packing requires either an
upstream scheduler fix or a fully opaque fused attention/backward extension,
and must beat that measured baseline before promotion.

The compiled joint path at `B=1,L=8192,C=512` measures 18.82 ms versus
273.26 ms eager, a 14.5x systems speedup and a reduction from 1.82GiB to
1.11GiB peak allocated. Typed hierarchical K/V reuse at inference reduces one
`B=1,prefix=4096,C=512` NFE from 12.14 ms with clean-prefix recomputation to
0.975 ms cached, a 12.45x speedup. At 48 NFE this is approximately 46.8 ms of
denoiser time versus 582.7 ms before prefix prefill/commit and sampler kernels;
12 NFE is approximately 11.7 ms versus 145.7 ms. End-to-end generated-byte
latency still depends on quality, early EOT, replay, and the chosen NFE.

Benchmark in this order: isolated global/local blocks, canvas-only batch sweep,
joint clean-plus-canvas bucket sweep, compile/CUDA-graph comparison, then
end-to-end inference at concurrency 1, 8, 32, 128, and 384 for NFE 12, 24, and
48 including the causal commit pass. All GPU benchmarks run through `mlq`.

### Proposal-only residuals

Do not reserve base-model parameters for proposal adapters. After the shared
model is competitive, implement an I-DLM-faithful residual arm that gates
low-rank updates by position mode in every linear it adapts, then choose its
rank from the remaining measured artifact budget. A decoder-only rank-16 arm
is a cheaper ablation, not the paper method. In exact introspective mode:

- clean/anchor positions use base weights only;
- `[MASK]` proposal positions use base plus residual weights;
- anchor positions cannot attend to proposal positions;
- acceptance is computed against base-only causal logits.

Zero-gating the residuals on clean positions provides a bit-for-bit base-only
anchor if the attention graph also prevents clean queries from reading
proposal states. Exactness of the generated distribution comes from I-DLM's
`p/q` acceptance correction, not from LoRA itself. Full shared weights may be
trained by the canvas loss as long as the AR loss and BPB gate hold; residuals
are considered only when objectives interfere or improve proposal acceptance
per serialized byte.

### Incremental state contract

`ByteDiffusionDecodeState` owns every mutable inference value:

- committed atomic ids, absolute position, EOT status, and RNG state;
- incremental UTF-8 DFA state for diagnostics, independent of patch offset;
- the fixed-patch offset, incomplete-patch byte buffer, and patch-local encoder
  state;
- per-layer global-prefix K/V pages and their committed latent length;
- per-layer sliding-window local-decoder K/V pages and RoPE positions;
- scratch pages for unresolved ISD proposals or one canvas;
- counters for proposal, verification, denoising, and causal-commit forwards.

The live backbone contains attention caches only. No BOLMo KDA/mLSTM state or
conversion teacher exists in the training or serving path. If a scientific
control introduces a recurrent module, it must add an explicit recurrent-state
field and step/reference parity tests.

The state API is transactional:

- `prefill(ids)` builds base-only causal state;
- `begin_isd(stride)` forks scratch pages without mutating committed state;
- `commit_isd(k)` promotes only the exact base-only introspection K/V for the
  accepted prefix, replays a rejection-resampled byte if necessary, updates
  any newly completed fixed patches, and frees later scratch pages;
- `begin_canvas(length)` reads prefix caches but writes only canvas scratch;
- `commit_canvas(ids)` truncates at EOT and performs an explicit causal append
  over the accepted bytes to update patch, global, and local caches;
- `abort_scratch()` releases all unresolved state and leaves the prefix
  bit-identical.

Proposal-position K/V produced with gated residuals is never promoted into the
base anchor. The canvas causal-append pass and any rejection replay are real
model work and must be included in throughput and NFE accounting.

The scored model remains a normalized distribution over all 261 output ids;
UTF-8 constraints are not hidden inside BPB. Report raw sampling first. An
optional serving mode may renormalize byte/EOT logits through the UTF-8 DFA to
prevent invalid transitions or EOT inside a code point, but that is a distinct
constrained distribution and exact `p/q` verification must constrain both
distributions identically.

## Objectives

### AR anchor loss

Use ordinary shifted next-atomic-id CE over clean bytes and specials. Report
challenge BPB with the repository's existing byte accounting. Padding and
the synthetic leading EOT/BOS are not scored.

### Introspective loss

Partition a clean region into proposal blocks and create an all-`[MASK]` copy.
The logical input contains noisy and clean copies, but it is not an ordinary
triangular mask over a concatenated sequence. For noisy block `b`:

- noisy query position `j` attends only noisy positions `<= j` in block `b`;
- it cross-attends to clean tokens from blocks strictly before `b`;
- it cannot attend clean tokens in block `b` or any later block.

Clean query position `i` attends only clean positions `<= i` and never attends
the noisy copy. This is the I-DLM mask: the proposal sees the committed prefix
without seeing its current clean targets, while the clean path remains an
ordinary causal anchor. Implement it as an explicit block mask; the physical
packing order must not determine information flow.

Apply the normal logit shift in both regions. A noisy position at absolute
clean index `i` predicts clean target `i + 1`; the final undefined target is
excluded. Compute dense shifted CE over the noisy positions, normal shifted CE
over the clean positions, and use
`loss = mask_ce + stop_gradient(mask_ce / clean_ce) * clean_ce`.

Tests must enumerate every permitted query/key edge for two adjacent blocks
and prove that changing the current clean block cannot change its noisy logits.

Track the two raw losses, the detached balance coefficient, and per-offset
proposal accuracy. Never report the balanced sum as AR BPB.

The hierarchy must enforce the same separation, not merely the byte decoder:

- Begin with proposal blocks of four bytes aligned to the fixed compute patch;
  strides 2 and 8 are later mask-parity ablations.
- The proposal local encoder uses strict causal byte attention at the original
  absolute positions. It starts from a fork of the committed zero-to-three-byte
  incomplete-patch buffer.
- Pool a proposal patch only after its fourth scratch position exists. Its
  scratch global latent may read cached clean latents strictly before the
  proposal block plus earlier proposal latents, never a clean latent containing
  any current-block target.
- Before a proposal patch closes, its decoder queries use the last committed
  clean global latent. At the closing proposal position, the usual shifted
  alignment allows the completed scratch latent to help predict the next byte.
- Proposal decoder attention is causal and cannot read the clean reference
  copy. Multiple sampled proposal branches cannot read one another.
- Scratch encoder/global/decoder K/V is never promoted. Accepted bytes are
  replayed through the base-only clean path, which rebuilds partial patches and
  appends a real global latent only when four accepted bytes or EOT close it.

Tests perturb current-block clean byte embeddings and clean patch/global states
independently. Neither may change proposal logits. A decoder-only edge test is
insufficient because a leaked clean patch latent would bypass it.

### Canvas diffusion loss

For clean sequence length `L`, canvas length `C`, and branch count `M`:

1. process one clean causal row and compute shifted AR CE once over every valid
   next-byte target in that full row;
2. sample up to `M` nonoverlapping valid canvas starts at committed patch
   boundaries;
3. sample an exact masked-position count `K` independently per canvas;
4. corrupt only those `C`-position canvases;
5. condition each branch only on the clean prefix before its own start;
6. predict all clean canvas ids or only corrupted ids according to the selected
   corruption objective;
7. combine mean canvas loss with `L_AR_clean_row` from the already computed
   clean row.

A canvas never crosses a document EOT. Prefer starts with `C` valid atomic
positions remaining in the same document. If a short suffix must be used,
truncate at its first EOT, pad the remainder, exclude padding from `K`, noise
level, attention keys, and loss, and prove that changing post-EOT storage
cannot change a pre-EOT logit.

Fixed-patch alignment is explicit. A clean hidden state at byte `i` predicts
byte `i + 1`: if `i` closes patch `p`, it may read global state `p`; otherwise
it reads `p - 1`. The synthetic initial state handles the first patch. A
same-position denoising query for byte `i` in noisy patch `p` reads that
branch's noisy global state `p` and never the clean state of `p`.

The primary target is `C = 512`. Initial controlled comparisons are
`C in {128, 256, 512}` and `M in {1, 2, 4}`. Run paired controls:
clean-byte-matched runs isolate
learning behavior, while device-time-matched runs answer the operational
question. Clean-prefix activations/K/V must be shared across the branches. Do
not construct a block at every patch start and do not recompute the prefix once
per branch.

#### Corruption granularity

Independent masked bytes are the Fast BLT reference, not an unquestioned
large-canvas default. At partial noise, visible UTF-8 continuation bytes,
spelling fragments, and other bytes in the same four-byte patch can make the
reconstruction loss locally easy. This may lower CE without teaching the
all-mask continuation needed at inference.

The first paper-supported difficulty control changes the noise distribution,
not the unit: compare `absorbing_rb` with `allmask_50`, where half of canvases
are fully masked and half use `absorbing_rb`. I-DLM supplies the all-mask
precedent; the 50/50 mixture is our declared ablation, not its objective or an
unbiased estimate of Fast BLT. This test precedes invented span corruption.

Compare three absorbing-mask selectors while holding the active byte count and
device time as close as possible:

1. `independent_byte`: choose exactly `K` individual byte positions uniformly;
2. `whole_patch`: choose `J` fixed four-byte patches and mask all four bytes,
   so no noisy latent retains within-patch clean clues;
3. `contiguous_patch_span`: mask one or more contiguous runs of whole patches
   with the same total masked-patch count, forcing longer-range reconstruction.

The first is the paper comparison. If `allmask_50` still shows a large gap
between easy partial CE and all-mask generation, the second is the primary
granularity ablation. The third tests whether removing nearby context helps
generation or merely makes optimization less sample-efficient. Always
include an explicitly measured all-mask bucket. Match arms by masked bytes,
not selected groups, and report denoising accuracy versus distance to the
nearest visible byte.

Whole-Unicode-code-point corruption is lower priority. It would still predict
bytes, but the repeated `[MASK]` count reveals the code point's UTF-8 width and
the corruption pipeline must parse clean Unicode, including arbitrary invalid
byte strings. Patch/span corruption is raw-byte-native and avoids that extra
side channel.

Two corruption variants are planned because the papers support different
choices:

- `absorbing`: replace selected positions with `[MASK]` and score only those
  positions;
- `uniform_replacement`: sample `t ~ Uniform(0, 1)` and independently replace
  each eligible position with probability `t` by an id drawn uniformly from
  output ids `0..260`; predict every clean canvas position.

Uniform replacement remains a standalone corruption oracle, but the production
run contract rejects it until its detached-prior self-conditioning forward is
implemented. The first baseline is absorbing corruption because it has
direct byte-level evidence and keeps random invalid UTF-8 bytes distinguishable
from clean context. `[MASK]` and PAD are never uniform replacements. A random
EOT or control id in `x_t` is noise, not structure: it does not truncate the
canvas or alter masks. Clean validity metadata controls training; EOT semantics
apply only when the final generated canvas is committed.

The `bd_blt16_ref` recipe implements a compute-bounded Fast BLT estimator:
sample block origins independently from every eligible fixed-patch origin,
apply the exact eligible-origin/sample-count inclusion weight, sample one shared
`t ~ Uniform(0, 1)` for all eligible positions in the concatenated sampled
blocks, mask every position independently with probability `t`, and weight the
summed masked loss by `1/t`. Its `paper_sum` arm is therefore an unbiased
estimate of the complete fixed-patch masked sum and adds the exact clean sum
before the batch mean. It is not an exact enumeration of all blocks and is not
assumed optimal at length 512.

The latent-canvas baseline, named `absorbing_rb`, uses a lower-variance
equivalent sampling view. Let `U` be the number of eligible positions. If one
unobserved `t` governs all `U` identical Bernoulli masks, the marginal count is
`K ~ Uniform{0, ..., U}`; conditional on nonzero `K`, the subset is uniform and
`E[1/t | K] = (U + 1) / K`. Therefore sample
`K ~ Uniform{1, ..., U}`, choose exactly `K` eligible positions without
replacement, and take a mean over masked positions within each canvas before
averaging canvases. Apart from the zero-contribution `K=0` event and a constant
scale, this Rao--Blackwellizes the Fast BLT estimator.

That equivalence requires all of the stated conditions: a single shared `t`,
no model input for `t`, identical masking eligibility, per-canvas reduction,
and no selective treatment of specials. If an objective excludes any clean
special from corruption, use its actual eligible count `U`; a global masked-
token mean across the batch would reweight examples by `K` and is not the same
objective.

For `whole_patch`, apply the same derivation to 128 patch groups: sample
`J ~ Uniform{1, ..., valid_patches}`, choose exactly `J` groups, mask every
valid byte in them, and average CE over the masked bytes. This matches the
noise-fraction distribution without pretending four bytes in one selected
group are independent observations.

Uniform-over-`K` gives only one all-mask example per `U` noise counts. If
all-mask starts are weak, compare `allmask_50` at equal device time; report it
as a changed optimization distribution
and, when estimating the reference objective, apply the proper importance
weights. Do not introduce heuristic per-noise loss weights silently. Log loss
and accuracy by `K/U` bucket, then measure objective-level shared-gradient
norms/cosine on a fixed batch. “UTF-8 role” always means the role
of the clean target byte, never a classification inferred from corrupted ids.
Report leading-byte, continuation-byte, EOT, and all-mask behavior separately.

Normalize `L_AR` by valid clean targets and `L_diff` by active masked targets,
then use `L = L_diff + L_AR` for `absorbing_rb`. This equal-mean normalization
is a deliberate optimization hypothesis, not Fast BLT's summed-loss equation;
compare it with `paper_sum` at matched device time. On a fixed diagnostic batch,
log each objective's shared-parameter gradient norm and cosine similarity. Only
then ablate
`lambda_ar in {0.5, 1, 2}` at equal device time. Dynamic gradient balancing is
not the baseline because it changes the optimization problem and can hide
objective conflict. If no fixed coefficient preserves AR BPB and denoising
quality, test proposal-only capacity rather than silently erasing one task.

### Self-conditioning

Self-conditioning is off for absorbing baselines. The planned uniform-noise arm follows
DiffusionGemma's core construction by forming `stopgrad(p) @ E_in[0:261]` at
width 256. Our budgeted
`RMSNorm -> 256 -> 64 -> 256` projection is a compressed, paper-inspired FFW,
not an exact reproduction. Match the documented training rule: on 50% of
uniform-noise examples feed a detached prediction from the prior no-grad pass;
on 50% feed zero. The terminal random canvas/first step uses zero. Count every
extra conditioning forward in device time. Neither Fast BLT nor DiffusionGemma
documents an explicit timestep embedding, so `t` conditioning is a separate
parameter-matched ablation. The production run contract currently rejects
uniform replacement so this incomplete construction cannot be selected
silently.

## Training phases

There is one scratch-pretraining phase, not two. Fast BLT trains clean and
diffusion objectives jointly, while I-DLM explicitly uses a single-stage
objective. DiffusionGemma's published two stages are denoising SFT followed by
sampler-distillation/RL; its second stage solves low-NFE generation and reward
alignment after high-NFE denoising works. That motivates an optional
post-training phase here, not a second default pretraining curriculum.

There is no BOLMo conversion and no AR-only warm-up. The causal-clean-row loss
is present from the first update so the same checkpoint exposes a normalized
BPB and verifier distribution.

### Scratch pretraining: analytic-noise byte denoising

Every 2,000-update scratch run contains one clean causal row plus noisy
branches sharing that row's prefix states. The named recipes differ
deliberately:

- `bd_blt16_ref`: iid fixed-patch-origin blocks totaling 512 corrupted
  positions, an inclusion weight for the full block bank, one shared continuous
  `t`, Bernoulli byte masks, sampled `1/t`, and `paper_sum` reduction;
- `bd_canvas{128,256,512}`: one latent canvas initially, `absorbing_rb`
  exact-`K` corruption, per-canvas mean diffusion loss, and mean clean AR loss.

The canvas recipe optimizes:

```text
L_pretrain = mean(L_canvas_1 ... L_canvas_M) + lambda_ar * L_AR_clean_row
```

Each model learns from both terms from random initialization. Start with
`M = 1` and `lambda_ar = 1`; there is no loss-weight warm-up and no earlier AR
checkpoint. Keep self-conditioning off in the first baseline so corruption,
alignment, and architecture can be debugged independently.

The trainer emits per-`K/U`-bucket raw NLL and accuracy on every update. Set
`BYTE_DIFFUSION_GRAD_DIAGNOSTIC=1` to run a non-perturbing fixed-validation-
batch measurement of AR and diffusion gradient norms/cosine on the shared
global trunk; the corruption RNG is restored before training begins.

Pretraining invariants:

- challenge AR BPB and diffusion losses reported separately;
- exact preservation and round-trip tests for atomic special ids;
- deterministic resume including data order, canvas starts, exact-`K` masks,
  and RNG;
- no clean query can attend an unresolved branch and no branch can read a
  separate clean copy of its current/future targets;
- all-mask generation quality reported separately from aggregate masked CE;
- no block-per-patch expansion or branch-local prefix recomputation.

### Post-training: sampler distillation and RL

This phase is optional and follows successful high-NFE pretraining. It still
counts against any end-to-end challenge training-time claim. DiffusionGemma
does not disclose enough to reproduce its SD·RL objective, so this family
defines and names its own algorithm rather than claiming a reproduction.

**Optional sampler-aligned replay ablation.** Before distillation, test whether
the analytic training distribution causes a measurable inference-state gap.
Keep half of states analytic. For the other half, run the fixed-quota sampler's
position selection under no-grad, teacher-force revealed values, sample a
forward index uniformly, and replay that state once with gradients. Use zero
self-conditioning at the first step and detached prior predictions afterward.
Free-running absorbing replay is not the baseline because an incorrectly
committed byte cannot be repaired; it requires explicit loss on committed
errors or a revisable corruption process. Promote this replay only if it beats
continued analytic training at equal device time.

**Distillation: online trajectory distillation.** For each transition, freeze
the immediately preceding sampler as teacher: the coherent pretrained 48-NFE
sampler teaches 24 NFE, then the accepted 24-NFE student is frozen to teach
12 NFE. Roll the current student at its fixed target schedule `K`. At a sampled
student state, run the teacher from the identical corrupted canvas for no more
than the teacher's named NFE budget, counting every additional teacher call.
Record the state, actual teacher NFE, teacher per-position distribution,
teacher final canvas, selected positions, and EOT. Optimize active student
positions with:

```text
L_SD = CE(student_logits, teacher_final_ids)
     + beta * KL(stop_gradient(teacher_probs) || student_probs)
     + lambda_ar * L_AR
```

Progress only after a held-out trajectory gate from `K = 48` to 24 and then
12. Position selection and stopping are deterministic functions of recorded
entropy for this first algorithm; they are not undocumented learned actions.

All rollouts are detached/no-grad. Replay only one, or a small fixed number,
of recorded student states with gradients. Retaining 24--48 rollout graphs
would multiply activation memory by NFE and exceed the 32GB development GPU.

**RL: on-policy group-relative optimization.** After distillation is stable,
sample a group of student trajectories for each prompt. Store every sampled
atomic id, its
log probability, corrupted canvas state, selected position, and denoising
step. Use final sequence reward:

```text
R = task_reward
  - c_nfe * denoising_forwards
  - c_len * committed_bytes
  - c_utf8 * invalid_utf8_events
  - c_eot * premature_or_missing_eot
  - c_rep * repetition_score
```

Normalize advantages within each prompt group. The policy term is the
advantage-weighted sum of recorded categorical byte log probabilities,
normalized by committed bytes. Add a per-state KL to the frozen B1 sampler and
the clean AR loss. Freeze the base-only causal anchor; update proposal/canvas
parameters and sampler residuals only. Deterministic position selection and
stopping contribute no policy log probability. UTF-8 and EOT receive
sequence-level credit through the trajectory advantage, and their coarse
credit assignment is reported as a limitation.

Start RL with absorbing diffusion, where a selected byte is committed once.
Uniform revision diffusion requires a separate path-likelihood implementation
and is not silently routed through this loss.

Do not begin post-training until scratch pretraining produces coherent high-NFE
generations.
DiffusionGemma shows that low-NFE SFT failures can look deceptively confident
because repetition collapses entropy.

## Inference modes

### AR

Generate one atomic id at a time from the base-only causal path. This is the
quality oracle, BPB evaluator, and fallback after sampler failure.

### Introspective strided decoding

Start with strides `N in {2, 4, 8}`. Bytes have a much smaller vocabulary than
subword tokens, but longer positional horizons, so I-DLM's token stride cannot
be copied without measurement.

Each step:

1. emits one exact causal byte and `N - 1` proposals;
2. on the next pass, obtains causal anchor probabilities for prior proposals
   while drafting new proposals;
3. accepts each proposal with `min(1, p(byte) / q(byte))`;
4. on rejection, samples from the normalized positive part of `p - q`, commits
   it, and discards later proposals;
5. commits a bonus causal byte when all proposals are accepted.

Measure acceptance by byte offset and by UTF-8 role: ASCII, leading byte,
continuation byte, atomic special. Aggregate acceptance alone can conceal a
failure on multibyte characters or EOT.

### Canvas diffusion

Initialize 512 unresolved positions and use a sampler contract matched to the
corruption process:

- absorbing mode predicts only `[MASK]` positions and permanently reveals the
  selected bytes; it cannot revise an earlier reveal;
- uniform-replacement mode may revise categorical byte ids and requires its own
  transition probabilities, self-conditioning, and path-likelihood accounting.

Do not route uniform revision through absorbing reveal/replay/RL code. Every
denoising forward re-encodes 128 fixed four-byte canvas patches, runs the
dual-mode global backbone over those canvas latents with cached prefix K/V,
and runs the local decoder over 512 byte positions. After convergence, truncate
the visible result at the first final atomic EOT and causally append the kept
bytes to update prefix caches. The absorbing sampler must support:

- entropy-bounded progressive unmasking;
- fixed-NFE schedules for controlled comparisons;
- adaptive entropy plus consecutive-argmax stability stopping;
- AR verification of the draft as a BLT-DV baseline;
- final-commit truncation at a stable EOT whose earlier prefix is resolved,
  without exposing later positions.

An absorbing sampler may reveal EOT only after every earlier position is
resolved. Once revealed, mark the suffix inactive immediately; suffix bytes do
not condition the kept prefix, enter caches, or appear in output. This avoids
training on document-bounded canvases but sampling with visible junk after EOT.

Diffusion steps are set for training targets and primary comparisons. Always
report fixed 48-, 24-, and 12-NFE results first. Adaptive inference is a later
serving policy with a hard cap of 48: it may change how many positions are
revealed from entropy and stop only after consecutive-argmax stability plus
valid EOT handling. Do not train a learned stopping policy until fixed
schedules are stable, or denoiser quality and stopping behavior become
confounded. Fixed NFE can replay one captured denoise graph. Adaptive batching
must use device-side active masks up to the cap or accept losing whole-loop
CUDA graph capture.

The fixed schedule is reproducible. For canvas valid length `V`, NFE budget
`S`, and step index `s` from zero through `S`, define
`remaining(s, S) = ceil(V * (1 - s / S) ** gamma)`, with `gamma = 1` in the
baseline. At forward `s`, reveal exactly
`remaining(s, S) - remaining(s + 1, S)` lowest-entropy masked positions; the
last forward therefore forces zero remaining masks. Evaluate `gamma in {1,
2}` only as a named ablation. Sampler-aligned replay samples `s` uniformly from
`0..S - 1` unless a different distribution is explicitly compared. A static
fixed-schedule batch may execute dummy graph iterations after a row reaches
EOT, but that row is inactive and its actual useful NFE is recorded. Adaptive
serving may remove or stop completed rows.

Generation speed is reported as committed raw bytes per full forward and raw
bytes per second, not canvas positions touched per forward. Valid-UTF-8 rate
is a separate quality metric.

## Implementation layout

Implemented files:

```text
pretraining/byte_diffusion/
    FAMILY.md
    __init__.py
    config.py              # versioned architecture/objective dataclasses
    data.py                # atomic vocabulary and document-aligned packing
    tokenizer.py           # strict UTF-8, typed controls, streaming decode
    corruption.py          # sampled-canvas and all-mask construction
    attention.py           # native varlen Flash and sparse Flex dispatch
    kernels.py             # fused 261-way categorical and 512 reveal kernels
    masks.py               # causal, canvas, and introspection masks
    model.py               # shared prefix backbone and dual-mode decoder
    objectives.py          # AR, introspective, and canvas losses
    export.py              # exact parameter table and quantized artifact
    state.py               # transactional prefix/speculation/canvas caches
    inference.py           # typed hierarchical K/V prefill and cached canvas NFE
    sampling.py            # AR, ISD, canvas, and canvas+verify samplers
    distillation.py        # optional trajectory distillation schema/losses
    rl.py                  # optional path log-probability and reward losses
    metrics.py             # BPB, acceptance, UTF-8, NFE, latency counters
    papers/
scripts/
    build_k3_pretrain_dataset_checkpointed.py  # accepts --tokenizer utf8_bytes
    build_byte_diffusion_dataset.py
    materialize_byte_corpus_gpt2_view.py
    train_byte_diffusion.py
    eval_byte_diffusion.py
    benchmark_byte_diffusion.py
    benchmark_byte_diffusion_inference.py
    export_byte_diffusion.py
pretraining/tests/
    test_byte_diffusion_corruption.py
    test_byte_diffusion_masks.py
    test_byte_diffusion_model.py
    test_byte_diffusion_objectives.py
    test_byte_diffusion_sampling.py
    test_byte_diffusion_state.py
    test_byte_diffusion_inference.py
    test_byte_diffusion_distillation.py
    test_byte_diffusion_rl.py
    test_byte_diffusion_export.py
```

Do not modify `pretraining/bolmo.py` or `pretraining/train_bolmo.py` to host the
family. They remain the reproducible source lineage. Extract genuinely shared
byte data/accounting code only when the new implementation needs it.

Every training, generation evaluation, latency benchmark, and model workload
is submitted through `mlq` with `--max-parallel-runs 1`. CPU-only static and
unit tests may run directly.

## Detailed implementation order

### 1. Contracts and reference implementations

- Define versioned configs and an explicit id manifest including `[MASK]` and
  padding.
- Implement pure-PyTorch corruption, masks, clean shifted labels, corrupted
  same-position labels, optional Fast-dLLM shifted corrupted labels, `p/q`
  acceptance, and canvas schedules.
- Add exhaustive small-tensor tests for information flow and target alignment.
- Add a slow categorical reference sampler used only as a correctness oracle.

Exit criterion: mask matrices and losses match hand-computed examples; exact
ISD sampling matches the AR categorical distribution statistically and under
fixed RNG cases.

### 2. Scratch dual-mode model and exact budget

- Implement the latent-global and local-decoder blocks with causal and canvas
  masks; initialize every parameter from scratch.
- Emit an exact named-tensor parameter table and a real quantized round-trip
  artifact before training. Fail if parameters exceed 24.8M or the complete
  artifact exceeds 16,000,000 bytes.
- Add shared clean-prefix global K/V, `M` branch views, and aligned global-patch
  conditioning in the local decoder.
- Prove that changing any unresolved canvas id cannot change a clean-prefix
  logit and that branch `m` cannot read its current clean targets.
- Implement actual sparse/window attention; a logical boolean mask over a dense
  long sequence does not satisfy this step.

Exit criterion: exact parameter/artifact accounting, bounded memory at
128 latent/512 byte canvas positions, transactional cache tests, and no prefix
recomputation per branch or denoising step.

### 3. Kernel and memory characterization

- Benchmark local/global blocks, `M in {1, 2, 4}`, and corrupted lengths 16,
  32, 128, 256, and 512 under the kernel plan above.
- Sweep microbatch and checkpoint policy per clean-length bucket.
- Record step time, achieved FLOP/s, positions/s, power, allocated/reserved
  VRAM, kernel gaps, and SDPA backend; fail on dense-mask fallback.
- Select the fastest configuration that preserves the exact reference output.

Exit criterion: a credible 32GB operating point and a measured path toward the
approximately four-second 5090-equivalent step needed for the challenge cap.

### 4. Scratch-pretraining ablations

- Train a causal-only control with the identical architecture and device-time
  budget; it is an experimental control, not a warm-start checkpoint.
- Train `bd_blt16_ref` from random initialization, then ablate block 32. These
  retain Fast BLT's decoder masks and same-position target while deliberately
  using fixed routing and sampled nonoverlapping blocks totaling 512 corrupted
  positions rather than entropy patching and every eligible patch start.
- Only after that reference works, enable noisy encoder/global latents at
  `C = 128`; compare `absorbing_rb` with `allmask_50`, then whole-patch and
  contiguous-patch-span corruption only if local reconstruction remains a
  failure mode. Take the winner to 256 and 512. Start every length with `M = 1`;
  test 2 and 4 only if extra branches improve quality per device time.
- Compare at both clean-byte exposure and device time. Report per-`K`,
  all-mask, UTF-8-role, EOT, and AR BPB metrics.
- Promote only measured changes; retain the Parameter Golf requirement of more
  than 0.005 BPB improvement at 2,000 steps for a pretraining architecture
  change.

Exit criterion: coherent 48-NFE all-mask continuation, normalized causal BPB,
deterministic resume, and a measured quality/device-time advantage over the
same model trained causal-only.

### 5. Optional sampler-aligned replay

- Mix 50% analytic exact-`K` states with 50% detached states from the model's
  fixed-quota 48-NFE position-selection trajectory, teacher-forcing revealed
  values in the baseline.
- Use zero self-conditioning at the first step and detached prior predictions
  afterward; compare a declared conditioning-dropout rate separately.
- Sample forward index uniformly, replay one state with gradients, and keep all
  rollout steps no-grad.
- Compare against continued analytic-noise training at identical device time;
  this is not a mandatory curriculum stage.
- Preserve fixed-schedule 48-NFE quality and the causal BPB guard before
  evaluating adaptive stopping.

Promotion criterion: better held-out generation or fewer fixed denoising
forwards at matched quality, with no AR BPB regression above 0.005.

### 6. Introspective proposal training and ISD

- Train proposal-only gated residuals from the accepted pretrained model.
- Evaluate strides 2, 4, and 8 at equal device time.
- Add exact and near-lossless modes.
- Benchmark batch/concurrency 1, 8, 32, 128, and 384 against byte AR, BOLMo,
  and the latest nanoGPT model.

Promotion criterion: unchanged base-only AR BPB/quality, introspective
acceptance high enough to improve end-to-end bytes/s at concurrency 384, and
no regression in atomic EOT or UTF-8 behavior.

### 7. Optional sampler distillation and RL

- Implement the recorded-state schema and `CE + KL + AR` objective exactly as
  specified above; record high-NFE states and distributions, not only final
  canvases.
- Distill 48 to 24 forwards, freeze the accepted 24-NFE model, then distill it
  to 12 with held-out trajectory gates.
- Implement RL only after imitation is stable; replay recorded action log
  probabilities in a reference loss test before any reward run.
- Compare online distillation, offline replay, and AR-verified canvas drafts at
  equal device time without changing the base-only causal anchor.

Promotion criterion: task-quality retention, stable adaptive stopping, and a
real end-to-end speedup after counting prompt processing, prefix commits,
rejected drafts, output length, and all sampler passes.

### 8. Hybrid routing

- Route high concurrency to AR/ISD and low concurrency to canvas diffusion.
- Fall back to AR after invalid sampler state, repeated low-entropy loops, or
  excessive NFE.
- Select routes using measured batch-size and output-length break-even curves,
  not fixed assumptions from the papers.

## Evaluation matrix

Every reported generation result includes:

- initialization seed, code revision, and checkpoint hash;
- model parameters and serialized bytes;
- scratch-pretraining and optional post-training device time, clean bytes seen,
  and noisy positions seen;
- AR challenge BPB and diffusion loss separately;
- task accuracy/reward and output length;
- valid UTF-8 fraction and replacement-character count;
- EOT precision, recall, and generated-position distribution;
- NFE, committed bytes/forward, rejected bytes, and commit-pass overhead;
- latency and throughput at concurrency 1, 8, 32, 128, and 384;
- peak allocated/reserved VRAM.

For Parameter Golf pretraining, a structural change is retained only if a
2,000-step wall-clock-matched ablation improves BPB by more than 0.005. For
post-training-only generation changes, BPB must be preserved and promotion is
based on the measured quality/throughput Pareto frontier.

## Design bug audit

The implementation received independent learning/correctness and systems red
teams after the first full pass. The review found and this revision fixes:

- mutually incompatible builder and loader schemas, plus an eager Python
  materialization of the whole corpus; v2 uses verified lazy shard caching and
  shard-local epoch order;
- filtering every document/tail shorter than the diffusion canvas and eagerly
  materializing validation; all rows now contribute AR/BPB and validation stays
  a lazy sequence with a manifest-derived identity;
- omission of every document's first-byte target and invalid atomic-target BPB
  normalization;
- BF16-reduced CE, FP32-only CUDA validation, dropped validation tails,
  duplicated DDP validation, and world-size-dependent validation corruption;
- inconsistent byte-versus-patch RoPE units between clean and canvas paths;
- a nominal Fast-BLT recipe that used mean AR CE and an unscaled 32-block
  sample; `paper_sum` now uses full clean sums and an inclusion-weighted block
  estimator;
- local-microbatch AR normalization rather than the update-global/DDP target
  denominator;
- accepted transactions rewinding their RNG, useful-NFE overcount after EOT,
  and divergent non-512 EOT semantics;
- checkpoints/artifacts that lost custom-special semantics and final exports
  that did not require completed-checkpoint/data provenance or dequantized
  evaluation;
- clean-prefix recomputation at every inference NFE; typed K/V caches now match
  the shared-bank reference on CPU and measure 12.45x faster on the 5090;
- hot-path GPU scalar extraction for full-row checks and per-row branch-start
  sampling; full-row metadata stays on CPU and starts are sampled vectorially.

The earlier specification audit also fixed:

- the high-variance sampled `1/t` estimator at large canvas length by defining
  both an inclusion-weighted Fast-BLT control and a Rao--Blackwellized
  `absorbing_rb` baseline;
- a parameter budget inferred by subtraction rather than counted from the new
  architecture;
- an intra-patch-only local encoder that did not implement BLT's causal byte
  window or patch cross-attention pooling;
- a local decoder that consumed about 35% of parameters, versus the heavy-
  global allocation supported by BLT and Fast BLT;
- an undefined decoder input, tied input/output tables, and ambiguous clean
  versus corrupted target alignment;
- conflation of the four-byte compute stride with UTF-8 code-point width, plus
  undefined partial-patch and block-tail handling;
- mandatory noise conditioning and self-conditioning unsupported by the
  absorbing Fast BLT baseline;
- unjustified microbatch capacities and an unjustified `M = 4` default;
- an AR-base-then-replace-its-backbone phase ordering error;
- ambiguous reuse of the 48-NFE teacher for the 24-to-12 transition;
- unspecified prefix sharing, which could nearly double canvas training cost;
- a logical sliding-window mask that could accidentally execute dense
  quadratic attention;
- an unbudgeted self-conditioning projection;
- free-running absorbing replay that could condition clean targets on
  irreversibly wrong committed bytes;
- an undefined fixed-NFE reveal quota and ambiguous self-conditioning state;
- cross-document/post-EOT target leakage and incomplete patch-state alignment;
- zero scratch gates that initially blocked gradients into new paths;
- update-count exposure math mislabeled as an 80/20 device-time split;
- use of board power as an optimization objective.

The following are blocking tests, not documentation niceties:

- enumerate every allowed attention edge for two clean blocks and two canvas
  branches, then perturb forbidden clean targets and prove logits do not move;
- verify clean shifted and corrupted same-position labels around byte `0x0A`,
  every UTF-8 width, EOT, mask, pad, and the last valid target;
- prove exact-`K` sampling never emits zero active targets, overlaps branches,
  corrupts padding, or loses determinism across resume;
- compare dense reference attention with Flash/Flex kernels in forward and
  backward before performance tests;
- prove `begin`, `commit`, rejection replay, EOT truncation, and `abort` leave
  transactional caches identical to full causal recomputation;
- generate the exact tensor table and quantized round-trip artifact, then
  evaluate BPB after dequantization;
- assert the chosen attention backend and detect graph breaks, recompiles, CPU
  synchronization, and hidden prefix recomputation in performance tests.

## Known risks

- Scratch diffusion may need more optimization steps than the 10-minute
  challenge budget permits.
- Fixed patching may lower BPB even if it improves proposal/anchor agreement.
- Learned dynamic patching may make proposal distributions too different from
  clean causal anchors for useful ISD acceptance.
- A 128-latent global canvas pass may still lack the capacity of
  DiffusionGemma's full-backbone 256-token canvas pass.
- An uncompressed 512-position global denoiser may be compute-bound and
  eliminate the inference advantage at rollout batch sizes.
- The model may learn UTF-8 local validity while failing long-range semantic
  consistency across a 512-byte canvas.
- Adaptive entropy stopping may terminate repetitive failures early and
  falsely report low NFE.
- Faster responses may simply be shorter responses; length-normalized quality
  and full-trajectory reward are mandatory.

## Decisions deliberately left to ablation

- fixed versus strictly causal learned patching;
- absorbing masks versus uniform atomic corruption;
- 128-latent hierarchical versus uncompressed 512-position global canvas;
- full fine-tuning versus proposal-only residual adaptation;
- canvas lengths 128, 256, and 512;
- one, two, or four canvas branches per clean row;
- ISD strides 2, 4, and 8;
- entropy-bounded, confidence-based, or distilled acceptance;
- pure canvas generation versus AR-verified canvas drafting.

The family target is a 512-byte canvas, but 512 is a target to measure rather
than a conclusion encoded into the data pipeline.

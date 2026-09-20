# K3-inspired pretraining campaign

This is the longer off-challenge foundation-model campaign intended to produce
a stronger base for post-training. Its 20,000 updates do not fit the
Parameter Golf submission's 10-minute training limit; the challenge run
remains a separate, shorter export recipe.

This campaign translates the text-side parts of Kimi K3 to the measured
`KKKDKKKD` nano backbone. It is inspired by K3, not a reproduction: the K3
report names Web, Code, Mathematics, and Knowledge as its text domains but
does not publish source weights.

The full campaign uses `k3_weights_quality.json`: 50% web, 20% code, 15%
math, and 15% knowledge. Raw FineWeb is only 5% of the stream; 45% is
quality-filtered and deduplicated FineWeb-Edu. This preserves a small
challenge-distribution anchor without allowing lower-quality generic web text
to dominate the base model.

## Corpus v5

`k3_sources.json` pins eleven sources and their revisions. The builder:

- allocates exact token budgets rather than sampling documents;
- interleaves sources by least completed token fraction;
- applies domain-aware document heuristics;
- performs exact cross-source deduplication and conservative
  formatting-insensitive deduplication for prose only;
- excludes exact DAPO/AIME problem keys where the source exposes problems;
- uses stable hash validation splits sized to at least one million tokens per
  domain (2% for the broad mix, 5% for `web80`);
- retains the original FineWeb validation set for challenge comparability;
- writes loader-aligned shards with one-token boundary overlap;
- records input hashes, selected source metadata, rejection counts, and
  per-shard source token counts.

Remote data and corpus construction are ML preprocessing workloads:

```bash
mlq submit --name k3_sources --cwd "$PWD" --max-parallel-runs 1 -- \
  python3 scripts/prepare_k3_pretrain_sources.py

mlq submit --name k3mix_v5_gpt2_2k --cwd "$PWD" --max-parallel-runs 1 -- \
  python3 scripts/build_k3_pretrain_dataset.py \
    --output data/datasets/k3mix_v5_gpt2_2k \
    --training-steps 2000

mlq submit --name k3mix_v5_web80_gpt2_2k --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  python3 scripts/build_k3_pretrain_dataset.py \
    --weights pretraining/k3_weights_web80.json \
    --output data/datasets/k3mix_v5_web80_gpt2_2k \
    --training-steps 2000 \
    --validation-permille 50
```

The generated token shards are for local research. They combine data with
different attribution and redistribution terms; do not redistribute them
without reviewing the source-level licenses and underlying-content rights
recorded in the source manifest.

## Byte-native diffusion corpus

Byte diffusion uses the production byte encoding during corpus selection. It
does not select documents under GPT-2 or ToaST token counts and decode them
again. The current 2,000-update baseline includes MathGLM v6 at 6% and the
audited arithmetic drills at 4%. It deliberately uses one shuffled mixture,
not an unmeasured late-stage anneal:

```bash
mlq submit --name bd_mathglm_v6_bytes_2k_data --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/build_k3_pretrain_dataset_checkpointed.py \
    --weights pretraining/k3_weights_math30_mathglm.json \
    --tokenizer utf8_bytes --full-source-set --training-steps 2000 \
    --train-batch-tokens 2097152 \
    --cache-dir data/datasets/bd_mathglm_v6_bytes_2k.cache \
    --output data/datasets/bd_mathglm_v6_bytes_2k

mlq submit --name bd_mathglm_v6_atomic_2k_data --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/build_byte_diffusion_dataset.py \
    --output data/byte_diffusion \
    --train 'data/datasets/bd_mathglm_v6_bytes_2k/fineweb_train_*.bin' \
    --validation 'data/datasets/bd_mathglm_v6_bytes_2k/fineweb_val_*.bin' \
    --chunk-size 8192 --chunks-per-shard 256 \
    --max-train-documents 512000 \
    --require-one-train-chunk-per-document
```

Here “2k” means 2,000 byte-diffusion updates. The default global batch is 256
document-aligned rows, so the atomic builder pins exactly 512,000 rows and the
trainer visits every row exactly once. The source build's 4,194,304,001-position
budget is a worst-case envelope: because each UTF-8-safe source document is at
most 8,192 atoms, it guarantees enough complete documents even when every one
fills a row. Short documents leave storage PAD, so the manifest records the
actual literal-byte, special-atom, scored-target, and PAD exposure rather than
claiming 4.194B clean bytes. The second command only adds patch-aligned storage
metadata; it performs no tokenization or lossy text round trip. The exact-budget
source stream may end inside its final document; that one recorded incomplete
tail is excluded rather than converted into a synthetic EOT target.

After the atomic build, copy `payload_sha256` from
`data/byte_diffusion/manifest.json` into the checked launch contract:

```bash
mlq submit --name bd_canvas512_scratch_v1_2k --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/ablation.py --steps 2000 \
    --name bd_canvas512_scratch_v1_2k \
    --script scripts/train_byte_diffusion.py \
    --env BYTE_DIFFUSION_PRESET=canvas512_scratch_v1 \
    --env BYTE_DIFFUSION_MICROBATCH=24 \
    --env BYTE_DIFFUSION_MICROBATCH_TOKEN_BUDGET=208896 \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env BYTE_DIFFUSION_VALIDATION_CHUNKS=2048 \
    --env BYTE_DIFFUSION_EXPECTED_DATA_SHA256=<payload_sha256> \
    --env BYTE_DIFFUSION_DATA_PATH=data/byte_diffusion
```

The preset fails closed if the recipe, absorbing corruption, 512-byte canvas,
single branch, equal-mean joint objective, global optimizer batch, optimizer,
schedule, seed, model layout, or dataset hash drifts. A staged data curriculum
must be introduced as its own ablation; shuffling a nominal final stage into
the bulk is not an anneal.

The periodic 2,048-row readout is emitted as `val_proxy_bpb`. It is an exact
causal codelength on a deterministic subset and is appropriate for paired
ablation curves, but it is not the full challenge score. After training,
evaluate the complete source-bound FineWeb validation split:

```bash
mlq submit --name bd_canvas512_scratch_v1_full_eval --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/eval_byte_diffusion.py \
    --checkpoint ablation_results/bd_canvas512_scratch_v1_2k/checkpoint.pt \
    --data-path data/byte_diffusion --full-validation
```

Only that command emits `val_challenge_bpb`. The separate
`val_diffusion_loss` is mean denoising CE, not an ELBO, likelihood, or BPB.

For a matched nanoGPT AR experiment, derive GPT-2 ids from those already
selected complete documents:

```bash
mlq submit --name bd_mathglm_v6_gpt2_view --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/materialize_byte_corpus_gpt2_view.py \
    --source data/datasets/bd_mathglm_v6_bytes_2k \
    --output data/datasets/bd_mathglm_v6_same_docs_gpt2
```

That view can contain a different number of model positions—that is the real
compression difference—but its manifest proves document selection did not
change.

The short reference runs use the sampled source set above. Full pretraining
downloads the expanded revision-pinned source set and builds a one-pass
10.49B-token stream. The checkpointed builder fails rather than cycling if any
source cannot fill its budget and atomically preserves every completed
source's preprocessing work.

## Full pretraining

Build the 20,000-step, 10.49B-token corpus:

```bash
mlq submit --name k3_quality_sources_full --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  python3 scripts/prepare_k3_pretrain_sources.py --full-source-set

mlq submit --name k3_quality_20k_data --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  python3 scripts/build_k3_pretrain_dataset_checkpointed.py \
    --weights pretraining/k3_weights_quality.json \
    --full-source-set \
    --training-steps 20000 \
    --cache-dir data/datasets/k3mix_quality_gpt2_20k.cache \
    --output data/datasets/k3mix_v7_quality_gpt2_20k
```

Then run the KKKDKKKD curriculum:

```bash
mlq submit --name k3_quality_20k_train --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  /home/marvin/Documents/repositories/parameter-golf/.venv/bin/python \
  scripts/run_k3_context_curriculum.py \
    --data data/datasets/k3mix_v7_quality_gpt2_20k \
    --run-id k3_quality_20k \
    --steps 20000 \
    --kda-heads 3 \
    --per-head-muon \
    --lr-schedule cosine
```

The curriculum keeps the 524,288-token global batch and 32,768-token
microbatches fixed while moving from 2K to 4K to 8K context over
15,000/3,750/1,250 steps. It uses cosine decay with a 1% linear warmup,
current quintic Newton-Schulz Muon, a 500-step momentum warmup, and per-head
Muon for Q/K/V. Stage boundaries resume model, optimizer, RNG, and exact
corpus position. Every update consumes a new part of the stream; no epoch
wrap occurs. Exact staged resume is intentionally single-GPU only because
Muon optimizer state is rank-sharded.

### K3 Stable LatentMoE variant

Pass `--latent-moe` to replace the two legacy dense MLPs with the repository's
compute-matched K3 Stable LatentMoE in all eight layers:

```bash
mlq submit --name k3_latent_moe_20k --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  /home/marvin/Documents/repositories/parameter-golf/.venv/bin/python \
  scripts/run_k3_context_curriculum.py \
    --data data/datasets/k3mix_v7_quality_gpt2_20k \
    --run-id k3_latent_moe_20k \
    --steps 20000 \
    --kda-heads 3 \
    --per-head-muon \
    --lr-schedule cosine \
    --latent-moe
```

The default MoE has 32 routed experts, top-2 routing, a 128-wide latent,
256-wide routed SiTU-GLU experts, and two full-input shared experts with 64
hidden units each. It grows the 64.05M-parameter backbone to 87.73M while its
positionwise matrix work is only 2.0% above the old two-MLP layout. Routing
uses FP32 sigmoid scores, selection-only correction bias, normalized unbiased
top-k weights, and the report's 1,000-bin Quantile Balancing update. The
correction bias is updated between optimizer steps and is checkpointed; SFT
and RL freeze the pretrained router weights while continuing causal Quantile
Balancing so the bias follows changes in the router's hidden inputs. The
experts and surrounding trunk continue to adapt. RL constructs its
from-scratch value trunk without MoE: sparse
random routing is a poor fit for the critic and would duplicate the actor's
extra optimizer/activation memory without improving policy capacity.

If corpus preprocessing is interrupted, submit the same build command again.
Verified source caches are reused; only the source that was incomplete at the
time of failure is restarted. Final loader-shard assembly is a cheap
sequential pass over those caches.

The runner streams all three stages into
`ablation_results/k3_quality_20k/metrics.jsonl` and
`tb_logs/k3_quality_20k`, so TensorBoard shows a single continuous 0–20,000
step curve. Corpus download and construction do not emit TensorBoard metrics.


## Corpus v8: math-30% with a mid-training anneal

The `quality` profile spends 15% of the stream on mathematics. A 2026-08-06
audit of what that buys found the shortfall is not in word problems but
below them: the corpus contains very little text that carries out an
arithmetic procedure step by step, which is the one thing post-training can
refine but cannot install. Corpus v8 addresses that in four ways.

**Math is 30% of the stream** (`k3_weights_math30.json`), paid for out of web,
which drops from 50% to 42% and keeps only the educationally filtered corpus
plus a small raw-FineWeb anchor.

**A worked-step drill corpus.** `scripts/build_math_drills.py` generates
deterministic column-addition, partial-product, long-division, fraction, and
percentage drills that show every intermediate step, across fifteen families
and a declared digit curriculum. Two thirds are written in expanded form and
one third in least-significant-digit-first column form, so the model sees more
than one procedure for the same fact. Every drill is checked against the
problem registry before it is written, and a disjoint held-out probe
(`postraining/arithmetic_probe.py`) is generated in the same command so
arithmetic accuracy can be reported per family and per digit count instead of
being averaged into a single math number.

**DeepMind modules are weighted rather than round-robined.** All 56
`train-easy` modules used to contribute equally; `module_weights` in
`k3_sources.json` now gives arithmetic 5-6x, place value and rounding 4x, and
measurement 3-4x the share of the modules the model already handles.

**A mid-training anneal.** `--anneal-weights` draws the final
`--anneal-fraction` of the token budget under a second profile
(`k3_weights_math30_anneal.json`: math 50%, no raw web). The two stages share
their source iterators, so the anneal continues each source rather than
restarting it and cannot turn the tail of a one-pass corpus into a second
epoch of its heaviest sources. This is distinct from `--cooldown-data`, which
swaps in a whole second dataset for the final context stage; the anneal is one
stream, with `stages` in the dataset manifest recording what each stage drew.

Embedding tying needs no change: `TIE_EMBEDDINGS` already defaults to on, and
51.5M of the 64.0M parameters are embeddings.

### Prerequisites

The corpus builders now fail closed without a problem registry, which is the
single global assignment of every math problem to one split:

```bash
mlq submit --name k3_v8_problem_registry --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/build_problem_registry.py \
  --output data/problem_registry/v1

mlq submit --name k3_v8_math_drills --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/build_math_drills.py \
  --count 2000000 --output data/math_drills/v1
```

Both are pure CPU, but still use the single-workload queue. The registry covers 344,309
distinct problems (2,472 eval, 204,430 RL, 137,407 SFT) and indexes 4.0M
protected 13-grams over the eval, RL, and SFT splits. `--index-splits`
narrower than that is refused by `ProblemGuard`, because exact key matching
alone catches a held-out problem only when an entire web page equals its
statement.

The registry build now ends with a positive control: it re-reads
`--control-samples` protected problems per source, queries the finished index
for each, and fails the build if any indexable one is not detected. That turns
`index.json`'s declared split coverage from a claim nothing verifies into a
measurement taken while the source text is still in hand. The shipped
`data/problem_registry/v1` predates the control and was verified separately
(2,229/2,229 sampled problems detected across all three splits), so it does
not need rebuilding.

**Residual exposure, and it is not the index.** 518,560 of 2,861,011 protected
rows yield no indexable 13-gram and are therefore protected by exact key alone
— which for web text catches a page only if the page *is* the problem. There
are two reasons, and the manifest's `short_rows` names only the first: a row
can be shorter than one window, or long but so repetitive that every window is
one token repeated (`Simplify (d*d*((d*d**3)/d)/d)/(d/d**6)` is 18 words of
which 12 are `d`). It concentrates in the RL template pools: 255 of 300
sampled `deepmind_interpolate_rl_full` rows are too short, none repetitive.
Those problems are short generated templates, so verbatim reproduction in web
text is unlikely, but this is the real hole in the decontamination, not
anything the n-gram index does.

Measure what the guard will remove before spending build time on it:

```bash
.venv/bin/python scripts/audit_problem_overlap.py \
  --bytes-per-source 25000000 --report data/problem_registry/v1/overlap.json
```

On a 25 MB-per-source sample this reports `fineweb` 0.00%, `fineweb_edu_dedup`
0.06%, `open_web_math` 1.58%, `finemath_4plus` 2.45%, `deepmind_math` 5.18%,
and `openmath_instruct` 16.45% exact plus 15.98% n-gram. The last two are
genuine overlap with the RL and SFT pools, not false positives: OpenMathInstruct's
GSM8K band is 97% GSM8K, and `deepmind_interpolate_rl_full` is drawn from the
same generator as `deepmind_math`.

Those are *document* rates, and a DeepMind document packs 8-24 independent
question/answer pairs. The builder therefore drops contaminated items rather
than whole documents wherever a source declares one quality key per segment,
so the 5.18% document rate costs far less than 5.18% of the pairs. Measured
over 4,000 DeepMind documents (63,970 items): 0.356% of items are
contaminated, 5.13% of documents hold at least one, and item-level trimming
releases 3,424 clean siblings — 5.35% of the stream — that whole-document
rejection would have discarded. No document lost every item, and no trimmed
document was newly rejected on the rejoin.

Item counts and document counts are different units, so the manifest keeps
them apart: `rejection_counts` counts documents, `item_rejection_counts`
counts items. Summing the two would overstate documents refused by roughly
16x on the one key. Sources whose segments are not item-aligned -- a web page,
an OpenMath row -- are still admitted or refused whole.

`sample_tokenizer_corpus.py` runs the same guard. A tokenizer is fitted to its
sample, so a held-out problem in that sample buys shorter encodings of exactly
the text the model is judged on -- quieter than training on it, and just as
much a leak.

### The four-arm ablation

Four 20,000-step runs isolate one change each. Every arm shares the model,
optimizer, curriculum, and problem registry; only the named factor moves.

| arm | tokenizer | corpus | isolates |
| --- | --- | --- | --- |
| A | GPT-2 50,257 | `quality` (math 15%) | control, comparable to `k3_quality_20k` |
| B | GPT-2 50,257 | `math30` + anneal | the data change alone |
| C | ToaST+TST 50,257 | `math30` + anneal | the tokenizer at a matched vocabulary |
| D | ToaST+TST 16,384 | `math30` + anneal | reallocating embedding budget to depth |

A to B answers "does more mathematics help"; B to C answers "does the
tokenizer help at the same embedding cost"; C to D answers "is 13% better
compression worth more than 34,000 embedding rows". Bits-per-byte is
tokenizer-independent, so all four arms are directly comparable on it, and the
existing runs in `ablation_results/` stay comparable too.

Arms C and D need their tokenizers trained first (pure CPU, still queued under
the repository workload policy; see `tokenization/README.md`):

```bash
mlq submit --name k3_v8_tokenizer_sample_math30 --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/sample_tokenizer_corpus.py \
  --weights pretraining/k3_weights_math30.json \
  --output data/tokenizer_corpus/train.jsonl \
  --validation-output data/tokenizer_corpus/val.jsonl \
  --bytes 200000000

mlq submit --name k3_v8_toast_tst_n1_50k --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" tokenization/train.py \
  --corpus data/tokenizer_corpus/train.jsonl \
  --validation data/tokenizer_corpus/val.jsonl \
  --output data/tokenizers/toast_tst_n1_50k \
  --vocab-size 50257 --group-size 1 --baseline-gpt2 \
  --min-count 60 --max-ngram 32 --max-trees 150000

mlq submit --name k3_v8_toast_tst_n1_16k --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" tokenization/train.py \
  --corpus data/tokenizer_corpus/train.jsonl \
  --validation data/tokenizer_corpus/val.jsonl \
  --output data/tokenizers/toast_tst_n1_16k \
  --vocab-size 16384 --group-size 1 --baseline-gpt2 \
  --min-count 60 --max-ngram 32 --max-trees 150000
```

Sample the tokenizer corpus under the *same* weight profile the model will
train on. A tokenizer fitted to a 15%-math mixture spends its vocabulary in
the wrong place for a 30%-math one.

Then build the four corpora. Each is a separate `mlq` job with
`--max-parallel-runs 1`, and each takes its own cache directory:

```bash
# Arm A -- control
mlq submit --name k3_v8_armA_data --cwd "$PWD" --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/build_k3_pretrain_dataset_checkpointed.py \
    --weights pretraining/k3_weights_quality.json \
    --full-source-set --training-steps 20000 \
    --cache-dir data/datasets/k3mix_v8_armA.cache \
    --output data/datasets/k3mix_v8_armA_quality_gpt2_20k

# Arm B -- math30 + anneal, GPT-2
mlq submit --name k3_v8_armB_data --cwd "$PWD" --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/build_k3_pretrain_dataset_checkpointed.py \
    --weights pretraining/k3_weights_math30.json \
    --anneal-weights pretraining/k3_weights_math30_anneal.json \
    --anneal-fraction 0.15 \
    --on-exhausted redistribute \
    --full-source-set --training-steps 20000 \
    --cache-dir data/datasets/k3mix_v8_armB.cache \
    --output data/datasets/k3mix_v8_armB_math30_gpt2_20k

# Arm C -- same corpus, ToaST+TST at a matched vocabulary
mlq submit --name k3_v8_armC_data --cwd "$PWD" --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/build_k3_pretrain_dataset_checkpointed.py \
    --weights pretraining/k3_weights_math30.json \
    --anneal-weights pretraining/k3_weights_math30_anneal.json \
    --anneal-fraction 0.15 \
    --on-exhausted redistribute \
    --tokenizer data/tokenizers/toast_tst_n1_50k \
    --full-source-set --training-steps 20000 \
    --cache-dir data/datasets/k3mix_v8_armC.cache \
    --output data/datasets/k3mix_v8_armC_math30_toast50k_20k

# Arm D -- ToaST+TST at 16,384
mlq submit --name k3_v8_armD_data --cwd "$PWD" --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/build_k3_pretrain_dataset_checkpointed.py \
    --weights pretraining/k3_weights_math30.json \
    --anneal-weights pretraining/k3_weights_math30_anneal.json \
    --anneal-fraction 0.15 \
    --on-exhausted redistribute \
    --tokenizer data/tokenizers/toast_tst_n1_16k \
    --full-source-set --training-steps 20000 \
    --cache-dir data/datasets/k3mix_v8_armD.cache \
    --output data/datasets/k3mix_v8_armD_math30_toast16k_20k
```

A corpus cache is bound to its tokenizer, weight profiles, anneal fraction,
registry hashes, and the decontamination code itself, so a cache built for one
arm will refuse to resume into another rather than mixing them.

**Build arm C first.** At a fixed token budget the ToaST-50k arm must read
~7.5% more source text than the GPT-2 arms (3.7288 vs 3.4673 bytes/token), so
it is the arm most likely to run a thin source dry. `build_source_caches`
fails closed on that -- `RuntimeError: source {name} exhausted at N / M
tokens` -- rather than cycling, which is the right behaviour but is a late and
expensive place to find out. Building the hungriest arm first surfaces any
source that needs another shard before three other corpora have been built
against it.

Then train each arm. The curriculum runner reads
`tokenizer_provenance.vocab_size` out of the dataset manifest and passes the
padded width to the trainer, so arms C and D need no extra flag:

```bash
mlq submit --name k3_v8_armA_train --cwd "$PWD" --max-parallel-runs 1 -- \
  .venv/bin/python scripts/run_k3_context_curriculum.py \
    --data data/datasets/k3mix_v8_armA_quality_gpt2_20k \
    --run-id k3_v8_armA --steps 20000 \
    --kda-heads 3 --per-head-muon --lr-schedule cosine
```

Substitute the arm's dataset and `--run-id` for B, C, and D. Run them one at
a time; they share one RTX 5090.

### Reading the result

Report, for every arm:

- bits per byte on each domain validation split, which is the only metric
  comparable across tokenizers;
- the arithmetic probe, per family and per digit count, from
  `data/math_drills/v1/probe.jsonl`;
- GSM8K-test and AIME accuracy from the sealed eval panels;
- the anneal boundary in the loss curve, to check the mixture shift did not
  destabilize training rather than consolidate it.

Do not read a bits-per-byte improvement as a reasoning improvement. A
tokenizer that packs more bytes into each token also gives the model fewer
forward passes per byte, and 6% of the stream being templated drills is a real
risk of a stylistic prior that does not transfer. The probe and the sealed
panels are what decide the ablation; the compression table is not.

**The arms are matched on tokens, not on text.** Every arm trains on
20,000 x 524,288 tokens, so an arm whose tokenizer compresses 7.5% better also
reads about 7.5% more text. That is a genuine part of what a better tokenizer
buys at a fixed compute budget, and it is not a confound to remove -- but it
does mean C-over-B is "better tokenizer *and* more text", not "better
tokenizer alone". The 7.5% is measured, not assumed: on the 4.0 MB held-out
slice of the sample the tokenizer was fit on, GPT-2 gets 3.4673 bytes/token
and ToaST+TST 3.7288 at 50,257. No 16,384 artifact has been built, so arm D
has no measured compression figure at all.

**That advantage does not survive the change of distribution.** Measured on
the raw-FineWeb canonical validation window every model is actually scored on,
over an identical 9,081,775 bytes: GPT-2 needs 2,053,772 tokens (4.4220
bytes/token) and ToaST+TST 2,097,152 (4.3305) -- ToaST is **2.1% worse** at
the same vocabulary. The fitting sample is math-30 weighted and
`numeric.group_size` is 1; the validation shard is raw web. Report the
in-distribution ratio and this one together, and do not carry the fitting-set
number into a claim about evaluation. `--baseline-gpt2` re-measures both on
whatever sample the tokenizer is actually trained from and writes
`compression_ratio` into `<tokenizer>/validation.json`; report that ratio
beside each arm's result so the size of the effect is on the page rather than
inferred.

## Bolmo byteification of the Arm-C checkpoint

`pretraining/bolmo.py` and `pretraining/train_bolmo.py` implement the two-stage
recipe from *Bolmo: Byteifying the Next Generation of Language Models*. This
is a new model lineage; it does not alter the source trainer or checkpoint.
The source is pinned by SHA-256 to the 1,000-step Arm-C KDA/NoPE checkpoint.
NextLat is absent because it was a training-only auxiliary head, while the
retained KDA/MHA trunk and NoPE configuration come directly from the source.

The scaled architecture keeps the paper's one mLSTM+SwiGLU encoder block,
one-byte-lookahead cosine boundary predictor, last-byte pooling, retained
longest-suffix source embeddings, latest-patch depooling, four
mLSTM+SwiGLU decoder blocks, and fused byte/boundary output. The source's five
registered control symbols remain atomic; ordinary text and TST numerics are
losslessly byteified. This is necessary to retain compatibility with later
SFT checkpoints without pretending control symbols are ordinary UTF-8 text.
As in the released recipe, the boundary immediately before EOT is removed;
teacher representations are gathered from their corresponding source-token
positions after this coalescing. Raw post-trunk states flow through depooling,
and the copied source final RMSNorm is applied once at the byte LM head.

Stage 1 freezes the global trunk and trains the local models with the paper's
4:1:1:1 boundary, four-layer stitching, patch-likelihood distillation, and
byte-CE losses. Stage 2 removes the teacher, uses predicted boundaries, and
trains the whole model with fused byte CE plus the externally supervised
boundary loss. A 2,000-update plan has fixed 667/1,333 phase horizons. A
1,000-update stop is the chronological prefix required by the paper schedule:
667 complete Stage-1 updates followed by 333 Stage-2 updates. It never rescales
the phase boundary while retaining longer LR horizons.

The byte dataset uses the source checkpoint's 2,048-position sequence length.
Exactly as in the released data path, each training row is the synthetic BOS
plus the first 2,047 tokens of a 2,048-token source item; the last source token
is skipped. Stage 1 uses 64 examples and Stage 2 uses 128, preserving Table 8's
131,072 and 262,144 patch-axis positions per update. The 12,288-atom cap
preserves the paper's 6x maximum byte expansion and truncates only at complete
source-token patch boundaries. Training is fail-closed if the unique dataset
cannot cover the requested run, and every tensor shard is hash-verified before
training or continuation.

On the RTX 5090, both stages use eight 2,048-position examples per microbatch,
preserving the old 16 × 1,024 source-token microbatch workload. Stage 1 computes
the frozen teacher's selected source-token log probabilities in 4,096-position
chunks. This preserves the paper's selected-token likelihood while
avoiding a multi-gigabyte full-sequence FP32 teacher-softmax tensor. The
selected global batch is sorted by byte length before it is split into
microbatches. This changes neither the sampled examples nor any loss weight;
it only removes padding work caused by placing a short row beside an unrelated
long row. The local mLSTMs use xLSTM's TFLA-derived
`chunkwise--triton_xl_chunk` backend with the paper's FP32 kernel autocast.
This implements the same stabilized recurrence as the released
`triton_limit_chunk` path, which is not viable in this environment: Triton 3.6
rejects its untyped FP64 launch scalar at chunk 128, and chunk 64 is slower.
Forward/backward parity against the native recurrence measured max errors of
1.33e-4 and 1.51e-8 respectively.

Displayed paper equations govern the two ambiguous release-only choices. The
stitch coefficient is 1 (the released 1B launch overrides its fourth internal
stitch term to 4), and the Dolma-2-specific ALM whitespace debias is omitted
because this source tokenizer is ToaST+TST and the paper's patch-likelihood KL
does not define that correction. These are explicit recipe variants to ablate,
not silently imported tokenizer assumptions.

Build the bounded byte dataset from the exact Arm-C math-30% stream:

```bash
mlq submit --name bolmo_armC_byte2048_paper_v3_data --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/build_bolmo_dataset.py \
    --tokenizer data/tokenizers/toast_tst_n1_50k \
    --output data/datasets/k3mix_v8_armC_bolmo_byte2048_paper_v3 \
    --train 'data/datasets/k3mix_v8_armC_math30_toast50k_stop1k/fineweb_train_*.bin' \
    --validation 'data/datasets/k3mix_v8_armC_math30_toast50k_stop1k/fineweb_val_*.bin' \
    --source-tokens 2048 --examples-per-shard 256 \
    --source-model-vocab-size 50304 --byte-length-multiple 128 \
    --max-atomic-tokens 12288 \
    --max-train-examples 240000 --max-validation-examples 1024
```

Run Stage 1 as its own process so the compiler cache and teacher are discarded
before Stage 2. The held-out gate uses the source-matched 2,097,152-token span,
requires at least 99% byte-weighted boundary accuracy, and saves the artifact
before failing closed:

```bash
mlq submit --name bolmo_armC_paper_v3_stage1 --cwd "$PWD" --max-parallel-runs 1 -- \
  "$PWD/.venv/bin/python" scripts/ablation.py \
    --steps 2000 --name bolmo_armC_paper_v3_stage1 \
    --script pretraining/train_bolmo.py \
    --env BOLMO_DATA_PATH=data/datasets/k3mix_v8_armC_bolmo_byte2048_paper_v3 \
    --env STOP_AFTER_STEP=667 --env RUN_ID=bolmo_armC_paper_v3
```

Canonical validation follows the released Bolmo evaluator: it marginalizes the
two fused output classes for each byte before computing BPB, and that
marginalized `val/canonical_bpb` is the number comparable against the paper's.
It is *not* comparable against a subword model's BPB: the boundary predictor is
non-causal, so routing at position `t` consumes a bit derived from byte `t + 1`,
and marginalizing that bit away means never paying for it. `val/joint_bpb`
charges the boundary at `t + 1` from position `t`, one step before routing
reads it, which makes it a valid causal codelength — and
`val/canonical_joint_bpb`, not `val/canonical_bpb`, is what a subword model's
BPB must be compared against. (`val/joint_bpb` is the 256-example proxy's
joint, not the canonical span's.) A 256-example proxy gives
the frequent learning curve; `val/canonical_bpb` uses the same 2,097,152 scored
source tokens as the source run at initialization, the Stage-1 boundary, and
the final step. Validation rows carry one unscored left-context source token,
matching the source trainer's shifted contiguous input/target windows. Every
train and validation row resets TST rendering at its synthetic BOS, so its
atomic stream losslessly decodes to that independent source row.

The performance reference is the completed source run, not an arbitrary
utilization target: its final steady readout was 2,094.88 ms for 524,288
source tokens, or about 250,275 source tokens/s, with 7,656 MiB peak allocated
VRAM. Real-patch Stage 2 initially took roughly 2.65 seconds/update in a clean
process. Two exact integration fixes are retained: local training calls no
longer materialize unused terminal C/N/M state copies, and the byte vocabulary
norm/projection/rational-softcap region is compiled separately just as the
source trainer compiles its vocabulary head. The production default compiler
takes about 2.18--2.31 seconds/update (roughly 112k--120k source tokens/s) and
allocates about 14.8 GiB. Max-autotune reached about 2.09--2.18 seconds on
isolated training updates, but made a 16-example validation probe take 16.4
seconds instead of about 2.1 seconds; at the normal validation cadence it is
slower overall and is not selected. The production path is close to
optimizer-update parity with the old trainer while also processing roughly one
million byte positions through five local recurrent blocks. It is not
source-token-throughput parity: Bolmo deliberately uses half as many source
tokens per update and does substantially more local work per source token.

The old 333/667 run is retained only as a performance diagnostic. It switched
phases halfway through the Stage-1 LR horizon, used 1,024-token data, evaluated
one eighth of the source validation span, and reported joint fused NLL as BPB.
Its checkpoints are rejected by the `paper_v3` provenance gate and must not be
used as quality evidence or as a Stage-2 starting point. Its useful systems
finding remains that mb16 with 4,096 teacher positions per head chunk was the
best measured RTX 5090 Stage-1 configuration.

The student predicts a 522-way fused byte/boundary vocabulary, not the
source's 50k-token vocabulary: 256 literal bytes plus five registered control
symbols form 261 atomic symbols, each paired with a boundary bit. The
paper-aligned model does retain a separate
50,304-row longest-suffix input embedding table: its lookup key is derived
from the raw byte prefix rather than external tokenization, but the table is
still an internal copy of the source vocabulary embeddings. The 50k output
softmax and all other teacher-only source tables are absent after Stage 1.
`stage_source_tokens_per_second` is the wall-clock interval gate (since the
previous log, so first-use compilation does not pollute the final probe);
the `stage_cumulative_*` rates retain whole-stage context.
`stage_source_tokens_per_device_second`, atomic-symbol throughput, CUDA-event
device-interval time, wall time, and peak VRAM are emitted alongside it to
distinguish GPU submission intervals from input/compile stalls. Canonical
`train_time` and `step_avg` use the synchronized wall clock, matching the source
trainer; `train_device_time_ms` is diagnostic only. Stage-2 microbatch 24 did
not beat mb16, and mb32 exceeded 32 GiB even with expandable allocator
segments. Whole-Stage-2 compilation, fused boundary Q/K projection, separately
compiled CE, max-autotune at the validation cadence, explicit XL warp
overrides, CUDA-graph mode, and the reference limit-chunk kernel were rejected
by measured latency, memory use, compiler failure, or runtime compatibility.
Saving all XL forward recurrence states
instead of recomputing them in backward was also slower. The xLSTM wrapper's
distinct terminal-state copies are now disabled because Bolmo never carries
state between training chunks; CPU parity tests cover outputs and every input
and parameter gradient.

Resetting Dynamo at the in-process phase boundary reduced the original
roughly 3.3-second cache-exhaustion path but did not fully isolate Stage 2.
Production continuation strict-loads the Stage-1 artifact in a fresh process,
validates its architecture, source, data, schedule, gate, batch, LR, RNG, and
cursor provenance, and deterministically skips the 42,688 unique examples
already consumed. Submit it only after Stage 1 succeeds:

```bash
mlq submit --name bolmo_armC_paper_v3_1k --cwd "$PWD" --max-parallel-runs 1 \
  --after-success <stage1-job-id> -- \
  "$PWD/.venv/bin/python" scripts/ablation.py \
    --steps 2000 --name bolmo_armC_paper_v3_1k \
    --script pretraining/train_bolmo.py \
    --env BOLMO_DATA_PATH=data/datasets/k3mix_v8_armC_bolmo_byte2048_paper_v3 \
    --env STOP_AFTER_STEP=1000 \
    --env RUN_ID=bolmo_armC_paper_v3 \
    --env BOLMO_STAGE2_RESUME_CHECKPOINT=logs/bolmo_armC_paper_v3_stage1_model.pt \
    --env BOLMO_BACKEND=triton --env BOLMO_CHUNK_SIZE=128 \
    --env BOLMO_AUTOCAST_KERNEL_DTYPE=float32 \
    --env BOLMO_COMPILE=1 --env BOLMO_COMPILE_BYTE_HEAD=1 \
    --env BOLMO_COMPILE_MODE=default \
    --env BOLMO_END_AFTER_GLOBAL_STEP=1000 \
    --env BOLMO_STAGE1_MICROBATCH_EXAMPLES=8 \
    --env BOLMO_STAGE2_MICROBATCH_EXAMPLES=8 \
    --env BOLMO_STAGE1_GLOBAL_EXAMPLES=64 \
    --env BOLMO_STAGE2_GLOBAL_EXAMPLES=128 \
    --env BOLMO_TEACHER_POSITIONS_PER_CHUNK=4096
```

Pin the runtime-only backend, recurrence precision, compiler, and batching
settings explicitly for a staged continuation. Runtime kernel choices are not
part of the paper architecture, while schedule, data, batching, optimizer, and
gate choices are bound into the checkpoint's `paper_v3` training contract.

## Latent feedback ("temporal residual") on nanogpt-mini

`pretraining/nanogpt_mini/nanogpt_mini_feedback_train.py` trains the mini GPT
with the previous position's top-layer state fed back into the stack
(`FB_MODE=glu`: the full-bandwidth transformer's gated fusion, arXiv:2608.08888;
`add`: additive control; `lam`: a decayed linear-attention memory over all
earlier top states read at every layer), with Jacobi passes (`FB_PASSES=2`),
an optional detached carry (`FB_DETACH=1`), jitter (`FB_NOISE`), and a sliding
attention window (`FB_WINDOW`). Validation reports Jacobi passes 1/2/3/8 and
the true sequential recurrence (`val_bpb_seq`). Pre-registration and results:
`NOTES.md`, "Latent feedback ("temporal residual") on the nanogpt-mini
transformer".

LAM keeps the ordinary six-layer, width-512 nanoGPT-mini blocks and adds memory
reads at selected layers of the second pass; it is not the paper's input-only
GLU architecture. CUDA reads use the pinned FLA kernels behind a graph-compatible
forward/backward adapter, and training requires full-graph compilation. Missing
FLA is an error, not a silent slow fallback. `FB_MEMORY_KERNEL=0` selects the
Torch reference explicitly; `FB_MEMORY_COMPILED=0` is the graph-breaking legacy
FLA reference used by the benchmark, not the full-graph trainer.

The matched 1,000-update runs `nanomini_fb_base_perf_1k` and
`nanomini_fb_lam_compiled_1k` scored **1.3433 / 1.3243 BPB** on the same
1M-token subset, at **529.5 / 1,343.9 ms/update** including training compilation
but excluding evaluation. LAM improves BPB by **0.0190** at **2.54x** training
time; this is a matched-token comparison, not a compute-efficiency win.
LAM's pass-one score is 1.3426, so almost all the observed gain needs feedback
at evaluation. Protocol, performance ablations, rejected-candidate source,
and comparisons are in `ablation_results/nanomini_lam_performance/`.

**Recommended next control: first-layer-only LAM**, `FB_MEMORY_LAYERS=0`.
`nanomini_fb_lam_first_1k` completed the same 1,000-update protocol at
**1.3265 BPB / 1,096.5 ms per update**: only **0.0022 BPB** worse than all-layer
LAM, with **18.4% lower training latency** and **2,626,560 fewer parameters**.
It remains 0.0168 BPB better than the plain backbone. `FB_MEMORY_LAYERS=all`
(the compatibility default) retains all six reads; comma-separated sorted
indices such as `0,2,4` select intermediate sites. The checkpoint records the
selection, and unselected query/output projections are not instantiated.

The matched fp32 Torch reference `nanomini_fb_lam_torch_1k` scored 1.3248 BPB
at 1,466.0 ms/update: effectively tied in quality with compiled FLA, rather
than a sustained optimization regression. Evaluating identical FLA-trained
weights with both backends changed BPB by only 0.0000079. See
`ablation_results/nanomini_lam_iteration/{comparison,trajectory_diagnosis}.json`.
The GLU training arm was cancelled by request; its benchmark is not a completed
quality result. Three-site LAM has throughput evidence only. AdaRMSNorm and
progressive-pass scheduling remain untested proposals, not enabled features.

Use fresh output names: the ablation runner replaces an existing run directory.

```bash
mlq submit --name fb_gpu_contracts --cwd "$PWD" --max-parallel-runs 1 --priority 1 \
  --max-attempts 1 --time-limit 40m -- .venv/bin/python -m pytest -q -s \
  pretraining/tests/test_nanogpt_mini_feedback.py pretraining/tests/test_nanogpt_mini_feedback_gpu.py \
  -k 'not trainer_runs and not real_shape_training'
mlq submit --name fb_lam_benchmark --cwd "$PWD" --max-parallel-runs 1 --priority 1 \
  --max-attempts 1 --time-limit 40m -- .venv/bin/python scripts/benchmark_feedback_lam.py \
  --output ablation_results/fb_lam_benchmark/benchmark.json --warmup 5 --samples 20 --accumulation 8
run() {  # fresh run name, mode, optional comma-separated memory layers
  mlq submit --name "$1" --cwd "$PWD" --max-parallel-runs 1 --priority 1 \
    --max-attempts 1 --time-limit 2h -- .venv/bin/python scripts/ablation.py \
    --script pretraining/nanogpt_mini/nanogpt_mini_feedback_train.py \
    --name "$1" --steps 1000 --val-every 20 \
    --env DATA_PATH=data/datasets/fineweb10B_sp1024 VAL_TOKENS=1048576 \
      VOCAB_SIZE=1024 SEQ_LEN=1024 MBS=64 SEED=1337 FB_MODE="$2" \
      FB_MEMORY_LAYERS="${3:-all}" FB_PASSES=2 FB_DETACH=0 FB_NOISE=0.02 FB_WINDOW=0 \
      FB_SEQ_TOKENS=262144 FB_SEQ_EVERY=0 FB_MEMORY_KERNEL=1 FB_MEMORY_COMPILED=1
}
run fb_base_next_1k none
run fb_lam_first_next_1k lam 0
```

## Full-bandwidth nanoGPT-mini pretraining

`scripts/train_nanogpt_mini_full_bandwidth.py` is the dedicated
[Full-bandwidth Transformer (2608.08888v1)](https://arxiv.org/pdf/2608.08888v1)
version. Its importable model and trainer live in
`pretraining/nanogpt_mini/nanogpt_mini_full_bandwidth_{model,train}.py`.
The `nanogpt_mini_feedback_*` LAM experiments are a separate architecture and trainer.

The implementation keeps the six-layer, width-512 mini attention/ReLU-square
backbone and SentencePiece-1024 FineWeb data; it is not a reproduction of the
paper's 1B GQA/SiLU architecture or Phi-4 data mixture. It adds:

- Bias-free `W_u h_previous * sigmoid(W_g normalized_embedding)` feedback,
  RMS-normalized before the stack, with no additive token shortcut.
- Parallel, right-shifted Jacobi passes with full cross-pass gradients.
  Equation (12) uses `loss_1 + mean(loss_2, ..., loss_K)`, not an unweighted sum.
- Independent random plain-prefix lengths per row and feedback pass, uniform
  carried-state jitter of ±0.02 in training, and tied embedding/readout weights.
- Residual branch scaling `1/sqrt(2L)`. This constant is a local choice:
  the paper calls for depth scaling without specifying its exact value.
- A delayed pass schedule with exactly 1,500/440/60 one-/two-/three-pass
  batches at 2,000 steps (75%/22%/3%, or 1.28× token-equivalent pass compute).
  The paper does not specify stage boundaries. Locally, the first 50% is
  single-pass; 50–88% mixes one/two passes equally; the last 12% mixes
  one/two/three passes 50/25/25. Seeded stage shuffles agree across ranks.
- NorMuon matrices plus fused AdamW for the tied embedding and scalars;
  200-step warmup, final-25% cooldown, cooldown-only z-loss `1e-5`, and
  cooldown weight-decay scaling. One-pass steps do not update the fusion
  optimizer or decay its unused weights.

Training and model evaluation require compiled CUDA/BF16; parameters remain
FP32. The default global batch is 524,288 tokens, context 1,024, and validation
cadence 20 steps. `WARMDOWN_ITERS` is ignored in favor of the explicit WSD
schedule. Run the read-only configuration/data check without touching CUDA:

```bash
python3 scripts/train_nanogpt_mini_full_bandwidth.py --preflight
python3 scripts/train_nanogpt_mini_full_bandwidth.py --help
```

Queue the 2,000-step experiment through the existing metrics/TensorBoard runner:

The recorded runs use `/usr/bin/python3` (PyTorch 2.14.0, CUDA 13.3).
Use an absolute interpreter path when submitting: this workstation's repo
virtual environment has a different PyTorch/CUDA build. Do not pool
benchmark or common-panel results across those runtimes.

```bash
mlq submit --name nanomini_full_bandwidth_2k --cwd "$PWD" \
  --max-parallel-runs 1 --priority 1 --max-attempts 1 --time-limit 4h -- \
  /usr/bin/python3 scripts/ablation.py \
    --script scripts/train_nanogpt_mini_full_bandwidth.py \
    --name nanomini_full_bandwidth_2k --steps 2000 --val-every 20
```

For a matched-token/step control, use a distinct run name and append
`--env FBT_SCHEDULE=single`. This is not a matched-compute control.
For the detached-carry ablation, append `--env FBT_DETACH_CARRY=1` instead
(direct trainer flag: `--detach-carry 1`). The default remains attached.
Only the incoming hidden state is detached, before jitter and the trainable
value projection. The fusion weights, embeddings/readout, every pass's CE,
and within-pass causal attention remain differentiable. This mirrors the
stop-gradient boundary in MiniCPM5's token-carry combiner, not its additive
fusion architecture or stored-rollout RL replay. Forward values, the pass
schedule, and Eq. (12) objective weighting do not change.

The matched 1,000-step pair is `nanomini_full_bandwidth_1k` (attached) and
`nanomini_full_bandwidth_detach_1k` (detached): seed 1337, 524,288 tokens/step,
and exactly 750/220/30 one-/two-/three-pass steps. `detach_carry` is saved in
the model configuration; the trainer also records it in run provenance.

`metrics.jsonl`, `result.json`, `train.log`, and the self-describing
`final_model.pt` belong under `ablation_results/<run_name>/`; the runner feeds
pretraining TensorBoard at `tb_logs/<run_name>/`. The direct trainer writes
the log/checkpoint; use the runner for canonical metrics and summaries.

Headline `val_loss`/`val_bpb` always use the ordinary first pass. Explicit
`val_bpb_p1/p2/p3` are reported every validation, and `val_bpb_p8` at the end;
`feedback_trained` distinguishes measurements before the channel is trained.
Optional true cached recurrence scoring is enabled with
`FBT_SEQ_TOKENS=4096 FBT_SEQ_LEN=128` (final-only unless `FBT_SEQ_EVERY` is set).
Its `val_bpb_seq` and matched-window `val_bpb_p1_seq` are separate from the
full-window parallel metrics.

Verification: five schedule tests and eight compiled CUDA/BF16 behavioral tests
passed (GPU queue job 8313). GPU checks cover prefix preservation, causality,
Eq. (12) under both gradient policies, cached recurrence/Jacobi agreement,
no-shortcut fusion, tied readout, and stochastic training under both modes.
An explicit gradient-boundary check verifies identical attached/detached
forward losses, no detached producer gradient, and retained reader gradients.
Both real-data preflights pass. No 2,000-step training result or BPB improvement
has been established for this version; do not promote it as a winning ablation.

### LayerScale carry fusion

`FBT_FUSION=layerscale FBT_DETACH_CARRY=1 FBT_LAYERSCALE_INIT=0.1`
selects the separate residual-fusion ablation:

```text
e_t = existing normalized token embedding
u_t = e_t + gamma * (W_token e_t + W_carry stop_gradient(h_previous))
```

`gamma` is a learned per-channel FP32 parameter, initialized to 0.1. Its
product is formed in FP32 before conversion to the residual input dtype.
There is no sigmoid or normalization after the residual addition. Plain
prefixes remain unchanged; at gamma zero the input is exactly `e_t`.
This applies [LayerScale (2103.17239v2)](https://arxiv.org/pdf/2103.17239v2)
to a token-conditioned carry residual; it is not the paper's original GLU.
The 0.1 initialization follows LayerScale's shallow-network setting, not the
MiniCPM combiner's 0.01/zero-projection initialization.

The Jacobi schedule, loss weighting, backbone, data, and optimizer are
unchanged. The channel scale uses the existing scalar AdamW group and is
inactive on plain one-pass updates, as are the fusion matrices.

Matched single-seed 1,000-step results (750/220/30 pass schedule):

| Fusion / carry gradients | Pass 1 BPB | Pass 2 BPB | Pass 3 BPB | Pass 8 BPB |
| --- | ---: | ---: | ---: | ---: |
| GLU / attached | 1.449260 | 1.484559 | 1.499752 | 1.506381 |
| GLU / detached | 1.446833 | 1.499779 | 1.518222 | 1.526676 |
| LayerScale / detached | **1.435961** | **1.433345** | **1.433276** | **1.433274** |

`nanomini_full_bandwidth_layerscale_detach_1k` improved pass 1 by 0.010872
BPB against detached GLU; additional passes now slightly improve validation.
Recorded training time was 778.663 seconds versus 784.306 seconds.
`comparison.json` preserves the matched metrics. This is not a 2,000-step
qualification or multi-seed result.

### Document-parallel sequential recurrence

`scripts/train_nanogpt_mini_full_bandwidth_stream.py` runs one current token
per independent document lane. There are no Jacobi or producer passes.
Incoming carry and historical KV are detached; current-token Q/K/V and the
trunk remain differentiable. The ring receives current KV/hidden only after
local backward. Actual document BOS resets carry, attention context, and
RoPE position; CPU page and optimizer boundaries do not reset them.

The default attention capacity is 1,024, including the current token.
BF16 ring storage is `4 * layers * lanes * capacity * kv_heads * head_dim`
bytes. Default four-head MHA uses 6 GiB at 512 lanes or 12 GiB at 1,024 lanes.
With `--kv-heads 1` (`FBT_KV_HEADS=1`), four query heads share one KV head:
the ring is four times smaller (3 GiB at 1,024 lanes). This is an architecture
ablation, not a behavior-preserving optimization. Omitted/zero KV heads retain
MHA, including loading older model configurations. Validation owns a separate
ring. Kernels read shared history in place, without expanding it to query heads,
and write only the new ring slot.
Training accumulates exactly 524,288 scored tokens per optimizer update.
Data uses the existing BOS-indexed document stream with bounded 32-tick
prefetch pages and deterministic document order, not a corpus-sized shuffle.

```bash
python3 scripts/train_nanogpt_mini_full_bandwidth_stream.py --preflight
mlq submit --name stream_benchmark --cwd "$PWD" \
  --max-parallel-runs 1 --priority 1 --time-limit 40m -- \
  /usr/bin/python3 scripts/train_nanogpt_mini_full_bandwidth_stream.py \
    --benchmark --run-id stream_benchmark --document-batch 1024 \
    --cache-capacity 1024 --benchmark-warmup 32 --benchmark-ticks 256
```

`--benchmark` exercises full-sized synthetic history, forward, local
backward, and commit; it performs no optimizer updates and is not quality
evidence. Training checkpoints preserve model/configuration and token
counts, not optimizer/cache/data-cursor state; they are not resumable.

Validation is a fixed 512-lane, 1,048,576-token document-stream panel.
BPB is the primary metric, canonically named `val_bpb_document_stream`.
The runner and watcher display it under the standard `val/bpb` and
`time/val_bpb` TensorBoard charts, while retaining the scoped metric.
`result.json` records `final_val_bpb_document_stream` with promotion scope
`document_stream_bpb`; challenge `final_val_bpb` remains null.
`--eval-checkpoint PATH --run-id UNIQUE_NAME`
evaluates existing FullBandwidth checkpoints on the identical panel without
training. Interpreter-pinned common-panel results are 1.486955 BPB for attached
`nanomini_full_bandwidth_1k`, 1.507146 for detached
`nanomini_full_bandwidth_detach_1k`, and **1.413219** for detached
`nanomini_full_bandwidth_layerscale_detach_1k`. These are not their packed
validation scores. Sequential training additionally changes packing and KV gradient
policy, so it is not a pure batching-only ablation.

Initial GPU qualification passed 11 combined LayerScale/FullBandwidth
contracts (job 8331) and four sequential contracts (job 8333), covering
SDPA value/current-gradient parity, detached history, ring wrap, document
isolation, BOS resets, and repeated local backward/commit.

Full-cache throughput investigation on the RTX 5090, pinned to PyTorch
2.14.0/CUDA 13.3, at 1,024 document lanes (32 warmup and 1,024 timed ticks):

| Execution | Tokens/second | Decision |
| --- | ---: | --- |
| Original sequential runtime | 56,537 | Retained |
| Save attention logits for backward | 55,325 | Rejected |
| Foreach parameter-gradient accumulation | 56,947 | Only 0.73% nominal gain; not retained |
| Both changes | 55,462 | Rejected |
| Best tested saved-query-Jacobian variant | 38,359 | Rejected |

The initial profile attributed 89.3% of GPU time to attention forward/backward.
An unchanged full-cache scan reads approximately 24 GiB of historical KV per
1,024-token tick across both directions. The Jacobian experiment removed
backward KV rereads and passed current-gradient/SDPA checks, including
capacities 1 and 1,024, but its forward tensor-core work and register pressure
cost more than it saved. Smaller document batches did not solve the throughput
gap. No approximation, shorter context, or KV quantization was substituted.
The original parallel detached run averaged about 668,000 tokens/second;
these measurements do **not** establish sequential training near parity.

`ablation_results/nanomini_document_stream_profile_baseline/comparison.json`
records pinned results, source snapshots, rejected candidates, and the
matched document panel. Earlier mixed-PyTorch performance measurements are
explicitly superseded.

### Shared-KV sequential ablation

Both FullBandwidth trainers accept `--kv-heads 1` / `FBT_KV_HEADS=1`.
The model retains four query heads, width 512, head dimension 128, six layers,
and the original MLP. Only K/V projections and storage shrink. Current-token
K/V gradients sum across their query group; historical K/V remain detached in
streaming training. The carry still contains all 512 channels.

The grouped Triton implementation processes a query group against its shared
history. Forward saves the current-token softmax coefficient for backward;
this preserves exact BOS value-gradient summation rather than recomputing
the coefficient through a rounded log-sum-exp.

Qualification passed all 22 compiled CUDA/BF16 tests (job 8393), including
MHA and grouped recurrence, SDPA value/current-gradient parity at full context,
query-group isolation, detached history, ring wrap, BOS resets, and complete
carry commits. Existing LayerScale and core FullBandwidth contracts also pass.

Full-cache benchmark, PyTorch 2.14.0/CUDA 13.3, 32 warmup/1,024 timed ticks:

| Shared-KV document lanes | Tokens/second | Peak allocated GiB |
| ---: | ---: | ---: |
| 1,024 | 173,077 | 3.14 |
| 2,048 | 191,096 | 6.14 |
| 4,096 | **198,759** | 12.15 |

The selected 4,096-lane configuration is 3.52× faster than the original MHA
sequential benchmark, but still about 3.36× slower than the earlier parallel
training average. Benchmarks exclude optimizer updates and the validation ring;
they do not establish training quality or end-to-end throughput parity.
Full results are in
`ablation_results/nanomini_shared_kv_bench_b4096/comparison.json`.

The shared-KV parallel control `nanomini_full_bandwidth_shared_kv_detach_1k`
(job 8403) completed 1,000 steps: first-pass packed BPB **1.459101**, versus
**1.446833** for detached MHA, a **0.012268 regression**. Training time was
739.810 seconds versus 784.306 seconds. Its common document-panel BPB is
**1.523839** (job 8405), versus **1.507146** for detached MHA.

The sequential run `nanomini_full_bandwidth_shared_kv_stream_1k` (job 8404)
was canceled at the user's request after 460 steps because runtime remained
unacceptable. Latest recorded document-panel BPB: **1.682126**; training time:
887.026 seconds; latest interval: **1.964 seconds/update**, 266,883 tokens/s.
The `last_model.pt` checkpoint and all 24 validation points remain available;
BPB was backfilled to the standard TensorBoard charts. This is an incomplete
run, not a matched 1,000-step quality result or a resumable checkpoint.
Both runs used detached GLU carry, seed 1337, and 524,288 tokens per update.
No further quality training is authorized. Shared-KV remains experimental;
no quality improvement is established.

### Shared-KV runtime investigation after stopping training

A fresh 4,096-lane full-cache benchmark measured **204,990 tokens/second**.
Attention forward/backward accounted for **78.65%** of GPU kernel time.
Tile/warp tuning stayed near 204,000 tokens/second; a SIMT variant passed
the stream contracts but fell to 153,397 tokens/second and was rejected.

Detached history permits batching backward across consecutive tokens while
keeping forward token-sequential. A temporal-backward primitive passed nine
gradient cases and reduced its measured component from 21.656 ms to 2.887 ms;
this is not a whole-model speedup.

The full deferred prototype measured **294,215 tokens/second**, **43.5% above**
the contemporaneous baseline, with 21.15 GiB peak reserved memory. This used
the full 1,024-token cache, 4,096 lanes, BF16, noise 0.02, 32 warmup ticks and
1,024 timed ticks. It excludes optimizer updates and the validation ring;
it does not establish end-to-end training throughput, BPB, or near-parallel speed.

**Not promoted.** The original tile-four qualification passed 3,936 checks,
but a wider-tile run exceeded its carry-error bound. Its fixtures and compiler
policy were subsequently aligned with production without relaxing tolerances.
Both revised qualifications then failed with an illegal CUDA memory access
after the first cold/hot replay pair; the failing runtime/kernel has not been
localized. An initial full-shape cold capture also failed before a warmed
repeat produced the benchmark above. Static reviews found no concrete cause,
but do not establish memory safety. Remaining sanitizer and frozen-checkpoint
evaluation jobs were canceled through `mlq`; dependent cold-capture and
RNG/optimizer-replay jobs did not execute.

Production retains the original immediate-backward runtime. Temporary candidates
and probes were moved out of `scripts/` into
`ablation_results/nanomini_shared_kv_profile_b4096/source/`.
`source_manifest.json` records original paths and checksums;
`optimization.json` records measurements, failures, and job outcomes.
The stopped step-460 checkpoint, metrics, and standard TensorBoard BPB charts
remain intact. No further training was started.

## FFN-only streaming recurrence with a future-bag carry

`scripts/train_future_credit_stream.py` trains
`pretraining/future_credit_stream/` on the non-GPT2 FineWeb
SentencePiece-1024 corpus. Six width-512 residual FFNs use the nanoGPT-mini
ReLU-square MLP convention, without attention, token-history buffers, or memory
slots. Each document carries one detached BF16 vector, injected before the
first FFN; FP32 master parameters train with fused AdamW. Every token is its
own local graph: no temporal gradient, no incoming-state VJP, no lookahead
graph.

The question this trainer asks is how a recurrent carry can be *chosen*
without gradients from the future, without future tokens as inputs, without
a critic, and without running the model again. Plain CE recursion
(`objective=ce`) carries the current hidden verbatim: the vector whose readout
through the head is the belief about `x_{t+1}`. The next reader will know
`x_{t+1}`; what it needs from the carry is what the past says about the tokens
*after* it. The `future_bag` objective adds a gated carry writer and asks the
carry to be the vector whose readout through the *same frozen head* is the
discounted distribution of the tokens from `x_{t+1}` on:

```text
h_t     = FFN stack(norm(embed x_t) + norm(c_{t-1}))
c_t     = g_t * unit(write(h_t)) + (1 - g_t) * unit(c_{t-1})    (writer; h_t detached; unit = RMS norm)
q_t     ~ sum_{j<=horizon} discount^j [document continues] onehot(x_{t+1+j})
L_t     = cross_entropy( softcap(head(final_norm(c_t))), q_t )  (head, norm frozen)
loss    = CE_t + L_t
```

`q_t` is the discounted return of future one-hots computed exactly from the
corpus: a horizon without a bootstrap, a learned value, or a vector residual
to regress. Future tokens are targets only, exactly as `x_{t+1}` is for CE.
The bag starts at the current target so that `discount=0` is exactly the
hidden's own job (the belief about `x_{t+1}`, what CE recursion carries) and
the identity-initialized writer starts near its optimum; larger discounts
lengthen the horizon and credit retaining context that pays off many tokens
later. The bag is truncated at the document's closing BOS, which is itself
its last member, so no row is empty. The writer normalizes both mixed terms,
so retention can only be expressed through the gate and never through the
relative norms of the write and the old carry; only a direction reaches the
reader anyway. The extra cost per token is two head matmuls (one
gradient-free, for the hidden's reference bag loss), a scatter into
`[batch, vocab]`, two softmaxes, and the writer's three `[dim, dim]` matmuls:
expect 10 to 15% of a step, no extra forward pass. The writer's 787,456
parameters (three `512 x 512` matrices plus biases, about 5.8% of the
backbone) run at inference too, so the arm is not parameter-matched to the
`ce` control; results report parameter counts and the step-time ratio.

Gradient ownership per token, one combined backward:

- current CE trains the backbone;
- `L_t` trains the writer through the frozen head's input gradient, and
  reaches the final norm gains, head weights, and the hidden only through
  `backbone_future_weight` (default 0: the backbone is exactly the
  CE-recursion backbone and the writer cannot damage the current prediction).

The writer starts as CE recursion (identity write, gate logit 3) so an
untrained writer reproduces the control. Only the carry's direction reaches
the reader (it is RMS-normalized) and only its direction is judged, so no
loss can be lowered by rescaling. Reset lanes read a zero carry and the writer
sees that reset state.

Objectives:

| `FUTURE_CREDIT_STREAM_OBJECTIVE` | Model | Trains | Role |
| --- | --- | --- | --- |
| `ce` | backbone | CE | recurrent-CE control |
| `future_bag` | backbone + writer | CE, writer | candidate recipe; `discount`, `horizon`, `backbone_future_weight` |
| `tbptt` | backbone | CE with true temporal gradients inside each 8-tick page | upper reference; never promotable |

If TBPTT-8 does not beat CE recursion, no local carry objective can be
expected to; `discount=0` isolates whether the horizon, rather than the
writer, is what matters.

Training lines log `bag_loss` (soft cross-entropy of the carry's readout to
the bag), `hidden_bag_loss` (the same cross-entropy for the hidden itself,
what CE recursion would have carried, gradient-free: the matched reference
the writer must beat), `bag_entropy` (entropy of the bag target; not a floor,
since the bag is a handful of samples from a future the carry cannot see),
`gate_mean` (the fraction of the carry that is the fresh write; a true
retention measure because both mixed terms have unit RMS), and
`carry_cosine` (cosine between the written carry and the hidden; 1 for CE
recursion). Validation reports `val_bpb`, `val_bpb_reset_latent`, and
`val_memory_gain_bpb` as before; validation and generation apply the writer.

```bash
mlq submit --name ffn_bag_gpu_tests --cwd "$PWD" --max-parallel-runs 1 --max-attempts 1 --time-limit 30m -- \
  .venv/bin/python -m pytest -q pretraining/tests/test_future_credit_stream_model_gpu.py \
  pretraining/tests/test_future_credit_stream_objective_gpu.py pretraining/tests/test_future_credit_stream_runtime_gpu.py
for arm in ce future_bag tbptt; do
  mlq submit --name ffn_bag_${arm}_sp1024_2k --cwd "$PWD" \
    --max-parallel-runs 1 --max-attempts 1 --time-limit 30m \
    --env FUTURE_CREDIT_STREAM_RUN_ID=ffn_bag_${arm}_sp1024_2k \
    --env FUTURE_CREDIT_STREAM_OBJECTIVE=${arm} -- \
    .venv/bin/python scripts/train_future_credit_stream.py
done
.venv/bin/python scripts/compare_future_credit_stream.py --candidate ffn_bag_future_bag_sp1024_2k \
  --control ffn_bag_ce_sp1024_2k --reference ffn_bag_tbptt_sp1024_2k
```

Use fresh run identifiers for new experiments. Configuration overrides use
`FUTURE_CREDIT_STREAM_<FIELD>`; `backbone_future_weight` is accepted only with
`objective=future_bag`. Resume with `--resume ablation_results/<run>/checkpoint.pt`
and identical configuration, data, tokenizer, and source hashes. Generate
through `mlq` with `--generate ablation_results/<run>/checkpoint.pt --prompt
"Some text"`. The architecture tag `streaming_ffn_future_bag_carry_v1` rejects
historical vector-credit, scalar-TD, and NextLat checkpoints. Resume source
fingerprints include the baseline utility that defines validation byte
accounting; that upstream source is not modified.

The mmap loader maintains 4,096 independent document lanes, preserves
cross-shard documents, prefetches one bounded page (with the `horizon` future
targets per position when the writer is trained), and checkpoints the consumed
rather than speculative cursor. A complete document ends at the following real
BOS target; incomplete corpus edges are excluded. The default 2,000 updates
consume 65,536,000 token targets, with validation every 20 updates.

### Dropped before running: hindsight-advantage critic

A critic-based design was built and queued but dropped before any arm ran:
a scalar critic of the carry's advantage over the zero carry, fitted by a
Bellman regression and differentiated into the writer. It needed a second
gradient-free zero-carry forward per token and a learned value that every
earlier critic in this lineage failed to fit; the future-bag objective above
replaces it with a signal the actor's own head provides. An imagined-rollout
variant (running the reader on the actor's belief for k future steps) was
rejected for cost: k extra reader passes per token exceed TBPTT itself.

### Historical scalar TD arm (retired objective)

`ffn_td_value_sp1024_2k` regressed a 64-unit critic onto the undiscounted sum
of future CE and differentiated it into the current hidden with the backbone
learning rate. It completed 2,000 updates at **2.725905 proxy BPB** against
**2.139952** for its matched CE control `ffn_td_value_ce_sp1024_2k`, and its
validation memory gain fell from 1.165 to 0.884 BPB. The logged critic never
fit: predictions stayed near 0.2 while targets were near 4.7, so the producer
followed an unfit critic's gradient. Its update time was 1.063x CE. The
objective was removed; its artifacts and `comparison.json` remain immutable.

### Historical vector-credit ablation

The matched CE control, `ffn_future_credit_ce_sp1024_2k`, completed 2,000
updates and 65,536,000 targets: **2.139481 proxy BPB**, 3.309343 reset-state BPB,
and 50.107 seconds of timed training. `ffn_future_credit_sp1024_2k`, which
predicted a synthetic credit vector, was externally cancelled by request; its
last validation was **2.485239 proxy BPB at step 660** against 2.3968 for CE at
the same step. There is no completed 2,000-update vector-credit score, so that
experiment is not eligible for a final matched comparison. Its wall-clock timing
was contaminated by concurrent, untracked GPU training.

### Historical 2,000-update references

The previous streaming NextLat implementation used SmoothL1 hidden matching and
forward-KL self-distillation, conditioned on the correct next token. Those
auxiliary losses and the old entrypoint have been replaced; historical
checkpoints, source hashes, metrics, and comparison artifacts remain unchanged.

| Recipe | Parameters | Proxy BPB | Reset-state BPB | Training-loop seconds |
| --- | ---: | ---: | ---: | ---: |
| Recurrent CE, actual-hidden carry | 13,652,480 | **2.139849** | 3.306123 | 46.246 |
| Delayed NextLat, predicted carry | 16,274,944 | 2.597596 | 3.067103 | 60.827 |
| Delayed NextLat, actual-hidden carry | 16,274,944 | 2.396413 | 3.018147 | 62.462 |
| Scalar TD value into hidden | 13,652,480 + critic | 2.725905 | 3.610297 | 50.179 |
| Recurrent CE control for the future-bag arms | 13,652,480 | 2.139688 | 3.306414 | 46.143 |
| Future-bag carry, discount 0 (writer judged on x_{t+1} only) | 14,439,936 | 2.141280 | 3.278168 | 56.803 |
| Future-bag carry, discount 0.5 | 14,439,936 | 2.146800 | 3.301365 | 57.621 |
| Future-bag carry, discount 0.9 | 14,439,936 | 2.168735 | 3.280604 | 56.928 |
| Future-bag carry, discount 0.9, bag gradient into backbone | 14,439,936 | 2.183606 | 3.276703 | 58.948 |
| TBPTT-8 reference (true temporal gradients per 8-tick page) | 13,652,480 | 2.175066 | 3.440089 | 53.780 |
| Stationary buffer, value_map entry, K=1 (read of the previous hidden) | 14,045,956 | 2.203452 | 2.975146 | 50.833 |
| Stationary buffer, value_map entry, K=10 | 14,045,956 | 2.228332 | 2.934474 | 56.044 |
| Stationary buffer, latent_norm entry, K=1 (bitwise the control's forward) | 13,783,812 | 2.138618 | 3.289755 | 48.321 |
| Stationary buffer, latent_norm entry, K=10, recency slope 1 | 13,783,848 | 2.151391 | 3.272516 | 50.599 |
| Stationary buffer, latent_norm entry, K=10, recency slope 4 | 13,783,848 | 2.139679 | 3.303107 | 48.118 |

Predicted-carry NextLat was 0.457747 BPB worse than CE. Switching NextLat to
actual carry improved BPB by 0.201183, but remained 0.256564 worse than CE.
Every recipe that pushed a future-proxy gradient into the CE backbone lost to
plain CE recursion. These are negative ablations, not promoted recipes.
The future-bag arms (2026-09-17) lost monotonically in the discount while
beating the hidden at their own head-readout judge; see the results under the
pre-registration entry in `NOTES.md`. The TBPTT-8 reference, the upper bound
for temporal credit in this architecture, also lost to detached CE at 2,000
updates, so the carry-objective line is closed at this scale.

## FFN-only streaming recurrence over a stationary buffer

`scripts/train_stationary_stream.py` trains `pretraining/stationary_stream/`,
the same six-FFN width-512 backbone, data pipeline, optimizer, schedule, and
validation panel as `future_credit_stream`, with one change in what a step
reads. CE recursion carried one detached vector that every step rewrote. Here
each document lane keeps the last `buffer_slots` post-final-norm hiddens
verbatim and detached (Transformer-XL's stop-gradient memory at a segment
length of one token) in a ring that the host overwrites oldest-first, and
every step reads them with one small attention keyed by slot age `a = 1..K`:

```text
q      = W_q norm(embed x_t)
k_a    = RoPE(age a)(W_k h_{t-a})                 rotation gathered from a precomputed age table; query unrotated
w      = softmax_a([b_null, q . k_a / sqrt(d_k)])   slots before the document's BOS are masked
read   = W_v(concat_h sum_a w_{h,a} slice_h(h_{t-a}))   value projection after mixing (exact)
h_t    = FFN stack(norm(embed x_t) + read)
```

No gradient crosses a tick and nothing judges the buffer: the read block trains
through the current tick's CE alone. Two read entries (`read_entry`) carry the
mixed slots into the residual stream. `value_map`, above, keeps a learned null
slot and a zero-initialized `W_v`, so an untrained model is the context-free
FFN. `latent_norm` (the default) has neither: the softmax runs over the
readable slots only, the mixed vector passes through the control's own
`latent_norm` and enters at unit RMS from step 0, the query starts at zero,
and a learned per-head recency bias `-slope * (age - 1)` (`read_recency_slope`,
default 1) makes the newest slot dominate at init; with `K=1` it is the control's forward bitwise (tested). The
backbone initialization is bitwise identical to
`StreamingFFNModel(use_writer=False)` under the same seed (tested), so the
CE-recursion arm of the sibling line is the matched control. Diagnostics per
training log: `null_mass`, `read_age`, `read_rms` (all over lanes with a
readable slot; `read_rms` excludes the value bias); validation reports
`val_bpb`, `val_bpb_reset_latent` (every slot masked), and
`val_memory_gain_bpb`.

The `value_map` arms lost to the control (K=1 2.203452, K=10 2.228332 against
2.139688) with the reader demonstrably in use (K=10 null mass 0.09, mean read
age 3.2 steps): both arms were *better* than the control with context removed
and recovered far less from it, the signature of a weak entry path (read RMS
0.23-0.29 against the control's unit-RMS carry). K=10 also ran at 1.20x the
control's step time; K=1 at 1.09x. With the `latent_norm` entry, K=1
reproduces the control (2.138618 against 2.139688) and K=10 with a recency
slope of 1 lands 0.012 worse: its read age stayed at the prior's own
expectation (1.53) for the whole run, so the blurred mixture cost early and
the query never sharpened it. With a recency slope of 4 (98% of the mass on
the newest slot at init) K=10 is the control to five decimals and its read
age moved from 1.019 to 1.033 over the run: the reader never used the older
slots. Step time at K=10 is 1.09x. The line is closed as a negative result;
`NOTES.md` has the reading.

```bash
mlq submit --name stat_gpu_tests --cwd "$PWD" --max-parallel-runs 1 --max-attempts 1 --time-limit 30m -- \
  .venv/bin/python -m pytest -q pretraining/tests/test_stationary_stream_model_gpu.py \
  pretraining/tests/test_stationary_stream_runtime_gpu.py
for slots in 1 10; do
  mlq submit --name stat_norm_k${slots}_sp1024_2k --cwd "$PWD" \
    --max-parallel-runs 1 --max-attempts 1 --time-limit 30m \
    --env STATIONARY_STREAM_RUN_ID=stat_norm_k${slots}_sp1024_2k \
    --env STATIONARY_STREAM_BUFFER_SLOTS=${slots} \
    --env STATIONARY_STREAM_READ_ENTRY=latent_norm -- \
    .venv/bin/python scripts/train_stationary_stream.py
done
mlq submit --name stat_norm_k10_s4_sp1024_2k --cwd "$PWD" --max-parallel-runs 1 --max-attempts 1 --time-limit 30m \
  --env STATIONARY_STREAM_RUN_ID=stat_norm_k10_s4_sp1024_2k --env STATIONARY_STREAM_BUFFER_SLOTS=10 \
  --env STATIONARY_STREAM_READ_ENTRY=latent_norm --env STATIONARY_STREAM_READ_RECENCY_SLOPE=4 -- \
  .venv/bin/python scripts/train_stationary_stream.py
.venv/bin/python scripts/compare_stationary_stream.py --candidate stat_norm_k10_sp1024_2k \
  --control ffn_bag_ce_sp1024_2k --reference stat_norm_k1_sp1024_2k
```

Configuration overrides use `STATIONARY_STREAM_<FIELD>` (`buffer_slots`,
`read_heads`, `read_key_dim`, `read_entry`, `read_recency_slope`, and the shared training
fields). Resume with
`--resume ablation_results/<run>/checkpoint.pt`; resume refuses changed
sources or a changed configuration. The pre-registration and results are in
`NOTES.md` under "Stationary buffer read for the streaming FFN".

## Slot-limited KV cache with a learned write policy (nanogpt-mini)

`pretraining/nanogpt_mini/nanogpt_mini_slot_train.py` forks the nanogpt-mini
baseline so that every layer attends over the token itself plus at most
`SLOT_M` (64) memory entries: after each position's top state, a
zero-initialised head samples a slot (or the null write) and that
position's per-layer key/value pair overwrites it. Replay uses the
`(T, M)` alive table (`alive[t, s] = max{i < t : σ_i = s}`, key `i`
visible iff `i == t` or `alive[t, σ_i] == i`) as a FlexAttention block
mask derived from the table. Training is two Jacobi passes (full-causal
reference pass whose head samples the slots, then the slot-restricted pass)
with a policy gradient on the slot choice whose advantage is the
discounted future CE gain of the restricted pass over the reference
(`SLOT_GAMMA` 0.9, `SLOT_HORIZON` 32, EMA baseline). `SLOT_MODE=fifo` is
the 64-token sliding window in slot form and `SLOT_MODE=full` is the base
model bitwise. Validation prints `val_bpb` (restricted), `val_bpb_full`,
`val_slot_null`, `val_slot_entropy`, `val_slot_age`, and at the last step
the true sequential regime (`val_bpb_seq`, `val_bpb_seq_greedy`). The
mechanism, loss, and pre-registration are in `NOTES.md` under
"Slot-limited KV cache with a learned write policy".

```bash
.venv/bin/python -m pytest -q pretraining/tests/test_nanogpt_mini_slot.py
mlq submit --name nanomini_slot_gpu_tests --cwd "$PWD" --max-parallel-runs 1 --max-attempts 1 \
  --time-limit 40m --priority 2 -- \
  .venv/bin/python -m pytest -q -s pretraining/tests/test_nanogpt_mini_slot_gpu.py
for mode in policy fifo full; do
  mlq submit --name nanomini_slot_${mode}_2k --cwd "$PWD" --max-parallel-runs 1 --max-attempts 1 \
    --time-limit 3h --priority 1 -- \
    .venv/bin/python scripts/ablation.py --script pretraining/nanogpt_mini/nanogpt_mini_slot_train.py \
      --name nanomini_slot_${mode}_2k --steps 2000 \
      --env DATA_PATH=data/datasets/fineweb10B_sp1024 VAL_TOKENS=1048576 SLOT_MODE=${mode}
done
```

## Model-native binary character channel (nanogpt-mini)

`pretraining/nanogpt_mini/nanogpt_mini_native_bits_model.py` implements:

```text
external UTF-8 text → shuffled opaque character IDs
 → lookup B[V,32] → hard learned character bits (straight-through gradients)
 → small causal compressor → 8 latent bits per character
 → Mini 6-layer/512-wide Transformer → 8 simultaneous bit logits

transmitted latent bits → local causal decoder → 32 character-code bits
 → small binary identity decoder → opaque-ID bit predictions
 → XOR identity residual → exact ID → external character lookup
```

There is no BPE, Unicode numerical feature, vocabulary output projection, or
vocabulary softmax. Attention still uses Mini's normal SDPA. Bits are a vector
channel, **not separate Transformer positions**. This version reduces channel
width, not sequence length: there is still one model position per character.
The compressor and local decoder each use a 15-character causal receptive field.
The prior sees only previous latent vectors; the decoder sees transmitted
current/past latents, never the original character.

The learned table need not be collision-free. Direct supervision reconstructs
opaque identity bits from each observed character's learned code, and a second
auxiliary reconstructs character codes from latents. These encourage useful
codes without a vocabulary-sized classifier. The XOR residual guarantees
recovery even if every learned code collapses to the same value.

The training objective is
`latent_BCE + identity_residual_BCE + CODEC_WEIGHT * reconstruction_auxiliaries`.
`val_bpb` counts **both** latent and residual nats divided by `ln(2)` and the
actual UTF-8 byte count. Auxiliary losses are excluded and logged separately.
This is a **two-part coding-cost bound**, not exact marginal character NLL.
Independent Bernoulli bits cannot generally represent multimodal symbol
uncertainty efficiently. In particular, redundant 256-bit random addresses
would make their summed BCE a poor proxy for character entropy.

`scripts/native_bits.py` provides `prepare`, `train`, `encode`, and `decode`.
Preparation strictly decodes UTF-8 without normalization, preserves every
scalar including NUL/newlines/combining marks, and shuffles the union of the
observed train/validation alphabet with a fixed seed. The alphabet is saved
with the dataset/checkpoint; it is **not all Unicode**, and encoding unseen
characters fails explicitly. A validation-only character has a row but no
training occurrences. Malformed UTF-8 and aliased train/validation files fail.

```bash
mlq submit --name native_bits_prepare --cwd "$PWD" --max-parallel-runs 1 \
  --priority 1 --time-limit 30m -- \
  .venv/bin/python scripts/native_bits.py prepare \
    --train-text /path/train.txt --val-text /path/val.txt \
    --output data/datasets/native_bits --alphabet-seed 1337

# Submit only after preparation and GPU contracts succeed.
mlq submit --name native_bits_frozen_2k --cwd "$PWD" --max-parallel-runs 1 \
  --priority 1 --time-limit 2h -- \
  .venv/bin/python scripts/native_bits.py train --data data/datasets/native_bits \
    --name native_bits_frozen_2k --steps 2000 --frozen-codes
mlq submit --name native_bits_learned_2k --cwd "$PWD" --max-parallel-runs 1 \
  --priority 1 --time-limit 2h -- \
  .venv/bin/python scripts/native_bits.py train --data data/datasets/native_bits \
    --name native_bits_learned_2k --steps 2000
```

The frozen control changes only whether the random character table learns;
both arms train the compressor, prior, and decoders. Defaults: 32 character
bits, 8 latent bits, 1,024 characters/window, 8,192 characters/update, one seed,
2,000 steps, validation every 20 steps on a fixed 65,536-character prefix.
Compare the **final step-2000** bound on identical data; require an improvement
greater than 0.005 BPB before promoting the learned table. This character-budget
experiment is not a matched-work comparison with the original sp1024 Mini.

Alternatively, `prepare --byte-cache
data/datasets/bd_mathglm_v6_bytes_2k.cache/00_fineweb` extracts the existing
authenticated length-prefixed uint16 byte cache, removing only EOT markers at
the external boundary and validating UTF-8 separately per document. Its
`validation_web.npy` split differs from the original sp1024 heldout split:
do not compare its numbers directly to `baseline_2k` or the challenge SOTA.

Metrics/results are owned by `scripts/ablation.py` under
`ablation_results/<name>/`; TensorBoard uses `tb_logs/<name>/` (pretraining).
`checkpoint.pt` contains model/config, alphabet/data/source provenance,
optimizer state, RNG state, and source cursor. `train --resume CHECKPOINT`
requires matching settings/data and a **new run name**, because the ablation
runner clears its destination directory. The runtime requires CUDA, BF16,
and compiled neural paths; it never falls back to CPU.

`encode --checkpoint CHECKPOINT --input SOURCE --output PACKET` and
`decode --checkpoint CHECKPOINT --input PACKET --output RESTORED` must also
run through `mlq`. NB01 packets bind the checkpoint and source with SHA-256
and **raw-bit-pack** the latent/residual streams; they are not entropy-coded.
The CLI reports actual stored size separately from modeled rate, verifies an
exact roundtrip before publishing a packet, and verifies the decoded source
digest before publishing text. Decoding requires the same checkpoint and a
numerically compatible backend; incompatible hard-decision rounding fails
the source digest rather than silently corrupting output.

Verification:

```bash
.venv/bin/python -m pytest -q pretraining/tests/test_native_bits_data.py \
  pretraining/tests/test_native_bits_wire.py
mlq submit --name native_bits_gpu_contracts --cwd "$PWD" --max-parallel-runs 1 \
  --priority 1 --time-limit 40m -- \
  .venv/bin/python -m pytest -q -s pretraining/tests/test_native_bits_gpu.py
```

The GPU contracts exercise the real Mini width with 172,808 opaque identities:
hard/ST gradients, causal encoding/prior/decoding, honest residual accounting,
exact recovery under total codebook collapse, and packet roundtrips.

### Initial 2,000-step result: rejected, latent collapse

The first paired runs used 3,099 observed characters from 208,307,257 training
characters. Both scored the same 65,536 heldout characters / 66,121 UTF-8 bytes:

| Character table | Two-part BPB | Latent BPB | Residual BPB |
| --- | ---: | ---: | ---: |
| Frozen random (`nanomini_native_bits_frozen_2k`) | 10.868301 | 0.00000175 | 10.868298 |
| Learned (`nanomini_native_bits_learned_2k`) | 10.868298 | 0.00000175 | 10.868295 |

The difference is below the 0.005 promotion threshold. Neither is a winning
model. The near-zero latent rate and residual-dominated cost expose the failed
representation; the overall latent bit mean of 0.5 did **not** establish useful
variation. New runs also log per-bit marginal entropy within each batch and
direct code-to-identity reconstruction accuracy to distinguish these failures.
The existing runner's proxy-metric convention is used so summaries do not
label this coding bound as challenge BPB.

Losslessness nevertheless held: the trained learned-code checkpoint encoded
and decoded all 3,099 alphabet characters in separate CUDA CLI processes,
recovering the original 8,828 UTF-8 bytes exactly (`native_bits_roundtrip/`).
That proves the residual mechanism, not language-model quality.

The full-width gradient diagnosis (`native_bits_diagnosis/gradient_paths.json`)
used 8,192 heldout characters. It found 226 latent vectors at initialization
and one vector, `10111000`, after training in both arms. Mean absolute encoder
logits exceeded 31; mean sigmoid surrogate derivatives fell below `5e-19`.
At the latent vector, the prior's target-gradient norm exceeded the residual
reconstruction gradient by about 7,000–39,000 times. Direct code-to-identity
accuracy was still 100% (frozen) / 96.9% (learned) on this batch.

The isolated correction stops gradients through **latent prediction targets**.
The encoder still receives reconstruction gradients and gradients through
latents used as the prior's context. The reported forward rate still includes
every latent/residual bit; only gradient routing changes. No entropy penalty,
new loss weight, bandwidth schedule, or decoder change is bundled with it.
The `nanomini_native_bits_{frozen,learned}_sg_2k` runs restart from seed 1337
rather than trying to revive saturated checkpoints. Their final results:

| Character table, detached prior targets | Two-part BPB | Latent BPB | Residual BPB | Identity accuracy |
| --- | ---: | ---: | ---: | ---: |
| Frozen (`nanomini_native_bits_frozen_sg_2k`) | 4.9824261 | 4.6525345 | 0.32989148 | 97.19% |
| Learned (`nanomini_native_bits_learned_sg_2k`) | 4.5065220 | 4.4451506 | 0.06137154 | 99.56% |

The learned table improves the matched frozen control by **0.4759041 BPB**,
exceeding the 0.005 gate. Mean per-bit entropies are 0.9677 and 0.9024,
respectively, rather than a constant bus. The corrected learned checkpoint
also passed a separate-process exact roundtrip of all 3,099 characters.
Eight compiled CUDA contracts and 52 CPU contracts pass, including a regression
that forbids current-target encoder credit while preserving prior-context credit.
This supports the correction against the observed collapse; it does **not**
establish that stop-gradient is an optimal learning rule or beat a matched
character-softmax baseline, which has not been run.

The user-requested extension is `nanomini_native_bits_learned_sg_8k`: 8,000
steps from seed 1337 with the current target-detachment rule, unchanged batch,
context, codec and validation settings, and a fresh 8k learning-rate schedule.
It does not resume the already-decayed 2k checkpoint.
The completed 8k run reaches **3.9294272 two-part BPB**: 3.9064688 latent
plus 0.0229579 residual, with 99.8001% identity reconstruction accuracy and
0.8807 mean per-bit entropy. The 8k experiment therefore remains noncollapsed;
nearly all remaining coding cost is in the latent prior. This is evidence for
longer training of this variant, not evidence that target detachment is optimal.

### Joint bit priors and bijective codec comparison

`bit_density.py` shares the causal Mini prior between the original native
codec and the new `nanogpt_mini_bitflow_model.py`. `--mixture-components 8`
predicts a mixture of eight product-Bernoulli distributions per character:

```text
P(z | past) = sum_m softmax(mixture_logits)[m] *
                     product_k Bernoulli(z[k]; bit_logits[m,k])
```

The likelihood marginalizes the component with stable `logsumexp`; it does
not choose an uncharged best component. Small deterministic component biases
break zero-head symmetry. `--model native --mixture-components 1` retains
the historical independent-bit control, including detached current targets.
Changing only the component count preserves the native codec's initialization
RNG stream. Old native checkpoints still load with one component.

`--model bitflow` instead uses:

```text
external UTF-8 → shuffled opaque IDs → minimum-width identity bits
 → alternating learned XOR couplings → binary vector → causal Mini joint prior
```

Each coupling leaves one half unchanged and XORs the other half with a mask
predicted only from the unchanged half. Reverse layer order gives the exact
inverse, even after learning. The mask has a hard binary forward and a selectable
straight-through derivative; both current-target and prior-context likelihood
gradients reach the codec. There is no stop-gradient target, reconstruction
decoder, auxiliary loss, balance penalty, or residual stream.
`--mask-surrogate sigmoid` (default) differentiates the Bernoulli-mask
probability relaxation; `--mask-surrogate identity` restores the original
logit surrogate. The choice is stored in model configuration and checked on
resume. Both are biased discrete-gradient estimators; invertibility guarantees
information preservation, not successful optimization.

The 3,099-character dataset needs 12 bits; all 4,096 patterns remain in the
probability domain. Reserved-ID mass is **charged**, not renormalized away,
and decoding a reserved ID fails explicitly. This is a code-space coding
bound, not a normalized likelihood over only the observed alphabet.
`--fixed-codec` freezes the identity-initialized coupling networks; the learned
arm starts with exactly the same codec and prior weights.

`--model softmax` is the matched character baseline in
`nanogpt_mini_character_model.py`. It uses the original Mini trunk and a
vocabulary-sized softmax, solely as a comparison against the proposed
binary-output models. All variants predict every source character, including
the first from zero-feature BOS. Context length, character budget, heldout
prefix, seed and learning-rate schedule match. The softmax embedding retains
the original Mini learning rate **0.7**, rather than the native code-logit
table's **0.004**; parameter counts and compute are not artificially matched.

All rates are divided by actual source UTF-8 bytes. `val_proxy_bpb` prevents
the ablation runner from labeling this prepared split as challenge heldout;
the additional labels are `val_two_part_bpb`, `val_code_space_bpb`, and
`val_character_bpb`. NB01 now supports zero-width residual rows for BitFlow
and softmax identity transport; nonzero-width historical packets are unchanged.
Packets still contain **raw packed bits**, not an entropy-coded realization
of the reported likelihood.

```bash
# Common: prepared dataset, 2,000 updates, validation every 20 updates.
mlq submit --name native_joint8_2k --cwd "$PWD" --max-parallel-runs 1 \
  --priority 1 --time-limit 2h -- \
  .venv/bin/python scripts/native_bits.py train \
    --data data/datasets/native_bits_fineweb --name native_joint8_2k \
    --model native --mixture-components 8
mlq submit --name bitflow_fixed_2k --cwd "$PWD" --max-parallel-runs 1 \
  --priority 1 --time-limit 2h -- \
  .venv/bin/python scripts/native_bits.py train \
    --data data/datasets/native_bits_fineweb --name bitflow_fixed_2k \
    --model bitflow --fixed-codec
mlq submit --name bitflow_learned_2k --cwd "$PWD" --max-parallel-runs 1 \
  --priority 1 --time-limit 2h -- \
  .venv/bin/python scripts/native_bits.py train \
    --data data/datasets/native_bits_fineweb --name bitflow_learned_2k \
    --model bitflow
mlq submit --name character_softmax_2k --cwd "$PWD" --max-parallel-runs 1 \
  --priority 1 --time-limit 2h -- \
  .venv/bin/python scripts/native_bits.py train \
    --data data/datasets/native_bits_fineweb --name character_softmax_2k \
    --model softmax
```

The integrated verification passes 72 CPU contracts and 31 compiled CUDA
contracts: full-domain probability normalization, paid mixture choice,
nontrivial bijection including odd-width splits, signed XOR probability-surrogate
derivatives, full target/context encoder credit, causal prediction, reserved
IDs, first/tail scoring, historical checkpoint metadata, and residual-free
packet framing. GPU tests run through `mlq`; no CPU neural fallback is used.

The first learned-flow arm (`nanomini_bitflow_learned_joint8_2k`) used an
identity straight-through derivative. It stopped shortly after step 170
because the gradient guard checked FP32 norms rather than gradient elements.
Replaying its step-160 checkpoint reproduced that guard at update 178: all
gradient elements remained finite, but the largest codec gradient was
`1.53e18`, its FP64 norm was `1.85e19`, and FP32 norm accumulation overflowed.
Coupling mask logits reached 5,504 while forward codes remained exactly binary.
The diagnosis is saved in that run's `failure_diagnosis.json`.
**This was a false-positive gradient-finiteness check, not evidence that the
original learning rule had failed.** The sigmoid-surrogate run
(`nanomini_bitflow_learned_sigmoid_joint8_2k`) is a separate experiment, not a
continuation or a required repair of the identity rule.

The guard now checks `isfinite` on every gradient element and reduces booleans
in a compiled function. It performs no norm calculation, clipping or mutation.
CUDA regressions accept finite `1e20` gradients and reject NaN and both signs
of infinity. The original step-160 checkpoint remains untouched; an annotated
copy in `ablation_results/nanomini_bitflow_identity_resume/` records its
identity surrogate, original SHA-256 and historical source hashes. Its model,
optimizer, RNG state and data cursor are preserved for the resumed experiment.
The original checkpoint also recovered all 3,099 characters / 8,828 UTF-8 bytes
with 3,099 distinct codes and zero residual bits.

With the guard corrected, `nanomini_bitflow_learned_identity_guardfix_2k`
continued improving to **4.5181253 BPB at step 220**. A second diagnostic
isolated a different overflow at update 222: every gradient and parameter was
finite before fused AdamW, but gradients near `1.85e19` produced NaNs in FP32
second moments and then five codec weights. This transition is recorded in
that run's `optimizer_diagnosis.json`.

For the identity surrogate, the codec's 3,352 master parameters and Adam
states now use FP64; all codec neural operations still cast to BF16, and the
Transformer is unchanged. Initialization happens before widening, preserving
the original initialization stream. There is no clipping, changed loss,
learning-rate adjustment, or replacement surrogate. A compiled CUDA regression
checks an actual Adam update with finite `1e20` gradients against its expected
update, in addition to checking state finiteness.

`nanomini_bitflow_learned_identity_fp64state_2k` resumes the valid step-220
checkpoint with its momentum, RNG, source cursor and original 2k schedule.
It **completed step 2000 at 2.7975670 BPB**. The model and every optimizer
state tensor are finite, and its separate-process alphabet roundtrip retained
3,099 distinct codes with zero residual bits (`verification.json` in that run).
This is 0.0413840 BPB worse than fixed codes and 0.1004021 worse than the
sigmoid surrogate. The original rule is now evaluated rather than rejected
because of an overflowing diagnostic; its completed endpoint does not clear
the promotion gate. Surrogate choice is explicit in new checkpoints; resuming
historical BitFlow checkpoints without that field is rejected rather than
silently guessing a different backward rule.

Verification for this repair: 33 compiled CUDA contracts covering both
surrogates, finite-element checks and the high-dynamic-range optimizer update.

### Completed 2,000-step comparison

All eleven completed arms use seed 1337, six 512-wide main Mini blocks, 8,192
unique characters/update, 1,024-character windows, and the same
65,536-character / 66,121-byte heldout prefix. Adaptive refresh additionally
has two 128-wide local blocks and evaluates two sampled routes per training
example; it is not a compute-matched run. Dynamics uses one backbone pass
plus cheap latent auxiliaries; its fresh fixed-prefix control is included.
The canonical cross-run summary is
`ablation_results/nanomini_binary_comparison/summary.json`; individual runs
retain `metrics.jsonl`, `result.json`, and their step-2000 checkpoints.

| Model | Heldout BPB | Parameters | Run |
| --- | ---: | ---: | --- |
| Native codec, independent prior | 4.4751207 | 19,099,484 | `nanomini_native_bits_independent_refactor_2k` |
| Native codec, joint 8-component prior | 2.5418896 | 19,132,316 | `nanomini_native_bits_joint8_2k` |
| Fixed bijection, joint 8-component prior | 2.7561830 | 18,972,544 | `nanomini_bitflow_fixed_joint8_2k` |
| Learned bijection, sigmoid surrogate, joint prior | 2.6971649 | 18,972,544 | `nanomini_bitflow_learned_sigmoid_joint8_2k` |
| Learned bijection, identity surrogate, FP64 codec state | 2.7975670 | 18,972,544 | `nanomini_bitflow_learned_identity_fp64state_2k` |
| Fixed bijection, fresh joint-8 control for prefix ablation | 2.7658979 | 18,972,544 | `nanomini_bitflow_fixed_joint8_prefix_control_2k` |
| Fixed bijection, shared prefix-conditioned head | 2.0747585 | 18,987,929 | `nanomini_bitflow_fixed_prefix128_2k` |
| Learned variable-length refresh, local character decoder | 2.2351199 | 19,508,738 | `nanomini_adaptive_refresh002_2k` |
| Fixed prefix head, fresh dynamics control | 2.0641140 | 18,987,929 | `nanomini_prefix128_dynamics_control_2k` |
| Normalized latent dynamics, learned refresh | 2.0659811 | 19,184,270 | `nanomini_dynamics_norm_h2_refresh002_2k` |
| Character softmax | 1.8111367 | 22,085,659 | `nanomini_character_softmax_2k` |

The joint prior improves the fresh independent control by **1.9332311 BPB**.
Its rate includes 2.4518676 latent BPB and 0.0900222 residual BPB; the residual
has not been hidden or dropped. The sigmoid-surrogate learned bijection improves its identical
fixed-codec control by **0.0590180 BPB**, above the 0.005 promotion threshold.
It retains 0.88827 mean per-bit entropy and requires no reconstruction
auxiliary, stop-gradient target, or residual. Both improvements are supported
by these single-seed endpoint comparisons; retain them as experimental wins.

The prefix-conditioned fixed-code model is now the best binary arm, but
character softmax is still **0.2529773 BPB better** than its fresh dynamics
control. This does not establish a
competitive replacement for softmax: the baseline also has more parameters
and a distinct input embedding and embedding optimizer convention. No new
8k or challenge-scale run was launched. None of these scores is challenge
BPB, and these unquantized checkpoints are not 16 MB submission artifacts.
The fresh prefix control is a paired rerun, not a new architecture or evidence
of a new algorithmic gain. Historical experiments retain their own paired controls.

The historical independent control was 4.5065220 BPB. Its initial residual
rate and latent bit mean match the new control, but the refactored run does
not reproduce its final trajectory bit-for-bit. The joint-head improvement
above therefore uses the **fresh** 4.4751207 control, not the historical score.

The original six completed checkpoints passed separate-process CLI encode/decode over
all 3,099 alphabet characters / 8,828 UTF-8 bytes. Both fixed and learned
bijections transmitted 3,099 distinct 12-bit codes with zero residual bits.
Transport evidence is in `ablation_results/nanomini_binary_comparison_roundtrip/`,
including the sigmoid arm's separate result JSON; the resumed identity arm
stores `verification.json` and its packet/restored text in its own run directory.
This demonstrates losslessness, not entropy compression; NB01 still
raw-bit-packs the streams.

### Prefix-conditioned address prediction

`--model bitflow --fixed-codec --density-head prefix --prefix-width 128`
replaces only the output density. It retains the shuffled opaque IDs, fixed
12-bit identity addresses, linear binary input projection, six 512-wide Mini
blocks, optimizer conventions, data order, and 2,000-step schedule. The
same-seed control has identical initial codec, input, and Transformer weights.
No learned address assignment, nonlinear input encoder, or character patching
is introduced by this ablation.

The head applies the chain rule rather than a mixture of independent bits:

```text
P(address | past) = product_j P(bit[j] | past, bit[:j])
```

`PrefixBinaryHead` projects the Transformer state once into 128 dimensions.
An exclusive cumulative sum of learned signed-bit contributions represents
each strictly earlier prefix. Adding a learned bit-position vector, applying
ReLU squared, and using one shared scalar readout produces each conditional
logit. The current and future bits cannot enter their own conditioning path.
All decisions are teacher-forced in parallel during training; sequential
prefix traversal gives the same likelihood without another main-Transformer
pass per bit. Neural operations remain compiled BF16, with FP32 likelihood
reductions and the existing softcap of 15.

The head has **68,737 parameters**, versus **53,352** for the mixture control:
15,385 additional parameters, not a vocabulary-sized output table. All 4,096
addresses remain in the probability domain; the 997 reserved addresses are
still charged. This implementation intentionally requires a fixed codec and
`mixture_components=1`. The CLI chooses that component default for prefix
heads and rejects unsupported combinations before launching training.
Density-head choice and width are checkpointed and checked on resume;
historical checkpoints without these fields retain the mixture density.

```bash
mlq submit --name nanomini_bitflow_fixed_prefix128_2k --cwd "$PWD" \
  --max-parallel-runs 1 --priority 1 --time-limit 30m -- \
  .venv/bin/python scripts/native_bits.py train \
    --data data/datasets/native_bits_fineweb \
    --name nanomini_bitflow_fixed_prefix128_2k \
    --model bitflow --fixed-codec --density-head prefix --prefix-width 128 \
    --steps 2000 --val-every 20
```

The fresh paired mixture control scored **2.7658979 BPB**, versus
**2.0747585 BPB** for the prefix head: a **0.6911394 BPB improvement**,
comfortably exceeding the 0.005 retention gate. Keep the prefix head as an
experimental win, not as a softmax replacement. It also improves on the
previous best binary arm by 0.4671311 BPB. The historical fixed mixture scored
2.7561830, so this comparison deliberately uses the fresh control rather than
assuming bit-for-bit trajectory reproduction.

Median steady-state update times were **25.945 ms** for the control and
**25.923 ms** for prefix, measured from training-time deltas between validation
checkpoints after step 100. These single runs show no material steady-update
overhead, not a statistically established speedup. Training time including
initial compilation was 61.905 / 65.861 seconds respectively; total runner
wall time including validation and checkpointing was 100.391 / 95.790 seconds.

Verification passed **84 CPU contracts** and **51 compiled CUDA contracts**,
including strict-prefix causality, sequential/teacher-forced likelihood
agreement, full-domain normalization, fixed-code invariance, finite backward
credit, and existing native/mixture/softmax behavior. The trained prefix
checkpoint and all optimizer states are finite. Separate-process CLI
encode/decode recovered all 3,099 characters / 8,828 UTF-8 bytes with 3,099
distinct 12-bit addresses and zero residual bits. Enumerating all 4,096
addresses at three heldout contexts yielded total probability within
2.4e-7 of one, including nonzero reserved mass.

Canonical evidence: `ablation_results/nanomini_prefix_comparison/summary.json`
and `ablation_results/nanomini_bitflow_fixed_prefix128_2k/verification.json`.
The latter preserves checkpoint SHA-256, transport evidence, and full-domain
mass checks. The 4,729-byte NB01 alphabet packet is raw bit packing, not an
entropy-coded realization of the 2.0747585 BPB likelihood.

### Learned variable-length character refresh

**Status: experimental, not promoted.** This directly implements learned
variable-length character generation; no fixed-four-character training stage
was run. There is no fixed or maximum chunk length and no periodic refresh
rule. The ordinary training context window still resets at its BOS, and
generation stops at the caller's total output limit, not at a chunk limit.

`nanogpt_mini_adaptive_model.py` adds `--model adaptive`:

- Two 128-wide causal Mini blocks process previous character addresses. A
  learned BOS embedding supplies the initial local state; no current/future
  character is visible when deciding to refresh.
- A learned binary router reads that local state. At evaluation and generation,
  a nonnegative logit requests a global update. BOS always has one update.
  The router does not consume route-dependent global feedback, allowing its
  causal local features and training route samples to be computed in parallel.
- Selected local states are packed chronologically into the six-layer,
  512-wide global Transformer. Other characters reuse the latest global state.
  Packing rounds allocations to 32 event slots and records the padding work;
  these allocation buckets neither set chunk lengths nor create boundaries.
- Local features and the held global state feed the existing prefix-conditioned
  binary head. It autoregresses over the address bits for every character.
  The local character state continues across global refreshes.
- `adaptive_generation.py` uses preallocated local/global KV caches. Local
  positions count characters; global positions count actual selected updates.
  The global neural step is not run and discarded on a continue decision.

Training uses two independent Bernoulli route samples per input window.
Character NLL is averaged across the two routes. The gate receives a
leave-one-out score-function gradient: each action's loss-to-go is compared
with the other independent route's loss-to-go. The explicit compute price is
`refresh_cost * (forced_BOS_updates + sum(refresh_probabilities))`, with an
analytic gradient. The default `--refresh-cost 0.02` is nats per global update,
not a desired span length. There is no straight-through gate, clipping,
entropy schedule, or supplied boundary label.

Evaluation uses deterministic routing and reports **character NLL only** as
BPB. Compute cost remains a separate metric. Deterministic boundaries are
reproducible from the already-known character prefix, so they do not require
an uncharged stochastic boundary stream. All 4,096 address patterns remain
in the density; reserved-ID probability is still paid. Generation reports and
stops on a reserved address rather than resampling or renormalizing it away.

The initial all-zero BOS design failed the activated-network backward
contract: its local BOS gradient reached `4.2554e26` while other-position
gradients stayed near 2.7, then overflowed at the local input. Giving the
diagnostic a nonzero initial state removed the overflow. The implementation
therefore uses a learned nonzero local BOS vector in both parallel and cached
execution. Mini normalization is unchanged; no epsilon adjustment or gradient
suppression was applied. Evidence is in
`ablation_results/nanomini_adaptive_diagnostics/bos_gradient.json`.

```bash
# Use a fresh name: the ablation runner owns its output directory.
mlq submit --name adaptive_refresh_example_2k --cwd "$PWD" \
  --max-parallel-runs 1 --priority 1 --time-limit 45m -- \
  .venv/bin/python scripts/native_bits.py train \
    --data data/datasets/native_bits_fineweb \
    --name adaptive_refresh_example_2k --model adaptive \
    --local-layers 2 --local-dim 128 --prefix-width 128 \
    --refresh-cost 0.02 --steps 2000 --val-every 20

mlq submit --name adaptive_generation_example --cwd "$PWD" \
  --max-parallel-runs 1 --priority 1 --time-limit 30m -- \
  .venv/bin/python scripts/native_bits.py generate \
    --checkpoint ablation_results/nanomini_adaptive_refresh002_2k/checkpoint.pt \
    --prompt "The purpose of language is" --characters 128 --temperature 0.8 \
    --seed 1337 --output ablation_results/adaptive_generation_example.json
```

The completed run scored **2.2351199 BPB**, **0.1603614 worse** than the
2.0747585 prefix reference and 0.4239832 worse than character softmax. The
quality retention gate is not met; the prefix reference remains selected,
and no longer run was launched. This whole-model ablation also changes local
representation, fusion, BOS, and parameter count; it does not isolate the
effect of the routing rule alone.

On 65,536 validation characters, the model used **41,491 global updates**:
**36.6898% fewer updates**, averaging **1.5795 characters/update**. Completed
spans ranged from 1 to 9 characters; 24,498 were length 1 and 12,146 length 2.
Final spans at the 64 context-window ends are explicitly right-censored, not
counted as learned stop decisions. Padded parallel global work was 0.6875
positions/character. Router entropy fell to 0.0058346 nats by step 500 and
0.0009059 at step 2000; this observed early determinism does not by itself
diagnose the quality gap.

Median steady training updates were **37.3825 ms**, versus **25.9230 ms** for
the prefix reference: **44.2062% slower**. Both sampled routes and their
padding are counted in training work. Total runner wall time was 189.986 s,
including compilation, validation, and checkpointing.

A same-model inference-cost diagnostic replayed the same 256 source
characters through the cached runtime. Learned routing used 161 global
updates; forcing an update for every character used 256. After warmup, the
three-repeat median was **0.192063 s versus 0.257643 s**, a **25.4539% reduction**
in cached context-production time. This excludes address sampling and cache
allocation; it is not an end-to-end generation speedup against softmax or a
fixed-length training control.

All 16 adaptive compiled-CUDA contracts and the 51 existing comparison CUDA
contracts passed, alongside 122 CPU contracts. The actual checkpoint and
optimizer states are finite; all 3,099 alphabet characters roundtrip with
zero residual bits. Cached and parallel execution selected identical routes
over the 256-character replay. Logits were not bitwise identical (mean
absolute difference 0.01541, maximum 0.16024); total NLL was 465.6295 versus
465.6852 nats. Full-address probability mass at three cached contexts was
within 3e-7 of one, including nonzero reserved mass.

The separate `generate` CLI emitted all 128 requested characters without a
reserved address, using 96 global updates for 154 prompt/output positions.
Its cold runtime includes compilation and is not the warmed context benchmark.
Canonical evidence: `ablation_results/nanomini_adaptive_comparison/summary.json`,
plus `verification.json` and `generation.json` in
`ablation_results/nanomini_adaptive_refresh002_2k/`. Packets remain raw bit
packing; the measured likelihood is not an implemented entropy-compressed file.

### NextLat-style character latent dynamics

**Historical experiment: unverified rollouts, not speculative decoding.**
The 8,192-character-batch `--model dynamics` ablation removed the two global
training routes and local attention stack. Its 5.23% steady training overhead
was a cost measurement, not the intended source of an inference speedup.
The retired sampler committed approximate predictions without teacher
verification. Its loose-budget quality collapse does **not** evaluate exact
speculative drafting. The corrected runtime and normal-batch results follow below.

`nanogpt_mini_dynamics_model.py` reuses the fixed-prefix Mini teacher's
constructor and raw opaque-ID addresses. A single dense teacher pass supplies
all causal hidden states. A 128-wide residual MLP advances the state from the
previous state and emitted character bits; its input is RMS-normalized.
A small causal predictor estimates the next prediction's bit-prefix KL
discrepancy from the candidate state and `log1p(age)`. Refresh occurs when
that estimate reaches `--refresh-cost` (default 0.02 nats). It cannot see the
next character, teacher error, or future state when choosing.

Training supervises two-step latent rollouts by default, using normalized
smooth-L1 state loss, Bernoulli KL on the actual target's causal bit prefixes,
and error-prediction MSE. This is not an exact vocabulary-level KL enumeration.
Teacher states, targets, and auxiliary head parameters are detached: the
teacher/head receive only their ordinary character NLL; auxiliaries train
the transition and error predictor. `train_teacher_bpb` distinguishes that
rate from the total auxiliary objective. Historical `val_bpb` scored the
**unverified deterministic rollout**. Canonical evaluation now scores the dense
teacher, matching the distribution of verified speculative generation.

The two-step supervision horizon is **not an inference chunk limit**.
There is no periodic refresh, forced maximum span, or character EOS.
Offline teacher-forced evaluation precomputes causal reference states once,
but consumes a reference only after the learned gate requests a refresh.
The historical cached diagnostic in `dynamics_generation.py` instead appends every
pending character input to the exact KV cache in one causal suffix pass,
using absolute-position RoPE and attention masks. It never silently drops
skipped characters or recomputes previously cached positions.

That retired policy batched backbone **calls**, not steady-state backbone
**positions**. It was not exact speculative sampling: cheaply generated
characters were retained and their own binary-head likelihood was charged.
Refresh restored the full-history teacher state conditioned on those characters;
it did not retroactively substitute teacher likelihoods for cheap predictions.

The first unnormalized transition failed validation after update 20.
All parameters and teacher states were finite, but a continuing latent
trajectory grew to `7.4156e21` at character position 43 and NaN at 44.
Input normalization fixes this recurrent positive-feedback overflow without
clipping, an epsilon change, a forced refresh interval, or an eager fallback.
A 65-character positive-feedback regression failed before the fix and
passes afterward. The captured failure is in
`ablation_results/nanomini_dynamics_diagnostics/rollout_failure.json`.
The failed run is not included as a completed comparison arm.

```bash
mlq submit --name dynamics_example_2k --cwd "$PWD" \
  --max-parallel-runs 1 --priority 1 --time-limit 45m -- \
  .venv/bin/python scripts/native_bits.py train \
    --data data/datasets/native_bits_fineweb --name dynamics_example_2k \
    --model dynamics --dynamics-width 128 --rollout-horizon 2 \
    --prefix-width 128 --refresh-cost 0.02 --steps 2000 --val-every 20
```

The full 2,000-step run scored **2.0659811 BPB**, versus **2.0641140** for
the fresh paired prefix control: 0.0018671 worse, not a promotion.
The dynamics checkpoint's own dense teacher scored **2.0647694**.
Independent compiled runs did not produce bitwise-identical trained teachers;
the fresh control and same-checkpoint teacher are therefore reported separately.

Median steady training updates were **27.6058 ms versus 26.2330 ms**, measured
from training-time deltas after step 100. Total training time including
compilation was 75.782 s versus 53.450 s. Full runner wall time was
**217.677 s versus 74.344 s**: dynamics performs sequential deployed
validation every 20 updates, whereas the ordinary prefix control evaluates
its density in parallel. The 5.23% figure is not an end-to-end wall-time claim.

The primary policy made **64,370 calls for 65,536 characters** (1.7792%
fewer calls), but processed **all 65,536 backbone positions** after catch-up.
Completed spans ranged from 1 to 4 characters; mean characters/call was
1.0181. The 64 final window spans remain explicitly right-censored.
On a warmed 256-character cached replay, context production plus address
acceptance took **0.223749 s versus 0.196868 s** when forcing every-character
refresh: **13.65% slower**, despite five fewer calls. These three-repeat
medians exclude address sampling and cache allocation and are not an
end-to-end softmax comparison.

Frozen-checkpoint diagnostics increased the inference error budget without
retraining or selecting a new default:

| Refresh threshold (nats) | Deployed BPB | Backbone calls / 65,536 chars | Positions processed |
| ---: | ---: | ---: | ---: |
| 0.02 (checkpoint policy) | 2.0659811 | 64,370 | 65,536 |
| 0.10 (diagnostic) | 39.8086746 | 38,567 | 43,591 |
| 0.50 (diagnostic) | 110.5011799 | 2,078 | 4,686 |

The lower position counts at loose budgets include long unrefreshed tails,
not successful sparse-memory compression. Those unverified policies are unusable
at these likelihoods. This is not evidence against target-preserving speculative
decoding. No longer training run or promotion was launched for that experiment.

Verification: **14 dynamics CUDA contracts**, **67 existing comparison CUDA
contracts**, and **195 CPU contracts** passed. Checkpoint and optimizer
states are finite. Separate-process CLI encode/decode exactly roundtripped
all 3,099 alphabet characters with 12-bit addresses and zero residual.
Cached and offline policies agreed on all 256 replay decisions; their NLLs
were 431.3235 and 431.0965 nats, respectively (0.001279 BPB difference).
BF16 logits were not bitwise identical. Enumerating all 4,096 addresses at
three cached contexts gave total mass within 6e-7 of one, including reserved
mass. Actual CLI generation emitted all 128 requested characters without
a reserved address, using 150 calls and 154 processed prompt/output positions.

Canonical evidence: `ablation_results/nanomini_dynamics_comparison/summary.json`,
plus `verification.json`, `generation.json`, and `budget_diagnostics.json`
under `ablation_results/nanomini_dynamics_norm_h2_refresh002_2k/`.
The fixed-prefix family remains the binary reference; all scores here are
prepared-character proxies, not challenge BPB or implemented entropy coding.

### Normal batch and target-verified speculative decoding

**Result: not promoted.** Both models completed 2,000 updates with **524,288
characters/update**, 1,024-character contexts, 64 rows/microbatch, eight accumulated
microbatches, seed 1337, and validation every 20 updates. Each consumed
1,048,576,000 characters, approximately 5.0338 passes over the 208,307,257-character
training stream. Validation uses the same 65,536 characters / 66,121 UTF-8 bytes.
The original BPE Mini job was canceled before starting and was kept canceled at
the user's request; the baseline below is **character-softmax Mini**, not BPE Mini.

| Model | Parameters | Dense target BPB | Steady update | Runner wall time |
| --- | ---: | ---: | ---: | ---: |
| Character-softmax Mini | 22,085,659 | **1.4019525** | 616.19 ms | 1,266.21 s |
| Binary-prefix Mini with dynamics | 19,184,270 | 1.4828257 | 650.50 ms | 1,350.94 s |

The binary system is **0.0808732 BPB worse**, with 13.14% fewer parameters and
5.57% slower steady updates. This compares complete architectures; it does not
isolate the effect of the auxiliary dynamics loss from the different input/output
heads. The earlier 8,192-character-batch scores were 1.8111367 for character
softmax and 2.0647694 for the dynamics checkpoint's dense teacher. The batch
increase gives 64 times more training characters at the same update count.

The first normal-batch dynamics attempt failed before its first update because
Inductor could not compile a symbolic mixed-reduction backward through horizon
slices (`CantSplit`). Specializing the fixed-shape training objective fixes that
compiler failure without reducing the batch, changing the objective, or using an
eager fallback. The successful run is
`nanomini_dynamics_batch524288_static2k`; the failed attempt remains archived
separately and is not a completed result.

#### Correct inference contract

`speculative_generation.py` uses the existing frozen transition only as a draft
model. One target suffix pass verifies the pending character plus proposed
characters. Each proposed bit is accepted with probability `min(1, p/q)`, using
the proposal logits captured during sampling. At the first rejection, the binary
positive-residual distribution selects the opposite bit; later bits are sampled
from the target conditioned on the accepted prefix and corrected bit. Later
drafts are discarded and the KV cache is rolled back. All-accepted proposals
receive a target-sampled bonus character.

No learned error gate decides what output to keep. `--draft-tokens` is an
inference-only proposal budget, not a training change; zero selects the matched
target-only sampler. The binary address domain is not renormalized: the first
**committed** reserved address terminates generation and is recorded once,
whereas an unaccepted reserved proposal does not terminate it.

Distribution preservation is mathematical, not a claim of bitwise identity
between differently shaped BF16 kernels. The runtime preserves Mini's explicit
BF16 arithmetic; enabling CUDA autocast had promoted RMSNorm/residual states to
FP32 and failed the cached-versus-dense regression. The fix preserves the actual
teacher precision rather than weakening the regression's dtype check.

`scripts/native_bits.py generate` now uses this verifier for dynamics checkpoints;
the old unverified generator has been removed. Canonical dynamics evaluation
also scores the dense teacher. Archived training logs still contain the old
approximate rollout metric (1.4836531 at the normal-batch endpoint), so the table
above uses the independently scored target likelihood, not that archived field.
Legacy rollout diagnostics remain explicit diagnostics, not the public sampler.

#### Matched full-sampling benchmark

RTX 5090, the same normal-batch checkpoint for every budget, three prompts, three
sampling repeats per prompt, 256 output characters, temperature 0.8. Every exact
seed/prompt/budget workload is warmed before measurement; budget order rotates.
All **36 measured generations completed**, without a committed reserved address.

| Draft budget | Median warm decode / 256 chars | Speedup vs target-only | Committed / proposed drafts | Target calls (9 runs) | Target positions (9 runs) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 258.184 ms | 1.000x | — | 2,295 | 2,295 |
| 1 | 256.821 ms | 1.005x | 54.53% | 1,488 | 2,968 |
| 2 | 257.844 ms | 1.001x | 41.85% | 1,253 | 3,743 |
| 4 | 309.511 ms | 0.834x | 24.25% | 1,170 | 5,809 |

Budgets one and two are effectively break-even, not a demonstrated meaningful
speedup. Their decode-time ranges overlap the target-only range substantially.
Budget four is 19.88% slower. Budget two saves 45.40% of target calls but processes
63.09% more target positions because rejected suffixes still cost work.
Call reduction alone is not a latency improvement.

Decode timing includes real character sampling, proposals, verification,
correction and host coordination; it excludes separately reported allocation/RNG
setup and prefill. Median end-to-end times were 259.269, 257.896, 259.046 and
310.729 ms for budgets 0/1/2/4. Cold calls, warmups and raw repeats are retained;
compilation is never silently subtracted. A separate cold public-CLI invocation
generated all 128 requested characters with budget two: 75 decode target calls,
224 decode positions, plus 27 prefill positions.

```bash
mlq submit --name dynamics_exact_generation_example --cwd "$PWD" \
  --max-parallel-runs 1 --priority 1 --time-limit 20m -- \
  .venv/bin/python scripts/native_bits.py generate \
    --checkpoint ablation_results/nanomini_dynamics_batch524288_static2k/checkpoint.pt \
    --prompt "The history of science is " --characters 128 \
    --temperature 0.8 --seed 1337 --draft-tokens 2 \
    --output ablation_results/dynamics_exact_generation_example.json

mlq submit --name dynamics_exact_benchmark_example --cwd "$PWD" \
  --max-parallel-runs 1 --priority 1 --time-limit 1h -- \
  .venv/bin/python scripts/speculate_dynamics.py benchmark \
    --checkpoint ablation_results/nanomini_dynamics_batch524288_static2k/checkpoint.pt \
    --prompt "The history of science is " --prompt "To solve this problem, " \
    --prompt "In the city, " --characters 256 --temperature 0.8 \
    --seed 1337 --draft-tokens 1 2 4 --repeats 3 --warmup 1 \
    --output ablation_results/dynamics_exact_benchmark_example.json
```

Verification: **27 dynamics/speculation CUDA contracts**, **16 adaptive CUDA
contracts**, and **195 CPU contracts** passed. Coverage includes actual normal-size
training backward, Bernoulli residual correctness, rollback against dense teacher
states, reserved-code handling, and gate-independent canonical teacher likelihood.
The full-prefix evaluation smoke matched the independently scored teacher BPB
exactly for both checkpoints; checkpoint and optimizer states are finite.

Canonical evidence: `ablation_results/nanomini_normal_batch_comparison/summary.json`
and `exact_speculation_benchmark.json` / `exact_generation.json` under
`ablation_results/nanomini_dynamics_batch524288_static2k/`. Checkpoint hashes,
training provenance, inference source hashes, device/precision settings and raw
measurements are recorded. Scores remain prepared-character proxies, not
challenge BPB. No 8k run, new embedding architecture, or promotion was launched.

### Dynamics spike mechanism: frozen batches and closed-loop traces

Three different quantities had been conflated: dense-teacher BPB, unverified
rollout BPB, and the total training objective (which includes auxiliary MSEs).
The public dynamics sampler remains target-verified; the learned error gate is
used only by the explicitly labeled legacy diagnostic.

**The catastrophic rollout spike is a recurrent-state radius feedback loop.**
The transition normalizes its concatenated input, but previously returned
`state + residual` without normalizing that resulting state. A bounded residual
increment does not bound its accumulated state. Increasing state radius
suppresses the unit-scale character bits inside the joint input normalization
and overwhelms the decoder's bit-prefix features. Meanwhile the error critic's
unnormalized ReLU-squared features can extrapolate toward large *negative* raw
scores, so softplus predicts near-zero error precisely when refresh is needed.

The replay reproduced this mechanism on the full 65,536-character validation
prefix, without changing training math:

| Checkpoint step | Dense teacher BPB | Legacy rollout BPB | Same routes, teacher-radius decoder input | Maximum candidate RMS | Maximum rollout age |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 11.18118 | 119.16634 | 12.19014 | 35.7910 | 1,023 |
| 40 | 10.03989 | 101.32557 | 10.83824 | 29.4519 | 1,022 |

The radius-only decoder intervention removes **99.07% of step 20's excess BPB
over the teacher**, while keeping the original routes and recurrent trajectory.
It uses the actual teacher radius and is therefore an **oracle diagnostic, not
a deployable correction or quality improvement**. At step 20 the teacher's
maximum RMS is 0.91786, the critic's median raw score is −9,152, its median
predicted KL is zero, and the measured median Bernoulli KL is 72.943. There are
65,426 false-safe positions, including 59,668 where every bit logit saturates
to the same sign. Only the initial BOS refresh occurs; the transition was
supervised at horizons one and two, not these thousand-step tails.

The initial critic update explains why this failure is especially accessible
early in training. Both output heads start at zero: teacher and draft predict
one-half, so the critic's KL label is zero but its softplus prediction is
`log(2)`. For a ReLU-squared hidden activation `a_j >= 0`, the initial MSE
gradient of output weight `v_j` is `log(2) * mean(a_j) >= 0`. The first AdamW
update therefore makes active output weights negative. Large out-of-training
activations then imply *smaller* estimated error. This is a confidence-model
extrapolation failure, not evidence that clipping the optimizer would fix it.
At the normal-batch final checkpoint, 795 of 1,024 signed coordinate rays still
have negative leading quadratic coefficients; this establishes bad radial
extrapolation directions, not that every such direction occurs in a rollout.

**Ordinary training-BPB jumps mostly follow batch difficulty.** Frozen final
weights were evaluated on the exact historical training windows, with no
updates or reshuffling:

| Run | Historical batch pair | Historical BPB increase | Increase at frozen final weights |
| --- | --- | ---: | ---: |
| Dynamics, batch 8,192 | 1,050 → 1,060 | 1.81840 | 1.66469 |
| Prefix control, batch 8,192 | 1,050 → 1,060 | 1.77643 | 1.47782 |
| Dynamics, batch 524,288 | 1,010 → 1,020 | 0.18434 | 0.17616 |
| Character softmax, batch 524,288 | 1,010 → 1,020 | 0.17445 | 0.16500 |

For the normal dynamics pair, 95.57% of the historical jump's magnitude remains
at fixed final weights; this is not an exact causal attribution at historical
weights. Dynamics and softmax training BPB correlate at 0.98023 after step 800.
At steps 1,910 → 1,920, the total dynamics objective rises by 0.57231 nats/char:
0.04796 teacher rate, 0.00072 latent loss, 0.10762 rollout KL, and **0.41601
critic MSE** (72.69% of the total increase). The critic's graph is detached from
the teacher and transition. Auxiliary MSE is not a coding rate; only actual
`*_nats` rate statistics now receive corresponding BPB fields.

The user stopped the observational replay after its last recorded checkpoint at
step 1,180. At that checkpoint teacher BPB is 1.59606, legacy rollout BPB is
1.59721, and maximum rollout age is five. It was not a quality candidate and
was not restarted. Initial model tensors and CPU/CUDA RNG states match the
historical normal run exactly, but subsequent numerical trajectories differ:
this is a reproduction of the failure mechanism, not a bitwise replay.

Evidence: `ablation_results/nanomini_dynamics_mechanistic/` contains
`frozen_batches.json`, `historical_spike_decomposition.json`,
`critic_parameter_geometry.json`, `replay_initialization_comparison.json`, and
`confirmed_mechanism.json`. The replay directory retains per-position `.npz`
traces, validation/update JSONL streams, and the preceding/current full
checkpoints for spike events. `scripts/diagnose_dynamics.py` reproduces the
frozen-batch and rollout analyses through the existing trainer.

#### Recurrent output normalization: mechanism confirmed, retrofit rejected

`nanogpt_mini_dynamics_bounded_model.py` isolates one architectural change:
`next_state = RMSNorm(state + residual)`. All existing parameters initialize
identically to the residual model; the added norm gains start at one. For fixed
finite gains, output RMS is bounded by their maximum absolute value (up to
roundoff), independently of rollout age. Four CUDA contracts passed, including
the normal 64×1,024 backward for both variants and a 256-step residual-drift
regression. This is an explicit experimental model kind, `dynamics_bounded`,
not a replacement for the baseline or a promoted quality result.

A second frozen-checkpoint probe reran the **whole recurrent trajectory** with
that norm, leaving every shared parameter and the learned gate unchanged:

| Checkpoint step | Original rollout BPB | Output-normalized rollout BPB | Normalized maximum state RMS | Teacher NLL change |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 119.16634 | 12.48226 | 1.00049 | 0 |
| 40 | 101.32557 | 10.55946 | 1.00061 | 0 |
| 1,180 | 1.59721 | **18.35923** | 1.00189 | 0 |

At the two early spikes this removes 98.80% and 99.43% of excess BPB and all
same-sign-saturated false-safe positions. No teacher-radius oracle, age cap,
periodic refresh, gradient clipping, or retraining is used. But inserting a
unit-radius norm into the later checkpoint destroys its learned state geometry
and worsens the gate's decisions: refresh falls from 98.66% to 14.83%.
**A radius bound is not error calibration, and the frozen retrofit is rejected.**
Training this representation from initialization was not run; neither a
2000-step BPB improvement nor a decoding speedup is established.

The operational correction remains target verification, which does not trust
the learned critic and preserves teacher probabilities. The new autoregressive
embedder instead makes its lossy recurrent representation part of the actual
language model, rather than assuming an unverified approximation to a separate
teacher. Its quality and compute cost require their own matched comparison.
Full intervention evidence: `bounded_intervention.json` in the mechanistic
evidence directory; reproduce with `scripts/diagnose_dynamics.py bounded-probe`.

### Autoregressive character embedder: capped screening, not promoted

The delegated implementation is `nanogpt_mini_embedder_model.py`, integrated as
`--model embedder` in `scripts/native_bits.py`. It uses a persistent
128-dimensional diagonal-affine character recurrence, evaluated with a parallel
scan during training. After consuming character `t`, a causal gate can emit its
summary to the six-layer, width-512 global Transformer before predicting `t+1`.
Only emitted summaries and BOS enter the packed global sequence; this is not a
dense character Transformer with an attention mask. Emission does not reset the
local state. A local-plus-held-global readout scores every next character with
an observed-alphabet softmax. There is no approximation to a separate teacher.

The fixed-stride-four and learned-gate variants have identical parameter
layouts and initialization, differing only in route selection. The learned
variant uses one sampled route, a detached suffix-return REINFORCE advantage,
an action-independent mean-future-NLL critic, and an analytic expected emission
charge of 0.02 nats/event. Its deterministic evaluation policy emits when the
gate logit is nonnegative. Critic regression inputs are detached; auxiliaries
and work counters are not reported as BPB. Useful and padded Transformer
positions, character-encoder positions, gate evaluations and scan compositions
are separate statistics.

**Both runs stopped at 200 updates at the user's request.** The runner used
`--steps 2000 --stop-at 200`: the original learning-rate schedule is preserved,
and validation/checkpoint writing precede termination. Each run consumed
104,857,600 characters at the normal 524,288-character batch. Seed, data,
1,024-character contexts, 64-row microbatches and validation prefix match the
historical character-softmax control. The two originally queued full runs were
canceled before starting; BPE stayed canceled.

| Model | Parameters | Step-200 proxy BPB | Eval useful global positions/char | Eval padded positions/char | Update time, steps 100–200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical character-softmax control | 22,085,659 | **1.7988339** | 1.0000 | 1.0000 | 615.04 ms |
| Fixed-stride-four embedder | 21,061,789 | 2.7708755 | 0.2500 | 0.2500 | 238.95 ms |
| Learned-gate embedder | 21,061,789 | 3.3559706 | 0.1425 | 0.2031 | 174.45 ms |

The capped runner wall times were 76.58 s fixed and 68.79 s learned, including
validation, checkpoints and compilation. Neither justifies extension. This is
not only a step-count disadvantage: the baseline already reaches 2.57052 BPB
after 51.43 s of recorded training time, versus the fixed embedder's 2.77088
after 60.74 s. These are historical timing comparisons, not simultaneous
hardware-controlled trials or 2,000-step promotion results.

The learned policy emits nothing at validations 20 and 40, with expected
emission probabilities only 0.00075–0.00089, then recovers to 14.17% deterministic
emissions by step 200. Fixed routing avoids that early collapse but still trails
the baseline by 0.97204 BPB. Thus gate learning is not the only unresolved issue;
the fixed representation/readout/training combination also needs a better
quality–compute tradeoff. The screen does not isolate which of those components
causes the gap. No clipping, forced-emission schedule or additional training
was added to conceal these results.

#### Actual cached generation and numerical checks

`embedder_generation.py` retains the local recurrence and a KV cache indexed by
**emission position**. `character_generation.py` is the genuine dense
CharacterGPT control, sharing only the Mini cache arithmetic and character
sampler through `mini_cached.py`. Both avoid consuming the final generated
character when no subsequent prediction is requested; a zero-character request
does no neural or prefill work. The public `generate` CLI supports both model
kinds and records current inference source hashes.

`scripts/benchmark_embedder.py` ran three heldout 128-character prompts, three
repeats each, 256 sampled output characters, temperature 0.8, identical seed
schedules and rotating variant order. Each exact workload was warmed before
its measured run. All **27 measured generations completed**:

| Model | Aggregate warm decode chars/s | Measured ratio | Decode global positions, nine runs | Decode encoder positions |
| --- | ---: | ---: | ---: | ---: |
| Character-softmax control | 1,287.44 | 1.000x | 2,295 | 2,295 |
| Fixed embedder | 2,703.64 | 2.100x | 567 | 2,295 |
| Learned embedder | 3,064.28 | 2.380x | 355 | 2,295 |

These are real sampling times, not FLOP estimates; allocation/prefill and
cold-first-use timings are retained separately. **This is not a matched-quality
speedup:** the available baseline checkpoint has 2,000 updates and the embedders
have 200, with substantially worse BPB. Learned-route cost also depends on
generated content.

Untouched trained weights were checked on three real 1,024-character heldout
prefixes. Both variants had **0/3,069 emission-decision mismatches** between the
parallel encoder and sequential generation recurrence. The maximum learned
gate-logit difference was 0.03125; maximum normalized-state differences were
0.17578 fixed and 0.03125 learned. No gate-margin intervention was used. This
finite check does not establish bitwise identity or exact dense/incremental
likelihood equality for arbitrary contexts.

Verification: **220 CPU contracts**, a **31-case embedder CUDA suite**, and the
additional **normal-size 64×1,024 backward regression** passed. The first
normal-size screening attempts failed before any update with Inductor
`CantSplit`; specializing the fixed batch/character axes while leaving packed
event capacity dynamic fixed the compiler failure without changing arithmetic
or falling back to eager execution. Failed attempts remain separate evidence.

Completed capped runs:
`nanomini_embedder_fixed4_batch524288_static_screen200` and
`nanomini_embedder_learned_batch524288_static_screen200`.
Canonical comparison and inference evidence:
`ablation_results/nanomini_embedder_screen200_comparison/{summary,inference}.json`.
The comparison records checkpoint/source hashes, complete validation curves,
training and generation work counters, queue jobs and limitations. No automatic
extension, 2,000-step quality win, challenge BPB improvement, or submission-size
claim is made.

## Paired Cola-DLM controls: approximately 120M parameters

`pretraining/cola/` implements the approved BPE and literal-byte research
controls; `scripts/train_cola.py` is the operational entrypoint. Architecture
version `cola_nanogpt_120m_v3` supersedes the earlier v2 controls. These are
**2,000 VAE-preparation updates followed by 2,000 joint updates per arm**,
not 2,000 updates total. No subsequent ablations have started.

### Architecture, data and runtime

| Configuration | Released-tokenizer BPE | Literal bytes |
| --- | ---: | ---: |
| Vocabulary, excluding encoder-only mask | 100,278 | 256 |
| Total trainable parameters during joint training | 117,727,718 | 119,612,336 |
| VAE parameters | 56,859,094 | 21,843,232 |
| DiT prior parameters | 60,868,624 | 97,769,104 |
| VAE encoder/decoder | 4 + 4 layers, width 256 | 4 + 4 layers, width 512 |
| DiT | 8 layers, width 640 | 13 layers, width 640 |
| Positions per sequence / prior block | 512 / 16 | 2,560 / 80 |

Both use 16-dimensional stochastic latents at every input position: **the VAE
does not compress sequence length**. Matching total parameter scale is not
matching backbones, VAE/prior budget allocation, FLOPs or wall time.

The source is the same literal `data/datasets/native_bits_fineweb/` stream:
209,641,042 training bytes and 124,766 validation bytes, without normalization,
BOS/EOT insertion or additional document isolation. This is the local FineWeb
web-validation proxy, not challenge heldout. BPE uses the released tokenizer at
`ByteDance-Seed/Cola-DLM@c1eafdd9cfd8064aeb917d569ef70a075b353eed`;
`scripts/prepare_cola_bpe.py` produces a roundtrip-verified byte-accounted cache.
Full provenance and source hashes are recorded per run.

Five times as many byte positions is approximate context matching, not exact
data matching. Each completed 2,000-update BPE stage consumes 151,584,126 source
bytes versus 163,840,000 for byte. Preparation and joint stages revisit the
training prefix; their sum is consumed bytes, not unique data.

Neural computation uses compiled BF16 with FP32 master parameters, reductions
and ODE states. The VAE uses causal Flash SDPA; full clean/noisy prior attention
uses the explicit TRITON Flex backend. No CPU or eager fallback is used.
Block matrices use five-step Newton–Schulz Muon; remaining parameters use fused
AdamW. Rates are 0.002 matrix/embedding, 0.0001 other and 0.0003 scalar, with
40-update warmup, a hold through 30%, then linear cooldown to 5%.
Each update has 32 sequences; joint microbatch is 4. Validation runs every
20 updates. Byte preparation uses microbatch 8; BPE preparation uses 4.

### Corrected runs and current interruption

| Run | Updates | Stage wall time | Final generative proxy BPB |
| --- | ---: | ---: | ---: |
| `cola_bpe_120m_v3_vae_2k` | 2,000 | 329.07 s | Not applicable |
| `cola_bpe_120m_v3_joint_2k` | 2,000 | 590.82 s | **4.77466258** |
| `cola_byte_120m_v3_vae_2k` | 2,000 | 540.40 s | Not applicable |
| `cola_byte_120m_v3_joint_2k` | Last logged update 830 | Interrupted | Not evaluated |

The BPE preparation-plus-joint wall time is 919.89 s on the RTX 5090, excluding
the separate CNF evaluation. Its generated examples remain incoherent; low VAE
reconstruction error is not evidence of good generation.

Queue job **8081** was cancelled by request with SIGTERM15, not a reported model
exception. Its dependent evaluation **8082** was skipped. The initiator is
unknown. The preserved byte recovery checkpoint is at **step 759**, earlier
than the last logged training update 830 and validation 820; its source hashes
match the current implementation. No automatic restart was submitted.
The completed preparation is retained. **There is no finished paired 2,000-step
quality comparison and no byte likelihood proxy or winner.**

Completed queue jobs: primitive parity 8075, CUDA contracts 8076, BPE
preparation/joint/evaluation 8077–8079, byte preparation 8080.
Canonical evidence:

- `ablation_results/cola_120m_paired_comparison.json` explicitly records the
  incomplete comparison, completed results, hashes and limitations.
- Each completed run has `metrics.jsonl`, `result.json` and `provenance.json`.
- `ablation_results/cola_byte_120m_v3_joint_2k/interruption.json` records the
  cancellation, final logs and recovery metadata.
- `ablation_results/cola_bpe_120m_v3_joint_2k/evaluation.json` retains both solver
  resolutions and actual generation; `proxy_bpb.json` records the published
  metric and its checkpoint/evaluation hashes.

### What the proxy measures and where to see it

`val/proxy_bpb` is a **final step-2000 point, not a live training curve**.
It is the numerical CNF negative-ELBO estimate
`(reconstruction NLL - prior log density + posterior log density) / (bytes * ln 2)`,
using one posterior sample and one Gaussian trace probe per context over the
entire literal heldout file, including the final short context.

The corrected BPE estimate is 4.77084760 with 16 Heun steps and **4.77466258**
with 32. Their difference, 0.00381498 BPB, is a solver-resolution diagnostic,
not an error bound. The finest estimate decomposes into 0.31166255
reconstruction + 3.42751566 prior NLL + 1.03548440 posterior log-density
bits/byte. This is not exact marginal likelihood, a certified numerical bound,
or challenge BPB. VAE reconstruction bits/byte, masked CE and flow MSE are
different metrics and must not be substituted.

Open [TensorBoard Time Series](http://127.0.0.1:6101/?runFilter=cola_.*joint_2k&tagFilter=%5Eval%2Fproxy_bpb%24#timeseries)
and **check `cola_bpe_120m_v3_joint_2k` in the run selector**. Filtering the list
does not select a newly discovered run. This interaction and the visible
4.7747 point were browser-verified; the screenshot is
`ablation_results/cola_dashboard_visibility.png`.
The older `cola_bpe_120m_joint_2k` point, 4.71956629, belongs to superseded v2.

After a completed joint run, submit the real evaluation through `mlq`, then
publish its result (the publisher does not execute a model):

```bash
mlq submit --name cola_bpe_v3_evaluation --cwd "$PWD" --max-parallel-runs 1 --priority 1 --time-limit 4h -- .venv/bin/python scripts/train_cola.py evaluate --checkpoint ablation_results/cola_bpe_120m_v3_joint_2k/checkpoint.pt --output ablation_results/cola_bpe_120m_v3_joint_2k/evaluation.json --resolutions 16 32
.venv/bin/python scripts/publish_cola_proxy.py --run-dir ablation_results/cola_bpe_120m_v3_joint_2k
```

Run the publisher only after evaluation succeeds. It verifies completion,
checkpoint identity, byte coverage and ELBO arithmetic, then writes the
canonical JSONL/TensorBoard point. Re-publishing the same evaluation is
idempotent; it does not promote the proxy to exact `final_val_bpb`.
All training, qualification and evaluation workloads must use `mlq` with
parallel limit 1 and priority 1. Resume uses `train --resume` into a fresh run
directory and requires exact source, data and configuration compatibility.

### Reference audit and verification limits

The audit compared reference source
`ByteDance-Seed/Cola-DLM@7d1daeea1455a6cb9e23ddd4f06b8a2e59e63a8c`
and the released checkpoint configs. V3 restores the kernel-1 input
projection's native Conv-equivalent initialization and separates VAE norm
epsilon `1e-6` from DiT epsilon `1e-5`. Chosen nanoGPT optimizers, compiled
runtime and zero heads remain declared control differences. The paper's full
numeric training recipe is not released; this is not an exact reproduction.

The original BPE-preparation spike at step 220 was masked reconstruction CE
(13.9571), not clean reconstruction or DiT flow loss. A complete unchanged
2,000-update instrumented replay did not reproduce it. Initial statistics and
logged mask counts matched, but later numerical trajectories were not
bitwise identical. Therefore neither the replay nor the corrected run
establishes the original spike's cause.

Verification completed: **47 CPU checks passed**, 17 CUDA cases skipped in that
CPU invocation; **32 CUDA checks passed**, one optional case skipped.
Reference primitive checks cover both projection sizes and normalization
epsilons, not full weight-mapped upstream model parity. Reports are
`ablation_results/cola_v3_reference_parity.json` and
`ablation_results/cola_bpe_reference_check.json`. Finished diagnostic programs
are archived in `ablation_results/cola_reference_diagnostics.zip`; the
throwaway `/tmp` copies are removed. No SOTA, ablation-win or 16MB submission
claim is made.

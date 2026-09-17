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

## FFN-only streaming TD value pretraining

`scripts/train_future_credit_stream.py` trains
`pretraining/future_credit_stream/` on the non-GPT2 FineWeb
SentencePiece-1024 corpus. Six width-512 residual FFNs use the nanoGPT-mini
ReLU-square MLP convention, without attention, token-history buffers, or memory
slots. Each document carries one detached BF16 vector, injected before the
first FFN; FP32 master parameters train with fused AdamW.

The actual current hidden produces next-token CE and becomes the detached
recurrent input. A tiny, training-only scalar critic predicts remaining
prediction loss, not a vector of synthetic gradients:

```text
V(h_t) estimates CE_(t+1) + discount * CE_(t+2) + ...
producer_t = mean(CE_t + discount * V(h_t; frozen critic weights))
target_(t-1) = stop_gradient(CE_t + discount * V(h_t))
critic_(t-1) = 0.5 * mean((V(stop_gradient(h_(t-1))) - target_(t-1))^2)
loss_t = producer_t + critic_(t-1)
```

This is ordinary teacher-forced pretraining: targets use observed corpus
tokens, not sampled rollouts or contrastive negatives. The 64-unit squared-ReLU
critic adds no inference backbone capacity. Its first projection uses BF16;
the final scalar accumulation uses FP32 so cumulative values can retain small
TD residuals. It is initialized after the common FFN parameters, with zero
output weights, preserving same-seed CE initialization.

The producer differentiates through a **frozen critic** into only the current
FFN. Critic regression sees the **original detached previous carry**, not the
reset state, and its CE/value teacher is detached. A current BOS target removes
the current future-value term while retaining CE. A document reset sets the
previous state's entire TD target to zero, including that terminal row in
regression.

Each token uses **one compiled forward and one combined backward**. There is
no incoming-state VJP, lookahead, boundary replay, additional history vector,
or linked temporal graph/TBPTT. The previous TD feature is the same vector
already needed for recurrence. A global `has_previous` flag skips fitting only
the first observation's nonexistent predecessor; it stays true across optimizer
pages and is restored from the saved optimizer-step count. Every observed
transition is eligible, including transitions across update boundaries. No
pending graph survives a token backward or enters a checkpoint. Carry and
accumulated gradients use external buffers across CUDA-graph replays.

The default `discount=1` targets the undiscounted finite-document continuation;
values in `[0,1]` are configurable. A calibrated value estimate is not a guarantee
of useful input gradients: critic exploitation and moving-target instability
remain algorithmic risks. Only the matched BPB ablation can establish benefit.

The mmap loader maintains 4,096 independent document lanes, preserves
cross-shard documents, prefetches one bounded page, and checkpoints the consumed
rather than speculative cursor. A complete document ends at the following real
BOS target; incomplete corpus edges are excluded. The default 2,000 updates
consume 65,536,000 token targets, with validation every 20 updates.

```bash
mlq submit --name ffn_td_value_sp1024_2k --cwd "$PWD" \
  --max-parallel-runs 1 --max-attempts 1 --time-limit 4h \
  --env FUTURE_CREDIT_STREAM_RUN_ID=ffn_td_value_sp1024_2k \
  --env FUTURE_CREDIT_STREAM_OBJECTIVE=td -- \
  .venv/bin/python scripts/train_future_credit_stream.py

mlq submit --name ffn_td_value_ce_sp1024_2k --cwd "$PWD" \
  --max-parallel-runs 1 --max-attempts 1 --time-limit 4h \
  --env FUTURE_CREDIT_STREAM_RUN_ID=ffn_td_value_ce_sp1024_2k \
  --env FUTURE_CREDIT_STREAM_OBJECTIVE=ce -- \
  .venv/bin/python scripts/train_future_credit_stream.py
```

Use fresh run identifiers for new experiments. Configuration overrides use
`FUTURE_CREDIT_STREAM_<FIELD>`. Resume with
`--resume ablation_results/<run>/checkpoint.pt` and identical configuration,
data, tokenizer, and source hashes. An improved validation at the resumed
starting step also publishes `best.pt`. Generate through `mlq` with
`--generate ablation_results/<run>/checkpoint.pt --prompt "Some text"`.
Neither validation nor generation invokes the critic. The architecture tag
`streaming_ffn_td_value_v1` rejects historical vector-credit and NextLat
checkpoints. Resume source fingerprints include the baseline utility that
defines validation byte accounting; that upstream source is not modified.

### Scalar TD acceptance gates

Control-like speed means at most **1.15x** the matched CE update time. The
throughput-only check uses full 4,096-lane, eight-tick optimizer pages, actual
prefetched corpus data, one process, 40 warmup updates per arm, and three
alternating 100-update timing rounds. It checks for unmanaged Python GPU
workloads and uses median round timings; this is not reduced-run BPB evidence.
The learning comparison remains 2,000 updates, with an improvement greater than
0.005 proxy BPB required for promotion. Neither speed nor learning benefit is
assumed from the implementation.

### Historical vector-credit ablation

The matched CE control, `ffn_future_credit_ce_sp1024_2k`, completed 2,000
updates and 65,536,000 targets: **2.139481 proxy BPB**, 3.309343 reset-state BPB,
and 50.107 seconds of timed training. Its source hashes, data/tokenizer
fingerprints, and configuration match the future-credit arm except objective
and run identifier.

`ffn_future_credit_sp1024_2k` was externally cancelled by request, not culled
by the trainer. Its last logged update was 670; its last validation was
**2.485239 proxy BPB at step 660**. `checkpoint.pt` preserves step 600 and
`best.pt` preserves step 660. There is no completed 2,000-update future-credit
score, so this experiment is **not eligible for promotion or a final matched
BPB comparison**. `result.json` and `comparison.json` explicitly record the
interruption rather than treating the last validation as a final score.

Future-credit wall-clock timing was contaminated by concurrent, untracked PGQA
GPU training; it is not an isolated throughput benchmark. The implementation
passed 26 CPU document-stream tests and 15 CUDA gradient/runtime contracts,
including compiled optimizer-page parity over three updates. The full-checkpoint
generation and resumed-best-publication checks were skipped because their
required training job was cancelled; those end-to-end checks remain unverified.

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

Predicted-carry NextLat was 0.457747 BPB worse than CE. Switching NextLat to
actual carry improved BPB by 0.201183, but remained 0.256564 worse than CE.
These are negative ablations, not promoted recipes. Future credit is a separate
ablation; the historical results do not establish its effectiveness.

Scores cover a fixed 32,768-target partial-document panel (76,307 bytes), not
full challenge validation or the old nanoGPT baseline window. Reset-state BPB
is a counterfactual using zero carry at every token, not a separately trained
model. Canonical metrics, run provenance, results, and checkpoints live under
`ablation_results/<run>/`; the same metrics stream feeds `tb_logs/<run>/`.
Current training records CE, scalar TD regression, predecessor values and targets;
timed training excludes validation and checkpoint writes.

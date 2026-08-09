# ToaST + TST tokenizer

A tokenizer that combines two ideas from recent papers into one artifact:

- **ToaST** (Tokenization with Split Trees, Schmidt et al., arXiv 2605.22705v1)
  builds a vocabulary-independent binary split tree per pretoken from byte
  n-gram counts, then chooses the vocabulary by solving an integer program
  over those trees rather than by greedy merging. Inference is a
  recursive descent that emits the first in-vocabulary node.
- **TST** (A Triadic Suffix Tokenization Scheme for Numerical Reasoning,
  Chetverina, arXiv 2604.11582v3) gives digits their own token space, grouped
  with explicit magnitude suffixes, so a number's place value is visible in
  its token identity instead of being an accident of merge frequency.

The two compose cleanly because they claim disjoint parts of the input: TST
takes every numeric span before pre-tokenization runs, so no digit fragment
ever competes for text vocabulary budget, and the split-tree integer program
optimizes only what is left.

## Measured result

The current 50,257-token artifact was trained on a 189.8 MB sample from the
math-30 source mixture and evaluated on a held-out 4.0 MB slice of that same
mixture:

| tokenizer | vocab | bytes/token | round-trip failures |
| --- | ---: | ---: | ---: |
| GPT-2 (current) | 50,257 | 3.4673 | — |
| ToaST + TST | 50,257 | **3.7288** | 0 |

At a matched 50,257-token budget that is **7.54% fewer tokens for the same
text**. An earlier exploratory artifact trained on a different 95.3 MB sample
reached 4.0001 bytes/token, so that number is not used as evidence for the
current math-30 artifact. The 16,384-vocabulary design remains a separate arm:
it reallocates embedding parameters to the trunk and must be trained and
evaluated on this same sample before making a current comparison.

The current 50,257-token LP was exactly integral: zero fractional variables,
zero relative gap, and identical relaxed and rounded objectives (41,865,188).

Bits-per-byte is the comparison metric for any model trained on this, and it
is tokenizer-independent by construction, so BPB numbers stay comparable to
every existing run in `ablation_results/`.

## Layout

| file | contents |
| --- | --- |
| `tst.py` | numeric scheme: digit grouping, magnitude suffixes, exact round trip |
| `split_tree.py` | n-gram counting, split-tree construction, recursive-descent inference |
| `vocab_lp.py` | the vocabulary integer program, its LP relaxation, and §4.3 rounding |
| `spec.py` | the serialized tokenizer: id layout, pretoken regex, provenance, hashing |
| `ngram_store.py` | the count dictionary the encoder needs at inference |
| `tokenizer.py` | encode and decode |
| `train.py` | the training pipeline and CLI |

`scripts/sample_tokenizer_corpus.py` draws the domain-weighted text sample.

## Two deliberate deviations from the papers

**Numbers stay lossless.** TST as published pads digit groups with leading
zeros, which makes `0.1`, `0.10`, and `0.100` collide. This implementation
uses variable-length boundary groups instead, so the mapping is a bijection
and `decode(encode(text)) == text` holds byte for byte on arbitrary input.
The padded form is still reachable (`leading_zero_padding=True`) for
comparison against the paper, and a test pins its lossy behaviour so the
difference stays visible rather than forgotten.

**Token identity is `(digits, power)`.** A numeric token is the pair of its
digit string and the base-ten exponent of its least significant digit, so its
value is exactly `int(digits) * 10**power`. This is what makes the scheme's
place-value claim mechanical rather than notational.

Group size N is a parameter. N=1 costs 340 numeric tokens, N=2 costs 1,780,
N=3 costs 12,220. The paper hypothesizes N=1 is optimal for small models and
explicitly defers the experiment; the ablation below is that experiment.

## Training a tokenizer

Sampling and training are pure CPU, but they still go through `mlq` under the
repository's single-workload queue policy.

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
```

`--baseline-gpt2` encodes the same validation text with GPT-2 and records
`compression_ratio` in `validation.json`. Bytes per token means nothing on its
own; it means something against the tokenizer it would replace, on the same
text.

The sampler applies the problem-registry guard, so the tokenizer is not fitted
to held-out problems. Vocabulary fitted to an evaluation set is a quieter leak
than training on it and just as real: it buys the model shorter encodings of
exactly the text it will be judged on.

Cost scales with `--max-trees`: 20,000 trees solve in 34 seconds, 150,000 in
15 minutes, almost all of it inside HiGHS. Coverage of pretoken occurrences
goes from 90.7% to 98.2% over that range, which is worth the wait for a
tokenizer that will be used for every subsequent run.

`--vocab-size` fails closed if the numeric scheme and the byte alphabet do not
both fit, and if the requested size exceeds the number of candidate tokens the
trees actually produce. Both are configuration errors, not conditions to
degrade around.

## Using one to build a corpus

`scripts/build_k3_pretrain_dataset*.py --tokenizer <directory>` builds the
pretraining stream under a trained tokenizer instead of GPT-2. Everything that
assumed GPT-2 ids follows the resolved one: document boundaries use the
tokenizer's own end-of-text id, and the pre-tokenized FineWeb shards -- which
are stored under GPT-2 ids -- are decoded and re-encoded rather than passed
through, so no part of the corpus is left in the old vocabulary.

The dataset manifest records `tokenizer_provenance` (kind, name, vocabulary
size, end-of-text id, directory, spec hash, n-gram hash).
`scripts/run_k3_context_curriculum.py` reads the vocabulary size from there and
pads it to a multiple of 128 for the trainer, so no flag has to be kept in sync
by hand. A checkpointed build's cache is bound to its tokenizer and refuses to
resume under another.

Two hashes, not one, because the split trees are rebuilt from the counts at
encode time: the counts decide the encoding as much as the vocabulary does.
`from_directory` verifies `ngrams.bin` against the digest in `tokenizer.json`
before loading it, so a directory whose counts were swapped underneath its spec
fails loudly instead of silently reinterpreting every shard. Anywhere two
corpora have to share one embedding table -- `--cooldown-data` against
`--data` -- they are compared on that content identity rather than on padded
vocabulary size, which is a 128-wide bucket that GPT-2 and a 50,257-token
trained tokenizer both land in.

Bits per byte is what makes an ablation across tokenizers readable, and its
byte denominator does not survive a tokenizer swap by itself: a TST token
carries `(digits, power)` and renders to bytes that depend on its neighbours,
so no per-token byte table exists for it. `pretraining/byte_accounting.py`
keeps the GPT-2 lookup table for GPT-2 corpora and decodes runs of tokens for
everything else, charging registered specials one byte each under both.

`tokenization.tokenizer.BatchEncoder` is the adapter that makes this one flag
rather than a second code path: it presents `SplitTreeNumericTokenizer` behind
the same list-in, list-out signature the builders already used, and encodes
corpus text with `allow_specials=False` so a document that happens to contain
`<|endoftext|>` stays text.

## What has and has not been established

Established, by direct measurement: compression on held-out text, exact
invertibility on that text and on the round-trip corpus in
`tests/test_tokenization.py`, the LP formulation's agreement with brute force
on a small instance, and the paper's no-cascade property (removing one token
from the vocabulary expands only the node that used it).

Not established: that any of this improves the model. Compression is not
capability, and a tokenizer that packs more bytes per token also gives the
model fewer steps of computation per byte. Nothing here should be adopted on
the strength of the bytes-per-token table alone.

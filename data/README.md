# Data Workflows

This directory contains the dataset download helpers and export scripts used for the challenge.

Canonical local layout:
- `data/datasets/<dataset_name>/`
- `data/tokenizers/`
- `data/manifest.json`
- `data/docs_selected.jsonl`
- `data/docs_selected.source_manifest.json`

## Downloading Published Data

Preferred entrypoint:

```bash
./data/scripts/get-datasets.sh
```

That wraps the Python downloader and fetches the baseline `sp1024` FineWeb
export (80 train shards ≈ 8B tokens + full val). Extra flags are forwarded:

```bash
./data/scripts/get-datasets.sh --variant sp1024 --train-shards 180
VARIANT=sp8192 TRAIN_SHARDS=40 ./data/scripts/get-datasets.sh
./data/scripts/get-datasets.sh --with-docs
```

Equivalent direct call:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This populates `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`.
By default it downloads the full validation split and 8B training tokens (80 train shards).

To fetch more training shards, pass `--train-shards`:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 180
```

The downloader is manifest-driven and can fetch only a prefix of train shards from a larger published export. With the current shard size of `100_000_000` tokens, `10B` retokenized training tokens is `100` train shards:

```bash
MATCHED_FINEWEB_REPO_ID=your-hf-username/your-dataset-repo \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=your_50B_export_root \
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 100
```

Validation is always downloaded in full from the fixed `fineweb_val_*` split. Training on the first `N` train shards means training on the prefix of the same frozen shuffled export, so the data order stays aligned with the baseline for that tokenizer family.

The default published repo is `willdepueoai/parameter-golf`, with the export rooted under the repo subdirectory `datasets/`.

## Rebuilding Tokenizers From Published Docs

To retrain a tokenizer or re-export shards from exactly the same selected documents, run the standalone retokenizer against the published docs cache:

```bash
python3 data/download_hf_docs_and_tokenize.py \
  --repo-id your-hf-username/your-dataset-repo \
  --remote-root your_50B_export_root \
  --output-root /tmp/my_custom_tokenizer_export \
  --tokenizer-config ./data/tokenizer_specs.json
```

The sidecar `docs_selected.source_manifest.json` includes `docs_sha256`, so users can verify they are rebuilding from the exact same document list and order as the baseline export.

## Useful Knobs

For CPU-heavy exports, useful knobs are:

```bash
MATCHED_FINEWEB_SP_BATCH_SIZE=2048
MATCHED_FINEWEB_TOKENIZER_THREADS=16
MATCHED_FINEWEB_TIKTOKEN_THREADS=16
MATCHED_FINEWEB_GPT2_DECODE_BATCH_SIZE=512
```

These control batched tokenizer encoding during shard export, tokenizer thread count, tiktoken thread count, and batched GPT-2 decode for the blobstore docs-cache path.

## MiniCPM5 English Source Pool

`data/pretraining_sources/minicpm5_refresh/` contains a pinned, verified
39.81 GB compressed Parquet source pool: 24 Ultra-FineWeb English shards
(31.19 GB), four Ultra-FineWeb-L3 English Q&A shards (4.31 GB), and four
English multi-style shards (4.31 GB). Together they contain 18,402,960 rows.
Shard selection is deterministic across each pinned English subset, not a
contiguous prefix. These are acquisition proportions, not validated training
mixture weights.

- `source_manifest.json`: repository revisions, file lists, byte sizes,
  upstream SHA-256 hashes, and selection policy; uses the existing source
  downloader format and the `content` text column.
- `verification.json`: full-file checksum results, Parquet schemas, row
  counts, and allocated storage accounting.
- `cleanup.json`: retired payload inventory and the measured 40.25 GB
  replacement budget. The final pool, retained provenance, and metadata
  occupy 39.92 GB, below that budget.
- `retired_provenance/`: K3 v7 preprocessing-cache metadata and validation
  snapshots; the completed K3 v7 training corpus remains in place.

Obsolete mathmix v1–v3, dense-v4 diffusion, Bolmo byte1024, MathGLM v1–v5,
and drill v2/v3 training payloads were retired. Their retained manifests,
validation, and probe files are historical records, not complete training
datasets. Failed builds and the completed K3 v7 intermediate cache were
removed; current trainer defaults and canonical source datasets were retained.

The new pool is **source data only**. Apply challenge-heldout decontamination
and existing mixture ablations before materializing or using it for training.
No training configuration was switched to these sources. Check upstream
source licenses; the L3 dataset card also prohibits unauthorized unchanged
redistribution.

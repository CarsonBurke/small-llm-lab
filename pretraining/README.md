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

If corpus preprocessing is interrupted, submit the same build command again.
Verified source caches are reused; only the source that was incomplete at the
time of failure is restarted. Final loader-shard assembly is a cheap
sequential pass over those caches.

The runner streams all three stages into
`ablation_results/k3_quality_20k/metrics.jsonl` and
`tb_logs/k3_quality_20k`, so TensorBoard shows a single continuous 0–20,000
step curve. Corpus download and construction do not emit TensorBoard metrics.

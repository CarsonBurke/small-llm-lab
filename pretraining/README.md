# K3-inspired pretraining campaign

This campaign translates the text-side parts of Kimi K3 to the measured
`KKKDKKKD` nano backbone. It is inspired by K3, not a reproduction: the K3
report names Web, Code, Mathematics, and Knowledge as its text domains but
does not publish source weights. Our 60/15/15/10 domain hypothesis is chosen
as the broad-capability arm. The 80/10/7/3 `web80` profile is the
FineWeb-preserving arm. Both must pass the 2,000-step gate; no mixture is
selected from intuition alone.

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
  python3 prepare_k3_pretrain_sources.py

mlq submit --name k3mix_v5_gpt2_2k --cwd "$PWD" --max-parallel-runs 1 -- \
  python3 build_k3_pretrain_dataset.py \
    --output data/datasets/k3mix_v5_gpt2_2k \
    --training-steps 2000

mlq submit --name k3mix_v5_web80_gpt2_2k --cwd "$PWD" \
  --max-parallel-runs 1 -- \
  python3 build_k3_pretrain_dataset.py \
    --weights pretraining/k3_weights_web80.json \
    --output data/datasets/k3mix_v5_web80_gpt2_2k \
    --training-steps 2000 \
    --validation-permille 50
```

The generated token shards are for local research. They combine data with
different attribution and redistribution terms; do not redistribute them
without reviewing the source-level licenses and underlying-content rights
recorded in the source manifest.

The 2K gates use the sampled source set above. Before an 8K campaign, download
and build with `--full-source-set`; this expands the large remote corpora from
13 sampled files to 31 revision-pinned files, in addition to all 21 SciCode
shards, so the longer run does not just consume more tokens from the same
narrow slices.

## Gated ablations

Every change is evaluated for 2,000 steps before combination:

1. unchanged 3-head KDA8 recipe on corpus v5;
2. full-width KDA (4 x 128 heads);
3. K3 full-rank KDA output gate;
4. per-head Muon for Q/K/V;
5. faithful NoPE Gated MLA in the two global layers;
6. one training-only future-token prediction head, removed at export;
7. independently retuned cosine + 1% warmup and weight decay.

FineWeb BPB remains the keep/discard gate. The four domain BPBs diagnose
whether a FineWeb regression buys real breadth; they do not replace the
challenge metric.

Prior K3 components are not being revived without evidence. SiTU-GLU finished
at 1.1745 BPB versus the 1.1734 dense reference, below the 0.005 keep
threshold. The AttnRes port was stopped at step 300 after reaching 1.5150 BPB
at step 200 versus 1.4183 for the reference. Earlier GDN2 variants were also
slower and worse than KDA. The new campaign therefore isolates only the K3
changes that have not yet received a valid 2,000-step test.

## Longer pretraining

Only the winning 2,000-step combination scales to 8,000 steps. Build the
4.19B-token corpus with `--training-steps 8000`, then run the curriculum
inside mlq:

```bash
mlq submit --name k3_v5_curriculum --cwd "$PWD" --max-parallel-runs 1 -- \
  /home/marvin/Documents/repositories/parameter-golf/.venv/bin/python \
  run_k3_context_curriculum.py \
    --data data/datasets/k3mix_v5_gpt2_8k \
    --run-id k3_v5_winner \
    --kda-heads 4
```

The curriculum keeps the 524,288-token global batch and 32,768-token
microbatches fixed while moving from 2K to 4K to 8K context. Stage boundaries
resume model, optimizer, RNG, and exact corpus position. Exact staged resume
is intentionally single-GPU only because Muon optimizer state is rank-sharded.
An optional `--cooldown-data` corpus can replace the broad mix only for the
last 500 steps, matching K3's high-quality cooldown idea; it is not enabled
unless a continuation ablation shows a BPB win.

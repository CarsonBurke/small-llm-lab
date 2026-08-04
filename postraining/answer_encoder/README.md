# LeJEPA answer encoder

This experiment tests whether a LeJEPA-trained text representation can turn a
reference answer into a dense scalar reward without a learned verifier. The
deployed reward is `exp((cosine - 1) / temperature)`, with temperature `0.1`
by default. This is an RBF kernel over normalized answer embeddings: exact
agreement scores one and increasingly distant representations approach zero.
The transform sharpens useful margins but cannot repair an encoder that ranks
an incorrect answer above a correct one.

For each underlying answer or random pretraining segment, the data pipeline
creates configurable token views. Tokens are the analogue of ViT patches: one
learned CLS token attends bidirectionally to the complete, dynamically padded
sequence, and its final normalized state is the answer embedding. There is no
token or chunk mean pooling. Sinusoidal positions permit inference beyond the
training length while retaining token order. Training uses the same two terms
as the minimal LeJEPA recipe:

```text
center = mean(projected view embeddings)
invariance = mean squared error(view, center)
loss = (1 - lambda) * invariance + lambda * SIGReg(views)
```

There are no correctness labels, explicit negatives, EMA targets, or
stop-gradient branches. The experiment mixes random GPT-2-tokenized K3
pretraining windows with verified DAPO answers and raw code. Broad pretraining
records are deliberately dominant; exact answer strings are deduplicated and
split into train and held-out validation sets. It does not
modify VAPO's production reward: cosine reward must pass the behavioral gate
before an RL integration ablation is justified.

## Training

All model workloads must run through `mlq`:

```bash
mlq submit \
  --name answer_lejepa_multicrop_v5_2k \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  python3 scripts/train_answer_encoder.py train \
    --name answer_lejepa_multicrop_v5_2k \
    --steps 2000
```

The first run, `answer_lejepa_v1_2k`, failed the behavioral gate and exposed
two important reference mismatches. LeJEPA uses its BatchNorm projection head
only for the training loss and evaluates the backbone representation. It also
regularizes substantially larger batches than v1's 32 samples. The corrected
reference-regime ablation is:

```bash
mlq submit \
  --name answer_lejepa_reference_2k \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  python3 scripts/train_answer_encoder.py train \
    --name answer_lejepa_reference_2k \
    --steps 2000 \
    --batch-size 256 \
    --projection-hidden-dim 2048 \
    --projection-dim 16 \
    --reference-projector \
    --global-views 4 \
    --local-views 0 \
    --global-scale-min 1.0 \
    --global-scale-max 1.0 \
    --mask-probability 0.05 \
    --dropout 0.0 \
    --learning-rate 0.002 \
    --reward-space backbone
```

That second run also failed, and a reference audit found that it still used
mostly duplicated one-token answers, identity-like 5% masking, arbitrary
chunk-mean pooling, identically initialized Transformer layers, a mismatched
validation distribution, and a nonreference learning-rate floor. The v3
run corrected those confounds with direct CLS pooling,
independent block initialization, 30% masking, a 20% deduplicated answer
mixture, held-out mixture validation, gradient scaling, and the minimal
reference's `1e-3` cosine floor.

Job 1209 (`answer_lejepa_cls_v3_2k`) completed the full ablation. SIGReg was
healthy: held-out projection effective rank reached 14.92/16 and validation
loss reached 0.0333. The direct backbone embedding nevertheless failed every
semantic gate. Its effective rank was 5.05/256, reordered functions scored
below a semantic bug by 0.000010 cosine, numeric equivalents scored below
non-equivalents by 0.00487, paraphrases scored below negations by 0.000436,
and taxonomy-related text scored below unrelated text by 0.000119. This is a
negative result for raw cosine correctness reward from same-document masking;
it rules out projected-space SIGReg collapse, but not an insufficient
reward-space objective or missing semantic augmentations. In particular, v3
used no block-shuffle views, so explicit function-order invariance remains a
separate ablation. The projected-space probe is retained as a diagnostic, not
as a deployable reward claim.

The next isolated ablation applies the unchanged loss directly to the CLS
backbone, keeping the seed, corpus, masking, and lack of block shuffling fixed:

```bash
mlq submit \
  --name answer_lejepa_cls_direct_sigreg_v4_2k \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  python3 scripts/train_answer_encoder.py train \
    --name answer_lejepa_cls_direct_sigreg_v4_2k \
    --steps 2000 \
    --objective-space backbone \
    --reward-space backbone
```

Job 1212 completed this test. Direct regularization raised held-out backbone
effective rank from 5.05 to 23.62/256, confirming that the nonlinear projector
had absorbed much of SIGReg. The resulting cosine reward was still unusable:
only the taxonomy check passed. Numeric equivalence margin was -0.00745,
paraphrase-minus-negation was -0.00527, and function-reorder-minus-bug was
-0.000504. The projected-space diagnostic for v3 likewise failed all four
checks. Block-shuffle views remain a separate function-order experiment, but
cannot explain the numeric or negation failures.

V3/v4 did not actually exercise the text analogue of LeJEPA's multi-crop
training: they used four complete sequences, no local views, and only token
masking. The v5 defaults fix that single central mismatch while restoring the
projector as the deployed embedding on both sides. Each source produces two
global crops covering 30-100% of its tokens and six local crops covering
5-30%, plus 10% masking. A static 256-sample FineWeb audit produced eight
unique views for 255/256 sources; global crops averaged about 93 tokens and
local crops about 26. Because this design deploys the projector rather than
discarding it, its hidden normalization is per-example LayerNorm instead of
reference BatchNorm; this avoids applying crop-batch running statistics to
single full-answer inference. The projected objective, corpus, model, batch,
optimizer, and seed otherwise remain fixed.

Job 1215 (`answer_lejepa_multicrop_v5_2k`) completed all 2,000 steps and
failed the behavioral gate. Unlike the masked-copy runs, it learned nontrivial
view geometry: held-out view-center cosine was 0.788, projected effective rank
was 11.76/16, and taxonomy-related text beat unrelated text by 0.05882 cosine.
The remaining relations were wrong. Numeric equivalence margin was -0.38541,
paraphrase-minus-negation was -0.29811, and function-reorder-minus-bug was
-0.00683. In particular, `10.0` clustered with `10.1` near 0.614 cosine while
the integer strings `50` and `100` remained above 0.999 against `10`. The
multi-crop correction therefore removed the identity-view shortcut, but this
2,000-step projected representation is not a usable correctness reward.

V6 replaces crop-level CLS agreement with aligned masked latent prediction,
without adding token cross-entropy, an EMA teacher, stop-gradient, or a new
tokenizer. One shared bidirectional encoder processes the complete sequence
and a same-length context whose contiguous target spans are replaced by
`[MASK]`. A training-only MLP predicts the complete-view hidden state at every
masked token and at CLS; predicted and complete states then pass through the
same attached projector. The exact two-view LeJEPA center-MSE scale and SIGReg
are averaged across separate token-patch and CLS populations. The final
checkpoint discards the predictor and retains the same projected CLS reward
API as v5.

```bash
mlq submit \
  --name answer_lejepa_masked_latent_v6_2k \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  python3 scripts/train_answer_encoder.py train \
    --name answer_lejepa_masked_latent_v6_2k \
    --steps 2000 \
    --training-objective masked-latent \
    --span-mask-probability 0.3 \
    --mean-mask-span-length 3.0 \
    --predictor-hidden-dim 1024
```

Job 1219 completed in 190 seconds. The missing-patch task learned cleanly:
held-out target and predicted patch effective ranks were 15.93/16 and
15.92/16, patch prediction cosine was 0.99034, projected CLS rank was
15.50/16, and global prediction cosine was 0.99860. This rules out collapse or
a failed predictor. It still failed three semantic checks. Relative to v5,
numeric equivalence margin improved from -0.38541 to -0.11033,
paraphrase-minus-negation from -0.29811 to -0.01179, and function-reorder
minus bug from -0.00683 to -0.000018; all remain on the wrong side. Taxonomy
passed by +0.00182. In particular, `10` remained closer to `9`, `50`, and
`100` (all about 0.998 cosine) than to `10.0` (0.888), while a paraphrase
scored 0.9819 against 0.9937 for its negation. Masked latent prediction is a
substantial improvement over multicrop geometry but not a deployable
correctness reward after 2,000 steps. Do not integrate v6 into RL.

V7 removes the remaining length-preserving shortcut. Missing spans are
deleted from the context rather than replaced by `[MASK]`; every surviving
token keeps its original zero-based position. A two-layer bidirectional-query
Transformer predictor cross-attends to the compacted context. Learned patch
queries receive the original positions of the missing targets, while a
learned global query predicts the complete-view CLS. Only valid queries enter
the shared attached projector, center MSE, and SIGReg populations. The target
and context encoders remain the same attached encoder, and the final reward
checkpoint still omits the training-only predictor.

```bash
mlq submit \
  --name answer_lejepa_variable_cardinality_v7_2k \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  python3 scripts/train_answer_encoder.py train \
    --name answer_lejepa_variable_cardinality_v7_2k \
    --steps 2000 \
    --training-objective masked-latent \
    --span-mask-probability 0.3 \
    --mean-mask-span-length 3.0 \
    --predictor-hidden-dim 1024 \
    --predictor-layers 2
```

Job 1221 completed all 2,000 steps in 196 seconds. The different-cardinality
prediction problem trained without collapse: held-out target and predicted
patch effective ranks were 15.91/16 and 15.92/16, patch cosine was 0.99039,
projected CLS rank was 15.30/16, and global prediction cosine was 0.96451.
The lower global cosine than v6's 0.99860 confirms that deleting patches made
complete-state prediction materially harder rather than reproducing the
same-length shortcut.

The reward gate nevertheless failed all four checks. Numeric equivalence
improved from v6's -0.11033 to -0.05279, but `10` was still closer to `9`,
`50`, and `100` (all about 0.999 cosine) than to `10.0` (0.94599).
Paraphrase-minus-negation was -0.01401, function-reorder-minus-bug was
-0.000013, and taxonomy-related-minus-unrelated was -0.00226. V7 is a more
faithful variable-cardinality JEPA and improves the numeric failure, but its
projected CLS is still not a usable correctness reward. Do not integrate v7
into RL.

Job 1223 tested whether v7's projected contextual token patches contained
useful local semantics that CLS pooling had erased. It compared patch sets
with exact one-to-one maximum-cosine matching, reporting both a matched-only
score and a cardinality-aware score whose unmatched patches receive zero
reward. The matched-only readout improved numeric equivalence from -0.05279
to -0.02507 and restored the taxonomy check at +0.00400, but paraphrase versus
negation remained wrong at -0.01083. Most importantly, function reorder versus
semantic bug worsened from the CLS margin of -0.000013 to -0.000068: the bug
still matched more closely. Cardinality-aware matching was dominated by GPT-2
token-count differences and made numeric and paraphrase behavior much worse.
The current patch geometry therefore does not contain the missing localized
program-semantic signal; replacing CLS with token-set matching is rejected.

V8 makes the deployed representation itself the only latent prediction
target. The complete answer is encoded into one projected 256-dimensional CLS
vector. A compact context is formed by deleting contiguous spans while keeping
the surviving tokens at their original positions; a training-only two-layer
Transformer predictor cross-attends those contextual states from one learned
global query. The predicted and complete-answer CLS states pass through the
same attached per-example projector. Exact two-view LeJEPA center MSE and
SIGReg are computed only over those two CLS populations: there are no patch
queries, patch losses, or patch rank metrics. The final checkpoint contains the
encoder/projector but omits the predictor.

The v8 projection default is 256 dimensions; historical multicrop runs retain
their 16-dimensional default. An explicit `--projection-dim` overrides either.

```bash
mlq submit \
  --name answer_lejepa_global_cls_v8_2k \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  python3 scripts/train_answer_encoder.py train \
    --name answer_lejepa_global_cls_v8_2k \
    --steps 2000 \
    --training-objective global-latent \
    --span-mask-probability 0.3 \
    --mean-mask-span-length 3.0 \
    --predictor-hidden-dim 1024 \
    --predictor-layers 2
```

Job 1229 completed the full 2,000-step v8 ablation. The 256-dimensional
projected CLS did not collapse completely, but its effective rank was only
8.55/256 (backbone rank 3.45/256). Global prediction cosine was 0.95557 and
validation loss was 0.22075. The behavioral gate failed 0/4: numeric
equivalence was -0.04306, taxonomy was -0.01053, paraphrase versus negation
was -0.00863, and function reorder versus bug was -0.000087. V8 modestly
improves v7's numeric and negation margins, but worsens taxonomy and the
function-bug distinction. A larger CLS alone therefore does not fix the
same-text JEPA geometry; do not integrate this checkpoint into RL.

For a pretrained upper baseline, job 1228 evaluated Qwen3-Embedding-8B over
the same probes plus systematic numerical errors and graded math/code cases.
Plain symmetric 256-D cosine passed the four small behavioral gates, with
margins +0.02593 numeric, +0.16650 taxonomy, +0.13591 negation, and +0.04981
function behavior. It achieved 0.7808 mean Spearman and 0.8654 pairwise
ordering accuracy over the five curated semantic/code/math cases. It is not a
correctness verifier: an answer-only correct `5` scored 0.36450 while a
verbose arithmetic slip scored 0.88509, and an off-by-one factorial scored
0.98139 versus 0.83686 for an equivalent library implementation. The
supported 256-D Qwen embedding is a strong semantic shaping baseline, but raw
cosine is not a safe standalone correctness reward.

Canonical metrics, the manifest, final probe report, and checkpoint are
written under `ablation_results/<name>/`; TensorBoard events go under
`tb_logs/<name>/`. Every evaluation also writes a resumable
`training_checkpoint.pt` containing optimizer, RNG, and corpus traversal
state.

## Behavioral gate

The frozen probe reports raw cosine and exponentially scaled `[0, 1]` reward for:

- `10` against `10`, `10.0`, `ten`, `9`, `10.1`, `50`, and `100`;
- `cat` against `kitten`, `dog`, and an unrelated noun;
- paraphrase versus inserted negation and reversed comparison;
- two independent functions in either order versus a one-operator bug.

Run it in the deployed projection space shared by target and answer:

```bash
python3 scripts/train_answer_encoder.py probe \
  --checkpoint ablation_results/answer_lejepa_multicrop_v5_2k/answer_encoder.pt \
  --reward-space projection
```

The current projector is deterministic per example for both pre-encoded
targets and candidate answers. To quantify train/eval BatchNorm mismatch on a
historical checkpoint, run the queued diagnostic through `mlq`:

```bash
mlq submit \
  --name answer_lejepa_diagnose \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  python3 scripts/train_answer_encoder.py diagnose \
    --checkpoint ablation_results/answer_lejepa_v1_2k/answer_encoder.pt \
    --output ablation_results/answer_lejepa_v1_2k/diagnostic.json
```

The primary gate is relative geometry rather than the representation loss:
equivalent function reordering should outrank the semantic bug, paraphrase
should outrank negation, and exact `10` should outrank every non-equivalent
numeric candidate. These conditions produce an explicit pass/fail result. A
failed checkpoint may be probed, but scoring and target caching reject it
unless `--allow-failed-gate` is supplied for diagnosis. No absolute cosine
threshold is assumed in v1. Authorization is bound to the embedding space
that passed the gate; a backbone pass cannot authorize projection scoring.

## Reward and pre-encoded targets

```bash
python3 scripts/train_answer_encoder.py score \
  --checkpoint ablation_results/passing_run/answer_encoder.pt \
  --target "10" \
  --answer "10" \
  --answer "9"

python3 scripts/train_answer_encoder.py cache-targets \
  --checkpoint ablation_results/passing_run/answer_encoder.pt \
  --target problem-1=10 \
  --output /tmp/answer-targets.pt
```

Target caches are bound to the checkpoint SHA-256 and embedding space, so a
stale target cannot silently be scored by a different encoder.

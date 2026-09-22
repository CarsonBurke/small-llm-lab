# Exact shared-state KDA routing

Experiment requested against `k3mix_v5_kda8_ref_2k`: retain the eight-block
`KKKDKKKD` backbone and its dense attention, but replace six privately owned
KDA memory banks with twelve globally shared banks. Each bank contains three
128-by-128 head matrices. Each KDA site selects exactly one bank per token,
updates that bank, and reads its updated state. Unselected banks are unchanged.

This borrows softmax/top-k routing from
[LatentMoE](https://arxiv.org/pdf/2601.18089v1), adapted to top-1 memory indices.
It does not implement the paper's latent FFN experts or claim that its results
establish this recurrent architecture's quality.

## Semantics

- Execution is strictly token-major, then layer-major within each token.
  Earlier layers see later-layer writes from previous tokens, and later
  layers see earlier-layer writes from the current token.
- KDA projections, decay/write controls, output gates, and short-convolution
  histories remain layer-local. Only the matrix memories are shared.
- Routers receive the layer's normalized hidden activation. The selected
  softmax probability scales the mixer output without top-1 renormalization;
  language loss therefore differentiates the routing scores. The discrete
  index itself is not differentiable, as in ordinary hard-routed MoE.
- A [Switch-style](https://arxiv.org/abs/2101.03961) balancing loss with coefficient 0.01 operates over all
  routing events in each checkpoint segment. Routers use AdamW at 0.001,
  without weight decay; inherited parameter groups are unchanged.
- All memories and dense caches are differentiable, including across the
  16-token activation-checkpoint boundaries. Those boundaries introduce no
  delay, detach, or truncated backpropagation.
- Dense attention retains its projections, RoPE, RMS normalization, SDPA
  scale, MLP, and causal visibility. Its token-step execution uses functional
  KV caches instead of a full-sequence call.

Exact sharing introduces a dependency from late layers at token t to early
layers at token t+1. The ordinary layer-wise FLA parallel scan cannot directly
evaluate that dependency. Token transitions and batched vocabulary losses
are compiled with full-graph capture; execution is not claimed to be an
optimized fused recurrent training kernel or CUDA-graph replay.

## Run contract

```bash
mlq submit --name kda_state12_contracts --cwd "$PWD" \
  --max-parallel-runs 1 --priority 1 --time-limit 10m -- \
  .venv/bin/python -m pytest -x -vv -o faulthandler_timeout=90 \
  pretraining/tests/test_kda_state_routing_gpu.py

mlq submit --name k3mix_v5_kda8_state12_exact_10m --cwd "$PWD" \
  --max-parallel-runs 1 --priority 1 --time-limit 10m \
  --after-success <contract-job-id> -- \
  .venv/bin/python -u scripts/train_kda_state_routing.py \
  --name k3mix_v5_kda8_state12_exact_10m --steps 1000 --seconds 600
```

The runner takes the reference's dataset and overrides, preserving a
524,288-token global batch, 1,024-token sequences, and 32-sequence
microbatches. Initialization, dense classes, the data loader, and Muon remain
identical to the reference source; static tests enforce this. The inherited
trainer uses a 1,000-update learning-rate schedule for this capped experiment,
not the historical 2,000-update schedule. A comparison past their common
constant-rate prefix would need a schedule-matched baseline.

The 600-second queue deadline includes process startup and compilation. An
internal deadline at 570 seconds reserves time to save the last completed
update; partial accumulation is discarded. The full reference validation
panel and 20-update cadence are retained, but initial untrained validation
is skipped. If no validation completes in the budget, no BPB claim is made.

Canonical metrics, result metadata, source snapshots, and the routed
checkpoint live in `ablation_results/<name>/`; TensorBoard lives in
`tb_logs/<name>/`. Microbatch losses include balancing and are diagnostics,
not completed optimizer updates or held-out accuracy. The checkpoint carries
an `exact_shared_state_kda` architecture tag. To reconstruct its parameter
tree, instantiate `KDAGPT(**payload['model_config'])` on CUDA, call
`install_shared_state_routing(model, **payload['routing_config'])`, then
strict-load `payload['model']`. Ordinary private-state KDA execution is not
a valid inference substitute for this checkpoint.

This remains a diagnostic GPT-2-vocabulary model, not a 16 MB challenge
submission.

The KDA numerical oracle uses FLA's native Triton backend. The optional
TileLang/TVM FFI extension failed to initialize in the current Python 3.14
environment; the shared-state implementation itself does not use that
extension or FLA's backward kernels.

## Result: exact execution does not meet the training-time objective

The completed run is
[`k3mix_v5_kda8_state12_exact_10m_v3`](../../ablation_results/k3mix_v5_kda8_state12_exact_10m_v3/result.json),
queue job **8780**. The queue recorded about **574.3 seconds** from attempt
start to exit, inside its hard 600-second limit. The runner stopped at its
570-second save deadline and exited successfully after saving.

| Measurement | Result |
|---|---:|
| Completed optimizer updates | 3 |
| Completed microbatches | 54 |
| Microbatches discarded from incomplete update 4 | 6 |
| Tokens represented in completed updates | 1,572,864 |
| Steady update time, mean of updates 2 and 3 | 155.986 s |
| Historical reference update time, updates 21–40 | 1.11135 s |
| Historical timing ratio | approximately 140× slower |
| Peak PyTorch allocated GPU memory | 17,741.5 MiB |
| Parameters, including routers | 64,085,182 |
| Held-out BPB | unavailable: no validation point reached |

The timing comparison uses the reference's recorded history, not a fresh
contemporaneous benchmark. The production shape fits memory and its
gradients are finite, but this exact token-major executor is too slow for a
useful quality test in ten minutes. It copies functional state/cache tensors
and launches many small sequential operations; compilation is not a fused
cross-layer recurrent kernel. This result does **not** demonstrate that the
underlying memory-sharing architecture is good or bad for BPB, and it does
not support a claim of only a few percent quality degradation.

Training losses including the balancing term were 10.8359, 10.5132, and
9.2768 for the three completed updates. All twelve banks were used in the
last completed microbatch, but routing was uneven. These are early learning
diagnostics, not validation or promotion evidence.

Artifacts include canonical `metrics.jsonl`, TensorBoard events, four source
snapshots, and `last_complete_update.pt` containing update-three weights and
both optimizer states. The partial fourth update was never applied. The
experiment is isolated and has not replaced any baseline.

Validation: **13 static/run-protocol tests and 5 GPU contracts passed**
(GPU job 8777). The GPU contracts cover dense forward/backward equivalence,
private-bank reduction to FLA KDA including gradients, causality and delayed
cross-layer credit, single-bank writes and router learning, and full BPTT
across checkpoint boundaries. The checkpoint test uses identical compiled
arithmetic for its tight comparison and also retains an independent
BF16 eager comparison.

Two startup-only attempts are preserved under the same run stem without
`_v3` and with `_v2`: an inherited manifest-schema assumption and an
unregistered execution-module namespace prevented training. Neither applied
an optimizer update. Both were fixed before job 8780; no dataset or dense
attention change was made to work around them.

# 4K context ablation

This experiment is authorized after the exact-execution speed investigation.
No architecture, memory width, forgetting rule, optimizer, or precision changes
are part of the context ablation.
All six temporal blocks remain GDN2; no conventional token-attention blocks are
introduced. Plain mini appears only as the throughput control.

Training keeps 524,288 tokens per optimizer update, 1,000 updates, validation
every 20 updates, seed 1337, summed cross-entropy, and the same ordered shards.
The initial execution shape is 16 sequences of 4,096 tokens per replay and
eight replays per optimizer update. The 64-token kernel chunk is an internal
computation boundary: neither state nor gradients reset there. State resets
between packed sequences, as in the 1K control.

The full-update 4K benchmark compares mini and GDN2 at the same 16x4096 shape,
including optimizers and co-resident training/validation graphs. It is queued as
job 8683 after successful completion of the fresh-autotuning qualification,
job 8682. Job 8682 failed numerical qualification, so 8683 was skipped; the 4K
quality run has not started. Shape-aware throughput reports
also bind the effective FLA/Triton autotuning policy to the subsequent trainer.

## Separating training context from evaluation context

Validation always uses the same first 1,048,576 target tokens and 2,524,883
bytes. Reshaping them into 4K rows removes three quarters of the reset boundaries;
that changes available context even though token and byte budgets match.

The end comparison therefore evaluates both completed checkpoints on both
panels, using a fresh process for each shape:

| Training context | Evaluation context | Purpose |
|---|---|---|
| 1K | 1K | Reproduce the historical control |
| 1K | 4K | Measure context extrapolation without longer-context training |
| 4K | 1K | Measure training effect at the original evaluation context |
| 4K | 4K | Measure the combined result against the same-context control |

`scripts/evaluate_gdn2_context.py` checks checkpoint completion, tokenizer,
canonical byte count, identical flattened panel hashes, compiled/CUDA-graph
loss agreement, and source provenance. It accumulates NLL in FP64 and does not
perform optimizer updates. The trainer explicitly marks its historical 1K LAM
reference as a different-context comparison; that reference cannot promote a
4K run as though the evaluation settings matched.

Longer sequences preserve total tokens per update, so they do not imply four
times the update cost. Fewer independent sequences and a longer recurrent scan
can nevertheless change occupancy and latency. Throughput must be measured.

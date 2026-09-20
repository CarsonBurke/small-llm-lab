# Latent reuse through a parallel encoder and recurrent refiner

This experiment is rejected as a quality improvement. Its parallel variants
train faster than mini but score worse at the matched 1,000-update budget.
Previous-state reuse gains only 0.00110 BPB over current-state processing.
The experiment distinguishes this limited source effect from the consequences
of replacing token attention; it does not disprove deeper latent feedback.

## Evidence motivating the refactor

The matched 1,000-update references are plain nanoGPT-mini at 1.3433 BPB and
first-layer-only LAM at 1.3265. LAM preserves the ordinary transformer and
retrieves previous-pass deep states. That result supports deep-state reuse;
it does not establish that recursively generated final states are necessary.

The scalar delta model reached 1.439372 BPB despite faster training. A narrow
softmax control reached 1.384314, so narrowing alone does not explain the whole
delta-model deficit. These comparisons still do not isolate compression from
the altered gates, normalization, and optimization.

The subsequent exact final-state delta-carry model measured 226,900 tokens/s
at B512 versus 863,891 for its paired mini benchmark. It was cancelled at 264
updates after the user requested a rethink; its latest validation was 1.618608
BPB at update 260. This is not a matched 1,000-update quality result.

## Contract

Four full-width causal transformer blocks compute `u[t]` in parallel. A shared
refiner then computes:

```text
g[t] = sigmoid(gate(u[t]))
z[t] = u[t] + g[t] * source[t]
s[t] = final_norm(z[t] + MLP(refiner_norm(z[t])))
```

The three source arms have identical parameters and initialization:

| Arm | Source after position zero | Question |
| --- | --- | --- |
| `refiner_current` | `u[t]` | What does the added gated transformation achieve? |
| `refiner_encoder` | `u[t-1]` | Does direct access to the preceding contextual representation help? |
| `refiner_refined` | `s[t-1]` | Does reusing the preceding refinement and inherited recurrence add value? |

All sources are zero at position zero. Each sequence starts with fresh state.
There is no age-based eviction/compression rule, stale chunk boundary, detached
carry, or substitution of previous-pass states. The 512-vector recurrent state
is itself a learned summary; ordinary attention separately preserves access to
the prefix. All recurrent segments retain full
temporal gradients. Segment checkpointing is an execution setting: stride zero
disables it, one recomputes every segment, and the current default of two
recomputes alternating segments. Retaining every activation exceeded available
memory at B256. Dense operations use BF16; state arithmetic uses FP32.
All arms use the same summed cross-entropy head with checkpointed 4,096-row
chunks. The full B256 head exceeded capacity when refiner activations were
retained; recomputing the head is the targeted execution tradeoff being tested.

Cached decoding preserves the same source alignment and absolute-position
causal encoder. It errors when cache capacity is exhausted. Decoding speed is
unmeasured.

## Explicit compromises

The recurrent state affects the refiner, not the encoder. This retains parallel
encoder training but prevents the carried state from changing its attention
queries or intermediate computations. The model also has four encoder blocks
instead of mini's six. Those are real architectural changes, not exact
implementations of first-layer feedback.

The initial gate is 0.05. Long chains through the refiner alone are consequently
weak at initialization; the model is not assumed to retain long-lived state.
Ordinary attention supplies access to history independently. A win by the
refined arm over the encoder arm would conflate access to the preceding MLP's
output with inherited recurrence; further ablation would be needed to separate
those effects.

## Qualification and decisions

1. CUDA tests check causal/source contracts, late-loss gradients, compiled
   parity, graph replay, and cached decoding.
2. Benchmark complete 524,288-token optimizer updates at sequence length 1,024:
   five warmups and at least five timed updates, paired with plain mini. Report
   actual memory and preparation cost separately. Arithmetic savings are not
   throughput evidence.
3. Use one common microbatch for any source-quality panel. Train 1,000 total
   updates, seed 1337, validate every 20, and use the canonical 1,048,576-token
   validation window and summed-loss convention.
4. A candidate needs more than 0.005 BPB improvement over first-only LAM, plus
   measured faster pretraining, before promotion. A completed slower benchmark
   does not itself justify a new quality run.

Entry points: `scripts/benchmark_latent_carry.py` and
`scripts/train_latent_carry.py`, selecting one of the three `refiner_*` arms.
All model execution belongs in `mlq` with priority 1 and maximum parallelism 1.
Benchmark source snapshots and hashes bind timing evidence to training code.

## Qualification results, 2026-09-20

The final source/gradient/cached-decoding suite passed 19 GPU tests (job 8604).
The shared split-head change separately passed two GPU tests, and the trainer
protocol suite passed 20 static tests.

| Execution | Training tokens/s | Paired mini | Outcome |
| --- | ---: | ---: | --- |
| Refined, B128, full segment recomputation | 592,613 | 964,189 | Too slow |
| Refined, B256, full segment recomputation | 958,661 | about 962k | No speed gain; before joint-graph capacity qualification |
| Refined, B256, no segment recomputation | — | — | Training preparation exceeded memory |
| Refined, B256, alternate recomputation | — | — | Timed training finished, but co-resident validation exceeded memory, including at validation B64; incomplete benchmark |
| Previous encoder source, B128, bounded head | 1,177,924 | 964,559 | +22.1%; training and validation graphs fit |
| Current encoder source, B128, bounded head | 1,186,288 | about 963k | +23.2%; training and validation graphs fit |

The final two arms each peaked at 13,479 MiB allocated. Their common settings
are training B128, validation B64, T1024, and a checkpointed 4096-row head. Their
matched quality runs are `nanomini_refiner_encoder_1k` (job 8609) and
`nanomini_refiner_current_1k` (job 8610). Exact recursive refinement is not in
this quality panel: it failed combined speed/capacity qualification. This
limits conclusions to reuse of the previous contextual encoder state.

Both quality runs completed 1,000 updates and are rejected:

| Source | Final BPB | Training seconds | Serialized checkpoint bytes |
| --- | ---: | ---: | ---: |
| Current encoder state | 1.3623702099 | 471.367 | 63,056,215 |
| Previous encoder state | 1.3612665863 | 470.454 | 63,056,215 |

Previous-state reuse improves only 0.0011036 BPB over the matched current-state
control, below the 0.005 threshold. It remains 0.0179666 worse than plain mini
and 0.0347666 worse than first-only LAM. The saved checkpoints are unquantized;
neither quality nor 16 MB submission compliance is established.

The encoder-source intervention reproduces training BPB exactly. Zeroing its
gates raises BPB to 1.4044108987 (+0.0431443); mean gate is 0.08416, injected
RMS / encoder RMS is 0.41231, and mean source/current cosine is 0.43659. Thus
the trained model uses the path. This distribution-shift intervention does not
contradict the weak gain over the separately trained current-source model,
and it does not establish the quality of retraining without the path.

The current-source diagnostic gives normal BPB 1.3623702456 (within 3.6e-8 of
training) and zero-source BPB 1.4060307471 (+0.0436605), with mean gate 0.15843
and RMS ratio 0.51545. Both learned gated paths matter to their respective
models. Their similar intervention penalties and tiny between-model quality
gap give no qualified evidence that shifting the source adds value here.
All compared diagnostic source hashes match training.

The defensible conclusion is narrower than "latent carry does not work."
This design retains ordinary attention but moves reuse to one terminal MLP
and reduces the encoder from six blocks to four. It cannot isolate the cause
of the remaining quality deficit. First-only LAM instead exposes deeper
states early enough to influence the whole stack. Preserving that downstream
computation is the better-supported research direction; neither exact
recurrence nor merely preserving the latest vector has been shown sufficient.

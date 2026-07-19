# xlayer — sub-quadratic self-attention over all prior layers

**Guiding idea:** every layer should be able to read token representations from *all*
prior layers — a growing memory of `L×T` tokens — through attention that is (a) fully
sub-quadratic in sequence length and (b) *rewired by gradient descent*, with the routing
staying plastic for the whole run. Where-to-look is a learned, content-dependent decision,
not a fixed pattern and not a heuristic.

## Principles (from ablation evidence + standing direction)

1. **Learnability above all.** No schedules, no EMA, no clamping/capping, no annealing,
   no utility thresholds. If a mechanism needs a knob that decays over time, the design
   is wrong — sharpening must *emerge* from the loss.
2. **Gradients must reach unselected candidates.** Any hard selection whose losers get
   zero gradient will freeze (v2 postmortem below). Every candidate needs a live,
   correctly-signed gradient path every step.
3. **Content-dependent routing.** Content-independent wiring (per-head `(layer, lag)`
   tables) cannot express induction or any content addressing, and empirically the model
   learns to *abandon* such reads (cross-layer mass fell 0.57→0.11 during v2 training).
4. **GEMM-shaped or it doesn't ship.** Sub-quadratic FLOPs mean nothing if the
   implementation is per-query gathers + fp32 atomics: the entmax gather kernels ran
   2.2–2.7 s/step vs the dense baseline's 0.89 s/step on the RTX 5090 *at fewer FLOPs*.
   Every op must map to dense GEMMs / SDPA on tensor cores, static shapes,
   `torch.compile(fullgraph=True)`-clean.
5. **Budget-aware.** 16 MB int8 export; routing parameters must be small (v2 spent
   1.48 MB on fp32 logit tables — never again). New params must be routed correctly
   between Muon (2D) and scalar Adam, and considered for int8 sensitivity.

## Family history

| arm | script | routing | result (5090) | verdict |
|---|---|---|---|---|
| within-layer entmax (reference) | `../sparse_entmax_fast_train_gpt.py` | content-dependent scores, fixed candidate set, same layer only | **1.4258 bpb @ 700**, 2705 ms/step | best bpb-per-step in repo; 3× too slow; proves sparse entmax attention itself helps |
| xlayer churn v0 | `../sparse_xlayer_train_gpt.py` | inherited churn wiring over `(layer', pos')` pool | — | pool too big for churn heuristics |
| persistent graph v1 | `../sparse_persistent_xlayer_train_gpt.py` | per-(layer,head) `(source_layer, lag)` templates, utility-threshold rewiring | 1.4741 @ 700 | heuristic layer self-defeating: new wires arrive gradient-blind |
| learned graph v2 | `../sparse_learned_xlayer_train_gpt.py` | Gumbel-top-K over learned logit tables, straight-through gate + score bias | 1.4928 @ 680, 2243 ms/step | content-independent ceiling; unselected wires starve → topology froze (θ-std 20–37 ≫ noise); model abandoned cross-layer reads |
| **v3 (current)** | `sparse_nsa_xlayer_train_gpt.py` | see below | — | in progress |

Dense baseline to beat: `pope_zero_fast_nogain` — 894 ms/step, 1.4565 @ 700, **1.3447 @ 2000**.
Gate: > 0.005 bpb better at step 2000, at ≤ ~1.2× baseline step time.

## v2 postmortem (what v3 must fix)

1. **Content independence** was the expressivity ceiling: the same 128 `(layer, lag)`
   wires for every token in every batch can only express fixed relative patterns.
2. **Gradient starvation froze the graph**: only selected wires got gradient; with a
   constant exploration noise scale the learned logit gaps (std → 20–37) made turnover
   exactly 0 by mid-run. "Self-annealing" became self-freezing.
3. **Replacing** the proven attention path instead of augmenting it meant the new
   mechanism had to beat dense attention before it had learned anything.
4. **Kernel economics**: gather + atomics lost 2.5× wall-clock to flash SDPA while
   doing ~8× fewer FLOPs.

## v3 design — `blockwire_train_gpt.py`

Synthesis of a three-report design panel (block-sparse designer, fully-differentiable
designer, first-principles red team) plus the user constraint that killed both pooled
designs: **no pooling, no down-projected values, random rewiring is cheap and unbiased.**

The v3 is the *winning within-layer arm's semantics* (sketch-scan selection →
entmax-1.5 + learned null over the selected set) extended to the cross-layer universe
and re-shaped to block granularity so every op is a dense batched matmul:

- **Universe**: all layers 0..L full-resolution rotary K/V, 32-token key blocks,
  64-query query blocks. No pooled representations anywhere — block importance is a
  top-8-mean over the block's queries of each query's *best exact-token* sketched score.
- **Discovery (zero params, zero state)**: a fixed random orthogonal sketch
  (head_dim→16, the winner's mechanism — a JL estimate of the exact q·k score, not a
  learned compression) scores every token under `no_grad`; top-12 blocks per
  (kv-head, query-block) + **4 fresh uniformly-random blocks per step** (deterministic
  hash on the optimizer-step clock) + the always-on 128-token local span.
- **Why rewiring can't freeze**: there is no selector state to freeze — selection is
  recomputed from the live q/k every forward. The random audit slots give every block
  full-resolution fine-pass gradient ~14×/step (batch × query-block multiplier), which
  breaks the cross-layer q-addressing starvation loop the red team identified.
- **Causality**: only fully-causal blocks are scannable/drawable; the sole straddling
  keys are the current layer's diagonal span under an explicit per-token mask. Earlier
  layers' near-diagonal is structurally unaddressable (leak-counter stat asserts 0;
  gradient-based causality test in `tests/test_blockwire_xlayer.py`).
- **Speed shape**: contiguous block gathers + batched matmuls. No Triton, no atomics.
  New params: one null bias per layer.
- **Memory shape**: entmax-1.5 uses a custom autograd.Function with the analytic
  Peters-Martins Jacobian (backward needs only the output), so under
  `torch.compile` the min-cut partitioner saves just the entmax output and the
  backward is sort-free. *No* activation checkpoint — verified (node-level graph
  probes) that a checkpoint here does the opposite: it recompute-tags the region
  and forces every query-chunk's sort/cumsum composite back into the backward
  graph co-live, which is what OOM'd the 32 GB card four times. The per-chunk
  query loop bounds the forward's transient fp32 score + int64 sort index to one
  chunk. Persistent cost = one saved entmax output per layer.

Knobs: `SPARSE_SEL/RAND/KEY_BLOCK/QUERY_BLOCK/COARSE_DIM/IMP_TOPR`,
`SPARSE_FINE=entmax|softmax`, `SPARSE_XLAYER=0` (within-layer-only control).

Planned 2k runs: `blockwire_2k` (main), `blockwire_within_2k` (SPARSE_XLAYER=0 — the
decisive cross-layer A/B), `blockwire_rand_2k` (SPARSE_SEL=0, SPARSE_RAND=16 — tests
the "random rewiring is enough" hypothesis directly).

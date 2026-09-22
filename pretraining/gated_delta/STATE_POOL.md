# Shared pool of routed GDN2 state banks

Status: implemented and run to 1,000 updates (job 8905); negative result on
routing, see Results.

## Hypothesis

The layerwise GDN2 model keeps six isolated recurrent states; nothing a lower
layer writes is visible to an upper layer except through the residual stream
of the same token. The final-to-first feedback experiment ([FEEDBACK.md](FEEDBACK.md))
showed that letting one layer read memory written by the whole network is
worth 0.038 BPB at 1,000 updates over the standalone GDN2 (1.3228 versus
1.3605). The user's proposal generalizes that: twelve GDN2 state banks are
shared by all six layers, every layer picks one bank per token by a mixture-of-
experts router, writes to it and reads from it, so the network learns which
layers share which memory. The expected outcome is a modest throughput cost
and a clearly better BPB than the standalone GDN2.

## What is implemented

`pretraining/nanogpt_mini/gated_delta_pool.py`, `SharedPoolGatedDeltaGPT`.

- Every layer keeps its exact private GDN2 recurrence, unchanged from the
  production model (custom operators, packed projections, FLA chunk kernels,
  saved backward intermediates). The pool is additive.
- Each sequence owns `pool_banks = 12` GDN2 banks of the private state's shape
  (four heads of 128 x 128). Events are the (token, layer) pairs, ordered by
  token first and layer second: layer L at token t sees every write with an
  earlier token and the writes of layers 0..L at token t, including its own,
  exactly like the private recurrence's inclusive read.
- Per layer and token, the router (512 -> 12, fp32 logits and softmax) takes
  the top-1 bank. The layer writes the same key, value, log-decay, erase and
  write fields that updated its private state; it reads the bank with its own
  pool query (512 -> 512 projection, width-4 SiLU short convolution, in-kernel
  L2 normalization).
- Executing (token, layer) order token by token is about 140x slower on these
  kernels (`pretraining/state_routing`, job 8780: 156 s per update). The model
  therefore uses two Jacobi passes, as the retained feedback design did: pass
  one is the private model and records every layer's writes, queries and
  routes; the pool then processes each bank's events in event order with one
  chunk-kernel call (below); pass two rebuilds the private model from the
  embeddings (block 0's mixer output is reused from pass one, since its input
  is the same embedding) and adds each layer's pool read through a gated RMS
  norm (gate from that pass's private projection), scaled by the router's
  confidence (the selected probability, Switch Transformer style) and a
  zero-initialized 512 x 512 readout. Both passes are scored,
  `loss = (CE_1 + CE_2) / 2 + 0.01 x tokens x sum of the six routers' Switch
  balance losses`; the headline BPB is pass two, pass one is reported as
  `val_bpb_pass1`.
- The pool call is packed, not padded to a capacity. Each sequence's 6,144
  events are laid out bank by bank: a bank's events, in event order, form one
  contiguous segment that starts at a 64-row chunk boundary (its count is
  rounded up to whole chunks with no-op rows: zero query, key, value, decay,
  erase and write; the kernel's L2 normalization has an additive epsilon, so
  zeros stay zeros). A filler segment of no-op rows completes the row to a
  static 108 chunks (6,144 / 64 + 12 banks; the rounding of twelve banks
  costs fewer than twelve chunks, so at least one filler chunk always
  remains). The layout is a handful of cumsum/gather ops on the routes
  (`pool_layout`), the events are copied layer by layer into the packed rows
  (`scatter_events`) and read back per layer (`gather_events`), and the FLA
  chunk kernels run once over the `[1, batch x 6,912]` row in their
  variable-length mode with the segment offsets (`cu_seqlens`, twelve banks
  plus the filler per sequence) and the per-chunk segment table
  (`chunk_indices`) computed on the device: every shape is static, nothing
  depends on the routing on the host, so the step compiles as one graph and
  CUDA-graphs, and no event is ever dropped whatever the routing. Against
  the earlier static capacity of 768 events per bank (9,216 rows per
  sequence, events beyond capacity dropped) the pool kernel does 25% less
  work and its buffers are 25% smaller. FLA's identity-keyed tensor cache
  is disabled for the process (`FLA_DISABLE_TENSOR_CACHE`), because the
  cache would splice a stale offsets tensor into the captured graph.
  Validation reports `val_pool_balance` and `val_pool_load_max` (the largest
  share of one sequence's events routed to one bank).
- Parameters: 27,904,536 versus 24,708,888 for the standalone GDN2 (per
  layer: pool query 262,144, its convolution 2,048, router 6,144, pool norm
  128, readout 262,144). Initialization follows the mixer: Xavier gain
  2^-2.5 for the pool query and router, FLA's convolution default, unit norm
  weight, zero readout, so the initial model is exactly the private model run
  twice. The pool modules are attached after the private backbone is built,
  so at the same seed the private parameters are bit-identical to the
  standalone model's: the comparison isolates the architecture from
  initialization noise. Optimizer groups: routers AdamW 0.001 without weight decay; pool
  matrices Muon; pool convolution in the convolution group; pool norm in the
  scalar group; everything else as the production model.
- Cost: pass two doubles the private model, and the pool call processes
  6,912 rows per sequence, 1.125x the six private layers' tokens. Memory
  forces a smaller microbatch than the production 64: with the static
  capacity layout at 32 the training graph's warmup ran out of memory beside
  the 8.7 GiB validation graph (job 8854; the process alone held 28.6 GiB).
  `scripts/scan_gdn2_pool_microbatch.py` (job 8872, unpinned kernels)
  settled the microbatch at 16: the packed pool still runs out of memory at
  32 (training graph beside a 16- or 32-row validation graph), and the
  private model itself is fastest at 16 (0.939 s per update against 0.974
  at 32 and 0.991 at 64: the state-passing kernels do not lose occupancy at
  16 x 4 heads, and the smaller activations help the rest). The pool at 16
  takes 2.588 s per update (202,570 tokens/s, 11.4 GiB allocated, 16.7 GiB
  reserved with both graphs resident), 2.76x the private model: two private
  passes (2 x 0.388 s of chunk kernels), the pool call (0.455 s, 108 chunks
  per row), and the doubled projections. Its profile showed the event
  dispatch costing 0.13 s per update in backward: tracking six in-place
  per-layer copies with autograd runs a pool-sized `index_fill` per layer
  and field in backward. The dispatch is now a pair of autograd functions
  whose backward is the exact inverse permutation (a gather for the scatter,
  a scatter for the gather).

Incremental decoding is not implemented for the pool (validation and training
use complete sequences); `forward_hidden(use_cache=True)` raises.

## Protocol

1. `pretraining/tests/test_gated_delta_pool_protocol.py` (CPU): the packed
   layout against a worked example and its invariants over random routes
   (unique slots, contiguous chunk-aligned bank segments in event order,
   consistent segment offsets and chunk table, a filler chunk per row),
   scatter/gather round trip and gradients, the whole dispatch traced as one
   graph, the balance loss, configuration rejection,
   initialization and parameter ownership, optimizer groups, loss wrapper
   selection, parameter count, the update-400 gate policy and reference
   loading. `pretraining/tests/test_gated_delta_protocol.py` binds the pool
   keys into the throughput-report verifier and the CLIs.
2. `pretraining/tests/test_gated_delta_pool_gpu.py` (mlq): pass one equals
   the standalone model bit for bit and zero readout makes both passes
   identical; pool reads match an fp32 literal per-bank recurrence in
   (token, layer) order over every event and banks are independent; the
   single packed variable-length kernel call equals one dense kernel call per
   bank, forward and backward; pass two is causal and every pool parameter
   trains under a fullgraph compile with no graph breaks; the production
   model compiles whole, replays co-resident training and validation CUDA
   graphs and leaves headroom on the 32 GiB device.
3. `scripts/pin_gdn2_autotune.py --variant shared_pool` tunes every FLA kernel
   fresh at the pool's shapes (private microbatch x T1024 and the packed
   pool row) inside the production executor and writes profile
   `rtx5090_gdn2_pool_b<microbatch>_t1024_custom_ops`.
4. `scripts/benchmark_gated_delta.py --shared-pool --microbatch <microbatch>`
   under the strict pinned policy produces the throughput report (exit 75 is
   expected: the pool cannot clear the 5% gate against plain mini).
5. `scripts/train_recurrent_slots.py --shared-pool --microbatch <microbatch>
   --allow-slow-diagnostic --decision-gate-reference gdn2_custom_ops_pinned_1k`:
   matched data, seed 1337, 1,000 updates, validation every 20. The
   update-400 decision gate binds the control to the same data, tokenizer,
   seed, context, budget and cadence, then compares the headline (pass-two)
   BPB with the control's 1.5082 at update 400 and prunes with exit 75 unless the pool is
   at least 0.005 better (`gate_decision.json`). The final comparison is
   against the pinned production run's 1.35857 BPB.

## Results

Static-capacity layout (superseded): contract job 8847 passed the four
semantic contracts and failed the production test on a reporting detail
(`1 - mean(kept)` was -3e-8 with nothing dropped). Job 8854 passed the same
four contracts and ran out of memory in the B32 production test: the B32
validation graph held 8.7 GiB and the B32 training warmup needed more than
the remaining 19 GiB. The capacity layout was then replaced by the packed
layout above (no drops, 25% less pool work). The microbatch scan (job 8872,
`ablation_results/gdn2_pool_microbatch_scan/scan.json`) fixed the
microbatch at 16 (numbers above). Contract job 8884 passed the four
semantic contracts on the packed layout and failed the production test on a
test defect: it asserted a nonzero gradient for every parameter while the
head and the MLP readouts still started at zero, so the cross-entropy reached
no block and `blocks.0.mlp.fc` received exactly zero (the balance loss alone
had reached the attention parameters). The test now randomizes those
readouts like the small-scale contracts, checks the compiled backward
without graph capture first, and compares the replayed gradients against it.
Job 8892 then passed every contract on the numerics (the replayed
gradients match the uncaptured compiled backward; 77.2 ms per B16
microbatch) and tripped only the memory headroom assertion, because the new
uncaptured pass left cached allocator blocks that graph capture cannot reuse
(27.9 GiB reserved against 16.7 GiB for the graphs alone in the scan); the
test now releases that cache before capture. Contracts job 8897 passed all
five (77.9 ms per B16 microbatch, 11.2 GiB allocated, 16.4 GiB reserved
with both graphs resident). Kernel profile job 8898 wrote
`rtx5090_gdn2_pool_b16_t1024_custom_ops` (14 kernels, 32 entries) and
measured 2.359 s per update under it, 9% under the scan's unpinned 2.588 s.
Throughput report job 8899 (`ablation_results/gdn2_pool_pinned_b16`) under
the strict pinned policy: 2.346 s per update, 223,506 tokens/s, 16.4 GiB
reserved, against plain mini's 0.544 s (963,817 tokens/s), speedup 0.232;
it exited 75 as expected (the pool is a learning experiment, not a
throughput candidate). The 1k run had been chained on that job's success
and was skipped; it was resubmitted directly as job 8905
(`ablation_results/gdn2_pool_pinned_1k`, update-400 gate against
`gdn2_custom_ops_pinned_1k`, `--allow-slow-diagnostic`).

Run 8905 (`ablation_results/gdn2_pool_pinned_1k`) passed the update-400 gate
(1.5013 against the control's 1.5082, threshold 1.5032) and completed 1,000
updates at 2.434 s per update: final pass-two BPB 1.3495, pass-one 1.3521,
against the control's 1.3586, plain nanoGPT mini's 1.3433 and the mini
feedback run's 1.3265. The pool did not specialize: every router stayed at
the balance loss's uniform minimum (summed 6.00 for six layers) with a
largest bank share of 0.105 per sequence throughout, router weights moved
from 0.011 to about 0.016 rms over the run under AdamW at 0.001 (a noise
gradient; a consistent one would have moved them tens of times further),
while Muon grew the pool readout to the private output projection's scale.
The read itself is worth 0.003 BPB (pass two over pass one, flat from
update 300); the rest of the gain over the control lives in pass one, i.e.
in the two-pass training dynamics rather than in memory.

Diagnosis. Switch-style routing learns because experts differ at
initialization; these banks have no parameters and are identical random
mixtures of every layer's writes under uniform routing, so "bank j helped"
is a coin flip, the task gradient on the router averages to zero, and the
balance loss (only needed under the retired static capacity; the packed
layout makes imbalance free) holds the uniform fixed point. Hard argmax on
near-uniform logits makes the assignment itself noise, so no bank acquires
an identity. The confidence multiplier (about 0.1) enters the read at a
tenth of a private mixer output's scale, and every layer writes in its own
private key space, so a query matches its own (redundant) writes and can
exploit other layers' only after all six projections co-adapt. Candidate
fixes, in order of leverage: banks with identities (single-writer banks, 6
layers x 2, routing only the reads; or per-bank low-rank write/read
adapters), a shared pool key projection, no balance loss, a straight-through
full-scale read, and top-2 soft reads through read-only events.

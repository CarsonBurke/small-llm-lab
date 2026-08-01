# Post-training

No base checkpoint is currently selected. The old 6-layer dense
`mathmix_v4` checkpoint was superseded because its four-source corpus was
narrow, its 2,000-step run was too short, and it did not provide KDA's
long-context memory scaling.

The replacement campaign uses the measured `KKKDKKKD` KDA skeleton, the
four-domain `k3mix_v5` corpus, and a measured 2K -> 4K -> 8K context
curriculum. A new `postraining/base_model.json` should be written only after:

1. the 2,000-step data/architecture gates complete;
2. the winning KDA recipe completes the 8,000-step curriculum;
3. the KDA post-training adapter strict-loads the checkpoint and matches dense
   prefill/decode logits;
4. FineWeb plus web/code/math/knowledge validation and the frozen-policy
   rollout gate pass.

The KDA adapter for gate 3 exists: `postraining/kda_backbone.py`
(`NanoKDABackbone` over the import-safe `nanogpt_mini_kda_model.py`) loads
`*_kda_*` checkpoints through `model_io.load_model` with zero key mapping.
Dense MHA layers keep KV caches; KDA mixers carry
`(conv_q, conv_k, conv_v, state)` decode caches — the delta-rule state is
always fp32. Rollout decode is a pure-PyTorch recurrence (stays inside the
fullgraph-compiled step artifact); prefill and teacher-forced replay dispatch
to FLA's `chunk_kda` on CUDA (the mixer is an eager region, so the replay
artifacts compile with graph breaks — set automatically). The
continuous-refill scheduler is KV-address machinery and is refused for KDA;
use lockstep. CPU parity/integration tests live in
`postraining/tests/test_kda_backbone.py`; the CUDA parity gate
(kernel-vs-reference, dense-vs-decode logits, left-pad invariance) is
`postraining/kda_gpu_parity.py`, run through mlq.

Run every GPU workload through `mlq`. Before a new training campaign, run the
frozen-policy learnability gate:

```bash
mlq submit \
  --name posttrain_base_gate \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  python3 -m postraining.train_latent_vapo \
    --checkpoint logs/SELECTED_K3_BASE_final_model.pt \
    --output postraining/runs/posttrain_base_gate \
    --reasoning-mode cot \
    --rollout-groups 4 \
    --rollout-only
```

Then start a run with an explicit output directory:

```bash
mlq submit \
  --name posttrain_cot \
  --cwd "$PWD" \
  --max-parallel-runs 1 \
  -- \
  python3 -m postraining.train_latent_vapo \
    --checkpoint logs/SELECTED_K3_BASE_final_model.pt \
    --output postraining/runs/posttrain_cot \
    --reasoning-mode cot \
    --steps 2000
```

## Latent thinking: deterministic hidden carry

`--reasoning-mode latent` trains the hidden-carry policy (execution schema
v28). A "thought" is the post-final-norm belief that produced a generated
token; when that token feeds back as input, its producing belief rides along
as a residual on the token embedding:

```
combined = embed(token) + hasThought * (W(belief) + type_bias)
input    = combined + MLP(RMSNorm(combined))   # per block, relu^2, identity at init
```

`hasThought` is 1 exactly where the input token was model-generated (prompt
tokens and the first generation step carry no hidden). `W` is
zero-initialized, so a fresh run is bit-exact with the pretrained token
policy while `W` gets a full-rank gradient from the first step. (The v1
form gated an orthogonal `W` behind one zero-init scalar gain — a
multiplicative saddle neither factor escaped in practice.) There is no
stochastic thought channel: the only actions are tokens, training is standard
token-level VAPO (DAPO clip + HL-Gauss critic), and the carried belief is
detached replay data — no BPTT. The separate critic re-derives combined
embeddings with its own combiner weights. `cot` and `none` remain token-only
control modes with zero-width hidden storage. Combiner geometry is set by
`--combined-mlp-hidden` (default 2048) and `--combined-mlp-blocks`
(default 1; 0 ablates to the pure residual).

Checkpoints from the pre-v28 stochastic thought/gate policy cannot be
resumed or evaluated; there are no migrations. Test-time-read-compute
(carrying hiddens for read tokens) is deliberately out of scope for this
schema.

The campaign uses GPT-2 BPE with token 50256 as both BOS and EOS. The final
checkpoint records its exact architecture, corpus manifest, and maximum
pretraining context.

The BPB guard evaluates one 8-row eval batch (8192 tokens) of a fixed
FineWeb validation prefix by default — a cheap catastrophic-drift canary,
paired against the same tokens every eval. Guard values are comparable only
within one `--bpb-val-tokens` setting; identity checks against a recorded
pretraining val_bpb (the init gate) need `--bpb-val-tokens 2097152`.

`train_latent_vapo` writes a manifest, source snapshot, JSONL metrics,
TensorBoard events, and exact-resume checkpoints beneath the output directory.
The base checkpoint file is never overwritten; the trainable actor copy and
the separate critic state live in the run directory.

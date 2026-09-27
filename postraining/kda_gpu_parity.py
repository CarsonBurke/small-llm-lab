"""GPU parity gate for the KDA post-training backbone.

Asserts, on CUDA, the agreements the CPU suite cannot check:

1. FLA ``chunk_kda`` under the training flags == the pure-PyTorch reference
   recurrence (outputs and final state) — validates the derivation the decode
   step and every CPU test stand on; and the fused Triton decode step
   (``kda_decode_kernel``) == the reference step over 96 steps, eager and
   inside a fullgraph compile.
2. Teacher-forced logits == stepwise decode logits on a 986-configuration
   trunk (8 layers, KDA mixers at 0,1,2,4,5,6, 3 heads): fp32 tight for
   decode from an empty cache at every position, and bf16-autocast loose for
   prefill + decode — the "dense prefill/decode logits match" half of the
   base-model gate, on random weights. CUDA prefill runs varlen
   FlashAttention, which is bf16/fp16 only, so prefill is checked in the
   production dtype rather than through an fp32 fallback.
3. Left-padded prefill == unpadded prefill through the CUDA kernel path.
4. Paged continuous-refill decode == per-row dense decode (bf16), with dead
   padding rows in every step (the scratch-lane redirect under CUDA), and
   the inductor-compiled ``paged_step_core`` == its eager self — the gate
   ``--rollout-scheduler continuous_refill`` and ``--rollout-graph-decode``
   stand on for KDA trunks.
5. The same BF16 teacher/decode and compiled paged-decode gates with the
   production 32-expert Stable LatentMoE active in every layer.
6. The deterministic hidden carry (``--reasoning-mode carry``) on the 986
   trunk with a live combiner, in the production bf16 regime: rollout token
   log-probabilities == parallel-replay ones, measured beside the cot
   rollout on the same trunk so the carry's own cost is the excess over the
   token-only rail; each stored carry == the replayed belief one slot
   earlier; and the compiled ``step_core`` (whose Gaussian outputs are None
   under carry) == the eager one, decoding the same carried stream.
7. Whole cot and live-carry rollouts collected the way the trainer collects
   them -- ``PinnedDecodeArena``'s compiled tick replayed as CUDA graphs over
   left-padded ragged prompts, with survivor compaction across captures and
   attention through the ranged decode kernel -- == eager parallel replay:
   each emitted token wins the counter-Gumbel race under the replay logits,
   except at bf16 near-ties.

Writes ``--output`` (default ``postraining/runs/kda_gpu_parity/result.json``)
and exits nonzero on any failed bound, so an mlq failure IS a parity failure.
Completed results are run artifacts: re-running after a model change writes a
new path rather than overwriting the old evidence, and an existing file is
refused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pretraining.nanogpt_mini import nanogpt_mini_kda_model as kda_model
from postraining.kda_backbone import NanoKDABackbone
from postraining.graph_decode import PinnedDecodeArena
from postraining.latent_rollout import (
    PAD_SLOT,
    LatentRolloutBatch,
    counter_gumbel_scores,
    generated_slot_mask,
    replay_beliefs,
    rollout_continuations,
    trim_stream,
)
from postraining.latent_thought import LatentThoughtModel

DEFAULT_RESULT_PATH = Path("postraining/runs/kda_gpu_parity/result.json")

MODEL_KWARGS = dict(
    vocab_size=512,
    num_layers=8,
    model_dim=512,
    mlp_hidden=2070,
    delta_num_heads=3,
    delta_layer_indices=[0, 1, 2, 4, 5, 6],
    delta_attention_type="kda",
    delta_full_rank_gate=False,
    delta_mlp_on_delta=False,
    dense_attention_type="mha",
)


def max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT_PATH)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; parity results are immutable")
    device = torch.device("cuda")
    torch.manual_seed(0)
    results: dict[str, float] = {}
    failures: list[str] = []

    def check(name: str, err: float, bound: float) -> None:
        results[name] = err
        results[f"{name}_bound"] = bound
        if not err <= bound:
            failures.append(f"{name}: {err:.3e} > {bound:.3e}")

    # ---- 1. chunk_kda vs reference recurrence -----------------------------
    from fla.ops.kda import chunk_kda

    B, T, H, D = 4, 384, 3, 128
    q = torch.randn(B, T, H, D, device=device)
    k = torch.randn(B, T, H, D, device=device)
    v = torch.randn(B, T, H, D, device=device)
    decay = torch.randn(B, T, H, D, device=device)
    beta = torch.randn(B, T, H, device=device)
    A_log = torch.zeros(H, device=device)
    dt_bias = torch.randn(H * D, device=device) * 0.5 - 3.0
    with torch.no_grad():
        kernel_out, kernel_state = chunk_kda(
            q=q, k=k, v=v, g=decay, beta=beta,
            A_log=A_log, dt_bias=dt_bias,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=kda_model.KDA_SAFE_GATE_LOWER_BOUND,
            state_v_first=True,
            disable_recompute=True,
        )
        reference_out, reference_state = kda_model.reference_kda_recurrence(
            q, k, v, decay, beta, A_log, dt_bias
        )
    check("chunk_vs_reference_out", max_err(kernel_out, reference_out), 5e-3)
    check(
        "chunk_vs_reference_state", max_err(kernel_state, reference_state), 5e-3
    )
    results["kernel_state_shape"] = list(kernel_state.shape)  # type: ignore[assignment]

    # bf16 inputs, the production activation dtype.
    with torch.no_grad():
        kernel_bf16, _ = chunk_kda(
            q=q.bfloat16(), k=k.bfloat16(), v=v.bfloat16(),
            g=decay.bfloat16(), beta=beta.bfloat16().float(),
            A_log=A_log, dt_bias=dt_bias,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=kda_model.KDA_SAFE_GATE_LOWER_BOUND,
            state_v_first=True,
            disable_recompute=True,
        )
        reference_bf16, _ = kda_model.reference_kda_recurrence(
            q.bfloat16(), k.bfloat16(), v.bfloat16(),
            decay.bfloat16(), beta.bfloat16().float(), A_log, dt_bias,
        )
    check("chunk_vs_reference_bf16", max_err(kernel_bf16, reference_bf16), 1e-1)

    # ---- 1b. fused Triton decode step vs the reference step ---------------
    # 96 consecutive steps from a nonzero state on bf16 head inputs (the
    # production activation dtype), eager and inside a fullgraph compile --
    # the op must trace into the rollout tick, not split it. Both sides are
    # fp32 recurrences differing only in reduction order.
    from pretraining.nanogpt_mini.kda_decode_kernel import fused_decode_supported

    rows, steps = 40, 96
    heads_in = torch.randn(steps, 3, rows, H, D, device=device).bfloat16()
    step_gates = kda_model.kda_decay_gate(
        torch.randn(steps, rows, H, D, device=device), A_log, dt_bias
    )
    step_betas = torch.rand(steps, rows, H, device=device)
    initial = torch.randn(rows, H, D, D, device=device) * 0.1

    def decode(step_fn, state):
        outs = []
        for t in range(steps):
            q_t, k_t, v_t = heads_in[t]
            outs.append(step_fn(q_t, k_t, v_t, step_gates[t], step_betas[t], state))
        return torch.stack(outs)

    compiled_step = torch.compile(
        kda_model.kda_decode_recurrent_step, fullgraph=True, dynamic=False
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference_state = initial.clone()
        reference_steps = decode(kda_model.kda_recurrent_step, reference_state)
        fused_state = initial.clone()
        fused_steps = decode(kda_model.kda_decode_recurrent_step, fused_state)
        compiled_state = initial.clone()
        compiled_steps = decode(compiled_step, compiled_state)
    if not fused_decode_supported(initial):
        failures.append("fused decode step does not cover the production shape")
    if fused_steps.dtype != torch.float32:
        failures.append(f"fused decode output dtype {fused_steps.dtype}")
    check("fused_step_vs_reference_out", max_err(fused_steps, reference_steps), 1e-4)
    check("fused_step_vs_reference_state", max_err(fused_state, reference_state), 1e-4)
    check("compiled_fused_step_vs_eager_out", max_err(compiled_steps, fused_steps), 1e-5)
    check(
        "compiled_fused_step_vs_eager_state",
        max_err(compiled_state, fused_state),
        1e-5,
    )

    # ---- 2. teacher-forced vs prefill+decode on the 986 layout ------------
    torch.manual_seed(1)
    backbone = NanoKDABackbone(**MODEL_KWARGS).float().to(device).eval()
    with torch.no_grad():
        for block in backbone.blocks:
            attn = block.attn
            if block.use_kda:
                attn.o_proj.weight.normal_(std=0.02)
            else:
                attn.proj.weight.normal_(std=0.02)
            if block.use_mlp:
                block.mlp.proj.weight.normal_(std=0.02)
        backbone.proj.weight.normal_(std=0.02)
    wrapper = LatentThoughtModel(backbone).to(device)
    ids = torch.randint(0, 512, (8, 320), device=device)
    prompt = 256
    # fp32 exactness through decode alone: every position from an empty
    # cache, so the recurrent state and KV cache carry the whole sequence.
    with torch.no_grad():
        reference = wrapper.policy_logits(ids).float()
        caches = wrapper.make_generation_cache(8, 320, device)
        step_err = 0.0
        for position in range(320):
            output = wrapper.token_step(ids[:, position], caches, position)
            step_err = max(
                step_err, max_err(output.logits, reference[:, position])
            )
    check("fp32_decode_vs_dense", step_err, 2e-3)

    # Autocast bf16: the production rollout regime, prefill included.
    # Rounded caches and bf16 GEMMs move logits at bf16 resolution; the bound
    # is a sanity rail, the meaningful exactness statement is fp32 above and
    # rollout-vs-replay (same dtype both sides) in training.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference_bf = wrapper.policy_logits(ids).float()
        caches = wrapper.make_generation_cache(
            8, 320, device, dtype=torch.bfloat16
        )
        output = wrapper.prefill(ids[:, :prompt], caches)
        step_err_bf = max_err(output.logits, reference_bf[:, prompt - 1])
        for position in range(prompt, 320):
            output = wrapper.token_step(ids[:, position], caches, position)
            step_err_bf = max(
                step_err_bf, max_err(output.logits, reference_bf[:, position])
            )
    check("bf16_decode_vs_dense", step_err_bf, 5e-1)

    # ---- 3. left-padded prefill through the CUDA kernels ------------------
    # CUDA left-padded production prefill deliberately uses native varlen
    # FlashAttention, whose supported execution dtypes are bf16/fp16.  Keep
    # this parity gate on the actual production dtype instead of asking the
    # optimized path to grow a slow fp32 fallback.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pad = 64
        padded = torch.zeros((8, 320 + pad), dtype=torch.long, device=device)
        padded[:, pad:] = ids
        key_valid = torch.zeros(
            (8, 320 + pad), dtype=torch.bool, device=device
        )
        key_valid[:, pad:] = True
        clean_caches = wrapper.make_generation_cache(8, 320, device)
        clean = wrapper.prefill(ids, clean_caches)
        padded_caches = wrapper.make_generation_cache(8, 320 + pad, device)
        shifted = wrapper.prefill(padded, padded_caches, key_valid)
        pad_err = max_err(shifted.logits, clean.logits)
        state_err = max(
            max_err(padded_tensor, clean_tensor)
            for padded_layer, clean_layer in zip(padded_caches, clean_caches)
            if len(padded_layer) == 4
            for padded_tensor, clean_tensor in zip(padded_layer, clean_layer)
        )
    check("leftpad_logits", pad_err, 5e-1)
    check("leftpad_state", state_err, 5e-1)

    # ---- 4. paged continuous-refill decode, eager and compiled ------------
    # Two ragged groups fanned into shuffled lanes of an 8-lane pool, then
    # 32 decode steps at width 8 with four dead padding rows per step (the
    # production bucket-padding shape). The dead rows name a LIVE lane, so
    # every step exercises the recurrent scratch-lane redirect; any leak
    # compounds through the 32-step recurrence instead of averaging out.
    torch.manual_seed(2)
    steps = 32
    prompt_width = 256
    pad_prompts = ids[:2, :prompt_width].clone()
    pad_prompts[0, :64] = 0
    lengths = torch.tensor([prompt_width - 64, prompt_width], device=device)
    slots = torch.tensor([[2, 0], [3, 1]], device=device)
    selected_groups = torch.tensor([1, 0], device=device)
    step_tokens = torch.randint(0, 512, (steps, 4), device=device)

    # Prompt admission prefills, so both sides run in the production bf16
    # regime; paged and dense then differ only in cache layout and
    # accumulation order, far inside the bf16 teacher-vs-decode rail.
    def run_paged() -> list[torch.Tensor]:
        paged = wrapper.make_paged_generation_cache(
            8, prompt_width + steps, device, dtype=torch.bfloat16
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bank = wrapper.build_prompt_prefix_bank(pad_prompts, lengths)
            wrapper.admit_prompt_prefixes(bank, selected_groups, slots, paged)
            live = torch.tensor([True] * 4 + [False] * 4, device=device)
            slot_ids = torch.cat(
                (
                    slots.flatten(),
                    torch.zeros(4, dtype=torch.long, device=device),
                )
            )
            logits = []
            for offset in range(steps):
                positions = torch.where(
                    live,
                    torch.tensor(prompt_width + offset, device=device),
                    torch.tensor(0, device=device),
                )
                tokens = torch.cat(
                    (
                        step_tokens[offset],
                        torch.zeros(4, dtype=torch.long, device=device),
                    )
                )
                stepped = wrapper.token_paged_step(
                    tokens,
                    paged,
                    slot_ids=slot_ids,
                    positions=positions,
                    live=live,
                )
                logits.append(stepped.logits[:4].clone())
        return logits

    key_valid = (
        torch.arange(prompt_width, device=device)[None]
        >= (prompt_width - lengths)[:, None]
    )
    dense_logits = [[] for _ in range(steps)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for group in selected_groups.tolist():
            for _ in range(2):
                dense_cache = wrapper.make_generation_cache(
                    1, prompt_width + steps, device, dtype=torch.bfloat16
                )
                wrapper.prefill(
                    pad_prompts[group : group + 1],
                    dense_cache,
                    key_valid[group : group + 1],
                )
                row = len(dense_logits[0])
                for offset in range(steps):
                    mask = torch.cat(
                        (
                            key_valid[group : group + 1],
                            torch.ones(
                                1,
                                1 + offset,
                                dtype=torch.bool,
                                device=device,
                            ),
                        ),
                        dim=1,
                    )
                    stepped = wrapper.token_step(
                        step_tokens[offset, row : row + 1],
                        dense_cache,
                        prompt_width + offset,
                        mask,
                    )
                    dense_logits[offset].append(stepped.logits)

    eager_paged = run_paged()
    paged_err = max(
        max_err(paged_step_logits, torch.cat(dense_step_logits))
        for paged_step_logits, dense_step_logits in zip(
            eager_paged, dense_logits
        )
    )
    check("bf16_paged_vs_dense", paged_err, 1e-1)

    # The production shadowing pattern: the trainer swaps the bound method
    # for the compiled artifact, so parity here is parity for the real
    # decode loop under --rollout-graph-decode's compile flags.
    original_paged_step_core = wrapper.paged_step_core
    wrapper.paged_step_core = torch.compile(
        original_paged_step_core, fullgraph=True, dynamic=False
    )
    try:
        compiled_paged = run_paged()
    finally:
        wrapper.paged_step_core = original_paged_step_core
    compiled_err = max(
        max_err(compiled_step, eager_step)
        for compiled_step, eager_step in zip(compiled_paged, eager_paged)
    )
    check("bf16_compiled_paged_vs_eager", compiled_err, 2e-2)

    # ---- 5. Stable LatentMoE in production BF16/fullgraph decode ----------
    moe_kwargs = {
        **MODEL_KWARGS,
        "moe_num_experts": 32,
        "moe_top_k": 2,
        "moe_latent_dim": 128,
        "moe_expert_hidden": 256,
        "moe_shared_hidden": 64,
        "moe_num_shared_experts": 2,
        "moe_layer_indices": list(range(8)),
    }
    torch.manual_seed(3)
    moe_backbone = NanoKDABackbone(**moe_kwargs).to(device).eval()
    with torch.no_grad():
        for block in moe_backbone.blocks:
            if block.use_kda:
                block.attn.o_proj.weight.normal_(std=0.02)
            else:
                block.attn.proj.weight.normal_(std=0.02)
            block.mlp.shared_expert.down_proj.weight.normal_(std=0.02)
            block.mlp.latent_up_proj.weight.normal_(std=0.02)
        moe_backbone.proj.weight.normal_(std=0.02)
    moe_wrapper = LatentThoughtModel(moe_backbone).to(device)
    moe_ids = ids[:8, :96]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        moe_reference = moe_wrapper.policy_logits(moe_ids).float()
        moe_caches = moe_wrapper.make_generation_cache(
            8, 96, device, dtype=torch.bfloat16
        )
        moe_output = moe_wrapper.prefill(moe_ids[:, :64], moe_caches)
        moe_decode_err = max_err(moe_output.logits, moe_reference[:, 63])
        for position in range(64, 96):
            moe_output = moe_wrapper.token_step(
                moe_ids[:, position], moe_caches, position
            )
            moe_decode_err = max(
                moe_decode_err,
                max_err(moe_output.logits, moe_reference[:, position]),
            )
    check("moe_bf16_decode_vs_dense", moe_decode_err, 5e-1)

    def run_moe_paged() -> list[torch.Tensor]:
        paged = moe_wrapper.make_paged_generation_cache(
            8, prompt_width + steps, device, dtype=torch.bfloat16
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bank = moe_wrapper.build_prompt_prefix_bank(pad_prompts, lengths)
            moe_wrapper.admit_prompt_prefixes(
                bank, selected_groups, slots, paged
            )
            live = torch.tensor([True] * 4 + [False] * 4, device=device)
            slot_ids = torch.cat(
                (
                    slots.flatten(),
                    torch.zeros(4, dtype=torch.long, device=device),
                )
            )
            logits = []
            for offset in range(steps):
                positions = torch.where(
                    live,
                    torch.tensor(prompt_width + offset, device=device),
                    torch.tensor(0, device=device),
                )
                tokens = torch.cat(
                    (
                        step_tokens[offset],
                        torch.zeros(4, dtype=torch.long, device=device),
                    )
                )
                stepped = moe_wrapper.token_paged_step(
                    tokens,
                    paged,
                    slot_ids=slot_ids,
                    positions=positions,
                    live=live,
                )
                logits.append(stepped.logits[:4].clone())
        return logits

    eager_moe_paged = run_moe_paged()
    original_moe_paged_core = moe_wrapper.paged_step_core
    moe_wrapper.paged_step_core = torch.compile(
        original_moe_paged_core, fullgraph=True, dynamic=False
    )
    try:
        compiled_moe_paged = run_moe_paged()
    finally:
        moe_wrapper.paged_step_core = original_moe_paged_core
    moe_compiled_err = max(
        max_err(compiled_step, eager_step)
        for compiled_step, eager_step in zip(
            compiled_moe_paged, eager_moe_paged
        )
    )
    check("moe_bf16_compiled_paged_vs_eager", moe_compiled_err, 2e-2)

    # ---- 6. deterministic hidden carry, bf16 rollout vs replay ------------
    # A fresh carry combiner is an identity, which would make this section a
    # second copy of the cot measurement; live weights make the stored carry
    # load-bearing in every generated input.
    carry_wrapper = LatentThoughtModel(backbone, hidden_carry=True).to(device)
    carry_wrapper.eval()
    torch.manual_seed(4)
    with torch.no_grad():
        combiner = carry_wrapper.combiner
        combiner.carry.weight.normal_(std=0.5 / MODEL_KWARGS["model_dim"] ** 0.5)
        combiner.type_bias.normal_(std=0.05)
        for mlp in combiner.mlps:
            mlp.proj.weight.normal_(std=0.02)
    # Four unique prompts fanned out to two samples each, with a stop set
    # that ends roughly half the rows early: prompt fan-out and finished-row
    # compaction both remap the rows each carry is written through.
    carry_prompts = ids[:4, :64]
    carry_repeats = 2
    carry_stop_ids = tuple(range(8))

    def roll(policy: LatentThoughtModel, hidden_carry: bool) -> LatentRolloutBatch:
        # One seed for both policies: cot and carry share the sampling lane,
        # so the two rails are measured over comparable draws.
        torch.manual_seed(5)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return trim_stream(
                rollout_continuations(
                    policy,
                    carry_prompts,
                    64,
                    64,
                    1.0,
                    1.0,
                    stop_ids=carry_stop_ids,
                    prompt_repeats=carry_repeats,
                    pin_emit=True,
                    hidden_carry=hidden_carry,
                    cache_dtype=torch.bfloat16,
                    sync_every=1,
                )
            )

    def replay_error(
        policy: LatentThoughtModel, batch: LatentRolloutBatch
    ) -> tuple[float, torch.Tensor]:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            stream_inputs, beliefs = replay_beliefs(policy, batch)
            logits = policy.backbone.logits_from_features(
                policy.renderer_features(stream_inputs, beliefs)
            ).float()
        targets = torch.zeros_like(batch.token_ids)
        targets[:, :-1] = batch.token_ids[:, 1:]
        logprobs = (
            logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
        )
        emits = batch.emit_mask.bool()
        return max_err(logprobs[emits], batch.old_token_logprobs[emits]), beliefs

    cot_batch = roll(wrapper, hidden_carry=False)
    cot_replay_err, _ = replay_error(wrapper, cot_batch)
    carry_batch = roll(carry_wrapper, hidden_carry=True)
    carry_replay_err, carry_beliefs = replay_error(carry_wrapper, carry_batch)
    # Reported side by side rather than differenced: the two policies sample
    # different trajectories, so the gap between their maxima is not itself
    # a statistic. A carry error far above cot's is the signal to inspect.
    check("cot_bf16_rollout_vs_replay_logprob", cot_replay_err, 5e-1)
    check("carry_bf16_rollout_vs_replay_logprob", carry_replay_err, 5e-1)
    action_counts = carry_batch.action_mask.sum(1)
    results["carry_rows_stopped_early"] = int((action_counts < 64).sum())
    carried = generated_slot_mask(carry_batch)[:, 1:]
    check(
        "carry_bf16_hidden_vs_replay_belief",
        max_err(
            carry_batch.hiddens[:, 1:][carried],
            carry_beliefs[:, :-1][carried],
        ),
        2.5e-1,
    )
    results["carry_generated_slots"] = int(carried.sum())

    # The lockstep rollout's compiled artifact (``generation_step``) under
    # the carry: its Gaussian outputs are None, and the carried input rides
    # the same fullgraph step. Both artifacts decode the eager rollout's own
    # stream teacher-forced -- identical inputs, so the comparison never
    # depends on a sampled token surviving a rounding difference.
    compiled_step_core = torch.compile(
        carry_wrapper.step_core, fullgraph=True, dynamic=True
    )
    prompt_length = carry_batch.prompt_length
    stream_length = carry_batch.stream_length

    def carry_decode(step_core) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        rows = carry_prompts.size(0) * carry_repeats
        caches = carry_wrapper.make_generation_cache(
            rows, stream_length, device, dtype=torch.bfloat16
        )
        beliefs, logits = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            carry_wrapper.prefill(
                carry_prompts.repeat_interleave(carry_repeats, dim=0), caches
            )
            for position in range(prompt_length, stream_length):
                belief, predicted, log_sigma, step_logits = step_core(
                    carry_wrapper.combined_input(
                        carry_batch.token_ids[:, position],
                        carry_batch.hiddens[:, position],
                    ),
                    caches,
                    position,
                )
                if predicted is not None or log_sigma is not None:
                    raise AssertionError(
                        "hidden-carry step produced Gaussian parameters"
                    )
                beliefs.append(belief.float())
                logits.append(step_logits.float())
        return beliefs, logits

    eager_beliefs, eager_logits = carry_decode(carry_wrapper.step_core)
    compiled_beliefs, compiled_logits = carry_decode(compiled_step_core)
    check(
        "carry_bf16_compiled_vs_eager_logits",
        max(map(max_err, compiled_logits, eager_logits)),
        2e-2,
    )
    check(
        "carry_bf16_compiled_vs_eager_belief",
        max(map(max_err, compiled_beliefs, eager_beliefs)),
        2e-2,
    )

    # ---- 7. graph-decode arena rollouts vs replay, cot and carry ----------
    # Whole rollouts collected the way the trainer collects cot/none/carry:
    # PinnedDecodeArena's compiled tick replayed as CUDA graphs, over
    # left-padded ragged prompts with fan-out (so rows attend different key
    # ranges), and row buckets small enough that survivors are compacted
    # through several captures within one rollout. The arena records no likelihoods,
    # so it is graded through its sampler instead: every emitted token must be
    # the counter-Gumbel race winner under the trainer's own replay logits
    # for the same (seed, slot). A disagreement is only admissible where the
    # replay race is a near-tie that bf16 reduction order can flip; a stale
    # graph buffer, a mis-compacted row, a wrong width mask or a mis-keyed
    # slot changes the winner at ordinary margins.
    ragged_lengths = torch.tensor([64, 48, 57, 40], device=device)
    ragged_prompts = torch.zeros_like(carry_prompts)
    for row, length in enumerate(ragged_lengths.tolist()):
        ragged_prompts[row, -length:] = carry_prompts[row, :length]
    arena_rows = ragged_prompts.size(0) * carry_repeats
    arena_new_tokens = 64

    def build_arena(
        policy: LatentThoughtModel, hidden_carry: bool
    ) -> PinnedDecodeArena:
        return PinnedDecodeArena(
            policy,
            rows=arena_rows,
            # Wider than any stream here so the attention kernel's split
            # count (sized by cache width) exceeds one in the small buckets.
            kv_width=1024,
            temperature=1.0,
            stop_ids=carry_stop_ids,
            device=device,
            hidden_carry=hidden_carry,
            row_buckets=(8, 6, 4, 2),
            sync_every=2,
        )

    def arena_roll(
        arena: PinnedDecodeArena,
        prompts: torch.Tensor,
        lengths: torch.Tensor,
        seeds: torch.Tensor,
    ) -> LatentRolloutBatch:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            batch = arena.rollout(
                prompts,
                lengths,
                prompt_repeats=carry_repeats,
                max_new_tokens=arena_new_tokens,
                max_stream_steps=arena_new_tokens,
                token_seeds=seeds,
            )
        return trim_stream(batch)

    def race_disagreement(
        policy: LatentThoughtModel,
        batch: LatentRolloutBatch,
        seeds: torch.Tensor,
    ) -> tuple[float, int, torch.Tensor]:
        """Largest replay race margin at a token the arena did not pick."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            stream_inputs, beliefs = replay_beliefs(policy, batch)
            logits = policy.backbone.logits_from_features(
                policy.renderer_features(stream_inputs, beliefs)
            ).float()
        worst_margin, mismatches = 0.0, 0
        emits = batch.emit_mask.bool()
        # The race is keyed on each row's unpadded slot.
        pads = (batch.kind != PAD_SLOT).long().argmax(1)
        for slot in range(batch.kind.size(1) - 1):
            rows = emits[:, slot].nonzero().squeeze(-1)
            if rows.numel() == 0:
                continue
            scores = counter_gumbel_scores(
                logits[rows, slot], seeds[rows], slot - pads[rows], 1.0
            )
            winner = scores.argmax(-1)
            sampled = batch.token_ids[rows, slot + 1]
            wrong = winner != sampled
            if not bool(wrong.any()):
                continue
            margin = (
                scores.gather(-1, winner[:, None])
                - scores.gather(-1, sampled[:, None])
            )[wrong]
            worst_margin = max(worst_margin, float(margin.max()))
            mismatches += int(wrong.sum())
        return worst_margin, mismatches, beliefs

    arena_seeds = torch.randint(
        0, 2**62, (arena_rows,), device=device, dtype=torch.int64
    )
    # The second rollout reuses the captured arena the way the next training
    # pool does: after an in-place parameter update (the optimizer's only
    # kind of write), with a narrower prompt set, fewer rows and a smaller
    # starting bucket. Stale KV, stale filler rows and replayed graphs that
    # missed the update all fail the same race check.
    narrow_lengths = torch.tensor([33, 21], device=device)
    narrow_prompts = torch.zeros(
        (2, int(narrow_lengths.max())), dtype=torch.long, device=device
    )
    for row, length in enumerate(narrow_lengths.tolist()):
        narrow_prompts[row, -length:] = carry_prompts[row + 2, :length]
    narrow_seeds = arena_seeds[: 2 * carry_repeats] ^ 0x5DEECE66D
    for policy, hidden_carry, mode in (
        (wrapper, False, "cot"),
        (carry_wrapper, True, "carry"),
    ):
        arena = build_arena(policy, hidden_carry)
        batch = arena_roll(arena, ragged_prompts, ragged_lengths, arena_seeds)
        saved = [parameter.detach().clone() for parameter in policy.parameters()]
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.mul_(1.03)
        reused = arena_roll(arena, narrow_prompts, narrow_lengths, narrow_seeds)
        reused_margin, reused_mismatches, _ = race_disagreement(
            policy, reused, narrow_seeds
        )
        with torch.no_grad():
            for parameter, value in zip(policy.parameters(), saved, strict=True):
                parameter.copy_(value)
        margin, mismatches, beliefs = race_disagreement(policy, batch, arena_seeds)
        # bf16 logits of this trunk differ between decode and replay by
        # ~1e-2 (section 2); a flip needs a race margin of that order.
        check(f"{mode}_bf16_arena_rollout_vs_replay_race_margin", margin, 5e-2)
        check(
            f"{mode}_bf16_reused_arena_after_update_race_margin",
            reused_margin,
            5e-2,
        )
        results[f"{mode}_arena_emitted_tokens"] = int(batch.emit_mask.sum())
        results[f"{mode}_arena_race_mismatches"] = mismatches
        results[f"{mode}_reused_arena_emitted_tokens"] = int(
            reused.emit_mask.sum()
        )
        results[f"{mode}_reused_arena_race_mismatches"] = reused_mismatches
        results[f"{mode}_arena_rows_stopped_early"] = int(
            (batch.action_mask.sum(1) < arena_new_tokens).sum()
        )
        results[f"{mode}_arena_row_buckets"] = len(arena.row_buckets)
        if hidden_carry:
            carried = generated_slot_mask(batch)[:, 1:]
            check(
                "carry_bf16_arena_hidden_vs_replay_belief",
                max_err(
                    batch.hiddens[:, 1:][carried],
                    beliefs[:, :-1][carried],
                ),
                2.5e-1,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"failures": failures, **results}, indent=2)
    )
    for name, value in results.items():
        print(f"{name}: {value}")
    if failures:
        raise SystemExit("KDA GPU parity FAILED: " + "; ".join(failures))
    print("KDA GPU parity: all bounds passed")


if __name__ == "__main__":
    main()

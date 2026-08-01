"""GPU parity gate for the KDA post-training backbone.

Asserts, on CUDA, the agreements the CPU suite cannot check:

1. FLA ``chunk_kda`` under the training flags == the pure-PyTorch reference
   recurrence (outputs and final state) — validates the derivation the decode
   step and every CPU test stand on.
2. Teacher-forced logits == prefill + stepwise decode logits on a
   986-configuration trunk (8 layers, KDA mixers at 0,1,2,4,5,6, 3 heads),
   fp32 tight and bf16-autocast loose — the "dense prefill/decode logits
   match" half of the base-model gate, on random weights.
3. Left-padded prefill == unpadded prefill through the CUDA kernel path.
4. Paged continuous-refill decode == per-row dense decode, with dead
   padding rows in every step (the scratch-lane redirect under CUDA), and
   the inductor-compiled ``paged_step_core`` == its eager self — the gate
   ``--rollout-scheduler continuous_refill`` and ``--rollout-graph-decode``
   stand on for KDA trunks.

Writes ``postraining/runs/kda_gpu_parity/result.json`` and exits nonzero on
any failed bound, so an mlq failure IS a parity failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import nanogpt_mini_kda_model as kda_model
from postraining.kda_backbone import NanoKDABackbone
from postraining.latent_thought import LatentThoughtModel

RESULT_PATH = Path("postraining/runs/kda_gpu_parity/result.json")

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
    wrapper = LatentThoughtModel(backbone)
    ids = torch.randint(0, 512, (8, 320), device=device)
    prompt = 256
    with torch.no_grad():
        reference = wrapper.policy_logits(ids).float()
        caches = wrapper.make_generation_cache(8, 320, device)
        output = wrapper.prefill(ids[:, :prompt], caches)
        prefill_err = max_err(output.logits, reference[:, prompt - 1])
        step_err = 0.0
        for position in range(prompt, 320):
            output = wrapper.token_step(ids[:, position], caches, position)
            step_err = max(
                step_err, max_err(output.logits, reference[:, position])
            )
    check("fp32_prefill_vs_dense", prefill_err, 2e-3)
    check("fp32_decode_vs_dense", step_err, 2e-3)

    # Autocast bf16: the production rollout regime. Rounded caches and bf16
    # GEMMs move logits at bf16 resolution; the bound is a sanity rail, the
    # meaningful exactness statements are fp32 above and rollout-vs-replay
    # (same dtype both sides) in training smokes.
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
    with torch.no_grad():
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
    check("leftpad_logits", pad_err, 2e-3)
    check("leftpad_state", state_err, 1e-3)

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

    def run_paged() -> list[torch.Tensor]:
        paged = wrapper.make_paged_generation_cache(
            8, prompt_width + steps, device
        )
        bank = wrapper.build_prompt_prefix_bank(pad_prompts, lengths)
        wrapper.admit_prompt_prefixes(bank, selected_groups, slots, paged)
        live = torch.tensor([True] * 4 + [False] * 4, device=device)
        slot_ids = torch.cat(
            (slots.flatten(), torch.zeros(4, dtype=torch.long, device=device))
        )
        logits = []
        with torch.no_grad():
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
    with torch.no_grad():
        for group in selected_groups.tolist():
            for _ in range(2):
                dense_cache = wrapper.make_generation_cache(
                    1, prompt_width + steps, device
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
    check("fp32_paged_vs_dense", paged_err, 2e-3)

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
    check("fp32_compiled_paged_vs_eager", compiled_err, 2e-3)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps({"failures": failures, **results}, indent=2)
    )
    for name, value in results.items():
        print(f"{name}: {value}")
    if failures:
        raise SystemExit("KDA GPU parity FAILED: " + "; ".join(failures))
    print("KDA GPU parity: all bounds passed")


if __name__ == "__main__":
    main()

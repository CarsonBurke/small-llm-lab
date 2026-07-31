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

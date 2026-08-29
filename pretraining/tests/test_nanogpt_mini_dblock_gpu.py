"""CUDA parity tests for the DiffusionBlocks two-stream constructions.

The CPU suite proves the read-only interleaved-slot semantics against the
pure-PyTorch KDA oracle; this suite proves the *released FLA chunk kernel*
and *flex_attention* implement the same contract under the trainer's exact
flags and dtypes. Queue through mlq:

    mlq submit --name dblock_gpu_parity --cwd "$PWD" --max-parallel-runs 1 -- \
        .venv/bin/python -m pytest pretraining/tests/test_nanogpt_mini_dblock_gpu.py -v
"""

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_dblock import (
    READ_ONLY_LOGIT,
    dblock_mask_mod,
    deinterleave_streams,
    interleave_streams,
)
from pretraining.nanogpt_mini.nanogpt_mini_kda_model import (
    reference_kda_recurrence,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _trainer_flag_chunk_kda(q, k, v, g, beta, A_log, dt_bias):
    from fla.ops.kda import chunk_kda

    out, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        state_v_first=True,
        disable_recompute=False,
    )
    return out


@cuda_only
def test_chunk_kda_readonly_interleave_matches_reference():
    """The kernel with READ_ONLY_LOGIT decay/beta on interleaved noisy slots
    must reproduce the fp32 reference recurrence: untouched clean trajectory,
    inclusive-prefix reads at noisy slots."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    B, T, H, D = 2, 96, 3, 128

    def draw(*shape):
        return torch.randn(*shape, device=device, dtype=torch.float32)

    clean = {name: draw(B, T, H, D) for name in ("q", "k", "v", "g")}
    noisy = {name: draw(B, T, H, D) for name in ("q", "k", "v")}
    beta_clean = draw(B, T, H)
    A_log = (draw(H).abs() * 0.1).contiguous()
    dt_bias = (draw(H * D) * 0.1 - 3.0).contiguous()

    q_int = interleave_streams(clean["q"], noisy["q"])
    k_int = interleave_streams(clean["k"], noisy["k"])
    v_int = interleave_streams(clean["v"], noisy["v"])
    g_int = interleave_streams(
        clean["g"], torch.full_like(clean["g"], READ_ONLY_LOGIT)
    )
    beta_int = interleave_streams(
        beta_clean, torch.full_like(beta_clean, READ_ONLY_LOGIT)
    )

    kernel_out = _trainer_flag_chunk_kda(
        q_int.bfloat16(),
        k_int.bfloat16(),
        v_int.bfloat16(),
        g_int.bfloat16(),
        beta_int.float(),
        A_log,
        dt_bias,
    ).float()
    reference_out, _ = reference_kda_recurrence(
        q_int, k_int, v_int, g_int, beta_int, A_log, dt_bias
    )

    kernel_clean, kernel_noisy = deinterleave_streams(kernel_out)
    reference_clean, reference_noisy = deinterleave_streams(
        reference_out.float()
    )
    scale = reference_out.abs().max().clamp(min=1.0)
    assert (kernel_clean - reference_clean).abs().max() / scale < 4e-2
    assert (kernel_noisy - reference_noisy).abs().max() / scale < 4e-2

    # The clean slots of the interleaved pass must also match a plain
    # clean-only kernel pass: the read-only slots are invisible to the state.
    clean_only = _trainer_flag_chunk_kda(
        clean["q"].bfloat16(),
        clean["k"].bfloat16(),
        clean["v"].bfloat16(),
        clean["g"].bfloat16(),
        beta_clean.float(),
        A_log,
        dt_bias,
    ).float()
    assert (kernel_clean - clean_only).abs().max() / scale < 4e-2


@cuda_only
def test_flex_attention_dblock_mask_matches_sdpa_oracle():
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )

    torch.manual_seed(1)
    device = torch.device("cuda")
    B, H, T, D = 2, 4, 128, 128
    q = torch.randn(B, H, 2 * T, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, 2 * T, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, 2 * T, D, device=device, dtype=torch.bfloat16)

    mask_mod = dblock_mask_mod(T)
    block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=2 * T, KV_LEN=2 * T, device="cuda"
    )
    flex_out = flex_attention(q, k, v, block_mask=block_mask, scale=0.12)

    dense_mask = torch.zeros(2 * T, 2 * T, dtype=torch.bool, device=device)
    for q_idx in range(2 * T):
        for kv_idx in range(2 * T):
            if q_idx < T:
                allowed = kv_idx <= q_idx
            else:
                allowed = (kv_idx < T and kv_idx <= q_idx - T) or (
                    kv_idx == q_idx
                )
            dense_mask[q_idx, kv_idx] = allowed
    sdpa_out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=dense_mask, scale=0.12
    )
    assert (flex_out.float() - sdpa_out.float()).abs().max() < 3e-2

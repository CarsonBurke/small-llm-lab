from __future__ import annotations

import os

import pytest
import torch

from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_yarn import (
    FixedTargetYarnRotary,
    FreshLeJEPASharedRMSV1FixedYarn,
)
from train_gpt import Rotary


KWARGS = dict(
    vocab_size=32,
    num_layers=3,
    model_dim=32,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2,
    tie_embeddings=True,
    tied_embed_init_std=0.01,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.5,
)


def test_fixed_yarn_preserves_pretrained_prefix_exactly():
    reference = Rotary(8)
    yarn = FixedTargetYarnRotary(8, pretrained_length=8, target_length=32)
    expected_cos, expected_sin = reference(8, torch.device("cpu"), torch.float32)
    actual_cos, actual_sin = yarn(8, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(actual_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(actual_sin, expected_sin, rtol=0, atol=0)


def test_fixed_yarn_full_and_absolute_position_tables_match():
    yarn = FixedTargetYarnRotary(8, pretrained_length=4, target_length=16)
    cos, sin = yarn(12, torch.device("cpu"), torch.float32)
    for position in range(12):
        frequency = yarn.frequencies(torch.tensor([position]))[0]
        torch.testing.assert_close(frequency.cos(), cos[0, 0, position])
        torch.testing.assert_close(frequency.sin(), sin[0, 0, position])


def test_65k_target_is_continuous_at_pretrained_boundary():
    yarn = FixedTargetYarnRotary(
        64, pretrained_length=1024, target_length=65_536
    )
    positions = torch.tensor([1023, 1024, 1025, 2048])
    frequencies = yarn.frequencies(positions)
    torch.testing.assert_close(frequencies[1], 1024 * yarn.inv_freq)
    torch.testing.assert_close(
        frequencies[2] - frequencies[1], yarn.yarn_inv_freq, rtol=1e-4, atol=3e-5
    )
    torch.testing.assert_close(
        frequencies[3] - frequencies[1],
        1024 * yarn.yarn_inv_freq,
        rtol=1e-5,
        atol=3e-5,
    )
    assert torch.all(yarn.yarn_inv_freq <= yarn.inv_freq)
    assert torch.any(yarn.yarn_inv_freq < yarn.inv_freq)


def test_fixed_yarn_incremental_matches_full_sequence(monkeypatch):
    monkeypatch.delenv("FRESH_YARN_INIT_CHECKPOINT", raising=False)
    monkeypatch.setattr(FreshLeJEPASharedRMSV1FixedYarn, "pretrained_context", 4)
    monkeypatch.setattr(FreshLeJEPASharedRMSV1FixedYarn, "target_context", 16)
    model = FreshLeJEPASharedRMSV1FixedYarn(**KWARGS).eval()
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, std=0.05)
            torch.nn.init.normal_(block.mlp.proj.weight, std=0.05)
        torch.nn.init.normal_(model.critic_probe.output.weight, std=0.05)
    ids = torch.randint(0, 32, (2, 10))
    with torch.no_grad():
        full_logits = model.policy_logits(ids)
        full_values = model.values(ids)
        caches = model.make_generation_cache(2, ids.size(1), ids.device)
        logits_steps, value_steps = [], []
        for position in range(ids.size(1)):
            logits, values, caches = model.generation_step(
                ids[:, position], caches, position
            )
            logits_steps.append(logits)
            value_steps.append(values)
    torch.testing.assert_close(torch.stack(logits_steps, 1), full_logits, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(torch.stack(value_steps, 1), full_values, rtol=2e-4, atol=2e-4)


def test_selected_checkpoint_strict_loads_into_fixed_yarn(monkeypatch):
    checkpoint = (
        "ablation_results/fresh_lejepa_shared_rms_v1_probes_1k/"
        "pretraining_checkpoint.pt"
    )
    monkeypatch.setenv("FRESH_YARN_INIT_CHECKPOINT", checkpoint)
    model = FreshLeJEPASharedRMSV1FixedYarn(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    torch.testing.assert_close(
        model.tok_emb.weight, payload["model"]["tok_emb.weight"].float()
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("RUN_CUDA_TESTS") != "1",
    reason="explicit idle-GPU validation only",
)
def test_compiled_cuda_cached_tensor_positions_cross_boundary(monkeypatch):
    monkeypatch.delenv("FRESH_YARN_INIT_CHECKPOINT", raising=False)
    monkeypatch.setattr(FreshLeJEPASharedRMSV1FixedYarn, "pretrained_context", 4)
    monkeypatch.setattr(FreshLeJEPASharedRMSV1FixedYarn, "target_context", 16)
    model = FreshLeJEPASharedRMSV1FixedYarn(**KWARGS).cuda().bfloat16().eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.float()
    ids = torch.randint(0, 32, (2, 10), device="cuda")
    compiled_step = torch.compile(model.generation_step, fullgraph=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full_logits = model.policy_logits(ids)
        caches = model.make_generation_cache(2, ids.size(1), ids.device)
        steps = []
        for position in range(ids.size(1)):
            logits, _, caches = compiled_step(
                ids[:, position], caches, torch.tensor(position, device="cuda")
            )
            steps.append(logits)
    torch.testing.assert_close(
        torch.stack(steps, 1), full_logits, rtol=3e-3, atol=3e-3
    )

"""Uncertain mask, recomputation, gradient and checkpoint boundaries (CPU only)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from postraining.uno import (
    UnoAdapterBank,
    UnoConfig,
    attach_uno_adapters,
    chunked_head_l1,
    load_uno_adapter,
    paired_uno_inputs,
    uno_checkpoint_payload,
    uno_mask_mod,
)


def _model():
    model = nn.Module()
    model.config = SimpleNamespace(
        model_type="llama",
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        vocab_size=13,
    )
    layer = nn.Module()
    layer.self_attn = nn.Module()
    for name, width in (("q_proj", 4), ("k_proj", 2), ("v_proj", 2), ("o_proj", 4)):
        setattr(layer.self_attn, name, nn.Linear(4, width, bias=False))
    layer.mlp = nn.Module()
    layer.mlp.gate_proj = nn.Linear(4, 8, bias=False)
    layer.mlp.up_proj = nn.Linear(4, 8, bias=False)
    layer.mlp.down_proj = nn.Linear(8, 4, bias=False)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([layer])
    model.lm_head = nn.Linear(4, 13, bias=False)
    model.requires_grad_(False)
    return model


def test_mask_preserves_causal_teacher_and_excludes_same_block_clean_leakage():
    mask = uno_mask_mod(8, 4)
    indices = torch.arange(16)
    dense = mask(None, None, indices[:, None], indices[None, :])
    assert torch.equal(dense[:8, :8], torch.ones(8, 8, dtype=torch.bool).tril())
    assert not dense[:8, 8:].any()
    # First noisy output of block two sees clean block one and itself only.
    assert dense[12].nonzero().flatten().tolist() == [0, 1, 2, 3, 12]
    assert dense[14].nonzero().flatten().tolist() == [0, 1, 2, 3, 12, 13, 14]
    assert dense[8].nonzero().flatten().tolist() == [8]


def test_paired_noise_uses_full_vocabulary_and_aligned_logical_positions():
    clean = torch.zeros((2, 1024), dtype=torch.long)
    ids, positions, gate = paired_uno_inputs(
        clean, 13, generator=torch.Generator().manual_seed(19)
    )
    expected = torch.randint(
        13, clean.shape, generator=torch.Generator().manual_seed(19)
    )
    assert torch.equal(ids[:, 1024:], expected)
    assert ids[:, 1024:].max() == 12
    assert torch.equal(ids[:, :1024], clean)
    assert torch.equal(positions[:, :1024], positions[:, 1024:])
    assert not gate[:, :1024].any()
    assert gate[:, 1024:].all()


def test_chunked_l1_matches_dense_loss_and_hidden_gradient_without_teacher_grad():
    torch.manual_seed(29)
    head = nn.Linear(4, 13, bias=True).requires_grad_(False)
    student = torch.randn(2, 5, 4, requires_grad=True)
    teacher = torch.randn(2, 5, 4, requires_grad=True)
    weights = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32)
    loss = chunked_head_l1(student, teacher, head, chunk_size=3, token_weights=weights)
    loss.backward()
    actual_gradient = student.grad.clone()
    reference_student = student.detach().requires_grad_()
    q = head(reference_student).softmax(-1)
    p = head(teacher.detach()).softmax(-1)
    reference = ((q - p).abs().sum(-1) * weights).sum() / weights.sum()
    reference.backward()
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(
        actual_gradient, reference_student.grad, atol=1e-7, rtol=1e-5
    )
    assert teacher.grad is None
    assert head.weight.grad is None
    assert not actual_gradient[weights == 0].any()


def test_checkpoint_recomputation_keeps_original_gate_and_backbone_frozen():
    torch.manual_seed(31)
    model = _model()
    bank = UnoAdapterBank(model, UnoConfig(rank=2, alpha=4))
    for projection in bank.projections.values():
        nn.init.normal_(projection.lora_b)
    router = attach_uno_adapters(model, bank)
    projection = model.model.layers[0].self_attn.q_proj
    inputs = torch.randn(1, 4, 4)
    gate = torch.tensor([[[0], [0], [1], [1]]])
    baseline = nn.functional.linear(inputs, projection.weight)
    router.set_gate(gate)
    output = checkpoint(
        projection, inputs, use_reentrant=False, context_fn=router.checkpoint_contexts
    )
    torch.testing.assert_close(output[:, :2], baseline[:, :2], rtol=0, atol=0)
    assert not torch.equal(output[:, 2:], baseline[:, 2:])
    router.set_gate(None)
    output.square().sum().backward()
    actual = {
        name: p.grad.clone()
        for name, p in bank.named_parameters()
        if p.grad is not None
    }
    bank.zero_grad(set_to_none=True)
    router.set_gate(gate)
    projection(inputs).square().sum().backward()
    for name, parameter in bank.named_parameters():
        if name in actual:
            torch.testing.assert_close(actual[name], parameter.grad)
    assert any(torch.count_nonzero(value) for value in actual.values())
    assert projection.weight.grad is None
    router.set_gate(None)
    torch.testing.assert_close(projection(inputs), baseline, rtol=0, atol=0)
    router.close()


def test_checkpoint_rejects_untrained_nonfinite_and_wrong_model(tmp_path):
    model = _model()
    bank = UnoAdapterBank(model, UnoConfig(rank=2, alpha=4))
    with torch.no_grad():
        for projection in bank.projections.values():
            projection.lora_b.fill_(0.125)
    payload = uno_checkpoint_payload(
        bank,
        model,
        model_id="model",
        revision="revision",
        trained_tokens=12,
        step=1,
        teacher_sha256="a" * 64,
        training={"corpus_sha256": "b" * 64},
    )
    path = tmp_path / "uno.pt"
    torch.save(payload, path)
    state_before = torch.get_rng_state().clone()
    loaded, _ = load_uno_adapter(path, model, model_id="model", revision="revision")
    assert torch.equal(torch.get_rng_state(), state_before)
    inputs = torch.randn(1, 2, 4)
    key = "model__layers__0__self_attn__q_proj"
    torch.testing.assert_close(
        loaded.projections[key](inputs), bank.projections[key](inputs)
    )
    with pytest.raises(ValueError, match="identity"):
        load_uno_adapter(path, model, model_id="other", revision="revision")
    payload["trained_tokens"] = 0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="trained_tokens"):
        load_uno_adapter(path, model, model_id="model", revision="revision")
    payload["trained_tokens"] = 12
    next(iter(payload["adapter"].values())).fill_(float("nan"))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="invalid Uno adapter tensor"):
        load_uno_adapter(path, model, model_id="model", revision="revision")


def test_owner_dtype_migration_does_not_quantize_adapter_masters():
    model = _model()
    bank = UnoAdapterBank(model, UnoConfig(rank=2, alpha=4))
    router = attach_uno_adapters(model, bank)
    before = {name: value.detach().clone() for name, value in bank.named_parameters()}
    model.bfloat16()
    assert model.lm_head.weight.dtype == torch.bfloat16
    for name, parameter in bank.named_parameters():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    router.close()

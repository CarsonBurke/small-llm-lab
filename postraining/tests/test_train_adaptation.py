from __future__ import annotations

import torch

from fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
from postraining.latent_thought import LatentThoughtModel
from postraining.model_io import _pope_construction
from postraining.train_adaptation import (
    SampledLatentEval,
    adaptation_losses,
    build_optimizers,
)

KWARGS = dict(
    vocab_size=32, num_layers=3, model_dim=32, num_heads=4, num_kv_heads=2,
    mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.01,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
)


def _wrapper() -> LatentThoughtModel:
    with _pope_construction():
        backbone = FreshLeJEPASharedRMSV1PoPE(**KWARGS)
    return LatentThoughtModel(backbone)


def _batch(batch: int = 2, length: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    trajectory = torch.randint(0, 32, (batch, length + 1))
    return trajectory[:, :-1], trajectory[:, 1:]


def test_losses_are_finite_scalars_with_thought_exposure():
    torch.manual_seed(31)
    wrapper = _wrapper()
    input_ids, target_ids = _batch()
    losses = adaptation_losses(wrapper, input_ids, target_ids, 0.25, 1.0, 0.5)
    for key in ("latent_nll", "renderer_ce", "prediction_mse", "copy_last_mse"):
        assert losses[key].shape == ()
        assert torch.isfinite(losses[key]), key
    assert losses["latent_nll"].requires_grad
    assert losses["renderer_ce"].requires_grad
    assert 0.0 < float(losses["thought_fraction"]) < 1.0
    (losses["latent_nll"] + losses["renderer_ce"]).backward()


def test_zero_exposure_renderer_ce_matches_pretraining_eval_loss():
    torch.manual_seed(37)
    wrapper = _wrapper()
    backbone = wrapper.backbone.eval()
    input_ids, target_ids = _batch()
    with torch.no_grad():
        pretraining_loss = backbone(input_ids, target_ids)
        losses = adaptation_losses(wrapper, input_ids, target_ids, 0.0, 1.0, 0.5)
    torch.testing.assert_close(losses["renderer_ce"], pretraining_loss)
    assert float(losses["thought_fraction"]) == 0.0


def test_renderer_ce_grads_reach_projector_and_probe_but_not_trunk():
    torch.manual_seed(41)
    wrapper = _wrapper()
    backbone = wrapper.backbone
    # The probe's output layer is zero-initialized in this architecture, which
    # blocks gradient flow through it on a fresh model; the pretrained
    # checkpoint has trained weights there, so emulate that.
    with torch.no_grad():
        backbone.policy_probe.output.weight.normal_(std=0.02)
    input_ids, target_ids = _batch()
    losses = adaptation_losses(wrapper, input_ids, target_ids, 0.0, 1.0, 0.5)
    losses["renderer_ce"].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in backbone.prediction_projector.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in backbone.policy_probe.parameters()
    )
    assert backbone.blocks[0].attn.c_qkv.weight.grad is None
    assert backbone.tok_emb.weight.grad is None


def test_latent_nll_grads_reach_trunk_embeddings_and_log_std_head():
    torch.manual_seed(43)
    wrapper = _wrapper()
    backbone = wrapper.backbone
    input_ids, target_ids = _batch()
    losses = adaptation_losses(wrapper, input_ids, target_ids, 0.0, 1.0, 0.5)
    losses["latent_nll"].backward()
    assert backbone.blocks[0].attn.c_qkv.weight.grad is not None
    assert backbone.tok_emb.weight.grad is not None
    assert wrapper.transition.log_std_head.weight.grad is not None
    assert all(p.grad is None for p in backbone.policy_probe.parameters())


def test_latent_target_branch_is_attached_like_pretraining():
    # The checkpoint architecture is attached-target: latent-loss gradients
    # must reach the embedding of a token that only ever appears as a target.
    torch.manual_seed(61)
    wrapper = _wrapper()
    backbone = wrapper.backbone
    trajectory = torch.randint(0, 31, (2, 13))
    trajectory[:, -1] = 31  # token 31 appears exclusively as the last target
    input_ids, target_ids = trajectory[:, :-1], trajectory[:, 1:]
    losses = adaptation_losses(wrapper, input_ids, target_ids, 0.0, 1.0, 0.5)
    losses["latent_nll"].backward()
    assert float(backbone.tok_emb.weight.grad[31].abs().sum()) > 0


def test_full_exposure_doubles_the_stream_and_stays_finite():
    torch.manual_seed(47)
    wrapper = _wrapper()
    input_ids, target_ids = _batch(batch=1, length=6)
    losses = adaptation_losses(wrapper, input_ids, target_ids, 1.0, 0.5, 0.5)
    assert float(losses["thought_fraction"]) == 1.0
    assert torch.isfinite(losses["latent_nll"])
    assert torch.isfinite(losses["renderer_ce"])


def test_optimizer_split_covers_every_trainable_parameter_exactly_once():
    wrapper = _wrapper()
    for parameter in wrapper.gate.parameters():
        parameter.requires_grad_(False)
    optimizers = build_optimizers(wrapper, 0.3, 3e-4)
    optimizer_tok, optimizer_muon, optimizer_scalar, optimizer_new = optimizers
    grouped: list[int] = []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            grouped.extend(id(p) for p in group["params"])
    assert len(grouped) == len(set(grouped))
    trainable = {id(p) for p in wrapper.parameters() if p.requires_grad}
    assert set(grouped) == trainable
    backbone = wrapper.backbone
    assert any(
        "delta_c" in name
        for optimizer in (optimizer_scalar,)
        for group in optimizer.param_groups
        for p in group["params"]
        for name, q in backbone.blocks.named_parameters()
        if q is p
    )
    muon_ids = {
        id(p) for group in optimizer_muon.param_groups for p in group["params"]
    }
    assert all(
        id(p) in muon_ids
        for p in backbone.prediction_projector.parameters()
        if p.ndim == 2
    )
    new_ids = {
        id(p) for group in optimizer_new.param_groups for p in group["params"]
    }
    assert {id(p) for p in wrapper.transition.parameters()} <= new_ids
    assert all(id(p) not in new_ids for p in wrapper.gate.parameters())


def test_training_step_reduces_losses_on_a_fixed_batch():
    torch.manual_seed(53)
    wrapper = _wrapper()
    # Pretraining-scale learning rates diverge on a tiny random-init model;
    # the test pins that a full step is wired correctly, not the LR schedule.
    optimizers = build_optimizers(wrapper, 0.02, 1e-3)
    input_ids, target_ids = _batch(batch=4, length=16)
    first = None
    for _ in range(30):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        losses = adaptation_losses(wrapper, input_ids, target_ids, 0.0, 1.0, 0.5)
        total = losses["renderer_ce"] + losses["latent_nll"]
        if first is None:
            first = float(total)
        total.backward()
        for optimizer in optimizers:
            optimizer.step()
    assert float(total) < first


def test_sampled_latent_eval_tracks_closed_loop_when_noise_is_tiny():
    torch.manual_seed(59)
    wrapper = _wrapper()
    with torch.no_grad():
        wrapper.transition.log_std_head.bias.fill_(wrapper.transition.log_std_min)
    backbone = wrapper.backbone.eval()
    shim = SampledLatentEval(wrapper).eval()
    input_ids, target_ids = _batch()
    with torch.no_grad():
        closed = backbone(input_ids, target_ids)
        sampled = shim(input_ids, target_ids)
    torch.testing.assert_close(sampled, closed, rtol=1e-3, atol=1e-3)

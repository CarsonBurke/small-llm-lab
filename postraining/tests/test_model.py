from __future__ import annotations

import torch

from pretraining.fresh_lejepa.fresh_lejepa_train import FreshLeJEPAGPT, SIGReg


def tiny_model():
    return FreshLeJEPAGPT(
        vocab_size=32, num_layers=3, model_dim=32, num_heads=4, num_kv_heads=2,
        mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.01,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    )


def test_probe_losses_do_not_reach_backbone_but_attached_target_does():
    model = tiny_model()
    ids = torch.randint(0, 32, (2, 5))
    logits = model.policy_logits(ids)
    logits.sum().backward()
    assert model.tok_emb.weight.grad is None
    assert model.policy_probe.output.weight.grad is not None

    model.zero_grad(set_to_none=True)
    token, predicted = model.latent_features(ids)
    target = torch.nn.functional.rms_norm(model.tok_emb(ids.roll(-1, 1)), (32,))
    torch.nn.functional.mse_loss(predicted, target).backward()
    assert model.tok_emb.weight.grad is not None


def test_incremental_generation_matches_full_causal_features():
    model = tiny_model().eval()
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, std=0.05)
            torch.nn.init.normal_(block.mlp.proj.weight, std=0.05)
        torch.nn.init.normal_(model.policy_probe.output.weight, std=0.05)
        torch.nn.init.normal_(model.critic_probe.output.weight, std=0.05)
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        full_logits = model.policy_logits(ids)
        full_values = model.values(ids)
        caches = model.make_generation_cache(ids.size(0), ids.size(1), ids.device)
        steps = []
        value_steps = []
        for position in range(ids.size(1)):
            logits, values, caches = model.generation_step(ids[:, position], caches, position)
            steps.append(logits)
            value_steps.append(values)
        incremental = torch.stack(steps, dim=1)
        incremental_values = torch.stack(value_steps, dim=1)
    torch.testing.assert_close(incremental, full_logits, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(incremental_values, full_values, rtol=2e-4, atol=2e-4)


def test_sigreg_position_chunking_preserves_full_reduction():
    embeddings = torch.randn(3, 7, 8)
    regularizer = SIGReg(knots=5, num_proj=6, proj_chunk=6, position_chunk=2)
    torch.manual_seed(123)
    actual = regularizer(embeddings)

    torch.manual_seed(123)
    directions = torch.randn(8, 6)
    directions /= directions.norm(dim=0, keepdim=True)
    samples = embeddings.transpose(0, 1)
    x_t = (samples @ directions).unsqueeze(-1) * regularizer.t
    error = (x_t.cos().mean(1) - regularizer.phi).square()
    error += x_t.sin().mean(1).square()
    expected = ((error @ regularizer.weights) * embeddings.size(0)).mean()
    torch.testing.assert_close(actual, expected)

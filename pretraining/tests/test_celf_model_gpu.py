"""Real compiled CUDA contracts for CELF. Execute through mlq; no eager substitutes."""

import pytest
import torch

from pretraining.celf.config import LossConfig, ModelConfig
from pretraining.celf.evaluation import estimate_nelbo, generate_bytes, sample_blocks
from pretraining.celf.model import CelfModel
from pretraining.celf.objective import Objective, summarize_statistics
from pretraining.celf.training import (
    TrainConfig,
    component_parameters,
    make_optimizers,
    optimizer_step,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _config():
    return ModelConfig(
        patch_size=2,
        latent_dim=8,
        byte_dim=16,
        codec_dim=32,
        codec_layers=2,
        codec_heads=2,
        codec_ffn_dim=64,
        prior_dim=32,
        prior_layers=2,
        prior_heads=2,
        prior_rope_dim=8,
        block_patches=4,
        seq_bytes=64,
    )


def _activate(model):
    # Zero-initialized heads and gates would make dependence tests vacuous.
    with torch.no_grad():
        for block in model.prior.blocks:
            torch.nn.init.normal_(block.modulation.weight, std=0.05)
        torch.nn.init.normal_(model.prior.final_modulation.weight, std=0.05)
        torch.nn.init.normal_(model.prior.head.weight, std=0.05)
        torch.nn.init.normal_(model.decoder.head.weight, std=0.05)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    model = CelfModel(_config()).cuda()
    _activate(model)
    model.compile_components()
    return model


def test_objective_runs_and_every_parameter_receives_gradient(model):
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (2, model.config.seq_bytes), device="cuda")
    objective = Objective(model, LossConfig())
    model.zero_grad(set_to_none=True)
    loss, statistics = objective(ids)
    loss.backward()
    summary = summarize_statistics(statistics.cpu().tolist())
    assert all(map(torch.isfinite, (loss,)))
    assert summary["bytes"] == ids.numel()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_flow_loss_reaches_encoder_through_history_stream(model):
    torch.manual_seed(2)
    ids = torch.randint(0, 256, (2, model.config.seq_bytes), device="cuda")
    patches = ids.view(2, model.config.seq_patches, model.config.patch_size)
    z = model.encoder(patches, torch.zeros(patches.shape[:2], dtype=torch.bool, device="cuda"))
    history = model.anchor(z, torch.randn_like(z)).detach().requires_grad_(True)
    noisy = torch.randn_like(z)
    times = torch.full(z.shape[:2], 0.6, device="cuda")
    velocity = model.prior(history, noisy, times)
    (grad,) = torch.autograd.grad(velocity.square().sum(), history)
    blocks = model.config.block_patches
    # First block has no history; later blocks' velocities depend on earlier history.
    assert grad[:, :blocks].abs().sum().item() > 0  # block 0 history feeds block 1
    assert grad[:, -blocks:].abs().sum().item() == 0  # last block history feeds nothing


def test_cached_block_velocity_matches_two_stream_forward(model):
    torch.manual_seed(3)
    config = model.config
    batch = 2
    history = torch.randn(batch, config.seq_patches, config.latent_dim, device="cuda")
    noisy = torch.randn_like(history)
    times = torch.full(history.shape[:2], 0.7, device="cuda")
    with torch.no_grad():
        full = model.prior(history, noisy, times)
        cache = None
        for block in range(config.blocks):
            start = block * config.block_patches
            stop = start + config.block_patches
            cached = model.prior.block_velocity(noisy[:, start:stop], times[:, start:stop], cache)
            torch.testing.assert_close(cached, full[:, start:stop], atol=3e-2, rtol=3e-2)
            if block + 1 < config.blocks:
                cache = model.prior.append_history(history[:, start:stop], cache)


def test_flow_stream_is_blind_to_its_own_and_future_history(model):
    torch.manual_seed(4)
    config = model.config
    history = torch.randn(1, config.seq_patches, config.latent_dim, device="cuda")
    noisy = torch.randn_like(history)
    times = torch.full(history.shape[:2], 0.5, device="cuda")
    with torch.no_grad():
        base = model.prior(history, noisy, times)
        perturbed = history.clone()
        perturbed[:, config.block_patches :] += 3.0
        changed = model.prior(perturbed, noisy, times)
    torch.testing.assert_close(changed[:, : config.block_patches], base[:, : config.block_patches])
    assert not torch.allclose(changed[:, config.block_patches :], base[:, config.block_patches :])


def test_evaluation_paths_are_finite(model):
    torch.manual_seed(5)
    ids = torch.randint(0, 256, (2, model.config.seq_bytes), device="cuda")
    record = estimate_nelbo(model, ids, steps=3, seed=1, probes=2)
    assert record["negative_elbo_estimate_bpb"] > 0
    samples = sample_blocks(model, ids, steps=3, seed=1)
    assert 0 <= samples["sampled_byte_accuracy"] <= 1
    prompt = bytes(range(2 * model.config.block_bytes))
    generated = generate_bytes(model, prompt, new_bytes=2 * model.config.block_bytes, steps=2, seed=1)
    assert generated["generated_bytes"] == 2 * model.config.block_bytes
    with pytest.raises(ValueError):
        generate_bytes(model, prompt[:-1], new_bytes=0, steps=1, seed=1)


def _grads(model, prefix):
    return {n: p.grad.clone() for n, p in model.named_parameters() if n.startswith(prefix)}


def test_codec_stage_excludes_prior_entirely(model):
    torch.manual_seed(6)
    ids = torch.randint(0, 256, (2, model.config.seq_bytes), device="cuda")
    objective = Objective(model, LossConfig())
    model.zero_grad(set_to_none=True)
    loss, statistics = objective(ids, torch.Generator(device="cuda").manual_seed(0), flow=False)
    loss.backward()
    summary = summarize_statistics(statistics.cpu().tolist())
    assert summary["flow_mse"] == 0.0
    assert loss.item() == pytest.approx(summary["objective"], rel=1e-4)
    assert all(p.grad is None for p in model.prior.parameters())
    assert all(p.grad is not None for p in model.encoder.parameters())
    assert all(p.grad is not None for p in model.decoder.parameters())


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def test_history_attachment_changes_only_the_encoder_gradient(model):
    torch.manual_seed(7)
    ids = torch.randint(0, 256, (2, model.config.seq_bytes), device="cuda")
    grads = {}
    for name, loss_config in (
        ("all", LossConfig(attachment="all")),
        ("history", LossConfig(attachment="history")),
        ("codec_only", LossConfig(flow_weight=0.0)),
    ):
        objective = Objective(model, loss_config)
        model.zero_grad(set_to_none=True)
        loss, _ = objective(ids, torch.Generator(device="cuda").manual_seed(0))
        loss.backward()
        grads[name] = (loss.item(), _grads(model, "encoder."), _grads(model, "prior."), _grads(model, "decoder."))
    # Same forward pass (same generator), so the loss and the prior/decoder gradients
    # coincide up to BF16 kernel rounding across the two compiled graphs ...
    assert grads["all"][0] == pytest.approx(grads["history"][0], rel=1e-4)
    for component in (2, 3):
        for name, grad in grads["all"][component].items():
            other = grads["history"][component][name]
            assert _cosine(grad, other) > 0.999, name
            assert other.norm().item() == pytest.approx(grad.norm().item(), rel=2e-2), name
    def encoder_distance(a, b):
        return sum((grads[a][1][n] - grads[b][1][n]).abs().sum().item() for n in grads[a][1])
    # ... while the encoder loses the noisy-block and target paths of the flow loss ...
    assert encoder_distance("all", "history") > 0
    # ... but keeps the history path: 'history' is not the same as no flow gradient at all.
    assert encoder_distance("history", "codec_only") > 0
    assert all(p.grad is not None for p in model.encoder.parameters())


def test_component_optimizers_partition_and_stage_one_freezes_prior(model):
    components = component_parameters(model)
    codec_ids = {id(p) for p in components["codec"][0]}
    prior_ids = {id(p) for p in components["prior"][0]}
    assert not codec_ids & prior_ids
    assert codec_ids | prior_ids == {id(p) for p in model.parameters()}
    snapshot = {n: p.detach().clone() for n, p in model.state_dict().items()}
    try:
        config = TrainConfig(name="staged", steps=4, codec_steps=2, val_every=1, log_every=1, warmup_steps=1, sample_every=1, nelbo_every=1, microbatch=2, batch_sequences=2, val_sequences=2, sample_sequences=2, nelbo_sequences=2)
        optimizers = {name: make_optimizers(model, config, name) for name in ("codec", "prior")}
        torch.manual_seed(8)
        ids = torch.randint(0, 256, (2, model.config.seq_bytes), device="cuda")
        objective = Objective(model, LossConfig())
        # Stage 1: even a prior parameter that somehow carries a gradient must not move.
        model.zero_grad(set_to_none=True)
        loss, _ = objective(ids, flow=False)
        loss.backward()
        for p in model.prior.parameters():
            p.grad = torch.ones_like(p)
        before_prior = [p.detach().clone() for p in model.prior.parameters()]
        before_codec = [p.detach().clone() for p in model.encoder.parameters()]
        optimizer_step(model, optimizers, config, 0)
        assert all(torch.equal(a, b) for a, b in zip(before_prior, model.prior.parameters(), strict=True))
        assert any(not torch.equal(a, b) for a, b in zip(before_codec, model.encoder.parameters(), strict=True))
        # Stage 2: a codec-only backward is rejected, a joint backward moves the prior.
        model.zero_grad(set_to_none=True)
        loss, _ = objective(ids, flow=False)
        loss.backward()
        with pytest.raises(RuntimeError):
            optimizer_step(model, optimizers, config, 2)
        model.zero_grad(set_to_none=True)
        loss, _ = objective(ids, flow=True)
        loss.backward()
        before_prior = [p.detach().clone() for p in model.prior.parameters()]
        optimizer_step(model, optimizers, config, 2)
        assert any(not torch.equal(a, b) for a, b in zip(before_prior, model.prior.parameters(), strict=True))
    finally:
        model.zero_grad(set_to_none=True)
        model.load_state_dict(snapshot, strict=True)

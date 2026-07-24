from __future__ import annotations

import torch
import torch.nn.functional as F

import nanogpt_mini_model
from postraining.latent_thought import LatentThoughtModel
from postraining.model_io import fresh_trunk, load_model
from postraining.nano_backbone import NanoGPTBackbone, NanoTiedDotBackbone
from postraining.value_model import SeparateCritic

# head_dim is fixed at 128, so the smallest multi-head trunk is model_dim 256.
KWARGS = dict(vocab_size=64, num_layers=2, model_dim=256)


def _backbone(cls=NanoGPTBackbone, seed: int = 3):
    torch.manual_seed(seed)
    backbone = cls(**KWARGS).float().eval()
    with torch.no_grad():
        # Zero-init readouts make every logit comparison trivially 0 == 0;
        # give the renderer real weights so parity tests are meaningful.
        if isinstance(backbone, NanoTiedDotBackbone):
            backbone.readout_scale.fill_(0.5)
            backbone.readout_bias.normal_(std=0.1)
        else:
            backbone.proj.weight.normal_(std=0.05)
            backbone.proj.bias.normal_(std=0.05)
    return backbone


def _teacher_forced(wrapper, input_ids):
    backbone = wrapper.backbone
    token_latent = backbone.embed_tokens(input_ids)
    beliefs = backbone.temporal_belief_from_token_latent(token_latent)
    logits = backbone.logits_from_features(
        wrapper.renderer_features(token_latent, beliefs)
    )
    return beliefs, logits


@torch.no_grad()
def test_stepwise_matches_teacher_forced_int_positions():
    for cls in (NanoGPTBackbone, NanoTiedDotBackbone):
        wrapper = LatentThoughtModel(_backbone(cls)).eval()
        input_ids = torch.randint(0, KWARGS["vocab_size"], (2, 7), generator=torch.Generator().manual_seed(5))
        beliefs, logits = _teacher_forced(wrapper, input_ids)
        caches = wrapper.make_generation_cache(2, 7, torch.device("cpu"), dtype=torch.float32)
        for t in range(7):
            out = wrapper.token_step(input_ids[:, t], caches, t)
            torch.testing.assert_close(out.belief, beliefs[:, t], rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(out.logits, logits[:, t], rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_stepwise_matches_teacher_forced_tensor_positions():
    wrapper = LatentThoughtModel(_backbone()).eval()
    input_ids = torch.randint(0, KWARGS["vocab_size"], (2, 6), generator=torch.Generator().manual_seed(7))
    beliefs, logits = _teacher_forced(wrapper, input_ids)
    caches = wrapper.make_generation_cache(2, 6, torch.device("cpu"), dtype=torch.float32)
    for t in range(6):
        out = wrapper.token_step(input_ids[:, t], caches, torch.tensor(t))
        torch.testing.assert_close(out.belief, beliefs[:, t], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out.logits, logits[:, t], rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_stepwise_full_cache_key_mask():
    """The static-shape path: 1-D key mask over a zero-filled full cache."""
    wrapper = LatentThoughtModel(_backbone()).eval()
    input_ids = torch.randint(0, KWARGS["vocab_size"], (2, 6), generator=torch.Generator().manual_seed(9))
    beliefs, logits = _teacher_forced(wrapper, input_ids)
    cache_length = 10
    caches = wrapper.make_static_generation_cache(
        2, cache_length, torch.device("cpu"), dtype=torch.float32
    )
    key_mask = torch.zeros(cache_length, dtype=torch.bool)
    for t in range(6):
        key_mask[t] = True
        out = wrapper.token_step(input_ids[:, t], caches, torch.tensor(t), key_mask)
        torch.testing.assert_close(out.belief, beliefs[:, t], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out.logits, logits[:, t], rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_prefill_matches_stepwise():
    wrapper = LatentThoughtModel(_backbone()).eval()
    input_ids = torch.randint(0, KWARGS["vocab_size"], (2, 5), generator=torch.Generator().manual_seed(11))
    prefill_caches = wrapper.make_generation_cache(2, 8, torch.device("cpu"), dtype=torch.float32)
    prefill_out = wrapper.prefill(input_ids, prefill_caches)
    step_caches = wrapper.make_generation_cache(2, 8, torch.device("cpu"), dtype=torch.float32)
    for t in range(5):
        step_out = wrapper.token_step(input_ids[:, t], step_caches, t)
    torch.testing.assert_close(prefill_out.belief, step_out.belief, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(prefill_out.logits, step_out.logits, rtol=1e-4, atol=1e-4)
    for prefill_cache, step_cache in zip(prefill_caches, step_caches):
        torch.testing.assert_close(
            prefill_cache[0][:, :, :5], step_cache[0][:, :, :5], rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            prefill_cache[1][:, :, :5], step_cache[1][:, :, :5], rtol=1e-4, atol=1e-4
        )


@torch.no_grad()
def test_left_padded_rollout_matches_unpadded():
    """2-D key-mask stepping: padded rows reproduce their unpadded stream."""
    wrapper = LatentThoughtModel(_backbone()).eval()
    length, pad = 5, 3
    row = torch.randint(0, KWARGS["vocab_size"], (1, length), generator=torch.Generator().manual_seed(13))
    reference_caches = wrapper.make_generation_cache(1, length + 2, torch.device("cpu"), dtype=torch.float32)
    reference = wrapper.prefill(row, reference_caches)

    padded = torch.cat([torch.zeros(1, pad, dtype=row.dtype), row], dim=1)
    key_valid = torch.cat(
        [torch.zeros(1, pad, dtype=torch.bool), torch.ones(1, length, dtype=torch.bool)],
        dim=1,
    )
    total = pad + length + 2
    padded_caches = wrapper.make_generation_cache(1, total, torch.device("cpu"), dtype=torch.float32)
    for cache in padded_caches:
        for tensor in cache:
            tensor.zero_()
    padded_out = wrapper.prefill(padded, padded_caches, key_valid)
    torch.testing.assert_close(padded_out.belief, reference.belief, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(padded_out.logits, reference.logits, rtol=1e-4, atol=1e-4)

    # Continue one step through the per-row masked branch and compare with the
    # unpadded stepwise continuation.
    next_token = torch.tensor([1])
    key_mask = torch.cat([key_valid, torch.ones(1, 1, dtype=torch.bool)], dim=1)
    padded_step = wrapper.token_step(
        next_token, padded_caches, torch.tensor(pad + length), key_mask
    )
    reference_step = wrapper.token_step(next_token, reference_caches, length)
    torch.testing.assert_close(padded_step.belief, reference_step.belief, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(padded_step.logits, reference_step.logits, rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_renderer_matches_pretraining_forward():
    """Adapter-assembled sum-CE equals the pretraining class's forward."""
    for backbone_cls, pretraining_cls in (
        (NanoGPTBackbone, nanogpt_mini_model.GPT),
        (NanoTiedDotBackbone, nanogpt_mini_model.TiedDotGPT),
    ):
        backbone = _backbone(backbone_cls)
        pretraining = pretraining_cls(**KWARGS).float()
        pretraining.load_state_dict(backbone.state_dict(), strict=True)
        pretraining.eval()
        generator = torch.Generator().manual_seed(17)
        inputs = torch.randint(0, KWARGS["vocab_size"], (2, 6), generator=generator)
        targets = torch.randint(0, KWARGS["vocab_size"], (2, 6), generator=generator)
        _, logits = _teacher_forced(LatentThoughtModel(backbone), inputs)
        assembled = F.cross_entropy(
            logits.float().view(targets.numel(), -1), targets.view(-1), reduction="sum"
        )
        torch.testing.assert_close(
            assembled, pretraining(inputs, targets), rtol=1e-5, atol=1e-5
        )


def test_checkpoint_roundtrip(tmp_path):
    """Script-style payload -> load_model -> strict fp32-master backbone."""
    torch.manual_seed(23)
    pretraining = nanogpt_mini_model.GPT(**KWARGS)  # bf16 embed, as the scripts save
    payload = {
        "model": pretraining.state_dict(),
        "model_config": dict(KWARGS, mlp_hidden=4 * KWARGS["model_dim"]),
        "architecture": "nanogpt_mini_v1",
    }
    checkpoint = tmp_path / "final_model.pt"
    torch.save(payload, checkpoint)
    model = load_model(checkpoint, torch.device("cpu"))
    assert isinstance(model, NanoGPTBackbone)
    assert model.architecture == "nanogpt_mini_v1"
    assert model.tok_emb.weight.dtype == torch.float32
    assert all(not parameter.requires_grad for parameter in model.parameters())
    # bf16 -> fp32 promotion is value-exact.
    torch.testing.assert_close(
        model.embed.weight, pretraining.embed.weight.float(), rtol=0, atol=0
    )
    trunk = fresh_trunk(model, torch.device("cpu"))
    assert type(trunk) is NanoGPTBackbone
    assert trunk.model_config == model.model_config
    assert not torch.equal(trunk.embed.weight, model.embed.weight)


def test_tieddot_checkpoint_dispatch(tmp_path):
    torch.manual_seed(29)
    pretraining = nanogpt_mini_model.TiedDotGPT(**KWARGS)
    payload = {
        "model": pretraining.state_dict(),
        "model_config": dict(KWARGS, mlp_hidden=4 * KWARGS["model_dim"]),
        "architecture": "nanogpt_mini_tieddot_v1",
    }
    checkpoint = tmp_path / "final_model.pt"
    torch.save(payload, checkpoint)
    model = load_model(checkpoint, torch.device("cpu"))
    assert isinstance(model, NanoTiedDotBackbone)


def test_nano_critic_smoke():
    """SeparateCritic on a fresh nano trunk decodes the prior value."""
    torch.manual_seed(31)
    trunk = NanoGPTBackbone(**KWARGS).float().eval()
    critic = SeparateCritic(trunk, num_bins=17, prior_value=0.25).eval()
    from postraining.latent_rollout import LatentRolloutBatch, PAD_SLOT, TOKEN_SLOT

    batch_size, stream = 2, 4
    kind = torch.full((batch_size, stream), TOKEN_SLOT, dtype=torch.long)
    kind[1, -1] = PAD_SLOT
    zeros = torch.zeros(batch_size, stream)
    batch = LatentRolloutBatch(
        kind=kind,
        token_ids=torch.randint(0, KWARGS["vocab_size"], (batch_size, stream)),
        thoughts=torch.zeros(batch_size, stream, KWARGS["model_dim"]),
        gate_actions=torch.zeros(batch_size, stream, dtype=torch.long),
        action_mask=torch.zeros(batch_size, stream, dtype=torch.bool),
        gate_mask=torch.zeros(batch_size, stream, dtype=torch.bool),
        emit_mask=torch.zeros(batch_size, stream, dtype=torch.bool),
        old_gate_logprobs=zeros.clone(),
        old_token_logprobs=zeros.clone(),
        old_thought_logprobs=torch.zeros(batch_size, stream, KWARGS["model_dim"]),
        old_thought_means=torch.zeros(batch_size, stream, 0),
        old_thought_log_sigmas=torch.zeros(batch_size, stream, 0),
        old_values=zeros.clone(),
        rewards=zeros.clone(),
        reward_scalar=torch.zeros(batch_size),
        prompt_length=2,
    )
    with torch.no_grad():
        values = critic.values(batch)
    assert values.shape == (batch_size, stream)
    torch.testing.assert_close(
        values, torch.full_like(values, 0.25), rtol=0, atol=0.02
    )

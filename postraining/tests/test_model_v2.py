from __future__ import annotations

import copy

import torch

from fresh_lejepa_train_v2 import ARCHITECTURE, FreshLeJEPAGPTV2
from fresh_lejepa_train_v3 import ARCHITECTURE as V3_ARCHITECTURE, FreshLeJEPAGPTV3
from fresh_lejepa_train_v5_jedi_denoising import (
    ARCHITECTURE as JEDI_ARCHITECTURE,
    FreshLeJEPAV5JEDIDenoising,
)
from fresh_lejepa_train_v9_belief_transition_jedi import (
    ARCHITECTURE as V9_ARCHITECTURE,
    FreshLeJEPAV9BeliefTransition,
)
from postraining.model_io import DEFAULT_MODEL_CONFIG, load_model
from postraining.core import TrajectoryBatch
from postraining.train_vapo import update_step, update_step_accumulated


def tiny_v2():
    return FreshLeJEPAGPTV2(
        vocab_size=32, num_layers=3, model_dim=32, num_heads=4, num_kv_heads=2,
        mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.01,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    )


def test_v2_probe_is_compact_and_ce_stays_detached():
    model = FreshLeJEPAGPTV2(
        vocab_size=1024, num_layers=3, model_dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.01,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    )
    assert sum(p.numel() for p in model.policy_probe.parameters()) < 1_100_000
    assert sum(p.numel() for p in model.critic_probe.parameters()) < 1_100_000
    ids = torch.randint(0, 1024, (2, 5))
    model.policy_logits(ids).sum().backward()
    assert model.tok_emb.weight.grad is None
    assert model.policy_probe.token.weight.grad is not None


def test_v2_incremental_logits_and_values_match_full_sequence():
    model = tiny_v2().eval()
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, std=0.05)
            torch.nn.init.normal_(block.mlp.proj.weight, std=0.05)
        torch.nn.init.normal_(model.critic_probe.output.weight, std=0.05)
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        full_logits = model.policy_logits(ids)
        full_values = model.values(ids)
        caches = model.make_generation_cache(ids.size(0), ids.size(1), ids.device)
        logits_steps, value_steps = [], []
        for position in range(ids.size(1)):
            logits, values, caches = model.generation_step(ids[:, position], caches, position)
            logits_steps.append(logits)
            value_steps.append(values)
    torch.testing.assert_close(torch.stack(logits_steps, 1), full_logits, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(torch.stack(value_steps, 1), full_values, rtol=2e-4, atol=2e-4)


def test_v2_pretraining_and_rl_checkpoints_reconstruct_architecture(tmp_path):
    model = FreshLeJEPAGPTV2(**DEFAULT_MODEL_CONFIG)
    pretraining = tmp_path / "pretraining.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": {"architecture": ARCHITECTURE, "model": DEFAULT_MODEL_CONFIG},
        },
        pretraining,
    )
    loaded = load_model(pretraining, torch.device("cpu"))
    assert isinstance(loaded, FreshLeJEPAGPTV2)
    assert loaded.policy_probe.token.weight.dtype == torch.float32
    assert loaded.policy_probe.token.weight.requires_grad
    assert not loaded.tok_emb.weight.requires_grad

    rl = tmp_path / "rl.pt"
    torch.save(
        {
            "model": loaded.state_dict(),
            "model_config": DEFAULT_MODEL_CONFIG,
            "architecture": ARCHITECTURE,
        },
        rl,
    )
    resumed = load_model(rl, torch.device("cpu"))
    assert isinstance(resumed, FreshLeJEPAGPTV2)


def test_v2_policy_compiles_with_dynamic_sequence():
    model = tiny_v2().eval()
    compiled = torch.compile(model.policy_logits, dynamic=True, fullgraph=False, backend="eager")
    assert compiled(torch.randint(0, 32, (2, 4))).shape == (2, 4, 32)
    assert compiled(torch.randint(0, 32, (2, 6))).shape == (2, 6, 32)


def test_v3_checkpoint_selects_dropout_probe(tmp_path):
    model = FreshLeJEPAGPTV3(**DEFAULT_MODEL_CONFIG)
    assert model.policy_probe.dropout.p == 0.1
    checkpoint = tmp_path / "v3.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": {"architecture": V3_ARCHITECTURE, "model": DEFAULT_MODEL_CONFIG},
        },
        checkpoint,
    )
    loaded = load_model(checkpoint, torch.device("cpu"))
    assert isinstance(loaded, FreshLeJEPAGPTV3)
    assert loaded.policy_probe.dropout.p == 0.1


def test_jedi_checkpoint_reconstructs_for_vapo_without_training_denoiser(tmp_path):
    config = {
        **DEFAULT_MODEL_CONFIG,
        "vocab_size": 32,
        "num_layers": 3,
        "model_dim": 32,
        "num_heads": 4,
        "num_kv_heads": 2,
    }
    model = FreshLeJEPAV5JEDIDenoising(**config)
    checkpoint = tmp_path / "jedi.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": {"architecture": JEDI_ARCHITECTURE, "model": config},
        },
        checkpoint,
    )

    loaded = load_model(checkpoint, torch.device("cpu"))
    assert isinstance(loaded, FreshLeJEPAV5JEDIDenoising)
    assert all(not parameter.requires_grad for parameter in loaded.latent_denoiser.parameters())
    assert all(parameter.requires_grad for parameter in loaded.policy_probe.parameters())
    assert all(parameter.requires_grad for parameter in loaded.critic_probe.parameters())


def test_v9_checkpoint_reconstructs_with_frozen_transition(tmp_path):
    config = {
        **DEFAULT_MODEL_CONFIG,
        "vocab_size": 32,
        "num_layers": 3,
        "model_dim": 32,
        "num_heads": 4,
        "num_kv_heads": 2,
    }
    model = FreshLeJEPAV9BeliefTransition(**config)
    checkpoint = tmp_path / "v9.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": {"architecture": V9_ARCHITECTURE, "model": config},
        },
        checkpoint,
    )
    loaded = load_model(checkpoint, torch.device("cpu"))
    assert isinstance(loaded, FreshLeJEPAV9BeliefTransition)
    assert all(
        not parameter.requires_grad
        for parameter in loaded.belief_transition.parameters()
    )


def test_chunked_update_matches_single_chunk_parameter_delta():
    base = tiny_v2().eval()
    input_ids = torch.randint(0, 32, (2, 5))
    target_ids = torch.randint(0, 32, (2, 5))
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    with torch.no_grad():
        logits = base.policy_logits(input_ids).float()
        old_logprobs = logits.log_softmax(-1).gather(-1, target_ids[..., None]).squeeze(-1)
        old_values = base.values(input_ids).float()
    rewards = torch.zeros_like(old_values)
    rewards[0, 2], rewards[1, 4] = 1.0, -1.0
    batch = TrajectoryBatch(
        input_ids, target_ids, mask, old_logprobs, old_values, rewards,
        torch.tensor([True, False]), ["correct", "wrong"],
    )
    chunked, whole = copy.deepcopy(base), copy.deepcopy(base)

    def run(model, chunk):
        actor = torch.optim.SGD(model.policy_probe.parameters(), lr=1e-3)
        critic = torch.optim.SGD(model.critic_probe.parameters(), lr=1e-3)
        metrics = update_step(model, batch, actor, critic, token_chunk_size=chunk)
        params = [p.detach().clone() for p in model.policy_probe.parameters()]
        params += [p.detach().clone() for p in model.critic_probe.parameters()]
        return metrics, params

    chunked_metrics, chunked_params = run(chunked, 3)
    whole_metrics, whole_params = run(whole, 100)
    for key in ("loss", "policy_loss", "value_loss", "positive_lm_loss"):
        torch.testing.assert_close(
            torch.tensor(chunked_metrics[key]), torch.tensor(whole_metrics[key]),
            rtol=2e-5, atol=2e-5,
        )
    for actual, expected in zip(chunked_params, whole_params, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_trajectory_microbatch_accumulation_matches_full_update():
    base = tiny_v2().eval()
    input_ids = torch.randint(0, 32, (4, 5))
    target_ids = torch.randint(0, 32, (4, 5))
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        logits = base.policy_logits(input_ids).float()
        old_logprobs = logits.log_softmax(-1).gather(-1, target_ids[..., None]).squeeze(-1)
        old_values = base.values(input_ids).float()
    rewards = torch.zeros_like(old_values)
    for row, last in enumerate((2, 4, 1, 3)):
        rewards[row, last] = 1.0 if row in (0, 3) else -1.0
    batch = TrajectoryBatch(
        input_ids, target_ids, mask, old_logprobs, old_values, rewards,
        torch.tensor([True, False, False, True]), ["a", "b", "c", "d"],
    )
    accumulated, whole = copy.deepcopy(base), copy.deepcopy(base)

    def run(model, microbatch):
        actor = torch.optim.SGD(model.policy_probe.parameters(), lr=1e-3)
        critic = torch.optim.SGD(model.critic_probe.parameters(), lr=1e-3)
        metrics = update_step_accumulated(
            model, batch, actor, critic, token_chunk_size=4,
            microbatch_trajectories=microbatch,
        )
        parameters = [p.detach().clone() for p in model.policy_probe.parameters()]
        parameters += [p.detach().clone() for p in model.critic_probe.parameters()]
        return metrics, parameters

    accumulated_metrics, accumulated_parameters = run(accumulated, 1)
    whole_metrics, whole_parameters = run(whole, 4)
    for key in ("loss", "policy_loss", "value_loss", "positive_lm_loss"):
        torch.testing.assert_close(
            torch.tensor(accumulated_metrics[key]), torch.tensor(whole_metrics[key]),
            rtol=3e-5, atol=3e-5,
        )
    for actual, expected in zip(accumulated_parameters, whole_parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)

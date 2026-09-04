from __future__ import annotations

import hashlib
import random

import pytest
import torch

from postraining.minicpm_vapo import DEFAULT_LORA_TARGETS, NextLatAuxiliaryHead
from postraining.train_minicpm_vapo import build_parser as build_trainer_parser
from scripts import preflight_minicpm_vapo as preflight
from scripts.migrate_minicpm_vapo_v6 import migrate_checkpoint


def _optimizer_group(parameters: int, lr: float) -> dict:
    return {
        "lr": lr,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": True,
        "params": list(range(parameters)),
    }


def _v4_checkpoint() -> dict:
    adapter = {
        "layer.lora_a": torch.randn(2, 4),
        "layer.lora_b": torch.randn(4, 2),
    }
    critic = {
        "norm.weight": torch.ones(8),
        "input.weight": torch.randn(4, 8),
        "input.bias": torch.zeros(4),
        "output.weight": torch.randn(1, 4),
        "output.bias": torch.zeros(1),
    }
    return {
        "policy": {
            "schema": "minicpm5_vapo_adapter/v4",
            "model_id": "model",
            "revision": "revision",
            "lora_config": {"rank": 2, "alpha": 4.0, "targets": ("q_proj",)},
            "lora_modules": ["q_proj"],
            "adapter": adapter,
            "critic": critic,
            "nextlat_projection_factor": 1.6,
            "nextlat_head": NextLatAuxiliaryHead(8, 1.6).state_dict(),
        },
        "actor_optimizer": {
            "state": {0: {"step": torch.tensor(3)}},
            "param_groups": [_optimizer_group(2, 2e-5)],
        },
        "critic_optimizer": {
            "state": {0: {"step": torch.tensor(3)}},
            "param_groups": [_optimizer_group(5, 1e-4)],
        },
        "nextlat_optimizer": {
            "state": {1: {"step": torch.tensor(7)}},
            "param_groups": [_optimizer_group(4, 4e-4)],
        },
        "step": 203,
        "cursor": 1012,
        "warmup_step": 10,
        "pending_records": None,
        "pending_epoch": 0,
        "data_sha256": "0" * 64,
        "args": {
            "actor_lr": 2e-5,
            "critic_lr": 1e-4,
            "nextlat_lr": 4e-4,
            "nextlat_horizon": 2,
            "nextlat_projection_factor": 1.6,
            "nextlat_draft_length": 2,
            "nextlat_rollout": False,
            "nextlat_kl_tokens": 64,
            "nextlat_mse_coefficient": 1.0,
            "nextlat_kl_coefficient": 1.0,
            "train_nextlat": False,
        },
    }


def test_v6_migration_preserves_actor_state_and_builds_independent_critic() -> None:
    migrated = migrate_checkpoint(_v4_checkpoint(), seed=7)
    payload = migrated["policy"]
    assert payload["schema"] == "minicpm5_vapo_adapter/v6"
    actor = payload["actor"]
    critic = payload["critic"]
    assert actor["nextlat_projection_factor"] == 1.6
    assert critic["nextlat_projection_factor"] == 1.6
    assert set(actor["adapter"]) == set(critic["adapter"])
    for name in actor["adapter"]:
        torch.testing.assert_close(actor["adapter"][name], critic["adapter"][name])
        assert actor["adapter"][name].data_ptr() != critic["adapter"][name].data_ptr()
    assert "value_head" not in actor
    assert "value_head" in critic
    assert set(actor["nextlat"]) == set(critic["nextlat"])
    assert "nextlat_optimizer" not in migrated
    assert 0 in migrated["actor_optimizer"]["state"]
    assert int(migrated["actor_optimizer"]["state"][3]["step"]) == 7
    assert 2 in migrated["critic_optimizer"]["state"]
    assert len(migrated["actor_optimizer"]["param_groups"]) == 2
    assert len(migrated["critic_optimizer"]["param_groups"]) == 2
    assert {
        group["lr"] for group in migrated["actor_optimizer"]["param_groups"]
    } == {1e-6}
    assert {
        group["lr"] for group in migrated["critic_optimizer"]["param_groups"]
    } == {2e-6}
    assert migrated["args"]["optimizer_minibatches"] == 4
    assert migrated["args"]["gradient_clip_norm"] == 1.0
    assert migrated["args"]["train_nextlat"] is True
    assert migrated["args"]["nextlat_horizon"] == 2
    assert migrated["args"]["nextlat_samples"] == 64
    assert migrated["args"]["nextlat_mse_coefficient"] == 1.0
    assert migrated["args"]["nextlat_kl_coefficient"] == 1.0
    assert migrated["args"]["actor_lr"] == 1e-6
    assert migrated["args"]["critic_lr"] == 2e-6
    assert "nextlat_lr" not in migrated["args"]
    assert "positive_coefficient" not in migrated["args"]
    assert migrated["args"]["prompts_per_rollout"] == 4
    assert migrated["args"]["top_k"] == 20
    assert migrated["args"]["fast_rollout"] is True
    assert "rollout_physical_prompts" not in migrated["args"]
    assert migrated["args"]["replay_checkpoint_interval"] == 2
    assert migrated["args"]["replay_token_budget"] == 8_192
    assert migrated["args"]["replay_attention_backend"] == "sdpa"
    assert migrated["args"]["replay_max_trajectories"] == 16
    assert migrated["args"]["max_new_tokens"] == 4_096


def test_v6_migration_requires_completed_rollout_boundary() -> None:
    source = _v4_checkpoint()
    source["pending_records"] = [object()]
    with pytest.raises(ValueError, match="completed rollout boundary"):
        migrate_checkpoint(source)


def _preflight_checkpoint(tmp_path):
    data_path = tmp_path / "data.parquet"
    data_path.write_bytes(b"fixed dataset bytes")
    checkpoint = migrate_checkpoint(_v4_checkpoint(), seed=7)
    checkpoint["data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    checkpoint["cpu_rng"] = torch.get_rng_state()
    checkpoint["cuda_rng"] = torch.get_rng_state().clone()
    checkpoint["python_rng"] = random.getstate()
    saved_args = vars(build_trainer_parser().parse_args([]))
    saved_args.update(checkpoint["args"])
    saved_args.update(
        data=str(data_path),
        model="model",
        revision="revision",
        prompts_per_rollout=4,
        samples_per_prompt=16,
        value_warmup_steps=10,
        ppo_epochs=1,
        temperature=0.8,
        seed=7,
        lora_rank=2,
        lora_alpha=4.0,
    )
    expected_lora = {
        "rank": 2,
        "alpha": 4.0,
        "targets": tuple(DEFAULT_LORA_TARGETS),
    }
    checkpoint["policy"]["actor"]["lora_config"] = expected_lora
    checkpoint["policy"]["critic"]["lora_config"] = expected_lora.copy()
    checkpoint["args"] = saved_args
    resume_path = tmp_path / "source" / "vapo_adapter_checkpoint.pt"
    resume_path.parent.mkdir()
    torch.save(checkpoint, resume_path)
    return checkpoint, resume_path, data_path


def test_preflight_resolves_additional_steps_and_builds_queue_command(tmp_path) -> None:
    checkpoint, resume_path, data_path = _preflight_checkpoint(tmp_path)
    output_path = tmp_path / "continuation"
    report = preflight.inspect_continuation(
        checkpoint,
        resume_path=resume_path,
        output_path=output_path,
        data_path=data_path,
        target_steps=None,
        additional_steps=200,
        job_name="balanced continuation",
        time_limit="10h",
    )
    assert report["saved_actor_step"] == 203
    assert report["target_actor_step"] == 403
    assert report["remaining_actor_updates"] == 200
    assert report["rollout_trajectories"] == 64
    assert report["output_state"] == "new"
    assert "--max-parallel-runs 1" in report["queue_command"]
    assert "--steps 403" in report["queue_command"]
    assert "--name balanced-continuation" in report["queue_command"]
    assert "--temperature 0.8" in report["queue_command"]
    assert "--seed 7" in report["queue_command"]


def test_preflight_rejects_noop_target_before_launch(tmp_path) -> None:
    checkpoint, resume_path, data_path = _preflight_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="must exceed saved actor step"):
        preflight.inspect_continuation(
            checkpoint,
            resume_path=resume_path,
            output_path=tmp_path / "continuation",
            data_path=data_path,
            target_steps=203,
            additional_steps=None,
            job_name=None,
            time_limit="10h",
        )


def test_preflight_refuses_unowned_nonempty_output(tmp_path) -> None:
    checkpoint, resume_path, data_path = _preflight_checkpoint(tmp_path)
    output_path = tmp_path / "unrelated"
    output_path.mkdir()
    (output_path / "events").write_text("existing run")
    with pytest.raises(ValueError, match="refusing to mix"):
        preflight.inspect_continuation(
            checkpoint,
            resume_path=resume_path,
            output_path=output_path,
            data_path=data_path,
            target_steps=None,
            additional_steps=1,
            job_name=None,
            time_limit="10h",
        )


def test_preflight_preserves_saved_relative_dataset_argument(
    tmp_path, monkeypatch
) -> None:
    checkpoint, resume_path, data_path = _preflight_checkpoint(tmp_path)
    checkpoint["args"]["data"] = "data.parquet"
    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    report = preflight.inspect_continuation(
        checkpoint,
        resume_path=resume_path,
        output_path=tmp_path / "continuation",
        data_path=data_path,
        target_steps=None,
        additional_steps=1,
        job_name=None,
        time_limit="10h",
    )
    assert "--data data.parquet" in report["queue_command"]


def test_preflight_rejects_inconsistent_pending_epoch(tmp_path) -> None:
    checkpoint, resume_path, data_path = _preflight_checkpoint(tmp_path)
    checkpoint["pending_epoch"] = 1
    with pytest.raises(ValueError, match="pending epoch without replay records"):
        preflight.inspect_continuation(
            checkpoint,
            resume_path=resume_path,
            output_path=tmp_path / "continuation",
            data_path=data_path,
            target_steps=None,
            additional_steps=1,
            job_name=None,
            time_limit="10h",
        )


def test_preflight_rejects_truncated_v6_payload(tmp_path) -> None:
    checkpoint, resume_path, data_path = _preflight_checkpoint(tmp_path)
    del checkpoint["policy"]["critic"]["value_head"]
    with pytest.raises(ValueError, match="critic is missing: value_head"):
        preflight.inspect_continuation(
            checkpoint,
            resume_path=resume_path,
            output_path=tmp_path / "continuation",
            data_path=data_path,
            target_steps=None,
            additional_steps=1,
            job_name=None,
            time_limit="10h",
        )


def test_preflight_rejects_empty_pending_replay(tmp_path) -> None:
    checkpoint, resume_path, data_path = _preflight_checkpoint(tmp_path)
    checkpoint["pending_records"] = []
    with pytest.raises(ValueError, match="pending records cannot be empty"):
        preflight.inspect_continuation(
            checkpoint,
            resume_path=resume_path,
            output_path=tmp_path / "continuation",
            data_path=data_path,
            target_steps=None,
            additional_steps=1,
            job_name=None,
            time_limit="10h",
        )

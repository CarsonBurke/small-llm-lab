"""Convert the stopped MiniCPM VAPO v4 checkpoint to the optimized v6 schema."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch

from checkpointing import atomic_torch_save
from postraining.minicpm_vapo import NextLatAuxiliaryHead


def _append_parameter_group(
    optimizer: dict[str, Any],
    second_optimizer: dict[str, Any],
    *,
    first_parameters: int,
    second_parameters: int,
    learning_rate: float,
) -> dict[str, Any]:
    migrated = deepcopy(optimizer)
    first = migrated["param_groups"][0]
    if len(first["params"]) != first_parameters:
        raise ValueError("optimizer parameter count differs from checkpoint payload")
    first["params"] = list(range(first_parameters))
    first["lr"] = learning_rate
    second_source = deepcopy(second_optimizer)
    second = second_source["param_groups"][0]
    old_second_ids = list(second["params"])
    if len(old_second_ids) != second_parameters:
        raise ValueError("NextLat optimizer does not match the dynamics head")
    second["params"] = list(
        range(first_parameters, first_parameters + second_parameters)
    )
    second["lr"] = learning_rate
    second["weight_decay"] = 0.1
    second["betas"] = (0.9, 0.95)
    migrated["state"].update(
        {
            first_parameters + index: second_source["state"][parameter_id]
            for index, parameter_id in enumerate(old_second_ids)
            if parameter_id in second_source["state"]
        }
    )
    migrated["param_groups"] = [first, second]
    return migrated


def _expand_critic_optimizer(
    optimizer: dict[str, Any],
    group_source: dict[str, Any],
    *,
    adapter_parameters: int,
    value_parameters: int,
    nextlat_parameters: int,
    critic_lr: float,
) -> dict[str, Any]:
    source = deepcopy(optimizer)
    source_group = source["param_groups"][0]
    old_value_ids = list(source_group["params"])
    if len(old_value_ids) != value_parameters:
        raise ValueError("critic optimizer does not match the value head")
    remapped_state = {
        adapter_parameters + index: source["state"][parameter_id]
        for index, parameter_id in enumerate(old_value_ids)
        if parameter_id in source["state"]
    }
    first_count = adapter_parameters + value_parameters
    first = deepcopy(source_group)
    first["params"] = list(range(first_count))
    first["lr"] = critic_lr
    second = deepcopy(group_source)
    second["params"] = list(
        range(first_count, first_count + nextlat_parameters)
    )
    second["lr"] = critic_lr
    second["weight_decay"] = 0.1
    second["betas"] = (0.9, 0.95)
    return {"state": remapped_state, "param_groups": [first, second]}


def migrate_checkpoint(source: dict[str, Any], *, seed: int = 1337) -> dict[str, Any]:
    payload = source.get("policy")
    if not isinstance(payload, dict) or payload.get("schema") != "minicpm5_vapo_adapter/v4":
        raise ValueError("source checkpoint must use the v4 MiniCPM policy schema")
    if source.get("pending_records") is not None or int(source.get("pending_epoch", 0)):
        raise ValueError("checkpoint migration requires a completed rollout boundary")
    data_sha256 = source.get("data_sha256")
    if not isinstance(data_sha256, str) or len(data_sha256) != 64:
        raise ValueError("source checkpoint is missing its dataset fingerprint")
    critic_state_raw = payload.get("critic")
    adapter_state_raw = payload.get("adapter")
    actor_nextlat_state_raw = payload.get("nextlat_head")
    if not all(
        isinstance(value, dict)
        for value in (critic_state_raw, adapter_state_raw, actor_nextlat_state_raw)
    ):
        raise ValueError("v4 adapter, value head, or NextLat state is missing")
    critic_state = cast(dict[str, torch.Tensor], critic_state_raw)
    adapter_state = cast(dict[str, torch.Tensor], adapter_state_raw)
    actor_nextlat_state = cast(dict[str, torch.Tensor], actor_nextlat_state_raw)
    hidden_size = int(critic_state["norm.weight"].numel())
    projection_factor = float(payload["nextlat_projection_factor"])
    reference_head = NextLatAuxiliaryHead(hidden_size, projection_factor)
    if reference_head.state_dict().keys() != actor_nextlat_state.keys():
        raise ValueError("v4 NextLat head does not match the reference architecture")
    actor_nextlat = {
        name: tensor.detach().cpu().clone()
        for name, tensor in actor_nextlat_state.items()
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        critic_nextlat_module = NextLatAuxiliaryHead(hidden_size, projection_factor)
    critic_nextlat = {
        name: tensor.detach().cpu()
        for name, tensor in critic_nextlat_module.state_dict().items()
    }
    common = {
        "model_id": payload["model_id"],
        "revision": payload["revision"],
        "lora_config": deepcopy(payload["lora_config"]),
        "lora_modules": deepcopy(payload["lora_modules"]),
        "nextlat_projection_factor": projection_factor,
    }
    actor_payload = {
        **common,
        "adapter": {
            name: tensor.detach().cpu().clone()
            for name, tensor in adapter_state.items()
        },
        "nextlat": actor_nextlat,
    }
    critic_payload = {
        **common,
        "adapter": {
            name: tensor.detach().cpu().clone()
            for name, tensor in adapter_state.items()
        },
        "value_head": {
            name: tensor.detach().cpu().clone()
            for name, tensor in critic_state.items()
        },
        "nextlat": critic_nextlat,
    }
    args = deepcopy(source["args"])
    for obsolete in (
        "nextlat_draft_length",
        "nextlat_rollout",
        "nextlat_kl_tokens",
        "nextlat_width",
        "nextlat_tokens",
        "nextlat_coefficient",
        "rollout_physical_prompts",
        "replay_gradient_checkpointing",
        "nextlat_lr",
        "positive_coefficient",
    ):
        args.pop(obsolete, None)
    args.update(
        actor_lr=1e-6,
        critic_lr=2e-6,
        prompts_per_rollout=4,
        nextlat_horizon=2,
        nextlat_projection_factor=projection_factor,
        nextlat_samples=64,
        nextlat_kl_chunk_tokens=16,
        nextlat_mse_coefficient=1.0,
        nextlat_kl_coefficient=1.0,
        train_nextlat=True,
        gradient_clip_norm=1.0,
        optimizer_minibatches=4,
        replay_checkpoint_interval=2,
        replay_token_budget=8_192,
        replay_attention_backend="sdpa",
        replay_max_trajectories=16,
        max_new_tokens=4_096,
        post_update_kl_interval=10,
    )
    args.setdefault("top_k", 20)
    args.setdefault("fast_rollout", True)
    actor_optimizer = _append_parameter_group(
        source["actor_optimizer"],
        source["nextlat_optimizer"],
        first_parameters=len(actor_payload["adapter"]),
        second_parameters=len(actor_payload["nextlat"]),
        learning_rate=float(args["actor_lr"]),
    )
    nextlat_source_group = source["nextlat_optimizer"]["param_groups"][0]
    critic_optimizer = _expand_critic_optimizer(
        source["critic_optimizer"],
        nextlat_source_group,
        adapter_parameters=len(critic_payload["adapter"]),
        value_parameters=len(critic_payload["value_head"]),
        nextlat_parameters=len(critic_payload["nextlat"]),
        critic_lr=float(args["critic_lr"]),
    )
    migrated = deepcopy(source)
    migrated["policy"] = {
        "schema": "minicpm5_vapo_adapter/v6",
        "actor": actor_payload,
        "critic": critic_payload,
    }
    migrated["actor_optimizer"] = actor_optimizer
    migrated["critic_optimizer"] = critic_optimizer
    migrated.pop("nextlat_optimizer", None)
    migrated["args"] = args
    migrated["v6_migration"] = {
        "source_schema": "minicpm5_vapo_adapter/v4",
        "critic_adapter_initialized_from_actor": True,
        "immutable_backbone_shared_at_runtime": True,
        "optimizer_state_preserved": True,
        "actor_nextlat_preserved": True,
        "critic_nextlat_initialized": True,
        "seed": seed,
    }
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("migration output must differ from its source")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(source, dict):
        raise ValueError("source checkpoint must be a dictionary")
    atomic_torch_save(migrate_checkpoint(source, seed=args.seed), output_path)


if __name__ == "__main__":
    main()

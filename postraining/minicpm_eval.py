"""Checkpoint-owned protocol and actor-only loading for MiniCPM evaluation.

Metadata resolution intentionally imports no model/runtime modules. Supported
training payloads have ``args`` and ``policy = {schema, actor, ...}``; critic,
optimizer and pending-rollout state are never needed for evaluation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


_NATIVE_SCHEMA = "minicpm5_vapo_adapter/v6"
_CARRY_SCHEMA = "minicpm5_vapo_token_carry/v4"
_SLOT_SCHEMA = "minicpm5_vapo_slot_memory/v1"
_CARRY_COMBINER_KEYS = frozenset({"token_delta.weight", "carry.weight", "scale"})
_SLOT_COMBINER_KEYS = frozenset({
    "token_delta.weight", "query_token.weight", "query_carry.weight", "key.weight",
    "value.weight", "output.weight", "null_key", "scale",
})
_SLOT_HEAD_KEYS = frozenset({"projection.weight", "projection.bias"})
_SLOT_GEOMETRY_ARGS = {"slots": "slot_memory_slots", "heads": "slot_memory_heads", "head_dim": "slot_memory_head_dim"}
_PROTOCOL_FIELDS = (
    "model", "revision", "thinking", "prompt_tokens", "max_new_tokens", "context_tokens",
    "samples_per_prompt", "prompts_per_rollout", "rollout_physical_batch_size",
    "temperature", "top_k", "top_p", "prompt_suffix", "answer_reserve_tokens", "seed",
)


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {name} must be a mapping")
    return value


def _integer(value: Any, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _positive_float(value: Any, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _checkpoint_actor(checkpoint: dict) -> tuple[dict, dict]:
    checkpoint = _mapping(checkpoint, "root")
    payload = _mapping(checkpoint.get("policy"), "policy")
    schema = payload.get("schema")
    if schema not in (_NATIVE_SCHEMA, _CARRY_SCHEMA, _SLOT_SCHEMA):
        raise ValueError(f"unsupported MiniCPM evaluation schema: {schema!r}")
    actor = _mapping(payload.get("actor"), "actor")
    args = _mapping(checkpoint.get("args"), "args")
    missing = [name for name in _PROTOCOL_FIELDS if name != "context_tokens" and name not in args]
    if missing:
        raise ValueError(f"checkpoint args missing evaluation protocol fields: {missing}")
    for name in ("model_id", "revision"):
        if not isinstance(actor.get(name), str) or not actor[name].strip():
            raise ValueError(f"actor {name} must be a nonempty string")
    if actor["model_id"] != args["model"] or actor["revision"] != args["revision"]:
        raise ValueError("actor base identity differs from saved args")
    carry = schema in (_CARRY_SCHEMA, _SLOT_SCHEMA)
    slot = schema == _SLOT_SCHEMA
    for source, label in ((actor, "actor"), (args, "args")):
        if type(source.get("token_carry", False)) is not bool or source.get("token_carry", False) != carry:
            raise ValueError(f"{label} token_carry differs from checkpoint schema")
        if source.get("latent_thinking", False) is not False:
            raise ValueError("Gaussian latent-thinking checkpoints are unsupported")
        if source.get("uno_rollout", False) is not False or source.get("uno_checkpoint") is not None:
            raise ValueError("Uno checkpoints are unsupported")
    if type(args.get("slot_memory", False)) is not bool or args.get("slot_memory", False) != slot:
        raise ValueError("args slot_memory differs from checkpoint schema")
    if (actor.get("slot_memory") is not None) != slot or ("slot_head" in actor) != slot:
        raise ValueError("actor slot-memory state differs from checkpoint schema")
    if slot:
        from postraining.slot_memory import SlotMemoryConfig

        geometry = SlotMemoryConfig.from_payload(_mapping(actor["slot_memory"], "actor.slot_memory"))
        for field, name in _SLOT_GEOMETRY_ARGS.items():
            if args.get(name) != getattr(geometry, field):
                raise ValueError(f"actor slot-memory {field} differs from saved args")
        if set(_mapping(actor["slot_head"], "actor.slot_head")) != _SLOT_HEAD_KEYS:
            raise ValueError("actor slot_head does not match slot memory v1")
    if any(name in actor for name in ("transition", "thinking_gate", "thought_adapter")):
        raise ValueError("actor contains unsupported Gaussian state")
    if carry:
        combiner = _mapping(actor.get("token_combiner"), "actor.token_combiner")
        if set(combiner) != (_SLOT_COMBINER_KEYS if slot else _CARRY_COMBINER_KEYS):
            raise ValueError(f"actor token_combiner does not match {'slot memory v1' if slot else 'carry v4'}")
    elif "token_combiner" in actor:
        raise ValueError("native checkpoint contains token-carry state")
    lora = _mapping(actor.get("lora_config"), "actor.lora_config")
    _integer(lora.get("rank"), "LoRA rank", 1)
    _positive_float(lora.get("alpha"), "LoRA alpha")
    targets = lora.get("targets")
    if (
        not isinstance(targets, (tuple, list)) or not targets
        or any(not isinstance(name, str) or not name for name in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("LoRA targets must be nonempty unique strings")
    initialization = lora.get("initialization", "standard")
    if initialization not in ("standard", "nora"):
        raise ValueError("unsupported LoRA initialization")
    for name, value in (("lora_rank", lora["rank"]), ("lora_alpha", lora["alpha"])):
        if name not in args or args[name] != value:
            raise ValueError(f"actor {name} differs from saved args")
    if args.get("lora_initialization", "standard") != initialization:
        raise ValueError("actor LoRA initialization differs from saved args")
    factor = actor.get("nextlat_projection_factor")
    _positive_float(factor, "NextLat projection factor")
    if args.get("nextlat_projection_factor") != factor:
        raise ValueError("actor NextLat projection factor differs from saved args")
    modules = actor.get("lora_modules")
    if (
        not isinstance(modules, (tuple, list)) or not modules
        or any(not isinstance(name, str) or not name for name in modules)
        or len(set(modules)) != len(modules)
    ):
        raise ValueError("actor lora_modules must be nonempty unique strings")
    adapter = _mapping(actor.get("adapter"), "actor.adapter")
    if set(adapter) != {f"{name}.{suffix}" for name in modules for suffix in ("lora_a", "lora_b")}:
        raise ValueError("actor adapter keys differ from lora_modules")
    if not _mapping(actor.get("nextlat"), "actor.nextlat"):
        raise ValueError("actor NextLat state is missing")
    return actor, args


def resolve_evaluation_config(checkpoint: dict, overrides: dict) -> dict:
    """Inherit every saved protocol option; only non-None overrides take effect.

    Historical checkpoints may omit context_tokens (response-only limits).
    Other protocol fields are required even if overridden. Model and revision
    must agree with actor metadata and cannot select a different base model.
    The same returned protocol is used for both stock and trained actors.
    """
    actor, args = _checkpoint_actor(checkpoint)
    overrides = _mapping(overrides, "overrides")
    unknown = overrides.keys() - set(_PROTOCOL_FIELDS)
    if unknown:
        raise ValueError(f"unknown evaluation overrides: {sorted(unknown)}")
    config = {name: args.get(name) if overrides.get(name) is None else overrides[name] for name in _PROTOCOL_FIELDS}
    if config["model"] != actor["model_id"] or config["revision"] != actor["revision"]:
        raise ValueError("evaluation cannot override actor base identity")
    config["model"], config["revision"] = actor["model_id"], actor["revision"]
    if type(config["thinking"]) is not bool:
        raise ValueError("thinking must be a boolean")
    if not isinstance(config["prompt_suffix"], str):
        raise ValueError("prompt_suffix must be a string")
    for name in ("prompt_tokens", "max_new_tokens", "samples_per_prompt", "prompts_per_rollout"):
        _integer(config[name], name, 1)
    for name in ("rollout_physical_batch_size", "answer_reserve_tokens"):
        _integer(config[name], name, 0)
    _integer(config["seed"], "seed", -(2**63))
    if config["seed"] >= 2**64:
        raise ValueError("seed must fit torch.manual_seed's 64-bit range")
    trajectories = config["samples_per_prompt"] * config["prompts_per_rollout"]
    if config["rollout_physical_batch_size"] > trajectories:
        raise ValueError("physical rollout batch cannot exceed logical trajectories")
    reserve = config["answer_reserve_tokens"]
    if reserve and (not config["thinking"] or config["max_new_tokens"] <= reserve + 1):
        raise ValueError("answer reserve requires thinking and room for thinking plus its delimiter")
    context = config["context_tokens"]
    if context is not None:
        _integer(context, "context_tokens", 1)
        remaining = min(config["max_new_tokens"], context - 1)
        if remaining < 1 or (reserve and remaining <= reserve + 1):
            raise ValueError("context_tokens must leave response and answer reserve after a nonempty prompt")
    for name in ("temperature", "top_p"):
        _positive_float(config[name], name)
    if config["top_p"] > 1:
        raise ValueError("top_p must lie in (0, 1]")
    top_k = config["top_k"]
    if type(top_k) is not int or not (top_k == -1 or 1 <= top_k <= 130_560):
        raise ValueError("top_k must be -1 or positive and fit the MiniCPM vocabulary")
    if top_k == -1 and config["top_p"] != 1:
        raise ValueError("unrestricted top_k requires top_p=1")
    return config


def load_evaluation_policy(
    checkpoint: dict, *, stock: bool, device: torch.device
) -> tuple[Any, Any]:
    """Load just the actor on CUDA, or its zero-effect-LoRA pretrained baseline."""
    if device.type != "cuda":
        raise RuntimeError("MiniCPM evaluation requires CUDA; CPU model execution is unsupported")
    if type(stock) is not bool:
        raise ValueError("stock must be a boolean")
    actor, _ = _checkpoint_actor(checkpoint)
    from postraining.vapo.policy import VAPOPolicy
    from postraining.vapo.model.lora import (
        LoRAConfig,
        load_adapter_state_dict,
    )
    from postraining.slot_memory import SlotMemoryConfig

    lora = actor["lora_config"]
    slot_memory = None
    if not stock and actor.get("slot_memory") is not None:
        slot_memory = SlotMemoryConfig.from_payload(actor["slot_memory"])
    policy, tokenizer = VAPOPolicy.from_family("minicpm5", 
        model_id=actor["model_id"],
        revision=actor["revision"],
        device=device,
        lora_config=LoRAConfig(
            rank=lora["rank"], alpha=lora["alpha"], targets=tuple(lora["targets"]),
            initialization="standard" if stock else lora.get("initialization", "standard"),
        ),
        nextlat_projection_factor=actor["nextlat_projection_factor"],
        gradient_checkpointing=False,
        latent_thinking=False,
        token_carry=not stock and actor.get("token_carry", False),
        slot_memory=slot_memory,
    )
    if tuple(actor["lora_modules"]) != policy.lora_modules:
        raise ValueError("actor LoRA module topology differs from pretrained base")
    if not stock:
        parameters = dict(policy.causal_lm.named_parameters())
        for name, tensor in actor["adapter"].items():
            if tensor.shape != parameters[name].shape:
                raise ValueError(f"actor adapter shape differs for {name}")
        load_adapter_state_dict(policy.causal_lm, actor["adapter"])
        policy.nextlat_head.load_state_dict(actor["nextlat"], strict=True)
        policy.load_token_carry_state_dict(actor)
    policy.requires_grad_(False)
    policy.eval()
    return policy, tokenizer

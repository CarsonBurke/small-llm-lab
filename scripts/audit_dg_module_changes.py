"""Attribute checkpoint policy movement to memory, trunk, or renderer.

Called by the native audit driver, which owns token histories, model loading,
compilation, autocast, and sequential teacher forcing. This module never runs
a training step. An evaluator must explicitly identify whether it recomputes
the entire prefix or conditions on stored behavior beliefs: ordinary memory
replay does the latter and must not be described as full policy evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PolicyEvaluator:
    """Evaluate the same ordered positions after every weight substitution.

    ``logits`` returns native softcapped logits [positions, vocabulary]. Its
    implementation must reset decode/cache/memory state on every invocation.
    Structural token groups overlap only if explicitly supplied by the driver.
    """

    name: str
    state_semantics: str
    logits: Callable[[nn.Module], Tensor]
    targets: Tensor
    groups: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if self.state_semantics not in {
            "full_prefix_recomputed",
            "stored_behavior_beliefs_conditioned",
        }:
            raise ValueError("Evaluator must declare its prefix-state semantics")
        if self.targets.ndim != 1 or self.targets.dtype != torch.long:
            raise ValueError("targets must be a one-dimensional int64 tensor")
        for name, mask in self.groups.items():
            if name in {"all", "_positions"}:
                raise ValueError(f"group name {name!r} is reserved")
            if mask.shape != self.targets.shape or mask.dtype != torch.bool:
                raise ValueError(f"group {name!r} requires a bool mask per target")


def forced_prefix_evaluator(
    *,
    name: str,
    token_ids: Tensor,
    prompt_length: int,
    prompt_valid: Tensor,
    positions: Tensor,
    groups: Mapping[str, Tensor],
    step_core: Callable,
) -> PolicyEvaluator:
    """Native sequential teacher forcing with newly computed memory writes.

    ``positions`` is CPU int64 [N, 2] containing row and TARGET token slot.
    Slot prompt_length is predicted by prefill; slot t > prompt_length is
    predicted after consuming t - 1. This avoids inspecting the target before
    computing its distribution. The driver supplies its compiled step_core
    and native bf16 autocast context; this helper supplies no eager fallback.
    Ranged attention and fixed bf16 cache shapes match native decode semantics.
    """
    if token_ids.ndim != 2 or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be [rows, stream] int64")
    if positions.device.type != "cpu" or positions.dtype != torch.long:
        raise ValueError("positions must be CPU int64")
    if positions.ndim != 2 or positions.shape[1] != 2 or not positions.numel():
        raise ValueError("positions must be nonempty [N, 2]")
    if prompt_valid.shape != (token_ids.shape[0], prompt_length):
        raise ValueError("prompt_valid must match the deterministic prefix")
    if prompt_valid.dtype != torch.bool or not prompt_valid[:, -1].all():
        raise ValueError("every prompt must end in a real token")
    # Native ranged decode requires contiguous left padding, as training does.
    if (prompt_valid[:, :-1] & ~prompt_valid[:, 1:]).any():
        raise ValueError("prompt validity must be contiguous after left padding")
    slots = positions.tolist()
    if any(row < 0 or row >= token_ids.shape[0] for row, _ in slots):
        raise ValueError("selected row outside captured histories")
    if any(slot < prompt_length or slot >= token_ids.shape[1] for _, slot in slots):
        raise ValueError("selected targets must be generated-token positions")
    targets = token_ids[
        positions[:, 0].to(token_ids.device), positions[:, 1].to(token_ids.device)
    ]
    selected: dict[int, list[tuple[int, int]]] = {}
    for ordinal, (row, slot) in enumerate(slots):
        selected.setdefault(slot, []).append((ordinal, row))
    last_target = max(selected)

    @torch.no_grad()
    def evaluate(wrapper: nn.Module) -> Tensor:
        if not wrapper.memory_carry:
            raise ValueError("this audit requires the native memory-carry actor")
        device = token_ids.device
        caches = wrapper.backbone.make_generation_cache(
            token_ids.shape[0], last_target + 1, device, dtype=torch.bfloat16
        )
        for layer in caches:
            for tensor in layer:
                tensor.zero_()
        output = wrapper.prefill(token_ids[:, :prompt_length], caches, prompt_valid)
        memory = wrapper.memory_prompt_state(output.prefix_beliefs, prompt_valid)
        belief, logits = output.belief, output.logits
        starts = prompt_valid.long().argmax(-1)
        position = torch.tensor(prompt_length, device=device, dtype=torch.long)
        recorded: list[Tensor | None] = [None] * len(slots)
        for target_slot in range(prompt_length, last_target + 1):
            for ordinal, row in selected.get(target_slot, ()):
                # Graph-backed outputs may be overwritten by the next step.
                recorded[ordinal] = logits[row].detach().clone()
            if target_slot == last_target:
                break
            input_latent = wrapper.memory_input(token_ids[:, target_slot], belief, memory)
            position.fill_(target_slot)
            belief, _, _, logits = step_core(
                input_latent, caches, position, key_starts=starts
            )
            memory = wrapper.memory_write(memory, belief)
        if any(value is None for value in recorded):
            raise RuntimeError("failed to observe every requested target distribution")
        return torch.stack(recorded)

    return PolicyEvaluator(
        name=name, state_semantics="full_prefix_recomputed",
        logits=evaluate, targets=targets, groups=groups,
    )


def policy_component(name: str) -> str:
    if name.startswith("combiner."):
        return "memory"
    if name.startswith("backbone.proj."):
        return "renderer"
    if name.startswith("backbone."):
        return "trunk"
    raise ValueError(f"Unclassified actor state entry {name!r}")


def _distribution_summary(values: Tensor) -> dict[str, float | int]:
    if not values.numel():
        return {"count": 0}
    quantiles = torch.quantile(
        values.float(), values.new_tensor([0.1, 0.5, 0.9, 0.99]).float()
    ).tolist()
    return {
        "count": values.numel(),
        "mean": float(values.mean()),
        "rms": float(values.square().mean().sqrt()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": quantiles[0],
        "p50": quantiles[1],
        "p90": quantiles[2],
        "p99": quantiles[3],
    }


@torch.no_grad()
def compare_policy_logits(
    reference: Tensor,
    candidate: Tensor,
    targets: Tensor,
    groups: Mapping[str, Tensor],
) -> dict[str, Any]:
    """Exact categorical KL, distinct from the sampled k3 estimate.

    Log-softmax and probability arithmetic are fp32; the driver's native
    readout determines the logits' GEMM precision and rational softcap.
    """
    if reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("reference and candidate require matching [N, V] logits")
    if reference.shape[0] != targets.numel():
        raise ValueError("one target is required for every policy position")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("nonfinite policy logits")
    old = reference.float().log_softmax(-1)
    new = candidate.float().log_softmax(-1)
    old_p, new_p = old.exp(), new.exp()
    targets = targets.to(device=old.device)
    picked_old = old.gather(-1, targets[:, None]).squeeze(-1)
    picked_new = new.gather(-1, targets[:, None]).squeeze(-1)
    log_ratio = picked_new - picked_old
    measurements = {
        "exact_kl_reference_to_candidate": (old_p * (old - new)).sum(-1),
        "exact_kl_candidate_to_reference": (new_p * (new - old)).sum(-1),
        "chosen_logprob_change": log_ratio,
        "chosen_probability_change": picked_new.exp() - picked_old.exp(),
        "sampled_k3": torch.expm1(log_ratio) - log_ratio,
        "entropy_change": -(new_p * new).sum(-1) + (old_p * old).sum(-1),
        "top_token_changed": (old.argmax(-1) != new.argmax(-1)).float(),
    }
    masks = {"all": torch.ones_like(targets, dtype=torch.bool), **groups}
    result = {
        group: {
            metric: _distribution_summary(values[mask.to(device=old.device)])
            for metric, values in measurements.items()
        }
        for group, mask in masks.items()
    }
    result["_positions"] = {
        "target_token_ids": targets.cpu().tolist(),
        **{metric: values.cpu().tolist() for metric, values in measurements.items()},
    }
    return result


@torch.no_grad()
def audit_module_changes(
    wrapper: nn.Module,
    before_state: Mapping[str, Tensor],
    after_state: Mapping[str, Tensor],
    evaluators: Sequence[PolicyEvaluator],
    *,
    restore_state: Mapping[str, Tensor],
    before_label: str,
    after_label: str,
) -> dict[str, Any]:
    """Apply all/memory/trunk/renderer changes on the earlier checkpoint.

    All copies mutate existing parameter/buffer storage, preserving compiled
    graph addresses. The caller supplies the current checkpoint to restore;
    it must own independent storage, not wrapper.state_dict() aliases.
    Each callback is a fresh evaluation on identical tokens and positions.
    Components' effects need not add because the policy is nonlinear.
    """
    current = wrapper.state_dict()
    if not evaluators:
        raise ValueError("at least one native evaluator is required")
    for label, state in (
        ("before", before_state), ("after", after_state), ("restore", restore_state)
    ):
        if set(state) != set(current):
            raise ValueError(f"{label} state keys differ from the loaded actor")
        for name, tensor in state.items():
            if tensor.shape != current[name].shape or tensor.dtype != current[name].dtype:
                raise ValueError(f"{label} shape or dtype differs for {name}")
            if label == "restore" and tensor.data_ptr() == current[name].data_ptr():
                raise ValueError("restore_state must not alias live actor storage")
    components = {name: policy_component(name) for name in current}

    def install(component: str | None) -> None:
        for name, destination in current.items():
            use_after = component == "all" or components[name] == component
            source = after_state[name] if use_after else before_state[name]
            destination.copy_(source)

    result: dict[str, Any] = {
        "before": before_label,
        "after": after_label,
        "component_elements": {
            component: sum(t.numel() for k, t in current.items() if components[k] == component)
            for component in ("memory", "trunk", "renderer")
        },
        "evaluators": {},
        "interpretation": (
            "A substitution attributes function change on identical observed prefixes; "
            "it does not establish which update caused the training collapse. "
            "Stored-belief replay conditions on behavior state and is not a "
            "full-prefix evaluation under the substituted policy."
        ),
    }
    try:
        install(None)
        references = {}
        for evaluator in evaluators:
            if evaluator.name in references:
                raise ValueError(f"duplicate evaluator name {evaluator.name!r}")
            references[evaluator.name] = evaluator.logits(wrapper).detach().float()
            result["evaluators"][evaluator.name] = {
                "state_semantics": evaluator.state_semantics,
                "position_count": evaluator.targets.numel(),
                "cases": {},
            }
        for component in ("all", "memory", "trunk", "renderer"):
            install(component)
            for evaluator in evaluators:
                logits = evaluator.logits(wrapper)
                result["evaluators"][evaluator.name]["cases"][component] = (
                    compare_policy_logits(
                        references[evaluator.name], logits,
                        evaluator.targets, evaluator.groups,
                    )
                )
                del logits
        # Separates reproducibility of native evaluation from attribution.
        # Report the actual repeat error rather than choosing a tolerance.
        install(None)
        for evaluator in evaluators:
            repeated = evaluator.logits(wrapper).detach().float()
            reference = references[evaluator.name]
            result["evaluators"][evaluator.name]["reference_repeat"] = {
                "bitwise_equal": torch.equal(reference, repeated),
                "logits_abs_change_max": float((repeated - reference).abs().max()),
                "metrics": compare_policy_logits(
                    reference, repeated, evaluator.targets, evaluator.groups
                ),
            }
    finally:
        for name, destination in current.items():
            destination.copy_(restore_state[name])
    return result

"""Low-rank adaptation shared by every model family.

Moved out of the MiniCPM policy module unchanged: the arithmetic here is
qualified (fp32 masters under bf16 autocast, power-of-two scale folding for
bit-identical forward and backward) and must stay identical for existing
checkpoints. Nothing in this module is Llama-specific; target module names
come from the caller's :class:`LoRAConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS
    initialization: Literal["standard", "nora"] = "nora"

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if self.initialization not in {"standard", "nora"}:
            raise ValueError("LoRA initialization must be standard or nora")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("LoRA targets must be nonempty and unique")


class LoRALinear(nn.Module):
    """Frozen linear plus fp32-master low-rank update.

    CUDA callers run under bf16 autocast. Keeping the trainable matrices fp32
    avoids losing small AdamW updates while autocast still executes both small
    GEMMs in the model's compute dtype.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.rank = config.rank
        self.scaling = config.alpha / config.rank
        # Scaling by a power of two commutes with every rounding step, so
        # folding it into the small ``lora_b`` operand before the GEMM gives
        # bit-identical forward values and gradients while dropping one
        # token-sized multiply per projection in each direction.
        self._fold_scaling = math.frexp(self.scaling)[0] == 0.5
        self.lora_a = nn.Parameter(
            torch.empty(config.rank, base.in_features, dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, config.rank, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        if config.initialization == "nora":
            with torch.no_grad():
                column_norms = torch.linalg.vector_norm(
                    self.lora_a, dim=0, keepdim=True
                )
                self.lora_a.div_(
                    column_norms.clamp_min(torch.finfo(self.lora_a.dtype).eps)
                )
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: Tensor) -> Tensor:
        if (
            inputs.dtype != self.lora_a.dtype
            and not torch.is_autocast_enabled(inputs.device.type)
        ):
            raise RuntimeError(
                "mixed-dtype LoRA requires autocast so fp32 masters are cast "
                "only inside the adapter GEMMs"
            )
        if self._fold_scaling:
            update = F.linear(
                F.linear(inputs, self.lora_a), self.lora_b * self.scaling
            )
            return self.base(inputs).add_(update)
        update = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.scaling

def inject_lora(model: nn.Module, config: LoRAConfig) -> tuple[str, ...]:
    """Freeze ``model`` and replace every requested projection exactly once."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in config.targets:
            replacements.append((name, module))
    if not replacements:
        raise ValueError(f"none of the LoRA targets exist: {config.targets}")
    found_targets = {name.rsplit(".", 1)[-1] for name, _ in replacements}
    missing = set(config.targets) - found_targets
    if missing:
        raise ValueError(f"missing LoRA target modules: {sorted(missing)}")
    for name, module in replacements:
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            child_name = name
            parent = model
        setattr(parent, child_name, LoRALinear(module, config))
    return tuple(name for name, _ in replacements)

def share_frozen_parameters_(
    destination: nn.Module, source: nn.Module
) -> tuple[str, ...]:
    """Alias every immutable destination parameter to matching source storage."""

    source_parameters = dict(source.named_parameters())
    shared: list[str] = []
    for name, parameter in list(destination.named_parameters()):
        if parameter.requires_grad:
            continue
        source_parameter = source_parameters.get(name)
        if source_parameter is None:
            raise ValueError(f"shared backbone source is missing {name}")
        if source_parameter.requires_grad:
            raise ValueError(f"shared backbone source parameter is trainable: {name}")
        if (
            source_parameter.shape != parameter.shape
            or source_parameter.dtype != parameter.dtype
        ):
            raise ValueError(f"shared backbone parameter differs: {name}")
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = destination.get_submodule(parent_name)
        else:
            child_name = name
            parent = destination
        setattr(parent, child_name, source_parameter)
        shared.append(name)
    if not shared:
        raise ValueError("no immutable backbone parameters were shared")
    return tuple(shared)

def merge_lora_for_inference(model: nn.Module) -> tuple[str, ...]:
    """Fold every LoRA update into its frozen base weight and remove the wrappers."""

    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    with torch.no_grad():
        for name, module in replacements:
            if torch.count_nonzero(module.lora_b):
                update = module.lora_b @ module.lora_a
                module.base.weight.add_(
                    update.to(module.base.weight.dtype), alpha=module.scaling
                )
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                child_name = name
                parent = model
            setattr(parent, child_name, module.base)
    return tuple(name for name, _ in replacements)


def adapter_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    }


def load_adapter_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    }
    if parameters.keys() != state.keys():
        missing = sorted(parameters.keys() - state.keys())
        unexpected = sorted(state.keys() - parameters.keys())
        raise ValueError(
            f"adapter state mismatch: missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


__all__ = [
    "DEFAULT_LORA_TARGETS",
    "LoRAConfig",
    "LoRALinear",
    "adapter_state_dict",
    "inject_lora",
    "load_adapter_state_dict",
    "merge_lora_for_inference",
    "share_frozen_parameters_",
]

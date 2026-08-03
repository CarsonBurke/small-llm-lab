"""Validate CUDA-graph gradient accumulation for torch.compile.

CUDA Graph Trees reuse the storage holding a compiled backward's outputs.
When ``Parameter.grad`` starts as ``None``, autograd may adopt that storage
directly. Replaying the graph for the next microbatch then overwrites the
already-accumulated gradient. A persistent, eagerly allocated gradient buffer
prevents that alias while retaining CUDA graphs.
"""

import copy
import json

import torch
from torch import nn


MICROBATCHES = 16
BATCH_SIZE = 32
WIDTH = 128


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(WIDTH, 4 * WIDTH, bias=False)
        self.down = nn.Linear(4 * WIDTH, WIDTH, bias=False)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        hidden = self.up(inputs).relu().square()
        return (self.down(hidden) - targets).square().mean()


@torch.no_grad()
def set_grads_none(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


@torch.no_grad()
def reset_persistent_grads(model: nn.Module) -> None:
    grads_to_clear = []
    for parameter in model.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(
                parameter,
                memory_format=torch.preserve_format,
            )
        else:
            grads_to_clear.append(parameter.grad)
    if grads_to_clear:
        torch._foreach_zero_(grads_to_clear)


def mark_step() -> None:
    torch.compiler.cudagraph_mark_step_begin()


def clone_grads(model: nn.Module) -> list[torch.Tensor]:
    return [parameter.grad.detach().clone() for parameter in model.parameters()]


def errors(
    names: list[str],
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
) -> dict[str, object]:
    per_parameter = []
    for name, actual_grad, expected_grad in zip(names, actual, expected):
        absolute = float((actual_grad - expected_grad).abs().max())
        reference = float(expected_grad.abs().max())
        per_parameter.append({
            "name": name,
            "max_abs": absolute,
            "max_abs_relative_to_reference": absolute / max(reference, 1e-30),
        })
    return {
        "max_abs": max(entry["max_abs"] for entry in per_parameter),
        "max_relative_error": max(
            entry["max_abs_relative_to_reference"]
            for entry in per_parameter
        ),
        "per_parameter": per_parameter,
    }


def main() -> None:
    torch.manual_seed(1337)
    device = torch.device("cuda")
    inputs = [
        torch.randn(BATCH_SIZE, WIDTH, device=device)
        for _ in range(MICROBATCHES)
    ]
    targets = [
        torch.randn(BATCH_SIZE, WIDTH, device=device)
        for _ in range(MICROBATCHES)
    ]

    model = TinyModel().to(device)
    parameter_names = [name for name, _ in model.named_parameters()]
    initial_state = copy.deepcopy(model.state_dict())
    compiled_model = torch.compile(model, dynamic=False, mode="reduce-overhead")

    # Same compiled kernels, but preserve each microbatch's gradient before the
    # next CUDA-graph replay can reuse its output storage.
    reference = [torch.zeros_like(parameter) for parameter in model.parameters()]
    for micro_inputs, micro_targets in zip(inputs, targets):
        set_grads_none(model)
        mark_step()
        compiled_model(micro_inputs, micro_targets).backward()
        for accumulated, micro_grad in zip(reference, clone_grads(model)):
            accumulated.add_(micro_grad)

    model.load_state_dict(initial_state)
    set_grads_none(model)
    for micro_inputs, micro_targets in zip(inputs, targets):
        mark_step()
        compiled_model(micro_inputs, micro_targets).backward()
    aliased = clone_grads(model)

    model.load_state_dict(initial_state)
    set_grads_none(model)
    reset_persistent_grads(model)
    persistent_addresses = [parameter.grad.data_ptr() for parameter in model.parameters()]
    for micro_inputs, micro_targets in zip(inputs, targets):
        mark_step()
        compiled_model(micro_inputs, micro_targets).backward()
    fixed = clone_grads(model)
    addresses_stable = persistent_addresses == [
        parameter.grad.data_ptr() for parameter in model.parameters()
    ]

    aliased_error = errors(parameter_names, aliased, reference)
    fixed_error = errors(parameter_names, fixed, reference)
    result = {
        "torch_version": torch.__version__,
        "microbatches": MICROBATCHES,
        "aliased_set_to_none": aliased_error,
        "persistent_grad_buffers": fixed_error,
        "persistent_addresses_stable": addresses_stable,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if aliased_error["max_relative_error"] < 1e-3:
        raise RuntimeError("The validator did not reproduce CUDA-graph corruption")
    if fixed_error["max_relative_error"] > 1e-5:
        raise RuntimeError("Persistent gradient buffers did not restore accumulation")
    if not addresses_stable:
        raise RuntimeError("Persistent gradient buffer addresses changed")


if __name__ == "__main__":
    main()

"""GPU identity helpers shared by authenticated benchmark telemetry."""

from __future__ import annotations

import os

import torch


def nvidia_smi_selector(device: torch.device) -> str:
    """Resolve a logical CUDA device to the physical ID consumed by nvidia-smi."""

    if device.type != "cuda":
        raise ValueError("telemetry requires a CUDA device")
    logical_index = (
        torch.cuda.current_device() if device.index is None else device.index
    )
    if logical_index < 0:
        raise ValueError("telemetry requires a valid CUDA device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(logical_index)
    devices = tuple(item.strip() for item in visible.split(","))
    if logical_index >= len(devices) or not devices[logical_index]:
        raise ValueError("logical CUDA device is absent from CUDA_VISIBLE_DEVICES")
    selector = devices[logical_index]
    if selector == "-1":
        raise ValueError("CUDA_VISIBLE_DEVICES disables the selected device")
    return selector


__all__ = ("nvidia_smi_selector",)

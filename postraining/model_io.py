"""Construct and load the fresh pretraining architecture."""

from __future__ import annotations

from pathlib import Path

import torch

from fresh_lejepa_train import FreshLeJEPAGPT
from train_gpt import CastedLinear, restore_low_dim_params_to_fp32


DEFAULT_MODEL_CONFIG = {
    "vocab_size": 1024,
    "num_layers": 9,
    "model_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 4,
    "mlp_mult": 2,
    "tie_embeddings": True,
    "tied_embed_init_std": 0.005,
    "logit_softcap": 30.0,
    "rope_base": 10000.0,
    "qk_gain_init": 1.5,
}


def load_model(checkpoint: str | Path, device: torch.device) -> FreshLeJEPAGPT:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dict(DEFAULT_MODEL_CONFIG)
    if isinstance(payload, dict) and "metadata" in payload:
        config.update(payload["metadata"]["model"])
    elif isinstance(payload, dict) and "model_config" in payload:
        config.update(payload["model_config"])
    model = FreshLeJEPAGPT(**config).to(device).bfloat16()
    model.model_config = config
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.policy_probe.parameters():
        parameter.requires_grad_(True)
    for parameter in model.critic_probe.parameters():
        parameter.requires_grad_(True)
    return model

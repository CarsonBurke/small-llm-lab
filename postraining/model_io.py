"""Construct and load the fresh pretraining architecture."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

import train_gpt as baseline
from fresh_lejepa_train import EXPERIMENT_ARCHITECTURE, FreshLeJEPAGPT
from train_gpt import CastedLinear, restore_low_dim_params_to_fp32


@contextmanager
def _pope_construction():
    """Reproduce the pretraining monkeypatches PoPE construction depends on.

    The PoPE block is instantiated through ``baseline.CausalSelfAttention``,
    and ``delta_c`` must join the control-tensor patterns so the fp32 restore
    treats phase offsets exactly as pretraining did.
    """
    import fresh_lejepa_train_v1_probe_shared_rms_pope as pope

    if pope.PolarCausalSelfAttention.position_mode != "pope":
        raise ValueError(
            "loading a PoPE checkpoint requires FRESH_POSITION_MODE=pope "
            f"(got {pope.PolarCausalSelfAttention.position_mode!r})"
        )
    original_attention = baseline.CausalSelfAttention
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    baseline.CausalSelfAttention = pope.PolarCausalSelfAttention
    baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns + ("delta_c",)
    try:
        yield
    finally:
        baseline.CausalSelfAttention = original_attention
        baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns


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


def load_model(
    checkpoint: str | Path, device: torch.device, payload: dict | None = None
) -> FreshLeJEPAGPT:
    if payload is None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dict(DEFAULT_MODEL_CONFIG)
    architecture = EXPERIMENT_ARCHITECTURE
    if isinstance(payload, dict) and "metadata" in payload:
        config.update(payload["metadata"]["model"])
        architecture = payload["metadata"].get("architecture", architecture)
    elif isinstance(payload, dict) and "model_config" in payload:
        config.update(payload["model_config"])
        architecture = payload.get("architecture", architecture)
    model_class = FreshLeJEPAGPT
    architecture = architecture or EXPERIMENT_ARCHITECTURE
    if architecture.endswith("additive_codebook_probes_v2"):
        from fresh_lejepa_train_v2 import FreshLeJEPAGPTV2

        model_class = FreshLeJEPAGPTV2
    elif architecture.endswith("additive_codebook_dropout01_probes_v3"):
        from fresh_lejepa_train_v3 import FreshLeJEPAGPTV3

        model_class = FreshLeJEPAGPTV3
    elif architecture.endswith("v2_sigreg_only_token_projector"):
        from fresh_lejepa_train_v2_sigreg_projector import FreshLeJEPAV2SIGRegProjector

        model_class = FreshLeJEPAV2SIGRegProjector
    elif architecture.endswith("v2_shared_token_projector"):
        from fresh_lejepa_train_v2_shared_projector import FreshLeJEPAV2SharedProjector

        model_class = FreshLeJEPAV2SharedProjector
    elif architecture.endswith("shared_bn_predproj_v1_large_probes"):
        from fresh_lejepa_train_v1_probe_shared_projector import (
            FreshLeJEPASharedProjectorV1Probes,
        )

        model_class = FreshLeJEPASharedProjectorV1Probes
    elif architecture.endswith("shared_rms_predproj_v1_large_probes"):
        from fresh_lejepa_train_v1_probe_shared_rms_projector import (
            FreshLeJEPASharedRMSProjectorV1Probes,
        )

        model_class = FreshLeJEPASharedRMSProjectorV1Probes
    elif architecture.endswith("v4_shared_rms_fp32_sigreg_dwide_learned_probes"):
        from fresh_lejepa_train_v4 import FreshLeJEPAV4

        model_class = FreshLeJEPAV4
    elif architecture.endswith("v4_predicted_only_probe"):
        from fresh_lejepa_train_v4_predicted_only import FreshLeJEPAV4PredictedOnly

        model_class = FreshLeJEPAV4PredictedOnly
    elif architecture.endswith("v4_predictor_dropout01"):
        from fresh_lejepa_train_v4_predictor_dropout import (
            FreshLeJEPAV4PredictorDropout,
        )

        model_class = FreshLeJEPAV4PredictorDropout
    elif architecture.endswith("v5_shared_rms_paired_b128_prenorm_swiglu_probes"):
        from fresh_lejepa_train_v5_swiglu import FreshLeJEPAV5SwiGLU

        model_class = FreshLeJEPAV5SwiGLU
    elif architecture.endswith("v5_prenorm_swiglu_predictor_dropout01_only"):
        from fresh_lejepa_train_v5_dropout_only import FreshLeJEPAV5DropoutOnly

        model_class = FreshLeJEPAV5DropoutOnly
    elif architecture.endswith("v5_belief_only_no_dropout"):
        from fresh_lejepa_train_v5_belief_only import FreshLeJEPAV5BeliefOnly

        model_class = FreshLeJEPAV5BeliefOnly
    elif architecture.endswith("v5_rms_jedi_edm_token_target"):
        from fresh_lejepa_train_v5_jedi_denoising import (
            FreshLeJEPAV5JEDIDenoising,
        )

        model_class = FreshLeJEPAV5JEDIDenoising
    elif architecture.endswith("v9_action_conditioned_belief_transition_edm"):
        from fresh_lejepa_train_v9_belief_transition_jedi import (
            FreshLeJEPAV9BeliefTransition,
        )

        model_class = FreshLeJEPAV9BeliefTransition
    elif architecture.endswith(
        "v6_belief_only_prenorm_swiglu_predictor_dropout01"
    ):
        from fresh_lejepa_train_v6_belief_dropout import (
            FreshLeJEPAV6BeliefDropout,
        )

        model_class = FreshLeJEPAV6BeliefDropout
    elif architecture.endswith("v7_pre_predproj_belief_only_dropout01"):
        from fresh_lejepa_train_v7_preproj_belief import (
            FreshLeJEPAV7PreProjBelief,
        )

        model_class = FreshLeJEPAV7PreProjBelief
    elif architecture.endswith("v8_shared_token_prenorm_swiglu_encoder"):
        from fresh_lejepa_train_v8_token_swiglu import FreshLeJEPAV8TokenSwiGLU

        model_class = FreshLeJEPAV8TokenSwiGLU
    elif architecture.endswith("v2_predicted_only_probe"):
        from fresh_lejepa_train_v2_predicted_only import FreshLeJEPAV2PredictedOnly

        model_class = FreshLeJEPAV2PredictedOnly
    construction = nullcontext()
    if "_pope_" in architecture:
        from fresh_lejepa_train_v1_probe_shared_rms_pope import (
            FreshLeJEPASharedRMSV1PoPE,
        )

        model_class = FreshLeJEPASharedRMSV1PoPE
        construction = _pope_construction()
    elif architecture.endswith("rope_scratch_1k_control"):
        from fresh_lejepa_train_v1_probe_shared_rms_projector import (
            FreshLeJEPASharedRMSProjectorV1Probes,
        )

        model_class = FreshLeJEPASharedRMSProjectorV1Probes
    with construction:
        model = model_class(**config).to(device).bfloat16()
        model.model_config = config
        model.architecture = architecture
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

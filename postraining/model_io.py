"""Construct and load the fresh pretraining architecture."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

import train_gpt as baseline
from pretraining.fresh_lejepa.fresh_lejepa_train import EXPERIMENT_ARCHITECTURE, FreshLeJEPAGPT


@contextmanager
def _pope_construction():
    """Reproduce the pretraining monkeypatches PoPE construction depends on.

    The PoPE block is instantiated through ``baseline.CausalSelfAttention``,
    and ``delta_c`` must join the control-tensor patterns so the fp32 restore
    treats phase offsets exactly as pretraining did.
    """
    from pretraining.fresh_lejepa import fresh_lejepa_train_v1_probe_shared_rms_pope as pope

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


def _checkpoint_train_context(payload) -> int:
    """Read pretraining context from current and legacy payload layouts."""
    if not isinstance(payload, dict):
        return 1024
    if "train_seq_len" in payload:
        return int(payload["train_seq_len"])
    metadata = payload.get("metadata", {})
    optimizer = metadata.get("optimizer", {}) if isinstance(metadata, dict) else {}
    return int(optimizer.get("train_seq_len", 1024))


def _load_nano_model(architecture: str, payload, device: torch.device):
    """Construct and strict-load a nanogpt-mini backbone.

    The nano branch never merges the fresh DEFAULT_MODEL_CONFIG — the two
    architectures share no constructor signature. All parameters are frozen
    (the trainer re-enables what it trains); there are no probes to unfreeze.
    """
    from postraining.nano_backbone import (
        NANO_DEFAULT_MODEL_CONFIG,
        NanoGPTBackbone,
        NanoTiedDotBackbone,
    )

    if "_gdn2_" in architecture:
        raise NotImplementedError(
            f"{architecture} requires a GDN-2 recurrent backbone; only the "
            "plain-KDA and dense nano architectures are implemented"
        )

    if "_kda_" in architecture:
        from postraining.kda_backbone import NanoKDABackbone

        # KDA checkpoints always carry their full model_config (mixer layout,
        # head count, gate rank); there is no sensible default to fall back
        # on, so a payload without one is refused rather than guessed at.
        if not (isinstance(payload, dict) and "model_config" in payload):
            raise ValueError(
                f"{architecture} checkpoint payload lacks model_config; "
                "cannot reconstruct the KDA mixer layout"
            )
        config = dict(payload["model_config"])
        model_class = NanoKDABackbone
    else:
        config = dict(NANO_DEFAULT_MODEL_CONFIG)
        if isinstance(payload, dict) and "model_config" in payload:
            config.update(payload["model_config"])
        if "tieddot" in architecture:
            model_class = NanoTiedDotBackbone
        else:
            model_class = NanoGPTBackbone
    model = model_class(**config).to(device)
    model.model_config = config
    model.architecture = architecture
    # RoPE positions seen in pretraining bound the usable RL context: the
    # half-truncate rotary has no extrapolation. Old checkpoints predate the
    # field and were all trained on 1024-token windows.
    model.train_context_tokens = _checkpoint_train_context(payload)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    # The checkpoint stores a bf16 embedding; the backbone keeps fp32 masters
    # (bf16 -> fp32 is value-exact, and load_state_dict casts on copy).
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


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
    if architecture.startswith("nanogpt_mini"):
        return _load_nano_model(architecture, payload, device)
    if architecture.endswith("additive_codebook_probes_v2"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v2 import FreshLeJEPAGPTV2

        model_class = FreshLeJEPAGPTV2
    elif architecture.endswith("additive_codebook_dropout01_probes_v3"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v3 import FreshLeJEPAGPTV3

        model_class = FreshLeJEPAGPTV3
    elif architecture.endswith("v2_sigreg_only_token_projector"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v2_sigreg_projector import FreshLeJEPAV2SIGRegProjector

        model_class = FreshLeJEPAV2SIGRegProjector
    elif architecture.endswith("v2_shared_token_projector"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v2_shared_projector import FreshLeJEPAV2SharedProjector

        model_class = FreshLeJEPAV2SharedProjector
    elif architecture.endswith("shared_bn_predproj_v1_large_probes"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_projector import (
            FreshLeJEPASharedProjectorV1Probes,
        )

        model_class = FreshLeJEPASharedProjectorV1Probes
    elif architecture.endswith("shared_rms_predproj_v1_large_probes"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_projector import (
            FreshLeJEPASharedRMSProjectorV1Probes,
        )

        model_class = FreshLeJEPASharedRMSProjectorV1Probes
    elif architecture.endswith("v4_shared_rms_fp32_sigreg_dwide_learned_probes"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v4 import FreshLeJEPAV4

        model_class = FreshLeJEPAV4
    elif architecture.endswith("v4_predicted_only_probe"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v4_predicted_only import FreshLeJEPAV4PredictedOnly

        model_class = FreshLeJEPAV4PredictedOnly
    elif architecture.endswith("v4_predictor_dropout01"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v4_predictor_dropout import (
            FreshLeJEPAV4PredictorDropout,
        )

        model_class = FreshLeJEPAV4PredictorDropout
    elif architecture.endswith("v5_shared_rms_paired_b128_prenorm_swiglu_probes"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v5_swiglu import FreshLeJEPAV5SwiGLU

        model_class = FreshLeJEPAV5SwiGLU
    elif architecture.endswith("v5_prenorm_swiglu_predictor_dropout01_only"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v5_dropout_only import FreshLeJEPAV5DropoutOnly

        model_class = FreshLeJEPAV5DropoutOnly
    elif architecture.endswith("v5_belief_only_no_dropout"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v5_belief_only import FreshLeJEPAV5BeliefOnly

        model_class = FreshLeJEPAV5BeliefOnly
    elif architecture.endswith("v5_rms_jedi_edm_token_target"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v5_jedi_denoising import (
            FreshLeJEPAV5JEDIDenoising,
        )

        model_class = FreshLeJEPAV5JEDIDenoising
    elif architecture.endswith("v9_action_conditioned_belief_transition_edm"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v9_belief_transition_jedi import (
            FreshLeJEPAV9BeliefTransition,
        )

        model_class = FreshLeJEPAV9BeliefTransition
    elif architecture.endswith(
        "v6_belief_only_prenorm_swiglu_predictor_dropout01"
    ):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v6_belief_dropout import (
            FreshLeJEPAV6BeliefDropout,
        )

        model_class = FreshLeJEPAV6BeliefDropout
    elif architecture.endswith("v7_pre_predproj_belief_only_dropout01"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v7_preproj_belief import (
            FreshLeJEPAV7PreProjBelief,
        )

        model_class = FreshLeJEPAV7PreProjBelief
    elif architecture.endswith("v8_shared_token_prenorm_swiglu_encoder"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v8_token_swiglu import FreshLeJEPAV8TokenSwiGLU

        model_class = FreshLeJEPAV8TokenSwiGLU
    elif architecture.endswith("v2_predicted_only_probe"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v2_predicted_only import FreshLeJEPAV2PredictedOnly

        model_class = FreshLeJEPAV2PredictedOnly
    construction = nullcontext()
    if "_pope_" in architecture:
        if architecture.endswith(
            "distributional_kernel_mean_perdim_nosigreg_onepass_2k"
        ):
            from energy_readout.fresh_lejepa_train_distributional_kernel_jepa import (
                DistributionalKernelLeJEPA,
            )

            model_class = DistributionalKernelLeJEPA
            state = (
                payload["model"]
                if isinstance(payload, dict) and "model" in payload
                else payload
            )
            frequency = None
            if isinstance(state, dict):
                frequency = next(
                    (
                        value
                        for key, value in state.items()
                        if key.endswith(".kernel_frequencies_q16")
                    ),
                    None,
                )
            if torch.is_tensor(frequency):
                config["kernel_features"] = 2 * int(frequency.shape[0])
        elif architecture.endswith(
            "pope_shared_point_energy_onepass_2k"
        ):
            from energy_readout.fresh_lejepa_train_shared_point_energy import (
                SharedPointEnergyLeJEPA,
            )

            model_class = SharedPointEnergyLeJEPA
        elif architecture.endswith(
            "pope_unified_energy_barycenter_onepass_2k"
        ):
            from energy_readout.fresh_lejepa_train_unified_energy_barycenter import (
                UnifiedEnergyBarycenterLeJEPA,
            )

            model_class = UnifiedEnergyBarycenterLeJEPA
        elif architecture.endswith(
            "pope_jepa_owned_geometry_rbf_decoder_onepass_2k"
        ):
            from energy_readout.fresh_lejepa_train_jepa_rbf_decoder import (
                GeometryPreservingRBFLeJEPA,
            )

            model_class = GeometryPreservingRBFLeJEPA
        elif architecture.endswith("probes_pope_belief_attached_ce_onepass_2k"):
            from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope_belief_attached import (
                FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE,
            )

            model_class = FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE
        elif architecture.endswith("probes_pope_attached_ce_scratch_2k"):
            from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope_attached import (
                FreshLeJEPASharedRMSV1PoPEAttachedCE,
            )

            model_class = FreshLeJEPASharedRMSV1PoPEAttachedCE
        else:
            from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope import (
                FreshLeJEPASharedRMSV1PoPE,
            )

            model_class = FreshLeJEPASharedRMSV1PoPE
        construction = _pope_construction()
    elif architecture.endswith("rope_scratch_1k_control"):
        from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_projector import (
            FreshLeJEPASharedRMSProjectorV1Probes,
        )

        model_class = FreshLeJEPASharedRMSProjectorV1Probes
    with construction:
        # Pretraining trains fp32 master weights and lets autocast (or
        # CastedLinear) drop precision per-op; a whole-body bf16 cast is NOT
        # equivalent — measured +0.32 val BPB on the PoPE 2k checkpoint
        # (phase math and norms are precision-sensitive) — so the loaded
        # model keeps the checkpoint's fp32 masters.
        model = model_class(**config).to(device)
        model.model_config = config
        model.architecture = architecture
        model.train_context_tokens = _checkpoint_train_context(payload)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model, "policy_probe"):
        for parameter in model.policy_probe.parameters():
            parameter.requires_grad_(True)
    if hasattr(model, "critic_probe"):
        for parameter in model.critic_probe.parameters():
            parameter.requires_grad_(True)
    if hasattr(model.blocks[-1], "rbf_token_bias"):
        for name, parameter in model.blocks[-1].named_parameters():
            if name.startswith("rbf_"):
                parameter.requires_grad_(True)
    if hasattr(model.blocks[-1], "energy_bias"):
        for name, parameter in model.blocks[-1].named_parameters():
            if name.startswith("energy_"):
                parameter.requires_grad_(True)
    return model


def fresh_trunk(
    reference,
    device: torch.device,
    *,
    model_config_overrides: dict | None = None,
    architecture_suffix: str = "",
    architecture_override: str | None = None,
):
    """A fresh, randomly initialized, fully trainable instance of a loaded
    model's architecture — same class and config, none of its weights.

    ``model_config_overrides`` supports deliberately simpler auxiliary trunks
    such as a dense critic behind a LatentMoE policy.  The effective config is
    retained on the object so manifests and checkpoints describe the critic
    that was actually constructed. ``architecture_override`` supplies the
    corresponding self-description when a suffix alone would be misleading.
    """
    construction = (
        _pope_construction() if "_pope_" in reference.architecture else nullcontext()
    )
    model_config = dict(reference.model_config)
    if model_config_overrides:
        model_config.update(model_config_overrides)
    with construction:
        model = type(reference)(**model_config).to(device)
    model.model_config = model_config
    model.architecture = (
        architecture_override
        if architecture_override is not None
        else reference.architecture + architecture_suffix
    )
    return model

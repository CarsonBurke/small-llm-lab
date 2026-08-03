"""Energy readout with a per-dimension (diagonal-precision) inverse-temperature.

Builds on the BatchNorm-projector arm
(``energy_readout/fresh_lejepa_train_energy_readout_bnproj.py``).  The scalar global
inverse-temperature ``s = exp(log_s)`` becomes a length-``model_dim`` vector
``s_d = exp(log_s_d)``, giving the codebook-energy head a learned diagonal
precision instead of a single isotropic temperature:

    logit_k = b_k - 0.5 * Σ_d s_d * (ẑ_d - c_{k,d})^2

At initialization every ``log_s_d`` is set to ``-0.5·ln(model_dim)`` (the exact
scalar init value), so the head is numerically identical to the scalar variant
until training pulls the dimensions apart.  The vector still has ``ndim < 2`` so
``train_gpt.py``'s optimizer split keeps routing it to the fused-Adam scalar
group with fp32 masters, exactly like the scalar parameter it replaces.

The codebook path (running-stats BatchNorm ``inference``, attached) is inherited
unchanged from the BN arm.  The only crash the scalar canary printer would hit
is ``float(exp(log_scale))`` on a vector; this module ships its own ``main()``
that mirrors ``energy.main()`` exactly but logs the scale as mean/min/max.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import Tensor, nn

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
from pretraining.fresh_lejepa import fresh_lejepa_train_v1_probe_shared_rms_pope as pope
import train_gpt as baseline
import energy_readout.fresh_lejepa_train_energy_readout as energy
from energy_readout.fresh_lejepa_train_energy_readout_bnproj import (
    EnergyReadoutBNProjLeJEPA,
)

ARCHITECTURE = "energy_readout_lejepa_tied_codebook_bnproj_perdim_onepass_2k"


class EnergyReadoutPerDimScaleLeJEPA(EnergyReadoutBNProjLeJEPA):
    """BN-projector energy readout with a per-dimension diagonal precision."""

    def __init__(self, *args, **kwargs):
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        super().__init__(*args, **kwargs)
        owner = self.blocks[-1]
        # Replace the parent's 0-d scale with a per-dimension vector.  The init
        # value is deterministic (no RNG is consumed by torch.full), so doing
        # this after super().__init__() leaves every other parameter — and the
        # RNG stream — bit-identical to the scalar variant.  Filling every
        # entry with the scalar init value makes the head numerically identical
        # to the scalar variant at step 0.
        del owner.energy_log_scale
        owner.energy_log_scale = nn.Parameter(
            torch.full(
                (model_dim,), -0.5 * math.log(model_dim), dtype=torch.float32
            )
        )

    def energy_logits(self, predicted: Tensor, input_ids: Tensor | None) -> Tensor:
        codebook = self.energy_codebook()
        owner = self.blocks[-1]
        log_scale = owner.energy_log_scale
        # Straight-through, applied elementwise: forward uses the clamped
        # vector, backward passes the unclamped gradient, so no dimension's
        # bound is sticky.
        log_scale = log_scale + (
            log_scale.clamp(math.log(energy.SCALE_MIN), math.log(energy.SCALE_MAX))
            - log_scale
        ).detach()
        scale = torch.exp(log_scale)  # (model_dim,), fp32
        # Same dtype discipline as the parent: the scale that enters the bf16
        # tensor-core matmul is cast to the prediction dtype; the scale that
        # weights the fp32 squared norms stays fp32.
        scaled_predicted = predicted * scale.to(predicted.dtype)
        dots = (scaled_predicted @ codebook.transpose(0, 1)).float()
        if energy.HEAD_FORM == "dot":
            logits = dots + owner.energy_bias
        else:
            scale_f = scale.float()
            z_sq = (predicted.float().square() * scale_f).sum(dim=-1, keepdim=True)
            c_sq = (codebook.float().square() * scale_f).sum(dim=-1)
            logits = owner.energy_bias - 0.5 * (z_sq - 2.0 * dots + c_sq)
        if energy.BIGRAM_TABLE:
            if input_ids is None:
                raise RuntimeError(
                    "ENERGY_BIGRAM_TABLE requires current-token ids at the readout"
                )
            vocab = self.tok_emb.num_embeddings
            logits = logits + owner.energy_bigram.view(vocab, vocab)[input_ids]
        return logits

    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int | float]:
        metadata = super().experiment_metadata()
        metadata.update(
            {
                "energy_scale_form": "per_dimension_diagonal",
                "energy_scale_dims": v1.FreshHyperparameters.model_dim,
            }
        )
        return metadata


def main() -> None:
    # Install this class as the energy family's model and architecture, then
    # mirror energy.main() — rather than call it — because energy.main() would
    # install a canary printer that assumes a 0-d scale and crash on the vector.
    energy.EnergyReadoutLeJEPA = EnergyReadoutPerDimScaleLeJEPA
    energy.ARCHITECTURE = ARCHITECTURE

    if pope.PolarCausalSelfAttention.position_mode != "pope":
        raise ValueError("the energy-readout variant is defined only for PoPE")
    if energy.HEAD_FORM not in {"distance", "dot"}:
        raise ValueError(f"unknown ENERGY_HEAD_FORM={energy.HEAD_FORM!r}")
    # Single-source the latent-MSE weight exactly as energy.main() does: the
    # forward reads FreshHyperparameters.latent_loss_weight and the patched
    # accumulation loop recomposes the logged loss from the model class
    # attribute; both must agree with ENERGY_LATENT_MSE_WEIGHT.
    v1.FreshHyperparameters.latent_loss_weight = energy.LATENT_MSE_WEIGHT
    EnergyReadoutPerDimScaleLeJEPA.latent_loss_weight = energy.LATENT_MSE_WEIGHT
    # Compile warmup would consume the corpus prefix twice; the one-pass
    # contract forbids that reuse.
    v1.FreshHyperparameters.warmup_steps = 0
    pope.FreshLeJEPASharedRMSV1PoPE = EnergyReadoutPerDimScaleLeJEPA
    pope.POPE_ARCHITECTURE = ARCHITECTURE
    pope.__file__ = str(Path(__file__).resolve())

    # Collapse canaries, once per validation.  Adapted for the per-dimension
    # scale: mean/min/max of exp(clamped log scale) plus the codebook norm and
    # spread.  v1.main wraps whatever baseline.eval_val is at call time, so
    # patching before pope.main composes cleanly.
    original_eval_val = baseline.eval_val

    def eval_val_logging_energy_canaries(*args, **kwargs):
        result = original_eval_val(*args, **kwargs)
        # The int8 round-trip eval at the end of baseline.main runs on
        # dequantized weights and prints no val line for the canaries to fold
        # into; recognize it by the `quant_state` local, as the lineage does.
        frame = inspect.currentframe().f_back
        in_quant_eval = False
        while frame is not None:
            if "quant_state" in frame.f_locals:
                in_quant_eval = True
                break
            frame = frame.f_back
        if in_quant_eval or int(os.environ.get("RANK", "0")) != 0:
            return result
        model = kwargs.get("model", args[1] if len(args) > 1 else None)
        base = getattr(model, "module", model)
        base = getattr(base, "_orig_mod", base)
        with torch.no_grad():
            log_scale = base.blocks[-1].energy_log_scale.detach().float()
            scale = torch.exp(
                log_scale.clamp(
                    math.log(energy.SCALE_MIN), math.log(energy.SCALE_MAX)
                )
            )
            scale_mean = float(scale.mean())
            scale_min = float(scale.min())
            scale_max = float(scale.max())
            codebook = base.energy_codebook().detach().float()
            norm_mean = float(codebook.norm(dim=-1).mean())
            pairdist_mean = float(torch.cdist(codebook, codebook).mean())
        # churn_stats is scripts/ablation.py's stepless fold channel: emitted before
        # baseline.main prints the val line, so these keys land inside that val
        # entry in metrics.jsonl and stream to TensorBoard.  Fixed-point
        # formatting is required — the extras parser rejects scientific
        # notation.
        print(
            f"churn_stats energy_scale_mean:{scale_mean:.8f} "
            f"energy_scale_min:{scale_min:.8f} "
            f"energy_scale_max:{scale_max:.8f} "
            f"codebook_norm_mean:{norm_mean:.4f} "
            f"codebook_pairdist_mean:{pairdist_mean:.4f}",
            flush=True,
        )
        return result

    baseline.eval_val = eval_val_logging_energy_canaries
    try:
        pope.main()
    finally:
        baseline.eval_val = original_eval_val


if __name__ == "__main__":
    main()

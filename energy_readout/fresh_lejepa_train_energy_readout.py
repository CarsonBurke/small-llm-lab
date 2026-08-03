"""Energy-readout LeJEPA: token emission straight from the tied codebook.

Fork of the belief-attached lejepa-ce variant.  The 4.2M-parameter policy
probe and the untrained 3.15M-parameter critic probe are deleted after parent
construction (all RNG consumption happens first, so every surviving parameter
is initialized bit-identically to the parent).  Vocabulary logits are the
negative squared distance between the JEPA prediction and the projected tied
codebook with a learned inverse-temperature and per-token bias:

    logit_k = -exp(log_s) * ||z_hat - c_k||^2 / 2 + b_k

CE through these logits carries both the discriminative signal and the
attractive pull toward the target code, so the explicit latent MSE defaults
to weight zero (``ENERGY_LATENT_MSE_WEIGHT`` restores it).  No softcap on
this head: distance logits are one-sided and tanh would saturate the target's
logit exactly when its code is far.  This family supports pretraining only;
post-training value/generation paths raise.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
from pretraining.fresh_lejepa import fresh_lejepa_train_v1_probe_shared_rms_pope as pope
import train_gpt as baseline
from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope_belief_attached import (
    FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE,
)

ARCHITECTURE = "energy_readout_lejepa_tied_codebook_onepass_2k"

LATENT_MSE_WEIGHT = float(os.environ.get("ENERGY_LATENT_MSE_WEIGHT", "0.0"))
HEAD_FORM = os.environ.get("ENERGY_HEAD_FORM", "distance")
DETACH_CODEBOOK = bool(int(os.environ.get("ENERGY_DETACH_CODEBOOK", "0")))
BIGRAM_TABLE = bool(int(os.environ.get("ENERGY_BIGRAM_TABLE", "0")))
# Guard rails on the learned inverse-temperature: wide enough to never bind
# in healthy training, tight enough to stop a runaway s from compensating a
# collapsing codebook.  Applied as a straight-through clamp so the gradient
# stays alive at the bounds and the parameter can re-enter the window.
SCALE_MIN = float(os.environ.get("ENERGY_SCALE_MIN", "1e-5"))
SCALE_MAX = float(os.environ.get("ENERGY_SCALE_MAX", "100.0"))
# Optional warm start b_k = log p(k) from a JSON list of per-token counts;
# pays off in early-step BPB at the 2k ablation scale.  Resolved at import
# time because v1.main may chdir into FRESH_LEJEPA_WORK_DIR before the model
# is constructed.
BIAS_INIT_COUNTS = os.environ.get("ENERGY_BIAS_INIT_COUNTS", "")
if BIAS_INIT_COUNTS:
    BIAS_INIT_COUNTS = str(Path(BIAS_INIT_COUNTS).resolve())


class EnergyReadoutLeJEPA(FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE):
    """Belief-attached lejepa-ce with probes replaced by a codebook energy head."""

    latent_loss_weight = LATENT_MSE_WEIGHT

    def __init__(self, *args, **kwargs):
        vocab_size = kwargs.get("vocab_size", args[0] if args else None)
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        super().__init__(*args, **kwargs)
        owner = self.blocks[-1]
        # The parent lineage direct-assigns both probes during construction;
        # deleting them afterwards leaves the RNG stream — and therefore every
        # remaining parameter's initialization — identical to the parent.
        del owner.policy_probe
        del owner.critic_probe
        # Registered under blocks[-1] so baseline.main's optimizer split sees
        # them: ndim < 2 routes both to the Adam scalar group, and
        # restore_low_dim_params_to_fp32 keeps fp32 masters.
        owner.energy_log_scale = nn.Parameter(
            torch.tensor(-0.5 * math.log(model_dim), dtype=torch.float32)
        )
        owner.energy_bias = nn.Parameter(
            torch.zeros(vocab_size, dtype=torch.float32)
        )
        if BIAS_INIT_COUNTS:
            counts = torch.tensor(
                json.loads(Path(BIAS_INIT_COUNTS).read_text()),
                dtype=torch.float64,
            )
            if counts.shape != (vocab_size,) or bool((counts < 0).any()):
                raise ValueError(
                    f"{BIAS_INIT_COUNTS} must hold {vocab_size} nonnegative counts"
                )
            # Laplace smoothing keeps zero-count tokens at a finite bias.
            probs = (counts + 1.0) / (counts.sum() + vocab_size)
            with torch.no_grad():
                owner.energy_bias.copy_(probs.log().float())
        if BIGRAM_TABLE:
            # Flat 1-D on purpose: a (V, V) parameter under blocks would be
            # Muon-orthogonalized (wrong for a frequency-indexed lookup table)
            # and bf16-mastered; ndim 1 lands in fused Adam with fp32 masters
            # and serializes with a single per-tensor int8 scale.
            owner.energy_bigram = nn.Parameter(
                torch.zeros(vocab_size * vocab_size, dtype=torch.float32)
            )
        leftover = [name for name, _ in self.named_parameters() if "probe" in name]
        if leftover:
            raise RuntimeError(f"probe parameters survived deletion: {leftover}")

    def energy_codebook(self) -> Tensor:
        raw = F.rms_norm(self.tok_emb.weight, (self.tok_emb.embedding_dim,))
        codebook = self.latent_projector(raw)
        return codebook.detach() if DETACH_CODEBOOK else codebook

    def energy_logits(self, predicted: Tensor, input_ids: Tensor | None) -> Tensor:
        codebook = self.energy_codebook()
        owner = self.blocks[-1]
        log_scale = owner.energy_log_scale
        # Straight-through: forward uses the clamped value, backward passes
        # the unclamped gradient, so a bound is never sticky.
        log_scale = log_scale + (
            log_scale.clamp(math.log(SCALE_MIN), math.log(SCALE_MAX)) - log_scale
        ).detach()
        scale = torch.exp(log_scale)
        # bf16 tensor-core matmul with fp32 accumulation, then fp32 for the
        # squared norms: at ||c||^2 ~ O(d) a bf16 square-sum loses ~0.2
        # absolute, which is free to avoid.
        dots = (predicted @ codebook.transpose(0, 1)).float()
        if HEAD_FORM == "dot":
            logits = scale * dots + owner.energy_bias
        else:
            z_sq = predicted.float().square().sum(dim=-1, keepdim=True)
            c_sq = codebook.float().square().sum(dim=-1)
            logits = owner.energy_bias - 0.5 * scale * (z_sq - 2.0 * dots + c_sq)
        if BIGRAM_TABLE:
            if input_ids is None:
                raise RuntimeError(
                    "ENERGY_BIGRAM_TABLE requires current-token ids at the readout"
                )
            vocab = self.tok_emb.num_embeddings
            logits = logits + owner.energy_bigram.view(vocab, vocab)[input_ids]
        return logits

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        token_latent, _belief, predicted, target_latent = (
            self.training_latents_with_belief(input_ids, target_ids)
        )
        logits = self.energy_logits(
            predicted, input_ids if BIGRAM_TABLE else None
        )
        policy_loss = F.cross_entropy(
            logits.float().flatten(0, 1), target_ids.flatten()
        )
        if not self.training:
            return policy_loss

        total_loss = policy_loss
        # type(self).latent_loss_weight is the single authority: the patched
        # accumulation loop recomposes the logged train loss from this same
        # attribute, so gradient and logs cannot diverge.
        latent_weight = type(self).latent_loss_weight
        if latent_weight != 0.0:
            latent_loss = F.mse_loss(predicted.float(), target_latent.float())
            total_loss = total_loss + latent_weight * latent_loss
            latent_component = latent_loss.detach()
        else:
            # Reported as component[1] purely as a diagnostic: how far the
            # CE-shaped prediction drifts from latent-space regression.
            with torch.no_grad():
                latent_component = F.mse_loss(
                    predicted.float(), target_latent.float()
                )
        if self.defer_sigreg:
            sigreg_loss = policy_loss.detach().new_zeros(())
        else:
            sigreg_loss = self.sigreg(
                self.training_sigreg_features(token_latent, target_latent)
            )
            total_loss = total_loss + type(self).sigreg_loss_weight * sigreg_loss
        if self.return_loss_components:
            return total_loss, torch.stack(
                (policy_loss.detach(), latent_component, sigreg_loss.detach())
            )
        return total_loss

    def logits_from_features(self, features: Tensor) -> Tensor:
        # Features are predicted latents; no softcap on the energy head.
        return self.energy_logits(features, None)

    def detached_probe_features(self, input_ids: Tensor) -> Tensor:
        token_latent = self.embed_tokens(input_ids)
        belief = self.temporal_belief_from_token_latent(token_latent)
        return self.prediction_latent(belief).detach()

    def generation_probe_features(
        self, token_latent: Tensor, belief: Tensor, predicted: Tensor
    ) -> Tensor:
        del token_latent, belief
        return predicted.detach()

    def values_from_features(self, features: Tensor) -> Tensor:
        raise RuntimeError(
            "the energy_readout family has no critic; post-training value "
            "paths are unsupported"
        )

    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int | float]:
        metadata = super().experiment_metadata()
        metadata.update(
            {
                "emission_head": "tied_codebook_energy",
                "energy_head_form": HEAD_FORM,
                "energy_codebook_gradient": (
                    "detached" if DETACH_CODEBOOK else "attached"
                ),
                "energy_log_scale_init": -0.5
                * math.log(v1.FreshHyperparameters.model_dim),
                "energy_softcap": "none",
                "energy_bigram_table": int(BIGRAM_TABLE),
                "energy_scale_clamp": f"[{SCALE_MIN:g}, {SCALE_MAX:g}]",
                "energy_bias_init": (
                    "log_unigram_laplace" if BIAS_INIT_COUNTS else "zeros"
                ),
                "latent_mse_weight": LATENT_MSE_WEIGHT,
                "probes": "deleted_policy_and_critic",
            }
        )
        return metadata


def main() -> None:
    if pope.PolarCausalSelfAttention.position_mode != "pope":
        raise ValueError("the energy-readout variant is defined only for PoPE")
    if HEAD_FORM not in {"distance", "dot"}:
        raise ValueError(f"unknown ENERGY_HEAD_FORM={HEAD_FORM!r}")
    # Single-source the latent-MSE weight: the forward reads
    # FreshHyperparameters.latent_loss_weight, and the patched accumulation
    # loop recomposes the logged train loss from the model class attribute.
    # Both must agree with ENERGY_LATENT_MSE_WEIGHT or the logged loss would
    # silently diverge from the optimized one.
    v1.FreshHyperparameters.latent_loss_weight = LATENT_MSE_WEIGHT
    EnergyReadoutLeJEPA.latent_loss_weight = LATENT_MSE_WEIGHT
    # Compile warmup would consume the corpus prefix twice even though its
    # state is restored; the parent's one-pass contract forbids that reuse.
    v1.FreshHyperparameters.warmup_steps = 0
    pope.FreshLeJEPASharedRMSV1PoPE = EnergyReadoutLeJEPA
    pope.POPE_ARCHITECTURE = ARCHITECTURE
    pope.__file__ = str(Path(__file__).resolve())

    # Collapse canaries, once per validation: the learned inverse-temperature
    # plus codebook norm/spread.  With the codebook undetached, CE is
    # scale-invariant and SIGReg alone holds the geometry open — a shrinking
    # pairwise distance with a compensating rising s is the joint-collapse
    # signature.  v1.main wraps whatever baseline.eval_val is at call time, so
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
            scale = float(
                torch.exp(
                    log_scale.clamp(math.log(SCALE_MIN), math.log(SCALE_MAX))
                )
            )
            codebook = base.energy_codebook().detach().float()
            norm_mean = float(codebook.norm(dim=-1).mean())
            pairdist_mean = float(torch.cdist(codebook, codebook).mean())
        # churn_stats is scripts/ablation.py's stepless fold channel: emitted before
        # baseline.main prints the val line, so these keys land inside that
        # val entry in metrics.jsonl and stream to TensorBoard.  Fixed-point
        # formatting is required — the extras parser rejects scientific
        # notation.
        print(
            f"churn_stats energy_scale:{scale:.8f} "
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

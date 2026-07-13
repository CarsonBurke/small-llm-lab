"""Clean shared-RMS LeJEPA with compact learned-output policy and critic probes."""

from __future__ import annotations

import inspect
import os
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import fresh_lejepa_train as v1
import train_gpt as baseline
from train_gpt import CastedLinear, RMSNorm


ARCHITECTURE = "fresh_lejepa_v4_shared_rms_fp32_sigreg_dwide_learned_probes"
SIGREG_PROJECTION_CHUNK = 64
SIGREG_POSITION_CHUNK = 64


class RMSTokenProjector(nn.Module):
    """Linear -> per-token RMSNorm -> GELU -> Linear, with no output norm."""

    def __init__(self, model_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.input = CastedLinear(model_dim, hidden_dim)
        self.norm = RMSNorm()
        self.output = CastedLinear(hidden_dim, model_dim)

    def forward(self, latent: Tensor) -> Tensor:
        return self.output(F.gelu(self.norm(self.input(latent))))

    def inference(self, latent: Tensor) -> Tensor:
        return self(latent)


class DWideFusion(nn.Module):
    """Fuse two D-wide latents and process them with a residual D-wide trunk."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.token = CastedLinear(model_dim, model_dim)
        self.predicted = CastedLinear(model_dim, model_dim)
        self.hidden1 = CastedLinear(model_dim, model_dim)
        self.hidden2 = CastedLinear(model_dim, model_dim)

    def forward(self, features: Tensor) -> Tensor:
        token, predicted = features.chunk(2, dim=-1)
        hidden = F.rms_norm(
            self.token(token) + self.predicted(predicted), (token.size(-1),)
        )
        hidden = hidden + F.silu(self.hidden1(hidden))
        hidden = hidden + F.silu(
            self.hidden2(F.rms_norm(hidden, (hidden.size(-1),)))
        )
        return F.rms_norm(hidden, (hidden.size(-1),))


class LearnedOutputPolicyProbe(DWideFusion):
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__(model_dim)
        self.output = CastedLinear(model_dim, vocab_size, bias=False)
        self.output._zero_init = True
        nn.init.zeros_(self.output.weight)

    def forward(self, features: Tensor) -> Tensor:
        return self.output(super().forward(features))


class LearnedOutputCriticProbe(DWideFusion):
    def __init__(self, model_dim: int):
        super().__init__(model_dim)
        self.output = CastedLinear(model_dim, 1, bias=False)
        self.output._zero_init = True
        nn.init.zeros_(self.output.weight)

    def forward(self, features: Tensor) -> Tensor:
        return self.output(super().forward(features))


class FreshLeJEPAV4(v1.FreshLeJEPAGPT):
    defer_sigreg = True
    sigreg_loss_weight = v1.FreshHyperparameters.sigreg_weight

    def make_policy_probe(self, model_dim: int, vocab_size: int) -> nn.Module:
        return LearnedOutputPolicyProbe(model_dim, vocab_size)

    def make_critic_probe(self, model_dim: int) -> nn.Module:
        return LearnedOutputCriticProbe(model_dim)

    def __init__(self, *args, **kwargs):
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        super().__init__(*args, **kwargs)
        if model_dim is None:
            raise ValueError("model_dim is required")
        owner = self.blocks[-1]
        owner.latent_projector = RMSTokenProjector(model_dim)
        owner.prediction_projector = RMSTokenProjector(model_dim)

    @property
    def latent_projector(self) -> RMSTokenProjector:
        return self.blocks[-1].latent_projector

    @property
    def prediction_projector(self) -> RMSTokenProjector:
        return self.blocks[-1].prediction_projector

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        folded = getattr(self, "_folded_token_latents", None)
        if folded is not None:
            return F.embedding(input_ids, folded)
        raw = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        return self.latent_projector(raw)

    @torch.no_grad()
    def fold_input_projector_for_inference(self) -> None:
        """Cache the frozen context-free token path as one projected table."""
        raw = F.rms_norm(self.tok_emb.weight, (self.tok_emb.embedding_dim,))
        table = self.latent_projector.inference(raw).detach()
        object.__setattr__(self, "_folded_token_latents", table)

    def clear_folded_input_projector(self) -> None:
        self.__dict__.pop("_folded_token_latents", None)

    def prediction_latent(self, predicted: Tensor) -> Tensor:
        return self.prediction_projector(predicted)

    def training_latents(
        self, input_ids: Tensor, target_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        trajectory_ids = torch.cat((input_ids, target_ids[:, -1:]), dim=1)
        trajectory_latent = self.embed_tokens(trajectory_ids)
        token_latent = trajectory_latent[:, :-1]
        target_latent = trajectory_latent[:, 1:]
        predicted = self.predict_from_token_latent(token_latent)
        return token_latent, predicted, target_latent

    def training_sigreg_features(
        self, token_latent: Tensor, target_latent: Tensor
    ) -> Tensor:
        return torch.cat((token_latent, target_latent[:, -1:]), dim=1)

    def deferred_sigreg_loss(
        self,
        first_input_ids: Tensor,
        first_target_ids: Tensor,
        second_input_ids: Tensor,
        second_target_ids: Tensor,
    ) -> Tensor:
        """Compute one exact B=128 statistic from two baseline B=64 batches."""
        first = torch.cat((first_input_ids, first_target_ids[:, -1:]), dim=1)
        second = torch.cat((second_input_ids, second_target_ids[:, -1:]), dim=1)
        trajectory_latent = self.embed_tokens(torch.cat((first, second), dim=0))
        return self.sigreg(trajectory_latent)


def _install_configurable_accumulation(default_steps: int = 8):
    """Fork baseline.main in memory without modifying the upstream file."""
    original = baseline.main
    source = textwrap.dedent(inspect.getsource(original))
    old = (
        "if 8 % world_size != 0:\n"
        "        raise ValueError(f\"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral\")\n"
        "    grad_accum_steps = 8 // world_size"
    )
    new = (
        f"total_grad_accum_steps = int(os.environ.get('GRAD_ACCUM_STEPS', '{default_steps}'))\n"
        "    if total_grad_accum_steps <= 0 or total_grad_accum_steps % world_size != 0:\n"
        "        raise ValueError(\n"
        "            f\"GRAD_ACCUM_STEPS={total_grad_accum_steps} must be positive and divisible by WORLD_SIZE={world_size}\"\n"
        "        )\n"
        "    grad_accum_steps = total_grad_accum_steps // world_size\n"
        "    if grad_accum_steps % 2 != 0:\n"
        "        raise ValueError(\"local grad accumulation must be even for paired B=128 SIGReg\")\n"
        "    local_sequences = args.train_batch_tokens // (world_size * grad_accum_steps * args.train_seq_len)\n"
        "    if 2 * local_sequences != 128:\n"
        "        raise ValueError(\n"
        "            f\"paired SIGReg requires two B=64 microbatches, got B={local_sequences}\"\n"
        "        )"
    )
    if source.count(old) != 1:
        raise RuntimeError("upstream grad accumulation block changed")
    source = source.replace(old, new)
    warmup_old = (
        "warmup_loss = model(x, y)\n"
        "                (warmup_loss * grad_scale).backward()"
    )
    warmup_new = (
        "warmup_loss, _warmup_components = model(x, y)\n"
        "                (warmup_loss * grad_scale).backward()\n"
        "                if micro_step % 2 == 0:\n"
        "                    pending_sigreg_batch = (x, y)\n"
        "                else:\n"
        "                    paired_sigreg = base_model.deferred_sigreg_loss(\n"
        "                        pending_sigreg_batch[0], pending_sigreg_batch[1], x, y\n"
        "                    )\n"
        "                    (paired_sigreg * base_model.sigreg_loss_weight * 2 * grad_scale).backward()"
    )
    train_init_old = 'train_loss = torch.zeros((), device=device)'
    train_init_new = (
        'train_loss = torch.zeros((), device=device)\n'
        '        train_components = torch.zeros(3, device=device)'
    )
    train_forward_old = (
        "loss = model(x, y)\n"
        "            train_loss += loss.detach()"
    )
    train_forward_new = (
        "loss, loss_components = model(x, y)\n"
        "            train_loss += loss.detach()\n"
        "            train_components += loss_components\n"
        "            (loss * grad_scale).backward()\n"
        "            if micro_step % 2 == 0:\n"
        "                pending_sigreg_batch = (x, y)\n"
        "            else:\n"
        "                paired_sigreg = base_model.deferred_sigreg_loss(\n"
        "                    pending_sigreg_batch[0], pending_sigreg_batch[1], x, y\n"
        "                )\n"
        "                (paired_sigreg * base_model.sigreg_loss_weight * 2 * grad_scale).backward()\n"
        "                train_loss += 2 * base_model.sigreg_loss_weight * paired_sigreg.detach()\n"
        "                train_components[2] += 2 * paired_sigreg.detach()"
    )
    train_mean_old = (
        '(loss * grad_scale).backward()\n'
        '        train_loss /= grad_accum_steps'
    )
    train_mean_new = (
        'train_loss /= grad_accum_steps\n'
        '        train_components /= grad_accum_steps'
    )
    log_old = (
        'f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "\n'
        '                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"'
    )
    log_new = (
        'f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "\n'
        '                f"policy_loss:{train_components[0].item():.4f} "\n'
        '                f"latent_loss:{train_components[1].item():.4f} "\n'
        '                f"sigreg_loss:{train_components[2].item():.4f} "\n'
        '                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"'
    )
    for expected, replacement, label in (
        (warmup_old, warmup_new, "warmup output"),
        (train_init_old, train_init_new, "component accumulator"),
        (train_forward_old, train_forward_new, "training output"),
        (train_mean_old, train_mean_new, "component mean"),
        (log_old, log_new, "component logging"),
    ):
        if source.count(expected) != 1:
            raise RuntimeError(f"upstream {label} block changed")
        source = source.replace(expected, replacement)
    exec(compile(source, inspect.getsourcefile(original) or "train_gpt.py", "exec"), baseline.__dict__)
    return original


def main() -> None:
    original_main = _install_configurable_accumulation(default_steps=8)
    FreshLeJEPAV4.return_loss_components = True
    v1.FreshHyperparameters.sigreg_proj_chunk = SIGREG_PROJECTION_CHUNK
    v1.FreshHyperparameters.sigreg_position_chunk = SIGREG_POSITION_CHUNK
    v1.FreshLeJEPAGPT = FreshLeJEPAV4
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    os.environ.setdefault("GRAD_ACCUM_STEPS", "8")
    try:
        v1.main()
    finally:
        baseline.main = original_main


if __name__ == "__main__":
    main()

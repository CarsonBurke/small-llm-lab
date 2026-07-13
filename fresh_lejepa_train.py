"""Fresh LeJEPA language-model experiment built only from the upstream baseline.

The latent backbone is trained by an attached-target next-embedding loss and
SIGReg.  Detached policy/critic probes consume the current token embedding and
the predicted next latent.  ``train_gpt.main`` remains the canonical loader,
optimizer, evaluator, and serializer; this module replaces only its model and
adds experiment hyperparameters.
"""

from __future__ import annotations

import json
import hashlib
import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

import train_gpt as baseline

EXPERIMENT_ARCHITECTURE = "fresh_lejepa_attached_target_detached_probes_v1"
EXPERIMENT_SOURCE = Path(__file__)


class FreshHyperparameters(baseline.Hyperparameters):
    latent_loss_weight = float(os.environ.get("FRESH_LEJEPA_LATENT_WEIGHT", "1.0"))
    sigreg_weight = float(os.environ.get("FRESH_LEJEPA_SIGREG_WEIGHT", "0.09"))
    sigreg_knots = int(os.environ.get("FRESH_LEJEPA_SIGREG_KNOTS", "17"))
    sigreg_num_proj = int(os.environ.get("FRESH_LEJEPA_SIGREG_NUM_PROJ", "1024"))
    sigreg_proj_chunk = int(os.environ.get("FRESH_LEJEPA_SIGREG_PROJ_CHUNK", "64"))
    sigreg_position_chunk = int(os.environ.get("FRESH_LEJEPA_SIGREG_POSITION_CHUNK", "64"))


class ResidualProbe(nn.Module):
    """Four linear layers with three non-compressing 2D hidden states."""

    def __init__(self, model_dim: int, output_dim: int):
        super().__init__()
        width = 2 * model_dim
        self.input = baseline.CastedLinear(width, width)
        self.hidden1 = baseline.CastedLinear(width, width)
        self.hidden2 = baseline.CastedLinear(width, width)
        self.output = baseline.CastedLinear(width, output_dim, bias=False)
        self.output._zero_init = True
        nn.init.zeros_(self.output.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = F.silu(self.input(x))
        x = x + F.silu(self.hidden1(F.rms_norm(x, (x.size(-1),))))
        x = x + F.silu(self.hidden2(F.rms_norm(x, (x.size(-1),))))
        return self.output(F.rms_norm(x, (x.size(-1),)))


def _sigreg_projection_statistic(
    embeddings: Tensor,
    directions: Tensor,
    t: Tensor,
    phi: Tensor,
    weights: Tensor,
    position_chunk: int,
) -> Tensor:
    """One projection shard, factored out so backward can recompute its activations."""
    batch, length, _ = embeddings.shape
    statistic = embeddings.new_zeros((), dtype=torch.float32)
    for position_start in range(0, length, position_chunk):
        samples = embeddings[
            :, position_start : position_start + position_chunk
        ].transpose(0, 1).float()
        projected = samples @ directions
        x_t = projected.unsqueeze(-1) * t
        err = (x_t.cos().mean(dim=1) - phi).square()
        err = err + x_t.sin().mean(dim=1).square()
        statistic = statistic + (err @ weights).sum() * batch
    return statistic


class SIGReg(nn.Module):
    """LeWM's Epps-Pulley Gaussian regularizer, streamed over projections."""

    def __init__(self, knots: int, num_proj: int, proj_chunk: int, position_chunk: int):
        super().__init__()
        self.num_proj = num_proj
        self.proj_chunk = proj_chunk
        self.position_chunk = position_chunk
        self.checkpoint_projection_chunks = True
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # The characteristic-function quadrature is a numerical statistic, not
        # model state.  Keep it in FP32 when the surrounding model is cast to
        # BF16, matching the LeJEPA/LeWM reference implementations.
        for name in ("t", "phi", "weights"):
            setattr(self, name, getattr(self, name).float())
        return self

    @torch.compiler.disable
    def forward(self, embeddings: Tensor) -> Tensor:
        # LeWM regularizes the batch distribution independently at every time.
        # Chunk both axes, but preserve the exact all-position/all-projection mean.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            batch, length, dim = embeddings.shape
            statistic = embeddings.new_zeros((), dtype=torch.float32)
            for projection_start in range(0, self.num_proj, self.proj_chunk):
                width = min(self.proj_chunk, self.num_proj - projection_start)
                directions = torch.randn(dim, width, device=embeddings.device, dtype=torch.float32)
                directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
                args = (
                    embeddings,
                    directions,
                    self.t,
                    self.phi,
                    self.weights,
                    self.position_chunk,
                )
                if self.checkpoint_projection_chunks and embeddings.requires_grad:
                    contribution = checkpoint(
                        _sigreg_projection_statistic, *args, use_reentrant=False
                    )
                else:
                    contribution = _sigreg_projection_statistic(*args)
                statistic = statistic + contribution
            return statistic / (length * self.num_proj)


class NonCachingRotary(baseline.Rotary):
    """RoPE tables local to a forward, safe across inference/train graphs."""

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        frequencies = torch.outer(positions, self.inv_freq.to(device))
        return frequencies.cos()[None, None].to(dtype), frequencies.sin()[None, None].to(dtype)


class FreshLeJEPAGPT(baseline.GPT):
    """Baseline causal predictor plus detached policy and critic probes."""

    return_loss_components = False
    defer_sigreg = False

    def __init__(self, *args, **kwargs):
        vocab_size = kwargs.get("vocab_size", args[0] if args else None)
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        super().__init__(*args, **kwargs)
        if vocab_size is None or model_dim is None:
            raise ValueError("vocab_size and model_dim are required")

        # Register these under blocks so the unmodified baseline optimizer sees
        # every experimental matrix.  The critic has no gradients in pretraining.
        owner = self.blocks[-1]
        self.install_probes(owner, model_dim, vocab_size)
        for block in self.blocks:
            rotary = NonCachingRotary(block.attn.head_dim)
            rotary.inv_freq.copy_(block.attn.rotary.inv_freq)
            block.attn.rotary = rotary
        self.sigreg = SIGReg(
            FreshHyperparameters.sigreg_knots,
            FreshHyperparameters.sigreg_num_proj,
            FreshHyperparameters.sigreg_proj_chunk,
            FreshHyperparameters.sigreg_position_chunk,
        )

    def make_policy_probe(self, model_dim: int, vocab_size: int) -> nn.Module:
        return ResidualProbe(model_dim, vocab_size)

    def make_critic_probe(self, model_dim: int) -> nn.Module:
        return ResidualProbe(model_dim, 1)

    def install_probes(
        self, owner: nn.Module, model_dim: int, vocab_size: int
    ) -> None:
        owner.policy_probe = self.make_policy_probe(model_dim, vocab_size)
        owner.critic_probe = self.make_critic_probe(model_dim)

    @property
    def policy_probe(self) -> ResidualProbe:
        return self.blocks[-1].policy_probe

    @property
    def critic_probe(self) -> ResidualProbe:
        return self.blocks[-1].critic_probe

    def train(self, mode: bool = True):
        # Validation runs under torch.inference_mode and Rotary caches its
        # tables.  Such tensors cannot later be saved for backward, so discard
        # them on the eval->train transition.
        was_training = self.training
        result = super().train(mode)
        if mode and not was_training:
            for block in self.blocks:
                block.attn.rotary._cos_cached = None
                block.attn.rotary._sin_cached = None
                block.attn.rotary._seq_len_cached = 0
        return result

    def latent_features(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        token_latent = self.embed_tokens(input_ids)
        return token_latent, self.predict_from_token_latent(token_latent)

    def temporal_belief_from_token_latent(self, token_latent: Tensor) -> Tensor:
        predicted = token_latent
        x0 = token_latent
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            predicted = self.blocks[i](predicted, x0)
            skips.append(predicted)
        for i in range(self.num_decoder_layers):
            if skips:
                predicted = predicted + self.skip_weights[i].to(predicted.dtype)[None, None, :] * skips.pop()
            predicted = self.blocks[self.num_encoder_layers + i](predicted, x0)
        return self.final_norm(predicted)

    def predict_from_token_latent(self, token_latent: Tensor) -> Tensor:
        return self.prediction_latent(
            self.temporal_belief_from_token_latent(token_latent)
        )

    def training_latents(
        self, input_ids: Tensor, target_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        token_latent, predicted = self.latent_features(input_ids)
        return token_latent, predicted, self.target_latent(target_ids)

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        return F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))

    def detached_probe_features(self, input_ids: Tensor) -> Tensor:
        token_latent, predicted = self.latent_features(input_ids)
        return self.probe_features(token_latent, predicted)

    def probe_features(self, token_latent: Tensor, predicted: Tensor) -> Tensor:
        return torch.cat((token_latent.detach(), predicted.detach()), dim=-1)

    def sigreg_features(self, token_latent: Tensor) -> Tensor:
        return token_latent

    def training_sigreg_features(
        self, token_latent: Tensor, target_latent: Tensor
    ) -> Tensor:
        return self.sigreg_features(token_latent)

    def prediction_latent(self, predicted: Tensor) -> Tensor:
        return predicted

    def target_latent(self, target_ids: Tensor) -> Tensor:
        return self.embed_tokens(target_ids)

    def policy_logits(self, input_ids: Tensor) -> Tensor:
        return self.logits_from_features(self.detached_probe_features(input_ids))

    def logits_from_features(self, features: Tensor) -> Tensor:
        raw = self.policy_probe(features)
        return self.logit_softcap * torch.tanh(raw / self.logit_softcap)

    def values(self, input_ids: Tensor) -> Tensor:
        return self.values_from_features(self.detached_probe_features(input_ids))

    def values_from_features(self, features: Tensor) -> Tensor:
        return self.critic_probe(features).squeeze(-1)

    def _attention_step(
        self,
        attention: baseline.CausalSelfAttention,
        x: Tensor,
        cache: tuple[Tensor, Tensor],
        position: int | Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """One-token GQA attention step used by post-training rollouts."""
        batch, _, dim = x.shape
        q_dim = attention.num_heads * attention.head_dim
        kv_dim = attention.num_kv_heads * attention.head_dim
        q, k, v = attention.c_qkv(x).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(batch, 1, attention.num_heads, attention.head_dim).transpose(1, 2)
        k = k.view(batch, 1, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
        v = v.view(batch, 1, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        frequency = position * attention.rotary.inv_freq.to(x.device)
        cos = frequency.cos()[None, None, None].to(q.dtype)
        sin = frequency.sin()[None, None, None].to(q.dtype)
        q = baseline.apply_rotary_emb(q, cos, sin)
        k = baseline.apply_rotary_emb(k, cos, sin)
        q = q * attention.q_gain.to(q.dtype)[None, :, None, None]
        if torch.is_tensor(position):
            index = position.reshape(1)
            cache[0].index_copy_(2, index, k)
            cache[1].index_copy_(2, index, v)
            prefix_k = torch.narrow(cache[0], 2, 0, position + 1)
            prefix_v = torch.narrow(cache[1], 2, 0, position + 1)
        else:
            cache[0][:, :, position : position + 1].copy_(k)
            cache[1][:, :, position : position + 1].copy_(v)
            prefix_k = cache[0][:, :, : position + 1]
            prefix_v = cache[1][:, :, : position + 1]
        y = F.scaled_dot_product_attention(
            q, prefix_k, prefix_v, is_causal=False,
            enable_gqa=attention.num_kv_heads != attention.num_heads,
        )
        y = y.transpose(1, 2).contiguous().view(batch, 1, dim)
        return attention.proj(y), cache

    def _block_step(
        self,
        block: baseline.Block,
        x: Tensor,
        x0: Tensor,
        cache: tuple[Tensor, Tensor],
        position: int | Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        mix = block.resid_mix.to(x.dtype)
        x = mix[0][None, None] * x + mix[1][None, None] * x0
        attn, cache = self._attention_step(block.attn, block.attn_norm(x), cache, position)
        x = x + block.attn_scale.to(x.dtype)[None, None] * attn
        x = x + block.mlp_scale.to(x.dtype)[None, None] * block.mlp(block.mlp_norm(x))
        return x, cache

    def generation_step(
        self,
        token_ids: Tensor,
        caches: list[tuple[Tensor, Tensor]],
        position: int | Tensor,
    ) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        """Consume one token and return next-token logits, value, and KV caches."""
        token_latent = self.embed_tokens(token_ids[:, None])
        predicted = token_latent
        skips: list[Tensor] = []
        next_caches = list(caches)
        for i in range(self.num_encoder_layers):
            predicted, next_caches[i] = self._block_step(
                self.blocks[i], predicted, token_latent, caches[i], position
            )
            skips.append(predicted)
        for j in range(self.num_decoder_layers):
            i = self.num_encoder_layers + j
            if skips:
                predicted = predicted + self.skip_weights[j].to(predicted.dtype)[None, None] * skips.pop()
            predicted, next_caches[i] = self._block_step(
                self.blocks[i], predicted, token_latent, caches[i], position
            )
        belief = self.final_norm(predicted)
        predicted = self.prediction_latent(belief)
        features = self.generation_probe_features(token_latent, belief, predicted)
        logits = self.logits_from_features(features).squeeze(1)
        value = self.values_from_features(features).squeeze(1)
        return logits, value, next_caches

    def generation_probe_features(
        self, token_latent: Tensor, belief: Tensor, predicted: Tensor
    ) -> Tensor:
        return self.probe_features(token_latent, predicted)

    def make_generation_cache(
        self, batch_size: int, max_length: int, device: torch.device
    ) -> list[tuple[Tensor, Tensor]]:
        caches = []
        for block in self.blocks:
            attention = block.attn
            shape = (batch_size, attention.num_kv_heads, max_length, attention.head_dim)
            caches.append(
                (
                    torch.empty(shape, device=device, dtype=self.tok_emb.weight.dtype),
                    torch.empty(shape, device=device, dtype=self.tok_emb.weight.dtype),
                )
            )
        return caches

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        token_latent, predicted, target_latent = self.training_latents(input_ids, target_ids)
        detached = self.probe_features(token_latent, predicted)
        logits = self.logits_from_features(detached)
        policy_loss = F.cross_entropy(logits.float().flatten(0, 1), target_ids.flatten())
        if not self.training:
            return policy_loss

        # Deliberately attached: gradients enter the target embedding branch.
        latent_loss = F.mse_loss(predicted.float(), target_latent.float())
        if self.defer_sigreg:
            sigreg_loss = policy_loss.detach().new_zeros(())
        else:
            sigreg_loss = self.sigreg(
                self.training_sigreg_features(token_latent, target_latent)
            )
        total_loss = (
            policy_loss
            + FreshHyperparameters.latent_loss_weight * latent_loss
            + FreshHyperparameters.sigreg_weight * sigreg_loss
        )
        if self.return_loss_components:
            return total_loss, torch.stack(
                (policy_loss.detach(), latent_loss.detach(), sigreg_loss.detach())
            )
        return total_loss


def main() -> None:
    work_dir = os.environ.get("FRESH_LEJEPA_WORK_DIR")
    if work_dir:
        data_path = Path(FreshHyperparameters.data_path).resolve()
        tokenizer_path = Path(FreshHyperparameters.tokenizer_path).resolve()
        FreshHyperparameters.data_path = str(data_path)
        FreshHyperparameters.train_files = str(data_path / "fineweb_train_*.bin")
        FreshHyperparameters.val_files = str(data_path / "fineweb_val_*.bin")
        FreshHyperparameters.tokenizer_path = str(tokenizer_path)
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(work_dir)
    tracked_optimizers = []
    original_adam = baseline.torch.optim.Adam
    original_muon = baseline.Muon
    original_compile = baseline.torch.compile

    def tracked_adam(*args, **kwargs):
        optimizer = original_adam(*args, **kwargs)
        tracked_optimizers.append(optimizer)
        return optimizer

    class TrackedMuon(original_muon):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            tracked_optimizers.append(self)

    def experiment_compile(target, *args, **kwargs):
        # Keep the transformer/probes compiled, but permit the exact chunked
        # SIGReg routine to run eagerly instead of unrolling hundreds of chunks
        # into an intractably large full graph.
        if isinstance(target, nn.Module):
            kwargs["fullgraph"] = False
        return original_compile(target, *args, **kwargs)

    baseline.Hyperparameters = FreshHyperparameters
    baseline.GPT = FreshLeJEPAGPT
    baseline.torch.optim.Adam = tracked_adam
    baseline.Muon = TrackedMuon
    baseline.torch.compile = experiment_compile
    try:
        baseline.main()
    finally:
        baseline.torch.optim.Adam = original_adam
        baseline.Muon = original_muon
        baseline.torch.compile = original_compile
    if int(os.environ.get("RANK", "0")) == 0:
        tokenizer = Path(FreshHyperparameters.tokenizer_path)
        metadata = {
            "architecture": EXPERIMENT_ARCHITECTURE,
            "model": {
                key: getattr(FreshHyperparameters, key)
                for key in (
                    "vocab_size", "num_layers", "model_dim", "num_heads",
                    "num_kv_heads", "mlp_mult", "tie_embeddings", "rope_base",
                    "logit_softcap", "qk_gain_init", "tied_embed_init_std",
                )
            },
            "loss": {
                "latent_weight": FreshHyperparameters.latent_loss_weight,
                "sigreg_weight": FreshHyperparameters.sigreg_weight,
                "sigreg_knots": FreshHyperparameters.sigreg_knots,
                "sigreg_num_proj": FreshHyperparameters.sigreg_num_proj,
                "sigreg_proj_chunk": FreshHyperparameters.sigreg_proj_chunk,
                "sigreg_position_chunk": FreshHyperparameters.sigreg_position_chunk,
            },
            "seed": FreshHyperparameters.seed,
            "tokenizer": {
                "path": FreshHyperparameters.tokenizer_path,
                "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            },
            "optimizer": {
                key: getattr(FreshHyperparameters, key)
                for key in (
                    "embed_lr", "head_lr", "tied_embed_lr", "matrix_lr", "scalar_lr",
                    "muon_momentum", "muon_backend_steps", "beta1", "beta2", "adam_eps",
                    "warmup_steps", "warmdown_iters", "iterations", "train_batch_tokens",
                    "train_seq_len",
                )
            },
            "grad_accum_steps": int(os.environ.get("GRAD_ACCUM_STEPS", "8")),
            "sigreg_compute_dtype": "float32",
        }
        if hasattr(FreshLeJEPAGPT, "experiment_metadata"):
            metadata["experiment"] = FreshLeJEPAGPT.experiment_metadata()
        metadata_path = Path("fresh_lejepa_metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        if "RUN_ID" in os.environ:
            run_dir = Path(__file__).resolve().parent / "ablation_results" / os.environ["RUN_ID"]
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2("final_model.pt", run_dir / "final_model.pt")
            if Path("final_model.int8.ptz").exists():
                shutil.copy2("final_model.int8.ptz", run_dir / "final_model.int8.ptz")
            shutil.copy2(metadata_path, run_dir / metadata_path.name)
            shutil.copy2(EXPERIMENT_SOURCE, run_dir / EXPERIMENT_SOURCE.name)
            checkpoint = {
                "step": FreshHyperparameters.iterations,
                "model": torch.load("final_model.pt", map_location="cpu", weights_only=True),
                "optimizers": [optimizer.state_dict() for optimizer in tracked_optimizers],
                "metadata": metadata,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(),
            }
            torch.save(checkpoint, run_dir / "pretraining_checkpoint.pt")


if __name__ == "__main__":
    main()

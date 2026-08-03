"""V4 ablation with LeWM-style dropout inside the temporal predictor only."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
from pretraining.fresh_lejepa.fresh_lejepa_train_v4 import (
    FreshLeJEPAV4,
    SIGREG_PROJECTION_CHUNK,
    SIGREG_POSITION_CHUNK,
    _install_configurable_accumulation,
)
import train_gpt as baseline


ARCHITECTURE = "fresh_lejepa_v4_predictor_dropout01"


class PredictorDropoutMixin:
    predictor_dropout = 0.1

    def _attention_with_dropout(
        self, attention: baseline.CausalSelfAttention, x: Tensor
    ) -> Tensor:
        batch, length, dim = x.shape
        q_dim = attention.num_heads * attention.head_dim
        kv_dim = attention.num_kv_heads * attention.head_dim
        q, k, value = attention.c_qkv(x).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.reshape(batch, length, attention.num_heads, attention.head_dim).transpose(1, 2)
        k = k.reshape(batch, length, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
        value = value.reshape(
            batch, length, attention.num_kv_heads, attention.head_dim
        ).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = attention.rotary(length, x.device, q.dtype)
        q = baseline.apply_rotary_emb(q, cos, sin)
        k = baseline.apply_rotary_emb(k, cos, sin)
        q = q * attention.q_gain.to(q.dtype)[None, :, None, None]
        output = F.scaled_dot_product_attention(
            q,
            k,
            value,
            is_causal=True,
            dropout_p=self.predictor_dropout if self.training else 0.0,
            enable_gqa=attention.num_kv_heads != attention.num_heads,
        )
        output = output.transpose(1, 2).contiguous().reshape(batch, length, dim)
        output = attention.proj(output)
        return F.dropout(output, self.predictor_dropout, self.training)

    def _block_with_dropout(
        self, block: baseline.Block, latent: Tensor, input_latent: Tensor
    ) -> Tensor:
        mix = block.resid_mix.to(latent.dtype)
        latent = mix[0][None, None] * latent + mix[1][None, None] * input_latent
        attention = self._attention_with_dropout(block.attn, block.attn_norm(latent))
        latent = latent + block.attn_scale.to(latent.dtype)[None, None] * attention
        hidden = torch.relu(block.mlp.fc(block.mlp_norm(latent))).square()
        hidden = F.dropout(hidden, self.predictor_dropout, self.training)
        mlp = block.mlp.proj(hidden)
        mlp = F.dropout(mlp, self.predictor_dropout, self.training)
        return latent + block.mlp_scale.to(latent.dtype)[None, None] * mlp

    def temporal_belief_from_token_latent(self, token_latent: Tensor) -> Tensor:
        predicted = token_latent
        skips: list[Tensor] = []
        for index in range(self.num_encoder_layers):
            predicted = self._block_with_dropout(
                self.blocks[index], predicted, token_latent
            )
            skips.append(predicted)
        for decoder_index in range(self.num_decoder_layers):
            if skips:
                predicted = predicted + self.skip_weights[decoder_index].to(
                    predicted.dtype
                )[None, None] * skips.pop()
            predicted = self._block_with_dropout(
                self.blocks[self.num_encoder_layers + decoder_index],
                predicted,
                token_latent,
            )
        return self.final_norm(predicted)

    def predict_from_token_latent(self, token_latent: Tensor) -> Tensor:
        return self.prediction_latent(
            self.temporal_belief_from_token_latent(token_latent)
        )


class FreshLeJEPAV4PredictorDropout(PredictorDropoutMixin, FreshLeJEPAV4):
    pass


def main() -> None:
    original_main = _install_configurable_accumulation(default_steps=8)
    FreshLeJEPAV4PredictorDropout.return_loss_components = True
    v1.FreshHyperparameters.sigreg_proj_chunk = SIGREG_PROJECTION_CHUNK
    v1.FreshHyperparameters.sigreg_position_chunk = SIGREG_POSITION_CHUNK
    v1.FreshLeJEPAGPT = FreshLeJEPAV4PredictorDropout
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    try:
        v1.main()
    finally:
        baseline.main = original_main


if __name__ == "__main__":
    main()

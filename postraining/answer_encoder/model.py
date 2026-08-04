"""Bidirectional text encoder and the unmodified LeJEPA representation loss.

The policy that eventually consumes this reward may be autoregressive.  The
reward encoder is not: it embeds the complete answer available at scoring
time and never observes a future answer state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


ANSWER_ENCODER_SCHEMA = "text_lejepa_answer_encoder/v2"
DEFAULT_REWARD_TEMPERATURE = 0.1
GPT2_EOT_ID = 50256
GPT2_MASK_ID = 50257
GPT2_PAD_ID = 50258
GPT2_ANSWER_ENCODER_VOCAB = 50259


def cosine_kernel_reward(
    cosine: Tensor,
    *,
    temperature: float = DEFAULT_REWARD_TEMPERATURE,
) -> Tensor:
    """Exponentially sharpen cosine while preserving its ordering."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("reward temperature must be finite and positive")
    return torch.exp((cosine.float().clamp(-1.0, 1.0) - 1.0) / temperature)


@dataclass(frozen=True)
class AnswerEncoderConfig:
    vocab_size: int = GPT2_ANSWER_ENCODER_VOCAB
    max_tokens: int = 128
    model_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    projection_hidden_dim: int = 1024
    projection_dim: int = 128
    reference_projector: bool = False
    projector_normalization: str = "batch"
    dropout: float = 0.1
    pad_token_id: int = GPT2_PAD_ID
    mask_token_id: int = GPT2_MASK_ID

    def __post_init__(self) -> None:
        if self.vocab_size <= max(self.pad_token_id, self.mask_token_id):
            raise ValueError("vocab_size must include the pad and mask tokens")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.model_dim < 1 or self.num_heads < 1 or self.model_dim % self.num_heads:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        if self.num_layers < 1 or self.mlp_ratio < 1:
            raise ValueError("num_layers and mlp_ratio must be positive")
        if self.projection_hidden_dim < 1 or self.projection_dim < 1:
            raise ValueError("projection dimensions must be positive")
        if self.projector_normalization not in {"batch", "layer"}:
            raise ValueError("projector_normalization must be 'batch' or 'layer'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LeJEPAObjectiveConfig:
    sigreg_weight: float = 0.02
    sigreg_knots: int = 17
    sigreg_slices: int = 256
    sigreg_t_max: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.sigreg_weight <= 1.0:
            raise ValueError("sigreg_weight must be in [0, 1]")
        if self.sigreg_knots < 2 or self.sigreg_slices < 1:
            raise ValueError("SIGReg needs at least two knots and one slice")
        if self.sigreg_t_max <= 0.0:
            raise ValueError("sigreg_t_max must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


class ProjectionHead(nn.Module):
    """LeJEPA projection MLP with batch- or example-local normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        exact_reference: bool,
        normalization: str,
    ):
        super().__init__()
        norm = nn.BatchNorm1d if normalization == "batch" else nn.LayerNorm
        if exact_reference:
            # Linear/ReLU topology and biases match torchvision.ops.MLP.
            layers = (
                nn.Linear(input_dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim, output_dim),
                nn.Dropout(0.0),
            )
        else:
            # Retain the v1 topology so its checkpoint remains loadable.
            layers = (
                nn.Linear(input_dim, hidden_dim, bias=False),
                norm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                norm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim, bias=False),
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.layers(embeddings)


class TextAnswerEncoder(nn.Module):
    """Variable-length bidirectional token encoder with learned CLS pooling."""

    def __init__(self, config: AnswerEncoderConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.model_dim,
            padding_idx=config.pad_token_id,
        )
        self.cls_embedding = nn.Parameter(torch.empty(config.model_dim))
        self.register_buffer(
            "position_frequencies",
            torch.exp(
                torch.arange(0, config.model_dim, 2, dtype=torch.float32)
                * (-math.log(10_000.0) / config.model_dim)
            ),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.model_dim * config.mlp_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.model_dim),
            enable_nested_tensor=False,
        )
        self.projector = ProjectionHead(
            config.model_dim,
            config.projection_hidden_dim,
            config.projection_dim,
            exact_reference=config.reference_projector,
            normalization=config.projector_normalization,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[self.config.pad_token_id].zero_()
        nn.init.normal_(self.cls_embedding, mean=0.0, std=0.02)

        # nn.TransformerEncoder deep-copies one layer, leaving every block
        # bit-identical unless it is explicitly reinitialized. ViT blocks are
        # independently initialized, and short LeJEPA ablations need the same.
        for layer in self.backbone.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            if layer.self_attn.in_proj_bias is not None:
                nn.init.zeros_(layer.self_attn.in_proj_bias)
            layer.self_attn.out_proj.reset_parameters()
            layer.linear1.reset_parameters()
            layer.linear2.reset_parameters()
            nn.init.ones_(layer.norm1.weight)
            nn.init.zeros_(layer.norm1.bias)
            nn.init.ones_(layer.norm2.weight)
            nn.init.zeros_(layer.norm2.bias)

    def position_embeddings(self, position_ids: Tensor, reference: Tensor) -> Tensor:
        """Create deterministic embeddings for arbitrary absolute positions."""
        if position_ids.ndim < 1:
            raise ValueError("position_ids must have at least one dimension")
        if bool((position_ids < 0).any()):
            raise ValueError("position_ids must be nonnegative")
        angles = (
            position_ids.to(device=reference.device, dtype=torch.float32).unsqueeze(-1)
            * self.position_frequencies
        )
        encoding = torch.zeros(
            *position_ids.shape,
            self.config.model_dim,
            device=reference.device,
            dtype=torch.float32,
        )
        encoding[..., 0::2] = angles.sin()
        encoding[..., 1::2] = angles.cos()[..., : encoding[..., 1::2].shape[-1]]
        return encoding.to(dtype=reference.dtype)

    def encode_hidden(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
        *,
        position_ids: Tensor | None = None,
        allow_empty_tokens: bool = False,
    ) -> Tensor:
        """Encode CLS and internal token states; explicit positions are zero-based."""
        if token_ids.ndim != 2 or attention_mask.shape != token_ids.shape:
            raise ValueError("token_ids and attention_mask must both have shape [batch, length]")
        if position_ids is not None and position_ids.shape != token_ids.shape:
            raise ValueError("position_ids must match token_ids")
        if not allow_empty_tokens and not bool(attention_mask.any(dim=1).all()):
            raise ValueError("every answer must contain at least one unpadded token")

        batch, length = token_ids.shape
        token_hidden = self.token_embedding(token_ids)
        cls = self.cls_embedding.view(1, 1, -1).expand(batch, 1, -1)
        hidden = torch.cat((cls, token_hidden), dim=1)
        if position_ids is None:
            position_ids = torch.arange(
                length, device=token_ids.device, dtype=torch.long
            ).expand(batch, -1)
        full_position_ids = torch.cat(
            (
                torch.zeros(batch, 1, device=token_ids.device, dtype=torch.long),
                position_ids.to(device=token_ids.device, dtype=torch.long) + 1,
            ),
            dim=1,
        )
        hidden = hidden + self.position_embeddings(full_position_ids, hidden)
        cls_mask = torch.ones(batch, 1, dtype=torch.bool, device=attention_mask.device)
        full_mask = torch.cat((cls_mask, attention_mask.bool()), dim=1)
        hidden = self.backbone(hidden, src_key_padding_mask=~full_mask)
        return hidden

    def encode_backbone(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.encode_hidden(token_ids, attention_mask)[:, 0]

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        backbone = self.encode_backbone(token_ids, attention_mask)
        projection = self.projector(backbone)
        return backbone, projection

    def encode(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
        *,
        space: str = "backbone",
        normalize: bool = True,
    ) -> Tensor:
        backbone, projection = self(token_ids, attention_mask)
        if space == "projection":
            embeddings = projection
        elif space == "backbone":
            embeddings = backbone
        else:
            raise ValueError(f"unknown embedding space {space!r}")
        return F.normalize(embeddings.float(), dim=-1) if normalize else embeddings


class SIGReg(nn.Module):
    """Sketched isotropic-Gaussian regularization from the LeJEPA recipe."""

    def __init__(self, config: LeJEPAObjectiveConfig):
        super().__init__()
        t = torch.linspace(0.0, config.sigreg_t_max, config.sigreg_knots)
        dt = config.sigreg_t_max / (config.sigreg_knots - 1)
        weights = torch.full((config.sigreg_knots,), 2.0 * dt)
        weights[[0, -1]] = dt
        gaussian_cf = torch.exp(-0.5 * t.square())
        self.slices = config.sigreg_slices
        self.register_buffer("t", t)
        self.register_buffer("gaussian_cf", gaussian_cf)
        self.register_buffer("weights", weights * gaussian_cf)

    def forward(self, views: Tensor) -> Tensor:
        """Regularize each view's batch distribution; views is [V, B, D]."""
        if views.ndim != 3 or views.size(1) < 2:
            raise ValueError("SIGReg expects [views, batch>=2, dimensions]")
        values = views.float()
        directions = torch.randn(
            values.size(-1), self.slices, device=values.device, dtype=values.dtype
        )
        directions = F.normalize(directions, dim=0)
        projected = values @ directions
        angles = projected.unsqueeze(-1) * self.t
        real_error = angles.cos().mean(dim=1) - self.gaussian_cf
        imaginary_error = angles.sin().mean(dim=1)
        error = real_error.square() + imaginary_error.square()
        statistic = (error @ self.weights) * values.size(1)
        return statistic.mean()


class LeJEPAObjective(nn.Module):
    """View invariance plus SIGReg, without labels, negatives, or stop-grad."""

    def __init__(self, config: LeJEPAObjectiveConfig):
        super().__init__()
        self.config = config
        self.sigreg = SIGReg(config)

    def forward(self, embeddings: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if embeddings.ndim != 3 or embeddings.size(0) < 2:
            raise ValueError("embeddings must have shape [views>=2, batch, dimensions]")
        values = embeddings.float()
        center = values.mean(dim=0, keepdim=True)
        invariance = (values - center).square().mean()
        sigreg = self.sigreg(values)
        weight = self.config.sigreg_weight
        total = (1.0 - weight) * invariance + weight * sigreg
        normalized = F.normalize(values, dim=-1)
        normalized_center = F.normalize(center, dim=-1)
        view_center_cosine = (normalized * normalized_center).sum(dim=-1).mean()
        diagnostics = {
            "invariance_loss": invariance.detach(),
            "sigreg_loss": sigreg.detach(),
            "view_center_cosine": view_center_cosine.detach(),
            "objective_norm": values.norm(dim=-1).mean().detach(),
            "objective_dimension_std": values.std(dim=(0, 1), correction=0).mean().detach(),
        }
        return total, diagnostics


class GlobalLatentPredictor(nn.Module):
    """Predict one complete-answer CLS from a variable-cardinality context."""

    def __init__(
        self,
        dimension: int,
        hidden_dimension: int,
        *,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        if dimension < 1 or hidden_dimension < 1 or num_layers < 1:
            raise ValueError("predictor dimensions must be positive")
        if num_heads < 1 or dimension % num_heads:
            raise ValueError("predictor heads must divide the latent dimension")
        layer = nn.TransformerDecoderLayer(
            d_model=dimension,
            nhead=num_heads,
            dim_feedforward=hidden_dimension,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(dimension),
        )
        self.global_query = nn.Parameter(torch.empty(dimension))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.global_query, mean=0.0, std=0.02)
        # TransformerDecoder clones its prototype layer, so initialize each
        # block independently just as the encoder does.
        for layer in self.decoder.layers:
            for attention in (layer.self_attn, layer.multihead_attn):
                nn.init.xavier_uniform_(attention.in_proj_weight)
                if attention.in_proj_bias is not None:
                    nn.init.zeros_(attention.in_proj_bias)
                attention.out_proj.reset_parameters()
            layer.linear1.reset_parameters()
            layer.linear2.reset_parameters()
            for norm in (layer.norm1, layer.norm2, layer.norm3):
                nn.init.ones_(norm.weight)
                nn.init.zeros_(norm.bias)

    def forward(
        self,
        context_embeddings: Tensor,
        context_attention_mask: Tensor,
    ) -> Tensor:
        """Return one global prediction per sample."""
        if context_embeddings.ndim != 3:
            raise ValueError("context embeddings must have shape [batch, states, dimensions]")
        if context_attention_mask.shape != context_embeddings.shape[:2]:
            raise ValueError("context attention must identify every context state")
        if not bool(context_attention_mask[:, 0].all()):
            raise ValueError("every context must include its CLS state")

        batch = context_embeddings.size(0)
        query = self.global_query.view(1, 1, -1).expand(batch, 1, -1)
        predictions = self.decoder(
            query,
            context_embeddings,
            memory_key_padding_mask=~context_attention_mask.bool(),
        )
        return predictions[:, 0]


class GlobalLatentObjective(nn.Module):
    """LeJEPA center MSE and SIGReg over complete-answer CLS embeddings.

    Both complete targets and compact-context predictions remain attached. This
    preserves LeJEPA's symmetric, no-teacher/no-stop-gradient optimization while
    making the deployed CLS the only representation optimized by the objective.
    """

    def __init__(self, config: LeJEPAObjectiveConfig):
        super().__init__()
        self.config = config
        self.sigreg = SIGReg(config)

    def forward(
        self,
        predicted_global: Tensor,
        target_global: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if predicted_global.ndim != 2 or target_global.shape != predicted_global.shape:
            raise ValueError("predicted and target globals must have shape [batch, dimensions]")
        batch = predicted_global.size(0)
        if batch < 2:
            raise ValueError("global latent prediction requires batch size at least two")

        predicted_global = predicted_global.float()
        target_global = target_global.float()

        global_mse = (predicted_global - target_global).square().mean()
        # For two views, LeJEPA's view-to-center MSE is exactly one quarter of
        # the direct pairwise MSE. Preserve that scale so lambda retains its
        # reference meaning.
        invariance = 0.25 * global_mse

        global_views = torch.stack((predicted_global, target_global))
        sigreg = self.sigreg(global_views)

        weight = self.config.sigreg_weight
        total = (1.0 - weight) * invariance + weight * sigreg
        global_cosine = F.cosine_similarity(predicted_global, target_global).mean()
        normalized_global_views = F.normalize(global_views, dim=-1)
        normalized_global_center = F.normalize(
            global_views.mean(dim=0, keepdim=True), dim=-1
        )
        global_center_cosine = (
            normalized_global_views * normalized_global_center
        ).sum(dim=-1).mean()
        diagnostics = {
            "invariance_loss": invariance.detach(),
            "global_prediction_mse": global_mse.detach(),
            "sigreg_loss": sigreg.detach(),
            "global_prediction_cosine": global_cosine.detach(),
            "view_center_cosine": global_center_cosine.detach(),
            "objective_norm": global_views.norm(dim=-1).mean().detach(),
            "objective_dimension": torch.tensor(
                predicted_global.size(-1),
                device=predicted_global.device,
                dtype=predicted_global.dtype,
            ).detach(),
        }
        return total, diagnostics


class AnswerSimilarityReward:
    """Frozen target embedding and temperature-scaled cosine-kernel reward."""

    def __init__(
        self,
        target_embedding: Tensor,
        *,
        temperature: float = DEFAULT_REWARD_TEMPERATURE,
    ):
        if target_embedding.ndim == 1:
            target_embedding = target_embedding.unsqueeze(0)
        if target_embedding.ndim != 2 or target_embedding.size(0) != 1:
            raise ValueError("target_embedding must describe exactly one target")
        self.target_embedding = F.normalize(target_embedding.float(), dim=-1).detach()
        # Validate at construction instead of waiting until the first rollout.
        cosine_kernel_reward(torch.ones(1), temperature=temperature)
        self.temperature = temperature

    def cosine(self, answer_embeddings: Tensor) -> Tensor:
        answers = F.normalize(answer_embeddings.float(), dim=-1)
        target = self.target_embedding.to(device=answers.device)
        return answers @ target.T

    def __call__(self, answer_embeddings: Tensor) -> Tensor:
        """Return exp((cosine - 1) / temperature), with an exact match at one."""
        cosine = self.cosine(answer_embeddings).squeeze(-1)
        return cosine_kernel_reward(cosine, temperature=self.temperature)

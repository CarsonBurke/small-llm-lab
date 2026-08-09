"""Bolmo-style byteification for the repository's KDA language model.

This module follows Minixhofer et al., *Bolmo: Byteifying the Next
Generation of Language Models* (arXiv:2512.15586v2): one causal mLSTM local
encoder, a one-byte-lookahead non-causal boundary predictor, last-state
pooling into the retained global model, and four causal mLSTM local-decoder
blocks with a fused byte/boundary vocabulary.

The source model is needed only during Stage 1.  Its global blocks become the
student's global model; its input/output tables are held by :class:`BolmoTeacher`
and are deliberately absent from inference checkpoints.  The retained suffix
embedding is an independently trainable copy of the source input table, as in
the released Bolmo implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pretraining.nanogpt_mini.nanogpt_mini_kda_model import KDAGPT, RMSNorm

try:
    from xlstm.xlstm_large.model import (
        mLSTMBackendConfig,
        mLSTMLayer,
        mLSTMLayerConfig,
    )
except ImportError as exc:  # pragma: no cover - exercised by deployment guard
    raise ImportError(
        "Bolmo requires xlstm==2.0.5 (and its mlstm_kernels dependency)"
    ) from exc


BYTE_VOCAB_SIZE = 256
BYTE_EOT_ID = 256
SOURCE_EOT_ID = 0
DEFAULT_TEACHER_POSITIONS_PER_CHUNK = 4096


@dataclass(frozen=True)
class BolmoArchitecture:
    """Architecture choices fixed by the paper, scaled to the 512-wide trunk."""

    model_dim: int = 512
    local_heads: int = 16
    local_ffn_hidden: int = 704  # 2816 / 2048 times the global width
    encoder_layers: int = 1
    decoder_layers: int = 4
    boundary_lookahead: int = 1
    gate_soft_cap: float = 15.0
    input_gate_bias: float = -10.0
    logit_soft_cap: float = 15.0
    backend: str = "triton"
    chunk_size: int = 128
    autocast_kernel_dtype: str = "float32"
    num_special_tokens: int = 5

    def __post_init__(self) -> None:
        if self.encoder_layers != 1 or self.decoder_layers != 4:
            raise ValueError("paper-aligned Bolmo uses one encoder and four decoder layers")
        if self.boundary_lookahead != 1:
            raise ValueError("paper-aligned Bolmo uses exactly one byte of lookahead")
        if self.model_dim % self.local_heads:
            raise ValueError("model_dim must be divisible by local_heads")
        if self.backend not in {"native", "triton"}:
            raise ValueError("backend must be 'native' or 'triton'")
        if self.autocast_kernel_dtype not in {"float32", "bfloat16"}:
            raise ValueError(
                "autocast_kernel_dtype must be 'float32' or 'bfloat16'"
            )
        if self.num_special_tokens < 1:
            raise ValueError("at least the EOT/BOS special token is required")

    @property
    def atomic_vocab_size(self) -> int:
        return BYTE_VOCAB_SIZE + self.num_special_tokens

    @property
    def byte_pad_id(self) -> int:
        return self.atomic_vocab_size

    @property
    def fused_vocab_size(self) -> int:
        return 2 * self.atomic_vocab_size


@dataclass
class BolmoBatch:
    """A padded batch in byte space with its oracle source-token alignment."""

    source_ids: Tensor
    source_valid_mask: Tensor
    byte_ids: Tensor
    expanded_ids: Tensor
    oracle_boundaries: Tensor
    valid_mask: Tensor
    score_mask: Tensor
    patch_lens: Tensor

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "BolmoBatch":
        return BolmoBatch(
            source_ids=self.source_ids.to(device, non_blocking=non_blocking),
            source_valid_mask=self.source_valid_mask.to(device, non_blocking=non_blocking),
            byte_ids=self.byte_ids.to(device, non_blocking=non_blocking),
            expanded_ids=self.expanded_ids.to(device, non_blocking=non_blocking),
            oracle_boundaries=self.oracle_boundaries.to(device, non_blocking=non_blocking),
            valid_mask=self.valid_mask.to(device, non_blocking=non_blocking),
            score_mask=self.score_mask.to(device, non_blocking=non_blocking),
            patch_lens=self.patch_lens.to(device, non_blocking=non_blocking),
        )

    def validate(self, source_vocab_size: int, atomic_vocab_size: int) -> None:
        shapes = {
            tuple(self.byte_ids.shape),
            tuple(self.expanded_ids.shape),
            tuple(self.oracle_boundaries.shape),
            tuple(self.valid_mask.shape),
            tuple(self.score_mask.shape),
        }
        if len(shapes) != 1 or self.byte_ids.ndim != 2:
            raise ValueError("byte tensors must be rank-2 and share a shape")
        if self.source_ids.ndim != 2 or self.patch_lens.shape != self.source_ids.shape:
            raise ValueError("source_ids and patch_lens must be aligned rank-2 tensors")
        if self.source_valid_mask.shape != self.source_ids.shape:
            raise ValueError("source_valid_mask and source_ids must share a shape")
        if self.source_ids.shape[0] != self.byte_ids.shape[0]:
            raise ValueError("source and byte batch sizes differ")
        if not torch.all(self.valid_mask[:, 0]):
            raise ValueError("every row must begin with a valid BOS/EOT patch")
        if not torch.all(self.oracle_boundaries[:, 0]):
            raise ValueError("the prepended BOS/EOT byte must be a boundary")
        if torch.any(self.score_mask & ~self.valid_mask):
            raise ValueError("score mask includes invalid byte positions")
        if torch.any(self.score_mask[:, 0]):
            raise ValueError("the prepended BOS/EOT byte cannot be scored")
        if torch.any(self.byte_ids[self.valid_mask] >= atomic_vocab_size):
            raise ValueError("a valid byte id is outside the atomic vocabulary")
        if torch.any(self.expanded_ids[self.valid_mask] > source_vocab_size):
            raise ValueError("an expanded id is outside source_vocab_size + null")
        if not torch.equal(self.patch_lens.sum(1), self.valid_mask.sum(1)):
            raise ValueError("patch lengths do not cover exactly the valid byte prefix")
        if not torch.equal(self.patch_lens > 0, self.source_valid_mask):
            raise ValueError("source validity differs from nonempty patch lengths")
        expected_boundaries = torch.zeros_like(self.oracle_boundaries)
        for row in range(self.patch_lens.shape[0]):
            lengths = self.patch_lens[row][self.source_valid_mask[row]]
            ends = lengths.cumsum(0) - 1
            expected_boundaries[row, ends] = True
            valid_count = int(self.valid_mask[row].sum())
            eot_positions = torch.nonzero(
                self.byte_ids[row, 1:valid_count] == BYTE_EOT_ID,
                as_tuple=False,
            ).flatten()
            expected_boundaries[row, eot_positions] = False
            expected_boundaries[row, 0] = True
        if not torch.equal(self.oracle_boundaries, expected_boundaries):
            raise ValueError("oracle boundaries differ from EOS-aware source patching")


@dataclass
class BolmoLoss:
    total: Tensor
    ce: Tensor
    boundary: Tensor
    encoder_stitch: Tensor | None = None
    encoder_stitch_cosine: Tensor | None = None
    decoder_distill: Tensor | None = None
    boundary_accuracy: Tensor | None = None
    boundary_precision: Tensor | None = None
    boundary_recall: Tensor | None = None
    bytes_per_patch: Tensor | None = None
    boundary_correct: Tensor | None = None
    boundary_true_positive: Tensor | None = None
    boundary_false_positive: Tensor | None = None
    boundary_false_negative: Tensor | None = None
    boundary_positions: Tensor | None = None
    valid_bytes: Tensor | None = None
    predicted_patches: Tensor | None = None


@dataclass
class BolmoValidationStatistics:
    """Additive validation statistics for paper-style byte evaluation.

    Bolmo trains its fused byte/boundary head with joint cross entropy.  At
    evaluation time the released implementation marginalizes the output
    boundary bit, because both halves of the fused vocabulary represent the
    same next byte.  Keeping the joint NLL alongside the marginalized byte NLL
    makes the training objective observable without misreporting it as BPB.
    """

    byte_nll: Tensor
    joint_nll: Tensor
    scored_bytes: Tensor
    valid_atoms: Tensor
    predicted_patches: Tensor
    boundary_correct: Tensor
    boundary_positions: Tensor
    boundary_true_positive: Tensor
    boundary_false_positive: Tensor
    boundary_false_negative: Tensor


def _log1mexp(log_p: Tensor) -> Tensor:
    """Stable log(1-exp(log_p)) for log_p <= 0."""

    split = -math.log(2.0)
    return torch.where(
        log_p < split,
        torch.log1p(-torch.exp(log_p)),
        torch.log(-torch.expm1(log_p)),
    )


def bernoulli_kl_from_log_probs(
    teacher_log_p: Tensor,
    student_log_p: Tensor,
    *,
    temperature: float = 5.0,
    epsilon: float = 1e-6,
) -> Tensor:
    """The released Bolmo Stage-1 patch-likelihood objective.

    Dividing log probabilities by ``temperature`` implements the paper's
    probability power transform.  Returning KL rather than cross entropy only
    subtracts the teacher entropy and therefore has identical student
    gradients; this matches ``div_fn=kl`` in AI2's launch recipe.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # This is the released implementation's exact stabilization: subtracting
    # epsilon keeps both Bernoulli outcomes finite without changing the power
    # transform into a normalized softmax temperature.
    log_q = teacher_log_p.float() / temperature - epsilon
    log_p = student_log_p.float() / temperature - epsilon
    q = torch.exp(log_q)
    one_minus_q = -torch.expm1(log_q)
    teacher_entropy_term = q * log_q + one_minus_q * _log1mexp(log_q)
    cross_entropy_term = q * log_p + one_minus_q * _log1mexp(log_p)
    return teacher_entropy_term - cross_entropy_term


def _backend_config(config: BolmoArchitecture) -> mLSTMBackendConfig:
    if config.backend == "triton":
        return mLSTMBackendConfig(
            # The xLSTM project's TFLA-derived kernel is the maintained fast
            # training path. Unlike the older limit-chunk implementation it
            # is compatible with Triton 3.6's strict scalar type promotion.
            # Both implement the same stabilized mLSTM recurrence.
            chunkwise_kernel="chunkwise--triton_xl_chunk",
            sequence_kernel="native_sequence__triton",
            step_kernel="triton",
            mode="train",
            # xLSTM's layer wrapper unconditionally unpacks the last state.
            return_last_states=True,
            autocast_kernel_dtype=config.autocast_kernel_dtype,
            chunk_size=config.chunk_size,
        )
    return mLSTMBackendConfig(
        chunkwise_kernel="chunkwise--native_autograd",
        sequence_kernel="native_sequence__native",
        step_kernel="native",
        mode="train",
        return_last_states=True,
        autocast_kernel_dtype=config.autocast_kernel_dtype,
        chunk_size=config.chunk_size,
    )


class SwiGLU(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int):
        super().__init__()
        self.gate_up = nn.Linear(model_dim, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class BolmoLocalBlock(nn.Module):
    """Pre-norm mLSTM + SwiGLU block used by both local models."""

    def __init__(self, config: BolmoArchitecture):
        super().__init__()
        self.mlstm_norm = RMSNorm(config.model_dim)
        self.mlstm = mLSTMLayer(mLSTMLayerConfig(
            embedding_dim=config.model_dim,
            num_heads=config.local_heads,
            qk_dim_factor=0.5,
            v_dim_factor=1.0,
            gate_soft_cap=config.gate_soft_cap,
            mlstm_backend=_backend_config(config),
            weight_mode="fused",
        ))
        self.ffn_norm = RMSNorm(config.model_dim)
        self.ffn = SwiGLU(config.model_dim, config.local_ffn_hidden)
        with torch.no_grad():
            # Fused layout is [input gates for H heads, forget gates for H].
            self.mlstm.ifgate_preact.bias[: config.local_heads].fill_(
                config.input_gate_bias
            )

    def forward(self, x: Tensor) -> Tensor:
        recurrent = _stateless_mlstm_forward(self.mlstm, self.mlstm_norm(x))
        x = x + recurrent
        return x + self.ffn(self.ffn_norm(x))


def _stateless_mlstm_forward(layer: mLSTMLayer, x: Tensor) -> Tensor:
    """xLSTM's fused training forward without unused final-state copies.

    The upstream :class:`mLSTMLayer` always unpacks a ``(hidden, state)`` pair,
    which forces the XL kernel to materialize contiguous copies of its final
    C/N/M states. Bolmo never carries state between training chunks. This is
    the fused-weight branch of xLSTM 2.0.5's forward with the backend's
    supported ``return_last_states=False`` override; all hidden-state math is
    otherwise identical.
    """

    if layer.config.weight_mode != "fused":
        raise ValueError("Bolmo local layers require fused xLSTM projections")
    batch, sequence, _ = x.shape
    qkv_o = layer.qkv_opreact(x)
    q, k, v, o_preact = torch.tensor_split(
        qkv_o,
        (
            layer.qk_dim,
            2 * layer.qk_dim,
            2 * layer.qk_dim + layer.v_dim,
        ),
        dim=-1,
    )
    gate_cap = layer.config.gate_soft_cap
    if_preact = gate_cap * torch.tanh(layer.ifgate_preact(x) / gate_cap)
    i_preact, f_preact = torch.tensor_split(
        if_preact, (layer.config.num_heads,), dim=-1
    )
    heads = layer.config.num_heads
    q = q.reshape(batch, sequence, heads, -1).transpose(1, 2)
    k = k.reshape(batch, sequence, heads, -1).transpose(1, 2)
    v = v.reshape(batch, sequence, heads, -1).transpose(1, 2)
    hidden = layer.mlstm_backend(
        q=q,
        k=k,
        v=v,
        i=i_preact.transpose(1, 2),
        f=f_preact.transpose(1, 2),
        return_last_states=False,
    )
    if isinstance(hidden, tuple):
        raise AssertionError("stateless xLSTM backend unexpectedly returned state")
    hidden = hidden.transpose(1, 2)
    hidden = layer.multihead_norm(hidden).reshape(batch, sequence, -1)
    hidden = layer.ogate_act_fn(o_preact) * hidden
    return layer.out_proj(hidden)


class NonCausalBoundaryPredictor(nn.Module):
    """Cosine-distance predictor with exactly one byte of future context."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        with torch.no_grad():
            identity = torch.eye(model_dim)
            self.q_proj.weight.copy_(identity)
            self.k_proj.weight.copy_(identity)

    def forward(self, hidden: Tensor, valid_mask: Tensor) -> Tensor:
        if hidden.shape[:2] != valid_mask.shape:
            raise ValueError("hidden and valid_mask sequence shapes differ")
        # Keep the wide projections and normalization in the autocast dtype,
        # as the reference does; only the scalar cosine is promoted for the
        # stable log-probability transform.
        q = F.normalize(self.q_proj(hidden[:, :-1]), dim=-1)
        k = F.normalize(self.k_proj(hidden[:, 1:]), dim=-1)
        cosine = (q * k).sum(-1).float()
        log_p = torch.log1p(-cosine.clamp(max=1.0 - 1e-3)) - math.log(2.0)
        log_p = F.pad(log_p, (0, 1), value=-100_000.0)
        paired_valid = valid_mask & F.pad(valid_mask[:, 1:], (0, 1), value=False)
        log_p = torch.where(paired_valid, log_p, torch.full_like(log_p, -100_000.0))
        # Each row starts with a synthetic BOS/EOT patch, exactly as AI2's
        # implementation forces its first byte to be a boundary.
        log_p[:, 0] = 0.0
        return log_p


class BolmoLocalEncoder(nn.Module):
    def __init__(
        self,
        source_embedding: Tensor,
        config: BolmoArchitecture,
    ):
        super().__init__()
        source_vocab_size, source_dim = source_embedding.shape
        if source_dim != config.model_dim:
            raise ValueError("source embedding width differs from Bolmo width")
        self.source_vocab_size = source_vocab_size
        self.byte_embedding = nn.Embedding(
            config.atomic_vocab_size + 1,
            config.model_dim,
            padding_idx=config.byte_pad_id,
            dtype=source_embedding.dtype,
        )
        # Last row is a causal "no suffix" value.  It remains exactly zero.
        self.expanded_embedding = nn.Embedding(
            source_vocab_size + 1,
            config.model_dim,
            padding_idx=source_vocab_size,
            dtype=source_embedding.dtype,
        )
        with torch.no_grad():
            source_float = source_embedding.float()
            source_mean = source_float.mean(0)
            source_std = source_float.std(0, unbiased=False).clamp_min(1e-6)
            sampled = torch.randn_like(self.byte_embedding.weight)
            self.byte_embedding.weight.copy_(
                sampled * source_std[None, :] + source_mean[None, :]
            )
            self.byte_embedding.weight[config.byte_pad_id].zero_()
            self.expanded_embedding.weight[:-1].copy_(source_embedding)
            self.expanded_embedding.weight[-1].zero_()
        self.blocks = nn.ModuleList(
            BolmoLocalBlock(config) for _ in range(config.encoder_layers)
        )
        self.final_norm = RMSNorm(config.model_dim)
        self.out_projection = nn.Linear(config.model_dim, config.model_dim, bias=True)
        self.boundary_predictor = NonCausalBoundaryPredictor(config.model_dim)

    def forward(
        self,
        byte_ids: Tensor,
        expanded_ids: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden = self.byte_embedding(byte_ids) + self.expanded_embedding(expanded_ids)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        boundary_log_probs = self.boundary_predictor(hidden, valid_mask)
        return hidden, boundary_log_probs

    def pool(
        self,
        hidden: Tensor,
        boundaries: Tensor,
        *,
        max_patches: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Select the last byte of each patch and left-pack valid patches."""

        batch, length, width = hidden.shape
        if boundaries.shape != (batch, length):
            raise ValueError("boundary mask shape differs from hidden states")
        patch_counts = boundaries.sum(1)
        # Training has a fixed source-token-axis upper bound. Passing it avoids
        # a GPU->CPU `.item()` synchronization on every microbatch, and matches
        # the reference implementation's fixed-size packed patch tensor.
        if max_patches is None:
            max_patches = length
        if not 0 < max_patches <= length:
            raise ValueError("max_patches must be within the byte sequence")
        indices = torch.arange(length, device=hidden.device)[None, :]
        sort_keys = indices + (~boundaries).long() * length
        selected = torch.argsort(sort_keys, dim=1)[:, :max_patches]
        pooled = torch.gather(
            hidden,
            1,
            selected[..., None].expand(-1, -1, width),
        )
        patch_valid = (
            torch.arange(max_patches, device=hidden.device)[None, :]
            < patch_counts.clamp(max=max_patches)[:, None]
        )
        pooled = self.out_projection(pooled) * patch_valid[..., None]
        return pooled, patch_valid


class BolmoLocalDecoder(nn.Module):
    def __init__(self, config: BolmoArchitecture):
        super().__init__()
        self.byte_projection = nn.Linear(config.model_dim, config.model_dim, bias=True)
        self.patch_norm = RMSNorm(config.model_dim)
        self.blocks = nn.ModuleList(
            BolmoLocalBlock(config) for _ in range(config.decoder_layers)
        )
        # Bolmo keeps the source trunk's final state unnormalized through
        # depooling and applies the source-style final norm at the byte head.
        # Its weights are initialized from the source model in BolmoModel.
        self.head_norm = RMSNorm(config.model_dim)
        self.lm_head = nn.Linear(
            config.model_dim, config.fused_vocab_size, bias=False
        )
        self.logit_soft_cap = config.logit_soft_cap

    def project_logits(self, hidden: Tensor) -> Tensor:
        """Normalize, project, and rational-softcap byte logits."""

        logits = self.lm_head(self.head_norm(hidden)).float()
        cap = self.logit_soft_cap
        return cap * logits * torch.rsqrt(logits.square() + cap * cap)

    def prepare_hidden(
        self,
        byte_hidden: Tensor,
        patch_hidden: Tensor,
        boundaries: Tensor,
    ) -> Tensor:
        """Depool patch states and combine them with projected byte states."""

        patch_ids = (boundaries.long().cumsum(1) - 1).clamp(
            min=0, max=patch_hidden.shape[1] - 1
        )
        expanded_patch = torch.gather(
            self.patch_norm(patch_hidden),
            1,
            patch_ids[..., None].expand(-1, -1, patch_hidden.shape[-1]),
        )
        return self.byte_projection(byte_hidden) + expanded_patch

    def forward(
        self,
        byte_hidden: Tensor,
        patch_hidden: Tensor,
        boundaries: Tensor,
    ) -> Tensor:
        hidden = self.prepare_hidden(byte_hidden, patch_hidden, boundaries)
        for block in self.blocks:
            hidden = block(hidden)
        return self.project_logits(hidden)


class BolmoTeacher:
    """Stage-1-only source-model pieces sharing the student's global blocks."""

    def __init__(self, source: KDAGPT, global_blocks: nn.Sequential):
        self.embedding = source.embed
        self.input_norm = source.norm1
        self.output_head = source.proj
        self.global_blocks = global_blocks
        self.output_norm = source.norm2
        for module in (
            self.embedding,
            self.input_norm,
            self.output_norm,
            self.output_head,
        ):
            module.requires_grad_(False)

    def to(self, device: torch.device | str) -> "BolmoTeacher":
        self.embedding.to(device)
        self.input_norm.to(device)
        self.output_norm.to(device)
        self.output_head.to(device)
        return self

    @torch.no_grad()
    def forward(self, source_ids: Tensor, stitch_depth: int) -> tuple[Tensor, Tensor]:
        hidden = self.input_norm(self.embedding(source_ids))
        stitched = None
        for index, block in enumerate(self.global_blocks):
            hidden = block(hidden)
            if index + 1 == stitch_depth:
                stitched = hidden
        if stitched is None:
            raise ValueError("stitch_depth exceeds global model depth")
        return stitched, hidden

    @torch.no_grad()
    def selected_log_probs(
        self,
        final_hidden: Tensor,
        targets: Tensor,
        *,
        positions_per_chunk: int = DEFAULT_TEACHER_POSITIONS_PER_CHUNK,
    ) -> Tensor:
        """Compute gold-token log probabilities without full-sequence logits."""

        if positions_per_chunk <= 0:
            raise ValueError("positions_per_chunk must be positive")
        normalized = self.output_norm(final_hidden).flatten(0, 1)
        flat_targets = targets.flatten()
        selected_chunks = []
        for start in range(0, normalized.shape[0], positions_per_chunk):
            stop = min(start + positions_per_chunk, normalized.shape[0])
            logits = self.output_head(normalized[start:stop]).float()
            logits = 15.0 * logits * torch.rsqrt(logits.square() + 15.0**2)
            selected_chunks.append(
                -F.cross_entropy(
                    logits,
                    flat_targets[start:stop],
                    reduction="none",
                )
            )
        return torch.cat(selected_chunks).view_as(targets)


class BolmoModel(nn.Module):
    # Arms whose per-row patch count the placement-blind floor can be matched
    # to. They disagree by about 5% on the canonical span, so the choice is
    # part of the diagnostic rather than an implementation detail.
    UNIFORM_PATCH_COUNTS = ("oracle", "predicted")

    def __init__(
        self,
        global_blocks: nn.Sequential,
        global_norm: nn.Module,
        source_embedding: Tensor,
        source_model_config: Mapping[str, Any],
        architecture: BolmoArchitecture,
    ):
        super().__init__()
        self.global_blocks = global_blocks
        self.local_encoder = BolmoLocalEncoder(source_embedding, architecture)
        self.local_decoder = BolmoLocalDecoder(architecture)
        self.local_decoder.head_norm.load_state_dict(global_norm.state_dict(), strict=True)
        self.source_model_config = dict(source_model_config)
        self.architecture_config = asdict(architecture)
        self.atomic_vocab_size = architecture.atomic_vocab_size
        self.byte_pad_id = architecture.byte_pad_id
        self.fused_vocab_size = architecture.fused_vocab_size
        self.stitch_depth = min(4, len(global_blocks))

    @classmethod
    def from_source_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        architecture: BolmoArchitecture | None = None,
        map_location: str | torch.device = "cpu",
    ) -> tuple["BolmoModel", BolmoTeacher, dict[str, Any]]:
        payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
        source_config = dict(payload["model_config"])
        source = KDAGPT(**source_config)
        source.load_state_dict(payload["model"], strict=True)
        architecture = architecture or BolmoArchitecture(
            model_dim=int(source_config["model_dim"])
        )
        if architecture.model_dim != int(source_config["model_dim"]):
            raise ValueError("Bolmo width must match the source global model")
        model = cls(
            global_blocks=source.blocks,
            global_norm=source.norm2,
            source_embedding=source.embed.weight.detach(),
            source_model_config=source_config,
            architecture=architecture,
        )
        teacher = BolmoTeacher(source, model.global_blocks)
        return model, teacher, payload

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BolmoModel":
        source_config = dict(config["source_model_config"])
        source = KDAGPT(**source_config)
        architecture = BolmoArchitecture(**dict(config["architecture"]))
        return cls(
            global_blocks=source.blocks,
            global_norm=source.norm2,
            source_embedding=source.embed.weight.detach(),
            source_model_config=source_config,
            architecture=architecture,
        )

    def export_config(self) -> dict[str, Any]:
        return {
            "source_model_config": self.source_model_config,
            "architecture": self.architecture_config,
            "atomic_vocab_size": self.atomic_vocab_size,
            "byte_pad_id": self.byte_pad_id,
            "fused_vocab_size": self.fused_vocab_size,
        }

    def freeze_global(self, frozen: bool = True) -> None:
        self.global_blocks.requires_grad_(not frozen)

    @torch.no_grad()
    def calibrate_encoder_output(
        self,
        batch: BolmoBatch,
        teacher: BolmoTeacher,
    ) -> None:
        """Match initial oracle patch mean/std to source input representations.

        AI2's released implementation performs this affine calibration before
        Stage 1.  Using a real byteified batch is more representative than its
        random-token estimate and does not alter the paper's objective.
        """

        byte_hidden, _ = self.local_encoder(
            batch.byte_ids, batch.expanded_ids, batch.valid_mask
        )
        pooled, patch_valid = self.local_encoder.pool(
            byte_hidden,
            batch.oracle_boundaries,
            max_patches=batch.source_ids.shape[1],
        )
        source_ids = batch.source_ids.clamp(
            max=self.local_encoder.source_vocab_size - 1
        )
        target = teacher.input_norm(teacher.embedding(source_ids))
        teacher_indices, teacher_patch_valid = self._teacher_patch_alignment(batch)
        target = torch.gather(
            target,
            1,
            teacher_indices[..., None].expand(-1, -1, target.shape[-1]),
        )
        mask = patch_valid & teacher_patch_valid[:, : patch_valid.shape[1]]
        current_values = pooled[mask].float()
        target_values = target[:, : patch_valid.shape[1]][mask].float()
        current_mean = current_values.mean(0)
        current_std = current_values.std(0, unbiased=False).clamp_min(1e-6)
        target_mean = target_values.mean(0)
        target_std = target_values.std(0, unbiased=False).clamp_min(1e-6)
        scale = target_std / current_std
        projection = self.local_encoder.out_projection
        projection.weight.mul_(scale[:, None])
        projection.bias.copy_(
            target_mean + scale * (projection.bias.float() - current_mean)
        )

    def global_forward(self, patches: Tensor) -> Tensor:
        return self.global_blocks(patches)

    @staticmethod
    def _boundary_metrics(
        log_probs: Tensor,
        oracle: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        predicted = log_probs > math.log(0.5)
        true_positive = (predicted & oracle & valid).sum().float()
        false_positive = (predicted & ~oracle & valid).sum().float()
        false_negative = (~predicted & oracle & valid).sum().float()
        pred_positive = (predicted & valid).sum().float()
        target_positive = (oracle & valid).sum().float()
        correct = ((predicted == oracle) & valid).sum().float()
        positions = valid.sum().float()
        accuracy = correct / positions.clamp_min(1)
        precision = true_positive / pred_positive.clamp_min(1)
        recall = true_positive / target_positive.clamp_min(1)
        return (
            accuracy,
            precision,
            recall,
            correct,
            true_positive,
            false_positive,
            false_negative,
            positions,
        )

    @staticmethod
    def _boundary_valid_mask(valid: Tensor) -> Tensor:
        """Positions whose one-byte-lookahead boundary score is defined."""

        boundary_valid = valid.clone()
        last_valid = valid.sum(1).sub(1)
        boundary_valid.scatter_(1, last_valid[:, None], False)
        return boundary_valid

    @staticmethod
    def _boundary_loss(log_probs: Tensor, oracle: Tensor, valid: Tensor) -> Tensor:
        log_probs = torch.clamp(log_probs, max=-1e-6)
        elementwise = torch.where(oracle, -log_probs, -_log1mexp(log_probs))
        return elementwise.masked_select(valid).mean()

    @staticmethod
    def _force_well_formed_boundaries(
        log_probs: Tensor,
        valid: Tensor,
    ) -> Tensor:
        predicted = (log_probs > math.log(0.5)) & valid
        predicted[:, 0] = True
        return predicted

    @staticmethod
    def _fixed_stride_boundaries(valid: Tensor, stride: int) -> Tensor:
        """Content-free patching: a boundary every ``stride`` valid bytes.

        The diagnostic floor for the learned boundary predictor. It keeps the
        well-formedness the pooler relies on — the first byte is a boundary,
        the last valid byte of every row closes its final patch, and nothing
        outside ``valid`` is a boundary — so the only variable against a
        predicted-boundary run is where the patches fall.
        """

        if stride < 1:
            raise ValueError("stride must be positive")
        offsets = valid.cumsum(1) - 1
        boundaries = ((offsets + 1) % stride == 0) & valid
        boundaries[:, 0] = True
        lengths = valid.sum(1)
        rows = torch.arange(valid.shape[0], device=valid.device)
        boundaries[rows, (lengths - 1).clamp_min(0)] = True
        return boundaries

    @staticmethod
    def _uniform_boundaries(valid: Tensor, patch_counts: Tensor) -> Tensor:
        """Placement-blind patching that matches a target patch count per row.

        The honest floor for the learned predictor. A global fixed stride
        cannot be compared against oracle patching here: the oracle spends
        exactly one patch per source token and so saturates the pooled patch
        budget, while any stride dense enough to match its mean density
        overruns that budget on byte-rich rows. Matching ``patch_counts``
        row-by-row holds the patch count, the budget, and the trunk's
        sequence length fixed, leaving placement as the variable the
        predictor actually decides.

        One residual asymmetry survives the count match. This tiles the row:
        the final split point always lands on the last valid byte.
        ``_force_well_formed_boundaries`` never closes that byte, because its
        lookahead score there is ``-100_000``, so the predicted arm leaves a
        trailing partial patch unpooled. At the measured ~4.6 bytes per patch
        against ~9k valid bytes per row that is under 0.1% of the row, but the
        floor is tiling a row that the arm it brackets does not, so the
        comparison is placement plus tail-closure convention, not placement
        alone.

        This is placement-blind, not content-blind: ``patch_counts`` is
        supplied per row, and every source of it is content-derived. The
        oracle count is the subword tokenizer's compression rate for that
        row's text; the predicted count is the model's own. A floor built
        this way therefore already knows how many patches the row deserves
        and is only denied where to put them. Read it as a bound on the
        value of *placement*, never as a bound on the value of knowing
        anything about the content.

        Match the count to the arm being bracketed. The predictor and the
        oracle do not agree on it — on the canonical validation span the
        oracle spends about 5% more patches than the predictor — so an
        oracle-count floor is a systematically finer patching than the
        predicted arm it would be compared against.

        Position 0 is its own patch, mirroring the oracle's BOS patch, and
        the remaining patches divide the rest of the row as evenly as the
        integer split allows.
        """

        lengths = valid.sum(1, keepdim=True)
        remaining_patches = patch_counts[:, None] - 1
        remaining_bytes = (lengths - 1).clamp_min(1)
        if int(remaining_patches.min()) < 1:
            # Reachable from predicted counts and not from oracle ones: a
            # checkpoint whose predictor fires nowhere still gets the forced
            # leading boundary, so its count is 1. Name the arm, or this
            # aborts mid-evaluation with no clue which one produced it.
            raise ValueError(
                "every row needs a BOS patch and at least one more; the "
                "supplied patch count leaves a row with none, which a "
                "predictor that fires nowhere will produce"
            )
        if int((remaining_patches > remaining_bytes).sum()):
            raise ValueError("more patches requested than the row has bytes")
        offsets = valid.cumsum(1) - 1
        # A byte closes a patch exactly when the even split crosses an integer
        # boundary at it, which places the final patch end on the last valid
        # byte and yields precisely ``remaining_patches`` of them.
        current = offsets * remaining_patches // remaining_bytes
        previous = (offsets - 1) * remaining_patches // remaining_bytes
        boundaries = (current != previous) & (offsets >= 1) & valid
        boundaries[:, 0] = True
        return boundaries

    def _fused_targets(self, byte_ids: Tensor, boundaries: Tensor) -> Tensor:
        return byte_ids + boundaries.long() * self.atomic_vocab_size

    @staticmethod
    def _original_boundaries(batch: BolmoBatch) -> Tensor:
        """Source-token ends before the released recipe's EOT coalescing."""

        ends = batch.patch_lens.cumsum(1).sub(1).clamp_min(0)
        counts = torch.zeros_like(batch.oracle_boundaries, dtype=torch.int32)
        counts.scatter_add_(
            1,
            ends.clamp(max=counts.shape[1] - 1),
            batch.source_valid_mask.to(torch.int32),
        )
        return counts > 0

    @staticmethod
    def _teacher_patch_alignment(batch: BolmoBatch) -> tuple[Tensor, Tensor]:
        """Map EOS-coalesced oracle patches back to source teacher positions."""

        keep = batch.source_valid_mask.clone()
        keep[:, :-1] &= ~(
            batch.source_valid_mask[:, 1:]
            & batch.source_ids[:, 1:].eq(SOURCE_EOT_ID)
        )
        keep[:, 0] = True
        width = batch.source_ids.shape[1]
        source_indices = torch.arange(width, device=batch.source_ids.device)[None, :]
        sortable = torch.where(keep, source_indices, width)
        aligned_indices = sortable.sort(dim=1).values
        aligned_valid = aligned_indices < width
        return aligned_indices.clamp(max=width - 1), aligned_valid

    @staticmethod
    def _ce(logits: Tensor, targets: Tensor, valid: Tensor) -> Tensor:
        return F.cross_entropy(logits[valid], targets[valid])

    def stage1(
        self,
        batch: BolmoBatch,
        teacher: BolmoTeacher,
        *,
        teacher_positions_per_chunk: int = DEFAULT_TEACHER_POSITIONS_PER_CHUNK,
    ) -> BolmoLoss:
        byte_hidden, boundary_log_probs = self.local_encoder(
            batch.byte_ids, batch.expanded_ids, batch.valid_mask
        )
        oracle_patches, patch_valid = self.local_encoder.pool(
            byte_hidden,
            batch.oracle_boundaries,
            max_patches=batch.source_ids.shape[1],
        )
        with torch.no_grad():
            teacher_stitched, teacher_final = teacher.forward(
                batch.source_ids.clamp(max=self.local_encoder.source_vocab_size - 1),
                self.stitch_depth,
            )
        teacher_indices, teacher_patch_valid = self._teacher_patch_alignment(batch)
        gather_index = teacher_indices[..., None].expand(
            -1, -1, teacher_final.shape[-1]
        )
        aligned_teacher_stitched = torch.gather(
            teacher_stitched, 1, gather_index
        )
        aligned_teacher_final = torch.gather(teacher_final, 1, gather_index)
        student_stitched = oracle_patches
        for block in list(self.global_blocks)[: self.stitch_depth]:
            student_stitched = block(student_stitched)
        aligned = patch_valid & teacher_patch_valid[:, : patch_valid.shape[1]]
        stitch_difference = (
            student_stitched[aligned].float()
            - aligned_teacher_stitched[:, : student_stitched.shape[1]][aligned].float()
        )
        encoder_stitch = (
            torch.linalg.vector_norm(stitch_difference, dim=-1)
            / math.sqrt(stitch_difference.shape[-1])
        ).mean()
        encoder_stitch_cosine = (
            1.0
            - F.cosine_similarity(
                student_stitched[aligned].float(),
                aligned_teacher_stitched[:, : student_stitched.shape[1]][
                    aligned
                ].float(),
                dim=-1,
            )
        ).mean()

        # The paper's oracle-patch Stage 1 feeds teacher final hidden states
        # directly to the local decoder; the student global path is optimized
        # independently by the stitching loss above.
        patch_hidden = aligned_teacher_final[:, : oracle_patches.shape[1]]
        logits = self.local_decoder(
            byte_hidden, patch_hidden, batch.oracle_boundaries
        )
        target_valid = batch.score_mask[:, 1:]
        targets = self._fused_targets(
            batch.byte_ids[:, 1:], batch.oracle_boundaries[:, 1:]
        )
        ce = self._ce(logits[:, :-1], targets, target_valid)

        student_log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
        selected = torch.gather(
            student_log_probs,
            -1,
            targets[..., None],
        ).squeeze(-1)
        # Distillation remains over the source tokenizer's original patches,
        # even though byte/global routing coalesces boundaries before EOT.
        original_boundaries = self._original_boundaries(batch)
        target_boundaries = original_boundaries[:, 1:]
        target_patch_ids = (
            original_boundaries.long().cumsum(1)[:, 1:]
            - target_boundaries.long()
            - 1
        ).clamp_min(0)
        num_target_patches = batch.source_ids.shape[1] - 1
        patch_log_probs = selected.new_zeros(
            selected.shape[0], num_target_patches
        )
        patch_log_probs.scatter_add_(
            1,
            target_patch_ids.clamp(max=num_target_patches - 1),
            selected * target_valid,
        )
        teacher_targets = batch.source_ids[:, 1:].clamp(
            max=self.local_encoder.source_vocab_size - 1
        )
        teacher_selected = teacher.selected_log_probs(
            teacher_final[:, :-1],
            teacher_targets,
            positions_per_chunk=teacher_positions_per_chunk,
        )
        decoder_distill_elementwise = bernoulli_kl_from_log_probs(
            teacher_selected,
            patch_log_probs,
            temperature=5.0,
        )
        target_source_valid = batch.source_valid_mask[:, 1:]
        decoder_distill = decoder_distill_elementwise[
            target_source_valid
        ].mean()
        boundary_valid = self._boundary_valid_mask(batch.valid_mask)
        boundary = self._boundary_loss(
            boundary_log_probs, batch.oracle_boundaries, boundary_valid
        )
        (
            accuracy,
            precision,
            recall,
            boundary_correct,
            boundary_true_positive,
            boundary_false_positive,
            boundary_false_negative,
            boundary_positions,
        ) = self._boundary_metrics(
            boundary_log_probs, batch.oracle_boundaries, boundary_valid
        )
        total = 4.0 * boundary + encoder_stitch + decoder_distill + ce
        return BolmoLoss(
            total=total,
            ce=ce,
            boundary=boundary,
            encoder_stitch=encoder_stitch,
            encoder_stitch_cosine=encoder_stitch_cosine,
            decoder_distill=decoder_distill,
            boundary_accuracy=accuracy,
            boundary_precision=precision,
            boundary_recall=recall,
            bytes_per_patch=batch.valid_mask.sum() / batch.oracle_boundaries.sum(),
            boundary_correct=boundary_correct,
            boundary_true_positive=boundary_true_positive,
            boundary_false_positive=boundary_false_positive,
            boundary_false_negative=boundary_false_negative,
            boundary_positions=boundary_positions,
            valid_bytes=batch.valid_mask.sum(),
            predicted_patches=batch.oracle_boundaries.sum(),
        )

    def stage2(self, batch: BolmoBatch, *, oracle_boundaries: bool = False) -> BolmoLoss:
        byte_hidden, boundary_log_probs = self.local_encoder(
            batch.byte_ids, batch.expanded_ids, batch.valid_mask
        )
        boundaries = (
            batch.oracle_boundaries
            if oracle_boundaries
            else self._force_well_formed_boundaries(boundary_log_probs, batch.valid_mask)
        )
        pooled, _ = self.local_encoder.pool(
            byte_hidden, boundaries, max_patches=batch.source_ids.shape[1]
        )
        patch_hidden = self.global_forward(pooled)
        logits = self.local_decoder(byte_hidden, patch_hidden, boundaries)
        target_valid = batch.score_mask[:, 1:]
        targets = self._fused_targets(batch.byte_ids[:, 1:], boundaries[:, 1:])
        ce = self._ce(logits[:, :-1], targets, target_valid)
        boundary_valid = self._boundary_valid_mask(batch.valid_mask)
        boundary = self._boundary_loss(
            boundary_log_probs, batch.oracle_boundaries, boundary_valid
        )
        (
            accuracy,
            precision,
            recall,
            boundary_correct,
            boundary_true_positive,
            boundary_false_positive,
            boundary_false_negative,
            boundary_positions,
        ) = self._boundary_metrics(
            boundary_log_probs, batch.oracle_boundaries, boundary_valid
        )
        return BolmoLoss(
            total=ce + 4.0 * boundary,
            ce=ce,
            boundary=boundary,
            boundary_accuracy=accuracy,
            boundary_precision=precision,
            boundary_recall=recall,
            bytes_per_patch=batch.valid_mask.sum() / boundaries.sum(),
            boundary_correct=boundary_correct,
            boundary_true_positive=boundary_true_positive,
            boundary_false_positive=boundary_false_positive,
            boundary_false_negative=boundary_false_negative,
            boundary_positions=boundary_positions,
            valid_bytes=batch.valid_mask.sum(),
            predicted_patches=boundaries.sum(),
        )

    @torch.no_grad()
    def validation_statistics(
        self,
        batch: BolmoBatch,
        *,
        oracle_boundaries: bool = False,
        fixed_stride: int | None = None,
        uniform_patching: str | None = None,
        causal_routing: bool = False,
        patch_budget: int | None = None,
    ) -> BolmoValidationStatistics:
        """Return additive paper-style byte and diagnostic joint statistics.

        ``oracle_boundaries`` is a Stage-1 emulation diagnostic,
        ``fixed_stride`` is a content-free patching at a chosen density, and
        ``uniform_patching`` selects the placement-blind floor, naming the
        arm whose per-row patch count it matches: ``"oracle"`` to bracket
        oracle patching, ``"predicted"`` to bracket the learned predictor.
        Those counts differ by about 5% on the canonical span, so a floor
        matched to one arm is not a controlled floor for the other. All
        three are diagnostics; canonical evaluation always uses predicted
        boundaries.

        ``causal_routing`` is the codelength diagnostic, and applies only to
        the predicted boundaries. The other three patchings are non-causal by
        construction and by much more than one byte: oracle boundaries come
        from the subword tokenizer, whose segmentation of a prefix can change
        arbitrarily far ahead, and the uniform and fixed-stride masks are
        derived from whole-row quantities. Shifting their routing would not
        make them codelengths, so the combination is refused rather than
        reported.

        The boundary
        predictor is non-causal by design (Section 3.1.1): ``log_p[t]`` reads
        ``hidden[t]`` and ``hidden[t + 1]``, so ``boundaries[t]`` is a function
        of byte ``t + 1``. Routing on it means position ``t`` is conditioned on
        a bit derived from the very byte it scores, which the paper accounts
        for as "the single bit of information leaked by discrete boundary
        predictions". The marginal byte distribution therefore does not
        normalize and ``byte_bpb`` is not a codelength. Under
        ``causal_routing`` a position reads the most recent patch that closed
        *strictly before* it, so everything scoring byte ``t + 1`` is a
        function of bytes ``0..t`` and the marginal is a valid codelength.

        The patch contents are unaffected either way: ``pool`` selects
        ``byte_hidden`` at patch ends, all of which are causal. Only which
        patch a position reads changes. Note that the model is trained with
        the non-causal routing, so this diagnostic charges a train/eval
        mismatch on top of the leak and is an upper bound on what a
        causally-routed model would cost.
        """

        if (
            uniform_patching is not None
            and uniform_patching not in self.UNIFORM_PATCH_COUNTS
        ):
            raise ValueError(
                "uniform_patching names the arm whose patch count the floor "
                f"matches, one of {self.UNIFORM_PATCH_COUNTS}"
            )
        chosen_patchings = (
            oracle_boundaries,
            fixed_stride is not None,
            uniform_patching is not None,
        )
        if sum(chosen_patchings) > 1:
            raise ValueError("these options select different patchings")
        if causal_routing and any(chosen_patchings):
            raise ValueError(
                "causal routing is only a codelength under the predicted "
                "boundaries; the oracle, uniform and fixed-stride patchings "
                "are derived from whole-row quantities"
            )
        byte_hidden, boundary_log_probs = self.local_encoder(
            batch.byte_ids, batch.expanded_ids, batch.valid_mask
        )
        if oracle_boundaries:
            boundaries = batch.oracle_boundaries
        elif fixed_stride is not None:
            boundaries = self._fixed_stride_boundaries(
                batch.valid_mask, fixed_stride
            )
        elif uniform_patching is not None:
            # The floor is only controlled against the arm it is count-matched
            # to, so the predicted count is recomputed here rather than reusing
            # the oracle's.
            patch_counts = (
                batch.oracle_boundaries.sum(1)
                if uniform_patching == "oracle"
                else self._force_well_formed_boundaries(
                    boundary_log_probs, batch.valid_mask
                ).sum(1)
            )
            boundaries = self._uniform_boundaries(batch.valid_mask, patch_counts)
        else:
            boundaries = self._force_well_formed_boundaries(
                boundary_log_probs, batch.valid_mask
            )
        # ``pool`` truncates silently past its patch budget, which would make a
        # denser patching look better than it is by dropping the row's tail.
        #
        # The default budget is the stored source width, which is exactly right
        # on the canonical panel, where the predictor emits fewer patches than
        # the tokenizer did. It is not a modelling limit: the trunk is NoPE and
        # `pool` only needs a fixed-size packed tensor. On out-of-distribution
        # panels the predictor over-segments and can exceed it, so evaluation
        # may raise the budget rather than report a truncated row. Doing so
        # lets the trunk see a slightly longer sequence than training did;
        # record the value used, and never lower it below the stored width.
        budget = batch.source_ids.shape[1] if patch_budget is None else patch_budget
        if budget < batch.source_ids.shape[1]:
            raise ValueError(
                "patch budget is below the stored source width, which would "
                f"truncate rows the training geometry admits: {budget} < "
                f"{batch.source_ids.shape[1]}"
            )
        if int(boundaries.sum(1).max()) > budget:
            raise ValueError(
                "patching exceeded the pooled patch budget: "
                f"{int(boundaries.sum(1).max())} > {budget}"
            )
        pooled, _ = self.local_encoder.pool(
            byte_hidden, boundaries, max_patches=budget
        )
        # ``prepare_hidden`` routes position ``t`` to patch
        # ``cumsum(boundaries)[t] - 1``, which consumes ``boundaries[t]``.
        # Shifting the routing mask right by one drops that term, so the
        # position reads the last patch closed at or before ``t - 1``. The
        # shifted-in ``False`` at index 0 makes the cumsum there ``-1``, and
        # ``prepare_hidden``'s ``clamp(min=0)`` — which the unshifted mask's
        # forced leading boundary never engages — maps it onto patch 0. That
        # is the one position reading a patch closed *at* rather than before
        # it, and it stays causal: patch 0 is ``byte_hidden[0]``, a function of
        # byte 0, while position 0 scores byte 1.
        routing_boundaries = (
            F.pad(boundaries[:, :-1], (1, 0), value=False)
            if causal_routing
            else boundaries
        )
        logits = self.local_decoder(
            byte_hidden, self.global_forward(pooled), routing_boundaries
        )
        # Padding carries ``byte_pad_id``, which sits one past the atomic
        # vocabulary, so select the scored positions before gathering exactly
        # as ``_ce`` does; gathering over the padded grid would index out of
        # bounds on the marginal byte distribution.
        valid = batch.score_mask[:, 1:]
        log_probs = F.log_softmax(logits[:, :-1][valid].float(), dim=-1)
        byte_targets = batch.byte_ids[:, 1:][valid]
        fused_targets = self._fused_targets(byte_targets, boundaries[:, 1:][valid])
        selected_joint = log_probs.gather(-1, fused_targets[:, None]).squeeze(-1)
        # This is the released Bolmo evaluation rule: add the probabilities of
        # ``byte`` and ``byte + boundary`` before scoring the observed byte.
        # The non-causal prefill boundary predictor still determines routing;
        # the causal output-boundary decision is a latent label at evaluation.
        marginal_byte_log_probs = torch.logaddexp(
            log_probs[:, : self.atomic_vocab_size],
            log_probs[:, self.atomic_vocab_size :],
        )
        selected_byte = marginal_byte_log_probs.gather(
            -1, byte_targets[:, None]
        ).squeeze(-1)
        # The challenge convention charges registered special symbols one byte,
        # matching ByteCounter and the GPT-2 EOT byte LUT. Padding is invalid.
        scored_bytes = valid.sum()
        boundary_valid = self._boundary_valid_mask(batch.valid_mask)
        boundary_predictions = boundary_log_probs > math.log(0.5)
        boundary_correct = (
            (boundary_predictions == batch.oracle_boundaries) & boundary_valid
        ).sum()
        boundary_true_positive = (
            boundary_predictions & batch.oracle_boundaries & boundary_valid
        ).sum()
        boundary_false_positive = (
            boundary_predictions & ~batch.oracle_boundaries & boundary_valid
        ).sum()
        boundary_false_negative = (
            ~boundary_predictions & batch.oracle_boundaries & boundary_valid
        ).sum()
        return BolmoValidationStatistics(
            byte_nll=-selected_byte.sum(),
            joint_nll=-selected_joint.sum(),
            scored_bytes=scored_bytes,
            valid_atoms=batch.valid_mask.sum(),
            predicted_patches=boundaries.sum(),
            boundary_correct=boundary_correct,
            boundary_positions=boundary_valid.sum(),
            boundary_true_positive=boundary_true_positive,
            boundary_false_positive=boundary_false_positive,
            boundary_false_negative=boundary_false_negative,
        )


def parameter_report(model: BolmoModel) -> dict[str, int]:
    groups = {
        "expanded_embedding": sum(
            p.numel()
            for p in model.local_encoder.expanded_embedding.parameters()
        ),
        "local_encoder": sum(p.numel() for p in model.local_encoder.parameters()),
        "global_model": sum(p.numel() for p in model.global_blocks.parameters()),
        "local_decoder": sum(p.numel() for p in model.local_decoder.parameters()),
    }
    groups["total"] = sum(p.numel() for p in model.parameters())
    return groups

"""Continue shared-RMS V1 probes with checkpoint-compatible long-context RoPE.

The fixed-target YaRN frequencies are selected once from the configured target
length.  Positions in the pretrained 1K window retain their exact original
RoPE phases; extrapolated phases continue from that boundary with the blended
YaRN frequencies.  The same absolute-position routine is used by full-sequence
attention and incremental KV-cache generation.
"""

from __future__ import annotations

import inspect
import math
import os
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import fresh_lejepa_train as v1
import train_gpt as baseline
from fresh_lejepa_train_v1_probe_shared_rms_projector import (
    FreshLeJEPASharedRMSProjectorV1Probes,
)


YARN_ARCHITECTURE = "fresh_lejepa_shared_rms_v1_probes_fixed_yarn_4k"
CONTROL_ARCHITECTURE = "fresh_lejepa_shared_rms_v1_probes_rope_4k_control"
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "ablation_results/fresh_lejepa_shared_rms_v1_probes_1k/pretraining_checkpoint.pt"
)


def _correction_dim(
    rotations: float, dim: int, base: float, pretrained_length: int
) -> float:
    return dim * math.log(pretrained_length / (rotations * 2 * math.pi)) / (
        2 * math.log(base)
    )


def _linear_ramp(low: float, high: float, width: int) -> Tensor:
    if low == high:
        high += 1e-3
    return ((torch.arange(width, dtype=torch.float32) - low) / (high - low)).clamp_(0, 1)


class FixedTargetYarnRotary(nn.Module):
    """Exact-prefix, continuous fixed-target YaRN rotary phases."""

    def __init__(
        self,
        dim: int,
        base: float = 10_000.0,
        pretrained_length: int = 1024,
        target_length: int = 65_536,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        super().__init__()
        if target_length < pretrained_length:
            raise ValueError("target_length must be at least pretrained_length")
        if dim % 2:
            raise ValueError("rotary dimension must be even")
        self.pretrained_length = pretrained_length
        self.target_length = target_length
        base_inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        factor = target_length / pretrained_length
        if factor == 1:
            yarn_inv_freq = base_inv_freq.clone()
        else:
            low = math.floor(
                _correction_dim(beta_fast, dim, base, pretrained_length)
            )
            high = math.ceil(
                _correction_dim(beta_slow, dim, base, pretrained_length)
            )
            low = max(low, 0)
            high = min(high, dim - 1)
            ramp = _linear_ramp(low, high, dim // 2)
            yarn_inv_freq = base_inv_freq * (1 - ramp) + base_inv_freq / factor * ramp
        # Keep the familiar non-persistent name so old checkpoints remain strict-load compatible.
        self.register_buffer("inv_freq", base_inv_freq, persistent=False)
        self.register_buffer("yarn_inv_freq", yarn_inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def frequencies(self, positions: Tensor) -> Tensor:
        positions = positions.to(dtype=self.inv_freq.dtype)
        prefix = positions.clamp_max(self.pretrained_length)
        extension = (positions - self.pretrained_length).clamp_min(0)
        return torch.outer(prefix, self.inv_freq) + torch.outer(
            extension, self.yarn_inv_freq
        )

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            positions = torch.arange(seq_len, device=device)
            frequencies = self.frequencies(positions)
            self._cos_cached = frequencies.cos()[None, None]
            self._sin_cached = frequencies.sin()[None, None]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype), self._sin_cached.to(dtype)


class FreshLeJEPASharedRMSV1FixedYarn(FreshLeJEPASharedRMSProjectorV1Probes):
    """Shared-RMS V1 model with strict checkpoint continuation and fixed YaRN."""

    pretrained_context = int(os.environ.get("FRESH_YARN_PRETRAINED_CONTEXT", "1024"))
    target_context = int(os.environ.get("FRESH_YARN_TARGET_CONTEXT", "65536"))
    position_mode = os.environ.get("FRESH_POSITION_MODE", "yarn")
    defer_sigreg = True
    sigreg_loss_weight = v1.FreshHyperparameters.sigreg_weight

    def __init__(self, *args, **kwargs):
        rope_base = float(kwargs.get("rope_base", 10_000.0))
        super().__init__(*args, **kwargs)
        if self.position_mode not in {"rope", "yarn"}:
            raise ValueError(f"unknown FRESH_POSITION_MODE={self.position_mode!r}")
        if self.position_mode == "yarn":
            for block in self.blocks:
                block.attn.rotary = FixedTargetYarnRotary(
                    block.attn.head_dim,
                    base=rope_base,
                    pretrained_length=self.pretrained_context,
                    target_length=self.target_context,
                )
        checkpoint = os.environ.get("FRESH_YARN_INIT_CHECKPOINT")
        if checkpoint:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = payload.get("model", payload)
            self.load_state_dict(state, strict=True)

    def _attention_step(
        self,
        attention: baseline.CausalSelfAttention,
        x: Tensor,
        cache: tuple[Tensor, Tensor],
        position: int | Tensor,
        key_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if not isinstance(attention.rotary, FixedTargetYarnRotary):
            return super()._attention_step(attention, x, cache, position, key_mask)
        batch, _, dim = x.shape
        q_dim = attention.num_heads * attention.head_dim
        kv_dim = attention.num_kv_heads * attention.head_dim
        q, k, value = attention.c_qkv(x).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(batch, 1, attention.num_heads, attention.head_dim).transpose(1, 2)
        k = k.view(batch, 1, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
        value = value.view(batch, 1, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        position_tensor = torch.as_tensor(position, device=x.device).reshape(1)
        frequency = attention.rotary.frequencies(position_tensor)[0]
        cos = frequency.cos()[None, None, None].to(q.dtype)
        sin = frequency.sin()[None, None, None].to(q.dtype)
        q = baseline.apply_rotary_emb(q, cos, sin)
        k = baseline.apply_rotary_emb(k, cos, sin)
        q = q * attention.q_gain.to(q.dtype)[None, :, None, None]
        attn_mask = None
        if key_mask is not None and key_mask.dim() == 2:
            # Per-row (batch, keys) validity for left-padded batched rollouts
            # (see FreshLeJEPAGPT._attention_step).
            position_length = int(position) + 1
            cache[0][:, :, position : position_length].copy_(k)
            cache[1][:, :, position : position_length].copy_(value)
            prefix_k = cache[0][:, :, :position_length]
            prefix_v = cache[1][:, :, :position_length]
            attn_mask = key_mask[:, None, None, :position_length]
        elif key_mask is not None:
            # Static full-cache path (see FreshLeJEPAGPT._attention_step).
            if not torch.is_tensor(position):
                raise ValueError("key_mask stepping requires a 0-dim tensor position")
            index = position.reshape(1)
            cache[0].index_copy_(2, index, k)
            cache[1].index_copy_(2, index, value)
            prefix_k, prefix_v = cache
            attn_mask = key_mask[None, None, None, :]
        elif torch.is_tensor(position):
            index = position.reshape(1)
            cache[0].index_copy_(2, index, k)
            cache[1].index_copy_(2, index, value)
            prefix_k = torch.narrow(cache[0], 2, 0, position + 1)
            prefix_v = torch.narrow(cache[1], 2, 0, position + 1)
        else:
            cache[0][:, :, position : position + 1].copy_(k)
            cache[1][:, :, position : position + 1].copy_(value)
            prefix_k = cache[0][:, :, : position + 1]
            prefix_v = cache[1][:, :, : position + 1]
        y = F.scaled_dot_product_attention(
            q,
            prefix_k,
            prefix_v,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=attention.num_kv_heads != attention.num_heads,
        )
        y = y.transpose(1, 2).contiguous().view(batch, 1, dim)
        return attention.proj(y), cache

    def deferred_sigreg_loss(
        self, batches: list[tuple[Tensor, Tensor]]
    ) -> Tensor:
        """One exact B=128 statistic assembled from memory-safe B=16 forwards."""
        expected_batch = int(os.environ.get("FRESH_SIGREG_EFFECTIVE_BATCH", "128"))
        trajectories = []
        for input_ids, target_ids in batches:
            trajectory_ids = torch.cat((input_ids, target_ids[:, -1:]), dim=1)
            trajectories.append(self.embed_tokens(trajectory_ids))
        embeddings = torch.cat(trajectories, dim=0)
        if embeddings.size(0) != expected_batch:
            raise ValueError(
                f"expected effective SIGReg B={expected_batch}, got {embeddings.size(0)}"
            )
        return self.sigreg(embeddings)

    @classmethod
    def experiment_metadata(cls) -> dict[str, int | str]:
        return {
            "position_encoding": (
                "fixed_target_yarn_exact_prefix"
                if cls.position_mode == "yarn"
                else "standard_rope_control"
            ),
            "pretrained_context": cls.pretrained_context,
            "target_context": cls.target_context,
            "initial_checkpoint": os.environ.get(
                "FRESH_YARN_INIT_CHECKPOINT", str(DEFAULT_CHECKPOINT)
            ),
        }


def _install_context_adaptation_loop(default_steps: int = 8):
    """Use B=16 forwards while preserving one exact B=128 SIGReg objective."""
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
        "        raise ValueError(\"GRAD_ACCUM_STEPS must be positive and divisible by WORLD_SIZE\")\n"
        "    grad_accum_steps = total_grad_accum_steps // world_size"
    )
    if source.count(old) != 1:
        raise RuntimeError("upstream accumulation block changed")
    source = source.replace(old, new)
    warmup_start = (
        "for warmup_step in range(args.warmup_steps):\n"
        "            zero_grad_all()\n"
        "            for micro_step in range(grad_accum_steps):"
    )
    warmup_start_new = (
        "for warmup_step in range(args.warmup_steps):\n"
        "            zero_grad_all()\n"
        "            sigreg_batches = []\n"
        "            for micro_step in range(grad_accum_steps):"
    )
    warmup_backward = "(warmup_loss * grad_scale).backward()\n            for opt in optimizers:"
    warmup_backward_new = (
        "(warmup_loss * grad_scale).backward()\n"
        "                sigreg_batches.append((x, y))\n"
        "            warmup_sigreg = base_model.deferred_sigreg_loss(sigreg_batches)\n"
        "            (warmup_sigreg * base_model.sigreg_loss_weight).backward()\n"
        "            for opt in optimizers:"
    )
    train_start = (
        "zero_grad_all()\n"
        "        train_loss = torch.zeros((), device=device)\n"
        "        for micro_step in range(grad_accum_steps):"
    )
    train_start_new = (
        "zero_grad_all()\n"
        "        train_loss = torch.zeros((), device=device)\n"
        "        sigreg_batches = []\n"
        "        for micro_step in range(grad_accum_steps):"
    )
    train_backward = (
        "(loss * grad_scale).backward()\n"
        "        train_loss /= grad_accum_steps"
    )
    train_backward_new = (
        "(loss * grad_scale).backward()\n"
        "            sigreg_batches.append((x, y))\n"
        "        sigreg_loss = base_model.deferred_sigreg_loss(sigreg_batches)\n"
        "        (sigreg_loss * base_model.sigreg_loss_weight).backward()\n"
        "        train_loss /= grad_accum_steps\n"
        "        train_loss += base_model.sigreg_loss_weight * sigreg_loss.detach()"
    )
    log_old = (
        'f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "\n'
        '                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"'
    )
    log_new = (
        'f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "\n'
        '                f"sigreg_loss:{sigreg_loss.item():.4f} "\n'
        '                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"'
    )
    for expected, replacement, label in (
        (warmup_start, warmup_start_new, "warmup batch collection"),
        (warmup_backward, warmup_backward_new, "warmup SIGReg backward"),
        (train_start, train_start_new, "training batch collection"),
        (train_backward, train_backward_new, "training SIGReg backward"),
        (log_old, log_new, "SIGReg logging"),
    ):
        if source.count(expected) != 1:
            raise RuntimeError(f"upstream {label} block changed")
        source = source.replace(expected, replacement)
    exec(compile(source, inspect.getsourcefile(original) or "train_gpt.py", "exec"), baseline.__dict__)
    return original


def main() -> None:
    checkpoint = Path(os.environ.setdefault("FRESH_YARN_INIT_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"initial checkpoint not found: {checkpoint}")
    original_main = _install_context_adaptation_loop()
    v1.FreshLeJEPAGPT = FreshLeJEPASharedRMSV1FixedYarn
    v1.EXPERIMENT_ARCHITECTURE = (
        YARN_ARCHITECTURE
        if FreshLeJEPASharedRMSV1FixedYarn.position_mode == "yarn"
        else CONTROL_ARCHITECTURE
    )
    v1.EXPERIMENT_SOURCE = Path(__file__)
    os.environ.setdefault("GRAD_ACCUM_STEPS", "8")
    try:
        v1.main()
    finally:
        baseline.main = original_main


if __name__ == "__main__":
    main()

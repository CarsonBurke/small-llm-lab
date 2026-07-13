"""Baseline pretraining with PoPE attention and the NextLat auxiliary objective.

Exactly two changes to ``train_gpt.py``, both from scratch:

1. PoPE Q/K geometry ported from ``fresh_lejepa_train_v1_probe_shared_rms_pope``
   with the zero phase-offset initialization of the ``_pope_zero`` variant
   (``POPE_PHASE_INIT=two_pi`` restores the reference two-pi interval init).
2. The NextLat next-latent auxiliary (arXiv 2511.05963) at its method defaults:
   ``lambda_mse=1.0``, ``lambda_kl=1.0``, ``mtp_horizon=1``, ``proj_factor=1.0``.
   ``lambda_ce`` defaults to 0 upstream (logging only) and is omitted here.

Everything else — architecture, optimizer split, schedules, quantized export —
is the unmodified baseline.  Adaptations forced by the baseline's conventions:

- Unlike the reference PoPE port (and the baseline's own Rotary), attention
  angles are computed in fp32 at forward time instead of through a buffer that
  baseline.main's .bfloat16() cast would round: bf16 positions quantize to
  multiples of 4 above 512, corrupting high-frequency phases — fatally so at
  the longer contexts this model is headed for.
- Teacher and student logits both pass through the baseline logit softcap, so
  the KL compares the model's actual output distribution.
- "Next token embeddings" are the baseline's true stream inputs,
  ``rms_norm(tok_emb(x))`` (the norm is parameter-free, so this differs from
  NextLat's raw ``wte`` output only by a per-position rescale).
- The fineweb shards delimit documents with BOS (id 1), never EOS, and BOS
  starts each document — positionally identical to NextLat's EOS convention,
  whose attention mask also assigns the delimiter to the following document.
  ``NEXTLAT_EOS_ID`` therefore defaults to 1; set -1 to disable the masks.
  Unlike the NextLat repo we keep the baseline's plain causal attention (no
  per-document masking), matching how the baseline treats packed documents.
- The dynamics model registers under ``blocks[-1]`` so the unmodified baseline
  optimizer split sees it (matrices via Muon, the norm gain via scalar Adam),
  the same idiom ``fresh_lejepa_train.py`` uses for its probes.  It is a
  training-only organ (~2.6M params at width 512): it inflates the reported
  int8 artifact size, and a submission-grade export would strip it.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import train_gpt as baseline


class PolarCausalSelfAttention(baseline.CausalSelfAttention):
    """Baseline GQA with PoPE geometry: softplus magnitudes, polar phases.

    The reference K angle is ``theta + delta`` for every Q head.  With GQA we
    instead use Q angle ``theta - delta``, which gives the identical score
    ``cos((theta_k + delta) - theta_q)`` without duplicating cached K/V.
    """

    block_size = int(os.environ.get("TRAIN_SEQ_LEN", "1024"))
    phase_init = os.environ.get("POPE_PHASE_INIT", "zero")
    # The PoPE reference has no query gain; the lejepa port deliberately kept
    # the baseline's q_gain (init 1.5, tuned for RMS-normed Q/K), which makes
    # the already-local positive-magnitude kernel ~1.5x sharper at init.
    # "off" matches the reference (leaves q_gain unused: single-GPU only,
    # DDP would object to the parameter receiving no gradient).
    qk_gain_mode = os.environ.get("POPE_QK_GAIN", "on")

    def __init__(self, *args, **kwargs):
        rope_base = float(kwargs.get("rope_base", args[3] if len(args) > 3 else 10_000.0))
        super().__init__(*args, **kwargs)
        if self.phase_init not in {"zero", "two_pi"}:
            raise ValueError(f"unknown POPE_PHASE_INIT={self.phase_init!r}")
        if self.qk_gain_mode not in {"on", "off"}:
            raise ValueError(f"unknown POPE_QK_GAIN={self.qk_gain_mode!r}")
        # Kept as a plain float, not a buffer: baseline.main's model-wide
        # .bfloat16() would cast a registered inv_freq buffer (as it does the
        # reference's), and bf16 angles corrupt the high-frequency phases at
        # large positions.  Frequencies are recomputed in fp32 every forward.
        self.rope_base = rope_base
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(self.head_dim, dtype=torch.float32) / self.head_dim)
        )
        if self.phase_init == "zero":
            delta = torch.zeros(1, self.num_heads, 1, self.head_dim)
        else:
            # Do not perturb initialization of any matched baseline weights.
            with torch.random.fork_rng(devices=[]):
                upper = torch.zeros_like(inv_freq)
                lower = (
                    -2
                    * math.pi
                    / torch.maximum(inv_freq, torch.tensor(1.0 / self.block_size))
                    * inv_freq
                )
                delta = torch.rand(1, self.num_heads, 1, self.head_dim)
                delta = delta * (upper - lower)[None, None, None] + lower[None, None, None]
        self.delta_c = nn.Parameter(delta)

    def _polar_components(
        self, q: Tensor, k: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return real/imaginary Q/K, retaining the unexpanded KV heads.

        Angles are computed entirely in fp32: positions are exact up to 2^24
        and the theta product carries ~1e-7 relative error, so phases stay
        accurate far beyond any target context length.  Only the bounded
        cos/sin outputs are later rounded to bf16, which is harmless.
        """
        exponents = torch.arange(self.head_dim, device=q.device, dtype=torch.float32)
        inv_freq = self.rope_base ** (-exponents / self.head_dim)
        theta = torch.outer(positions.to(dtype=torch.float32), inv_freq)[None, None]
        delta = self.delta_c.clamp(-2 * math.pi, 0).to(dtype=theta.dtype)
        q_theta = theta - delta
        q_mag = F.softplus(q.float())
        k_mag = F.softplus(k.float())
        q_real = q_mag * q_theta.cos()
        q_imag = q_mag * q_theta.sin()
        k_real = k_mag * theta.cos()
        k_imag = k_mag * theta.sin()
        if self.qk_gain_mode == "off":
            return q_real, q_imag, k_real, k_imag
        gain = self.q_gain.float()[None, :, None, None]
        return q_real * gain, q_imag * gain, k_real, k_imag

    def _complex_attention(
        self,
        q_real: Tensor,
        q_imag: Tensor,
        k_real: Tensor,
        k_imag: Tensor,
        value: Tensor,
    ) -> Tensor:
        # Concatenation turns Re(conj(Q)K) into one ordinary dot product.  The
        # explicit scale is sqrt(complex feature width), not sqrt(2D).  CUDA
        # Flash SDPA requires Q/K/V to have equal feature widths, so append a
        # zero-valued half to V and discard the corresponding zero output.
        query = torch.cat((q_real, q_imag), dim=-1).to(value.dtype)
        key = torch.cat((k_real, k_imag), dim=-1).to(value.dtype)
        padded_value = torch.cat((value, torch.zeros_like(value)), dim=-1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            padded_value,
            is_causal=True,
            enable_gqa=self.num_kv_heads != self.num_heads,
            scale=self.head_dim**-0.5,
        )
        return output[..., : self.head_dim]

    def forward(self, x: Tensor) -> Tensor:
        batch, length, dim = x.shape
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, value = self.c_qkv(x).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        positions = torch.arange(length, device=x.device)
        q_real, q_imag, k_real, k_imag = self._polar_components(q, k, positions)
        y = self._complex_attention(q_real, q_imag, k_real, k_imag, value)
        y = y.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.proj(y)


class NextLatDynamicsModel(nn.Module):
    """NextLat's residual dynamics MLP: (h_t, e(x_{t+1})) -> h_{t+1} estimate."""

    def __init__(self, model_dim: int, proj_factor: float):
        super().__init__()
        input_dim = model_dim * 2
        hidden_dim = 128 * round(proj_factor * input_dim / 128)
        # NextLat's LayerNorm(bias=False) dispatches to F.rms_norm with a
        # learned gain (model_base.py), so this must be RMSNorm, not LayerNorm.
        self.norm_x = nn.RMSNorm(input_dim, eps=1e-5)
        # Construction and NextLat init (normal std 0.02) both draw from the
        # RNG, so the whole block is forked to keep the global stream — and
        # with it every weight shared with a plain baseline run — untouched.
        with torch.random.fork_rng(devices=[]):
            self.mlp = nn.Sequential(
                baseline.CastedLinear(input_dim, hidden_dim, bias=False),
                nn.GELU(),
                baseline.CastedLinear(hidden_dim, hidden_dim, bias=False),
                nn.GELU(),
                baseline.CastedLinear(hidden_dim, model_dim, bias=False),
            )
            for module in self.mlp:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, current_states: Tensor, next_token_embeds: Tensor) -> Tensor:
        x = torch.cat([next_token_embeds, current_states], dim=-1)
        return current_states + self.mlp(self.norm_x(x))


class NextLatGPT(baseline.GPT):
    """Baseline GPT whose training loss adds the NextLat auxiliary.

    In eval mode ``forward`` returns the pure next-token cross-entropy, so
    ``eval_val``'s loss/BPB stay directly comparable with every other run.
    """

    lambda_mse = float(os.environ.get("NEXTLAT_LAMBDA_MSE", "1.0"))
    lambda_kl = float(os.environ.get("NEXTLAT_LAMBDA_KL", "1.0"))
    mtp_horizon = int(os.environ.get("NEXTLAT_HORIZON", "1"))
    proj_factor = float(os.environ.get("NEXTLAT_PROJ_FACTOR", "1.0"))
    eos_token_id = int(os.environ.get("NEXTLAT_EOS_ID", "1"))

    def __init__(self, *args, **kwargs):
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        super().__init__(*args, **kwargs)
        if model_dim is None:
            raise ValueError("model_dim is required")
        if self.mtp_horizon < 1:
            raise ValueError("NEXTLAT_HORIZON must be at least 1")
        # Register under blocks so the unmodified baseline optimizer split
        # sees every dynamics matrix (fresh_lejepa_train.py's probe idiom).
        self.blocks[-1].nextlat_dynamics = NextLatDynamicsModel(model_dim, self.proj_factor)

    @property
    def nextlat_dynamics(self) -> NextLatDynamicsModel:
        return self.blocks[-1].nextlat_dynamics

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        # Baseline trunk, kept unflattened so the hidden states are reusable.
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        hidden = self.final_norm(x)

        if self.tie_embeddings:
            logits_proj = F.linear(hidden, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(hidden)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        ntp_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1)
        )
        if not self.training:
            # eval_val consumes this value directly as the BPB cross-entropy.
            return ntp_loss
        return ntp_loss + self._nextlat_loss(input_ids, x0, hidden, logits)

    def _student_logits(self, predicted: Tensor) -> Tensor:
        # Detached head weights: the auxiliary must not update the embedding
        # through the logit path (NextLat detaches lm_head the same way).
        weight = self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight
        student_proj = F.linear(predicted, weight.detach())
        return self.logit_softcap * torch.tanh(student_proj / self.logit_softcap)

    def _nextlat_loss(self, input_ids: Tensor, input_latents: Tensor, hidden: Tensor, logits: Tensor) -> Tensor:
        if self.eos_token_id >= 0:
            boundary = input_ids == self.eos_token_id
        else:
            boundary = torch.zeros_like(input_ids, dtype=torch.bool)
        token_pred_mask = ~boundary[:, :-1]
        predicted = hidden
        next_inputs = input_latents
        target_states = hidden
        teacher_logits = logits[:, :-1]
        total_mse = 0.0
        total_kl = 0.0
        # Recursive d-step prediction: each iteration feeds the previous
        # prediction back and shifts every target one position right.
        for i in range(self.mtp_horizon):
            predicted = predicted[:, :-1]
            next_inputs = next_inputs[:, 1:]
            target_states = target_states[:, 1:]
            teacher_logits = teacher_logits[:, 1:]
            token_pred_mask = token_pred_mask[:, 1:]
            predicted = self.nextlat_dynamics(predicted, next_inputs)

            # Skip states whose input token is the document delimiter: their
            # prediction crosses a document boundary.
            mse_mask = ~boundary[:, i + 1 :]
            mse_elem = F.smooth_l1_loss(
                predicted.float(), target_states.detach().float(), reduction="none"
            )
            mse_weight = mse_mask[..., None].to(mse_elem.dtype)
            total_mse = total_mse + (mse_elem * mse_weight).sum() / mse_weight.expand_as(
                mse_elem
            ).sum().clamp_min(1.0)

            # KL(teacher || student): ground the predicted state by matching
            # the distribution the trunk would emit from the true state.
            log_teacher = F.log_softmax(teacher_logits.detach().float(), dim=-1)
            log_student = F.log_softmax(self._student_logits(predicted[:, :-1]).float(), dim=-1)
            kl_per_token = F.kl_div(
                log_student, log_teacher, log_target=True, reduction="none"
            ).sum(-1)
            kl_weight = token_pred_mask.to(kl_per_token.dtype)
            total_kl = total_kl + (kl_per_token * kl_weight).sum() / kl_weight.sum().clamp_min(1.0)
        return (self.lambda_mse * total_mse + self.lambda_kl * total_kl) / self.mtp_horizon


def main() -> None:
    original_attention = baseline.CausalSelfAttention
    original_gpt = baseline.GPT
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    original_int8_patterns = baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    baseline.CausalSelfAttention = PolarCausalSelfAttention
    baseline.GPT = NextLatGPT
    baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns + ("delta_c",)
    baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns + ("delta_c",)
    try:
        baseline.main()
    finally:
        baseline.CausalSelfAttention = original_attention
        baseline.GPT = original_gpt
        baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns
        baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns


if __name__ == "__main__":
    main()

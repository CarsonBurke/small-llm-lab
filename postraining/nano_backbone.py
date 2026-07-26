"""Post-training backbone adapters for the nanogpt-mini architecture.

``NanoGPTBackbone``/``NanoTiedDotBackbone`` subclass the pretraining model
classes (``nanogpt_mini_model``) so construction, parameter registration, and
state-dict keys are byte-identical to the training scripts — a pretraining
``final_model.pt`` payload strict-loads with zero key mapping. On top of the
pretrained trunk they add the interface the latent-VAPO stack consumes
(``LatentThoughtModel``/``latent_rollout``/``SeparateCritic``): token
embedding, dense belief, stepwise KV-cache decode, dense prefill, and the
renderer.

Conventions chosen for the latent-thought family:

- ``embed_tokens(ids) = norm1(embed(ids))``: the input norm belongs to the
  TOKEN path, mirroring fresh_lejepa's rms-normed ``tok_emb``. Injected
  thoughts (after their policy/critic adapter) and PAD zeros bypass ``norm1`` so the
  continuous thought coordinates reach the block stack unchanged.
  ``temporal_belief_from_token_latent`` starts at the block loop, and
  ``embed_tokens`` composed with it reproduces the pretraining forward exactly.
- Belief = the post-``norm2`` final hidden state — precisely what the
  pretrained readout consumes, so no new probe is needed.
- ``logits_from_features`` keeps the standard ``cat(input_latent, belief)``
  renderer feature layout and slices the belief half: nano's readout is
  structurally belief-only, and ``proj(belief)`` (or the tied codebook dot)
  followed by the EXACT pretraining rational softcap is the pretrained LM.
  Every path — step, prefill, replay — prices the same softcapped logits.
- The embedding is promoted to fp32 masters at construction (bf16 -> fp32 is
  value-exact; checkpoint loads cast losslessly). Plain-AdamW RL steps at
  ~3e-4 on std-1 embeddings would vanish at bf16 resolution, and the fresh
  path keeps fp32 masters for the same reason (see model_io.py).
- ``__init__`` applies the training scripts' seeded init recipe (zero proj,
  std-1 embed, 0.33**0.5/sqrt(fan) trunk, unit gains, zero bias/scale), so
  ``fresh_trunk`` critics start nano-native with an identity-like residual
  stream instead of PyTorch defaults.

The stepwise/prefill attention mirrors fresh_lejepa_train.py's
``_attention_step``/``_attention_prefill`` branch-for-branch (int position
slice-copy; 0-dim tensor position ``index_copy_`` + narrow; 1-D full-cache
key mask; 2-D per-row left-pad key mask), with the nano deltas: separate
q/k/v Linears, unweighted ``F.rms_norm`` on q/k, half-truncate RoPE applied
at the absolute cache position, SDPA scale 0.12, no GQA, no q_gain.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import nanogpt_mini_model

NANO_DEFAULT_MODEL_CONFIG = {
    "vocab_size": 1024,
    "num_layers": 6,
    "model_dim": 512,
    "mlp_hidden": 2048,
}


class _NanoPostrainingMixin:
    """Post-training interface shared by both nano readout variants.

    Mixed in FRONT of the concrete pretraining class (``GPT``/``TiedDotGPT``)
    — never as a second base beside it, which would break the pretraining
    classes' plain ``super().__init__()`` chains.
    """

    logit_softcap = 15.0

    def _finalize_construction(self, num_layers: int, model_dim: int) -> None:
        self.model_dim = model_dim
        self.num_encoder_layers = num_layers
        self.num_decoder_layers = 0
        # fp32 masters: exact promotion of the bf16 pretraining storage.
        self.embed.float()
        self._apply_pretraining_init()

    def _apply_pretraining_init(self) -> None:
        """The training scripts' seeded init recipe, for fresh (critic) trunks."""
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                w = parameter.data
                if name.endswith("weight"):
                    if "proj" in name:
                        w.zero_()
                    elif "embed" in name:
                        w.normal_()
                    else:
                        w.normal_(std=0.33**0.5 / w.size(-1)**0.5)
                elif name.endswith("bias"):
                    w.zero_()
                elif name.endswith("scale"):
                    w.zero_()
                elif name.endswith("gains"):
                    w.normal_(mean=1, std=0)
                else:
                    raise ValueError(f"uninitialized parameter: {name}")

    @property
    def tok_emb(self) -> nn.Embedding:
        return self.embed

    @property
    def final_norm(self) -> nn.Module:
        return self.norm2

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        return self.norm1(self.embed(input_ids))

    def temporal_belief_from_token_latent(self, token_latent: Tensor) -> Tensor:
        x = token_latent
        for block in self.blocks:
            x = block(x)
        return self.norm2(x)

    def logits_from_features(self, features: Tensor) -> Tensor:
        # Standard renderer features are cat(input_latent, belief); the nano
        # readout is belief-only, so slice the contextual half.
        belief = features[..., -self.model_dim:]
        raw = self._raw_logits(belief)
        return 15 * raw * (raw.square() + 15**2).rsqrt()

    def policy_logits(self, input_ids: Tensor) -> Tensor:
        """Teacher-forced vocab logits — exactly the pretraining readout."""
        token_latent = self.embed_tokens(input_ids)
        belief = self.temporal_belief_from_token_latent(token_latent)
        return self.logits_from_features(torch.cat((token_latent, belief), dim=-1))

    @staticmethod
    def _rotate_step(x_B1HD: Tensor, position: "int | Tensor",
                     angular_freq: Tensor) -> Tensor:
        """Half-truncate RoPE for one step at an absolute cache position.

        The dense ``Rotary`` derives positions from ``x.size(1)``; here the
        position is the cache slot. The zeroed frequency half gives cos=1 and
        sin=0, so the generic rotation is an exact identity there, matching
        the dense path.
        """
        theta = position * angular_freq
        cos = theta.cos()[None, None, None, :]
        sin = theta.sin()[None, None, None, :]
        x1, x2 = x_B1HD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), -1).type_as(x_B1HD)

    def _attention_step(
        self,
        attention: nanogpt_mini_model.CausalSelfAttention,
        x: Tensor,
        cache: tuple[Tensor, Tensor],
        position: "int | Tensor",
        key_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """One-token attention step; branch structure mirrors fresh_lejepa.

        With ``key_mask`` (bool, (cache_length,), True = attend) the step
        attends over the WHOLE preallocated cache under the mask instead of
        narrowing to ``:position + 1`` — shapes independent of ``position``.
        ``position`` must be a 0-dim tensor in that mode, and masked cache
        slots must hold finite values. A 2-D (batch, keys) ``key_mask``
        keeps the narrow shapes but masks each row's left-padded prefix.
        """
        batch, _, dim = x.shape
        num_heads, head_dim = attention.num_heads, attention.head_dim
        q = attention.q(x).view(batch, 1, num_heads, head_dim)
        k = attention.k(x).view(batch, 1, num_heads, head_dim)
        v = attention.v(x).view(batch, 1, num_heads, head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        angular_freq = attention.rotary.angular_freq
        q = self._rotate_step(q, position, angular_freq).transpose(1, 2)
        k = self._rotate_step(k, position, angular_freq).transpose(1, 2)
        v = v.transpose(1, 2)
        # Autocast puts rms_norm on the fp32 list and linear on the bf16 one,
        # so k arrives fp32 and v bf16 while the cache holds one dtype: eager
        # index_copy_ rejects the mismatch outright. Cast as _attention_prefill
        # does. Every branch below reads its SDPA keys back out of the cache,
        # so rounding here is what the attention already sees. q is cast for
        # the same reason one step later — it never enters the cache, but eager
        # SDPA demands all three agree, and autocast casts it to the cache
        # dtype anyway (scaled_dot_product_attention is on the bf16 list).
        q = q.to(cache[0].dtype)
        k = k.to(cache[0].dtype)
        v = v.to(cache[1].dtype)
        attn_mask = None
        if key_mask is not None and key_mask.dim() == 2:
            if torch.is_tensor(position):
                index = position.reshape(1)
                position_length = key_mask.shape[-1]
                cache[0].index_copy_(2, index, k)
                cache[1].index_copy_(2, index, v)
                prefix_k = torch.narrow(cache[0], 2, 0, position_length)
                prefix_v = torch.narrow(cache[1], 2, 0, position_length)
                attn_mask = torch.narrow(
                    key_mask, 1, 0, position_length
                )[:, None, None, :]
            else:
                position_length = position + 1
                cache[0][:, :, position:position_length].copy_(k)
                cache[1][:, :, position:position_length].copy_(v)
                prefix_k = cache[0][:, :, :position_length]
                prefix_v = cache[1][:, :, :position_length]
                attn_mask = key_mask[:, None, None, :position_length]
        elif key_mask is not None:
            if not torch.is_tensor(position):
                raise ValueError("key_mask stepping requires a 0-dim tensor position")
            index = position.reshape(1)
            cache[0].index_copy_(2, index, k)
            cache[1].index_copy_(2, index, v)
            prefix_k, prefix_v = cache[0], cache[1]
            attn_mask = key_mask[None, None, None, :]
        elif torch.is_tensor(position):
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
            q, prefix_k, prefix_v, attn_mask=attn_mask, is_causal=False, scale=0.12
        )
        y = y.transpose(1, 2).contiguous().view(batch, 1, dim)
        return attention.proj(y), cache

    def _block_step(
        self,
        block: nanogpt_mini_model.Block,
        x: Tensor,
        x0: Tensor,
        cache: tuple[Tensor, Tensor],
        position: "int | Tensor",
        key_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        del x0  # no encoder/decoder skip input in the nano trunk
        attn, cache = self._attention_step(
            block.attn, block.norm1(x), cache, position, key_mask
        )
        x = x + attn
        x = x + block.mlp(block.norm2(x))
        return x, cache

    @staticmethod
    def _prefill_attention_mask(key_valid: Tensor) -> Tensor:
        """Causal mask that excludes left padding without NaN pad queries."""
        length = key_valid.size(1)
        causal = torch.ones(
            (length, length), dtype=torch.bool, device=key_valid.device
        ).tril_()
        mask = causal[None, None] & key_valid[:, None, None, :]
        # A left-pad query has no semantically meaningful output, but a fully
        # masked SDPA row produces NaN that contaminates deeper-layer K/V.
        # Match stepwise prefill by letting such queries attend their causal
        # padded prefix; real queries still exclude every padded key.
        return torch.where(
            key_valid[:, None, :, None], mask, causal[None, None]
        )

    def _attention_prefill(
        self,
        attention: nanogpt_mini_model.CausalSelfAttention,
        x: Tensor,
        cache: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        """Evaluate a prefix densely and populate its generation cache."""
        batch, length, dim = x.shape
        num_heads, head_dim = attention.num_heads, attention.head_dim
        q = attention.q(x).view(batch, length, num_heads, head_dim)
        k = attention.k(x).view(batch, length, num_heads, head_dim)
        v = attention.v(x).view(batch, length, num_heads, head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        # The dense Rotary derives positions 0..length-1 from x.size(1) —
        # exactly the absolute cache slots the prefix occupies.
        q, k = attention.rotary(q), attention.rotary(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cache[0][:, :, :length].copy_(k.to(cache[0].dtype))
        cache[1][:, :, :length].copy_(v.to(cache[1].dtype))
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            is_causal=attention_mask is None, scale=0.12,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, dim)
        return attention.proj(y)

    def _block_prefill(
        self,
        block: nanogpt_mini_model.Block,
        x: Tensor,
        cache: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        x = x + self._attention_prefill(
            block.attn, block.norm1(x), cache, attention_mask
        )
        return x + block.mlp(block.norm2(x))

    def prefill_belief(
        self,
        token_latent: Tensor,
        caches: list[tuple[Tensor, Tensor]],
        key_valid: Tensor | None = None,
    ) -> Tensor:
        """Compute a whole deterministic prefix and fill every layer cache."""
        attention_mask = (
            self._prefill_attention_mask(key_valid)
            if key_valid is not None
            else None
        )
        x = token_latent
        for i, block in enumerate(self.blocks):
            x = self._block_prefill(block, x, caches[i], attention_mask)
        return self.norm2(x)

    def make_generation_cache(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Allocate KV storage independently of the master-weight dtype."""
        cache_dtype = self.tok_emb.weight.dtype if dtype is None else dtype
        caches = []
        for block in self.blocks:
            attention = block.attn
            shape = (batch_size, attention.num_heads, max_length, attention.head_dim)
            caches.append(
                (
                    torch.empty(shape, device=device, dtype=cache_dtype),
                    torch.empty(shape, device=device, dtype=cache_dtype),
                )
            )
        return caches


class NanoGPTBackbone(_NanoPostrainingMixin, nanogpt_mini_model.GPT):
    """nanogpt-mini baseline (untied linear head) with the post-training interface."""

    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 mlp_hidden: "int | None" = None):
        super().__init__(vocab_size, num_layers, model_dim, mlp_hidden)
        self._finalize_construction(num_layers, model_dim)

    def renderer_parameters(self):
        """Readout parameters (the fresh path's policy-probe analogue)."""
        yield from self.proj.parameters()

    def _raw_logits(self, belief: Tensor) -> Tensor:
        return self.proj(belief).float()


class NanoTiedDotBackbone(_NanoPostrainingMixin, nanogpt_mini_model.TiedDotGPT):
    """nanogpt-mini tied attached dot-product readout variant."""

    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 mlp_hidden: "int | None" = None):
        super().__init__(vocab_size, num_layers, model_dim, mlp_hidden)
        self._finalize_construction(num_layers, model_dim)

    def renderer_parameters(self):
        yield self.readout_scale
        yield self.readout_bias

    def _raw_logits(self, belief: Tensor) -> Tensor:
        # attached codebook: gradients flow into embed through the readout
        codebook = F.rms_norm(self.embed.weight, (self.embed.embedding_dim,))
        return (
            belief @ codebook.type_as(belief).t()
        ).float() * self.readout_scale + self.readout_bias

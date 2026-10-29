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

The stepwise/prefill attention mirrors pretraining/fresh_lejepa/fresh_lejepa_train.py's
``_attention_step``/``_attention_prefill`` branch-for-branch (int position
slice-copy; 0-dim tensor position ``index_copy_`` + narrow; 1-D full-cache
key mask; 2-D per-row left-pad key mask), with the nano deltas: separate
q/k/v Linears, unweighted ``F.rms_norm`` on q/k, half-truncate RoPE applied
at the absolute cache position, SDPA scale 0.12, no GQA, no q_gain.
Padded CUDA prefill replaces the quadratic left-pad mask with native bf16/fp16
varlen Flash; CPU retains the exact dense-mask oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from postraining.decode_attention import ranged_decode_attention
from pretraining.nanogpt_mini import nanogpt_mini_model

NANO_DEFAULT_MODEL_CONFIG = {
    "vocab_size": 1024,
    "num_layers": 6,
    "model_dim": 512,
    "mlp_hidden": 2048,
}


@dataclass(frozen=True)
class _PrefillAttentionLayout:
    """One batch-wide attention layout, shared by every dense layer.

    CPU keeps the exact dense oracle mask.  CUDA stores only the O(B)
    cumulative-sequence metadata consumed by native varlen Flash;
    in particular, it never materializes the old O(BL²) boolean mask.
    ``max_seqlen`` is the padded width, a static upper bound accepted by the
    Flash kernel, so deriving the layout does not synchronize for a CUDA
    ``lengths.max().item()``.
    """

    attention_mask: Tensor | None = None
    cu_seqlens: Tensor | None = None
    max_seqlen: int | None = None

    @property
    def is_varlen(self) -> bool:
        return self.cu_seqlens is not None


def rational_softcap(raw: Tensor, cap: float) -> Tensor:
    """The pretraining logit softcap, ``cap * x / sqrt(x^2 + cap^2)``."""
    return cap * raw * (raw.square() + cap**2).rsqrt()


def _shifted_softcapped_logits(
    raw: Tensor, cap: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One shared capped row for normalization and selected-token scores.

    Keeping the row maximum separate avoids adding a tiny normalization
    correction to a large capped score. No clipping or nonfinite repair is
    applied: invalid logits remain invalid and must fail downstream checks.
    """
    compute_dtype = torch.promote_types(raw.dtype, torch.float32)
    capped = rational_softcap(raw.to(compute_dtype), cap)
    row_max = capped.amax(-1)
    shifted = capped - row_max[:, None]
    log_norm = shifted.logsumexp(-1)
    return capped, row_max, shifted, log_norm


class SoftcappedTargetLogprobs(torch.autograd.Function):
    """Target log-probabilities of rationally softcapped logits.

    Only one scalar per row of a (rows, vocab) readout is consumed, but the
    composed autograd graph of softcap -> fp32 log-softmax -> gather keeps
    vocabulary-wide fp32 tensors alive and runs its backward as zeros +
    scatter-add + log-softmax backward + softcap backward: several fp32
    passes over (rows, vocab). This function saves only the raw logits, in
    the readout GEMM's own dtype, and two (rows,) normalization terms. Its
    backward is a single elementwise pass that reforms the softmax from them::

        d raw = (onehot(target) - softmax(capped)) * g * cap'(raw)
        cap'(x) = cap^3 / (x^2 + cap^2)^(3/2)

    and writes the gradient straight back in the GEMM's dtype. The softcap
    and the softmax are evaluated in at least fp32, as in
    ``logits_from_features``.
    """

    @staticmethod
    def forward(ctx, raw: Tensor, targets: Tensor, cap: float) -> Tensor:
        if raw.ndim != 2 or targets.shape != raw.shape[:1]:
            raise ValueError("expected raw [rows, vocab] and targets [rows]")
        _, row_max, shifted, log_norm = _shifted_softcapped_logits(raw, cap)
        selected = shifted.gather(-1, targets[:, None]).squeeze(-1)
        ctx.save_for_backward(raw, targets, row_max, log_norm)
        ctx.cap = cap
        return selected - log_norm

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad: Tensor):
        # Saved normalization terms hide their dependence on raw from a
        # create_graph backward, so a second derivative must refuse.
        raw, targets, row_max, log_norm = ctx.saved_tensors
        cap = ctx.cap
        x = raw.to(log_norm.dtype)
        inverse_norm = (x.square() + cap**2).rsqrt()
        probabilities = (
            (cap * x * inverse_norm - row_max[:, None]) - log_norm[:, None]
        ).exp()
        vocabulary = torch.arange(raw.size(-1), device=raw.device)
        is_target = vocabulary[None, :] == targets[:, None]
        grad_capped = grad[:, None] * (is_target.to(x.dtype) - probabilities)
        grad_raw = grad_capped * cap**3 * inverse_norm**3
        return grad_raw.to(raw.dtype), None, None


def policy_centered_token_values(
    logits: Tensor, targets: Tensor, directions: Tensor, action_latents: Tensor
) -> Tensor:
    """Q(s, a_t) - sum_a pi(a | s) Q(s, a) for Q(s, a) = c + u . x(a).

    ``logits`` are the policy's (rows, vocab) logits, ``directions`` the
    critic's (rows, dim) u, ``action_latents`` its (vocab, dim) x. The state
    term c cancels: A = u . (x(a_t) - sum_a pi(a) x(a)), the critic's own
    offset (``policy_action_offsets_``) read by u. The offset is one GEMM
    of pi - onehot(a_t), so as pi concentrates on a_t, A goes to zero with
    its error rather than leaving the error of a separately rounded mean.
    In a TF32 emulation (d 512, V 50,304) its relative error matched the
    whole-vocabulary score GEMM's, about 3e-4 from pi(a_t) = 0.05 to 0.999,
    without that GEMM's (rows, vocab) scores.
    A detached statistic of the old policy; it has no gradient.
    """
    if logits.ndim != 2 or targets.shape != logits.shape[:1]:
        raise ValueError("expected logits [rows, vocab] and targets [rows]")
    if action_latents.shape != (logits.size(-1), directions.size(-1)):
        raise ValueError("action latents must be [vocab, dim]")
    with torch.no_grad(), torch.autocast(
        device_type=logits.device.type, enabled=False
    ):
        offsets = policy_action_offsets_(logits.float().softmax(-1), targets, action_latents)
        return (directions.to(offsets.dtype) * offsets).sum(-1)


def policy_action_offsets_(probabilities: Tensor, targets: Tensor,
                           action_latents: Tensor) -> Tensor:
    """x(a_t) - sum_a pi(a | s) x(a) for the policy's (rows, vocab) fp32
    ``probabilities``, taken ``targets`` and the critic's (vocab, dim)
    action table: the offset that centres the token Q critic's fit
    (``value_model.policy_centered_q``). Detached, in their dtype.

    Formed whole, as -(pi - onehot(a_t)) @ x in one GEMM, so its rounding
    shrinks with 1 - pi(a_t) the way the actor's advantage does
    (``policy_centered_token_values``). Consumes ``probabilities``: they are
    overwritten in place, which keeps a second (rows, vocab) tensor out of
    every readout that calls this last.
    """
    if probabilities.ndim != 2 or targets.shape != probabilities.shape[:1]:
        raise ValueError("expected probabilities [rows, vocab] and targets [rows]")
    with torch.no_grad(), torch.autocast(device_type=probabilities.device.type, enabled=False):
        probabilities.scatter_add_(
            -1, targets[:, None], probabilities.new_full((targets.numel(), 1), -1.0))
        return -(probabilities @ action_latents.to(probabilities.dtype))


class SoftcappedTargetLogprobsEntropy(torch.autograd.Function):
    """:class:`SoftcappedTargetLogprobs` plus each row's policy entropy.

    The target log-probability is the same expression, so it is bit-for-bit
    the single-output function's. The entropy of the capped softmax,
    ``H = -sum(p * log p)``, avoids cancellation for peaked rows and costs
    one more vocabulary pass in each direction. Its gradient is::

        d H / d capped = -p * (log p + H)

    so the backward reforms ``p`` once, as the single-output function does,
    and adds this term before the shared softcap derivative. Only the raw
    logits and three (rows,) statistics are saved. All three custom readouts
    share the same selected-logit arithmetic; their target outputs must agree
    bit-for-bit when evaluated by the same numerical implementation.
    """

    @staticmethod
    def forward(ctx, raw: Tensor, targets: Tensor, cap: float):
        if raw.ndim != 2 or targets.shape != raw.shape[:1]:
            raise ValueError("expected raw [rows, vocab] and targets [rows]")
        _, row_max, shifted, log_norm = _shifted_softcapped_logits(raw, cap)
        log_probabilities = shifted - log_norm[:, None]
        entropy = -(log_probabilities.exp() * log_probabilities).sum(-1)
        selected = shifted.gather(-1, targets[:, None]).squeeze(-1)
        ctx.save_for_backward(raw, targets, row_max, log_norm, entropy)
        ctx.cap = cap
        return selected - log_norm, entropy

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_logprob: Tensor, grad_entropy: Tensor):
        return SoftcappedTargetLogprobsEntropy.raw_gradient(
            ctx, grad_logprob, grad_entropy
        ), None, None

    @staticmethod
    def raw_gradient(ctx, grad_logprob: Tensor, grad_entropy: Tensor) -> Tensor:
        raw, targets, row_max, log_norm, entropy = ctx.saved_tensors
        cap = ctx.cap
        x = raw.to(log_norm.dtype)
        inverse_norm = (x.square() + cap**2).rsqrt()
        log_probabilities = (
            (cap * x * inverse_norm - row_max[:, None]) - log_norm[:, None]
        )
        probabilities = log_probabilities.exp()
        vocabulary = torch.arange(raw.size(-1), device=raw.device)
        is_target = vocabulary[None, :] == targets[:, None]
        grad_capped = grad_logprob[:, None] * (
            is_target.to(x.dtype) - probabilities
        ) - grad_entropy[:, None] * probabilities * (
            log_probabilities + entropy[:, None]
        )
        grad_raw = grad_capped * cap**3 * inverse_norm**3
        return grad_raw.to(raw.dtype)


class SoftcappedTargetLogprobsEntropyAdvantages(torch.autograd.Function):
    """:class:`SoftcappedTargetLogprobsEntropy` plus the token Q critic's
    policy-centered advantage (:func:`policy_centered_token_values`) and the
    taken token's action-row offset (:func:`policy_action_offsets_`) under
    the same capped softmax.

    The advantage is formed inside the forward from the capped logits the
    log-probabilities already hold, so the readout GEMM runs once; it is a
    detached statistic with no gradient. Owning it here also keeps the
    softcap inside one traced subgraph: computed outside the function, a
    dynamic-shape compile shares the cap's symbolic float across the
    function boundary and fails to build the graph.
    """

    @staticmethod
    def forward(
        ctx,
        raw: Tensor,
        targets: Tensor,
        cap: float,
        directions: Tensor,
        action_latents: Tensor,
    ):
        if raw.ndim != 2 or targets.shape != raw.shape[:1]:
            raise ValueError("expected raw [rows, vocab] and targets [rows]")
        capped, row_max, shifted, log_norm = _shifted_softcapped_logits(raw, cap)
        selected = shifted.gather(-1, targets[:, None]).squeeze(-1)
        log_probabilities = shifted - log_norm[:, None]
        probabilities = log_probabilities.exp()
        entropy = -(probabilities * log_probabilities).sum(-1)
        # The offsets consume the probabilities, and the advantage is u
        # read against them (``policy_centered_token_values``): one GEMM,
        # one (slots, vocab) tensor.
        offsets = policy_action_offsets_(probabilities, targets, action_latents)
        advantages = (directions.to(offsets.dtype) * offsets).sum(-1)
        ctx.save_for_backward(raw, targets, row_max, log_norm, entropy)
        ctx.cap = cap
        ctx.mark_non_differentiable(advantages, offsets)
        return (
            selected - log_norm, entropy, advantages, offsets
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_logprob: Tensor, grad_entropy: Tensor, _, _offsets):
        return SoftcappedTargetLogprobsEntropy.raw_gradient(
            ctx, grad_logprob, grad_entropy
        ), None, None, None, None


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
        return rational_softcap(self._raw_logits(belief).float(), self.logit_softcap)

    def target_logprobs_from_features(
        self, features: Tensor, targets: Tensor
    ) -> Tensor:
        """``logits_from_features(features).log_softmax(-1)`` at ``targets``.

        The same value without materializing the vocabulary-wide fp32 logits
        for autograd: see :class:`SoftcappedTargetLogprobs`.
        """
        belief = features[..., -self.model_dim:]
        return SoftcappedTargetLogprobs.apply(
            self._raw_logits(belief), targets, self.logit_softcap
        )

    def target_logprobs_and_entropy_from_features(
        self, features: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``target_logprobs_from_features`` and the full policy entropy:
        see :class:`SoftcappedTargetLogprobsEntropy`."""
        belief = features[..., -self.model_dim:]
        return SoftcappedTargetLogprobsEntropy.apply(
            self._raw_logits(belief), targets, self.logit_softcap
        )

    def target_logprobs_entropy_and_advantages_from_features(
        self,
        features: Tensor,
        targets: Tensor,
        directions: Tensor,
        action_latents: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``target_logprobs_and_entropy_from_features``, the policy-centered
        token Q and the taken token's action-row offset, from one readout GEMM: see
        :class:`SoftcappedTargetLogprobsEntropyAdvantages`."""
        belief = features[..., -self.model_dim:]
        return SoftcappedTargetLogprobsEntropyAdvantages.apply(
            self._raw_logits(belief), targets, self.logit_softcap,
            directions, action_latents,
        )

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
        block_mask: BlockMask | None = None,
        key_starts: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """One-token attention step; branch structure mirrors fresh_lejepa.

        With ``key_mask`` (bool, (cache_length,), True = attend) the step
        attends over the WHOLE preallocated cache under the mask instead of
        narrowing to ``:position + 1`` — shapes independent of ``position``.
        ``position`` must be a 0-dim tensor in that mode, and masked cache
        slots must hold finite values. A 2-D (batch, keys) ``key_mask``
        keeps the narrow shapes but masks each row's left-padded prefix.

        ``block_mask`` replaces a 2-D ``key_mask`` when every row's attendable
        keys form one contiguous range, which is always true here: left padding
        is a prefix and the write head is a suffix bound. It is not a
        convenience — handing SDPA any ``attn_mask`` at all disqualifies the
        fused backends and drops this step onto the memory-efficient kernel,
        which the v25 profile measured at 32% of pool device time with 12672 of
        its 12684 calls coming from here. The same information as a BlockMask
        reaches flex decoding, measured 45-64% faster across L=128..2048 on the
        production shapes. One mask serves all layers, so it is built by the
        caller, not here.

        ``key_starts`` (int64, (batch,)) is the same range without a table:
        row ``b`` attends ``[key_starts[b], position]`` over the whole cache,
        and ``ranged_decode_attention`` reads only those slots -- not the
        left padding before them nor the cache past the write head.
        """
        batch, _, dim = x.shape
        num_heads, head_dim = attention.num_heads, attention.head_dim
        q = attention.q(x).view(batch, 1, num_heads, head_dim)
        k = attention.k(x).view(batch, 1, num_heads, head_dim)
        v = attention.v(x).view(batch, 1, num_heads, head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        if getattr(attention, "use_rope", True):
            angular_freq = attention.rotary.angular_freq
            q = self._rotate_step(q, position, angular_freq)
            k = self._rotate_step(k, position, angular_freq)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
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
        if key_starts is not None:
            if not torch.is_tensor(position):
                raise ValueError("key_starts stepping requires a 0-dim tensor position")
            if key_mask is not None or block_mask is not None:
                raise ValueError("key_starts excludes key_mask and block_mask")
            index = position.reshape(1)
            cache[0].index_copy_(2, index, k)
            cache[1].index_copy_(2, index, v)
            y = ranged_decode_attention(
                q.squeeze(2), cache[0], cache[1], key_starts, position, 0.12
            )
            return attention.proj(y.reshape(batch, 1, dim)), cache
        if block_mask is not None:
            if not torch.is_tensor(position):
                raise ValueError("block_mask stepping requires a 0-dim tensor position")
            if key_mask is not None:
                # The block table already carries the live range; honouring a
                # boolean mask too would silently pick one and drop the other.
                raise ValueError("block_mask and key_mask are mutually exclusive")
            index = position.reshape(1)
            cache[0].index_copy_(2, index, k)
            cache[1].index_copy_(2, index, v)
            y = flex_attention(q, cache[0], cache[1], block_mask=block_mask, scale=0.12)
            y = y.transpose(1, 2).contiguous().view(batch, 1, dim)
            return attention.proj(y), cache
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
        block_mask: BlockMask | None = None,
        key_starts: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        del x0  # no encoder/decoder skip input in the nano trunk
        attn, cache = self._attention_step(
            block.attn, block.norm1(x), cache, position, key_mask, block_mask,
            key_starts,
        )
        x = x + attn
        x = x + block.mlp(block.norm2(x))
        return x, cache

    @staticmethod
    def _write_paged_cache(
        cache: Tensor,
        value: Tensor,
        addresses: Tensor,
    ) -> None:
        cache.index_copy_(
            2, addresses, value.permute(2, 1, 0, 3).contiguous()
        )

    def _attention_paged_step(
        self,
        attention: nanogpt_mini_model.CausalSelfAttention,
        x: Tensor,
        cache: tuple[Tensor, Tensor],
        positions: Tensor,
        block_mask: BlockMask,
        cache_addresses: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """One-token attention for independently advancing packed cache lanes."""
        batch, _, dim = x.shape
        num_heads, head_dim = attention.num_heads, attention.head_dim
        q = attention.q(x).view(batch, 1, num_heads, head_dim)
        k = attention.k(x).view(batch, 1, num_heads, head_dim)
        value = attention.v(x).view(batch, 1, num_heads, head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        if getattr(attention, "use_rope", True):
            angular_freq = attention.rotary.angular_freq
            theta = positions[:, None].to(angular_freq.dtype) * angular_freq[None]
            cos, sin = theta.cos()[:, None, None], theta.sin()[:, None, None]

            def rotate(tensor: Tensor) -> Tensor:
                first, second = tensor.float().chunk(2, dim=-1)
                return torch.cat(
                    (first * cos + second * sin, first * (-sin) + second * cos),
                    dim=-1,
                ).type_as(tensor)

            q = rotate(q)
            k = rotate(k)
        q = q.transpose(1, 2).to(cache[0].dtype)
        k = k.transpose(1, 2).to(cache[0].dtype)
        value = value.transpose(1, 2).to(cache[1].dtype)
        self._write_paged_cache(cache[0], k, cache_addresses)
        self._write_paged_cache(cache[1], value, cache_addresses)
        y = flex_attention(
            q,
            cache[0],
            cache[1],
            block_mask=block_mask,
            scale=0.12,
        )
        y = y.transpose(1, 2).contiguous().view(batch, 1, dim)
        return attention.proj(y), cache

    def _block_paged_step(
        self,
        block: nanogpt_mini_model.Block,
        x: Tensor,
        x0: Tensor,
        cache: tuple[Tensor, Tensor],
        positions: Tensor,
        block_mask: BlockMask,
        cache_addresses: Tensor,
        lane_rows: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # lane_rows addresses recurrent arenas; a pure-KV block has none.
        del x0, lane_rows
        attn, cache = self._attention_paged_step(
            block.attn,
            block.norm1(x),
            cache,
            positions,
            block_mask,
            cache_addresses,
        )
        x = x + attn
        return x + block.mlp(block.norm2(x)), cache

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

    @classmethod
    def _prefill_attention_layout(
        cls,
        token_latent: Tensor,
        key_valid: Tensor | None,
    ) -> _PrefillAttentionLayout:
        """Build dense-CPU or packed-CUDA metadata once for the whole trunk."""

        if key_valid is None:
            return _PrefillAttentionLayout()
        if (
            key_valid.dtype != torch.bool
            or key_valid.ndim != 2
            or key_valid.shape != token_latent.shape[:2]
        ):
            raise ValueError("key_valid must be boolean [batch, length]")
        if key_valid.device != token_latent.device:
            raise ValueError("key_valid and token_latent must share a device")
        if token_latent.device.type != "cuda":
            return _PrefillAttentionLayout(
                attention_mask=cls._prefill_attention_mask(key_valid)
            )

        # Preserve the original contiguous [B, L] token storage and describe
        # each row as exactly two causal sequences: its invalid left prefix
        # and its valid suffix.  The former reproduces the dense oracle's
        # finite padded-query states; the latter cannot see padded keys.  This
        # fixed 2B-sequence representation is fully packed (cu[-1] == B*L),
        # handles the longest row through a zero-length pad sequence, and
        # avoids both dynamic-shape ``nonzero`` synchronization and gather /
        # scatter kernels.  Callers promise left padding; CPU remains the
        # general arbitrary-mask oracle.
        valid_lengths = key_valid.sum(dim=1, dtype=torch.int32)
        pad_lengths = token_latent.shape[1] - valid_lengths
        segment_lengths = torch.stack(
            (pad_lengths, valid_lengths), dim=1
        ).flatten()
        cu_seqlens = F.pad(
            segment_lengths.cumsum(dim=0, dtype=torch.int32), (1, 0)
        )
        return _PrefillAttentionLayout(
            cu_seqlens=cu_seqlens,
            max_seqlen=token_latent.shape[1],
        )

    def _attention_prefill(
        self,
        attention: nanogpt_mini_model.CausalSelfAttention,
        x: Tensor,
        cache: tuple[Tensor, Tensor],
        layout: _PrefillAttentionLayout,
    ) -> Tensor:
        """Evaluate a prefix and populate its generation cache.

        A padded CUDA batch is true packed causal attention: every physical
        row contributes separate pad-prefix and valid-suffix sequences to
        native varlen Flash.  The split reproduces even the dense oracle's
        padded-query outputs while making it structurally impossible for a
        valid query to see a padded key.  Q/K/V storage and physical cache
        addresses remain unchanged.
        """
        batch, length, dim = x.shape
        num_heads, head_dim = attention.num_heads, attention.head_dim
        q = attention.q(x).view(batch, length, num_heads, head_dim)
        k = attention.k(x).view(batch, length, num_heads, head_dim)
        v = attention.v(x).view(batch, length, num_heads, head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        # The dense Rotary derives positions 0..length-1 from x.size(1) —
        # exactly the absolute cache slots the prefix occupies.
        if getattr(attention, "use_rope", True):
            q, k = attention.rotary(q), attention.rotary(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cache[0][:, :, :length].copy_(k.to(cache[0].dtype))
        cache[1][:, :, :length].copy_(v.to(cache[1].dtype))
        if layout.is_varlen:
            if q.device.type != "cuda":
                raise RuntimeError("packed prefill requires CUDA")
            if v.dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError(
                    "packed CUDA prefill requires fp16 or bf16 Q/K/V; run "
                    "generation under CUDA autocast"
                )
            try:
                from torch.nn.attention.varlen import varlen_attn
            except Exception as error:
                raise RuntimeError(
                    "native varlen Flash attention is required for padded "
                    "CUDA prefill"
                ) from error
            assert layout.cu_seqlens is not None
            assert layout.max_seqlen is not None
            # Autocast keeps linear V in bf16/fp16 but rms_norm Q/K in fp32.
            # SDPA would autocast all three internally; native varlen requires
            # an explicit common dtype before dispatch.
            q_tokens = q.transpose(1, 2).reshape(-1, num_heads, head_dim)
            k_tokens = k.transpose(1, 2).reshape(-1, num_heads, head_dim)
            v_tokens = v.transpose(1, 2).reshape(-1, num_heads, head_dim)
            packed_y = varlen_attn(
                q_tokens.to(v.dtype),
                k_tokens.to(v.dtype),
                v_tokens,
                layout.cu_seqlens,
                layout.cu_seqlens,
                layout.max_seqlen,
                layout.max_seqlen,
                scale=0.12,
                window_size=(-1, 0),
            )
            y = packed_y.view(batch, length, num_heads, head_dim).reshape(
                batch, length, dim
            )
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=layout.attention_mask,
                is_causal=layout.attention_mask is None,
                scale=0.12,
            )
            y = y.transpose(1, 2).contiguous().view(batch, length, dim)
        return attention.proj(y)

    def _block_prefill(
        self,
        block: nanogpt_mini_model.Block,
        x: Tensor,
        cache: tuple[Tensor, Tensor],
        layout: _PrefillAttentionLayout,
    ) -> Tensor:
        x = x + self._attention_prefill(
            block.attn, block.norm1(x), cache, layout
        )
        return x + block.mlp(block.norm2(x))

    def prefill_belief(
        self,
        token_latent: Tensor,
        caches: list[tuple[Tensor, Tensor]],
        key_valid: Tensor | None = None,
    ) -> Tensor:
        """Compute a whole deterministic prefix and fill every layer cache."""
        layout = self._prefill_attention_layout(token_latent, key_valid)
        x = token_latent
        for i, block in enumerate(self.blocks):
            x = self._block_prefill(block, x, caches[i], layout)
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
        return self.proj(belief)


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

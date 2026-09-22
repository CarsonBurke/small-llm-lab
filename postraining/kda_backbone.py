"""Post-training backbone adapter for the KDA-mixer nanogpt-mini architecture.

``NanoKDABackbone`` subclasses the import-safe pretraining model
(``nanogpt_mini_kda_model.KDAGPT``) so construction and state-dict keys are
identical to the training script's exports — a ``*_kda_*`` ``final_model.pt``
strict-loads with zero key mapping. On top of the pretrained trunk it adds the
same interface ``nano_backbone`` gives the dense models, with one structural
difference: caches are heterogeneous per layer.

- Dense (MHA) layers keep the nano KV pair ``(k_cache, v_cache)`` and reuse
  ``_NanoPostrainingMixin``'s step/prefill attention verbatim (the KDA
  script's ``CausalSelfAttention`` matches nano's attribute contract:
  ``q/k/v/proj`` Linears, ``rotary``, scale 0.12, unweighted q/k rms_norm).
- KDA layers carry ``(conv_q, conv_k, conv_v, state)``: three ``[B, D, W]``
  causal-conv windows and one fp32 ``[B, H, Dv, Dk]`` delta-rule state
  (value-major, FLA's ``state_v_first=True`` layout). The state is always
  fp32 regardless of the requested cache dtype — it is an accumulating
  recurrence, and the prefill kernel computes it in fp32; storing it rounded
  would make decode drift from a dense re-evaluation of the same prefix.

Position arguments, key masks, and block masks are dense-attention concepts;
KDA steps ignore them (a recurrent state has no addressable history). That is
correct for every decode schedule this stack runs: each lockstep decode step
feeds one real token per live lane, and lanes that already finished keep
stepping but their outputs — and therefore their state corruption — are never
read. Left-padded prefill is handled inside the KDA mixer by zeroing conv
inputs at padded positions, which keeps the recurrent state exactly zero
until each row's first real token (see ``KimiDeltaAttention.forward``).

The paged/continuous-refill decode path treats KDA layers as lane-indexed
recurrent arenas: page tables, block masks, and token addresses are
KV-address machinery that recurrent layers never touch. Each lane owns one
row of conv windows and state; admission overwrites the row wholesale, the
decode step gathers, mutates, and scatters it by ``lane_rows``, and dead
rows are redirected to scratch lanes the same way dead KV writes go to
scratch pages (see ``LatentThoughtModel.paged_step``).
"""

from __future__ import annotations

import math
import os

import torch
from torch import Tensor

# fla dispatches the KDA chunked backward to its TileLang backend by default,
# and in this environment that backend cannot run: TileLang vendors its own
# TVM while tvm_ffi is also installed standalone, so both register the FFI
# TypeAttr ``__ffi_repr__`` for the same TVM type index. The first backward
# then dies with "TypeAttr `__ffi_repr__` is already registered for type
# index 132" and the process deadlocks unwinding it -- a hang, not a clean
# crash. fla's Triton KDA backward is its reference implementation and is
# what every KDA run in this repository already uses; the pretraining
# ablations and GPU tests all pin FLA_TILELANG=0 the same way. Forward-only
# paths never notice, which is why this only ever bites training.
#
# fla caches BaseBackend.can_use() per class, and that cache is populated
# when ``fla.ops.kda.backends`` is imported, so this must be set before the
# first KDA op runs. Importing this module is that point: it is the only way
# a KDA checkpoint enters the post-training stack.
os.environ.setdefault("FLA_TILELANG", "0")


def _assert_kda_backward_is_usable() -> None:
    """Fail loudly if the unusable TileLang KDA backend won the dispatch.

    Setting the environment variable above is only effective while fla's
    cached capability check has not run yet. If some earlier import already
    resolved it the other way, the run would otherwise proceed to its first
    backward and hang there instead of saying why.
    """
    from fla.ops.kda.backends.tilelang import KDATileLangBackend

    if KDATileLangBackend.can_use():
        raise RuntimeError(
            "fla resolved the TileLang KDA backend, whose backward cannot "
            "run here (duplicate TVM FFI __ffi_repr__ registration). "
            "FLA_TILELANG=0 must be set before fla.ops.kda.backends is "
            f"first imported; it is currently "
            f"{os.environ.get('FLA_TILELANG')!r}."
        )

from pretraining.nanogpt_mini import nanogpt_mini_kda_model as kda_model
from postraining.nano_backbone import _NanoPostrainingMixin


class NanoKDABackbone(_NanoPostrainingMixin, kda_model.KDAGPT):
    """KDA-mixer nanogpt-mini trunk with the post-training interface."""

    def __init__(self, **model_config):
        _assert_kda_backward_is_usable()
        super().__init__(**model_config)
        self._finalize_construction(
            model_config["num_layers"], model_config["model_dim"]
        )

    def freeze_moe_routing_(self) -> int:
        """Freeze pretrained route selection while fine-tuning expert values.

        Updating both the router matrix and its changing hidden inputs would
        make post-training route drift needlessly abrupt. SFT/RL therefore
        keep the projection fixed while causal Quantile Balancing continues
        to track input-distribution drift as the experts and trunk adapt.
        """

        frozen = 0
        for block in self.blocks:
            if getattr(block, "use_moe", False):
                block.mlp.router_weight.requires_grad_(False)
                frozen += block.mlp.router_weight.numel()
        return frozen

    # ------------------------------------------------------------------ init
    def _apply_pretraining_init(self) -> None:
        """The training script's init recipe for fresh (critic) trunks.

        RNG *pairing* with the dense baseline is a pretraining-ablation
        concern and is deliberately dropped; what a fresh critic needs is the
        same distributions: normal(0.33**0.5/sqrt(fan)) projections, zeroed
        output projections, identity causal convs, A_log = 0 (safe-gate
        A = 1), dt_bias from the softplus-inverse of dt ~ LogUniform(1e-3,
        1e-1), unit norm gains, std-1 embedding.
        """

        def normal_weight(weight: Tensor) -> None:
            weight.normal_(std=0.33**0.5 / weight.size(-1) ** 0.5)

        def zero_linear(linear) -> None:
            linear.weight.zero_()
            if linear.bias is not None:
                linear.bias.zero_()

        with torch.no_grad():
            self.embed.weight.normal_()
            for block in self.blocks:
                attn = block.attn
                if block.use_kda:
                    for linear in (
                        attn.q_proj,
                        attn.k_proj,
                        attn.v_proj,
                        attn.f_a_proj,
                        attn.f_b_proj,
                        attn.b_proj,
                        *(
                            (attn.g_proj,)
                            if attn.full_rank_gate
                            else (attn.g_a_proj, attn.g_b_proj)
                        ),
                    ):
                        normal_weight(linear.weight)
                    for conv in (attn.q_conv1d, attn.k_conv1d, attn.v_conv1d):
                        conv.weight.zero_()
                        conv.weight[:, 0, -1] = 1
                    attn.A_log.zero_()
                    log_dt = torch.empty_like(attn.dt_bias).uniform_(
                        math.log(0.001), math.log(0.1)
                    )
                    dt = log_dt.exp().clamp(min=1e-4)
                    attn.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
                    attn.o_norm.weight.fill_(1)
                    zero_linear(attn.o_proj)
                else:
                    for linear in (attn.q, attn.k, attn.v):
                        normal_weight(linear.weight)
                        linear.bias.zero_()
                    zero_linear(attn.proj)
                block.norm1.gains.fill_(1)
                if block.use_mlp:
                    if block.use_moe:
                        moe = block.mlp
                        for weight in (
                            moe.router_weight,
                            moe.latent_down_proj.weight,
                            moe.expert_gate_up_weight,
                            moe.expert_down_weight,
                            moe.shared_expert.gate_up_proj.weight,
                        ):
                            normal_weight(weight)
                        zero_linear(moe.shared_expert.down_proj)
                        zero_linear(moe.latent_up_proj)
                        moe.routed_norm.weight.fill_(1)
                        moe.correction_bias.zero_()
                    else:
                        normal_weight(block.mlp.fc.weight)
                        block.mlp.fc.bias.zero_()
                        zero_linear(block.mlp.proj)
                    block.norm2.gains.fill_(1)
            zero_linear(self.proj)
            self.norm1.gains.fill_(1)
            self.norm2.gains.fill_(1)

    # ------------------------------------------------------------- readout
    def renderer_parameters(self):
        """Readout parameters (the fresh path's policy-probe analogue)."""
        yield from self.proj.parameters()

    def _raw_logits(self, belief: Tensor) -> Tensor:
        return self.proj(belief).float()

    # -------------------------------------------------------------- caches
    def make_generation_cache(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> list[tuple[Tensor, ...]]:
        """Heterogeneous per-layer decode caches (see module docstring)."""
        cache_dtype = self.tok_emb.weight.dtype if dtype is None else dtype
        caches: list[tuple[Tensor, ...]] = []
        for block in self.blocks:
            attention = block.attn
            if block.use_kda:
                conv_shape = (
                    batch_size,
                    attention.projection_size,
                    attention.conv_size,
                )
                state_shape = (
                    batch_size,
                    attention.num_heads,
                    attention.head_dim,
                    attention.head_dim,
                )
                caches.append(
                    (
                        torch.zeros(conv_shape, device=device, dtype=cache_dtype),
                        torch.zeros(conv_shape, device=device, dtype=cache_dtype),
                        torch.zeros(conv_shape, device=device, dtype=cache_dtype),
                        torch.zeros(
                            state_shape, device=device, dtype=torch.float32
                        ),
                    )
                )
            else:
                shape = (
                    batch_size,
                    attention.num_heads,
                    max_length,
                    attention.head_dim,
                )
                caches.append(
                    (
                        torch.empty(shape, device=device, dtype=cache_dtype),
                        torch.empty(shape, device=device, dtype=cache_dtype),
                    )
                )
        return caches

    # ---------------------------------------------------------------- step
    def _block_step(
        self,
        block: kda_model.Block,
        x: Tensor,
        x0: Tensor,
        cache: tuple[Tensor, ...],
        position: "int | Tensor",
        key_mask: Tensor | None = None,
        block_mask=None,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        del x0  # no encoder/decoder skip input in the nano trunks
        if block.use_kda:
            # position/key_mask/block_mask are KV addressing; a recurrent
            # step has no history to address. Caches mutate in place.
            x = x + block.attn.step(block.norm1(x), cache)
        else:
            attn, cache = self._attention_step(
                block.attn, block.norm1(x), cache, position, key_mask, block_mask
            )
            x = x + attn
        if block.use_mlp:
            x = x + block.mlp(block.norm2(x))
        return x, cache

    def _block_paged_step(
        self,
        block: kda_model.Block,
        x: Tensor,
        x0: Tensor,
        cache: tuple[Tensor, ...],
        position: Tensor,
        block_mask,
        cache_addresses: Tensor,
        lane_rows: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Advance independent lanes; recurrent layers index rows, not pages.

        A KDA layer's decode continuation is one row per lane — three conv
        windows and the fp32 delta-rule state — so the paged machinery
        reduces to gather → step → scatter on ``lane_rows``. The gathered
        copies are what ``attn.step`` mutates in place; the scatter is the
        single write back to the arena. ``lane_rows`` is duplicate-free by
        construction (``paged_step`` redirects dead rows to per-row scratch
        lanes and checks uniqueness on CPU), which keeps the ``index_copy_``
        deterministic — a repeated index would pick a winner
        nondeterministically.
        """
        del x0
        if block.use_kda:
            gathered = tuple(
                tensor.index_select(0, lane_rows) for tensor in cache
            )
            x = x + block.attn.step(block.norm1(x), gathered)
            for arena, updated in zip(cache, gathered, strict=True):
                arena.index_copy_(0, lane_rows, updated)
        else:
            attn, cache = self._attention_paged_step(
                block.attn,
                block.norm1(x),
                cache,
                position,
                block_mask,
                cache_addresses,
            )
            x = x + attn
        if block.use_mlp:
            x = x + block.mlp(block.norm2(x))
        return x, cache

    # -------------------------------------------------------------- prefill
    def prefill_belief(
        self,
        token_latent: Tensor,
        caches: list[tuple[Tensor, ...]],
        key_valid: Tensor | None = None,
    ) -> Tensor:
        """Compute a whole deterministic prefix and fill every layer cache."""
        layout = self._prefill_attention_layout(token_latent, key_valid)
        x = token_latent
        for i, block in enumerate(self.blocks):
            if block.use_kda:
                mixed, states = block.attn(
                    block.norm1(x),
                    key_valid=key_valid,
                    output_final_state=True,
                )
                for slot, fresh in zip(caches[i], states, strict=True):
                    if slot.shape != fresh.shape:
                        raise ValueError(
                            f"KDA prefill cache shape mismatch at layer {i}: "
                            f"cache {tuple(slot.shape)} vs prefill "
                            f"{tuple(fresh.shape)}"
                        )
                    slot.copy_(fresh.to(slot.dtype))
                x = x + mixed
                if block.use_mlp:
                    x = x + block.mlp(block.norm2(x))
            else:
                x = self._block_prefill(block, x, caches[i], layout)
        return self.norm2(x)

"""NanoGPT-mini residual/MLP backbone with one official GDN-2 mixer per layer.

Every layer has its own causal recurrent state. The GDN-2 read includes the
current token's update, as defined upstream; there is no token attention,
slot bank, or whole-network feedback pass. Ordinary training evaluates complete
sequences through the chunk kernel, while cached evaluation uses the official
short-sequence recurrent kernel and short-convolution caches.

The vendored GDN-2 code is licensed for research/evaluation under NVIDIA's
Source Code License-NC. See pretraining/gated_delta/vendor/LICENSE.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from fla.models.utils import Cache
from fla.modules import FusedRMSNormSwishGate, ShortConvolution

from pretraining.gated_delta.vendor.gdn2 import GatedDeltaNet2
from pretraining.nanogpt_mini import gated_delta_ops
from pretraining.nanogpt_mini.nanogpt_mini_model import Linear, MLP, RMSNorm


class CustomOpShortConvolution(ShortConvolution):
    """FLA's Triton short convolution as a compilable custom operator.

    Full training sequences without caches use the registered operator, so the
    packed-projection slice feeds the kernel directly and the surrounding graph
    stays whole. Cached and incremental evaluation keep FLA's own dispatch.
    """

    def forward(self, x: Tensor, residual: Tensor | None = None, mask: Tensor | None = None,
                cache: Tensor | None = None, output_final_state: bool = False,
                cu_seqlens: Tensor | None = None, chunk_indices: Tensor | None = None, **kwargs):
        if (cache is None and not output_final_state and residual is None and mask is None
                and cu_seqlens is None and chunk_indices is None and not kwargs and x.shape[1] > 1):
            weight = self.weight.view(self.hidden_size, self.kernel_size[0])
            return gated_delta_ops.causal_conv1d_silu(x, weight), None
        return super().forward(x, residual=residual, mask=mask, cache=cache,
                               output_final_state=output_final_state, cu_seqlens=cu_seqlens,
                               chunk_indices=chunk_indices, **kwargs)


class CustomOpRMSNormSwishGate(FusedRMSNormSwishGate):
    """FLA's gated output normalization as a compilable custom operator."""

    def forward(self, x: Tensor, g: Tensor, residual: Tensor | None = None,
                prenorm: bool = False, residual_in_fp32: bool = False) -> Tensor:
        if residual is None and not prenorm and not residual_in_fp32:
            return gated_delta_ops.rms_norm_swish_gate(x, g, self.weight, self.eps)
        return super().forward(x, g, residual=residual, prenorm=prenorm, residual_in_fp32=residual_in_fp32)


def adopt_custom_ops(layer: GatedDeltaNet2):
    """Swap the layer's opaque FLA modules for custom-operator subclasses.

    Replacements are built on the meta device and then share the layer's
    already-initialized parameters, so parameter names, values, and the random
    number stream are all unchanged.
    """
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        conv = getattr(layer, name)
        if conv.bias is not None or conv.activation != "silu" or conv.backend != "triton":
            raise ValueError("Custom operators cover bias-free SiLU Triton short convolutions")
        replacement = CustomOpShortConvolution(hidden_size=conv.hidden_size, kernel_size=conv.kernel_size[0],
                                               bias=False, activation="silu", backend="triton", device="meta")
        replacement.weight = conv.weight
        setattr(layer, name, replacement)
    norm = layer.o_norm
    if norm.bias is not None or norm.weight is None or norm.activation != "swish":
        raise ValueError("Custom operators cover the affine bias-free swish-gated RMS norm")
    replacement = CustomOpRMSNormSwishGate(norm.hidden_size, elementwise_affine=True, eps=norm.eps, device="meta")
    replacement.weight = norm.weight
    layer.o_norm = replacement


class GatedDeltaNet2Execution(GatedDeltaNet2):
    """Select exact-recurrence kernels without changing frontend parameters.

    state_v_first chooses storage layout, not state capacity. disable_recompute
    retains forward intermediates for backward, trading memory for execution
    time. custom_ops registers the same installed FLA kernels as compilable
    operators so the whole training step compiles as one graph. gate_in_kernel
    hands the raw decay projection to the chunk kernels, which compute
    -exp(A_log) * softplus(g + dt_bias) themselves instead of the surrounding
    graph. None of these options change projections, gates, or initialization.
    """

    def __init__(self, *args, gdn_backend="vendor", state_v_first=False,
                 disable_recompute=False, custom_ops=False, gate_in_kernel=False, **kwargs):
        if gdn_backend not in {"vendor", "fla"}:
            raise ValueError("gdn_backend must be vendor or fla")
        if not all(isinstance(flag, bool) for flag in (state_v_first, disable_recompute, custom_ops,
                                                       gate_in_kernel)):
            raise ValueError("state_v_first, disable_recompute, custom_ops and gate_in_kernel must be booleans")
        if custom_ops and (gdn_backend != "fla" or not disable_recompute):
            raise ValueError("custom_ops wraps the installed FLA kernels with saved intermediates")
        if gate_in_kernel and not custom_ops:
            raise ValueError("gate_in_kernel selects a kernel path of the custom-operator execution")
        super().__init__(*args, **kwargs)
        if gate_in_kernel and self.num_v_heads != self.num_heads:
            # The frontend repeats g across value-head groups before dispatch,
            # while the kernels index A_log and dt_bias by that repeated head.
            raise ValueError("gate_in_kernel indexes the decay parameters per key head; grouped value heads "
                             "are not supported")
        self.gdn_backend = gdn_backend
        self.state_v_first = state_v_first
        self.disable_recompute = disable_recompute
        self.custom_ops = custom_ops
        self.gate_in_kernel = gate_in_kernel
        if custom_ops:
            if not self.use_short_conv:
                raise ValueError("custom_ops expects the short-convolution configuration")
            adopt_custom_ops(self)

    def _log_decay(self, f_input: Tensor, mode: str) -> Tensor:
        # The chunk kernels take the raw projection when they compute the gate;
        # the short-sequence recurrent kernel keeps the frontend's fp32 decay.
        if self.gate_in_kernel and mode == "chunk":
            return self.f_proj[1](f_input)
        return super()._log_decay(f_input, mode)

    def _chunk_recurrence(self, **kwargs):
        if kwargs["use_gate_in_kernel"]:
            raise ValueError("the frontend hands in-kernel gate selection to the execution subclass")
        if self.custom_ops and kwargs["initial_state"] is None and not kwargs["output_final_state"]:
            if kwargs.get("cu_seqlens") is not None or not kwargs["use_qk_l2norm_in_kernel"]:
                raise ValueError("custom_ops training recurrence expects fixed-length in-kernel L2 normalization")
            output = gated_delta_ops.chunk_gdn2_training(
                kwargs["q"], kwargs["k"], kwargs["v"], kwargs["g"], kwargs["b"], kwargs["w"],
                A_log=kwargs["A_log"] if self.gate_in_kernel else None,
                dt_bias=kwargs["dt_bias"] if self.gate_in_kernel else None,
                state_v_first=self.state_v_first)
            return output, None
        if self.gdn_backend == "fla":
            from fla.ops.gdn2 import chunk_gdn2
            kwargs.update(use_gate_in_kernel=self.gate_in_kernel)
            return chunk_gdn2(**kwargs, state_v_first=self.state_v_first,
                              disable_recompute=self.disable_recompute)
        return super()._chunk_recurrence(**kwargs, transpose_state_layout=self.state_v_first,
                                        disable_recompute=self.disable_recompute)

    def _recurrent_recurrence(self, **kwargs):
        if self.gdn_backend == "fla":
            from fla.ops.gdn2 import fused_recurrent_gdn2
            return fused_recurrent_gdn2(**kwargs, state_v_first=self.state_v_first)
        return super()._recurrent_recurrence(**kwargs, transpose_state_layout=self.state_v_first)


class FusedGatedDeltaNet2(GatedDeltaNet2Execution):
    """Pack independent input projections while retaining their parameters.

    This subclasses NVIDIA's research/evaluation-licensed implementation; its
    license remains applicable (pretraining/gated_delta/vendor/LICENSE).
    Packing occurs inside every forward, so captured execution observes later
    optimizer updates and autograd returns gradients to the original leaves.
    """

    def _project_inputs(self, hidden_states: Tensor) -> tuple[Tensor, ...]:
        weight = torch.cat((
            self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
            self.b_proj.weight, self.w_proj.weight,
            self.f_proj[0].weight, self.g_proj[0].weight,
        ), dim=0)
        projected = F.linear(hidden_states, weight)
        return projected.split((self.key_dim, self.key_dim, self.value_dim,
                                self.key_dim, self.value_dim,
                                self.head_v_dim, self.head_v_dim), dim=-1)


class GatedDeltaBlock(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        dim = config["model_dim"]
        mixer_class = FusedGatedDeltaNet2 if config["fused_projections"] else GatedDeltaNet2Execution
        self.attn = mixer_class(
            hidden_size=dim, expand_v=config["expand_v"],
            head_dim=config["head_dim"], num_heads=config["mixer_dim"] // config["head_dim"],
            mode="chunk", use_short_conv=config["use_short_conv"],
            conv_size=config["conv_size"], conv_bias=False,
            allow_neg_eigval=config["allow_neg_eigval"], layer_idx=layer_idx,
            norm_eps=config["mixer_norm_eps"],
            gdn_backend=config["gdn_backend"], state_v_first=config["state_v_first"],
            disable_recompute=config["disable_recompute"], custom_ops=config["custom_ops"],
            gate_in_kernel=config["gate_in_kernel"],
        )
        if config["use_short_conv"]:
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                if getattr(self.attn, name).backend != "triton":
                    raise ValueError(
                        "GDN-2 runs require the Triton short-convolution backend; "
                        "remove other FLA_CONV_BACKEND overrides"
                    )
        self.mlp = MLP(dim)
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)

    def forward(self, x: Tensor, state: Cache | None = None, use_cache: bool = False):
        mixed, _, state = self.attn(self.norm1(x), past_key_values=state, use_cache=use_cache)
        x = x + mixed
        return x + self.mlp(self.norm2(x)), state


def gated_delta_config(vocab_size: int = 1024, num_layers: int = 6,
                       model_dim: int = 512, head_dim: int = 128, expand_v: float = 1.0,
                       use_short_conv: bool = True, conv_size: int = 4,
                       allow_neg_eigval: bool = False, mixer_norm_eps: float = 1e-5,
                       kernel_chunk_size: int = 64, mixer_dim: int | None = None,
                       fused_projections: bool = False, gdn_backend: str = "vendor",
                       state_v_first: bool = False, disable_recompute: bool = False,
                       custom_ops: bool = False, gate_in_kernel: bool = False) -> dict:
    """Validate and record the layerwise GDN-2 model configuration."""
    mixer_dim = model_dim if mixer_dim is None else mixer_dim
    if min(vocab_size, num_layers, model_dim, mixer_dim, head_dim, conv_size) < 1:
        raise ValueError("All model dimensions must be positive")
    if mixer_dim % head_dim or head_dim > 256:
        raise ValueError("head_dim must divide mixer_dim and must not exceed the kernel limit of 256")
    if not math.isfinite(expand_v) or expand_v <= 0 or not math.isfinite(mixer_norm_eps) or mixer_norm_eps <= 0:
        raise ValueError("expand_v and mixer_norm_eps must be positive and finite")
    if kernel_chunk_size != 64:
        raise ValueError("The unmodified official GDN-2 chunk kernel uses chunk_size=64")
    if gdn_backend not in {"vendor", "fla"}:
        raise ValueError("gdn_backend must be vendor or fla")
    if not all(isinstance(flag, bool) for flag in (state_v_first, disable_recompute, custom_ops,
                                                   gate_in_kernel)):
        raise ValueError("state_v_first, disable_recompute, custom_ops and gate_in_kernel must be booleans")
    if custom_ops and (gdn_backend != "fla" or not disable_recompute or not use_short_conv):
        raise ValueError("custom_ops requires the FLA backend, saved intermediates and short convolutions")
    if gate_in_kernel and not custom_ops:
        raise ValueError("gate_in_kernel is a kernel path of the custom-operator execution")
    return dict(vocab_size=vocab_size, num_layers=num_layers, model_dim=model_dim,
                mixer_dim=mixer_dim,
                head_dim=head_dim, expand_v=expand_v, use_short_conv=use_short_conv,
                conv_size=conv_size, allow_neg_eigval=allow_neg_eigval,
                mixer_norm_eps=mixer_norm_eps, kernel_chunk_size=kernel_chunk_size,
                fused_projections=fused_projections, gdn_backend=gdn_backend,
                state_v_first=state_v_first, disable_recompute=disable_recompute,
                custom_ops=custom_ops, gate_in_kernel=gate_in_kernel)


@torch.no_grad()
def initialize_backbone(model: nn.Module, owned):
    """nanoGPT-mini initialization for every parameter ``owned`` admits.

    The official mixer initializes its own projections, short convolutions,
    output norm, A_log and inverse-softplus dt_bias. In particular, its many
    *_proj names MUST NOT enter mini's zero-projection initializer, so callers
    exclude them (and any other self-initializing modules) through ``owned``.
    """
    for name, parameter in model.named_parameters():
        if not owned(name):
            continue
        if name == "embed.weight":
            parameter.normal_()
        elif name.endswith("gains"):
            parameter.fill_(1)
        elif name.endswith("weight"):
            if "proj" in name:
                parameter.zero_()
            else:
                parameter.normal_(std=math.sqrt(0.33 / parameter.shape[-1]))
        else:
            parameter.zero_()


class GatedDeltaGPT(nn.Module):
    """Keep residual/MLP width fixed while optionally narrowing recurrent mixing.

    mixer_dim is the total query/key width; value width is mixer_dim * expand_v.
    Its default is model_dim. Head size and mixer width change capacity and
    arithmetic, never the immediate per-token recurrence or memory-age policy.
    """

    block_class = GatedDeltaBlock

    def __init__(self, **options):
        super().__init__()
        self.config = self._configure(**options)
        vocab_size, num_layers, model_dim = (self.config[key] for key in ("vocab_size", "num_layers", "model_dim"))
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(self.block_class(self.config, index) for index in range(num_layers))
        self.proj = Linear(model_dim, vocab_size)
        self.norm1, self.norm2 = RMSNorm(model_dim), RMSNorm(model_dim)
        initialize_backbone(self, self._backbone_owns)

    def _configure(self, **options) -> dict:
        return gated_delta_config(**options)

    @staticmethod
    def _backbone_owns(name: str) -> bool:
        return ".attn." not in name

    @staticmethod
    def new_cache() -> Cache:
        """Create independent zero-state recurrent and short-convolution caches."""
        return Cache()

    def forward_hidden(self, inputs: Tensor, state: Cache | None = None,
                       segment_size: int | None = None, use_cache: bool = False):
        """Return hidden states and optionally an updated evaluation cache.

        segment_size is accepted as a positive runtime scheduling hint for the
        common LM interface. It does not split this model's sequence or reset
        state: the official recurrence's internal chunk size remains 64.
        Cached execution is evaluation-only; fresh full-sequence training has
        independent zero states for each packed row and full causal gradients.
        """
        if inputs.ndim != 2 or inputs.shape[1] == 0:
            raise ValueError("Expected nonempty [batch,time] token inputs")
        if inputs.device.type != "cuda":
            raise ValueError("GDN-2 requires CUDA execution; no CPU fallback")
        if segment_size is not None and segment_size < 1:
            raise ValueError("segment_size must be positive when supplied")
        if state is not None and not use_cache:
            raise ValueError("Pass use_cache=True when providing an existing cache")
        if use_cache:
            if self.training or torch.is_grad_enabled():
                raise ValueError("Cached GDN-2 execution requires eval mode and disabled gradients")
            if state is None:
                state = self.new_cache()
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x, state = block(x, state, use_cache=use_cache)
        return self.norm2(x), state

    def logits(self, hidden: Tensor) -> Tensor:
        logits = self.proj(hidden).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()

    @torch.no_grad()
    def step(self, tokens: Tensor, state: Cache | None = None):
        """Incremental next-token logits plus the updated per-layer cache."""
        if tokens.ndim != 1:
            raise ValueError("step expects one token per batch row")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, state = self.forward_hidden(tokens[:, None], state, use_cache=True)
            return self.logits(hidden[:, 0]), state


def build_gated_delta_model(config: dict) -> "GatedDeltaGPT":
    """Rebuild a model from the config a checkpoint recorded (``model.config``).

    Shared-pool configs record ``shared_pool=True`` and are rejected by the
    private constructor, so checkpoints of the two architectures never rebuild
    as each other.
    """
    if config.get("shared_pool"):
        from pretraining.nanogpt_mini.gated_delta_pool import SharedPoolGatedDeltaGPT
        return SharedPoolGatedDeltaGPT(**config)
    return GatedDeltaGPT(**config)

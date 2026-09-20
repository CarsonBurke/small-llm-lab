"""Scalar-gated DeltaNet control on the shared nanoGPT-mini backbone.

Uses MIT-licensed FLA GatedDeltaNet without modifying its kernels. Optional
projection packing uses the separately licensed, provenance-tracked frontend in
pretraining/gated_delta/vendor/fla_scalar; the default uses the installed class.
Each head has scalar forgetting and delta-update strength, shared across its
state channels. This restricts both GDN-2's channel-wise decay and its separate
key/value write gates; it is not an isolated change to a single gate. FLA also
uses a direct scalar decay projection instead of GDN-2's factorized decay
projection, and a dense output gate instead of its factorized projection,
and fuses compatible q/k/v convolutions during uncached sequence execution.
Matching parameter initialization distributions does not match the resulting
gate activation distributions across these different projection structures.

Every layer updates and reads its own state at the current token. There is no
ordinary token attention, delayed write, slot bank, or whole-network recurrence.
Backbone execution and initialization are inherited without first constructing
or replacing a GDN-2 model. The imported backbone retains its documented license
dependencies; the scalar mixer itself comes from flash-linear-attention.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from fla.layers.gated_deltanet import GatedDeltaNet
from pretraining.gated_delta.vendor.fla_scalar.layer import GatedDeltaNet as HookedGatedDeltaNet

from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaBlock, GatedDeltaGPT
from pretraining.nanogpt_mini.nanogpt_mini_model import Linear, MLP, RMSNorm


class FusedScalarDeltaNet(HookedGatedDeltaNet):
    """One packed input GEMM, with original independent trainable leaves."""

    def _project_inputs(self, hidden_states):
        weights = [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
                   self.a_proj.weight, self.b_proj.weight]
        widths = [2 * self.key_dim + self.value_dim, self.num_v_heads, self.num_v_heads]
        if self.use_gate:
            weights.append(self.g_proj.weight)
            widths.append(self.value_dim)
        # Scalar gate rows can make N unaligned (514 for M128/H1). Add only
        # constant zero rows to the GEMM, never parameters or real outputs.
        width = sum(widths)
        packed_weight = F.pad(torch.cat(weights, dim=0), (0, 0, 0, -width % 16))
        projected = F.linear(hidden_states, packed_weight)[..., :width]
        outputs = projected.split(widths, dim=-1)
        return outputs if self.use_gate else (*outputs, None)


class ScalarDeltaBlock(GatedDeltaBlock):
    """Reuse the residual block contract with an independently built mixer."""

    def __init__(self, config: dict, layer_idx: int):
        nn.Module.__init__(self)
        dim = config["model_dim"]
        mixer_class = FusedScalarDeltaNet if config["fused_projections"] else GatedDeltaNet
        self.attn = mixer_class(
            hidden_size=dim, expand_v=config["expand_v"],
            head_dim=config["head_dim"], num_heads=config["mixer_dim"] // config["head_dim"],
            mode="chunk", use_gate=True, use_short_conv=config["use_short_conv"],
            conv_size=config["conv_size"], conv_bias=False,
            allow_neg_eigval=config["allow_neg_eigval"], layer_idx=layer_idx,
            norm_eps=config["mixer_norm_eps"],
        )
        if config["use_short_conv"]:
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                if getattr(self.attn, name).backend != "triton":
                    raise ValueError("Scalar DeltaNet requires the Triton short-convolution backend")
        self._initialize_mixer()
        self.mlp = MLP(dim)
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)

    @torch.no_grad()
    def _initialize_mixer(self):
        # Match the GDN-2 initialization distributions, not FLA's default
        # Linear initialization or its wider A ~ Uniform[0,16) distribution.
        # Preserve convolution/norm initialization and no-weight-decay flags.
        for module in self.attn.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
                if module.bias is not None:
                    module.bias.zero_()
        self.attn.A_log.copy_(torch.empty_like(self.attn.A_log).uniform_(1, 16).log_())
        dt = torch.empty_like(self.attn.dt_bias).uniform_(math.log(.001), math.log(.1)).exp_()
        self.attn.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))


class ScalarDeltaGPT(GatedDeltaGPT):
    """Same LM/cache API as GatedDeltaGPT, with scalar-gated FLA mixers.

Default: six D512 blocks, four heads of width64, total mixer width256.
Full-sequence training starts each row with fresh state and retains full BPTT;
the inherited eval-only cache and step interfaces preserve immediate writes.
"""

    def __init__(self, vocab_size: int = 1024, num_layers: int = 6,
                 model_dim: int = 512, head_dim: int = 64, expand_v: float = 1.0,
                 use_short_conv: bool = True, conv_size: int = 4,
                 allow_neg_eigval: bool = False, mixer_norm_eps: float = 1e-5,
                 kernel_chunk_size: int = 64, mixer_dim: int = 256,
                 memory_rule: str = "scalar_delta",
                 initialization: str = "gdn2_matched_distributions",
                 fused_projections: bool = False):
        nn.Module.__init__(self)
        if memory_rule != "scalar_delta" or initialization != "gdn2_matched_distributions":
            raise ValueError("Unsupported scalar DeltaNet recurrence or initialization")
        if min(vocab_size, num_layers, model_dim, mixer_dim, head_dim, conv_size) < 1:
            raise ValueError("All model dimensions must be positive")
        if mixer_dim % head_dim or head_dim > 256:
            raise ValueError("head_dim must divide mixer_dim and not exceed 256")
        if not math.isfinite(expand_v) or expand_v <= 0 or not math.isfinite(mixer_norm_eps) or mixer_norm_eps <= 0:
            raise ValueError("expand_v and mixer_norm_eps must be positive and finite")
        if kernel_chunk_size != 64:
            raise ValueError("The installed scalar DeltaNet chunk kernel uses chunk_size=64")
        self.config = dict(vocab_size=vocab_size, num_layers=num_layers, model_dim=model_dim,
                           mixer_dim=mixer_dim, head_dim=head_dim, expand_v=expand_v,
                           use_short_conv=use_short_conv, conv_size=conv_size,
                           allow_neg_eigval=allow_neg_eigval, mixer_norm_eps=mixer_norm_eps,
                           kernel_chunk_size=kernel_chunk_size, memory_rule=memory_rule,
                           initialization=initialization, fused_projections=fused_projections)
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(ScalarDeltaBlock(self.config, index) for index in range(num_layers))
        self.proj = Linear(model_dim, vocab_size)
        self.norm1, self.norm2 = RMSNorm(model_dim), RMSNorm(model_dim)
        self._initialize_backbone()

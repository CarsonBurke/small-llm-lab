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

from pretraining.gated_delta.vendor.gdn2 import GatedDeltaNet2
from pretraining.nanogpt_mini.nanogpt_mini_model import Linear, MLP, RMSNorm


class GatedDeltaNet2Execution(GatedDeltaNet2):
    """Select exact-recurrence kernels without changing frontend parameters.

    state_v_first chooses storage layout, not state capacity. disable_recompute
    retains forward intermediates for backward, trading memory for execution
    time. These options do not change projections, gates, or initialization.
    """

    def __init__(self, *args, gdn_backend="vendor", state_v_first=False,
                 disable_recompute=False, **kwargs):
        if gdn_backend not in {"vendor", "fla"}:
            raise ValueError("gdn_backend must be vendor or fla")
        if not isinstance(state_v_first, bool) or not isinstance(disable_recompute, bool):
            raise ValueError("state_v_first and disable_recompute must be booleans")
        super().__init__(*args, **kwargs)
        self.gdn_backend = gdn_backend
        self.state_v_first = state_v_first
        self.disable_recompute = disable_recompute

    def _chunk_recurrence(self, **kwargs):
        if self.gdn_backend == "fla":
            from fla.ops.gdn2 import chunk_gdn2
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
            disable_recompute=config["disable_recompute"],
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


class GatedDeltaGPT(nn.Module):
    """Keep residual/MLP width fixed while optionally narrowing recurrent mixing.

    mixer_dim is the total query/key width; value width is mixer_dim * expand_v.
    Its default is model_dim. Head size and mixer width change capacity and
    arithmetic, never the immediate per-token recurrence or memory-age policy.
    """

    def __init__(self, vocab_size: int = 1024, num_layers: int = 6,
                 model_dim: int = 512, head_dim: int = 128, expand_v: float = 1.0,
                 use_short_conv: bool = True, conv_size: int = 4,
                 allow_neg_eigval: bool = False, mixer_norm_eps: float = 1e-5,
                 kernel_chunk_size: int = 64, mixer_dim: int | None = None,
                 fused_projections: bool = False, gdn_backend: str = "vendor",
                 state_v_first: bool = False, disable_recompute: bool = False):
        super().__init__()
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
        if not isinstance(state_v_first, bool) or not isinstance(disable_recompute, bool):
            raise ValueError("state_v_first and disable_recompute must be booleans")
        self.config = dict(vocab_size=vocab_size, num_layers=num_layers, model_dim=model_dim,
                           mixer_dim=mixer_dim,
                           head_dim=head_dim, expand_v=expand_v, use_short_conv=use_short_conv,
                           conv_size=conv_size, allow_neg_eigval=allow_neg_eigval,
                           mixer_norm_eps=mixer_norm_eps, kernel_chunk_size=kernel_chunk_size,
                           fused_projections=fused_projections, gdn_backend=gdn_backend,
                           state_v_first=state_v_first, disable_recompute=disable_recompute)
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(GatedDeltaBlock(self.config, index) for index in range(num_layers))
        self.proj = Linear(model_dim, vocab_size)
        self.norm1, self.norm2 = RMSNorm(model_dim), RMSNorm(model_dim)
        self._initialize_backbone()

    @torch.no_grad()
    def _initialize_backbone(self):
        # The official mixer initializes its own projections, short convolutions,
        # output norm, A_log and inverse-softplus dt_bias. In particular, its
        # many *_proj names MUST NOT enter mini's zero-projection initializer.
        for name, parameter in self.named_parameters():
            if ".attn." in name:
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

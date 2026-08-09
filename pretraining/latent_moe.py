"""Kimi K3-style Stable LatentMoE for small language models.

The routed experts operate on a compact latent representation while a fused
full-width shared expert preserves a dense path.  CUDA BF16 execution sorts
selected token/expert pairs once and uses differentiable ``torch._grouped_mm``
for both expert projections.  The reference path deliberately favors clarity
and portability over memory efficiency.

Routing follows Kimi K3 exactly: an FP32 sigmoid router sees the full-width
token, a non-trainable correction bias affects Top-k selection only, and the
mixture weights are the normalized *unbiased* router scores.  Bias updates are
explicit and causal: callers compute a next bias from a completed batch and
install it for the following optimizer step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import torch
from torch import Tensor, nn
import torch.nn.functional as F


Implementation = Literal["auto", "grouped_mm", "selected_bmm", "reference"]


@dataclass(frozen=True)
class LatentMoEConfig:
    """Dimensions and numerical choices for :class:`StableLatentMoE`."""

    model_dim: int = 512
    latent_dim: int = 128
    routed_hidden_dim: int = 256
    num_routed_experts: int = 32
    experts_per_token: int = 2
    shared_hidden_dim: int = 64
    num_shared_experts: int = 2
    rms_norm_eps: float = 1e-6
    situ_gate_cap: float = 4.0
    situ_up_cap: float = 25.0

    def __post_init__(self) -> None:
        positive = {
            "model_dim": self.model_dim,
            "latent_dim": self.latent_dim,
            "routed_hidden_dim": self.routed_hidden_dim,
            "num_routed_experts": self.num_routed_experts,
            "experts_per_token": self.experts_per_token,
            "shared_hidden_dim": self.shared_hidden_dim,
            "num_shared_experts": self.num_shared_experts,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.experts_per_token > self.num_routed_experts:
            raise ValueError(
                "experts_per_token cannot exceed num_routed_experts: "
                f"{self.experts_per_token} > {self.num_routed_experts}"
            )
        if self.num_routed_experts >= 1024:
            raise ValueError("torch._grouped_mm requires fewer than 1024 groups")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.situ_gate_cap <= 0 or self.situ_up_cap <= 0:
            raise ValueError("SiTU soft caps must be positive")


@dataclass(frozen=True)
class RouterTelemetry:
    """Detached routing diagnostics, shaped like the input token dimensions."""

    expert_indices: Tensor
    expert_weights: Tensor
    expert_loads: Tensor
    load_fraction: Tensor
    load_cv_squared: Tensor


@dataclass(frozen=True)
class QuantileBalanceHistogram:
    """Additive K3 Quantile Balancing statistics for one layer and step.

    ``counts`` and ``token_count`` are the only fields that should be summed
    across micro-batches or data-parallel ranks.  Bounds are deterministic from
    the installed correction bias and therefore must agree across ranks.
    """

    counts: Tensor
    token_count: Tensor
    lower_bound: Tensor
    upper_bound: Tensor


def situ_glu(
    gate: Tensor, up: Tensor, gate_cap: float = 4.0, up_cap: float = 25.0
) -> Tensor:
    """Kimi K3's bounded SiTU-GLU activation."""

    capped_gate = gate_cap * torch.tanh(gate / gate_cap)
    capped_up = up_cap * torch.tanh(up / up_cap)
    return capped_gate * torch.sigmoid(gate) * capped_up


class _SiTUGLUMLP(nn.Module):
    """Bias-free SiTU-GLU MLP with a fused gate/up projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        gate_cap: float,
        up_cap: float,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            input_dim, 2 * hidden_dim, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            hidden_dim, output_dim, bias=False, device=device, dtype=dtype
        )
        self.gate_cap = gate_cap
        self.up_cap = up_cap

    def forward(self, x: Tensor) -> Tensor:
        gate_up = F.linear(x, self.gate_up_proj.weight.to(x.dtype))
        gate, up = gate_up.chunk(2, dim=-1)
        activated = situ_glu(gate, up, self.gate_cap, self.up_cap)
        return F.linear(activated, self.down_proj.weight.to(activated.dtype))


class _RMSNorm(nn.Module):
    """Affine RMSNorm with FP32 reduction for low-precision activations."""

    def __init__(
        self,
        dim: int,
        *,
        eps: float,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        reduction_dtype = (
            torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        )
        normalized = F.rms_norm(x.to(reduction_dtype), (x.shape[-1],), eps=self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class StableLatentMoE(nn.Module):
    """Normalized LatentMoE with auxiliary-loss-free K3 routing.

    The default construction keeps FP32 optimizer-master parameters, matching
    the surrounding KDA model, and casts projection weights to activation dtype
    at use sites.  Construct with ``dtype=torch.bfloat16`` only when true BF16
    parameters (for example, inference weights) are desired.  Router arithmetic
    is FP32 in either case.
    """

    def __init__(
        self,
        config: LatentMoEConfig = LatentMoEConfig(),
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        implementation: Implementation = "auto",
    ) -> None:
        super().__init__()
        if implementation not in ("auto", "grouped_mm", "selected_bmm", "reference"):
            raise ValueError(f"unknown implementation: {implementation!r}")
        self.config = config
        self.implementation = implementation

        # K3 routes from the original full-width token, not the latent input.
        self.router_weight = nn.Parameter(
            torch.empty(
                config.num_routed_experts,
                config.model_dim,
                device=device,
                dtype=torch.float32,
            )
        )
        self.register_buffer(
            "correction_bias",
            torch.zeros(config.num_routed_experts, device=device, dtype=torch.float32),
        )
        # Full-step Quantile Balancing collection is opt-in.  Empty,
        # nonpersistent buffers impose no checkpoint or forward cost until the
        # fixed histogram shape is configured before torch.compile.
        self.register_buffer(
            "_qb_counts",
            torch.empty(config.num_routed_experts, 0, device=device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_qb_token_count",
            torch.zeros((), device=device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_qb_expert_offsets",
            torch.empty(0, device=device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_qb_route_loads",
            torch.zeros(config.num_routed_experts, device=device, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_qb_lower_bound",
            torch.zeros((), device=device, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_qb_upper_bound",
            torch.zeros((), device=device, dtype=torch.float32),
            persistent=False,
        )
        self._qb_collection_enabled = False
        self._qb_collect_in_eval = False

        self.latent_down_proj = nn.Linear(
            config.model_dim, config.latent_dim, bias=False, device=device, dtype=dtype
        )
        self.expert_gate_up_weight = nn.Parameter(
            torch.empty(
                config.num_routed_experts,
                2 * config.routed_hidden_dim,
                config.latent_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.expert_down_weight = nn.Parameter(
            torch.empty(
                config.num_routed_experts,
                config.latent_dim,
                config.routed_hidden_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.routed_norm = _RMSNorm(
            config.latent_dim,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        # Concatenating the hidden widths of the shared experts is exactly
        # equivalent to summing independent bias-free expert outputs, while
        # requiring only one gate/up GEMM and one down GEMM.
        self.shared_expert = _SiTUGLUMLP(
            config.model_dim,
            config.shared_hidden_dim * config.num_shared_experts,
            config.model_dim,
            gate_cap=config.situ_gate_cap,
            up_cap=config.situ_up_cap,
            device=device,
            dtype=dtype,
        )
        self.latent_up_proj = nn.Linear(
            config.latent_dim, config.model_dim, bias=False, device=device, dtype=dtype
        )
        self.reset_parameters()

    def _apply(self, fn, recurse: bool = True):  # type: ignore[no-untyped-def]
        """Preserve K3's FP32 routing tensors across whole-model casts.

        Device moves still apply normally. As with any PyTorch dtype change,
        callers must cast a model before constructing its optimizer; retaining
        router precision does not make post-optimizer whole-model casts safe.
        """

        super()._apply(fn, recurse=recurse)
        if self.router_weight.dtype != torch.float32:
            self.router_weight.data = self.router_weight.data.float()
            if self.router_weight.grad is not None:
                self.router_weight.grad.data = self.router_weight.grad.data.float()
        if self.correction_bias.dtype != torch.float32:
            self.correction_bias.data = self.correction_bias.data.float()
        if self._qb_lower_bound.dtype != torch.float32:
            self._qb_lower_bound = self._qb_lower_bound.float()
        if self._qb_upper_bound.dtype != torch.float32:
            self._qb_upper_bound = self._qb_upper_bound.float()
        return self

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.router_weight, a=5**0.5)
        self.latent_down_proj.reset_parameters()
        self.shared_expert.gate_up_proj.reset_parameters()
        self.shared_expert.down_proj.reset_parameters()
        self.latent_up_proj.reset_parameters()
        with torch.no_grad():
            gate_up_bound = self.config.latent_dim**-0.5
            self.expert_gate_up_weight.uniform_(-gate_up_bound, gate_up_bound)
            down_bound = self.config.routed_hidden_dim**-0.5
            self.expert_down_weight.uniform_(-down_bound, down_bound)
            self.routed_norm.weight.fill_(1.0)
            self.correction_bias.zero_()

    def _router_scores(self, flat_x: Tensor) -> Tensor:
        # Autocast is explicitly bypassed: both router matmul and sigmoid are FP32.
        device_type = flat_x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            return torch.sigmoid(F.linear(flat_x.float(), self.router_weight.float()))

    def _route(
        self, flat_x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        scores = self._router_scores(flat_x)
        biased_scores = scores + self.correction_bias
        collect_qb = self._qb_collection_enabled and (
            self.training or self._qb_collect_in_eval
        )
        top_count = self.config.experts_per_token + int(collect_qb)
        selected = biased_scores.topk(top_count, dim=-1)
        expert_indices = selected.indices[:, : self.config.experts_per_token]
        cutoff = selected.values[:, -1] if collect_qb else None
        selected_scores = scores.gather(1, expert_indices)
        denominator = selected_scores.sum(dim=-1, keepdim=True)
        normalized = selected_scores / denominator.clamp_min(
            torch.finfo(selected_scores.dtype).tiny
        )
        # Avoid NaNs only in the extreme FP32-underflow case; ordinary routing
        # is exactly the normalized unbiased sigmoid score from the paper.
        uniform = torch.full_like(normalized, 1.0 / self.config.experts_per_token)
        expert_weights = torch.where(denominator > 0, normalized, uniform)
        return scores, expert_indices, expert_weights, cutoff

    @torch.no_grad()
    def _accumulate_qb_forward_statistics(
        self,
        scores: Tensor,
        cutoff: Tensor,
        expert_loads: Tensor,
    ) -> None:
        num_bins = self._qb_counts.shape[1]
        required_bias = cutoff[:, None] - scores.detach()
        scaled = (required_bias - self._qb_lower_bound) / (
            self._qb_upper_bound - self._qb_lower_bound
        )
        bin_indices = (
            torch.floor(scaled * num_bins).to(torch.int64).clamp_(0, num_bins - 1)
        )
        bin_indices.add_(self._qb_expert_offsets[None, :])
        self._qb_counts.view(-1).scatter_add_(
            0,
            bin_indices.reshape(-1),
            torch.ones_like(bin_indices, dtype=torch.int64).reshape(-1),
        )
        self._qb_token_count.add_(scores.shape[0])
        self._qb_route_loads.add_(expert_loads)

    def _can_use_grouped_mm(self, latent: Tensor) -> bool:
        if not hasattr(torch, "_grouped_mm"):
            return False
        if not latent.is_cuda or latent.dtype != torch.bfloat16:
            return False
        if not self.expert_gate_up_weight.is_floating_point():
            return False
        # CUTLASS grouped GEMM requires K and N to be 16-byte aligned.
        alignment = 8  # 16 bytes / sizeof(BF16)
        dims = (
            self.config.latent_dim,
            2 * self.config.routed_hidden_dim,
            self.config.routed_hidden_dim,
        )
        return all(dim % alignment == 0 for dim in dims)

    def _grouped_experts(
        self,
        latent: Tensor,
        expert_indices: Tensor,
        expert_weights: Tensor,
        expert_loads: Tensor,
    ) -> Tensor:
        routes_per_token = self.config.experts_per_token
        flat_experts = expert_indices.reshape(-1)
        sort_order = flat_experts.argsort()
        sorted_token_indices = torch.div(sort_order, routes_per_token, rounding_mode="floor")
        routed_latent = latent.index_select(0, sorted_token_indices)
        offsets = expert_loads.cumsum(0, dtype=torch.int32)

        gate_up = torch._grouped_mm(
            routed_latent,
            self.expert_gate_up_weight.to(routed_latent.dtype).transpose(1, 2),
            offs=offsets,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = situ_glu(
            gate,
            up,
            self.config.situ_gate_cap,
            self.config.situ_up_cap,
        )
        sorted_output = torch._grouped_mm(
            activated,
            self.expert_down_weight.to(activated.dtype).transpose(1, 2),
            offs=offsets,
        )

        sorted_weights = expert_weights.reshape(-1).index_select(0, sort_order)
        weighted_output = sorted_output * sorted_weights.to(sorted_output.dtype)[:, None]
        # Aggregate directly in token order, avoiding an inverse permutation
        # and a materialized [tokens, top_k, latent_dim] output.
        aggregate = torch.zeros_like(latent)
        return aggregate.index_add(0, sorted_token_indices, weighted_output)

    def _reference_experts(self, latent: Tensor, expert_indices: Tensor) -> Tensor:
        routes_per_token = self.config.experts_per_token
        flat_experts = expert_indices.reshape(-1)
        routed_latent = (
            latent[:, None, :]
            .expand(-1, routes_per_token, -1)
            .reshape(-1, self.config.latent_dim)
        )
        route_output = latent.new_empty(flat_experts.shape[0], self.config.latent_dim)
        # The portable path loops over the static expert count to avoid gathering
        # a separate copy of both expert matrices for every selected route.  It
        # is intentionally not the optimized CUDA path, but remains memory-safe
        # for realistic token counts and fully differentiable.
        for expert_index in range(self.config.num_routed_experts):
            route_positions = torch.where(flat_experts == expert_index)[0]
            expert_input = routed_latent.index_select(0, route_positions)
            gate_up = F.linear(
                expert_input,
                self.expert_gate_up_weight[expert_index].to(latent.dtype),
            )
            gate, up = gate_up.chunk(2, dim=-1)
            activated = situ_glu(
                gate,
                up,
                self.config.situ_gate_cap,
                self.config.situ_up_cap,
            )
            expert_output = F.linear(
                activated,
                self.expert_down_weight[expert_index].to(activated.dtype),
            )
            route_output.index_copy_(0, route_positions, expert_output)
        return route_output.view(-1, routes_per_token, self.config.latent_dim)

    def _selected_bmm_experts(self, latent: Tensor, expert_indices: Tensor) -> Tensor:
        """Vectorized selected-route backend for low-token-count CUDA decode."""

        routes_per_token = self.config.experts_per_token
        flat_experts = expert_indices.reshape(-1)
        routed_latent = (
            latent[:, None, :]
            .expand(-1, routes_per_token, -1)
            .reshape(-1, self.config.latent_dim)
        )
        gate_up_weight = self.expert_gate_up_weight.index_select(0, flat_experts).to(
            latent.dtype
        )
        gate_up = torch.bmm(gate_up_weight, routed_latent.unsqueeze(-1)).squeeze(-1)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = situ_glu(
            gate,
            up,
            self.config.situ_gate_cap,
            self.config.situ_up_cap,
        )
        down_weight = self.expert_down_weight.index_select(0, flat_experts).to(
            activated.dtype
        )
        route_output = torch.bmm(down_weight, activated.unsqueeze(-1)).squeeze(-1)
        return route_output.view(-1, routes_per_token, self.config.latent_dim)

    def _resolve_implementation(
        self, latent: Tensor, requested: Implementation | None
    ) -> Implementation:
        implementation = self.implementation if requested is None else requested
        if implementation not in ("auto", "grouped_mm", "selected_bmm", "reference"):
            raise ValueError(f"unknown implementation: {implementation!r}")
        can_group = self._can_use_grouped_mm(latent)
        if implementation == "grouped_mm" and not can_group:
            raise RuntimeError(
                "grouped_mm requires CUDA BF16 activations, torch._grouped_mm, "
                "and 16-byte-aligned latent/expert dimensions"
            )
        if implementation == "auto":
            return "grouped_mm" if can_group else "reference"
        return implementation

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        return_telemetry: Literal[False] = False,
        implementation: Implementation | None = None,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        return_telemetry: Literal[True],
        implementation: Implementation | None = None,
    ) -> tuple[Tensor, RouterTelemetry]: ...

    def forward(
        self,
        x: Tensor,
        *,
        return_telemetry: bool = False,
        implementation: Implementation | None = None,
    ) -> Tensor | tuple[Tensor, RouterTelemetry]:
        if x.ndim < 2 or x.shape[-1] != self.config.model_dim:
            raise ValueError(
                f"expected [..., {self.config.model_dim}] input, got {tuple(x.shape)}"
            )
        token_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.config.model_dim)
        shared_output = self.shared_expert(flat_x)

        scores, expert_indices, expert_weights, cutoff = self._route(flat_x)
        flat_experts = expert_indices.reshape(-1)
        expert_loads = torch.zeros(
            self.config.num_routed_experts,
            device=flat_experts.device,
            dtype=torch.int64,
        ).scatter_add(
            0, flat_experts, torch.ones_like(flat_experts, dtype=torch.int64)
        )
        if cutoff is not None:
            self._accumulate_qb_forward_statistics(scores, cutoff, expert_loads)
        latent = F.linear(flat_x, self.latent_down_proj.weight.to(flat_x.dtype))
        if flat_x.shape[0] == 0:
            routed_aggregate = latent
        else:
            selected_implementation = self._resolve_implementation(latent, implementation)
            if selected_implementation == "grouped_mm":
                routed_aggregate = self._grouped_experts(
                    latent, expert_indices, expert_weights, expert_loads
                )
            else:
                route_output = (
                    self._selected_bmm_experts(latent, expert_indices)
                    if selected_implementation == "selected_bmm"
                    else self._reference_experts(latent, expert_indices)
                )
                routed_aggregate = (
                    route_output * expert_weights.to(route_output.dtype)[..., None]
                ).sum(1)

        normalized_routed = self.routed_norm(routed_aggregate)
        routed_output = F.linear(
            normalized_routed, self.latent_up_proj.weight.to(normalized_routed.dtype)
        )
        output = (shared_output + routed_output).reshape(*token_shape, self.config.model_dim)
        if not return_telemetry:
            return output

        total_routes = expert_loads.sum()
        load_fraction = expert_loads.float() / total_routes.clamp_min(1)
        uniform_load = 1.0 / self.config.num_routed_experts
        load_cv_squared = ((load_fraction - uniform_load).square().mean() / uniform_load**2)
        telemetry = RouterTelemetry(
            expert_indices=expert_indices.reshape(
                *token_shape, self.config.experts_per_token
            ).detach(),
            expert_weights=expert_weights.reshape(
                *token_shape, self.config.experts_per_token
            ).detach(),
            expert_loads=expert_loads.detach(),
            load_fraction=load_fraction.detach(),
            load_cv_squared=load_cv_squared.detach(),
        )
        return output, telemetry

    @torch.no_grad()
    def compute_next_correction_bias(self, x: Tensor) -> Tensor:
        """Compute K3's exact single-batch Quantile Balancing update.

        The returned mean-centered bias is not installed automatically.  This
        preserves the paper's causal contract: route the current batch with the
        old bias, then call :meth:`set_correction_bias_` for the next step.
        Distributed training should compute the equivalent global quantile (or
        K3's histogram approximation) before installing it.
        """

        if self.config.experts_per_token == self.config.num_routed_experts:
            raise RuntimeError(
                "Quantile Balancing is undefined when every expert is selected"
            )
        if x.ndim < 2 or x.shape[-1] != self.config.model_dim:
            raise ValueError(
                f"expected [..., {self.config.model_dim}] input, got {tuple(x.shape)}"
            )
        flat_x = x.reshape(-1, self.config.model_dim)
        if flat_x.shape[0] == 0:
            raise ValueError("cannot estimate Quantile Balancing from an empty batch")
        scores = self._router_scores(flat_x)
        biased_scores = scores + self.correction_bias
        cutoff = biased_scores.topk(self.config.experts_per_token + 1, dim=-1).values[:, -1]
        margins = scores - cutoff[:, None]
        quantile = 1.0 - self.config.experts_per_token / self.config.num_routed_experts
        next_bias = -torch.quantile(margins, quantile, dim=0)
        return next_bias - next_bias.mean()

    def enable_qb_collection(
        self, *, num_bins: int = 1000, collect_in_eval: bool = False
    ) -> None:
        """Enable fixed-shape full-step QB collection before compilation.

        The histogram shape is immutable after the first call because changing
        it would invalidate compiled graphs and CUDA-graph storage addresses.
        Collection occurs only in ``training`` mode unless a trainer whose
        modules stay permanently in eval mode passes ``collect_in_eval=True``
        before compilation. Call
        :meth:`reset_qb_accumulators_` at each step boundary after installing
        the bias that will remain fixed throughout that step.
        """

        if torch.compiler.is_compiling():
            raise RuntimeError("enable QB collection before calling torch.compile")
        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}")
        if self.config.experts_per_token == self.config.num_routed_experts:
            raise RuntimeError(
                "Quantile Balancing is undefined when every expert is selected"
            )
        existing_bins = self._qb_counts.shape[1]
        if existing_bins not in (0, num_bins):
            raise RuntimeError(
                "QB histogram shape is immutable after configuration: "
                f"already {existing_bins} bins, requested {num_bins}"
            )
        if existing_bins == 0:
            self._qb_counts = torch.zeros(
                self.config.num_routed_experts,
                num_bins,
                device=self.correction_bias.device,
                dtype=torch.int64,
            )
            self._qb_expert_offsets = (
                torch.arange(
                    self.config.num_routed_experts,
                    device=self.correction_bias.device,
                    dtype=torch.int64,
                )
                * num_bins
            )
        self._qb_collection_enabled = True
        self._qb_collect_in_eval = collect_in_eval
        self.reset_qb_accumulators_()

    @torch.no_grad()
    def reset_qb_accumulators_(self) -> None:
        """Reset full-step statistics and freeze bin bounds from current bias."""

        if not self._qb_collection_enabled:
            raise RuntimeError("QB collection must be enabled before reset")
        self._qb_counts.zero_()
        self._qb_token_count.zero_()
        self._qb_route_loads.zero_()
        self._qb_lower_bound.copy_(self.correction_bias.min() - 1.0)
        self._qb_upper_bound.copy_(self.correction_bias.max() + 1.0)

    @torch.no_grad()
    def get_accumulated_qb_histogram(self) -> QuantileBalanceHistogram:
        """Return a snapshot suitable for cross-rank summation and recovery."""

        if not self._qb_collection_enabled:
            raise RuntimeError("QB collection is disabled")
        return QuantileBalanceHistogram(
            counts=self._qb_counts.clone(),
            token_count=self._qb_token_count.clone(),
            lower_bound=self._qb_lower_bound.clone(),
            upper_bound=self._qb_upper_bound.clone(),
        )

    @torch.no_grad()
    def get_accumulated_route_loads(self) -> Tensor:
        """Return routed-expert loads pooled across the current step."""

        if not self._qb_collection_enabled:
            raise RuntimeError("QB collection is disabled")
        return self._qb_route_loads.clone()

    @torch.no_grad()
    def compute_qb_histogram(
        self, x: Tensor, *, num_bins: int = 1000
    ) -> QuantileBalanceHistogram:
        """Build the paper's additive per-expert Quantile Balancing histogram.

        Required biases ``cutoff - raw_score`` are binned over the proven range
        ``[correction_bias.min() - 1, correction_bias.max() + 1]``.  The method
        neither communicates nor mutates module state.  Training code may sum
        ``counts`` and ``token_count`` across micro-batches/ranks before passing
        the result to :meth:`correction_bias_from_qb_histogram`.
        """

        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}")
        if self.config.experts_per_token == self.config.num_routed_experts:
            raise RuntimeError(
                "Quantile Balancing is undefined when every expert is selected"
            )
        if x.ndim < 2 or x.shape[-1] != self.config.model_dim:
            raise ValueError(
                f"expected [..., {self.config.model_dim}] input, got {tuple(x.shape)}"
            )
        flat_x = x.reshape(-1, self.config.model_dim)
        if flat_x.shape[0] == 0:
            raise ValueError(
                "cannot build Quantile Balancing histogram from an empty batch"
            )
        scores = self._router_scores(flat_x)
        biased_scores = scores + self.correction_bias
        cutoff = biased_scores.topk(self.config.experts_per_token + 1, dim=-1).values[:, -1]
        required_bias = cutoff[:, None] - scores

        lower_bound = self.correction_bias.min() - 1.0
        upper_bound = self.correction_bias.max() + 1.0
        scaled = (required_bias - lower_bound) / (upper_bound - lower_bound)
        bin_indices = (
            torch.floor(scaled * num_bins).to(torch.int64).clamp_(0, num_bins - 1)
        )
        expert_offsets = (
            torch.arange(
                self.config.num_routed_experts,
                device=scores.device,
                dtype=torch.int64,
            )
            * num_bins
        )
        bin_indices.add_(expert_offsets[None, :])
        flat_indices = bin_indices.reshape(-1)
        flat_counts = torch.zeros(
            self.config.num_routed_experts * num_bins,
            device=scores.device,
            dtype=torch.int64,
        ).scatter_add(
            0, flat_indices, torch.ones_like(flat_indices, dtype=torch.int64)
        )
        return QuantileBalanceHistogram(
            counts=flat_counts.view(self.config.num_routed_experts, num_bins),
            token_count=torch.full(
                (), flat_x.shape[0], device=scores.device, dtype=torch.int64
            ),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    @torch.no_grad()
    def correction_bias_from_qb_histogram(
        self, histogram: QuantileBalanceHistogram
    ) -> Tensor:
        """Recover a centered next-step bias from a summed QB histogram.

        Recovery selects the first bin whose cumulative count reaches the
        target load ``tokens * top_k / num_experts`` and linearly interpolates
        within that bin, matching Appendix D of the Kimi K3 report.
        """

        counts = histogram.counts
        if counts.ndim != 2 or counts.shape[0] != self.config.num_routed_experts:
            raise ValueError(
                "histogram counts must have shape "
                f"[{self.config.num_routed_experts}, num_bins], got {tuple(counts.shape)}"
            )
        if counts.shape[1] == 0:
            raise ValueError("histogram must contain at least one bin")
        if histogram.token_count.numel() != 1:
            raise ValueError("histogram token_count must be scalar")
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if counts.dtype not in integer_dtypes:
            raise ValueError("histogram counts must use an integer dtype")
        if histogram.token_count.dtype not in integer_dtypes:
            raise ValueError("histogram token_count must use an integer dtype")
        if histogram.lower_bound.numel() != 1 or histogram.upper_bound.numel() != 1:
            raise ValueError("histogram bounds must be scalar")
        count_values = counts
        torch._assert_async(
            histogram.token_count > 0,
            "histogram token_count must be positive",
        )
        torch._assert_async(
            (count_values >= 0).all(),
            "histogram counts must be nonnegative",
        )
        torch._assert_async(
            (count_values.sum(dim=1) == histogram.token_count).all(),
            "each expert histogram must count every pooled token exactly once",
        )
        valid_bounds = (
            torch.isfinite(histogram.lower_bound)
            & torch.isfinite(histogram.upper_bound)
            & (histogram.upper_bound > histogram.lower_bound)
        )
        torch._assert_async(valid_bounds, "histogram bounds must be finite and increasing")

        cumulative = count_values.cumsum(dim=1)
        target = (
            histogram.token_count.to(torch.float32)
            * self.config.experts_per_token
            / self.config.num_routed_experts
        )
        target_rank = torch.ceil(target).to(cumulative.dtype)
        selected_bin = (cumulative >= target_rank).to(torch.int64).argmax(dim=1)
        selected_count = count_values.gather(1, selected_bin[:, None]).squeeze(1)
        cumulative_before = F.pad(cumulative, (1, 0))
        count_before = cumulative_before.gather(1, selected_bin[:, None]).squeeze(1)
        fraction = (
            (target - count_before.to(target.dtype))
            / selected_count.clamp_min(1).to(target.dtype)
        ).clamp(0.0, 1.0)
        bin_width = (histogram.upper_bound - histogram.lower_bound) / counts.shape[1]
        next_bias = histogram.lower_bound + (
            selected_bin.to(bin_width.dtype) + fraction.to(bin_width.dtype)
        ) * bin_width
        return next_bias - next_bias.mean()

    @torch.no_grad()
    def set_correction_bias_(self, correction_bias: Tensor) -> None:
        """Install a finite, mean-centered correction bias for a future batch."""

        if correction_bias.shape != self.correction_bias.shape:
            raise ValueError(
                f"expected correction bias shape {tuple(self.correction_bias.shape)}, "
                f"got {tuple(correction_bias.shape)}"
            )
        torch._assert_async(
            torch.isfinite(correction_bias).all(),
            "correction bias must contain only finite values",
        )
        centered = correction_bias.to(
            device=self.correction_bias.device, dtype=self.correction_bias.dtype
        )
        self.correction_bias.copy_(centered - centered.mean())


__all__ = [
    "Implementation",
    "LatentMoEConfig",
    "QuantileBalanceHistogram",
    "RouterTelemetry",
    "StableLatentMoE",
    "situ_glu",
]

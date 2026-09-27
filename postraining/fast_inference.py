from __future__ import annotations
import copy
import gc

from collections.abc import Callable, Sequence
from importlib import import_module
from dataclasses import dataclass
import time
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.library import wrap_triton
import triton
import triton.language as tl
from transformers.cache_utils import Cache, StaticLayer

from postraining.core import top_p_sample
from postraining.vapo.policy import VAPOPolicy
from postraining.vapo.model.lora import (
    LoRALinear,
    merge_lora_for_inference,
)
from postraining.invariant_linear import (
    INVARIANT_ARITHMETIC,
    LEGACY_ARITHMETIC,
    OPTIMIZED_ARITHMETIC,
    InvariantLinear,
    compile_invariant,
    install_invariant_linears,
)
from postraining.invariant_attention import INVARIANT_ATTENTION, invariant_suffix_attention
from postraining.slot_memory import SlotMemoryConfig, SlotMemoryRolloutState
from postraining.split_kv_plan import (
    plan_split_kv,
    split_kv_metadata,
    split_kv_plan_shapes,
)
from postraining.vapo.rollout.results import (
    ContinuousTrainingGeneration,
    FastTrainingDecodeStats,
    response_token_limits,
)
from postraining.thinking_budget import force_thinking_end_, validate_thinking_budget


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max

ROLLOUT_WORKSPACE_RESERVE_BYTES = 1 << 30
DEFAULT_OPTIMIZED_DECODE = True


class _FrozenParameterStash:
    """Keep immutable training weights on the host while rollout KV is resident."""

    def __init__(self, module: nn.Module) -> None:
        self.parameters = tuple(
            parameter for parameter in module.parameters() if not parameter.requires_grad
        )
        if not self.parameters:
            raise ValueError("rollout source has no frozen parameters to offload")
        devices = {parameter.device for parameter in self.parameters}
        if len(devices) != 1:
            raise ValueError("rollout source frozen parameters span multiple devices")
        self.device = devices.pop()
        self.host_backing: tuple[Tensor, ...] | None = None
        self.resident = True
        self.bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.parameters
        )

    def restore(self) -> None:
        if self.resident:
            return
        if self.host_backing is None:
            raise RuntimeError("frozen rollout source has no host backing")
        for parameter, host_value in zip(
            self.parameters, self.host_backing, strict=True
        ):
            # Pinned backing: the copy is stream-ordered, so consumers on the
            # current stream see the restored values without a host stall.
            parameter.data = host_value.to(self.device, non_blocking=True)
        self.resident = True

    def offload(self) -> None:
        if not self.resident:
            return
        if self.host_backing is None:
            self.host_backing = tuple(
                parameter.detach().to("cpu", copy=True).pin_memory()
                for parameter in self.parameters
            )
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        for parameter, host_value in zip(
            self.parameters, self.host_backing, strict=True
        ):
            parameter.data = host_value
        self.resident = False
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _cuda_allocatable_bytes(device: torch.device) -> int:
    free_bytes, _ = torch.cuda.mem_get_info(device)
    reclaimable_bytes = max(
        torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device),
        0,
    )
    return free_bytes + reclaimable_bytes


class _FusedFirstProjection(nn.Module):
    fused: nn.Module
    def __init__(
        self,
        projections: tuple[nn.Linear, ...],
        output_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        if len(projections) != len(output_sizes):
            raise ValueError("fused projection metadata differs")
        first = projections[0]
        if any(
            projection.in_features != first.in_features
            or projection.bias is not None
            for projection in projections
        ):
            raise ValueError("fused projections must be compatible and bias-free")
        fused = nn.Linear(
            first.in_features,
            sum(output_sizes),
            bias=False,
            device=first.weight.device,
            dtype=first.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(
                torch.cat(
                    [projection.weight for projection in projections], dim=0
                )
            )
        self.fused = fused
        self.output_sizes = output_sizes
        self._cached: Tensor | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        self._cached = self.fused(inputs)
        return self._slice(0)

    def take(self, index: int) -> Tensor:
        return self._slice(index)

    def _slice(self, index: int) -> Tensor:
        if self._cached is None:
            raise RuntimeError("fused projection slices were called out of order")
        result = self._cached.narrow(
            -1,
            sum(self.output_sizes[:index]),
            self.output_sizes[index],
        )
        if index == len(self.output_sizes) - 1:
            self._cached = None
        return result


class _FusedProjectionSlice(nn.Module):
    _owner: _FusedFirstProjection

    def __init__(self, owner: _FusedFirstProjection, index: int) -> None:
        super().__init__()
        object.__setattr__(self, "_owner", owner)
        self.index = index

    def forward(self, _: Tensor) -> Tensor:
        return self._owner.take(self.index)


def fuse_llama_projections_(causal_lm: nn.Module) -> tuple[str, ...]:
    """Fuse QKV and gate/up projections into one GEMM per decoder layer."""

    layers = cast(Any, causal_lm.get_submodule("model")).layers
    fused_names: list[str] = []
    for index, layer in enumerate(layers):
        attention = layer.self_attn
        qkv = (attention.q_proj, attention.k_proj, attention.v_proj)
        if not all(isinstance(projection, nn.Linear) for projection in qkv):
            raise TypeError("QKV fusion requires merged linear projections")
        qkv_sizes = tuple(projection.out_features for projection in qkv)
        fused_qkv = _FusedFirstProjection(qkv, qkv_sizes)
        attention.q_proj = fused_qkv
        attention.k_proj = _FusedProjectionSlice(fused_qkv, 1)
        attention.v_proj = _FusedProjectionSlice(fused_qkv, 2)
        fused_names.append(f"model.layers.{index}.self_attn.qkv")

        mlp = layer.mlp
        gate_up = (mlp.gate_proj, mlp.up_proj)
        if not all(
            isinstance(projection, nn.Linear) for projection in gate_up
        ):
            raise TypeError("MLP fusion requires merged linear projections")
        gate_up_sizes = tuple(
            projection.out_features for projection in gate_up
        )
        fused_gate_up = _FusedFirstProjection(gate_up, gate_up_sizes)
        mlp.gate_proj = fused_gate_up
        mlp.up_proj = _FusedProjectionSlice(fused_gate_up, 1)
        fused_names.append(f"model.layers.{index}.mlp.gate_up")
    return tuple(fused_names)


def _copy_merged_lora_weight_(destination: Tensor, source: LoRALinear) -> None:
    destination.copy_(source.base.weight)
    destination.addmm_(
        source.lora_b.to(destination.dtype),
        source.lora_a.to(destination.dtype),
        alpha=source.scaling,
    )


@torch.no_grad()
def synchronize_fused_lora_policy_(
    destination: VAPOPolicy,
    source: VAPOPolicy,
) -> int:
    """Refresh a fused inference replica from the live actor LoRA."""

    if getattr(destination, "token_carry", False) != getattr(source, "token_carry", False):
        raise ValueError("rollout replica token-carry mode differs from the actor")
    if getattr(destination, "slot_memory", None) != getattr(source, "slot_memory", None):
        raise ValueError("rollout replica slot-memory geometry differs from the actor")

    source_layers = cast(Any, source.causal_lm.get_submodule("model")).layers
    destination_layers = cast(
        Any, destination.causal_lm.get_submodule("model")
    ).layers
    if len(source_layers) != len(destination_layers):
        raise ValueError("rollout replica layer count differs from the actor")

    synchronized = 0
    for source_layer, destination_layer in zip(
        source_layers, destination_layers, strict=True
    ):
        source_attention = source_layer.self_attn
        destination_attention = destination_layer.self_attn
        fused_qkv = destination_attention.q_proj
        if not isinstance(fused_qkv, _FusedFirstProjection):
            raise TypeError("rollout replica QKV projections are not fused")
        fused_qkv_weight = cast(nn.Linear, fused_qkv.fused).weight
        offset = 0
        for projection_name in ("q_proj", "k_proj", "v_proj"):
            source_projection = getattr(source_attention, projection_name)
            if not isinstance(source_projection, LoRALinear):
                raise TypeError("actor attention projection is not LoRA")
            size = source_projection.base.out_features
            _copy_merged_lora_weight_(
                fused_qkv_weight[offset : offset + size],
                source_projection,
            )
            offset += size
            synchronized += 1

        source_output = source_attention.o_proj
        destination_output = destination_attention.o_proj
        if not isinstance(source_output, LoRALinear) or not isinstance(
            destination_output, nn.Linear
        ):
            raise TypeError("attention output projection layout differs")
        _copy_merged_lora_weight_(destination_output.weight, source_output)
        synchronized += 1

        source_mlp = source_layer.mlp
        destination_mlp = destination_layer.mlp
        fused_gate_up = destination_mlp.gate_proj
        if not isinstance(fused_gate_up, _FusedFirstProjection):
            raise TypeError("rollout replica gate/up projections are not fused")
        fused_gate_up_weight = cast(nn.Linear, fused_gate_up.fused).weight
        offset = 0
        for projection_name in ("gate_proj", "up_proj"):
            source_projection = getattr(source_mlp, projection_name)
            if not isinstance(source_projection, LoRALinear):
                raise TypeError("actor MLP projection is not LoRA")
            size = source_projection.base.out_features
            _copy_merged_lora_weight_(
                fused_gate_up_weight[offset : offset + size],
                source_projection,
            )
            offset += size
            synchronized += 1

        source_down = source_mlp.down_proj
        destination_down = destination_mlp.down_proj
        if not isinstance(source_down, LoRALinear) or not isinstance(
            destination_down, nn.Linear
        ):
            raise TypeError("MLP output projection layout differs")
        _copy_merged_lora_weight_(destination_down.weight, source_down)
        synchronized += 1

    if getattr(source, "token_carry", False):
        modules = [("token_combiner", source.token_combiner, destination.token_combiner)]
        if getattr(source, "slot_memory", None) is not None:
            modules.append(("slot_head", source.slot_head, destination.slot_head))
        for name, source_module, destination_module in modules:
            destination_state = dict(destination_module.named_parameters())
            for parameter_name, parameter in source_module.named_parameters():
                if parameter_name not in destination_state:
                    raise ValueError(f"rollout replica {name} layout differs from the actor")
                destination_state[parameter_name].copy_(parameter)
                synchronized += 1

    return synchronized


class _CompactStaticLayer(StaticLayer):
    """Static cache with sequence-major backing for fixed-shape FA4 varlen decode."""

    def __init__(
        self,
        max_cache_len: int,
        sequence_lengths: Tensor,
        prefill_mask: Tensor,
        *,
        indexed_decode: bool = False,
    ) -> None:
        super().__init__(max_cache_len)
        self.sequence_lengths = sequence_lengths
        self.prefill_mask = prefill_mask
        self.prefilling = True
        self.indexed_decode = indexed_decode
        self.key_backing: Tensor
        self.value_backing: Tensor

    def lazy_initialization(
        self, key_states: Tensor, value_states: Tensor
    ) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size, self.num_heads = key_states.shape[:2]
        self.k_head_dim = key_states.shape[-1]
        self.v_head_dim = value_states.shape[-1]
        self.key_backing = torch.empty(
            (
                self.batch_size,
                self.max_cache_len,
                self.num_heads,
                self.k_head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        self.value_backing = torch.empty(
            (
                self.batch_size,
                self.max_cache_len,
                self.num_heads,
                self.v_head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        self.keys = self.key_backing.permute(0, 2, 1, 3)
        self.values = self.value_backing.permute(0, 2, 1, 3)
        self.cumulative_length = self.cumulative_length.to(self.device)
        self.is_initialized = True

    def reset(self) -> None:
        """Invalidate by length; stale KV beyond each sequence is never read."""
        self.cumulative_length.zero_()

    def _append_suffix(
        self, key_states: Tensor, value_states: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        # index_put is reinplaceable by Inductor; functional scatter is not.
        # Derive views here instead of capturing aliases of mutated backings.
        rows = torch.arange(key_states.size(0), device=key_states.device)[:, None]
        self.key_backing.index_put_((rows, positions), key_states.transpose(1, 2))
        self.value_backing.index_put_((rows, positions), value_states.transpose(1, 2))
        return self.key_backing.permute(0, 2, 1, 3), self.value_backing.permute(0, 2, 1, 3)

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        *args,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        width = key_states.shape[-2]
        if self.prefilling:
            valid = self.prefill_mask[:, :width]
            compact_positions = valid.cumsum(dim=1).sub(1)
            rows, source_positions = valid.nonzero(as_tuple=True)
            destination_positions = compact_positions[rows, source_positions]
            self.key_backing.index_put_(
                (rows, destination_positions),
                key_states.transpose(1, 2)[rows, source_positions],
            )
            self.value_backing.index_put_(
                (rows, destination_positions),
                value_states.transpose(1, 2)[rows, source_positions],
            )
            self.cumulative_length.copy_(self.sequence_lengths.max())
            return key_states, value_states

        safe_lengths = self.sequence_lengths.clamp_max(self.max_cache_len - 1)
        if self.indexed_decode:
            keys, values = self._append_suffix(
                key_states, value_states,
                safe_lengths[:, None],
            )
            self.cumulative_length.copy_(
                self.sequence_lengths.max().add(1).clamp_max(self.max_cache_len)
            )
            return keys, values
        key_indices = safe_lengths[:, None, None, None].expand(
            -1, 1, self.num_heads, self.k_head_dim
        )
        value_indices = safe_lengths[:, None, None, None].expand(
            -1, 1, self.num_heads, self.v_head_dim
        )
        self.key_backing.scatter_(1, key_indices, key_states.transpose(1, 2))
        self.value_backing.scatter_(
            1, value_indices, value_states.transpose(1, 2)
        )
        self.cumulative_length.copy_(
            self.sequence_lengths.max().add(1).clamp_max(self.max_cache_len)
        )
        if self.keys is None or self.values is None:
            raise RuntimeError("compact rollout cache was not initialized")
        return self.keys, self.values


def _split_kv_decode_geometry(config: Any) -> bool:
    """Whether the SM120 split-KV decode kernel serves this attention shape."""
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    return (
        int(config.num_attention_heads) == 16
        and int(config.num_key_value_heads) == 2
        and int(head_dim) == 128
    )


def _hybrid_fa4_mask(**kwargs):
    if kwargs["q_length"] == 1:
        return None
    from transformers.masking_utils import sdpa_mask

    return sdpa_mask(**kwargs)


@torch.compiler.disable
def _fixed_varlen_fa4_attention(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[Tensor, None]:
    if not getattr(module, "_rollout_cached_append", False):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        if attention_mask is not None:
            attention_mask = attention_mask[..., : key.shape[2]]
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    rollout = cast(Any, module)
    attention_output, _ = rollout._rollout_flash_varlen(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        max_seqlen_q=1,
        max_seqlen_k=rollout._rollout_max_cache_len,
        seqused_k=rollout._rollout_sequence_lengths,
        softmax_scale=scaling,
        causal=True,
        pack_gqa=True,
    )
    return attention_output, None

def build_fused_rollout_replica(
    source: VAPOPolicy,
) -> tuple[VAPOPolicy, tuple[str, ...]]:
    """Create the persistent merged policy used only for actor rollouts."""

    nextlat_head = source.nextlat_head
    cast(Any, source).nextlat_head = nn.Identity()
    try:
        replica = copy.deepcopy(source)
    finally:
        cast(Any, source).nextlat_head = nextlat_head
    flash_attn_varlen_func = cast(
        Any, import_module("flash_attn.cute.interface")
    ).flash_attn_varlen_func
    import transformers.modeling_flash_attention_utils as flash_utils

    flash_utils._flash_varlen_fn = flash_attn_varlen_func
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    implementation = "parameter_golf_fa4_decode"
    ALL_ATTENTION_FUNCTIONS.register(
        implementation, _fixed_varlen_fa4_attention
    )
    ALL_ATTENTION_FUNCTIONS.register(INVARIANT_ATTENTION, invariant_suffix_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(implementation, _hybrid_fa4_mask)
    replica.causal_lm.config._attn_implementation = implementation
    replica.causal_lm.gradient_checkpointing_disable()
    # Disabling checkpointing alone can leave the training embedding hook installed.
    replica.causal_lm.disable_input_require_grads()
    replica.causal_lm.config.use_cache = True
    merge_lora_for_inference(replica.causal_lm)
    fused = fuse_llama_projections_(replica.causal_lm)
    for parameter in replica.parameters():
        parameter.requires_grad_(False)
    replica.eval()
    synchronize_fused_lora_policy_(replica, source)
    return replica, fused




@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 16, "BLOCK_K": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 256}, num_warps=8),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_warps=8),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=8),
    ],
    key=["rows", "in_features", "out_features"],
)
@triton.jit
def _w8a16_linear_kernel(
    inputs,
    weight,
    weight_scale,
    output,
    rows: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    output_block = tl.program_id(0)
    row = tl.program_id(1)
    output_offsets = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    output_mask = output_offsets < out_features
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, in_features, BLOCK_K):
        input_offsets = start + tl.arange(0, BLOCK_K)
        input_mask = input_offsets < in_features
        activations = tl.load(
            inputs + row * in_features + input_offsets,
            mask=input_mask,
            other=0.0,
        ).to(tl.float32)
        weights = tl.load(
            weight
            + output_offsets[:, None] * in_features
            + input_offsets[None, :],
            mask=output_mask[:, None] & input_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(weights * activations[None, :], axis=1)
    scales = tl.load(
        weight_scale + output_offsets,
        mask=output_mask,
        other=0.0,
    )
    tl.store(
        output + row * out_features + output_offsets,
        accumulator * scales,
        mask=output_mask,
    )


@torch.library.triton_op(
    "parameter_golf::w8a16_linear",
    mutates_args={},
)
def w8a16_linear(
    inputs: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
) -> Tensor:
    if (
        inputs.ndim != 2
        or weight.ndim != 2
        or weight_scale.shape != (weight.shape[0], 1)
        or inputs.shape[1] != weight.shape[1]
    ):
        raise ValueError("W8A16 tensor dimensions are invalid")
    if (
        inputs.dtype != torch.bfloat16
        or weight.dtype != FP8_DTYPE
        or weight_scale.dtype != torch.float32
    ):
        raise TypeError("W8A16 tensor dtypes are invalid")
    if (
        not inputs.is_cuda
        or inputs.device != weight.device
        or inputs.device != weight_scale.device
    ):
        raise ValueError("W8A16 tensors must share one CUDA device")
    if (
        not inputs.is_contiguous()
        or not weight.is_contiguous()
        or not weight_scale.is_contiguous()
    ):
        raise ValueError("W8A16 tensors must be contiguous")
    rows, in_features = inputs.shape
    out_features = weight.shape[0]
    output = torch.empty(
        (rows, out_features), dtype=torch.bfloat16, device=inputs.device
    )

    def grid(meta):
        return (triton.cdiv(out_features, meta["BLOCK_N"]), rows)

    # A bare kernel name: AOTAutograd's cache key finds triton_op kernels by
    # that spelling, and misses one hidden behind a cast.
    wrap_triton(_w8a16_linear_kernel)[grid](
        inputs,
        weight,
        weight_scale,
        output,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    )
    return output


class W8A16Linear(nn.Module):
    """Row-scaled FP8 weights with BF16 activations for bandwidth-bound GEMV."""

    quantized_weight: Tensor
    weight_scale: Tensor

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        if linear.bias is not None:
            raise ValueError("W8A16 inference requires bias-free linears")
        weight = linear.weight.detach()
        scale = (
            weight.abs().amax(dim=1, keepdim=True).float() / FP8_MAX
        ).clamp_min(torch.finfo(torch.float32).tiny)
        quantized = (
            (weight.float() / scale)
            .clamp(-FP8_MAX, FP8_MAX)
            .to(FP8_DTYPE)
        )
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.register_buffer("quantized_weight", quantized)
        self.register_buffer("weight_scale", scale)

    def reconstructed_weight(self) -> Tensor:
        return self.quantized_weight.float() * self.weight_scale

    def forward(self, inputs: Tensor) -> Tensor:
        shape = inputs.shape[:-1]
        output = w8a16_linear(
            inputs.reshape(-1, self.in_features).contiguous(),
            self.quantized_weight,
            self.weight_scale,
        )
        return output.reshape(*shape, self.out_features)


def quantize_lm_head_w8a16_(causal_lm: nn.Module) -> None:
    lm_head = cast(Any, causal_lm).lm_head
    if not isinstance(lm_head, nn.Linear):
        raise TypeError("W8A16 requires an unquantized LM head")
    cast(Any, causal_lm).lm_head = W8A16Linear(lm_head)


def quantize_linear_layers_w8a16_(
    model: nn.Module,
) -> tuple[str, ...]:
    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    for name, module in replacements:
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            child_name = name
            parent = model
        setattr(parent, child_name, W8A16Linear(module))
    return tuple(name for name, _ in replacements)

@dataclass(frozen=True)
class FastDecodeStats:
    prefill_seconds: float
    decode_seconds: float
    target_decode_calls: int


def top_k_top_p_sample(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Tensor:
    """Sample a bounded nucleus, or the full categorical with top_k=-1/top_p=1."""

    if top_k == -1:
        if top_p != 1.0:
            raise ValueError("unrestricted top-k sampling requires top-p=1")
        return top_p_sample(logits, temperature, top_p)

    values, token_ids = logits.topk(top_k, dim=-1, sorted=True)
    probabilities = (values.float() / temperature).softmax(dim=-1)
    if top_p < 1.0:
        preceding_mass = probabilities.cumsum(dim=-1) - probabilities
        probabilities = torch.where(
            preceding_mass < top_p,
            probabilities,
            torch.zeros((), device=probabilities.device),
        )
        probabilities /= probabilities.sum(dim=-1, keepdim=True)
    selected = torch.multinomial(probabilities, 1)
    return token_ids.gather(1, selected).squeeze(1)


def selected_token_logprobs(logits: Tensor, token_ids: Tensor) -> Tensor:
    """Exact untempered policy log-probability for already selected tokens."""
    policy_logits = logits.float()
    return (
        policy_logits.gather(1, token_ids[:, None]).squeeze(1)
        - policy_logits.logsumexp(dim=-1)
    )


class FixedLengthInferenceEngine:
    """Inference-only fixed-length decoder with sampling fused into each target step."""

    def __init__(
        self,
        policy: VAPOPolicy,
        *,
        batch_size: int,
        cache_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        compile_decode: bool,
    ) -> None:
        from transformers import StaticCache

        if getattr(policy, "token_carry", False):
            raise ValueError("token carry requires CapturedTrainingRolloutEngine")
        if batch_size < 1 or cache_length < 2:
            raise ValueError("batch size and cache length must be positive")
        if temperature <= 0 or (top_k != -1 and top_k < 1) or not 0 < top_p <= 1:
            raise ValueError("sampling dimensions are invalid")
        if top_k == -1 and top_p != 1.0:
            raise ValueError("unrestricted top-k sampling requires top-p=1")
        config: Any = policy.causal_lm.config
        if top_k > int(config.vocab_size):
            raise ValueError("top-k exceeds the model vocabulary")

        self.policy = policy
        self.batch_size = batch_size
        self.cache_length = cache_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.decode_schedule = (
            "captured_static_step" if compile_decode else "eager_split"
        )
        device = next(policy.parameters()).device
        self.cache = StaticCache(
            config=config,
            max_batch_size=batch_size,
            max_cache_len=cache_length,
            device=device,
            dtype=torch.bfloat16,
        )
        self.cache_positions = torch.arange(cache_length, device=device)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self.output_position = torch.zeros(1, dtype=torch.long, device=device)
        self.position_ids = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=device
        )
        self.attention_mask = torch.zeros(
            (batch_size, cache_length), dtype=torch.bool, device=device
        )
        self.generated = torch.empty(
            (batch_size, cache_length), dtype=torch.long, device=device
        )
        self._graph_logits = torch.empty(
            (batch_size, int(config.vocab_size)),
            dtype=torch.bfloat16,
            device=device,
        )
        self._decode_graph: torch.cuda.CUDAGraph | None = None
        self._capture_stream = torch.cuda.Stream(device=device)
        self._compile_decode = compile_decode
        if hasattr(torch, "_dynamo"):
            for tensor in (
                self.cache_position,
                self.output_position,
                self.position_ids,
                self.attention_mask,
                self.generated,
                self._graph_logits,
            ):
                torch._dynamo.mark_static_address(tensor)

        def sample(logits: Tensor) -> Tensor:
            return top_k_top_p_sample(
                logits,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )

        def decode(token: Tensor) -> Tensor:
            self.generated.index_copy_(1, self.output_position, token[:, None])
            self.attention_mask.index_fill_(1, self.cache_position, True)
            hidden = policy.cached_hidden(
                token[:, None],
                past_key_values=self.cache,
                cache_position=self.cache_position,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
            )[:, -1]
            self.output_position.add_(1)
            self.cache_position.add_(1)
            self.position_ids.add_(1)
            return policy.logits(hidden)

        self.sample = (
            torch.compile(
                sample,
                fullgraph=True,
            )
            if compile_decode
            else sample
        )
        self.decode = (
            torch.compile(
                decode,
                fullgraph=False,
            )
            if compile_decode
            else decode
        )

    def _split_decode_step(self) -> None:
        token = self.sample(self._graph_logits)
        self._graph_logits.copy_(self.decode(token))

    def _capture_static_decode_schedule(self) -> None:
        torch.cuda.current_stream().synchronize()
        with torch.cuda.stream(self._capture_stream):
            self._split_decode_step()
        self._capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self._capture_stream):
            self._split_decode_step()
        torch.cuda.current_stream().wait_stream(self._capture_stream)
        self._decode_graph = graph

    def _run_decode_schedule(self, logits: Tensor, calls: int) -> Tensor:
        self._graph_logits.copy_(logits)
        if calls < 1:
            return self._graph_logits
        if not self._compile_decode:
            for _ in range(calls):
                self._split_decode_step()
            return self._graph_logits
        if self._decode_graph is None:
            if calls == 1:
                self._split_decode_step()
                return self._graph_logits
            self._capture_static_decode_schedule()
            calls -= 2
        if self._decode_graph is None:
            raise RuntimeError("static decode schedule capture failed")
        for _ in range(calls):
            self._decode_graph.replay()
        return self._graph_logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
    ) -> tuple[Tensor, FastDecodeStats]:
        if prompt_ids.ndim != 1 or prompt_ids.numel() < 1:
            raise ValueError("prompt must be a nonempty rank-one tensor")
        prompt_length = int(prompt_ids.numel())
        if max_new_tokens < 1 or prompt_length + max_new_tokens > self.cache_length:
            raise ValueError("prompt and response exceed the inference cache")

        device = self.generated.device
        prompt = prompt_ids.to(device)[None].expand(self.batch_size, -1)
        position_ids = torch.arange(prompt_length, device=device)[None].expand(
            self.batch_size, -1
        )
        self.cache.reset()
        self.attention_mask.zero_()
        self.attention_mask[:, :prompt_length] = True
        self.cache_position.fill_(prompt_length)
        self.output_position.zero_()
        self.position_ids.fill_(prompt_length)

        started = torch.cuda.Event(enable_timing=True)
        prefill_complete = torch.cuda.Event(enable_timing=True)
        decode_complete = torch.cuda.Event(enable_timing=True)
        started.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = self.policy.cached_hidden(
                prompt,
                past_key_values=self.cache,
                cache_position=self.cache_positions[:prompt_length],
                attention_mask=self.attention_mask,
                position_ids=position_ids,
            )[:, -1]
            logits = self.policy.logits(hidden)
            prefill_complete.record()
            logits = self._run_decode_schedule(
                logits, max_new_tokens - 1
            )
            final_token = self.sample(logits)
            self.generated.index_copy_(
                1, self.output_position, final_token[:, None]
            )
        decode_complete.record()
        decode_complete.synchronize()
        return (
            self.generated[:, :max_new_tokens].cpu(),
            FastDecodeStats(
                prefill_seconds=started.elapsed_time(prefill_complete) / 1_000.0,
                decode_seconds=prefill_complete.elapsed_time(decode_complete) / 1_000.0,
                target_decode_calls=max_new_tokens - 1,
            ),
        )


@dataclass(frozen=True)
class PromptPrefixBank:
    """Prompt KV reused when continuous lanes are refilled."""

    lengths: Tensor
    logits: Tensor
    values: Tensor
    layer_keys: Tensor
    layer_values: Tensor
    hidden: Tensor | None = None

    @property
    def prompts(self) -> int:
        return int(self.lengths.numel())


def _retire_inactive_flash_rows_(
    flash_sequence_lengths: Tensor,
    active: Tensor,
) -> None:
    """Keep completed static lanes from scanning their now-irrelevant KV history."""
    flash_sequence_lengths.masked_fill_(~active, 1)


def _take_refill_rows(
    free_slots: Sequence[int],
    *,
    pending_row: int,
    total_rows: int,
) -> tuple[list[int], int]:
    """Fill every available physical lane from a stable row request queue."""
    if pending_row < 0 or total_rows < 0:
        raise ValueError("rollout row counters must be nonnegative")
    remaining = max(total_rows - pending_row, 0)
    rows = min(len(free_slots), remaining)
    return list(free_slots[:rows]), rows

def _completion_poll_chunk(
    host_output_positions: Sequence[int],
    occupied_slots: Sequence[int],
    *,
    max_new_tokens: int,
    poll_steps: int,
    maximum_tokens_per_step: int = 1,
    response_limits: Sequence[int] | None = None,
) -> int:
    """Poll when a lane can first reach its known output limit."""
    if not occupied_slots or min(max_new_tokens, poll_steps, maximum_tokens_per_step) < 1:
        raise ValueError("completion polling dimensions must be positive")
    remaining = min(
        (max_new_tokens if response_limits is None else response_limits[slot])
        - host_output_positions[slot]
        for slot in occupied_slots
    )
    if remaining < 1:
        raise ValueError("occupied rollout lane already exhausted its token limit")
    remaining_steps = (remaining + maximum_tokens_per_step - 1) // maximum_tokens_per_step
    return min(poll_steps, remaining_steps)



class CapturedTrainingRolloutEngine:
    """Persistent fused rollout replica with a captured full-batch decode step.

    ``record_carry_history=False`` omits replay-only hidden history while
    preserving the recurrent carry state used to generate every token.
    """

    token_carry = False
    slot_memory: SlotMemoryConfig | None = None
    slot_state: SlotMemoryRolloutState | None = None

    _invariant_projections: tuple[InvariantLinear, ...] = ()
    invariant_decode: bool = False
    optimized_decode: bool = False

    def __init__(
        self,
        source_policy: VAPOPolicy,
        *,
        stop_ids: Sequence[int],
        prompts_per_rollout: int,
        samples_per_prompt: int,
        cache_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        compile_decode: bool,
        physical_batch_size: int | None = None,
        invariant_decode: bool = False,
        optimized_decode: bool = DEFAULT_OPTIMIZED_DECODE,
        answer_reserve_tokens: int = 0,
        thinking_end_token_id: int | None = None,
        record_carry_history: bool = True,
        capture_logprobs: bool = False,
    ) -> None:

        validate_thinking_budget(answer_reserve_tokens, thinking_end_token_id)
        self.answer_reserve_tokens = answer_reserve_tokens
        self.thinking_end_token_id = thinking_end_token_id
        logical_batch_size = prompts_per_rollout * samples_per_prompt
        if prompts_per_rollout < 1 or samples_per_prompt < 1:
            raise ValueError("rollout batch dimensions must be positive")
        if physical_batch_size is None:
            physical_batch_size = logical_batch_size
        if not 1 <= physical_batch_size <= logical_batch_size:
            raise ValueError(
                "physical rollout batch must fit the logical rollout batch"
            )
        if not stop_ids:
            raise ValueError("rollout stop ids cannot be empty")
        if temperature <= 0 or (top_k != -1 and top_k < 1) or not 0 < top_p <= 1:
            raise ValueError("sampling dimensions are invalid")
        if top_k == -1 and top_p != 1.0:
            raise ValueError("unrestricted top-k sampling requires top-p=1")
        if invariant_decode and not compile_decode:
            raise ValueError("invariant rollout requires compiled CUDA decode")
        self.token_carry = bool(getattr(source_policy, "token_carry", False))
        self.slot_memory = getattr(source_policy, "slot_memory", None)
        self.record_carry_history = record_carry_history
        if self.token_carry and invariant_decode:
            raise ValueError(
                "token-carry rollout does not support invariant/Uno decoding; "
                "use the ordinary optimized native rollout"
            )
        if self.token_carry and not compile_decode:
            raise ValueError("token-carry rollout requires compiled CUDA decode")
        if capture_logprobs and invariant_decode:
            raise ValueError("invariant/Uno decoding does not capture behavior log-probabilities")
        self.capture_logprobs = capture_logprobs

        self.source_policy = source_policy
        self.policy, self.fused_projection_groups = build_fused_rollout_replica(
            source_policy
        )
        self.invariant_decode = invariant_decode
        # An explicit eager request keeps the ordinary compatibility path.
        self.optimized_decode = invariant_decode or (optimized_decode and compile_decode)
        self.arithmetic = (
            INVARIANT_ARITHMETIC
            if invariant_decode
            else OPTIMIZED_ARITHMETIC if self.optimized_decode else LEGACY_ARITHMETIC
        )
        if invariant_decode:
            self._invariant_projections = install_invariant_linears(
                self.policy.causal_lm
            )
            self._pending = torch.zeros(
                physical_batch_size,
                dtype=torch.long,
                device=next(self.policy.parameters()).device,
            )
            self._prompt_last_tokens: Tensor | None = None
        self.prompts_per_rollout = prompts_per_rollout
        self.samples_per_prompt = samples_per_prompt
        self.batch_size = physical_batch_size
        self.cache_length = cache_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.stop_ids = tuple(int(token_id) for token_id in stop_ids)
        config: Any = self.policy.causal_lm.config
        if top_k > int(config.vocab_size):
            raise ValueError("top-k exceeds the model vocabulary")

        device = next(self.policy.parameters()).device
        self._runtime_device = device
        self._rollout_resident = True
        self._prepared_for_generation = False
        self._num_hidden_layers = int(config.num_hidden_layers)
        self.estimated_cache_bytes = (
            self.batch_size
            * cache_length
            * int(config.num_hidden_layers)
            * 2
            * int(config.num_key_value_heads)
            * int(config.head_dim)
            * torch.empty((), dtype=torch.bfloat16).element_size()
        )
        self._source_stash = _FrozenParameterStash(source_policy.causal_lm)
        self.offloaded_source_bytes = self._source_stash.bytes
        self.cache_positions = torch.arange(cache_length, device=device)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self.output_position = torch.zeros(
            self.batch_size, dtype=torch.long, device=device
        )
        self.active = torch.zeros(
            self.batch_size, dtype=torch.bool, device=device
        )
        self.thinking_closed = torch.zeros_like(self.active)
        self.response_limit = torch.ones(
            self.batch_size, dtype=torch.long, device=device
        )
        self.stop_tensor = torch.tensor(
            self.stop_ids, dtype=torch.long, device=device
        )
        self.position_ids = torch.zeros(
            (self.batch_size, 1), dtype=torch.long, device=device
        )
        self.attention_mask = torch.zeros(
            (self.batch_size, cache_length), dtype=torch.bool, device=device
        )
        self.sequence_lengths = torch.zeros(
            self.batch_size, dtype=torch.long, device=device
        )
        # This buffer crosses into the opaque FA4 operator; keep its
        # inference-mode identity from the warmup copies through recapture.
        with torch.inference_mode():
            self.flash_sequence_lengths = torch.zeros(
                self.batch_size, dtype=torch.int32, device=device
            )
        # Split-KV decode partitions every row's live prefix from the same
        # lengths in all layers. Plan once per step here; the attention layers
        # read these buffers instead of replanning 24 times per step.
        self._split_kv_metadata: tuple[Tensor, Tensor, Tensor] | None = None
        self._split_kv_offsets: Tensor | None = None
        self._split_kv_live: Tensor | None = None
        if (
            self.optimized_decode
            and not self.invariant_decode
            and device.type == "cuda"
            and torch.cuda.get_device_capability(device) == (12, 0)
            and _split_kv_decode_geometry(config)
        ):
            offsets_size, live_size = split_kv_plan_shapes(self.batch_size)
            with torch.inference_mode():
                self._split_kv_metadata = split_kv_metadata(
                    self.batch_size, cache_length, device
                )
                self._split_kv_offsets = torch.zeros(
                    offsets_size, dtype=torch.int32, device=device
                )
                self._split_kv_live = torch.zeros(
                    live_size, dtype=torch.int32, device=device
                )
        self.cache = self._new_cache()
        self.generated = torch.zeros(
            (self.batch_size, cache_length), dtype=torch.long, device=device
        )
        self.logprobs = torch.zeros(
            (self.batch_size, cache_length), dtype=torch.float32, device=device
        )
        self.values = torch.zeros_like(self.logprobs)
        self._graph_logits = torch.empty(
            (self.batch_size, int(config.vocab_size)),
            dtype=torch.bfloat16,
            device=device,
        )
        self._graph_values = torch.empty(
            self.batch_size, dtype=torch.float32, device=device
        )
        self.carry_hidden: Tensor | None = None
        self._carry_history: Tensor | None = None
        self.last_carry_hiddens: Tensor | None = None
        self.last_slot_choices: Tensor | None = None
        self.slot_state: SlotMemoryRolloutState | None = None
        # Written by the sampler, read by the same graph's advance: a forced
        # delimiter is an environment intervention and never writes a slot.
        self._forced_now = torch.zeros(self.batch_size, dtype=torch.bool, device=device)
        self._allocate_carry_storage()
        self._decode_graph: torch.cuda.CUDAGraph | None = None
        self._continuous_decode_graph: torch.cuda.CUDAGraph | None = None
        self._capture_stream = torch.cuda.Stream(device=device)
        self._compile_decode = compile_decode
        if hasattr(torch, "_dynamo"):
            for tensor in (
                self.cache_position,
                self.output_position,
                self.active,
                self.thinking_closed,
                self.response_limit,
                self.stop_tensor,
                self.position_ids,
                self.attention_mask,
                self.generated,
                self.sequence_lengths,
                self.flash_sequence_lengths,
                self.logprobs,
                self.values,
                self._graph_logits,
                self._graph_values,
                self._forced_now,
                *(self._split_kv_metadata or ()),
                *(
                    (self._split_kv_offsets, self._split_kv_live)
                    if self._split_kv_offsets is not None
                    else ()
                ),
            ):
                torch._dynamo.mark_static_address(tensor)

        def sample_tokens(logits: Tensor) -> Tensor:
            token = top_k_top_p_sample(
                logits,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            if self.answer_reserve_tokens:
                token, forced = force_thinking_end_(
                    token,
                    self.output_position,
                    self.thinking_closed,
                    self.active,
                    self.response_limit - self.answer_reserve_tokens - 1,
                    cast(int, self.thinking_end_token_id),
                )
                if self.slot_memory is not None:
                    self._forced_now.copy_(forced)
            return token

        def sample(logits: Tensor) -> tuple[Tensor, Tensor]:
            token = sample_tokens(logits)
            selected_logprob = selected_token_logprobs(logits, token)
            return token, selected_logprob

        def advance(
            token: Tensor,
            cache: Cache,
        ) -> Tensor:
            active = self.active
            safe_output = self.output_position.clamp_max(
                self.generated.size(1) - 1
            )[:, None]
            previous_tokens = self.generated.gather(1, safe_output).squeeze(1)
            actual = torch.where(active, token, previous_tokens)
            self.generated.scatter_(1, safe_output, actual[:, None])
            cache_rows = self.sequence_lengths.clamp_max(
                self.attention_mask.size(1) - 1
            )[:, None]
            previous_mask = self.attention_mask.gather(1, cache_rows)
            self.attention_mask.scatter_(
                1,
                cache_rows,
                torch.where(active[:, None], True, previous_mask),
            )
            self._plan_split_kv_()
            if self.token_carry:
                if self.record_carry_history:
                    # Save the producer of this action, before consuming its token.
                    # Match compact KV's reinplaceable index_put: functional
                    # scatter would copy the entire B x capacity x H bank.
                    # Inactive lanes preserve even their final capacity row.
                    history_rows = torch.arange(
                        self.batch_size, device=self.generated.device
                    )
                    history_positions = safe_output.squeeze(1)
                    previous_carry = self._carry_history[
                        history_rows, history_positions
                    ]
                    self._carry_history.index_put_(
                        (history_rows, history_positions),
                        torch.where(active[:, None], self.carry_hidden, previous_carry),
                    )
                if self.slot_memory is not None:
                    slot_state = cast(SlotMemoryRolloutState, self.slot_state)
                    mixed, slot_logprob = slot_state.step(
                        self.policy.token_combiner,
                        self.policy.slot_head,
                        token_embedding=self.policy.token_embeddings(actual),
                        producer_hidden=self.carry_hidden,
                        position=self.output_position,
                        active=active,
                        forced=self._forced_now,
                    )
                    # Joint action: the stored likelihood is log pi_tok + log pi_slot.
                    logprob_rows = (slot_state.rows, safe_output.squeeze(1))
                    self.logprobs.index_put_(
                        logprob_rows, self.logprobs[logprob_rows] + slot_logprob
                    )
                    inputs_embeds = mixed[:, None, :]
                else:
                    inputs_embeds = self.policy.carry_embeddings(
                        actual[:, None], self.carry_hidden[:, None, :]
                    )
                hidden = self.policy.cached_hidden(
                    inputs_embeds=inputs_embeds,
                    past_key_values=cache,
                    cache_position=self.cache_position,
                    attention_mask=self.attention_mask,
                    position_ids=self.position_ids,
                )[:, -1]
                self.carry_hidden.copy_(
                    torch.where(active[:, None], hidden, self.carry_hidden)
                )
            else:
                hidden = self.policy.cached_hidden(
                    actual[:, None],
                    past_key_values=cache,
                    cache_position=self.cache_position,
                    attention_mask=self.attention_mask,
                    position_ids=self.position_ids,
                )[:, -1]
            step = active.long()
            self.sequence_lengths.add_(step)
            self.flash_sequence_lengths.add_(step.to(torch.int32))
            self.flash_sequence_lengths.clamp_max_(self.cache_length)
            self.output_position.add_(step)
            self.position_ids.add_(step[:, None])
            stopped = active & (
                (actual[:, None] == self.stop_tensor[None]).any(dim=1)
            )
            exhausted = self.output_position >= self.response_limit
            self.active.logical_and_(~(stopped | exhausted))
            _retire_inactive_flash_rows_(
                self.flash_sequence_lengths,
                self.active,
            )
            return hidden

        def decode(
            token: Tensor,
            selected_logprob: Tensor,
            state_value: Tensor,
            cache: Cache,
        ) -> tuple[Tensor, Tensor]:
            safe_output = self.output_position.clamp_max(
                self.values.size(1) - 1
            )[:, None]
            previous_values = self.values.gather(1, safe_output).squeeze(1)
            actual_values = torch.where(
                self.active, state_value, previous_values
            )
            previous_logprobs = self.logprobs.gather(
                1, safe_output
            ).squeeze(1)
            actual_logprobs = torch.where(
                self.active, selected_logprob, previous_logprobs
            )
            self.logprobs.scatter_(
                1, safe_output, actual_logprobs[:, None]
            )
            self.values.scatter_(1, safe_output, actual_values[:, None])
            hidden = advance(token, cache)
            return self.policy.logits(hidden), self.policy.rollout_values(hidden)

        def decode_without_statistics(
            token: Tensor,
            cache: Cache,
        ) -> Tensor:
            hidden = advance(token, cache)
            return self.policy.logits(hidden)

        compile_forward = compile_invariant if self.optimized_decode else (
            lambda function: torch.compile(function, fullgraph=False)
        )
        self.sample_tokens = (
            torch.compile(sample_tokens, fullgraph=True)
            if compile_decode
            else sample_tokens
        )
        self.sample = (
            torch.compile(sample, fullgraph=True) if compile_decode else sample
        )
        self.decode = (
            compile_forward(decode)
            if compile_decode
            else decode
        )
        self.decode_without_statistics = (
            compile_forward(decode_without_statistics)
            if compile_decode
            else decode_without_statistics
        )
        if capture_logprobs:

            def commit_logprob(selected_logprob: Tensor) -> None:
                safe_output = self.output_position.clamp_max(
                    self.logprobs.size(1) - 1
                )[:, None]
                previous = self.logprobs.gather(1, safe_output).squeeze(1)
                self.logprobs.scatter_(
                    1,
                    safe_output,
                    torch.where(self.active, selected_logprob, previous)[:, None],
                )

            self._commit_logprob = (
                torch.compile(commit_logprob, fullgraph=True)
                if compile_decode
                else commit_logprob
            )
        if invariant_decode:

            def predict_pending() -> Tensor:
                hidden = self.policy.cached_hidden(
                    self._pending[:, None],
                    past_key_values=self.cache,
                    cache_position=self.cache_position,
                    position_ids=self.sequence_lengths[:, None],
                )[:, 0]
                return self.policy.logits(hidden)

            def commit_pending(token: Tensor) -> None:
                destination = self.output_position.clamp_max(
                    self.generated.size(1) - 1
                )[:, None]
                previous = self.generated.gather(1, destination).squeeze(1)
                self.generated.scatter_(
                    1, destination, torch.where(self.active, token, previous)[:, None]
                )
                self._pending.copy_(torch.where(self.active, token, self._pending))
                step = self.active.long()
                self.sequence_lengths.add_(step)
                self.output_position.add_(step)
                self.position_ids[:, 0].copy_(self.sequence_lengths)
                stopped = (token[:, None] == self.stop_tensor[None]).any(-1)
                self.active.logical_and_(
                    ~stopped & (self.output_position < self.response_limit)
                )

            self._predict_pending = compile_invariant(predict_pending)
            self._commit_pending = torch.compile(commit_pending, fullgraph=True)

    @property
    def trunk(self) -> Any:
        """The adapter this engine samples through.

        Part of the rollout-engine interface: the trainer reads geometry and
        identity from here rather than from a Hugging Face ``config``.
        """
        return self.policy.trunk

    def _new_cache(self) -> Cache:
        return Cache(
            layers=[
                _CompactStaticLayer(
                    self.cache_length,
                    self.sequence_lengths,
                    self.attention_mask,
                    indexed_decode=self.optimized_decode or self.invariant_decode,
                )
                for _ in range(self._num_hidden_layers)
            ]
        )

    def _allocate_carry_storage(self) -> None:
        if not self.token_carry:
            return
        hidden_size = int(self.policy.causal_lm.config.hidden_size)
        self.carry_hidden = torch.zeros(
            (self.batch_size, hidden_size),
            dtype=torch.bfloat16,
            device=self._runtime_device,
        )
        if self.record_carry_history:
            self._carry_history = torch.zeros(
                (self.batch_size, self.cache_length, hidden_size),
                dtype=torch.bfloat16,
                device=self._runtime_device,
            )
        if self.slot_memory is not None:
            self.slot_state = SlotMemoryRolloutState(
                self.slot_memory,
                batch_size=self.batch_size,
                capacity=self.cache_length,
                device=self._runtime_device,
            )
        if hasattr(torch, "_dynamo"):
            torch._dynamo.mark_static_address(self.carry_hidden)
            if self._carry_history is not None:
                torch._dynamo.mark_static_address(self._carry_history)
            if self.slot_state is not None:
                for tensor in self.slot_state.buffers():
                    torch._dynamo.mark_static_address(tensor)

    def _restore_rollout_cache(self) -> None:
        if self._rollout_resident:
            return
        self.cache = self._new_cache()
        self._allocate_carry_storage()
        self._rollout_resident = True

    def _validate_cache_capacity(self) -> None:
        available_bytes = _cuda_allocatable_bytes(self._runtime_device)
        resident_cache_bytes = sum(
            layer.key_backing.nbytes + layer.value_backing.nbytes
            for cache_layer in self.cache.layers
            if (layer := cast(_CompactStaticLayer, cache_layer)).is_initialized
        )
        additional_cache_bytes = self.estimated_cache_bytes - resident_cache_bytes
        required_bytes = additional_cache_bytes + ROLLOUT_WORKSPACE_RESERVE_BYTES
        if required_bytes > available_bytes:
            raise MemoryError(
                "captured rollout requires "
                f"{additional_cache_bytes / 2**30:.2f} GiB of additional KV plus "
                f"{ROLLOUT_WORKSPACE_RESERVE_BYTES / 2**30:.2f} GiB workspace, "
                f"but only {available_bytes / 2**30:.2f} GiB is allocatable "
                f"after offloading {self.offloaded_source_bytes / 2**30:.2f} GiB "
                "of frozen training weights"
            )

    def synchronize(self) -> None:
        self._source_stash.restore()
        synchronize_fused_lora_policy_(self.policy, self.source_policy)
        self._source_stash.offload()
        self._restore_rollout_cache()
        self._validate_cache_capacity()

    def prepare_generation(self) -> None:
        self.synchronize()
        self._prepared_for_generation = True

    def _synchronize_generation(self) -> None:
        if self._prepared_for_generation:
            self._prepared_for_generation = False
            return
        self.synchronize()

    def _bind_flash_cache(self) -> None:
        import transformers.modeling_flash_attention_utils as flash_utils

        self._set_invariant_decode(True)
        attention_layers = self.policy.causal_lm.model.layers
        if len(attention_layers) != len(self.cache.layers):
            raise RuntimeError("rollout policy and cache layer counts differ")
        for decoder_layer, cache_layer in zip(
            attention_layers, self.cache.layers
        ):
            if not isinstance(cache_layer, _CompactStaticLayer):
                raise TypeError("rollout cache layer layout differs")
            if not cache_layer.is_initialized:
                raise RuntimeError("rollout cache was not initialized by prefill")
            attention = cast(Any, decoder_layer.self_attn)
            cache_layer.prefilling = False
            attention._rollout_cached_append = True
            attention._rollout_flash_varlen = flash_utils._flash_varlen_fn
            attention._rollout_sequence_lengths = self.flash_sequence_lengths
            attention._rollout_max_cache_len = self.cache_length
            attention._rollout_optimized_decode = (
                self.optimized_decode and not self.invariant_decode
            )
            attention._rollout_split_kv_offsets = self._split_kv_offsets
            attention._rollout_split_kv_live = self._split_kv_live


    def release_cache(self) -> None:
        """Swap rollout KV for the frozen training backbone between phases."""
        self._prepared_for_generation = False
        if self._rollout_resident:
            self._decode_graph = None
            self._continuous_decode_graph = None
            for decoder_layer in self.policy.causal_lm.model.layers:
                attention = cast(Any, decoder_layer.self_attn)
                attention._rollout_sequence_lengths = None
                attention._rollout_split_kv_offsets = None
                attention._rollout_split_kv_live = None
            # Dynamo's ModelOutput bookkeeping can retain an obsolete Cache
            # until cyclic GC. Release its GPU payload at this phase boundary.
            self.cache.layers.clear()
            self.cache = self._new_cache()
            self.carry_hidden = None
            self._carry_history = None
            self.slot_state = None
            self._rollout_resident = False
            if self._compile_decode:
                # Traced layer/tensor aliases can also survive in unreachable
                # Dynamo cycles. Collect before allocating the restored actor.
                gc.collect()
        self._source_stash.restore()

    def _plan_split_kv_(self) -> None:
        """Refresh the shared split-KV partition plan for this decode step.

        Runs inside the compiled decode step before the trunk, from the same
        ``flash_sequence_lengths`` every layer's attention consumes, so the
        plan equals what each layer would have derived on its own.
        """
        if self._split_kv_metadata is None:
            return
        offsets, live = plan_split_kv(
            self.flash_sequence_lengths, *self._split_kv_metadata
        )
        cast(Tensor, self._split_kv_offsets).copy_(offsets)
        cast(Tensor, self._split_kv_live).copy_(live)

    def _set_invariant_decode(self, decoding: bool) -> None:
        if self.optimized_decode or self.invariant_decode:
            self.policy.causal_lm.config._attn_implementation = (
                INVARIANT_ATTENTION if decoding else "parameter_golf_fa4_decode"
            )
        for projection in self._invariant_projections:
            projection.decoding = decoding

    def _prepare_prompts(
        self, prompt_ids_cpu: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor, int]:
        self._set_invariant_decode(False)
        if len(prompt_ids_cpu) != self.prompts_per_rollout:
            raise ValueError("rollout prompt count differs from the configured batch")
        lengths = [prompt.numel() for prompt in prompt_ids_cpu]
        if not lengths or min(lengths) < 1:
            raise ValueError("rollout prompts cannot be empty")
        prompt_width = max(lengths)
        if prompt_width >= self.cache_length:
            raise ValueError("prompt exhausts the rollout cache")
        for layer in self.cache.layers:
            cast(_CompactStaticLayer, layer).prefilling = True
        for layer in self.policy.causal_lm.model.layers:
            cast(Any, layer.self_attn)._rollout_cached_append = False

        device = self.generated.device
        pad_token_id = int(self.policy.causal_lm.config.pad_token_id)
        prompt_batch = torch.full(
            (self.batch_size, prompt_width),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        prefill_position_ids = torch.zeros_like(prompt_batch)
        self.attention_mask.zero_()
        for group, (prompt, length) in enumerate(zip(prompt_ids_cpu, lengths)):
            row_start = group * self.samples_per_prompt
            row_stop = row_start + self.samples_per_prompt
            token_start = prompt_width - length
            prompt_batch[row_start:row_stop, token_start:].copy_(
                prompt.to(device)[None].expand(self.samples_per_prompt, -1)
            )
            self.attention_mask[row_start:row_stop, token_start:prompt_width] = True
            prefill_position_ids[row_start:row_stop, token_start:].copy_(
                torch.arange(length, device=device)[None].expand(
                    self.samples_per_prompt, -1
                )
            )
        repeated_lengths = torch.tensor(
            lengths, dtype=torch.long, device=device
        ).repeat_interleave(self.samples_per_prompt)
        self.position_ids[:, 0].copy_(repeated_lengths)
        self.sequence_lengths.copy_(repeated_lengths)
        self.flash_sequence_lengths.copy_(repeated_lengths.to(torch.int32).add_(1))
        return prompt_batch, prefill_position_ids, prompt_width

    def _split_decode_step(self) -> None:
        token, selected_logprob = self.sample(self._graph_logits)
        next_logits, next_values = self.decode(
            token, selected_logprob, self._graph_values, self.cache
        )
        self._graph_logits.copy_(next_logits)
        self._graph_values.copy_(next_values)

    def _capture_static_decode_schedule(self) -> None:
        torch.cuda.current_stream().synchronize()
        with torch.cuda.stream(self._capture_stream):
            self._split_decode_step()
        self._capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self._capture_stream):
            self._split_decode_step()
        torch.cuda.current_stream().wait_stream(self._capture_stream)
        self._decode_graph = graph

    def _run_decode_schedule(
        self,
        logits: Tensor,
        values: Tensor | None,
        calls: int,
        *,
        started: float,
        progress_callback: (
            Callable[[int, dict[str, float | int]], None] | None
        ),
    ) -> None:
        self._graph_logits.copy_(logits)
        collect_statistics = values is not None
        graph_name = (
            "_decode_graph"
            if collect_statistics
            else "_continuous_decode_graph"
        )
        if values is not None:
            self._graph_values.copy_(values)
        completed = 0
        if calls < 1:
            return
        if not self._compile_decode:
            for completed in range(1, calls + 1):
                if collect_statistics:
                    self._split_decode_step()
                else:
                    self._continuous_split_decode_step()
                self._report_progress(completed, started, progress_callback)
            return
        if getattr(self, graph_name) is None:
            if calls == 1:
                if collect_statistics:
                    self._split_decode_step()
                else:
                    self._continuous_split_decode_step()
                self._report_progress(1, started, progress_callback)
                return
            if collect_statistics:
                self._capture_static_decode_schedule()
            else:
                self._capture_continuous_decode_schedule()
            completed = 2
            self._report_progress(completed, started, progress_callback)
        graph = getattr(self, graph_name)
        if graph is None:
            raise RuntimeError("static decode schedule capture failed")
        for completed in range(completed + 1, calls + 1):
            graph.replay()
            self._report_progress(completed, started, progress_callback)

    def _report_progress(
        self,
        completed: int,
        started: float,
        progress_callback: Callable[[int, dict[str, float | int]], None] | None,
    ) -> None:
        if progress_callback is None or completed % 256:
            return
        wall_seconds = time.perf_counter() - started
        scheduled_tokens = completed * self.batch_size
        progress_callback(
            completed,
            {
                "decode_steps": completed,
                "batch_rows": self.batch_size,
                "prompt_groups": self.prompts_per_rollout,
                "scheduled_tokens": scheduled_tokens,
                "wall_seconds": wall_seconds,
                "scheduled_tokens_per_second": scheduled_tokens
                / max(wall_seconds, 1e-9),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(
                    self.generated.device
                ),
            },
        )

    @torch.inference_mode()
    def generate_prompts(
        self,
        prompt_ids_cpu: Sequence[Tensor],
        *,
        max_new_tokens: int,
        collect_statistics: bool = True,
        progress_callback: (
            Callable[[int, dict[str, float | int]], None] | None
        ) = None,
    ) -> tuple[Tensor, Tensor, Tensor, int, float, FastTrainingDecodeStats]:
        validate_thinking_budget(
            self.answer_reserve_tokens, self.thinking_end_token_id, max_new_tokens
        )
        started_wall = time.perf_counter()
        self._synchronize_generation()
        prompt_batch, prefill_position_ids, prompt_width = self._prepare_prompts(
            prompt_ids_cpu
        )
        if prompt_width + max_new_tokens > self.cache_length:
            raise ValueError("prompt and response exceed the rollout cache")
        self.cache.reset()
        self.cache_position.zero_()
        self.output_position.zero_()
        self.active.fill_(True)
        self.thinking_closed.zero_()
        self.response_limit.fill_(max_new_tokens)
        self.generated.zero_()
        self.logprobs.zero_()
        self.values.zero_()
        self.last_carry_hiddens = None
        self.last_slot_choices = None
        if self.token_carry and self.record_carry_history:
            self._carry_history.zero_()
        if self.slot_state is not None:
            self.slot_state.reset_all()
            self._forced_now.zero_()
        started = torch.cuda.Event(enable_timing=True)
        prefill_complete = torch.cuda.Event(enable_timing=True)
        decode_complete = torch.cuda.Event(enable_timing=True)
        started.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = self.policy.cached_hidden(
                prompt_batch,
                past_key_values=self.cache,
                cache_position=self.cache_positions[:prompt_width],
                attention_mask=self.attention_mask,
                position_ids=prefill_position_ids,
            )[:, -1]
            if self.token_carry:
                self.carry_hidden.copy_(hidden)
            self._bind_flash_cache()
            logits = self.policy.logits(hidden)
            if self.invariant_decode:
                # Prefill and decode have different numerical geometry. Recompute
                # the final prompt token through the same target as every suffix.
                self._pending.copy_(prompt_batch[:, -1])
                self.sequence_lengths.sub_(1)
                self.flash_sequence_lengths.sub_(1)
                if collect_statistics:
                    logits = self._predict_pending()
                    self.sequence_lengths.add_(1)
                    self.flash_sequence_lengths.add_(1)
            state_values = (
                self.policy.rollout_values(hidden) if collect_statistics else None
            )
            prefill_complete.record()
            self._run_decode_schedule(
                logits,
                state_values,
                max_new_tokens,
                started=started_wall,
                progress_callback=progress_callback,
            )
        decode_complete.record()
        decode_complete.synchronize()
        if progress_callback is not None:
            wall_seconds = time.perf_counter() - started_wall
            progress_callback(
                max_new_tokens,
                {
                    "decode_steps": max_new_tokens,
                    "batch_rows": self.batch_size,
                    "prompt_groups": self.prompts_per_rollout,
                    "scheduled_tokens": max_new_tokens * self.batch_size,
                    "wall_seconds": wall_seconds,
                    "scheduled_tokens_per_second": (
                        max_new_tokens * self.batch_size
                    )
                    / max(wall_seconds, 1e-9),
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(
                        self.generated.device
                    ),
                },
            )
        if self.token_carry and self.record_carry_history:
            # Replay needs ordinary detached tensors, not inference tensors,
            # and must own values independently of the next rollout/cache.
            with torch.inference_mode(False):
                self.last_carry_hiddens = self._carry_history[
                    :, :max_new_tokens
                ].detach().to(device="cpu", copy=True)
                if self.slot_state is not None:
                    self.last_slot_choices = self.slot_state.history[
                        :, :max_new_tokens
                    ].detach().to(device="cpu", copy=True)
        return (
            self.generated[:, :max_new_tokens].cpu(),
            self.logprobs[:, :max_new_tokens].cpu(),
            self.values[:, :max_new_tokens].cpu(),
            int(self.policy.causal_lm.config.vocab_size),
            self.top_p,
            FastTrainingDecodeStats(
                prefill_seconds=started.elapsed_time(prefill_complete) / 1_000.0,
                decode_seconds=prefill_complete.elapsed_time(decode_complete) / 1_000.0,
                target_decode_calls=max_new_tokens,
                target_decode_positions=max_new_tokens * self.batch_size,
            ),
        )

    @torch.inference_mode()
    def build_prompt_prefix_bank(
        self,
        prompt_ids_cpu: Sequence[Tensor],
        *,
        prefill_batch_prompts: int = 8,
    ) -> PromptPrefixBank:
        """Prefill every unique prompt once and retain compact KV on the host."""
        return self._build_prompt_prefix_bank(
            prompt_ids_cpu,
            prefill_batch_prompts=prefill_batch_prompts,
            collect_values=True,
            storage_device="cpu",
        )

    def _build_prompt_prefix_bank(
        self,
        prompt_ids_cpu: Sequence[Tensor],
        *,
        prefill_batch_prompts: int,
        collect_values: bool,
        storage_device: torch.device | str,
    ) -> PromptPrefixBank:
        self._set_invariant_decode(False)
        for layer in self.policy.causal_lm.model.layers:
            cast(Any, layer.self_attn)._rollout_cached_append = False
        if not prompt_ids_cpu:
            raise ValueError("prompt prefix bank cannot be empty")
        if prefill_batch_prompts < 1:
            raise ValueError("prefill_batch_prompts must be positive")
        lengths = torch.tensor(
            [int(prompt.numel()) for prompt in prompt_ids_cpu],
            dtype=torch.long,
        )
        if bool((lengths < 1).any()):
            raise ValueError("prompt prefix bank contains an empty prompt")
        prompt_width = int(lengths.max())
        if prompt_width >= self.cache_length:
            raise ValueError("prompt prefix bank exhausts the rollout cache")
        if self.invariant_decode:
            self._prompt_last_tokens = torch.stack(
                [prompt[-1] for prompt in prompt_ids_cpu]
            ).to(self._runtime_device)

        device = self.generated.device
        bank_device = torch.device(storage_device)
        if bank_device.type not in {"cpu", "cuda"}:
            raise ValueError("prompt prefix bank storage must be CPU or CUDA")
        if bank_device.type == "cuda" and bank_device != device:
            raise ValueError("CUDA prompt prefix bank must use the rollout device")
        host_resident = bank_device.type == "cpu"

        def empty_bank(shape: tuple[int, ...], *, dtype: torch.dtype) -> Tensor:
            if host_resident:
                return torch.empty(
                    shape,
                    dtype=dtype,
                    device="cpu",
                    pin_memory=True,
                )
            return torch.empty(shape, dtype=dtype, device=bank_device)

        pad_token_id = int(self.policy.causal_lm.config.pad_token_id)
        prompt_count = len(prompt_ids_cpu)
        bank_logits = empty_bank(
            (prompt_count, self._graph_logits.size(1)),
            dtype=self._graph_logits.dtype,
        )
        bank_values = empty_bank(
            (prompt_count,) if collect_values else (0,),
            dtype=self._graph_values.dtype,
        )
        bank_hidden = (
            empty_bank(
                (prompt_count, self.carry_hidden.size(1)),
                dtype=self.carry_hidden.dtype,
            )
            if self.token_carry
            else None
        )
        bank_keys: Tensor | None = None
        bank_layer_values: Tensor | None = None

        for start in range(0, prompt_count, prefill_batch_prompts):
            stop = min(start + prefill_batch_prompts, prompt_count)
            chunk_lengths_cpu = lengths[start:stop]
            chunk_lengths = chunk_lengths_cpu.to(device)
            rows = stop - start
            prompt_batch = torch.full(
                (rows, prompt_width),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (rows, prompt_width), dtype=torch.bool, device=device
            )
            position_ids = torch.zeros_like(prompt_batch)
            for row, (prompt, length_tensor) in enumerate(
                zip(
                    prompt_ids_cpu[start:stop],
                    chunk_lengths_cpu,
                    strict=True,
                )
            ):
                length = int(length_tensor)
                token_start = prompt_width - length
                prompt_batch[row, token_start:].copy_(prompt.to(device))
                attention_mask[row, token_start:] = True
                position_ids[row, token_start:].copy_(
                    torch.arange(length, device=device)
                )

            chunk_cache = Cache(
                layers=[
                    _CompactStaticLayer(
                        prompt_width,
                        chunk_lengths,
                        attention_mask,
                    )
                    for _ in self.cache.layers
                ]
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = self.policy.cached_hidden(
                    prompt_batch,
                    past_key_values=chunk_cache,
                    cache_position=self.cache_positions[:prompt_width],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[:, -1]
                if bank_hidden is not None:
                    bank_hidden[start:stop].copy_(
                        hidden, non_blocking=host_resident
                    )
                bank_logits[start:stop].copy_(
                    self.policy.logits(hidden),
                    non_blocking=host_resident,
                )
                if collect_values:
                    bank_values[start:stop].copy_(
                        self.policy.rollout_values(hidden),
                        non_blocking=host_resident,
                    )

            compact_layers = [
                cast(_CompactStaticLayer, layer) for layer in chunk_cache.layers
            ]
            if bank_keys is None or bank_layer_values is None:
                first_layer = compact_layers[0]
                if any(
                    layer.num_heads != first_layer.num_heads
                    or layer.k_head_dim != first_layer.k_head_dim
                    or layer.v_head_dim != first_layer.v_head_dim
                    for layer in compact_layers
                ):
                    raise ValueError(
                        "continuous prefix bank requires uniform KV dimensions"
                    )
                bank_keys = empty_bank(
                    (
                        prompt_count,
                        prompt_width,
                        len(compact_layers),
                        first_layer.num_heads,
                        first_layer.k_head_dim,
                    ),
                    dtype=first_layer.key_backing.dtype,
                )
                bank_layer_values = empty_bank(
                    (
                        prompt_count,
                        prompt_width,
                        len(compact_layers),
                        first_layer.num_heads,
                        first_layer.v_head_dim,
                    ),
                    dtype=first_layer.value_backing.dtype,
                )
            for layer_index, layer in enumerate(compact_layers):
                bank_keys[start:stop, :, layer_index].copy_(
                    layer.key_backing,
                    non_blocking=host_resident,
                )
                bank_layer_values[start:stop, :, layer_index].copy_(
                    layer.value_backing,
                    non_blocking=host_resident,
                )

        torch.cuda.current_stream(device).synchronize()
        if bank_keys is None or bank_layer_values is None:
            raise RuntimeError("prompt prefix bank was not initialized")
        return PromptPrefixBank(
            lengths=lengths,
            logits=bank_logits,
            values=bank_values,
            layer_keys=bank_keys,
            layer_values=bank_layer_values,
            hidden=bank_hidden,
        )

    def _ensure_continuous_cache(self, bank: PromptPrefixBank) -> None:
        device = self.generated.device
        for layer_index, cache_layer in enumerate(self.cache.layers):
            bank_keys = bank.layer_keys[:, :, layer_index]
            bank_values = bank.layer_values[:, :, layer_index]
            layer = cast(_CompactStaticLayer, cache_layer)
            if layer.is_initialized:
                continue
            key_template = torch.empty(
                (
                    self.batch_size,
                    bank_keys.size(2),
                    1,
                    bank_keys.size(3),
                ),
                dtype=bank_keys.dtype,
                device=device,
            )
            value_template = torch.empty(
                (
                    self.batch_size,
                    bank_values.size(2),
                    1,
                    bank_values.size(3),
                ),
                dtype=bank_values.dtype,
                device=device,
            )
            layer.lazy_initialization(key_template, value_template)
        self._bind_flash_cache()

    def _admit_prompt_rows(
        self,
        bank: PromptPrefixBank,
        prompt_index: int,
        slots: Sequence[int],
        *,
        max_new_tokens: int,
    ) -> None:
        if not slots:
            raise ValueError("admission requires at least one rollout lane")
        if self.token_carry and bank.hidden is None:
            raise ValueError("token-carry admission requires prompt final hidden states")
        device = self.generated.device
        slot_ids = torch.tensor(slots, dtype=torch.long, device=device)
        length = int(bank.lengths[prompt_index])
        self.attention_mask.index_fill_(0, slot_ids, False)
        self.attention_mask[slot_ids, :length] = True
        self.generated.index_fill_(0, slot_ids, 0)
        if bank.values.numel():
            self.logprobs.index_fill_(0, slot_ids, 0)
            self.values.index_fill_(0, slot_ids, 0)
        key_sources = bank.layer_keys[prompt_index, :length].to(
            device, non_blocking=True
        )
        value_sources = bank.layer_values[prompt_index, :length].to(
            device, non_blocking=True
        )
        for layer_index, cache_layer in enumerate(self.cache.layers):
            layer = cast(_CompactStaticLayer, cache_layer)
            layer.key_backing[slot_ids, :length] = key_sources[:, layer_index]
            layer.value_backing[slot_ids, :length] = value_sources[:, layer_index]
        self.sequence_lengths.index_fill_(0, slot_ids, length)
        self.flash_sequence_lengths.index_fill_(0, slot_ids, length + 1)
        self.position_ids.index_fill_(0, slot_ids, length)
        self.output_position.index_fill_(0, slot_ids, 0)
        # Carry history needs no capacity-sized clear on refill: every valid
        # action overwrites its producer row, and exports use this new length.
        self.response_limit.index_fill_(0, slot_ids, max_new_tokens)
        self._graph_logits[slot_ids] = bank.logits[prompt_index].to(
            device, non_blocking=True
        )
        if bank.values.numel():
            self._graph_values[slot_ids] = bank.values[prompt_index].to(
                device, non_blocking=True
            )
        if self.token_carry:
            self.carry_hidden[slot_ids] = cast(Tensor, bank.hidden)[prompt_index].to(
                device, non_blocking=True
            )
        if self.slot_state is not None:
            self.slot_state.reset_lanes(slot_ids)
            self._forced_now.index_fill_(0, slot_ids, False)
            # The joint log-prob accumulates in place; a re-admitted lane must
            # not inherit the previous episode's slot terms.
            self.logprobs.index_fill_(0, slot_ids, 0)
        self.active.index_fill_(0, slot_ids, True)
        self.thinking_closed.index_fill_(0, slot_ids, False)
        if self.invariant_decode:
            if self._prompt_last_tokens is None:
                raise RuntimeError("invariant admission requires prompt token identity")
            self.sequence_lengths.index_fill_(0, slot_ids, length - 1)
            self.flash_sequence_lengths.index_fill_(0, slot_ids, length)
            self.position_ids.index_fill_(0, slot_ids, length - 1)
            self._pending[slot_ids] = self._prompt_last_tokens[prompt_index]

    def _continuous_split_decode_step(self) -> None:
        if self.invariant_decode:
            self.sequence_lengths.masked_fill_(~self.active, 0)
            self.flash_sequence_lengths.copy_((self.sequence_lengths + 1).int())
            logits = self._predict_pending()
            self._commit_pending(self.sample_tokens(logits))
            return
        if self.capture_logprobs:
            # Same sampler and RNG draw as the token-only path; the exact
            # untempered replica log-probability lands at this output position
            # before ``advance`` moves it.
            token, selected_logprob = self.sample(self._graph_logits)
            self._commit_logprob(selected_logprob)
        else:
            token = self.sample_tokens(self._graph_logits)
        next_logits = self.decode_without_statistics(token, self.cache)
        self._graph_logits.copy_(next_logits)

    def _capture_continuous_decode_schedule(self) -> None:
        torch.cuda.current_stream().synchronize()
        with torch.cuda.stream(self._capture_stream):
            self._continuous_split_decode_step()
        self._capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self._capture_stream):
            self._continuous_split_decode_step()
        torch.cuda.current_stream().wait_stream(self._capture_stream)
        self._continuous_decode_graph = graph

    def _continuous_decode_once(self) -> None:
        if not self._compile_decode:
            self._continuous_split_decode_step()
            return
        if self._continuous_decode_graph is None:
            self._capture_continuous_decode_schedule()
            return
        self._continuous_decode_graph.replay()

    @torch.inference_mode()
    def generate_prompt_pool(
        self,
        prompt_ids_cpu: Sequence[Tensor],
        *,
        max_new_tokens: int,
        context_tokens: int | None = None,
        prefill_batch_prompts: int = 8,
        completion_poll_steps: int = 16,
        progress_callback: (
            Callable[[int, dict[str, float | int]], None] | None
        ) = None,
    ) -> ContinuousTrainingGeneration:
        """Generate an ordered logical pool over reusable physical lanes.

        Completion polling batches device-to-host synchronization. Completed
        lanes are refilled immediately after each poll.
        Responses include their first stop token or reach their own response
        limit; log-probabilities are exact replica values when the engine
        captures them, otherwise zero placeholders for replay-time refresh.
        """
        validate_thinking_budget(
            self.answer_reserve_tokens, self.thinking_end_token_id, max_new_tokens
        )
        if not prompt_ids_cpu:
            raise ValueError("continuous prompt pool cannot be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if completion_poll_steps < 1:
            raise ValueError("completion_poll_steps must be positive")
        prompt_lengths = [int(prompt.numel()) for prompt in prompt_ids_cpu]
        prompt_limits = response_token_limits(
            prompt_lengths, max_new_tokens=max_new_tokens,
            context_tokens=context_tokens,
            answer_reserve_tokens=self.answer_reserve_tokens,
            thinking_end_token_id=self.thinking_end_token_id,
        )
        if context_tokens is not None and context_tokens > self.cache_length:
            raise ValueError("context_tokens exceeds the allocated rollout cache")
        if any(
            length + limit > self.cache_length
            for length, limit in zip(prompt_lengths, prompt_limits, strict=True)
        ):
            raise ValueError("prompt pool and responses exceed the rollout cache")

        started = time.perf_counter()
        self._synchronize_generation()
        self.last_carry_hiddens = None
        self.last_slot_choices = None
        bank = self._build_prompt_prefix_bank(
            prompt_ids_cpu,
            prefill_batch_prompts=prefill_batch_prompts,
            collect_values=False,
            storage_device=self.generated.device,
        )
        self._ensure_continuous_cache(bank)
        for cache_layer in self.cache.layers:
            cast(_CompactStaticLayer, cache_layer).cumulative_length.zero_()
        self.cache_position.zero_()
        self.output_position.zero_()
        self.active.zero_()
        self.thinking_closed.zero_()
        self.response_limit.fill_(max_new_tokens)
        self.position_ids.zero_()
        self.sequence_lengths.zero_()
        self.flash_sequence_lengths.fill_(1)
        self.attention_mask.zero_()
        self.generated.zero_()
        self._graph_logits.zero_()
        self.logprobs.zero_()
        if self.token_carry:
            self.carry_hidden.zero_()
        if self.slot_state is not None:
            self.slot_state.reset_all()
            self._forced_now.zero_()
        if self._compile_decode and self._continuous_decode_graph is None:
            self._capture_continuous_decode_schedule()
            self._graph_logits.zero_()
        prefill_complete = time.perf_counter()

        prompt_count = len(prompt_ids_cpu)
        total_rows = prompt_count * self.samples_per_prompt
        completed_responses: list[Tensor | None] = [None] * total_rows
        completed_logprobs: list[Tensor | None] = [None] * total_rows
        completed_carries: list[Tensor | None] | None = (
            [None] * total_rows
            if self.token_carry and self.record_carry_history
            else None
        )
        completed_slot_choices: list[Tensor | None] | None = (
            [None] * total_rows
            if completed_carries is not None and self.slot_state is not None
            else None
        )
        slot_prompt = [-1] * self.batch_size
        slot_sample = [-1] * self.batch_size
        free_slots = list(range(self.batch_size))
        occupied_slots: list[int] = []
        pending_row = 0
        decode_steps = 0
        useful_tokens = 0
        admission_events = 0
        minimum_active_with_backlog = self.batch_size
        host_output_positions = [0] * self.batch_size
        host_response_limits = [max_new_tokens] * self.batch_size
        next_progress = 256

        def admit() -> None:
            nonlocal pending_row, free_slots, occupied_slots
            nonlocal admission_events
            selected, rows = _take_refill_rows(
                free_slots,
                pending_row=pending_row,
                total_rows=total_rows,
            )
            if not rows:
                return
            offset = 0
            while offset < rows:
                request_row = pending_row + offset
                prompt_index = request_row // self.samples_per_prompt
                prompt_stop = min(
                    rows,
                    (prompt_index + 1) * self.samples_per_prompt - pending_row,
                )
                slots = selected[offset:prompt_stop]
                self._admit_prompt_rows(
                    bank,
                    prompt_index,
                    slots,
                    max_new_tokens=prompt_limits[prompt_index],
                )
                for local_offset, slot in enumerate(slots, start=offset):
                    row = pending_row + local_offset
                    slot_prompt[slot] = row // self.samples_per_prompt
                    slot_sample[slot] = row % self.samples_per_prompt
                    host_output_positions[slot] = 0
                    host_response_limits[slot] = prompt_limits[prompt_index]
                offset = prompt_stop
            pending_row += rows
            selected_set = set(selected)
            free_slots = [slot for slot in free_slots if slot not in selected_set]
            occupied_slots = sorted((*occupied_slots, *selected))
            admission_events += 1

        while pending_row < total_rows or occupied_slots:
            admit()
            if pending_row < total_rows:
                minimum_active_with_backlog = min(
                    minimum_active_with_backlog, len(occupied_slots)
                )
            if not occupied_slots:
                raise RuntimeError("pending rollout rows cannot fit rollout lanes")
            decode_chunk = _completion_poll_chunk(
                host_output_positions,
                occupied_slots,
                max_new_tokens=max_new_tokens,
                poll_steps=completion_poll_steps,
                maximum_tokens_per_step=getattr(self, "_output_slots_per_cycle", 1),
                response_limits=host_response_limits,
            )
            for _ in range(decode_chunk):
                self._continuous_decode_once()
            decode_steps += decode_chunk
            slots = torch.tensor(
                occupied_slots,
                dtype=torch.long,
                device=self.generated.device,
            )
            status = torch.stack(
                (
                    self.active.index_select(0, slots).long(),
                    self.output_position.index_select(0, slots),
                ),
                dim=1,
            ).to("cpu").tolist()
            completed_slots: list[int] = []
            completed_lengths: list[int] = []
            completed_indices: list[int] = []
            for slot, (is_active, length) in zip(
                occupied_slots, status, strict=True
            ):
                host_output_positions[slot] = int(length)
                if is_active:
                    continue
                prompt_index = slot_prompt[slot]
                sample = slot_sample[slot]
                if prompt_index < 0 or sample < 0 or length < 1:
                    raise RuntimeError("completed rollout lane lost request identity")
                if length > prompt_limits[prompt_index]:
                    raise RuntimeError("completed rollout lane exceeded its response limit")
                completed_slots.append(slot)
                completed_lengths.append(int(length))
                completed_indices.append(
                    prompt_index * self.samples_per_prompt + sample
                )
                useful_tokens += int(length)
                slot_prompt[slot] = -1
                slot_sample[slot] = -1
            if completed_slots:
                completed_slot_ids = torch.tensor(
                    completed_slots,
                    dtype=torch.long,
                    device=self.generated.device,
                )
                max_completed_length = max(completed_lengths)
                completed_batch = self.generated.index_select(
                    0, completed_slot_ids
                )[:, :max_completed_length].to("cpu")
                completed_logprob_batch = (
                    self.logprobs.index_select(0, completed_slot_ids)[
                        :, :max_completed_length
                    ].to("cpu")
                    if self.capture_logprobs
                    else None
                )
                for position, (result_index, row, length) in enumerate(zip(
                    completed_indices,
                    completed_batch,
                    completed_lengths,
                    strict=True,
                )):
                    completed_responses[result_index] = row[:length].clone()
                    if completed_logprob_batch is not None:
                        completed_logprobs[result_index] = (
                            completed_logprob_batch[position, :length].clone()
                        )
                if completed_carries is not None:
                    # Slice before gathering so padding up to cache capacity
                    # is never copied. CPU clones own each logical request.
                    with torch.inference_mode(False):
                        carry_batch = self._carry_history[
                            :, :max_completed_length
                        ].index_select(0, completed_slot_ids).detach().to("cpu")
                        for result_index, row, length in zip(
                            completed_indices,
                            carry_batch,
                            completed_lengths,
                            strict=True,
                        ):
                            completed_carries[result_index] = row[:length].clone()
                        if completed_slot_choices is not None:
                            choice_batch = cast(SlotMemoryRolloutState, self.slot_state).history[
                                :, :max_completed_length
                            ].index_select(0, completed_slot_ids).detach().to("cpu")
                            for result_index, row, length in zip(
                                completed_indices,
                                choice_batch,
                                completed_lengths,
                                strict=True,
                            ):
                                completed_slot_choices[result_index] = row[:length].clone()
            if completed_slots:
                completed_set = set(completed_slots)
                occupied_slots = [
                    slot
                    for slot in occupied_slots
                    if slot not in completed_set
                ]
                free_slots = sorted((*free_slots, *completed_slots))
            admit()
            if pending_row < total_rows:
                minimum_active_with_backlog = min(
                    minimum_active_with_backlog, len(occupied_slots)
                )
            if progress_callback is not None and decode_steps >= next_progress:
                while next_progress <= decode_steps:
                    next_progress += 256
                wall_seconds = time.perf_counter() - started
                progress_callback(
                    decode_steps,
                    {
                        "decode_steps": decode_steps,
                        "batch_rows": self.batch_size,
                        "prompt_groups": prompt_count,
                        "completed_rows": sum(
                            response is not None
                            for response in completed_responses
                        ),
                        "active_rows": len(occupied_slots),
                        "pending_rows": total_rows - pending_row,
                        "pending_groups": (
                            total_rows
                            - pending_row
                            + self.samples_per_prompt
                            - 1
                        )
                        // self.samples_per_prompt,
                        "scheduled_tokens": (
                            decode_steps * self.batch_size
                            * getattr(self, "_output_slots_per_cycle", 1)
                        ),
                        "useful_completed_tokens": useful_tokens,
                        "wall_seconds": wall_seconds,
                        "scheduled_tokens_per_second": (
                            decode_steps * self.batch_size
                            * getattr(self, "_output_slots_per_cycle", 1)
                        )
                        / max(wall_seconds, 1e-9),
                        "peak_vram_bytes": torch.cuda.max_memory_allocated(
                            self.generated.device
                        ),
                    },
                )

        if any(response is None for response in completed_responses):
            raise RuntimeError("continuous rollout pool lost completed responses")
        if completed_carries is not None and any(
            carry is None for carry in completed_carries
        ):
            raise RuntimeError("continuous rollout pool lost completed carries")
        if completed_slot_choices is not None and any(
            choices is None for choices in completed_slot_choices
        ):
            raise RuntimeError("continuous rollout pool lost completed slot choices")
        responses = tuple(
            cast(Tensor, response) for response in completed_responses
        )
        placeholder_logprobs = tuple(
            cast(Tensor, completed_logprobs[index])
            if self.capture_logprobs
            else torch.zeros(response.numel(), dtype=torch.float32)
            for index, response in enumerate(responses)
        )
        return ContinuousTrainingGeneration(
            responses=responses,
            logprobs=placeholder_logprobs,
            response_limits=tuple(
                limit for limit in prompt_limits for _ in range(self.samples_per_prompt)
            ),
            prefill_seconds=prefill_complete - started,
            decode_seconds=time.perf_counter() - prefill_complete,
            decode_steps=decode_steps,
            useful_tokens=useful_tokens,
            capacity_row_steps=(
                decode_steps * self.batch_size
                * getattr(self, "_output_slots_per_cycle", 1)
            ),
            admission_events=admission_events,
            minimum_active_rows_with_backlog=minimum_active_with_backlog,
            carry_hiddens=(
                tuple(cast(Tensor, carry) for carry in completed_carries)
                if completed_carries is not None
                else None
            ),
            slot_choices=(
                tuple(cast(Tensor, choices) for choices in completed_slot_choices)
                if completed_slot_choices is not None
                else None
            ),
        )

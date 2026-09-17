"""Memory-efficient MiniCPM5 policy support for standard-token VAPO.

MiniCPM5 remains a native ``LlamaForCausalLM``. Actor and critic own disjoint
LoRA adapters and heads while aliasing the immutable pretrained parameters.
NextLat follows the reference residual dynamics model and objective geometry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
import math
from importlib import import_module
from typing import Any, Literal, cast
import numpy as np
from scipy.signal import lfilter

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from postraining.hf_runtime import prepare_text_only_transformers_runtime
from postraining.latent_thought import (
    CombinedEmbedding,
    GaussianTransitionHead,
    StopThinkingGate,
)
from postraining.token_carry import (
    TokenCarryCombiner,
    load_token_carry_state_dict,
    token_carry_replay_hidden,
)




MINICPM5_MODEL_ID = "openbmb/MiniCPM5-1B"
MINICPM5_REVISION = "87179e5c1f455ef22e6223592d2d61351b525bfc"
MINICPM5_VOCAB_SIZE = 130_560
TOKEN_ACTION = 0
FIRST_THOUGHT = 1
CONTINUE_THOUGHT = 2
STOP_THINKING = 3
FORCED_STOP_THINKING = 4
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS
    initialization: Literal["standard", "nora"] = "nora"

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if self.initialization not in {"standard", "nora"}:
            raise ValueError("LoRA initialization must be standard or nora")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("LoRA targets must be nonempty and unique")


class LoRALinear(nn.Module):
    """Frozen linear plus fp32-master low-rank update.

    CUDA callers run under bf16 autocast. Keeping the trainable matrices fp32
    avoids losing small AdamW updates while autocast still executes both small
    GEMMs in the model's compute dtype.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.rank = config.rank
        self.scaling = config.alpha / config.rank
        self.lora_a = nn.Parameter(
            torch.empty(config.rank, base.in_features, dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, config.rank, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        if config.initialization == "nora":
            with torch.no_grad():
                column_norms = torch.linalg.vector_norm(
                    self.lora_a, dim=0, keepdim=True
                )
                self.lora_a.div_(
                    column_norms.clamp_min(torch.finfo(self.lora_a.dtype).eps)
                )
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: Tensor) -> Tensor:
        if (
            inputs.dtype != self.lora_a.dtype
            and not torch.is_autocast_enabled(inputs.device.type)
        ):
            raise RuntimeError(
                "mixed-dtype LoRA requires autocast so fp32 masters are cast "
                "only inside the adapter GEMMs"
            )
        update = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.scaling


@torch.library.custom_op("parameter_golf::replay_silu_backward", mutates_args=())
def _replay_silu_backward(gradient: Tensor, inputs: Tensor) -> Tensor:
    # Keep the native fused derivative: decomposition changes bf16 rounding.
    return torch.ops.aten.silu_backward.default(gradient, inputs)


@_replay_silu_backward.register_fake
def _fake_replay_silu_backward(gradient: Tensor, inputs: Tensor) -> Tensor:
    return torch.empty_like(inputs)


class _ReplaySiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor) -> Tensor:
        ctx.save_for_backward(inputs)
        return F.silu(inputs)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> Tensor:
        return _replay_silu_backward(gradient, ctx.saved_tensors[0])


class _ReplaySiLU(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return _ReplaySiLUFunction.apply(inputs)


def enable_replay_mlp_compilation(causal_lm: nn.Module) -> None:
    """Compile replay MLPs without renaming parameters or changing SiLU backward."""
    from transformers.activations import SiLUActivation
    from postraining.invariant_linear import compile_invariant

    layers = cast(Any, causal_lm.get_submodule("model")).layers
    for layer in layers:
        if not isinstance(layer.mlp.act_fn, (nn.SiLU, SiLUActivation, _ReplaySiLU)):
            raise TypeError("compiled replay requires a SiLU MLP")
    for layer in layers:
        if (
            isinstance(layer.mlp.act_fn, _ReplaySiLU)
            and layer.mlp._compiled_call_impl is not None
        ):
            continue
        layer.mlp.act_fn = _ReplaySiLU()
        # nn.Module's compilation slot is excluded from deepcopy/serialization.
        # A forward closure would keep an inference replica bound to the actor.
        layer.mlp._compiled_call_impl = compile_invariant(layer.mlp._call_impl)


def inject_lora(model: nn.Module, config: LoRAConfig) -> tuple[str, ...]:
    """Freeze ``model`` and replace every requested Llama projection exactly once."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in config.targets:
            replacements.append((name, module))
    if not replacements:
        raise ValueError(f"none of the LoRA targets exist: {config.targets}")
    found_targets = {name.rsplit(".", 1)[-1] for name, _ in replacements}
    missing = set(config.targets) - found_targets
    if missing:
        raise ValueError(f"missing LoRA target modules: {sorted(missing)}")
    for name, module in replacements:
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            child_name = name
            parent = model
        setattr(parent, child_name, LoRALinear(module, config))
    return tuple(name for name, _ in replacements)


def share_frozen_parameters_(
    destination: nn.Module, source: nn.Module
) -> tuple[str, ...]:
    """Alias every immutable destination parameter to matching source storage."""

    source_parameters = dict(source.named_parameters())
    shared: list[str] = []
    for name, parameter in list(destination.named_parameters()):
        if parameter.requires_grad:
            continue
        source_parameter = source_parameters.get(name)
        if source_parameter is None:
            raise ValueError(f"shared backbone source is missing {name}")
        if source_parameter.requires_grad:
            raise ValueError(f"shared backbone source parameter is trainable: {name}")
        if (
            source_parameter.shape != parameter.shape
            or source_parameter.dtype != parameter.dtype
        ):
            raise ValueError(f"shared backbone parameter differs: {name}")
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = destination.get_submodule(parent_name)
        else:
            child_name = name
            parent = destination
        setattr(parent, child_name, source_parameter)
        shared.append(name)
    if not shared:
        raise ValueError("no immutable backbone parameters were shared")
    return tuple(shared)


def _packed_replay_mask(**_: Any) -> None:
    return None


@torch.compiler.disable
def _packed_replay_attention(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **_: Any,
) -> tuple[Tensor, None]:
    if query.shape[0] != 1 or attention_mask is not None or dropout:
        raise ValueError("packed replay attention requires one unpadded sequence batch")
    attention = cast(Any, module)
    boundaries = attention._packed_sequence_boundaries
    if boundaries is None:
        boundaries = tuple(int(item) for item in attention._packed_cu_seqlens.tolist())
    outputs = []
    for start, stop in zip(boundaries, boundaries[1:]):
        query_slice = query[:, :, start:stop]
        key_slice = key[:, :, start:stop]
        value_slice = value[:, :, start:stop]
        outputs.append(
            F.scaled_dot_product_attention(
                query_slice,
                key_slice,
                value_slice,
                dropout_p=0.0,
                is_causal=True,
                scale=scaling,
                enable_gqa=query_slice.shape[1] != key_slice.shape[1],
            )
        )
    return torch.cat(outputs, dim=2).transpose(1, 2), None


@torch.compiler.disable
def _packed_replay_attention_fa4(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **_: Any,
) -> tuple[Tensor, None]:
    if query.shape[0] != 1 or attention_mask is not None or dropout:
        raise ValueError("packed replay attention requires one unpadded sequence batch")
    attention = cast(Any, module)
    query_varlen = query.transpose(1, 2).squeeze(0).contiguous()
    key_varlen = key.transpose(1, 2).squeeze(0).contiguous()
    value_varlen = value.transpose(1, 2).squeeze(0).contiguous()
    output = attention._packed_flash_varlen(
        query_varlen,
        key_varlen,
        value_varlen,
        cu_seqlens_q=attention._packed_cu_seqlens,
        cu_seqlens_k=attention._packed_cu_seqlens,
        max_seqlen_q=attention._packed_max_sequence_length,
        max_seqlen_k=attention._packed_max_sequence_length,
        softmax_scale=scaling,
        causal=True,
        pack_gqa=True,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output.unsqueeze(0), None


def enable_packed_replay_attention(
    causal_lm: nn.Module, *, backend: str = "sdpa"
) -> None:
    """Install segmented SDPA or experimental FA4 for packed trajectories."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    implementations = {
        "sdpa": ("parameter_golf_sdpa_packed_replay", _packed_replay_attention),
        "fa4": ("parameter_golf_fa4_packed_replay", _packed_replay_attention_fa4),
    }
    if backend not in implementations:
        raise ValueError(f"unsupported packed replay attention backend: {backend}")
    implementation, attention_function = implementations[backend]
    ALL_ATTENTION_FUNCTIONS.register(implementation, attention_function)
    ALL_MASK_ATTENTION_FUNCTIONS.register(implementation, _packed_replay_mask)
    flash_varlen = (
        cast(Any, import_module("flash_attn.cute.interface")).flash_attn_varlen_func
        if backend == "fa4"
        else None
    )
    cast(Any, causal_lm).config._attn_implementation = implementation
    cast(Any, causal_lm)._packed_replay_attention_implementation = implementation
    for layer in cast(Any, causal_lm).model.layers:
        attention = cast(Any, layer.self_attn)
        attention._packed_flash_varlen = flash_varlen
        attention._packed_cu_seqlens = None
        attention._packed_sequence_boundaries = None
        attention._packed_max_sequence_length = 0


def use_packed_replay_attention(causal_lm: nn.Module, *, enabled: bool) -> None:
    """Switch a prepared model between packed replay and ordinary SDPA."""
    implementation = getattr(
        causal_lm, "_packed_replay_attention_implementation", None
    )
    if not isinstance(implementation, str):
        raise RuntimeError("packed replay attention was not installed")
    cast(Any, causal_lm).config._attn_implementation = (
        implementation if enabled else "sdpa"
    )


def merge_lora_for_inference(model: nn.Module) -> tuple[str, ...]:
    """Fold every LoRA update into its frozen base weight and remove the wrappers."""

    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    with torch.no_grad():
        for name, module in replacements:
            if torch.count_nonzero(module.lora_b):
                update = module.lora_b @ module.lora_a
                module.base.weight.add_(
                    update.to(module.base.weight.dtype), alpha=module.scaling
                )
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                child_name = name
                parent = model
            setattr(parent, child_name, module.base)
    return tuple(name for name, _ in replacements)


def adapter_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    }


def load_adapter_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    }
    if parameters.keys() != state.keys():
        missing = sorted(parameters.keys() - state.keys())
        unexpected = sorted(state.keys() - parameters.keys())
        raise ValueError(
            f"adapter state mismatch: missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


class ValueHead(nn.Module):
    """Scalar return readout owned exclusively by the critic."""

    def __init__(self, hidden_size: int, width: int = 256) -> None:
        super().__init__()
        if hidden_size < 1 or width < 1:
            raise ValueError("critic dimensions must be positive")
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6, dtype=torch.float32)
        self.input = nn.Linear(hidden_size, width, dtype=torch.float32)
        self.output = nn.Linear(width, 1, dtype=torch.float32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def features(self, hidden: Tensor) -> Tensor:
        return F.silu(self.input(self.norm(hidden.float())))

    def forward(self, hidden: Tensor) -> Tensor:
        return self.output(self.features(hidden)).squeeze(-1)


class NextLatAuxiliaryHead(nn.Module):
    """Reference NextLat residual dynamics MLP."""

    def __init__(self, hidden_size: int, projection_factor: float = 1.6) -> None:
        super().__init__()
        if hidden_size < 1 or not math.isfinite(projection_factor) or projection_factor <= 0:
            raise ValueError("NextLat dimensions must be positive")
        input_size = hidden_size * 2
        projected_size = max(
            128, 128 * round(projection_factor * input_size / 128)
        )
        self.hidden_size = hidden_size
        self.projection_factor = projection_factor
        self.norm = nn.LayerNorm(
            input_size,
            eps=1e-5,
            elementwise_affine=True,
            bias=False,
            dtype=torch.float32,
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_size, projected_size, bias=False, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(projected_size, projected_size, bias=False, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(projected_size, hidden_size, bias=False, dtype=torch.float32),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, hidden: Tensor, next_token_embedding: Tensor) -> Tensor:
        combined = torch.cat((next_token_embedding.float(), hidden.float()), dim=-1)
        delta = self.mlp(self.norm(combined))
        return hidden + delta.to(hidden.dtype)


def _thought_embeddings(side: Any, raw: Tensor) -> Tensor:
    if not side.latent_thinking:
        raise ValueError("thought embeddings require latent thinking")
    if raw.ndim != 2 or raw.shape[1] != int(side.causal_lm.config.hidden_size):
        raise ValueError("thought vectors must have shape [thoughts, hidden_size]")
    raw = raw.detach()
    dtype = side.causal_lm.get_input_embeddings().weight.dtype
    return side.thought_adapter(raw.to(dtype=dtype), raw)


def _replay_inputs(
    side: Any,
    input_ids: Tensor | None,
    inputs_embeds: Tensor | None,
    latent_vectors: Tensor | None,
    latent_input_positions: Tensor | None,
) -> dict[str, Tensor]:
    if (latent_vectors is None) != (latent_input_positions is None):
        raise ValueError("latent replay requires both vectors and input positions")
    if latent_vectors is not None:
        if inputs_embeds is not None or input_ids is None:
            raise ValueError("latent replay substitutes token inputs, not inputs_embeds")
        if not side.latent_thinking:
            raise ValueError("latent replay requires a latent-thinking model")
        assert latent_input_positions is not None
        if (
            latent_input_positions.ndim != 1
            or latent_input_positions.dtype != torch.long
            or latent_input_positions.numel() != latent_vectors.shape[0]
        ):
            raise ValueError("one packed input position is required per thought")
        # Token weights are frozen; discard the training-only input-gradient leaf
        # before in-place replacement, without detaching the thought adapter.
        embeddings = side.token_embeddings(input_ids).detach()
        thought_embeddings = side.thought_embeddings(latent_vectors)
        inputs_embeds = embeddings.flatten(0, 1).index_copy_(
            0, latent_input_positions, thought_embeddings
        ).view_as(embeddings)
    if inputs_embeds is not None:
        return {"inputs_embeds": inputs_embeds}
    if input_ids is None:
        raise ValueError("replay requires token ids or input embeddings")
    return {"input_ids": input_ids}


def _load_latent_state(side: Any, payload: dict[str, Any], *, actor: bool) -> None:
    enabled = payload.get("latent_thinking", False)
    if type(enabled) is not bool or enabled != side.latent_thinking:
        raise ValueError("checkpoint latent-thinking mode differs from the model")
    keys = {
        "thought_sigma", "init_stop_thinking_probability",
        "transition", "thinking_gate", "thought_adapter",
    }
    if not enabled:
        if keys.intersection(payload):
            raise ValueError("native checkpoint contains latent-thinking state")
        return
    modules = {"thought_adapter": side.thought_adapter}
    if actor:
        for key in ("thought_sigma", "init_stop_thinking_probability"):
            if payload.get(key) != getattr(side, key):
                raise ValueError(f"checkpoint {key} differs from the model")
        modules.update(transition=side.transition, thinking_gate=side.thinking_gate)
    elif (keys - {"thought_adapter"}).intersection(payload):
        raise ValueError("critic checkpoint contains actor latent-thinking state")
    for name, module in modules.items():
        if name not in payload:
            raise ValueError(f"checkpoint is missing {name}")
        module.load_state_dict(payload[name], strict=True)


class MiniCPMVAPOPolicy(nn.Module):
    """MiniCPM actor with its own LoRA and NextLat dynamics model."""

    def __init__(
        self,
        causal_lm: Any,
        lora_config: LoRAConfig,
        *,
        nextlat_projection_factor: float = 1.6,
        latent_thinking: bool = False,
        token_carry: bool = False,
        thought_sigma: float = 1.0,
        init_stop_thinking_probability: float = 0.9,
    ) -> None:
        super().__init__()
        if latent_thinking and token_carry:
            raise ValueError("token carry cannot be combined with latent thinking")
        self.causal_lm = causal_lm
        config: Any = causal_lm.config
        if getattr(config, "model_type", None) != "llama":
            raise ValueError("MiniCPM VAPO requires a standard LlamaForCausalLM")
        if int(config.vocab_size) != MINICPM5_VOCAB_SIZE:
            raise ValueError(
                f"unexpected MiniCPM vocabulary: {config.vocab_size} != "
                f"{MINICPM5_VOCAB_SIZE}"
            )
        self.lora_config = lora_config
        self.lora_modules = inject_lora(causal_lm, lora_config)
        self.nextlat_projection_factor = nextlat_projection_factor
        self.nextlat_head = NextLatAuxiliaryHead(
            int(config.hidden_size), nextlat_projection_factor
        )
        self.latent_thinking = latent_thinking
        self.token_carry = token_carry
        if token_carry:
            self.token_combiner = TokenCarryCombiner(int(config.hidden_size))
        if latent_thinking:
            self.thought_sigma = float(thought_sigma)
            self.init_stop_thinking_probability = float(init_stop_thinking_probability)
            self.transition = GaussianTransitionHead(int(config.hidden_size), thought_sigma)
            self.thinking_gate = StopThinkingGate(
                int(config.hidden_size), init_stop_thinking_probability
            )
            self.thought_adapter = CombinedEmbedding(int(config.hidden_size))
        self.model_id = MINICPM5_MODEL_ID
        self.revision = MINICPM5_REVISION
        self.causal_lm.config.use_cache = False

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = MINICPM5_MODEL_ID,
        revision: str = MINICPM5_REVISION,
        device: torch.device,
        lora_config: LoRAConfig,
        nextlat_projection_factor: float = 1.6,
        gradient_checkpointing: bool = True,
        latent_thinking: bool = False,
        token_carry: bool = False,
        thought_sigma: float = 1.0,
        init_stop_thinking_probability: float = 0.9,
    ) -> tuple["MiniCPMVAPOPolicy", Any]:
        prepare_text_only_transformers_runtime()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        loaded: Any = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        causal_lm = loaded.to(device)
        policy = cls(
            causal_lm,
            lora_config,
            nextlat_projection_factor=nextlat_projection_factor,
            latent_thinking=latent_thinking,
            token_carry=token_carry,
            thought_sigma=thought_sigma,
            init_stop_thinking_probability=init_stop_thinking_probability,
        ).to(device)
        policy.model_id = model_id
        policy.revision = revision
        if len(tokenizer) != int(causal_lm.config.vocab_size):
            raise ValueError("tokenizer and checkpoint vocabularies differ")
        if tokenizer.chat_template is None:
            raise ValueError("MiniCPM tokenizer is missing its native chat template")
        if gradient_checkpointing:
            causal_lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        return policy, tokenizer

    @property
    def lm_head_weight(self) -> Tensor:
        weight = self.causal_lm.lm_head.weight
        if weight.requires_grad:
            raise RuntimeError("the 130,560-token output head must remain frozen")
        return weight

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.causal_lm.named_parameters():
            if name.endswith(("lora_a", "lora_b")) and parameter.requires_grad:
                yield parameter
        if self.token_carry:
            yield from (
                parameter for parameter in self.token_combiner.parameters()
                if parameter.requires_grad
            )
        if self.latent_thinking:
            for module in (self.transition, self.thinking_gate, self.thought_adapter):
                yield from (parameter for parameter in module.parameters() if parameter.requires_grad)

    def token_embeddings(self, token_ids: Tensor) -> Tensor:
        return self.causal_lm.get_input_embeddings()(token_ids)

    def carry_embeddings(self, token_ids: Tensor, previous_hidden: Tensor) -> Tensor:
        if not self.token_carry:
            raise ValueError("carry embeddings require token carry")
        return self.token_combiner(self.token_embeddings(token_ids), previous_hidden)

    def token_carry_replay_hidden(self, batch: "ReplayMicrobatch") -> Tensor:
        return token_carry_replay_hidden(self, batch)

    def thought_embeddings(self, raw: Tensor) -> Tensor:
        return _thought_embeddings(self, raw)

    def nextlat_hidden(self, hidden: Tensor, token_ids: Tensor) -> Tensor:
        return self.nextlat_head(hidden, self.token_embeddings(token_ids))

    def replay_hidden(
        self,
        input_ids: Tensor | None,
        attention_mask: Tensor | None,
        *,
        position_ids: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        sequence_boundaries: tuple[int, ...] | None = None,
        max_sequence_length: int = 0,
        inputs_embeds: Tensor | None = None,
        latent_vectors: Tensor | None = None,
        latent_input_positions: Tensor | None = None,
    ) -> Tensor:
        if self.token_carry and inputs_embeds is None:
            raise ValueError("token carry requires token_carry_replay_hidden with stored carries")
        if cu_seqlens is not None:
            boundaries = sequence_boundaries
            if boundaries is None:
                boundaries = tuple(int(item) for item in cu_seqlens.tolist())
            for layer in self.causal_lm.model.layers:
                attention = cast(Any, layer.self_attn)
                attention._packed_cu_seqlens = cu_seqlens
                attention._packed_max_sequence_length = max_sequence_length
                attention._packed_sequence_boundaries = boundaries
        outputs = self.causal_lm.model(
            **_replay_inputs(self, input_ids, inputs_embeds, latent_vectors, latent_input_positions),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def cached_hidden(
        self,
        input_ids: Tensor | None = None,
        *,
        past_key_values: Any,
        cache_position: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        outputs = self.causal_lm.model(
            **_replay_inputs(self, input_ids, inputs_embeds, None, None),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def logits(self, hidden: Tensor) -> Tensor:
        return self.causal_lm.lm_head(hidden)

    def action_logprobs(
        self, action_hidden: Tensor, batch: "ReplayMicrobatch", *, chunk_tokens: int = 64
    ) -> Tensor:
        """Score each action using only its phase's distribution."""
        if action_hidden.ndim != 2 or action_hidden.shape[0] != batch.action_count:
            raise ValueError("one hidden state is required per replay action")
        if batch.action_kinds is None:
            if self.latent_thinking:
                raise ValueError("latent policy cannot replay native-token records")
            return chunked_frozen_head_logprobs(
                action_hidden, batch.targets, self.lm_head_weight, chunk_tokens=chunk_tokens
            )
        if not self.latent_thinking:
            raise ValueError("native policy cannot replay latent records")
        if (
            batch.latent_vectors is None or batch.latent_action_indices is None
            or batch.token_action_indices is None or batch.gate_action_indices is None
            or batch.gate_stop_actions is None
        ):
            raise ValueError("latent likelihood requires stored actions and phase indices")
        result = action_hidden.new_zeros(batch.action_count, dtype=torch.float32)
        token_indices = batch.token_action_indices
        if token_indices.numel():
            result = result.index_copy(
                0, token_indices,
                chunked_frozen_head_logprobs(
                    action_hidden[token_indices], batch.targets[token_indices],
                    self.lm_head_weight, chunk_tokens=chunk_tokens,
                ),
            )
        thought_indices = batch.latent_action_indices
        thought_hidden = action_hidden[thought_indices]
        mean = self.transition.predict_mean(thought_hidden)
        thought_logprobs = self.transition.log_prob(
            batch.latent_vectors.detach(), mean,
            self.transition.predict_log_sigma(thought_hidden),
        )
        result = result.index_copy(0, thought_indices, thought_logprobs)
        gate_indices = batch.gate_action_indices
        if gate_indices.numel():
            gate_logprobs = self.thinking_gate.log_prob(
                batch.gate_stop_actions, action_hidden[gate_indices]
            )
            result = result.index_add(0, gate_indices, gate_logprobs)
        return result

    def rollout_values(self, hidden: Tensor) -> Tensor:
        """Value placeholder; the independent critic refreshes returns post-rollout."""
        return torch.zeros(hidden.shape[:-1], dtype=torch.float32, device=hidden.device)


    def checkpoint_payload(self) -> dict[str, Any]:
        payload = {
            "model_id": self.model_id,
            "revision": self.revision,
            "lora_config": asdict(self.lora_config),
            "lora_modules": list(self.lora_modules),
            "nextlat_projection_factor": self.nextlat_projection_factor,
            "adapter": adapter_state_dict(self.causal_lm),
            "nextlat": {
                name: tensor.detach().cpu()
                for name, tensor in self.nextlat_head.state_dict().items()
            },
        }
        if self.token_carry:
            payload.update(
                token_carry=True,
                token_combiner={
                    name: tensor.detach().cpu()
                    for name, tensor in self.token_combiner.state_dict().items()
                },
            )
        if self.latent_thinking:
            payload.update(
                latent_thinking=True,
                thought_sigma=self.thought_sigma,
                init_stop_thinking_probability=self.init_stop_thinking_probability,
                transition={name: tensor.detach().cpu() for name, tensor in self.transition.state_dict().items()},
                thinking_gate={name: tensor.detach().cpu() for name, tensor in self.thinking_gate.state_dict().items()},
                thought_adapter={name: tensor.detach().cpu() for name, tensor in self.thought_adapter.state_dict().items()},
            )
        return payload

    def load_latent_state_dict(self, payload: dict[str, Any]) -> None:
        _load_latent_state(self, payload, actor=True)

    def load_token_carry_state_dict(self, payload: dict[str, Any]) -> None:
        load_token_carry_state_dict(self, payload)


class MiniCPMVAPOCritic(nn.Module):
    """Independent critic adapters over actor-shared immutable base weights."""

    def __init__(
        self,
        causal_lm: Any,
        lora_config: LoRAConfig,
        *,
        critic_width: int = 256,
        nextlat_projection_factor: float = 1.6,
        latent_thinking: bool = False,
        token_carry: bool = False,
    ) -> None:
        super().__init__()
        if latent_thinking and token_carry:
            raise ValueError("token carry cannot be combined with latent thinking")
        self.causal_lm = causal_lm
        config: Any = causal_lm.config
        if getattr(config, "model_type", None) != "llama":
            raise ValueError("MiniCPM critic requires a standard LlamaForCausalLM")
        if not isinstance(causal_lm.lm_head, nn.Linear):
            raise TypeError("MiniCPM critic output head layout differs")
        self.lora_config = lora_config
        self.lora_modules = inject_lora(causal_lm, lora_config)
        self.value_head = ValueHead(int(config.hidden_size), critic_width)
        self.nextlat_projection_factor = nextlat_projection_factor
        self.nextlat_head = NextLatAuxiliaryHead(
            int(config.hidden_size), nextlat_projection_factor
        )
        self.latent_thinking = latent_thinking
        self.token_carry = token_carry
        if token_carry:
            self.token_combiner = TokenCarryCombiner(int(config.hidden_size))
        if latent_thinking:
            self.thought_adapter = CombinedEmbedding(int(config.hidden_size))
        self.shared_frozen_parameters: tuple[str, ...] = ()
        self.model_id = MINICPM5_MODEL_ID
        self.revision = MINICPM5_REVISION
        self.causal_lm.config.use_cache = False

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = MINICPM5_MODEL_ID,
        revision: str = MINICPM5_REVISION,
        device: torch.device,
        lora_config: LoRAConfig,
        critic_width: int = 256,
        nextlat_projection_factor: float = 1.6,
        gradient_checkpointing: bool = True,
        shared_frozen_source: nn.Module | None = None,
        latent_thinking: bool = False,
        token_carry: bool = False,
    ) -> "MiniCPMVAPOCritic":
        prepare_text_only_transformers_runtime()
        from transformers import AutoModelForCausalLM

        loaded: Any = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        critic = cls(
            loaded,
            lora_config,
            critic_width=critic_width,
            nextlat_projection_factor=nextlat_projection_factor,
            latent_thinking=latent_thinking,
            token_carry=token_carry,
        )
        if shared_frozen_source is not None:
            critic.shared_frozen_parameters = share_frozen_parameters_(
                critic.causal_lm, shared_frozen_source
            )
        critic.to(device)
        critic.model_id = model_id
        critic.revision = revision
        if gradient_checkpointing:
            critic.causal_lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        return critic

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.causal_lm.named_parameters():
            if name.endswith(("lora_a", "lora_b")) and parameter.requires_grad:
                yield parameter
        if self.token_carry:
            yield from (
                parameter for parameter in self.token_combiner.parameters()
                if parameter.requires_grad
            )
        if self.latent_thinking:
            yield from (
                parameter for parameter in self.thought_adapter.parameters()
                if parameter.requires_grad
            )

    @property
    def lm_head_weight(self) -> Tensor:
        weight = self.causal_lm.lm_head.weight
        if weight.requires_grad:
            raise RuntimeError("the critic lexical grounding head must remain frozen")
        return weight

    def replay_hidden(
        self,
        input_ids: Tensor | None,
        attention_mask: Tensor | None,
        *,
        position_ids: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        sequence_boundaries: tuple[int, ...] | None = None,
        max_sequence_length: int = 0,
        inputs_embeds: Tensor | None = None,
        latent_vectors: Tensor | None = None,
        latent_input_positions: Tensor | None = None,
    ) -> Tensor:
        if self.token_carry and inputs_embeds is None:
            raise ValueError("token carry requires token_carry_replay_hidden with stored carries")
        if cu_seqlens is not None:
            boundaries = sequence_boundaries
            if boundaries is None:
                boundaries = tuple(int(item) for item in cu_seqlens.tolist())
            for layer in self.causal_lm.model.layers:
                attention = cast(Any, layer.self_attn)
                attention._packed_cu_seqlens = cu_seqlens
                attention._packed_max_sequence_length = max_sequence_length
                attention._packed_sequence_boundaries = boundaries
        outputs = self.causal_lm.model(
            **_replay_inputs(self, input_ids, inputs_embeds, latent_vectors, latent_input_positions),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def token_embeddings(self, token_ids: Tensor) -> Tensor:
        return self.causal_lm.get_input_embeddings()(token_ids)

    def carry_embeddings(self, token_ids: Tensor, previous_hidden: Tensor) -> Tensor:
        if not self.token_carry:
            raise ValueError("carry embeddings require token carry")
        return self.token_combiner(self.token_embeddings(token_ids), previous_hidden)

    def token_carry_replay_hidden(self, batch: "ReplayMicrobatch") -> Tensor:
        return token_carry_replay_hidden(self, batch)

    def thought_embeddings(self, raw: Tensor) -> Tensor:
        return _thought_embeddings(self, raw)

    def values(self, hidden: Tensor) -> Tensor:
        return self.value_head(hidden)

    def checkpoint_payload(self) -> dict[str, Any]:
        payload = {
            "model_id": self.model_id,
            "revision": self.revision,
            "lora_config": asdict(self.lora_config),
            "lora_modules": list(self.lora_modules),
            "nextlat_projection_factor": self.nextlat_projection_factor,
            "adapter": adapter_state_dict(self.causal_lm),
            "value_head": {
                name: tensor.detach().cpu()
                for name, tensor in self.value_head.state_dict().items()
            },
            "nextlat": {
                name: tensor.detach().cpu()
                for name, tensor in self.nextlat_head.state_dict().items()
            },
        }
        if self.token_carry:
            payload.update(
                token_carry=True,
                token_combiner={
                    name: tensor.detach().cpu()
                    for name, tensor in self.token_combiner.state_dict().items()
                },
            )
        if self.latent_thinking:
            payload.update(
                latent_thinking=True,
                thought_adapter={name: tensor.detach().cpu() for name, tensor in self.thought_adapter.state_dict().items()},
            )
        return payload

    def load_latent_state_dict(self, payload: dict[str, Any]) -> None:
        _load_latent_state(self, payload, actor=False)

    def load_token_carry_state_dict(self, payload: dict[str, Any]) -> None:
        load_token_carry_state_dict(self, payload)


class _ChunkedFrozenHeadLogProbs(torch.autograd.Function):
    """Exact selected-token log-probabilities without retaining full logits."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden: Tensor,
        targets: Tensor,
        weight: Tensor,
        chunk_tokens: int,
    ) -> Tensor:
        if hidden.ndim != 2 or weight.ndim != 2:
            raise ValueError("hidden and output weight must be matrices")
        if targets.shape != hidden.shape[:1] or targets.dtype != torch.long:
            raise ValueError("targets must be int64 with one id per hidden row")
        if hidden.shape[1] != weight.shape[1]:
            raise ValueError("hidden width and output-head width differ")
        if weight.requires_grad:
            raise ValueError("chunked output head must be frozen")
        if chunk_tokens < 1:
            raise ValueError("logit chunk size must be positive")
        if targets.device.type == "cpu" and targets.numel() and (
            int(targets.min()) < 0 or int(targets.max()) >= weight.shape[0]
        ):
            raise ValueError("target token lies outside the vocabulary")

        ctx.save_for_backward(hidden, targets, weight)
        ctx.chunk_tokens = chunk_tokens
        result = torch.empty(hidden.shape[0], device=hidden.device, dtype=torch.float32)
        for start in range(0, hidden.shape[0], chunk_tokens):
            stop = min(start + chunk_tokens, hidden.shape[0])
            logits = F.linear(hidden[start:stop], weight).float()
            result[start:stop] = logits.gather(
                1, targets[start:stop, None]
            ).squeeze(1) - logits.logsumexp(dim=1)
        return result


    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor):
        if len(grad_outputs) != 1:
            raise RuntimeError("chunked log-probability backward expects one gradient")
        grad_output = grad_outputs[0]
        hidden, targets, weight = ctx.saved_tensors
        grad_hidden = torch.empty_like(hidden)
        for start in range(0, hidden.shape[0], ctx.chunk_tokens):
            stop = min(start + ctx.chunk_tokens, hidden.shape[0])
            logits = F.linear(hidden[start:stop], weight).float()
            probabilities = -logits.softmax(dim=1)
            del logits
            probabilities[
                torch.arange(stop - start, device=hidden.device),
                targets[start:stop],
            ] += 1.0
            probabilities.mul_(grad_output[start:stop, None].float())
            grad_hidden[start:stop] = F.linear(
                probabilities.to(weight.dtype), weight.transpose(0, 1)
            ).to(hidden.dtype)
        return grad_hidden, None, None, None


def chunked_frozen_head_logprobs(
    hidden: Tensor,
    targets: Tensor,
    weight: Tensor,
    *,
    chunk_tokens: int,
) -> Tensor:
    return _ChunkedFrozenHeadLogProbs.apply(hidden, targets, weight, chunk_tokens)


@dataclass(frozen=True)
class SamplingStats:
    scanned_vocabulary: int
    nucleus_mass_lower_bound: float


def _nucleus_membership(
    scaled_logits: Tensor,
    probabilities: Tensor,
    sampled: Tensor,
    top_p: float,
) -> Tensor:
    """Return membership in a deterministic descending-logit nucleus.

    Ties are ordered by ascending token id. A token belongs to the nucleus
    exactly when the probability mass preceding it in that total order is
    below ``top_p``.
    """
    vocabulary_ids = torch.arange(
        scaled_logits.shape[1], device=scaled_logits.device
    )
    thresholds = scaled_logits.gather(1, sampled[:, None])
    precedes = (scaled_logits > thresholds) | (
        (scaled_logits == thresholds)
        & (vocabulary_ids[None] < sampled[:, None])
    )
    preceding_mass = torch.where(
        precedes, probabilities, torch.zeros((), device=probabilities.device)
    ).sum(dim=-1)
    return preceding_mass < top_p


@torch.no_grad()
def exact_top_p_sample(
    logits: Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, SamplingStats]:
    """Sample an exact nucleus without sorting the 130,560-token vocabulary.

    Proposals come from the full temperature-scaled categorical distribution.
    Rejecting proposals outside the deterministic nucleus samples exactly from
    that distribution conditioned on the nucleus. Its acceptance probability
    is at least ``top_p``; at the configured 0.95 it needs about 1.053 proposals
    on average while replacing repeated adaptive top-k sorts with linear GPU
    reductions.
    """
    if logits.ndim != 2:
        raise ValueError("sampling logits must be [batch, vocabulary]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")

    policy_logits = logits.float()
    scaled_logits = policy_logits.div(temperature)
    probabilities = scaled_logits.softmax(dim=-1)
    sampled = torch.empty(
        logits.shape[0], dtype=torch.long, device=logits.device
    )
    pending = torch.ones(
        logits.shape[0], dtype=torch.bool, device=logits.device
    )
    rows = torch.arange(logits.shape[0], device=logits.device)
    while True:
        proposals = torch.multinomial(
            probabilities.index_select(0, rows),
            1,
            generator=generator,
        ).squeeze(1)
        accepted = (
            torch.ones_like(proposals, dtype=torch.bool)
            if top_p == 1.0
            else _nucleus_membership(
                scaled_logits.index_select(0, rows),
                probabilities.index_select(0, rows),
                proposals,
                top_p,
            )
        )
        accepted_rows = rows[accepted]
        sampled[accepted_rows] = proposals[accepted]
        pending[accepted_rows] = False
        if not bool(pending.any()):
            break
        rows = pending.nonzero(as_tuple=False).squeeze(1)
    selected = (
        policy_logits.gather(1, sampled[:, None]).squeeze(1)
        - policy_logits.logsumexp(dim=-1)
    )
    return (
        sampled,
        selected,
        SamplingStats(
            scanned_vocabulary=logits.shape[1],
            nucleus_mass_lower_bound=top_p,
        ),
    )

@torch.no_grad()
def dense_top_p_probabilities(
    logits: Tensor,
    *,
    temperature: float,
    top_p: float,
) -> Tensor:
    """Materialize an exact nucleus distribution with adaptive top-k search.

    The top-k search only locates the boundary logit. Membership is then
    reconstructed over the full vocabulary, including ascending-token-id tie
    handling, so the result exactly matches :func:`_nucleus_membership`.
    """
    if logits.ndim != 2:
        raise ValueError("sampling logits must be [batch, vocabulary]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")
    scaled = logits.float().div(temperature)
    probabilities = scaled.softmax(dim=-1)
    if top_p == 1.0:
        return probabilities

    vocabulary = scaled.shape[1]
    search_width = min(256, vocabulary)
    while True:
        top_logits, top_ids = torch.topk(
            scaled, search_width, dim=-1, sorted=True
        )
        top_probabilities = probabilities.gather(1, top_ids)
        cumulative = top_probabilities.cumsum(dim=-1)
        if bool((cumulative[:, -1] >= top_p).all()):
            break
        if search_width == vocabulary:
            raise RuntimeError("top-p boundary search failed")
        search_width = min(2 * search_width, vocabulary)

    boundary_indices = (cumulative < top_p).sum(dim=-1).clamp_max(
        search_width - 1
    )
    boundary_logits = top_logits.gather(
        1, boundary_indices[:, None]
    ).squeeze(1)
    strictly_higher = scaled > boundary_logits[:, None]
    equal_boundary = scaled == boundary_logits[:, None]
    higher_mass = torch.where(
        strictly_higher, probabilities, torch.zeros_like(probabilities)
    ).sum(dim=-1)
    boundary_probability = top_probabilities.gather(
        1, boundary_indices[:, None]
    ).squeeze(1)
    needed_ties = torch.ceil(
        (top_p - higher_mass).clamp_min(0.0)
        / boundary_probability.clamp_min(torch.finfo(torch.float32).tiny)
    ).long().clamp_min_(1)
    tie_rank = equal_boundary.long().cumsum(dim=-1)
    membership = strictly_higher | (
        equal_boundary & (tie_rank <= needed_ties[:, None])
    )
    probabilities.masked_fill_(~membership, 0.0)
    probabilities.div_(probabilities.sum(dim=-1, keepdim=True))
    return probabilities


@torch.no_grad()
def maximal_coupling_verify(
    target_probabilities: Tensor,
    draft_probabilities: Tensor,
    proposals: Tensor,
    active: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Verify independent proposals and directly sample rejected residuals."""
    if target_probabilities.shape != draft_probabilities.shape:
        raise ValueError("target and draft distributions must have equal shapes")
    if proposals.shape != active.shape:
        raise ValueError("proposal and active masks must have equal shapes")
    target = target_probabilities.gather(1, proposals[:, None]).squeeze(1)
    draft = draft_probabilities.gather(1, proposals[:, None]).squeeze(1)
    accepted = (
        torch.rand(
            active.shape,
            device=target.device,
            generator=generator,
        )
        < (target / draft.clamp_min(1e-30)).clamp_max(1.0)
    ) & active
    rejected = active & ~accepted
    residual = (
        target_probabilities - draft_probabilities
    ).clamp_min_(0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    residual.div_(residual_mass.clamp_min(1e-30))
    residual = torch.where(
        residual_mass <= 0,
        target_probabilities,
        residual,
    )
    corrections = torch.multinomial(
        residual,
        1,
        generator=generator,
    ).squeeze(1)
    committed = torch.where(rejected, corrections, proposals)
    return committed, accepted

def _precompute_advantages(old_values: Tensor, correct: bool) -> Tensor:
    """Vectorized VAPO GAE for one terminal-reward trajectory."""
    values = old_values.detach().float().cpu().numpy()
    length = values.size
    if length < 1:
        raise ValueError("trajectory values cannot be empty")
    horizon = max(0.05 * length, min(float(length), 20.0))
    policy_lambda = min(max(1.0 - 1.0 / horizon, 0.0), 1.0)
    terminal_reward = 1.0 if correct else -1.0
    deltas = np.empty(length, dtype=np.float32)
    if length > 1:
        deltas[:-1] = values[1:] - values[:-1]
    deltas[-1] = terminal_reward - values[-1]
    reversed_advantages = np.asarray(
        lfilter([1.0], [1.0, -policy_lambda], deltas[::-1]),
        dtype=np.float32,
    )
    return torch.from_numpy(reversed_advantages[::-1].copy())


@dataclass(frozen=True)
class TrajectoryRecord:
    """Compact host replay record; no vocabulary-sized tensor is retained."""

    token_ids: Tensor
    prompt_length: int
    old_logprobs: Tensor
    advantages: Tensor
    correct: bool
    text: str
    forced_token_index: int = -1
    action_kinds: Tensor | None = None
    latent_vectors: Tensor | None = None
    controller_observations: Tensor | None = None
    carry_hiddens: Tensor | None = None

    def __post_init__(self) -> None:
        if self.token_ids.device.type != "cpu" or self.token_ids.dtype != torch.int32:
            raise ValueError("replay token ids must be CPU int32")
        if self.old_logprobs.device.type != "cpu" or self.old_logprobs.dtype != torch.float32:
            raise ValueError("replay log-probabilities must be CPU fp32")
        if self.advantages.device.type != "cpu" or self.advantages.dtype != torch.float32:
            raise ValueError("replay advantages must be CPU fp32")
        if any(tensor.ndim != 1 for tensor in (self.token_ids, self.old_logprobs, self.advantages)):
            raise ValueError("replay token ids, log-probabilities, and advantages must be vectors")
        response_length = self.token_ids.numel() - self.prompt_length
        if self.prompt_length < 1 or response_length < 1:
            raise ValueError("trajectory must contain prompt and response tokens")
        if self.old_logprobs.numel() != response_length:
            raise ValueError("one old log-probability is required per response token")
        if self.advantages.numel() != response_length:
            raise ValueError("one advantage is required per response token")
        if not -1 <= self.forced_token_index < response_length:
            raise ValueError("forced token index must be -1 or a response token index")
        if (self.action_kinds is None) != (self.latent_vectors is None):
            raise ValueError("latent replay requires both action kinds and vectors")
        carries = self.carry_hiddens
        if carries is not None:
            if self.action_kinds is not None:
                raise ValueError("token carry cannot be combined with latent trajectories")
            if (
                carries.device.type != "cpu" or carries.dtype != torch.bfloat16
                or carries.ndim != 2 or carries.shape[0] != response_length
                or carries.shape[1] < 1 or carries.requires_grad or carries.is_inference()
                or not bool(torch.isfinite(carries).all())
            ):
                raise ValueError("carry hiddens must be ordinary detached finite CPU BF16 [response_length, hidden_size]")
        observations = self.controller_observations
        if observations is not None:
            if (
                observations.device.type != "cpu"
                or not observations.is_floating_point()
                or observations.ndim != 2
                or observations.shape[0] != response_length
                or observations.shape[1] < 1
                or observations.requires_grad
            ):
                raise ValueError(
                    "controller observations must be detached CPU floating "
                    "[response_length, hidden_size]"
                )
            if self.latent_vectors is None:
                raise ValueError("controller observations require a latent trajectory")
        if self.action_kinds is None:
            return
        kinds = self.action_kinds
        vectors = self.latent_vectors
        assert vectors is not None
        if (
            kinds.device.type != "cpu" or kinds.dtype != torch.int8
            or kinds.ndim != 1 or kinds.numel() != response_length
        ):
            raise ValueError("action kinds must be CPU int8 with one entry per response slot")
        if (
            vectors.device.type != "cpu" or vectors.dtype != torch.float32
            or vectors.ndim != 2 or vectors.shape[1] < 1 or vectors.requires_grad
            or not bool(torch.isfinite(vectors).all())
        ):
            raise ValueError("latent vectors must be detached finite CPU fp32 [thoughts, hidden_size]")
        if observations is not None and observations.shape[1] != vectors.shape[1]:
            raise ValueError("controller observations must match the latent hidden size")
        thought_count = vectors.shape[0]
        if thought_count < 1 or response_length < thought_count + 2:
            raise ValueError("latent response requires thoughts, a close slot, and answer tokens")
        if (
            int(kinds[0]) != FIRST_THOUGHT
            or not bool((kinds[1:thought_count] == CONTINUE_THOUGHT).all())
            or int(kinds[thought_count]) not in (STOP_THINKING, FORCED_STOP_THINKING)
            or not bool((kinds[thought_count + 1:] == TOKEN_ACTION).all())
        ):
            raise ValueError("latent action kinds must be first, continue*, close, answer+")
        expected_forced = (
            thought_count if int(kinds[thought_count]) == FORCED_STOP_THINKING else -1
        )
        if self.forced_token_index != expected_forced:
            raise ValueError("forced token index must identify exactly the forced latent close")
        if expected_forced >= 0 and float(self.old_logprobs[expected_forced]) != 0.0:
            raise ValueError("forced latent close must have zero policy log-probability")

    @property
    def response_length(self) -> int:
        return self.token_ids.numel() - self.prompt_length

    @property
    def input_length(self) -> int:
        return self.token_ids.numel() - 1

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.token_ids, self.old_logprobs, self.advantages,
            self.action_kinds, self.latent_vectors, self.controller_observations,
            self.carry_hiddens,
        )
        return sum(
            tensor.numel() * tensor.element_size() for tensor in tensors if tensor is not None
        )

    @classmethod
    def from_device(
        cls,
        *,
        token_ids: Tensor,
        prompt_length: int,
        old_logprobs: Tensor,
        old_values: Tensor,
        correct: bool,
        text: str,
        forced_token_index: int = -1,
        action_kinds: Tensor | None = None,
        latent_vectors: Tensor | None = None,
        controller_observations: Tensor | None = None,
        carry_hiddens: Tensor | None = None,
    ) -> "TrajectoryRecord":
        if action_kinds is not None:
            action_kinds = action_kinds.detach().to(device="cpu")
            if (
                action_kinds.dtype not in (
                    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
                )
                or action_kinds.ndim != 1
                or not bool(((action_kinds >= TOKEN_ACTION) & (action_kinds <= FORCED_STOP_THINKING)).all())
            ):
                raise ValueError("action kinds must contain integer latent action codes")
        token_ids_cpu = token_ids.detach().to(device="cpu", dtype=torch.int32)
        response_length = token_ids_cpu.numel() - prompt_length
        if old_values.numel() != response_length:
            raise ValueError("one old value is required per response token")
        if carry_hiddens is not None:
            if carry_hiddens.dtype != torch.bfloat16:
                raise ValueError("rollout carry hiddens must already be BF16")
            with torch.inference_mode(False):
                carry_hiddens = carry_hiddens.detach().to(device="cpu", copy=True)
        return cls(
            token_ids=token_ids_cpu,
            prompt_length=prompt_length,
            old_logprobs=old_logprobs.detach().to(device="cpu", dtype=torch.float32),
            advantages=_precompute_advantages(old_values, correct),
            correct=correct,
            text=text,
            forced_token_index=forced_token_index,
            action_kinds=(
                None if action_kinds is None
                else action_kinds.detach().to(device="cpu", dtype=torch.int8)
            ),
            latent_vectors=(
                None if latent_vectors is None
                else latent_vectors.detach().to(device="cpu", dtype=torch.float32)
            ),
            controller_observations=(
                None if controller_observations is None
                else controller_observations.detach().to(device="cpu")
            ),
            carry_hiddens=carry_hiddens,
        )




def replay_storage_bytes(records: Sequence[TrajectoryRecord]) -> int:
    return sum(record.storage_bytes for record in records)


def plan_replay_microbatches(
    records: Sequence[TrajectoryRecord],
    order: Sequence[int],
    *,
    token_budget: int,
    max_trajectories: int,
) -> list[tuple[int, ...]]:
    """First-fit-decreasing packing under a real-token activation budget."""

    if token_budget < 1 or max_trajectories < 1:
        raise ValueError("replay limits must be positive")
    if not order or len(set(order)) != len(order) or any(
        index < 0 or index >= len(records) for index in order
    ):
        raise ValueError("replay order must contain unique valid trajectory indices")
    oversized = [
        index
        for index in order
        if records[index].input_length > token_budget
    ]
    if oversized:
        largest = max(records[index].input_length for index in oversized)
        raise ValueError(
            f"replay trajectory length {largest} exceeds token budget "
            f"{token_budget}"
        )
    plan: list[list[int]] = []
    token_counts: list[int] = []
    for index in sorted(
        order, key=lambda item: records[item].input_length, reverse=True
    ):
        length = records[index].input_length
        for shard_index, (shard, count) in enumerate(
            zip(plan, token_counts)
        ):
            if len(shard) < max_trajectories and count + length <= token_budget:
                shard.append(index)
                token_counts[shard_index] += length
                break
        else:
            plan.append([index])
            token_counts.append(length)
    packed = [tuple(shard) for shard in plan]
    flattened = [index for microbatch in packed for index in microbatch]
    if sorted(flattened) != sorted(order):
        raise RuntimeError("replay planner lost or duplicated a trajectory")
    return packed


@dataclass
class ReplayMicrobatch:
    input_ids: Tensor
    attention_mask: Tensor | None
    position_ids: Tensor
    cu_seqlens: Tensor
    sequence_boundaries: tuple[int, ...]
    max_sequence_length: int
    action_batch_indices: Tensor
    action_positions: Tensor
    targets: Tensor
    policy_mask: Tensor
    old_logprobs: Tensor
    advantages: Tensor
    value_targets: Tensor
    nextlat_sequence_ranges: tuple[tuple[int, int], ...]
    action_kinds: Tensor | None = None
    latent_vectors: Tensor | None = None
    latent_input_positions: Tensor | None = None
    latent_action_indices: Tensor | None = None
    token_action_indices: Tensor | None = None
    gate_action_indices: Tensor | None = None
    gate_stop_actions: Tensor | None = None
    carry_hiddens: Tensor | None = None
    carry_input_positions: Tensor | None = None

    @property
    def action_count(self) -> int:
        return self.targets.numel()




def collate_replay_microbatch(
    records: Sequence[TrajectoryRecord],
    indices: Sequence[int],
    *,
    pad_token_id: int,
    device: torch.device,
) -> ReplayMicrobatch:
    del pad_token_id
    if not indices:
        raise ValueError("cannot collate an empty replay microbatch")
    selected = [records[index] for index in indices]
    latent = selected[0].action_kinds is not None
    if any((record.action_kinds is not None) != latent for record in selected):
        raise ValueError("cannot mix native and latent trajectories in one replay microbatch")
    if latent and len({record.latent_vectors.shape[1] for record in selected if record.latent_vectors is not None}) != 1:
        raise ValueError("latent trajectories must share a hidden size")
    carry = selected[0].carry_hiddens is not None
    if any((record.carry_hiddens is not None) != carry for record in selected):
        raise ValueError("cannot mix native, Gaussian, and token-carry trajectories")
    if carry:
        if latent:
            raise ValueError("token carry cannot be combined with latent trajectories")
        if len({record.carry_hiddens.shape[1] for record in selected}) != 1:
            raise ValueError("carry trajectories must share a hidden size")
    lengths = [record.input_length for record in selected]
    total = sum(lengths)
    maximum = max(lengths)
    pin = device.type == "cuda"
    input_ids = torch.empty((1, total), dtype=torch.long, pin_memory=pin)
    position_ids = torch.empty_like(input_ids)
    positions: list[Tensor] = []
    targets: list[Tensor] = []
    policy_masks: list[Tensor] = []
    old_logprobs: list[Tensor] = []
    advantages: list[Tensor] = []
    value_targets: list[Tensor] = []
    action_kinds: list[Tensor] = []
    latent_vectors: list[Tensor] = []
    latent_input_positions: list[Tensor] = []
    latent_action_indices: list[Tensor] = []
    token_action_indices: list[Tensor] = []
    gate_action_indices: list[Tensor] = []
    gate_stop_actions: list[Tensor] = []
    carry_hiddens: list[Tensor] = []
    carry_input_positions: list[Tensor] = []
    nextlat_sequence_ranges: list[tuple[int, int]] = []
    action_offset = 0
    boundaries = [0]
    offset = 0

    for record in selected:
        length = record.input_length
        stop = offset + length
        input_ids[0, offset:stop].copy_(record.token_ids[:-1])
        position_ids[0, offset:stop].copy_(torch.arange(length))
        action_start = offset + record.prompt_length - 1
        action_stop = action_start + record.response_length
        positions.append(torch.arange(action_start, action_stop, dtype=torch.long))
        targets.append(record.token_ids[record.prompt_length:].long())
        policy_mask = torch.ones(record.response_length, dtype=torch.bool)
        if record.forced_token_index >= 0:
            policy_mask[record.forced_token_index] = False
        policy_masks.append(policy_mask)
        if carry:
            assert record.carry_hiddens is not None
            carry_hiddens.append(record.carry_hiddens[:-1])
            carry_input_positions.append(torch.arange(action_start + 1, action_stop))
        if latent:
            assert record.action_kinds is not None and record.latent_vectors is not None
            thought_count = record.latent_vectors.shape[0]
            action_kinds.append(record.action_kinds)
            latent_vectors.append(record.latent_vectors)
            # Action state precedes the sampled input by one stream position.
            latent_input_positions.append(
                torch.arange(action_start + 1, action_start + 1 + thought_count)
            )
            latent_action_indices.append(
                torch.arange(action_offset, action_offset + thought_count)
            )
            token_action_indices.append(torch.arange(
                action_offset + thought_count + 1, action_offset + record.response_length,
            ))
            learned_stop = record.forced_token_index < 0
            gate_count = thought_count - 1 + int(learned_stop)
            gate_action_indices.append(torch.arange(
                action_offset + 1, action_offset + 1 + gate_count,
            ))
            gate_actions = torch.zeros(gate_count, dtype=torch.long)
            if learned_stop:
                gate_actions[-1] = 1
            gate_stop_actions.append(gate_actions)
            nextlat_sequence_ranges.append((action_start + thought_count + 1, action_stop))
        else:
            nextlat_sequence_ranges.append((action_start, action_stop))
        action_offset += record.response_length
        old_logprobs.append(record.old_logprobs)
        advantages.append(record.advantages)
        value_targets.append(
            torch.full(
                (record.response_length,),
                1.0 if record.correct else -1.0,
                dtype=torch.float32,
            )
        )
        offset = stop
        boundaries.append(offset)

    def transfer(tensor: Tensor) -> Tensor:
        return tensor.to(device, non_blocking=pin)

    return ReplayMicrobatch(
        input_ids=transfer(input_ids),
        attention_mask=None,
        position_ids=transfer(position_ids),
        cu_seqlens=transfer(torch.tensor(boundaries, dtype=torch.int32)),
        sequence_boundaries=tuple(boundaries),
        max_sequence_length=maximum,
        action_batch_indices=transfer(
            torch.zeros(sum(record.response_length for record in selected), dtype=torch.long)
        ),
        action_positions=transfer(torch.cat(positions)),
        targets=transfer(torch.cat(targets)),
        policy_mask=transfer(torch.cat(policy_masks)),
        old_logprobs=transfer(torch.cat(old_logprobs)),
        advantages=transfer(torch.cat(advantages)),
        value_targets=transfer(torch.cat(value_targets)),
        action_kinds=transfer(torch.cat(action_kinds)) if latent else None,
        latent_vectors=transfer(torch.cat(latent_vectors)) if latent else None,
        latent_input_positions=transfer(torch.cat(latent_input_positions)) if latent else None,
        latent_action_indices=transfer(torch.cat(latent_action_indices)) if latent else None,
        token_action_indices=transfer(torch.cat(token_action_indices)) if latent else None,
        gate_action_indices=transfer(torch.cat(gate_action_indices)) if latent else None,
        gate_stop_actions=transfer(torch.cat(gate_stop_actions)) if latent else None,
        carry_hiddens=transfer(torch.cat(carry_hiddens)) if carry else None,
        carry_input_positions=transfer(torch.cat(carry_input_positions)) if carry else None,
        nextlat_sequence_ranges=tuple(nextlat_sequence_ranges),
    )


class StaticCachePool:
    """One reusable HF static cache with an explicit reset boundary per group."""

    def __init__(self, factory: Callable[[], Any], batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("cache batch size must be positive")
        self._factory = factory
        self._batch_size = batch_size
        self._cache: Any | None = None
        self.reset_count = 0

    def acquire(self, batch_size: int) -> Any:
        if batch_size != self._batch_size:
            raise ValueError("cache pool batch size changed")
        if self._cache is None:
            self._cache = self._factory()
        else:
            self._cache.reset()
            self.reset_count += 1
        return self._cache

    def clear(self) -> None:
        """Release cache tensors before replay needs the rollout VRAM."""
        self._cache = None

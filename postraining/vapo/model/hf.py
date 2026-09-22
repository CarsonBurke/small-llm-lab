"""Hugging Face causal-LM family.

MiniCPM5 is a stock ``LlamaForCausalLM``; so is every other model this
adapter serves. Nothing below names MiniCPM: the pinned model id, revision,
and vocabulary live in an :class:`HFModelSpec`, and the registry binds one
spec per family key so a checkpoint's lineage stays exact.

The runtime surgery kept here (packed replay attention, compiled replay
MLPs) is qualified against measured evidence in ``postraining/TODO_MINICPM5.md``
and must not be re-derived; it is Llama-shaped, which is why the adapter
declares the matching capabilities only after verifying the module layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from postraining.hf_runtime import prepare_text_only_transformers_runtime
from postraining.vapo.model.protocols import (
    Capability,
    ModelFamily,
    Readout,
    TrunkAdapter,
    TrunkGeometry,
)
from postraining.vapo.model.readout import FrozenLinearReadout
from postraining.vapo.model.registry import register_family


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
    # Concatenate token-major views so the [1, tokens, heads, dim] result is
    # already contiguous: the caller's head merge becomes a view instead of a
    # second full copy of the activations.
    return torch.cat([output.transpose(1, 2) for output in outputs], dim=1), None


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

@dataclass(frozen=True)
class HFModelSpec:
    """Pinned identity of one Hugging Face checkpoint.

    ``vocab_size`` is asserted rather than read so that a silently reshaped
    or re-tokenized upload cannot be adopted by an existing run.
    """

    key: str
    model_id: str
    revision: str
    vocab_size: int
    model_type: str = "llama"
    requires_chat_template: bool = True

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision:
            raise ValueError("a model spec pins both an id and a revision")
        if int(self.vocab_size) < 1:
            raise ValueError("vocabulary size must be positive")


class HFCausalTrunk(TrunkAdapter):
    """Adapter over a ``*ForCausalLM`` whose body is ``model`` and head ``lm_head``."""

    def __init__(self, causal_lm: Any, spec: HFModelSpec) -> None:
        config: Any = causal_lm.config
        if getattr(config, "model_type", None) != spec.model_type:
            raise ValueError(
                f"{spec.key} requires a {spec.model_type} causal LM, got "
                f"{getattr(config, 'model_type', None)!r}"
            )
        if int(config.vocab_size) != int(spec.vocab_size):
            raise ValueError(
                f"unexpected {spec.key} vocabulary: {config.vocab_size} != "
                f"{spec.vocab_size}"
            )
        if not isinstance(causal_lm.lm_head, nn.Linear):
            raise TypeError(f"{spec.key} output head layout differs")
        self.module = causal_lm
        self.spec = spec
        self._config = config
        self._readout = FrozenLinearReadout(causal_lm.lm_head)
        causal_lm.config.use_cache = False

    # -- identity ---------------------------------------------------------

    @property
    def family(self) -> str:
        return self.spec.key

    @property
    def causal_lm(self) -> Any:
        """The wrapped Hugging Face model, for family-specific call sites."""
        return self.module

    @property
    def config(self) -> Any:
        return self._config

    @property
    def hidden_size(self) -> int:
        return int(self._config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self._config.vocab_size)

    @property
    def geometry(self) -> TrunkGeometry:
        config = self._config
        return TrunkGeometry(
            hidden_size=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
            num_layers=len(self.layers()),
            num_key_value_heads=int(config.num_key_value_heads),
            head_dim=int(config.head_dim),
            pad_token_id=int(config.pad_token_id),
        )

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {
                Capability.LORA_ADAPTERS,
                Capability.FROZEN_OUTPUT_HEAD,
                Capability.GRADIENT_CHECKPOINTING,
                Capability.PACKED_REPLAY_ATTENTION,
                Capability.COMPILED_REPLAY_MLP,
                Capability.STATIC_KV_CACHE,
                Capability.FUSED_PROJECTIONS,
                Capability.FA4_DECODE,
                Capability.SPLIT_KV_DECODE,
                Capability.W8A16_HEAD,
            }
            | ({Capability.CHAT_TEMPLATE} if self.spec.requires_chat_template else set())
        )

    def identity(self) -> dict[str, Any]:
        return {
            "family": self.spec.key,
            "model_id": self.spec.model_id,
            "revision": self.spec.revision,
            "model_type": self.spec.model_type,
            "vocab_size": int(self.spec.vocab_size),
            "hidden_size": self.hidden_size,
            "num_hidden_layers": len(self.layers()),
        }

    # -- forward paths ----------------------------------------------------

    @property
    def readout(self) -> Readout:
        return self._readout

    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        return self.module.get_input_embeddings()(token_ids)

    @property
    def input_embedding_dtype(self) -> torch.dtype:
        return self.module.get_input_embeddings().weight.dtype

    def hidden_states(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        sequence_boundaries: tuple[int, ...] | None = None,
        max_sequence_length: int = 0,
    ) -> Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("replay takes exactly one of token ids or embeddings")
        if cu_seqlens is not None:
            boundaries = sequence_boundaries
            if boundaries is None:
                boundaries = tuple(int(item) for item in cu_seqlens.tolist())
            for attention in self.attention_modules():
                packed = cast(Any, attention)
                packed._packed_cu_seqlens = cu_seqlens
                packed._packed_max_sequence_length = max_sequence_length
                packed._packed_sequence_boundaries = boundaries
        inputs = (
            {"input_ids": input_ids} if inputs_embeds is None
            else {"inputs_embeds": inputs_embeds}
        )
        outputs = self.module.model(
            **inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def cached_hidden_states(
        self,
        *,
        past_key_values: Any,
        cache_position: Tensor,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("decode takes exactly one of token ids or embeddings")
        inputs = (
            {"input_ids": input_ids} if inputs_embeds is None
            else {"inputs_embeds": inputs_embeds}
        )
        outputs = self.module.model(
            **inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def new_kv_cache(
        self,
        *,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> Any:
        """A plain static cache.

        The captured rollout engine allocates its own compact layers; this is
        the ordinary path used by evaluation and paired-rollout diagnostics.
        """
        from transformers import StaticCache

        return StaticCache(
            config=self._config,
            max_batch_size=batch_size,
            max_cache_len=cache_length,
            device=device,
            dtype=dtype or torch.bfloat16,
        )

    # -- runtime surgery --------------------------------------------------

    def layers(self) -> Sequence[nn.Module]:
        return cast(Any, self.module.get_submodule("model")).layers

    def set_attention_implementation(self, implementation: str) -> None:
        self._config._attn_implementation = implementation

    def set_gradient_checkpointing(self, interval: int) -> int:
        layers = self.layers()
        if interval <= 0:
            self.module.gradient_checkpointing_disable()
            for layer in layers:
                layer.gradient_checkpointing = False
            return 0
        self.module.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        enabled = 0
        for index, layer in enumerate(layers):
            checkpointed = index % interval == 0
            layer.gradient_checkpointing = checkpointed
            enabled += int(checkpointed)
        return enabled


class HFCausalFamily(ModelFamily):
    """Family that loads one pinned Hugging Face checkpoint."""

    def __init__(self, spec: HFModelSpec) -> None:
        self.spec = spec
        self.key = spec.key

    def build_tokenizer(
        self, *, model_id: str | None = None, revision: str | None = None, **options: Any
    ) -> Any:
        prepare_text_only_transformers_runtime()
        from transformers import AutoTokenizer

        spec = self.resolved_spec(model_id=model_id, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
        if len(tokenizer) != int(spec.vocab_size):
            raise ValueError("tokenizer and checkpoint vocabularies differ")
        if spec.requires_chat_template and tokenizer.chat_template is None:
            raise ValueError(f"{self.key} tokenizer is missing its native chat template")
        return tokenizer

    def resolved_spec(
        self, *, model_id: str | None = None, revision: str | None = None
    ) -> HFModelSpec:
        """The family's spec, or an explicitly overridden one.

        An override is recorded in the spec itself, so a run that points at a
        different upload cannot later be mistaken for the pinned lineage. The
        vocabulary assertion is unchanged and still applies.
        """
        if model_id is None and revision is None:
            return self.spec
        return replace(
            self.spec,
            model_id=model_id or self.spec.model_id,
            revision=revision or self.spec.revision,
        )

    def build_actor_trunk(
        self,
        *,
        device: torch.device,
        gradient_checkpointing: bool = True,
        model_id: str | None = None,
        revision: str | None = None,
        **options: Any,
    ) -> HFCausalTrunk:
        prepare_text_only_transformers_runtime()
        from transformers import AutoModelForCausalLM

        spec = self.resolved_spec(model_id=model_id, revision=revision)
        loaded: Any = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        trunk = HFCausalTrunk(loaded.to(device), spec)
        if gradient_checkpointing:
            trunk.set_gradient_checkpointing(1)
        return trunk

    def build_critic_trunk(
        self, *, device: torch.device, actor: TrunkAdapter, **options: Any
    ) -> HFCausalTrunk:
        """A second adapter set over the actor's immutable pretrained weights.

        The caller shares frozen storage through
        :func:`postraining.vapo.model.lora.share_frozen_parameters_` after
        injecting the critic's own adapters, so this loads a fresh module and
        does not itself alias anything.
        """
        return self.build_actor_trunk(device=device, **options)

    def build_rollout_engine(self, **options: Any) -> Any:
        from postraining.fast_inference import CapturedTrainingRolloutEngine

        return CapturedTrainingRolloutEngine(**options)


MINICPM5_SPEC = HFModelSpec(
    key="minicpm5",
    model_id="openbmb/MiniCPM5-1B",
    revision="87179e5c1f455ef22e6223592d2d61351b525bfc",
    vocab_size=130_560,
)

register_family(MINICPM5_SPEC.key, lambda: HFCausalFamily(MINICPM5_SPEC))


__all__ = [
    "HFCausalFamily",
    "HFCausalTrunk",
    "HFModelSpec",
    "MINICPM5_SPEC",
    "enable_packed_replay_attention",
    "enable_replay_mlp_compilation",
    "use_packed_replay_attention",
]

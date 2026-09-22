"""Model-side contracts for the post-training RL library.

The VAPO trainer, its replay machinery, and its rollout engines were written
against MiniCPM5's ``LlamaForCausalLM`` internals: ``causal_lm.model.layers``,
``causal_lm.lm_head``, and a Llama ``config``. Those reach-throughs are what
kept nanoGPT-mini and KDA backbones out of the same stack.

Everything the algorithm actually needs from a model is declared here:

``TrunkGeometry``
    Static shape facts used for cache sizing, budget validation, and padding.
``Readout``
    Turning trunk features into vocabulary log-probabilities. Families differ:
    MiniCPM5 scores against a frozen ``lm_head`` weight, the nano tied-dot
    readout scores against a normalized codebook with a softcap.
``TrunkAdapter``
    The trunk itself: embeddings, teacher-forced replay hidden states, cached
    decode hidden states, layer access for runtime surgery, and a KV cache.
``RolloutEngine``
    Producing sampled responses from prompts. Backends differ radically
    (HF static-cache continuous batching versus the nano paged lockstep
    decoder) and are not unified beyond this interface.
``ModelFamily``
    The registry entry that builds a tokenizer, an actor trunk, a critic
    trunk, and a rollout engine, and that binds checkpoint lineage.

Capabilities are declared per family rather than inferred from class names,
so an optimization that only exists for one runtime (fused projections, FA4
decode, packed replay attention) is gated on a fact the adapter asserts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from postraining.vapo.rollout.results import ContinuousTrainingGeneration


class Capability(StrEnum):
    """Runtime features a trunk may support.

    Declared, never guessed. A trainer that wants packed replay attention
    asks the adapter rather than testing for a Llama module path.
    """

    LORA_ADAPTERS = "lora_adapters"
    FROZEN_OUTPUT_HEAD = "frozen_output_head"
    GRADIENT_CHECKPOINTING = "gradient_checkpointing"
    PACKED_REPLAY_ATTENTION = "packed_replay_attention"
    COMPILED_REPLAY_MLP = "compiled_replay_mlp"
    STATIC_KV_CACHE = "static_kv_cache"
    PAGED_KV_CACHE = "paged_kv_cache"
    FUSED_PROJECTIONS = "fused_projections"
    FA4_DECODE = "fa4_decode"
    SPLIT_KV_DECODE = "split_kv_decode"
    W8A16_HEAD = "w8a16_head"
    CHAT_TEMPLATE = "chat_template"


@dataclass(frozen=True)
class TrunkGeometry:
    """Static shape facts shared by every family.

    ``num_key_value_heads`` and ``head_dim`` describe the decode cache, not
    the replay path; a family without attention KV state reports zero for both
    and must not claim :data:`Capability.STATIC_KV_CACHE`.
    """

    hidden_size: int
    vocab_size: int
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    pad_token_id: int

    def __post_init__(self) -> None:
        for name in ("hidden_size", "vocab_size", "num_layers"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_key_value_heads < 0 or self.head_dim < 0:
            raise ValueError("cache geometry must be non-negative")
        if bool(self.num_key_value_heads) != bool(self.head_dim):
            raise ValueError("KV head count and head dimension travel together")
        if not 0 <= int(self.pad_token_id) < int(self.vocab_size):
            raise ValueError("pad token must lie inside the vocabulary")

    def kv_cache_bytes(
        self, *, batch_size: int, cache_length: int, element_size: int = 2
    ) -> int:
        """Exact dense K/V storage for one decode cache."""
        if batch_size < 1 or cache_length < 1 or element_size < 1:
            raise ValueError("cache dimensions must be positive")
        if not self.num_key_value_heads:
            raise ValueError("this trunk has no attention KV cache")
        return (
            batch_size
            * cache_length
            * self.num_layers
            * 2
            * self.num_key_value_heads
            * self.head_dim
            * element_size
        )


@runtime_checkable
class Readout(Protocol):
    """Vocabulary scoring for trunk features.

    ``features`` is whatever :meth:`TrunkAdapter.hidden_states` returned for
    the scored positions. A family whose readout needs more than the final
    hidden state (the nano renderer consumes ``cat(token_latent, belief)``)
    packs that into its own feature tensor rather than widening this contract.
    """

    @property
    def frozen(self) -> bool:
        """Whether the readout parameters are excluded from the actor."""

    def logits(self, features: Tensor) -> Tensor:
        """Dense vocabulary logits, fp32."""

    def target_logprobs(
        self, features: Tensor, targets: Tensor, *, chunk_tokens: int
    ) -> Tensor:
        """Log-probability of ``targets`` without materializing dense logits.

        ``chunk_tokens`` bounds the vocabulary-sized intermediate: the whole
        point of this method is that a 130k-token head never allocates a
        ``[actions, vocab]`` tensor for either the forward or the backward.
        """


class TrunkAdapter(ABC):
    """A model's trunk, expressed in the terms the RL library uses.

    Implementations own the concrete module and are responsible for keeping
    arithmetic identical to that model's qualified runtime. The adapter is a
    view, not a container: it never copies parameters and never changes
    dtypes behind the caller.
    """

    #: The wrapped module. Optimizers and checkpointing address it directly.
    module: nn.Module

    # -- identity ---------------------------------------------------------

    @property
    @abstractmethod
    def family(self) -> str:
        """Registry key this adapter was built from."""

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Width of the features :meth:`hidden_states` returns.

        Separate from :attr:`geometry` because every trunk has one, while a
        trunk used only for replay need not describe a decode cache.
        """

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def geometry(self) -> TrunkGeometry: ...

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[Capability]: ...

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Checkpoint-bound lineage: family, source, revision, architecture.

        Written into every derived checkpoint and compared on resume, so it
        must contain only deterministic, JSON-safe values.
        """

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def require(self, *capabilities: Capability) -> None:
        missing = [item for item in capabilities if item not in self.capabilities]
        if missing:
            names = ", ".join(sorted(str(item) for item in missing))
            raise RuntimeError(f"{self.family} trunk lacks: {names}")

    # -- family-specific escape hatch -------------------------------------

    @property
    def causal_lm(self) -> Any:
        """The wrapped Hugging Face model, for Hugging Face call sites only.

        Overridden by :class:`HFCausalTrunk`. Any other family raises here
        rather than letting a bare ``AttributeError`` from deep inside a
        rollout engine stand in for "this entrypoint is Hugging Face only".
        """
        raise RuntimeError(
            f"{self.family} trunk has no causal_lm: this call site reaches "
            "into Hugging Face internals and only supports the hf family; "
            "family-agnostic code goes through the TrunkAdapter methods"
        )

    # -- parameters -------------------------------------------------------

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.module.parameters()

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        return self.module.named_parameters()

    @property
    def device(self) -> torch.device:
        return next(self.module.parameters()).device

    # -- forward paths ----------------------------------------------------

    @property
    @abstractmethod
    def readout(self) -> Readout: ...

    @abstractmethod
    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        """Input-side representation consumed by the first layer."""

    @abstractmethod
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
        """Teacher-forced replay features for every input position.

        ``cu_seqlens`` selects the packed (variable-length) layout used by
        replay microbatches; adapters without
        :data:`Capability.PACKED_REPLAY_ATTENTION` must reject it.
        """

    @abstractmethod
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
        """One decode step (or prefill) against a mutable KV cache."""

    @abstractmethod
    def new_kv_cache(
        self,
        *,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> Any:
        """Allocate decode KV storage in this family's cache representation."""

    # -- runtime surgery --------------------------------------------------

    @abstractmethod
    def layers(self) -> Sequence[nn.Module]:
        """Ordered transformer blocks, for per-layer runtime configuration."""

    def attention_modules(self) -> Iterable[nn.Module]:
        for layer in self.layers():
            attention = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attention is None:
                raise AttributeError(f"{type(layer).__name__} exposes no attention module")
            yield attention

    def set_gradient_checkpointing(self, interval: int) -> int:
        """Enable checkpointing on every ``interval``-th layer; return the count.

        ``interval <= 0`` disables it. The default refuses rather than
        silently ignoring a caller that asked for memory it will not get.
        """
        if interval > 0:
            raise RuntimeError(f"{self.family} trunk does not support gradient checkpointing")
        return 0

    def set_attention_implementation(self, implementation: str) -> None:
        raise RuntimeError(f"{self.family} trunk has a fixed attention implementation")


@runtime_checkable
class RolloutEngine(Protocol):
    """Sampling backend that turns prompts into replayable trajectories.

    Backends are not interchangeable in performance or in memory lifecycle,
    only in this interface. Implementations must release decode KV storage in
    :meth:`release_cache` before the trainer allocates update activations.
    """

    @property
    def trunk(self) -> TrunkAdapter: ...

    def prepare_generation(self) -> None:
        """Refresh the inference replica from the current actor weights."""

    def release_cache(self) -> None:
        """Free decode KV storage so the update step can use the memory."""

    def generate_prompt_pool(
        self,
        prompt_ids_cpu: Sequence[Tensor],
        *,
        max_new_tokens: int,
        context_tokens: int | None = None,
    ) -> "ContinuousTrainingGeneration":
        """Sample one response per prompt, in prompt order.

        Backends may accept further keyword options of their own (lane
        refill policy, prefill batching); the trainer only relies on what is
        declared here. The result is the shared currency in
        ``postraining.vapo.rollout.results``, not a backend type.
        """


class ModelFamily(ABC):
    """Registry entry: how to build one model's RL components."""

    #: Unique registry key, also written into checkpoints.
    key: str

    @abstractmethod
    def build_tokenizer(self, **options: Any) -> Any: ...

    @abstractmethod
    def build_actor_trunk(self, *, device: torch.device, **options: Any) -> TrunkAdapter:
        """Load the pretrained trunk the actor trains on top of."""

    @abstractmethod
    def build_critic_trunk(
        self, *, device: torch.device, actor: TrunkAdapter, **options: Any
    ) -> TrunkAdapter:
        """Build the critic trunk.

        Families that share immutable pretrained weights between actor and
        critic alias ``actor``'s parameters here rather than loading twice.
        """

    @abstractmethod
    def build_rollout_engine(self, **options: Any) -> RolloutEngine: ...


__all__ = [
    "Capability",
    "ModelFamily",
    "Readout",
    "RolloutEngine",
    "TrunkAdapter",
    "TrunkGeometry",
]

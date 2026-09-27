"""Custom-backbone family: nanoGPT-mini, the tied-dot readout variant, and KDA.

These backbones already share a post-training interface
(``postraining.nano_backbone._NanoPostrainingMixin``): token embedding,
temporal belief, a softcapped renderer, a paged generation cache, and
per-block prefill/step entry points. This module expresses that interface as
a :class:`~postraining.vapo.model.protocols.TrunkAdapter` so the VAPO trainer
and its replay machinery reach a KDA checkpoint through exactly the calls
they make against a Hugging Face model.

Two differences from the Hugging Face family are structural, not incidental:

* There is no frozen output head. The renderer is trainable and, for the
  tied-dot variant, gradients flow back into the embedding through it. The
  actor therefore trains the whole trunk rather than an adapter, and the
  family does not claim :data:`Capability.LORA_ADAPTERS`.
* Decode state is a per-layer ``(key, value)`` pair the caller preallocates,
  not a ``transformers`` ``Cache``. It is a paged cache, not a static one.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from postraining.vapo.model.protocols import (
    Capability,
    ModelFamily,
    Readout,
    TrunkAdapter,
    TrunkGeometry,
)
from postraining.vapo.model.readout import TrunkRenderReadout
from postraining.vapo.model.registry import register_family


class NanoTrunk(TrunkAdapter):
    """Adapter over a ``_NanoPostrainingMixin`` backbone.

    ``pad_token_id`` is supplied by the caller because these tokenizers carry
    no padding row of their own; the trainer pads with the end-of-sequence id
    and masks those positions out of every loss.
    """

    def __init__(self, backbone: Any, *, pad_token_id: int) -> None:
        for attribute in (
            "embed_tokens",
            "temporal_belief_from_token_latent",
            "logits_from_features",
            "target_logprobs_from_features",
            "make_generation_cache",
            "prefill_belief",
        ):
            if not hasattr(backbone, attribute):
                raise TypeError(
                    f"{type(backbone).__name__} is not a post-training nano "
                    f"backbone: missing {attribute}"
                )
        self.module = backbone
        self._pad_token_id = int(pad_token_id)
        self._readout = TrunkRenderReadout(backbone)

    # -- identity ---------------------------------------------------------

    @property
    def family(self) -> str:
        return "nano"

    @property
    def architecture(self) -> str:
        return str(self.module.architecture)

    @property
    def hidden_size(self) -> int:
        return int(self.module.model_dim)

    @property
    def vocab_size(self) -> int:
        return int(self.module.tok_emb.num_embeddings)

    @property
    def geometry(self) -> TrunkGeometry:
        blocks = self.layers()
        attention = getattr(blocks[0], "attn", None)
        return TrunkGeometry(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            num_layers=len(blocks),
            num_key_value_heads=int(getattr(attention, "num_heads", 0) or 0),
            head_dim=int(getattr(attention, "head_dim", 0) or 0),
            pad_token_id=self._pad_token_id,
        )

    @property
    def capabilities(self) -> frozenset[Capability]:
        if not self.geometry.num_key_value_heads:
            # A recurrent mixer carries state instead of keys and values.
            return frozenset()
        return frozenset({Capability.PAGED_KV_CACHE})

    def identity(self) -> dict[str, Any]:
        geometry = self.geometry
        return {
            "family": self.family,
            "architecture": self.architecture,
            "model_config": dict(self.module.model_config),
            "vocab_size": geometry.vocab_size,
            "hidden_size": geometry.hidden_size,
            "num_hidden_layers": geometry.num_layers,
            "train_context_tokens": int(getattr(self.module, "train_context_tokens", 0)),
        }

    # -- forward paths ----------------------------------------------------

    @property
    def readout(self) -> Readout:
        return self._readout

    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        return self.module.embed_tokens(token_ids)

    @property
    def input_embedding_dtype(self) -> torch.dtype:
        return self.module.tok_emb.weight.dtype

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
        if cu_seqlens is not None or sequence_boundaries is not None:
            raise RuntimeError(
                "nano replay has no packed variable-length attention; pad the "
                "microbatch and mask instead"
            )
        if attention_mask is not None or position_ids is not None:
            # The nano trunk derives positions from the sequence axis and has
            # no mask argument; accepting either silently would train on the
            # wrong positions.
            raise RuntimeError("nano replay takes neither an attention mask nor positions")
        latent = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        return self.module.temporal_belief_from_token_latent(latent)

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
        """One decode step against preallocated per-layer caches.

        ``attention_mask`` is the nano ``key_mask`` (True where a cache slot
        may be attended), and ``cache_position`` is the absolute cache slot.
        ``position_ids`` has no meaning here: the slot is the position.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("decode takes exactly one of token ids or embeddings")
        if position_ids is not None:
            raise RuntimeError("the nano cache slot is the position")
        caches = cast(list, past_key_values)
        if len(caches) != self.geometry.num_layers:
            raise ValueError("one key/value cache is required per layer")
        latent = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        if latent.ndim != 3 or latent.shape[1] != 1:
            raise ValueError("a decode step consumes exactly one position")
        hidden = latent
        for index, block in enumerate(self.layers()):
            hidden, caches[index] = self.module._block_step(
                block, hidden, latent, caches[index], cache_position, attention_mask
            )
        return self.module.final_norm(hidden)

    def new_kv_cache(
        self,
        *,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        return self.module.make_generation_cache(batch_size, cache_length, device, dtype)

    # -- runtime surgery --------------------------------------------------

    def layers(self) -> Sequence[nn.Module]:
        return self.module.blocks


class NanoFamily(ModelFamily):
    """Loads a pretrained nano/KDA checkpoint and its matching tokenizer.

    The concrete architecture is read from the checkpoint rather than from a
    family key, which is what makes one family serve every nano variant while
    still binding the exact architecture into checkpoint lineage.
    """

    key = "nano"

    def build_tokenizer(
        self,
        *,
        architecture: str,
        sp_model_path: str,
        think_tokens: bool = False,
        answer_tokens: bool = False,
        tokenizer_provenance: dict | None = None,
        **options: Any,
    ) -> Any:
        from postraining.core import load_posttraining_tokenizer

        return load_posttraining_tokenizer(
            architecture,
            sp_model_path,
            think_tokens=think_tokens,
            answer_tokens=answer_tokens,
            tokenizer_provenance=tokenizer_provenance,
        )

    def build_actor_trunk(
        self,
        *,
        device: torch.device,
        checkpoint: str | Path | None = None,
        payload: dict | None = None,
        pad_token_id: int = 0,
        **options: Any,
    ) -> NanoTrunk:
        from postraining.model_io import load_model

        if checkpoint is None and payload is None:
            raise ValueError("a nano actor is loaded from a checkpoint")
        backbone = load_model(cast(Any, checkpoint), device, payload)
        for parameter in backbone.parameters():
            parameter.requires_grad_(True)
        return NanoTrunk(backbone, pad_token_id=pad_token_id)

    def build_critic_trunk(
        self,
        *,
        device: torch.device,
        actor: TrunkAdapter,
        model_config_overrides: dict | None = None,
        **options: Any,
    ) -> NanoTrunk:
        """A fresh, randomly initialized trunk of the actor's architecture.

        Unlike the Hugging Face family, nothing is shared with the actor: the
        critic reads the same stream through its own weights, as
        ``postraining.value_model.SeparateCritic`` has always done.
        """
        from postraining.model_io import fresh_trunk

        if not isinstance(actor, NanoTrunk):
            raise TypeError("a nano critic is built from a nano actor")
        backbone = fresh_trunk(
            actor.module,
            device,
            model_config_overrides=model_config_overrides,
            architecture_suffix="_critic",
        )
        return NanoTrunk(backbone, pad_token_id=actor.geometry.pad_token_id)

    def build_rollout_engine(self, **options: Any) -> Any:
        from postraining.vapo.rollout.nano_engine import NanoRolloutEngine

        return NanoRolloutEngine(**options)


register_family(NanoFamily.key, NanoFamily)


__all__ = ["NanoFamily", "NanoTrunk"]

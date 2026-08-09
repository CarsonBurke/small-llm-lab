"""Byte-native tokenization for training, serving, and post-training.

Literal text is always encoded as its exact UTF-8 octets. Control tokens are
typed values rather than magic substrings: the text ``"<think>"`` therefore
stays seven literal bytes unless the caller explicitly supplies
``SpecialAtom("<think>")``. This avoids prompt-content injection through a
special-token scanner and gives every phase one encoding contract.
"""

from __future__ import annotations

import codecs
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from .data import AtomicDocument, AtomicIdManifest, SpecialAtom


TokenPart = str | SpecialAtom
DecodedPart = str | SpecialAtom
SpecialDecodeMode = Literal["error", "skip", "surface"]


class ByteTokenizer:
    """The model's complete external tokenizer.

    No normalization, pre-tokenization, or learned vocabulary is involved.
    Strict UTF-8 rejects unpaired surrogates instead of silently changing the
    input. EOT is appended only when explicitly requested.
    """

    def __init__(self, manifest: AtomicIdManifest | None = None) -> None:
        self.manifest = manifest or AtomicIdManifest.reference()

    def encode_text(self, text: str, *, add_eot: bool = False) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        ids = tuple(text.encode("utf-8", errors="strict"))
        return (*ids, self.manifest.eot_id) if add_eot else ids

    def encode_parts(
        self,
        parts: Iterable[TokenPart],
        *,
        add_eot: bool = False,
    ) -> tuple[int, ...]:
        by_name = self.manifest.special_by_name
        encoded: list[int] = []
        for part in parts:
            if isinstance(part, str):
                encoded.extend(part.encode("utf-8", errors="strict"))
            elif isinstance(part, SpecialAtom):
                try:
                    encoded.append(by_name[part.name].atomic_id)
                except KeyError as error:
                    raise ValueError(f"unknown atomic special {part.name!r}") from error
            else:
                raise TypeError(
                    "parts must be str or SpecialAtom, "
                    f"got {type(part).__name__}"
                )
        if add_eot:
            encoded.append(self.manifest.eot_id)
        return tuple(encoded)

    def encode_document(self, text: str, *, key: str) -> AtomicDocument:
        document = AtomicDocument(
            key=key,
            atomic_ids=self.encode_text(text, add_eot=True),
        )
        document.validate(self.manifest)
        return document

    def encode_prompt(self, text: str, *, add_bos: bool = False) -> tuple[int, ...]:
        """Encode untrusted prompt text without recognizing control surfaces.

        The model supplies BOS virtually. ``add_bos`` exists only for external
        formats that explicitly store that boundary; serving leaves it false.
        """

        body = self.encode_text(text)
        return (self.manifest.eot_id, *body) if add_bos else body

    def encode_reasoning_completion(
        self,
        reasoning: str,
        answer: str,
        *,
        add_eot: bool = True,
    ) -> tuple[int, ...]:
        """Compose trusted SFT/RL fields into typed reasoning controls.

        The fields remain literal even if they contain strings that resemble
        controls. Corpus preparation should reject those strings when its
        structural reward contract forbids them, but tokenization never grants
        them control semantics implicitly.
        """

        return self.encode_parts(
            (
                SpecialAtom("<think>"),
                reasoning,
                SpecialAtom("</think>"),
                SpecialAtom("<answer>"),
                answer,
                SpecialAtom("</answer>"),
            ),
            add_eot=add_eot,
        )

    def decode_parts(self, atomic_ids: Iterable[int]) -> tuple[DecodedPart, ...]:
        decoder = IncrementalByteDecoder(self.manifest)
        decoded: list[DecodedPart] = []
        for atomic_id in atomic_ids:
            decoded.extend(decoder.push(atomic_id))
        decoded.extend(decoder.finish())
        return _coalesce_text(decoded)

    def decode_text(
        self,
        atomic_ids: Iterable[int],
        *,
        specials: SpecialDecodeMode = "error",
    ) -> str:
        if specials not in {"error", "skip", "surface"}:
            raise ValueError(f"unknown special decode mode {specials!r}")
        output: list[str] = []
        for part in self.decode_parts(atomic_ids):
            if isinstance(part, str):
                output.append(part)
            elif specials == "skip":
                continue
            elif specials == "surface":
                output.append(part.name)
            else:
                raise ValueError(
                    f"cannot decode atomic special {part.name!r} as plain text"
                )
        return "".join(output)

    def encode_batch(
        self, texts: Sequence[str], *, add_eot: bool = False
    ) -> list[list[int]]:
        return [list(self.encode_text(text, add_eot=add_eot)) for text in texts]


class IncrementalByteDecoder:
    """Strict streaming UTF-8 decoder that preserves atomic control events.

    A control token may occur only at a UTF-8 code-point boundary. This makes
    an invalid generated prefix observable instead of replacing it with U+FFFD
    and hiding a model or sampler defect.
    """

    def __init__(self, manifest: AtomicIdManifest | None = None) -> None:
        self.manifest = manifest or AtomicIdManifest.reference()
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._finished = False

    def push(self, atomic_id: int) -> tuple[DecodedPart, ...]:
        if self._finished:
            raise RuntimeError("incremental byte decoder is already finished")
        self.manifest.validate_clean_id(atomic_id)
        if atomic_id < self.manifest.byte_count:
            text = self._decoder.decode(bytes((atomic_id,)), final=False)
            return (text,) if text else ()

        text = self._decoder.decode(b"", final=True)
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        special = self.manifest.special_by_id[atomic_id]
        prefix: tuple[DecodedPart, ...] = (text,) if text else ()
        return (*prefix, SpecialAtom(special.name))

    def finish(self) -> tuple[DecodedPart, ...]:
        if self._finished:
            return ()
        self._finished = True
        text = self._decoder.decode(b"", final=True)
        return (text,) if text else ()


@dataclass(frozen=True)
class UTF8ByteBatchEncoder:
    """Adapter for the deterministic K3 corpus builder's batch API.

    Its ids are already model ids: 0..255 are octets and 256 is EOT/BOS. It
    makes bytes the budgeting/sharding unit; it is not a learned tokenizer.
    """

    manifest: AtomicIdManifest = AtomicIdManifest.reference()
    is_utf8_bytes: bool = True

    @property
    def vocab_size(self) -> int:
        return self.manifest.output_size

    @property
    def eot_id(self) -> int:
        return self.manifest.eot_id

    def encode(self, texts: Sequence[str], *, out_type: type = int) -> list[list[int]]:
        if out_type is not int:
            raise ValueError("UTF8ByteBatchEncoder supports only out_type=int")
        return ByteTokenizer(self.manifest).encode_batch(texts)

    def decode(self, batches: Sequence[Sequence[int]]) -> list[str]:
        tokenizer = ByteTokenizer(self.manifest)
        return [tokenizer.decode_text(ids, specials="skip") for ids in batches]


def _coalesce_text(parts: Iterable[DecodedPart]) -> tuple[DecodedPart, ...]:
    output: list[DecodedPart] = []
    for part in parts:
        if isinstance(part, str) and output and isinstance(output[-1], str):
            output[-1] += part
        elif part != "":
            output.append(part)
    return tuple(output)


__all__ = [
    "ByteTokenizer",
    "DecodedPart",
    "IncrementalByteDecoder",
    "SpecialDecodeMode",
    "TokenPart",
    "UTF8ByteBatchEncoder",
]

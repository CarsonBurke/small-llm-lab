from __future__ import annotations

import pytest

from pretraining.byte_diffusion.data import AtomicIdManifest, SpecialAtom
from pretraining.byte_diffusion.tokenizer import (
    ByteTokenizer,
    IncrementalByteDecoder,
    UTF8ByteBatchEncoder,
)


def test_text_is_exact_utf8_and_special_surfaces_are_not_scanned() -> None:
    tokenizer = ByteTokenizer()
    text = "🙂 literal <think>\n"

    ids = tokenizer.encode_text(text)

    assert ids == tuple(text.encode("utf-8"))
    assert tokenizer.decode_text(ids) == text
    assert AtomicIdManifest.reference().special_by_name["<think>"].atomic_id not in ids


def test_typed_specials_and_terminal_eot_are_atomic() -> None:
    tokenizer = ByteTokenizer()

    ids = tokenizer.encode_parts(
        ("question\n", SpecialAtom("<think>"), "work", SpecialAtom("</think>")),
        add_eot=True,
    )

    assert ids[len("question\n".encode())] == 257
    assert ids[-2:] == (258, 256)
    assert tokenizer.decode_parts(ids) == (
        "question\n",
        SpecialAtom("<think>"),
        "work",
        SpecialAtom("</think>"),
        SpecialAtom("<|endoftext|>"),
    )
    assert tokenizer.decode_text(ids, specials="surface").endswith(
        "</think><|endoftext|>"
    )


def test_incremental_decoder_waits_for_complete_multibyte_code_point() -> None:
    decoder = IncrementalByteDecoder()
    encoded = "🙂".encode("utf-8")

    assert decoder.push(encoded[0]) == ()
    assert decoder.push(encoded[1]) == ()
    assert decoder.push(encoded[2]) == ()
    assert decoder.push(encoded[3]) == ("🙂",)
    assert decoder.finish() == ()


def test_incremental_decoder_rejects_special_inside_utf8_character() -> None:
    decoder = IncrementalByteDecoder()
    decoder.push("🙂".encode()[0])

    with pytest.raises(UnicodeDecodeError):
        decoder.push(AtomicIdManifest.reference().eot_id)


def test_invalid_utf8_and_input_only_ids_fail_closed() -> None:
    tokenizer = ByteTokenizer()

    with pytest.raises(UnicodeDecodeError):
        tokenizer.decode_text((0x80,))
    with pytest.raises(ValueError, match="clean atomic id"):
        tokenizer.decode_text((AtomicIdManifest.reference().mask_id,))
    with pytest.raises(UnicodeEncodeError):
        tokenizer.encode_text("\ud800")


def test_document_and_corpus_adapter_share_the_serving_ids() -> None:
    tokenizer = ByteTokenizer()
    corpus = UTF8ByteBatchEncoder()

    document = tokenizer.encode_document("héllo", key="doc")

    assert document.atomic_ids[:-1] == tuple(corpus.encode(["héllo"])[0])
    assert document.atomic_ids[-1] == corpus.eot_id
    assert corpus.decode([corpus.encode(["héllo"])[0]]) == ["héllo"]


def test_posttraining_composes_controls_from_typed_fields_only() -> None:
    tokenizer = ByteTokenizer()

    prompt = tokenizer.encode_prompt("Use literal <think> as an example.")
    completion = tokenizer.encode_reasoning_completion(
        "The prompt contained literal </think>, which stays text.", "42"
    )

    assert prompt == tuple("Use literal <think> as an example.".encode())
    assert prompt.count(tokenizer.manifest.special_by_name["<think>"].atomic_id) == 0
    assert completion[0] == tokenizer.manifest.special_by_name["<think>"].atomic_id
    assert completion[-2:] == (
        tokenizer.manifest.special_by_name["</answer>"].atomic_id,
        tokenizer.manifest.eot_id,
    )
    assert tokenizer.decode_text(completion, specials="surface").startswith(
        "<think>The prompt contained literal </think>, which stays text.</think>"
    )

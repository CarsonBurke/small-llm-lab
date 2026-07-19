from __future__ import annotations

import pytest

from postraining.sample_latent import decode_trace_with_think_markers


class _Tokenizer:
    def decode(self, ids: list[int]) -> str:
        return {(): "", (7,): "hello"}.get(tuple(ids), "decoded")

    def id_to_piece(self, token: int) -> str:
        return {5: "</s>", 7: "▁hello"}[token]


def test_cpu_trace_decode_marks_thoughts_and_preserves_piece_spacing():
    tokenizer = _Tokenizer()
    actual = decode_trace_with_think_markers(
        tokenizer, [7, 5], "tEE", stop_ids=(5,)
    )

    assert actual == "1🪙 hello</s>"


def test_cpu_trace_decode_rejects_misaligned_capture():
    tokenizer = _Tokenizer()
    with pytest.raises(ValueError, match="more EMITs"):
        decode_trace_with_think_markers(tokenizer, [7], "EE")

from __future__ import annotations

import numpy as np

from scripts.build_byte_diffusion_dataset import ChallengeDocumentReader
from scripts.build_math_mix_dataset import write_shard
from scripts.materialize_byte_corpus_gpt2_view import materialize_split


class _Target:
    def encode(self, texts, out_type=int):
        assert out_type is int
        return [[1_000 + byte for byte in text.encode("utf-8")] for text in texts]


def test_view_preserves_complete_documents_and_drops_only_source_tail(tmp_path) -> None:
    source = tmp_path / "source.bin"
    output = tmp_path / "target.bin"
    source_tokens = np.asarray(
        [256, *"héllo".encode(), 256, *"🙂".encode(), 256, *b"cut"],
        dtype=np.int32,
    )
    write_shard(source, source_tokens)

    stats = materialize_split(
        name="train",
        source_paths=(source,),
        output_path=output,
        source_eot_id=256,
        source_overlap=0,
        target_encoder=_Target(),
        target_eot_id=9,
        batch_size=1,
    )

    documents = list(ChallengeDocumentReader((output,), eot_id=9).iter_documents())
    assert documents == [
        (9,),
        (*[1_000 + byte for byte in "héllo".encode()], 9),
        (*[1_000 + byte for byte in "🙂".encode()], 9),
    ]
    assert stats["documents"] == 2
    assert stats["literal_utf8_bytes"] == len("héllo🙂".encode())
    assert stats["source_incomplete_tail_tokens"] == 3

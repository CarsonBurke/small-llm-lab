"""Literal byte stream views for CELF; sources are verified against their manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pretraining.nanogpt_mini.native_bits_data import sha256_file

BOUNDARY_POLICY = (
    "existing concatenated literal stream; no BOS/EOT insertion or document isolation"
)


def source_manifest(root: Path) -> dict:
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("normalization") != "none":
        raise ValueError("CELF requires an unnormalized literal byte stream")
    sources = {}
    for split, filename in (("train", "train.txt"), ("validation", "val.txt")):
        path = root / filename
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != metadata[split]["sha256"] or size != metadata[split]["bytes"]:
            raise ValueError(f"{split} byte stream differs from its manifest")
        if size == 0:
            raise ValueError(f"empty {split} byte stream")
        sources[split] = {"bytes": size, "sha256": digest}
    return sources


@torch.compile(fullgraph=True, dynamic=False)
def _cyclic_batch(source, offsets, cursor):
    return source[(offsets + cursor) % source.numel()].long()


class ByteCorpus:
    train: torch.Tensor
    validation: torch.Tensor

    def __init__(self, root: str | Path, device: torch.device):
        self.root = Path(root).resolve()
        sources = source_manifest(self.root)
        self.metadata: dict = {
            "root": str(self.root),
            "source_manifest_sha256": sha256_file(self.root / "metadata.json"),
            "boundary_policy": BOUNDARY_POLICY,
            "normalization": "none",
            "evaluation_scope": "FineWeb byte-cache validation_web proxy, not challenge heldout",
            "tokenization": "byte",
            "resident_dtype": "uint8",
        }
        for split, filename in (("train", "train.txt"), ("validation", "val.txt")):
            path = self.root / filename
            mapped = np.memmap(path, dtype=np.uint8, mode="r")
            tensor = torch.from_numpy(np.array(mapped, copy=True)).to(device)
            setattr(self, split, tensor)
            self.metadata[split] = {
                "path": str(path),
                **sources[split],
                "positions": tensor.numel(),
            }
        self._offsets: dict[int, torch.Tensor] = {}
        self._cursor = torch.empty((), device=device, dtype=torch.long)

    def training_batch(self, cursor: int, count: int, seq_bytes: int) -> torch.Tensor:
        if cursor < 0 or seq_bytes < 1 or count < 1 or count % seq_bytes:
            raise ValueError("invalid consumed-byte cursor or training shape")
        if count not in self._offsets:
            self._offsets[count] = torch.arange(count, device=self.train.device)
        self._cursor.fill_(cursor % self.train.numel())
        return _cyclic_batch(self.train, self._offsets[count], self._cursor).view(
            -1, seq_bytes
        )

    def validation_prefix(self, count: int, seq_bytes: int) -> torch.Tensor:
        if seq_bytes < 1 or not 0 < count <= self.validation.numel() or count % seq_bytes:
            raise ValueError("validation must be an in-range prefix of complete contexts")
        return self.validation[:count].long().view(-1, seq_bytes)

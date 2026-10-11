"""Streams and validation sets for byte/token comparisons on identical text.

Training: the byte corpus ``bd_mathglm_v6_bytes_2k`` and its GPT-2 view
``bd_mathglm_v6_gpt2_2k_v4`` hold the same documents in the same order, so a
contiguous stream from the start of each consumes the same text per step to
within the bytes-per-token ratio.

Validation: the matched token reference (`nanogpt_latest_mathglm_v6_matched_2k_v7`)
scores GPT-2 targets ``tokens[1 : 2_097_153]`` of the FineWeb validation
shard in 1,024 independent rows of 2,048 tokens. Its byte decoding is the
first ``sum(lut[tokens[:2_097_153]])`` bytes of the byte validation shard
(checked by aligning every end-of-text). The primary byte panel scores
exactly those bytes in rows cut at the same token boundaries, so every byte
has the same context start as the token that contains it (validation text
runs 4.42 bytes/token, so these rows average ~9,050 bytes). A secondary
panel cuts the same bytes into fixed rows of the training length.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

MAGIC = 20240520
HEADER_BYTES = 256 * 4
BYTE_DATA = Path("data/datasets/bd_mathglm_v6_bytes_2k")
GPT2_DATA = Path("data/datasets/bd_mathglm_v6_gpt2_2k_v4")
GPT2_BYTE_LUT = Path("data/tokenizers/gpt2_byte_lut.pt")
REFERENCE_VAL_TOKENS = 2_097_152
BYTE_EOT = 256
GPT2_EOT = 50_256
DOMAINS = ("web", "code", "math", "knowledge")


def shard_tokens(path: Path) -> np.memmap:
    header = np.fromfile(path, dtype=np.int32, count=256)
    if header[0] != MAGIC or header[1] != 1:
        raise ValueError(f"{path}: bad shard header")
    count = int(header[2])
    if path.stat().st_size != HEADER_BYTES + 2 * count:
        raise ValueError(f"{path}: size does not match header")
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES, shape=(count,))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContiguousStream:
    """One pass over the train shards in order; each step is a contiguous
    ``rows * seq_len + 1`` window reshaped into (inputs, targets) rows."""

    def __init__(self, data_dir: Path, rows: int, seq_len: int):
        self.files = sorted(data_dir.glob("fineweb_train_*.bin"))
        if not self.files:
            raise FileNotFoundError(f"no train shards in {data_dir}")
        self.rows, self.seq_len = rows, seq_len
        self.step_atoms = rows * seq_len
        self._file = -1
        self._pos = 0
        self._buf: Tensor | None = None

    def _load_next(self) -> None:
        self._file += 1
        if self._file >= len(self.files):
            raise RuntimeError("training stream exhausted; runs must not wrap an epoch")
        tokens = shard_tokens(self.files[self._file])
        self._buf = torch.from_numpy(np.asarray(tokens)).pin_memory()
        self._pos = 0

    def next_batch(self) -> tuple[Tensor, Tensor]:
        if self._buf is None or self._pos + self.step_atoms + 1 > self._buf.numel():
            self._load_next()
        window = self._buf[self._pos : self._pos + self.step_atoms + 1]
        self._pos += self.step_atoms
        window = window.to("cuda", non_blocking=True).long()
        return (
            window[:-1].view(self.rows, self.seq_len),
            window[1:].view(self.rows, self.seq_len),
        )


@dataclass
class EvalSet:
    """Rows of (inputs, targets); ignored targets are -100."""

    name: str
    inputs: Tensor
    targets: Tensor
    scored_bytes: int  # UTF-8 bytes of the scored targets (specials count 1)

    @property
    def rows(self) -> int:
        return self.inputs.size(0)


def _rows_from_stream(stream: np.ndarray, seq_len: int, pad: int) -> tuple[Tensor, Tensor]:
    """Split ``stream`` (N + 1 atoms) into rows scoring every target once."""
    stream = torch.from_numpy(stream.astype(np.int64))
    n_targets = stream.numel() - 1
    rows = -(-n_targets // seq_len)
    inputs = torch.full((rows * seq_len,), pad, dtype=torch.long)
    targets = torch.full((rows * seq_len,), -100, dtype=torch.long)
    inputs[:n_targets] = stream[:-1]
    targets[:n_targets] = stream[1:]
    return inputs.view(rows, seq_len), targets.view(rows, seq_len)


def _token_aligned_rows(byte_stream: np.ndarray, offsets: np.ndarray, tokens_per_row: int, multiple: int) -> tuple[Tensor, Tensor]:
    """Byte rows whose context starts where the token rows' do.

    Token row ``r`` reads ``tokens[r*n : (r+1)*n]`` and scores
    ``tokens[r*n+1 : (r+1)*n+1]``. Byte row ``r`` reads the bytes of the same
    input tokens and scores only the bytes of the same target tokens; the
    trailing bytes of its first token are context (row ``r-1`` scored them).
    """
    n_rows = (len(offsets) - 1 - 1) // tokens_per_row
    starts = offsets[np.arange(n_rows) * tokens_per_row]
    first_scored = offsets[np.arange(n_rows) * tokens_per_row + 1]
    ends = offsets[(np.arange(n_rows) + 1) * tokens_per_row + 1]  # exclusive target end
    length = int((ends - 1 - starts).max())
    length = -(-length // multiple) * multiple
    stream = torch.from_numpy(byte_stream.astype(np.int64))
    inputs = torch.full((n_rows, length), BYTE_EOT, dtype=torch.long)
    targets = torch.full((n_rows, length), -100, dtype=torch.long)
    for r, (a, f, e) in enumerate(zip(starts.tolist(), first_scored.tolist(), ends.tolist())):
        n = e - 1 - a
        inputs[r, :n] = stream[a : e - 1]
        targets[r, :n] = stream[a + 1 : e]
        targets[r, : f - 1 - a] = -100
    return inputs, targets


def reference_validation(atom: str, seq_len: int, aligned: bool = True) -> tuple[EvalSet, dict]:
    """Validation matched to the token reference's 2,097,152 scored tokens.

    For bytes, ``aligned`` cuts rows at the reference's token-row boundaries
    (rows padded to a multiple of 128); otherwise rows are ``seq_len`` bytes.
    """
    lut = torch.load(GPT2_BYTE_LUT).numpy().astype(np.int64)
    tokens = np.asarray(shard_tokens(GPT2_DATA / "fineweb_val_000000.bin")[: REFERENCE_VAL_TOKENS + 1]).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(lut[tokens])])
    span = int(offsets[-1])
    byte_stream = np.asarray(shard_tokens(BYTE_DATA / "fineweb_val_000000.bin")[:span]).astype(np.int64)
    token_eot = np.nonzero(tokens == GPT2_EOT)[0]
    byte_eot = np.nonzero(byte_stream == BYTE_EOT)[0]
    if not np.array_equal(offsets[token_eot], byte_eot):
        raise RuntimeError("byte and GPT-2 validation streams are not the same text")
    scored_bytes = span - int(lut[tokens[0]])
    if atom == "gpt2":
        if seq_len != 2048 or REFERENCE_VAL_TOKENS % seq_len:
            raise ValueError("the token reference scores 2048-token rows")
        inputs, targets = _rows_from_stream(tokens, seq_len, pad=GPT2_EOT)
    elif aligned:
        inputs, targets = _token_aligned_rows(byte_stream, offsets, 2048, multiple=128)
    else:
        inputs, targets = _rows_from_stream(byte_stream, seq_len, pad=BYTE_EOT)
        if int((targets >= 0).sum()) != scored_bytes:
            raise RuntimeError("byte validation target count drifted from the token span")
    info = {
        "reference_val_tokens": REFERENCE_VAL_TOKENS,
        "scored_bytes": scored_bytes,
        "documents": int(len(token_eot)),
        "val_token_shard": str(GPT2_DATA / "fineweb_val_000000.bin"),
        "val_byte_shard": str(BYTE_DATA / "fineweb_val_000000.bin"),
        "val_token_slice_sha256": hashlib.sha256(tokens.astype(np.uint16).tobytes()).hexdigest(),
        "val_byte_slice_sha256": hashlib.sha256(byte_stream.astype(np.uint16).tobytes()).hexdigest(),
        "rows": inputs.size(0),
        "row_length": inputs.size(1),
    }
    name = "val" if atom == "gpt2" or aligned else f"val_fixed{seq_len}"
    return EvalSet(name, inputs, targets, scored_bytes), info


def domain_validation(seq_len: int) -> list[EvalSet]:
    """Per-domain held-out byte panels (byte models only)."""
    sets = []
    for domain in DOMAINS:
        stream = np.asarray(shard_tokens(BYTE_DATA / f"domainval_{domain}_000000.bin"))
        inputs, targets = _rows_from_stream(stream, seq_len, pad=BYTE_EOT)
        sets.append(EvalSet(f"domain_{domain}", inputs, targets, int((targets >= 0).sum())))
    return sets


def data_identity(atom: str) -> dict:
    data_dir = GPT2_DATA if atom == "gpt2" else BYTE_DATA
    manifest = data_dir / "mix_manifest.json"
    return {
        "data_dir": str(data_dir),
        "mix_manifest_sha256": file_sha256(manifest),
        "train_shards": [
            {"path": str(p), "bytes": p.stat().st_size}
            for p in sorted(data_dir.glob("fineweb_train_*.bin"))
        ],
        "gpt2_byte_lut_sha256": file_sha256(GPT2_BYTE_LUT),
    }

"""Build per-token unigram counts for ENERGY_BIAS_INIT_COUNTS.

Reads challenge-format token shards (256 int32 header + uint16 payload) and
writes a JSON list of length ``--vocab-size``.  CPU-only and read-only.

    python3 energy_readout/build_unigram_counts.py \
        --data-glob 'data/datasets/mathmix_v4_sp1024/fineweb_train_*.bin' \
        --out energy_readout/unigram_counts_mathmix_v4.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

HEADER_BYTES = 256 * 4


def shard_counts(path: Path, vocab_size: int) -> np.ndarray:
    payload_bytes = path.stat().st_size - HEADER_BYTES
    if payload_bytes < 0 or payload_bytes % 2:
        raise ValueError(f"invalid token shard size: {path}")
    tokens = np.fromfile(path, dtype=np.uint16, offset=HEADER_BYTES)
    if tokens.size and int(tokens.max()) >= vocab_size:
        raise ValueError(
            f"{path} holds token id {int(tokens.max())} >= vocab {vocab_size}"
        )
    return np.bincount(tokens, minlength=vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-glob", required=True)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    files = [Path(path) for path in sorted(glob.glob(args.data_glob))]
    if not files:
        raise FileNotFoundError(f"no shards match {args.data_glob!r}")
    totals = np.zeros(args.vocab_size, dtype=np.int64)
    for path in files:
        totals += shard_counts(path, args.vocab_size)
    Path(args.out).write_text(json.dumps(totals.tolist()) + "\n")
    covered = int((totals > 0).sum())
    print(
        f"wrote {args.out}: {int(totals.sum()):,} tokens over {len(files)} "
        f"shards; {covered}/{args.vocab_size} vocab ids observed"
    )


if __name__ == "__main__":
    main()

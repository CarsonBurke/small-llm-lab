#!/usr/bin/env python3
"""Recover missing fixed-panel inputs for the original latent-JEPA sum bench.

Usage: python3 scripts/recover_latent_jepa_references.py --run-dir PATH

This is CPU-only data reconstruction, not model inference. New sample writers
should save `numbers` directly. This deliberately supports only the audited
task/bench snapshots and RTX 5090 launch layout, rather than guessing when a
sampler changes. It replays only the FIRST torch.randint CUDA call: 2048 x 12
values, seed 1, offset 0. That launch uses one thread per output element, each
consuming curand4().x. Subsequent rejection-sampler calls are unnecessary for
the four saved examples and are not implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_HASHES = {
    "task.py": "1014a92350f02bbc60e57394b31380f350d1f7420ceb3140cffec2ad70025b34",
    "bench.py": "b12ffca01c50290926d7b559e5ce71dfaabbd9e552bb8cd9d0db7aa815d57fed",
    "pretrain.py": "337c4e2832c42babf2fbd9318dbe7fbf56f51a42091c9c027783e1a16022ff17",
}
MASK = (1 << 32) - 1


def philox(seed: int, subsequence: int) -> tuple[int, int, int, int]:
    """cuRAND Philox4x32-10 at offset zero (four 32-bit words)."""
    x, y, z, w = 0, 0, subsequence & MASK, subsequence >> 32
    k0, k1 = seed & MASK, seed >> 32
    for _ in range(10):
        p0, p1 = 0xD2511F53 * x, 0xCD9E8D57 * z
        x, y, z, w = (p1 >> 32) ^ y ^ k0, p1 & MASK, (p0 >> 32) ^ w ^ k1, p0 & MASK
        k0, k1 = (k0 + 0x9E3779B9) & MASK, (k1 + 0xBB67AE85) & MASK
    return x, y, z, w


def first_panel_rows(count: int = 4) -> list[list[int]]:
    if not 1 <= count <= 4:
        raise ValueError("only the first four saved benchmark examples are supported")
    rows = []
    for j in range(2048):
        row = [10 + philox(1, j * 12 + i)[0] % 90 for i in range(12)]
        mixed = sum(v * (1_000_003 * (i + 1)) for i, v in enumerate(row))
        # Match signed int64 overflow and arithmetic right shift in task.py.
        mixed = ((mixed ^ (mixed >> 7)) * 2_654_435_761) & ((1 << 64) - 1)
        if mixed >= 1 << 63:
            mixed -= 1 << 64
        if (mixed >> 11) % 64 == 0:
            rows.append(row)
            if len(rows) == count:
                return rows
    raise ValueError("first CUDA draw did not contain enough held-out rows")


def recover(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("device") != "NVIDIA GeForce RTX 5090":
        raise ValueError("unverified CUDA launch layout: expected RTX 5090")
    if manifest.get("torch") != "2.13.0+cu130":
        raise ValueError("unverified torch CUDA RNG implementation")
    for name, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((run_dir / "source" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"unsupported saved sampler source: {name}")
    task = manifest["contract"]["task"]
    if task != dict(schema="latent_jepa_sum_task_v1", terms=12, write_every=4, vocab=256, heldout_modulus=64):
        raise ValueError("unsupported task contract")
    numbers = first_panel_rows()
    counts: dict[tuple[int, str], int] = {}
    checked = 0
    for line in (run_dir / "samples.jsonl").read_text().splitlines():
        sample = json.loads(line)
        panel = sample["panel"]
        if panel in ("free_mean", "greedy"):
            key = sample["step"], panel
            index = counts.get(key, 0)
            counts[key] = index + 1
        elif panel.startswith("free_s"):
            index = 0
        else:
            raise ValueError(f"unsupported saved panel: {panel}")
        if index >= len(numbers) or sum(numbers[index]) != sample["target"]:
            raise ValueError(f"reconstructed inputs disagree with target at step {sample['step']}, {panel}")
        checked += 1
    if not checked:
        raise ValueError("no saved targets available to cross-check reconstruction")
    return dict(schema="latent_jepa_sum_panel_v1", task=task, seed=1, size=1024,
                device="cuda", numbers=numbers, source_hashes=SOURCE_HASHES,
                method="CPU replay of the saved CUDA Philox sampler's first draw; source hashes checked, "
                       f"all {checked} saved target sums cross-checked. No model execution.",
                coverage="First four fixed-panel inputs only; not the full benchmark panel")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = recover(args.run_dir)
    output = args.output or args.run_dir / "references.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()

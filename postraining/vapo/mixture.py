"""Immutable, deterministic multi-source prompt schedules for VAPO."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from postraining.core import load_unique_math_rows


VAPO_MIXTURE_SCHEMA = "vapo_verifiable_mixture/v1"
VERIFIER_KINDS = {"math", "python_mbpp"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MixtureSource:
    name: str
    path: Path
    quota: int
    verifier: str
    rows: tuple[dict, ...]


def _balanced_schedule(quotas: dict[str, int]) -> tuple[str, ...]:
    """Spread each source across a cycle while preserving exact quotas."""
    if not quotas or any(quota < 1 for quota in quotas.values()):
        raise ValueError("mixture source quotas must be positive")
    remaining = dict(quotas)
    used = Counter()
    total = sum(quotas.values())
    schedule = []
    for position in range(total):
        candidates = [name for name, count in remaining.items() if count]
        # Largest deficit from its ideal cumulative allocation wins. Stable
        # lexical tie-breaking makes the schedule independent of JSON order.
        name = max(
            candidates,
            key=lambda candidate: (
                (position + 1) * quotas[candidate] / total - used[candidate],
                candidate,
            ),
        )
        schedule.append(name)
        remaining[name] -= 1
        used[name] += 1
    if Counter(schedule) != Counter(quotas):
        raise AssertionError("balanced schedule changed source quotas")
    return tuple(schedule)


def load_mixture_manifest(
    path: str | Path,
) -> tuple[list[dict], list[MixtureSource], dict]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != VAPO_MIXTURE_SCHEMA:
        raise ValueError(f"unsupported VAPO mixture schema in {manifest_path}")
    entries = manifest.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError("VAPO mixture manifest needs nonempty sources")
    names = [str(entry.get("name", "")) for entry in entries]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("VAPO mixture source names must be unique and nonempty")

    sources = []
    all_rows = []
    global_identities: dict[str, str] = {}
    for entry in entries:
        name = str(entry["name"])
        source_path = Path(entry["path"])
        quota = int(entry["quota"])
        verifier = str(entry["verifier"])
        if quota < 1:
            raise ValueError(f"source {name!r} has a nonpositive quota")
        if verifier not in VERIFIER_KINDS:
            raise ValueError(f"source {name!r} has unknown verifier {verifier!r}")
        if file_sha256(source_path) != entry.get("sha256"):
            raise ValueError(f"source {name!r} bytes differ from its manifest")
        rows = load_unique_math_rows(source_path)
        if len(rows) != int(entry.get("rows", -1)):
            raise ValueError(f"source {name!r} logical row count changed")
        stamped = []
        for row in rows:
            info = row.get("extra_info") or {}
            identity = str(info.get("index", row["prompt"][0]["content"]))
            qualified = f"{name}:{identity}"
            if qualified in global_identities:
                raise ValueError(f"duplicate qualified prompt identity {qualified!r}")
            global_identities[qualified] = name
            stamped.append(
                {
                    **row,
                    "_rl_source": name,
                    "_verifier_kind": verifier,
                    "_qualified_identity": qualified,
                }
            )
        source = MixtureSource(
            name=name,
            path=source_path,
            quota=quota,
            verifier=verifier,
            rows=tuple(stamped),
        )
        sources.append(source)
        all_rows.extend(stamped)

    expected_cycle = int(manifest.get("groups_per_cycle", -1))
    observed_cycle = sum(source.quota for source in sources)
    if observed_cycle != expected_cycle:
        raise ValueError(
            f"mixture quotas sum to {observed_cycle}, not {expected_cycle}"
        )
    return all_rows, sources, manifest


def mixture_identity(path: str | Path, manifest: dict) -> str:
    digest = hashlib.sha256()
    digest.update(VAPO_MIXTURE_SCHEMA.encode())
    digest.update(b"\0")
    digest.update(file_sha256(path).encode())
    for entry in manifest["sources"]:
        digest.update(b"\0")
        digest.update(str(entry["name"]).encode())
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode())
    return "sha256:" + digest.hexdigest()


class MixedPromptSampler:
    """Exact-quota source cycle with independent, resumable row streams."""

    def __init__(
        self,
        sources: list[MixtureSource],
        seed: int,
        dataset_identity: str,
        cursor: int = 0,
    ):
        if cursor < 0:
            raise ValueError("sampler cursor must be nonnegative")
        self.sources = {source.name: source for source in sources}
        self.schedule = _balanced_schedule(
            {source.name: source.quota for source in sources}
        )
        self.seed = seed
        self.cursor = cursor
        self.dataset_identity = dataset_identity
        self.rows = [row for source in sources for row in source.rows]
        self._orders: dict[tuple[str, int], list[int]] = {}

    @property
    def epoch(self) -> int:
        return self.cursor // len(self.schedule)

    def _consumed_before(self, source: str, position: int) -> int:
        cycles, remainder = divmod(position, len(self.schedule))
        per_cycle = self.schedule.count(source)
        return cycles * per_cycle + self.schedule[:remainder].count(source)

    def _source_order(self, source: MixtureSource, epoch: int) -> list[int]:
        key = (source.name, epoch)
        if key not in self._orders:
            order = list(range(len(source.rows)))
            if epoch > 0:
                source_seed = int.from_bytes(
                    hashlib.sha256(source.name.encode()).digest()[:8], "big"
                )
                random.Random(
                    self.seed * 1_000_003 + source_seed + epoch
                ).shuffle(order)
            self._orders[key] = order
        return self._orders[key]

    def next_rows(self, count: int) -> list[dict]:
        if count < 0:
            raise ValueError("prompt count must be nonnegative")
        picked = []
        for position in range(self.cursor, self.cursor + count):
            source_name = self.schedule[position % len(self.schedule)]
            source = self.sources[source_name]
            local = self._consumed_before(source_name, position)
            epoch, offset = divmod(local, len(source.rows))
            order = self._source_order(source, epoch)
            picked.append(source.rows[order[offset]])
        self.cursor += count
        return picked

    def source_counts(self, start: int, count: int) -> dict[str, int]:
        return dict(
            Counter(
                self.schedule[position % len(self.schedule)]
                for position in range(start, start + count)
            )
        )

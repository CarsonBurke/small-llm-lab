"""Immutable, deterministic multi-source prompt schedules for VAPO."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from postraining.core import (
    load_unique_math_rows,
    math_corpus_identity,
    math_corpus_policy_sha256,
)
from postraining.vapo.choice_sampling import CHOICE_PRESENTATION_SCHEMA, randomize_choice_row


# v2 is an exact-pass mixture: a source has no quota, one cycle serves every
# row of every source exactly once, and sources are interleaved in proportion
# to their sizes. v1 manifests carried free per-source quotas, which revisit
# small sources many times per pass over a large one.
VAPO_MIXTURE_SCHEMA = "vapo_verifiable_mixture/v2"
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
    verifier: str
    rows: tuple[dict, ...]


def _balanced_schedule(counts: dict[str, int]) -> tuple[str, ...]:
    """Interleave sources so every prefix tracks its proportional share.

    Each position goes to the source with the largest deficit from its ideal
    cumulative allocation, so every prefix, and hence every window, tracks
    each source's exact share to within about one prompt. The bound is not
    a hard constant: exhaustive checks put it below 1 prompt per prefix (1.5
    per window) for up to three sources, including the production corpus
    (0.77 and 1.17), and slightly above 1 for four or more sources.
    """
    if not counts or any(count < 1 for count in counts.values()):
        raise ValueError("mixture sources must be nonempty")
    total = sum(counts.values())
    names = sorted(counts)
    used = dict.fromkeys(names, 0)
    schedule = []
    for position in range(1, total + 1):
        # Stable lexical tie-breaking makes the schedule independent of
        # manifest order. A source with rows left always has a positive
        # deficit before any exhausted one, so exhaustion needs no check.
        name = max(
            names,
            key=lambda candidate: (
                position * counts[candidate] / total - used[candidate],
                candidate,
            ),
        )
        schedule.append(name)
        used[name] += 1
    if used != counts:
        raise AssertionError("balanced schedule changed source counts")
    return tuple(schedule)


def load_mixture_manifest(
    path: str | Path,
) -> tuple[list[dict], list[MixtureSource], dict]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != VAPO_MIXTURE_SCHEMA:
        raise ValueError(
            f"unsupported VAPO mixture schema in {manifest_path}; "
            "prepare a new immutable mixture manifest"
        )
    if manifest.get("math_corpus_policy_sha256") != math_corpus_policy_sha256():
        raise ValueError(
            "mixture manifest lacks the current math corpus policy; "
            "prepare a new immutable mixture manifest, do not reuse its cursor"
        )
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
        if "quota" in entry:
            raise ValueError(
                f"source {name!r} has a quota; {VAPO_MIXTURE_SCHEMA} mixtures "
                "take each source's share from its row count"
            )
        verifier = str(entry["verifier"])
        if verifier not in VERIFIER_KINDS:
            raise ValueError(f"source {name!r} has unknown verifier {verifier!r}")
        if not entry.get("math_corpus_identity"):
            raise ValueError(
                f"source {name!r} lacks an effective corpus identity; "
                "prepare a new immutable mixture manifest"
            )
        if file_sha256(source_path) != entry.get("sha256"):
            raise ValueError(f"source {name!r} bytes differ from its manifest")
        rows = load_unique_math_rows(source_path)
        if math_corpus_identity(rows) != entry["math_corpus_identity"]:
            raise ValueError(
                f"source {name!r} effective corpus changed; prepare a new "
                "immutable mixture manifest, do not reuse its cursor"
            )
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
        if not stamped:
            raise ValueError(f"source {name!r} has no rows")
        source = MixtureSource(
            name=name,
            path=source_path,
            verifier=verifier,
            rows=tuple(stamped),
        )
        sources.append(source)
        all_rows.extend(stamped)

    if int(manifest.get("prompts_per_cycle", -1)) != len(all_rows):
        raise ValueError(
            f"mixture sources hold {len(all_rows)} rows, not the manifest's "
            f"prompts_per_cycle {manifest.get('prompts_per_cycle')}"
        )
    return all_rows, sources, manifest


def mixture_identity(
    path: str | Path, manifest: dict, *, randomize_choice_options: bool = False,
) -> str:
    digest = hashlib.sha256()
    digest.update(VAPO_MIXTURE_SCHEMA.encode())
    digest.update(b"\0")
    digest.update(file_sha256(path).encode())
    digest.update(b"\0")
    digest.update(math_corpus_policy_sha256().encode())
    for entry in manifest["sources"]:
        digest.update(b"\0")
        digest.update(str(entry["name"]).encode())
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode())
        digest.update(b"\0")
        digest.update(str(entry["math_corpus_identity"]).encode())
    if randomize_choice_options:
        digest.update(b"\0")
        digest.update(CHOICE_PRESENTATION_SCHEMA.encode())
    return "sha256:" + digest.hexdigest()


class MixedPromptSampler:
    """Exact-pass traversal of a mixture, resumable from a prompt cursor.

    Cycle ``k`` serves every row of every source exactly once: each source's
    rows in a per-cycle order (file order in cycle 0, then a deterministic
    reshuffle from ``seed``) and the sources interleaved by the balanced
    schedule, so any rollout window holds each source's share of the corpus
    to within about one prompt.
    """

    def __init__(
        self,
        sources: list[MixtureSource],
        seed: int,
        dataset_identity: str,
        cursor: int = 0,
        *,
        randomize_choice_options: bool = True,
    ):
        if cursor < 0:
            raise ValueError("sampler cursor must be nonnegative")
        self.sources = {source.name: source for source in sources}
        self.schedule = _balanced_schedule(
            {source.name: len(source.rows) for source in sources}
        )
        # A position's rank within its own source's stream, precomputed so a
        # draw is O(1) over a cycle as long as the corpus.
        seen: Counter[str] = Counter()
        ranks = []
        for name in self.schedule:
            ranks.append(seen[name])
            seen[name] += 1
        self._ranks = tuple(ranks)
        self.seed = seed
        self.cursor = cursor
        self.dataset_identity = dataset_identity
        self.randomize_choice_options = randomize_choice_options
        self.rows = [row for source in sources for row in source.rows]
        self._orders: dict[tuple[str, int], list[int]] = {}

    @property
    def epoch(self) -> int:
        return self.cursor // len(self.schedule)

    def _source_order(self, source: MixtureSource, epoch: int) -> list[int]:
        key = (source.name, epoch)
        if key not in self._orders:
            # The cursor only advances, so earlier cycles' orders are dead.
            for stale in [
                cached for cached in self._orders
                if cached[0] == source.name and cached[1] < epoch
            ]:
                del self._orders[stale]
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
            epoch, phase = divmod(position, len(self.schedule))
            source = self.sources[self.schedule[phase]]
            order = self._source_order(source, epoch)
            row = source.rows[order[self._ranks[phase]]]
            if self.randomize_choice_options:
                row = randomize_choice_row(row, seed=self.seed, cursor=position)
            picked.append(row)
        self.cursor += count
        return picked

    def source_counts(self, start: int, count: int) -> dict[str, int]:
        return dict(
            Counter(
                self.schedule[position % len(self.schedule)]
                for position in range(start, start + count)
            )
        )

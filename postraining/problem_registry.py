"""One global assignment of every math problem to exactly one split.

The `--heldout` list the corpus builders use today has two holes that a
2026-08-06 audit measured directly. GSM8K was never on the list, so 97.5% of
`openmath_gsm8k`, 97.0% of `gsm8k_socratic`, 96.7% of `had653_gold` and 96.9%
of the GSM8K RL pool were problems the base model had already pretrained on.
And for web sources the exclusion is structurally inert: `quality_keys` is the
whole document chunk, so it only fires when an entire web page equals a bare
problem statement.

A per-builder flag list cannot fix that, because the invariant is global: a
problem must belong to one split across pretraining, SFT, RL, and evaluation
at once. This module owns that invariant. Every builder consults the registry
instead of maintaining its own idea of what to exclude.

Priority runs ``eval > rl > sft > pretrain``. A problem claimed by a
higher-priority split is excluded from every lower one, so evaluation
integrity wins over post-training convenience, which in turn wins over corpus
volume.

Identity uses the same normalization and personalised digest the corpus
builder already uses for held-out keys, so registry keys and existing
`pgolf-holdout` keys agree on what "the same problem" means.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REGISTRY_SCHEMA = "math_problem_registry/v1"
DIGEST_PERSON = b"pgolf-holdout"
DIGEST_SIZE = 16

_WHITESPACE = re.compile(r"\s+")

# Highest priority first. A problem is assigned to the first split that claims
# it, so eval membership can never be overridden by a training source.
SPLIT_PRIORITY = ("eval", "rl", "sft", "pretrain")

# The same clauses `postraining.math_prompt.SOURCE_INSTRUCTIONS` removes. They
# are duplicated rather than imported so the registry stays importable without
# torch, which every corpus builder that consults it would otherwise pull in.
# `tests/test_problem_registry.py` asserts the two lists agree, so a new source
# template cannot be added to one and forgotten in the other.
FRAMING_INSTRUCTIONS = (
    "Solve the following math problem step by step.",
    "The last line of your response should be of the form Answer: $Answer "
    "(without quotes) where $Answer is the answer to the problem.",
    'Remember to put your answer on its own line after "Answer:".',
    "请以“Answer: \\boxed{<final_answer>}”的格式输出最终答案。",
    "让我们一步一步地思考。",
    "Start your response with <think> and reason until </think>, then end it "
    "with only the final answer inside <answer></answer>.",
    "Remember to start with <think> and put only the final answer inside "
    "<answer></answer>.",
    "请以 <think> 开始思考，并仅将最终答案放在 <answer></answer> 中。",
)


def normalize_problem(text: str) -> str:
    """Case and spacing normalization that preserves mathematical operators."""
    collapsed = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()
    return collapsed.casefold()


def strip_framing(text: str) -> str:
    """Remove every recognized instruction wrapper from a prompt.

    Unlike `math_prompt.strip_math_prompt_framing` this never raises on a
    surviving ``Answer:`` demand. The registry ingests raw source pools whose
    problem statements may legitimately contain that word, and refusing to
    hash such a problem would silently drop it from decontamination -- the
    opposite of failing closed.
    """
    for instruction in FRAMING_INSTRUCTIONS:
        if instruction in text:
            text = text.replace(instruction, "")
    return text


def problem_key(text: str) -> bytes:
    """Canonical 16-byte identity of a problem statement."""
    return hashlib.blake2b(
        normalize_problem(strip_framing(text)).encode("utf-8", errors="ignore"),
        digest_size=DIGEST_SIZE,
        person=DIGEST_PERSON,
    ).digest()


def prompt_content(value: object) -> str | None:
    """Pull the problem text out of the several shapes the parquets use."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        return content if isinstance(content, str) else None
    if isinstance(value, list):
        # A chat-formatted prompt: the problem is the user turn. Preferring it
        # over the first message keeps a system preamble out of the key, and
        # joining every turn would make an identical problem hash differently
        # depending on which template wrapped it.
        for item in value:
            if isinstance(item, dict) and item.get("role") == "user":
                content = prompt_content(item)
                if content:
                    return content
        for item in value:
            content = prompt_content(item)
            if content:
                return content
    return None


def read_problems(path: Path, columns: Iterable[str]) -> Iterator[str]:
    """Yield problem statements from the first present column of ``columns``."""
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    chosen = next((name for name in columns if name in available), None)
    if chosen is None:
        raise ValueError(
            f"{path} has none of the problem columns {list(columns)}; it has "
            f"{sorted(available)}"
        )
    for batch in parquet.iter_batches(batch_size=8192, columns=[chosen]):
        for value in batch[chosen].to_pylist():
            text = prompt_content(value)
            if text:
                yield text


@dataclass(frozen=True)
class SourceDeclaration:
    """One file's problems and the split they belong to."""

    name: str
    path: Path
    split: str
    columns: tuple[str, ...] = ("problem", "prompt", "question")

    def __post_init__(self) -> None:
        if self.split not in SPLIT_PRIORITY:
            raise ValueError(
                f"source {self.name!r} declares unknown split {self.split!r}; "
                f"expected one of {SPLIT_PRIORITY}"
            )

    def digest(self) -> str:
        """SHA-256 of the declared file, streamed."""
        hasher = hashlib.sha256()
        with self.path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(block)
        return hasher.hexdigest()

    @classmethod
    def from_json(cls, payload: dict, root: Path) -> SourceDeclaration:
        return cls(
            name=payload["name"],
            path=root / payload["path"],
            split=payload["split"],
            columns=tuple(
                payload.get("columns", ("problem", "prompt", "question"))
            ),
        )


class ProblemRegistry:
    """Immutable key to split mapping, with the provenance that produced it."""

    def __init__(self, assignment: dict[bytes, str], provenance: dict):
        self._assignment = assignment
        self.provenance = provenance

    def __len__(self) -> int:
        return len(self._assignment)

    def split_of(self, text: str) -> str | None:
        return self._assignment.get(problem_key(text))

    def key_split(self, key: bytes) -> str | None:
        return self._assignment.get(key)

    @property
    def split_counts(self) -> dict[str, int]:
        """Distinct problems per split, counted from the assignment itself.

        Derived rather than read from provenance, so a consumer deciding what
        needs protection cannot be misled by a stale manifest.
        """
        counts = {split: 0 for split in SPLIT_PRIORITY}
        for split in self._assignment.values():
            counts[split] += 1
        return counts

    def keys_for(self, split: str) -> set[bytes]:
        return {key for key, value in self._assignment.items() if value == split}

    def excluded_from(self, split: str) -> set[bytes]:
        """Keys a builder for ``split`` must refuse.

        Everything claimed by a split of higher priority. Pretraining, the
        lowest priority, is therefore excluded from every problem that any
        evaluation, RL, or SFT set uses.
        """
        if split not in SPLIT_PRIORITY:
            raise ValueError(f"unknown split {split!r}")
        rank = SPLIT_PRIORITY.index(split)
        higher = set(SPLIT_PRIORITY[:rank])
        return {key for key, value in self._assignment.items() if value in higher}

    # -- construction ----------------------------------------------------

    @classmethod
    def build(
        cls, sources: list[SourceDeclaration], *, log=lambda message: None
    ) -> ProblemRegistry:
        assignment: dict[bytes, str] = {}
        per_source: dict[str, dict] = {}
        conflicts: dict[str, int] = {}
        # Highest priority first, so a later, lower-priority claim never wins.
        ordered = sorted(sources, key=lambda s: SPLIT_PRIORITY.index(s.split))
        for source in ordered:
            rows = 0
            claimed = 0
            # Several pools repeat each problem for rollout batching, so rows
            # are a poor unit for conflict reporting; count distinct problems.
            lost: set[bytes] = set()
            distinct: set[bytes] = set()
            for text in read_problems(source.path, source.columns):
                rows += 1
                key = problem_key(text)
                distinct.add(key)
                current = assignment.get(key)
                if current is None:
                    assignment[key] = source.split
                    claimed += 1
                elif current != source.split:
                    lost.add(key)
            per_source[source.name] = {
                "split": source.split,
                "path": str(source.path),
                # The bytes this claim came from. Without it, a registry can
                # be reproduced only by trusting that a path still holds what
                # it held, which is the assumption every other artifact in
                # this repository refuses to make.
                "sha256": source.digest(),
                "rows": rows,
                "distinct_problems": len(distinct),
                "claimed": claimed,
                "yielded_to_higher_priority": len(lost),
            }
            if lost:
                conflicts[source.name] = len(lost)
            log(
                f"{source.name:40s} {source.split:8s} rows {rows:>9,} "
                f"distinct {len(distinct):>9,} claimed {claimed:>9,} "
                f"yielded {len(lost):>8,}"
            )
        registry = cls(assignment, {})
        registry.provenance = {
            "schema": REGISTRY_SCHEMA,
            "sources": per_source,
            "split_counts": registry.split_counts,
            "conflicts": conflicts,
            "total_problems": len(assignment),
        }
        return registry

    # -- storage ---------------------------------------------------------

    def write(self, directory: Path) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        items = sorted(self._assignment.items())
        table = pa.table(
            {
                "key": pa.array([key for key, _ in items], type=pa.binary(DIGEST_SIZE)),
                "split": pa.array([value for _, value in items], type=pa.string()),
            }
        )
        path = directory / "registry.parquet"
        pq.write_table(table, path, compression="zstd")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Update in place so provenance read off a freshly built registry is
        # the same dict as provenance read back from disk.
        self.provenance["registry_sha256"] = digest
        (directory / "manifest.json").write_text(
            json.dumps(self.provenance, indent=2, sort_keys=True) + "\n"
        )
        return digest

    @classmethod
    def read(cls, directory: Path) -> ProblemRegistry:
        directory = Path(directory)
        path = directory / "registry.parquet"
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["schema"] != REGISTRY_SCHEMA:
            raise ValueError(
                f"registry schema {manifest['schema']!r} is not {REGISTRY_SCHEMA!r}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["registry_sha256"]:
            raise ValueError(
                f"{path} hashes to {digest}, manifest records "
                f"{manifest['registry_sha256']}; the registry is not the one "
                "its manifest describes"
            )
        table = pq.read_table(path)
        assignment = dict(
            zip(table["key"].to_pylist(), table["split"].to_pylist(), strict=True)
        )
        return cls(assignment, manifest)


def load_sources(path: Path, root: Path) -> list[SourceDeclaration]:
    payload = json.loads(Path(path).read_text())
    sources = [
        SourceDeclaration.from_json(entry, root) for entry in payload["sources"]
    ]
    # Per-source provenance is keyed by name, so a duplicate would overwrite
    # another source's record and make the manifest quietly incomplete.
    duplicates = sorted(
        name for name, count in Counter(s.name for s in sources).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"{path} declares duplicate source names: {duplicates}")
    return sources

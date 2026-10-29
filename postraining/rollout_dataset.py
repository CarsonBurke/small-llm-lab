"""Frozen-policy rollout datasets: every trajectory of a rollout-only run.

A critic learnability study fits several critics to the same frozen-policy
trajectories. Re-collecting them per critic would cost a rollout per arm
and pair the arms only through the RNG; recording them once makes every
arm see byte-identical data, and a held-out panel's trajectories can be
revisited by later tools (branch rollouts, evaluation) without regenerating
them.

The trainer's ``--rollout-only --rollout-dataset`` writes one parquet per
rollout pool, atomically, as each pool finishes scoring, so a crash keeps
every completed pool. ``manifest.json`` binds the directory to the run that
wrote it: the checkpoint, the ``--curriculum-init`` snapshot whose actor
sampled it if any (``policy_state_sha256``), the prompt stream, and the
rollout contract the rewards were scored under, and it lists every
completed pool with its digest. It is marked complete when its last planned pool is published;
readers refuse an incomplete dataset (a collection cut short by a crash or
a time limit), whose missing pools would silently shrink every study fit
on it.

Only token policies are recorded (reasoning mode ``cot``: pinned EMIT, no
carried belief). Their stream is a function of the token ids alone, so
``group_batch`` rebuilds exactly the ``LatentRolloutBatch`` the rollout
produced, field for field, apart from the behavior statistics it never
filled.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from postraining.latent_rollout import (
    PAD_SLOT,
    TOKEN_SLOT,
    LatentRolloutBatch,
    assign_terminal_rewards,
    emitted_token_rows,
)
from postraining.latent_thought import EMIT

ROLLOUT_DATASET_SCHEMA = "frozen_policy_rollouts/v1"

POOL_SCHEMA = pa.schema(
    [
        ("pool", pa.int32()),
        ("group", pa.int32()),
        ("sample", pa.int32()),
        ("source", pa.string()),
        ("prompt_index", pa.string()),
        ("prompt_text", pa.string()),
        ("prompt_sha256", pa.string()),
        ("ground_truth", pa.string()),
        ("answer_style", pa.string()),
        ("module", pa.string()),
        ("prompt_tokens", pa.list_(pa.int32())),
        ("completion_tokens", pa.list_(pa.int32())),
        ("reward", pa.float32()),
        ("structural_format_ok", pa.bool_()),
        ("terminated", pa.bool_()),
        ("parsed_answer", pa.string()),
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, write) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prompt_sha256(text: str) -> str:
    """The prompt identity a dataset groups and splits trajectories by."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pool_filename(pool: int) -> str:
    return f"pool_{pool:05d}.parquet"


class RolloutDatasetWriter:
    """Record each scored rollout group; publish one parquet per pool.

    ``write`` is called once per scored prompt group, possibly from the
    trainer's scoring worker thread; ``finish_pool`` runs on the main thread
    after ``collect`` has joined that worker, so the pool's records are
    complete and in scoring order when they are written.
    """

    def __init__(self, directory: Path, binding: dict, stop_ids: Sequence[int], planned_pools: int):
        self.directory = directory
        self.stop_ids = frozenset(int(token) for token in stop_ids)
        if not self.stop_ids:
            raise ValueError("a rollout dataset needs the scorer's stop ids")
        if planned_pools < 1:
            raise ValueError("a rollout dataset plans at least one pool")
        # A new destination per collection: appending pools to an existing
        # dataset would mix runs under one manifest.
        directory.mkdir(parents=True, exist_ok=False)
        self.manifest = {
            "schema": ROLLOUT_DATASET_SCHEMA,
            "binding": binding,
            "planned_pools": planned_pools,
            "complete": False,
            "pools": [],
        }
        self._save_manifest()
        self.lock = threading.Lock()
        self.pool = 0
        self.groups = 0
        self.records: list[dict] = []

    def _save_manifest(self) -> None:
        payload = json.dumps(self.manifest, indent=2, sort_keys=True) + "\n"
        _atomic_write(
            self.directory / "manifest.json",
            lambda path: path.write_text(payload, encoding="utf-8"),
        )

    def write(self, batch: LatentRolloutBatch, verdicts, row: dict, prompt: str) -> None:
        if batch.carry_injected or batch.memory_carry or batch.thoughts.size(-1):
            raise ValueError("a rollout dataset records token-policy streams only")
        if bool((batch.kind == TOKEN_SLOT).logical_or(batch.kind == PAD_SLOT).logical_not().any()):
            raise ValueError("a rollout dataset records pinned-EMIT streams only")
        if row.get("_verifier_kind", "math") != "math":
            raise ValueError("a rollout dataset records math-verified rows only")
        prompt_slots = batch.kind[0, : batch.prompt_length] != PAD_SLOT
        prompt_tokens = batch.token_ids[0, : batch.prompt_length][prompt_slots]
        if bool((batch.kind[:, : batch.prompt_length] != batch.kind[:1, : batch.prompt_length]).any()) or bool(
            (batch.token_ids[:, : batch.prompt_length] != batch.token_ids[:1, : batch.prompt_length]).any()
        ):
            raise ValueError("a rollout group shares one prompt")
        prompt_tokens = prompt_tokens.tolist()
        completions = emitted_token_rows(batch)
        rewards = batch.reward_scalar.tolist()
        info = row.get("extra_info") or {}
        truth = row["reward_model"]["ground_truth"]
        with self.lock:
            group = self.groups
            self.groups += 1
            for sample, (tokens, verdict, reward) in enumerate(
                zip(completions, verdicts, rewards, strict=True)
            ):
                self.records.append(
                    {
                        "pool": self.pool,
                        "group": group,
                        "sample": sample,
                        "source": str(row.get("_rl_source", "math")),
                        "prompt_index": None if info.get("index") is None else str(info["index"]),
                        "prompt_text": prompt,
                        "prompt_sha256": prompt_sha256(prompt),
                        "ground_truth": str(truth),
                        "answer_style": str(row["reward_model"].get("style", "")),
                        "module": None if info.get("module") is None else str(info["module"]),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": tokens,
                        "reward": float(reward),
                        "structural_format_ok": bool(verdict.format_ok),
                        # The rollout records the stop token it ended on.
                        "terminated": any(token in self.stop_ids for token in tokens),
                        "parsed_answer": None if verdict.parsed_answer is None else str(verdict.parsed_answer),
                    }
                )

    def finish_pool(self) -> Path:
        """Publish the current pool's records and advance to the next pool."""
        with self.lock:
            records, self.records = self.records, []
            self.groups = 0
        if not records:
            raise ValueError(f"pool {self.pool} recorded no trajectories")
        if self.manifest["complete"]:
            raise ValueError(f"the dataset already holds its {self.pool} planned pools")
        path = self.directory / pool_filename(self.pool)
        table = pa.Table.from_pylist(records, schema=POOL_SCHEMA)
        _atomic_write(path, lambda staging: pq.write_table(table, staging))
        self.manifest["pools"].append(
            {
                "pool": self.pool,
                "file": path.name,
                "sha256": _sha256(path),
                "trajectories": len(records),
                "groups": len({record["group"] for record in records}),
                "reward_mean": sum(record["reward"] for record in records) / len(records),
            }
        )
        self.pool += 1
        self.manifest["complete"] = self.pool == self.manifest["planned_pools"]
        self._save_manifest()
        return path


@dataclass(frozen=True)
class RolloutGroup:
    """One prompt's trajectories from one pool."""

    pool: int
    group: int
    source: str
    prompt_index: str | None
    prompt_text: str
    prompt_sha256: str
    ground_truth: str
    answer_style: str
    module: str | None
    prompt_tokens: tuple[int, ...]
    completions: tuple[tuple[int, ...], ...]
    rewards: tuple[float, ...]
    structural_format_ok: tuple[bool, ...]
    terminated: tuple[bool, ...]

    @property
    def samples(self) -> int:
        return len(self.completions)


def read_manifest(directory: Path, allow_incomplete: bool = False) -> dict:
    """A dataset's manifest; incomplete datasets only on request."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != ROLLOUT_DATASET_SCHEMA:
        raise ValueError(f"{directory} is not a {ROLLOUT_DATASET_SCHEMA} dataset")
    pools = [entry["pool"] for entry in manifest["pools"]]
    if pools != list(range(len(pools))):
        raise ValueError(f"{directory} lists non-contiguous pools {pools}")
    if manifest["complete"] != (len(pools) == manifest["planned_pools"]):
        raise ValueError(f"{directory} has an inconsistent completion marker")
    if not manifest["complete"] and not allow_incomplete:
        raise ValueError(
            f"{directory} holds {len(pools)} of {manifest['planned_pools']} planned pools; "
            "its collection did not finish"
        )
    return manifest


def dataset_digest(manifest: dict) -> str:
    """One identity for a dataset: its binding and every pool digest."""
    payload = json.dumps(
        {"binding": manifest["binding"], "pools": [entry["sha256"] for entry in manifest["pools"]]},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_pool(directory: Path, entry: dict, verify: bool = True) -> list[RolloutGroup]:
    """One pool's prompt groups, in scoring order."""
    path = directory / entry["file"]
    if verify and _sha256(path) != entry["sha256"]:
        raise ValueError(f"{path} does not match its manifest digest")
    rows = pq.read_table(path, schema=POOL_SCHEMA).to_pylist()
    if len(rows) != entry["trajectories"]:
        raise ValueError(f"{path} holds {len(rows)} trajectories, manifest says {entry['trajectories']}")
    by_group: dict[int, list[dict]] = {}
    for row in rows:
        if row["pool"] != entry["pool"]:
            raise ValueError(f"{path} holds a row of pool {row['pool']}")
        by_group.setdefault(row["group"], []).append(row)
    groups = []
    for group_id in sorted(by_group):
        members = sorted(by_group[group_id], key=lambda row: row["sample"])
        if [row["sample"] for row in members] != list(range(len(members))):
            raise ValueError(f"{path} group {group_id} has missing or repeated samples")
        first = members[0]
        shared = ("source", "prompt_index", "prompt_text", "prompt_sha256", "ground_truth", "answer_style", "module", "prompt_tokens")
        if any(row[key] != first[key] for row in members for key in shared):
            raise ValueError(f"{path} group {group_id} mixes prompts")
        groups.append(
            RolloutGroup(
                pool=entry["pool"],
                group=group_id,
                source=first["source"],
                prompt_index=first["prompt_index"],
                prompt_text=first["prompt_text"],
                prompt_sha256=first["prompt_sha256"],
                ground_truth=first["ground_truth"],
                answer_style=first["answer_style"],
                module=first["module"],
                prompt_tokens=tuple(first["prompt_tokens"]),
                completions=tuple(tuple(row["completion_tokens"]) for row in members),
                rewards=tuple(float(row["reward"]) for row in members),
                structural_format_ok=tuple(bool(row["structural_format_ok"]) for row in members),
                terminated=tuple(bool(row["terminated"]) for row in members),
            )
        )
    return groups


def iter_pools(directory: Path, verify: bool = True) -> Iterator[list[RolloutGroup]]:
    manifest = read_manifest(directory)
    for entry in manifest["pools"]:
        yield read_pool(directory, entry, verify=verify)


def group_batch(
    prompt_tokens: Sequence[int],
    completions: Sequence[Sequence[int]],
    rewards: Sequence[float],
    trim_multiple: int = 1,
) -> LatentRolloutBatch:
    """The pinned-EMIT rollout batch of one prompt group, rebuilt from ids.

    Stream layout follows ``rollout_continuations``: the prompt fills slots
    ``[0, P)``, the action at slot ``P - 1 + i`` emits completion token ``i``
    into slot ``P + i``, and the terminal reward sits at the last action slot
    (``assign_terminal_rewards``). The stream is trimmed to its used length,
    rounded up to ``trim_multiple`` as ``trim_stream`` does. Behavior
    statistics stay zero: a dataset carries no old policy or values.
    """
    prompt_length = len(prompt_tokens)
    if prompt_length < 1:
        raise ValueError("a rollout group needs a nonempty prompt")
    if not completions or any(len(tokens) < 1 for tokens in completions):
        raise ValueError("every trajectory emits at least one token")
    if len(rewards) != len(completions):
        raise ValueError("one reward per trajectory is required")
    rows = len(completions)
    used = prompt_length + max(len(tokens) for tokens in completions)
    stream = -(-used // trim_multiple) * trim_multiple
    kind = torch.full((rows, stream), PAD_SLOT, dtype=torch.long)
    token_ids = torch.zeros((rows, stream), dtype=torch.long)
    action_mask = torch.zeros((rows, stream), dtype=torch.float32)
    kind[:, :prompt_length] = TOKEN_SLOT
    token_ids[:, :prompt_length] = torch.tensor(prompt_tokens, dtype=torch.long)
    for row, tokens in enumerate(completions):
        end = prompt_length + len(tokens)
        kind[row, prompt_length:end] = TOKEN_SLOT
        token_ids[row, prompt_length:end] = torch.tensor(tokens, dtype=torch.long)
        action_mask[row, prompt_length - 1 : end - 1] = 1.0
    actions = (action_mask * EMIT).long()
    zero = torch.zeros_like(action_mask)
    empty = torch.zeros((rows, stream, 0), dtype=torch.float32)
    batch = LatentRolloutBatch(
        kind=kind,
        token_ids=token_ids,
        thoughts=empty,
        hiddens=empty.clone(),
        actions=actions,
        action_mask=action_mask,
        stop_mask=zero.clone(),
        emit_mask=action_mask.clone(),
        old_stop_logprobs=zero.clone(),
        old_token_logprobs=zero.clone(),
        old_token_log_odds=zero.clone(),
        old_thought_logprobs=empty.clone(),
        old_thought_means=empty.clone(),
        old_thought_log_sigmas=empty.clone(),
        old_values=zero.clone(),
        old_action_values=zero.clone(),
        rewards=zero.clone(),
        reward_scalar=torch.zeros(rows, dtype=torch.float32),
        prompt_length=prompt_length,
    )
    assign_terminal_rewards(batch, torch.tensor(rewards, dtype=torch.float32))
    return batch


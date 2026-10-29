"""Teacher-forced memory probe: does dense supervision adopt the memory carry?

Under RL the gated-delta memory carry (``postraining/memory_carry.py``,
reasoning mode ``memory``) stayed nearly unused: its read reached about 2%
of the input RMS under the shared fine-tune AdamW and about 9% at
``--fresh-learning-rate 3e-3``, while bench went flat (NOTES 2026-10-21,
"Memory as a fresh module"). Three explanations compete:

- (A) signal: one reward bit per episode at gradient SNR ~0.02 cannot grow
  a new read circuit. Dense next-token CE should adopt it fast.
- (B) redundancy: the final belief ``h_{t-1}`` mostly encodes the token
  ``x_t`` the next slot already embeds, and its whole prefix is visible to
  slot ``t`` through attention/KDA. Even dense CE gains little; writing a
  mid-layer state may do better.
- (C) rate: Muon moves ``W_o`` a fixed amount per step, so adoption speed
  tracks the fresh learning rate.

This probe trains the memory on gold worked drill traces with next-token CE
on completion tokens, from the drill RL start checkpoint, and reports
held-out CE overall and on the error anatomy's high-hazard step types.

**Model.** ``load_probe_actor`` builds the actor exactly as
``train_latent_vapo`` does for ``--reasoning-mode memory``: the checkpoint's
backbone under ``LatentThoughtModel(hidden_carry=True, memory_carry=True)``
at the trainer's default geometry. Stream inputs are built with the
trainer's replay calls (``memory_write_mask`` and
``GatedDeltaMemory.stream_inputs``, as ``assemble_stream_latents`` does):
every real slot after the first writes the belief computed one slot
earlier, prompt slots never read, and completion slots read.

**Beliefs (the Jacobian-free fixed point).** In RL the written beliefs are
the decode's own record: the exact causal fixed point of
belief -> write -> read -> belief. Teacher forcing has no decode, so the
written stream is iterated with the current weights and no gradient:

    c_0 = source(pass with the memory off)
    c_k = source(pass writing c_{k-1}),  k = 1 .. K-1
    training pass writes c_{K-1} (detached, as replay detaches its record)

``source`` is the post-final-norm belief (the trainer's write) or, for the
mid-layer arm, the residual stream after block ``write_layer``. Slot ``t``'s
read sees writes of slots ``< t``, so ``c_k`` is exact through completion
position ``P + k - 1`` and the training pass's beliefs are exact through
``P + K - 1``; beyond that they carry the iteration's residual. Every pass
logs ``|c_k - c_{k-1}| / |c_{k-1}|`` over read slots, and evaluation also
logs the training pass's own residual ``|c_K - c_{K-1}|`` and the CE change
from two extra passes, so the approximation's error is measured, not
assumed. Like RL, no gradient flows through the written beliefs: the trunk
is never trained to put anything into them.

**Arms** (same data order, same memory initialization, same steps):
``off`` (memory never read; the baseline), ``mem`` at the fresh rate L, at
L/4 and 4L (C), ``mem_midlayer`` (B), ``mem_shuffled`` (the capacity
control: every row's carried stream is another row's, tiled to length, so
the read keeps its parameters and statistics but carries nothing about this
trajectory), ``trunk_only`` and ``mem_plus_trunk`` (co-adaptation, with the
trunk at the SFT's fine-tune rates). Fresh modules use the trainer's
partition: matrices under Muon at ``L * (0.025/0.015) * 1.45``, vectors
under AdamW at L.

Subcommands: ``build-data`` writes the versioned training traces, ``train``
runs arms, ``analyze`` compares finished arms with paired per-problem SEs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postraining.core import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    GPT2BPETokenizer,
    encode_many,
    encode_prompt,
    frame_prompt,
)
from postraining.latent_rollout import PAD_SLOT, TOKEN_SLOT
from postraining.latent_thought import LatentThoughtModel
from postraining.math_drills import BUILDERS, FAMILY_ORDERS, Drill, FamilySpec, generate
from postraining.math_prompt import (
    ANSWER_FENCE_SUFFIX,
    answer_fence_prompt,
    require_answer_fence_prompt_schema,
)
from postraining.memory_carry import (
    DEFAULT_MEMORY_HEAD_DIM,
    DEFAULT_MEMORY_HEADS,
    GatedDeltaMemory,
    delta_scan,
    masked_write_terms,
    memory_write_mask,
)
from postraining.muon import Muon
from postraining.problem_registry import problem_key
from postraining.vapo.config import POLAR_EXPRESS_STEP_COMPENSATION
from scripts.analyze_drill_step_errors import (
    NUMBER,
    RAW_DRAWS_PER_CELL,
    cell_seed,
    step_lines,
)

PROBE_SCHEMA = "memory_ce_probe/v1"
PROBE_DATA_SCHEMA = "memory_ce_probe_drills/v1"
# The RL study pools draw each cell from cell_seed(pool seed, family, digits)
# with pool seeds 20261015 (train) and 20261016 (held-out); the probe's
# training traces use a third, disjoint pool seed.
PROBE_TRAIN_POOL_SEED = 20261021
STUDY_POOL_SEEDS = (20261015, 20261016)
RL_START_CHECKPOINT = (
    "postraining/runs/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt"
)
# The trainer's fresh-module Muon rate rule (vapo/config.py).
MUON_RATE_FACTOR = (0.025 / 0.015) * POLAR_EXPRESS_STEP_COMPENSATION
_AUTOCAST_DTYPE = torch.bfloat16

# Token classes scored at evaluation, all over target (predicted) tokens.
TOKEN_CLASSES = (
    "all",  # every completion token plus the stop target
    "comp_line",  # tokens on computation lines (step_lines' definition)
    "prose",  # solution tokens on any other line
    "answer",  # the <answer> span's value tokens
    "borrow_equal",  # column-subtraction decision token, equal digits
    "borrow_equal_after",  # ... equal digits after a borrow
    "borrow_real",  # ... top < bottom (a real borrow)
    "borrow_none",  # ... top > bottom
    "pp_readout",  # partial-product readout number (digit-list reversal)
    "pp_shift",  # the shifted partial product (appended zeros)
    "pp_add_leftmost",  # operand digits of the leftmost column in pp addition
    "pp_add_operand",  # every operand digit read in pp addition
    "flat",  # the union of FLAT_CLASSES: the pre-registered primary class
)
DECISION_CLASSES = ("borrow_equal", "borrow_equal_after", "borrow_real", "borrow_none")
# The step types whose errors drill RL left flat (NOTES 2026-10-20/21):
# equal-digit and phantom borrow decisions, partial-product readouts and
# leftmost-column reads. Real borrows are excluded -- RL's trunk fixed them.
FLAT_CLASSES = ("borrow_equal", "borrow_equal_after", "borrow_none", "pp_readout", "pp_add_leftmost")

_COLUMN_SUBTRACTION = re.compile(
    r"^[a-z0-9^][a-z0-9^ ]*: (-?\d+)( after the borrow)?( is less than| -) (\d+)"
)
_PP_READOUT = re.compile(
    r"^\s*Digits from the ones place: [\d ]+ -> (\d+)"
    r"(?:, shifted \d+ places? -> (\d+))?$"
)
_PP_ADD_HEADER = re.compile(r"^Add the next partial product: (\d+) \+ (\d+)$")
_PP_ADD_COLUMN = re.compile(r"^\s+[a-z0-9^][a-z0-9^ ]*: (\d) \+ (\d)")


# --------------------------------------------------------------------- data


def drill_document(problem: str, solution: str, answer: str) -> str:
    """The canonical SFT document (``prepare_sft_corpus.build_document``)."""
    return (
        f"{problem}{ANSWER_FENCE_SUFFIX}{THINK_OPEN}\n{solution}\n"
        f"{THINK_CLOSE}\n{ANSWER_OPEN}{answer}{ANSWER_CLOSE}"
    )


def study_cells(pool: Path) -> list[tuple[str, int]]:
    """The (family, digits) cells an RL study pool lists, sorted."""
    infos = pq.read_table(pool, columns=["extra_info"]).column("extra_info").to_pylist()
    return sorted({(info["family"], int(info["digits"])) for info in infos})


def rebuild_pool_drills(pool: Path) -> list[Drill]:
    """Every pool problem's worked drill under its recorded digit order.

    The pool builder drew each cell as ``math_drills.generate`` on
    ``cell_seed(pool seed, family, digits)``, skipping excluded draws, so a
    pool problem is matched by text in the raw stream (the rule
    ``scripts/analyze_drill_step_errors.py`` uses). The rebuilt drill must
    reproduce the recorded digit order and ground truth.
    """
    rows = pq.read_table(pool).to_pylist()
    wanted: dict[tuple[str, int, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        info = row["extra_info"]
        seed = int(info["index"].split("/")[3])
        wanted[(info["family"], int(info["digits"]), seed)][row["prompt"][0]["content"]] = row
    rebuilt: dict[str, Drill] = {}
    for (family, digits, seed), problems in wanted.items():
        rng = random.Random(cell_seed(seed, family, digits))
        missing = set(problems)
        for _ in range(RAW_DRAWS_PER_CELL):
            rng.choices([family], [1.0])
            order = rng.choice(FAMILY_ORDERS[family])
            rng.randint(digits, digits)
            drill = BUILDERS[family](rng, order, digits)
            if drill.problem not in missing:
                continue
            missing.discard(drill.problem)
            row = problems[drill.problem]
            if drill.digit_order != row["extra_info"]["digit_order"]:
                raise ValueError(f"digit order mismatch for {drill.problem!r}")
            if drill.answer != row["reward_model"]["ground_truth"]:
                raise ValueError(f"ground truth mismatch for {drill.problem!r}")
            rebuilt[drill.problem] = drill
            if not missing:
                break
        if missing:
            raise ValueError(f"{len(missing)} {family}:{digits} problems not rebuilt")
    return [rebuilt[row["prompt"][0]["content"]] for row in rows]


def module_name(drill: Drill) -> str:
    return f"{drill.family}_d{int(drill.difficulty['digits'])}"


def generate_cell_drills(
    family: str,
    digits: int,
    *,
    pool_seed: int,
    count: int,
    excluded_keys: frozenset[bytes],
) -> list[Drill]:
    """Up to ``count`` distinct drills of one cell, the pool builder's draw rule.

    ``generate`` raises once a cell's problem space is exhausted; a small
    cell simply yields fewer drills.
    """
    drills = []
    stream = generate(
        seed=cell_seed(pool_seed, family, digits),
        count=count,
        curriculum=(FamilySpec(family, 1.0, digits, digits),),
        excluded_keys=excluded_keys,
    )
    try:
        drills.extend(stream)
    except RuntimeError:
        pass
    return drills


def drill_record(drill: Drill) -> dict:
    return {
        "module": module_name(drill),
        "family": drill.family,
        "digits": int(drill.difficulty["digits"]),
        "digit_order": drill.digit_order,
        "problem": drill.problem,
        "solution": drill.solution,
        "answer": drill.answer,
        "problem_key": drill.key.hex(),
        "document": drill_document(drill.problem, drill.solution, drill.answer),
    }


def build_training_table(
    train_pool: Path, heldout_pool: Path, *, pool_seed: int, per_cell: int
) -> tuple[pa.Table, dict]:
    """Fresh gold traces for the study's cells, held-out problems excluded."""
    if pool_seed in STUDY_POOL_SEEDS:
        raise ValueError("the probe's pool seed must differ from the study pools'")
    heldout_rows = pq.read_table(heldout_pool).to_pylist()
    heldout_texts = {row["prompt"][0]["content"] for row in heldout_rows}
    excluded = frozenset(problem_key(text) for text in heldout_texts)
    train_texts = {
        row["prompt"][0]["content"] for row in pq.read_table(train_pool).to_pylist()
    }
    cells = study_cells(train_pool)
    if cells != study_cells(heldout_pool):
        raise ValueError("the study's train and held-out pools list different cells")
    records = []
    per_cell_counts = {}
    for family, digits in cells:
        drills = generate_cell_drills(
            family, digits, pool_seed=pool_seed, count=per_cell, excluded_keys=excluded
        )
        per_cell_counts[f"{family}_d{digits}"] = len(drills)
        for drill in drills:
            if drill.problem in heldout_texts:
                raise AssertionError("a held-out problem survived the key exclusion")
            records.append(drill_record(drill))
    table = pa.Table.from_pylist(records)
    manifest = {
        "schema": PROBE_DATA_SCHEMA,
        "pool_seed": pool_seed,
        "per_cell_requested": per_cell,
        "per_cell": per_cell_counts,
        "rows": len(records),
        "train_pool": str(train_pool),
        "train_pool_sha256": file_sha256(train_pool),
        "heldout_pool": str(heldout_pool),
        "heldout_pool_sha256": file_sha256(heldout_pool),
        "heldout_problems_excluded": len(heldout_texts),
        "overlap_with_rl_train_pool": sum(r["problem"] in train_texts for r in records),
        "cell_seed_rule": "blake2b(f'{seed}:{family}:{digits}', 8 bytes, little)",
    }
    return table, manifest


# The column-method cells whose step types the error anatomy flagged: the
# borrow decision (subtraction) and the partial-product readout and addition
# chain (multiplication).
HAZARD_FAMILIES = ("sub_integer", "sub_decimal", "mul_integer", "mul_decimal")
PROBE_HAZARD_PANEL_SEED = 20261022
PROBE_PANEL_SCHEMA = "memory_ce_probe_hazard_panel/v1"


def build_hazard_panel(
    train_data: Path,
    train_pool: Path,
    heldout_pool: Path,
    *,
    seed: int,
    per_cell: int,
    family_per_cell: dict[str, int] | None = None,
) -> tuple[pa.Table, dict]:
    """A column-method evaluation panel for the hazard step types.

    The held-out pool holds few hazard tokens (about 50 equal-digit borrow
    decisions), because half its subtraction problems use the expanded
    method and equal digits are rare. This panel draws the study's
    subtraction and multiplication cells under the column method only
    (``digit_order = reversed``), from its own seed, excluding every problem
    in the probe's training traces, the RL train pool and the held-out pool.
    ``family_per_cell`` overrides ``per_cell`` for named families (v2 draws
    more subtraction rows, where equal-digit decisions are rare).
    """
    family_per_cell = dict(family_per_cell or {})
    unknown = set(family_per_cell) - set(HAZARD_FAMILIES)
    if unknown:
        raise ValueError(f"unknown hazard families {sorted(unknown)}")
    if seed in STUDY_POOL_SEEDS or seed == PROBE_TRAIN_POOL_SEED:
        raise ValueError("the panel seed must differ from every pool seed")
    excluded_texts = set(pq.read_table(train_data, columns=["problem"]).column("problem").to_pylist())
    for pool in (train_pool, heldout_pool):
        excluded_texts |= {row["prompt"][0]["content"] for row in pq.read_table(pool).to_pylist()}
    excluded = {problem_key(text) for text in excluded_texts}
    cells = [cell for cell in study_cells(heldout_pool) if cell[0] in HAZARD_FAMILIES]
    records, per_cell_counts = [], {}
    for family, digits in cells:
        wanted = family_per_cell.get(family, per_cell)
        rng = random.Random(cell_seed(seed, family, digits))
        seen: set[bytes] = set()
        drills = []
        for _ in range(64 * wanted):
            drill = BUILDERS[family](rng, "reversed", digits)
            if drill.key in seen or drill.key in excluded:
                continue
            seen.add(drill.key)
            drills.append(drill)
            if len(drills) == wanted:
                break
        per_cell_counts[f"{family}_d{digits}"] = len(drills)
        records.extend(drill_record(drill) for drill in drills)
    manifest = {
        "schema": PROBE_PANEL_SCHEMA,
        "seed": seed,
        "per_cell_requested": per_cell,
        "family_per_cell_requested": family_per_cell,
        "per_cell": per_cell_counts,
        "rows": len(records),
        "digit_order": "reversed",
        "excluded_problems": len(excluded),
        "train_data_sha256": file_sha256(train_data),
        "train_pool_sha256": file_sha256(train_pool),
        "heldout_pool_sha256": file_sha256(heldout_pool),
    }
    return pa.Table.from_pylist(records), manifest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------- tokenization and classes


def is_step_line(line: str) -> bool:
    """``step_lines``' computation-line predicate for one line."""
    stripped = line.strip()
    return bool(NUMBER.search(line)) and (
        "=" in line or "->" in line or stripped.startswith("Digits from")
    )


def annotate_solution(
    solution: str, family: str, digit_order: str
) -> dict[str, list[tuple[int, int, bool]]]:
    """Character spans of each token class in ``solution``.

    A span is ``(start, end, first_token_only)``. Decision spans mark the
    text the decision emits (" is less than" or " -" right after the top
    digit) and select only its first token.
    """
    spans: dict[str, list[tuple[int, int, bool]]] = defaultdict(list)
    column_subtraction = family in ("sub_integer", "sub_decimal") and digit_order == "reversed"
    offset = 0
    pp_block: dict | None = None
    for line in solution.split("\n"):
        start, end = offset, offset + len(line)
        offset = end + 1
        if line.strip():
            spans["comp_line" if is_step_line(line) else "prose"].append((start, end, False))
        if column_subtraction and (match := _COLUMN_SUBTRACTION.match(line)):
            top, borrowed, _, bottom = match.groups()
            top_value, bottom_value = int(top), int(bottom)
            if top_value < bottom_value:
                name = "borrow_real"
            elif top_value > bottom_value:
                name = "borrow_none"
            else:
                name = "borrow_equal_after" if borrowed else "borrow_equal"
            spans[name].append((start + match.start(3), start + match.end(3), True))
        if match := _PP_READOUT.match(line):
            spans["pp_readout"].append((start + match.start(1), start + match.end(1), False))
            if match.group(2) is not None:
                spans["pp_shift"].append((start + match.start(2), start + match.end(2), False))
        if match := _PP_ADD_HEADER.match(line):
            width = max(len(match.group(1)), len(match.group(2)))
            pp_block = {"width": width, "column": 0}
            continue
        if pp_block is not None:
            if column := _PP_ADD_COLUMN.match(line):
                operands = [
                    (start + column.start(group), start + column.end(group), False)
                    for group in (1, 2)
                ]
                spans["pp_add_operand"].extend(operands)
                if pp_block["column"] == pp_block["width"] - 1:
                    spans["pp_add_leftmost"].extend(operands)
                pp_block["column"] += 1
            elif "Digits from the ones place" in line or not line.startswith(" "):
                pp_block = None
    return dict(spans)


@dataclass(frozen=True)
class EncodedTrace:
    """One gold episode framed exactly as RL replays it."""

    module: str
    problem: str
    ids: np.ndarray  # int32, [BOS] + prompt then completion (no stop token)
    prompt_length: int
    # (completion tokens + 1,) bool per class over the supervised targets:
    # target i is completion token i, and the last is the stop target.
    classes: dict[str, np.ndarray]

    @property
    def length(self) -> int:
        return int(self.ids.shape[0])

    @property
    def supervised(self) -> int:
        return self.length - self.prompt_length + 1


def completion_token_spans(tokenizer: GPT2BPETokenizer, completion: str):
    """Completion ids and their character offsets.

    The offsets come from the same fast tokenizer ``tokenizer.encode`` wraps;
    the ids are asserted equal to ``encode``'s.
    """
    encoded = tokenizer._tokenizer(completion, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    if ids != tokenizer.encode(completion):
        raise AssertionError("offset encoding disagrees with tokenizer.encode")
    return ids, list(encoded["offset_mapping"])


def encode_trace(
    tokenizer: GPT2BPETokenizer,
    *,
    module: str,
    family: str,
    digit_order: str,
    problem: str,
    solution: str,
    answer: str,
    annotate: bool,
) -> EncodedTrace:
    """Tokenize one gold trace the way ``sft_trace_train`` frames RL episodes.

    The prompt is the bare problem (``answer_fence_prompt``), encoded alone
    and BOS-framed; the completion is the rest of the canonical document,
    encoded alone.
    """
    prompt = answer_fence_prompt(problem) + ANSWER_FENCE_SUFFIX
    if prompt != problem:
        raise ValueError(f"drill problem is not canonical: {problem!r}")
    document = drill_document(problem, solution, answer)
    completion = document[len(prompt):]
    prompt_ids = frame_prompt(tokenizer, tokenizer.encode(prompt))
    supervised = None
    if annotate:
        completion_ids, offsets = completion_token_spans(tokenizer, completion)
        supervised = len(completion_ids) + 1
        solution_offset = len(f"{THINK_OPEN}\n")
        if completion[solution_offset:solution_offset + len(solution)] != solution:
            raise AssertionError("completion does not hold the solution verbatim")
        classes = {name: np.zeros(supervised, dtype=bool) for name in TOKEN_CLASSES}
        classes["all"][:] = True
        spans = annotate_solution(solution, family, digit_order)
        answer_start = completion.rindex(ANSWER_OPEN) + len(ANSWER_OPEN)
        spans["answer"] = [(answer_start - solution_offset,
                            answer_start - solution_offset + len(answer), False)]
        token_starts = np.array([s for s, _ in offsets])
        token_ends = np.array([e for _, e in offsets])
        for name, class_spans in spans.items():
            for span_start, span_end, first_only in class_spans:
                lo, hi = span_start + solution_offset, span_end + solution_offset
                hit = np.flatnonzero((token_starts < hi) & (token_ends > lo))
                if hit.size == 0:
                    raise AssertionError(f"{name} span maps to no token")
                classes[name][hit[:1] if first_only else hit] = True
        classes["flat"] = np.logical_or.reduce([classes[name] for name in FLAT_CLASSES])
    else:
        completion_ids = tokenizer.encode(completion)
        classes = {}
    if not completion_ids or completion_ids[0] != tokenizer.think_open_id:
        raise ValueError("the completion must open with the registered <think> token")
    if completion_ids[-1] != tokenizer.answer_close_id:
        raise ValueError("the completion must close with the registered </answer> token")
    ids = np.asarray(prompt_ids + completion_ids, dtype=np.int32)
    return EncodedTrace(module, problem, ids, len(prompt_ids), classes)


def encode_table(
    tokenizer: GPT2BPETokenizer, rows: Sequence[dict], *, annotate: bool
) -> list[EncodedTrace]:
    if not annotate:
        # The batched encoder gives encode()'s ids (``encode_many``); the
        # framing and fence checks are encode_trace's.
        prompts = [answer_fence_prompt(row["problem"]) + ANSWER_FENCE_SUFFIX for row in rows]
        documents = [drill_document(r["problem"], r["solution"], r["answer"]) for r in rows]
        completions = [doc[len(prompt):] for doc, prompt in zip(documents, prompts)]
        traces = []
        for row, prompt, prompt_ids, completion_ids in zip(
            rows, prompts, encode_many(tokenizer, prompts), encode_many(tokenizer, completions),
            strict=True,
        ):
            if prompt != row["problem"]:
                raise ValueError(f"drill problem is not canonical: {row['problem']!r}")
            if completion_ids[0] != tokenizer.think_open_id or completion_ids[-1] != tokenizer.answer_close_id:
                raise ValueError("a completion is not fenced by the registered tokens")
            framed = frame_prompt(tokenizer, prompt_ids)
            ids = np.asarray(framed + list(completion_ids), dtype=np.int32)
            traces.append(EncodedTrace(row["module"], row["problem"], ids, len(framed), {}))
        return traces
    return [
        encode_trace(
            tokenizer,
            module=row["module"],
            family=row["family"],
            digit_order=row["digit_order"],
            problem=row["problem"],
            solution=row["solution"],
            answer=row["answer"],
            annotate=annotate,
        )
        for row in rows
    ]


def drill_rows(drills: Sequence[Drill]) -> list[dict]:
    return [
        {
            "module": module_name(drill),
            "family": drill.family,
            "digit_order": drill.digit_order,
            "problem": drill.problem,
            "solution": drill.solution,
            "answer": drill.answer,
        }
        for drill in drills
    ]


# ------------------------------------------------------------------ batches


@dataclass
class StreamBatch:
    """Right-padded teacher-forced episodes on the device."""

    ids: Tensor  # (B, T) long
    real: Tensor  # (B, T) bool
    read: Tensor  # (B, T) bool: generated (completion) slots
    lengths: Tensor  # (B,) long, real slots per row
    positions: Tensor  # (N,) flat indices of supervised slots
    labels: Tensor  # (N,) next-token targets
    row_of: np.ndarray  # (N,) row index of each supervised slot (host)
    classes: dict[str, Tensor]  # name -> (N,) bool
    traces: list[EncodedTrace]

    @property
    def kind(self) -> Tensor:
        return torch.where(self.real, TOKEN_SLOT, PAD_SLOT)


def collate(
    traces: Sequence[EncodedTrace], device: torch.device, *, stop_id: int, pad_multiple: int = 64
) -> StreamBatch:
    longest = max(trace.length for trace in traces)
    width = -(-longest // pad_multiple) * pad_multiple
    ids = np.full((len(traces), width), stop_id, dtype=np.int64)
    real = np.zeros((len(traces), width), dtype=bool)
    read = np.zeros((len(traces), width), dtype=bool)
    positions, labels, rows = [], [], []
    for row, trace in enumerate(traces):
        length, prompt = trace.length, trace.prompt_length
        ids[row, :length] = trace.ids
        real[row, :length] = True
        read[row, prompt:length] = True
        slots = np.arange(prompt - 1, length)
        positions.append(row * width + slots)
        targets = np.empty(slots.size, dtype=np.int64)
        targets[:-1] = trace.ids[prompt:length]
        targets[-1] = stop_id
        labels.append(targets)
        rows.append(np.full(slots.size, row))
    classes = {}
    if traces[0].classes:
        classes = {
            name: torch.from_numpy(np.concatenate([t.classes[name] for t in traces])).to(device)
            for name in TOKEN_CLASSES
        }
    upload = lambda array: torch.from_numpy(array).to(device)
    return StreamBatch(
        ids=upload(ids),
        real=upload(real),
        read=upload(read),
        lengths=torch.tensor([t.length for t in traces], device=device),
        positions=upload(np.concatenate(positions)),
        labels=upload(np.concatenate(labels)),
        row_of=np.concatenate(rows),
        classes=classes,
        traces=list(traces),
    )


class CellBalancedSampler:
    """Cell-uniform draws, each cell cycled through in shuffled epochs.

    ``macro`` batches are drawn together, sorted by length and split, so a
    padded batch holds traces of similar length; the macro group's batch
    order is then shuffled. Deterministic in ``seed``.
    """

    def __init__(self, traces: Sequence[EncodedTrace], batch: int, macro: int, seed: int):
        self.rng = random.Random(seed)
        self.batch, self.macro = batch, macro
        by_cell: dict[str, list[int]] = defaultdict(list)
        for index, trace in enumerate(traces):
            by_cell[trace.module].append(index)
        self.cells = sorted(by_cell)
        self.members = {cell: by_cell[cell] for cell in self.cells}
        self.queues: dict[str, list[int]] = {cell: [] for cell in self.cells}
        self.traces = traces
        self.pending: list[list[int]] = []

    def _draw(self) -> int:
        cell = self.rng.choice(self.cells)
        queue = self.queues[cell]
        if not queue:
            queue.extend(self.members[cell])
            self.rng.shuffle(queue)
        return queue.pop()

    def __iter__(self) -> Iterator[list[EncodedTrace]]:
        while True:
            if not self.pending:
                draws = sorted(
                    (self._draw() for _ in range(self.batch * self.macro)),
                    key=lambda index: self.traces[index].length,
                )
                self.pending = [
                    draws[i:i + self.batch] for i in range(0, len(draws), self.batch)
                ]
                self.rng.shuffle(self.pending)
            yield [self.traces[index] for index in self.pending.pop()]


def evaluation_batches(
    traces: Sequence[EncodedTrace], slot_budget: int, min_rows: int = 2
) -> list[list[EncodedTrace]]:
    """Length-sorted held-out batches of at most ``slot_budget`` padded slots.

    Every batch has at least ``min_rows`` rows, so the shuffle control always
    has a donor: a short tail batch joins the one before it (exceeding the
    budget by at most ``min_rows - 1`` rows).
    """
    ordered = sorted(traces, key=lambda trace: trace.length)
    if len(ordered) < min_rows:
        raise ValueError(f"an evaluation panel needs at least {min_rows} traces")
    if min_rows * (-(-ordered[-1].length // 64) * 64) > slot_budget:
        raise ValueError(f"--eval-slot-budget {slot_budget} cannot fit {min_rows} of the longest trace")
    batches, current = [], []
    for trace in ordered:
        width = -(-trace.length // 64) * 64
        if current and width * (len(current) + 1) > slot_budget:
            batches.append(current)
            current = []
        current.append(trace)
    if batches and len(current) < min_rows:
        batches[-1].extend(current)
    else:
        batches.append(current)
    return batches


# -------------------------------------------------------------------- model


# Eval-time ablations of a trained memory's read at slot t, which normally
# sees every write through slot t (the last one is the belief at t - 1):
#   drop_last_write  the state through slot t - 1 only (beliefs to t - 2),
#                    so the one-step final-layer carry is removed
#   last_write_only  only slot t's write, from an empty state: the one-step
#                    carry alone
#   foreign_writes   the shuffle's writes, with the row's own belief in the
#                    query, so only the written content is foreign
READ_ABLATIONS = ("drop_last_write", "last_write_only", "foreign_writes")


@dataclass(frozen=True)
class ProbeConfig:
    """How one forward builds its stream."""

    memory: bool
    write_layer: int | None = None  # None: the post-final-norm belief
    shuffle: bool = False
    passes: int = 2
    read_ablation: str | None = None

    def __post_init__(self):
        if self.read_ablation not in (None, *READ_ABLATIONS):
            raise ValueError(f"unknown read ablation {self.read_ablation!r}")
        if self.read_ablation is not None and (self.shuffle or not self.memory):
            raise ValueError("read ablations apply to an unshuffled memory arm")


@dataclass(frozen=True)
class Carried:
    """The streams one memory pass consumes.

    ``written`` enters the memory and ``query`` is the belief in the reader's
    query; ``own`` is the row's own stream, which the fixed point refines.
    The shuffle control writes and queries the donor's stream; the
    ``foreign_writes`` ablation writes the donor's and queries its own.
    """

    written: Tensor
    query: Tensor
    own: Tensor


@dataclass
class StreamOutput:
    token_latent: Tensor
    inputs: Tensor
    belief: Tensor
    source: Tensor  # the stream this pass would write next


def shift_carried(carried: Tensor, write_mask: Tensor) -> Tensor:
    """The replay record: ``hiddens[:, j]`` is the stream at slot ``j - 1``.

    Slots that write nothing hold zero, as a rollout's record does.
    """
    hiddens = torch.zeros_like(carried)
    hiddens[:, 1:] = carried[:, :-1]
    return hiddens * write_mask[..., None].to(hiddens.dtype)


def derangement(rows: int, generator: torch.Generator) -> Tensor:
    """A uniformly ordered cycle: row ``order[i]`` takes ``order[i + 1]``'s."""
    if rows < 2:
        raise ValueError("a shuffle control needs at least two rows")
    order = torch.randperm(rows, generator=generator)
    donor = torch.empty(rows, dtype=torch.long)
    donor[order] = order.roll(-1)
    return donor


def shuffle_carried(carried: Tensor, lengths: Tensor, donor: Tensor) -> Tensor:
    """Each row's carried stream replaced by its donor's, tiled to length.

    Recipient slot ``t`` gets donor slot ``t mod L_donor``: every written
    vector is a real slot of another trajectory, positions are kept where
    the donor is long enough, and nothing in the stream comes from this row.
    """
    donor = donor.to(carried.device)
    columns = torch.arange(carried.size(1), device=carried.device)
    source_columns = columns[None, :] % lengths[donor][:, None]
    return carried[donor[:, None], source_columns]


def memory_inputs(
    memory: GatedDeltaMemory,
    token_latent: Tensor,
    carried: Carried,
    write_mask: Tensor,
    read_mask: Tensor,
    ablation: str | None,
) -> Tensor:
    """``stream_inputs``' formula with separate write and query streams.

    Without an ablation and with one stream it is ``stream_inputs`` itself,
    the trainer's path; otherwise the same writer terms, queries and
    ``reader.add`` around an ablated read.
    """
    hiddens = shift_carried(carried.written, write_mask)
    if ablation is None and carried.query is carried.written:
        return memory.stream_inputs(token_latent, hiddens, write_mask, read_mask)
    k, v, log_decay, beta = masked_write_terms(memory.writer, hiddens.detach(), write_mask)
    q = memory.reader.queries(token_latent, shift_carried(carried.query, write_mask).detach())
    if ablation == "last_write_only":
        # S_t = beta_t k_t v_t^T from an empty state, so S_t^T q_t is this.
        reads = beta[..., None] * (q * k).sum(-1, keepdim=True) * v
    elif ablation == "drop_last_write":
        # Slot t's read of S_{t-1}: query t scanned at slot t - 1.
        shifted = torch.cat((q[:, 1:], torch.zeros_like(q[:, :1])), dim=1)
        scanned = delta_scan(shifted, k, v, log_decay, beta)[0]
        reads = torch.cat((torch.zeros_like(scanned[:, :1]), scanned[:, :-1]), dim=1)
    else:
        reads = delta_scan(q, k, v, log_decay, beta)[0]
    return torch.where(read_mask[..., None], memory.reader.add(token_latent, reads), token_latent)


def run_stream(
    wrapper: LatentThoughtModel,
    batch: StreamBatch,
    carried: Carried | None,
    write_layer: int | None,
    ablation: str | None = None,
) -> StreamOutput:
    """One teacher-forced pass, writing ``carried`` (or reading nothing).

    The input path is ``assemble_stream_latents``' for a memory batch whose
    record is ``shift_carried(carried.written)``; the trunk loop is
    ``temporal_belief_from_token_latent``'s, with the residual stream after
    block ``write_layer`` kept for the mid-layer write.
    """
    backbone = wrapper.backbone
    token_latent = wrapper.embed_tokens(batch.ids)
    pad_scale = batch.real[..., None].to(token_latent.dtype)
    if carried is None:
        inputs = token_latent * pad_scale
    else:
        write_mask = memory_write_mask(batch.kind, PAD_SLOT)
        inputs = memory_inputs(
            wrapper.combiner, token_latent, carried, write_mask, batch.read, ablation
        ) * pad_scale
    x, source = inputs, None
    for index, block in enumerate(backbone.blocks):
        x = block(x)
        if write_layer is not None and index + 1 == write_layer:
            source = x
    belief = backbone.final_norm(x)
    return StreamOutput(token_latent, inputs, belief, belief if source is None else source)


def relative_movement(new: Tensor, old: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """(sum |new - old|^2, sum |old|^2) over ``mask`` slots, in fp32."""
    weight = mask[..., None].float()
    return (
        ((new.float() - old.float()).square() * weight).sum(),
        (old.float().square() * weight).sum(),
    )


def fixed_point_carried(
    wrapper: LatentThoughtModel,
    batch: StreamBatch,
    config: ProbeConfig,
    donor: Tensor | None,
) -> tuple[Carried, list[tuple[Tensor, Tensor]]]:
    """``c_{K-1}``, the stream the training pass writes, and each pass's movement.

    Movement ``k`` compares ``c_k`` with ``c_{k-1}`` over read slots (prompt
    slots never read, so they never move). Shuffled arms write the donor's
    stream at every pass; an ablated read is ablated at every pass.
    """
    if config.passes < 1:
        raise ValueError("the fixed point needs at least one pass")
    foreign = config.shuffle or config.read_ablation == "foreign_writes"
    if foreign and donor is None:
        raise ValueError("a shuffled stream needs a donor")

    def arrange(own: Tensor) -> Carried:
        if not foreign:
            return Carried(own, own, own)
        donated = shuffle_carried(own, batch.lengths, donor)
        return Carried(donated, donated if config.shuffle else own, own)

    with torch.no_grad():
        own = run_stream(wrapper, batch, None, config.write_layer).source
        movements = []
        for _ in range(config.passes - 1):
            refined = run_stream(
                wrapper, batch, arrange(own), config.write_layer, config.read_ablation
            ).source
            movements.append(relative_movement(refined, own, batch.read))
            own = refined
    return arrange(own), movements


def probe_forward(
    wrapper: LatentThoughtModel,
    batch: StreamBatch,
    config: ProbeConfig,
    donor: Tensor | None = None,
) -> tuple[StreamOutput, Carried | None, list[tuple[Tensor, Tensor]]]:
    """The training pass: (output, consumed streams, fixed-point movements)."""
    if not config.memory:
        return run_stream(wrapper, batch, None, None), None, []
    carried, movements = fixed_point_carried(wrapper, batch, config, donor)
    out = run_stream(wrapper, batch, carried, config.write_layer, config.read_ablation)
    return out, carried, movements


def supervised_features(wrapper: LatentThoughtModel, out: StreamOutput, batch: StreamBatch) -> Tensor:
    features = wrapper.renderer_features(out.inputs, out.belief)
    return features.flatten(0, 1).index_select(0, batch.positions)


def target_logprobs(
    wrapper: LatentThoughtModel, out: StreamOutput, batch: StreamBatch, chunk: int | None = None
) -> Tensor:
    """log P(target) at every supervised slot, the SFT readout.

    ``chunk`` bounds the vocabulary-wide logits held at once (no-grad
    evaluation; a training pass keeps every chunk's logits for backward).
    """
    features = supervised_features(wrapper, out, batch)
    readout = wrapper.backbone.target_logprobs_from_features
    if chunk is None:
        return readout(features, batch.labels).float()
    return torch.cat([
        readout(part, labels).float()
        for part, labels in zip(features.split(chunk), batch.labels.split(chunk))
    ])


def autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type, dtype=_AUTOCAST_DTYPE, enabled=device.type == "cuda"
    )


def load_probe_actor(
    checkpoint: Path,
    device: torch.device,
    *,
    memory_heads: int = DEFAULT_MEMORY_HEADS,
    memory_head_dim: int = DEFAULT_MEMORY_HEAD_DIM,
    seed: int,
) -> tuple[LatentThoughtModel, dict]:
    """The memory actor as ``train_latent_vapo`` builds it, and SFT metadata."""
    from postraining.model_io import load_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sft = payload.get("sft") or {}
    sft_args = sft.get("args") or {}
    if not (sft_args.get("think_tokens") and sft_args.get("answer_fence")):
        raise ValueError("the probe needs a think/answer-fence SFT checkpoint")
    require_answer_fence_prompt_schema(sft, answer_fence=True, source=str(checkpoint))
    backbone = load_model(checkpoint, device, payload=payload)
    torch.manual_seed(seed)
    wrapper = LatentThoughtModel(
        backbone,
        hidden_carry=True,
        memory_carry=True,
        memory_heads=memory_heads,
        memory_head_dim=memory_head_dim,
    ).to(device)
    wrapper.eval()
    return wrapper, sft


# --------------------------------------------------------------- optimizers


def fresh_muon_learning_rate(fresh_learning_rate: float) -> float:
    return fresh_learning_rate * MUON_RATE_FACTOR


def memory_optimizers(
    wrapper: LatentThoughtModel, fresh_learning_rate: float
) -> dict[str, torch.optim.Optimizer]:
    """The trainer's fresh-module partition (``build_optimizers``).

    Matrices (ndim >= 2, gate projections included) under Muon at the
    derived rate; vectors under AdamW at ``fresh_learning_rate``.
    """
    parameters = [p for p in wrapper.new_parameters() if p.requires_grad]
    vectors = [p for p in parameters if p.ndim < 2]
    matrices = [p for p in parameters if p.ndim >= 2]
    fused = all(p.is_cuda for p in parameters)
    optimizers: dict[str, torch.optim.Optimizer] = {}
    if vectors:
        optimizers["actor_fresh"] = torch.optim.AdamW(
            vectors, lr=fresh_learning_rate, weight_decay=0.0, fused=fused
        )
    if matrices:
        optimizers["actor_fresh_muon"] = Muon(
            matrices, lr=fresh_muon_learning_rate(fresh_learning_rate), weight_decay=0.0
        )
    return optimizers


# --------------------------------------------------------------------- arms


@dataclass(frozen=True)
class Arm:
    name: str
    memory: bool
    fresh_learning_rate: float | None = None
    write_layer: int | None = None
    shuffle: bool = False
    trunk_lr_scale: float | None = None

    @property
    def trains(self) -> bool:
        return self.fresh_learning_rate is not None or self.trunk_lr_scale is not None


def standard_arms(
    fresh_learning_rate: float, write_layer: int, trunk_lr_scale: float
) -> dict[str, Arm]:
    rate = fresh_learning_rate
    arms = (
        Arm("off", memory=False),
        Arm("mem", memory=True, fresh_learning_rate=rate),
        Arm("mem_lr_quarter", memory=True, fresh_learning_rate=rate / 4),
        Arm("mem_lr_x4", memory=True, fresh_learning_rate=rate * 4),
        Arm("mem_midlayer", memory=True, fresh_learning_rate=rate, write_layer=write_layer),
        Arm("mem_shuffled", memory=True, fresh_learning_rate=rate, shuffle=True),
        Arm("trunk_only", memory=False, trunk_lr_scale=trunk_lr_scale),
        Arm("mem_plus_trunk", memory=True, fresh_learning_rate=rate,
            trunk_lr_scale=trunk_lr_scale),
    )
    return {arm.name: arm for arm in arms}


def set_trainable(wrapper: LatentThoughtModel, arm: Arm) -> None:
    for parameter in wrapper.backbone.parameters():
        parameter.requires_grad_(arm.trunk_lr_scale is not None)
    for parameter in wrapper.new_parameters():
        parameter.requires_grad_(arm.memory and arm.fresh_learning_rate is not None)


def arm_optimizers(wrapper: LatentThoughtModel, arm: Arm) -> dict[str, torch.optim.Optimizer]:
    optimizers: dict[str, torch.optim.Optimizer] = {}
    if arm.memory and arm.fresh_learning_rate is not None:
        optimizers.update(memory_optimizers(wrapper, arm.fresh_learning_rate))
    if arm.trunk_lr_scale is not None:
        from postraining.sft_trace_train import build_optimizers as sft_optimizers

        adamw, muon = sft_optimizers(wrapper.backbone, arm.trunk_lr_scale)
        optimizers["trunk"], optimizers["trunk_muon"] = adamw, muon
    return optimizers


# ---------------------------------------------------------------- telemetry


def memory_weight_telemetry(wrapper: LatentThoughtModel, initial: dict[str, Tensor]) -> dict:
    rms = lambda tensor: float(tensor.detach().float().square().mean().sqrt())
    memory = wrapper.combiner
    values = {
        "weights/read_out_rms": rms(memory.reader.out.weight),
        "weights/query_rms": rms(memory.reader.query.weight),
        "weights/key_rms": rms(memory.writer.key.weight),
        "weights/value_rms": rms(memory.writer.value.weight),
        "weights/write_gate_rms": rms(memory.writer.write_gate.weight),
        "weights/decay_gate_rms": rms(memory.writer.decay_gate.weight),
    }
    for name, parameter in memory.named_parameters():
        start = initial[name]
        values[f"drift/{name}"] = rms(parameter - start) / max(rms(start), 1e-12)
    return values


class EvalAccumulator:
    """Held-out sums: CE and greedy hits per class, memory and movement terms."""

    def __init__(self, heads: int):
        self.ce = defaultdict(float)
        self.hits = defaultdict(float)
        self.count = defaultdict(float)
        self.module_ce = defaultdict(float)
        self.module_count = defaultdict(float)
        self.read_delta = 0.0
        self.read_token = 0.0
        self.read_ratios: list[np.ndarray] = []
        self.beta = np.zeros(heads)
        self.log_timescale = np.zeros(heads)
        self.writes = 0.0
        self.source_top_share = 0.0
        self.source_tokens = 0.0
        self.movement = defaultdict(lambda: [0.0, 0.0])
        self.problems: dict[str, dict] = {}

    def add(
        self,
        wrapper: LatentThoughtModel,
        batch: StreamBatch,
        out: StreamOutput,
        carried: Tensor | None,
        movements: list[tuple[Tensor, Tensor]],
        logprobs: Tensor,
        greedy: Tensor,
    ) -> None:
        nll = (-logprobs).double().cpu().numpy()
        hit = (greedy == batch.labels).cpu().numpy()
        masks = {name: mask.cpu().numpy() for name, mask in batch.classes.items()}
        # Per-row sums by bincount over each supervised slot's row.
        rows = len(batch.traces)
        per_row = {
            name: (
                np.bincount(batch.row_of[mask], weights=nll[mask], minlength=rows),
                np.bincount(batch.row_of[mask], minlength=rows),
                np.bincount(batch.row_of[mask], weights=hit[mask], minlength=rows),
            )
            for name, mask in masks.items()
        }
        for name, mask in masks.items():
            self.ce[name] += float(nll[mask].sum())
            self.hits[name] += float(hit[mask].sum())
            self.count[name] += float(mask.sum())
        for row, trace in enumerate(batch.traces):
            record = {"module": trace.module}
            for name, (ce, count, hits) in per_row.items():
                record[name] = [float(ce[row]), int(count[row]), int(hits[row])]
            self.problems[trace.problem] = record
            self.module_ce[trace.module] += record["all"][0]
            self.module_count[trace.module] += record["all"][1]
        for index, (delta, base) in enumerate(movements):
            self.movement[f"pass{index + 1}"][0] += float(delta)
            self.movement[f"pass{index + 1}"][1] += float(base)
        if carried is None:
            return
        # The fixed point's own stream: for the shuffle, the row's c_K
        # against its c_{K-1}, not against the donor's.
        moved, base = relative_movement(out.source, carried.own, batch.read)
        self.movement["residual"][0] += float(moved)
        self.movement["residual"][1] += float(base)
        read = batch.read
        delta = (out.inputs.float() - out.token_latent.float())[read]
        token = out.token_latent.float()[read]
        delta_sq, token_sq = delta.square().sum(-1), token.square().sum(-1)
        self.read_delta += float(delta_sq.sum())
        self.read_token += float(token_sq.sum())
        self.read_ratios.append((delta_sq / token_sq.clamp_min(1e-30)).sqrt().cpu().numpy())
        write_mask = memory_write_mask(batch.kind, PAD_SLOT)
        response = (write_mask & batch.read)
        hiddens = shift_carried(carried.written, write_mask)
        _, _, log_decay, beta = wrapper.combiner.writer(hiddens[response])
        # Outlier channels in the write source: the share of each vector's
        # squared norm in its top 1% of channels (a raw mid-layer residual
        # dominated by a few channels would starve the unit-RMS keys).
        source = out.source.float()[read].square()
        top = source.topk(max(1, source.size(-1) // 100), dim=-1).values.sum(-1)
        self.source_top_share += float((top / source.sum(-1).clamp_min(1e-30)).sum())
        self.source_tokens += float(source.size(0))
        log_timescale = -torch.log((-torch.expm1(log_decay)).clamp_min(1e-12))
        self.beta += beta.double().sum(0).cpu().numpy()
        self.log_timescale += log_timescale.double().sum(0).cpu().numpy()
        self.writes += float(beta.size(0))

    def summary(self) -> dict:
        result = {}
        for name in TOKEN_CLASSES:
            if self.count.get(name):
                result[f"ce/{name}"] = self.ce[name] / self.count[name]
                result[f"acc/{name}"] = self.hits[name] / self.count[name]
                result[f"count/{name}"] = self.count[name]
        for module in sorted(self.module_ce):
            result[f"module_ce/{module}"] = self.module_ce[module] / self.module_count[module]
        for name, (delta, base) in self.movement.items():
            result[f"fixed_point/{name}"] = math.sqrt(delta / max(base, 1e-30))
        if self.read_token:
            ratios = np.concatenate(self.read_ratios)
            result["memory/read_input_rms_ratio"] = math.sqrt(self.read_delta / self.read_token)
            for quantile in (10, 50, 90, 99):
                result[f"memory/read_ratio_p{quantile}"] = float(np.percentile(ratios, quantile))
            beta = self.beta / max(self.writes, 1)
            timescale = self.log_timescale / max(self.writes, 1)
            result["memory/write_source_top1pct_share"] = self.source_top_share / max(self.source_tokens, 1)
            result["memory/write_beta_mean"] = float(beta.mean())
            result["memory/decay_log_timescale_mean"] = float(timescale.mean())
            for head in range(beta.size):
                result[f"memory/write_beta_h{head}"] = float(beta[head])
                result[f"memory/decay_log_timescale_h{head}"] = float(timescale[head])
        return result


def evaluate(
    wrapper: LatentThoughtModel,
    batches: Sequence[StreamBatch],
    config: ProbeConfig,
    seed: int,
    readout_chunk: int = 4096,
) -> tuple[dict, dict[str, dict]]:
    """Held-out CE and greedy accuracy per class, plus memory telemetry."""
    accumulator = EvalAccumulator(wrapper.combiner.heads)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for batch in batches:
            device = batch.ids.device
            foreign = config.shuffle or config.read_ablation == "foreign_writes"
            donor = derangement(len(batch.traces), generator) if foreign else None
            with autocast(device):
                out, carried, movements = probe_forward(wrapper, batch, config, donor)
                logprobs = target_logprobs(wrapper, out, batch, readout_chunk)
                features = supervised_features(wrapper, out, batch)
                greedy = torch.cat([
                    wrapper.backbone.logits_from_features(chunk).argmax(-1)
                    for chunk in features.split(readout_chunk)
                ])
            accumulator.add(wrapper, batch, out, carried, movements, logprobs, greedy)
    return accumulator.summary(), accumulator.problems


# -------------------------------------------------------------------- train


RL_RUN_STEPS = 340  # the fresh-rate drill RL run's length (NOTES 2026-10-21)


def eval_schedule(steps: int) -> set[int]:
    """Doubling steps 3, 6, 12, 25, 50, ... plus 0, the RL run's length and the end.

    Doubling puts every eval of the L/4 arm at four times an eval of the L
    arm and every eval of the 4L arm at a quarter of one, so curves can be
    compared at matched lr x step.
    """
    doubling = {round(25 * 2.0 ** k) for k in range(-3, 40) if 25 * 2.0 ** k <= steps}
    return {0, steps} | doubling | ({RL_RUN_STEPS} if RL_RUN_STEPS <= steps else set())


def evaluate_panels(
    wrapper: LatentThoughtModel,
    panels: dict[str, Sequence[StreamBatch]],
    config: ProbeConfig,
    seed: int,
) -> tuple[dict, dict[str, dict]]:
    """``evaluate`` on every panel, keys prefixed ``<panel>/``."""
    summary, problems = {}, {}
    for name, batches in panels.items():
        panel_summary, problems[name] = evaluate(wrapper, batches, config, seed)
        summary.update({f"{name}/{key}": value for key, value in panel_summary.items()})
    return summary, problems


def train_arm(
    wrapper: LatentThoughtModel,
    initial_state: dict[str, Tensor],
    arm: Arm,
    *,
    train_traces: Sequence[EncodedTrace],
    panels: dict[str, Sequence[StreamBatch]],
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    loss_normalizer: float,
) -> dict:
    from torch.utils.tensorboard import SummaryWriter

    from postraining.sft_trace_train import lr_scale_at, muon_momentum_at

    wrapper.load_state_dict(initial_state, strict=True)
    set_trainable(wrapper, arm)
    optimizers = arm_optimizers(wrapper, arm)
    config = ProbeConfig(
        memory=arm.memory, write_layer=arm.write_layer, shuffle=arm.shuffle,
        passes=args.fixed_point_passes,
    )
    memory_initial = {
        name: parameter.detach().clone()
        for name, parameter in wrapper.combiner.named_parameters()
    }
    arm_dir = run_dir / arm.name
    arm_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = arm_dir / "metrics.jsonl"
    writer = SummaryWriter(str(run_dir / "tensorboard" / arm.name))
    # Every arm draws the same batches in the same order.
    sampler = iter(CellBalancedSampler(train_traces, args.batch_trajectories, args.macro_batches, args.seed))
    shuffle_generator = torch.Generator().manual_seed(args.seed + 1)
    steps = args.steps if arm.trains else 0
    schedule = eval_schedule(steps)
    started = time.monotonic()
    train_window: list[dict] = []

    def log(record: dict, step: int) -> None:
        record = {"arm": arm.name, "step": step, "elapsed_s": time.monotonic() - started, **record}
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        for key, value in record.items():
            if isinstance(value, (int, float)) and key not in ("step", "arm"):
                writer.add_scalar(f"{record['kind']}/{key}", value, step)

    for step in range(steps + 1):
        if step in schedule:
            eval_started = time.monotonic()
            summary, _ = evaluate_panels(wrapper, panels, config, args.seed + 2)
            if arm.memory:
                summary.update(memory_weight_telemetry(wrapper, memory_initial))
            summary["eval_seconds"] = time.monotonic() - eval_started
            log({"kind": "eval", **summary}, step)
            print(f"[{arm.name}] step {step} heldout ce {summary['heldout/ce/all']:.5f} "
                  f"hazard ce {summary['hazard/ce/all']:.5f} read "
                  f"{summary.get('heldout/memory/read_input_rms_ratio', 0):.4f} "
                  f"({summary['eval_seconds']:.1f}s)", flush=True)
        if step == steps:
            break
        step_started = time.monotonic()
        if "trunk_muon" in optimizers:
            for group in optimizers["trunk_muon"].param_groups:
                group["mu"] = muon_momentum_at(step)
            # The SFT schedule's shape: linear warmup, linear decay to zero.
            # The fresh memory keeps the trainer's constant rate.
            scale = lr_scale_at(step, steps, args.trunk_warmup_steps)
            for name in ("trunk", "trunk_muon"):
                for group in optimizers[name].param_groups:
                    group["lr"] = group["initial_lr"] * scale
        # One optimizer step per batch, accumulated over micro-batches of at
        # most --train-slot-budget padded slots (each with >= 2 rows for the
        # shuffle's derangement); the loss normalizer is the whole batch's.
        sums = defaultdict(float)
        micro_batches = evaluation_batches(next(sampler), args.train_slot_budget)
        for micro in micro_batches:
            batch = collate(micro, device, stop_id=args.stop_id)
            donor = derangement(len(batch.traces), shuffle_generator) if arm.shuffle else None
            with autocast(device):
                out, carried, movements = probe_forward(wrapper, batch, config, donor)
                logprobs = target_logprobs(wrapper, out, batch)
            (-logprobs.sum() / loss_normalizer).backward()
            with torch.no_grad():
                sums["nll"] += float(-logprobs.detach().sum())
                sums["tokens"] += batch.labels.numel()
                sums["slots"] += batch.ids.numel()
                if carried is not None:
                    read = batch.read
                    delta = (out.inputs.float() - out.token_latent.float())[read]
                    sums["read_delta"] += float(delta.square().sum())
                    sums["read_token"] += float(out.token_latent.float()[read].square().sum())
                    for index, (moved, base) in enumerate(movements):
                        sums[f"moved{index}"] += float(moved)
                        sums[f"base{index}"] += float(base)
            del out, carried, logprobs
        for optimizer in optimizers.values():
            optimizer.step()
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        record = {
            "ce": sums["nll"] / sums["tokens"],
            "tokens": sums["tokens"],
            "slots": sums["slots"],
            "micro_batches": len(micro_batches),
        }
        if sums["read_token"]:
            record["read_input_rms_ratio"] = math.sqrt(sums["read_delta"] / sums["read_token"])
            for index in range(config.passes - 1):
                record[f"fixed_point_pass{index + 1}"] = math.sqrt(
                    sums[f"moved{index}"] / max(sums[f"base{index}"], 1e-30)
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        record["step_seconds"] = time.monotonic() - step_started
        train_window.append(record)
        if (step + 1) % args.log_every == 0:
            window = {
                key: float(np.mean([r[key] for r in train_window]))
                for key in train_window[0]
            }
            log({"kind": "train", **window}, step + 1)
            train_window.clear()
    # Final: the per-problem tables, and the same eval with two extra passes
    # to price the fixed point's truncation.
    final, problems = evaluate_panels(wrapper, panels, config, args.seed + 2)
    result = {"arm": dataclasses.asdict(arm), "steps": steps, "final": final}
    if arm.memory:
        deeper = dataclasses.replace(config, passes=config.passes + 2)
        extra, _ = evaluate_panels(wrapper, panels, deeper, args.seed + 2)
        result["final_extra_passes"] = {
            key: value for key, value in extra.items()
            if any(part in key for part in ("/ce/", "/acc/", "/fixed_point/", "/memory/read"))
        }
        if not arm.shuffle:
            result["final_ablations"] = {}
            for ablation in READ_ABLATIONS:
                ablated = dataclasses.replace(config, read_ablation=ablation)
                summary, ablated_problems = evaluate_panels(wrapper, panels, ablated, args.seed + 2)
                result["final_ablations"][ablation] = {
                    key: value for key, value in summary.items()
                    if any(part in key for part in ("/ce/", "/acc/", "/fixed_point/", "/memory/read_input"))
                }
                (arm_dir / f"final_problems_{ablation}.json").write_text(json.dumps(ablated_problems))
        torch.save(
            {k: v.detach().cpu() for k, v in wrapper.combiner.state_dict().items()},
            arm_dir / "memory_final.pt",
        )
    (arm_dir / "final_problems.json").write_text(json.dumps(problems))
    (arm_dir / "result.json").write_text(json.dumps(result, indent=2))
    writer.close()
    return result


def train_main(args: argparse.Namespace) -> None:
    from postraining.provenance import capture_source_provenance

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    run_dir = Path(args.output)
    run_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    args.stop_id = tokenizer.eos_id()
    train_rows = pq.read_table(args.train_data).to_pylist()
    hazard_rows = pq.read_table(args.hazard_panel).to_pylist()
    heldout_drills = rebuild_pool_drills(Path(args.heldout_data))
    evaluation_texts = {drill.problem for drill in heldout_drills} | {
        row["problem"] for row in hazard_rows
    }
    if any(row["problem"] in evaluation_texts for row in train_rows):
        raise ValueError("training traces contain evaluation problems")
    train_traces = [
        trace for trace in encode_table(tokenizer, train_rows, annotate=False)
        if trace.length <= args.max_length
    ]
    panel_traces = {
        "heldout": encode_table(tokenizer, drill_rows(heldout_drills), annotate=True),
        "hazard": encode_table(tokenizer, hazard_rows, annotate=True),
    }
    if max(t.length for traces in panel_traces.values() for t in traces) > args.max_length:
        raise ValueError("an evaluation trace exceeds --max-length")
    wrapper, sft = load_probe_actor(
        Path(args.checkpoint), device, memory_heads=args.memory_heads,
        memory_head_dim=args.memory_head_dim, seed=args.seed,
    )
    if args.write_layer >= len(wrapper.backbone.blocks):
        raise ValueError(
            f"--write-layer {args.write_layer} must be a middle block "
            f"(1..{len(wrapper.backbone.blocks) - 1})"
        )
    if args.trunk_lr_scale is None:
        args.trunk_lr_scale = float(sft["args"]["lr_scale"])
    arms = standard_arms(args.fresh_learning_rate, args.write_layer, args.trunk_lr_scale)
    unknown = set(args.arms) - set(arms)
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}; choose from {sorted(arms)}")
    cells = sorted({trace.module for trace in train_traces})
    per_cell_supervised = [
        np.mean([t.supervised for t in train_traces if t.module == cell]) for cell in cells
    ]
    loss_normalizer = args.batch_trajectories * float(np.mean(per_cell_supervised))
    manifest = {
        "schema": PROBE_SCHEMA,
        "args": {k: v for k, v in vars(args).items() if k != "func"},
        "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "train_data_sha256": file_sha256(Path(args.train_data)),
        "heldout_data_sha256": file_sha256(Path(args.heldout_data)),
        "hazard_panel_sha256": file_sha256(Path(args.hazard_panel)),
        "train_traces": len(train_traces),
        "train_traces_dropped_over_max_length": len(train_rows) - len(train_traces),
        "panel_traces": {name: len(traces) for name, traces in panel_traces.items()},
        "panel_class_counts": {
            panel: {
                name: int(sum(t.classes[name].sum() for t in traces)) for name in TOKEN_CLASSES
            }
            for panel, traces in panel_traces.items()
        },
        "loss_normalizer": loss_normalizer,
        "fresh_muon_learning_rate": fresh_muon_learning_rate(args.fresh_learning_rate),
        "arms": {name: dataclasses.asdict(arms[name]) for name in args.arms},
        "memory_geometry": [wrapper.combiner.heads, wrapper.combiner.head_dim],
        "sft_lineage": {k: sft.get(k) for k in ("base_checkpoint", "traces", "traces_sha256")},
    }
    manifest["source"] = capture_source_provenance(
        run_dir / "provenance", Path(__file__).resolve().parents[1]
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: manifest[k] for k in (
        "train_traces", "panel_traces", "panel_class_counts", "loss_normalizer")}), flush=True)
    panels = {
        name: [
            collate(batch, device, stop_id=args.stop_id)
            for batch in evaluation_batches(traces, args.eval_slot_budget)
        ]
        for name, traces in panel_traces.items()
    }
    initial_state = {k: v.detach().clone() for k, v in wrapper.state_dict().items()}
    for name in args.arms:
        result = train_arm(
            wrapper, initial_state, arms[name], train_traces=train_traces,
            panels=panels, args=args, run_dir=run_dir, device=device,
            loss_normalizer=loss_normalizer,
        )
        print(f"[{name}] final heldout ce {result['final']['heldout/ce/all']:.5f} "
              f"hazard ce {result['final']['hazard/ce/all']:.5f}", flush=True)


# ------------------------------------------------------------------ analyze


def paired_difference(
    arm: dict[str, dict], base: dict[str, dict], name: str
) -> tuple[float, float, float]:
    """(arm CE - base CE per token, its SE clustered by problem, base CE)."""
    problems = sorted(set(arm) & set(base))
    deltas = np.array([arm[p][name][0] - base[p][name][0] for p in problems])
    counts = np.array([base[p][name][1] for p in problems], dtype=float)
    if counts.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    mean = deltas.sum() / counts.sum()
    residual = deltas - mean * counts
    se = math.sqrt((residual ** 2).sum() * len(problems) / max(len(problems) - 1, 1)) / counts.sum()
    base_ce = sum(base[p][name][0] for p in problems) / counts.sum()
    return float(mean), float(se), float(base_ce)


CURVE_KEYS = (
    "ce/all", "ce/flat", "ce/comp_line", "ce/prose", "ce/borrow_equal", "ce/borrow_equal_after",
    "ce/borrow_none", "ce/pp_readout", "ce/pp_shift", "ce/pp_add_leftmost", "acc/flat",
    "acc/borrow_equal", "acc/pp_readout", "memory/read_input_rms_ratio", "fixed_point/pass1",
    "fixed_point/residual",
)
SHOWN_CLASSES = ("flat", "all", "comp_line", "prose", "borrow_equal", "borrow_equal_after",
                 "borrow_none", "borrow_real", "pp_readout", "pp_add_leftmost")


def load_runs(runs: Sequence[str]) -> tuple[dict, dict, dict, dict]:
    """Per-arm (problem tables, results, eval curves, ablation tables).

    Refuses a run whose manifest names an arm with no result (a job that hit
    its time limit) and trained arms with different step counts.
    """
    problems, results, curves, ablations = {}, {}, {}, {}
    for run in map(Path, runs):
        manifest = json.loads((run / "manifest.json").read_text())
        for name in manifest["arms"]:
            arm_dir = run / name
            if not (arm_dir / "result.json").exists():
                raise FileNotFoundError(f"{arm_dir} has no result.json; the arm did not finish")
            if name in results:
                raise ValueError(f"arm {name} appears in more than one run")
            problems[name] = json.loads((arm_dir / "final_problems.json").read_text())
            results[name] = json.loads((arm_dir / "result.json").read_text())
            curves[name] = [
                record for record in map(json.loads, (arm_dir / "metrics.jsonl").read_text().splitlines())
                if record["kind"] == "eval"
            ]
            ablations[name] = {
                ablation: json.loads((arm_dir / f"final_problems_{ablation}.json").read_text())
                for ablation in READ_ABLATIONS
                if (arm_dir / f"final_problems_{ablation}.json").exists()
            }
    trained = {result["steps"] for result in results.values() if result["steps"]}
    if len(trained) > 1:
        raise ValueError(f"trained arms ran different step counts {sorted(trained)}")
    return problems, results, curves, ablations


def paired_table(arm: dict[str, dict], base: dict[str, dict]) -> dict:
    return {
        panel: {
            cls: dict(zip(("delta", "se", "base"), paired_difference(arm[panel], base[panel], cls)))
            for cls in TOKEN_CLASSES
        }
        for panel in base
    }


def print_table(title: str, rows: dict[str, dict], base_label: str) -> None:
    print(f"\n{title}")
    print(" | ".join(["arm"] + list(SHOWN_CLASSES)))
    base = next(iter(rows.values()))
    print(" | ".join([base_label] + [f"{base[c]['base']:.4f}" for c in SHOWN_CLASSES]))
    for name, row in rows.items():
        print(" | ".join([name] + [f"{row[c]['delta']:+.4f}±{row[c]['se']:.4f}" for c in SHOWN_CLASSES]))


def analyze_main(args: argparse.Namespace) -> None:
    problems, results, curves, ablations = load_runs(args.runs)
    report: dict = {"vs": {}, "ablations": {}, "curves": {}}
    for reference in ("off", "trunk_only", "mem_shuffled"):
        if reference in problems:
            report["vs"][reference] = {
                name: paired_table(problems[name], problems[reference])
                for name in problems if name != reference
            }
    # Each ablated read against the same arm's full read (positive: the
    # ablated part was carrying information).
    for name, tables in ablations.items():
        if not tables:
            continue
        report["ablations"][name] = {
            ablation: paired_table(table, problems[name]) for ablation, table in tables.items()
        }
    report["final"] = {name: result["final"] for name, result in results.items()}
    report["final_extra_passes"] = {
        name: result.get("final_extra_passes") for name, result in results.items()
    }
    # Read share is only evidence relative to the capacity-matched control.
    if "mem_shuffled" in results:
        shuffled = results["mem_shuffled"]["final"]
        report["read_share_vs_shuffled"] = {
            name: {
                panel: result["final"].get(f"{panel}/memory/read_input_rms_ratio", float("nan"))
                / shuffled[f"{panel}/memory/read_input_rms_ratio"]
                for panel in ("heldout", "hazard")
            }
            for name, result in results.items() if f"hazard/memory/read_input_rms_ratio" in result["final"]
        }
    for name, evals in curves.items():
        curve = {"steps": [r["step"] for r in evals]}
        for panel in ("heldout", "hazard"):
            for key in CURVE_KEYS:
                curve[f"{panel}/{key}"] = [r.get(f"{panel}/{key}") for r in evals]
        # Adoption speed on the primary class: the first eval step reaching
        # half / 90% of the arm's final hazard flat-class CE drop.
        ce = curve["hazard/ce/flat"]
        drop = ce[0] - ce[-1] if ce and None not in (ce[0], ce[-1]) else 0.0
        for fraction, label in ((0.5, "half"), (0.9, "90pct")):
            curve[f"steps_to_{label}_drop"] = next(
                (step for step, value in zip(curve["steps"], ce) if ce[0] - value >= fraction * drop),
                None,
            ) if drop > 0 else None
        curve["weights/read_out_rms"] = [r.get("weights/read_out_rms") for r in evals]
        report["curves"][name] = curve
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    for reference, arms in report["vs"].items():
        for panel in ("hazard", "heldout"):
            print_table(
                f"[{panel}] CE minus {reference} (nats/token, SE clustered by problem)",
                {name: table[panel] for name, table in arms.items()}, f"{reference} CE",
            )
    for name, tables in report["ablations"].items():
        print_table(
            f"[hazard] {name}: ablated read minus full read",
            {ablation: table["hazard"] for ablation, table in tables.items()}, f"{name} CE",
        )
    for name, shares in report.get("read_share_vs_shuffled", {}).items():
        print(f"read share / mem_shuffled's  {name}: hazard {shares['hazard']:.3f} "
              f"heldout {shares['heldout']:.3f}")
    for name, curve in report["curves"].items():
        print(f"\n{name}: steps {curve['steps']}")
        for key in ("hazard/ce/flat", "hazard/ce/all", "heldout/ce/all",
                    "hazard/memory/read_input_rms_ratio", "hazard/fixed_point/residual"):
            print(f"  {key}: " + " ".join("-" if v is None else f"{v:.4f}" for v in curve[key]))
        print(f"  hazard flat steps to half / 90% of final drop: "
              f"{curve['steps_to_half_drop']} / {curve['steps_to_90pct_drop']}")


# ------------------------------------------------------------------ data cli


def build_data_main(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"{output} exists; data artifacts are immutable")
    table, manifest = build_training_table(
        Path(args.train_pool), Path(args.heldout_pool),
        pool_seed=args.pool_seed, per_cell=args.per_cell,
    )
    pq.write_table(table, output)
    manifest["sha256"] = file_sha256(output)
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def build_panel_main(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"{output} exists; data artifacts are immutable")
    table, manifest = build_hazard_panel(
        Path(args.train_data), Path(args.train_pool), Path(args.heldout_pool),
        seed=args.seed, per_cell=args.per_cell,
        family_per_cell=dict(parse_family_count(item) for item in args.family_per_cell),
    )
    pq.write_table(table, output)
    manifest["sha256"] = file_sha256(output)
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def parse_family_count(item: str) -> tuple[str, int]:
    family, _, count = item.partition("=")
    if not family or not count.isdigit() or int(count) < 1:
        raise ValueError(f"expected FAMILY=N, got {item!r}")
    return family, int(count)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("build-data")
    data.add_argument("--train-pool", default="postraining/data/drill_rl_study_train_20261014_v1.parquet")
    data.add_argument("--heldout-pool", default="postraining/data/drill_rl_study_heldout_20261014_v1.parquet")
    data.add_argument("--pool-seed", type=int, default=PROBE_TRAIN_POOL_SEED)
    data.add_argument("--per-cell", type=int, default=4096)
    data.add_argument("--output", required=True)
    data.set_defaults(func=build_data_main)

    panel = commands.add_parser("build-panel")
    panel.add_argument("--train-data", required=True)
    panel.add_argument("--train-pool", default="postraining/data/drill_rl_study_train_20261014_v1.parquet")
    panel.add_argument("--heldout-pool", default="postraining/data/drill_rl_study_heldout_20261014_v1.parquet")
    panel.add_argument("--seed", type=int, default=PROBE_HAZARD_PANEL_SEED)
    panel.add_argument("--per-cell", type=int, default=80)
    panel.add_argument("--family-per-cell", nargs="*", default=[], metavar="FAMILY=N")
    panel.add_argument("--output", required=True)
    panel.set_defaults(func=build_panel_main)

    train = commands.add_parser("train")
    train.add_argument("--checkpoint", default=RL_START_CHECKPOINT)
    train.add_argument("--train-data", required=True)
    train.add_argument("--heldout-data", default="postraining/data/drill_rl_study_heldout_20261014_v1.parquet")
    train.add_argument("--hazard-panel", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--arms", nargs="+", required=True)
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--batch-trajectories", type=int, default=32)
    train.add_argument("--macro-batches", type=int, default=4)
    train.add_argument("--trunk-warmup-steps", type=int, default=20,
                       help="trunk arms' linear warmup (the SFT run's)")
    train.add_argument("--log-every", type=int, default=25)
    train.add_argument("--eval-slot-budget", type=int, default=32768)
    train.add_argument("--train-slot-budget", type=int, default=16384,
                       help="padded slots per training micro-batch (gradient accumulation)")
    train.add_argument("--fixed-point-passes", type=int, default=2)
    train.add_argument("--fresh-learning-rate", type=float, default=3e-3)
    train.add_argument("--write-layer", type=int, default=4)
    train.add_argument("--trunk-lr-scale", type=float, default=None,
                       help="default: the checkpoint's SFT lr_scale")
    train.add_argument("--memory-heads", type=int, default=DEFAULT_MEMORY_HEADS)
    train.add_argument("--memory-head-dim", type=int, default=DEFAULT_MEMORY_HEAD_DIM)
    train.add_argument("--max-length", type=int, default=2048)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--device", default="cuda")
    train.set_defaults(func=train_main)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("runs", nargs="+")
    analyze.add_argument("--json", default=None)
    analyze.set_defaults(func=analyze_main)
    return parser


def validate_train_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command != "train":
        return
    if args.fixed_point_passes < 1:
        parser.error("--fixed-point-passes must be at least 1")
    if args.batch_trajectories < 2:
        parser.error("--batch-trajectories must be at least 2 (the shuffle control)")
    if not 1 <= args.write_layer:
        parser.error("--write-layer must name a block (1-based)")
    if args.steps < 0 or args.log_every < 1 or args.trunk_warmup_steps < 0:
        parser.error("--steps and --trunk-warmup-steps must be >= 0, --log-every positive")
    if len(set(args.arms)) != len(args.arms):
        parser.error("--arms must not repeat")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_train_args(parser, args)
    args.func(args)


if __name__ == "__main__":
    main()

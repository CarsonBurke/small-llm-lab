"""Admission, hygiene, and ownership screens shared by math RL prompt pools.

A math RL pool is a parquet of verifier rows (``prompt``, ``reward_model``,
``extra_info``) that ``prepare_vapo_mixture`` hash-binds as one source. Every
pool builder applies the same screens, in this order, and counts every drop:

1. ``uncanonicalizable`` -- the answer-fence canonicalizer refuses the prompt
   (a literal ``Answer:`` in the problem body, an unregistered template).
   Quarantined with the reason, never reshaped.
2. ``control_character`` -- the problem carries a C0 control byte other than
   tab/newline/carriage return. In DAPO these are JSON-escape corruption
   (``"\\frac"`` decoded as form feed + ``rac``) or PDF ligature loss
   (``\x0crst`` for "first"); the two are not separable by rule, so the row is
   dropped rather than repaired.
3. ``unverifiable_target`` -- the ground truth does not grade as correct when
   it is itself the fenced answer, under the row's own answer style and the
   exact call the trainer makes (``verify_answer("Answer: " + span, ...,
   window=None)``). Such a row can never pay reward.
   ``equation_target`` -- a Minerva-graded truth that states no single value
   (``core.graded_answer_field`` is None): an equation between expressions
   (``x + 2y - 5 = 0``), several equations (``a = b = c``), or an ``=`` inside
   a subscript. Minerva's normalization keeps only the text after the last
   ``=``, so such a row would pay a bare ``0`` or ``c`` and punish the
   equation actually asked for. Quarantined.
4. ``contaminated`` / ``contains_reference_problem`` -- the evaluation screens
   the SFT corpora use: exact normalized match or any shared word 8-gram with
   GSM8K test, the DeepMind interpolate-easy bench panel and AIME 2024/2025/
   2026, plus per-reference containment of GSM8K (the RL prompt pool) and
   KodCode.
5. ``prompt_over_budget`` -- the BOS-framed GPT-2 BPE prompt exceeds the RL
   prompt budget. ``encode_prompt`` keeps the *last* tokens of an over-long
   prompt, so such a problem would be served with its opening cut off.
6. ``owned_by:<pool>`` -- the problem restates a row of a pool that owns it
   (``problem_overlap``, any matcher), so this pool yields it.
7. Restatements within the pool, in two tiers.

   *Same text* (equal ``skeleton_key``, which whitespace equality implies):
   the copies ask the same question, so agreeing targets collapse to the
   lowest identity and disagreeing targets quarantine the whole group -- at
   least one label is wrong and nothing here can tell which.

   *Near-duplicate text* (shingle containment): a row collapses into an
   earlier kept row it matches *directly* when their targets agree. Matches
   are not chained, so one loose edge cannot fuse distinct problems into a
   component. A shingle neighbour with a *different* target is kept: on
   inspection those are answer-form rewrites ("... in the form
   \\frac{m}{n}; provide m + n") or symbolic siblings the digit guard cannot
   separate (``+`` against ``-``, third against fourth derivative), each
   correct for its own question, not label errors.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from postraining.core import (
    POSTTRAIN_REWARD_SCHEMA,
    answer_style,
    encode_prompt,
    graded_answer_field,
    math_corpus_policy_sha256,
    verify_answer,
)
from postraining.math_prompt import canonicalize_answer_fence_rows
from postraining.problem_overlap import (
    ProblemOverlapIndex,
    skeleton_key,
    template_shingles,
)

MATH_RL_POOL_SCHEMA = "math_rl_prompt_pool/v1"
ULTRADATA_EXTRACTION_SCHEMAS = frozenset({"ultradata_subset/v1", "verifiable_corpus/v1"})
# ``scripts/launch_kda8_posttrain.sh`` runs RL at ``--prompt-tokens 256``.
PROMPT_TOKEN_BUDGET = 256
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Modules whose code decides what a pool contains; their bytes are bound into
# every pool manifest.
POLICY_SOURCES = (
    Path(__file__),
    Path(__file__).with_name("problem_overlap.py"),
    Path(__file__).with_name("math_prompt.py"),
)


def policy_provenance() -> dict:
    return {
        "sources": [
            {"path": str(path.relative_to(path.parents[1])), "sha256": file_sha256(path)}
            for path in POLICY_SOURCES
        ],
        "math_corpus_policy_sha256": math_corpus_policy_sha256(),
        # The target self-verification and equation screens run the
        # trainer's grader, whose semantics this names.
        "reward_schema": POSTTRAIN_REWARD_SCHEMA,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ultradata_math(extractions: Sequence[Path]) -> tuple[list[dict], Counter, dict]:
    """Math rows of UltraData-RL-2609 extractions, one per source query.

    Both splits of each extraction are candidates: its train/validation cut
    served that extraction's own assessment, and the RL bench panels are
    separate held-out files. Returns (rows, counts, provenance).
    """
    revisions, schemas, reward_identities = set(), set(), set()
    rows, counts, input_hashes = [], Counter(), []
    seen: dict[str, dict] = {}
    for directory in map(Path, extractions):
        manifest = json.loads((directory / "manifest.json").read_text())
        for name in ("manifest.json", "train.parquet", "validation.parquet"):
            input_hashes.append(
                {"path": str(directory / name), "sha256": file_sha256(directory / name)}
            )
        revisions.add(
            manifest.get("source_revision")
            or next(
                source["revision"]
                for source in manifest["sources"]
                if source["dataset"] == "openbmb/UltraData-RL-2609"
            )
        )
        schemas.add(manifest["schema"])
        reward_identities.add(manifest["reward_identity"])
        for split in ("train", "validation"):
            for row in pq.read_table(directory / f"{split}.parquet").to_pylist():
                counts["extracted"] += 1
                info = row["extra_info"]
                if info.get("domain") != "Math":
                    counts["non_math"] += 1
                    continue
                identity = info.get("original_query_sha256")
                if not identity:
                    # A None key would collapse every such row into one
                    # "duplicate".
                    counts["missing_identity"] += 1
                    continue
                if identity in seen:
                    # First-seen would make the pool depend on extraction
                    # order unless every copy carries the same task.
                    if _task(seen[identity]) != _task(row):
                        raise ValueError(
                            f"query {identity} differs between extractions"
                        )
                    counts["duplicate_identity"] += 1
                    continue
                seen[identity] = row
                rows.append(row)
    if len(revisions) != 1:
        raise ValueError(f"extractions disagree on source_revision: {revisions}")
    if not schemas <= ULTRADATA_EXTRACTION_SCHEMAS:
        raise ValueError(f"unexpected extraction schema: {schemas}")
    if len(reward_identities) != 1:
        raise ValueError(f"extractions disagree on reward_identity: {reward_identities}")
    provenance = {
        "dataset": "openbmb/UltraData-RL-2609",
        "source_revision": revisions.pop(),
        "source_schemas": sorted(schemas),
        "reward_identity": reward_identities.pop(),
        "domain": "Math",
        "extractions": [str(path) for path in extractions],
        "input_sha256": input_hashes,
    }
    return rows, counts, provenance


def _task(row: dict) -> tuple:
    return (
        json.dumps(row["prompt"], sort_keys=True),
        json.dumps(row["reward_model"], sort_keys=True),
    )


def canonical_problem(row: dict) -> str:
    """The prompt text the trainer serves for this row, framing removed."""
    (canonical,) = canonicalize_answer_fence_rows([row])
    return "\n".join(message["content"] for message in canonical["prompt"])


def target_self_verifies(row: dict) -> bool:
    """Whether the ground truth, emitted as the fenced answer, earns reward."""
    truth = row["reward_model"]["ground_truth"]
    if not isinstance(truth, str) or not truth.strip():
        return False
    correct, _ = verify_answer(
        "Answer: " + truth, truth, answer_style(row), window=None
    )
    return bool(correct)


class EvaluationGuard:
    """The SFT corpora's evaluation screens, applied to canonical problems."""

    def __init__(self):
        from postraining.prepare_sft_corpus import (
            CONTAINMENT_MIN_OVERLAP,
            CONTAINMENT_TARGETS,
            build_containment_index,
            read_reference_texts,
        )
        from postraining.prepare_sft_traces import (
            DECONTAMINATION_TARGETS,
            build_decontamination_index,
        )

        self.exact, self.ngrams = build_decontamination_index()
        references: list[str] = []
        self.containment_counts: dict[str, int] = {}
        for target, column in CONTAINMENT_TARGETS:
            if not target.exists():
                raise FileNotFoundError(
                    f"{target} is missing; rows would enter without its screen"
                )
            texts = read_reference_texts(target, column)
            self.containment_counts[str(target)] = len(texts)
            references.extend(texts)
        self.reference_exact = {" ".join(text.split()).lower() for text in references}
        self.gram_to_ids, self.sizes = build_containment_index(references)
        self.min_overlap = CONTAINMENT_MIN_OVERLAP
        self.targets = [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in (
                *DECONTAMINATION_TARGETS,
                *(target for target, _ in CONTAINMENT_TARGETS),
            )
        ]

    def reason(self, problem: str) -> str | None:
        from postraining.prepare_sft_corpus import contains_reference_problem
        from postraining.prepare_sft_traces import contaminated

        if contaminated(problem, self.exact, self.ngrams):
            return "contaminated"
        if contains_reference_problem(
            problem, self.reference_exact, self.gram_to_ids, self.sizes,
            self.min_overlap,
        ):
            return "contains_reference_problem"
        return None

    def provenance(self) -> dict:
        return {
            "rule": "exact normalized text or any shared word 8-gram with the "
            "math evaluation targets; exact or >= "
            f"{self.min_overlap:.0%} per-reference 8-gram containment of the "
            "containment targets",
            "targets": self.targets,
            "containment_rows": self.containment_counts,
        }


class PromptBudget:
    """Token length of a problem exactly as the RL trainer frames it."""

    def __init__(self, budget: int = PROMPT_TOKEN_BUDGET):
        from postraining.core import GPT2BPETokenizer

        self.budget = budget
        self.tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)

    def tokens(self, problem: str) -> int:
        return len(encode_prompt(self.tokenizer, problem))

    def provenance(self) -> dict:
        return {
            "prompt_tokens": self.budget,
            "tokenizer": "gpt2 byte-level BPE, <think>/<answer> registered",
            "framing": "core.encode_prompt: BOS + canonical problem",
        }


@dataclass
class Candidate:
    row: dict
    problem: str
    identity: str
    tokens: int


@dataclass
class ScreenResult:
    kept: list[Candidate]
    counts: Counter = field(default_factory=Counter)
    quarantined: list[dict] = field(default_factory=list)
    near_duplicates: dict = field(default_factory=dict)
    prompt_tokens: list[int] = field(default_factory=list)
    # reason -> ids, for every row the screens dropped without quarantine.
    dropped: dict[str, list] = field(default_factory=dict)


def _targets_agree(rows: Sequence[dict]) -> bool:
    from postraining.prepare_sft_traces import answers_agree

    truths = [row["reward_model"]["ground_truth"] for row in rows]
    return all(answers_agree(truth, truths[0]) for truth in truths[1:])


def screen_math_pool(
    rows: Sequence[dict],
    *,
    identity: Callable[[dict], str],
    guard: EvaluationGuard,
    budget: PromptBudget,
    owners: dict[str, Sequence[tuple[str, str]]] | None = None,
    describe: Callable[[dict], str] = lambda row: str(
        (row.get("extra_info") or {}).get("index")
    ),
) -> ScreenResult:
    """Apply the module's screens in order; see the module docstring.

    ``owners`` maps a pool name to the ``(id, canonical problem)`` pairs it
    owns. Template shingles are learned from both this pool and the owner, so
    a clause either pool stamps on hundreds of problems is not taken as
    evidence of identity.
    """
    result = ScreenResult(kept=[])

    def drop(reason: str, row: dict, detail=None) -> None:
        result.counts[reason] += 1
        entry = describe(row) if detail is None else {"id": describe(row), **detail}
        result.dropped.setdefault(reason, []).append(entry)

    survivors: list[Candidate] = []
    for row in rows:
        result.counts["seen"] += 1
        try:
            problem = canonical_problem(row)
        except ValueError as error:
            result.counts["uncanonicalizable"] += 1
            result.quarantined.append(
                {"id": describe(row), "reason": str(error)[:200]}
            )
            continue
        if _CONTROL.search(problem):
            drop("control_character", row)
            continue
        if not target_self_verifies(row):
            result.counts["unverifiable_target"] += 1
            result.quarantined.append(
                {
                    "id": describe(row),
                    "reason": "unverifiable_target: "
                    + repr(row["reward_model"]["ground_truth"])[:120],
                }
            )
            continue
        if (
            answer_style(row) == "minerva"
            and graded_answer_field(row["reward_model"]["ground_truth"]) is None
        ):
            result.counts["equation_target"] += 1
            result.quarantined.append(
                {
                    "id": describe(row),
                    "reason": "equation_target: "
                    + repr(row["reward_model"]["ground_truth"])[:120],
                }
            )
            continue
        reason = guard.reason(problem)
        if reason:
            drop(reason, row)
            continue
        tokens = budget.tokens(problem)
        result.prompt_tokens.append(tokens)
        if tokens > budget.budget:
            drop("prompt_over_budget", row)
            continue
        survivors.append(Candidate(row, problem, identity(row), tokens))

    problems = [candidate.problem for candidate in survivors]
    ownership = {}
    for name, owned in (owners or {}).items():
        owned_ids = [owner_id for owner_id, _ in owned]
        owned_problems = [problem for _, problem in owned]
        template = template_shingles([problems, owned_problems])
        index = ProblemOverlapIndex(owned_problems, template=template)
        ownership[name] = index.provenance()
        yielded = []
        for candidate in survivors:
            match = index.matches(candidate.problem)
            if match:
                result.counts[f"owned_by:{name}:{match[0].matcher}"] += 1
                drop(
                    f"owned_by:{name}",
                    candidate.row,
                    {
                        "owner": owned_ids[match[0].reference],
                        "matcher": match[0].matcher,
                        "containment": round(match[0].containment, 4),
                    },
                )
            else:
                yielded.append(candidate)
        survivors = yielded
    result.near_duplicates["ownership"] = ownership

    survivors.sort(key=lambda candidate: candidate.identity)
    groups: dict[object, list[Candidate]] = {}
    for candidate in survivors:
        key = skeleton_key(candidate.problem) or ("unkeyed", candidate.identity)
        groups.setdefault(key, []).append(candidate)
    representatives, collapsed, conflicts = [], [], []
    for members in groups.values():
        if len(members) == 1:
            representatives.append(members[0])
        elif _targets_agree([member.row for member in members]):
            representatives.append(members[0])
            result.counts["near_duplicate:same_text"] += len(members) - 1
            collapsed.extend(
                {"kept": members[0].identity, "dropped": member.identity,
                 "matcher": "skeleton"}
                for member in members[1:]
            )
        else:
            result.counts["conflicting_same_text_quarantine"] += len(members)
            conflicts.append(
                {
                    "ids": [describe(member.row) for member in members],
                    "targets": [
                        member.row["reward_model"]["ground_truth"]
                        for member in members
                    ],
                }
            )
    representatives.sort(key=lambda candidate: candidate.identity)

    problems = [candidate.problem for candidate in representatives]
    index = ProblemOverlapIndex(problems, template=template_shingles([problems]))
    absorbed = [False] * len(representatives)
    distinct_targets = 0
    for position, keeper in enumerate(representatives):
        if absorbed[position]:
            continue
        result.kept.append(keeper)
        for match in index.matches(keeper.problem, exclude=position):
            other = match.reference
            if other < position or absorbed[other]:
                continue
            if _targets_agree([keeper.row, representatives[other].row]):
                absorbed[other] = True
                result.counts["near_duplicate:shingle"] += 1
                collapsed.append(
                    {"kept": keeper.identity,
                     "dropped": representatives[other].identity,
                     "matcher": match.matcher,
                     "containment": round(match.containment, 4)}
                )
            else:
                distinct_targets += 1
    result.near_duplicates.update(
        {
            "within_pool_index": index.provenance(),
            "same_text_rule": "equal skeleton_key; agreeing targets keep the "
            "lowest identity, disagreeing targets quarantine the group",
            "shingle_rule": "collapse into an earlier kept row matched "
            "directly (no chaining) with an agreeing target; different "
            "targets are kept as distinct questions",
            "shingle_pairs_kept_with_distinct_targets": distinct_targets,
            "collapsed": collapsed,
            "conflicting_same_text_groups": conflicts,
        }
    )
    result.counts["kept"] = len(result.kept)
    return result


def prompt_token_audit(tokens: Sequence[int], budget: int) -> dict:
    """Length distribution of the problems that reached the budget screen."""
    if not tokens:
        return {"measured": 0}
    ordered = sorted(tokens)

    def quantile(q: float) -> int:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    over = sum(1 for value in ordered if value > budget)
    return {
        "measured": len(ordered),
        "over_budget": over,
        "over_budget_fraction": over / len(ordered),
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "p99": quantile(0.99),
        "max": ordered[-1],
    }

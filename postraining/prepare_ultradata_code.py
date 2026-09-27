"""UltraData-Code L3/py: pinned mirror, then disjoint verified SFT and RL pools.

openbmb/UltraData-Code L3 turns algorithmic GitHub files into exercises with
four generated fields -- ``task``, ``analysis``, ``solution`` and ``test`` --
and the dataset card calls the tests "test candidates": nothing upstream
executed them. Measured on one 16,949-row group, only 60.6% of reference
solutions pass this repository's Python reward policy at all, so rows are
admitted here only after this repository's own verifier agrees with them:

* the reference ``solution`` passes ``python_candidate_allowed`` -- the exact
  AST policy the RL reward applies to policy samples -- so SFT never teaches a
  program shape the reward would reject;
* the whole test module, split into its top-level statements, passes in the
  bwrap sandbox twice (determinism) within ``MAX_REFERENCE_WALL_SECONDS``;
* a stub defining every solution function as ``return None`` *fails* the
  same tests, so a suite that constrains nothing cannot certify a row.

A passing reference certifies that solution and tests agree, not that either
is right; whole-module agreement is kept rather than per-assert salvage
because the data is abundant and a partial pass is evidence the pair
disagrees somewhere.

Tasks are screened before any sandbox time is spent: evaluation containment
(KodCode, MBPP, HumanEval, plus the GSM8K/DeepMind/AIME math index), fence
literals, meta-references to the snippet the exercise was synthesised from
("the snippet", "the original code") -- a think trace that cites code the
prompt never shows teaches the model to hallucinate context -- and entry
points the task never names, since a prompt that does not say which function
to write is not a well-posed exercise. Exact and MinHash near-duplicate
tasks collapse to one representative before sandboxing.

Word-shingle similarity does not see paraphrase: in a 3,726-row verified
sample, 34% of rows share their entry-point name with another row, and those
pairs are the same problem reworded (``factorial`` 30 times, ``max_profit``
17) at a median estimated Jaccard of only 0.056. The entry-point name set is
therefore the *problem identity*: an identity belongs to exactly one pool,
the RL pool holds one row per identity and the SFT pool at most
``SFT_PER_IDENTITY``, so RL prompts are not paraphrases of SFT documents and
neither pool repeats a classic exercise hundreds of times.

Outputs (all immutable, refused if present):

* ``<prefix>-sft-pool.parquet``: verified exercises for the
  ``ultradata_code_l3`` adapter of ``prepare_sft_corpus``, which renders
  them in the canonical answer-fence schema;
* ``<prefix>-rl.parquet``: RL rows in the MBPP row schema, whose prompt is
  the bare task plus one or two self-contained example assertions, graded on
  every kept top-level test statement so shown examples cannot be hardcoded;
* ``<prefix>.manifest.json``: shard hashes, every drop count, the pass-rate
  audit and both output hashes.

CPU only (tokenizer, AST, bwrap); no model runs, so no mlq submission.

    .venv/bin/python -m postraining.prepare_ultradata_code download --shards 19
    .venv/bin/python -m postraining.prepare_ultradata_code targets
    .venv/bin/python -m postraining.prepare_ultradata_code build \\
        --sample-rows 2681998 --rl-rows 20000 --sft-rows 120500 --workers 14 \\
        --output-prefix postraining/data/ultradata-code-l3-v3
"""

from __future__ import annotations

import argparse
import ast
import builtins
import gc
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request
import warnings
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from postraining.core import GPT2BPETokenizer
from postraining.math_prompt import strip_math_prompt_framing
from postraining.prepare_sft_corpus import (
    CODE_CONTAINMENT_TARGETS,
    CONTAINMENT_MIN_OVERLAP,
    CONTAINMENT_TARGETS,
    build_containment_index,
    contains_reference_problem,
    read_reference_texts,
)
from postraining.prepare_sft_traces import (
    DECONTAMINATION_TARGETS,
    FENCE_STRINGS,
    build_decontamination_index,
    contaminated,
)
from postraining.vapo.code_reward import (
    PYTHON_REWARD_SCHEMA,
    SAFE_IMPORT_ROOTS,
    python_candidate_allowed,
    python_test_result,
)

DATASET_ID = "openbmb/UltraData-Code"
DATASET_REVISION = "85182d829f2ce7ea07cca72ebfc509deea1d9f5f"
PUBLISHED_SHARDS = 147
SHARD_PATH = (
    "data/UltraData-Code-L3/py/"
    "UltraData-Code-L3-py-part-{index:05d}-of-00147.parquet"
)
MIRROR = Path("postraining/data/instruction_corpus_shards/ultradata_code_hf")
DOWNLOAD_MANIFEST = MIRROR / "l3_py.download.manifest.json"
DOWNLOAD_SCHEMA = "ultradata_code_l3_py_download/v1"
# v2: RL rows declare verification_info.entry_points (Python reward v6).
BUILD_SCHEMA = "ultradata_code_l3_verified_pools/v2"
SOURCE_NAME = "ultradata_code_l3"
PROVENANCE = "L3/py"
DOWNLOAD_ATTEMPTS = 5
# Hashed into the build manifest: the screening, sandbox and reward rules.
SOURCE_FILES = (
    Path("postraining/prepare_ultradata_code.py"),
    Path("postraining/prepare_sft_corpus.py"),
    Path("postraining/prepare_sft_traces.py"),
    Path("postraining/math_prompt.py"),
    Path("postraining/vapo/code_reward.py"),
)

# Evaluation targets for code that ``prepare_sft_corpus`` did not already
# carry. Built by the ``targets`` subcommand from pinned sources so the
# containment rule can read them like the KodCode target.
MBPP_SOURCE = Path("postraining/data/vapo_broad_v7_bare_mbpp_source.jsonl")
MBPP_SOURCE_SHA256 = (
    "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f"
)
MBPP_TARGET = Path("postraining/data/mbpp-problems.parquet")
HUMANEVAL_REPO = "openai/openai_humaneval"
HUMANEVAL_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
HUMANEVAL_FILE = "openai_humaneval/test-00000-of-00001.parquet"
HUMANEVAL_SHA256 = (
    "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6"
)
HUMANEVAL_TARGET = Path("postraining/data/humaneval-problems.parquet")
TARGET_SCHEMA = "code_eval_decontamination_target/v1"

# A test module must make at least this many distinct top-level assertions.
# RL shows up to two of them; an RL prompt is also only admitted when the
# unshown statements call an entry point on arguments no example revealed
# (see ``hidden_calls_discriminate``), so hardcoding the shown outputs fails.
MIN_TOP_LEVEL_ASSERTS = 3
MAX_EXAMPLES = 2
MAX_EXAMPLE_CHARS = 160
# encode_prompt keeps BOS plus the last ``prompt_tokens - 1`` tokens, and the
# production RL split is 256 prompt + 768 response.
RL_PROMPT_TOKENS = 256
# Rows per problem identity (sorted entry-point names; see module docstring).
# One in RL, so every RL prompt is a distinct problem; three in SFT, which
# keeps real variants (recursive vs iterative factorial) without letting a
# classic exercise act like a second epoch.
RL_PER_IDENTITY = 1
VERIFIER_INFRA_ATTEMPTS = 3
# Paraphrases under a different function name survive both the 0.5 dedupe
# and the name identity (``cigar_party`` vs ``party_success``). An SFT row
# whose task reaches this estimated Jaccard with any RL task is dropped;
# 63 bands of 2 rows propose a pair at 0.3 with probability 0.997.
CROSS_POOL_JACCARD = 0.3
CROSS_POOL_BANDS = 63
SFT_PER_IDENTITY = 3
# The verifier's wall limit is 3 s under RL-time contention; a reference
# that needs a third of it here is too close to be a stable positive.
MAX_REFERENCE_WALL_SECONDS = 1.0

# Tests are verifier code, not policy output, so they may use deterministic
# stdlib modules the candidate cannot. Nondeterministic or environment-bound
# modules (random, time, os, io, sys beyond maxsize) are excluded: a test that
# passes only by luck is not a label.
TEST_IMPORT_ROOTS = (SAFE_IMPORT_ROOTS - {"sys"}) | {
    "copy",
    "decimal",
    "fractions",
    "functools",
    "statistics",
    "string",
    "typing",
}

# The analysis was generated while looking at a source file the exercise
# never shows. Measured: 11.9% of analyses cite it ("the snippet" 2,252,
# "original code" 852, "reference solution" 256 in 16,949 rows).
META_REFERENCE = re.compile(
    r"\b(?:the snippet|code snippet|(?:reference|provided|original|given|source)"
    r" (?:solution|implementation|code|function))\b",
    re.I,
)

BUILTIN_NAMES = frozenset(dir(builtins))
SHINGLE = 5
# 42 bands of 3 rows: a pair at Jaccard J becomes a candidate with
# probability 1 - (1 - J**3)**42 -- 0.996 at the 0.5 threshold, 0.68 at 0.3.
# (16 bands of 4 proposed only 64% of pairs at 0.5.)
MINHASH_PERMUTATIONS = 126
LSH_BANDS = 42
NEAR_DUPLICATE_JACCARD = 0.5
BUCKET_COMPARISONS = 32
_MINHASH_RNG = np.random.default_rng(0x5EED_C0DE)
_MINHASH_A = _MINHASH_RNG.integers(1, 2**63, MINHASH_PERMUTATIONS, dtype=np.uint64) | 1
_MINHASH_B = _MINHASH_RNG.integers(0, 2**63, MINHASH_PERMUTATIONS, dtype=np.uint64)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def atomic_parquet(table: pa.Table, path: Path) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------


def spread_shard_indices(count: int, published: int = PUBLISHED_SHARDS) -> list[int]:
    """``count`` 1-based shard indices evenly spaced over the whole release.

    The shard order is not documented as random, so a prefix could be a
    biased slice of whatever the upstream pipeline sorted by. Even spacing
    samples the full ordering whatever it is, and always includes both ends.
    """

    if not 1 <= count <= published:
        raise ValueError(f"count must be in 1..{published}")
    if count == 1:
        return [1]
    indices = [1 + round(k * (published - 1) / (count - 1)) for k in range(count)]
    if len(set(indices)) != count:
        raise AssertionError("spread produced duplicate shard indices")
    return indices


def _http_fetch(url: str, target: Path, expected_sha256: str) -> Path:
    """Stream ``url`` to ``target``; rename into place only on a digest match.

    The token, when present, is read from the environment and sent as a
    header only; it is never written anywhere.
    """

    if target.exists():
        if file_sha256(target) != expected_sha256:
            raise SystemExit(f"{target} exists with a different sha256")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    staging = target.with_name(target.name + f".{os.getpid()}.part")
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(
                request, timeout=120
            ) as response, staging.open("wb") as out:
                while chunk := response.read(8 * 1024 * 1024):
                    digest.update(chunk)
                    out.write(chunk)
            break
        except OSError as error:  # URLError and socket timeouts included
            staging.unlink(missing_ok=True)
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            print(f"{target.name}: attempt {attempt} failed ({error}); retrying",
                  file=sys.stderr, flush=True)
            time.sleep(10 * attempt)
    if digest.hexdigest() != expected_sha256:
        staging.unlink()
        raise SystemExit(
            f"{url}: sha256 {digest.hexdigest()} != published {expected_sha256}"
        )
    staging.replace(target)
    return target


def download(args: argparse.Namespace) -> None:
    """Mirror evenly spread L3/py shards at the pinned revision.

    Plain HTTP against the pinned ``resolve`` URL in parallel streams: the
    hub client's chunked (xet) transfer stalled on this host.
    """

    from huggingface_hub import HfApi

    if DOWNLOAD_MANIFEST.exists():
        raise SystemExit(f"{DOWNLOAD_MANIFEST} exists; the mirror is immutable")
    indices = spread_shard_indices(args.shards)
    info = HfApi().dataset_info(
        DATASET_ID, revision=DATASET_REVISION, files_metadata=True
    )
    if info.sha != DATASET_REVISION:
        raise SystemExit(f"resolved revision {info.sha} != pinned {DATASET_REVISION}")
    published = {
        sibling.rfilename: sibling
        for sibling in info.siblings
        if sibling.rfilename.startswith("data/UltraData-Code-L3/py/")
    }
    if len(published) != PUBLISHED_SHARDS:
        raise SystemExit(
            f"expected {PUBLISHED_SHARDS} L3/py shards, found {len(published)}"
        )
    plan = []
    for index in indices:
        relative = SHARD_PATH.format(index=index)
        sibling = published[relative]
        if not (sibling.lfs and sibling.lfs.sha256):
            raise SystemExit(f"{relative} has no LFS sha256 at the pinned revision")
        plan.append((index, relative, sibling.lfs.sha256))

    def fetch(item):
        index, relative, expected = item
        local = _http_fetch(
            f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
            f"{DATASET_REVISION}/{relative}",
            MIRROR / relative,
            expected,
        )
        metadata = pq.ParquetFile(local).metadata
        print(f"{relative}: {metadata.num_rows} rows, sha256 ok",
              file=sys.stderr, flush=True)
        return {
            "index": index,
            "path": relative,
            "bytes": local.stat().st_size,
            "sha256": expected,
            "rows": metadata.num_rows,
            "row_groups": metadata.num_row_groups,
        }

    with ThreadPoolExecutor(max_workers=args.connections) as pool:
        files = list(pool.map(fetch, plan))
    manifest = {
        "schema": DOWNLOAD_SCHEMA,
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "subset": "UltraData-Code-L3/py",
        "published_shards": PUBLISHED_SHARDS,
        "selection": (
            f"{args.shards} shards evenly spaced over 1..{PUBLISHED_SHARDS} "
            "(1 + round(k * 146 / (n - 1))); upstream shard order is not "
            "documented as random"
        ),
        "license": (
            "apache-2.0 plus each source repository's license (dataset card); "
            "no unchanged redistribution"
        ),
        "files": files,
        "rows": sum(entry["rows"] for entry in files),
        "bytes": sum(entry["bytes"] for entry in files),
    }
    atomic_write_bytes(
        DOWNLOAD_MANIFEST, (json.dumps(manifest, indent=2) + "\n").encode()
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


# --------------------------------------------------------------------------
# evaluation targets
# --------------------------------------------------------------------------


def build_targets(_args: argparse.Namespace) -> None:
    """Materialise the MBPP and HumanEval containment targets.

    MBPP is covered whole (974 tasks): 601-974 are the RL ``mbpp`` source and
    the rest are its evaluation splits. HumanEval's 164 prompts are fetched
    at a pinned revision for decontamination only.
    """

    for path in (MBPP_TARGET, HUMANEVAL_TARGET):
        for output in (path, path.with_suffix(".manifest.json")):
            if output.exists():
                raise SystemExit(f"{output} exists; targets are immutable")
    if file_sha256(MBPP_SOURCE) != MBPP_SOURCE_SHA256:
        raise SystemExit(f"{MBPP_SOURCE} does not match the pinned MBPP bytes")
    mbpp = [json.loads(line) for line in MBPP_SOURCE.read_text().splitlines()]
    if len(mbpp) != 974:
        raise SystemExit(f"expected 974 MBPP tasks, found {len(mbpp)}")
    mbpp_rows = [
        {"task_id": int(row["task_id"]), "problem": str(row["text"]).strip()}
        for row in mbpp
    ]
    atomic_parquet(pa.Table.from_pylist(mbpp_rows), MBPP_TARGET)
    staging = HUMANEVAL_TARGET.with_name(HUMANEVAL_TARGET.name + ".source")
    _http_fetch(
        f"https://huggingface.co/datasets/{HUMANEVAL_REPO}/resolve/"
        f"{HUMANEVAL_REVISION}/{HUMANEVAL_FILE}",
        staging,
        HUMANEVAL_SHA256,
    )
    humaneval = pq.read_table(staging).to_pylist()
    staging.unlink()
    if len(humaneval) != 164:
        raise SystemExit(f"expected 164 HumanEval tasks, found {len(humaneval)}")
    humaneval_rows = [
        {"task_id": str(row["task_id"]), "problem": str(row["prompt"])}
        for row in humaneval
    ]
    atomic_parquet(pa.Table.from_pylist(humaneval_rows), HUMANEVAL_TARGET)
    for path, payload in (
        (
            MBPP_TARGET,
            {
                "dataset": "google-research/mbpp (mbpp.jsonl, all 974 tasks)",
                "source": str(MBPP_SOURCE),
                "source_sha256": MBPP_SOURCE_SHA256,
                "purpose": (
                    "decontamination target for code SFT/RL rows; MBPP 601-974 "
                    "is an RL source and the rest are its evaluation splits"
                ),
                "problems": len(mbpp_rows),
            },
        ),
        (
            HUMANEVAL_TARGET,
            {
                "dataset": HUMANEVAL_REPO,
                "revision": HUMANEVAL_REVISION,
                "source_file": HUMANEVAL_FILE,
                "source_sha256": HUMANEVAL_SHA256,
                "purpose": "decontamination target for code SFT/RL rows",
                "problems": len(humaneval_rows),
            },
        ),
    ):
        manifest = {
            "schema": TARGET_SCHEMA,
            **payload,
            "column": "problem",
            "output_sha256": file_sha256(path),
        }
        atomic_write_bytes(
            path.with_suffix(".manifest.json"),
            (json.dumps(manifest, indent=2) + "\n").encode(),
        )
        print(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------
# static screening (pure; runs in worker processes)
# --------------------------------------------------------------------------


def sample_key(seed: int, uuid: str) -> int:
    """Order-independent 64-bit key: sampling and ordering both derive from it."""

    digest = hashlib.blake2b(f"{seed}:{uuid}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def normalized_task(text: str) -> str:
    return " ".join(text.split()).lower()


def sft_identity(problem: str) -> str:
    """``prepare_sft_corpus``'s dedupe key, so the adapter drops nothing new."""

    return problem.strip().lower()[:160]


def task_shingles(text: str) -> np.ndarray:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    grams = {
        " ".join(words[i : i + SHINGLE])
        for i in range(max(1, len(words) - SHINGLE + 1))
    }
    return np.array(
        [
            int.from_bytes(
                hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big"
            )
            for gram in sorted(grams)
        ],
        dtype=np.uint64,
    )


def minhash(text: str) -> np.ndarray:
    """MINHASH_PERMUTATIONS min-hashes of the task's word 5-grams (uint32).

    Multiply-add in uint64 wraps modulo 2**64; the high 32 bits of an odd
    multiplier's product are a sound universal hash for this purpose.
    """

    shingles = task_shingles(text)
    with np.errstate(over="ignore"):
        mixed = shingles[None, :] * _MINHASH_A[:, None] + _MINHASH_B[:, None]
    return (mixed >> np.uint64(32)).astype(np.uint32).min(axis=1)


_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def module_bindings(tree: ast.Module, *, imports: bool = True) -> set[str]:
    """Names bound at module scope; nested function and class bodies excluded.

    Only module scope matters: a test statement ``result = f(x)`` does not
    collide with a local ``result`` inside the solution's function.
    """

    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            return
        if isinstance(node, _SCOPES):
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if imports:
                names.update(
                    (alias.asname or alias.name).split(".", 1)[0]
                    for alias in node.names
                )
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in tree.body:
        visit(statement)
    return names


def _statement_source(source: str, node: ast.stmt) -> str:
    """Exact source of one top-level statement, decorators included.

    ``get_source_segment`` starts a decorated ``def``/``class`` at its
    ``def`` line, which would silently drop ``@lru_cache`` or ``@dataclass``
    from a test helper.
    """

    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ValueError("statement has no source segment")
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        # Split exactly as the tokenizer counts lines; ``str.splitlines``
        # also breaks on form feeds and other separators ast ignores.
        lines = io.StringIO(source, newline="").readlines()
        # Top-level, so the decorator block starts at column 0.
        first = min(decorator.lineno for decorator in decorators) - 1
        segment = "".join(lines[first : node.lineno - 1]) + segment
    return segment


def all_bindings(tree: ast.AST) -> set[str]:
    """Every name the module binds anywhere: assignments, parameters,
    definitions, imports, handlers and match captures, at any depth."""

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                (alias.asname or alias.name).split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names


def entry_calls(node: ast.AST, entry_points: set[str]) -> set[str]:
    """Structural identities of every entry-point call under ``node``."""

    return {
        ast.dump(child)
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id in entry_points
    }


def hidden_calls_discriminate(
    body: list[ast.stmt], shown: list[ast.stmt], entry_points: set[str]
) -> bool:
    """Whether the unshown statements call an entry point in a way no shown
    example does -- otherwise returning the shown outputs passes every test."""

    shown_ids = {id(node) for node in shown}
    revealed = set().union(*(entry_calls(node, entry_points) for node in shown))
    hidden = set().union(
        *(entry_calls(node, entry_points) for node in body if id(node) not in shown_ids)
    )
    return bool(hidden - revealed)


def example_assertion(
    node: ast.stmt, source: str, entry_points: set[str]
) -> str | None:
    """A top-level assert a reader can run with only the entry points.

    Shown in the RL prompt, so it must be self-contained (no names from
    earlier test statements), actually call an entry point, and be one short
    line.
    """

    if not isinstance(node, ast.Assert) or node.msg is not None:
        return None
    names = {
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    }
    if not names & entry_points or names - entry_points - BUILTIN_NAMES:
        return None
    calls_entry = any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id in entry_points
        for child in ast.walk(node)
    )
    if not calls_entry:
        return None
    text = _statement_source(source, node)
    if "\n" in text or len(text) > MAX_EXAMPLE_CHARS:
        return None
    return text


def render_rl_prompt(task: str, examples: list[str]) -> str:
    label = "Example:" if len(examples) == 1 else "Examples:"
    return f"{task}\n\n{label}\n" + "\n".join(examples)


def stub_module(solution: str) -> str:
    """The solution with every top-level function body replaced by
    ``return None``.

    Imports, classes and constants stay, so a test that also reads them
    fails the stub on the functions' behaviour rather than on a NameError.
    """

    tree = ast.parse(solution)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = [ast.Return(value=ast.Constant(value=None))]
    return ast.unparse(ast.fix_missing_locations(tree))


# Screening state, set in the parent before the pool forks (copy-on-write).
_TOKENIZER: GPT2BPETokenizer | None = None
_MATH_EXACT: set[str] | None = None
_MATH_NGRAMS: set[tuple[str, ...]] | None = None
_REF_EXACT: set[str] | None = None
_REF_GRAM_TO_IDS: dict | None = None
_REF_SIZES: list[int] | None = None


def screen_exercise(row: dict) -> tuple[dict | None, str]:
    """Every order-independent, sandbox-free admission check for one row."""

    fields = {}
    for name in ("uuid", "task", "analysis", "solution", "test"):
        value = row.get(name)
        if not isinstance(value, str) or not value.strip():
            return None, f"empty_{name}"
        fields[name] = value
    solution = fields["solution"].strip()
    analysis = fields["analysis"].strip()
    test_source = fields["test"]
    try:
        task, removed = strip_math_prompt_framing(fields["task"])
    except ValueError:
        return None, "uncanonicalizable_task"
    if removed:
        # A registered instruction inside a synthesized exercise would be
        # silently rewritten; drop rather than reshape.
        return None, "task_carries_instruction"
    if any(
        fence in text
        for text in (task, analysis, solution, test_source)
        for fence in FENCE_STRINGS
    ):
        return None, "fence_literal"
    if "```" in solution:
        return None, "solution_contains_fence"
    if META_REFERENCE.search(task) or META_REFERENCE.search(analysis):
        return None, "meta_reference"
    if not python_candidate_allowed(solution):
        return None, "solution_policy_rejected"
    solution_tree = ast.parse(solution)
    functions = {
        node.name
        for node in solution_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # Imports are excluded: a test re-importing ``math`` is harmless, while a
    # test rebinding a function or global the solution defines is vacuous.
    solution_names = module_bindings(solution_tree, imports=False)
    solution_module_names = module_bindings(solution_tree)
    try:
        test_tree = ast.parse(test_source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None, "test_syntax_error"
    if not test_tree.body:
        return None, "empty_test"
    for node in ast.walk(test_tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots = {(node.module or "").split(".", 1)[0]} if not node.level else {""}
        else:
            continue
        if not roots <= TEST_IMPORT_ROOTS:
            return None, "test_import_unsupported"
    if module_bindings(test_tree) & solution_names:
        # A test that redefines the function it checks passes vacuously.
        return None, "test_rebinds_solution_name"
    used = {
        node.id for node in ast.walk(test_tree) if isinstance(node, ast.Name)
    }
    entry_points = functions & used
    if not entry_points:
        return None, "tests_call_no_solution_function"
    unresolved = used - entry_points - BUILTIN_NAMES - all_bindings(test_tree)
    if unresolved & solution_module_names:
        # The test reads a solution import, class or constant the task never
        # asks for: a correct answer without it scores 0, and the candidate
        # may define it however it likes.
        return None, "test_reads_solution_name"
    if unresolved:
        return None, "test_unbound_name"
    if any(
        not re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", task)
        for name in entry_points
    ):
        return None, "entry_point_not_in_task"
    # Distinct by structure: a repeated statement is not a second check.
    asserts = len(
        {ast.dump(node) for node in test_tree.body if isinstance(node, ast.Assert)}
    )
    if asserts < MIN_TOP_LEVEL_ASSERTS:
        return None, "too_few_top_level_asserts"
    try:
        tests = [_statement_source(test_source, node) for node in test_tree.body]
    except ValueError:
        return None, "test_source_segment"
    if contaminated(task, _MATH_EXACT, _MATH_NGRAMS):
        return None, "contaminated_math_index"
    if contains_reference_problem(
        task, _REF_EXACT, _REF_GRAM_TO_IDS, _REF_SIZES, CONTAINMENT_MIN_OVERLAP
    ):
        return None, "contains_reference_problem"

    examples: list[tuple[str, ast.stmt]] = []
    for node in test_tree.body:
        example = example_assertion(node, test_source, entry_points)
        if example is not None and example not in (text for text, _ in examples):
            examples.append((example, node))
        if len(examples) == MAX_EXAMPLES:
            break
    rl_prompt = None
    rl_prompt_tokens = None
    shown = 0
    # Two examples when they fit the prompt budget, else one.
    for count in range(len(examples), 0, -1):
        if not hidden_calls_discriminate(
            test_tree.body, [node for _, node in examples[:count]], entry_points
        ):
            continue
        prompt = render_rl_prompt(task, [text for text, _ in examples[:count]])
        try:
            canonical, prompt_removed = strip_math_prompt_framing(prompt)
        except ValueError:
            continue
        if prompt_removed or canonical != prompt:
            continue
        tokens = len(_TOKENIZER.encode(prompt))
        if tokens + 1 <= RL_PROMPT_TOKENS:
            rl_prompt, rl_prompt_tokens, shown = prompt, tokens, count
            break
    return (
        {
            "uuid": fields["uuid"],
            "task": task,
            "analysis": analysis,
            "solution": solution,
            "tests": tests,
            "entry_points": sorted(entry_points),
            "top_level_asserts": asserts,
            "rl_prompt": rl_prompt,
            "rl_prompt_tokens": rl_prompt_tokens,
            "shown_examples": shown,
            "minhash": minhash(task),
        },
        "",
    )


BODY_FIELDS = ("analysis", "solution", "tests")


def pack_body(payload: dict) -> bytes:
    """Move the bulky text fields into one zlib blob.

    The parent holds every screened row until the seeded sort and dedupe,
    about a million at full rate; compressed, that fits beside other jobs.
    """

    body = [payload.pop(name) for name in BODY_FIELDS]
    return zlib.compress(json.dumps(body).encode(), 6)


def unpack_body(item: dict) -> dict:
    """``item`` with its text fields restored (a new dict; ``item`` unchanged)."""

    if "_body" not in item:
        return item
    full = {key: value for key, value in item.items() if key != "_body"}
    full.update(zip(BODY_FIELDS, json.loads(zlib.decompress(item["_body"]))))
    return full


def _screen_chunk(chunk: list[dict]) -> list[tuple[dict | None, str, int]]:
    # Upstream sources are arbitrary Python; their invalid escape sequences
    # are not this build's concern and would flood the log.
    warnings.simplefilter("ignore", SyntaxWarning)
    results = []
    for row in chunk:
        try:
            payload, reason = screen_exercise(row)
        except RecursionError:
            # Machine-generated expressions nested past the interpreter's
            # recursion limit; the AST checks cannot certify them.
            payload, reason = None, "ast_too_deep"
        except Exception as error:  # noqa: BLE001 -- counted, never admitted
            payload, reason = None, f"screen_error_{type(error).__name__}"
        if payload is not None:
            payload["_key"] = row["_key"]
            payload["_shard"] = row["_shard"]
            payload["_body"] = pack_body(payload)
        results.append((payload, reason, row["_shard"]))
    return results


# --------------------------------------------------------------------------
# near-duplicate clustering
# --------------------------------------------------------------------------


def near_duplicate_representatives(
    signatures: np.ndarray, threshold: float = NEAR_DUPLICATE_JACCARD
) -> tuple[np.ndarray, int]:
    """Keep the first row (in input order) of every near-duplicate cluster.

    Banded LSH proposes pairs, and a pair is joined only when its estimated
    Jaccard (signature agreement) reaches ``threshold``; union-find turns
    pairs into clusters, so a later row joined transitively to an earlier
    one is dropped as well. Returns (keep mask, pairs joined).
    """

    count, permutations = signatures.shape
    rows_per_band = permutations // LSH_BANDS
    if rows_per_band * LSH_BANDS != permutations:
        raise ValueError(f"{permutations} permutations do not split into {LSH_BANDS} bands")
    parent = np.arange(count)

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    joined = 0
    for band in range(LSH_BANDS):
        block = signatures[:, band * rows_per_band : (band + 1) * rows_per_band]
        buckets: dict[bytes, list[int]] = {}
        for index in range(count):
            members = buckets.setdefault(block[index].tobytes(), [])
            if members:
                # Bounded so one boilerplate-heavy bucket cannot go quadratic.
                earlier = members[:BUCKET_COMPARISONS]
                agreement = (signatures[earlier] == signatures[index]).mean(axis=1)
                for other in np.asarray(earlier)[agreement >= threshold]:
                    a, b = find(int(other)), find(index)
                    if a != b:
                        # The smaller root stays the root, so the earliest
                        # row of every cluster is its representative.
                        parent[max(a, b)] = min(a, b)
                        joined += 1
            members.append(index)
    keep = np.array([find(index) == index for index in range(count)])
    return keep, joined


# --------------------------------------------------------------------------
# sandbox verification
# --------------------------------------------------------------------------


def verify_exercise(item: dict) -> dict:
    """Reference twice (determinism, wall time), stub once (vacuity)."""

    info = {
        "schema": PYTHON_REWARD_SCHEMA,
        "entry_points": item["entry_points"],
        "test_setup": [],
        "tests": item["tests"],
    }
    walls = []
    results = []
    for attempt in range(2):
        start = time.perf_counter()
        result = python_test_result(item["solution"], info)
        walls.append(time.perf_counter() - start)
        results.append(result)
        if result != "pass":
            break
        if attempt == 0:
            try:
                stub_source = stub_module(item["solution"])
            except RecursionError:
                return {"uuid": item["uuid"], "verdict": "stub_unavailable",
                        "first": result, "walls": walls}
            stub = python_test_result(stub_source, info)
            if stub == "pass":
                return {"uuid": item["uuid"], "verdict": "stub_passes",
                        "first": result, "walls": walls}
    if results[0] != "pass":
        verdict = f"reference_{results[0]}"
    elif results[1] == "timeout":
        # Load-bound rather than nondeterministic; counted apart so the
        # audit does not confuse the two.
        verdict = "reference_rerun_timeout"
    elif results[1] != "pass":
        verdict = "reference_nondeterministic"
    elif max(walls) > MAX_REFERENCE_WALL_SECONDS:
        verdict = "reference_too_slow"
    else:
        verdict = "verified"
    return {"uuid": item["uuid"], "verdict": verdict, "first": results[0],
            "walls": walls}


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def problem_identity(item: dict) -> tuple[str, ...]:
    return tuple(sorted(name.lower() for name in item["entry_points"]))


class PoolAssigner:
    """Order-dependent, online RL/SFT assignment of verified rows.

    The first verified row of a problem identity decides its pool: RL while
    RL has room and the row has an RL prompt, otherwise SFT. Later rows of
    that identity follow it or are dropped at the per-identity cap -- never
    moved to the other pool -- so the pools share no identity. Online, so
    the sandbox can stop the moment both pools are full.
    """

    def __init__(self, rl_rows: int, sft_rows: int) -> None:
        self.rl_rows = rl_rows
        self.sft_rows = sft_rows
        self.rl: list[dict] = []
        self.sft: list[dict] = []
        self.pool_of: dict[tuple[str, ...], str] = {}
        self.taken: Counter = Counter()
        self.dropped: Counter = Counter()

    def offer(self, item: dict) -> str:
        identity = problem_identity(item)
        pool = self.pool_of.get(identity)
        if pool is None:
            pool = (
                "rl"
                if len(self.rl) < self.rl_rows and item["rl_prompt"] is not None
                else "sft"
            )
            self.pool_of[identity] = pool
        if pool == "rl":
            if (
                self.taken[identity] >= RL_PER_IDENTITY
                or len(self.rl) >= self.rl_rows
                or item["rl_prompt"] is None
            ):
                self.dropped["identity_repeat_rl"] += 1
                return "dropped"
            self.rl.append(item)
        else:
            if self.taken[identity] >= SFT_PER_IDENTITY:
                self.dropped["identity_repeat_sft"] += 1
                return "dropped"
            if len(self.sft) >= self.sft_rows:
                self.dropped["sft_full"] += 1
                return "dropped"
            self.sft.append(item)
        self.taken[identity] += 1
        return pool

    def saturated(self, item: dict) -> bool:
        """Whether ``offer`` would drop this row whatever its verdict."""

        identity = problem_identity(item)
        pool = self.pool_of.get(identity)
        if pool == "rl":
            return (
                self.taken[identity] >= RL_PER_IDENTITY
                or len(self.rl) >= self.rl_rows
            )
        if pool == "sft":
            return (
                self.taken[identity] >= SFT_PER_IDENTITY
                or len(self.sft) >= self.sft_rows
            )
        # A new identity goes to SFT once RL is full or it has no prompt.
        return len(self.sft) >= self.sft_rows and (
            len(self.rl) >= self.rl_rows or item["rl_prompt"] is None
        )

    @property
    def full(self) -> bool:
        return len(self.rl) >= self.rl_rows and len(self.sft) >= self.sft_rows


def near_twins(
    queries: np.ndarray,
    index: np.ndarray,
    threshold: float = CROSS_POOL_JACCARD,
    bands: int = CROSS_POOL_BANDS,
) -> np.ndarray:
    """Mask of ``queries`` rows whose estimated Jaccard with some ``index``
    row reaches ``threshold`` (LSH-proposed, checked on full signatures)."""

    rows_per_band = index.shape[1] // bands
    if rows_per_band * bands != index.shape[1]:
        raise ValueError(f"{index.shape[1]} permutations do not split into {bands} bands")
    hit = np.zeros(len(queries), dtype=bool)
    for band in range(bands):
        cols = slice(band * rows_per_band, (band + 1) * rows_per_band)
        buckets: dict[bytes, list[int]] = {}
        for position, row in enumerate(index[:, cols]):
            buckets.setdefault(row.tobytes(), []).append(position)
        for position, row in enumerate(queries[:, cols]):
            if hit[position]:
                continue
            members = buckets.get(row.tobytes())
            if members and (
                (index[members] == queries[position]).mean(axis=1) >= threshold
            ).any():
                hit[position] = True
    return hit


def _verify_or_reject(item: dict) -> dict:
    """``verify_exercise`` for the pool: an unexpected verifier failure on
    one row rejects that row (counted by exception type) instead of
    discarding a whole build's screening."""

    item = unpack_body(item)
    for _ in range(VERIFIER_INFRA_ATTEMPTS - 1):
        try:
            return verify_exercise(item)
        except RuntimeError:
            # python_test_result raises only for sandbox infrastructure
            # failures (bwrap/prlimit, subprocess launch), not for the row.
            time.sleep(1.0)
    try:
        return verify_exercise(item)
    except Exception as error:  # noqa: BLE001 -- fail closed, per row
        return {"uuid": item["uuid"], "verdict": f"verifier_error_{type(error).__name__}",
                "first": "error", "walls": [0.0]}


def rl_row(item: dict) -> dict:
    return {
        "data_source": SOURCE_NAME,
        "prompt": [{"role": "user", "content": item["rl_prompt"]}],
        "ability": "coding",
        # The generic VAPO loader requires this field; the Python verifier
        # ignores it and executes verification_info instead.
        "reward_model": {"ground_truth": "ALL_TESTS_PASS", "style": "python-exec"},
        "extra_info": {
            "index": f"{SOURCE_NAME}_{item['uuid']}",
            "module": SOURCE_NAME,
            "prompt_contract": "bare",
            "shown_examples": item["shown_examples"],
            "graded_statements": len(item["tests"]),
        },
        "verification_info": {
            "schema": PYTHON_REWARD_SCHEMA,
            # Screening admits only tests that read these solution functions
            # and nothing else the solution binds.
            "entry_points": item["entry_points"],
            "test_setup": [],
            # Every kept top-level statement, shown examples included: the
            # prompt reveals at most two assertions of at least three.
            "tests": item["tests"],
        },
    }


SFT_POOL_SCHEMA = pa.schema(
    [
        ("uuid", pa.string()),
        ("problem", pa.string()),
        ("analysis", pa.string()),
        ("solution", pa.string()),
        ("tests", pa.list_(pa.string())),
        ("entry_points", pa.list_(pa.string())),
        ("provenance", pa.string()),
        ("shard", pa.int32()),
    ]
)


def percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "mean": round(sum(ordered) / n, 1),
        "p10": ordered[n // 10],
        "p50": ordered[n // 2],
        "p90": ordered[int(0.9 * n)],
        "p99": ordered[min(n - 1, int(0.99 * n))],
        "max": ordered[-1],
    }


def load_code_references() -> tuple[list[str], dict[str, int]]:
    """Reference problems for containment: the SFT tool's global targets
    (KodCode, GSM8K RL prompts) plus its code targets (MBPP, HumanEval)."""

    texts: list[str] = []
    counts: dict[str, int] = {}
    for target, column in (*CONTAINMENT_TARGETS, *CODE_CONTAINMENT_TARGETS):
        if not target.exists():
            raise SystemExit(f"{target} is missing; run `targets` first")
        target_texts = read_reference_texts(target, column)
        texts.extend(target_texts)
        counts[str(target)] = len(target_texts)
    return texts, counts


def load_download_manifest() -> dict:
    if not DOWNLOAD_MANIFEST.exists():
        raise SystemExit(f"{DOWNLOAD_MANIFEST} is missing; run `download` first")
    manifest = json.loads(DOWNLOAD_MANIFEST.read_text())
    if manifest.get("schema") != DOWNLOAD_SCHEMA or manifest.get(
        "revision"
    ) != DATASET_REVISION:
        raise SystemExit("download manifest does not match the pinned revision")
    return manifest


def iter_sampled_rows(files: list[dict], seed: int, rate: float):
    columns = ["uuid", "task", "analysis", "solution", "test"]
    threshold = int(rate * 2**64)
    for entry in files:
        parquet = pq.ParquetFile(MIRROR / entry["path"])
        for group in range(parquet.metadata.num_row_groups):
            uuids = parquet.read_row_group(group, columns=["uuid"]).column(0)
            keys = [sample_key(seed, str(uuid)) for uuid in uuids.to_pylist()]
            chosen = [index for index, key in enumerate(keys) if key < threshold]
            if not chosen:
                continue
            table = parquet.read_row_group(group, columns=columns).take(chosen)
            for index, row in zip(chosen, table.to_pylist(), strict=True):
                row["_key"] = keys[index]
                row["_shard"] = entry["index"]
                yield row


def chunked(rows, size: int):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def windowed_imap(pool, function, items, window: int):
    """``pool.imap`` with at most ``window`` items in flight, in order.

    ``Pool.imap`` drains its input iterator eagerly on a feeder thread, which
    would read every sampled row of the mirror into the task queue at once.
    """

    iterator = iter(items)
    while batch := list(islice(iterator, window)):
        yield from pool.imap(function, batch)


def build(args: argparse.Namespace) -> None:
    prefix = Path(args.output_prefix)
    sft_path = prefix.with_name(prefix.name + "-sft-pool.parquet")
    rl_path = prefix.with_name(prefix.name + "-rl.parquet")
    manifest_path = prefix.with_name(prefix.name + ".manifest.json")
    for path in (sft_path, rl_path, manifest_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite immutable output {path}")
    if "." in prefix.name:
        raise SystemExit("--output-prefix must not contain a dot")
    download_manifest = load_download_manifest()
    files = download_manifest["files"]
    for entry in files:
        digest = file_sha256(MIRROR / entry["path"])
        if digest != entry["sha256"]:
            raise SystemExit(f"{entry['path']}: mirror bytes changed ({digest})")
    total_rows = download_manifest["rows"]
    rate = min(1.0, args.sample_rows / total_rows)

    global _TOKENIZER, _MATH_EXACT, _MATH_NGRAMS
    global _REF_EXACT, _REF_GRAM_TO_IDS, _REF_SIZES
    _TOKENIZER = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    _MATH_EXACT, _MATH_NGRAMS = build_decontamination_index()
    reference_texts, reference_counts = load_code_references()
    _REF_EXACT = {" ".join(text.split()).lower() for text in reference_texts}
    _REF_GRAM_TO_IDS, _REF_SIZES = build_containment_index(reference_texts)

    counts: Counter = Counter()
    per_shard: dict[int, Counter] = {entry["index"]: Counter() for entry in files}
    survivors: list[dict] = []
    started = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        for screened in windowed_imap(
            pool,
            _screen_chunk,
            chunked(iter_sampled_rows(files, args.seed, rate), args.chunk_rows),
            window=args.workers * 8,
        ):
            for payload, reason, shard in screened:
                counts["sampled"] += 1
                per_shard[shard]["sampled"] += 1
                if payload is None:
                    counts[reason] += 1
                    continue
                counts["screened"] += 1
                per_shard[shard]["screened"] += 1
                survivors.append(payload)
            if counts["sampled"] % 50_000 < args.chunk_rows:
                print(
                    f"screen: sampled {counts['sampled']} kept "
                    f"{counts['screened']} ({time.time() - started:.0f}s)",
                    file=sys.stderr,
                    flush=True,
                )

    # Verdicts, the split and the pools all key on uuid.
    if len({item["uuid"] for item in survivors}) != len(survivors):
        raise SystemExit("duplicate uuid among screened rows")
    # Seeded, order-independent order: every later first-wins decision
    # (dedupe representative, RL/SFT assignment) follows it.
    survivors.sort(key=lambda item: item["_key"])
    unique: list[dict] = []
    seen_exact: set[str] = set()
    for item in survivors:
        keys = (normalized_task(item["task"]), "prefix:" + sft_identity(item["task"]))
        if any(key in seen_exact for key in keys):
            counts["duplicate_task_exact"] += 1
            continue
        seen_exact.update(keys)
        unique.append(item)
    signatures = np.stack([item["minhash"] for item in unique])
    keep, joined = near_duplicate_representatives(signatures)
    counts["duplicate_task_near"] = int((~keep).sum())
    counts["near_duplicate_pairs_joined"] = joined
    candidates = [item for item, kept in zip(unique, keep, strict=True) if kept]
    for item in candidates:
        del item["minhash"]
    counts["deduplicated"] = len(candidates)
    print(f"dedupe: {len(survivors)} -> {len(candidates)}", file=sys.stderr,
          flush=True)
    # Workers fork next; drop what they never read and freeze the rest so
    # the collector's refcount writes do not copy every page per worker.
    del survivors, unique, signatures, keep
    gc.collect()
    gc.freeze()

    # Sandboxing is the expensive stage. Candidates go in seeded order and
    # the run stops once both pools are full. A candidate whose identity is
    # already saturated is skipped unsandboxed: ``offer`` would drop it
    # whatever its verdict, so given the same verdicts the pools are the
    # same for any window size (--workers). Verdicts themselves are
    # wall-clock bound (1 s reference limit, 3 s sandbox timeout), so heavy
    # contention can still flip a borderline row.
    verdict_list: list[dict] = []
    submitted: list[dict] = []
    assigner = PoolAssigner(args.rl_rows, args.sft_rows)
    verified_so_far = 0
    started = time.time()

    def pending():
        for item in candidates:
            if assigner.saturated(item):
                counts["skipped_saturated_identity"] += 1
                continue
            submitted.append(item)
            yield item

    with get_context("fork").Pool(args.workers) as pool:
        for verdict in windowed_imap(
            pool, _verify_or_reject, pending(), args.workers * 64
        ):
            item = submitted[len(verdict_list)]
            if verdict["uuid"] != item["uuid"]:
                raise AssertionError("imap returned verdicts out of order")
            verdict_list.append(verdict)
            if verdict["verdict"] == "verified":
                verified_so_far += 1
                assigner.offer(item)
            if len(verdict_list) % 10_000 == 0:
                print(f"verify: {len(verdict_list)} sandboxed, "
                      f"{counts['skipped_saturated_identity']} skipped of "
                      f"{len(candidates)}, {verified_so_far} verified, "
                      f"{len(assigner.rl)} RL, {len(assigner.sft)} SFT "
                      f"({time.time() - started:.0f}s)", file=sys.stderr,
                      flush=True)
            if assigner.full:
                break
        pool.terminate()
    gc.unfreeze()
    # imap yields in input order; the window may have pulled a few more.
    sandboxed_items = submitted[: len(verdict_list)]
    verdicts = {verdict["uuid"]: verdict for verdict in verdict_list}
    counts["sandboxed"] = len(sandboxed_items)
    counts["pools_full"] = int(assigner.full)
    verdict_counts = Counter(v["verdict"] for v in verdicts.values())
    first_counts = Counter(v["first"] for v in verdicts.values())
    walls_ms = [
        round(1000 * max(v["walls"]))
        for v in verdicts.values()
        if v["verdict"] == "verified"
    ]
    verified = [
        item
        for item in sandboxed_items
        if verdicts[item["uuid"]]["verdict"] == "verified"
    ]
    for item in verified:
        per_shard[item["_shard"]]["verified"] += 1
    for item in sandboxed_items:
        per_shard[item["_shard"]]["sandboxed"] += 1

    rl_items = [unpack_body(item) for item in assigner.rl]
    sft_items = [unpack_body(item) for item in assigner.sft]
    counts.update(assigner.dropped)
    twins = near_twins(
        np.stack([minhash(item["task"]) for item in sft_items]),
        np.stack([minhash(item["task"]) for item in rl_items]),
    )
    counts["sft_near_twin_of_rl"] = int(twins.sum())
    sft_items = [item for item, twin in zip(sft_items, twins, strict=True) if not twin]
    if len(rl_items) < args.rl_rows:
        raise SystemExit(
            f"only {len(rl_items)} RL rows after every candidate; raise --sample-rows"
        )
    if len(sft_items) < args.sft_rows:
        # Every candidate was considered; a short SFT pool is a measurement,
        # recorded in the manifest, not a failure.
        print(f"SFT pool short of target: {len(sft_items)} < {args.sft_rows}",
              file=sys.stderr, flush=True)
    # Disjointness is by construction; assert it on both identities anyway.
    rl_ids = {normalized_task(item["task"]) for item in rl_items}
    rl_problems = {problem_identity(item) for item in rl_items}
    if any(
        normalized_task(item["task"]) in rl_ids or problem_identity(item) in rl_problems
        for item in sft_items
    ):
        raise AssertionError("RL and SFT pools share a problem")

    atomic_parquet(
        pa.Table.from_pylist([rl_row(item) for item in rl_items]), rl_path
    )
    atomic_parquet(
        pa.Table.from_pylist(
            [
                {
                    "uuid": item["uuid"],
                    "problem": item["task"],
                    "analysis": item["analysis"],
                    "solution": item["solution"],
                    "tests": item["tests"],
                    "entry_points": item["entry_points"],
                    "provenance": PROVENANCE,
                    "shard": item["_shard"],
                }
                for item in sft_items
            ],
            schema=SFT_POOL_SCHEMA,
        ),
        sft_path,
    )
    sandboxed = len(sandboxed_items)
    manifest = {
        "schema": BUILD_SCHEMA,
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "download_manifest": str(DOWNLOAD_MANIFEST),
        "download_manifest_sha256": file_sha256(DOWNLOAD_MANIFEST),
        "shard_sha256": {entry["path"]: entry["sha256"] for entry in files},
        "seed": args.seed,
        "arguments": {
            "sample_rows": args.sample_rows,
            "rl_rows": args.rl_rows,
            "sft_rows": args.sft_rows,
            "seed": args.seed,
            "workers": args.workers,
            "chunk_rows": args.chunk_rows,
        },
        # The rules below are code; bind the bytes that implemented them.
        "source_sha256": {
            str(path): file_sha256(path) for path in SOURCE_FILES
        },
        "decontamination_target_sha256": {
            str(path): file_sha256(path)
            for path in (
                *DECONTAMINATION_TARGETS,
                *(target for target, _ in CONTAINMENT_TARGETS),
                *(target for target, _ in CODE_CONTAINMENT_TARGETS),
            )
        },
        "sample_rate": rate,
        "mirror_rows": total_rows,
        "python_reward_schema": PYTHON_REWARD_SCHEMA,
        "rules": {
            "solution_policy": "python_candidate_allowed (the RL reward's AST policy)",
            "tests": "top-level statements of the upstream test module, all kept",
            "min_top_level_asserts": MIN_TOP_LEVEL_ASSERTS,
            "test_import_roots": sorted(TEST_IMPORT_ROOTS),
            "verification": (
                "reference passes twice in the bwrap sandbox, each within "
                f"{MAX_REFERENCE_WALL_SECONDS}s wall; the solution with every "
                "top-level function body replaced by return None fails"
            ),
            "test_names": (
                "every name a test reads is an entry point, a builtin, or bound "
                "by the test module itself"
            ),
            "entry_points": "every solution function the tests use is named in the task",
            "meta_reference": META_REFERENCE.pattern,
            "decontamination": {
                "containment_targets": reference_counts,
                "containment_rule": (
                    f">= {CONTAINMENT_MIN_OVERLAP:.0%} of a reference problem's "
                    "word 8-grams contained in the task, or exact normalized match"
                ),
                "math_index": "prepare_sft_traces exact + any word 8-gram",
            },
            "dedupe": (
                "exact normalized task and the SFT corpus's 160-char identity; "
                f"MinHash {MINHASH_PERMUTATIONS} perms over word {SHINGLE}-grams, "
                f"{LSH_BANDS} bands, estimated Jaccard >= {NEAR_DUPLICATE_JACCARD}; "
                "first in seeded order is the representative, before the split"
            ),
            "rl_prompt": (
                "bare task + up to two self-contained one-line example asserts "
                f"(<= {MAX_EXAMPLE_CHARS} chars), BOS + prompt <= "
                f"{RL_PROMPT_TOKENS} GPT-2 tokens; the unshown statements must "
                "call an entry point with arguments no example shows; graded "
                "on every kept statement"
            ),
            "cross_pool": (
                f"SFT rows with estimated Jaccard >= {CROSS_POOL_JACCARD} to any RL "
                f"task are dropped ({CROSS_POOL_BANDS}-band LSH, full-signature check)"
            ),
            "split": (
                "problem identity = sorted entry-point names; in seeded order "
                "the first verified row of an identity puts it in RL (while RL "
                "has room and the row has a prompt) or SFT, for good; at most "
                f"{RL_PER_IDENTITY} RL and {SFT_PER_IDENTITY} SFT rows per identity"
            ),
        },
        "counts": dict(counts),
        "verification_audit": {
            "stop_rule": (
                "both pools full (--rl-rows, --sft-rows) or candidates "
                "exhausted; rows of a saturated identity are skipped unsandboxed"
            ),
            "sandboxed": sandboxed,
            "verdicts": dict(verdict_counts),
            "first_run_results": dict(first_counts),
            "verified_fraction_of_sandboxed": round(len(verified) / sandboxed, 4),
            "verified_wall_ms": percentiles(walls_ms),
        },
        "per_shard": {str(index): dict(value) for index, value in per_shard.items()},
        "rl": {
            "path": str(rl_path),
            "rows": len(rl_items),
            "sha256": file_sha256(rl_path),
            "prompt_tokens": percentiles(
                [item["rl_prompt_tokens"] for item in rl_items]
            ),
            "shown_examples": dict(Counter(item["shown_examples"] for item in rl_items)),
            "graded_statements": percentiles([len(item["tests"]) for item in rl_items]),
        },
        "sft_pool": {
            "path": str(sft_path),
            "rows": len(sft_items),
            "sha256": file_sha256(sft_path),
            "problem_identities": len({problem_identity(item) for item in sft_items}),
            "rl_eligible_rows_left_in_pool": sum(
                item["rl_prompt"] is not None for item in sft_items
            ),
        },
    }
    atomic_write_bytes(
        manifest_path, (json.dumps(manifest, indent=2) + "\n").encode()
    )
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in {"shard_sha256", "per_shard"}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="mirror pinned L3/py shards")
    fetch.add_argument("--shards", type=int, default=19)
    fetch.add_argument("--connections", type=int, default=8)
    commands.add_parser("targets", help="build MBPP/HumanEval containment targets")
    make = commands.add_parser("build", help="verified SFT pool and RL rows")
    make.add_argument("--output-prefix", required=True)
    make.add_argument("--sample-rows", type=int, required=True)
    make.add_argument("--rl-rows", type=int, default=20_000)
    make.add_argument(
        "--sft-rows", type=int, required=True,
        help="stop sandboxing once the SFT pool holds this many rows (and RL is full)",
    )
    make.add_argument("--seed", type=int, default=0)
    make.add_argument(
        "--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 6))
    )
    make.add_argument("--chunk-rows", type=int, default=100)
    args = parser.parse_args()
    if args.command == "download":
        download(args)
    elif args.command == "targets":
        build_targets(args)
    else:
        if min(args.sample_rows, args.rl_rows, args.sft_rows, args.workers) < 1:
            parser.error(
                "--sample-rows, --rl-rows, --sft-rows and --workers must be positive"
            )
        build(args)


if __name__ == "__main__":
    main()

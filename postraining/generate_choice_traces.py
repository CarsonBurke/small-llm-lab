"""Distil verified single-choice worked solutions from a local teacher.

The job 9040 SFT policy scores about 0.2% on the ``science_mc`` RL pool
against 25% chance, because nothing it was trained on answers with a letter
(job 9466, NOTES 2026-09-23). This module writes the missing SFT source. The
teacher is the strongest model available locally, Qwen3.8-27B (unsloth
UD-IQ4_XS GGUF), served by the local llama.cpp build with continuous
batching; MiniCPM5-1B, the only teacher with an in-repo generation path, is a
far weaker science answerer, and at four options a weak teacher's correct
letters come with post-hoc reasoning much more often.

Two stages, one CLI:

* ``generate`` (GPU, through ``mlq``) starts ``llama-server`` in the job's
  own process group, sends every question of the SFT partition written by
  ``scripts/build_science_mc_rl_prompts.py`` ``--samples`` times, and appends
  one JSON line per completion. The teacher sees the rendered problem the RL
  policy sees -- question, blank line, ``A. option`` lines -- as the user
  turn; the system turn asks for brief reasoning and a final ``Answer: X``
  line, and thinking mode is off. Resume is append-only and refuses a
  changed teacher, server, prompt, sampling setting or question file.
* ``select`` (CPU) verifies and screens every completion and keeps one
  trace per question as a pool parquet for the ``science_mc_traces`` adapter
  of ``prepare_sft_corpus``. Verification is outcome-only, as in
  rejection-sampling fine-tuning (RFT, STaR): a completion is correct when
  its last line is ``Answer: X`` and X is the gold letter, and the reward's
  own grader (``core.verify_answer``, ``exact`` style) agrees. The prose is
  not parsed for a conclusion. A right letter reached by a guess is
  controlled by self-consistency instead: a question's traces are admitted
  only if at least ``ceil(--min-correct-share * k)`` of its k samples are
  correct. A correct completion is kept if it passes the mechanical screens
  of ``choice_prompt.choice_trace_defect`` (no claim that the gold option is
  wrong, no second answer line or drafted response, no talk of the answer
  format the student never sees, of the teacher's instructions or of an answer
  key, nothing degenerate, cut off
  or looping) and the canonical document fits ``--max-doc-tokens`` GPT-2
  tokens as the trainer frames it. Among the survivors of one question the
  kept trace is the one with the smallest content hash, so the choice is
  deterministic and blind to length and sample order.

    mlq submit --name science_mc_teacher --cwd "$PWD" --max-parallel-runs 1 -- \\
        .venv/bin/python -m postraining.generate_choice_traces generate \\
        --output postraining/data/choice_traces/science-mc-qwen38-27b-v1.jsonl
    .venv/bin/python -m postraining.generate_choice_traces select \\
        --generations postraining/data/choice_traces/science-mc-qwen38-27b-v1.jsonl \\
        --output postraining/data/science-mc-traces-v1-pool.parquet
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import re
import socket
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from postraining.choice_prompt import choice_trace_defect, split_options
from postraining.choice_rl_pool import sha256_file, split_draw
from postraining.core import GPT2BPETokenizer, answer_style, verify_answer
from postraining.prepare_sft_corpus import (
    build_document,
    refuse_evaluation_only,
    trained_length,
)
from postraining.prepare_sft_traces import FENCE_STRINGS

GENERATION_SCHEMA = "choice_trace_generation/v1"
RECORD_FIELDS = {
    "schema", "key", "sample", "seed", "content", "reasoning_content",
    "finish_reason", "prompt_tokens", "completion_tokens",
}
POOL_SCHEMA = "choice_trace_pool/v2"
# Self-consistency thresholds every pool manifest reports yield at, whatever
# ``--min-correct-share`` admits: k/4, k/2 and 3k/4 correct samples.
CONSISTENCY_REPORT_SHARES = (0.25, 0.5, 0.75)

DEFAULT_QUESTIONS = Path("postraining/data/science-mc-sft-questions-v3.parquet")
DEFAULT_SERVER = Path.home() / "llm/llama.cpp-src/build-gemm/bin/llama-server"
DEFAULT_MODEL = (
    Path.home() / "models/Qwen3.8-27B-GGUF-IQ4XS/Qwen3.8-27B-UD-IQ4_XS.gguf"
)
# The hub identity of DEFAULT_MODEL, from the download's own metadata file
# (``.cache/huggingface/download/*.gguf.metadata``): the file's LFS sha256
# and the repository commit it was fetched at. ``generate`` checks the bytes.
DEFAULT_MODEL_IDENTITY = {
    "repo": "unsloth/Qwen3.8-27B-GGUF",
    "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
    "file": "Qwen3.8-27B-UD-IQ4_XS.gguf",
    "sha256": "40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199",
    "base_model": "Qwen/Qwen3.8-27B",
    "license": "apache-2.0",
}

# Value-blind: identical for every question, so no gold information reaches
# the teacher. It is phrased so the reasoning has no reason to discuss it;
# ``select`` drops the traces that do anyway.
SYSTEM_PROMPT = (
    "You are writing worked solutions used to train a very small student "
    "model. Each question is a multiple-choice science question followed by "
    "its lettered options. Reason step by step in clean, complete sentences: "
    "recall the fact or principle the question turns on, apply it, and say "
    "briefly why the other options are wrong when that helps. Be as brief as "
    "the question allows, with no headings, no lists, no bold text and no "
    "restating of the question. Refer to an option by its letter, as in "
    "\"option B\". End the reasoning with one sentence that names the "
    "correct option by its letter, such as \"So the correct option is B.\" "
    "Then write the final answer on its own last line in exactly the form "
    "'Answer: X', where X is that single letter, and write nothing after "
    "that line."
)
# The first wording ("Finish the reasoning with a sentence that names the
# correct option by its letter") was ignored: in the 256-question timing run
# 398 of 512 completions, 96% of them correct, concluded by option text
# ("Therefore, the atom is the correct answer.") and failed the conclusion
# screen then in use. That screen is gone (verification is outcome-only),
# but the prompt is part of the job 9506 generation contract and stays.

# The teacher's last line, or the end of it after a finished sentence ("So
# the correct option is B. Answer: B"). It is the completion's one structural
# requirement and is removed from the <think> body, whose answer is the
# <answer> span; an answer line anywhere in the rest is dropped by
# ``choice_prompt.choice_trace_defect``, which the SFT adapter re-applies.
ANSWER_LINE = re.compile(
    r"(?P<sentence>.*?[.!?][\"'”’)]*)?[ \t]*Answer:[ \t]*(?P<letter>[A-Z])"
)


def split_answer_line(visible: str) -> tuple[str, str] | str:
    """(reasoning, letter) from a completion, or a drop reason."""

    lines = visible.rstrip().split("\n")
    final = ANSWER_LINE.fullmatch(lines[-1].strip())
    if final is None:
        return "no_answer_line"
    reasoning = "\n".join([*lines[:-1], final["sentence"] or ""]).strip()
    if not reasoning:
        return "no_reasoning"
    return reasoning, final["letter"]


def question_rows(path: Path) -> list[dict]:
    """The question partition, one dict per question, in a content order.

    Ordered by a hash of the choice identity, so any prefix (``--limit``)
    is a uniform sample across sources.
    """

    rows = []
    for row in pq.read_table(refuse_evaluation_only(path)).to_pylist():
        problem = row["prompt"][-1]["content"]
        parsed = split_options(problem)
        if parsed is None:
            raise ValueError(f"{path}: a row is not a single-choice problem")
        rows.append({
            "key": row["extra_info"]["original_query_sha256"],
            "source": row["data_source"].removeprefix("science_mc_"),
            "module": row["extra_info"]["module"],
            "problem": problem,
            "choice": parsed,
            "gold": row["reward_model"]["ground_truth"],
            "style": answer_style(row),
        })
    if len({row["key"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate question keys")
    return sorted(rows, key=lambda row: split_draw(row["key"], "teacher_order"))


def request_seed(key: str, sample: int, seed: int) -> int:
    digest = hashlib.sha256(f"{key}\x00{sample}\x00{seed}".encode()).hexdigest()
    return int(digest[:8], 16)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class LlamaServer:
    """``llama-server`` as a child of this process, for one ``with`` block.

    It stays in the job's process group, so cancelling the ``mlq`` job stops
    it too; its log goes next to the generation output.
    """

    def __init__(self, command: list[str], port: int, log: Path, timeout: float):
        self.command = command
        self.url = f"http://127.0.0.1:{port}"
        self.log = log
        self.timeout = timeout
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "LlamaServer":
        self.stream = self.log.open("a")
        self.process = subprocess.Popen(
            self.command, stdout=self.stream, stderr=subprocess.STDOUT
        )
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with {self.process.returncode}; "
                    f"see {self.log}"
                )
            try:
                if requests.get(f"{self.url}/health", timeout=5).status_code == 200:
                    return self
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(2)
        self.__exit__(None, None, None)
        raise TimeoutError(f"llama-server not healthy after {self.timeout}s")

    def __exit__(self, *_) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.stream.close()


def server_flags(args: argparse.Namespace) -> list[str]:
    """Every server setting but the model file and the address."""

    flags = [
        "--n-gpu-layers", "999", "--flash-attn", "on",
        "--parallel", str(args.parallel),
        "--ctx-size", str(args.parallel * args.slot_context),
        # No host-side prompt cache (8 GiB by default, about 0.5 GiB an
        # entry for this hybrid model) and no recurrent-state checkpoints,
        # which are held in host memory too: every prompt is a different
        # question, so neither is reused, and the first pilot's server
        # reached 14.5 GB of host memory before the kernel OOM killer
        # stopped it (job 9499).
        "--cache-ram", "0", "--ctx-checkpoints", "0",
        "--reasoning", "off", "--no-webui",
    ]
    if args.draft_model is not None:
        flags += [
            "--spec-type", args.spec_type,
            "--model-draft", str(args.draft_model),
            "--n-gpu-layers-draft", "999",
        ]
    elif args.spec_type != "none":
        flags += ["--spec-type", args.spec_type]
    return flags


def server_command(args: argparse.Namespace, port: int) -> list[str]:
    return [
        str(args.server), "--model", str(args.model),
        "--host", "127.0.0.1", "--port", str(port), *server_flags(args),
    ]


def server_version(server: Path) -> str:
    result = subprocess.run(
        [str(server), "--version"], capture_output=True, text=True, timeout=60
    )
    lines = (result.stdout + result.stderr).strip().splitlines()
    version = [line for line in lines if line.startswith("version:")]
    if not version:
        raise RuntimeError(f"{server} --version printed no version line")
    return version[0]


def generation_meta(args: argparse.Namespace) -> dict:
    """Everything a resumed run must share with the run it continues."""

    model_sha256 = sha256_file(args.model)
    if args.model == DEFAULT_MODEL and model_sha256 != DEFAULT_MODEL_IDENTITY["sha256"]:
        raise SystemExit(
            f"{args.model} sha256 {model_sha256} is not the pinned teacher"
        )
    return {
        "schema": GENERATION_SCHEMA,
        "questions": str(args.questions),
        "questions_sha256": sha256_file(args.questions),
        "teacher": {
            **(DEFAULT_MODEL_IDENTITY if args.model == DEFAULT_MODEL else {}),
            "path": str(args.model),
            "sha256": model_sha256,
            "mode": "chat, thinking off (--reasoning off, enable_thinking false)",
        },
        "draft_model": (
            {"path": str(args.draft_model),
             "sha256": sha256_file(args.draft_model)}
            if args.draft_model is not None else None
        ),
        "spec_type": args.spec_type,
        "server": {
            "path": str(args.server),
            "sha256": sha256_file(args.server),
            "version": server_version(args.server),
            "flags": server_flags(args),
        },
        "system_prompt": SYSTEM_PROMPT,
        "sampling": {
            "samples_per_question": args.samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": 0.0,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "request_seed": "first 8 hex of sha256(key NUL sample NUL seed)",
        },
    }


def drop_torn_tail(output: Path) -> int:
    """Cut a record torn by a crash mid-write; returns the bytes removed.

    Records are appended whole and newline-terminated, so only the bytes
    after the last newline can be torn. Appending after them would fuse the
    next record onto the fragment.
    """

    if not output.exists():
        return 0
    data = output.read_bytes()
    keep = data.rfind(b"\n") + 1
    if keep == len(data):
        return 0
    with output.open("r+b") as stream:
        stream.truncate(keep)
    return len(data) - keep


def parse_records(data: bytes, output: Path) -> list[dict]:
    """Every record in ``data``, refusing any line that is not one whole record."""

    records = []
    lines = data.split(b"\n")
    if lines[-1]:
        raise SystemExit(f"{output}:{len(lines)} is torn; resume generate to repair it")
    for number, line in enumerate(lines[:-1], 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{output}:{number}: {error}") from None
    return records


def read_records(output: Path) -> list[dict]:
    return parse_records(output.read_bytes(), output)


def finished(output: Path) -> set[tuple[str, int]]:
    """(key, sample) pairs already written."""

    if not output.exists():
        return set()
    return {(record["key"], record["sample"]) for record in read_records(output)}


def complete(
    session: requests.Session, url: str, question: dict, sample: int,
    args: argparse.Namespace,
) -> dict:
    seed = request_seed(question["key"], sample, args.seed)
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question["problem"]},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": 0.0,
        "max_tokens": args.max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(5):
        try:
            response = session.post(
                f"{url}/v1/chat/completions", json=payload, timeout=args.timeout
            )
        except requests.exceptions.RequestException:
            if attempt == 4:
                raise
            time.sleep(2.0 ** attempt)
            continue
        if response.status_code == 200:
            break
        if response.status_code < 500 or attempt == 4:
            raise RuntimeError(
                f"HTTP {response.status_code}: {response.text[:300]}"
            )
        time.sleep(2.0 ** attempt)
    body = response.json()
    choice = body["choices"][0]
    return {
        "schema": GENERATION_SCHEMA,
        "key": question["key"],
        "sample": sample,
        "seed": seed,
        "content": choice["message"].get("content") or "",
        "reasoning_content": choice["message"].get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": body["usage"]["prompt_tokens"],
        "completion_tokens": body["usage"]["completion_tokens"],
    }


def generate(args: argparse.Namespace) -> None:
    questions = question_rows(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    meta = generation_meta(args)
    meta_path = args.output.with_suffix(".meta.json")
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous != meta:
            changed = sorted(
                key for key in meta.keys() | previous.keys()
                if meta.get(key) != previous.get(key)
            )
            raise SystemExit(
                f"{args.output} was generated under a different contract "
                f"({', '.join(changed)} changed); write a new output"
            )
    elif args.output.exists():
        raise SystemExit(f"{args.output} exists without {meta_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    with args.output.open("a") as stream:
        # Two writers appending to one output would interleave and duplicate
        # records; the second one to start refuses instead.
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{args.output} is being written by another run") from None
        append(args, questions, stream)


def append(args: argparse.Namespace, questions: list[dict], stream) -> None:
    """Generate every missing (question, sample) into the locked ``stream``."""

    if torn := drop_torn_tail(args.output):
        print(f"dropped a torn {torn}-byte record; it is regenerated", flush=True)
    done = finished(args.output)
    work = [
        (question, sample)
        for sample in range(args.samples)
        for question in questions
        if (question["key"], sample) not in done
    ]
    print(
        f"{len(work)} completions pending of {len(questions) * args.samples} "
        f"({len(questions)} questions x {args.samples})",
        flush=True,
    )
    if not work:
        return
    port = free_port()
    lock = threading.Lock()
    local = threading.local()
    written = completion_tokens = 0
    started = time.monotonic()

    def run(item: tuple[dict, int], url: str) -> None:
        nonlocal written, completion_tokens
        if not hasattr(local, "session"):
            local.session = requests.Session()
        record = complete(local.session, url, *item, args)
        with lock:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            written += 1
            completion_tokens += record["completion_tokens"]
            if written % args.log_every == 0 or written == len(work):
                elapsed = time.monotonic() - started
                rate = written / elapsed
                print(
                    f"{written}/{len(work)} completions, "
                    f"{completion_tokens / elapsed:.0f} completion tok/s, "
                    f"{rate:.2f} completions/s, "
                    f"eta {(len(work) - written) / rate / 3600:.2f} h",
                    flush=True,
                )

    with LlamaServer(
        server_command(args, port), port,
        args.output.with_suffix(".server.log"), args.startup_timeout,
    ) as server:
        started = time.monotonic()
        # Sample 0 of every question comes first, so a stopped run still
        # covers every question once.
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(run, item, server.url) for item in work]
            try:
                for future in futures:
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    elapsed = time.monotonic() - started
    print(json.dumps({
        "completions": written,
        "seconds": round(elapsed, 1),
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": round(completion_tokens / elapsed, 1),
        "questions_per_hour": round(written / args.samples / elapsed * 3600, 1),
    }), flush=True)


def select(args: argparse.Namespace) -> None:
    meta_path = args.generations.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    if meta["schema"] != GENERATION_SCHEMA:
        raise SystemExit(f"{meta_path} is not {GENERATION_SCHEMA}")
    questions_path = Path(meta["questions"])
    if sha256_file(questions_path) != meta["questions_sha256"]:
        raise SystemExit(f"{questions_path} changed since generation")
    for path in (args.output, args.output.with_suffix(".manifest.json")):
        if path.exists():
            raise SystemExit(f"{path} exists; trace pools are immutable")
    if "." in args.output.stem:
        raise SystemExit("--output stem must not contain a dot")

    questions = {row["key"]: row for row in question_rows(questions_path)}
    # Read once, under a shared lock a running generate's exclusive one
    # refuses, and hash the bytes that were parsed: the manifest must
    # describe exactly the generations the pool was selected from.
    with args.generations.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                f"{args.generations} is being written by another run"
            ) from None
        data = stream.read()
    generations_sha256 = hashlib.sha256(data).hexdigest()
    candidates: dict[str, dict[int, dict]] = defaultdict(dict)
    samples_per_question = meta["sampling"]["samples_per_question"]
    for record in parse_records(data, args.generations):
        if record.get("schema") != GENERATION_SCHEMA or not RECORD_FIELDS <= record.keys():
            raise SystemExit(f"a record is not a {GENERATION_SCHEMA} record: {record!r:.200}")
        if not (
            isinstance(record["sample"], int)
            and 0 <= record["sample"] < samples_per_question
        ):
            raise SystemExit(
                f"record {record['key']}/{record['sample']} is outside the "
                f"{samples_per_question} samples generated per question"
            )
        if record["key"] not in questions:
            raise SystemExit(f"generation for unknown question {record['key']}")
        if record["seed"] != request_seed(
            record["key"], record["sample"], meta["sampling"]["seed"]
        ):
            raise SystemExit(f"record {record['key']}/{record['sample']} seed mismatch")
        if record["sample"] in candidates[record["key"]]:
            raise SystemExit(
                f"record {record['key']}/{record['sample']} written twice"
            )
        candidates[record["key"]][record["sample"]] = record

    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    required = required_correct(args.min_correct_share, samples_per_question)
    reasons = Counter()
    per_source = defaultdict(Counter)
    per_module = defaultdict(Counter)
    passing_per_question = Counter()
    correct_per_question = Counter()
    # Questions admitted, and traces kept, at other consistency thresholds,
    # overall, per source and per option count: the cost of guess control in
    # yield, and whether it starves a group.
    admitted_at = {share: defaultdict(Counter) for share in CONSISTENCY_REPORT_SHARES}
    complete_in = Counter()
    per_option_count = defaultdict(Counter)
    complete = 0
    question_letters = Counter()
    rows = []
    for key, question in questions.items():
        samples = candidates.get(key)
        if not samples:
            continue
        tally = per_source[question["source"]]
        module = per_module[question["module"]]
        tally["questions"] += 1
        module["questions"] += 1
        verdicts = []
        for sample, record in sorted(samples.items()):
            tally["candidates"] += 1
            verdicts.append((sample, record, judge(
                record, question, tokenizer, args.max_doc_tokens
            )))
        correct = sum(verdict["correct"] for _, _, verdict in verdicts)
        survivors = [
            {**verdict, "sample": sample, "record": record}
            for sample, record, verdict in verdicts if "reason" not in verdict
        ]
        tally["correct_candidates"] += correct
        tally["questions_any_correct"] += correct > 0
        module["questions_any_correct"] += correct > 0
        correct_per_question[(correct, len(samples))] += 1
        passing_per_question[len(survivors)] += 1
        for _, _, verdict in verdicts:
            if "reason" in verdict:
                reasons[verdict["reason"]] += 1
        if len(samples) < samples_per_question:
            # A question the generation has not finished cannot be judged
            # for consistency; its survivors are counted, not kept.
            reasons["incomplete_question"] += len(survivors)
            tally["incomplete"] += 1
            continue
        complete += 1
        tally["complete"] += 1
        module["complete"] += 1
        options = per_option_count[len(question["choice"].labels)]
        options["complete"] += 1
        question_letters[question["gold"]] += 1
        groups = (
            ("all", None),
            ("per_source", question["source"]),
            ("per_option_count", len(question["choice"].labels)),
        )
        complete_in.update(groups)
        for share, admitted in admitted_at.items():
            if correct >= required_correct(share, samples_per_question):
                for group in groups:
                    admitted[group]["questions"] += 1
                    admitted[group]["traces"] += bool(survivors)
        if correct < required:
            # Counted only where a trace would otherwise have been kept, so
            # the tallies are what consistency itself removed.
            reasons["below_consistency"] += len(survivors)
            tally["below_consistency"] += bool(survivors)
            continue
        if not survivors:
            continue
        kept = min(
            survivors,
            key=lambda item: hashlib.sha256(
                item["record"]["content"].encode("utf-8")
            ).hexdigest(),
        )
        tally["kept"] += 1
        module["kept"] += 1
        options["kept"] += 1
        rows.append({
            "key": key,
            "source": question["source"],
            "module": question["module"],
            "problem": question["problem"],
            "reasoning": kept["reasoning"],
            "letter": kept["letter"],
            "option_count": len(question["choice"].labels),
            "sample": kept["sample"],
            "seed": kept["record"]["seed"],
            "teacher_completion_tokens": kept["record"]["completion_tokens"],
            "doc_tokens": kept["doc_tokens"],
            "correct_candidates": correct,
            "passing_candidates": len(survivors),
            "candidates": len(samples),
            "verified": True,
        })
    if not rows:
        raise SystemExit("no trace survived")
    rows.sort(key=lambda row: (row["source"], row["key"]))

    letters = Counter(row["letter"] for row in rows)
    tokens = sorted(row["doc_tokens"] for row in rows)
    staging = args.output.with_name(args.output.name + ".tmp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), staging)
    digest = sha256_file(staging)
    manifest = {
        "schema": POOL_SCHEMA,
        "generation": meta,
        "generations": str(args.generations),
        "generations_sha256": generations_sha256,
        "traces": len(rows),
        "questions_generated": len(candidates),
        "max_doc_tokens": args.max_doc_tokens,
        "selection": "one trace per admitted question: among completions that "
        "verify and pass every screen, the smallest sha256 of the completion "
        "text",
        "verifier": "outcome only: last line 'Answer: X' with X the gold "
        "letter, and core.verify_answer(completion, gold, style='exact') true; "
        "the reasoning is not parsed for a conclusion",
        "self_consistency": {
            "min_correct_share": args.min_correct_share,
            "samples_per_question": samples_per_question,
            "required_correct": required,
            "rule": "a question's traces are admitted only if at least "
            "ceil(min_correct_share * k) of its k samples are correct",
            "questions_complete": complete,
            "correct_per_question": {
                f"{correct}/{samples}": count
                for (correct, samples), count in sorted(correct_per_question.items())
            },
            "admitted_by_share": {
                str(share): {
                    "required_correct": required_correct(share, samples_per_question),
                    **admission(admitted[("all", None)], complete),
                    **{
                        kind: {
                            str(group[1]): admission(admitted[group], complete_in[group])
                            for group in sorted(g for g in complete_in if g[0] == kind)
                        }
                        for kind in ("per_source", "per_option_count")
                    },
                }
                for share, admitted in admitted_at.items()
            },
        },
        "screens": "choice_prompt.choice_trace_defect (mechanical), fence "
        "literals, finish_reason 'stop', empty reasoning channel, "
        "prepare_sft_corpus.trained_length <= max_doc_tokens",
        "rejections": dict(reasons.most_common()),
        "per_source": {
            name: {
                **dict(tally),
                "yield": tally["kept"] / tally["complete"] if tally["complete"] else None,
                "candidate_accuracy": tally["correct_candidates"] / tally["candidates"],
            }
            for name, tally in sorted(per_source.items())
        },
        "per_module": {
            name: {
                **dict(tally),
                "yield": tally["kept"] / tally["complete"] if tally["complete"] else None,
            }
            for name, tally in sorted(per_module.items())
        },
        "per_option_count": {
            str(count): {**dict(tally), "yield": tally["kept"] / tally["complete"]}
            for count, tally in sorted(per_option_count.items())
        },
        "passing_candidates_per_question": dict(sorted(passing_per_question.items())),
        "kept_letters": dict(sorted(letters.items())),
        "question_letters": dict(sorted(question_letters.items())),
        # A letter's share of the kept traces over its share of the complete
        # questions: the teacher saw uniformly shuffled options, so a letter
        # it keeps less often is where it errs or is screened more.
        "kept_share_over_question_share": {
            letter: (letters[letter] / len(rows))
            / (question_letters[letter] / complete)
            for letter in sorted(question_letters)
        },
        "modal_kept_letter_share": max(letters.values()) / len(rows),
        "doc_tokens": {
            "median": tokens[len(tokens) // 2],
            "p90": tokens[int(0.9 * (len(tokens) - 1))],
            "max": tokens[-1],
            "total": sum(tokens),
        },
        "output_sha256": digest,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    staging.replace(args.output)
    print(json.dumps({k: v for k, v in manifest.items() if k != "generation"},
                     indent=2))


def admission(admitted: Counter, complete: int) -> dict:
    """Questions admitted, those with a kept trace, and that share of ``complete``."""

    return {
        "questions": admitted["questions"],
        "questions_with_a_trace": admitted["traces"],
        "yield": admitted["traces"] / complete,
    }


def required_correct(share: float, samples: int) -> int:
    """Correct samples of ``samples`` a question needs at ``share``: at least one."""

    return max(1, math.ceil(share * samples - 1e-9))


def judge(
    record: dict, question: dict, tokenizer: GPT2BPETokenizer, max_doc_tokens: int
) -> dict:
    """{"reason": ...} for a dropped completion, else its trace and length.

    ``correct`` is reported either way, so teacher accuracy and
    self-consistency are measured before the screens: a completion is
    correct when it stopped, its last line is ``Answer: X`` with X the gold
    letter, and the reward's grader agrees.
    """

    content = record["content"]
    stopped = record["finish_reason"] == "stop"
    split = split_answer_line(content) if stopped else None
    graded = verify_answer(content, question["gold"], style=question["style"])[0]
    verdict: dict = {
        "correct": isinstance(split, tuple) and split[1] == question["gold"] and graded
    }
    if not stopped:
        return {**verdict, "reason": "truncated"}
    if record["reasoning_content"].strip():
        return {**verdict, "reason": "reasoning_channel"}
    if isinstance(split, str):
        return {**verdict, "reason": split}
    reasoning, letter = split
    if letter not in question["choice"].labels:
        return {**verdict, "reason": "letter_not_an_option"}
    if letter != question["gold"]:
        return {**verdict, "reason": "wrong_answer"}
    # The structural letter and the reward's own grading must agree.
    if not graded:
        return {**verdict, "reason": "grader_disagrees"}
    if any(fence in reasoning for fence in FENCE_STRINGS):
        return {**verdict, "reason": "fence_literal"}
    if defect := choice_trace_defect(reasoning, question["choice"], letter):
        return {**verdict, "reason": defect}
    document = build_document(question["problem"], reasoning, letter)
    if trained_length(question["problem"], document, tokenizer) > max_doc_tokens:
        return {**verdict, "reason": "over_doc_tokens"}
    doc_tokens = len(tokenizer.encode(document))
    return {**verdict, "reasoning": reasoning, "letter": letter,
            "doc_tokens": doc_tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="sample the teacher (GPU, mlq)")
    gen.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    gen.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    gen.add_argument("--draft-model", type=Path, default=None)
    gen.add_argument(
        "--spec-type", default="none",
        help="llama-server --spec-type (draft-dflash needs --draft-model; "
        "draft-mtp uses the teacher's own next-token head)",
    )
    gen.add_argument("--parallel", type=int, default=32, help="server slots")
    gen.add_argument(
        "--slot-context", type=int, default=2048,
        help="tokens per slot: system turn, problem and completion",
    )
    gen.add_argument("--samples", type=int, default=4)
    # Qwen's recommended non-thinking sampling.
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--top-p", type=float, default=0.8)
    gen.add_argument("--top-k", type=int, default=20)
    gen.add_argument(
        "--max-tokens", type=int, default=1024,
        help="teacher tokens per completion; longer traces cannot fit the "
        "document cap anyway and are dropped as truncated",
    )
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--limit", type=int, default=0, help="first N questions")
    gen.add_argument("--timeout", type=float, default=600.0)
    gen.add_argument("--startup-timeout", type=float, default=900.0)
    gen.add_argument("--log-every", type=int, default=200)
    sel = commands.add_parser("select", help="verify, screen, keep (CPU)")
    sel.add_argument("--generations", type=Path, required=True)
    sel.add_argument("--output", type=Path, required=True)
    sel.add_argument("--max-doc-tokens", type=int, default=1024)
    sel.add_argument(
        "--min-correct-share", type=float, default=0.5,
        help="admit a question's traces only if at least ceil(share * k) of "
        "its k samples are correct (self-consistency against lucky guesses)",
    )
    args = parser.parse_args()
    if args.command == "generate":
        if args.spec_type.startswith("draft-") and args.spec_type != "draft-mtp" \
                and args.draft_model is None:
            parser.error(f"--spec-type {args.spec_type} needs --draft-model")
        generate(args)
    else:
        if not 0 < args.min_correct_share <= 1:
            parser.error("--min-correct-share must be in (0, 1]")
        select(args)


if __name__ == "__main__":
    sys.exit(main())

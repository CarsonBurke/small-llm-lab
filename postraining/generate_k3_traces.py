"""Generate verified K3 reasoning traces over our own RL problem sources.

The round-5 SFT corpus needs a frontier-teacher math core that does not
exist on Hugging Face (K3-exhaustive survey, NOTES.md 2026-08-01): the
only genuine K3 math data is 810 olympiad rows. This script produces the
missing core by querying K3 directly over the two problem sources the RL
stage already trains on — GSM8K train and deepmind-interpolate — so the
SFT prior lands exactly on the RL distribution, with every trace verified
against exact ground truth BY THE RL VERIFIER ITSELF before it is kept.

Protocol per problem (bevangelista AIME recipe, adapted down-difficulty):
attempts walk an effort-escalation schedule and stop on the first attempt
whose visible answer verifies; a length-truncated attempt retries the
SAME effort with a doubled token cap (truncation is a budget problem, not
an effort problem). Both the provider's native reasoning channel and the
visible solution are stored; the compose-time choice of which becomes the
<think> span is deliberately deferred to prepare_sft_traces (the native
channel is authentic but telegraphic; the visible solution is clean
pedagogy — see the survey's register caveat).

Money-safety invariants (red-team reviewed):
- The budget meter fails CLOSED: a response without usable usage numbers
  is a fatal contract error, never a free request.
- Attempts the provider may have billed but could not report (timeouts,
  dropped connections, parse failures) are charged at a conservative
  estimate.
- A per-style yield canary halts a style whose verified yield collapses
  (e.g. a deterministic answer-format mismatch) instead of walking the
  full escalation schedule across thousands of problems.
- The pool fails fast: the first fatal error cancels all queued work.

Resume is append-only: rerun with the same --output and finished problems
are skipped; failed attempts retry from scratch.

    python3 -m postraining.generate_k3_traces \
        --output postraining/data/k3_traces/gsm8k_deepmind.jsonl \
        --gsm8k-rows 7217 --deepmind-rows 13000 \
        --budget-usd 40 --price-in-per-mtok 2 --price-out-per-mtok 10 \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from postraining.core import (
    GPT2BPETokenizer,
    answer_style,
    load_unique_math_rows,
    verify_answer,
)
from postraining.math_prompt import (
    ANSWER_FIELD_DEMAND,
    DAPO_HEADER,
    strip_math_prompt_framing,
)

GENERATION_SCHEMA = "k3_trace_generation/v1"

# The RL parquets bake the DAPO instruction sentences into the prompt;
# the teacher gets the bare problem plus our own contract instead. The
# rewrite table is the trainer's authoritative template set (includes the
# Chinese dapo template), so a source with any known layout strips clean.
SYSTEM_PROMPT = (
    "You are writing worked solutions used to train a very small student "
    "model. Reason step by step in clean, complete sentences. Be as brief "
    "as the problem allows — aim for under 250 words, with no headings, "
    "no recap of the problem, and no filler. End your response with the "
    "final answer on its own last line in exactly the form 'Answer: X', "
    "and write nothing after that line."
)
# mathematics_dataset grading is byte-exact string equality, so the
# teacher must land the canonical form, not just the right value. This
# hint is value-blind (identical for every problem): no ground-truth
# information reaches the teacher.
EXACT_STYLE_HINT = (
    " This problem expects one exact canonical answer: give fractions in "
    "lowest terms like 3/5 rather than decimal approximations (unless the "
    "problem itself works in decimals), answer multiple-choice questions "
    "with just the option letter, give multiple values comma-space "
    "separated in the order asked, and write booleans exactly as True or "
    "False."
)


class ContractError(RuntimeError):
    """Provider response violated the API contract; retrying cannot help."""


def bare_problem(row: dict) -> str:
    """The problem text alone, stripped of the baked-in DAPO instructions.

    Fail-loud: a row whose content still carries an Answer:-field demand
    after stripping would leak the plain-text contract into the teacher
    prompt (and mark a template this function does not know about).
    """
    content = "".join(message["content"] for message in row["prompt"])
    problem, _ = strip_math_prompt_framing(content)
    if not problem:
        raise ValueError("row reduced to an empty problem after stripping")
    return problem


def split_visible_solution(visible: str) -> tuple[str, str] | None:
    """(reasoning prose, final answer value) from a visible solution.

    The contract asks for a terminal ``Answer: X`` line; everything above
    it is the reasoning prose that compose-time may use as the think
    span. None when the contract was not honored (no Answer: line, or
    text AFTER the final answer line).
    """
    lines = visible.rstrip().splitlines()
    if not lines:
        return None
    match = ANSWER_FIELD_DEMAND.match(lines[-1].strip())
    if match is None:
        return None
    value = lines[-1].strip()[match.end():].strip()
    if not value:
        return None
    return "\n".join(lines[:-1]).strip(), value


def effort_body(effort: str, effort_key: str) -> dict:
    """Request-body fragment for one effort label.

    ``effort_key`` is dotted for providers that nest the control (e.g.
    OpenRouter's ``reasoning.effort``); the label ``none`` sends nothing.
    """
    if effort == "none":
        return {}
    body: dict = {}
    cursor = body
    *parents, leaf = effort_key.split(".")
    for parent in parents:
        cursor = cursor.setdefault(parent, {})
    cursor[leaf] = effort
    return body


def interleave_by_source(problems: list[dict]) -> list[dict]:
    """Round-robin across source prefixes.

    Both budget exhaustion and ``--limit`` pilots take a prefix of this
    list, so a head slice must sample every source — an all-GSM8K pilot
    would never exercise the exact-style half where format-mismatch risk
    lives (red-team finding 4).
    """
    groups: dict[str, deque] = defaultdict(deque)
    for problem in problems:
        groups[problem["key"].split("/")[0]].append(problem)
    queues = list(groups.values())
    order: list[dict] = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.popleft())
    return order


def finished_keys(output: Path, retry_exhausted: bool = False) -> set[str]:
    """Problem keys already resolved in an existing output file.

    A problem is finished once any attempt verified correct or an
    ``exhausted`` record marks the schedule as spent — reruns skip both.
    ``retry_exhausted`` counts only verified-correct problems as
    finished: after fixing a format contract, the problems whose
    failures DETECTED the broken contract (they exhausted their schedule
    before the canary fired) must not stay burned out of the corpus
    (red-team round 2, NEW-1). A torn trailing line (writer killed
    mid-record) is skipped, not fatal: its problem simply retries.
    """
    done: set[str] = set()
    if not output.exists():
        return done
    with output.open() as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("correct") or (
                record.get("exhausted") and not retry_exhausted
            ):
                done.add(record["key"])
    return done


def usage_int(usage: dict, key: str) -> int:
    """Usage count coerced defensively.

    A provider sending ``null`` (or any non-numeric junk) must fall into
    the fail-closed no-usage branch — which charges the estimate — not
    crash uncharged before the meter sees the attempt (red-team round 2,
    NEW-4).
    """
    value = usage.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class CostMeter:
    """Thread-safe running cost with a hard budget ceiling."""

    def __init__(self, budget_usd: float, price_in: float, price_out: float):
        self.budget_usd = budget_usd
        self.price_in = price_in
        self.price_out = price_out
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()

    def charge(self, usage: dict) -> None:
        with self._lock:
            self.prompt_tokens += usage_int(usage, "prompt_tokens")
            self.completion_tokens += usage_int(usage, "completion_tokens")

    def spent_usd(self) -> float:
        with self._lock:
            return (
                self.prompt_tokens * self.price_in
                + self.completion_tokens * self.price_out
            ) / 1e6

    def exhausted(self) -> bool:
        return self.spent_usd() >= self.budget_usd


class YieldCanary:
    """Per-style verified-yield floor with a hard abort.

    A deterministic failure (canonical-form mismatch, habitual teacher
    sign-off after the answer line) makes every attempt of every problem
    of that style fail while billing normally; without a canary the run
    walks the full escalation schedule across thousands of problems
    before anyone notices (red-team finding 3). Aborted styles leave
    their remaining problems unwritten so a fixed rerun retries them.
    """

    def __init__(self, min_resolved: int, min_yield: float):
        self.min_resolved = min_resolved
        self.min_yield = min_yield
        self._resolved: dict[str, int] = defaultdict(int)
        self._correct: dict[str, int] = defaultdict(int)
        self._aborted: set[str] = set()
        self._lock = threading.Lock()

    def resolve(self, style: str, correct: bool) -> None:
        with self._lock:
            self._resolved[style] += 1
            self._correct[style] += int(correct)
            resolved = self._resolved[style]
            hits = self._correct[style]
            # A deterministic contract failure yields exactly ZERO hits,
            # so it fires at half the sample size — detection cost
            # matters because every detection attempt is billed
            # (red-team round 2, NEW-2).
            tripped = (
                resolved >= self.min_resolved
                and hits / resolved < self.min_yield
            ) or (resolved >= max(1, self.min_resolved // 2) and hits == 0)
            if tripped and style not in self._aborted:
                self._aborted.add(style)
                print(
                    f"CANARY: style {style!r} yield {hits}/{resolved} — "
                    "halting that style; fix the format contract and "
                    "rerun with --retry-exhausted so the detection "
                    "sample is regenerated too",
                    flush=True,
                )

    def aborted(self, style: str) -> bool:
        with self._lock:
            return style in self._aborted

    def aborted_styles(self) -> set[str]:
        with self._lock:
            return set(self._aborted)


class TeacherClient:
    """Minimal OpenAI-compatible chat client (requests; no SDK dep).

    Money-safety: every response that reaches parsing has its usage
    charged to the meter FIRST; a response without positive usage counts
    is a fatal ``ContractError`` (a silent zero would disarm the budget
    ceiling entirely — red-team finding 1). Attempts that fail before a
    usage report exists (timeout, dropped connection) are charged at a
    conservative estimate, because the provider may have billed them
    anyway (finding 2). Rejected requests (429/5xx) are not charged.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        meter: CostMeter,
        max_retries: int = 4,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.model = model
        self.timeout = timeout
        self.meter = meter
        self.max_retries = max_retries

    def _post(self, payload: dict) -> tuple[int, str, dict | None]:
        """(status_code, text_snippet, parsed_json_or_None). Test seam."""
        response = requests.post(
            self.url, json=payload, headers=self.headers, timeout=self.timeout
        )
        try:
            body = response.json()
        except ValueError:
            body = None
        return response.status_code, response.text[:200], body

    def complete(
        self, system: str, problem: str, temperature: float,
        max_tokens: int, extra: dict,
    ) -> dict:
        """One attempt: returns reasoning/visible channels plus usage."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": problem},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra,
        }
        # Conservative upper bound charged whenever an attempt may have
        # been billed but could not report usage: ~3 chars/token
        # overestimates the prompt, and the completion is capped.
        billed_estimate = {
            "prompt_tokens": len(system + problem) // 3 + 16,
            "completion_tokens": max_tokens,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                status, text, body = self._post(payload)
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as error:
                # The server may have generated (and billed) the whole
                # completion even though we never saw it.
                self.meter.charge(billed_estimate)
                last_error = error
                time.sleep(min(2.0 ** attempt, 30.0))
                continue
            if status == 429 or status >= 500:
                # Rejected before generation: retried, never charged.
                last_error = RuntimeError(f"HTTP {status}: {text}")
                time.sleep(min(2.0 ** attempt, 30.0))
                continue
            if status != 200 or body is None:
                raise ContractError(f"HTTP {status} with body {text!r}")
            usage = body.get("usage") or {}
            if (
                usage_int(usage, "prompt_tokens") <= 0
                or usage_int(usage, "completion_tokens") <= 0
            ):
                # Billed but unmeterable — charge the estimate, then stop
                # the run before the pool spends blind.
                self.meter.charge(billed_estimate)
                raise ContractError(
                    f"response carried no usable usage ({str(usage)[:120]}); "
                    "enable usage reporting on the provider before rerunning"
                )
            self.meter.charge(usage)
            try:
                choice = body["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError, TypeError) as error:
                # Usage was charged above; the body is still unusable.
                raise ContractError(
                    f"unparseable response body: {str(body)[:200]}"
                ) from error
            return {
                "visible": message.get("content") or "",
                # Moonshot exposes reasoning_content; OpenRouter
                # normalizes to reasoning. Keep whichever exists.
                "reasoning": (
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or ""
                ),
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
            }
        raise RuntimeError(f"teacher request failed after retries: {last_error}")


def load_problems(args: argparse.Namespace) -> list[dict]:
    """Deterministic problem list: GSM8K train + a deepmind subsample.

    deepmind rows are decontaminated against the bench-default parquet by
    normalized problem text — the index namespaces overlap between files,
    so indices cannot be used for exclusion. Sources are interleaved so
    any prefix (budget exhaustion, --limit pilots) samples all of them.
    """
    problems: list[dict] = []
    if args.gsm8k_rows:
        rows = load_unique_math_rows(args.gsm8k_data)[: args.gsm8k_rows]
        for index, row in enumerate(rows):
            problems.append(
                {
                    "key": f"gsm8k/{index}",
                    "problem": bare_problem(row),
                    "ground_truth": row["reward_model"]["ground_truth"],
                    "style": answer_style(row),
                }
            )
    if args.deepmind_rows:
        bench_texts = {
            " ".join(bare_problem(row).split())
            for row in load_unique_math_rows(args.bench_data)
        }
        rows = load_unique_math_rows(args.deepmind_data)
        order = list(range(len(rows)))
        random.Random(args.seed).shuffle(order)
        taken = 0
        for index in order:
            if taken >= args.deepmind_rows:
                break
            row = rows[index]
            problem = bare_problem(row)
            if " ".join(problem.split()) in bench_texts:
                continue
            problems.append(
                {
                    "key": f"deepmind/{index}",
                    "problem": problem,
                    "ground_truth": row["reward_model"]["ground_truth"],
                    "style": answer_style(row),
                }
            )
            taken += 1
    if not problems:
        raise ValueError("no problems selected; pass --gsm8k-rows/--deepmind-rows")
    return interleave_by_source(problems)


def run_problem(
    problem: dict,
    client: TeacherClient,
    schedule: list[str],
    args: argparse.Namespace,
    meter: CostMeter,
    tokenizer,
    write,
    canary: YieldCanary,
) -> None:
    """Escalate through the schedule, stopping on first verified answer.

    A length-truncated attempt retries the SAME effort slot with a
    doubled token cap (up to --max-tokens-cap): more effort makes the
    reasoning channel longer and truncation MORE likely, so escalating
    on truncation compounds the failure (red-team finding 6).
    """
    system = SYSTEM_PROMPT + (
        EXACT_STYLE_HINT if problem["style"] == "exact" else ""
    )
    for attempt_index, effort in enumerate(schedule):
        max_tokens = args.max_tokens
        while True:
            if meter.exhausted() or canary.aborted(problem["style"]):
                return
            result = client.complete(
                system,
                problem["problem"],
                temperature=args.temperature,
                max_tokens=max_tokens,
                extra=effort_body(effort, args.effort_key),
            )
            split = split_visible_solution(result["visible"])
            # Graded exactly as the RL trainer grades plain-text
            # emissions: per-row style, default tail window.
            correct, prediction = verify_answer(
                result["visible"], problem["ground_truth"], style=problem["style"]
            )
            record = {
                "schema": GENERATION_SCHEMA,
                "key": problem["key"],
                "problem": problem["problem"],
                "ground_truth": problem["ground_truth"],
                "style": problem["style"],
                "attempt": attempt_index,
                "effort": effort,
                "max_tokens": max_tokens,
                "model": client.model,
                "reasoning_channel": result["reasoning"],
                "visible": result["visible"],
                "visible_prose": split[0] if split else None,
                "final_answer": split[1] if split else None,
                "prediction": prediction,
                # A verified value with prose after the Answer: line is
                # NOT kept: compose-time needs the clean prose/answer
                # split, so the contract violation retries instead (and
                # a habitual violation trips the yield canary).
                "correct": bool(correct and split is not None),
                "finish_reason": result["finish_reason"],
                "usage": result["usage"],
                "gpt2_prose_tokens": (
                    len(tokenizer.encode(split[0])) if split else None
                ),
                "gpt2_reasoning_channel_tokens": len(
                    tokenizer.encode(result["reasoning"])
                )
                if result["reasoning"]
                else 0,
            }
            write(record)
            if record["correct"]:
                canary.resolve(problem["style"], True)
                return
            if (
                result["finish_reason"] == "length"
                and max_tokens < args.max_tokens_cap
            ):
                max_tokens = min(max_tokens * 2, args.max_tokens_cap)
                continue
            break
    write(
        {
            "schema": GENERATION_SCHEMA,
            "key": problem["key"],
            "style": problem["style"],
            "exhausted": "schedule",
        }
    )
    canary.resolve(problem["style"], False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--gsm8k-data", default="postraining/data/gsm8k_rl_prompts.parquet"
    )
    parser.add_argument(
        "--deepmind-data",
        default="postraining/data/deepmind-interpolate-rl.parquet",
    )
    parser.add_argument(
        "--bench-data",
        default="postraining/data/deepmind-interpolate-easy.parquet",
        help="decontamination target: no generated problem may match it",
    )
    parser.add_argument("--gsm8k-rows", type=int, default=0)
    parser.add_argument("--deepmind-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--base-url", default="https://api.moonshot.ai/v1")
    parser.add_argument(
        "--api-key-env",
        default="MOONSHOT_API_KEY",
        help="environment variable holding the API key (never a flag)",
    )
    parser.add_argument(
        "--effort-schedule",
        default="low,low,high",
        help="comma list per attempt; 'none' sends no effort control",
    )
    parser.add_argument(
        "--effort-key",
        default="reasoning_effort",
        help="request field for effort; dotted nests (reasoning.effort)",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-tokens-cap",
        type=int,
        default=8192,
        help="ceiling for the truncation-retry doubling",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--budget-usd", type=float, required=True)
    # Set from the provider's price sheet at launch time; deliberately
    # required so a stale default can never under-meter a run.
    parser.add_argument("--price-in-per-mtok", type=float, required=True)
    parser.add_argument("--price-out-per-mtok", type=float, required=True)
    parser.add_argument(
        "--canary-min-resolved",
        type=int,
        default=50,
        help="resolved problems per style before the yield floor applies",
    )
    parser.add_argument(
        "--canary-min-yield",
        type=float,
        default=0.2,
        help="verified-yield floor per style; below it the style halts",
    )
    parser.add_argument("--limit", type=int, default=0, help="cap problems (pilot)")
    parser.add_argument(
        "--retry-exhausted",
        action="store_true",
        help="retry schedule-exhausted problems (after fixing a format "
        "contract that tripped the canary)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    schedule = [
        effort.strip() for effort in args.effort_schedule.split(",") if effort.strip()
    ]
    if not schedule:
        # An empty schedule would stamp every problem exhausted with zero
        # requests, permanently poisoning the resume file.
        parser.error("--effort-schedule must name at least one attempt")
    problems = load_problems(args)
    if args.limit:
        problems = problems[: args.limit]
        source_count = len({p["key"].split("/")[0] for p in problems})
        if args.limit < source_count * args.canary_min_resolved:
            print(
                f"WARNING: --limit {args.limit} gives under "
                f"{args.canary_min_resolved} problems per style, so the "
                "yield canary may never arm during this pilot",
                flush=True,
            )
    output = Path(args.output)

    if args.dry_run:
        print(f"{len(problems)} problems, schedule {schedule}")
        for problem in problems[:4]:
            print(f"--- {problem['key']} [{problem['style']}] "
                  f"truth={problem['ground_truth']!r}")
            print(problem["problem"][:300])
        print(f"effort body sample: {effort_body(schedule[0], args.effort_key)}")
        skipped = finished_keys(output, retry_exhausted=args.retry_exhausted)
        print(f"resume would skip {len(skipped)} finished keys")
        return

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        parser.error(
            f"no API key in ${args.api_key_env}; export it before launching"
        )
    done = finished_keys(output, retry_exhausted=args.retry_exhausted)
    pending = [problem for problem in problems if problem["key"] not in done]
    print(
        f"{len(pending)} pending of {len(problems)} problems "
        f"({len(done)} already finished); budget ${args.budget_usd:.2f}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    meter = CostMeter(
        args.budget_usd, args.price_in_per_mtok, args.price_out_per_mtok
    )
    client = TeacherClient(
        args.base_url, api_key, args.model, args.timeout, meter
    )
    canary = YieldCanary(args.canary_min_resolved, args.canary_min_yield)
    tokenizer = GPT2BPETokenizer()
    write_lock = threading.Lock()
    completed = 0

    def write(record: dict) -> None:
        nonlocal completed
        with write_lock:
            with output.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record.get("correct") or record.get("exhausted"):
                completed += 1
                if completed % 50 == 0:
                    print(
                        f"{completed}/{len(pending)} resolved, "
                        f"${meter.spent_usd():.2f} spent",
                        flush=True,
                    )

    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        futures = [
            pool.submit(
                run_problem,
                problem, client, schedule, args, meter, tokenizer, write, canary,
            )
            for problem in pending
        ]
        for future in as_completed(futures):
            # Fail fast: a systematic error (bad effort key, dead
            # provider, usage contract) must not drain the whole queue
            # before surfacing (red-team finding 7).
            future.result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True)
        print(
            f"spend: {meter.prompt_tokens} prompt + "
            f"{meter.completion_tokens} completion tokens, "
            f"${meter.spent_usd():.2f} of ${meter.budget_usd:.2f}"
            + (" (BUDGET EXHAUSTED)" if meter.exhausted() else ""),
            flush=True,
        )
    print(f"done: {completed} of {len(pending)} pending problems resolved")
    aborted = canary.aborted_styles()
    if aborted:
        # A canary abort must be unmissable in queued/log-scraped runs
        # (red-team round 2, NEW-3): fail the process, not just a line.
        print(
            f"CANARY ABORTED STYLES: {sorted(aborted)} — fix the format "
            "contract and rerun with --retry-exhausted",
            flush=True,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()

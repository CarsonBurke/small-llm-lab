#!/usr/bin/env python3
"""Render saved MiniCPM rollout responses as a standalone HTML transcript.

Reads TensorBoard text only; no model, CUDA, or checkpoint loading. Re-run the
command to refresh a live run's latest saved examples.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import html
from pathlib import Path
import re
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.types_pb2 import DT_STRING

from postraining.benchmark_report import _atomic_write_text
from postraining.minicpm_tensorboard_schema import (
    LEGACY_TEXT_TAG_MAP,
    TEXT_SUMMARY_SUFFIX,
)


CSS = "\nbody{overflow-wrap:anywhere}\n:root{color-scheme:dark;--page:#0d1117;--surface:#161b22;--surface-muted:#11161d;--ink:#e6edf3;--muted:#9aa7b5;--line:#303846;--link:#79b8ff;--correct:#3fb950;--incorrect:#f85149}*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:17px/1.65 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:36px 24px 80px}h1{font-size:clamp(27px,4vw,40px);line-height:1.2;margin-bottom:12px}h2{font-size:26px;margin:0 0 20px}h3{font-size:15px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 12px}nav{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0 36px}a{color:var(--link);text-underline-offset:3px}nav a{background:var(--surface);border:1px solid var(--line);padding:7px 14px;border-radius:8px}section{margin-bottom:48px;scroll-margin-top:20px}.question{padding:24px;background:var(--surface);border:1px solid var(--line);border-radius:12px;margin-bottom:16px}.prose{white-space:pre-wrap;overflow-wrap:anywhere;tab-size:4}.reference{margin:20px 0 0;padding-top:16px;border-top:1px solid var(--line)}.responses{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:18px;margin-bottom:24px}.response{min-width:0;background:var(--surface);border:1px solid var(--line);border-top:4px solid;border-radius:10px;padding:22px}.response.correct{border-top-color:var(--correct)}.response.incorrect{border-top-color:var(--incorrect)}.response header{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:18px}.response header span,.note{color:var(--muted);font-size:15px}.correct header strong{color:#7ee787}.incorrect header strong{color:#ff7b72}details{margin:16px 0;border:1px solid var(--line);border-radius:10px;padding:16px;background:var(--surface-muted)}summary{cursor:pointer;font-size:20px;font-weight:650;padding:4px}details[open] summary{margin-bottom:20px}a:focus-visible,summary:focus-visible{outline:3px solid #58a6ff;outline-offset:4px}footer{margin-top:36px;color:var(--muted);font-size:14px}@media(max-width:600px){main{padding:24px 14px}.question,.response{padding:16px}}@media print{body{background:white;color:#182334;font-size:11pt}main{max-width:none;padding:0}nav{display:none}.responses{display:block}.response{break-inside:avoid;margin-bottom:14px}section{break-before:page}h2,h3{break-after:avoid}}\n"

_SAMPLE_PREFIX = re.compile(
    r"Reward: (?P<reward>-?1)\n\nResponse tokens: (?P<tokens>\d+)\n\n"
)
_PROMPT_HEADER = "Prompt:\n"
_GROUND_TRUTH_HEADER = "\n\nGround truth:\n"
_MODEL_RESPONSE_HEADER = "\n\nModel response:\n"
_MIDDLE_TRUNCATION = "[... middle truncated ...]"


@dataclass(frozen=True)
class Sample:
    step: int
    label: str
    tokens: int
    prompt: str
    truth: str
    response: str
    wall_time: float
    middle_truncated: bool


def parse_sample(text: str, *, step: int, label: str, wall_time: float) -> Sample:
    prefix = _SAMPLE_PREFIX.match(text)
    if prefix is None:
        raise ValueError(f"step {step} {label}: unrecognized saved sample headers")
    fields = prefix.groupdict()
    if (fields["reward"] == "1") != (label == "correct"):
        raise ValueError(f"step {step}: sample label disagrees with saved reward")
    prompt_start = text.find(_PROMPT_HEADER, prefix.end())
    if prompt_start < 0:
        raise ValueError(f"step {step} {label}: unrecognized saved sample headers")
    payload = text[prompt_start + len(_PROMPT_HEADER) :]
    if _GROUND_TRUTH_HEADER in payload and _MODEL_RESPONSE_HEADER in payload:
        prompt, remainder = payload.split(_GROUND_TRUTH_HEADER, 1)
        truth, response = remainder.split(_MODEL_RESPONSE_HEADER, 1)
        middle_truncated = _MIDDLE_TRUNCATION in text
    elif _MIDDLE_TRUNCATION in payload:
        prompt, response = payload.split(_MIDDLE_TRUNCATION, 1)
        prompt = prompt.rstrip() + "\n\n" + _MIDDLE_TRUNCATION
        response = response.lstrip()
        truth = "[omitted from truncated TensorBoard sample]"
        middle_truncated = True
    else:
        raise ValueError(f"step {step} {label}: unrecognized saved sample headers")
    return Sample(
        step=step,
        label=label,
        tokens=int(fields["tokens"]),
        prompt=prompt,
        truth=truth,
        response=response,
        wall_time=wall_time,
        middle_truncated=middle_truncated,
    )


def load_samples(event_directory: Path) -> list[Sample]:
    if not event_directory.is_dir():
        raise ValueError(f"TensorBoard directory does not exist: {event_directory}")
    events = EventAccumulator(str(event_directory), size_guidance={"tensors": 0})
    events.Reload()
    samples: dict[tuple[int, str], Sample] = {}
    for tag in events.Tags()["tensors"]:
        logical = tag.removesuffix(TEXT_SUMMARY_SUFFIX)
        logical = LEGACY_TEXT_TAG_MAP.get(logical, logical)
        if logical not in {"samples/rollout_correct", "samples/rollout_incorrect"}:
            continue
        label = logical.removeprefix("samples/rollout_")
        for event in events.Tensors(tag):
            tensor = event.tensor_proto
            if tensor.dtype != DT_STRING or len(tensor.string_val) != 1:
                raise ValueError(
                    f"step {event.step}: expected one text value for {tag}"
                )
            sample = parse_sample(
                tensor.string_val[0].decode("utf-8"),
                step=event.step,
                label=label,
                wall_time=event.wall_time,
            )
            key = (sample.step, sample.label)
            previous = samples.get(key)
            if previous is None or sample.wall_time >= previous.wall_time:
                samples[key] = sample
    if not samples:
        raise ValueError(f"no saved rollout response samples in {event_directory}")
    return sorted(samples.values(), key=lambda sample: (-sample.step, sample.label))


def parse_steps(value: str) -> set[int]:
    steps: set[int] = set()
    for item in value.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", item.strip())
        if match is None:
            raise argparse.ArgumentTypeError(
                "use step numbers or inclusive ranges, e.g. 44,64-84"
            )
        start = int(match[1])
        stop = int(match[2]) if match[2] is not None else start
        if stop < start:
            raise argparse.ArgumentTypeError("step ranges must be increasing")
        steps.update(range(start, stop + 1))
    return steps


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def render_step(samples: Sequence[Sample]) -> str:
    groups: dict[tuple[str, str], list[Sample]] = {}
    for sample in samples:
        groups.setdefault((sample.prompt, sample.truth), []).append(sample)
    parts: list[str] = []
    for (prompt, truth), group in groups.items():
        parts.append(
            '<div class="question"><h3>Question</h3><div class="prose">'
            + html.escape(prompt)
            + '</div><p class="reference">Reference answer: <strong>'
            + html.escape(truth)
            + '</strong></p></div><div class="responses">'
        )
        for sample in sorted(group, key=lambda item: item.label):
            parts.append(
                f'<article class="response {sample.label}"><header>'
                f"<strong>{sample.label.capitalize()}</strong>"
                f"<span>{sample.tokens:,} tokens</span></header>"
                '<div class="prose">' + html.escape(sample.response) + "</div>"
            )
            if sample.middle_truncated:
                parts.append(
                    '<p class="note">The original TensorBoard sample omitted its middle; '
                    "the original omission marker is preserved.</p>"
                )
            parts.append("</article>")
        parts.append("</div>")
    return "".join(parts)


def render_report(
    samples: Sequence[Sample],
    *,
    run: Path,
    last: int,
    compare_steps: set[int],
) -> str:
    by_step: dict[int, list[Sample]] = {}
    for sample in samples:
        by_step.setdefault(sample.step, []).append(sample)
    available = sorted(by_step, reverse=True)
    latest = available[:last]
    missing = compare_steps - by_step.keys()
    if missing:
        raise ValueError(
            "comparison steps have no saved samples: "
            + ", ".join(map(str, sorted(missing)))
        )
    comparisons = sorted(compare_steps - set(latest), reverse=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    heading = str(latest[0]) if len(latest) == 1 else f"{latest[-1]}–{latest[0]}"
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>MiniCPM training — response samples</title><style>",
        CSS,
        "</style></head><body><main><h1>MiniCPM training — response samples</h1>"
        "<p>Questions, reference answers, and full saved model responses. "
        f"Latest examples: <strong>steps {heading}</strong>.</p>"
        '<p class="note">Run: '
        + html.escape(run.name)
        + f" · Snapshot: {timestamp}</p>"
        '<p class="note">These are the first correct and first incorrect examples saved per '
        "rollout, not random samples. Different questions within one step remain separate. "
        "Only saved text is shown; a live writer may not have flushed every example yet. "
        "Correctness refers to the final-answer verifier, not the validity of every reasoning "
        "step. Original mathematical notation is preserved.</p>"
        '<nav aria-label="Response sections">',
    ]
    parts.extend(f'<a href="#step-{step}">Step {step}</a>' for step in latest)
    if comparisons:
        parts.append('<a href="#earlier">Comparison samples</a>')
    parts.append("</nav>")
    for step in latest:
        parts.append(f'<section id="step-{step}"><h2>Step {step}</h2>')
        parts.append(render_step(by_step[step]))
        parts.append("</section>")
    if comparisons:
        parts.append(
            '<h2 id="earlier">Comparison samples</h2>'
            '<p class="note">Expand a step to read the full saved examples.</p>'
        )
        for step in comparisons:
            parts.append(f'<details id="step-{step}"><summary>Step {step}</summary>')
            parts.append(render_step(by_step[step]))
            parts.append("</details>")
    parts.append(
        "<footer>Static snapshot from "
        + html.escape(str(run))
        + ". Refresh by re-running scripts/render_minicpm_responses.py. "
        "No model responses regenerated or rewritten.</footer></main></body></html>"
    )
    return "".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="run directory containing tensorboard/, or an event directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="HTML destination, replaced on refresh (default: RUN/responses.html)",
    )
    parser.add_argument(
        "--last",
        type=positive_integer,
        default=4,
        help="number of latest saved rollout steps to show (default: 4)",
    )
    parser.add_argument(
        "--compare-steps",
        type=parse_steps,
        default=set(),
        help="additional collapsible steps, e.g. 44,64-84",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run = args.run.expanduser().resolve()
    events = run / "tensorboard" if (run / "tensorboard").is_dir() else run
    if events == run and run.name == "tensorboard":
        run = run.parent
    output = (args.output or run / "responses.html").expanduser().resolve()
    if output.suffix.lower() != ".html":
        parser.error("--output must be an .html file")
    try:
        samples = load_samples(events)
        document = render_report(
            samples, run=run, last=args.last, compare_steps=args.compare_steps
        )
        _atomic_write_text(output, document)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    latest_step = max(sample.step for sample in samples)
    print(f"Updated {output} — latest saved rollout step {latest_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

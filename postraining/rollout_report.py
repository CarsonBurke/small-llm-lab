"""Human-readable transcripts of training rollouts.

The periodic benchmark (``benchmark_report``) shows held-out answers; this
module shows what the policy is actually being trained on. Every
``--rollout-sample-every`` actor steps the KDA trainer arms a
``RolloutSampleRecorder`` for one pool, which keeps one correct and one
incorrect trajectory of every RL source as the pool is scored. Each
capture lands in ``<run>/rollout_samples/step_XXXXXX.json`` and the HTML
report ``<run>/rollout_samples.html`` is re-rendered from the latest
captures.

The same renderer reads MiniCPM runs, whose trainer saves the first correct
and first incorrect sample per rollout as TensorBoard text
(``samples/rollout_{correct,incorrect}``); no model is loaded either way:

    .venv/bin/python -m postraining.rollout_report --run postraining/runs/<run>
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from postraining.benchmark_report import (
    _LEGEND,
    STREAM_CSS,
    _atomic_write_text,
    _fence_pairs_broken,
    _render_stream,
)
from postraining.core import emitted_display_segments
from postraining.latent_rollout import LatentRolloutBatch, emitted_token_rows


ROLLOUT_SAMPLE_SCHEMA = "training_rollout_samples/v1"
SAMPLE_DIRECTORY = "rollout_samples"
REPORT_NAME = "rollout_samples.html"
# Captures the trainer's re-rendered report shows; older ones stay on disk.
REPORT_CAPTURES = 8
LABELS = ("correct", "incorrect")


def capture_due(step: int, pool_updates: int, every: int) -> bool:
    """Whether a pool covering actor steps [step, step + pool_updates) is due.

    True when any step in the pool is a multiple of ``every``, so step 0 is
    always captured and a pool spanning several updates is never skipped
    over.
    """
    if every <= 0:
        return False
    return (-step) % every < pool_updates


@dataclass(frozen=True)
class RowVerdict:
    """The reward gate's decision for one scored row.

    Scorers return one per row so transcripts report what the reward
    actually decided instead of re-deriving it. ``parsed_answer`` is the
    verifier's own prediction for the field it graded (as ``latent_eval``
    reports it); a gate-zeroed row carries its counterfactual's prediction —
    the value the gate-zeroed-correct alarm scored — and an unterminated
    row None.
    """

    format_ok: bool
    parsed_answer: str | None


def _pick_key(step: int, prompt: str, row: int) -> bytes:
    return hashlib.blake2b(
        f"{step}\0{prompt}\0{row}".encode(), digest_size=16
    ).digest()


class RolloutSampleRecorder:
    """One correct and one incorrect trajectory per source, per pool.

    ``offer`` runs wherever the trainer scores a group — the collection
    thread or its scoring worker — so state is lock-guarded. Within each
    (source, label) slot the trajectory with the smallest hash of (step,
    prompt, row) wins: a uniform pick over the pool's matching trajectories
    that is deterministic and independent of scoring order. (The collector
    scores groups shortest prompt first, so first-match picks would always
    show the pool's easiest problems.)

    ``solution_prefix_ids`` are the teacher-forced solution tokens at the
    end of the prompt (the none-mode ``Answer:`` prefix); transcripts show
    them as a prefix segment, since the verifier reads them with the
    emission.
    """

    def __init__(
        self,
        tokenizer,
        stop_ids: Sequence[int],
        sources: Sequence[str],
        solution_prefix_ids: Sequence[int],
    ):
        self._tokenizer = tokenizer
        self._stop_ids = frozenset(int(token) for token in stop_ids)
        self._sources = tuple(sources)
        self._prefix_ids = [int(token) for token in solution_prefix_ids]
        self._prefix_segments = (
            emitted_display_segments(self._prefix_ids, tokenizer, kind="prefix")
            if self._prefix_ids
            else []
        )
        self._lock = threading.Lock()
        self._step: int | None = None
        self._metrics_step: int | None = None
        self._picked: dict[tuple[str, str], tuple[bytes, dict[str, Any]]] = {}
        self._offered: dict[str, int] = {}

    @property
    def armed(self) -> bool:
        return self._step is not None

    def begin(self, step: int, metrics_step: int) -> None:
        """Arm for the pool collected after ``step`` actor updates.

        ``metrics_step`` is where the trainer logs that pool's rollout
        metrics, so the report can point from a transcript to its curves.
        """
        with self._lock:
            if self._step is not None:
                raise RuntimeError("rollout sample capture is already armed")
            if metrics_step < step:
                raise ValueError("metrics_step precedes the capture step")
            self._step = int(step)
            self._metrics_step = int(metrics_step)
            self._picked = {}
            self._offered = {}

    def offer(
        self,
        batch: LatentRolloutBatch,
        verdicts: Sequence[RowVerdict],
        prompt: str,
        ground_truth: str,
        source: str,
    ) -> None:
        with self._lock:
            if self._step is None:
                return
            if source not in self._sources:
                raise ValueError(f"unregistered rollout source {source!r}")
            rewards = [float(value) for value in batch.reward_scalar.tolist()]
            if len(verdicts) != len(rewards):
                raise ValueError(
                    f"{len(verdicts)} verdicts for a {len(rewards)}-row group"
                )
            self._offered[source] = self._offered.get(source, 0) + 1
            # Exact verifier success; bounded numeric proximity (< 1) stays
            # incorrect, as the pool's accuracy metrics count it.
            correct = [reward == 1.0 for reward in rewards]
            rows = None
            for label in LABELS:
                candidates = [
                    (_pick_key(self._step, prompt, row), row)
                    for row, is_correct in enumerate(correct)
                    if is_correct == (label == "correct")
                ]
                if not candidates:
                    continue
                key, index = min(candidates)
                held = self._picked.get((source, label))
                if held is not None and held[0] <= key:
                    continue
                if rows is None:
                    rows = emitted_token_rows(batch)
                self._picked[(source, label)] = (
                    key,
                    self._describe(
                        rows[index],
                        verdicts[index],
                        source=source,
                        label=label,
                        prompt=prompt,
                        ground_truth=ground_truth,
                        reward=rewards[index],
                        group_correct=sum(correct),
                        group_size=len(rewards),
                        group_mean_reward=sum(rewards) / len(rewards),
                    ),
                )

    def _describe(
        self, emitted: list[int], verdict: RowVerdict, **fields: Any
    ) -> dict[str, Any]:
        stop_cut = next(
            (
                index
                for index, token in enumerate(emitted)
                if token in self._stop_ids
            ),
            None,
        )
        terminated = stop_cut is not None
        visible = emitted[: stop_cut + 1] if terminated else emitted
        return {
            **fields,
            "terminated": terminated,
            "structural_format_ok": bool(verdict.format_ok),
            "parsed_answer": verdict.parsed_answer,
            "emitted_token_count": len(emitted),
            # What the verifier decoded: the teacher-forced prefix, then the
            # emission through its stop token.
            "emitted_text": self._tokenizer.decode(self._prefix_ids + visible),
            "emitted_segments": [
                *self._prefix_segments,
                *emitted_display_segments(emitted, self._tokenizer),
            ],
            "middle_truncated": False,
        }

    def finish(self) -> dict[str, Any]:
        """Disarm and return the capture payload."""
        with self._lock:
            if self._step is None:
                raise RuntimeError("rollout sample capture was never armed")
            samples = [
                self._picked[(source, label)][1]
                for source in self._sources
                for label in LABELS
                if (source, label) in self._picked
            ]
            payload = {
                "schema": ROLLOUT_SAMPLE_SCHEMA,
                "step": self._step,
                "metrics_step": self._metrics_step,
                "sources": list(self._sources),
                "groups_offered": {
                    source: self._offered.get(source, 0)
                    for source in self._sources
                },
                "samples": samples,
            }
            self._step = None
            self._metrics_step = None
            self._picked = {}
            self._offered = {}
            return payload


def _capture_paths(output: Path) -> list[tuple[int, Path]]:
    directory = output / SAMPLE_DIRECTORY
    if not directory.is_dir():
        return []
    paths = []
    for path in directory.glob("step_*.json"):
        match = re.fullmatch(r"step_(\d+)", path.stem)
        if match is not None:
            paths.append((int(match[1]), path))
    return sorted(paths)


def load_json_captures(output: Path) -> list[dict[str, Any]]:
    captures = []
    for step, path in _capture_paths(output):
        payload = json.loads(path.read_text())
        if payload.get("schema") != ROLLOUT_SAMPLE_SCHEMA:
            raise ValueError(
                f"{path}: schema {payload.get('schema')!r} is not "
                f"{ROLLOUT_SAMPLE_SCHEMA!r}"
            )
        if int(payload["step"]) != step:
            raise ValueError(f"{path}: payload step {payload['step']} != {step}")
        captures.append(payload)
    return captures


def _write_report(output: Path, captures: Sequence[Mapping[str, Any]]) -> Path:
    report = output / REPORT_NAME
    if captures:
        _atomic_write_text(
            report,
            render_rollout_report(
                captures, run=output, last=REPORT_CAPTURES, compare_steps=set()
            ),
        )
    elif report.exists():
        report.unlink()
    return report


def write_rollout_samples(output: str | Path, payload: Mapping[str, Any]) -> Path:
    """Persist one capture and re-render the run's report."""
    output = Path(output)
    step = int(payload["step"])
    _atomic_write_text(
        output / SAMPLE_DIRECTORY / f"step_{step:06d}.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    return _write_report(output, load_json_captures(output))


def purge_rollout_samples_from(output: str | Path, step: int) -> int:
    """Drop captures at or after a resume step and re-render the report.

    A capture at step s describes the pool collected before update s. A run
    resumed at s re-collects that pool, so the stale captures would describe
    trajectories the continuing run never trained on.
    """
    output = Path(output)
    removed = 0
    for capture_step, path in _capture_paths(output):
        if capture_step >= step:
            path.unlink()
            removed += 1
    if removed or (output / REPORT_NAME).exists():
        _write_report(output, load_json_captures(output))
    return removed


# MiniCPM TensorBoard text samples ------------------------------------------

_SAMPLE_PREFIX = re.compile(
    r"Reward: (?P<reward>-?1)\n\nResponse tokens: (?P<tokens>\d+)\n\n"
)
_PROMPT_HEADER = "Prompt:\n"
_GROUND_TRUTH_HEADER = "\n\nGround truth:\n"
_MODEL_RESPONSE_HEADER = "\n\nModel response:\n"
_MIDDLE_TRUNCATION = "[... middle truncated ...]"


def parse_tensorboard_sample(text: str, *, step: int, label: str) -> dict[str, Any]:
    """One MiniCPM ``samples/rollout_*`` text summary as a normalized sample."""
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
    return {
        "source": "rollout",
        "label": label,
        "prompt": prompt,
        "ground_truth": truth,
        # MiniCPM saves +1/-1 verifier rewards; the report shows them as is.
        "reward": float(fields["reward"]),
        "terminated": None,
        "structural_format_ok": None,
        "parsed_answer": None,
        "emitted_token_count": int(fields["tokens"]),
        "emitted_text": response,
        "emitted_segments": None,
        "group_correct": None,
        "group_size": None,
        "group_mean_reward": None,
        "middle_truncated": middle_truncated,
    }


def load_tensorboard_captures(event_directory: Path) -> list[dict[str, Any]]:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
    from tensorboard.compat.proto.types_pb2 import DT_STRING

    from postraining.minicpm_tensorboard_schema import (
        LEGACY_TEXT_TAG_MAP,
        TEXT_SUMMARY_SUFFIX,
    )

    events = EventAccumulator(str(event_directory), size_guidance={"tensors": 0})
    events.Reload()
    latest: dict[tuple[int, str], tuple[float, dict[str, Any]]] = {}
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
            sample = parse_tensorboard_sample(
                tensor.string_val[0].decode("utf-8"), step=event.step, label=label
            )
            # A resumed writer can re-log a step; the newest write wins.
            key = (event.step, label)
            if key not in latest or event.wall_time >= latest[key][0]:
                latest[key] = (event.wall_time, sample)
    by_step: dict[int, list[dict[str, Any]]] = {}
    for (step, _), (_, sample) in sorted(latest.items()):
        by_step.setdefault(step, []).append(sample)
    return [
        {
            "schema": "minicpm_tensorboard_rollout_samples",
            "step": step,
            "sources": ["rollout"],
            "groups_offered": None,
            "samples": samples,
        }
        for step, samples in sorted(by_step.items())
    ]


# Rendering -------------------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _render_sample(sample: Mapping[str, Any]) -> str:
    label = str(sample["label"])
    if label not in LABELS:
        raise ValueError(f"unknown sample label {label!r}")
    segments = sample.get("emitted_segments") or []
    fence_broken = _fence_pairs_broken(segments)
    badges = []
    if sample.get("reward") is not None:
        badges.append(f'<span class="badge">reward {float(sample["reward"]):g}</span>')
    if sample.get("terminated") is not None:
        badges.append(
            '<span class="badge">'
            + ("terminated" if sample["terminated"] else "unterminated")
            + "</span>"
        )
    if sample.get("structural_format_ok") is False:
        badges.append('<span class="badge incorrect">format gate failed</span>')
    if fence_broken:
        badges.append('<span class="badge incorrect">broken fences</span>')
    stats = [f"<div><dt>Tokens</dt><dd>{int(sample['emitted_token_count']):,}</dd></div>"]
    if sample.get("group_size"):
        stats.append(
            "<div><dt>Group correct</dt><dd>"
            f"{int(sample['group_correct'])}/{int(sample['group_size'])}"
            f" · mean reward {float(sample['group_mean_reward']):.3f}</dd></div>"
        )
    parsed = sample.get("parsed_answer")
    parsed_answer = (
        '<div><dt>Parsed answer</dt><dd><code>'
        + _esc(parsed if parsed is not None else "none")
        + "</code></dd></div>"
        if sample.get("terminated") is not None
        else ""
    )
    note = (
        '<p class="note">The saved sample omitted its middle; the original '
        "omission marker is preserved.</p>"
        if sample.get("middle_truncated")
        else ""
    )
    return f"""
        <article class="attempt {label}">
          <header><h3>{label.capitalize()}</h3><div class="badges">{''.join(badges)}</div></header>
          <dl class="answers"><div><dt>Expected answer</dt><dd><code>{_esc(sample['ground_truth'])}</code></dd></div>{parsed_answer}</dl>
          <details class="prompt" open><summary>Prompt</summary><pre>{_esc(sample['prompt'])}</pre></details>
          <div class="response"><h4>Model response</h4><div class="stream">{_render_stream(sample, tint=not fence_broken)}</div></div>
          <dl class="stats">{''.join(stats)}</dl>{note}
        </article>"""


def _render_capture(capture: Mapping[str, Any]) -> str:
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for sample in capture["samples"]:
        by_source.setdefault(str(sample["source"]), []).append(sample)
    offered = capture.get("groups_offered") or {}
    parts = []
    for source in capture["sources"]:
        samples = sorted(
            by_source.get(source, []), key=lambda item: LABELS.index(item["label"])
        )
        found = {sample["label"] for sample in samples}
        missing = [label for label in LABELS if label not in found]
        groups = offered.get(source)
        heading_note = f"{groups} groups in this pool" if groups is not None else ""
        if missing and groups:
            heading_note += " · no " + " or ".join(missing) + " trajectory"
        parts.append(
            f'<div class="source"><h3>{_esc(source)}'
            f'<span class="note">{_esc(heading_note)}</span></h3>'
            f'<div class="attempt-grid">{"".join(_render_sample(s) for s in samples)}</div></div>'
        )
    return "".join(parts)


def _metrics_note(capture: Mapping[str, Any]) -> str:
    metrics_step = capture.get("metrics_step")
    if metrics_step is None:
        return ""
    return (
        f' <span class="note">pool metrics logged at step {int(metrics_step)}</span>'
    )


def render_rollout_report(
    captures: Sequence[Mapping[str, Any]],
    *,
    run: Path,
    last: int,
    compare_steps: set[int],
) -> str:
    """Latest ``last`` captures in full, ``compare_steps`` collapsed below."""
    if not captures:
        raise ValueError("no rollout sample captures to render")
    by_step = {int(capture["step"]): capture for capture in captures}
    missing = compare_steps - by_step.keys()
    if missing:
        raise ValueError(
            "comparison steps have no saved samples: "
            + ", ".join(map(str, sorted(missing)))
        )
    latest = sorted(by_step, reverse=True)[:last]
    comparisons = sorted(compare_steps - set(latest), reverse=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    any_segments = any(
        sample.get("emitted_segments")
        for capture in captures
        for sample in capture["samples"]
    )
    nav = "".join(f'<a href="#step-{step}">Step {step}</a>' for step in latest)
    if comparisons:
        nav += '<a href="#earlier">Comparison steps</a>'
    sections = "".join(
        f'<section id="step-{step}"><h2>Step {step}{_metrics_note(by_step[step])}</h2>'
        f"{_render_capture(by_step[step])}</section>"
        for step in latest
    )
    if comparisons:
        sections += '<h2 id="earlier">Comparison steps</h2>' + "".join(
            f'<details id="step-{step}" class="earlier"><summary>Step {step}</summary>'
            f"{_render_capture(by_step[step])}</details>"
            for step in comparisons
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Training rollouts · {_esc(run.name)}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b0e14; --panel:#151c29; --panel-2:#101620; --line:#303b50; --text:#e8edf7; --muted:#9ca9bc; --good:#47d18c; --bad:#ff7885; --accent:#a6c4ff; --think:#c4a7ff; --answer:#6fe3ae; --eos:#ffab70; --think-bg:#c4a7ff12; --answer-bg:#47d18c12; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; overflow-wrap:anywhere; }}
    main {{ width:min(1100px,calc(100% - 2rem)); margin:0 auto; padding:2.5rem 0 5rem; }}
    h1 {{ margin:0 0 .4rem; font-size:clamp(1.8rem,4vw,2.8rem); letter-spacing:-.03em; }}
    h2 {{ margin:2.2rem 0 1rem; font-size:1.5rem; }}
    h3 {{ display:flex; flex-wrap:wrap; align-items:baseline; gap:.3rem .8rem; margin:0 0 .7rem; font-size:1rem; }}
    .note,.subtitle {{ color:var(--muted); font-size:.85rem; font-weight:400; }}
    nav {{ display:flex; flex-wrap:wrap; gap:.5rem; margin:1.2rem 0; }}
    nav a {{ color:var(--accent); padding:.3rem .7rem; border:1px solid var(--line); border-radius:8px; background:var(--panel); text-decoration:none; }}
    {STREAM_CSS}
    .guide {{ margin:0 0 1.5rem; }}
    .guide > summary {{ font-size:.85rem; }}
    .guide .legend {{ margin-top:.75rem; }}
    .source {{ margin:1.6rem 0 2.5rem; }}
    .source > h3 {{ padding-bottom:.6rem; border-bottom:1px solid var(--line); font-size:1.15rem; }}
    .attempt-grid {{ display:grid; gap:1rem; }}
    .attempt {{ min-width:0; padding:1.3rem 1.5rem; border:1px solid var(--line); border-left:4px solid var(--bad); border-radius:12px; background:var(--panel); }}
    .attempt.correct {{ border-left-color:var(--good); }}
    .attempt header {{ display:flex; justify-content:space-between; align-items:center; gap:.8rem; margin-bottom:1rem; }}
    .attempt h3 {{ margin:0; font-size:1.1rem; }}
    .badges {{ display:flex; flex-wrap:wrap; justify-content:flex-end; gap:.35rem; }}
    .badge {{ padding:.14rem .46rem; border:1px solid var(--line); border-radius:999px; color:var(--muted); font-size:.72rem; white-space:nowrap; }}
    .badge.incorrect {{ border-color:#ff6b7a88; color:var(--bad); }}
    code {{ color:#c3e88d; }}
    summary {{ cursor:pointer; color:var(--accent); font-weight:650; }}
    pre {{ margin:.55rem 0 0; padding:1rem 1.15rem; white-space:pre-wrap; overflow-wrap:anywhere; border:1px solid var(--line); border-radius:9px; background:var(--panel-2); color:var(--text); font:14px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace; }}
    .answers {{ display:flex; flex-wrap:wrap; gap:.7rem 2rem; margin:0 0 1.1rem; padding:.8rem 1rem; border-radius:9px; background:var(--panel-2); }}
    .answers > div {{ min-width:8rem; }}
    .answers dd {{ font-size:1.05rem; }}
    .prompt {{ margin:0 0 1.2rem; }}
    h4 {{ margin:0 0 .55rem; color:var(--muted); font-size:.75rem; font-weight:650; text-transform:uppercase; letter-spacing:.07em; }}
    .response .stream {{ padding:1.1rem 1.2rem; border:1px solid var(--line); background:var(--panel-2); font-size:14px; line-height:1.7; }}
    .response .tok {{ margin:0 .1rem; }}
    .stats {{ display:flex; flex-wrap:wrap; gap:.5rem 1.5rem; margin:1rem 0 0; padding-top:.8rem; border-top:1px solid var(--line); }}
    dt {{ color:var(--muted); font-size:.7rem; text-transform:uppercase; letter-spacing:.06em; }}
    dd {{ margin:0; font-variant-numeric:tabular-nums; }}
    details.earlier {{ margin:1rem 0; padding:.8rem 1rem; border:1px solid var(--line); border-radius:12px; }}
    footer {{ margin-top:2rem; color:var(--muted); font-size:.85rem; }}
    @media (max-width:500px) {{ main {{ width:calc(100% - 1.5rem); padding-top:1.5rem; }} .attempt {{ padding:1rem; }} .attempt header {{ display:block; }} .badges {{ justify-content:flex-start; margin-top:.5rem; }} .response .stream {{ padding:.8rem; }} }}
  </style>
</head>
<body>
<main>
  <h1>Training rollouts</h1>
  <p class="subtitle">{_esc(run.name)} · snapshot {timestamp}</p>
  <p class="note">Up to one correct and one incorrect training example per source and step. These samples do not measure held-out accuracy.</p>
  <nav aria-label="Captured steps">{nav}</nav>
  <details class="guide"><summary>How samples are selected and graded</summary>
    <p class="note">Each sample is selected uniformly by a hash of (step, prompt, row), so reruns choose the same one. Correct means exact verifier reward 1; bounded numeric proximity counts as incorrect. The parsed answer is the verifier's prediction for the graded field.</p>
    {_LEGEND if any_segments else ''}
  </details>
  {sections}
  <footer>Static snapshot of {_esc(run)}. Refresh with <code>python -m postraining.rollout_report --run {_esc(run)}</code>.</footer>
</main>
</body>
</html>
"""


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


def load_captures(run: Path) -> list[dict[str, Any]]:
    """KDA JSON captures when present, else MiniCPM TensorBoard text."""
    if (run / SAMPLE_DIRECTORY).is_dir():
        captures = load_json_captures(run)
    else:
        events = run / "tensorboard" if (run / "tensorboard").is_dir() else run
        captures = load_tensorboard_captures(events)
    if not captures:
        raise ValueError(f"no saved rollout samples under {run}")
    return captures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="run directory")
    parser.add_argument(
        "--output",
        type=Path,
        help=f"HTML destination, replaced on refresh (default: RUN/{REPORT_NAME})",
    )
    parser.add_argument(
        "--last",
        type=positive_integer,
        default=4,
        help="number of latest captured steps shown in full (default: 4)",
    )
    parser.add_argument(
        "--compare-steps",
        type=parse_steps,
        default=set(),
        help="additional collapsible steps, e.g. 0,250-500",
    )
    args = parser.parse_args(argv)
    run = args.run.expanduser().resolve()
    if not run.is_dir():
        parser.error(f"run directory does not exist: {run}")
    # A MiniCPM event directory names its run by its parent.
    if run.name == "tensorboard":
        run = run.parent
    output = (args.output or run / REPORT_NAME).expanduser().resolve()
    if output.suffix.lower() != ".html":
        parser.error("--output must be an .html file")
    try:
        captures = load_captures(run)
        document = render_rollout_report(
            captures, run=run, last=args.last, compare_steps=args.compare_steps
        )
        _atomic_write_text(output, document)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Updated {output} — latest captured step {captures[-1]['step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

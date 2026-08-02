"""Persistent, human-readable samples from the periodic math benchmark."""

from __future__ import annotations

import argparse
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


CAPTURE_PROBLEMS = 4
CAPTURE_SAMPLES_PER_PROBLEM = 4
CAPTURE_ATTEMPTS = CAPTURE_PROBLEMS * CAPTURE_SAMPLES_PER_PROBLEM
BENCHMARK_ANSWER_SCHEMA = "deterministic_hidden_carry_answers/v4"


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace ``path`` atomically so readers never observe partial reports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = file.name
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _percent(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def _validate_attempts(
    attempts: Sequence[Mapping[str, Any]],
    *,
    problem_count: int = CAPTURE_PROBLEMS,
    samples_per_problem: int = CAPTURE_SAMPLES_PER_PROBLEM,
) -> None:
    if problem_count <= 0 or samples_per_problem <= 0:
        raise ValueError("benchmark report dimensions must be positive")
    expected = {
        (problem, sample)
        for problem in range(problem_count)
        for sample in range(samples_per_problem)
    }
    observed = {
        (int(attempt["problem_index"]), int(attempt["sample_index"]))
        for attempt in attempts
    }
    if len(attempts) != len(expected) or observed != expected:
        if (
            problem_count == CAPTURE_PROBLEMS
            and samples_per_problem == CAPTURE_SAMPLES_PER_PROBLEM
        ):
            requirement = (
                "exactly the first four samples from the first four problems"
            )
        else:
            requirement = (
                f"a complete {problem_count}-problem by "
                f"{samples_per_problem}-sample grid"
            )
        raise ValueError(
            f"benchmark report requires {requirement}; got "
            f"{len(attempts)} attempts and coordinates {sorted(observed)!r}"
        )


# Fence roles toggle a background tint on the text between them so the
# think/answer structure reads at a glance; every other special role is a
# standalone chip. (role -> (tint class, opens?)).
_SEGMENT_SPAN_ROLES = {
    "think_open": ("in-think", True),
    "think_close": ("in-think", False),
    "answer_open": ("in-answer", True),
    "answer_close": ("in-answer", False),
}
# Stored artifacts are data, not code: any role outside the set the
# segmenter can produce renders as a generic chip, so a crafted JSON
# cannot smuggle extra CSS class tokens through the role field.
_KNOWN_ROLES = frozenset(_SEGMENT_SPAN_ROLES) | {"eos", "bos"}
_FENCE_PAIRS = (("think_open", "think_close"), ("answer_open", "answer_close"))


def _fence_pairs_broken(segments: Sequence[Any]) -> bool:
    """True when a fence pair is present but not exactly-once well-ordered.

    Mirrors the spirit of ``single_fence_span`` without duplicating the RL
    gate's anchoring rules: a tinted span must at least be THE span the
    grader would extract. Broken structure suppresses tinting so the report
    never dresses up fences the reward gate rejected.
    """
    positions: dict[str, list[int]] = {}
    for index, segment in enumerate(segments):
        if isinstance(segment, Mapping) and segment.get("kind") == "special":
            positions.setdefault(str(segment.get("role", "")), []).append(index)
    for open_role, close_role in _FENCE_PAIRS:
        opens = positions.get(open_role, [])
        closes = positions.get(close_role, [])
        if not opens and not closes:
            continue
        if len(opens) != 1 or len(closes) != 1 or closes[0] < opens[0]:
            return True
    return False


def _render_text_segment(text: str, classes: Sequence[str]) -> str:
    escaped = html.escape(str(text), quote=True)
    # Line breaks are content the model chose to emit; mark them so an
    # empty-looking line is distinguishable from wrapped text.
    escaped = escaped.replace("\n", '<span class="nl">&#9166;</span>\n')
    return f'<span class="{" ".join(classes)}">{escaped}</span>'


def _render_stream(attempt: Mapping[str, Any], tint: bool = True) -> str:
    """Render the emitted stream with every special token kept visible.

    Segments come pre-split on token ids (``emitted_display_segments``);
    chip rendering never pattern-matches decoded text, so model output
    that literally types "<think>" stays plain escaped text. Attempts
    from older artifacts or text-only sources fall back to the stripped
    ``emitted_text``. Malformed segment entries degrade to being skipped
    rather than failing the whole report.
    """
    segments = attempt.get("emitted_segments")
    if not segments:
        return _render_text_segment(attempt["emitted_text"], ["txt"])
    parts: list[str] = []
    open_tints: set[str] = set()
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        if segment.get("kind") == "special":
            role = str(segment.get("role", "special"))
            if role not in _KNOWN_ROLES:
                role = "special"
            span = _SEGMENT_SPAN_ROLES.get(role)
            if span is not None and span[1]:
                open_tints.add(span[0])
            chip_classes = f"tok tok-{role}"
            if segment.get("source") == "prefix":
                chip_classes += " prefix"
            parts.append(
                f'<span class="{chip_classes}">'
                f'{html.escape(str(segment.get("text", "")), quote=True)}</span>'
            )
            if span is not None and not span[1]:
                open_tints.discard(span[0])
        else:
            classes = ["txt"]
            if segment.get("kind") == "prefix":
                classes.append("prefix")
            if tint:
                classes.extend(sorted(open_tints))
            parts.append(
                _render_text_segment(str(segment.get("text", "")), classes)
            )
    return "".join(parts)


def _render_attempt(attempt: Mapping[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    state = "correct" if attempt["correct"] else "incorrect"
    terminated = "terminated" if attempt["terminated"] else "unterminated"
    segments = attempt.get("emitted_segments") or []
    fence_broken = _fence_pairs_broken(segments)
    broken_badge = (
        '\n            <span class="badge incorrect">broken fences</span>'
        if fence_broken
        else ""
    )
    return f"""
      <article class="attempt {state}">
        <header>
          <h3>Sample {int(attempt['sample_index']) + 1}</h3>
          <div class="badges">
            <span class="badge {state}">{state}</span>
            <span class="badge">{esc(terminated)}</span>{broken_badge}
          </div>
        </header>
        <div class="stream">{_render_stream(attempt, tint=not fence_broken)}</div>
        <dl class="stats">
          <div><dt>Parsed answer</dt><dd>{esc(attempt['parsed_answer'])}</dd></div>
          <div><dt>Grader</dt><dd>{esc(attempt['answer_style'])}</dd></div>
          <div><dt>Emitted tokens</dt><dd>{int(attempt['emitted_token_count'])}</dd></div>
        </dl>
      </article>"""


_LEGEND = """
    <div class="legend" aria-label="Token stream legend">
      <span class="legend-item"><span class="tok tok-think_open">&lt;think&gt;</span><span class="txt in-think">thinking span</span><span class="tok tok-think_close">&lt;/think&gt;</span></span>
      <span class="legend-item"><span class="tok tok-answer_open">&lt;answer&gt;</span><span class="txt in-answer">graded answer span</span><span class="tok tok-answer_close">&lt;/answer&gt;</span></span>
      <span class="legend-item"><span class="tok tok-eos">&lt;|endoftext|&gt;</span> stop token</span>
      <span class="legend-item"><span class="txt prefix">teacher-forced prefix</span></span>
      <span class="legend-item"><span class="nl">&#9166;</span> emitted line break</span>
    </div>"""


def render_benchmark_report(payload: Mapping[str, Any]) -> str:
    """Render a self-contained report for one periodic benchmark evaluation."""
    metrics = payload["metrics"]
    attempts = payload["attempts"]
    problem_count = int(payload.get("problem_count", CAPTURE_PROBLEMS))
    samples_per_problem = int(
        payload.get("samples_per_problem", CAPTURE_SAMPLES_PER_PROBLEM)
    )
    _validate_attempts(
        attempts,
        problem_count=problem_count,
        samples_per_problem=samples_per_problem,
    )
    by_problem: dict[int, list[Mapping[str, Any]]] = {}
    for attempt in attempts:
        by_problem.setdefault(int(attempt["problem_index"]), []).append(attempt)

    any_segments = any(attempt.get("emitted_segments") for attempt in attempts)
    sections = []
    for problem_index in range(problem_count):
        problem_attempts = sorted(
            by_problem[problem_index], key=lambda item: int(item["sample_index"])
        )
        exemplar = problem_attempts[0]
        solved = sum(int(bool(item["correct"])) for item in problem_attempts)
        attempt_html = "\n".join(_render_attempt(item) for item in problem_attempts)
        sections.append(
            f"""
    <section class="problem">
      <div class="problem-heading">
        <p class="eyebrow">Dataset problem {problem_index + 1} · ID {html.escape(str(exemplar['dataset_index']))}</p>
        <div class="problem-title">
          <h2>Ground truth: <code>{html.escape(str(exemplar['ground_truth']))}</code></h2>
          <span class="badge solved {'correct' if solved else 'incorrect'}">{solved}/{len(problem_attempts)} correct</span>
        </div>
      </div>
      <details class="prompt">
        <summary>Prompt / question</summary>
        <pre>{html.escape(str(exemplar['prompt']))}</pre>
      </details>
      <div class="attempt-grid">{attempt_html}
      </div>
    </section>"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Latent benchmark answers · step {int(payload['step'])}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b0e14; --panel:#141a26; --panel-2:#0d121c; --line:#293246; --text:#e8edf7; --muted:#94a2b8; --good:#47d18c; --bad:#ff6b7a; --accent:#82aaff; --think:#c4a7ff; --answer:#6fe3ae; --eos:#ffab70; --think-bg:#c4a7ff1f; --answer-bg:#47d18c1f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at top,#161f33 0,var(--bg) 34rem); color:var(--text); font:15px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ width:min(1500px,calc(100% - 2rem)); margin:0 auto; padding:2.5rem 0 5rem; }}
    h1,h2,h3,p {{ margin-top:0; }}
    h1 {{ margin-bottom:.4rem; font-size:clamp(1.9rem,4.5vw,3.2rem); letter-spacing:-.04em; }}
    .subtitle,.eyebrow {{ color:var(--muted); }}
    .eyebrow {{ margin:0 0 .25rem; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:.4rem; margin:.8rem 0 0; }}
    .meta .badge {{ font-size:.74rem; }}
    .summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(13rem,1fr)); gap:.8rem; margin:1.8rem 0 1.4rem; }}
    .summary div,.problem,.attempt {{ border:1px solid var(--line); background:color-mix(in srgb,var(--panel) 94%,transparent); border-radius:14px; }}
    .summary div {{ padding:.9rem 1.1rem; }}
    .summary span {{ display:block; color:var(--muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; }}
    .summary strong {{ display:block; margin-top:.3rem; font-size:1.45rem; font-variant-numeric:tabular-nums; }}
    .legend {{ display:flex; flex-wrap:wrap; align-items:center; gap:1rem .9rem; margin:0 0 1rem; padding:.65rem .9rem; border:1px solid var(--line); border-radius:12px; background:var(--panel-2); color:var(--muted); font:12.5px/1.9 ui-monospace,SFMono-Regular,Consolas,monospace; }}
    .legend-item {{ white-space:nowrap; }}
    .problem {{ margin:1.4rem 0; padding:clamp(1rem,3vw,1.7rem); box-shadow:0 18px 45px #0005; }}
    .problem-title {{ display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:.6rem .9rem; margin-bottom:.8rem; }}
    .problem-title h2 {{ margin:0; font-size:1.25rem; }}
    code {{ color:#c3e88d; }}
    details {{ border-top:1px solid var(--line); padding-top:.8rem; }}
    summary {{ cursor:pointer; color:var(--accent); font-weight:650; }}
    pre {{ margin:.8rem 0 0; padding:1rem; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; border-radius:9px; background:var(--panel-2); color:#d7deec; font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace; }}
    .attempt-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; margin-top:1.2rem; }}
    .attempt {{ min-width:0; padding:1rem; border-left:4px solid var(--bad); display:flex; flex-direction:column; }}
    .attempt.correct {{ border-left-color:var(--good); }}
    .attempt header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:.8rem; margin-bottom:.75rem; }}
    .attempt h3 {{ margin:0; }}
    .badges {{ display:flex; flex-wrap:wrap; justify-content:flex-end; gap:.35rem; }}
    .badge {{ padding:.16rem .48rem; border:1px solid var(--line); border-radius:999px; color:var(--muted); font-size:.72rem; white-space:nowrap; }}
    .badge.correct,.badge.solved.correct {{ border-color:#47d18c88; color:var(--good); }}
    .badge.incorrect,.badge.solved.incorrect {{ border-color:#ff6b7a88; color:var(--bad); }}
    .stream {{ flex:1; margin:0; padding:.9rem 1rem; border-radius:9px; background:var(--panel-2); color:#d7deec; font:13px/1.9 ui-monospace,SFMono-Regular,Consolas,monospace; white-space:pre-wrap; overflow-wrap:anywhere; overflow-x:auto; }}
    .tok {{ display:inline-block; margin:0 .12rem; padding:0 .34rem; border-radius:5px; border:1px solid var(--line); font-size:.86em; line-height:1.55; vertical-align:baseline; color:var(--muted); background:#ffffff08; white-space:nowrap; }}
    .tok-think_open,.tok-think_close {{ color:var(--think); border-color:#c4a7ff66; background:var(--think-bg); }}
    .tok-answer_open,.tok-answer_close {{ color:var(--answer); border-color:#47d18c66; background:var(--answer-bg); }}
    .tok-eos,.tok-bos {{ color:var(--eos); border-color:#ffab7066; background:#ffab701a; }}
    .txt.in-think {{ background:var(--think-bg); border-radius:3px; }}
    .txt.in-answer {{ background:var(--answer-bg); border-radius:3px; }}
    .txt.prefix {{ color:var(--muted); font-style:italic; }}
    .tok.prefix {{ font-style:italic; opacity:.7; }}
    .nl {{ color:#5b688044; user-select:none; }}
    .stats {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.65rem; margin:.9rem 0 0; }}
    .stats div {{ min-width:0; }}
    dt {{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }}
    dd {{ margin:.1rem 0 0; overflow-wrap:anywhere; font-variant-numeric:tabular-nums; }}
    footer {{ margin-top:2rem; color:var(--muted); text-align:center; font-size:.85rem; }}
    @media (max-width:850px) {{ .attempt-grid {{ grid-template-columns:1fr; }} }}
    @media (max-width:500px) {{ main {{ width:min(100% - 1rem,1500px); padding-top:1.5rem; }} .stats {{ grid-template-columns:1fr; }} .attempt header {{ display:block; }} .badges {{ justify-content:flex-start; margin-top:.6rem; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <p class="eyebrow">Held-out automatic benchmark</p>
    <h1>What the model answered</h1>
    <p class="subtitle">Step {int(payload['step'])} · {problem_count} dataset problems · {samples_per_problem} rollout samples each</p>
    <div class="meta">
      <span class="badge">reward schema: {html.escape(str(payload.get('reward_schema', 'unspecified')))}</span>
      <span class="badge">sampling: {html.escape(str(metrics.get('sampling_schema', 'legacy')))}</span>
      <span class="badge">finished-row compaction: {html.escape(str(metrics.get('finished_compaction', 'legacy')))}</span>
    </div>
  </header>
  <section class="summary" aria-label="Benchmark summary">
    <div><span>All-rollout accuracy</span><strong>{_percent(metrics['accuracy'])}</strong></div>
    <div><span>Policy accuracy</span><strong>{_percent(metrics['policy_accuracy'])}</strong></div>
    <div><span>Prompts solved at least once</span><strong>{_percent(metrics.get('prompt_any_correct_fraction', 0.0))}</strong></div>
    <div><span>Mixed-reward prompt groups</span><strong>{_percent(metrics.get('prompt_mixed_reward_fraction', 0.0))}</strong></div>
    <div><span>Mean within-group reward std</span><strong>{float(metrics.get('within_group_reward_std', 0.0)):.4f}</strong></div>
    <div><span>Best constant-answer baseline</span><strong>{_percent(metrics.get('dataset_modal_answer_accuracy', 0.0))}</strong></div>
  </section>
  {_LEGEND if any_segments else ''}
  {''.join(sections)}
  <footer>Correctness requires explicit BOS/EOS termination · full-dataset modal answer: {html.escape(str(metrics.get('dataset_modal_answer', 'unavailable')))} · reward schema: {html.escape(str(payload.get('reward_schema', 'unspecified')))} · sampling: {html.escape(str(metrics.get('sampling_schema', 'legacy')))} · finished-row compaction: {html.escape(str(metrics.get('finished_compaction', 'legacy')))}</footer>
</main>
</body>
</html>
"""


def write_benchmark_report(
    output: str | Path,
    step: int,
    metrics: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    reward_schema: str = "unspecified",
    schema: str | None = None,
) -> dict[str, Path]:
    """Atomically persist JSON history/latest and the latest HTML report.

    ``schema`` defaults to the current answers schema; a rewrite of an
    older stored payload (the resume purge path) passes the source's own
    tag through so "answers/v4" stays a reliable has-segments signal on
    disk.
    """
    ordered = sorted(
        (dict(attempt) for attempt in attempts),
        key=lambda item: (int(item["problem_index"]), int(item["sample_index"])),
    )
    _validate_attempts(ordered)
    payload = {
        "schema": schema or BENCHMARK_ANSWER_SCHEMA,
        "step": int(step),
        "reward_schema": reward_schema,
        "problem_count": CAPTURE_PROBLEMS,
        "samples_per_problem": CAPTURE_SAMPLES_PER_PROBLEM,
        "metrics": dict(metrics),
        "attempts": ordered,
    }
    output = Path(output)
    history = output / "bench_answers" / f"step_{step:06d}.json"
    latest = output / "bench_answers" / "latest.json"
    report = output / "bench_answers.html"
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(history, serialized)
    _atomic_write_text(latest, serialized)
    _atomic_write_text(report, render_benchmark_report(payload))
    return {"history": history, "latest": latest, "report": report}


def sample_json_to_report_payload(
    sampled: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize ``sample_latent --math-rows`` output for this report.

    This powers cheap, one-off inspection panels without rerunning the full
    automatic benchmark. The caller controls row selection; each selected
    problem must contain the same positive number of samples.
    """
    records = sampled.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("sample JSON requires at least one problem record")
    first_samples = records[0].get("samples")
    if not isinstance(first_samples, list) or not first_samples:
        raise ValueError("each sampled problem requires at least one attempt")
    problem_count = len(records)
    declared_samples = sampled.get("samples_per_problem")
    if declared_samples is None:
        # Compatibility with older sample artifacts that predate the
        # self-describing count.
        samples_per_problem = len(first_samples)
    elif (
        isinstance(declared_samples, bool)
        or not isinstance(declared_samples, int)
        or declared_samples <= 0
    ):
        raise ValueError("samples_per_problem must be a positive integer")
    else:
        samples_per_problem = declared_samples
    sampled_metrics = sampled.get("metrics")
    pin_emit = bool(
        sampled_metrics.get("pin_emit", False)
        if isinstance(sampled_metrics, Mapping)
        else False
    )
    attempts: list[dict[str, Any]] = []
    for problem_index, record in enumerate(records):
        samples = record.get("samples")
        if (
            not isinstance(samples, list)
            or len(samples) != samples_per_problem
        ):
            raise ValueError(
                "each sampled problem requires exactly the same number of "
                f"attempts ({samples_per_problem})"
            )
        for sample_index, sample in enumerate(samples):
            attempt = {
                "problem_index": problem_index,
                "dataset_index": record["row_index"],
                "sample_index": sample_index,
                "prompt": record["problem"],
                "ground_truth": str(record["ground_truth"]),
                "answer_style": str(record.get("answer_style", "minerva")),
                "emitted_text": sample["text"],
                "parsed_answer": sample["prediction"],
                "correct": bool(sample["correct"]),
                "terminated": bool(sample["terminated"]),
                "termination_token_id": None,
                "emitted_token_count": int(sample["emits"]),
            }
            if sample.get("segments"):
                attempt["emitted_segments"] = sample["segments"]
            attempts.append(attempt)

    total = len(attempts)
    prompt_correct_counts = [0] * problem_count
    for attempt in attempts:
        prompt_correct_counts[attempt["problem_index"]] += int(
            attempt["correct"]
        )
    metrics = {
        "evaluation_metric_schema": (
            "deterministic_hidden_carry_token_actions/v4"
        ),
        "accuracy": sum(int(attempt["correct"]) for attempt in attempts) / total,
        "policy_accuracy": sum(
            int(attempt["correct"]) for attempt in attempts
        )
        / total,
        "policy_samples": total,
        "pin_emit": pin_emit,
        "prompt_correct_counts": prompt_correct_counts,
        "samples": total,
    }
    return {
        "schema": BENCHMARK_ANSWER_SCHEMA,
        "step": int(sampled.get("wrapper_step") or 0),
        "reward_schema": str(sampled.get("reward_schema", "unspecified")),
        "problem_count": problem_count,
        "samples_per_problem": samples_per_problem,
        "metrics": metrics,
        "attempts": attempts,
    }


def write_sample_json_report(source: str | Path, output: str | Path) -> Path:
    """Render a self-contained HTML panel from a sampling JSON artifact."""
    source = Path(source)
    sampled = json.loads(source.read_text(encoding="utf-8"))
    payload = sample_json_to_report_payload(sampled)
    output = Path(output)
    _atomic_write_text(output, render_benchmark_report(payload))
    return output


def bench_json_to_report_payload(
    payload: Mapping[str, Any], tokenizer
) -> dict[str, Any]:
    """Upgrade a stored bench-answers JSON for the segment-aware renderer.

    Histories written before answers/v4 stored raw ``emitted_token_ids``
    but no display segments; rebuilding them here gives finished runs the
    fence-visible report without re-running the benchmark. Attempts
    already carrying segments pass through untouched.

    The reconstruction is cross-checked against the stored ``emitted_text``
    (written by the run's OWN tokenizer): the rebuilt non-special text must
    be its suffix, and any leftover head is exactly the teacher-forced
    prefix whose ids were never persisted — it rejoins the stream as a
    ``kind="prefix"`` segment instead of silently vanishing. A mismatch
    means ``tokenizer`` carries the wrong special-token registration for
    this artifact and raises rather than fabricating fence chips. The
    check cannot catch a run that never registered fences yet sampled raw
    padded-vocab ids (both tokenizers strip them from text), so the fence
    flags remain the caller's assertion about the run.
    """
    from postraining.core import emitted_display_segments

    upgraded = dict(payload)
    attempts = []
    for attempt in payload["attempts"]:
        attempt = dict(attempt)
        if not attempt.get("emitted_segments") and attempt.get(
            "emitted_token_ids"
        ):
            segments = emitted_display_segments(
                attempt["emitted_token_ids"], tokenizer
            )
            joined = "".join(
                str(segment["text"])
                for segment in segments
                if segment.get("kind") != "special"
            )
            emitted_text = str(attempt.get("emitted_text", ""))
            if not emitted_text.endswith(joined):
                raise ValueError(
                    "segment reconstruction for problem "
                    f"{attempt.get('problem_index')} sample "
                    f"{attempt.get('sample_index')} does not reproduce the "
                    "stored emitted_text; the artifact's run used a "
                    "different special-token registration — rerun with "
                    "matching --think-tokens/--answer-tokens flags"
                )
            head = emitted_text[: len(emitted_text) - len(joined)]
            if head:
                segments.insert(0, {"kind": "prefix", "text": head})
            attempt["emitted_segments"] = segments
        attempts.append(attempt)
    upgraded["attempts"] = attempts
    return upgraded


def write_bench_json_report(
    source: str | Path,
    output: str | Path,
    think_tokens: bool = True,
    answer_tokens: bool = True,
) -> Path:
    """Re-render a stored bench-answers JSON (latest.json or step_*.json).

    Reconstructing segments needs the artifact's fence-token registration,
    which only the gpt2vocab family has; the flags default to the fully
    fenced setup every fenced run to date used, and MUST be turned off for
    artifacts from runs without --think-tokens/--answer-fence — a
    fence-less run can still sample raw padded-vocab ids that the fenced
    registration would mislabel as fence chips. The tokenizer import lives
    here so the module stays import-light for the trainer.
    """
    from postraining.core import GPT2BPETokenizer

    source = Path(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    tokenizer = GPT2BPETokenizer(
        think_tokens=think_tokens, answer_tokens=answer_tokens
    )
    upgraded = bench_json_to_report_payload(payload, tokenizer)
    output = Path(output)
    _atomic_write_text(output, render_benchmark_report(upgraded))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a benchmark-answer HTML report from a stored "
        "JSON artifact"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--sample-json", help="sample_latent --math-rows output"
    )
    source.add_argument(
        "--bench-json",
        help="bench_answers/step_*.json or latest.json from a training run",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--think-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--bench-json only: the artifact's run registered the "
        "<think></think> fence pair",
    )
    parser.add_argument(
        "--answer-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--bench-json only: the artifact's run registered the "
        "<answer></answer> fence pair",
    )
    args = parser.parse_args()
    if args.sample_json is not None:
        report = write_sample_json_report(args.sample_json, args.output)
    else:
        report = write_bench_json_report(
            args.bench_json,
            args.output,
            think_tokens=args.think_tokens,
            answer_tokens=args.answer_tokens,
        )
    print(report)


if __name__ == "__main__":
    main()

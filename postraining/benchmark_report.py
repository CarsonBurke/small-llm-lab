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
BENCHMARK_ANSWER_SCHEMA = "latent_benchmark_answers/v1"


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


def _render_attempt(attempt: Mapping[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    state = "correct" if attempt["correct"] else "incorrect"
    forced = "forced initial THINK" if attempt["forced_initial_think"] else "unforced"
    terminated = "terminated" if attempt["terminated"] else "unterminated"
    runs = attempt["think_run_lengths"] or []
    run_text = ", ".join(str(length) for length in runs) if runs else "none"
    return f"""
      <article class="attempt {state}">
        <header>
          <h3>Sample {int(attempt['sample_index']) + 1}</h3>
          <div class="badges">
            <span class="badge {state}">{state}</span>
            <span class="badge">{esc(forced)}</span>
            <span class="badge">{esc(terminated)}</span>
          </div>
        </header>
        <dl class="stats">
          <div><dt>Parsed answer</dt><dd>{esc(attempt['parsed_answer'])}</dd></div>
          <div><dt>Grader</dt><dd>{esc(attempt['answer_style'])}</dd></div>
          <div><dt>Thoughts</dt><dd>{int(attempt['optional_thought_count'])} optional / {int(attempt['total_thought_count'])} total</dd></div>
          <div><dt>Think runs</dt><dd>{esc(run_text)}</dd></div>
          <div><dt>Emitted tokens</dt><dd>{int(attempt['emitted_token_count'])}</dd></div>
        </dl>
        <details open>
          <summary>Full emitted text</summary>
          <pre>{esc(attempt['emitted_text'])}</pre>
        </details>
        <details>
          <summary>Action trace (T = THINK, E = EMIT)</summary>
          <pre class="trace">{esc(attempt['action_trace'])}</pre>
        </details>
      </article>"""


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

    sections = []
    for problem_index in range(problem_count):
        problem_attempts = sorted(
            by_problem[problem_index], key=lambda item: int(item["sample_index"])
        )
        exemplar = problem_attempts[0]
        attempt_html = "\n".join(_render_attempt(item) for item in problem_attempts)
        sections.append(
            f"""
    <section class="problem">
      <div class="problem-heading">
        <p class="eyebrow">Dataset problem {problem_index + 1} · ID {html.escape(str(exemplar['dataset_index']))}</p>
        <h2>Ground truth: <code>{html.escape(str(exemplar['ground_truth']))}</code></h2>
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
    :root {{ color-scheme: dark; --bg:#0b0e14; --panel:#131924; --line:#293246; --text:#e8edf7; --muted:#9aa8bd; --good:#47d18c; --bad:#ff6b7a; --accent:#82aaff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at top,#172035 0,var(--bg) 32rem); color:var(--text); font:15px/1.55 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ width:min(1500px,calc(100% - 2rem)); margin:0 auto; padding:3rem 0 5rem; }}
    h1,h2,h3,p {{ margin-top:0; }}
    h1 {{ margin-bottom:.4rem; font-size:clamp(2rem,5vw,3.7rem); letter-spacing:-.04em; }}
    .subtitle,.eyebrow {{ color:var(--muted); }}
    .eyebrow {{ margin:0 0 .25rem; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; }}
    .summary {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.8rem; margin:2rem 0 3rem; }}
    .summary div,.problem,.attempt {{ border:1px solid var(--line); background:color-mix(in srgb,var(--panel) 94%,transparent); border-radius:14px; }}
    .summary div {{ padding:1rem 1.1rem; }}
    .summary span {{ display:block; color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }}
    .summary strong {{ display:block; margin-top:.3rem; font-size:1.5rem; }}
    .problem {{ margin:1.5rem 0; padding:clamp(1rem,3vw,1.7rem); box-shadow:0 18px 45px #0005; }}
    .problem-heading h2 {{ margin-bottom:.8rem; }}
    code {{ color:#c3e88d; }}
    details {{ border-top:1px solid var(--line); padding-top:.8rem; }}
    summary {{ cursor:pointer; color:var(--accent); font-weight:650; }}
    pre {{ margin:.8rem 0 0; padding:1rem; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; border-radius:9px; background:#080b11; color:#d7deec; font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace; }}
    .attempt-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; margin-top:1.2rem; }}
    .attempt {{ min-width:0; padding:1rem; border-left:4px solid var(--bad); }}
    .attempt.correct {{ border-left-color:var(--good); }}
    .attempt header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:.8rem; }}
    .attempt h3 {{ margin:0; }}
    .badges {{ display:flex; flex-wrap:wrap; justify-content:flex-end; gap:.35rem; }}
    .badge {{ padding:.16rem .48rem; border:1px solid var(--line); border-radius:999px; color:var(--muted); font-size:.72rem; }}
    .badge.correct {{ border-color:#47d18c88; color:var(--good); }}
    .badge.incorrect {{ border-color:#ff6b7a88; color:var(--bad); }}
    .stats {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.65rem; margin:1rem 0; }}
    .stats div {{ min-width:0; }}
    dt {{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }}
    dd {{ margin:.1rem 0 0; overflow-wrap:anywhere; }}
    .attempt details + details {{ margin-top:.8rem; }}
    .trace {{ letter-spacing:.1em; color:#c792ea; }}
    footer {{ margin-top:2rem; color:var(--muted); text-align:center; }}
    @media (max-width:850px) {{ .summary,.attempt-grid {{ grid-template-columns:1fr; }} .summary {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width:500px) {{ main {{ width:min(100% - 1rem,1500px); padding-top:1.5rem; }} .summary,.stats {{ grid-template-columns:1fr; }} .attempt header {{ display:block; }} .badges {{ justify-content:flex-start; margin-top:.6rem; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <p class="eyebrow">Held-out automatic benchmark</p>
    <h1>What the model answered</h1>
    <p class="subtitle">Step {int(payload['step'])} · {problem_count} dataset problems · {samples_per_problem} policy samples each</p>
  </header>
  <section class="summary" aria-label="Benchmark summary">
    <div><span>Overall accuracy</span><strong>{_percent(metrics['accuracy'])}</strong></div>
    <div><span>Forced accuracy</span><strong>{_percent(metrics['forced_initial_accuracy'])}</strong></div>
    <div><span>Unforced accuracy</span><strong>{_percent(metrics['unforced_initial_accuracy'])}</strong></div>
    <div><span>Optional think fraction</span><strong>{_percent(metrics['think_fraction'])}</strong></div>
  </section>
  {''.join(sections)}
  <footer>Correctness requires explicit BOS/EOS termination · reward schema: {html.escape(str(payload.get('reward_schema', 'unspecified')))} · the action trace excludes prompt tokens.</footer>
</main>
</body>
</html>
"""


def write_benchmark_report(
    output: str | Path,
    step: int,
    metrics: Mapping[str, float | int | Mapping[str, float]],
    attempts: Sequence[Mapping[str, Any]],
    reward_schema: str = "unspecified",
) -> dict[str, Path]:
    """Atomically persist JSON history/latest and the latest HTML report."""
    ordered = sorted(
        (dict(attempt) for attempt in attempts),
        key=lambda item: (int(item["problem_index"]), int(item["sample_index"])),
    )
    _validate_attempts(ordered)
    payload = {
        "schema": BENCHMARK_ANSWER_SCHEMA,
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
            trace = str(sample["trace"]).upper()
            forced = bool(sample["forced_initial_think"])
            total_thoughts = int(sample["thinks"])
            attempts.append(
                {
                    "problem_index": problem_index,
                    "dataset_index": record["row_index"],
                    "sample_index": sample_index,
                    "prompt": record["problem"],
                    "ground_truth": str(record["ground_truth"]),
                    "answer_style": str(record.get("answer_style", "minerva")),
                    # The marked rendering exposes where latent slots occurred;
                    # raw emitted text remains in the source JSON.
                    "emitted_text": sample["text"],
                    "parsed_answer": sample["prediction"],
                    "correct": bool(sample["correct"]),
                    "terminated": bool(sample["terminated"]),
                    "termination_token_id": None,
                    "emitted_token_count": int(sample["emits"]),
                    "forced_initial_think": forced,
                    "forced_thought_count": int(forced),
                    "optional_thought_count": int(
                        sample.get(
                            "optional_thought_count",
                            total_thoughts - int(forced),
                        )
                    ),
                    "total_thought_count": total_thoughts,
                    "think_run_lengths": list(sample["think_run_lengths"]),
                    "action_trace": trace,
                }
            )

    total = len(attempts)
    forced_attempts = [
        attempt for attempt in attempts if attempt["forced_initial_think"]
    ]
    unforced_attempts = [
        attempt for attempt in attempts if not attempt["forced_initial_think"]
    ]
    optional_thoughts = sum(
        int(attempt["optional_thought_count"]) for attempt in attempts
    )
    optional_gate_actions = optional_thoughts + sum(
        int(attempt["emitted_token_count"]) for attempt in attempts
    )
    metrics = {
        "accuracy": sum(int(attempt["correct"]) for attempt in attempts) / total,
        "forced_initial_accuracy": sum(
            int(attempt["correct"]) for attempt in forced_attempts
        )
        / max(len(forced_attempts), 1),
        "unforced_initial_accuracy": sum(
            int(attempt["correct"]) for attempt in unforced_attempts
        )
        / max(len(unforced_attempts), 1),
        "think_fraction": optional_thoughts / max(optional_gate_actions, 1),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a benchmark-answer HTML report from sample_latent JSON"
    )
    parser.add_argument("--sample-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = write_sample_json_report(args.sample_json, args.output)
    print(report)


if __name__ == "__main__":
    main()

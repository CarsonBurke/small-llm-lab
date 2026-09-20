"""Table and pre-registered readings for the latent-feedback line (NOTES.md, 2026-09-17).

Reads ``ablation_results/<run>/metrics.jsonl`` (the harness's canonical stream)
and prints the final validation of every arm plus the differences the
pre-registration fixed: gating (glu vs add), gradient through the state (glu
vs glu_detach), window (feedback gain under the 65-token window vs at full
attention), memory (lam vs glu), and recurrence (sequential vs pass 8).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path("ablation_results")
COLUMNS = ("val_bpb", "val_bpb_p1", "val_bpb_p2", "val_bpb_p3", "val_bpb_p8", "val_bpb_seq", "val_bpb_p1_seq", "step_avg_ms")
ARMS = {
    "base": "nanomini_slot_full_2k",
    "base_w64": "nanomini_slot_fifo_2k",
    "glu": "nanomini_fb_glu_2k",
    "lam": "nanomini_fb_lam_2k",
    "glu_w64": "nanomini_fb_glu_w64_2k",
    "lam_w64": "nanomini_fb_lam_w64_2k",
    "add": "nanomini_fb_add_2k",
    "glu_detach": "nanomini_fb_glu_detach_2k",
}
# (label, minuend, subtrahend, metric): negative means the first arm is better.
READINGS = (
    ("feedback (glu - base), sequential", "glu", "base", "val_bpb_seq"),
    ("feedback (glu - base), pass 1 only", "glu", "base", "val_bpb_p1"),
    ("gating (glu - add), sequential", "glu", "add", "val_bpb_seq"),
    ("gradient through state (glu - glu_detach), sequential", "glu", "glu_detach", "val_bpb_seq"),
    ("memory (lam - glu), sequential", "lam", "glu", "val_bpb_seq"),
    ("windowed feedback (glu_w64 - base_w64), sequential", "glu_w64", "base_w64", "val_bpb_seq"),
    ("windowed memory (lam_w64 - base_w64), sequential", "lam_w64", "base_w64", "val_bpb_seq"),
    ("window cost (base_w64 - base)", "base_w64", "base", "val_bpb"),
)


def final_validation(run_dir: Path) -> dict | None:
    """The last validation row whose step is the run's final step, or None if the run has not finished."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    vals = [r for r in rows if r.get("type") == "val"]
    result = run_dir / "result.json"
    if not vals or not result.exists():
        return None
    steps = json.loads(result.read_text()).get("steps")
    last = vals[-1]
    return last if last.get("step") == steps else None


def load(results: Path = RESULTS) -> dict[str, dict]:
    return {arm: row for arm, name in ARMS.items() if (row := final_validation(results / name)) is not None}


def table(rows: dict[str, dict]) -> str:
    head = "arm".ljust(11) + "".join(c.replace("val_bpb", "bpb").rjust(11) for c in COLUMNS)
    lines = [head]
    for arm, row in rows.items():
        cells = "".join((f"{row[c]:.4f}" if isinstance(row.get(c), float) and c != "step_avg_ms" else
                         f"{row[c]:.0f}" if c in row else "-").rjust(11) for c in COLUMNS)
        lines.append(arm.ljust(11) + cells)
    return "\n".join(lines)


def readings(rows: dict[str, dict]) -> list[str]:
    out = []
    for label, a, b, metric in READINGS:
        if a in rows and b in rows and metric in rows[a] and metric in rows[b]:
            out.append(f"{label}: {rows[a][metric] - rows[b][metric]:+.4f}")
        else:
            out.append(f"{label}: pending")
    for arm, row in rows.items():
        if "val_bpb_seq" in row and "val_bpb_p8" in row:
            out.append(f"recurrence gap {arm} (seq - p8): {row['val_bpb_seq'] - row['val_bpb_p8']:+.4f}")
    if "lam" in rows:
        decays = [rows["lam"][k] for k in sorted(rows["lam"]) if k.startswith("mem_decay")]
        temps = [rows["lam"][k] for k in sorted(rows["lam"]) if k.startswith("mem_temp")]
        out.append(f"lam decays {decays} temperatures {temps}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    rows = load(args.results)
    print(table(rows))
    print()
    print("\n".join(readings(rows)))


if __name__ == "__main__":
    main()

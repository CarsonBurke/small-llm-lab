import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import compare_feedback_arms as cfa  # noqa: E402


def _run(root: Path, name: str, steps: int, rows: list[dict], finished: bool = True) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if finished:
        (d / "result.json").write_text(json.dumps({"steps": steps}))


def test_final_validation_requires_the_last_step_and_a_result(tmp_path):
    _run(tmp_path, cfa.ARMS["base"], 2000, [{"type": "val", "step": 1980, "val_bpb": 1.3}, {"type": "val", "step": 2000, "val_bpb": 1.2, "val_bpb_seq": 1.2}])
    _run(tmp_path, cfa.ARMS["glu"], 2000, [{"type": "val", "step": 1980, "val_bpb": 1.4}])
    _run(tmp_path, cfa.ARMS["add"], 2000, [{"type": "val", "step": 2000, "val_bpb": 1.4}], finished=False)
    rows = cfa.load(tmp_path)
    assert set(rows) == {"base"} and rows["base"]["val_bpb"] == 1.2


def test_readings_are_signed_differences_and_pending_when_missing(tmp_path):
    _run(tmp_path, cfa.ARMS["base"], 2000, [{"type": "val", "step": 2000, "val_bpb": 1.30, "val_bpb_p1": 1.30, "val_bpb_seq": 1.30, "step_avg_ms": 580.0}])
    _run(tmp_path, cfa.ARMS["glu"], 2000, [{"type": "val", "step": 2000, "val_bpb": 1.28, "val_bpb_p1": 1.29, "val_bpb_p8": 1.27,
                                            "val_bpb_seq": 1.275, "step_avg_ms": 1200.0}])
    _run(tmp_path, cfa.ARMS["lam"], 2000, [{"type": "val", "step": 2000, "val_bpb": 1.25, "val_bpb_seq": 1.25, "val_bpb_p8": 1.25,
                                            "mem_decay0": 0.5, "mem_decay1": 0.9, "mem_temp0": 1.0, "mem_temp1": 1.1}])
    rows = cfa.load(tmp_path)
    lines = cfa.readings(rows)
    assert "feedback (glu - base), sequential: -0.0250" in lines
    assert "feedback (glu - base), pass 1 only: -0.0100" in lines
    assert "memory (lam - glu), sequential: -0.0250" in lines
    assert "gating (glu - add), sequential: pending" in lines
    assert "recurrence gap glu (seq - p8): +0.0050" in lines
    assert "lam decays [0.5, 0.9] temperatures [1.0, 1.1]" in lines
    text = cfa.table(rows)
    assert "glu" in text and "1.2750" in text and "1200" in text

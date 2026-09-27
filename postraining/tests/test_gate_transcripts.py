import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch
import pytest

from postraining import gate_transcripts


def test_capture_keeps_every_verdict_and_serializes_parallel_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(gate_transcripts, "emitted_token_rows", lambda batch: [[7, 9], [8]])
    tokenizer = SimpleNamespace(decode=lambda tokens: str(tokens))
    path = tmp_path / "gate.jsonl"
    writer = gate_transcripts.GateTranscriptWriter(path, tokenizer, [9])
    batch = SimpleNamespace(reward_scalar=torch.tensor([1., 0.]))
    verdicts = [SimpleNamespace(format_ok=True, parsed_answer="A"),
                SimpleNamespace(format_ok=False, parsed_answer=None)]
    row = {"prompt": [{"role": "user", "content": "question"}],
           "reward_model": {"ground_truth": "A"},
           "extra_info": {"module": "sciq_choice_4"}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: writer.write(batch, verdicts, row), range(8)))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 16
    assert sum(record["terminated"] for record in records) == 8
    assert sum(record["reward"] for record in records) == 8
    assert all(record["extra_info"] == row["extra_info"] for record in records)
    assert [record["token_ids"] for record in records[:2]] == [[7, 9], [8]]
    with pytest.raises(FileExistsError):
        gate_transcripts.GateTranscriptWriter(path, tokenizer, [9])


def test_summary_rejects_partial_or_duplicate_samples(tmp_path):
    from scripts.summarize_posttraining_gates import summarize

    path = tmp_path / "gate.jsonl"
    record = {"schema": "frozen_policy_gate_transcript/v1", "prompt": ["Q"],
              "sample": 0, "extra_info": {"module": "choice_4", "chance": .25},
              "reward": 1., "structural_format_ok": True, "terminated": True,
              "parsed_answer": "A", "reward_model": {"ground_truth": "A"}}
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        summarize(path, 1, 2)
    path.write_text((json.dumps(record) + "\n") * 2)
    with pytest.raises(ValueError):
        summarize(path, 1, 2)
    second = {**record, "sample": 1, "reward": 0., "parsed_answer": "B"}
    path.write_text(json.dumps(record) + "\n" + json.dumps(second) + "\n")
    result = summarize(path, 1, 2)["modules"]["choice_4"]
    assert result["contract_accuracy"] == .5
    assert result["terminated_parsed_letter_accuracy"] == .5
    assert result["mixed_prompt_fraction"] == 1.
    # DAPO/UltraData Math have source identities but no topic/module label.
    record = {**record, "extra_info": {"index": "q1"}, "source": "dapo"}
    path.write_text(json.dumps(record) + "\n")
    assert summarize(path, 1, 1)["modules"]["dapo"]["contract_accuracy"] == 1.

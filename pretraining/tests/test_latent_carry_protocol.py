"""Pure scheduling/evidence contracts for causal diagnostics; no GPU execution."""
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import train_latent_carry as trainer


def parse(architecture="carry", *extra):
    return trainer.parse_args(["--name", "contract", "--architecture", architecture,
                               "--steps", "1000", "--throughput-report", "benchmark.json", *extra])


def test_diagnostic_defaults_and_explicit_budget():
    assert parse().microbatch == 512
    assert parse("narrow").microbatch == 64
    assert parse().val_every == 20
    with pytest.raises(SystemExit):
        trainer.parse_args(["--name", "contract", "--architecture", "carry",
                            "--throughput-report", "benchmark.json"])
    with pytest.raises(SystemExit):
        trainer.parse_args(["--name", "contract", "--architecture", "carry", "--steps", "1000"])


@pytest.mark.parametrize("extra", [["--steps", "999"], ["--steps", "1001"], ["--val-every", "40"],
                                  ["--name", "../escape"], ["--microbatch", "128"]])
def test_invalid_protocol_rejected(extra):
    with pytest.raises(SystemExit):
        parse("carry", *extra)


def test_narrow_control_cannot_change_execution_batch():
    with pytest.raises(SystemExit):
        parse("narrow", "--microbatch", "512")


@pytest.mark.parametrize("source", ["current", "encoder", "refined"])
def test_refiner_source_contract_and_batch_choices(source):
    architecture = f"refiner_{source}"
    assert parse(architecture).microbatch == 128
    for batch in (64, 128, 256):
        assert parse(architecture, "--microbatch", str(batch)).microbatch == batch
    with pytest.raises(SystemExit):
        parse(architecture, "--microbatch", "512")


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "source_hashes", lambda: {"fixture.py": "digest"})
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "contract-gpu")
    config = {"heads": 4, "key_dim": 32, "value_dim": 128}
    candidate = dict(model="carry", model_config=config, microbatch=512,
                     tokens_per_second=524288 / 2, update_seconds=[2.0] * 5,
                     warmup_optimizer_updates=5, measured_optimizer_updates=5)
    baseline = dict(model="nanogpt_mini", microbatch=64, tokens_per_second=524288,
                    update_seconds=[1.0] * 5, warmup_optimizer_updates=5, measured_optimizer_updates=5)
    report = dict(status="completed", architecture="carry", batch_tokens=524288,
                  seq_len=1024, optimizer_included=True, compiled=True, cuda_graph=True,
                  source_sha256={"fixture.py": "digest"}, gpu="contract-gpu", torch=str(torch.__version__),
                  candidate=candidate, baseline=baseline, repeats=5, gate_passed=False)
    args = SimpleNamespace(throughput_report=tmp_path / "benchmark.json", architecture="carry", microbatch=512)

    def check():
        args.throughput_report.write_text(json.dumps(report))
        return trainer.throughput_evidence(args, config)
    return args, report, check


def test_completed_slower_benchmark_allows_diagnosis_but_not_promotion(evidence):
    _, _, check = evidence
    speed = check()
    assert speed["available"] and not speed["speed_passed"]
    result = trainer.promotion_status("completed", 1000, 1.30, 1.3265, True, speed)
    assert result["quality_passed"] and not result["retained"]


def test_speed_and_quality_needed_for_promotion(evidence):
    _, report, check = evidence
    report["candidate"].update(tokens_per_second=524288 / 0.8, update_seconds=[0.8] * 5)
    speed = check()
    assert speed["speed_passed"]
    assert trainer.promotion_status("completed", 1000, 1.30, 1.3265, True, speed)["retained"]
    assert not trainer.promotion_status("completed", 1000, 1.325, 1.3265, True, speed)["retained"]
    assert not trainer.promotion_status("running", 80, 1.0, 1.3265, True, speed)["retained"]


@pytest.mark.parametrize("mutation", ["unfinished", "uncompiled", "stale", "config", "microbatch", "samples", "warmups", "rate"])
def test_invalid_benchmark_evidence_rejected(evidence, mutation):
    _, report, check = evidence
    if mutation == "unfinished":
        report["status"] = "running"
    elif mutation == "uncompiled":
        report["compiled"] = False
    elif mutation == "stale":
        report["source_sha256"] = {}
    elif mutation == "config":
        report["candidate"]["model_config"] = {"heads": 1}
    elif mutation == "microbatch":
        report["candidate"]["microbatch"] = 64
    elif mutation == "samples":
        report["candidate"]["update_seconds"].pop()
    elif mutation == "warmups":
        report["candidate"]["warmup_optimizer_updates"] = 2
    else:
        report["candidate"]["tokens_per_second"] *= 2
    with pytest.raises(ValueError):
        check()

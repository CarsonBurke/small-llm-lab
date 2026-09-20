"""Pure report validation; no model construction or CUDA execution."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pretraining.nanogpt_mini.gated_delta_runtime import (
    runtime_dependency_versions, verify_throughput_report,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def gate(tmp_path, monkeypatch):
    from pretraining.nanogpt_mini import gated_delta_runtime as runtime
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "contract-gpu")
    provenance = {"source_sha256": {"fla-core:fla/ops/gdn2/chunk.py": "fixture-digest"}}
    monkeypatch.setattr(runtime, "gdn2_dependency_provenance", lambda: provenance)
    config = dict(vocab_size=1024, num_layers=6, model_dim=512, head_dim=128,
                  mixer_dim=512, expand_v=1.0, use_short_conv=True, conv_size=4,
                  allow_neg_eigval=False, mixer_norm_eps=1e-5, kernel_chunk_size=64, fused_projections=False,
                  gdn_backend="vendor", state_v_first=False, disable_recompute=False)
    baseline = dict(model="nanogpt_mini", microbatch=64, tokens_per_second=524288,
                    measured_optimizer_updates=5, warmup_optimizer_updates=5, update_seconds=[1.0] * 5)
    candidate = dict(model="gated_delta", model_config=config, microbatch=64, chunk_size=64,
                     tokens_per_second=655360, measured_optimizer_updates=5, warmup_optimizer_updates=5,
                     validation_microbatch=64, validation_residency_passed=True, update_seconds=[0.8] * 5)
    sources = [ROOT / p for p in (
        "pretraining/nanogpt_mini/gated_delta_model.py",
        "pretraining/nanogpt_mini/gated_delta_runtime.py",
        "pretraining/nanogpt_mini/chunk_memory_runtime.py",
        "pretraining/nanogpt_mini/nanogpt_mini_model.py",
        "pretraining/nanogpt_mini/recurrent_slots_runtime.py",
        "scripts/train_recurrent_slots.py", "scripts/benchmark_gated_delta.py",
        "scripts/benchmark_chunk_memory.py")]
    sources.extend(p for p in (ROOT / "pretraining/gated_delta/vendor").rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    report = dict(status="completed", gate_passed=True, batch_tokens=524288, seq_len=1024,
                  optimizer_included=True, compiled=True, cuda_graph=True, checkpointing=False,
                  repeats=5, baseline=baseline, candidate=candidate, gpu="contract-gpu",
                  torch=str(torch.__version__), dependency_versions=runtime_dependency_versions(),
                  gdn2_dependency_provenance=json.loads(json.dumps(provenance)),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources})
    path = tmp_path / "benchmark.json"
    args = SimpleNamespace(throughput_report=path, segment_size=64, microbatch=64, memory_head_dim=128,
                           memory_width=512, fused_projections=False, architecture="gdn2", value_expansion=1.0,
                           gdn_backend="vendor", state_v_first=False, disable_recompute=False,
                           allow_slow_diagnostic=False)

    def check():
        path.write_text(json.dumps(report))
        return verify_throughput_report(args)
    return args, report, check


def test_valid_gate_records_evidence(gate):
    _, _, check = gate
    result = check()
    assert result["candidate_tokens_per_second"] == 655360
    assert result["baseline_tokens_per_second"] == 524288
    assert len(result["sha256"]) == 64


@pytest.mark.parametrize("key,value", [
    ("status", "failed"), ("gate_passed", False), ("batch_tokens", 65536),
    ("seq_len", 256), ("optimizer_included", False), ("compiled", False),
    ("cuda_graph", False), ("checkpointing", True), ("repeats", 4),
    ("gpu", "different"), ("torch", "different"),
])
def test_invalid_benchmark_contract_rejected(gate, key, value):
    _, report, check = gate
    report[key] = value
    with pytest.raises(ValueError):
        check()


@pytest.mark.parametrize("mutation", ["source", "dependency", "config", "microbatch", "short_samples", "rate", "overlap"])
def test_stale_or_unmatched_evidence_rejected(gate, mutation):
    _, report, check = gate
    if mutation == "source":
        report["source_sha256"]["pretraining/nanogpt_mini/gated_delta_runtime.py"] = "stale"
    elif mutation == "dependency":
        report["dependency_versions"]["triton"] = "different"
    elif mutation == "config":
        report["candidate"]["model_config"]["head_dim"] = 64
    elif mutation == "microbatch":
        report["candidate"]["microbatch"] = 32
    elif mutation == "short_samples":
        report["candidate"]["update_seconds"].pop()
    elif mutation == "rate":
        report["candidate"]["tokens_per_second"] *= 2
    else:
        report["candidate"]["update_seconds"][0] = 1.01
    with pytest.raises(ValueError):
        check()


@pytest.mark.parametrize('name', ['chunk_gdn2', 'causal_conv1d_fwd', 'layer_norm_gated_fwd'])
def test_audit_accepts_verified_opaque_kernels(monkeypatch, name):
    from torch._dynamo.utils import counters
    from pretraining.nanogpt_mini.gated_delta_runtime import CompiledGatedDeltaLoss
    reason = f'Skip calling torch.compiler.disable function <function {name} at 0x1234>'
    monkeypatch.setitem(counters, 'graph_break', {reason: 1})
    audit = object.__new__(CompiledGatedDeltaLoss)
    audit._breaks_before = {}
    audit.model = SimpleNamespace(config={})
    assert audit.audit_graph_breaks() == {reason: 1}


@pytest.mark.parametrize('reason', [
    'torch.compiler.disable <function causal_conv1d_fwd_other at 0x1234>',
    'unsupported operation in chunk_gdn2',
    'torch.compiler.disable without an identifiable kernel',
    'torch.compiler.disable <function eager_fallback at 0x1234> chunk_gdn2',
])
def test_audit_rejects_unrelated_graph_breaks(monkeypatch, reason):
    from torch._dynamo.utils import counters
    from pretraining.nanogpt_mini.gated_delta_runtime import CompiledGatedDeltaLoss
    monkeypatch.setitem(counters, 'graph_break', {reason: 1})
    audit = object.__new__(CompiledGatedDeltaLoss)
    audit._breaks_before = {}
    audit.model = SimpleNamespace(config={})
    with pytest.raises(RuntimeError):
        audit.audit_graph_breaks()


@pytest.mark.parametrize('width', [256, 512])
def test_narrowed_memory_gate_matches_training_configuration(gate, width):
    args, report, check = gate
    args.memory_head_dim = 64
    args.memory_width = width
    report['candidate']['model_config'].update(head_dim=64, mixer_dim=width)
    assert check()['candidate_tokens_per_second'] == 655360
    args.memory_width = 128
    with pytest.raises(ValueError):
        check()


def test_fused_projection_gate_matches_training_configuration(gate):
    args, report, check = gate
    args.fused_projections = True
    report["candidate"]["model_config"]["fused_projections"] = True
    check()
    args.fused_projections = False
    with pytest.raises(ValueError):
        check()


@pytest.fixture
def scalar_gate(gate, monkeypatch):
    from pretraining.nanogpt_mini import gated_delta_runtime as runtime
    args, report, check = gate
    args.architecture = "gdn1"
    report["candidate"]["model"] = "scalar_delta"
    config = report["candidate"]["model_config"]
    for key in ("gdn_backend", "state_v_first", "disable_recompute"):
        del config[key]
    config.update(memory_rule="scalar_delta", initialization="gdn2_matched_distributions")
    source = ROOT / "pretraining/nanogpt_mini/scalar_delta_model.py"
    report["source_sha256"][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    provenance = {"distribution_versions": {"fla-core": "0.5.2", "flash-linear-attention": "0.5.2"},
                  "source_sha256": {"fla-core:fla/ops/gated_delta_rule/chunk.py": "fixture-digest"},
                  "wheel_record_mismatches": []}
    monkeypatch.setattr(runtime, "scalar_dependency_provenance", lambda: provenance)
    report["scalar_dependency_provenance"] = json.loads(json.dumps(provenance))
    return args, report, check


def test_scalar_gate_accepts_verified_model_and_dependency_sources(scalar_gate):
    _, _, check = scalar_gate
    assert check()["candidate_tokens_per_second"] == 655360


@pytest.mark.parametrize("mutation", ["wrong_rule", "wrong_init", "fused_key", "source", "dependency_source", "identity"])
def test_scalar_gate_rejects_model_or_dependency_drift(scalar_gate, mutation):
    _, report, check = scalar_gate
    if mutation == "wrong_rule":
        report["candidate"]["model_config"]["memory_rule"] = "channel_delta"
    elif mutation == "wrong_init":
        report["candidate"]["model_config"]["initialization"] = "default"
    elif mutation == "fused_key":
        del report["candidate"]["model_config"]["fused_projections"]
    elif mutation == "source":
        del report["source_sha256"]["pretraining/nanogpt_mini/scalar_delta_model.py"]
    elif mutation == "identity":
        report["candidate"]["model"] = "gated_delta"
    else:
        report["scalar_dependency_provenance"]["source_sha256"]["fla-core:fla/ops/gated_delta_rule/chunk.py"] = "changed"
    with pytest.raises(ValueError):
        check()


@pytest.mark.parametrize("rule,operator,accepted", [
    ("scalar_delta", "chunk_gated_delta_rule", True),
    ("scalar_delta", "causal_conv1d_fwd", True),
    ("scalar_delta", "layer_norm_gated_fwd", True),
    ("scalar_delta", "chunk_gdn2", False),
    ("gdn2", "chunk_gated_delta_rule", False),
    ("unknown", "chunk_gdn2", False),
])
def test_model_specific_kernel_boundary_audit(monkeypatch, rule, operator, accepted):
    from torch._dynamo.utils import counters
    from pretraining.nanogpt_mini.gated_delta_runtime import CompiledGatedDeltaLoss
    reason = f"torch.compiler.disable <function {operator} at 0x1234>"
    monkeypatch.setitem(counters, "graph_break", {reason: 1})
    audit = object.__new__(CompiledGatedDeltaLoss)
    audit.model = SimpleNamespace(config={"memory_rule": rule})
    audit._breaks_before = {}
    if accepted:
        assert audit.audit_graph_breaks() == {reason: 1}
    else:
        with pytest.raises(RuntimeError):
            audit.audit_graph_breaks()


@pytest.mark.parametrize("fused", [False, True])
def test_scalar_projection_packing_requires_matching_configuration(scalar_gate, fused):
    args, report, check = scalar_gate
    args.fused_projections = fused
    report["candidate"]["model_config"]["fused_projections"] = fused
    assert check()["candidate_tokens_per_second"] == 655360
    args.fused_projections = not fused
    with pytest.raises(ValueError, match="configuration"):
        check()


def test_scalar_near_miss_still_fails_five_percent_gate(scalar_gate):
    _, report, check = scalar_gate
    speedup = 1.048615
    report["candidate"]["tokens_per_second"] = 524288 * speedup
    report["candidate"]["update_seconds"] = [1 / speedup] * 5
    with pytest.raises(ValueError, match="five percent"):
        check()


@pytest.mark.parametrize("expansion", [1.0, 2.0])
def test_scalar_value_expansion_requires_matching_configuration(scalar_gate, expansion):
    args, report, check = scalar_gate
    args.memory_width = 128
    args.memory_head_dim = 64
    args.value_expansion = expansion
    report["candidate"]["model_config"].update(mixer_dim=128, head_dim=64, expand_v=expansion)
    assert check()["candidate_tokens_per_second"] == 655360
    args.value_expansion = 1.0 if expansion == 2.0 else 2.0
    with pytest.raises(ValueError, match="configuration"):
        check()


@pytest.mark.parametrize("key,value", [("gdn_backend", "fla"), ("state_v_first", True), ("disable_recompute", True)])
def test_gdn2_execution_requires_matching_configuration(gate, key, value):
    args, report, check = gate
    setattr(args, key, value)
    with pytest.raises(ValueError, match="configuration"):
        check()
    report["candidate"]["model_config"][key] = value
    check()


@pytest.mark.parametrize("mutation", ["dependency", "warmup", "validation", "count"])
def test_gdn2_rejects_incomplete_execution_evidence(gate, mutation):
    _, report, check = gate
    if mutation == "dependency":
        report["gdn2_dependency_provenance"]["source_sha256"].clear()
    elif mutation == "warmup":
        report["baseline"]["warmup_optimizer_updates"] = 2
    elif mutation == "validation":
        report["candidate"]["validation_residency_passed"] = False
    else:
        report["candidate"]["measured_optimizer_updates"] = 6
    with pytest.raises(ValueError):
        check()


def test_candidate_batch_can_differ_from_fixed_mini_reference(gate):
    args, report, check = gate
    args.microbatch = 32
    report["candidate"].update(microbatch=32, validation_microbatch=32)
    check()
    report["baseline"]["microbatch"] = 32
    with pytest.raises(ValueError, match="configuration"):
        check()


def test_explicit_slow_diagnostic_keeps_speed_failure(gate):
    args, report, check = gate
    report["candidate"].update(tokens_per_second=524288 / 1.2, update_seconds=[1.2] * 5)
    report["gate_passed"] = False
    with pytest.raises(ValueError, match="five percent"):
        check()
    args.allow_slow_diagnostic = True
    evidence = check()
    assert evidence["speed_passed"] is False
    assert evidence["diagnostic_only"] is True
    report["status"] = "failed"
    with pytest.raises(ValueError, match="status"):
        check()


def test_diagnostic_still_rejects_invented_speed_pass(gate):
    args, report, check = gate
    args.allow_slow_diagnostic = True
    report["candidate"].update(tokens_per_second=524288 / 1.2, update_seconds=[1.2] * 5)
    with pytest.raises(ValueError, match="gate_passed"):
        check()

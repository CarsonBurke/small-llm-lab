"""Pure report validation; no model construction or CUDA execution."""
import hashlib
import json
import os
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
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args, **kwargs: "contract-gpu")
    provenance = {"source_sha256": {"fla-core:fla/ops/gdn2/chunk.py": "fixture-digest"}}
    monkeypatch.setattr(runtime, "gdn2_dependency_provenance", lambda: provenance)
    policy = dict(fla_cache_results=True, fla_cache_mode="disabled", fla_config_dir=None, fla_config_dir_sha256=None,
                  triton_cache_autotuning=False, autotune_cache_kwargs={"cache_results": True},
                  effective_triton_persistent_results=True)
    monkeypatch.setattr(runtime, "runtime_autotuning_policy", lambda: policy)
    config = dict(vocab_size=1024, num_layers=6, model_dim=512, head_dim=128,
                  mixer_dim=512, expand_v=1.0, use_short_conv=True, conv_size=4,
                  allow_neg_eigval=False, mixer_norm_eps=1e-5, kernel_chunk_size=64, fused_projections=False,
                  gdn_backend="vendor", state_v_first=False, disable_recompute=False, custom_ops=False,
                  gate_in_kernel=False)
    baseline = dict(model="nanogpt_mini", microbatch=64, tokens_per_second=524288,
                    measured_optimizer_updates=5, warmup_optimizer_updates=5, update_seconds=[1.0] * 5)
    candidate = dict(model="gated_delta", model_config=config, microbatch=64, chunk_size=64,
                     tokens_per_second=655360, measured_optimizer_updates=5, warmup_optimizer_updates=5,
                     validation_microbatch=64, validation_residency_passed=True, update_seconds=[0.8] * 5)
    sources = [ROOT / p for p in (
        "pretraining/nanogpt_mini/gated_delta_model.py",
        "pretraining/nanogpt_mini/gated_delta_ops.py",
        "pretraining/nanogpt_mini/gated_delta_pool.py",
        "pretraining/nanogpt_mini/gated_delta_bank_linear.py",
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
                  autotuning_policy=json.loads(json.dumps(policy)),
                  torch=str(torch.__version__), dependency_versions=runtime_dependency_versions(),
                  gdn2_dependency_provenance=json.loads(json.dumps(provenance)),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources})
    path = tmp_path / "benchmark.json"
    args = SimpleNamespace(throughput_report=path, segment_size=64, microbatch=64, memory_head_dim=128,
                           memory_width=512, fused_projections=False, architecture="gdn2", value_expansion=1.0,
                           gdn_backend="vendor", state_v_first=False, disable_recompute=False, custom_ops=False,
                           gate_in_kernel=False, shared_pool=False, pool_banks=None, pool_writer="routed",
                           pool_heads=2,
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
    for key in ("gdn_backend", "state_v_first", "disable_recompute", "custom_ops", "gate_in_kernel"):
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


@pytest.mark.parametrize("key,value", [("gdn_backend", "fla"), ("state_v_first", True),
                                       ("disable_recompute", True), ("custom_ops", True),
                                       ("gate_in_kernel", True)])
def test_gdn2_execution_requires_matching_configuration(gate, key, value):
    args, report, check = gate
    setattr(args, key, value)
    with pytest.raises(ValueError, match="configuration"):
        check()
    report["candidate"]["model_config"][key] = value
    check()


def test_shared_pool_gate_binds_pool_geometry(gate):
    args, report, check = gate
    pool = dict(gdn_backend="fla", disable_recompute=True, custom_ops=True, fused_projections=True)
    for key, value in pool.items():
        setattr(args, key, value)
    report["candidate"]["model_config"].update(pool)
    check()
    args.shared_pool = True
    with pytest.raises(ValueError, match="configuration"):
        check()
    policy = dict(shared_pool=True, pool_banks=12, pool_writer="routed", pool_heads=2, pool_reads=2, pool_passes=2,
                  pool_balance_coefficient=0.01, pool_z_coefficient=0.001)
    report["candidate"]["model_config"].update(policy)
    assert check()["candidate_tokens_per_second"] == 655360
    args.pool_banks = 8
    with pytest.raises(ValueError, match="configuration"):
        check()
    report["candidate"]["model_config"]["pool_banks"] = 8
    check()
    args.pool_writer, args.pool_banks = "layer", None
    with pytest.raises(ValueError, match="configuration"):
        check()
    report["candidate"]["model_config"].update(pool_writer="layer", pool_banks=6)
    check()
    args.pool_heads = 4
    with pytest.raises(ValueError, match="configuration"):
        check()
    report["candidate"]["model_config"]["pool_heads"] = 4
    check()
    report["candidate"]["model_config"]["pool_passes"] = 3
    with pytest.raises(ValueError, match="configuration"):
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


def four_k_report(gate):
    args, report, check = gate
    args.seq_len = 4096
    args.microbatch = 16
    report.update(seq_len=4096, train_seq_len=4096, val_seq_len=4096)
    for arm in (report['candidate'], report['baseline']):
        arm.update(microbatch=16, seq_len=4096, validation_seq_len=4096,
                   validation_microbatch=16, validation_residency_passed=True,
                   microsteps_per_update=8)
    return args, report, check


def test_four_k_report_matches_full_context_and_token_budget(gate):
    _, _, check = four_k_report(gate)
    assert check()['candidate_tokens_per_second'] == 655360


@pytest.mark.parametrize('arm,key,value', [
    ('candidate', 'seq_len', 1024), ('baseline', 'seq_len', 1024),
    ('candidate', 'validation_seq_len', 1024), ('baseline', 'validation_seq_len', 1024),
    ('candidate', 'microsteps_per_update', 32), ('baseline', 'microbatch', 64),
    ('candidate', 'validation_microbatch', 64), ('baseline', 'validation_residency_passed', False),
])
def test_four_k_rejects_mismatched_graph_shapes(gate, arm, key, value):
    _, report, check = four_k_report(gate)
    report[arm][key] = value
    with pytest.raises(ValueError):
        check()


def test_four_k_rejects_one_k_report(gate):
    args, _, check = gate
    args.seq_len = 4096
    args.microbatch = 16
    with pytest.raises(ValueError, match='seq_len'):
        check()


def test_four_k_training_cli_preserves_fixed_update_protocol():
    from scripts.train_recurrent_slots import parse_args
    common = ['--name', 'four_k', '--steps', '1000', '--architecture', 'gdn2',
              '--segment-size', '64', '--seq-len', '4096', '--microbatch', '16',
              '--throughput-report', 'unused.json']
    args = parse_args(common)
    assert args.seq_len == 4096 and args.microbatch == 16 and args.val_every == 20
    assert 524288 // (args.seq_len * args.microbatch) == 8
    for extra in (['--architecture', 'gdn1'], ['--microbatch', '256'], ['--steps', '2000']):
        with pytest.raises(SystemExit):
            parse_args(common + extra)


@pytest.mark.parametrize("key,value", [
    ("fla_cache_results", False), ("fla_cache_mode", "full"),
    ("fla_config_dir", "pretraining/gated_delta/autotune/other"), ("fla_config_dir_sha256", "0" * 64),
    ("triton_cache_autotuning", True), ("autotune_cache_kwargs", {"cache_results": False}),
    ("effective_triton_persistent_results", False),
])
def test_autotuning_policy_mismatch_rejects_timing(gate, key, value):
    _, report, check = gate
    report["autotuning_policy"][key] = value
    with pytest.raises(ValueError, match="autotuning policy"):
        check()


def test_missing_autotuning_policy_fails_closed(gate):
    _, report, check = gate
    del report["autotuning_policy"]
    with pytest.raises(ValueError, match="autotuning policy"):
        check()


def test_whole_graph_models_reject_every_kernel_boundary(monkeypatch):
    from torch._dynamo.utils import counters
    from pretraining.nanogpt_mini.gated_delta_runtime import CompiledGatedDeltaLoss
    reason = 'Skip calling torch.compiler.disable function <function chunk_gdn2 at 0x1234>'
    monkeypatch.setitem(counters, 'graph_break', {reason: 1})
    audit = object.__new__(CompiledGatedDeltaLoss)
    audit._breaks_before = {}
    audit.model = SimpleNamespace(config={"custom_ops": True})
    with pytest.raises(RuntimeError, match="graph breaks"):
        audit.audit_graph_breaks()
    audit._breaks_before = {reason: 1}
    assert audit.audit_graph_breaks() == {}


def test_custom_ops_cli_is_specific_to_gdn2():
    from scripts.train_recurrent_slots import parse_args
    common = ['--name', 'ops', '--steps', '1000', '--segment-size', '64', '--microbatch', '64',
              '--throughput-report', 'unused.json', '--custom-ops']
    args = parse_args(common + ['--architecture', 'gdn2'])
    assert args.custom_ops is True and args.gate_in_kernel is False
    with pytest.raises(SystemExit):
        parse_args(common + ['--architecture', 'gdn1'])
    args = parse_args(common + ['--architecture', 'gdn2', '--gate-in-kernel'])
    assert args.gate_in_kernel is True
    with pytest.raises(SystemExit):
        parse_args([a for a in common if a != '--custom-ops'] + ['--architecture', 'gdn2', '--gate-in-kernel'])
    with pytest.raises(SystemExit):  # FLA's sub-chunk kernels need bounded log-decay; GDN2's is unbounded
        parse_args(common + ['--architecture', 'gdn2', '--safe-gate'])


def test_shared_pool_cli_requires_the_custom_operator_packed_execution():
    from scripts.benchmark_gated_delta import main as benchmark_main
    from scripts.train_recurrent_slots import parse_args
    common = ['--name', 'pool', '--steps', '1000', '--segment-size', '64', '--microbatch', '16',
              '--throughput-report', 'unused.json', '--architecture', 'gdn2', '--gdn-backend', 'fla',
              '--disable-recompute']
    args = parse_args(common + ['--custom-ops', '--fused-projections', '--shared-pool'])
    assert args.shared_pool and args.pool_banks is None and args.pool_writer == "routed" and args.pool_heads == 2
    args = parse_args(common + ['--custom-ops', '--fused-projections', '--shared-pool',
                                '--pool-banks', '8', '--decision-gate-reference', 'control'])
    assert args.pool_banks == 8 and args.decision_gate_reference == 'control'
    args = parse_args(common + ['--custom-ops', '--fused-projections', '--shared-pool', '--pool-writer', 'layer',
                                '--pool-heads', '1'])
    assert args.pool_writer == "layer" and args.pool_heads == 1
    for extra in (['--custom-ops', '--shared-pool'], ['--fused-projections', '--shared-pool'],
                  ['--custom-ops', '--fused-projections', '--shared-pool', '--gate-in-kernel'],
                  ['--custom-ops', '--fused-projections', '--shared-pool', '--pool-banks', '1'],
                  ['--custom-ops', '--fused-projections', '--shared-pool', '--pool-writer', 'layer', '--pool-banks', '12'],
                  ['--custom-ops', '--fused-projections', '--shared-pool', '--pool-heads', '0'],
                  ['--custom-ops', '--fused-projections', '--shared-pool', '--pool-writer', 'top2'],
                  ['--custom-ops', '--fused-projections', '--pool-banks', '8'],
                  ['--custom-ops', '--fused-projections', '--pool-writer', 'layer'],
                  ['--custom-ops', '--fused-projections', '--pool-heads', '4'],
                  ['--custom-ops', '--fused-projections', '--shared-pool', '--decision-gate-reference', 'pool'],
                  ['--custom-ops', '--fused-projections', '--shared-pool', '--decision-gate-reference', 'a/b']):
        with pytest.raises(SystemExit):
            parse_args(common + extra)
    with pytest.raises(SystemExit):
        benchmark_main(['--output', 'unused', '--architecture', 'gdn2', '--gdn-backend', 'fla',
                        '--disable-recompute', '--custom-ops', '--shared-pool'])
    with pytest.raises(SystemExit):
        benchmark_main(['--output', 'unused', '--architecture', 'gdn2', '--pool-banks', '8'])


def test_shared_pool_variant_is_measured_but_never_qualified_as_an_execution_schedule():
    from scripts.benchmark_gdn2_execution import EXECUTION_VARIANTS, VARIANTS, gradient_reference
    assert VARIANTS["shared_pool"]["model"] == "shared_pool" and VARIANTS["shared_pool"]["microbatch"] == 16
    assert "shared_pool" not in EXECUTION_VARIANTS and "custom_ops" in EXECUTION_VARIANTS
    with pytest.raises(ValueError, match="execution schedule"):
        gradient_reference("shared_pool", None, None)


def test_kernel_options_belong_to_custom_operator_execution():
    from scripts.benchmark_gated_delta import main as benchmark_main
    from scripts.benchmark_gdn2_execution import EXECUTION_CONFIG_KEYS, VARIANTS
    with pytest.raises(SystemExit):
        benchmark_main(['--output', 'unused', '--architecture', 'gdn2', '--gdn-backend', 'fla',
                        '--disable-recompute', '--fused-projections', '--gate-in-kernel'])
    assert VARIANTS["gate_in_kernel"]["custom_ops"] and VARIANTS["gate_in_kernel"]["gate_in_kernel"]
    # Kernel paths are execution options: the gradient gate compares them against the reference model.
    assert "gate_in_kernel" in EXECUTION_CONFIG_KEYS
    # FLA's safe_gate sub-chunk kernels exponentiate log-decay differences in both directions and need
    # per-token log-decay within [-5, 0); trained GDN2 checkpoints reach -20, so the option has no variant.
    assert not any("safe_gate" in variant for variant in VARIANTS.values()) and "safe_gate" not in EXECUTION_CONFIG_KEYS
    from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT
    with pytest.raises(TypeError):
        GatedDeltaGPT(gdn_backend="fla", disable_recompute=True, custom_ops=True, safe_gate=True)
    with pytest.raises(ValueError, match="custom-operator"):
        GatedDeltaGPT(gdn_backend="fla", disable_recompute=True, fused_projections=True, gate_in_kernel=True)
    with pytest.raises(ValueError, match="booleans"):
        GatedDeltaGPT(gdn_backend="fla", disable_recompute=True, custom_ops=True, gate_in_kernel=1)


def test_pinned_autotune_digest_binds_kernel_config_bytes(tmp_path):
    from pretraining.nanogpt_mini.gated_delta_runtime import pinned_autotune_digest
    with pytest.raises(ValueError):
        pinned_autotune_digest(tmp_path / "missing")
    with pytest.raises(ValueError):
        pinned_autotune_digest(tmp_path)
    (tmp_path / "kernel_a.json").write_text('{"autotune_entries": {}}')
    first = pinned_autotune_digest(tmp_path)
    (tmp_path / "kernel_b.json").write_text('{"autotune_entries": {}}')
    second = pinned_autotune_digest(tmp_path)
    (tmp_path / "kernel_a.json").write_text('{"autotune_entries": {"x": {}}}')
    third = pinned_autotune_digest(tmp_path)
    (tmp_path / "notes.txt").write_text("ignored")
    assert len({first, second, third}) == 3 and pinned_autotune_digest(tmp_path) == third


def test_runtime_policy_binds_pinned_config_directory(tmp_path, monkeypatch):
    from pretraining.nanogpt_mini.gated_delta_runtime import pinned_autotune_digest, runtime_autotuning_policy
    monkeypatch.delenv("FLA_CONFIG_DIR", raising=False)
    assert runtime_autotuning_policy()["fla_config_dir"] is None
    profile = ROOT / "pretraining/gated_delta/autotune/_policy_test"
    monkeypatch.setenv("FLA_CONFIG_DIR", str(profile))
    with pytest.raises(ValueError):
        runtime_autotuning_policy()
    (tmp_path / "kernel.json").write_text("{}")
    monkeypatch.setenv("FLA_CONFIG_DIR", str(tmp_path))
    policy = runtime_autotuning_policy()
    assert policy["fla_config_dir_sha256"] == pinned_autotune_digest(tmp_path)
    assert policy["fla_config_dir"] == str(Path(os.path.relpath(tmp_path.resolve(), ROOT)))


def selections(config, plain_entries=None):
    entry = dict(autotune_key=[64, 1024, 4, 128, "torch.bfloat16"], config=config)
    plain = dict(kernel_name="l2norm_fwd_kernel1", config_file_support=False, keys=["N"],
                 cache_results=True, entries=plain_entries or {})
    return {"fla.ops.gdn2.chunk_bwd.wy": dict(kernel_name="chunk_gdn2_bwd_kernel_wy", config_file_support=True,
                                              keys=["BT", "H"], cache_results=True, entries={"hash": entry}),
            "fla.ops.gdn2.empty": dict(kernel_name="chunk_gdn2_unused", config_file_support=True,
                                       keys=[], cache_results=True, entries={}),
            "fla.modules.l2norm.l2norm_fwd_kernel1": plain}


def test_pinned_profile_round_trip_and_mismatch_detection(tmp_path):
    from pretraining.nanogpt_mini.gated_delta_runtime import (
        pinned_autotune_mismatches, unpinnable_selections, write_pinned_autotune_profile)
    config = dict(kwargs=dict(BK=32, BV=32), num_warps=8, num_stages=2, num_ctas=1, maxnreg=None)
    plain = {"p": dict(autotune_key=[128, "torch.bfloat16"], config=dict(config, num_warps=4))}
    observed = selections(config, plain)
    with pytest.raises(ValueError):
        write_pinned_autotune_profile(tmp_path / "empty", selections(config) | {"fla.ops.gdn2.chunk_bwd.wy": dict(
            observed["fla.ops.gdn2.chunk_bwd.wy"], entries={})})
    assert write_pinned_autotune_profile(tmp_path / "profile", observed) == ["chunk_gdn2_bwd_kernel_wy"]
    written = json.loads((tmp_path / "profile/chunk_gdn2_bwd_kernel_wy.json").read_text())
    assert set(written) == {"kernel_name", "triton_version", "autotune_entries"}
    assert written["autotune_entries"]["hash"]["config"] == config
    assert not (tmp_path / "profile/l2norm_fwd_kernel1.json").exists()
    assert unpinnable_selections(observed) == {"fla.modules.l2norm.l2norm_fwd_kernel1": plain}
    assert pinned_autotune_mismatches(tmp_path / "profile", observed) == []
    retuned = selections(dict(config, num_warps=4), plain)
    mismatches = pinned_autotune_mismatches(tmp_path / "profile", retuned)
    assert [m["kernel"] for m in mismatches] == ["chunk_gdn2_bwd_kernel_wy"]
    assert mismatches[0]["pinned"]["config"] == config and mismatches[0]["observed"]["num_warps"] == 4
    unseen = selections(config) | {"fla.ops.gdn2.other": dict(
        kernel_name="chunk_gdn2_other", config_file_support=True, keys=[], cache_results=True,
        entries={"h2": dict(autotune_key=[1], config=config)})}
    assert [m["pinned"] for m in pinned_autotune_mismatches(tmp_path / "profile", unseen)] == [None]
    with pytest.raises(FileExistsError):
        write_pinned_autotune_profile(tmp_path / "profile", observed)


def artifact(gradients, noise, loss=2.0):
    return dict(loss=loss, gradients=gradients, noise=noise, microbatches=8,
                model_config={"custom_ops": False, "head_dim": 128}, microbatch_losses=[loss] * 8)


def test_gradient_gate_admits_only_sub_noise_kernel_differences():
    from scripts.benchmark_gdn2_execution import GRADIENT_BOUNDS, compare_gradients
    torch.manual_seed(0)
    reference = {"a": torch.randn(64), "b": torch.randn(16)}
    direction = {name: torch.nn.functional.normalize(torch.randn_like(g), dim=0) for name, g in reference.items()}
    norms = {name: float(g.norm()) for name, g in reference.items()}
    noise = {name: 0.5 * norms[name] for name in reference}  # signal-to-noise 2 per parameter
    expected = artifact(reference, noise)

    def candidate(relative, loss=2.0):
        return artifact({name: g + relative * norms[name] * direction[name] for name, g in reference.items()}, noise, loss)

    exact = compare_gradients(expected, expected)
    assert exact["passed"] and exact["strict_parameters"] == 2 and exact["maximum_noise_ratio"] == 0
    strict = compare_gradients(candidate(.01), expected)
    assert strict["passed"] and strict["strict_parameters"] == 2 and strict["noise_qualified_parameters"] == 0
    # 5% relative error is 10% of the sampling noise here: admitted by the noise clause only.
    within = compare_gradients(candidate(.05), expected)
    assert within["passed"] and within["strict_parameters"] == 0 and within["noise_qualified_parameters"] == 2
    assert within["maximum_noise_ratio"] == pytest.approx(.1, rel=1e-6)
    assert within["minimum_cosine_similarity"] < 1 and within["failed_parameters"] == []
    # 5% relative error is a third of a much smaller sampling noise: rejected.
    quiet = artifact(reference, {name: .15 * norms[name] for name in reference})
    assert not compare_gradients(candidate(.05), quiet)["passed"]
    assert compare_gradients(candidate(.05), quiet)["failed_parameters"] == ["a", "b"]
    # Bounded relative error even when noise is enormous.
    loud = artifact(reference, {name: 100 * norms[name] for name in reference})
    assert not compare_gradients(candidate(GRADIENT_BOUNDS["noise_relative_error"] + .01), loud)["passed"]
    assert compare_gradients(candidate(GRADIENT_BOUNDS["noise_relative_error"] - .01), loud)["passed"]
    # Zero noise leaves only the strict clause.
    silent = artifact(reference, {name: 0.0 for name in reference})
    assert compare_gradients(candidate(.01), silent)["passed"]
    assert not compare_gradients(candidate(.05), silent)["passed"]
    assert compare_gradients(candidate(.05), silent)["undefined_noise_parameters"] == ["a", "b"]
    assert not compare_gradients(candidate(0, loss=2.0 * 1.002), expected)["passed"]
    with pytest.raises(RuntimeError):
        compare_gradients(artifact({"a": reference["a"]}, {"a": noise["a"]}), expected)
    assert compare_gradients(dict(expected, model_config={"custom_ops": True, "head_dim": 128}), expected)["passed"]
    with pytest.raises(RuntimeError, match="configurations"):
        compare_gradients(dict(expected, model_config={"custom_ops": False, "head_dim": 64}), expected)
    with pytest.raises(RuntimeError):
        compare_gradients(candidate(float("nan")), expected)


def test_custom_operator_fake_kernels_declare_production_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from pretraining.nanogpt_mini import gated_delta_ops as ops
    batch, time, heads, key_dim, value_dim = 2, 128, 4, 128, 96
    with FakeTensorMode():
        q = torch.empty(batch, time, heads, key_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.empty(batch, time, heads, value_dim, dtype=torch.bfloat16, device="cuda")
        g = torch.empty(batch, time, heads, key_dim, dtype=torch.float32, device="cuda")
        raw_g = torch.empty(batch, time, heads, key_dim, dtype=torch.bfloat16, device="cuda")
        A_log = torch.empty(heads, dtype=torch.float32, device="cuda")
        dt_bias = torch.empty(heads * key_dim, dtype=torch.float32, device="cuda")
        for state_v_first in (False, True):
            for gate_in_kernel in (False, True):
                decay = (A_log, dt_bias) if gate_in_kernel else (None, None)
                gate = raw_g if gate_in_kernel else g
                outputs = ops.chunk_fwd(q, q, v, gate, q, v, *decay, None, None, key_dim ** -.5, state_v_first)
                shapes = [tuple(o.shape) for o in outputs]
                dtypes = [o.dtype for o in outputs]
                state = (batch, time // 64, heads) + ((value_dim, key_dim) if state_v_first else (key_dim, value_dim))
                assert shapes == [(batch, time, heads, value_dim), (batch, time, heads, key_dim), (batch, time, heads),
                                  (batch, time, heads, key_dim), (batch, time, heads), (batch, time, heads, key_dim),
                                  (batch, time, heads, 64), (batch, time, heads, 64), (batch, time, heads, key_dim),
                                  (batch, time, heads, value_dim), (batch, time, heads, key_dim),
                                  (batch, time, heads, key_dim), (batch, time, heads, value_dim), state]
                assert dtypes == [torch.bfloat16, torch.bfloat16, torch.float32, torch.bfloat16, torch.float32,
                                  torch.float32] + [torch.bfloat16] * 8
                grads = ops.chunk_bwd(outputs[0], *outputs[1:5], v, outputs[5], q, v, *outputs[6:],
                                      raw_g if gate_in_kernel else None, *decay, None, None, key_dim ** -.5,
                                      state_v_first)
                # dg takes the raw projection's dtype when the kernels computed the gate; the decay
                # parameter gradients are empty otherwise.
                expected = (q, q, v, gate, q, v, A_log, dt_bias) if gate_in_kernel else (q, q, v, g, q, v)
                assert [tuple(t.shape) for t in grads[:len(expected)]] == [tuple(t.shape) for t in expected]
                assert [t.dtype for t in grads[:3]] == [torch.bfloat16] * 3
                assert grads[3].dtype == (torch.bfloat16 if gate_in_kernel else torch.float32)
                assert [t.dtype for t in grads[4:6]] == [torch.float32] * 2
                if gate_in_kernel:
                    assert [t.dtype for t in grads[6:]] == [torch.float32] * 2
                else:
                    assert [tuple(t.shape) for t in grads[6:]] == [(0,), (0,)]
        x = torch.empty(batch, time, 3 * key_dim, dtype=torch.bfloat16, device="cuda")[..., key_dim:2 * key_dim]
        weight = torch.empty(key_dim, 4, dtype=torch.float32, device="cuda")
        y = ops.causal_conv1d_silu(x, weight)
        assert y.shape == x.shape and y.dtype == x.dtype and y.is_contiguous()
        dx, dw = ops.conv_bwd(x, y, weight)
        assert dx.shape == x.shape and dw.shape == weight.shape
        gate_weight = torch.empty(value_dim, dtype=torch.float32, device="cuda")
        y = ops.rms_norm_swish_gate(v, v, gate_weight, 1e-5)
        assert y.shape == v.shape and y.dtype == v.dtype
        _, rstd = ops.norm_fwd(v, v, gate_weight, 1e-5)
        assert rstd.shape == (batch * time * heads,) and rstd.dtype == torch.float32
        dx, dg, dw = ops.norm_bwd(y, v, v, gate_weight, rstd, 1e-5)
        assert dx.shape == dg.shape == v.shape and dw.shape == gate_weight.shape


def test_custom_operators_require_exact_strides():
    from pretraining.nanogpt_mini import gated_delta_ops as ops
    library = getattr(torch.ops, ops.LIBRARY)
    names = ("chunk_fwd", "chunk_bwd", "causal_conv1d_silu_fwd", "causal_conv1d_silu_bwd",
             "rms_norm_swish_gate_fwd", "rms_norm_swish_gate_bwd")
    for name in names:
        tags = set(getattr(library, name).default.tags)
        assert torch.Tag.needs_exact_strides in tags, name
        assert torch.Tag.pt2_compliant_tag in tags, name
        assert torch.Tag.flexible_layout not in tags, name


PMON = """# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command
# Idx           #    C/G      %      %      %      %      %      %     MB     MB    name
    0      10454     G      -      -      -      -      -      -    261      0    niri
    0      11952   C+G      -      -      -      -      -      -    588      0    renderD128 --cr
    0      48756     C     99     58      -      -      -      -  21642      0    trading_bot_0
    0      {own}     C     98     62      -      -      -      -   3016      0    python
    0      [N/A]     C      -      -      -      -      -      -      -      0    hidden
    0      77777     C      3      1      -      -      -      -      -      0    partial
    0      44206     G     12      2      -      -      -      -    664      0    chromium-app --
"""


def test_process_activity_parses_pmon_and_ignores_this_process(monkeypatch):
    import os
    import subprocess
    from pretraining.nanogpt_mini import gated_delta_runtime as runtime
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: PMON.format(own=os.getpid()))
    activity = runtime.gpu_process_activity()
    assert activity["unobservable"] == 2
    use = runtime.foreign_gpu_use(activity)
    assert use == dict(memory_mib=261 + 588 + 21642 + 664, compute_sm_percent=99, busiest_compute_pid=48756,
                       graphics_sm_percent=12, busiest_graphics_pid=44206, unobservable=2)


def test_process_activity_reports_observer_failure(monkeypatch):
    import subprocess
    from pretraining.nanogpt_mini import gated_delta_runtime as runtime
    def fail(*a, **k):
        raise subprocess.CalledProcessError(1, "nvidia-smi")
    monkeypatch.setattr(subprocess, "check_output", fail)
    assert runtime.gpu_process_activity() is None
    assert runtime.gpu_utilization_percent() is None


def idle(memory=2000, sm=0, kind="C"):
    return dict(processes=[dict(pid=1, kind=kind, sm_percent=sm, memory_mib=memory, command="job")], unobservable=0)


def test_exclusive_gpu_wait_requires_consecutive_idle_polls(monkeypatch):
    import time
    from pretraining.nanogpt_mini import gated_delta_runtime as runtime
    readings = iter([idle(26000, 99), idle(), idle(2000, 40), idle(), idle(), idle()])
    utilization = iter([100, 0, 100, 0, 0, 0])
    clock = [0.0]
    monkeypatch.setattr(runtime, "gpu_process_activity", lambda device_index=0: next(readings))
    monkeypatch.setattr(runtime, "gpu_utilization_percent", lambda device_index=0: next(utilization))
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    result = runtime.wait_for_exclusive_gpu(consecutive=3, interval_seconds=10.0, deadline_seconds=3600)
    assert result["polls"] == 6 and result["waited_seconds"] == 50.0 and result["observer_errors"] == 0
    assert result["memory_mib"] == 2000 and result["utilization_percent"] == 0
    monkeypatch.setattr(runtime, "gpu_process_activity", lambda device_index=0: idle(26000, 0))
    monkeypatch.setattr(runtime, "gpu_utilization_percent", lambda device_index=0: 0)
    with pytest.raises(RuntimeError, match="still in use"):
        runtime.wait_for_exclusive_gpu(consecutive=3, interval_seconds=10.0, deadline_seconds=35)


def test_exclusive_gpu_wait_refuses_to_run_blind(monkeypatch):
    import time
    from pretraining.nanogpt_mini import gated_delta_runtime as runtime
    monkeypatch.setattr(runtime, "gpu_utilization_percent", lambda device_index=0: 0)
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    scattered = iter([None, idle(), None, idle(), None, idle(), None, idle(), None, idle(), idle(), idle()])
    monkeypatch.setattr(runtime, "gpu_process_activity", lambda device_index=0: next(scattered))
    result = runtime.wait_for_exclusive_gpu(consecutive=3, interval_seconds=10.0, deadline_seconds=3600)
    assert result["observer_errors"] == 5 and result["polls"] == 12
    monkeypatch.setattr(runtime, "gpu_process_activity", lambda device_index=0: None)
    with pytest.raises(RuntimeError, match="observe"):
        runtime.wait_for_exclusive_gpu(consecutive=3, interval_seconds=10.0, deadline_seconds=3600)
    hidden = dict(processes=[dict(pid=7, kind="C", sm_percent=0, memory_mib=None, command="x")], unobservable=1)
    monkeypatch.setattr(runtime, "gpu_process_activity", lambda device_index=0: hidden)
    with pytest.raises(RuntimeError, match="still in use"):
        runtime.wait_for_exclusive_gpu(consecutive=3, interval_seconds=10.0, deadline_seconds=35)


def settle(sampler, count, key="samples", deadline_seconds=5.0):
    """Block until the sampler has taken ``count`` samples (or samples plus observer errors)."""
    import time
    start = time.monotonic()
    while True:
        summary = sampler.summary()
        observed = summary["samples"] + (summary["observer_errors"] if key == "observed" else 0)
        if observed >= count:
            return summary
        if time.monotonic() - start > deadline_seconds:
            raise AssertionError(f"sampler took {observed} of {count} samples in {deadline_seconds}s: {summary}")
        time.sleep(0.005)


def test_sampler_records_peaks_and_fails_closed(monkeypatch):
    from pretraining.nanogpt_mini import gated_delta_runtime as runtime
    readings = iter([idle(), idle(3000, 40), None, idle()])
    monkeypatch.setattr(runtime, "gpu_process_activity",
                        lambda device_index=0: next(readings, idle()))
    with runtime.ForeignGpuSampler(interval_seconds=0.01) as sampler:
        settle(sampler, 4, 'observed')
    summary = sampler.summary()
    assert summary["peak_foreign_memory_mib"] == 3000 and summary["peak_foreign_compute_sm_percent"] == 40
    assert summary["busiest_foreign_compute_pid"] == 1 and summary["observer_errors"] == 1 and not summary["exclusive"]
    with pytest.raises(RuntimeError, match="used the device"):
        sampler.require_exclusive()
    quiet = runtime.ForeignGpuSampler()
    with pytest.raises(RuntimeError, match="never observed"):
        quiet.require_exclusive()
    monkeypatch.setattr(runtime, "gpu_process_activity", lambda device_index=0: idle())
    with runtime.ForeignGpuSampler(interval_seconds=0.01) as sampler:
        settle(sampler, 2, 'samples')
    assert sampler.require_exclusive()["exclusive"]
    # Desktop redraws: a graphics row at 12% is recorded, not contention; a sustained 40% is.
    monkeypatch.setattr(runtime, "gpu_process_activity", lambda device_index=0: idle(600, 12, kind="G"))
    with runtime.ForeignGpuSampler(interval_seconds=0.01) as sampler:
        settle(sampler, 6, 'samples')
    # ...unless it persists: five consecutive samples above the compute limit is a graphics workload.
    with pytest.raises(RuntimeError, match="used the device"):
        sampler.require_exclusive()
    assert sampler.summary()["longest_foreign_graphics_burst_samples"] >= 5
    # Two short redraw bursts in twenty samples pass; both are recorded.
    bursts = iter(([idle(600, 12, kind="G")] * 2 + [idle(600, 0, kind="G")] * 8) * 2)
    monkeypatch.setattr(runtime, "gpu_process_activity", lambda device_index=0: next(bursts, idle(600, 0, kind="G")))
    with runtime.ForeignGpuSampler(interval_seconds=0.01) as sampler:
        settle(sampler, 20, 'samples')
    summary = sampler.require_exclusive()
    assert summary["peak_foreign_graphics_sm_percent"] == 12 and summary["longest_foreign_graphics_burst_samples"] == 2
    assert summary["foreign_graphics_samples_above_compute_limit"] == 4 and summary["foreign_graphics_bursts"] == 2
    # A workload pulsing four samples on and one off satisfies the run-length rule but not the duty share.
    pulses = iter(([idle(600, 12, kind="G")] * 4 + [idle(600, 0, kind="G")]) * 3)
    monkeypatch.setattr(runtime, "gpu_process_activity", lambda device_index=0: next(pulses, idle(600, 0, kind="G")))
    with runtime.ForeignGpuSampler(interval_seconds=0.01) as sampler:
        settle(sampler, 15, 'samples')
    with pytest.raises(RuntimeError, match="used the device"):
        sampler.require_exclusive()
    summary = sampler.summary()
    assert summary["longest_foreign_graphics_burst_samples"] == 4 and summary["foreign_graphics_bursts"] == 3
    assert summary["foreign_graphics_samples_above_compute_limit"] == 12
    # A blind sample extends a burst rather than ending it: 2 + blind + 2 is a five-sample burst.
    blind = iter([idle(600, 12, kind="G")] * 2 + [None] + [idle(600, 12, kind="G")] * 2)
    monkeypatch.setattr(runtime, "gpu_process_activity", lambda device_index=0: next(blind, idle(600, 0, kind="G")))
    with runtime.ForeignGpuSampler(interval_seconds=0.01) as sampler:
        settle(sampler, 5, 'samples')
    with pytest.raises(RuntimeError, match="used the device"):
        sampler.require_exclusive()
    assert sampler.summary()["longest_foreign_graphics_burst_samples"] == 5 and sampler.summary()["observer_errors"] == 1
    monkeypatch.setattr(runtime, "gpu_process_activity", lambda device_index=0: idle(600, 40, kind="C+G"))
    with runtime.ForeignGpuSampler(interval_seconds=0.01) as sampler:
        settle(sampler, 2, 'samples')
    with pytest.raises(RuntimeError, match="used the device"):
        sampler.require_exclusive()
    hidden = dict(processes=[dict(pid=7, kind="C", sm_percent=0, memory_mib=None, command="x")], unobservable=1)
    monkeypatch.setattr(runtime, "gpu_process_activity", lambda device_index=0: hidden)
    with runtime.ForeignGpuSampler(interval_seconds=0.01) as sampler:
        settle(sampler, 2, 'samples')
    with pytest.raises(RuntimeError, match="attributed"):
        sampler.require_exclusive()

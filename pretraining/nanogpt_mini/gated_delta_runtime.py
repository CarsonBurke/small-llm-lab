"""Compiled dense regions around verified opaque Gated Delta Triton operators.

The released operator explicitly disables Dynamo tracing. That declared kernel
boundary is intentional; compilation errors and unrelated graph breaks remain
fatal. The complete forward/backward is captured by the CUDA graph executor.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F


class CompiledGatedDeltaLoss:
    def __init__(self, model, segment_size=64):
        import torch._dynamo.config as config
        from torch._dynamo.utils import counters
        config.suppress_errors = False
        config.fail_on_recompile_limit_hit = True
        self.model = model
        self.segment_size = segment_size
        self._breaks_before = dict(counters["graph_break"])
        self._compiled = torch.compile(self._loss, fullgraph=False, dynamic=False)

    def _loss(self, inputs, targets, diagnostics=False):
        hidden, _ = self.model.forward_hidden(inputs, segment_size=self.segment_size)
        loss = F.cross_entropy(self.model.logits(hidden).flatten(0, 1),
                               targets.flatten(), reduction="sum")
        if diagnostics:
            # No training-state diagnostics are exposed by the official module.
            return (loss,)
        return loss

    def __call__(self, inputs, targets, *, diagnostics=False):
        if inputs.device.type != "cuda" or targets.device != inputs.device:
            raise ValueError("Gated Delta requires CUDA inputs and targets")
        if inputs.ndim != 2 or inputs.shape != targets.shape:
            raise ValueError("inputs and targets must share [batch,time] shape")
        return self._compiled(inputs, targets, diagnostics)

    def audit_graph_breaks(self):
        """Allow only verified opaque GDN/FLA kernel dispatch boundaries."""
        from torch._dynamo.utils import counters
        changes = {str(reason): count - self._breaks_before.get(reason, 0)
                   for reason, count in counters["graph_break"].items()
                   if count > self._breaks_before.get(reason, 0)}
        # FLA's dispatch decorator disables tracing for these Triton kernels.
        # They are captured by CUDA graphs, not substituted with eager math.
        memory_rule = self.model.config.get("memory_rule", "gdn2")
        if memory_rule not in {"gdn2", "scalar_delta"}:
            raise RuntimeError(f"Unknown compiled memory rule: {memory_rule}")
        operator = "chunk_gated_delta_rule" if memory_rule == "scalar_delta" else "chunk_gdn2"
        allowed = {operator, "causal_conv1d_fwd", "layer_norm_gated_fwd"}
        unexpected = []
        for reason in changes:
            functions = set(re.findall(r"<function ([A-Za-z_][A-Za-z_0-9]*) at ", reason))
            if "torch.compiler.disable" not in reason or not functions or not functions <= allowed:
                unexpected.append(reason)
        if unexpected:
            raise RuntimeError(f"Unexpected Gated Delta graph breaks: {unexpected}")
        return changes


def build_gated_delta_optimizers(model):
    """Matched-mini optimizer adapted explicitly for convolution and decay gates."""
    from scripts.train_recurrent_slots import Muon
    embedding, head = model.embed.weight, model.proj.weight
    special = {id(embedding), id(head)}
    scalars, matrices, convolutions, no_decay = [], [], [], []
    for name, parameter in model.named_parameters():
        if id(parameter) in special:
            continue
        if getattr(parameter, "_no_weight_decay", False):
            no_decay.append(parameter)
        elif "_conv1d." in name:
            convolutions.append(parameter)
        elif parameter.ndim == 2:
            matrices.append(parameter)
        elif parameter.ndim < 2:
            scalars.append(parameter)
        else:
            raise ValueError(f"Unclassified Gated Delta parameter: {name}")
    groups = [{"params": [embedding], "lr": 0.7},
              {"params": [head], "lr": 0.004}]
    for parameters, options in ((scalars, {"lr": 0.015}),
                                (convolutions, {"lr": 0.002}),
                                (no_decay, {"lr": 0.002, "weight_decay": 0.0})):
        if parameters:
            groups.append({"params": parameters, **options})
    if not matrices or not convolutions or not no_decay:
        raise ValueError("Expected dense, short-convolution and no-decay gate groups")
    result = [torch.optim.AdamW(groups, betas=(0.8, 0.95), eps=1e-10,
                               weight_decay=0.001, fused=True), Muon(matrices)]
    grouped = [p for optimizer in result for group in optimizer.param_groups for p in group["params"]]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(model.parameters()):
        raise ValueError("Gated Delta optimizers must cover every parameter exactly once")
    return result


def runtime_dependency_versions():
    from importlib.metadata import PackageNotFoundError, version
    result = {}
    for distribution in ("fla-core", "flash-linear-attention", "triton"):
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError:
            result[distribution] = None
    return result


def _fla_dependency_provenance(required):
    """Hash installed FLA sources without importing its model/frontend modules.

    RECORD mismatches are disclosed rather than silently called upstream code.
    Benchmark and training must agree on the exact installed sources.
    """
    import base64
    import hashlib
    from importlib.metadata import distribution

    versions, sources, mismatches = {}, {}, []
    for name in ("fla-core", "flash-linear-attention"):
        installed = distribution(name)
        versions[name] = installed.version
        files = installed.files
        if files is None:
            raise ValueError(f"Missing installed file manifest for {name}")
        for entry in files:
            relative = str(entry)
            if not relative.startswith("fla/") or not relative.endswith(".py"):
                continue
            digest = hashlib.sha256(installed.locate_file(entry).read_bytes()).digest()
            key = f"{name}:{relative}"
            sources[key] = digest.hex()
            recorded = entry.hash
            actual = base64.urlsafe_b64encode(digest).decode().rstrip("=")
            if recorded is None or recorded.mode != "sha256" or recorded.value != actual:
                mismatches.append(key)
    for suffix in required:
        if not any(key.endswith(":" + suffix) for key in sources):
            raise ValueError(f"Missing installed FLA source: {suffix}")
    return {"distribution_versions": versions, "source_sha256": dict(sorted(sources.items())),
            "wheel_record_mismatches": sorted(mismatches)}


def scalar_dependency_provenance():
    return _fla_dependency_provenance(("fla/layers/gated_deltanet.py", "fla/ops/gated_delta_rule/chunk.py"))


def gdn2_dependency_provenance():
    """Bind optimized GDN2 and its installed FLA helpers to executed source."""
    return _fla_dependency_provenance(("fla/ops/gdn2/chunk.py",))


def verify_throughput_report(args):
    """Require current complete timing evidence; explicitly mark slow diagnostics."""
    import hashlib
    import json
    import math
    import statistics
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    path = Path(args.throughput_report)
    if not path.is_absolute():
        path = root / path
    raw = path.read_bytes()
    report = json.loads(raw)
    candidate, baseline = report["candidate"], report["baseline"]
    if args.architecture not in {"gdn1", "gdn2"}:
        raise ValueError("Throughput report must target gdn1 or gdn2")
    scalar = args.architecture == "gdn1"
    diagnostic = args.allow_slow_diagnostic
    if diagnostic and scalar:
        raise ValueError("Slow diagnostics are only supported for GDN2")
    expected_config = dict(vocab_size=1024, num_layers=6, model_dim=512,
                           head_dim=args.memory_head_dim, mixer_dim=args.memory_width, expand_v=args.value_expansion, use_short_conv=True,
                           conv_size=4, allow_neg_eigval=False, mixer_norm_eps=1e-5,
                           kernel_chunk_size=64, fused_projections=args.fused_projections)
    if scalar:
        expected_config.update(memory_rule="scalar_delta", initialization="gdn2_matched_distributions")
    else:
        expected_config.update(gdn_backend=args.gdn_backend, state_v_first=args.state_v_first,
                               disable_recompute=args.disable_recompute)
    expected_flags = {"status": "completed",
                      "batch_tokens": 524288, "seq_len": 1024,
                      "optimizer_included": True, "compiled": True,
                      "cuda_graph": True, "checkpointing": False}
    for key, value in expected_flags.items():
        if report.get(key) != value:
            raise ValueError(f"Gated Delta throughput report has invalid {key}")
    expected_model = "scalar_delta" if scalar else "gated_delta"
    if candidate.get("model") != expected_model:
        raise ValueError("Gated Delta throughput candidate model differs")
    if report.get("repeats", 0) < 5 or baseline.get("model") != "nanogpt_mini":
        raise ValueError("Gated Delta gate needs at least five updates against plain mini")
    if (candidate.get("model_config") != expected_config or args.segment_size != 64
            or candidate.get("chunk_size") != 64 or candidate.get("microbatch") != args.microbatch
            or baseline.get("microbatch") != 64):
        raise ValueError("Gated Delta throughput configuration does not match training")
    if scalar and report.get("scalar_dependency_provenance") != scalar_dependency_provenance():
        raise ValueError("Scalar delta installed-source provenance differs")
    if not scalar and report.get("gdn2_dependency_provenance") != gdn2_dependency_provenance():
        raise ValueError("GDN2 installed-source provenance differs")
    if not scalar and (candidate.get("validation_microbatch") != args.microbatch
                       or candidate.get("validation_residency_passed") is not True):
        raise ValueError("GDN2 needs co-resident training and validation graph qualification")
    if report.get("dependency_versions") != runtime_dependency_versions():
        raise ValueError("Gated Delta throughput dependency versions differ")
    if report.get("torch") != str(torch.__version__):
        raise ValueError("Gated Delta throughput PyTorch version differs")
    if report.get("gpu") != torch.cuda.get_device_name():
        raise ValueError("Gated Delta throughput GPU differs")
    baseline_rate, candidate_rate = baseline["tokens_per_second"], candidate["tokens_per_second"]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
               for value in (baseline_rate, candidate_rate)):
        raise ValueError("Gated Delta throughput must be finite and positive")
    if not diagnostic and candidate_rate < 1.05 * baseline_rate:
        raise ValueError("Gated Delta throughput improvement is below five percent")
    for arm in (baseline, candidate):
        samples = arm.get("update_seconds", [])
        if (len(samples) < 5 or arm.get("measured_optimizer_updates", 0) != len(samples)
                or arm.get("warmup_optimizer_updates", 0) < 5
                or not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                           for value in samples)):
            raise ValueError("Gated Delta gate requires complete positive timing samples")
        if not math.isclose(arm["tokens_per_second"], 524288 / statistics.median(samples), rel_tol=1e-6):
            raise ValueError("Gated Delta throughput rate disagrees with measured update times")
    separated = max(candidate["update_seconds"]) < min(baseline["update_seconds"])
    speed_passed = candidate_rate >= 1.05 * baseline_rate and separated
    if report.get("gate_passed") is not speed_passed:
        raise ValueError("Gated Delta gate_passed disagrees with measured update times")
    if not diagnostic and not separated:
        raise ValueError("Gated Delta repeated timings overlap; repeat benchmark before training")
    files = [root / name for name in (
        "pretraining/nanogpt_mini/gated_delta_model.py",
        "pretraining/nanogpt_mini/gated_delta_runtime.py",
        "pretraining/nanogpt_mini/chunk_memory_runtime.py",
        "pretraining/nanogpt_mini/nanogpt_mini_model.py",
        "pretraining/nanogpt_mini/recurrent_slots_runtime.py",
        "scripts/train_recurrent_slots.py", "scripts/benchmark_gated_delta.py",
        "scripts/benchmark_chunk_memory.py")]
    if scalar:
        files.append(root / "pretraining/nanogpt_mini/scalar_delta_model.py")
    vendor = root / "pretraining/gated_delta/vendor"
    if not vendor.is_dir():
        raise ValueError("Missing pinned GDN-2 kernel directory")
    files.extend(p for p in vendor.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for source in files:
        key = str(source.relative_to(root))
        if report.get("source_sha256", {}).get(key) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError(f"Gated Delta throughput source mismatch: {key}")
    return dict(gpu=report["gpu"], path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                candidate_tokens_per_second=candidate_rate, baseline_tokens_per_second=baseline_rate,
                speed_passed=speed_passed, speedup=candidate_rate / baseline_rate,
                diagnostic_only=not speed_passed)

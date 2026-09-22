"""Compare equivalent GDN2 execution schedules; every invocation belongs in mlq.

Controls bracket candidates to expose clock/thermal drift. All measurements
include full optimizer updates and co-resident validation/training CUDA graphs.
No learning hyperparameters, widths, sequence lengths or dtypes are changed.

Candidates are qualified on the complete production update gradient (eight
B64 x T1024 microbatches). Kernel schedules legitimately differ at the level
of floating-point accumulation order, so beyond a strict relative bound a
difference is accepted only when it is small next to the gradient's own
microbatch sampling noise, which is what learning can actually resolve.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch._inductor.config as inductor_config

from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT
from pretraining.nanogpt_mini.gated_delta_pool import SharedPoolGatedDeltaGPT
from pretraining.nanogpt_mini.gated_delta_runtime import compiled_gated_delta_loss, gdn2_dependency_provenance
from scripts.benchmark_gated_delta import benchmark_candidate
from scripts.train_recurrent_slots import atomic_json


VARIANTS = {
    "reference": dict(fused_projections=True, autotune=False),
    "reference_repeat": dict(fused_projections=True, autotune=False),
    "unpacked": dict(fused_projections=False, autotune=False),
    "custom_ops": dict(fused_projections=True, custom_ops=True, autotune=False),
    # Installed-kernel option of the custom-operator execution. FLA's safe_gate
    # sub-chunk kernels are not a variant: trained GDN2 log-decay leaves the
    # [-5, 0) range they require (pretraining/gated_delta/EXECUTION_OPTIMIZATION.md).
    "gate_in_kernel": dict(fused_projections=True, custom_ops=True, gate_in_kernel=True, autotune=False),
    # Not an execution schedule of the same model: the two-pass shared pool changes the
    # architecture, so it is measured and pinned at its own microbatch but never qualified
    # against the reference gradient.
    "shared_pool": dict(fused_projections=True, custom_ops=True, autotune=False,
                        model="shared_pool", microbatch=16),
}
MICROBATCH = 64
UPDATE_ROWS = 512
EXECUTION_VARIANTS = tuple(name for name, options in VARIANTS.items() if "model" not in options)
# Execution options select kernels, layouts and graph structure for the same
# mathematical model; qualification compares candidates across them.
EXECUTION_CONFIG_KEYS = frozenset({"custom_ops", "fused_projections", "gdn_backend", "state_v_first",
                                   "disable_recompute", "gate_in_kernel"})
GRADIENT_BOUNDS = dict(loss_relative_error=1e-3, strict_relative_error=.015,
                       noise_relative_error=.10, noise_fraction=.25)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def execution_sources(*scripts):
    """Every source whose bytes determine a GDN2 execution measurement."""
    paths = [Path(script).resolve() for script in scripts]
    paths += [ROOT / "scripts" / name for name in
              ("benchmark_gdn2_execution.py", "benchmark_gated_delta.py",
               "benchmark_chunk_memory.py", "train_recurrent_slots.py")]
    paths += [ROOT / "pretraining/nanogpt_mini" / name for name in
              ("gated_delta_model.py", "gated_delta_ops.py", "gated_delta_pool.py", "gated_delta_bank_linear.py", "gated_delta_runtime.py",
               "recurrent_slots_runtime.py", "chunk_memory_runtime.py", "nanogpt_mini_model.py")]
    paths += [p for p in (ROOT / "pretraining/gated_delta/vendor").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts]
    return sorted(set(paths))


def snapshot_sources(paths, destination):
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    for path in paths:
        target = Path(destination) / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    return hashes


def build_variant(name):
    """The variant's model plus its microbatch; every variant is a fresh seed-1337 construction."""
    options = dict(VARIANTS[name])
    options.pop("autotune")
    kind = options.pop("model", "gated_delta")
    microbatch = options.pop("microbatch", MICROBATCH)
    torch.manual_seed(1337)
    if kind == "shared_pool":
        return SharedPoolGatedDeltaGPT(gdn_backend="fla", disable_recompute=True, **options), microbatch
    return GatedDeltaGPT(gdn_backend="fla", disable_recompute=True, **options), microbatch


def measure(name, inputs, targets, repeats):
    autotune = VARIANTS[name]["autotune"]
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    model, microbatch = build_variant(name)
    with inductor_config.patch(max_autotune=autotune, max_autotune_gemm_backends="ATEN,TRITON"):
        result = benchmark_candidate(model, inputs, targets, microbatch=microbatch, chunk_size=64, repeats=repeats)
        result["inductor_settings"] = dict(max_autotune=inductor_config.max_autotune,
                                           max_autotune_gemm=inductor_config.max_autotune_gemm,
                                           max_autotune_gemm_backends=inductor_config.max_autotune_gemm_backends)
    result["execution_options"] = VARIANTS[name]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def gradient_reference(name, inputs, targets):
    """Full production update derivatives; CPU holds only comparison artifacts.

    Returns the summed gradient of all eight microbatches, exactly what the
    optimizer consumes, plus each parameter's microbatch sampling noise: the
    standard deviation of that sum under independent microbatch resampling,
    sqrt(8/7 * sum_i ||G_i - mean(G)||^2), estimated from the same eight
    microbatch gradients. Sums are formed in float64.
    """
    if name not in EXECUTION_VARIANTS:
        raise ValueError(f"{name} is not an execution schedule of the reference model")
    autotune = VARIANTS[name]["autotune"]
    if inputs.shape != (UPDATE_ROWS, 1024) or targets.shape != inputs.shape:
        raise ValueError("Qualification requires one complete production update of inputs")
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    model, _ = build_variant(name)
    model = model.cuda().train()
    with torch.no_grad():
        # Nonzero readout and MLP output weights exercise every backward path;
        # training initialization would leave earlier MLP derivatives zero.
        model.proj.weight.normal_(std=.01)
        for block in model.blocks:
            block.mlp.proj.weight.normal_(std=.01)
    parameters = dict(model.named_parameters())
    sums = {key: torch.zeros_like(p, dtype=torch.float64) for key, p in parameters.items()}
    squares = {key: torch.zeros((), device="cuda", dtype=torch.float64) for key in parameters}
    microbatch_losses = []
    with inductor_config.patch(max_autotune=autotune, max_autotune_gemm_backends="ATEN,TRITON"):
        loss_fn = compiled_gated_delta_loss(model, 64)
        for row in range(0, UPDATE_ROWS, MICROBATCH):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                loss = loss_fn(inputs[row:row + MICROBATCH], targets[row:row + MICROBATCH])
            loss.backward()
            microbatch_losses.append(float(loss.detach()))
            for key, parameter in parameters.items():
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"Missing/nonfinite qualification gradient: {key}")
                gradient = parameter.grad.double()
                sums[key] += gradient
                squares[key] += gradient.square().sum()
            del loss
        loss_fn.audit_graph_breaks()
    count = UPDATE_ROWS // MICROBATCH
    gradients, noise = {}, {}
    for key in parameters:
        total = sums[key]
        if float(total.norm()) == 0:
            raise RuntimeError(f"Unexercised qualification gradient: {key}")
        # sum_i ||G_i - mean||^2 = sum_i ||G_i||^2 - ||sum_i G_i||^2 / n
        scatter = float(squares[key]) - float(total.square().sum()) / count
        noise[key] = math.sqrt(max(scatter, 0.0) * count / (count - 1))
        gradients[key] = total.float().cpu()
    result = dict(loss=sum(microbatch_losses) / (UPDATE_ROWS * 1024), microbatch_losses=microbatch_losses,
                  gradients=gradients, noise=noise, microbatches=count,
                  model_config=dict(model.config))
    del loss_fn, model, parameters, sums, squares
    gc.collect()
    torch.cuda.empty_cache()
    return result


def compare_gradients(actual, expected):
    """Learning-relevant parity of one complete update gradient.

    A parameter passes when its relative error is within the strict bound, or
    when a larger (still bounded) error is a small fraction of that
    parameter's microbatch sampling noise. Both estimates come from the
    reference arm so the candidate cannot loosen its own gate.

    The noise clause establishes that one update cannot distinguish the
    candidate's gradient from the reference's. A kernel discrepancy could
    still be a bias that persists across updates while sampling noise
    averages out, so this gate screens candidates; only the matched
    full-length training run verifies learning.
    """
    bounds = GRADIENT_BOUNDS
    if actual["gradients"].keys() != expected["gradients"].keys():
        raise RuntimeError("Qualification parameter sets differ")
    def learning_config(artifact):
        return {key: value for key, value in artifact["model_config"].items() if key not in EXECUTION_CONFIG_KEYS}
    if learning_config(actual) != learning_config(expected):
        raise RuntimeError("Qualification model configurations differ beyond execution options")
    if actual["microbatches"] != expected["microbatches"]:
        raise RuntimeError("Qualification microbatch counts differ")
    parameters = {}
    for name, gradient in actual["gradients"].items():
        reference = expected["gradients"][name].double()
        gradient = gradient.double()
        difference = float((gradient - reference).norm())
        reference_norm = float(reference.norm())
        noise = expected["noise"][name]
        cosine = float(torch.dot(gradient.flatten(), reference.flatten()) /
                       (gradient.norm() * reference.norm()).clamp_min(1e-300))
        if not all(math.isfinite(v) for v in (difference, reference_norm, noise, cosine)) or reference_norm <= 0:
            raise RuntimeError(f"Nonfinite or degenerate qualification gradient: {name}")
        relative = difference / reference_norm
        # A zero noise estimate means identical microbatch gradients; only the
        # strict bound can then qualify the parameter.
        noise_ratio = difference / noise if noise > 0 else None
        strict = relative <= bounds["strict_relative_error"]
        within_noise = (noise_ratio is not None and relative <= bounds["noise_relative_error"]
                        and noise_ratio <= bounds["noise_fraction"])
        parameters[name] = dict(
            passed=strict or within_noise, strict=strict, within_noise=within_noise,
            relative_error=relative, difference_norm=difference, reference_norm=reference_norm,
            actual_norm=float(gradient.norm()), cosine_similarity=cosine,
            reference_noise=noise, actual_noise=actual["noise"][name], noise_ratio=noise_ratio,
            reference_signal_to_noise=reference_norm / noise if noise > 0 else None)
    loss_error = abs(actual["loss"] - expected["loss"]) / abs(expected["loss"])
    if not math.isfinite(loss_error):
        raise RuntimeError("Nonfinite qualification loss")
    relative_errors = {name: p["relative_error"] for name, p in parameters.items()}
    noise_ratios = {name: p["noise_ratio"] for name, p in parameters.items() if p["noise_ratio"] is not None}
    passed = loss_error <= bounds["loss_relative_error"] and all(p["passed"] for p in parameters.values())
    return dict(passed=passed, loss_relative_error=loss_error,
                maximum_parameter_gradient_relative_error=max(relative_errors.values()),
                worst_parameter=max(relative_errors, key=relative_errors.get),
                maximum_noise_ratio=max(noise_ratios.values()) if noise_ratios else None,
                worst_noise_ratio_parameter=max(noise_ratios, key=noise_ratios.get) if noise_ratios else None,
                undefined_noise_parameters=sorted(set(parameters) - set(noise_ratios)),
                strict_parameters=sum(p["strict"] for p in parameters.values()),
                noise_qualified_parameters=sum(not p["strict"] and p["within_noise"] for p in parameters.values()),
                failed_parameters=sorted(name for name, p in parameters.items() if not p["passed"]),
                minimum_cosine_similarity=min(p["cosine_similarity"] for p in parameters.values()),
                parameter_comparisons=parameters, bounds=dict(bounds),
                microbatches=expected["microbatches"], batch_size=MICROBATCH, seq_len=1024,
                noise_definition="standard deviation of the summed update gradient under microbatch resampling, "
                                 "sqrt(n/(n-1) * sum_i ||G_i - mean||^2) over the reference arm's n microbatch gradients")


def qualification_summary(comparison):
    return {key: value for key, value in comparison.items()
            if key not in {"parameter_comparisons", "bounds", "noise_definition"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=[v for v in EXECUTION_VARIANTS if v != "reference"],
                        required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 5 or len(set(args.variants)) != len(args.variants):
        parser.error("Use at least five repeats and unique variants")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required; submit through mlq")
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    args.output.mkdir(parents=True)
    paths = execution_sources(Path(__file__))
    hashes = snapshot_sources(paths, args.output / "sources")
    report = dict(status="running", gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  batch_tokens=524288, seq_len=1024, microbatch=MICROBATCH,
                  gradient_qualification_bounds=dict(GRADIENT_BOUNDS),
                  warmups=5, repeats=args.repeats, validation_training_graphs_co_resident=True,
                  optimizer_included=True, host_data_transfer_included=False,
                  precision="unchanged BF16 compute, FP32 non-embedding parameters",
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  source_sha256=hashes, installed_fla=gdn2_dependency_provenance(), cases={})
    output = args.output / "benchmark.json"
    atomic_json(output, report)
    try:
        torch.manual_seed(1337)
        inputs = torch.randint(1024, (UPDATE_ROWS, 1024), device="cuda", dtype=torch.int32)
        targets = inputs.roll(-1, 1).long()
        print("Qualifying production-shape derivatives", flush=True)
        expected = gradient_reference("reference", inputs, targets)
        report["gradient_qualification"] = {}
        for name in args.variants:
            actual = gradient_reference(name, inputs, targets)
            comparison = compare_gradients(actual, expected)
            del actual
            report["gradient_qualification"][name] = comparison
            atomic_json(output, report)
            print(json.dumps(dict(qualification=name, **qualification_summary(comparison))), flush=True)
            if not comparison["passed"]:
                raise RuntimeError(f"Production-shape gradient parity failed: {name}")
        del expected
        sequence = [("reference_before", "reference"), *[(name, name) for name in args.variants],
                    ("reference_after", "reference")]
        for label, name in sequence:
            print(f"Measuring {label}", flush=True)
            result = measure(name, inputs, targets, args.repeats)
            report["cases"][label] = result
            atomic_json(output, report)
            print(json.dumps(dict(case=label, tokens_per_second=result["tokens_per_second"],
                                  update_seconds=result["update_seconds"])), flush=True)
        controls = [report["cases"][name] for name in ("reference_before", "reference_after")]
        fastest_control_median = min(c["median_update_seconds"] for c in controls)
        fastest_control_sample = min(t for c in controls for t in c["update_seconds"])
        comparisons = {}
        for name in args.variants:
            candidate = report["cases"][name]
            speedup = fastest_control_median / candidate["median_update_seconds"]
            separated = max(candidate["update_seconds"]) < fastest_control_sample
            comparisons[name] = dict(speedup_vs_faster_control=speedup,
                                     samples_faster_than_both_controls=separated,
                                     improvement_gate_passed=speedup >= 1.05 and separated)
        report.update(status="completed", comparisons=comparisons,
                      reference_drift_ratio=max(c["median_update_seconds"] for c in controls) / fastest_control_median)
        if any(sha256(ROOT / path) != digest for path, digest in hashes.items()):
            raise RuntimeError("Benchmark sources changed during execution")
        if gdn2_dependency_provenance() != report["installed_fla"]:
            raise RuntimeError("Installed FLA sources changed during execution")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(output, report)


if __name__ == "__main__":
    main()

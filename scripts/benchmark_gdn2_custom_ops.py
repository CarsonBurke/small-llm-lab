"""Bracket custom-operator GDN2 execution against production in fresh processes.

Every arm is a separate process because FLA and Triton read their autotuning
policy at import time. Each child qualifies the complete production update
gradient, snapshots every FLA kernel selection, then times full optimizer
updates. The parent compares CPU gradient artifacts under the learning-relevant
gate of scripts/benchmark_gdn2_execution.py and only then reads timings.

Brackets (--bracket):
  custom_ops
    reference_before   graph-break execution, production autotuning policy
    custom_ops         whole-graph custom operators, production autotuning policy
    custom_ops_pinned  whole-graph custom operators, strict pinned FLA profile
    reference_after    drift control
  kernel_options
    reference_before, custom_ops and reference_after as above, plus the
    installed-kernel option of the custom-operator execution under the
    production policy: gate_in_kernel. Fresh tuning throughout, because the
    pinned profile fixes selections for the kernels the production path
    launches, not for this variant.

Run via mlq.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_recurrent_slots import atomic_json  # noqa: E402

POLICY_VARIABLES = ("FLA_CACHE_RESULTS", "FLA_CACHE_MODE", "TRITON_CACHE_AUTOTUNING", "FLA_CONFIG_DIR")
BRACKETS = {
    "custom_ops": (("reference_before", "reference", "production"),
                   ("custom_ops", "custom_ops", "production"),
                   ("custom_ops_pinned", "custom_ops", "pinned"),
                   ("reference_after", "reference", "production")),
    "kernel_options": (("reference_before", "reference", "production"),
                       ("custom_ops", "custom_ops", "production"),
                       ("gate_in_kernel", "gate_in_kernel", "production"),
                       ("reference_after", "reference", "production")),
}
CONTROLS = ("reference_before", "reference_after")


def candidates(arms):
    return tuple(label for label, _, _ in arms if label not in CONTROLS)


def arm_environment(policy, profile):
    environment = {key: value for key, value in os.environ.items() if key not in POLICY_VARIABLES}
    if policy == "pinned":
        environment.update(FLA_CACHE_RESULTS="0", FLA_CACHE_MODE="strict", TRITON_CACHE_AUTOTUNING="0",
                           FLA_CONFIG_DIR=str(profile))
    return environment


def expected_policy(policy, profile):
    from pretraining.nanogpt_mini.gated_delta_runtime import pinned_autotune_digest
    if policy == "pinned":
        return dict(fla_cache_results=False, fla_cache_mode="strict",
                    fla_config_dir=os.path.relpath(profile, ROOT), fla_config_dir_sha256=pinned_autotune_digest(profile),
                    triton_cache_autotuning=False, effective_triton_persistent_results=False)
    return dict(fla_cache_results=True, fla_cache_mode="disabled", fla_config_dir=None, fla_config_dir_sha256=None,
                triton_cache_autotuning=False, effective_triton_persistent_results=True)


def child(args):
    # Imports intentionally follow the parent's environment setup.
    import torch
    from pretraining.nanogpt_mini.gated_delta_runtime import (
        ForeignGpuSampler, fla_autotuner_selections, gdn2_dependency_provenance, pinned_autotune_mismatches,
        runtime_autotuning_policy, unpinnable_selections, wait_for_exclusive_gpu)
    from scripts.benchmark_gdn2_execution import gradient_reference, measure
    label, variant, policy = next(arm for arm in BRACKETS[args.bracket] if arm[0] == args.child)
    expected = expected_policy(policy, args.profile)
    actual = runtime_autotuning_policy()
    if {key: actual[key] for key in expected} != expected:
        raise RuntimeError(f"Effective autotuning policy {actual} does not match {expected}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required; submit through mlq")
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    provenance = gdn2_dependency_provenance()
    manifest = None
    if policy == "pinned":
        manifest = json.loads((args.profile / "manifest.json").read_text())
        if manifest["installed_fla"] != provenance or manifest["gpu"] != torch.cuda.get_device_name():
            raise RuntimeError("Pinned profile provenance does not match this process")
    # Work launched outside the queue can hold the device; the child blocks until it is idle and
    # samples other processes throughout so contended measurements never qualify.
    exclusivity = dict(before=wait_for_exclusive_gpu())
    torch.manual_seed(1337)
    inputs = torch.randint(1024, (512, 1024), device="cuda", dtype=torch.int32)
    targets = inputs.roll(-1, 1).long()
    report = dict(status="running", label=label, variant=variant, policy_name=policy, gpu_exclusivity=exclusivity,
                  environment={key: os.environ.get(key) for key in POLICY_VARIABLES}, policy=actual,
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), installed_fla=provenance,
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    output = args.output / "child.json"
    atomic_json(output, report)
    try:
        with ForeignGpuSampler() as sampler:
            try:
                artifact = gradient_reference(variant, inputs, targets)
                torch.save(artifact, args.output / "gradients.pt")
                report["qualification_loss"] = artifact["loss"]
                del artifact
                report["qualification_selections"] = fla_autotuner_selections()
                atomic_json(output, report)
                report["measurement"] = measure(variant, inputs, targets, args.repeats)
                report["timing_selections"] = fla_autotuner_selections()
            finally:
                exclusivity["during"] = sampler.summary()
        sampler.require_exclusive()
        if policy == "pinned":
            if manifest["model_config"] != report["measurement"]["model_config"]:
                raise RuntimeError("Pinned profile was tuned for a different model configuration")
            for phase in ("qualification_selections", "timing_selections"):
                mismatches = pinned_autotune_mismatches(args.profile, report[phase])
                if mismatches:
                    report["pinned_mismatches"] = mismatches
                    raise RuntimeError(f"Kernel selections departed from the pinned profile during {phase}")
            # Plain Triton autotuners retune in-process; differences from the
            # profile's tuning run are reported, not gated.
            observed = unpinnable_selections(report["timing_selections"])
            report["unpinnable_selection_differences"] = {
                name: dict(profile=manifest["unpinnable_selections"].get(name), observed=entries)
                for name, entries in observed.items() if manifest["unpinnable_selections"].get(name) != entries}
        if gdn2_dependency_provenance() != provenance:
            raise RuntimeError("Installed FLA changed during child")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bracket", choices=sorted(BRACKETS), default="custom_ops")
    parser.add_argument("--profile", type=Path, help="pinned FLA profile directory from scripts/pin_gdn2_autotune.py; "
                                                     "required by brackets with a pinned arm")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--child", choices=sorted({arm[0] for arms in BRACKETS.values() for arm in arms}),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("At least five complete optimizer updates are required")
    arms = BRACKETS[args.bracket]
    pinned = any(policy == "pinned" for _, _, policy in arms)
    if pinned and args.profile is None:
        parser.error(f"--bracket {args.bracket} has a pinned arm; pass --profile")
    if args.profile is not None:
        args.profile = args.profile.resolve()
        if not (args.profile / "manifest.json").is_file():
            parser.error(f"{args.profile} is not a pinned profile directory")
    if args.child:
        child(args)
        return
    from scripts.benchmark_gdn2_execution import (
        GRADIENT_BOUNDS, compare_gradients, execution_sources, qualification_summary, sha256, snapshot_sources)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    hashes = snapshot_sources(execution_sources(Path(__file__)), args.output / "sources")
    profile_files = sorted(p for p in args.profile.iterdir() if p.is_file()) if pinned else []
    profile_hashes = {p.name: sha256(p) for p in profile_files}
    for path in profile_files:
        destination = args.output / "profile" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    report = dict(status="running", qualified=False, bracket=args.bracket, source_sha256=hashes,
                  profile=os.path.relpath(args.profile, ROOT) if pinned else None, profile_sha256=profile_hashes,
                  arms=[dict(label=label, variant=variant, policy=policy) for label, variant, policy in arms],
                  cases={}, batch_tokens=524288, seq_len=1024, microbatch=64, warmups=5, repeats=args.repeats,
                  optimizer_included=True, validation_training_graphs_co_resident=True,
                  gradient_qualification_bounds=dict(GRADIENT_BOUNDS),
                  timing_contract="Each child qualifies the full update gradient then times; "
                                  "timings are discarded unless every arm passes the learning-relevant gate")
    output = args.output / "benchmark.json"
    atomic_json(output, report)
    try:
        for label, variant, policy in arms:
            directory = args.output / label
            directory.mkdir()
            with (directory / "stdout.log").open("w") as log:
                print(f"Running {label} ({variant}, {policy} policy)", flush=True)
                command = [sys.executable, str(Path(__file__).resolve()), "--bracket", args.bracket, "--child", label,
                           "--output", str(directory.resolve()), "--repeats", str(args.repeats)]
                if args.profile is not None:
                    command += ["--profile", str(args.profile)]
                result = subprocess.run(command, cwd=ROOT, env=arm_environment(policy, args.profile),
                                        stdout=log, stderr=subprocess.STDOUT)
            if (directory / "child.json").exists():
                report["cases"][label] = json.loads((directory / "child.json").read_text())
            atomic_json(output, report)
            if result.returncode or report["cases"].get(label, {}).get("status") != "completed":
                raise RuntimeError(f"{label} failed: exit {result.returncode}; see {directory}/stdout.log")
            if any(sha256(ROOT / p) != value for p, value in hashes.items()):
                raise RuntimeError("Benchmark sources changed")
            if pinned and {p.name: sha256(p) for p in args.profile.iterdir() if p.is_file()} != profile_hashes:
                raise RuntimeError("Pinned profile changed")
        # The parent only reads CPU artifacts: no model construction or CUDA execution.
        import torch
        reference = torch.load(args.output / "reference_before/gradients.pt", map_location="cpu", weights_only=True)
        qualifications = {}
        for label in (*candidates(arms), "reference_after"):
            actual = torch.load(args.output / label / "gradients.pt", map_location="cpu", weights_only=True)
            qualifications[label] = compare_gradients(actual, reference)
            print(json.dumps(dict(qualification=label, **qualification_summary(qualifications[label]))), flush=True)
        report["gradient_qualification"] = qualifications
        atomic_json(output, report)
        if not all(q["passed"] for q in qualifications.values()):
            raise RuntimeError("Learning-relevant gradient parity failed; timing evidence is unqualified")
        cases = report["cases"]
        for label in (*candidates(arms), "reference_after"):
            for key in ("installed_fla", "gpu", "torch", "float32_matmul_precision", "cuda_matmul_allow_tf32"):
                if cases[label][key] != cases["reference_before"][key]:
                    raise RuntimeError(f"Child provenance differs: {key}")
        controls = [cases[label]["measurement"] for label in CONTROLS]
        fastest_control = min(c["median_update_seconds"] for c in controls)
        fastest_control_sample = min(t for c in controls for t in c["update_seconds"])
        comparisons = {}
        custom_ops = cases["custom_ops"]["measurement"]
        for label in candidates(arms):
            measured = cases[label]["measurement"]
            speedup = fastest_control / measured["median_update_seconds"]
            separated = max(measured["update_seconds"]) < fastest_control_sample
            comparisons[label] = dict(speedup_vs_faster_control=speedup, samples_faster_than_both_controls=separated,
                                      improvement_gate_passed=speedup >= 1.05 and separated,
                                      tokens_per_second=measured["tokens_per_second"],
                                      peak_allocated_mib=measured["peak_allocated_mib"])
            if label != "custom_ops":
                # Kernel-option arms improve on the custom-operator arm they extend, measured in the same session.
                comparisons[label].update(
                    speedup_vs_custom_ops=custom_ops["median_update_seconds"] / measured["median_update_seconds"],
                    samples_faster_than_custom_ops=max(measured["update_seconds"]) < min(custom_ops["update_seconds"]))
            print(json.dumps(dict(case=label, **comparisons[label])), flush=True)
        report.update(status="completed", qualified=True, comparisons=comparisons,
                      control_tokens_per_second=[c["tokens_per_second"] for c in controls],
                      reference_drift_ratio=max(c["median_update_seconds"] for c in controls) / fastest_control)
    except BaseException as error:
        report.update(status="failed", qualified=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(output, report)


if __name__ == "__main__":
    main()

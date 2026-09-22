"""Pin production-shape FLA kernel selections for custom-ops GDN2; run via mlq.

Triton autotuning runs once per process and, by default, persists its winners
in a cache keyed without tensor geometry, so earlier small-shape experiments
can silently dictate production kernel tiles. This script tunes every FLA
kernel fresh at the production shape, inside the production CUDA-graph
executor, and writes FLA strict-mode config files that fix those selections.
Runs bind the profile by content digest through the autotuning policy.
"""
from __future__ import annotations

import os

POLICY = dict(FLA_CACHE_RESULTS="0", FLA_CACHE_MODE="disabled", TRITON_CACHE_AUTOTUNING="0")
# Policy must precede every FLA/Triton import; both read it at import time.
os.environ.update(POLICY)
os.environ.pop("FLA_CONFIG_DIR", None)

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import triton  # noqa: E402

from pretraining.nanogpt_mini.gated_delta_runtime import (  # noqa: E402
    fla_autotuner_selections, gdn2_dependency_provenance, pinned_autotune_digest, runtime_autotuning_policy,
    runtime_dependency_versions, unpinnable_selections, write_pinned_autotune_profile)
from scripts.benchmark_gdn2_execution import VARIANTS, execution_sources, measure, sha256  # noqa: E402
from scripts.train_recurrent_slots import atomic_json  # noqa: E402

PROFILES = ROOT / "pretraining/gated_delta/autotune"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help=f"new profile directory name under {PROFILES.relative_to(ROOT)}")
    parser.add_argument("--variant", default="custom_ops", choices=sorted(VARIANTS))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("Use at least five complete optimizer updates after warmup")
    output = PROFILES / args.profile
    if output.exists():
        raise FileExistsError(output)
    policy = runtime_autotuning_policy()
    expected = dict(fla_cache_results=False, fla_cache_mode="disabled", fla_config_dir=None,
                    fla_config_dir_sha256=None, triton_cache_autotuning=False,
                    effective_triton_persistent_results=False)
    if {key: policy[key] for key in expected} != expected:
        raise RuntimeError(f"Fresh in-process autotuning policy not in effect: {policy}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required; submit through mlq")
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    sources = execution_sources(Path(__file__))
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    provenance = gdn2_dependency_provenance()
    torch.manual_seed(1337)
    inputs = torch.randint(1024, (512, 1024), device="cuda", dtype=torch.int32)
    targets = inputs.roll(-1, 1).long()
    print(f"Tuning {args.variant} at production shape", flush=True)
    measurement = measure(args.variant, inputs, targets, args.repeats)
    selections = fla_autotuner_selections()
    if not any(tuner["entries"] for tuner in selections.values()):
        raise RuntimeError("No FLA autotuner selections were populated")
    if gdn2_dependency_provenance() != provenance or any(sha256(ROOT / p) != d for p, d in hashes.items()):
        raise RuntimeError("Sources changed during tuning")
    kernels = write_pinned_autotune_profile(output, selections)
    manifest = dict(
        profile=args.profile, variant=args.variant, kernels=kernels,
        gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), triton=triton.__version__,
        dependency_versions=runtime_dependency_versions(), installed_fla=provenance,
        tuning_policy=policy, model_config=measurement["model_config"],
        microbatch=measurement["microbatch"], seq_len=measurement["seq_len"], batch_tokens=524288,
        executor="co-resident validation and training CUDA graphs, full optimizer updates",
        warmup_optimizer_updates=measurement["warmup_optimizer_updates"],
        measured_optimizer_updates=measurement["measured_optimizer_updates"],
        tokens_per_second=measurement["tokens_per_second"],
        median_update_seconds=measurement["median_update_seconds"],
        update_seconds=measurement["update_seconds"],
        graph_breaks=measurement["graph_breaks"], source_sha256=hashes, selections=selections,
        unpinnable_selections=unpinnable_selections(selections),
        unpinnable_note="plain Triton autotuners retune in-process at production shape under every policy; "
                        "their selections are recorded for comparison but no config file can fix them")
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(dict(profile=str(output.relative_to(ROOT)), kernels=len(kernels),
                          entries=sum(len(t["entries"]) for t in selections.values()),
                          unpinnable_kernels=len(manifest["unpinnable_selections"]),
                          sha256=pinned_autotune_digest(output),
                          median_update_seconds=measurement["median_update_seconds"])), flush=True)


if __name__ == "__main__":
    main()

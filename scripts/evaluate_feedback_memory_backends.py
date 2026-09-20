#!/usr/bin/env python3
"""Compare one canonical LAM checkpoint's p1/p2 BPB under both memory backends.

Submit through mlq with --max-parallel-runs 1. Both isolated workers evaluate
all 1,048,576 canonical validation targets with fullgraph-compiled pass_losses.
No training, eager evaluation, CPU model execution, or reduced validation mode.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import struct
import subprocess
import sys
import tempfile
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
TRAINER = "pretraining/nanogpt_mini/nanogpt_mini_feedback_train.py"
BACKENDS = ("torch_fp32", "fla_compiled")
VAL_TOKENS = 1_048_576
SOURCES = (
    "scripts/evaluate_feedback_memory_backends.py",
    "scripts/profile_minicpm_latent_controller.py",
    TRAINER,
    "pretraining/nanogpt_mini/nanogpt_mini_feedback_model.py",
    "pretraining/nanogpt_mini/nanogpt_mini_model.py",
    "train_gpt.py",
)


def persist(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path.resolve()), sha256=digest, size_bytes=path.stat().st_size)


def repo_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO / path).resolve()


def shard_header(path: Path) -> tuple[int, str]:
    with path.open("rb") as stream:
        header = stream.read(1024)
    if len(header) != 1024:
        raise ValueError(f"Truncated shard header: {path}")
    magic, version, count = struct.unpack_from("<iii", header)
    if magic != 20240520 or version != 1 or count <= 0:
        raise ValueError(f"Invalid shard header: {path}")
    if path.stat().st_size != 1024 + 2 * count:
        raise ValueError(f"Shard payload does not match token count: {path}")
    return count, hashlib.sha256(header).hexdigest()


def describe_run(name: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in (".", ".."):
        raise ValueError("--run must be a canonical ablation name, not a path")
    result_path = REPO / "ablation_results" / name / "result.json"
    result = json.loads(result_path.read_text())
    if result.get("name") != name or result.get("script") != TRAINER:
        raise ValueError("Run name/trainer does not identify a canonical feedback run")
    if result.get("script_args", []) not in ([], ["1"]):
        raise ValueError("Only single-trial runs have an unambiguous final checkpoint")
    if (result.get("returncode") != 0 or result.get("metric_integrity_errors")
            or result.get("early_stop") is not None or result.get("stop_at") is not None
            or result.get("completed_steps") != result.get("steps")):
        raise ValueError("Run must have completed its requested schedule and final validation")
    overrides = result["overrides"]
    required = ("DATA_PATH", "TOKENIZER_PATH", "SEQ_LEN", "MBS", "VAL_TOKENS", "VOCAB_SIZE", "SEED")
    missing = [key for key in required if key not in overrides]
    if missing:
        raise ValueError(f"Missing recorded overrides; refusing to guess: {missing}")
    settings = {key: int(overrides[key]) for key in required[2:]}
    if settings["VAL_TOKENS"] != VAL_TOKENS:
        raise ValueError(f"Require the full canonical VAL_TOKENS={VAL_TOKENS}, not a substitute")
    if min(settings[key] for key in ("SEQ_LEN", "MBS", "VOCAB_SIZE")) <= 0:
        raise ValueError("SEQ_LEN, MBS and VOCAB_SIZE must be positive")
    if VAL_TOKENS % (settings["SEQ_LEN"] * settings["MBS"]):
        raise ValueError("Canonical validation must contain complete recorded microbatches")
    if overrides.get("FB_MODE") != "lam" or int(overrides.get("FB_PASSES", "2")) != 2:
        raise ValueError("This comparison requires a two-pass LAM training run")
    finals = [entry for entry in result["val_entries"]
              if entry.get("type") == "val" and entry.get("step") == result["completed_steps"]]
    if len(finals) != 1:
        raise ValueError("Require exactly one recorded final validation")
    for key in ("val_bpb_p1", "val_bpb_p2"):
        value = finals[0].get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Missing/nonfinite recorded final {key}")

    # The historical trainer divides globally reduced loss by a local byte count.
    # Do not silently compare that number to a corrected multi-GPU metric.
    log_path = REPO / "logs" / f"{name}.txt"
    banners = re.findall(r"^Running PyTorch .+ with world_size (\d+)\s*$", log_path.read_text(), re.MULTILINE)
    if not banners or set(banners) != {"1"}:
        raise ValueError("Require recorded world_size=1 to reproduce historical byte accounting exactly")
    dataset = repo_path(overrides["DATA_PATH"])
    files = sorted(dataset.glob("fineweb_val_*.bin"))
    if not files:
        raise FileNotFoundError(f"No validation shards in {dataset}")
    first_count, _ = shard_header(files[0])
    # Exactly the generator's first next(): it advances ONCE when >= holds,
    # including equality, and never concatenates shards to fill a window.
    index = (1 % len(files)) if VAL_TOKENS + 1 >= first_count else 0
    selected = files[index]
    count, header_sha = shard_header(selected)
    if count < VAL_TOKENS + 1:
        raise ValueError("Trainer's first validation shard window is too short for full val1M")
    artifacts = dict(
        result=fingerprint(result_path),
        checkpoint=fingerprint(REPO / "logs" / f"{name}_final_model.pt"),
        tokenizer=fingerprint(repo_path(overrides["TOKENIZER_PATH"])),
        training_log=fingerprint(log_path),
        validation_shard=fingerprint(selected),
    )
    return dict(
        run=name, overrides=overrides, settings=settings,
        training=dict(steps=result["steps"], completed_steps=result["completed_steps"],
                      val_every=result["val_every"], final_validation=finals[0], world_size=1,
                      memory_kernel=overrides.get("FB_MEMORY_KERNEL"),
                      memory_compiled=overrides.get("FB_MEMORY_COMPILED")),
        artifacts=artifacts, source_hashes={path: fingerprint(REPO / path) for path in SOURCES},
        validation=dict(dataset=str(dataset), shard_index=index, shard_tokens=count,
                        shard_header_sha256=header_sha, first_shard_tokens=first_count,
                        shard_order=[str(path) for path in files], token_offset=0,
                        tokens=VAL_TOKENS, source_tokens=VAL_TOKENS + 1,
                        sequence_length=settings["SEQ_LEN"], microbatch=settings["MBS"],
                        microbatches=VAL_TOKENS // (settings["SEQ_LEN"] * settings["MBS"])),
    )


def verify_files(request: dict) -> None:
    for expected in (*request["artifacts"].values(), *request["source_hashes"].values()):
        if fingerprint(Path(expected["path"])) != expected:
            raise RuntimeError(f"Pinned input/source changed: {expected['path']}")


def validation_window(request: dict, torch):
    import numpy as np
    import sentencepiece as spm

    settings = request["settings"]
    path = Path(request["artifacts"]["validation_shard"]["path"])
    raw = np.fromfile(path, dtype="<u2", count=VAL_TOKENS + 1, offset=1024)
    if raw.size != VAL_TOKENS + 1 or int(raw.max()) >= settings["VOCAB_SIZE"]:
        raise ValueError("Validation token window is truncated or outside model vocabulary")
    inputs = torch.from_numpy(raw[:-1].astype(np.int32)).reshape(-1, settings["SEQ_LEN"]).cuda()
    targets = torch.from_numpy(raw[1:].astype(np.int64)).reshape(-1, settings["SEQ_LEN"]).cuda()
    sp = spm.SentencePieceProcessor(model_file=request["artifacts"]["tokenizer"]["path"])
    if sp.vocab_size() != settings["VOCAB_SIZE"]:
        raise ValueError("Recorded tokenizer and model vocabulary sizes differ")

    from train_gpt import build_sentencepiece_luts

    base, leading, boundary = build_sentencepiece_luts(
        sp, vocab_size=settings["VOCAB_SIZE"], device=torch.device("cuda")
    )
    previous, target = inputs.reshape(-1).long(), targets.reshape(-1)
    byte_counts = base[target].long()
    byte_counts += (leading[target] & ~boundary[previous]).long()
    byte_count = int(byte_counts.sum().item())
    if byte_count <= 0:
        raise ValueError("Validation byte count must be positive")
    domain = dict(request["validation"], bytes=byte_count,
                  window_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                  window_encoding="little-endian uint16: input[0:N] and target[1:N+1]",
                  byte_accounting="train_gpt.build_sentencepiece_luts + feedback_train previous-token boundary rule")
    return inputs, targets, domain


def package_versions() -> dict:
    versions = {}
    for name in ("torch", "triton", "fla-core", "flash-linear-attention", "sentencepiece", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def worker(args, report: dict) -> None:
    request = json.loads(args.manifest.read_text())
    if request["run"] != args.run:
        raise ValueError("Worker manifest/run mismatch")
    verify_files(request)
    os.environ["FB_MEMORY_KERNEL"] = "0" if args.backend == "torch_fp32" else "1"
    os.environ["FB_MEMORY_COMPILED"] = "1"
    for name in ("TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE"):
        if os.environ.get(name, "0").lower() not in ("", "0", "false"):
            raise RuntimeError(f"{name} disables the required compiled execution")
    import torch
    from pretraining.nanogpt_mini import nanogpt_mini_feedback_model as feedback

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with bf16 support is required; no CPU/fp32 model fallback")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")  # Trainer default; fp32 memory reference.
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    if torch._dynamo.config.disable:
        raise RuntimeError("Dynamo is disabled; refusing eager evaluation")
    torch.manual_seed(request["settings"]["SEED"])
    if args.backend == "fla_compiled" and feedback.chunk_simple_gla is None:
        raise RuntimeError("FLA is required; no reference fallback")
    payload = torch.load(request["artifacts"]["checkpoint"]["path"], map_location="cpu", weights_only=True)
    if payload.get("architecture") != "nanogpt_mini_feedback_v1":
        raise ValueError("Checkpoint architecture is not nanogpt_mini_feedback_v1")
    config = dict(payload["model_config"])
    required_config = {"vocab_size", "num_layers", "model_dim", "mode", "detach", "noise",
                       "mlp_hidden", "memory_head_dim", "decays", "attn_window"}
    if required_config - config.keys():
        raise ValueError(f"Incomplete checkpoint model_config: {sorted(required_config - config.keys())}")
    if config["mode"] != "lam" or config["vocab_size"] != request["settings"]["VOCAB_SIZE"]:
        raise ValueError("Checkpoint mode/vocabulary disagrees with the recorded run")
    if payload.get("train_seq_len") != request["settings"]["SEQ_LEN"]:
        raise ValueError("Checkpoint training context differs from recorded SEQ_LEN")
    config["decays"] = tuple(config["decays"])
    if config.get("memory_layers") is not None:
        config["memory_layers"] = tuple(config["memory_layers"])
    for override, field, convert in (("FB_DETACH", "detach", lambda value: bool(int(value))),
                                     ("FB_NOISE", "noise", float), ("FB_WINDOW", "attn_window", int)):
        if override in request["overrides"] and convert(request["overrides"][override]) != config[field]:
            raise ValueError(f"Checkpoint {field} differs from recorded {override}")
    if "FB_MEMORY_LAYERS" in request["overrides"]:
        layer_spec = request["overrides"]["FB_MEMORY_LAYERS"]
        layers = None if layer_spec == "all" else tuple(int(part) for part in layer_spec.split(","))
        if layers != config.get("memory_layers"):
            raise ValueError("Checkpoint insertion sites differ from recorded FB_MEMORY_LAYERS")
    model = feedback.FeedbackGPT(**config).cuda().eval()
    state = payload["model"]
    # load_state_dict(strict=True) checks names/shapes, but would silently cast
    # dtypes. Refuse that too: the comparison must retain the checkpoint weights.
    for name, value in model.state_dict().items():
        if name in state and state[name].dtype != value.dtype:
            raise ValueError(f"Checkpoint dtype mismatch for {name}: {state[name].dtype} != {value.dtype}")
    model.load_state_dict(state, strict=True)
    del payload, state
    inputs, targets, domain = validation_window(request, torch)
    compile_settings = dict(backend="inductor", fullgraph=True, dynamic=False, mode="default")
    evaluate = torch.compile(model.pass_losses, **compile_settings)
    report.update(
        model_config=config, validation=domain, compile=compile_settings,
        checkpoint=request["artifacts"]["checkpoint"], source_hashes=request["source_hashes"],
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                         packages=package_versions(), cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                         matmul_precision=torch.get_float32_matmul_precision(),
                         matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                         cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                         deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                         seed=request["settings"]["SEED"],
                         environment={key: value for key, value in os.environ.items()
                                      if key.startswith(("CUDA_", "TORCH", "TRITON_", "NVIDIA_TF32", "CUBLAS_", "FB_MEMORY_"))}),
        memory_arithmetic=("Torch chunked fp32 operands/state" if args.backend == "torch_fp32"
                           else "FLA bf16 operands/mixed-precision state; fp32 normalizer"),
    )
    persist(args.output, report)
    torch.cuda.synchronize()
    started = time.perf_counter()
    sums = torch.zeros(2, device="cuda")
    mbs = request["settings"]["MBS"]
    # Same sum-CE and fp32 batch-order accumulation as the trainer; no autocast.
    # Passes 3..8 cannot feed back into p1/p2 and are not needed for this question.
    with torch.no_grad():
        for offset in range(0, len(inputs), mbs):
            losses = evaluate(inputs[offset:offset + mbs], targets[offset:offset + mbs], 2)
            sums += torch.stack(losses)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    graphs = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    if graphs < 1:
        raise RuntimeError("No compiled graph observed; refusing an uncompiled result")
    if not torch.isfinite(sums).all().item():
        raise RuntimeError("Nonfinite validation loss")
    losses = (sums / VAL_TOKENS).tolist()
    bpb = [(loss / math.log(2.0)) * (VAL_TOKENS / domain["bytes"]) for loss in losses]
    historical = request["training"]["final_validation"]["val_bpb_p2"]
    report.update(
        status="passed", pass1_bpb=bpb[0], pass2_bpb=bpb[1], pass2_minus_pass1_bpb=bpb[1] - bpb[0],
        pass_mean_nats=losses, pass_sum_nats=sums.tolist(), compiled_graphs=graphs,
        evaluation_seconds_including_compilation=elapsed,
        recorded_final_p2_agreement=dict(recorded_bpb=historical, delta_bpb=bpb[1] - historical,
                                        recorded_4dp=f"{historical:.4f}", evaluated_4dp=f"{bpb[1]:.4f}",
                                        matches_at_recorded_precision=f"{historical:.4f}" == f"{bpb[1]:.4f}"),
    )
    verify_files(request)


def driver(args, report: dict) -> None:
    request = describe_run(args.run)
    report.update(request)
    report["backends"] = {}
    report["limitations"] = [
        "Same-weights inference comparison, not evidence that either training backend is superior.",
        "Both pass_losses paths are fullgraph compiled; historical eager validation may round differently.",
        "Fixed validation inputs/bytes and seed; no claim of cross-device bitwise determinism.",
        "Only p1/p2 are evaluated; later Jacobi passes cannot influence these passes.",
        "Elapsed evaluation time includes compilation; this is not a steady-state training speed benchmark.",
        "Current source hashes and original training-log hash are recorded; historical code may differ.",
    ]
    persist(args.output, report)
    with tempfile.TemporaryDirectory(prefix="feedback-memory-eval-") as directory:
        manifest = Path(directory) / "manifest.json"
        persist(manifest, request)
        for backend in BACKENDS:
            output = Path(directory) / f"{backend}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--run", args.run,
                       "--output", str(output), "--backend", backend, "--manifest", str(manifest)]
            print(f"Evaluating {args.run}: {backend}, full val1M", flush=True)
            # No setsid/start_new_session: mlq retains the entire foreground group.
            child = subprocess.run(command, cwd=REPO, check=False)
            entry = (json.loads(output.read_text()) if output.exists()
                     else dict(status="failed", error="Worker exited without a JSON result"))
            entry["exit_code"] = child.returncode
            if child.returncode:
                entry["status"] = "failed"
            report["backends"][backend] = entry
            persist(args.output, report)
    if any(entry["status"] != "passed" for entry in report["backends"].values()):
        raise RuntimeError("A backend failed; incomplete results are not a BPB comparison")
    reference, fused = (report["backends"][backend] for backend in BACKENDS)
    for key in ("validation", "checkpoint", "source_hashes", "model_config"):
        if reference[key] != fused[key]:
            raise RuntimeError(f"Backend {key} identities differ; refusing comparison")
    report["comparison"] = dict(
        same_weights=True, identical_validation_domain=True,
        delta_convention="fla_compiled minus torch_fp32",
        pass1_delta_bpb=fused["pass1_bpb"] - reference["pass1_bpb"],
        pass2_delta_bpb=fused["pass2_bpb"] - reference["pass2_bpb"],
    )
    report["status"] = "passed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path (never overwritten)")
    parser.add_argument("--backend", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    # Avoid overwriting canonical result.json, a checkpoint, or a previous report,
    # including when error reporting itself would otherwise overwrite that file.
    if args.output.exists():
        print(json.dumps(dict(status="failed", error=f"Output already exists: {args.output}")), file=sys.stderr)
        return 1
    report = dict(schema="feedback_memory_backend_evaluation_v1", status="running", run=args.run,
                  backend=args.backend, started_unix=time.time())

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    exit_code = 0
    try:
        sys.path.insert(0, str(REPO))
        # This helper module imports only stdlib; no trainer import or GPU work.
        from scripts.profile_minicpm_latent_controller import require_mlq_runner
        report["queue"] = require_mlq_runner()
        if bool(args.backend) != bool(args.manifest):
            raise ValueError("Internal --backend and --manifest must be supplied together")
        if args.backend:
            worker(args, report)
        else:
            driver(args, report)
    except BaseException as error:
        exit_code = ((error.code or 1) if isinstance(error, SystemExit) and isinstance(error.code, int)
                     else 130 if isinstance(error, KeyboardInterrupt) else 1)
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    report["finished_unix"] = time.time()
    persist(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "run", "comparison", "error") if key in report}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

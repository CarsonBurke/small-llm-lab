"""Hard-budget runner for the exact 12-bank shared-state KDA experiment.

The queue enforces the outer 600-second wall limit. This in-process runner
reserves 30 seconds for saving the last complete optimizer state. It does
not count partial gradient accumulation as an optimizer update.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    args = parser.parse_args()
    if not 0 < args.steps <= 1000 or args.seconds != 600:
        parser.error(
            "this experiment requires 1..1000 updates and exactly 600 wall seconds"
        )
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be one directory component")
    output = ROOT / "ablation_results" / args.name
    for existing in (
        output,
        ROOT / "logs" / f"{args.name}.txt",
        ROOT / "tb_logs" / args.name,
    ):
        if existing.exists():
            parser.error(f"refusing to reuse experiment namespace: {existing}")
    output.mkdir(exist_ok=False)
    reference_path = ROOT / "ablation_results/k3mix_v5_kda8_ref_2k/result.json"
    reference = json.loads(reference_path.read_text())
    environment = dict(reference["overrides"])
    environment.update(
        {
            "ITERATIONS": str(args.steps),
            "VAL_LOSS_EVERY": "20",
            "TRAIN_LOG_EVERY": "1",
            "RUN_ID": args.name,
            "STATE_ROUTING_DEADLINE": str(STARTED + args.seconds - 30),
            # Retain the full reference validation panel and cadence. Starting
            # with its 33M-token evaluation would consume this experiment before
            # training begins; untrained step-0 quality is already known.
            "SKIP_INITIAL_VALIDATION": "1",
            "GLOBAL_BATCH_TOKENS": "524288",
            "SEQ_LEN": "1024",
            "VAL_TOKENS": str(64 * 524288),
            "DOMAIN_VAL_TOKENS": "1048576",
            "SEED": "1337",
            "MASTER_PORT": "29649",
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MOE_NUM_EXPERTS": "0",
            "MOE_LAYER_INDICES": "",
            "MOE_QB_INTERVAL": "0",
            "MTP_NUM_HEADS": "0",
            "NEXTLAT": "0",
            "NOPE": "0",
            "DENSE_ATTENTION_TYPE": "mha",
            "KDA_NUM_HEADS": "3",
            "KDA_FULL_RANK_GATE": "0",
            "PER_HEAD_MUON": "0",
        }
    )
    os.environ.update(environment)
    os.chdir(ROOT)
    trainer = ROOT / "pretraining/state_routing/train.py"
    sources = [
        Path(__file__),
        trainer,
        ROOT / "pretraining/state_routing/model.py",
        ROOT / "pretraining/tests/test_kda_state_routing_gpu.py",
    ]
    hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        destination = output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes()
        destination.write_bytes(content)
        hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    result = {
        "name": args.name,
        "reference": reference["name"],
        "steps": args.steps,
        "completed_steps": 0,
        "wall_limit_seconds": args.seconds,
        "state_banks": 12,
        "heads_per_bank": 3,
        "top_k": 1,
        "visibility": "exact_token_major",
        "dense_attention_changed": False,
        "balance_weight": 0.01,
        "router_lr": 0.001,
        "router_weighting": "selected_softmax_probability_without_renormalization",
        "overrides": environment,
        "source_sha256": hashes,
        "reference_result_sha256": hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest(),
        "status": "starting",
        "final_val_bpb": None,
        "quality_conclusion": "No matched validation comparison yet.",
    }
    manifest = ROOT / environment["DATA_PATH"] / "mix_manifest.json"
    result["dataset_manifest_sha256"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    result_path = output / "result.json"

    def save_result():
        result["elapsed_seconds"] = time.monotonic() - STARTED
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(result_path)

    save_result()
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from scripts.ablation import parse_log_line
    from pretraining.state_routing.model import RunDeadlineReached

    writer = SummaryWriter(str(ROOT / "tb_logs" / args.name), flush_secs=10)
    stream = (output / "metrics.jsonl").open("a")
    module_name = "pretraining.state_routing._active_trainer"
    spec = importlib.util.spec_from_file_location(module_name, trainer)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load trainer from {trainer}")
    training_module = importlib.util.module_from_spec(spec)
    # Dynamo resolves module globals when constructing guards. A detached
    # exec dictionary named __main__ points its guards at this runner instead.
    sys.modules[module_name] = training_module
    scope = training_module.__dict__
    microbatches = 0

    def record_microbatch(step, index, loss, counts):
        nonlocal microbatches
        if not math.isfinite(loss) or not bool(counts.isfinite().all()):
            raise FloatingPointError("nonfinite microbatch loss or routing counts")
        microbatches += 1
        entry = {
            "type": "microbatch",
            "completed_steps": step,
            "microbatch_index": index,
            "loss_including_balance": loss,
            "route_counts": counts.tolist(),
            "elapsed_seconds": time.monotonic() - STARTED,
        }
        stream.write(json.dumps(entry, allow_nan=False) + "\n")
        stream.flush()
        writer.add_scalar(
            "diagnostic/microbatch_loss_including_balance", loss, microbatches
        )
        writer.add_scalar("perf/wall_seconds", entry["elapsed_seconds"], microbatches)
        result.update(
            status="training",
            completed_microbatches=microbatches,
            completed_steps=scope.get("completed_updates", 0),
        )
        save_result()
        print(
            f"microbatch {microbatches}: update={step} index={index} loss={loss:.5f} wall={entry['elapsed_seconds']:.1f}s",
            flush=True,
        )

    scope["record_microbatch"] = record_microbatch
    sys.argv = [str(trainer)]
    failure = None
    try:
        spec.loader.exec_module(training_module)
        result["status"] = "completed"
    except RunDeadlineReached:
        result["status"] = "time_limit"
    except BaseException as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        failure = exc
    finally:
        result["completed_steps"] = scope.get("completed_updates", 0)
        model = scope.get("model")
        if model is not None and not scope.get("optimizer_update_in_progress", False):
            torch.cuda.synchronize()
            result["parameter_count"] = sum(p.numel() for p in model.parameters())
            result["peak_vram_allocated_mib"] = (
                torch.cuda.max_memory_allocated() / 2**20
            )
            checkpoint_path = output / "last_complete_update.pt"
            torch.save(
                {
                    "architecture": "exact_shared_state_kda",
                    "model_config": {
                        "vocab_size": 50304,
                        "num_layers": 8,
                        "model_dim": 512,
                        "mlp_hidden": 2070,
                        "delta_num_heads": 3,
                        "delta_layer_indices": [0, 1, 2, 4, 5, 6],
                    },
                    "routing_config": {
                        "banks": 12,
                        "checkpoint_tokens": 16,
                        "balance_weight": 0.01,
                    },
                    "model": model.state_dict(),
                    "completed_steps": result["completed_steps"],
                    "optimizer_states": [
                        opt.state_dict() for opt in scope.get("optimizers", [])
                    ],
                    "experiment_config": result,
                },
                checkpoint_path,
            )
            result["checkpoint"] = str(checkpoint_path.relative_to(ROOT))
        log = ROOT / "logs" / f"{args.name}.txt"
        validations = []
        if log.exists():
            for line in log.read_text().splitlines():
                if not line.startswith("step:"):
                    continue
                entry = parse_log_line(line)
                if entry:
                    if entry.get("type") == "metric_error":
                        result["status"] = "error"
                        result["error"] = f"Invalid trainer metric: {entry}"
                        failure = ValueError(result["error"])
                        continue
                    stream.write(json.dumps(entry, allow_nan=False) + "\n")
                    if entry.get("type") == "val":
                        validations.append(entry)
                        writer.add_scalar("val/bpb", entry["val_bpb"], entry["step"])
                    elif entry.get("type") == "train":
                        writer.add_scalar(
                            "train/loss_including_balance",
                            entry["train_loss"],
                            entry["step"],
                        )
        result["val_entries"] = validations
        if validations:
            result["final_val_bpb"] = validations[-1]["val_bpb"]
        else:
            result["quality_conclusion"] = (
                "Insufficient updates/evaluation within the hard time budget; no BPB claim."
            )
        save_result()
        stream.close()
        writer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    print(json.dumps(result, indent=2), flush=True)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()

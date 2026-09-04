"""Validate a MiniCPM VAPO continuation before reserving a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


POLICY_SCHEMA = "minicpm5_vapo_adapter/v6"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _required_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {label} must be a mapping")
    return value


def _require_fields(
    mapping: Mapping[str, Any], fields: tuple[str, ...], label: str
) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"checkpoint {label} is missing: {', '.join(missing)}")


def _saved_trainer_options(saved_args: Mapping[str, Any]) -> list[str]:
    from postraining.train_minicpm_vapo import (
        RESUME_MUTABLE_OPTIONS,
        build_parser as build_trainer_parser,
    )

    parser = build_trainer_parser()
    defaults = vars(parser.parse_args([]))
    operational = {"help", "resume", "output", "data", "steps"}
    options: list[str] = []
    for action in parser._actions:
        if action.dest in operational:
            continue
        if action.dest in RESUME_MUTABLE_OPTIONS:
            value = defaults[action.dest]
        elif action.dest in saved_args:
            value = saved_args[action.dest]
        else:
            continue
        long_options = [
            option
            for option in action.option_strings
            if option.startswith("--") and not option.startswith("--no-")
        ]
        if not long_options:
            continue
        if action.nargs == 0:
            if isinstance(action, argparse.BooleanOptionalAction):
                option = long_options[0] if value else f"--no-{long_options[0][2:]}"
                options.append(option)
            elif value:
                options.append(long_options[0])
            continue
        if value is not None:
            options.extend((long_options[0], str(value)))
    return options


def inspect_continuation(
    checkpoint: Mapping[str, Any],
    *,
    resume_path: Path,
    output_path: Path,
    data_path: Path,
    target_steps: int | None,
    additional_steps: int | None,
    job_name: str | None,
    time_limit: str,
) -> dict[str, Any]:
    _require_fields(
        checkpoint,
        (
            "policy",
            "actor_optimizer",
            "critic_optimizer",
            "args",
            "step",
            "cursor",
            "warmup_step",
            "pending_records",
            "pending_epoch",
            "cpu_rng",
            "cuda_rng",
            "python_rng",
            "data_sha256",
        ),
        "root",
    )
    policy = _required_mapping(checkpoint["policy"], "policy")
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"resume policy must use {POLICY_SCHEMA}")
    _require_fields(policy, ("actor", "critic"), "policy")
    actor = _required_mapping(policy["actor"], "actor")
    critic = _required_mapping(policy["critic"], "critic")
    saved_args = _required_mapping(checkpoint["args"], "args")
    side_fields = (
        "model_id",
        "revision",
        "lora_config",
        "lora_modules",
        "nextlat_projection_factor",
        "adapter",
        "nextlat",
    )
    _require_fields(actor, side_fields, "actor")
    _require_fields(critic, (*side_fields, "value_head"), "critic")
    expected_lora_config = {
        "rank": saved_args.get("lora_rank"),
        "alpha": saved_args.get("lora_alpha"),
        "targets": tuple(saved_args.get("lora_targets", ())),
    }
    if not expected_lora_config["targets"]:
        from postraining.minicpm_vapo import DEFAULT_LORA_TARGETS

        expected_lora_config["targets"] = tuple(DEFAULT_LORA_TARGETS)
    for side, label in ((actor, "actor"), (critic, "critic")):
        lora_config = _required_mapping(side["lora_config"], f"{label} LoRA config")
        if dict(lora_config) != expected_lora_config:
            raise ValueError(f"checkpoint {label} LoRA config differs from saved args")
        for field in ("adapter", "nextlat"):
            state = _required_mapping(side[field], f"{label} {field}")
            if not state:
                raise ValueError(f"checkpoint {label} {field} is empty")
        if not isinstance(side["lora_modules"], (list, tuple)) or not side[
            "lora_modules"
        ]:
            raise ValueError(
                f"checkpoint {label} LoRA modules must be a nonempty sequence"
            )
        if side["model_id"] != saved_args.get("model"):
            raise ValueError(f"checkpoint {label} model id differs from saved args")
        if side["revision"] != saved_args.get("revision"):
            raise ValueError(f"checkpoint {label} revision differs from saved args")
        if side["nextlat_projection_factor"] != saved_args.get(
            "nextlat_projection_factor"
        ):
            raise ValueError(
                f"checkpoint {label} NextLat factor differs from saved args"
            )
    value_head = _required_mapping(critic["value_head"], "critic value head")
    if not value_head:
        raise ValueError("checkpoint critic value head is empty")
    if tuple(actor["lora_modules"]) != tuple(critic["lora_modules"]):
        raise ValueError("actor and critic LoRA module sets differ")
    for field in ("actor_optimizer", "critic_optimizer"):
        optimizer = _required_mapping(checkpoint[field], field.replace("_", " "))
        _require_fields(optimizer, ("state", "param_groups"), field.replace("_", " "))
        _required_mapping(optimizer["state"], f"{field} state")
        if not isinstance(optimizer["param_groups"], list):
            raise ValueError(f"checkpoint {field} parameter groups must be a list")
    if not isinstance(checkpoint["cpu_rng"], torch.Tensor) or not isinstance(
        checkpoint["cuda_rng"], torch.Tensor
    ):
        raise ValueError("checkpoint CPU and CUDA RNG states must be tensors")
    if not isinstance(checkpoint["python_rng"], tuple):
        raise ValueError("checkpoint Python RNG state must be a tuple")
    try:
        random.Random().setstate(checkpoint["python_rng"])
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint Python RNG state is invalid") from error
    if actor["model_id"] != critic["model_id"]:
        raise ValueError("actor and critic model ids differ")
    if actor["revision"] != critic["revision"]:
        raise ValueError("actor and critic revisions differ")

    step = int(checkpoint["step"])
    warmup_step = int(checkpoint["warmup_step"])
    cursor = int(checkpoint["cursor"])
    pending_epoch = int(checkpoint["pending_epoch"])
    if min(step, warmup_step, cursor, pending_epoch) < 0:
        raise ValueError("resume progress counters must be nonnegative")
    value_warmup_steps = int(saved_args.get("value_warmup_steps", -1))
    ppo_epochs = int(saved_args.get("ppo_epochs", -1))
    if value_warmup_steps < 0 or warmup_step > value_warmup_steps:
        raise ValueError("resume warmup progress is invalid")
    if ppo_epochs < 1:
        raise ValueError("checkpoint PPO epoch count is missing or invalid")
    pending_records = checkpoint["pending_records"]
    if pending_records is not None and not isinstance(pending_records, list):
        raise ValueError("checkpoint pending records must be a list or null")
    if pending_records is None and pending_epoch != 0:
        raise ValueError("resume has a pending epoch without replay records")
    if pending_records == []:
        raise ValueError("checkpoint pending records cannot be empty")
    if pending_records is not None and not 0 <= pending_epoch < ppo_epochs:
        raise ValueError("resume pending PPO epoch is invalid")
    pending_record_count = 0 if pending_records is None else len(pending_records)

    if (target_steps is None) == (additional_steps is None):
        raise ValueError("choose exactly one of target_steps or additional_steps")
    if additional_steps is not None:
        if additional_steps < 1:
            raise ValueError("additional_steps must be positive")
        resolved_target = step + additional_steps
    else:
        assert target_steps is not None
        resolved_target = target_steps
    if resolved_target <= step:
        raise ValueError(
            f"target step {resolved_target} must exceed saved actor step {step}"
        )

    expected_data_sha256 = checkpoint.get("data_sha256")
    if not isinstance(expected_data_sha256, str) or len(expected_data_sha256) != 64:
        raise ValueError("checkpoint dataset fingerprint is missing or malformed")
    if not data_path.is_file():
        raise ValueError(f"dataset does not exist: {data_path}")
    actual_data_sha256 = file_sha256(data_path)
    if actual_data_sha256 != expected_data_sha256:
        raise ValueError("dataset bytes differ from the resume checkpoint")

    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"output path is not a directory: {output_path}")
    output_entries = tuple(output_path.iterdir()) if output_path.exists() else ()
    resume_parent = resume_path.parent.resolve()
    if output_entries and output_path.resolve() != resume_parent:
        raise ValueError(
            "refusing to mix a continuation into a nonempty output directory "
            "that does not own the resume checkpoint"
        )
    if output_path.resolve() == resume_parent:
        output_state = "resume_in_place"
    elif output_path.exists():
        output_state = "empty"
    else:
        output_state = "new"

    prompts = int(saved_args.get("prompts_per_rollout", 0))
    samples = int(saved_args.get("samples_per_prompt", 0))
    if prompts < 1 or samples < 1:
        raise ValueError("checkpoint rollout dimensions are missing or invalid")
    requested_name = job_name or output_path.name
    normalized_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", requested_name).strip("-.")
    if not normalized_name:
        raise ValueError("job name is empty after normalization")
    saved_data_argument = saved_args.get("data")
    if not isinstance(saved_data_argument, str) or not saved_data_argument:
        raise ValueError("checkpoint dataset argument is missing or invalid")
    if _resolve_repo_path(saved_data_argument) != data_path.resolve():
        raise ValueError(
            "dataset path must resolve from the checkpoint's saved argument; "
            "the trainer treats that argument as immutable"
        )
    trainer_command = [
        "python3",
        "-m",
        "postraining.train_minicpm_vapo",
        *_saved_trainer_options(saved_args),
        "--resume",
        str(resume_path),
        "--output",
        str(output_path),
        "--data",
        saved_data_argument,
        "--steps",
        str(resolved_target),
    ]
    from postraining.train_minicpm_vapo import (
        build_parser as build_trainer_parser,
        validate_resume_configuration,
    )

    generated_args = build_trainer_parser().parse_args(trainer_command[3:])
    validate_resume_configuration(dict(checkpoint), generated_args)
    queue_command = [
        "mlq",
        "submit",
        "--name",
        normalized_name,
        "--cwd",
        str(REPO_ROOT),
        "--max-parallel-runs",
        "1",
        "--time-limit",
        time_limit,
        "--",
        *trainer_command,
    ]
    return {
        "schema": "minicpm_vapo_preflight/v1",
        "resume": str(resume_path),
        "checkpoint_bytes": resume_path.stat().st_size,
        "policy_schema": POLICY_SCHEMA,
        "model_id": actor.get("model_id"),
        "revision": actor.get("revision"),
        "saved_actor_step": step,
        "target_actor_step": resolved_target,
        "remaining_actor_updates": resolved_target - step,
        "warmup_step": warmup_step,
        "cursor": cursor,
        "pending_epoch": pending_epoch,
        "pending_records": pending_record_count,
        "dataset": str(data_path),
        "data_sha256": actual_data_sha256,
        "output": str(output_path),
        "output_state": output_state,
        "rollout_trajectories": prompts * samples,
        "queue_command": shlex.join(queue_command),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-steps", type=int)
    target.add_argument("--additional-steps", type=int)
    parser.add_argument("--job-name")
    parser.add_argument("--time-limit", default="10h")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resume_path = _resolve_repo_path(args.resume)
    output_path = _resolve_repo_path(args.output)
    if not resume_path.is_file():
        raise ValueError(f"resume checkpoint does not exist: {resume_path}")
    checkpoint = torch.load(
        resume_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("resume checkpoint must be a mapping")
    saved_args = _required_mapping(checkpoint.get("args"), "args")
    configured_data = args.data or saved_args.get("data")
    if not configured_data:
        raise ValueError("dataset path is absent from both CLI and checkpoint")
    report = inspect_continuation(
        checkpoint,
        resume_path=resume_path,
        output_path=output_path,
        data_path=_resolve_repo_path(configured_data),
        target_steps=args.target_steps,
        additional_steps=args.additional_steps,
        job_name=args.job_name,
        time_limit=args.time_limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

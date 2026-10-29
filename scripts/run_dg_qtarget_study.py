#!/usr/bin/env python3
"""Supervise the existing matched DG study without executing models locally.

Normal mode records queue observations and submits exact checkpoint resumes.
--once is a read-only pass: report decisions, without writing or submitting.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time


TERMINAL_ARMS = {"complete", "culled", "failed", "needs_attention"}
LIVE_JOBS = {"queued", "running", "held", "starting", "recovering"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(path: Path, value: dict, *, jsonl: bool = False) -> None:
    atomic_text(path, json.dumps(value, indent=None if jsonl else 2, sort_keys=True) + "\n")


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def mlq(arguments: list[str], *, as_json: bool = True):
    result = subprocess.run(
        ["mlq", *arguments], check=True, capture_output=True, text=True,
        timeout=60,
    )
    return json.loads(result.stdout) if as_json else result.stdout


def resume_argv(initial: list[str], checkpoint: Path) -> list[str]:
    result = []
    index = 0
    while index < len(initial):
        argument = initial[index]
        if argument in {"--actor-init", "--resume"}:
            index += 2
        elif argument == "--fresh-critic":
            index += 1
        else:
            result.append(argument)
            index += 1
    if "--reset-optimizers-on-resume" in result:
        raise ValueError("this study must preserve optimizer state")
    return [*result, "--resume", str(checkpoint)]


def trainer_arguments(argv: list[str]) -> list[str]:
    index = argv.index("-m")
    if argv[index + 1] != "postraining.train_latent_vapo":
        raise ValueError("unexpected trainer module")
    return argv[index + 2:]


def verify_source(study: dict) -> None:
    source = Path(study["source_root"])
    for relative, expected in study["source_sha256"].items():
        if sha256(source / relative) != expected:
            raise ValueError(f"frozen source changed: {relative}")


def frozen_trainer(study: dict):
    source = str(Path(study["source_root"]).resolve())
    sys.path.insert(0, source)
    module = importlib.import_module("postraining.train_latent_vapo")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise ValueError("resume validation imported the live checkout")
    return module


def verify_checkpoint_source(study: dict, arm: dict, payload: dict) -> None:
    provenance = payload.get("source_provenance") or {}
    archive = Path(arm["output"]) / provenance["source_archive"]
    if sha256(archive) != provenance["source_archive_sha256"]:
        raise ValueError("checkpoint source archive digest changed")
    with tarfile.open(archive, "r:gz") as stream:
        for relative, expected in study["source_sha256"].items():
            member = stream.extractfile(relative)
            if member is None:
                raise ValueError(f"checkpoint source archive lacks {relative}")
            with member:
                if hashlib.file_digest(member, "sha256").hexdigest() != expected:
                    raise ValueError(f"checkpoint used different source: {relative}")


def resolve_runtime_budget(module, args, manifest: dict) -> None:
    """Mirror backbone-dependent CLI resolution using recorded metadata."""
    architecture = manifest["base"]["architecture"]
    if not architecture.startswith("nanogpt_mini"):
        raise ValueError("this supervisor requires the study's nano backbone")
    context = module.resolve_rollout_defaults(
        args, is_nano=True,
        checkpoint_context_tokens=manifest["checkpoint_context_limit"],
        trained_context_tokens=manifest["trained_context_tokens"],
    )
    if context != manifest["context_tokens"]:
        raise ValueError("resume changed the recorded runtime context")
    args.resolved_train_max_new_tokens, args.resolved_train_max_stream_steps = module.training_rollout_budget(
        args.reasoning_mode, args.continuation_tokens,
        answer_tokens=args.answer_tokens, prompt_tokens=args.prompt_tokens,
        context_tokens=context,
    )


def checkpoint_metadata(study: dict, arm: dict, checkpoint: Path) -> dict:
    # Trusted local trainer checkpoint; CPU mapping never constructs a model.
    import torch

    module = frozen_trainer(study)
    digest = sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    verify_checkpoint_source(study, arm, payload)
    saved = payload["args"]
    if not isinstance(saved, dict):
        saved = vars(saved)
    parser = module.build_arg_parser()
    args = parser.parse_args(trainer_arguments(resume_argv(arm["argv"], checkpoint)))
    module.validate_args(parser, args)
    manifest = json.loads((Path(arm["output"]) / "manifest.json").read_text())
    resolve_runtime_budget(module, args, manifest)
    module.validate_resume_arg_contract(saved, args)
    if args.steps != study["joint_updates"] or args.value_warmup_steps != study["critic_warmup_updates"]:
        raise ValueError("resume changed the cumulative update budget")
    if Path(saved["output"]).resolve() != Path(arm["output"]).resolve():
        raise ValueError("checkpoint belongs to another output directory")
    if payload["base_checkpoint_sha256"] != study["inputs"]["--checkpoint"]["sha256"]:
        raise ValueError("checkpoint has a different base")
    expected_schemas = {
        "resume_arg_contract_schema": module.RESUME_ARG_CONTRACT_SCHEMA,
        "actor_objective_schema": module.run_actor_objective_schema(args),
        "critic_schema": module.critic_schema_for(True),
        "optimizer_schema": module.optimizer_schema_for_trunk_optimizer(args.trunk_optimizer),
        "execution_schema": module.execution_schema_for(args.reasoning_mode, args.rollout_scheduler),
        "replay_numerics_schema": module.replay_numerics_schema_for(args.reasoning_mode),
    }
    for field, expected in expected_schemas.items():
        if payload.get(field) != expected:
            raise ValueError(f"checkpoint {field} changed")
    for field in ("model", "critic", "optimizers", "cpu_rng", "cuda_rng", "python_rng", "actor_init_provenance"):
        if field not in payload or payload[field] is None:
            raise ValueError(f"checkpoint lacks exact-resume state: {field}")
    expected_optimizers = {"actor", "critic", "critic_action"}
    if args.trunk_optimizer == "muon":
        expected_optimizers |= {"actor_muon", "critic_muon"}
    if set(payload["optimizers"]) != expected_optimizers:
        raise ValueError("checkpoint has an unexpected optimizer layout")
    provenance = payload["actor_init_provenance"]
    if provenance.get("checkpoint_sha256") != study["inputs"]["--actor-init"]["sha256"]:
        raise ValueError("checkpoint has a different actor initialization")
    if "target_sampler_cursor" not in provenance:
        raise ValueError("checkpoint cannot prove its warmup rewind cursor")
    step = int(payload["step"])
    warmup = int(payload["value_warmup_step"])
    cursor = int(payload["sampler_cursor"])
    if not 0 <= step <= study["joint_updates"] or not 0 <= warmup <= study["critic_warmup_updates"] or cursor < 0:
        raise ValueError("checkpoint counters are outside the study budget")
    if step and warmup != study["critic_warmup_updates"]:
        raise ValueError("joint training started before critic warmup completed")
    metadata = {
        "path": str(checkpoint), "sha256": digest, "step": step,
        "value_warmup_step": warmup, "sampler_cursor": cursor,
        "optimizer_names": sorted(payload["optimizers"]),
        "q_critic_lambda": saved.get("q_critic_lambda"),
    }
    del payload
    return metadata


def metrics_summary(output: Path, through_step: int | None = None) -> dict:
    path = output / "metrics.jsonl"
    if not path.exists():
        return {}
    summary = {}
    with path.open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    break  # Writer can be partway through its last record.
                raise
            kind = record.get("type")
            step = record.get("step", 0)
            if through_step is not None and kind != "value_warmup" and step > through_step:
                continue
            if kind in {"train", "bpb", "bench", "aime", "value_warmup"}:
                summary[kind] = {key: record[key] for key in (
                    "step", "val_bpb", "accuracy", "contract_accuracy", "policy_accuracy", "reward_mean", "explained_variance"
                ) if key in record}
    return summary


def time_limit_failure(job: dict) -> bool:
    reasons = [job.get("stateReason", "")]
    reasons.extend(attempt.get("message", "") for attempt in job.get("attempts", [])[-1:])
    return job["state"] == "failed" and any(
        re.search(r"time limit(?: .*?)? exceeded", reason, re.IGNORECASE) for reason in reasons
    )


def native_cull(stdout: str, threshold: int) -> bool:
    return threshold > 0 and bool(re.search(
        rf"{threshold} consecutive constant-reward pools at step \d+/\d+; stopping at the pool boundary",
        stdout,
    ))


def valid_score(value, *, bpb: bool = False) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and (value > 0 if bpb else 0 <= value <= 1)
    )


def decide(study: dict, arm: dict, job: dict, stdout: str, metadata: dict | None) -> tuple[str, str]:
    state = job["state"]
    if state in LIVE_JOBS:
        return "waiting", state
    if state != "succeeded" and not time_limit_failure(job):
        return "failed", f"job {job['id']} {state}: {job.get('stateReason', 'operator review required')}"
    arguments = arm["argv"]
    threshold = int(arguments[arguments.index("--zero-reward-stop-pools") + 1])
    if native_cull(stdout, threshold):
        return "culled", "native constant-reward stop; will not resume"
    if metadata is None:
        return "needs_attention", "terminal session has no valid recovery checkpoint"
    if metadata["step"] == study["joint_updates"]:
        return "training_complete", "cumulative joint and warmup budgets reached; final evaluations required"
    used = [submission.get("resume_checkpoint_sha256") for submission in arm["submissions"]]
    if metadata["sha256"] in used:
        return "needs_attention", "session made no saved progress; will not resubmit the same checkpoint"
    previous = arm["submissions"][-1].get("resume_checkpoint")
    if previous and (metadata["step"], metadata["value_warmup_step"]) <= (previous["step"], previous["value_warmup_step"]):
        return "needs_attention", "session made no optimizer-update progress despite rewriting its checkpoint"
    if state == "succeeded" and not re.search(r"--max-train-hours .* reached at step \d+(?:/\d+)?; stopping at the pool boundary", stdout):
        return "needs_attention", "incomplete successful session has no native wall-clock stop"
    return "resume", "native wall-clock boundary" if state == "succeeded" else "time limit with valid recovery state"


def plan_resume(study_path: Path, study: dict, name: str, arm: dict, metadata: dict) -> dict:
    session = len(arm["submissions"]) + 1
    identity = f"{study_path.resolve()}:{name}:{metadata['sha256']}"
    key = "dg-qtarget-" + hashlib.sha256(identity.encode()).hexdigest()
    return {
        "idempotency_key": key, "resume_checkpoint_sha256": metadata["sha256"],
        "resume_checkpoint": metadata, "session": session,
        "command": [
            "submit", "--json", "--idempotency-key", key,
            "--name", f"{Path(arm['output']).name}_s{session}",
            "--cwd", study["source_root"], "--priority", "1",
            "--max-parallel-runs", "1", "--max-attempts", "1",
            "--time-limit", "30m", "--", *resume_argv(arm["argv"], Path(metadata["path"])),
        ],
    }


def submit_pending(study_path: Path, study: dict, arm: dict) -> None:
    verify_source(study)
    pending = arm["pending_submission"]
    # Pending command is durable BEFORE mutation. Repeating this exact key is
    # safe if mlq accepted the command but the supervisor lost its response.
    result = mlq(pending["command"])
    if not isinstance(result, dict) or "id" not in result:
        raise ValueError("unrecognized mlq submission response; pending key retained")
    result["resume_checkpoint_sha256"] = pending["resume_checkpoint_sha256"]
    result["idempotency_key"] = pending["idempotency_key"]
    if "resume_checkpoint" in pending:
        result["resume_checkpoint"] = pending["resume_checkpoint"]
    arm["submissions"].append(result)
    del arm["pending_submission"]
    arm["supervisor_state"] = "waiting"
    atomic_save(study_path, study)


def plan_evaluation(study_path: Path, study: dict, name: str, arm: dict, kind: str) -> dict:
    metadata = arm["last_observation"]["checkpoint"]
    if sha256(Path(metadata["path"])) != metadata["sha256"]:
        raise ValueError("final evaluation checkpoint changed after training completed")
    output = Path(arm["output"]) / "final_eval" / kind
    argv = resume_argv(arm["argv"], Path(metadata["path"]))
    argv[argv.index("--output") + 1] = str(output)
    if kind != "aime":
        argv += [f"--{kind}-only"]
    if kind == "bench":
        argv += ["--bench-only-repeats", "1"]
    identity = f"{study_path.resolve()}:{name}:final-{kind}:{metadata['sha256']}"
    key = "dg-qtarget-" + hashlib.sha256(identity.encode()).hexdigest()
    return {
        "idempotency_key": key, "resume_checkpoint_sha256": metadata["sha256"],
        "resume_checkpoint": metadata,
        "command": ["submit", "--json", "--idempotency-key", key,
                    "--name", f"{Path(arm['output']).name}_final_{kind}",
                    "--cwd", study["source_root"], "--priority", "1",
                    "--max-parallel-runs", "1", "--max-attempts", "1",
                    "--time-limit", "30m", "--", *argv],
    }


def final_evaluations(study_path: Path, study: dict, name: str, arm: dict, *, apply: bool) -> dict:
    tasks = arm.setdefault("final_evaluations", {})
    kinds = ["bench", "bpb"]
    canonical_aime = metrics_summary(Path(arm["output"])).get("aime", {})
    aime_valid = canonical_aime.get("step") == study["joint_updates"] and all(
        valid_score(canonical_aime.get(field)) for field in ("accuracy", "contract_accuracy")
    )
    if "aime" in tasks or not aime_valid:
        # Full resume at the saved ceiling skips both optimizer loops and
        # executes the frozen trainer's terminal AIME evaluation only.
        kinds.append("aime")
    for kind in kinds:
        task = tasks.setdefault(kind, {"submissions": [], "supervisor_state": "pending"})
        if task["supervisor_state"] == "pending":
            task["pending_submission"] = plan_evaluation(study_path, study, name, arm, kind)
            if apply:
                atomic_save(study_path, study)
        if task.get("pending_submission"):
            if not apply:
                continue
            submit_pending(study_path, study, task)
        if task["supervisor_state"] in TERMINAL_ARMS:
            continue
        job = mlq(["show", str(task["submissions"][-1]["id"]), "--json"])
        if job["args"] != task["submissions"][-1]["args"] or Path(job["cwd"]).resolve() != Path(study["source_root"]).resolve():
            raise ValueError("final evaluation job changed its command or source")
        task["last_job_state"] = job["state"]
        if job["state"] in LIVE_JOBS:
            continue
        if job["state"] != "succeeded":
            task["supervisor_state"] = "failed"
            task["reason"] = f"final {kind} job {job['id']} {job['state']}; operator review required"
            continue  # Evaluation failures, including timeouts, are not blind retried.
        output = Path(arm["output"]) / "final_eval" / kind
        if kind == "bpb":
            stdout = mlq(["logs", str(job["id"])], as_json=False)
            records = []
            for line in stdout.splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("type") == "bpb" and record.get("step") == study["joint_updates"] and valid_score(record.get("val_bpb"), bpb=True):
                    records.append(record)
            if not records:
                task["supervisor_state"] = "needs_attention"
                task["reason"] = "successful final BPB job did not emit its step-950 score"
                continue
            if apply:
                atomic_save(output / "metrics.jsonl", records[-1], jsonl=True)
        else:
            summary = metrics_summary(output)
            score = summary.get(kind, {})
            if score.get("step") != study["joint_updates"] or not all(valid_score(score.get(field)) for field in ("accuracy", "contract_accuracy")):
                task["supervisor_state"] = "needs_attention"
                task["reason"] = f"successful final {kind} job lacks its step-950 metrics"
                continue
        task["supervisor_state"] = "complete"
    states = [task["supervisor_state"] for task in tasks.values()]
    if any(state in {"failed", "needs_attention"} for state in states):
        decision = "needs_attention"
    elif all(state == "complete" for state in states):
        decision = "complete"
    else:
        decision = "training_complete"
    observation = {"arm": name, "decision": decision, "final_evaluations": tasks}
    if apply:
        arm["supervisor_state"] = decision
        atomic_save(study_path, study)
    return observation


def supervise_once(study_path: Path, study: dict, *, apply: bool) -> list[dict]:
    verify_source(study)
    observations = []
    for name, arm in study["arms"].items():
        if arm.get("supervisor_state") in TERMINAL_ARMS:
            observations.append({"arm": name, "decision": arm["supervisor_state"]})
            continue
        if arm.get("supervisor_state") == "training_complete":
            observations.append(final_evaluations(study_path, study, name, arm, apply=apply))
            continue
        if arm.get("pending_submission"):
            if apply:
                submit_pending(study_path, study, arm)
            else:
                observations.append({"arm": name, "decision": "recover_pending", "submission": arm["pending_submission"]})
                continue
        job = mlq(["show", str(arm["submissions"][-1]["id"]), "--json"])
        if Path(job["cwd"]).resolve() != Path(study["source_root"]).resolve():
            raise ValueError(f"{name}: tracked job uses a different source checkout")
        expected = arm["submissions"][-1]["args"]
        if job["args"] != expected:
            raise ValueError(f"{name}: tracked job argv changed")
        metadata = None
        stdout = ""
        if job["state"] == "succeeded" or time_limit_failure(job):
            stdout = mlq(["logs", str(job["id"])], as_json=False)
            checkpoint = Path(arm["output"]) / "latent_vapo_checkpoint.pt"
            if checkpoint.exists():
                metadata = checkpoint_metadata(study, arm, checkpoint)
        decision, reason = decide(study, arm, job, stdout, metadata)
        observation = {"arm": name, "job_id": job["id"], "job_state": job["state"], "decision": decision, "reason": reason}
        if metadata:
            observation["checkpoint"] = metadata
        observation["metrics"] = metrics_summary(Path(arm["output"]), metadata["step"] if metadata else None)
        if decision == "resume":
            observation["submission"] = plan_resume(study_path, study, name, arm, metadata)
        observations.append(observation)
        if apply:
            arm["last_observation"] = observation
            arm["supervisor_state"] = decision
            if decision == "resume":
                arm["pending_submission"] = observation["submission"]
            atomic_save(study_path, study)
            if decision == "resume":
                submit_pending(study_path, study, arm)
    if apply:
        refresh_analysis(study_path)
    return observations


def refresh_analysis(study_path: Path) -> None:
    utility = Path(__file__).with_name("dg_qtarget_study.py")
    spec = importlib.util.spec_from_file_location("dg_qtarget_evidence", utility)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.build_report(study_path)
    atomic_save(study_path.parent / "analysis.json", report)
    atomic_text(study_path.parent / "analysis.txt", module.render(report) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1] / "postraining/runs/dg_qtarget_study_20261012/study.json")
    parser.add_argument("--once", action="store_true", help="inspect one pass without writing or submitting")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be between 1 and 60")
    study_path = args.study.resolve()
    with study_path.with_suffix(".supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another supervisor holds this study lock")
        while True:
            study = json.loads(study_path.read_text())
            if study.get("schema") != "dg_qtarget_matched_ablation/v1":
                raise ValueError("unsupported study schema")
            try:
                observations = supervise_once(study_path, study, apply=not args.once)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                if not args.once:
                    study["supervisor_error"] = {"message": str(error), "time": time.time()}
                    atomic_save(study_path, study)
                print(json.dumps({"supervisor_error": str(error)}), file=sys.stderr, flush=True)
                return 1
            print(json.dumps(observations), flush=True)
            if args.once or all(arm.get("supervisor_state") in TERMINAL_ARMS for arm in study["arms"].values()):
                return 1 if any(row.get("decision") in {"failed", "needs_attention"} for row in observations) else 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end OPSD training over a project-native language-model checkpoint."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters
from postraining.core import (
    answer_style,
    deterministic_math_subset,
    load_posttraining_tokenizer,
    load_unique_math_rows,
    structural_format_ok,
)
from postraining.latent_eval import evaluate_latent_math, verify_terminated_answer
from postraining.latent_rollout import emitted_token_rows, rollout_continuations
from postraining.latent_thought import LatentThoughtModel
from postraining.model_io import load_model
from postraining.math_prompt import (
    ANSWER_FENCE_PROMPT_SCHEMA,
    canonicalize_answer_fence_rows,
)
from postraining.opsd.config import resolved_distillation_temperature
from postraining.opsd.authorize import (
    AUTHORIZATION_GATE_CONTRACT,
    OPSD_AUTHORIZATION_SCHEMA,
)
from postraining.opsd.data import (
    OPSDExample,
    ShuffledExampleSampler,
    SourceQuotaSampler,
    TEACHER_PROMPT_SCHEMA,
    file_sha256,
    load_examples,
    resolve_reference_column,
    tokenize_example,
)
from postraining.opsd.manifest import (
    source_quotas,
    validate_final_answer_manifest,
)
from postraining.opsd.loss import backward_opsd_example
from postraining.opsd.schemas import (
    OPSD_CHECKPOINT_SCHEMA,
    OPSD_DATA_ORDER_SCHEMA,
    OPSD_EXPORT_SCHEMA,
    OPSD_OBJECTIVE_SCHEMA,
    OPSD_OPTIMIZER_SCHEMA,
    OPSD_PROMPT_SCHEMA,
)
from postraining.provenance import capture_source_provenance


PAPER_URL = "https://arxiv.org/abs/2601.18734v3"
PAPER_CODE_COMMIT = "7448751f307a9cdbcc1246dd1565a1a605b443df"

_RESUME_EXACT_FIELDS = (
    "checkpoint",
    "dataset",
    "reference_column",
    "data_manifest",
    "authorization",
    "allow_failed_authorization",
    "effective_batch_size",
    "rollout_batch_size",
    "max_completion_length",
    "max_prompt_length",
    "temperature",
    "top_p",
    "top_k",
    "distillation_temperature",
    "pointwise_kl_clip",
    "logit_chunk_tokens",
    "rollout_compile",
    "learning_rate",
    "max_grad_norm",
    "weight_decay",
    "seed",
    "think_tokens",
    "answer_fence",
)

_RESUME_EVAL_FIELDS = (
    "eval_data",
    "eval_rows",
    "eval_samples",
    "eval_batch_trajectories",
    "eval_temperature",
    "eval_top_p",
    "eval_top_k",
    "eval_think_min_tokens",
)

OPSD_REWARD_MIN_THINK_TOKENS = 33

def validate_data_manifest(args, payload: dict) -> dict[str, object] | None:
    """Bind final-answer arms to one immutable, decontaminated data build."""
    controlled_columns = {"solution", "permuted_solution"}
    if args.data_manifest is None:
        if args.reference_column in controlled_columns:
            raise ValueError(
                "--data-manifest is required for explicit final-answer OPSD arms"
            )
        return None
    manifest_path = Path(args.data_manifest)
    manifest = json.loads(manifest_path.read_text())
    validate_final_answer_manifest(manifest)
    dataset_sha256 = file_sha256(args.dataset)
    if dataset_sha256 != manifest.get("train_sha256"):
        raise ValueError("OPSD dataset does not match manifest train bytes")
    checkpoint_traces_sha256 = (payload.get("sft") or {}).get("traces_sha256")
    if checkpoint_traces_sha256 != manifest.get("sft_corpus_sha256"):
        raise ValueError(
            "OPSD checkpoint SFT corpus does not match the data manifest"
        )
    gate_path = Path(manifest.get("gate", ""))
    if not gate_path.is_file() or file_sha256(gate_path) != manifest.get(
        "gate_sha256"
    ):
        raise ValueError("OPSD manifest gate bytes are missing or changed")
    return {
        "path": str(manifest_path),
        "sha256": file_sha256(manifest_path),
        "schema": manifest["schema"],
        "split_schema": manifest["split_schema"],
        "train_sha256": manifest["train_sha256"],
        "gate_sha256": manifest["gate_sha256"],
        "gate": str(gate_path),
        "gate_rows": int(manifest["gate_rows"]),
        "sft_corpus_sha256": manifest["sft_corpus_sha256"],
        "source_quotas": source_quotas(manifest),
        "groups_per_cycle": manifest.get("groups_per_cycle"),
    }


def validate_authorization(
    args, payload: dict, data_manifest: dict[str, object] | None
) -> dict[str, object] | None:
    controlled_columns = {"solution", "permuted_solution"}
    if args.authorization is None:
        if getattr(args, "allow_failed_authorization", False):
            raise ValueError(
                "--allow-failed-authorization requires --authorization"
            )
        if args.reference_column in controlled_columns:
            raise ValueError(
                "--authorization is required for explicit final-answer OPSD arms"
            )
        return None
    if data_manifest is None:
        raise ValueError("OPSD authorization requires a validated data manifest")
    path = Path(args.authorization)
    authorization = json.loads(path.read_text())
    if authorization.get("schema") != OPSD_AUTHORIZATION_SCHEMA:
        raise ValueError("OPSD authorization has an incompatible schema")
    decision = authorization.get("decision")
    if decision not in {"pass", "fail"}:
        raise ValueError("OPSD authorization has an invalid decision")
    override_applied = decision == "fail"
    if override_applied and not getattr(
        args, "allow_failed_authorization", False
    ):
        raise ValueError(
            "OPSD authorization decision is not pass; an explicitly requested "
            "experimental run must preserve the failed artifact and pass "
            "--allow-failed-authorization"
        )
    if authorization.get("checkpoint_sha256") != file_sha256(args.checkpoint):
        raise ValueError("OPSD authorization used different checkpoint bytes")
    if authorization.get("data_manifest_sha256") != data_manifest["sha256"]:
        raise ValueError("OPSD authorization used a different data manifest")
    if authorization.get("gate_sha256") != data_manifest["gate_sha256"]:
        raise ValueError("OPSD authorization used different held-out gate bytes")
    gate_contract = authorization.get("gate_contract")
    if gate_contract != AUTHORIZATION_GATE_CONTRACT:
        raise ValueError("OPSD authorization has a different frozen-gate contract")
    input_artifacts = authorization.get("input_artifacts")
    if not isinstance(input_artifacts, dict) or set(input_artifacts) != {
        "generation_gate",
        "logit_gate",
    }:
        raise ValueError("OPSD authorization lacks frozen-gate provenance")
    for artifact in input_artifacts.values():
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            raise ValueError("OPSD authorization has malformed gate provenance")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("OPSD authorization has malformed gate digest")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "schema": authorization["schema"],
        "decision": decision,
        "gate_decisions": authorization.get("gate_decisions"),
        "failed_authorization_override": override_applied,
        "checkpoint_sha256": authorization["checkpoint_sha256"],
        "data_manifest_sha256": authorization["data_manifest_sha256"],
        "gate_sha256": authorization["gate_sha256"],
        "gate_contract": gate_contract,
        "input_artifacts": input_artifacts,
    }


def infer_source_contract(args, payload: dict) -> None:
    """Resolve dataset and tokenizer flags from source checkpoint provenance."""
    sft = payload.get("sft") or {}
    sft_args = sft.get("args") or {}
    if (
        args.think_tokens is not None
        and "think_tokens" in sft_args
        and args.think_tokens != bool(sft_args["think_tokens"])
    ):
        raise ValueError(
            "--think-tokens must match the source checkpoint's SFT provenance"
        )
    if (
        args.answer_fence is not None
        and "answer_fence" in sft_args
        and args.answer_fence != bool(sft_args["answer_fence"])
    ):
        raise ValueError(
            "--answer-fence must match the source checkpoint's SFT provenance"
        )
    if args.think_tokens is None:
        args.think_tokens = bool(sft_args.get("think_tokens", False))
    if args.answer_fence is None:
        args.answer_fence = bool(sft_args.get("answer_fence", False))
    if args.answer_fence and not args.think_tokens:
        raise ValueError("answer-fenced checkpoints require think tokens")
    if args.answer_fence and float(
        sft.get("answer_fence_document_fraction") or 0.0
    ) < 0.99:
        raise ValueError(
            "answer-fenced OPSD requires source SFT provenance measuring "
            ">=99% structurally valid answer-fenced documents"
        )
    if (
        args.answer_fence
        and sft.get("answer_fence_prompt_schema")
        != ANSWER_FENCE_PROMPT_SCHEMA
    ):
        raise ValueError(
            "answer-fenced OPSD requires source SFT prompt schema "
            f"{ANSWER_FENCE_PROMPT_SCHEMA!r}; got "
            f"{sft.get('answer_fence_prompt_schema')!r}"
        )
    if args.dataset is None:
        args.dataset = sft.get("traces") or "postraining/data/sft_traces_v1.parquet"
    if args.distillation_temperature is None:
        args.distillation_temperature = args.temperature


def validate_source_contract(args, payload: dict) -> dict[str, object]:
    """Static compatibility gate used by both --validate-only and training."""
    # Resolve the effective arm before validating its manifest/authorization.
    # Otherwise ``auto`` on DAPO silently resolves to the controlled
    # ``solution`` arm only after those fail-closed gates have been skipped.
    args.reference_column = resolve_reference_column(
        args.dataset, args.reference_column
    )
    architecture = payload.get("architecture")
    model_config = payload.get("model_config")
    if not architecture or not isinstance(model_config, dict):
        raise ValueError("checkpoint lacks architecture/model_config metadata")
    if args.think_tokens and "gpt2vocab" not in architecture:
        raise ValueError("think tokens require a gpt2vocab checkpoint")
    vocab_size = int(model_config.get("vocab_size", 0))
    if args.top_k > vocab_size:
        raise ValueError(
            f"--top-k {args.top_k} exceeds checkpoint vocabulary {vocab_size}"
        )
    if args.eval_top_k > vocab_size:
        raise ValueError(
            f"--eval-top-k {args.eval_top_k} exceeds checkpoint vocabulary "
            f"{vocab_size}"
        )
    context_tokens = int(payload.get("train_seq_len", 1024))
    if args.max_prompt_length + args.max_completion_length > context_tokens:
        raise ValueError(
            "max prompt plus completion length exceeds the checkpoint's "
            f"pretraining context ({context_tokens})"
        )
    tokenizer = load_posttraining_tokenizer(
        architecture,
        FreshHyperparameters.tokenizer_path,
        think_tokens=args.think_tokens,
        answer_tokens=args.answer_fence,
        tokenizer_provenance=model_config.get("tokenizer_provenance"),
    )
    data_manifest = validate_data_manifest(args, payload)
    authorization = validate_authorization(args, payload, data_manifest)
    eval_contract = None
    if args.eval_every > 0:
        eval_rows = load_unique_math_rows(args.eval_data)
        styles = {answer_style(row) for row in eval_rows}
        if styles != {"exact"}:
            raise ValueError(
                "--eval-data must contain only strict exact-answer rows; "
                f"resolved styles were {sorted(styles)}"
            )
        if args.eval_rows > len(eval_rows):
            raise ValueError(
                f"--eval-rows {args.eval_rows} exceeds evaluation set size "
                f"{len(eval_rows)}"
            )
        eval_contract = {
            "path": args.eval_data,
            "sha256": file_sha256(args.eval_data),
            "rows": len(eval_rows),
        }
    examples = load_examples(
        args.dataset,
        answer_fence=args.answer_fence,
        reference_column=args.reference_column,
    )
    quotas = data_manifest.get("source_quotas") if data_manifest else None
    if quotas:
        observed_sources = Counter(example.source for example in examples)
        if set(observed_sources) != set(quotas):
            raise ValueError(
                "OPSD mixture dataset sources do not match manifest quotas: "
                f"{sorted(observed_sources)} != {sorted(quotas)}"
            )
        cycle = sum(quotas.values())
        if args.effective_batch_size % cycle:
            raise ValueError(
                "OPSD mixture effective batch size must be a multiple of its "
                f"{cycle}-trajectory source cycle"
            )
    first_valid = None
    valid_examples = 0
    rejection_counts: Counter[str] = Counter()
    for example in examples:
        tokenized, reason = tokenize_example(
            example,
            tokenizer,
            max_prompt_length=args.max_prompt_length,
            context_tokens=context_tokens,
            max_completion_length=args.max_completion_length,
        )
        if tokenized is not None:
            valid_examples += 1
            if first_valid is None:
                first_valid = tokenized
            continue
        assert reason is not None
        rejection_counts[reason] += 1
    if first_valid is None:
        raise ValueError("no dataset row fits the configured context budgets")
    if quotas and rejection_counts:
        raise ValueError(
            "prepared OPSD mixture contains runtime-invalid rows, which would "
            f"change its exact source quotas: {dict(rejection_counts)}"
        )
    return {
        "architecture": architecture,
        "context_tokens": context_tokens,
        "verified_examples": len(examples),
        "valid_examples": valid_examples,
        "first_valid_student_prompt_tokens": len(
            first_valid.student_prompt_ids
        ),
        "first_valid_teacher_prompt_tokens": len(
            first_valid.teacher_prompt_ids
        ),
        "rejections": dict(rejection_counts),
        "think_tokens": args.think_tokens,
        "answer_fence": args.answer_fence,
        "answer_fence_prompt_schema": (
            ANSWER_FENCE_PROMPT_SCHEMA if args.answer_fence else None
        ),
        "dataset": args.dataset,
        "data_manifest": data_manifest,
        "source_rows": dict(
            sorted(Counter(example.source for example in examples).items())
        ),
        "authorization": authorization,
        "evaluation": eval_contract,
    }


def _state_dict_cpu(module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
    }


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def opsd_tensorboard_metrics(entry: dict[str, object]) -> dict[str, float]:
    """Stable live dashboard tags for one canonical OPSD train record."""
    fields = {
        "loss/clipped_objective": "train_loss",
        "loss/raw_forward_kl": "forward_kl",
        "loss/clipped_entry_fraction": "clipped_entry_fraction",
        "loss/clipped_token_fraction": "clipped_token_fraction",
        "loss/max_pointwise_contribution": "max_pointwise_contribution",
        "optimization/preclip_gradient_norm": "gradient_norm",
        "rollout/mean_response_tokens": "response_tokens_mean",
        "rollout/termination_fraction": "terminated_fraction",
        "rollout/empty_response_fraction": "empty_response_fraction",
        "throughput/response_tokens_per_second": "response_tokens_per_second",
        "throughput/step_seconds": "step_time_seconds",
        "system/gpu_memory_allocated_gib": "gpu_memory_allocated_gib",
        "system/gpu_memory_reserved_gib": "gpu_memory_reserved_gib",
        "system/gpu_peak_memory_allocated_gib": (
            "gpu_peak_memory_allocated_gib"
        ),
        "system/gpu_peak_memory_reserved_gib": "gpu_peak_memory_reserved_gib",
        "data/rejected_examples": "rejected_examples",
        "data/sampler_cursor": "sampler_cursor",
        "reward/exact_accuracy": "exact_accuracy",
        "reward/raw_exact_accuracy": "raw_exact_accuracy",
        "reward/contract_accuracy": "contract_accuracy",
        "reward/structural_format_fraction": "structural_format_fraction",
        "reward/graded_trajectories": "graded_trajectories",
        "perf/reward_scoring_seconds": "reward_scoring_seconds",
    }
    return {
        tag: float(entry[field])
        for tag, field in fields.items()
        if field in entry
    }


def grade_on_policy_response(
    emitted: list[int],
    example: OPSDExample,
    tokenizer,
    *,
    stop_ids: tuple[int, ...],
    think_fence_ids: tuple[int, int] | None,
    answer_fence_ids: tuple[int, int] | None,
    min_think_tokens: int,
) -> dict[str, int] | None:
    """Grade an existing OPSD rollout without changing the OPSD objective."""
    if example.ground_truth is None or example.grading_style is None:
        return None
    exact, _ = verify_terminated_answer(
        emitted,
        example.ground_truth,
        tokenizer,
        stop_ids,
        example.grading_style,
        answer_fence_ids=answer_fence_ids,
    )
    stop_set = set(stop_ids)
    stop_cut = next(
        (index for index, token in enumerate(emitted) if token in stop_set),
        None,
    )
    if think_fence_ids is not None and answer_fence_ids is not None:
        structurally_valid = bool(
            stop_cut is not None
            and structural_format_ok(
                emitted[: stop_cut + 1],
                think_fence_ids,
                answer_fence_ids,
                min_think_tokens,
            )
        )
    else:
        structurally_valid = stop_cut is not None
    return {
        "raw_exact": int(exact),
        "exact": int(exact and structurally_valid),
        "structurally_valid": int(structurally_valid),
    }


def opsd_stop_ids(tokenizer) -> tuple[int, ...]:
    """Use one terminal-token contract for rollout, grading, and trimming."""
    return tuple(
        dict.fromkeys((int(tokenizer.eos_id()), int(tokenizer.bos_id())))
    )


def opsd_eval_tensorboard_metrics(entry: dict[str, object]) -> dict[str, float]:
    """Quality indicators from one held-out question-only policy evaluation."""
    fields = {
        "eval/accuracy": "accuracy",
        "eval/contract_accuracy": "contract_accuracy",
        "eval/descriptive_accuracy_delta_from_step0": "accuracy_delta_from_step0",
        "eval/descriptive_contract_accuracy_delta_from_step0": (
            "contract_accuracy_delta_from_step0"
        ),
        "eval/paired_delta_bootstrap_low": "paired_delta_bootstrap_low",
        "eval/paired_delta_bootstrap_high": "paired_delta_bootstrap_high",
        "eval/structural_format_fraction": "structural_format_fraction",
        "eval/termination_fraction": "ended_fraction",
        "eval/prompt_any_correct_fraction": "prompt_any_correct_fraction",
        "eval/prompt_zero_correct_fraction": "prompt_zero_correct_fraction",
        "eval/within_group_reward_std": "within_group_reward_std",
        "eval/mean_emitted_tokens": "emitted_tokens_mean",
        "perf/eval_seconds": "eval_seconds",
    }
    return {
        tag: float(entry[field])
        for tag, field in fields.items()
        if field in entry
    }


def paired_prompt_accuracy_delta(
    current: list[int],
    baseline: list[int],
    *,
    samples: int,
    seed: int,
    resamples: int = 5000,
) -> dict[str, float]:
    """Prompt-paired descriptive delta with a deterministic bootstrap band."""
    if len(current) != len(baseline) or not current:
        raise ValueError("paired accuracy counts must be nonempty and aligned")
    if samples < 1 or resamples < 1:
        raise ValueError("samples and resamples must be positive")
    deltas = (
        torch.tensor(current, dtype=torch.float64)
        - torch.tensor(baseline, dtype=torch.float64)
    ) / samples
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        len(deltas),
        (resamples, len(deltas)),
        generator=generator,
    )
    bootstrap = deltas[indices].mean(dim=1)
    return {
        "paired_delta_mean": float(deltas.mean()),
        "paired_delta_bootstrap_low": float(torch.quantile(bootstrap, 0.025)),
        "paired_delta_bootstrap_high": float(torch.quantile(bootstrap, 0.975)),
    }


def _purge_jsonl_after(path: Path, step: int) -> int:
    if not path.exists():
        return 0
    retained: list[str] = []
    removed = 0
    lines = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            if index != len(lines) - 1:
                raise ValueError(
                    f"interior JSONL corruption in {path} at line {index + 1}"
                ) from error
            # An interrupted append can only tear the final record. It is
            # newer than the last atomic checkpoint and is safe to discard.
            removed += 1
            continue
        if int(record.get("step", 0)) > step:
            removed += 1
        else:
            retained.append(line)
    if removed:
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_text("".join(retained))
        os.replace(temporary, path)
    return removed


def _export_payload(
    student,
    source_payload: dict,
    args,
    *,
    step: int,
    source_sha256: str,
    dataset_sha256: str,
    data_manifest_sha256: str | None = None,
    authorization_sha256: str | None = None,
    eval_data_sha256: str | None = None,
) -> dict:
    is_permuted_control = getattr(args, "reference_column", None) == (
        "permuted_solution"
    )
    metadata = {
        "schema": OPSD_EXPORT_SCHEMA,
        "objective_schema": OPSD_OBJECTIVE_SCHEMA,
        "prompt_schema": OPSD_PROMPT_SCHEMA,
        "answer_fence_prompt_schema": (
            ANSWER_FENCE_PROMPT_SCHEMA
            if getattr(args, "answer_fence", False)
            else None
        ),
        "paper": PAPER_URL,
        "paper_code_commit": PAPER_CODE_COMMIT,
        "teacher": (
            "frozen_step0_same_checkpoint_permuted_answer_control"
            if is_permuted_control
            else "frozen_step0_same_checkpoint_privileged_context"
        ),
        "reference_column": getattr(args, "reference_column", "auto"),
        "control_role": (
            "split_local_permuted_answer_negative_control"
            if is_permuted_control
            else None
        ),
        "student_rollouts": "on_policy_one_per_prompt",
        "divergence": "full_vocabulary_forward_kl",
        "pointwise_kl_clip": args.pointwise_kl_clip,
        "temperature": args.distillation_temperature,
        "stop_token_distilled": False,
        "step": step,
        "source_checkpoint": args.checkpoint,
        "source_checkpoint_sha256": source_sha256,
        "dataset": args.dataset,
        "dataset_sha256": dataset_sha256,
        "data_manifest": getattr(args, "data_manifest", None),
        "data_manifest_sha256": data_manifest_sha256,
        "authorization": getattr(args, "authorization", None),
        "authorization_sha256": authorization_sha256,
        "eval_data": getattr(args, "eval_data", None),
        "eval_data_sha256": eval_data_sha256,
        "args": vars(args),
        "parent_opsd": source_payload.get("opsd"),
    }
    payload = {
        "model": _state_dict_cpu(student),
        "model_config": dict(student.model_config),
        "architecture": student.architecture,
        "train_seq_len": int(
            getattr(
                student,
                "train_context_tokens",
                source_payload.get("train_seq_len", 1024),
            )
        ),
        "opsd": metadata,
    }
    if source_payload.get("sft") is not None:
        # Downstream VAPO reconstructs trained fence tokens from this exact
        # provenance; OPSD must not erase that interface contract.
        payload["sft"] = source_payload["sft"]
    return payload


class OPSDTrainer:
    def __init__(self, args, source_payload: dict):
        if not torch.cuda.is_available():
            raise RuntimeError("OPSD training requires CUDA; submit it through mlq")
        data_manifest_contract = validate_data_manifest(args, source_payload)
        validate_authorization(args, source_payload, data_manifest_contract)
        self.args = args
        # Keep lineage metadata, not a third long-lived CPU copy of the base
        # weights. The two GPU roles are constructed from ``source_payload``
        # below before the caller releases it.
        self.source_payload = {
            key: value for key, value in source_payload.items() if key != "model"
        }
        self.device = torch.device("cuda")
        self.output = Path("postraining/runs") / args.name
        self.output.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output / "metrics.jsonl"
        self.generations_path = self.output / "generations.jsonl"
        self.source_sha256 = file_sha256(args.checkpoint)
        self.dataset_sha256 = file_sha256(args.dataset)
        self.data_manifest_sha256 = (
            file_sha256(args.data_manifest) if args.data_manifest else None
        )
        self.authorization_sha256 = (
            file_sha256(args.authorization) if args.authorization else None
        )
        self.eval_data_sha256 = (
            file_sha256(args.eval_data) if args.eval_every > 0 else None
        )
        self.context_tokens = int(source_payload.get("train_seq_len", 1024))

        self.tokenizer = load_posttraining_tokenizer(
            source_payload["architecture"],
            FreshHyperparameters.tokenizer_path,
            think_tokens=args.think_tokens,
            answer_tokens=args.answer_fence,
            tokenizer_provenance=source_payload["model_config"].get(
                "tokenizer_provenance"
            ),
        )
        self.examples = load_examples(
            args.dataset,
            answer_fence=args.answer_fence,
            reference_column=args.reference_column,
        )
        self.eval_rows: list[dict] = []
        if args.eval_every > 0:
            eval_rows = load_unique_math_rows(args.eval_data)
            styles = {answer_style(row) for row in eval_rows}
            if styles != {"exact"}:
                raise ValueError(
                    "--eval-data must contain only strict exact-answer rows; "
                    f"resolved styles were {sorted(styles)}"
                )
            if args.answer_fence:
                eval_rows = canonicalize_answer_fence_rows(eval_rows)
            self.eval_rows = deterministic_math_subset(
                eval_rows, args.eval_rows
            )
        quotas = (
            data_manifest_contract.get("source_quotas")
            if data_manifest_contract
            else None
        )
        if quotas:
            self.sampler = SourceQuotaSampler(
                self.examples, quotas, args.seed
            )
        else:
            self.sampler = ShuffledExampleSampler(self.examples, args.seed)
        self._tokenized_cache = {}
        self.rejection_counts: Counter[str] = Counter()

        # Both roles instantiate the identical source checkpoint. The teacher
        # is then frozen forever; unlike the authors' LoRA implementation,
        # a separate copy is cheap and exact for this 16 MB submission model.
        self.student = load_model(
            args.checkpoint, self.device, payload=source_payload
        )
        self.teacher = load_model(
            args.checkpoint, self.device, payload=source_payload
        )
        for parameter in self.student.parameters():
            parameter.requires_grad_(True)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()
        self.rollout_policy = LatentThoughtModel(
            self.student, num_blocks=0
        ).to(self.device)
        self.rollout_step_core = None
        if args.rollout_compile:
            # The current VAPO production path uses this same dynamic,
            # no-cudagraph recurrent decode artifact. OPSD has variable
            # prompt and survivor widths, so static specialization would
            # compile repeatedly across the 100-step run.
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit, 64
            )
            self.rollout_step_core = torch.compile(
                self.rollout_policy.step_core,
                mode="max-autotune-no-cudagraphs",
                fullgraph=True,
                dynamic=True,
            )
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
            fused=True,
        )
        self.rollout_generator = torch.Generator(device=self.device)
        self.rollout_generator.manual_seed(args.seed)
        self.step = 0
        if args.resume:
            self._restore(Path(args.resume))
        else:
            self.metrics_path.write_text("")
            self.generations_path.write_text("")
        self.eval_baseline_accuracy: float | None = None
        self.eval_baseline_contract_accuracy: float | None = None
        self.eval_baseline_prompt_counts: list[int] | None = None
        if args.resume and self.metrics_path.exists():
            for line in self.metrics_path.read_text().splitlines():
                record = json.loads(line)
                if record.get("type") == "eval" and record.get("step") == 0:
                    self.eval_baseline_accuracy = float(record["accuracy"])
                    self.eval_baseline_contract_accuracy = float(
                        record["contract_accuracy"]
                    )
                    self.eval_baseline_prompt_counts = [
                        int(value) for value in record["prompt_correct_counts"]
                    ]
                    break
        self.tensorboard = SummaryWriter(
            self.output / "tensorboard",
            purge_step=self.step + 1 if args.resume else None,
        )

    def _evaluate_policy(self, step: int) -> dict[str, object]:
        """Evaluate question-only accuracy on a fixed, SFT-decontaminated gate."""
        if not self.eval_rows:
            raise RuntimeError("OPSD policy evaluation has no held-out rows")
        # Training has already consumed this step's gradients. Releasing them
        # keeps policy evaluation independent of full-vocabulary backward
        # memory while preserving optimizer moments and parameters.
        self.optimizer.zero_grad(set_to_none=True)
        self.rollout_policy.eval()
        started = time.perf_counter()
        metrics = evaluate_latent_math(
            self.rollout_policy,
            self.tokenizer,
            self.eval_rows,
            self.args.eval_samples,
            self.args.max_completion_length,
            self.args.max_completion_length,
            self.args.eval_samples,
            self.args.seed,
            self.device,
            self.args.max_prompt_length,
            batch_trajectories=self.args.eval_batch_trajectories,
            # Keep evaluation execution independent from the compiled
            # training decoder so fallback cannot alter resume behavior.
            compiled_step_core=None,
            temperature=self.args.eval_temperature,
            top_p=self.args.eval_top_p,
            top_k=self.args.eval_top_k or None,
            pin_emit=True,
            think_fence_ids=(
                (self.tokenizer.think_open_id, self.tokenizer.think_close_id)
                if self.args.think_tokens
                else None
            ),
            answer_fence_ids=(
                (self.tokenizer.answer_open_id, self.tokenizer.answer_close_id)
                if self.args.answer_fence
                else None
            ),
            min_think_tokens=self.args.eval_think_min_tokens,
        )
        torch.cuda.synchronize(self.device)
        accuracy = float(metrics["accuracy"])
        contract_accuracy = float(metrics["contract_accuracy"])
        if step == 0:
            self.eval_baseline_accuracy = accuracy
            self.eval_baseline_contract_accuracy = contract_accuracy
            self.eval_baseline_prompt_counts = [
                int(value) for value in metrics["prompt_correct_counts"]
            ]
        if (
            self.eval_baseline_accuracy is None
            or self.eval_baseline_contract_accuracy is None
            or self.eval_baseline_prompt_counts is None
        ):
            raise RuntimeError("OPSD evaluation is missing its step-0 baseline")
        paired = paired_prompt_accuracy_delta(
            [int(value) for value in metrics["prompt_correct_counts"]],
            self.eval_baseline_prompt_counts,
            samples=self.args.eval_samples,
            seed=self.args.seed + step,
        )
        entry: dict[str, object] = {
            "type": "eval",
            "step": step,
            "eval_seconds": time.perf_counter() - started,
            "accuracy_delta_from_step0": (
                accuracy - self.eval_baseline_accuracy
            ),
            "contract_accuracy_delta_from_step0": (
                contract_accuracy - self.eval_baseline_contract_accuracy
            ),
            **paired,
            **metrics,
        }
        _append_jsonl(self.metrics_path, entry)
        for tag, value in opsd_eval_tensorboard_metrics(entry).items():
            self.tensorboard.add_scalar(tag, value, step)
        self.tensorboard.flush()
        print(
            f"step {step}: heldout accuracy {accuracy:.4f}, "
            f"contract {contract_accuracy:.4f}, "
            f"delta {entry['accuracy_delta_from_step0']:+.4f}",
            flush=True,
        )
        return entry

    def _restore(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != OPSD_CHECKPOINT_SCHEMA:
            raise ValueError("resume checkpoint has an incompatible schema")
        if payload.get("objective_schema") != OPSD_OBJECTIVE_SCHEMA:
            raise ValueError("resume checkpoint has a different objective")
        if payload.get("prompt_schema") != OPSD_PROMPT_SCHEMA:
            raise ValueError("resume checkpoint has a different prompt schema")
        if payload.get("data_order_schema") != OPSD_DATA_ORDER_SCHEMA:
            raise ValueError("resume checkpoint has a different data order")
        if (
            self.args.answer_fence
            and payload.get("answer_fence_prompt_schema")
            != ANSWER_FENCE_PROMPT_SCHEMA
        ):
            raise ValueError(
                "resume checkpoint has a different answer-fence prompt schema"
            )
        saved_args = payload["args"]
        for field in _RESUME_EXACT_FIELDS:
            if saved_args.get(field) != getattr(self.args, field):
                raise ValueError(
                    f"resume requires the saved --{field.replace('_', '-')}: "
                    f"{saved_args.get(field)!r} != {getattr(self.args, field)!r}"
                )
        if self.args.eval_every > 0:
            if int(saved_args.get("eval_every", 0)) <= 0:
                raise ValueError(
                    "cannot enable inline evaluation on resume without a "
                    "saved step-0 baseline; run it as a separate evaluation"
                )
            for field in _RESUME_EVAL_FIELDS:
                if saved_args.get(field) != getattr(self.args, field):
                    raise ValueError(
                        "resumed inline evaluation requires the saved "
                        f"--{field.replace('_', '-')}: "
                        f"{saved_args.get(field)!r} != "
                        f"{getattr(self.args, field)!r}"
                    )
        if payload["source_checkpoint_sha256"] != self.source_sha256:
            raise ValueError("source checkpoint bytes changed since the run began")
        if payload["dataset_sha256"] != self.dataset_sha256:
            raise ValueError("OPSD dataset bytes changed since the run began")
        if payload.get("data_manifest_sha256") != self.data_manifest_sha256:
            raise ValueError("OPSD data manifest changed since the run began")
        if payload.get("authorization_sha256") != self.authorization_sha256:
            raise ValueError("OPSD authorization changed since the run began")
        # Evaluation is observability-only and saves/restores RNG state. It
        # may be disabled or reconfigured on an exact optimization resume;
        # when enabled against the same path, its bound bytes must still be
        # unchanged.
        if (
            self.args.eval_every > 0
            and saved_args.get("eval_data") == self.args.eval_data
            and payload.get("eval_data_sha256") not in {
                None,
                self.eval_data_sha256,
            }
        ):
            raise ValueError("OPSD evaluation data changed since the run began")
        self.student.load_state_dict(payload["model"], strict=True)
        self.teacher.load_state_dict(payload["teacher_model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step = int(payload["step"])
        if self.args.steps < self.step:
            raise ValueError("--steps cannot precede the resume step")
        self.sampler.cursor = int(payload["sampler_cursor"])
        if isinstance(self.sampler, SourceQuotaSampler):
            if self.sampler.cursor != self.step * self.args.effective_batch_size:
                raise ValueError(
                    "mixture sampler cursor does not equal step times effective "
                    "batch size"
                )
            if payload.get("sampler_source_quotas") != self.sampler.quotas:
                raise ValueError("resume checkpoint has different source quotas")
            if payload.get(
                "sampler_source_consumed"
            ) != self.sampler.consumed_counts():
                raise ValueError("resume checkpoint source cursors are inconsistent")
        self.rejection_counts = Counter(payload.get("rejection_counts", {}))
        self.rollout_generator.set_state(payload["rollout_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        torch.cuda.set_rng_state(payload["cuda_rng_state"], self.device)
        _purge_jsonl_after(self.metrics_path, self.step)
        _purge_jsonl_after(self.generations_path, self.step)

    def _next_tokenized(self, count: int):
        selected = []
        attempts = 0
        # Static validation proves at least one row fits. Scanning up to one
        # partial plus ``count`` complete epochs therefore guarantees enough
        # accepted rows even in the sparsest valid dataset; tokenization
        # outcomes are cached after the first visit.
        maximum_attempts = len(self.examples) * (count + 1)
        while len(selected) < count and attempts < maximum_attempts:
            example = self.sampler.next()
            attempts += 1
            cached = self._tokenized_cache.get(example)
            if cached is None and example not in self._tokenized_cache:
                cached = tokenize_example(
                    example,
                    self.tokenizer,
                    max_prompt_length=self.args.max_prompt_length,
                    context_tokens=self.context_tokens,
                    max_completion_length=self.args.max_completion_length,
                )
                self._tokenized_cache[example] = cached
            tokenized, reason = cached
            if tokenized is None:
                assert reason is not None
                self.rejection_counts[reason] += 1
                continue
            selected.append(tokenized)
        if len(selected) != count:
            raise RuntimeError(
                "not enough dataset rows fit the OPSD context budgets; "
                f"accepted {len(selected)}/{count}, rejections "
                f"{dict(self.rejection_counts)}"
            )
        if isinstance(self.sampler, SourceQuotaSampler):
            factor = count // len(self.sampler.schedule)
            expected = {
                source: quota * factor
                for source, quota in self.sampler.quotas.items()
            }
            observed = Counter(
                example.example.source for example in selected
            )
            if observed != Counter(expected):
                raise RuntimeError(
                    f"OPSD source quota drift: {dict(observed)} != {expected}"
                )
        return selected

    def _rollout(self, examples):
        prompt_lengths = torch.tensor(
            [len(example.student_prompt_ids) for example in examples],
            dtype=torch.long,
            device=self.device,
        )
        width = int(prompt_lengths.max())
        eos = int(self.tokenizer.eos_id())
        stop_ids = opsd_stop_ids(self.tokenizer)
        prompts = torch.full(
            (len(examples), width), eos, dtype=torch.long, device=self.device
        )
        for row, example in enumerate(examples):
            ids = torch.tensor(
                example.student_prompt_ids,
                dtype=torch.long,
                device=self.device,
            )
            prompts[row, width - ids.numel():] = ids
        self.rollout_policy.eval()
        original_step_core = self.rollout_policy.step_core
        if self.rollout_step_core is not None:
            self.rollout_policy.step_core = self.rollout_step_core
        try:
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                batch = rollout_continuations(
                    self.rollout_policy,
                    prompts,
                    self.args.max_completion_length,
                    self.args.max_completion_length,
                    self.args.temperature,
                    self.args.top_p,
                    generator=self.rollout_generator,
                    stop_ids=stop_ids,
                    prompt_lengths=prompt_lengths,
                    tensor_positions=self.rollout_step_core is not None,
                    replay_storage=False,
                    record_likelihoods=False,
                    cache_dtype=torch.bfloat16,
                    pin_emit=True,
                    top_k=self.args.top_k or None,
                )
        finally:
            self.rollout_policy.step_core = original_step_core
        return emitted_token_rows(batch)

    def _save(self, *, final: bool) -> None:
        export = _export_payload(
            self.student,
            self.source_payload,
            self.args,
            step=self.step,
            source_sha256=self.source_sha256,
            dataset_sha256=self.dataset_sha256,
            data_manifest_sha256=self.data_manifest_sha256,
            authorization_sha256=self.authorization_sha256,
            eval_data_sha256=self.eval_data_sha256,
        )
        resume_payload = {
            "schema": OPSD_CHECKPOINT_SCHEMA,
            "objective_schema": OPSD_OBJECTIVE_SCHEMA,
            "prompt_schema": OPSD_PROMPT_SCHEMA,
            "optimizer_schema": OPSD_OPTIMIZER_SCHEMA,
            "data_order_schema": OPSD_DATA_ORDER_SCHEMA,
            "answer_fence_prompt_schema": (
                ANSWER_FENCE_PROMPT_SCHEMA
                if self.args.answer_fence
                else None
            ),
            "model": export["model"],
            "teacher_model": _state_dict_cpu(self.teacher),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "sampler_cursor": self.sampler.cursor,
            "sampler_source_quotas": (
                self.sampler.quotas
                if isinstance(self.sampler, SourceQuotaSampler)
                else None
            ),
            "sampler_source_consumed": (
                self.sampler.consumed_counts()
                if isinstance(self.sampler, SourceQuotaSampler)
                else None
            ),
            "rejection_counts": dict(self.rejection_counts),
            "rollout_rng_state": self.rollout_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(self.device),
            "source_checkpoint_sha256": self.source_sha256,
            "dataset_sha256": self.dataset_sha256,
            "data_manifest_sha256": self.data_manifest_sha256,
            "authorization_sha256": self.authorization_sha256,
            "eval_data_sha256": self.eval_data_sha256,
            "args": vars(self.args),
        }
        checkpoint_dir = self.output / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        _atomic_torch_save(
            export, checkpoint_dir / f"step_{self.step:06d}.pt"
        )
        # Commit the versioned artifact before advancing exact recovery
        # state. An interrupted export leaves the older resume point intact;
        # an interrupted final alias is regenerated from the new resume.
        _atomic_torch_save(
            resume_payload, self.output / "opsd_checkpoint.pt"
        )
        if final:
            _atomic_torch_save(export, self.output / "opsd_final_model.pt")

    def train(self) -> None:
        if self.args.eval_every > 0 and not self.args.resume and self.step == 0:
            self._evaluate_policy(0)
        started = time.perf_counter()
        stop_ids = opsd_stop_ids(self.tokenizer)
        think_fence_ids = (
            (self.tokenizer.think_open_id, self.tokenizer.think_close_id)
            if self.args.think_tokens
            else None
        )
        answer_fence_ids = (
            (self.tokenizer.answer_open_id, self.tokenizer.answer_close_id)
            if self.args.answer_fence
            else None
        )
        while self.step < self.args.steps:
            step_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(self.device)
            examples = self._next_tokenized(self.args.effective_batch_size)
            self.optimizer.zero_grad(set_to_none=True)
            per_example_metrics = []
            generated_for_log = []
            empty_responses = 0
            reward_rows: list[dict[str, int | str]] = []
            reward_scoring_seconds = 0.0
            for start in range(
                0, len(examples), self.args.rollout_batch_size
            ):
                chunk = examples[start:start + self.args.rollout_batch_size]
                responses = self._rollout(chunk)
                for example, response in zip(chunk, responses, strict=True):
                    emitted_response = response
                    reward_started = time.perf_counter()
                    reward = grade_on_policy_response(
                        emitted_response,
                        example.example,
                        self.tokenizer,
                        stop_ids=stop_ids,
                        think_fence_ids=think_fence_ids,
                        answer_fence_ids=answer_fence_ids,
                        min_think_tokens=OPSD_REWARD_MIN_THINK_TOKENS,
                    )
                    reward_scoring_seconds += time.perf_counter() - reward_started
                    if reward is not None:
                        reward["source"] = example.example.source
                        reward_rows.append(reward)
                    terminated = bool(response and response[-1] in stop_ids)
                    # Generation stop markers are not completion content in
                    # the paper code (EOS doubles as its padding token), so
                    # the distillation trajectory excludes the terminal EOT.
                    if terminated:
                        response = response[:-1]
                    if not response:
                        empty_responses += 1
                        continue
                    response_tensor = torch.tensor(
                        response, dtype=torch.long, device=self.device
                    )
                    self.student.train()
                    metrics = backward_opsd_example(
                        self.student,
                        self.teacher,
                        torch.tensor(
                            example.student_prompt_ids,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        torch.tensor(
                            example.teacher_prompt_ids,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        response_tensor,
                        batch_denominator=self.args.effective_batch_size,
                        temperature=resolved_distillation_temperature(
                            self.args
                        ),
                        pointwise_clip=(
                            self.args.pointwise_kl_clip
                            if self.args.pointwise_kl_clip > 0
                            else None
                        ),
                        logit_chunk_tokens=self.args.logit_chunk_tokens,
                    )
                    metrics["terminated"] = int(terminated)
                    per_example_metrics.append(metrics)
                    if (
                        len(generated_for_log)
                        < self.args.generation_log_samples
                    ):
                        generated_for_log.append(
                            {
                                "problem": example.example.problem,
                                "source": example.example.source,
                                "response": self.tokenizer.decode(response),
                                "response_tokens": len(response),
                                "response_token_ids": response,
                                "terminated": terminated,
                            }
                        )
            if not per_example_metrics:
                raise RuntimeError("every on-policy response was empty")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.args.max_grad_norm
            )
            if not math.isfinite(float(grad_norm)):
                raise FloatingPointError(f"non-finite gradient norm {grad_norm}")
            self.optimizer.step()
            self.step += 1

            def mean(name: str) -> float:
                values = [row[name] for row in per_example_metrics]
                first = values[0]
                if torch.is_tensor(first):
                    return float(torch.stack(values).mean())
                return sum(float(value) for value in values) / len(values)

            def algorithm_batch_mean(name: str) -> float:
                values = [row[name] for row in per_example_metrics]
                first = values[0]
                if torch.is_tensor(first):
                    return float(
                        torch.stack(values).sum()
                        / self.args.effective_batch_size
                    )
                return sum(float(value) for value in values) / (
                    self.args.effective_batch_size
                )

            # Make timing and peak-memory telemetry cover the complete CUDA
            # update rather than only host-side dispatch.
            torch.cuda.synchronize(self.device)
            step_seconds = time.perf_counter() - step_started
            response_tokens_total = sum(
                int(row["response_tokens"]) for row in per_example_metrics
            )
            entry = {
                "type": "train",
                "step": self.step,
                "train_loss": algorithm_batch_mean("loss"),
                "forward_kl": algorithm_batch_mean("forward_kl"),
                "clipped_entry_fraction": mean("clipped_entry_fraction"),
                "clipped_token_fraction": mean("clipped_token_fraction"),
                "max_pointwise_contribution": float(
                    torch.stack(
                        [
                            row["max_pointwise_contribution"]
                            for row in per_example_metrics
                        ]
                    ).max()
                ),
                "response_tokens_mean": mean("response_tokens"),
                "response_tokens_total": response_tokens_total,
                "response_tokens_per_second": (
                    response_tokens_total / step_seconds
                ),
                "terminated_fraction": mean("terminated"),
                "empty_response_fraction": (
                    empty_responses / self.args.effective_batch_size
                ),
                "gradient_norm": float(grad_norm),
                "sampler_cursor": self.sampler.cursor,
                "rejected_examples": sum(self.rejection_counts.values()),
                "source_trajectories": dict(
                    sorted(
                        Counter(
                            example.example.source for example in examples
                        ).items()
                    )
                ),
                "step_time_seconds": step_seconds,
                "train_time_ms": (time.perf_counter() - started) * 1000,
                "gpu_memory_allocated_gib": (
                    torch.cuda.memory_allocated(self.device) / (1024 ** 3)
                ),
                "gpu_memory_reserved_gib": (
                    torch.cuda.memory_reserved(self.device) / (1024 ** 3)
                ),
                "gpu_peak_memory_allocated_gib": (
                    torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                ),
                "gpu_peak_memory_reserved_gib": (
                    torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)
                ),
            }
            if reward_rows:
                graded = len(reward_rows)
                exact_accuracy = sum(
                    row["exact"] for row in reward_rows
                ) / graded
                entry.update(
                    {
                        # Match VAPO's reward/exact_accuracy: correctness only
                        # earns exact reward when the response also satisfies
                        # the trained structural contract.
                        "exact_accuracy": exact_accuracy,
                        "contract_accuracy": exact_accuracy,
                        "raw_exact_accuracy": sum(
                            row["raw_exact"] for row in reward_rows
                        ) / graded,
                        "structural_format_fraction": sum(
                            row["structurally_valid"] for row in reward_rows
                        ) / graded,
                        "graded_trajectories": graded,
                        "reward_scoring_seconds": reward_scoring_seconds,
                    }
                )
                source_reward_metrics = {}
                for source in sorted(
                    {str(row["source"]) for row in reward_rows}
                ):
                    source_rows = [
                        row
                        for row in reward_rows
                        if row["source"] == source
                    ]
                    source_graded = len(source_rows)
                    source_reward_metrics[source] = {
                        "graded_trajectories": source_graded,
                        "exact_accuracy": sum(
                            int(row["exact"]) for row in source_rows
                        )
                        / source_graded,
                        "raw_exact_accuracy": sum(
                            int(row["raw_exact"]) for row in source_rows
                        )
                        / source_graded,
                        "structural_format_fraction": sum(
                            int(row["structurally_valid"])
                            for row in source_rows
                        )
                        / source_graded,
                    }
                entry["source_reward_metrics"] = source_reward_metrics
            _append_jsonl(self.metrics_path, entry)
            for tag, value in opsd_tensorboard_metrics(entry).items():
                self.tensorboard.add_scalar(tag, value, self.step)
            for source, count in entry["source_trajectories"].items():
                self.tensorboard.add_scalar(
                    f"data/source_fraction/{source}",
                    count / self.args.effective_batch_size,
                    self.step,
                )
            for source, metrics in entry.get(
                "source_reward_metrics", {}
            ).items():
                for metric, value in metrics.items():
                    self.tensorboard.add_scalar(
                        f"reward/source/{source}/{metric}", value, self.step
                    )
            self.tensorboard.add_scalar(
                "config/rollout_batch_size",
                self.args.rollout_batch_size,
                self.step,
            )
            self.tensorboard.flush()
            for generation in generated_for_log:
                _append_jsonl(
                    self.generations_path,
                    {"step": self.step, **generation},
                )
            print(
                f"step {self.step}: loss {entry['train_loss']:.6f}, "
                f"forward_kl {entry['forward_kl']:.6f}, "
                f"tokens {entry['response_tokens_mean']:.1f}, "
                f"grad {entry['gradient_norm']:.4f}",
                flush=True,
            )
            if (
                self.args.eval_every > 0
                and self.step % self.args.eval_every == 0
            ):
                self._evaluate_policy(self.step)
            if (
                self.step % self.args.save_every == 0
                and self.step < self.args.steps
            ):
                self._save(final=False)

        # Always commit the exact state before the final export. This also
        # repairs a prior run interrupted between those two atomic writes.
        self._save(final=True)
        result = {
            "schema": OPSD_EXPORT_SCHEMA,
            "objective_schema": OPSD_OBJECTIVE_SCHEMA,
            "answer_fence_prompt_schema": (
                ANSWER_FENCE_PROMPT_SCHEMA
                if self.args.answer_fence
                else None
            ),
            "checkpoint": str(self.output / "opsd_final_model.pt"),
            "steps": self.step,
            "sampler_cursor": self.sampler.cursor,
            "source_checkpoint": self.args.checkpoint,
            "source_checkpoint_sha256": self.source_sha256,
            "dataset": self.args.dataset,
            "dataset_sha256": self.dataset_sha256,
            "data_manifest": self.args.data_manifest,
            "data_manifest_sha256": self.data_manifest_sha256,
            "authorization": self.args.authorization,
            "authorization_sha256": self.authorization_sha256,
            "eval_data": self.args.eval_data if self.args.eval_every > 0 else None,
            "eval_data_sha256": self.eval_data_sha256,
            "rejections": dict(self.rejection_counts),
            "args": vars(self.args),
        }
        _atomic_json(result, self.output / "result.json")

    def close(self) -> None:
        self.tensorboard.close()


def write_manifest(output: Path, args, contract: dict[str, object]) -> None:
    provenance = capture_source_provenance(output / "provenance", Path.cwd())
    is_permuted_control = args.reference_column == "permuted_solution"
    manifest = {
        "schema": OPSD_EXPORT_SCHEMA,
        "checkpoint_schema": OPSD_CHECKPOINT_SCHEMA,
        "objective_schema": OPSD_OBJECTIVE_SCHEMA,
        "optimizer_schema": OPSD_OPTIMIZER_SCHEMA,
        "prompt_schema": OPSD_PROMPT_SCHEMA,
        "answer_fence_prompt_schema": (
            ANSWER_FENCE_PROMPT_SCHEMA if args.answer_fence else None
        ),
        "data_order_schema": OPSD_DATA_ORDER_SCHEMA,
        "paper": PAPER_URL,
        "paper_code_commit": PAPER_CODE_COMMIT,
        "paper_alignment": {
            "on_policy_student_rollouts": True,
            "same_model_teacher_student": True,
            "teacher_privileged_solution": not is_permuted_control,
            "permuted_answer_negative_control": is_permuted_control,
            "teacher_frozen_at_step0": True,
            "full_vocabulary_forward_kl": True,
            "pointwise_vocabulary_contribution_clipping": (
                args.pointwise_kl_clip > 0
            ),
            "student_logits_only_receive_gradients": True,
            "one_generation_per_prompt": True,
            "terminal_eos_excluded_like_authors_padding_mask": True,
            "paper_algorithm1_per_response_then_batch_mean": True,
        },
        "project_adaptations": {
            "full_model_finetune_instead_of_lora": (
                "The challenge model is small enough for an exact separate "
                "step-0 teacher; no adapter approximation is needed."
            ),
            "prompt_mode": (
                "The KDA backbone has no Qwen thinking-mode switch. Preserve "
                "the selected SFT checkpoint's trained think/answer fence "
                "contract; the privileged teacher alone receives the "
                "independent-reasoning transition."
            ),
        },
        "contract": contract,
        "args": vars(args),
        "provenance": provenance,
    }
    _atomic_json(manifest, output / "manifest.json")

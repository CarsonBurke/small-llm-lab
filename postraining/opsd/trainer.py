"""End-to-end OPSD training over a project-native language-model checkpoint."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import torch

from fresh_lejepa_train import FreshHyperparameters
from postraining.core import load_posttraining_tokenizer
from postraining.latent_rollout import emitted_token_rows, rollout_continuations
from postraining.latent_thought import LatentThoughtModel
from postraining.model_io import load_model
from postraining.math_prompt import ANSWER_FENCE_PROMPT_SCHEMA
from postraining.opsd.config import resolved_distillation_temperature
from postraining.opsd.data import (
    ShuffledExampleSampler,
    file_sha256,
    load_examples,
    tokenize_example,
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

_DAPO_MANIFEST_SCHEMA = "dapo_opsd_final_answer_privilege/v1"
_DAPO_SPLIT_SCHEMA = (
    "sha256_clean_gate_then_split_local_answer_derangement/v2"
)


def validate_data_manifest(args, payload: dict) -> dict[str, object] | None:
    """Bind explicit DAPO arms to one immutable, decontaminated data build."""
    controlled_columns = {"solution", "permuted_solution"}
    if args.data_manifest is None:
        if args.reference_column in controlled_columns:
            raise ValueError(
                "--data-manifest is required for explicit final-answer OPSD arms"
            )
        return None
    manifest_path = Path(args.data_manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != _DAPO_MANIFEST_SCHEMA:
        raise ValueError("OPSD data manifest has an incompatible schema")
    if manifest.get("split_schema") != _DAPO_SPLIT_SCHEMA:
        raise ValueError(
            "OPSD data manifest lacks the split-local permutation contract"
        )
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
        "sft_corpus_sha256": manifest["sft_corpus_sha256"],
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
    )
    data_manifest = validate_data_manifest(args, payload)
    examples = load_examples(
        args.dataset,
        answer_fence=args.answer_fence,
        reference_column=args.reference_column,
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
        self.context_tokens = int(source_payload.get("train_seq_len", 1024))

        self.tokenizer = load_posttraining_tokenizer(
            source_payload["architecture"],
            FreshHyperparameters.tokenizer_path,
            think_tokens=args.think_tokens,
            answer_tokens=args.answer_fence,
        )
        self.examples = load_examples(
            args.dataset,
            answer_fence=args.answer_fence,
            reference_column=args.reference_column,
        )
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

    def _restore(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != OPSD_CHECKPOINT_SCHEMA:
            raise ValueError("resume checkpoint has an incompatible schema")
        if payload.get("objective_schema") != OPSD_OBJECTIVE_SCHEMA:
            raise ValueError("resume checkpoint has a different objective")
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
        if payload["source_checkpoint_sha256"] != self.source_sha256:
            raise ValueError("source checkpoint bytes changed since the run began")
        if payload["dataset_sha256"] != self.dataset_sha256:
            raise ValueError("OPSD dataset bytes changed since the run began")
        if payload.get("data_manifest_sha256") != self.data_manifest_sha256:
            raise ValueError("OPSD data manifest changed since the run began")
        self.student.load_state_dict(payload["model"], strict=True)
        self.teacher.load_state_dict(payload["teacher_model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step = int(payload["step"])
        if self.args.steps < self.step:
            raise ValueError("--steps cannot precede the resume step")
        self.sampler.cursor = int(payload["sampler_cursor"])
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
        return selected

    def _rollout(self, examples):
        prompt_lengths = torch.tensor(
            [len(example.student_prompt_ids) for example in examples],
            dtype=torch.long,
            device=self.device,
        )
        width = int(prompt_lengths.max())
        eos = int(self.tokenizer.eos_id())
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
                    stop_ids=eos,
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
        )
        resume_payload = {
            "schema": OPSD_CHECKPOINT_SCHEMA,
            "objective_schema": OPSD_OBJECTIVE_SCHEMA,
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
            "rejection_counts": dict(self.rejection_counts),
            "rollout_rng_state": self.rollout_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(self.device),
            "source_checkpoint_sha256": self.source_sha256,
            "dataset_sha256": self.dataset_sha256,
            "data_manifest_sha256": self.data_manifest_sha256,
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
        started = time.perf_counter()
        eos = int(self.tokenizer.eos_id())
        while self.step < self.args.steps:
            examples = self._next_tokenized(self.args.effective_batch_size)
            self.optimizer.zero_grad(set_to_none=True)
            per_example_metrics = []
            generated_for_log = []
            empty_responses = 0
            for start in range(
                0, len(examples), self.args.rollout_batch_size
            ):
                chunk = examples[start:start + self.args.rollout_batch_size]
                responses = self._rollout(chunk)
                for example, response in zip(chunk, responses, strict=True):
                    terminated = bool(response and response[-1] == eos)
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
                "terminated_fraction": mean("terminated"),
                "empty_response_fraction": (
                    empty_responses / self.args.effective_batch_size
                ),
                "gradient_norm": float(grad_norm),
                "sampler_cursor": self.sampler.cursor,
                "rejected_examples": sum(self.rejection_counts.values()),
                "train_time_ms": (time.perf_counter() - started) * 1000,
            }
            _append_jsonl(self.metrics_path, entry)
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
            "rejections": dict(self.rejection_counts),
            "args": vars(self.args),
        }
        _atomic_json(result, self.output / "result.json")


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

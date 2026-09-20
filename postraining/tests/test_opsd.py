"""Paper objective, prompt, data, sampling, and export contracts for OPSD."""

from __future__ import annotations

import copy
import hashlib
import json
from argparse import Namespace
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn

from postraining.core import parse_numeric_answer, top_p_sample
from postraining.math_prompt import ANSWER_FENCE_PROMPT_SCHEMA
from postraining.model_io import load_model
from postraining.nano_backbone import NanoGPTBackbone
from postraining.opsd.config import build_arg_parser, validate_args
from postraining.opsd.compare_policy import classify
from postraining.opsd.data import (
    ShuffledExampleSampler,
    SourceQuotaSampler,
    build_teacher_prompt,
    load_examples,
    resolve_reference_column,
    tokenize_example,
)
from postraining.opsd.loss import (
    backward_opsd_example,
    pointwise_clipped_forward_kl,
    response_features,
)
from postraining.opsd.prepare_dapo import (
    canonical_dapo_record,
    deduplicate_dapo,
    deranged_answer_donors,
    nonderangeable_bucket_keys,
    require_fresh_outputs,
)
from postraining.opsd.schemas import OPSD_EXPORT_SCHEMA
from postraining.opsd.trainer import (
    OPSDTrainer,
    _export_payload,
    _purge_jsonl_after,
    grade_on_policy_response,
    infer_source_contract,
    opsd_tensorboard_metrics,
    opsd_eval_tensorboard_metrics,
    opsd_stop_ids,
    paired_prompt_accuracy_delta,
    validate_authorization,
    validate_data_manifest,
)
from postraining.opsd.teacher_uplift import (
    gate_decision,
    repeated_ngram_fraction,
    summarize_attempts,
    terminal_loop,
)
from postraining.opsd.teacher_logit_gate import (
    auc,
    build_self_rationalized_prompt_arms,
    clipped_observed_update,
    extract_fixed_think_prefix,
    numeric_equivalent_mask,
    response_features_batch,
    response_features_varied_batch,
    response_regions,
)
from postraining.prepare_sft_traces import (
    INSTRUCTION_SUFFIX,
    INSTRUCTION_SUFFIX_ANSWER,
)


class _Tokenizer:
    def bos_id(self) -> int:
        return 7

    def eos_id(self) -> int:
        return 7

    def encode(self, text: str) -> list[int]:
        return [10 + byte % 31 for byte in text.encode()]


def test_forward_kl_matches_definition_and_teacher_is_detached():
    student = torch.tensor([[0.2, -0.3, 0.7]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.1, -0.4]], requires_grad=True)
    metrics = pointwise_clipped_forward_kl(
        student, teacher, pointwise_clip=None
    )
    teacher_logp = teacher.detach().log_softmax(-1)
    expected = (
        teacher_logp.exp() * (teacher_logp - student.log_softmax(-1))
    ).sum(-1)
    assert torch.allclose(metrics.clipped_token_loss, expected)
    metrics.clipped_token_loss.sum().backward()
    assert student.grad is not None
    assert teacher.grad is None
    assert float(metrics.forward_kl.detach()) >= 0


def test_pointwise_clip_happens_before_vocabulary_sum():
    student = torch.tensor([[-5.0, -5.0, 10.0]])
    teacher = torch.tensor([[5.0, 5.0, -10.0]])
    unclipped = pointwise_clipped_forward_kl(
        student, teacher, pointwise_clip=None
    )
    clipped = pointwise_clipped_forward_kl(
        student, teacher, pointwise_clip=0.05
    )
    student_logp = student.log_softmax(-1)
    teacher_logp = teacher.log_softmax(-1)
    contributions = teacher_logp.exp() * (teacher_logp - student_logp)
    expected = contributions.clamp(max=0.05).sum(-1)
    assert torch.allclose(clipped.clipped_token_loss, expected)
    assert clipped.clipped_entries == (contributions > 0.05).sum()
    assert not torch.allclose(
        clipped.clipped_token_loss,
        unclipped.forward_kl.clamp(max=0.05),
    )


class _ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(17, 8)
        self.proj = nn.Linear(16, 17, bias=False)

    def embed_tokens(self, ids):
        return self.embed(ids)

    def temporal_belief_from_token_latent(self, latent):
        return latent.cumsum(dim=1)

    def logits_from_features(self, features):
        return self.proj(features)


def _toy_gradient(chunk: int):
    torch.manual_seed(4)
    student = _ToyBackbone()
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        teacher.proj.weight.add_(0.1 * torch.randn_like(teacher.proj.weight))
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    metrics = backward_opsd_example(
        student,
        teacher,
        torch.tensor([1, 2, 3]),
        torch.tensor([1, 5, 6, 2, 3]),
        torch.tensor([4, 8, 9, 7]),
        batch_denominator=2,
        temperature=1.1,
        pointwise_clip=0.05,
        logit_chunk_tokens=chunk,
    )
    return metrics, [parameter.grad.clone() for parameter in student.parameters()]


def test_chunked_backward_is_objective_and_gradient_invariant():
    one, one_grad = _toy_gradient(1)
    all_at_once, all_grad = _toy_gradient(32)
    assert one["loss"] == pytest.approx(all_at_once["loss"], abs=1e-7)
    assert one["forward_kl"] == pytest.approx(
        all_at_once["forward_kl"], abs=1e-7
    )
    for left, right in zip(one_grad, all_grad, strict=True):
        assert torch.allclose(left, right, atol=2e-7, rtol=2e-6)


def test_logit_gate_batched_response_features_match_training_alignment():
    torch.manual_seed(5)
    model = _ToyBackbone()
    prompt = [1, 2, 3]
    responses = [[4, 5, 6], [7, 8]]
    batched, lengths = response_features_batch(
        model, prompt, responses, torch.device("cpu")
    )
    expected = torch.cat(
        [
            response_features(
                model,
                torch.tensor(prompt),
                torch.tensor(response),
            )
            for response in responses
        ]
    )
    assert lengths == [3, 2]
    assert torch.allclose(batched, expected)


def test_logit_gate_varied_prompts_match_training_alignment():
    torch.manual_seed(6)
    model = _ToyBackbone()
    prompts = [[1, 2, 3], [1, 9, 10, 11, 12]]
    responses = [[4, 5, 6], [7, 8]]
    batched, lengths = response_features_varied_batch(
        model, prompts, responses, torch.device("cpu")
    )
    expected = torch.cat(
        [
            response_features(
                model,
                torch.tensor(prompt),
                torch.tensor(response),
            )
            for prompt, response in zip(prompts, responses, strict=True)
        ]
    )
    assert lengths == [3, 2]
    assert torch.allclose(batched, expected)


def test_self_rationalized_context_uses_fixed_clean_position_matched_prefixes():
    attempt = {
        "emitted_token_ids": [10, *range(100, 140), 11, 12, 7, 13, 99],
        "terminated": True,
        "structural_format_ok": True,
    }
    rationale, reason = extract_fixed_think_prefix(
        attempt,
        think_ids=(10, 11),
        forbidden_ids={7, 10, 11, 12, 13, 99},
        rationale_tokens=32,
        max_repeated_4gram_fraction=0.35,
    )
    assert reason is None
    assert rationale == list(range(100, 132))
    prompts = build_self_rationalized_prompt_arms(
        correct_base=[1, 2],
        permuted_base=[1, 3],
        question_rationale=rationale,
        correct_rationale=[token + 100 for token in rationale],
        permuted_rationale=[token + 200 for token in rationale],
        transition_ids=[4, 5],
        response_tokens=9,
        context_tokens=64,
    )
    assert {len(prompt) for prompt in prompts.values()} == {36}
    assert prompts["direct"][2:34] == rationale

    short = {**attempt, "emitted_token_ids": [10, 100, 11, 99]}
    assert extract_fixed_think_prefix(
        short,
        think_ids=(10, 11),
        forbidden_ids={7, 10, 11, 12, 13, 99},
        rationale_tokens=32,
        max_repeated_4gram_fraction=0.35,
    )[1] == "short_think"


def test_teacher_prompt_places_reference_before_independent_solve():
    prompt = "Problem?" + INSTRUCTION_SUFFIX_ANSWER
    teacher = build_teacher_prompt(
        "Problem?", "<think>proof</think><answer>3</answer>"
    )
    assert teacher.startswith("Problem?")
    assert teacher.index("proof") < teacher.index("Do not copy")
    assert teacher.index("Do not copy") < teacher.index("Begin the new solution")
    assert teacher.endswith("Begin the new solution now:")
    assert INSTRUCTION_SUFFIX_ANSWER == ""
    assert prompt == "Problem?"

    from postraining.math_prompt import LEGACY_ANSWER_FENCE_SUFFIX

    with pytest.raises(ValueError, match="bare problem"):
        build_teacher_prompt(
            "Problem?" + LEGACY_ANSWER_FENCE_SUFFIX, "reference"
        )

    plain = build_teacher_prompt(
        "Problem?", "reference", instruction_suffix=INSTRUCTION_SUFFIX
    )
    assert plain.endswith(INSTRUCTION_SUFFIX)
    assert plain.count(INSTRUCTION_SUFFIX) == 1


def test_final_answer_teacher_prompt_does_not_claim_a_solution_trace():
    teacher = build_teacher_prompt("Problem?", "34", "final_answer")
    assert "verified final answer" in teacher.lower()
    assert "reference solution" not in teacher.lower()
    assert "every step" not in teacher.lower()
    assert "34" in teacher


def test_verified_trace_loading_and_context_rejection(tmp_path):
    problem = "What is 1+2?"
    prompt = problem + INSTRUCTION_SUFFIX_ANSWER
    rows = [
        {
            "source": "ok",
            "problem": problem,
            "document": prompt + "<think>1+2=3</think><answer>3</answer>",
            "verified": True,
        },
        {
            "source": "unverified",
            "problem": "Bad?",
            "document": "Bad?" + INSTRUCTION_SUFFIX_ANSWER + "wrong",
            "verified": False,
        },
        {
            "source": "missing-problem",
            "problem": None,
            "document": "unused",
            "verified": True,
        },
    ]
    path = tmp_path / "traces.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    [example] = load_examples(path, answer_fence=True)
    assert example.problem == problem
    assert example.ground_truth is None
    assert example.grading_style is None
    tokenized, reason = tokenize_example(
        example,
        _Tokenizer(),
        max_prompt_length=10_000,
        context_tokens=20_000,
        max_completion_length=100,
    )
    assert reason is None and tokenized is not None
    rejected, reason = tokenize_example(
        example,
        _Tokenizer(),
        max_prompt_length=2,
        context_tokens=20_000,
        max_completion_length=100,
    )
    assert rejected is None and reason == "student_prompt_overflow"


def test_reference_data_without_verification_fails_closed(tmp_path):
    path = tmp_path / "unverified.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"problem": "P", "solution": "S"}]), path
    )
    with pytest.raises(ValueError, match="verified column"):
        load_examples(path, answer_fence=False)


def test_verified_trace_loading_rejects_legacy_prompt_inside_completion(
    tmp_path,
):
    from postraining.math_prompt import LEGACY_ANSWER_FENCE_SUFFIX

    problem = "What is 1+2?"
    path = tmp_path / "legacy-traces.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source": "legacy",
                    "problem": problem,
                    "document": (
                        problem
                        + LEGACY_ANSWER_FENCE_SUFFIX
                        + "<think>1+2=3</think><answer>3</answer>"
                    ),
                    "verified": True,
                }
            ]
        ),
        path,
    )
    with pytest.raises(ValueError, match="completion boundary"):
        load_examples(path, answer_fence=True)


def test_explicit_reference_column_selects_true_or_permuted_answer(tmp_path):
    path = tmp_path / "dapo.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "problem": "P?",
                    "solution": "3",
                    "permuted_solution": "8",
                    "reference_kind": "final_answer",
                    "ground_truth": "3",
                    "reward_style": "rule-lighteval/MATH_v2",
                    "verified": True,
                }
            ]
        ),
        path,
    )
    [correct] = load_examples(
        path, answer_fence=True, reference_column="solution"
    )
    [control] = load_examples(
        path, answer_fence=True, reference_column="permuted_solution"
    )
    assert correct.reference_solution == "3"
    assert control.reference_solution == "8"
    assert correct.reference_kind == control.reference_kind == "final_answer"
    assert correct.ground_truth == control.ground_truth == "3"
    assert correct.grading_style == control.grading_style == "minerva"


def test_auto_reference_resolves_to_controlled_solution_arm(tmp_path):
    path = tmp_path / "dapo.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"problem": "P?", "solution": "3", "verified": True}]
        ),
        path,
    )
    assert resolve_reference_column(path, "auto") == "solution"


def _dapo_row(example_id: str, problem: str, truth: str) -> dict:
    return {
        "data_source": "math_dapo",
        "prompt": [
            {
                "content": (
                    "Solve the following math problem step by step. The last "
                    "line of your response should be of the form Answer: "
                    "$Answer (without quotes) where $Answer is the answer "
                    f"to the problem.\n\n{problem}\n\nRemember to put your "
                    "answer on its own line after \"Answer:\"."
                )
            }
        ],
        "ability": "MATH",
        "reward_model": {
            "ground_truth": truth,
            "style": "rule-lighteval/MATH_v2",
        },
        "extra_info": {"index": example_id},
    }


def test_dapo_deduplication_merges_content_and_quarantines_conflicts(tmp_path):
    path = tmp_path / "dapo.parquet"
    row = _dapo_row("a", "What is 1+2?", "3")
    duplicate_id = _dapo_row("b", "What is 1+2?", "3")
    pq.write_table(pa.Table.from_pylist([row, row, duplicate_id]), path)
    unique, physical = deduplicate_dapo(path)
    assert physical == 3
    assert [record["extra_info"]["index"] for record in unique] == ["a"]
    assert unique[0]["reward_model"]["ground_truth"] == "3"

    conflicting = copy.deepcopy(row)
    conflicting["reward_model"]["ground_truth"] = "4"
    survivor = _dapo_row("c", "What is 2+3?", "5")
    pq.write_table(pa.Table.from_pylist([row, conflicting, survivor]), path)
    unique, physical = deduplicate_dapo(path)
    assert unique == [survivor]
    assert physical == 3

    pq.write_table(
        pa.Table.from_pylist([row, _dapo_row("a", "Different problem?", "3")]), path
    )
    with pytest.raises(ValueError, match="ambiguous DAPO source id"):
        deduplicate_dapo(path)


def test_dapo_canonicalization_and_answer_derangement():
    records = [
        canonical_dapo_record(_dapo_row(f"id-{i}", f"P{i}?", truth))
        for i, truth in enumerate(("1", "1", "2", "2", "3", "3"))
    ]
    donors = deranged_answer_donors(records, seed=7)
    assert set(donors) == {record["example_id"] for record in records}
    assert sorted(donor["solution"] for donor in donors.values()) == sorted(
        record["solution"] for record in records
    )
    assert all(
        donors[record["example_id"]]["solution"] != record["solution"]
        for record in records
    )
    assert nonderangeable_bucket_keys(
        [
            {"solution": "1", "bucket": 0},
            {"solution": "1", "bucket": 0},
            {"solution": "2", "bucket": 0},
        ],
        lambda record: record["bucket"],
    ) == {0}
    assert records[0]["problem"] == "P0?"

    bucketed = [
        {**record, "bucket": index // 4}
        for index, record in enumerate(records + records[:2])
    ]
    for index, record in enumerate(bucketed):
        record["example_id"] = f"bucket-{index}"
    bucket_donors = deranged_answer_donors(
        bucketed, seed=11, bucket_key=lambda record: record["bucket"]
    )
    assert all(
        donor["bucket"] == record["bucket"]
        for record in bucketed
        for donor in [bucket_donors[record["example_id"]]]
    )


def test_dapo_outputs_are_immutable(tmp_path):
    paths = (tmp_path / "train.parquet", tmp_path / "gate.parquet")
    require_fresh_outputs(paths)
    paths[1].write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        require_fresh_outputs(paths)


def test_dapo_manifest_binds_train_gate_and_sft_bytes(tmp_path):
    from postraining.core import math_corpus_policy_sha256
    train = tmp_path / "train.parquet"
    gate = tmp_path / "gate.parquet"
    train.write_bytes(b"train")
    gate.write_bytes(b"gate")

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "dapo_opsd_final_answer_privilege/v2",
                "math_corpus_policy_sha256": math_corpus_policy_sha256(),
                "math_corpus_identity": "effective-source-fixture",
                "split_schema": (
                    "sha256_clean_gate_then_split_local_token_length_"
                    "answer_derangement/v3"
                ),
                "teacher_prompt_schema": (
                    "privilege_then_bare_problem_token_completion/v3"
                ),
                "opsd_prompt_schema": (
                    "privilege_then_bare_problem_token_completion/v3"
                ),
                "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
                "train_sha256": sha256(train),
                "gate": str(gate),
                "gate_sha256": sha256(gate),
                "gate_rows": 1,
                "sft_corpus_sha256": "sft-hash",
            }
        )
    )
    args = Namespace(
        dataset=str(train),
        data_manifest=str(manifest),
        reference_column="solution",
        authorization=None,
        allow_failed_authorization=False,
    )
    validated = validate_data_manifest(
        args, {"sft": {"traces_sha256": "sft-hash"}}
    )
    assert validated is not None
    assert validated["train_sha256"] == sha256(train)
    assert validated["gate_sha256"] == sha256(gate)

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    authorization = tmp_path / "authorization.json"
    from postraining.opsd.authorize import (
        AUTHORIZATION_GATE_CONTRACT,
        OPSD_AUTHORIZATION_SCHEMA,
    )

    gate_provenance = {
        "generation_gate": {"path": "generation.json", "sha256": "a" * 64},
        "logit_gate": {"path": "logit.json", "sha256": "b" * 64},
    }

    authorization.write_text(
        json.dumps(
            {
                "schema": OPSD_AUTHORIZATION_SCHEMA,
                "decision": "pass",
                "checkpoint_sha256": sha256(checkpoint),
                "data_manifest_sha256": sha256(manifest),
                "gate_sha256": sha256(gate),
                "gate_contract": AUTHORIZATION_GATE_CONTRACT,
                "input_artifacts": gate_provenance,
            }
        )
    )
    args.checkpoint = str(checkpoint)
    args.authorization = str(authorization)
    authorized = validate_authorization(args, {}, validated)
    assert authorized is not None
    assert authorized["sha256"] == sha256(authorization)
    assert authorized["decision"] == "pass"
    assert not authorized["failed_authorization_override"]

    authorization.write_text(
        json.dumps(
            {
                "schema": OPSD_AUTHORIZATION_SCHEMA,
                "decision": "fail",
                "gate_decisions": {"generation": "fail", "logit": "fail"},
                "checkpoint_sha256": sha256(checkpoint),
                "data_manifest_sha256": sha256(manifest),
                "gate_sha256": sha256(gate),
                "gate_contract": AUTHORIZATION_GATE_CONTRACT,
                "input_artifacts": gate_provenance,
            }
        )
    )
    with pytest.raises(ValueError, match="decision is not pass"):
        validate_authorization(args, {}, validated)
    args.allow_failed_authorization = True
    overridden = validate_authorization(args, {}, validated)
    assert overridden is not None
    assert overridden["decision"] == "fail"
    assert overridden["failed_authorization_override"]
    assert overridden["gate_decisions"] == {
        "generation": "fail",
        "logit": "fail",
    }

    train.write_bytes(b"changed")
    with pytest.raises(ValueError, match="train bytes"):
        validate_data_manifest(
            args, {"sft": {"traces_sha256": "sft-hash"}}
        )

    args.data_manifest = None
    args.authorization = None
    with pytest.raises(ValueError, match="data-manifest"):
        validate_data_manifest(args, {"sft": {}})


def test_uplift_repetition_and_loop_metrics():
    assert repeated_ngram_fraction([1, 2, 3, 4, 1, 2, 3, 4]) > 0
    assert repeated_ngram_fraction([1, 2, 3]) == 0
    assert terminal_loop([9, 8, 9, 8, 9, 8])
    assert not terminal_loop([1, 2, 3, 4, 5, 6])


def test_uplift_donor_diagnostic_resolves_dapo_reward_style(monkeypatch):
    styles = []

    def verify(*args, **kwargs):
        styles.append(args[4])
        return False, "[INVALID]"

    monkeypatch.setattr(
        "postraining.opsd.teacher_uplift.verify_terminated_answer", verify
    )
    attempts = [
        {
            "emitted_token_ids": [9, 8] * 20 + [7],
            "problem_index": 0,
            "structural_format_ok": True,
        }
    ]
    rows = [
        {
            "permuted_solution": "3",
            "reward_style": "rule-lighteval/MATH_v2",
        }
    ]
    summary = summarize_attempts(attempts, rows, None, (7,), (8, 9))
    assert styles == ["minerva"]
    assert summary["terminal_loop_fraction"] == 1.0


def test_uplift_gate_pass_and_uncertain_failure_decisions():
    question = {
        "structural_format_fraction": 0.94,
        "ended_fraction": 0.97,
        "terminal_loop_fraction": 0.0,
        "repeated_4gram_fraction_mean": 0.02,
        "all_samples_identical_prompt_fraction": 0.1,
        "unique_transcript_fraction": 0.9,
    }
    correct = {
        **question,
        "structural_format_fraction": 0.95,
        "ended_fraction": 0.98,
    }
    arms = {
        "question_only": question,
        "correct_answer": correct,
        "permuted_answer": question,
    }
    significant = {
        "mean_delta": 0.02,
        "bootstrap_ci_low": 0.005,
        "permutation_p": 0.01,
    }
    comparisons = {
        "correct_vs_question": significant,
        "correct_vs_permuted": significant,
    }
    assert gate_decision(arms, comparisons) == ("pass", [])
    inconclusive = {
        **significant,
        "bootstrap_ci_low": -0.001,
        "permutation_p": 0.08,
    }
    decision, _ = gate_decision(
        arms,
        {
            "correct_vs_question": inconclusive,
            "correct_vs_permuted": inconclusive,
        },
    )
    assert decision == "fail"


def test_opsd_policy_classification_requires_causal_and_absolute_uplift():
    baseline = {
        "structural_format_fraction": 0.95,
        "ended_fraction": 0.98,
        "terminal_loop_fraction": 0.0,
        "repeated_4gram_fraction_mean": 0.02,
        "all_samples_identical_prompt_fraction": 0.0,
        "unique_transcript_fraction": 0.9,
    }
    arms = {
        "baseline": baseline,
        "correct": baseline,
        "permuted": baseline,
    }
    significant = {
        "mean_delta": 0.02,
        "bootstrap_ci_low": 0.005,
        "bootstrap_ci_high": 0.04,
        "permutation_p": 0.01,
    }
    comparisons = {
        "correct_vs_baseline": significant,
        "correct_vs_permuted": significant,
    }
    assert classify(arms, comparisons) == ("beneficial", [])

    comparisons["correct_vs_permuted"] = {
        **significant,
        "mean_delta": 0.0,
        "bootstrap_ci_low": -0.01,
        "permutation_p": 1.0,
    }
    decision, reasons = classify(arms, comparisons)
    assert decision == "neutral_or_inconclusive"
    assert reasons


def test_teacher_logit_gate_auc_and_regions_exclude_answer_leakage():
    assert auc([2.0], [1.0]) == 1.0
    assert auc([1.0], [1.0]) == 0.5
    tokens = [10, 20, 1, 2, 3, 4, 5, 6, 21, 30, 7, 31]
    regions = response_regions(tokens, (10, 21), (30, 31), {(3, 4)})
    assert regions["answer"] == [10]
    assert 3 not in regions["prethink"]
    assert 4 not in regions["prethink"]
    assert len(regions["prethink"]) < len(regions["think_q1"] + regions["think_q2"] + regions["think_q3"])

    class NumericTokenizer:
        def decode(self, token_ids):
            return "".join({40: "1,", 41: "000"}[token] for token in token_ids)

    numeric_mask = numeric_equivalent_mask(
        [40, 41], NumericTokenizer(), {parse_numeric_answer("1000")}, halo=0
    )
    assert numeric_mask == [True, True]


def test_clipped_observed_update_matches_autograd():
    temperature = 1.1
    clip = 0.05
    student_logits = torch.tensor([[0.2, -0.4, 0.7]], requires_grad=True)
    teacher_logits = torch.tensor([[1.0, 0.1, -0.3]])
    student_logp = (student_logits / temperature).log_softmax(-1)
    teacher_logp = (teacher_logits / temperature).log_softmax(-1)
    contributions = teacher_logp.exp() * (teacher_logp - student_logp)
    loss = contributions.clamp(max=clip).sum()
    loss.backward()
    expected = -student_logits.grad[0, 2]
    observed, _ = clipped_observed_update(
        student_logp.detach(),
        teacher_logp,
        torch.tensor([2]),
        temperature=temperature,
        pointwise_clip=clip,
    )
    assert observed[0] == pytest.approx(float(expected), abs=1e-7)


def test_resume_cleanup_discards_only_a_torn_final_jsonl_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1}\n{"step": 2}\n{"step":')
    assert _purge_jsonl_after(path, 1) == 2
    assert path.read_text() == '{"step": 1}\n'

    path.write_text('{"step":\n{"step": 2}\n')
    with pytest.raises(ValueError, match="interior JSONL corruption"):
        _purge_jsonl_after(path, 1)


def test_opsd_tensorboard_metrics_expose_raw_kl_clipping_and_throughput():
    dashboard = opsd_tensorboard_metrics(
        {
            "train_loss": -0.01,
            "forward_kl": 0.12,
            "clipped_token_fraction": 0.75,
            "gradient_norm": 0.4,
            "response_tokens_per_second": 1000.0,
            "step_time_seconds": 12.0,
            "gpu_memory_allocated_gib": 7.5,
            "gpu_peak_memory_allocated_gib": 15.25,
            "exact_accuracy": 0.125,
            "raw_exact_accuracy": 0.25,
            "structural_format_fraction": 0.75,
            "reward_scoring_seconds": 0.01,
        }
    )
    assert dashboard["loss/clipped_objective"] == -0.01
    assert dashboard["loss/raw_forward_kl"] == 0.12
    assert dashboard["loss/clipped_token_fraction"] == 0.75
    assert dashboard["optimization/preclip_gradient_norm"] == 0.4
    assert dashboard["throughput/response_tokens_per_second"] == 1000.0
    assert dashboard["system/gpu_memory_allocated_gib"] == 7.5
    assert dashboard["system/gpu_peak_memory_allocated_gib"] == 15.25
    assert dashboard["reward/exact_accuracy"] == 0.125
    assert dashboard["reward/raw_exact_accuracy"] == 0.25
    assert dashboard["reward/structural_format_fraction"] == 0.75
    assert dashboard["perf/reward_scoring_seconds"] == 0.01


def test_on_policy_reward_matches_vapo_exact_contract():
    class RewardTokenizer:
        def decode(self, token_ids):
            return "".join("3" if token == 20 else "x" for token in token_ids)

    example = Namespace(ground_truth="3", grading_style="minerva")
    think = (10, 11)
    answer = (12, 13)
    valid = [10, *range(30, 63), 11, 12, 20, 13, 7]
    valid_reward = grade_on_policy_response(
        valid,
        example,
        RewardTokenizer(),
        stop_ids=(7,),
        think_fence_ids=think,
        answer_fence_ids=answer,
        min_think_tokens=33,
    )
    assert valid_reward == {
        "raw_exact": 1,
        "exact": 1,
        "structurally_valid": 1,
    }

    invalid_structure = [12, 20, 13, 7]
    invalid_reward = grade_on_policy_response(
        invalid_structure,
        example,
        RewardTokenizer(),
        stop_ids=(7,),
        think_fence_ids=think,
        answer_fence_ids=answer,
        min_think_tokens=33,
    )
    assert invalid_reward == {
        "raw_exact": 1,
        "exact": 0,
        "structurally_valid": 0,
    }


def test_opsd_stop_ids_deduplicate_aliases_and_preserve_distinct_terminals():
    aliased = Namespace(eos_id=lambda: 7, bos_id=lambda: 7)
    distinct = Namespace(eos_id=lambda: 7, bos_id=lambda: 8)
    assert opsd_stop_ids(aliased) == (7,)
    assert opsd_stop_ids(distinct) == (7, 8)


def test_opsd_eval_tensorboard_metrics_expose_quality_and_step0_delta():
    dashboard = opsd_eval_tensorboard_metrics(
        {
            "accuracy": 0.25,
            "contract_accuracy": 0.20,
            "accuracy_delta_from_step0": 0.03,
            "contract_accuracy_delta_from_step0": 0.02,
            "paired_delta_bootstrap_low": 0.01,
            "paired_delta_bootstrap_high": 0.05,
            "structural_format_fraction": 0.9,
            "ended_fraction": 0.95,
            "prompt_any_correct_fraction": 0.5,
            "prompt_zero_correct_fraction": 0.5,
            "within_group_reward_std": 0.1,
            "emitted_tokens_mean": 300,
            "eval_seconds": 12,
        }
    )
    assert dashboard["eval/accuracy"] == 0.25
    assert dashboard["eval/contract_accuracy"] == 0.20
    assert dashboard["eval/descriptive_accuracy_delta_from_step0"] == 0.03
    assert dashboard["eval/paired_delta_bootstrap_low"] == 0.01
    assert dashboard["eval/termination_fraction"] == 0.95
    assert dashboard["perf/eval_seconds"] == 12


def test_paired_prompt_accuracy_delta_is_aligned_and_deterministic():
    first = paired_prompt_accuracy_delta(
        [2, 0, 4, 1], [1, 0, 2, 1], samples=4, seed=9
    )
    second = paired_prompt_accuracy_delta(
        [2, 0, 4, 1], [1, 0, 2, 1], samples=4, seed=9
    )
    assert first == second
    assert first["paired_delta_mean"] == pytest.approx(3 / 16)
    assert first["paired_delta_bootstrap_low"] <= first["paired_delta_mean"]
    assert first["paired_delta_bootstrap_high"] >= first["paired_delta_mean"]


def test_sampler_cursor_exactly_resumes_seeded_shuffle():
    examples = [Namespace(problem=str(index)) for index in range(8)]
    first = ShuffledExampleSampler(examples, seed=19)
    prefix = [first.next().problem for _ in range(11)]
    resumed = ShuffledExampleSampler(examples, seed=19, cursor=6)
    assert [resumed.next().problem for _ in range(5)] == prefix[6:]


def test_source_quota_sampler_is_exact_shuffled_and_cursor_resumable():
    examples = [
        Namespace(problem=f"{source}-{index}", source=source)
        for source, count in (("deepmind_math", 7), ("gsm8k", 5), ("dapo_math_17k", 3))
        for index in range(count)
    ]
    quotas = {"deepmind_math": 24, "gsm8k": 18, "dapo_math_17k": 6}
    sampler = SourceQuotaSampler(examples, quotas, seed=19)
    first = [sampler.next() for _ in range(96)]
    assert Counter(example.source for example in first[:48]) == quotas
    assert Counter(example.source for example in first[48:]) == quotas
    assert sampler.source_counts(48, 48) == quotas

    resumed = SourceQuotaSampler(examples, quotas, seed=19, cursor=37)
    assert [resumed.next().problem for _ in range(35)] == [
        example.problem for example in first[37:72]
    ]


def test_top_k_sampling_never_leaves_candidate_set():
    logits = torch.tensor([[9.0, 8.0, 7.0, 6.0]])
    generator = torch.Generator().manual_seed(2)
    samples = {
        int(
            top_p_sample(
                logits, 1.0, 1.0, generator=generator, top_k=2
            )
        )
        for _ in range(100)
    }
    assert samples <= {0, 1}


def test_config_defaults_and_source_inference():
    from postraining.math_prompt import ANSWER_FENCE_PROMPT_SCHEMA

    parser = build_arg_parser()
    args = parser.parse_args(["--name", "unit"])
    validate_args(args)
    payload = {
        "sft": {
            "traces": "traces.parquet",
            "args": {"think_tokens": True, "answer_fence": True},
            "answer_fence_document_fraction": 1.0,
            "answer_fence_prompt_schema": ANSWER_FENCE_PROMPT_SCHEMA,
        }
    }
    infer_source_contract(args, payload)
    assert args.dataset == "traces.parquet"
    assert args.think_tokens is True
    assert args.answer_fence is True
    assert args.distillation_temperature == args.temperature
    assert args.rollout_compile is True
    assert args.checkpoint_interval_seconds == 480.0

    mismatch = parser.parse_args(
        ["--name", "unit", "--no-answer-fence"]
    )
    with pytest.raises(ValueError, match="SFT provenance"):
        infer_source_contract(mismatch, payload)

    invalid_name = parser.parse_args(["--name", "../escape"])
    with pytest.raises(ValueError, match="run-directory"):
        validate_args(invalid_name)


def test_export_round_trips_through_project_model_loader(tmp_path):
    torch.manual_seed(8)
    model = NanoGPTBackbone(
        vocab_size=64, num_layers=2, model_dim=32, mlp_hidden=64
    )
    model.model_config = {
        "vocab_size": 64,
        "num_layers": 2,
        "model_dim": 32,
        "mlp_hidden": 64,
    }
    model.architecture = "nanogpt_mini_dense_unit"
    model.train_context_tokens = 128
    args = Namespace(
        pointwise_kl_clip=0.05,
        distillation_temperature=1.1,
        checkpoint="base.pt",
        dataset="data.parquet",
    )
    payload = _export_payload(
        model,
        {"sft": {"schema": "unit"}},
        args,
        step=3,
        source_sha256="a",
        dataset_sha256="b",
    )
    assert payload["opsd"]["schema"] == OPSD_EXPORT_SCHEMA
    assert payload["opsd"]["prompt_schema"].endswith("/v3")
    assert payload["sft"] == {"schema": "unit"}
    path = tmp_path / "opsd.pt"
    torch.save(payload, path)
    loaded = load_model(path, torch.device("cpu"))
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_resume_rejects_missing_or_stale_prompt_schema(tmp_path):
    from postraining.opsd.schemas import (
        OPSD_CHECKPOINT_SCHEMA,
        OPSD_OBJECTIVE_SCHEMA,
    )

    trainer = OPSDTrainer.__new__(OPSDTrainer)
    trainer.args = Namespace(answer_fence=True)
    for index, prompt_schema in enumerate((None, "stale/v1")):
        path = tmp_path / f"resume-{index}.pt"
        torch.save(
            {
                "schema": OPSD_CHECKPOINT_SCHEMA,
                "objective_schema": OPSD_OBJECTIVE_SCHEMA,
                "prompt_schema": prompt_schema,
            },
            path,
        )
        with pytest.raises(ValueError, match="prompt schema"):
            trainer._restore(path)

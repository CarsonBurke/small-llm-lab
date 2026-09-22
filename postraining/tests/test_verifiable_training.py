from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from postraining.core import JsonlLogger
from postraining.vapo.policy import (
    FIRST_THOUGHT,
    STOP_THINKING,
    TOKEN_ACTION,
    TrajectoryRecord,
)
import postraining.train_minicpm_vapo as trainer
from postraining.verifiable_tasks import VERIFIABLE_TASK_SCHEMA


def task_row(domain="Math", index=0, target="42", kind="math"):
    query = f"Question {domain} {index}"
    return {
        "prompt": [
            {"role": "user", "content": query + "\n\nYou have a budget of 9k tokens"}
        ],
        "data_source": "fixture_tasks",
        "reward_model": {"style": VERIFIABLE_TASK_SCHEMA, "ground_truth": target},
        "extra_info": {
            "index": str(uuid5(NAMESPACE_URL, query)),
            "domain": domain,
        },
        "verification_info": {"schema": VERIFIABLE_TASK_SCHEMA, "kind": kind},
    }


class TextTokenizer:
    """Tiny reversible tokenizer; never constructs a model or touches CUDA."""

    bos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def decode(self, ids, *, skip_special_tokens=False):
        text = "".join(chr(token) for token in ids)
        if skip_special_tokens:
            for special in ("<think>", "</think>", "<|im_end|>"):
                text = text.replace(special, "")
        return text

    def apply_chat_template(self, messages, *, enable_thinking, **kwargs):
        text = "<user>" + "".join(message["content"] for message in messages)
        text += "<assistant>" + ("<think>\n" if enable_thinking else "")
        return {"input_ids": self.encode(text)}


def tokens(text):
    return torch.tensor(TextTokenizer().encode(text), dtype=torch.long)


def test_global_full_pass_preserves_retained_domain_proportions(tmp_path):
    source = [task_row("Math", n) for n in range(5)]
    source += [task_row("Knowledge", n) for n in range(2)]
    source += [task_row("Long_Context")]
    duplicate = deepcopy(source[0])
    duplicate["extra_info"]["index"] = str(uuid5(NAMESPACE_URL, "duplicate"))
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(source + [duplicate]), path)
    ordered, audit, _ = trainer.load_training_math_corpus(path, seed=17)
    consumed = [
        row
        for cursor in range(0, 2 * len(ordered), 2)
        for row in trainer.select_corpus_rows(ordered, cursor=cursor, count=2)
    ]
    for full_pass in (consumed[:8], consumed[8:]):
        assert len({row["prompt"][0]["content"] for row in full_pass}) == 8
        assert Counter(trainer.training_row_domain(row) for row in full_pass) == {
            "Math": 5,
            "Knowledge": 2,
            "Long_Context": 1,
        }
    assert consumed[:8] == consumed[8:]
    assert audit["domain_proportions"] == {
        "Math": 5 / 8,
        "Knowledge": 2 / 8,
        "Long_Context": 1 / 8,
    }
    assert trainer.corpus_progress(9, 8) == {
        "corpus_questions_seen": 9,
        "corpus_completed_epochs": 1,
        "corpus_cursor": 1,
        "corpus_epoch_progress": 1 / 8,
    }
    with pytest.raises(ValueError):
        trainer.select_corpus_rows(ordered, cursor=0, count=9)


def test_reward_semantics_change_rejects_resume_without_changing_dataset(
    tmp_path, monkeypatch
):
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([task_row()]), path)
    _, _, identity = trainer.load_training_math_corpus(path, seed=17)
    fingerprint = trainer.file_sha256(path)
    resume = {"data_sha256": fingerprint, "math_corpus_identity": identity}
    monkeypatch.setattr(trainer, "verifiable_reward_identity", lambda: "changed-grader")
    _, _, changed = trainer.load_training_math_corpus(path, seed=17)
    with pytest.raises(ValueError, match="different effective"):
        trainer.validate_resume_dataset(resume, fingerprint, changed)


def test_invalid_task_schema_fails_before_cuda_or_model(tmp_path, monkeypatch):
    row = task_row()
    row["reward_model"]["style"] = "unknown"
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    args = trainer.build_parser().parse_args(
        ["--data", str(path), "--output", str(tmp_path / "output")]
    )
    monkeypatch.setattr(
        trainer.argparse.ArgumentParser, "parse_args", lambda self: args
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA or model construction before schema validation")

    monkeypatch.setattr(trainer.torch, "manual_seed", forbidden)
    monkeypatch.setattr(trainer.VAPOPolicy, "from_family", forbidden)
    with pytest.raises(ValueError):
        trainer.main()


def test_task_prompt_is_complete_and_bound_includes_chat_framing():
    tokenizer = TextTokenizer()
    row = task_row()
    expected = tokenizer.apply_chat_template(row["prompt"], enable_thinking=True)[
        "input_ids"
    ]
    encoded = trainer.encode_math_prompt(
        tokenizer,
        row,
        prompt_tokens=len(expected),
        enable_thinking=True,
        prompt_suffix="must not be appended",
    )
    assert encoded.tolist() == expected
    with pytest.raises(ValueError, match="instead of truncating"):
        trainer.encode_math_prompt(
            tokenizer, row, prompt_tokens=len(expected) - 1, enable_thinking=True
        )
    legacy = deepcopy(row)
    legacy["data_source"] = "legacy"
    legacy["reward_model"]["style"] = "rule"
    legacy["verification_info"] = None
    assert (
        trainer.encode_math_prompt(
            tokenizer,
            legacy,
            prompt_tokens=12,
            enable_thinking=True,
        ).numel()
        == 12
    )
    with pytest.raises(ValueError, match="instead of truncating"):
        trainer.encode_math_prompt(
            tokenizer, legacy, prompt_tokens=12, enable_thinking=True,
            truncate_prompt=False,
        )
    row["extra_info"]["prompt_token_cap"] = len(expected) - 1
    with pytest.raises(ValueError, match="instead of truncating"):
        trainer.encode_math_prompt(
            tokenizer, row, prompt_tokens=6144, enable_thinking=True,
        )


@pytest.mark.parametrize(
    "domain,kind,target,answer",
    [
        ("arbitrary_numeric_label", "math", "42", r"\boxed{42}"),
        ("arbitrary_text_label", "text", "Paris", "Answer: Paris"),
    ],
)
def test_task_score_requires_final_section_but_legacy_keeps_its_contract(
    domain, kind, target, answer
):
    tokenizer = TextTokenizer()
    row = task_row(domain, target=target, kind=kind)
    prompt = tokens("<assistant><think>\n")
    assert not trainer.score_training_response(tokenizer, row, prompt, tokens(answer))[
        1
    ]
    text, correct, _ = trainer.score_training_response(
        tokenizer,
        row,
        prompt,
        tokens("reasoning\n</think>\n" + answer + "<|im_end|>"),
    )
    assert correct
    assert "</think>" in text
    assert not trainer.score_training_response(
        tokenizer,
        row,
        prompt,
        tokens(answer + "\n</think>\nNo final answer."),
    )[1]
    legacy = deepcopy(row)
    legacy["data_source"] = "legacy"
    legacy["reward_model"]["style"] = "rule"
    legacy["verification_info"] = None
    assert trainer.score_training_response(
        tokenizer, legacy, prompt, tokens("Answer: " + target)
    )[1]


def test_unfinished_code_scratchpad_is_not_executed():
    row = task_row(
        "arbitrary_code_label", target="execution_tests", kind="python_stdio"
    )
    row["verification_info"] = {
        "schema": VERIFIABLE_TASK_SCHEMA,
        "kind": "python_stdio",
        "call_type": "std",
        "fn_name": None,
        "inputs": ["hello\n"],
        "outputs": ["hello\n"],
    }
    _, correct, _ = trainer.score_training_response(
        TextTokenizer(),
        row,
        tokens("<assistant><think>\n"),
        tokens("```python\nprint(input())\n```"),
    )
    assert not correct


def test_token_and_latent_collection_share_final_answer_grading():
    tokenizer = TextTokenizer()
    row = task_row()
    prompt = tokens("<assistant><think>\n")
    response = tokens("</think>\nAnswer: 42")
    reasons = []
    records, _ = trainer._build_group_records(
        tokenizer,
        row,
        prompt,
        response.unsqueeze(0),
        torch.zeros(1, response.numel()),
        torch.zeros(1, response.numel()),
        samples_per_prompt=1,
        stop_ids=(0,),
        outcome_reasons=reasons,
    )
    assert records[0].correct
    assert records[0].text.startswith("<think></think>")

    # Latent thoughts are not lexical; the native close still reaches grading.
    # One mock close token expands to </think> in the tokenizer fixture.
    class LatentTokenizer(TextTokenizer):
        def decode(self, ids, **kwargs):
            return super().decode(ids, **kwargs).replace("~", "</think>")

    latent_response = torch.cat((tokens("?~"), tokens("\nAnswer: 42")))
    kinds = torch.tensor(
        [FIRST_THOUGHT, STOP_THINKING] + [TOKEN_ACTION] * (latent_response.numel() - 2),
        dtype=torch.int8,
    )
    generation = SimpleNamespace(
        responses=[latent_response],
        action_kinds=[kinds],
        latent_vectors=[torch.zeros(1, 2)],
        logprobs=[torch.zeros(latent_response.numel())],
        controller_observations=[None],
        capacity_row_steps=latent_response.numel(),
        admission_events=1,
        minimum_active_rows_with_backlog=1,
        prefill_seconds=0.0,
        decode_seconds=1.0,
        decode_steps=1,
    )
    engine = SimpleNamespace(
        thinking_start_token_id=ord(">"),
        samples_per_prompt=1,
        top_k=1,
        top_p=1.0,
        generate_prompt_pool=lambda *args, **kwargs: generation,
    )
    result = trainer.collect_latent_rollouts(
        engine,
        LatentTokenizer(),
        [(row, tokens("<think>"))],
        max_new_tokens=32,
    )
    assert result.records[0].correct
    assert result.group_domains == ("Math",)
    assert result.outcome_reasons == tuple(reasons)


def make_record(correct, length, *, capped=False):
    response = [20] * length
    response[-1] = 20 if capped else 99
    return TrajectoryRecord.from_device(
        token_ids=torch.tensor([10] + response),
        prompt_length=1,
        old_logprobs=torch.zeros(length),
        old_values=torch.zeros(length),
        correct=correct,
        text="answer",
    )


def test_prompt_major_domain_metrics_survive_jsonl_and_tensorboard(tmp_path):
    records = [
        make_record(True, 2),
        replace(make_record(False, 4, capped=True), text="A " * 20),
        make_record(False, 3),
        make_record(False, 5),
        make_record(True, 6),
        make_record(True, 8),
    ]
    result = trainer.RolloutResult(
        records=records,
        generated_tokens=28,
        scheduled_tokens=48,
        elapsed_seconds=1.0,
        sampling_scanned_vocabulary=100,
        sampling_candidate_support=100,
        sampling_full_policy_mass_lower_bound=1.0,
        sampling_conditional_mass_lower_bound=1.0,
        admission_events=1,
        minimum_active_rows_with_backlog=6,
        decoding=trainer.DecodeStats(decode_seconds=1.0),
        group_domains=("Math", "Knowledge", "Math"),
        response_limits=(4, 4, 5, 5, 8, 8),
        outcome_reasons=(
            "correct",
            "unfinished_thinking",
            "incorrect",
            "incorrect",
            "correct",
            "correct",
        ),
    )
    metrics = trainer.rollout_diagnostics(result, samples_per_prompt=2, stop_ids=(99,))
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["domain/Math/attempts"] == 4
    assert metrics["domain/Math/accuracy"] == pytest.approx(0.75)
    assert metrics["domain/Math/positive_groups"] == 2
    assert metrics["domain/Math/mixed_groups"] == 1
    assert metrics["domain/Math/response_length_mean"] == 5
    assert metrics["domain/Math/capped_trajectories"] == 1
    assert metrics["domain/Knowledge/accuracy"] == 0
    assert metrics["domain/Math/outcome/unfinished_thinking"] == 1
    assert metrics["response_limit_mean"] == pytest.approx(34 / 6)
    assert metrics["repetition_max_identical_word_run_mean"] == pytest.approx(25 / 6)
    assert metrics["repetition_max_identical_word_run_max"] == 20
    assert metrics["domain/Math/repetition_max_identical_word_run_mean"] == 23 / 4
    assert metrics["domain/Knowledge/repetition_3gram_fraction_max"] == 0
    path = tmp_path / "metrics.jsonl"
    JsonlLogger(path).log(type="rollout", step=7, **metrics)
    assert json.loads(path.read_text())["domain/Math/accuracy"] == 0.75
    assert json.loads(path.read_text())["repetition_16gram_fraction_max"] == 4 / 5
    writer = SummaryWriter(tmp_path / "tensorboard")
    trainer.tensorboard_scalars(writer, "rollout", metrics, 7)
    trainer.tensorboard_scalars(writer, "value_warmup", metrics, 2)
    writer.close()
    events = EventAccumulator(str(tmp_path / "tensorboard"))
    events.Reload()
    point = events.Scalars("rollout_domains/Math/accuracy")[0]
    assert (point.step, point.value) == (7, 0.75)
    assert events.Scalars("rollout_domains_warmup/Knowledge/attempts")[0].value == 2
    assert events.Scalars("rollout_quality/repetition_max_identical_word_run_max")[0].value == 20
    assert events.Scalars("rollout_domains/Math/repetition_max_identical_word_run_mean")[0].value == 23 / 4
    with pytest.raises(ValueError, match="prompt-major"):
        trainer.rollout_diagnostics(
            replace(result, group_domains=("Math",)),
            samples_per_prompt=2,
            stop_ids=(99,),
        )

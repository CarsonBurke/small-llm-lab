from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from postraining.eval_hf_math import (
    _causal_conv1d_reference,
    _causal_conv1d_update_reference,
    has_terminal_loop,
    last_boxed_answer,
    load_vapo_adapter_for_evaluation,
    normalized_token_ids,
    left_padded_position_ids,
    prepare_prompt_ids,
    relaxed_verify,
    repeated_ngram_fraction,
    resolved_eos_ids,
    summarize_attempts,
    truncate_prompt,
    write_suite_metrics_to_tensorboard,
)


def test_last_boxed_answer_handles_nested_latex_and_uses_last_box() -> None:
    text = r"first \boxed{1} then \boxed{\frac{3}{\sqrt{4}}}"
    assert last_boxed_answer(text) == r"\frac{3}{\sqrt{4}}"
    assert last_boxed_answer(r"\boxed{\frac{3}{4}") is None
    assert last_boxed_answer("no answer") is None


def test_relaxed_verify_prefers_answer_contract_before_boxed_fallback() -> None:
    assert relaxed_verify(r"\[\boxed{540}\]", "540", "aime") == (
        True,
        "540",
        "boxed",
    )
    assert relaxed_verify(
        "Answer: 7\nlater boxed: \\boxed{540}", "540", "aime"
    ) == (False, "7", "answer_field")


def test_truncate_prompt_preserves_bos_and_tail() -> None:
    assert truncate_prompt([1, 2, 3, 4, 5], 4, 1) == [1, 3, 4, 5]
    assert truncate_prompt([2, 3, 4, 5], 3, 1) == [3, 4, 5]
    assert truncate_prompt([1, 2], 4, 1) == [1, 2]
    with pytest.raises(ValueError, match="positive"):
        truncate_prompt([1], 0, 1)


def test_left_padded_positions_restart_at_first_real_token() -> None:
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert left_padded_position_ids(mask).tolist() == [
        [0, 0, 0, 1, 2],
        [0, 1, 2, 3, 4],
    ]


def test_generation_token_diagnostics_detect_repetition() -> None:
    assert repeated_ngram_fraction([1, 2, 3, 4, 1, 2, 3, 4]) == pytest.approx(
        0.2
    )
    assert repeated_ngram_fraction([1, 2], width=4) == 0
    assert has_terminal_loop([8, 9, 8, 9, 8, 9])
    assert has_terminal_loop([1, 2, 3, 4, 5, 6]) is False
    with pytest.raises(ValueError):
        repeated_ngram_fraction([1], width=0)


def test_normalized_token_ids_dedupes_and_accepts_scalar() -> None:
    assert normalized_token_ids(None) == ()
    assert normalized_token_ids(11) == (11,)
    assert normalized_token_ids([11, 228, 11]) == (11, 228)


class _FakeTokenizer:
    eos_token_id = 11
    unk_token_id = 999

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        return 228 if token == "<|im_end|>" else 999

    @staticmethod
    def convert_ids_to_tokens(token: int) -> str:
        return "<|im_end|>" if token == 228 else "<unk>"


def test_resolved_eos_ids_adds_chat_turn_stop_only_for_chat() -> None:
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[11])
    )
    tokenizer = _FakeTokenizer()
    assert resolved_eos_ids(model, tokenizer, "chat") == (11, 228)
    assert resolved_eos_ids(model, tokenizer, "raw") == (11,)


def test_prepare_chat_prompt_accepts_mapping_tokenizer_output() -> None:
    class MappingTokenizer:
        bos_token_id = 17

        @staticmethod
        def apply_chat_template(*args, **kwargs):
            return {"input_ids": [17, 20, 21], "attention_mask": [1, 1, 1]}

    assert prepare_prompt_ids(
        MappingTokenizer(),
        "problem",
        prompt_mode="chat",
        prompt_tokens=8,
    ) == [17, 20, 21]


def test_prepare_chat_prompt_forwards_native_thinking_mode() -> None:
    class ThinkingTokenizer:
        bos_token_id = 17
        received: dict = {}

        @classmethod
        def apply_chat_template(cls, *args, **kwargs):
            cls.received = kwargs
            return [17, 8, 20]

    assert prepare_prompt_ids(
        ThinkingTokenizer(),
        "problem",
        prompt_mode="chat",
        prompt_tokens=8,
        enable_thinking=True,
    ) == [17, 8, 20]
    assert ThinkingTokenizer.received["enable_thinking"] is True


def test_prepare_raw_prompt_rejects_thinking_mode() -> None:
    with pytest.raises(ValueError, match="chat prompt"):
        prepare_prompt_ids(
            SimpleNamespace(bos_token_id=17),
            "problem",
            prompt_mode="raw",
            prompt_tokens=8,
            enable_thinking=True,
        )

def test_vapo_adapter_evaluation_loads_bf16_policy_weights(tmp_path) -> None:
    class TinyPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ):
                setattr(self, name, nn.Linear(4, 4, bias=False))

    from postraining.hf_vapo import (
        LoRAConfig,
        adapter_state_dict,
        inject_lora,
    )

    config = LoRAConfig(rank=2, alpha=4.0)
    source = TinyPolicy()
    inject_lora(source, config)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if name.endswith("lora_b"):
                parameter.fill_(0.25)
    checkpoint = tmp_path / "adapter.pt"
    torch.save(
        {
            "policy": {
                "model_id": "model",
                "revision": "revision",
                "lora_config": {
                    "rank": config.rank,
                    "alpha": config.alpha,
                    "targets": config.targets,
                },
                "adapter": adapter_state_dict(source),
            },
            "step": 7,
            "args": {"thinking": True},
        },
        checkpoint,
    )
    target = TinyPolicy().to(dtype=torch.bfloat16)
    metadata = load_vapo_adapter_for_evaluation(
        target,
        checkpoint,
        model_id="model",
        revision="revision",
    )
    assert metadata["step"] == 7
    assert metadata["thinking"] is True
    adapter_parameters = {
        name: parameter
        for name, parameter in target.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    }
    assert adapter_parameters
    assert all(
        parameter.dtype == torch.bfloat16
        for parameter in adapter_parameters.values()
    )
    assert all(
        torch.all(parameter == torch.tensor(0.25, dtype=torch.bfloat16))
        for name, parameter in adapter_parameters.items()
        if name.endswith("lora_b")
    )


def test_periodic_suite_metrics_append_to_training_tensorboard(tmp_path) -> None:
    write_suite_metrics_to_tensorboard(
        tmp_path,
        5,
        {
            "aime_2024": {
                "contract_accuracy": 0.25,
                "relaxed_accuracy": 0.3,
                "terminated_fraction": 0.8,
                "capped_fraction": 0.2,
            }
        },
    )
    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    accuracy = accumulator.Scalars("aime_2024/accuracy")
    assert [(entry.step, entry.value) for entry in accuracy] == [
        (5, pytest.approx(0.25))
    ]


def test_reference_causal_convolution_updates_state_consistently() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 3, 5)
    weight = torch.randn(3, 4)
    bias = torch.randn(3)
    expected = _causal_conv1d_reference(
        x, weight, bias, activation="silu"
    )
    state = torch.zeros(2, 3, 4)
    outputs = [
        _causal_conv1d_update_reference(
            x[..., index], state, weight, bias, activation="silu"
        )
        for index in range(x.shape[-1])
    ]
    actual = torch.stack(outputs, dim=-1)
    assert torch.allclose(actual, expected)


def test_summarize_attempts_reports_group_and_collapse_metrics() -> None:
    attempts = []
    for problem in range(2):
        for sample in range(2):
            correct = problem == 0 and sample == 0
            attempts.append(
                {
                    "problem_index": problem,
                    "contract_correct": correct,
                    "contract_correct_unterminated": correct,
                    "relaxed_correct": correct,
                    "relaxed_correct_unterminated": correct,
                    "terminated": True,
                    "answer_field_count": 1,
                    "boxed_answer": "4",
                    "emitted_text": r"\boxed{4}" if sample == 0 else "Answer: 4",
                    "terminal_loop": False,
                    "repeated_4gram_fraction": 0.0,
                    "relaxed_prediction": "4",
                    "relaxed_source": "answer_field",
                    "emitted_token_count": 5,
                }
            )
    metrics = summarize_attempts(
        attempts,
        problem_count=2,
        samples_per_problem=2,
        elapsed_seconds=2.0,
        peak_allocated_bytes=10,
        peak_reserved_bytes=20,
    )
    assert metrics["contract_accuracy"] == 0.25
    assert metrics["prompt_any_correct_fraction"] == 0.5
    assert metrics["prompt_mixed_correct_fraction"] == 0.5
    assert metrics["prompt_zero_correct_fraction"] == 0.5
    assert metrics["modal_prediction_fraction"] == 1.0
    assert metrics["generated_tokens_per_second"] == 10.0

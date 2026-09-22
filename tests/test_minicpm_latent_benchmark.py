"""CPU workload-contract regressions; never load a model or access CUDA."""

from types import SimpleNamespace

import pytest
import torch

from scripts import benchmark_minicpm_latent as benchmark


def arguments(*options):
    return benchmark.build_parser().parse_args(["--output", "result.json", *options])


@pytest.mark.parametrize("enabled,warn_only", [(False, False), (True, True)])
def test_deterministic_check_restores_timing_configuration_after_failure(
    monkeypatch, enabled, warn_only
):
    workspace = 1024

    def workspace_size(size=None):
        nonlocal workspace
        if size is not None:
            workspace = size
        return workspace

    monkeypatch.setattr(torch.backends.cuda, "cublas_workspace_size", workspace_size)
    original = torch.are_deterministic_algorithms_enabled()
    original_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        with pytest.raises(RuntimeError, match="qualification failure"):
            with benchmark.deterministic_replay_check():
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                workspace_size(2048)
                raise RuntimeError("qualification failure")
        assert torch.are_deterministic_algorithms_enabled() == enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
        assert workspace == 1024
    finally:
        torch.use_deterministic_algorithms(original, warn_only=original_warn)


def test_stream_work_does_not_count_thoughts_as_lexical_tokens():
    args = arguments(
        "--prompts",
        "2",
        "--samples-per-prompt",
        "4",
        "--physical-batch-size",
        "3",
        "--thought-steps",
        "5",
        "--answer-tokens",
        "7",
    )
    work = benchmark.validate_args(args)
    assert work["logical_batch"] == 8
    assert work["physical_batch"] == 3
    assert work["stream_positions_per_row"] == 13
    assert work["total_stream_positions"] == 104
    assert work["latent_thought_actions"] == 40
    assert work["latent_close_actions"] == 8
    assert work["latent_lexical_tokens"] == 56
    assert work["native_lexical_tokens"] == 104
    assert work["cache_length"] == args.context_tokens + 13


@pytest.mark.parametrize(
    "options, message",
    [
        (("--physical-batch-size", "5"), "physical batch"),
        (
            ("--samples-per-prompt", "5", "--replay-batch-size", "2"),
            "replay batch must divide",
        ),
        (("--warmups", "0"), "warmups must be positive"),
        (("--thought-steps", "0"), "thought_steps must be positive"),
        (("--engines", "host", "host"), "engines must be unique"),
    ],
)
def test_invalid_comparison_configuration_rejected_before_loading(options, message):
    with pytest.raises(ValueError, match=message):
        benchmark.validate_args(arguments(*options))


def generation(*, thoughts=2, answers=3, rows=4, forced=True):
    from postraining.vapo.policy import (
        CONTINUE_THOUGHT,
        FIRST_THOUGHT,
        FORCED_STOP_THINKING,
        STOP_THINKING,
        TOKEN_ACTION,
    )

    kinds = torch.tensor(
        [FIRST_THOUGHT]
        + [CONTINUE_THOUGHT] * (thoughts - 1)
        + [FORCED_STOP_THINKING if forced else STOP_THINKING]
        + [TOKEN_ACTION] * answers,
        dtype=torch.int8,
    )
    return SimpleNamespace(
        responses=tuple(torch.arange(kinds.numel()) for _ in range(rows)),
        action_kinds=tuple(kinds.clone() for _ in range(rows)),
        latent_vectors=tuple(
            torch.arange(thoughts * 4, dtype=torch.float32).reshape(thoughts, 4) / 7
            for _ in range(rows)
        ),
        logprobs=tuple(torch.zeros(kinds.numel()) for _ in range(rows)),
    )


def test_records_sidecar_cannot_overwrite_the_source_checkpoint():
    args = arguments("--checkpoint", "result.records.pt")
    with pytest.raises(ValueError, match="overwrite its checkpoint"):
        benchmark.validate_args(args)


def test_nonfinite_temperature_cannot_enter_comparison():
    with pytest.raises(ValueError, match="sampling configuration"):
        benchmark.validate_args(arguments("--temperature", "inf"))


def test_natural_short_rollout_cannot_enter_equal_work_measurement():
    args = arguments("--thought-steps", "2", "--answer-tokens", "3")
    with pytest.raises(RuntimeError, match="natural early termination"):
        benchmark.validate_generation(
            generation(answers=2), "optimized", args, benchmark.validate_args(args)
        )


def test_same_length_wrong_phase_mix_is_not_equal_latent_work():
    args = arguments("--thought-steps", "2", "--answer-tokens", "3")
    with pytest.raises(RuntimeError, match="fixed latent workload"):
        benchmark.validate_generation(
            generation(thoughts=1, answers=4),
            "host",
            args,
            benchmark.validate_args(args),
        )


def test_raw_action_quantization_is_rejected():
    args = arguments("--thought-steps", "2", "--answer-tokens", "3")
    result = generation()
    result.latent_vectors = tuple(row.bfloat16() for row in result.latent_vectors)
    with pytest.raises(RuntimeError, match="precision violated"):
        benchmark.validate_generation(
            result, "optimized", args, benchmark.validate_args(args)
        )


def test_natural_records_preserve_real_close_and_exact_actions():
    args = arguments()
    result = generation(thoughts=2, forced=False)
    records = benchmark.make_records(result, [torch.tensor([8, 9])], args, "optimized")
    for record, raw in zip(records, result.latent_vectors, strict=True):
        assert record.forced_token_index == -1
        torch.testing.assert_close(record.latent_vectors, raw, atol=0, rtol=0)
        assert record.latent_vectors.data_ptr() != raw.data_ptr()


def test_native_records_are_token_only_not_relabelled_latent():
    args = arguments()
    result = SimpleNamespace(
        responses=tuple(torch.tensor([1, 2, 3]) for _ in range(4)),
        logprobs=tuple(torch.zeros(3) for _ in range(4)),
    )
    records = benchmark.make_records(result, [torch.tensor([8, 9])], args, "native")
    assert all(
        record.action_kinds is None and record.latent_vectors is None
        for record in records
    )
    assert all(record.forced_token_index == -1 for record in records)


def test_fixed_gate_is_restored_even_when_workload_fails():
    policy = SimpleNamespace(thinking_gate=SimpleNamespace(head=torch.nn.Linear(4, 1)))
    before = {
        name: value.clone()
        for name, value in policy.thinking_gate.head.state_dict().items()
    }
    with pytest.raises(RuntimeError, match="failed generation"):
        with benchmark.fixed_gate(policy):
            assert torch.equal(
                policy.thinking_gate.head(torch.ones(2, 4)).sigmoid(), torch.zeros(2, 1)
            )
            raise RuntimeError("failed generation")
    for name, value in policy.thinking_gate.head.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_native_checkpoint_cannot_silently_drop_or_initialize_latent_heads(tmp_path):
    path = tmp_path / "native.pt"
    torch.save({"policy": {"schema": "minicpm5_vapo_adapter/v6"}}, path)
    with pytest.raises(ValueError, match="latent/v1"):
        benchmark.load_policy(arguments("--checkpoint", str(path)))


def test_existing_measurement_is_not_overwritten(tmp_path):
    output = tmp_path / "measurement.json"
    output.write_text('{"status": "earlier evidence"}\n')
    args = arguments("--output", str(output))
    with pytest.raises(ValueError, match="already exists"):
        benchmark.validate_args(args)
    assert output.read_text() == '{"status": "earlier evidence"}\n'


def test_natural_replay_budget_fits_later_longer_responses():
    from postraining.vapo.policy import plan_replay_microbatches

    args = arguments()
    prompts = [torch.tensor([8, 9])]
    short = benchmark.make_records(
        generation(answers=1, forced=False),
        prompts,
        args,
        "optimized",
    )[0]
    long = benchmark.make_records(
        generation(answers=3, forced=False),
        prompts,
        args,
        "optimized",
    )[0]
    records = [short, long]
    options = benchmark.training_options(args, records)
    batches = plan_replay_microbatches(
        records,
        [0, 1],
        token_budget=options["replay_token_budget"],
        max_trajectories=options["replay_max_trajectories"],
    )
    assert sorted(index for batch in batches for index in batch) == [0, 1]

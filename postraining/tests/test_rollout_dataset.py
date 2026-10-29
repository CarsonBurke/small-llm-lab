from __future__ import annotations

import json
from dataclasses import fields

import pytest
import torch

from postraining.latent_rollout import (
    PAD_SLOT,
    LatentRolloutBatch,
    assign_terminal_rewards,
    rollout_continuations,
    split_rollout_groups,
    trim_stream,
)
from postraining.rollout_dataset import (
    ROLLOUT_DATASET_SCHEMA,
    RolloutDatasetWriter,
    dataset_digest,
    group_batch,
    iter_pools,
    prompt_sha256,
    read_manifest,
)
from postraining.rollout_report import RowVerdict
from postraining.tests.test_latent_rollout import _wrapper

STOP = 3
TRIM = 4
SAMPLES = 4


def _row(truth: str, index: int) -> dict:
    return {
        "reward_model": {"ground_truth": truth, "style": "minerva"},
        "extra_info": {"index": index, "module": "add_int"},
        "_rl_source": "drills",
    }


def _scored_groups(prompts: list[list[int]], seed: int = 5) -> list[LatentRolloutBatch]:
    """Rollout-only groups exactly as ``finish_group`` sees them."""
    width = max(len(prompt) for prompt in prompts)
    prompt_ids = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        prompt_ids[row, width - len(prompt) :] = torch.tensor(prompt)
    lengths = torch.tensor([len(prompt) for prompt in prompts])
    batch = rollout_continuations(
        _wrapper(),
        prompt_ids,
        max_new_tokens=9,
        max_stream_steps=9,
        temperature=1.0,
        top_p=1.0,
        generator=torch.Generator().manual_seed(seed),
        stop_ids=(STOP,),
        prompt_lengths=lengths,
        prompt_repeats=SAMPLES,
        pin_emit=True,
    )
    groups = split_rollout_groups(batch, SAMPLES, lengths.repeat_interleave(SAMPLES))
    scored = []
    for number, group in enumerate(groups):
        group = trim_stream(group, multiple=TRIM)
        rewards = torch.tensor(
            [float((number + sample) % 2) for sample in range(SAMPLES)]
        )
        assign_terminal_rewards(group, rewards)
        scored.append(group)
    return scored


def _verdicts(batch: LatentRolloutBatch) -> list[RowVerdict]:
    return [RowVerdict(format_ok=row % 2 == 0, parsed_answer=str(row)) for row in range(batch.kind.size(0))]


def _record(directory, pools: list[list[list[int]]]) -> list[list[LatentRolloutBatch]]:
    writer = RolloutDatasetWriter(directory, {"run": "test"}, (STOP,), planned_pools=len(pools))
    recorded = []
    for pool, prompts in enumerate(pools):
        groups = _scored_groups(prompts, seed=pool)
        for number, group in enumerate(groups):
            writer.write(group, _verdicts(group), _row(str(number), number), f"prompt {pool}/{number}")
        writer.finish_pool()
        recorded.append(groups)
    return recorded


def test_rebuilt_groups_equal_the_rollout_field_for_field(tmp_path):
    # Unequal prompt lengths: the left-padded multi-group rollout the
    # trainer actually runs, split back into per-group streams.
    prompts = [[5, 9, 11, 2, 7], [6, 1, 8], [12, 4, 4, 30, 1, 9, 2]]
    recorded = _record(tmp_path / "data", [prompts])
    (pool,) = list(iter_pools(tmp_path / "data"))
    assert len(pool) == len(prompts)
    for original, group, prompt in zip(recorded[0], pool, prompts, strict=True):
        assert group.prompt_tokens == tuple(prompt)
        rebuilt = group_batch(group.prompt_tokens, group.completions, group.rewards, trim_multiple=TRIM)
        for field in fields(LatentRolloutBatch):
            if field.name in {"replay_layout_token", "think_gate_zeroed_correct", "verifier_status", "source_id"}:
                continue
            expected = getattr(original, field.name)
            actual = getattr(rebuilt, field.name)
            if field.name.startswith("old_"):
                # Rollout-only collection never refreshes behavior statistics.
                assert actual.shape == expected.shape
                assert not actual.any()
                continue
            if isinstance(expected, torch.Tensor):
                assert actual.dtype == expected.dtype, field.name
                assert torch.equal(actual, expected), field.name
            else:
                assert actual == expected, field.name


def test_records_carry_verdicts_termination_and_identity(tmp_path):
    prompts = [[5, 9, 11, 2], [6, 1, 8, 8]]
    recorded = _record(tmp_path / "data", [prompts])
    (pool,) = list(iter_pools(tmp_path / "data"))
    for number, (original, group) in enumerate(zip(recorded[0], pool, strict=True)):
        assert group.prompt_text == f"prompt 0/{number}"
        assert group.prompt_sha256 == prompt_sha256(group.prompt_text)
        assert group.ground_truth == str(number)
        assert group.prompt_index == str(number)
        assert (group.source, group.module, group.answer_style) == ("drills", "add_int", "minerva")
        assert group.rewards == tuple(original.reward_scalar.tolist())
        assert group.structural_format_ok == tuple(row % 2 == 0 for row in range(SAMPLES))
        assert group.terminated == tuple(tokens[-1] == STOP for tokens in group.completions)
        assert all(STOP not in tokens[:-1] for tokens in group.completions)


def test_manifest_binds_every_pool_and_detects_tampering(tmp_path):
    directory = tmp_path / "data"
    _record(directory, [[[5, 9, 11]], [[6, 1, 8], [2, 2]]])
    manifest = read_manifest(directory)
    assert manifest["schema"] == ROLLOUT_DATASET_SCHEMA
    assert manifest["binding"] == {"run": "test"}
    assert [entry["pool"] for entry in manifest["pools"]] == [0, 1]
    assert [entry["groups"] for entry in manifest["pools"]] == [1, 2]
    assert [entry["trajectories"] for entry in manifest["pools"]] == [SAMPLES, 2 * SAMPLES]
    # Groups number from zero within each pool.
    assert [[group.group for group in pool] for pool in iter_pools(directory)] == [[0], [0, 1]]
    digest = dataset_digest(manifest)
    assert dataset_digest(read_manifest(directory)) == digest
    pool_file = directory / manifest["pools"][1]["file"]
    pool_file.write_bytes(pool_file.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="manifest digest"):
        list(iter_pools(directory))
    assert not list(directory.glob(".*.tmp"))


def test_writer_refuses_reused_directories_and_foreign_streams(tmp_path):
    directory = tmp_path / "data"
    writer = RolloutDatasetWriter(directory, {}, (STOP,), planned_pools=1)
    with pytest.raises(FileExistsError):
        RolloutDatasetWriter(directory, {}, (STOP,), planned_pools=1)
    with pytest.raises(ValueError, match="no trajectories"):
        writer.finish_pool()
    (group,) = _scored_groups([[5, 9, 11]])
    with pytest.raises(ValueError, match="math-verified"):
        writer.write(group, _verdicts(group), {**_row("1", 0), "_verifier_kind": "python_mbpp"}, "p")
    group.carry_injected = True
    with pytest.raises(ValueError, match="token-policy"):
        writer.write(group, _verdicts(group), _row("1", 0), "p")
    with pytest.raises(ValueError, match="stop ids"):
        RolloutDatasetWriter(tmp_path / "other", {}, (), planned_pools=1)
    with pytest.raises(ValueError, match="at least one pool"):
        RolloutDatasetWriter(tmp_path / "none", {}, (STOP,), planned_pools=0)


def test_group_batch_places_terminal_rewards_at_last_actions():
    batch = group_batch([7, 8, 9], [[1, 2], [4, 5, 6, 3]], [0.25, 1.0], trim_multiple=4)
    assert batch.prompt_length == 3
    assert batch.kind.shape == (2, 8)
    assert batch.action_mask.tolist() == [
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ]
    assert batch.rewards[0, 3].item() == 0.25 and batch.rewards[1, 5].item() == 1.0
    assert batch.rewards.sum().item() == 1.25
    assert (batch.kind[0, 5:] == PAD_SLOT).all() and (batch.kind[1, 7:] == PAD_SLOT).all()
    with pytest.raises(ValueError):
        group_batch([7], [[]], [0.0])
    with pytest.raises(ValueError):
        group_batch([7], [[1]], [0.0, 1.0])


def test_manifest_is_valid_json_after_every_pool(tmp_path):
    directory = tmp_path / "data"
    writer = RolloutDatasetWriter(directory, {"k": 1}, (STOP,), planned_pools=2)
    assert json.loads((directory / "manifest.json").read_text())["pools"] == []
    (group,) = _scored_groups([[5, 9, 11]])
    writer.write(group, _verdicts(group), _row("1", 0), "p")
    writer.finish_pool()
    assert len(read_manifest(directory, allow_incomplete=True)["pools"]) == 1


def test_readers_refuse_a_collection_cut_short(tmp_path):
    """A crash or time limit after some pools leaves a valid but partial
    dataset; every consumer reads it through ``read_manifest``."""
    directory = tmp_path / "data"
    writer = RolloutDatasetWriter(directory, {"k": 1}, (STOP,), planned_pools=2)
    (group,) = _scored_groups([[5, 9, 11]])
    writer.write(group, _verdicts(group), _row("1", 0), "p")
    writer.finish_pool()
    with pytest.raises(ValueError, match="1 of 2 planned pools"):
        read_manifest(directory)
    with pytest.raises(ValueError, match="1 of 2 planned pools"):
        list(iter_pools(directory))
    writer.write(group, _verdicts(group), _row("1", 0), "p")
    writer.finish_pool()
    manifest = read_manifest(directory)
    assert manifest["complete"] and len(manifest["pools"]) == 2
    writer.write(group, _verdicts(group), _row("1", 0), "p")
    with pytest.raises(ValueError, match="planned pools"):
        writer.finish_pool()


def _rollout_cli(*extra):
    from postraining.vapo.config import build_arg_parser, validate_args

    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "c", "--output", "o", "--reasoning-mode", "cot",
                              "--rollout-only", *extra])
    validate_args(parser, args)
    return args


def test_a_curriculum_policy_may_be_recorded_but_not_a_branch_or_resume(capsys):
    assert _rollout_cli("--rollout-dataset").rollout_dataset
    # Its snapshot's hash is bound, so readers can tell its policy apart.
    assert _rollout_cli("--rollout-dataset", "--curriculum-init", "s.pt").curriculum_init == "s.pt"
    for flags, message in (
        (("--actor-init", "a.pt"), "refuses --actor-init and --resume"),
        (("--resume", "r.pt"), "refuses --actor-init and --resume"),
        (("--reasoning-mode", "memory"), "requires --reasoning-mode cot"),
    ):
        with pytest.raises(SystemExit):
            _rollout_cli("--rollout-dataset", *flags)
        assert message in capsys.readouterr().err
    from postraining.vapo.config import build_arg_parser, validate_args

    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "c", "--output", "o", "--rollout-dataset"])
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert "requires --rollout-only" in capsys.readouterr().err

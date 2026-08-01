"""The replay tail's host syncs, and the bit-exactness of removing them.

Boolean-mask indexing has a data-dependent output shape, so every separate
``values[mask]`` on CUDA blocks the host until the stream drains -- in the
forward pass and again in the backward. The eager replay tail ran about a
dozen of them per shard. Replacing them with one ``nonzero`` per mask plus
``index_select``/``index_copy_`` is only admissible if it moves no number
at all: the age-0 zero-clip canary requires ``refresh_old_statistics`` and
the update step to stay bit-identical to each other.

Two properties are pinned here, both of which need a real GPU:

- The compaction is bit-exact, including through a bf16 matmul and the
  backward pass, because that is the claim the canary rests on.
- The tail's sync count no longer scales with the shard count. Counting
  syncs per shard rather than in total is what distinguishes the fix from
  a run that happened to be sharded differently.
"""

from __future__ import annotations

import inspect
import os

import pytest
import torch

from postraining.latent_rollout import (
    build_replay_plan,
    compact_slots,
    iter_length_aware_microbatches,
    plan_length_aware_shards,
    refresh_old_statistics,
    scatter_slots,
    slot_index,
)
from postraining.runtime.profiling import SyncDetector
from postraining.train_latent_vapo import update_minibatch

from postraining.tests.test_latent_rollout import (  # noqa: E402
    _bf16_wrapper,
    _critic,
    _optimizers,
    _rollout,
)
from postraining.latent_rollout import assign_terminal_rewards

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="needs CUDA and RUN_CUDA_TESTS=1",
)


def _source_lines(function, needle) -> set[int]:
    """Line numbers inside ``function`` whose text contains ``needle``.

    Found by inspection rather than written down so the assertions below
    survive edits above them, which is the failure mode that made an
    earlier line-numbered sync report unreadable.
    """
    source, first = inspect.getsourcelines(function)
    return {
        first + offset
        for offset, line in enumerate(source)
        if needle in line
    }


def _rollout_sync_lines(sites: dict[str, int]) -> set[int]:
    return {
        int(location.rsplit(":", 1)[1])
        for location in sites
        if location.rsplit(":", 1)[0].endswith("/latent_rollout.py")
    }


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16]
)
def test_slot_compaction_is_bit_exact_on_cuda(dtype):
    device = torch.device("cuda")
    torch.manual_seed(4)
    rows, stream, width = 5, 64, 32
    mask = torch.rand(rows, stream, device=device) < 0.3
    mask[0] = False
    index = slot_index(mask)

    values = torch.randn(rows, stream, width, device=device, dtype=dtype)
    assert torch.equal(compact_slots(values, index), values[mask])

    # Bit-exactness has to survive the consumer, not just the gather: a
    # differently ordered or strided operand would change a bf16 reduction.
    weight = torch.randn(width, 48, device=device, dtype=dtype)
    assert torch.equal(
        values[mask] @ weight, compact_slots(values, index) @ weight
    )

    by_mask = torch.zeros(rows, stream, device=device, dtype=dtype)
    by_mask[mask] = values[mask][:, 0]
    by_index = scatter_slots(
        torch.zeros(rows, stream, device=device, dtype=dtype),
        index,
        compact_slots(values, index)[:, 0],
    )
    assert torch.equal(by_mask, by_index)


def test_slot_compaction_gradients_are_bit_exact_on_cuda():
    device = torch.device("cuda")
    torch.manual_seed(9)
    mask = torch.rand(4, 48, device=device) < 0.4
    index = slot_index(mask)
    weights = torch.randn(4, 48, 16, device=device, dtype=torch.bfloat16)

    grads = []
    for compact in (
        lambda values: values[mask],
        lambda values: compact_slots(values, index),
    ):
        torch.manual_seed(9)
        values = torch.randn(
            4, 48, 16, device=device, dtype=torch.bfloat16
        ).requires_grad_(True)
        (compact(values) * weights[mask]).sum().backward()
        grads.append(values.grad)
    assert torch.equal(grads[0], grads[1])


def _cuda_batch(wrapper, critic, rows=6, seed=7):
    batch = _rollout(wrapper, batch=rows, prompt=5, new_tokens=4, seed=seed)
    assign_terminal_rewards(batch, torch.rand(rows))
    return batch.to(torch.device("cuda")), wrapper.cuda(), critic.cuda()


def _count_syncs(detector: SyncDetector, call):
    before = sum(detector.sites.values())
    detector.enable()
    try:
        call()
    finally:
        detector.disable()
    sites = {
        location: count
        for location, count in detector.sites.items()
        if "/postraining/" in location
    }
    return sum(detector.sites.values()) - before, sites


def _detector() -> SyncDetector:
    detector = SyncDetector(torch.device("cuda"))
    detector.run_controls()
    assert detector.controls_passed, detector.control_detail
    return detector


def test_refresh_syncs_do_not_scale_past_two_per_shard():
    wrapper = _bf16_wrapper()
    critic = _critic()
    batch, wrapper, critic = _cuda_batch(wrapper, critic, rows=8)
    detector = _detector()
    measured = {}
    for max_trajectories in (8, 1):
        shards = len(plan_length_aware_shards(batch, max_trajectories, 1 << 22))
        detector.sites.clear()
        count, _ = _count_syncs(
            detector,
            lambda: refresh_old_statistics(
                wrapper, critic, batch, max_trajectories=max_trajectories
            ),
        )
        measured[shards] = count
    assert len(measured) == 2, measured
    few, many = sorted(measured)
    slope = (measured[many] - measured[few]) / (many - few)
    # The fallback takes one batch-level metadata snapshot; shard count no
    # longer adds any data-dependent host round-trips.
    assert slope <= 1.0, measured


def test_only_the_fallback_metadata_snapshot_blocks_before_replay():
    wrapper = _bf16_wrapper()
    critic = _critic()
    batch, wrapper, critic = _cuda_batch(wrapper, critic, rows=6)
    detector = _detector()
    _, sites = _count_syncs(
        detector,
        lambda: refresh_old_statistics(
            wrapper, critic, batch, max_trajectories=2
        ),
    )
    allowed = _source_lines(build_replay_plan, 'to("cpu")')
    assert len(allowed) == 1, allowed
    assert _rollout_sync_lines(sites) <= allowed, (sites, allowed)


def test_cpu_built_replay_plan_removes_replay_tail_host_syncs():
    wrapper = _bf16_wrapper()
    critic = _critic()
    host_batch = _rollout(
        wrapper, batch=6, prompt=5, new_tokens=4, seed=23
    )
    assign_terminal_rewards(host_batch, torch.rand(6))
    plan = build_replay_plan(host_batch, 2, 1 << 22)
    batch = host_batch.to(torch.device("cuda"))
    plan = plan.to(torch.device("cuda"))
    wrapper = wrapper.cuda()
    critic = critic.cuda()
    detector = _detector()
    _, sites = _count_syncs(
        detector,
        lambda: refresh_old_statistics(
            wrapper,
            critic,
            batch,
            max_trajectories=2,
            replay_plan=plan,
        ),
    )
    assert not _rollout_sync_lines(sites), sites


def test_the_age_zero_canary_holds_on_cuda():
    # The property the whole rewrite is constrained by: refresh and the
    # update step must agree to the last bit, so behavior-age-0 ratios are
    # exactly one and nothing clips.
    wrapper = _bf16_wrapper()
    critic = _critic()
    with torch.no_grad():
        wrapper.backbone.policy_probe.output.weight.normal_(std=0.02)
        wrapper.combiner.carry.weight.normal_(std=0.02)
    host_batch = _rollout(
        wrapper, batch=6, prompt=5, new_tokens=4, seed=7
    )
    assign_terminal_rewards(host_batch, torch.rand(6))
    replay_plan = build_replay_plan(host_batch, 2, 1 << 22)
    batch = host_batch.to(torch.device("cuda"))
    replay_plan = replay_plan.to(torch.device("cuda"))
    wrapper = wrapper.cuda()
    critic = critic.cuda()
    # More than one shard, so the shared planner and the per-shard indices
    # are both exercised.
    assert len(plan_length_aware_shards(batch, 2, 1 << 22)) > 1
    refresh_old_statistics(
        wrapper,
        critic,
        batch,
        max_trajectories=2,
        replay_plan=replay_plan,
    )
    metrics = update_minibatch(
        wrapper,
        critic,
        batch,
        _optimizers(wrapper, critic, learning_rate=1e-4),
        replay_max_trajectories=2,
        replay_plan=replay_plan,
    )
    assert metrics["policy_clip_fraction"] == 0.0
    assert metrics["token_behavior_kl"] == 0.0
    assert metrics["token_abs_log_ratio_max"] == 0.0
    assert metrics["harmful_positive_log_ratio_max"] == 0.0


def test_update_syncs_do_not_scale_past_two_per_shard():
    wrapper = _bf16_wrapper()
    critic = _critic()
    batch, wrapper, critic = _cuda_batch(wrapper, critic, rows=8)
    detector = _detector()
    measured = {}
    for max_trajectories in (8, 1):
        shards = len(plan_length_aware_shards(batch, max_trajectories, 1 << 22))
        refresh_old_statistics(
            wrapper, critic, batch, max_trajectories=max_trajectories
        )
        optimizers = _optimizers(wrapper, critic, learning_rate=1e-5)
        detector.sites.clear()
        count, _ = _count_syncs(
            detector,
            lambda: update_minibatch(
                wrapper,
                critic,
                batch,
                optimizers,
                replay_max_trajectories=max_trajectories,
            ),
        )
        measured[shards] = count
    assert len(measured) == 2, measured
    few, many = sorted(measured)
    slope = (measured[many] - measured[few]) / (many - few)
    assert slope <= 1.0, measured


def test_the_planner_uploads_shard_rows_without_blocking():
    wrapper = _bf16_wrapper()
    critic = _critic()
    batch, wrapper, critic = _cuda_batch(wrapper, critic, rows=8)
    detector = _detector()
    plan = plan_length_aware_shards(batch, 2, 1 << 22)
    detector.sites.clear()
    count, sites = _count_syncs(
        detector, lambda: plan_length_aware_shards(batch, 2, 1 << 22)
    )
    # One transfer of the used lengths, whatever the shard count.
    assert count == 1, sites
    assert len(plan) > 1
    for host_rows, _, rows in plan:
        assert rows.tolist() == host_rows
    yielded = list(iter_length_aware_microbatches(batch, 2, 1 << 22))
    assert [rows for _, _, _, rows in yielded] == [
        host_rows for host_rows, _, _ in plan
    ]

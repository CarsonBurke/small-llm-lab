"""Schedule contracts that do not execute model workloads."""

from collections import Counter

import pytest

from pretraining.nanogpt_mini.nanogpt_mini_full_bandwidth_train import PassSchedule


def test_progressive_schedule_preserves_paper_mixture_and_delayed_feedback():
    schedule = PassSchedule()
    passes = [schedule.passes_at(step, 2000, 1337) for step in range(2000)]
    assert Counter(passes) == {1: 1500, 2: 440, 3: 60}
    assert set(passes[:1000]) == {1}
    assert set(passes[1000:1760]) == {1, 2}
    assert set(passes[1760:]) == {1, 2, 3}
    assert sum(passes) == 2560


def test_schedule_replay_does_not_depend_on_call_order():
    schedule = PassSchedule()
    expected = {step: schedule.passes_at(step, 2000, 19) for step in range(990, 2000)}
    replay = {step: schedule.passes_at(step, 2000, 19) for step in reversed(expected)}
    assert replay == expected
    alternate = [schedule.passes_at(step, 2000, 20) for step in expected]
    assert alternate != list(expected.values())


@pytest.mark.parametrize("step,total", [(-1, 2000), (2000, 2000), (0, 0)])
def test_schedule_rejects_outside_training_horizon(step, total):
    with pytest.raises(ValueError):
        PassSchedule().passes_at(step, total, 1337)

"""Collection bookkeeping tests; no model or GPU workload is executed."""
import torch

from postraining.frozen_critic_rollouts import _BranchArena, reasoning_positions
from postraining.graph_decode import PinnedDecodeArena


def test_reasoning_actions_exclude_fences_and_answer():
    assert reasoning_positions([7, 8, 100, 20, 21, 101, 200, 22], 2, 100, 101) == [2, 3]
    assert reasoning_positions([7, 8, 22], 2, 100, 101) == []


def test_forced_prefix_and_action_arms_use_ordinary_decode_tick(monkeypatch):
    # Construct only the interception state: fake tick records its pending
    # inputs, proving generated prefixes cannot enter prompt-prefill instead.
    arena = object.__new__(_BranchArena)
    arena.device = torch.device('cpu')
    arena.temperature = 1.0
    arena.pending = torch.tensor([9, 9, 4, 4])
    arena.live_rows = torch.arange(4)
    arena.policy_logits = torch.tensor([[0., 1., 2., 3., 4., 5.]]).expand(4, -1)
    observed = []

    def tick(self, rows):
        observed.append(self.pending[:rows].clone())
        self.pending[:rows].copy_(torch.tensor([3, 4, 5, 2])[:rows])

    monkeypatch.setattr(PinnedDecodeArena, '_run', tick)
    arena.configure([1, 2], taken=3, repeats=2)
    for _ in range(4):
        arena._run(4)
    assert observed[0].tolist() == [1, 1, 1, 1]
    assert observed[1].tolist() == [2, 2, 2, 2]
    assert observed[2].tolist() == [3, 3, 5, 5]
    assert observed[3].tolist() == [3, 4, 5, 2]
    torch.testing.assert_close(arena.branch_probabilities,
                               arena.policy_logits[0].softmax(-1))


def test_repeated_action_is_preserved_as_no_contrast(monkeypatch):
    arena = object.__new__(_BranchArena)
    arena.device = torch.device('cpu')
    arena.temperature = 1.0
    arena.pending = torch.tensor([2, 2, 2, 2])
    arena.live_rows = torch.arange(4)
    arena.policy_logits = torch.zeros((4, 3))
    monkeypatch.setattr(PinnedDecodeArena, '_run', lambda self, rows: None)
    arena.configure([], taken=2, repeats=2)
    arena._run(4)
    assert arena.branch_tokens.tolist() == [2, 2]


def test_unclosed_reasoning_never_trains_on_answer_fences_or_stop():
    assert reasoning_positions([7, 100, 20, 200, 33, 0], 1, 100, 101,
                               forbidden_ids=(200, 201, 0)) == [1]
    assert reasoning_positions([7, 100, 20, 0], 1, 100, 101,
                               forbidden_ids=(200, 201, 0)) == [1]


def test_independent_reference_rows_keep_unforced_policy_actions(monkeypatch):
    arena = object.__new__(_BranchArena)
    arena.device = torch.device('cpu')
    arena.temperature = 1.0
    arena.pending = torch.tensor([1, 2, 3, 4, 5, 6])
    arena.live_rows = torch.arange(6)
    arena.policy_logits = torch.zeros((6, 7))
    monkeypatch.setattr(PinnedDecodeArena, '_run', lambda self, rows: None)
    arena.configure([], taken=None, repeats=2)
    arena._run(6)
    assert arena.branch_tokens.tolist() == [1, 3]
    assert arena.pending.tolist() == [1, 1, 3, 3, 5, 6]


def test_distinct_probe_rejects_before_consuming_branch_action(monkeypatch):
    import pytest
    from postraining.frozen_critic_rollouts import UninformativeBranch

    arena = object.__new__(_BranchArena)
    arena.device = torch.device('cpu')
    arena.temperature = 1.0
    arena.pending = torch.tensor([2, 2, 2, 2, 5, 6])
    arena.live_rows = torch.arange(6)
    arena.policy_logits = torch.zeros((6, 7))
    called = []
    monkeypatch.setattr(PinnedDecodeArena, '_run', lambda self, rows: called.append(rows))
    arena.configure([], None, 2, require_distinct=True, forbidden_tokens=(0,))
    with pytest.raises(UninformativeBranch, match='identical_action_pairs'):
        arena._run(6)
    assert not called
    assert arena.prefix_cursor == 0
    # A distinct draw containing a structural token is rejected identically.
    arena.pending[2] = 0
    arena.configure([], None, 2, require_distinct=True, forbidden_tokens=(0,))
    with pytest.raises(UninformativeBranch, match='nonreasoning_action_pairs'):
        arena._run(6)
    assert not called
    # Passing choices still leave the policy reference rows unforced.
    arena.pending.copy_(torch.tensor([1, 2, 3, 4, 5, 6]))
    arena.configure([], None, 2, require_distinct=True, forbidden_tokens=(0,))
    arena._run(6)
    assert called == [6]
    assert arena.pending.tolist() == [1, 1, 3, 3, 5, 6]

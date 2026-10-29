"""Study protocol and split tests; no model execution."""
import pytest

from scripts.run_frozen_critic_study import (
    arm_specs, candidate_splits, group_predictions, parse_args, select_learnable,
)


def row(i, truth='3'):
    return {'prompt': [{'role': 'user', 'content': f'calculate {i}'}],
            'reward_model': {'ground_truth': truth}, 'extra_info': {'index': i}}


def test_prompt_identity_prevents_split_leakage_and_conflicting_labels():
    rows = [row(i) for i in range(24)]
    rows.append({**row(0), 'extra_info': {'index': 99}})
    split = candidate_splits(rows, seed=1, max_candidates=24)
    identities = [{r['prompt_id'] for r in group} for group in split.values()]
    assert sum(map(len, identities)) == 24
    assert not identities[0] & identities[1]
    assert not identities[1] & identities[2]
    with pytest.raises(ValueError, match='ambiguous'):
        candidate_splits([row(1), row(1, truth='4')], seed=1, max_candidates=24)


def test_calibration_uses_binary_draw_rates_with_complete_samples():
    rows = [{'prompt_id': 'a'}, {'prompt_id': 'b'}]
    records = [{'prompt_id': p, 'reward': int(i < count)}
               for p, count in [('a', 8), ('b', 1)] for i in range(16)]
    assert select_learnable(rows, records, samples=16, quota=5) == rows[:1]
    with pytest.raises(ValueError):
        select_learnable(rows, records[:-1], samples=16, quota=5)


def test_factorial_arms_and_explicit_update_budget():
    assert len(arm_specs()) == 8
    required = ['--checkpoint','base.pt','--actor-init','actor.pt','--data','math.parquet',
                '--output','postraining/runs/test','--steps','1000']
    args = parse_args(required)
    assert args.steps == 1000 and args.eval_every == 20
    assert args.rollout_batch_size >= 3 * args.branch_samples
    with pytest.raises(SystemExit):
        parse_args(required[:-1] + ['1001'])


def test_predictions_require_both_actions_and_preserve_centering():
    first = {'prompt_id':'p', 'pair_id':'s', 'arm':0, 'forced_token':1,
             'q':.5,'advantage':.1,'rewards':[0,1],'reference_rewards':[0,1]}
    second = {**first,'arm':1,'forced_token':2,'q':.3,'advantage':-.1}
    joined = group_predictions([first, second])
    assert joined[0]['actions'][1]['advantage'] == -.1
    with pytest.raises(ValueError, match='both'):
        group_predictions([first])
    with pytest.raises(ValueError, match='duplicate'):
        group_predictions([first,first,second])


def test_branch_journal_recovery_only_discards_interrupted_final_append(tmp_path):
    from scripts.run_frozen_critic_study import recover_partial_branches
    path = tmp_path / 'partial.jsonl'
    path.write_text('{"pair_id":"a"}\n{"pair_id":')
    assert recover_partial_branches(path) == [{'pair_id':'a'}]
    assert path.read_text() == '{"pair_id": "a"}\n'
    path.write_text('{"pair_id":"a"}')
    recover_partial_branches(path)
    assert path.read_text().endswith('\n')
    path.write_text('{"broken":\n{"pair_id":"a"}\n')
    with pytest.raises(ValueError, match='corruption'):
        recover_partial_branches(path)

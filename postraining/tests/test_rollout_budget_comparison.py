"""Keep long-budget attribution separate from lenient MC parsing."""

from scripts.compare_rollout_budgets import choice_stats, compare_choice_views, paired_delta, stats


def row(*, correct=False, length=30, format=True, valid=True, content=0, gold=0):
    return dict(correct=correct, length=length, format=format, terminated=True,
                valid_letter=valid, choice_count=4, content_answer=content,
                gold_content=gold, letter='A' if valid else None, module='science_choice_4',
                prefix_hash='identical')


def test_exact_prefix_counts_only_successes_ended_inside_budget():
    rows = {i: r for i, r in enumerate([
        row(correct=True, length=768), row(correct=True, length=769),
        row(correct=False, length=4096), row(correct=False, length=20),
    ])}
    result = stats(rows, 768)
    assert result['correct'] == 2
    assert result['exact_prefix_correct'] == 1
    assert result['correct_completions_longer_than_short_budget'] == 1


def test_lenient_mc_correctness_does_not_enter_strict_contract_excess():
    rows = {('science', 'p', 0): row(format=False),
            ('science', 'q', 0): row(correct=True)}
    result = choice_stats(rows)['science_choice_4']
    assert result['valid_letter_accuracy'] == 1
    assert result['parsed_letter_guess_excess_per_response'] == .75
    assert result['strict_contract_guess_excess_per_response'] == .375


def test_content_consistency_distinguishes_reward_eligibility():
    left = {('science', 'p', 0): row(correct=False, format=False)}
    right = {('science', 'p', 0): row(correct=True)}
    result = compare_choice_views(left, right)
    assert result['both_parsed_correct'] == 1
    assert result['strict_both_correct'] == 0


def test_paired_delta_clusters_samples_by_prompt():
    left = {('s', p, i): row() for p in ('p', 'q') for i in range(16)}
    right = {k: row(correct=k[1] == 'p') for k in left}
    result = paired_delta(left, right)
    assert result['accuracy_delta'] == .5
    assert result['prompt_cluster_standard_error'] == .5
    assert result['matching_short_token_prefixes'] == 32

"""Compare complete frozen gates and the exact short-budget prefix of long runs."""

import argparse
from collections import Counter, defaultdict
import json
import hashlib
import math
from pathlib import Path


def read_run(path, short_budget=768):
    rows = {}
    with (path / 'gate_transcripts.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['source'], str(row['extra_info']['index']), row['sample'])
            if key in rows:
                raise ValueError(f'duplicate trajectory {key}')
            # Retain only small per-trajectory fields, not long generated text.
            info = row['extra_info']
            order = info.get('option_order')
            answer = row['parsed_answer']
            valid = bool(order and isinstance(answer, str) and len(answer) == 1
                         and 'A' <= answer < chr(ord('A') + len(order)))
            rows[key] = {
                'prefix_hash': hashlib.sha256(json.dumps(row['token_ids'][:short_budget]).encode()).hexdigest(),
                'correct': row['reward'] == 1, 'terminated': row['terminated'],
                'format': row['structural_format_ok'], 'length': len(row['token_ids']),
                'module': info.get('module', row['source']),
                'gold_text': row['reward_model']['ground_truth'],
                'choice_count': len(order) if order else None,
                'valid_letter': valid,
                'content_answer': order[ord(answer) - ord('A')] if valid else None,
                'gold_content': order[ord(row['reward_model']['ground_truth']) - ord('A')] if order else None,
                'letter': answer if valid else None,
            }
    groups = defaultdict(set)
    for source, identity, sample in rows:
        groups[source, identity].add(sample)
    if len(groups) != 512 or any(samples != set(range(16)) for samples in groups.values()):
        raise ValueError(f'incomplete panel: {path}')
    return rows


def stats(rows, short_budget):
    values = list(rows.values())
    n = len(values)
    correct = sum(r['correct'] for r in values)
    return {
        'samples': n, 'correct': correct, 'accuracy': correct / n,
        'most_common_correct_gold_strings': dict(Counter(r.get('gold_text', '') for r in values if r['correct']).most_common(10)),
        'truncated': sum(not r['terminated'] for r in values),
        'finished_invalid_format': sum(r['terminated'] and not r['format'] for r in values),
        'mean_tokens': sum(r['length'] for r in values) / n,
        'exact_prefix_correct': sum(r['correct'] and r['length'] <= short_budget for r in values),
        'correct_completions_longer_than_short_budget': sum(r['correct'] and r['length'] > short_budget for r in values),
    }


def paired_delta(left, right):
    if left.keys() != right.keys():
        raise ValueError('panels differ')
    groups = defaultdict(list)
    for key, row in left.items():
        groups[key[:2]].append(int(right[key]['correct']) - int(row['correct']))
    values = [sum(g) / len(g) for g in groups.values()]
    mean = sum(values) / len(values)
    se = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) * (len(values) - 1)))
    return {'matching_short_token_prefixes': sum(left[k]['prefix_hash'] == right[k]['prefix_hash'] for k in left),
            'accuracy_delta': mean, 'prompt_cluster_standard_error': se,
            'approx_95_interval': [mean - 1.96 * se, mean + 1.96 * se]}


def choice_stats(rows):
    modules = defaultdict(list)
    for key, row in rows.items():
        if row['choice_count']:
            modules[row['module']].append((key, row))
    results = {}
    for module, entries in modules.items():
        valid = [(key, r) for key, r in entries if r['valid_letter']]
        correct = sum(r['content_answer'] == r['gold_content'] for _, r in valid)
        # Invalid outputs earn zero in both model and format-matched guesser.
        residuals = defaultdict(list)
        for key, row in entries:
            residuals[key[:2]].append(
                (int(row['content_answer'] == row['gold_content']) - 1 / row['choice_count'])
                if row['valid_letter'] else 0.0)
        rates = [sum(v) / len(v) for v in residuals.values()]
        mean = sum(rates) / len(rates)
        se = math.sqrt(sum((x - mean) ** 2 for x in rates) / (len(rates) * (len(rates) - 1))) if len(rates) > 1 else None
        results[module] = {
            'samples': len(entries), 'valid_letters': len(valid), 'valid_letter_correct': correct,
            'valid_letter_accuracy': correct / len(valid) if valid else None,
            'valid_letter_uniform_chance': sum(1 / r['choice_count'] for _, r in valid) / len(valid) if valid else None,
            'letter_counts': dict(Counter(r['letter'] for _, r in valid)),
            'parsed_letter_guess_excess_per_response': mean,
            'prompt_cluster_standard_error': se,
            'approx_95_interval': [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None,
            'strict_contract_guess_excess_per_response': sum(
                int(r['correct']) - (1 / r['choice_count'] if r['format'] and r['terminated'] and r['valid_letter'] else 0)
                for _, r in entries) / len(entries),
        }
    return results


def compare_choice_views(fixed, shuffled):
    if fixed.keys() != shuffled.keys():
        raise ValueError('choice views differ')
    valid = both_correct = same_content = strict_both_correct = 0
    for key, left in fixed.items():
        right = shuffled[key]
        strict_both_correct += left['correct'] and right['correct']
        if left['gold_content'] != right['gold_content']:
            raise ValueError('option permutation changed gold content')
        if left['valid_letter'] and right['valid_letter']:
            valid += 1
            same_content += left['content_answer'] == right['content_answer']
            both_correct += left['content_answer'] == left['gold_content'] == right['content_answer']
    return {'paired_valid_letters': valid, 'same_content': same_content,
            'both_parsed_correct': both_correct, 'strict_both_correct': strict_both_correct, 'samples': len(fixed),
            'caveat': 'Shared sampling seeds can correlate views; do not use independent-guess 1/K^2 as their null.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-prefix', default='postraining/runs/kda8_readiness_')
    parser.add_argument('--new-prefix', default='postraining/runs/kda8_readiness_long4096_')
    parser.add_argument('--new-run-map', type=Path)
    parser.add_argument('--suffix', default='_20260926')
    parser.add_argument('--sources', nargs='+', default=['deepmind_easy', 'ultradata_math', 'dapo', 'ultradata_code_l3', 'science_mc', 'ultradata_knowledge'])
    parser.add_argument('--short-budget', type=int, default=768)
    parser.add_argument('--shuffled-science', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    new_run_map = json.loads(args.new_run_map.read_text()) if args.new_run_map else None
    if new_run_map is not None and not set(args.sources) <= set(new_run_map):
        parser.error('new-run-map does not cover every requested source')
    result = {'schema': 'rollout_budget_comparison/v1', 'sources': {},
              'limitations': 'Exact prefix counts are conditional on saved long trajectories, not independently generated short answers. See each run context metadata for extrapolation. Panels are RL-pool diagnostics, not held-out generalization.'}
    for source in args.sources:
        old_path = Path(args.old_prefix + source + args.suffix)
        new_path = Path(new_run_map[source] if new_run_map is not None else args.new_prefix + source + args.suffix)
        old = read_run(old_path, args.short_budget)
        new = read_run(new_path, args.short_budget)
        result['sources'][source] = {'old': stats(old, args.short_budget), 'long': stats(new, args.short_budget),
                                     'paired': paired_delta(old, new), 'choice': choice_stats(new)}
        for label, path in [('old_context', old_path), ('new_context', new_path)]:
            manifest = json.loads((path / 'manifest.json').read_text())
            result['sources'][source][label] = {key: manifest.get(key) for key in (
                'context_tokens', 'trained_context_tokens', 'checkpoint_context_limit', 'eval_context_extrapolation')}
            result['sources'][source][label]['checkpoint'] = manifest['args']['checkpoint']
            result['sources'][source][label]['rollout_groups'] = manifest['args']['rollout_groups']
            result['sources'][source][label]['run_path'] = str(path)
        if source == 'science_mc' and args.shuffled_science:
            shuffled = read_run(args.shuffled_science, args.short_budget)
            result['science_option_permutation'] = {
                'stats': stats(shuffled, args.short_budget), 'choice': choice_stats(shuffled),
                'paired_accuracy': paired_delta(new, shuffled),
                'content_consistency': compare_choice_views(new, shuffled)}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

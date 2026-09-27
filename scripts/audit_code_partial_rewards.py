"""Regrade strict saved code completions and trivial controls with fractional rewards.

Run through mlq. No model generation or relaxed answer extraction occurs.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from postraining.core import GPT2BPETokenizer, fenced_answer_text, structural_format_ok


def aggregate(results):
    by_prompt = defaultdict(list)
    for row in results:
        by_prompt[row['id']].append(row['reward'])
    values = [row['reward'] for row in results]
    return {
        'samples': len(results), 'prompts': len(by_prompt),
        'mean_fraction_passed': sum(values) / len(values),
        'full_pass_samples': sum(value == 1 for value in values),
        'full_pass_rate': sum(value == 1 for value in values) / len(values),
        'partial_positive_samples': sum(0 < value < 1 for value in values),
        'any_positive_samples': sum(value > 0 for value in values),
        'prompts_with_positive': sum(any(v > 0 for v in values) for values in by_prompt.values()),
        'prompts_with_mixed_fractional_reward': sum(len(set(values)) > 1 for values in by_prompt.values()),
        'prompts_with_mixed_binary_reward': sum(len({v == 1 for v in values}) > 1 for values in by_prompt.values()),
        'statuses': dict(Counter(row['status'] for row in results)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcripts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--think-min-tokens', type=int, default=33)
    parser.add_argument('--trivial-baselines', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite evidence')
    if args.workers < 1:
        parser.error('--workers must be positive')
    from postraining.vapo.code_reward import python_test_score, python_test_cases, PYTHON_PARTIAL_REWARD_SCHEMA
    import postraining.vapo.code_reward as scorer_module
    scorer_sha256 = hashlib.sha256(Path(scorer_module.__file__).read_bytes()).hexdigest()
    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    think_ids = (tokenizer.think_open_id, tokenizer.think_close_id)
    answer_ids = (tokenizer.answer_open_id, tokenizer.answer_close_id)
    records, tasks, problems, seen = [], [], {}, set()
    digest = hashlib.sha256()
    with args.transcripts.open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            identity = str(row['extra_info']['index'])
            key = identity, row['sample']
            if key in seen:
                raise ValueError(f'duplicate trajectory: {key}')
            seen.add(key)
            verification = row['verification_info']
            if identity in problems and problems[identity] != verification:
                raise ValueError(f'inconsistent tests for {identity}')
            problems[identity] = verification
            tokens = row['token_ids']
            eligible = (bool(tokens) and tokens[-1] == tokenizer.eos_id()
                        and structural_format_ok(tokens, think_ids, answer_ids, args.think_min_tokens))
            if eligible != bool(row['terminated'] and row['structural_format_ok']):
                raise ValueError(f'saved format contract differs for {key}')
            record = {
                'id': identity, 'sample': row['sample'], 'old_binary_reward': row['reward'],
                'tokens': len(tokens), 'format_eligible': eligible, 'passed': 0,
                'total': len(python_test_cases(verification['tests'])[0]), 'reward': 0.0, 'status': 'format_ineligible',
            }
            records.append(record)
            if eligible:
                answer = fenced_answer_text(tokens, tokenizer, answer_ids)
                if answer is None:
                    raise ValueError('eligible response has no answer span')
                tasks.append((len(records) - 1, answer, verification))
    if len(problems) != 512 or len(seen) != 8192 or any(
        {sample for name, sample in seen if name == identity} != set(range(16))
        for identity in problems
    ):
        raise ValueError('expected complete512-prompt by16-sample frozen panel')

    def score(task):
        index, code, verification = task
        result = python_test_score(code, verification)
        return index, {'passed': result.passed, 'total': result.total, 'reward': result.reward, 'status': result.status}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in pool.map(score, tasks):
            records[index].update(result)
    print(json.dumps({'phase': 'saved_completions', **aggregate(records)}), flush=True)
    baselines = {}
    if args.trivial_baselines:
        for label, expression in [('return_none', 'None'), ('return_zero', '0'), ('return_false', 'False'), ('return_empty_list', '[]')]:
            baseline_rows, baseline_tasks = [], []
            for identity, verification in problems.items():
                names = verification['entry_points']
                if not all(isinstance(name, str) and name.isidentifier() for name in names):
                    raise ValueError('invalid entry point identifier')
                code = '\n\n'.join(f'def {name}(*args, **kwargs):\n    return {expression}' for name in names)
                baseline_tasks.append((len(baseline_rows), code, verification))
                baseline_rows.append({'id': identity})
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for index, result in pool.map(score, baseline_tasks):
                    baseline_rows[index].update(result)
            baselines[label] = {'summary': aggregate(baseline_rows), 'results': baseline_rows}
            print(json.dumps({'phase': label, **baselines[label]['summary']}), flush=True)
    result = {
        'schema': 'saved_code_partial_reward_audit/v1',
        'transcripts': str(args.transcripts), 'transcripts_sha256': digest.hexdigest(),
        'reward_schema': PYTHON_PARTIAL_REWARD_SCHEMA,
        'scorer_sha256': scorer_sha256,
        'extraction': 'Only terminated strict special-token think/answer contract; fenced_answer_text fromtoken_ids, no relaxed parsing.',
        'think_min_tokens': args.think_min_tokens,
        'summary': aggregate(records), 'format_eligible': len(tasks),
        'binary_verdict_disagreements': [row for row in records if (row['reward'] == 1) != (row['old_binary_reward'] == 1)],
        'trivial_baselines': baselines,
        'limitations': [
            'RL-pool diagnostic, not held-out generalization. Reuses existing generated text.',
            'Trivial controls bypass generation/format failure and implement every entry point; their scores measure reward exposure, not a model sampling baseline.',
            'Partial credit on a trivial program can be legitimate for some cases; it does not by itself prove a task or verifier is defective.',
        ],
        'results': records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'binary_verdict_disagreements': len(result['binary_verdict_disagreements'])}), flush=True)


if __name__ == '__main__':
    main()

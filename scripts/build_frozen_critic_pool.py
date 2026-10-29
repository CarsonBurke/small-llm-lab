"""Build a reproducible free-response pool; learnability is measured by screening."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postraining.math_prompt import canonicalize_answer_fence_rows

OPERATIONS = ('add', 'sub', 'mul', 'div', 'add3')


def row_for(operation: str, digits: int, operands: tuple[int, ...]) -> dict:
    if operation in {'add', 'mul', 'add3'}:
        operands = tuple(sorted(operands))
    a, b = operands[:2]
    if operation == 'add':
        question, answer = f'What is {a} + {b}?', a + b
    elif operation == 'sub':
        if a == b:
            raise ValueError('identity subtraction is excluded')
        question, answer = f'What is {a} - {b}?', a - b
    elif operation == 'mul':
        question, answer = f'What is {a} * {b}?', a * b
    elif operation == 'div':
        # Difficulty records factor/quotient digits. Dividend can have 2d
        # digits; this construction guarantees an exact integer answer.
        question, answer = f'What is {a*b} / {b}?', a
    elif operation == 'add3':
        c = operands[2]
        question, answer = f'What is {a} + {b} + {c}?', a + b + c
    else:
        raise ValueError(operation)
    if any(value < 2 for value in operands):
        raise ValueError('operands 0 and 1 are excluded')
    module = f'synthetic_{operation}_d{digits}'
    identity = hashlib.sha256(question.encode()).hexdigest()
    return {'prompt': [{'role': 'user', 'content': question}],
            'reward_model': {'ground_truth': str(answer), 'style': 'rule'},
            'extra_info': {'module': module, 'index': identity,
                           'prompt_contract': 'bare', 'pool_source': 'synthetic',
                           'difficulty_digits': digits}}


def round_robin(buckets: dict[str, list[dict]], count: int) -> list[dict]:
    pending = deque((key, deque(values)) for key, values in sorted(buckets.items()) if values)
    result = []
    while pending and len(result) < count:
        key, values = pending.popleft()
        result.append(values.popleft())
        if values:
            pending.append((key, values))
    if len(result) != count:
        raise ValueError(f'only {len(result)} distinct questions available, need {count}')
    return result


def synthetic_rows(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets = {}
    # Each non-exhausted module can absorb its share after small digit-one
    # spaces run out. Bounded attempts make generation termination explicit.
    target = max(1, (count + 14) // 15)
    for digits in range(1, 5):
        lo, hi = max(2, 10**(digits-1)), 10**digits - 1
        for operation in OPERATIONS:
            unique = {}
            arity = 3 if operation == 'add3' else 2
            if digits == 1:
                candidates = (itertools.combinations_with_replacement(range(lo, hi+1), arity)
                              if operation in {'add', 'mul', 'add3'}
                              else itertools.product(range(lo, hi+1), repeat=arity))
            else:
                candidates = (tuple(rng.randint(lo, hi) for _ in range(arity))
                              for _ in range(target * 50))
            for operands in candidates:
                if operation == 'sub' and operands[0] == operands[1]:
                    continue
                row = row_for(operation, digits, tuple(operands))
                unique[row['prompt'][0]['content']] = row
                if digits > 1 and len(unique) >= target:
                    break
            values = list(unique.values())
            rng.shuffle(values)
            buckets[f'{digits}:{operation}'] = values
    return round_robin(buckets, count)


def eligible_module(module: str) -> bool:
    return (module.startswith('arithmetic__') or module.startswith('algebra__linear_')
            or module == 'numbers__round_number')


def build_pool(source: Path, output: Path, count: int = 4096, seed: int = 1337) -> dict:
    source_rows = canonicalize_answer_fence_rows([
        row for row in pq.read_table(source).to_pylist()
        if eligible_module(row['extra_info']['module'])])
    rng = random.Random(seed + 1)
    rng.shuffle(source_rows)
    buckets = defaultdict(list)
    unique = set()
    synthetic = synthetic_rows(count, seed)
    unique.update(row['prompt'][0]['content'] for row in synthetic)
    for row in source_rows:
        question = '\n'.join(part['content'] for part in row['prompt'])
        if question in unique:
            continue
        unique.add(question)
        row['extra_info']['pool_source'] = 'deepmind_easy'
        row['extra_info']['difficulty_digits'] = None
        buckets[row['extra_info']['module']].append(row)
    deepmind = round_robin(buckets, count)
    rows = synthetic + deepmind
    rng.shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)
    manifest = {'schema': 'frozen_critic_free_response_pool/v1',
                'generator_seed': seed, 'rows': len(rows),
                'source': str(source.resolve()),
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'per_module_counts': dict(sorted(Counter(row['extra_info']['module'] for row in rows).items())),
                'per_source_counts': dict(Counter(row['extra_info']['pool_source'] for row in rows)),
                'learnability': 'unmeasured_until_independent_frozen_policy_screen',
                'synthetic_division_difficulty': 'digits of divisor and integer quotient; dividend is their product',
                'synthetic_digit_one': 'finite spaces exhausted; other modules fill remaining quota'}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('postraining/data/reviewed_candidate_20260926_v2/deepmind_easy.parquet'))
    parser.add_argument('--output', type=Path, default=Path('postraining/data/frozen_critic_pool_20261014.parquet'))
    parser.add_argument('--synthetic-count', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()
    print(json.dumps(build_pool(args.source, args.output, args.synthetic_count, args.seed), indent=2))


if __name__ == '__main__':
    main()

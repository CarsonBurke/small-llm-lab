"""Regrade one gold-independent code candidate per excluded response, through mlq."""

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

from postraining.core import GPT2BPETokenizer
from postraining.vapo.code_reward import python_test_result

BLOCK = re.compile(r'```(?:python|py|python3)?[ \t]*\n(.*?)```', re.DOTALL | re.IGNORECASE)


def one_code_field(text):
    if '```' not in text:
        return text.strip()
    blocks = BLOCK.findall(text)
    return blocks[0].strip() if len(blocks) == 1 and text.count('```') == 2 else None


def candidate(row, tokenizer):
    tokens = row['token_ids']
    try:
        start = tokens.index(50259)
        end = tokens.index(50260, start + 1)
    except ValueError:
        # Diagnostic only: a uniquely completed markdown block can occur in
        # unfinished reasoning. Never search multiple candidates for a pass.
        text = row['text']
        return one_code_field(text) if '```' in text else None
    return one_code_field(tokenizer.decode(tokens[start + 1:end]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcripts', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite evidence')
    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    counts = Counter()
    candidates = []
    with args.transcripts.open() as stream:
        for line in stream:
            row = json.loads(line)
            counts['samples'] += 1
            counts['strict_correct'] += row['reward'] == 1
            if row['terminated'] and row['structural_format_ok']:
                continue
            counts['excluded_responses'] += 1
            code = candidate(row, tokenizer)
            if code is None:
                continue
            counts['extracted'] += 1
            try:
                ast.parse(code)
            except (SyntaxError, ValueError, RecursionError):
                continue
            counts['ast_valid'] += 1
            candidates.append({'code': code, 'verification_info': row['verification_info'],
                               'id': row['extra_info']['index'], 'sample': row['sample'],
                               'terminated': row['terminated'], 'length': len(row['token_ids'])})
    def grade(row):
        return {**row, 'sandbox_result': python_test_result(row['code'], row['verification_info'])}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(grade, candidates))
    output = {'counts': dict(counts), 'outcomes': dict(Counter(r['sandbox_result'] for r in results)),
              'passes': [r for r in results if r['sandbox_result'] == 'pass'],
              'limitations': 'One candidate selected without gold; scratchpad code can be provisional. No model generation or reward-policy change.'}
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'passes'}))


if __name__ == '__main__':
    main()

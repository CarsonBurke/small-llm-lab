"""Sandbox conservative extractions from saved gate responses; run through mlq."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from postraining.vapo.code_reward import python_test_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite evidence')
    rows = [json.loads(line) for line in args.input.open()]
    def run(row):
        return {**row, 'sandbox_result': python_test_result(row['code'], row['verification_info'])}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, rows))
    summary = {'candidates': len(results), 'outcomes': dict(Counter(row['sandbox_result'] for row in results)),
               'passes': [row for row in results if row['sandbox_result'] == 'pass'], 'results': results}
    with args.output.open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key:value for key,value in summary.items() if key not in {'results','passes'}}))


if __name__ == '__main__':
    main()

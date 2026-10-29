#!/usr/bin/env python3
"""Launch standalone token-Q MC-DG after successful verified code SFT.

Submit through mlq with an after-success dependency on the SFT job.
The new SFT backbone initializes the actor; the critic starts from scratch.
Its 50 warmup updates consume 3,200 prompts before 950 joint updates;
fresh-backbone initialization does not support the actor-init rewind flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def prepare_rl(root: Path, sft_name: str, output_name: str) -> list[str]:
    for name in (sft_name, output_name):
        if not name or Path(name).name != name or name in {'.', '..'}:
            raise ValueError('run names must be single directory components')
    sft_run = root / 'postraining/runs' / sft_name
    result = json.loads((sft_run / 'result.json').read_text())
    if result['args']['epochs'] != 1 or result['steps'] <= 0:
        raise ValueError('expected a completed one-epoch SFT run')
    checkpoint = sft_run / 'sft_final_model.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    traces = root / f'postraining/data/{sft_name}_traces.parquet'
    traces_manifest = json.loads(traces.with_suffix('.manifest.json').read_text())
    traces_hash = file_sha256(traces)
    if traces_hash != traces_manifest['output_sha256']:
        raise ValueError('SFT corpus bytes changed after preparation')
    import torch
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    metadata = payload['sft']
    if metadata['schema'] != 'sft_trace_train/v1':
        raise ValueError('unexpected SFT checkpoint schema')
    if metadata['traces_sha256'] != traces_hash or metadata['steps'] != result['steps']:
        raise ValueError('completed SFT checkpoint does not match its corpus/result')
    for record in (result, metadata):
        settings = record['args']
        if (settings['epochs'] != 1 or settings.get('allow_unverified', False)
                or not settings['think_tokens'] or not settings['answer_fence']
                or not settings['skip_sampling_gate']):
            raise ValueError('expected one-epoch verified code SFT with trained fences')
    if metadata['answer_fence_document_fraction'] != 1.0:
        raise ValueError('code SFT checkpoint did not train every answer fence')
    del payload
    audit_path = root / traces_manifest['rl_separation_audit']
    if file_sha256(audit_path) != traces_manifest['rl_separation_audit_sha256']:
        raise ValueError('SFT/RL separation audit changed')
    audit = json.loads(audit_path.read_text())
    original_path = root / audit['rl_mixture']
    if file_sha256(original_path) != audit['rl_mixture_sha256']:
        raise ValueError('RL mixture changed since SFT separation checks')
    mixture = json.loads(original_path.read_text())
    if not any(source['name'] == 'ultradata_code_l3' for source in mixture['sources']):
        raise ValueError('code SFT follow-on must include audited code RL tasks')
    for source in mixture['sources']:
        source_path = root / source['path']
        if file_sha256(source_path) != source['sha256']:
            raise ValueError(f"RL source bytes changed: {source['name']}")
        source['path'] = str(source_path.resolve())
    mixture.update(sft_corpus=str(traces.resolve()), sft_corpus_sha256=traces_hash)
    mixture['code_sft_lineage'] = {
        'run': sft_name, 'epochs': 1, 'optimizer_updates': result['steps'],
        'checkpoint_sha256': file_sha256(checkpoint),
        'rl_separation_audit': str(audit_path.resolve()),
        'rl_separation_audit_sha256': file_sha256(audit_path),
        'rl_critic_family': 'standalone_token_q',
        'q_critic_lambda': 1.0,
        'critic_initialization': 'scratch',
        'warmup_prompt_policy': 'consume_first_3200_then_continue',
        'rl_optimizer_update_budget': 1000,
        'planned_combined_optimizer_updates': result['steps'] + 1000,
    }
    output = root / 'postraining/runs' / output_name
    if output.exists():
        raise FileExistsError(f'refusing to reuse RL output: {output}')
    destination = root / f'postraining/data/{output_name}.mixture.manifest.json'
    with destination.open('x') as stream:
        json.dump(mixture, stream, indent=2)
        stream.write('\n')
    return [sys.executable, '-u', '-m', 'postraining.train_latent_vapo',
            '--checkpoint', str(checkpoint.resolve()),
            '--rl-mixture-manifest', str(destination.resolve()),
            '--reasoning-mode', 'memory', '--think-tokens', '--answer-fence',
            '--delightful-policy-gradient', '--python-reward-mode', 'test-fraction',
            '--randomize-choice-options', '--prompt-tokens', '256',
            '--continuation-tokens', '2048', '--bench-max-tokens', '2048',
            '--aime-max-tokens', '2048', '--answer-tokens', '24',
            '--steps', '950', '--value-warmup-steps', '50',
            '--value-warmup-scope', 'critic', '--critic-init', 'scratch',
            '--critic-q-head-hidden', '1024', '--q-critic-lambda', '1.0',
            '--critic-action-head-learning-rate', '1e-4',
            '--prompts-per-rollout', '64', '--prompts-per-minibatch', '64',
            '--samples-per-prompt', '16', '--rollout-groups', '64',
            '--bench-every', '20', '--aime-every', '20', '--bpb-every', '20',
            '--rollout-sample-every', '25', '--zero-reward-stop-pools', '32',
            '--snapshot-every', '50',
            '--checkpoint-interval-seconds', '300', '--seed', '1337',
            '--output', str(output.resolve())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--trainer-root', type=Path, required=True)
    parser.add_argument('--sft-name', required=True)
    parser.add_argument('--output-name', required=True)
    args = parser.parse_args()
    trainer_root = args.trainer_root.resolve()
    if trainer_root != Path(__file__).resolve().parents[1]:
        parser.error('wrapper must execute from the frozen trainer source directory')
    command = prepare_rl(args.root.resolve(), args.sft_name, args.output_name)
    print(json.dumps({'command': command}), flush=True)
    subprocess.run(command, cwd=trainer_root, check=True)


if __name__ == '__main__':
    main()

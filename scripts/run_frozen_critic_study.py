#!/usr/bin/env python3
"""Frozen memory-policy critic study. Model execution must run through mlq.

Calibration labels select prompts only. Fresh critic labels and branch labels
are independent; train/tune/test identities are fixed before calibration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def write_jsonl(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    os.replace(temporary, path)


def read_jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def recover_partial_branches(path):
    """Commit complete records and discard only an interrupted final append."""
    lines = Path(path).read_text().splitlines()
    last = max((i for i, line in enumerate(lines) if line.strip()), default=-1)
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index != last:
                raise ValueError('branch journal corruption before its final record')
            break
        records.append(record)
    # Normalize the final newline too, so an interrupted newline cannot join
    # an otherwise complete record to the next append.
    write_jsonl(Path(path), records)
    return records


def prompt_identity(row):
    # Same text with another dataset index cannot cross a split.
    content = '\n'.join(m['content'].strip() for m in row['prompt'])
    return hashlib.sha256(content.encode()).hexdigest()


def candidate_splits(rows, *, seed, max_candidates):
    """Deterministic prompt-disjoint pools, stratified by source module."""
    groups = defaultdict(list)
    seen = {}
    for row in rows:
        identity = prompt_identity(row)
        truth = json.dumps(row['reward_model']['ground_truth'], sort_keys=True)
        if identity in seen:
            if seen[identity] != truth:
                raise ValueError('ambiguous duplicate prompt rewards')
            continue
        seen[identity] = truth
        info = row.get('extra_info') or {}
        module = str(info.get('module', info.get('task', row.get('data_source', 'math'))))
        groups[module].append({**row, 'prompt_id': identity})
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    ordered = []
    while groups:
        for name in sorted(list(groups)):
            ordered.append(groups[name].pop())
            if not groups[name]:
                del groups[name]
    # Allocate 1/2 fit, 1/4 tune, 1/4 final before seeing any reward.
    splits = {'train': [], 'tune': [], 'test': []}
    for index, row in enumerate(ordered[:max_candidates]):
        split = ('train', 'train', 'tune', 'test')[index % 4]
        splits[split].append({**row, 'split': split})
    return splits


def select_learnable(rows, calibration, *, samples, quota, low=.2, high=.8):
    grouped = defaultdict(list)
    for record in calibration:
        grouped[record['prompt_id']].append(record['reward'])
    selected = []
    for row in rows:
        rewards = grouped.get(row['prompt_id'], [])
        if len(rewards) != samples:
            raise ValueError('calibration sample count mismatch')
        rate = sum(rewards) / samples
        if low <= rate <= high:
            selected.append(row)
    return selected[:quota]


def group_predictions(flat):
    """Join two action records into a state; never infer a missing arm."""
    pairs = defaultdict(dict)
    for record in flat:
        key = (str(record['prompt_id']), str(record['pair_id']))
        arm = int(record['arm'])
        if arm not in (0, 1) or arm in pairs[key]:
            raise ValueError('invalid or duplicate branch action')
        pairs[key][arm] = record
    output = []
    for (prompt, pair), actions in pairs.items():
        if set(actions) != {0, 1}:
            raise ValueError('each branch state must contain both actions')
        if actions[0]['reference_rewards'] != actions[1]['reference_rewards']:
            raise ValueError('both arms must share an independent policy reference')
        output.append({'prompt_id': prompt, 'state_id': pair,
                       'reference_rewards': actions[0]['reference_rewards'], 'actions': [
            {'token_id': actions[i]['forced_token'], 'q': actions[i]['q'],
             'advantage': actions[i]['advantage'], 'rewards': actions[i]['rewards']}
            for i in (0, 1)]})
    return output


def arm_specs():
    return [
        {'name': f"{'gold' if gold else 'blind'}_lr{lr:g}_{weight}",
         'gold_context': gold, 'head_learning_rate': lr, 'loss_weighting': weight}
        for gold, lr, weight in itertools.product((False, True), (2e-5, 1e-4), ('rollout', 'token'))
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--actor-init', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='free-response math only, no multiple choice')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--eval-every', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1401)
    parser.add_argument('--max-candidates', type=int, default=1024)
    parser.add_argument('--calibration-samples', type=int, default=16)
    parser.add_argument('--train-prompts', type=int, default=32)
    parser.add_argument('--tune-prompts', type=int, default=32)
    parser.add_argument('--test-prompts', type=int, default=32)
    parser.add_argument('--samples-per-prompt', type=int, default=16)
    parser.add_argument('--positions-per-prompt', type=int, default=2)
    parser.add_argument('--branch-samples', type=int, default=64)
    parser.add_argument('--prompt-tokens', type=int, default=256)
    parser.add_argument('--continuation-tokens', type=int, default=2048)
    parser.add_argument('--rollout-batch-size', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--top-p', type=float, default=1.)
    parser.add_argument('--reuse-pool-from', type=Path, help='reuse calibration/fresh trajectories only; always collect new branches')
    parser.add_argument('--branch-position-candidates', type=int, default=None)
    parser.add_argument('--distinct-action-pairs', action='store_true')
    parser.add_argument('--branch-seed-offset', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='resume the same protocol and cumulative update budget')
    parser.add_argument('--dry-run', action='store_true', help='validate protocol only; no model')
    args = parser.parse_args(argv)
    for key in ('steps', 'eval_every', 'max_candidates', 'calibration_samples', 'train_prompts', 'tune_prompts', 'test_prompts', 'samples_per_prompt', 'positions_per_prompt', 'branch_samples', 'prompt_tokens', 'continuation_tokens', 'rollout_batch_size'):
        if getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if args.steps > 1000:
        parser.error('this study has an explicit maximum of 1000 updates per arm')
    if args.branch_position_candidates is not None and args.branch_position_candidates < args.positions_per_prompt:
        parser.error('branch candidates must cover the requested positions')
    if args.rollout_batch_size < 3 * args.branch_samples:
        parser.error('rollout batch must fit both forced arms and policy reference: >= 3 * branch-samples')
    if args.temperature != 1 or args.top_p != 1:
        parser.error('this preregistered study requires untruncated temperature-one policy')
    if min(args.tune_prompts, args.test_prompts) < 16:
        parser.error('at least 16 prompt clusters required in tune and test')
    if args.calibration_samples < 16 or args.branch_samples < 64:
        parser.error('calibration requires >=16 draws; branches require >=64 continuations/action')
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    allowed = (ROOT / 'postraining/runs').resolve()
    if not output.is_relative_to(allowed):
        raise ValueError(f'canonical nano study output must be under {allowed}')
    protocol = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(schema='frozen_token_q_study/v1', arms=arm_specs(),
        actor_updates=0, critic_input='independent causal token trunk; optional prepended gold context',
        critic_head='inverse-model-dimension direction readout',
        credit_scope='reasoning tokens strictly before </think>; no answer credit',
        branch_selection='random candidate order, retain first distinct interior-action pairs before observing outcomes' if args.distinct_action_pairs else 'uniform reasoning positions, independent pi action draws',
        branch_reference='K fresh continuations with next action drawn independently from full pi, separate from both forced arms',
        pool_selection='observed calibration success in [0.2,0.8]; separate fresh labels',
        limitations=['local adaptation holdout; base-pretraining exposure possible',
                      'K=64 has worst-case pair difference SE=0.088; small effects may be inconclusive',
                      'association scope is interior reasoning-action pairs conditional on surviving fence filtering'],
        arm_selection='highest tune sign-agreement lower confidence bound, then smallest association p-value; final test only for selected arm',
        quality_gates='lower 95% bounds: centered sign >0.5; prompt-fixed and state-fixed prediction slopes >0; >=16 prompt clusters',
        exploitation='not run until final sign and prediction gates pass')
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    protocol['input_sha256'] = {}
    for path in (args.checkpoint, args.actor_init, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open('rb') as stream:
            protocol['input_sha256'][str(path.resolve())] = hashlib.file_digest(stream, 'sha256').hexdigest()
    source_manifest = ROOT / 'source_manifest.json'
    if source_manifest.exists():
        protocol['source_manifest_sha256'] = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise FileExistsError(f'new output required or --resume: {output}')
        previous = json.loads((output / 'protocol.json').read_text())
        compare = lambda p: {k:v for k,v in p.items() if k not in ('resume','dry_run')}
        if compare(previous) != compare(protocol):
            raise ValueError('resume changed the preregistered protocol')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'protocol.json', protocol)
    if args.reuse_pool_from is not None and not args.resume:
        previous_dir = args.reuse_pool_from.resolve()
        previous_protocol = json.loads((previous_dir / 'protocol.json').read_text())
        matching = ('checkpoint','actor_init','data','seed','max_candidates','calibration_samples',
                    'train_prompts','tune_prompts','test_prompts','samples_per_prompt',
                    'prompt_tokens','continuation_tokens','temperature','top_p','input_sha256')
        if any(previous_protocol.get(key) != protocol.get(key) for key in matching):
            raise ValueError('reused pool does not match the frozen policy/data/sampling protocol')
        receipts = {}
        for split in ('train','tune','test'):
            for kind in ('calibration','rollouts'):
                src = previous_dir / f'{split}_{kind}.jsonl'
                if not src.is_file():
                    raise FileNotFoundError(src)
                shutil.copy2(src, output / src.name)
                receipts[src.name] = hashlib.sha256(src.read_bytes()).hexdigest()
        atomic_json(output / 'pool_reuse.json', {'source':str(previous_dir),
            'source_protocol_sha256':hashlib.sha256((previous_dir / 'protocol.json').read_bytes()).hexdigest(),
            'files':receipts, 'reused_branches':False, 'reused_critic_weights':False})
    cached_pool = args.resume or args.reuse_pool_from is not None
    started = time.time()
    result = {'schema': protocol['schema'], 'status': 'loading', 'started': started, 'actor_updates': 0}
    atomic_json(output / 'result.json', result)
    def progress(stage, **fields):
        result.update(status=stage, elapsed_seconds=time.time()-started, **fields)
        atomic_json(output / 'result.json', result)
        print(json.dumps({'stage': stage, **fields}), flush=True)
    try:
        from postraining.core import load_unique_math_rows
        from postraining.math_prompt import canonicalize_answer_fence_rows
        from postraining.frozen_critic_rollouts import FrozenPolicy
        from postraining.frozen_critic_fit import FitConfig, fit_study_arm
        from postraining.frozen_critic_gates import evaluate_predictions
        import torch
        from torch.utils.tensorboard import SummaryWriter
        torch.set_float32_matmul_precision('high')
        torch.manual_seed(args.seed)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 CUDA required; no CPU model fallback')
        policy = FrozenPolicy(args)
        atomic_json(output / 'policy_provenance.json', policy.provenance)
        pools = candidate_splits(canonicalize_answer_fence_rows(load_unique_math_rows(args.data)), seed=args.seed, max_candidates=args.max_candidates)
        atomic_json(output / 'candidate_ids.json', {k: [r['prompt_id'] for r in v] for k,v in pools.items()})
        selected = {}
        for split_index, (split, rows) in enumerate(pools.items()):
            quota = getattr(args, f'{split}_prompts')
            calibration_file = output / f'{split}_calibration.jsonl'
            calibration = read_jsonl(calibration_file) if cached_pool and calibration_file.exists() else []
            if len(calibration) % args.calibration_samples:
                raise ValueError('partial calibration prompt group')
            processed = rows[:len(calibration)//args.calibration_samples]
            expected = [r['prompt_id'] for r in processed for _ in range(args.calibration_samples)]
            if [r['prompt_id'] for r in calibration] != expected:
                raise ValueError('cached calibration prompt order changed')
            selected[split] = select_learnable(processed, calibration, samples=args.calibration_samples, quota=quota)
            # Stop calibration once quotas are reached, at fixed batch boundaries.
            for start in range(len(processed), len(rows), 16):
                if len(selected[split]) == quota:
                    break
                chunk = rows[start:start+16]
                progress('calibration', split=split, candidates_seen=len(processed), selected=len(selected[split]))
                fresh = policy.screen(chunk, samples=args.calibration_samples, seed=args.seed + 100 + split_index*10000 + start)
                calibration.extend(fresh)
                processed.extend(chunk)
                write_jsonl(output / f'{split}_calibration.jsonl', calibration)
                selected[split] = select_learnable(processed, calibration, samples=args.calibration_samples, quota=quota)
                if len(selected[split]) == quota:
                    break
            if len(selected[split]) < quota:
                progress('unlearnable_pool', failed_split=split, selected_counts={k:len(v) for k,v in selected.items()}, required=quota)
                return 2
        atomic_json(output / 'selected_ids.json', {k: [r['prompt_id'] for r in v] for k,v in selected.items()})
        rollouts, branches = {}, {}
        for index, (split, rows) in enumerate(selected.items()):
            progress('fresh_rollouts', split=split)
            rollout_file = output / f'{split}_rollouts.jsonl'
            rollouts[split] = (read_jsonl(rollout_file) if cached_pool and rollout_file.exists() else
                policy.screen(rows, samples=args.samples_per_prompt, seed=args.seed+100000+index*10000))
            expected = [r['prompt_id'] for r in rows for _ in range(args.samples_per_prompt)]
            if [r['prompt_id'] for r in rollouts[split]] != expected:
                raise ValueError('cached fresh rollout prompt order changed')
            write_jsonl(rollout_file, rollouts[split])
            if split != 'train':
                progress('branch_rollouts', split=split)
                branch_file = output / f'{split}_branches.jsonl'
                partial_file = output / f'{split}_branches_partial.jsonl'
                if args.resume and branch_file.exists():
                    branches[split] = read_jsonl(branch_file)
                else:
                    cached = recover_partial_branches(partial_file) if args.resume and partial_file.exists() else []
                    if not args.resume:
                        partial_file.unlink(missing_ok=True)
                    def save_partial(record):
                        with partial_file.open('a') as stream:
                            stream.write(json.dumps(record, allow_nan=False) + '\n')
                            stream.flush()
                            os.fsync(stream.fileno())
                    policy.progress_callback = save_partial
                    policy.completed_pair_ids = {r['pair_id'] for r in cached}
                    fresh = policy.branches(rollouts[split], positions_per_prompt=args.positions_per_prompt,
                                            K=args.branch_samples, seed=args.seed+200000+args.branch_seed_offset+index*10000)
                    branches[split] = cached + fresh
                    write_jsonl(branch_file, branches[split])
                    atomic_json(output / 'collection_stats.json', policy.collection_stats)
                    policy.progress_callback = None
                    policy.completed_pair_ids = set()
                informative_prompts = {r['prompt_id'] for r in branches[split]
                    if r['actions'][0]['token_id'] != r['actions'][1]['token_id']}
                if len(informative_prompts) < 16:
                    progress('insufficient_action_contrasts', failed_split=split,
                             informative_prompts=len(informative_prompts))
                    return 2
        policy.release_arenas()
        # No branches or calibration rewards are training labels.
        reports = {}
        writer = SummaryWriter(str(output / 'tensorboard'))
        try:
            for spec in arm_specs():
                progress('critic_fit', arm=spec['name'])
                config = FitConfig(steps=args.steps, eval_every=args.eval_every, seed=args.seed,
                    gold_context=spec['gold_context'], head_learning_rate=spec['head_learning_rate'],
                    loss_weighting=spec['loss_weighting'])
                arm_output = output / 'arms' / spec['name']
                arm_result = fit_study_arm(policy.wrapper, rollouts['train'], branches['tune'], arm_output, config)
                tune = evaluate_predictions(group_predictions(arm_result['predictions']), seed=args.seed)
                reports[spec['name']] = {'tune': tune, 'fit': arm_result}
                atomic_json(arm_output / 'tune_gates.json', tune)
                for name, value in tune.items():
                    if isinstance(value, (int,float)) and not isinstance(value,bool):
                        writer.add_scalar(f"{spec['name']}/{name}", value, args.steps)
                atomic_json(output / 'tuning.json', reports)
        finally:
            writer.close()
        # Never use the final test to select among arms or training checkpoints.
        def rank(name):
            tune = reports[name]['tune']
            interval = tune.get('sign_agreement_ci95')
            lower = float('-inf') if not interval or interval[0] is None else interval[0]
            pvalue = tune.get('arm_swap_pvalue_one_sided')
            return (lower, -pvalue if pvalue is not None else float('-inf'))
        winner = max(reports, key=lambda name: (rank(name), name))
        progress('final_test', selected_arm=winner)
        from postraining.frozen_critic_fit import predict_saved_critic
        test_predictions = predict_saved_critic(policy.wrapper, output / 'arms' / winner, branches['test'])
        write_jsonl(output / 'test_predictions.jsonl', test_predictions)
        final = evaluate_predictions(group_predictions(test_predictions), seed=args.seed+1)
        atomic_json(output / 'test_gates.json', final)
        progress('complete', selected_arm=winner, final_gates=final,
                 next_action='frozen-critic exploitation study' if final.get('passed', False) else 'actor remains frozen; improve critic or measurement precision')
        return 0
    except BaseException as error:
        progress('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())

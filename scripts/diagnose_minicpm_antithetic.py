#!/usr/bin/env python3
"""Queue-only, no-learning B64 controller coupling diagnostic.

Usage (the arguments after -- must match the v3 checkpoint):
  mlq submit --name antithetic --max-parallel-runs 1 --cwd "$PWD" -- \
    python scripts/diagnose_minicpm_antithetic.py --output /absolute/new/probe \
    --arm both --seed 1337 -- --output /absolute/new/unused-trainer-output \
    --resume /absolute/checkpoint.pt --bridge /checkpoint/matched/bridge.pt \
    [other checkpoint-matched trainer arguments]

Each requested arm runs exactly ONE real 4-prompt x 16-sample batch, physical64,
then the independent checkpoint critic's real refresh. DiagnosticComplete exits
at controller_update BEFORE any optimizer operation; it is not a successful
training step. No initial/final evaluation, critic warmup, actor update, critic
update, checkpoint save, or hidden-state replay substitute is allowed.

Both arms use counter-based independent noise across response positions/roles.
Independent keys each logical lane; antithetic keys neighboring lanes by lane//2,
negates odd-lane Gaussian noise, and shares gate/answer uniforms. These change
the JOINT law, not the intended marginal Gaussian sigma or categorical law.
They are not bitwise equivalent to stock torch RNG. Finite-precision RNG
qualification is a separate prerequisite; this script does not claim variance
improvement. No refill is supported in this deliberately exact 64/64 scope.
"""
from __future__ import annotations

import argparse
import atexit
from contextlib import ExitStack, contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticComplete(BaseException):
    """Successful diagnostic boundary, explicitly NOT a completed training step."""


@contextmanager
def replace_attribute(target, name, value):
    owned = name in vars(target)
    original = vars(target).get(name)
    setattr(target, name, value)
    try:
        yield
    finally:
        if owned:
            setattr(target, name, original)
        else:
            delattr(target, name)


def atomic_report(path, payload):
    """Publish strict JSON only after the entire report is serialized and synced."""
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, indent=2, allow_nan=False, default=str)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pair_statistics(left, right):
    """Sufficient statistics; no inference of population variance improvement."""
    count = len(left)
    sx, sy = sum(left), sum(right)
    xx, yy = sum(x*x for x in left), sum(y*y for y in right)
    xy = sum(x*y for x, y in zip(left, right, strict=True))
    vx, vy = xx - sx*sx/count, yy - sy*sy/count
    covariance = xy - sx*sy/count
    return {'count': count, 'sum_left': sx, 'sum_right': sy,
            'sum_sq_left': xx, 'sum_sq_right': yy, 'sum_products': xy,
            'correlation': covariance / math.sqrt(vx*vy) if vx > 0 and vy > 0 else None,
            'pair_mean_sample_variance':
                sum(((x+y)/2 - (sx+sy)/(2*count))**2
                    for x, y in zip(left, right, strict=True)) / (count-1)}


def make_diagnostic_engine(mode, seed, state):
    from postraining.minicpm_paired_rollout import PairedControllerRolloutEngine

    class DiagnosticEngine(PairedControllerRolloutEngine):
        def __init__(self, policy, **kwargs):
            required = {'prompts_per_rollout': 4, 'samples_per_prompt': 16,
                        'physical_batch_size': 64, 'cache_length': 11024,
                        'answer_reserve_tokens': 1024, 'temperature': 0.9,
                        'top_k': 20, 'top_p': 0.95}
            if any(kwargs.get(key) != value for key, value in required.items()):
                raise ValueError('diagnostic requires exact B64=4x16/physical64, '
                                 '11024 cache, 1024 reserve, T=.9/top-k20/top-p.95')
            super().__init__(policy, pair_samples=mode == 'antithetic',
                             counter_seed=seed, **kwargs)
            state['engines'].append(self)

        def generate_prompt_pool(self, prompt_ids, max_new_tokens, progress_callback=None):
            if state['generations'] != 0:
                raise ValueError('exactly one generation per diagnostic arm is allowed')
            if len(prompt_ids) != 4 or max_new_tokens != 10000:
                raise ValueError('diagnostic requires four prompts and the unchanged 10000 response budget')
            lengths = [prompt.numel() for prompt in prompt_ids]
            if any(not 1 <= length <= 1024 for length in lengths):
                raise ValueError('prompt length exceeds unchanged 1024-token budget')
            state['generations'] += 1
            started = time.perf_counter()
            result = super().generate_prompt_pool(prompt_ids, max_new_tokens, progress_callback)
            state['generation'] = {
                'wall_seconds': time.perf_counter()-started,
                'prefill_seconds': result.prefill_seconds, 'decode_seconds': result.decode_seconds,
                'window': [started, time.perf_counter()],
                'admission_events': result.admission_events, 'decode_steps': result.decode_steps,
                'prompt_lengths': lengths, 'telemetry': dict(self.telemetry),
            }
            return result

    return DiagnosticEngine


def export_batch(torch, save_analysis, policy, records, args, path, hashes, coupling):
    if len(records) != 64:
        raise ValueError('analysis must contain exactly 64 real trajectories')
    temporary = path.with_suffix('.partial.pt')
    events = []
    recorder = SimpleNamespace(torch=torch, analysis_saved=False,
                               emit=lambda name, **values: events.append({'event': name, **values}))
    try:
        save_analysis(recorder, policy, records, args, temporary, hashes)
        payload = torch.load(temporary, map_location='cpu', weights_only=False)
        payload['coupling'] = coupling
        payload['pair_ids'] = torch.arange(64, dtype=torch.int64)//2
        # The shared compact schema crops observations to genuine controller
        # actions. Keep every original token/kind/score/advantage, including the
        # forced close and frozen answer actions, without giant answer states.
        for index, (compact, record) in enumerate(zip(payload['records'], records, strict=True)):
            compact.update(pair_id=index//2, logical_lane=index, coupling_mode=coupling['mode'],
                           prompt_length=record.prompt_length,
                           forced_token_index=record.forced_token_index, text=record.text,
                           token_ids=record.token_ids.detach().clone(),
                           all_action_kinds=record.action_kinds.detach().clone(),
                           all_old_logprobs=record.old_logprobs.detach().clone(),
                           all_advantages=record.advantages.detach().clone())
        with temporary.open('wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return events
    finally:
        temporary.unlink(missing_ok=True)


def close_trainer_frame(error, trainer):
    """Stock main has no finally around its writer; close on our boundary too."""
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        if frame.f_code is trainer.main.__code__:
            for name in ('metrics_file', 'tensorboard'):
                resource = frame.f_locals.get(name)
                if resource is not None:
                    resource.close()
                    if name == 'tensorboard':
                        atexit.unregister(resource.close)
        trace = trace.tb_next


def run_arm(torch, trainer, rollout, profile, args, mode, seed, path, hashes, provenance):
    from postraining.runtime.profiling import DEVICE_SAMPLE_FIELDS, DeviceSampler

    state = {'engines': [], 'generations': 0, 'critic_refreshes': 0,
             'actor_updates': 0, 'critic_updates': 0, 'optimizer_steps': 0}
    started = time.perf_counter()
    sampler = DeviceSampler(250, torch.device('cuda', torch.cuda.current_device()))
    coupling = {'mode': mode, 'counter_seed': seed,
                'key': 'logical_lane//2' if mode == 'antithetic' else 'logical_lane',
                'pairing': 'neighboring logical lanes within each 16-sample prompt',
                'position': 'decoder.lengths[lane] - original_prompt_length[lane]',
                'roles': {'Gaussian': 1, 'Bernoulli_gate': 2, 'categorical_answer': 3},
                'Gaussian_sign': 'even +1, odd -1' if mode == 'antithetic' else 'all +1',
                'joint_law_changed': mode == 'antithetic',
                'marginal_law': 'intended unchanged; finite-precision RNG qualification is separate',
                'answer_inverse_cdf': 'stock top-k20 then top-p.95, temperature .9; '
                                      'ascending token IDs after filtering; FP64 CDF',
                'answer_score': 'stock untempered full-vocabulary likelihood with delimiters restored',
                'no_learning': True, 'provenance': provenance}
    original_refresh = trainer.refresh_critic

    def refresh(critic, records, token_budget):
        if state['generations'] != 1 or state['critic_refreshes'] != 0 or token_budget != 11024:
            raise ValueError('expected exactly one real post-rollout critic refresh at token budget 11024')
        engine = state['engines'][0]
        if critic is engine.source_policy or critic.value_head is None:
            raise ValueError('the restored independent critic is mandatory')
        before = time.perf_counter()
        result = original_refresh(critic, records, token_budget)
        torch.cuda.synchronize()
        state['critic_refresh_seconds'] = time.perf_counter()-before
        state['critic_refresh_window'] = [before, time.perf_counter()]
        state['critic_refreshes'] += 1
        return result

    def forbidden(*unused_args, **unused_kwargs):
        raise RuntimeError('NO-LEARNING diagnostic forbids optimizer updates or training checkpoint saves')

    def boundary(policy, optimizer, records, **kwargs):
        if state['critic_refreshes'] != 1:
            raise ValueError('controller boundary reached without the real independent critic refresh')
        state['export_events'] = export_batch(torch, profile.save_analysis, policy, records,
                                             args, path/'analysis.pt', hashes, coupling)
        correctness = [bool(record.correct) for record in records]
        thoughts = [record.latent_vectors.size(0) for record in records]
        response_lengths = [record.response_length for record in records]
        stops = state['engines'][0].stop_ids
        termination = ['stop_token' if int(record.token_ids[-1]) in stops else
                       'response_budget' if record.response_length == 10000 else 'unexpected'
                       for record in records]
        if 'unexpected' in termination:
            raise ValueError('a trajectory ended without a stop token or full response budget')
        first_scores = [float(record.old_logprobs[0]) for record in records]
        agreement = sum(correctness[i] == correctness[i+1] for i in range(0, 64, 2))
        state['batch'] = {'trajectories': 64, 'correct': sum(correctness), 'correctness': correctness,
                          'response_lengths': response_lengths, 'thought_lengths': thoughts,
                          'termination': termination,
                          'forced_close': [record.forced_token_index >= 0 for record in records],
                          'pair_reward_agreement_count': agreement,
                          'pair_reward_agreement_fraction': agreement/32,
                          'first_mandatory_Gaussian_logscore_pairs':
                              pair_statistics(first_scores[::2], first_scores[1::2]),
                          'score_statistic_scope': 'first mandatory Gaussian action only; no gate term; '
                                                   'logscore correlation is NOT gradient-estimator variance',
                          'analysis_path': str(path/'analysis.pt')}
        raise DiagnosticComplete('DIAGNOSTIC COMPLETE: one B64 batch and real critic refresh; '
                                 'stopped before controller_update; zero optimizer steps')

    parser = trainer.parser()
    # Parse the exact supplied configuration once, then inject only explicitly
    # mutable execution/output arguments; stock resume validation remains intact.
    parser.parse_args = lambda *unused_args, **unused_kwargs: args
    diagnostic_engine = make_diagnostic_engine(mode, seed, state)
    try:
        sampler.start()
        with ExitStack() as stack:
            for target, name, replacement in (
                (trainer, 'parser', lambda: parser),
                (trainer, 'MiniCPMLatentRolloutEngine', diagnostic_engine),
                (trainer, 'PairedControllerRolloutEngine', diagnostic_engine),
                (trainer, 'refresh_critic', refresh), (trainer, 'controller_update', boundary),
                (trainer, 'update_step', forbidden), (trainer, 'atomic_torch_save', forbidden),
                (torch.optim.AdamW, 'step', forbidden),
            ):
                stack.enter_context(replace_attribute(target, name, replacement))
            try:
                trainer.main()
            except BaseException as error:
                close_trainer_frame(error, trainer)
                if not isinstance(error, DiagnosticComplete):
                    raise
                state['status'] = 'diagnostic_complete_before_controller_update'
                state['boundary'] = str(error)
            else:
                raise RuntimeError('trainer returned without reaching the diagnostic boundary')
    finally:
        for engine in state.pop('engines'):
            engine.release_cache()
        ended = time.perf_counter()
        sampler.stop()
        windows = {'whole_arm': [(started, ended)]}
        if 'generation' in state:
            windows['rollout'] = [state['generation']['window']]
        if 'critic_refresh_window' in state:
            windows['critic_refresh'] = [state['critic_refresh_window']]
        state['device_metrics'] = {
            phase: sampler.window(intervals, 400.0)
            for phase, intervals in windows.items()
        }
        state['device_sampler_error'] = sampler.error
        atomic_report(path/'device_samples.json', {
            'fields': ['perf_counter', *DEVICE_SAMPLE_FIELDS],
            'samples': [[stamp, *values] for stamp, values in sampler.samples],
            'windows': windows, 'power_floor_watts': 400.0, 'error': sampler.error,
        })
        state['wall_seconds'] = time.perf_counter()-started
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True, help='new diagnostic directory')
    parser.add_argument('--arm', choices=('both', 'independent', 'antithetic'), default='both')
    parser.add_argument('--seed', type=int, help='counter seed; default SHA256-derived from restored CUDA RNG')
    parser.add_argument('trainer_args', nargs=argparse.REMAINDER,
                        help='-- followed by checkpoint-matched trainer arguments, including resume and bridge')
    options = parser.parse_args(argv)
    trainer_argv = options.trainer_args[1:] if options.trainer_args[:1] == ['--'] else options.trainer_args
    if not trainer_argv:
        parser.error('checkpoint-matched trainer arguments after -- are required')
    if options.seed is not None and not 0 <= options.seed < 2**63:
        parser.error('counter seed must be in [0, 2**63)')
    options.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'minicpm-antithetic-diagnostic/v1', 'status': 'running',
              'no_learning': True, 'arms': {}, 'trainer_argv': trainer_argv,
              'scope': {'prompts': 4, 'samples_per_prompt': 16, 'physical_batch': 64,
                        'prompt_budget': 1024, 'response_budget': 10000, 'answer_reserve': 1024,
                        'batches_per_arm': 1, 'critic_refreshes_per_arm': 1,
                        'actor_updates': 0, 'critic_updates': 0, 'heldout_evaluations': 0},
              'marginal_reasoning': [
                  'Distinct (seed,lane-or-pair,response-position,role,component) counters provide '
                  'fresh pseudorandom innovations independently of row ordering and head padding.',
                  'For each lane, symmetric standard normal z and -z have the same intended law; '
                  'existing sample_latent uses unchanged mean + exp(log_sigma)*z and sum log likelihood.',
                  'Per-lane gate decision U < sigmoid(logit) preserves its Bernoulli probability. '
                  'Gate, Gaussian, and answer draws have disjoint role domains.',
                  'Inverse CDF of the stock filtered categorical probabilities preserves that '
                  'intended distribution; sharing U correlates pairs without changing either marginal.',
                  'No bitwise stock RNG equivalence or measured variance improvement is asserted; '
                  'finite-precision normal/uniform qualification must run before this workload.',
              ]}
    started = time.perf_counter()
    old_path = list(sys.path)
    old_signal = signal.getsignal(signal.SIGTERM)
    torch = None
    rng = None
    current_arm = None
    precision = None

    def terminate(signum, frame):
        raise SystemExit(128+signum)

    try:
        signal.signal(signal.SIGTERM, terminate)
        sys.path.insert(0, str(ROOT))
        from scripts import profile_minicpm_latent_controller as profile
        report['queue'] = profile.require_mlq_runner()
        import torch
        from scripts import train_minicpm_latent_controller as trainer
        from postraining import minicpm_latent_rollout as rollout
        from postraining.runtime.coupled_rng import rng_metadata
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is mandatory; there is no CPU model fallback')
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), random.getstate())
        precision = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
        args = trainer.parser().parse_args(trainer_argv)
        if args.resume is None or args.bridge is None or args.eval_only:
            raise ValueError('a supplied v3 checkpoint and bridge are mandatory; eval-only is forbidden')
        if (args.prompts_per_rollout, args.samples_per_prompt, args.physical_batch_size) != (4, 16, 64):
            raise ValueError('initial diagnostic scope is exactly B64=4 prompts x16 even samples, physical64')
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['schema'] != 'minicpm-frozen-latent-controller/v3':
            raise ValueError('only restored v3 controller checkpoints are supported')
        if saved['warmups'] < args.warmup_steps:
            raise ValueError('checkpoint critic warmups are unfinished; diagnostic will not run any warmup')
        cuda_rng_bytes = saved['cuda_rng'].numpy().tobytes()
        cuda_rng_sha = hashlib.sha256(cuda_rng_bytes).hexdigest()
        counter_seed = options.seed if options.seed is not None else int(cuda_rng_sha[:16], 16) % 2**63
        provenance = {'checkpoint': str(args.resume.resolve()), 'checkpoint_sha256': sha256(args.resume),
                      'bridge': str(args.bridge.resolve()), 'bridge_sha256': sha256(args.bridge),
                      'checkpoint_step': int(saved['step']), 'checkpoint_cursor': int(saved['cursor']),
                      'checkpoint_warmups': int(saved['warmups']), 'restored_cuda_rng_sha256': cuda_rng_sha,
                      'seed_source': 'explicit counter seed' if options.seed is not None else
                                     'SHA256(restored checkpoint CUDA RNG bytes), first64bits modulo2^63',
                      'counter_seed': counter_seed, 'trainer_seed': args.seed,
                      'rng_implementation': rng_metadata()}
        args.steps = int(saved['step']) + 1
        del saved
        args.skip_evaluation = True
        # The boundary, not a timer, stops the batch; startup/capture is unrestricted.
        args.train_seconds = float('1e12')
        sources = [Path(__file__), Path(trainer.__file__), Path(rollout.__file__), Path(profile.__file__),
                   ROOT/'scripts/minicpm_coupled_rng.py', ROOT/'postraining/runtime/coupled_rng.py',
                   ROOT/'postraining/minicpm_paired_rollout.py', ROOT/'postraining/train_minicpm_vapo.py',
                   ROOT/'postraining/minicpm_vapo.py', ROOT/'postraining/latent_thought.py',
                   ROOT/'postraining/fast_inference.py', ROOT/'postraining/invariant_linear.py',
                   ROOT/'postraining/invariant_attention.py', ROOT/'scripts/diagnose_minicpm_latent_bridge.py']
        hashes = {str(path.resolve().relative_to(ROOT)): sha256(path) for path in sources}
        report.update(provenance=provenance, source_hashes=hashes, torch_version=str(torch.__version__),
                      cuda_version=torch.version.cuda,
                      comparison='same restored checkpoint, prompt cursor and counter seed for both arms')
        modes = ('independent', 'antithetic') if options.arm == 'both' else (options.arm,)
        for current_arm in modes:
            path = options.output/current_arm
            path.mkdir()
            arm_args = argparse.Namespace(**vars(args))
            arm_args.output = path/'trainer-construction'
            arm_args.tensorboard_dir = path/'tensorboard'
            arm_started = time.perf_counter()
            try:
                report['arms'][current_arm] = run_arm(torch, trainer, rollout, profile, arm_args,
                                                       current_arm, counter_seed, path, hashes, provenance)
            except BaseException:
                report['arms'][current_arm] = {'status': 'failed',
                                              'wall_seconds': time.perf_counter()-arm_started}
                raise
            gc.collect()
            torch.cuda.empty_cache()
            atomic_report(options.output/'result.json', report)
        report['status'] = 'diagnostic_complete_no_learning'
        print('DIAGNOSTIC COMPLETE: one real B64 batch per requested arm; '
              'one independent critic refresh per arm; zero optimizer steps.', flush=True)
    except BaseException as error:
        report['status'] = 'failed'
        report['failure'] = {'arm': current_arm, 'type': type(error).__name__,
                             'message': str(error), 'traceback': traceback.format_exc()}
        raise
    finally:
        try:
            if rng is not None:
                torch.set_rng_state(rng[0])
                torch.cuda.set_rng_state_all(rng[1])
                random.setstate(rng[2])
            if precision is not None:
                torch.set_float32_matmul_precision(precision[0])
                torch.backends.cuda.matmul.allow_tf32 = precision[1]
        finally:
            sys.path[:] = old_path
            signal.signal(signal.SIGTERM, old_signal)
            report['wall_seconds'] = time.perf_counter()-started
            atomic_report(options.output/'result.json', report)


if __name__ == '__main__':
    main()

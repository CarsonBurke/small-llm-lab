#!/usr/bin/env python3
"""Queue-only phase profiling of the unchanged frozen MiniCPM controller trainer.

Usage (retain every checkpoint-matched trainer argument):
  mlq submit --name controller_profile --max-parallel-runs 1 --cwd "$PWD" \
    --env TORCH_LOGS=recompiles -- python scripts/profile_minicpm_latent_controller.py \
    --profile-output /absolute/new/profile.json \
    --analysis-batch-output /absolute/new/controller_analysis.pt -- \
    --output /absolute/new/training --resume /absolute/checkpoint.pt [trainer args]

The trainer still uses its real B64/full-10k response path. No budgets, precision,
RNG, optimizer settings, compilation flags, or model computations are replaced.
The JSONL stream is PROFILE_OUTPUT.jsonl; PROFILE_OUTPUT is the exit summary.
CUDA event intervals are stream elapsed time, NOT busy time or GPU utilization.
They can include CPU launch gaps, other stream work, and waits. Counters are
process-global evidence, not proof of recompilation: also collect TORCH_LOGS.
SIGTERM/KeyboardInterrupt produce summaries; SIGKILL or host loss cannot do so.
There is no torch profiler, token hook, tracing hook, or automatic import workload.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def require_mlq_runner():
    """Check actual Linux ancestry; mlqueue does not inject a job-ID variable."""
    pid = os.getppid()
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        proc = Path('/proc') / str(pid)
        try:
            argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
            executable = proc.joinpath('exe').resolve().name.removesuffix(' (deleted)')
            if executable == 'mlqd' and b'__runner' in argv and b'--attempt-dir' in argv:
                return {'runner_pid': pid, 'runner_executable': executable}
            # Fields after the final ')' begin at field 3; PPID is field 4.
            pid = int(proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError):
            break
    raise RuntimeError('This real GPU workload must be submitted through mlq (mlqd __runner ancestry required)')


def strict_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): strict_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def counter_delta(before, after):
    return {key: after.get(key, 0) - before.get(key, 0)
            for key in sorted(before.keys() | after.keys())
            if after.get(key, 0) != before.get(key, 0)}


class PhaseRecorder:
    """Main-thread nested spans; no hooks are installed inside compiled functions."""

    def __init__(self, output):
        self.output = output
        self.stream_path = Path(str(output) + '.jsonl')
        output.parent.mkdir(parents=True, exist_ok=True)
        # Never destroy a previous measurement, even when the trainer is resumed.
        if output.exists():
            raise FileExistsError(output)
        self.stream = self.stream_path.open('x', buffering=1)
        self.started = time.perf_counter()
        self.started_unix = time.time()
        self.owner = threading.get_ident()
        self.stack = []
        self.totals = {}
        self.pending = []
        self.errors = []
        self.patched = []
        self.next_id = 0
        self.torch = None
        self.counters = None
        self.inductor_metrics = None
        self.initial_counters = {}
        self.overhead_seconds = 0.0
        self.analysis_saved = False

    def error(self, where, error):
        self.errors.append({'where': where, 'type': type(error).__name__, 'message': str(error)})

    def emit(self, event, **fields):
        started = time.perf_counter()
        try:
            self.stream.write(json.dumps(strict_value({
                'event': event, 'unix_seconds': time.time(),
                'elapsed_seconds': time.perf_counter() - self.started, **fields,
            }), allow_nan=False, separators=(',', ':')) + '\n')
        except Exception as error:
            self.error('jsonl_write', error)
        finally:
            self.overhead_seconds += time.perf_counter() - started

    def snapshot(self):
        started = time.perf_counter()
        try:
            if self.counters is None:
                return {}
            result = {f'dynamo/{group}/{key}': int(value)
                      for group, counts in list(self.counters.items())
                      for key, value in list(counts.items())
                      if isinstance(value, (int, float)) and math.isfinite(value)}
            # These public integer fields supplement the shared Dynamo counter map.
            for name in ('generated_kernel_count', 'generated_cpp_vec_kernel_count',
                         'ir_nodes_pre_fusion', 'num_bytes_accessed'):
                value = getattr(self.inductor_metrics, name, None)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    result[f'inductor_metrics/{name}'] = value
            return result
        except Exception as error:
            self.error('counter_snapshot', error)
            return {}
        finally:
            self.overhead_seconds += time.perf_counter() - started

    def poll_cuda(self, wait=False):
        if not self.pending:
            return
        started = time.perf_counter()
        overhead_before = self.overhead_seconds
        remaining = []
        for span_id, key, begin, end in self.pending:
            try:
                if wait:
                    end.synchronize()
                if not end.query():
                    remaining.append((span_id, key, begin, end))
                    continue
                seconds = begin.elapsed_time(end) / 1000.0
                self.totals[key]['cuda_event_seconds'] += seconds
                self.totals[key]['cuda_event_samples'] += 1
                self.emit('cuda_interval', span_id=span_id, phase=key, seconds=seconds)
            except Exception as error:
                self.error('cuda_event_query', error)
        self.pending = remaining
        # emit() already charges its own time; count this outer interval once.
        self.overhead_seconds = overhead_before + time.perf_counter() - started

    @contextmanager
    def phase(self, name, *, cuda=False, counters=True, metadata=None):
        if threading.get_ident() != self.owner:
            yield {}
            return
        self.poll_cuda()
        before = self.snapshot() if counters else None
        self.next_id += 1
        span = {'id': self.next_id, 'name': name, 'children_seconds': 0.0,
                'parent_id': self.stack[-1]['id'] if self.stack else None,
                'metadata': metadata or {}}
        self.emit('phase_begin', span_id=span['id'], parent_id=span['parent_id'],
                  phase=name, counters=before, metadata=span['metadata'])
        events = None
        if cuda and self.torch is not None and self.torch.cuda.is_initialized():
            try:
                begin = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)
                begin.record()
                events = begin, end
            except Exception as error:
                self.error('cuda_event_begin', error)
        self.stack.append(span)
        span['started'] = time.perf_counter()
        failure = None
        try:
            yield span['metadata']
        except BaseException as error:
            failure = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            wall = time.perf_counter() - span['started']
            self.stack.pop()
            if self.stack:
                self.stack[-1]['children_seconds'] += wall
            exclusive = wall - span['children_seconds']
            total = self.totals.setdefault(name, {
                'calls': 0, 'failures': 0, 'wall_seconds': 0.0,
                'exclusive_wall_seconds': 0.0, 'max_wall_seconds': 0.0,
                'cuda_event_seconds': 0.0, 'cuda_event_samples': 0,
            })
            total['calls'] += 1
            total['failures'] += int(failure is not None)
            total['wall_seconds'] += wall
            total['exclusive_wall_seconds'] += exclusive
            total['max_wall_seconds'] = max(total['max_wall_seconds'], wall)
            if events is not None:
                try:
                    events[1].record()
                    self.pending.append((span['id'], name, *events))
                except Exception as error:
                    self.error('cuda_event_end', error)
            after = self.snapshot() if counters else None
            self.emit('phase_end', span_id=span['id'], parent_id=span['parent_id'], phase=name,
                      wall_seconds=wall, exclusive_wall_seconds=exclusive,
                      counters=after, counter_delta=counter_delta(before, after) if counters else None,
                      failure=failure, metadata=span['metadata'])
            self.poll_cuda()

    def finish(self, failure, metadata):
        self.poll_cuda(wait=failure is None)
        final_counters = self.snapshot()
        main = self.totals.get('trainer.main', {})
        summary = {
            'schema': 'minicpm-controller-phase-profile/v1',
            'status': 'failed' if failure else ('diagnostics_failed' if self.errors else 'completed'),
            'failure': failure, 'diagnostic_errors': self.errors,
            'started_unix_seconds': self.started_unix, 'finished_unix_seconds': time.time(),
            'elapsed_seconds': time.perf_counter() - self.started,
            'phases': self.totals, 'patched_symbols': self.patched,
            'trainer_main_seconds': main.get('wall_seconds'),
            'trainer_main_unaccounted_seconds': main.get('exclusive_wall_seconds'),
            'instrumentation_overhead_observed_seconds': self.overhead_seconds,
            'initial_counters': self.initial_counters, 'final_counters': final_counters,
            'counter_delta': counter_delta(self.initial_counters, final_counters),
            'unresolved_cuda_intervals': [{'span_id': span_id, 'phase': name}
                                          for span_id, name, _, _ in self.pending],
            'analysis_batch_saved': self.analysis_saved, 'jsonl': str(self.stream_path),
            'limitations': [
                'CUDA event intervals include stream waits and CPU launch gaps; they are not GPU busy time/utilization.',
                'No phase-entry CUDA synchronization; prior queued work can affect wall/event comparisons.',
                'Inclusive nested phase times overlap. Use exclusive times, not their inclusive sum.',
                'trainer.main exclusive is startup, nested evaluation/save closure overhead, logging and other uninstrumented work.',
                'generate_prompt_pool exclusive includes scheduler/CPU per-position copying, decode, drain and graph replay, not only kernels.',
                'Graph capture spans include the existing two compile/allocator warmups; capture counts are not Dynamo recompilations.',
                'Process-global counters can include unrelated compilation; use timestamped TORCH_LOGS=recompiles for definitive guard reasons.',
                'Counter/JSONL/event-query overhead is measured approximately and is already included in parent/unaccounted time; do not add it.',
                'Only the main thread is instrumented; no per-token hook, profiler trace, NVML sample, or internal verifier-worker timing.',
                'Analysis export is one extra CPU crop/copy and one head-state copy; its measured cost is explicit and consumes trainer wall budget.',
                'Noise displacement relative to saved state includes the learned mean residual; it is not an estimate of Gaussian sigma.',
                'Final event synchronization is diagnostic exit overhead; failures only query ready events. SIGKILL cannot produce an exit summary.',
            ], **metadata,
        }
        self.emit('profile_summary', summary=summary)
        try:
            with self.output.open('x') as handle:
                json.dump(strict_value(summary), handle, allow_nan=False, indent=2)
                handle.write('\n')
        finally:
            self.stream.close()


def caller_context(trainer):
    """Observe existing trainer locals at outer phase boundaries, without tracing."""
    context = {}
    frame = sys._getframe(1)
    try:
        while frame is not None:
            if frame.f_code.co_filename == trainer.__file__:
                name = frame.f_code.co_name
                if name in ('evaluate', 'save'):
                    context['activity'] = name
                if name == 'main':
                    for key in ('step', 'warmups', 'training_seconds', 'cursor'):
                        if key in frame.f_locals:
                            context[key] = frame.f_locals[key]
            frame = frame.f_back
    finally:
        del frame
    context.setdefault('activity', 'training_or_warmup')
    return context


@contextmanager
def patched(recorder, target, name, replacement):
    # Preserve descriptors and delete temporary instance overrides on restoration.
    namespace = vars(target)
    owned = name in namespace
    original = namespace.get(name)
    label = f'{getattr(target, "__module__", "")}.{getattr(target, "__qualname__", getattr(target, "__name__", type(target).__name__))}.{name}'.strip('.')
    if label not in recorder.patched:
        recorder.patched.append(label)
    setattr(target, name, replacement)
    try:
        yield
    finally:
        if owned:
            setattr(target, name, original)
        else:
            delattr(target, name)


def wrap_phase(recorder, original, name, *, cuda=False, counters=True, context=None, metadata=None, observe=None):
    @wraps(original)
    def wrapper(*args, **kwargs):
        details = context() if context else {}
        if metadata:
            details.update(metadata(*args, **kwargs))
        with recorder.phase(name, cuda=cuda, counters=counters, metadata=details) as details:
            result = original(*args, **kwargs)
            if observe:
                try:
                    details.update(observe(result, *args, **kwargs))
                except Exception as error:
                    recorder.error(name + '.observe', error)
            return result
    return wrapper


def save_analysis(recorder, policy, records, args, output, source_hashes):
    torch = recorder.torch
    from postraining.vapo.policy import FIRST_THOUGHT, CONTINUE_THOUGHT, STOP_THINKING

    compact = []
    noise = {'sampled_states': 0, 'elements': 0, 'raw_minus_saved_state_sum_sq': 0.0,
             'raw_bf16_roundtrip_sum_sq': 0.0, 'raw_bf16_equals_saved_state_elements': 0}
    original_observation_bytes = cropped_observation_bytes = 0
    for index, record in enumerate(records):
        observations, raw = record.controller_observations, record.latent_vectors
        if observations is None or observations.dtype != torch.bfloat16 or observations.device.type != 'cpu':
            raise ValueError('analysis requires original CPU BF16 controller observations')
        if raw is None or raw.dtype != torch.float32 or raw.device.type != 'cpu':
            raise ValueError('analysis requires original CPU FP32 Gaussian actions')
        thoughts = raw.size(0)
        end = thoughts + int(record.forced_token_index < 0)
        kinds = record.action_kinds[:end]
        if thoughts < 1 or int(kinds[0]) != FIRST_THOUGHT:
            raise ValueError('missing mandatory FIRST action')
        if not bool((kinds[1:thoughts] == CONTINUE_THOUGHT).all()):
            raise ValueError('Gaussian prefix is not contiguous')
        if kinds.numel() != end or (end > thoughts and int(kinds[-1]) != STOP_THINKING):
            raise ValueError('analysis suffix must be a genuine STOP, never forced close or answer')
        fields = {'observations': observations[:end], 'raw': raw,
                  'kinds': kinds, 'advantages': record.advantages[:end],
                  'old_logprobs': record.old_logprobs[:end]}
        compact.append({**{key: value.detach().clone() for key, value in fields.items()},
                        'correct': bool(record.correct), 'group_index': index // args.samples_per_prompt})
        original_observation_bytes += observations.numel() * observations.element_size()
        cropped_observation_bytes += end * observations.size(1) * observations.element_size()
        # Bounded stratified prefix diagnostic only; no head/trunk replay or draws.
        count = min(thoughts, 4)
        selected_raw = raw[:count]
        selected_state = observations[:count].float()
        rounded = selected_raw.to(torch.bfloat16).float()
        noise['sampled_states'] += count
        noise['elements'] += selected_raw.numel()
        noise['raw_minus_saved_state_sum_sq'] += (selected_raw - selected_state).square().sum().item()
        noise['raw_bf16_roundtrip_sum_sq'] += (selected_raw - rounded).square().sum().item()
        noise['raw_bf16_equals_saved_state_elements'] += (rounded == selected_state).sum().item()
    elements, states = noise['elements'], noise['sampled_states']
    noise.update({
        'sampling': 'first up to four genuine Gaussian actions per real trajectory',
        'raw_minus_saved_state_component_rms': math.sqrt(noise['raw_minus_saved_state_sum_sq'] / elements) if elements else None,
        'raw_minus_saved_state_vector_rms': math.sqrt(noise['raw_minus_saved_state_sum_sq'] / states) if states else None,
        'raw_bf16_roundtrip_component_rms': math.sqrt(noise['raw_bf16_roundtrip_sum_sq'] / elements) if elements else None,
        'raw_bf16_equals_saved_state_fraction': noise['raw_bf16_equals_saved_state_elements'] / elements if elements else None,
        'interpretation': 'State-relative displacement includes the learned mean; use geometry analysis for current-mean-centered noise.',
    })
    payload = {
        'schema': 'minicpm-controller-analysis/v1',
        'saved_point': 'after refresh_critic, before controller_update',
        'model_dim': int(policy.transition.model_dim),
        'component_std': float(policy.transition.component_std),
        'vector_sigma': float(policy.transition.vector_sigma),
        'transition_state': {key: value.detach().to('cpu', copy=True) for key, value in policy.transition.state_dict().items()},
        'gate_state': {key: value.detach().to('cpu', copy=True) for key, value in policy.thinking_gate.state_dict().items()},
        'records': compact, 'source_hashes': source_hashes, 'seed': args.seed,
        'samples_per_prompt': args.samples_per_prompt, 'noise_diagnostics': noise,
    }
    with output.open('xb') as handle:
        torch.save(payload, handle)
    recorder.analysis_saved = True
    recorder.emit('analysis_batch_saved', path=str(output), records=len(compact),
                  controller_states=sum(row['observations'].size(0) for row in compact),
                  original_observation_bytes=original_observation_bytes,
                  cropped_observation_bytes=cropped_observation_bytes, noise_diagnostics=noise)


def generation_details(result, engine, *args, **kwargs):
    observations = result.controller_observations
    positions = sum(value.size(0) for value in observations)
    thoughts = sum(value.size(0) for value in result.latent_vectors)
    from postraining.vapo.policy import STOP_THINKING
    genuine_stops = sum(int((kinds == STOP_THINKING).sum()) for kinds in result.action_kinds)
    return {
        'engine_telemetry': dict(engine.telemetry),
        'engine_reported_prefill_seconds': result.prefill_seconds,
        'engine_reported_decode_seconds': result.decode_seconds,
        'emitted_positions': positions, 'gaussian_positions': thoughts,
        'genuine_stop_positions': genuine_stops,
        'saved_observation_bytes': sum(value.numel() * value.element_size() for value in observations),
        'controller_relevant_observation_positions': thoughts + genuine_stops,
        'noncontroller_observation_positions': positions - thoughts - genuine_stops,
        'per_position_observation_clones_source_derived': positions,
        'clone_count_note': 'Source _generate clones one observation tensor per emitted position; count derived from real output, not a clone hook.',
    }


def install_and_run(recorder, trainer, rollout, runtime, args, analysis_output, hashes):
    from contextlib import ExitStack

    context = lambda: caller_context(trainer)
    first_update = True
    gradient_iterator = None
    original_update = trainer.controller_update
    original_batches = trainer.controller_batches

    @wraps(original_batches)
    def controller_batches(*batch_args, **kwargs):
        nonlocal gradient_iterator
        # The generator spans the real caller's scoring/backward between yields.
        iterator = original_batches(*batch_args, **kwargs)
        caller = sys._getframe(1).f_code.co_name
        if caller != 'controller_update':
            return iterator

        def measured():
            with recorder.phase('controller.gradient_loop'):
                yield from iterator
        gradient_iterator = measured()
        return gradient_iterator

    @wraps(original_update)
    def controller_update(policy, optimizer, records, **kwargs):
        nonlocal first_update, gradient_iterator
        if first_update:
            first_update = False
            with recorder.phase('diagnostic.analysis_batch', metadata=context()):
                try:
                    save_analysis(recorder, policy, records, args, analysis_output, hashes)
                except Exception as error:
                    recorder.error('analysis_batch', error)
                    recorder.emit('analysis_batch_failed', error=str(error))
        step = wrap_phase(recorder, optimizer.step, 'controller.optimizer_step', counters=False)
        with patched(recorder, optimizer, 'step', step):
            with recorder.phase('controller.update', cuda=True, metadata=context()) as details:
                try:
                    result = original_update(policy, optimizer, records, **kwargs)
                    details.update(result)
                    return result
                finally:
                    # A traceback can retain a partially consumed generator. Close
                    # its span before the enclosing update span on failures too.
                    if gradient_iterator is not None:
                        gradient_iterator.close()
                        gradient_iterator = None

    def capture_metadata(chunks, entry, steps):
        return {'steps': steps, 'thinking_lanes': entry.thought_indices.numel(),
                'answering_lanes': entry.answer_indices.numel(),
                'cached_graphs_before': len(chunks.graphs), 'max_cached_graphs': chunks.engine.max_cached_graphs}

    specs = [
        (trainer, 'collect_rollouts', 'rollout.collect', False, True, None, None),
        (runtime, 'collect_latent_rollouts', 'rollout.latent_collect', False, True, None, None),
        (runtime, 'verify_answer', 'rollout.verify_answer', False, False, None, None),
        (runtime, '_decode_text', 'rollout.decode_text', False, False, None, None),
        (trainer, 'rollout_diagnostics', 'rollout.diagnostics', False, False, None, None),
        (rollout.MiniCPMLatentRolloutEngine, '__init__', 'engine.initialize', False, True, None, None),
        (rollout.MiniCPMLatentRolloutEngine, 'generate_prompt_pool', 'rollout.generate_prompt_pool', True, True, None, generation_details),
        (rollout.MiniCPMLatentRolloutEngine, 'synchronize_from', 'replica.synchronize_from', False, True, None, None),
        (rollout, 'synchronize_fused_lora_policy_', 'replica.merge_and_copy', False, True, None, None),
        (rollout, 'build_fused_rollout_replica', 'replica.build', False, True, None, None),
        (rollout.MiniCPMLatentRolloutEngine, 'release_cache', 'cache.release', False, True, None, None),
        (rollout.MiniCPMLatentRolloutEngine, '_release_decode_storage', 'cache.release_decode_storage', False, False, None, None),
        (rollout._FrozenParameterStash, 'restore', 'source.restore', False, True, None, None),
        (rollout._FrozenParameterStash, 'offload', 'source.offload', False, True, None, None),
        (rollout._LatentStateDecoder, 'prefill', 'decoder.prefill', True, True, None, None),
        (rollout._LatentStateDecoder, 'release_cache', 'decoder.release_cache', False, False, None, None),
        (rollout._LatentChunks, '_capture', 'cuda_graph.capture_with_warmup', False, True, capture_metadata, None),
        (rollout._LatentChunks, 'release', 'cuda_graph.release', False, False, None, None),
        (trainer, 'refresh_critic', 'critic.refresh', True, True, None, None),
        (trainer, 'controller_kl', 'controller.kl', False, True, None, lambda result, *a, **k: result),
        (trainer, 'update_step', 'critic.update_step', True, True, None, None),
        (trainer, 'atomic_torch_save', 'checkpoint.serialize', False, True, None, None),
    ]
    with ExitStack() as stack:
        for target, symbol, name, cuda, counters, meta, observe in specs:
            replacement = wrap_phase(recorder, getattr(target, symbol), name, cuda=cuda,
                                     counters=counters, context=context if counters else None,
                                     metadata=meta, observe=observe)
            stack.enter_context(patched(recorder, target, symbol, replacement))
        record_class = runtime.TrajectoryRecord
        original_from_device = record_class.from_device

        def from_device(cls, **kwargs):
            with recorder.phase('rollout.build_record', counters=False):
                return original_from_device(**kwargs)
        stack.enter_context(patched(recorder, record_class, 'from_device', classmethod(from_device)))
        stack.enter_context(patched(recorder, trainer, 'controller_batches', controller_batches))
        stack.enter_context(patched(recorder, trainer, 'controller_update', controller_update))
        with recorder.phase('trainer.main'):
            trainer.main()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument('--profile-output', type=Path, required=True)
    parser.add_argument('--analysis-batch-output', type=Path, required=True)
    parser.add_argument('trainer_args', nargs=argparse.REMAINDER,
                        help='-- followed by unchanged train_minicpm_latent_controller.py arguments')
    options = parser.parse_args(argv)
    trainer_argv = options.trainer_args
    if trainer_argv[:1] == ['--']:
        trainer_argv = trainer_argv[1:]
    if not trainer_argv:
        parser.error('pass the real trainer arguments after --')
    recorder = PhaseRecorder(options.profile_output)
    failure = None
    metadata = {'trainer_argv': trainer_argv, 'analysis_batch_output': str(options.analysis_batch_output),
                'environment': {key: os.environ.get(key) for key in
                                ('TORCH_LOGS', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_MODULE_LOADING')}}
    old_argv = sys.argv
    old_path = list(sys.path)
    old_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, terminate)
        metadata['queue'] = require_mlq_runner()
        paths = [options.profile_output.resolve(), Path(str(options.profile_output) + '.jsonl').resolve(),
                 options.analysis_batch_output.resolve()]
        if len(set(paths)) != len(paths):
            raise ValueError('profile, JSONL and analysis output paths must be distinct')
        if options.analysis_batch_output.exists():
            raise FileExistsError(options.analysis_batch_output)
        options.analysis_batch_output.parent.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(ROOT))
        with recorder.phase('startup.imports', counters=False):
            import torch
            import torch._dynamo.utils
            import torch._inductor.metrics
            from scripts import train_minicpm_latent_controller as trainer
            from postraining import minicpm_latent_rollout as rollout
            from postraining import train_minicpm_vapo as runtime
        recorder.torch = torch
        recorder.counters = torch._dynamo.utils.counters
        recorder.inductor_metrics = torch._inductor.metrics
        recorder.initial_counters = recorder.snapshot()
        args = trainer.parser().parse_args(trainer_argv)
        sys.argv = [trainer.__file__, *trainer_argv]
        sources = [Path(__file__), Path(trainer.__file__), Path(rollout.__file__), Path(runtime.__file__),
                   ROOT / 'postraining/vapo/policy.py', ROOT / 'postraining/latent_thought.py',
                   ROOT / 'postraining/fast_inference.py', ROOT / 'postraining/invariant_linear.py',
                   ROOT / 'postraining/invariant_attention.py']
        hashes = {str(path.resolve().relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        metadata.update({'source_hashes': hashes, 'seed': args.seed, 'torch_version': torch.__version__,
                         'cuda_version': torch.version.cuda, 'trainer_args': vars(args)})
        recorder.emit('profile_start', **metadata)
        install_and_run(recorder, trainer, rollout, runtime, args, options.analysis_batch_output, hashes)
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        raise
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
        signal.signal(signal.SIGTERM, old_sigterm)
        try:
            recorder.finish(failure, metadata)
        except Exception as error:
            # Do not replace a trainer exception with a secondary export failure.
            if failure is None:
                raise
            print(f'Could not finalize controller profile: {error}', file=sys.stderr)


if __name__ == '__main__':
    main()

"""Pure metadata tests: expensive training must fail closed on stale evidence."""
import hashlib
import json

import pytest

import scripts.train_recurrent_slots as trainer


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, 'ROOT', tmp_path)
    hashes = {}
    for name in ('chunk_memory.py', 'chunk_memory_runtime.py', 'nanogpt_mini_model.py',
                 'recurrent_slots_runtime.py'):
        relative = 'pretraining/nanogpt_mini/' + name
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ('scripts/train_recurrent_slots.py', 'scripts/benchmark_chunk_memory.py'):
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(relative)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    path = tmp_path / 'report.json'
    report = dict(status='completed', gate_passed=True, source_sha256=hashes,
                  batch_tokens=524288, seq_len=1024, optimizer_included=True, repeats=5,
                  torch=str(trainer.torch.__version__),
                  candidate=dict(microbatch=64, chunk_size=256, slots=32,
                                 tokens_per_second=120, model_config=dict(
                                     vocab_size=1024, num_layers=6, model_dim=512,
                                     head_dim=128, slots=32, chunk_size=256,
                                     checkpoint_chunks=False)),
                  baseline=dict(tokens_per_second=100, model='nanogpt_mini'))
    path.write_text(json.dumps(report))
    args = trainer.parse_args(['--name', 'test', '--steps', '1000', '--architecture',
                               'chunk', '--segment-size', '256', '--throughput-report', str(path)])
    return args, path, report


def test_chunk_requires_measured_throughput():
    with pytest.raises(SystemExit):
        trainer.parse_args(['--name', 'test', '--steps', '1000', '--architecture', 'chunk'])


def test_valid_evidence_is_bound_to_report(evidence):
    args, path, _ = evidence
    result = trainer.verify_throughput_gate(args)
    assert result['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('change', ['failed', 'slow', 'nonfinite', 'batch', 'shape', 'source'])
def test_training_rejects_invalid_or_stale_evidence(evidence, change):
    args, path, report = evidence
    if change == 'failed':
        report['gate_passed'] = False
    elif change == 'slow':
        report['candidate']['tokens_per_second'] = 104
    elif change == 'nonfinite':
        report['candidate']['tokens_per_second'] = float('nan')
    elif change == 'batch':
        report['candidate']['microbatch'] = 32
    elif change == 'shape':
        report['candidate']['model_config']['model_dim'] = 256
    else:
        (trainer.ROOT / 'pretraining/nanogpt_mini/chunk_memory.py').write_text('changed')
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        trainer.verify_throughput_gate(args)

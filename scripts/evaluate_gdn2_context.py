"""Evaluate a completed GDN2 checkpoint on the canonical panel; use mlq.

Run each context/checkpoint combination in a fresh process so autotuning cannot
reuse an in-memory configuration selected for another execution shape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sentencepiece as spm
import torch

from pretraining.nanogpt_mini.gated_delta_model import build_gated_delta_model
from pretraining.nanogpt_mini.gated_delta_runtime import CompiledGatedDeltaLoss, gdn2_dependency_provenance
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphValidation
from scripts.train_recurrent_slots import PackedBatches, atomic_json, build_sentencepiece_luts


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seq-len', type=int, choices=(1024, 4096), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required; submit through mlq')
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    run = args.run_dir.resolve()
    training = json.loads((run / 'config.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    checkpoint_path = run / 'model.pt'
    checkpoint_hash = digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (result.get('status') != 'completed' or result.get('completed_steps') != 1000
            or checkpoint.get('completed_steps') != 1000
            or checkpoint.get('train_seq_len') != training['seq_len']
            or checkpoint['model_config'] != training['model_config']):
        raise ValueError('Require a completed matching 1000-update checkpoint')
    tokenizer_path = resolve(training['tokenizer'])
    if digest(tokenizer_path) != training['tokenizer_sha256']:
        raise ValueError('Training tokenizer changed')
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    loader = PackedBatches(str(resolve(training['data_path']) / 'fineweb_val_*.bin'),
                           1048576, args.seq_len)
    inputs, targets = loader.next()
    del loader
    panel_hash = hashlib.sha256(inputs.cpu().numpy().tobytes() + targets.cpu().numpy().tobytes()).hexdigest()
    base, leading, boundary = build_sentencepiece_luts(tokenizer, vocab_size=1024, device='cuda')
    byte_counts = base[targets].long() + (leading[targets] & ~boundary[inputs.long()]).long()
    val_bytes = int(byte_counts.sum())
    if val_bytes != training['val_bytes'] or val_bytes != 2524883:
        raise ValueError('Validation bytes differ from canonical training panel')
    del base, leading, boundary, byte_counts
    model = build_gated_delta_model(checkpoint['model_config']).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    loss_fn = CompiledGatedDeltaLoss(model, 64)
    microbatch = 65536 // args.seq_len
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / 'scripts/train_recurrent_slots.py',
               *(ROOT / 'pretraining/nanogpt_mini' / name for name in
                 ('gated_delta_model.py', 'gated_delta_runtime.py', 'recurrent_slots_runtime.py',
                  'nanogpt_mini_model.py'))]
    sources += [path for path in (ROOT / 'pretraining/gated_delta/vendor').rglob('*')
                if path.is_file() and '__pycache__' not in path.parts]
    report = dict(status='running', run_dir=str(run), completed_training_updates=1000,
                  train_seq_len=training['seq_len'], val_seq_len=args.seq_len,
                  val_tokens=targets.numel(), val_bytes=val_bytes, microbatch=microbatch,
                  validation_panel_sha256=panel_hash, checkpoint_sha256=checkpoint_hash,
                  model_config=model.config, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  training_config_sha256=digest(run / 'config.json'),
                  training_result_sha256=digest(run / 'result.json'),
                  dynamo_suppress_errors=False, dynamo_fail_on_recompile_limit_hit=True,
                  tokenizer_sha256=digest(tokenizer_path), loss_reduction='sum',
                  environment={key: os.environ.get(key) for key in
                               ('FLA_CACHE_RESULTS', 'FLA_CACHE_MODE', 'TRITON_CACHE_AUTOTUNING')},
                  installed_fla=gdn2_dependency_provenance(),
                  source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sources})
    atomic_json(args.output, report)
    try:
        graph = CUDAGraphValidation(loss_fn, batch_size=microbatch, seq_len=args.seq_len)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            direct = loss_fn(inputs[:microbatch], targets[:microbatch])
            captured = graph.replay(inputs[:microbatch], targets[:microbatch])[0]
            relative_error = float((direct - captured).abs() / direct.abs())
            if not math.isfinite(relative_error) or relative_error > 1e-5:
                raise RuntimeError(f'Validation graph differs from compiled loss: {relative_error}')
            total = torch.zeros((), device='cuda', dtype=torch.float64)
            for row in range(0, inputs.shape[0], microbatch):
                total += graph.replay(inputs[row:row + microbatch], targets[row:row + microbatch])[0].double()
        nll = float(total)
        if not math.isfinite(nll):
            raise FloatingPointError('Nonfinite validation loss')
        report.update(status='completed', val_loss=nll / targets.numel(),
                      val_bpb=nll / val_bytes / math.log(2),
                      graph_loss_relative_error=relative_error, graph_breaks=loss_fn.audit_graph_breaks())
        if digest(checkpoint_path) != checkpoint_hash:
            raise RuntimeError('Checkpoint changed during evaluation')
        if (digest(run / 'config.json') != report['training_config_sha256']
                or digest(run / 'result.json') != report['training_result_sha256']):
            raise RuntimeError('Training metadata changed during evaluation')
        if gdn2_dependency_provenance() != report['installed_fla']:
            raise RuntimeError('Installed FLA changed during evaluation')
        if any(digest(ROOT / path) != value for path, value in report['source_sha256'].items()):
            raise RuntimeError('Evaluation sources changed')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in ('status', 'train_seq_len', 'val_seq_len', 'val_bpb')}))


if __name__ == '__main__':
    main()

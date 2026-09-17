#!/usr/bin/env python3
"""Queue-only initial-policy real-reward length ablation; no learning claim."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from postraining.core import load_unique_math_rows
from postraining.minicpm_vapo import MiniCPMVAPOPolicy, LoRAConfig
from postraining.minicpm_latent_rollout import MiniCPMLatentRolloutEngine
from postraining.fast_inference import CapturedTrainingRolloutEngine
from postraining.train_minicpm_vapo import (
    collect_rollouts, rollout_diagnostics, _stop_ids,
    resolve_thinking_start_token, resolve_thinking_end_token, encode_math_prompt,
)

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--data', default='postraining/data/dapo-math-17k.parquet')
parser.add_argument('--prompts', type=int, default=32)
parser.add_argument('--samples', type=int, default=4)
parser.add_argument('--group-prompts', type=int, default=16)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--gate-probabilities', type=float, nargs='+', default=[0.9, 1/64, 1/1024])
parser.add_argument('--native', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--normalize-thought-input', action='store_true')
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
assert torch.cuda.is_available()
assert args.prompts % args.group_prompts == 0
assert all(0 < p < 1 for p in args.gate_probabilities)
torch.manual_seed(args.seed)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
rows = load_unique_math_rows(args.data)
random.Random(args.seed).shuffle(rows)
rows = rows[:args.prompts]
policy, tokenizer = MiniCPMVAPOPolicy.from_pretrained(
    device=torch.device('cuda'), lora_config=LoRAConfig(initialization='nora'),
    gradient_checkpointing=False, latent_thinking=True,
)
policy.eval()
stops = _stop_ids(policy, tokenizer)
close = resolve_thinking_end_token(tokenizer, stop_ids=stops)
opening = resolve_thinking_start_token(tokenizer, stop_ids=stops)
with torch.inference_mode():
    calibration_ids = torch.cat([
        encode_math_prompt(tokenizer, row, prompt_tokens=1024, enable_thinking=True)
        for row in rows
    ]).to('cuda')
    reference_norm = policy.token_embeddings(calibration_ids).float().norm(dim=-1).median().item()


class NormalizedThoughtInput(torch.nn.Module):
    def __init__(self, adapter, output_norm):
        super().__init__()
        self.adapter = adapter
        self.rms = output_norm / math.sqrt(policy.causal_lm.config.hidden_size)

    def forward(self, base, raw):
        embedded = self.adapter(base, raw)
        return (F.rms_norm(embedded.float(), (embedded.size(-1),)) * self.rms).to(embedded.dtype)


if args.normalize_thought_input:
    policy.thought_adapter = NormalizedThoughtInput(policy.thought_adapter, reference_norm)
report = {
    'scope': 'initial-policy real-reward exploration; training-pool sample, not held-out learning evidence',
    'data': args.data, 'seed': args.seed, 'prompt_count': len(rows), 'samples_per_prompt': args.samples,
    'prompt_tokens': 1024, 'max_new_tokens': 10000, 'answer_reserve_tokens': 1024,
    'thought_sigma': policy.transition.vector_sigma, 'component_std': policy.transition.component_std,
    'prompt_sha256': hashlib.sha256(json.dumps([row['prompt'] for row in rows], sort_keys=True).encode()).hexdigest(),
    'normalize_thought_input': args.normalize_thought_input,
    'prompt_token_embedding_median_norm': reference_norm,
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'arms': {},
}
args.output.parent.mkdir(parents=True, exist_ok=True)


def flush():
    args.output.write_text(json.dumps(report, indent=2) + '\n')


def exercise(engine, name):
    arm = {'groups': [], 'responses': []}
    report['arms'][name] = arm
    for start in range(0, len(rows), args.group_prompts):
        torch.manual_seed(args.seed + start)
        began = time.perf_counter()
        result = collect_rollouts(
            engine, tokenizer, rows[start:start+args.group_prompts],
            prompt_tokens=1024, max_new_tokens=10000, enable_thinking=True,
        )
        metrics = rollout_diagnostics(result, samples_per_prompt=args.samples, stop_ids=stops)
        metrics['wall_seconds'] = time.perf_counter() - began
        arm['groups'].append(metrics)
        for i, record in enumerate(result.records):
            arm['responses'].append({
                'prompt_index': start+i//args.samples, 'sample_index': i%args.samples,
                'correct': record.correct, 'text': record.text,
                'stream_positions': record.response_length,
                'thoughts': record.latent_vectors.size(0) if record.latent_vectors is not None else None,
                'forced_close': record.forced_token_index >= 0,
            })
        flush()
        print(name, start, json.dumps(metrics), flush=True)
        engine.release_cache()
        del result
        gc.collect()
    arm['accuracy'] = sum(row['correct'] for row in arm['responses']) / len(arm['responses'])
    arm['wall_seconds'] = sum(group['wall_seconds'] for group in arm['groups'])
    flush()


for probability in args.gate_probabilities:
    with torch.no_grad():
        policy.thinking_gate.head.weight.zero_()
        policy.thinking_gate.head.bias.fill_(math.log(probability/(1-probability)))
    engine = MiniCPMLatentRolloutEngine(
        policy, stop_ids=stops, thinking_start_token_id=opening, thinking_end_token_id=close,
        prompts_per_rollout=args.group_prompts, samples_per_prompt=args.samples,
        physical_batch_size=args.group_prompts*args.samples, cache_length=11024,
        temperature=0.9, top_k=20, top_p=0.95, answer_reserve_tokens=1024,
    )
    exercise(engine, f'latent_stop_{probability:g}')
    del engine
    gc.collect()
    torch.cuda.empty_cache()
if args.native:
    policy.latent_thinking = False
    engine = CapturedTrainingRolloutEngine(
        policy, stop_ids=stops, prompts_per_rollout=args.group_prompts,
        samples_per_prompt=args.samples, cache_length=11024,
        temperature=0.9, top_k=20, top_p=0.95,
        thinking_end_token_id=close, answer_reserve_tokens=1024, compile_decode=True,
    )
    exercise(engine, 'native_cot')
report['status'] = 'completed'
flush()

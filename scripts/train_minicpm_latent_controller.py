#!/usr/bin/env python3
"""Queue-only frozen-dynamics latent-controller ablation with an exact KL constraint.

This is NOT end-to-end MiniCPM PPO. The fixed bf16 actor and input adapter define
state transitions; fp32 Gaussian/gate heads learn from actual rollout observations.
The independent critic still reconstructs every exact raw continuous action.
"""
from __future__ import annotations

import argparse
import atexit
import copy
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from postraining.controller_metrics import write_controller_metrics

from checkpointing import atomic_torch_save
from postraining.core import load_unique_math_rows
from postraining.latent_thought import GaussianTransitionHead, StopThinkingGate
from postraining.vapo.policy import (
    FIRST_THOUGHT,
    VAPOCritic,
    VAPOPolicy,
    STOP_THINKING,
    collate_replay_microbatch,
    plan_replay_microbatches,
)
from postraining.vapo.model.hf import (
    MINICPM5_SPEC,
    enable_packed_replay_attention,
    enable_replay_mlp_compilation,
)
from postraining.vapo.model.lora import (
    LoRAConfig,
    adapter_state_dict,
    load_adapter_state_dict,
)
from postraining.minicpm_latent_rollout import MiniCPMLatentRolloutEngine
from postraining.minicpm_paired_rollout import PairedControllerRolloutEngine
from postraining.train_minicpm_vapo import (
    collect_rollouts, rollout_diagnostics, _stop_ids, encode_math_prompt,
    resolve_thinking_start_token, resolve_thinking_end_token,
    _replay_hidden, update_step,
)
from scripts.diagnose_minicpm_latent_bridge import RidgeThoughtInput, digest

# Promote to 50 only after the queued real-logit performance gate passes.
DEFAULT_TOP_K = 20


class WhitenedGaussianController(GaussianTransitionHead):
    def predict_mean(self, belief):
        with torch.autocast(device_type=belief.device.type, enabled=False):
            features = F.rms_norm(belief.float(), (self.model_dim,))
            return belief.float() + self.component_std * self.mean_head(features)


class NormalizedStopGate(StopThinkingGate):
    def stop_logit(self, belief):
        with torch.autocast(device_type=belief.device.type, enabled=False):
            features = F.rms_norm(belief.float(), (belief.shape[-1],))
            return self.head(features).squeeze(-1)


class NormalizedThoughtInput(nn.Module):
    def __init__(self, adapter, norm, dimension):
        super().__init__()
        self.adapter = adapter
        self.rms = float(norm) / math.sqrt(dimension)

    def forward(self, base, raw):
        embedded = self.adapter(base, raw)
        return (F.rms_norm(embedded.float(), (embedded.size(-1),)) * self.rms).to(embedded.dtype)


def controller_batches(records, batch_size):
    for record in records:
        observations = record.controller_observations
        if observations is None:
            raise ValueError('actual rollout observations are required; replay states are not substitutes')
        thoughts = record.latent_vectors.size(0)
        end = thoughts + int(record.forced_token_index < 0)
        for start in range(0, end, batch_size):
            stop = min(start + batch_size, end)
            yield (
                observations[start:stop].to('cuda'),
                record.action_kinds[start:stop].to('cuda'),
                record.latent_vectors[start:min(stop, thoughts)].to('cuda'),
                record.advantages[start:stop].to('cuda'),
                record.old_logprobs[start:stop].to('cuda'),
            )


def controller_scores(policy, observations, kinds, raw, *, noise_sum_squares=None):
    count = raw.size(0)
    result = observations.new_zeros(observations.size(0), dtype=torch.float32)
    if count:
        states = observations[:count]
        mean = policy.transition.predict_mean(states)
        result = torch.cat((policy.transition.log_prob(raw.detach(), mean,
                             policy.transition.predict_log_sigma(states)), result[count:]))
        if noise_sum_squares is not None:
            # Measured pre-update action residuals, not the configured noise scale.
            # Reuse the scoring mean; no extra controller or trunk forward.
            with torch.no_grad():
                noise = raw.detach().float() - mean.detach()
                noise_sum_squares.add_(noise.square_().sum(dtype=torch.float64))
    # First thought is mandatory. Forced closure is excluded by controller_batches.
    gate_mask = kinds != FIRST_THOUGHT
    gate = policy.thinking_gate.log_prob((kinds == STOP_THINKING).long(), observations)
    return result + torch.where(gate_mask, gate, 0.0)


@torch.no_grad()
def controller_kl(policy, old_transition, old_gate, records, batch_size):
    maximum_gaussian = torch.zeros((), device='cuda', dtype=torch.float64)
    maximum_joint = torch.zeros_like(maximum_gaussian)
    total_joint = torch.zeros_like(maximum_gaussian)
    count = 0
    for states, kinds, _, _, _ in controller_batches(records, batch_size):
        old_mean = old_transition.predict_mean(states)
        new_mean = policy.transition.predict_mean(states)
        gaussian = 0.5 * ((new_mean.double() - old_mean.double()) / old_transition.component_std).square().sum(-1)
        old_logit = old_gate.stop_logit(states).double()
        new_logit = policy.thinking_gate.stop_logit(states).double()
        probability = old_logit.sigmoid()
        gate = probability * (F.logsigmoid(old_logit) - F.logsigmoid(new_logit))
        gate += (1-probability) * (F.logsigmoid(-old_logit) - F.logsigmoid(-new_logit))
        joint = torch.where(kinds == FIRST_THOUGHT, gaussian, gate + (1-probability)*gaussian)
        maximum_gaussian = torch.maximum(maximum_gaussian, gaussian.max())
        maximum_joint = torch.maximum(maximum_joint, joint.max())
        total_joint += joint.sum()
        count += states.size(0)
    return {'max_gaussian_kl': maximum_gaussian.item(), 'max_joint_kl': maximum_joint.item(),
            'mean_joint_kl': (total_joint/max(count, 1)).item(), 'controller_states': count}


def controller_update(policy, optimizer, records, *, batch_size, max_kl):
    parameters = list(policy.transition.parameters()) + list(policy.thinking_gate.parameters())
    optimizer.zero_grad(set_to_none=True)
    count = len(records)
    max_drift = 0.0
    loss_value = 0.0
    noise_sum_squares = torch.zeros((), device=parameters[0].device, dtype=torch.float64)
    noise_components = gaussian_actions = 0
    for states, kinds, raw, advantages, old_scores in controller_batches(records, batch_size):
        scores = controller_scores(policy, states, kinds, raw, noise_sum_squares=noise_sum_squares)
        noise_components += raw.numel()
        gaussian_actions += raw.size(0)
        max_drift = max(max_drift, (scores.detach()-old_scores).abs().max().item())
        loss = -(scores * advantages.detach()).sum()/count
        loss.backward()
        loss_value += loss.detach().item()
    if max_drift > 0.05:
        raise RuntimeError(f'controller sampling/replay density mismatch: {max_drift:g}; no behavior refresh allowed')
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        raise RuntimeError('nonfinite on-policy controller gradient')
    old_parameters = [p.detach().clone() for p in parameters]
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    old_transition = copy.deepcopy(policy.transition).requires_grad_(False)
    old_gate = copy.deepcopy(policy.thinking_gate).requires_grad_(False)
    optimizer.step()
    proposal = [p.detach().clone() for p in parameters]
    fraction = 1.0
    accepted = False
    for attempt in range(16):
        with torch.no_grad():
            for p, before, after in zip(parameters, old_parameters, proposal, strict=True):
                p.copy_(torch.lerp(before, after, fraction))
        metrics = controller_kl(policy, old_transition, old_gate, records, batch_size)
        bound = max(metrics['max_gaussian_kl'], metrics['max_joint_kl'])
        if math.isfinite(bound) and bound <= max_kl:
            accepted = True
            break
        # A full affine Gaussian proposal has quadratic KL. Verify rather than assume
        # the resulting step also satisfies the Bernoulli/mixture constraint.
        fraction *= min(0.5, math.sqrt(max_kl/bound)*0.95) if math.isfinite(bound) and bound > 0 else 0.5
    if not accepted:
        with torch.no_grad():
            for p, before in zip(parameters, old_parameters, strict=True):
                p.copy_(before)
        optimizer.load_state_dict(old_optimizer)
        raise RuntimeError('no KL-feasible controller proposal; state restored')
    delta_norm = torch.sqrt(sum((p.detach().double()-before.double()).square().sum()
                           for p, before in zip(parameters, old_parameters, strict=True))).item()
    optimizer.zero_grad(set_to_none=True)
    noise_metrics = {'sampled_gaussian_actions': gaussian_actions}
    if noise_components:
        squared_noise = noise_sum_squares.item()
        second_moment = squared_noise / noise_components / policy.transition.component_std**2
        noise_metrics.update(
            sampled_epsilon_second_moment=second_moment,
            sampled_epsilon_rms=math.sqrt(second_moment),
            sampled_noise_component_rms=math.sqrt(squared_noise / noise_components),
            sampled_noise_vector_rms=math.sqrt(squared_noise / gaussian_actions),
        )
    return {**metrics, **noise_metrics, 'controller_loss': loss_value, 'behavior_logprob_max_error': max_drift,
            'accepted_step_fraction': fraction, 'backtracks': attempt,
            'controller_parameter_delta_norm': delta_norm}


@torch.no_grad()
def refresh_critic(critic, records, token_budget):
    critic.eval()
    updated = list(records)
    plan = plan_replay_microbatches(records, list(range(len(records))),
                                   token_budget=token_budget, max_trajectories=16)
    for indices in plan:
        batch = collate_replay_microbatch(records, indices, pad_token_id=0, device=torch.device('cuda'))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hidden = _replay_hidden(critic, batch)
            values = critic.values(hidden[batch.action_batch_indices, batch.action_positions]).float().cpu()
        offset = 0
        for index in indices:
            record = records[index]
            updated[index] = replace(record, advantages=(
                (1.0 if record.correct else -1.0) - values[offset:offset+record.response_length]
            ))
            offset += record.response_length
    return updated


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', default='postraining/data/dapo-math-17k.parquet')
    p.add_argument('--bridge', type=Path)
    p.add_argument('--eval-data', type=Path)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--train-seconds', type=float, default=1200)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--warmup-steps', type=int, default=10)
    p.add_argument('--prompts-per-rollout', type=int, default=4)
    p.add_argument('--samples-per-prompt', type=int, default=16)
    p.add_argument('--physical-batch-size', type=int, default=64)
    p.add_argument('--sampling', choices=('independent', 'antithetic'), default='independent',
                   help='Training sampler; heldout evaluation always uses independent stock sampling.')
    p.add_argument('--top-k', type=int,
                   help='Answer sampling support; new runs use the qualified default, resume inherits checkpoint (legacy 20). Explicit eval-only overrides are recorded.')
    p.add_argument('--head-bucket-size', type=int, default=16)
    p.add_argument('--gate-probability', type=float, default=1/64)
    p.add_argument('--thought-sigma', type=float, default=1.0)
    p.add_argument('--normalize-thought-input', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--controller-lr', type=float, default=1e-3)
    p.add_argument('--critic-lr', type=float, default=1e-5)
    p.add_argument('--controller-batch-size', type=int, default=512)
    p.add_argument('--max-controller-kl', type=float, default=0.02)
    p.add_argument('--tensorboard-dir', type=Path)
    p.add_argument('--skip-evaluation', action='store_true',
                   help='Profile a resumed training boundary without initial, periodic, or final evaluation.')
    p.add_argument('--eval-prompts', type=int, default=32)
    p.add_argument('--eval-every-seconds', type=float, default=600)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--resume', type=Path)
    return p


def validate_resume_configuration(saved, args):
    mutable = {'output', 'resume', 'train_seconds', 'steps', 'eval_every_seconds',
               'eval_only', 'tensorboard_dir', 'skip_evaluation', 'head_bucket_size'}
    if args.eval_only:
        mutable.update({'eval_data', 'eval_prompts', 'prompts_per_rollout', 'samples_per_prompt', 'physical_batch_size'})
    # The coupling mode is immutable; an explicit eval-only top-k override is
    # recorded separately and never rewrites the checkpoint's training policy.
    prior_top_k = saved['args'].get('top_k', 20)
    if args.top_k is None:
        args.top_k = prior_top_k
    if not args.eval_only and args.top_k != prior_top_k:
        raise ValueError(
            f'controller checkpoint used top-k={prior_top_k}; resume with --top-k {prior_top_k} '
            'or start a new output without --resume to change the sampling policy'
        )
    for key, value in vars(args).items():
        if key in mutable or key == 'top_k' or (key == 'sampling' and key not in saved['args']):
            continue
        if saved['args'][key] != value:
            raise ValueError(f'controller checkpoint configuration differs: {key}')


@contextmanager
def evaluation_sampling(engine, sampling):
    """Isolate heldout RNG and restore the training sampler even on failure."""
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    try:
        with engine.stock_sampling() if sampling == 'antithetic' else nullcontext():
            yield
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)


def main():
    args = parser().parse_args()
    saved = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if saved is not None:
        validate_resume_configuration(saved, args)
    elif args.top_k is None:
        args.top_k = DEFAULT_TOP_K
    sampling_contract = {
        'training_top_k': saved['args'].get('top_k', 20) if saved else args.top_k,
        'evaluation_top_k': args.top_k,
        'evaluation_override': bool(saved and args.eval_only and args.top_k != saved['args'].get('top_k', 20)),
        'filter': 'legacy_exact_k_descending_nucleus',
    }
    if args.train_seconds <= 0 or args.max_controller_kl <= 0:
        raise ValueError('positive duration and KL budget required')
    if not 1 <= args.top_k <= MINICPM5_SPEC.vocab_size:
        raise ValueError('top-k must be positive and fit the MiniCPM vocabulary')
    if args.eval_prompts % args.prompts_per_rollout:
        raise ValueError('evaluation must contain complete prompt groups')
    if args.skip_evaluation and (args.eval_only or not args.resume):
        raise ValueError('evaluation skipping is only for resumed profiling')
    if (args.sampling == 'antithetic' and not args.eval_only
            and (args.prompts_per_rollout, args.samples_per_prompt, args.physical_batch_size) != (4, 16, 64)):
        raise ValueError('antithetic training requires 4 prompts x 16 samples and physical batch size 64')
    if args.output.exists() and (not args.resume or args.eval_only):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # Heads are small fp32 controllers; bf16 trunk tensorcore arithmetic stays unchanged.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    bridge_payload = torch.load(args.bridge, map_location='cpu', weights_only=False) if args.bridge else None
    if bridge_payload and bridge_payload['schema'] != 'minicpm_hidden_to_input_ridge_experiment/v1':
        raise ValueError('unsupported bridge checkpoint')
    excluded = {item for split in bridge_payload['splits'].values()
                for item in split['prompt_sha256']} if bridge_payload else set()
    all_rows = [row for row in load_unique_math_rows(args.data) if digest(row['prompt']) not in excluded]
    random.Random(args.seed).shuffle(all_rows)
    if args.eval_data:
        evaluation = load_unique_math_rows(args.eval_data)
        random.Random(args.seed).shuffle(evaluation)
        evaluation = evaluation[:args.eval_prompts]
        rows = all_rows
    else:
        evaluation, rows = all_rows[:args.eval_prompts], all_rows[args.eval_prompts:]
    if len(evaluation) != args.eval_prompts:
        raise ValueError('insufficient evaluation prompts')
    excluded.update(digest(row['prompt']) for row in evaluation)
    rows = [row for row in rows if digest(row['prompt']) not in excluded]
    data_sha = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    policy, tokenizer = VAPOPolicy.from_family("minicpm5", 
        device=torch.device('cuda'), lora_config=LoRAConfig(), gradient_checkpointing=False,
        latent_thinking=True, thought_sigma=args.thought_sigma,
        init_stop_thinking_probability=args.gate_probability,
    )
    if bridge_payload and (bridge_payload['model_id'], bridge_payload['revision']) != (policy.model_id, policy.revision):
        raise ValueError('bridge and actor base revisions differ')
    excluded_encoded = {item for split in bridge_payload['splits'].values()
                        for item in split['encoded_prompt_sha256']} if bridge_payload else set()
    excluded_encoded.update(digest(encode_math_prompt(tokenizer, row, prompt_tokens=1024,
                            enable_thinking=True).tolist()) for row in evaluation)
    encoded_rows = [(row, digest(encode_math_prompt(tokenizer, row, prompt_tokens=1024,
                     enable_thinking=True).tolist())) for row in rows]
    rows = [row for row, encoded in encoded_rows if encoded not in excluded_encoded]
    training_encoded = [encoded for _, encoded in encoded_rows if encoded not in excluded_encoded]
    if not rows:
        raise ValueError('no training prompts remain after exclusions')
    dimension = policy.causal_lm.config.hidden_size
    with torch.no_grad():
        calibration = torch.cat([encode_math_prompt(tokenizer, row, prompt_tokens=1024,
                              enable_thinking=True) for row in rows[:32]]).to('cuda')
        input_norm = policy.token_embeddings(calibration).float().norm(dim=-1).median().item()
    if saved:
        input_norm = saved['input_norm']
    policy.requires_grad_(False)
    policy.transition = WhitenedGaussianController(dimension, args.thought_sigma).to('cuda')
    policy.thinking_gate = NormalizedStopGate(dimension, args.gate_probability).to('cuda')
    if bridge_payload:
        policy.thought_adapter = RidgeThoughtInput(**bridge_payload['state_dict']).to('cuda')
    elif args.normalize_thought_input:
        policy.thought_adapter = NormalizedThoughtInput(policy.thought_adapter, input_norm, dimension)
    torch.manual_seed(args.seed+1)
    critic = VAPOCritic.from_family("minicpm5", 
        device=torch.device('cuda'), lora_config=LoRAConfig(), critic_width=256,
        gradient_checkpointing=False, shared_frozen_source=policy.causal_lm, latent_thinking=True,
    )
    if args.normalize_thought_input:
        critic.thought_adapter = NormalizedThoughtInput(critic.thought_adapter, input_norm, dimension)
    critic.nextlat_head.requires_grad_(False)
    enable_packed_replay_attention(critic.causal_lm)
    enable_replay_mlp_compilation(critic.causal_lm)
    controller_parameters = list(policy.transition.parameters()) + list(policy.thinking_gate.parameters())
    actor_optimizer = torch.optim.AdamW(controller_parameters, lr=args.controller_lr, weight_decay=0, fused=True)
    critic_optimizer = torch.optim.AdamW(
        list(critic.backbone_parameters())+list(critic.value_head.parameters()),
        lr=args.critic_lr, weight_decay=0, fused=True,
    )
    step = cursor = warmups = 0
    previous_training_seconds = 0.0
    if args.resume:
        if saved['schema'] != 'minicpm-frozen-latent-controller/v3' or saved['data_sha256'] != data_sha:
            raise ValueError('incompatible controller checkpoint')
        if saved['bridge_sha256'] != (hashlib.sha256(args.bridge.read_bytes()).hexdigest() if args.bridge else None):
            raise ValueError('bridge contents differ from checkpoint')
        previous_training_seconds = saved['total_training_seconds']
        if (saved['model_id'], saved['revision']) != (policy.model_id, policy.revision):
            raise ValueError('controller base model revision changed')
        evaluation_encoded = {digest(encode_math_prompt(tokenizer, row, prompt_tokens=1024,
                              enable_thinking=True).tolist()) for row in evaluation}
        if evaluation_encoded.intersection(saved['training_encoded_prompt_sha256']):
            raise ValueError('evaluation overlaps checkpoint training prompts')
        if not args.eval_only and training_encoded != saved['training_encoded_prompt_sha256']:
            raise ValueError('controller training split changed')
        policy.transition.load_state_dict(saved['transition'])
        policy.thinking_gate.load_state_dict(saved['thinking_gate'])
        load_adapter_state_dict(critic.causal_lm, saved['critic']['adapter'])
        critic.value_head.load_state_dict(saved['critic']['value_head'])
        critic.thought_adapter.load_state_dict(saved['critic']['thought_adapter'])
        actor_optimizer.load_state_dict(saved['actor_optimizer'])
        critic_optimizer.load_state_dict(saved['critic_optimizer'])
        step, cursor, warmups = saved['step'], saved['cursor'], saved['warmups']
    stops = _stop_ids(policy, tokenizer)
    engine_type = PairedControllerRolloutEngine if args.sampling == 'antithetic' else MiniCPMLatentRolloutEngine
    engine = engine_type(
        policy, stop_ids=stops,
        thinking_start_token_id=resolve_thinking_start_token(tokenizer, stop_ids=stops),
        thinking_end_token_id=resolve_thinking_end_token(tokenizer, stop_ids=stops),
        prompts_per_rollout=args.prompts_per_rollout, samples_per_prompt=args.samples_per_prompt,
        physical_batch_size=args.physical_batch_size, cache_length=11024,
        head_bucket_size=args.head_bucket_size,
        temperature=0.9, top_k=args.top_k, top_p=0.95, answer_reserve_tokens=1024,
    )
    source_paths = [Path(__file__), Path('postraining/vapo/policy.py'),
                    Path('postraining/minicpm_latent_rollout.py'), Path('postraining/train_minicpm_vapo.py')]
    if args.sampling == 'antithetic':
        source_paths.extend([Path('postraining/minicpm_paired_rollout.py'),
                             Path('postraining/runtime/coupled_rng.py')])
    manifest = {
        'schema': 'minicpm-frozen-latent-controller/v3', 'args': vars(args), 'data_sha256': data_sha,
        'input_embedding_median_norm': input_norm, 'training_rows': len(rows), 'heldout_rows': len(evaluation),
        'architecture': 'frozen bf16 actor dynamics and answer decoder; trainable whitened fp32 controller; independent trainable raw-action critic',
        'return_contract': 'terminal +1 correct / -1 incorrect; pre-update critic Monte Carlo baseline; sum scores per trajectory',
        'max_new_tokens': 10000, 'answer_reserve_tokens': 1024,
        'evaluation_scope': str(args.eval_data) if args.eval_data else 'fixed DAPO prompt holdout excluded from this experiment training, not an external benchmark',
        'evaluation_sha256': hashlib.sha256(args.eval_data.read_bytes()).hexdigest() if args.eval_data else None,
        'bridge_sha256': hashlib.sha256(args.bridge.read_bytes()).hexdigest() if args.bridge else None,
        'training_prompt_sha256': [digest(row['prompt']) for row in rows],
        'evaluation_prompt_sha256': [digest(row['prompt']) for row in evaluation],
        'bridge_and_evaluation_exclusion': 'raw and encoded prompts excluded from controller training',
        'sources': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        'previous_training_seconds': previous_training_seconds,
        'sampling_contract': sampling_contract,
    }
    (args.output/'manifest.json').write_text(json.dumps(manifest, default=str, indent=2)+'\n')
    tensorboard_dir = args.tensorboard_dir or Path('postraining/runs') / args.output.name / 'tensorboard'
    tensorboard = SummaryWriter(str(tensorboard_dir), flush_secs=2)
    atexit.register(tensorboard.close)
    tensorboard.add_text('samples/configuration', json.dumps(manifest, default=str, indent=2), step)
    tensorboard.flush()
    metrics_file = (args.output/'metrics.jsonl').open('a')
    last_save = time.monotonic()
    training_seconds = 0.0
    last_evaluated_step = None

    def log(phase, metrics):
        record = {'phase': phase, 'step': step, 'warmups': warmups, 'time': time.time(), **metrics}
        write_controller_metrics(tensorboard, record)
        line = json.dumps(record, allow_nan=False)
        metrics_file.write(line+'\n')
        metrics_file.flush()
        print(line, flush=True)

    def save():
        nonlocal last_save
        engine.release_cache()
        atomic_torch_save({
            'schema': manifest['schema'], 'data_sha256': data_sha, 'args': vars(args),
            'transition': policy.transition.state_dict(), 'thinking_gate': policy.thinking_gate.state_dict(),
            # NextLat is frozen and never executed by this controller experiment.
            'critic': {
                'adapter': adapter_state_dict(critic.causal_lm),
                'value_head': {name: value.detach().cpu() for name, value in critic.value_head.state_dict().items()},
                'thought_adapter': {name: value.detach().cpu() for name, value in critic.thought_adapter.state_dict().items()},
            }, 'input_norm': input_norm,
            'actor_optimizer': actor_optimizer.state_dict(), 'critic_optimizer': critic_optimizer.state_dict(),
            'step': step, 'cursor': cursor, 'warmups': warmups,
            'bridge_sha256': manifest['bridge_sha256'],
            'bridge': bridge_payload,
            'model_id': policy.model_id, 'revision': policy.revision,
            'total_training_seconds': previous_training_seconds + training_seconds,
            'training_encoded_prompt_sha256': training_encoded,
            'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(), 'python_rng': random.getstate(),
        }, args.output/'checkpoint.pt')
        last_save = time.monotonic()

    def rollout(selected):
        policy.eval()
        result = collect_rollouts(engine, tokenizer, selected, prompt_tokens=1024,
                                  max_new_tokens=10000, enable_thinking=True)
        diagnostics = rollout_diagnostics(result, samples_per_prompt=args.samples_per_prompt, stop_ids=stops)
        diagnostics.update(component_std=policy.transition.component_std,
                           vector_sigma=policy.transition.vector_sigma)
        engine.release_cache()
        return result.records, diagnostics

    def evaluate():
        nonlocal last_evaluated_step
        started = time.monotonic()
        correct = total = 0
        transcript = []
        with evaluation_sampling(engine, args.sampling):
            for offset in range(0, len(evaluation), args.prompts_per_rollout):
                torch.manual_seed(args.seed+100000+offset)
                records, diagnostics = rollout(evaluation[offset:offset+args.prompts_per_rollout])
                correct += sum(record.correct for record in records)
                total += len(records)
                log('evaluation_group', {'prompt_offset': offset, **diagnostics})
                transcript.extend({'prompt_index': offset+i//args.samples_per_prompt,
                                   'correct': r.correct, 'text': r.text, 'thoughts': r.latent_vectors.size(0)}
                                  for i,r in enumerate(records))
        summary = {'accuracy': correct/total, 'correct': correct, 'trajectories': total,
                   'evaluation_seconds': time.monotonic()-started}
        (args.output/f'evaluation_{step}.json').write_text(json.dumps({'summary': summary, 'sampling_contract': sampling_contract, 'responses': transcript}, indent=2)+'\n')
        log('evaluation', summary)
        last_evaluated_step = step
        return summary

    if saved is not None:
        # Replica construction consumes RNG too; restore only after all setup.
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
    del saved

    if not args.skip_evaluation:
        evaluate()
    if args.eval_only:
        metrics_file.close()
        tensorboard.close()
        return
    while warmups < args.warmup_steps:
        selected = [rows[(cursor+i)%len(rows)] for i in range(args.prompts_per_rollout)]
        cursor += args.prompts_per_rollout
        records, diagnostics = rollout(selected)
        if warmups == 0:
            # Calibrate a constant return prior on training data before any actor
            # update. Critic replay targets use the same +1/-1 reward convention.
            prior = sum((1.0 if r.correct else -1.0) * r.response_length for r in records)
            prior /= sum(r.response_length for r in records)
            with torch.no_grad():
                critic.value_head.output.bias.fill_(prior)
            diagnostics['critic_initial_reward_prior'] = prior
        critic_metrics = update_step(policy, critic, records, actor_optimizer, critic_optimizer,
            optimizer_minibatches=4, replay_token_budget=11024, replay_max_trajectories=16,
            logit_chunk_tokens=128, clip_low=0.2, clip_high=0.28, value_coefficient=1.0,
            nextlat_horizon=2, nextlat_samples=64, nextlat_mse_coefficient=1.0,
            nextlat_kl_coefficient=1.0, nextlat_kl_chunk_tokens=16, train_nextlat=False,
            grad_clip_norm=1.0, value_only=True)
        warmups += 1
        log('warmup', {**diagnostics, **critic_metrics})
        if time.monotonic()-last_save >= 300:
            save()
    last_evaluation = time.monotonic()
    while step < args.steps and training_seconds < args.train_seconds:
        cycle = time.monotonic()
        selected = [rows[(cursor+i)%len(rows)] for i in range(args.prompts_per_rollout)]
        cursor += args.prompts_per_rollout
        records, diagnostics = rollout(selected)
        records = refresh_critic(critic, records, 11024)
        controller_metrics = controller_update(policy, actor_optimizer, records,
            batch_size=args.controller_batch_size, max_kl=args.max_controller_kl)
        critic_metrics = update_step(policy, critic, records, actor_optimizer, critic_optimizer,
            optimizer_minibatches=4, replay_token_budget=11024, replay_max_trajectories=16,
            logit_chunk_tokens=128, clip_low=0.2, clip_high=0.28, value_coefficient=1.0,
            nextlat_horizon=2, nextlat_samples=64, nextlat_mse_coefficient=1.0,
            nextlat_kl_coefficient=1.0, nextlat_kl_chunk_tokens=16, train_nextlat=False,
            grad_clip_norm=1.0, value_only=True)
        step += 1
        cycle_seconds = time.monotonic()-cycle
        training_seconds += cycle_seconds
        log('train', {**diagnostics, **critic_metrics, **controller_metrics,
                      'cycle_seconds': cycle_seconds, 'elapsed_training_seconds': training_seconds,
                      'total_training_seconds': previous_training_seconds + training_seconds})
        if time.monotonic()-last_save >= 300:
            save()
        if not args.skip_evaluation and time.monotonic()-last_evaluation >= args.eval_every_seconds:
            save()
            evaluate()
            last_evaluation = time.monotonic()
    save()
    if not args.skip_evaluation and last_evaluated_step != step:
        evaluate()
    metrics_file.close()
    tensorboard.close()


if __name__ == '__main__':
    main()

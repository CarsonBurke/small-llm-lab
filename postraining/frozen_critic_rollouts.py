"""Frozen-policy data collection with exact memory-carry branch interventions.

Invoke only inside an mlq job. Prefixes after the original prompt are forced
through the compiled decode tick; treating those tokens as prompt prefill would
silently change the memory policy being studied.
"""
from __future__ import annotations

import gc
import hashlib
import random
from typing import Any

import torch

from postraining.core import encode_prompt, load_posttraining_tokenizer
from postraining.graph_decode import PinnedDecodeArena
from postraining.latent_rollout import emitted_token_rows, trajectory_token_seeds, split_rollout_groups
from postraining.latent_thought import LatentThoughtModel, combiner_init_kwargs_from_checkpoint
from postraining.math_prompt import canonicalize_answer_fence_rows
from postraining.model_io import load_model
from postraining.train_latent_vapo import score_math_rollout
from postraining.train_vapo import prompt_text
from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters


def reasoning_positions(tokens: list[int], prompt_length: int, think_open: int,
                        think_close: int, *, forbidden_ids: tuple[int, ...] = ()) -> list[int]:
    """Pre-action indices whose next token is strictly inside <think>."""
    generated = tokens[prompt_length:]
    try:
        start = generated.index(think_open)
    except ValueError:
        return []
    end = next((i for i in range(start + 1, len(generated))
                if generated[i] in {think_open, think_close, *forbidden_ids}), len(generated))
    return [prompt_length + i - 1 for i in range(start + 1, end)]


class UninformativeBranch(RuntimeError):
    """A preregistered action-only rejection, before any continuation label."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class _BranchArena(PinnedDecodeArena):
    """Teacher forcing around the ordinary captured tick, without new kernels."""
    def configure(self, prefix: list[int], taken: int | None, repeats: int, *,
                  require_distinct: bool = False, forbidden_tokens: tuple[int, ...] = ()) -> None:
        self.require_distinct = require_distinct
        self.forbidden_tokens = forbidden_tokens
        self.forced_prefix = prefix
        self.taken = taken
        self.repeats = repeats
        self.prefix_cursor = 0
        self.branch_tokens = None
        self.branch_probabilities = None
        self._forced_device = torch.tensor(prefix, device=self.device, dtype=torch.long)

    def _run(self, rows: int) -> None:
        index = self.prefix_cursor
        if index < len(self.forced_prefix):
            self.pending[:rows].copy_(self._forced_device[index].expand(rows))
        elif index == len(self.forced_prefix):
            # Both action draws are fresh and independent under pi. The
            # source token selected the state, never either treatment arm.
            assert self.policy_logits is not None
            self.branch_probabilities = torch.softmax(
                self.policy_logits[0] / self.temperature, -1).clone()
            first = (self.pending[0].clone() if self.taken is None
                     else torch.tensor(self.taken, device=self.device))
            second = self.pending[self.repeats].clone()
            self.branch_tokens = torch.stack((first, second))
            if self.require_distinct:
                # One host synchronization at the intervention point. No
                # branch token has been consumed and no reward or Q was read.
                drawn = self.branch_tokens.cpu().tolist()
                if any(token in self.forbidden_tokens for token in drawn):
                    raise UninformativeBranch('nonreasoning_action_pairs')
                if drawn[0] == drawn[1]:
                    raise UninformativeBranch('identical_action_pairs')
            arm = torch.div(self.live_rows[:rows], self.repeats,
                            rounding_mode='floor')
            self.pending[:rows].copy_(torch.where(
                arm < 2, self.branch_tokens[arm.clamp_max(1)], self.pending[:rows]))
        self.prefix_cursor += 1
        super()._run(rows)


class FrozenPolicy:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError('frozen critic collection requires CUDA; CPU fallback is forbidden')
        self.device = torch.device('cuda')
        self.args = args
        self.branch_position_candidates = getattr(args, 'branch_position_candidates', None)
        self.distinct_action_pairs = bool(getattr(args, 'distinct_action_pairs', False))
        if self.branch_position_candidates is not None and self.branch_position_candidates < 1:
            raise ValueError('branch_position_candidates must be positive')
        self.temperature = float(getattr(args, 'temperature', 1.0))
        self.top_p = float(getattr(args, 'top_p', 1.0))
        if self.top_p != 1.0:
            raise ValueError('study needs the full policy, top_p must be 1')
        payload = torch.load(args.actor_init, map_location='cpu', weights_only=False)
        saved = payload.get('args', {})
        if not saved.get('think_tokens') or not saved.get('answer_fence'):
            raise ValueError('study actor must use think and answer fences')
        from postraining.train_latent_vapo import file_sha256
        if payload.get('base_checkpoint_sha256') != file_sha256(args.checkpoint):
            raise ValueError('actor init does not descend from the supplied base checkpoint')
        self.backbone = load_model(args.checkpoint, self.device)
        self.wrapper = LatentThoughtModel(
            self.backbone, **combiner_init_kwargs_from_checkpoint(payload)).to(self.device)
        self.wrapper.load_state_dict(payload['model'], strict=True)
        self.wrapper.eval().requires_grad_(False)
        self.model = self.wrapper
        if not self.wrapper.memory_carry:
            raise ValueError('this study expects the approved memory-carry actor')
        self.tokenizer = load_posttraining_tokenizer(
            self.backbone.architecture, FreshHyperparameters.tokenizer_path,
            think_tokens=True, answer_tokens=True,
            tokenizer_provenance=self.backbone.model_config.get('tokenizer_provenance'))
        self.stop_ids = tuple(dict.fromkeys(x for x in
                              (self.tokenizer.eos_id(), self.tokenizer.bos_id()) if x >= 0))
        self.prompt_budget = int(args.prompt_tokens)
        self.max_new_tokens = int(args.continuation_tokens)
        self.capacity = int(getattr(args, 'rollout_batch_size', 128))
        self.min_think_tokens = int(saved.get('think_min_tokens', 1))
        self.progress_callback = None
        self.completed_pair_ids = set()
        self.collection_stats = {'branch_states': 0, 'nonreasoning_action_pairs': 0,
                                 'identical_action_pairs': 0, 'retained_branch_pairs': 0,
                                 'no_usable_reasoning_prompts': 0, 'resumed_pairs_skipped': 0,
                                 'attempted_branch_positions': 0, 'prompts_with_retained_pairs': 0,
                                 'prompts_exhausted_before_quota': 0}
        self._arena = None
        self._branch_arena = None
        self.provenance = {'checkpoint': str(args.checkpoint), 'actor_init': str(args.actor_init),
                           'actor_init_sha256': file_sha256(args.actor_init),
                           'temperature': self.temperature, 'top_p': self.top_p,
                           'memory_prefix_replay': 'original_prompt_then_forced_compiled_ticks',
                           'advantage_reference': 'K_independent_unforced_policy_continuations',
                           'branch_position_candidates': self.branch_position_candidates,
                           'distinct_action_pairs': self.distinct_action_pairs,
                           'branch_selection': 'random_positions_then_first_distinct_interior_pi_draws_before_labels'
                               if self.distinct_action_pairs else 'uniform_positions_unfiltered_pi_draws'}
        # KDA chunk prefill intentionally graph-breaks; compile with fullgraph
        # disabled, as in the trainer. Decode itself remains fullgraph+captured.
        self.wrapper.prefill_core = torch.compile(self.wrapper.prefill_core, fullgraph=False,
                                                 dynamic=True)
        del payload

    def _make_arena(self, branch: bool = False):
        cls = _BranchArena if branch else PinnedDecodeArena
        return cls(self.wrapper, rows=self.capacity,
                   kv_width=self.prompt_budget + self.max_new_tokens,
                   temperature=self.temperature, top_p=self.top_p,
                   stop_ids=self.stop_ids, device=self.device, hidden_carry=True,
                   compile=True, cuda_graphs=True, replay_storage=False,
                   record_policy_logits=branch)

    def _collect(self, prompts: list[list[int]], repeats: int, seed: int,
                 branch: bool = False):
        attr = '_branch_arena' if branch else '_arena'
        arena = getattr(self, attr)
        if arena is None:
            arena = self._make_arena(branch)
            setattr(self, attr, arena)
        width = max(map(len, prompts))
        ids = torch.zeros((len(prompts), width), device=self.device, dtype=torch.long)
        for index, tokens in enumerate(prompts):
            ids[index, width-len(tokens):] = torch.tensor(tokens, device=self.device)
        batch = arena.rollout(
            ids, torch.tensor([len(p) for p in prompts]), prompt_repeats=repeats,
            max_new_tokens=self.max_new_tokens, max_stream_steps=self.max_new_tokens,
            token_seeds=trajectory_token_seeds(seed, range(len(prompts)), repeats, self.device),
            offload_device=torch.device('cpu'))
        return batch

    @torch.inference_mode()
    def screen(self, rows: list[dict], samples: int, seed: int) -> list[dict[str, Any]]:
        """Fresh independent rollouts, usable for screening OR fitting, never both."""
        if samples < 1 or samples > self.capacity:
            raise ValueError('samples must lie within rollout arena capacity')
        output = []
        chunk_size = self.capacity // samples
        for chunk_start in range(0, len(rows), chunk_size):
            sources = rows[chunk_start:chunk_start + chunk_size]
            canonical = canonicalize_answer_fence_rows(sources)
            prompts = [encode_prompt(self.tokenizer, prompt_text(row), self.prompt_budget)
                       for row in canonical]
            combined = self._collect(prompts, samples, seed + chunk_start * 104729)
            groups = split_rollout_groups(
                combined, samples, torch.tensor([len(p) for p in prompts]).repeat_interleave(samples))
            for source, row, prompt, batch in zip(sources, canonical, prompts, groups, strict=True):
                truth = str(row['reward_model']['ground_truth'])
                self._score(batch, truth)
                prompt_id = str(row.get('prompt_id') or hashlib.sha256(
                    prompt_text(row).encode()).hexdigest())
                gold_context = self.tokenizer.encode('\nGold answer: ' + truth + '\n')
                for sample_index, generated in enumerate(emitted_token_rows(batch)):
                    full = prompt + generated
                    output.append({'prompt_id': prompt_id, 'split': source.get('split'),
                                   'prompt_tokens': prompt, 'prompt_length': len(prompt),
                                   'generated_tokens': generated, 'token_ids': full,
                                   'reasoning_positions': reasoning_positions(
                                       full, len(prompt), self.tokenizer.think_open_id,
                                       self.tokenizer.think_close_id,
                                       forbidden_ids=(self.tokenizer.answer_open_id,
                                                      self.tokenizer.answer_close_id, *self.stop_ids)),
                                   'reward': float(batch.reward_scalar[sample_index]),
                                   'gold': truth, 'gold_context_ids': gold_context,
                                   'gold_context_tokens': gold_context, 'sample_index': sample_index})
        return output

    def release_arenas(self):
        self._arena = None
        self._branch_arena = None
        gc.collect()
        torch.cuda.empty_cache()

    rollouts = screen

    def _score(self, batch, gold):
        score_math_rollout(batch, gold, self.tokenizer, self.stop_ids,
                           nearby_reward_max=0.0,
                           think_fence_ids=(self.tokenizer.think_open_id,
                                            self.tokenizer.think_close_id),
                           answer_fence_ids=(self.tokenizer.answer_open_id,
                                             self.tokenizer.answer_close_id),
                           min_think_tokens=self.min_think_tokens)

    @torch.inference_mode()
    def branches(self, records: list[dict], positions_per_prompt: int, K: int,
                 seed: int) -> list[dict]:
        if K < 1 or 3 * K > self.capacity:
            raise ValueError('branch K must fit two action arms and an independent policy reference')
        rng = random.Random(seed)
        seen = set()
        output = []
        for record_index, record in enumerate(records):
            if record['prompt_id'] in seen:
                continue
            eligible = record['reasoning_positions']
            if not eligible:
                continue
            seen.add(record['prompt_id'])
            candidate_count = self.branch_position_candidates or positions_per_prompt
            positions = rng.sample(eligible, min(candidate_count, len(eligible)))
            if self.branch_position_candidates is None:
                positions.sort()  # Preserve the original protocol's ordering.
            accepted = 0
            for position in positions:
                if accepted >= positions_per_prompt:
                    break
                pair_id = f"{record['prompt_id']}:{position}"
                if pair_id in self.completed_pair_ids:
                    self.collection_stats['resumed_pairs_skipped'] += 1
                    accepted += 1
                    continue
                if self._branch_arena is None:
                    self._branch_arena = self._make_arena(True)
                prompt = record['prompt_tokens']
                forced = record['token_ids'][len(prompt):position + 1]
                forbidden = (self.tokenizer.think_open_id, self.tokenizer.think_close_id,
                             self.tokenizer.answer_open_id, self.tokenizer.answer_close_id,
                             *self.stop_ids)
                self._branch_arena.configure(forced, None, K,
                                             require_distinct=self.distinct_action_pairs,
                                             forbidden_tokens=forbidden)
                # Two forced-action arms and one unforced independent pi reference.
                # All generated-prefix tokens pass through the same memory tick
                # as the source trajectory, preserving writes and recurrent KV.
                self.collection_stats['attempted_branch_positions'] += 1
                try:
                    batch = self._collect([prompt, prompt, prompt], K,
                                          seed + record_index * 104729 + position, True)
                except UninformativeBranch as rejection:
                    # The arena workspace context releases graphs/caches on
                    # exceptions, so the next candidate starts with fresh state.
                    self.collection_stats[rejection.reason] += 1
                    continue
                self._score(batch, record['gold'])
                arena = self._branch_arena
                tokens = arena.branch_tokens.cpu().tolist()
                probabilities = arena.branch_probabilities.cpu().tolist()
                rewards = batch.reward_scalar.tolist()
                expected = torch.tensor(forced, dtype=torch.long)
                actual = batch.token_ids[:, len(prompt):len(prompt) + len(forced)]
                if not torch.equal(actual, expected[None].expand(actual.size(0), -1)):
                    raise RuntimeError('branch teacher forcing changed the source prefix')
                self.collection_stats['branch_states'] += 1
                forbidden = {self.tokenizer.think_open_id, self.tokenizer.think_close_id,
                             self.tokenizer.answer_open_id, self.tokenizer.answer_close_id,
                             *self.stop_ids}
                if any(token in forbidden for token in tokens):
                    # Preserve the unconditioned draws; never redraw until a
                    # reasoning token appears. Report discarded coverage.
                    self.collection_stats['nonreasoning_action_pairs'] += 1
                    continue
                self.collection_stats['identical_action_pairs'] += int(tokens[0] == tokens[1])
                self.collection_stats['retained_branch_pairs'] += 1
                accepted += 1
                output.append({'prompt_id': record['prompt_id'], 'split': record.get('split'),
                               'pair_id': f"{record['prompt_id']}:{position}",
                               'prefix_ids': record['token_ids'][:position + 1],
                               'prompt_tokens': prompt, 'position': position,
                               'policy_probabilities': probabilities,
                               'reference_rewards': rewards[2*K:3*K],
                               'reference_sample_count': K,
                               'gold': record['gold'],
                               'gold_context_ids': record['gold_context_ids'],
                               'gold_context_tokens': record['gold_context_ids'],
                               'actions': [{'token_id': token,
                                            'probability': probabilities[token],
                                            'rewards': rewards[arm*K:(arm+1)*K]}
                                           for arm, token in enumerate(tokens)]})
                if self.progress_callback is not None:
                    self.progress_callback(output[-1])
            self.collection_stats['prompts_with_retained_pairs'] += int(accepted > 0)
            self.collection_stats['prompts_exhausted_before_quota'] += int(accepted < positions_per_prompt)
        self.collection_stats['no_usable_reasoning_prompts'] += (
            len({record['prompt_id'] for record in records}) - len(seen))
        return output

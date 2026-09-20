"""Exact binary speculative decoding of the frozen dynamics character teacher.

The cache excludes the last committed character, whose exact predicting context
anchors each rollout. A round appends pending + drafts in one causal target call;
only an accepted prefix and one target correction/bonus can become output.

Exactness is the Bernoulli acceptance/residual identity, not bitwise floating-point
identity between differently shaped BF16 kernels. In particular, batched target
scoring and single-character sampling can round differently near a decision
boundary. Proposal probabilities are always captured during sampling, not rescored.
"""

import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from pretraining.nanogpt_mini.bit_density import (
    PrefixBinaryHead,
    sample_prefix_code_and_logits,
)
from pretraining.nanogpt_mini.dynamics_generation import (
    _catch_up,
    _model_device,
    _nonnegative_integer,
)
from pretraining.nanogpt_mini.nanogpt_mini_dynamics_model import DynamicsGPT


@torch.compile(dynamic=True, fullgraph=True)
def acceptance_mask(
    draft_codes: Tensor,
    proposal_logits: Tensor,
    target_logits: Tensor,
    uniforms: Tensor,
    temperature: float,
) -> Tensor:
    """Return per-bit decisions; only the prefix before the FIRST false is usable."""
    if temperature == 0:
        return draft_codes.bool() == (target_logits > 0)
    sign = 2 * draft_codes.float() - 1
    log_q = F.logsigmoid(sign * proposal_logits.float() / temperature)
    log_p = F.logsigmoid(sign * target_logits.float() / temperature)
    log_ratio = torch.minimum(log_p - log_q, torch.zeros_like(log_p))
    # Log comparison avoids underflow in a tiny, but nonzero, acceptance ratio.
    # In particular U=0 accepts positive mass, but not genuinely zero target mass.
    return uniforms.float().log() < log_ratio


@torch.compile(dynamic=True, fullgraph=True)
def correct_code(
    head: PrefixBinaryHead,
    context: Tensor,
    draft_code: Tensor,
    rejected_bit: int,
    uniforms: Tensor,
    temperature: float,
) -> Tensor:
    """Keep accepted bits, flip the rejected Bernoulli, sample its TARGET suffix."""
    positions = torch.arange(draft_code.shape[-1], device=draft_code.device)[None]
    fixed_code = torch.where(positions == rejected_bit, 1 - draft_code, draft_code)
    # For two outcomes, positive (p-q) residual mass lies entirely on the other
    # bit. Sampling from p instead at the rejected bit would bias the result.
    return sample_prefix_code_and_logits(
        head,
        context,
        uniforms,
        temperature,
        prefix_code=fixed_code,
        prefix_mask=positions <= rejected_bit,
    )[0]


@torch.compile(dynamic=True, fullgraph=True)
def _propose(
    model: DynamicsGPT,
    anchor: Tensor,
    pending: Tensor,
    uniforms: Tensor,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Roll out a nonempty budget without computing the learned error predictor."""
    context, previous = anchor, pending
    codes, logits = [], []
    for index in range(uniforms.shape[0]):
        context = model.transition(context, previous)
        previous, probability_logits = sample_prefix_code_and_logits(
            model.head, context, uniforms[index : index + 1], temperature
        )
        codes.append(previous)
        logits.append(probability_logits)
    return torch.cat(codes, dim=0), torch.cat(logits, dim=0)


@torch.compile(dynamic=True, fullgraph=True)
def _verify_proposals(
    model: DynamicsGPT,
    contexts: Tensor,
    draft_codes: Tensor,
    proposal_logits: Tensor,
    uniforms: Tensor,
    temperature: float,
) -> Tensor:
    # All known prefixes are scored in ONE batched head call, not K sequential
    # sampling calls. The backbone has already verified all contexts together.
    target_logits = model.score(contexts, draft_codes)
    accepted = acceptance_mask(
        draft_codes, proposal_logits, target_logits, uniforms, temperature
    ).flatten()
    positions = torch.arange(accepted.numel(), device=accepted.device)
    return torch.where(accepted, accepted.numel(), positions).amin()


@torch.compile(dynamic=True, fullgraph=True)
def _write_suffix(pending: Tensor, drafts: Tensor, destination: Tensor) -> None:
    destination[:, :1].copy_((2 * pending - 1).to(torch.bfloat16))
    destination[:, 1:].copy_((2 * drafts - 1).to(torch.bfloat16))


@torch.compile(dynamic=True, fullgraph=True)
def _output_addresses(drafts: Tensor, tail: Tensor, shifts: Tensor) -> Tensor:
    # Fusing concatenation and address reduction avoids materializing a second
    # floating-point code batch just to transfer the committed addresses to CPU.
    codes = torch.cat((drafts, tail), dim=0).to(torch.int64)
    return torch.bitwise_left_shift(codes, shifts).sum(dim=-1)


def _random_tapes(
    device: torch.device,
    characters: int,
    budget: int,
    bits: int,
    temperature: float,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    rounds = characters - 1
    draft_shape = (rounds, budget, bits)
    if temperature == 0:
        # Greedy graphs ignore these values. Expanded storage still communicates
        # proposal lengths to the compiled rollout without allocating O(N*L*K).
        unused = torch.empty((1, bits), device=device)
        return (
            unused.expand(characters, bits),
            unused[None].expand(draft_shape),
            unused[None].expand(draft_shape),
        )
    seed_mask = (1 << 64) - 1
    target_rng = torch.Generator(device=device).manual_seed(seed)
    proposal_rng = torch.Generator(device=device).manual_seed(
        (seed ^ 0x9E3779B97F4A7C15) & seed_mask
    )
    acceptance_rng = torch.Generator(device=device).manual_seed(
        (seed ^ 0xD1B54A32D192ED03) & seed_mask
    )
    return (
        torch.rand((characters, bits), device=device, generator=target_rng),
        torch.rand(draft_shape, device=device, generator=proposal_rng),
        torch.rand(draft_shape, device=device, generator=acceptance_rng),
    )


@torch.inference_mode()
def generate(
    model: DynamicsGPT,
    prompt_ids: list[int],
    max_new_characters: int,
    temperature: float = 0.8,
    seed: int = 1337,
    draft_tokens: int = 2,
    trace: bool = False,
) -> dict:
    """Generate on CUDA/BF16; budget zero is the matched exact teacher baseline.

    Calls/positions count actual backbone work, including discarded verification.
    Character-category counters count committed attempts, including the first
    reserved address; ``target_sampled_characters`` also includes a sampled tail
    discarded because an earlier accepted draft was reserved. IDs exclude that
    offending address, whereas codes/addresses include it exactly once.

    Timings are synchronized and do not hide compilation. No internal warmup is
    performed: callers must separately retain the cold call and benchmark warmed
    calls. Decode includes initial teacher-head sampling; prefill is BOS + prompt.
    """
    _nonnegative_integer(max_new_characters, "max_new_characters")
    _nonnegative_integer(draft_tokens, "draft_tokens")
    if not isinstance(model, DynamicsGPT):
        raise TypeError("speculative generation requires a DynamicsGPT")
    if not isinstance(prompt_ids, list) or any(
        not isinstance(identity, int)
        or isinstance(identity, bool)
        or not 0 <= identity < model.config.vocab_size
        for identity in prompt_ids
    ):
        raise ValueError("prompt_ids must be valid integer alphabet identities")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be finite and nonnegative")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not -(1 << 63) <= seed < (1 << 64):
        raise ValueError("seed must be in the torch.Generator 64-bit seed range")
    if not isinstance(trace, bool):
        raise TypeError("trace must be a boolean")
    temperature = float(temperature)
    device = _model_device(model)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = {
        "prompt_ids": list(prompt_ids),
        "generated_ids": [],
        "generated_codes": [],
        "generated_addresses": [],
        "reserved_code": None,
        "stop_reason": "output_limit",
        "requested_characters": max_new_characters,
        "valid_generated_characters": 0,
        "draft_tokens": draft_tokens,
        "target_calls": 0,
        "target_positions": 0,
        "prefill_calls": 0,
        "prefill_positions": 0,
        "proposed_draft_characters": 0,
        "accepted_draft_characters": 0,
        "corrected_characters": 0,
        "bonus_characters": 0,
        "initial_characters": 0,
        "target_only_characters": 0,
        "target_sampled_characters": 0,
        "dynamics_steps": 0,
        "discarded_target_positions": 0,
        "initial_cache_length": 0,
        "final_cache_length": 0,
        "cache_capacity": 0,
        "setup_seconds": 0.0,
        "prefill_seconds": 0.0,
        "decode_seconds": 0.0,
        "elapsed_seconds": 0.0,
        "characters_per_second": 0.0,
        "temperature": temperature,
        "seed": seed,
        "precision": "bfloat16",
        "rate_label": "code_space",
        "timing_includes_compilation": True,
        "timing_scope": "Synchronized call; no internal warmup or compilation subtraction.",
        "semantics": (
            "Bitwise speculative acceptance with binary residual correction; target "
            "distribution invariant mathematically, subject to BF16 batch-shape "
            "rounding. Independent proposal/acceptance/target random streams. "
            "Reserved mass is neither renormalized nor resampled. Character-category "
            "counters include committed reserved attempts; IDs exclude them. Target "
            "calls/positions are decode-only actual work, including discarded suffixes. "
            "The final committed character remains pending outside the KV cache."
        ),
    }
    if trace:
        result["rounds"] = []
    if max_new_characters == 0:
        # In particular, do not prefill a prompt that cannot produce any output.
        result["elapsed_seconds"] = time.perf_counter() - started
        result["setup_seconds"] = result["elapsed_seconds"]
        return result

    bits = model.config.code_bits
    prompt_length = len(prompt_ids)
    capacity = prompt_length + max_new_characters
    budget = min(draft_tokens, max(max_new_characters - 2, 0))
    target_uniforms, proposal_uniforms, acceptance_uniforms = _random_tapes(
        device, max_new_characters, budget, bits, temperature, seed
    )
    keys, values = [], []
    for block in model.prior.blocks:
        shape = (1, block.attn.num_heads, capacity, block.attn.head_dim)
        keys.append(torch.empty(shape, dtype=torch.bfloat16, device=device))
        values.append(torch.empty(shape, dtype=torch.bfloat16, device=device))
    keys, values = tuple(keys), tuple(values)
    features = torch.empty((1, capacity, bits), dtype=torch.bfloat16, device=device)
    features[:, 0].zero_()
    if prompt_ids:
        identities = torch.tensor(prompt_ids, dtype=torch.int64, device=device)
        prompt_codes = (identities[:, None] >> model.identity_shifts) & 1
        features[:, 1 : prompt_length + 1].copy_(
            (2 * prompt_codes - 1).to(torch.bfloat16)
        )
    empty_codes = torch.empty((0, bits), dtype=torch.float32, device=device)
    cache_length = prompt_length + 1
    result["cache_capacity"] = capacity
    result["initial_cache_length"] = cache_length

    def commit(address: int) -> bool:
        result["generated_addresses"].append(address)
        result["generated_codes"].append(
            [(address >> shift) & 1 for shift in range(bits - 1, -1, -1)]
        )
        if address >= model.config.vocab_size:
            result["reserved_code"] = address
            result["stop_reason"] = "reserved_code"
            return False
        result["generated_ids"].append(address)
        return True

    torch.cuda.synchronize(device)
    prefill_started = time.perf_counter()
    result["setup_seconds"] = prefill_started - started
    # Mini sets BF16 explicitly. Autocast would instead promote RMSNorm and
    # residual states to FP32, changing the frozen teacher's arithmetic.
    with torch.autocast(device_type="cuda", enabled=False):
        anchor = _catch_up(model, features[:, :cache_length], keys, values, 0)
        result["prefill_calls"] = 1
        result["prefill_positions"] = cache_length
        torch.cuda.synchronize(device)
        decode_started = time.perf_counter()
        result["prefill_seconds"] = decode_started - prefill_started
        pending, _ = sample_prefix_code_and_logits(
            model.head, anchor, target_uniforms[:1], temperature
        )
        initial_address = int(
            _output_addresses(empty_codes, pending, model.identity_shifts).item()
        )
        result["initial_characters"] = 1
        result["target_sampled_characters"] = 1
        commit(initial_address)

        round_index = 0
        while (
            result["reserved_code"] is None
            and len(result["generated_addresses"]) < max_new_characters
        ):
            output_start = len(result["generated_addresses"])
            remaining = max_new_characters - output_start
            length = min(budget, remaining - 1)
            cache_start = cache_length
            cache_end = cache_start + length + 1
            if length:
                drafts, proposal_logits = _propose(
                    model,
                    anchor,
                    pending,
                    proposal_uniforms[round_index, :length],
                    temperature,
                )
                result["proposed_draft_characters"] += length
                result["dynamics_steps"] += length
            else:
                drafts = empty_codes
            _write_suffix(pending, drafts, features[:, cache_start:cache_end])
            contexts = _catch_up(
                model,
                features[:, cache_start:cache_end],
                keys,
                values,
                cache_start,
                return_all=True,
            )[0]
            result["target_calls"] += 1
            result["target_positions"] += length + 1
            first_rejected_bit = None
            accepted_full = length
            if length:
                rejection = int(
                    _verify_proposals(
                        model,
                        contexts[:length],
                        drafts,
                        proposal_logits,
                        acceptance_uniforms[round_index, :length],
                        temperature,
                    ).item()
                )
                if rejection < length * bits:
                    first_rejected_bit = rejection
                    accepted_full = rejection // bits
            # Index the independent target stream by the candidate's output slot.
            # The selected row has never been read; acceptance cannot depend on it.
            tail_index = output_start + accepted_full
            tail_context = contexts[accepted_full : accepted_full + 1]
            if first_rejected_bit is None:
                tail, _ = sample_prefix_code_and_logits(
                    model.head,
                    tail_context,
                    target_uniforms[tail_index : tail_index + 1],
                    temperature,
                )
                tail_kind = "bonus_characters" if length else "target_only_characters"
            else:
                tail = correct_code(
                    model.head,
                    tail_context,
                    drafts[accepted_full : accepted_full + 1],
                    first_rejected_bit % bits,
                    target_uniforms[tail_index : tail_index + 1],
                    temperature,
                )
                tail_kind = "corrected_characters"
            result["target_sampled_characters"] += 1
            addresses = _output_addresses(
                drafts[:accepted_full], tail, model.identity_shifts
            ).tolist()
            committed_drafts = 0
            for index, address in enumerate(addresses):
                if index < accepted_full:
                    result["accepted_draft_characters"] += 1
                    committed_drafts += 1
                else:
                    result[tail_kind] += 1
                if not commit(address):
                    break
            committed = len(result["generated_addresses"]) - output_start
            # Retain input positions only BEFORE the new pending character.
            # This also rolls back when an accepted draft, not the tail, is reserved.
            # Subsequent appends overwrite stale cells; _catch_up only attends :end.
            cache_length = cache_start + committed
            result["discarded_target_positions"] += cache_end - cache_length
            if committed <= accepted_full:
                # The committed reserved address was an accepted draft; even on
                # termination its input stays outside the logical cache.
                pending = drafts[committed - 1 : committed]
                anchor = contexts[committed - 1 : committed]
            else:
                pending, anchor = tail, tail_context
            if trace:
                result["rounds"].append(
                    {
                        "kind": "speculative" if length else "target_only",
                        "outcome": (
                            "corrected"
                            if first_rejected_bit is not None
                            else "accepted"
                            if length
                            else "target_only"
                        ),
                        "proposal_length": length,
                        "first_rejected_bit": first_rejected_bit,
                        "accepted_full_drafts": committed_drafts,
                        "verified_accepted_full_drafts": accepted_full,
                        "cache_start": cache_start,
                        "cache_end": cache_end,
                        "cache_committed_length": cache_length,
                        "output_start": output_start,
                        "output_count": len(result["generated_addresses"]),
                    }
                )
            round_index += 1
    torch.cuda.synchronize(device)
    finished = time.perf_counter()
    result["decode_seconds"] = finished - decode_started
    result["elapsed_seconds"] = finished - started
    result["final_cache_length"] = cache_length
    result["valid_generated_characters"] = len(result["generated_ids"])
    result["characters_per_second"] = (
        result["valid_generated_characters"] / result["decode_seconds"]
        if result["decode_seconds"] > 0
        else 0.0
    )
    return result

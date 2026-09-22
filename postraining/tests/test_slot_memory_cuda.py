"""Slot-memory CUDA validation: compiled FlexAttention replay and real rollouts.

RUN_SLOT_MEMORY_CUDA_VALIDATION=1 enables the real MiniCPM workload. Run only
inside an exclusive mlq job. These are numerical correctness checks, never a
learning-quality or throughput result.
"""

from __future__ import annotations

import gc
import json
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOT_MEMORY_CUDA_VALIDATION") != "1",
    reason="real model workload: enable explicitly inside an exclusive mlq job",
)


def _report(name, **values):
    print(json.dumps({"check": name, **values}, sort_keys=True), flush=True)


def _seed():
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)


def test_compiled_flex_replay_matches_dense_reference_with_gradients():
    from postraining.slot_memory import (
        NO_WRITE,
        SlotMemoryCombiner,
        SlotMemoryConfig,
        build_alive_table,
        replay_read,
    )

    _seed()
    device = torch.device("cuda")
    config = SlotMemoryConfig(slots=64, heads=1, head_dim=128)
    hidden_size, length = 1536, 2_500
    combiner = SlotMemoryCombiner(hidden_size, config).to(device)
    with torch.no_grad():
        combiner.output.weight.normal_(std=0.02)
        combiner.null_key.normal_()
    choices = torch.randint(NO_WRITE, config.slots, (length,), device=device)
    choices[:40] = NO_WRITE  # empty memory at the start: null-only reads
    table = build_alive_table(choices, config.slots)
    carries = torch.randn(length, hidden_size, device=device, dtype=torch.bfloat16)
    embeddings = torch.randn(length, hidden_size, device=device, dtype=torch.bfloat16)
    positions = torch.arange(length, device=device)
    reads = {}
    grads = {}
    for backend in ("dense", "flex"):
        combiner.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            read = replay_read(
                combiner, token_embeddings=embeddings, carries=carries, positions=positions,
                alive_table=table, key_choices=choices, backend=backend,
            )
            mixed = combiner.combine(embeddings, read)
        (mixed.float() * torch.linspace(-1, 1, mixed.numel(), device=device).view_as(mixed)).sum().backward()
        reads[backend] = mixed.detach().float()
        grads[backend] = {
            name: parameter.grad.detach().float().clone()
            for name, parameter in combiner.named_parameters()
        }
    error = (reads["flex"] - reads["dense"]).abs().max().item()
    assert torch.equal(reads["flex"][:40], embeddings[:40].float())
    gradient_errors = {
        name: ((grads["flex"][name] - grads["dense"][name]).norm() / grads["dense"][name].norm().clamp_min(1e-12)).item()
        for name in grads["dense"]
    }
    _report("compiled_flex_vs_dense", read_max_abs=error, gradient_relative_errors=gradient_errors)
    assert error < 5e-2, error
    assert all(value < 5e-2 for value in gradient_errors.values()), gradient_errors
    assert all(grads["dense"][name].norm() > 0 for name in grads["dense"]), "every combiner parameter must receive gradient"


def test_real_rollout_records_replay_to_the_same_joint_likelihood():
    from postraining.fast_inference import CapturedTrainingRolloutEngine
    from postraining.vapo.policy import (
        VAPOCritic,
        VAPOPolicy,
        TrajectoryRecord,
        collate_replay_microbatch,
    )
    from postraining.vapo.model.hf import (
        enable_packed_replay_attention,
        enable_replay_mlp_compilation,
    )
    from postraining.vapo.model.lora import LoRAConfig
    from postraining.slot_memory import NO_WRITE, SlotMemoryConfig
    from postraining.train_minicpm_vapo import (
        _action_logprobs,
        _replay_hidden,
        _stop_ids,
        refresh_behavior_statistics,
        resolve_thinking_end_token,
    )

    _seed()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    config = SlotMemoryConfig(slots=64, heads=1, head_dim=128)
    policy, tokenizer = VAPOPolicy.from_family("minicpm5", 
        device=device, lora_config=LoRAConfig(), gradient_checkpointing=False,
        token_carry=True, slot_memory=config,
    )
    policy.eval()
    with torch.no_grad():
        # Non-identity read so the memory actually shapes the sampled stream.
        policy.token_combiner.output.weight.normal_(std=0.02)
        policy.slot_head.projection.weight.normal_(std=0.02)
    prompts = [
        torch.tensor(tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=True, return_dict=True,
        )["input_ids"], dtype=torch.long)
        for text in ("What is 2 + 3?", "What is three times seven? Explain briefly.")
    ]
    stop_ids = _stop_ids(policy, tokenizer)
    max_new_tokens, reserve = 24, 8
    engine = CapturedTrainingRolloutEngine(
        policy, stop_ids=stop_ids, prompts_per_rollout=2, samples_per_prompt=1,
        cache_length=max(map(len, prompts)) + max_new_tokens,
        temperature=0.9, top_k=20, top_p=0.95, compile_decode=True,
        answer_reserve_tokens=reserve,
        thinking_end_token_id=resolve_thinking_end_token(tokenizer, stop_ids=stop_ids),
    )
    _seed()
    responses, logprobs, _, _, _, _ = engine.generate_prompts(prompts, max_new_tokens=max_new_tokens)
    carries = engine.last_carry_hiddens
    choices = engine.last_slot_choices
    assert carries is not None and choices is not None
    assert engine._decode_graph is not None
    assert choices.shape == (2, max_new_tokens) and bool((choices >= NO_WRITE).all()) and bool((choices < config.slots).all())
    forced_position = max_new_tokens - reserve - 1
    thinking_end = engine.thinking_end_token_id
    records = []
    for row, prompt in enumerate(prompts):
        response = responses[row]
        stopping = [i + 1 for i, token in enumerate(response.tolist()) if token in stop_ids]
        length = stopping[0] if stopping else len(response)
        forced = -1
        closes = [i for i, token in enumerate(response[:length].tolist()) if token == thinking_end]
        if closes and closes[0] == forced_position:
            forced = forced_position
        if forced >= 0:
            assert int(choices[row, forced]) == NO_WRITE
        records.append(TrajectoryRecord.from_device(
            token_ids=torch.cat((prompt, response[:length])), prompt_length=len(prompt),
            old_logprobs=logprobs[row, :length], old_values=torch.zeros(length),
            correct=False, text=tokenizer.decode(response[:length].tolist()),
            forced_token_index=forced, carry_hiddens=carries[row, :length],
            slot_choices=choices[row, :length], slot_count=config.slots,
        ))
    _report(
        "rollout",
        forced=[record.forced_token_index for record in records],
        lengths=[record.response_length for record in records],
        null_fraction=float(sum((record.slot_choices < 0).sum() for record in records)) / sum(record.response_length for record in records),
    )
    # Continuous refill: two seeded pools must agree bit-for-bit, including slot histories.
    pool_prompts = prompts + prompts[::-1]
    engine.generate_prompt_pool(pool_prompts, max_new_tokens=max_new_tokens, prefill_batch_prompts=2)
    _seed()
    pool_a = engine.generate_prompt_pool(pool_prompts, max_new_tokens=max_new_tokens, prefill_batch_prompts=2)
    _seed()
    pool_b = engine.generate_prompt_pool(pool_prompts, max_new_tokens=max_new_tokens, prefill_batch_prompts=2)
    assert pool_a.slot_choices is not None and pool_b.slot_choices is not None
    for first, second, response in zip(pool_a.slot_choices, pool_b.slot_choices, pool_b.responses, strict=True):
        assert torch.equal(first, second) and first.shape == (response.numel(),)
    for first, second in zip(pool_a.responses, pool_b.responses, strict=True):
        assert torch.equal(first, second)
    assert pool_b.admission_events >= 2
    engine.release_cache()
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    enable_packed_replay_attention(policy.causal_lm)
    enable_replay_mlp_compilation(policy.causal_lm)
    batch = collate_replay_microbatch(records, [0, 1], pad_token_id=1, device=device)
    frozen_carries = batch.carry_hiddens.clone()

    def joint(side):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = _replay_hidden(side, batch)
            actions = hidden[batch.action_batch_indices, batch.action_positions]
            return _action_logprobs(side, actions, batch, chunk_tokens=16)

    with torch.no_grad():
        behavior = joint(policy)
    rollout = torch.cat([record.old_logprobs for record in records]).to(device)
    # The terminal slot choice is never read: replay excludes its term and the
    # rollout may not even have sampled it (a cap-ended lane records no write).
    compared = batch.policy_mask & batch.slot_action_mask
    error = torch.where(compared, (behavior - rollout).abs(), torch.zeros_like(rollout))
    mean_error = error.sum().item() / compared.sum().item()
    _report("rollout_vs_replay_joint_logprob", max_abs=error.max().item(), mean_abs=mean_error,
            compared=int(compared.sum()))
    # Fused FA4 decode versus packed BF16 replay are not bit-identical, and the
    # slot head amplifies hidden drift into the joint term. Drift diagnostic,
    # not a bit-exactness claim: a contract bug shows up as nats, not tenths.
    assert error.max().item() < 0.5, error.max().item()
    assert mean_error < 0.1, mean_error
    differentiable = joint(policy)
    (-differentiable.mean()).backward()
    norms = {}
    for module_name, module in (("combiner", policy.token_combiner), ("slot_head", policy.slot_head)):
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, f"{module_name}.{name}"
            norms[f"{module_name}.{name}"] = parameter.grad.float().norm().item()
    assert all(0 < value < float("inf") for value in norms.values()), norms
    _report("actor_gradients", **norms)
    torch.testing.assert_close(batch.carry_hiddens, frozen_carries, rtol=0, atol=0)
    policy.zero_grad(set_to_none=True)

    critic = VAPOCritic.from_family("minicpm5", 
        device=device, lora_config=LoRAConfig(), gradient_checkpointing=False,
        shared_frozen_source=policy.causal_lm, token_carry=True, slot_memory=config,
    )
    enable_packed_replay_attention(critic.causal_lm)
    assert policy.token_combiner.output.weight.data_ptr() != critic.token_combiner.output.weight.data_ptr()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        critic_hidden = _replay_hidden(critic, batch)
        critic.values(critic_hidden[batch.action_batch_indices, batch.action_positions]).float().mean().backward()
    assert critic.token_combiner.key.weight.grad is not None and policy.token_combiner.key.weight.grad is None
    critic.zero_grad(set_to_none=True)
    refreshed, metrics = refresh_behavior_statistics(
        policy, critic, records, replay_token_budget=4096, replay_max_trajectories=2, logit_chunk_tokens=16,
    )
    assert all(torch.isfinite(record.old_logprobs).all() for record in refreshed)
    assert "slot_null_fraction" in metrics and "carry_actor_probe_null_mass" in metrics
    _report("refresh", **{key: value for key, value in metrics.items() if isinstance(value, (int, float))})

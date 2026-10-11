"""Gated-delta memory carry on the production CUDA path.

The CPU suite pins the memory's semantics against the sequential reference;
this pins the two numerics it cannot reach: fla's chunked scan (the replay
kernel) against that reference, forward and backward, and the compiled,
CUDA-graph-captured bf16 rollout arena against the replay that trains on it,
on the real KDA8 SFT checkpoint the memory run starts from.

Run only through mlq: .venv/bin/python -m pytest -q this_file.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from postraining.graph_decode import PinnedDecodeArena
from postraining.latent_rollout import (
    PAD_SLOT,
    generated_slot_mask,
    pack_rollout_groups_for_replay,
    replay_beliefs,
    replay_head_inputs,
    split_rollout_groups,
    trim_stream,
)
from postraining.latent_thought import LatentThoughtModel
from postraining.memory_carry import (
    _fla_delta_scan,
    memory_write_mask,
    reference_delta_scan,
)
from postraining.value_model import SeparateCritic

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires queued CUDA"
)


@pytest.fixture(autouse=True)
def _training_matmul_precision():
    """The trainer's TF32 settings (train_latent_vapo.main), so the decode
    read and the replay scan's surrounding GEMMs run as they do in a run."""
    saved = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    yield
    torch.backends.cuda.matmul.allow_tf32 = saved[0]
    torch.backends.cudnn.allow_tf32 = saved[1]
    torch.set_float32_matmul_precision(saved[2])

CHECKPOINT = Path(
    "postraining/runs/kda8_sft_omi2_drills_code30k_science_v1_e1/sft_final_model.pt"
)


def _terms(batch, length, heads, dim, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    unit = lambda x: x / x.norm(dim=-1, keepdim=True)
    shape = (batch, length, heads, dim)
    q = unit(torch.randn(shape, device=device, generator=generator))
    k = unit(torch.randn(shape, device=device, generator=generator))
    v = torch.randn(shape, device=device, generator=generator)
    log_decay = torch.nn.functional.logsigmoid(
        3.0 + torch.randn(shape[:3], device=device, generator=generator)
    )
    beta = torch.rand(shape[:3], device=device, generator=generator)
    # Masked slots, as the replay writes them: a left-pad run and holes.
    mask = torch.ones(shape[:3], device=device)
    mask[0, :37] = 0
    mask[1, 200:203] = 0
    return q, k, v, log_decay * mask, beta * mask


def test_fla_scan_matches_the_reference_forward_and_backward():
    """Across chunk boundaries, masked slots and per-row resets."""
    device = torch.device("cuda")
    inputs = _terms(3, 333, 4, 64, seed=0, device=device)
    fla_inputs = [x.clone().requires_grad_() for x in inputs]
    ref_inputs = [x.double().clone().requires_grad_() for x in inputs]
    fla_reads, fla_state = _fla_delta_scan(*fla_inputs)
    ref_reads, ref_state = reference_delta_scan(*ref_inputs)
    relative = lambda x, ref: (
        (x.double() - ref).abs().max() / ref.abs().max()
    ).item()
    report = {
        "reads": relative(fla_reads, ref_reads),
        "state": relative(fla_state, ref_state),
    }
    # Different reads at every slot, so every gradient path is exercised.
    weight = torch.randn_like(ref_reads)
    (fla_reads * weight.float()).sum().backward()
    (ref_reads * weight).sum().backward()
    for name, fla, ref in zip(
        ("q", "k", "v", "log_decay", "beta"), fla_inputs, ref_inputs, strict=True
    ):
        report[name] = relative(fla.grad, ref.grad)
    print(report)
    # fla's Triton dots run TF32 on fp32 inputs (10-bit mantissa, ~1e-3):
    # the bound is that precision, against a float64 reference.
    for name, error in report.items():
        assert error < 5e-3, (name, report)


def _wrapper(device):
    from postraining.model_io import load_model

    wrapper = LatentThoughtModel(
        load_model(CHECKPOINT, device),
        hidden_carry=True,
        memory_carry=True,
        memory_heads=4,
        memory_head_dim=64,
    ).to(device).eval()
    return wrapper


def _make_live(wrapper, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    memory = wrapper.combiner
    with torch.no_grad():
        for parameter, std in (
            (memory.reader.out.weight, 0.02),
            (memory.writer.write_gate.weight, 0.02),
            (memory.writer.decay_gate.weight, 0.02),
        ):
            parameter.copy_(
                torch.randn(
                    parameter.shape, device="cuda", generator=generator
                )
                * std
            )


def _rollout(arena, prompts, lengths, seeds, new_tokens):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return arena.rollout(
            prompts, lengths, prompt_repeats=4,
            max_new_tokens=new_tokens, max_stream_steps=new_tokens,
            token_seeds=seeds,
        )


def _relative_rms(left, right):
    return ((left.float() - right.float()).pow(2).mean().sqrt()
            / right.float().pow(2).mean().sqrt()).item()


def _replay_unit(batch, lengths, samples):
    """The trainer's replay input: per-prompt groups, left padding trimmed,
    packed. Replay masks no padding, so a mixed-pad batch is not replayable
    as one stream -- its pad slots would run through the trunk."""
    return pack_rollout_groups_for_replay(
        [trim_stream(group) for group in split_rollout_groups(batch, samples, lengths)]
    )


def _decode_vs_replay(wrapper, packed):
    written = memory_write_mask(packed.kind, PAD_SLOT)[:, 1:]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, beliefs = replay_beliefs(wrapper, packed)
    return _relative_rms(packed.hiddens[:, 1:][written], beliefs[:, :-1][written])


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the SFT checkpoint")
def test_compiled_arena_records_what_the_compiled_replay_recomputes():
    """bf16 decode/replay agreement with a live memory is no worse than the
    trunk's own decode/replay floor, and the memory is load-bearing."""
    device = torch.device("cuda")
    torch.fx.experimental._config.use_duck_shape = False
    wrapper = _wrapper(device)
    _make_live(wrapper, seed=1)
    generator = torch.Generator(device="cuda").manual_seed(2)
    prompts, samples, width, new_tokens = 16, 4, 96, 384
    lengths = torch.randint(
        24, width + 1, (prompts,), device=device, generator=generator
    ).cpu()
    prompt_ids = torch.zeros((prompts, width), dtype=torch.long, device=device)
    for row, length in enumerate(lengths.tolist()):
        prompt_ids[row, width - length :] = torch.randint(
            1, 50000, (length,), device=device, generator=generator
        )
    seeds = torch.arange(prompts * samples, device=device, dtype=torch.int64)
    row_lengths = lengths.repeat_interleave(samples)

    def arena_for(policy):
        return PinnedDecodeArena(
            policy, rows=prompts * samples, kv_width=width + new_tokens,
            temperature=1.0, stop_ids=(50256,), device=device, hidden_carry=True,
        )

    batch = _rollout(arena_for(wrapper), prompt_ids, lengths, seeds, new_tokens)
    assert batch.memory_carry
    packed = _replay_unit(batch, row_lengths, samples)
    generated = generated_slot_mask(packed)
    written = memory_write_mask(packed.kind, PAD_SLOT)
    assert int(generated.sum()) > prompts * samples * 32

    compiled = torch.compile(replay_head_inputs, dynamic=True, fullgraph=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager_beliefs, eager_inputs = replay_head_inputs(wrapper, packed)
        compiled_beliefs, compiled_inputs = compiled(wrapper, packed)
        zeroed = copy.deepcopy(packed)
        zeroed.hiddens.zero_()
        _, zeroed_inputs = replay_head_inputs(wrapper, zeroed)
    report = {
        "decode_vs_replay": _decode_vs_replay(wrapper, packed),
        "compiled_vs_eager_beliefs": _relative_rms(
            compiled_beliefs[written], eager_beliefs[written]
        ),
        "compiled_vs_eager_inputs": _relative_rms(
            compiled_inputs[generated], eager_inputs[generated]
        ),
        "memory_read_share": _relative_rms(
            eager_inputs[generated], zeroed_inputs[generated]
        ),
    }

    # The trunk's own floor: the same rollout policy with the read silenced
    # (exactly cot), through the same arena and replay unit.
    silent = copy.deepcopy(wrapper)
    with torch.no_grad():
        silent.combiner.reader.out.weight.zero_()
    silent_batch = _rollout(arena_for(silent), prompt_ids, lengths, seeds, new_tokens)
    report["cot_floor_decode_vs_replay"] = _decode_vs_replay(
        silent, _replay_unit(silent_batch, row_lengths, samples)
    )
    print(report)
    assert report["memory_read_share"] > 0.05, report
    assert report["compiled_vs_eager_inputs"] < 1e-2, report
    assert report["decode_vs_replay"] < 0.1, report
    assert report["decode_vs_replay"] < 2 * report["cot_floor_decode_vs_replay"] + 1e-3, report


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the SFT checkpoint")
def test_compiled_critic_reads_the_actor_memory_and_trains_only_its_reader():
    from postraining.model_io import load_model

    device = torch.device("cuda")
    wrapper = _wrapper(device)
    _make_live(wrapper, seed=3)
    critic = SeparateCritic(
        load_model(CHECKPOINT, device), actor_memory=wrapper.combiner
    ).to(device)
    with torch.no_grad():
        critic.combiner.out.weight.normal_(std=0.02)
        critic.head.weight.normal_(std=0.02)
    prompts, width, new_tokens = 8, 64, 128
    prompt_ids = torch.randint(1, 50000, (prompts, width), device=device)
    lengths = torch.full((prompts,), width)
    arena = PinnedDecodeArena(
        wrapper, rows=prompts * 4, kv_width=width + new_tokens,
        temperature=1.0, stop_ids=(), device=device, hidden_carry=True,
    )
    batch = _rollout(
        arena, prompt_ids, lengths,
        torch.arange(prompts * 4, device=device, dtype=torch.int64), new_tokens,
    )
    compiled_values = torch.compile(critic.values, dynamic=True, fullgraph=False)

    def compile_error():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            eager = critic.values(batch).float()
            compiled = compiled_values(batch).float()
        return _relative_rms(compiled, eager), eager

    # The floor: the same random bf16 trunk with the memory read silenced.
    live_reader = critic.combiner.out.weight.detach().clone()
    with torch.no_grad():
        critic.combiner.out.weight.zero_()
    floor, _ = compile_error()
    with torch.no_grad():
        critic.combiner.out.weight.copy_(live_reader)
    error, eager = compile_error()
    with torch.no_grad():
        critic.combiner.out.weight.zero_()
        _, silent = compile_error()
        critic.combiner.out.weight.copy_(live_reader)
    report = {
        "compiled_vs_eager": error,
        "compiled_vs_eager_floor": floor,
        "memory_value_share": _relative_rms(eager, silent),
    }
    print(report)
    assert report["memory_value_share"] > 0.05, report
    assert error < 2 * floor + 1e-3, report
    with torch.autocast("cuda", dtype=torch.bfloat16):
        values = compiled_values(batch)
    values[generated_slot_mask(batch)].float().square().mean().backward()
    assert critic.combiner.out.weight.grad.abs().sum() > 0
    assert critic.combiner.query.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in wrapper.parameters())


def test_offloading_a_strided_stream_prefix_stages_nothing_on_the_device():
    """The arena hands ``compact_stream_to_device`` views of its buffers.

    A strided CUDA source crosses to the host through a contiguous device
    temporary of its full size; for the replay record that was 4.5 GiB and
    ran a live run out of memory. The prefix must arrive intact with no
    device allocation beyond noise.
    """
    from dataclasses import fields as dataclass_fields

    from postraining.latent_rollout import (
        TOKEN_SLOT,
        LatentRolloutBatch,
        compact_stream_to_device,
    )

    device = torch.device("cuda")
    rows, buffer_rows, width, stream, used, dim = 64, 80, 1024, 900, 700, 512
    generator = torch.Generator(device=device).manual_seed(3)

    def buffer(*trailing, dtype=torch.float32):
        if dtype.is_floating_point:
            return torch.randn(
                (buffer_rows, width, *trailing), generator=generator,
                device=device, dtype=dtype,
            )
        return torch.randint(
            0, 50, (buffer_rows, width, *trailing), generator=generator,
            device=device, dtype=dtype,
        )

    kind = torch.full((buffer_rows, width), PAD_SLOT, device=device)
    kind[:, :used] = TOKEN_SLOT
    hiddens = buffer(dim)
    stream_fields = {
        "kind": kind,
        "token_ids": buffer(dtype=torch.long),
        "thoughts": buffer(0),
        "hiddens": hiddens,
        "actions": buffer(dtype=torch.long),
        "old_thought_logprobs": buffer(0),
        "old_thought_means": buffer(0),
        "old_thought_log_sigmas": buffer(0),
    }
    scalar_stream = (
        "action_mask", "stop_mask", "emit_mask", "old_stop_logprobs",
        "old_token_logprobs", "old_token_log_odds", "old_values",
        "old_action_values", "rewards",
    )
    for name in scalar_stream:
        stream_fields[name] = buffer()
    views = {
        name: value[:rows, :stream] for name, value in stream_fields.items()
    }
    assert not views["hiddens"].is_contiguous()
    batch = LatentRolloutBatch(
        **views,
        reward_scalar=torch.randn(rows, device=device),
        prompt_length=16,
        carry_injected=True,
        memory_carry=True,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)

    host = compact_stream_to_device(batch, torch.device("cpu"))

    staged = torch.cuda.max_memory_allocated(device) - baseline
    record_bytes = rows * used * dim * 4
    assert staged < record_bytes // 64, (staged, record_bytes)
    for field in dataclass_fields(LatentRolloutBatch):
        value = getattr(batch, field.name)
        if not isinstance(value, torch.Tensor):
            continue
        copied = getattr(host, field.name)
        assert copied.device.type == "cpu"
        # Zero-width fields own no storage to pin.
        assert copied.is_pinned() or copied.numel() == 0, field.name
        expected = value[:, :used] if value.dim() >= 2 else value
        assert torch.equal(copied, expected.cpu()), field.name

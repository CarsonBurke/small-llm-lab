"""Evaluation-only memory read ablations (``--memory-eval-ablation``).

Pins the foreign donor map (a cross-prompt bijection), that a foreign read is
the donor's state at the same tick and a finished donor's final state, that
the CUDA-graph arena and the eager loop implement the same ablation through
compaction, that zero_read is exactly the token-only input path on the same
trunk and restores the read-out bitwise, and the CLI contract that keeps both
out of training.
"""

from __future__ import annotations

import pytest
import torch

from postraining import latent_rollout
from postraining.graph_decode import PinnedDecodeArena
from postraining.latent_rollout import rollout_continuations
from postraining.memory_eval_ablation import (
    foreign_donor_rows,
    memory_eval_ablation,
    record_foreign_state,
    zeroed_memory_read,
)
from postraining.tests.test_hidden_carry import BACKBONES
from postraining.tests.test_memory_carry import _memory_wrapper
from postraining.vapo.config import build_arg_parser, validate_args

STOP_IDS = tuple(range(1, 5))


@pytest.mark.parametrize("prompts,repeats", [(2, 1), (2, 16), (3, 4), (64, 16), (5, 1)])
def test_foreign_state_donors_are_a_cross_prompt_bijection(prompts, repeats):
    donors = foreign_donor_rows(prompts, repeats, "foreign_state")
    rows = torch.arange(prompts * repeats)
    assert torch.equal(donors.sort().values, rows)
    assert bool((donors // repeats != rows // repeats).all())
    # The same sample index of another prompt.
    assert torch.equal(donors % repeats, rows % repeats)


@pytest.mark.parametrize("prompts,repeats", [(1, 2), (2, 16), (3, 4), (64, 16), (5, 3)])
def test_foreign_sample_donors_are_a_within_prompt_bijection(prompts, repeats):
    donors = foreign_donor_rows(prompts, repeats, "foreign_sample")
    rows = torch.arange(prompts * repeats)
    assert torch.equal(donors.sort().values, rows)
    assert torch.equal(donors // repeats, rows // repeats)
    assert bool((donors != rows).all())


def test_donor_maps_refuse_degenerate_batches():
    with pytest.raises(ValueError, match="at least two prompts"):
        foreign_donor_rows(1, 16, "foreign_state")
    with pytest.raises(ValueError, match="at least two samples"):
        foreign_donor_rows(4, 1, "foreign_sample")
    with pytest.raises(ValueError, match="not a foreign-read ablation"):
        foreign_donor_rows(4, 4, "zero_read")


def test_record_keeps_a_finished_rows_final_state():
    states = torch.zeros(4, 1, 2, 2)
    live = torch.tensor([3, 0, 2])
    new = torch.arange(3.0)[:, None, None, None].expand(3, 1, 2, 2) + 1
    record_foreign_state(states, live, new, torch.tensor([True, False, True]))
    assert states[3].eq(1).all() and states[2].eq(3).all()
    # Row 0 was not active and row 1 was not live: both untouched.
    assert states[0].eq(0).all() and states[1].eq(0).all()


def _distinct_prompts(kind, lengths, seed=5):
    _, vocab = BACKBONES[kind]
    generator = torch.Generator().manual_seed(seed)
    width = max(lengths)
    prompt_ids = torch.zeros((len(lengths), width), dtype=torch.long)
    for row, length in enumerate(lengths):
        prompt_ids[row, width - length :] = torch.randint(
            1, vocab, (length,), generator=generator
        )
    return prompt_ids, torch.tensor(lengths)


def _eager(wrapper, prompt_ids, lengths, repeats, seeds, new_tokens, **kwargs):
    with torch.no_grad():
        return rollout_continuations(
            wrapper, prompt_ids, new_tokens, new_tokens, 1.0, 1.0,
            stop_ids=STOP_IDS, prompt_lengths=lengths, pin_emit=True,
            record_likelihoods=False, replay_storage=False,
            prompt_repeats=repeats, token_seeds=seeds,
            hidden_carry=kwargs.pop("hidden_carry", True), **kwargs,
        )


def _arena(wrapper, rows, *, foreign: str | None):
    return PinnedDecodeArena(
        wrapper, rows=rows, kv_width=24, temperature=1.0, stop_ids=STOP_IDS,
        device=torch.device("cpu"), hidden_carry=True,
        cache_dtype=torch.float32, autocast_dtype=None, compile=False,
        cuda_graphs=False, row_buckets=(rows, 12, 8, 4, 2), sync_every=2,
        replay_storage=False,
        memory_read_ablation=foreign or "none",
    )


def _tokens(batch):
    return batch.token_ids * batch.action_mask.long()


def _seeds(count, seed=9):
    return torch.randint(0, 2**31, (count,), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("ablation", ["foreign_state", "foreign_sample"])
@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_arena_and_eager_foreign_reads_agree_and_are_load_bearing(kind, ablation):
    wrapper = _memory_wrapper(kind)
    prompt_ids, lengths = _distinct_prompts(kind, [3, 6, 4])
    repeats, new_tokens = 4, 13
    seeds = _seeds(12)
    eager = _eager(wrapper, prompt_ids, lengths, repeats, seeds, new_tokens,
                   foreign_memory_reads=ablation)
    arena = _arena(wrapper, 16, foreign=ablation)
    graph = arena.rollout(
        prompt_ids, lengths, prompt_repeats=repeats,
        max_new_tokens=new_tokens, max_stream_steps=new_tokens, token_seeds=seeds,
    )
    for name in ("kind", "token_ids", "action_mask"):
        assert torch.equal(getattr(eager, name), getattr(graph, name)), name
    assert len(set(graph.action_mask.sum(1).tolist())) > 2  # rows end apart
    assert arena.foreign_states is None  # nothing outlives the rollout
    own = _eager(wrapper, prompt_ids, lengths, repeats, seeds, new_tokens)
    assert not torch.equal(_tokens(own), _tokens(eager))


@pytest.mark.parametrize("ablation", ["foreign_state", "foreign_sample"])
@pytest.mark.parametrize("path", ["eager", "arena"])
def test_foreign_reads_of_identical_trajectories_are_the_policy(path, ablation):
    """Seeds and prompts chosen so the donor IS the row's trajectory (same
    prompt; same seed per sample index across prompts, or one seed per
    prompt group): a read misaligned by even one tick would show."""
    kind = "nano"
    wrapper = _memory_wrapper(kind)
    repeats, new_tokens = 4, 13
    if ablation == "foreign_state":
        prompt, _ = _distinct_prompts(kind, [5])
        prompt_ids = prompt.expand(3, -1).contiguous()
        lengths = torch.tensor([5, 5, 5])
        seeds = _seeds(repeats).repeat(3)
    else:
        # Distinct prompts: the donor is a sample of the same prompt with
        # the same seed. Rows end apart across prompts.
        prompt_ids, lengths = _distinct_prompts(kind, [3, 6, 4])
        seeds = _seeds(3).repeat_interleave(repeats)
    if path == "eager":
        foreign = _eager(wrapper, prompt_ids, lengths, repeats, seeds, new_tokens,
                         foreign_memory_reads=ablation)
    else:
        foreign = _arena(wrapper, 16, foreign=ablation).rollout(
            prompt_ids, lengths, prompt_repeats=repeats,
            max_new_tokens=new_tokens, max_stream_steps=new_tokens, token_seeds=seeds,
        )
    own = _eager(wrapper, prompt_ids, lengths, repeats, seeds, new_tokens)
    assert torch.equal(_tokens(own), _tokens(foreign))
    assert len(set(own.action_mask.sum(1).tolist())) > 1


@pytest.mark.parametrize("ablation", ["foreign_state", "foreign_sample"])
def test_a_foreign_read_is_the_donors_state_at_the_same_tick(monkeypatch, ablation):
    """Rebuild every read from the rows' own writes, independently of the
    side buffer: a live donor's latest write, a finished donor's last one."""
    wrapper = _memory_wrapper("nano")
    prompt_ids, lengths = _distinct_prompts("nano", [3, 6, 4, 5])
    repeats, new_tokens = 3, 14
    batch = len(lengths) * repeats
    donors = foreign_donor_rows(len(lengths), repeats, ablation)
    events = []
    prompt_state = []
    original_prompt_state = wrapper.memory_prompt_state
    monkeypatch.setattr(
        wrapper, "memory_prompt_state",
        lambda *a: prompt_state.append(original_prompt_state(*a)) or prompt_state[-1],
    )
    read, record = latent_rollout.foreign_read_state, latent_rollout.record_foreign_state

    def spy_read(states, donor_map, live_rows):
        value = read(states, donor_map, live_rows)
        events.append(("read", live_rows.clone(), value.clone()))
        return value

    def spy_record(states, live_rows, state, active):
        events.append(("write", live_rows.clone(), state.clone(), active.clone()))
        record(states, live_rows, state, active)

    monkeypatch.setattr(latent_rollout, "foreign_read_state", spy_read)
    monkeypatch.setattr(latent_rollout, "record_foreign_state", spy_record)
    rolled = _eager(wrapper, prompt_ids, lengths, repeats, _seeds(batch), new_tokens,
                    foreign_memory_reads=ablation, sync_every=2)

    latest = dict(enumerate(prompt_state[0].repeat_interleave(repeats, 0)))
    finished: set[int] = set()
    frozen_reads = live_reads = 0
    for event in events:
        if event[0] == "write":
            _, live_rows, state, active = event
            for slot, row in enumerate(live_rows.tolist()):
                if active[slot]:
                    latest[row] = state[slot]
                else:
                    finished.add(row)
            continue
        _, live_rows, value = event
        for slot, row in enumerate(live_rows.tolist()):
            if row in finished:
                continue  # its read feeds nothing recorded
            donor = int(donors[row])
            torch.testing.assert_close(value[slot], latest[donor], rtol=0, atol=0)
            frozen_reads += donor in finished
            live_reads += donor not in finished
    assert frozen_reads > 0 and live_reads > 0
    assert len(set(rolled.action_mask.sum(1).tolist())) > 2


def test_zero_read_is_the_token_input_path_and_restores_bitwise():
    kind = "nano"
    wrapper = _memory_wrapper(kind)
    before = {name: p.detach().clone() for name, p in wrapper.named_parameters()}
    prompt_ids, lengths = _distinct_prompts(kind, [3, 6, 4])
    seeds = _seeds(12)
    with memory_eval_ablation(wrapper, "zero_read"):
        for name, parameter in wrapper.named_parameters():
            if name == "combiner.reader.out.weight":
                assert parameter.eq(0).all()
            else:
                assert torch.equal(parameter, before[name]), name
        ablated = _eager(wrapper, prompt_ids, lengths, 4, seeds, 13)
    # The memory policy with its read zeroed is the trunk's token-only policy.
    token_only = _eager(wrapper, prompt_ids, lengths, 4, seeds, 13, hidden_carry=False)
    assert torch.equal(_tokens(ablated), _tokens(token_only))
    for name, parameter in wrapper.named_parameters():
        assert torch.equal(parameter, before[name]), name
    live = _eager(wrapper, prompt_ids, lengths, 4, seeds, 13)
    assert not torch.equal(_tokens(live), _tokens(token_only))


def test_zero_read_restores_on_error_and_refuses_training_or_no_memory():
    wrapper = _memory_wrapper("nano")
    weight = wrapper.combiner.reader.out.weight.detach().clone()
    with pytest.raises(KeyError):
        with zeroed_memory_read(wrapper):
            raise KeyError("boom")
    assert torch.equal(wrapper.combiner.reader.out.weight, weight)
    wrapper.train()
    with pytest.raises(RuntimeError, match="evaluation ablation"):
        with zeroed_memory_read(wrapper):
            pass
    assert torch.equal(wrapper.combiner.reader.out.weight, weight)
    from postraining.tests.test_hidden_carry import _carry_wrapper

    with pytest.raises(ValueError, match="memory-carry"):
        with zeroed_memory_read(_carry_wrapper("nano")):
            pass


def test_foreign_reads_refuse_training_shaped_decodes():
    wrapper = _memory_wrapper("nano")
    with pytest.raises(ValueError, match="evaluation ablation"):
        PinnedDecodeArena(
            wrapper, rows=8, kv_width=24, temperature=1.0, stop_ids=STOP_IDS,
            device=torch.device("cpu"), hidden_carry=True, compile=False,
            cuda_graphs=False, memory_read_ablation="foreign_state",
        )
    with pytest.raises(ValueError, match="only by foreign reads"):
        _arena_kwargs = dict(replay_storage=False, memory_read_ablation="zero_read")
        PinnedDecodeArena(
            wrapper, rows=8, kv_width=24, temperature=1.0, stop_ids=STOP_IDS,
            device=torch.device("cpu"), hidden_carry=True, compile=False,
            cuda_graphs=False, **_arena_kwargs,
        )
    prompt_ids, lengths = _distinct_prompts("nano", [3, 4])
    with pytest.raises(ValueError, match="evaluation ablation"):
        with torch.no_grad():
            rollout_continuations(
                wrapper, prompt_ids, 4, 4, 1.0, 1.0, stop_ids=STOP_IDS,
                prompt_lengths=lengths, pin_emit=True, hidden_carry=True,
                prompt_repeats=2, token_seeds=_seeds(4), foreign_memory_reads="foreign_state",
            )
    with pytest.raises(ValueError, match="at least two prompts"):
        _eager(wrapper, prompt_ids[:1], lengths[:1], 4, _seeds(4), 4,
               foreign_memory_reads="foreign_state")


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--reasoning-mode", "memory", "--memory-eval-ablation", "zero_read"],
         "requires --bench-only"),
        (["--reasoning-mode", "memory", "--rollout-only",
          "--memory-eval-ablation", "foreign_state"], "requires --bench-only"),
        (["--reasoning-mode", "cot", "--bench-only",
          "--memory-eval-ablation", "zero_read"], "--reasoning-mode memory"),
    ],
)
def test_cli_keeps_the_ablation_out_of_training(extra, message, capsys):
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "c", "--output", "o", *extra])
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["none", "zero_read", "foreign_state", "foreign_sample"])
def test_cli_accepts_a_bench_only_memory_ablation(mode):
    parser = build_arg_parser()
    args = parser.parse_args([
        "--checkpoint", "c", "--output", "o", "--reasoning-mode", "memory",
        "--bench-only", "--memory-eval-ablation", mode,
    ])
    validate_args(parser, args)
    assert args.memory_eval_ablation == mode
    assert build_arg_parser().parse_args(
        ["--checkpoint", "c", "--output", "o"]
    ).memory_eval_ablation == "none"


# --- CUDA (queued through mlq) ---------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires queued CUDA")
def test_compiled_graph_arena_foreign_reads_are_aligned():
    """The compiled, graph-captured tick: with identical trajectories the
    foreign arena agrees with an unablated arena as well as two unablated
    captures agree with each other; with distinct prompts it does not."""
    from postraining.tests.test_memory_carry_cuda import (
        CHECKPOINT,
        _make_live,
        _wrapper,
    )

    if not CHECKPOINT.exists():
        pytest.skip("needs the SFT checkpoint")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    wrapper = _wrapper(device)
    _make_live(wrapper, seed=1)
    generator = torch.Generator(device="cuda").manual_seed(2)
    prompts, samples, width, new_tokens = 16, 4, 64, 256
    one = torch.randint(1, 50000, (1, width), device=device, generator=generator)
    distinct = torch.randint(1, 50000, (prompts, width), device=device, generator=generator)
    lengths = torch.full((prompts,), width)
    per_sample = torch.arange(samples, device=device, dtype=torch.int64).repeat(prompts)

    def roll(prompt_ids, foreign):
        arena = PinnedDecodeArena(
            wrapper, rows=prompts * samples, kv_width=width + new_tokens,
            temperature=1.0, stop_ids=(50256,), device=device, hidden_carry=True,
            replay_storage=False,
            memory_read_ablation=foreign or "none",
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            batch = arena.rollout(
                prompt_ids, lengths, prompt_repeats=samples,
                max_new_tokens=new_tokens, max_stream_steps=new_tokens,
                token_seeds=per_sample,
            )
        return _tokens(batch)

    def agreement(left, right):
        return (left == right).all(1).float().mean().item()

    same = one.expand(prompts, -1).contiguous()
    reference = roll(same, None)
    floor = agreement(roll(same, None), reference)
    aligned = agreement(roll(same, "foreign_state"), reference)
    swapped = agreement(roll(distinct, "foreign_state"), roll(distinct, None))
    print({"floor": floor, "aligned": aligned, "distinct_prompts": swapped})
    assert aligned >= floor - 0.1
    assert swapped < 0.5

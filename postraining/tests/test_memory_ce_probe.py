"""CPU contracts of the teacher-forced memory probe (``memory_ce_probe``)."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from postraining import memory_ce_probe as probe
from postraining.core import ANSWER_CLOSE, THINK_OPEN, GPT2BPETokenizer, encode_prompt
from postraining.latent_rollout import PAD_SLOT, generated_slot_mask
from postraining.latent_thought import LatentThoughtModel
from postraining.math_drills import Drill, column_multiplication, column_subtraction
from postraining.math_prompt import answer_fence_prompt
from postraining.memory_carry import delta_write, memory_write_mask
from postraining.tests.test_hidden_carry import BACKBONES, PARITY_TOLERANCE, _prompts, _roll
from postraining.tests.test_memory_carry import HEAD_DIM, HEADS, _live_memory, _memory_wrapper

TRAIN_POOL = "postraining/data/drill_rl_study_train_20261014_v1.parquet"
HELDOUT_POOL = "postraining/data/drill_rl_study_heldout_20261014_v1.parquet"
STOP = 0


@pytest.fixture(scope="module")
def tokenizer():
    return GPT2BPETokenizer(think_tokens=True, answer_tokens=True)


# --- data -------------------------------------------------------------------


def test_training_table_is_canonical_and_excludes_heldout():
    table, manifest = probe.build_training_table(
        probe.Path(TRAIN_POOL), probe.Path(HELDOUT_POOL), pool_seed=probe.PROBE_TRAIN_POOL_SEED,
        per_cell=6,
    )
    rows = table.to_pylist()
    heldout = {row["problem"] for row in probe.drill_rows(probe.rebuild_pool_drills(probe.Path(HELDOUT_POOL)))}
    assert len(manifest["per_cell"]) == 22 and manifest["rows"] == len(rows) == 22 * 6
    assert not heldout & {row["problem"] for row in rows}
    assert len({row["problem_key"] for row in rows}) == len(rows)
    for row in rows:
        assert row["document"] == probe.drill_document(row["problem"], row["solution"], row["answer"])
        completion = row["document"][len(row["problem"]):]
        assert completion.startswith(THINK_OPEN + "\n") and completion.endswith(ANSWER_CLOSE)
        assert completion.endswith(f"<answer>{row['answer']}</answer>")
        assert "Answer:" not in row["problem"]
    with pytest.raises(ValueError, match="differ from the study"):
        probe.build_training_table(
            probe.Path(TRAIN_POOL), probe.Path(HELDOUT_POOL), pool_seed=20261016, per_cell=1
        )


def test_cell_generation_skips_excluded_problems():
    first = probe.generate_cell_drills(
        "sub_integer", 4, pool_seed=probe.PROBE_TRAIN_POOL_SEED, count=3, excluded_keys=frozenset()
    )
    skipped = probe.generate_cell_drills(
        "sub_integer", 4, pool_seed=probe.PROBE_TRAIN_POOL_SEED, count=3,
        excluded_keys=frozenset({first[0].key}),
    )
    assert first[0].problem not in {drill.problem for drill in skipped}
    assert first[1].problem == skipped[0].problem


def test_heldout_rebuild_matches_the_pool():
    drills = probe.rebuild_pool_drills(probe.Path(HELDOUT_POOL))
    rows = probe.pq.read_table(HELDOUT_POOL).to_pylist()
    assert len(drills) == len(rows) == 550
    for drill, row in zip(drills, rows):
        assert drill.problem == row["prompt"][0]["content"]
        assert drill.answer == row["reward_model"]["ground_truth"]
        assert drill.digit_order == row["extra_info"]["digit_order"]
        assert probe.module_name(drill) == row["extra_info"]["module"]


def test_encoding_is_the_sft_and_rl_framing(tokenizer):
    from postraining.sft_trace_train import tokenize_documents

    drill = probe.rebuild_pool_drills(probe.Path(HELDOUT_POOL))[0]
    row = probe.drill_rows([drill])[0]
    trace = probe.encode_table(tokenizer, [row], annotate=True)[0]
    reference = tokenize_documents(
        tokenizer,
        [{"problem": drill.problem, "document": probe.drill_document(drill.problem, drill.solution, drill.answer)}],
        seq_len=4096,
        instruction_suffix=probe.ANSWER_FENCE_SUFFIX,
    )[0]
    assert trace.prompt_length == reference.prompt_length
    assert np.array_equal(trace.ids, reference.ids)
    rl_prompt = encode_prompt(tokenizer, answer_fence_prompt(drill.problem))
    assert trace.ids[: trace.prompt_length].tolist() == rl_prompt
    assert trace.ids[trace.prompt_length] == tokenizer.think_open_id
    assert trace.ids[-1] == tokenizer.answer_close_id
    assert trace.classes["all"].size == trace.supervised
    assert trace.classes["all"].all()
    assert not any(mask[-1] for name, mask in trace.classes.items() if name != "all")


def _class_text(tokenizer, trace, name):
    completion = trace.ids[trace.prompt_length:]
    return [tokenizer._tokenizer.decode([int(completion[i])]) for i in np.flatnonzero(trace.classes[name][:-1])]


def _encode_solution(tokenizer, family, order, problem, lines, answer):
    return probe.encode_trace(
        tokenizer, module=f"{family}_dX", family=family, digit_order=order, problem=problem,
        solution="\n".join(lines), answer=answer, annotate=True,
    )


def test_borrow_decision_tokens_are_classified(tokenizer):
    # 5340 - 1236: ones real borrow; tens 3 after the borrow vs 3 (equal after);
    # hundreds 3 vs 2; thousands 5 vs 1.  4452 - 1432: ones 2 vs 2 and
    # hundreds 4 vs 4 are equal, tens and thousands are not.
    trace = _encode_solution(
        tokenizer, "sub_integer", "reversed", "What is 5340 - 1236?",
        column_subtraction(5340, 1236), "4104",
    )
    assert _class_text(tokenizer, trace, "borrow_real") == [" is"]
    assert _class_text(tokenizer, trace, "borrow_equal_after") == [" -"]
    assert _class_text(tokenizer, trace, "borrow_none") == [" -", " -"]
    assert _class_text(tokenizer, trace, "borrow_equal") == []
    trace = _encode_solution(
        tokenizer, "sub_integer", "reversed", "What is 4452 - 1432?",
        column_subtraction(4452, 1432), "3020",
    )
    assert _class_text(tokenizer, trace, "borrow_equal") == [" -", " -"]
    assert _class_text(tokenizer, trace, "borrow_none") == [" -", " -"]
    # The expanded method has no decision token.
    from postraining.math_drills import expanded_subtraction

    trace = _encode_solution(
        tokenizer, "sub_integer", "forward", "What is 4452 - 1432?",
        expanded_subtraction(4452, 1432), "3020",
    )
    assert not any(trace.classes[name].any() for name in probe.DECISION_CLASSES)


def test_line_classes_follow_step_lines(tokenizer):
    from scripts.analyze_drill_step_errors import step_lines

    lines = column_multiplication(438, 99)
    solution = "\n".join(lines)
    spans = probe.annotate_solution(solution, "mul_integer", "reversed")
    marked = [solution[s:e].strip() for s, e, _ in spans["comp_line"]]
    assert marked == step_lines(solution)
    prose = [solution[s:e] for s, e, _ in spans["prose"]]
    assert all(not probe.is_step_line(line) for line in prose) and prose


def test_partial_product_classes(tokenizer):
    lines = column_multiplication(438, 99)
    trace = _encode_solution(tokenizer, "mul_integer", "reversed", "What is 438 * 99?", lines, "43362")
    assert "".join(_class_text(tokenizer, trace, "pp_readout")).replace(" ", "") == "39423942"
    assert "".join(_class_text(tokenizer, trace, "pp_shift")).replace(" ", "") == "39420"
    # 3942 + 39420: five columns, the leftmost reads 0 + 3.
    assert [t.strip() for t in _class_text(tokenizer, trace, "pp_add_leftmost")] == ["0", "3"]
    assert len(_class_text(tokenizer, trace, "pp_add_operand")) == 10


# --- batches ----------------------------------------------------------------


def _random_traces(vocab, lengths, prompt=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    traces = []
    for index, length in enumerate(lengths):
        ids = torch.randint(1, vocab, (length,), generator=generator).numpy().astype(np.int32)
        traces.append(probe.EncodedTrace(f"m{index % 2}", f"p{index}", ids, prompt, {}))
    return traces


def test_collate_supervises_completion_and_stop():
    traces = _random_traces(32, [9, 6])
    batch = probe.collate(traces, torch.device("cpu"), stop_id=STOP, pad_multiple=4)
    assert batch.ids.shape == (2, 12)
    assert batch.read[0].tolist() == [False] * 4 + [True] * 5 + [False] * 3
    flat = batch.ids.flatten()
    first = batch.positions[: traces[0].supervised]
    assert first.tolist() == list(range(3, 9))
    assert batch.labels[: traces[0].supervised - 1].tolist() == flat[first[:-1] + 1].tolist()
    assert batch.labels[traces[0].supervised - 1] == STOP


def test_sampler_is_deterministic_and_length_grouped():
    traces = _random_traces(32, [5 + (i * 7) % 40 for i in range(60)])
    draw = lambda: [[t.problem for t in b] for _, b in zip(range(8), probe.CellBalancedSampler(traces, 4, 4, 9))]
    assert draw() == draw()
    for batch in [b for _, b in zip(range(4), probe.CellBalancedSampler(traces, 4, 4, 9))]:
        assert len(batch) == 4


# --- model contracts --------------------------------------------------------


def _wrapper(kind, live=False, seed=3):
    factory, _ = BACKBONES[kind]
    wrapper = LatentThoughtModel(
        factory(seed), hidden_carry=True, memory_carry=True,
        memory_heads=HEADS, memory_head_dim=HEAD_DIM,
    ).eval()
    if live:
        _live_memory(wrapper.combiner, seed + 100)
    return wrapper


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_off_arm_is_the_plain_model(kind):
    wrapper = _wrapper(kind)
    vocab = BACKBONES[kind][1]
    traces = _random_traces(vocab, [11, 7, 9])
    batch = probe.collate(traces, torch.device("cpu"), stop_id=STOP, pad_multiple=8)
    with torch.no_grad():
        out, carried, movements = probe.probe_forward(wrapper, batch, probe.ProbeConfig(memory=False))
        logprobs = probe.target_logprobs(wrapper, out, batch)
        expected = []
        for trace in traces:
            ids = torch.from_numpy(trace.ids.astype(np.int64))[None]
            logits = wrapper.backbone.policy_logits(ids).float().log_softmax(-1)[0]
            targets = torch.cat([ids[0, trace.prompt_length:], torch.tensor([STOP])])
            expected.append(logits[trace.prompt_length - 1:].gather(-1, targets[:, None]).squeeze(-1))
    assert carried is None and movements == []
    torch.testing.assert_close(logprobs, torch.cat(expected), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_every_memory_arm_starts_bitwise_at_off(kind):
    wrapper = _wrapper(kind)  # W_o = 0, the trainer's init
    traces = _random_traces(BACKBONES[kind][1], [11, 7, 9])
    batch = probe.collate(traces, torch.device("cpu"), stop_id=STOP, pad_multiple=8)
    donor = probe.derangement(3, torch.Generator().manual_seed(0))
    with torch.no_grad():
        base = probe.target_logprobs(
            wrapper, probe.probe_forward(wrapper, batch, probe.ProbeConfig(memory=False))[0], batch
        )
        for config in (
            probe.ProbeConfig(memory=True),
            probe.ProbeConfig(memory=True, passes=1),
            probe.ProbeConfig(memory=True, write_layer=1),
            probe.ProbeConfig(memory=True, shuffle=True),
            *(probe.ProbeConfig(memory=True, read_ablation=a) for a in probe.READ_ABLATIONS),
        ):
            out, carried, movements = probe.probe_forward(wrapper, batch, config, donor)
            assert torch.equal(probe.target_logprobs(wrapper, out, batch), base), config
            assert all(float(moved) == 0.0 for moved, _ in movements)


def test_shuffle_control_preserves_batch_statistics():
    generator = torch.Generator().manual_seed(4)
    carried = torch.randn(5, 6, 3, generator=generator)
    lengths = torch.full((5,), 6)
    donor = probe.derangement(5, generator)
    assert not bool((donor == torch.arange(5)).any())
    shuffled = probe.shuffle_carried(carried, lengths, donor)
    # Equal lengths: an exact row permutation, no row keeps its own stream.
    assert torch.equal(shuffled, carried[donor])
    assert torch.equal(
        shuffled.flatten(0, 1).sort(0).values, carried.flatten(0, 1).sort(0).values
    )
    # Unequal lengths: every written vector is a real slot of another row.
    lengths = torch.tensor([6, 2, 4, 3, 5])
    shuffled = probe.shuffle_carried(carried, lengths, donor)
    for row in range(5):
        source = int(donor[row])
        assert source != row
        for slot in range(int(lengths[row])):
            assert torch.equal(shuffled[row, slot], carried[source, slot % int(lengths[source])])


def test_midlayer_write_wires_the_block_output(monkeypatch):
    wrapper = _wrapper("kda", live=True)
    traces = _random_traces(BACKBONES["kda"][1], [10, 8])
    batch = probe.collate(traces, torch.device("cpu"), stop_id=STOP, pad_multiple=8)
    with torch.no_grad():
        out = probe.run_stream(wrapper, batch, None, write_layer=2)
        x = out.inputs
        for block in wrapper.backbone.blocks[:2]:
            x = block(x)
        torch.testing.assert_close(out.source, x, rtol=0, atol=0)
        assert not torch.allclose(out.source, out.belief)
        recorded = []
        original = wrapper.combiner.stream_inputs
        monkeypatch.setattr(
            wrapper.combiner, "stream_inputs",
            lambda base, hiddens, *rest: recorded.append(hiddens) or original(base, hiddens, *rest),
        )
        carried, _ = probe.fixed_point_carried(
            wrapper, batch, probe.ProbeConfig(memory=True, write_layer=2, passes=1), None
        )
        probe.run_stream(wrapper, batch, carried, write_layer=2)
    write_mask = memory_write_mask(batch.kind, PAD_SLOT)
    assert carried.written is carried.query is carried.own
    torch.testing.assert_close(carried.written, x, rtol=0, atol=0)
    torch.testing.assert_close(recorded[-1], probe.shift_carried(x, write_mask), rtol=0, atol=0)
    assert float(recorded[-1][:, 0].abs().sum()) == 0.0


@pytest.mark.parametrize("kind", sorted(BACKBONES))
def test_fixed_point_reaches_the_decode_and_is_exact_through_p_plus_k(kind):
    wrapper = _memory_wrapper(kind)
    rollout = _roll(wrapper, _prompts(kind, 4, 5), stop_ids=(), new_tokens=6)
    assert bool((rollout.kind != PAD_SLOT).all())
    prompt = rollout.prompt_length
    traces = [
        probe.EncodedTrace("m", f"p{row}", rollout.token_ids[row].numpy().astype(np.int32), prompt, {})
        for row in range(rollout.token_ids.size(0))
    ]
    batch = probe.collate(traces, torch.device("cpu"), stop_id=STOP, pad_multiple=1)
    assert torch.equal(batch.read, generated_slot_mask(rollout))
    length = batch.ids.size(1)
    tolerance = PARITY_TOLERANCE[kind]
    with torch.no_grad():
        converged, _, movements = probe.probe_forward(
            wrapper, batch, probe.ProbeConfig(memory=True, passes=length)
        )
        # The record holds the belief computed one slot earlier.
        written = memory_write_mask(rollout.kind, PAD_SLOT)[:, 1:]
        torch.testing.assert_close(
            converged.belief[:, :-1][written], rollout.hiddens[:, 1:][written],
            rtol=tolerance, atol=tolerance,
        )
        # Supervised slots are P-1 .. L-1; the last predicts the stop target,
        # which the rollout never sampled.
        logprobs = probe.target_logprobs(wrapper, converged, batch).view(4, -1)[:, :-1]
        emits = rollout.emit_mask.bool()[:, prompt - 1:-1]
        assert bool(emits.all())
        torch.testing.assert_close(
            logprobs, rollout.old_token_logprobs[:, prompt - 1:-1],
            rtol=tolerance, atol=tolerance,
        )
        assert float(movements[-1][0]) <= 1e-10 * float(movements[-1][1])
        for passes in (1, 2, 3):
            out, _, _ = probe.probe_forward(
                wrapper, batch, probe.ProbeConfig(memory=True, passes=passes)
            )
            exact = prompt + passes
            torch.testing.assert_close(
                out.belief[:, :exact], converged.belief[:, :exact], rtol=1e-5, atol=1e-5
            )
            assert (out.belief[:, exact] - converged.belief[:, exact]).abs().max() > 1e-4


# --- optimizers and arms ----------------------------------------------------


def test_memory_optimizers_are_the_trainers_fresh_partition():
    from postraining.train_latent_vapo import build_optimizers
    from postraining.value_model import SeparateCritic

    wrapper = _wrapper("nano")
    critic = SeparateCritic(BACKBONES["nano"][0](11), actor_memory=wrapper.combiner)
    rate = 3e-3
    assert probe.fresh_muon_learning_rate(rate) == pytest.approx(7.25e-3)
    trainer = build_optimizers(
        wrapper, critic, 2e-5, trunk_optimizer="muon", muon_learning_rate=4.8e-5,
        critic_muon_learning_rate=4.8e-5, fused=False, fresh_learning_rate=rate,
        fresh_muon_learning_rate=probe.fresh_muon_learning_rate(rate),
    )
    ours = probe.memory_optimizers(wrapper, rate)
    assert set(ours) == {"actor_fresh", "actor_fresh_muon"}
    for name, optimizer in ours.items():
        ids = lambda opt: {id(p) for g in opt.param_groups for p in g["params"]}
        assert ids(optimizer) == ids(trainer[name])
        assert type(optimizer) is type(trainer[name])
        assert optimizer.param_groups[0]["lr"] == trainer[name].param_groups[0]["lr"]


def test_a_frozen_trunk_step_trains_only_the_memory():
    wrapper = _wrapper("kda")
    arm = probe.standard_arms(3e-3, 2, 0.1)["mem"]
    probe.set_trainable(wrapper, arm)
    optimizers = probe.arm_optimizers(wrapper, arm)
    trunk = {k: v.clone() for k, v in wrapper.backbone.state_dict().items()}
    memory = {k: v.clone() for k, v in wrapper.combiner.state_dict().items()}
    traces = _random_traces(BACKBONES["kda"][1], [11, 9, 7])
    batch = probe.collate(traces, torch.device("cpu"), stop_id=STOP, pad_multiple=8)
    config = probe.ProbeConfig(memory=True)
    for step in range(2):
        out, _, _ = probe.probe_forward(wrapper, batch, config)
        (-probe.target_logprobs(wrapper, out, batch).sum()).backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        state = wrapper.combiner.state_dict()
        assert all(torch.isfinite(v).all() for v in state.values())
        if step == 0:
            # W_o = 0 gates every other memory gradient on the first step.
            assert not torch.equal(state["reader.out.weight"], memory["reader.out.weight"])
            assert torch.equal(state["writer.key.weight"], memory["writer.key.weight"])
    assert not torch.equal(wrapper.combiner.state_dict()["writer.key.weight"], memory["writer.key.weight"])
    for key, value in wrapper.backbone.state_dict().items():
        assert torch.equal(value, trunk[key]), key


def test_standard_arms_and_cli_validation():
    arms = probe.standard_arms(3e-3, 4, 0.1)
    assert set(arms) == {
        "off", "mem", "mem_lr_quarter", "mem_lr_x4", "mem_midlayer", "mem_shuffled",
        "trunk_only", "mem_plus_trunk",
    }
    assert not arms["off"].trains and arms["mem_lr_x4"].fresh_learning_rate == pytest.approx(1.2e-2)
    parser = probe.build_parser()
    args = parser.parse_args([
        "train", "--train-data", "x.parquet", "--hazard-panel", "h.parquet", "--output", "out",
        "--arms", "off", "mem", "--steps", "10",
    ])
    probe.validate_train_args(parser, args)
    assert args.fixed_point_passes == 2 and args.write_layer == 4
    bad = parser.parse_args([
        "train", "--train-data", "x", "--hazard-panel", "h", "--output", "o", "--arms", "mem", "mem",
        "--steps", "1",
    ])
    with pytest.raises(SystemExit):
        probe.validate_train_args(parser, bad)
    assert probe.eval_schedule(120) == {0, 3, 6, 12, 25, 50, 100, 120}
    assert {340, 400, 800, 1600, 2000} <= probe.eval_schedule(2000)
    assert probe.eval_schedule(0) == {0}


def test_batched_training_encoding_matches_the_annotated_path(tokenizer):
    drills = probe.rebuild_pool_drills(probe.Path(HELDOUT_POOL))[::50]
    rows = probe.drill_rows(drills)
    fast = probe.encode_table(tokenizer, rows, annotate=False)
    slow = probe.encode_table(tokenizer, rows, annotate=True)
    for left, right in zip(fast, slow, strict=True):
        assert np.array_equal(left.ids, right.ids)
        assert left.prompt_length == right.prompt_length and left.module == right.module


def test_hazard_panel_is_column_method_and_disjoint(tmp_path):
    train = tmp_path / "train.parquet"
    table, _ = probe.build_training_table(
        probe.Path(TRAIN_POOL), probe.Path(HELDOUT_POOL), pool_seed=probe.PROBE_TRAIN_POOL_SEED,
        per_cell=4,
    )
    probe.pq.write_table(table, train)
    panel, manifest = probe.build_hazard_panel(
        train, probe.Path(TRAIN_POOL), probe.Path(HELDOUT_POOL),
        seed=probe.PROBE_HAZARD_PANEL_SEED, per_cell=5,
    )
    rows = panel.to_pylist()
    assert len(manifest["per_cell"]) == 13 and len(rows) == 65
    assert {row["digit_order"] for row in rows} == {"reversed"}
    assert {row["family"] for row in rows} == set(probe.HAZARD_FAMILIES)
    excluded = set(table.column("problem").to_pylist())
    for pool in (TRAIN_POOL, HELDOUT_POOL):
        excluded |= {r["prompt"][0]["content"] for r in probe.pq.read_table(pool).to_pylist()}
    assert not excluded & {row["problem"] for row in rows}
    bigger, manifest = probe.build_hazard_panel(
        train, probe.Path(TRAIN_POOL), probe.Path(HELDOUT_POOL),
        seed=probe.PROBE_HAZARD_PANEL_SEED, per_cell=5, family_per_cell={"sub_integer": 9},
    )
    for cell, count in manifest["per_cell"].items():
        assert count == (9 if cell.startswith("sub_integer") else 5)
    # Overrides extend each cell's draw sequence, so the default cells are unchanged.
    assert [r for r in bigger.to_pylist() if r["family"] != "sub_integer"] == [
        r for r in rows if r["family"] != "sub_integer"
    ]
    assert probe.parse_family_count("sub_decimal=320") == ("sub_decimal", 320)
    with pytest.raises(ValueError, match="unknown hazard families"):
        probe.build_hazard_panel(
            train, probe.Path(TRAIN_POOL), probe.Path(HELDOUT_POOL),
            seed=probe.PROBE_HAZARD_PANEL_SEED, per_cell=1, family_per_cell={"add_integer": 2},
        )
    with pytest.raises(ValueError, match="differ from every pool seed"):
        probe.build_hazard_panel(
            train, probe.Path(TRAIN_POOL), probe.Path(HELDOUT_POOL),
            seed=probe.PROBE_TRAIN_POOL_SEED, per_cell=1,
        )


def test_train_and_analyze_end_to_end(tmp_path, monkeypatch, capsys):
    """The CLI path on a tiny real-vocabulary trunk: both panels scored, every
    arm written, frozen arms leave the trunk alone, and analyze reads it all."""
    from postraining import provenance
    from postraining.nano_backbone import NanoGPTBackbone

    def tiny_actor(checkpoint, device, *, memory_heads, memory_head_dim, seed):
        torch.manual_seed(seed)
        backbone = NanoGPTBackbone(vocab_size=50304, num_layers=2, model_dim=128).float()
        with torch.no_grad():
            backbone.proj.weight.normal_(std=0.02)
        wrapper = LatentThoughtModel(
            backbone, hidden_carry=True, memory_carry=True,
            memory_heads=memory_heads, memory_head_dim=memory_head_dim,
        ).eval()
        return wrapper, {"args": {"lr_scale": 0.1}}

    full_heldout = probe.rebuild_pool_drills(probe.Path(HELDOUT_POOL))
    monkeypatch.setattr(probe, "load_probe_actor", tiny_actor)
    monkeypatch.setattr(probe, "rebuild_pool_drills", lambda pool: full_heldout[::110])
    monkeypatch.setattr(provenance, "capture_source_provenance", lambda out, repo: {"stub": True})
    train = tmp_path / "train.parquet"
    probe.pq.write_table(
        probe.pq.read_table("postraining/data/memory_ce_probe_drills_20261021_v1.parquet").slice(0, 24),
        train,
    )
    panel = tmp_path / "panel.parquet"
    probe.pq.write_table(
        probe.pq.read_table("postraining/data/memory_ce_probe_hazard_panel_20261021_v1.parquet").take(
            list(range(0, 1040, 160))
        ),
        panel,
    )
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"stub")
    run = tmp_path / "run"
    arms = ["off", "mem", "mem_shuffled", "mem_midlayer", "trunk_only", "mem_plus_trunk"]
    probe.main([
        "train", "--checkpoint", str(checkpoint), "--train-data", str(train),
        "--hazard-panel", str(panel), "--output", str(run), "--device", "cpu",
        "--arms", *arms, "--steps", "3", "--batch-trajectories", "4", "--macro-batches", "1",
        "--log-every", "1", "--write-layer", "1",
        "--memory-heads", "2", "--memory-head-dim", "16", "--eval-slot-budget", "4096",
        "--train-slot-budget", "1536",
    ])
    manifest = probe.json.loads((run / "manifest.json").read_text())
    assert set(manifest["panel_traces"]) == {"heldout", "hazard"}
    assert manifest["hazard_panel_sha256"] == probe.file_sha256(panel)
    results = {arm: probe.json.loads((run / arm / "result.json").read_text()) for arm in arms}
    off = results["off"]["final"]
    for arm in ("mem", "mem_shuffled", "mem_midlayer"):
        evals = [
            probe.json.loads(line) for line in (run / arm / "metrics.jsonl").read_text().splitlines()
            if '"eval"' in line
        ]
        assert [record["step"] for record in evals] == [0, 3]
        for panel_name in ("heldout", "hazard"):
            # The read starts at zero, so step 0 is the off arm exactly.
            assert evals[0][f"{panel_name}/ce/all"] == pytest.approx(off[f"{panel_name}/ce/all"], abs=1e-6)
        assert "heldout/fixed_point/residual" in results[arm]["final_extra_passes"]
        assert ("final_ablations" in results[arm]) == (arm != "mem_shuffled")
        assert (run / arm / "memory_final.pt").exists()
    problems = probe.json.loads((run / "mem" / "final_problems.json").read_text())
    assert set(problems) == {"heldout", "hazard"}
    out = tmp_path / "report.json"
    probe.main(["analyze", str(run), "--json", str(out)])
    report = probe.json.loads(out.read_text())
    assert set(report["vs"]["off"]) == set(arms) - {"off"}
    assert set(report["vs"]["off"]["mem"]) == {"heldout", "hazard"}
    assert "flat" in report["vs"]["trunk_only"]["mem_plus_trunk"]["hazard"]
    assert report["curves"]["mem"]["steps"] == [0, 3]
    assert set(report["ablations"]["mem"]) == set(probe.READ_ABLATIONS)
    assert "mem" in report["read_share_vs_shuffled"]
    # A run whose arm never finished is refused, not silently dropped.
    (run / "mem" / "result.json").unlink()
    with pytest.raises(FileNotFoundError, match="did not finish"):
        probe.main(["analyze", str(run)])


def test_evaluation_batches_always_give_the_shuffle_a_donor():
    def trace(length):
        return probe.EncodedTrace("m", "p", np.arange(length, dtype=np.int32), 4, {})

    traces = [trace(n) for n in (60, 60, 60, 120)]
    batches = probe.evaluation_batches(traces, slot_budget=256)
    # 3 x 64 fits, the 128-wide tail would be alone and joins the batch before it.
    assert [len(b) for b in batches] == [4]
    assert sorted(t.length for b in batches for t in b) == [60, 60, 60, 120]
    assert all(len(b) >= 2 for b in probe.evaluation_batches([trace(n) for n in range(10, 700, 7)], 1500))
    with pytest.raises(ValueError, match="cannot fit"):
        probe.evaluation_batches(traces, slot_budget=192)
    with pytest.raises(ValueError, match="at least 2"):
        probe.evaluation_batches(traces[:1], slot_budget=1024)


# --- read ablations -----------------------------------------------------------


def _ablation_setup(seed=5):
    wrapper = _wrapper("nano", live=True, seed=seed)
    traces = _random_traces(BACKBONES["nano"][1], [11, 9, 7], seed=seed)
    batch = probe.collate(traces, torch.device("cpu"), stop_id=STOP, pad_multiple=4)
    generator = torch.Generator().manual_seed(seed)
    token_latent = wrapper.embed_tokens(batch.ids)
    own = torch.randn(*token_latent.shape, generator=generator)
    donor = probe.derangement(3, generator)
    write_mask = memory_write_mask(batch.kind, PAD_SLOT)
    return wrapper, batch, own, donor, token_latent, write_mask


def _reference_reads(memory, token_latent, written, query, write_mask, keep):
    """Sequential state per slot, keeping only writes ``keep(t, j)`` for slot t's read."""
    hiddens = probe.shift_carried(written, write_mask)
    queries = probe.shift_carried(query, write_mask)
    k, v, log_decay, beta = probe.masked_write_terms(memory.writer, hiddens, write_mask)
    q = memory.reader.queries(token_latent, queries)
    reads = torch.zeros_like(v)
    for t in range(k.size(1)):
        state = torch.zeros(k.size(0), k.size(2), k.size(3), v.size(-1))
        for j in range(t + 1):
            if keep(t, j):
                state = delta_write(state, k[:, j], v[:, j], log_decay[:, j], beta[:, j])
        reads[:, t] = (q[:, t, :, :, None] * state).sum(-2)
    return reads


@pytest.mark.parametrize("ablation", [None, *probe.READ_ABLATIONS])
def test_read_ablations_read_the_intended_state(ablation):
    wrapper, batch, own, donor, token_latent, write_mask = _ablation_setup()
    memory = wrapper.combiner
    donated = probe.shuffle_carried(own, batch.lengths, donor)
    if ablation == "foreign_writes":
        carried = probe.Carried(donated, own, own)
    else:
        carried = probe.Carried(own, own, own)
    keep = {
        None: lambda t, j: True,
        "foreign_writes": lambda t, j: True,
        "drop_last_write": lambda t, j: j < t,
        # From an empty state: earlier writes are dropped, not decayed.
        "last_write_only": lambda t, j: j == t,
    }[ablation]
    with torch.no_grad():
        got = probe.memory_inputs(memory, token_latent, carried, write_mask, batch.read, ablation)
        reads = _reference_reads(memory, token_latent, carried.written, carried.query, write_mask, keep)
        expected = torch.where(batch.read[..., None], memory.reader.add(token_latent, reads), token_latent)
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
        if ablation is None:
            # One stream: the trainer's own path.
            trainer = memory.stream_inputs(
                token_latent, probe.shift_carried(own, write_mask), write_mask, batch.read
            )
            torch.testing.assert_close(got, trainer, rtol=0, atol=0)
        else:
            full = probe.memory_inputs(memory, token_latent, probe.Carried(own, own, own),
                                       write_mask, batch.read, None)
            assert (got - full)[batch.read].abs().max() > 1e-4


def test_foreign_writes_keep_the_query_and_shuffle_residual_is_own():
    wrapper, batch, own, donor, token_latent, write_mask = _ablation_setup(seed=8)
    with torch.no_grad():
        carried, _ = probe.fixed_point_carried(
            wrapper, batch, probe.ProbeConfig(memory=True, read_ablation="foreign_writes"), donor
        )
        assert carried.query is carried.own and carried.written is not carried.own
        torch.testing.assert_close(
            carried.written, probe.shuffle_carried(carried.own, batch.lengths, donor), rtol=0, atol=0
        )
        shuffled, _ = probe.fixed_point_carried(
            wrapper, batch, probe.ProbeConfig(memory=True, shuffle=True), donor
        )
        assert shuffled.query is shuffled.written
        out = probe.run_stream(wrapper, batch, shuffled, None)
        accumulator = probe.EvalAccumulator(wrapper.combiner.heads)
        accumulator.movement.clear()
        moved, base = probe.relative_movement(out.source, shuffled.own, batch.read)
        logprobs = probe.target_logprobs(wrapper, out, batch)
        scored = probe.dataclasses.replace(batch, classes={"all": torch.ones_like(batch.labels, dtype=torch.bool)})
        accumulator.add(wrapper, scored, out, shuffled, [], logprobs, torch.zeros_like(batch.labels))
        assert accumulator.movement["residual"] == [float(moved), float(base)]
    with pytest.raises(ValueError, match="unshuffled memory arm"):
        probe.ProbeConfig(memory=True, shuffle=True, read_ablation="drop_last_write")
    with pytest.raises(ValueError, match="needs a donor"):
        probe.fixed_point_carried(
            wrapper, batch, probe.ProbeConfig(memory=True, read_ablation="foreign_writes"), None
        )


def test_flat_class_is_the_union_of_the_flat_step_types(tokenizer):
    drills = probe.rebuild_pool_drills(probe.Path(HELDOUT_POOL))
    traces = probe.encode_table(tokenizer, probe.drill_rows(drills[::7]), annotate=True)
    for trace in traces:
        union = np.logical_or.reduce([trace.classes[name] for name in probe.FLAT_CLASSES])
        assert np.array_equal(trace.classes["flat"], union)
        assert not (trace.classes["flat"] & trace.classes["borrow_real"]).any()


def test_micro_batched_gradient_equals_the_whole_batch():
    """Right padding and per-row memory make the accumulated gradient exact."""
    wrapper = _wrapper("nano", live=True, seed=6)
    traces = _random_traces(BACKBONES["nano"][1], [9, 13, 17, 21, 30, 6], seed=6)
    config = probe.ProbeConfig(memory=True)
    params = [p for p in wrapper.combiner.parameters()]

    def gradient(groups):
        for parameter in params:
            parameter.grad = None
        for group in groups:
            batch = probe.collate(group, torch.device("cpu"), stop_id=STOP)
            out, _, _ = probe.probe_forward(wrapper, batch, config)
            (-probe.target_logprobs(wrapper, out, batch).sum()).backward()
        return [parameter.grad.clone() for parameter in params]

    whole = gradient([traces])
    groups = probe.evaluation_batches(traces, slot_budget=128)
    assert len(groups) > 1 and all(len(g) >= 2 for g in groups)
    for left, right in zip(whole, gradient(groups), strict=True):
        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-6)


def test_chunked_readout_matches_the_whole_readout():
    wrapper = _wrapper("nano", live=True, seed=9)
    traces = _random_traces(BACKBONES["nano"][1], [12, 9, 15], seed=9)
    batch = probe.collate(traces, torch.device("cpu"), stop_id=STOP)
    with torch.no_grad():
        out, _, _ = probe.probe_forward(wrapper, batch, probe.ProbeConfig(memory=True))
        whole = probe.target_logprobs(wrapper, out, batch)
        torch.testing.assert_close(probe.target_logprobs(wrapper, out, batch, 5), whole, rtol=0, atol=1e-6)

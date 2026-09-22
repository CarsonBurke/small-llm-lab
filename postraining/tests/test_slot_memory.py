"""Deterministic contracts of the slot-choice latent memory (host only)."""

from __future__ import annotations

import copy
import io
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask

import postraining.vapo.policy as vapo
import postraining.train_minicpm_vapo as trainer
from postraining.vapo.model.hf import HFCausalTrunk, HFModelSpec
from postraining.vapo.policy import (
    VAPOCritic,
    VAPOPolicy,
    TrajectoryRecord,
    collate_replay_microbatch,
)
from postraining.vapo.model.lora import LoRAConfig
from postraining.slot_memory import (
    NO_WRITE,
    SlotChoiceHead,
    SlotMemoryCombiner,
    SlotMemoryConfig,
    SlotMemoryRolloutState,
    build_alive_table,
    replay_read,
    slot_block_mask,
    slot_choice_statistics,
    slot_mask_mod,
    slot_memory_replay_hidden,
    visibility_reference,
)
from postraining.token_carry import token_carry_replay_hidden

CPU = torch.device("cpu")


def _loop_alive(choices: list[int], slots: int) -> list[list[int]]:
    table, state = [], [NO_WRITE] * slots
    for step, choice in enumerate(choices):
        if choice >= 0:
            state[choice] = step
        table.append(list(state))
    return table


def _loop_visible(choices: list[int], slots: int) -> list[set[int]]:
    table = _loop_alive(choices, slots)
    return [{key for key in range(len(choices)) if choices[key] >= 0 and table[q][choices[key]] == key} for q in range(len(choices))]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_alive_table_matches_loop_reference(seed):
    generator = torch.Generator().manual_seed(seed)
    slots = 5
    choices = torch.randint(NO_WRITE, slots, (37,), generator=generator)
    table = build_alive_table(choices, slots)
    assert table.tolist() == _loop_alive(choices.tolist(), slots)
    shifted = build_alive_table(choices, slots, offset=100)
    assert torch.equal(shifted, torch.where(table >= 0, table + 100, table))
    assert build_alive_table(torch.tensor([], dtype=torch.long), slots).shape == (0, slots)
    with pytest.raises(ValueError, match="slot choices"):
        build_alive_table(torch.tensor([slots]), slots)


@pytest.mark.parametrize("seed", [3, 4])
def test_visibility_rule_matches_explicit_loop_and_is_bounded_by_slots(seed):
    generator = torch.Generator().manual_seed(seed)
    slots, length = 4, 53
    choices = torch.randint(NO_WRITE, slots, (length,), generator=generator)
    table = build_alive_table(choices, slots)
    visible = visibility_reference(table, choices)
    expected = _loop_visible(choices.tolist(), slots)
    for query in range(length):
        assert set(visible[query].nonzero().flatten().tolist()) == expected[query]
    assert int(visible.sum(dim=1).max()) <= slots
    # Point-wise mask_mod agrees with the dense rule everywhere.
    mask_mod = slot_mask_mod(table, choices)
    q_idx, k_idx = torch.meshgrid(torch.arange(length), torch.arange(length), indexing="ij")
    assert torch.equal(mask_mod(torch.zeros(()), torch.zeros(()), q_idx, k_idx), visible)
    # Null writes are never readable; a written key is visible exactly until overwritten.
    assert not visible[:, choices < 0].any()
    for key in range(length):
        if choices[key] < 0:
            continue
        later = [step for step in range(key + 1, length) if choices[step] == choices[key]]
        end = later[0] if later else length
        assert visible[key:end, key].all() and not visible[:key, key].any() and not visible[end:, key].any()


def test_block_mask_from_table_matches_dense_and_flex_reference():
    generator = torch.Generator().manual_seed(11)
    slots, length = 6, 300
    choices = torch.randint(NO_WRITE, slots, (length,), generator=generator)
    table = build_alive_table(choices, slots)
    block_mask = slot_block_mask(table, choices, block_size=64)
    dense = visibility_reference(table, choices)
    padded = torch.zeros(320, 320, dtype=torch.bool)
    padded[:length, :length] = dense
    block_presence = padded.view(5, 64, 5, 64).any(dim=3).any(dim=1)
    assert torch.equal(block_mask.to_dense()[0, 0].bool(), block_presence)
    reference = create_block_mask(
        slot_mask_mod(table, choices), None, None, length, length, BLOCK_SIZE=64, device="cpu",
    )
    assert torch.equal(block_mask.to_dense(), reference.to_dense())
    # Sparse: each query block needs at most the blocks its <= slots occupants live in.
    assert int(block_mask.kv_num_blocks.max()) <= min(slots * 64, (length + 63) // 64)
    with pytest.raises(ValueError, match="one query per stored key"):
        slot_block_mask(table[:-1], choices)


def test_flex_read_matches_dense_reference_including_empty_and_null_rows():
    torch.manual_seed(5)
    config = SlotMemoryConfig(slots=3, heads=2, head_dim=8)
    combiner = SlotMemoryCombiner(16, config)
    with torch.no_grad():
        combiner.null_key.normal_()
    choices = torch.tensor([NO_WRITE, NO_WRITE, 0, 1, NO_WRITE, 0, 2, 2, 1, NO_WRITE])
    table = build_alive_table(choices, config.slots)
    carries = torch.randn(10, 16)
    embeddings = torch.randn(10, 16)
    positions = torch.arange(10)
    with torch.no_grad():
        dense = replay_read(combiner, token_embeddings=embeddings, carries=carries, positions=positions,
                            alive_table=table, key_choices=choices, backend="dense")
        flex = replay_read(combiner, token_embeddings=embeddings, carries=carries, positions=positions,
                           alive_table=table, key_choices=choices, backend="flex")
    torch.testing.assert_close(flex, dense, rtol=1e-4, atol=1e-5)
    # Nothing was written before step 2: those reads see only the null entry.
    assert torch.equal(dense[:2], torch.zeros(2, config.width))
    assert dense[2:].abs().sum() > 0
    with pytest.raises(ValueError, match="backend"):
        replay_read(combiner, token_embeddings=embeddings, carries=carries, positions=positions,
                    alive_table=table, key_choices=choices, backend="eager")


def test_relative_position_enters_the_score_only_through_write_and_read_offsets():
    config = SlotMemoryConfig(slots=2, heads=1, head_dim=8)
    combiner = SlotMemoryCombiner(4, config)
    hidden = torch.randn(1, 4)
    embedding = torch.randn(1, 4)
    scores = []
    for write, read in ((0, 7), (100, 107), (3, 4)):
        keys, _ = combiner.keys_values(hidden, torch.tensor([write]))
        queries = combiner.queries(embedding, hidden, torch.tensor([read]))
        scores.append(float((queries * keys).sum()))
    assert scores[0] == pytest.approx(scores[1], abs=1e-4)
    assert scores[0] != pytest.approx(scores[2], abs=1e-3)


def test_rollout_state_matches_replay_visibility_and_reads_across_episode_boundaries():
    """The live slot banks and the recorded-table replay must expose identical keys."""
    torch.manual_seed(21)
    hidden_size, config = 12, SlotMemoryConfig(slots=3, heads=2, head_dim=6)
    combiner = SlotMemoryCombiner(hidden_size, config)
    head = SlotChoiceHead(hidden_size, config.slots)
    with torch.no_grad():
        combiner.output.weight.normal_(std=0.3)
        combiner.null_key.normal_()
        head.projection.weight.normal_()
    batch, steps = 2, 14
    state = SlotMemoryRolloutState(config, batch_size=batch, capacity=steps, device=CPU, dtype=torch.float32)
    producers = torch.randn(steps, batch, hidden_size)
    embeddings = torch.randn(steps, batch, hidden_size)
    position = torch.zeros(batch, dtype=torch.long)
    active = torch.ones(batch, dtype=torch.bool)
    forced = torch.zeros(batch, dtype=torch.bool)
    forced_step, restart_step = 5, 8
    live_visible: dict[int, list[set[int]]] = {0: [], 1: []}
    live_reads: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    episodes: dict[int, list[tuple[int, int]]] = {0: [(0, steps)], 1: [(0, restart_step), (restart_step, steps)]}
    exported: dict[tuple[int, int, int], torch.Tensor] = {}
    with torch.no_grad():
        for step in range(steps):
            if step == restart_step:
                # Lane 1 finishes its episode: export its history (as the engine
                # does before refill), then admit a new prompt with every slot empty.
                exported[(1, 0, restart_step)] = state.history[1, :restart_step].clone()
                state.reset_lanes(torch.tensor([1]))
                position[1] = 0
                assert (state.alive[1] == NO_WRITE).all() and (state.history[1] == NO_WRITE).all()
            forced[:] = False
            forced[0] = step == forced_step
            mixed, logprob = state.step(
                combiner, head, token_embedding=embeddings[step], producer_hidden=producers[step],
                position=position, active=active, forced=forced,
            )
            assert logprob[0] == 0 if step == forced_step else True
            for lane in range(batch):
                live_visible[lane].append({int(v) for v in state.alive[lane] if v >= 0})
                live_reads[lane].append(mixed[lane].clone())
            position += 1
    exported[(0, 0, steps)] = state.history[0, :steps].clone()
    exported[(1, restart_step, steps)] = state.history[1, :steps - restart_step].clone()
    assert int(exported[(0, 0, steps)][forced_step]) == NO_WRITE
    assert (state.history >= NO_WRITE).all() and (state.history < config.slots).all()
    for lane in range(batch):
        for start, stop in episodes[lane]:
            choices = exported[(lane, start, stop)]
            table = build_alive_table(choices, config.slots)
            visible = visibility_reference(table, choices)
            for local, step in enumerate(range(start, stop)):
                assert {v + 0 for v in visible[local].nonzero().flatten().tolist()} == {v for v in live_visible[lane][step]}
            for backend in ("dense", "flex"):
                with torch.no_grad():
                    read = replay_read(
                        combiner, token_embeddings=embeddings[start:stop, lane],
                        carries=producers[start:stop, lane], positions=torch.arange(stop - start),
                        alive_table=table, key_choices=choices, backend=backend,
                    )
                    replayed = combiner.combine(embeddings[start:stop, lane], read)
                torch.testing.assert_close(replayed, torch.stack(live_reads[lane][start:stop]), rtol=1e-4, atol=1e-5)


def test_slot_head_joint_logprob_and_null_choice():
    torch.manual_seed(2)
    head = SlotChoiceHead(6, slots=4)
    hidden = torch.randn(5, 6)
    uniform = head.log_prob(hidden, torch.tensor([0, 3, NO_WRITE, 2, NO_WRITE]))
    torch.testing.assert_close(uniform, torch.full((5,), -torch.log(torch.tensor(5.0))))
    with torch.no_grad():
        head.projection.weight.normal_()
    log_probabilities = head.logits(hidden).log_softmax(dim=-1)
    torch.testing.assert_close(head.log_prob(hidden, torch.tensor([NO_WRITE] * 5)), log_probabilities[:, 4])
    torch.testing.assert_close(head.log_prob(hidden, torch.tensor([1, 1, 1, 1, 1])), log_probabilities[:, 1])
    choices, logprobs = head.sample(hidden)
    assert choices.shape == (5,) and bool((choices >= 0).all()) and bool((choices <= 4).all())
    torch.testing.assert_close(logprobs, log_probabilities.gather(1, choices[:, None]).squeeze(1))
    with pytest.raises(ValueError, match="one slot choice"):
        head.log_prob(hidden, torch.tensor([0, 1]))


def test_combiner_starts_as_identity_and_never_trains_the_producer():
    config = SlotMemoryConfig(slots=4, heads=1, head_dim=4)
    combiner = SlotMemoryCombiner(8, config)
    embedding = torch.randn(3, 8, requires_grad=True)
    producer = torch.randn(3, 8, requires_grad=True)
    choices = torch.tensor([0, 1, 0])
    table = build_alive_table(choices, config.slots)
    read = replay_read(combiner, token_embeddings=embedding, carries=producer, positions=torch.arange(3),
                       alive_table=table, key_choices=choices, backend="dense")
    output = combiner.combine(embedding, read)
    torch.testing.assert_close(output, embedding, rtol=0, atol=0)
    (output * torch.arange(24.0).view(3, 8)).sum().backward()
    assert producer.grad is None
    assert combiner.output.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(combiner.scale.grad) == 0
    assert combiner.query_carry.weight.grad is not None and combiner.query_carry.weight.grad.abs().sum() == 0


class _CausalTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(17, 8)
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.layers = nn.ModuleList()
        self.calls = 0

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, use_cache=False, **kwargs):
        assert not use_cache
        self.calls += 1
        inputs = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        transformed = inputs + self.q_proj(inputs)
        starts = (position_ids[0] == 0).nonzero(as_tuple=True)[0].tolist()
        stops = starts[1:] + [transformed.shape[1]]
        running = torch.cat([
            transformed[:, start:stop].cumsum(dim=1) for start, stop in zip(starts, stops)
        ], dim=1)
        return SimpleNamespace(last_hidden_state=F.layer_norm(running, (8,)))


class _LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="llama", vocab_size=17, hidden_size=8, pad_token_id=0)
        self.model = _CausalTrunk()
        self.lm_head = nn.Linear(8, 17, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens


CONFIG = SlotMemoryConfig(slots=3, heads=2, head_dim=4)



TEST_SPEC = HFModelSpec(
    key="test",
    model_id="test/fixture",
    revision="0" * 40,
    vocab_size=17,
    requires_chat_template=False,
)


def _trunk(model):
    """Wrap a fixture causal LM in the Hugging Face trunk adapter."""
    return HFCausalTrunk(model, TEST_SPEC)


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    torch.manual_seed(19)
    base = _LM()
    lora = LoRAConfig(rank=2, alpha=4, targets=("q_proj",))
    actor = VAPOPolicy(_trunk(copy.deepcopy(base)), lora, token_carry=True, slot_memory=CONFIG)
    critic = VAPOCritic(_trunk(copy.deepcopy(base)), lora, token_carry=True, slot_memory=CONFIG, critic_width=5)
    for side in (actor, critic):
        side.slot_replay_backend = "dense"
        with torch.no_grad():
            side.token_combiner.output.weight.normal_(std=0.3)
            side.token_combiner.null_key.normal_()
    return actor, critic


def _record(tokens, prompt, choices, forced=-1):
    actions = len(tokens) - prompt
    return TrajectoryRecord(
        token_ids=torch.tensor(tokens, dtype=torch.int32), prompt_length=prompt,
        old_logprobs=torch.zeros(actions), advantages=torch.linspace(-1, 1, actions),
        correct=True, text="", forced_token_index=forced,
        carry_hiddens=(torch.arange(actions * 8).reshape(actions, 8).float() / 7 + tokens[0]).bfloat16(),
        slot_choices=torch.tensor(choices, dtype=torch.int16), slot_count=CONFIG.slots,
    )


def _batch(*records):
    return collate_replay_microbatch(records, range(len(records)), pad_token_id=0, device=CPU)


def test_record_contract_rejects_invalid_slot_choices():
    record = _record([1, 2, 3, 4, 5, 6], 2, [0, NO_WRITE, 2, 1])
    for invalid in ([0, 1, 2], [0, 1, 3, 1], [0, -2, 1, 1]):
        with pytest.raises(ValueError, match="slot choices"):
            replace(record, slot_choices=torch.tensor(invalid, dtype=torch.int16))
    with pytest.raises(ValueError, match="slot choices"):
        replace(record, slot_choices=torch.tensor([0, 1, 2, 1], dtype=torch.int64))
    with pytest.raises(ValueError, match="travel together"):
        replace(record, slot_count=0)
    with pytest.raises(ValueError, match="travel together"):
        replace(record, slot_choices=None)
    with pytest.raises(ValueError, match="require stored carry"):
        replace(record, carry_hiddens=None, slot_count=CONFIG.slots)
    with pytest.raises(ValueError, match="forced token never writes"):
        replace(record, forced_token_index=2)
    replace(record, forced_token_index=1)
    with torch.inference_mode():
        converted = TrajectoryRecord.from_device(
            token_ids=torch.tensor([1, 2, 3, 4]), prompt_length=1,
            old_logprobs=torch.zeros(3), old_values=torch.zeros(3), correct=False, text="",
            carry_hiddens=torch.zeros(3, 8, dtype=torch.bfloat16),
            slot_choices=torch.tensor([2, NO_WRITE, 0]), slot_count=3,
        )
    assert converted.slot_choices.dtype == torch.int16 and not converted.slot_choices.is_inference()
    assert converted.storage_bytes - replace(converted, slot_choices=None, slot_count=0).storage_bytes == 6
    stream = io.BytesIO()
    torch.save(converted, stream)
    stream.seek(0)
    torch.load(stream, weights_only=False).__post_init__()


def test_collate_builds_per_trajectory_tables_in_global_key_space():
    first = _record([1, 2, 3, 4, 5, 6], 2, [0, 1, 0, 2])
    second = _record([7, 8, 9, 10], 1, [1, NO_WRITE, 1])
    batch = _batch(first, second)
    assert batch.carry_input_positions.tolist() == [2, 3, 4, 6, 7]
    assert batch.slot_key_choices.tolist() == [0, 1, 0, 1, NO_WRITE]
    assert batch.slot_positions.tolist() == [0, 1, 2, 0, 1]
    assert batch.slot_actions.tolist() == [0, 1, 0, 2, 1, NO_WRITE, 1]
    assert batch.slot_action_mask.tolist() == [True, True, True, False, True, True, False]
    expected = torch.cat((
        build_alive_table(torch.tensor([0, 1, 0]), 3),
        build_alive_table(torch.tensor([1, NO_WRITE]), 3, offset=3),
    ))
    assert torch.equal(batch.slot_alive_table, expected)
    visible = visibility_reference(batch.slot_alive_table, batch.slot_key_choices)
    assert not visible[:3, 3:].any() and not visible[3:, :3].any()
    assert visible[4, 3] and visible[3, 3]
    plain = replace(first, slot_choices=None, slot_count=0)
    with pytest.raises(ValueError, match="cannot mix slot-memory"):
        _batch(first, plain)
    with pytest.raises(ValueError, match="share a slot count"):
        _batch(first, replace(second, slot_count=5))


def _oracle(side, records):
    outputs = []
    for record in records:
        ids = record.token_ids[:-1].long().unsqueeze(0)
        plain = side.token_embeddings(ids)
        choices = record.slot_choices[:-1].long()
        read = replay_read(
            side.token_combiner, token_embeddings=plain[0, record.prompt_length:],
            carries=record.carry_hiddens[:-1], positions=torch.arange(choices.numel()),
            alive_table=build_alive_table(choices, record.slot_count), key_choices=choices, backend="dense",
        )
        mixed = side.token_combiner.combine(plain[0, record.prompt_length:], read)
        embeddings = torch.cat((plain[:, :record.prompt_length], mixed[None]), dim=1)
        outputs.append(side.causal_lm.model(
            inputs_embeds=embeddings, position_ids=torch.arange(record.input_length).unsqueeze(0),
        ).last_hidden_state)
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("side_index", [0, 1])
def test_packed_replay_matches_per_trajectory_oracle_and_trains_only_its_own_side(models, side_index):
    side = models[side_index]
    reference = copy.deepcopy(side)
    records = [
        _record([1, 2, 3, 4, 5, 6], 2, [0, 1, 0, 2]),
        _record([7, 8, 9, 10], 3, [NO_WRITE]),
        _record([3, 4, 5, 6, 7], 1, [2, 2, NO_WRITE, 1]),
    ]
    batch = _batch(*records)
    batch.carry_hiddens.requires_grad_()
    actual = side.token_carry_replay_hidden(batch)
    assert side.causal_lm.model.calls == 1
    torch.testing.assert_close(actual, _oracle(reference, records))
    weights = torch.linspace(-0.7, 1.3, actual.numel()).view_as(actual)
    (actual * weights).sum().backward()
    assert batch.carry_hiddens.grad is None
    for name in ("output", "query_token", "query_carry", "key", "value", "token_delta"):
        assert getattr(side.token_combiner, name).weight.grad.abs().sum() > 0, name
    assert side.token_combiner.null_key.grad.abs().sum() > 0
    assert side.token_combiner.scale.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in models[1 - side_index].parameters())
    assert side.token_combiner.output.weight.data_ptr() != models[1 - side_index].token_combiner.output.weight.data_ptr()


def test_plain_token_carry_replay_refuses_slot_records_and_vice_versa(models, monkeypatch):
    actor, _ = models
    batch = _batch(_record([1, 2, 3, 4], 1, [0, 1, 2]))
    with pytest.raises(ValueError, match="cannot consume slot-memory"):
        token_carry_replay_hidden(actor, batch)
    plain = replace(_record([1, 2, 3, 4], 1, [0, 1, 2]), slot_choices=None, slot_count=0)
    with pytest.raises(ValueError, match="slot choices"):
        slot_memory_replay_hidden(actor, _batch(plain), backend="dense")
    with pytest.raises(ValueError, match="slot-memory mode"):
        trainer._replay_hidden(actor, _batch(plain))
    with pytest.raises(ValueError, match="embeddings come from the rollout"):
        actor.carry_embeddings(torch.tensor([[1]]), torch.zeros(1, 8))


def test_checkpoint_roundtrip_and_geometry_mismatch(models):
    actor, critic = models
    payload = actor.checkpoint_payload()
    assert payload["slot_memory"] == CONFIG.payload() and "slot_head" in payload
    assert "slot_head" not in critic.checkpoint_payload()
    restored = copy.deepcopy(actor)
    with torch.no_grad():
        actor.slot_head.projection.weight.normal_()
        actor.token_combiner.key.weight.normal_()
    restored.load_token_carry_state_dict(actor.checkpoint_payload())
    batch = _batch(_record([1, 2, 3, 4, 5], 1, [0, 2, 1, 1]))
    torch.testing.assert_close(restored.token_carry_replay_hidden(batch), actor.token_carry_replay_hidden(batch))
    torch.testing.assert_close(restored.slot_head.projection.weight, actor.slot_head.projection.weight)
    wrong = VAPOPolicy(_trunk(_LM()), actor.lora_config, token_carry=True, slot_memory=SlotMemoryConfig(slots=4, heads=2, head_dim=4))
    with pytest.raises(ValueError, match="geometry"):
        wrong.load_token_carry_state_dict(actor.checkpoint_payload())
    plain = VAPOPolicy(_trunk(_LM()), actor.lora_config, token_carry=True)
    with pytest.raises(ValueError, match="slot-memory mode"):
        plain.load_token_carry_state_dict(actor.checkpoint_payload())
    with pytest.raises(ValueError, match="slot-memory mode"):
        actor.load_token_carry_state_dict(plain.checkpoint_payload())
    with pytest.raises(ValueError, match="slot head presence"):
        critic.load_token_carry_state_dict(actor.checkpoint_payload())
    with pytest.raises(ValueError, match="extends token carry"):
        VAPOPolicy(_trunk(_LM()), actor.lora_config, slot_memory=CONFIG)


def test_checkpoint_schema_selection():
    assert trainer.policy_checkpoint_schema(latent_thinking=False, token_carry=True, slot_memory=True) == "minicpm5_vapo_slot_memory/v1"
    assert trainer.policy_checkpoint_schema(latent_thinking=False, token_carry=True, slot_memory=False) == "minicpm5_vapo_token_carry/v4"
    assert trainer.policy_checkpoint_schema(latent_thinking=False, token_carry=False, slot_memory=False) == "minicpm5_vapo_adapter/v6"
    with pytest.raises(ValueError):
        trainer.policy_checkpoint_schema(latent_thinking=False, token_carry=False, slot_memory=True)
    args = SimpleNamespace(slot_memory=True, slot_memory_slots=64, slot_memory_heads=1, slot_memory_head_dim=128)
    assert trainer.slot_memory_config_from_args(args) == SlotMemoryConfig()
    assert trainer.slot_memory_config_from_args(SimpleNamespace(slot_memory=False)) is None


def test_joint_action_likelihood_and_forced_masking(models):
    actor, _ = models
    with torch.no_grad():
        actor.slot_head.projection.weight.normal_()
    record = _record([1, 2, 3, 4, 5, 6], 2, [0, NO_WRITE, 2, 1], forced=1)
    batch = _batch(record)
    hidden = actor.token_carry_replay_hidden(batch)[batch.action_batch_indices, batch.action_positions]
    joint = trainer._action_logprobs(actor, hidden, batch, chunk_tokens=2)
    token_only = vapo.chunked_frozen_head_logprobs(hidden, batch.targets, actor.lm_head_weight, chunk_tokens=2)
    slot_only = actor.slot_head.log_prob(hidden, torch.tensor([0, NO_WRITE, 2, 1]))
    # The terminal slot choice is never read: it contributes no likelihood.
    torch.testing.assert_close(joint[:-1], (token_only + slot_only)[:-1])
    torch.testing.assert_close(joint[-1], token_only[-1])
    assert bool(slot_only[-1] < -1e-3)
    assert batch.slot_action_mask.tolist() == [True, True, True, False]
    assert batch.policy_mask.tolist() == [True, False, True, True]
    assert bool(slot_only[1] < 0) and int(batch.slot_actions[1]) == NO_WRITE
    plain_batch = _batch(replace(record, slot_choices=None, slot_count=0))
    with pytest.raises(ValueError, match="recorded slot choices"):
        trainer._action_logprobs(actor, hidden, plain_batch, chunk_tokens=2)


_REPLAY_OPTIONS = dict(replay_token_budget=64, replay_max_trajectories=16, logit_chunk_tokens=3)


def _records(count):
    torch.manual_seed(count)
    records = []
    for index in range(count):
        length = 4 + index % 3
        prompt = 1 + index % 2
        response = length - prompt
        choices = torch.randint(NO_WRITE, CONFIG.slots, (response,), dtype=torch.int16)
        records.append(TrajectoryRecord(
            token_ids=(torch.arange(length, dtype=torch.int32) + index) % 8 + 1,
            prompt_length=prompt, old_logprobs=torch.full((response,), -2.0),
            advantages=torch.linspace(0.3, 1.0, response), correct=index % 2 == 0, text=str(index),
            carry_hiddens=(torch.arange(response * 8).reshape(response, 8) / 50 + index).bfloat16(),
            slot_choices=choices, slot_count=CONFIG.slots,
        ))
    return records


def test_refresh_materializes_joint_likelihoods_independent_of_packing(models):
    actor, critic = models
    with torch.no_grad():
        actor.slot_head.projection.weight.normal_()
    records = _records(7)
    saved = [(record.carry_hiddens.clone(), record.slot_choices.clone()) for record in records]
    refreshed, metrics = trainer.refresh_behavior_statistics(actor, critic, records, **_REPLAY_OPTIONS)
    narrow, _ = trainer.refresh_behavior_statistics(
        actor, critic, records, **{**_REPLAY_OPTIONS, "replay_token_budget": 7},
    )
    assert {"slot_null_fraction", "slot_write_entropy", "carry_actor_probe_null_mass", "carry_critic_probe_read_age_mean"} <= set(metrics)
    for wide, small, (carries, choices) in zip(refreshed, narrow, saved):
        torch.testing.assert_close(wide.old_logprobs, small.old_logprobs)
        torch.testing.assert_close(wide.carry_hiddens, carries, rtol=0, atol=0)
        assert torch.equal(wide.slot_choices, choices) and torch.equal(small.slot_choices, choices)
        batch = _batch(wide)
        hidden = actor.token_carry_replay_hidden(batch)[batch.action_batch_indices, batch.action_positions]
        torch.testing.assert_close(wide.old_logprobs, trainer._action_logprobs(actor, hidden, batch, chunk_tokens=3))
    before = trainer.measure_post_update_behavior_kl(actor, refreshed, **_REPLAY_OPTIONS)
    assert before["post_update_ratio_abs_log_max"] == pytest.approx(0.0, abs=1e-6)
    with torch.no_grad():
        actor.slot_head.projection.bias[0].add_(0.5)
    after = trainer.measure_post_update_behavior_kl(actor, refreshed, **_REPLAY_OPTIONS)
    assert after["post_update_approximate_kl"] > 0


@pytest.mark.parametrize("value_only", [False, True])
def test_update_trains_slot_head_and_combiners_without_touching_history(models, value_only):
    actor, critic = models
    records = _records(8)
    saved = [(record.carry_hiddens.clone(), record.slot_choices.clone()) for record in records]
    head_before = actor.slot_head.projection.weight.detach().clone()
    actor_before = actor.token_combiner.output.weight.detach().clone()
    critic_before = critic.token_combiner.output.weight.detach().clone()
    options = dict(
        **_REPLAY_OPTIONS, optimizer_minibatches=2, clip_low=0.2, clip_high=0.2,
        value_coefficient=1.0, nextlat_horizon=1, nextlat_samples=1,
        nextlat_mse_coefficient=1.0, nextlat_kl_coefficient=1.0, nextlat_kl_chunk_tokens=3,
        train_nextlat=False, grad_clip_norm=100.0, value_only=value_only,
    )
    actor_optimizer = torch.optim.SGD(list(actor.actor_parameters()), lr=0.01)
    critic_optimizer = torch.optim.SGD(list(critic.backbone_parameters()) + list(critic.value_head.parameters()), lr=0.01)
    for _ in range(2):
        trainer.update_step(actor, critic, records, actor_optimizer, critic_optimizer, **options)
    assert not torch.equal(critic.token_combiner.output.weight, critic_before)
    assert torch.equal(actor.token_combiner.output.weight, actor_before) == value_only
    assert torch.equal(actor.slot_head.projection.weight, head_before) == value_only
    for record, (carries, choices) in zip(records, saved):
        torch.testing.assert_close(record.carry_hiddens, carries, rtol=0, atol=0)
        assert torch.equal(record.slot_choices, choices)


def test_slot_choice_statistics():
    stats = slot_choice_statistics([torch.tensor([0, 1, NO_WRITE, 1]), torch.tensor([NO_WRITE, NO_WRITE])], slots=4)
    assert stats["null_fraction"] == pytest.approx(0.5)
    assert stats["distinct_slots_per_trajectory"] == pytest.approx(1.0)
    assert 0 < stats["write_entropy_ratio"] < 1
    assert slot_choice_statistics([], 4) == {}


def test_slot_count_is_bounded_by_int16_record_storage():
    from postraining.slot_memory import MAX_SLOTS

    SlotMemoryConfig(slots=MAX_SLOTS)
    with pytest.raises(ValueError, match="int16"):
        SlotMemoryConfig(slots=MAX_SLOTS + 1)

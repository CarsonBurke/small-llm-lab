"""KDA backbone: checkpoint loading, decode parity, and stack integration.

The KDA trunk is the first recurrent backbone behind the post-training
interface, so these tests pin the properties the dense suites get for free
from KV caches: stepwise decode must reproduce the teacher-forced forward,
left-padded prefill must match unpadded evaluation, rollout logprobs must
replay, and the optimizer partition must keep the conv windows out of Muon.

Everything here runs the pure-PyTorch reference paths on CPU; the
chunk_kda/fused-kernel agreement is asserted by the GPU parity job
(``postraining/kda_gpu_parity.py``) queued through mlq.
"""

import math

import pytest
import torch

import nanogpt_mini_kda_model as kda_model
from postraining.kda_backbone import NanoKDABackbone
from postraining.latent_thought import LatentThoughtModel
from postraining.latent_rollout import (
    replay_beliefs,
    rollout_continuations,
    trim_stream,
)
from postraining.model_io import load_model

MODEL_KWARGS = dict(
    vocab_size=32,
    num_layers=4,
    model_dim=128,
    mlp_hidden=192,
    delta_num_heads=2,
    delta_layer_indices=[0, 2],
    delta_attention_type="kda",
    delta_full_rank_gate=False,
    delta_mlp_on_delta=False,
    dense_attention_type="mha",
)

ARCHITECTURE = "nanogpt_mini_gpt2vocab_kda_kdkd_mixers_v3"


def _seeded_backbone(seed: int = 3) -> NanoKDABackbone:
    torch.manual_seed(seed)
    backbone = NanoKDABackbone(**MODEL_KWARGS).float().eval()
    with torch.no_grad():
        # The fresh init zeroes every output projection (identity trunk) and
        # the readout; give the trunk live signal so decode parity is a real
        # statement about the recurrence rather than about zeros.
        for block in backbone.blocks:
            attn = block.attn
            if block.use_kda:
                attn.o_proj.weight.normal_(std=0.02)
            else:
                attn.proj.weight.normal_(std=0.02)
                attn.proj.bias.zero_()
            if block.use_mlp:
                block.mlp.proj.weight.normal_(std=0.02)
        backbone.proj.weight.normal_(std=0.05)
        backbone.proj.bias.normal_(std=0.05)
    return backbone


def _wrapper(seed: int = 3) -> LatentThoughtModel:
    return LatentThoughtModel(_seeded_backbone(seed))


def test_kda_checkpoint_payload_strict_loads_through_model_io(tmp_path):
    torch.manual_seed(11)
    reference = _seeded_backbone(11)
    payload = {
        "model": {
            key: value.clone()
            for key, value in reference.state_dict().items()
        },
        "model_config": dict(MODEL_KWARGS),
        "architecture": ARCHITECTURE,
        "train_seq_len": 96,
    }
    path = tmp_path / "kda_final_model.pt"
    torch.save(payload, path)
    loaded = load_model(path, torch.device("cpu"))
    assert isinstance(loaded, NanoKDABackbone)
    assert loaded.architecture == ARCHITECTURE
    assert loaded.train_context_tokens == 96
    assert all(not p.requires_grad for p in loaded.parameters())
    input_ids = torch.randint(0, 32, (2, 12))
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.float().policy_logits(input_ids),
            reference.policy_logits(input_ids),
        )


def test_model_io_refuses_gdn2_and_configless_kda(tmp_path):
    payload = {
        "model": {},
        "model_config": dict(MODEL_KWARGS),
        "architecture": "nanogpt_mini_gpt2vocab_gdn2_kdkd_mixers_v3",
    }
    path = tmp_path / "gdn2.pt"
    torch.save(payload, path)
    with pytest.raises(NotImplementedError, match="GDN-2"):
        load_model(path, torch.device("cpu"))
    # Route a KDA architecture through the metadata-style payload, which
    # carries no model_config: the loader must refuse to guess a mixer layout.
    configless = {
        "model": {},
        "metadata": {"model": {}, "architecture": ARCHITECTURE},
    }
    path = tmp_path / "configless.pt"
    torch.save(configless, path)
    with pytest.raises(ValueError, match="model_config"):
        load_model(path, torch.device("cpu"))


def test_policy_logits_match_the_pretraining_forward():
    """The backbone reproduces the standalone pretraining module bit-for-bit.

    ``embed_tokens`` composed with ``temporal_belief_from_token_latent`` and
    the softcapped readout must BE the pretraining forward — this is the
    dense-logit half of the base-model gate, on the reference CPU path.
    """
    backbone = _seeded_backbone()
    input_ids = torch.randint(0, 32, (2, 17))
    targets = torch.randint(0, 32, (2, 17))
    with torch.no_grad():
        logits = backbone.policy_logits(input_ids).float()
        expected_loss = kda_model.KDAGPT.forward(backbone, input_ids, targets)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="sum"
        )
    torch.testing.assert_close(loss, expected_loss)


def test_stepwise_decode_matches_teacher_forced_logits():
    """Prefill + one-token steps reproduce the parallel forward.

    This is the recurrent analogue of KV-cache exactness: the delta-rule
    state carried across steps must price every continuation exactly as a
    dense re-evaluation of the whole prefix would.
    """
    wrapper = _wrapper()
    input_ids = torch.randint(0, 32, (3, 14))
    prompt = 8
    with torch.no_grad():
        reference = wrapper.policy_logits(input_ids).float()
        caches = wrapper.make_generation_cache(3, 14, torch.device("cpu"))
        output = wrapper.prefill(input_ids[:, :prompt], caches)
        torch.testing.assert_close(
            output.logits.float(),
            reference[:, prompt - 1],
            rtol=1e-4,
            atol=1e-5,
        )
        for position in range(prompt, 14):
            output = wrapper.token_step(
                input_ids[:, position], caches, position
            )
            torch.testing.assert_close(
                output.logits.float(),
                reference[:, position],
                rtol=1e-4,
                atol=1e-5,
            )


def test_left_padded_prefill_matches_unpadded():
    """Left padding must not leak into the recurrent state or the beliefs."""
    wrapper = _wrapper()
    ids = torch.randint(1, 32, (2, 9))
    pad = 4
    padded = torch.zeros((2, 9 + pad), dtype=torch.long)
    padded[:, pad:] = ids
    key_valid = torch.zeros((2, 9 + pad), dtype=torch.bool)
    key_valid[:, pad:] = True
    device = torch.device("cpu")
    with torch.no_grad():
        clean_caches = wrapper.make_generation_cache(2, 9, device)
        clean = wrapper.prefill(ids, clean_caches)
        padded_caches = wrapper.make_generation_cache(2, 9 + pad, device)
        shifted = wrapper.prefill(padded, padded_caches, key_valid)
    torch.testing.assert_close(
        shifted.logits.float(), clean.logits.float(), rtol=1e-4, atol=1e-5
    )
    for clean_layer, padded_layer in zip(clean_caches, padded_caches):
        if len(clean_layer) != 4:
            continue
        # Conv windows and states are position-free, so the padded run must
        # land on the identical decode cache, not merely close logits.
        for clean_tensor, padded_tensor in zip(clean_layer, padded_layer):
            torch.testing.assert_close(
                padded_tensor, clean_tensor, rtol=1e-5, atol=1e-6
            )


def test_replay_reproduces_kda_rollout_logprobs():
    """Age-0 canary on the recurrent trunk.

    The rollout prices tokens through the pure-PyTorch step recurrence; the
    replay prices them through the full-sequence reference recurrence. The
    PPO ratio at behavior age 0 is exp of their difference, so the two code
    paths must agree on every recorded action.
    """
    wrapper = _wrapper()
    prompt_ids = torch.randint(1, 32, (2, 6))
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        batch = trim_stream(
            rollout_continuations(
                wrapper,
                prompt_ids,
                4,
                24,
                1.0,
                1.0,
                generator=generator,
                prompt_repeats=2,
            )
        )
        stream_inputs, beliefs = replay_beliefs(wrapper, batch)
        features = wrapper.renderer_features(stream_inputs, beliefs)
        logits = wrapper.backbone.logits_from_features(features).float()
        token_targets = torch.zeros_like(batch.token_ids)
        token_targets[:, :-1] = batch.token_ids[:, 1:]
        token_logprobs = (
            logits.log_softmax(-1)
            .gather(-1, token_targets[..., None])
            .squeeze(-1)
        )
    actions = batch.action_mask.bool()
    assert bool(actions.any())
    torch.testing.assert_close(
        token_logprobs[actions],
        batch.old_token_logprobs[actions],
        rtol=2e-4,
        atol=2e-4,
    )
    # Latent rollouts on the KDA trunk store their carried beliefs like any
    # other backbone: the hidden channel is architecture-independent.
    assert batch.carry_injected and batch.hiddens.size(-1) == wrapper.backbone.model_dim


def test_fresh_kda_trunk_initializes_every_parameter():
    torch.manual_seed(5)
    backbone = NanoKDABackbone(**MODEL_KWARGS)
    for name, parameter in backbone.named_parameters():
        assert bool(parameter.isfinite().all()), name
    for block in backbone.blocks:
        if not block.use_kda:
            continue
        attn = block.attn
        for conv in (attn.q_conv1d, attn.k_conv1d, attn.v_conv1d):
            # Identity causal conv: current token only, so silu(proj) at init.
            assert float(conv.weight[:, 0, :-1].abs().sum()) == 0.0
            assert bool((conv.weight[:, 0, -1] == 1).all())
        assert float(attn.A_log.abs().sum()) == 0.0
        assert float(attn.o_proj.weight.abs().sum()) == 0.0
        dt = attn.dt_bias
        assert bool((dt > math.log(math.expm1(1e-4))).all())


def test_muon_partition_keeps_conv_windows_and_gates_out_of_muon():
    from postraining.train_latent_vapo import (
        build_optimizers,
        muon_matrix_parameters,
        non_trunk_parameter_ids,
    )
    from postraining.value_model import SeparateCritic

    wrapper = _wrapper()
    backbone = wrapper.backbone
    matrices = muon_matrix_parameters(
        backbone.blocks, non_trunk_parameter_ids(backbone)
    )
    assert matrices, "the KDA trunk must still hand its 2-D matrices to Muon"
    assert all(parameter.ndim == 2 for parameter in matrices)
    conv_ids = {
        id(conv.weight)
        for block in backbone.blocks
        if block.use_kda
        for conv in (
            block.attn.q_conv1d,
            block.attn.k_conv1d,
            block.attn.v_conv1d,
        )
    }
    assert conv_ids.isdisjoint({id(p) for p in matrices})
    critic = SeparateCritic(_seeded_backbone(4), num_bins=8)
    for parameter in wrapper.parameters():
        parameter.requires_grad_(True)
    optimizers = build_optimizers(
        wrapper,
        critic,
        3e-4,
        trunk_optimizer="muon",
        muon_learning_rate=1e-3,
        critic_muon_learning_rate=1e-3,
        fused=False,
    )
    assert set(optimizers) == {"actor", "critic", "actor_muon", "critic_muon"}


@torch.no_grad()
def test_paged_refill_matches_independent_masked_decode_on_kda_trunk():
    """Shuffled lanes on the hybrid trunk decode exactly like isolated rows.

    The recurrent half of the paged parity statement: admission must fan the
    bank's conv windows and delta-rule state into arbitrary lanes, and two
    paged steps at ragged occupancy must reproduce per-row dense decode. Two
    steps rather than one so the conv-window shift and the state recurrence
    both run from lane storage, not just from the freshly admitted rows.
    """
    wrapper = _wrapper().eval()
    prompts = torch.randint(
        1, 32, (2, 5), generator=torch.Generator().manual_seed(37)
    )
    lengths = torch.tensor([3, 5])
    prompts[0, :2] = 0
    slots = torch.tensor([[2, 0], [3, 1]])
    paged = wrapper.make_paged_generation_cache(
        4, 12, torch.device("cpu"), dtype=torch.float32, page_size=4
    )
    bank = wrapper.build_prompt_prefix_bank(prompts, lengths, dtype=torch.float32)
    selected_groups = torch.tensor([1, 0])
    prefilled = wrapper.admit_prompt_prefixes(bank, selected_groups, slots, paged)

    references = []
    reference_caches = []
    key_valid = torch.arange(5)[None] >= (5 - lengths)[:, None]
    for group in selected_groups.tolist():
        for _ in range(2):
            dense_cache = wrapper.make_generation_cache(
                1, 12, torch.device("cpu"), dtype=torch.float32
            )
            reference = wrapper.prefill(
                prompts[group : group + 1],
                dense_cache,
                key_valid[group : group + 1],
            )
            references.append(reference)
            reference_caches.append(dense_cache)
    torch.testing.assert_close(
        prefilled.logits,
        torch.cat([reference.logits for reference in references]),
        rtol=1e-5,
        atol=1e-5,
    )

    for offset, next_tokens in enumerate(
        (torch.tensor([7, 11, 13, 17]), torch.tensor([19, 23, 5, 29]))
    ):
        positions = torch.full((4,), 5 + offset, dtype=torch.long)
        paged_step = wrapper.token_paged_step(
            next_tokens,
            paged,
            slot_ids=slots.flatten(),
            positions=positions,
        )
        dense_steps = []
        for row, dense_cache in enumerate(reference_caches):
            group = int(selected_groups[row // 2])
            mask = torch.cat(
                (
                    key_valid[group : group + 1],
                    torch.ones(1, 1 + offset, dtype=torch.bool),
                ),
                dim=1,
            )
            dense_steps.append(
                wrapper.token_step(
                    next_tokens[row : row + 1],
                    dense_cache,
                    5 + offset,
                    mask,
                )
            )
        torch.testing.assert_close(
            paged_step.belief,
            torch.cat([step.belief for step in dense_steps]),
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            paged_step.logits,
            torch.cat([step.logits for step in dense_steps]),
            rtol=1e-4,
            atol=1e-4,
        )


@torch.no_grad()
def test_padding_rows_leave_recurrent_lanes_untouched():
    """Dead rows naming a live lane must not touch its conv windows or state.

    The scheduler pads decode batches with slot 0 — a live, occupied lane —
    under ``live=False``. KV padding is made inert by empty read ranges and
    scratch addresses; the recurrent analogue is the scratch-lane redirect.
    Step lanes 2/3 while two padding rows name lane 0, and demand lanes 0/1
    stay bit-identical: nothing in the batch legitimately writes them, so any
    change is a padding row punching through ``live``.
    """
    wrapper = _wrapper().eval()
    prompts = torch.randint(
        1, 32, (2, 4), generator=torch.Generator().manual_seed(41)
    )
    paged = wrapper.make_paged_generation_cache(
        4, 12, torch.device("cpu"), dtype=torch.float32, page_size=4
    )
    wrapper.prefill_into_paged_slots(
        prompts,
        torch.tensor([4, 4]),
        torch.tensor([[0, 1], [2, 3]]),
        paged,
    )
    recurrent_snapshots = [
        tuple(tensor[:2].clone() for tensor in layer)
        for layer in paged.layers
        if len(layer) == 4
    ]

    stepped = wrapper.token_paged_step(
        torch.tensor([7, 11, 0, 0]),
        paged,
        slot_ids=torch.tensor([2, 3, 0, 0]),
        positions=torch.tensor([4, 4, 0, 0]),
        live=torch.tensor([True, True, False, False]),
    )
    assert torch.isfinite(stepped.logits[:2]).all()
    recurrent_layers = [
        layer for layer in paged.layers if len(layer) == 4
    ]
    assert recurrent_snapshots  # the hybrid fixture must have KDA layers
    for snapshot, layer in zip(recurrent_snapshots, recurrent_layers):
        for before, after in zip(snapshot, layer):
            assert torch.equal(after[:2], before)


@torch.no_grad()
def test_recurrent_lane_recycling_overwrites_previous_occupant():
    """A recycled lane must decode as if its previous occupant never existed.

    Dense lanes hide a stale suffix behind ``kv_starts``; a recurrent state
    has no such mask, so readmission is sound only because the admission copy
    overwrites the lane's conv windows and state wholesale. Dirty the lanes
    with a first occupant and several steps, readmit, and demand exact parity
    with a never-used cache.
    """
    wrapper = _wrapper().eval()
    prompts = torch.randint(
        1, 32, (2, 6), generator=torch.Generator().manual_seed(43)
    )
    lengths = torch.tensor([6, 3])
    prompts[1, :3] = 0
    paged = wrapper.make_paged_generation_cache(
        2, 12, torch.device("cpu"), dtype=torch.float32, page_size=4
    )
    bank = wrapper.build_prompt_prefix_bank(prompts, lengths, dtype=torch.float32)
    wrapper.admit_prompt_prefixes(
        bank, torch.tensor([0]), torch.tensor([[0, 1]]), paged
    )
    for offset, tokens in enumerate(
        (torch.tensor([7, 11]), torch.tensor([13, 17]))
    ):
        wrapper.token_paged_step(
            tokens,
            paged,
            slot_ids=torch.tensor([0, 1]),
            positions=torch.full((2,), 6 + offset, dtype=torch.long),
        )

    fresh = wrapper.make_paged_generation_cache(
        2, 12, torch.device("cpu"), dtype=torch.float32, page_size=4
    )
    recycled_output = wrapper.admit_prompt_prefixes(
        bank, torch.tensor([1]), torch.tensor([[1, 0]]), paged
    )
    fresh_output = wrapper.admit_prompt_prefixes(
        bank, torch.tensor([1]), torch.tensor([[1, 0]]), fresh
    )
    torch.testing.assert_close(recycled_output.logits, fresh_output.logits)
    step_recycled = wrapper.token_paged_step(
        torch.tensor([19, 23]),
        paged,
        slot_ids=torch.tensor([1, 0]),
        positions=torch.full((2,), 6, dtype=torch.long),
    )
    step_fresh = wrapper.token_paged_step(
        torch.tensor([19, 23]),
        fresh,
        slot_ids=torch.tensor([1, 0]),
        positions=torch.full((2,), 6, dtype=torch.long),
    )
    torch.testing.assert_close(step_recycled.logits, step_fresh.logits)
    torch.testing.assert_close(step_recycled.belief, step_fresh.belief)


def test_kda_paged_step_core_is_one_full_graph_across_ragged_positions():
    """The recurrent gather/step/scatter must not break the compiled core."""
    wrapper = _wrapper().eval()
    paged = wrapper.make_paged_generation_cache(
        2, 8, torch.device("cpu"), dtype=torch.float32, page_size=4
    )
    wrapper.prefill_into_paged_slots(
        torch.randint(1, 32, (1, 3)),
        torch.tensor([3]),
        torch.tensor([[0, 1]]),
        paged,
    )
    slot_ids = torch.tensor([0, 1])
    input_latent = wrapper.embed_tokens(torch.tensor([[3], [5]]))
    compiled_graphs = []

    def backend(graph, _example_inputs):
        compiled_graphs.append(graph)
        return graph.forward

    compiled = torch.compile(
        wrapper.paged_step_core, backend=backend, fullgraph=True
    )
    with torch.no_grad():
        for positions in (
            torch.tensor([3, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6, 6]),
        ):
            block_mask = paged.block_mask(slot_ids, positions + 1)
            output = compiled(
                input_latent,
                paged.layers,
                positions,
                block_mask,
                paged.token_addresses(slot_ids, positions),
                slot_ids,
            )
            assert torch.isfinite(output[-1]).all()
    assert len(compiled_graphs) == 1

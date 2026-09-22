from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini import nanogpt_mini_model
from postraining.latent_thought import LatentThoughtModel
from postraining.model_io import copy_trunk, fresh_trunk, load_model
from postraining.nano_backbone import NanoGPTBackbone, NanoTiedDotBackbone
from postraining.value_model import SeparateCritic

# head_dim is fixed at 128, so the smallest multi-head trunk is model_dim 256.
KWARGS = dict(vocab_size=64, num_layers=2, model_dim=256)


def _backbone(cls=NanoGPTBackbone, seed: int = 3):
    torch.manual_seed(seed)
    backbone = cls(**KWARGS).float().eval()
    with torch.no_grad():
        # Zero-init readouts make every logit comparison trivially 0 == 0;
        # give the renderer real weights so parity tests are meaningful.
        if isinstance(backbone, NanoTiedDotBackbone):
            backbone.readout_scale.fill_(0.5)
            backbone.readout_bias.normal_(std=0.1)
        else:
            backbone.proj.weight.normal_(std=0.05)
            backbone.proj.bias.normal_(std=0.05)
    return backbone


def _teacher_forced(wrapper, input_ids):
    backbone = wrapper.backbone
    token_latent = backbone.embed_tokens(input_ids)
    beliefs = backbone.temporal_belief_from_token_latent(token_latent)
    logits = backbone.logits_from_features(
        wrapper.renderer_features(token_latent, beliefs)
    )
    return beliefs, logits


@torch.no_grad()
def test_stepwise_matches_teacher_forced_int_positions():
    for cls in (NanoGPTBackbone, NanoTiedDotBackbone):
        wrapper = LatentThoughtModel(_backbone(cls)).eval()
        input_ids = torch.randint(0, KWARGS["vocab_size"], (2, 7), generator=torch.Generator().manual_seed(5))
        beliefs, logits = _teacher_forced(wrapper, input_ids)
        caches = wrapper.make_generation_cache(2, 7, torch.device("cpu"), dtype=torch.float32)
        for t in range(7):
            out = wrapper.token_step(input_ids[:, t], caches, t)
            torch.testing.assert_close(out.belief, beliefs[:, t], rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(out.logits, logits[:, t], rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_stepwise_matches_teacher_forced_tensor_positions():
    wrapper = LatentThoughtModel(_backbone()).eval()
    input_ids = torch.randint(0, KWARGS["vocab_size"], (2, 6), generator=torch.Generator().manual_seed(7))
    beliefs, logits = _teacher_forced(wrapper, input_ids)
    caches = wrapper.make_generation_cache(2, 6, torch.device("cpu"), dtype=torch.float32)
    for t in range(6):
        out = wrapper.token_step(input_ids[:, t], caches, torch.tensor(t))
        torch.testing.assert_close(out.belief, beliefs[:, t], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out.logits, logits[:, t], rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_stepwise_full_cache_key_mask():
    """The static-shape path: 1-D key mask over a zero-filled full cache."""
    wrapper = LatentThoughtModel(_backbone()).eval()
    input_ids = torch.randint(0, KWARGS["vocab_size"], (2, 6), generator=torch.Generator().manual_seed(9))
    beliefs, logits = _teacher_forced(wrapper, input_ids)
    cache_length = 10
    caches = wrapper.make_static_generation_cache(
        2, cache_length, torch.device("cpu"), dtype=torch.float32
    )
    key_mask = torch.zeros(cache_length, dtype=torch.bool)
    for t in range(6):
        key_mask[t] = True
        out = wrapper.token_step(input_ids[:, t], caches, torch.tensor(t), key_mask)
        torch.testing.assert_close(out.belief, beliefs[:, t], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out.logits, logits[:, t], rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_prefill_matches_stepwise():
    wrapper = LatentThoughtModel(_backbone()).eval()
    input_ids = torch.randint(0, KWARGS["vocab_size"], (2, 5), generator=torch.Generator().manual_seed(11))
    prefill_caches = wrapper.make_generation_cache(2, 8, torch.device("cpu"), dtype=torch.float32)
    prefill_out = wrapper.prefill(input_ids, prefill_caches)
    step_caches = wrapper.make_generation_cache(2, 8, torch.device("cpu"), dtype=torch.float32)
    for t in range(5):
        step_out = wrapper.token_step(input_ids[:, t], step_caches, t)
    torch.testing.assert_close(prefill_out.belief, step_out.belief, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(prefill_out.logits, step_out.logits, rtol=1e-4, atol=1e-4)
    for prefill_cache, step_cache in zip(prefill_caches, step_caches):
        torch.testing.assert_close(
            prefill_cache[0][:, :, :5], step_cache[0][:, :, :5], rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            prefill_cache[1][:, :, :5], step_cache[1][:, :, :5], rtol=1e-4, atol=1e-4
        )


@torch.no_grad()
def test_left_padded_rollout_matches_unpadded():
    """2-D key-mask stepping: padded rows reproduce their unpadded stream."""
    wrapper = LatentThoughtModel(_backbone()).eval()
    length, pad = 5, 3
    row = torch.randint(0, KWARGS["vocab_size"], (1, length), generator=torch.Generator().manual_seed(13))
    reference_caches = wrapper.make_generation_cache(1, length + 2, torch.device("cpu"), dtype=torch.float32)
    reference = wrapper.prefill(row, reference_caches)

    padded = torch.cat([torch.zeros(1, pad, dtype=row.dtype), row], dim=1)
    key_valid = torch.cat(
        [torch.zeros(1, pad, dtype=torch.bool), torch.ones(1, length, dtype=torch.bool)],
        dim=1,
    )
    total = pad + length + 2
    padded_caches = wrapper.make_generation_cache(1, total, torch.device("cpu"), dtype=torch.float32)
    for cache in padded_caches:
        for tensor in cache:
            tensor.zero_()
    padded_out = wrapper.prefill(padded, padded_caches, key_valid)
    torch.testing.assert_close(padded_out.belief, reference.belief, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(padded_out.logits, reference.logits, rtol=1e-4, atol=1e-4)

    # Continue one step through the per-row masked branch and compare with the
    # unpadded stepwise continuation.
    next_token = torch.tensor([1])
    key_mask = torch.cat([key_valid, torch.ones(1, 1, dtype=torch.bool)], dim=1)
    padded_step = wrapper.token_step(
        next_token, padded_caches, torch.tensor(pad + length), key_mask
    )
    reference_step = wrapper.token_step(next_token, reference_caches, length)
    torch.testing.assert_close(padded_step.belief, reference_step.belief, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(padded_step.logits, reference_step.logits, rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_paged_refill_matches_independent_masked_decode():
    """Shuffled physical lanes may advance at different logical positions."""
    wrapper = LatentThoughtModel(_backbone()).eval()
    prompts = torch.randint(
        1,
        KWARGS["vocab_size"],
        (2, 5),
        generator=torch.Generator().manual_seed(37),
    )
    lengths = torch.tensor([3, 5])
    prompts[0, :2] = 0
    slots = torch.tensor([[2, 0], [3, 1]])
    paged = wrapper.make_paged_generation_cache(
        4, 12, torch.device("cpu"), dtype=torch.float32, page_size=4
    )
    bank = wrapper.build_prompt_prefix_bank(
        prompts, lengths, dtype=torch.float32
    )
    selected_groups = torch.tensor([1, 0])
    prefilled = wrapper.admit_prompt_prefixes(
        bank, selected_groups, slots, paged
    )

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

    next_tokens = torch.tensor([7, 11, 13, 17])
    positions = torch.full((4,), 5, dtype=torch.long)
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
            (key_valid[group : group + 1], torch.ones(1, 1, dtype=torch.bool)),
            dim=1,
        )
        dense_steps.append(
            wrapper.token_step(
                next_tokens[row : row + 1], dense_cache, 5, mask
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
    # Four lanes of 12, plus the scratch page padding rows write into so they
    # never address through a slot that a page pool may have reassigned.
    assert paged.layers[0][0].shape[2] == 4 * 12 + paged.page_size
    mask = paged.block_mask(slots.flatten(), positions + 1)
    # A range touches at most two partial pages, but the tables must still
    # share their last axis: the Triton decode kernel offsets FULL_KV_IDX by
    # stride("KV_IDX") and bounds it by size("KV_IDX", -1), so a narrower
    # partial table makes it read the full table at the wrong row stride, and
    # full pages skip mask_mod so nothing downstream corrects it.
    assert mask.kv_indices.shape == mask.full_kv_indices.shape
    assert mask.full_kv_indices.shape[-1] == paged.pages_per_lane


@torch.no_grad()
def test_prompt_prefix_bank_prefills_once_across_repeated_admissions(monkeypatch):
    wrapper = LatentThoughtModel(_backbone()).eval()
    prompts = torch.randint(
        1,
        KWARGS["vocab_size"],
        (3, 6),
        generator=torch.Generator().manual_seed(41),
    )
    lengths = torch.tensor([2, 4, 6])
    prompts[0, :4] = 0
    prompts[1, :2] = 0
    calls = 0
    original = wrapper.backbone.prefill_belief

    def counted_prefill(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        wrapper.backbone, "prefill_belief", counted_prefill
    )
    bank = wrapper.build_prompt_prefix_bank(
        prompts, lengths, dtype=torch.float32
    )
    assert calls == 1
    assert bank.prompt_width == 6
    assert bank.kv_starts.tolist() == [4, 2, 0]

    paged = wrapper.make_paged_generation_cache(
        4, 10, torch.device("cpu"), dtype=torch.float32, page_size=4
    )
    first = wrapper.admit_prompt_prefixes(
        bank,
        torch.tensor([2, 0]),
        torch.tensor([[3, 1], [2, 0]]),
        paged,
    )
    second = wrapper.admit_prompt_prefixes(
        bank,
        torch.tensor([1]),
        torch.tensor([[1, 3]]),
        paged,
    )
    assert calls == 1
    assert first.logits.shape == (4, KWARGS["vocab_size"])
    torch.testing.assert_close(
        second.logits,
        bank.output.logits[1:2].repeat_interleave(2, dim=0),
    )
    assert paged.kv_starts.tolist() == [4, 2, 4, 2]


@torch.no_grad()
def test_paged_step_core_is_one_full_graph_across_ragged_positions():
    """BlockMask values may change without recompiling the fixed-shape core."""
    wrapper = LatentThoughtModel(
        NanoGPTBackbone(vocab_size=32, num_layers=1, model_dim=128).float()
    ).eval()
    paged = wrapper.make_paged_generation_cache(
        2, 8, torch.device("cpu"), dtype=torch.float32, page_size=4
    )
    wrapper.prefill_into_paged_slots(
        torch.randint(0, 32, (1, 3)),
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


@torch.no_grad()
def test_eager_tensor_position_step_accepts_a_bf16_cache():
    """The step path must write its own dtype into the cache, not assume one.

    Eager ``index_copy_`` refuses a dtype mismatch, so without the cast this
    raises the moment the rollout falls out of compile.
    """
    for cls in (NanoGPTBackbone, NanoTiedDotBackbone):
        wrapper = LatentThoughtModel(_backbone(cls)).eval()
        caches = wrapper.make_static_generation_cache(
            2, 12, torch.device("cpu"), dtype=torch.bfloat16
        )
        key_mask = torch.zeros(12, dtype=torch.bool)
        input_ids = torch.randint(
            0, KWARGS["vocab_size"], (2, 5),
            generator=torch.Generator().manual_seed(47),
        )
        for t in range(5):
            key_mask[t] = True
            out = wrapper.token_step(
                input_ids[:, t], caches, torch.tensor(t), key_mask
            )
            assert torch.isfinite(out.logits).all()
        for cache in caches:
            assert cache[0].dtype == torch.bfloat16
            assert cache[1].dtype == torch.bfloat16


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("RUN_CUDA_TESTS") != "1",
    reason="set RUN_CUDA_TESTS=1 on CUDA host",
)
@torch.no_grad()
def test_eager_step_under_cuda_bf16_autocast():
    """The real asymmetry: autocast norms k to fp32 and leaves v bf16.

    ``rms_norm`` is on autocast's fp32 list and ``linear`` is not, so only
    CUDA autocast produces the mismatched pair the cache has to absorb.
    """
    # .cuda() on the WRAPPER, not on the backbone alone: LatentThoughtModel
    # builds its own transition and adapter parameters in __init__, so moving
    # only the argument leaves those on the host and the first projection
    # fails with mat2 on cpu.
    wrapper = LatentThoughtModel(_backbone()).cuda().eval()
    device = torch.device("cuda")
    caches = wrapper.make_static_generation_cache(
        2, 12, device, dtype=torch.bfloat16
    )
    key_mask = torch.zeros(12, dtype=torch.bool, device=device)
    input_ids = torch.randint(0, KWARGS["vocab_size"], (2, 5), device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for t in range(5):
            key_mask[t] = True
            out = wrapper.token_step(
                input_ids[:, t], caches, torch.tensor(t, device=device), key_mask
            )
            assert torch.isfinite(out.logits).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("RUN_CUDA_TESTS") != "1",
    reason="set RUN_CUDA_TESTS=1 on CUDA host",
)
@torch.no_grad()
def test_cuda_ragged_prefill_uses_varlen_and_matches_independent_rows(
    monkeypatch,
):
    """CUDA padding is packed once and never becomes a dense B×L×L mask.

    Comparing the next cached step as well as the prefill result catches both
    valid-token attention errors and a subtle cache-position error: the
    packed valid suffix is an isolated sequence, but K/V must still land in
    the original physical left-padded slots for generation.
    """

    from torch.nn.attention import varlen as varlen_module

    device = torch.device("cuda")
    wrapper = LatentThoughtModel(_backbone()).to(device).eval()
    lengths = torch.tensor([3, 6], device=device)
    width = 6
    ids = torch.randint(
        1,
        KWARGS["vocab_size"],
        (2, width),
        generator=torch.Generator(device=device).manual_seed(53),
        device=device,
    )
    key_valid = torch.arange(width, device=device)[None] >= (
        width - lengths[:, None]
    )
    ids = torch.where(key_valid, ids, torch.zeros_like(ids))

    calls: list[tuple[int, list[int]]] = []
    native_varlen = varlen_module.varlen_attn

    def counted_varlen(query, key, value, cu_seq_q, cu_seq_k, *args, **kwargs):
        calls.append((query.shape[0], cu_seq_q.detach().cpu().tolist()))
        return native_varlen(
            query, key, value, cu_seq_q, cu_seq_k, *args, **kwargs
        )

    def dense_mask_forbidden(_key_valid):
        raise AssertionError("CUDA prefill constructed the dense padding mask")

    monkeypatch.setattr(varlen_module, "varlen_attn", counted_varlen)
    monkeypatch.setattr(
        NanoGPTBackbone,
        "_prefill_attention_mask",
        staticmethod(dense_mask_forbidden),
    )

    batched_caches = wrapper.make_generation_cache(
        2, width + 2, device, dtype=torch.bfloat16
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        batched = wrapper.prefill(ids, batched_caches, key_valid)
    assert len(calls) == KWARGS["num_layers"]
    assert calls == [(12, [0, 3, 6, 6, 12])] * KWARGS["num_layers"]

    reference_outputs = []
    reference_caches = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for row, length in enumerate((3, 6)):
            row_cache = wrapper.make_generation_cache(
                1, length + 2, device, dtype=torch.bfloat16
            )
            reference_outputs.append(
                wrapper.prefill(ids[row : row + 1, -length:], row_cache)
            )
            reference_caches.append(row_cache)
    torch.testing.assert_close(
        batched.belief,
        torch.cat([output.belief for output in reference_outputs]),
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(
        batched.logits,
        torch.cat([output.logits for output in reference_outputs]),
        rtol=3e-2,
        atol=3e-2,
    )

    next_ids = torch.tensor([7, 11], device=device)
    next_valid = torch.cat(
        (key_valid, torch.ones(2, 1, dtype=torch.bool, device=device)), dim=1
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        batched_step = wrapper.token_step(
            next_ids, batched_caches, torch.tensor(width, device=device), next_valid
        )
        reference_steps = [
            wrapper.token_step(
                next_ids[row : row + 1], reference_caches[row], length
            )
            for row, length in enumerate((3, 6))
        ]
    torch.testing.assert_close(
        batched_step.belief,
        torch.cat([output.belief for output in reference_steps]),
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(
        batched_step.logits,
        torch.cat([output.logits for output in reference_steps]),
        rtol=3e-2,
        atol=3e-2,
    )

    # Production compiles this tensor-only surface with dynamic batch/length
    # shapes.  Full-graph capture here prevents metadata construction or the
    # native varlen call from silently falling back to Python between layers.
    monkeypatch.undo()
    compiled_prefill = torch.compile(
        wrapper.prefill_core, fullgraph=True, dynamic=True
    )
    compiled_caches = wrapper.make_generation_cache(
        2, width + 2, device, dtype=torch.bfloat16
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        (
            compiled_belief,
            compiled_predicted,
            compiled_log_sigma,
            compiled_logits,
        ) = compiled_prefill(
            wrapper.embed_tokens(ids), compiled_caches, key_valid
        )
    torch.testing.assert_close(
        compiled_belief, batched.belief, rtol=3e-2, atol=3e-2
    )
    torch.testing.assert_close(
        compiled_predicted, batched.predicted, rtol=3e-2, atol=3e-2
    )
    torch.testing.assert_close(
        compiled_log_sigma,
        batched.thought_log_sigma,
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(
        compiled_logits, batched.logits, rtol=3e-2, atol=3e-2
    )


@torch.no_grad()
def test_renderer_matches_pretraining_forward():
    """Adapter-assembled sum-CE equals the pretraining class's forward."""
    for backbone_cls, pretraining_cls in (
        (NanoGPTBackbone, nanogpt_mini_model.GPT),
        (NanoTiedDotBackbone, nanogpt_mini_model.TiedDotGPT),
    ):
        backbone = _backbone(backbone_cls)
        pretraining = pretraining_cls(**KWARGS).float()
        pretraining.load_state_dict(backbone.state_dict(), strict=True)
        pretraining.eval()
        generator = torch.Generator().manual_seed(17)
        inputs = torch.randint(0, KWARGS["vocab_size"], (2, 6), generator=generator)
        targets = torch.randint(0, KWARGS["vocab_size"], (2, 6), generator=generator)
        _, logits = _teacher_forced(LatentThoughtModel(backbone), inputs)
        assembled = F.cross_entropy(
            logits.float().view(targets.numel(), -1), targets.view(-1), reduction="sum"
        )
        torch.testing.assert_close(
            assembled, pretraining(inputs, targets), rtol=1e-5, atol=1e-5
        )


def test_checkpoint_roundtrip(tmp_path):
    """Script-style payload -> load_model -> strict fp32-master backbone."""
    torch.manual_seed(23)
    pretraining = nanogpt_mini_model.GPT(**KWARGS)  # bf16 embed, as the scripts save
    payload = {
        "model": pretraining.state_dict(),
        "model_config": dict(KWARGS, mlp_hidden=4 * KWARGS["model_dim"]),
        "architecture": "nanogpt_mini_v1",
    }
    checkpoint = tmp_path / "final_model.pt"
    torch.save(payload, checkpoint)
    model = load_model(checkpoint, torch.device("cpu"))
    assert isinstance(model, NanoGPTBackbone)
    assert model.architecture == "nanogpt_mini_v1"
    assert model.tok_emb.weight.dtype == torch.float32
    assert all(not parameter.requires_grad for parameter in model.parameters())
    # bf16 -> fp32 promotion is value-exact.
    torch.testing.assert_close(
        model.embed.weight, pretraining.embed.weight.float(), rtol=0, atol=0
    )
    trunk = fresh_trunk(model, torch.device("cpu"))
    assert type(trunk) is NanoGPTBackbone
    assert trunk.model_config == model.model_config
    assert not torch.equal(trunk.embed.weight, model.embed.weight)

    # --critic-init actor: the loaded actor's exact values in storage of its
    # own, trainable while the frozen reference stays untouched by its steps.
    copy = copy_trunk(model, torch.device("cpu"))
    assert type(copy) is NanoGPTBackbone
    assert copy.architecture == model.architecture
    reference = model.state_dict()
    copied = copy.state_dict()
    assert copied.keys() == reference.keys()
    reference_storage = {
        tensor.untyped_storage().data_ptr() for tensor in reference.values()
    }
    for name, tensor in copied.items():
        assert tensor.dtype == reference[name].dtype, name
        torch.testing.assert_close(tensor, reference[name], rtol=0, atol=0)
        assert tensor.untyped_storage().data_ptr() not in reference_storage, name
    assert all(parameter.requires_grad for parameter in copy.parameters())
    snapshot = {name: tensor.clone() for name, tensor in reference.items()}
    with torch.no_grad():
        for parameter in copy.parameters():
            parameter.add_(1.0)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, snapshot[name]), name


def test_tieddot_checkpoint_dispatch(tmp_path):
    torch.manual_seed(29)
    pretraining = nanogpt_mini_model.TiedDotGPT(**KWARGS)
    payload = {
        "model": pretraining.state_dict(),
        "model_config": dict(KWARGS, mlp_hidden=4 * KWARGS["model_dim"]),
        "architecture": "nanogpt_mini_tieddot_v1",
    }
    checkpoint = tmp_path / "final_model.pt"
    torch.save(payload, checkpoint)
    model = load_model(checkpoint, torch.device("cpu"))
    assert isinstance(model, NanoTiedDotBackbone)


def test_nano_critic_smoke():
    """SeparateCritic on a fresh nano trunk predicts exactly zero."""
    torch.manual_seed(31)
    trunk = NanoGPTBackbone(**KWARGS).float().eval()
    critic = SeparateCritic(trunk).eval()
    from postraining.latent_rollout import LatentRolloutBatch, PAD_SLOT, TOKEN_SLOT

    batch_size, stream = 2, 4
    kind = torch.full((batch_size, stream), TOKEN_SLOT, dtype=torch.long)
    kind[1, -1] = PAD_SLOT
    zeros = torch.zeros(batch_size, stream)
    batch = LatentRolloutBatch(
        kind=kind,
        token_ids=torch.randint(0, KWARGS["vocab_size"], (batch_size, stream)),
        thoughts=torch.zeros(batch_size, stream, KWARGS["model_dim"]),
        hiddens=torch.zeros(batch_size, stream, 0),
        actions=torch.zeros((batch_size, stream), dtype=torch.long),
        action_mask=torch.zeros(batch_size, stream, dtype=torch.bool),
        stop_mask=zeros.clone(),
        emit_mask=zeros.clone(),
        old_stop_logprobs=zeros.clone(),
        old_token_logprobs=zeros.clone(),
        old_token_log_odds=zeros.clone(),
        old_thought_logprobs=torch.zeros(batch_size, stream, KWARGS["model_dim"]),
        old_thought_means=torch.zeros(batch_size, stream, KWARGS["model_dim"]),
        old_thought_log_sigmas=torch.zeros(
            batch_size, stream, KWARGS["model_dim"]
        ),
        old_values=zeros.clone(),
        rewards=zeros.clone(),
        reward_scalar=torch.zeros(batch_size),
        prompt_length=2,
    )
    with torch.no_grad():
        values = critic.values(batch)
    assert values.shape == (batch_size, stream)
    assert torch.equal(values, torch.zeros_like(values))

"""Contracts of the hierarchical byte trainer (CPU; the GPU causality test
lives in test_hbyte_gpu.py and runs through mlq)."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
import torch

from pretraining.hbyte import data as hdata
from pretraining.hbyte.model import (
    HashConfig,
    LocalConfig,
    ModelConfig,
    NextLatDecoder,
    PatchLayout,
    TrunkConfig,
    build_model,
    budgeted_ends,
    capped_ends,
    softcap,
    hashed_ngram_ids,
    initialize,
)
from pretraining.hbyte.train import BYTE_CLASSES, build_optimizers, byte_classes, set_schedule

REPO = Path(__file__).resolve().parents[2]


def test_ngram_ids_are_causal_and_order_local():
    torch.manual_seed(0)
    x = torch.randint(0, 257, (2, 64))
    orders = (3, 5, 8)
    ids = hashed_ngram_ids(x, orders, 1024)
    assert ids.shape == (3, 2, 64)
    for i in range(3):
        assert ids[i].min() >= i * 1024 and ids[i].max() < (i + 1) * 1024
    s = 40
    y = x.clone()
    y[:, s] = (y[:, s] + 1) % 257
    changed = hashed_ngram_ids(y, orders, 1024) != ids
    assert not changed[:, :, :s].any()
    for i, n in enumerate(orders):
        # Every n-gram covering byte s changes (collisions are ~1/1024).
        assert changed[i, :, s : s + n].float().mean() > 0.95
        assert not changed[i, :, s + n :].any()


def test_static_layout_routes_to_newest_completed_patch():
    P, T = 4, 32
    layout = PatchLayout.static(1, T, P, "cpu")
    ends = layout.end_index[0].tolist()
    for t in range(T):
        k = int(layout.route_index[0, t])
        completed = [i for i, e in enumerate(ends) if e <= t]
        assert k == (completed[-1] + 1 if completed else 0)
        # offset 0 marks a target that is the first byte after a patch end.
        assert (int(layout.offset[0, t]) == 0) == (t in ends)


def test_capped_ends_bound_patch_length():
    torch.manual_seed(0)
    natural = torch.rand(3, 200) < 0.1
    natural[1] = False  # a row with no natural boundary at all
    ends = capped_ends(natural, 8)
    assert (ends | ~natural).all(), "natural ends are kept"
    for row in ends:
        positions = torch.cat([torch.tensor([-1]), row.nonzero().flatten()])
        assert int(torch.diff(positions).max()) <= 8
    # Forced ends only where a run would exceed the cap.
    assert torch.equal(ends[1].nonzero().flatten(), torch.arange(7, 200, 8))


def test_dynamic_layout_matches_static_for_regular_ends():
    T, P = 64, 4
    ends = ((torch.arange(T) + 1) % P == 0).expand(2, -1)
    dynamic = PatchLayout.from_ends(ends, budget=32)
    static = PatchLayout.static(2, T, P, "cpu")
    k = static.end_index.size(1)
    assert torch.equal(dynamic.end_index[:, :k], static.end_index)
    assert torch.equal(dynamic.route_index, static.route_index)
    assert torch.equal(dynamic.offset, static.offset)
    assert dynamic.patch_valid[:, :k].all() and not dynamic.patch_valid[:, k:].any()


def test_dynamic_layout_routes_to_newest_completed_patch():
    torch.manual_seed(1)
    ends = capped_ends(torch.rand(2, 100) < 0.3, 6)
    layout = PatchLayout.from_ends(ends, budget=48)
    assert layout.end_index.size(1) == 48
    for b in range(2):
        e = ends[b].nonzero().flatten().tolist()
        assert layout.end_index[b, : len(e)].tolist() == e
        assert int(layout.patch_valid[b].sum()) == len(e)
        for t in range(100):
            completed = [i for i, p in enumerate(e) if p <= t]
            assert int(layout.route_index[b, t]) == len(completed)
            last = e[completed[-1]] if completed else -1
            assert int(layout.offset[b, t]) == t - last
            assert int(layout.offset[b, t]) < 6


def test_budgeted_ends_bound_patches_and_keep_the_cap():
    torch.manual_seed(5)
    T, cap, budget = 120, 6, 30
    natural = torch.rand(4, T) < 0.4
    natural[1] = True  # every byte wants to end a patch
    natural[2] = False  # only forced ends
    natural[3] = torch.rand(T) < 0.1  # sparse: fits without dropping
    ends = budgeted_ends(natural, cap, budget)
    for b in range(4):
        e = ends[b].nonzero().flatten()
        assert len(e) <= budget
        assert int(torch.diff(torch.cat([torch.tensor([-1]), e])).max()) <= cap
        # Natural ends survive up to the first dropped one; the cap alone
        # ends patches after it.
        dropped = (natural[b] & ~ends[b]).nonzero().flatten()
        if len(dropped):
            prefix = natural[b] & (torch.arange(T) < dropped[0])
            assert torch.equal(ends[b], capped_ends(prefix[None], cap)[0])
    assert torch.equal(ends[2:], capped_ends(natural[2:], cap))
    # The reserve shrinks along the row, so a saturating row uses the budget.
    assert int(ends[1].sum()) >= budget - 1
    # Causal: changing natural ends after t never changes ends at or before t.
    moved = natural.clone()
    moved[:, 60:] = ~moved[:, 60:]
    assert torch.equal(budgeted_ends(moved, cap, budget)[:, :60], ends[:, :60])


def test_spacelike_ends_are_causal_first_of_run():
    from pretraining.hbyte.model import spacelike_ends

    text = "Hi  there, x=1.\n\nnaïve end"
    x = torch.tensor([list(text.encode())])
    ends = spacelike_ends(x)[0]
    # End after the first spacelike byte of each run; the bytes of "ï"
    # (c3 af) stay inside the word.
    cut, start = [], 0
    for t in ends.nonzero().flatten().tolist():
        cut.append(bytes(x[0, start : t + 1].tolist()))
        start = t + 1
    assert cut == [b"Hi ", b" there,", b" x=", b"1.", b"\n\nna\xc3\xafve "]
    # Byte t's decision reads bytes t - 1 and t only.
    y = torch.randint(0, 257, (4, 300))
    base = spacelike_ends(y)
    for t in (0, 1, 150, 299):
        z = y.clone()
        z[:, t:] = torch.randint(0, 257, (4, 300 - t))
        assert torch.equal(spacelike_ends(z)[:, :t], base[:, :t])


def test_block_prefix_ids_hash_the_block_string_so_far():
    from pretraining.hbyte.model import block_prefix_ids, prefix_powers, spacelike_ends

    torch.manual_seed(0)
    T, bits = 300, 20
    powers = prefix_powers(T, "cpu")
    words = [b"the ", b"cat ", b"sat. ", b"a ", b"zebra, "]
    rows = []
    for _ in range(3):
        text = b"".join(words[i] for i in torch.randint(0, len(words), (100,)).tolist())
        rows.append(list(text[:T]))
    x = torch.tensor(rows)
    ends = spacelike_ends(x)
    ids = block_prefix_ids(x, ends, bits, powers)
    assert ids.min() >= 1 and ids.max() <= 1 << bits
    # The id is a function of the string since the last end before t: equal
    # strings collide wherever they sit, different ones (almost) never do.
    seen: dict[bytes, int] = {}
    for b in range(x.size(0)):
        start = 0
        for t in range(T):
            key = bytes(x[b, start : t + 1].tolist())
            assert seen.setdefault(key, int(ids[b, t])) == int(ids[b, t]), key
            if ends[b, t]:
                start = t + 1
    assert len(set(seen.values())) == len(seen)
    # Causal in both the bytes and the ends.
    for t in (0, 1, 77, 299):
        y, e = x.clone(), ends.clone()
        y[:, t + 1 :] = torch.randint(0, 257, (3, T - t - 1))
        e[:, t + 1 :] = torch.rand(3, T - t - 1) < 0.3
        assert torch.equal(block_prefix_ids(y, e, bits, powers)[:, : t + 1], ids[:, : t + 1])


def test_block_ngram_and_context_hashes_are_causal_string_functions():
    from pretraining.hbyte.model import block_context_hash, block_ngram_hash, block_prefix_hash, prefix_powers, spacelike_ends

    torch.manual_seed(1)
    T = 400
    powers = prefix_powers(T, "cpu")
    words = [b"the ", b"cat ", b"sat. ", b"a ", b"zebra, "]
    text = b"".join(words[i] for i in torch.randint(0, len(words), (150,)).tolist())
    x = torch.tensor([list(text[:T])])
    ends = spacelike_ends(x)
    prefix = block_prefix_hash(x, ends, powers)
    end_pos = ends[0].nonzero().flatten()
    blocks = [bytes(x[0, s:e + 1].tolist()) for s, e in zip([0, *(end_pos[:-1] + 1).tolist()], end_pos.tolist())]
    block_hash = prefix[:, end_pos]
    for order in (1, 2, 3):
        h = block_ngram_hash(block_hash, order)[0].tolist()
        seen: dict[tuple, int] = {}
        for k in range(len(blocks)):
            key = tuple(blocks[max(0, k - order + 1) : k + 1])
            if k >= order - 1:  # windows that reach before the row start are their own keys
                assert seen.setdefault(key, h[k]) == h[k], key
        assert len(set(seen.values())) == len(seen)
        # Causal over slots.
        for k in (0, 5, 40):
            other = block_hash.clone()
            other[:, k + 1 :] = torch.randint(-(2**62), 2**62, other[:, k + 1 :].shape)
            assert torch.equal(block_ngram_hash(other, order)[:, : k + 1], block_ngram_hash(block_hash, order)[:, : k + 1])
    # Orders hash the same newest block differently.
    assert not torch.equal(block_ngram_hash(block_hash, 1), block_ngram_hash(block_hash, 2))
    # Context: (previous block, current prefix), a function of those strings.
    ctx = block_context_hash(prefix, ends)[0].tolist()
    seen = {}
    start, previous = 0, b""
    for t in range(T):
        current = b"" if ends[0, t] else bytes(x[0, start : t + 1].tolist())
        done_block = bytes(x[0, start : t + 1].tolist()) if ends[0, t] else previous
        key = (done_block, current)
        if previous or ends[0, t]:  # skip the partial first block
            assert seen.setdefault(key, ctx[t]) == ctx[t], key
        if ends[0, t]:
            previous, start = bytes(x[0, start : t + 1].tolist()), t + 1
    assert len(set(seen.values())) == len(seen)


def test_local_mixers_are_one_letter_per_layer():
    assert LocalConfig(decoder_layers=2).mixers("decoder") == "aa"
    assert LocalConfig(encoder_layers=1, encoder_mixers="k").mixers("encoder") == "k"
    for bad in ("k", "kax", "ab"):
        with pytest.raises(ValueError, match="decoder_mixers"):
            LocalConfig(decoder_layers=2, decoder_mixers=bad)


def test_encoder_window_defaults_to_the_local_window():
    assert LocalConfig(window=1024).stage_window("encoder") == 1024
    local = LocalConfig(window=1024, encoder_window=512)
    assert (local.stage_window("encoder"), local.stage_window("decoder")) == (512, 1024)
    with pytest.raises(ValueError, match="encoder_window"):
        LocalConfig(encoder_window=-1)


def test_block_ngram_needs_prefix_memory():
    with pytest.raises(ValueError, match="need prefix_bits"):
        ModelConfig(block_ngram=2)
    with pytest.raises(ValueError, match="need prefix_bits"):
        ModelConfig(prefix_context=True)
    with pytest.raises(ValueError, match="need prefix_bits"):
        ModelConfig(memory_gate=True)
    with pytest.raises(ValueError, match="need prefix_bits"):
        ModelConfig(prefix_local=True)


def test_memory_gate_scales_reads_by_agreement():
    from pretraining.hbyte.model import memory_read

    torch.manual_seed(0)
    state, gain = torch.randn(3, 64), torch.full((64,), 0.5)
    memory = torch.randn(3, 64)
    assert torch.equal(memory_read(gain, memory, state, gated=False), gain * memory)
    gated = memory_read(gain, memory, state, gated=True)
    assert torch.allclose(gated / (gain * memory), torch.sigmoid((torch.nn.functional.rms_norm(state, (64,)) * torch.nn.functional.rms_norm(memory, (64,))).sum(-1, keepdim=True) / 8).expand(-1, 64), atol=1e-5)
    # Aligned memory passes almost whole; opposed memory is shut.
    assert torch.allclose(memory_read(gain, state, state, gated=True), gain * state, rtol=1e-3)
    assert memory_read(gain, -state, state, gated=True).abs().max() < 1e-3


def test_trunk_logits_need_the_attention_decoder():
    with pytest.raises(ValueError, match="trunk_logits"):
        ModelConfig(decoder="nextlat", trunk_logits=True)


def test_chunk_budget_is_static_per_row_length():
    config = ModelConfig(seq_len=8192, patcher="learned", target_patch=6.0, max_patch=64, chunk_slack=1.25)
    assert config.chunk_budget(8192) == 1728  # 64 * ceil(8192 * 1.25 / 6 / 64)
    # Always room for a row of forced ends only.
    tight = ModelConfig(seq_len=8192, patcher="entropy", target_patch=8.0, max_patch=8, chunk_slack=1.0)
    assert tight.chunk_budget(8192) >= 8192 // 8
    assert ModelConfig(seq_len=8192).chunk_budget(8192) == 2048  # static stride 4


@pytest.mark.parametrize("patcher", ["entropy", "learned"])
@pytest.mark.parametrize("target", [2.5, 4.0, 6.0])
def test_rate_threshold_controls_realized_patch_length(patcher, target):
    config = ModelConfig(seq_len=512, patcher=patcher, target_patch=target, max_patch=8)
    model = build_model(config)
    torch.manual_seed(3)
    gate = torch.rand(4, 512).square() * (5 if patcher == "entropy" else 1)
    valid = torch.ones(4, 512, dtype=torch.bool)
    budget = config.chunk_budget(512)
    model.patcher.update_threshold(gate, valid, budget)
    realized = 4 * 512 / int(budgeted_ends(model.patcher.natural_ends(gate), 8, budget).sum())
    assert realized == pytest.approx(target, rel=0.02)


def test_fixed_threshold_is_h_net_half():
    config = ModelConfig(seq_len=64, patcher="learned", threshold_mode="fixed")
    model = build_model(config)
    initialize(model, 0)
    model.patcher.update_threshold(torch.rand(2, 64), torch.ones(2, 64, dtype=torch.bool), config.chunk_budget(64))
    assert float(model.patcher.threshold) == 0.5
    with pytest.raises(ValueError):
        ModelConfig(patcher="entropy", threshold_mode="fixed")


def test_nextlat_decoder_matches_sequential_recurrence():
    torch.manual_seed(2)
    config = ModelConfig(seq_len=48, local=LocalConfig(dim=32), trunk=TrunkConfig(dim=64), patcher="entropy", max_patch=5, decoder="nextlat")
    decoder = NextLatDecoder(config)
    with torch.no_grad():
        for p in decoder.parameters():
            p.normal_(std=0.2)
    ends = capped_ends(torch.rand(2, 48) < 0.3, 5)
    layout = PatchLayout.from_ends(ends, budget=24)
    h = torch.randn(2, 48, 32)
    z = torch.randn(2, layout.end_index.size(1), 64)
    targets = torch.randint(0, 257, (2, 48))
    nll = decoder(h, z, layout.route_index, layout.offset, targets)
    g = torch.cat([decoder.bos.expand(2, 1, -1), z], dim=1)
    for b in range(2):
        state = decoder.lift(decoder.route_norm(g[b, 0]))
        for t in range(48):
            if int(layout.offset[b, t]) == 0:
                state = decoder.lift(decoder.route_norm(g[b, layout.route_index[b, t]]))
            else:
                state = decoder.step(state, h[b, t])
            logits = softcap(decoder.head(decoder.out_norm(state)).float())
            ref = torch.nn.functional.cross_entropy(logits, targets[b, t])
            assert torch.allclose(nll[b, t], ref, rtol=1e-4, atol=1e-4), (b, t, nll[b, t].item(), ref.item())


def test_byte_classes():
    prev = torch.tensor([[32, 97, 97, 49, 97, 32, 32, 200]])
    tgt = torch.tensor([[97, 98, 32, 50, 44, 256, 195, 169]])
    names = [BYTE_CLASSES[i] for i in byte_classes(prev, tgt)[0].tolist()]
    assert names == ["word_initial", "word_internal", "space", "digit", "punct", "eot", "non_ascii", "non_ascii"]


def test_schedule_is_cosine_with_one_percent_warmup():
    param = torch.nn.Parameter(torch.zeros(2, 2))
    opt = torch.optim.SGD([param], lr=1.0)
    opt.param_groups[0]["initial_lr"] = 1.0
    assert set_schedule([opt], 0, 2000) == pytest.approx(1 / 20)
    assert set_schedule([opt], 19, 2000) == pytest.approx(1.0)
    mid = set_schedule([opt], 1000, 2000)
    assert mid == pytest.approx(0.5 * (1 + math.cos(math.pi * 980 / 1980)))


def _tiny_config(atom: str, **extra) -> ModelConfig:
    return ModelConfig(
        atom=atom,
        seq_len=64,
        trunk=TrunkConfig(),
        local=LocalConfig(dim=128, window=16),
        hashing=HashConfig(orders=(3, 4), buckets=64),
        patch_size=4,
        **extra,
    )


MEMORY = {
    "patcher": "space",
    "target_patch": 6.0,
    "max_patch": 16,
    "prefix_bits": 6,
    "trunk_logits": True,
    "block_ngram": 3,
    "prefix_context": True,
    "memory_gate": True,
    "prefix_local": True,
}


@pytest.mark.parametrize("atom,extra,mixers", [
    ("byte", {}, {}),
    ("gpt2", {}, {}),
    ("byte", MEMORY, {}),
    ("byte", MEMORY, {"encoder_mixers": "k", "decoder_mixers": "ka"}),
])
def test_every_parameter_is_initialized_and_owned_once(atom, extra, mixers):
    config = _tiny_config(atom, **extra)
    model = build_model(dataclasses.replace(config, local=dataclasses.replace(config.local, **mixers)))
    initialize(model, 0)

    class Args:
        embed_lr, head_lr, scalar_lr, conv_lr, muon_lr = 0.7, 0.004, 0.015, 0.004, 0.025
        adam_wd, muon_wd = 0.001, 0.05

    build_optimizers(model, Args)


@pytest.mark.skipif(not (REPO / hdata.BYTE_DATA).exists(), reason="corpus not present")
def test_byte_validation_is_the_reference_text(monkeypatch):
    monkeypatch.chdir(REPO)
    token_set, token_info = hdata.reference_validation("gpt2", 2048)
    assert int((token_set.targets >= 0).sum()) == hdata.REFERENCE_VAL_TOKENS
    lut = torch.load(hdata.GPT2_BYTE_LUT).long()
    assert int(lut[token_set.targets].sum()) == token_info["scored_bytes"]
    stream = torch.from_numpy(
        hdata.shard_tokens(REPO / hdata.BYTE_DATA / "fineweb_val_000000.bin")[: token_info["scored_bytes"] + 1].astype("int64")
    )
    for aligned in (True, False):
        byte_set, info = hdata.reference_validation("byte", 8192, aligned=aligned)
        assert info["scored_bytes"] == token_info["scored_bytes"] == byte_set.scored_bytes
        valid = byte_set.targets >= 0
        # Every byte after the first is scored exactly once, in order, from
        # its true predecessor.
        assert torch.equal(byte_set.targets[valid], stream[1:])
        assert torch.equal(byte_set.inputs[valid], stream[:-1])
    # Aligned rows start their context where the token rows do.
    tokens = torch.cat([token_set.inputs.flatten(), token_set.targets[-1, -1:]])
    starts = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lut[tokens], 0)])[::2048][: token_set.rows]
    byte_set, _ = hdata.reference_validation("byte", 8192, aligned=True)
    assert byte_set.rows == token_set.rows
    for r in (0, 1, 500, token_set.rows - 1):
        n = int((byte_set.targets[r] >= 0).nonzero().max()) + 1
        assert torch.equal(byte_set.inputs[r, :n], stream[int(starts[r]) : int(starts[r]) + n])


def test_nextlat_confidence_carries_router_gradient():
    # The straight-through confidence must reach the NLL: scaling the whole
    # state would be divided back out by the readout norm.
    torch.manual_seed(6)
    config = ModelConfig(seq_len=48, local=LocalConfig(dim=32), trunk=TrunkConfig(dim=64), patcher="learned", max_patch=5, decoder="nextlat")
    decoder = NextLatDecoder(config)
    with torch.no_grad():
        for p in decoder.parameters():
            p.normal_(std=0.2)
    ends = capped_ends(torch.rand(2, 48) < 0.3, 5)
    layout = PatchLayout.from_ends(ends, budget=24)
    h, z = torch.randn(2, 48, 32), torch.randn(2, 24, 64)
    targets = torch.randint(0, 257, (2, 48))
    p = torch.rand(2, 48, requires_grad=True)
    c = torch.where(ends, p, 1 - p)
    delta = c - c.detach()
    nll = decoder(h, z, layout.route_index, layout.offset, targets, delta)
    assert torch.equal(nll, decoder(h, z, layout.route_index, layout.offset, targets)), "forward unchanged"
    nll.sum().backward()
    # Inside a patch the state mixes trunk and recurrence, so the gate gets a
    # real gradient. At a patch end the state is the trunk lift alone and the
    # norm removes its scale; there the end decision learns through chunk
    # smoothing instead (and through ``nextlat_local_reset`` if enabled).
    inside = layout.offset > 0
    assert p.grad[inside].abs().min() > 1e-4
    assert p.grad[~inside].abs().max() < 1e-5


def test_patch_pool_is_a_per_patch_softmax():
    from pretraining.hbyte.model import PatchPool

    torch.manual_seed(4)
    config = ModelConfig(seq_len=40, local=LocalConfig(dim=32), patcher="entropy", patch_pool="attn", pool_heads=4)
    pool = PatchPool(config)
    with torch.no_grad():
        for p in pool.parameters():
            p.normal_(std=0.3)
    ends_mask = capped_ends(torch.rand(2, 40) < 0.25, 6)
    layout = PatchLayout.from_ends(ends_mask, budget=16)
    h = torch.randn(2, 40, 32)
    patch_id = layout.route_index - (layout.offset == 0).long()
    end_states = torch.gather(h, 1, layout.end_index.unsqueeze(-1).expand(-1, -1, 32))
    out = pool(h, patch_id, layout.end_index, end_states)
    # Padding patches get no pooled content.
    pad = ~layout.patch_valid
    assert pad.any() and torch.equal(out[pad], end_states[pad])
    x = pool.norm(h)
    k = pool.key(x).view(2, 40, 4, 8)
    v = pool.value(x).view(2, 40, 4, 8)
    q = pool.query.view(4, 8)
    for b in range(2):
        e = ends_mask[b].nonzero().flatten().tolist()
        start = 0
        for p, end in enumerate(e):
            idx = torch.arange(start, end + 1)
            w = ((k[b, idx] * q).sum(-1) / 8**0.5).softmax(0)  # (n, H)
            ref = (w[..., None] * v[b, idx]).sum(0).reshape(32) + h[b, end]
            assert torch.allclose(out[b, p], ref, atol=1e-5), (b, p)
            start = end + 1


def test_patch_lengths_assign_bytes_to_their_patch():
    from pretraining.hbyte.train import bin_names, length_bins, matched_bernoulli_rate, patch_lengths

    # Ends at 1, 4, 5, 9: positions 1-3 predict the 3-byte patch 2-4,
    # position 4 the 1-byte patch 5, positions 5-8 the 4-byte patch 6-9;
    # positions 0 (no end yet) and 9-10 (patch never completes) are excluded.
    ends = torch.zeros(1, 11, dtype=torch.bool)
    ends[0, [1, 4, 5, 9]] = True
    assert patch_lengths(ends).tolist() == [[0, 3, 3, 3, 1, 4, 4, 4, 4, 0, 0]]
    assert length_bins(64) == (0, 1, 2, 3, 4, 8, 16, 32, 64)
    assert bin_names(length_bins(64), 64)[-3:] == ["16-31", "32-63", "64"]
    assert bin_names(length_bins(4), 4) == ["0", "1", "2", "3", "4"]
    assert bin_names((0, 1, 2, 4, 8), 11) == ["0", "1", "2-3", "4-7", "8-11"]
    valid = torch.ones(64, 4096, dtype=torch.bool)
    rate = matched_bernoulli_rate(0.25, 8, valid)
    realized = capped_ends(torch.rand(valid.shape, generator=torch.Generator().manual_seed(1)) < rate, 8).float().mean()
    assert abs(realized - 0.25) < 0.005


def test_metric_log_time_axis(tmp_path):
    import json

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from pretraining.hbyte.train import MetricLog

    log = MetricLog(tmp_path, tmp_path / "tb")
    log.write("val", 0, {"bpb": 8.0})
    log.write("train", 10, {"bpb": 2.0, "train_seconds": 12.4})
    log.write("val", 10, {"bpb": 1.9, "samples": "a|b"})
    log.writer.close()
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["train_seconds"] for r in records] == [0.0, 12.4, 12.4]
    events = EventAccumulator(str(tmp_path / "tb")).Reload()
    assert [(e.step, e.value) for e in events.Scalars("time/val_bpb")] == [(0, 8.0), (12, pytest.approx(1.9))]
    assert [e.step for e in events.Scalars("time/train_bpb")] == [12]


def test_smooth_chunks_matches_sequential_ema():
    from pretraining.hbyte.model import smooth_chunks

    torch.manual_seed(0)
    for K in (1, 7, 64, 100):
        z = torch.randn(2, K, 5)
        w = torch.rand(2, K)
        out = smooth_chunks(z, w, steps=8)
        ref, prev = [], torch.zeros(2, 5)
        for k in range(K):
            prev = w[:, k, None] * z[:, k] + (1 - w[:, k, None]) * prev
            ref.append(prev)
        assert torch.allclose(out, torch.stack(ref, 1), atol=1e-5), K


def test_learned_router_is_causal_and_ratio_loss_is_minimal_at_target():
    from pretraining.hbyte.model import LearnedRouter, initialize, ratio_loss

    config = ModelConfig(seq_len=64, local=LocalConfig(dim=32), patcher="learned", max_patch=16)
    router = LearnedRouter(config)
    with torch.no_grad():
        for p in router.parameters():
            p.normal_(std=0.3)
    h = torch.randn(2, 64, 32)
    p = router(h)
    assert p.shape == (2, 64) and (p >= 0).all() and (p <= 1).all()
    assert torch.allclose(p[:, 0], torch.full((2,), 0.5))  # no previous byte
    moved = h.clone()
    moved[:, 30] += 1
    q = router(moved)
    # p_t reads bytes t and t-1 only.
    assert torch.equal(p[:, :30], q[:, :30]) and torch.equal(p[:, 32:], q[:, 32:])
    router.threshold.fill_(0.5)
    assert torch.equal(router.natural_ends(p), p > 0.5)
    valid = torch.ones(1, 400, dtype=torch.bool)
    at = torch.full((1, 400), 0.2)
    at[0, ::4] = 0.8  # F = G = 1/4
    loss = ratio_loss(at, valid, 4.0)
    assert loss.item() == pytest.approx(1.0)
    # Too many ends and a high mean p: the loss pushes p down.
    g = torch.full((1, 400), 0.6, requires_grad=True)
    ratio_loss(g, valid, 4.0).backward()
    assert (g.grad > 0).all()
    # F is the router's own rate. A saturated router still gets pushed down,
    # though a rate controller would hold the realized ends at 1/4 (where a
    # loss on those ends would be flat in G).
    sat = torch.full((1, 400), 0.97, requires_grad=True)
    ratio_loss(sat, valid, 4.0).backward()
    assert (sat.grad > 0).all()
    # Invalid positions count neither toward F nor G.
    mask = valid.clone()
    mask[0, 200:] = False
    mixed = at.clone()
    mixed[0, 200:] = 0.99
    assert ratio_loss(mixed, mask, 4.0).item() == pytest.approx(1.0)
